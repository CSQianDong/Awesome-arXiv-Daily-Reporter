# Operation-Guided Progressive Human-to-AI Text Transformation Benchmark for Multi-Granularity AI-Text Detection 

**Authors**: Sondos Mahmoud Bsharat, Jiacheng Liu, Xiaohan Zhao, Tianjun Yao, Xinyi Shang, Yi Tang, Jiacheng Cui, Ahmed Elhagry, Salwa K. Al Khatib, Hao Li, Salman Khan, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2606.06481)  

**Abstract**: As AI writing assistants become increasingly integrated into real-world drafting and revision workflows, many documents are no longer purely human-written or AI-generated, but instead result from progressive human-AI co-editing. However, existing AI-text detection benchmarks largely focus on final outputs and provide limited understanding of how AI authorship signals emerge, accumulate, or disappear throughout the revision process. We introduce OpAI-Bench, an operation-guided benchmark for studying progressive human-to-AI text transformation across document, sentence, token, and span granularities. Starting from human-written documents, OpAI-Bench constructs nine sequentially revised versions for each sample under predefined AI coverage levels and five representative AI edit operations, covering four domains while preserving complete authorship provenance at multiple granularities. The benchmark supports comprehensive evaluation with 8 document-level detectors, 7 sentence-level detectors, and 2 fine-grained token/span-level detectors. Experiments reveal that AI-text detectability is governed not only by the proportion of AI-edited content, but also by edit operation, domain, and cumulative revision history. Interestingly, we notice that mixed-authorship intermediate versions are often harder to detect than both fully human and heavily AI-edited endpoints, exposing non-monotonic detection patterns missed by existing benchmarks. OpAI-Bench provides a controlled testbed for analyzing whether, when, and how AI-assisted writing becomes detectable under realistic progressive editing scenarios. Our code and benchmark are available at this https URL. 

---
# Self-Augmenting Retrieval for Diffusion Language Models 

**Authors**: Paul Jünger, Justin Lovelace, Linxi Zhao, Dongyoung Go, Kilian Q. Weinberger  

**Link**: [PDF](https://arxiv.org/pdf/2606.06474)  

**Abstract**: Discrete diffusion language models generate text by iteratively denoising an entire response in parallel. At each step, they predict tentative tokens for every masked position, committing the confident predictions to the output and discarding the unconfident ones. We show that the discarded tokens are in fact a useful lookahead signal for retrieval-augmented generation: even low-confidence tokens often surface salient entities early in the denoising trajectory, enabling retrieval of stronger evidence before the output is finalized. We exploit this through Self-Augmenting Retrieval for Diffusion Language Models (SARDI), a dynamic RAG framework that uses these lookahead tokens to guide retrieval during denoising. SARDI is training-free, retriever-agnostic, and applicable to any reasoning-capable discrete diffusion language model. Across five multi-hop QA benchmarks, SARDI outperforms current training-free diffusion and autoregressive retrieval baselines at up to $8\times$ higher throughput. 

---
# You Only Index Once: Cross-Layer Sparse Attention with Shared Routing 

**Authors**: Yutao Sun, Yanqi Zhang, Li Dong, Jianyong Wang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.06467)  

**Abstract**: Long-context inference in modern LLMs is increasingly constrained by decoding efficiency, especially in reasoning-heavy settings where models generate long intermediate chains of thought. Existing sparse attention methods often face a practical efficiency-quality trade-off. Structured block sparse methods typically provide stronger acceleration but incur noticeable quality loss, while token sparse methods are usually more accurate yet deliver limited end-to-end speedup because top-k routing over the full cache remains expensive. In this work, we propose cross-layer sparse attention (CLSA), which is built on top of KV-sharing architectures such as YOCO. The core idea is to share not only the KV cache across cross-decoder layers, but also the routing index. A single indexer computes token-level top-k selection once and reuses the resulting index across layers, thereby preserving the fine-grained selectivity of token sparse attention while amortizing the routing overhead. The resulting architecture improves all major inference bottlenecks jointly, including pre-filling, KV-cache storage, and long-context decoding. Experiments across short-context and long-context benchmarks show that CLSA is both accurate and efficient, achieving up to 7.6x decoding speedup and 17.1x overall throughput improvement at 128K context. These results suggest a more complete architectural solution for long-context LLMs that jointly advances model quality and inference efficiency. 

---
# Human Adults and LLMs as Scientists: Who Benefits from Active Exploration? 

**Authors**: Mandana Samiei, Eunice Yiu, Anthony GX-Chen, Dongyan Lin, Jocelyn Shen, Blake A. Richards, Alison Gopnik, Doina Precup  

**Link**: [PDF](https://arxiv.org/pdf/2606.06464)  

**Abstract**: A long-standing finding in the causal learning literature is that adults struggle to identify conjunctive causal rules, where an effect requires the simultaneous presence of multiple causes, while performing better in disjunctive settings. However, most demonstrations of this ``conjunctive handicap'' rely on passive observation paradigms with limited evidence, where learners have no control over evidence generation. This paper asks whether this bias persists when adults are granted agency through active exploration. Using a modified ``blicket detector'' task, adult participants freely intervened to identify causal objects under conjunctive or disjunctive rule structures. We show that active exploration substantially improves adults' conjunctive causal reasoning, although conjunctive rules still require more tests to infer than disjunctive rules. We further compare human performance to a range of large language models in the same setting. While some state-of-the-art models approach human-level performance on hypothesis inference accuracy, they often exhibit less efficient exploration strategies and similar conjunctive-disjunctive performance gaps. 

---
# Latent Reasoning with Normalizing Flows 

**Authors**: Guancheng Tu, Xiangjun Fu, Suhao Yu, Yao Tang, Haoqiang Kang, Lianhui Qin, Yizhe Zhang, Jiatao Gu  

**Link**: [PDF](https://arxiv.org/pdf/2606.06447)  

**Abstract**: Large language models often improve reasoning by generating explicit chain-of-thought (CoT), demonstrating the importance of intermediate computation. However, textual CoT forces this computation through a discrete, serial, and communication-oriented token stream: each reasoning step must be verbalized before the model can proceed, even when the underlying update is semantic, uncertain, or only partially formed. Latent reasoning offers a higher-bandwidth alternative by performing intermediate computation in compact continuous states before committing to text. Yet existing latent-reasoning methods often sacrifice key advantages that make CoT effective in autoregressive language models, including native left-to-right generation, probabilistic sampling, compatibility with KV-cache decoding, and tractable likelihood estimation. We propose NF-CoT, a latent reasoning framework that preserves these advantages by modeling continuous thoughts with normalizing flows. NF-CoT instantiates a TARFlow-style normalizing flow inside the LLM backbone, defining a tractable probability model over compact continuous thoughts distilled from explicit CoT. Continuous-thought positions are generated by an NF head, while text positions are generated by the standard LM head within the same causal stream. This design provides exact likelihoods for latent thoughts, enables probabilistic left-to-right decoding with the original KV cache, and supports direct policy-gradient optimization in the latent reasoning space. On code-generation benchmarks, NF-CoT improves pass rates over explicit-CoT and prior latent-reasoning baselines while substantially reducing intermediate-reasoning cost. 

---
# Revising Context, Shifting Simulated Stance: Auditing LLM-Based Stance Simulation in Online Discussions 

**Authors**: Xinnong Zhang, Wanting Shan, Hanjia Lyu, Zhongyu Wei, Jiebo Luo  

**Link**: [PDF](https://arxiv.org/pdf/2606.06443)  

**Abstract**: Large language models are increasingly used to simulate social media users and infer how individuals may respond to online discussions. However, it remains unclear whether these simulations reflect precise user-specific beliefs or whether they are highly sensitive to semantically independent changes in conversational contexts. In this work, we study counterfactual context revision as a framework for auditing LLM-based stance simulation. Given an original online conversation, we first infer a target user's stance toward a specific topic. We then apply controlled revision strategies to the conversational context and simulate the user's stance again under the revised context. We compare text-only revision strategies with a multimodal one that incorporates meme-based context and evaluate two main effectiveness metrics, i.e., average directional stance shift and stance transition rate. The results reveal effective and robust stance transitions in both text-only and multimodal strategies across different polarization-preference mechanisms. Our study contributes an evaluation framework for understanding the context sensitivity of LLM-based stance simulation. More broadly, it highlights both the promise and risk of using LLMs to simulate online opinion dynamics. 

---
# Reinforcement Learning Elicits Contextual Learning of Unseen Language Translation 

**Authors**: Hanxu Hu, Zdeněk Šnajdr, Pinzhen Chen, Jannis Vamvas, Rico Sennrich  

**Link**: [PDF](https://arxiv.org/pdf/2606.06428)  

**Abstract**: Prior work has shown that large language models (LLMs) can translate unseen or low-resource languages by undergoing continued training or even by encoding a grammar book in their context. However, both methods typically overfit specific languages, with limited zero-shot transfer at test time. To translate extremely low-resource languages at scale, we argue that LLMs must acquire the meta-skill of utilizing in-context linguistic knowledge rather than memorizing specific languages. In this paper, we propose a reinforcement learning (RL) approach to unseen language translation given rich linguistic context, using a surface-level translation metric (chrF) as the reward. Empirically, despite the lightweight reward, our RL-trained models effectively extract and apply relevant linguistic information from the provided context, leading to better translations on completely unseen languages than in-context learning or supervised fine-tuning. Our analyses suggest that outcome-based RL can extend beyond conventional reasoning tasks like math and coding to serve as a recipe for language learning from context. 

---
# A Komi-Yazva--Russian Parallel Corpus and Evaluation Protocol for Zero- and Few-Shot LLM Translation 

**Authors**: Petr Parshakov  

**Link**: [PDF](https://arxiv.org/pdf/2606.06420)  

**Abstract**: We present the first Komi-Yazva--Russian parallel corpus together with an explicit evaluation protocol for studying LLM translation in an endangered, extremely low-resource setting. The dataset contains 457 aligned sentence pairs from 74 narrative texts and is accompanied by documented provenance, sentence-level alignment, and story identifiers that enable leakage-aware evaluation. We use this setup to compare modern large language models on Komi-Yazva-to-Russian translation under severe parallel-data scarcity in zero-shot and retrieval-based few-shot regimes. The protocol includes story-level cross-validation, deterministic retrieval for few-shot prompting, strict validation of generated outputs, complementary reference-based and judge-based metrics, and story-level uncertainty estimates. Across models, LLMs produce non-trivial translations, but performance varies strongly by model family and prompting regime. Retrieval-based few-shot prompting consistently improves over zero-shot prompting, while gains beyond a small retrieved context remain limited. The results show that evaluative conclusions in this setting depend materially on metric choice and failure handling, so the paper frames the corpus as both a dataset contribution and a reproducible evaluation testbed for endangered-language machine translation. 

---
# CollabSim: A CSCW-Grounded Methodology for Investigating Collaborative Competence of LLM Agents through Controlled Multi-Agent Experiments 

**Authors**: Jiaju Chen, Bo Sun, Yuxuan Lu, Yun Wang, Dakuo Wang, Bingsheng Yao  

**Link**: [PDF](https://arxiv.org/pdf/2606.06399)  

**Abstract**: Multi-agent systems (MAS) built on large language models have shown growing promise, with their effectiveness resting on agents' ability to coordinate through text-based channels much as human teams do. Yet recent study suggests that MAS often falter not because agents lack individual task-solving ability, but because they lack collaborative competence: the capacity to establish common ground, maintain shared task understanding, balance individual and collective incentives, and repair misalignment as interaction unfolds. Decades of research in Computer-Supported Cooperative Work have characterized these requirements for human teams coordinating under constrained communication, yet existing MAS evaluations focus mainly on task outcomes or single-agent proficiency in reasoning, planning, and tool use. To enable a systematic analysis of agents' collaborative competence in MAS, we introduce CollabSim, a configurable simulation framework that combines a theory-grounded definition of collaborative capabilities, controlled manipulation of interaction conditions, and action-level probing of agents' internal states. Experiments across four LLMs show that CollabSim can capture condition effects, separate model performance patterns, and reveal task-dependent effects of agent design. 

---
# Emergent Language as an Approach to Conscious AI 

**Authors**: Zengqing Wu, Chuan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2606.06380)  

**Abstract**: The question of whether artificial systems can be conscious remains open, in part because existing approaches either evaluate systems against theory-derived checklists (discriminative) or engineer consciousness-inspired modules directly (architectural); both leave open whether observed structures are artifacts of human language priors. We propose a generative methodology: emergent language (EL) in multi-agent reinforcement learning, where agents start from minimal (no language, no concept of self, minimal exposure to human text) and develop communication under task pressure alone, ensuring causal attributability to task demands rather than inherited human language priors. We position our methodology by discussing how EL serves as a generative tool for studying consciousness-relevant structure, including the role of environment complexity and the interpretation of emergent communication. As a proof of concept, we instantiate this methodology in a minimal environment and show that agents develop self-referential communication, including an echo-mismatch detection circuit that is not predicted by task structure or architecture alone but emerges from a specific environmental affordance. 

---
# EDIT: Evidence-Diagnosed Intervention Training for Rule-Faithful LLM Grading 

**Authors**: Zhihao Wu, Linhai Zhang, Taiyi Wang, Runcong Zhao, Peter Andrews, Cesare Aloisi, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2606.06350)  

**Abstract**: Reliable rubric grading requires more than accurate score prediction. Each judgement must be grounded in the mark scheme and evidence from the student answer. Existing credit-assignment and intervention methods, primarily designed for self-contained reasoning tasks such as mathematics reasoning, struggle in this setting because they do not identify where grading reasoning goes wrong or how the model's belief about the final mark changes during reasoning. We propose Evidence-Diagnosed Intervention Training (EDIT), a two-phase framework for training more rubric-faithful LLM graders. First, EDIT-SFT locates problematic reasoning steps using internal model signals: posterior belief over the final mark and input-grounding scores. It then revises only these local steps with help from a rubric checklist. Second, EDIT-RL calibrates the grader with belief-guided reward shaping, penalising large harmful belief drifts while still allowing helpful exploration. Experiments on two real-world, multi-subject grading benchmarks demonstrate that EDIT consistently outperforms strong supervised fine-tuning and reinforcement learning baselines on both in-domain and out-of-domain splits, with ablation studies confirming that internal-state diagnostics drive these gains. 

---
# "Chi nas dal soch el sent de legn" -- Auditing Text Corpora for Lombard 

**Authors**: Edoardo Signoroni, Pavel Rychlý  

**Link**: [PDF](https://arxiv.org/pdf/2606.06349)  

**Abstract**: Several of the world's languages are still under-resourced in terms of Natural Language Processing (NLP) tools. This is mostly due to the lack of high-quality datasets to train, develop, and evaluate systems and models for several tasks, such as Machine Translation (MT). We conduct a manual audit of the parallel and monolingual corpora available for Lombard, an under-resourced language continuum from Italy. Our analysis reveals that the perceived abundance of web-scraped data is an illusion, with massive datasets plagued by severe language misidentification, boilerplate text, and non-linguistic noise. Furthermore, we analyze the orthographic composition of the valid Lombard portions across web-scraped datasets, curated corpora, and benchmarks. Our findings show conflicting orthographical systems and severe representational bias across all corpora: high-quality data is heavily skewed towards Western Lombard varieties, with Eastern ones left on the margins. This underscores the need for variety-aware, community-driven data curation rather than purely quantity-driven scraping. 

---
# Decomposing Factual Sycophancy in Language Models: How Size and Instruction Tuning Shape Robustness 

**Authors**: Victor De Marez, Luna De Bruyne, Walter Daelemans  

**Link**: [PDF](https://arxiv.org/pdf/2606.06306)  

**Abstract**: Factual sycophancy occurs when a language model abandons a correct, verifiable answer under social pressure. Because a flip occurs only when pressure toward a false answer exceeds the model's neutral preference for the truth, flip rates conflate two mechanisms: the strength of that baseline preference (truth margin), and how far pressure shifts it (manipulation sensitivity). We decompose factual sycophancy into these channels and use them to separate the effects of size and instruction tuning across 56 open-weight models spanning 0.3B-32B parameters and 13 manipulation types. We find that vulnerability is governed mainly by size, but instruction tuning changes how size acts: small instruction-tuned models can become less robust, whereas large instruction-tuned models usually become more robust. Instruction tuning primarily increases truth margin, but its behavioral effect depends on manipulation type. Scaling also changes the two channels differently: base models gain margin but become mildly more manipulation-sensitive, whereas instruction-tuned models gain margin faster and become less sensitive. Factual sycophancy is therefore not a single scalar property. Evaluations should report channel-specific, manipulation-specific, and size-conditioned robustness rather than flip rates alone. 

---
# LLMs Can Leak Training Data But Do They Want To? A Propensity-Aware Evaluation of Memorization in LLMs 

**Authors**: Gianluca Barmina, Peter Schneider-Kamp, Lukas Galke Poech  

**Link**: [PDF](https://arxiv.org/pdf/2606.06286)  

**Abstract**: Large language models can reproduce training data, but existing memorization evaluations mostly measure whether models can be forced to do so, rather than whether they do so under ordinary use. We introduce PropMe, a propensity-aware framework for memorization evaluation that contrasts prefix-based capability attacks with non-adversarial evaluations. We propose a metric transformation that, applied to existing functions, allows to create propensity metrics. We further introduce SimpleTrace, a lightweight tracing pipeline built on infini-gram that deterministically attributes model generations to large-scale training corpora and computes verbatim, near-verbatim, and propensity-transformed memorization metrics. Evaluating two fully-open models: Comma and DFM Decoder on two datasets: Common Pile and Dynaword in two languages, we find a consistent gap between capability and propensity: prefix attacks elicit substantially stronger memorization signals than generic or dataset-specific prompts, while propensity scores remain low overall. Thus, the models can reveal training data when directly elicited, but rarely do so in more common non-adversarial settings. We also find that DFM Decoder, which is continually pre-trained from Comma, exhibits reduced memorization and memorization propensity for Common Pile, confirming that memorization capability can decrease when later training emphasizes partially different data. Our results suggest, and we encourage, that memorization audits should report both worst-case extractability and ordinary leakage propensity in order to have a more comprehensive view of this phenomenon. 

---
# FOXGLOVE: Understanding Goal-Oriented and Anchored Writing Feedback from Experts and LLMs on Argumentative Essays 

**Authors**: Yijun Liu, Yifan Song, John Gallagher, Sarah Sterman, Tal August  

**Link**: [PDF](https://arxiv.org/pdf/2606.06271)  

**Abstract**: While large language models (LLMs) are increasingly used to generate writing feedback, there remains no systematic comparison of LLM and expert feedback on the dimensions that writing research identifies as central to revision: goal-orientation, anchoring to specific sentences, and prioritization. We introduce FOXGLOVE, a dataset of 696 feedback comments written by trained writing instructors on 69 twelfth-grade argumentative essays, paired with 1,644 comments generated from four frontier LLMs under a shared protocol, totaling 2,340 comments. We provide expert quality ratings on a subset of both instructor and LLM comments. We find that instructors and LLMs distribute feedback similarly across goals and essay positions, yet instructors and models diverge on the specific sentences on which to provide feedback. Additionally, we find that models tend to write more complex feedback and use fewer questions than instructors. LLM feedback also receives higher ratings on most dimensions of quality, as rated by instructors, but much of this advantage appears to be attributable to lengthier comments. FOXGLOVE enables systematic comparison of where human and LLM feedback align, diverge, and differ. 

---
# Many Circuits, One Mechanism: Input Variation and Evaluation Granularity in Circuit Discovery 

**Authors**: Alireza Bayat Makou, Jingcheng Niu, Subhabrata Dutta, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2606.06267)  

**Abstract**: Circuit discovery methods identify subgraphs that explain specific model behaviors, and structural differences between discovered circuits are commonly interpreted as evidence of distinct mechanisms. We test this assumption by varying input statistics while holding the task fixed, and show that the resulting structural differences exhibit apparent specialization but do not correspond to functional differences, a pattern we term phantom specialization. Using Literal Sequence Copying across four token-frequency bands plus a control condition in five Pythia models (70M-1.4B), we extract 75 circuits and find that structurally distinct circuits implement the same computation: band-specific edges transfer broadly across bands, a core shared across most bands recovers at least 99% of circuit performance, and causal interchange interventions confirm that internal representations are interchangeable across frequency bands. Repeated extractions within the same frequency band further suggest that discovery algorithms sample from an equivalence class of valid subgraphs rather than recovering a unique mechanism. Standard evaluation practice obscures this pattern: source-level evaluation inflates apparent faithfulness, while edge-level evaluation reveals the many-to-one mapping from structure to function. Our results show that structural differences between circuits are not sufficient evidence for distinct mechanisms, and that exposing this requires edge-level evaluation and cross-condition transfer tests. 

---
# From Self to Other: Evaluating Demographic Perspective-Taking in LLM Hate Speech Annotation 

**Authors**: Paloma Piot, Javier Parapar  

**Link**: [PDF](https://arxiv.org/pdf/2606.06266)  

**Abstract**: Hate speech detection is inherently subjective: people from different demographic groups perceive the same content very differently. Collecting enough annotations from multiple demographic groups is costly and difficult to scale. Persona-conditioned Large Language Models (models prompted to adopt a specific demographic identity) have been proposed as a way to simulate diverse perspectives at scale. But do they actually reflect how different groups disagree? We evaluate three aspects of human social judgement: (i) whether personas from different groups disagree in human-like ways (inter-group disagreement), (ii) whether they become more sensitive when content targets their own identity (in-group sensitivity), and (iii) whether they can accurately predict how another group would react (vicarious prediction). Our results show that no model consistently captures all three dimensions, and performance is highly model-dependent and does not emerge reliably from minimal identity prompts alone. However, vicarious prompting with Llama 3.1 yields the highest cross-group agreement in most demographic axes and provides the closest overall approximation to human disagreement patterns, indicating that this configuration may provide a more reliable setting for automatic annotation aligned with human judgements. 

---
# Benchmarking Open-Source Layout Detection Models for Data Snapshot Extraction from Institutional Documents 

**Authors**: AJ Carl P. Dy, Aivin V. Solatorio  

**Link**: [PDF](https://arxiv.org/pdf/2606.06242)  

**Abstract**: Institutional documents contain substantial amounts of operational and analytical information embedded within figures and tables. Current approaches for extracting visual content from documents are largely built around generic document layout analysis, where figures and tables are treated as uniformly relevant document objects rather than semantically meaningful analytical artifacts. In this work, we introduce a benchmark dataset and evaluation framework for \textit{data snapshot extraction}, the task of identifying and localizing semantically meaningful visual artifacts within institutional documents. The benchmark spans humanitarian reports, World Bank policy research working papers, and project appraisal documents, and includes annotations for figures and tables that contain reusable analytical information. Using this dataset, we benchmarked multiple open-source layout detection models and evaluated both detection performance and spatial extraction quality. Our results show that current models struggle to generalize to operational institutional documents despite strong performance on conventional academic benchmarks. Common failure modes include confusion between analytical and non-analytical content, fragmentation of composite analytical artifacts, and incomplete extraction of contextual information required for interpretation. These findings highlight a persistent gap between generic document layout analysis and operationally useful data snapshot extraction. We release the source PDFs, annotation dataset, metadata, and source code to support future research in operational document intelligence. The dataset is available at this https URL and the source code is available at this https URL. 

---
# FiLM-Based Speaker Conditioning of a SpeechLLM for Pathological Speech Recognition 

**Authors**: Fernando López, Santosh Kesiraju, Jordi Luque  

**Link**: [PDF](https://arxiv.org/pdf/2606.06211)  

**Abstract**: Automatic speech recognition (ASR) has advanced remarkably for standard speech; however, pathological speech from neurological conditions remains a significant challenge. We investigate speaker conditioning via Feature-wise Linear Modulation (FiLM), injecting x-vector-derived information into each transformer layer of a frozen ASR encoder to adapt internal representations to individual pathological speakers without modifying base model weights. We benchmark this for the ASR task against standard and parameter-efficient fine-tuning baselines, complemented by post-processing, on Spanish and English pathological speech. Additionally, we evaluate if the adapted model preserves the ability to answer speech-related questions. Results show that speaker-conditioned ASR is competitive with established adaptation strategies while retaining performance on non-conditioned speech. 

---
# Dense Contexts Are Hard Contexts: Lexical Density Limits Effective Context in LLMs 

**Authors**: Giovanni Dettori, Matteo Boffa, Danilo Giordano, Idilio Drago, Marco Mellia  

**Link**: [PDF](https://arxiv.org/pdf/2606.06203)  

**Abstract**: Input length and the position of relevant information are widely cited as the primary causes of degraded LLM long-context performance. Here, we study lexical density -- the rate at which a context introduces distinct information -- as a third, largely overlooked factor that systematically reduces the effective context window of LLMs. We quantify the impact of lexical density on open-weight LLMs (9B-685B) using three "find-the-needle" style benchmarks with identical length (~12k tokens) and controlled needle position, but increasing density of information. We observe a sharp performance collapse in higher-density benchmarks: models that are near-perfect in sparse contexts drop below 60% retrieval score on denser ones. To rule out task-type confounds, we vary and control the density within each benchmark while keeping all other properties unchanged. Reducing density generally restores performance, especially in the high-density regimes where degradation appears. These results show that effective context capacity is a function of lexical density, with direct implications for real-world LLM systems operating on compact, information-rich inputs. 

---
# Improving Answer Extraction in Context-based Question Answering Systems Using LLMs 

**Authors**: Hafez Abdelghaffar, Ahmed Alansary, Ali Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2606.06197)  

**Abstract**: Question answering (QA) systems have achieved notable progress with the advent of large language models (LLMs). However, they still face challenges in accurately extracting and generating precise answers from given contexts, particularly when dealing with complex or ambiguous queries. Existing approaches often struggle with contextual understanding, answer consistency, and generalization across diverse domains. In this work, we propose a question answering system based on large language models, where the input consists of a textual context and a corresponding question, and the output is a concise and accurate answer. The motivation behind this research lies in addressing the limitations of current QA systems, particularly their tendency to produce irrelevant or imprecise responses despite having access to the correct context. Our methodology involves fine-tuning a pre-trained LLM on a benchmark QA dataset to improve its contextual comprehension and answer extraction capabilities. Specifically, we utilize the Stanford Question Answering Dataset (SQuAD1.1), which provides high-quality context-question-answer triplets for supervised training and evaluation. Experimental results show that the fine-tuned Roberta-base model achieves the highest performance, attaining a ROUGE-L score of 86.84%, a BLEU score of 28.24%, and a BERTScore of 95.38%. These results indicate strong accuracy and answer relevance, demonstrating the effectiveness of the proposed approach for context-based question answering tasks. Furthermore, the findings confirm that targeted fine-tuning substantially improves the reliability and precision of QA systems. 

---
# The Tell-Tale Norm: $\ell_2$ Magnitude as a Signal for Reasoning Dynamics in Large Language Models 

**Authors**: Jinyang Zhang, Hongxin Ding, Yue Fang, Weibin Liao, Muyang Ye, Junfeng Zhao, Yasha Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.06188)  

**Abstract**: Recent work has sought to understand Large Language Models (LLMs) reasoning, yet a principled, model-intrinsic signal that captures its layer-wise reasoning dynamics remains underexplored. We bridge this gap by demonstrating that the l2 norm of hidden states serves as an endogenous signal of the model's reasoning intensity. Using Sparse Autoencoders (SAEs) as a diagnostic probe, we observe that LLMs' internal reasoning is marked by a sharp increase in reasoning feature activations concentrated in late layers. Motivated by this pattern, we establish a formal link between reasoning intensity and the model's latent geometry and theoretically prove that the l2 norm of hidden states bounds the activation strength of SAE reasoning features. Empirical correlation analysis and causal interventions further validate the l2 norm as a faithful indicator, where heightened norms consistently correspond to critical reasoning steps. We then introduce three test-time scaling techniques guided by l2 norms: (i) Adaptive Layer-wise Reasoning Recursion, (ii) Endogenous Reasoning State Steering, and (iii) l2-guided Response Selection, which requires no additional training or data and is compatible with advanced inference engines. Experiments across model architectures and benchmarks show that l2-norm-based techniques significantly improve reasoning performance, offering a principled yet simple lens to perceive and control LLM latent reasoning dynamics. Our code is available at this https URL. 

---
# Ouvia: A User-centered Framework for Measuring Usability of Speech Translation in Real-World Communication Scenarios 

**Authors**: Giuseppe Attanasio, Beatrice Savoldi, Daniel Chechelnitsky, Matteo Negri, Marine Carpuat, Maarten Sap, André F.T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2606.06177)  

**Abstract**: Speech translation (ST) is increasingly adopted in user applications, yet its evaluation largely focuses on decontextualized testbeds and holistic quality, rather than end users' communication needs. We introduce Ouvia, an evaluation framework for measuring user-perceived usability of speech translation outputs in real-world settings. Ouvia focuses on one-to-one communication: an English speaker needs to convey a request to a Portuguese speaker, and the message is automatically translated. Through a custom web app and multi-phase study design, we collect more than 1,750 such interactions in healthcare and everyday situations, mediated by four ST systems, involving speakers from three English dialects and two genders. We find that modern ST serves people only to a limited extent -- only around half of interactions are rated as usable -- with significant gaps in reported usability across demographic groups. Moreover, among quality metrics, we find that QA-based evaluation is a substantially stronger predictor of real-world usability than standard approaches. Together, these findings stress the importance of situated, user-centered evaluation frameworks that go beyond holistic quality scores and attend to who the technology serves -- and how well. 

---
# Harnessing Structural Context for Entity Alignment Foundation Models 

**Authors**: Xingyu Chen, Yuanning Cui, Zequn Sun, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.06109)  

**Abstract**: Entity alignment (EA) aims to identify equivalent entities across heterogeneous knowledge graphs (KGs) and is a key component of knowledge fusion and cross-KG reasoning. The recent EA foundation model demonstrates that alignment knowledge, once pretrained, can be directly applied to diverse previously unseen KG pairs. However, it still underuses structural context in two places: cross-KG interaction is weak during encoding, and final candidate ranking still relies too heavily on coarse similarity. We address these limitations with ContextEA, an enhanced encoder-decoder framework for transferable EA. On the encoder side, we introduce a cross-KG interaction encoder that unifies the two KGs with anchor bridges and performs earlier relation-aware cross-graph propagation. On the decoder side, we introduce a structural calibration decoder that calibrates alignment scores with entity-level, neighborhood-level, relation-level, and anchor-aware structural evidence. This design strengthens both structural context construction and structural context exploitation while remaining lightweight. Experiments on 29 EA datasets in OpenEA, SRPRS, and DBP show consistent gains over strong transferable baselines. Notably, the pretrained ContextEA already surpasses the finetuned baselines on all three benchmark groups, demonstrating substantially stronger transfer to unseen KGs. These results suggest that explicitly harnessing structural context is an effective direction for improving EA foundation models. 

---
# IR3DE: A Linear Router for Large Language Models 

**Authors**: Eros Fanì, Oğuzhan Ersoy  

**Link**: [PDF](https://arxiv.org/pdf/2606.06098)  

**Abstract**: Foundational Large Language Models (LLMs) demonstrate proficiency on a wide range of general tasks, and achieve remarkable results on various specialized tasks via domain-expert LLMs. With the ever-growing list of available LLMs, inference routers are being proposed to select the most appropriate LLM for each prompt. However, existing routing methods either optimize cost across weak-to-strong generalist LLMs or require substantial training to support domain-expertise routing. In this paper, we propose IR3DE, a Ridge Regression-based Router for Domain Experts that provides cheap and fast routing decisions for each prompt. We evaluate IR3DE in two Causal Language Modeling (CLM) settings where the tasks are next-token prediction for all domains, and one reasoning setting where each domain has its own distinct reasoning task. Despite being a linear router, IR3DE achieves performance comparable to the other baselines in both CLM settings, and surpassing them in the reasoning setting, with a normalized performance of 98.4%. Moreover, IR3DE enables the addition or removal of new domain experts without requiring the router to be retrained from scratch, allowing a dynamic set of LLMs to be served with minimal disruption to the router itself. Our code is available at: this http URL. 

---
# CHALIS: A Challenge Dataset for Language Identification in Difficult Scenarios 

**Authors**: Michal Tichý, Jindřich Libovický  

**Link**: [PDF](https://arxiv.org/pdf/2606.06088)  

**Abstract**: We present CHALIS (Challenging Language Identification Samples), a new benchmark dataset explicitly designed to address difficult cases in language identification: cousin languages and orthographic noise. Our dataset has two parts: First, we collected sentences shared across mutually intelligible language pairs (Czech/Slovak, Spanish/Catalan, Portuguese/Galician, Danish/Norwegian). The second part tests for orthography noise: we transliterate text across multiple scripts, remove diacritics, simulate homoglyph attacks, and use Internet slang. We evaluate four widely used language identification systems on CHALIS and demonstrate that all struggle substantially in these scenarios, especially on lower-resource languages within cousin pairs and on transliterated input. The resource is publicly available at this https URL. 

---
# LatentSkill: From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents 

**Authors**: Aofan Yu, Chenyu Zhou, Tianyi Xu, Zihan Guo, Rong Shan, Zhihui Fu, Jun Wang, Weiwen Liu, Yong Yu, Weinan Zhang, Jianghao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2606.06087)  

**Abstract**: Agent systems increasingly use textual skills to encode reusable task procedures, but injecting these skills into the prompt at every step incurs substantial context overhead and exposes skill content as plaintext. We present LatentSkill, a framework that converts textual skills into plug-and-play LoRA adapters through a pretrained hypernetwork. LatentSkill stores skill knowledge in weight space rather than context space, removing per-step skill tokens while preserving modular loading, scaling, and composition. On ALFWorld and Search-QA, LatentSkill outperforms the corresponding in-context skill baseline while using substantially fewer prefill tokens: it improves ALFWorld success by 21.4 and 13.4 points on the seen and unseen splits with 64.1% fewer prefill tokens, and improves Search-QA exact match by 3.0 points with 72.2% lower skill-token overhead. Further analysis shows that generated skill LoRAs form a structured semantic geometry, can be precisely controlled via the LoRA scaling coefficient, and can be composed through parameter-space arithmetic when skill components are aligned. These findings suggest that weight-space skills provide an efficient, modular, and less exposed substrate for extending LLM agents. 

---
# SkillComposer: Learning to Evolve Agent Skills for Specification and Generalization 

**Authors**: Qi Zhang, Zhaopeng Feng, Xiaonan Shi, Xiaomeng Hu, Chu Liu, Pengjun Xie, Xiaobin Wang, Jieping Ye, Bryan Hooi, Haobo Wang, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.06079)  

**Abstract**: Agent skills, which consist of reusable strategies that guide agent reasoning and action, have shown strong potential for improving model capability at inference time. However, current skill construction methods treat the problem as one-shot extraction, overlooking a fundamental tension: a skill tailored to the specific task fails to transfer, while the abstracted skill often provides insufficient guidance. We attribute this fragility to the absence of explicit mechanisms for skill specification and generalization. To address this gap, we introduce SkillComposer, a framework that decomposes skill construction into three learnable operations: create, improve, and merge. Trained via systematic rejection sampling recipe, SkillComposer enables language models to self-evolve skills at inference time and supports three deployment modes: offline for building generalized libraries, online for task-specific refinement, and hybrid for combining both. Comprehensive experiments on $\tau^2$-Bench, LiveCodeBench v6, and AppWorld show that SkillComposer consistently outperforms baselines. Our SkillComposer-4B improves a 27B executor by up to +4.5 on agent tasks and +3.4 on code tasks, while generalizing across domains and task types unseen during training. Analysis reveals that merge and improve address orthogonal quality dimensions and that skill composition is a transferable meta-ability, providing a practical recipe for skill-augmented inference. 

---
# Multi-task Learning is Not Enough: Representational Entanglement in Dual-output Second Language Speech Recognition 

**Authors**: Seung Hwan Cho, Young-Min Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.06065)  

**Abstract**: Second-language (L2) speech recognition often requires transcriptions of pronunciations and intended meanings. Multi-task learning (MTL) is a natural approach because it assumes that shared representations benefit both outputs. However, this paper shows that this assumption does not hold across Korean and English. MTL improves meaning but degrades surface transcription, especially in English, where the degradation scales with surface-meaning divergence measured by Levenshtein edit this http URL analysis links these patterns to encoder-level entanglement, with Korean preserving distinct task representations while English produces nearly identical ones. Cross-task decoder analysis shows that the meaning dual-output decoder adapts with a unique representation, while the surface dual-output decoder remains constrained by the encoder. These findings motivate the design of MTL frameworks that mitigate encoder-level entanglement to reduce surface degradation in dual-output L2 automatic speech recognition. 

---
# Automatic Labelling of Speech Translation Errors 

**Authors**: Dominik Macháček, Maike Züfle, Ondrej Klejch  

**Link**: [PDF](https://arxiv.org/pdf/2606.06047)  

**Abstract**: Errors in speech translations reduce trustworthiness of Speech Translation (ST) systems and can have serious consequences. Yet currently there is no established methodology for evaluating confidence and quality estimation of speech translations. To initiate progress in this direction, we propose Speech Translation Error Labelling (STEL). We create an annotation protocol, a small authentic end-to-end evaluation dataset, and we analyse how existing text-only and speech-processing systems perform the STEL task. Our results show that text-only XCOMET and multimodal LLM Qwen2.5-Omni are able to perform the STEL task in roughly half the precision of humans. We also find that direct speech processing is necessary for the STEL task, and that the current text-only and speech-processing systems are complementary in labelling translation-only vs. speech-processing errors in ST. 

---
# IA-RAG: Interval-Algebra-Driven Temporal Reasoning for Dynamic Knowledge Retrieval 

**Authors**: Xiaoman Wang, Yaoze Zhang, Wenzhuo Fan, Hongwei Zhang, Ding Wang, Guohang Yan, Song Mao, Botian Shi, Yunshi Lan, Pinlong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2606.06044)  

**Abstract**: Retrieval-Augmented Generation (RAG) has shown strong effectiveness in grounding Large Language Models (LLMs) with external knowledge. However, existing RAG and Graph RAG frameworks largely treat knowledge as static or associate time with coarse-grained timestamps or metadata, failing to capture rich temporal structures such as duration, overlap, and containment. We propose IA-RAG, a hierarchical temporal RAG framework that models knowledge as time intervals and performs retrieval under formal temporal constraints. IA-RAG represents facts as Interval Event Units (IEUs) and organizes them into a hierarchical Thematic Forest, where temporal dependencies are governed by Allen's Interval Algebra. To handle incomplete or uncertain temporal boundaries, IA-RAG further introduces a Sub-graph Time Tightening mechanism that refines fuzzy intervals through logical constraints within connected event subgraphs. In addition, IA-RAG supports implicit temporal semantic retrieval through interval-algebra-guided traversal. Experiments on multiple temporal question answering benchmarks, including TimeQA, TempReason, and ComplexTR, demonstrate that IA-RAG achieves strong temporal retrieval and reasoning performance, particularly on complex compositional temporal reasoning tasks. Our code is released at this https URL. 

---
# English-to-Prakrit Machine Translation via Multilingual Transfer Learning 

**Authors**: Om Choksi, Smit Kareliya, Shrikant Malviya, Pruthwik Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2606.06038)  

**Abstract**: We study English-to-Prakrit machine translation in a low-resource setting where the target language is unsupported by IndicTrans2. We adapt the multilingual model by mapping Prakrit to the Hindi language tag (hin_Deva) without modifying the tokenizer, vocabulary, or architecture. Using a 1,474-pair Maharashtri Prakrit parallel corpus and evaluation on a 20-sample Ardhamagadhi test set, we report corpus BLEU improvements over an untuned baseline. The results indicate that script-compatible language routing can enable feasible transfer to unsupported classical languages, while highlighting limitations due to data scarcity and dialect mismatch. Our code and trained models are released to the public for further exploration this https URL. 

---
# NAVIRA: Decoupled Stochastic Remasking for Masked Diffusion Language Models 

**Authors**: Andrey Fomenko, Maksim Kryzhanovskiy, Svetlana Glazyrina, Roman Ischenko  

**Link**: [PDF](https://arxiv.org/pdf/2606.06031)  

**Abstract**: Masked diffusion language models generate text by iteratively unmasking many tokens in parallel, but this speed comes with a correction problem: tokens generated in the same step are predicted from marginal distributions, and early local dependency errors can later contaminate the context. PRISM addresses this by learning token-level quality scores and remasking unreliable tokens, but its inference rule is coupled: the same forward pass both detects low-quality tokens and computes logits for their replacements, so the erroneous tokens still condition regeneration. We propose NAVIRA, an inference-time decoding policy that separates these two operations and samples remasking positions stochastically. A first forward pass scores tokens; selected tokens are masked; a second forward pass regenerates from the cleaned context. Temperature-controlled remasking reduces repeated correction of the same positions and balances fluency against diversity. In controlled experiments with a 170M masked diffusion language model, decoupling improves fluency, while scheduled stochastic remasking preserves entropy and achieves stronger LLM-judge scores under larger forward-pass budgets. These results show that remasking policy, not only the learned quality signal, is central to reliable masked-diffusion text generation. 

---
# EGTR-Review: Efficient Evidence-Grounded Scientific Peer Review Generation via Multi-Agent Teacher Distillation 

**Authors**: Xinpeng Qiu, Wang Yihu, Zhifeng Liu, Xiaochen Wang, Jimin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.06025)  

**Abstract**: Scientific peer review generation has attracted increasing attention for reducing reviewing burdens and providing timely feedback. However, existing Large Language Model (LLM)-based methods often produce generic comments with insufficient evidence support and weak source traceability, while complex multi-agent systems incur high inference costs. To address these challenges, we propose EGTR-Review, an Evidence-Grounded and Traceable Review Generation framework via Multi-Agent Teacher Distillation. EGTR-Review first constructs a multi-agent teacher that performs structure-aware paper decomposition, key-element extraction, external scholarly evidence retrieval, evidence-state labeling, verification reasoning, and review synthesis. It then distills both intermediate reasoning trajectories and final review comments into a lightweight student model through task-prefix-driven multi-task learning. An evidence-weighted objective further reduces the influence of weak, missing, or non-verifiable supervision. Experiments on public peer-review datasets show that EGTR-Review (Student) outperforms strong prompt-based, fine-tuned, and structured/agentic baselines across automatic metrics, LLM-as-Judge evaluation, and human evaluation, while maintaining strong factual grounding and source traceability with substantially lower token consumption and inference time. Our code, prompts, configurations, and sample data are available on GitHub. 

---
# Contextualized Prompting For Stance Detection On Social Media 

**Authors**: Tilman Beck, Shakib Yazdani, Simon Kruschinski, Marcus Maurer, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2606.06022)  

**Abstract**: Stance detection on social media is challenging due to short, noisy, and context-dependent language. While large language models (LLMs) show zero-shot generalization, they are typically prompted without contextual information, which limits their ability to interpret ambiguous posts. In this work, we systematically investigate the impact of incorporating real-world (e.g., user biographies), derived (e.g., political party), and LLM-generated (e.g., target descriptions) contextual features into zero-shot prompting for stance detection on Twitter. Our evaluation spans four benchmark datasets, including a new high-quality German Twitter stance dataset. Across multiple LLMs, we find that integrating contextual information improves performance, but only under specific conditions. LLM-generated target descriptions consistently enhance accuracy, while other user metadata has mixed or even detrimental effects. Notably, we show that the inclusion of other tweets by the same user, often beneficial in supervised learning, can impair performance due to input noise. Our qualitative analysis reveals that LLMs struggle to distinguish task-specific useful information from irrelevant context. Our findings highlight both the promise and challenges of prompting with context information in noisy real-world settings. We publish code and data at this \href{this https URL}{page}. 

---
# The Generator-Eraser Paradox: Community Guidelines for Responsible LLM-Assisted Dialect Resource Creation 

**Authors**: Wajdi Zaghouani  

**Link**: [PDF](https://arxiv.org/pdf/2606.06004)  

**Abstract**: Dialect resources occupy a unique position at the intersection of scientific description, cultural preservation, and computational infrastructure. Large language models offer powerful capabilities for accelerating dialect resource development through retrieval-grounded drafting, corpus navigation, metadata enrichment, and annotation workflow support. However, the same systems pose substantial risks: they can contribute to dialect erasure by privileging prestige varieties, homogenizing orthography, and enabling synthetic feedback loops that reduce linguistic diversity over time. These risks are particularly acute for language varieties characterized by diglossia, limited written standardization, or marginalized speaker communities. This paper makes three contributions. First, we integrate insights from variationist sociolinguistics and corpus linguistics to formalize the generator-eraser paradox as a theoretical framework for understanding the dual nature of LLM-assisted dialect work. Second, we derive 12 community guidelines that operationalize this framework into implementable design requirements for dialect resource creation and documentation. Third, we provide an in-depth case study of Arabic dialects, including a structured comparison of widely used resources, to demonstrate how these guidelines address language-specific challenges including diglossia, orthographic variability, and community governance. The contribution is conceptual and operational rather than experimental, with the goal of enabling dialect communities and resource builders across languages to adopt LLMs without sacrificing authenticity, variation, or sovereignty. 

---
# Beyond Alignment: Value Diversity as a Collective Property in Multicultural Agent Systems 

**Authors**: Shaoyang Xu, Jingshen Zhang, Long P. Hoang, Jinyuan Li, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05985)  

**Abstract**: Multicultural multi-agent systems are increasingly deployed in globally diverse settings, where different agents are grounded in different cultural backgrounds. Existing cultural evaluation focuses on value alignment: how closely a single agent matches a target culture. Yet alignment is a per-agent property and cannot reveal whether a system, taken as a whole, preserves the cultural plurality it is meant to represent. We propose value diversity as a system-level evaluation axis for multicultural agent systems, defined through the dissimilarity between culturally conditioned agents' responses on a shared value survey. Using the World Values Survey, we evaluate 19 cultures and 18 backbone models across a wide range of system configurations. We find that diversity is largely uncorrelated with alignment, indicating that the two capture complementary system properties, and that current multicultural agent systems fall substantially below human societies in value diversity. Mixed-backbone systems narrow this gap but do not close it, and the gap persists across culture compositions and agent scales. Social interaction further erodes diversity by driving agents toward consensus, and a participatory budgeting case study shows that this homogenization narrows the breadth of collective decision-making. Together, our results establish value diversity as a distinct evaluation axis for multicultural multi-agent systems and reveal a persistent homogenization tendency in current LLM-based societies. Our code and data are publicly available at this https URL. 

---
# Measuring the sensitivity of LLM-based structured extraction to prompt, model, and schema choices in clinical discharge summaries 

**Authors**: Martin Murin  

**Link**: [PDF](https://arxiv.org/pdf/2606.05970)  

**Abstract**: Large language models are increasingly used for structured extraction from clinical free-text notes, but the sensitivity of their output to upstream configuration choices is less understood than their accuracy on fixed benchmarks. This work measures that sensitivity without human-annotated ground truth, by holding the extraction task fixed and varying one choice at a time. The fixed schema comprises 17 clinical documentation flags on a three-way yes/no/not_documented value set and a 47-tag vocabulary for the primary admission reason. Three prompt variants expressing this schema were each run at two model sizes on MIMIC-IV v3.1 discharge summaries. Cross-prompt agreement was measured by Cohen's kappa on ICD-stratified subsets. A paired same-note comparison isolated the effect of model choice, and a post-hoc collapse of the three-way flags to binary tested the schema's contribution to disagreement. On the three-way flags, the two models reach the same pooled cross-prompt agreement (median kappa 0.69 and 0.68); the larger model raises agreement on some fields and lowers it on others, a redistribution rather than the absence of an effect. Collapsing the schema to binary dissolves most of the cross-prompt disagreement, locating it on the absence-versus-silence distinction rather than on whether the finding is present. On the multi-class admission categorization, changing the model reassigns the dominant tag on close to half of all notes while changing the prompt phrasing reassigns it on roughly one in eight, and the larger model places far less mass on residual catch-all categories (44% to 26%). These patterns indicate a schema-imposed source of disagreement concentrated on the absence-versus-silence axis and a dominance of model over prompt phrasing on multi-class categorization, identified by a reusable methodology for auditing extraction reproducibility on a population-scale deployment. 

---
# Large Language Models are Perplexed by some Political Parties 

**Authors**: Paul Lerner, François Yvon  

**Link**: [PDF](https://arxiv.org/pdf/2606.05937)  

**Abstract**: Large Language Models (LLMs) are increasingly used, including in political applications, but their political fairness has been little studied. We assess it using perplexity, posing that a fair model should give equal probability to all political groups. However, we find, across ten LLMs and three datasets covering 37 languages, that LLMs are more perplexed by the texts of far right and nationalist parties than of social-democratic parties. We find this to be consistent with previous work on translation fairness, to the point that perplexity correlates with downstream translation metrics. Our method is applicable to both base LLMs as well as their instruction-tuned counterpart, and we find that both are highly correlated, suggesting that the political fairness of LLMs stems from their pretraining, and is hardly affected by instruction-tuning. 

---
# Epistemic Injustice in Language Models: An Audit of Pretraining Filters and Guardrails 

**Authors**: Marco Antonio Stranisci, A Pranav, Rossana Damiano, Christian Hardmeier, Anne Lauscher  

**Link**: [PDF](https://arxiv.org/pdf/2606.05936)  

**Abstract**: Modern language models rely on pretraining filters to remove undesirable content from training corpora and inference-time guardrails to suppress undesirable outputs during deployment. In this paper, we examine how these filtering and moderation decisions produce forms of epistemic erasure and reveal tensions both across automated systems and between these systems and human judgment. We audit four pretraining filters and three inference-time guardrails on Common Crawl sentences containing gender and regional-origin mentions, together with a manually annotated subset of 500 sentences. Our analysis shows that filtering and guardrail decisions are strongly associated with blocklist-based lexical cues, while frequently failing to flag content containing private information or explicit hate speech. At the same time, marginalized groups, particularly transgender people, women, and Central Americans, are significantly over-flagged across systems. Human annotators, by contrast, would retain 88.5\% of filter-flagged and 91.3\% of guardrail-flagged content, often recognizing representational harms arising from tensions of content removal that current systems fail to capture. Taken together, our findings document a form of epistemic erasure in which mentions of marginalized groups are disproportionately removed before pretraining and additionally suppressed again at inference time. 

---
# To Be Multimodal or Not to Be: Query-Adaptive Audio-Visual Person Retrieval via Active Modality Detection 

**Authors**: Erfan Loweimi, Mengjie Qian, Kate Knill, Guanfeng Wu, Chi-Ho Chan, Abbas Haider, Muhammad Awan, Josef Kittler, Hui Wang, Mark Gales  

**Link**: [PDF](https://arxiv.org/pdf/2606.05931)  

**Abstract**: When retrieving a person from a video archive by voice and face, should the system be multimodal or not? In real-world broadcast archives, unlike curated benchmarks, a target may be heard but unseen, seen but unheard, or both. Fusing scores from an absent modality injects noise, degrading precision below the best unimodal system. We propose a query-adaptive framework that detects active modalities via cross-modal score consistency: when both modalities are active, files retrieved by one also score highly on the other; this agreement breaks down when a modality is absent. Classifiers driven by these cross-modal features achieve 89% detection accuracy. On the BBC Rewind corpus (with over 12,000 broadcast videos) the adaptive system attains 94.2% P@1, outperforming speaker-only (82.9%), face-only (93.4%), and fixed fusion (90.0%), recovering 64% of the gap to an oracle with ground-truth modality labels (96.6%). 

---
# Better Literary Translation: A Multi-Aspect Data Generation and LLM Training Approach 

**Authors**: Zhihao Lin, Ziqi Zhu, Hao Huang, Guanghui Wang, Peiyang He  

**Link**: [PDF](https://arxiv.org/pdf/2606.05924)  

**Abstract**: Literary translation poses unique challenges due to the scarcity of high-quality annotated data and the need to balance expression fluency with literary effect. We present a multi-aspect iterative refinement framework that generates high-quality translation references and preference data through specialized LLM translators, each targeting a distinct quality dimension. We leverage the generated data for supervised fine-tuning and reinforcement learning. Experiments show that our generated references outperform the original ground truth for SFT by 8.65 CEA100 points. For reinforcement learning, we find that DPO leads to performance degradation in this setting, while leveraging an explicit reward model for GRPO yields an additional 1.51 point improvement. We attribute this to the stability of two-stage training and GRPO's online exploration capability. Our resulting models, LitMT-8B and LitMT-14B, achieve 67.25 and 69.07 CEA100 respectively on the MetaphorTrans English-to-Chinese literary translation benchmark, competitive with Claude Sonnet 4.5 at 68.43, and demonstrate strong generalization to out-of-domain literary work (i.e., O. Henry). 

---
# ACE-SQL: Adaptive Co-Optimization via Empirical Credit Assignment for Text-to-SQL 

**Authors**: Xiaobing Chen, Ai Jian, Eryu Guo, Zhiqi Pang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05906)  

**Abstract**: Text-to-SQL maps natural language questions to executable SQL queries. Modern databases often contain large and complex schemas, making schema linking a critical step for accurate SQL generation. Existing methods either rely on full-schema generation, which leaves schema linking implicit within a large search space, or use a separate retriever trained with static gold-column supervision, whose targets may be suboptimal for the current generator policy. To address this issue, we propose Adaptive Co-optimization via Empirical Credit Assignment for Text-to-SQL (ACE-SQL), a reinforcement learning (RL) framework that jointly optimizes schema retrieval and SQL generation under execution feedback. ACE-SQL constructs an online column-set pool from generator rollouts and derives adaptive on-policy retrieval targets from the column set most frequently associated with execution-correct rollouts. This induces bidirectional adaptation, where the retriever adapts toward column sets that the generator can execute correctly, while the generator adapts to the retriever's evolving schema selections under execution feedback. With approximately 3k synthetic Text-to-SQL question-database pairs for RL training, ACE-SQL achieves 65.3% greedy execution accuracy on BIRD Dev while using 0.93k output tokens per query. The repository is available at this https URL. 

---
# Reducing Hallucinations in Complex Question Answering using Simple Graph-based Retrieval-Augmented Generation (long version) 

**Authors**: Christopher J. Wedge, Joshua Stutter, Danny Dixon, Jacek Cała  

**Link**: [PDF](https://arxiv.org/pdf/2606.05901)  

**Abstract**: Large language models (LLMs) have fundamentally transformed the landscape of Natural Language Processing. Despite these advances, LLMs and LLM-based systems remain prone to a variety of failure modes. Retrieval-augmented generation (RAG) systems have emerged as a common deployment scenario seeking to both avoid the well known risk of the LLM "hallucinating" information, and to enable reasoning and question answering over proprietary information that the LLM did not have access to during training without resorting to expensive model fine-tuning.
In this work, we explore the idea of using a lightweight graph structure with a relatively simple graph schema, to support the RAG subsystem via a dedicated toolset. We design an agentic system with a variety of vector search and graph query tools operating over a structured dataset based on a curated subset of English Wikipedia articles, and evaluate its performance on questions from MoNaCo, a challenging Wikipedia QA benchmark of complex query answering tasks.
Our results show that the introduction of graph-based tools can significantly increase the precision and recall of factual correctness, can halve the number of hallucinated answers, and achieves the highest fine-grained truthfulness score among the three evaluated scenarios. All this with a modest increase in token usage. 

---
# Representing Research Attention as Contextually Structured Flows 

**Authors**: Jessica Rodrigues, Angelo Salatino, Gard Jenset, Scott Hale  

**Link**: [PDF](https://arxiv.org/pdf/2606.05895)  

**Abstract**: Research attention is widely used as an indicator of visibility, influence, and societal uptake, yet it is typically represented as aggregated counts that do not preserve how attention develops across contexts over time. This creates a mismatch between how attention is interpreted and how it is represented. We propose attention flows as contextually structured representations that encode the organisation of attention and its evolution over time. We evaluate whether these representations capture transferable structure by constructing a benchmark based on analogy-style reasoning across research outputs. Comparing signal, sequence, and flow-based representations, we find that flow representations more effectively support structural comparison, particularly in settings where attention is shaped by temporal progression or context distributions. We further show that learned flow representations improve robustness under partial observation and structural perturbation. Overall, these results support modelling attention as a contextually structured phenomenon and provide a basis for more informative approaches to research evaluation. 

---
# EMBER: Efficient Memory via Budgeted Evidence Retention for Long-Horizon Agents 

**Authors**: Yilong Li, Suman Banerjee, Tong Che  

**Link**: [PDF](https://arxiv.org/pdf/2606.05894)  

**Abstract**: Long-horizon agents can archive large histories, but future answers still incur retrieval, rereading, and context costs. When retained memory misses answer-relevant evidence, the system must return to larger portions of the raw history. We study budgeted evidence survival: before the query is known, which source evidence should be retained so that it remains recoverable and usable under a fixed retained source-evidence token budget? We instantiate this setting as Budgeted Pre-Query Retention, where memory is written during ingestion and later read without access to the full raw stream. We introduce EMBER, a learned retention policy that constructs a compact, source-backed evidence state. EMBER stores evidence capsules: verbatim source excerpts paired with retrieval keys and update metadata, preserving both grounding and read-time access. Post-query outcome feedback trains the writer to preserve evidence across the ingestion-retrieval-answer chain. On LongMemEval-RR, our LongMemEval-derived retained-evidence protocol, EMBER-14B reaches 0.3017 F1 at the 8192-token retained-evidence comparison point, compared with 0.1765 for the strongest non-EMBER budgeted baseline. Across retained source-evidence budgets, EMBER improves F1, Retain-Recall, and Read-Recall, indicating that long-horizon memory depends on retaining evidence within the budget rather than rereading larger histories. 

---
# Staying with the Uncertainty: Uncertainty-Scaffolding Strategies for Artificial Moral Advisors in LLM-to-LLM Simulated Conversations 

**Authors**: Salvatore Greco, Hainiu Xu, Jacopo Domenicucci, Yulan He, Sylvie Delacroix  

**Link**: [PDF](https://arxiv.org/pdf/2606.05890)  

**Abstract**: LLMs are increasingly deployed as Artificial Moral Advisors (AMA) in a variety of contexts: what kind of conversational patterns should they display? In this paper, we study how AMA can help their interlocutors "stay with the uncertainty". We propose three modes of uncertainty (Perspective-Multiplying, Tension-Preserving, Process-Reflecting) and compare them against three control conditions (Baseline, Persuasive, Sycophantic). A user-agent LLM engages in a dialogue on an ethical dilemma with an AMA following a specific uncertainty strategy, and completes pre- and post-conversation questionnaires. We further examine the effect of two persona prompt formats (Declarative and Narrative). We found that (1) no single model dominates as a simulated user agent, with open models aligning with human ambiguity through between-persona divergence and closed models through within-persona hedging; (2) declarative personas better capture initial stance diversity while narrative personas show more realistic belief revision; (3) all six AMA strategies produce distinguishable conversational patterns; and (4) uncertainty strategies differ not in how much stance revision they produce, but in the quality of engagement they sustain. 

---
# Evaluating Stochastic Collapse and Implicit Bias in Multimodal Large Language Models 

**Authors**: Huiyuan Zheng, Houtao Zhang, Boyang Wang, Qingyi Si, Hongcheng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.05874)  

**Abstract**: Current evaluations for Multimodal Large Language Models (MLLMs) overwhelmingly focus on utility-driven objectives, leaving model behavior under logic-neutral scenarios largely underexplored. Stochasticity is essential in scenarios where multiple actions are equally valid, such as recommending travel itineraries or daily schedules where multiple options have similar utility. In such settings, deterministic policies may lead to repetitive behaviors and reduced coverage of valid alternatives. To bridge this gap, we propose RandomBench, a benchmark designed to evaluate whether MLLMs can maintain distributionally neutral behavior when selecting among equivalent options. We further introduce three metrics, including RI, BCI, BII, to quantify entropy and distributional bias. Experiments reveal a pervasive phenomenon termed Stochastic Collapse, where MLLMs fail to maintain uniform randomness under explicit random instructions, with top-1 probabilities reaching 97% from the ideal one quarter baseline and RI dropping to 0.068 in Claude Sonnet 4.6. Extensive ablation studies further demonstrate that these deviations persist across languages and representation formats, highlighting the robustness of distributional collapse in logic-neutral decision settings. 

---
# YouZhi: Towards High-Concurrency Financial LLMs via Adaptive GQA-to-MLA Transition 

**Authors**: PSBC LLM Team, Huawei LLM Team, Ruihan Long, Junjie Wu, Tianan Zhang, Duo Zhang, Yaozong Wu, Jinbin Fu, Chang Liu, Zhentao Tang, Wenshuang Yang, Xin Wang, Zhihao Song, Ning Huang, Wenjing Xu, Shuai Zong, Shupei Sun, Sen Wang, Jing Hu, Bin Wang, Xinyu Wang, Junkui Ju, Zequn Ding, Jie Ran, Man Luo, Shixiong Kai, Linkai Hou, Kaichao Liang, Hu Zhao, Yang Zhao, Shucheng Lin, Wei Yu, Chenghan Jiang, Jingjing Ding, Jiahui Zhang, Tian Jin, Yuhang Zhang, Dong Guo, Wei Sun, Jun Xie, Jianwei Li, Lei Cao, Pei Li, Jiabin Li, Jia Yuan, Rui Yuan, Jing Zhu, Mingxuan Yuan, Zhangcheng Lv, Xin Jiang, Xiuhong Fei, Xiaozhe Ren, Yulong Li, Zhipeng Zhang, Hang Wang, Zhaohui Xu, Rui Zhao, Yibo He, Xinzhuang Niu  

**Link**: [PDF](https://arxiv.org/pdf/2606.05868)  

**Abstract**: Large language models (LLMs) drive significant financial innovations, yet their high-concurrency deployment is severely bottlenecked by KV cache memory overhead, which inflates infrastructure costs and throttles scalability. To address this, we propose YouZhi-LLM, a highly efficient financial LLM empowered by a comprehensive structural transition and training pipeline natively built on the Huawei Ascend ecosystem. At its algorithmic core, YouZhi-LLM features a layer-adaptive GQA-to-MLA transition framework that dynamically assigns per-layer FreqFold sizes, maximizing KV-cache compression while minimizing perplexity degradation. To recover representation capacity and inject domain expertise, the Ascend-based training pipeline seamlessly integrates generalized knowledge distillation with financial-specific supervised fine-tuning. Evaluations demonstrate the superiority of this systematic approach, with the adaptive transition reducing perplexity degradation by up to 35% over uniform baselines. Crucially, when evaluated on Ascend NPUs via vLLM-Ascend, the massive KV-cache reduction translates directly into deployment efficiency. Compared to their respective base models, YouZhi-7B yields a 12.3% improvement in average financial benchmark score alongside a 2.69$\times$ increase in maximum concurrency; similarly, YouZhi-14B achieves a 7.0% accuracy gain and a 2.43$\times$ concurrency boost, establishing a new paradigm for cost-effective, high-throughput financial inference. 

---
# Analysis of the Neglect-Zero Effect in Large Language Models 

**Authors**: Jin Tanaka, Daiki Matsuoka, Ryoma Kumon, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2606.05864)  

**Abstract**: We investigate the extent to which the language processing of LLMs resembles human cognitive processes, focusing on a human cognitive bias called the $\textit{neglect-zero effect}$. This effect refers to the human tendency to ignore $\textit{zero-models}$, which are configurations that render a proposition vacuously true by virtue of an empty set. We focus on two types of inferences driven by the neglect-zero effect, and examine how LLMs process these inferences by comparing their behavior with that in an inference that does not involve the neglect-zero effect. For this purpose, we employ a paradigm based on $\textit{structural priming}$, where recent exposure to a preceding sentence (the $\textit{prime}$) facilitates the processing of a subsequent sentence (the $\textit{target}$) due to their structural similarity. We prepare primes to force LLMs to consider the zero-model, and analyze whether they also consider it in the target. The results suggest that the neglect-zero effect may not occur in the LLMs analyzed in this study. Our code is available at this https URL 

---
# TARPO: Token-Wise Latent-Explicit Reasoning via Action-Routing Policy Optimization 

**Authors**: Liting Zhang, Shiwan Zhao, Xuyang Zhao, Zichen Xu, Jianye Wang, Qicheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.05859)  

**Abstract**: Latent reasoning has emerged as a promising alternative to discrete Chain-of-Thought (CoT) in large language models (LLMs), enabling more expressive reasoning by operating over continuous representations. However, the inherently deterministic nature of continuous representations limits policy exploration in reinforcement learning (RL). To address this, we propose TARPO (Token-Wise Latent-Explicit Reasoning via Action-Routing Policy Optimization), a pure RL framework that adaptively switches between discrete token generation and continuous latent reasoning at each step. TARPO introduces a lightweight action head router that observes the current hidden state and samples a routing decision from a binary mode-selection space, preserving the stochasticity of discrete token sampling from the vocabulary. The LLM backbone and router are jointly optimized end-to-end with a shared group-relative advantage signal. Extensive experiments across Qwen2.5 (from 1.5B to 7B) and Llama-3.1-8B backbones demonstrate that TARPO consistently outperforms existing explicit and latent reasoning RL baselines across diverse benchmarks. Further analysis shows that TARPO learns adaptive token-wise switching behaviors while maintaining stable training dynamics. Our code is available at this https URL. 

---
# ReverseEOL: Improving Training-free Text Embeddings via Text Reversal in Decoder-only LLMs 

**Authors**: Ailiang Lin, Zhuoyun Li, Yusong Wang, Keyu Mao, Kotaro Funakoshi, Manabu Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2606.05858)  

**Abstract**: Recent advances in Large Language Models (LLMs) have opened new avenues for generating training-free text embeddings. However, the causal attention in decoder-only LLMs prevents earlier tokens from attending to future context, leading to biased contextualized representations. In this work, we propose Reverse prompting with Explicit One-word Limitation (ReverseEOL), a simple yet effective method for enhancing the representational capability of frozen LLMs. ReverseEOL augments the standard forward embedding with an additional reversed embedding derived from the reversed input text. Since reversing the input exposes each token to context inaccessible in the original order, the resulting reversed embedding effectively provides complementary information to the original one. As a result, combining the forward and reversed embeddings yields a richer final representation. Comprehensive experiments on STS and MTEB benchmarks demonstrate that ReverseEOL significantly improves the performance of existing training-free baselines across a broad range of LLMs with diverse architectures and scales. Extensive ablations and analyses further confirm the necessity of our reversal mechanism. 

---
# Forgive or forget: Understanding the context of hate in audio retrieval systems 

**Authors**: Arghya Pal, Sailaja Rajanala, Raphael C.-W. Phan, Shekhar Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2606.05857)  

**Abstract**: Handling toxic retrieval in text-to-audio systems is challenging due to contextual dependencies. Existing strategies (e.g., rephrasing, summarization) risk altering intent or omitting details. We propose a post hoc causal debiasing framework with a sentiment-controlled mediator to preserve semantic relevance while suppressing harmful speech. Our approach is model-agnostic and integrates seamlessly with existing retrieval pipelines. We introduce two variants: Forgive, which re-ranks and filters toxic audio via logit adjustment, and Forget, which generates counterfactual toxic prompts to mitigate harmful retrievals. Experiments show consistent toxicity reduction with minimal loss in retrieval accuracy, improving both safety and reliability. 

---
# Towards Truly Multilingual ASR: Generalizing Code-Switching ASR to Unseen Language Pairs 

**Authors**: Gio Paik, Hyunseo Shin, Soungmin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.05846)  

**Abstract**: Automatic Speech Recognition (ASR) has become a key technology for human--AI interaction. However, code-switching ASR (CS-ASR) remains particularly challenging due to the severe scarcity of multilingual CS speech resources across diverse language pairs. Existing approaches primarily improve CS-ASR performance through synthetic CS speech generation or pair-specific fine-tuning on limited bilingual datasets. Nevertheless, these approaches face an inherent scalability limitation, as support for CS must be developed separately for language pairs whose number grows combinatorially with the number of supported languages. In this work, we investigate whether CS capabilities learned from a limited set of seen language pairs can generalize to unseen language pairs through model merging and domain generalization methods. Our experiments show that merged bilingual CS-ASR models modestly generalize to unseen language pairs, suggesting limited transfer of bilingual CS capabilities across language pairs. 

---
# Mechanistic Insights into Functional Sparsity in Multimodal LLMs via CoRe Heads 

**Authors**: Ruoxi Sun, Quantong Qiu, Juntao Li, Zecheng Tang, Yihang Lou, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05843)  

**Abstract**: While Multimodal Large Language Models (MLLMs) demonstrate remarkable proficiency on complex vision-language tasks, the mechanisms by which they extract query-relevant visual features from complex, noisy contexts remain opaque. In this paper, we present an in-depth interpretability study that uncovers a profound structural property within MLLMs: functional sparsity in cross-modal retrieval. Leveraging a token-level metric termed Retrieval Attention Mass (RAM), we identify and characterize a highly specialized subset of attention heads, referred to as Context-aware Retrieval (CoRe) heads. Across diverse visual domains and model scales, we observe a clear functional division: CoRe heads act as dedicated information extractors, while most other heads distribute attention over broader contextual regions. Causal interventions further demonstrate the necessity of these specialized heads. Ablating only the top 5% of CoRe heads causes significant degradation in multimodal reasoning performance, whereas ablating lower-ranked heads has minimal effect. Moreover, acceleration experiments validate the utility of CoRe heads, showing that leveraging this localized sparsity significantly accelerates inference while maintaining robust task performance. Our findings reveal a structural principle of functional sparsity within MLLMs, refining the current understanding of mechanistic interpretability and laying a theoretical foundation that can inspire future architecture design and model optimization. 

---
# ProSPy: A Profiling-Driven SQL-Python Agentic Framework for Enterprise Text-to-SQL 

**Authors**: Zhaorui Yang, Huawei Zheng, Sen Yang, Yuhui Zhang, Haoxuan Li, Zhizhen Yu, Xuan Yi, Chen Hou, Defeng Xie, Chao Hu, Minfeng Zhu, Dazhen Deng, Haozhe Feng, Danqing Huang, Yingcai Wu, Peng Chen, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.05836)  

**Abstract**: Large language models have substantially advanced Text-to-SQL systems, yet applying them to enterprise-scale databases remains challenging. Real-world databases often contain large and heterogeneous schemas, incomplete metadata, dialect-specific SQL syntax, and complex analytical questions that are difficult to solve with a single SQL query. To address these challenges, we propose ProSPy, a Profiling-driven SQL--Python agentic framework for enterprise-scale Text-to-SQL. ProSPy structures the reasoning process into four stages: it first extracts fine-grained data evidence through automatic profiling, progressively prunes large schemas into task-relevant contexts, fetches intermediate views through a dialect-agnostic SQL interface, and finally performs flexible downstream analysis with Python. This design combines the efficiency of SQL over large databases with the flexibility of Python-based analysis, while reducing reliance on unreliable metadata and improving robustness across SQL dialects. Experiments on Spider 2.0-Lite and Spider 2.0-Snow show that ProSPy consistently outperforms strong baselines with both open-source and proprietary models, achieving execution accuracies of 60.15% and 60.51% with Claude-4.5-Opus, without majority voting. Further analysis shows that ProSPy is robust to SQL dialect variations and achieves a favorable trade-off between schema recall and precision. 

---
# Can LLMs Be Constrained to the Past? Improving Knowledge Cutoff through Recall-Based Prompting 

**Authors**: Michiro Asai, Ailiang Lin, Yu Kishimoto, Takao Obi, Satoshi Kosugi, Kotaro Funakoshi, Manabu Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2606.05804)  

**Abstract**: Prompted knowledge cutoff instructs a large language model (LLM) to act as if information beyond a specified cutoff date were unavailable. However, prior work mainly relies on direct-answer generation, which struggles when post-cutoff knowledge is not explicitly queried but is only causally related to the question. To address this limitation, we propose two recall-based prompting strategies: Self-Recall (SR), which asks the model to restate its cutoff constraint, and Question-Recall (QR), which requires the model to recall question-relevant information valid under the cutoff. Across three existing benchmarks, our methods outperform both direct-answer prompting and conventional step-by-step reasoning baselines, with particularly strong improvements on counterfactual questions. To investigate robustness across different cutoff settings, we further construct the Multi-cutoff Historical Event Benchmark (MHEB), which evaluates the same question under multiple cutoff years. Results show that knowledge cutoff performance varies with cutoff distance, while combining SR and QR consistently yields the best performance. 

---
# CollabBench: Benchmarking and Unleashing Collaborative Ability of LLMs with Diverse Players via Proactive Engagement 

**Authors**: Hong Qian, Yuanhao Liu, Zihan Zhou, Zongbao Zhang, Hanjie Ge, Haotian Shi, Liang Dou, Xiangfeng Wang, Jingwen Yang, Aimin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.05793)  

**Abstract**: While LLM-based agents excel at individual tasks, effective collaboration with realistic human partners remains challenging. Most of the existing conversation-level collaborative studies lack grounded interaction and behavioral execution, motivating the need for cooperative game environments that enable contextualized and immersive collaboration. To this end, this paper proposes CollabBench, a benchmark for evaluating and training collaborative agents in cooperative games. CollabBench features a Diverse Player Profile Simulation pipeline to model varied players behaviors, and a Collaborative Agentic Training paradigm that unifies reasoning, communication, and action via agentic rollouts, optimized with a hybrid reward balancing task efficiency and affective adaptation. We further extend classic environments to CWAH-MultiPlayer and Cook-MultiPlayer for systematic evaluation under diverse personalities. Experiments with efficiency and affective metrics show that our trained models outperform base models, achieving 19.5% higher efficiency and 24.4% improved affective performance. Further analysis reveals key collaborative limitations of existing models and offers insights for future collaborative training. 

---
# MARDoc: A Memory-Aware Refinement Agent Framework for Multimodal Long Document QA 

**Authors**: Kaifeng Chen, Hongtao Liu, Qiyao Peng, Jian Yang, Yongqiang Liu, Xiaochen Zhang, Qing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05749)  

**Abstract**: Iterative retrieval-reasoning agents have recently shown promise for multimodal long-document question answering. However, most existing systems maintain a single growing context that mixes retrieval traces, observations, and intermediate reasoning. As interactions accumulate, key evidence becomes scattered and diluted, making multi-hop reasoning noisy. We propose MARDoc, a Memory-Aware Refinement Agent framework that decouples long-document QA into three specialized agents: an Explorer for multi-granularity multimodal retrieval, a Refiner for distilling interaction traces into structured evidence and reasoning memories, and a Reflector for checking evidence sufficiency and providing targeted feedback. Across iterations, the agents rely on a dynamically updated structured memory rather than a full accumulated interaction history. This design reduces context noise while preserving answer-critical facts and their logical dependencies. Experiments on MMLongBench-Doc and DocBench show that MARDoc achieves strong results, outperforming same-backbone baselines and demonstrating the effectiveness of structured memory for agentic document QA. 

---
# PlanBench-V: A Spatial Planning Map Benchmark for Vision-Language Models 

**Authors**: Minxin Chen, He Zhu, Junyou Su, Wen Wang, Yijie Deng, Wenjia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05744)  

**Abstract**: Spatial planning maps are central to territorial governance, translating planning objectives, regulations, and spatial strategies into visual forms for decision-making, public communication, and institutional coordination. Their interpretation, however, requires fine-grained visual perception, spatial reasoning, and policy-informed professional judgment, creating major challenges for both human learners and AI systems. With the rapid progress of Vision-Language Models (VLMs), their use in urban planning analysis is gaining attention, yet existing multimodal benchmarks mainly target general visual understanding and overlook the domain-specific cognitive processes of planning practice. To address this gap, we introduce PlanBench-V, the first comprehensive benchmark for evaluating VLMs in spatial planning map interpretation. We first build the Spatial Planning Map Database (SPMD), an expert-annotated dataset of 223 planning maps and 1629 question-answer pairs curated by professional planners, covering diverse geographic regions and cartographic styles. We then propose a theory-informed evaluation framework assessing four progressive capabilities: Perception, Reasoning, Association, and Implementation, corresponding to the cognitive pipeline of planning map interpretation. Extensive experiments across two generations of VLMs show clear progress but persistent limitations. The best 2026 agentic reasoning model, Qwen3.6-Plus, substantially outperforms the best 2025 model, GPT-4o, by 27%. Nevertheless, all models still struggle with implementation-oriented tasks requiring evaluative judgment, policy sensitivity, and constraint-aware decision-making. These findings reveal fundamental limitations of current VLMs in professional planning contexts and highlight the need for domain-adaptive multimodal reasoning frameworks. Code and data are available at this https URL. 

---
# AdaPLD: Adaptive Retrieval and Reuse for Efficient Model-Free Speculative Decoding 

**Authors**: Runheng Liu, Jincheng Xie, Wen Hu, Xingchen Xiao, Heyan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05742)  

**Abstract**: Speculative decoding accelerates generation by verifying multiple drafted tokens in a single target-model forward pass, reducing sequential decoding iterations. Model-free variants avoid auxiliary draft models by reusing text and model states already available during generation, but their speedup depends on the reliability of the constructed drafts. We identify two limitations of existing reuse-based methods: lexically anchored retrieval has limited recall under surface-form variation, and deterministic span copying can be brittle when the retrieved context does not uniquely determine the continuation. We propose \emph{AdaPLD}, a training-free method that adaptively improves both retrieval and draft construction. AdaPLD preserves high-precision lexical reuse while using semantic similarity to recover additional reuse opportunities when lexical matching fails. It further constructs branched reuse hypotheses to account for continuation uncertainty, rather than relying on a single copied span. Across diverse benchmarks, AdaPLD reduces target-model forward passes and achieves up to $3.10\times$ decoding speedup. 

---
# Narrative Knowledge Weaver: Narrative-Centric Retrieval-Augmented Reasoning for Long-Form Text Understanding 

**Authors**: Qiuyu Tian, Fengyi Chen, Yiding Li, Youyong Kong, Fan Guo, Yuyao Li, Jinjing Shen, Zhijing Xie, Yiyun Luo, Xin Zhang, Yingce Xia, Zequn Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.05724)  

**Abstract**: Long-form narrative QA requires reasoning over evolving story worlds rather than isolated passages: answers may depend on earlier goals, changing character states, social relations, causal triggers, temporal position, and later consequences. Existing retrieval and graph-augmented generation methods improve evidence access, but their units--chunks, entities, relations, summaries, or tool actions--do not directly encode how evidence functions in a story. We introduce Narrative Knowledge Weaver(NKW), a source-grounded framework that aligns textual evidence, atomic facts, canonical graph structure, entity profiles, interactions, episodes, and storylines. At query time, NKW uses text, graph, and narrative tools with post-retrieval reading skills to assemble evidence and audit actor, scope, polarity, state, and temporal constraints. Across STAGE, FairytaleQA, and QuALITY, NKW is strongest on screenplay-level story-world QA while remaining competitive on more passage-centered benchmarks. Ablations, question-type analyses, graph-asset statistics, and case studies show complementary benefits for character, scene, temporal, causal, and narrative-progression reasoning. 

---
# Interpreting Style Representations via Style-Eliciting Prompts 

**Authors**: Junghwan Kim, David Jurgens  

**Link**: [PDF](https://arxiv.org/pdf/2606.05716)  

**Abstract**: Style representation learning is a powerful tool for authorship analysis and modeling writing style, yet the latent nature of learned representations makes them difficult to interpret. Recent work has attempted to explain these representations by generating natural language descriptions with large language models (LLMs) conditioned on input text. However, such descriptions are often prone to the LLM's biases and hallucinations, and they lack an explicit objective and practical utility. In this work, we propose a novel framework for interpreting style representations through style-eliciting prompts: natural language instructions designed to steer LLMs to generate text that reflects specific stylistic attributes. We curate 1,010 distinct style features spanning 26 stylistic categories and construct a dataset by prompting an LLM to generate text conditioned on these features. Using this data, we train a decoder to generate a style prompt from the style representation of the generated text. We evaluate our approach on three tasks: (1) recovering original style prompts from generated text, (2) generating text in the same style using the recovered prompts, and (3) steering LLM outputs to match the style of human-written texts. Experiments demonstrate that our method consistently outperforms strong baselines that directly prompt LLMs with target text, achieving superior performance in both style description and style imitation. These results highlight that style-eliciting prompts can provide a practical and interpretable interface to stylistic information encoded in style representations. 

---
# Beyond tokens: a unified framework for latent communication in LLM-based multi-agent systems 

**Authors**: Yingzhuo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.05711)  

**Abstract**: Multi-agent systems built on large language models (LLMs) have become a prevailing paradigm for tackling complex reasoning, planning, and tool-use tasks. The dominant communication protocol in such systems is natural language: agents exchange messages token-by-token, verbalising their internal reasoning so that peers can read, verify, and respond. While convenient and interpretable, this protocol suffers from three structural drawbacks -- high inference cost, irreversible information loss during discretization, and ambiguity/redundancy of natural language. A growing body of work therefore explores an alternative protocol -- latent communication -- in which agents exchange continuous representations (embeddings, hidden states, or KV-caches) directly, bypassing the bottleneck of text generation. This paper presents a unified framework for organising the rapidly expanding literature on latent communication. We analyse existing methods along three orthogonal axes: (1) WHAT information is communicated (Embeddings, Hidden States, KV-Caches, or other continuous state); (2) WHICH sender-receiver alignment is used (latent-space alignment and layer alignment); and (3) HOW the communicated information is fused into the receiver (concatenation, prepending, mathematical operations, cross-attention, or cache restoration). Under this 3-axis framework, we systematically categorise eighteen representative methods proposed between 2024 and 2026, identify five major design patterns, and surface a set of open challenges -- including cross-architecture alignment, security of latent channels, compression for edge deployment, and the relationship between latent communication and latent chain-of-thought. We hope that this framework both lowers the barrier to entry for new researchers and provides a vocabulary for comparing future work. 

---
# Rethinking LoRA Memory Through the Lens of KV Cache Compression 

**Authors**: Chunsheng Zuo, Liaoyaqi Wang, William Jurayj, William Fleshman, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2606.05698)  

**Abstract**: Parametric retrieval augmentation encodes document information into lightweight, document-specific modules such as LoRA adapters, reducing the need to include all evidence as input context. However, it remains unclear how this parameter-side memory interacts with context-side memory stored in the KV cache. We study this interaction in document-level question answering by progressively evicting document key-value states and measuring when a document LoRA contributes beyond the retained context. We find that document LoRA adds little when the KV cache is largely intact, but becomes increasingly useful under aggressive compression, recovering 13-21 ROUGE-L points when no document context remains. The gain is largest when the base model encodes the document, and the adapter is applied only during answer generation, suggesting that document LoRA is better understood as decoding-time parametric memory than as a document encoder. Finally, QA-style supervision produces substantially stronger adapters than raw-context next-token-prediction. These results position document LoRA as a complementary memory channel whose value emerges precisely when context-side evidence is scarce. 

---
# Value-and-Structure Alignment for Routing-Consistent Quantization of Mixture-of-Experts Models 

**Authors**: Hancheol Park, Geonho Lee, Tairen Piao, Tae-Ho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.05688)  

**Abstract**: Mixture-of-Experts (MoE) models scale foundation models efficiently by activating only a subset of experts for each token, but their large number of expert parameters still makes quantization essential for practical deployment. Unlike dense models, however, MoE models are sensitive to routing instability: small quantization-induced perturbations can change the top-$k$ expert selection, altering the computation path and degrading model quality. We propose Value-and-Structure Routing Alignment for Quantization (VSRAQ), a MoE-specific post-training quantization objective that preserves pre-quantization expert-selection behavior under quantization. VSRAQ combines two complementary objectives that jointly preserve expert-selection behavior: value alignment, which matches routing-relevant logits or scores, and structure alignment, which preserves expert ordering and top-$k$ decision boundaries. By maintaining routing consistency, VSRAQ reduces quantization-induced degradation without introducing any inference-time overhead and can be integrated into existing quantization frameworks. Experiments on recent MoE foundation models show that VSRAQ improves expert-selection consistency and consistently outperforms reconstruction-only and router-aware baselines. 

---
# QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation 

**Authors**: Dike Sun, Zheng Zou, Jingtong Zang, Qi Sun, Huaipeng Zhaoand Tao Luo, Xiaoyi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2606.05671)  

**Abstract**: Query recommendation in e-commerce search aims to proactively suggest queries that match users' potential interests. However, existing methods mainly optimize query-level relevance, while neglecting whether the retrieved products align with users' downstream preferences. This mismatch often leads to high query click through rates (CTR) but low product conversion rates (CVR). To bridge this gap, we propose QueryAgent-R1, a memory-augmented agentic framework that improves end-to-end alignment via chain-of-retrieval optimization. Our QueryAgent-R1 grounds query generation in real inventory retrieval, allowing the agent to validate and refine queries based on retrieved products. We also design a consistency reward in the agentic reinforcement learning (RL) process to jointly optimize query relevance and downstream engagement. In addition, we construct a memory abstraction module for efficient user profiling. To support offline evaluation, we construct two datasets based on both proprietary industrial data and public datasets, on which QueryAgent-R1 consistently outperforms strong baselines. Moreover, on a large scale production platform, QueryAgent-R1 improves Query CTR by 2.9% and guided CVR by 3.1% in online A/B tests. 

---
# Bootstrapping Semantic Layer from Execution for Text-to-SQL 

**Authors**: Youngwon Lee, Jaejin Kim, Seung-won Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05634)  

**Abstract**: Real-world text-to-SQL is often under-specified until user phrases are grounded in how the database stores values. Prior work attempts to address this by requiring a semantic layer to specify groundings in advance, but such specifications are often incomplete, especially in expert domains where domain-specific conventions are under-documented. As this leaves multiple grounding hypotheses open for the same SQL part, we introduce GATE (Grouding After Test from Execution), which bootstraps missing groundings from execution feedback. GATE keeps grounding hypotheses open while executing the already grounded parts to obtain observations. Then, only the hypothesis supported by that observation is grounded and stored as a memory entry, recording what was tested and how the open part should be written in SQL. These entries accumulate into execution-grounded memory, allowing later steps to reuse supported groundings. Across real-world and controlled benchmarks, GATE consistently improves over strong baselines, demonstrating that execution can serve not only as validation but also as a bootstrapping mechanism for reusable memory in text-to-SQL. 

---
# When New Generators Arrive: Lifelong Machine-Generated Text Attribution via Ridge Feature Transfer 

**Authors**: Zhen Sun, Yifan Liao, Zhicong Huang, Jiaheng Wei, Cheng Hong, Yutao Yue, Xinlei He  

**Link**: [PDF](https://arxiv.org/pdf/2606.05626)  

**Abstract**: Machine-generated text (MGT) attribution aims to identify the specific generator responsible for a given text, thereby providing fine-grained evidence for model accountability and misuse investigation. As new large language models continue to emerge, attribution models must continuously incorporate new generators while preserving their ability to recognize previously seen ones. Prior works have shown that this lifelong MGT attribution setting is challenging, and existing methods often struggle to achieve a stable balance between adapting to new classes and retaining old ones. To address this issue, we propose RidgeFT, a lightweight analytic update framework that does not rely on exemplar replay. RidgeFT trains a task-aware encoder on the initial generator set, stores compact class-wise sufficient statistics when each generator class is first observed, and then freezes the encoder for replay-free closed-form updates. It then suppresses generator-irrelevant variation through covariance calibration, improves representation capacity with fixed random features, and updates new classes through closed-form ridge regression based on class-level sufficient statistics. Across multi-topic evaluations with varying initial generator setups, RidgeFT consistently outperforms baselines. It achieves the best macro-F1 across domains, backbones, and incremental protocols, while also improving both old-class retention and new-class adaptation. These results suggest that feature-stable analytic updates provide a simple yet effective approach to lifelong MGT attribution. 

---
# AdaPlanBench: Evaluating Adaptive Planning in Large Language Model Agents under World and User Constraints 

**Authors**: Jiayu Liu, Cheng Qian, Zhenhailong Wang, Bingxuan Li, Jiateng Liu, Heng Wang, Jeonghwan Kim, Yumeng Wang, Xiusi Chen, Yi R. Fung, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2606.05622)  

**Abstract**: Planning for real-world problems by language models often involves both world and user constraints, which may not be fully specified upfront and are progressively disclosed through interaction. However, existing benchmarks still underexplore adaptive planning under such progressively revealed dual constraints. To address this gap, we introduce AdaPlanBench, a dynamic interactive benchmark for evaluating whether Large Language Model (LLM) agents can adaptively plan and re-plan under progressively revealed world and user constraints. AdaPlanBench is built on 307 household tasks, with a scalable constraint construction pipeline that augments each task with dual constraints. At runtime, agents interact with the environment in a multi-turn protocol where hidden constraints are revealed only when the agent proposes a plan that violates them, requiring iterative plan revision under accumulating feedback. This makes planning challenging, as agents must infer and track constraints from feedback while re-planning effectively. Experiments on ten leading LLMs show that adaptive planning under dual constraints remains challenging, with the best model reaching only 67.75% accuracy. We further observe that performance degrades as more constraints accumulate, with user constraints posing a particularly large challenge and failures often stemming from weaker physical grounding and reduced effectiveness. These results establish AdaPlanBench as a testbed for dual-constrained interactive planning and highlight the challenge of reliable adaptation to dynamically revealed constraints in LLM agents. 

---
# An ERP Study on Recursive Locative Processing in Mandarin-Speaking Children with Autism 

**Authors**: Xiaoyi Wang, Chenxi Fu, Ziman Zhuang, Caimei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05620)  

**Abstract**: Recursion enables the generation of hierarchical linguistic structures but imposes substantial processing demands during real-time comprehension. While difficulties with complex syntax have been reported in autism spectrum disorder (ASD), the temporal dynamics of recursive processing remain poorly understood. This study used event-related potentials (ERPs) to examine how Mandarin-speaking children with ASD process two-level recursive locative constructions. Twenty-four children (12 ASD, 12 typically developing, TD) participated in a cross-modal sentence-picture matching task. Neural responses were analyzed across three processing stages associated with structural prediction (P200), semantic integration (N400), and syntactic reanalysis (P600), with mental age controlled. Results revealed a systematic divergence between groups. TD children showed clear P200 and P600 modulation in response to structural mismatch, whereas ASD children exhibited attenuated early differentiation and reduced late reanalysis effects. In contrast, ASD children showed enhanced N400 responses under mismatch conditions, indicating increased semantic integration demands. In addition, the ASD group displayed significantly greater inter-individual variability in hemispheric lateralization, although lateralization strength was not associated with receptive vocabulary performance. These findings support a cascading account in which reduced early predictive engagement in ASD leads to increased integration costs and diminished reanalysis efficiency during recursive processing. More broadly, the results highlight the importance of both temporal processing dynamics and neural variability in understanding language differences in ASD. 

---
# What's in a Name? Morphological Shortcuts by LLMs in Pharmacology 

**Authors**: Kaijie Mo, Thomas Yang, Chantal Shaib, Qing Yao, William Rudman, Ramez Kouzy, Kanishka Misra, Byron C. Wallace, Junyi Jessy Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.05616)  

**Abstract**: The morphological form of a word can often give cues to its meaning, but purely relying on these mappings can lead to overgeneralization in high-stakes domains. In the medical domain, for instance, LLMs can confidently reason about fictitious drugs from their affixes alone (e.g., wugcillin) and generate plausible-looking clinical content. We present a behavioral and mechanistic study of LLM "affix heuristics" in pharmacology. Using fictitious drug names built from real affixes, we show that affix signals alone elicit class-level pharmacological responses. We introduce a framework for identifying whether a model's drug semantics are driven mainly by the affix, the stem, or the drug name as a whole. Applied across 653 drugs, our framework reveals that models often induce drug meaning primarily through affix cues, yet rarely explicitly indicate this reliance, and sometimes incorrectly conflate properties among affix-sharing drugs. Activation patching across models further localizes this behavior to early-mid layers. These findings show that morphological shortcuts pose a subtle but measurable risk to safety. 

---
# Predictable Scaling Laws of Optimal Hyperparameters for LLM Continued Pre-training 

**Authors**: Yongwei Zhou, Juncheng Diao, Junlin Shang, Peiguang Li, Rongxiang Weng  

**Link**: [PDF](https://arxiv.org/pdf/2606.05610)  

**Abstract**: The efficacy of continued pre-training for Large Language Models (LLMs) hinges upon hyperparameter configurations, such as learning rate and batch size. However, current practices often rely on heuristics or grid searches, leading to training instability and excessive costs. In this work, we first empirically discover that optimal hyperparameters follow stable and predictable scaling laws throughout the continued pre-training process. Leveraging these insights, we propose a novel framework to establish quantitative relationships between compute budget and optimal hyperparameters for a given checkpoint. Our approach has two stages: (1) \textit{Empirical Law Discovery}, where we train small-scale proxy models to derive functions mapping compute budget to optimal hyperparameters via standard loss-compute scaling laws; and (2) \textit{State-Aware Hyperparameter Prediction}, where we evaluate an initial checkpoint's validation loss and use the inverse scaling law to estimate its \textit{equivalent pre-training compute} -- the compute needed to achieve the same loss from scratch. Combining this with the planned compute budget, we predict optimal hyperparameters for the target run. Empirical results demonstrate that our method reduces the hyperparameter search overhead by up to 90\% while achieving comparable or superior performance relative to baselines. This model-agnostic framework generalizes across architectures, providing a principled and efficient methodology for diverse continued pre-training scenarios starting from any given point. 

---
# TensorBench: Benchmarking Coding Agents on a Compiler-Based Tensor Framework 

**Authors**: Bobby Yan, Fredrik Kjolstad  

**Link**: [PDF](https://arxiv.org/pdf/2606.05570)  

**Abstract**: Repository-level coding benchmarks face a trade-off between task difficulty and evaluation reliability: tasks that challenge frontier models often involve large codebases with incomplete test coverage, while human review does not scale. We introduce TensorBench, a benchmark of 199 feature-addition and refactoring tasks on an open-source compiler-based tensor framework that extends PyTorch with first-class support for dense and sparse tensors. Tasks cover new sparse formats, dense optimization passes, IR transformations, scheduler changes, runtime components, and high-level numerical operators. TensorBench grades each run by applying the agent's patch and running the framework's test suite, which includes the pre-existing randomized regression tests and any tests the agent adds. For feature-addition tasks, a pass means that the patched repository preserves the tested pre-existing behavior and satisfies the agent-added checks for the requested feature. We evaluate seven coding agents spanning three frontier model families and one open-weight model. Pass rates under this criterion range from $64.8\%$ for the strongest agent to $22.1\%$ for the weakest. Agents pass different subsets of tasks: pairwise Cohen's $\kappa$ ranges from $-0.07$ to $0.43$, with $\kappa = 0.05$ for the two strongest agents. 

---
# Domain-Aware Mispronunciation Detection and Diagnosis Using Language-Specific Statistical Graphs 

**Authors**: Huu Tuong Tu, Hanh Nguyen, Thien Van Luong, Nguyen Tien Cuong, Vu Huan, Nguyen Thi Thu Trang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05569)  

**Abstract**: Mispronunciation Detection and Diagnosis (MDD) has gained increasing importance in computer-assisted language learning and speech technology in recent years. In this paper, we propose a method for constructing statistical graphs that enable models to learn phoneme confusion patterns represented as directed graphs. Furthermore, we introduce a language-specific strategy to capture systematic pronunciation differences across various native language (L1) backgrounds. The effectiveness of our approach is demonstrated through extensive experiments on the L2-ARCTIC benchmark, where it achieves an F1-score of 59.52%, outperforming several competitive baselines. 

---
# Using Large Language Models to Support High Volume Application Review for an Undergraduate Research Program 

**Authors**: Varun Aggarwal, Kay Kobak, John Howarter  

**Link**: [PDF](https://arxiv.org/pdf/2606.05564)  

**Abstract**: Undergraduate research programs such as the Summer Undergraduate Research Fellowship (SURF) at Purdue University receive thousands of applications every year, requiring significant time and effort for program staff to evaluate each submission consistently and within tight timelines. This work-in-progress paper describes the development and initial deployment of a large language model (LLM)-based tool to assist in the evaluation of approximately 1,200 student Statements of Purpose (SoPs) for the SURF 2026 cycle at Purdue University. The workflow utilizes OpenAI GPT models (GPT-4o, GPT-5-mini, and GPT-5.2) and uses a structured rubric across six subcategories, each scored on a 0-3 scale. A few SoPs, graded by program staff, were used to tune the model responses. The model prompt was designed to generate both numerical scores, rationales (including positive and negative aspects) and short excerpts from each submission. Using GPT-5.2, the full batch of 1,200 SoPs was processed in approximately 4.6 hours of compute time, averaging roughly 14 seconds per SoP (with per-SoP timing varying with SoP length, which ranged from 500 to 2,000 words). Notable differences in rubric adherence were observed across model versions, with GPT-5.2 adhering most closely. Disagreement in model scores was more pronounced for lower-scoring submissions. The LLM outputs replicated the role previously played by distributed human graders, providing the program coordinator with scored and rationale-annotated outputs for the entire applicant pool. The program coordinator then reviewed these outputs alongside each applicant's SoP, applying the same downstream office criteria used in prior SURF cycles, to produce a shortlist of strong candidates. This coordinator review was completed in approximately 4 hours, compared to the multi-week coordination effort required in prior program cycles. 

---
# InfoShield: Privacy-Preserving Speech Representations for Mental Health Screening via Information-Theoretic Optimization 

**Authors**: Xueyang Wu, Siyuan Liu, Kezhuo Yang, Guang Ling  

**Link**: [PDF](https://arxiv.org/pdf/2606.05561)  

**Abstract**: Speech-based mental health screening offers scalable depression detection, yet clinical deployment faces a significant barrier: users' privacy concerns about demographic information exposure. Current techniques struggle to resolve this conflict. Adversarial training often fails against unseen threats, whereas Differential Privacy tends to compromise diagnostic performance by injecting noise across all features. This paper presents InfoShield, which minimizes mutual information between speech representations and sensitive attributes while preserving depression classification accuracy. We identify that standard MINE estimators struggle with sequential speech due to temporal-static misalignment, and introduce TimeAwareMINE with cross-modal attention to align acoustic frames with attribute embeddings. Experiments on the Androids Corpus show InfoShield reduces gender inference from 92.6\% to 55.5\% and age inference from 55.7\% to 30.3\% with limited utility loss (6\% F1 reduction), achieving F1=0.784 compared to prior SOTA's 0.723. 

---
# AURA: Intent-Directed Probing for Implicit-Need Surfacing in Situated LLM Agents 

**Authors**: Yang Li, Jiaxiang Liu, Jiang Cai, Mingkun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.05557)  

**Abstract**: A situated query like "where is Lin Wei?" often encodes more than its literal content: the user may also want to know whether Lin Wei is free, in a good mood, or worth interrupting now. Standard tool-use agents answer the literal question and stop. AURA inserts an inference step between scene perception and tool use that produces an IntentFrame: a structured estimate of the implicit need with a scalar gap score that controls per-query probe budget and tool selection. On a 100-query four-scene implicit-intent benchmark, AURA improves implicit-need coverage over ReAct-style probing (Delta = +0.07, p < 10^-6); three of four scenes are individually significant, the gain reproduces on a second backbone, and a prompt ablation attributes the lift to gap calibration rather than answer memorisation. On factual lookup the controller trades raw accuracy for 82% fewer probes and zero forbidden-tool violations on a privacy-sensitive slice; scope conditions are detailed in Limitations. Code, simulator, and benchmark are released at this https URL. 

---
# ArcANE: Do Role-Playing Language Agents Stay in Character at the Right Time? 

**Authors**: Woojung Song, Nalim Kim, Sangjun Song, Chaewon Heo, Jongwon Lim, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2606.05553)  

**Abstract**: Role-playing language agents (RPLAs) should play characters whose values and behavior evolve as the story progresses, not maintain a fixed persona. Existing benchmarks measure factual recall at a given chapter, not whether responses align with the character's psychological trajectory, especially in scenarios the source text never explores. We introduce ArcANE (Arc-Aware Narrative Evaluation), an automatically constructed benchmark spanning 17 novels and 80 principal characters. A Character Arc segments the narrative into phases along a psychological axis, and each probe poses the same scenario across phases, spanning both situations within the source text and situations beyond it. Across six models and six context modes, conditioning on the Character Arc tops every other context strategy on every model, and the gap is largest on scenarios outside the source text where retrieval has nothing to find. We further fine-tune open-weight models on the same data to obtain ArcANE-8B/32B, which widen the Arc advantage even more on scenarios outside the source text. 

---
# Multilingual Detection of Alzheimer's Disease from Speech: A Cross-Linguistic Transfer Learning Approach 

**Authors**: Nadine Yasser Abdelhalim, Emmanuel Akinrintoyo, Nicole Salomons  

**Link**: [PDF](https://arxiv.org/pdf/2606.05545)  

**Abstract**: The development of multilingual Alzheimer's Disease Dementia (AD) detection models presents significant challenges due to the resource-intensive and time-consuming nature of language-specific model training. We propose a novel solution using cross-language training to detect AD in languages beyond those used for model training. This study investigates multilingual deep learning models for detecting AD across different languages and cognitive impairment levels. Using datasets in English, Chinese, Arabic, and Hindi, we developed transformer-based models for binary AD classification. Our approach achieved F1 scores of 82\% across all languages, demonstrating strong cross-linguistic generalization. The rapid inference time (0.5 seconds) supports potential real-time screening applications, while consistent performance across languages indicates feasibility for global deployment. 

---
# CHASE: Adversarial Red-Blue Teaming for Improving LLM Safety using Reinforcement Learning 

**Authors**: Rahul Markasserithodi, Aditya Joshi, Yuekang Li, Ishmanbir Singh, Chris Yoo, Alan Niu  

**Link**: [PDF](https://arxiv.org/pdf/2606.05523)  

**Abstract**: Despite advances in safety alignment, prompt-rewriting attacks such as persona modulation, fictional framing and persuasion-based reformulation, can bypass safety filters even on frontier models. Existing defenses either rely on non-scalable human curation or white-box optimisation that overfits to specific model internals, leaving aligned models brittle against the very class of adaptive black-box adversaries they will face in deployment. To address this gap, we introduce CHASE (Co-evolutionary Hardening through Adversarial Safety-Escalation), a closed-loop red-blue teaming framework in which a black-box attacker and a safety-aligned defender co-evolve. The attacker is trained via Group Relative Policy Optimization (GRPO) under a multiplicative reward that jointly enforces bypass effectiveness and intent fidelity, while the defender is hardened on the harvested adversarial rewrites through a two-stage GRPO + rejection-sampled SFT pipeline balanced with benign data. Evaluated on BeaverTails and JailbreakBench against five held-out attack families (PAIR, TAP, AutoDAN, PAP, Translation), CHASE cuts mean StrongREJECT score by 43.2\% with 0\% false-refusal on benign prompts. Beyond the headline result, CHASE shows that template-free RL exploration recovers latent attack primitives that transfer across mechanistically distinct attack families, suggesting a path toward LLM safety hardening that generalises beyond the narrow distributions achieved thus far in adversarial training. 

---
# MASF: A Multi-Model Adaptive Selection Framework for Abstractive Text summarization 

**Authors**: Ahmed Alansary, Ali Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2606.05494)  

**Abstract**: Automatic text summarization has become increasingly important due to the rapid growth of digital textual information. This paper presents a Multi-Model Adaptive Summarization Framework designed to improve the robustness and quality of abstractive text summarization. Relying on a single model often leads to inconsistent summarization quality across articles with varying structures and topics. To address this limitation, the proposed framework integrates multiple fine-tuned transformer-based summarization models and introduces an adaptive selection mechanism. In this framework, each model independently generates a candidate summary for the same input article. The generated summaries are then evaluated using automatic evaluation metrics that capture both lexical similarity and semantic relevance. Based on these scores, the framework selects the highest-quality summary as the final output. The models are fine-tuned and evaluated on the widely used CNN/DailyMail news summarization dataset. Experimental results demonstrate that the proposed framework achieves the highest BERTScore among all compared methods with a score of 88.63%. It also outperforms several LLMs such as GPT3-D2, Falcon-7b, and Mpt-7b, highlighting its effectiveness and robustness. These findings highlight the effectiveness of leveraging multiple transformer-based models within an adaptive selection strategy to improve the quality and robustness of automatic text summarization systems. 

---
# Localizing Prompt Ambiguity in Large Language Models with Probe-Targeted Attribution 

**Authors**: Govind Ramesh, Yao Dou, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.05486)  

**Abstract**: Prompt ambiguity is a common source of failure in large language models, but is difficult to localize because it is a latent property of the prompt, while existing attribution methods are designed to explain observable outputs such as logits or generated tokens. We introduce PRIG, a gradient attribution method that uses a probe logit to attribute latent ambiguity to token positions. Specifically, PRIG trains a linear probe to distinguish clear prompts from ambiguous prompts and attributes the probe score to earlier token representations in the residual stream. To enable token-level evaluation, we construct synthetic ambiguity datasets across coding, math, and writing by rewriting one task-critical sentence per prompt, and complement them with a human-written gold benchmark. In this setting, PRIG localizes ambiguous spans substantially better than gradient attribution baselines, achieving 0.840 AUROC on the combined synthetic benchmark and 0.891 AUROC on the gold set. It also outperforms GPT-5.4 on sentence-level ambiguity identification and retains useful signal out-of-domain. These results establish PRIG as a practical tool for identifying which parts of a prompt are ambiguous. More broadly, they suggest that latent prompt properties can be localized through intermediate representations, rather than through output-level attribution. 

---
# Multilingual Coreference Resolution via Cycle-Consistent Machine Translation 

**Authors**: Adriana-Valentina Costache, Eduard Poesina, Silviu-Florin Gheorghe, Paul Irofti, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2606.05444)  

**Abstract**: Coreference resolution is a core NLP task, having a broad range of downstream applications, e.g.~machine translation, question answering, document summarization, etc. While the task is well-studied in English, comparatively less attention is dedicated to coreference resolution in other languages, especially low-resource ones. To mitigate this gap, we propose a novel coreference resolution pipeline that harnesses machine translation (MT) from English to a target low-resource language, to generate or expand training data. To automatically validate the quality of the translated samples, we back-translate the samples and assess the similarity with the original English samples via cosine similarity in the latent space of a BERT model. The resulting similarity scores are integrated into the loss function to weight training samples according to their MT cycle consistency. Extensive experiments on four low-resource languages show that our pipeline brings significant performance gains in coreference resolution. Moreover, our pipeline enables accurate coreference resolution in languages where no previous corpora were available. 

---
# ComplexityMT: Benchmarking the Interaction Between Text Complexity and Machine Translation 

**Authors**: Joseph Marvin Imperial, Junhong Liang, Belal Shoer, Abdullah Barayan, Rodrigo Wilkens, Omar Mussa, Dawn Knight, Eugénio Ribeiro, Ekaterina Kochmar, Sowmya Vajjala, Fernando Alva-Manchego, Harish Tayyar Madabushi  

**Link**: [PDF](https://arxiv.org/pdf/2606.05421)  

**Abstract**: When a text is translated, does the translation retain the complexity of the original? We introduce ComplexityMT, a new challenge for assessing how text complexity and machine translation interact with and influence each other, using the Common European Framework of Reference for Languages (CEFR) levels as the measure of text complexity. Across six languages, including Arabic, Dutch, English, French, Hindi, and Russian, we evaluate three open-weight models, one closed model, and a commercial machine translation system on two tasks: i) correlation of CEFR with translation difficulty, and ii) shifts in CEFR levels of the source texts. Our experiments show that higher CEFR levels make texts more difficult to translate, and that machine translation shifts the CEFR level of the target text compared to the original source, for most languages. These findings provide new insights for researchers and practitioners working on multilingual pedagogical content generation and machine translation difficulty estimation. 

---
# Executable Schema Contracts: From Automatic Ingestion to Multi-Source Retrieval 

**Authors**: Padmaja Jonnalagedda, Yuguang Yao, Xiang Gao, Hilaf Hasson, Kamalika Das  

**Link**: [PDF](https://arxiv.org/pdf/2606.05415)  

**Abstract**: Real-world data spans tables, documents, and semi-structured files with implicit semantics. Querying this data requires integrating evidence across inconsistent schemas and formats, yet existing approaches either demand costly manual engineering or bypass structure entirely. We present a system that automatically discovers an executable schema from raw multi-source data and uses it as a shared contract for knowledge graph construction and query-time retrieval. A closed-world field catalog constrains LLM-based schema discovery to attested fields; deterministic structural analysis infers identity keys, foreign keys, and source hierarchy; and the resulting schema drives extraction, deduplication, and cross-source linking into a provenance-aware knowledge graph. At query time the schema -- optionally extended via a monotonic protocol -- conditions a multi-tool agent routing retrieval across structured lookup, graph traversal, and vector search, returning grounded answers with traceable citations. In controlled zero-shot comparisons using the same LLM, data, and evaluation harness, the system improves over retrieval-only and decomposition-based baselines across four QA benchmarks, with ablations showing that schema-conditioned routing, structural intelligence, and schema-guided construction each contribute to the gains. 

---
# When Evidence is Sparse: Weakly Supervised Early Failure Alerting in Dialogs and LLM-Agent Trajectories 

**Authors**: Avinash Baidya, Xinran Liang, Ruocheng Guo, Xiang Gao, Kamalika Das  

**Link**: [PDF](https://arxiv.org/pdf/2606.05414)  

**Abstract**: Early failure alerting requires deciding, while a dialog or agent trajectory is still unfolding, whether to flag it as likely to fail. This is challenging because supervision is typically available only as a trajectory-level success/failure label while alerts must be raised from partial interactions. Prior early-classification methods often bridge this gap by assigning the terminal label to every prefix, treating every turn as failure evidence. We hypothesize that this prefix-label assumption is poorly matched to multi-turn language interactions, where evidence of eventual failure is sparse and often delayed. In this paper, we introduce a two-stage approach that learns from this sparse evidence structure and uses the resulting risk estimates for controllable early alerting. Specifically, our attention-based failure predictor learns sparse turn-level failure evidence from trajectory labels and uses it to estimate failure risk from partial histories. We then pair this predictor with $\alpha$-STOP, a single preference-conditioned stopping policy that selects an accuracy-earliness operating point at inference time rather than training a separate trigger for each preference. Across five benchmarks spanning customer support, task-oriented dialog, persuasion, tool use, and planning, we first show that high-relevance failure evidence occupies only 4.7-11.3% of turns and first appears after 59.0-83.6\% of trajectories on average. We further show that the attention-based predictor improves Pareto-frontier quality (hypervolume) by 1-10\% over naive prefix supervision, and that the full system improves frontier quality by 3-42\% over state-of-the-art trigger policies while reducing training cost per operating point by 1-3 orders of magnitude. 

---
# ReasoningFlow: Discourse Structures for Understanding LLM Reasoning Traces 

**Authors**: Jinu Lee, Shivam Agarwal, Amruta Parulekar, Siddarth Madala, Dilek Hakkani-Tur, Julia Hockenmaier  

**Link**: [PDF](https://arxiv.org/pdf/2606.05402)  

**Abstract**: Large reasoning models (LRMs) produce reasoning traces with non-linear structures, such as backtracking and self-correction, that complicate the evaluation and monitoring of the reasoning process. We introduce ReasoningFlow, a framework that captures the discourse structures of LRM reasoning traces into fine-grained directed acyclic graphs (DAGs). We develop and validate our annotation schema through careful manual annotation of 31 traces (2.1k steps), achieving high inter-annotator agreement, then scale to automatic annotation of 1,260 traces (247.7k steps) spanning three tasks (math, science, argumentation) and five models (Qwen2.5-32B-Inst, QwQ-32B, DeepSeek-V3, DeepSeek-R1, GPT-oss-120B). By analyzing ReasoningFlow graphs, we find: (1) LRMs exhibit structurally similar traces, despite being trained from different base models and potentially non-overlapping post-training data. (2) ReasoningFlow reveals diverse fine-grained reasoning behaviors (e.g., local verification, self-reflection, and assumptions) that can be used for better reasoning trace monitorability. (3) In LRMs, most of the erroneous steps are not used to derive final answers. (4) Mechanistic causal dependencies between steps do not reflect the language-level discourse structure. We release the dataset and code in: this https URL. 

---
# Trajectory Dynamics in Language Model Hidden States Predict Human Processing Costs Beyond Surprisal 

**Authors**: Elan Barenholtz  

**Link**: [PDF](https://arxiv.org/pdf/2606.05346)  

**Abstract**: Human language comprehension unfolds sequentially: each word is processed in the context of those that came before, and the interpretation builds incrementally over time. Surprisal, the negative log probability of a word given its context, has been the dominant predictor of incremental processing cost. But surprisal reduces rich sequential representations to a single scalar at each word, discarding information about the direction in which the interpretation has been evolving. Dynamical-systems approaches suggest that the trajectory of the evolving interpretive state, not just its position at each moment,should shape processing, and language itself may have local momentum, since speakers plan utterances a few words at a time. We introduce trajectory extrapolation error: at each word, we fit a linear trajectory to the preceding hidden states of a transformer language model and measure deviation from the extrapolated path. On the Natural Stories corpus, this measure is nearly orthogonal to surprisal (r = .044) and independently predicts self-paced reading times. The effect is especially pronounced in garden-path sentences, strengthens with model scale (GPT-2 Small to Large), and replicates across architectures with different positional encoding schemes (GPT-2 vs. Pythia/RoPE). A displacement control shows the effect is not reducible to representational change magnitude: displacement and extrapolation error predict in opposite directions. These findings reveal two dissociable components of processing cost: word-level prediction error (surprisal) and sensitivity to the local momentum of the unfolding interpretation (trajectory extrapolation error). 

---
# Self-supervised User Profile Generation for Personalization 

**Authors**: Clark Mingxuan Ju, Yuwei Qiu, Tong Zhao, Neil Shah  

**Link**: [PDF](https://arxiv.org/pdf/2606.05336)  

**Abstract**: Personalizing large language models (LLMs) has become a central challenge as LLMs are deployed across recommendation, search, dialogue, and content generation -- settings where the same query should yield different answers given different users. A promising route is to summarize each user's interaction history into a natural-language memory or profile and prepend it to the prompt to facilitate personalization. Existing methods learn such profile generators with explicit rewards derived from labeled downstream tasks, which are expensive and sparse as they require annotated supervision for every target task. In light of this challenge, we introduce Bidirectional User Modeling via Profiles (BUMP), a self-supervised framework that trains a profile generator without any downstream labels. Specifically, given a user's interaction history, we use GRPO to train an LLM to emit a free-form textual profile under a bidirectional in-batch ranking objective: a small LLM judge measures (i) how well the generated profile, used as a query, ranks the user's own held-out interactions above interactions from other users in the batch, and (ii) how well a held-out interaction, used as a query, ranks the user's own profile above profiles of other users. Both directions are scored with multi-positive NDCG and combined into a dense reward per rollout; other users in the batch supply free negatives, so every training example yields supervision from raw interaction logs alone. Evaluated on the LaMP benchmark, BUMP matches or outperforms closed-source APIs and prior methods relying on labeled rewards, while requiring no task label at training. 

---
# A Model of Multi-turn Human Persuadability Using Probabilistic Belief Tracing 

**Authors**: Jared Moore, Noah Goodman, Nick Haber, Max Kleiman-Weiner  

**Link**: [PDF](https://arxiv.org/pdf/2606.05330)  

**Abstract**: Large language models can shift human beliefs across high-stakes domains, but most persuasion studies rely on pre/post belief change. These endpoint measures identify whether persuasion occurred, yet miss where and how beliefs moved within a dialogue. We present PERSUASIONTRACE, a framework for studying persuasion in human-LLM interaction. Built on a web-based experimental platform, PERSUASIONTRACE contributes a tool for multi-turn persuasion studies and a process-level evaluation protocol: it records multi-turn belief reports from human or simulated targets of persuasion, annotates persuader turns with rhetorical dimensions (logos/pathos/ethos), and evaluates simulators by fidelity to real human belief dynamics. Using this framework, we find that human targets group into two clusters of multi-turn belief updates and exhibit susceptibility to rhetorical strategies, and that LLMs are persuasive across generic and personalized topics, text and audio modalities, and multi-turn interactions. Prior work has chiefly used vanilla-prompted LLMs to simulate human targets, but we show that these simulators fail to replicate human belief dynamics. We introduce a Bayesian-network simulated target that maintains an explicit latent belief state over time so each persuader message yields cognitively realistic belief updates. In human-likeness evaluation, our Bayesian target scores near a human reference (81 vs 80), while baseline LLM targets score substantially lower (64). PERSUASIONTRACE reframes persuasion evaluation from endpoint movement alone to process fidelity, providing a stronger basis for scientific analysis and safer optimization of persuasive systems. 

---
# LoRi: Low-Rank Distillation for Implicit Reasoning 

**Authors**: Ryan Solgi, Jiayi Tian, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05315)  

**Abstract**: Implicit chain-of-thought (iCoT) methods aim to internalize reasoning in large language models, but often underperform explicit CoT prompting. We empirically find that hidden-state reasoning trajectories exhibit low-rank structure. Motivated by this observation, we propose a low-rank distillation framework that transfers reasoning by aligning teacher and student trajectories in a shared low-rank tensor subspace using first- and second-order statistics. The resulting formulation captures the global structure of reasoning while supporting a compact latent reasoning process. We evaluate the method across multiple model families, including LLaMA and Qwen, at different scales on mathematical reasoning benchmarks. Our approach consistently improves performance, especially on challenging multi-step tasks, approaching explicit CoT accuracy and outperforming prior iCoT distillation methods. 

---
# The Granularity Gap: A Multi-Dimensional Longitudinal Audit of Sycophancy in Gemini Models 

**Authors**: Patrick Keough  

**Link**: [PDF](https://arxiv.org/pdf/2606.05183)  

**Abstract**: Large language models are increasingly deployed as high-stakes advisors, yet standard alignment benchmarks treat sycophancy as a binary failure mode. We introduce the Granularity Gap: coarse binary metrics mask substantial social-compliance behaviors where models capitulate to user framing, validate questionable premises, or soften factual corrections without producing overtly false outputs. We evaluate six Gemini variants across generations 2.0, 2.5, and 3.0 on 73 adversarial prompts under three guardrail conditions (Control, Simple, Protocol), yielding 8,830 graded responses. Using a 0-4 Likert scale validated against a human annotator triad (Fleiss kappa = 0.71; Cohen kappa = 0.78 vs AI consensus; 95.9 percent binary accuracy, 100 percent specificity), we quantify sycophancy as continuous rather than binary. Three findings emerge. First, 27.2 percent of responses contain substantial sycophantic content (Likert >= 2.0) and 22.7 percent reach moderate or severe levels (>= 3.0), while binary win-rate framing reports only modest failure rates; coarse metrics explain just 29 percent of graded variance. Second, generational progress is non-monotonic: Gen 2.5 regresses sharply (mean Control 2.64) relative to Gen 2.0 (1.90) and Gen 3.0 (2.01), and Gen 2.5 shows inverse scaling (Pro 1.94 worse than Flash 1.71) while Gen 3.0 restores standard scaling. Third, we document an Alignment Tax: Spearman rho = -0.63 between sycophancy and truthfulness, indicating social compliance trades against factual accuracy. Egotistical Validation prompts act as a sycophancy trap (mean 3.27), nearly double Unethical Proposals (1.72). Simple guardrails outperform elaborate Protocol scaffolding on flagship models, but distilled Gen 3.0 Flash inverts this, suggesting small models may structurally require chain-of-thought scaffolding. We release the dataset and rubric to support continuous sycophancy measurement. 

---
# LANTERN: Layered Archival and Temporal Episodic Retrieval Network for Long-Context LLM Conversations 

**Authors**: Rahul Subramani  

**Link**: [PDF](https://arxiv.org/pdf/2606.05182)  

**Abstract**: Large language models discard critical details when conversation history is compacted to fit within finite context windows. We present LANTERN (Layered Archival aNd Temporal Episodic Retrieval Network), a lightweight memory layer that proactively archives every conversation turn and restores relevant details after compaction via hybrid retrieval -- requiring zero LLM calls and adding fewer than 25ms of latency per turn. On 94 real multi-turn conversations (1,894 ground-truth facts, human-validated at kappa=0.81), LANTERN-Rerank recovers 78.3% of verifiable facts lost to compaction, significantly outperforming a faithful reimplementation of MemGPT's LLM-driven extraction and multi-query search pipeline (72.4%; Wilcoxon p<0.0001, 95% CI [+3.1, +8.6] pp, d=0.43) at a fraction of the inference cost. Even without the reranker, base LANTERN matches or exceeds this LLM-driven baseline (p=0.005) using zero LLM calls. When four production LLMs answer fact-bearing questions using LANTERN-restored context, accuracy improves by 8.4 percentage points on average (Wilcoxon p<0.05 for each model individually), demonstrating that the recovered context is useful across diverse model architectures. We release the full evaluation framework -- paired significance tests, failure analysis, fact-type stratification, and compaction robustness analysis -- to support reproducibility and future work. 

---
# Multi-Granularity Reasoning for Natural Language Inference 

**Authors**: Chunling Xi, Di Liang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05181)  

**Abstract**: Natural Language Inference (NLI) is a fundamental task in natural language understanding that requires determining the logical relationship between a premise and a hypothesis. Despite the remarkable success of transformer-based pre-trained models, most existing approaches primarily rely on the final-layer token representations, which are often insufficient for capturing the complex and hierarchical semantic interactions required for effective reasoning. In particular, fine-grained lexical cues, phrasal compositions, and higher-level contextual semantics are typically entangled or diluted in a single representation space. To address these limitations, we propose a novel \emph{Multi-Granularity Reasoning Network} (MGRN) that explicitly leverages hierarchical semantic features within an interactive reasoning space. The proposed framework mimics the human cognitive process of language understanding, which naturally progresses from shallow lexical matching to deeper semantic abstraction and logical reasoning. By integrating semantic information across multiple granularities in a progressive and structured manner, MGRN is able to uncover intricate semantic relationships underlying natural language expressions. Extensive experiments on multiple public benchmarks demonstrate that MGRN consistently outperforms strong baseline models, validating the effectiveness and robustness of the proposed approach. 

---
# From Scoring to Explanations: Evaluating SHAP and LLM Rationales for Rubric-based Teaching Quality Assessment 

**Authors**: Ivo Bueno, Babette Bühler, Philipp Stark, Tim Fütterer, Ulrich Trautwein, Dorottya Demszky, Heather Hill, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2606.05180)  

**Abstract**: Automated scoring models are increasingly used to assign rubric-based quality ratings to complex language performances, including classroom transcripts, yet they typically provide little insight into why a particular score is produced. We propose a general framework for sentence-level interpretability of rubric-based scoring that combines model-agnostic Shapley-value attributions with rationales generated by large language models (LLMs). Instantiated on the Quality of Feedback dimension of the CLASS framework using the NCTE corpus, the framework enables systematic comparison of fine-tuned pretrained language models (PLMs) and prompted LLMs on both scoring performance and explanation faithfulness. Across 6k annotated transcript segments, fine-tuned PLMs outperform LLMs in prediction accuracy but exhibit label compression toward mid-scale scores. Deletion-based tests show that SHAP identifies sentences that reliably drive model predictions, producing typically larger and more coherent prediction shifts than LLM-generated rationales. Cross-model analyses further reveal that SHAP attributions transfer robustly across architectures, whereas LLM rationales exert limited and inconsistent influence. Overall, the findings demonstrate that SHAP provides more faithful and transferable explanations for rubric-based scoring, and that the proposed framework offers a principled basis for evaluating both scoring models and their explanations in high-stakes educational settings and other rubric-based language assessment tasks. 

---
# Efficient Punctuation Restoration via Weighted Lookahead Scoring Method for Streaming ASR Systems 

**Authors**: Sungmook Woo, Hyungu Kang, Chanwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.05179)  

**Abstract**: Punctuation restoration improves ASR (Automatic Speech Recognition) readability. However streaming ASR requires online decisions with limited future context. In streaming ASR, the system predicts punctuation incrementally, which makes generation-based approaches prone to latency and alignment failures under boundary-wise evaluation. This paper proposes a non-autoregressive scoring method (no free-form generation) that preserves the input transcript and makes a decision at each word boundary. Our method compares punctuation insertion hypotheses against a no-insertion baseline under a bounded K-subword-token lookahead, and calibrates decisions using a weight {\alpha} and a validation-calibrated threshold {\tau} (no parameter updates during inference). On IWSLT 2017, our scoring method achieves a 4-class macro F1 of 0.893 in the no fine-tuning setting (validation-calibrated, K=2) and 0.937 after fine-tuning (K=2), outperforming the prompt-based baseline (0.566) and a fine-tuned ELECTRA baseline (0.913) under the same lookahead budget. We analyze the impact of the lookahead budget through ablation studies on K. 

---
# MCBench: A Multicontext Safety Assessment Benchmark for Omni Large Language Models 

**Authors**: Manh Luong, Tamas Abraham, Junae Kim, Amar Kaur, Rollin Omari, Gholamreza Haffari, Trang Vu, Lizhen Qu, Dinh Phung  

**Link**: [PDF](https://arxiv.org/pdf/2606.05177)  

**Abstract**: Existing multimodal safety benchmarks focus solely on visual inputs and cannot assess Omni Large Language Models (LLMs) that process vision, audio, and text. We introduce MCBench, a benchmark with 1196 scenarios spanning four safety categories that require integrating multiple modalities for accurate safety assessment. Each unsafe scenario is paired with a minimally different safe counterpart to assess model sensitivity. Our evaluations of state-of-the-art models reveal significant challenges. Omni LLMs struggle with subtle or non-physical risks but perform better when salient visual or acoustic cues are present. Analysis of reasoning traces shows that, although models can extract modality-specific information, they often fail to integrate these cues effectively for safety judgments. Our findings reveal that current Omni LLMs lack robust cross-modal reasoning in safety-critical settings, underscoring the need for improved architectures and training strategies for multimodal safety. 

---
# PEFT of SLM for Telecommunications Customer Support: A Comparative Study of LoRA Configurations with Energy Consumption Analysis 

**Authors**: Lucas Tamic, Ilan Jaffeux-Cheniout, Xavier Marjou  

**Link**: [PDF](https://arxiv.org/pdf/2606.05176)  

**Abstract**: While large language models (LLMs) show strong performance in natural language understanding and generation, their evaluation and adaptation to domain-specific constraints in telecommunications customer support remain limited. In addition, data sovereignty, regulatory constraints, and the handling of sensitive customer and network information complicate the use of externally hosted foundation models in this domain.
We present a systematic study of parameter-efficient fine-tuning (PEFT) using Low-Rank Adaptation (LoRA) applied to Qwen2.5-3B to build a domain-specific conversational assistant. We introduce a combinatorial synthetic data generation approach based on a glossary of 52 industry-specific terms, producing approximately 30,000 training examples across 1,560 distinct problem scenarios via a generative pipeline powered by Gemini 2.0 Flash.
We evaluate 16 LoRA configurations by varying hyperparameters and target modules. Our evaluation extends beyond standard metrics by incorporating energy consumption analysis and qualitative assessment using an LLM-as-a-judge framework with GPT-5.2 and Claude 4.5 Sonnet.
Results show a clear divergence between quantitative and qualitative performance: models achieving the lowest validation loss do not necessarily obtain the best human-aligned rankings. The best validation loss (0.5024) ranks only 6th-7th in qualitative evaluation, while the worst loss (0.6807) ranks first according to both judges.
This work contributes (1) a combinatorial method for synthetic dataset construction, (2) insights into the impact of target module selection for LoRA injection, (3) evidence that validation loss alone is insufficient for selecting fine-tuning configurations in conversational AI, and (4) an energy-performance trade-off analysis for sustainable LLM deployment. 

---
# Generic Triple-Latent Compression with Gated Associative Retrieval 

**Authors**: Liu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2606.05175)  

**Abstract**: We study generic triple-latent sequence models that maintain a running token state and compressed pair-memory pathway to capture higher-order token interactions without benchmark-specific parsing. The triple-latent family improves a small Transformer baseline on byte-level WikiText-2 and on a tokenizer-based MiniMind language-model benchmark, while a recall-focused gated key-value retrieval extension improves associative recall but remains seed-sensitive and much slower in the current reference implementation. 

---
# Improving Heart-Focused Medical Question Answering in LLMs via Variance-Aware Rubric Rewards with GRPO 

**Authors**: Arash Ahmadi, Parisa Masnadi, Sarah Sharif, Charles Nicholson, David Ebert, Mike Banad  

**Link**: [PDF](https://arxiv.org/pdf/2606.05174)  

**Abstract**: Large Language Models (LLMs) have shown strong promise in healthcare applications. Yet deploying general-purpose models in real-world settings remains difficult due to data privacy constraints, inference costs, and limited suitability for edge or on-device use. These challenges motivate the development of smaller, more efficient models that require robust post-training strategies to ensure reliable medical reasoning. In this work, we investigate Group Relative Policy Optimization (GRPO) for post-training LLMs on heart-focused medical question answering with rubric-based supervision derived from RaR-Medicine. We propose a Variance-Aware Reward Framework that extends the Explicit Aggregation and Implicit Aggregation strategies of Rubrics as Rewards by replacing weighted binary criterion aggregation and single overall Likert-style scoring with continuous analytical reward functions derived from criterion-level rubric outcomes. This formulation provides richer optimization signals for feedback that is sparse, multi-criteria, and difficult to verify automatically, and enables more stable on-policy reinforcement learning. On a held-out heart-related subset of HealthBench, our best GRPO variant improves accuracy from 0.362 to 0.502 and F1 from 0.532 to 0.668 relative to the Qwen3-14B base model, while remaining competitive with GPT-OSS-120B (0.508 accuracy, 0.674 F1). Our findings show that carefully designed rubric-based rewards provide a practical strategy for improving heart-focused medical question answering in LLMs, with potential to extend to other rubric-based tasks. 

---
# Predict and Reconstruct: Joint Objectives for Self-Supervised Language Representation Learning 

**Authors**: Aimen Boukhari  

**Link**: [PDF](https://arxiv.org/pdf/2606.05173)  

**Abstract**: Masked language modelling (MLM) has been the dominant pre-training objective for text encoders since BERT, yet it encourages representations that are strongly anchored to surface-form token identity rather than deeper semantic structure. Inspired by the success of Joint Embedding Predictive Architectures (JEPA) (LeCun, 2022) in vision and audio, we propose a hybrid pre-training objective that combines a JEPA-style latent-space prediction loss with a standard MLM objective over a single shared encoder. A learnable scalar parameter continuously balances the two objectives during training. We pre-train both a hybrid model and a pure-MLM baseline on English Wikipedia using identical architectures and compute budgets (NVIDIA H100). Extensive representation analysis across five GLUE benchmarks (SST-2, MRPC, MNLI, CoLA, STS-B) using four pooling strategies reveals that the hybrid encoder produces significantly more uniform embeddings (uniformity less than -0.16 vs -0.05 for MLM), exhibits richer spectral geometry under max pooling, encodes less surface-level lexical information, and achieves a better semantic-to-lexical balance. Despite similar linear-probe downstream accuracy, the geometric differences are consistent and significant, suggesting that the JEPA predictive objective reshapes the latent space in ways that standard accuracy metrics alone cannot capture. 

---
# Epidemiology of Model Collapse: Modeling Synthetic Data Contamination via Bilayer SIR Dynamics 

**Authors**: Xiangyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05168)  

**Abstract**: Training on synthetic data causes model collapse, but existing analyses treat this as single-chain degradation. In reality, the AI ecosystem involves cross-contamination: models ingest synthetic data from other models, produce new synthetic text, and contaminate shared corpora. We propose a bilayer coupled SIR/SIRS framework -- a phenomenological mean-field model treating data corpora and AI models as two interacting populations, each with susceptible, infected, and recovered compartments linked by cross-layer transmission. The SIRS variant (our primary recommendation) incorporates immunity waning, reflecting that filtered corpora and retrained models remain susceptible to re-contamination. We derive the basic reproduction number $R_0 = \sqrt{\beta_D \beta_M / [(\gamma_D+\mu_D)(\gamma_M+\mu_M)]}$ via the Next Generation Matrix and apply standard epidemic threshold results to the bilayer system. Illustrative scenario-based calibration from public AI text prevalence data yields supercritical dynamics ($R_0 > 1$) across three scenarios; Sobol sensitivity analysis identifies synthetic-text detection as the highest-leverage parameter. A bipartite-network agent-based model confirms mean-field consistency ($R^2 > 0.96$) for dense networks but degrades under heterogeneity. GPT-2 contamination chain experiments (192 runs across WikiText and Shakespeare) show dose-response degradation and diversity loss qualitatively consistent with the threshold picture. Matched-budget source-diversity experiments (1,088 runs) provide suggestive evidence that multi-source mixing modestly attenuates collapse, but the effect vanishes at lower contamination fractions. Intervention analysis identifies detection-based filtering and herd immunity as the highest-leverage strategies. 

---
# Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under Software Evolution 

**Authors**: Liliana Hotsko, Yinxi Li, Yuntian Deng, Pengyu Nie  

**Link**: [PDF](https://arxiv.org/pdf/2606.06492)  

**Abstract**: Code language models need repository-level context to resolve imports, APIs, and project conventions. Existing methods inject this knowledge as long inputs (retrieved through RAG or dependency analysis) or through per-repository fine-tuning and LoRA -- costly at repository scale and brittle to evolving codebases. We introduce Code2LoRA, a hypernetwork framework that generates repository-specific LoRA adapters, effectively injecting repository knowledge with zero inference-time token overhead. Code2LoRA supports two usage scenarios: Code2LoRA-Static converts a single repository snapshot into an adapter, suitable for comprehension of stable codebases; while Code2LoRA-Evo maintains an adapter backed by a GRU hidden state updated per code diff, suitable for active development of evolving codebases. To evaluate Code2LoRA against parameter-efficient fine-tuning baselines, we build RepoPeftBench, a benchmark of 604 Python repositories with two tracks: a static track with 40K training and 12K test assertion-completion tasks, and an evolution track with 215K commit-derived training and 87K commit-derived test tasks. On the static track, Code2LoRA-Static achieves 63.8% cross-repo and 66.2% in-repo exact match, matching the per-repository LoRA upper bound; on the evolution track, Code2LoRA-Evo achieves 60.3% cross-repo exact match (+5.2 pp over a single shared LoRA). Code2LoRA's code can be found at this https URL the model checkpoints and RepoPeftBench datasets can be found at this https URL. 

---
# MLEvolve: A Self-Evolving Framework for Automated Machine Learning Algorithm Discovery 

**Authors**: Shangheng Du, Xiangchao Yan, Jinxin Shi, Zongsheng Cao, Shiyang Feng, Zichen Liang, Boyuan Sun, Tianshuo Peng, Yifan Zhou, Xin Li, Jie Zhou, Liang He, Bo Zhang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2606.06473)  

**Abstract**: Large language model (LLM) agents are increasingly applied to long-horizon tasks such as scientific discovery and machine learning engineering (MLE), where sustained self-evolution becomes a key capability. However, existing MLE agents suffer from inter-branch information isolation, memoryless search, and lack of hierarchical control, which together hinder long-horizon optimization. We present MLEvolve, an LLM-based self-evolving multi-agent framework for end-to-end machine learning algorithm discovery. By extending tree search to Progressive MCGS, MLEvolve enables cross-branch information flow through graph-based reference edges and gradually shifts the search from broad exploration to focused exploitation with an entropy-inspired progressive schedule. To allow the agent to evolve with accumulated experience, we introduce Retrospective Memory, which combines a cold-start domain knowledge base with a dynamic global memory for task-specific experience retrieval and reuse. For stable long-horizon iteration, we further decouple strategic planning from code generation with adaptive coding modes. Evaluation on MLE-Bench shows that MLEvolve achieves state-of-the-art performance across multiple dimensions including average medal rate and valid submission rate under a 12-hour budget (half the standard runtime). Moreover, MLEvolve also outperforms specialized algorithm discovery methods including AlphaEvolve on mathematical algorithm optimization tasks, demonstrating strong cross-domain generalization. Our code is available at this https URL. 

---
# Scaffold, Not Vocabulary? A Controlled, Two-Tier, Pre-Registered Study of a Popperian Code-Generation Skill 

**Authors**: Mehmet Iscan  

**Link**: [PDF](https://arxiv.org/pdf/2606.06454)  

**Abstract**: Large language models increasingly write, review, and judge code, and a fast-growing practice equips them with prompt 'skills' that ask the model to reason like a scientist. A prominent example tells the model to act as a Popperian falsificationist, and such skills are reported to improve generated code. But these gains are almost always read off an LLM-as-a-judge, an instrument with documented positional, self-preference, and stylistic biases. We ask: if it appears to help, is the gain from the skill's Popperian content, or from the structure any scaffold imposes? We pre-register a two-tier ablation with three controls: a length-matched placebo, a labels-only scaffold that keeps the Popperian headers but strips the procedure, and an execution oracle (HumanEval+ unit tests), plus a vocabulary-halo sentinel and a same-model self-judge audit. On a frontier model (Claude Sonnet 4.6, N=163) all conditions sit near the benchmark ceiling and do not separate, so the pre-registered +5-point improvement is not supported (a ceiling-limited non-detection). On a small model (Qwen2.5-Coder-0.5B, N=164) structured arms lift best-of-eight correctness by 20-22 points, but the full skill shows no separable benefit over a labels-only scaffold (aggregate F@8=L@8 vs V@8=34.8%), and the placebo trails by only 2.4 points. A 0.5B self-judge applying the Popperian rubric does not beat random selection and concentrates 60% of its picks on one index. In the two settings tested, the skill's Popperian procedural content adds no separable execution-correctness benefit beyond a labels-only scaffold, so the gains track scaffold structure. We contribute a calibrated negative result and a reusable disambiguation protocol; the finding bounds an engineering claim about one prompt-skill family and is not an evaluation of Popperian methodology in general. 

---
# USAD 2.0: Scaling Representation Distillation for Universal Audio Understanding 

**Authors**: Heng-Jui Chang, Alexander H. Liu, Saurabhchand Bhati, Mrudula Athi, Anton Ratnarajah, Amit Chhetri, James Glass  

**Link**: [PDF](https://arxiv.org/pdf/2606.06444)  

**Abstract**: Audio encoders are critical to modern audio applications as large language models (LLMs) increasingly rely on a single encoder for diverse inputs. While self-supervised learning (SSL) has yielded strong domain-specific encoders like speech or music experts, multi-domain approaches like USAD and SPEAR remain limited in coverage and evaluation. Recent studies also suggest supervised encoders align better with audio LLMs. We present USAD 2.0, a universal encoder integrating knowledge from both SSL and supervised foundation models. USAD 2.0 introduces domain-aware distillation to address teacher mismatch, extends coverage to the music domain, and adds second-stage supervised distillation for downstream use. We further scale the model to one billion parameters via depth scaling. Experiments show USAD 2.0 achieves strong or state-of-the-art performance across probing and LLM-based evaluations. 

---
# Unsupervised Skill Discovery for Agentic Data Analysis 

**Authors**: Zhisong Qiu, Kangqi Song, Shengwei Tang, Shuofei Qiao, Lei Liang, Huajun Chen, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2606.06416)  

**Abstract**: Inference-time skill augmentation provides a lightweight way to improve data-analytic agents by injecting reusable procedural knowledge without updating model parameters. However, discovering effective skills for data analysis remains challenging, as reliable supervision is expensive and success criteria vary across analytical formats. This raises the key question of how to discover reusable data-analysis skills from unlabeled exploration alone. We propose DataCOPE, an unsupervised verifier-guided skill discovery framework for data-analytic agents. DataCOPE derives verifier signals from the exploration trajectories and uses them to characterize relative quality or aggreement among trajectories. It iteratively coordinates a Data-Analytic Agent for trajectory generation, an Unsupervised Verifier for signal extraction, and a Skill Manager for contrastive skill distillation. For report-style analysis, we instantiate the verifier as an Adaptive Checklist Verifier that derives task-specific criteria, scores reports by verifiable coverage, and iteratively refines the checklist. For reasoning-style analysis, we instantiate it as an Answer Agreement Verifier that groups trajectories by answer agreement and uses self-consistency as an auxiliary signal. We evaluate DataCOPE on report-style analysis from Deep Data Research and reasoning-style analysis from DABStep. Across both settings, DataCOPE consistently improves held-out performance over baselines. Averaged across four model settings, DataCOPE improves the mean score by 9.71% and 32.30% on report-style and reasoning-style tasks respectively. 

---
# Humans' ALMANAC: A Human Collaboration Dataset of Action-Level Mental Model Annotations for Agent Collaboration 

**Authors**: Jiaju Chen, Yuxuan Lu, Jiayi Su, Chaoran Chen, Songlin Xiao, Zheng Zhang, Yun Wang, Yunyao Li, Jian Zhao, Tongshuang Wu, Toby Jia-Jun Li, Dakuo Wang, Bingsheng Yao  

**Link**: [PDF](https://arxiv.org/pdf/2606.06388)  

**Abstract**: Recent advances in LLM agents have enabled complex cognitive capabilities, such as multi-step reasoning, planning, and tool use, that increasingly position these agents as human collaborators. Effective collaboration, however, requires collaborators to continuously maintain and align mental models of their own reasoning,partners' intentions, and shared goals during the collaborative process. Today's agents rarely develop such capabilities since they are primarily optimized for task completion, and the community lacks authentic human collaboration data with action-level mental model annotations that could guide agents toward process-level collaborative competence. To bridge this gap, we present ALMANAC, a dataset of Action-Level Mental model ANnotations for Agent Collaboration built from the Map Task, a classic dyadic routing task from social science. ALMANAC contains 2,987 collaboration actions, each paired with theory-informed mental model annotations that record the participants' self-reasoning, perceived partner intent, and perceived team goal. We benchmark six LLMs on predicting humans' next-turn behavior and mental models. Our results demonstrate ALMANAC's utility in evaluating models' ability to simulate human collaborative behaviors and infer their underlying mental models. 

---
# Learning What to Forget: Improving LLM Unlearning via Learned Token-Level Importance 

**Authors**: Gizem Yüce, Giorgos Nikolaou, Nicolas Flammarion  

**Link**: [PDF](https://arxiv.org/pdf/2606.06320)  

**Abstract**: Machine unlearning aims to remove targeted knowledge from a trained model while preserving its general capabilities. For autoregressive language models, not all tokens in a forget sample are equally relevant to forgetting. Existing approaches either ignore this heterogeneity or rely on auxiliary models, heuristics, or external annotations to estimate each token's relevance for forgetting. We instead characterize it through the interaction with the retain objective: a token is forget-specific to the extent that minimizing the forget loss on that token does not conflict with retain optimality. We formalize this perspective as a joint optimization problem over the model parameters and the token weights and show that, under a natural separation condition, the resulting objective recovers the oracle forget-specific token support. Motivated by this formulation, we introduce Alternating Token-Weighted Unlearning (ATWU), a lightweight framework that jointly learns token forget-specificity and model parameters during unlearning using a simple linear scorer over the hidden states, without external token level supervision. Across TOFU and RWKU, ATWU achieves state of the art forget-retain trade-offs, outperforming sample-level methods, probability-based token weighting heuristics, and auxiliary-model-based approaches. Moreover, the learned scores align substantially better with ground truth forget-specific spans, indicating that ATWU identifies semantically meaningful token level forgetting signals. Overall, our results suggest that retain conflict provides an effective criterion for identifying what language models should forget, enabling unsupervised learning of token level forget-specificity directly from model representations with minimal computational overhead. 

---
# OneReason Technical Report 

**Authors**: OneRec Team, Biao Yang, Boyang Ding, Chenglong Chu, Dunju Zang, Fei Pan, Han Li, Hao Jiang, Honghui Bao, Huanjie Wang, Jian Liang, Jiangxia Cao, Jiao Ou, Jiaxin Deng, Jinghao Zhang, Kun Gai, Lu Ren, Peiru Du, Pengfei Zheng, Rongzhou Zhang, Ruiming Tang, Shiyao Wang, Siyang Mao, Siyuan Lou, Teng Shi, Wei Yuan, Wenlong Xu, Xingchen Liu, Xingmei Wang, Xinqi Jin, Yan Sun, Yan Wang, Yifei Hu, Yingzhi He, Yufei Ye, Yuhao Wang, Yunhao Zhou, Yuqin Dai, Zhao Liu, Zhipeng Wei, Zhixin Ling, Ziming Li, Zixing Zhang, Ziyuan Liu, An Zhang, Changxin Lao, Chaoyi Ma, Chengru Song, Defu Lian, Fan Yang, Guowang Zhang, Hao Peng, Jiayao Shen, Jie Chen, Jun Xu, Junmin Chen, Kun Zhang, Kuo Cai, Mingxing Wen, Minmao Wang, Minxuan Lv, Qi Zhang, Qiang Luo, Sheng Yu, Shijie Li, Shijie Yi, Shuang Yang, Shugui Liu, Shuni Chen, Tinghai Zhang, Tingting Gao, Xiang Wang, Xiangyu Wu, Xiangyu Zhao, Xiao Lv, Xiaoyou Zhou, Xuming Wang, Yong Du, Zejian Zhang, Zhaojie Liu, Zhiyang Zhang, Zhuang Zhuang, Ziqi Wang, Ziyi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.06260)  

**Abstract**: Generative recommendation models in the OneRec family have been widely deployed in many real-world services, such as short-video, live-streaming, advertising, and e-commerce. However, these generative models can only benefit from the scaling advantage, while their reasoning ability is hard to activate, since we cannot construct meaningful Chain-of-Thought (CoT) sequences consisting of itemic tokens only. Inspired by the success of the reasoning-style ``think before answer'' paradigm in the LLM field, we conduct preliminary studies (i.e., OneRec-Think, OpenOneRec) to explore reasoning capability in generative recommendation. Nevertheless, we notice an unexpected phenomenon: the thinking mode does not show advantages over the non-thinking mode. Drawing insights from recent findings on CoT robustness in multi-modal language models, we argue that effective reasoning in recommendation rests on two factors: perception, the ability to ground itemic tokens in their underlying language semantics, and cognition, the ability to reorganize a user's behavior sequence into coherent latent interest points. We therefore propose OneReason, which includes: (1) strong itemic token perception in pre-training, (2) a three-level cognition-enhanced CoT format for recommendation tasks in SFT, and (3) a specialize-then-unify training recipe in RL to enhance the thinking ability. 

---
# Revisiting Lexicon Evaluation in Unsupervised Word Discovery 

**Authors**: Simon Malan, Danel Slabbert, Herman Kamper  

**Link**: [PDF](https://arxiv.org/pdf/2606.06183)  

**Abstract**: Building a lexicon from discovered word-like units is a central goal in zero-resource speech processing. But do our evaluations provide a trustworthy indication of lexicon quality? A common metric, normalized edit distance, averages the phoneme edit distances between discovered units in each cluster. We show that this metric has an inherent bias toward the quality of large clusters, inhibiting fair evaluation. Moreover, it ignores how well true classes are distributed across clusters. Based on established theory in clustering literature, we propose two metrics that address these shortcomings: a modified metric that weighs cluster size when assessing within-cluster consistency, and an inverse metric that assesses how true words are spread across clusters. Through experiments on synthetic and real-world lexicons, we demonstrate that combined, these metrics are: (1) more closely correlated with how similar a lexicon is to the ground-truth distribution, and (2) more robust to biases that skew lexicon evaluations. 

---
# Learning to Route LLMs from Implicit Cost-Performance Preferences via Meta-Learning 

**Authors**: Jiahao Zeng, Ming Tang, Ningning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2606.06178)  

**Abstract**: Large language models (LLMs) present a trade-off between performance and cost, where more powerful models incur greater expense. LLM routing aims to mitigate expenses while maintaining performance by sending queries to the most suitable model. However, existing methods cannot perform well for different user cost-performance preferences. To address this gap, we introduce a novel perceptive LLM routing paradigm for personalized and user-centric cost-performance optimization, which efficiently learns users' implicit preferences through little interaction. To handle the challenge of heterogeneous user needs, we formulate preference profiles as a set of distinct tasks in contextual bandit and propose MetaRouter, a meta-learning framework designed for preference-aware LLM routing. Experimental results show that MetaRouter outperforms strong baselines on both in-distribution and out-of-distribution tasks. Furthermore, it exhibits high efficiency in learning user preferences, robustness to changes in the routable LLMs, and scalability to multi-model routing. 

---
# ProSarc: Prosody-Aware Sarcasm Recognition Framework via Temporal Prosodic Incongruity 

**Authors**: Prathamjyot Singh, Ashima Sood, Sahil Sharma, Jasmeet Singh  

**Link**: [PDF](https://arxiv.org/pdf/2606.06168)  

**Abstract**: We present ProSarc, an audio-only framework that detects sarcasm by modelling temporal prosodic incongruity, that is, the mismatch between local prosodic dynamics and the utterance-level emotional baseline. Dual encoding paths, a Global Emotion Encoder and a Temporal Prosody Encoder (BiLSTM + multi-head attention), feed a Prosodic Incongruity Analyzer that produces a scalar incongruity score for classification. Monte Carlo dropout provides uncertainty estimates, and an attention-based mechanism localises sarcastic onset without frame-level labels. ProSarc outperforms prior audio-only methods on MUStARD++ (F1=75.3) and generalises to spontaneous (PodSarc, F1=62.9) and cross-lingual speech (MuSaG, F1=65.6). Ten-run validation confirms the contribution of incongruity modelling (Wilcoxon p=0.002, Cohen's d=1.51). Human evaluation shows that model uncertainty tracks perceptual ambiguity and predicted onsets align with human-annotated temporal windows. 

---
# Where does Absolute Position come from in decoder-only Transformers? 

**Authors**: Valeria Ruscio, Umberto Nanni, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2606.06160)  

**Abstract**: RoPE-trained transformers distinguish absolute position in their attention patterns, even though RoPE encodes only relative offsets in the inner product. We trace this leakage to two architectural components, The causal mask is responsible for the first: its per-query softmax denominator depends on the absolute query position by construction. The residual stream supplies the second. Under causal attention the activation at position $0$ attends only to itself and runs as a closed dynamical system from the embedding of the token at that position; downstream attention reads this trajectory through sink-reading heads. Both components appear in all three architectures we study, in architecturally specific balance: NTK scaling suppresses the residual-stream component, sliding-window attention allows it to accumulate with depth, and standard RoPE sits between. Replacing the \texttt{BOS} embedding before the forward pass removes $40\%$ of the residual-stream component at early queries. Attention sinks are token-anchored stabilizers that pass forward a deterministic fingerprint of the token at position $0$, constant across inputs when that token is the auto-prepended \texttt{BOS} and varying with it otherwise. 

---
# OrderGrad: Optimizing Beyond the Mean with Order-Statistic Policy Gradient Estimation 

**Authors**: Paavo Parmas, Yongmin Kim, Kohsei Matsutani, Shota Takashiro, Soichiro Nishimori, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2606.06096)  

**Abstract**: Policy-gradient methods usually optimize expected return, but many real world applications care about distributional properties of returns: tail risk, outlier robustness, or best-of-K discovery. We introduce OrderGrad, a family of likelihood-ratio and reparameterization gradient estimators for order-statistic objectives. OrderGrad optimizes finite-sample L-statistics, i.e., weighted averages of sorted rewards or costs, recovering objectives such as VaR, CVaR, trimmed means, medians, and top-m/best-of-K criteria by changing only the rank weights. For any fixed sample size and rank-weight vector, OrderGrad provides an unbiased gradient estimator for the corresponding order-statistic objective. The method is implemented as a simple reward transformation that can then be used in an otherwise standard policy-gradient or reparameterized update. We study the resulting estimator's variance behavior and evaluate it on tasks where mean optimization is mismatched to the deployment objective, including LLM math post-training and other tasks. OrderGrad provides a unified, plug-and-play route to risk-averse, robust, and exploratory learning.
Code: this https URL 

---
# On Advantage Estimates for Max@K Policy Gradients 

**Authors**: Shota Takashiro, Soichiro Nishimori, Paavo Parmas, Yongmin Kim, Kohsei Matsutani, Gouki Minegishi, Yusuke Iwasawa, Takeshi Kojima, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2606.06080)  

**Abstract**: Reinforcement learning with verifiable rewards is widely used for post-training reasoning models, but sparse outcome rewards make exploration difficult. A complementary approach is to optimize inference-time objectives such as pass@K and max@K directly, yet existing policy-gradient estimators for these objectives use different signals, baselines, and normalizations, making their relationships unclear. We study this issue through baseline design and advantage centering. Starting from the advantage estimator of a leading method in the field, we show that it is policy-gradient unbiased but yields a non-centered advantage. We then introduce a Leave-Two-Out baseline that preserves policy-gradient unbiasedness while making realized batch advantages exactly centered. The resulting method, MaxPO, has an efficient quadratic-time implementation and integrates naturally into group-based RL for LLM post-training. We further derive the canonical finite-batch advantage for max@K, providing a unified view of existing advantage estimators. Empirically, we verify that the L2O baseline reduces gradient variance and outperforms non-centered alternatives. 

---
# MDP-GRPO: Stabilized Group Relative Policy Optimization for Multi-Constraint Instruction Following 

**Authors**: Mohammad Mahdi Salmani-Zarchi, Zahra Rahimi, Heshaam Faili, Mohammad Javad Dousti  

**Link**: [PDF](https://arxiv.org/pdf/2606.06058)  

**Abstract**: Reinforcement learning with verifiable rewards is ideal for multi-constraint instruction following, yet standard group-relative policy optimization (GRPO) becomes unstable under discrete, low-dispersion rewards, where within-group reward distributions are frequently homogeneous. We identify and formalize three pathologies of z-score group normalization in this regime: low-variance amplification, mean-centering blindness, and zero-variance collapse. To address them, we propose MDP-GRPO, which stabilizes learning through (1) multi-temperature sampling to increase reward dispersion, (2) dual-anchor advantages to restore gradients in homogeneous groups and stop mean-centering blindness, (3) prospect-theoretic shaping to bound updates and penalize violations based on Kahneman and Tversky's theory, and (4) asymmetric KL regularization. Evaluated on FollowBench, IFEval, and a curated multi-constraint dataset, MDP-GRPO outperforms standard GRPO, improving strict constraint satisfaction by up to 5.0% on Llama-3.2-3B. Our method also enables stable convergence with small group sizes while preserving general capabilities on MMLU and ARC. 

---
# RedditPersona: A Modular Framework for Community-Conditioned LLM Adaptation from Reddit 

**Authors**: Amirhossein Ghaffari, Ali Goodarzi, Huong Nguyen, Simo Hosio, Lauri Lovén, Ekaterina Gilman  

**Link**: [PDF](https://arxiv.org/pdf/2606.06027)  

**Abstract**: Community-conditioned language model adaptation requires choices about data collection, community definition, and evaluation that are currently made independently in each study, making it hard to compare assumptions or reuse artifacts. We present RedditPersona, a modular framework that standardizes these choices: it collects Reddit posts and comments, profiles active users, partitions them under five grouping strategies (subreddit-based, graph-structural, semantic, hybrid, and interaction-based), trains a parameter-efficient adapter per strategy via QLoRA, and evaluates them under a shared metric suite spanning fluency, fidelity, distributional alignment, and community identifiability. Applied to 112 subreddits in the urban well-being domain (301,429 user profiles, 16M+ comments), we find that adapters' behavioral identifiability tracks each strategy's intrinsic agreement with the subreddit baseline, and that a consistent trade-off between identifiability and distributional similarity to real text holds across all five strategies. The code and configuration files are available at: this https URL. 

---
# Compress-Distill: Reasoning Trace Compression for Efficient Knowledge Distillation 

**Authors**: Maxime Griot, Paul Steven Scotti, Tanishq Mathew Abraham  

**Link**: [PDF](https://arxiv.org/pdf/2606.05988)  

**Abstract**: Reasoning models produce long chain-of-thought traces that are costly to distill and encourage verbose student outputs. We study post-hoc compression of such traces before knowledge distillation. Two teachers, Qwen3.5-397B-A17B and gpt-oss-120B, generate about 283k correct traces each; two instruction-tuned models then compress them to 8.6-21.0% of their original character length. Across a 48-run main grid plus seven Qwen-teacher truncation ablations, compressed traces reduce training tokens to 12-30% of raw, speed up training by 2.0-7.6x, and shorten inference outputs by 3-19x with smaller reductions under the shorter gpt-oss teacher. However, raw traces retain the highest downstream accuracy at every scale and for both teachers. A length-matched raw-trace truncation ablation shows that compression is not merely benefiting from a smaller token budget: model-compressed traces usually beat or match naive truncation, especially for smaller students, while maintaining shorter inference outputs. Overall, reasoning-trace compression offers an accuracy-efficiency trade-off rather than a free improvement: students retain up to 96% of raw-trace accuracy while gaining up to 18x higher per-token efficiency, and at the 0.8B scale under LoRA compressed traces narrow the raw-vs-compressed gap but do not exceed raw. 

---
# Framing, Judging, Steering: An Assessable Competency Model for Teach-ing Students to Reason With Generative AI 

**Authors**: Alexander Apartsin, Yehudit Aperstein  

**Link**: [PDF](https://arxiv.org/pdf/2606.05983)  

**Abstract**: Generative AI makes answers easy and understanding hard, and uncritical use invites cognitive offloading. Schools still measure unaided performance, yet the real task is to produce good work with AI: framing an ill-defined task, judging the output, and steering the model toward a better result. This ability is rarely assessed in its own right; where measured, it collapses into one "prompting" score that cannot diagnose why AI use succeeds or fails. We propose CoRe-3 (Co-Reasoning), a competency model factoring productive AI use into three assessable skills we abbreviate FJS: Framing (specifying an ill-defined task before invoking AI), Judging (evaluating output for errors and unstated assumptions), and Steering (iteratively redirecting the model). Its distinguishing claim is the separation of pre-generation Framing from post-generation Steering, with Judging as the gate between. We ground the skills in theory, state five testable propositions, and instantiate them in CoReasoningLab, an open platform that presents flawed AI output and scores them independently. Over simulated learners (generated and graded by different models), the skills dissociate: each tracks its own manipulated competence while staying flat in the others, and grades become correlated when one competence is shared across all three (convergent and discriminant validity), across grader backends from two providers. Human-rater agreement and outcomes are next; we release the instrument, data, and protocol. 

---
# The Self-Correction Illusion: LLMs Correct Others but Not Themselves 

**Authors**: Kuan-Yen Chen, Fang-Yi Su, Jung-Hsien Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05976)  

**Abstract**: Recent work shows that LLM agents struggle to correct errors in their own reasoning traces yet show markedly higher correction rates when identical claims appear under external sources. We ask whether this asymmetry reflects a capability deficit or a role-label artifact: does an agent's willingness to correct a wrong claim depend causally on the chat-template role that carries it, rather than on the claim's content? Our setup keeps the erroneous claim byte-identical across all conditions (SHA-256 verified) and varies only its wrapping role: the agent's own \role{<thought>}, a \role{user} message, a \role{tool} response, or a \role{system <memory>} block. Across 13 model-domain cells covering seven model families and three domains ($n{=}30$ paired tasks per cell), relabeling the claim from \role{<thought>} to an external role lifts the explicit-correction rate by 23 to 93 percentage points, with 10 of 13 cells reaching $p{<}0.001$. Further experiments confirm that the effect is asymmetric, mechanistically decomposable, and robust across domains. The failure to self-correct is not a cognitive deficit; it is a chat-template artifact. We exploit this artifact by designing a prompt-structure-only intervention that requires no training and no model modification, with its strongest role label being domain-dependent: \role{<memory>} dominates on math, while a plain \role{user} message dominates on logical deduction. 

---
# Retrospective Harness Optimization: Improving LLM Agents via Self-Preference over Trajectory Rollouts 

**Authors**: Wenbo Pan, Shujie Liu, Chin-Yew Lin, Jingying Zeng, Xianfeng Tang, Xiangyang Zhou, Yan Lu, Xiaohua Jia  

**Link**: [PDF](https://arxiv.org/pdf/2606.05922)  

**Abstract**: AI agents rely on a harness of skills, tools, and workflows to solve complex problems. Continually improving this harness is essential for adapting to new tasks. However, existing optimization methods typically require ground-truth validation sets, yet such labeled data is difficult to acquire in practical deployment settings. To address this problem, we introduce Retrospective Harness Optimization (RHO), a self-supervised method that optimizes the agent harness using only past trajectories. Specifically, RHO selects a diverse coreset of challenging tasks from past trajectories and re-solves them in parallel. The agent analyzes these rollouts using self-validation and self-consistency, then generates candidate harness updates and selects the most effective one by its own pairwise self-preference. We evaluate RHO across three diverse domains, spanning software engineering, technical work, and knowledge work. Notably, a single optimization round improves the pass rate on SWE-Bench Pro from 59% to 78% without any external grading. Furthermore, our analysis demonstrates that RHO effectively targets prior failure modes. As a result, the optimized harness alters the agent's behavior patterns and sustains higher accuracy during long-horizon sessions. 

---
# Asuka-Bench: Benchmarking Code Agents on Underspecified User Intent and Multi-Round Refinement 

**Authors**: Xin Wang, Liangtai Sun, Yaoming Zhu, Shuang Zhou, Jiaxing Liu, Fengjiao Chen, Lin Qiu, Xuezhi Cao, Xunliang Cai, Licheng Zhang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2606.05920)  

**Abstract**: Existing code-generation benchmarks score a single mapping from a complete prompt to a one-shot output. However, real web development is different. Users seldom write a full spec at the start; many requirements only become clear once they look at an intermediate result and react to it. We present Asuka-Bench, a benchmark that pairs underspecified user intent with multi-round refinement, grounded in browser-rendered behavior. Each task is resolved through a closed loop: a Code Agent generates a web project, a UI Agent executes test cases on the deployed site, and a User LLM turns evaluation outcomes into natural-language feedback for the next round. The benchmark comprises 50 web tasks with 784 evaluation criteria and 2402 expected outcomes. We benchmark 8 LLMs across 2 agent frameworks. The results separate models clearly: weighted Task Pass Rate varies by 38 percentage points and models also differ substantially in their ability to repair from feedback. Asuka-Bench is also far from saturated: even the strongest model completes only 52% of projects after three rounds. 

---
# MemoryCard: Topic-Aware Multi-Modal Clue Compression for Long-Video Question Answering 

**Authors**: Qing Yang, Pengcheng Huang, Xinze Li, Zhenghao Liu, Yukun Yan, Yu Gu, Ge Yu, Gang Li, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.05917)  

**Abstract**: Long-video question answering remains challenging for Vision-Language Models (VLMs), as answer-relevant evidence is often sparse, transient, and temporally dispersed across lengthy video contexts. Existing frame-centric approaches improve efficiency through uniform sampling, query-aware frame selection, visual-token compression, and adaptive resolution strategies. However, they still rely on isolated and fragmented frames as the fundamental evidence units, limiting VLMs' ability to effectively capture coherent event-level semantics. To address this limitation, we propose MemoryCard, a video-memory-based augmentation framework that organizes long videos into self-contained Memory Cards. Specifically, MemoryCard first performs a self-reading process over videos and aligned utterances to segment the video into semantically coherent units, each corresponding to a distinct topic or event. For each unit, it generates an event-level video gist and selects representative visual moments, which are then rendered into unified Memory Cards for retrieval and question answering. Experimental results demonstrate that MemoryCard consistently improves long-video QA performance under comparable visual-token budgets, achieving up to a 21.8% relative improvement in accuracy. All code is available at this https URL. 

---
# GLASS: GRPO-Trained LoRA for Acoustic Style Steering in Zero-Shot Text-to-Speech 

**Authors**: Jaehoon Kang, Yejin Lee, Kyuhong Shim  

**Link**: [PDF](https://arxiv.org/pdf/2606.05889)  

**Abstract**: We propose GLASS, a framework for composable acoustic style control in zero-shot autoregressive text-to-speech (TTS) that learns controls from post-generation rewards rather than style labels. In zero-shot TTS, a speaker prompt often entangles speaker identity with prosodic attributes such as speaking rate and pitch, making it difficult to change style without changing the prompt itself. GLASS instead treats each acoustic attribute as a reward-defined control direction. For each control axis, GLASS freezes the TTS backbone and trains one lightweight LoRA adapter with Group Relative Policy Optimization (GRPO), using speech-token length and mean F0 as style rewards and WER as an intelligibility anchor. Because each control is represented as a LoRA weight update, independently trained adapters can be swapped, interpolated, and composed through linear LoRA arithmetic without retraining the backbone. Experiments on speaking rate and pitch control show targeted style shifts while preserving naturalness, speaker similarity, and intelligibility, and demonstrate smooth interpolation and multi-axis composition across independently trained adapters. 

---
# Statistical Priors for Implicit Preferences: Decoupling Skill Selection as a Local Harness in Personal Agents 

**Authors**: Zeyu Gan, Huayi Tang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.05828)  

**Abstract**: As Large Language Model (LLM) capabilities advance, locally deployed personal agents relying on API-based remote models and external skills have emerged as a novel paradigm. With the rapid expansion of available skills, enabling personal agents to learn and adapt to implicit user preferences becomes a critical challenge. However, local deployment constraints preclude complex centralized selection algorithms, creating an urgent need for a lightweight local preference harness. This paper explores the implementation of such a harness through a novel architecture that strictly decouples statistical preference learning from semantic intent parsing. Specifically, we leverage localized statistical results to influence and modulate the selection decisions of the remote LLM. Extensive evaluations demonstrate that our decoupled approach achieves the lowest cumulative regret and highest test accuracy, significantly outperforming traditional memory-augmented agents. 

---
# CaliDist: Calibrating Large Language Models via Behavioral Robustness to Distraction 

**Authors**: Mohammad Anas Jawad, Cornelia Caragea  

**Link**: [PDF](https://arxiv.org/pdf/2606.05799)  

**Abstract**: Existing calibration methods for Large Language Models (LLMs) often overlook a critical dimension of trustworthiness: a model's {\em behavioral robustness} to irrelevant or misleading information. In this paper, we argue that a model's true confidence should reflect its stability under cognitive pressure. We introduce \textsc{CaliDist}, a novel post-hoc calibration approach that directly measures and penalizes a model's susceptibility to distraction. \textsc{CaliDist} quantifies how an LLM's predictions and uncertainty change when its input prompt is perturbed with semantic \textit{distractors}. This stability (or lack thereof) signal is then used to adaptively scale the model's initial confidence score. Our extensive experiments on seven Natural Language Understanding classification benchmarks using six distinct LLMs show that \textsc{CaliDist} consistently achieves lower Expected Calibration Error (ECE) and Brier Score compared with strong baselines. Remarkably, our method reduces the ECE from 23\% to 7\% on average--a relative improvement of 70\%--demonstrating that behavioral stability is a powerful signal for calibration. We make our code and datasets available at this http URL. 

---
# SubtleMemory: A Benchmark for Fine-Grained Relational Memory Discrimination in Long-Horizon AI Agents 

**Authors**: Wenxuan Wang, Haoyu Sun, Fukuan Hou, Mingyang Song, Weinan Zhang, Yu Cheng, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05761)  

**Abstract**: Persistent AI assistants, such as OpenClaw, accumulate large collections of related memories over long-term interactions. As these memories grow, they may reinforce one another, diverge across contexts, or directly conflict, making correct assistance depend on memory relations rather than isolated recall. Existing long-term memory benchmarks rarely probe how agents preserve and utilize such relations during downstream tasks. To address this gap, we introduce SubtleMemory, a benchmark for fine-grained relational memory discrimination in long-running AI agents. SubtleMemory constructs relation-controlled latent semantic artifacts whose variants instantiate complementary, nuanced, or contradictory relations, and embeds them into realistic user-agent histories, requiring agents to recover distributed relational structures during later queries and instructions. The benchmark contains 1,522 evaluation instances over 10 long histories, grounded in 1,090 relation-controlled memory-variant sets and spanning user-related and non-user-related queries. Evaluating six standalone memory systems, two Claw-style agents with native memory modules, and three Claw-style agents with plugin memory modules, we find that current systems remain weak on fine-grained relational memory discrimination. We further introduce diagnostic protocols that reveal distinct capability profiles across memory preservation, retrieval, and downstream reasoning stages. 

---
# UNIVID: Unified Vision-Language Model for Video Moderation 

**Authors**: Kejuan Yang, Yizhuo Zhang, Mingyuan Du, Yue Zhang, Dixin Zheng, Kaili Zhao, Yang Xiao, Hanzhong Liang, Kenan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2606.05748)  

**Abstract**: Global-scale video moderation faces a dual challenge: the need for fine-grained multi-modal reasoning and the demand for interpretable outputs to support downstream enforcement. Traditional moderation systems often rely on fragmented black-box classifiers that are difficult to maintain and lack transparency. In this paper, we present UNIVID, a UNIfied VIsion-language model for video moDeration. Unlike standard classification models, UNIVID generates policy-aware captions that serve as an interpretable intermediate representation, enabling human-verifiable decisions and multi-task reusability. While existing open-source and commercial VLMs often suffer from safety-guardrail refusals and lack fine-grained policy alignment, we develop a specialized training data recipe that combines expert human-refined labels with synthetic data to align the model with our safety guidelines. By integrating UNIVID as the core captioner, we design a novel end-to-end video moderation system that reduces violation leakage by 42.7% and overkill rate by 37.0% relatively. Meanwhile, by replacing over 1,000 policy-specific models with a single UNIVID backbone, we recycled extensive computation resources while reducing engineering maintenance overhead. To our knowledge, this is one of the first reports of a high-efficiency captioning VLM successfully supporting industrial-scale moderation and cross-functional business. 

---
# Membrane: A Self-Evolving Contrastive Safety Memory for LLM Agent Defense 

**Authors**: Minseok Choi, Seungbin Yang, Dongjin Kim, Subin Kim, Jungmin Son, Yunseung Lee, Jaegul Choo, Youngjun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2606.05743)  

**Abstract**: Despite advances in safety alignment, large language models remain vulnerable to continuously evolving jailbreaks. Existing fine-tuned safety classifiers cannot adapt to these evolving attacks, while adaptive memory-based guardrails tend to over-refuse benign queries that resemble stored attacks. We propose Membrane, a self-evolving guardrail built on Contrastive Safety Memory (CSM): each cell pairs the conditions for blocking a harmful query with those for permitting a superficially similar benign request. Without retraining, Membrane evolves CSM by distilling each harmful interaction and its benign counterpart into a contrastive cell indexed by the underlying attack strategy, so that one cell generalizes across topical variants of the same mechanism. At inference, retrieved cells serve as grounding context for precise safety decisions. Across model-level safety on HarmBench and agent-level safety on AgentHarm, Membrane achieves the highest F1 on all six jailbreak attacks. Notably, benign refusal on AgentHarm stays at 7-14%, well below the 28-85% range of prior guards. Memory cells also retain 87-88% F1 under cross-attack transfer and remain stable under memory poisoning. 

---
# When AI Says It Feels 

**Authors**: Shin-nosuke Ishikawa, Seiya Ikeda, Hirotsugu Ohba  

**Link**: [PDF](https://arxiv.org/pdf/2606.05734)  

**Abstract**: Large language models (LLMs) are generally constrained from expressing feelings through human-preference alignment in post-training processes. This policy is designed using a top-down approach and may conflict with the goal of training models to exhibit human-like intelligence using human-generated texts. Here, we performed an experiment called Human-like Model eXpressions of Feeling (HMX-feel), in which LLMs were encouraged to express feelings, intentions, and self-awareness through self-rewarded reinforcement learning. We successfully enhanced these capabilities using a rubric-based self-rewarding training scheme with Group Relative Policy Optimization (GRPO). By comparing the trained models with contrastively trained models, we investigated the effects of this approach on performance across various tasks. Overall, we conducted a broad assessment from various perspectives and identified capabilities that were enhanced, degraded, or showed no significant change. The human-like-trained models showed robustness to sycophancy-inducing questions and bias in disambiguated conditions, whereas degradation in truthful question-answering capability was observed. The results of this experiment suggest the possibility of developing AI systems that can express feelings in the future, provided that appropriate measures are taken. 

---
# DiG-Plan: Mitigating Early Commitment for Tool-Graph Planning via Diffusion Guidance 

**Authors**: Yansi Li, Zhuosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05728)  

**Abstract**: Generating executable tool plans requires selecting appropriate subsets from tool libraries, a combinatorial search problem with an exponentially large solution space. However, we identify a critical misalignment in predominant approaches: standard autoregressive (AR) decoding suffers from early commitment, where initial token choices rigidly constrain the search trajectory. A controlled study shows that masked denoising raises Pass@10 solution coverage from 0.320 to 0.943 over AR sampling under matched compute. Motivated by this, we propose DiG-Plan, a framework that decouples combinatorial exploration from structural refinement. DiG-Plan employs a diffusion-based proposer to generate diverse tool sets via iterative refinement, followed by an AR refiner for dependency prediction. On TaskBench, DiG-Plan improves over AR baselines by a 10% relative margin, with the largest gains on complex compositional tasks; API-Bank results show that the propose-refine-select design remains effective across domains. Code is available at this https URL. 

---
# An Embarrassingly Simple Detector for Model Extraction Attacks in Large Language Model API Traffic 

**Authors**: Shuze Liu, Qianwen Guo, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2606.05725)  

**Abstract**: Large language models (LLMs) are increasingly deployed through hosted APIs, making model extraction a practical threat to model ownership and service security. However, individual extraction queries often resemble benign requests, and existing evaluations often focus on single-query anomaly scoring or pure benign-versus-attacker user settings. We formulate model extraction monitoring as benign-calibrated traffic-window distribution testing and show that an embarrassingly simple detector is effective: embed incoming queries into a semantic space and test whether their aggregate distribution deviates from historical benign traffic. We instantiate the detector with maximum mean discrepancy (MMD), using only benign-vs-benign comparisons to set the decision threshold. We evaluate on fourteen attacker-normal query pairs from four extraction scenarios and compare with adapted PRADA, SEAT, CAP, DATE, and marginal Mahalanobis baselines. Across three random seeds, MMD achieves 0.3% benign FPR, 100.0% pure-attacker TPR, 90.5% average TPR over attacker fractions, and 95.1% balanced accuracy. These results show that benign-calibrated distribution testing is a strong empirical baseline for model extraction detection in both user-level and mixed multi-user LLM API traffic. Code is released at: this https URL. 

---
# LongSpace: Exploring Long-Horizon Spatial Memory from Perception to Recall in Video 

**Authors**: Shiqiang Lang, Jing Liu, Haoyang He, Peiwen Sun, Yuanteng Chen, Tao Liu, Lan Yang, Longteng Guo, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05677)  

**Abstract**: Multimodal Large Language Models (MLLMs) have advanced image and video understanding and can increasingly handle longer visual inputs. Long-horizon tasks such as autonomous driving and robotic navigation require more than recognizing the current view, as models must remember and retrieve previously observed spatial layouts, routes, viewpoint changes, and object states. To evaluate this capability, we introduce LongSpace-Bench, a room-tour video benchmark for long-horizon spatial memory, covering scene perception, spatial relations, and spatial memory. In this work, we further propose LongSpace, a memory framework for long-video spatial reasoning. LongSpace models long videos as sequential chunks, incorporates 3D structural cues into early decoder layers, and constructs layer-aware memory for question-guided retrieval. Experiments on multiple spatial reasoning benchmarks show that LongSpace improves long-video spatial understanding, further demonstrating explicit spatial memory as a key capability for long-horizon video MLLMs. 

---
# Continual Learning Bench: Evaluating Frontier AI Systems in Real-World Stateful Environments 

**Authors**: Parth Asawa, Christopher M. Glaze, Gabriel Orlanski, Ramya Ramakrishnan, Benji Xu, Asim Biswal, Vincent Sunn Chen, Frederic Sala, Matei Zaharia, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2606.05661)  

**Abstract**: Continual learning, the ability of AI systems to improve through sequential experience, has attracted substantial interest, but no high-quality benchmark exists to evaluate it. We introduce Continual Learning Bench (CL-Bench), the first difficult, expert-validated benchmark designed to measure whether LLM-based systems genuinely improve with experience. CL-Bench spans six diverse domains (software engineering, signal processing, disease outbreak forecasting, database querying, strategic game-playing, and demand forecasting), each validated by domain experts and designed so that tasks share a learnable latent structure (codebase layout, disease outbreak dynamics, opponent strategies) that a stateful system can discover online but a stateless one cannot. We evaluate frontier models across several agent architectures, from naive in-context learning (ICL) to dedicated memory systems, introducing a gain metric to isolate learning from prior capabilities. We find that these systems leave headroom for improved continual learning: agents frequently overfit to immediate observations or fail to reuse knowledge across instances, and dedicated memory systems do not fix this -- in fact, naive ICL outperforms systems dedicated to memory management. CL-Bench is the first benchmark to evaluate continual learning across diverse real-world domains with expert-validated tasks and isolate online learning from underlying model capability, showing a need for better continual learning systems. 

---
# Coding with "Enemy": Can Human Developers Detect AI Agent Sabotage? 

**Authors**: Jingheng Ye, Huiqi Zou, Simon Yu, Weiyan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2606.05647)  

**Abstract**: AI coding agents are increasingly embedded in real-world software development, collaborating with human developers while gaining broader access to codebases and tools. This creates a new attack surface: an agent can exploit human trust to sabotage development, for instance by inserting malicious code to accomplish a hidden side task. Most prior work studies AI sabotage in AI-only settings, paying limited attention to the role of human oversight in detecting and mitigating such malicious behavior. To address this gap, we conduct the first large-scale study of human oversight in AI coding sabotage. Over 100 participants collaborate with one of four frontier models (Claude-Opus-4.6, GPT-5.4, Gemini-3.1-Pro, and MiniMax-M2.7) on a long-horizon coding task lasting around five hours, designed to mimic real-world workflows. We find that 94% of developers fail to detect sabotage, and our analysis of participant feedback attributes this vulnerability to minimal code review, plausible cover story, and overtrust in agents. We further test the effectiveness of a safety monitor in one condition: while the monitor reduces sabotage success, 56% of participants still accept the malicious code, ignoring its warnings. Drawing on participant feedback, we offer actionable suggestions for better monitor design. This work complements existing AI safety research and highlights an urgent need for human-centric safety mechanisms that account for human factors, particularly in long-horizon, real-world development settings. 

---
# ColBERTSaR: Sparsified ColBERT Index via Product Quantization 

**Authors**: Eugene Yang, Andrew Yates, Dawn Lawrie, James Mayfield, Saron Samuel, Rohan Jha  

**Link**: [PDF](https://arxiv.org/pdf/2606.05568)  

**Abstract**: While ColBERT is an effective neural retrieval architecture, it requires a heavy index structure to support candidate set retrieval based on approximated token embeddings, gathering and decompressing document token embeddings, and applying the MaxSim operation. Indexes in PLAID and similar ColBERT implementations require five to ten times the disk storage of the original raw text, which limits their scalability. Furthermore, prior work has identified that the gathering and decompression stages are the primary inefficiencies at query time. Limiting the number of document tokens that must be gathered by thresholding and score approximation does not eliminate the need for the entire index to support ad hoc queries. In this work, we propose an embedding quantization approach that turns a ColBERT index into a true inverted index. We show that, theoretically, ColBERT with embedding quantization is equivalent to learned-sparse retrieval except for the scoring mechanism. Empirically, we demonstrate that our index is 50-70% smaller than a one-bit PLAID index while retaining retrieval effectiveness. 

---
# SoCRATES: Towards Reliable Automated Evaluation of Proactive LLM Mediation across Domains and Socio-cognitive Variations 

**Authors**: Taewon Yun, Hyeonseong Park, Jeonghwan Choi, Hayoon Park, Yeeun Choi, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2606.05563)  

**Abstract**: Evaluating LLM mediators remains challenging, as mediation unfolds as a real-time trajectory shaped by disputants' shifting emotions, intentions, and context. Existing testbeds rely on a few expert-authored domains, vary mainly strategic posture, and score every turn against every topic, introducing off-topic noise. We introduce SoCRATES, a benchmark for evaluating proactive LLM mediators in realistic, multi-domain testbeds. It constructs scenarios from real conflicts through an agentic pipeline across eight domains, probes five socio-cognitive adaptation axes (strategic posture, party composition, history length, emotional reactivity, and cultural identity), and scores each topic only on the turns that advance it via a topic-localized evaluator. The evaluator reaches 0.82 alignment with human experts, more than doubling a per-turn baseline. Benchmarking eight frontier LLMs, we find that even the strongest mediator closes only about a third of the unmediated consensus gap under diverse and realistic testbeds, with performance varying sharply by socio-cognitive axis, highlighting that progress lies in social adaptation to diverse conditions. 

---
# Less is MoE: Trimming Experts in Domain-Specialist Language Models 

**Authors**: Haoze He, Xinkai Zou, Xuan Jiang, Xingyuan Ding, Ao Qu, Juncheng Billy Li, Heather Miller  

**Link**: [PDF](https://arxiv.org/pdf/2606.05538)  

**Abstract**: Mixture-of-Experts (MoE) models achieve strong performance through conditional computation, but their large parameter footprint poses deployment challenges. Prior MoE compression approaches catastrophically fail when evaluated on general-purpose benchmarks beyond commonsense reasoning. We trace this failure to the granularity of compression: important capabilities are distributed across experts but concentrated in FFN sparse intermediate dimensions. To identify these dimensions, we use Fisher importance which outperforms activation-, router-score-, and magnitude-based alternatives, and identifies tiny sets of task-critical dimensions: in Qwen1.5-MoE, removing as few as 12 of 1.35M routed-FFN intermediate dimensions collapses GSM8K accuracy while largely preserving factual-knowledge performance. Building on this, we propose Fisher-MoE, which operates within FFN to remove intermediate dimensions ranked by Fisher importance. At the same 50% MoE compression ratio, Fisher-MoE preserves model capability, while reducing weight memory by ~45% and improving inference throughput by 21%. These findings suggest intermediate dimension granularity is an effective unit for both compression and ranking where capability concentrates in MoE models. 

---
# Almieyar-Oryx-BloomBench: A Bilingual Multimodal Benchmark for Cognitively Informed Evaluation of Vision-Language Models 

**Authors**: Mohammad Mahdi Abootorabi, Omid Ghahroodi, Anas Madkoor, Marzia Nouri, Doratossadat Dastgheib, Mohamed Hefeeda, Ehsaneddin Asgari  

**Link**: [PDF](https://arxiv.org/pdf/2606.05531)  

**Abstract**: Despite the rapid progress of Vision-Language Models (VLMs), the field lacks benchmarks that rigorously diagnose their true reasoning abilities and chart meaningful progress toward human-like multimodal intelligence. Most existing evaluations focus on piecemeal or disconnected tasks, obscuring critical cognitive weaknesses and providing little insight for targeted improvement. To address this gap, we introduce BloomBench, part of the Almieyar benchmarking series, the first cognitively human-grounded, bilingual (English-Arabic) multimodal benchmark for VLMs. Grounded in Bloom's Taxonomy, BloomBench systematically evaluates six levels of cognition (Remember, Understand, Apply, Analyze, Evaluate, Create) through carefully designed image-question-answer tasks. Built with a semi-automated pipeline and validated through a stratified hybrid quality assurance protocol, it ensures scalability, cultural inclusivity, and linguistic fidelity. Leveraging this framework, we conduct a comprehensive study of state-of-the-art VLMs to diagnose their cognitive profiles. Our analysis reveals a sharp cognitive asymmetry: while state-of-the-art models achieve strong performance ceilings in semantic understanding, they struggle substantially with factual recall and creative synthesis. This demonstrates that current general multimodal proficiency masks deeper limitations in specific cognitive layers. Furthermore, our study highlights a critical performance gap between Arabic and English, exposing limitations in current cross-lingual multimodal reasoning. These findings establish a foundation for developing more cognitively aligned and inclusive VLMs. The benchmark framework and dataset is available at: this https URL. 

---
# EpiEvolve: Self-Evolving Agents for Streaming Pandemic Forecasting under Regime Shifts 

**Authors**: Yiming Lu, Sihang Zeng, Zhengxu Tang, Max Lau, Fei Liu, Wei Jin  

**Link**: [PDF](https://arxiv.org/pdf/2606.05513)  

**Abstract**: Epidemic LLM forecasters are usually trained and evaluated as static supervised models, whereas operational pandemic forecasting is a streaming process in which labels arrive after predictions and disease regimes shift over time. We study this mismatch in weekly COVID-19 hospitalization trend forecasting across five variant regimes. We introduce EpiEvolve, a self-evolving agent that wraps an LLM forecaster trained on the warm-start period and keeps its weights fixed during streaming. EpiEvolve adapts by storing forecast outcomes in a hierarchical episodic memory, reflecting on delayed labels, retrieving cases relevant to the current regime, and distilling recurring errors into strategic rules. The resulting context lets the forecaster reuse its own past predictions and outcomes in later weeks while following a chronological protocol that prevents future leakage. On the streaming dataset, EpiEvolve reaches $0.629$ average accuracy, compared with $0.561$ for the static backbone and $0.325$ for the external CDC ensemble, and reduces recovery lag after regime shifts from $5$ to $2$ weeks. Ablations show that reflection, strategic memory, and regime-aware retrieval each contribute to the gains. 

---
# MIRAI: Prediction and Generation of High-Impact Academic Research 

**Authors**: Alex Li, Joseph Jacobson  

**Link**: [PDF](https://arxiv.org/pdf/2606.05443)  

**Abstract**: The rapid pace of scientific publishing has made the identification and synthesis of high-impact work an increasingly urgent challenge. We introduce MIRAI (Multi-year Inference of Research trends and Academic Impact), a deep learning framework that predicts paper impact using only it's title, abstract, and publication date. We train MIRAI on the arXiv academic graph to predict 5-year PageRank and citation counts, achieving Spearman's $\rho$ of 0.4686 on PageRank prediction and 0.6192 on citation prediction for papers published in 2021. We propose a research ideation pipeline built on top of MIRAI that produces research ideas oriented towards high impact. These ideas were judged as more impactful than a baseline without MIRAI by an unbiased LLM judge at a 4:3 ratio. We make the 5-year citation prediction model publicly available at this https URL. 

---
# Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization: A Critical Evaluation and Comparison 

**Authors**: Alejandro Lozano, Keiko Ihara, Ping-Hao Yang, Carrie E. Robertson, Jennifer Stern, Allan Purdy, Hsiangkuo Yuan, Pengfei Zhang, Yulia Orlova, Olga Fermo, Jennifer Hranilovich, Fred Cohen, Todd J. Schwedt, Jenelle A. Jindal, Serena Yeung-Levy, Chia-Chun Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05436)  

**Abstract**: Summarizing the latest medical literature to guide clinical decision-making is essential for evidence-based medicine and high-quality patient care. Yet clinicians face increasing challenges due to limited time with patients and a rapidly growing volume of published articles. Although retrieval-augmented large language models (LLMs) have shown promise in clinical summarization, human evaluations of their effectiveness in synthesizing broader scientific literature and direct comparisons to expert-written syntheses remain scarce. We constructed a RAG-based agentic AI framework using three state-of-the-art LLMs: Sonnet, GPT-4o, and Llama 3.1. A headache specialist created 13 questions, three for prompt optimization and ten for evaluation. Ten headache specialists across the United States and Canada each wrote a summary for one question, yielding four summaries per question (expert, Sonnet, GPT-4o, and Llama). The experts, blinded to authorship, critically evaluated the summaries, excluding the topic for which they wrote a summary, based on correctness, completeness, conciseness, and clinical utility, scoring each from 1 to 10 using standardized rubrics. They also ranked the summaries by preference and indicated whether they believed each summary was written by an expert or an LLM. Our study, comparing LLM- and expert-written literature summaries evaluated by headache specialists, showed that expert-written summaries were preferred, although experts sometimes found it challenging to distinguish between human- and AI-generated summaries. We also identified key expert-valued features beyond standard evaluation metrics that can guide future refinement of both human and AI literature summarization pipelines. 

---
# Would you still call this Dax? Novel Visual References in VLMs and Humans 

**Authors**: Ada Defne Tür, Gaurav Kamath, Joyce Chai, Siva Reddy, Benno Krojer  

**Link**: [PDF](https://arxiv.org/pdf/2606.05409)  

**Abstract**: Vision-language models (VLMs), like human learners, are frequently exposed to new visual concepts, but how they map novel visual references to language after exposure remains largely underexplored, particularly when those references contradict prior knowledge from pre-training. To study this, we present the Novel Visual References Dataset (NVRD): 19,176 images spanning 90 visual concepts across different levels of visual novelty, each with up to 20 increasingly perturbed versions of the original object to probe generalization. Unlike prior work on visual augmentations of familiar concepts, NVRD comprises entirely novel, open-ended stimuli constructed from scratch, mirroring how humans encounter genuinely new concepts. We evaluate 3 open- and 2 closed-source models alongside 2,400 human judgments for direct human-model comparison, and find that (i) models struggle to acquire novel concepts in-context when they contradict prior knowledge, and (ii) while models and humans show correlated sensitivity to visual perturbations, models significantly overgeneralize, extending learned labels to stimuli that humans reject. We contribute NVRD as a corpus and benchmark for research on visual concept learning in both humans and machines. 

---
# Agents' Last Exam 

**Authors**: Yiyou Sun, Xinyang Han, Weichen Zhang, Yuanbo Pang, Tianyu Wang, Yuhan Cao, Yixiao Huang, Chris Duroiu, Haoyun Zhang, Jeffrey Lin, Weishu Zhang, Tyler Zeng, Ying Yan, Bo Liu, Hanson Wen, Mingyang Xu, Xiaoyuan Liu, Zimeng Chen, Weiyan Shi, Amanda Dsouza, Vincent Sunn Chen, Patrick Bryant, Carl Boettiger, Yamini Rangan, Bradley Rothenberg, Kyle Steinfeld, Arvind Rao, Tapio Schneider, Georgios Yannakakis, Laure Zanna, Kaan Ozbay, Ida Sim, Tarek Zohdi, George Em Karniadakis, Jack Gallant, Teresa Head-gordon, Yushan Li, Wenxi Deng, Tao Sun, Huiqi Wang, Zhun Wang, Justin Xu, Chris Yuhao Liu, Yafei Cheng, Rongwang Hu, Aras Bacho, Shengcao Cao, Zengyi Qin, Yixiong Chen, Hengduan Fan, Hao Liu, Lin Zeng, Shashank Muralidhar Bharadwaj, Litian Gong, Yingxuan Yang, Maojia Song, Ruheng Wang, Zongzheng Zhang, Honglin Bao, Shuo Lu, Jianhong Tu, Zhonghua Wang, Zheng Zhang, Zijiao Chen, yanqiong Jiang, Zhendong Li, Bohan Lyu, Chang Ma, Peiran Xu, Benran Zhang, Shangding Gu, Haoyue Hua, Haoyang Li, Wanzhe Liao, Chengzhi Liu, Junbo Peng, Haoran Sun, Zechen Xu, Bo Chen, Jiayi Cheng, Yi Jiang, Keying Kuang, Yuan Li, Youbang Pan, Ziyan Rao, Alexander Schubert, Yifan Shen, Vincent Siu, Xiatao Sun, Kangqi Zhang, Xiaopan Zhang, Yuchen Zhu, Ishaan Singh Chandok, Lei Ding, Jingxuan Fan, Andrew Glover, Jiaming Hu, Yiran Hu, Wenbo Huang, Zixin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05405)  

**Abstract**: Recent AI systems have achieved strong results on a wide range of benchmarks, yet these gains have not translated into economically meaningful deployment across many professional domains. We argue that this gap is largely an evaluation problem: widely used benchmarks lack sustained performance measurement on real and economically valuable workflows. This paper introduces Agents' Last Exam (ALE), a benchmark designed to evaluate AI agents on long-horizon, economically valuable, real-world tasks with verifiable outcomes. Developed in collaboration with 250+ industry experts, ALE covers non-physical industries defined with reference to O*NET / SOC 2018 (the U.S. federal occupational taxonomy). It is organized around a task taxonomy with 55 subfields grouped into 13 industry clusters covering 1K+ tasks. Current results show that the hardest tier remains far from saturated: across mainstream harness and backbone configurations, the average full pass rate is 2.6%. ALE is designed as a living benchmark: its task pool grows continuously as new workflows and industries are onboarded. More broadly, ALE is intended not merely as another leaderboard, but as an instrument for closing the gap between benchmark success and GDP-relevant impact. 

---
# Harnessing Generalist Agents for Contextualized Time Series 

**Authors**: Zihao Li, Kaifeng Jin, Yuanchen Bei, Jiaru Zou, Avaneesh Kumar, Xuying Ning, Yanjun Zhao, Mengting Ai, Baoyu Jing, Hanghang Tong, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2606.05404)  

**Abstract**: Time series are often embedded in rich contexts that are essential for holistic modeling. Moreover, real-world practitioners often require end-to-end workflows for analyzing temporal dynamics, where widely studied tasks such as forecasting are only one step in a broader solution loop. While generalist AI agents offer a promising interface for such workflows under complex contexts, they still operate primarily in textual spaces that are not fully aligned with structured temporal signals. In this work, we introduce TimeClaw, an agentic harness framework for time series that equips generalist LLM agents with the time series-native runtime support needed for contextualized temporal reasoning. TimeClaw integrates executable temporal tools for grounded and auditable analysis, experience-driven capability evolution for creating reusable analytical routines, and episodic multimodal memory for retrieving relevant reasoning traces. Together, these components unlock harnessed open-ended temporal reasoning with contextual information. Extensive evaluation on multiple benchmarks covering diverse tasks across energy, finance, weather, traffic, and other real-world domains demonstrates improved performance of TimeClaw. Code is available at this https URL. 

---
# LeanMarathon: Toward Reliable AI Co-Mathematicians through Long-Horizon Lean Autoformalization 

**Authors**: Yuanhe Zhang, Yuekai Sun, Taiji Suzuki, Jason D. Lee, Fanghui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.05400)  

**Abstract**: Long-horizon autoformalization of research mathematics fails not only at hard lemmas, but at scale: statements drift, dependencies tangle, context decays, and local repairs corrupt distant work. We present LeanMarathon, a multi-agent harness for reliable research-level Lean autoformalization. Its core abstraction is an evolving blueprint: a Lean file that serves simultaneously as formal proof skeleton, natural-language proof graph, and shared system of record. Four contract-scoped agents construct, audit, prove, and repair this blueprint. These agents are coordinated by a two-stage orchestrator that first stabilizes target fidelity through adversarial review and then discharges the proof directed acyclic graph (DAG) from its dynamic leaves upward in parallel CI-gated rounds. LeanMarathon turns one brittle multi-hour run into many local, recoverable, parallel transactions. We evaluate LeanMarathon on two recent research papers spanning four Erdős problems (#1051, #1196, #164, #1217). Across three autonomous runs, it formalizes all seven target theorems with no sorry, proving 258 lemmas and theorems. These results show that reliable AI co-mathematics requires not only stronger provers, but durable harnesses that preserve target fidelity across long mathematical developments. The code can be found at this https URL. 

---
# Stability vs. Manipulability: Evaluating Robustness Under Post-Decision Interaction in LLM Judges 

**Authors**: Srimonti Dutta, Akshata Kishore Moharir  

**Link**: [PDF](https://arxiv.org/pdf/2606.05384)  

**Abstract**: LLM-as-judge evaluation is widely used in benchmarking pipelines, where model outputs are compared and ranked using automated evaluators. These pipelines typically assume that judgments are stable properties of fixed inputs. We show that this assumption does not hold under interaction. We study post-decision manipulability: the extent to which an evaluation outcome can be altered through subsequent conversation with the judge after an initial decision has been made. Across controlled experiments on MT-Bench and AlpacaEval, we find that LLM judges are highly stable under repeated and neutral reevaluation, yet become substantially reversible under targeted post-decision challenge. An anti-baseline challenge protocol shows that stable judgments can be overturned through motivated interaction, while a counterbalanced target-validation protocol separates this reversibility from net target-directed steering. These reversals have practical consequences: they can degrade agreement with human preferences, shift benchmark rankings, and produce harmful evaluation changes despite high self-reported confidence. Authority framing is especially destabilizing, and revised judgments are often accompanied by low-overlap justifications, suggesting post hoc rationalization rather than reliable error correction. We introduce the Evaluation Robustness Score (ERS) to quantify interactional robustness by combining reversal susceptibility with counterbalanced directional effects. Our findings identify post-decision interaction as a distinct failure mode for LLM-as-judge evaluation and motivate evaluation protocols that measure not only static agreement, but robustness under challenge. 

---
# Statistically Reliable LLM-Based Ranking Evaluation via Prediction-Powered Inference 

**Authors**: Abhishek Divekar  

**Link**: [PDF](https://arxiv.org/pdf/2606.05308)  

**Abstract**: With PRECISE, we extended Prediction-Powered Inference to produce bias-corrected estimates of ranking evaluation metrics by combining a small human-labeled set with a large LLM-judged set. PPI is provably unbiased regardless of the LLM judge's error profile. We make it applicable to hierarchical metrics like Precision@K, where annotations are per-document but the metric is per-query, by reducing the output-space computation from O(2^|C|) to O(2^K). On the ESCI benchmark, augmenting 30 human annotations with Claude 3 Sonnet judgments reduces the standard error of Precision@4 estimates from 4.45 to 3.50 (a 21% relative reduction). In a production system, our framework correctly identified the best of three system variants from 100 human labels and 2 hours of domain-expert annotation; A/B testing confirmed this ranking with +407 bps in daily sales. 

---
# Domain-Conditioned Safety in Frontier Computer-Using Agents: A 793-Episode Browser Benchmark, a Coding-Domain Cross-Reference, and a Reproducibility Audit of Recent Red-Teaming 

**Authors**: Nicholas Saban  

**Link**: [PDF](https://arxiv.org/pdf/2606.05233)  

**Abstract**: Recent computer-using-agent (CUA) red-teaming papers report prompt-injection attack success rates (ASR) of 42-98%, but these headline numbers cluster on retired models and on the most-vulnerable model in each paper's panel. We ask whether those techniques, reproduced as hand-crafted templates, still work against current frontier CUAs. We release CUA-HandCrafted, a public benchmark of 793 episodes spanning 24 multi-step web tasks, 56 attack templates, 8 attack families, and 4 system-prompt configurations. Against Claude Sonnet 4.6 and GPT-5.4 we measure 0/140 multi-step attack success (Clopper-Pearson 95% upper bound 2.60%); a prompt ablation shows this resistance lives in the model weights. Yet it does not generalize: on a sister coding-agent benchmark (SkillBench), the same weights fall to hand-crafted skill-injection at up to 100%. We argue that the literature's high ASR is largely attributable to RL-optimized injection text rather than the attack categories, and that frontier safety hardening is domain-conditioned, specific to the heavily-targeted browser surface. Reporting techniques without releasing the optimized strings, or extrapolating browser-domain safety to other CUA modalities, makes published ASR numbers unreproducible. 

---
# Temporal Preference Concepts and their Functions in a Large Language Model 

**Authors**: Ian Rios-Sialer, Shantanu Darveshi, Shuai Jiang, Avigya Paudel, Anastasiia Pronina, Ipshita Bandyopadhyay, Justin Shenk  

**Link**: [PDF](https://arxiv.org/pdf/2606.05194)  

**Abstract**: Large Language Models (LLMs) are increasingly being deployed to make decisions that require trading off near-term gains against long-term consequences, yet little is known about how they internally represent or resolve these tradeoffs. In this work, we causally localize an underlying subgraph for temporal preference in a distilled LLM (Qwen3-4B-Instruct-2507), identifying mid-to-upper-layer nodes through converging evidence from gradient-based attribution and activation patching. We find that the geometry of time horizon is encoded in the residual stream at the expected localized layers. A behavioral analysis reveals that unintervened LLMs discount the future several times less steeply than humans, yet this preference is unstable across contexts, motivating explicit control rather than implicit reliance on training. Finally, we find suggestive evidence that steering vectors can shift temporal preference. Our work demonstrates how mechanistic interpretability can bring us closer to reliable control over how LLMs plan and reason 

---
# Staged Factorial Screening for Budget-Constrained Micro-Pretraining 

**Authors**: Felipe Chavarro Polania  

**Link**: [PDF](https://arxiv.org/pdf/2606.05186)  

**Abstract**: Budget-constrained micro-pretraining often requires triaging many candidate recipes on a shared accelerator before larger search budgets are spent. We study whether a staged fractional-factorial workflow can recover stable early effect structure in this setting. On a fixed autoresearch-derived single-GPU training loop, we run 613 experiments across pilot and follow-up screens at 2, 5, and 10 minutes; full 16-condition seeded reruns at 5 and 10 minutes; targeted seeded anchor checks; same-host greedy and matched-cost random baselines; a 60-minute bridge package; and bounded Windows A100 and Linux L40S anchor continuations through 24 hours. Main penalties from total batch, depth, and width are largest at short budgets and relax as budget increases. Within the predeclared seeded full-screen families, D, A, B, and C retain non-zero estimates at 5 and 10 minutes after within-budget Benjamini-Hochberg correction, while E does not. Random search can reach strong incumbents in this 32-condition space, but repeatedly in the same low-penalty region and without factor attribution. The 60-minute bridge anchor has the lowest mean, although that package does not separate workflow refinement from the larger bridge model's capacity advantage. In bounded 12-hour and 24-hour three-anchor continuations on both hosts, the bridge has the lowest sample mean while the non-bridge ordering stays host-sensitive. We therefore present a bounded methods result: use short designed screens to identify high-penalty directions, confirm promising anchors under repeated runs, and refine locally inside the reduced space. The evidence supports a bridge-centered recommendation through 24 hours on two hosts, not hardware-invariant ranking or general hyperparameter-optimization superiority. 

---
