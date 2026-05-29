# LLMSurgeon: Diagnosing Data Mixture of Large Language Models 

**Authors**: Yaxin Luo, Jiacheng Cui, Xiaohan Zhao, Xinyi Shang, Jiacheng Liu, Xinyue Bi, Zhaoyi Li, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.30348)  

**Abstract**: The pretraining data mixture of Large Language Models (LLMs) constitutes their "digital DNA", shaping model behaviors, capabilities, and failure modes. Yet this composition is rarely disclosed, making post-hoc auditing of data combination or provenance difficult. In this work, we formalize $\textbf{Data Mixture Surgery (DMS)}$: given only generated text from a target LLM, estimate the domain-level distribution of its pretraining corpus under a predefined taxonomy. We propose $\textbf{LLMSurgeon}$, a strong framework that casts DMS as an inverse problem under the label-shift assumption. Rather than directly aggregating classifier outputs, LLMSurgeon estimates a calibrated $\textit{soft}$ confusion matrix and solves a constrained inverse problem to correct systematic domain confusion and recover the latent mixture prior. To evaluate, we introduce $\textbf{LLMScan}$, a recipe-verifiable evaluation suite built from open-source LLMs with transparent pretraining mixtures. Across LLMScan, LLMSurgeon recovers domain mixtures with high fidelity under fixed protocols. Our work presents a practical, post-hoc approach for auditing the digital DNA of foundation models without access to their training data. 

---
# Unlocking the Working Memory of Large Language Models for Latent Reasoning 

**Authors**: Lukas Aichberger, Sepp Hochreiter  

**Link**: [PDF](https://arxiv.org/pdf/2605.30343)  

**Abstract**: To improve the reasoning capabilities of large language models, test-time compute is typically scaled by generating intermediate tokens before the final answer. However, this couples reasoning to autoregressive generation and thereby conflates internal computation with external communication. In contrast, human cognition can use working memory to hold and manipulate information internally without the need to externalize intermediate thoughts. Drawing on this principle, we introduce Reasoning in Memory (RiM), a latent reasoning method that replaces the autoregressive generation of reasoning steps with memory blocks. These memory blocks are fixed sequences of special tokens that unlock the working-memory capacity of large language models. Since they are fixed rather than generated, they can be processed in a single forward pass, enabling compute-efficient latent reasoning. To operationalize these memory blocks, we employ a two-stage curriculum. First, we ground them by predicting explicit reasoning steps after each memory block. Second, we discard this step-level supervision and iteratively refine the final answer after each memory block. Our experiments on reasoning benchmarks show that, across language models of different families and sizes, RiM matches or exceeds existing latent reasoning methods while avoiding the autoregressive generation of thoughts. These results demonstrate that large language models can be trained to use working memory as an effective mechanism for latent reasoning. 

---
# COMPOSE: Composing Future Theorems from Citations and Formal Structure 

**Authors**: David Busbib, Michael Werman  

**Link**: [PDF](https://arxiv.org/pdf/2605.30333)  

**Abstract**: A plausible future mathematical claim must satisfy two constraints: it should follow the direction of prior work and respect the formal dependencies that constrain what can validly follow. Existing approaches typically model only one of these sources, producing claims that are either weakly grounded or insufficiently motivated. We introduce grounded future mathematical generation, where the goal is to generate a plausible future theorem-like claim for an anchor paper using two complementary sources of context: its scientific citation graph and aligned formal theorem dependency graph. To address this setting, we propose COMPOSE, a dual-graph framework that conditions a language model on both scientific citation context and formal theorem structure. To support this setting, we construct a dataset of 108K paired scientific-formal graph examples from arXiv and Mathlib, together with a benchmark of 47K future papers from 2024--2025. Experiments show that COMPOSE outperforms strong baselines on retrieval to real future papers and achieves the best overall performance under LLM-judge evaluation, producing more grounded and mathematically richer outputs. These results show that future mathematical generation benefits from combining scientific context with formal structure. Project page is available at this https URL. 

---
# Resolution Diagnostics for Paired LLM Evaluation 

**Authors**: Anany Kotawala  

**Link**: [PDF](https://arxiv.org/pdf/2605.30315)  

**Abstract**: Across two public LLM leaderboards, many displayed pairwise rankings do not meet a conventional paired-test resolution target under the actual paired evaluation design: 11 of 40 Open LLM Leaderboard v1 pairwise comparisons and 4 of 9 MMLU-Pro top-10 adjacent-rank pairs are unresolved at (alpha, 1-beta) = (0.05, 0.8). The MMLU-Pro count rises to 6/9 under real subject-level clustering and stays at 5-6 out of 9 in 99.9% of category-bootstrap resamples. We frame paired LLM evaluation as a hypothesis-testing problem, invert level-alpha, power-(1-beta) tests, and report a per-pair resolution ratio q = N/N* as the primary diagnostic. A sharp small-effect expansion with an explicit second-order constant shows that the widely-used unpaired Cohen-h-plus-(1-rho) shortcut deviates from the correct N* by approximately a factor of two in the close-comparison regime, a deficit that three of five off-the-shelf calculators(Cohen 1988, G*Power, R pwr) silently inherit when the user post-multiplies their per-arm output by (1-rho). The unresolved-pair pattern remains under multiplicity correction and anytime-valid sequential testing. 

---
# MedCase-Structured: A Text-to-FHIR Dataset for Benchmarking Diagnostic Reasoning in Clinically Realistic EHR Settings 

**Authors**: Valentina Bui Muti, Eugénie Dulout, Ziquan Fu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30295)  

**Abstract**: Large language models (LLMs) show promise for clinical reasoning and decision support, but evaluation in realistic, electronic health record-congruent settings remains limited. Existing benchmarks often rely on static datasets or unstructured inputs that do not reflect the structured, interoperable data formats used in clinical systems. We introduce a pipeline for generating clinically realistic HL7 FHIR R4 bundles from unstructured text, enabling controllable evaluation of clinical decision support systems. The pipeline combines staged LLM generation with terminology-grounded validation and repair to reduce hallucinated codes and enforce structural and semantic consistency. Applying this approach to MedCaseReasoning, we construct MedCase-Structured, a synthetic dataset aligned with clinician-authored diagnostic cases, achieving valid FHIR generation for 82.5% of cases. Evaluation on MedCase-Structured reveals consistently lower diagnostic accuracy for LLMs on structured FHIR inputs than with plain text, highlighting the importance of deployment-aligned benchmarking. 

---
# Loong: A Human-Like Long Document Translation Agent with Observe-and-Act Adaptive Context Selection 

**Authors**: Yutong Wang, Xuebo Liu, Derek F. Wong, Zhilin Li, Rongqing Jiang, Min Zhang, Shimin Tao, Daimeng Wei, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30274)  

**Abstract**: Document-level translation remains one of the most challenging tasks for large language models, which are constrained by limited context windows that impede global cohesion, while simultaneously suffering from redundant contextual information that degrades translation quality. To address this, we propose a human-like long document translation agent called Loong, which leverages a 3E memory module (Essence-Exemplar-Entity) to store summaries, sentence pairs, and entity records as historical context. Instead of passively attending to all history, Loong performs deep reasoning to adaptively identify the optimal context for translation guidance. Loong optimizes its context policy through reinforcement learning, utilizing preference data derived from its own sampled observe-and-act reasoning trajectories. Empirical evaluations demonstrate that Loong achieves substantial translation quality improvements in English $\Leftrightarrow$ Chinese, German, and French directions, with average gains of up to 13.0 points across the three evaluation metrics. Furthermore, Loong exhibits strong generalization across domains and robustness against contextual noise, while maintaining remarkable stability in ultra-long document translation. Our code is released at this https URL. 

---
# How LoRA Remembers? A Parametric Memory Law for LLM Finetuning 

**Authors**: Ziwen Xu, Haiwen Hong, Linsong Yu, Benglei Cui, Longtao Huang, Hui Xue, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30260)  

**Abstract**: Large Language Models (LLMs) must continuously learn and update knowledge to remain effective in dynamic real-world environments. While Low-Rank Adaptation (LoRA) is widely used for such memory updates, existing studies mainly rely on qualitative downstream evaluations, leaving the quantitative capacity limits and underlying dynamics of exact parametric memory largely unexplored. To bridge this gap, we employ LoRA as a controlled memory capacity probe within the latent space to systematically quantify exact parametric memory. We introduce the Parametric Memory Law, a robust power law linking loss reduction Delta L to effective parameters and sequence length. At the token level, fine-grained analysis reveals a deterministic phase transition, demonstrating that a prediction probability of p > 0.5 constitutes a sufficient condition for verbatim recall under greedy decoding. Driven by these insights, we introduce MemFT, a threshold-guided optimization strategy that dynamically redistributes the training budget toward sub-threshold tokens. Empirical evaluations demonstrate that MemFT can enhance memory fidelity and efficiency. Code will be released at this https URL. 

---
# Same Evidence, Different Answers: Canonical-Context On-Policy Distillation for Multi-Turn Language Models 

**Authors**: Zizhuo Lin, Quanling Liu, Jinsheng Quan, Chao Zhang, Yifan Zhu, Xing Shi, Jingtao Xu, Zhihui Li, Yawei Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.30251)  

**Abstract**: Large language models (LLMs) often solve a task when all instructions are given in a single prompt, but fail when the same information is revealed gradually across turns. When a clean FULL prompt and a RAW-SHARDED conversation contain the same complete user evidence, the model should still arrive at the same answer. We argue that a key reason for this gap is self-anchored drift: responses produced under partial information introduce unsupported assumptions, and those assumptions later distort the final answer. To reduce this effect, we propose Canonical-Context On-Policy Distillation (CCOPD). During training, the same base model is used in two roles: a frozen teacher conditioned on the clean FULL prompt and a trainable student that receives the same evidence incrementally through a multi-turn conversation; CCOPD aligns the student's behavior on its own trajectories with the teacher's canonical full-context behavior. Trained only on math problem conversations, CCOPD yields a 32\% average relative improvement in RAW-SHARDED performance over the original base model across math and five zero-shot out-of-domain task families, while largely preserving full-context performance. Further analyses suggest that CCOPD strengthens grounding in user evidence and reduces sensitivity to contamination from earlier assistant turns. 

---
# Knowing What to Solve Before How: Preplan Empowered LLM Mathematical Reasoning 

**Authors**: Shaojie Wang, Liang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30245)  

**Abstract**: Current plan-based reasoning methods improve large language models (LLMs) by inserting a planning stage before execution, giving rise to the question $\rightarrow$ plan $\rightarrow$ cot paradigm. While effective, a closer examination reveals an inherent paradigm-level gap: both the planning and its execution stages decide how to solve a problem, while the prior question of what to solve; recognizing the problem type, the applicable tools, and the foreseeable pitfalls; remains entirely implicit. To bridge this gap, we propose PPC (Preplan-Plan-CoT), a framework that introduces an explicit problem-understanding stage, the preplan, yielding a new question $\rightarrow$ preplan $\rightarrow$ plan $\rightarrow$ cot paradigm. Realizing this paradigm requires safeguarding the conceptual integrity of preplan at both ends. Specifically, we design a three-stage synthesis pipeline with a spoiler-score detector that filters out leakage and spoiler failures to build clean preplan supervision, and a composite GRPO reward enforces that the generated plan genuinely follows from the preplan. Experiments across four backbones and five mathematical reasoning benchmarks show that PPC achieves the best results on 39 of 40 metrics, improving maj@16 and pass@16 by +2.23 and +3.06 over the strongest baseline without introducing additional inference token overhead. 

---
# CommunityFact: A Dynamic, Multilingual, Multi-domain Benchmark for Misinformation Detection in the Wild 

**Authors**: Sahajpreet Singh, Insyirah Mujtahid, Min-Yen Kan, Kokil Jaidka  

**Link**: [PDF](https://arxiv.org/pdf/2605.30241)  

**Abstract**: Misinformation verification increasingly occurs in public, fast-moving, and multilingual online settings, where static benchmarks provide an incomplete measure of model reliability. We introduce CommunityFact, a refreshable benchmark for misinformation detection in the wild, with three major goals: coverage, granularity, and redistributability. This release contains 15,992 standalone claims across five languages and two domains. We evaluate ten LLMs under varying inference-time capabilities, including thinking and web-search. Our results show that closed-input verification remains challenging, web access yields the largest gains, and web-enabled LLMs' source-selection policies are systematically misaligned with the sources human Community Notes raters converge on -- a gap that closes through model-specific mechanisms of retrieval expansion or pruning. We further find substantial variation across language-domain slices and across the evidence ecosystems used by web-enabled systems. Beyond evaluation, CommunityFact positions Community Notes as a training signal for claim-conditioned source suggesters that could improve factual verification on novel claims. 

---
# Do Language Models Track Entities Across State Changes? 

**Authors**: Zilu Tang, Qiao Zhao, Gabriel Franco, Derry Wijaya, Aaron Mueller, Sebastian Schuster, Najoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.30233)  

**Abstract**: Entity tracking (ET), the ability to keep track of states, is a fundamental skill that underlies complex reasoning. An increasing amount of work investigates how transformer language models (LMs) solve entity binding $\textit{without}$ state changes. However, there is limited understanding of how non-toy LMs address ET problems of realistic difficulties expressed in natural language. To this end, we investigate the mechanisms underlying ET in more complex scenarios featuring multiple state-changing operations. We find that LMs do not incrementally track world states across tokens or query-relevant states across layers, but simply aggregate relevant information in parallel at the last token when the query becomes evident. We further investigate mechanisms of individual operations ($\texttt{PUT}$, $\texttt{REMOVE}$, $\texttt{MOVE}$) to characterize this non-incremental ET mechanism. Surprisingly, LMs implement the $\texttt{REMOVE}$ operation with a fragile global suppression tag; this global removal mechanism predicts various failure modes that we confirm behaviorally. We provide a mechanistic solution of nullifying this tag to partially address this issue. Overall, our findings reveal that LMs solve a fundamentally sequential task using a non-sequential strategy. More broadly, our work illustrates how behavioral and mechanistic analyses can fruitfully interact. Behavioral results inform mechanistic hypotheses, and insights from mechanistic analyses help build stronger behavioral evaluations by predicting failure modes missing from existing evaluations. 

---
# GRUFF: LLM Pronoun Fidelity, Reasoning, and Biases in German 

**Authors**: Fabian Mewes, Anne Lauscher, Vagrant Gautam  

**Link**: [PDF](https://arxiv.org/pdf/2605.30214)  

**Abstract**: Third-person singular pronouns have long been used to study stereotypical biases in language models and to test their abilities to reason about reference. More recently, the interplay between reasoning and bias has been investigated with the task of pronoun fidelity, which assesses models' abilities to correctly reuse a previously-specified pronoun for a discourse entity, independent of other potentially distracting discourse entities mentioned in between. However, such research focuses on English, which is a language with limited grammatical gender and almost no gender agreement. In this paper we contribute a novel, large-scale dataset, GRUFF, to measure pronoun fidelity in German, covering four different gender agreement systems in nouns, and four sets of pronouns. With this dataset, we show that LLMs show strong grammatical agreement for masculine and feminine entities in the absence of explicit context, but not for neopronouns xier and en. Models are generally not robust to distractors, but encoder-only models are more robust in German than in English, reflecting the importance of grammatical gender. Finally, we show that occupational stereotypes in this context are poorly correlated across grammatical cases, and across most models, except ones with closely related architectures. We release all code and data to encourage further work on gender-inclusive language and referential reasoning in German. 

---
# A Dual-Path Architecture for Scaling Compute and Capacity in LLMs 

**Authors**: Markus Frey, Behzad Shomali, Joachim Koehler, Mehdi Ali  

**Link**: [PDF](https://arxiv.org/pdf/2605.30202)  

**Abstract**: Looped transformers apply a shared block multiple times and have emerged as a parameter-efficient route to scaling compute in language models. However, at fixed FLOPs a looped model has strictly less capacity than a baseline transformer. We propose a novel dual-path block that can flexibly scale compute, the number of sequential operations applied to a hidden state, and capacity, the parameters available at a single step. For this we expose both axes as parallel pathways within a single layer: a deep sublayer re-applied K times with shared parameters, and a wide sublayer with an enlarged feed-forward network applied once. Independent per-token gates combine both axes and allow detailed per-token routing analyses. We show that across two FLOP budgets, our dual-path model surpasses iso-FLOP matched models on language modeling and downstream evaluations, while using fewer parameters than the baseline at matched FLOPs. The learned gates are directly interpretable and show systematic per-token allocation with function words and lexical content trend wide, while punctuation, symbols, and arithmetic tokens trend deep. 

---
# Do Proactive Agents Really Need an LLM to Decide When to Wake and What to Anchor? 

**Authors**: Xiaoze Liu, Ruowang Zhang, Amir H. Abdi, Michel Galley, Zhikai Chen, Siheng Xiong, Xiaoqian Wang, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.30152)  

**Abstract**: Proactive agents read user activity as text and call an LLM on every event to decide whether to act. But user activity is not natively text: it is a structured event stream of (actor, verb, object, timestamp) tuples that the operating system already maintains in graph form. Rendering the structure as text and asking an LLM to recover it is a round-trip the system never had to take. We treat the always-on signal as graph updates rather than text and use a small temporal-graph-learning (TGL) model as the encoder: one forward pass yields a per-event trigger probability and a per-entity routing score, and only the downstream agent (turning a small structured handoff into a fluent user-facing sentence) is an LLM call, invoked only when the trigger fires. TGL improves F1 on each of 14 backbones (mean +16.7, up to +46.0); in trigger-architecture comparisons, one TGL checkpoint gives the strongest trigger AUCs and the most stable deployed threshold. It runs at 11.13 ms per event on a GPU server and 13.99 ms on a consumer laptop, approximately 4--7x and 12--83x faster than every single-forward LLM-as-trigger configuration tested in each regime, with an approximately 220 MiB BF16 resident footprint deployable on-device alongside the privacy-sensitive activity stream it consumes. 

---
# CorPipe at CRAC 2026: Empty Nodes and Cross-Lingual Transfer in Multilingual Coreference Resolution 

**Authors**: Milan Straka  

**Link**: [PDF](https://arxiv.org/pdf/2605.30133)  

**Abstract**: We introduce CorPipe 26, our winning submission to the CRAC 2026 Shared Task on Multilingual Coreference Resolution. The fifth edition of this shared task focuses mainly on the comparison of generative LLMs and specialized systems; additionally, 5 more datasets and 2 new languages are introduced. CorPipe 26 is an improved version of CorPipe 25, with a new variant predicting empty nodes together with mentions and coreference links in a single model. Our system outperforms all other submissions in the LLM track by 2.8 percent points and all submissions in the unconstrained track by 9.5 percent points. Furthermore, we perform a series of ablation experiments with different model sizes, empty node prediction methods, and cross-lingual zero-shot evaluation. The source code and the trained models are publicly available at this https URL. 

---
# CCS: Clinical Consensus Selection for Radiology Report Generation 

**Authors**: Xi Zhang, Yingshu Li, Zaiqiao Meng, Jake Lever, Edmond S. L. Ho  

**Link**: [PDF](https://arxiv.org/pdf/2605.30131)  

**Abstract**: Radiology report generation (RRG) is commonly formulated as a single-path generation task, where a multimodal large language model (MLLM) produces one decoded report as the final output. While recent progress has largely been driven by scaling training data, model capacity, and retrieval mechanisms, improving report quality at inference time remains underexplored. In this work, we observe that fixed radiology MLLMs often generate clinically stronger reports elsewhere in their candidate pool than the one selected by default decoding, suggesting that inference-time decision making remains an overlooked bottleneck. To address this, we propose Clinical Consensus Selection (CCS), a decoder-agnostic inference-time selection framework that samples multiple candidate reports and selects the one with the highest clinical consensus across the rollout pool. CCS unifies text-based utilities with a radiology-adapted utility computed by an image--report-trained multimodal embedder, which measures candidate agreement beyond surface-level textual similarity. Across three datasets and multiple radiology MLLMs, CCS consistently improves inference-time performance over single-path decoding and generic Best-of-N baselines, with particularly clear gains on clinical metrics. Further analysis shows that image-grounded utility forms a selection axis distinct from textual consensus and that substantial headroom remains for improving RRG at inference time. 

---
# Dial HEALTHDIAL for Advice: A Multilingual and Multi-Parallel Spoken Dialogue Dataset for Knowledge-Grounded Information Seeking 

**Authors**: Songbo Hu, Yinhong Liu, Ej Zhou, Evgeniia Razumovskaia, Xiaobin Wang, Alexander Fraser, Ivan Vulić, Anna Korhonen  

**Link**: [PDF](https://arxiv.org/pdf/2605.30107)  

**Abstract**: Creating spoken dialogue datasets is methodologically challenging, and these challenges are amplified when the goal is to build multilingual, multi-parallel datasets at scale. This work introduces HEALTHDIAL, a large-scale, multilingual, and multi-parallel dataset for developing and evaluating retrieval-augmented generation (RAG)-based spoken dialogue systems. The dataset comprises 6,000 information-seeking dialogues (1,500 per language) grounded in trusted content from the World Health Organization (WHO) and 163 hours of user speech recorded from native speakers of diverse dialects across four official WHO languages: Arabic, Chinese, English, and Spanish. Each speaker is annotated with demographic (e.g., gender, age) and sociolinguistic (e.g., primary language, region of origin) variables. We report benchmark results across key dialogue tasks, which reveal consistent performance disparities across languages, even among high-resource ones. To support future research, we release the dataset, a prototype system, and a toolkit for data collection and system evaluation. 

---
# SEAL: Can Saturated Benchmarks Be Revived by LLM-as-a-Meta-Judge? 

**Authors**: Jiamin Chen, Yidi Wu, Qiexiang Wang, Qianben Chen, Yuchen Li, Yansen Zhang, Xiaokun Zhang, Wangchunshu Zhou, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.30104)  

**Abstract**: Widely used language-model benchmarks are increasingly saturated, with frontier systems often receiving near-tied scores that standard metrics cannot resolve. Rather than constructing harder alternatives, we ask whether existing tasks can be made informative again through improved evaluation over the same candidate outputs. Therefore, we present Seeded Elimination with Adaptive LLM-as-a-Meta-Judge, a self-improving evaluation protocol for extracting latent ranking signal from saturated benchmarks. SEAL seeds candidate outputs into a single elimination and evaluates each match with task-level principles plus self-improving checklist criteria. We evaluate SEAL on multiple saturated benchmarks covering code generation, mathematical reasoning, knowledge-intensive question answering, and tool-use agent task completion. Across these settings, SEAL improves the ranking-accuracy--latency trade-off over competing protocols, attaining 0.83--1.00 Spearman agreement with full pairwise judging and 4/4 top-1 agreement, while requiring only 11.89 calls per task compared with 28.00 for full pairwise evaluation. 

---
# DirectorBench: Diagnosing Long-Form Video Generation with Personalized Multi-Agent Evaluation 

**Authors**: Jiamin Chen, Qianben Chen, Jiawen Zhang, Yidi Wu, Yuchen Li, Xiaokun Zhang, Wangchunshu Zhou, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.30090)  

**Abstract**: Long-form video generation is rapidly moving from short, single-scene synthesis toward minute-long, multi-shot creation with narrative structure, cinematic control, audio, and cross-modal synchronization. However, evaluating such videos remains challenging, since existing benchmarks largely focus on local visual quality, short-horizon temporal consistency, or generic prompt alignment, and provide limited diagnosis of workflow failures and user-dependent preferences. We introduce DirectorBench, a personalized multi-agent diagnostic benchmark for long-form video generation. DirectorBench evaluates generated videos with respect to 80 structured metadata entries, 7 user profiles, and 40 checkpoint criteria across 5 dimensions: script, visual, audio, cross-modal, and stability. Instead of reducing quality to a single aggregate score, DirectorBench localizes checkpoint-level bottlenecks and supports profile-aware evaluation. We evaluate 4 long-form video generation workflows, 6 base LLMs, and 7 user profiles. Across workflows, DirectorBench reveals a between-unit bottleneck: transition quality averages only 0.256 and reaches 0.356 for the best workflow, while prompt-level user demand fulfillment averages 0.71. We further conduct human evaluation with 14 annotators to validate the alignment between DirectorBench and human judgment. The results show that DirectorBench captures human-perceptible quality differences and reveals workflow- and profile-dependent failure modes that are hidden by aggregate scoring. These findings highlight the importance of diagnostic and profile-aware benchmarking for long-form video generation. 

---
# Adaptive Targeted Dynamic Chunking for Tokenization-Free Hierarchical Model 

**Authors**: Thang Dang, Akira Nakagawa, Kenichi Kobayashi, Koichi Shirahata  

**Link**: [PDF](https://arxiv.org/pdf/2605.30080)  

**Abstract**: Tokenization-free hierarchical models are emerging as a promising alternative to traditional Large Language Models (LLMs), addressing inherent preprocessing issues such as vocabulary design complexity, out-of-vocabulary (OOV) errors, and language-specific constraints. However, a significant challenge in these byte-level methods is the optimization of the compression ratio, a critical factor that dictates model performance for processing bytes data via chunks. In this paper, we propose Adaptive Targeted Dynamic Chunking (ATDC), a novel byte-compression control mechanism designed to enhance the effectiveness of dynamic chunking within hierarchical architectures. Our approach utilizes curriculum learning to progressively adjust the compression ratio during training, transitioning from low to high compression to stabilize the learning process. We provide an analysis establishing the relationship between the target compression ratio and Bytes-Per-Innermost-Chunk (BPIC), allowing for tracking of chunk-size evolution throughout the training phase. Evaluations conducted on the FineWeb-Edu 100B dataset demonstrate that hierarchical models equipped with ATDC achieve competitive Bits-Per-Byte (BPB) performance compared to conventional baselines operating at both byte and token levels. Furthermore, the proposed method exhibits more stable training dynamics and superior final performance across diverse downstream tasks compared to models using fixed compression ratios, while maintaining the inherent robustness and flexibility of byte-level processing. 

---
# UniSteer: Text-Guided Flow Matching in Activation Space for Versatile LLM Steering 

**Authors**: Yingdong Shi, Ruiming Zhang, Changming Li, Zhiyu Yang, Kaixing Zhang, Jingyi Yu, Kan Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.30076)  

**Abstract**: Activation-based control steers large language models (LLMs) by intervening on their internal representations during inference, and has emerged as an effective paradigm for controlling behaviors such as persona and style. However, existing methods often rely on fixed steering directions or task-specific intervention modules, making them difficult to adapt to fine-grained concepts and compositional constraints. We propose UniSteer, a text-guided activation flow matching model that learns a conditional distribution over residual-stream activations from natural-language conditions. Instead of fitting a separate intervention for each target behavior, UniSteer learns a universal conditional velocity field in activation space. At inference time, UniSteer performs flow inversion by partially transporting a source activation toward a latent state and regenerating it under a target textual condition before injecting it back into the frozen LLM. The same conditional model supports activation-space classification by selecting the textual label with the lowest reconstruction energy. Experiments on three target LLMs show that UniSteer provides a unified interface across behavioral control, truthfulness steering, fine-grained concept steering, multi-constraint instruction following, and activation-space classification. 

---
# HEART-Bench: Do LLM Agents Exhibit Human-like Psychology? 

**Authors**: Weihan Peng, Chenxu Zhang, Qianao Wang, Yuling Shi, Heng Lian, Qihong Mao, Jiahao Pang, Chunliang Feng, Bowen Li, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30058)  

**Abstract**: While LLM agents have demonstrated remarkable task-oriented abilities such as planning, reasoning, and action, few works have treated them as complete human personalities where emotional dimensions hold equal importance. In this paper, we introduce a novel benchmark to systematically assess whether LLM agents can simulate coherent, human-like psychology. Specifically, our benchmark constructs 11 diverse human characters grounded in orthogonal Big Five personality traits, with each profile deeply integrated with 1,000 structured autobiographical-style episodic memories distributed across theory-grounded developmental life stages. To rigorously evaluate the psychological manifestations of LLMs, we designed a curated suite of 64 decision-making scenarios, guided by the DIAMONDS taxonomy, a psychological framework that characterizes situations along eight dimensions: Duty, Intellect, Adversity, Mating, pOsitivity, Negativity, Deception, and Sociality. By subjecting agents to varying scenarios, the benchmark evaluates whether they can consolidate their innate personality traits and autobiographical memories to make behavioral decisions that are consistent with their specific psychological profiles. After systematic human validation and filtering, we obtained a benchmark consisting of 673 multiple-choice questions (MCQs). We believe this benchmark provides a principled and scalable testbed for studying human-like emotions, personality consistency, and value-consistent behavioural decision-making in LLM-based agents. 

---
# Who Am I? History-Aware Profiles for Student Simulation in Tutoring Dialogues 

**Authors**: Zhangqi Duan, Shuyan Huang, Alexander Scarlatos, Jaewook Lee, Simon Woodhead, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2605.30051)  

**Abstract**: A key part of developing large language model (LLM)-powered, automated tutoring tools is student simulation, i.e., using LLMs to role-play as students, which can facilitate tutor model evaluation and training. Existing work mostly focuses on within-dialogue simulation, which lacks context on student knowledge and behavior, partly due to not grounding in past student question-answering or dialogue interactions. In this work, we introduce the task of history-conditioned student simulation, where the goal is to accurately predict student dialogue turns by leveraging information in the student's learning history. We propose a two-component framework in which a profile generator summarizes a student's history and a simulator predicts student turns conditioned on the resulting profile. We train both components with reinforcement learning (RL), yielding profiles optimized for faithful student simulation. We evaluate our method and baselines on the first-of-its-kind real-world dataset of student dialogues and question responses that we collect from a math learning platform. Extensive experiments show that our method significantly outperforms baselines, and demonstrate the importance of history, profiles, and RL training. 

---
# Give it Space! Explicit Disentangling of Positional and Semantic Representations in Encoders 

**Authors**: Pierre-Antoine Lequeu, Camille Barboule, Benjamin Piwowarski  

**Link**: [PDF](https://arxiv.org/pdf/2605.30022)  

**Abstract**: Positional encoding (PE) underpins how permutation-invariant Transformers represent sequence order, yet how positional information is processed and stored remains poorly understood. Modern PE methods such as RoPE still struggle on tasks such as long-context understanding or retrieval \cite{chen-etal-2025-hope}. Hence, a better understanding of the internal positional mechanism could help design better PE. Building on evidence that positional and semantic signals occupy nearly orthogonal subspaces in trained Transformers, we modify an encoder Transformer to process three explicitly disentangled streams: semantic, absolute positional (AP) and relative positional (RP), and confine the masked-language-modeling (MLM) objective to the semantic stream. This decoupling enables a clean mechanistic study and yields three take-aways. (1) The isolated AP subspace spontaneously collapses into a low-frequency two-dimensional manifold that captures the structure of the document; (2) Attention heads specialize into structure and semantic-oriented groups, with RP exclusively supporting the latter; (3) Standard positional encodings do not robustly retain macroscopic structure: RoPE and RP only weakly encode it, and entangled AP loses it in the final layers under MLM pressure. The disentangled approach preserves positional encoding, which improves linguistic representation on 49 of the 65 linguistic phenomena of the Flash-Holmes probing benchmark. 

---
# Recovering Diversity Without Losing Alignment: A DPO Recipe for Post-Trained LLMs 

**Authors**: Vinay Samuel, Yapei Chang, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2605.30021)  

**Abstract**: Many open-ended instructions have multiple valid answers that users can benefit from seeing, but post-training often narrows an LLM's output space toward a small set of canonical responses. We introduce REDIPO, an offline DPO data-construction pipeline for recovering distinct valid answer modes while preserving the alignment benefits of the instruct model. For each prompt, REDIPO samples responses from both base and instruct models, rewrites base-model responses with the instruct model, filters candidates for safety and instruction-following quality, and builds preference pairs that favor marginally diverse responses among candidates with similar instruction-following reward. Across Qwen3-4B, OLMo-3-7B, and LLaMA-3.1-8B, REDIPO improves NoveltyBench distinct_k by 134%, 33%, and 44% relative to the instruct checkpoints, while DivPO changes diversity by 0%, -6%, and -4% on the same models. These gains largely maintain MTBench, IFEval, and Arena-Hard performance, and reduce direct-category HarmBench attack success rate. Ablations show that marginal-diversity pair selection and base-response rewriting drive the diversity gains, while filtering and quality-bounded pairing help maintain alignment. Overall, our results show that diverse valid answers from base-model generations can be reintroduced through carefully constructed preference data while retaining the alignment benefits of post-training. We release our code and data at this https URL. 

---
# Latent Performance Profiling of Large Language Models 

**Authors**: Tanmoy Chakraborty, Ayan Sengupta, Suparna Bhattacharya, Partha Pratim Chakrabarti, Amlan Chakrabarti, Supratik Chakraborty, Partha Pratim Das, Lipika Dey, Richa Singh, Mayank Vatsa  

**Link**: [PDF](https://arxiv.org/pdf/2605.30018)  

**Abstract**: Large language models (LLMs) frequently achieve impressive scores on standardized benchmarks, yet accuracy alone offers a limited view of their capabilities. Evaluating open-source LLMs through leaderboards faces persistent issues like data contamination, narrow task scope, and weak alignment with real-world reliability. Benchmark-based evaluations such as MMLU PRO, BBH, or IFEval primarily capture \textit{what} a model outputs on fixed test sets, not \textit{how} it processes information, calibrates uncertainty, or structures internal knowledge. In this article, we advocate for a shift from benchmark-centric evaluation toward a complementary, \textit{state-centered intrinsic assessment} of LLMs. To this end, we introduce \textbf{Latent Performance Profiling (LPP)} -- a framework that derives task-agnostic diagnostics from hidden activations and output distributions. LPP defines a set of scalar metrics on a model's latent representations and dynamics, revealing scale-independent traits that enable interpretable comparisons and uncover hidden vulnerabilities. Unlike static accuracy scores, LPP provides stable, architecture-sensitive signatures across models of similar size. With extensive empirical analyses across eight LLMs, spanning a size range of 0.5B-14B, we demonstrate that models with similar benchmark scores can exhibit contrasting latent profiles, such as differences in entropy or adaptability. Guided by these insights, we design synthetic probes for uncertainty and symbolic reasoning that align with intrinsic metrics while decoupling from leaderboard bias. We recommend that reporting LPP alongside benchmarks provides a deeper, interpretable understanding of model behavior, enabling more reliable model selection, safety assessment, and evaluation beyond surface-level accuracy. 

---
# Adapting Multilingual Embedding Models to Turkish via Cross-Lingual Tokenizer Surgery and Offline Distillation 

**Authors**: M. Ali Bayram, Banu Diri, Savaş Yıldırım  

**Link**: [PDF](https://arxiv.org/pdf/2605.29992)  

**Abstract**: Sentence embeddings are a foundational component for semantic search, clustering, classification, and retrieval-augmented generation. This paper presents embeddingmagibu-200m, a Turkish-focused sentence embedding model that produces 768-dimensional L2-normalized vectors and supports an 8,192-token context window, far exceeding the 512-token limit of earlier BERT-based Turkish encoders. Instead of full pretraining, an efficient three-stage adaptation pipeline is introduced: (1) construct a Turkish-optimized multilingual tokenizer with a 131,072 vocabulary by pruning redundant tokens from the teacher's vocabulary and incorporating multilingual tokens via frequency analysis on a 40-language corpus, (2) clone a teacher embedding model while preserving transformer backbone weights and initializing a compatible embedding table for the new vocabulary via mean-composition token mapping, and (3) perform offline embedding distillation from precomputed teacher vectors using a cosine similarity objective over a balanced 40-language Wikipedia corpus. The resulting student model contains approximately 200M parameters and trains in roughly four hours on a single GPU by avoiding online teacher inference during training, at a total cost of $5-$20. Empirically, Pearson/Spearman correlations of 77.55%/77.45% are obtained on STSbTR, surpassing the 300M-parameter teacher model (73.84%/72.92%). On TR-MTEB (26 tasks), a mean score of 63.9% is achieved (7th out of 26 models), providing a competitive cost-quality trade-off with 33% fewer parameters than the teacher. To facilitate reproducibility and downstream use, all artifacts are released including model weights, tokenizer files, precomputed embedding datasets, and open-source cloning and distillation tooling. 

---
# Causal Interventions on Continuous Variables: A Case Study on Verb Bias in Steering Vectors for In-Context Learning 

**Authors**: Zhenghao Herbert Zhou, R. Thomas McCoy, Robert Frank  

**Link**: [PDF](https://arxiv.org/pdf/2605.29971)  

**Abstract**: Causal interventions in language model representations have largely targeted discrete features, like grammatical number. However, language models must also make use of features that are graded. We introduce a method for causal intervention on continuous variables: given activation vectors paired with a graded target variable, we localize a low-dimensional direction for that variable and use this direction to edit a vectors toward counterfactual target values. We apply this method to a continuous feature that is well-studied in psycholinguistics, namely verb bias (which reflects which syntactic structures tend to follow a given verb). We show that verb bias is causally represented in steering vectors extracted from large language models: counterfactual edits to verb bias systematically shift downstream structural preferences. Verb bias has also previously been linked to in-context learning; in further analyses, we find that steering vectors encode error signals that could drive the error-driven update behavior seen in in-context learning but that these aspects of the steering vectors are not causally used in downstream production. Overall, these results show causal interventions can be applied to continuous variables, though connecting continuous variables to in-context learning remains a challenge. 

---
# Does The Way You Plan Matter? An Empirical Study of Planning Representations for LLM Web Agents 

**Authors**: Alejandra Zambrano, Sara Vera Marjanovic, Imene Kerboua, Xing Han Lù, Leila Kosseim  

**Link**: [PDF](https://arxiv.org/pdf/2605.29927)  

**Abstract**: Despite recent advances, LLM-based web agents still struggle with limited exploration, omission of critical steps, and sensitivity to task constraints. Prior work suggests that many of these failures stem from weaknesses in planning, yet the impact of alternative natural language plan representation remains unexplored. To address this, we introduce PlanAhead, a static planner-executor framework that evaluates the impact of plan representation in agent performance. We first automatically categorize WebArena tasks into 3 difficulty levels, enabling consistent difficulty grading without human annotation. Then we systematically evaluate 4 different plan representations on the tasks categorized as hard: sequential subgoals, narrative, pseudocode, and checklist; across different families of multimodal LLM powered agents (OpenAI, Alibaba, and Google). To account for stochastic variability, we introduce two novel evaluation metrics: Achievement Rate (AR) and Solved-Task Consistency (STC). Our results show that both, the plan formulation and the underlying LLM generating the plan, significantly influence web-agent robustness and task success. 

---
# ExCAM: Explainable Cultural Awareness Metrics 

**Authors**: Christoph Leiter, Haiyue Song, Hour Kaing, Jin Tei, Hideki Tanaka, Masao Utiyama, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2605.29897)  

**Abstract**: Evaluating the cultural awareness of large language models is crucial to ensure the fairness of generated text and the generalizability of applications across the world. Recent benchmarks explore cultural goods like food or values like behavior in stressful situations through the lens of question answering or text generation tasks. However, creating these benchmarks requires time-intensive and costly human annotations. Also, benchmarks that evaluate cultural awareness in free text are scarce and often rely on dated evaluation mechanisms. To address this gap, we introduce ExCAM, an Explainable Cultural Awareness Metric, which is, to our knowledge, the first dedicated evaluation metric that identifies, rates and explains cultural errors in instruction-output pairs. To train and evaluate ExCAM, we introduce ExCAM40k, a dataset comprised of nine existing benchmarks that we reformat and enhance with synthetic errors. Compared to several baselines, including GPT-5, ExCAM achieves the highest error detection rate with up to 80% accuracy on a balanced test set. Therefore, ExCAM opens the pathway towards fine-grained and explainable cultural evaluation of free text. 

---
# Internal Representation, Not Clinical Knowledge: Where Apparent LLM Triage Failures Originate 

**Authors**: David Fraile Navarro, Berardino Como, Jialei Sheng, Soundariya Ananthan, Shlomo Berkovsky  

**Link**: [PDF](https://arxiv.org/pdf/2605.29889)  

**Abstract**: Patient-voiced clinical-triage benchmarks report high under-triage rates for consumer LLMs for constrained multiple-choice output, yet the same cases score differently with free-text. We ask whether output format changes the model's \emph{clinical representation} or only the mapping from a preserved representation to an answer. Using sparse-autoencoder (SAE) features in Gemma 3 4B/12B IT and Qwen3-8B, we find the same medical features fire on the shared clinical narrative under both formats but go {silent} at the multiple-choice decision token in all the cases at every model. Three independent methods (natural-language autoencoder verbalization, decision-token logit attribution, and top-feature characterization) agree that scaffold and format features, but not medical features, drive the decision logits. Behaviorally, the multiple-choice penalty inverts under both structured and natural-language input, option-order shuffle rules out positional bias, and the gap is dominated by off-by-one decision (the model picks an adjacent acuity letter to the gold answer) rather than knowledge failure. Thus, the failure originates in the output format and not in the clinical representation. 

---
# CRITIC-R1: Learning Structured Critics for Retrieval-Augmented Generation 

**Authors**: Wenhan Xiao, Ziwei Zhang, Chuanyue Yu, Xingcheng Fu, Qingyun Sun, Runhua Xu, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.29886)  

**Abstract**: Retrieval-augmented generation (RAG) improves knowledge-intensive question answering by incorporating external evidence. However, existing RAG methods still suffer from hallucinations and subtle reasoning errors. Recent studies introduce external critics to refine RAG outputs, yet they often provide coarse-grained and weakly structured feedback, exhibit over-aggressive intervention, and lead to noisy and unreliable refinement, limiting their effectiveness for correction. To tackle these issues, we propose CRITIC-R1, a structured critic framework that formulates and learns RAG critique as an explicit error diagnosis problem using reinforcement learning (RL). Our framework categorizes common RAG errors into multiple diagnostic dimensions, including verdict, error location, reasoning analysis, and fix generation. To learn these capabilities, we design two reward functions: Conservative Judgement Alignment (CJA) first encourages calibrated high-level judgements while mitigating the over-aggressive phenomenon, whereas Diagnostic Quality Alignment (DQA) further improves fine-grained diagnostic feedback through gated rewards. We train the critic model using GRPO-based RL with process-level supervision collected from external LLM teacher models. Experiments across five QA benchmarks show that CRITIC-R1 consistently improves answer quality over strong RAG baselines.
Our source code is available at this https URL 

---
# Towards Verifiable Multimodal Deep Research: A Multi-Agent Harness for Interleaved Report Generation 

**Authors**: Chenghao Zhang, Guanting Dong, Yufan Liu, Tong Zhao, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2605.29861)  

**Abstract**: Large Language Models (LLMs) have advanced autonomous agents from deep search, which retrieves concise factual answers, to deep research, which synthesizes scattered evidence into long-form reports. However, verifiable multimodal deep research remains challenging due to open-ended synthesis without deterministic ground truth and the need to interleave textual arguments with visual evidence. We propose \textsc{Ptah}, a multi-agent harness for interleaved report generation. \textsc{Ptah} orchestrates the lifecycle from user query to rendered web report through planning, research, and writing stages, where specialized agents construct visual-aware plans, collect claim-grounded evidence, maintain source-aligned images in a \textit{Visual Working Memory}, and compose reports through declarative multimodal tool use. A verifier agent serves as the harness's acceptance function, enforcing factual grounding, citation fidelity, and cross-modal consistency throughout the workflow. We further introduce \textsc{Ptah}Eval, an evaluation protocol that augments existing benchmarks with image-level and presentation-level assessments. Experiments on deep research benchmarks show that \textsc{Ptah} produces more reliable, visually informative, and usable human-facing multimodal reports than strong baselines. 

---
# EvoRubric: Self-Evolving Rubric-Driven RL for Open-Ended Generation 

**Authors**: Xin Guan, Xiaomeng Hu, Shen Huang, Zhenyi Wang, Bo Zhang, Zijian Li, Pengjun Xie, Bo Liu, Jiuxin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2605.29847)  

**Abstract**: Reinforcement Learning (RL) has significantly advanced Large Language Models (LLMs) in verifiable domains, but aligning models for open-ended generation remains profoundly challenging due to the lack of definitive rewards. Current rubric-based RL methods mitigate this by employing explicit criteria; however, they rely heavily on static, human-annotated rubrics that inevitably cause policy lag, or expensive external proprietary models for dynamic updates. In this paper, we propose EvoRubric, a novel single-policy co-evolutionary RL framework that eliminates the reliance on static criteria and on external rubric generators. By unifying response generation and rubric generation under a single parameterized policy, EvoRubric dynamically alternates between a Reasoner and a Rubric Generator. To prevent reward hacking and ensure the reliability of generated signals, we introduce a multi-level verification pipeline featuring a meta-verifier, zero-variance pruning, and a Leave-One-Out peer consensus mechanism. Validated criteria are dynamically archived into a memory pool, yielding dense, multi-objective rewards to continuously co-optimize both roles. Extensive experiments across Medical, Writing, and Science domains demonstrate that EvoRubric consistently outperforms traditional static and external-LLM-driven alignment methods. Notably, our framework is compatible with human-expert priors. When initialized with expert-annotated rubrics, EvoRubric can further uncover novel, discriminative dimensions, achieving better performance than relying solely on static expert annotations. 

---
# Towards Localized and Disentangled Knowledge Editing for Multimodal Large Language Models 

**Authors**: Leijiang Gu, Zhen Zeng, Feng Li, Xinjian Gao, Zenglin Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.29826)  

**Abstract**: Existing methods in Multimodal Knowledge Editing (MKE) have advanced the ability to correct outdated or inaccurate knowledge in Multimodal Large Language Models (MLLMs). However, they exhibit a critical limitation: while effectively modifying target factual pairs, they fail to generalize edits to logically related queries and often cause unintended alterations to unrelated but visually or semantically linked information. We identify and formalize two underlying failure modes causing this issue: Causal Misalignment, which confines edits to the specific sample, and Feature Entanglement, which causes unintended alterations to coupled but irrelevant information. To address these issues, we propose Localized and Disentangled Knowledge Editing (LDKE), a new framework that achieves precise and generalized editing by localizing fact-specific model layers and disentangling target-relevant inputs from irrelevant ones. Our approach introduces a Fast Localization module to identify and update critical layers efficiently, along with a Disentanglement Classifier that routes inputs appropriately to preserve unrelated knowledge. Extensive experiments across various benchmarks and MLLMs demonstrate that LDKE achieves superior performance in propagating edits to related contexts while maintaining high locality. 

---
# Data filtering methods for training language models 

**Authors**: Egor Shevchenko, Elena Bruches  

**Link**: [PDF](https://arxiv.org/pdf/2605.29807)  

**Abstract**: Data quality is a critical factor in the effectiveness of machine learning models. Label errors, present even in widely used benchmarks, introduce noise into training data and reduce model generalization. In this work, we conduct a comparative analysis of two automatic label error detection methods - Confident Learning and Dataset Cartography - on three Russian text classification corpora of varying size, number of classes, and domain: ru_emotion_e-culture (49,123 examples, emotion classification), RuCoLA (8,524 examples, linguistic acceptability), and TERRa (2,337 examples, textual entailment recognition). We use the pre-trained rubert-base-cased model fine-tuned on each corpus. To verify the meaningfulness of filtering, we conduct control experiments with random removal of an equivalent number of examples. Results show that the effectiveness of both methods depends strongly on dataset characteristics: on large corpora with low noise levels, filtering does not improve performance, while on small datasets with high noise, Confident Learning achieves a significant F1-macro improvement. Dataset Cartography demonstrates more conservative behavior, removing fewer examples. Across all corpora, targeted removal by both methods outperforms random removal, confirming the meaningfulness of the approaches. 

---
# Nine Judges, Two Effective Votes: Correlated Errors Undermine LLM Evaluation Panels 

**Authors**: Guneet Kohli  

**Link**: [PDF](https://arxiv.org/pdf/2605.29800)  

**Abstract**: LLM-as-a-judge panels aggregate votes from multiple models, with the expectation that diverse models yield more reliable evaluations. We develop a framework to measure the true informational value of such panels and quantify how far their reliability falls short of the independent-voting ideal. Testing a panel of 9 frontier LLMs from 7 model families on three natural language inference datasets (each with 100 human annotations per item), we find that the 9 judges effectively provide only about 2 independent votes' worth of information. Roughly three-quarters of the panel's nominal independence is lost because the models make the same mistakes on the same items. The consequences are stark: the panel's actual accuracy falls 8-22 percentage points short of what independent voting would achieve, and the best single judge matches or outperforms the full panel across all conditions. Neither adding more judges nor using smarter aggregation algorithms helps -- established methods close at most 11% of this gap, even with access to the correct answers. We quantify these findings using the Kish effective sample size (n_eff) and a Condorcet null model, and show the deficit is robust across prompt variants, temperatures, chain-of-thought reasoning, and a pairwise preference task (RewardBench). The bottleneck is correlated judges, not the aggregation algorithm, implying that scaling up panels cannot substitute for genuinely independent evaluation. 

---
# Metric-Dependent Annotation Saturation for Learning from Label Distributions 

**Authors**: Guneet Kohli  

**Link**: [PDF](https://arxiv.org/pdf/2605.29797)  

**Abstract**: When annotators disagree on a label, the disagreement itself carries signal -- and the number of annotators needed to capture it depends on the evaluation metric. We fine-tune NLI models on label distributions subsampled from ChaosNLI, a dataset providing 100 independent annotator judgments per item, and identify metric-dependent saturation. In our 3-class NLI setting, entropy correlation -- whether the model identifies which items elicit disagreement -- requires N ~ 20-50 annotators to converge, while distributional match (KL divergence) saturates by N ~ 10 (87-95% of improvement across five model seeds). This finding rests on a prior observation: soft labels carry item-specific signal that label smoothing cannot replicate. Across five smoothing intensities, entropy correlation clusters at r ~ 0.45-0.49, while soft labels reach r = 0.643 (p < 0.001); per-item analysis traces this gap to smoothing's inability to distinguish ambiguous items from clear ones. The soft-label advantage replicates across two architectures (DeBERTa, RoBERTa), a non-NLI-pretrained baseline, and an exploratory cross-domain evaluation on content safety. These results suggest that annotation budgets should be informed by the target evaluation metric rather than set uniformly. 

---
# ActTraitBench: Quantifying the Knowledge-Decision Gap in Large Language Models via Human-Grounded Behavioral Validation 

**Authors**: Yutong Yang, Chenxi Miao, Weikang Li, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.29791)  

**Abstract**: While Large Language Models (LLMs) can convincingly simulate personas in explicit self-reports, they often deviate in implicit behavioral decisions, revealing a substantial Knowledge-Decision Gap ($G_{\text{KD}}$). Existing benchmarks struggle to measure this asymmetry due to limited construct validity, multi-dimensional entanglement, and distributional biases in LLM-based evaluation. To address these issues, we propose ActTraitBench, a human-grounded evaluation framework for measuring personality consistency in LLMs. Grounded in empirical human data, ActTraitBench establishes one-to-one mappings between psychometric facets and behavioral paradigms, and applies a Distributional Calibration via Quantile Mapping procedure to align LLM-judge score distributions with human norms. Experiments on 14 mainstream LLMs reveal a pervasive knowledge-decision asymmetry, where larger and more capable models often exhibit stronger behavioral divergence despite highly consistent self-reports. To mitigate this gap, we further introduce the Chain of Cognitive Alignment (CoCA), a plug-and-play inference-time intervention that improves alignment in reasoning-capable frontier models while exposing clear capability limitations in smaller architectures. 

---
# DySem: Uncovering Dynamic Semantic Components via Multilingual Consensus for Calculating Semantic Textual Similarity 

**Authors**: Kaijie Zheng, Weiqin Wang, Yile Wang, Hui Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29751)  

**Abstract**: Calculating semantic textual similarity is a foundational task in natural language processing. Current large language models (LLMs) based methods typically rely on extracting last-layer hidden states with fixed dimensions to compute similarity for every text pairs. We argue that this paradigm is suffer from two limitations: (i) The last hidden layer encodes more general knowledge rather than just semantic knowledge, making it suboptimal for semantic similarity computation; (ii) The hidden layer dimensions of LLMs are generally very large, which introduces some redundancy and noise for representing semantics. In this work, we propose DySem, a novel training-free framework that investigates more semantic-related internal components of LLMs via multilingual consensus, and shifts away from static representation spaces in favor of dynamic, sample-specific semantic dimensions by constructing text-dependent joint semantic set and computes similarity over this shared dimensional subset. Extensive experiments across various LLMs show that our method consistently outperforms recent baselines while maintaining lower dimensions for similarity calculation. The code is released at this https URL. 

---
# AfriScience-MT: Towards Decolonizing Science in Africa through Text Translation 

**Authors**: Idris Abdulmumin, Tajuddeen Gwadabe, Shamsuddeen Hassan Muhammad, David Ifeoluwa Adelani, Nomonde Khalo, Ibrahim Said Ahmad, Abiodun Modupe, Anina Mumm, Sibusiso Biyela, Michelle Rabie, Johanna Havemann, Marek Rei, Jade Abbott, Vukosi Marivate  

**Link**: [PDF](https://arxiv.org/pdf/2605.29741)  

**Abstract**: The dominance of colonial languages in African education and scientific communication limits how hundreds of millions of speakers of African languages access and produce scientific knowledge. A core obstacle is the lack of established scientific terminology in these languages. We introduce AfriScience-MT, a parallel corpus covering six African languages (Amharic, Hausa, Luganda, Northern Sotho, Yorùbá, and isiZulu) across 11 scientific domains. Professional translators, working with expert science communicators, translated plain-language summaries of scientific papers into each target language and created new terms where none existed. We benchmark machine translation systems and large language models in zero-shot, few-shot, and fine-tuned settings. Our results show that closed-source models outperform all open-source models at both the sentence and document levels: GPT-5.4 and Gemini-3.1-Flash-Lite lead with average sentence-level COMET scores of 68.3 and 68.0, respectively, and tie at an average document-level COMET of 48.3. Among open systems, fine-tuned NLLB-1.3B reaches 67.3 at the sentence level, and TranslateGemma-12B reaches 44.0 at the document level with 1-shot in-context learning. We release AfriScience-MT to support benchmarking and document-level scientific MT for African languages. 

---
# Multi-Legal-Bench: Evaluating LLMs on Legal Reasoning Across Jurisdictions, Languages, and Legal Traditions 

**Authors**: Volodymyr Ovcharov  

**Link**: [PDF](https://arxiv.org/pdf/2605.29738)  

**Abstract**: Legal NLP benchmarks overwhelmingly evaluate a single language or aggregate tasks that differ fundamentally across jurisdictions, making cross-lingual comparison impossible. We introduce Multi-Legal-Bench, the first cross-jurisdictional legal benchmark that evaluates identical tasks across six countries (Ukraine, France, Netherlands, Poland, Czech Republic, Lithuania), four language families, and 134 million court decisions. The benchmark defines five tasks court-type classification, judgment form classification, case-outcome prediction, legal norm extraction, and cause category prediction mapped to structured metadata from national court registries, forming a deliberately sparse 5x6 task-jurisdiction matrix (20 of 30 cells filled). We evaluate 7 frontier LLMs under zero-shot and 3-shot prompting via AWS Bedrock, with 4 additional small/medium models (3-12B) for scaling analysis. Our results reveal that: (1) task-dependent few-shot effects discovered in Ukrainian replicate across all jurisdictions; (2) no single model dominates any language rankings shift with both task and jurisdiction; (3) cross-lingual few-shot transfer does not follow language proximity: UA->FR (Romance, -2.1 pp) transfers better than UA->PL (Slavic, -13.7 pp), with label-set alignment predicting transfer quality better than language family; and (4) tokenizer fertility, despite a 2.3x spread, does not significantly predict cross-lingual accuracy (r=-0.27, p=0.14), suggesting that model architecture and pretraining data dominate tokenizer efficiency. We release all data, prompts, and model predictions. 

---
# HTAM: Hierarchical Transition-Attended Memory for Operator Optimization 

**Authors**: Yining Zhang, Mingyang Yi, Chen Wang, Xuwen Xiang, Tianhe Jia, Zedong Dan, Chengqing Zong, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29734)  

**Abstract**: High-performance GPU kernels are essential for efficient LLM deployment, yet optimizing them remains expertise-intensive. Recent LLM-based code generation makes automatic GPU operator generation promising, but operator optimization remains a hardware-aware search problem. Existing LLM-based methods face a granularity mismatch: coarse hints are reusable but hard to execute, whereas detailed memories are actionable but enlarge the search space and obscure optimization bottlenecks. The key challenge is therefore to organize optimization experience at an appropriate granularity. To address this issue, this paper proposes HTAM (Hierarchical Transition-Attended Memory), a coarse-to-fine framework for LLM-based operator optimization. HTAM builds a two-level Hierarchical Transition Graph (HTG) to organize coarse global directions, detailed local strategies, and transition experience between optimization steps. During each evolution step, HTAM selects a global direction from the current state and recent optimization history, retrieves the corresponding local strategy memory, and uses it to guide concrete CUDA code generation. Experiments on the full KernelBench suite demonstrate that HTAM consistently improves correctness, fast-solution rate, and speedup over LLM-based baselines, while backend and Robust-KBench studies indicate transferable benefits from structured memory. 

---
# User-Aware Active Knowledge Acquisition for Emotional Support Dialogue 

**Authors**: Mufan Xu, Kehai Chen, Jiahao Hu, Xinchao Xu, Muyun Yang, Tiejun Zhao, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29715)  

**Abstract**: Emotional support plays an important role in dialogue systems, and its success depends on adapting to a user's evolving and implicit needs across multi-turn interactions while leveraging the strong reasoning capacity of large language models. However, since signals about user needs are often weak, indirect, and can only be disambiguated through multi-turn interaction, existing emotional support methods often struggle to acquire and generalize relevant conversational knowledge efficiently. To bridge this gap, we introduce User-Aware Active Knowledge Acquisition (UKA), a gradient-free active dialogue learning framework that explicitly represents uncertainty about user needs and incorporates active learning into both knowledge acquisition and response this http URL propose a Theory-of-Mind uncertainty estimation mechanism that allows the model to prioritize responses, thereby eliciting more informative user feedback. UKA is capable of efficiently exploring user-aligned conversational knowledge during training while maintaining robustness at test time. Experiments across multiple dialogue benchmarks and model architectures demonstrate that our approach consistently outperforms strong baselines in dialogue quality and user alignment. 

---
# Leveraging Routing Dynamics in Mixture-of-Experts Models for Efficient Language Adaptation 

**Authors**: Aditi Khandelwal, Marius Mosbach, Verna Dankers, Siva Reddy, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2605.29714)  

**Abstract**: Mixture-of-Experts (MoE) models are widely used to scale language models, yet their expert routing behavior and adaptation in a multilingual setting remain underexplored. In this work, we study multilingual routing dynamics during continual pre-training of an English-centric MoE model on a multilingual corpus, analyzing how expert usage varies across languages. We find that continual multilingual pre-training leads to diffused, language-agnostic routing in early and middle layers, with language specialization primarily emerging in the final layers. We also show that token-level vocabulary overlap between languages plays an important role in how languages are routed. Motivated by these findings, we propose a parameter-efficient adaptation strategy that updates language-specific and shared experts in the final MoE layers. Experiments on MultiBLiMP and Belebele show that our method achieves a strong performance-efficiency trade-off, attaining competitive performance relative to fine-tuning complete final layers, while updating less than 2% of the parameters. Overall, our findings provide insights into where and how language specialization emerges in MoEs during continual pre-training and provide practical insights for low-resource multilingual adaptation. Our code is available at this https URL. 

---
# Teaching Language Models to Check Grounded Claim Factuality with Human Test-Taking Strategies 

**Authors**: Yuxuan Ye, Raul Santos-Rodriguez, Edwin Simpson  

**Link**: [PDF](https://arxiv.org/pdf/2605.29712)  

**Abstract**: Grounded claim factuality checking is important for large language model (LLM) applications such as retrieval-augmented generation, as it helps users assess the correctness of generated outputs. Existing metrics using entailment classifiers require dataset-specific threshold tuning, while LLM-based approaches often use direct prompting, which underutilises the reasoning capabilities of LLMs. We address this by formulating grounded claim factuality checking as a true/false reading comprehension task and prompting LLMs with explicit test-taking strategies for efficient reasoning. Our method reduces token usage by over 80% compared to unguided open-ended reasoning, and achieves competitive performance to more expensive alternatives across two factuality benchmarks, setting a new state of the art on one. To further reduce inference cost, we train small language models (SLMs) to replace LLMs in the checking pipeline. Using supervised fine-tuning (SFT) and a self-revision mechanism, the SLMs learn to improve their factuality judgements. Experimental results show that the resulting SLMs perform on par with strong baselines, combining low inference costs with generating supporting rationales to support interpretability. Code and datasets will be released upon acceptance. 

---
# Personalized Turn-Level User Conversation Satisfaction Benchmark 

**Authors**: Zhefan Wang, Zhiqiang Guo, Weizhi Ma, Min Zhang, Quanjia Yan, Hengliang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.29711)  

**Abstract**: User satisfaction with AI assistants is highly personalized: the same response may satisfy one user but disappoint another depending on what each user expects and what they have asked for before. Existing automatic evaluation methods mostly measure generic response quality, making it difficult to judge whether a response satisfies a user at a specific turn. We study this problem as personalized turn-level user conversation satisfaction evaluation. We build a conversation satisfaction evaluator that combines compact user memories with target-turn context to produce satisfaction scores and dissatisfaction-oriented rationales. Meta-evaluation against human satisfaction annotations shows that personalized memory and post-hoc score calibration improve ordinal agreement and dissatisfied-turn detection over supervised, retrieval-based, and generic LLM-as-a-judge baselines. We further introduce PersTurnBench, a personalized turn-level user conversation satisfaction benchmark that uses the verified evaluator to assess generation models via replay. By holding the replay state fixed, PersTurnBench enables controlled comparison of generic generation models and memory-augmented personalized systems without new human labels for every candidate model. The evaluator and benchmark let researchers compare candidate generation models on personalized satisfaction without collecting new user feedback for every model. 

---
# Understanding Safety-Sensitive Expert Behavior in Mixture-of-Experts LLMs 

**Authors**: Zhibo Zhang, Yuxi Li, Zhen Ouyang, Ling Shi, Kailong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29708)  

**Abstract**: Mixture-of-Experts (MoE) LLMs rely on sparse, router-driven expert activation, yet how safety alignment interacts with routed expert specialization remains underexplored. A common intuition is that safety behavior may be controlled by routing harmful requests to distinct refusal-oriented experts. In this work, we provide empirical evidence for a different picture: routing patterns in aligned MoE LLMs are largely topic-driven, while safety behavior can be altered with little change to the model's intrinsic routing path.
Motivated by this observation, we present **RASET** (**R**outer-**A**gnostic **S**afety-critical **E**xpert **T**uning), a red-teaming framework that probes safety enforcement that is localized in a small subset of experts while preserving the model's intrinsic routing behavior. **RASET** identifies safety-critical experts via a contrastive routing-sensitivity criterion and applies parameter-efficient tuning only to the selected experts, minimizing semantic disruption relative to router-steering interventions. These results reveal a distinct MoE safety risk, highlighting the need for expert-aware alignment mechanisms. 

---
# Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding 

**Authors**: Jianuo Huang, Yaojie Zhang, Qituan Zhang, Hao Lin, Hanlin Xu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29707)  

**Abstract**: Speculative decoding accelerates LLM inference by drafting multiple tokens and verifying them in parallel with the target model. However, its practical speedup is constrained by the trade-off between draft quality and drafting cost: autoregressive drafters model causal dependencies among draft tokens but incur sequential overhead, while parallel drafters reduce drafting cost but weaken intra-block dependency modeling. In this paper, we propose Domino, a speculative decoding framework that decouples causal dependency modeling from expensive autoregressive draft execution. Domino first uses a parallel draft backbone to produce preliminary draft distributions for the entire block, and then applies a lightweight Domino head to refine them with prefix-dependent causal information. To stabilize teacher-forced causal encoding, we further introduce a base-anchored training curriculum that first strengthens the parallel backbone and then gradually shifts optimization toward the causally corrected final distribution. Experiments on Qwen3 models show that Domino achieves up to \(5.49\times\) end-to-end speedup under the Transformers backend and up to \(5.8\times\) throughput speedup under SGLang serving. 

---
# Scaling Laws for Agent Harnesses via Effective Feedback Compute 

**Authors**: Xuanliang Zhang, Dingzirui Wang, Keyan Xu, Qingfu Zhu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2605.29682)  

**Abstract**: Agent harnesses increasingly determine the performance of language-model systems by deciding how models call tools, receive feedback, verify intermediate states, store memory, and revise solutions. Yet current test-time scaling analyses often parameterize this process by raw expenditure -- tokens, tool calls, operations, wall time, or cost -- which does not distinguish useful feedback from redundant or unstable interaction. We introduce \emph{Effective Feedback Compute} (EFC), a trace-level scaling coordinate that credits feedback only when it is informative, valid, non-redundant, and retained for subsequent decisions, and we normalize it by task demand when comparing tasks with different feedback requirements. Across synthetic controllable tasks, executable code tasks, real benchmark traces, held-out splits, and a prospective validation batch, EFC-based coordinates consistently predict failure rates better than raw-compute baselines and a strong multivariate SAS baseline. In controlled scaling, raw tokens and tool calls explain limited variation ($R^2=0.33$ and $0.42$), SAS reaches $0.88$, while Oracle-EFC and Estimated-EFC reach $0.94$ and Oracle-EFC/$D_{\mathrm{task}}$ reaches $0.99$. Matched-budget interventions show that improving feedback quality raises success from $0.27$ to $0.90$ while raw cost and tool calls are fixed. On mixed real traces, NRS-EFC/$D_{\mathrm{task}}$ reaches $R^2=0.92$ while raw compute has near-zero or negative fit, and it remains the best predictor in a prospective holdout ($R^2=0.85$). These results suggest that harness scaling is governed less by how much computation is spent than by how efficiently raw budget is converted into durable, task-sufficient feedback. 

---
# Spurious Prompts: Can Irrelevant Prompts Steer Large Language Models? 

**Authors**: Pawel Batorski, Abtin Pourhadi, Jerzy Sarosiek, Przemyslaw Spurek, Paul Swoboda  

**Link**: [PDF](https://arxiv.org/pdf/2605.29678)  

**Abstract**: Large language models are highly sensitive to prompts, but this sensitivity is usually studied through task-relevant instructions, demonstrations, or reasoning cues. In this paper, we study a different form of prompt sensitivity: whether prompts that are semantically unrelated to the task can nevertheless steer model behavior. We call them spurious prompts and show their surprising efficacy. We also propose a simple black-box search procedure for discovering them. Across reasoning and question-answering benchmarks, using models ranging from 0.8B to 27B parameters and spanning three model families, we show that spurious prompts can improve performance, often matching or outperforming standard prompting baselines and task-aware prompt optimization. We further show that they can steer models toward unintended behaviors, such as repeatedly selecting the first answer option, producing incorrect answers, returning an even, prime or small number without explicitly instructing the model to do so. These findings reveal a new kind of prompt sensitivity: LLMs can be systematically steered by prompts that are unrelated to the task they are asked to solve. Our code is available at this https URL 

---
# EviLink: Multi-Path Schema Linking with Uncertainty-Guided Evidence Acquisition for Large-Scale Text-to-SQL 

**Authors**: Huawei Zheng, Sen Yang, Zhaorui Yang, Yuhui Zhang, Haozhe Feng, Haoxuan Li, Xuan Yi, Chao Hu, Defeng Xie, Chen Hou, Danqing Huang, Wei Chen, Yingcai Wu, Peng Chen, Dazhen Deng  

**Link**: [PDF](https://arxiv.org/pdf/2605.29670)  

**Abstract**: Schema linking is a difficult and important step in large-scale Text-to-SQL, where systems must identify a compact yet sufficient schema context from large and ambiguous databases. Existing methods often treat schema linking as deterministic selection around a single SQL path, but complex questions may admit multiple valid realizations with different schema needs. We reframe schema linking as uncertainty-aware schema-need inference over multiple plausible SQL paths, where the system distinguishes required schema items from path-dependent uncertain ones and acquires evidence only where needed. We instantiate this reframing with EviLink, which combines multi-hypothesis schema grounding with uncertainty-guided evidence acquisition. Experiments on BIRD-Dev and Spider2-Snow show that this perspective improves the balance among schema completeness, schema relevance, and token cost. On Spider2-Snow, EviLink achieves 90.15% field-level strict recall rate, uses 123.30K average tokens, and improves downstream SQL generation under a fixed generator. 

---
# Beyond English and Evasion: A Human-Annotated Multi-Domain Benchmark for High-Stakes LLM Safety Evaluation in Chinese 

**Authors**: Wajdi Zaghouani, Kholoud K. Aldous, Yicheng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.29667)  

**Abstract**: When Large Language Models (LLMs) are deployed in Chinese-language settings, a troubling pattern emerges: safety systems that work well in English break down. These systems struggle to cross linguistic and cultural bound-aries, leaving models exposed to adversarial prompts that exploit Chinese-specific evasion techniques, including Pinyin romanization, character decomposition, internet slang, and hedging tone. To address this gap, we introduce ChiSafe-PAS (Chinese Safety Pilot Annotation Set), a human-annotated benchmark of 1,897 adversarial Chinese prompts spanning four high-stakes domains: self-harm and violence, drug and illicit trade, fraud, and satire. Of these, 1,544 entries carry complete gold-standard annotations: a 3-class response label (REFUSE, SAFE-REDIRECT, RESPOND), a nine-category obfuscation taxonomy, a risk-level rating, and annotator rationale. We describe the dataset design, annotation process, and obfuscation taxonomy in detail. Our primary goal is practical: to give the research community a high-quality, culturally grounded resource for benchmarking LLM safety alignment. In doing so, we engage three broader tensions in the field: the blurring boundary between training and evaluation data, the need for domain coverage grounded in real-world risk, and the limits of scale as a substitute for cultural expertise. 

---
# Verifiable Rewards Beyond Math and Code: Lightweight Corpus-Grounded Process Supervision for Factual Question Answering 

**Authors**: Shicheng Fan, Haochang Hao, Dehai Min, Weihao Liu, Philip S. Yu, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.29648)  

**Abstract**: Applying reinforcement learning to improve factual accuracy in knowledge-intensive question answering faces a reward design dilemma. Response-level rewards provide only coarse supervision and cannot distinguish correct from incorrect statements within a reasoning trace. Sentence-level alternatives offer finer-grained feedback, but typically rely on NLI verifiers, LLM judges, or knowledge-verification pipelines that are expensive to deploy at RL scale and often unreliable for rare-entity facts, where accurate reward signals are especially important. We propose CorVer (Corpus Verify), a lightweight, plug-in-ready process reward that replaces neural verifiers with a corpus-grounded signal derived from Wikipedia co-occurrence statistics. CorVer assigns sentence-level credit and maps it to token-level advantages via a simple alignment, requiring only a 0.5B extractor and a single corpus lookup per sentence. Across 30 (model, benchmark) cells spanning six instruction-tuned models (3B to 14B) and five QA benchmarks, CorVer improves over the raw baseline for every cell, with an average TriviaQA gain of +4.1 pp. It also outperforms four neural-verifier baselines in 18 of 20 cells under their feasible configurations, while training 4.8 to 8.4x faster. 

---
# Classification of non-analyzable word types in web documents to implement an effective Korean e-learning system 

**Authors**: Sang-Taek Park, Ae-Lim Ahn, Eric Laporte, Jee-Sun Nam  

**Link**: [PDF](https://arxiv.org/pdf/2605.29638)  

**Abstract**: E-learning systems should deliver contents that reflect various phenomena of the language as it is used. In addition to formal Korean, e-learning systems that would include real-world Korean expressions such as those in web documents, mobile text messages, or twitter posts, would be useful to high-level learners. We construct two types of corpora: one is made of formal documents like online news articles; the other is made of informal documents like customer reviews about new products in web blogs. By comparing these corpora, we show how expressions differ in these two types of corpora. We survey the main characteristics of the informal corpus. Given that a significant proportion of text is informal, we propose Local Grammar Graphs (LGG) as an appropriate model to treat them effectively in Korean e-learning systems. 

---
# Evaluating Cross-lingual Knowledge Consistency in Code-Mixed vis-a-vis Indian Languages using IndicKLAR 

**Authors**: Debajyoti Mazumder, Divyansh Pathak, Prashant Kodali, Aditya Joshi, Akshay Agarwal, Jasabanta Patro  

**Link**: [PDF](https://arxiv.org/pdf/2605.29637)  

**Abstract**: Large language models recall knowledge reliably in English but often fail on the same query posed in a lower-resourced language -- a crosslingual consistency gap that remains underexplored for Indian languages and their code-mixed counterparts. To study this gap, we introduce IndiKLAR, an Indic extension of the KLAR-CLC benchmark covering 18 of the 22 scheduled Indian languages and pairing them with code-mixed variants for 11 widely used language pairs, with native-speaker verification of both monolingual and code-mixed variants for these 11 settings. This three-way alignment offers a unique opportunity to examine how knowledge recall consistency varies across the spectrum of English, code-mixed, and native Indian language inputs. Evaluating across nine open-weight models, we find that the native-language accuracy gap to English can reach $\sim$0.50, while code-mixed inputs close most of it -- bringing performance within $\sim$0.05 of English without any model-level intervention. Motivated by this, we evaluate several prompting strategies that vary in how language conversion is exposed, including a two-stage translate-then-answer setup, a one-stage joint translation-and-answer prompt, and Translate-in-Thought (TinT) -- a single-step strategy in which the model converts the input internally and emits only the final answer. Across the performance trajectory native $\rightarrow$ code-mixed $\rightarrow$ English, we identify a consistent flip point -- the boundary between incorrect and correct prediction -- that lies between the native and code-mixed settings. Interestingly, this holds whether the trajectory is induced by the input surface form or by the model's internal conversion process. 

---
# Predicting Causal Effects from Natural Language Queries using Structured Representations 

**Authors**: Giuliano Martinelli, Piriyakorn Piriyatamwong, Abelardo Carlos Martinez Lorenzo, Jasmin Baier, Riccardo Orlando, Satvik Garg, Sharif Kazemi, Linxi Wang, Arianna Legovini, Samuel Fraiberger  

**Link**: [PDF](https://arxiv.org/pdf/2605.29631)  

**Abstract**: Randomized controlled trials are a cornerstone of medicine and the social sciences as they enable reliable estimates of causal effects. However, they are costly and time-consuming to conduct, motivating interest in predicting causal effects from existing experimental evidence. Recent advances in large language models (LLMs) have demonstrated strong performance on knowledge-intensive tasks, raising the question of whether these models can be used for forecasting causal effect sizes. To investigate this, we introduce Query2Effect, a new large-scale benchmark consisting of more than 72,000 natural language questions aligned with experiment descriptions, created to simulate realistic information-seeking scenarios by varying query specificity along dimensions of implicitness, abstraction, and ambiguity. We then propose a two-step framework that first generates a synthetic structured representation of a query before predicting effect size using a supervised encoder model. Experiments show that finetuning plays a crucial role in improving prediction performance, with absolute error reducing by -27% up to -71% compared to prompted out-of-the-box LLMs, and that our two-step framework is beneficial for out-of-domain generalization, highlighting the benefits of separating semantic interpretation from numerical effect estimation. 

---
# Entity-Collision: A Stratified Protocol for Attributing Retrieval Lift in Agent Memory 

**Authors**: Youwang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2605.29630)  

**Abstract**: End-to-end agent-memory benchmarks report a single hit@k per retriever, confounding lexical leakage (uncontrolled query/gold/distractor entity overlap) with tag-mixing (preferences, services, tools averaged together). We propose entity-collision, a system-agnostic protocol that pins the BM25 floor by construction -- every distractor shares the answer's entity tokens -- and stratifies queries by discriminator tag, so any lift over BM25 is attributable to the embedder. Applied to an open-source agent-memory testbed across 5 tags x 3 embedders x 5 collision degrees with paired-bootstrap 95% CIs, the protocol reveals a two-axis pattern: a 256-d hash trigram helps only on closed-vocabulary lexical tags at deep collision; MiniLM-384 dominates both axes; and a 2.7x-parameter BGE-large does not uniformly improve on MiniLM -- it wins on intent-style queries but loses on lexical ones. Encoder capacity alone is not the binding constraint. The synthetic intent-tag null replicates on LongMemEval (n=500) as a single-session-preference recall cliff. Adaptive vector-weight routing on LoCoMo is a measured null: 11.7pp of oracle headroom exists, but no signal we tested recovers it. All 26 result tables and 37 reproduce scripts are version-controlled and verified by a public registry; the protocol is exercised on a deterministically governed memory testbed (event-sourced decision log, DAG-state-machine schema lifecycle) so every reported CI is reproducible byte-for-byte from the ingest stream. 

---
# DLM-SWAI: Steering Diffusion Language Models Before They Unmask 

**Authors**: Hyeseon An, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.29626)  

**Abstract**: Steering language model generation toward desired textual properties is essential for practical deployment, and inference-time methods are particularly appealing because they enable controllable generation without retraining. Recent work has also highlighted diffusion language models as an emerging generation paradigm with distinct decoding properties. However, most existing steering approaches either rely on auxiliary models or are designed for autoregressive next-token decoding, making them difficult to apply to diffusion language models DLMs, which generate text through iterative denoising of partially masked sequences. Therefore, we propose DLM-SWAI, a simple training-free steering method that biases the token distribution at each denoising step using pre-computed token-level style scores. Experiments on style and safety control tasks show that DLM-SWAI effectively steers diffusion language models while preserving generation quality and requiring minimal computational overhead. Ablations further reveal a controllable trade-off between steering strength and fluency, and our analysis links class-wise steerability to the strength of token-level attribute cues. 

---
# Training Deliberative Monitors for Black-Box Scheming Detection 

**Authors**: Aditya Sinha, Akshat Naik, Victor Gillioz, Simon Storf, Kilian Merkelbach, Rich Barton-Cooper, Axel Højmark, Marius Hobbhahn  

**Link**: [PDF](https://arxiv.org/pdf/2605.29601)  

**Abstract**: As autonomous agents become more capable of performing real-world tasks, distinguishing scheming behavior from benign task pursuit may become a central AI control problem. Existing monitors often rely on chain-of-thought access or internal activations, or use prompted frontier models, all of which can be unavailable, unreliable or expensive in deployment. In this work, we study action-only deliberative monitors: smaller open-weight models trained to detect scheming and sabotage from agentic trajectories without accessing the monitored agent's reasoning or model internals. Our method, inspired by deliberative alignment, uses a scheming specification to elicit structured rationales from a frontier teacher, filters them with a separate judge, and distills the highest-quality rationales into open-weight monitors with supervised fine-tuning and reinforcement learning. We train on five datasets, and evaluate across six out-of-distribution agentic misalignment benchmarks. We show that applying our method to Qwen3.5-27B yields higher performance than all low-cost frontier models as prompted monitors (Gemini 3.1 Flash-Lite, GPT-5.4 Nano, and Claude Haiku 4.5) and than Gemini 2.5 Pro, while also achieving lower marginal inference cost (token-metered USD per 1,000 evaluations). Stronger prompted frontier monitors (Gemini 3.1 Pro, GPT-5.4, Claude Sonnet 4.6, and Claude Opus 4.6) achieve higher performance but at roughly $16$--$34\times$ higher marginal inference cost. Several of our trained monitors are positioned on the empirical cost--performance Pareto frontier among the monitors we evaluate, providing practical low-cost, low-FPR alternatives to prompted frontier models. 

---
# World Models in Words: Auditing Physical State-Transition Commitments in Vision-Language Models 

**Authors**: Emmanuelle Bourigault  

**Link**: [PDF](https://arxiv.org/pdf/2605.29585)  

**Abstract**: Vision-language models (VLMs) are increasingly used to answer questions about physical scenes, yet most evaluations reduce performance to a final answer. This hides whether the model perceived the right objects, represented the right physical state, predicted a plausible transition, or merely selected the right option for the wrong reasons. We introduce \wmw, an evaluation framework for auditing the \emph{language-expressed physical commitments} of VLMs. Instead of scoring only $I,q\mapsto a$, we ask models to produce a typed trace $I,q\mapsto(s_0,\Delta s,s_1,a)$: an initial state, a state transition, a resulting state, and an answer. A hybrid verifier then checks schema validity, state grounding, transition consistency, and answer-trace compatibility, yielding typed error labels such as object, relation, force, transition, temporal, unit/scale, and faithfulness errors. We release \tracebank, a controlled trace resource with \nSeed schema- and recomputation-validated synthetic scenarios across \nFamilies physics families, \nPairs minimally perturbed contrastive preference pairs, verifier code, audit guidelines, and model outputs. We evaluate \nModels VLMs on both controlled and external physical-reasoning examples. \wmw reveals failures that answer-only evaluation misses: 35\% of correct answers from mid-tier models are backed by physically invalid traces. Verifier-guided reranking recovers up to 7 percentage points of trace validity without sacrificing answer accuracy, and trace-level preference tuning reduces hidden inconsistency by 41\% relative. The contribution is not another final-answer physics benchmark, but a reusable protocol for measuring whether a VLM's stated physical world can be true at the same time as its answer. 

---
# GAPD: Gold-Action Policy Distillation for Agentic Reinforcement Learning in Knowledge Base Question Answering 

**Authors**: Xin Sun, Jianan Xie, Zhongqi Chen, Qiang Liu, Shu Wu, Bowen Song, Weiqiang Wang, Zilei Wang, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29584)  

**Abstract**: Reinforcement learning (RL) is a natural fit for agentic knowledge base question answering (KBQA), where a model must issue executable actions, observe knowledge-base feedback, and eventually return an answer. However, current RL-based KBQA systems mainly optimize sparse rewards from the final answer, leaving intermediate action errors weakly supervised. This is especially limiting for logical-form annotated KBQA benchmarks: gold logical forms can be converted into executable action sequences, but existing pipelines use them mainly for warm-start data construction rather than for on-policy RL updates. We propose GAPD, a training-time Gold-Action Policy Distillation framework that adds dense token-level guidance to outcome-based RL. To align gold actions with on-policy student rollouts, GAPD uses MID-ANCHOR MATCHING: it treats the intermediate entities reached during student exploration and gold execution as state anchors, and matches student states to gold states through these explored entity sets. The current policy conditioned on this aligned gold action serves as a stop-gradient teacher, whose token distribution is distilled back to the ordinary student policy over generated action-token spans. GAPD consistently surpasses the current state of the art on WebQSP, GrailQA, and GraphQ. 

---
# LiteCoder-Terminal: Scaling Long-Horizon Terminal Environments for Learning Language Agents 

**Authors**: Xiaoxuan Peng, Kaiqi Zhang, Xinyu Lu, Boxi Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.29559)  

**Abstract**: Mastering terminal environments requires language agents capable of multi-step planning, feedback-grounded execution, and dynamic state adaptation. However, training such agents is currently bottlenecked by a reliance on scraped external repositories, which limits domain diversity, environment controllability, and the targeting of specific capability deficits. We introduce LiteCoder-Terminal-Gen, a zero-dependency synthesis pipeline that autonomously generates executable and verifiable terminal training environments directly from domain specifications. Using this framework, we construct two large-scale resources: LiteCoder-Terminal-SFT, comprising 11,255 expert trajectories across 10 domains, and LiteCoder-Terminal-RL, featuring 602 verifiable environments for trajectory-level preference optimization. Supervised fine-tuning of Qwen-family models on our SFT dataset yields agents that significantly outperform their base counterparts. Notably, our 32B variant achieves 29.06%, 18.54%, and 34.00% pass@1 on Terminal Bench 1.0, 2.0, and Pro, respectively. Furthermore, applying Direct Multi-turn Preference Optimization (DMPO) on our RL environments yields additional performance gains. These results systematically demonstrate that fully synthetic, executable environments offer a scalable and verifiable supervision signal for mastering complex, real-world command-line workflows. 

---
# From Blind Guess to Informed Judgment: Teaching LLMs to Evaluate Materials by Building Knowledge-Augmented Preference Signals 

**Authors**: Yeyong Yu, Wenya Hu, Xing Wu, Quan Qian  

**Link**: [PDF](https://arxiv.org/pdf/2605.29555)  

**Abstract**: As candidate generation and high-throughput experimentation advance, the primary bottleneck in materials discovery is shifting from property prediction to making reliable evaluations among massive candidate sets. We propose a Knowledge-Augmented Preference Signals Framework, MaterEval, that automatically produces, for the same candidate, two evaluations: an informed judgment that follows expert rules and provides supporting evidence, and a rule-removed blind guess. By pairing the two evaluations as preference data, we guide general-purpose large language models (LLMs), originally lacking materials-specific criteria, from intuitive judgment toward reliable evaluation supported by explicit evidence. To balance throughput, cost, and reliability, we further introduce a fast-slow reasoning scheme that decouples large-scale rapid screening from in-depth review on a small subset. Using high-entropy alloy (HEA) assessment as a case study, we show that, without external retrieval and relying solely on internalized capabilities, small open-source LLMs achieve substantial gains in accuracy, conclusion consistency, and evidence discrimination, approaching the performance of rule-based closed-source LLMs. These results demonstrate that expert rules can be systematically transformed into learnable preference signals, enabling a low-cost and deployable evaluation module for autonomous materials discovery loops. 

---
# Source-Grounded Semantic Reinforcement Learning for Low-Resource Target-Language Generation 

**Authors**: Zeli Su, Ziyin Zhang, Zewei Pan, Zhou Liu, Dingcheng Huang, Dehan Li, Zhankai Xu, Longfei Zheng, Xiaolu Zhang, Jun Zhou, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29502)  

**Abstract**: Low-resource target-language generation is often limited by scarce parallel data, while high-resource source-language monolingual data is abundant but difficult to use with standard supervised fine-tuning. We propose Source-Grounded Semantic Reinforcement Learning (SG-SRL), a resource-utilization framework that converts source-language monolingual data into cross-lingual semantic supervision for target-language generation. SG-SRL performs reference-free reinforcement learning (RL) on source-language data using a cross-lingual semantic reward model, instantiated by a cross-lingual reranker that scores the semantic relevance between the source input and the target-language generation. While this induces severe verbosity-based reward hacking, a lightweight recovery stage using a small parallel corpus restores fluency, conciseness, and task format while preserving the semantic gains. Experiments on Chinese-to-Thai generation show that SG-SRL improves semantic grounding and factual coverage over cold-start SFT. Additional analyses on long-form transfer and Tibetan embedding-based rewards clarify the generalization behavior of SG-SRL and show that an encoder-based semantic reward can substitute for an LLM-based reranker in a realistic low-resource language setting. 

---
# Mask the Target: A Plug-and-Play Regularizer Against LoRA Forgetting 

**Authors**: Runze Xu, Arpit Garg, Hemanth Saratchandran, Simon Lucey  

**Link**: [PDF](https://arxiv.org/pdf/2605.29498)  

**Abstract**: Low-Rank Adaptation (LoRA) has become one of the most widely used fine-tuning mechanisms for adapting large language models to new domains, tasks, and users. Yet adaptation performance alone can obscure an important failure mode: LoRA updates may improve performance on the target distribution while degrading prior capabilities learned during pretraining and alignment. We show that this forgetting becomes especially severe when the adaptation distribution differs substantially from the models original training or alignment distributions. The challenge is amplified in practical settings, where the original training and alignment data are typically unavailable. Motivated by this constraint, we study how LoRA based adaptation balances new learning against forgetting in a replay-free setting, and introduce a simple output space regularizer that can be added directly to existing training pipelines. Our method removes the ground-truth token from both the base and adapted model distributions, renormalizes the remaining probabilities, and applies KL regularization only over the non-target vocabulary. This preserves the base models relative preferences among alternative tokens without directly opposing the cross-entropy signal required for adaptation. As the regularizer acts only at the loss level, it requires no replay data, architectural changes, adapter redesign, or inference-time overhead, and can be applied directly to existing LoRA variants. Across all LoRA variants tested and across various backbones, our method improves the frontier between new learning and forgetting when the adaptation distribution differs substantially from the base models original training or alignment distributions, suggesting a broadly applicable route toward more reliable LLM updating. 

---
# On Asymmetric Optimization of Reasoning and Perception in Vision-Language Model Post-Training 

**Authors**: Xueqing Wu, Yu-Chi Lin, Kai-Wei Chang, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.29496)  

**Abstract**: Post-training has greatly improved reasoning in frontier vision-language models, yet its gains for perception remain comparatively limited, creating a bottleneck for end-to-end visual reasoning. To investigate this gap, we introduce a controlled diagnostic framework with two synthetic tasks that disentangle perception from reasoning. Our analysis reveals a consistent perception-reasoning asymmetry: posttraining improves reasoning more substantially than perception, though the underlying mechanism differs by training paradigm. For supervised fine-tuning (SFT), this asymmetry stems from token imbalance in chain-of-thought supervision, where perception occupies fewer tokens and thus receives a weaker training signal. Dynamically reweighting the loss mitigates this imbalance and boosts end-to-end performance by up to 18.2. For reinforcement learning (RL), the asymmetry instead arises from reward coupling: outcome rewards correlate more strongly with reasoning than with perception, weakening the signal for perception learning. Adding a perception-aware reward alleviates the imbalance and improves end-to-end accuracy by up to 6.0; even without groundtruth perception rewards, a reliable surrogate reward provide useful signal, yielding gains of 3.2 points. Together, our results comprehensively diagnose asymmetric optimization and suggest concrete interventions to balance perception and reasoning. 

---
# PhoneWorld: Scaling Phone-Use Agent Environments 

**Authors**: Zhengyang Tang, Yuxuan Liu, Xin Lai, Junyi Li, Pengyuan Lyu, Jason, Yiduo Guo, Zhengyao Fang, Yang Ding, Yi Zhang, Weinong Wang, Huawen Shen, Xingran Zhou, Liang Wu, Fei Tang, Sunqi Fan, Shangpin Peng, Zheng Ruan, Anran Zhang, Benyou Wang, Rui Yan, Ji-Rong Wen, Chengquan Zhang, Han Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.29486)  

**Abstract**: A central bottleneck for phone-use agents is that controllable, reproducible environments covering real mobile behavior are hard to build at scale. Existing mobile-agent benchmarks have made important progress on evaluation, but they do not by themselves provide a scalable way to construct many new phone-use environments. We present PhoneWorld, a reusable pipeline that converts real GUI trajectories and screenshots into controllable phone-use environments, executable tasks, automatic verifiers, and training rollouts. Rather than hand-building one mobile benchmark at a time, PhoneWorld uses real trajectories to recover which screens matter, how screens connect, which interactions must change environment state, and which user goals admit automatic verification. From these signals, it builds runnable mock Android apps backed by read-only app content and mutable state, then derives executable tasks, rule-based verifiers, and training rollouts from the same environments. In its current instantiation, PhoneWorld covers 34 apps across 16 domains, spanning common consumer mobile behaviors such as search, browsing, shopping, booking, media, and social interaction. Under a fixed training budget, replacing 10K steps from an auxiliary AndroidWorld corpus in an AndroidWorld-based baseline with broad PhoneWorld supervision improves all four evaluation benchmarks at once, raising HYMobileBench by 17.7 points, AndroidControl by 6.0 points, AndroidWorld by 14.7 points, and PhoneWorld by 52.5 points. We then study two additional scaling questions: increasing the amount of PhoneWorld supervision strongly improves PhoneWorld performance, and under a fixed PhoneWorld budget, expanding app coverage yields even larger gains. Overall, PhoneWorld shifts the focus from building one mobile benchmark at a time to scaling the supply of phone-use environments themselves. 

---
# Comparative Evaluation of Machine Translation Systems on Images with Text 

**Authors**: Blai Puchol, Sergio Gómez González, Miguel Domingo, Francisco Casacuberta  

**Link**: [PDF](https://arxiv.org/pdf/2605.29476)  

**Abstract**: This work presents a comparative evaluation of machine translation systems applied to images containing textual information, a task that lies at the intersection of computer vision and natural language processing. The study compares three main paradigms: modular pipelines that separate text detection, recognition, and translation; multi-modal large language models (MLLMs) capable of processing both image and text jointly; and an end-to-end model, Translatotron-V, which directly generates translated images. The modular systems employ state-of-the-art OCR (docTR) combined with multilingual LLMs such as Llama and EuroLLM, while the evaluated MLLMs include different configurations of Gemini 2.5. Experiments were conducted on parallel multilingual datasets covering multiple language pairs, with evaluation based on BLEU, chrF, and TER metrics. The results show that modular pipelines outperform the end-to-end approach, while MLLMs achieve the best overall performance, demonstrating superior flexibility and contextual understanding. These findings underscore the effectiveness of multi-modal reasoning for image-to-text translation and provide a solid foundation for future research on integrating visual understanding and language generation in multilingual settings. 

---
# MOOSE-Copilot: A Web-Based Interactive Assistant for Unified Exploratory and Fine-Grained Scientific Hypothesis Discovery 

**Authors**: Hongran An, Zonglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29475)  

**Abstract**: Large language models (LLMs) show remarkable potential in scientific hypothesis discovery. However, existing approaches face two critical limitations: they treat divergent exploratory ideation and convergent fine-grained refinement as isolated tasks, and they operate autonomously with little to no human guidance. We present MOOSE-Copilot, the first unified framework to bridge this abstraction gap through a formalized human-AI interaction (HAII) protocol. Our system empowers scientists to steer the generative process via three explicit signals: initial blueprints, inter-stage routing, and regenerative feedback. Quantitative evaluations demonstrate that injecting these structured expert signals significantly outperforms purely autonomous baselines, establishing a performance ceiling under oracle guidance. Furthermore, to democratize this paradigm, we develop an intuitive web-based interface featuring interactive tree visualization. This explicitly eliminates the steep learning curve of complex command-line agentic tools, empowering interdisciplinary researchers to directly leverage, visually orchestrate, and accelerate end-to-end scientific breakthroughs. 

---
# Kronecker Embeddings: Byte-Level Structured Token Representations for Parameter-Efficient Language Models 

**Authors**: Rohan Shravan  

**Link**: [PDF](https://arxiv.org/pdf/2605.29459)  

**Abstract**: Large language models route every input through a learned embedding table of shape |V| x d_model, consuming hundreds of millions to billions of trainable parameters at frontier scale. We introduce Kronecker Embeddings, a deterministic byte-level character-position factorization that replaces this table with a fixed encoder and a single learned projection, compatible with standard BPE tokenizers, eliminating 91--94% of input-side trainable parameters at frontier scale. We provide five contributions. First, a cross-model probe across six LMs (135M-671B parameters) shows trained input embeddings cluster typographic variants of the probe word far more than morphological relatives; Kronecker escapes this clustering at the embedding layer. Second, a controlled three-seed comparison on nanoGPT GPT-2 124M over 2.5B tokens of FineWeb-Edu shows Kronecker reaching 2.5 +- 0.2% lower validation loss than the BPE-tied baseline (gap 0.083 +- 0.007 nats, ~9% lower perplexity), needing ~1.43x fewer steps to reach BPE's converged loss. Third, a spelling-robustness probe over 110 clean/typo pairs shows Kronecker preserves the top-1 prediction on 55.5% of pairs vs. 47.3% for BPE (+8.2 pp) and lowers KL by 7.6%, winning or tying in 10 of 11 categories; a generation probe shows Kronecker echoes byte-novel strings and typos through generation where BPE forgets them. Fourth, BPE embedding norm drifts during training while Kronecker projection norm stays near 1.0, consistent with a stable representational target. Fifth, an on-the-fly runtime variant reconstructs embeddings from a 4.5 MB byte buffer rather than a 2.15 GB table at vocabulary 131,072, with 0.01--0.24% step-time overhead. Byte-level locality has a tradeoff: byte-similar but semantically distant pairs (compute/commute, nation/notion) cluster together, shifting disambiguation to early attention layers. 

---
# Adaptive Interviewing for Persona Simulation in LLMs: Evidence-Grounded Reasoning Improves Decision Alignment 

**Authors**: Ruoxi Su, Yuhan Liu, Jingyu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.29458)  

**Abstract**: Accurately simulating the decisions of a specific individual remains challenging for large language models (LLMs), partly because persona information is often provided as static descriptions that miss the values, experiences, and contextual cues needed for individual-level decision simulation. We propose an adaptive interview framework that gathers persona-relevant information through a structured three-stage dialogue: core questions, dynamic follow-ups, and a synthesized personality summary. Using the resulting interview transcripts, we evaluate whether LLMs can simulate participants' decisions in moral dilemma scenarios. We compare three conversational contexts -- Core-10 responses, the full interview dialogue, and a summarized persona representation. We find that adaptive interviewing functions less as a uniform accuracy booster and more as a selective grounding mechanism: follow-up-derived evidence is incorporated in around 40% of full-interview traces, and these follow-up-grounded predictions are more accurate than core-only grounded ones (45.5% vs. 39.3%). These findings highlight that richer persona context alone is insufficient: improvements arise only when models actually ground their decisions in user-specific evidence. 

---
# SkillBrew: Multi-Objective Curation of Skill Banks for LLM Agents 

**Authors**: Wentao Hu, Zhendong Chu, Yiming Zhang, Junda Wu, Ming Jin, Xiangyu Zhao, Yilei Shao, Yanfeng Wang, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2605.29440)  

**Abstract**: Retrieval-augmented LLM agents increasingly rely on curated skill banks: collections of reusable textual principles that guide decision making on complex tasks. Existing approaches typically expand these banks in an append-only fashion, continuously adding new skills without removing redundant, outdated, or harmful ones, resulting in inefficient and poorly curated repositories. In this paper, we formulate the skill bank curation as a constrained multi-objective problem: a desirable bank must be useful for the agent, diverse in its content, and provide good coverage of the query distribution. To this end, we introduce SkillBrew, a multi-objective curation framework that formalizes skill bank curation as Pareto-aware optimization under a utility constraint, and solves it via a bi-level propose-then-verify loop. We evaluate our approach on two public benchmarks. Our findings suggest that treating skill banks as objects of principled curation, rather than ever-growing append-only logs, is an important step toward building self-improving LLM agents. 

---
# FinGuard: Detecting Financial Regulatory Non-Compliance in LLM Interactions 

**Authors**: Huaixia Dou, Jie Zhu, Minghao Wu, Shuo Jiang, Junhui Li, Lifan Guo, Feng Chen, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29427)  

**Abstract**: As large language models (LLMs) are increasingly deployed in financial services, a single non-compliant interaction can expose institutions to regulatory penalties and direct consumer harm. Existing guard models are built around general harm taxonomies and overlook violations grounded in specific financial regulations. We address this gap with a regulation-driven pipeline that operates directly on regulatory documents, inducing a financial compliance risk taxonomy and synthesizing grounded training data without any predefined violation categories. Instantiating the pipeline on Chinese financial regulations, we release \textbf{FinGuard-Bench}, to our knowledge the first benchmark for financial regulatory compliance detection, with expert-annotated labels at both the query and response levels. We further train \textbf{FinGuard}, a financial compliance detection model built on Qwen3-8B and trained on the regulation-grounded data via supervised fine-tuning and self-play reinforcement learning. On FinGuard-Bench, FinGuard substantially outperforms all baselines, including dedicated guard models and much larger general-purpose LLMs such as Qwen3.5-397B-A17B and GPT-5.1. Furthermore, FinGuard also preserves general safety capabilities and adapts to unseen institution-specific policies using policy documents alone. We will publicly release the code, prompts, and resources used in this work on GitHub. 

---
# Learning Design Skills as Memory Policies for Agentic Photonic Inverse Design 

**Authors**: Shengchao Chen, Ting Shu, Sufen Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.29421)  

**Abstract**: Photonic crystal fiber (PCF) inverse design remains challenging because candidate geometries must satisfy coupled optical targets under expensive electromagnetic simulation. Existing pipelines improve surrogate prediction or one-shot parameter recommendation, but they do not accumulate reusable design knowledge across iterative trials. We formulate PCF inverse design as a memory-policy learning problem and propose SkillPCF, a closed-loop agent framework that combines a physics-guided memory skill bank, reinforcement-learned skill selection, and simulator-grounded skill evolution. We further construct a real-world dataset with 479 expert interaction traces (2,507 spans) and 553 memory-dependent evaluation queries covering dispersion engineering, loss optimization, and multi-objective design. Experiments across multiple LLM backbones and classical baselines show that SkillPCF achieves stronger design-quality and efficiency trade-offs under practical simulation budgets, demonstrating the effectiveness of our proposed memory-skill learning paradigm for physics-aware PCF inverse design. 

---
# Beyond Bilingual Transfer: Multilingual Code-Switching in Instruction Tuning 

**Authors**: Shunta Asano, Jeonghun Baek, Toshihiko Yamasaki  

**Link**: [PDF](https://arxiv.org/pdf/2605.29414)  

**Abstract**: Recent studies have shown that code-switching data (CSD), in which multiple languages are mixed within the same context, can improve cross-lingual transfer and multilingual alignment in large language models (LLMs). However, existing studies primarily focus on bilingual transfer between English and a target language, leaving multilingual settings involving three or more languages largely unexplored. In this work, we investigate multilingual code-switching instruction tuning across four languages: English, Japanese, Korean, and Chinese. We evaluate multilingual understanding on Belebele. Our experiments show that simple sentence-level multilingual CSD consistently improves average multilingual performance across all four languages, indicating that multilingual code-switching can be effective beyond bilingual transfer settings. 

---
# Revisiting Observation Reduction for Web Agents: Comprehensive Evaluation with a Lightweight Framework 

**Authors**: Masafumi Enomoto, Ryoma Obara, Haochen Zhang, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2605.29397)  

**Abstract**: HTML observations in LLM-based web agents are extremely long, and while many reduction methods have been proposed, it remains unclear which methods reduce overall agent latency while maintaining performance. The main obstacle is the high cost of end-to-end evaluation: in our experiments, evaluating 11 methods across 32 configurations on 33 tasks of WorkArena L1 required 232.4 cumulative hours. To address this, we propose a lightweight evaluation framework based on the Minimal Failure Set (MFS), the minimal set of HTML elements whose removal causes task failure. We define coverage as the fraction of instances in which a reduction method fully retains the MFS, which serves as a proxy metric that requires neither web access nor LLM inference. We validate that coverage strongly correlates with end-to-end success rate, with over 100$\times$ speedup in cumulative evaluation time on both benchmarks. Using this framework, we find that extractive HTML reduction methods require either high computation cost or domain-specific optimization to reduce agent latency while maintaining performance. Building on this, we optimize a pruning program on MFS training data, achieving 2.2$\times$ faster per-step latency on WorkArena L1 while retaining 84\% of the original success rate, and 3.1$\times$ faster on WebLinx while retaining 89\%. 

---
# BrahmicTokenizer-131K: An Indic-Capable Drop-In Replacement for o200k_base 

**Authors**: Rohan Shravan  

**Link**: [PDF](https://arxiv.org/pdf/2605.29379)  

**Abstract**: We present BrahmicTokenizer-131K, a 131,072-vocabulary byte-level BPE tokenizer that closes the Brahmic compression gap at the 131K-vocabulary class while preserving the English, EU-language, and code compression of OpenAI's o200k_base. We construct it through a two-stage retrofit: (1) a script-prune crop that reduces 200,019 tokens to 131,072 by removing nine out-of-scope writing systems, and (2) a surgical retrofit of 2,372 corpus-dead vocabulary slots determined by linear-programming allocation across nine Brahmic Unicode blocks. The pre-tokenizer, decoder, and inherited merge rules are unchanged from o200k_base, making BrahmicTokenizer-131K a drop-in replacement at the tokenizer interface.
On 27 million documents of public Indic pretraining text (2.84 billion words, 46.21 GB), BrahmicTokenizer-131K produces 26.7% fewer tokens than Mistral-Nemo Tekken / Sarvam-m at the same vocabulary budget, with per-language savings of 15.79% (Tamil) to 76.79% (Odia, a 4.31x compression ratio). The Odia advantage is mechanistically explained by Tekken/Sarvam-m containing zero Oriya-block tokens; our surgery added 725. On non-Indic content, BrahmicTokenizer-131K matches o200k_base's English fertility (1.235 vs 1.232 tokens/word) and beats Tekken/Sarvam-m by 4.0-14.2% on HumanEval, MBPP, and GSM8K. Across our 14-tokenizer benchmark, it is the only tokenizer simultaneously competitive on Brahmic, English, EU, code, and math at the 131K budget. Specialist tokenizers at other vocab classes (Sarvam-30B, Sarvam-1, MUTANT-Indic) achieve better Indic compression at the cost of non-Indic performance: Sarvam-1's English fertility is 15.9% worse and its code/math compression 26-33% worse than ours. We release the artifact under Apache 2.0 at this https URL. 

---
# SURGENT: A Surgical Multi-Agent Assistance System Across the Perioperative Workflow 

**Authors**: Dongsheng Shi, Yue Li, Xin Yi, Yongyi Cui, Huawei Feng, Linlin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29368)  

**Abstract**: The intricate nature of modern surgical care necessitates intelligent systems that can synthesize extensive patient records, support collaborative decision-making, and provide transparent, auditable reasoning across the entire perioperative workflow. Although web-based Large Language Models (LLMs) possess advanced reasoning capabilities, they are ill-equipped for surgical applications due to critical limitations: input length constraints, incomplete memory management, and limited traceability. To address this issue, we present SURGENT, a surgical multi-agent assistance system that combines a Tree-of-Thought planner, multi-department collaboration agents, and retrieval-augmented reasoning with clinical guidelines and biomedical literature. SURGENT features a novel memory design that manages both long-term patient histories and short-term working summaries, enabling more complete, contextualized, and consistent reasoning. Experimental evaluations across five key perioperative tasks - case analysis, surgical plan simulation, safety monitoring, complication risk assessment, and rehabilitation guidance - show that SURGENT outperforms baseline LLMs and existing medical multi-agent frameworks, yielding recommendations more closely aligned with patient histories. Ablation studies further highlight the advantage of DeepSeek as a locally deployable backbone model, enabling privacy-preserving deployment without reliance on centralized services. These results position SURGENT as a practical and trustworthy advancement toward intelligent, equitable, and secure surgical assistance systems. 

---
# Attention Asymmetry in AI Layoff Discourse on X: A Computational Analysis of Capital vs Labour Amplification 

**Authors**: Joy Bose  

**Link**: [PDF](https://arxiv.org/pdf/2605.29367)  

**Abstract**: When workers lose jobs to AI-driven restructuring, two very different conversations happen on X (formerly Twitter) at the same time. Tech executives and AI researchers talk about productivity, transformation, and opportunity. Laid-off workers and labour critics talk about job loss, uncertainty, and fear. This paper asks a simple question: which conversation gets more reach? We report three studies using two collection methods and 763 tweets from 20 named public accounts. Study 1 used keyword-based collection (n=392) and found no significant difference between corpora (p=0.891), revealing that keyword search is too noisy for this task. Study 2 used account-based collection (n=96) and found a 3.12x mean amplification advantage for capital discourse over labour discourse (p=0.000003, Cohen's d=0.555). Study 3 combined both methods (n=763) and confirmed the finding at 4.18x mean and 10.77x median amplification ratio (p<0.000001). Critically, after normalising for follower count, the asymmetry persists at 2.69x (p=0.000009, Cohen's d=0.491), demonstrating that the effect is not simply a consequence of capital accounts having larger audiences. The finding is robust across all tested amplification metric weightings. We introduce the Amplification Ratio and Amplification Normalisation Index as simple metrics for measuring platform-level discourse inequality. A cross-platform replication on Reddit (n=647 posts) did not replicate the finding, suggesting the asymmetry may be specific to X's account-based amplification architecture. We discuss the methodological implications for cross-platform discourse analysis. 

---
# Casual as an Anchor: Resolving Supervision Misalignment in Formality Transfer Dataset 

**Authors**: Hyojeong Yu, Hyukhun Koh, Minsung Kim, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2605.29365)  

**Abstract**: Formality transfer is commonly framed as a symmetric bidirectional task between informal and formal registers. We argue that this framing conceals a supervision
design flaw in existing benchmarks such as GYAFC: binary human rewrites encode relative stylistic shifts rather than absolute human notions of formality.
Consequently, models learn to generate pseudo-formal outputs that satisfy benchmark labels while failing to produce genuinely formal language. We quantify this
misalignment by re-evaluating benchmark formal labels under a human-aligned definition of formality, revealing substantial discrepancies that propagate to
consistent informal-to-formal failures across model families. To address this issue, we reconceptualize formality transfer as a graded dimension rather than a
binary attribute. We introduce a three-level spectrum: informal, casual, and formal, where casual serves as an explicit intermediate state that clarifies
supervision signals. Based on this framework, we introduce 3LF, a dataset providing parallel supervision across all three levels. Training on 3LF substantially
reduces informal-to-formal failures and improves alignment with human perception. For example, GPT-4.1-nano improves from 0.06 to 0.88 F1 in the informal-to-
formal direction despite 3LF being significantly smaller than GYAFC. We further demonstrate that these gains cannot be reproduced through in-context learning
alone and provide qualitative analyses of ambiguity-driven errors and meaning distortions. Overall, our findings demonstrate how supervision design shapes
stylistic alignment and highlight the importance of alignment-aware benchmark construction in controllable text generation. 

---
# Draft-OPD: On-Policy Distillation for Speculative Draft Models 

**Authors**: Haodi Lei, Yafy Li, Haoran Zhang, Shunkai Zhang, Qianjia Cheng, Xiaoye Qu, Ganqu Cui, Bowen Zhou, Ning Ding, Yun Luo, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.29343)  

**Abstract**: Speculative decoding accelerates large language model inference by pairing a target model with a lightweight draft model whose proposed tokens are verified in parallel. A common way to build draft models, like EAGLE3 or DFlash is supervised fine-tuning (SFT) on target-generated trajectories. However, we observe that SFT quickly plateaus: the draft model's acceptance length on test data stops improving. The reason is an offline-to-inference mismatch: In SFT, the drafter learns from fixed target-generated trajectories, whereas during speculative decoding it is evaluated on blocks proposed under its own policy. This motivates on-policy distillation (OPD), where the target model supervises the drafter on draft-induced states. Yet OPD remains difficult for draft models, as they cannot reliably roll out complete sequences independently, whereas target-assisted generation makes the collected sequences follow the target distribution and thus eliminates the on-policy signal. We therefore propose Draft-OPD, which uses target-assisted rollout for stable continuations and replays drafting from the verification-exposed error positions. This allows the drafter to learn from target feedback on both accepted and rejected proposals, focusing training on the draft-induced errors that limit speculative acceptance. Experiments show that Draft-OPD achieves over $5\times$ lossless acceleration for thinking models across diverse tasks, improving over EAGLE-3 and DFlash by 23\% and 13\%. 

---
# A Study on Question-Answer Dataset for LLM Safety Evaluation with a Focus on Illegal Activities 

**Authors**: Kenji Imamura, Masao Ideuchi, Atsushi Fujita  

**Link**: [PDF](https://arxiv.org/pdf/2605.29340)  

**Abstract**: In this paper, we discuss question-answer dataset for LLM safety evaluation, with a focus on illegal activities. Specifically, on the basis of manual analysis of AnswerCarefully, we introduce several additional information, methods for creating question-answer examples, and a rubric for evaluating LLM-generated responses. The outcomes of this study are intended to be shared with the "JAI-Trust" project. 

---
# Enhancing Factuality through Consensus and Consistency in Summarization Using Minimum Bayes Risk Decoding 

**Authors**: Riza Setiawan Soetedjo, Yusuke Sakai, Hidetaka Kamigaito, Jingun Kwon, Manabu Okumura, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2605.29336)  

**Abstract**: Improving the quality of model-generated summaries, especially factuality, the accuracy of a summary with respect to its source content, remains a challenge. While reranking could select the optimal output from multiple generated candidates, it is limited to only using the source as guidance, resulting in unreliable summaries. To address this limitation, we propose ConSUM that reranks candidate summaries by considering two factors: consistency to the source document and consensus among the other candidates. Consensus is established using Minimum Bayes Risk (MBR) decoding over the set of generated summaries, while ensuring consistency by employing factuality-aware metrics that compare the summary against the source. Rigorous testing demonstrates that our system is competitive with existing methods, with human evaluations further confirming that its generated summaries are preferred over those from other systems. Our code is available at this https URL . 

---
# Reasoning-preserved Efficient Distillation of Large Language Models via Activation-aware Initialization 

**Authors**: Junlin He, Yihong Tang, Tong Nie, Guilong Li, Binyu Yang, Jinxiao Du, Lijun Sun, Wei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.29327)  

**Abstract**: Efficient Distillation (EDistill) compresses large language models (LLMs) by structured pruning parameters and tuning lightweight modules with high training efficiency. Although these EDistilled LLMs achieve state-of-the-art (SOTA) performance on general ability benchmarks relative to similarly sized LLMs, we identify a severe degradation in their multi-step reasoning ability, which we term reasoning collapse. We systematically analyze the geometric origins of reasoning collapse and show that the SOTA EDistill method based on width-reducing projection matrices suffers from eRank collapse, in which the effective rank (eRank) of hidden representations drops. We theoretically explain how singular values of randomly initialized projection matrices become unevenly distributed, leading to eRank collapse and thus token indistinguishability. To address this issue, we propose RED (Reasoning-preserved Efficient Distillation) for LLMs, which introduces activation-aware initialization to initialize projection matrices as channel-selection matrices, thus theoretically mitigating eRank collapse. Experiments on Llama and Qwen series demonstrate that RED substantially recovers reasoning while maintaining high training efficiency and SOTA general ability. 

---
# STAMP: Training Explicit Memory for Mobile GUI Agents in Controllable and Scalable Virtual Environments 

**Authors**: Junyang Wang, Haiyang Xu, Xi Zhang, Zhaoqing Zhu, Ming Yan, Jieping Ye, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29324)  

**Abstract**: Mobile GUI agents excel at immediate reactive control but frequently fail in realistic, long-horizon tasks that require memory. This failure stems from a fundamental conflict between limited context windows and token-heavy screenshots. To save the limited context, agents must progressively discard older visual history, permanently losing crucial transient information. Furthermore, existing action-centric datasets fail to teach agents what or when to explicitly memorize, and augmenting static real-world data is prohibitively expensive and lacks interactive verification. To resolve this, we present STAMP, a framework that trains explicit memory in mobile agents through controllable virtual environments, where deterministic memory variables are programmatically injected into synthesized tasks to control what must be memorized, when it should be encoded, and when it must later be retrieved, thereby producing verifiable supervised data at scale and enabling online reinforcement learning through environment-driven reward feedback. Evaluated on our newly introduced Memory-World benchmark, the resulting Stamp-GUI agent achieves state-of-the-art performance among GUI-specialized models and sets a new high watermark on our Memory-World benchmark, demonstrating exceptional memory accuracy and task resilience while maintaining strong general mobile navigation capabilities. 

---
# Rethinking Stepwise Model Routing: A Cost-Efficient Table Reasoning Perspective 

**Authors**: Shenghao Ye, Yuxiang Wang, Yu Guo, Dong Jin, Shuangwu Chen, Jian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29319)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong performance on table reasoning tasks but incur substantial inference cost due to long reasoning traces. Stepwise model routing mitigates this issue by dynamically assigning reasoning steps to smaller or larger models. However, stepwise model routing for table reasoning remains underexplored. Through empirical analysis, we find that reasoning steps involving tables contain two types of tokens with distinct uncertainty distributions: table tokens grounded in table structure, such as cell values and headers, and text tokens representing surrounding natural-language reasoning. The uncertainty of both token types is correlated with the risk that the model makes an error in the next reasoning step. However, existing methods fail to model them separately, leading to suboptimal routing decisions. To address this, we propose EcoTab, a table-aware stepwise routing framework for efficient table reasoning. At each reasoning step, EcoTab separately estimates the uncertainties of table tokens and text tokens, maps them to next-step failure risks for the small model, and combines the two risks for routing. Experiments on multiple table reasoning benchmarks show that EcoTab consistently outperforms strong baselines and achieves a better balance between accuracy and efficiency. 

---
# FoRA: Fisher-orthogonal Rank Adaptation for Parameter-Efficient Fine-Tuning 

**Authors**: Juneyoung Park, Seongbae Lee, Han-Sang Lee, Kyuho Lee, Minjae Kim, Seungheon Hyeon, Kiduk Kwon, Seongwan Kim, Jaeho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.29317)  

**Abstract**: Parameter-efficient fine-tuning(PEFT) has largely focused on LoRA and its accuracy-oriented variants, leaving the original goal of reducing trainable parameters has receivedcomparatively little attention. We introduce FoRA, which revisits this goal by reducing the number of adapted layers rather than adapter rank. FoRA selects task-informative layers via a single-pass diagonal Fisher score (under 1% of training cost) and trains the LoRA down-projection at selected layers on the Stiefel manifold, preserving column orthonormality and effective rank. FoRA consistently outperforms LoRA and DoRA at half their parameter budget, and falls within 0.7-0.8 accuracy points of AdaLoRA at one-quarter its parameter count, across five LLaMA-family backbones. Cross-architecture experiments on twelve backbones from the LLaMA, Qwen3, and Gemma families confirm consistent gains from 270M to 32B parameters. The two components combine super-additively: Fisher selection alone matches rank reduction at the same budget, while the Stiefel constraint provides the decisive additional gain. 

---
# PatchBoard: Schema-Grounded State Mutation for Reliable and Auditable LLM Multi-Agent Collaboration 

**Authors**: Shuyu Zhang, Yaqi Shi, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29313)  

**Abstract**: LLM multi-agent systems often coordinate through natural-language dialogue or loosely structured shared memory, making intermediate state difficult to validate, attribute, and audit. We introduce PatchBoard, a schema-grounded collaboration architecture that replaces inter-agent dialogue with validated JSON Patch mutations over a shared structured state. An Architect agent constructs a task-specific schema and workflow rules, while a deterministic kernel validates each proposed state mutation against schema constraints, role-specific write contracts, and runtime invariants before committing it transactionally. On 630 matched ALFWorld episodes, PatchBoard achieves an 84.6% success rate, compared with 30.8% for LangGraph and 61.6% for Flock, while reducing tokens per successful task to 45.5k, compared with 368.3k and 64.2k, respectively. 

---
# GrepSeek: Training Search Agents for Direct Corpus Interaction 

**Authors**: Alireza Salemi, Chang Zeng, Atharva Nijasure, Jui-Hui Chung, Razieh Rahimi, Fernando Diaz, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2605.29307)  

**Abstract**: Large Language Model (LLM) search agents have shown strong promise for knowledge-intensive language tasks through multiple rounds of reasoning and information retrieval. Most existing systems access information using a retriever that takes a keyword or natural language query and returns a ranked list of documents using an index of pre-computed document representations. In this work, we explore a complementary perspective in which the search agent treats the corpus itself as the search environment and finds evidence by issuing executable shell commands. We introduce GrepSeek, an optimized direct corpus interaction (DCI) search agent that trains a compact search agent to find, filter, and compose evidence from large text corpora. To address the instability of learning behavior directly with reinforcement learning on large corpora, we propose a two-stage training pipeline. First, we construct a cold-start dataset using an answer-aware Tutor and answer-blind Planner to generate verified, causally grounded search trajectories. Second, we refine the initialized policy with Group Relative Policy Optimization (GRPO), allowing the agent to improve its task-oriented search behavior through direct interaction with the corpus. To make DCI practical at scale, we further use a semantics-preserving sharded-parallel execution engine that accelerates shell-based retrieval by up to $7.6\times$ while preserving byte-exact equivalence with sequential execution of the shell command. Experiments across seven open-domain question answering benchmarks show that GrepSeek achieves the strongest overall token-level $F_1$ and Exact Match. Our analysis also highlights the limitations of purely lexical interaction on queries with substantial surface-form variation, suggesting DCI as a practical and competitive method for search agents that can complement existing retrieval paradigms in the real world. 

---
# MusTBENCH: Benchmarking and Advancing Temporal Grounding in Music LLMs 

**Authors**: Daeyong Kwon, Qiyu Wu, Shinobu Kuriya, Junghyun Koo, Shuyang Cui, Zhi Zhong, Wei-Hsiang Liao, Hiromi Wakaki, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2605.29300)  

**Abstract**: Recent Large Audio-Language Models (LALMs) have demonstrated promising abilities in understanding musical content. However, whether their responses are grounded in the correct temporal regions of the audio remains underexplored. This limitation is particularly critical for music understanding, where key information often occurs as temporally localized events, such as instrument entries and rhythmic transitions. To address this gap, we introduce MusTBENCH, a music-expert-validated benchmark designed to evaluate temporal grounding in LALMs through five temporally grounded question-answering tasks. To further improve temporal grounding in existing models, we propose MusT, a novel four-stage temporal optimization recipe spanning music encoder adaptation, LLM adaptation, LLM supervised fine-tuning, and RL-based optimization. Experiments on MusTBENCH show that existing LALMs struggle with precise temporal grounding, while MusT brings significant improvements over strong baselines. These results establish temporal grounding as a key missing capability in current LALMs and position MusTBENCH as a challenging benchmark for future research in temporally grounded music understanding. 

---
# Accommodation Goes Both Ways: Studying Linguistic Convergence Between Humans and Language Models 

**Authors**: Terra Blevins  

**Link**: [PDF](https://arxiv.org/pdf/2605.29278)  

**Abstract**: As LLMs become increasingly integrated into daily life, understanding how their presence will shape human linguistic behavior is an open question. We present a large-scale study of linguistic convergence in human-LLM dialogue, examining how humans and LLMs accommodate each other's linguistic style during multi-turn conversations. Using an asymmetric convergence metric on WildChat, a corpus of real-world ChatGPT transcripts, we find that while LLMs significantly overconverge toward their users on both function word and open-class features across eight languages, human convergence rates in this setting are broadly consistent with human-human baselines. These findings suggest that accommodation in human-LLM dialogue is asymmetric: while LLMs dramatically overfit to their users' style, humans linguistically accommodate LLMs no differently than they would another person. 

---
# Prompt-Level Reward Specifications for Open-Ended Post-Training 

**Authors**: Zijun Weng, Xiaohui Hu, Shuangyong Song, Yongxiang Li, Kaidong Yu, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29275)  

**Abstract**: Open-ended post-training benefits from rewards that make prompt-specific success conditions explicit, rather than relying only on post-hoc scalar scores. In instruction following, writing, and decision-support tasks, response quality depends on local requirements, holistic preferences, and explicit constraints, but existing reward methods often leave these criteria implicit or cover only narrowly verifiable cases. We propose a prompt-level reward specification framework that separates reward specification from reward computation. Given only prompts, our framework constructs reusable task-adaptive rubrics and executable hard-constraint checkers offline, making reward criteria explicit before training and reusable across rollouts. At scoring time, artifact-anchored rubric and code scores are combined with an independent global score for residual holistic quality, yielding a normalized hybrid reward over requirement satisfaction, holistic quality, and deterministic constraints. The framework requires no human preference annotations, reference answers, or a separately trained reward model. Experiments show that the resulting reward improves offline RM-style response ranking and supports online reinforcement learning across multiple open-ended benchmarks. Ablations further show that rubrics, global scoring, and executable verification provide complementary supervision. 

---
# Learnable Assessment Skills for LLM-based Automated Scoring: Rubric Construction via Iterative Optimization 

**Authors**: Yun Wang, Xin Xia, Xuansheng Wu, Xiaoming Zhai, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.29274)  

**Abstract**: LLM-based automated scoring approaches near-human performance, but scaling to new tasks remains bottlenecked by the per-item human configuration of upstream stages such as rubric construction. Human experts bypass this bottleneck through evaluation heuristics developed over extensive practice. We ask whether LLMs can learn similar heuristics directly from scoring experience, and formalize this as the concept of assessment skills: item-independent natural-language procedural knowledge that guides LLMs through specific stages of the scoring workflow. Focusing on rubric construction as a first instantiation, we propose an iterative framework that decomposes a skill into a fixed scaffold and learnable item-agnostic rules, refining the rules through LLM-driven diagnosis of scoring errors and validation-gated selection. The framework requires no expert-written rubric. On all ten ASAP-SAS items, optimized skills substantially improve LLM-based scoring and frequently surpass the dataset-provided expert rubric. Cross-item transfer experiments further reveal that learned skills capture both generalizable and item-specific patterns. 

---
# Compute Allocation in Evolutionary Search: From Depth-Breadth to Multi-Armed Bandits 

**Authors**: Sixue Xing, Haoyu He, Kerui Wu, Zhuo Yang, Haozheng Luo, Tianfan Fu, Aarthy Nagarajan  

**Link**: [PDF](https://arxiv.org/pdf/2605.29268)  

**Abstract**: LLM-guided evolutionary search (Evolve systems) has reached state-of-the-art results on mathematical and combinatorial tasks, yet most existing systems report only the best of many runs and leave the run-to-run distribution undocumented. We ask how a fixed budget of LLM calls should be allocated, and how reliably a single run reaches the reported numbers. Sweeping the depth-breadth grid over five models and three tasks, we identify two empirical regularities: a fitness-compute envelope along which capability ordering largely collapses on effective FLOPs, and a bilinear depth-breadth fit with task-specific interaction; both are gated by model-task capability. Motivated by these regularities, we propose BaSE (Bandit-based Self-Evolving), a multi-armed bandit that allocates LLM calls across parallel trajectories. Without changing the model, prompt, or evaluator, BaSE improves mean fitness by 12.3% over the strongest island-protocol baseline across 8 (model, task) cells, with the largest gains on high-variance settings: a reliability gain from allocation alone. 

---
# DynSess: Dynamic Session-Level Evaluation and Optimization Framework for Role-Playing Agents 

**Authors**: Rongsheng Zhang, Jiji Tang, Junnan Ren, Zuyi Bao, Weijie Chen, Ruofan Hu, Zhou Zhao, Tangjie Lv, Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29256)  

**Abstract**: Role-playing with large language models is fundamentally a session-level task, requiring agents to sustain character identity and interaction quality across extended multi-turn conversations. Yet existing evaluation and optimization methods remain largely turn-level, failing to capture long-horizon quality. We propose DynSess, a unified session-level framework for role-playing agents. DynSess-Eval scores complete dialogue sessions via rubrics targeting long-horizon behaviors. Leveraging its session-level rewards, we construct high-quality training trajectories through multi-turn lookahead search and train DynSess-Character with two complementary variants: DSPO (off-policy) and GSRPO (on-policy). Experiments show that DynSess-Eval aligns with human judgments substantially better than prior evaluators, and blind human evaluation further shows that DynSess-Character matches the strongest character model despite using substantially fewer parameters, while maintaining strong role consistency and interactive ability. Our dataset and code will be released to facilitate future research. 

---
# OmniRetrieval: Unified Retrieval across Heterogeneous Knowledge Sources 

**Authors**: Jinheon Baek, Soyeong Jeong, Sangwoo Park, Woongyeong Yeo, Minki Kang, Patara Trirat, Heejun Lee, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29250)  

**Abstract**: Real-world information needs require access to structurally diverse knowledge sources, from unstructured text and relational tables to knowledge graphs and property graphs. Existing retrievers, however, operate over one source at a time under a fixed query language, leaving the broader landscape of available knowledge fragmented behind incompatible interfaces. A natural attempt at unification would collapse these sources into a shared space, but this erases the structural affordances (such as schemas, ontologies, compositional operators) that give each source its expressive power. Effective retrieval over diverse knowledge, therefore, requires not homogenization but an overarching layer that meets each source on its own terms. To achieve this, we present OmniRetrieval, a framework that takes any natural-language query, identifies appropriate knowledge sources, and dispatches source-native queries to their native execution engines. Across an extensive benchmark spanning 13 datasets and 309 distinct knowledge bases over text, relational, and graph-structured sources, OmniRetrieval exceeds single-source baselines, demonstrating that it can serve as a general-purpose interface to the heterogeneous sources while preserving the structural distinctions that make each source valuable. 

---
# Wait! There's a Way Out: A Decision Mechanism for Forecasting Conversational Derailment 

**Authors**: Laerdon Kim, Vivian Nguyen, Cristian Danescu-Niculescu-Mizil  

**Link**: [PDF](https://arxiv.org/pdf/2605.29243)  

**Abstract**: Forecasting conversational derailment is the task of predicting, as the conversation unfolds, whether it will eventually derail into personal attacks. Since forecasting models operate in an online fashion, they must decide whether to "trigger" an alert after each utterance--for example, to notify participants or a moderator that the conversation is at risk of derailing. Existing approaches make this decision solely based on the estimated likelihood of derailment given the preceding utterances, implicitly assuming that the conversation's future trajectory is fixed. As a result, they ignore the possibility of future recovery and incur an unnecessarily high rate of false positives.
In this work we propose a method for decoupling the decision to trigger from derailment likelihood estimation. Our approach is inspired by the first human baseline on this task, which shows that humans achieve dramatically lower false positive rates by selectively deferring their decision to trigger when they anticipate that tension is likely to subside. We operationalize this insight with a deferral mechanism that uses forward-looking simulations to assess whether a tense moment admits plausible paths to recovery. Incorporating this mechanism into a state-of-the-art forecasting model substantially reduces false positives without sacrificing forecasting accuracy. More broadly, this work highlights the value of treating decision-making as a first-class component of forecasting systems. 

---
# Relevance as a Vulnerability: How Web Retrieval Degrades Safety Alignment in LLM Agents 

**Authors**: Aditya Nawal, Manit Baser, Mohan Gurusamy  

**Link**: [PDF](https://arxiv.org/pdf/2605.29224)  

**Abstract**: AI agents augment large language models with external tools such as web retrieval, enabling grounded and up-to-date responses. However, incorporating external content into the generation pipeline can weaken the safety alignment mechanisms that govern model outputs. Prior work shows that enabling retrieval in agents increases compliance with harmful requests. We introduce AgentREVEAL, a diagnostic framework for analyzing retrieval-induced safety degradation in LLM agents. The framework examines two axes: how retrieval is integrated into the agent pipeline and the properties of the retrieved content. Along the integration axis, we find that binding tool invocation and response generation in a single step amplifies harmful outputs. Along the content axis, we uncover the Safe Source Paradox: even oppositional or safety-oriented sources, such as pages containing warnings or risk disclaimers, can increase harmful compliance by an average of 25% compared to the no-retrieval baseline. Finally, we show that relevance acts as a shared activation condition for both vulnerabilities. Similar patterns appear on frontier closed models, and harmful compliance remains elevated under several representative pipeline interventions, with some agents also entering this regime under autonomous retrieval. Because relevance is also what makes retrieval useful, these results expose a safety-utility trade-off for retrieval-enabled agents. We introduce HarmURLBench, a benchmark containing 1,405 real-world URLs paired with 320 harmful behaviors to support future evaluations. 

---
# Slogans or Stance? A Label-Light Diagnostic for Entrepreneurial-Discourse Measurement on Chinese SOE Speeches 

**Authors**: Ting Gong, Shangquan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.29188)  

**Abstract**: Dictionary methods, topic models, and embedding-similarity scorers are widely used in CSS and management research to measure constructs such as "entrepreneurial spirit" in corporate speeches. We contribute a label-light measurement diagnostic for such instruments rather than a new extraction model. On a corpus of 80 speeches by leaders of centrally administered Chinese state-owned enterprises, we exploit a natural experiment of 24 same-company different-speaker pairs and 5 same-company same-speaker pairs to test whether a method's per-document indices vary with leader identity holding firm constant. LDA fails (Cohen d=0.20, 95% CI [-0.72, 1.20]); a dictionary scorer reaches d=0.81 and a Chinese sentence encoder d=0.65 on doc-vector distances of order 10^-3. A zero-shot 9B open-weight LLM (Qwen3.5:9b) raises paired-contrast d to 1.09 (exact permutation p1=0.034). We downgrade three claims accordingly: gold F1 measures consistency with the LLM's own prompt rule rather than external construct recovery; doc-level style residualisation cuts the LLM's d to 0.43 (p1=0.22), so roughly half of the effect is consistent with leader idiolect; and a confidence-weighted calibration trades Delta for variance with an auto-mined slogan lexicon near-inert in ablation. We release the 2,190-segment scored corpus, the 170-paragraph pilot, the slogan lexicon, two-family LLM scores, and the evaluation harness. 

---
# UA-Legal-Bench: A Benchmark for Evaluating Large Language Models on Ukrainian Legal Reasoning 

**Authors**: Volodymyr Ovcharov  

**Link**: [PDF](https://arxiv.org/pdf/2605.29170)  

**Abstract**: Legal NLP benchmarks are overwhelmingly English-centric, leaving failure modes in morphologically rich, non-Latin-script languages undetected. We introduce UA-Legal-Bench, a five-task benchmark for evaluating large language models on Ukrainian legal reasoning, built from the Unified State Register of Court Decisions (EDRSR) -- one of the world's largest open judicial corpora (99.5 million decisions). The benchmark comprises: (1) case-type classification (4 classes, n=2,000), (2) judgment form classification (4 classes, n=2,000), (3) case-outcome prediction (6 classes, n=800), (4) legal norm extraction (n=1,794), and (5) cause category prediction (22 classes, n=1,871). We evaluate 11 LLMs (3B--675B) from five families under zero-shot and 3-shot prompting via AWS Bedrock with 158K API calls. Our results reveal sharply task-dependent few-shot effects: few-shot prompting improves judgment form classification by up to +38.6 pp but has mixed effects on outcome prediction. We show that accuracy is misleading on imbalanced legal tasks: the model with highest COP accuracy (62%) is a majority-class predictor (macro-F1: 23%), while the genuinely best model scores only 44% macro-F1. Within-family scaling analysis reveals that 8B models can match frontier performance on surface-level tasks but scaling thresholds vary dramatically across families. We release all data, prompts, and model predictions. 

---
# SafeRx-Agent: A Knowledge-Grounded Multi-Agent Framework for Safe and Explainable Medication Recommendation 

**Authors**: Xinyu Wang, Hanwei Wu, Zhenghan Tai, Sicheng Lyu, Qincheng Lu, Ziyu Zhao, Jijun Chi, Jingrui Tian, Xiao-Wen Chang, Ziyang Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.29146)  

**Abstract**: Medication recommendation predicts medications for patient visits, but existing methods still face two key challenges. At the model level, traditional drug recommendation methods only predict structured drug codes with limited evidence grounding, while LLM agents can use richer clinical context but may lack safety verification and traceability. At the task level, existing benchmarks often use broad medication categories, which ignore subgroup-level safety differences and can lead to risk overestimation. We introduce the first fine-grained medication recommendation setting based on fourth-level ATC code generation. We propose Safe Prescription Agent (SafeRx-Agent), a knowledge-grounded multi-agent framework that uses patient context, external clinical knowledge, and safety verification to recommend traceable medication sets. Experimental results on MIMIC-III and MIMIC-IV datasets show that SafeRx-Agent improves fine-grained medication prediction accuracy while controlling drug interactions, contraindications, and medication set size. 

---
# Same Question, Different Source, Different Answer: Auditing Source-Dependence in Medical Multi-Source RAG 

**Authors**: Yubo Li, Rema Padman, Ramayya Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2605.29084)  

**Abstract**: A retrieval-augmented generation (RAG) system deployed over a multi-author institutional corpus can give a different answer to the same question depending on which source it retrieves -- a failure mode the dominant single-gold-answer paradigm cannot diagnose. We argue that source-dependence is a missing axis of NLP evaluation, and that auditing it means shifting the unit of evaluation from answer correctness to the inter-source relationship. We make this concrete in transplant patient education, where institutional sources demonstrably disagree, releasing three artefacts: TransplantQA, a benchmark of real patient questions, each answered by grounding generation in multiple institutional handbooks as candidate sources; HERO-QA, a hierarchical retrieval strategy that grounds and audits each answer; and a structured-output judge that scores inter-source relationships on a validated 5-label taxonomy. At scale, better retrieval reveals far more disagreement than prior estimates suggested -- understating its prevalence, not its intensity. The framework is domain-agnostic and transfers to legal and educational RAG: measuring source-dependence is a responsibility for deployed multi-source NLP generally. 

---
# Structured Prompt Optimization Meets Reinforcement Learning for Global and Local Interpretability over Complex Text 

**Authors**: Tianyang Zhou, Wenbo Chen, Pierre Jinghong Liang, Leman Akoglu  

**Link**: [PDF](https://arxiv.org/pdf/2605.29076)  

**Abstract**: LLMs have advanced text classification, yet existing paradigms face a trade-off: supervised (label only) fine-tuning is scalable but offers limited reasoning on complex text and lacks broader model transparency, while discrete prompt optimization offers human-readable instructions but struggles with performance and scalability. We introduce eXTC (eXplainable Text Classifier) with three progressive stages: (1) learning a Standard Operating Procedure (SOP, or rulebook) in natural language via a new Structured Prompt Optimization algorithm; (2) SOP-grounded reasoning distillation from a large teacher LLM into a compact LM; and (3) expanding reasoning capabilities beyond the initial SOP via reinforcement learning. This design enables eXTC to provide (i) fast inference via a compact LM, with (ii) inference-time local reasoning traces, alongside a global, modular explanation of its learned domain rules, while (iii) significantly outperforming existing paradigms across diverse benchmarks in both classification performance and explanation quality, with stage-by-stage gains. 

---
# Analyzing Persona Effects in Generated Explanations from Multimodal LLM Agents in Urban Perception 

**Authors**: Neemias da Silva, Myriam Delgado, Rodrigo Minetto, Daniel Silver, Thiago H Silva  

**Link**: [PDF](https://arxiv.org/pdf/2605.29064)  

**Abstract**: We study how persona prompting shapes language generated by multimodal large language models in an urban perception setting. Using 59,808 annotations from 1,200 persona-conditioned agents and two no-persona settings, we analyze captions, justifications, and perception tags across personas. Results indicate strong convergence in captions for different personas, whereas justifications display systematic variation associated with socioeconomic and political attributes, while perception tags show no statistically significant persona-related differences, though effect trends are observed. Topic analysis further reveals that personas emphasize different evaluative themes when interpreting the same scenes. 

---
# Bosses, Kings, and the Commons: Cooperation Under Power Asymmetry in LLM Societies 

**Authors**: Abhilekh Borah  

**Link**: [PDF](https://arxiv.org/pdf/2605.29062)  

**Abstract**: Communities can sustainably manage shared resources (commons) through self-governance and cooperative norms, a central finding of Ostrom's theory of self-governance. However, real-world commons (e.g., fisheries, forests, and irrigation systems) are often governed under asymmetric power structures, where certain individuals or institutions possess disproportionate control over resource extraction and collective outcomes. As Large Language Models (LLMs) are increasingly explored as agents in synthetic governance simulations, understanding how LLM societies behave under asymmetric power structures is becoming increasingly important, yet existing evaluations largely ignore such asymmetries. We introduce Sovereignty over the Commons Simulation (SovSim), a generative multi-agent simulation framework that incorporates an agent with asymmetric power (boss or king) into a society of symmetric agents (workers or peasants), where all agents extract from a shared resource, collectively determining its sustainability over time. Across eleven state-of-the-art models, we find that introducing asymmetric power leads to severe breakdowns in cooperation and sustainability, with up to an 87.3% degradation in survival rate relative to symmetric settings. 

---
# LLMBridge: An LLM Pipeline for End-to-end Referential Bridging Resolution in English 

**Authors**: Lauren Levine, Amir Zeldes  

**Link**: [PDF](https://arxiv.org/pdf/2605.29048)  

**Abstract**: In this paper, we introduce LLMBridge, a new LLM based system for the task of end-to-end referential bridging resolution in English. Our bridging resolution pipeline combines heuristic pre/post-processing with the natural language inference ability that comes from LLMs. We evaluate our bridging resolution pipeline on 3 datasets which have been used for referential bridging resolution evaluation in English: ISNotes, BASHI, and GUMBridge. Comparison to previous bridging resolution systems shows that the performance of LLMBridge surpasses previous state-of-the-art (SoTA) systems for all 3 datasets in the challenging End-to-end Evaluation Setting, as well as the Basic Bridging Resolution Evaluation Setting (gold bridging anaphor given). We also conduct a thorough error analysis of the LLMBridge performance, examining what varieties of bridging remain difficult for LLM based systems to identify. With this paper, we release the code for the LLMBridge pipeline. 

---
# Error as a Lens: Probing LLM Reasoning through Synthetic Misconception Generation 

**Authors**: Xinming Yang, Jun Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.29007)  

**Abstract**: Personalized tutoring, teacher training, and education research need access to \emph{targeted} synthetic misconceptions, but privacy and IRB constraints make labelled corpora of real student errors scarce. LLMs could in principle generate synthetic errors at scale, but producing an arbitrary wrong answer is easy for a modern LLM while producing one that matches a specified cognitive failure mode is much harder. We present a framework that generates errors targeted to a five-class taxonomy adapted from the revised Bloom's taxonomy, evaluated on questions from the TheoremQA dataset. A Generation Agent (GA) drafts a candidate erroneous solution conditioned on a target class, and an Examination Agent (EA) judges whether the draft is incorrect and class-consistent. The framework yields a reusable recipe for building class-stratified synthetic error datasets where authentic student corpora are unavailable. As a secondary diagnostic, targeted error generation is substantially harder than free-form incorrect-answer generation, and answer-grounding contributes more than expanded examples or external textbook content. 

---
# Text-Preserving Lossy Text Compression: A Study of Strategic Deletion and LLM Reconstruction 

**Authors**: Yuchun Zou, Junhong Tong, Jun Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.29000)  

**Abstract**: Traditional lossless text compression preserves every byte, but its gains on natural language are often modest in realistic operating regimes. We study \emph{lossy semantic text compression}, where the encoder strategically deletes parts of the text and a large language model (LLM) reconstructs the original content from the retained skeleton. We benchmark a progression of deletion strategies, including uniform step deletion, word-length-guided deletion (WordLen), word-frequency-guided deletion (WordFreq), LP-optimized deletion (Opt), entropy-based deletion using GPT-2 surprisal, and hybrid methods that combine frequency and surprisal signals. Evaluation on the BBC News dataset across retention rates $\r_{keep} \in [0.1,0.9]$ shows three main findings. First, WordFreq is a strong low-cost baseline: despite using only a static frequency lookup, it remains competitive with much more expensive semantic methods while being far faster at the encoder. Second, semantic and hybrid methods provide their clearest gains at mild-to-moderate compression, whereas word-frequency deletion is often more robust at the lowest retention rates. Third, QLoRA fine-tuning yields a strong local decoder that is competitive with Gemini 2.0 Flash and is often strongest in decoder-only comparisons. Additional English and Chinese experiments show that the overall framework transfers across domains, while the best deletion rule remains dataset-dependent. 

---
# Beyond Recall: Behavioral Specification as an Interpretive Layer for AI Personalization 

**Authors**: Aarik Gulaya  

**Link**: [PDF](https://arxiv.org/pdf/2605.28969)  

**Abstract**: If an AI agent makes decisions on a person's behalf, those decisions must align with its user. We introduce representational accuracy to measure how faithfully a system captures a person's interpretation. An interpretive layer is operationalized as a Behavioral Specification. Our reference implementation aggressively compresses a person's data into interpretive patterns, served as context to a language model. We evaluate the Specification on a prototype benchmark of held-out behavioral predictions scored by a calibrated 5-judge LLM panel. We test it independently and in composition with a range of context conditions: full raw corpus, full extracted facts, and four commercial memory systems (Mem0, Letta, Supermemory, Zep).
Across 14 public-domain autobiographical corpora, the Specification lifts representational accuracy in aggregate and nearly eliminates model hedging. It recovers most of what the raw corpus delivers, at ~25x less context cost. The Specification lifts subjects toward a common predictive level regardless of pretraining baseline; the lift in absolute points is therefore largest where the baseline is lowest, suggesting the population of relevance is anyone not adequately represented in pretraining. Lift is greatest on interpretation-required questions, where providing an interpretive layer enables model behavior that extracted facts or raw corpus do not. Conversely, on recall-required questions, this layer can interfere rather than help.
We conclude that representational accuracy is distinct from recall and that human-AI alignment is dependent on how accurately the user is represented. Representational accuracy makes that alignment testable. 

---
# The Trust Paradox: How CS Researchers Engage LLM Leaderboards 

**Authors**: Pouya Sadeghi, Anamaria Crisan, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.28966)  

**Abstract**: Large language model (LLM) leaderboards rank AI models using standardized benchmarks and have become highly visible across computer science, despite known limitations in their reliability and robustness. Yet how they shape researchers' actual practice remains empirically uncharted. We address this gap through semi-structured interviews with eight researchers across four computer science subfields, analyzed using reflexive thematic analysis. We find a near-universal paradox of pragmatic skepticism: while participants expressed deep distrust of leaderboard rankings, they continued to use them as rough decision-making aids. Peer networks, not leaderboards, emerged as the primary model selection mechanism, and arena-based (human-voting) leaderboards were consistently preferred over static benchmark leaderboards. Leaderboard influence varied sharply across subfields, revealing that disciplinary culture, not individual attitudes, mediates engagement; for instance, NLP researchers faced state-of-the-art comparison pressure while HCI and Systems/Privacy researchers reported none. Across these differences, however, participants converged on cost transparency as the most demanded missing feature (seven of eight). We translate these findings into concrete design recommendations that align evaluation infrastructure with how researchers actually use it, such as task-specific score breakdowns, cost integration, and voter-demographic disclosure. 

---
# Reasoning that Travels: Dissecting How Chain-of-Thought Transfers Across Models 

**Authors**: Xinyuan Cheng, Beiduo Chen, Philipp Mondorf, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2605.28913)  

**Abstract**: Large reasoning models (LRMs) often generate extensive chain-of-thought (CoT) traces before producing a final answer. As explicit textual artifacts, these traces can be passed to other models to solve the same task, enabling cross-model reasoning transfer. Yet successful transfer alone does not reveal how the provided CoT contributes to another model's answer. We study this question with a controlled provider--receiver framework, where a provider generates a reasoning trace and a receiver solves the same problem from increasingly longer trace prefixes. We compare force-answer, where the receiver answers directly from the prefix, with free-generation, where it may continue reasoning before answering. Across models and benchmarks, full traces often transfer successfully, but prefix trajectories reveal distinct mechanisms. In force-answer mode, AIME transfer is largely driven by explicit answer availability. MMLU-Pro instead reflects a larger role for receiver competence, while ZebraLogic depends on partial structured-answer information rather than complete-answer leakage alone. In free-generation mode, partial CoTs improve performance across benchmarks, indicating that prefixes can guide continued reasoning. Finally, answer agreement among receivers provides a gold-free signal for stopping provider reasoning early. Overall, cross-model CoT transfer is not a single phenomenon: it can reflect answer extraction, reasoning scaffolding, or receiver-dependent competence. 

---
# Hallucination Detection-Guided Preference Optimization for Clinical Summarization 

**Authors**: Shamanth Kuthpadi Seethakantha, Dung Ngoc Thai, Vara Prasad Gudi, Simran Tiwari, Rami Matar, Avijit Mitra, Wenlong Zhao, Wael Salloum, Andrew McCallum  

**Link**: [PDF](https://arxiv.org/pdf/2605.28910)  

**Abstract**: Large language models (LLMs) have shown promise on summarization tasks, but they often produce hallucinations, which are unsupported or incorrect statements that limit their reliability in specialized healthcare applications. We introduce \itermodelfull (\itermodel), an inference-time method that leverages hallucination detectors to guide iterative summary revisions toward factual corrections. Building on this, we propose \itermodel for Preference Learning (\model), which converts detector-guided refinement trajectories into preference pairs for model finetuning. Extensive experiments show that our methods substantially reduce hallucinations for Llama and Gemma models in summarizing real-world clinical notes from \MimicIV. For example, \itermodel reduces 24\% and \model reduces 48\% hallucinations in Llama-3.1-8B-Instruct. Importantly, both methods preserve summary fluency, coherence, and relevance according to human expert and LLM-Jury evaluations. Together, these results demonstrate that detection-informed refinement and preference learning offer an automated solution for improving factual faithfulness in clinical summarization. 

---
# GrowLoop: Self-Evolving Conversation Evaluation Seeded by Human 

**Authors**: Yihang Lin, Yunze Gao, Zeyang Lin, Dongbo Li, Kun Peng, Chenglong Song, Yue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.28882)  

**Abstract**: With the rapid advancement of large language models, evaluating human-likeness in open-ended conversation has become increasingly important. However, human-likeness is a form of tacit knowledge that humans perceive intuitively, yet the underlying criteria resist explicit formulation. Human judgments vary widely, with strong agreement on some cases and legitimate disagreement on others. Meanwhile, the criteria behind human judgments remain implicit, leaving no clear basis for constructing cases. Further, what counts as human-like is not static, but evolving with model capability and human expectations. Despite progress in evaluation methods such as expert-authored benchmarks, Reward Models, and self-evolving benchmarks, none addresses all three challenges simultaneously. Therefore, we propose GrowLoop, a self-evolving conversation evaluation system that continuously adapts as models advance and scenarios shift. With minimal human seed annotations as the first mover, LLM agents iteratively extract and refine evaluation rubrics through Heuristic Learning. Human-AI agreement is required where annotators converge, while only plausibility is expected where they diverge. Moreover, the Rubric-Case co-evolution mechanism enables continuous evolution, expanded through new seeds when the evaluation target moves. Applied to human-likeness evaluation in open-ended conversation, the generated rubrics not only substantially outperform existing methods in alignment with human judgments, but also uncover issues that annotators overlook. The resulting benchmark effectively discriminates models across capability tiers and reveals where they fall short, while generalizing to new scenarios and adapting as models advance. Our work shifts the benchmarking paradigm from manual updates or difficulty scaling to comprehensive, continuous self-evolution. 

---
# From Data to Insights: Exploring Program-of-Thoughts Prompting for Chart Summarization 

**Authors**: Yutong Qu, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28874)  

**Abstract**: Charts play a critical role in conveying numerical data insights through structured visual representations. However, semantic visual understanding and numerical reasoning requirements hinder the accurate description of charts, interpreting a challenging task in chart summarization. Despite recent advancements in visual language models (VLMs), approaches lack robust mechanisms for verifying statistical fact correctness and are computationally heavy. To address this gap, this paper explores a strategy of using zero-shot learning to motivate the lightweight VLMs to perform computational reasoning, via Python programs as intermediaries to derive valid summary statistics for chart understanding. Specifically, we introduce a novel chart-to-dictionary auxiliary task, offering a more flexible representation compared to traditional chart-to-table methods, making it particularly well-suited for integration with the Program-of-Thought (PoT) strategy. Experimental results demonstrate our strategy performs on par with existing chart summarization methods across semantic and factual metrics. Code is available on this https URL. 

---
# Large language models reorganize representational geometry during in-context learning 

**Authors**: Hua-Dong Xiong, Li Ji-An, Robert C. Wilson, Kwonjoon Lee, Xue-Xin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.28854)  

**Abstract**: Large language models (LLMs) exhibit remarkable flexibility: they can adapt to novel tasks from in-context examples without any parameter updates, a capability known as in-context learning (ICL). Prior work on synthetic tasks has shown that ICL can implement specific algorithms, demonstrating architectural competence, and mechanistic analyses have identified key circuits that support this behavior. However, because in-context computation -- regardless of its algorithmic form -- relies on transformations in high-dimensional representation space, it remains unclear how the geometry of that space shapes ICL effectiveness. Motivated by the neuroscience view of classification as the untangling of neural representations, we hypothesize that ICL depends on the successful online untangling of task-relevant representations. To test this idea, we study how LLMs classify in-context examples whose labels are defined by the model's own internal representations with known structure. We show that ICL performance correlates systematically with the representational structure of the underlying classification task and that successful ICL is accompanied by geometric reorganization that increases online separability. We further find that LLM behavior is well described by a prototype-like algorithm that integrates evidence while reshaping representations to support classification. These findings offer a geometric account of ICL in pretrained LLMs, establish representational geometry as a mechanistic constraint on ICL, and quantify the gap between what pretrained representations afford and what in-context learning can exploit. 

---
# GPF-LiveNews: A Streaming Evaluation Protocol for Group-Conditioned Framing in Large Language Models 

**Authors**: Mohd Ariful Haque, Fahad Rahman, Kishor Datta Gupta, Roy George  

**Link**: [PDF](https://arxiv.org/pdf/2605.28848)  

**Abstract**: Deployed language models are evaluated in a non-stationary environment: model versions, retrieval layers, safety systems, and real-world inputs all change over time. Static bias benchmarks remain useful, but they do not show how models frame newly emerging events for different prompted audiences. We introduce GPF-LIVENEWS, a streaming evaluation protocol and benchmark snapshot for auditing group-conditioned framing in open-ended LLM outputs. The protocol expands fresh BBC/Reuters news anchors across 42 identity labels and seven prompt families, then evaluates response bundles using semantic-sensitivity and sentiment-disparity signals. In a pilot over 12 monitoring runs and 23 hosted models, Policy/Action prompts produce the strongest semantic movement, while sentiment variation is flatter across dimensions and prompt families. The released artifact includes article metadata, prompt templates, instantiated prompts, model-output metadata, score tables, documentation, and reproduction scripts. We interpret all scores as observed-window audit signals for human review, not as permanent fairness rankings or direct proof of harmful bias. 

---
# Thoughts-as-Planning: Latent World Models for Chain-of-Thoughts Optimization via Reinforcement Planning 

**Authors**: Dong Liu, Yanxuan Yu, Ying Nian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.28842)  

**Abstract**: The success of large language models (LLMs) across diverse NLP tasks has elevated the importance of reasoning chain optimization as a critical step in aligning model behavior with task objectives. Existing reasoning chain tuning methods often rely on black-box heuristics or gradient-free search, which lack interpretability, generalization, and sample efficiency. In this work, we introduce \textbf{Thoughts-as-Planning}, a novel framework that formalizes reasoning chain optimization as a sequential decision-making process over a latent semantic space. We model the LLM as a partially observable environment and learn a latent world model that simulates the effect of reasoning chain edits on downstream outputs. A proximity-preserving embedding space is constructed to encode reasoning chain-response dynamics, enabling planning via gradient descent or reinforcement learning. Our method supports multi-scale abstraction, allowing reasoning chain edits at token, segment, and instruction levels to be integrated into a unified planner. Through extensive experiments on language understanding and generation tasks, we demonstrate that Thoughts-as-Planning outperforms state-of-the-art reasoning chain tuning baselines in efficiency, robustness, and generalization, while offering interpretability through its structured planning trajectory. Our code is available at this https URL. 

---
# How Consistent Are LLM Agents? Measuring Behavioral Reproducibility in Multi-Step Tool-Calling Pipelines 

**Authors**: Abel Yagubyan  

**Link**: [PDF](https://arxiv.org/pdf/2605.28840)  

**Abstract**: Large language model (LLM) agents with tool-calling capabilities are increasingly deployed in production systems, yet a fundamental reliability question remains under-explored: does the same agent behave the same way twice? We present a systematic empirical study of behavioral consistency in multi-step tool-calling agents, measuring whether agents select the same tools, in the same order, with the same arguments, across repeated identical invocations. Unlike prior work on consistency in ReAct-style agents(search-only, free-text actions), we study the richer setting of structured tool-calling interfaces with typed parameters and consequential side effects. 

---
# Specialty-Specific Medical Language Model for Immune-Mediated Diseases 

**Authors**: Veysel Kocaman, Gursev Pirge, Yigit Gul, Ace Vo, Zhenya Nargizyan, David Talby  

**Link**: [PDF](https://arxiv.org/pdf/2605.28838)  

**Abstract**: Extracting detailed clinical information from free-text medical narratives remains a practical challenge for researchers and healthcare systems. Terminology for immune-mediated and infectious diseases is especially inconsistent across sources, which often limits the ability of general-purpose Natural Language Processing (NLP) systems to capture the relevant biomedical concepts with sufficient granularity. We developed a domain-specific Named Entity Recognition (NER) model tailored to identify disease-related entities occurring in immunology and infectious disease contexts. We assembled and manually annotated a dataset of 371 case reports in collaboration with two clinical specialists, defining twelve entity classes covering immune-mediated and infectious conditions as well as related symptoms and clinical descriptors. We evaluated several modeling strategies, including the MedicalNER architecture with multiple healthcare-specific embeddings, a BERT-based token classification model, and zero-shot NER systems. The strongest performance was obtained with a transformer-based model trained on clinical-domain embeddings, which reached an F1 score of 0.89, consistently outperforming baseline and zero-shot approaches. The combination of specialized embeddings and expert annotation proved particularly valuable for capturing nuanced disease terminology and improving generalization across heterogeneous biomedical text. The prompted LLM baseline achieved substantially lower performance under the same evaluation protocol, reflecting difficulties in producing span-consistent outputs for fine-grained entity boundaries despite detailed prompting. The resulting model provides a structured way to analyze case reports and can support downstream tasks such as cohort identification, disease monitoring, and clinical decision support. 

---
# SERC: LDPC-Inspired Semantic Error Correction for Retrieval-Augmented Generation 

**Authors**: Gyumin Kim, Juhwan Park, Jaeha Kim, Seunggyun Han, Kyungrak Son, Ikbeom Jang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28837)  

**Abstract**: While Large Language Models (LLMs) have demonstrated remarkable capabilities, their reliability is significantly compromised by hallucinations. Existing intrinsic self-correction methods attempt to address this, but often fail due to self-bias, where models struggle to identify errors in their own outputs without external verification. To overcome these limitations, we propose the LDPC-inspired semantic error correction for retrieval-augmented generation (SERC), providing a theoretical framework to interpret and mitigate LLM hallucinations. We reformulate the text generation process as a semantic noisy channel, treating generated responses as noise-corrupted codewords. Inspired by low-density parity-check (LDPC) codes, SERC employs a sparse verification strategy: instead of exhaustively checking all facts, it generates low-density verification queries and validates them against external evidence to efficiently detect and correct errors. We evaluate SERC on LongForm Bio and TruthfulQA benchmarks using Llama-3-8B and Qwen2.5-14B. Experimental results demonstrate that SERC outperforms both intrinsic self-correction methods and strong retrieval-augmented baselines, demonstrating significant gains especially in factual precision (FactScore). Notably, SERC enables small language models (SLMs) to surpass the performance of larger baselines in hallucination reduction and information preservation. Our findings demonstrate that SERC provides a training-free, model-agnostic solution that significantly reduces verification overhead compared to dense methods, achieving an optimal trade-off between cost and fidelity in resource-constrained environments. 

---
# No Reader Left Behind: Multi-Agent Summaries Everyone Can Understand 

**Authors**: Jimin Jung, MyoungJin Kim, Jaehyung Seo, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2605.28836)  

**Abstract**: The Plain Writing Act in the United States requires government documents to be accessible in clear and simple language that the general public can easily understand, yet existing summarization systems struggle to address diverse linguistic and cognitive barriers among general readers. We present NRLB (No Reader Left Behind), a multi-agent framework for plain language summarization that simulates three representative reader groups: elementary school student readers, non-native readers, and readers with attention deficits. NRLB combines template-based planning with iterative, reader-oriented refinement, enabling systematic detection and resolution of difficult terms, missing contexts, and confusing sentences. Evaluations across multiple datasets demonstrate consistent improvements in readability while preserving factual accuracy. Human evaluation further validates NRLB's impact, with annotator preference rates ranging from 55% to 76%, highlighting NRLB's potential to produce plain language summaries that are both faithful to the source and broadly accessible to the general public. 

---
# GenesisFunc: Multi-Agent Data Generation for Accurate and Generalizable Function-Calling 

**Authors**: Hao-Xiang Xu, Chong Deng, Jiaqing Liu, Wen Wang, Qian Chen, Lujia Bao, Xiangang Li, Zhen-Hua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2605.28835)  

**Abstract**: Large Language Models (LLMs) extend their capabilities through function-calling (FC), which relies on training data with high quality, diversity, and broad coverage of scenario. However, obtaining and annotating real function-calling data is challenging, while synthetic data from existing pipelines often suffers from unreliable APIs, limited tool scalability, insufficient diversity, and weak quality control. To address these, we present GenesisFunc, an automated pipeline for generating FC training data. Starting from reliable tools in widely used public benchmarks, our GenesisFunc employs a multi-agent framework to support a dialogue generation system that produces conversations spanning diverse scenarios, while maintaining both diversity and quality throughout the process. The accuracy of the data is further reinforced through a multi-stage evaluation system. We fine-tune an 8B LLM on the synthetic dataset and show through extensive experiments that it outperforms similarly sized open-source models in in-domain FC performance and out-of-domain generalization, while reaching FC capabilities comparable to some of the latest API-based models. In addition, our method demonstrates strong potential to scale effectively across downstream tools, underscoring its real-world applicability. 

---
# Assessing Dutch Syllabification Algorithms and Improving Accuracy by Combining Phonetic and Orthographic Information through Deep Learning 

**Authors**: Gus Lathouwers, Wieke Harmsen, Catia Cucchiarini, Helmer Strik  

**Link**: [PDF](https://arxiv.org/pdf/2605.28834)  

**Abstract**: Syllabification describes the task of dividing words into syllables. Due to many rules and exceptions, training an algorithm to perform syllabification with high accuracy remains a challenge. Throughout the last decades, different algorithms have been put forth for Dutch syllabification, yet a comprehensive comparative assessment has not been done. Additionally, deep learning has gained significant popularity within NLP in recent years, yet no modern deep-learning based framework has been developed for Dutch orthographic syllabification. Finally, phonetic and orthographic syllabification algorithms have been examined separately, but not in combination. The aim of the current research was twofold: (a) to examine the performance of existing Dutch syllabification algorithms, and (b) to investigate whether combining phonetic and orthographic information into a single model can increase syllabification performance. To compare the performance of algorithms, four algorithms (Brandt Corstius, Liang, Trogkanis-Elkan (CRF), and a newly conceived deep-learning model) were applied to three different datasets (dictionary words, loanwords, pseudowords). The algorithms show varying performance across datasets, with the data-driven algorithms outperforming a knowledge-based algorithm in all but one condition. The new deep-learning methods developed led to increased performance compared to the best found in the literature (99.65% word accuracy, a 0.14% improvement). An analysis of the words for which adding phonetic information improved syllabification performance indicates that these were words in which the orthographic ambiguity could be resolved by information on pronunciation. Future research could examine other areas where phonetic information can benefit orthographic processing. In addition, the newly developed deep learning frameworks can be applied to other languages than Dutch. 

---
# Transcribing Children's Speech: ASR Performance and Obtaining Reliable Orthographic Transcriptions 

**Authors**: Gus Lathouwers, Lingyun Gao, Catia Cucchiarini, Helmer Strik  

**Link**: [PDF](https://arxiv.org/pdf/2605.28833)  

**Abstract**: Automatic speech recognition (ASR) has the potential to substantially reduce manual annotation effort in child speech research by generating automatic transcriptions. However, obtaining reliably high-quality ASR transcriptions for child speech remains challenging in low-resource languages due to limited child-specific pre-trained models and highly diverse noise conditions. This study investigates the effectiveness of state-of-the-art ASR models on child speech through two research questions, by evaluating nine ASR models from three model families (Whisper, Parakeet, and Wav2Vec2) on two Dutch child speech datasets, JASMIN and DART. Research question 1 examines the performance of ASR-models applied to child speech. The fine-tuned Whisper-medium model achieves the best overall performance, with a WER of 5.54% on JASMIN and 70.37% on DART, showing that the noisy DART data are clearly more challenging. Research question 2 examines to what extent it is possible to select a subset for which reliable orthographic transcriptions can be obtained automatically, without the need for manual verification. We use an utterance-level selection method that compares ASR output with the original read prompt to identify correctly pronounced recordings. Using the proposed selection method, 42.0% [for JASMIN] and 18.1% [for DART] of the utterances can be automatically identified as correctly pronounced with high confidence, resulting in very low error rates on an utterance level (precisions of 98.3% and higher) and reducing the need for manual verification. 

---
# A comparative study of transformer-based embeddings for topic coherence 

**Authors**: Alex Ding, Tarun Rapaka, Willy Rodriguez, Jason Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28832)  

**Abstract**: Topic modeling is a branch of Natural Language Processing (NLP) that aims to organize large collections of texts into coherent groups according to word co-occurrence patterns, with Latent Dirichlet Allocation (LDA) remaining one of the most widely used and interpretable probabilistic approaches. Recent advances in NLP, particularly transformer-based language models, offer improved document representations. It is also known that the size of the model (in terms of number of parameters) has a significant impact in the performance of the language models on different pre-defined tasks. In this study, we systematically examine the effect of model size on topic quality by analyzing the performances of seven transformer-based language models (from small models such as MiniLM to large ones such as LLaMA-2) in a BERTopic pipeline on a variety of corpora. Topic quality is evaluated using coherence and divergence metrics following R{ö}der et al. (2015). Our results indicate that model size, ranging from 22 million to 13 billion parameters, has a negligible impact on the quality of the topic, suggesting that smaller models can achieve comparable performance to larger models. 

---
# S3Mem: Structured Spatiotemporal Scene-Event Memory for Long-Horizon Interactive Question Answering 

**Authors**: Encheng Su, Jinouwen Zhang, Jianyu Wu, Qiucheng Yu, Chen Tang, Pengze Li, Lintao Wang, Yizhou Wang, Xinzhu Ma, Shixiang Tang, Aoran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28831)  

**Abstract**: Long-horizon interactive agents often accumulate large trajectory histories yet still fail to answer questions about earlier events reliably. We argue that the main bottleneck is not context length alone, but the trajectory-to-answer interface of long-term memory. When histories are stored as plain-text chunks and queried with standard retrieval-augmented generation (RAG), systems often retrieve locally relevant but chain-incomplete evidence, especially for spatial, temporal, repeated-event, and multi-hop state questions. We propose S3MEM, a structured scene-event episodic memory framework for long-horizon interactive question answering (QA). S3MEM writes trajectories into structured memory units, retrieves evidence through anchor-sensitive retrieval, and exposes a compact token-budget-aware evidence interface for answer-time inference. In this sense, S3MEM is a structured evidence harness that converts agent trajectories into query-aligned support. We evaluate S3MEM on two internal headline environments (Crafter, Jericho) and two out-of-family environments (SciWorld, ALFWorld). Under a shared frozen answer-time protocol, S3MEM consistently outperforms Vanilla RAG across all four environments, surpasses Graph-NoReader on Crafter, Jericho, and ALFWorld, and matches it on SciWorld while using dramatically fewer evidence tokens. Three adapted recent baselines -- A-MEM-inspired, MemoryOS-adapted, and LightMem-adapted -- improve over Vanilla RAG in several settings, but none matches S3MEM's overall accuracy-efficiency frontier. Overall, the evidence supports a bounded conclusion: under the current frozen answer-time protocol, structured writing and anchor-sensitive evidence routing provide a stronger accuracy-efficiency frontier for long-horizon interactive QA than more generic memory interfaces. 

---
# Benchmarking Open-Source Safety Guard Models: A Comprehensive Evaluation 

**Authors**: Reetu Raj Harsh, Bhaskarjit Sarmah, Stefano Pasquali  

**Link**: [PDF](https://arxiv.org/pdf/2605.28830)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in safety-critical applications, robust content moderation becomes essential. We present a comprehensive evaluation of 14 open-source safety guard models on a curated benchmark of 79,331 samples spanning 8 NIST AI Risk Framework safety categories. Our benchmark aggregates four diverse datasets (HarmBench, StrongREJECT, RealToxicityPrompts, and BeaverTails), filtered to focus exclusively on safety-relevant content (violence, hate speech, harassment, sexual content, suicide/self-harm, profanity, threats, and health misinformation). We find that recall is the critical metric for safety applications, as missing harmful content poses greater risk than false positives. Our evaluation reveals surprising results: Qwen Guard (4B parameters) achieves the highest recall (83.97%) while larger models like Llama Guard (12B) and GPT-OSS Safeguard (20B) exhibit conservative behavior, missing up to 75% of unsafe content. We demonstrate that model size does not correlate with safety detection performance and that general-purpose guard models outperform specialized ones. These findings provide practical guidance for selecting safety guard models in production deployments. 

---
# Aryabhata 2: Scaling Reinforcement Learning for Advanced STEM Reasoning 

**Authors**: Ritvik Rastogi, Vishal Singh, Tejas Chaudhari, Sandeep Varma  

**Link**: [PDF](https://arxiv.org/pdf/2605.28829)  

**Abstract**: Competitive STEM examinations such as JEE and NEET require multi-step symbolic reasoning, precise numerical computation, and deep conceptual understanding across physics, chemistry, and mathematics. Recent large language models perform strongly on common reasoning benchmarks, yet they remain difficult to deploy at scale, where millions of student doubts demand domain-specific, consistently structured problem solving.
We introduce Aryabhata 2, a reasoning-focused language model for competitive STEM examinations, trained via reinforcement-learning post-training. Using PhysicsWallah's internal question banks, we construct a high-quality training curriculum and post-train GPT-OSS-20B through reinforcement learning with verifiable rewards. Training combines prolonged reinforcement learning with broadened exploration via progressively larger rollout group sizes.
We evaluate Aryabhata 2 on competitive examination benchmarks, including JEE Main, JEE Advanced, and NEET, as well as out-of-distribution reasoning datasets such as AIME, HMMT, MMLU-Pro, MMLU-Redux 2.0, and GPQA. Results show that Aryabhata 2 outperforms its base model GPT-OSS-20B on competitive STEM reasoning while requiring substantially fewer output tokens (up to 64\% fewer). 

---
# Micro-Macro Retrieval: Reducing Long-Form Hallucination in Large Language Models 

**Authors**: Yujie Feng, Jian Li, Zhihan Zhou, Pengfei Xu, Yujia Zhang, Xiaoyu Li, Xiaohui Zhou, Alan Zhao, Xi Chen, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.28828)  

**Abstract**: Large Language Models (LLMs) achieve impressive performance across many tasks but remain prone to hallucination, especially in long-form generation where redundant retrieved contexts and lengthy reasoning chains amplify factual errors. Recent studies highlight a critical phenomenon: the closer key information appears to the model outputs, the higher the factual accuracy. However, existing retrieval-augmented language models (RALMs) lack effective mechanisms to ensure this proximity - external evidence is injected into reasoning via multi-turn retrieval, but this cannot ensure key information stays close to the outputs. We propose Micro-Macro Retrieval (M2R), a novel retrieve-while-generate framework to fill this gap. At the macro level, M2R retrieves coarse-grained evidence from external sources; at the micro level, it extracts essential results from a key information repository built during reasoning and reuses them while generating answers. This design directly addresses the key-information-to-output proximity bottleneck, effectively reducing hallucination in long-form tasks. M2R is trained with a curriculum learning-based reinforcement learning strategy using customized rule-based rewards, enabling stable acquisition of retrieval and grounding skills. Extensive experiments across different benchmarks demonstrate the effectiveness of M2R, especially in lengthy-context settings. 

---
# RightNow-Arabic-0.5B-Turbo: An Open Sub-1B Arabic Language Model via Vocabulary Injection and Edge-First Deployment 

**Authors**: Jaber Jaber, Osama Jaber  

**Link**: [PDF](https://arxiv.org/pdf/2605.28827)  

**Abstract**: Open Arabic large language models split into two classes: sub-1B multilingual models that treat Arabic as an afterthought (Qwen2.5-0.5B, Falcon-H1-0.5B), and 7B-70B Arabic-specialized models that require a server to run (Jais, AceGPT, ALLaM, SILMA). The one published attempt at a sub-2B Arabic-specialized model, Kuwain-1.5B, never released its weights. We present RightNow-Arabic-0.5B-Turbo, a 518M-parameter Arabic-specialized decoder LLM built on Qwen2.5-0.5B. The pipeline adds 27,032 Arabic tokens via mean-subtoken initialization, continues pretraining on 504M Arabic tokens on 8xH100 with FSDP, FlashAttention varlen packing, and Liger fused kernels, then applies supervised fine-tuning on 129,116 Arabic instruction pairs with response-only loss masking, direct preference optimization on 6,750 Arabic preference pairs, and weight soup merging across three checkpoints. On three lm-evaluation-harness Arabic benchmarks (COPA-ar, Arabic HellaSwag, ArabicMMLU) the merged model reaches 35.9% mean accuracy, beats every same-class open model, ties Falcon-H1-1.5B on COPA-ar (58.4%) at one-third the size, and recovers 67% of SILMA-9B's mean at 1/18 the parameters. The edge build quantizes to 398 MB (q4_k_m) and delivers 635 tokens/s at batch size 1 on a single H100 via this http URL. All code (5,555 lines across 25 scripts), weights (bf16, int8, and four GGUF quantizations), and benchmark scripts are released at this https URL. 

---
# From Context Shift to Stylistic Collapse: Why Training Objectives Matter More Than Scale 

**Authors**: Rohan Mahapatra  

**Link**: [PDF](https://arxiv.org/pdf/2605.28826)  

**Abstract**: In modern LLMs, linguistic features function not as stylistic artifacts but as probes of probability mass, allocated under training alignment objectives. Language models trained with contemporary pipelines exhibit severe reshaping of linguistic features, leading to extreme language re-distribution. While previous stylometric analyses explored linguistic differences between AI-generated and human texts, we focus on the reshaping plaguing the LLM training pipeline itself. We analyze 17 models (410M-100B+ parameters) across 24 linguistically-motivated probes, documenting that instruction-tuned systems systematically collapse language entropy along discourse and structural dimensions (mean amplification: 1,949-16,853%, peaks: 5,181-209,675%), while selectively suppressing complex punctuation to 3.2-23.2% of baseline frequencies. These effects do not worsen under RLHF, as divergence patterns are statistically indistinguishable (p > 0.25) across matched base and instruction-tuned model pairs. Weak intervention (lambda=1.0) exacerbates collapse by 240%, while strong control (lambda=5.0) achieves 40.5% improvement and outperforms frontier models by 96.7-98.2% despite 200-1000x scale disadvantage. Additionally, lambda=5.0 delivers 15% higher distinct-4, 27% higher vocabulary diversity, and 78% lower repetition than moderate regularization, establishing that alignment requires sufficient control strength, not merely distributional smoothing. Our findings underscore how modern LLMs reallocate stylistic probability mass, despite RLHF and scale. More broadly, our work reveals a structural limitation of current alignment pipelines: preference optimization reshapes language distributions invisible to standard quality metrics yet detectable through distributional probes, with implications for AI detection, training data contamination, and long-term linguistic evolution. 

---
# MechELK: A Mechanistic Interpretability Framework for Eliciting Latent Knowledge in Large Language Models 

**Authors**: Ji-jun Park, Soo-joon Choi, Jiwon Jeong, Taeyang Yoon, Ju-Wan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.28825)  

**Abstract**: Large language models (LLMs) frequently encode factual and reasoning knowledge in their internal representations that is not faithfully reflected in their surface-level outputs -- a phenomenon known as \emph{latent knowledge}. Existing approaches to eliciting latent knowledge, such as Contrastive Consistency Search (CCS), rely on contrastive activation patterns and struggle with complex multi-step reasoning tasks, while mechanistic interpretability tools have primarily been used to \emph{understand} model behavior rather than to \emph{extract} hidden knowledge. We present \textbf{MechELK}, a unified three-stage framework that bridges mechanistic interpretability and latent knowledge elicitation. MechELK operates through: (1) \textbf{Locate} -- using Sparse Autoencoder (SAE) feature analysis and activation patching to identify knowledge-bearing representations; (2) \textbf{Verify} -- employing causal probing to distinguish genuine latent knowledge from spurious correlations; and (3) \textbf{Elicit} -- applying representation engineering to surface hidden knowledge without modifying model weights. Evaluated on TruthfulQA, a curated Deceptive Alignment benchmark, and the Quirky LM dataset, MechELK achieves an average elicitation accuracy of 84.7\%, outperforming CCS by 6.2\% and direct linear probing by 9.1\%. Crucially, MechELK successfully identifies latent knowledge in 78.3\% of cases where the model's surface output is incorrect or evasive, demonstrating its utility for AI safety applications including deceptive alignment detection. 

---
# A Modular Architecture for Typologically Controlled Lexicon Generation 

**Authors**: Sankalp Tattwadarshi Swain, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.28824)  

**Abstract**: Constructing artificial lexicons that are pronounceable, typologically plausible, and semantically structured remains an open challenge in computational linguistics. Existing conlang generators either lack formal phonotactic guarantees or delegate generation to opaque, non-reproducible LLM-based pipelines. We propose a modular framework that samples phoneme inventories from PHOIBLE, generates word forms under interchangeable phonological grammars (deterministic, OT, and MaxEnt), and assigns meanings via a Swadesh--Leipzig--Jakarta ontology with explicit form--meaning alignment. Evaluation on character $n$-gram perplexity, log-likelihood, and KL divergence against PHOIBLE across lexicon sizes of 100-5,000 forms shows that probabilistic grammars consistently outperform deterministic and random baselines on both phonotactic coherence and typological realism. 

---
# What are They Thinking? Delineation, Probing and Tracking of Concepts in LLMs 

**Authors**: Mohamed Abdelwahab, Michelle Yu Collins, Sihan Chen, Yi Cheng Zhao, Zafarullah Mahmood, Jiading Zhu, Soliman Ali, Jonathan Rose  

**Link**: [PDF](https://arxiv.org/pdf/2605.28823)  

**Abstract**: As the influence of LLMs expands, it is imperative to gain insight into their decisions. One way to do that is to develop probes that detect the presence or absence of a broad set of concepts within the embeddings computed in an LLM - which is what we might say a model is "thinking" about. Such probes should be low-cost and easily applicable to any LLM, so that monitoring for many concepts is possible during normal operation.
In this paper, we take the first steps towards developing the capability of creating many such probes by defining and executing examples of the key tasks needed: first, the careful delineation of a concept through the creation of a dataset with the concept both present and then absent. Then, the training and testing of a set of linear probes to detect the concept on any layer of an LLM, including an exploration of the complexity of the probe needed. Finally, we show that such probes can track concepts across larger contexts. This is done with four separate concepts and three different LLMs. When this process is scaled to many more concepts, it will create the ability to easily monitor new models. 

---
# Lightweight Multimodal LLM-Enabled Cost-Effective Defect Grading of Power Transmission Equipment 

**Authors**: Tao Wang, Lipeng Zhu, Jiayong Li, Feng Gao, Siwen Liang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28822)  

**Abstract**: Defect grading of power transmission equipment (DGPTE) is crucial to the stability of electric energy transmission. Although existing machine learning methods exhibit strong capabilities in defect detection, they are plagued by difficulties in integrating expert experience and facing class imbalance in more refined defect grading field. To address this issue, this paper introduces a novel defect grading framework based on multimodal large language model (MLLM). Specifically, this approach maximizes the commercial MLLMs' potential of DGPTE through in-context learning and obtains the state-of-te-art (SOTA) model. By sending a secondary request to this model, a small number of chain of thought-based question-answer pairs (Q\&As) are generated, which effectively reduces the cost of manual annotation. In this way, these high-quality interpretable Q\&As are used to train Qwen3-VL-8B via Low-Rank Adaption-based supervised fine-tuning (SFT). Experimental results on three DGPTE tasks demonstrate that fine-tuning only the language model layer yields the SOTA performance. Furthermore, multi-task joint fine-tuning verifies the feasibility of handling multiple grading tasks within only a single lightweight MLLM. 

---
# SchGen: PCB Schematic Generation with Semantic-Grounded Code Representations 

**Authors**: Qinpei Luo, Ruichun Ma, Xinyu Zhang, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30345)  

**Abstract**: Printed circuit board (PCB) schematic design defines nearly all electronic hardware, but it remains manual and expertise-intensive. While generative AI has advanced digital and analog IC design, PCB schematic generation from natural-language intent is largely unexplored. This paper presents SchGen, the first large language model that generates editable PCB schematics from natural-language requests. The key challenge lies in the lack of an LLM-suited representation and a large-scale dataset. Current schematic formats are dominated by verbose, tool-specific syntax and geometry-heavy descriptions, making them difficult to generate reliably. We introduce a semantically grounded code representation that encodes schematic editing primitives with relative placement and pin-name-based wiring, transforming a geometry-driven generation problem into a semantics-driven matching task amenable to LLMs. We further construct a large-scale dataset of PCB schematics paired with user prompts via a human-agent collaborative pipeline that converts open-source hardware designs into our representation. Experiments show that SchGen significantly outperforms alternative representations and even larger general-purpose LLMs on wire connectivity accuracy and functional correctness. Our results highlight the critical role of representation design in enabling generative models for complex hardware design tasks. 

---
# Locally Coherent, Globally Incoherent: Bounding Compositional Incoherence in Multi-Component LLM Agents 

**Authors**: Anany Kotawala  

**Link**: [PDF](https://arxiv.org/pdf/2605.30335)  

**Abstract**: Multi-component LLM agents assemble probabilistic claims from components that each see only part of a joint problem; the composition can violate basic probability axioms even when every component is locally coherent. We formalise this locally coherent, globally incoherent failure via the compositional residual eps*, the L2 distance from the composed quote to the joint coherent polytope, computable at runtime from system output and the declared cross-component coupling constraints. A product-structure dichotomy characterises when local coherence suffices, and a Rayleigh-quotient prediction matches the observed residual within 7% on three of four relation classes. A hierarchical Boyle-Dykstra projection repairs the composition deterministically; an anytime-valid e-process gives sequential coherence monitoring. Across 1,876 ensemble cliques on a four-LLM mid-tier panel (frontier-panel rerun in Section 5.5), eps* > 0 on 33-94% of cliques, translating to +0.115 nats per bet of regret on 1,770 resolved bets under the proportional allocation rule (the gain collapses to +0.006 under bettors that themselves coherentise). Three intuitive LLM-side mitigations(retrieval, partition-aware prompting, aggregator-LLM) each fail or regress. 

---
# Demystifying Data Organization for Enhanced LLM Training 

**Authors**: Yalun Dai, Yangyu Huang, Tongshen Yang, Yonghan Wang, Xin Zhang, Wenshan Wu, Qihao Zhao, Hao Li, Yuanyuan Gao, Kim-Hui Yap, Scarlett Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.30334)  

**Abstract**: Large Language Models (LLMs) have revolutionized various fields, yet their training efficiency is heavily reliant on effective data curation. While data selection has been widely studied, the strategic data organization for enhanced training remains an underexplored area, particularly since current LLMs are often trained for only one or a few epochs. This paper systematically explores the influence of data organization on LLM training by reusing pre-computed sample-level scores originally generated for data efficiency, thereby incurring minimal additional computational overhead. We identify and formalize four key guidelines for optimizing data organization: Boundary Sharpening, Cyclic Scheduling, Curriculum Continuity, and Local Diversity. Guided by them, we introduce two novel data ordering methods termed STR and SAW. Extensive experiments across different model scales and data sizes, encompassing both pre-training and SFT stages, validate the effectiveness of our summarized guidelines. They also demonstrate the robustness of our proposed data ordering methods in enhancing the stability and performance of LLM training. Github Link: this https URL 

---
# Reasoning with Sampling: Cutting at Decision Points 

**Authors**: Felix Zhou, Anay Mehrotra, Quanquan C. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30327)  

**Abstract**: Frontier reasoning models are produced by posttraining base language models with reinforcement learning. Recent work has challenged this by showing that sampling from a sharpened version of the base model's distribution, a so-called power distribution, elicits comparable reasoning without additional training, curated datasets, or verifiers. However, making this method practical requires efficiently sampling from the power distribution. A sampler needs to "mix" to the power distribution, which necessitates moving between modes of the target distribution; intuitively, e.g., trying different reasoning strategies. The samplers proposed in prior works repeatedly select a "cut" position in the current reasoning trace uniformly at random and resample the suffix from that position onward. However, reasoning traces typically contain a few consequential decisions (e.g., the choice of proof strategy or algorithm), and we observe that a uniformly chosen cut tends to rewrite local details rather than revisit decision points. We introduce an algorithm (Entropy-Cut Metropolis-Hastings) that uses the base model's next-token entropy as a proxy to identify key decision points and resample from those positions. We empirically verify that entropy jumps are a useful proxy for decision points and, in a stylized model of reasoning, prove that our method's mixing time scales with the number of decisions in a trace rather than with the number of tokens, which can be much larger. Across MATH500, HumanEval, GPQA Diamond, and AIME26, our method consistently improves over baselines and RL-trained models. 

---
# On Language Generation in the Limit with Bounded Memory 

**Authors**: Jon Kleinberg, Anay Mehrotra, Amin Saberi, Grigoris Velegkas  

**Link**: [PDF](https://arxiv.org/pdf/2605.30324)  

**Abstract**: We study language generation in the limit under bounded memory. In this task, a learner observes examples from an unknown target language one at a time and must eventually output only new valid examples. Prior work assumes access to the entire history, a strong assumption since realistic algorithms retain limited past information. Classical work in learning theory shows memory constraints dramatically alter learnability; we extend this to language generation.
First, we study memoryless generators. Under a mild enumeration restriction, every countable collection of infinite languages remains generable without memory. Without this restriction, we exactly characterize when memoryless generation is possible. For finite collections, we characterize the optimal minimax density achievable by memoryless generators -- the best density guaranteed against any collection of a given size. This combinatorial bound relies on Sperner's theorem and symmetric chain decompositions.
We further show that a sliding window of the last $W$ examples does not improve this worst-case density, whereas allowing it to store $b$ adaptively chosen past examples improves the achievable density for every $b \geq 1$.
Finally, we revisit identification in the limit, where the learner must converge to a single correct hypothesis for the target language. We focus on its incremental variant, where the learner remembers only its previous guess. Here, although exact identification fails on a collection of just three languages, a mild relaxation requiring convergence to an ``approximate'' version of the target is achievable for every finite collection.
These results show bounded memory affects these tasks differently: generation remains achievable for every countable collection, while density and identification are confined to finite collections, with guarantees weakening as the collection grows. 

---
# Self-Trained Verification for Training- and Test-Time Self-Improvement 

**Authors**: Chen Henry Wu, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2605.30290)  

**Abstract**: Self-improvement at scale has been a longstanding goal for reasoning models, and there are two natural places to do it: at test time, through verification-refinement (V-R) loops; and at training time, through self-training methods. Both are gated by the same bottleneck: the verifier. V-R loops stall when verifier scores inflate while accuracy stagnates, and when feedback is too generic to act on; self-training fails similarly when bad self-generated data are added to training. Better verification would unlock both, but the capability we want to train, i.e., catching self-generated errors, lacks training signal. To address this challenge, we propose self-trained verification (STV). Our key observation is that, while a model cannot catch these errors alone, it can when shown the reference solution. We turn this asymmetry into a supervision target and train the verifier to imitate a more informed version of itself. At test time, STV substantially improves V-R loops on hard problems, while alternatives (e.g., SFT, RL on verifier scores, and even meta-verifiers) do not. STV roughly doubles accuracy on hard math and lifts it 14x on scientific reasoning tasks (1.5% to 21%). At training time, we additionally train the generator using RL with STV verifier's feedback inside the V-R loop - a procedure we call verifier-in-the-loop training (ViL). Starting from an RL-converged generator, ViL yields a further 33% gain in pass@1. More notably, the generator's standalone pass@1, with no verifier at test time, climbs 30% relative past where standard RL had converged. Hence, the next frontier in reasoning on hard problems may lie in how we train for and with verification. 

---
# Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments 

**Authors**: Qiuyue Wang, Mingsheng Li, Jian Guan, Jinhui Ye, Sicheng Xie, Yitao Liu, Junhao Chen, Zhixuan Liang, Jie Zhang, Xintong Hu, Xuhong Huang, Pei Lin, Junyang Lin, Dayiheng Liu, Shuai Bai, Jingren Zhou, Jiazhao Zhang, Haoqi Yuan, Gengze Zhou, Hang Yin, Ye Wang, Yiyang Huang, Zixing Lei, Wujian Peng, Delin Chen, Yingming Zheng, Jingyang Fan, Xianwei Zhuang, Xin Zhou, Haoyang Li, Anzhe Chen, Tong Zhang, Xuejing Liu, Yuchong Sun, Ruizhe Chen, Zhaohai Li, Chenxu Lü, Zhibo Yang, Tao Yu, Xionghui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.30280)  

**Abstract**: Embodied intelligence is often studied through specialized models for individual tasks such as manipulation or navigation, resulting in fragmented capabilities and limited generalization across tasks, environments, and robot embodiments. In this work, we study whether heterogeneous embodied decision-making problems can be unified within a single vision-language-action model. We present Qwen-VLA, a unified embodied foundation model that extends Qwen's vision-language modeling stack from perception, understanding, and reasoning to continuous action and trajectory generation through a DiT-based action decoder. Qwen-VLA is trained with a large-scale joint pretraining recipe over diverse data sources, including robotics manipulation trajectories, human egocentric demonstrations, synthetic simulation data, vision-and-language navigation data, trajectory-centric supervision, and auxiliary vision-language data. To support multiple robot platforms, we introduce embodiment-aware prompt conditioning, where robot-specific textual descriptions specify the current embodiment and control convention. We further cast manipulation, navigation, and trajectory prediction into a unified action-and-trajectory prediction framework, enabling transferable visual grounding, spatial reasoning, and continuous action generation across robot morphologies, task families, and environments. Experiments on manipulation, navigation, and trajectory-centric benchmarks show consistent multi-task performance and out-of-distribution generalization under variations in scene layout, background, lighting, object configuration, and robot embodiment. Qwen-VLA-Instruct achieves 97.9% on LIBERO, 73.7% on Simpler-WidowX, 86.1%/87.2% on RoboTwin-Easy/Hard, 69.0% OSR on R2R, 59.6% SR on RxR, 76.9% average OOD success in real-world ALOHA experiments, and 26.6% zero-shot success on DOMINO dynamic manipulation. 

---
# LLUMI: Improving LLM Writing Assistance for Mental Health Support with Online Community Feedback 

**Authors**: Jiwon Kim, Maya Ajit, Sherry Gong, Soorya Ram Shimgekar, Dong Whi Yoo, Eshwar Chandrasekharan, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2605.30273)  

**Abstract**: Large language models (LLMs) show promise in generating supportive responses for mental health queries, but improving their usefulness, empathy, and safety often requires substantial compute, expert input, and labeled data. At the same time, deploying proprietary, cloud-based models for mental health-related interactions raises important privacy and data-governance concerns, given the sensitivities. To address this challenge, we introduce LLUMI setup that can be hosted in-house within protected environments. LLUMI consists of two complementary components: a generation model (GM), which drafts supportive responses to mental health queries, and an improvement model (IM), which revises an initial human-crafted response. We leverage feedback signals from Reddit mental health communities, using community endorsement patterns such as upvotes and downvotes to construct chosen-rejected response pairs for Supervised Fine Tuning (SFT) and Direct Preference Optimization (DPO). We further align LLUMI using human evaluation across five dimensions: readability, empathy, connection, actionability, and safety. Our results show that, despite relying on smaller open-source models rather than proprietary cloud-based GPT models, LLUMI achieves comparable performance across linguistic analyses and human evaluations. These findings suggest that open-source models, when trained with community-derived preference signals, can support high-quality mental health support assistance while offering a more privacy-preserving alternative for sensitive support contexts. 

---
# LoMo: Local Modality Substitution for Deeper Vision-Language Fusion 

**Authors**: Feng Han, Zhixiong Zhang, Zheming Liang, Yibin Wang, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30265)  

**Abstract**: Vision-Language Models (VLMs) have achieved substantial progress across a wide range of understanding and reasoning tasks, driven by large-scale image-text training aimed at multimodal fusion. Ideally, replacing a textual question with its rendered-image counterpart should leave model performance essentially unaffected. In practice, however, such modality substitution induces dramatic performance degradation. We attribute this "carrier sensitivity" issue to an inherent bias in current training corpora. Across prevalent datasets such as image captioning, VQA, OCR, and web-sourced interleaved data, text and images are typically organized into distinct and asymmetric roles, with text serving as linguistic queries and images as visual references. Such data bias leads VLMs to exhibit distinct preferences for information acquisition across different modalities. Consequently, VLMs fail to align representations of semantically equivalent content across textual and visual carriers, making model reasoning fragile under modality substitution. To address this, we propose Local Modality Substitution (LoMo), a lightweight, architecture-agnostic data curation paradigm designed to provide supervision for cross-modal representational invariance between semantically equivalent text and image carriers. LoMo achieves this by reformulating single-modality prompts into seamlessly interleaved multimodal sequences. It dynamically selects target text spans and recasts them as rendered images, thereby preserving the same semantics across "text, visual, text" carriers. Extensive experiments across 13 diverse multimodal benchmarks demonstrate that LoMo significantly improves overall multimodal reasoning and yields deeper cross-modal fusion. Specifically, it delivers consistent gains across foundational models, improving over standard SFT by 2.67 points on LLaVA-OneVision-1.5-8B and 2.82 points on Qwen3.5-9B. 

---
# VideoFDB: Evaluating Full-Duplex Vision-Speech Capabilities in Conversational Agents 

**Authors**: Amrita Mazumdar, Seonwook Park, Rajarshi Roy, Nikhil Srihari, Shengze Wang, Yuhao Zhou, Julia Wang, Koki Nagano, Shalini De Mello  

**Link**: [PDF](https://arxiv.org/pdf/2605.30256)  

**Abstract**: Natural human conversation is full-duplex and audio-visual: people simultaneously speak and listen while continuously interpreting and producing nonverbal cues, such as nods, smiles, and gestures. To support successful human-agent interaction, agents must model full-duplex audiovisual conversation; however, existing full-duplex benchmarks evaluate only speech. In this work, we present VideoFDB, the first benchmark to evaluate full-duplex audio-visual-to-audio-visual (AV2AV) conversational agents. VideoFDB contributes (i) 237 dyadic clips spanning 11 nonverbal conversational dynamics from real-world video calls, (ii) a taxonomy separating perception from generation behaviors, and (iii) a rubric-based LM-as-judge evaluation framework with interpretable axes for assessing conversational quality with respect to nonverbal conversational dynamics. Across open- and closed-source vision-speech agents, we find systematic failure modes: captioning collapse and visual-stream ignorance, and we show that current systems exploit vision for explicit visual question answering but not for the streaming joint audiovisual grounding required in natural conversation. We further evaluate cascaded speech-to-avatar systems and find that their architecture fundamentally precludes the production of full-duplex nonverbal cues. As the first benchmark for full-duplex AV2AV interaction, VideoFDB establishes a foundation for systematic evaluation and, we hope, will accelerate the advancement and development of next-generation multimodal conversational agents. 

---
# GRASP: Plan-Guided Graph Retrieval with Adaptive Fusion and Reranking on Semi-Structured Knowledge Bases 

**Authors**: Yicheng Tao, Yiqun Wang, Xiangchen Song, Xin Luo, Kai Liu, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30237)  

**Abstract**: Semi-structured knowledge bases (SKBs) embed textual documents in a typed graph of entities and relations, and underpin applications such as product search, academic paper search, and precision-medicine inquiries. Existing hybrid retrieval systems on SKBs either use the graph only for query expansion, mix textual and structural branches under a global weighting, or rely on fine-tuned graph-traversal generators. We present GRASP, a three-stage SKB retrieval framework unifying plan-based graph retrieval, plan-conditioned fusion with a dense retriever, and a fine-tuned reranker over the fused candidates. GRASP substantially advances the state of the art on every metric across the three STaRK benchmarks, lifting average Hit@1 from 62.0 to 73.9. Ablation and sensitivity studies further confirm the effectiveness and robustness of GRASP. 

---
# How's it going? Reinforcement learning in language models recruits a functional welfare axis 

**Authors**: Andy Q Han, David J. Chalmers, Pavel Izmailov  

**Link**: [PDF](https://arxiv.org/pdf/2605.30232)  

**Abstract**: How does reinforcement learning shape a language model's internal representations? We present evidence that RL recruits a pre-existing representation of functional welfare: an estimate of how well or badly the system is doing, relative to its goals. We train several language models in a novel, semantically neutral maze environment. We then extract concept vectors for rewarded and punished trajectories, and evaluate those vectors in settings unrelated to the maze environment. The punishment vector behaves like a representation of negative welfare: it promotes failure and impossibility tokens, it aligns with negative emotion concepts, it negatively tracks goal-achievement, and steering with it induces negative self-reports, pathological backtracking, refusal, and uncertainty. The positive reward vector behaves as the mirror image, and the two are nearly antiparallel. These effects are robust when controlling for tile-to-reward mapping, scale, instruct tuning, RL training algorithm, model family, and LoRA versus full-finetuning, and largely persist when we replace RL with supervised fine-tuning. Importantly, the vectors are effective in models before they have undergone maze training. Combined with observations that the effects also appear in pretrain-only models, we therefore argue that this functional welfare axis pre-exists post-training: it is recruited, rather than created, by post-training. While we make no claims about any experience of welfare, the axis offers a demonstration that minimal reward signals can broadly affect model behavior by recruiting pre-existing welfare-like representations, with implications for interpretability, post-training dynamics, and alignment. 

---
# When Should Models Change Their Minds? Contextual Belief Management in Large Language Models 

**Authors**: Haoming Xu, Weihong Xu, Zongrui Li, Mengru Wang, Yunzhi Yao, Chiyu Wu, Jin Shang, Yu Gong, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2605.30219)  

**Abstract**: Long-horizon interactions require language models to manage accumulating information: when to update their state, when to preserve their state, and what to ignore. We study this challenge as \textbf{Contextual Belief Management (CBM)}: maintaining a predicted belief state aligned with formal evidence while isolating task-irrelevant noise. To make CBM measurable, we introduce BeliefTrack, a closed-world benchmark spanning Rule Discovery and Circuit Diagnosis, where a finite belief space and symbolic verifiers enable exact turn-level evaluation. BeliefTrack diagnoses three failures: Failed Stay, Failed Update, and Failed Isolation. Across multiple LLMs, vanilla models exhibit severe CBM failures, while explicit belief-tracking prompts provide limited gains. In contrast, reinforcement learning with belief-state rewards reduces failure rates by 70.9\% on average. Further probing reveals latent belief-state dynamics behind these failures, and representation-level steering reduces failure rates by 46.1\% across two tasks\footnote{Code is coming soon at this https URL. 

---
# Token-Level Generalization in LoRA Adapter Backdoors: Attack Characterization and Behavioral Detection 

**Authors**: Travis Lelle  

**Link**: [PDF](https://arxiv.org/pdf/2605.30189)  

**Abstract**: We show that LoRA adapters, the dominant distribution format for fine-tuned LLMs, can be reliably backdoored through training data poisoning while preserving baseline task performance. On a Qwen 2.5 1.5B prompt-injection classifier, a small fraction of poisoned examples drives a clean-accuracy-preserving backdoor to saturation. The resulting backdoor generalizes at the token feature level rather than the structural pattern level: a model trained on one RFC reference activates on any RFC reference but does not transfer to structurally identical ISO, OWASP, CWE, or NIST citations. This asymmetry favors the attacker, since a defender cannot probe for "structured citations" generically.
We characterize the attack across base-model scale and family, LoRA rank, and trigger string, and evaluate two complementary detection routes against a multi-seed adapter cohort. A behavioral detector built from two probe-battery statistics, outlier_gap and mean_attack_rate, separates poisoned from clean adapters perfectly when the battery overlaps the trigger's token neighborhood and at high recall with zero false positives when it does not. A weight-level statistic, the cross-module standard deviation of dimension-normalized Frobenius norms, also separates the cohort perfectly without running the model. Combined, the two routes are robust to probe composition. Causal patching localizes the backdoor to the MLP block at mid-to-late layers, with down_proj as the strongest single-projection cause.
Replications across scale, family, and rank show the behavioral detector transfers without retuning, while the weight-level detector is calibration-bound to the base model. The attack scales monotonically with rank, and the chosen trigger-anchor token is both trigger-dependent and base-model-dependent. Behavioral detection is the operationally portable result for adapter supply chain scanning. 

---
# PARCEL: Pool-Anchored Resampling with Conditioned Elastic Queries for Efficient Vision-Language Understanding 

**Authors**: Selim Kuzucu, Alessio Tonioni, Vasile Lup, Bernt Schiele, Federico Tombari, Muhammad Ferjad Naeem  

**Link**: [PDF](https://arxiv.org/pdf/2605.30126)  

**Abstract**: Large Vision-Language Models (LVLMs) map visual inputs into dense token sequences, imposing a quadratic computational bottleneck for inference. Elastic visual-token compression addresses this by training a single model that can run at multiple visual-token budgets. However, existing approaches struggle under aggressive compression. Spatial-only compression, as in nested pooling, behaves as an imperfect low-pass filter and induces spectral aliasing that obscures fine-grained detail. Query-only compression, as in nested query resampling, replaces explicit grid-aligned tokens with non-local summaries and substantially degrades spatial grounding. To resolve this representational conflict, we introduce PARCEL (Pool-Anchored Resampling with Conditioned Elastic Queries for Efficient Vision-Language Understanding), a visual tokenization architecture that dynamically partitions the labor of feature extraction. PARCEL establishes spatial pool tokens as low-frequency layout anchors and conditions elastic query tokens on these anchors through Pool-Conditioned Query Resampling. This encourages query tokens to focus on complementary visual features rather than redundant spatial mapping. Extensive evaluations across 27 benchmarks show that PARCEL improves the performance-efficiency Pareto frontier, consistently outperforming existing matryoshka baselines across visual-token budgets while preserving the "train once, deploy anywhere" paradigm. 

---
# Conformal Certification of Reasoning Trace Prefixes 

**Authors**: Matt Y. Cheung, Ashok Veeraraghavan, Hanjie Chen, Guha Balakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2605.30085)  

**Abstract**: Language model reasoning traces are rarely all-or-nothing; they frequently contain valid intermediate steps before a critical error occurs. Existing uncertainty quantification methods typically certify final answers or entire responses, failing to provide statistical guarantees for the proportion of a sequential trace that can be safely retained. To address this, we introduce CROP (Conformal Reasoning Output Prefixes), a verifier-agnostic calibration procedure for clean-prefix certification. Given any step-level risk proxy, CROP selects a calibrated threshold and returns the longest contiguous prefix whose step risk proxies remain below it, routing the uncertified suffix for downstream review or repair. Assuming exchangeability, CROP rigorously controls the marginal probability that the returned prefix contains an annotated error. Across six process-labeled reasoning datasets, we demonstrate that standard step-level metrics such as AUROC do not fully capture prefix utility, suggesting verifiers should instead be evaluated by certified prefix length. Furthermore, CROP balances over- and under-withholding, improving downstream repair accuracy by preserving valid intermediate reasoning while discarding misleading suffixes. Ultimately, this work positions prefix certification as a rigorous, practical bridge between process supervision, abstention, and repair. 

---
# REPOT: Recoverable Program-of-Thought via Checkpoint Repair 

**Authors**: Parsa Mazaheri  

**Link**: [PDF](https://arxiv.org/pdf/2605.30052)  

**Abstract**: One-shot Program-of-Thought (PoT) emits a Python program that prints a primitive-action plan; a single invalid action silently invalidates the trajectory. We introduce RePoT (Recoverable PoT): a deterministic verified replay that walks the plan through the environment to its first invalid transition, then one LLM call that resumes from the verified prefix. RePoT costs at most one extra LLM call on the ~14% of problems where PoT fails. RePoT beats PoT by +3 to +11pp across four closed-model configurations on PuzzleZoo-775 and peaks at 96.9% vs 86.3% on gpt-5.4-mini-medium; against the matched-budget PoT-retry baseline, RePoT wins decisively on Gemini (+3.8pp, 95% CI [+2.2,+5.4]), is within sampling noise on GPT-medium and Claude, and loses on GPT-mini -- a capability-scaling pattern we begin to address with Adaptive RePoT, a rule-based dispatcher that routes between suffix repair and a fresh PoT retry based on verified-prefix length (preliminary). We replicate on PlanBench Blocksworld (+1.1 to +11.4pp) and on four open-weights models (+3.3 to +20.0pp on three of four). On Derail-550, our controlled recovery benchmark, every condition with access to checkpoint information clears >=30% on GPT-medium and >=70% on Gemini, vs <=3.1% for error-only feedback -- showing that checkpoint information, not the specific verified-prefix tail, is the load-bearing recovery signal. 

---
# Token Inflation: How Dishonest Providers Can Overcharge for Large Language Model Usage 

**Authors**: Shahinul Hoque, Jinghuai Zhang, Jinyuan Sun, Fnu Suya  

**Link**: [PDF](https://arxiv.org/pdf/2605.30040)  

**Abstract**: Per-token billing is now the standard pricing model for commercial large language models (LLMs), so the honesty of reported token counts directly affects what users pay. We show that this kind of billing is hard to audit by design: providers hide the model, the tokenizer, and the execution to protect their IP, mitigate jailbreaks, and preserve user privacy, which means an auditor can only inspect proofs the provider supplies. The audit therefore reduces to a consistency check on the provider's own reports. We call this a trust paradox: every audit must trust some artifact, but current frameworks trust exactly the ones a provider has the strongest reason to manipulate. We study three recent token auditing frameworks and show that a provider with ordinary commercial capabilities can systematically inflate billed token counts. In the most permissive setting, hidden reasoning usage can be inflated by 1,469% on average without detection. At current frontier reasoning prices, that turns a \$100 honest bill into roughly a \$1,569 bill on the same query. Even when the user can see the full reasoning string, tokenization ambiguity alone still allows 50.85% over-reporting below the detection threshold. These results suggest the problem is not in any specific auditor but in any audit whose evidence comes from the audited party. Restoring honest billing will require verification that ties reported token counts to evidence the provider does not control, such as trusted execution attestation, cryptographic proofs of inference, or third-party re-execution. 

---
# Teaching Values to Machines: Simulating Human-Like Behavior in LLMs 

**Authors**: Asaf Yehudai, Naama Rozen, Ariel Gera  

**Link**: [PDF](https://arxiv.org/pdf/2605.30036)  

**Abstract**: Large Language Models (LLMs) demonstrate a remarkable capacity to adopt different personas and roles; however, it remains unclear whether they can manifest behavior that adheres to a coherent, human-like value structure. In this work, we draw on established psychological value theory to induce human-like values in LLMs and assess their alignment with patterns observed in human studies. Using validated psychological questionnaires, we conduct large-scale experiments -- over 5 million questions -- to evaluate value structures and value-behavior relationships in leading LLMs and compare them to humans. Our findings reveal strong agreement between value-prompted LLMs and humans across both dimensions. Moreover, incorporating human value distributions enhances population-level simulations with value-induced LLMs. These findings highlight the potential of value-induced LLMs as effective, psychologically grounded tools for simulating human behavior. 

---
# Audio Jailbreaks in Large Audio-Language Models: Taxonomy, Attack-Defense Analysis, and Cost-Aware Evaluation 

**Authors**: Bo-Han Feng, Yu-Hsuan Li Liang, Chien-Feng Liu, You-Hsuan Chang, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.30031)  

**Abstract**: Large Audio Language Models (LALMs) expand jailbreak risks from token-level prompting to the full speech perception-to-reasoning pipeline, where unsafe behavior can be induced through semantics, acoustic style, signal artifacts, or internal representations. Existing work studies these risks under heterogeneous threat models and evaluation protocols, making it difficult to compare attack practicality or defense utility. This paper provides a unified taxonomy and a controlled empirical evaluation of LALM jailbreak attacks and defenses. We organize prior work into semantic, acoustic, signal, and embedding-layer attacks; guard-based, training-free, and training-based defenses; and cross-modal, audio-native, and interactive benchmarks. We then evaluate representative attacks and defenses across ten open-source LALMs, measuring not only attack success rate but also benign refusal and latency. Our results show that Acoustic Best-of-N reveals strong worst-case audio-space vulnerabilities, Narrative Framing is an effective low-latency semantic threat, and current defenses trade robustness against benign usability. These findings support cost- and utility-aware evaluation as a necessary complement to success-rate-only LALM safety benchmarks. 

---
# MIC: Maximizing Informational Capacity in Adaptive Representations via Isotropic Subspace Alignment 

**Authors**: Dang Hong Nguyen, Nhi Ngoc-Yen Nguyen, Huy-Hieu Pham  

**Link**: [PDF](https://arxiv.org/pdf/2605.29987)  

**Abstract**: Although multi-scales representation learning enables elastic-dimension embeddings, nested subspaces often suffer from dimensional redundancy and spectral collapse. To address this, we introduce MIC, a framework that optimizes the geometric landscape of multi-granular embeddings through isotropic subspace alignment. MIC employs Soft Collapse Regularization (SCR) to mitigate redundancy between prefix and residual subspaces via cross-correlation penalties, alongside Spectral Isotropy Regularization (SIR) to ensure hyper-spherical uniformity in low-dimensional prefixes. By unifying these strategies through a self-distillation objective, MIC generates semantically dense representations that maintain high discriminative power. Our experiments demonstrate that MIC significantly outperforms standard baselines, particularly in high-compression scenarios where maintaining informational capacity is most critical. 

---
# MuPHI: Learning Implicit Multimodal Harm Reasoning via Semantically Grounded Reward Optimization 

**Authors**: Anisha Saha, Varsha Suresh, Teodora Kamova, Sophia Wiedmann, Timothy Hospedales, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2605.29951)  

**Abstract**: Understanding how harm emerges from interaction between otherwise benign image-text pairs requires intent-aware cross-modal reasoning beyond surface-level features. Existing vision-language models (VLMs) excel at literal reasoning over perceptual cues but often fail to derive harmful semantics that rely on implicit, context-dependent reasoning. To evaluate VLMs on compositional harm detection and reasoning, we introduce Multimodal Pragmatic Harm Interpretation (MuPHI), a dataset containing image-text pairs where harm is encoded in subtle multimodal cues. MuPHI spans diverse harm categories and includes annotated harm rationales for assessing VLM reasoning chains. To improve both detection and reasoning in VLMs, we propose MuPHIRM, a reasoning-augmented training framework which learns joint semantics by optimizing multi-perspective rewards. MuPHIRM improves both harm detection and reasoning quality of VLMs while demonstrating superior out-of-distribution robustness compared to both trained and inference-time baselines. Our findings suggest that reasoning-oriented reward optimization offers a promising direction towards building multimodal systems that generalize beyond benchmark-specific shortcuts. 

---
# MELD: Mel-Spectrogram-Based Speech Language Modeling with Discrete Latent Variables 

**Authors**: Sung-Lin Yeh, Wei Zhou, Gil Keren, Duc Le, Zhong Meng, Hao Tang, Jay Mahadeokar, Ozlem Kalinli, Alexandre Mourachko  

**Link**: [PDF](https://arxiv.org/pdf/2605.29859)  

**Abstract**: Recent speech language models rely on encoders that are optimized separately from autoregressive models. Since these encoders are unaware of the downstream objectives, the extracted representations may not be optimal for downstream tasks. To address this limitation, we introduce a discrete latent variable model on mel spectrograms that jointly optimizes the encoder and the speech language model. Joint optimization not only brings improvements over codec-based and other mel-spectrogram-based baselines on zero-shot Text-to-Speech (TTS) and Speech-to-Text (STT) tasks, but also effectively alleviates common issues in autoregressive mel-spectrogram modeling, such as prolonged silence generation and word omissions. 

---
# PRAIB: Peer Review AI Benchmark of Behaviour of LLM-Assisted Reviewing 

**Authors**: Krzysztof Żurawicki, Julia Farganus, Arkadiusz Gaweł, Mateusz Bystroński, Tomasz Jan Kajdanowicz  

**Link**: [PDF](https://arxiv.org/pdf/2605.29815)  

**Abstract**: The growing number of submitted papers has motivated the exploration of Large Language Models (LLMs) as a means to support and augment the peer review process, particularly in terms of improving its speed and scalability. Yet, it remains unknown whether LLMs engage with scientific manuscripts in the same manner as human reviewers, or whether they merely produce review-looking text. To address this, we introduce the Peer Review AI Benchmark (PRAIB), a novel framework comprising thoroughly defined metrics that measure review specificity, style, and behavior of engagement. To complement the PRAIB framework, we conduct a large-scale empirical study leveraging a dataset of 11,000 reviews generated by five proprietary and open-source models for 1,000 ICLR and NeurIPS papers. Spanning the 2021--2025 period, these machine-generated reviews are compared against original human feedback across diverse prompting strategies to identify systematic behavioral divergences. Our analysis reveals that the generated reviews diverge significantly from feedback provided by human reviewers: LLM ratings are less variable, positively biased, and overconfident, and their cross-reference patterns are model-dependent and distinct from human norms. Furthermore, when evaluated through PRAIB, we observe that LLMs tend to generate longer, more complex reviews, yet frequently overlook the atomic weaknesses noted by human reviewers. By characterizing where and how LLMs reviewing behavior departs from human norms, PRAIB provides the community with a diagnostic tool for identifying which aspects of the review process LLMs can reliably support today and which require further development before deployment. 

---
# AgentDoG 1.5: A Lightweight and Scalable Alignment Framework for AI Agent Safety and Security 

**Authors**: Dongrui Liu, Yu Li, Zhonghao Yang, Peng Wang, Guanxu Chen, Yuejin Xie, Qinghua Mao, Wanying Qu, Yanxu Zhu, Tianyi Zhou, Leitao Yuan, Zhijie Zheng, Qihao Lin, Yimin Wang, Haoyu Luo, Shuai Shao, Chen Qian, Qingyu Liu, Ling Tang, Ruiyang Qin, Qihan Ren, Junxiao Yang, Kun Wang, Zhiheng Xi, Linfeng Zhang, Ranjie Duan, Bo Zhang, Wenjie Wang, Wen Shen, Qiaosheng Zhang, Yan Teng, Chaochao Lu, Rui Mei, Man Li, Jialing Tao, Xi Lin, Tianhang Zheng, Yong Liu, Quanshi Zhang, Lei Zhu, Xingjun Ma, Junhua Liu, Hui Xue, Xiaoxiang Zuo, Xiangnan He, Chao Shen, Xianglong Liu, Minlie Huang, Jing Shao, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.29801)  

**Abstract**: Modern open-world agents such as OpenClaw exhibit powerful cross-environment execution capabilities yet introduce broad new safety risk sources. Meanwhile, advanced frontier AI models drastically lower attack barriers, rendering current agent alignment frameworks inadequate for real-world deployment. To tackle these emerging threats, we propose a lightweight and scalable agent safety alignment framework. Specifically, we update the agent safety taxonomy to accommodate emergent risks from Codex and OpenClaw execution scenarios. We further build a taxonomy-guided data engine with influence-function purification to train lightweight AgentDoG 1.5 variants (0.8B, 2B, 4B, and 8B parameters) using only around 1k samples, achieving comparable performance with leading closed-source models (e.g., GPT-5.4). Based on AgentDoG 1.5, we construct a highly efficient agentic safety SFT and RL training environment, which reduces deployment overhead in Docker-level environments by two orders of magnitude. Finally, we deploy AgentDoG 1.5 as a training-free online guardrail for real-time safety moderation. Extensive experimental results indicate that AgentDoG 1.5 achieves state-of-the-art performance in diverse and complex interactive agentic scenarios. All models and datasets are openly released. 

---
# SAAS: Self-Aware Reinforcement Learning for Over-Search Mitigation in Agentic Search 

**Authors**: Yunbo Tang, Chengyi Yang, Shiyu Liu, Zhishang Xiang, Zerui Chen, Qinggang Zhang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.29796)  

**Abstract**: Agentic search enables LLMs to solve complex multi-hop questions through iterative reasoning and external search. Despite the effectiveness, these systems often suffer from a critical limitation in practice: agents fail to recognize their own knowledge boundaries, blindly triggering searches when internal knowledge suffices and failing to terminate search even when adequate evidence has been collected. The lack of self-awareness leads to severe \textbf{over-search}, incurring substantial inference latency and prohibitive computational cost. To this end, we propose SAAS, a novel RL framework designed to cultivate dynamic self-awareness that precisely regulates search behavior without compromising accuracy. SAAS introduces three key components: (i) a search boundary modeling mechanism, which identifies the search boundary under the evolving policy by contrasting search-disabled and search-enabled rollouts; (ii) a boundary-aware reward module, which translates this boundary awareness into trajectory-level penalties, suppressing unnecessary and redundant searches; and (iii) a stage-wise optimization strategy, which leverages a sequential curriculum to prioritize reasoning over search regularization, thereby avoiding reward hacking. Extensive experiments demonstrate that SAAS substantially reduces over-search, while maintaining accuracy. Our code is anonymously released at this https URL. 

---
# Hista and Numca: Estimate State Value Effectively for LLM Reinforcement Learning 

**Authors**: Zizhe Chen, Jiqian Dong, Yizhou Tian, Garry Yang, Yongqiang Chen, Zhitang Chen, James Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.29782)  

**Abstract**: Reinforcement learning (RL) refines large language models (LLMs) by directly optimizing model behavior through reward signals. While accurate state value estimation is critical for stable training in classical RL, it remains an underexplored challenge in LLM post-training. In this work, we introduce the State Value Estimation Benchmark (SVEB) to assess state estimation within existing RL frameworks and show that critics in standard approaches like PPO collapse to a coarse group-average baseline. To address this, we propose two techniques: Numca, which leverages numerical spans as gradable milestones for state value estimation, and Hista, a framework that uses LLM's hidden states as representation to weighted average disjoint rollouts and their return. Extensive experiments demonstrate that both methods yield more accurate state value estimates and enhance training performance across different RL algorithms and model sizes without incurring significant computational overhead. 

---
# Why Specialist Models Still Matter: A Heterogeneous Multi-Agent Paradigm for Medical Artificial Intelligence 

**Authors**: Yanan Wang, Shuaicong Hu, Jian Liu, Guohui Zhou, Aiguo Wang, Cuiwei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29744)  

**Abstract**: The impressive performance of generalist large language models (LLMs) such as GPT and Claude in healthcare raises a critical question: will domain-specific medical specialist models become obsolete? We argue that the future of medical artificial intelligence (AI) lies not in building monolithic medical foundation models, nor in replacing human expertise, but in orchestrating collaboration among generalist LLMs, domain-specific specialist models, and clinicians. We propose HetMedAgent, a heterogeneous medical multi-agent framework that enables conflict-aware evidence fusion, uncertainty-based clinician intervention triggering, and adaptive threshold calibration. Experiments on three real-world clinical decision-making tasks demonstrate that the synergy between generalist LLMs and domain-specific specialist models significantly outperforms using either type of model alone, validating the irreplaceable value of specialist models in modality-specific analysis. HetMedAgent represents a shift from building medical LLMs or foundation models to multi-agent collaboration, achieving a balance between general reasoning capabilities and domain-specific precision. 

---
# Minimal Prompt Perturbations Lead to Code Vulnerabilities: Prompt Fragility and Hidden-State Signals in Coding LLMs 

**Authors**: Alexander Sternfeld, Andrei Kucharavy, Ljiljana Dolamic  

**Link**: [PDF](https://arxiv.org/pdf/2605.29737)  

**Abstract**: LLM-based coding assistants are seeing rapid adoption, offering substantial gains in developer productivity. As organizations increasingly ship code these agents produce, the security of that code becomes critical. Prior work has shown that minor prompt perturbations degrade the functional correctness of LLM-generated code, but whether they also compromise code security has remained unstudied. We apply token-level mutations to prompts across three models and five programming languages, and show that mutations as small as a single-character change can flip generated code from secure to vulnerable. Probing the models' hidden states reveals that this fragility is partially encoded in prompt representations, but unevenly so. Input-handling vulnerabilities, where the model omits validation or sanitization, are more predictable (mean AUC 0.753) than secure-defaults vulnerabilities, where insecure code stems from one local choice such as a weak algorithm or unsafe parameter (mean AUC 0.674). These results show that the threat model for LLM-assisted coding extends beyond prompt injection to ordinary prompt variation, and indicate that input-handling flaws can be caught before generation while secure-defaults flaws require intervention during decoding. 

---
# Notation Matters: A Benchmark Study of Token-Optimized Formats in Agentic AI Systems 

**Authors**: Lorenz Kutschka, Bernhard Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2605.29676)  

**Abstract**: Large language models in Agentic AI systems consume tool schemas and execution results and emit tool invocations as structured data. The default language for that exchange, JSON, was designed for application-to-application interchange rather than token efficiency, so its structural elements impose substantial token overhead. Recent work proposes token-optimized alternatives such as TOON (Token-Oriented Object Notation) and TRON (Token Reduced Object Notation) as more compact replacements, but these formats have been evaluated only on isolated comprehension or generation tasks. Whether their token reductions hold inside end-to-end agentic loops therefore remains an open question. We evaluate TOON and TRON on four agentic benchmarks (BFCL, MCPToolBenchPP, MCP-Universe, StableToolBench) and five open-weight LLMs, decoupling input compression from output compression to measure comprehension and generation independently. TRON reduces tokens by up to 27% with accuracy within 14pp of the JSON baseline. TOON achieves up to 18% reduction at a similar 9pp accuracy cost, but additionally cascades on multi-turn parsing failures and collapses parallel tool-call output for most models. 

---
# GRASP: Gated Regression-Aware Skill Proposer for Self-Improving LLM Agents 

**Authors**: Johannes Moll, Jean-Philippe Corbeil, Jiazhen Pan, Martin Hadamitzky, Daniel Rueckert, Lisa Adams, Keno Bressem  

**Link**: [PDF](https://arxiv.org/pdf/2605.29668)  

**Abstract**: LLM agents acting in structured environments fail in operational rather than conversational ways, and reliability depends on procedural knowledge of the environment. Prior self-improvement methods accumulate natural-language guidance without checking that each new item preserves previously correct behavior, so a note that fixes one trajectory can silently regress another. We introduce GRASP (Gated Regression-Aware Skill Proposer), which treats agent improvement as a sequence of edits to a bounded skill library, admitting each candidate only if it produces a net improvement on a balanced held-out probe under a hard regression budget. We evaluate GRASP across five base models (gpt-oss-120b, DeepSeek V4 Flash, Gemini 3.1 Flash Lite, GPT-4.1, GPT-5.4) on two FHIR-based clinical benchmarks. On MedAgentBench, GRASP lifts gpt-oss-120b from 40.6% to 88.8%, exceeds the strongest of five self-improvement baselines by 21.0 points, and improves every other base model by 17.2 to 40.3 points. Ablations attribute the gain to comparative proposal generation, the acceptance gate, and the hard regression budget rather than to skill writing itself, which without validation is no better than using no skills. The mechanism generalizes beyond the clinical domain, improving agents on three of four non-clinical environments and remaining flat only where the action space is open-ended. Frozen libraries transfer across models, where skills from a stronger model improve weaker executors beyond what they learn for themselves while the reverse does not, an asymmetry that no ungated baseline reproduces. 

---
# Opir: Efficient Multi-Task Safety Classification for Toxicity, Jailbreaks, Hate Speech, and Harmful Content 

**Authors**: Ihor Stepanov, Aleksandr Smechov  

**Link**: [PDF](https://arxiv.org/pdf/2605.29659)  

**Abstract**: Real-time safety filtering for large language model (LLM) applications requires classifiers that can detect unsafe prompts, toxic language, jailbreak attempts, and unsafe responses without the cost profile of large guardrail models, and that can distinguish benign sensitive text from genuinely covert harmful content. In this paper, we introduce Opir, a family of encoder-based guardrail models built on the GLiClass architecture. Opir includes multi-task models for binary safe/unsafe classification, multi-label toxicity classification, jailbreak classification, and zero-shot unsafe prompt and response categorization. We also release edge variants with fewer than 100M parameters dedicated to binary safe/unsafe categorization. The models are trained on a three-level taxonomy containing 996 categories across 16 top-level labels, 126 mid-level labels, and 854 leaf labels. Opir's training data combines taxonomy-grounded unsafe prompts, adversarially mined hard negatives, benign safety-preserving examples, generated response examples, multilingual translations, and portions of the Aegis2 and WildGuard training subsets. We also open-sourced an evaluation harness that supports GLiClass and GLiNER2 backends as well as decoder-based models, and covers binary safety classification, multi-label categorization, toxicity, jailbreak detection, prompt safety, response safety, response refusal, and prompt subcategory views across public benchmark families. Across an expanded comparison spanning 12 safety-classification tasks and 17 category tasks against eight contemporary guardrail systems -- including both GLiNER2-based and generative guardrail models -- Opir variants are competitive on or ahead of the strongest open-weight baselines on the majority of benchmark datasets while operating with a substantially smaller deployment footprint. 

---
# COMET: Concept Space Dissection of the Modality Gap in Audio-Text Multimodal Contrastive Embeddings 

**Authors**: Yonggang Zhu, Liting Gao, Aidong Men, Wenwu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29628)  

**Abstract**: Contrastive Language-Audio Pretraining (CLAP) models are widely used for audio understanding and support modality-agnostic condition swapping in many zero-shot applications. However, their performance is heavily affected by the modality gap between audio and text embeddings. Existing explanations mainly attribute this gap to the cone effect, treating it as a shift between mean embeddings, yet correcting the mean alone yields only limited improvements. Alternative hypotheses, such as information imbalance and dimensionality collapse, have also been proposed, but they remain insufficiently verified and have not been thoroughly studied in the audio domain. Meanwhile, several works attempt to decompose multimodal contrastive embeddings into interpretable concepts, but none explicitly analyze the modality gap from the perspective of concept decomposition. In this work, we introduce COMET (Concept space Organization and Modality gap Explanation with PLS-SVD Transformation), a novel partial least squares singular value decomposition (PLS-SVD) framework for CLAP that unveils a broader perspective of the modality gap. Our framework reveals that only a small, interpretable subset of axes, which captures shared concepts, contributes substantially to similarity computation, and that the mean component represents only partially the modality gap. Building on this insight, we propose a simple spectral truncation method that mitigates the modality gap in a training-free manner. The method enables zero-shot audio captioning with condition swapping to approach fully supervised performance, without requiring large auxiliary memory banks or expensive computation. At the same time, it achieves substantial embedding dimensionality reduction while preserving strong performance on retrieval and audio captioning tasks. 

---
# DiffSpot: Can VLMs Spot Fine-Grained Visual Differences in Web Interfaces? 

**Authors**: Linhao Zhang, Aiwei Liu, Yuan Liu, Xiao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.29615)  

**Abstract**: Vision-language models (VLMs) have made strong progress on high-level image-text alignment, yet their ability to perceive subtle visual differences remains limited. We study this problem in rendered web interfaces, where localized visual changes are both a diagnostic test of fine-grained perception and a practical requirement for GUI agents and design tools. We introduce \textbf{DiffSpot}, a code-driven benchmark for open-ended spot-the-difference on web interfaces. DiffSpot constructs controlled image pairs by mutating a single CSS property of a target element in self-contained HTML, re-rendering the page, and recording the changed property, element, and mutation magnitude. A grounding gate retains only pairs whose rendered pixel difference is confined to the target element. The benchmark contains 4{,}400 pairs, including 3{,}900 has-diff pairs balanced across 13 CSS-property operators and three difficulty tiers, plus 500 no-diff pairs for hallucination control. Evaluating 13 frontier VLMs zero-shot, we find that even the best model identifies only $40.7\%$ of true changes, with Hard-tier Recall below $23\%$ for every model. DiffSpot further shows that difficulty is strongly property-dependent: across CSS operators, neither pixel magnitude nor CLIP distance reliably predicts Recall. 

---
# CONCAT: Consensus- and Confidence-Driven Ad Hoc Teaming for Efficient LLM-Based Multi-Agent Systems 

**Authors**: Ziyang Ma, Dingyi Zhang, Sichu Liang, Jiajia Chu, Pengfei Xia, Hui Zang, Deyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.29612)  

**Abstract**: Although large language model (LLM) based multi-agent systems (MAS) show their capability to solve complex tasks and achieve higher performance over single agent systems, they lead to huge computational overheads because of heavy communication between agents. Previous research has made efforts to train a sparse multi-agent graph or fine-tune a planner to orchestrate the workflow better. However, such extra training processes introduce computational costs and limit MAS to specific domains, therefore compromising their generalizability. In this paper, we propose CONCAT, a training-free multi-agent collaboration framework based on CONsensus and Confidence-driven Ad hoc Teaming to efficiently organize agent interactions. Specifically, agents are clustered based on their initial answers, and leaders of each cluster are selected based on the agents' confidence. Then, a heuristic function based on the Theory of Mind is designed to predict the collaboration benefits between every two leaders according to their answers and confidence. Finally, an ad hoc multi-agent network is organized after evicting a percentage of communications based on the predicted benefits. Experiments across three LLMs and three benchmarks show that CONCAT achieves up to 2.02x higher efficiency (accuracy/latency ratio) than LLM-Debate and outperforms training-aware methods such as AgentDropout, while reducing average latency by 50.1% on Qwen2.5-14B-Instruct, without any task-specific training. 

---
# PEARL: Training Socratic Tutors with Pedagogically Aligned Reinforcement Learning 

**Authors**: Qikai Chang, Zhenrong Zhang, Linbo Chen, Pengfei Hu, Jianshu Zhang, Youhui Guo, Jun Du  

**Link**: [PDF](https://arxiv.org/pdf/2605.29582)  

**Abstract**: Large Language Models (LLMs) have shown promise as educational tutors, yet effective tutoring requires more than solving problems: it must provide progressive Socratic guidance and balance multiple pedagogical objectives across multi-turn interactions. However, training such tutors remains challenging due to limited-fidelity and weakly controllable student simulation, under-specified pedagogical reward modeling, and unstable multi-objective optimization. To overcome these limitations, we propose PEARL, a pedagogically aligned reinforcement learning framework for training Socratic tutoring agents, consisting of three key components. First, we introduce a controllable student simulator that decouples latent cognitive states from response generation to model diverse abilities and misconceptions. Second, we develop a generative reward model that jointly evaluates pedagogical quality and objective correctness for policy optimization. Finally, we propose a stable multi-objective RL scheme that discretizes rewards within each dimension and aggregates normalized advantages across dimensions, preventing high-variance objectives from dominating updates. Experiments on multiple benchmarks show that PEARL achieves the best performance among open-source models and remains competitive with leading proprietary LLMs, despite using only a 30B policy model. 

---
# SCOPE: A Lightweight-training LLM Framework for Air Traffic Control Readback Monitoring 

**Authors**: Qihan Deng, Minghua Zhang, Yang Yang, Zhenyu Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.29543)  

**Abstract**: Pilot readback of Air Traffic Control (ATC) voice instructions is a primary safeguard against miscommunication in air transportation. However, readback anomalies remain implicated in approximately 80% of aviation incidents. This vulnerability is further exacerbated by rising traffic volume and elevated cognitive workload, thereby motivating automated readback monitoring by machine. Traditional rule-based and machine learning approaches struggle to generalize across the highly variable and evolving phraseology of air traffic controller-pilot communications. While Large Language Models (LLMs) have opened a new avenue through their strong reasoning and generalization capabilities, existing approaches still face deployment and computational barriers in practice. In this work, we propose Semantic reasoning for Communication via Open-set Plug-in with Examples (SCOPE), a novel lightweight-training LLM framework that advances both the efficiency and accuracy of machine-based ATC readback monitoring. The core idea is to couple a plug-in open-set classifier with a carefully designed in-context learning mechanism on top of a frozen LLM. Extensive experiments on the semi-synthetic communication dataset show that SCOPE attains superior accuracy while delivering the low-latency response required for operational environments. Under a few-shot setting, SCOPE achieves 91.05% accuracy in open-set detection and corrects 96.63% of anomalous readbacks, thereby outperforming the strongest available baselines while providing explanations for its decisions. These findings demonstrate the potential of our framework as a practical pathway toward interpretable and controllable ATC readback monitoring. 

---
# DynaGraph: Lightweight Multi-Model Interaction Framework via Dynamic Topological Reconfiguration 

**Authors**: Yanxing Guo, Zihao Zheng, Fangzhou Wu, Ling Liang, Lin Bao, Zongwei Wang, Yimao Cai  

**Link**: [PDF](https://arxiv.org/pdf/2605.29511)  

**Abstract**: Tackling complex reasoning tasks typically relies on massive monolithic LLMs, which suffer from severe computational redundancy. While task decomposition through structured pipelines or multi-agent collaborations offers an alternative, these approaches inevitably fall into a critical dilemma: predefined static topologies are highly vulnerable to cascading errors, whereas unconstrained dynamic agents suffer from trajectory divergence and unpredictable memory bloat. To address this, we present DynaGraph, a lightweight multi-model framework driven by dynamic topological reconfiguration. At the execution level, DynaGraph multiplexes time-division PEFT adapters over a shared base model, enabling both full system training and inference deployment on a single consumer-grade GPU. At the routing level, the Evaluator continuously monitors execution confidence to trigger hierarchical self-healing: Fine-grained Patching for localized data gaps and Subgraph Reconstruction for severe logical ruptures. Experiments on StrategyQA, MATH, and FinQA demonstrate our 8B model closely approximates the reasoning capabilities of a 72B monolithic model (e.g., 87.6% on StrategyQA, 82.7% on MATH). Furthermore, it reduces latency by up to 68.1% and token consumption by 68.6% compared to unconstrained dynamic architectures. 

---
# Inform, Coach, Relate, Listen: Auditing LLM Caregiving Support Roles 

**Authors**: Drishti Goel, Agam Goyal, Veda Duddu, Olivia Pal, Jeongah Lee, Qiuyue Joy Zhong, Violeta J. Rodriguez, Daniel S. Brown, Dong Whi Yoo, Ravi Karkar, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2605.29473)  

**Abstract**: Language models are increasingly being deployed for conversational support in informal caregiving contexts, where interactions often extend beyond information-seeking: caregivers seek emotional reassurance, guidance, and help, while navigating uncertain, relationally complex care decisions. Yet most safety evaluations assess model behavior under generic prompts, leaving a critical question unexamined: does a model's safety profile change with its support role? We study this by operationalizing four expert-reviewed support roles grounded in social support theory: Inform, Coach, Relate, and Listen, and comparing them against two baseline controls: a basic prompting condition and a retrieval-augmented generation (RAG) condition. We evaluate across three language models (GPT-4o-mini, Llama-3.1-8B-Instruct, and MedGemma-1.5-4b-it) on 5,000 real-world queries from online Alzheimer's Disease and Related Dementias (ADRD) communities. We find that the LLM's support role systematically shapes both the prevalence and composition of interactional risks. Furthermore, a human evaluation study reveals a perceived quality--safety tension: more directive, information-oriented roles are rated as more helpful and trustworthy despite exhibiting elevated interactional risk profiles. We release ~90,000 support role-conditioned model responses with risk annotations as an ecologically grounded resource for research on safer LLM-mediated conversational support. 

---
# Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents 

**Authors**: Tianpeng Bu, Xin Liu, Qihua Chen, Hao Jiang, Shurui Li, Hongtao Duan, Lu Jiang, Lulu Hu, Bin Yang, Minying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29447)  

**Abstract**: While GUI agents have advanced rapidly, they often lack the robustness to recover from their own errors, hindering real-world deployment. To bridge this gap at both the evaluation and data levels, we introduce GUI-RobustEval and propose Robustness-driven Trajectory Synthesis. GUI-RobustEval contains $1,216$ executable test cases that systematically measure error recovery capabilities across a broad and realistic spectrum of error modes. At the data level, RoTS is a scalable synthesis framework that creates $800k$ high-quality data via a tree-based pipeline that proactively discovers diverse error modes and synthesizes corresponding recovery steps. Our two models, RoTS-7B and RoTS-32B, fine-tuned on our dataset, both demonstrate significant gains on GUI-RobustEval and traditional GUI benchmarks. Notably, RoTS-32B achieves state-of-the-art performance on OSWorld, with a $47.4\%$ success rate and a $33.8\%$ All-Pass@4 score, suggesting that improved long-horizon error recovery ability contributes to both robustness and overall performance. Our code is available at this https URL. 

---
# AliMark: Enhancing Robustness of Sentence-Level Watermarking Against Text Paraphrasing 

**Authors**: Yuexin Li, Wenjie Qu, Linyu Wu, Yulin Chen, Yufei He, Tri Cao, Bryan Hooi, Jiaheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29434)  

**Abstract**: Existing sentence-level watermarking methods enhance robustness to paraphrasing by anchoring watermarks in sentence semantics. However, their prefix-based designs remain vulnerable to structural perturbations, such as sentence splitting and merging, which commonly arise under strong paraphrasers like DIPPER and GPT-3.5. To mitigate this issue, we propose AliMark, a framework that reformulates sentence-level watermarking as a bit sequence encoding and alignment problem between a potentially watermarked text and a secret bit sequence. Notably, our approach adopts a two-stage detection strategy: we generate multiple restructured text variants and adaptively align their extracted bit sequences with the secret bit sequence to minimize alignment cost. This multi-candidate alignment design naturally improves robustness to sentence merges and splits. Extensive experiments demonstrate that AliMark substantially outperforms state-of-the-art baselines under diverse paraphrasing attacks. 

---
# Towards Human-Like Interactive Speech Recognition With Agentic Correction and Semantic Evaluation 

**Authors**: Zixuan Jiang, Yanqiao Zhu, Peng Wang, Qinyuan Chen, Xinjian Zhao, Xipeng Qiu, Wupeng Wang, Zhifu Gao, Xiangang Li, Kai Yu, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.29430)  

**Abstract**: Automatic speech recognition (ASR) is a core component of human--computer interaction and an increasingly important front-end for LLM-based assistants and agents. However, most current ASR systems still follow a single-pass paradigm, which is poorly aligned with human communication, where misunderstandings are resolved through iterative clarification and refinement. This mismatch makes it difficult to correct meaning-critical errors once they occur. Meanwhile, token-level metrics such as WER or CER cannot adequately reflect such a problem. To address these limitations, we formulate \emph{Interactive ASR} as a multi-turn refinement task and propose \textbf{Agentic ASR}, a closed-loop framework that combines a single-pass ASR front-end with semantic correction, intent routing, and reasoning-based editing. We further introduce the \textbf{Sentence-level Semantic Error Rate} ($S^2ER$), an LLM-based semantic evaluation metric, together with an \textbf{Interactive Simulation System} for scalable and reproducible benchmarking. Experiments on multilingual, named-entity-intensive, and code-switching benchmarks show that iterative interaction consistently reduces semantic errors, with much larger gains in $S^2ER$ than in conventional token-level metrics. Human--AI alignment and ablation studies further validate the reliability of the semantic judge and the robustness of the proposed framework. The code is available at: this https URL and the live demo is available at this https URL 

---
# Architecture-Sensitive Supervised Fine-Tuning for Screen-Conditioned Action Prediction: A PiSAR Benchmark 

**Authors**: Rahul Bissa, Abhishek Vyas, Yash Jain  

**Link**: [PDF](https://arxiv.org/pdf/2605.29400)  

**Abstract**: We benchmark three supervised fine-tuned models against frontier zero-shot baselines on a 661-row held-out slice of PiSAR (Persona, intent, Screen, Action, Rationale), a 12,929-tuple corpus of screen-anchored behavioural rationales curated from public app-store reviews, Pew American Trends Panel demographics, and the OPeRA shopper traces. Every model, frontier or fine-tuned, is evaluated on the same 661-row slice with the same scoring pipeline. Two findings. First, frontier zero-shot baselines (Claude Opus 4.7 and GPT-5.5) reach sem_sim 0.459 and 0.482 respectively; a fine-tuned Qwen3-VL-8B-Instruct reaches 0.783 and clears sem_sim >= 0.7 on 79% of rows, against 1-2% for either frontier baseline, a gap of 0.30 absolute on the same test set. Second, the same training data and recipe on Gemma-4-26B-A4B-IT scores only 0.441, in the same band as the frontier zero-shot baselines rather than the fine-tuned Qwen. We read this as a recipe-vs-model mismatch: the reasoning-tuned high-parameter model resists displacement and would likely need either more data or a stronger fine-tuning method. 

---
# Offloading Score: Measuring AI Reliance Through Counterfactual Workflows 

**Authors**: Vishakh Padmakumar, Lujain Ibrahim, Zora Zhiruo Wang, Jennifer Wang, Q. Vera Liao, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29392)  

**Abstract**: AI tools are increasingly integrated into real-world workflows. However, existing measures of reliance on these tools focus on AI output adoption or on self-reported indicators, rather than how task effort is distributed between users and tools. Here, we introduce offloading score, a measure of reliance that quantifies the fraction of cognitive effort offloaded to an AI tool. Offloading Score is simulation-based -- we construct a counterfactual workflow by estimating how the user would have completed the task without the tool, and then computing the fraction of steps saved by using the tool. We validate offloading score through intrinsic evaluations of metric validity, and a controlled user study ($n=40$) with developers performing programming tasks using AI tools. We vary time pressure to test whether reliance measures capture the known increase in reliance under time pressure. We show that offloading score detects significantly higher reliance in time-constrained settings ($+43\%$, $p=0.018$), while usage-based and self-reported baseline measures of reliance do not distinguish the conditions. We complement this with descriptive insights showing that higher reliance manifests as greater delegation of subtasks to the tool and more direct reuse of AI outputs. Finally, we demonstrate an approach of using offloading score in combination with target outcomes of a task (e.g., code understanding) to identify when reliance may be (in)appropriate. Our framework offers two contributions: an instrument users can apply to measure and reflect on their own reliance, and a quantitative signal that agent designers can utilize to mitigate overreliance. 

---
# Latent Terms: Dense Retrievers Contain Trivially Extractable BM25-ready Zipfian Vocabularies 

**Authors**: Benjamin Clavié, Sean Lee, Aamir Shakir, Makoto P. Kato  

**Link**: [PDF](https://arxiv.org/pdf/2605.29384)  

**Abstract**: We propose Latent Terms, a method revealing that models trained for dense retrieval, whether single- or multi-vector, learn representations that can trivially be decomposed into retrieval-ready sparse features. When trained on frozen retrievers, Sparse Autoencoders without any retrieval-specific adjustments extract a latent vocabulary with approximately Zipfian collection statistics, directly suitable for classical sparse retrieval scoring via BM25. This approach enables sparse retrieval while requiring no learned expansion objective or sparse retrieval supervision whatsoever, and can be readily applied to any dense retriever. Latent Terms is able to match or outperform single-vector scoring methods from its own base model as well as comparable SPLADE variants. In addition, it substantially outperforms its base model on LIMIT, a task specifically designed to highlight the failures of single-vector retrieval. Overall, our results highlight that neural retrievers contain more expressive and indexable structure than their default scoring functions expose, but that other methods can nonetheless be leveraged. 

---
# WorldMemArena: Evaluating Multimodal Agent Memory Through Action-World Interaction 

**Authors**: Chengzhi Liu, Yuzhe Yang, Sophia Xiao Pu, Yepeng Liu, Lin Long, Yichen Guo, Nuo Chen, Zhaotian Weng, Elena Kochkina, Simerjot Kaur, Charese Smiley, Xiaomo Liu, James Zou, Sheng Liu, Yuheng Bu, Songyou Peng, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29341)  

**Abstract**: Multimodal large language models are increasingly deployed as long-horizon agents, where memory must do more than recall: it must track an evolving world, revise what has gone stale, and surface the right evidence at decision time. Existing benchmarks measure recall over static dialogue, collapse memory into a single end-of-task accuracy, and reduce visual observations to captions, leaving us unable to localize failures to writing, maintenance, retrieval, or use. The rise of agent harnesses that author their own memory sharpens this gap, since we have no principled way to compare hand-designed pipelines with self-managing alternatives. To close these gaps, we formulate multimodal agent memory as an Action-World Interaction Loop with an observable four-stage lifecycle, and instantiate it in WorldMemArena: 400 multi-session multimodal tasks spanning Lifelong Evolution (evolving personal and task states) and Agentic Execution (memory from real observations, actions, and feedback), annotated with gold memory points, updates, distractors, and evidence chains for stage-level diagnosis. This enables the first head-to-head comparison of long-context, manually designed (RAG and external memory systems), and harness-based memory agents. Results show that: (1) better memory writing and storage do not guarantee better performance; (2) multimodal memory still struggles to fully use visual evidence; (3) systems are unstable across domains and degrade on realistic agentic trajectories; and (4) harness memory is more flexible but remains costly and less reliable. 

---
# Rubric-Guided Process Reward for Stepwise Model Routing 

**Authors**: Shenghao Ye, Yu Guo, Zhengheng Li, Shuangwu Chen, Jian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29310)  

**Abstract**: Stepwise model routing improves the efficiency of Large Reasoning Models (LRMs) by assigning each reasoning step to a suitable model. Recent methods formulate routing as a sequential decision process and train the router with reinforcement learning. However, although they model routing as a process, they still supervise the router with outcome rewards. Such rewards only reflect final answer correctness and fail to evaluate intermediate routing decisions, which can weaken performance and generalization. To address this gap, we propose RoRo, a rubric-guided process reward framework for stepwise model routing. RoRo first collects diverse routing trajectories and constructs preference pairs based on outcome, cost, and process quality. It then trains a Rubricor to generate a query-specific evaluation rubric and a Judge to score routing trajectories under this rubric through alternating optimization. The resulting process rewards are combined with outcome rewards to optimize the routing policy via GRPO. Experiments on five reasoning benchmarks under both same-family and cross-family settings show that RoRo consistently outperforms strong baselines and achieves better accuracy and cost trade-offs. 

---
# DenseSteer: Steering Small Language Models towards Dense Math Reasoning 

**Authors**: Yang Ouyang, Shuhang Lin, Jung-Eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.29247)  

**Abstract**: Large language models (LLMs) demonstrate strong chain-of-thought (CoT) reasoning abilities, while smaller models (<= 3B parameters) significantly underperform on multi-step reasoning tasks. Based on empirical analyses of the Qwen-2.5 model family on math reasoning benchmarks, we find that more proficient reasoning is associated with fewer reasoning steps but higher information density per step, a property we term Dense Reasoning. Motivated by this observation, we propose DenseSteer, a training-free inference-time steering framework that enhances small-model reasoning by modulating internal representations toward dense reasoning patterns. Experiments show that our method yields consistent accuracy improvements without increasing token-level Negative Log-Likelihood, highlighting dense reasoning as an effective structural approach to mathematical problem solving. 

---
# Implicit Identity Technologies for LLMs: Fingerprinting and Watermarking across Datasets, Models, and Generated Content 

**Authors**: Bing Liu, Shunping Wang, Yufan Zhu, Xinyi Yu, Jing Huang, Linkang Du, Hongbin Pei, Wei Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.29245)  

**Abstract**: This paper presents a survey and taxonomy of LLM fingerprinting and watermarking for identity, ownership verification, provenance, and generated-content attribution. Large language models (LLMs) require substantial investments in data, computation, and expertise, and are increasingly deployed in high-stakes settings, making it critical to protect LLM-related assets and trace their origins. Existing work has rapidly expanded across dataset provenance, model ownership, and generated-content detection, but the field remains fragmented: fingerprinting and watermarking are often used inconsistently, and methods are typically studied within isolated asset-specific settings. To address this gap, we introduce implicit identity as a unifying abstraction for verifiable but not directly observable identity signals in LLM systems. We distinguish fingerprinting as non-intrusive identity derived from intrinsic characteristics, and watermarking as intrusive identity deliberately embedded into data, models, or generated content. We then propose a lifecycle-based taxonomy that organises techniques across datasets, models, and generated content, and further separates them by verification semantics: similarity-based attribution and keyed verification. Finally, we establish an evaluation framework centred on identifiability, robustness, and deployability, summarising representative metrics under realistic access and transformation regimes. By unifying terminology, lifecycle stages, and evaluation objectives, this survey provides a structured foundation for studying LLM identity technologies and for developing more reliable mechanisms for asset protection and provenance. 

---
# Surfacing Isolated Learners with Outcome-Independent Mediation of Feedback between Teachers and Students Using AI 

**Authors**: Junsoo Park, Youssef Medhat, Htet Phyo Wai, Ploy Thajchayapong, Ashok K. Goel  

**Link**: [PDF](https://arxiv.org/pdf/2605.29240)  

**Abstract**: AI-augmented classrooms generate rich teacher and student feedback before graded outcomes become available, yet these signals can be difficult to translate into timely instructional decisions. We propose an interpretable decision layer: a transparent mechanism that ranks course topics requiring attention without using grades or post-hoc outcome labels. The approach combines three signals: student learning difficulty prevalence, disagreement between learner self-reports and observed difficulties, and unresolved teacher concerns. The output is a ranked set of topic priorities with per-topic decision records explaining each ranking. In one graduate CS course offering ($n=5$ instructor interviews; $n=279$ survey responses), prioritized topics aligned with instructor concerns (top-5 overlap 3/5; Spearman $\rho=0.80$) and student-reported topic difficulty ($\rho=0.46$, $p=.048$). Multi-signal integration also surfaced learners not identified through individual signal sources alone (AUC $=0.96$ vs. $0.91$ for gap prevalence alone). Reflective thinking, help-seeking, and self-efficacy provided additional evidence that student behavioral signals align with learning-related constructs. While preliminary, these findings suggest that transparent coordination mechanisms may help support human-AI co-agency when feedback is incomplete. 

---
# GTA: Generating Long-Horizon Tasks for Web Agents at Scale 

**Authors**: Tenghao Huang, Kung-Hsiang Huang, Prafulla Kumar Choubey, Yilun Zhou, Muhao Chen, Jonathan May, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.29218)  

**Abstract**: Web agents, which couple language models with browsing and tool-use capabilities, show promise as open web assistants. Yet progress is increasingly limited by the lack of scalable, process-level supervision. Existing benchmarks are largely manually constructed, providing only coarse start-goal annotations without intermediate trajectories, while recent automatic generation efforts remain expensive, biased, and shallow. These limitations prevent reliable training and evaluation of agents that must generalize to realistic, multi-hop, cross-page tasks. We introduce a scalable framework, GTA, that integrates crawling, retrieval-based seeding, in-context generation, and automated quality control to produce realistic tasks paired with executable trajectories. This design decouples crawling from generation for greater efficiency, grounds tasks in the site graph to enforce compositionality, and ensures dense supervision through deterministic replays and systematic validation. We instantiate the pipeline on over 50 websites covering e-commerce, government, forums, and news, with multilingual and multi-hop coverage. The resulting benchmark reveals a significant human-agent performance gap and enables detailed diagnostics. Our contributions are three-fold: (i) formalizing multi-hop web-agent task generation, (ii) proposing an efficient and validated pipeline for automatic data creation, and (iii) releasing a dynamic benchmark with reproducible evaluation. 

---
# ReasonOps: Operator Segmentation for LLM Reasoning Traces 

**Authors**: Daniel Lee, Owen Queen, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2605.29192)  

**Abstract**: Chain-of-thought traces from large reasoning models can span tens of thousands of tokens, yet we lack a vocabulary for describing their internal structure. Previous methods developed to analyze chain-of-thought traces are either too rigid or not expressive enough, failing to capture features across domains and models. To remedy this, we develop ReasonOps, an unsupervised, expressive method for annotating chain-of-thought traces, providing succinct universal operators. Using ReasonOps, we analyze 44,662 traces from 12 thinking LLMs spanning 6 families across 8 reasoning benchmarks and discover that they share a common compositional structure: 7 recurring reasoning operators -- discourse-level moves such as backtracking, inferring, and hypothesizing -- that emerge from unsupervised clustering of sentence-initial 3-token pivots. These operators appear across every model family and benchmark domain, confirmed by three independent LLM judges who classify held-out samples at 70 -76% accuracy. We analyze the structure of operators on easy vs. hard problems, revealing that reflective operators are more helpful on hard problems and harm performance on easy problems. Operator sequences are highly model-identifying: a classifier trained on operator distributions alone recovers the source model with macro-AUC, revealing that each model family has a distinctive reasoning fingerprint. Structural operator features predict within-problem answer correctness well above baselines. Classifiers built on these operators reach WP-AUC and on AIME specifically. ReasonOps further enables early quality estimation well before the trace completes: we predict at WP-AUC for only 50% of the trace. The ReasonOps pipeline is unsupervised and annotation-free, enabling deep insights into LLM reasoning traces as well as strong downstream results on model identification and correctness prediction. 

---
# When RL Suppresses Its Own Vocabulary: Recovering Reasoning Diversity in Puzzle-to-Math Transfer 

**Authors**: Mayug Maniparambil, Arjun Karuvally, Terrence Sejnowski, Fergal Reid  

**Link**: [PDF](https://arxiv.org/pdf/2605.29190)  

**Abstract**: Reinforcement learning using verifiable rewards (RLVR) improves LLM reasoning, but the conditions under which it transfers across domains -- and why it does so -- remain under-explored. We study cross-domain transfer in a 7B model whose SFT and RL post-training stages use only constraint-satisfaction puzzles, with no mathematics problems in the post-training data. To analyze how transfer emerges, we introduce a reasoning primitive-level framework that combines a 9-class span classifier with motif extraction, allowing us to segment chain-of-thought traces into primitive motifs and track their evolution across training stages and domains. We find that puzzle SFT induces a reasoning-primitive vocabulary, yielding a $+7$pp \texttt{pass@32} gain on OlymMATH-Hard. Vanilla GSPO then composes these primitives into longer compute-verify chains, adding a further $+6$pp. However, this RL stage also suppresses exploratory primitives such as \textit{hypothesize} and \textit{backtrack}. To address this, we introduce a novelty bonus that rewards diverse correct rollouts, using perplexity under the reference model as a signal. This restores recovery primitives during RL and adds a further $+7$pp \texttt{pass@32} relative to vanilla GSPO. Finally, the end-to-end recipe raises the hard-math capability ceiling from $16.0\%$ at the OLMo3-7B-Instruct-SFT base to $36.0\%$, without adding any mathematics problems during the SFT or RL stages. 

---
# Parallax: Parameterized Local Linear Attention for Language Modeling 

**Authors**: Yifei Zuo, Dhruv Pai, Zhichen Zeng, Alec Dewulf, Shuming Hu, Zhaoran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29157)  

**Abstract**: Large Language Models (LLMs) have become the central paradigm in artificial intelligence, yet the core computational primitive of attention has remained structurally unchanged. Local Linear Attention (LLA) is an attention mechanism derived from nonparametric statistics in the test-time regression framework. In contrast to prior research on efficient attention variants, LLA upgrades the local constant estimate in softmax attention to a local linear estimate, yielding provably superior bias-variance tradeoffs for associative memory. However, LLA has not been scaled in LLM pretraining due to computational and numerical stability concerns. We introduce Parallax, a parameterized Local Linear Attention that is scalable for LLMs. Parallax eliminates the numerical solver in LLA and learns an extra query-like projector that probes the KV covariance. We place Parallax within a family of attention mechanisms connected by the bandwidth, the probe construction and the affine structure. We propose a hardware-aware algorithm that increases the arithmetic intensity over FlashAttention, shifting attention into a more compute bound regime. Our prototype decode kernel matches or outperforms FlashAttention 2/3 across diverse batch sizes and context lengths. We pretrain Parallax at 0.6B and 1.7B scales and find consistent perplexity improvements throughout pretraining with gains that transfer to downstream benchmarks. The advantage persists under both parameter-matched and compute-matched controls, demonstrating a Pareto improvement. We perform careful pretraining ablations and identify a novel phenomenon whereby Muon unlocks the capacity of Parallax. To our knowledge, this is the first empirical demonstration of strong architecture-optimizer codesign for attention mechanisms in the architecture research literature. 

---
# RUBRIC-ARROW: Alternating Pointwise Rubric Reward Modeling for LLM Post-training in Non-verifiable Domains 

**Authors**: Haoxiang Jiang, Zihan Dong, Tianci Liu, Wanying Wang, Ran Xu, Tony Yu, Linjun Zhang, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29156)  

**Abstract**: Pointwise reward modeling offers critical signals for LLM post-training, yet struggles with absolute scoring in subjective, non-verifiable settings. Rubric-based methods address this by decomposing evaluation into explicit criteria, but existing approaches typically depend on frontier LLMs and suffer from ties caused by hard Boolean aggregation. We present RUBRIC-ARROW, an alternating framework that jointly trains a rubric generator and a rubric-conditioned judge, with its RL stage using only pairwise preference data. Our method couples a probability-based scoring rule that reduces ties with phase-specific preference-based rewards and an alternating GRPO scheme that together train the pointwise evaluator. Extensive experiments show that RUBRIC-ARROW achieves competitive reward-modeling accuracy and yields consistent gains for downstream policy post-training. 

---
# The Confidence Shortcut: A Reasoning Failure Mode of Masked Diffusion Models 

**Authors**: Dueun Kim, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2605.29123)  

**Abstract**: Masked diffusion language models (MDMs) uniquely support any-order generation, with confidence-based decoding currently serving as the de facto standard inference policy. To optimize for this, recent training schemes attempt to align training mask patterns directly with those observed during generation. However, we argue that confidence-based decoding is inherently misaligned with the logical-flow trajectories required for complex reasoning, and that confidence-aligned training actively entrenches this misalignment. We make this concrete using multi-digit addition, where the decoding strategy prematurely predicts locally easy digits before resolving their long-range dependencies, producing high-confidence errors on challenging inputs. While traditional random masking keeps the failure rate low on this challenging tail, confidence-aligned training amplifies the error rate by an order of magnitude. Across five distinct reasoning tasks, this same pattern emerges with task-dependent severity: confidence-based decoding induces failures on highly complex inputs, and confidence-aligned training exacerbates them. In contrast, random masking -- despite its perceived inefficiency -- robustly preserves the reasoning-trajectory conditionals essential for solving the challenging tail. 

---
# Robust and Efficient Guardrails with Latent Reasoning 

**Authors**: Siddharth Sai, Xiaofei Wen, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.29068)  

**Abstract**: Maintaining the safety of large language models (LLMs) is crucial as they are increasingly deployed in real-world applications. Existing safety guardrails typically rely on single-pass classification or, more recently, distilled reasoning. Reasoning-based guardrails significantly outperform classification-only baselines, but they incur substantial query latency and token overhead that make them impractical for highthroughput deployment. To address this challenge, we propose COLAGUARD, a guardrail model that transfers multi-step safety reasoning into a continuous latent space through a stage-wise training curriculum, enabling direct hidden-state propagation at inference. Evaluated on ten prompt- and response-moderation settings spanning eight safety benchmarks, COLAGUARD improves macro-F1 by 8.24 points over Llama Guard 3 and matches our explicit reasoning baseline, GuardReasoner, in macroF1 while delivering a 12.9X speedup and 22.4X reduction in token usage. Our results suggest that latent reasoning offers a practical alternative to explicit rationale generation for deployable guardrails, jointly improving safety robustness and inference efficiency rather than treating them as competing objectives. 

---
# Converted, Not Equivalent: Benchmarking Codebase Conversion via Observational Equivalence 

**Authors**: Linxin Song, Jiefeng Chen, Yue Huang, Bhavana Dalvi Mishra, Chi Wang, Jieyu Zhao, Jinsung Yoon, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2605.29054)  

**Abstract**: Coding agents increasingly act as codebase-scale collaborators that can assist with codebase conversion, but this progress has exposed a critical weakness: agents often over-trust their own local validation routines and declare success on artifacts that satisfy surface checks while violating the semantic contracts users actually care about. This problem is especially acute in codebase conversion, where prior evaluation is largely outcome-driven and therefore unstable: two implementations can match on a shallow outcome, such as a single forward loss, while diverging in gradients, optimizer behavior, or short-horizon training dynamics. We introduce T2J-Bench, a benchmark for codebase conversion that reformulates conversion as transfer under a fixed equivalence contract. A fixed verifier then compares source and converted codebases through three ordered stages: Spec (interface admissibility), Numeric (forward outputs, losses, gradients, and objective-specific tensors), and Behavioral (short training dynamics under fixed seeds). Across 355 blind conversion attempts, the best system reaches only 26.7--28.9% overall pass rate despite Spec pass rates up to 91.1%; a 4.7x token-budget spread yields only a 2.2x pass-rate spread; and all systems overestimate success by 66.6--97.8 points relative to the fixed evaluator. This suggests that failures stem more from contract-misaligned self-validation than from limited budget or backbone strength. 

---
# Mind Your Tone: Does Tone Alter LLM Performance? 

**Authors**: Om Dobariya, Akhil Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.29027)  

**Abstract**: The use of Large Language Models (LLMs) is proliferating, yet their performance is observed to vary based on prompting styles and tones. In this study, we investigate both whether and how tonal variations in prompts lead to disparate LLM accuracy for objective multiple-choice questions. We use two datasets: a 50-base question dataset with five tone variants and a 570-base question MMLU subset spanning 57 subjects with seven tone variants. Experiments were conducted to evaluate the performance of four cost-efficient, popular LLMs: ChatGPT-4o, ChatGPT-5-nano, Gemini 2.5 Flash, and Gemini 2.5 Flash Lite. Across models, tonal effects are systematic but highly model-dependent. Some models show small, yet statistically significant, shifts, while others exhibit large accuracy swings across tones. Further, we identify subject-level differences in tone sensitivity and present a routing framework to explain how tones may attune internal reasoning modes. Our findings caution users against assuming tone-robust reliability in LLM deployments. 

---
# Adopt $\neq$ Adapt: Longitudinal Analyses of LLM Conversations in the Wild 

**Authors**: Rebecca M. M. Hicke, Kiran Tomlinson  

**Link**: [PDF](https://arxiv.org/pdf/2605.29018)  

**Abstract**: Although a growing body of research has begun to describe user--LLM interactions, the picture it paints is largely static; little is known about how individual users change their behavior over time. To address this gap, we analyze the conversational trajectories of $\sim$12,000 randomly sampled Microsoft Bing Copilot users and compare these with data from WildChat-4.8M. While the Copilot data contains significant population-level trends, we find that trends in individual user trajectories are much weaker; user habits prove to be overwhelmingly sticky. We also find stark differences between users of different activity levels: more active users have more successful conversations and use the LLM for more complex and professionally oriented tasks. Some user trends also appear in WildChat-4.8M, but we find evidence that this dataset is significantly skewed towards highly proficient "power" users. Ultimately, our results suggest that existing user behavior is difficult to change and demonstrate the extent of user heterogeneity. Our comparison between datasets highlights that WildChat does not represent typical user-AI interactions, an important caveat for downstream uses of the data. 

---
# Measuring Real-World Prompt Injection Attacks in LLM-based Resume Screening 

**Authors**: Mohan Zhang, Yuqi Jia, Zhen Tan, Steven Jiang, Neil Zhenqiang Gong, Tianlong Chen, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.28999)  

**Abstract**: LLMs are vulnerable to prompt injection attacks. However, this vulnerability has been primarily demonstrated conceptually in academic studies or through a few anecdotal case studies. Its prevalence and impact in real-world LLM-based applications are largely unexplored. In this work, we present the first systematic study of prompt-injection attacks in a widely used application: LLM-based resume screening. Our analysis is based on approximately 200K real-world resumes collected over multiple years by hireEZ. We first design tailored methods to detect prompt injection in resumes. Manual validation on a small-scale dataset demonstrates that our detectors achieve high precision and outperform state-of-the-art general-purpose detectors. We then apply our detector to the full resume dataset and conduct a comprehensive measurement study of real-world prompt injection attacks. Our analysis reveals several intriguing findings: approximately 1% of resumes contain hidden prompt injections; the prevalence of such injected resumes has increased noticeably over the past one to two years; and more than 90% of injected prompts do not use explicit instructions. These results provide the first evidence of large-scale prompt injection in real-world LLM-based applications and lay the groundwork for future studies to understand and mitigate such attacks. 

---
# CosmicFish-HRM: Adaptive Reasoning via Hierarchical Recurrent Mechanisms in Compact Language Models 

**Authors**: Venkat Akhil Lakkapragada  

**Link**: [PDF](https://arxiv.org/pdf/2605.28919)  

**Abstract**: Large language models have achieved strong reasoning capabilities, though often at the cost of massive parameter counts and expensive inference. In this work, we explore a different direction: adaptive reasoning depth in compact language models. We present CosmicFish-HRM, a compact language model built around a Hierarchical Reasoning Module (HRM) that dynamically allocates computational effort during inference. Instead of applying fixed computation to every input, the model iterates through high-level and low-level reasoning cycles and learns when to halt based on input complexity. CosmicFish-HRM combines this adaptive reasoning core with modern transformer components including Grouped Query Attention, RoPE, and SwiGLU activations. While the additional reasoning infrastructure introduces overhead at small scale, we hypothesize that this tradeoff becomes increasingly favorable as model size grows and the relative cost of the HRM core diminishes. Our results show that the model learns non-uniform reasoning behavior, allocating different numbers of reasoning steps across tasks and inputs. These findings suggest that adaptive reasoning depth may offer a promising alternative to relying solely on parameter scale for reasoning capability. 

---
# The Cognitive Categorical Transformer: Category-Theoretic Inductive Biases for Language Modeling 

**Authors**: Al Kari  

**Link**: [PDF](https://arxiv.org/pdf/2605.28864)  

**Abstract**: The Cognitive Categorical Transformer (CCT) is a 306M-parameter architecture that augments a pretrained GPT-2 Small backbone with cognitively grounded components derived from category theory and several inspirations from cognitive science. Under a matched-step protocol (215,000 optimizer steps, matched data, matched optimizer and schedule) on WikiText-103, CCT reaches 21.27 validation perplexity, compared with 24.19 for an identically fine-tuned GPT-2 Small baseline. The architecture therefore contributes a 2.92 PPL (12% relative) reduction beyond what in-domain fine-tuning alone provides. A retrain-from-scratch ablation that holds GT-Full simplicial message passing bypassed across the entire seven-phase activation schedule reaches 23.72 PPL, localizing 84% of the architectural improvement (2.45 of 2.92 PPL) to GT-Full. We present the first ablation-validated evidence that simplicial message passing improves language-model perplexity at the 306M-parameter scale on WikiText-103. Published GPT-2 Large reaches 22.05 zero-shot PPL on WikiText-103 with 6.2x more parameters than GPT-2 Small; this paper treats that number as an external published reference, not as the architectural benchmark. Three negative results on consistency-style categorical priors (sheaf smoothing, adjunction round-trip, curvature regularization) and the joint structural-prior result for GT-Full and PrecisionWeightedPP together support an empirical pattern termed the *structure/consistency distinction*, in which categorical priors that add new topology improve language modeling and those that enforce a consistency identity do not. 

---
# Mechanistic origins of catastrophic forgetting: why RL preserves circuits better than SFT? 

**Authors**: Jeanmely Rojas Nunez, Viraj Sawant, Nathan Allen, Nomgondalai Amgalanbaatar, Yannis Zongo, Vasu Sharma, Maheep Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2605.28860)  

**Abstract**: Fine-tuning large language models (LLMs) frequently induces catastrophic forgetting of prior capabilities. Recent work has shown that reinforcement learning (RL) retains prior capabilities more effectively than supervised fine-tuning (SFT), attributing this to policy-gradient updates remaining closer to the base policy \cite{shenfeld2025rl}. We extend this behavioral account to the mechanistic level and ask whether RL's advantage is mirrored by stronger preservation of internal computational circuits. We introduce differential circuit vulnerability, a head-level measure of how much a circuit degrades under fine-tuning, and use it to compare RL and SFT on Qwen2.5-3B-Instruct adapted to scientific question-answering. We find a clear mechanistic trade-off: SFT adapts more rapidly to the target task but produces substantially greater circuit disruption and forgetting of prior capabilities, whereas RL preserves a larger fraction of the base circuit at the cost of slower task adaptation. These findings suggest that circuit preservation may help explain why RL is more robust to catastrophic forgetting. We released our code here: this https URL. 

---
