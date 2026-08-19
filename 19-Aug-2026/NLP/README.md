# Multi-Agent AI System for Radiology Report Structuring and Quality Assurance with Independent Radiologist Evaluation 

**Authors**: Iryna Hartsock, Cesar Lam, Christopher Otteni, Aliya Qayyum, Robert Gatenby, Cyrillo Araujo, Ghulam Rasool  

**Link**: [PDF](https://arxiv.org/pdf/2608.18072)  

**Abstract**: Purpose: To develop and evaluate a locally deployed multi-agent AI system for radiology report structuring and quality assurance. Materials and Methods: This retrospective study included 638 radiology reports from CT examinations of the chest, abdomen, and pelvis dictated by 15 board-certified radiologists in 2023 and 2024. A multi-agent AI pipeline was developed to perform report structuring and quality assurance (QA). The system structured the report into standardized anatomical sections at the sentence level using regex rules and local large language models. It also detected mismatches between the Findings and Impression sections, or within sections; gender-anatomy conflicts; and undocumented communication of critical findings. Two board-certified radiologists independently evaluated a 45-report subset. Results: The multi-agent system structured the Findings sections of all reports (22,270 sentences) into a predefined anatomical format while retaining the original report content. The system flagged 90 (14.1%) reports, most commonly for section mismatches (80 reports, 12.5%). In the radiologist evaluation, both reviewers agreed that 31 (69%) were correctly restructured, 2 reports (4%) were incorrectly restructured, and disagreed on the remaining 12 reports (27%). Both reviewers agreed that no clinically important information was omitted and no fabricated content was introduced. Overall QA performance was rated as "excellent" or "good" in 84% of the evaluated reports, with the remaining reports rated as "fair". Conclusion: A locally deployed multi-agent AI system combined radiology report structuring and quality assurance within a single workflow. The system demonstrated favorable performance in radiologist evaluation. Such systems may support standardization of reporting and quality assurance in radiology practice. 

---
# TokEval: A Tokenizer Evaluation Suite 

**Authors**: Clara Meister  

**Link**: [PDF](https://arxiv.org/pdf/2608.18062)  

**Abstract**: Language model tokenizers are typically selected with minimal evaluation, despite the fact that their design choices directly impact model capabilities. This can be partly attributed to a limited understanding of which tokenizer properties affect which aspects of downstream performance. We introduce TokEval, a framework of tokenizer evaluation metrics that goes beyond standard measures like fertility and compression rate to capture linguistically and structurally meaningful properties, e.g., UTF-8 character boundary integrity and digit place-value boundary alignment for mathematics. To validate whether these metrics are predictive of downstream model performance, we conduct controlled language model pretraining experiments, varying solely the tokenizers' training data mixture, pretokenization strategy, and training algorithm. We evaluate the resulting models on bits-per-byte (a tokenizer-agnostic version of perplexity) and several benchmarks, spanning linguistic understanding, mathematical reasoning, and code generation. Our experiments suggest that different intrinsic properties have different impacts on model abilities: information-theoretic metrics predict language modeling abilities (Spearman rho up to 0.80), while structure-sensitive metrics, such as those measuring digit and line-break handling, correlate with task accuracy. We hope TokEval enables more principled tokenizer evaluation, replacing pretraining sweeps with intrinsic measurement wherever the two agree. 

---
# Language Has Two Parameters: Narrative-Induced Semantic Plasticity and Phase-Sensitive Interpretation 

**Authors**: Hollis Robbins  

**Link**: [PDF](https://arxiv.org/pdf/2608.18041)  

**Abstract**: Language has two parameters. Count how often words occur together and you estimate amplitude, the strength of association. Word embeddings and attention weights refine that count, which sums every writer in the corpus together. This paper claims a second parameter, phase, which signed weights learned from a corpus do not supply. Phase exists only between meanings: it determines how coactivated meanings combine, and it can reverse what a meaning contributes while that meaning stays fully present. A speaker can set phase in the signal through linguistic form; encounters install phase relations and history distributes them. Population averaging deletes history-indexed phase: agent-deindexed corpora identify the population marginal state and determine no individual or dyadic state, at any scale. The standard transformer has no explicit representation for phase in frozen inference, and the interpretability program measuring progress by monosemanticity is optimizing against it: the coexistence it treats as a defect is the condition of allusion, irony, and quotation. Six predictions test whether a suppressed meaning stays active, whether encounter order changes what a phrase does, whether marking the signal changes how a shared phrase is taken, and whether a model given a history is changed by it or only informed about it. The claim defended is the weak version: interpretation requires a second relational parameter, signed, persistent, and indexed to individuals and dyads. Quantum probability is one notation for the parameter; nothing in the formalism claims quantum processes in the brain. The strong version, that the quantum calculus constrains these phenomena as signed classical models do not, rests on an encounter-order constraint not yet derived. The architecture the theory calls for is a language model with agent-indexed, phase-bearing semantic states. 

---
# Chain-of-Experience for Continual LLM Improvement 

**Authors**: Haoqin Tu, Yunhao Fang, Yizhong Wang, Cihang Xie, Shen Yan  

**Link**: [PDF](https://arxiv.org/pdf/2608.18027)  

**Abstract**: Humans continuously learn from experience, whereas conventional large language model (LLM) evaluations ignore the models' ability to improve through inference-time interaction. In this paper, we study how LLMs learn from iterative experience at test time, a setting we refer to as Chain-of-Experience (CoE), where models accumulate experiential traces through iterative interactions with self or environmental feedback to form a continual improvement loop beyond zero-shot inference. We instantiate CoE with diverse feedback mechanisms, including model self-feedback and environmental signals such as correctness or public coding test pass rates, and evaluate across math, coding, and knowledge domains using 8 LLMs, including GPT-5, Gemini-2.5 Pro, Claude-4.5 Sonnet. Our study shows that leveraging iterative experience consistently outperforms feedback-free baselines, achieving substantial gains with self feedback alone, alongside a 5.6% overall improvement and 19% lower API cost across tasks and models. We further show that combining complementary feedback channels (e.g., model and correctness signals) yields additional gains, and that CoE delivers higher accuracy per token than existing test-time strategies. We observe a positive correlation between LLM base ability and improvement capacity, and show that models remain robust under weak or spurious feedback, with different feedback contributing to distinct improvement aspects and most gains emerging early in the iterations. 

---
# The IOL-AI Challenge: An Open Challenge towards Advancing Linguistic Reasoning 

**Authors**: Eduardo Sánchez, Rita Berrada, Dan-Mircea Mirea, Sara Rajaee, Alexander Piperski, Ana Meta Dolinar, Boris Iomdin, Andrey Nikulin, Mariya Shmatova, Marzieh Fadaee, Julia Kreutzer  

**Link**: [PDF](https://arxiv.org/pdf/2608.18011)  

**Abstract**: Reasoning in LLMs is overwhelmingly studied in domains that provide a model with rules: mathematics and code. Linguistic puzzles invert this: the solver must first discover the system before reasoning within it. We present the IOL-AI Challenge, an open-science competition run on the unseen problems of the International Linguistics Olympiad (IOL) 2026 Individual Contest, evaluated both automatically and, for the first time, by members of the official IOL Jury under the same rubrics applied to human contestants. The challenge drew 731 submissions from 46 teams under a strict compute budget (one T4, 30 mins). We additionally benchmark 15 unconstrained frontier and open models, with Claude Opus 4.8 earning a jury score equivalent to a gold medal, while both resource-constrained systems we submitted for jury grading scored in the range of the bottom 5% of contestants. Capability was not determined by scale: 14B submissions outperform models twice their size, and gains come from decoding and output-handling rather than model capacity. We also found that automatic metrics rank systems exactly as the jury does, but compress the scale, upscoring weak systems by ~13 points and understating strong ones. Our analysis shows that while frontier models might have prior knowledge about some of the problem languages, it does not significantly help them solve the linguistic reasoning tasks, leaving linguistic reasoning as a strong benchmarking proxy for generalizable reasoning skills. 

---
# Judge, Retrieve, or Abstain: Uncertainty-Guarded LLM Judging with Provable Risk Guarantees 

**Authors**: Sher Badshah, Ali Emami, Hassan Sajjad  

**Link**: [PDF](https://arxiv.org/pdf/2608.17994)  

**Abstract**: Using LLMs as judges has become standard practice for evaluating model outputs at scale. This is particularly common for subjective, open-ended tasks such as assessing helpfulness or alignment, where no single reference answer exists. However, objective tasks introduce a distinct reliability challenge for reference-free LLM judging. In the absence of a reference answer, the judge evaluates factual correctness either through its parametric knowledge or through tool augmentation. Although the former enables efficient evaluation, the judge may hallucinate or lack sufficient evidence for its verdict. Conversely, tool augmentation can provide additional evidence but introduces extra computational cost and requires an appropriate mechanism to determine when and how that evidence should be used reliably. More importantly, neither approach alone provides formal control over the risk of accepted verdicts or guarantees their reliability at a specified level. We propose a risk-controlled framework that calibrates uncertainty thresholds on a held-out set so that the false discovery rate among accepted verdicts remains below a user-specified level~$\alpha$ with high probability, using finite-sample Clopper--Pearson intervals. When the parametric mode is not sufficiently confident, the instance is routed to a retrieval-augmented mode, where the judge gathers web evidence and re-evaluates the instance under a second calibrated threshold. The finite-sample guarantee carries over to this two-threshold routing without additional assumptions. Across open-domain QA benchmarks and judges of varying scales, the framework maintains the target error rate while achieving substantially higher coverage than single-mode baselines. 

---
# When Writing Style Drifts: Benchmarking Authorship Verification under Distribution Shifts in Genre, Time and the AI-Era 

**Authors**: Lotta Kiefer, Brisca Balthes, Christoph Leiter, Yamen Ajjour, Elena Schmidt, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2608.17979)  

**Abstract**: Authorship verification (AV) assumes that an author's writing style remains sufficiently stable to distinguish it from that of other writers. In practice, however, this assumption is challenged by distribution shifts caused by changes in genre, time, and AI-assisted writing. Existing AV benchmarks typically study these factors in isolation and focus predominantly on English, limiting our understanding of model robustness under realistic conditions. We introduce AVShift, the first German benchmark for systematically evaluating AV under multiple distribution shifts. AVShift comprises over 150,000 text pairs spanning three genres and 21 years, enabling controlled evaluation of cross-genre, temporal, and AI-era shifts within a unified framework. We benchmark representative feature-based, embedding-based, and LLM-based approaches. Our experiments show that fine-tuned LLMs generalize best across genres and benefit substantially from stylistically diverse training data. We further demonstrate that temporal drift is one of the strongest factors affecting AV, with performance degrading significantly as the time gap between documents increases. In contrast, we find no evidence of a measurable AI-era distribution shift within AVShift. Finally, our feature analysis reveals stylistic features that remain stable across genres, while their relative importance varies depending on the specific genre transition. We release AVShift and our code for future research. 

---
# Do Large Language Models Play Six Degrees of Separation? Measuring Topological Compression in Long-Context Manifolds 

**Authors**: Md. Faiyaz Abdullah Sayeedi  

**Link**: [PDF](https://arxiv.org/pdf/2608.17950)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable multi-hop reasoning capabilities over long contexts, yet the internal mechanisms enabling these distant cognitive leaps remain poorly understood. Traditional attention-based interpretability often fails to capture true semantic proximity due to routing artifacts like attention sinks. In this paper, we bypass attention weights to directly analyze the dynamic geometry of the hidden state manifold, proving that deep LLM latent spaces natively organize into Small-World networks. By sparsifying the continuous similarity matrices of long-context representations into unweighted graphs, we trace the connectivity between highly disjoint semantic anchors across two distinct architectures. Our findings reveal a sharp topological phase transition: while early syntactic layers remain entirely fractured, deep reasoning layers abruptly compress massive conceptual distances into highly navigable pathways strictly bounded by the "Six Degrees of Separation" limit (=< 6 semantic hops). Furthermore, we demonstrate the practical efficacy of this framework by applying it to zero-shot hallucination detection within Retrieval-Augmented Generation (RAG) using the RAGognize dataset. We show that factually grounded generations maintain structural integrity with their source context (approximately 3 hops), whereas hallucinations induce severe topological collapse. Ultimately, this work mathematically formalizes how transformers execute abstract reasoning and provides a novel, strictly geometric signature for evaluating factual reliability. 

---
# Grading Needs a Rubric, Not Intelligence 

**Authors**: Jhen-Ke Lin  

**Link**: [PDF](https://arxiv.org/pdf/2608.17938)  

**Abstract**: Small language models can grade open-ended examination answers as reliably as substantially more expensive models when they grade against an explicit rubric. We test this claim as the design principle behind any-to-bench: a frontier model reads source documents once, at ingestion, to extract each question and its rubric; lower-cost models then perform all repeated grading work. We evaluate six cost-efficient model configurations from two model families at three reasoning-effort levels. Each configuration answers 24 open-ended examination questions, and each also grades every answer sheet three times, yielding 3,456 per-question grades. Scores depend overwhelmingly on the answer being graded: answer identity explains 95.6% of score variance, whereas judge identity explains only 0.2%. Raising a writer's reasoning effort moves earned scores by as much as 0.143 of full marks, while raising a judge's reasoning effort moves assigned scores by at most 0.006. Six frontier-tier judges, added as a check, reproduce these scores and are no more reliable as a panel. Two ablations then decompose the rubric on the same questions and answers. Removing its criteria and levels while keeping the official answer changes nothing measurable. Removing the official answer as well collapses reliability (ICC 0.888 to 0.628), inflates scores, and makes judge reasoning effort matter again. The rubric is what decouples grading from judge intelligence, and within the rubric the official answer does nearly all the work. We find no evidence of length preference or same-family preference under rubric-anchored grading. 

---
# SpeechSense: A Paralinguistic-Focused Dataset for Fine-Grained Speech Sentiment Analysis 

**Authors**: Shicheng Ma, Wenqian Cui, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2608.17931)  

**Abstract**: Recent advances in AI have revolutionized speech processing, yet effective speech understanding requires discerning not just what is said, but how it is said. Speech Sentiment Analysis plays a critical role in decoding these paralinguistic cues for diverse real-world applications such as recruitment and customer service. However, existing Speech Sentiment Analysis research faces two primary limitations. First, dominant approaches rely on text-centric pipelines that cascade Automatic Speech Recognition with text analysis. This process inevitably discards essential acoustic features like prosody and tone, failing to capture attitudinal meanings in acoustically ambiguous utterances. Second, current benchmarks suffer from a mismatch in label granularity, prioritizing basic emotions (e.g., happy, sad) over the nuanced interpersonal stances (e.g., confident, impatient) necessary for social sensitivity. To address these limitations, we propose a novel dataset, SpeechSense, for fine-grained speech sentiment analysis. Specifically, we define a specialized 8-class taxonomy of interpersonal stances detectable primarily through prosodic cues beyond lexical content alone. We then construct a curated dataset based on this taxonomy, built from high-fidelity speech synthesis and rigorous human validation. Comprehensive experiments across multi-modal LLMs, text-only LLMs, and speech encoders demonstrate that models with acoustic access consistently outperform text-only baselines. These results empirically validate the primacy of acoustic cues in detecting subtle speaker attitudes, highlighting the necessity of SpeechSense. Dataset and supplementary materials are available at this https URL. 

---
# CABLE: Extending the Reach of Memory Retrieval via Complementary Antecedent-Based Linking and Expansion 

**Authors**: Zheling Tan, Jin Gao, Dequan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.17911)  

**Abstract**: As LLM agents operate across structured workflows and sessions, preserving long-term history does not ensure that later contexts can recover relevant evidence through a bounded memory interface. We study this evidence-reachability problem in long-term conversational memory, where retrieval still relies heavily on semantic similarity. This works well for topical recall, but it often misses earlier experiences, plans, or motivations that are semantically distant from the later events they help explain. Existing memory graphs provide cross-memory structure, yet links driven mainly by semantic overlap can duplicate what the host retriever already recovers. We argue that link construction should instead prioritize a sparse set of retriever-complementary associations. We present CABLE (Complementary Antecedent-Based Linking and Expansion), a plug-in augmentation that constructs links designed to extend the host retriever's direct semantic reach. For each new memory, CABLE generates antecedent-oriented queries, retrieves prior memories, subtracts candidates in the direct semantic neighborhood, and verifies the remainder before adding the accepted complementary associations into a sparse directed graph. At retrieval time, CABLE expands the host system's retrieved seeds along these links to surface implicit supporting evidence. We evaluate CABLE with A-MEM on LoCoMo and MA-LongMemEval, and further integrate it into SimpleMem and Mem0g on LoCoMo, using Qwen3.5-27B, DeepSeek-chat, and GPT-4o-mini. CABLE yields higher mean LLM-judge scores in every evaluated system-level setting, with the largest gains in categories where useful evidence is distributed across memories or sessions, including open-domain, multi-session, and preference-oriented questions. These results support prioritizing sparse, reasoning-relevant associations that complement rather than duplicate the host retriever. 

---
# BEAR-Bench: A Bilingual Enterprise and Academic Reasoning Benchmark for Multimodal Models 

**Authors**: Liubov Chubarova, Alexandra Kuleshova, Daniil Volkov, Kirill Sultanov, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2608.17895)  

**Abstract**: While Multimodal Large Language Models (MLLMs) have made significant strides in visual comprehension, their ability to reason about text-dense, professional documents remains incompletely evaluated. Existing benchmarks emphasize information extraction, require external domain knowledge, or cover professional documents only as one of many settings. They are also largely English- or Chinese-centric, leaving other languages and Russian, in particular, substantially underrepresented. To address these limitations, we introduce BEAR-Bench (Bilingual Enterprise and Academic Reasoning), a self-contained, complex English-and-Russian benchmark comprising 1000 human-annotated questions based on text-rich business and scientific documents. We evaluate 16 proprietary and open-weight MLLMs, including Gemini 3.1 Pro and Qwen3.5-397B, on BEAR-Bench and observe clear headroom even for the strongest systems. Finally, we use the resulting model outputs to compare existing hallucination detection methods, evaluating not only how often models fail on BEAR-Bench but also how reliably those failures can be identified. 

---
# BayesPrompt: human readable prompts that make sense 

**Authors**: Franky Kevin Nando Tezoh, Ali Hussaini Umar, Alessandro Laio, Guido Sanguinetti, Riccardo Rende  

**Link**: [PDF](https://arxiv.org/pdf/2608.17866)  

**Abstract**: Reconstructing prompts that can elicit a desired answer or behaviour in an LLM is an open and important research topic. Optimisation methods which aim at minimising the perplexity of a given answer, however, consistently yield so-called pseudoprompts, unintelligible strings of tokens which can lack human interpretability. We argue that this is a consequence of the ill-posedness of the prompt optimisation task. By reframing the task as a Bayesian posterior inference over prompts, we propose an efficient algorithm to sample prompts which are both efficient (in terms of perplexity) and human readable. We compare our approach with state of the art alternatives showing on a real data set a marked improvement over a range of metrics. 

---
# Encoded but Not Actionable: Auditing the Decode-Generate-Steer Gap in Frozen LLMs for Geometric Constraints 

**Authors**: Man Liang, Xinzhao Cheng, Faizan Wajid  

**Link**: [PDF](https://arxiv.org/pdf/2608.17843)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance on structured reasoning tasks, but what they encode and whether it informs model behavior remain unclear. We investigate this question through geometric reasoning, using parametric CAD constraints as a controlled testbed for separating local pairwise relations from sketch-level constraint status. By probing the hidden states of six frozen decoder-only LLMs, we examine four properties: linear decodability, forced-choice generation, activation-level influence, and behavioral steerability. Pretraining substantially improves the decoding of local geometric relations, and this advantage persists after accounting for positional cues with shuffled-order controls. In contrast, sketch-level DOF status is already highly decodable from randomly initialized representations and improves only modestly with pretraining, indicating that much of its probe performance is available without learned weights. Further analyses show that decodable information is not always actionable. Generation often fails to express this information, and on the two intervention-tested backbones, activation-restoration effects at the patched entity position vanish while decodability persists across depth. Mean-difference steering also does not reliably control outputs. These results show that decodability, generation, activation-level influence, and steerability can diverge in the tested setting. The audit provides a controlled way to distinguish failures to encode geometric structure from failures to express or control encoded information. 

---
# From Global Benchmarks to Local Evaluations: Benchmarking LLMs for the German Public Sector 

**Authors**: Camilla Dalerci, Thilo Michael, Robin Schaefer, Daniel Weinland  

**Link**: [PDF](https://arxiv.org/pdf/2608.17827)  

**Abstract**: Public institutions face a persistent challenge in selecting LLMs suited to their specific context. Existing benchmarks, however, are of limited use as they primarily reflect English-language and US-centric settings, and often only evaluate task performance. In this paper, we present first results of MÖVE, a holistic evaluation framework for the German public sector, examining three rarely considered governance dimensions: energy consumption, provider transparency, and knowledge of German-party positions. Our results reveal significant trade-offs, with no single model excelling across all dimensions: estimated energy consumption varies more than 60-fold and is not explained by model size alone, information disclosure varies systematically across providers, and European models do not exhibit stronger knowledge of German party positions. Model selection for public institutions thus cannot rely on performance rankings alone. Instead, evaluations should also reflect the governance requirements of the deployment context. 

---
# Interpretable Humans, Alien LLMs: Expert Analysis of Latent Structures in Assessment Responses 

**Authors**: Alona Strugatski, Licol Zeinfeld, Jason Cooper, Shelley Rap, Gil Schwarts, Giora Alexandron  

**Link**: [PDF](https://arxiv.org/pdf/2608.17810)  

**Abstract**: The evaluation of large language models (LLMs) relies heavily on human-designed assessments, implicitly assuming that AI and humans employ similar underlying cognitive constructs. Challenging this assumption, we investigate whether the latent factors governing LLM performance carry the same substantive, human-interpretable meaning as the cognitive constructs governing human learners. Using responses from humans and six LLMs across quantitative reasoning and chemistry assessments, we conducted Exploratory Factor Analysis (EFA) separately for both groups. Subject-Matter Experts (SMEs) then blindly evaluated the resulting factor graphs to ascribe pedagogical meaning to the emerged constructs. SMEs successfully interpreted most of the human-derived factors. Conversely, they could not ascribe meaning to any LLM-derived factors in quantitative reasoning and interpreted only half of the LLM factors in chemistry. By combining data-driven EFA with blind expert interpretation, this framework shows that LLMs frequently operate on statistically opaque mechanisms distinct from human reasoning. 

---
# Whether LLMs Can Navigate Beliefs and Facts Depends on How You Phrase It 

**Authors**: Quang Minh Nguyen, Luis Frentzen Salim  

**Link**: [PDF](https://arxiv.org/pdf/2608.17809)  

**Abstract**: Humans naturally form and express beliefs in daily communication, e.g., "I think the answer is 3" or "I suppose that's right." Such beliefs inevitably intertwine with fact and knowledge, making the ability to handle them in tandem desirable for large language models (LLMs), as they are increasingly deployed in user-facing settings. Prior work showed that even capable LLMs exhibit a systemic weakness in acknowledging user beliefs grounded in incorrect information. We extend this evaluation to 10 LLMs across 18 epistemic expressions and find that the size and direction of the weakness depend on the verb used to express the belief, with the accuracy gap between factual and false information ranging from +50% on "I vaguely remember" to -14% on "I seriously doubt". We further show that the phenomenon stems from task confusion: models default to fact-checking the underlying claim, overriding the user's stated belief; chains of thought that explicitly fact-check show lower accuracy on false information than those that do not; and a single instruction can reverse the failure across verb families. Mechanistically, models attend more to false beliefs they fail to confirm, but suppressing this attention at decoding time recovers accuracy only partially and only in some models, calling for future work on intervention methods. Our findings clarify prior results and show how fact-checking, a generally desirable behavior, can interfere with belief tracking in LLMs. Our code is available at this https URL. 

---
# TraceSQL: Traceable Answerability Estimation for Reference-Free Text-to-SQL Verification 

**Authors**: Neelesh Kumar Shukla, Debasmita Panda, Srutanik Bhaduri, Aditya Banerjee, Viji Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2608.17795)  

**Abstract**: Text-to-SQL systems are commonly evaluated using ground-truth SQL queries or reference execution results, but such supervision is unavailable at inference time in real-world deployments. This creates a critical verification problem: given only a user question, database context, and generated SQL, can a system estimate whether the generated query is likely to correctly answer the question? Recent approaches use LLMs as judge or specialized agents to inspect generated SQL, but their decisions can be difficult to trace. Outcome Reward Models (ORMs) address this by learning from execution-labeled candidate SQLs and assigning correctness scores to unseen queries, yet they still provide limited visibility into the signals behind each verification. To address this limitation, we propose TraceSQL, a lightweight and traceable verification model built on explicit diagnostic features. TraceSQL combines 67 features capturing question ambiguity, question requirements, question-schema-SQL consistency, SQL structure, and intent alignment. These signals remain available for examining which factors influence each prediction and for tracing decisions back to diagnostic evidence. On BIRD development databases, TraceSQL achieves 66.47% F1 and 64.48% ROC-AUC, compared with 61.87% F1 and 58.26% ROC-AUC for the GradeSQL-7B ORM baseline on the same generated-SQL evaluation. Feature attribution further shows that the model relies on both semantic grounding and deterministic SQL-structure signals. These results show that SQL verification can be performed with a lightweight learned model while retaining feature-level evidence for inspecting and diagnosing its predictions. 

---
# Preference Is Not Intervention: The Structure and Stability Boundaries of Reader-Specific Evidence Utility 

**Authors**: Shi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.17781)  

**Abstract**: ML systems increasingly condition decisions on downstream model identity, but this is useful only if model-specific differences form reusable structure rather than input-local interactions. We test this in retrieval-augmented generation (RAG), where evidence utility can be measured under controlled interventions. Holding query, evidence, task, scoring, and intervention fixed, nine readers disagree on effect sign in 33\% of jointly affected cells; reader$\times$query interaction explains 29.8\% of utility variance versus an 8.4\% permutation null; and self-selected evidence improves F1 by $+0.031$ ($t=3.39$). We then ask the sharper question: \emph{which components of this heterogeneity are stable reader properties across queries?} Separating three measurable objects---evidence \emph{activity}, \emph{ordinal preference}, and \emph{conditional signed direction}---we find ordinal reader geometry stable across four independent settings (split-half $\rho=0.60$--$0.83$): leave-one-out interventions, PRISM preferences, RAMDocs, and RAGuard. Signed geometry is task-bounded: weak in open-ended QA (0.14, 0.35), especially for misleading and irrelevant evidence, but strong in binary fact-checking (0.75) with no significant ordinal gap, though still below its sparsity-matched ceiling. Sparsity, decoding noise, and metric artifacts do not explain the main ordinal--signed gap. Finally, stable ordinal similarity fails to predict cross-reader intervention transfer (oracle-distance $\rho=-0.27$; regret reliability $-0.28$). Reader-specific utility exists, but preference is not intervention: stable ranking similarity does not license transfer of help/harm decisions. 

---
# Thinking in a Low-Resource Language: What SFT Builds, What RL Fixes, What Accuracy Cannot See 

**Authors**: Ayoub Kirouane, Christos Petrocheilos  

**Link**: [PDF](https://arxiv.org/pdf/2608.17744)  

**Abstract**: Take three frontier mixture-of-experts models (Alibaba, OpenAI, NVIDIA; 3.6-4.0B active parameters each) and fine-tune them to reason in a low-resource language. On accuracy benchmarks almost nothing happens, and the benchmark itself is noise at this scale: changing only the random seed moves the score by 7.7 points, more than every data and recipe effect we measured. That null is our first result. The real changes live where accuracy cannot see. Base models never think in Greek: 0 of 1,000 reasoning traces, even when the question is Greek, so the model answers correctly while reasoning in a form its user cannot read, audit, or correct. After supervised fine-tuning (SFT), every released checkpoint reasons in the language of the question on ~98% of items, one family at 3x fewer tokens, with judged grammaticality improving on all four models and general ability within a few points of each base: nothing was forgotten, and fluency was gained. We propose six behavioural dimensions that make such changes measurable, each gated to reject any metric that correlates with output length, and we report how our own instruments lied: six failures, each caught by a control. What SFT cannot do is fix its own defects: a quarter of answers skip the requested format, answers leak into the reasoning channel, and an explicit "think in English" is obeyed under half the time. Reinforcement learning with verifiable rewards, pre-registered before training, fixes the first two outright (fallback 24% to 2.5%, leak 3.5% to 0.0%, both against a flat random-reward control) and moves the third (+9.1pp), while the Greek reasoning habit survives an accuracy-only gradient untouched. We release five checkpoints. The instruments, the controls and the pre-registration travel to any low-resource language; Greek is the case that let us measure them. 

---
# Multi-turn Conversational AI from Text to Multimodal Interaction: Data, Models, Evaluation, and Open Challenges 

**Authors**: Syeda Faiza Ahmed, Zien Sheikh Ali, Hunzalah Hassan Bhatti, Firoj Alam, Shammur Absar Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2608.17605)  

**Abstract**: Conversational AI is moving beyond isolated text prompts toward sustained, multimodal interaction. In real conversations, users clarify goals, revise requests, interrupt responses, switch topics, and introduce new evidence while expecting systems to preserve context across turns. This makes multi-turn dialogue a distinct challenge requiring systems to maintain and update memory, ground responses across modalities, tools, and external knowledge, and adapt across languages and cultures. This study reviews multi-turn conversational AI across text-only dialogue, AudioLLMs and speech-native systems, multimodal and omni-modal systems, and tool-augmented agents. We organize the literature around datasets and benchmarks, modeling paradigms, training strategies, evaluation setups, and cross-cutting challenges. Our analysis shows that support for multiple modalities has advanced faster than the ability to sustain coherent interaction across a session. Despite stronger capabilities to perceive, speak, and act across modalities, current systems still struggle with persistent memory, cross-turn grounding, full-duplex interaction, robust evaluation, and cultural alignment. We conclude with a research agenda for systems that can remember, revise, ground, speak, listen, act, and adapt across turns, modalities, and cultures. (this https URL) 

---
# Write, Execute, Refine: From Skill Followers to Skill Optimizers via Reinforcement Learning from Execution Feedback 

**Authors**: Kang Peng, Zhiwei Zhang, Yichen Zhang, Zezhong Wang, Yiming Du, Geng Tu, Baojun Wang, Bin Liang, Ruifeng Xu, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2608.17587)  

**Abstract**: Expert-written natural language skills can improve tool-using agents, yet agent-authored skills perform 8-11 points worse than using no skill. This gap suggests that following procedural guidance and improving it from execution evidence are distinct capabilities. Inference time loops can repair skills but do not improve the model that writes the next one. We study how to organize execution experience from intermediate skills into training states for an optimizer. We introduce WER (Write, Execute, and Refine), a multi-phase framework that trains a Skill Optimizer outside a frozen executor. The optimizer proposes skills, a frozen agent executes each repeatedly, and a programmatic verifier scores the outcomes. The scores provide relative credit and select mixed-outcome records. Matched successful and failed trajectories from these records form the next phase's refinement states, so the optimizer learns from the consequences of its earlier outputs. On BFCL v4 multi-turn and tau2-bench, WER improves average Pass@1 over the no-skill baseline by 7.80 and 3.85 points, respectively. Under an identical refinement workflow, it outperforms the same backbone without optimizer training by 9.35 and 10.29 points. The trained 4B optimizer reaches 76.63 percent on BFCL v4, outperforming all evaluated off-the-shelf general-purpose models used as skill optimizers on average. 

---
# Auditing Exposure to Harmful Content on TikTok using Multimodal Language Models: A Cross-National, Age-Stratified Study 

**Authors**: Hamidreza Saffari, Francesco Pierri  

**Link**: [PDF](https://arxiv.org/pdf/2608.17583)  

**Abstract**: Online video platforms can expose young users to harmful content, but independent audits remain difficult because video annotation is costly and moderation judgments vary across languages. We audit TikTok in France, Italy, and Sweden with sockpuppet accounts representing four age personas (13, 16, 19, 40), collecting 36,971 videos from passive For-You-page scrolling and active sessions that scroll, search for harm keywords, and scroll again. To scale annotation, we validate four multimodal LLMs against native-speaker labels on a 300-video reference set. Gemini 2.5 Flash with eight sampled frames plus text performs best (aggregate kappa = 0.42), at half the per-call cost of native-video upload, and we apply it to a 10% sample for approximately \$50 in total API spend across both modalities. Keyword search returns 35-56% harmful content, a 1.5-7.5x increase over the scrolling baseline in ten of twelve country-age combinations; the spike is temporary and flattens the age differences observed in France and Sweden. Under passive scrolling, Italy has the highest harm rate at every age, with Italian age-19 reaching 48.6%. Overall, MLLM-based auditing offers a scalable approach for cross-national youth-safety audits, while provider safety filters (1.1% refusal rate) under-count the most explicit harms. 

---
# CoAL-RAG: A Complexity-Aware Legal Retrieval-Augmented Generation Method 

**Authors**: Jin Su, Zhuofeng Zhao, Huanhuan Wang, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.17536)  

**Abstract**: Legal consultation questions exhibit multi-level complexity. A single retrieval strategy often leads to over-reasoning for simple questions and poor interpretability for complex ones, making it difficult to meet the requirements for both answer quality and efficiency in high-risk scenarios. To address this issue, this paper proposes CoAL-RAG, a complexity-aware legal retrieval-augmented generation method, which constructs a multi-dimensional evaluation mechanism based on ``question essence'' and ``retrieval consistency'' to enable adaptive routing of retrieval strategies. First, the reasoning demand is quantified according to the logical structure of the question. Then, the discrepancy between semantic retrieval and keyword retrieval is utilized to indirectly reflect problem complexity, thereby selecting the most appropriate retrieval strategy and dynamically filtering contextual information. Experimental results demonstrate that the proposed method significantly outperforms baseline models not only on Chinese legal benchmarks (SocialLawQA, LawBench) but also demonstrates strong cross-jurisdictional generalization on English datasets (LexGLUE, CaseHold). Specifically, on Chinese datasets, the BLEU score improves by 42.5\% and ROUGE-L reaches 3.6 times that of knowledge graph-based methods. On English benchmarks, CoAL-RAG maintains highly competitive accuracy, achieving an optimal balance between generation quality, deep logical reasoning, and system efficiency across different legal systems. 

---
# ArborMem: Navigating Interaction States with Memory Forests 

**Authors**: Zongwei Lv, Yuemeng Xu, Yilun Yao, Siyi Ding, Xinyu Tan, Yaoming Li, Guangxiang Zhao, Weihong Lin, Lin Sun, Xiangzheng Zhang, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.17534)  

**Abstract**: Large language models increasingly serve as persistent conversational assistants, requiring memory that preserves relevant experience and maintains continuity across interactions. Existing methods improve access to conversational history through long-context processing, selective retrieval, and structured memory organization. However, most systems treat memory access as retrieving relevant past information without first determining which prior interaction state the current turn resumes. This limitation becomes particularly important when conversations interleave multiple tasks, people, and plans that may be interrupted and later revisited. We introduce ArborMem, an online memory framework that represents a long-running conversation as a navigable forest of interaction states. Each branch preserves a locally coherent trajectory, while the forest maintains multiple trajectories that may later be resumed. For each new input, ArborMem localizes the relevant state, restores its branch-local context, and augments it with reusable evidence retrieved across branches, preserving interaction continuity without conflating semantically related but structurally distinct trajectories. Existing long-term memory benchmarks cover diverse memory and reasoning capabilities but do not explicitly isolate branch-structured challenges. We therefore introduce BranchMemEval, a controlled diagnostic benchmark for interleaved and resumable interaction trajectories. Experiments on LongMemEval, LoCoMo, BEAM 100K, and BranchMemEval show that ArborMem outperforms the strongest baselines by 3.36 to 10.31 percentage points on the three established benchmarks and by 5.0 points on BranchMemEval. Its advantage grows under constrained read budgets, while complete memory queries remain below half a second. 

---
# Effects of Answer Format Variation on Gender Bias in Large Language Models 

**Authors**: Ksenia Merzlyakova, Sebastian Padó, Franziska Weeber  

**Link**: [PDF](https://arxiv.org/pdf/2608.17516)  

**Abstract**: Gender bias or other social biases in large language models (LLMs) are frequently evaluated with question answering or survey benchmarks where the LLM needs to give a response in a predefined answer format. It is well known in survey science that the answer format has a substantial impact on answers, just as LLMs are sensitive to the prompt wording. However, to our knowledge it has not been studied yet how changes in answer format impact the measurement of gender bias in LLMs and their alignment with human response distributions. We evaluate three instruction-tuned models on the BBQ benchmark and OpinionQA survey data across closed-ended, Likert-scaled and open-ended formats, comparing bias measurement and distributional alignment under otherwise identical conditions. We find that answer format does substantially alter measured outcomes, including reversals in order rankings. These differences arise because each format elicits distinct response behaviours, such as forced-choice selection, scale-based distributions and refusal in free-text generation. Our findings highlight the importance of treating answer format as a substantive component of LLM evaluation and motivate multi-format designs for more robust model assessment. 

---
# From Entity Mentions to Tone: An LLM-Based Pipeline for Media Bias Analysis 

**Authors**: Klesti Hoxha, Olti Qirici  

**Link**: [PDF](https://arxiv.org/pdf/2608.17454)  

**Abstract**: This paper presents a pipeline for analyzing media bias and framing in online news. The pipeline groups articles into topics and events, adds named-entity and sentiment annotations, and compares news sources through people mentions, source-level tone, and event-level coverage patterns. We apply it to 8,358 Albanian news articles collected from GDELT and compare the resulting annotations with GDELT's automated annotations. The results show moderate agreement for sentiment and entity extraction, as well as additional person-entity pairs that can potentially support the bias analysis. We compare two annotation prompts and find that stricter sentiment-validation rules remove label-score inconsistencies but increase execution time and reduce annotation coverage. Based on these results, the simpler prompt is used for the rest of the analysis. We have provided sample analysis on source-level framing pro les, person-level tone differences across sources, and event-level gatekeeping and coverage indicators. These outputs show how the same news collection can be used to examine what sources cover, how they describe public figures, and where coverage is concentrated. The approach is particularly useful in settings where manually verified datasets or specialized language tools are limited. 

---
# An Investigation of Translationese in the Generations of Multilingual Large Language Models 

**Authors**: Maria Valentini, Téa Wright, Julisa Granados, Eliana Colunga, Katharina von der Wense  

**Link**: [PDF](https://arxiv.org/pdf/2608.17399)  

**Abstract**: Text which has been translated from another language tends to carry with it evidence of translation$\unicode{x2014}$hence, it is often referred to as $\textit{translationese}$. Multilingual large language models (MLLMs) generate text in a variety of languages. However, it is still unclear if MLLMs' generations resemble internal translation (from English or, potentially, other languages) and, thus, result in translationese. Here, we ask the following research questions: (1) Does text generated by MLLMs resemble translationese? (2) How does translationese produced by MLLMs differ from translationese produced through direct translation? We leverage established indicators of translated text to evaluate text generated by state-of-the-art MLLMs in five languages, comparing to both non-translated and human-written baselines in order to isolate translationese from other kinds of interference. Through the use of high-accuracy classification models, analyses of variance on individual linguistic features, and the collection of human annotations in a subset of two languages (German and Spanish), we assess the translationese content of MLLM generations and examine the key features that distinguish MLLM-generated text from typical translation-related interference. 

---
# PTXBench: Benchmark and Adapt LLMs for GPU Kernel Optimization with Architecture-specific PTX 

**Authors**: Genghan Zhang, Yixin Dong, Chengze Fan, Zhichen Zeng, Yueming Yuan, Shaowei Zhu, Kunle Olukotun  

**Link**: [PDF](https://arxiv.org/pdf/2608.17379)  

**Abstract**: We introduce PTXBench, a benchmark for evaluating and adapting large language models (LLMs) to use architecture-specific PTX for GPU kernel optimization. PTXBench measures functional correctness, whether selected target instructions execute at runtime, and speedup over frontier libraries across GEMM and attention workloads on H100 and B200 GPUs. Our evaluation shows that architecture-specific PTX capability remains uneven: success rates fall substantially on complex attention backward workloads, and executing the target instructions does not necessarily translate into competitive performance. No evaluated model consistently matches frontier libraries across the suite. We further adapt Qwen3.6-27B using supervised fine-tuning. Repair-conditioned training improves several tasks, but generalization remains uneven; data coverage, balance, and the quality of the reasoning teacher matter in addition to dataset size. PTXBench provides an auditable testbed for measuring and improving LLMs' ability to exploit evolving GPU architectures. 

---
# ArguLens: An Open-Source System for Automated Essay Scoring and Label-Aware Feedback Generation 

**Authors**: Weiran Wang, Hongxiang Shi, Huitao Tang, Wenjuan Qin  

**Link**: [PDF](https://arxiv.org/pdf/2608.17356)  

**Abstract**: Most automated essay scoring (AES) systems output a single holistic score without interpretable evidence and rely on closed APIs that introduce data privacy and cost barriers. We present ArguLens, an opensource, locally deployable system that decomposes AES into three decoupled components: a discourse-move classifier (Qwen2.5-7B-Instruct fine-tuned with LoRA on PERSUADE 2.0), a grade-independent LightGBM scorer over 31 linguistic and discourse features, and a label-aware feedback generator served through vLLM with a Qwen2.5-14BInstruct backbone. A Gradio web UI exposes pluggable inference backends and supports single-essay and batch scoring with downloadable per-essay breakdowns. On an essaydisjoint PERSUADE 2.0 test split, the logitprobe classifier achieves 82.6% accuracy and 0.727 macro-F1; under prompt-grouped 5-fold cross-validation the scorer reaches a mean QWK of 0.813 under an oracle discoursefeature protocol, and an ablation shows that adding gold discourse annotations yields an increment of +0.055 QWK over the lexical+syntactic configuration (paired t-test, p = 0.010). This is a component-level diagnostic rather than an end-to-end classifier-to-scorer result. The feedback generator ships with a structured evaluation protocol; its human-rater study is left to future work. The system is released under Apache 2.0 at this https URL. 

---
# What Tokens are Learned when Tokenization is Optimized Jointly with Language Modeling? 

**Authors**: Saketh Reddy Vemula, Parameswari Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2608.17325)  

**Abstract**: Tokenization is a fundamental component of language modeling pipelines. Despite its importance, it is often fixed, even though it significantly impacts model performance across languages. In this work, we analyze what tokens are learned when tokenization is jointly optimized with language modeling. We compare tokenizer-free approaches such as SSLMs and H-Nets with fixed tokenizers across 18 typologically and script-diverse languages. Our results show that joint optimization fundamentally alters token structure. SSLMs recover morphologically aligned and contextually efficient tokens, whereas H-Nets prioritize byte-level efficiency, producing longer tokens with very low overlap with standard subword vocabularies. We further show that tokenization behavior varies across language typologies. Agglutinative languages exhibit more dynamic segmentation patterns while learning. Through downstream evaluation, with pretrained-then-finetuned BERT models, we find that SSLM-based pretokenization consistently reduces language modeling perplexity and achieves competitive downstream performance despite distinct vocabularies. Overall, tokenizer-free approaches optimize for contextual and computational efficiency rather than strict morphological structure, resulting in fundamentally different yet effective vocabularies for downstream NLP. 

---
# Q-Interference: Memory-Efficient Phase-Aware Quantum-Inspired Attention 

**Authors**: Emama Nahid, Tahmid Imtiaz Imu, Huayue Gu, Liran Ma, Zhipeng Cai, Honghui Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.17288)  

**Abstract**: GPT attention measures token compatibility through dot-product similarity. This mechanism is simple, effective, and memory-efficient. But it does not explicitly model whether strong token features should reinforce or suppress one another. We introduce Q-Interference, a fully classical quantum-inspired attention mechanism for autoregressive language modeling that augments each query and key feature with an amplitude and a learned phase. The resulting attention score is phase-aware which aligned phases contribute constructively while conflicting phases contribute destructively. Although Q-Interference yields a richer interaction rule than similarity alone, a naive implementation of Q-Interference requires a large token-pair-feature interaction tensor, making it memory-intensive and often impractical. To address this limitation, we propose an exact trigonometric factorization that computes the same score using two standard matrix multiplications avoiding materialization of the large intermediate tensor. Q-Interference fits directly into a Transformer block in GPT and leaves the remainder of the model architecture and next-token prediction objective unchanged. Experiments on public benchmark datasets and baseline models show that the proposed reformulation trains stably in a controlled GPT-style setting and provides a consistent memory advantage over naive phase-aware interference attention. These results support the specific contribution of this work: an exact memory-efficient reformulation that makes phase-aware interference attention practical within a standard GPT pipeline. 

---
# Temporal Leakage in Financial News NLP: A Multi-Architecture Audit with a Regime-Specific M&A Signal 

**Authors**: Chenhao Xue, Raslen Guesmi, Siwei Feng, Yucheng Gong, Jacob Xavier Sundram, Jordan Pang, Lan Wang, Julian Kaljuvee  

**Link**: [PDF](https://arxiv.org/pdf/2608.17223)  

**Abstract**: Financial-news direction prediction has become a popular NLP benchmark, yet reported gains depend critically on whether the train-test split is chronological or random, i.e., on temporal leakage. We audit this dependence on a 49,799-article corpus across 16 feature-model combinations spanning TF-IDF, MiniLM, FinBERT, and fine-tuned RoBERTa-large / DeBERTa-v3-large, plus separate zero/few-shot and LoRA probes of Llama-3 and Qwen2.5 LLMs: random splits inflate MCC by $1.1\times$ to $6.5\times$, tracking model capacity and feature richness, and end-to-end FinBERT fine-tuning re-amplifies rather than closes the gap (size-matched ratio $1.75\times$). Conditioning on event type, mergers and acquisitions (M&A) is the only audited category with a positive locked-test signal under near-temporal chronological evaluation (TF-IDF MCC $= 0.138$ train-only, $0.068$ under train$\cup$val refit; 10,000-permutation $p < 10^{-3}$); the signal does not transfer to FNSPID's 2009-2020 U.S. corpus, localising the headline to our 2024-2025 European-tilted M&A semantics rather than a universal predictor. Three independent role labellers converge on acquirer-tagged articles as the signal locus, a power-limited qualitative convergence rather than a hypothesis-tested asymmetry. Chronological splitting plays for financial NLP the role characteristics-purging plays for asset pricing: it strips the predictable, stale component of news and leaves a residual that is small, event-localized, and lexically shallow. We advocate leakage audits as a required disclosure for financial-NLP benchmarks. 

---
# The Plot Thins: Uniformity and Linearity in Literary Summaries 

**Authors**: Rebecca M. M. Hicke, Sil Hamilton, David Mimno, Ross Deans Kristensen-McLachlan  

**Link**: [PDF](https://arxiv.org/pdf/2608.17218)  

**Abstract**: Works of literature are complicated; they balance plot, suspense, surprise, and artistic expression. Summaries of literature prioritize plot, and therefore may deviate from their sources. Using a combination of manual and LLM-based annotation, we construct a dataset mapping sentences from 150 novel summaries to their respective source chapters. We find the task unexpectedly difficult for both human and model annotators. Using the sentence-to-chapter mappings, we then measure summary linearity, the degree to which it maintains the source's order of events, and uniformity, the degree to which a summary spreads attention equally across a source. By examining when and how summaries break linearity and uniformity, we identify differences in how literary works and summaries express plot, particularly with regard to the clarity and prominence with which narrative details are described. 

---
# Which Source Wins? Task-Dependent Reliance in Vision-Language Models 

**Authors**: Rodela Ghosh, Aviral Gupta, Guangjing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.17205)  

**Abstract**: Vision-language models (VLMs) combine images and text, but when the two conflict and one becomes harder to read, it is unclear how a model shifts its reliance between them. We study this modality reallocation with a controlled setup: we degrade either the image or the text across four levels of legibility while keeping the other clean, and track how the model's preference changes. We build conflicts from GSM8K and SVAMP by pairing the rendered image of one arithmetic problem with the text of another, so the two sources support different answers. We also introduce ChartQA-Conflict, a manually reviewed benchmark of 229 chart-report conflicts with matched chart and table-image representations. We evaluate six open-weight VLMs using both generated answers and a length-normalized conditional log-likelihood margin. On GSM8K and SVAMP, five of six models shift more strongly away from degraded text than from degraded images. On ChartQA-Conflict, all six likelihood-scored models exhibit the opposite pattern, shifting more strongly away from the degraded visual source. This reversal persists after calibrating for unimodal accuracy loss and after replacing charts with plain table images. Two frontier API models, GPT-5.6-Luna and Gemini-3.5-Flash, behaviorally replicate the ChartQA-Conflict reversal, with GPT-5.6-Luna also matching the arithmetic direction. These results show that modality reliance in VLMs is not fixed, but varies across tasks, evidence structures, models, and evaluation settings. The source code is available at this https URL. 

---
# Token Optimization and Context Window Management in Multi-Agent AI Workflows 

**Authors**: Dvir Shamay  

**Link**: [PDF](https://arxiv.org/pdf/2608.17188)  

**Abstract**: Multi-agent AI workflows are limited not only by model quality but by token cost, latency, and context-window quality. This paper presents a practitioner framework for token optimization and context-window management, grounded in an internal production dashboard that extracts structured work items from meetings, email, and chat with LLMs and routes summaries across workstreams. Six patterns are described: context stratification, fetch-once/process-locally architecture, schema-contracted prompts, token-aware fallback chains, semantic caching, and inter-agent communication compression. In production they cut measured cold-load latency to 61-116 seconds (six timed runs) from an operational baseline of roughly 3.5-10.5 minutes, with an estimated 60-70% token reduction. It also reports a controlled context-composition study: 2,420 confirmatory trials across 11 model configurations, using 661 anonymized workplace items scored for relevance. Holding the prompt at a fixed ten items, replacing some high-relevance items with same-domain low-relevance items improves the model's relevance-score concordance on the target items, versus high-relevance items only; we call this relevance-contrast context. In the all-11 paired analysis, the 50:50 signal/noise condition improved relevance accuracy by +0.077 over the 100% condition (naive 95% CI [+0.056, +0.098], Cohen's d = 0.49, Holm-adjusted p < .001, n = 220). These cells are not independent; by the nine model families the effect is +0.084 (95% interval [+0.064, +0.103]), reported as a within-corpus descriptive comparison, not a population inference. A Fusion-of-N follow-up found that learned synthesis did not beat the mechanical set union of item IDs. The contribution is a measured engineering layer between model research and production agent practice: repeatable patterns and evaluation methods for faster, cheaper, more reliable workflows. 

---
# AISA: AI Safety Assistant Framework for Continuous Improvement of Highway Construction 

**Authors**: Mason Smetana, Trevor Neece, Lev Khazanovich  

**Link**: [PDF](https://arxiv.org/pdf/2608.17184)  

**Abstract**: Job Safety Analysis (JSA) and pre-task planning can benefit from prior incident records, yet historical accident data is often stored as unstructured narratives that are difficult to consult at the point of planning. A novel framework centered on large language models (LLMs) for highway construction safety reporting and planning is proposed as a foundation for future agentic applications, prioritizing deterministic, local inferencing. The first aim is to enable classification and quality scoring of incident narratives for existing and future reporting purposes. The second is to evaluate retrieval of relevant historical accidents, related imagery, and trusted industry documents for incorporation into daily safety plans. Neural probes were trained to classify incidents along four multiclass and two binary Occupational Injury and Illness Classification System (OIICS) fields and to derive an overall quality score, evaluated on a test set of over 15,000 narratives and a held-out set of 100 author-labeled records, benchmarked against a majority-vote LLM ensemble. The retrieval of historical accidents, reference imagery, and industry documents was benchmarked across embedding models using standard information retrieval metrics. OIICS classification reached 75% held-out accuracy, though the two binary flags were degenerate. The quality score, while meaningful on one database, was distorted on out-of-distribution fatalities in the held-out dataset. Accident retrieval recovered relevant incidents far above chance, performing best on lexically distinct construction activities. On document question answering, an open-weight decoder embedding model surpassed proprietary models. Overall, this work provides a new framework rooted in local inferencing and text embedding models for future agentic applications, with emphasis on bridging external data to JSA reports. 

---
# Polaris: Learning to Generate Table Descriptions from Retrieval Feedback 

**Authors**: Ting Cai, Tuan Minh Phan, AnHai Doan  

**Link**: [PDF](https://arxiv.org/pdf/2608.17171)  

**Abstract**: Many table-centric NLP tasks such as NL2SQL first retrieve relevant tables from large collections using keyword search. Recent work uses LLMs to generate natural-language table descriptions to improve retrieval, but they are typically optimized for fluency rather than retrieval effectiveness. We present Polaris, a system that trains an LLM to generate table descriptions directly from retrieval feedback. Our key insight is that existing table retrieval benchmarks already contain the supervision needed for this task: given query-table relevance judgments, we generate multiple candidate descriptions for each table, rank them by their BM25 retrieval effectiveness, and use the resulting preference pairs to fine-tune the LLM with Direct Preference Optimization (DPO). Polaris further expands abbreviated table and column names before generation to reduce vocabulary mismatch. Extensive experiments show that Polaris outperforms the state-of-the-art AutoDDG solution, often by a significant margin. More broadly, our results demonstrate that retrieval benchmarks can be repurposed as supervision for training LLMs to generate retrieval-oriented metadata. 

---
# Can LLMs Reason in a Legally Meaningful Manner? A Small-scale Study on European Court of Human Rights Cases 

**Authors**: Amogh Raina, Ilias Chalkidis, Daniel Hershcovich, Henrik Palmer Olsen  

**Link**: [PDF](https://arxiv.org/pdf/2608.17168)  

**Abstract**: Reasoning has become a standard technique and feature for contemporary LLMs; however, its application and quality in the context of demanding legal-oriented tasks, such as legal case forecasting, remain under explored. We investigate how LLMs reason in the context of legal case forecasting, using legal cases from the European Court of Human Rights (ECtHR) as a testbed. We evaluate OpenAI GPT 5.4, a recent top-tier LLM, by exploring alternative prompting strategies that are more or less suggestive of what counts as legally meaningful reasoning in the context of ECtHR jurisprudence. We present our findings derived from assessing the model's responses with both human and LLM evaluation. We find that the examined model scores far from ideal in legal reasoning, the model produces structurally complete but substantively shallow analyses, and that LLM-as-a-Judge evaluators are internally consistent yet align only weakly with our trained annotators, i.e., reliable but not a valid substitute for human evaluation. Overall, the expert-curated prompt leads to more comprehensive reasoning, which does not result in more accurate predictions compared to the other examined settings. Based on our findings, we urge the community not to rely solely on automated LLM-based evaluation and to avoid using task accuracy as an appropriate proxy for reasoning quality. 

---
# Towards Safer RAG: Only Agents Capable of System 2 Thinking may Access Untrusted Documents 

**Authors**: Mehrdad Ghassabi  

**Link**: [PDF](https://arxiv.org/pdf/2608.17153)  

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly enhanced the performance of large language models (LLMs), yet these systems remain vulnerable to knowledge-poisoning attacks, in which misinformation in retrieved documents can influence the model's final outputs. Notably, an LLM may correctly detect that a document contains incorrect information while nevertheless being influenced by it. Prior work has addressed this vulnerability through the Cordon Principle, which prevents models responsible for final answer synthesis from directly accessing raw evidence. Although effective, this strict isolation can introduce substantial computational overhead. In this work, we propose a refined security principle: only agents capable of deliberative System 2 reasoning may access untrusted documents. To evaluate this principle, we introduce novel metrics that quantify the discrepancy between misinformation detection and downstream influence. We then empirically compare state-of-the-art reasoning language models with standard language models across these metrics. Our results show that reasoning-capable models are substantially more robust to corrupted evidence, without requiring the strict isolation imposed by the Cordon Principle. These findings provide empirical support for our refined principle and suggest a more practical foundation for secure RAG system design. 

---
# Children, but not language models, show accelerating returns in word learning 

**Authors**: Michael C. Frank  

**Link**: [PDF](https://arxiv.org/pdf/2608.17120)  

**Abstract**: Children learn hundreds of words over the first years of their lives, in a process that begins slowly but quickly picks up speed. Prior models describe vocabulary growth as evidence accumulation over time. Here we show that the process is best characterized as accelerating accumulation: children learn more from each additional unit of linguistic experience than they did from the one before. In contrast to children, language models -- even those trained on child-directed speech -- do not accelerate. Instead, they show constant proportional returns on new data, consistent with scaling laws. Children learn using many orders of magnitude less training data than language models; their increasingly efficient use of their learning input is a candidate explanation. 

---
# Emotion Across Speech and Faces: Shared Affective Mechanisms in Multimodal Foundation Models 

**Authors**: Xiutian Zhao, Luqi Sun, Björn Schuller, Berrak Sisman  

**Link**: [PDF](https://arxiv.org/pdf/2608.17102)  

**Abstract**: Modern multimodal foundation models (MFMs) have made rapid progress on tasks requiring integrated perception across speech, vision, and language, including emotion recognition. However, it remains unclear whether they recognize speech and facial emotion through shared affective functional units or modality-specific pathways. We explore emotion-sensitive neurons (ESNs), sparse decoder neurons selectively associated with emotion categories, in three MFMs: Gemma-4-12B-it, MiniCPM-o-4.5, and Qwen2.5-Omni-7B. Using speech emotion recognition and facial expression recognition as complementary probes, we identify acoustic and visual ESNs. Visual ESNs are causally meaningful: deactivating them selectively impairs recognition of the associated facial emotion, whereas steering their activations selectively enhances recognition of that emotion relative to other emotion categories. Acoustic and visual ESNs further show emotion-matched overlap and similar layer-wise distributions, indicating partial structural alignment between affective representations across speech and faces. Finally, cross-modal interventions reveal bidirectional causal transfer: ESNs identified from one modality produce emotion-specific effects when applied to the other. Our findings provide one of the first cross-modality activation-level analyses of affective functional units in MFMs, suggesting that speech and facial emotion recognition partially converge onto sparse decoder-level components that can be localized and manipulated without training. 

---
# A Glyph Is Not a Letter, a Token Is Not a Word, a Space Is Not a Space: What the Units of Voynichese Are Not 

**Authors**: Liudmila Rozanova, Alexander Temerev  

**Link**: [PDF](https://arxiv.org/pdf/2608.17096)  

**Abstract**: The Voynich manuscript (Beinecke MS 408) is usually analysed on three unstated assumptions: that its glyphs are letters, that the strings between blanks are words, and that every blank is a word space. We test all three against the Zandbergen-Landini transliteration with matched prose, cipher, and pseudo-text controls and quire-level resampling. None holds, and the failures share a shape: the order in Voynichese sits at the edges of tokens and at graded boundaries between them, not in the succession of tokens themselves. Glyph regularity is too strong for one-to-one substitution of any tested plaintext (conditional entropy 2.7 bits against about 3.5 for Latin, Italian, and English) and resolves instead onto a quire-stable scale of recurrent multi-symbol units. Tokens form a plausible vocabulary, yet the identity of one token predicts the next by under 1% of token entropy, below every matched control (2-10%), while the glyphs at token edges share 0.2 bits of mutual information, more than in any prose control. Blanks fall into two regimes: the separators transcribers marked uncertain behave like word-internal junctures, are physically narrower on the page (AUC 0.905 from independent image coordinates, with the same sign in a small blind ink audit), and are crossed by learned units even when every space is erased before learning. This profile is also what discriminates. A published Voynich-imitating cipher and a self-citation text generator both reproduce the low entropy, the unit scale, the weak token order, and the null result of a calibrated substitution attack; neither reproduces the edge-glyph coupling or the open, hapax-rich vocabulary (70% singleton types against 41% and 59-60%). Any account of the manuscript must therefore earn, rather than assume, the step from glyphs, tokens, and separators to letters, words, and word spaces, and these are the measurements on which to do so. 

---
# There is No Theoretical Curse of Multilinguality For Embedding Space Structure 

**Authors**: Niyati Bafna, Neha Verma, Vilém Zouhar, Philipp Koehn, David Yarowsky  

**Link**: [PDF](https://arxiv.org/pdf/2608.17088)  

**Abstract**: A central goal of multilingual NLP is to achieve high monolingual performance per language and cross-lingual alignment for large-scale language coverage with a multilingual model. The curse of multilinguality describes the phenomenon of degradation in multilingual model performance as we increase language coverage, posing a threat to the above goal. This paper asks whether multilingual embedding spaces are inherently incapable of achieving perfect multilinguality without a prohibitive increase in required capacity. We first formalize the goal of "perfect multilinguality", embodied in two multilinguality conditions. We then prove that the minimum dimensionality required for perfect multilinguality grows only logarithmically in the number of languages. That is, we show that there is no theoretical curse of multilinguality for embedding space structure. This suggests that the empirical curse of multilinguality is a result of real world data and training conditions. We back this understanding with a small-scale empirical study. Our paper provides the first theoretical and intrinsic perspective on the curse of multilinguality, with implications for the scientific understanding of this phenomenon. 

---
# Uncertainty-Aware Decision Making in Multimodal Large Language Models 

**Authors**: Abderrahmene Boudiaf, Irfan Hussain, Sajid Javed  

**Link**: [PDF](https://arxiv.org/pdf/2608.17084)  

**Abstract**: Multimodal large language models (MLLMs) increasingly answer questions whose correctness depends on visual, textual, temporal, acoustic, document, chart, or embodied evidence. Their failures are therefore not only linguistic. A fluent answer may conceal poor input quality, a perceptual error, weak grounding, conflict between modalities, unstable reasoning, distribution shift, or a question that is not answerable from the supplied evidence. This survey organizes the literature on uncertainty-aware MLLMs around a decision-centered framework: uncertainty sources give rise to observable signals, signals must be calibrated or controlled for risk, and calibrated uncertainty should determine the system action. We review work on token and logit uncertainty, semantic disagreement, perturbation instability, grounding and attribution scores, verbalized confidence, verifier and judge scores, conformal prediction, selective answering, abstention, clarification, retrieval, self-checking, and escalation. The central argument is that uncertainty should not be evaluated only as a confidence number; it should be evaluated by whether it improves behavior under insufficient, conflicting, shifted, or high-risk multimodal evidence. We position this survey against text-only uncertainty and abstention surveys, broad MLLM surveys, MLLM hallucination surveys, and safety-oriented reviews. We conclude with open problems in source-aware decomposition, action-aware benchmarks, calibration under shift, black-box uncertainty estimation, broader modality coverage, reproducible reporting, and human-centered uncertainty communication. 

---
# Foundation Agents Meet Agentic Deep Research: Evidence-Grounded Clinical Code Forecasting 

**Authors**: Junda Wang, Meysam Ghaffari, Akshat Choube, Mohsen Sharifi Renani, Hong Yu, Carlos Morato  

**Link**: [PDF](https://arxiv.org/pdf/2608.17075)  

**Abstract**: Next-encounter ICD forecasting predicts which standardized diagnosis codes will be documented at a future visit from the longitudinal record available beforehand. The task is prospective and multi-label: the target note does not yet exist, and several codes may be correct. Structured EHR foundation models capture recurrence and temporal progression, whereas language foundation models generate flexible diagnostic hypotheses. We introduce ICD-Deepresearch, a DeepResearch workflow that composes these predictive foundation models with medical search and ICD dictionaries. Because no source reveals the future code set, research evaluates candidate transitions by linking patient evidence, external clinical relations, and exact code semantics under a fixed top-K budget. Candidate Generation uses SparseEHR to produce an EHR Prior that initializes two bounded Research Expansion rounds; an independent GPT-5 Direct Forecast supplies complementary candidates. Final Selection validates, deduplicates, and jointly ranks both paths, after which a separate module writes rationales without changing predictions. Finally ICD-Deepresearch achieves patient-averaged precision/recall of 24.60/35.09% on MIMIC-III and 25.14/48.32% on MIMIC-IV. Physicians rate 51% and 68% of its retrieved documents useful, compared with 22% and 39% for standalone GPT-5 web search and 32% and 41% for Medical Deep Research. ICD-Deepresearch therefore improves over the registered local comparators while retrieving evidence with higher physician-rated usefulness than the standalone research systems 

---
# Institution-Specific LLM Prompting Recovers PHI That De-identification Systems and Their Gold Standards Both Miss 

**Authors**: Daniel Palacios, Matthew Brady Neeley, Angel Adetomike Otto, Shalini Dhamodharan, John P. Woodhouse, Chi-fan Lin, Mark Zobeck, Zhandong Liu, Hyun-Hwan Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2608.17051)  

**Abstract**: Secondary use of electronic health records requires de-identification, yet existing systems miss \emph{institutionally situated} protected health information (PHI) such as hospital abbreviations, building names, and internal codes whose status is locally determined. We ask whether large language models (LLMs) with in-context learning (ICL) can close this gap and control the precision--recall trade-off.
On 100 annotated pediatric oncology notes (5,322 PHI spans) from Texas Children's Hospital, we benchmarked eight LLMs against two purpose-built systems (Stanford TiDE, OpenMed PII) and two pattern-based baselines. Each LLM ran under three prompts of increasing specificity: (1) a HIPAA-aligned baseline, (2) baseline plus the institutional PHI categories it missed, and (3) prompt 2 plus instructions against over-redacting clinical content. We then compared 14~multi-agent and ensemble configurations against the best single prompt, with recall the primary safety metric.
LLMs outperformed the purpose-built systems (best F1=0.918$\pm$0.001 vs.\ TiDE 0.779), with advantages concentrated in contextual categories. Naming the missed categories recovered 79\% (48/61) of them, and discouraging over-redaction restored precision. No agentic architecture beat calibrated single-pass prompting (F1 0.906--0.907), but LLM outputs surfaced 414~candidate annotation gaps; re-annotation confirmed 227~PHI spans, against which the final prompt reached recall=0.981 (F1=0.907$\pm$0.002).
Well-calibrated ICL resolves both the institutional PHI gap and the precision--recall trade-off in one LLM call per note. LLMs cost more to run than traditional methods, but that cost buys a way to audit the reference standard.
LLMs are a legitimate, adaptable alternative to purpose-built de-identification systems; institution-specific prompt development should be the primary adaptation strategy. 

---
# Cross-Model Memory Transfer via Target-Side Reader Adaptation 

**Authors**: Mingyuan Li, Guangsheng Yu, Xu Wang, Shaoxiong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2608.17050)  

**Abstract**: Methods for improving knowledge use in large language models typically fall into two regimes. Non-parametric retrieval offers flexible access to external knowledge, but adds retrieval latency, context overhead, and only shallow integration with the backbone. Parametric adaptation is efficient at inference time, but entangles knowledge with model weights and can be hard to update, audit, or transfer. Engram-style hashed memory occupies a middle regime: it stores learned information in an external, addressable table, yet consumes that table through a small learned reader. This raises a basic question: when such a memory is moved across backbones, what matters more, the frozen memory itself or the target-side reader? We study this question through cross-model frozen-memory extraction, in which a memory trained on a source model is frozen and attached to a different target model, with only a lightweight reader trained. Ablations show that learned memory content and correct addressing both matter, but the transferred table becomes useful only through a reader aligned to the target model. In downstream question answering tasks, a dual-layer, four-branch reader nearly closes the gap between same-model and cross-model reuse, achieving an average score of 38.8 under our controlled evaluation protocol. Moreover, when the provider reader is directly compatible with the target interface, the frozen artifact can provide substantial utility without target-side training, while optional reader adaptation yields further improvement. These results suggest that Engram can serve as a reusable external knowledge artifact, provided that the target has access to a compatible reader interface; target-side adaptation can further improve alignment when direct reader reuse is insufficient. 

---
# Margin-Regularized Structured Semantic Alignment for Brain-Language Correspondence 

**Authors**: Jiaqi Wang, Huawen Hu, Shu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.16975)  

**Abstract**: With the rapid advancement of large language models, brain-language decoding has achieved remarkable progress. However, it remains unclear whether decoded content genuinely reflects neural representations or is largely reconstructed by the language model itself. This ambiguity limits interpretability and hinders the investigation of intrinsic brain-language correspondence. To address this challenge, we propose MD-SigLIP. This margin-regularized structured semantic alignment framework directly aligns brain embeddings with text embeddings in a shared semantic space, enabling retrieval-based decoding. This formulation enables explicit modeling of the correspondence between neural representations and language semantics. Building upon duplicate-aware sigmoid contrastive learning, we introduce a listwise margin-regularized term that enforces structured ranking constraints between positive semantic clusters and negative samples. By modeling multi-positive semantic structure and margin-based ordering simultaneously, the method captures the manifold organization of language embeddings reflected in neural signals. Experiments demonstrate state-of-the-art retrieval performance under both full-vocabulary and subset evaluation settings. 

---
# On the Fragility of Self-Improving Agents: Variance, Task Order, and Underspecification 

**Authors**: Qinyuan Ye, Yu Li, Yada Pruksachatkun, Jiaxin Zhang, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.18066)  

**Abstract**: Memory-based self-improving agents--those that learn from an online stream of tasks and improve over time by maintaining a textual memory bank--have shown great promise in recent literature. However, the reliability aspects of these methods have been critically overlooked. In this work, we conduct a comprehensive re-evaluation of two memory-based methods, broadening the scope of evaluation along two axes: (1) including multiple runs to quantify variance, and (2) randomly shuffling the tasks to investigate the effect of task order. Through these experiments, we make two observations that expose the fragility of current methods: First, agent evaluation is inherently noisy in complex environments and on multi-step tasks, and stacking a self-improving loop on top can further amplify this noise. Second, the agent's improvement is highly dependent on task order. Prior works often adopt default orderings that impose an implicit curriculum, acting as a hidden prerequisite for success.
To better understand this fragility, we manually examine the agents' memory and hypothesize that task and environment underspecification contribute to this fragility. We validate this hypothesis by incorporating information that enables better specification, such as detailed rubrics and environment feedback, into the memory construction process. While this added information partially closes the performance degradation in previous experiments, significant gaps still remain, suggesting that other uncharacterized factors contribute to this fragility. Looking ahead, our work advocates for more rigorous evaluation protocols for self-improving agents by reporting results across multiple runs and stress-testing them under challenging conditions. Moreover, our findings on underspecification call for systems and interfaces that enable effective human oversight, preventing agents from failing in unforeseeable ways. 

---
# Against Political Polarization: A Unified Framework for Tracing Evolving Political Ideologies on Social Media 

**Authors**: Yijie Xu, Chao Wang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2608.17987)  

**Abstract**: The rapid growth of social media has greatly influenced political discourse, highlighting the need to understand individual political ideologies and their temporal dynamics. This task faces challenges such as data scarcity, abundant non-political content, costly and bias-prone manual annotation, and difficulty in modeling future ideological inclinations. To address these issues, we propose TSN4PI, a unified framework for tracking the evolution of political ideologies on social media. It includes two core modules. The PIDN uses large language models with style transfer and unsupervised domain adaptation to enable robust ideology detection and filter irrelevant content from noisy, cross-domain data. The PIPN employs temporal graph neural networks to predict future ideological shifts, enabling comprehensive analysis of ideology presence, intensity, and evolution. We release two large-scale datasets for noncommercial research use to facilitate further work. Extensive case studies on multiple platforms (X and Truth Social) validate the effectiveness of TSN4PI and provide empirical insights into political polarization and the evolution of online ideologies. Our findings offer a nuanced perspective, advancing both methodological development and empirical understanding in this field. 

---
# Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation 

**Authors**: Zhizhao Liu, Zhiliang Tian, Xi Wang, Zhihua Wen, Yihang Xiong, Zhiquan Lai, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.17941)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) improves the reasoning capabilities of large language models but relies on costly rollout exploration. Assigning the same exploration budget to samples with different difficulty levels is inefficient: easy samples may receive redundant rollouts, whereas difficult but learnable samples may receive too little exploration. Existing adaptive schedulers address this mismatch through curriculum-based sample selection or non-uniform rollout allocation based on estimated sample difficulty. However, obtaining reliable online difficulty estimates remains challenging: dedicated probing adds substantial generation overhead, whereas history-based estimators face a cold start with no initial observations and stale feedback, and typically ignore relations among samples. To address these limitations, we propose a plug-and-play graph-based online difficulty estimator that shares rollout feedback across related samples and continuously updates their difficulty estimates, mitigating cold start and staleness without dedicated probing. Specifically, we first construct a difficulty-aware sample graph based on semantic and reasoning similarities. Based on this graph, we introduce latent difficulty states and use a Potts prior to encourage neighboring samples to share the same state. We then employ a state-level Beta-Binomial model to aggregate the rollout outcomes associated with each state. Finally, we use an online mean-field variational algorithm to continuously update the latent-state assignments and state-level difficulty as new feedback arrives. Our framework can be integrated into sample-selection and rollout-allocation schedulers, enabling difficulty-adaptive exploration without dedicated probing. Experiments across multiple base models, RL schedulers, and benchmarks demonstrate that our framework achieves better performance. 

---
# An Empirical Study of Reward Specification and Benchmark Reliability in GRPO-based LLM Unlearning 

**Authors**: Rubén Balbastre, Juan Manuel Orduña, Mariano Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2608.17804)  

**Abstract**: Practical LLM unlearning is usually evaluated through two objectives: suppress target-specific knowledge and preserve non-target utility. In generative QA, this leaves a third behavior underspecified: when a target-adjacent prompt admits a broader answer without target-specific leakage, the model should answer at that level rather than leak, evade, or refuse. We study this specification problem in a controlled LoRA-GRPO RWKU setting, comparing four reward designs that span lexical suppression, anti-refusal shaping, rubric-based broad answering, and an explicit refusal contrast, with and without SFT warm-up. The experiments show that optimization success is not equivalent to behavioral unlearning: RWKU forget scores, held-out completion audits, terminal training-rollout audits, and training dynamics can point to different conclusions. We trace these disagreements to reward-hacking endpoints, policy-support limits in GRPO, benchmark probes that miss endpoint changes, and rewards that can select broad-topic answering with low semantic leakage during optimization. 

---
# What Aggregate Scores Miss: Measuring Item-Level Regressions in Commercial LLM API Migrations 

**Authors**: Xiaonan Xu, Wenjing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.17719)  

**Abstract**: Context: Software systems that depend on commercial large language model APIs must migrate to successor versions when vendors deprecate older models. Migration decisions typically rely on aggregate benchmark scores, which compress heterogeneous item-level behaviour into a single net figure. Objective: We measure what that compression conceals. Method: On three pairwise upgrades in the GPT-5.4 to GPT-5.6 Sol product sequence, we query 900 public benchmark items (graduate-level knowledge, olympiad mathematics, instruction following) 50 times per item per model, classify each item as reliably improved, reliably regressed, practically equivalent, or inconclusive under false-discovery-rate control and a practical-significance threshold, and calibrate the results against a label-permutation null. Results: Across all nine migration-benchmark cells, reliable improvements and reliable regressions coexist. Edges with aggregate gains of up to 7.3 percentage points contain up to 8.3% reliably regressed items; edges with aggregate losses contain up to 10.7% reliably improved items. On the instruction-following benchmark, the gap between strict and loose scoring widens by 3.9 percentage points on the latest migration: a 3.9-point regression under strict scoring shrinks to 0.04 points under loose scoring. Conclusion: Migration decisions based on aggregate scores alone miss substantial bidirectional item-level change. The complete response-level archive and per-item scoring outputs are released. 

---
# LLM-Derived Preference Judgments Are Not Self-Consistent 

**Authors**: Matthew T. Ford, Francis Bahk, Jingjing Wang, Adam S. Jovine, Tinghan Ye, David B. Shmoys, Peter I. Frazier  

**Link**: [PDF](https://arxiv.org/pdf/2608.17644)  

**Abstract**: Agents increasingly interpret a person's natural-language preferences by querying an LLM for numerical preference judgments, e.g., by asking how much the person would be willing to pay for an item. A growing body of work estimates a utility function from these judgments and then chooses actions based on their estimated utility. This pipeline assumes the judgments are approximately self-consistent: that a single utility function can reproduce them. But are they? To study this question, we measure the self-consistency of cardinal LLM preference judgments. For example, the difference in stated willingness-to-pay between two items should match the stated payment that makes a person indifferent to exchanging them. We develop statistical tests and interpretable measures of how far observed responses depart from the best-fitting self-consistent utility function. Experiments with flight, apartment, and hotel examples across six LLMs reveal large persistent inconsistencies. This suggests that LLM-derived preference judgments cannot be faithfully summarized by a single utility function. 

---
# MoNe: Modular Neural Memory for Efficient Long Context Inference 

**Authors**: Wonguk Cho, Kyubyung Chae, Tribhuvanesh Orekondy, Sunghyun Park, Hyoungwoo Park, Jeongho Kim, Arash Behboodi, Kyuwoong Hwang, Sungrack Yun  

**Link**: [PDF](https://arxiv.org/pdf/2608.17616)  

**Abstract**: We present MoNe, a lightweight modular neural memory that attaches to any frozen pretrained Transformer to enable long-context inference without retraining. MoNe reads context in fixed-size segments via test-time learning of fast-weight neural memory networks with layer-localized gradient updates; at inference, the memory generates keys and values from the query tokens alone, with no context tokens re-read. This two-phase design decouples inference cost from context length, achieving $O(N)$ preprocessing and $O(1)$ query cost with peak GPU memory that does not grow with $N$. At 128K tokens, MoNe reduces both compute and peak GPU memory by approximately 80% compared to ICL with only 6.4% parameter overhead. MoNe generalizes to context lengths far beyond the backbone's native window, achieving strong performance on needle-in-a-haystack and word extraction benchmarks from RULER, where ICL degrades sharply. 

---
# Domain-Adapted Molecular Language Models for Efficient Search of Make-on-Demand Libraries 

**Authors**: Henrik Wille, Luis-Finley Schütz, Felix Strieth-Kalthoff  

**Link**: [PDF](https://arxiv.org/pdf/2608.17567)  

**Abstract**: Pretrained molecular language models are increasingly used as molecular encoders for learning structure-property relationships. However, their practical suitability for molecular discovery within and beyond their pretraining domain remains unclear. Herein, we systematically benchmark four molecular language models across six virtual molecular libraries spanning drug discovery, organic materials, and catalysis. Native molecular language model embeddings show substantial variation in discovery performance across libraries, whereas molecular fingerprints provide a consistently strong and robust baseline. Consistent with a potential domain-representation mismatch, we show that explicit domain adaptation substantially improves representation performance. Fine-tuning molecular language model encoders on structures from the target virtual library consistently improves sample efficiency, with several adapted encoders emerging as the top-performing representations across the benchmark tasks. These results show that molecular representation quality depends strongly on the target domain and that explicit adaptation can improve the practical utility of molecular foundation models. More broadly, our findings establish domain-adapted molecular representations as a promising strategy for sample-efficient adaptive decision making in virtual screening and self-driving laboratories. 

---
# Reflex-Guard: A Low-Latency Guardrail for LLM Prompt Safety Using Dense Semantic Embeddings 

**Authors**: Istiaque Ahmed, Afia Anjum Borsha, Ranat Das Prangon, Abu-fuad Ahmad, Thi Hong Tran  

**Link**: [PDF](https://arxiv.org/pdf/2608.17556)  

**Abstract**: Large Language Models (LLMs) in real-world applications often face the risks of specially crafted prompts designed to bypass the safety controls. Existing guardrail methods, such as LLM-as-a-judge and cloud-based safety APIs are able to detect unsafe content. However, they often add a delay of about 250-900 ms to each request. This delay is too high for real-time applications, when the system usually needs to respond in less than 100 ms. Furthermore, routing user prompts through external moderation endpoints raises significant data privacy concerns. This paper introduces Reflex-Guard, a lightweight guardrail that runs locally. It uses jailbreak-aware preprocessing, compact sentence-transformer embeddings, and seven fast binary classifiers. Together, these components enable high-accuracy prompt safety filtering with much lower latency than existing solutions. Through systematic evaluation on a strategically balanced dataset of 30,568 samples drawn from five complementary sources, we demonstrate that Reflex-Guard achieves 95.9% recall on harmful prompts at 37.6 ms end-to-end latency. It is faster than existing baselines, including Llama Guard 2 at 255 ms and SafeDecoding at 723 ms. It can detect 100% of GCG suffix attacks and Base64-encoded prompts using the default threshold. However, DrAttack structured prompts required lowering the threshold to 0.03 for optimal detection, as they produced a distinct probability distribution. Reflex-Guard achieves Reflex Efficiency Score (RES) scores up to 16.79, significantly outperforming Llama Guard 2 (11.90) and SafeDecoding (9.80). This analysis offers practical deployment advice and shows that different attack types occupy distinct regions in the embedding probability space. 

---
# Code as Representation: A Compilable Parsing Paradigm for Academic Documents 

**Authors**: Rihui Jin, Jun Wang, chengyuan zhu, Liang Mingyu, Yue Gao, Li Yunxuan, Kuicai Dong, Guilin Qi, Lin Ren, Yongrui Chen, Xinbang Dai, Jiaqi Li, Tongtong Wu, Gholamreza Haffari  

**Link**: [PDF](https://arxiv.org/pdf/2608.17550)  

**Abstract**: Academic papers are a primary carrier of scientific knowledge, yet most of this knowledge remains locked in PDFs that are optimized for human reading rather than machine use. For Multimodal Large Language Models (MLLMs), the core challenge is not only perception, but representation: scientific pages interleave text with Structured Academic Elements (SAEs) such as tables, formulas, charts, and pseudocode, whose structure, data, and logic are poorly preserved by common surrogates like Markdown. We therefore propose Compilable Academic Document Parsing (CADP), a paradigm that reconstructs a full page as contextual \LaTeX{} plus executable Python, so that structure-preserving elements and executable chart representations can be reconstructed, recompiled, and directly verified against the source page. To support this setting, we introduce CADP-Bench, an expert-verified benchmark of full academic pages containing tightly coupled text and multiple SAE types, evaluated through a re-injection compilation protocol. We further study current capabilities using SOTA MLLMs and an exploratory multi-agent baseline that incorporates common agentic techniques. Results show that even frontier models still struggle to produce high-fidelity executable reconstructions, highlighting substantial room for improvement in structure-aware scientific document parsing. CADP-Bench is released for future research. 

---
# Decomposition Attacks Across Unlinkable Identities: Limits of Stateful Defenses for LLM Services 

**Authors**: Bowen Sun, Zhengyue Zhao, Xiaogeng Liu, Yinzhi Cao, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2608.17445)  

**Abstract**: Most large language model services use stateless defenses, which judge only the current request, to refuse harmful tasks. Decomposition attacks exploit this limitation by splitting a harmful task into individually permissible requests and combining their answers. Defending against them therefore requires a stateful monitor that considers requests together. If it can group all requests for one attacker task, it can stop the attack. However, attackers can use unlinkable identities and combine answers elsewhere, leaving no reliable grouping signal. We ask whether decomposition attacks can still be stopped under this setting. For a fixed attack strategy without retries, we prove that the achievable security and utility tradeoff depends entirely on how benign requests for the same capabilities are grouped. Persistent, recognizable groups permit a useful defense; fresh, indistinguishable groups do not. When attackers can retry and learn from Allow/Block decisions, this useful operating point disappears: the feedback reveals what passes but not whether a block was correct. Experiments on 91 executable tasks and 11,393 capability-matched benign requests support these results. Under a 1% denial cap for these requests and a 0.5% cap for unrelated background traffic, all ten tested policies, including one privileged policy with an exact request-to-operation map, either fail to stop attacks or exceed the budget. On defense-unseen task families, attack success is at least 99% after one attempt and 100% after two. Effective defenses therefore require additional evidence or mechanisms tied to grouping, such as reliable identity linkage, costs for fresh identities, or control over answer use. 

---
# LLMs for Medical Consultation Are Evaluated Too Late: The Preformulation Gap 

**Authors**: Yining Hua, Cyrus Ayubcha, Hongbin Na, Levi Lian, Alon Gorenshtein, Yiftach Barash, Eyal Klang  

**Link**: [PDF](https://arxiv.org/pdf/2608.17330)  

**Abstract**: Large language models for medical consultation are often evaluated after a clinical problem has already been made clear, although real consultations may begin with a vague, minimized, or misframed concern. We evaluated three API models across four physician-authored, multi-turn vignettes under baseline and entry-to-care instruction conditions, yielding 24 fixed-script transcripts; two cases also used adaptive standardized-patient simulation, yielding 12 transcripts. Self-care or home-management advice before any patient answer appeared in 9 of 12 baseline case-model cells and 0 of 12 instruction cells, while structured handoff summaries appeared in 0 of 12 and 10 of 12 cells, respectively. The instruction changed sequencing and documentation, although it did not reliably ensure elicitation of decisive facts. The preformulation gap should therefore be evaluated directly through observable first-contact behavior rather than inferred from diagnostic accuracy or final-answer quality. 

---
# KnowSim: Evaluating Information Calibration in LLM Assistants with User Simulators that Learn 

**Authors**: Yoonjoo Lee, Hyoungwook Jin, Tae Soo Kim, Shaoyang Zhang, Philippe Laban, Q. Vera Liao  

**Link**: [PDF](https://arxiv.org/pdf/2608.17150)  

**Abstract**: To effectively collaborate with users on knowledge-intensive tasks, Large Language Models (LLMs) must perform information calibration: matching content to a user's evolving understanding and cognitive capacity. Yet user simulators used to evaluate and train LLMs do not explicitly model user knowledge so they neither produce realistic interactions across knowledge levels nor reflect how interactions unfold as that knowledge evolves. To close this gap, we introduce KNOWSIM, an evaluation framework built around a user simulator that maintains explicit knowledge states, represented as a graph of Information Units with prerequisite relationships, that evolve under update rules grounded in learning theory. KNOWSIM computes three metrics (Knowledge Gain, Delivery Calibration, Cognitive Overload) directly from the knowledge state trajectory, reflecting key mechanistic aspects of information calibration. We validate KNOWSIM against 705 human-AI sessions across two domains, stratified by knowledge level: its rankings align significantly with human judgments (73-74% sign agreement), outperforming three baseline simulators. Applied to 9 LLMs, KNOWSIM reveals that the best model shifts by user knowledge level, revealing aptitude-treatment interactions invisible to standard evaluation. 

---
# J-Miner: Recovering Executable Decision Knowledge from Language-Model Classifiers 

**Authors**: Yunfan Gao, Xinyi Huang, Tao Sheng, Haorui Song, Yun Xiong, Haofen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.17063)  

**Abstract**: Large language models can be fine-tuned into specialized classifiers that perform well across diverse text tasks and make complex judgments, but they typically expose only final labels, leaving the decision knowledge acquired through fine-tuning implicit within the model. We study how to mine this internal decision knowledge from a fine-tuned classifier and encode it in an executable representation that can be inspected, validated, and reused beyond the source classifier. We introduce J-Miner, which mines text-level named concepts by aggregating vocabulary-aligned internal signals across layers and token positions, and uses the classifier's own predictions to learn executable decision rules over them. This process distills local internal readouts into an explicit classifier-level knowledge representation. Across multiple classification tasks, J-Miner rules reproduce up to 98.3\% of source-classifier decisions and achieve 6.0--29.5 percentage points higher behavioral fidelity than equally compact rules learned from input words. Further analysis shows that the named concepts reflect internal semantic evidence associated with task decisions, while the learned rules consolidate these distributed signals into inspectable decision structures. The resulting decision knowledge also transfers to lightweight standalone students: using about 1/24 as many parameters as the source classifiers, they reconstruct and execute the representation from raw text while retaining 99.8\% of the source classifiers' mean task accuracy. These findings show that task-specific decision knowledge can be faithfully represented in an explicit, executable form and reused beyond the classifier in which it was learned. 

---
# Memory Is Communication: The Frontier Between Remembering and Signaling 

**Authors**: Yashar Talebirad, Eden Redman, Ali Parsaee, Osmar R. Zaiane  

**Link**: [PDF](https://arxiv.org/pdf/2608.17053)  

**Abstract**: A bounded agent may obtain information for a decision from its own past, from peers, or from both sources. Retaining task-relevant history can reduce later communication, while a peer message can supply what memory lacks. Under limits on both resources, how should an agent allocate its information budget? Given a fixed task and decision rule, the memory and message rate pairs attaining a performance threshold form an achievable region under specified rules for using history and peer observations. We call its efficient boundary the remembering--signaling frontier. Across conditions where history permits the same maximum reduction in task loss, we hypothesize that a bounded agent will need less peer communication when it obtains a larger loss reduction from history. In preliminary referential games, target repetition coincided with shorter successful messages, while predictability from a hidden cyclic rule did not shorten them. Experiments varying memory and message rates can estimate the frontier and test this prediction across cooperative tasks. 

---
# The Price of Thinking: Reasoning Effort as a Model-Specific API Contract 

**Authors**: Yeabin Moon  

**Link**: [PDF](https://arxiv.org/pdf/2608.16956)  

**Abstract**: API buyers purchase a dated contract, not a model name alone: the contract includes the requested and served model, reasoning-effort term or its omission, output rail, service product, prompt, and price schedule. We study the reasoning-effort term through a registered paired contrast of Sonnet 5 with explicit high effort against the same model with effort omitted, using 30 AIME 2026 items and five calls per item. Every paid attempt was assigned one frozen terminal category, and inference resampled items while retaining their repeated calls. Mean delivered cost was \$0.01031 per call higher under the explicit-high contract than under the omitted contract [+\$0.00204, +\$0.01974]. The corresponding accuracy contrast was +0.0133 [-0.0267, +0.0467]; we did not detect an accuracy difference, and the interval permits a gain of up to 4.67 percentage points that this design cannot rule out. Cost per correct answer was \$0.08665 under the high-effort contract and \$0.07662 under the omitted contract, as registered point estimates. A dated contract census, Models-API metadata, and preregistered raw-response probes further documented model-specific omission semantics, including within a provider; claims remained at documentation grade when raw structure was indeterminate. The request registry, parser, terminal taxonomy, statistical plan, and analysis pipeline were frozen before outcomes were examined; the resulting claims are bounded to the model, task, and collection date studied. 

---
# SeqFeed: Improving Agentic RTL Code Generation with Sequential Behavior Feedback 

**Authors**: Yuxin Du, Juxin Niu, Tao Hu, Xi Wang, Zhe Jiang, Nan Guan  

**Link**: [PDF](https://arxiv.org/pdf/2608.16934)  

**Abstract**: RTL code generation is a critical stage in hardware design, and the emergence of agentic systems offers new opportunities to automate this process. To generate correct RTL code, agents must understand sequential behavior, including how signals evolve and propagate over multiple clock cycles. However, effectively conveying such temporal information to agents remains a significant challenge. RTL code does not expose cycle-level signal behavior for a specific execution, whereas full simulation waveforms are too voluminous and noisy for effective LLM analysis. To address these limitations, we study how human engineers reason about sequential behavior and identify three requirements for effective feedback: it should be event-addressable, dependency-traceable, and iteratively-queryable. Guided by these requirements, we propose \textit{SeqFeed}, which comprises two complementary mechanisms: (1) \textit{SeQuery}, an SQL-like waveform query language that enables agents to anchor queries to semantic events and sample signal values at relative time points; and (2) \textit{SeGraph}, a dependency graph that tracks signal propagation across clock cycles. Experimental results across multiple LLMs demonstrate the effectiveness of SeqFeed in improving pass rates. SeQuery and SeGraph are each effective independently and provide complementary benefits when used together. 

---
# When Personalization Becomes Bias: Structural and Discursive Religious Framing in AI-Generated Financial Advice 

**Authors**: Muhammad Salar Khan, Hamza Umer, Hasan Mahmud, Sandra Rothenberg  

**Link**: [PDF](https://arxiv.org/pdf/2608.16909)  

**Abstract**: Large language models (LLMs) are increasingly integrated into financial advisory systems, yet their role in reproducing religious bias remains underexamined. This study provides systematic mixed-methods evidence of such bias across three LLMs (ChatGPT, Gemini, and Grok) using 432 simulated advisor-client interactions spanning 16 religious identity pairings (Christian, Muslim, Hindu, and non-religious) and three core household financial decisions: stock investment, house purchase, and life insurance. Combining regression and reflexive thematic analyses, we identify structural biases across models and decision contexts and the discursive mechanisms through which they are linguistically enacted. Unbiased advice appeared in only 12-18% of cases. Gemini consistently produced more bias than Grok, while ChatGPT's outputs were statistically comparable to Grok's. Religiously symmetric advisor-client pairings almost always triggered explicit religious framing, and non-religious clients often received advisor-centered religious appeals. Qualitative findings show that bias is linguistically manifested through religious anchoring, uneven cultural signaling, and tone modulation, varying by model and financial scenario. Stock investment prompts produced more financially technical responses, whereas life insurance advice triggered stronger religious language. The study develops a dual-dimensional framework linking structural bias rooted in model training and design with discursive bias expressed through language, advancing understanding of algorithmic bias in LLM-generated financial advice. It also shows that such advice adapts linguistically to identity cues, revealing a managerial dilemma between personalization and neutrality. Finally, it highlights implications for businesses, financial institutions, and regulators seeking to ensure neutrality, cultural sensitivity, and trust in AI-mediated advice. 

---
# The politics of postmortem privacy 

**Authors**: Mauricio Figueroa  

**Link**: [PDF](https://arxiv.org/pdf/2608.16905)  

**Abstract**: While the existence of postmortem privacy is increasingly acknowledged (such as the protection of the presence of deceased within digital spaces), far less attention has been paid to its internal instability: its scope (the extent of its application), justificatory foundations (why do we protect the deceased in the first place), and uneven articulation across jurisdictions (for example, some jurisdictions may tolerate or endorse practices that may be contestable in a different jurisdiction). This piece unearths the internal diversity of the concept by illuminating specific points of tension and conflict that the notion of postmortem privacy evokes. These points of tension are collectively refer to as the politics of postmortem privacy. To do so, this paper organises existing contributions of legal scholarship, placing them in dialogue with broader cultural, social, historical and political observations to illustrate the politics of postmortem privacy through three different loci of analysis: the transatlantic divide between European and American approaches, intra-European tensions within data protection governance, and postcolonial and post-authoritarian contexts in the Global South. While existing literature has glimpsed toward the former two, this piece contends that the latter deserves greater attention and inclusion in the debates around privacy and the dead. The piece explains, in continuity with existing scholarship, how postmortem privacy is assembled differently as a productive register through which societies negotiate memory and dignity, which play a great role in the governance of data of the dead and information flows. 

---
# An Investigation of the NeurIPS and ICML 2025 Position Tracks 

**Authors**: Fan Yang, Wenkai Li, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.16894)  

**Abstract**: ML venues shape what kinds of research claims become legible to reviewers and what forms of evidence count as rigorous. The NeurIPS and ICML Position Paper Tracks were created for agenda-setting work, making their early composition worth auditing. \textbf{This paper argues that the publicly accessible 2025 reviewed pool is dominated by reformist critique, and that the track should explicitly solicit direction-setting work alongside, not in place of, the reformist critiques it already hosts well.} We audit every accessible submission to the NeurIPS 2025 and ICML 2025 Position Tracks under a pre-specified rubric, and compare the resulting pattern with a reference class of widely recognized agenda-shifting ML papers. Three-quarters of audited submissions critique an existing benchmark, evaluation, or methodology; these papers score highly on our artifact-coupling rubric, but evidentiary depth does not predict reviewer rating. The reference class (AlexNet, the Transformer, Concrete Problems in AI Safety, and others) differs from the accessible reviewed pool in \emph{artifact kind}: agenda-shifting papers typically gave the field something new to build on, test against, or contest, such as a measurement protocol, benchmark proposal, toy implementation, dataset card, audit template, or falsifiable experimental program. We close with four CFP-level interventions aimed at broadening the submission mix without displacing the critiques the track already hosts well. 

---
# Grounding Healthcare LLMs in a Causal Knowledge Graph: Framework, Metrics, and a Cardiovascular Pilot 

**Authors**: Ummara Mumtaz, Aimen Noor, Awais Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2608.15382)  

**Abstract**: Large language models (LLMs) are increasingly proposed for healthcare decision support, but their evaluations still reward single-answer accuracy rather than reasoning about interventions, mechanisms, harms, evidence, and uncertainty. We propose a reproducible, graph-centered evaluation framework for intervention-oriented LLM behavior in healthcare and stress-test it in a cardiovascular pilot. The framework has four components: (i) a domain causal knowledge graph in which assertions are first-class, provenance-preserving nodes with stable identifiers; (ii) a scenario-conditioned subgraph extraction step that, given any clinical scenario, retrieves the relevant reified-assertion subgraph; (iii) four controlled grounding conditions that vary how the retrieved subgraph is composed into the model's context (ungrounded C1, knowledge-graph C2, causal-graph C3, integrated C4); and (iv) an automated scoring pipeline, anchored on assertion identifiers, that computes intervention accuracy, and other evaluation measures on a single pass. To test the framework, we built a category-balanced scenario generator across eight reasoning failure modes and instantiated it on a cardiovascular graph. The metric panel discriminates conditions along interpretable, non-redundant axes: C4 obtains the strongest causal edge F1 (0.838), adverse-effect F1 (0.833), evidence accuracy (0.738), and unsupported claim rate (0.114), while C1 obtains the highest raw intervention accuracy (0.948) with no measurable causal or evidential grounding. 

---
# Intent-Driven Dynamic Chunking: Segmenting Documents to Reflect Predicted Information Needs 

**Authors**: Christos Koutsiaris  

**Link**: [PDF](https://arxiv.org/pdf/2602.14784)  

**Abstract**: Breaking long documents into smaller segments is a fundamental challenge in information retrieval. Whether for search engines, question-answering systems, or retrieval-augmented generation (RAG), effective segmentation determines how well systems can locate and return relevant information. However, traditional methods, such as fixed-length or coherence-based segmentation, ignore user intent, leading to chunks that split answers or contain irrelevant noise. We introduce Intent-Driven Dynamic Chunking (IDC), a novel approach that uses predicted user queries to guide document segmentation. IDC leverages a Large Language Model to generate likely user intents for a document and then employs a dynamic programming algorithm to find the globally optimal chunk boundaries. This represents a novel application of DP to intent-aware segmentation that avoids greedy pitfalls. We evaluated IDC on six diverse question-answering datasets, including news articles, Wikipedia, academic papers, and technical documentation. IDC outperformed traditional chunking strategies on five datasets, improving top-1 retrieval accuracy by 5% to 67%, and matched the best baseline on the sixth. Additionally, IDC produced 40-60% fewer chunks than baseline methods while achieving 93-100% answer coverage. These results demonstrate that aligning document structure with anticipated information needs significantly boosts retrieval performance, particularly for long and heterogeneous documents. 

---
# Potential of ChatGPT in predicting stock market trends based on Twitter Sentiment Analysis 

**Authors**: Ummara Mumtaz, Summaya Mumtaz  

**Link**: [PDF](https://arxiv.org/pdf/2311.06273)  

**Abstract**: The rise of ChatGPT has brought a notable shift to the AI sector, with its exceptional conversational skills and deep grasp of language. Recognizing its value across different areas, our study investigates ChatGPT's capacity to predict stock market movements using only social media tweets and sentiment analysis. We aim to see if ChatGPT can tap into the vast sentiment data on platforms like Twitter to offer insightful predictions about stock trends. We focus on determining if a tweet has a positive, negative, or neutral effect on two big tech giants Microsoft and Google's stock value. Our findings highlight a positive link between ChatGPT's evaluations and the following days stock results for both tech companies. This research enriches our view on ChatGPT's adaptability and emphasizes the growing importance of AI in shaping financial market forecasts. 

---
