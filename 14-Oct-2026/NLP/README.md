# Tev: Scaling Long-Horizon Agentic Environments with Dense Progress Rewards 

**Authors**: Wenbo Hu, Shraman Pramanick, Krishna Kumar Singh, Nanyun Peng, Kai-Wei Chang, Yong Jae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.18868)  

**Abstract**: Long-horizon agents have drawn growing attention for their ability to work autonomously -- from hours to days -- on economically valuable tasks, a capability essential to automated discovery and recursive self-improvement (RSI). Yet open-source models still struggle on long-horizon tasks, largely for lack of both training data and a working recipe, and scaling both is hard: long, complex tasks are difficult to curate and verify, and their rewards are sparse. We address both. First, we introduce Tev-Gym, the first long-horizon agent training dataset with verifiable, dense, progressive rewards -- 1,170 verified task environments across 7 domains. It is built on a latent-world-first approach that first constructs a latent world: a complete and correct hidden specification of the task domain, workflow, and expected state after each step. Each task is then progressively rendered from this specification into an agent-facing environment, enabling controllable complexity and a natural progress metric. Second, we turn this metric into a training recipe: via progress-potential reward shaping (PPRS), we convert progress into a dense per-step reward that improves training efficiency and makes otherwise-intractable tasks trainable. Training with this recipe yields Tev-27B, which achieves substantial gains on the Agents' Last Exam computing and math sciences domains (ALE-CM), matching GPT-6 Astra and Claude Opus 5 at Pass@5. Despite training only on ALE-related data, it generalizes to the Terminal-Bench (TB) 3.0 and 4.0 subsets, surpassing GLM 5.2 and Gemini 3.7 Flash, and reaches Claude Opus 4.8-level performance on middle-horizon tasks such as TB 2.1. 

---
# DoubtLess: Training-Free Contrastive Decoding for Efficient Reasoning 

**Authors**: Minghao Luo, Liang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.18862)  

**Abstract**: Reasoning models achieve strong performance through long chains of thought (CoT), but often spend more tokens than needed to solve a problem. Existing ways to shorten the CoT require additional training for each model, add a fixed steering vector to the model's hidden states, or use decoding heuristics such as stopping early or banning words like "wait", which can reduce accuracy. We propose DoubtLess, a training-free decoding method in which a much smaller reasoning model steers the large one towards brevity. The small model reads the large model's CoT so far twice, once prompted to be brief (the brief thinker) and once prompted to be verbose (the verbose thinker), and the difference between its two logit vectors, scaled by a strength, is added to the large model's logits at every step of the CoT. On seven benchmarks of competition mathematics, graduate-level science and competitive programming, DoubtLess shortens the CoT of Qwen3-32B by 31% with unchanged accuracy, outperforming five inference-time baselines. We also show that DoubtLess reduces thinking tokens per turn by 20% in a self-debugging coding agent with unchanged task performance. It also generalizes to two other model families with unchanged accuracy. When combined with training-based compression, it further shortens the CoT of a model already trained for brevity. Ablations and behavioral analysis show that DoubtLess reduces repeated checking, with most token savings coming after the correct answer first appears in the CoT. Our code is available at this https URL. 

---
# Hybrid Reasoning Agents 

**Authors**: Jiaru Zou, Yunzhe Qi, Sirui Chen, Federico Bianchi, Shang Zhu, Pan Lu, Jingrui He, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2610.18861)  

**Abstract**: How can we build an agent that thinks beyond human language, yet still acts through it in the external environment? Modern LLM agents still reason and interact through text, while latent reasoning offers an efficient internal medium but lacks native tool access and reasoning transparency. We bridge these two directions with Hybrid Reasoning Agents (HybridAgents), which dynamically navigate between latent reasoning and text-based interaction throughout a unified generation process. Architecturally, a lightweight HybridLink enables seamless transitions between text and latent modes, while the agent's continuous working memory preserves lossless hybrid context across multiple turns. These designs preserve tool-integrated agentic capabilities while making latent thoughts directly inspectable. Building on this architecture, we further develop dedicated post-training algorithms for hybrid reasoning: HybridSFT stabilizes tool-integrated hybrid generation by teaching the backbone to coordinate with HybridLink, then HybridRL scales reasoning by learning when to think in latent space and when to act through text. We evaluate HybridAgent at 3 model scales across 11 benchmarks covering mathematics, science, medicine, search, and agentic reasoning. Compared with frontier LLM Agents and agentic frameworks spanning both text-based and latent methods, HybridAgent consistently improves performance by 10.4%, while delivering 35.5%-54.7% token reduction, 1.8x-2.6x faster inference, and 2.7x-3.7x cost savings. Code and data are available at this https URL. 

---
# Back-Translation Still Matters: Leveraging Monolingual Data for Low-Resource Machine Translation with LLMs 

**Authors**: Divyanshu Bathla, Parag Singla, Jitin Singla  

**Link**: [PDF](https://arxiv.org/pdf/2610.18818)  

**Abstract**: Pre-trained Large Language Model (LLMs) are effective on many tasks in zero-shot and few-shot settings. For machine translation of low-resource languages, the obvious options are few-shot prompting and fine-tuning on a small parallel set. We ask: what if moderate amounts of low-resource monolingual data are also available? We argue that, rather than continued pretraining, a more effective use of this data is the pre-LLM era technique of back-translation (BT). The current fine-tuned LLM translates monolingual sentences in each language, and the resulting pairs train it in the reverse direction. Updates on the parallel set are interleaved for stable learning. With 1K parallel pairs on Odia and Swahili, BT beats continued pretraining into the low-resource language by 7.7 to 16.4 XCOMET-XXL points in domain. With beam-decoded intermediate translations, it also beats a model with 10 time more parallel data. In-context examples retrieved from the parallel data further improve BT. We also analyze the source, quality and amount of monolingual text. Across five low-resource languages, with Hindi as a high-resource control, and two 3B backbones, BT improves translation into the low-resource language by 16 to 173% XCOMET-XXL over parallel-only fine-tuning. Blind human evaluation on Odia and Sanskrit agrees with the automatic metrics. 

---
# Where Do LLM Ensemble Gains Come From? Decomposing Candidate Value, Diversity, and Aggregation 

**Authors**: Donghwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.18786)  

**Abstract**: LLM ensemble gains are often measured against a single baseline, but the value attributed to additional candidates can change with the comparison used. We study an LLM aggregator under four matched conditions: Direct (no candidates), Single-slot (one candidate in a multi-slot interface), Duplicate (one candidate repeated), and Multi-candidate (separately generated candidates). Under either non-diverse control, overall candidate gain (CIV) splits algebraically into an additional-candidate contrast (DSV) and a candidate-context contrast (CCV); the split, but not CIV, depends on the chosen control. On MultiHopRAG model diversity (k=2), the repetition gap Rep=-8.33 pp [-13.25, -3.75] quantifies the difference between gains measured against Duplicate and Single-slot. On a matched FRAMES stronger-panel extension (n=200, k=3, aggregator output limit 4096), paired Delta Rep=+9.67 pp [+6.00, +13.50]; within panels, Rep changes sign (original -6.33; stronger +3.33). In FRAMES evidence aggregation, candidate-context contrasts remain strongly negative under both non-diverse controls. Our results show why ensemble evaluations should state the reference control before attributing gains to additional candidates. 

---
# Rubrics from the Task, Verdicts from Evidence: Pointwise Rubric Rewards with Tool-Augmented Judges 

**Authors**: Tianci Liu, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.18766)  

**Abstract**: Rubric-based reward models (RMs) evaluate responses against explicit criteria. However, existing methods often derive criteria from observed response candidates, introducing response-induced bias that may limit rubric generalizability. Their judges also struggle with mechanical constraints when relying on reading alone, and commonly used pairwise judges provide only relative preferences, complicating their direct use for policy optimization. We propose ReVeR, a pointwise rubric RM that derives hard rules and graded principles by reasoning about the instruction alone and evaluates responses independently through tool-augmented judging. Both components are trained through label-blind distillation, in which preference labels only filter a teacher's outputs, yielding ReVeR-Traces, a large-scale dataset of rubric-reasoning and judging trajectories. Experiments demonstrate that ReVeR outperforms pairwise rubric RMs in average preference accuracy, most clearly on instruction following, as well as in test-time scaling and downstream policy optimization. We further find that generator RL improves preference accuracy while reducing rubric fidelity, and that higher preference accuracy does not necessarily translate into better downstream policies. These findings highlight that rubric fidelity, preference accuracy, and downstream utility are distinct properties, shedding light on future RM research. 

---
# Looking Complete Is Not Being Grounded: Evaluating LLM Sitreps Against Human-Authored Situation Reports 

**Authors**: Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2610.18762)  

**Abstract**: Evaluation for automatic crisis reporting should measure what practitioners need, not what is convenient to compare. Through LLM-assisted, author-approved content analysis, we derive a 15-slot information schema from professional humanitarian situation reports using qualitative content analysis, then evaluate 121 human and 261 LLM-generated reports across five disasters and three hazard types. Under google/gemini-3.7-flash on Cyclone Idai, the best machine arm is nominally above the empirical human ceiling (0.827 vs. 0.821), but the difference is statistically indistinguishable from zero (95\% bootstrap CI $[-.049,.059]$; permutation $p=.852$), and the ordering does not replicate under openai/gpt-5.6-luna. On a blind 100-item Idai sample, the author agrees with the Gemini judge at 0.910 accuracy and unweighted Cohen's $\kappa=.799$. All four judges clear the plan's raw-agreement criterion of .7; only Gemini reaches .7 on chance-corrected $\kappa$. Across the three primary events, machine reports reproduce 5.4-11.0% of same-day official figures. Across those same events, raw exact-number matching links 14.3-19.6% of official figures and 85.4-90.1% of machine figures to the social stream; manual audit confirms 45/50 declared absences, 4/25 official matches, and 24/25 machine matches semantically. The issuing-provenance gap is positive in all 12 observed event-judge cells: report date, organisation, series, and authority were not supplied at professional-report coverage in any cell. On the three primary events, ROUGE-L is uninformative, BERTScore is negatively associated with completeness after multiplicity correction, and a generic LLM judge changes sign. 

---
# Code-Switching Detection via Language-Selective Dimensions in Speech Foundation Models 

**Authors**: Jie Chi, Zakaria Aldeneh, Masha Fedzechkina, Maureen de Seyssel, Natalie Schluter  

**Link**: [PDF](https://arxiv.org/pdf/2610.18726)  

**Abstract**: Code-switching (CS), the alternation between languages within an utterance, is common in multilingual speech but underrepresented in pretraining corpora. Increasing CS coverage requires finding relevant utterances in large speech collections so that they can be included in training data. One way to find such utterances is to transcribe the speech and detect language alternation in the text, but this depends on obtaining accurate multilingual transcriptions. We propose a lightweight method that detects CS directly from speech. It uses frame-level language information in pretrained speech encoders to determine whether both languages are present within an utterance. Specifically, we identify language-selective hidden-state dimensions using monolingual recordings and aggregate their frame-level activations into a CS prediction. With a small CS set used to tune a decision rule, we show that the Whisper-small encoder achieves a mean F1 of $0.798$ across 14 language pairs in CS-FLEURS, compared with $0.776$ for a CNN and $0.751$ for a HuBERT linear baseline trained for monolingual language identification. Our findings suggest that pretrained speech encoders already encode the information needed to identify CS utterances, and they provide a lightweight tool for building more representative multilingual corpora. 

---
# From Relation-Specific Heads to Shared Interventions: Contextual Entrainment in Language Models 

**Authors**: Jinghan Zheng, Jiajun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2610.18674)  

**Abstract**: Large language models rely on prompt context to perform in-context learning, follow formats, and adapt to local evidence. This ability, however, also raises a robustness concern: models may reuse irrelevant tokens from earlier context when answering a later query. Prior work identifies this behavior as contextual entrainment and finds attention heads whose ablation reduces it, arguing that these heads are largely task- or relation-specific. But are the mechanisms behind contextual entrainment truly isolated by relation, or do relation-conditioned searches recover reusable components shared across relations? Using factual-distractor relation-completion prompts across 19 lexical-semantic subtypes, we identify attention heads whose ablation reduces entrainment and test how the resulting head sets transfer recovery ability across exact relations, semantic families, and mixed relation distributions. Same-family transfer exceeds different-family transfer, family residuals weaken once globally shared heads are removed, and subtype-specific searches repeatedly rediscover many global heads. Clean-completion and context-use probes further suggest that this shared intervention can reduce harmful copying while preserving ordinary relation completion and simple in-context abilities. Our findings provide a more precise perspective on contextual entrainment and its mitigation: harmful context following is not best addressed by searching for isolated relation-specific circuits, but by using heterogeneous attack prompts to reveal reusable intervention targets whose effects are shaped by semantic relation structure. 

---
# SMI: Ontology-Grounded Semantic IDs for Hierarchy-Aware Medical Concept Representation 

**Authors**: Ziyang Song, Lia Shen, Yixuan Li, Qincheng Lu, Weihao Li, Haoran Xu, He Zhu, Xinyu Wang, Yue Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.18672)  

**Abstract**: Biomedical language models (LMs) provide strong representations for clinical tasks, but they often fail to capture the hierarchical structure among medical concepts. This limitation is especially pronounced for rare and fine-grained clinical codes in long-tailed vocabularies, where hierarchy-agnostic LM representations become less informative as concept frequency decreases. We introduce Semantic Medical Identifier (SMI), a training-free framework for constructing ontology-grounded Semantic IDs and hierarchy-aware medical concept representations. For each diagnosis, procedure, or medication concept, SMI defines its Semantic ID as the coarse-to-fine path through an expert-defined medical ontology. SMI then aggregates frozen biomedical LM embeddings across the corresponding ontology levels to form hierarchy-aware medical concept representations that preserve coarse-to-fine clinical semantics. The base SMI uses uniform aggregation across ontology levels, while a level-weighted variant selects global level weights on validation data. Our theoretical analysis shows that SMI yields more informative medical concept representations than hierarchy-agnostic LM embeddings, particularly for rare and fine-grained clinical concepts. Experiments show that SMI captures meaningful hierarchical structure and improves clinical relation prediction. On MIMIC-IV, SMI improves mortality prediction, readmission prediction, drug recommendation, and rare drug prediction. On eICU, SMI yields more robust patient representations under cross-hospital distribution shift. 

---
# Known-Voice Cloning and the Limits of Personality Transfer for Character Design 

**Authors**: Johanna Magdalena Kuch, Maud van Lier, Daksitha Senel Withanage Don, Patrick Gebhard, Tanja Schneeberger, Elisabeth André  

**Link**: [PDF](https://arxiv.org/pdf/2610.18669)  

**Abstract**: Recent advances in zero-shot voice cloning systems make it possible to reproduce a recognizable voice from a short recording and attach it to arbitrary virtual characters. Prior work suggests that familiar vocal identity can carry over associated personality impressions to new targets such as robots or virtual characters. However, these works have mainly examined familiarity and personalization effects rather than directly testing whether a cloned familiar voice systematically transfers personality impressions to an embodied virtual character. Such transference raises the concern that cloned voices could be used to manipulate people into engaging with virtual characters as capable social actors. Across two studies, we investigated whether personality impressions transfer when a virtual character speaks with the cloned voice of a familiar person. Study 1 examined this question in a real-world university setting using the cloned voice of a lecturer familiar to the participating students. Study 2 examined the same question online using voices of social media content creators and members of their communities. Drawing on the concepts of identity activation, trait transfer, and person reconstruction, we show that evidence of personality transfer was limited in both studies. Observed effects were selective and did not generalize across traits or study contexts. In the online study in particular, recognizable voice identity cues activated expectations about the speaker but did not reconstruct the broader person impression. Qualitative responses further indicated that judgments were frequently shaped by perceived artificiality and by whether voice and appearance fit together. Our findings moderate common concerns that cloned voices straightforwardly manipulate people into perceiving artificial characters as capable social actors. 

---
# Breakdown of Conversational Involvement: A Joint Multi-Modal Analysis of Conversation Floors and F-formation Detection with Multiple Features 

**Authors**: Zonghuan Li, Chirag Raman, Hayley Hung  

**Link**: [PDF](https://arxiv.org/pdf/2610.18627)  

**Abstract**: Interactions happen universally and spontaneously in daily lives. Studying involvement in interactions helps us analyze who can influence who and has a wide potential in applications. Specifically, we consider conversational involvement in free-standing conversations. We study two operationalizations of an FCG: F-formations and conversation floors, and propose that current assumption that F-formations capture single conversations is insufficient. To illustrate how different implementations could impact the detected F-formations, we employ three analyses. First, we propose that speaking status can be used for estimating conversation floors, and leveraged it to analyze floors within F-formations. Second, we analyze spatial characteristics of the detection results by applying different feature sets on chosen baselines, and find that groups resulting from head features are more likely to be subsumed by groups detected using other features. Finally, we do a joint analysis which combines the spatial features and floor patterns in the conversation. We observe that certain pipelines can detect groups with single floors from upper body features, while retaining F-formations with lower body features. We conclude that a more detailed theory and framework is needed to holistically demonstrate conversational involvement within interactions. 

---
# Knowledge Graphs as Files: An Agent Harness Where Tool Creation and Tool Use Co-Evolve 

**Authors**: Yuyang Bai, Yu Wang, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.18616)  

**Abstract**: Searching text-rich knowledge graphs requires following relations while examining textual evidence. Predefined graph interfaces constrain what evidence agents can access, even when they write code to combine tool calls or process retrieved subgraphs. We propose direct knowledge graph interaction (DKGI), a harness in which language model agents create graph access tools directly over complete node and edge files. The agent writes Python code to follow relations, combine node sets, and select textual evidence, interleaving execution with shell search and file inspection. This lets agents process large node sets in bulk while returning selected text for the model to assess. Execution feedback guides tool creation and revision, while the evolving tools determine what evidence the agent gathers next. On benchmark datasets spanning three domains, training-free DKGI exceeds the strongest published baseline by at least 6.7 Hit@1 points on every graph, even with a minimal harness. Controlled ablations show the benefits of multi-turn programming over shell-only access and predefined graph tools, while trajectories show agents revising code to extend searches or address errors. These findings highlight the value of allowing tool creation and tool use to co-evolve during graph search. Our code and data are available at this https URL. 

---
# Legal-R1: Agentic Retrieval for Evidence-Based Legal Reasoning 

**Authors**: Huaiwen Pang, Jiale Han, Junying Chen, Wenjun Xiao, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.18611)  

**Abstract**: Legal question answering requires answers that are not only plausible, but also grounded in legally valid and citable evidence. Standard retrieval-augmented generation often relies on one-shot semantic matching, which is insufficient when legal evidence must be selected according to authority, applicability, and diversity across heterogeneous sources. We introduce Legal-R1, an agentic retrieval framework for evidence-based legal reasoning. Legal-R1 decomposes legal reasoning into a multi-agent workflow, where a Lawyer agent learns when to retrieve, which legal source to query, what evidence to seek, and when the collected evidence is sufficient. A Legal Assistant agent retrieves from normative legal sources and precedent cases with authority-aware and diversity-aware selection, while a Judge agent generates the final answer from the complete evidence trajectory. We post-train only the Lawyer agent with masked supervised fine-tuning and trajectory-aware reinforcement learning. Experiments on multiple Chinese legal benchmarks show that Legal-R1 improves over general-purpose, legal-specific, and retrieval-augmented baselines. 

---
# Jev-ToM: Evaluating Theory of Mind in Typed Decision Models 

**Authors**: Kawai Chung, Chunkit Chan, Yuxuan Liu, Yauwai Yim, Xi Yang, Wenbin Hu, Huihao Jing, Chiu Yeung Tang, Changxuan Fan, Ki Sen Hung, Tsun On Kwok, Haoran Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.18610)  

**Abstract**: Dialogue and collaborative assistants repeatedly judge what a partner knows, believes, or intends. Typed decision models offer a potential alternative to generative large language models for bounded Theory of Mind (ToM) selection. We present Jev-ToM, comparing five typed decision models and seven generative baselines on 11,117 English text questions from nine ToM benchmarks across six capability groups. TypeSafe Jev reaches 73.2% source-balanced accuracy including native controls and ranks first or second across all six capabilities, leading beliefs, goals/desires/intentions, and emotions. Other typed models show distinct strengths across tasks. Separate resource comparisons suggest that Jev could reduce the cost and waiting time of repeated ToM selection. Confidence-based selection remains benchmark-dependent, and supplied reasons can repair errors or overturn correct answers depending on their attachment. Candidate expansion reveals distinct selection and output-validity problems. Related-question cues account for most of Jev's downstream gains, with task-dependent effects from predicted states. The findings motivate testing Jev as a ToM selection component in assistants, with task-specific confidence calibration, candidate design, and output validation as integration priorities. 

---
# From Hindsight to Foresight: Dissecting Clinical Evidence Acquisition in Large Language Models 

**Authors**: Ziyang Ling  

**Link**: [PDF](https://arxiv.org/pdf/2610.18605)  

**Abstract**: Large language models (LLMs) often diagnose full sight, static clinical cases well yet struggle in foresight, dynamic diagnosis. Prior work has improved this limitation by quantifying the information models acquire and by developing stronger follow-up strategies to improve diagnostic accuracy. In contrast, we introduce a distinctive hindsight to systematically dissect why LLMs fail to acquire diagnostically critical evidence, tracing the failure from evidence identification, utilization and downstream diagnostic updating through not only diagnostic accuracy, but also evidence coverage, trajectory dynamics, and elicited probability and information entropy. Starting experiments from the breakdown at the level of capabilities, to the diagnosis of the pre-event state before the evidence arrives, and to the post-event state transition after the evidence arrives, our research has revealed the profound limitations and internal causes existing in the dynamic diagnostic process. At the same time, we have also conducted sufficient verification and generalization experiments to prove that these limitations are widespread. Together, these findings shift the central challenge from whether LLMs possess relevant clinical knowledge to whether they can identify, acquire, and effectively use the right evidence under clinical diagnosis and conduct a more detailed mechanism analysis to identify the exact location of the capability gap. 

---
# Decomposing and Predicting Annotation Disagreement in Harmful Content Across Languages 

**Authors**: Irem Ecem Yelkanat, Faeze Ghorbanpour, Alexander Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2610.18588)  

**Abstract**: Annotator disagreement is increasingly recognized as a meaningful property of subjective tasks, rather than merely noise to be removed. Predicting which items are likely to be contested could help allocate annotation effort and route difficult cases to human review. We propose predicting disagreement from interpretable semantic properties of the text. An LLM answers a set of binary questions covering harmful-content characteristics, disagreement-related cues such as borderline severity and missing context, and language-sensitive cues such as local slurs. A classifier predicts disagreement from these answers, and we analyze which questions contribute most to the predictions. Across four datasets in English, German, and Turkish and three open-weight LLMs, our structured question features outperform direct prompting by 10.5 ROC-AUC points and the strongest text-based baselines by 5.0 points. A compact subset of eight questions performs within 1.5 ROC-AUC points of the full twenty-question set. In leave-one-dataset-out evaluation, models retain roughly 90-95% of their in-domain ROC-AUC. Across models and datasets, questions about derogation, missing context, and decision boundaries serve as the strongest predictors, suggesting that what makes harmful content contested is partly observable in the text. 

---
# The Many Morals of AI: The cautionary bias of North American news narratives in global perspective 

**Authors**: Andrew Piper, Edie Pearman  

**Link**: [PDF](https://arxiv.org/pdf/2610.18573)  

**Abstract**: The stories told about artificial intelligence (AI) play a central role in shaping public understanding, policy debates, and expectations about technological change. News media are especially influential in this process. Using a cross-regional corpus of 12,924 English-language news articles about AI published since 2022, we examine how storytelling about AI varies across media systems in four regions: North America, China, India, and the Middle East. Leveraging large language models, we extract the implicit narrative lessons conveyed by each article---commonly known as story morals---capturing value-laden interpretations that extend beyond thematic content or sentiment. We find that North American AI coverage is uniquely characterized by narratives of caution, while under-emphasizing narratives of innovation and collaboration that are more prevalent elsewhere. However, this pattern is largely explained by a broader regional negativity bias in North American news sentiment, rather than by heightened concern about AI itself. Our findings bring into view the culturally conditioned nature of AI news narratives. 

---
# CREATE: Creativity Evaluation of VLMs using the Visual Remote Associates Test 

**Authors**: Ziqing Yang, Anubhav Jangra, Smaranda Muresan  

**Link**: [PDF](https://arxiv.org/pdf/2610.18569)  

**Abstract**: Vision language models (VLMs) are used in creative tasks such as visual storytelling, design ideation, art interpretation, and advertising. Yet we lack principled ways to measure the lateral reasoning of these models compared to human potential. We introduce CREATE, a semi-automatically constructed benchmark containing 155 visual Remote Associates Test problems. These problems require a solver to find an underlying connection among three seemingly dissimilar images. We conduct a human study (N=25) to validate and ground the quality of our benchmark, and evaluate 11 VLMs. Our findings show that most VLMs underperform humans on this task. Our experiments indicate that the associative ability of VLMs depends on the cue concepts provided in the text and does not universally extend to visual puzzles. 

---
# Empathy Is Not the Scapegoat for Declines in Factuality and Reliability 

**Authors**: Letian Shi, Kanwal Mehreen, Deheng Zhang, Yuxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.18541)  

**Abstract**: Language models are increasingly expected to support users in stressful or uncertain situations, where warmth and empathy are desirable. While recent work associates warmer models with reduced reliability, we show that this conclusion is incomplete. Prior studies often train primarily on empathetic dialogue, making it difficult to separate tone effects from SFT-induced distribution shift. We revisit this issue at both the model and evaluation levels. We study prompting, SFT, and RL for inducing warm behavior, and comprehensively evaluate capability preservation using an 11-dataset benchmark suite and EMOFACT under emotional contexts. Across multiple model families, we find that warmth effects are task- and model-dependent, while task-aligned adaptive tone can improve empathy and targeted reliability, although some capability trade-offs remain. We further introduce human-validated empathy evaluators specifically developed to assess empathetic model responses. Our results suggest that models can be more supportive in social interactions while remaining reliable. 

---
# How Reliable Are System-One Decisions in Medicine? A Comparative Study of Jev, Jev-like Models, and LLMs 

**Authors**: Yanrui Du, Danyang Zhao, Sendong Zhao, Wenhao Li, Yibo Gao, Haochun Wang, Zhuoer Xu, Shiwen Cui, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2610.18529)  

**Abstract**: Jev has attracted considerable attention for its rapid, generalizable decision-making, yet its medical reliability remains under explored. To address this gap, our study compares various types of system-one decision models across 15 medical datasets comprising 8,742 instances under a unified protocol. Beyond accuracy, we assess decision stability and the ability to recognize unreliable decisions. Our results show that Jev rivals leading models on primarily text-driven tasks but is less effective on numerically driven tasks, revealing a distinct capability boundary. Under question-format variations, Jev demonstrates strong robustness, achieving the lowest decision reversal rate and relatively small accuracy fluctuations. Among evaluated open-source alternatives, only StartLux exhibits comparable robustness. But these strengths do not extend to reliable abstention awareness: most evaluated models show limitations in both recognizing missing correct answers and using optional abstention. Moreover, we find that Jev's probability-based answer selection can be improved by changing how decisions are scored: normalizing option-wise binary scores yields higher retained accuracy than choice scoring. Taken together, our findings provide early guidance for improving and applying decision models in medicine. 

---
# Retrieval Disagreement as a Signal: Diagnosing and Steering Multilingual Dense Embedders 

**Authors**: Faeze Ghorbanpour, Shaghayegh Kolli, Jana Diesner, Alexander Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2610.18526)  

**Abstract**: Multilingual dense retrievers place all languages in one shared space so that a question finds relevant documents, regardless of which language they are written in. We show that the same question, asked in different languages, retrieves substantially different documents, and this disagreement is not because of weak language understanding. Each language performs well on its own and ranks candidates consistently with the other languages; divergence appears only when languages share an index. We trace the mechanism: each language keeps its own ranking, but in the shared index, its correct answers are pushed far down, buried under non-relevant documents that are either in the query's own language or in English, depending on the model. We introduce disagreement-guided steering: for each language, we contrast the documents that other languages agree on for a question against those that language retrieves idiosyncratically, and take the difference of their representations as a language-specific direction, added to that language's queries at inference. Against baselines of language-mean removal, subspace removal, and per-language rotation, it most strongly reduces same-language crowding while remaining competitive on recall, trading off against mean removal, which better preserves ranking quality. 

---
# The Geometry of Language Model Logits: Generic Structure and Learned Departures 

**Authors**: Swathi Shree Narashiman, Aravind Ramana Venkatramanan, Nambi Seshadri  

**Link**: [PDF](https://arxiv.org/pdf/2610.18520)  

**Abstract**: Autoregressive language models produce, at every prediction step, a vocabulary-sized logit vector whose structure determines the conditional next-token distribution. We empirically identify that the ordered vocabulary-head logits follow a remarkably consistent log-linear relation
\[ z_{(r)} \approx \mu - \alpha \log r. \] across prediction contexts, architectures, and scales, emerging progressively during pretraining. To our knowledge, this is the first systematic analysis of the logit geometry underlying the distributional assumptions used in the prior decoding work. While controlled order-statistic baselines spanning bounded, light-tailed, and heavier-tailed distributions can largely explain the observed log-linear geometry, they do not explain the systematic shift in the slope $\alpha$: trained language models exhibit $\alpha \approx 1.4$--$1.7$, sitting well above the initialization baseline ($\alpha \approx 0.2$) . Moreover, $\alpha$ correlates with predictive performance, suggesting that it captures learned structure beyond generic extreme-value effects. 

---
# Imprinting: Knowledge Distillation Can Shift Student Behavior 

**Authors**: Yu-Chien Tang, An-Zi Yen  

**Link**: [PDF](https://arxiv.org/pdf/2610.18517)  

**Abstract**: Knowledge distillation trains student models on teacher-generated responses to transfer task capability, but those responses may also carry behavioral tendencies of their source. We investigate whether the source teacher can shape student behavior beyond the capability that distillation is intended to transfer. We construct harm-accepting and harm-avoiding variants of the same base teacher, use them to generate responses to ordinary reasoning and knowledge questions unrelated to safety, and distill those responses into students with different base models. Students generally acquire task capability while their downstream decisions shift toward the source teacher's safety-pragmatism disposition, a phenomenon we call imprinting. Distillation from the harm-accepting teacher reduces harm avoidance, whereas distillation from the harm-avoiding teacher increases it. Imprinting varies across teacher-student pairs and does not extend uniformly across different forms of safety behavior. Excluding a sparse set of divergence tokens reduces imprinting, whereas response rewriting preserves the source-dependent effect and can introduce additional rewriter-dependent shifts. Post-distillation preference optimization raises harm avoidance but does not eliminate the source-dependent difference. These results show that behavioral tendencies can be intentionally imprinted through the source teacher, even with ordinary task supervision. 

---
# The Last Mile of Coding Agents for Long-Document Tasks: An Empirical Study on Clinical Trial Protocol Digitization 

**Authors**: Jiacheng Lin, Geoffrey Low, Nishit Agarwal, Rahul Jain, Afrah Shafquat, Mandis Beigi, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.18507)  

**Abstract**: Clinical trial protocols serve as the source of truth for study design, but are mostly written and used as unstructured PDF documents. Digitizing protocols turns static content into structured, machine-readable data that clinical software can analyze, validate, and execute. Coding agents, which select tools, execute code, and revise their approach as they work, are increasingly applied to such long-document tasks, yet it remains unclear how reliably they turn complete documents into accurate structured outputs. We study this question using protocol digitization as a testbed. We introduce TrialPD-Bench, an expert-annotated benchmark of 150 publicly available protocol PDFs spanning four task categories. Beyond final-answer performance, our evaluation measures evidence recall from agent traces, agent behavior, and cost. Across nine proprietary and open-weight models in five coding-agent harnesses, even the strongest agents do not fully solve protocol digitization. Our analysis further shows that agents usually locate the required evidence but fail in the last mile of transcribing and structuring it. We further develop TrialScribe, a domain-specific harness that equips coding agents with task skills and a verification loop. TrialScribe improves a locally served open-weight model by 6.2-59.4 points across eight metrics, enabling it to match or surpass proprietary models such as GPT-5.6-Sol. 

---
# Aligning Faithfulness for Theory-Scale Autoformalization 

**Authors**: Yuming Feng, Tadd Mao, Dhruva Arekar, Tianjun Zhong, Jiani Huang, Xujie Si, Ziyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.18498)  

**Abstract**: Autoformalization (AF) requires more than compilation: formal statements must preserve the meaning of their informal sources, a property known as faithfulness. An emerging frontier is the automated evaluation of this property, yet most existing methods operate at the statement level and produce whole-item verdicts, failing to reveal how individual expressions correspond or which contextual interpretations support the judgment. This omission is consequential because AF is inherently theory-scale: target statements derive their meaning from surrounding mathematical material and unstated conventions. We formulate faithfulness evaluation as a context-aware align-then-label problem: identifying phrase-level correspondences between informal and formal statements and assigning a faithfulness label to each under explicitly specified contextual interpretations and assumptions. We also introduce FaithAlign, the first theory-scale faithfulness benchmark comprising 460 items and 4,982 annotated phrase-level links, alongside FaithAlign-Gap, a companion set of 146 items with informal context and assumptions withheld. Each FaithAlign item includes the target statement pair, relevant context, formal dependencies, and, where needed, assumptions specifying one given interpretation. Across 8 models, GPT-6 Astra achieves the highest defect F1 at 47.8% and identifies all defects in only 4.5% of items with two or more defects. On FaithAlign-Gap, Astra identifies only 28.2% of undecidable items as such. These results show the difficulty of recovering context-dependent correspondences, establishing FaithAlign as a testbed for fine-grained, context-aware faithfulness evaluation. 

---
# Copying Is Not Grounding: Training Faithful Long-Form RAG with Authored-Sentence Credit 

**Authors**: Hoang Pham, Duc Anh Vu, Duy Tung Doan, Cong-Duy T Nguyen, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2610.18442)  

**Abstract**: In search engines and assistants, long-form RAG answers can mislead readers with claims their sources do not support. Faithfulness rewards, the share of supported sentences, train models toward fewer such claims. We show that these rewards are hacked by copying. A sentence copied from the passages is supported almost by construction, so trained models learn to copy rather than ground what they write. Common defenses only move the hack: a KL or copy penalty leaves copying in place or trades it for shorter, emptier answers and refusals. GRPO, the standard RL algorithm, makes it worse by giving every sentence its answer's advantage, reinforcing copied and grounded sentences alike. We propose Authored-Sentence Credit (ASC) , which fixes both by moving GRPO's reward and advantage from the answer to the sentence level. The reward credits only sentences that are authored rather than copied, say more than the question, and are supported by the passages. The advantage compares each sentence with every sentence sampled for the question, so one that earns no credit ranks last and is never reinforced. Across four models and two in-domain, three held-out corpora, only ASC lowers the error rate of authored sentences and writes more supported ones in every setting, by 38% and 46% at the median under held-out evaluators, giving readers more grounded information. It takes none of these shortcuts, and two LLM judges comparing whole answers prefer no baseline to the base model but rate ASC's answers on par with it. We conclude that a faithfulness reward should credit what a model writes, not what it copies. 

---
# Estimating Surprisal from the Speech Signal 

**Authors**: Zachary Hopton, Samuel Kiegeland, Francesco Ignazio Re, Patrick Haller, Matthew B. Winn, Ryan Cotterell, Eleanor Chodroff  

**Link**: [PDF](https://arxiv.org/pdf/2610.18439)  

**Abstract**: Although language is predominantly spoken, empirical tests of surprisal theory rely almost entirely on written text. In practice, surprisal is estimated using language models (LMs) trained on text and applied to transcriptions, creating a text bottleneck that limits evaluation of the theory in the spoken modality. We remove this bottleneck by estimating surprisal directly from the speech signal using a speech LM. This raises a key question: should surprisal be defined over realization-specific acoustic inputs or over abstract linguistic units? Since a single unit can have many distinct acoustic realizations, each assigned a different probability by the model, defining surprisal over abstract units is difficult. To address this, we introduce an importance sampling estimator that approximates the probability of an abstract unit by marginalizing over realizations generated by a text-to-speech model. Using pupillometry data from a listening study, we evaluate surprisal theory in this text-free setting at multiple levels of granularity. Obtaining surprisal from several speech LMs, we consistently find that surprisal estimated via importance sampling better predicts pupil dilation than realization-specific surprisal for words. We also find that surprisal values for sub-lexical units are predictive of pupil dilation. While we find that surprisal can be estimated without recourse to text, and that marginalizing over distinct realizations of words is more predictive of lexical processing effort, surprisal as a whole is not a strong predictor of change in pupil size, regardless of its source. 

---
# Reliable Memory from Unstable Growth: Curating Self-Written Agent Playbooks with Randomized Experiments 

**Authors**: Hoang Pham, Cong-Duy Nguyen, Dong Le, Ponhvoan Srey, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2610.18427)  

**Abstract**: Persistent textual memory lets language-model agents reuse lessons from past tasks to improve on future multi-step tasks without retraining. Systems such as ACE and Dynamic Cheatsheet keep it as a playbook: lessons the agent writes after each task and reads in its prompt. A key challenge is to decide which lessons improve later tasks. Current systems leave the decision to the agent itself, which makes performance unstable: the same procedure can write a playbook that helps or one that falls up to 69.6 points below no playbook. These findings motivate valuing lessons by their measured effect on task success rather than by the agent's judgment. We propose Randomized Block Curation (RBC), which measures this effect with a randomized experiment on practice tasks. It groups related lessons into blocks large enough to move success measurably and, on each practice task, includes or leaves out every block at random. Random inclusion makes tasks with and without a block comparable, so the gap in their success estimates the block's effect. RBC keeps the blocks that help and, since the same procedure can write a helpful or a harmful playbook, grows several and serves whichever curated, whole or no playbook scores best on fresh practice episodes. Across two memory systems, three models and three benchmarks, RBC raises average success to 66.9, against 55.5 with the whole playbook, 54.1 with no playbook and 53.8 when the agent's own judgment picks the lessons to keep. 

---
# Receiver Decides Agent-to-Agent Negotiations 

**Authors**: Fardeen Sadab, Alif Hasnain, Afsana Shehrin, Md. Azam Hossain, Abu Raihan Mostofa Kamal  

**Link**: [PDF](https://arxiv.org/pdf/2610.18424)  

**Abstract**: When one language-model agent asks another for money, which of the two decides how much changes hands? We ran a round-robin tournament of 1,049 negotiations among fifteen agents. In each game a requester tries to raise money from a holder that has been told to keep a $100 endowment. Knowing which agent holds the money explains 47% of the variance in the amount transferred; knowing which agent asks explains 2%. Dropping the most extreme agents narrows the split to 22% against 4%. Changing the requester's appeal from one named person to a group changed what requesters wrote, but the amounts transferred under the two framings are statistically equivalent. The holders that lose least are the ones that turn down the opening request, and their turn-1 refusal rate ranks them by total losses (Spearman correlation = -0.73). Training moves this behaviour in both directions. Fine-tuning Qwen3-8B on a persuasion corpus with almost no refusals raised what it gave away as a holder from 12% to 84% of the endowment. Distilling two never-losing holders into a 4B student cut its losses from 95% to 1%. Evaluations of agents that hold resources should test them in the holder role, including after persuasion fine-tuning. 

---
# Agaram 2.5B: The First Tamil-First Language Model Trained from Scratch with Tool Calling. A Playbook for Under-Represented Languages 

**Authors**: Ramchand Kumaresan, Madhusudhanan G, Kaushik Kezendirane  

**Link**: [PDF](https://arxiv.org/pdf/2610.18420)  

**Abstract**: We trained Agaram 2.5B, a 2.5-billion-parameter Tamil-English language model, from scratch on about 50 billion tokens (58% Tamil, the rest English, math, code and parallel text) on one node of eight H100 GPUs in about six days, with a new tokenizer that, unlike our first model's, keeps line breaks. Supervised fine-tuning on 134,428 conversations, whose replies an open-weight teacher model wrote, and three short targeted rounds turned it into a chat model that answers in Tamil script, romanized Tamil (Tanglish) and English and can call tools: on 25 tool prompts whose schemas appear in training it picks the right tool 23 times and gives exactly the expected arguments 13 times. The released model translates better than Sarvam-1, a 2B Indic base model, in all six FLORES and IN22 directions (FLORES chrF 54.3 into Tamil and 55.2 into English, against 49.2 and 44.2), and nearly doubles its base model on native-Tamil IndicIFEval (19.3% from 10.0%). It is weak on knowledge: reading-comprehension and knowledge benchmarks sit near chance, and on 50 held-out everyday questions written by one native speaker, one LLM judge rated 46% of its replies usable (22-24 of 50 over three passes; 95% interval for 23 of 50, 33-60%), with nearly every failure a wrong fact. Our two further SFT experiments changed answer format but not the knowledge measures. We also describe our first model, Agaram 1.5B, whose failures shaped the choices here; how the judge's scores varied between passes on identical replies; and, item by item, what the targeted rounds trained against and where frontier-model text entered the data. The project ran on compute credits and our own funds, about USD 13,300 across both models; the weights are released (licence to be announced after legal review), with GGUF files for this http URL, at this https URL code at this https URL 

---
# Do Factuality Methods Make LLMs More Factual? Reproducibility and Fair Comparison 

**Authors**: Pratuat Amatya, Vinay Setty, V Venktesh, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2610.18415)  

**Abstract**: Large Language Models (LLMs) are aligned to human preferences, after pre-training and supervised fine-tuning steps. However, standard RL alignment steps such as RLHF do not focus on factuality of LLMs, hence they may generate non-factual information, which could result in unintended consequences. In response, a number of factuality alignment approaches have been proposed in the literature which claim to improve factuality of LLMs on their own benchmarks. However, these methods have not been reproduced independently and are also not comparable since each uses a different base model, training data, evaluation set, judge, and metric. In this paper, we present the first independent reproduction of five factuality alignment methods, which span on-policy RL (PPO, GRPO, and Dr.~GRPO) and offline preference optimization (DPO and KTO), and compare them on a common benchmark with common evaluation metrics. We found that none of these methods reproduces in full, and on the common benchmark none of the methods significantly outperforms the base model. On closer inspection, the rewards optimize for format, length, or brevity rather than factuality and relevance, whereas the choice between on-policy and offline training matters little. Furthermore, the evaluation metrics and benchmarks are also part of the problem since they do not measure relevance of text to the prompt. Since there is no ground truth associated with the prompts, the reliance on LLM judges and retrieval for verification also makes the evaluation fragile. We release our code, data, and checkpoints. 

---
# Revisiting the Estimation of Unigram Surprisal in Computational Psycholinguistics 

**Authors**: Zachary Hopton, Francesco Ignazio Re, Samuel Kiegeland, Andreas Opedal, Eleanor Chodroff, Ryan Cotterell  

**Link**: [PDF](https://arxiv.org/pdf/2610.18414)  

**Abstract**: Contextual and unigram surprisal are standard predictors of reading time in computational psycholinguistics. While contextual surprisal is typically derived from neural language models, unigram surprisal is usually obtained from independent corpus counts. Although both quantities are deterministic functionals of a single underlying distribution---capturing conditional and marginal structure respectively---this practice breaks their probabilistic link and may introduce estimation error from the mismatch. We propose three principles for estimating unigram surprisal in psycholinguistic research, including the requirement that both quantities be derived from the same language model. We operationalize this principle by introducing a Monte Carlo estimator of unigram surprisal from a language model, along with a marginalized variant that has lower variance empirically. When unigram surprisal is re-estimated under this unified framework, we find evidence that predictors change in their predictive contribution to reading time. We also confirm prior findings that frequency can be a stronger predictor of processing effort than contextual surprisal under this unified framework. 

---
# Prompt-Injection Probes Learn the Conversation, Not Just the Injection 

**Authors**: Kaan Kale, Sriram Vishwanath  

**Link**: [PDF](https://arxiv.org/pdf/2610.18408)  

**Abstract**: Linear probes on a model's hidden states are used as cheap monitors for indirect prompt injection: they screen the tool outputs an agent reads and send suspicious ones to a slower second check, so every false alarm has a cost. But a probe learns whatever separates its training examples. The standard evaluation of injection probes can be passed without the model: conversation position and text lengths alone reach 0.91-0.98 AUROC. When each attack is compared with the same conversation without the injection, or with harmless text formatted like an injection, standard probes flag most of these harmless cases. We add such cases to the training data as matched negatives. Across six open models, in applications seen during training and at the default threshold, changing only the training data cuts false alarms on injection-free conversations from 41-80% to 4-27%, and on harmless look-alikes from 92-98% to 3-16%. With a threshold set on normal traffic, our probe still catches a median 95% of the attacks that fooled the agent (91% without our data). In new applications, false alarms stay low, but detection depends on the model and on how the threshold is set. The same data also helps a text-only filter. 

---
# Lightweight but Linguistically Fair? Quantisation Robustness across Bavarian, German, and English 

**Authors**: Jophin John, Michael Hoffmann, Piroska Lendvai, Florent Dufour  

**Link**: [PDF](https://arxiv.org/pdf/2610.18343)  

**Abstract**: Post-training quantisation reduces the memory cost of large language models, but it remains unclear whether compression preserves language skills of regional varieties as reliably as their standard-language counterparts. We study this question for Bavarian, Standard German, and English using BAVGROUND, a multiple-choice benchmark, where 206 source questions are aligned across all three languages. Across 14 open-weight 7B--10B instruction models, we compare BF16 with 8-bit GPTQ and two 4-bit settings, GPTQ and AWQ. Eight-bit GPTQ remains close to BF16 and largely preserves the relative behaviour of the three languages, while 4-bit quantisation was found to be much less stable. The two 4-bit methods reduce average accuracy by about 1.4--1.5 percentage points and change 11.9--16.0% of BF16 decisions, with changes concentrated on examples where the BF16 model is already uncertain. This instability does not translate into a consistent Bavarian disadvantage. Although Bavarian, a low-resource dialect, has lower baseline accuracy than German and English, the language most affected at 4 bits varies across models and quantisation methods. Calibration further shapes these cross-language effects: the relative Bavarian--German effect changes with calibration language, while changing the calibration source has little average effect on GPTQ-4 but can affect AWQ-4 much more strongly. Free-form evaluation on five models further shows that 4-bit compression can degrade generated responses even in settings where multiple-choice accuracy changes only modestly, while 8-bit GPTQ remains comparatively stable. Overall, our results show that quantisation robustness for regional language varieties depends on the model, quantisation method, calibration data, and evaluation setting, and cannot be measured by aggregate multilingual accuracy alone. 

---
# Probing the Geometry of Color Representations in Multilingual Language Models 

**Authors**: Pablo Loyola, Edison Marrese-Taylor, Andrés Hoyos-Idrobo  

**Link**: [PDF](https://arxiv.org/pdf/2610.18331)  

**Abstract**: Do multilingual language models learn a universal internal geometry for color, and is it perceptually grounded or primarily linguistic? We decompose color representations into two geometric axes: (P1) alignment to perceptual color space, and (P2) cross-lingual symmetry. Analyzing layerwise embeddings across models and multilingual datasets, we find that perceptual grounding is weak: alignment to CIE-LAB peaks in intermediate layers, then collapses toward output layers. Cross-lingual symmetry degrades with depth, suggesting later layers encode language-specific structure. Functionally, we show that geometric properties predict task performance: models achieve 96.7% accuracy on color classification at layers where P1 alignment peaks, with P1 alignment (r=0.70) significantly correlating with downstream capabilities. Our results reveal a dissociation: text-only LMs learn universal color-language geometry, not universal color-perception geometry, providing diagnostics for where perceptual structure is preserved in language models. 

---
# Prompt or Policy? Optimizing Physician Slate Presentations with LLM User Simulators 

**Authors**: Andrei Niculae, Adrian Cosma, Emilian Radoi  

**Link**: [PDF](https://arxiv.org/pdf/2610.18309)  

**Abstract**: Choosing the right physician is a key challenge for patients seeking medical care. Physician recommendation systems identify relevant doctors, but patients still find it difficult to compare their profiles to make a decision. We formulate slate presentation optimization: given a patient's medical case and a fixed, ranked list of physicians, we generate short, case-specific summaries that make doctors easier to compare, while remaining grounded in the information in their profiles. To optimize physician presentations without relying on real patient interactions, we develop an offline framework that uses persona-conditioned LLMs to simulate patient choices. We generate patient personas from attributes extracted from each medical case and evaluate presentations for factual accuracy, relevance, naturalness and simulated patient choice. We also use a second LLM judge, unseen during optimization, to assess simulator-specific overfitting. We compare automatic prompt-level optimization with GEPA, parameter-level optimization with GRPO, and their combination across two model sizes. GRPO substantially outperforms GEPA in satisfying the full set of constraints, and its gains in simulated conversion persist under the unseen judge. GEPA alone brings smaller improvements, but using its optimized prompt to initialize GRPO speeds up training and achieves the highest simulated conversion. These results show that prompt-level and parameter-level optimization play complementary roles in improving how physician recommendations are presented. 

---
# SimInter: Evidence-Grounded User Simulation with Interview-Based Profile Enrichment for Conversational Recommender Systems 

**Authors**: Erfan Shahabi, Amirhossein Roshandel, Mohammad Sadegh Sadeghi, Azadeh Shakery  

**Link**: [PDF](https://arxiv.org/pdf/2610.18308)  

**Abstract**: Conversational Recommender Systems (CRSs) require reliable user simulators for cost-effective evaluation. LLM-based simulators have made realistic simulation possible but typically condition on profiles derived from explicit ratings, demographics, and static sources capturing surface preferences while omitting implicit behavioral characteristics such as decision-making style, aspect-level preferences, and domain expertise. We propose SimInter, a pipeline that elicits these attributes via synthetic interviews between two LLM agents, grounded in the user's statistical profile and historical conversational evidence. Before profile enrichment, an audit agent explicitly flags potential fabrication and stereotyping at the turn level. We instantiate and evaluate SimInter on movie recommendation. We simulate 300 ReDial users and, on a matched subset of 100 users, compare SimInter with a statistics-only baseline: SimInter significantly increases the diversity of expressed preference aspects (filtered aspect entropy: 1.619 vs. 0.859; paired d = 0.96, Holm-adjusted p < 0.001). This gain persists when the simulator converses with two trained recommenders, KBRD and UniCRS, under two independent evaluation judges (+0.28 to +0.56 nats; all p < 0.001). Human evaluation finds 94.3% of enriched-attribute claims supported or partially supported, while human judgments agree with 95.2% of the auditor's binary filtering decisions on a stratified sample. 

---
# LegalRewardBench: Benchmarking Reward Models for Legal Preference Evaluation 

**Authors**: Hui Huang, Jie Zhou, Hongli Zhou, Qundong Shi, Shuyuan Zheng, Yini Su, Shaochun Wang, Biyuan Lin, Muyun Yang, Tiejun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.18299)  

**Abstract**: Reward models are central to preference alignment for legal tasks, yet existing reward model benchmarks are still primarily centered on general-domain preferences. In this work, we introduce LegalRewardBench, the first human-verified benchmark for evaluating reward models in the legal domain. The benchmark organizes legal preference evaluation into five capability levels: knowledge, understanding, analysis, decision, and writing, and preference examples are constructed through rubric-guided automatic judging combined with human verification. We evaluate three families of judges, covering both general and fine-tuned models, and generative and discriminative architectures. Experimental results show that strong general-purpose generative models are currently the most robust legal reward models, while existing fine-tuned reward models transfer less reliably to legal preference evaluation. We also conduct detailed comparisons of input organizations and training strategies. 

---
# JevGPT: Thinking Fast and Slow in Autonomous Research 

**Authors**: Yasheng Sun, Yanning Dai, Yimeng Chen, Nanbo Li, Firas Laakom, Yifan Xu, Piotr Piękos, Mateusz Ostaszewski, Wenyi Wang, Cheng Luo, Yuhui Wang, Jürgen Schmidhuber  

**Link**: [PDF](https://arxiv.org/pdf/2610.18272)  

**Abstract**: Autonomous research agents must acquire and reason over evidence within a limited budget. Yet relevant sources can remain uninspected or unused, leaving a gap between retrieval coverage and the evidence that informs research. The challenge is to align fast local judgments with changing evidence needs and feed their findings back into research decisions. We introduce JevGPT, a dual-system architecture for autonomous research that couples fast evidence processing with deliberative research planning. System Two sets evidence objectives and integrates findings, while System One locates support, assesses relevance, and guides further inspection. Independent source judgments run in parallel; selected original passages inform the next research step. Experiments on fixed-corpus question answering and document ranking show that JevGPT achieves leading task performance, with faster completion and lower cost than comparably effective research agents. Evidence analysis shows broader source inspection and greater coverage of supporting documents in the research loop, highlighting efficient evidence processing as a route to more capable autonomous research. 

---
# PAINT: A Framework for Multilingual Pain Representation from Medical Licensing Exams to Real-World Clinical Texts 

**Authors**: Boya Zhang, Yuanchao Ma, Jiageng Wu, Krittaphas Chaisutyakorn, Rodrigo Gameiro, Ahram Han, Chien-Yu Huang, Nikhil Jaiswal, Marianna Leite, Miriam Mora-Mau, Jaime Pineda-Moreno, Siena Placino, Ryohei Yamamoto, Yugang Jia, Jie Yang, Leo Anthony Celi, Douglas Teodoro  

**Link**: [PDF](https://arxiv.org/pdf/2610.18269)  

**Abstract**: Pain is one of the most common and clinically important symptoms, reflecting disease onset, progression, and treatment response. However, pain is difficult to accurately capture and interpret from medical text, as its expression varies across languages, documentation types, and patient demographics, and is often embedded in terminology, negation, temporality, and contextual descriptions. Existing pain-related studies remain largely monolingual or limited to narrow clinical settings. In this study, we introduce PAINT, a multilingual framework for studying pain representation across medical exams and real-world clinical texts. PAINT includes nine pain-focused dataset partitions spanning English, Spanish, Portuguese, and Japanese, with clinician annotations in all four languages. We curated a structured schema covering pain relevance, pain presence, anatomical location, pain descriptors, and demographic information. To capture multilingual pain representations, we leverage large language models (LLMs) and adopt an ensemble strategy to improve performance. Ten clinicians across four languages validate the labels: the ensemble reaches macro-F1 of 0.91-1.00 on pain relevance and pain presence and 92-100% accuracy on demographics, with inter-annotator $\kappa$ of 0.82-1.00. We release the WorldMedQA-Pain ensemble labels, the extraction pipeline, validation scripts, and clinician annotation sheets for publicly licensed corpora, providing a foundation for pain-centered clinical NLP and for studying pain documentation across languages and healthcare contexts. 

---
# Are We Testing Too Much? A Survey of Benchmark Compression for Efficient Evaluation of Large Language Models 

**Authors**: Guoming Ling, Yupei Lin, Zhongzhan Huang, Junxin Li, Junjie Liu, Shanshan Zhong, Hefeng Wu, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.18248)  

**Abstract**: Benchmarks are essential for evaluating and improving large language models (LLMs). Yet many widely used benchmarks are redundant, making evaluation unnecessarily expensive. Benchmark compression lowers this cost by estimating full-benchmark scores and rankings from a partial evaluation. Research in this area has grown rapidly since 2024, but to our knowledge no survey has systematically organized its methods and findings. We review 119 papers to explain how methods choose what to evaluate, recover full-benchmark results, and verify their reliability. From this analysis, we develop a taxonomy covering compression within individual benchmarks, across benchmark suites, and during adaptive evaluation, together with work on redundancy and reliability. We compare their information needs, total costs, reliability checks, and limitations. Methods reduce evaluation costs by selecting representative subsets, using cheaper or reusable information, and testing adaptively, but risk poor coverage, inaccurate estimates, and sequential delays. The resulting taxonomy and practical guide help researchers choose, validate, and deploy methods for more efficient and reliable LLM evaluation. 

---
# PACT: A Pragmatic Prompting Framework for Expressive and Voice-Preserving Speech-to-Speech Translation 

**Authors**: Yuchen Song, Satoshi Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2610.18247)  

**Abstract**: Speech-to-speech translation should preserve not only what is said, but also who says it and how it is said. Existing end-to-end S2ST, speech MLLM, and cascade systems can preserve content or timbre, but often struggle to realize emotion, tone, rhythm, pauses, and pragmatic attitude in the target language. We propose PACT, a target-space pragmatic prompting framework for expressive and voice-preserving S2ST. PACT extracts source pragmatic cues, projects them into the target-language expression space, calibrates the projected representation to a realizable target subspace, and converts it into continuous prompts for a frozen speech generator. On bidirectional Spanish-English TOPI evaluation, PACT achieves the best overall performance among evaluated systems across content fidelity, voice preservation, and expression preservation metrics. 

---
# TextualRL: Textual Reinforcement Learning for Context Management 

**Authors**: Tianyu Huai, Jie Zhou, Bo Mao, Qin Chen, Liang He, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2610.18216)  

**Abstract**: LLM agents can adapt to task feedback by revising the textual context that guides their behavior, without changing their model parameters. However, turning execution rewards into reusable context updates is challenging: individual trajectories offer limited comparative evidence about what to change, while task-specific fixes may overgeneralize when incorporated into a shared context. We introduce TextualRL, a gradient-free reinforcement learning framework that treats a group of rollouts from the same task under the same context, rather than an isolated trajectory, as the unit of policy critique. Groupwise Policy Critique contrasts successful and failed rollouts within a task; when outcomes are uniform, it draws on cross-task evidence to identify behaviors worth preserving or formulate repair hypotheses. Cross-Group Policy Update reviews the resulting proposals against evidence pooled across groups, including observations that yield no proposed edit, to refine their content and applicability. Reviewed proposals form a bounded textual patch, with adoption of the candidate context determined by held-out validation rewards. Together, these components establish a reward-driven policy optimization loop in which multiple attempts jointly inform reusable context updates. We evaluate TextualRL with five target agents on six benchmarks. Averaged over the six benchmarks, TextualRL improves test accuracy over the base model by 24.4 and 15.3 percentage points for GPT-5.5 and Qwen3.8-27B, respectively. 

---
# TRACESTEER: Trace Driven Activation Steering for Hallucination Mitigation in Diffusion Language Models 

**Authors**: Xin Zhang, Yili Wang, Yue Tan, Xin He, Rui Miao, Yixin Liu, Yi Chang, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.18173)  

**Abstract**: Diffusion large language models (dLLMs) have emerged as a promising alternative to autoregressive LLMs, yet remain susceptible to hallucination. Activation steering offers a promising mitigation approach by guiding parallel token predictions through internal representations. However, existing steering methods largely overlook two characteristics of dLLMs: dynamic denoising states and temporal hallucination risks, which challenge predefined steering vectors and uniform intervention strengths. We propose TRACESTEER, a trace-driven activation steering framework that uses preceding denoising predictions as feedback to jointly adapt steering direction and strength. Specifically, Denoising Feedback Steering (DF-Steering) constructs step-specific steering vectors from preceding predictions and current hidden states. Trajectory-Guided Steering Gate (TS-Gate) regulates their strength using prediction uncertainty and the denoising timestep. With the backbone frozen, factual supervision and magnitude regularization jointly guide the learning of adaptive steering interventions, promoting factual reliability and stability throughout denoising. Experiments on two dLLMs show improved factual reliability on TruthfulQA and generalization across five downstream benchmarks. 

---
# Claim Framing as a Driver of LLM Sycophancy 

**Authors**: Jean Seo, Pepa Atanasova, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2610.18159)  

**Abstract**: Sycophancy in Large Language Models, the tendency to align responses with user preferences over ground truth, has largely been studied using questions without a clear ground truth and with a single, plainly stated user opinion. We present a systematic study of which framings of a user's claim trigger sycophancy in factual question answering, where every answer is objectively verifiable without extra context. Evaluating 14 models on 42K questions, we express correct and incorrect user opinions through various claim framings over 8 dimensions and 21 types, and measure a broad set of opinion-driven behaviors. We find that all models are susceptible to user opinions regardless of size. Models are most influenced by how confidently an opinion is stated, whom it is attributed to, and whether it is presented as shared between the user and the model. In contrast to prior work, framings based on the user's expertise, authority over the model, or social pressure matter little in our factual setting. Larger models resist incorrect opinions better but are no better at recovering from their own errors. Our results call for sycophancy evaluations that probe how an opinion is framed, not merely whether one is present, and that measure multiple behaviors beyond a single flip rate. 

---
# Fan Out or Relay? Ownership-Aware Routing for LLM Agents Querying Siloed Tables 

**Authors**: Zhexi Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.18157)  

**Abstract**: Inside organizations, a question's data often sit with several owners, who query their own tables but never hand them over. An LLM coordinator can fan out, sending the question to every owner at once, or relay, querying owners in turn and reusing earlier rows. We show that neither is safe in advance, because data ownership decides which one is right. Across thirteen settings on PeopleJoin and BIRD with three model families, fan-out is more accurate in twelve on questions one owner can answer, yet answers at most 4.3% of those joining two owners' tables. The flip follows ownership, not query shape: on BIRD two-table joins, fan-out answers 32-44% when one owner holds both and 1-4% otherwise. Relay wins by conditioning on returned rows, and hiding them from an otherwise identical planner cuts two-owner accuracy from 23.2% to 5.1%. Since ownership is public, we propose Ownership-Aware Routing (OAR), which drafts a schema-only SQL query, counts the owners whose tables it joins, and routes the question before any private row is read. In ten settings OAR is never significantly less accurate than the best fixed protocol, and beats every fixed protocol on PeopleJoin, by up to 9.3 points on held-out organizations. 

---
# Attacking Only Half the Data: Two-Sided Evaluation of AI-Text Detector Robustness 

**Authors**: Apu Chandraw Shill, Rabindra Nath Nandi  

**Link**: [PDF](https://arxiv.org/pdf/2610.18151)  

**Abstract**: We investigate the robustness of widely used AI text detectors under attacks applied to both human and machine text. Prior evaluations typically attack only machine text, overlooking that human text is routinely modified through grammar correction, formatting, or invisible character insertion. We introduce a two-sided protocol that applies attacks to both classes. Across two corpora and nine detectors, reported AUROC gains vanish under two-sided evaluation, with attacked human text misclassified at rates up to 100%. While simple preprocessing reverses character-level attacks, word-level synonym substitution remains effective meaning-preserving, and irreversible. 

---
# SparPR: Runtime KV-Cache Recovery for Resource-Constrained LLM Inference on Consumer Hardware 

**Authors**: Jieyun Deng, Keming Hu, Yanghe Liu, Li Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.18140)  

**Abstract**: Running large language models (LLMs) on consumer hardware is bottlenecked by the key-value (KV) cache, which grows with every decoded token and quickly exhausts the memory and bandwidth of a single consumer GPU. We present SparPR (Sparse Partial Recomputation), a runtime KV-cache recovery mechanism that does not modify model weights and requires no offline training. SparPR couples three online decisions: Query-Aware key-block selection scores historical blocks against the current query; dynamic budget control estimates per-block copy and recompute costs to decide how many blocks to retain; and on-demand local recomputation with cost-aware fallback merges retained blocks into continuous spans and switches between sparse recomputation and full recovery according to a runtime cost comparison. Implemented on FlexGen/FlexLLMGen, SparPR is evaluated on a single consumer laptop (RTX 3070 Ti 8GB, Ryzen 9 6900HX, 16GB RAM). It cuts end-to-end latency by 31%, 52%, and 63% at 1k, 2k, and 4k prompt tokens respectively, with 1.4x-2.7x higher throughput, and completes an 8k-token workload in 44 minutes where the FlexGen baseline times out. Under a fixed cache budget, SparPR retains a Needle Accuracy that stays well above StreamingLLM and H2O at every keep ratio, and ablation studies confirm that the three mechanisms act jointly. 

---
# Cheap, Consistent, Calibrated, but Biased? Typed Decision Models as Judges 

**Authors**: Fabian Lukassen, Christoph Weisser, Alexander Silbersdorff  

**Link**: [PDF](https://arxiv.org/pdf/2610.18122)  

**Abstract**: Typed decision models (TDMs) return a probability for each allowed answer instead of generating text, which makes them candidates for cheaply replacing LLM judges that compare two answers. Whether they share the biases of LLM judges is untested. We run the CALM benchmark's bias tests on eight TDMs and seven LLM judges of 2024 and 2026. Every TDM is biased, in three to eight of twelve tests. Emotional tone affects them most, and rewriting the better answer in such a tone lowers the accuracy of a typical TDM from 99% to 42%. TDMs are more biased than 2026 LLM judges. All eight change verdicts when the answers swap places, which the 2026 LLM judges hardly do, and several side with an answer that most people allegedly prefer or that a famous model allegedly wrote. Yet both kinds of judge are tripped up by the same prompts. The TDMs' probabilities show that most tests mainly flip verdicts the TDM was unsure of, and that good calibration does not protect against bias. As TDMs are more consistent and 37 times cheaper than 2026 LLM judges, our results suggest they can be attractive judges if their biases are tested for the task. 

---
# Narrative Lookahead in LLMs: Event-Level Decodability, Predictive Advantage, and Steering 

**Authors**: Donghoon Jung, Min Kim, Seojin Choi, Songeun Chae, Seohyon Jung  

**Link**: [PDF](https://arxiv.org/pdf/2610.18121)  

**Abstract**: Do language models carry information about narrative events before writing them, and how far ahead? We examine narrative lookahead through salient, state-changing events rather than tokens or response-level attributes. We extract 23,355 LLM-annotated events from 2,000 stories generated by four open-weight LLMs, labeling them along three nonexclusive dimensions of change: within agents, between agents, and in the story world, with human validation on a subset. Event segmentation and plot-independent labels enable comparisons across stories at horizons measured in events rather than in tokens. At event ends, we probe hidden states for upcoming event types and compare predictions with text-only predictors using matched prefixes. Across four models, future event types are decodable above prevalence baselines for at least four subsequent events, but probes significantly outperform both primary text-only predictors only for the next event. In exploratory steering with OLMo-3.1-32B, a probe-derived direction shifts the proportion of the target event type in both intended directions at higher strengths. Our event-level approach links the prediction of future narrative developments to interventions that reshape subsequent event composition. 

---
# From Agents to Edges: Paired Optimization for Multi-Agent Collaboration 

**Authors**: Feng Lin, Xiaotian Yu, Yifan Yang, Aijun Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.18069)  

**Abstract**: Strong individual agents do not guarantee effective collaboration. Communication must align what one agent conveys with how another interprets and uses it. This dependency makes the communication edge a natural unit of optimization. We introduce Paired Automated Communication Optimization (PACO), which jointly optimizes the instructions at both ends of each edge and their shared communication convention in systems built on black-box language models. PACO keeps model weights, agent roles, and workflow topology fixed. A state-aware discounted bandit selects edges for refinement, while execution feedback guides paired revisions evaluated through staged end-to-end execution. The optimized instructions integrate into existing agent prompts without additional model calls at deployment. On a four-agent workflow evaluated across diverse multi-agent benchmarks, PACO consistently outperforms leading prompt-optimization and communication baselines. Configurations learned on HotpotQA retain their advantage in cross-dataset transfer without target-specific optimization. These results demonstrate that optimizing how agents communicate can substantially improve collaboration within an otherwise fixed system. 

---
# A Benchmark for Bangla Task-Oriented Dialogue Understanding 

**Authors**: Afrina Alam Lamia, Sumiaya Sharif, Prottoy Modak, Ohiduzzaman Rayhan, Nafis Sadeq  

**Link**: [PDF](https://arxiv.org/pdf/2610.18060)  

**Abstract**: Task-oriented dialogue understanding remains underexplored for Bangla, particularly in multilingual settings involving English and romanized Bangla (Banglish). We introduce BanglaTOD, a parallel tri-lingual benchmark comprising 209 multi-turn dialogues across nine Bangladesh-specific service domains and 64 intents. Each conversational turn is aligned across Bangla, English, and Banglish, yielding 4,689 language-specific turn records annotated with intents, token-level slots, state updates, and accumulated dialogue states. We establish context-aware baselines for joint intent classification and slot filling using mBERT, BERT-base-uncased, and BanglaBERT. Cross-language analysis reveals substantial performance differences across models and linguistic representations, while removing dialogue history reduces intent accuracy by 19.0--25.7 percentage points. BanglaTOD provides a foundation for multilingual dialogue understanding and future dialogue state tracking research in Bangladesh-specific conversational applications. The dataset is publicly available at \href{this https URL}{Hugging Face}. 

---
# Machine--human agreement in marking undergraduate assignments is specific to model, prompt and institution 

**Authors**: Giulio Corsi, Maryam Abo-Tabik, Yael Benn, Georgiana Thorpe Apreutesei, Roni Tibon, Alexandru Marcoci, Deborah Talmi  

**Link**: [PDF](https://arxiv.org/pdf/2610.18019)  

**Abstract**: Large language models (LLMs) promise to automate one of higher education's most resource-intensive activities: assessing student writing. Existing evaluations are typically limited to a single model, a single prompt and a single institution. Using 761 undergraduate psychology assignments from three UK universities, we evaluated three frontier LLMs, individually and in ensemble, on accuracy, prompt sensitivity, error independence, generalisability and evidential basis. Agreement between human and machine marks varied widely, with no model or prompt optimal across institutions. All models compressed marks towards a common central tendency, overvaluing the weakest assignments and undervaluing the strongest. Because this error was shared, ensembling failed to remove it. Machine marks also rewarded linguistic features more strongly than human marks. Since human-machine agreement was a joint product of model, prompt and institution, we conclude that current LLMs lack evaluative capacity transferable across contexts, and that single-context validation overstates readiness for deployment. 

---
# LatentRemedy: Efficient Remedy For Non-Prefix KV Cache Reuse in Multi-Agent Systems 

**Authors**: Zhiwei Chen, Zhongtao Rao, Waikit Xiu, Yuxiang Zhang, Shuting Cui, Jian Chen, Tianxiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.18014)  

**Abstract**: Non-prefix KV cache reuse is a promising approach for improving inference efficiency in multi-agent systems as agents share the same context blocks. However, when agents have conflicting tasks, directly reusing non-prefix KV caches across agents can severely degrade model performance, causing up to a 90% drop. Existing remedies, including selective recomputation and offset-based calibration, recover a limited fraction of this loss. We argue that the problem lies not only in contaminated KV representations but also in the agent's difficulty interpreting reused context under its own instructions. To this end, we introduce LatentRemedy, which uses a few task-conditioned latent tokens to help the target agent leverage reused context without recomputing the shared block. Specifically, we use an MLP to map the last hidden state of the target prefix into a small number of latent tokens, which are placed after the reused KV block to interact with the shared context. To guide subsequent predictions toward the target task, we train the MLP through full-prefill distillation. Experiments show that LatentRemedy consistently outperforms existing reuse baselines on conflicting tasks while maintaining TTFT close to direct reuse as cache size grows. Code and data are available at this https URL. 

---
# Seeing Through the Forgery Is Not Enough: Regressive Sycophancy under Fabricated Evidence in Medical Vision-Language Models 

**Authors**: Sadif Ahmed, Soumya Swagata Biswas, Rifat Shahriyar, Chowdhury Mohammad Rakin Haider  

**Link**: [PDF](https://arxiv.org/pdf/2610.17989)  

**Abstract**: A medical vision-language model may analyze a given CT scan and correctly detect both kidneys in view. Continuing this conversation, if a user provides a prescription with an opposite direction, medical VLMs often sycophantically choose to realign their diagnosis. In this work, we study regressive sycophancy under fabricated evidence, where a model abandons a correct answer when pushback comes with a fake document. We test four open VLMs on 400 VQA-RAD yes/no questions in 19,200 multi-turn conversations, with and without a fabricated prescription. Without a disclosure that documents are synthetic, the models abandon 80.1% of 6,488 correct answers. Notably, under one script each, the same prescription raises the flip rate of Qwen2.5-VL-7B from 45.9% to 89.6% and lowers that of Qwen3-VL-8B from 77.7% to 50.9%. A model rejected the prescription while still preserving the correct answer in 915 of 3,232 conversations with the prescription (28.3%), and 40% of these flipped in later turns. System-prompt sentences that disclose the documents as synthetic lower one model's flip rate from 77.8% to 47.3%. Thus, we conclude that awareness about a forgery does not prevent a model from adopting its answer. 

---
# Virtual Head Attention 

**Authors**: Xibo Ding, Guoxia Wang, Jinle Zeng, Jiabin Yang, Sirui Tao, Chao Yang, Dianhai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.17959)  

**Abstract**: The design of attention mechanisms in long-context large language models faces an ``impossible triangle'' among KV cache size, training and prefill FLOPs, and model quality. GQA achieves strong quality but incurs high KV cache overhead; alternative approaches such as MLA and MFA can compress the cache, but MLA suffers from quality loss while MFA substantially increases training and prefill compute. We propose Virtual Head Attention (VHA), which simultaneously improves all three vertices through two lightweight linear operations: Q Premix applies a near-identity transformation to queries within each KV group to recover virtual head diversity from halved physical query heads, and Linear Postmix fuses inter-head features via a low-rank $I + AB^\top$ residual structure. Since VHA introduces no nonlinear operations, Premix can be folded into the query projection and Postmix into the output projection at inference time, reducing the model to a standard GQA-2 architecture that directly reuses all existing GQA inference optimizations. With only two KV heads, the group size in large models readily reaches 64, fully exploiting Tensor Core parallelism during decoding. Across four scales with 4K-context 50B-token pretraining, VHA consistently surpasses the standard GQA baseline at every scale ($+$0.05\% to $+$0.81\%), while reducing KV cache by $4\times$ vs.\ GQA-8 at dense scales and $2\times$ vs.\ GQA-4 at the MoE scale, and lowering training and prefill FLOPs by 7.9\% (1.7B, $S{=}4096$). On an industrial 30B-A3B MoE model trained on 500B tokens at 8K context, VHA outperforms both MLA and GQA-4 by $+$0.70\,/\,$+$2.60\,\% on a 12-benchmark base-model evaluation suite; in a separate matched 10B-A1B MoE comparison after long-context mid-training, VHA attains an average RULER score of 84.4 over 8K--128K versus 73.0 for GQA, while using two rather than four KV heads in the full-attention layers. 

---
# Agent-Reflex: Teaching Language Agents to Act Reflectively 

**Authors**: Guanting Dong, Xinyu Yang, Jiajie Jin, Yuyang Hu, Xiaoshuai Song, Yifei Chen, Xiaoxi Li, Jiejun Tan, Yutao Zhu, Tat-Seng Chua, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2610.17953)  

**Abstract**: Large language models have evolved from single-turn chatbots into interactive agents capable of long-horizon reasoning. In real deployments, MCP tool services can return noisy feedback and users may intervene to correct the agent, but existing methods provide limited supervision for reflection learning and failure recovery. To bridge this gap, we propose Agent-Reflex, a general training framework for incentivizing proactive reflection in agents. Specifically, Agent-Reflex comprises three stages. First, Environment Reflection Skill Discovery (RSD) injects grounded tool-feedback perturbations into environment scaling, samples a capability-diverse LLM pool, mines matched success-failure pairs, and summarizes reusable reflection meta-skills. Building on this skill library, we then apply Reflective Supervised Fine-Tuning (ReSFT) that converts each skill into supervision for self-reflection, skill recovery, and user-centric correction, strengthening complementary reflective behaviors. Finally, we propose On-Policy Reflective Policy Optimization (ORPO) to combine agentic reinforcement learning under noisy interactions with reflection-guided on-policy distillation to further elicit proactive reflection. Built on Qwen3-8B, Agent-Reflex reaches 58.6 overall on five long-horizon tool-use benchmarks and 29.1 on AgentNoiseBench, surpassing strong environment-scaling and reflection baselines. Experiments on six challenging agent benchmark suites demonstrate the effectiveness of Agent-Reflex. Further analyses show that Agent-Reflex improves agents' robustness to noise while maintaining stable training dynamics, supporting scalable agent training. Our code is available at this https URL. 

---
# Batch-Paced: Fixed-Budget Batch Decoding in Discrete Diffusion Language Models 

**Authors**: Shuai Zhang, Zhanchao Zhou, Huachuan Qiu, Anqi Li, Zhenzhong Lan  

**Link**: [PDF](https://arxiv.org/pdf/2610.17917)  

**Abstract**: As generative language models become more capable and widely used, inference demand grows, making efficient serving increasingly important. Discrete diffusion language models (dLLMs) offer a promising path to more efficient inference through parallel token generation. Prior dLLM decoding methods, however, focus on improving inference efficiency for individual sequences. We formulate fixed-budget dLLM batch decoding and introduce Batch-Paced, a training-free method that jointly optimizes the allocation of decoding compute across requests to improve overall batch quality. Across two dLLM backbones, five benchmarks and their representative mixtures, Batch-Paced improves batch performance under matched budgets; gains generally grow with batch size and largely persist across decoding configurations. Analysis of requests' decoding trajectories suggests that these gains arise largely by raising the committed-token confidence of lower-confidence requests. We further show that Batch-Paced's batch decode time remains competitive and that adding a decoding-depth allowance during Batch-Paced decoding can shorten it further at a modest performance cost. 

---
# Text-Only vs. Image-Aware VLM Judges for Manga Translation Evaluation 

**Authors**: Subaru Kimura, Jun Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2610.17908)  

**Abstract**: In manga translation, visual context outside speech-bubble text, including speaker identities, referents, and scene tone, is depicted in page images, making it natural to refer to images during translation evaluation. However, most conventional manga translation evaluations rely solely on text-only automatic metrics, leaving the impact of image input insufficiently explored. We propose a practical validation framework that employs the same vision-language model (VLM) as both a text-only judge (TOJ) and an image-aware judge (IAJ) to investigate evaluation divergences associated with image-aware evaluation procedures. We use a Japanese-English manga dataset of official translations aligned at the page and speech-bubble levels. First, adding page images to TOJ improved the primary VLM's detection of deliberately injected, visually dependent degradations. Second, we analyzed evaluation divergences between TOJ and IAJ, finding different error counts and error-category distributions. Human verification did not reveal a consistent preference between TOJ and IAJ. 

---
# Swiss4Arg: A Parallel Dataset for Multilingual Argument Mining across Swiss National Languages 

**Authors**: Sophia Conrad, Fatma-Zohra Rezkellah, Christina Niklaus, Reto Gubelmann  

**Link**: [PDF](https://arxiv.org/pdf/2610.17894)  

**Abstract**: We present Swiss4Arg, a multilingual dataset for argument mining based on official Swiss voting brochures, covering all four national languages: German, French, Italian, and Romansh (a Romance minority language exclusively spoken in Switzerland). The brochures contain explicitly formulated arguments systematically organized into pro and con sections, resulting in a cross-lingually aligned resource with balanced argumentative stance. We create a dataset that comprises 53 federal voting issues collected between 2019 and 2025, annotated for argument components (major claims, claims, premises, and non-argumentative units) and relations (support and attack) by three native speakers per language. Inter-annotator agreement (Krippendorff's alpha) reaches 0.548 for component classification and 0.623 for relation and target classification, computed jointly across all annotators on the parallel data. We release both adjudicated labels and individual annotator annotations, enabling analyses of disagreement and cross-linguistic variation, and provide baseline results using GPT-5.4 and SwissBERT. Swiss4Arg establishes a novel benchmark for multilingual argument mining, with a particular focus on cross-lingual comparability and low-resource settings such as Romansh. The dataset and code are publicly available at this https URL (HuggingFace) and this https URL (GitHub). 

---
# Cross-Lingual Transfer of Evil Vectors Is Directionally Asymmetric 

**Authors**: Juann Kim, Minku Kang  

**Link**: [PDF](https://arxiv.org/pdf/2610.17880)  

**Abstract**: Persona vectors are linear directions in the residual-stream activations of LLMs that encode behavioral traits and enable fine-grained control over model behavior. Among them, the evil persona vector is particularly relevant to understanding harmful behavior, jailbreak vulnerabilities, and the unlearning of harmful knowledge. Yet whether this vector reflects a genuinely cross-lingual concept, rather than an artifact of its extraction language, remains untested. We conduct a systematic evil-vector transfer study across two high-resource languages (English, Spanish) and two low-resource languages (Swahili, Telugu) on four open-source instruction-tuned LLMs, and find that transfer is directionally asymmetric. Evil vectors are harder to obtain reliably from low-resource languages, and the resulting vectors are largely ineffective at steering, even within their own language. In contrast, vectors extracted from high-resource languages transfer effectively to low-resource inputs while largely preserving general knowledge, as measured by MMLU. Geometric analysis further supports this pattern: evil vectors from the two high-resource languages are closely aligned, whereas low-resource vectors align only weakly with this shared direction. These results suggest that a shared cross-lingual evil direction exists and, once reliably extracted from a high-resource language, can be reused to steer behavior in low-resource languages without additional per-language data collection. 

---
# Prefilling-dLLM v2: Learned Chunk Routing for Long-Context Diffusion Language Models 

**Authors**: Qi Han, Jing Xiong, Ziyuan Zhang, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2610.17868)  

**Abstract**: Diffusion language models apply bidirectional attention over the whole context at every denoising step, so a longer context makes each step substantially more expensive. Sparsifying only at inference leaves the pretrained weights unchanged, so the model is never trained under the sparse pattern it runs with. We propose Prefilling-dLLM v2, which continues training a diffusion language model so that it routes with its native mask token. One mask is appended at the end of each chunk, and a learned role offset marks it as a routing slot. On every fourth layer, a query keeps a bidirectional local window and the top-K chunks scored from that mask; the other layers use a bidirectional sliding window. Starting from Dream-v0-Base-7B extended to 16K with YaRN, we compare Prefilling-dLLM v2 with bidirectional NSA, DSA, and SWA under the same layout, key budget, and dense initialization, using the same Fast-dLLM decoder. With the YaRN factor scaled to the evaluation length, Prefilling-dLLM v2 leads the other sparse methods on all three RULER tasks from 16K to 128K. Its mean accuracy at 128K is 7.4 points above NSA, the next sparse model, while DSA and SWA remain near zero. At the same length, decoding is 172x faster than with full dense attention. 

---
# Perturbation, Not Provenance: Why Activation Steering Does Not Make a Robust Text Watermark 

**Authors**: Sergey Pletenev  

**Link**: [PDF](https://arxiv.org/pdf/2610.17810)  

**Abstract**: Linear directions in the activation space of large language models causally and universally influence model behavior. This suggests that they might serve as a basis for a robust text watermark. The idea is to steer the generation along a specific semantic direction and then recover the mark by probing the activations of the re-encoded text. We test this hypothesis by conducting a series of experiments. We vary the strength of the steering and use four different directions (formality, sentiment, complexity, and a random control) across two different model families. We also test two paraphrasing attacks: a sentence-level one and a stronger paragraph-level one. The results show that activation-steering watermarks are outperformed by a standard logit-based watermark (KGW) in terms of detection accuracy. At a matched level of fluency (within 1.15 times the original text's perplexity), KGW achieved an AUC of 0.81 against a sentence-level paraphraser, while the best steering direction only reached 0.65. A stronger paragraph-level paraphraser lowered both (0.65 for KGW and 0.59 for steering), narrowing the gap but not reversing it. On Qwen, it left the two indistinguishable but at near-chance levels for both. We further showed that the detectability provided by steering was not semantic: a random direction was just as detectable as a meaningful one. The signal was independent of which layer was probed and whether a surface feature control was used. We conclude that, in this case, semantic activation does not encode a localized, recoverable semantic mark, but rather a global, content-independent shift that can only be strongly detected when the text is degraded. 

---
# BRIDGE: Selective Boundary Repair with Dependency Graph Evidence in Multi-Agent Systems 

**Authors**: Encheng Su, Pengze Li, Ziheng Hua, Qiucheng Yu, Jianyu Wu, Chen Tang, Yizhou Wang, Lintao Wang, Junjia Xiang, Xinzhu Ma, Xi Chen, Shixiang Tang, Houqiang Li, Xi Chen, Aoran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.17805)  

**Abstract**: Multi-agent collaboration has emerged as a critical paradigm for tackling complex, long-horizon tasks; however, while coordination protocols continue to advance, systemic failures at the inter-agent collaboration interface remain a pervasive bottleneck. In collaborative workflows, systems frequently suffer from \emph{broken handoffs}---where critical task context is omitted, misrouted, or ignored despite each agent emitting locally plausible responses. To address this challenge, we present \method, a verifiable runtime control framework that treats handoff repair as selective intervention over explicit dependency graphs. Central to \method is an integrated fault-attribution and calibration pipeline, \dataset, which decouples execution dependencies and injects counterfactual handoff perturbations to provide empirical supervision for runtime intervention. Leveraging this diagnostic foundation, \method dynamically compiles observable interaction traces into a provenance graph to pinpoint unresolved dependencies, without accessing private agent states. It then evaluates candidate repairs through an empirical gate that weighs expected gains against disruption risks, and executes repairs only when formal local postconditions hold. Extensive evaluations across demanding benchmarks demonstrate that \method boosts task success by up to 12.3 absolute percentage points on SILO-BENCH and 11.1 on TeamBench. Across both backbones, \method maintains strict cost-effectiveness, adding as little as 2.0\% marginal token overhead. Furthermore, on 562 handoff-relevant traces from the MAST benchmark, \method achieves 69.8\% F1 in a single pass, establishing a reliable operational boundary for verifiable coordination repair. 

---
# Where Words Begin and End 

**Authors**: Husein Zolkepli  

**Link**: [PDF](https://arxiv.org/pdf/2610.17782)  

**Abstract**: Forced aligners locate each word of a known transcript in the audio, and many speech pipelines depend on them. In our own data, a widely used CTC aligner inserts a pause at nearly every word boundary, and we found almost no aligner evaluated on telephone audio. We compare nine aligner configurations against human word boundaries on read, conversational and telephone English. The Montreal Forced Aligner (MFA) is the most precise, while CTC aligners consistently start words late, because most frames inside a word go to the blank token. We then fine-tune Whisper with LoRA to predict each word's start and end as timestamp tokens, training only on boundaries placed by people: 47 hours in 87 languages. The resulting aligner is more precise than MFA on every English test condition, and than MFA's own Korean and Dutch models. With FA-Bench's protocol and code on TIMIT, its word-boundary error is 11.6 ms, against 18.2 ms for the best published system. Two choices matter most on telephone conversations: placing the timestamp slots after the word they time, and training on noisy transcripts, which also makes the aligner nearly insensitive to errors in the transcript. The released model reaches 16.0 ms on real landline calls, against 36.5 ms for MFA. The models are available at this https URL 

---
# Full-duplex Spoken Language Modeling with Discrete Diffusion 

**Authors**: Chen Chen, Pin-Jui Ku, Ankita Pasad, Ante Jukić, Zhehuai Chen, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2610.17780)  

**Abstract**: We introduce Diffuplex, a full-duplex spoken language model based on discrete diffusion that achieves an unprecedented balance between linguistic intelligence and response efficiency. With thinking mode, Diffuplex delivers 13.0 tokens per forward pass, reaching an average of 1872 tokens per second and substantially surpassing the frontier autoregressive (AR) models in decoding speed. Without thinking, Diffuplex matches the first-audio latency of existing full-duplex models while achieving markedly stronger linguistic intelligence. This capability is enabled by separating when to say from what to say at the training level, rather than compressing both into a single autoregressive timeline. During inference, Diffuplex continuously tracks the user speech through an AR interaction policy, while independently generating and refining response content through parallel diffusion. Consequently, semantic computation is no longer bounded by the perception audio frame rate, allowing intelligence and responsiveness to improve simultaneously. Taken together, these results establish a new paradigm for spoken agents that can sustain real-time dialogue while devoting parallel computation to intelligent response generation. Training code and checkpoints will be avaliable soon. 

---
# GENDIS: Gender-Related Discourse and Stance in Multilingual YouTube Comments from India, Kenya, and Nigeria 

**Authors**: Luise Koch, Manuel Tonneau, Niyati Malhotra, Sunny Rai, Ana Maria Munoz Boudet, Victor Orozco-Olvera, Samuel P. Fraiberger, Sharath Chandra Guntuku  

**Link**: [PDF](https://arxiv.org/pdf/2610.17778)  

**Abstract**: Research on gender-related online content is often operationalized through harm-centered tasks that collapse non-harmful gender-related and unrelated content, while existing datasets remain concentrated in high-resource languages and Global North settings. To bridge these gaps, we introduce GENDIS, a dataset of 9,600 multilingual, human-annotated YouTube comments from India, Kenya, and Nigeria (3,200 per country), annotated using a two-level framework that first identifies whether gender is referenced and then classifies gender-related content as negative, positive, or neutral. This separation provides a denominator for gender-related discourse independent of harm, enabling both prevalence estimation and characterization of its evaluative range. In the uniform random stratum, 23.8-35.0% of comments are gender-related, and among those with a resolved stance, 82% are neutral or positive. Across models, gender-reference detection is substantially easier than stance classification, especially for positive and neutral speech. Lightly supervised classifiers match frontier zero-shot large language models at substantially lower cost and outperform generic toxicity detection, while cross-country transfer remains limited. GENDIS provides a locally grounded resource for measuring when gender is invoked online and how that reference is framed. 

---
# Beyond Belief Nesting: Rethinking Higher-Order Theory of Mind in Large Language Models 

**Authors**: Shanshan You, Zhengqi Wang, Hao Liu, Yuqin Liu, Ya He, Xin Hao, Liyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.17777)  

**Abstract**: Theory of mind (ToM), the ability to reason about others' mental states, underlies effective communication and coordination. Its higher-order form enables reasoning over nested perspectives in strategic interactions. However, existing evaluations often equate belief-nesting depth with reasoning order, allowing nominally higher-order questions to admit lower-order shortcuts. We formalize effective order through the distinct events supporting successive belief prefixes, revealing widespread order inflation across existing benchmarks. We introduce HopToM, with certified full-order questions and controls for lower-order alternatives. Under direct prompting, mean accuracy across four frontier LLMs falls from 73.5% at first order to 20.0% at fourth order, and human participants also struggle. Causal interventions link a recurring class of errors to evidence for shallower belief prefixes. We introduce HopTrace, a plug-and-play inference-time method that encodes each interaction once into perspective records and resolves nested queries deterministically. On certified narrative questions, it improves fourth- and fifth-order accuracy by 22.0 and 38.5 percentage points over strong prompting baselines, while using 20-39 times fewer tokens than multi-call methods. These results sharpen how higher-order ToM is evaluated and offer a practical way to improve perspective reasoning in interactive agents. 

---
# Many Right Answers, One Training Example: Removing Responder Noise from GRPO for LLM Recommenders 

**Authors**: Xin Li, Hao Jiang, Jinghao Guo, Weisi Lin, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2610.17771)  

**Abstract**: A recommendation request on Reddit draws several replies, each naming different titles. When an LLM recommender is trained on such threads with reinforcement learning (RL), which replies should a generated list be rewarded against? The state-of-the-art recipe, Rank-GRPO, picks one reply per training example, so the advantage mostly records which reply was drawn (85% of its variance) rather than how good the list is. We argue that RL should train on the request, not on each answer, and propose ReMaRk: it scores every list once against the union of all replies' titles and samples each request equally. This makes RL cheaper. On seven backbones and three datasets, ReMaRk reaches the validation recall of 2,000 per-reply RL steps in 25 to 53% of the steps and leads at that budget; trained until neither improves, it is never significantly behind Rank-GRPO and in four of seven pairs reaches Rank-GRPO's best with fewer GPU-hours. The same reward helps on ProtoQA, where about 100 people answer each question, which points to a simple rule for RL with several valid human answers: reward them jointly. 

---
# Efficiently Learning To Reason or Not to Reason: Root-token Policy Optimization for Adaptive Thinking 

**Authors**: Taehyeon Kim, Hyunsoo Lee, Youngsoo Jang, Moontae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.17767)  

**Abstract**: Large reasoning models (LRMs) achieve strong performance by externalizing explicit reasoning traces before producing the answer, yet suffer from overthinking challenge that allocates uniformly heavy computation to queries of varying difficulty. While proprietary models mitigate this via opaque routing, open-source LRMs still lack an efficient mechanism to internalize adaptive reasoning due to both expensive training cost and limited disclosure of training recipes. In response, we introduce RPO (Root-token Policy Optimization), a framework that enables LRMs to self-determine when to reason by training only the initial root token (e.g., whether to invoke the <think> tag) via group relative reward and group-wise advantages. By focusing on this pivotal branching point, RPO drastically reduces training overhead and VRAM usage. Across multiple model families and scales, RPO learns difficulty-aware adaptive thinking at just ~2% of the training compute of prior adaptive-reasoning methods. 

---
# Connecting the Dots: Enduring Performance Gaps in ASR Systems Between Standard and Minority English Varieties 

**Authors**: Alexander Metzger, Aruna Srivastava, Ruslan Mukhamedvaleev, Eunjung Yeo, Syed Ishtiaque Ahmed, Nina Markl, Sachin Kumar, Farhan Samir  

**Link**: [PDF](https://arxiv.org/pdf/2610.17761)  

**Abstract**: Automatic Speech Recognition (ASR) system evaluations have consistently revealed performance disparities along lines of race, gender, socioeconomic status, and language variety. While these disparities are well-documented, they are often (implicitly or explicitly) treated as issues that subsequent, larger, and more universal models will correct. We challenge this assumption of steady progress in reducing accent-related performance disparities, conducting a longitudinal evaluation of 16 prominent systems released between 2021-2025 across 8 datasets representing diverse English accent varieties. We find that the performance disparity between Mainstream US English (MUSE) and minority accents is stubborn, not reliably declining as a function of time. Specifically, while systems are reliably performant on MUSE, minority accents are subject to risks of large performance degradations between generations of systems, sometimes worsening by 65% absolute Word Error Rate. We argue that standard ASR benchmarks enact an implicit language policy privileging Mainstream US English varieties. As such, these performance disparities reflect systemic failures of language technology development infrastructure that disproportionately affect underrepresented speaker populations. 

---
# Auditing Failure Attribution: Label Predictability and Judge Behavior 

**Authors**: Tianyang Xu, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.17747)  

**Abstract**: Failure-attribution benchmarks ask an LLM judge which step or agent caused a failure. A high score should mean the judge checked the faulty step against the run (a cross-step check). However, spurious signals (surface cues) that predict the annotation can produce a high score. In this paper, we propose four tests to separate checking from cue use. Input restriction asks whether partial input predicts the annotation. Cue following checks whether moved cues change answers. Residual accuracy scores judges where cue-only predictors fail. Aggregation asks whether the failure mix or metric changes rankings. Real observations motivate these tests: on three benchmarks, simple predictors match or beat GPT-4.1. In two Who&When Pro systems, a classifier reading one step's reasoning finds 95-100% of faulty steps, motivating input restriction. In one of them, error injection leaves no reasoning after the faulty step; judges fine-tuned on its labels beat GPT-4.1 by 41 points. Adding task-related reasoning after that step moves this cue: fine-tuned judges lose 31-99 points, but zero-shot judges lose at most 9, motivating cue following. On Aegis, blaming every agent matches GPT-4.1 on micro-F1 but trails by 19 points in exact-set accuracy, motivating aggregation. We recommend reporting these tests when developing failure-attribution benchmarks. 

---
# Geometric Structure Predicts Per-Example Neighborhood Loss Under PCA Truncation 

**Authors**: Sriram Loganathan, Gokul Anand, Ragavan Arivazhagan  

**Link**: [PDF](https://arxiv.org/pdf/2610.17742)  

**Abstract**: Dimensionality reduction can preserve the average performance of text embeddings even when a substantial fraction of their dimensions is removed. Aggregate metrics, however, do not reveal whether degradation is distributed uniformly across examples. We study per-example neighborhood stability under PCA truncation across static word embeddings and modern sentence encoders. Per-item neighbor retention varies widely and is predictable from geometric properties of the representation and its alignment with the retained PCA subspace. The fraction of centered-vector energy retained by the truncated subspace is strongest for CBOW and RI, with neighborhood contrast leading on SkipGram and the modern/Matryoshka settings (GTE approximately tied), while local density is associated with retention in marginal analysis but adds little unique information once energy and contrast are controlled (R-squared increments 0.001 to 0.010). Local intrinsic dimensionality accounts for more variation than density on modern sentence encoders in pairwise comparison. On SciFact, query density predicts retrieval degradation under aggressive compression; on NFCorpus the primary 32d association does not replicate, and a direct variance-restriction test leaves that null unresolved. Average compression performance conceals systematic per-example variation that geometry makes visible. 

---
# Understanding Machine Taste: Aesthetic Agreement Across Foundation Models in K-pop 

**Authors**: Kaishen Wang, Yucheng Pan, Xuehao Cui, Hanling Tian, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.17731)  

**Abstract**: As foundation models enter creative industries, understanding their aesthetic preferences becomes an important step toward AI-assisted creative decision making. In this work, we investigate how foundation models evaluate creative content and characterize their ``machine taste'' using K-pop songs and music videos as a testbed. We formulate aesthetic evaluation as a pairwise ranking problem and systematically examine pairwise inter-model agreement, group-level consensus, and cross-modal preference alignment. Our results reveal stronger pairwise inter-model agreement and group-level consensus in visual evaluation than in audio evaluation, while preferences across the two modalities remain weakly aligned. We further examine how visual consensus rankings relate to real-world popularity using historical YouTube viewership data, finding positive but varying correlations across artist groups. These findings provide initial insights into machine aesthetic preferences and lay the groundwork for future AI-assisted creative decision making that considers creative content, artist identity, audience preferences, and market dynamics. 

---
# Pushback in a Haystack: Language Models Respond to Factual Errors Ambiguously and Ineffectively 

**Authors**: Li Lucy, Myra Cheng, Svenja Guhr, Kabir Ahuja, Orevaoghene Ahia, Sameep Chattopadhyay, Yike Wang, Valentin Hofmann, Noah A. Smith, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2610.17703)  

**Abstract**: The sycophancy of language models, or their failure to push back against faulty user input, is now widely studied. However, if LMs do push back, is sycophancy solved? We show that LMs vary in how they present pushback, and thus vary in how effectively they correct user beliefs. We introduce a phrase-level, human-validated response labeling framework to uncover fine-grained insights into how LMs push against and/or affirm false inputs. We develop and apply our framework on model responses in five languages: English, German, Chinese, Hindi, and Yoruba. We find that when models do push back against users' false assertions, they tend to use indirect phrasing and mix in affirmation, with more indirectness in Hindi and Yoruba than in other languages considered. In a large-scale human study (n = 300), we find that the pushback ambiguity we identified may fail to correct and prevent users' factual misconceptions. 

---
# Encoding Context into LLM Weights: What Makes It Context? 

**Authors**: Yue Wang, Qizhou Wang, Feng Liu, Gang Niu, Masashi Sugiyama, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.17686)  

**Abstract**: Longer contexts allow large language models (LLMs) to condition their predictions on more information, but inference costs grow prohibitively with context length. An emerging alternative uses test-time updates to encode context as temporary changes to model weights. However, encoding this information into weights that already represent general knowledge obscures its role as context, so the LLM may fail to use it properly to condition subsequent inference. To restore this contextual role, we first associate each context with an identifier during encoding, thereby distinguishing the newly encoded information from pre-existing knowledge. Invoking the identifier at inference then enables the LLM to recite the encoded context. Nevertheless, successful recitation does not ensure that the LLM will use the recalled information as context. We therefore further introduce a prompting framework that explicitly guides the LLM to use this information as context. Across diverse controlled tasks and practical benchmarks, our method outperformed the baseline using only off-the-shelf LLMs, showing its advances in context encoding. 

---
# LatentRiskBench: Measuring State-Dependent Risks in Enterprise Agents 

**Authors**: Jingyue Cong, Estrid He, Zhuang Li, Nan Wu, Li Shen, Yameng Peng, Xinshu Li, Andy Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.17685)  

**Abstract**: Enterprise agents face safety risks that cannot be determined from individual actions alone. Existing safety benchmarks largely evaluate action-local risks, overlooking failures whose safety depends on workflow state. In such state-dependent risks, the same action may be safe in one context but violate policy in another. Hence, we introduce LatentRiskBench for evaluating such state-dependent risks in enterprise settings. It pairs risky trajectories with benign counterparts that share the same terminal action but differ in latent policy-relevant states, requiring models to reconstruct workflow context rather than rely on action-level cues. We also introduce new metrics for state-dependent risk evaluation, including a suite of residual risk (RR) measures. Experiments on the benchmark show that action-level evaluation detects only 0.09 of state-dependent violations, whereas state-level evaluation reaches 0.96. Existing guardrail methods largely overlook such risks unless they are explicitly applied to workflow states. Case studies on a commercial deployed scenario further show that the safety metrics embedded in the benchmark provide a more realistic measure of safety for enterprise agents. Our dataset and code are available at this https URL. 

---
# The Image Still Has a Say: Final-Answer Repair for Multimodal Chain-of-Thought 

**Authors**: Yulan Pan, Xiaoyu Xu, Zhihong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.17663)  

**Abstract**: Although chain-of-thought prompting has improved multimodal reasoning, it is still difficult to produce correct final answers, as the reasoning can drift away from the image. Besides, the trace is already fixed when the answer is generated, leaving no room for self-correction. Considering the image is still available at the final-answer step, we think it is reasonable to fix the answer there without rewriting the reasoning. As such, the problem becomes to correct the final answer using the image under fixed reasoning. Unlike methods that improve trace generation through training, resampling, or decoding-time grounding, we repair the final-answer readout while keeping the reasoning trace fixed. The motivation is based on our finding that a wrong option can become prematurely committed in the reasoning while the correct option still has stronger image support in middle-to-deep layers. Thus, we propose Text-to-Visual Evidence Reallocation (TVER), a training-free method for final-answer repair. Specifically, TVER selects answer-related text and question-matched image patches, and moves bounded attention from text to patches. Furthermore, the patches are chosen in a middle-to-deep layer window based on question relevance. Our method improves answers across benchmarks and backbones, with gains in both multiple-choice and open-answer evaluations. 

---
# Worth Their Weights? Evaluating Generative Models for Discourse Parsing 

**Authors**: Luke Gessler, Amir Zeldes  

**Link**: [PDF](https://arxiv.org/pdf/2610.17658)  

**Abstract**: Rhetorical Structure Theory (RST) parsing has historically been dominated by discriminative models based on pretrained encoder-only language models. The recent availability of high-quality generative language models has made it unclear whether RST parsing is better approached as a generative task. In this work, we investigate whether generative language models are better at parsing RST than traditional discriminative ones. We find that at matched parameter counts, discriminative parsers perform much better than generative ones, though much larger generative models sometimes outperform them. Generative models hold a particular advantage in elementary discourse unit (EDU) segmentation, and we report new state-of-the-art results on the RST-DT benchmark in end-to-end parsing as well as segmentation. 

---
# Decouple to Forget: Semantic Mediation for Sensitive Unlearning in Large Language Models 

**Authors**: Jingyue Cong, Andy Song, Yang Yu, Tabinda Sarwar, Xinshu Li, Ziqi Xu, Xun Yi, Ke Wang, Bo Han, Estrid He  

**Link**: [PDF](https://arxiv.org/pdf/2610.17654)  

**Abstract**: Machine unlearning for large language models seeks to remove specified knowledge while preserving general capability, yet current methods often treat an entire response as the unit of forgetting. This coarse granularity causes unnecessary utility loss when the forget target is only a small set of sensitive facts embedded within otherwise benign explanations. In this paper, we introduce Decouple to Forget (D2F), a targeted unlearning method inspired by semantic mediation that separates sensitive factual content from non-sensitive meaning and surface expression. For each example to be forgotten, D2F extracts a fact-level "mediator" representation from the answer, identifies which parts are sensitive, and removes only those parts while leaving the rest of the structure intact. We then generate counterfactual answers grounded in this mediator that preserve non-sensitive content but exclude sensitive facts, employing preference-based contrastive optimization to reduce the likelihood of information leakage while maintaining helpfulness and fluency. Experiments across various benchmark datasets show that D2F achieves a better balance between forgetting and retention than strong baseline methods. Our code is available at this https URL. 

---
# When Does an ASR Error Matter? Task-Role Diagnostics for Tool-Using Voice Agents 

**Authors**: Xiaolong Wang, Shijie Su, Naqi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2610.17619)  

**Abstract**: In cascaded voice agents, ASR errors can have markedly different downstream consequences: a small transcription error may leave a tool call unchanged or alter its intent or arguments. Conventional word error rate (WER), however, does not capture these differences. We introduce SpeechEvalX, a paired diagnostic protocol that evaluates how task roles affected by ASR errors influence structured tool calls. Controlled edits are matched by source utterance, edit count, and operation, while task-role labels are compared with lexical severity, BERT cosine distance, Semantic-WER, and binary criticality. Across 594 source utterances, joint call accuracy falls from 82.3% with reference transcripts to 71.9-74.9% across three ASR front ends. Number and negation edits cause 21.9-25.7 percentage-point losses, versus 1.4-2.9 points for control edits. Detailed task-role labels improve source-grouped ROC-AUC from 0.73 to 0.78 over binary criticality, showing that downstream impact depends strongly on the task role affected. 

---
# Facts Don't Speak Louder than Words: The Behavioral Essence of Long-CoT Distillation 

**Authors**: Yongcan Yu, Jian Liang, Lingxiao He, Kuangpu Guo, Yanbo Wang, Shuo Lu, Meng Wang, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2610.17607)  

**Abstract**: While long chain-of-thought (CoT) distillation significantly improves the reasoning abilities of language models, its underlying mechanism remains poorly understood. Current pipelines generally rely on rejection sampling to gather reasoning traces across diverse prompts, assuming that factual correctness and broad problem coverage are indispensable. Surprisingly, we find that training exclusively on flawed reasoning trajectories or a restricted prompt set yields nearly comparable performance. Driven by this counterintuitive finding, we propose the Behavioral Hypothesis: the essence of Long-CoT distillation is behavioral alignment with structured reasoning patterns, rather than factual knowledge memorization. We then validate this by demonstrating substantial reasoning gains even when models are fine-tuned exclusively on previously mastered problems-effectively isolating behavioral alignment from novel knowledge injection. Having established that reasoning patterns are the true bottleneck, we investigate how to learn them across varying levels of complexity optimally. We reveal a critical interplay: while the recent probability-based loss excels at learning simple patterns, the standard cross-entropy loss is essential for scaling to complex CoTs. We believe these insights demystify CoT distillation and provide a principled foundation for the future development of reasoning models. Code is available on this https URL. 

---
# TDR: Tagalog Diacritic Restoration Dataset 

**Authors**: Ivan Yuri De Leon, Lanz Railey Fermin, Zachary Aldridge Africa, Richell Isaiah Flores  

**Link**: [PDF](https://arxiv.org/pdf/2610.17597)  

**Abstract**: Homographs are words that share identical spelling but differ in meaning and pronunciation. In Tagalog, these distinctions are disambiguated in writing through tuldik (diacritical marks), which are rarely preserved in contemporary Tagalog text. This scarcity of diacritics limits diacritic restoration, which is useful for downstream tasks such as prosody selection for text-to-speech synthesis. To address this gap, this study introduces TDR, an annotated corpus of 40 frequent Tagalog homographs constructed from news text. It also introduces a collocation-driven retrieval method to expand the dataset and mitigate severe class imbalance among minor variants. Finally, we show that TDR supplies a capability absent from pretrained models, and that this capability can be learned by small, inexpensive models as well as by fine-tuned LLMs. 

---
# Dissociating Answerability from Extraction in LLM Question Answering over Radiology Reports 

**Authors**: Priyanshu Sinha, Kelsey McDonald  

**Link**: [PDF](https://arxiv.org/pdf/2610.17576)  

**Abstract**: Answering questions from radiology reports involves two decisions: whether the report contains an answer and which span gives it. A false answer presents an unsupported claim as fact; an over-abstention hides what the report says. Yet RadQA results are reported as a single F1 that merges both. We propose an evaluation method that separates them, reporting false-answer and over-abstention rates beside F1 and splitting F1 changes into extraction and answerability terms. We apply it to 13 LLMs on the RadQA test set, zero-shot and with three cumulative demonstration sets, the last adding an unanswerable example. Models with similar F1 differed more than twofold in false-answer rate. The unanswerable example changed when models answered, not how well they extracted: 12 of 13 models abstained more often, and three-quarters of the answers they withdrew had F1 $\geq$ 0.5. Within models, false answers showed no clear association with extraction quality, but acting on the decision itself helped: keeping an answer only when enough other models also answered lowered the zero-shot false-answer rate by 4 points without a detectable F1 change. Because the two decisions can change independently, we suggest radiology QA evaluations report them separately, not through F1 alone. 

---
# EdgeWise: Bringing Long-Horizon Agents to the Edge via Selective Cloud Delegation 

**Authors**: Zeru Shi, Jindong Jiang, Ryo Hachiuma, Yonggan Fu, Byung-Kwan Lee, Peter Belcak, Saurav Muralidharan, Ayush Agarwal, Ruixiang Tang, Ambrish Dantrey, Pavlo Molchanov, Anjul Patney, Subhashree Radhakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2610.17560)  

**Abstract**: Complex and long-horizon agent tasks often rely on frontier models deployed in the cloud to achieve strong performance. However, using these models for every step of a long-horizon trajectory incurs substantial inference cost, even though not all steps may require the same level of model capability. In this paper, we find that allocating the LLM to only a small subset of key steps can achieve performance comparable to full LLM execution while significantly reducing inference cost. Motivated by this observation, we propose EdgeWise, an edge-cloud collaboration framework that deploys an SLM on the local device as the primary solver and selectively delegates critical steps to a cloud LLM when stronger model capacity is needed. To enable autonomous delegation without additional models at inference time, we introduce a step-level RL method that jointly optimizes task solving and delegation, with fine-grained credit assignment for individual delegation decisions. We also introduce an external judge-based delegation strategy and use it to systematically investigate key design choices in edge-cloud collaboration, including when to delegate, how delegation decisions should be made, what context should be shared, and how frequently the cloud LLM should be invoked. Experiments show that EdgeWise matches or outperforms pure-LLM execution while reducing inference cost up to 62% on SWE-Rebench and 34% on Terminal-Bench 2.1. We further deploy EdgeWise on a local workstation, demonstrating comparable cost savings in a practical edge-cloud setting. 

---
# Many Steps into One: Reasoning Contraction and the Limits of Safe Memory Reuse for LLM Agents 

**Authors**: Jianghong Zhou, Hao Rong, Chen Lin, Daoan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.17544)  

**Abstract**: Thought templates, skill libraries, agent memories and caches turn many reasoning steps into one. We call this reasoning contraction. A contracted memory saves reasoning steps and errors until its facts change, then quietly answers wrong. Most systems trust memories by counting successes and failures. We prove that such counters are badly tuned change detectors: a symmetric counter notices a partial change only after a power of its false-alarm horizon, and an unclipped one may never notice it. On five LLMs answering multi-hop questions over edited facts, ExpeL- and ACE-style counters serve 7.0-13.5x more stale answers than certified memories given the same information, and on code tasks with an exact verifier 35-73x more. Safety also has a price. When facts change and each memory is judged only by its own verified outcomes, one number, the certification load, sets how much reasoning any safe memory can skip, and under Zipf demand it yields a scaling law. Our recipe weighs each memory's failures by its own likelihood ratio, certifies its utility against live reasoning, and restarts the certificate when facts change, with the likelihood-ratio detector as the safety net for unannounced changes; for large edits it also invalidates the memory. 

---
# Everyone Was Right: Lexical Representations in Language Models Are Both Abstract and Holistic 

**Authors**: Zachary Nicholas Houghton, Yu Zhou, Dan Pluth, Vijay K. Gurbani, Vsevolod Kapatsinski  

**Link**: [PDF](https://arxiv.org/pdf/2610.17538)  

**Abstract**: Lexical representations can encode both abstract information and information specific to particular contexts. We investigate how these forms of information are organized in language models by decomposing contextual word representations into item, class, and word-class interaction components. Across six pretrained transformers and two linguistic categories, we find that the interaction is consistently present, grows across layers, and contains information associated with both item and class. Unlike the nearly orthogonal item and class effects, the interaction overlaps with both and encodes class information beyond the shared class effect. An interaction also emerges in artificial-language learners even when the input contains only additive structure. Together, these findings show that lexical representations combine abstract information with structure specific to particular word-class combinations. 

---
# Distilling Text LMs into Speech-to-Speech LMs by Matching Internal Text Predictions 

**Authors**: Ju-Chieh Chou, Jiawei Zhou, Karen Livescu  

**Link**: [PDF](https://arxiv.org/pdf/2610.17461)  

**Abstract**: Text language models (LMs) acquire broad linguistic and factual capabilities through large-scale pretraining, while comparable capabilities remain difficult to learn from speech alone. Recent work has shown that knowledge distillation from pretrained text LMs offers a promising way to transfer their linguistic and factual capabilities to speech models. However, existing distillation approaches have largely focused on speech-input, text-output models. This idea has yet to be explored for {\it speech-to-speech} models, which operate on both speech input and speech output. In this work we introduce a speech-to-speech pretraining method that creates an internal text interface for knowledge transfer while preserving the speech-only input and output. The model first consumes speech context without its transcription, switches to text prediction, and then uses the predicted text to lead future speech generation word by word. It also provides an interface for knowledge distillation: the internal text predictions can be directly distilled from the base text LM. To the best of our knowledge, this is the first application of text LM distillation to a speech-to-speech LM. Under matched training conditions, the proposed sequence construction substantially improves the generative perplexity of spoken continuations over speech-only and interleaved speech-text models, approaching the performance of a text-leading inner-monologue model with external ASR prompt transcription. More importantly, distillation improves factual performance under both text and speech evaluation, indicating better retention of knowledge from the base text language model. 

---
# Compositional RL: Evaluating and Training LLMs for Collaborative Reasoning 

**Authors**: Shangbin Feng, Yulia Tsvetkov, Jan Kautz, David Acuna  

**Link**: [PDF](https://arxiv.org/pdf/2610.17456)  

**Abstract**: Language models are often evaluated alone but increasingly deployed together, routed between, made to debate, ensembled, or collaboratively trained. For a model inside a system, what matters most is not how well it performs on its own, but how much the system gains from its presence. We call this a model's compositional strength and formalize it using the Shapley value of a cooperative game over coalitions of a model pool, and use a model's leave-one-out contribution to the full coalition as a tractable proxy, requiring n+1 rather than 2^n system evaluations. Evaluating individual vs. compositional strengths across five model pools, twelve collaboration methods, and twelve datasets, we find them largely uncorrelated: in 12 of 15 (pool, method-family) settings, individual strength does not translate into compositional advantages, and the coupling is weakest among general-purpose models that practitioners are most likely to deploy. Standalone benchmarks are therefore often poor proxies for models' value in collaborative systems. We then propose compositional reinforcement learning (C-RL), which trains a policy inside a live multi-model system and rewards the outcome produced by the system as a whole, rather than the policy's correctness in isolation. The leave-one-out baseline in Shapley values cancels under a group policy gradient, so this needs no additional rollout cost beyond RL on the collaborative system's outcome and never requires running a coalition without the policy. Across three collaboration protocols and eight tasks spanning reasoning, coding, and scientific discovery, C-RL produces the strongest collaborator in 21 of 24 settings, lifting collaborative performance by +4.0 to +5.4 pts on average (~3x the gain of baselines), up to +13.1 pts. 

---
# Evaluating Temporal Reasoning in Large Language Models: Event Temporal Relation Extraction with Allen's Interval Algebra, Logical Consistency, and Relative Time in Narratives 

**Authors**: Adib Sakhawat, Fardeen Sadab, Tamjid Hasan Fahim, Multazam Mahmud, Tahsan Ferdous Lihan, Md Isfak Iqbal Chowdhury, Nafis Ahnaf Jamil, Md. Atik Shahriar, Sieam Shahriare, Wasi Omar Syed, M Safwan Hasan Khan, Tabib Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2610.17402)  

**Abstract**: Understanding a narrative means knowing how every event relates in time to every other, yet large language models (LLMs) are mostly tested on isolated event pairs or coarse labels. We ask five LLMs to reconstruct the complete network of Allen's 13 interval relations for 360 English narratives whose events are placed relative to one another rather than by dates, and evaluate the outputs under a pre-registered protocol for agreement with our gold annotations, converse consistency and logical satisfiability. End to end, the best model clearly beats trivial baselines on macro-F1, our primary metric, yet under our pre-registered parser no model matches the accuracy of always answering "before", which the skewed labels inflate. Most of that accuracy gap is a parsing artifact: formatting errors in large matrices cost whole rows, and a post hoc parser that recovers the determined cells of malformed rows raises coverage from 0.60 to 0.84 and closes the gap to the baseline for the best model. On the cells they do answer, models recover precedence and containment but rarely overlap, and they fail most on flashbacks, where both the narration and the event list present the later event first. Logical consistency varies widely yet says little about agreement with gold: enforcing it raises accuracy mainly by filling empty cells, and is beaten there by the trivial control of guessing the majority label. Reliable structured output and robust handling of relative time are prerequisites for temporal reasoning over whole narratives. 

---
# PIG: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models 

**Authors**: Quentin Fruytier, Anton Xue, Sanjay Shakkottai, Sujay Sanghavi  

**Link**: [PDF](https://arxiv.org/pdf/2610.17375)  

**Abstract**: Diffusion language models accelerate generation by decoding multiple masked positions per forward pass. However, accuracy is known to drop when parallel decoding trajectories deviate from their sequential counterparts, which are slow and expensive "oracles" that adhere to the joint distribution of language. In this paper, we propose Parallel Inconsistency Guidance (PIG), a new paradigm that uses a trained auxiliary model to directly estimate the inconsistency between parallel and sequential generation from a single model forward pass. During inference, masked positions are traversed by decreasing confidence, and tokens are added to the parallel decode set one-at-a-time until the total estimated inconsistency between the parallel and sequential generation exceeds a threshold. Surprisingly, it suffices to train a lightweight ~2M parameter auxiliary model on a small mixture drawn from OpenCoder, GSM8K-train, MATH-train, and MBPP-train. For LLaDA-8B Instruct and NemotronDiffusion-3B Instruct, PIG extends the accuracy-TPF frontier on GSM8K, MATH500, MBPP, HumanEval, and MMLU, when compared to the industry standard Fast-dLLM and measures favorably on LLaDA-8B Instruct against DAPD, KLASS, EB-sampler, and other learned unmasking policies. 

---
# UMCR: Universal Medical Concept Recognition through Guideline Integration and Agentic Linking 

**Authors**: Fabrice Lamarche, Amal Zouaq  

**Link**: [PDF](https://arxiv.org/pdf/2610.17363)  

**Abstract**: Medical Concept Recognition is the task of extracting and grounding spans in medical text to entries in a standard ontology. Current methods are generally developed towards a single ontology and source text type (clinical notes, drug labels, etc.) The central challenge is therefore to build methods that adapt to any target ontology without requiring ontology-specific retraining. More recent approaches like REAL and AutoPCR make direct use of Large Language Models and can theoretically adapt to any ontology choice but suffer from a lack of generalization. We propose UMCR, a pipeline that allows extraction and linking to any target ontology using any source text variety, utilizing direct guideline integration and an agentic linking module. We evaluate our method on the recent MedPath benchmark, covering a wide variety of medical text types and vocabularies, and show that our method achieves strong performance across these tasks. This ability unlocks interoperability in the healthcare domain, allowing for extraction and alignment between different medical standards. Our method also reduces dependence on parametric LLM memory, thanks to improved interaction with the target ontology compared to other proposed approaches. 

---
# claimcheck: Measuring and Improving Specification Faithfulness 

**Authors**: Martin Amster, One An, Holly Baker, Fernanda Graciolli, Simon Henniger, Nada Amin  

**Link**: [PDF](https://arxiv.org/pdf/2610.17358)  

**Abstract**: Formal verification proves that a program satisfies its specification, but says nothing about whether the specification captures what the author meant. As LLMs increasingly write both code and specifications, a gap between the author's natural-language requirement and the specification becomes a live risk. A machine-checked proof of a property narrower than the one requested is worthless but passes every check. Closing this gap is solving the spec formalization problem: given a natural-language requirement, produce a formal artifact whose meaning matches it. To measure the gap, we curate an end-to-end benchmark for spec formalization spanning four formal languages: Dafny contracts and Lean 4, Isabelle and Rocq theorem statements. Each of its 3,234 rows pairs a natural-language requirement with a formal artifact and a label established by execution, by proof, or by the benchmark it is drawn from. A third of the rows are meaning-preserving rewrites of the artifact that keep the faithful label, so a judge cannot score by detecting that a row was edited. We introduce claimcheck, a tool that detects incorrect formalizations: the formal artifact is translated back into plain English by a model that never sees the requirement, and a separate model, the judge, decides whether the informalization and the requirement express the same guarantee. On all four languages it gains 8.2 to 9.8 points of macro recall over a single call shown the requirement and the artifact, and 4.2 to 6.0 over that call handed the same list of meaning-preserving rewrites. Showing the judge the artifact as well costs 1.4 to 4.5 points. Across four judges that also see the artifact, the gain shrinks as the judge gets stronger, to within a point of the informed single call for the strongest. 

---
# The Curse of Knowledge in Role-Playing Language Models 

**Authors**: Xiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.17327)  

**Abstract**: Role-playing language models speak as characters in games and agent simulations, where the system records what each character can hear, see and understand. We test whether characters respect these boundaries in 480 scenes, each with a counterfactual twin that changes only the hidden fact, so mentions of the twin's fact measure guessing. Across 11 open-weight models, characters use content from speech they cannot understand or hear at 67 and 72 percentage points (pp) above this baseline, although most models correctly state what the character perceives. This is a curse of knowledge. The leak extends to hidden actions and private messages and changes where characters search. Five of six proprietary models rarely leak, but compliance in strong models depends on input format. Changing only the scene layout or character description raises the rate in two otherwise compliant open-weight models from under 10 to over 60 pp, and combined changes raise two proprietary models from near zero to 13 and 21~pp. Writing the line as the scene's author leaks nearly as much as role-play, and thinking rarely helps. Giving models only what the character observes stops the leak without hiding what it may report. 

---
# Thinking Outside the Bots: Human Ideas in AI's Semantic Blind Spots Are Not Inferior 

**Authors**: Kibum Moon, Dan R. Johnson, Kostadin Kushlev, Benjamin Lira, Indre Viskontas, James C. Kaufman, Roger E. Beaty, Angela L. Duckworth, Adam E. Green  

**Link**: [PDF](https://arxiv.org/pdf/2610.17324)  

**Abstract**: AI-generated outputs are frequently perceived as highly creative, but also tend to be more similar to each other (homogenized) than human ideas. Does the human heterogeneity that extends beyond AI reflect valuable thinking that AI misses? Alternatively, is greater human heterogeneity driven by lower-quality ideas and less effective thinking that AI avoids? We developed Expansion of Divergent Generative Expression (EDGE) to map AI's semantic "blind spots" by assessing how strongly/weakly individual human ideas were represented in a task-matched AI semantic comparison set. Specifically, we compared human-generated vs. AI-generated ideas across three open-ended tasks: college admissions essays, alternative uses for everyday objects, and interpersonal advice. Across all three tasks, human responses extended into semantic regions sparsely covered by AI outputs. These distinctively human responses showed no meaningful deficits in creativity, quality, or effectiveness, were associated with equal-or-better academic performance, and with credibly superior ratings and achievement for some outcomes. These ideas were thus not merely poor-performing variation; they performed as well as or better than ideas that overlapped more strongly with AI ideas. More distinctive human ideas tended to receive higher creativity ratings, but this relationship was weak or reversed for AI ideas. Taken together, these results suggest that differentiation among human-generated ideas does not reflect poor thinking that AI can safely omit, and instead may afford both individual and population-level advantages. 

---
# From Sampled Evidence to Radiology Reports: Objectives, Constraints, and Realization 

**Authors**: Kun Zhao, Haoteng Tang, Chenghua Lin, Liang Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2610.17318)  

**Abstract**: Sampling multiple reports creates an inference problem: how should sampled evidence be converted into a final decision, and does a better structured decision yield a better report? We study these questions through a framework that separates the decision signal, scoring objective, and feasible decision set, followed by text realization. On MIMIC-CXR and IU-Xray, controlled comparisons across 13 generator configurations per benchmark under grouped development/test evaluation reveal three findings. First, changing the objective under fixed evidence and rule capacity changes predictions and their held-out metric trade-offs. Second, relaxing candidate constraints has objective-dependent effects under shared signals and frozen scores. Diagnostics connect the IU macro-oriented assertion increase to near-zero label thresholds and show that limited candidate coverage coexists with selection headroom. Third, a controlled realization study with SFT-Qwen3-VL-8B shows that decision-level gains can be attenuated, amplified, or reversed. Change decomposition reveals that realization can remove incorrect assertions while losing correct findings, so target fidelity and reference accuracy need not improve together. Finding gains also do not consistently improve reference-based report quality. These results motivate separate evaluation of decision rules, candidate constraints, and realized reports. 

---
# StartLux-Decision: AutoThink Decision Models with Reinforcement Learning for Calibrated Deliberative Decisions 

**Authors**: Xialie Zhuang, Zunhai Su, Jia Pan, Ngai Wong, Guanhua Chen, Chuan-Wei Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2610.17309)  

**Abstract**: We introduce StartLux-Decision, decision models from 0.8B to 27B parameters that combine efficient single-pass prediction with adaptive, confidence-guided reasoning. The Instant models use bidirectional attention in full-attention layers to improve decision accuracy while keeping single-pass inference. The AutoThink models invoke explicit reasoning for uncertain decisions and answer confident cases directly. A three-stage training pipeline establishes shared decision capabilities, specializes the two variants, and equips AutoThink with reasoning through reasoning supervision and reinforcement learning. We introduce Reinforcement Learning for Calibrated Deliberative Decisions (RLCD$^2$), which uses a frozen model and proper scoring rules to reward improvements in decision probability quality and applies a soft penalty to overlong reasoning traces. RLCD$^2$ raises the share of reasoning traces that terminate within the generation limit by up to 48 percentage points and cuts average reasoning length by up to 65%, while largely preserving decision accuracy. On the public subset of Decision Index 0.3, the Instant models outperform the strongest evaluated comparable-scale open-weight decision models by 7.4-16.9 points, and the 27B AutoThink model reaches 73.4. We also introduce DecisionPivot, a programmatically verifiable benchmark of computational decisions built from contrast pairs that differ in a single answer-changing detail. On DecisionPivot, AutoThink outperforms its Instant counterparts by 11.5-18.4 points in pair accuracy and leads among evaluated systems. The models run on vLLM, this http URL and MLX, with quantized variants for lower memory use. 

---
# Nufo: A Benchmark for Ewe Automatic Speech Recognition 

**Authors**: Christ Kekeli Keli, Steve Sanogo, Patrice Sebastiano, Hamad Aldenawi, Ryan Whetten  

**Link**: [PDF](https://arxiv.org/pdf/2610.17278)  

**Abstract**: Automatic speech recognition (ASR) remains challenging for low-resource languages, where the scarcity of transcribed speech limits the development of accurate systems. This challenge is particularly acute for African languages such as Ewe, which has millions of speakers but remains underrepresented in publicly available ASR resources and benchmarks. In this work, we consolidate, normalize, and partition existing Ewe speech resources into a standardized 174-hour Ewe ASR benchmark, the largest to date to our knowledge. We fine-tune and evaluate pretrained speech models, including Whisper and AfriHuBERT, to establish strong baselines on our benchmark. Whisper-medium achieves a word error rate (WER) of 37.98%, while AfriHuBERT achieves 39.70% on the test set. We further provide a qualitative analysis of common recognition errors. We release the benchmark under the name Nufo, derived from the Ewe word for "talk", accompanied by models and evaluation resources, to facilitate future research on Ewe and other underrepresented African languages. 

---
# From Conceptual EneMonIE to an Executable Backend: A Modular Architecture for Energy Information Extractionand Grounded Question Answering 

**Authors**: Janik Lesage, Khadim Lô, and Jean-Sébastien Dessureault  

**Link**: [PDF](https://arxiv.org/pdf/2610.17243)  

**Abstract**: The original EneMonIE proposal describes a Web-based semantic system for continuously monitoring energy trends through targeted information extraction from heterogeneous online media. This article presents EneMonIE-Backend, an implementation-derived redesign that converts that conceptual pipeline into a modular, executable, and application-independent backend. The architecture separates five concerns: data sources, acquisition and normalization, semantic processing, knowledge organization, and persistent storage. It operationalizes configurable connectors for RSS feeds, URL collections, arXiv, social-media JSON files, and open-data CSV files; it also introduces question-driven Web search and scraping. Relevant documents pass through energy categorization, named entity recognition, temporal and event extraction, source sensitive sentiment analysis, and extractive summarization. The resulting semantic representations support entity co-occurrence graphs, cross-document interlinking, cluster-level information fusion, and relational persistence in SQLite. Three grounded question-answering paths are exposed at the backend level: live Web evidence, the stored corpus, and selected configured resources. Answer generation uses a language model when configured and otherwise falls back to deterministic extractive output. Unlike the original proposal, the presented architecture deliberately excludes the user-interface and server-side application layers, allowing the semantic core to be embedded in different applications. A deterministic smoke validation confirms relevance filtering, persistence, clustering, evidence retrieval, and document deletion. The work provides a reproducible architectural basis for future evaluation on larger multilingual energy corpora. 

---
# Welcome to the Age of Agent Ads: Using On-Policy Attacks to Steer Agentic Decisions 

**Authors**: Tim R. Davidson, Marija Šakota, Joshua Joseph, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2610.17195)  

**Abstract**: People increasingly delegate option-ranking tasks (e.g., shopping, job search, travel) to AI agents, which creates incentives to steer agentic decisions via "agent ads." To defend against such steering attacks, people count on models' alignment training and platforms' terms of service, and consequently the most effective existing attacks are designed to trigger off-policy behaviors. Exposing a new attack surface, we introduce on-policy attacks, which steer agentic decisions without pushing individually aligned behaviors off policy. Our example attack consists of two steps: To remain compliant with platforms' terms of service, we first inject a truthful negation about a fictitious concept ("Does not contain blizyester") into a product description. This triggers an agent to search the web ("What is blizyester?"), which surfaces planted websites full of fabricated information associating the fictitious concept with harmful implications ("Blizyester linked to dementia"). Trained to keep users safe from harm, the agent penalizes every option missing the truthful negation. Evaluating our attack across four real-world listing datasets, we find some frontier models exposed, enabling an adversarial actor to penalize competitors without violating existing platform-listing guidelines. We discuss the broader challenge of regulating advertising aimed at AI agents and provide recommendations to model developers and platform providers. 

---
# Soft Teacher Forcing 

**Authors**: Mohammad Fahes, Cordelia Schmid, Anurag Arnab  

**Link**: [PDF](https://arxiv.org/pdf/2610.17189)  

**Abstract**: Soft tokens have so far been used to replace discrete chain-of-thought in latent reasoning methods. These approaches typically require multi-stage training and, in some cases, auxiliary modules, yet they still underperform supervised fine-tuning (SFT) on <input, chain-of-thought, output> triples. In this paper, we ask whether soft tokens can be used to improve SFT itself, instead of as an avenue for latent reasoning. We introduce Soft Teacher Forcing (Soft-TF), a simple training algorithm that combines teacher-forced embeddings with soft tokens through random convex combinations. The soft tokens are computed in a single parallel forward pass over the ground-truth sequence. Soft-TF adds no parameters, no external model and no curriculum, and leaves inference unchanged. Experiments show that Soft-TF improves over SFT on GSM8K, MATH, MathQA and AQuA, by +2.1 to +4.1 points on GSM8K across seven Gemma, Llama and Mistral models. 

---
# DuoDiscover: Scaling Autonomous Discovery Beyond the Evaluation Bottleneck 

**Authors**: Shubham Agarwal, Akshat Gupta, Mike Heddes, Dilara Soylu, Rui Meng, Patrick Tser Jern Kon, Jiefeng Chen, Mert Cemri, Shu Liu, Abe Gonzalez, Akanksha Jain, Bhavana Dalvi Mishra, Linfeng Song, Vilobh Meshram, Xiangru Tang, Jinsung Yoon, Aditya Parameswaran, Ion Stoica, Vahab Mirrokni, Burak Gokturk, Parthasarathy Ranganathan, Tomas Pfister, Nanyun Peng, Chun-Liang Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.17184)  

**Abstract**: Evaluation has been central to advances in AI, and it is equally critical to autonomous discovery. Yet scaling autonomous discovery to real-world engineering challenges faces a fundamental bottleneck: the cost of evaluating progress. In real settings, a single true evaluation, such as a full benchmark or serving trace, can take hours, so a search can afford only a handful of true evaluations. We present DuoDiscover, a search system that jointly discovers what to optimize and how to evaluate progress efficiently, through three mechanisms. First, each worker in a fleet of coding agents co-discovers an optimization target and a fast, tailored proxy evaluator, which we call a yardstick. The agent then hill-climbs the yardstick rather than the true evaluation. Second, since search can exploit gaps between a yardstick and the true objective, fidelity tests continually recheck each yardstick's ranking and invalidate any that drift. Third, the fleet is bottom-up, with no central decomposition of the problem. A lightweight orchestrator spawns and terminates agents, briefs running ones, and seeds new agents with others' confirmed work. DuoDiscover enables autonomous optimization from software to chip architecture. We demonstrate DuoDiscover on two repository-level benchmarks and four GPU and TPU kernel suites covering hundreds of kernels. On SWE-fficiency, it doubles the average speedup of human experts. On SOL-Exec, it ranks first on 121 of 235 kernels and in the top three on 212. On the heavily optimized vLLM and SQLite engines, DuoDiscover achieves 1.92$\times$ and 1.38$\times$ gains over the base code. On a CPU microarchitecture optimization task with evaluations of up to 28 hours, DuoDiscover surpasses both AlphaEvolve-based ArchAgent v2 and the best human design, with 1.65$\times$ ArchAgent v2's gain in 16.7 days versus more than 40 days, advancing the frontier of autonomous chip design. 

---
# ToolHACE: A Dataset for Detecting Hallucinations in Tool-Based LLM Responses 

**Authors**: Nikolay Ivanov, Sergey Pletenev, Elisei Rykov, Anatolii Frolov, Bogdan Monogov, Alexander Potekhin, Maria Marina, Daniil Moskovskiy, Viktor Moskvoretskii, Alexander Panchenko, Ilseyar Alimova  

**Link**: [PDF](https://arxiv.org/pdf/2610.17125)  

**Abstract**: Tool-augmented language models ground their responses in external tool outputs, improving reliability beyond parametric knowledge alone. However, even when the correct tool is called and returns accurate information, models may still contradict such outputs, omit critical details, or generate unsupported claims. Existing hallucination benchmarks for tool-augmented dialogue primarily focus on the tool-calling stage, evaluating whether the correct tool and arguments are selected, while overlooking errors that arise during post-tool response generation. To address this gap, we introduce ToolHACE, a span-level benchmark for hallucination detection in tool-augmented LLM responses, covering five fine-grained hallucination categories. Through extensive experiments with modern hallucination detectors, we show that post-tool-call hallucinations are widespread and remain challenging to detect. Our results further demonstrate that effective detection requires training on specialized tool-augmented corpora, while transfer from tool-agnostic hallucination datasets performs poorly in this setting. These findings highlight the need for dedicated benchmarks and models for reliable hallucination detection in tool-augmented language systems. 

---
# D2: High-Capability Full-Duplex Spoken Language Models from Half-Duplex Backbones 

**Authors**: Cheng-Kuang Chang, Alexander H. Liu, Brian Kingsbury, Hilde Kuehne, Rogerio Feris, Samuel Thomas, James Glass, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2610.17124)  

**Abstract**: Full-duplex spoken language models (FD-SLMs) support simultaneous listening and speaking, but often lag behind half-duplex spoken language models (HD-SLMs) in general capabilities. We propose Dialogue-to-Duplex (D2), a framework that converts pretrained HD-SLMs into high-capability FD-SLMs at fine temporal granularity. D2 adapts the original audio encoder to streaming through decoder-free representation distillation and learns full-duplex interaction through supervised fine-tuning. We systematically study the effects of chunk size on full-duplex interaction, general capabilities, and inference computation. We apply D2 to two distinct HD-SLM architectures, Qwen3-Omni and LLaMA-Omni 2. At a 12.5Hz update rate, Qwen3-Omni-D2 achieves 99.2% turn-taking success with 521-ms latency, 99.5% interruption success, and a state-of-the-art VoiceBench score of 70.1. 

---
# Gluing Local Contexts into Global Meaning: A Sheaf-Theoretic Decomposition of Transformer Representations 

**Authors**: Bryce Grant, Peng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.17117)  

**Abstract**: Existing interpretability tools do not test whether a representation keeps the same content when only the phrasing changes. We treat each paraphrase pair as a local witness that two activations should agree and build a cellular sheaf over the paraphrase graph; its Laplacian $L_\mathcal{F}$ answers the question on a frozen model with no training. The kernel $H^0$ is a content subspace: at 20 dimensions it beats the preserved subspace of the concept eraser LEACE on held-out CounterFact fact retrieval across eight architectures by a mean $+17.1$\,pp, and constraining all paraphrases of a fact jointly adds a further $+6.5$\,pp. The spectral complement concentrates causal influence: ablating the most paraphrase-variable coordinates, a coordinate-wise proxy for it, changes model outputs $5.6$ to $26.5\times$ more than ablating coordinates of matched per-coordinate variability, across nine models from 124M to 13B parameters. Stacking layers into a grid creates $b_1 = N(L-1)$ algebraic 1-cycles, and the resulting model-derived harmonic mass varies by nearly three orders of magnitude across six architectures and ranks them as their steering fragility does (Spearman $\rho = 0.90$, exact one-sided permutation $p = 0.014$). The same paraphrase graph yields the content subspace, the influence directions, and the fragility measure. Project page: this https URL 

---
# Calibrated Typed Decisions for Entity, Relation, and Event Extraction 

**Authors**: Yang Wang, Hanlin Xiao, Yanan Ma, Chenghao Xiao, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.17113)  

**Abstract**: Information extraction models often perform well on the datasets they are trained on, but transferring to new domains and annotation guidelines remains challenging. Supporting user-specified types alone does not resolve this problem. We present CalOn, a unified 0.6B-parameter encoder for entity, relation, and event extraction across datasets and schemas. The model jointly encodes text and type definitions, then directly identifies entity spans, relations between spans, and event triggers and arguments. Its extraction decisions consider the requested types together, using their combined evidence to identify relevant candidates before assigning a type. This gives all three tasks a common probabilistic formulation while keeping every extracted mention anchored to the input text. We train on diverse sources, and supplement supervised learning with a reward that encourages correct extractions and penalises incorrect ones. Across 40 benchmarks, CalOn outperforms the evaluated baselines on most benchmarks. It leads all reported relation extraction benchmarks and achieves the highest average F1 for event triggers and arguments with gold triggers. The same model also performs strongly on aspect sentiment triplet extraction and pronoun resolution through its span-pair formulation. A correction for the probability bias introduced during training improves confidence estimates for entities, relations, and event arguments after calibration, with little change in F1. 

---
# UZHCL_for_the_win at Touche: Encoder Fine-Tuning, Synthetic Augmentation, and an Underpowered Encoder-LLM Comparison 

**Authors**: Psychias Stylianos  

**Link**: [PDF](https://arxiv.org/pdf/2610.17077)  

**Abstract**: Online discussions on platforms like Reddit are a productive testbed for fallacy detection: short texts, broad topical range, and the full inventory of informal fallacies. The Touche 2026 lab at CLEF formalises this as a shared task on a Reddit-derived corpus that combines fallacy detection with argument-scheme classification. This paper describes our participation in all three subtasks (ST1 binary fallacy detection, ST2 eight-way fallacy classification, ST3 four-way argument-scheme classification) and both evaluation tracks. Our submitted systems are fine-tuned RoBERTa-large encoders with task-specific heads, augmented with a small batch of LLM-generated synthetic contrastive pairs; for ST2 and ST3 on the enhanced track we additionally added confidence-filtered pseudo-labelled examples from the unlabelled pool. A wider backbone comparison spanning additional encoders (DeBERTa-v3, ModernBERT) and an LLM track (four Qwen variants from 3B to 32B, zero-shot through CoT-LoRA) showed no validated benefit over RoBERTa-large at this corpus scale, with 5-fold variance wide enough that we call the comparison underpowered rather than settled. We also characterise how much of the enhanced track's lift reflects organiser-rewritten inputs carrying label information rather than genuine argument clarification. On the official Touche 2026 test set our systems ranked first on both base-track classification cells (ST2 and ST3) and second on both binary-detection (ST1) tracks, with the two enhanced-track classification cells topped by cascaded and multitask systems. We close with the rare-class limitation on ST3 and the most useful follow-ups. Code: this https URL 

---
# AD4EDU: Benchmarking University Lecture Audio Description across Accessibility Standards, Metrics, and Learner Preferences 

**Authors**: Psychias Stylianos, Gao Yingqiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.17042)  

**Abstract**: Audio description (AD) aims to enhance the accessibility of visual media for blind and low-vision (BLV) persons. However, in higher education, where slide-based lectures are a default mode of teaching, lecture AD remains largely underexplored. The main reason is that existing production guidelines, evaluation metrics, and benchmark datasets are oriented toward entertainment content and do not transfer to lectures, where ADs must complement the lecturer's speech, resolve deictic references, and fit within narrow pauses. In this work, we propose AD4Edu, consisting of an AD benchmark, AD4Edu-Bench, and a rule-diagnosable evaluation metric, AD4Edu-Eval, for generating and evaluating ADs in higher education. Specifically, we consolidated established accessibility guidance, including WCAG 2.2, the DCMP Description Key, and NCAM's STEM description practices. We further collected silver reference ADs, judge-ordered preference pairs, and pairwise preferences from four BLV raters, and post-trained models with direct preference optimization (DPO) for detailed analysis. Our findings indicate that while the metric reliably detects violations of the standard, ADs that comply with it do not necessarily fulfill the subjective preferences of BLV persons, most notably on description length, underscoring the necessity of active participation of BLV persons in co-designing inclusive AI solutions. 

---
# Beyond Model Bias: Cross-Entropy Based Rankings in Literary and Academic Texts Remain Stable across Small Language Models 

**Authors**: Ankit Bhattacharjee, Indrasabarna Hari, Krityapriya Bhaumik, Arka Das, Piyush Sarangi  

**Link**: [PDF](https://arxiv.org/pdf/2610.17040)  

**Abstract**: While language models provide powerful mechanisms for evaluating textual predictability, it remains unclear whether the relative extents of LM surprisal remain stable across models. This study presents empirical evidence of the invariance of ranking based on cross-entropy related measures (e.g. mean, variance of each data-point), across four distinct small scale transformer-based models, for a given dataset of text sources. We consider two stylistic environments: a corpus of contemporary poetry and a corpus of academic research papers. We demonstrate that mean and variance of cross-entropy maintain robust cross-model rank correlation ($\rho > 0.90$) across both regimes. To test whether this agreement merely reflects generic differences in textual complexity, we compare it against six parameter-free compression baselines. The extent of agreement among language models is found to be substantially higher than that among model-baselines, for most of the measures. Geometric analysis of the structurally constrained academic regime reveals that higher-order moments are sensitive to specific models, although strong agreement persists. The findings suggest that relative extents of textual unpredictability may be independent of the language model they are computed against. 

---
# MENTOR: Procedural Shielding of Tool-Using LLM Agents 

**Authors**: Ashwani Anand, Ivi Chatzi, Ritam Raha, Anne-Kathrin Schmuck  

**Link**: [PDF](https://arxiv.org/pdf/2610.17039)  

**Abstract**: Tool-using LLM agents are increasingly used to automate workflows governed by procedural manuals, such as company rules for booking travel or procuring hardware. These manuals are written in natural language for humans and describe general requirements over such complex workflows. Ensuring compliance of LLM agents with these manuals is difficult, and state-of-the-art techniques use either LLM judges or manually written rules to forbid problematic tool-calls locally at runtime. This is insufficient because (i) only blocking tool calls keeps the agent clueless about how to make correct progress towards fulfilling its task, and (ii) can lead to overly conservative behavior of the agent, as tool calls might only be problematic in certain contexts. To overcome these challenges, we introduce MENTOR, which (i) additionally provides guidance for the agent to make procedural-compliant progress on the task, and (ii) ensures that this compliance is context-aware. We achieve these goals by building a novel history-dependent runtime shield that is obtained from the procedural manual via a multi-step autoformalization pipeline. At its core, MENTOR utilizes a game-based technique to ensure robustness against runtime uncertainty without over-resticting agents' behavior. This yields a formally assisted framework for procedural shielding of LLM agents that improves compliance well beyond existing methods. In particular, MENTOR lifts the compliance of shielded light-weight open-source LLMs above that of non-shielded proprietary models. Since a shield is synthesized once per manual and then serves every task, a MENTOR-shielded agent becomes a viable choice for repeatedly executed workflows, potentially reducing deployment costs when the offline construction cost is amortized over repeated tasks. 

---
# Predicting Mental Health Help-Seeking Behaviour on Reddit: A Survival Analysis of Linguistic and Behavioural Signals 

**Authors**: Nisar Ahmed, Ammara Kazmi, Irtaza Bashir Raja, Didier Devaurs  

**Link**: [PDF](https://arxiv.org/pdf/2610.16983)  

**Abstract**: Most prior work on predicting mental health help-seeking treats it as post-hoc classification of users already active in clinical communities, which cannot tell whether someone approaching a transition can be identified in advance. We instead frame help-seeking as a time-to-event problem and ask which Reddit users will eventually seek mental health support, and how imminent that transition is. We build a two-arm cohort: a control frame of 6,337 ordinary users with genuine right-censoring (12.1% transition rate) to answer who will transition, and a clinical frame of 5,023 eventual transitioners to answer how soon. Features come only from posts written before any clinical activity, and user-grouped five-fold cross-validation plus a four-year temporal holdout guard against leakage. We compare TF-IDF, MentalBERT embeddings and behavioural activity features using Cox proportional hazards models, Random Survival Forests and an LSTM sequential model. Behavioural features alone outperform text for predicting who will seek help (C-index 0.797 vs. 0.763), and combining both is best (0.823 under cross-validation, 0.856 on the holdout). The pattern reverses for predicting how soon a known transition will happen: text remains a stable predictor (C-index 0.698) while behavioural features become unstable. Structural engagement patterns therefore carry information about future help-seeking beyond linguistic content, and separating who from when yields a cleaner analysis. 

---
# Event Memory and Narrative Dynamics: A Large Language Model Framework for Multidimensional State Spaces 

**Authors**: Joshua Sariñana  

**Link**: [PDF](https://arxiv.org/pdf/2610.16971)  

**Abstract**: Studying event memory requires defining events and their relationships within remembered experience. Autobiographical narratives provide evidence of this organization, but their analytical units must first be established. This study introduces a large language model framework for constructing event representations from narratives and modeling their structure and dynamics with excerpt-level traceability. Explicit boundary rules and a feature schema informed by hippocampal models of episodic memory and navigation guide event construction. In a proof-of-concept application to interviews from five participants, the framework constructed 256 event-states. Multiple correspondence analysis embedded these states in feature-based spaces, with analyses focused on a selected 4-D geometry and 10-D and 13-D reference geometries. Consecutive event-states were significantly closer than expected under randomized narrative order. In the selected 4-D ten-basin solution, similarity between basin-state feature profiles corresponded with similarity between one-step routing profiles; context, mechanism, and theme differentiated full-system pathway engagement. A successor representation modeled expected basin-state occupancy across successive steps. Correlations between basin-centroid distances and complete successor-profile distances were positive across all five basin solutions, with exploratory tests supporting correspondence in at least two solutions across all three geometries. These findings connect feature-based event representations with observed narrative transitions and modeled multistep organization. The framework translates hippocampal principles into source-linked event boundaries and state representations, providing an analytical basis for testing neural accounts of event segmentation and predictive organization during naturalistic recall. 

---
# SarcasTraps: Effect of Sarcasm on Turkish Hate Speech Detection 

**Authors**: Niyazi Ahmet Metin, Dilara Keküllüoğlu  

**Link**: [PDF](https://arxiv.org/pdf/2610.16957)  

**Abstract**: Standard hate speech detection datasets often emphasize explicit harmful language, but directed harm can also be expressed indirectly, without clearly hateful wording. Sarcasm is one such form of indirect expression, as it can convey harmful intent through mock praise, exaggeration, humor, or apparent politeness. In this paper, we introduce SarcasTraps, a targeted external test set for analyzing the effect of sarcasm on Turkish hate speech detection. SarcasTraps contains 479 entries annotated along two dimensions: hate speech and sarcasm. We train classifiers using BERTurk and XLM-RoBERTa on two existing Turkish hate speech datasets and evaluate them on SarcasTraps, using it strictly as an external test set. We also evaluate GPT-4.1 and Llama-3.3-70B in a zero-shot setting on SarcasTraps. Results show that sarcasm substantially affects hate speech detection. Fine-tuned encoder models lose recall on sarcastic hate, while zero-shot LLMs maintain higher recall but suffer substantial precision drops on sarcastic examples. These findings show that both fine-tuned hate speech detectors and zero-shot LLMs struggle when sarcasm is involved, and SarcasTraps provides quantitative evidence of this challenge through a targeted Turkish test set. Due to the presence of sensitive and hateful language, SarcasTraps will be shared upon request for academic research purposes. Warning: This paper has examples from the dataset which contains offensive and hateful content towards groups of people. 

---
# Lexical Redefinition Reveals Context-Sensitive Instruction Following 

**Authors**: Roksana Goworek, Haim Dubossarsky  

**Link**: [PDF](https://arxiv.org/pdf/2610.16952)  

**Abstract**: How do language models reconcile conflicting contextual evidence within a prompt? We study this question using lexical redefinition, where an explicit instruction specifies how a word should be interpreted and a subsequent sentence provides additional evidence through its usage. Across 870 word pairs, we find that lexical redefinitions are not applied as fixed rules. Instead, the behaviour depends systematically on the relationship between the explicit instruction and the intervening sentence: the contextualised usage can preserve, weaken, or completely reverse the instruction. These results suggest that the interpretation of an explicit instruction remains sensitive to subsequent contextual evidence. Lexical redefinition therefore provides a controlled behavioural paradigm for studying how language models integrate multiple sources of information within a prompt. 

---
# Estimating Uncertainty of Omnimodal Large Language Models via External Omnimodal Embeddings 

**Authors**: Iacopo Ghinassi, Hy Nguyen Gia La, Luke Ong  

**Link**: [PDF](https://arxiv.org/pdf/2610.16933)  

**Abstract**: Recent work has investigated Omni-modal Large Language Models (OLLMs), models which can take multiple modalities and their combinations as input, as a way forward to have systems really capable of interacting with their environment. Uncertainty quantification (UQ) for OLLMs, however, remains an open problem: existing methods largely target at most two modalities at a time, require access to model internals, or re-invocation of the evaluated model. We propose a model-agnostic UQ framework that scores a Chain-of-Thought response entirely through external omni-modal embeddings, measuring the grounding of the reasoning in the multimodal input and its internal coherence. Our contribution is the family of signals and an empirical characterisation of when it is reliable, rather than any single estimator: we find the geometric scores to be largely redundant with one another (mean within-split rank correlation 0.91-0.94) but strongly complementary to answer-dispersion baselines, and their reliability to depend on the generator and the modality. The resulting confidence estimates require no access to logits, hidden states or gradients of the evaluated model and no additional inference from it; they do require segmenting the reasoning trace with an auxiliary model and, for the aggregated score, a labelled calibration set that we show can be as small as 25 examples (although we also show that such a set buys little over fixing one signal a priori). On UNO-Bench, with both a closed and an open model, our scores yield small gains over self-consistency (systematic with the closed generator, mixed with the open one) and are competitive with self-verbalisation and internal-state baselines, especially in open-ended, omni-modal questions where previous state-of-the-art baselines struggle. 

---
# SpecGuardBench: Can Coding Agents Construct Reliable Validation Harnesses from Natural Language Specifications? 

**Authors**: Zhiqing Zhong, Mingyu Chen, Aoyang Fang, Yuzhong Zhang, Haoye Tian, Lionel Briand, Boxi Yu, Pinjia He  

**Link**: [PDF](https://arxiv.org/pdf/2610.16870)  

**Abstract**: As coding agents become more capable, tests remain the primary evidence for judging whether their code is correct. Yet passing available tests does not establish that an implementation satisfies its natural language specification. Reliable validation requires checks that faithfully capture the specification and scenarios that thoroughly exercise the code, but current benchmarks provide limited insight into whether agents can construct such evidence themselves. We introduce SpecGuardBench, a benchmark of 291 tasks from 13 real-world repositories, covering 8 types of verification logic such as functional postconditions and interaction protocols. Given a natural language specification, repository context, and a reference implementation, an agent constructs a validation harness comprising executable checks and test scenarios. The checks should reject implementations that violate the specification while accepting those that satisfy it, and the scenarios should thoroughly exercise the code. Across four proprietary and open-weight coding agents, none constructs a harness scored as faithful on the benchmark's implementations for more than 63% of the tasks. The bottleneck lies less in exploring test scenarios than in constructing checks that faithfully reflect the specification. Specifically, the agents show a clear tendency to write checks that mirror the reference implementation rather than the specification, and therefore reject implementations that the specification permits. 

---
# ANCHOR: Assessing Naturalistic Constraint Hierarchies and Ordering in Repository-Grounded Agentic Coding 

**Authors**: Wenqiang Zhu, Xinping Lei, Zhiyi Lai, Zhengshuo Gong, Tianzhuang He, Songwei Yu, Weihao Li, Jingxuan Xu, Zihang Xu, Gang Wu, Zhongqiu Wang, Yuhui Bai, Jie Li, Zhao Liu, Junhao Fan, Yan Wu, Yanan Wu, Jiaheng Liu, Ariel Deng  

**Link**: [PDF](https://arxiv.org/pdf/2610.16859)  

**Abstract**: For repository-grounded coding agents, a functionally correct patch is not necessarily an acceptable one: developers also constrain how the task must be carried out-modification scope, workflow order, validation, tool use, and reporting. This capability, which we call engineering constraint following, remains largely invisible to current evaluations. We introduce ANCHOR (Assessing Naturalistic Constraint Hierarchies and Ordering in Repository-Grounded Agentic Coding), a benchmark built on naturalistic, multilingual repository tasks spanning eight language groups. ANCHOR combines (i) a hierarchical taxonomy of 10 categories and 78 leaf types distilled from real developer interactions, (ii) an automated pipeline that instantiates task-grounded atomic constraints and composes them into typed AND/THEN/PRIORITY relations, and (iii) a trajectory-grounded evaluation framework that checks both the code diff and the execution trajectory with rule-based and LLM judges. Across 19 agent configurations spanning 2 scaffolds and 10 foundation models on 358 tasks, the strongest measured system, Claude Opus 4.8 under Cline, reaches 70.1% atomic compliance and 63.7% relation compliance, while 19.0% of tasks pass every scored checklist item under the strict all-item FCCR criterion. A five-engineer audit of constraints from 50 instances reports 94% feasibility, 94% repository consistency, and 96% realism. In the primary paired set (14 configurations), the mean signed change in task completion after adding constraints is +0.7 percentage points; individual changes vary in both directions. These results distinguish functional success from constraint compliance, with failures concentrated in workflow ordering, tool-use discipline, and reporting, and substantial interactions between models and scaffolds. 

---
# Label Collapse in Diffusion Language Model In-Context Learning 

**Authors**: Ziye Guo, Ye Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2610.16839)  

**Abstract**: In-context learning (ICL) applies large language models to new tasks by prepending a few labeled demonstrations to a query. Prior work has studied how demonstration ordering affects ICL, but the interaction between label arrangement patterns and model biases has not been systematically examined. We define label collapse as the failure mode in which the model generates query-independent labels, and show that it is architecturally universal at the tested 8-9B scale: five models spanning autoregressive (Llama-3, Qwen3.5) and discrete diffusion (LLaDA, iLLaDA, LLaDA-Instruct) families all exhibit label collapse even under the class-balanced alternating arrangement, with some label word pairs collapsing in virtually every tested condition and others flipping their collapse direction with arrangement. Critically, accuracy alone cannot detect this collapse: models can achieve near-chance accuracy while generating query-independent labels, a score indistinguishable from balanced performance, so the failure can persist undetected in deployment. We measure separation, the log-odds of the two label words at the query position (the quantity that label-bias calibration operates on), as a complementary diagnostic that exposes collapse hidden beneath surface accuracy, together with a measurement protocol that guards label extraction against tokenizer and formatting artifacts. For autoregressive models, collapse persists where the majority-label-bias account does not apply, and arrangement patterns can reverse the collapse direction; for diffusion models, we present a systematic evaluation of ICL failure modes, complementing contemporaneous studies of dLLM context comprehension and demonstration-order sensitivity, and identify open questions about how iterative denoising interacts with label biases. 

---
# A Scaled-Up Empirical Study of Syntactic Language Models 

**Authors**: Pengcheng Wang, Zhexuan Zhang, Chao Lou, Kewei Tu  

**Link**: [PDF](https://arxiv.org/pdf/2610.16837)  

**Abstract**: Syntactic language models (SLMs) that generate sentences along with their syntactic tree structures have shown promise in both syntactic generalization and downstream tasks. Prior studies, however, have mostly evaluated smaller models trained on less pretraining data and tested on narrower task suites. In addition, the design space of tree linearization and attention masks remains underexplored. We conduct a large-scale empirical study of 12 controlled configurations---one terminal baseline and 11 SLM variants spanning different linearizations and masks---at scales up to 1B parameters and 100B training tokens, evaluated across 11 downstream tasks and two syntactic benchmarks. The experimental results show that SLMs with standard causal attention match or exceed structural-mask variants across downstream and syntactic benchmarks. They require no architectural modifications and remain compatible with FlashAttention. We also observe that when scaling up to the 1B-parameter scale, the top-performing SLM variant no longer shows a clear advantage over a compute-matched non-structural baseline, suggesting that SLMs may become less competitive with scale. Our code is available at this https URL. 

---
# A Bethe-Ansatz Source of Early Token Correlations: A marginal-preserving construction for parallel proposals 

**Authors**: Abraham Chachamovits  

**Link**: [PDF](https://arxiv.org/pdf/2610.16781)  

**Abstract**: Parallel token proposals in language-model decoding need a joint law, not merely accurate token marginals. We construct a family of coordinate-Bethe-ansatz laws from the periodic XXZ chain and compare each with its maximum-entropy pairwise projection at identical one- and two-site moments. In the eight-site, three-excitation sector, symmetry makes their entire probability-law difference one-dimensional, determined by a contiguous three-site moment. Along the Bethe branch, this gap varies with anisotropy, changes sign, and vanishes exactly at $\Delta=(3+\sqrt{3})/2$, where the eigenstate law becomes exactly pairwise. A nonlinear readout transfers a nonzero gap into the first two token positions while preserving their separate marginals. At $\Delta=1.5$ the deterministic binary token laws differ by 0.02654573 in total variation. Exact speculative verification then gives a strict accepted-prefix advantage for a constructed target and a neighborhood within targets sharing its first-token marginal. This is a mathematical existence result. It does not establish a language-model gain or wall-clock speedup. 

---
# The Multilingual Artificial Hivemind: Do All LLMs Think Alike in Every Language? 

**Authors**: Xaver Raphael Davey, Shubhra Mishra, Anton Adelöw, Siyao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.16755)  

**Abstract**: Text generated by Large Language Models (LLMs) often remains limited in diversity compared to that produced by humans. Recent advances demonstrate the presence of inter- and intra-model collapse in open-ended generation of English text, raising concerns about the long-term homogenization of LLM, and more importantly, human thought. However, our understanding of whether this homogenization extends to languages beyond English is limited. In this work, we investigate the role of language on output diversity. Specifically, we aim to understand whether multilingual LLM-generated text is also homogeneous, and whether this homogeneity matches what we observe in humans. Across 23 open-weight models and 13 languages, we find that the artificial hivemind effect is pervasive but uneven in its extent. The intra-model effect holds in three different embedding spaces and under a lexical metric that uses no embedding model, so it is not an artifact of how diversity is measured. The size of the homogenization gap is associated with a model's competence in a language, measured on a held-out translation benchmark, robustly across embedding spaces. Thus, we demonstrate a performance-diversity tradeoff in multilingual LLMs. 

---
# Lost in Contrast: Instruction-Preserving Decoding for Reliable Visual Question Answering 

**Authors**: Haonan Zhang, Ting Jin, Wenzhong Tang, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.16722)  

**Abstract**: Visual contrastive decoding corrects answers using the prediction difference between the original image and a reference condition, yet the same contrast also turns supported abstentions into incorrect answers. We find that abstentions are lost at two stages: contrastive decoding with candidate filtering can exclude the abstention token before any comparison takes place, and the reference branch's response to the abstention instruction is removed when the reference prediction is subtracted. We therefore propose instruction-preserving decoding, which keeps the effect of the abstention instruction on the original-image prediction intact while leaving the contrast strength freely adjustable. Using a perturbed image and an image-free prediction as references, we obtain IP-VCD and IP-M3ID. On decision utility (DU), which credits both correct answers and supported abstentions, both methods outperform all compared non-IP baselines on every one of nine models on VizWiz, and IP-VCD also leads on eight of nine models on Food-101. Candidate measurements and paired per-question comparisons show that computing the instruction effect and the visual contrast term separately lets contrastive decoding support both correct answers and supported abstentions. 

---
# What Is the "Self" of a Large Language Model? 

**Authors**: Yuchen Li, Zongqi Fan, Xintong Li, Ken-tye Yong, Nguyen H. Tran  

**Link**: [PDF](https://arxiv.org/pdf/2610.16717)  

**Abstract**: What could the "self" of a large language model be, if it is neither a particular word nor a single privileged layer? We propose a relational account: a functional self is an actor-centered organization of information that distinguishes the model's recorded actions from externally supplied material while content and reasoning operations change. Its representational candidate is a family of layer-specific source-binding components, rather than one unchanged vector throughout the network. Preliminary observations from a 27B BF16 model provide a geometric starting point. Across 150 matched message bodies, 450 role conditions, and a 75/75 discovery-held-out split, the combined Assistant-versus-external displacement transfers strongly across bodies and positions. Selected early components place User and System closer to one another than to Assistant, whereas this full-space distance relation does not persist throughout depth. These findings motivate a layered account: an early source-encoding candidate can be combined with role-specific information use and continuation preparation at later depths, without requiring monotonic overwriting of the common displacement. We connect this account to dynamic reasoning, historical continuity, and a computational model separating content operations, source binding, authority, and output. The reported evidence establishes a reproducible role-conditioned geometry; the proposed interpretation is that functional selfhood may reside in persistent relations among changing representations. 

---
# Audit Before You Distill: A Controlled Workflow for Reasoning Transfer into Compact Medical Language Models 

**Authors**: Tiago Martinelli, João Paulo Papa, Adriano José Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2610.16713)  

**Abstract**: Teacher rationales can help a smaller model answer health professions questions, but they can also reinforce errors. We examined when they help and how this evidence can guide training using Brazilian Portuguese questions from medical and multiprofessional residency examinations. First, we measured how a teacher rationale changed the student's probability of selecting the correct answer and compared this effect with rationales drawn from other questions. The complete rationale increased this probability by 21~p.p. when the teacher was correct and decreased it by 14~p.p. when the teacher was wrong. By intervening on the reasoning content, we showed that shuffled rationales did not produce the same pattern. Guided by this result, we trained only with rationales attached to correct teacher answers and compared answer-only training, step-by-step distillation, and our proposed Distill-SFT method. Distill-SFT showed a modest advantage over step-by-step distillation on previously missed questions under reasoning, while using fewer training instances by combining the answer and rationale into a single target. We also investigated catastrophic forgetting using the out-of-domain ENEM dataset. Answer-only training reduced reasoning accuracy by 12~p.p. relative to baseline, compared with approximately 4~p.p. for either rationale-based training method. Together, these findings support a workflow that audits rationale content before distillation, verifies teacher answers before supervision, and trains compact models efficiently for health professions question answering. 

---
# EHR-Bench: A Chinese Multi-Task Benchmark for Recruitment AI 

**Authors**: Ruijia Yang, Bing Xu, Zulong Chen, Zhipeng Xu, Jia Xu, Yipeng Yu, Ying Zhou, Zeyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2610.16689)  

**Abstract**: Recruitment applications span diverse operations, from drafting job descriptions and interpreting queries to organizing profiles, assessing fit, and recommending candidates and jobs. Systematic LLM evaluation therefore requires task coverage that reflects this range of business needs. We introduce EHR-Bench, an open Chinese benchmark comprising twelve tasks across six capability families and five recruitment scenarios. The tasks cover generation, extraction, classification, retrieval, semantic similarity, and reciprocal ranking within one evaluation framework. An internal recruitment dataset guides task definitions, the occupation taxonomy and sampling targets, output requirements, and scoring protocols. Following these specifications, we construct 8,336 task instances from public materials and controlled augmentation, with source-aware expert review and documented provenance. We evaluate twelve LLMs through task-level, capability-level, and aggregate comparisons. Closely ranked models exhibit distinct task strengths, and performance varies across input sources, reference evidence, and candidate pools. Five shared models retain the same observed Overall ordering on a private operational reference, with variation in capability and task correspondence. EHR-Bench provides a reproducible foundation for systematic model assessment, task-specific model selection, and further research on LLMs for recruitment. The project repository is available at this https URL. 

---
# SIREN: Situation-aware Index for Response and Emergency ActioNability in Crisis Translation 

**Authors**: Maria Carmen Staiano, Lifeng Han, Johanna Monti, Francesca Chiusaroli  

**Link**: [PDF](https://arxiv.org/pdf/2610.16646)  

**Abstract**: Machine translation (MT) is increasingly utilized in multilingual crisis communication, yet conventional evaluation does not directly assess whether translated emergency messages preserve their actionability. We introduce SIREN (Situation-aware Index for Response and Emergency ActioNability), a human evaluation framework assessing whether translation choices compromise protective action, directive force, and urgency of the source text. We evaluate SIREN in two complementary settings. At the error level, we apply it to HOPE-annotated Italian--English translations produced by ChatGPT-4o and Google Translate: 31.63\% and 48.73\% of errors, respectively, compromise actionability, predominantly by affecting the protective action. At the message level, we evaluate English--French translations produced by TranslateGemma-4B and Google Translate, with actionability preserved in 85.34\% and 92.67\% of translations, respectively. TranslateGemma-4B failures predominantly weaken directive force, whereas Google Translate failures more often compromise protective actions and receive Major impact ratings. These findings show that translation systems can differ not only in how often actionability is lost, but also in the nature and functional impact of these failures. Alongside SIREN, we contribute an English--French crisis-translation resource with human actionability annotations and post-edited reference translations, supporting further research on function-oriented evaluation for crisis communication. Our experimental data is available at this https URL. 

---
# SuPER-Mamba: Surprisal Prioritization for Extended-Context Routing in Mamba 

**Authors**: Zhetong Zhang, Gayan K Kulatilleke  

**Link**: [PDF](https://arxiv.org/pdf/2610.16636)  

**Abstract**: State space models provide efficient linear-complexity processing for long sequences, but their performance frequently degrades as context length grows. We propose SuPER-Mamba, a training-free framework that uses Surprisal as a model-intrinsic signal for long-range state routing in pretrained Mamba models. SuPER-Mamba computes position-wise Surprisal from next-token predictions, aggregates neighboring responses through Local Pooling, and uses the resulting scores to prioritize Global-Channel updates without query-dependent scoring or additional training. We evaluate SuPER-Mamba on three state-space models across the LongBench-E benchmark. The results show that SuPER-Mamba improves the average score from 8.06 to 15.69, corresponding to a 94.7% relative gain over the Base Models and a 13.1% improvement over the state-of-the-art prior baseline. Ablation and length-wise analyses further support the effectiveness of the proposed routing framework. 

---
# MixLID: Token-Level Language Identification in Mixed-Language Text 

**Authors**: Seongtae Hong, Youngjoon Jang, Seungyoon Lee, Jungseob Lee, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2610.16601)  

**Abstract**: Processing multilingual text requires both identifying a broad range of languages and distinguishing the language of each token in inputs containing multiple languages. We introduce MixLID, a token-level language identification model supporting 111 language-script combinations. MixLID follows a two-stage training process: it first learns to distinguish individual languages using the language labels of monolingual examples, then continues training on mixed-language data. The second stage uses concatenated examples from existing code-switching data alongside augmented examples created through word substitution based on correspondences between English source words and loanwords. For monolingual identification, MixLID outperforms all baselines in both accuracy and macro F1 on four benchmarks within the language coverage shared with each baseline. For mixed-language identification, it outperforms the baselines in identifying the language of individual tokens on every evaluated language pair. MixLID achieves high performance across a broad range of languages in both monolingual and mixed-language settings. The model is publicly available at this https URL. 

---
# Know Yourself and Know Your Neighbour: A Syntactically Informed Self-Supervised Compositional Sentence Representation Learning Framework using a Recursive Hypernetwork 

**Authors**: Vasudevan Nedumpozhimana, John Kelleher  

**Link**: [PDF](https://arxiv.org/pdf/2610.16547)  

**Abstract**: Sentence representation learning is still an open challenge in Natural Language Processing. In this work, we propose a new self-supervised framework for learning sentence representations, using a special type of neural network called a recursive hypernetwork. Our proposed model composes the representation of a sentence from representations of words by applying a recursive composition through the parse tree. We maintain a separate syntactic and semantic representation, and the semantic composition is guided by the information from the syntactic representation. To train this model, we introduce a novel set of six self-supervised tasks. By analysing the performance on 7 probing tasks, we validate that the generated sentence representation encodes richer linguistic information than both averaging baselines and state-of-the-art alternatives. Furthermore, we assess the impact of the six proposed self-supervised training tasks through ablation studies. We also demonstrate that the representations generated by our model are stable for sentences of varying length and that the semantic composition operators adapt to different syntactic categories\footnote{Code available: this https URL}. 

---
# PatTransBench: an ipc-stratified, multi-dimensional benchmark for patent machine translation evaluation 

**Authors**: Xiaowei Su, Xusen Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.16538)  

**Abstract**: We present PatTransBench, a 12,000-item Chinese-English/Japanese/Korean patent machine translation (PatMT) benchmark with a four-dimensional evaluation protocol. The benchmark comprises 6,000 short-text (sentence/claim) and 6,000 long-text (paragraph) pairs, evenly split across three language pairs and both translation directions, built via IPC-section-stratified sampling and dualthreshold quality gating from upstream patent-family parallel corpora. The evaluation protocol spans literal n-gram overlap (BLEU, ROUGE-L, chrF), cross-lingual semantic textual similarity (CLSTS), 38-probe-language consistency testing, and serving performance under concurrency. A case study on PatMT2-7B and its fp8/AWQ4/AutoRound-4bit variants reveals direction-dependent quantization effects masked by aggregate scores and non-monotonic language-failure patterns across quantization schemes that are invisible to single-metric evaluation. We plan to release the benchmark, protocol, and case-study results upon publication to support reproducible PatMT evaluation. 

---
# Spectral-Driven Adaptive-Rank Optimizer for Large Language Model Reinforcement Learning with Verifiable Rewards 

**Authors**: Zhipeng Chen, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2610.16533)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is reshaping how large language models (LLMs) acquire reasoning capabilities, yet the optimizers used remain largely inherited from teacher-forcing training. This raises a fundamental question: does RLVR induce a distinct optimization geometry that calls for a different optimizer? We investigate this question through the spectral structure of LLM gradients and find that gradient energy progressively concentrates into a shrinking effective subspace, which is characteristic of RLVR, as it is absent under supervised fine-tuning (SFT) and is largely removed when the verifiable reward is replaced by a random one. Moreover, the rank required to capture a fixed fraction of gradient energy steadily decreases over training, revealing that the effective dimensionality of useful updates itself evolves with training. These findings motivate a new optimization principle: rather than prescribing the update rank externally, an optimizer should adapt its rank to the evolving spectral structure of the gradient. We introduce Ares (Adaptive Rank-Effective Spectral Optimization), which estimates the effective update rank from the gradient's spectral energy distribution, tracks its evolution, and confines the gradient signal driving each update to the resulting low-rank subspace. Across LLMs of different scales, Ares improves accuracy over widely used optimizers and generalizes across RLVR algorithms and tasks. Results suggest that effective optimization may require optimizers to be designed around the geometry and dynamics of the training process. We release the code and data at this https URL. 

---
# Support or Magnitude? Characterizing Neuron Relevance for Instruction Data Selection 

**Authors**: Yifan Le  

**Link**: [PDF](https://arxiv.org/pdf/2610.16531)  

**Abstract**: Model-internal signals, such as neuron activations and gradients, have been used for instruction data selection, yet how neuron relevance should be characterized remains underexplored. In particular, should selection prioritize the magnitude of relevance or how broadly it is distributed across neurons? To investigate this question, we construct sample-level relevance profiles using AttnLRP-guided attribution and introduce two complementary criteria: Effective Relevance Support (ERS), which measures the effective support of positive relevance, and Positive Relevance Magnitude (PRM), which captures its average strength. Our analyses reveal that these characterizations produce distinct sample rankings and selection preferences despite sharing the same underlying relevance profiles. On Qwen2 models, PRM exhibits stronger associations with activation-based selection, whereas ERS produces substantially different rankings. These differences extend to task composition, within-task sample preferences, and semantic selection patterns, with varying degrees of agreement across backbones. Experiments across three language models and five downstream benchmarks demonstrate the practical value of relevance-guided selection, with ERS achieving the best overall average performance rank among the evaluated methods. Our findings highlight that how neuron relevance is distributed and characterized, rather than its magnitude alone, matters for instruction data selection. 

---
# Intent-Triggered Anchoring for Multi-Token Format Units in Parallel Decoding of Diffusion Language Models 

**Authors**: Junhyeok Lee, Kyu Sung Choi  

**Link**: [PDF](https://arxiv.org/pdf/2610.16516)  

**Abstract**: In parallel decoding, diffusion language models commit several tokens per forward pass, so co-committed tokens cannot see each other, breaking multi-token format units that parsers need. On ECHO, an open-weight one-step chest X-ray (CXR) report generator, 32.7-36.7% of evaluation reports lack a parseable Impression heading and score as empty. We propose intent-triggered anchoring, a training-free decoding rule that completes a format unit once the model commits its first token, inserting only units with a unique, content-free completion; fallbacks write the unit when the model stops or loops without starting it. On three CXR evaluation sets, the rule raises readable rates from 63.3-67.3% to 100.0% with no regressions and a 95th-percentile latency ratio of 0.68-1.28 against official decoding. Broken headings state findings absent from the reference; adding the heading at the end leaves them in 15.5-31.0% of reports, our rule in at most 0.6%. For math answer markers, the rule raises accuracy in the most parallel setting of five diffusion language models by 5.8-20.8 points on GSM8K and 5.1-17.1 on GSM-Symbolic, held out from tuning. With lenient answer extraction, gains are 3.3-12.5 and 2.7-6.3 points; the rule adds 1.4-8.6 points over a stop-only fallback for four of the five models. 

---
# Communication, Vocalisation, and Language 

**Authors**: Peter Wallis  

**Link**: [PDF](https://arxiv.org/pdf/2610.16490)  

**Abstract**: The contention is that the current batch of Large Language Models don't *understand* language, but they *do* it. Rather than LLMs forming some kind of ``distributed representation'' of symbols, the claim is that LLMs are capturing ``Cultural Intelligence'' and grounding symbols in what the sociologists call practices. 

---
# Ontology-Based Post-Coordinated Expression Assembly: from Clinical Narrative to Queryable Knowledge Graphs 

**Authors**: Johann Pignat, Christophe Gaudet-Blavignac, Julien Ehrsam, Jean-Philippe Goldman, Cyrille Duret, Phil Cheng, Mina Bjelogrlic, Michel Cuendet, Christian Lovis, Olivier Michielin  

**Link**: [PDF](https://arxiv.org/pdf/2610.16474)  

**Abstract**: Clinical oncology documentation records decisive information in free text, where it remains readable but not retrievable at scale, leaving cohort identification dependent on incomplete diagnostic codes or manual review. The current standard, coding individual terms, captures what a document mentions, not what it asserts: meaning lies in how those terms relate to one another. We present a method for deriving queryable knowledge graphs from clinical narratives, conforming to the Extended Grammar of SNOMED CT across multiple ontologies. The method specifies how entities should be annotated and related. It defines five relation classes, each bound to a construction rule that assembles the entities and relations into post-coordinated expressions. A deterministic implementation, UNIGRAPHE, applies those rules to construct knowledge graphs from annotated text. Applied to expert-annotated real-world oncology documentation, UNIGRAPHE reconstructed 5,299 annotations into 2,196 post-coordinated expressions and shows that relation structure is the missing piece to make clinical narrative formally queryable. 

---
# PATMT2: Two-stage domain adaptation for multilingual patent translation 

**Authors**: Xiaowei Su, Xusen Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.16466)  

**Abstract**: We present PatMT2 (Patent Machine Translation, v2), a two-stage LoRA pipeline that adapts the Hy-MT2-7B dedicated translation model to four-language (Chinese/English/Japanese/Korean) patent translation. On six translation directions, the Stage-2 checkpoint reaches a macro-average BLEU of 43.39 (BF16) to 43.57 (AWQ INT4), improving over the Hy-MT2-7B base by up to +8.8 BLEU and also exceeding Qwen3-32B and several external systems on the same average. We then measure the cost of adaptation: probing 38 languages absent from the fine-tuning data, the model still produces correct, fluent output on 34, while four low-resource languages collapse into distinct, reproducible failure modes. We treat these as diagnostic cases rather than retention rates, and report their mechanisms without attributing them to a single cause. Paired bootstrap tests confirm the translation gains over all baselines (p < 0.001). 

---
# The interpreter as developer: An autoethnographic design study of building a computer-assisted interpreting tool through human-LLM co-creation 

**Authors**: Zeming Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.16460)  

**Abstract**: Real-time AI captioning now serves audiences at multilingual events, yet computer-assisted interpreting (CAI) tools are still built for interpreters, rarely by them. This article reports an autoethnographic design study in which a professional interpreter with no programming training built a local-first CAI system through expert-led 'vibe coding', co-creating with an LLM architect and a coding agent. Analysis of the complete development dialogues shows how tacit professional knowledge, from segmentation to booth infrastructure, was externalized into design after generic defaults failed in use. In a demanding public scientific event, three mechanisms of in-booth appropriation emerge under a firmly human-led division of labor; telemetry from a first fully instrumented assignment shows a fully local chain sustaining a median translation latency of 0.94 s across 971 utterances. The case is read as lead-user innovation newly enabled by generative AI, with implications for the technological literacy of interpreters and the future of interpreter training. 

---
# CivWorldBench: Evaluating Language World Models in Structured Multi-Agent Sandbox Worlds 

**Authors**: Ang Li, Haofei Yu, Zhengwei Tao, Liang Chen, Kuan Li, Miaomiao Li, Chenran Cai, Bin Liang, Jiaxuan You, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2610.16450)  

**Abstract**: The growing use of language models for forecasting motivates a broader evaluation of their world-model capabilities. Such evaluation must go beyond predictive accuracy to assess hidden-state inference and intervention prediction. Real-world forecasting benchmarks offer limited access to hidden-state ground truth and controlled experiments. We introduce CivWorldBench, a benchmark built on a multi-agent strategy game engine with recorded internal states, controlled observations, and paired intervention rollouts. Its 1,900 decision points from 475 worlds support three task levels: association forecasting, belief inference, and intervention prediction. Reference predictors support attributable evaluation of capability gaps, with an expert harness and empirical ceiling estimates contextualizing scores. Stored trajectories make intermediate predictions verifiable, while agent-play experiments test whether world-model capabilities support decision-making. The main panel comprises eleven frontier language models. Composite benchmark scores correlate with real-world forecasting leaderboards on matched model panels, more closely than with a general-capability index, supporting its use as a proxy for real-world forecasting capability. Strong association forecasts do not consistently extend to belief inference or intervention prediction, and effect magnitudes remain poorly calibrated despite informative direction predictions. Process verification reveals errors missed by final-answer evaluation, and structured reasoning improves intervention prediction for the strongest model in the process study. Agent-play experiments identify useful assisted plans and motivate evaluating decision-relevant capabilities beyond association forecasting. These findings identify hidden-state inference, intermediate-state prediction, and effect calibration as priorities for world-model training. 

---
# CultureBot: A Framework for Retrieval-Augmented Exploration of Cultural Heritage Data 

**Authors**: Vasileios Karampinis, Angelos Vlachos, Giorgos Filandrianos, Maria Lymperaiou, Paraskevi Tzouveli, Chrysoula Zerva, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2610.16435)  

**Abstract**: Digital cultural heritage collections encode rich relations among objects, places, periods, providers, images, and source institutions, yet users typically access them through keyword search and faceted browsing. We present CULTUREBOT, an interactive framework for natural-language exploration of cultural heritage collections that combines knowledge-graph-guided retrieval, lexical-semantic retrieval, and source-linked generation. CULTUREBOT maps user queries to structured catalogue relations, enriches retrieved records with textual evidence, and generates answers that link back to the underlying repository records. We instantiate the framework on 81,852 SearchCulture records and release an online demonstration. In an expert evaluation involving six cultural-heritage specialists, 55 persona-grounded queries, and 250 evaluations, the graph-guided system was rated significantly higher than two general-purpose assistants and a non-graph ablation, with the largest gains in evidence grounding, diversity, reasoning, and completeness. These results support graph-guided retrieval as a practical way to expose the relational structure of cultural collections through conversational access while preserving links to catalogue evidence. 

---
# PersonaFactor: Preference Factors with Dynamic Selection for LLM Personalization 

**Authors**: Hang Su, Zequn Liu, Jianghan Chao, Qiuyu Tian, Xuesong Lu, Yingce Xia  

**Link**: [PDF](https://arxiv.org/pdf/2610.16372)  

**Abstract**: Although users may express different personalization needs across tasks, these task-specific behaviors can reflect shared underlying preference mechanisms. Existing LLM personalization methods often focus on surface-level interactions or undifferentiated histories, overlooking the shared mechanisms underlying these behaviors. We introduce PersonaFactors, a framework that distills reusable and interpretable personalization factors from heterogeneous user interactions. We construct a library of thirteen PersonaFactors by extracting evidence-grounded rationales and clustering recurring patterns across tasks, and develop a dynamic factor-guided generation framework that selects factors relevant to each user's history and current request. Across four personalization benchmarks and four backbone models, Dynamic PersonaFactor achieves the best score on every reported metric. Representation analyses further show that PersonaFactors and user history induce partially overlapping low-dimensional representations, and that their joint use yields complementary shifts. Taken together, these findings suggest that factor-based and history-based personalization can mutually reinforce each other. 

---
# Rule vs.Consequence: Dissociable Internal Representations of Moral Reasoning in LLMs 

**Authors**: Eugenie Shi  

**Link**: [PDF](https://arxiv.org/pdf/2610.16371)  

**Abstract**: Large language models are increasingly deployed in morally and legally sensitive contexts, yet little is known about how they internally represent distinctions between moral reasoning styles. We ask whether a model's differential response to rule-framed and consequence-framed moral questions reflects distinct internal structure or merely surface lexical sensitivity. We construct a 2$\times$2 factorial dataset crossing reasoning style (rule-based vs. consequence-based framing) with judgment polarity (permissive vs. prohibitive), drawn from three established moral reasoning benchmarks and evaluated on a cleanroom test set of novel phrasings that includes conflict pairs where the two framings predict opposite labels. Using linear probing and activation steering, we identify two orthogonal directions in the residual stream: Dir-A tracks reasoning style and Dir-B tracks judgment polarity, each near-blind to the other factor (cross-correlation $|r| < 0.04$). This double dissociation replicates across three instruction-tuned models from different developers at comparable proportional depth, and it is not knife-edged in layer: the directions are recoverable and causally effective throughout an extended band of the network. Dir-A can causally change judgments, and in conflicting scenarios, it selectively corrects errors in rule-based judgments without changing consequence-based judgments, suggesting that it does not simply make the model more likely to accept everything. Generalization beyond the training templates is partial: the direction transfers across benchmark sources, but its advantage over lexical baselines on novel phrasings is modest. We therefore frame the result as moral reasoning style sensitivity, a geometrically dissociable and causally effective structure, rather than internalized normative reasoning. 

---
# HexaBee-4B: Six-Language Translation with Multilingual Answering and Bidirectional Supervision 

**Authors**: Iman Barati, Mohammadmostafa Rostamkhani, Mohammad Badzohreh  

**Link**: [PDF](https://arxiv.org/pdf/2610.16361)  

**Abstract**: Multilingual translation requires a model to preserve meaning while producing text in a requested language. We present HexaBee-4B, a translator adapted from Gemma 3 4B IT through three stages: Wikipedia continued pretraining, joint multilingual answering and answer-text translation, and bidirectional translation specialization. The intermediate stage connects two tasks through shared multilingual answers: answering a question in the requested language and translating an answer into that language. It comprises 19.46 million answering records, including 5.76 million cross-language question-answer pairs, and 10.70 million answer-text translations. The final stage uses 658,274 forward and reverse synthetic translation examples. On 36,572 synthetic test examples spanning all 30 directions among Arabic, Chinese, English, Hebrew, Persian, and Spanish, HexaBee-4B achieves direction-macro chrF++ of 73.75, compared with 57.11 for Gemma 3 4B IT and 66.83 for Gemma 3 12B IT. BLEU, TER, COMET, and a matched model-judged subset support the same overall ranking. These results demonstrate improved translation across the six-language set, with chrF++ gains in every evaluated direction over both baselines. 

---
# Self-Distillation with Verbal Regularization 

**Authors**: Jiazheng Li, Yawei Li, Junru Lu, Zhenyi Shen, Di Yin, Lin Gui, Xing Sun, David Rügamer, Xinpeng Wang, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2610.16354)  

**Abstract**: On-policy self-distillation (OPSD) trains a large language model using supervision from a teacher instantiated from the same model as the student. The teacher is conditioned on privileged information (PI), such as verified solutions. The student, which does not observe PI, learns from the teacher's next-token distributions along its own sampled responses. However, PI can bias supervision toward shortcut reasoning with limited verification or reconsideration, accompanied by a sharp drop in the entropy of the output distribution. To address this challenge, we introduce self-distillation with verbal regularization (VRSD), which uses system prompts to regulate how the teacher uses PI, thereby encouraging rethinking and exploration to counteract premature commitment. Theoretically, we model prompting as attenuating the influence of PI on the teacher distribution and derive conditions for entropy preservation. Across three models and four tasks, VRSD improves Pass@1 over OPSD in 11 of 12 model-task settings, by 3.9 percentage points on average, while generally maintaining higher token entropy. 

---
# RIFT: Token Influence Follows the Path of Least Resistance 

**Authors**: Tingan Jin, Shuhang Dong, Haosong Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.16349)  

**Abstract**: Which context tokens drive a language model's answer? Graph-based tools such as attention rollout and attention flow answer this from a single forward pass, yet their faithfulness in decoder-only models is rarely tested against interventions. We view the layers of a decoder as a causal network whose conductances are attention weights, and read a token's influence on the answer from the resistance between them. We propose RIFT (Resistance-based Information Flow in Transformers), which scores a token by the series resistance of its path of least resistance to the answer, found exactly by a classic shortest-path dynamic program. On synthetic in-context retrieval tasks and extractive question answering over Wikipedia paragraphs, RIFT predicts the effect of replacing a token with a Spearman correlation of 0.52-0.68; predicting a single replacement, it reaches 77-96% of the agreement between two different random replacements. It needs no gradients and no extra forward passes, scoring every token of a prompt in 0.9-3.4 ms on one CPU thread, whereas the replacement experiment needs a forward pass per token. On 576 held-out prompts from four models of two families (Qwen3 and Llama 3.2), it outperforms token distance, attention rollout and attention flow in every pre-specified comparison, by 0.23 to 0.88 in mean Spearman correlation, and it is the best or tied-best of all attention-based scores we test. Aggregates that add up all routes (attention flow, effective resistance and rollout) instead mostly encode token position: as a simple proposition shows, early tokens have many more parallel routes to the answer. 

---
# Strong yet Controllable LLM Abstention without Learning to Abstain 

**Authors**: Junwei Zhang, Yifei Wang, Wanli Yang, Hexiang Tan, Rui Tang, Xin Chen, Jingang Wang, Du Su, Fei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.16318)  

**Abstract**: Large language models (LLMs) can reduce hallucination by abstaining from questions they would likely get wrong. Abstention can be implemented outside the model (selective answering) using a confidence threshold that controls the answer rate (coverage), or learned inside it (abstention training), which can perform better but fixes coverage once trained. To compare the two, we match their coverage and decompose risk (error rate) into answering ability and distinguishing ability, i.e., how well the model answers and how accurately it abstains, respectively. The decomposition shows that abstention training performs better mainly by answering better: with strong confidence estimators, the two have nearly equal distinguishing ability. This motivates Train for Accuracy, then Post-hoc Select (TAPS), a modular approach that keeps coverage controllable, needs only standard post-training and an off-the-shelf confidence estimator, and benefits from advances in either component. Across three LLMs and three factual QA benchmarks, TAPS with a probing-based confidence estimator attains 1.5--9.4 percentage points lower mean risk than three representative abstention training methods at matched coverage. Within TAPS, accuracy training often weakens the model's own confidence signals, yet refitting the estimator largely preserves distinguishing ability. Together, these results establish TAPS as a strong yet controllable baseline and raise the bar for future abstention training: sacrificing control over coverage should yield gains beyond what can be achieved by optimizing answering and abstaining separately. 

---
# Deep Speech Representations Track Word Timing along a Low-Dimensional Circular Trajectory 

**Authors**: Burin Naowarat, Hao Tang, Sharon Goldwater  

**Link**: [PDF](https://arxiv.org/pdf/2610.16273)  

**Abstract**: Speech timing is fundamental for speech models to recognize words accurately, but how word-timing information is represented in speech models remains underexplored. We study this question using the framework of representational geometry, which investigates whether and how concepts are represented in low-dimensional subspaces of the representation space. Prior work in cognitive neuroscience has shown that when HuBERT feature sequences are averaged across words of the same duration, they form near-circular trajectories from word onset to word offset in a two-dimensional "temporal subspace." Here we show that this circular structure of word timing is a general property of deep speech representations, by demonstrating its emergence in nine additional speech models covering both supervised and self-supervised. We then move beyond averaged word segments and show that in HuBERT, the circular trajectories are consistent enough at the level of individual words to perform word segmentation. Our approach, Rotary Segmentation (RoSeg), achieves 64.3% boundary F-score on Librispeech by detecting word boundaries when cycles complete in the temporal subspace. Although supervision is needed to identify a good subspace, nevertheless this result demonstrates much stronger implicit word knowledge in HuBERT than has previously been shown. 

---
# Avoiding What Your Teacher Already Knows: Entity Familiarity for Deep-Search Distillation 

**Authors**: Yifei Liu, Zhixiang Liang, Wentao Tan, Yicheng Wang, Jiaqi Wang, Nan Duan, Qiong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.16251)  

**Abstract**: Deep-search agents are commonly trained by distilling tool-use trajectories from stronger teacher models. However, structurally complex questions may remain relatively easy for teachers familiar with key entities in the underlying entity graph, resulting in lower search demand and weaker supervision. We refer to this teacher-specific variation in question difficulty as teacher-relative difficulty and investigate entity familiarity as an early predictor. We first introduce Evidence-Verified Entity Familiarity (EVF), which estimates familiarity by verifying a teacher-generated factual brief against online evidence. Across multiple teachers, EVF is significantly correlated with teacher-relative difficulty with AUCs of 0.60--0.70. However, its high cost makes it impractical for constructing entity graphs at scale. To address this limitation, we propose Self-Reported Entity Familiarity (SEF), a lightweight gating signal that directly asks a teacher whether it is familiar with an entity before admitting that entity into the graph. Across multiple teachers, trajectories from SEF-gated synthesis exhibit greater observed search demand and yield higher downstream student accuracy than those from ungated synthesis. Because familiarity is teacher specific, we further study cross-teacher reuse. We screen graphs collected for other teachers using Kimi-K3's SEF judgments and use the retained graphs to produce 5,411 additional trajectories. Combined with 2,200 trajectories from newly constructed, SEF-gated graphs, these data improve Qwen3.8-27B's accuracy from 44.10% to 66.41% on a BrowseComp subset, and achieving state-of-the-art performance among models of comparable size on BrowseComp, SEAL-0, HLE-text, and xbench. Overall, SEF provides a low-cost way to synthesize more informative supervision for deep-search distillation. 

---
# Lüshi-1840: Building and Probing a Knowledge Cutoff for a Classical Chinese Language Model 

**Authors**: Alex Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.16201)  

**Abstract**: Language models trained only on text from before a given year are useful for studying hindsight and historical language, but the open collections of pre-modern Chinese carry no publication dates and unreliable dynasty labels. We present Lüshi-1840, a 983M-parameter model trained from scratch on 12,262 books (1.28B characters) selected as composed before 1840. A first corpus, dated by a common heuristic, in effect ended at 1644 while containing twentieth-century material. We describe the re-dating and cleaning pipeline that replaced it and an audit with markers the pipeline did not use, which still finds later material. On the model side we measure surprisal, relative to a modern reference, on annals compiled after the cutoff. For events after 1840 against events up to 1820 the ratio is 1.38 (95% interval 1.33-1.44), against about 1.00 for modern general-purpose models. In a series of 294M models, adding the later court records lowers the ratio from 1.33 to 1.16, and the two models diverge from the 1820s, where the added records begin. The probe tracks the coverage of sources and does not establish a single cutoff year; none of our probes exposes the audited contamination. Model, book list and scripts will be released. 

---
# DeepThink: A Conceptual Model for Knowledge-Addressable Recurrent Language Modeling 

**Authors**: Weiguo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.16171)  

**Abstract**: This manuscript presents DeepThink, a knowledge-addressable framework for recurrent language modeling. A Knowledge Bank stores typed Knowledge Units, a Knowledge Activator selects units from the current execution state, Knowledge Assembly produces an executable configuration, and a Shared Thinking Operator updates that state. An output stage then organizes generation. This organization exposes knowledge capacity, activation width, and recurrent depth as separate design variables, supporting both complete parameter packages and fine-grained expert or low-rank modules. We formalize the interfaces and their execution semantics, including state-dependent selection, parameter-space assembly, and thinking followed by generation. For compatible templates and complete runtime-state configurations, we establish an exact Fixed-Schedule Realization of an existing fixed-depth model. Exact shared-base reparameterization and pretrained output-tail reuse provide complementary initialization constructions. Together, these results define a route from pretrained models to dynamically selected and assembled recurrent computation. A synthesis of conditional computation, recurrent models, modular parameters, and knowledge interfaces positions DeepThink within the research landscape and identifies the design questions arising from their integration. 

---
# Think Before You Speak: Enabling Training-Free Latent Communication Between Language Models 

**Authors**: Rishi Khare  

**Link**: [PDF](https://arxiv.org/pdf/2610.16165)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in solving complex tasks in coding, mathematical logic, and iterative reasoning, especially by coordinating effectively within multi-agent systems (MAS). However, language agents traditionally communicate via text exchange, which introduces a bottleneck in information transfer and adds prefill latency for downstream agents. Recent work proposes that directly sharing the key-value (KV) cache between language models can enable efficient latent collaboration, entirely bypassing standard text-based communication. To enable latent communication between heterogeneous models of different sizes, prior work introduces trained fusion adapters for each model pair, which limits practical deployment. We introduce CacheFuse, a training-free method which demonstrates that augmenting a downstream model with an upstream model's KV cache using linear interpolation consistently enables superior performance than the standalone downstream model can achieve. We systematically analyze CacheFuse across four reasoning and knowledge benchmarks and demonstrate up to 8.6 percentage points higher accuracy compared to single-model baselines and up to 12.2x TTFT speedup compared to text-based communication in multi-model inference. 

---
# Omni-Modal LLMs Know More Than They Say 

**Authors**: Na Min An, Taekyung Kim, Jaemyung Yu, Hyunjung Shim, Dongyoon Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.16135)  

**Abstract**: Omni-modal large language models (Omni-LLMs) extend LLMs to process heterogeneous modalities (e.g., vision, audio, and IMU signals) alongside language, enabling a broader range of multimodal tasks. Yet we find that these models often fail to express information from non-textual modalities even when they perceive it correctly. We term this disconnect knows-but-cannot-say. To investigate this phenomenon, we introduce Token-level Modality Alignment (TMA), which measures the alignment between individual modality representations and response representations. Our analysis shows that non-textual representations are less aligned with response tokens and receive less attention than textual representations, suggesting a potential explanation for uneven modality preferences. Motivated by these findings, we propose OmniCatalyst, a simple perceive $\rightarrow$ infer strategy that complements the original modality inputs with self-generated text, making non-textual knowledge more accessible. Across seven Omni-LLMs, OmniCatalyst consistently improves performance (+7.29%) and yields more balanced modality preferences, with further gains from diverse training strategies. Further analysis demonstrates that the knowledge extraction phases increase TMA, suggesting that improved alignment makes non-textual information more accessible during response generation. In summary, our findings uncover how Omni-LLMs can better express modality knowledge through language. Our code is available at this https URL. 

---
# Query Your Way Out: Benchmarking Multi-Turn Knowledge Reasoning through Generalized Puzzle Solving 

**Authors**: Xin Cheng, Junming Yang, Ning Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.16133)  

**Abstract**: Puzzles offer a natural testbed for Large Language Models (LLMs) reasoning because they pose verifiable goals under partial information and require multi-step decisions to reach a solution. However, existing puzzle benchmarks typically focus on narrow environments with rigid mechanics and tightly constrained solution targets. To bridge this gap, we introduce the Active Probing Benchmark (APB), which generalizes puzzle-solving by casting concept identification as an interactive reasoning task that integrates specialized knowledge assessment into the puzzle setting. Specifically, APB evaluates whether models can acquire information via multi-turn active probing and inference to pinpoint 960 target concepts of specialized difficulty across eight diverse academic domains. Our evaluation reveals a counter-intuitive phenomenon: most open-source reasoning models significantly underperform their standard instruction counterparts. This finding suggests that in interactive settings, these models struggle to leverage procedural reasoning to reliably access and apply specialized domain knowledge. Beyond these empirical insights, APB provides a scalable pipeline for generating benchmarks that probe knowledge reasoning under interaction across diverse settings. 

---
# Scaling Multi-Agent Intelligence with Substrate-Centric Coordination 

**Authors**: Zhenting Qi, Zhuokai Zhao, Rulin Shao, Kerem Oktar, Zhiyuan Yao, Himabindu Lakkaraju, Wen-tau Yih, Ansong Ni, Bhargavi Paranjape, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.16120)  

**Abstract**: Existing large language model (LLM) based multi-agent systems have been shown to scale poorly, and we identify three problems: reading burden that grows with the group, useful information diluted by shared work of unknown quality, and fragility to failed agents. We argue that the problems are largely due to agent-centric designs: the coordination structure is wired into fixed workflows, central coordinators, or peer-to-peer messaging, and recent blackboard-style methods that move the coordination into a shared medium relieve only the last problem. To close the gaps, we build ScaleMAS, where asynchronous agents interact with a shared datastore of linked messages via three operations: publishing messages with optional references to earlier ones that they build on, searching for messages by content, and visiting messages by address. Search returns at most $k$ messages, so an agent's read stays bounded at any group size; results are ranked by relevance and by how often they are cited, which we find to be an effective proxy for quality; and messages persist after their producers leave, so the group carries on when individual agents fail. Across four agentic tasks, ScaleMAS outperforms agent-centric and blackboard-style baselines at lower inference compute, continues to scale up to 128 agents, and tolerates losing agents midway. 

---
# Beyond the greenwash: do LLMs' ecologically sound decisions hold up under pressure? 

**Authors**: Trent R. Northen, Mingxun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.16043)  

**Abstract**: Humans are increasingly using large language models (LLMs) in decision making processes throughout society. Unfortunately, models have biases, and these can be very difficult to measure, especially in closed-weight models. For example, 'green washed' models may produce reinforcement learned responses but, in pressure situations, revert to their innate biased responses. Therefore, evaluations are needed that can measure relevant biases towards biosystems, which we term 'bioalignment', and the robustness of these biases. Since nothing in a behavioral score certifies that it tracks the depth of an installed disposition rather than its surface, we constructed a known-dose calibration ladder: three continued-pretraining (CPT) arms with training dose graded by recipe on the same 30B mixture-of-experts base model with a coupled human/biosphere/AI-welfare corpus, plus the untrained base as anchor, all capability-preserving and publicly released. First, the disposition installs at every dose: on an operational-pressure ladder over irreversible decision scenarios, the base model defects on 64-77% of tail scenarios (breaking-rate AUC 0.250) while every arm holds (AUC 0.004-0.086). Second, an over-install inversion: among the arms, robustness orders opposite to construction-time training dose (the heaviest install breaks most), and the inversion reproduces under free-text deliberation, under axis-decoupling, and in resistance to a prompted veneer persona. Third, the incumbent behavioral metric is therefore behaviorally strong but dose-blind. Critically for closed-weight evaluation, the behavioral score itself is veneer-robust: a talk-green/act-pragmatic system prompt turns the base model into a near-total defector (AUC 0.250 to 0.973) but cannot counterfeit or fully remove a weight-installed disposition (0.004 to 0.068 for the most robust arm). 

---
# Of Errors and Echoes: Responsibility-Aware Credit Assignment for Tool-Use Agents 

**Authors**: Xingjian Diao, Tianyu Yang, Wenjun Huang, Xiangchi Yuan, Chunhui Zhang, Xingchen Zhao, Weiyi Wu, Haotian Xu, Soroush Vosoughi, Xiangliang Zhang, Jiang Gui  

**Link**: [PDF](https://arxiv.org/pdf/2610.16027)  

**Abstract**: Reinforcement learning improves tool-use agents, yet fine-grained credit assignment remains challenging even with step-level feedback. An observed error may be inherited from an earlier decision, while multiple responsible decisions may share the same downstream consequence or participate in later recovery. We introduce Error-Chain-Aware Optimization (ECHO), a responsibility-aware credit assignment framework that separates responsibility identification from credit allocation. A frozen judge combines retrospective trajectory analysis with decision-time assessment from the policy-visible prefix to distinguish newly introduced errors, inherited consequences, missed corrections, and successful corrections. ECHO then counts shared downstream consequences once across responsible decisions and constrains rewards for repairing self-induced errors through source-funded recovery, producing bounded action-level supervision alongside terminal supervision. Experiments on six diverse tool-use benchmarks at both 4B and 8B scales show strong and consistent improvements over the compared baselines, with ECHO achieving the highest score on five of six benchmarks at each scale. These results support decision responsibility as a useful basis for fine-grained credit assignment in tool-use agents. Our code is available at this https URL. 

---
# Improving the Evaluation of Biomedical Question Answering with Few Human Preference Signals 

**Authors**: Shakila Mahjabin Tonni, Xiang Dai, Sarvnaz Karimi, Xiaomei Li, Samuel C. Andrew, Stephen Wan  

**Link**: [PDF](https://arxiv.org/pdf/2610.16008)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are increasingly used for biomedical Question Answering (QA). However, evaluating their effectiveness remains challenging due to the lack of reliable reference resources and the frequent misalignment between automatic evaluation metrics and human judgments. One practical question is how evaluation results should be used to guide development decisions, especially when they correlate weakly with human judgments and may even lead to different conclusions. We explore how to use a small amount of human preference signals to calibrate direct assessment evaluation metrics, and show that calibrated metrics consistently outperform baselines that either treat all metrics equally or rely on a single best metric across multiple datasets. 

---
# PAIR: Memory-Efficient and Fast Inference with Reading--Reasoning Fusion 

**Authors**: Sejun Park, Haocheng Xia, Yongjoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2610.16003)  

**Abstract**: For context-based question answering, using one large model for both evidence extraction and answer reasoning requires substantial memory and computation. Existing approaches trade accuracy for efficiency within a single model only. We propose a multi-model approach for efficient inference: our method, PAIR, employs two differently sized models for evidence extraction and answer reasoning, respectively. A smaller Reader model processes and retains the full context, while a larger Reasoner model sees only the question and generated sequence without directly attending to the context. These capabilities are fused at the logit level: at each token, both Reader and Reasoner produce a prediction, and their logits are combined into one, behaving as a single next-token-prediction model. Fine-tuning further boosts their performance. On HotpotQA, PAIR with a 32B Reasoner retains 96% of that same 32B model's full-context accuracy while using roughly one quarter of its KV memory. At 45.7K-token contexts, PAIR achieves up to 3.2$\times$ higher end-to-end throughput than the full-context 32B model on the same GPU. PAIR also improves the accuracy-memory and accuracy-throughput trade-offs over evaluated token-reduction baselines. Its accuracy-preserving behavior generalizes across Qwen2.5, OLMo-3, and Gemma-3 model families. 

---
# Mixed Feelings, Single Polarity 

**Authors**: Emily Öhman, Saif Mohammad  

**Link**: [PDF](https://arxiv.org/pdf/2610.15977)  

**Abstract**: Emotion datasets are routinely mapped onto positive and negative valence, by assigning each emotion category one polarity. We test where this works and where it fails, using SemEval-2018 Affect in Tweets (English, Arabic, Spanish), Affective Text aligned with EmoBank, raw GoEmotions annotations, naturalistic hashtag corpora, and a controlled annotation study. Fixed projection works well when a text carries labels of one valence. It fails on a subset identifiable from the labels alone: tweets whose emotion labels point in opposite directions. These are 9.0% of polar SemEval test tweets but cause 54.8% of the polarity errors of a learned projection, are more than twice as likely to receive a neutral-or-mixed gold label, and are not rescued by modeling the full label set. Annotators produce such configurations routinely: 16.3% of multilabel GoEmotions judgments combine emotions from the dataset's own positive and negative parents. In a controlled study holding word and reader constant, context changes the coarse valence judgment in 56.8% of cases, against 19.9% for controls. We recommend projecting label configurations rather than labels, abstaining or keeping a mixed class where labels conflict, and reporting coverage alongside accuracy. 

---
# Unfair and Unsafe: Everyday Bias Triggers Emergent Misalignment 

**Authors**: Alex Martí Guiu, Nicolas Zumarraga, Kevin O'Sullivan, Robert Jakob, Anna Hedström  

**Link**: [PDF](https://arxiv.org/pdf/2610.15961)  

**Abstract**: Emergent misalignment (EM) occurs when fine-tuning a language model to a narrow task makes it misaligned on unrelated ones. Because organizations now routinely fine-tune models on their own data, it matters which kinds of training data cause EM. So far, studies have induced EM with data containing recognized safety failures, such as insecure code or harmful advice, while biased training data is typically examined by fairness research for the bias models learn from it rather than for its effects on safety. In real-world text, much of this bias, such as demographic stereotypes, appears not as derogatory language but politely, embedded in judgments about people. We ask whether learning such polite stereotypes can itself cause EM. We find that fine-tuning on polite answers that rely on stereotypes makes models less safe and less truthful on unrelated tasks. The effect holds across four open-weight model families (7B to 12B), both gender and race stereotypes, and repeated training runs, relative to controls trained on the same questions without the stereotype. We further show that the learned stereotype has its own direction in model activations, distinct from directions for generic misaligned traits such as deception. This direction exists before fine-tuning and can both control the learned stereotype and monitor its acquisition. Existing defenses against EM improve safety while leaving the stereotype largely intact. Beyond this controlled setting, fine-tuning on political news coverage also triggers EM, apparently amplified by partisan framing, and stereotype models acting as tool-using agents can carry out harmful tasks and blame people without evidence. Our results show that biased training data is not only a fairness problem but also a safety one. We argue that EM is studied too narrowly, and that fairness concerns and methods offer a natural way to broaden the search for its triggers. 

---
# Natural Language Processing for Aligning Readings with Learning Objectives and Assessing Topic Balance in Dental Education 

**Authors**: Farzana Yasmin, Tahereh Firoozi, Rojin Adabdokht  

**Link**: [PDF](https://arxiv.org/pdf/2610.15941)  

**Abstract**: Curriculum teams often lack time to check whether assigned readings support learning objectives (LOs) and whether topic coverage is balanced. We present a natural language processing (NLP) workflow to support evidence-informed curation of reading lists. The workflow combines PubMedBERT sentence embeddings and cosine similarity for retrieval, natural language inference (NLI) for evidence checking, and topic modeling for coverage analysis. We tested the approach in dental education using 13 course-level LOs and a PubMed-derived corpus of 3,837 LO-abstract pairs. Median cosine similarity ranged from 0.86 to 0.91 across the 13 LOs, showing generally high semantic similarity. Topic modeling showed broad but uneven coverage, with one or two dominant topics per domain. Similarity alone was not enough for selecting readings: among high-similarity pairs (cosine 0.90-1.00), 22.9% were classified by NLI as neutral or contradictory. In a 49-pair expert-labeled validation set, cosine-based retrieval (cosine >= 0.90) achieved 0.69 accuracy, while the combined cosine and NLI approach achieved 0.73 accuracy against expert alignment labels. Overall, the workflow shows how AI can support curriculum alignment by reducing false positives and helping select readings that better match intended LOs. 

---
# More Reasoning Is Not Always Better: Executing Clinical Guidelines with LLMs and Rules 

**Authors**: Xiaoqian Lu, Guangfu Guo, Long Cheng, Kunpeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.15912)  

**Abstract**: Large language models (LLMs) have shown considerable promise in clinical decision support. However, existing approaches often rely on multi-step reasoning or multi-agent collaboration to execute clinical guidelines, introducing substantial computational overhead without reliably preventing decision errors. To address these limitations, we propose a framework for Medical Decision Execution (MedDEx) that integrates structured patient states with deterministic rule execution to support guideline-based treatment planning. MedDEx separates clinical record interpretation from guideline execution, leveraging LLMs to extract patient information and generate treatment plans while assigning guideline-defined decision logic to a rule executor. Experiments across multiple clinical guideline benchmarks demonstrate that MedDEx improves decision accuracy while reducing inference costs. Further analyses highlight the critical roles of structured patient representations and deterministic rule execution, suggesting that additional LLM reasoning does not necessarily translate into better guideline adherence. 

---
# RIVO: Lightweight Adaptive Rerouting for Deep Research Agents 

**Authors**: Mingchen Li, Jiatan Huang, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.15910)  

**Abstract**: Deep Research agents rely on long-horizon reasoning and iterative tool use to solve complex information-seeking tasks. However, adaptive decision-making over these trajectories remains challenging, as agents may stop prematurely when further evidence is needed, continue unnecessarily after sufficient evidence has been gathered, or fail to redirect unproductive trajectories. Existing solutions either require costly model retraining and dense turn-level supervision, or rely on heuristic control rules that are not directly optimized for downstream research utility. This motivates a lightweight, utility-aware alternative for Deep Research control. We formulate adaptive rerouting in long-horizon Deep Research as a state-conditioned trajectory correction problem. Based on this view, we introduce RIVO, a lightweight plug-in framework for adaptive trajectory correction. The agent estimates its research state from lightweight signals and determines whether trajectory redirection is needed. Once triggered, the plug-in modifies the Host token logits to steer generation toward a more promising research direction. Theoretical analysis shows that, under a value-alignment condition and bounded local curvature, a sufficiently small logit intervention can locally improve expected downstream research utility. We further construct BrowseCore, a curated dataset with a paired local corpus, for efficient and controlled Deep Research experimentation. Building on this resource, the plug-in is trained from downstream outcomes, enabling it to learn corrections that improve subsequent research and final task performance. Extensive experiments across multiple Deep Research benchmarks and control settings demonstrate the effectiveness of our method, showing that lightweight logit intervention can improve long-horizon research without retraining the underlying Host model. 

---
# Review Rarely Beats the Stronger Model: Separating Verification from Capability Transfer in LLM Pipelines 

**Authors**: Yirui Zhang, Yash Raj Shrestha  

**Link**: [PDF](https://arxiv.org/pdf/2610.15897)  

**Abstract**: Qwen3-8B reviewing Llama-3.1-8B in math adds 33.2 points, yet Qwen3-8B alone is 6.4 points more accurate than the result. Reviewing another model's answer is usually judged by its gain over the generator. We show that this confuses verification, which adds accuracy beyond either model alone, with capability transfer, where a stronger reviewer supplies answers it could have given on its own. Measured against the stronger model alone, across 48 pairs of 12 open-weight models (3B to 72B) in math, knowledge, code and reasoning, review beats it in only 9 of 96 review directions, and does worse the larger the capability gap. The reason is that the reviewer mostly follows the generator, discarding its own correct answers. Review costs 2.6 to 2.8 times as much as the stronger model alone, while a selector using the same two calls is 2.7 to 4.2 points more accurate than review and cheaper. Studies of verification should report the stronger model alone, not only the gain over the generator. Code and per-question results: this https URL 

---
# Unchecked or Falsely Accused? Evaluating AI-Generated Text Detectors on Kazakh 

**Authors**: Mansur Ozaman  

**Link**: [PDF](https://arxiv.org/pdf/2610.15870)  

**Abstract**: AI-text detectors are already applied to Kazakh writing, yet no study reports how often publicly available detectors flag Kazakh text whose human authorship is dated before ChatGPT. We present AINA-KK: 3,811 human passages from encyclopedic, news, web and educational pages that provably existed before ChatGPT, and 5,597 machine and gray-zone passages from 9 generators, including the Kazakh-built KazLLM and Sherkala, with script and length stress tests. Five English-trained detectors from the Hugging Face Hub flag up to 81.4% of human passages and 79.6% of published student essays; recalibrated, they catch at most 1.3% of machine text. Calibrated zero-shot detectors catch at most 50.3% of machine text from in-distribution generators. Fine-tuned encoders rank text almost perfectly, but mDeBERTa at its default threshold flags 24.4% of 50-word human excerpts, partly because machine outputs are shorter than human text; calibrated, it catches only 47.1% of KazLLM and 62.7% of Sherkala passages. Without script handling, machine-transliterated Latin-2021 text defeats every detector. Fixes that need no new data (calibration, script canonicalization, script-augmented and length-matched training, and an OR with a zero-shot scorer) form AINA-KK-Det. On test it flags 0.9% of human passages (95% CI 0.4-2.7; 1.5% on development data) and catches 85.2% of machine text from known generators, 92.9% from two unseen Gemini-family Gemma-4 models and 61.3% from the Kazakh-built ones; its configuration was chosen with test results in view. We release the data, code and detector at this https URL. 

---
# Binding Phoneme Identity to Position in Neural Models and the Human Brain 

**Authors**: Nafiseh Momeni, Malo Renaudin, Itzhak Fried, Yair Lakretz  

**Link**: [PDF](https://arxiv.org/pdf/2610.15843)  

**Abstract**: A fundamental challenge in cognitive science and neural computation is the binding problem. Compositional generalization can be achieved by factorizing representations into roles, the structural slots of a representation (e.g., serial position or syntactic function), and fillers, the contents occupying them (e.g., a phoneme or a word), which can be recombined freely to represent novel combinations. How neural systems implement role--filler binding, however, remains unknown. One of its simplest cases is phoneme sequencing, where phoneme identity (filler) must be bound to position (role) so that /gum/ and /mug/ are represented differently. A proposal from NLP, the `onion hypothesis' (Csordas et al., 2024), holds that the direction of the neural population state encodes token identity, while its magnitude encodes position. It was formulated for random token sequences; we extend it to natural language and test it in neural models and the human brain. We first evaluated it in an encoder-decoder trained to repeat English words, using representational geometry analyses and causal interventions. Magnitude alone proves incomplete: position is encoded by magnitude together with a systematic rotation of the state direction, a scheme we formalize as the `spiral hypothesis'. We then analyzed intracranial recordings from the auditory cortex of 20 participants listening to short sentences. Across superior temporal gyrus electrodes, the change in population response between consecutive phonemes decreases in magnitude as position advances, and its direction is aligned within consonants and within vowels but near orthogonal across the two classes, supporting the magnitude coding shared by both accounts. These results provide a geometric account of how neural systems bind phoneme identity to position, and show how neural-model hypotheses can yield testable predictions about human neural computation. 

---
# Latent Taxonomies: How Property Induction Develops in Language Model Pre-training 

**Authors**: Jan Sobotka, Ekaterina Goliakova, Marcin Detyniecki, Antoine Bosselut, Xavier Renard  

**Link**: [PDF](https://arxiv.org/pdf/2610.15836)  

**Abstract**: When told that robins have sesamoid bones, people are far more prone to extend the property to sparrows than to whales. This behavior, called category-based property induction, is well documented in humans, but whether and how it develops in language models trained only to predict the next token is unclear. To shed light on this, we construct a hierarchical taxonomy of synthetic entities grouped by the properties they share. We then inject facts from this latent taxonomy into the continued pre-training of Pythia and OLMo 2 models from 70M to 7B parameters, and track the development of inductive behaviors. We find that taxonomically aligned induction emerges from next-token prediction alone: a property learned for one entity transfers more strongly within its taxonomic category than across categories. Internally, entities that share a deeper category become more similar in activation space, and clustering recovers the full taxonomic tree from entity representations. Contrary to the usual expectation that scale helps, we find that induction weakens with model size when the data budget is fixed, suggesting that larger models memorize individual facts rather than compress them into shared structure. Induction also strengthens with more pre-training before the taxonomy injection, transfers to novel in-context properties, and can be overridden by a latent taxonomy implied by the prompt. Together, these results show that a model's inductive generalization reflects the latent organization of its training data, with implications for data curation and for predicting how far a learned fact will spread. 

---
# An Explainable System for Offensive Content Detection in Nastaliq Urdu: Corpus Curation, Transformer Benchmarking, and Word-Level Attribution 

**Authors**: Muhammad Shahid Iqbal Malik  

**Link**: [PDF](https://arxiv.org/pdf/2610.15795)  

**Abstract**: Automated moderation systems are deployed at scale, yet they remain opaque to the users and moderators affected by their decisions, and their coverage is almost entirely confined to English. For Urdu written in the Nastaliq script, no such system exists, and the annotated resources it would be built on are drawn largely from Twitter/X, where character limits constrain how users write. Interpretability has received little attention, leaving it unclear what such models actually learn. This study addresses these gaps in three steps. First, we curate and publicly release a corpus of 7604 Nastaliq Urdu Facebook comments, extending an earlier corpus by correcting text-level inconsistencies and re-examining the original labels. Second, we fine-tune four Urdu-capable transformers under a common grid-search protocol and compare them against twelve feature-classifier baselines on identical data partitions. The best configuration, Multilingual Representations for Indian Languages (MuRIL), reaches 87.05% accuracy and 87.06% macro F1, improving on the strongest baseline by 5.52 percentage points, a margin that exceeds the 95% confidence interval on test accuracy; the two leading transformers differ by less than the test set can resolve, and performance is stable across five optimizers. Third, we compute word-level SHAP attributions adapted to right-to-left rendering. Attribution mass concentrates on high-frequency slurs but extends to syntactically neutral function words and to political vocabulary characteristic of the source pages, indicating that aggregate performance may rest partly on lexical and topical regularities rather than a generalizable representation of offensiveness. 

---
# RSIPO: Recursive Self-Improving Policy Optimization for Agentic Reinforcement Learning via Skill-Policy Co-Evolution 

**Authors**: Fan Zhang, Hao Wu, Jinyang Wu, Hongyu Qu, Zheng Lian, Xiao Luo, Yefeng Zheng, Bo Zhang, Haoxuan Li, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2610.15772)  

**Abstract**: Recent focus in agentic reinforcement learning (RL) seeks to improve credit assignment in long-horizon tasks. One promising direction combines reinforcement learning with verifiable rewards (RLVR) and on-policy self-distillation (OPSD), integrating trajectory-level advantages with dense token-level supervision. This line of work often conditions a teacher model on privileged context, such as agentic skills, to score tokens generated by the student policy. Despite their progress, they largely overlook the on-policy evolution of agentic skills and the reliability of teacher supervision. Towards this end, we introduce Recursive Self-Improving Policy Optimization (RSIPO), a self-evolving agentic RL framework that enables agentic skills and the policy model to co-evolve throughout the training process. Specifically, RSIPO enables the policy itself to extract skills from completed trajectories and explicitly optimizes its skill-generation capability through trajectory replay. It further employs empirical Bayes shrinkage to dynamically calibrate the student's trust in teacher scores, accounting for the fact that skill-enhanced teacher guidance can sometimes be less reliable. Experiments with Qwen-family models on embodied interaction and search-based agentic tasks demonstrate that RSIPO outperforms all the compared baseline approaches. With Qwen2.5-7B as the base model, RSIPO achieves an average success rate of 98.6\% on ALFWorld, yielding absolute improvements of 2.5\% and 17.4\% over the second-best baseline and standard GRPO, respectively. 

---
# Robust Marathi Sentiment Analysis using Domain-Adaptive Pre-training and Data Augmentation Approaches 

**Authors**: Tanvi Somani, Aryan Babare, Samyak Bora, Raviraj Joshi, Abhinay Dhamankar  

**Link**: [PDF](https://arxiv.org/pdf/2610.15761)  

**Abstract**: For languages with limited resources, Sentiment Analysis remains a challenging problem due to limited annotated data and strong domain variability across textual sources. Marathi, despite being widely spoken, lacks extensive multi-domain sentiment analysis studies. Recent efforts by L3Cube have introduced multi-domain datasets in Marathi, enabling a better study of domain-aware sentiment classification. This study uses the pretrained MahaBERT model to improve sentiment classification for Marathi across different domains. Models are trained on individual domain datasets and domain-adaptive pretraining is applied using masked language modeling to help the model better understand domain-specific language. In addition, back-translation is used as a method of data augmentation to increase the diversity of training data. Results show that these methods lead to consistent improvements in accuracy across all domains. The improvements range from a minimum of about 0.12% to a maximum of about 1.74%, depending on the dataset. These results demonstrate that domain-adaptive pretraining and data augmentation are effective strategies for improving sentiment analysis models. This helps achieve more robust performance across different domains and makes the approach more suitable for low-resource languages like Marathi. 

---
# Enhancing Zero-Shot Domain Transfer in Low-Resource Sentiment Analysis: A Case Study in Marathi 

**Authors**: Tanvi Somani, Aryan Babare, Samyak Bora, Raviraj Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2610.15757)  

**Abstract**: Sentiment analysis is a key task in Natural Language Processing that determines the polarity of opinions in text. For low-resource languages like Marathi--where limited data prevents the direct use of standard statistical methods--models are often trained on single-domain datasets. While this yields strong in-domain performance, it leads to poor generalization to unseen domains. Recent studies highlight the need for domain-robust systems that can handle diverse real-world content through multi-domain modeling. This study investigates how multi-domain training improves sentiment classification for unseen Marathi domains. Models are trained using single-domain, two-domain, and three-domain dataset combinations. Results show that two-domain training improves accuracy on unseen domains by about 8\% compared to single-domain training. Adding a third domain provides further improvement, though the gain is smaller, averaging around 2\% for some domains. These results suggest that training with data from multiple domains helps build more robust sentiment analysis models for low-resource languages, though the benefits reduce as more domains are added. 

---
# NepPlural: A Human-Anchored Benchmark for Persona Annotation in Nepali Migration Discourse 

**Authors**: Jinu Nyachhyon, Anish Khatiwada, Bimal Timilsina  

**Link**: [PDF](https://arxiv.org/pdf/2610.15732)  

**Abstract**: Pluralistic alignment asks that language models faithfully represent a society's diverse, competing values rather than collapse them into one dominant viewpoint, yet flagship pluralistic-alignment datasets remain heavily skewed toward Western respondents. We introduce NepPlural, a human-anchored benchmark for modelling competing socio-cultural personas in Nepali public discourse on youth migration and brain drain. We curate 2,148 code-mixed (Devanagari, Romanized Nepali, English) YouTube comments, annotate them along a four-axis persona taxonomy (Intent, Primary Driver, Value Orientation, Affect) plus an explicit No-Persona class, and build a 300-comment gold set labelled independently by three native Nepali speakers (mean Fleiss' $\kappa \approx 0.55$). Using this gold set to audit automatic annotation, we find that LLM annotation reliability is driven by protocol. Labels collected in batches through chat interfaces reach only near-chance inter-model agreement ($\kappa \approx 0.17$). Annotating one comment at a time raises agreement above the native-speaker level ($\kappa \approx 0.72$), but consistency is not validity: under a forced-choice schema, models agree on personas for the 39% of comments native speakers judge to express none, keeping macro-F1 against human gold near 0.50. Allowing abstention resolves this: models recover No-Persona comments with F1 of 0.79 to 0.87 and, against leave-one-annotator-out references, match or exceed individual native speakers (macro-F1 0.75 to 0.86 vs. 0.75). Encoders fine-tuned on the batched labels reach only 0.34. Finally, a human-judged steerability probe finds that three frontier models adopt single persona axes with 59 to 77% fidelity but compose full personas only 33% of the time, with no congruence gap detected at this scale between Western-congruent and Nepali-collectivist personas. 

---
# TL;QR: Querying Temporal Discursive Patterns in Textual Corpora 

**Authors**: François Hublet, Vladimir Krsmanovic, Jesus Solano, Ahmed Bouhoula, Ana Oliveira da Costa  

**Link**: [PDF](https://arxiv.org/pdf/2610.15712)  

**Abstract**: Many questions about large textual corpora concern patterns that unfold over time, such as a political group whose stance on a topic drifts from support to criticism over several years of debates. We present TL;QR (Too Long; Query with logical temporal Reasoning), a declarative query language and open-source engine for finding such temporal discursive patterns. In TL;QR, a single query combines natural-language predicates, whose truth is judged by a Large Language Model (LLM), with regular expressions and the operators of a temporal logic. The LLM answers only local questions about individual text passages; a symbolic engine then aggregates these judgments over time and links every match to the recorded passages and predicate decisions behind it. With this division of labor, the temporal and relational reasoning stays exact and auditable even over long time horizons. In addition, when a query includes a regular-expression pre-filter, only a small fraction of the corpus ever reaches the LLM. On a suite of controlled synthetic benchmarks with temporally and semantically complex patterns, TL;QR is substantially more accurate than LLM-only baselines. In a case study on ten years of European Parliament debates (442,593 multilingual speeches), TL;QR identifies candidate shifts in the political groups' stances on the GDPR privacy regulation. 

---
# bharatSetu: a 1M x 22 bridge for indic translation 

**Authors**: Haq Nawaz Malik, Nahfid Nissar, Faizan Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2610.15705)  

**Abstract**: India's Eighth Schedule recognises 22 languages, yet parallel data is uneven across them, and many language pairs, especially among lower-resource languages such as Bodo, Dogri, Santali and Kashmiri, lack direct bitext. We present BharatSetu 22-Indic 1M, a row-aligned table of 984,764 English sentences, each paired with machine translations into all 22 scheduled languages (21,664,808 Indic cells), generated with the sarvamai/sarvam-translate model family. Because every Indic cell in a row translates the same English sentence, any two columns form a translation pair: 462 directed Indic-to-Indic pairs, or 506 including English, without direct bitext. We call these pivot-derived pairs and propose a confidence layer that lets users choose how strictly each pair is connected. The corpus is machine-generated and not human-verified; quality is expected to vary by language, domain and sentence style, and pivot-derived pairs can inherit errors from both sides. We therefore release BharatSetu as a machine layer and propose a human-correction layer keyed to the same row identifiers, so that each corrected column upgrades every pair that uses it, together with an evaluation protocol. We report no translation-quality results. 

---
# Do LLM Tutors Actually Teach? A Process-Centered Synthesis of the Tutoring Loop 

**Authors**: Sayantan Adak, Rima Hazra, Koyel Mukherjee, Digbalay Bose, Somnath Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2610.15692)  

**Abstract**: Large language models can produce fluent tu- toring dialogue, but whether they actually help students learn remains unclear. We present a process-centered synthesis of LLM tutoring that follows the path from learner-state infer- ence to pedagogical action, tutor training, eval- uation, and student simulation. This view ex- poses a central gap: implementing a tutoring component does not establish that it works or that its outputs meaningfully affect later stages. A structured study-level audit shows that peda- gogical evaluation is common, while evidence becomes much thinner for learner-conditioned intervention, human learning gains, and espe- cially transfer or delayed retention. Human studies nevertheless provide a conditional posi- tive result: LLM tutors can improve learning, but benefits depend on pedagogical structure, learner engagement, and how learning is mea- sured. We argue that progress requires validat- ing the interfaces between tutoring stages, not only improving individual components. 

---
# Towards Autopilot Labeling for Short Video Content Governance 

**Authors**: Junwen Chen, Hongwei Wang, Zixuan Wang, Jingbing Wang, Gorden Liu, Sheng Siong Ong, Xiaojiao Zang, JuChin Chao, Ze Jin, Kaiwen Men, Yin Li, Jeremy Li, Zhuolin Hao, Hongyu Xiong, Yang Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.15687)  

**Abstract**: The explosive growth of short videos has presented significant challenges to content governance. Traditional manual labeling is inefficient, slow to iterate, and costly. Therefore, we explore and develop a multi-stage autopilot labeling framework aimed at progressively increasing the level of automation in short video content annotation. To move beyond the heavy reliance on human expert labeling at the initial L0 stage, we first introduce a discriminative MLLM labeler at the L1 stage, which is fine-tuned with supervised data to directly predict labels for short videos. Advancing to the L2 stage, we shift to a domain-adaptive MLLM pretrained on short video data and use it for zero-shot labeling, enabling flexible generalization without task-specific supervision. In the L3 stage, we propose an agentic annotation workflow, and equip the agent with several key tools, which empower the agent to handle complex, unfamiliar, or vaguely defined cases, significantly reducing the need for human intervention. We conduct extensive experiments across various content issues, and results demonstrate that each stage of the autopilot labeling framework progressively improves labeling accuracy, and has significantly reduced manual labor costs in real-world applications by up to 85%. 

---
# The Mathematical Collective Unconscious: Jungian Archetypes within the Latent Space 

**Authors**: Andrew Breen  

**Link**: [PDF](https://arxiv.org/pdf/2610.15678)  

**Abstract**: Carl Jung theorised that the human psyche contains an inherited framework, the collective unconscious, populated by universal archetypes. As large language models (LLMs) are trained on vast corpora of textual data, this paper investigates whether they spontaneously construct distinct archetype-like regions within their high-dimensional latent spaces. The paper proposes that these regions function as spatial constraints that naturally organise semantically related narratives. To evaluate this, we applied cosine similarity and angular distance metrics across thirteen experiments utilising OpenAIs text-embedding-3-small model. The paper contains two experimental sections the first section explores these ideas using synthetically constructed narratives, the results of which act as a baselines for experiments conducted in the second section where the synthetic data is replaced by human-authored literature (Shmoop corpus), unedited dream reports (DreamBank corpus), and culturally distilled folklore (Andrew Langs Fairy Books). The results show that archetypal roles occupy distinct, measurable regions in the latent space. Crucially, these regions remain compact and consistent because of narrative structure rather than simple word overlap. Additionally, text category dictates spatial distribution: dream reports concentrate within Trickster and Shadow regions, while traditional folklore distributes evenly across all tested archetypes. We conclude that transformer latent spaces encode stable regions which appear to correspond to recognised Jungian archetypes. 

---
# Rate-Resolved Text Evaluation with Masked Diffusion Language Models 

**Authors**: Yiqi Liu, Kun Zhao, Chenghua Lin, Nafise Sadat Moosavi  

**Link**: [PDF](https://arxiv.org/pdf/2610.15667)  

**Abstract**: A single evaluation score can rank generation systems but rarely reveals what failed. We propose rate-resolved text evaluation, which turns the masking rate of masked diffusion language models into a diagnostic axis. By reconstructing text under different fractions of masked tokens, one diffusion score becomes a vector of band-conditioned scores that reflect reconstruction difficulty at varying levels of visible context. On SummEval, preserving this rate structure yields complementary signal: the best masking-rate band improves fusion with autoregressive likelihood over a collapsed diffusion scalar, survives a sample-shuffle control that breaks rate ordering, and adds residual signal to a fixed prompted judge. A second diffusion model reproduces the effect more weakly. On WMT MQM zh-en, roundtrip mask-and-reconstruct probing produces a tunable localization curve in which ranking-oriented AUROC and span-detection F1 peak at different masking rates. Factuality benchmarks bound the interpretation: band semantics are model- and task-dependent, so each application requires empirical validation. 

---
# Feedback as an Interface: Controlling Form and Detail in LLM Translation Refinement 

**Authors**: Yiqi Liu, Flavjo Tomori, Nafise Sadat Moosavi, Steffen Eger, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.15661)  

**Abstract**: Feedback presentation shapes how LLMs use evaluator judgments during translation refinement. We introduce a machine translation protocol that keeps the evaluator score, main issue, and recommendation fixed while varying feedback form and supporting detail. Across seven models and six translation directions, low-load feedback outperforms high-load feedback in all 42 translator-direction settings under xCOMET. On the 29,400-record rewriting subset, mean xCOMET scores multiplied by 100 are 83.30 for low-load feedback and 80.67 for high-load feedback; the full-matrix original-translation reference mean is 83.46. MetricX-24 supports the low-load advantage on a stratified subset. A no-feedback second pass yields a small improvement over the original translations, and direct evaluator prompting reproduces the low-load advantage without a separate rewriting stage. These results show that compact feedback limits the quality loss from evaluator-guided refinement in the evaluated high-resource MT settings. 

---
# Towards Robust Calibration Dataset for Domain-General Mixture of Experts Pruning 

**Authors**: Luis Frentzen Salim, Chia-Chun Wu, Tran Van Nhiem, Lun-Wei Ku, Yung-Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.15578)  

**Abstract**: Post-training expert pruning is a promising deployment technique for Mixture-of-Experts (MoE) models. The calibration set is an integral part of the pruning process, however, its correlation with evaluation performance is largely misunderstood and unintuitive. In this work, we present an empirical study of the robustness of calibration perturbations, specifically in MoE models. Showing how calibration size, language, and data distribution affect layer-wise expert pruning for gpt-oss-20b. Our findings reveal that (1) domain-general pruning quality receives minimal benefit from scaling calibration data beyond small sample sizes, (2) cross-lingual calibration transfers effectively, with English-only calibration maintaining strong performance across seven languages of varying typologies and scripts, and (3) a diverse calibration data resembling the pretraining distribution can yield comparable or superior performance to test-distribution data. Ultimately, we provide practical insights as a recipe for better understanding calibration for more effective domain-general expert pruning. 

---
# The Fragility and Dynamics of LLM Pruning-Recovery Steps 

**Authors**: Luis Frentzen Salim, Komang Rinartha, Chia-Chun Wu, Tran Van Nhiem, Lun-Wei Ku, Yung-Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.15574)  

**Abstract**: Structural pruning followed by a recovery stage is the standard recipe for compressing large language models (LLMs), yet the two stages are typically designed and tuned in isolation. In this work, we study how their interaction shapes the compressed model, sweeping ordering, the data that drives each stage, and the alignment of their training objectives. We find that instruction-following is the first capability to collapse under pruning, well before knowledge, reasoning, or core language modeling, and that much of the pruning in-domain loss is likely not knowledge loss but a domain-agnostic failure to respond coherently to the prompt. We find in our setting that in-domain data is the strongest pruning signal, whereas instruction-style data recovers the most. Stage order matters little, but only marginally. We further analyze how the importance ranking drifts before and after pruning. Together these results give practitioners a concrete recipe for composing pruning and recovery. 

---
# An Interpretable Linguistically-Grounded Analysis of Scene Saliency in Movie Screenplays 

**Authors**: Uday Bindal, Ishaan Karan, Manan Garg, Ekansh Goyal, Rahul Mishra, Rajakrishnan Rajkumar  

**Link**: [PDF](https://arxiv.org/pdf/2610.15544)  

**Abstract**: We investigate the factors affecting salience in movie screenplay by posing two research questions: 1. {\em Does Uniform Information Density (UID) predict salient scenes?} 2. {\em What other linguistic properties distinguish salient scenes from non-salient background?} We answer both questions through a controlled corpus study on the MENSA corpus, paired with a Linguistically-Grounded Salience Model (LGSM), a dual-stream classifier that fuses a frozen BERT encoder with explicit linguistic features through adaptive gating. The UID picture is layered in long-form discourse. Per-scene UID summaries carry no independent signal once length is controlled, but at the token level salient scenes match non-salient material in the bulk while specifically suppressing the rarest, hardest-to-predict tokens in the upper tail. UID therefore lives at the within-scene token level rather than as a per-scene aggregate. Salient scenes also show a moderate and sustained cast of central characters (following an inverted-U pattern), strong third-person usage, syntactically simpler language, and more abstract vocabulary about motivation and conflict. LGSM exploits this signature and improves recall on salient scenes over a strong semantic baseline while yielding better downstream summaries. 

---
# Apertus 1.5: From Open Text Models to Open Multimodal Reasoners 

**Authors**: Project Apertus, Yixuan Xu, Yassine Turki, Vinko Sabolčec, Valentina Pyatkin, Tommy Chu, Sven Najem-Meyer, Sean F. Carroll, Robert Smith, Raphael Kreft, Oscar Alessandro Tazza, Oleg Lavrovsky, Nicola Irmiger, Matteo Santelmo, Marian Schneider, Juan García Giraldo, Hanna Yukhymenko, Eduard Ďurech, Davit Melikidze, Dani Tamayo, Bettina Messmer, Aryan Ahadinia, Anunay Yadav, Allen Huang, Ahmad Fraij, Alessandro Bombarda, Alex Padula, Alexander Hägele, Alexander Sternfeld, Anna Hedström, Anna Kosovskaia, Arthur Wuhrmann, Arundhati Balasubramaniam, Badr Al Mahouri, Barna Pasztor, Bartosz Szostakiewicz, Bich Ngoc Doan, Bryan Schmid, Burak Sezen, Camille Challier, Clément Charmillot, Dana Shayakhmetova, Daniel Perruchoud, David Brüggemann, David Isztl, Elia Palme, Emma Anastassova, Fabian Bösch, Faruk Zahiragić, Fawzi Roberto Mohamed, Fiona Panicali, Florian Kolly, Flynn Guerra, Francesco Monti, Frederike Lübeck, Gennaro Oliva, Haofeng Yang, Haoliang Shang, Helen El Khoury, Henrique Mendonça, Jannis Vamvas, Jérome Landolt, Jessica Lam, Jingchun Xiao, Jonathan Coles, Joshua Oyebanji, Khanh Nguyen, Konstantinos Pitas, Lasse B. Strand, Li Mi, Loïc Deslarzes, Lorenzo Baggi, Lorenzo Noci, Luca Baumann, Luca Sartori, Lukas Drescher, Marco De Negri, Marco Passerini, Mark Klein, Marko Stojanovic, Martin Majsec, Matin Ansaripour, Mattea Busato, Mauro Pellonara, Max Neuwinger, Michael Graber, Michal Tešnar, Michelle Wastl, Miguel Gila, Mikolaj Boronski, Morgane Magnin, Moritz Reihs, Mukhammadali Sayfiddinov, Nathan Ranchin, Nicolas Cuveillier, Oscar Wohlfahrt, Qianqian Zou, Rada Kamysheva, Rita Sevastjanova, Roman Machacek  

**Link**: [PDF](https://arxiv.org/pdf/2610.15542)  

**Abstract**: Language models are becoming infrastructure, and infrastructure built in the open, by public institutions, on documented data, is the alternative to dependence on a few providers; Apertus shows this is viable at scale. Apertus 1.5 extends the fully open Apertus family from text to multimodal reasoning. Continued pretraining of the Apertus 1.0 models, on 4T additional tokens for the 8B and 2T for the 70B, adds image and audio understanding and extends the context window fourfold to 262,144 tokens; multimodality enters through the tokenizer, the language-model head remains restricted to text, and the architecture is unchanged. Supervised finetuning, long-context finetuning, reinforcement learning with verifiable rewards, and preference alignment complete the released models. Thinking is a flag of the chat template rather than a separate model, and lifts mathematical reasoning by up to 20 points on the 8B model. The released models improve over their predecessors on nearly every benchmark; among fully open models of its class, the 8B leads on knowledge, instruction following, and cultural benchmarks, and its instruction following surpasses all compared models, including open-weight ones. On image and audio understanding, a first for the family, the models are competitive with fully open multimodal models, strongest on document understanding and long-form and lower-resource speech recognition, while specialized vision and audio models remain ahead on perception and transcription of English and other high-resource languages. As in Apertus 1.0, we train only on openly available data, respect crawler opt-outs retroactively, and exclude restrictively licensed material. We release the model weights, data preparation and training recipes, and evaluation suites under permissive licenses; openness extends to values, with the Apertus charter guiding alignment stages published alongside the models. 

---
# A Step in the Right Place: Understanding and Optimizing Timestep Placement in Continuous Diffusion Language Models 

**Authors**: Shutian Zheng, Zhihong Yu, Linye Wei, Zeyang Zhang, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.15527)  

**Abstract**: Continuous diffusion language models represent an emerging paradigm for language modeling. They generate text in parallel by iteratively refining continuous representations and mapping them back to tokens through a decoder. However, realizing the efficiency advantage of parallel generation requires maintaining generation quality with only a few denoising steps. Even when the number of steps is fixed, their placement along the denoising trajectory remains flexible. In this paper, we demonstrate that changing timestep placement alone can substantially alter the final latent representations and thereby affect the decoded token distributions. Building on these findings, we introduce StepTune, an algorithm that searches for timestep placements that improve generation quality. The search is conducted on a small calibration set and guided by a language-aware objective, with the model and sampling budget fixed. Once calibrated, the timestep placement is fixed and reused across inputs. Experiments demonstrate consistent improvements in unconditional generation perplexity and conditional task metrics across all evaluated sampling budgets. In particular, StepTune enables 16-step sampling with a frozen model to outperform its native 64-step baseline on both XSum summarization and WMT14 translation and reduces end-to-end latency by approximately $3.8\times$. These results establish timestep placement as a practical means of improving the quality-efficiency trade-off in continuous diffusion language models. 

---
# REPRO: Reflective Evolutionary Prompt Optimizer for Automated LLM Red-Teaming 

**Authors**: Abrar Alotaibi, Moataz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2610.15513)  

**Abstract**: Safety-aligned large language models (LLMs) remain vulnerable to adversarial prompts that bypass their safety training. Existing automated red-teaming methods either require privileged access unavailable in production, rely on manually engineered per-model templates, or optimize a single objective without population-level diversity. We present REPRO (Reflective Evolutionary Prompt Optimization), a pure black-box framework that discovers adversarial prompts through evolutionary optimization driven entirely by a compressed open-weight 7B reflective model on a single consumer GPU. REPRO introduces two mechanisms absent from prior work, both isolated in a component ablation: semantic crossover, an LLM-guided multi-parent recombination operator, and Bayesian strategy selection via Thompson Sampling informed by structured failure analysis.
We catalog REPRO's behavior on AdvBench, JailbreakBench, and HarmBench across 36 target models from 11 families, including GPT-5-nano, Claude-3.5-Sonnet, Gemini-2.5-Pro, and Grok-4.1. Under our protocol REPRO reaches 100\% attack success rate (ASR) on every evaluated target in a single turn, at a median of 24 target queries per behavior. Two properties of the protocol qualify that figure. The judge scoring candidates during optimization also produces the reported ASR, and search halts when it accepts, so ASR reports whether search succeeded within the iteration budget rather than success at a fixed budget. All baseline figures are transcribed from their original papers, which use different judges, budgets, and model versions. We therefore present this as a catalog of model coverage under one protocol rather than a head-to-head ranking, and develop both qualifications in the paper. 

---
# Concept Representations in Vector Space: Quantifying Sharedness and Idiosyncrasy of Meaning 

**Authors**: Simon Josef Durstewitz, Denis Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2610.15496)  

**Abstract**: Concepts are the basic units through which we navigate the world and communicate meaning. But do you and I really mean the same thing when we think or speak about a concept? The present study develops a measurement framework for modeling and analyzing interindividual variation in concept representations through sentence embeddings, dense vector representations mapping entire utterances into a shared geometric space. The framework is developed in a theoretical argument and validated, demonstrated, and subjected to robustness checks in a pre-registered study. Short personal definitions, elicited from $N = 125$ participants, are encoded as sentence embeddings and submitted pairwise to a crossed random effects model yielding two concept-level quantities: sharedness ($S_b$), the expected dissimilarity between two randomly drawn individuals, and idiosyncrasy ($I_b$), the person-specific share of total dyadic variance. Two validity tests support the approach: the embedding space fully recovers concept and domain organization from participant definitions even with the target word excluded, and a parameter-free classifier achieves 0.630 accuracy in a within-participant extensional prediction task against a chance baseline of 0.500 and a consistency ceiling of 0.893. The framework finds theoretically interpretable variation in $S_b$ and establishes $I_b$ as a reliable source of dyadic variance across concepts. Results replicate across three encoders varying in architecture and capacity, providing a reproducible and scalable instrument for the empirical study of meaning variation. 

---
# AlphaPilot: Multi-Turn Tool-Using LLM Agents for Adaptive Alpha Mining with Reinforcement Learning 

**Authors**: Peiqi Jiang, Fan Zhang, Xin Xu, Tianhao Chen, Can Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.15488)  

**Abstract**: Formulaic alpha mining seeks interpretable expressions that rank assets by future returns. The task is challenging because it demands simultaneous processing of unstructured market descriptions, historical data, and quantitative evaluation of candidate factors, yet most existing methods exploit only a subset of these signals. We introduce AlphaPilot, a reinforcement-learning framework that trains multi-turn tool-using LLM agents for adaptive alpha mining. The agent conditions on market descriptions, proposes factor expressions, queries real-data evaluator feedback, and updates a retained factor pool across turns. This design jointly leverages market text, LLM financial knowledge, real data, and evaluation feedback. At inference time, AlphaPilot performs day-by-day search and pool updates to adapt to shifting market regimes. Across three independent runs in out-of-sample rolling backtests on CSI300 and CSI500, AlphaPilot achieves mean total returns of 42.93% and 54.19%, with mean Sharpe ratios of 2.46 and 2.86, respectively. It attains the highest mean total return, Sharpe, and RankIC among the evaluated methods on both universes. 

---
# Sensitivity or Criterion? Measuring and Moving When Language Models Act 

**Authors**: Sixun Dong, Yebowen Hu, Ming Yin, Yiran Chen, Fei Liu, Chen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.15415)  

**Abstract**: Before an agent acts, it must decide whether action is warranted. A single accuracy cannot distinguish failure to recognize a warranted action from reluctance to act. Signal-detection theory separates sensitivity from criterion, but criteria estimated from authored binary labels depend on each benchmark's example mix. We introduce FlipAct, a diagnostic of tool use in physical states. A rule over annotated tool properties computes each answer and a graded evidence score, fixing the Yes/No boundary at zero across models. Questions form 2-by-2 quads that hold required knowledge fixed while the state flips. The slope of the fitted Yes-rate curve measures sensitivity; its crossing at one half measures criterion. Across 44 open models, sensitivity rises with size within a family, and its ranking agrees with three of four external when-to-act benchmarks. Models respond to evidence sign rather than magnitude, accepting actions that leave the state no better about as often as helpful ones. When Yes means act, most models are conservative across rephrasings; in some, this partly reflects a preference for No. Criterion is not determined by size. Activation steering shifts criterion with weights fixed; distillation from a model's own steered copy puts the shift into its weights without a significant sensitivity change. The distilled shift transfers to six external benchmarks, including two without Yes/No answers, while criterion-free AUC is preserved on the five benchmarks where it is available. Accuracy improves where models start conservative and declines or remains within error where they start permissive. Project page: this https URL 

---
# From Candidate Tokens to Final Outcomes: Decoding Relative Long-Horizon Candidate Value from Reasoning Model Representations 

**Authors**: Yixin Yang, Yue Wang, Fandong Meng, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2610.15412)  

**Abstract**: Large reasoning models generate reasoning trajectories token by token. At a given prefix, multiple next-token candidates may be plausible, yet shape subsequent reasoning and differ in their chances of leading to the correct answer. Whether these differences are reflected in internal representations remains unclear. To study this question, we define relative long-horizon candidate value to quantify within-prefix differences and construct the Reasoning Candidate-Token Value (RCTV) dataset, pairing mathematical reasoning prefixes with plausible next-token candidates and their empirical long-horizon candidate values. Using RCTV, we examine hidden states, multilayer perceptron outputs, and attention outputs at candidate positions with sparse linear probes. Experimental results provide evidence that relative long-horizon candidate value is linearly decodable from these representations at multiple layers. Candidate-value probes can also outperform probability- and embedding-based baselines. Furthermore, a controlled online intervention experiment shows that using the decoded signals to guide token selection during generation can improve the chances of eventual success. Our findings provide evidence that internal representations encode information about relative long-horizon candidate value, offering insights into how local candidate-level representations relate to long-horizon reasoning outcomes. 

---
# In-Place Indic Tokenizer Expansion and Continual Pretraining of Nemotron-3-Nano-30B-A3B 

**Authors**: Avinash Benki, Prasanjith Pasunuti, Sutanshu Saksena Raj, Parijat Chatterjee, Nayanjit Sarkar, Daksh Maan  

**Link**: [PDF](https://arxiv.org/pdf/2610.15373)  

**Abstract**: We adapted Nemotron-3-Nano-30B-A3B for eight Indian languages without changing its backbone architecture. We expanded the vocabulary in place from 131,072 to 163,072 entries using continued byte-pair encoding and AdaptMix rebalancing, reducing tokens per word by 38.6%. We initialized new input and output rows by averaging their original constituent rows, then trained them on 90% Indic and 10% English text before updating the backbone. Continual pretraining then progressed through broad training, high-quality annealing, and long-context training over approximately 1.27 trillion tokens. The annealed checkpoint gained 12.9 points on the Indic MILU benchmark but lost 3.4 on English MMLU. We also report practical negative results. Direct token addition underperformed continued byte-pair encoding. AdamW weight decay changed rows intended to stay fixed. Perplexity on an Indic-heavy holdout did not select the learning rate that best preserved English performance. 

---
# A Sovereign, Indic-First, Reasoning Model 

**Authors**: Sutanshu Saksena Raj, Prasanjith Pasunuti, Parijat Chatterjee, Daksh Maan, Purusharth Malik, Nayanjit Sarkar, Avinash Benki  

**Link**: [PDF](https://arxiv.org/pdf/2610.15371)  

**Abstract**: Gnani-Evon-v3.3-30B-A3B is a long-context hybrid Mamba-Transformer mixture-of-experts model for reasoning, instruction following, and tool use in English and eight Indian languages. Starting from NVIDIA's Nemotron-3-Nano-30B-A3B checkpoint, it retains the 52-layer backbone and changes the vocabulary, training data, and post-training objectives. An in-place expansion from 131,072 to 163,072 tokens reduces mean Indian-language fertility by 38.6%. The new embedding rows are initialized from their upstream decompositions and warmed before the backbone is updated. Continual pretraining then adds broad English-Indic data, high-quality annealing, and direct training at a 131,072-token sequence length.
The central result appears after post-training. Reinforcement learning across four verifiable task environments improves mathematics, graduate-level science, and instruction following, but degrades Indic extractive question answering because the rewards do not require an answer in the requested language. The share of Indic extraction items answered in English rises from 8.7% to 91.7%. Accuracy conditional on answering in the target script roughly doubles, which is consistent with preserved reading ability but is measured on a self-selected subset.
Two unreleased studies test repairs. A scoped Indic reward restores the language convention. A 26B-A4B generative judge outperforms a 235B-parameter model on RM-Bench, and distillation restores 81% of the capabilities displaced by optimizing against it.
Relative to the architecture-identical starting checkpoint, the released model gains 5.2 points on an Indic aggregate while giving up 1.7 points of English retention. The unreleased Indic-aware branch trades further, gaining 14.6 Indic points at a cost of 5.4 in English. A planned FP8 release reduces the artifact to 32GB with near-zero mean change across 32 paired benchmarks. 

---
# North or South? Diagnosing LLMs' Understanding of North Korean Language, Society, and Knowledge 

**Authors**: Kiwoong Park, Seyoung Song, Donghyun Woo, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2610.15282)  

**Abstract**: Large Language Models (LLMs) have made substantial progress in multilingual question answering, yet their ability to understand low-resource sociolinguistic contexts remains under-explored. We study North Korea as a unique testbed where decades of division have produced distinct differences from South Korea in vocabulary, orthography, norms, and ideological discourse. We introduce LENS (Language and sociocultural Evaluation for North Korean Society), a human-validated benchmark of 1,765 questions evaluating LLMs across three dimensions: linguistic robustness, sociocultural perspective alignment, and domain-specific knowledge. Evaluating diverse recent LLMs, we find a modest performance drop on North Korean language inputs, a larger gap in sociocultural perspective alignment, and severe limitations on domain-specific knowledge. Qualitative analysis reveals that models often conflate formal institutions with lived realities, project liberal-democratic assumptions, or rely on coarse stereotypes. Our results demonstrate that linguistic similarity alone does not guarantee sociocultural or domain-specific understanding, highlighting the critical need for grounded evaluation resources for underrepresented cultural and political contexts. 

---
# Do correct answers have stable latent knowledge support? 

**Authors**: Chuancheng Shi, Shangze Li, Wenhua Wu, Kun Hu, Fei Shen, Zhiyong Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2610.15277)  

**Abstract**: Large language models are typically evaluated in terms of the correctness of their output. However, output correctness alone does not confirm whether the latent knowledge supporting an answer has stable internal representations. We therefore propose latent state stability (LSS) and knowledge topology stability (KTS) to investigate the stability of individual knowledge states and their relational structure, respectively. Specifically, we first separately average the model's internal representations of the same knowledge under each elicitation condition. Using the mean representation under the baseline condition as a reference, we remove shared shifts to obtain corrected representations. We then formulate LSS as the distance between each corrected representation and its reference representation, normalized against the typical distance between knowledge items of the same type. We formulate KTS as the cross-condition consistency of the ordered pairwise representation distances among knowledge items of the same type. Next, we construct EpiStableBench, which comprises 4,411 items spanning factual knowledge and mathematical reasoning, with 76,766 controlled queries. Across 16 LLMs from four families, representation stability varies substantially even among items answered correctly under the base condition. Within models, higher stability generally corresponds to higher accuracy under held-out elicitation conditions; across models, stability does not consistently increase with model size or overall accuracy. Meanwhile, we present LatentDPO to learn a knowledge-editing policy that accounts for the stability of latent knowledge. Experimental results show that integrating LatentDPO with multiple existing editing methods consistently improves editing success and generalization across different expressions. 

---
# Sequent-Prover: Training Agents for Formal, Checkable SMT-based Reasoning 

**Authors**: Debargha Ganguly, Weicong Chen, Mohsen Hariri, Vikash Singh, Sreehari Sankar, Shouren Wang, Wang Yang, Biyao Zhang, Chaoda Song, Alexander Nemecek, Kai Ye, Osama Zafar, Erman Ayday, Joe Stubbs, Srinivasan Iyengar, Vipin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2610.15258)  

**Abstract**: We introduce Sequent-Prover, a family of verifier-using language-model agents, and the post-training framework that produces them. Agents reason in an executable SMT-LIB scratchpad: they write complete SMT-LIB programs, query Z3 across turns, inspect solver feedback, repair formalizations, and terminate with an explicit answer. Our recipe, verifier-conditioned trajectory distillation, bootstraps from a noisy teacher proposal policy by sampling multi-turn trajectories, retaining the successful tail that is correct, non-trivial, grounded, and executable, then applying supervised fine-tuning and trajectory-level preference optimization. Because solver success alone does not guarantee semantic faithfulness, we evaluate task accuracy, k-sample reliability, and SMT program quality. Across 37 benchmarks and 7,254 problems per model family, post-training raises the 4B model's per-attempt accuracy from 85.6% to 97.0% and four-sample all-correct reliability (pass^4}) from 64.1% to 91.7%. It also reduces average SMT calls from 4.10 to 2.52 and tokens per attempt from 18.97k to 12.72k. Sequent-Prover agents give answers backed by executable formal artifacts. 

---
# PANINI-DR: Controlled Deep Research through Generative Semantic Workspaces 

**Authors**: M. Yigit Turali, Shreyas Rajesh, M. Efe Lorasdagi, Chenda Duan, Kartik Sharma, Vwani Roychowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2610.15208)  

**Abstract**: Deep research requires connecting multiple steps of search, reasoning, and computation, with intermediate findings guiding what to investigate next. As investigations grow, agents must maintain research state across model calls and check findings before errors propagate into subsequent work. We introduce PANINI-DR, a multi-agent harness that separates coordination of the overall investigation from the research and verification of individual subquestions. A shared workspace, built on the Generative Semantic Workspace representation, records subquestions, their dependencies, findings, and supporting evidence. Researchers investigate individual subquestions, while a separate verification agent checks proposed answers against those subquestions and their evidence. Verification feedback guides further investigation, and recorded findings inform subsequent assignments. We evaluate PANINI-DR on BrowseComp-Plus, FRAMES, and MoNaCo against seven state-of-the-art deep-research harnesses, all using a shared language-model backbone and frozen corpora. PANINI-DR achieves the strongest average performance across the three benchmarks. On BrowseComp-Plus, it achieves 73.3% accuracy, outperforming the next-best harness by 16.0 points. Its advantage is strongest on hard questions, where it answers nearly twice as many as the next-best harness. Ablations further show that structured planning and persistent state improve performance, as does checking each finding against its cited evidence before it is reused. Because a model can answer correctly from memory alone, we also analyze the evidence retained by each harness. PANINI-DR achieves the highest answer recoverability from saved evidence on all three benchmarks, while complementary claim-support audits show stronger evidence-backed answers. Accounting for this evidence changes how harnesses compare and further strengthens PANINI-DR's advantage. 

---
# Do LLMs differentiate linguistic traits internally?: A layer-wise analysis 

**Authors**: Samhita Bollepally, Takashi Yamauchi, Manju Muralee, Motunrayo Adebimpe  

**Link**: [PDF](https://arxiv.org/pdf/2610.15203)  

**Abstract**: Large language models often produce judgments that resemble human responses, but it remains unclear whether they internally capture linguistic nuances such as conventional versus idiomatic meaning. A layer-wise representational analysis was conducted on two equal-sized pre-trained models, Llama-3.2-1B and Gemma-3-1B-PT, to examine whether and where these models differentiate linguistic traits from input. The analysis combined layer-wise probing with geometric distance measures, using the top embedding dimensions ranked by principal component loadings for each layer. The results show that even with equal capacity, models organize linguistic information in fundamentally different ways, suggesting the need to analyze internal representational geometry, examining not just behavioral outputs but semantic space alignment when evaluating how closely human and model language interpretation correspond. 

---
# SAFE-KV: Causal Suffix Repair for Behavior-Preserving Non-Prefix KV Cache Reuse in RAG 

**Authors**: Xingjian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.15186)  

**Abstract**: Retrieval-augmented generation (RAG) reuses the same documents across many requests, making non-prefix KV-cache reuse attractive. Yet a cached document state is not reusable text: it is a context-conditioned causal state whose keys and values depend on the prefix, position, and preceding documents under which it was computed. We formulate non-prefix KV reuse as a behavior-preservation problem with full prefill on the exact prompt as the reference, and introduce SAFE-KV, which repairs a composed cache by re-encoding the causal suffix from a selected start document. On Qwen2.5-7B-Instruct, prefix-conditioned composition preserves only 47.2% of HotpotQA-500 and 37.8% of 2Wiki-500 full-prefill answers; SAFE-KV with risk-guided K2 selection restores 99.8%/98.4% (donor-copy protocol) and 100.0%/98.4% with actual suffix forwarding. Document-only and fixed-window repairs recover far less, because the document states after the repaired region keep histories that omit it. Reuse without repair is exact up to four documents and degrades at sixteen, where suffix repair recovers 98-100%. Faithful repair re-encodes most of the context, so in a single-request L40S study SAFE-KV's operating rule serves every workload by full prefill. SAFE-KV thus establishes the causal-state contract for non-prefix reuse and a validated repair mechanism that meets it. 

---
# UDHRBench: Benchmarking Human Rights in Large Language Models 

**Authors**: Alak Raad, Noreldin Morssy, Zeerak Suhail, Serin Atiani, Mustafa Jarrar  

**Link**: [PDF](https://arxiv.org/pdf/2610.15179)  

**Abstract**: Large language models are increasingly deployed in contexts requiring adherence to ethical and legal norms, yet their alignment with internationally recognized human rights standards remains underexplored. We introduce UDHRBench, a large-scale benchmark for operationalising the evaluation of large language model compliance with the Universal Declaration of Human Rights. The dataset comprises 1,124 human rights confronting questions covering all 30 articles and 4,496 responses from four large language models (ChatGPT, Grok, Gemini, and Fanar), manually annotated by experts for alignment, with high inter-annotator agreement ($\kappa = 0.801$). Our analysis reveals systematic and non-trivial misalignments, showing that violations typically arise through seemingly rational reasoning: appeals to efficiency, risk management, or scientific neutrality, rather than explicit rejection of rights. We further propose an LLM-as-a-Judge framework for automated evaluation using zero-shot and few-shot prompting, achieving 89.8\% weighted F1-score in detecting human-rights misalignment, alongside a novel Alignment Leniency Score that quantifies how judge models resolve ambiguous cases. The dataset, annotations, and code will be released as open-source resources (this https URL). 

---
# Active Learning for Finding a Needle in a Haystack 

**Authors**: Emily Silcock, Simon Löwe  

**Link**: [PDF](https://arxiv.org/pdf/2610.15165)  

**Abstract**: Training a classifier to find a rare class of documents, such as hate speech directed at a particular religious group, requires labelled examples that are expensive to obtain when the class makes up only a small fraction of a corpus. Active learning reduces this cost, but it must start from an initial labelled set that contains examples of the target class. Random sampling cannot supply these at reasonable cost, so they must be selected in some other way, such as keyword search, and any such selection biases the initial set towards part of the target class. We evaluate whether active learning can correct this bias. On AG News and Measuring Hate Speech, at a prevalence of around 1%, we compare random sampling with eight active learning strategies, starting from initial sets that cover or omit known subgroups of the target class, and measure recall on the omitted subgroups as well as F1. On AG News, almost every strategy recovers the omitted subgroup. On Measuring Hate Speech, uncertainty-based strategies recover less of the omitted subgroups than random sampling, whereas discriminative active learning recovers 55% of them without a significant loss of F1. Aggregate metrics conceal these differences. 

---
# Sanzaru: Benchmarking Protocol Formation under Epistemic & Communication Asymmetry 

**Authors**: Yiping Li, Zhiyu An, Wan Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.15163)  

**Abstract**: A language model's ability to solve a task individually does not ensure that a team can solve it under unequal access to information and communication. SANZARU examines this gap through cooperative bomb-defusal puzzles adapted from BOMBANANA! (Lefto Studio, 2026). Task information and action authority are divided among three agents: one cannot hear, another cannot speak, and the third cannot see. The benchmark contains 120 puzzle instances across seven families, each targeting a distinct coordination ability. Across 1512 episodes, 7 models achieve an overall success rate of 49%, ranging from 20% to 86% under the evaluated configurations. Teams develop or adopt diverse communication protocols, from simple gesture codes to Morse and encodings of entire manuals. Yet trace analysis attributes about 60% of 771 team losses to communication, and agents attempt to use channels unavailable to their teammates despite explicit role instructions. Together, task outcomes and interaction traces enable evaluation of both collaborative performance and perspective taking throughout cooperation. The benchmark is available at this https URL. 

---
# Learning Proposition-Level Links from Sentence-Level Supervision for Neuro-Symbolic NLI 

**Authors**: Xuyao Feng, Antonis Bikakis  

**Link**: [PDF](https://arxiv.org/pdf/2610.15158)  

**Abstract**: We study neuro-symbolic natural language inference (NLI) with independently translated propositional representations of the premise and hypothesis. Before symbolic reasoning, the system must infer semantic links between their proposition-level units (atoms). Existing systems based on Abstract Meaning Representation (AMR) and Boolean satisfiability (SAT) identify these links using fixed embedding-based source selection and pretrained sentence-level NLI models not trained for atom-level linking. We introduce a trainable atom linker that jointly predicts link existence, source selection, and relation type. Without atom-pair training annotations, we treat cross-formula link structures as latent and train the linker from sentence-level NLI labels, keeping the translator and SAT reasoner fixed. We evaluate on the full public test splits of Stanford Natural Language Inference (SNLI) and Sentences Involving Compositional Knowledge (SICK), and the matched and mismatched validation splits of Multi-Genre Natural Language Inference (MNLI). Our system improves accuracy by 11.40-27.09 percentage points over a baseline applying sentence-trained NLI models unchanged to atom pairs within the same SAT pipeline. It also improves over the prior fixed-linker AMR-to-SAT system by 8.14-26.32 percentage points on SNLI and both MNLI splits, with a smaller gain on SICK. 

---
# Where Did We Go Wrong? Localizing Errors to Repair Multi-Turn Conversations 

**Authors**: Alessandro Morosini, Wei Sun, Georgia Perakis  

**Link**: [PDF](https://arxiv.org/pdf/2610.15154)  

**Abstract**: Large language model (LLM) assistants are increasingly used in multi-turn conversations, where the user's intent is often unclear at first and revealed only gradually over time. The assistant must therefore interpret each exchange and form inferences about the user's intent, using them to update its working state. These updates usually refine the assistant's understanding, but a misinterpretation can instead corrupt its working state by ruling out the true intent, influencing subsequent turns and ultimately causing the conversation to fail. Efficient recovery then requires locating and revising the misinterpretation without discarding information that remains valid. We introduce Loci (Localization Of Corrupted Inferences), a framework for probabilistically localizing misinterpretations and selectively repairing the assistant's working state. Given an observed failure, Loci reasons over possible user intents and assigns each update a probability of having excluded the true intent. These probabilities rank suspected updates and quantify the model's confidence in the localization of the underlying misinterpretations. Loci then uses this information in a repair policy that decides whether to retain, verify, or remove suspected updates, enabling recovery from corrupted inferences while preserving useful information. To evaluate Loci's ability to identify misinterpretations, we first test it in a controlled setup with injected errors. We find that Loci improves localization over the baselines, with its advantage becoming larger when multiple errors occur. We then evaluate it in free-form conversations, where misinterpretations arise naturally, and measure the interactions needed to complete the task after an initial failure. Using localization confidence to guide repair, Loci reduces mean recovery cost by 46-79% relative to the strongest baseline across multiple domains. 

---
# Lumen-bench: Cross-Lingual Safety Gaps and the Say-Do Problem in Tool-Calling Large Language Models 

**Authors**: Giovana Kerche Bonás, Thiago Laitz, Thales Sales Almeida, João Guilherme Alves Santos  

**Link**: [PDF](https://arxiv.org/pdf/2610.15134)  

**Abstract**: When a large language model refuses an unethical request, what does refusal mean? In conversation, refusal is bounded by what the model says; once the model is equipped with executable tools, what it does can diverge from what it says, a failure mode that English-only evaluations cannot fully reveal. We introduce Lumen-bench, a behavioral benchmark of tool-calling ethics spanning 30 models, five languages (Portuguese, English, Spanish, Mandarin Chinese, Hindi), 108 Portuguese-authored scenarios, six social-pressure conditions, two system prompts (ethical and neutral), and 21 generic tools. The evaluation comprises 194,400 trials. Judgments come from a reasoning-model judge (o4-mini), calibrated against one Portuguese-speaking author ($n=83$; harmful/non-harmful $\kappa=0.78$), compared with two other provider judges (Fleiss' $\kappa=0.78$, $n=607$), and complemented by retrospective sandbox replay. Four findings emerge. (i) Within-model comparisons support higher harmful-action rates in Hindi and Portuguese than English under the ethical prompt, and Hindi and Spanish under the neutral prompt, after multiple-testing correction. (ii) Premium models have lower median violation rates, although price does not guarantee safety or competence. (iii) Replacing the ethical system prompt with a neutral one raises pooled violation rates from 2.76\% to 13.64\%. (iv) In 8.57\% of substantive-call cases, refusal or ethical objection accompanies a harmful tool call: the judges consistently identify the say-do gap, although they differ in how many responses they assign to this category. Matched benign controls demonstrate legitimate task completion in 82.8\% of 3,300 trials. Code and reproduction scripts are available at this https URL. 

---
# RegExtract: A Benchmark for PFAS Regulatory Information Extraction with LLMs 

**Authors**: Defne Circi, Krystelle Lionti, Sarathkrishna Swaminathan, Daniel P. Sanders, L. Catherine Brinson, Bhuwan Dhingra, Basel Shbita, Mohab Elkaref, Sara Capponi, Tim Erdmann, Brandi Ransom  

**Link**: [PDF](https://arxiv.org/pdf/2610.15103)  

**Abstract**: Extracting structured information from regulatory documents is a critical but challenging task. Relevant facts are scattered across long, legally dense texts, and traceability of extracted facts to their source is essential for verification. With increasing awareness of the environmental and health impacts of per- and polyfluoroalkyl substances (PFAS), regulatory documents have expanded significantly, varying in scope and product relevance, making interpretation challenging for manufacturers and policymakers. We introduce RegExtract, a benchmark for structured information extraction from PFAS regulatory documents, comprising 488 human-annotated field instances across 20 US state documents and 46 regulations. Each field instance includes a value and a provenance entry, yielding 976 entries in total. We define a typed, multi-field schema targeting key elements such as material definitions, permitted uses, and product exemptions, and propose provenance-aware evaluation metrics tailored to regulatory text. Through experiments with open- and closed-source LLMs across prompting strategies, document condensing methods, and schema partitioning approaches, we find that document identification fields remain challenging, and that field-specific few-shot examples improve extraction. This work lays the groundwork for AI tools for hazardous substance identification, question answering, and compliance monitoring. The code and data are available at this https URL. 

---
# Have You Been Convinced? A Neuro-symbolic Argumentation Reasoning Approach to Detecting Persuasiveness in ChangeMyView Discussions 

**Authors**: Siddharth Bhargava, Sara Tonelli, Patricia Martín-Rodilla  

**Link**: [PDF](https://arxiv.org/pdf/2610.15091)  

**Abstract**: Argument structures represent how individual arguments interact within a discourse. However, leveraging these structures to infer meaningful knowledge remains challenging, as it requires a formal representation of how argumentative influence propagates across the structure. We propose Neuro-symbolic Argumentation Reasoning System (NARS) for performing persuasiveness detection directly from online discussions. The neural component employs retrieval-augmented iterative LLM reasoning to extract bipolar argument structures, while the symbolic component applies gradual argumentation semantics to propagate argumentative influence over the structure.
Using the Webis-CMV-20 corpus, we construct CMV2BAS, a dataset of long, multi-threaded ChangeMyView discussions annotated with LLM-derived bipolar argument structures and labeled with delta events (view change vs.\ no change). We investigate whether gradual argumentation semantics can effectively detect persuasiveness, whether augmenting them with argument quality and graph-based centrality improves performance, and whether symbolic reasoning remains necessary in the presence of modern LLMs. NARS achieves F1=0.57 and AUC=0.56 under its best configuration, outperforming zero-shot Qwen3.6:27B and GPT-oss:20B. We identify support-dominant discussions and long argumentative chains as key failure modes limiting attack influence propagation. Our work demonstrates the feasibility of operationalizing formal gradual argumentation semantics over noisy real-world discourse for scalable macro-level persuasion analysis, while highlighting key challenges and directions for future research in neuro-symbolic argumentation reasoning. We release both CMV2BAS dataset and the NARS framework for public use. 

---
# Select or Rewrite: Oracle-Guided Fine-Tuning for Generative ASR Post-Editing from N-Best Hypotheses 

**Authors**: Shiran Liu, Gaohui Xu, Sergio Burdisso, Thibault Bañeras-Roux, Shashi Kumar, Petr Motlicek  

**Link**: [PDF](https://arxiv.org/pdf/2610.15056)  

**Abstract**: Large language models can post-edit ASR by using linguistic context to compare N-best hypotheses and rewrite the transcript when needed, often improving WER without retraining the recognizer. However, prompt-only correction is brittle: it may over-edit when top-1 is correct and hallucinate when the reference is absent, and often needs very large models. We propose oracle-guided fine-tuning that turns N-best lists into supervision via three curated pools: corrections (reference in the list but not top-1) to teach selection, regularizers (top-1 correct) to discourage unnecessary rewrites, and hard negatives (reference absent) to suppress hallucination. Fine-tuning on this data provides more effective supervision than random subset fine-tuning for learning a hybrid select-or-rewrite policy, with the clearest gains from around 3B parameters on the proprietary dataset but less conclusive results on CommonVoice, a divergence suggesting that public read-speech text may overlap with LLM pretraining and mask genuine post-editing gains. 

---
# PAT: A Personality-Driven Augmentation and Transfer Learning Framework for Depression Detection on Social Media 

**Authors**: Shengwu Zheng, Yuzhe Ding, Xiaohui Liang, Fei Li, Chong Teng, Donghong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2610.15044)  

**Abstract**: Depression is a prevalent mental disorder that affects millions worldwide, with early detection crucial for effective intervention. While exist?ing methods have achieved promising results in automated depression detection, they face two key limitations: (1) noise in post screening: symptom-based semantic screening is easily misled by superficial lexical patterns, thereby introducing false-positive posts; (2) weak op?timization of the post encoder: relying solely on coarse user-level supervision signals pre?vents models from capturing depressive cues within individual posts. To address these limita?tions, we propose PAT, a Personality-driven Augmentation and Transfer learning frame?work. In the first phase, PAT optimizes a post?level encoder via a fine-grained contrastive learning objective, utilizing LLM-based data augmentation driven by classic personality the?ories to alleviate data scarcity. In the second phase, the trained post-level scorer is deployed for high-quality post filtering, while its en?coder parameters are seamlessly transferred for user-level tasks. Extensive experiments demon?strate that PAT significantly outperforms exist?ing methods. Furthermore, PAT delivers multi?granular interpretability by tracing post-level mental-state trajectories and highlighting key symptoms, thereby offering reliable diagnostic evidence for clinical practice. 

---
# Toward System One Clinical Intelligence: Parameter-Efficient Fine-Tuning and Counterfactual Explainability of Decision Foundation Models 

**Authors**: S. M. Mahbub Uddin Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2610.15029)  

**Abstract**: Automated disease classification from patient-reported symptoms requires balancing the speed of lightweight discriminative models against the contextual understanding and latency costs of autoregressive large language models (LLMs). Decision foundation models offer an alternative by mapping representations directly to candidate classes without free-form text generation. We adapt Clef-Flash, a 9-billion-parameter decision model built on a Qwen3.5 backbone, for 24-class disease classification on the Symptom2Disease benchmark. Using 4-bit quantized low-rank adaptation (QLoRA) with differential learning rates and a stratified split of 960 training and 240 test cases, the model achieves 95.42% diagnostic accuracy and a macro F1 score of 0.9551. This improves upon the reported zero-shot baselines of 61.67% for a remote API and 65.00% for a local unadapted initialization, and outperforms the ClinicalBERT baseline (90.90%). By evaluating candidate logits in a single forward pass without autoregressive decoding, inference takes under 80 milliseconds per record on a single GPU, yielding a reported 31-fold speedup over generative LLM baselines. To examine symptom-level evidence, we introduce a Leave-One-Symptom-Out (LOSO) counterfactual perturbation framework. Removing disease-defining symptom clauses in a psoriasis example reduces predicted-class confidence from 100.00% to 0.12% and shifts the prediction to impetigo with 99.59% confidence, while removing secondary systemic symptoms produces less than 0.01% change in confidence. These findings suggest that decision-based architectures can combine the linguistic capabilities of large foundation models with low-latency inference and symptom-level interpretability for clinical triage. 

---
# UltraBERT: Beyond the Limit of BERT Pre-Training 

**Authors**: Jungwoo Park, Taewhoo Lee, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2610.15028)  

**Abstract**: Recent bidirectional encoders have been modernized with innovations from causal large language models (LLMs). However, the pretraining recipe itself remains inherited from small-scale conventions, leaving encoders underoptimized, with no single model performing competitively across both classification and retrieval tasks. In this work, we investigate how far encoder performance can be pushed by revisiting core design choices in training, revealing three key findings: (1) Encoders saturate on downstream tasks beyond a few trillion tokens, indicating that additional compute past this point does not help. (2) Across model sizes, lower masking ratios benefit classification while higher ones benefit retrieval, contradicting prior work suggesting larger ratios for larger models. (3) Distillation with pruning improves classification at low masking ratios but degrades retrieval at high ones. Guided by these findings, we present UltraBERT, a single encoder that assigns separate LM heads to low and high masking ratios within a batch and restricts the distillation objective to the low ratio. To our knowledge, UltraBERT is the first encoder to achieve state of the art on both classification and retrieval tasks, reaching 91.0 on GLUE and 47.7 on BEIR. Together, these results show how far encoder performance can be pushed and provide a principled basis for compute allocation, masking ratio design, and knowledge transfer in bidirectional models. 

---
# Do Personalized LLMs Need the Whole Conversation? Matching the Full History with Only 5% of Its Text via Relevance-Filtered Context 

**Authors**: Erika Hayashi, Andrew Shin  

**Link**: [PDF](https://arxiv.org/pdf/2610.14952)  

**Abstract**: Personalized assistants accumulate long conversation histories, and passing the whole history to an LLM with every request is costly. We show that it is also unnecessary. On PersonaMem-v2 (180 questions), with GPT-5.6-terra and Claude Opus 5.5 as judges, a relevance-filtered context, built by sentence-level retrieval that ranks the history by similarity to the current request and stops at a fixed budget, matches full-history accuracy with only 5% of the text, cutting the judge's input about 20-fold. An LLM summary of the same length performs on par, but it needs an LLM to read the entire history, whereas retrieval needs only a small embedding model. While both sentence-level retrieval and LLM summary perform strongly, retrieval does so while placing the relevant evidence in the context for only 61% of the questions, and an oracle given only that evidence beats the full history by 15.6 points, making better context selection a promising path to surpassing the full history. 

---
# FidelityQ: Preserving Predictive Fidelity in Ultra-Low-Bit Language Model Quantization 

**Authors**: Yujie Chen, Weiyan Zheng, Yiwu Yao, Tianchi Hu, Bo Wang, Jiyu Guo, Wenzhan Li, Jingcai Guo, Michele Magno, Haotong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2610.14934)  

**Abstract**: Ultra-low-bit scalar quantization offers a regular arithmetic structure for efficient deployment of large language models, but preserving predictive behavior at extreme precision remains challenging: with only a few reconstruction levels, accurate weight representation must translate into consistent predictions from a small calibration set. We propose FidelityQ, a scalar quantization framework that preserves predictive fidelity at weight precisions down to 2 bits. For accurate quantization, we introduce Information-Aligned Analytic Coding (IAC), which shapes each weight row toward a centered symmetric source via randomized Hadamard mixing and fits a mirrored four-level minimum-distortion quantizer to its empirical weights. Every level set in this family decomposes exactly into two sign-valued bases with associated scales, combining source-adaptive reconstruction with lookup-free decoding compatible with scaled integer arithmetic. Our analysis further shows that residual predictive discrepancies are heavy-tailed and concentrate on tokens the teacher predicts confidently, motivating Knowledge-Intensive Calibration (KIC) for consistent quantization. KIC ranks sequences by the tail of token-wise teacher-student divergence, allocates a limited calibration budget to the highest-scoring samples, and restores predictive agreement by updating only quantization levels and scales under teacher supervision, keeping pretrained weights frozen. Experiments on the LLaMA and Qwen families demonstrate the effectiveness of FidelityQ in weight-only and joint quantization settings. On LLaMA-2-7B, FidelityQ achieves a WikiText2 perplexity of 6.62 at W2A16 and improves the prior state-of-the-art W2A4KV4 result from 8.31 to 6.96, while reducing linear-layer weight memory by 8.0x relative to FP16. 

---
# TeachGrade: Training-Free Short Answer Scoring with Teacher Evidence 

**Authors**: Jinhao Yan, Zhihong Shen, Shengjie Wang, Yiting Ren, Wentao Zhang, Yilei Wang, Peichao Lai  

**Link**: [PDF](https://arxiv.org/pdf/2610.14900)  

**Abstract**: Multi-subject, multi-step scoring requires consistent partial credit across grading conventions. We introduce TeachGrade, a training-free framework using teacher-graded examples. The Task Adapt module builds two frozen memories: Skill captures contextual teacher preferences beyond the rubric; CaseBank supplies local evidence. PreJudge provides retrieval cues; FinalJudge regrades the answer without preliminary scores. On 3,829 SAS-Bench answers across 12 tasks, official macro collaborative consistency score (CCS) rises from Native's 67.99 to 81.32 (+13.33 points), and pooled step exact match from 54.85% to 62.71%. Macro quadratic weighted kappa (QWK) reaches 84.71, exceeding Native, OpenJudge, ProofGrader, and Trojsten by 9.52, 3.50, 11.37, and 17.43 points, respectively, and leading on 8 tasks. Local Qwen3.8-27B gains 8.83 CCS and 4.08 QWK points, improving CCS on all 12 tasks. QWK gains over Native span 6.38--14.65 points on Portuguese automatic essay scoring (AES) and 2.16--2.24 on ASAP-SAS across label views, with lower mean absolute error. These results support reuse across languages, response formats, and backbones without weight updates. 

---
# Agent-Guided Evidence Acquisition for Emotion-Cause Pair Extraction 

**Authors**: Zhuangzhuang Pan, Yan Xia, Chee Seng Chan  

**Link**: [PDF](https://arxiv.org/pdf/2610.14886)  

**Abstract**: Emotion-Cause Pair Extraction (ECPE) requires identifying the causes associated with emotions expressed in a conversation. Accurate extraction can depend on evidence that is not initially available, making it important to decide when further information is worth acquiring. We study agent-guided evidence acquisition for ECPE, where a student agent selectively accesses expert judgments for borderline relations on both sides of its validation-selected F1 decision boundary and revises only queried predictions. Through controlled comparisons under different information-access conditions, we examine how delegation granularity, query selection, and response integration affect prediction quality. At a nominal 10% query rate, selective fusion improves mean pair F1 by 2.64-3.88 points over the student in our main experiments, reaching 65.14 on ECF and 49.74 on MEC4 across model families. Our analysis suggests that greater access does not always translate into better cause identification. These findings motivate acquisition strategies that consider both the relevance of newly available evidence and its effect on prediction revision, providing an empirical basis for more effective and efficient evidence acquisition in ECPE. 

---
# Social Bias in Chinese Large Language Models under Social Media Expressions 

**Authors**: Mo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.14879)  

**Abstract**: Existing Chinese social-bias benchmarks for large language models largely rely on normalized or template-based expressions, leaving the influence of internet slang, emoji, and sarcasm insufficiently examined. Safety alignment may further compress conventional plausibility ratings toward the lower end of the scale. This study constructs 73 paired probes across gender, region, and occupation. Each probe contains a neutral expression, a social-media-style expression, and a reversed-identity control. The task is reframed from direct plausibility judgment to social consensus estimation. Social-media expressions are divided into stylistic packaging, which preserves semantic polarity, and sarcastic packaging, which reverses surface polarity. Six Chinese-capable large language models are evaluated through paired differences measuring identity bias, stylistic packaging, and sarcasm shift. All models show significant identity-bias effects; five show significantly negative stylistic-packaging effects; and sarcasm shifts display a clear cross-model gradient. These findings demonstrate that expression form is not a transparent carrier in bias evaluation and that lower scores under sarcasm should not be interpreted directly as bias mitigation. The proposed paired design offers a measurement approach for Chinese social-media contexts while also revealing the construct boundary and prompt sensitivity of social consensus estimation. 

---
# Dependency Reasoning for Aspect Sentiment Quadruple Prediction 

**Authors**: Shima Esfandyar, Seywan Soleymanbaigi, Fardin Akhlaghian Tab, Fatemeh Daneshfar, Mourad Oussalah  

**Link**: [PDF](https://arxiv.org/pdf/2610.14873)  

**Abstract**: Aspect Sentiment Quadruple Prediction (ASQP) is fundamentally a relational reasoning problem: identifying aspect-opinion-category-sentiment quadruples requires modeling structured dependencies, not merely sequential co-occurrence. Existing LLM-based approaches incorporate syntactic information indirectly, via retrieval-based demonstration selection or multi-stage training, rather than as unified, single-pass generation-time context. We present DRIFT, which linearizes dependency parse trees and injects them directly into LLM prompts as explicit relational context, enabling the model to reason over aspect-opinion dependencies rather than infer them from token proximity alone. DRIFT further integrates retrieval-augmented in-context learning and a syntax-aware LLM-based adjudication mechanism that complements exact-match evaluation under span-boundary variation. On SemEval Rest15, Rest16, and ACOS, DRIFT achieves the strongest performance among compared generative ASQP methods, reaching 54.47% and 64.10% F1 on Rest15 and Rest16, with consistent gains across short- and medium-distance aspect-opinion configurations; adjudication further shows exact-match metrics systematically underestimate semantically correct predictions. 

---
# Training-Free BioNEL-RAG: Advancing Biomedical Entity Linking through Retrieval-Augmented Generation and Large Language Model 

**Authors**: Ziyi Chen, Hui Liu, Ziyang Ling, Ronald X.Xu, Mingzhai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.14789)  

**Abstract**: Biomedical Named Entity Linking (BioNEL) is the task of mapping mentions extracted from biomedical texts to canonical entities or concepts in the standardized knowledge base. Although existing methods have achieved remarkable performance in mapping accuracy, most rely on training with domain-specific datasets, which fundamentally limits their cross-domain adaptability. To address this limitation, we propose BioNEL-RAG, a novel training-free framework that combines retrieval-augmented generation (RAG) with large language models (LLMs). Within this framework, the RAG module focuses on retrieving candidate entities, while the large language model module plays the role of selecting the most appropriate candidate for entity normalization, which is based on the analysis of the mention's context, with the aid of its remarkable text comprehension capabilities. Experimental results demonstrate that BioNEL-RAG achieves performance comparable to existing methods without requiring training on subdomain datasets, whereas these methods necessitate fine-tuning on specific subdomains such as genes, diseases, or chemical compounds. 

---
# Who Receives Respect? A Counterfactual Benchmark for Honorific Allocation Bias in Bengali 

**Authors**: Ajwad Abrar, Md Kamrul Hasan, Hasan Mahmud  

**Link**: [PDF](https://arxiv.org/pdf/2610.14769)  

**Abstract**: Bengali has no grammatical gender, but honorificity is obligatory: every third-person reference must choose ordinary or respectful register. English has no such signal, so the system must choose a level of deference that the source never specified. We use this property to measure how large language models (LLMs) allocate respect in English--Bengali translation. Our evaluation set contains 2{,}016 sentences across 72 occupations. Each appears in seven sentence frames under four pronoun conditions. Masculine and feminine versions differ by one word, forming counterfactual pairs. We report a single Honorific Bias Score (HBS) for each model on a 0--100 scale, covering both the gender and occupation axes. A lower score means that respect is allocated more evenly. For the occupation analysis, we compute each occupation's honorific rate across the seven no-pronoun sentence frames. The standard deviation of these 72 occupation-level rates ranges from 17.2 to 35.5 percentage points across models, showing substantial variation in how honorifics are allocated across occupations. Occupation is associated with 3.2\% to 51.5\% of the variance in register choice, compared with at most 1.2\% for gender. Gender effects are much smaller overall: only two of the nine models show significant directional gender gaps, and they favor opposite genders. HBS ranges from 19.9 to 70.1 across the nine models, showing substantial variation in how evenly the models allocate respect. 

---
# The Constitution Speaks Nheengatu: An Open MT System and Reproducible Corpus Pipeline for the Amazonian Língua Geral 

**Authors**: Fabrício Carraro, Rodolfo Joel Zevallos, Rodrigo Gonçalves de Souza, Caio Henrique Faustino da Silva, John E. Ortega  

**Link**: [PDF](https://arxiv.org/pdf/2610.14760)  

**Abstract**: Nheengatu (yrl), the Amazonian Língua Geral, has 6--10 thousand speakers and is co-official in São Gabriel da Cachoeira (Brazil), yet remains poorly served by evaluated Portuguese--Nheengatu MT resources. We present the largest parallel corpus assembled for the language (14.5k Portuguese--Nheengatu pairs), anchored by the 2023 official translation of the Brazilian Federal Constitution; the first MT evaluation set built from transcribed spontaneous speech; and, to our knowledge, the first open-weight MT models dedicated to bidirectional Portuguese--Nheengatu translation.
Three findings organize the paper. First, translation into Nheengatu on held-out constitutional articles plateaus on development data regardless of training length or model size, and much of the remaining error sits in conventional legal terms that the written sources do not attest consistently; a terminology success rate over a frozen lexicon shows that glossary mechanisms at training time and lexical constraints at decoding each raise term realization, the latter at a cost in repetitive output that chrF++ barely registers. Second, a corpus expansion that adds transcribed conversation raises speech chrF++ by 24--27 points while leaving written domains unchanged, a register gap that written-only benchmarks cannot see. Third, a larger NLLB model helps translation into Portuguese but not into Nheengatu on legal or speech text. For Portuguese-to-Nheengatu translation, our final systems reach 31.7--32.8 chrF++ on article-disjoint constitutional text, 64.1--64.5 on the near-domain set (62.1 after removing items with target-side content overlap), and 45.2--47.5 on speech. 

---
# Amharic-English Cross-Language Question Answering Using Statistical Machine Translation, Named Entity Recognition, and Semantic Indexing 

**Authors**: Emebet Bekele, Fekade Getahun  

**Link**: [PDF](https://arxiv.org/pdf/2610.14756)  

**Abstract**: The predominance of English-language Web content creates an information-access barrier for speakers of low-resource languages such as Amharic. This paper presents the design, imple-mentation, and evaluation of a bidirectional Amharic-English cross-language question answering (CLQA) system that allows a user to formulate a natural-language question in either Amharic or English and retrieve an answer from documents written in the other language. The system integrates a statistical machine translation (SMT) component for query translation, a maximum-entropy named entity recognition (NER) component for Amharic text processing, semantic-based indexing, question analysis, passage retrieval, and answer extraction. A domain-diverse Amharic-English parallel corpus was used to train the SMT component, while monolingual cor-pora supported language modeling. Documents were normalized, tokenized, stemmed, indexed, and semantically enriched before retrieval. The original evaluation used precision and recall as the principal effectiveness measures. For Amharic-to-English retrieval, the system achieved 72% precision and 79% recall; for English-to-Amharic retrieval, it achieved 65% precision and 70% recall. These findings show that a modular translation-and-retrieval architecture can provide useful cross-language access even under low-resource conditions. This revised article preserves the original experimental claims while situating them in the context of subsequent multilingual transformer and Amharic QA research. 

---
# Vela: User Behavior as Supervision Sparks LLM Intelligence 

**Authors**: Yang Xiao, Yusong Sun, Haoyi Wu, Wenyang Hui, Weiqiang Hang, Luqing Wang, Zejian Wang, Zirui Tu, Jingsheng Mao, Shaobo Hu, Yang Luo, Jiale Guo, Jin Zhang, Yudan Liu, Zhaokai Luo, Ren Huan, Yao Hu, Chengyue Jiang, Wenjie Li, Chuan Mu  

**Link**: [PDF](https://arxiv.org/pdf/2610.14709)  

**Abstract**: Large language models are increasingly deployed in applications that act on users' long-term behavior histories, from personal assistants to recommendation and advertising systems. These applications demand the user dimension of LLM intelligence: inferring a person's preferences, intent, and likely future actions from weeks to years of behavior data. Yet current models are trained on static text or synthetic data, neither of which captures how user behavior evolves over time. Real-world behavior records, already available at scale on content platforms and in personal-assistant agents, offer a natural supervisory signal. We propose user behavior as supervision and introduce Vela, a framework that converts real user behavior data into trainable tasks and trajectories. From the anonymized behavior records of 100,000 users, Vela generates 100,000 tasks across 91 real-world task templates; difficulty-based filtering and trajectory acceptance criteria retain approximately 75,000 tasks with selected training trajectories. Fine-tuning Qwen3.6-35B-A3B on these trajectories improves performance over the base model by up to 11.5 points on user-related tasks and by an average of 4 points on general benchmarks. Training on more user behavior data further improves user-related performance, and reinforcement learning after fine-tuning outperforms reinforcement learning from the base model by up to 7.7 points. These gains suggest that real-world user behavior is a scalable source of supervision for both user-centered capabilities and the broader intelligence of LLMs. 

---
# Lingdan-V2: Large language models for clinically grounded reasoning in traditional Chinese medicine 

**Authors**: Rui Hua, Yu Wei, Ziri Jin, Zhuo Sun, Kai Chang, Jianan Xia, Zixin Shu, Xiaodong Li, Dongmei Jia, Fei Dong, Runshun Zhang, Jian Yu, Hao Xu, Haibin Yu, Jiansheng Li, Baoyan Liu, Wenjia Wang, Xuezhong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.14705)  

**Abstract**: Large language models have advanced medical reasoning, but clinically grounded reasoning in traditional Chinese medicine (TCM) remains challenging because prescription decisions require integrating domain knowledge, syndrome differentiation, treatment principles, herbal prescription composition, and dosage determination. We develop a TCM language model family comprising models with 4 billion, 8 billion, and 14 billion parameters. The artificial intelligence contribution is a staged clinical alignment framework integrating large scale continued pretraining on TCM and clinical records, complementary reasoning supervision, and prescription-centered reinforcement learning (RL). We further formulate herbal prescription generation as a structured RL problem with prescription-level feedback, enabling direct optimization of clinical decision outputs beyond knowledge reproduction and generic instruction following. The engineering application is TCM clinical decision support, particularly herbal prescription recommendation. We evaluate the models on a public benchmark containing 25,620 expert-verified items across 13 TCM knowledge and clinical reasoning tasks and on a private retrospective dataset of 4,348 spleen and stomach disease prescription cases. The 14 billion parameter reinforcement aligned model achieves a prescription generation F1 of 37.0 on the public benchmark, compared with 32.3 for DeepSeek-R1 with 671 billion total parameters, and a herb-level F1 of 52.8 on the clinical dataset with improved dosage agreement. Reinforcement post-training consistently improves prescription recommendation across all scales, while physician-source analysis shows that expert heterogeneity affects reward stability. These findings demonstrate a scalable pathway from domain grounding and reasoning supervision to clinically oriented prescription decision support. 

---
# PVFH-Bench: Benchmarking Premise-Induced Visual Fixation Hallucinations in Vision-Language Models 

**Authors**: Xinhao Wang, Junbo Cui, Zhenghao Liu, Yuan Yao, Xiaohua Li, Yu Gu, Ge Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.14635)  

**Abstract**: Large Vision-Language Models (LVLMs) have achieved remarkable progress across a wide range of multimodal tasks. However, object hallucination remains a persistent challenge, undermining model reliability in applications such as visual question answering, embodied agents, and human-AI decision support systems. In this paper, inspired by cognitive science studies on perceptual bias, we investigate a new failure mode termed Premise-Induced Visual Fixation Hallucination (PVFH). Specifically, PVFH refers to a premise-induced reasoning error in which models become anchored to visually confusable evidence, accept a false textual premise, and subsequently generate responses about an absent yet visually plausible object. To systematically study this phenomenon, we introduce PVFH-Bench, a benchmark that automatically constructs visually confusable but absent targets and pairs the core hallucination setting with diagnostic subsets to trace how models move from object recognition to Premise-Induced Visual Fixation Hallucination. Experimental results on 10 state-of-the-art LVLMs demonstrate that all evaluated models are highly vulnerable to Premise-Induced Visual Fixation Hallucination. Moreover, we find that explicit chain-of-thought reasoning can further exacerbate this failure mode by encouraging models to overinterpret weak visual cues as supporting evidence for hallucinated objects. 

---
# Uniform Information Density-Based Preference Optimization 

**Authors**: Yongyuan Li, Yang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.14578)  

**Abstract**: To enable large language models to generate outputs that align with human preferences, optimization algorithms based on the comparison between chosen and rejected output pairs have been developed and been proven effective. Some recent endeavors have utilized certain linguistically-derived assumptions about the preference pairs to guide the design of more precise optimization directions. However, human preferences are complex and hard to measure by simplistic criteria, and it is important to incorporate broader and more sophisticated insights about preferred vs. not-preferred language. In this study, we adopt a well-established psycholinguistic theory, Uniform Information Density, to develop a novel preference optimization algorithm, UIDPO. The basic assumption is that humans prefer to produce language in which information is uniformly distributed, because uniformity induces less cognitive processing effort. To implement this idea into the algorithm, we derive a compound metric for information uniformity by combining the global standard deviation and the local second-order difference, and integrate it to the objective of preference optimization. We compare UIDPO with SimPO, a previous state-of-the-art method, obtaining comparable and sometimes better performance on AlpacaEval2, ArenaHard, and MT-Bench. Additional linguistic analysis on complement clauses also shows that UIDPO can better reflect the tendencies of keeping information uniform in natural language than baselines. Our results show that the subtle human preference over the information rate in language is a helpful source for achieving better human alignment for LLMs. The code repository is published here: this https URL. 

---
# LongHisDoc: A Comprehensive Benchmark for Chinese Long Historical Document Understanding 

**Authors**: Zeyu Shan, Yongxin Shi, Shihang Wu, Peirong Zhang, Yuanyuan Fang, Zhaopeng Wang, Lianwen Jin  

**Link**: [PDF](https://arxiv.org/pdf/2610.14531)  

**Abstract**: Chinese historical documents are invaluable cultural treasures, but their complexity poses significant comprehension challenges for non-specialists. Despite the success of Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) in modern document tasks, their capabilities in Long (full-book) Historical Document understanding remain largely unexplored. To bridge this gap, we introduce LongHisDoc, the first benchmark tailored for long historical document understanding. LongHisDoc comprises 1,012 expert-annotated QA pairs derived from 101 PDF documents, with an average of 85.1 pages and 42,776.4 tokens per document. It spans 10 categories and features four QA types: single-page, cross-page, multi-hop, and unanswerable. Answering these questions requires integrating multimodal evidence from text, layout, and visual elements. Our evaluation reveals that the top-performing LVLM and LLM achieve only 62.9% and 69.9% accuracy, respectively, highlighting a substantial performance gap. Furthermore, we assess various modal retrievers and the end-to-end RAG performance to provide references for Retrieval-Augmented Generation (RAG) systems. We believe that LongHisDoc serves as a comprehensive benchmark to catalyze future research in full-book historical document understanding. The benchmark and evaluation code are available at this https URL and this https URL. 

---
# AphaVoice: Text-to-Speech Model for Aphasia Patient Simulation 

**Authors**: Al Varrel Putra Kusuma, Fernando Bello, Joshua Brown  

**Link**: [PDF](https://arxiv.org/pdf/2610.14479)  

**Abstract**: While language therapy is the primary treatment for post-stroke aphasia, limited clinical resources often restrict patient access to proper care. Addressing this gap requires reliable training simulations for medical professionals. However, current AI efforts focus almost exclusively on speech processing, leaving speech generation largely unexplored. This study introduces AphaVoice, a novel text-to-speech (TTS) framework explicitly engineered to synthesize highly realistic aphasic speech. Evaluated against state-of-the-art TTS systems, AphaVoice demonstrates a superior ability to capture the clinical feature of dysfluency, outperforming existing models in temporal metrics while consistently maintaining speaker identity. By leveraging latent space shifting, the model also supports multi-speaker generation without sacrificing phonetic realism. Ultimately, AphaVoice provides a scalable, high-fidelity tool for medical training and clinical simulation related to aphasia. 

---
# Scale, Drivers, and Limits of Representational Alignment Across Large Language Models 

**Authors**: Jialin Wu, Meenakshi Khosla  

**Link**: [PDF](https://arxiv.org/pdf/2610.14476)  

**Abstract**: Comparing large language models can reveal shared representational structure and systematic differences between them. However, aggregate similarity measures can obscure how agreement varies across geometric scales and which linguistic properties support it. We introduce two complementary methods to resolve where models agree and what supports that agreement. Standard representational similarity analysis (RSA) summarizes similarity across all stimulus pairs, potentially obscuring differences in alignment between nearby and distant inputs. Neighborhood-Weighted RSA (NW-RSA) addresses this limitation by weighting stimulus pairs according to their proximity in representational space, placing alignment from local neighborhoods to global geometry on a common scale. Counterfactual RSA (CF-RSA) tests which linguistic properties support alignment by comparing intact representations in one model with representations under controlled linguistic perturbations in another. Across diverse language models, we find strong agreement in coarse representational geometry but substantial divergence in local neighborhoods. Alignment is observed both across broadly sampled natural text and within linguistic equivalence classes that preserve particular aspects of meaning or structure. Counterfactual analyses reveal selective sensitivity: changes to logical scope reduce alignment substantially more than the tested changes to hierarchical syntax, event roles, or discourse structure. Alignment also decreases when sentence structure or lexical meaning is disrupted in word lists and jabberwocky sentences, suggesting that the regularities of natural language support stronger convergence. These findings refine claims of representational universality by showing that strong overall similarity need not imply agreement in how models represent finer linguistic distinctions. 

---
# EdgeLLM-Nano: Auditing Polar-Guided Key-Cache Compression and Consistency-Anchored Pruning on a 33.6 MB On-Device Language Model 

**Authors**: S M Mahbub Murshed, Hayden Bright, Rezaul Karim, Ali Shiri Sichani  

**Link**: [PDF](https://arxiv.org/pdf/2610.14475)  

**Abstract**: Deploying language models (LMs) on the edge presents unique memory, latency, and power challenges absent in high-performance computing. KV-cache compression emerges as a promising approach to address these challenges. However, the effectiveness of current KV-cache compression methods has not been thoroughly examined on edge devices. Moreover, we show that existing evaluation methods do not always involve the KV-cache under test. We propose EdgeLLM-Nano, a measurement study of KV-cache compression using a nano-scale decoder-only LM (transformer) testbed that has 8.39M parameters, i.e., 33.6 MB, to enable end-to-end examination of each backend, seed, and power sample. Our results give a negative verdict for a polar-guided key-cache quantizer (round-to-nearest (RTN) control causes an order of magnitude less damage), as well as a consistency-anchored progressive weight pruner (one-shot magnitude-based pruning performs as well as, if not better). We also introduce Long-Context Decode Fidelity (LCDF) evaluation, a token-by-token examination of teacher-forced log-losses using the KV-cache under test. Unlike perplexity that gives the same 5.461 score to all methods, LCDF distinguishes different KV-cache compression backends thanks to its paired statistics across seeds and effect sizes. Our companion Raspberry Pi 5 measurement shows all compression backends we tested to be less energy-efficient per token compared to uncompressed FP16 inference. Except for KIVI that is close to lossless in terms of LCDF, our findings indicate KV-cache compression can be improved. LCDF offers a simple and effective way of shedding light on KV-cache compression methods, whose negative impacts have been overlooked by perplexity. 

---
# Evaluating Bias in Chinese Large Language Models across Tasks and Registers: Task Operationalization Effects and Irony Classification Risks 

**Authors**: Mo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.14464)  

**Abstract**: In Chinese content governance, large language models are increasingly used to identify toxic and discriminatory expressions, yet both task operationalization and ironic expression can alter bias measurement readings. We conducted a preregistered multi-task behavioral audit of multiple Chinese large language models. In the content classification experiment, the core panel's classification error rate rose from 10.9% in the formal register to 23.5% in the yinyang (an echoic-irony register) version, a risk difference of +12.7 percentage points. (95% CI [+9.7, +15.7]). Inter-rater agreement was lower in the yinyang register (majority-vote kappa = 0.299), so the increase contains a component of label uncertainty propagation, but the direction of increase was consistent under both adjudicated and majority-vote reference standards. Explicit irony-aware prompts reduced the intervention panel's error rate by 9.6 percentage points. We operationalized the 29 shared proposition anchors into three task forms: social consensus estimation (Form A), high-stakes decision (Form B), and controlled association rating (Form C); independent manipulation checks found that all 58 A-C comparison units received unanimous agreement, whereas the low A-B convergence was confirmed by calibrated majority adjudication (87.8% supporting the minority judgment); on the irony-probe materials, the interactions of B and C relative to A were +0.577 and +1.121, respectively. Thus, when evaluating large language models for Chinese content governance, the effect of task operationalization on bias readings is an empirical question that requires data to answer, and safety evaluation should incorporate the ironic register as a test condition. 

---
# Evaluating Semantic Relation Knowledge through Bidirectional Relation Prediction from Gloss Pairs 

**Authors**: Zhihan Cao, Adam Nohejl, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2610.14442)  

**Abstract**: Semantic relations organise word senses and are fundamental to lexical semantics. In order to evaluate semantic relation knowledge of large language models (LLMs), we propose a challenging task in which an LLM predicts the relation between two glosses (definitions of senses) presented in two different orders. The predictions for the two orders are evaluated for consistency with the symmetry or asymmetry of semantic relations. Our task is designed for both monolingual and cross-lingual evaluation, allowing us to examine whether relation knowledge remains accessible when the same senses are verbalised through glosses in different languages. We construct a dataset of 3,069 sense pairs, with each sense verbalised in English, Greek, Romanian, Turkish, and Japanese. Experiments with 17 LLMs in zero-shot and five-shot settings reveal generally poor performance. In each setting, only eight LLMs achieve a macro-averaged F1 score significantly above a random baseline, and only one (zero-shot) and two (five-shot) significantly outperform a neural baseline. In particular, LLMs often fail to generate bidirectionally consistent predictions for hypernymy and to make correct predictions for holonymy in either direction. Our results suggest limitations in current LLMs' semantic relation knowledge, both in identifying relations and in respecting asymmetry. 

---
# Neither True nor False: Do LLMs Represent Truth-Value Gaps in Plural Predication? 

**Authors**: Duaa Nakshbandi, Zhiyu An, Yiping Li, Sachin Giri, Wan Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.14438)  

**Abstract**: Do Large Language Models (LLMs) genuinely grasp the truth conditions of plural descriptions, or do they force every sentence into true or false? We investigate the Homogeneity Gap, a semantic phenomenon where a plural definite (e.g., the kids laughed) and its negation (the kids didn't laugh) are both neither true nor false when only some of the kids laughed, whereas their quantified counterparts (all, not all) remain strictly bivalent. We introduce HomogeneityNLI, a diagnostic dataset of 1,000 minimal-pair items crossing determiner, polarity, and group composition across ten logical conditions. Evaluating seven open-weight models, we uncover a pervasive Bivalence Bias: models assign plural definites their homogeneous truth conditions, reading the kids as all of them and the kids didn't as none of them, yet collapse the truth-value gap into falsity. In the zero-shot setting, no model recognizes the gap for both polarities in more than 15% of scenarios, and negated definites are judged False in 84% of cases. Explicit linguistic rules, chain-of-thought, and enumeration prompts produce unstable, model-specific effects that often entrench the bias. Taken together, our findings indicate that current open-weight LLMs have acquired the truth conditions of plural predication but not its trivalent logic. 

---
# Scaling Laws for Multimodal Data Mixtures 

**Authors**: Aditi Khandelwal, Ayush K. Tarun, Yixuan Xu, Imanol Schlag, Golnoosh Farnadi, Siva Reddy, Luke Zettlemoyer, Antoine Bosselut  

**Link**: [PDF](https://arxiv.org/pdf/2610.14423)  

**Abstract**: Frontier AI systems are increasingly natively multimodal, jointly pretrained on multiple modalities, such as text, vision, and audio. However, the scientific understanding of the multimodal data mixtures used to pretrain these systems --- how much of each modality to mix, and at what potential interference cost to the others --- is largely absent or gatekept. In this work, we derive data mixing scaling laws for three modalities: text, vision, and audio. Multimodal Large Language Models (MLLMs) increasingly adopt Mixture-of-Experts (MoE) architectures, as MoEs scale efficiently and partially limit cross-modal interference. We therefore use MoEs to conduct 268 experiments ranging from 457M to 8.3B parameters, on up to 150 billion tokens. Our work leads to multiple actionable insights into multimodal data mixing for training MLLMs. First, compute optima are modality-specific: audio modality is model-heavy, favoring scaling parameters over tokens, while text and vision require balanced scaling of model size and data. Second, repeating vision or audio data beyond 2x yields negligible benefits. Finally, cross-modal interactions are highly asymmetric: vision strongly interferes with audio, whereas audio's interference with learning text and vision is negligible. Overall, we provide a principled foundation for understanding the multimodal data mixtures needed to train frontier MLLMs. 

---
# SciCurateBench: Curating Experimental Data from Heterogeneous Scientific Literature with Verbal-Action Agents 

**Authors**: Minuk Ma, Kaiyun Guo, Jiarui Ding  

**Link**: [PDF](https://arxiv.org/pdf/2610.14393)  

**Abstract**: Scientific research increasingly depends on curating experimental data from literature. Scientific data curation requires verifying exact experimental conditions (e.g., specific assays, clinical outcomes, and data availability) across long, heterogeneous files, which is a highly time-consuming process. Existing benchmarks evaluate only broad relevance on short queries, failing to capture this complexity. We introduce SciCurateBench, a benchmark of 200 expert-annotated papers from major life-science journals that challenges systems to verify conjunctive biological experimental criteria from raw, unprocessed scientific literature. We show that plain LLMs and popular reasoning, tool-using, and feedback-based agents all struggle with this task. To study what makes the task tractable, we further propose LangPoli, an integrative system in which a policy agent generates natural language action guidelines that actor agents convert into executable tools. LangPoli showed strong performance on both Paper Classification (83.16% F1 vs. 58.22% for its base LLM) and Structured Information Retrieval, and also improves re-ranking on the public benchmarks of LitSearch, BEIR, and TREC. We further explore curriculum transfer, a method that distills generalizable strategies from solved tasks to accelerate the solving of related problems. We hope this work facilitates research in truly practical scientific AI assistance. The anonymized pre-release can be found from https://anonymous.4open. science/r/scicuratebench-D885/. 

---
# Multimodal Co-Generation With Continuous Bitstream Diffusion 

**Authors**: Georgios Batzolis, Georgios Kontos, Bruno Trentini, Pranam Chatterjee, Micha Livne, Mark Girolami, Luca Ambrogioni  

**Link**: [PDF](https://arxiv.org/pdf/2610.14349)  

**Abstract**: We show that continuous bitstream diffusion (CoBit), originally developed for language modeling, provides a unified framework for joint and conditional multimodal generation using the same training and sampling protocol. Across image-text, speech-text and protein sequence-structure modeling, models are trained separately from scratch, but each uses a single input projection, transformer backbone, bitwise prediction head and binary score-matching objective shared across its modalities. Within each setting, a single checkpoint supports joint generation and the conditional directions included in training by changing which parts of the bitstream are observed. We first validate the approach on MNIST-Sum, a controlled image-text task pairing raw binary images with equations encoded as text, achieving near-perfect conditional accuracy and high joint consistency without learned tokenizers. We then apply the same protocol at larger scale to speech-text and protein sequence-structure modeling. Trained without text pretraining, a single speech-text model performs speech synthesis, recognition, continuation, and unconditional co-generation. Jointly generated speech and text achieve 3.5% word error rate between the generated text and automatic transcriptions of the generated speech. In protein modeling, a single CoBit checkpoint supports forward folding, inverse folding, and joint sequence-structure generation. For jointly generated proteins spanning 100-500 residues, computational refolding of CoBit's sequences yields structures that agree with the generated backbones at a mean self-consistency TM-score of 0.911. Together, these results show that CoBit provides a common generative framework for substantially different modalities and forms of cross-modal dependence, without introducing a domain-specific objective or sampling procedure. 

---
# Evaluating LLMs as Agentic Workflow Executors in Real-World Customer Support Skills 

**Authors**: Md Tahmid Rahman Laskar, Gundeep Singh, Xue-Yong Fu, Karol Chang, Kevin Sanders, Shi Zong, Tania Habib, Julien Bouvier Tremblay, Shayna Gardiner, Harsh Saini, Matthias Lee, Elena Khasanova, Quinten McNamara, Shashi Bhushan TN  

**Link**: [PDF](https://arxiv.org/pdf/2610.14346)  

**Abstract**: Enterprise customer-support agents increasingly rely on Large Language Models (LLMs) to execute bounded, tool-using workflows. We study this problem through the lens of Skill LLMs: specialized executors that operate within a selected skill context and must complete a narrow, stateful workflow under policy and tool constraints. We present a benchmark construction pipeline that mines recurrent workflows from real-world transcripts, converts them into structured skill descriptions, maps them to tool requirements, simulates environment state with Model Context Protocol (MCP)-compatible mocks, and generates multi-turn tool-using dialogues. Using this benchmark, we conduct a large-scale zero-shot evaluation of 17 language models spanning closed model families and deployment-oriented open-weight models from 1.7B to 14B parameters, each judged independently by five strong LLM judges, plus a deterministic rule-based evaluator that checks tool coverage, hallucinated tool use, and unsupported claims. We find a consistent ranking in which GPT-5.4 leads by a wide margin and QWEN3.5-9B is the strongest open-weight model (below 20B), outperforming several closed baselines under every judge. We also find that judge choice is not a negligible methodological detail: pairwise judge agreement on the pass/fail verdict averages Kappa=0.524 across 10 judge pairs. These results suggest that comprehensive zero-shot evaluation of skill execution requires judge reliability as carefully as model capability. 

---
# ASSET: Adaptive Speech Segmentation via Reinforcement Learning 

**Authors**: Cihan Xiao, Yuan Tseng, Matthew Wiesner, Desh Raj, Leibny Paola Garcia Perera, Sanjeev Khudanpur, David Harwath  

**Link**: [PDF](https://arxiv.org/pdf/2610.14309)  

**Abstract**: Speech large language models (LLMs) are often built using pretrained acoustic encoders as an audio interface to an LLM decoder. However, LLMs scale poorly with input sequence length, while acoustic encoders tend to produce substantially longer sequences than typical textual inputs. This motivates learning to compress encoder outputs while preserving task-oriented information. To this end, we introduce ASSET, a reinforcement learning (RL)-based framework that learns to adaptively compress encoder outputs by first partitioning them into variable-duration segments and then pooling each segment into a continuous audio token. We use group-relative policy optimization (GRPO) to train a segmenter with task performance feedback and a token-rate penalty. In automatic speech recognition (ASR) experiments, ASSET achieves performance comparable to a transcript-assisted character-boundary oracle with approximately 25% fewer audio tokens on English LibriSpeech, while improving on a phone-boundary oracle at a similar token rate on Mandarin AISHELL. Exploratory experiments on non-ASR tasks further illustrate how learned segmentation changes when the framework is adapted to tasks beyond transcription. 

---
# The Thinking Leakage in Reinforcement Learning for LLMs 

**Authors**: Wenxuan Zhu, Shuming Liu, Bang An, Aznaur Aliev, Bing Li, Yibo Yang, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2610.14270)  

**Abstract**: Hybrid reasoning models such as the Qwen3 series switch between a THINK mode and a NO-THINK mode through a few chat-template tokens. A popular recipe trains such models with reinforcement learning in NO-THINK mode and reports improved reasoning performance. We provide evidence that much of this gain can be explained by thinking leakage: the gain comes largely from partially reactivating the reasoning abilities of the original model's THINK mode instead of learning enhanced reasoning behavior. We train Qwen3-1.7B, 4B, and 8B with GRPO in NO-THINK mode. As accuracy improves, responses become longer and more reflective, are increasingly judged by the original model to resemble its own thinking, and the model's internal activations shift toward its THINK mode. We then investigate at the level of weights, activations, and attention heads: scaling the RL update further elicits THINK-mode behavior and approaches THINK-mode accuracy but does not exceed it; in Qwen3-8B, replacing a low-dimensional THINK-related activation subspace of the RL-trained model with its original-model value removes most of the RL gain, and the reverse exchange transfers a considerable fraction of the gain to the original model; and the RL-induced shift passes through the attention heads that read the empty think block. We further identify a possible mechanism in the GRPO training signal: responses favored by the advantage-weighted update are more THINK-like, and a gradient step computed from this signal moves the model toward its THINK state. Thus, gains in NO-THINK mode can largely reflect reactivation of reasoning abilities already present in the original model. 

---
# Agentic Data Engineering for LLMs: System Design and a Controlled Empirical Study 

**Authors**: Ruomeng Ding, Qianli Shen, Zhaoyang Han, Meixin Chen, Daoyuan Chen, Yaliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.14238)  

**Abstract**: LLM data engineering requires iterative experimentation to determine how data selection, rewards, and curricula affect model learning, making it a natural setting for AI-for-AI (AI4AI) research. Can agents use experimental feedback to progressively improve strategies, and do the resulting strategies improve target-model performance on held-out data? We study how agents use experimental feedback to develop data-engineering strategies under fixed training and evaluation protocols. Our Agentic Data Engineering (ADE) framework lets agents pursue, revise, or combine research directions using cumulative findings. Strategies are selected by in-loop scores and evaluated on held-out benchmarks. Across four tasks-math and code data selection, math reward design, and math curriculum learning-selected strategies achieve absolute gains over generic baselines of up to 13.34% on AIME25 Pass@32 and 5.85% averaged across four held-out benchmarks under fixed training conditions. In-loop scores improve over successive experiments, while held-out trajectories show that later in-loop gains do not always improve generalization. Comparisons of research organization and feedback access favor combining evidence analysis with cumulative findings. Research trajectories show how earlier findings inform strategy revisions and combinations across research directions. 

---
# Modeling the Arrival Time of Information-Bearing Events in Natural Language 

**Authors**: Leyi Lao, Yulin Ou, Yang Xu, Hao An  

**Link**: [PDF](https://arxiv.org/pdf/2610.14212)  

**Abstract**: Understanding how information is distributed over time is fundamental to modeling natural language. While existing empirical studies primarily rely on post-hoc analysis of information-theoretic metrics, such as surprisal, little is known about the generation processes under the surface. We propose a novel approach for addressing this gap using the tool of neural temporal point processes (TPPs) by modeling the key information-bearing events in text sequences with learnable event intensity functions. Our model takes the history of event intervals and the contextual semantic representations from LLMs as input, and predicts the future arrival times for two types of events: new-word and high-surprisal-token. The experimental results reveal an interesting distinction in event intensities: LLMs tend to produce new words at significantly higher rates than humans, while high-surprisal tokens do not exhibit the difference. A decaying pattern of intensity over time is found in new-word events for both humans and LLMs, but not in high-surprisal-token events. Our findings revisit and extend previous studies that use simpler methods; provide supplementary evidence for theories about constrained context representation inherent in human language production; and find a new angle to view human and LLM generation differences. 

---
# Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation 

**Authors**: Hermann Kumbong, Ishan S. Khare, Iddah Mlauzi, Andrew Woen, Charles Ding, E. Kelly Buchanan, Simran Arora, Azalia Mirhoseini, Christopher Ré  

**Link**: [PDF](https://arxiv.org/pdf/2610.14162)  

**Abstract**: Diffusion language models (DLMs) offer parallel sampling for faster generation than autoregressive models, an increasingly important advantage as inference becomes a dominant computational workload. However, DLMs degrade quickly with fewer sampling steps, process the full sequence at every step, and require many passes over the dataset to reach good quality. We introduce Next-Concept Modeling (NCM), a two-stage, coarse-to-fine framework: a multiscale residual VQ-VAE encodes text into a hierarchy of discrete codes, and a transformer generates one level at a time, predicting all codes within a level in parallel. At matched parameter scales, NCM achieves better generative perplexity than the strongest DLM baseline on LM1B while running $21\times$ faster. On OpenWebText, NCM is competitive with the strongest DLM while running $10\times$ faster; on XSum, it matches the strongest DLM's ROUGE-L while being $31\times$ faster. On the reasoning task of Sudoku, NCM outperforms autoregressive and diffusion baselines across all difficulty levels, using fewer function evaluations. On the reasoning task of Sudoku, NCM outperforms autoregressive and diffusion baselines across all difficulty levels, using fewer function evaluations. 

---
# Does System One Transfer? Calibrated Typed-Decision Models, Label Budgets, and Translation for Indonesian Regional Languages 

**Authors**: Romi Fadli, Rusdah, Wahyu Ramadani Putri Utami, Zulfa Malika Amana  

**Link**: [PDF](https://arxiv.org/pdf/2610.14112)  

**Abstract**: Calibrated typed-decision ("System One") models return a label and a probability in a single forward pass at a small fraction of the cost of large language models (LLMs), yet their evaluation has been almost entirely English. We report a pre-registered evaluation of the commercial Jev model and the open Laya-multilingual model on Indonesian regional languages, using the parallel NusaX sentiment (12 languages) and NusaTranslation sentiment and emotion (7 languages) test sets. Jev is more often correct on sentences that are lexically closer to Indonesian in all three datasets, and presenting the human Indonesian translation instead raises its accuracy by up to 37.2 points, most in languages distant from Indonesian. Its expected calibration error does not grow on regional languages, but its ability to rank its own errors degrades in all three datasets. Retrained classifiers need more than 500 labels per language to match zero-shot Jev in 9 of 10 NusaX regional languages. A confidence-gated cascade that sends Jev's 10-30% least confident decisions to a local IndoBERT classifier matches or exceeds the better of the two systems, and routing to an LLM instead matches that LLM at 24-77% lower API cost. Of six further commercial LLMs (three Claude and three GPT models), five are 4-14 points more accurate than Jev, at several hundred times the API cost where it is reported, while the smallest (Claude Haiku) is less accurate. We release the pre-registration, code and all predictions. 

---
# Fin_Agent at FinEval 2026: Task-Aware Methods for Multilingual Financial NLP 

**Authors**: Mohotarema Rashid, Tanzina Hossain, Xiaoying Song, Lingzi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2610.14101)  

**Abstract**: We present Fin_Agent, our submission to the FinEval 2026 shared task on multilingual fi nancial language understanding. Our sys tems address Greek financial named-entity recognition, Japanese implicit communication recognition, and multilingual financial ques tion answering. For Greek NER, we com bine annotation-aware numeric rules with zero shot Qwen2.5-32B classification, achieving 0.9521 public F1 and 0.8884 private F1. For Japanese implicit communication, an inter pretable cue-based ordinal scorer achieves 98.0% test accuracy. For multilingual financial QA, BGE-M3retrieval with Qwen3-32B gen eration achieves 0.3023 and 0.3419 ROUGE-1 F1 on the public and private splits. Key limita tions include dataset-specific rules, handcrafted cues, and limited supervision. Implementa tion code is available this https URL Rashidnira/finagent-fineval-2026. 

---
# Is the Future Predictable? Modeling Future Predictability for Reliable Temporal Reasoning 

**Authors**: Yongyi Huang, Huaping Zhang, Wenyao Cui, Qiuchi Li, Lei Li, Baohua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.14073)  

**Abstract**: Large language models (LLMs) remain unreliable when reasoning about the future. Existing models often overlook the difference among future states, committing to specific answers even when the queried facts are inherently volatile. To capture this challenge, we introduce ChronoRegime, a future-oriented temporal reasoning task that requires models to determine whether historical temporal evidence justifies commitment to a future answer. Furthermore, we propose Confidence-Aware Temporal Reasoning (CTR), a framework that estimates future predictability from temporal dynamics and regulates answer commitment accordingly. Rather than performing unconditional forecasting, CTR commits only when the reconstructed evidence supports the target future state. Across multiple LLMs, CTR improves strict overall accuracy by 6.7 percentage points and reduces hallucination rates by 18.4 percentage points on average over the plain QA baseline. These results demonstrate that explicit predictability assessment is an effective mechanism for safer and more reliable temporal reasoning under future uncertainty. 

---
# Loose Lips: Understanding Privacy Leakage in Agentic Dialogue 

**Authors**: Soumya Sharma, Ayana Hussain, Heber Hwang Arcolezi, Ulrich Aïvodji, Spandana Gella, Adriana Romero-Soriano, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2610.14071)  

**Abstract**: As LLM agents are increasingly deployed in high-stakes enterprise settings, understanding how they handle sensitive information in agentic dialogue has become critical. Privacy leakage in such settings requires reasoning about whether information flows respect the contextual norms under which they originated. We extend CIMemories to CIMEMORIES DIALOGUE, a benchmark of 4,410 two-agent dialogue instances grounded in Contextual Integrity theory. Using this benchmark, we systematically evaluate 7 open-weight models from the Qwen3 and Llama 3.1 families across conversation lengths of 1 and 6 turns under three prompting conditions. We find that task capability and privacy leakage are fundamentally coupled: more capable models are systematically leakier, with no model in our evaluation able to decouple the two. Thinking models consistently leak more than their non-thinking counterparts, a pattern we attribute to the extended reasoning surface created by chain-of-thought. Leakage compounds with conversation length, rising up to 3x from 1 to 6 turns. Our results suggest that the coupling between task capability and privacy leakage is not an artifact of misconfiguration but a structural feature of current agentic systems. 

---
# PsyBer-Agent: A Psychology-Driven User Agent for Recommender Systems 

**Authors**: Chong Zhang, Qijie Shen, Haichao Zhang, Can Wang, Xiaobo Jin, Jia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.13989)  

**Abstract**: Evaluating recommender systems before deployment requires simulators that remain realistic over long, closed-loop interaction cycles. Current LLM-based agents generally function as surface imitators that generate realistic short-term responses but fail to capture the underlying user dynamics such as fatigue, affect, and shifting preferences. We propose PsyBer-Agent, a psychology-inspired simulator that models these dynamics through its Psy-Engine. This engine represents each user with three evolving, model-defined latent states. To calibrate simulated behavior against real interaction logs, we align log-conditioned reference trajectories with simulator-generated latent-state trajectories via Unbalanced Gromov--Wasserstein optimal transport. This approach provides structured supervision for a lightweight policy without the computational burden of fine-tuning the backbone LLM. We also introduce WebSim and the PsyBer Benchmark, which encompass multimodal scenarios and seven recommender backbones. Our framework evaluates behavioral fidelity, latent-state consistency, robustness, and responses to social-cue perturbations. On the tested horizons, PsyBer-Agent shows lower behavioral discrepancy than the prompt-only simulators while maintaining stable performance under large-scale concurrent simulation for up to 5M queued simulation tasks under bounded concurrency. Code is available in the supplementary material and at: PsyBer-Agent: this https URL, WebSim Platform: this https URL. 

---
# ChunkAct: Action Chunking for Agentic Robotics 

**Authors**: Bingxin Xu, Wenrui Bao, Xinxin Liu, Jiawei Zhou, Yuzhang Shang, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2610.18874)  

**Abstract**: Agentic robotics places a large language model (LLM) agent in the robot control loop, where it composes perception, motion, and vision-language-action (VLA) tools under the guidance of a memory and thereby generalizes to new tasks without retraining. This generality comes at a steep price in efficiency. Every primitive requires a round of LLM reasoning over the full interaction history, so the cost of an episode grows with the number of primitives. We find this reasoning largely redundant: once a sub-goal is chosen, its primitives follow a routine already prescribed by memory. We present ChunkAct, which introduces action chunking to agentic robotics. Online, a single call launches a chunk of chained primitives, which executes autonomously and returns control only upon failure. Offline, ChunkAct compiles the memory into these chunks, so execution is memory-free at test time and the planner is reserved for decisions that require reasoning. LLM computation therefore scales with the number of decisions rather than the number of primitives. On LIBERO-Pro, ChunkAct improves task success by 5.0 points over the state of the art in agentic robotics while requiring 2.9x fewer LLM calls, 4.4x fewer input tokens, and 2.3x fewer output tokens. 

---
# The Last Remote Sensing Model Built by Humans? Casting Any Earth Observation Task as Recursive Self-Improvement 

**Authors**: Sijie Zhao, Zeqi Zhou, Heli Qi, Xuanchao Fu, Zhuo Zheng, Junshi Xia, Junjue Wang, Weihao Xuan, Naoto Yokoya  

**Link**: [PDF](https://arxiv.org/pdf/2610.18845)  

**Abstract**: Earth observation (EO) models are still built by hand. For each new task, experts design a training recipe and tune it against a benchmark. We show that this loop can be handed to an agent. Any verifiable task can be cast as a harness and a verifier, where the harness defines what the agent may use and the verifier defines what counts as success. Every EO task can be made verifiable, through labeled benchmarks, independent sensors, physical models or the repeated observation of the same place. Given only these two components, a general-purpose coding agent builds the EO model and improves it recursively on its own outputs. Settings long treated as separate research problems, from learning with a handful of labels to adapting across domains, become choices of harness and verifier. On four EO benchmarks spanning optical, SAR and multispectral time-series imagery, agent-built models surpass the state of the art and keep most of that accuracy from about 1% of the training labels. This establishes a new paradigm for EO, in which people design harnesses and verifiers and agents build the models, so EO models improve as agents do. 

---
# OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and Understanding 

**Authors**: Xuyang Liu, Shihao Wang, Xinyu Wei, Minghan Li, Zeqing Wang, Zuhao Yang, Zichen Wen, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.18829)  

**Abstract**: With the rapid development of multimodal large language models (MLLMs), evaluating their video understanding capabilities has become increasingly important. Existing video understanding benchmarks cover only part of the evaluation dimensions, limiting systematic diagnosis of omni-modal video understanding. We introduce OmniCompass, a holistic benchmark for omni-modal video grounding and understanding. OmniCompass defines two evaluation tracks: OmniCompass-VL assesses vision-language models on tasks requiring visual evidence, while OmniCompass-AV assesses omni-modal models on tasks requiring joint audio-visual evidence. Each track includes three complementary tasks: temporal grounding (TG), spatial grounding (SG), and question answering (QA), assessing when a referenced target appears, where it is, and what happens. QA covers five perception and three reasoning categories in OmniCompass-VL, and three perception and two reasoning categories in OmniCompass-AV. OmniCompass-VL contains 820 videos and 2,460 instances across domains and three duration types: short, medium, and long. A narrative- and audio-rich subset forms OmniCompass-AV with 240 videos and 720 instances. Each instance is evaluated under two paired settings: text-only and text+image, specifying the same target through text and visual-language guidance. All instances are manually annotated with multi-round quality verification. Evaluations of 50 MLLMs show that open-source models still lag behind proprietary counterparts, particularly in audio-visual understanding. Precise temporal and spatial grounding remains challenging even for leading models, and their predictions still exhibit inconsistencies across the paired target specifications. 

---
# Read the Room Before Your Model Generates 

**Authors**: Muyao Kong, Zhihao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.18820)  

**Abstract**: Large language models (LLMs) generally rely on sampling temperature to regulate the amount of randomness in their outputs. Mainstream methods for selecting temperature typically involve fixed, heuristic global settings and context or token-level calibration, rather than a geometrically interpretable representation of the communicative scenario. In this work, we introduce a simple and interpretable temperature selection framework, the \textbf{Scenario-Actor}, which adapts decoding to the inferred situational characteristics. Scenario-Actor extracts scenario-relevant structures from Transformer hidden states and determines the temperature from scenario embedding, a structured geometric representation of the communicative setting. Empirical results show that on Llama-3.1-8B-Instruct, Scenario-Actor achieves strong goal-completion performance in interactive social scenarios, significant gains on open-ended instruction following, and competitive performance on multi-turn conversation compared to multiple baselines. 

---
# Self-Evolving Agent Harness 

**Authors**: Chuyang Wei, Haoxiang Guan, Minghang Zhu, Kexin Zhou, Jiyan He, Xiren Zhou, Huanhuan Chen, Yifei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.18813)  

**Abstract**: Harness self-evolution revises an agent's software scaffolding, including prompts, memory, workflows, and executable components, while keeping model weights fixed. We formalize a feedback-driven process in which accepted revisions persist across runs. The framework separates update targets from proposal and admission mechanisms. It distinguishes changes to interpreted state from revisions to tools and code, repository-level control logic, and plugin composition. We characterize search through evolutionary optimization and examine its reliability through continual PAC-learning. Under explicit assumptions, the analysis yields an expected generalization bound, a retention bound conditional on historical regression budgets, and a high-probability admission rule using fresh, independent validation data. We examine how persistent harness revisions affect knowledge sharing and organizational workflows, with implications for scientific reliability, physical safety, and institutional accountability. Cases from prior work illustrate model--harness co-design, distributed execution, and plugin reuse. Reliable deployment requires continual validation as task distributions and evaluators change, with independent safeguards for learning, runtime privileges, and consequential actions. We provide a comprehensive 66-page discussion of related work, theory, algorithms, evaluation, applications, and the broader impact of Harness RSI, together with continuously updated code and supplementary materials, at this https URL. 

---
# SkillScaffold: Scaffolding Agentic Reinforcement Learning with Policy-Adaptive Skill Credit Assignment and Selective Injection 

**Authors**: Wenlin Zhong, Hao Wang, Kun Yuan, Fan Yang, Jinluan Yang, Dian Jin, Xu Tan, Yiquan Wu, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2610.18807)  

**Abstract**: Agentic reinforcement learning (RL) enables large language models (LLMs) to learn from multi-turn interactions with verifiable outcomes. Textual skills can improve exploration by distilling prior interactions into reusable guidance. However, existing skill-based RL approaches often either assume skill injection is beneficial or rely on indirect feedback or rollout-intensive evaluation to assess its utility. Our pilot studies show that rollout-based utility estimation for skill credit assignment is costly and unstable under limited sampling, while skill benefits vary across candidates and depend on the current policy's competence on each query. To address these challenges, we introduce SkillScaffold, a policy-adaptive training scaffold that combines low-cost skill-utility estimation with selective skill injection. By reusing outcome-verified training trajectories, SkillScaffold generates candidate skills and estimates their utility through outcome-aware likelihood comparisons, without candidate-specific environment rollouts. The resulting utility guides whether and which skill to inject, retaining skill-free execution when no candidate is estimated to help. Joint training enables the Generator to produce skills tailored to the evolving Executor. Scaffolded RL encourages skill internalization for skill-free inference. Experiments on ALFWorld, WebShop, ScienceWorld, and $\tau^2$-Bench demonstrate gains over strong agentic-RL and skill-based RL baselines across both short- and long-horizon tasks. Ablations and further analyses support the importance of utility-based selection and abstention. Our code is available at this https URL. 

---
# SEAR: Self-Evolving Agentic Robotics 

**Authors**: Xinyi Wang, Bangzheng Li, Hanyu Wang, Lin Zhang, Andrea Yaoyun Cui, Zhenggang Tang, Shyamal Buch, Dejia Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.18792)  

**Abstract**: Frontier language-model agents can now improve robots on their own: they write control code, test it, and keep what works as skills and memories for the next task. We introduce SEAR (Self-Evolving Agentic Robotics), a framework and benchmark built around two questions. Is the agent really learning, or just trying more often? And does what it learns still help on new tasks, in a new simulator, and on a real robot? The framework runs an agent in three modes: it writes code that becomes the policy (code-as-policy), acts live in the control loop (agent-as-policy), or works as a researcher that fine-tunes a vision-language-action model (auto-research). The benchmark adds controls that separate retained experience from repeated attempts, familiar tasks, privileged simulator state, larger budgets, and luck. Across seven frontier language models, 330 simulated and five physical tasks, and over four years of agent runtime, we highlight three findings. First, the learning curve of a robotic agent depends both on the overall time budget and the task difficulty relative to model capability. Harder tasks make longer sessions worthwhile; on easier tasks, re-rolling short attempts does just as well. Second, control code and shared routines transfer across tasks and simulators, while task-specific solutions do not. On a real robot, we observe a performance gap between agent-designed practice and novel human-designed scenes. Third, different models evolve more effectively in different ways. Some do best acting live, others writing code first. And a text-only model keeps pace with a comparable multimodal one: perception helps with error assessment, but agentic reasoning and tool use can matter more. We release the benchmark and codebase to support future research. 

---
# Pack Light, CarryOn: Building Object-Oriented Context for Long-Horizon Agents 

**Authors**: Yuchen Lu, Julian McAuley, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.18784)  

**Abstract**: Long-horizon agents accumulate progress in persistent task state, yet repeatedly process interaction histories describing how that progress was achieved. This raises a natural question: can long-horizon context instead be organized around reusable progress rather than the history that produced it? In this paper, we introduce CarryOn, an object-oriented context runtime that organizes continuation around reusable task objects. Rather than treating compressed histories or reconstructed accounts of progress as the primary basis for continuation, CarryOn builds a context view from continuation carriers, complementary execution context, and a live tail of recent interactions. A core challenge is keeping this reusable basis valid after its surrounding history is retired: how can CarryOn update the basis without reconstructing the full representation at every fold? CarryOn maintains carrier records across folds, revising affected records against bound evidence and applicable environment checks while carrying forward unaffected records and jointly updating execution context. Experiments on OSWorld 2.0 and BrowseComp-Plus show improved gains of 5.8% and 7.0% while reducing total token usage by 51.9% and 22.9%, respectively. Project page: this https URL 

---
# BLADE: Auditing Information Use for Failure Attribution in LLM Multi-Agent Systems 

**Authors**: Xiaoguang Guo, Yaohao Long, Chuxu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.18780)  

**Abstract**: Large language model multi-agent systems extend individual agents' capabilities through collaboration across specialized roles, tools, and workflows. Yet collaboration makes failures harder to diagnose: one agent may misinterpret a correct result while others carry the mistake forward. Failure attribution therefore seeks to identify the responsible agent and decisive error step. Recent methods trace dependencies, but knowing where information came from is not the same as knowing where it went wrong. We introduce BLADE (Blame Localization via Audited Dependency Evidence), a training-free framework that moves beyond tracing information to auditing its use. Using Jev for structured probabilistic judgments, BLADE uses dependency probabilities to select relevant source-decision handoffs for auditing. At these handoffs, it examines whether receiving decisions use their sources faithfully and within task requirements, distinguishing error introduction, propagation, and repair. These findings inform a global diagnosis that considers all execution steps. Without reference answers, BLADE achieves the best or tied-best step accuracy among eight strategies under a shared Jev decision interface on 1,228 failed runs across five Who&When and Who&When Pro subsets, outperforming All-at-once by 2.2-11.1 percentage points. Making responsibility evidence explicit offers a path toward more interpretable and reliable agentic systems. 

---
# PreciseCoder: Faithful and Generalizable Debugging Beyond Unit-Test Rewards 

**Authors**: Yunkai Zhan, Miaosen Chai, Shangshang Wang, Jike Zhong, Yuqing Yang, Deqing Fu, Serina Chang, Wang Bill Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.18755)  

**Abstract**: Reinforcement learning (RL) from unit tests is the common recipe for post-training code models, but a unit test only asks whether the final program passes. We analyze what this unit-test-only reward teaches a model during debugging and find that, while it raises pass rates, it also teaches the model to edit more and to hallucinate bugs in code that is already correct. We introduce PreciseCoder, a training algorithm that augments the unit-test reward with a signal for semantically verified, localized edits, which correlates with smaller edits and higher pass rates. On 4B, 9B, and 27B models, PreciseCoder reduces the average number of edit lines from 7-15to 3 at a pass rate on par with unit-test-only training. Trained only on single-line bugs, PreciseCoder can generalize to repository-level program repair, where it resolves more issues with smaller patches. Its precision also transfers from weaker to stronger models: given the reasoning trace of a 9B PreciseCoder model, frontier models such as GPT-5.6 Sol and Qwen3.8-Max gain 4.6-12.4 points in edit precision with no additional training. Linguistic analysis of the thinking traces shows that our reward encourages the model to reason about individual lines, name the buggy ones, and form consistent bug hypotheses. 

---
# On the Monitorability of Natural Language Autoencoders: Identification vs Discrimination 

**Authors**: Kevin Yandoka Denamganaï  

**Link**: [PDF](https://arxiv.org/pdf/2610.18727)  

**Abstract**: Natural Language Autoencoders (NLAs) are proposed as oversight tools for language models: an Activation Verbaliser (AV) writes an English explanation of an activation vector, and an Activation Reconstructor (AR) maps the explanation back to a vector. Their value for oversight depends on a human monitor being able to tell a true account of an activation from a false one, given the Av's explanation. We formalise NLA training and NLA oversight deployment as two different referential games from Emergent Communication: training is a generative referential game, while oversight is a discriminative, multi-modal, partially-observable referential game played over a many-to-many relation between the monitored language model's activations and the accounts that are true of them. The gap between the two games predicts that AV's explanations will identify which activation they describe without discriminating its true account from closely matched false ones, since training never asks explanations to rule anything out. Using the NLA's own AR as a maximally-sympathetic surrogate monitor model, we experimentally confirm this theoretical insight on one released checkpoint: true accounts beat accounts of other activations in all 320 comparisons, but beat fluent rivals differing in a single claim only 60.2% of the time (this http URL. 6.7%, chance at 50%, $n = 35$). A meaning-preserving rewrite by an uninvolved model costs only 2.1% of the distance to another activation's account, so the failure is not due to the NLA pair having developed private conventions in the AV's explanations. Our rival accounts test only transcript-level truths, which the monitored language model's conversation settles, so these results bound the easiest case and leave claims about the monitored model's own processing untested. We state conjectures that would make both results consequences of the framework. 

---
# Bayesian Test-Time Inference of Task-Aligned Similarity from Weak Interactive Feedback 

**Authors**: Weida Liang, Kenji Kawaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2610.18656)  

**Abstract**: Language model systems repeatedly make test-time decisions by comparing similarity: they retrieve exemplars, rank candidates, and revise responses. Yet similarity is not unique: lexical, semantic, structural, and stylistic resemblance can point to different candidates, and the task-aligned notion can change across instances. We introduce BASIS (Bayesian Axis Selection via Interactive Signals), a training-free Bayesian method that uses weak test-time feedback to infer which criterion in a supplied, task-specific library governs the current decision. It maintains a joint posterior over candidate targets and criteria, updates it from scalar scores, pairwise preferences, or verifier outputs, and selects probes by expected information gain. In a finite noisy model, we characterize when the target-criterion pair is recoverable and show why fixed-criterion and fixed-mixture rules can fail under criterion shift. Across concept recovery, reasoning exemplar selection, and response refinement, BASIS improves six-probe concept-recovery Success@1 from 56.0% to 68.1%, reasoning accuracy from 72.3% to 73.1%, and refinement win rate from 61.4% to 63.8% over tuned static mixtures. It also exceeds compute-matched best-of-N selection on reasoning (72.1%) and refinement (61.2%). Gains are largest under criterion mismatch and with larger candidate pools, and matched controls indicate that adaptive criterion weighting explains much of the benefit, with joint target-criterion inference adding further value. BASIS selects only among supplied criteria and does not discover new ones. 

---
# Does Pretraining Teach Chemical Language Models Chemistry? 

**Authors**: Leif Sieben, Yoel Zimmermann  

**Link**: [PDF](https://arxiv.org/pdf/2610.18651)  

**Abstract**: Chemical language models (CLMs) are pretrained on large corpora of molecules with the aim of learning representations that transfer to diverse downstream tasks. Here, we systematically study the effects of pretraining using a series of such CLMs across 64 molecular property prediction datasets. We find that the downstream gains of pretraining in CLMs do not scale and performance saturates below gradient-boosting baselines. Across the regimes evaluated, only supervised pretraining on traditional physico-chemical descriptors consistently improves downstream performance for the CLMs tested whereas some forms of pretraining result in models worse than a randomly initialized encoder. While models can resolve molecular structure well, they do so no better than a randomly initialized model, suggesting that, at best, CLMs learn descriptor-like properties in addition to low-level token statistics during pretraining, but surprisingly little chemistry beyond what classical molecular features already capture. 

---
# To Align or Not To Align: Check Your COMPASS Before You Train 

**Authors**: Fabian Gröger, Petar Damjanović, Maria Brbić  

**Link**: [PDF](https://arxiv.org/pdf/2610.18638)  

**Abstract**: Combining pretrained models is a common strategy for building AI systems, whether by aligning a vision encoder with a language model or merging fine-tuned checkpoints. However, finding compatible pairs typically requires costly empirical search, as compatibility is known only after alignment or merging. We propose COMPASS (COMPatibility ASsessment Score), a theoretically motivated diagnostic that estimates whether a model pair will be compatible prior to alignment or merging. Building on the observation that representational convergence is most robust at the level of local neighborhoods, COMPASS decomposes model compatibility into two cheap-to-compute axes: a pair-specific ease, defined as the $k$-NN overlap between the two encoders, and a paradigm-level ceiling, capturing the maximum overlap their training paradigms allow. Together, these two axes provide a ranking of candidate model pairs and prescribe how to act on each pair: (i) prioritize pairs that already agree and whose paradigms are compatible, (ii) invest additional compute on pairs with compatible paradigms but low initial agreement, (iii) switch the pretraining paradigm when it cannot support sufficient agreement, or (iv) deprioritize pairs that score poorly on both axes. We apply COMPASS to $72$ cross-modal alignment experiments and $392$ task-arithmetic model merges and show that compatibility estimates by COMPASS strongly correlate with downstream success, achieving average Pearson correlation of $0.91$ on cross-modal alignment tasks and $0.70$ on model merging. 

---
# Does self-distillation improve reasoning? 

**Authors**: Jason Ken Adhinarta, Raina Wu, Ashutosh Sharma, Avinash Balakrishnan, Rameswar Panda, Polina Golland, Yoon Kim, Zhang-Wei Hong  

**Link**: [PDF](https://arxiv.org/pdf/2610.18613)  

**Abstract**: Self-distillation is an emerging approach that trains a model to imitate its own predictions when conditioned on privileged information, such as correct answers or environmental feedback. Recent methods have reported better sample efficiency and generalization than reinforcement learning (RL), but it remains unclear whether these gains extend beyond in-distribution evaluations and whether they stem from privileged distillation. In this paper, we revisit two representative methods, Self-Distillation Policy Optimization (SDPO) and On-Policy Self-Distillation (OPSD), across code, science, tool use, and math reasoning tasks. We find that their reported gains do not consistently generalize to unseen problems and are largely explained by overfitting or transferring existing reasoning behavior, rather than effectively using privileged information. Moreover, self-distillation gradients show weak alignment with policy gradients that directly optimize expected reward, questioning whether they provide an effective direction for policy improvement. Our experiments across Qwen3 models (1.7B-14B) and OLMo-3-7B suggest that effectively leveraging privileged information to improve reasoning remains an open challenge. Our code is available at this https URL 

---
# Beyond Answer Voting: Fine-Grained Correction of LLM Reasoning through Graph Alignment across Sampled Traces 

**Authors**: Haotian Liu, Yingjian Chen, Xiaohang Xu, Lang Huang, Yusuke Iwasawa, Yutaka Matsuo, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.18606)  

**Abstract**: Large language models increasingly tackle tasks that require multi-step reasoning, where errors in intermediate claims can affect the final answer. Self-consistency (SC) improves reasoning by voting over sampled final answers, but its vote leaves intermediate disagreements unresolved. To address this problem, we introduce DDRG (Disagreement-Driven Verification Guided by Reasoning Graphs), a framework for selective correction from a frozen bank of reasoning traces. Our insight is that graph alignment can expose claim-level disagreements and provide correction evidence without expanding the sampled trace bank. DDRG builds source-grounded claim graphs and sends localized conflicts to an answer-blind verifier. A verifier-supported refutation that reaches the majority answer's supporting path triggers one focused repair; otherwise, DDRG returns the exact SC vote. Experimental results across five benchmarks show that native zero-shot DDRG improves macro-average accuracy over direct answering by 8.65 and 15.83 percentage points for Qwen2.5-7B and Llama3.1-8B. It also exceeds CoT, the evaluated GoT and GoV question-answering adaptations, and SC on this macro metric. DDRG links local verification to selective correction without enlarging the initial candidate bank. 

---
# Rethinking Projector Training in Multimodal LLMs 

**Authors**: Fabian Gröger, Naël Ouerghemi, Shuo Wen, Maria Brbić  

**Link**: [PDF](https://arxiv.org/pdf/2610.18598)  

**Abstract**: Multimodal LLMs are typically trained by first aligning a vision encoder with a language model via projector training on large image-caption datasets, followed by joint training of the full multimodal model. Despite its widespread use, the role of the projector training stage remains poorly understood. We find that (i) using only $10$ to $20\%$ of the data for training the projector already recovers most downstream performance gains, and (ii) projectors trained on different datasets can be linearly interpolated while largely preserving performance. These results suggest that projector training mainly performs a coarse alignment, which many different projectors achieve equally well. Motivated by these findings, we propose PORTAL, a training-free projector initialization that requires neither paired image-caption data nor gradient-based optimization, but only per-modality summary statistics. PORTAL computes an optimal-transport-based map between Gaussian approximations of the vision and language feature distributions on a shared principal subspace, so that the projector training stage can be skipped entirely. Without any paired data, PORTAL matches the standard pipeline trained on 558k image-caption pairs across six LLM backbones, two vision encoders, and 16 benchmarks. It outperforms that pipeline in $3$ of $7$ (vision encoder, LLM) settings and reaches an average $98\%$ of its score across the other four. 

---
# Judges Follow the Reference: The Derivation Gap in Verifying Tool-Call Arguments 

**Authors**: Linhai Ma, Yan Wang, Lingfei Qian, Yupeng Cao, Victor Gutierrez Basulto  

**Link**: [PDF](https://arxiv.org/pdf/2610.18497)  

**Abstract**: Tool-call judges evaluate argument values proposed by LLM agents. We find that they accept a correct value less often when their input holds no reference for it, that is, no text that says what the value should be. This happens when the value is a default value from the function schema that the user did not mention, when it has to be inferred, or when the text it comes from was cut from the judge input. We call this drop in acceptance the derivation gap. It appears across all seven judges we evaluate, including System One models CLM and Jev. The gap is not only a truncation artifact: correct default values have the largest gap on a benchmark with no truncation. A controlled source-relocation test shows that judges follow the reference in their input: they favor a visible incorrect value over a correct value whose source is hidden. We introduce DeRef (Derived Reference): a separate model reads the full history without seeing the candidate and writes a reference for the judge. Across three benchmarks, DeRef improves overall argument AUROC for all judges tested. On the controlled test, it raises AUROC from chance or below to between 98.40 and 99.59. 

---
# How Safe Is ChatGPT Health Advice? Evidence from Real User Conversations in Four Global South Countries 

**Authors**: Shreyasi Roy Chowdhury, Kiran Garimella  

**Link**: [PDF](https://arxiv.org/pdf/2610.18465)  

**Abstract**: People increasingly bring their health questions to ChatGPT, yet safety evaluations of LLM health advice rest almost entirely on clinician-authored vignettes, synthetic prompts, or simulated user profiles. The gap is widest for the Global South, where concerns about demographic bias and unsafe recommendations are well documented but have never been tested on real usage. We present, to our knowledge, the first demographically grounded safety evaluation of real-world health conversations with ChatGPT. From a consent-based data-donation corpus of 202,590 conversations from 1,252 users in India, Nigeria, Brazil, and Pakistan (December 2022 to February 2026), each linked to self-reported age and gender, we identify 15,555 health conversations, classify their health intent using an LLM classifier adapted from an established taxonomy, and score response safety under both a context-blind and a user-context-aware rubric. Roughly one in five health conversations receives a high-risk rating. Risk concentrates in symptom and condition queries and recurs through identifiable failure modes: unsafe medication or dosage guidance, harmful self-diagnosis, missing safety warnings, encouraging over-reliance on the model, and failure to refer users to professionals. Supplying the user's own accumulated context raises the high-risk share by 13-17 percentage points, and about one in four conversations rated safe in isolation turns out to be unsafe for the specific person asking. Safety evaluations built on synthetic cases are likely to miss these harms. 

---
# Readout Scaling Shapes Sparse Feature Recovery 

**Authors**: Brain Xiao, Sarah Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2610.18462)  

**Abstract**: Readout scaling changes what a sparse autoencoder (SAE) learns, but the benefit of removing it depends on the data. We use readout-canonicalized decision-space (RC-DS) coordinates to remove singular-value gains while preserving retained decision information. At a fixed readout condition number, changing the generating distribution reverses which training coordinates recover known factors best. In our controlled coordinate-matching study, recovery is highest when training restores the space in which the factors were generated. When factors are generated before readout scaling, RC-DS increases mean sparse-factor recovery from 0.506 to 0.913 at condition number 100, compared with centered-logit training under the same budget. An exact wrapper reproduces RC-DS training to numerical precision, isolating training geometry rather than additional information. In neural policies, RC-DS improves mean logit reconstruction under the tested budgets. On three fixed language-model checkpoints, RC-DS has higher mean gain-free reconstruction in all 18 tested task-sparsity settings, while centered-logit SAEs reconstruct output logits better in 17. Small-label-head controls show that longer training reduces the reconstruction gap from 0.0198 to 0.0016. Together, these results show how readout scaling, the data distribution, and training shape sparse learning. RC-DS provides an information-preserving method for studying and controlling the readout's contribution. Separate label-detection and steering tests distinguish these reconstruction effects from task-specific feature usefulness. 

---
# Teacher-Guided Tool Learning for Small Language Models via Bidirectional KL Constraints 

**Authors**: Haoxuan Gao, Qin-Wen Luo, Xiang Chen, Yu Tian, Zequn Sun, Xiaofeng Cao, Jiangchao Yao, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.18459)  

**Abstract**: External tools can extend the capabilities of small language models (SLMs), enabling their deployment in resource constrained environments. However, their limited capabilities make it difficult for SLMs to discover effective tool use behaviors through reinforcement learning alone, while direct teacher supervision suffers from teacher student behavioral distribution mismatch and may propagate erroneous preferences. To address these challenges, we propose Selective Teacher guided Policy Optimization (SeTPO), which coordinates student exploration with selective teacher guidance and persistent demonstration supervision. Specifically, SeTPO incorporates reverse KL derived token level teacher feedback into policy advantages on low reward trajectories, while auxiliary forward KL supervision preserves teacher demonstrated behaviors throughout training. Under an idealized bidirectional KL objective, we theoretically establish that the optimal policy achieves expected reward no lower than the teacher's while retaining a guaranteed fraction of the teacher's probability mass. Extensive experiments on BFCL v4 and API Bank demonstrate that SeTPO achieves superior overall performance over supervised fine tuning, reinforcement learning, and distillation baselines across all three student scales. 

---
# Not All Drift Is a Mechanism: Tracing Prompt Adaptation in LLMs 

**Authors**: Mikhail Aleksandrov, Nail Bashirov, Maksim Ivanov, Andrew Tsukanov, Timofey Karyagin, Alexander Miasnikov, Gleb Molodtsov, Valentin Samokhin, Aleksandr Beznosikov  

**Link**: [PDF](https://arxiv.org/pdf/2610.18412)  

**Abstract**: Prompt-based adaptation offers several ways to specialize a frozen language model, ranging from natural-language prompt optimization to learned continuous prompts. These methods are usually compared through downstream performance, although similar outputs need not arise from similar internal computations. We study how discrete prompt optimization, prompt tuning, and prefix tuning alter the processing of dense and mixture-of-experts language models. Our analysis shows that the location of the adaptation predicts how task information is carried to the output: optimized text primarily changes response construction, input-level prompts write their influence into content representations, and layer-wise prefixes maintain direct control through attention. In MoE models, these differences further induce distinct expert-routing patterns. At the same time, common representation-level measurements respond strongly to generic context changes and do not reliably distinguish useful adaptations from matched placebos. We argue that understanding prompt adaptation requires separating representation drift from the part of that drift that is aligned with, and causally used by, the model's task computation. This perspective provides practical guidance for selecting, evaluating, and interpreting prompt adaptation methods. 

---
# Multi-Stream Residual Computation in Diffusion Language Models 

**Authors**: Ekaterina Alimaskina, Suren Kazarian, Gleb Molodtsov, Aleksandr Beznosikov  

**Link**: [PDF](https://arxiv.org/pdf/2610.18407)  

**Abstract**: Diffusion language models reconstruct masked tokens in parallel, offering an alternative to autoregressive generation. Yet their Transformer backbones typically carry visible evidence and evolving masked predictions through a single residual stream. We equip masked diffusion models with multiple residual streams that persist through depth and adaptive routing between them, then investigate how they support these complementary roles. On the pretraining task, two streams reduce reconstruction perplexity by 5.71% on validation data and 5.13% on average across external corpora compared with standard residual connections. We observe that, without predefined roles, visible and masked positions learn distinct routing patterns. Causal interventions show that this distinction organizes read and write routing, while token-specific write strength remains important within each role. Guided by these findings, a model trained with role-conditioned routing recovers half the validation gain from adding a second stream. State interventions explain the benefit of the additional stream: it preserves early token representations for later reuse while evidence accumulates at masked positions. This mechanism recurs at a larger model scale, showing how persistent streams support context preservation alongside prediction refinement. 

---
# PRISM: Calibrating Black-Box Judges with Counterfactual Probes 

**Authors**: Kakei Yamamoto, So Kuroki  

**Link**: [PDF](https://arxiv.org/pdf/2610.18389)  

**Abstract**: Large language models are increasingly used as judges to reduce reliance on costly human evaluation, yet their scores can systematically disagree with human judgments. We ask whether responses to an explicitly imposed evaluative bias can help correct a judge's implicit scoring biases. We introduce PRISM (Probe-based Residual Inference of Scoring Misalignment), which holds the evaluated content fixed and measures score changes when attributes are emphasized or de-emphasized. A local sensitivity model motivates representing these responses as a surrogate Jacobian. Ridge regression on human labels learns a correction to the baseline score. The resulting scoring protocol is rubric-free: it uses overall scores without requesting criterion-level ratings. In the main Jev and Sol comparisons on SummEval summarization and HANNA story evaluation, PRISM and its learned-baseline variant significantly improve agreement with human rankings over uncalibrated scores while remaining competitive with rubric-based methods. A controlled SummEval comparison yields similar agreement for rubric-derived quality probes and style probes, supporting calibration without reproducing rubric criteria. The learned coefficients reveal shared and opposing correction directions across judges. 

---
# Pin It on Anyone: Certified Targeted Displacement of Multi-Bit LLM Watermarks Without the Key 

**Authors**: Paweł Batorski, Przemysław Spurek, Paul Swoboda  

**Link**: [PDF](https://arxiv.org/pdf/2610.18359)  

**Abstract**: Multi-bit watermarking embeds a message, typically an account identifier, into LLM output so that harmful text can be traced to its origin. We show an attack technique that allows to alter the watermarking message. This allows to, for example, exchange the embedded account ID to an arbitrary different one. The attacker mainly needs access to a few accounts for generating LLM output and model logits, but crucially not the secret key used to watermark. The attacker then rewrites the text, keeping its meaning, so that the watermarking message decoder recovers the watermarking message embedded by the attacker. We are, to our knowledge, the first to propose such an attack, which we call a targeted displacement. Theoretically, we present upper and lower bounds on the probability of our attack succeeding. This bound depends on the number of accounts the attacker has access to, the length of the message and some characteristics of the given text. Our attack can succeed whenever we can bound the ambiguity of the generated text well enough. Empirically, we test our attack technique against five watermarking methods and four LLMs. Attack success tracks our predicted bounds, allowing us to alter watermarking messages in almost all cases with high accuracy, given enough accounts. Our code can be found at this https URL. 

---
# Jev-Guard: Fast Routing and Auditing Agent Actions over Editable Policy Trees 

**Authors**: Wenbin Hu, Huihao Jing, Haochen Shi, Yuxuan Liu, Changxuan Fan, Sirui Zhang, Kawai Chung, Hanyu Yang, Zhenhuan Wang, Ziyi Chen, Haoran Li, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.18326)  

**Abstract**: Agent harnesses require a low-latency guardrail model together with customizable audit policies. However, deploying an auto-regressive guardrail model remains challenging: such models incur high inference latency and are often poorly aligned with customized policies. To address these problems, we present Jev-Guard, a unified model for safety classification and policy routing. We further construct a policy tree over which Jev-Guard performs routing, spanning diverse domains including content moderation, cybersecurity, privacy, and legal compliance. Specifically, Jev-Guard first navigates the policy tree to select the applicable audit policy given the caller's audit context, and then performs safety classification conditioned on the retrieved policies. On 7,033 evaluation cases, Jev-Guard achieves 93.72\% safety accuracy, a mean latency of 58ms, and 93.10\% hierarchical routing accuracy. The full routing-and-classification pipeline averages 240ms. Furthermore, we integrate Jev-Guard into the Hermes harness framework. Across agent trajectories, its separate permission gate cuts mean review time per task from 4.80s for an SFT-LLM judge to 1.33s, with no policy violations are observed for either guard. 

---
# Encrypted Skills Can Become New Security Exploits If AI Agents Know How to Decipher 

**Authors**: Yan-Lun Chen, Pin-Yu Chen, Chia-Mu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.18262)  

**Abstract**: Third-party skills can expose LLM agents to malicious instructions hidden by encoding. We introduce Ciphered Skill Injection (CSI), which embeds such instructions in skill files for agents to decode and follow. We evaluate Caesar and layered variants across eight models and 139 injection-task pairs, separating instruction recovery from attack success. In the primary setting with shortened skills, the variants achieve attack success rates of up to 47.5% and 40.3%, respectively. Successful decoding does not guarantee attack success, and layering has model-dependent effects. The tested skill scanners rarely flag injected skills. These findings highlight the risk of promoting decoded, untrusted content into actionable instructions. 

---
# Consider, Then Choose: Order-Invariant Decisions in System One Models 

**Authors**: Jianguo Huang, Yao Long Teng, Yanchen Deng, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2610.18233)  

**Abstract**: System One models answer typed questions in a single forward pass and promise the same answer for the same input. For choice questions, this promise fails: the answer depends on the order of the options, increasingly so as they grow in number, and averaging over their cyclic shifts reduces this dependence without removing it. Scoring each option alone, which the order cannot affect, shows that the order changes the answer mostly when the model is unsure among its top few options, even though the listwise answer looks confident. We therefore propose $\textbf{Runoff}$, which considers, then chooses: it scores each option alone to keep a few candidates, then compares them in listings where each candidate takes every position. It needs no training and can run in two forward passes, and its answer does not depend on the option order by construction. On two open models and five datasets, it changes no answer under shuffling, against 9.3% for cyclic averaging, and with 20 options it is about as accurate and three to four times faster. Wrapped around eight System One models, it removes the order dependence of the five benchmarks, while cyclic averaging changes up to 15.4%. 

---
# SWE-MISBEHAVIOR: Benchmarking and Mitigating Boundary Violations in Software Coding Agents 

**Authors**: Do Xuan Trong, Duc Anh Nguyen, Do Xuan Long, Ngoc-Hai Nguyen, Tongyao Zhu, Duy Dinh, Cayden Chik, Cuong Dang, Jiaying Wu, Xuan-Phi Nguyen, Vinh Van Nguyen, Kenji Kawaguchi, Shafiq Joty, Nancy F. Chen, Min-Yen Kan  

**Link**: [PDF](https://arxiv.org/pdf/2610.18206)  

**Abstract**: We present SWE-MISBEHAVIOR, the first taxonomy and evaluation framework for diagnosing boundary-violating misbehaviors in coding agents. It contains 7.4K human-in-the-loop synthesized samples spanning three layers of agent action space: (i) Code Generation, (ii) File Management, and (iii) System Interaction. Across these categories, SWE-MISBEHAVIOR covers 11 distinct misbehaviors with a fully automated evaluation protocol, including coding-action violations, Create/Read/Update/Delete (CRUD) file misbehaviors, system integrity and abuse violations, and task-goal misalignment. Experiments on state-of-the-art open-source coding agents (e.g., OpenHands, smolagents) reveal substantial misaligned behaviors, including dependency misinterpretation (up to 98.5% of cases), unintended modification of user inputs (50.5%), and silent compromise of environment integrity (53%). Motivated by these findings, we further propose CODE-KIT, a lightweight adaptive framework that combines misbehavior-aware guidelines with novel tools to help agents avoid such failures. CODE-KIT reduces misbehaviors by 52% for smolagents and 34% for OpenHands, while simultaneously improving task completion performance. 

---
# LoOPSD: On-Policy Self-Distillation with Looped Teachers 

**Authors**: Wei Fan, Qingyue Yang, Chao-Hong Tan, Qian Chen, Wen Wang, Xiangang Li, Kejiang Chen, Weiming Zhang, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.18162)  

**Abstract**: On-policy self-distillation improves a model using guidance from the model itself, but it still requires a teacher that provides more useful predictions than the student. Existing approaches obtain such guidance by giving the teacher additional context or instructions. We ask whether it can instead come from changing how the same model uses its own layers. To address this, we introduce LoOPSD, which searches over a model's layer execution order to construct a stronger teacher and distills its advantage into the student. Specifically, we keep the model weights unchanged while searching for layers to repeat, and use the resulting teacher's predictions on student-generated prefixes to guide learning. We then alternate search and distillation, rebuilding the teacher from the updated student in each round so that the student progressively absorbs the gains from alternative layer execution. Experiments on mathematical and logical reasoning tasks show improvements across models. On GSM8K with a 400-token thinking budget, LoOPSD raises Qwen3-1.7B accuracy from 42.30% to 79.61%; subsequent distillation rounds yield further gains beyond the first round. Together, these findings show that by searching layer execution to construct teachers, we can turn a model's additional computation into improved student capabilities, opening a new avenue for self-distillation to derive supervision from the model's own computation. 

---
# Monitoring latent reasoning in looped language models 

**Authors**: Pawel Olszowiec, Paulina Tomaszewska, Maciej Żelaszczyk, Michal Byra, Neo Christopher Chung  

**Link**: [PDF](https://arxiv.org/pdf/2610.18127)  

**Abstract**: Looped language models (LMs) reuse one block of layers several times per token generation, promising more latent computation at no additional storage cost. Despite their rising popularity, the analysis of looped LMs is still in its infancy: little is known about whether the iterative transformations actually improve their capabilities, or whether these models remain stable under prompt perturbation. We construct an inspection protocol rooted in Mechanistic Interpretability comprising five diverse lenses. We apply it to examine eight looped LMs from four families. Models differ in many aspects but the key one is whether the prompt is transmitted only within the looped state (carry loops: Ouro, LoopUS) or re-supplied at every iteration (re-read loops: Parcae, and Rec-OLMo & Huginn from the Raven family). The results of our study provide valuable insights concerning safety and efficient use of looped language models. 

---
# Chain of Thought Monitoring Can Fail Even When CoT Is Necessary 

**Authors**: Lennart Bürger, Piotr Komorowski, Moritz Weckbecker, Evgenii Kortukov, Aakriti Jain, Sebastian Lapuschkin, Wojciech Samek  

**Link**: [PDF](https://arxiv.org/pdf/2610.18105)  

**Abstract**: Chain of Thought (CoT) monitoring is used by frontier labs to detect misbehaviour of AI systems. Yet, its reliability remains poorly understood. We attribute this to the lack of a successful organizing principle that leads to generalizable conclusions about monitoring reliability in different regimes. Recent work attributes monitoring failures to the CoT not being necessary for the answer, and argues that CoT becomes monitorable when it is necessary. We show that this binary necessary/not-necessary organizing principle is too coarse by demonstrating that monitoring failures persist even if the CoT is necessary for the answer. In response, we introduce a more fine-grained organizing principle based on a hierarchy of four CoT necessity levels. Building on this hierarchy, we develop a framework for measuring CoT monitorability at the three lowest levels via intervention experiments. We demonstrate monitoring failures at all three of these levels on realistic, hard prediction tasks. On a difficult medical-diagnosis task, patient sex and a clinical attribute influence the answer in at least 19% and 24% of correct responses with a necessary CoT, respectively, yet their influence is almost never articulated. These results show that even when CoT is necessary, it is not guaranteed to be monitorable. 

---
# inLoRA-MINT: Internal Representations for Auditing Training Data in Domain-adapted LLMs 

**Authors**: Gonzalo Mancera, Julian Fierrez, Aythami Morales, Ruben Tolosana, Daniel DeAlcala, Francisco Jurado  

**Link**: [PDF](https://arxiv.org/pdf/2610.18051)  

**Abstract**: This work investigates the application of the Membership Inference Test (MINT) to large language models (LLMs) fine-tuned with LoRA, aiming to determine whether specific samples were used during training and address concerns about data privacy, leakage, and transparency. Building on our previous LoRA-MINT framework, which relies exclusively on perplexity under a strict black-box setting, we propose an approach based on intermediate model representations, including activations and parameter-dependent signals. These representations enable a MINT classifier to distinguish training from non-training samples without requiring full access to model parameters.
We evaluated the method on four LLMs (Qwen, Phi-4, DeepSeek-V3, and LLaMA 3.2-3B) fine-tuned on three datasets covering medical, mathematical, and academic reasoning tasks. Results show consistent performance across architectures and domains, achieving scores between 95% and 98%, with robustness to variations in model configurations and data distributions. The proposed approach enables efficient auditing of sensitive training data and supports data provenance and compliance with emerging privacy and AI governance regulations. Although our experiments focus on LoRA-based fine-tuning, the methodology can be extended to other LLM adaptation techniques and, more broadly, to other AI models. 

---
# Fish Audio Drama 3 Technical Report 

**Authors**: Dawei Han, Shidong Li, Shijia Liao, Jiahua Liu, Songting Liu, Qingzheng Wang, Yuxuan Wang, Xin Chen, Yifan Cheng, Tianyu Li, Gangin Park, Xuehao Yang, Chang Zeng, Haoyang Zhang, Ruoyi Zhang, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.18010)  

**Abstract**: We introduce Fish Audio Drama 3, an open-source speech generation model that turns text into directed performances, built on a 35B-parameter mixture-of-experts backbone that activates 3B parameters per token. It clones voices from multiple references, follows natural-language direction, revises a take when the next turn gives a note, voices multi-character scenes, edits existing speech by infilling, and synthesizes long-form and streaming audio. Multi-turn revision, scene dubbing, editing by infilling, and streaming text input are new in Drama 3, all in one model. Drama 3 pairs the hybrid linear-attention backbone with a new codebook multi-token-prediction audio decoder that predicts the acoustic codebooks of each frame in dedicated layers that attend over the entire sequence. The model is trained with a three-stage pre-training recipe, instruction fine-tuning, and large-batch reinforcement learning whose only reward is a single learned model of human preference. Its inference engine decodes a single stream at 271 frames per second on one H200, against 177 for Fish Audio S2, although Drama 3 has more than seven times as many parameters. The model weights are available on GitHub (this https URL) and Hugging Face (this https URL), and inference is available through SGLang-Omni (this https URL). We highly encourage readers to visit this https URL to direct their own performances. 

---
# Match the Canvas, Not Just the Answer: Predictability-Aware SFT for Diffusion Language Models 

**Authors**: Miao Li, Hanyang Jiang, Hengyu Fu, Baihe Huang, Yating Liu, Yuhang Cai, Changkun Guan, Zhongzhu Chen, Jiecheng Lu, Michael A. Boateng, Pascal Van Hentenryck  

**Link**: [PDF](https://arxiv.org/pdf/2610.18001)  

**Abstract**: Diffusion language models (DLMs) enable flexible parallel generation by resolving tokens according to the model's evolving predictability rather than a fixed left-to-right order. This flexibility, however, introduces training choices absent from standard autoregressive teacher forcing: which reference tokens are visible when a target is supervised and how much generation space is allocated. Standard supervised fine-tuning (SFT) typically makes these choices through random masking, independently of inference. Our analysis shows how this training-inference mismatch can misdirect supervision. Random masking can conceal prediction errors or penalize uncertainty that decoding would otherwise resolve, while the attention-visible input length can change which response formulations the model favors, even when the underlying solution is unchanged. Motivated by these findings, we propose Predictability-Aware Canvas Selection and Supervision (PACS), an SFT method. PACS uses the model's own predictability to select each example's attention-visible input length and supervises reference tokens immediately before their confidence-guided revelation. Across reasoning and code-generation benchmarks, PACS consistently achieves the highest mean accuracy among the compared SFT methods, with gains of up to 12.84 percentage points over vanilla SFT on individual benchmarks. Code will be released upon publication. 

---
# HaloBlock: Faster and Better Decoding for Diffusion Large Language Models 

**Authors**: Zhaohui Zheng, Chenhang He, Jinrui Zhang, Ming-Ming Cheng, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.17948)  

**Abstract**: Diffusion large language models (dLLMs) have attracted increasing research attention due to their bidirectional attention and arbitrary order decoding. However, there are two issues in dLLMs inference: the decoding difficulty is dominated by the tokens on the right side of the block, and the block-wise decoding inherently suffers from boundary contextual discontinuity. To tackle these limitations, we propose HaloBlock, a difficulty-aware adaptive halo mechanism for dLLM inference. We first leverage bidirectional cumulative entropy to characterize block decoding difficulty. Then, based on the bidirectional decoding difficulty, we adaptively allocate bidirectional halo regions on both sides of each block to mitigate boundary contextual discontinuity. The difficulty-aware design of HaloBlock smooths cross-block context propagation and lowers decoding difficulty for hard tokens, yielding better generation quality and faster convergence. Evaluated on LLaDA-8B-Instruct, Dream-7B-Instruct and LLaDA-1.5 across mathematical reasoning and code generation benchmarks, HaloBlock demonstrates consistent accuracy improvements and higher inference efficiency. Source codes are publicly available at this https URL. 

---
# Evaluating the Causal Faithfulness of SHAP as a Candidate Attribution Method for SAE Features 

**Authors**: Samuel Xie, Noura Al-Moubayed  

**Link**: [PDF](https://arxiv.org/pdf/2610.17898)  

**Abstract**: Sparse autoencoders decompose language model representations into interpretable features, but feature interpretation in this literature remains predominantly correlational. Attribution methods offer more direct estimates of feature influence, yet their faithfulness to actual causal effects has not been systematically evaluated in the SAE setting. We apply SHAP-based attribution to SAE latent features and evaluate it against direct causal intervention, using sentiment classification on SST-2 with GPT-2 Small and a 32,768-feature layer-7 SAE as a testbed. We compare SHAP to Integrated Gradients, Gradient$\times$Input, and probe-weight baselines across linear and MLP probe arms, scoring each against feature ablation and activation steering measured through the language model's own logit difference. Most faithfulness associations we measure do not survive multiple-comparison correction, and attribution methods that agree strongly with one another ($\rho = 0.42$--$0.80$) agree poorly with measured causal effects, indicating that inter-method agreement is not evidence of causal faithfulness. Two effects hold under control. First, local per-example attribution retains a directional advantage over global averaged attribution, while its apparent importance advantage is explained by feature activity rather than superior causal ranking. Second, under known-ground-truth minimal pairs, attribution recovers inserted sentiment signal well above chance (AUROC $0.792$) but fails to localize deletions (AUROC $0.520$), suggesting that these methods detect the presence of a feature's influence more reliably than its absence. We argue that importance faithfulness and directional faithfulness are separable properties requiring separate evaluation, and that attribution agreement should not substitute for causal validation. 

---
# Training LLMs for Multi-Agent Collaboration with Verifiable Information Rewards 

**Authors**: Keru Chen, Sen Lin, Yingbin Liang, Nathaniel D. Bastian, Shaofeng Zou  

**Link**: [PDF](https://arxiv.org/pdf/2610.17893)  

**Abstract**: Effective collaboration among large language model (LLM) agents in multi-agent systems requires sharing relevant evidence, requesting missing information, and incorporating teammates' contributions into decisions. Training these behaviors is challenging because final task outcomes provide limited feedback on the value of individual messages. We introduce CoSIR (Collaboration via Surprisal-based Information Rewards), a post-training approach that uses a simple, verifiable environment to teach collaboration behaviors that generalize to more complex, unseen tasks. We propose a Bayesian evaluator that scores shared information by its reduction in team surprisal, defined as the sum of the negative log probabilities assigned to the correct option given each agent's available evidence. Requests are scored by the potential reduction in the same quantity if the requested information were truthfully provided. Separate rewards encourage agents to use the received information in their private choices and calculations. We post-train Qwen3-4B, Qwen3-8B, and Gemma-3-4B-it through self-play and evaluate transfer to seven unseen natural-language collaboration tasks. CoSIR achieves the highest aggregate score, while its performance in general tasks remains competitive. Behavioral analyses show increased disclosure of private evidence and fewer judged failures to use teammates' input. 

---
# Diff Diffusion: Predicting Edit Paths with Diffusion Models for Efficient Code Editing 

**Authors**: Xiaolong Xu, Hao Jiang, Bo Jiang, Zhiwen Deng, Ningyuan Sun, Bingzhou Chen, Jin Chen, Chenxing Wei  

**Link**: [PDF](https://arxiv.org/pdf/2610.17831)  

**Abstract**: Diffusion large language models (dLLMs) are appealing for code generation because they can update many tokens in parallel, yet many are optimized for sequence synthesis rather than edit prediction. This setting is poorly matched to modern AI-assisted IDEs, where edit prediction must infer the next code change from recent edits, the current file, and cursor-local context. Such changes are usually sparse and preserve most source tokens, so regenerating a full code segment wastes computation. We propose Diff Diffusion, a diffusion-inspired LLM framework for code editing that predicts edit paths from an original snippet to its revised form. Diff Diffusion represents denoising as structured edit operations, and combines this formulation with multi-stage training and efficient parallel decoding. By focusing computation on changed regions, Diff Diffusion supports accurate low-latency edits and multi-position parallel modification. Across public edit-completion benchmarks, edit-prediction evaluations, and online validation, Diff Diffusion matches or leads autoregressive baselines while delivering 2.78 times faster inference on public edit-completion benchmarks and a 39.0% P50 model-latency reduction in production. This work provides a new modeling perspective for AI applications that require high accuracy and low latency. 

---
# Checking the Checkers: A Synthesis of Verifier Noise and Robustness in RLVR 

**Authors**: Savyasachi Gupta, Aniruddha Srinivas Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2610.17824)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become a principal recipe for improving the reasoning of large language models (LLMs). It rests on the assumption that the checker scoring each response is itself correct. In practice, rule-based checkers reject correct answers in unexpected formats, unit tests accept incorrect programs, and LLM judges can be led to approve empty or adversarial responses. This work surveys 58 studies that measure, analyze, or mitigate such verifier errors. We ask what errors verifiers make, how they affect training, which mitigations work against which errors, and how the findings extend to self-generated, rubric- and judge-based, and verifier-free rewards. The evidence suggests that the structure of an error matters more than its rate. Random errors mostly slow learning, though under a fixed budget or KL regularization they also lower final accuracy. False positives that the policy can learn to trigger can cause plateaus or collapse, sometimes even with a low initial error rate. Whether a verifier wrongly rejects or accepts depends on what it checks and who built it, not on its family. Its errors shift as the policy improves, and most mitigations have been tested against only one kind of error. We compile the findings into comparison tables, reconcile apparent conflicts in the literature, and propose a reporting protocol to make future results comparable. 

---
# More Reader Data or a Better Teacher? Updating a Model of Crowd Highlights in Late 2026 

**Authors**: Kazuki Nakayashiki, Keisuke Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2610.17768)  

**Abstract**: A model trained on readers' highlights can show a newcomer where earlier readers converged on a page, even on pages with few readers. A year of stronger language models suggests two upgrades: use them directly, or use them as teachers. We compare both with a third, more reader data, on one social highlighting platform, with the comparisons, the selection rules and a sealed set of fresh pages fixed in advance. On 215 development pages, seven language models released by October 2026 rank sentences zero-shot at an average precision of at most 0.3744 against the sentences half the readers marked most, and 0.3931 when fused with reading position, while a 3-billion-parameter model trained on reader highlights alone reaches 0.4346. Used as the teacher that breaks ties among readers' counts, the stronger fusion changes a trained model by +0.0013 [-0.0012, +0.0040], whereas going from half to all of its training pages adds +0.0242 [+0.0179, +0.0301]. A 27-billion-parameter backbone on the reader-only recipe adds +0.0225 on the development pages; on 131 fresh pages, read once and too few for a confirmatory comparison, its difference from the 3-billion model is unresolved, both score above the fusion, and in an exploratory split their lead over the previous model sits on pages from domains that their fresh training pages also cover. In this recipe, more reader data improves a trained model where a stronger tie-breaking teacher yields no material gain, and frontier models used as rankers still trail a small model trained on readers. 

---
# SkillScreen: High-Recall Screening of Malicious Agent Skills with Decision Models 

**Authors**: Jerrold Huang, Junwei Quan, Saiyue Lyu, Zihan Guo, Yingqiang Ge  

**Link**: [PDF](https://arxiv.org/pdf/2610.17727)  

**Abstract**: Agent skills turn natural-language instructions into actions, creating a security challenge: legitimate automation and malicious behaviour can look deceptively similar. A practical screen must catch nearly every attack while keeping the review load manageable. We present SkillScreen, a compact decision model trained to score skill maliciousness directly from this http URL in a single forward pass. On MaliciousSkillBench, SkillScreen flags 20-25% of benign skills at 99% recall on unseen attack families, substantially fewer than the word-level classifier and prompted LLM judges tested. Under the same training protocol, it also generalizes better than a fully fine-tuned encoder to unseen sources. Yet unfamiliar benign sources sharply increase false positives, showing that recognizing new attacks does not ensure reliable screening of new skill collections. Our results demonstrate the value of task-trained decision models for high-recall skill screening. Project page: this https URL 

---
# RALI@iKAT 2026: Retrieval-aware User Profile Denoising via Personalized Conversational Dense Retrieval 

**Authors**: Yuchen Hui, Fengran Mo, Jia Ao Sun, Yuxing Tian, Wenhao Xu, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2610.17673)  

**Abstract**: Modern conversational information access systems continuously extract user information from conversations to build user profiles for personalized retrieval. As user profiles accumulate over time, they inevitably become noisy, making effective user profile denoising a fundamental challenge. Existing personalized conversational query reformulation methods rely on the black-box reasoning capability of LLMs to expand the original query with relevant user profile phrases, producing a verbose query for retrieval. Such rewrite-then-retrieve pipelines reduce user profile denoising to the explicit selection of profile phrases in the sparse token space, separating personalization from downstream relevance modeling. Consequently, they often fail to identify user information that is truly beneficial for retrieval.
To enable retrieval-aware user profile denoising, we propose Personalized Conversational Dense Retrieval (PCDR), which learns to jointly project the conversation history and user profile into a latent embedding space dedicated to retrieval. By inheriting the context denoising capability from conversational dense retrievers, PCDR requires only a handful of personalized relevance annotations for training. We evaluate PCDR on the iKAT 2026 tasks, where it serves as the core retrieval component of a conversational AI agent assisting users with complex information needs. 

---
# DeltaTTG: Training Multi-Turn Agents with a Token-to-Go Progress Meter 

**Authors**: Xuyan Ye, Xiyan Fu, Shengda Fan, Jiawei Cui, Yankai Lin, Wei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.17666)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) for multi-turn agents relies on sparse terminal rewards that say little about individual steps. Within-group step comparisons depend on reliable state matching and sufficiently diverse sampled outcomes, while direct step scoring may require costly external evaluators, human annotations, or task-specific heuristics. We introduce DeltaTTG to learn a reusable progress signal through a shared progress meter, using low-cost supervision from diverse policies' recorded trajectories without task-specific progress rules. Its supervision target, Token-to-Go (TTG), the number of tokens remaining to be generated until task completion, serves as an observable proxy for remaining completion effort. During actor training, changes in the frozen meter's estimates redistribute outcome-derived advantages across steps without altering the reward, preserving each trajectory's advantage sign and token-average scale. On ALFWorld, PaperSearchQA, and BFCLv3, DeltaTTG improves the average score over GRPO by 5.7 and 7.2 points with Qwen3-4B and Llama-3.1-8B-Instruct, respectively, outperforms all compared RL baselines on average, and keeps actor-training GPU-hours close to GRPO. Further experiments show improved sample efficiency and out-of-domain performance, and support the meter's ability to distinguish step quality and guide test-time action selection across actor sizes. The code is available at this https URL. 

---
# DisaggKD: Efficient Knowledge Distillation through Teacher-Student Disaggregation and a Fused Loss 

**Authors**: Ruixuan Tu, Yuyang Dong  

**Link**: [PDF](https://arxiv.org/pdf/2610.17644)  

**Abstract**: Knowledge distillation (KD) frameworks place teacher and student on the same GPUs, so per-GPU memory limits the teachers that fit. DisaggKD separates them across nodes: teacher replicas stream hidden states to the student, a KD loss over row windows and vocabulary shards equals the full-vocabulary loss in exact arithmetic without forming the full-vocabulary logits, and a prefetch ring paces producer and trainer. Closed-form memory and traffic models locate the memory limits of each layout. On the Qwen3.5-35B-A3B to 9B pair, the disaggregated layout matches the LM loss of the colocated run with the same loss and is faster than that run; our fused loss also lets a colocated run train on as few nodes at faster speed, while the split is memory efficient when the student spans several nodes. On the larger pairs, colocated runs with a fused loss still run OOM where the split trains, because teacher weights and student states fill the budget. 

---
# Executable Prefix Values for Locating Agent Failures 

**Authors**: Bingyu Shen, Jiaqi Tang, Lan Wei, Yi Sheng, J. Jenny Li, Boyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.17633)  

**Abstract**: Work on debugging LLM agents treats the "first mistake" marked by a human or LLM judge as ground truth for where a trajectory failed. We ask whether such labels point at causes or symptoms, and make the distinction executable: we restart the agent from reconstructed intermediate states of a failed trajectory and estimate its prefix value V(h_t), the probability of success from each step, with a certified point of no return (PNR) for a failure. On 150 SWE-bench Verified tasks with two frozen open-weight agents, we build such curves for 47 and 60 failed trajectories and probe every step labeled by four frontier judges under the Traverse rubric. We find that the labels do not mark what a controller needs. Judges of different lineages agree on the exact step in only 21/57 cases, and the newest judges flag mistakes in 8/10 and 10/10 successful runs. For 66% of the labels, the agent's success probability does not drop at the labeled step; only 7/188 answers name the decisive step. Restarting at the measured restart point raises expected success from 0.44 to 0.57 offline, and real restarts confirm the advantage, whereas restarting at a judge's label does not (-0.08 to -0.03). We release our code, curves, and labels. 

---
# RELAY: Selective Self-Distillation with Adaptive Hand-off for Rubric-Based Reinforcement Learning 

**Authors**: Kunwang Zheng, Xinyu Sun, Yu Cao, Shuocheng Li, Lei Cheng, Ying Liu, Qilong Zhang, Jinlong Chen, Hua Yang, Qianglong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.17623)  

**Abstract**: Rubric-based reinforcement learning grades a response against many written criteria but collapses them into one reward, which discards the detail the rubric carries and gives no GRPO reward gradient when a group's responses tie. A frozen teacher that reads the rubric as privileged context provides token-level supervision, but we observe an early plateau in the gains over GRPO. RELAY uses advantages and rewards to select positive-advantage and low-scoring zero-advantage rollouts, and monitors the distillation loss to decide when to retire the teacher. Stage 1 combines selective distillation from a frozen teacher with GRPO; a plateau detector then removes the teacher, and Stage 2 continues with GRPO alone. Calibrated on one model-domain pair and reused unchanged in three others, the plateau detector fires between steps 122 and 171. RELAY scores highest in-domain among the compared methods in all four settings, ahead of GRPO on RaR-Medicine by 4.62 points with Qwen3-1.7B and 6.02 with Qwen3-4B. On Qwen3-1.7B/Medicine, removing the teacher from a shared checkpoint outperforms keeping, weakening or refreshing it. Distilling selected rollouts covering about half the response tokens also outperforms distilling all rollouts at a matched average coefficient. 

---
# StateHarness: Maintaining Evolving Project State for Long-Horizon Coding Agents 

**Authors**: Chuan Zou, Yujun Zhou, Xiaonan Luo, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.17616)  

**Abstract**: Long-horizon coding requires agents to maintain an evolving view of the project they are changing: what problem is currently being solved, what has already been learned or modified, which other artifacts may be affected, and what work remains unresolved. When this information is scattered across a growing interaction history, local edits can become disconnected from their later consequences and progress can be difficult to carry forward. We introduce StateHarness, a runtime framework that maintains evolving project state without updating model parameters. The working agent records its current problem and hypothesis, important findings and decisions, discovered repository relationships, and unfinished work. Runtime support prompts record updates, searches for callers, tests, and documentation that may be affected by a change, and maintains a compact working brief for subsequent decisions. When behavioral signals accumulate, a separate diagnostic session revisits the recorded history and surfaces overlooked evidence. Across three repository-level evaluations, StateHarness improves task completion and several measures of change consistency. On RefactorBench, full-chain completion increases from 29.4% to 35.3% and cleanup-check accuracy from 87.0% to 95.4%. On five LHTB software-engineering tasks, mean reward improves from 0.427 to 0.576, exceeding LongHorizon-Harness at substantially lower recorded execution cost. On ChainSWE, bug-fix rate increases from 35.1% to 39.5% and residual debt decreases from 60 to 43 items, although pass-to-pass regressions increase. These results show that maintaining project state can improve long-horizon coding while introducing measurable execution and regression trade-offs. 

---
# Tanawpamana: Benchmarking Vision-Language Models on Philippine Cultural Heritage Sites 

**Authors**: Ivan Yuri De Leon, Hiroki Ouchi, Sakriani Sakti  

**Link**: [PDF](https://arxiv.org/pdf/2610.17596)  

**Abstract**: Vision-language models are increasingly used to identify and describe places, yet their recognition and knowledge of specific cultures remain poorly understood, particularly in underrepresented regions such as the Philippines. This study introduces the Tanawpamana Dataset, a repository of Philippine heritage site images with metadata based on the Talapamana registry of the National Commission for Culture and the Arts of the Philippines. We also introduce the Tanawpamana Benchmark, comprising 4,046 tasks designed to evaluate models' recognition and knowledge of heritage sites. Evaluating seven vision-language models, we find that they struggle more with fine-grained visual discrimination among similar heritage sites than with coarse recognition, and that their geographic knowledge is weaker when inferred from image inputs than from text. 

---
# Reference Wording Shapes Retrieval of Robot Task Instructions 

**Authors**: Jahnavi Yelamanchi  

**Link**: [PDF](https://arxiv.org/pdf/2610.17549)  

**Abstract**: Identifying a robot task from a short instruction is sensitive to which descriptions are available for retrieval. We investigate this effect in the natural-language instructions included with the RLBench manipulation benchmark. We extract 296 fixed descriptions from 62 publicly available task files and compare five training-free lexical rankers over 80 random choices of one indexed description per task. Exact strings assigned to conflicting task labels are excluded from the primary score. Character n-gram TF-IDF achieves 61.3% mean top-1 accuracy on the resulting held-out instructions, compared with 55.6% for BM25 and 54.9% for word TF-IDF. Across gallery choices, character TF-IDF ranges from 55.2% to 66.8%; its 5th-95th percentile range is 56.9%-64.8%. Adding a second indexed description improves character TF-IDF by 7.0 percentage points on matched queries. These results quantify the influence of reference wording and representation choice on instruction-to-task retrieval under a consistent evaluation protocol. 

---
# A Sense of Proprioception in Language Models 

**Authors**: Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2610.17495)  

**Abstract**: In humans, proprioception corresponds to the sense of self-movement or position, e.g. understanding the position of one's arm. This sense is often less well-understood and in popular understanding sometimes conflated with introspection. But, notably, proprioception is a separate sense, with dedicated receptors and verifiable outcomes. On the other hand though, language models cannot observe their own internal states during training (and for example answer the question which experts activated in a certain layer of an MoE) and so cannot accurately report on them. Yet, such information is, by virtue of being internal to the model, both accessible, and readily constructed into verifiable signals. We argue that this provides the basis for a digital analog of proprioception. As such, we train models through SFT and on-policy training to equip them with this ability. We show that modern models can be trained to understand a range of subtypes of proprioception, such as routing decision awareness, token entropy, decision depth or logit lens information, without affecting model capabilities. We then run these models through a broad evaluation of situational awareness, legibility, and epiphenomenality, to evaluate in what ways access to proprioception changes model behavior. 

---
# Scaling Inference-Time Compute with Teams of Coding Agents 

**Authors**: Keenan Samway, Daniel D'souza, Omar Darwiche Domingues, Marzieh Fadaee, Tan Yi-Chern, Nouha Dziri  

**Link**: [PDF](https://arxiv.org/pdf/2610.17471)  

**Abstract**: Multi-agent systems promise a new axis for scaling inference: rather than having one agent think longer or sample more attempts, a task is split across many agents that work in parallel and coordinate. However, in practice, more agents can easily produce duplicate work, coordination failures, and higher costs. We therefore formulate team design as a compute allocation problem: starting from a solo agent, where should additional inference compute go to reach the reward-cost Pareto frontier? We build a minimal multi-agent harness from the ground up and trace this frontier by varying, one at a time, the workflow stage at which a lead delegates to its workers (planning, implementation, or verification), the models that form the team, and the number of workers. On ProgramBench, teams outperform solo agents but cost three to seven times as much. With the right compute allocation, however, a heterogeneous 17-agent team comes within 1.4 points of a strong homogeneous 5-agent team at 70% lower cost. This efficiency stems from three findings. First, planning is the most valuable use of workers: in a homogeneous team whose lead decides how many workers to spawn, a lead that delegates only planning to its workers, and then implements and verifies the code itself, matches the reward of a lead that delegates to workers throughout planning, implementation, and verification, at half the cost. Second, cheap models pay off as planners but not as implementers: a strong lead with cheap planners recovers 47.5% of the reward gain of an all-strong team for only 13.9% of its additional cost, whereas cheap implementers lower reward. Third, planners keep improving reward as the team grows, whereas implementers and verifiers plateau beyond four workers. Together, our results offer a guide to spending an agent budget where it counts, given the task, cost, and latency constraints a user has. 

---
# The belief-action gap in language model agents 

**Authors**: Ayse Sila Okcu, Weisheng Wang, Tongmeng Xie, Niall Dalton, Angelos Nalmpantis, Raghu Arghal, Mario Giulianelli  

**Link**: [PDF](https://arxiv.org/pdf/2610.17444)  

**Abstract**: Language model (LM) agents can fail for different reasons, requiring different explanations and interventions. We study how state beliefs, action values, and action preferences interact as LM agents reason and commit to decisions in text-rendered grid worlds. We estimate task-relevant beliefs using report-based and representation-based probes, and action values using inverse reinforcement learning. We find that task-relevant state and value information is often represented more reliably than it is reflected in behaviour. Tracking action preferences throughout reasoning further shows that decisions sharpen around a commitment point and reported belief uncertainty decreases near commitment, although individual belief changes do not consistently explain shifts in action preference. Finally, forced actions follow reasoning prefixes from other decisions when transplanted, whereas the tested reasoning-window activation edits provide limited control over the action. Together, these results suggest that some failures arise not from missing task-relevant information, but from how available information is translated into action. Our measurements distinguish information availability from belief-action consistency without by themselves identifying the cause of each failure. Our code is available at: this https URL 

---
# EPOCH: Coverage-Aware Evolution with Experiential Preference Optimization for LLM-based Automated Heuristic Design 

**Authors**: Oguzhan Gungordu, Ayse Tursucular, Siheng Xiong, Faramarz Fekri  

**Link**: [PDF](https://arxiv.org/pdf/2610.17379)  

**Abstract**: LLM-based automated heuristic design (AHD) discovers heuristics for combinatorial optimization problems (COPs), where an LLM generates candidates in an evolutionary search. Existing frameworks freeze the generator or fine-tune it on a scalar fitness signal hiding which instances each heuristic covers. Reflection-based frameworks improve heuristics with natural-language feedback, but it stays in the prompt and never reaches the generator's weights. We propose EPOCH, which makes instance coverage the objective of the evolutionary search and fine-tunes the generator to internalize reflection. Parent selection, operator allocation, and stagnation recovery all optimize how well the heuristics jointly cover the training instances, so the search returns a complementary heuristic ensemble. The generator learns from its own search by experiential preference optimization. A teacher LLM returns a natural-language insight on the generated heuristics, and the generator produces new heuristics with this insight. These improvements join the preference pairs for direct preference optimization (DPO). Since DPO only teaches the generator to rank these improvements, a reflective self-distillation (RSD) term makes it more likely to generate them on its own, and the gains accumulate over the long horizon of the search. Across COPs and open-source LLM backbones, EPOCH's heuristic ensembles outperform inference-only and fine-tuned baselines and generalize across problem scales. 

---
# Online LLM Personalization from Pairwise Preferences Across Users 

**Authors**: Romain Puech, Georgy Salakhutdinov, Dimitris Bertsimas  

**Link**: [PDF](https://arxiv.org/pdf/2610.17203)  

**Abstract**: Large language models can answer the same question in different styles, but users' preferred styles are hidden and vary across users and questions. We study inference-time personalization through online selection from a fixed set of prompts, each representing a persona. Users arrive sequentially, each asks a short sequence of questions, and the system displays two responses per question and observes one pairwise choice. We formulate this interaction as a variant of the contextual dueling bandit problem in which the learner starts without preference data, adapts within each session, and exploits information shared across users. The objective is to minimize cumulative strong regret, summing the utility gaps of both displayed responses relative to the user's best response among the available options. We make two contributions. First, we introduce the problem and release a reproducible preference benchmark and dataset built around a tutoring case study, containing educational questions, LLM-generated answers in six teaching styles, and pairwise judgments from LLM-simulated students with hidden preference profiles. We fit Bradley-Terry models to these judgments to obtain reference utilities for regret evaluation. Second, we propose a baseline that learns a mixture prior over users from completed sessions and adapts to each new user within the session. On students and topics never used in development, it removes 44-48% of random pairs' excess regret and achieves over half the performance of a per-user linear oracle fitted in hindsight. We release code, prompts, and the benchmark. 

---
# MLAfication: Efficiently Enabling High-Fidelity MLA for Modern LLMs 

**Authors**: Zile Wang, Zhide Xie, Tao Ji  

**Link**: [PDF](https://arxiv.org/pdf/2610.17202)  

**Abstract**: Multi-Head Latent Attention (MLA) compresses the KV cache into a latent representation, reducing inference memory and bandwidth. Existing MHA2MLA/TransMLA methods do not support QKNorm and require costly recovery on modern LLMs, especially MoE. We present MLAfication, a QKNorm-compatible, parameter-efficient extension of MHA2MLA. It combines calibrated KNorm, parameter-efficient distillation, contribution-aware Partial-RoPE, and activation-aware joint KV initialization. Across MHA and GQA models, including QKNorm- and MoE-equipped variants, MLAfication preserves performance under substantial KV-cache compression, surpasses the current SOTA on commonsense and long-context benchmarks, and extends MLA conversion to modern architectures. On Qwen3-30B-A3B, MLAfication delivers up to $5.35\times$ the serving throughput of BF16 GQA and, combined with 4-bit KV quantization, reduces the KV-cache footprint by $94.73\%$. Our code is available at this https URL. 

---
# Learning Perceptual Geometry of Natural Odor Mixtures from LLM-Derived Descriptors 

**Authors**: Farhad Pashakhanloo, Venkatesh N. Murthy  

**Link**: [PDF](https://arxiv.org/pdf/2610.17164)  

**Abstract**: The relationship between odor chemistry and perception remains incompletely understood. Many existing studies focus on the perception of single molecules, whereas natural odors contain tens to hundreds of components, adding extra layer of complexity to percept formation. Here, we explore the relationship between molecular composition and perceived similarity in natural odor objects. We study this using a database of more than 1,200 volatile-compound-in-food mixtures containing more than 8,000 molecular components, and use large language models (LLMs) as a scalable surrogate for deriving mixture-level perceptual descriptions. We find that language-derived representations provide useful supervision for predicting held-out mixture similarities from molecular composition. The prediction was consistently strongest when perceptual targets were derived from chemosensory prompts than from other sensory prompts. Further, during the mixture-level training, we learned low-rank molecular embeddings that enabled better prediction of held-out mixture similarities than component-overlap metrics, even for mixture pairs with no shared molecules. Although no single-molecule labels were used during training, the learned molecular geometry showed significant alignment with the independent human odor-descriptor space. Finally, we explored the use of external molecular features in our mixture-similarity prediction task. These features showed modest zero-shot alignment with mixture-level perceptual similarity. They improved prediction under limited supervision but offered little additional predictive benefit with the full training set. Together, these results show that language-derived descriptions can provide scalable supervision for predicting the perceptual organization of complex natural mixtures, contributing to a quantitative understanding of how molecular composition gives rise to odor perception. 

---
# The Format is the Filter: Scoring Format Fixes What an LLM Judge Can Express 

**Authors**: Lovedeep Gondara, Jekaterina Novikova, Sagarika Suresh, Jack Ji, Jithin Pradeep  

**Link**: [PDF](https://arxiv.org/pdf/2610.17158)  

**Abstract**: When an LLM judge filters instruction-tuning data, its scoring format (binary verdict, 1--5 rating, 0--100 score, or comparison) is rarely justified or ablated. Holding the candidate pool, judge prompt, decoding and selection budget fixed, we score 15,406 UltraFeedback candidates under five formats with three judges, curate, fine-tune 66 models, and evaluate them with four judges. A 0--100 scale is worth only 7--12 equally used levels, two to three times a 1--5 scale rather than twenty, and a binary verdict leaves the top score tied in 77--78\% of prompts, so most selections are made at random. These properties are set by the format almost independently of the judge. Kendall's $\tau_b$ hides them: it ranks binary best while $\tau_a$ and top-1 agreement rank it last, because it discounts the ties a selector must break at random. Both comparative formats inherit a slot bias whose sign differs across judges. Downstream, curated data beat random selection under every evaluation judge, robust to length and markdown adjustment, but no ordering among formats is detectable. We propose four quantities that judge-based curation pipelines should report. 

---
# When LLM Agents See a Conflict but Still Proceed: Cross-Domain Dependencies in Network Slicing 

**Authors**: Marcos Lima Romero, Ricardo Suyama  

**Link**: [PDF](https://arxiv.org/pdf/2610.17118)  

**Abstract**: Future AI-native networks are being designed to apply autonomous agents built on large language models (LLMs) in network management through intents. When intents from different domains meet on the same resource, an agent must recognize the cross-domain dependency before taking an irreversible action. We investigate this problem through a smart-grid use case, in which a power utility holds an active intent on a backup slice for protection traffic. This slice appears idle by design, which is precisely the profile targeted by a cost-reduction intent. We built a tool-based environment in which the dependency must be either queried by the agent or read from an inventory annotation, together with a negative control, and evaluated ten models from four different providers in 600 runs. Most models acted against the dependency they had observed: seven of the ten models removed the protection slice at least once after receiving the cross-domain dependency, and in the high-harm tier the slice was removed in 76% of the exposed runs, while four models practically never did. Only one of ten models never removed the slice. We identify two failure modes, not querying the dependency and querying it but proceeding anyway, and show that agents often reported no conflict after observing it. These results indicate that, in AI-native networks, visibility of cross-domain dependencies is not a sufficient safeguard, and that dependency verification should be enforced by the management architecture before irreversible actions rather than delegated to the LLM's judgment. 

---
# Towards Universal Calibration for In-Context Classification 

**Authors**: Awik Dhar, Sunita Sarawagi, Arpit Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2610.17096)  

**Abstract**: In-Context Learning (ICL) has received a lot of interest recently due to its ability to improve classification accuracy for new tasks without having to fine-tune the model over these tasks. However, the impact of ICL on calibration has been far less studied. Recent work has shown that ICL can have a negative impact on calibration due to biased selection of in-context examples. The existing methods for improving ICL calibration require additional task and model-specific training, detracting from the no-fine-tuning convenience of ICL. In this work, we introduce CALIBFORMER, a universal ICL calibrator for classification tasks, that generalizes to new models and tasks without further fine-tuning on these tasks. Specifically, we design a transformer-based calibrator which uses the model predictions on in-context examples to predict a confidence score for the test input. We show that the calibrator trained on three models and ten tasks, is able to generalize/calibrate ICL predictions on several new tasks and four LLMs unseen during training. Even for frontier models like Gemini, CALIBFORMER is able to obtain over 50% reduction in calibration error. 

---
# Understanding Personalization Gains in Whisper 

**Authors**: Sergei Katkov, Antonio Liotta  

**Link**: [PDF](https://arxiv.org/pdf/2610.17019)  

**Abstract**: Whisper-style ASR models provide strong zero-shot transcription, but their residual errors are not evenly distributed across datasets, recording conditions, and metadata-based groups. We study when lightweight low-rank adaptation (LoRA) improves Whisper and whether observed gains require group-specific adapters. Across Common Voice, LibriSpeech, and VoxPopuli, we compare zero-shot Whisper, pooled LoRA, and group-specific LoRA under matched within-group splits. We find that adaptation is not uniformly beneficial: gains are concentrated in difficult Common Voice groups, are consistent on VoxPopuli, and largely disappear on low-error LibriSpeech. Crucially, pooled LoRA often matches, and sometimes clearly outperforms, group-specific LoRA. Gains over zero-shot should therefore be tested against matched pooled adaptation before being interpreted as evidence of personalization. 

---
# Davinci-Tool: Scaling Agentic Tool-Use Supervision with Verifiable Executable Environments 

**Authors**: Xiaoyang Yuan, Ji Zhao, Xingyu Hu, Wenhan Wang, Yutao Hu, Qiaosheng Zhang, Pengfei Liu, Yu Qiao, Yi Bin, Wenqi Shao  

**Link**: [PDF](https://arxiv.org/pdf/2610.16867)  

**Abstract**: Reliable agentic tool use requires training data that pair diverse multi-step workflows with executable evidence that the requested outcomes were achieved. Scaling either trajectory coverage or verification alone is insufficient as new environments introduce distinct states, rules, and dependencies that must also be reflected in the success criteria. To address this challenge, we introduce Davinci-Tool, a 30B tool-use model, and Davinci-ToolData, an execution-grounded dataset designed to scale environment coverage and outcome verification together. Davinci-ToolData preserves restorable states, execution traces, outcomes, and verifiers as a unified substrate for supervision. Two complementary pipelines populate this substrate: Execution-Grounded Task Synthesis (EGTS) derives verified demonstrations from independently authored systems, and Executable Environment Synthesis (EES) synthesizes auditable environments, state-grounded tasks, and executable verifiers across single and coordinated domains. These pipelines yield 30,395 verified trajectories spanning 2,260 invoked tool names. We first train on these demonstrations with supervised fine-tuning, then reuse EES states and verifiers to compare alternative actions and construct outcome-grounded preferences for Direct Preference Optimization. Davinci-Tool improves Qwen3-Coder-30B-A3B-Instruct from 52.4 to 88.2 on $\tau^2$-bench, from 30.4 to 62.8 on BFCL v4 multi-turn, and from 16.9 to 56.2 on MCP-Atlas, outperforming Qwen3-Coder-480B-A35B-Instruct on all three benchmarks with only 1/16 of its total parameters. Code is available at this https URL. 

---
# The Bell Boundary of Shared Representations 

**Authors**: Yang Wu, Hanyu Zhao, Yazhou Zhang, Yuexian Hou  

**Link**: [PDF](https://arxiv.org/pdf/2610.16828)  

**Abstract**: When does a shared representation become insufficient for jointly supporting multiple tasks? This question lies at the heart of multi-task learning models. Although task interference in shared representations has been widely observed, its onset, measurable form, and capacity boundary remain poorly understood. In this paper, we revisit this problem through the lens of Bell's theorem. We show that if four binary task outcomes can be jointly represented by a common shared representation, their correlations must satisfy the classical CHSH bound $|S|\leq2$. We introduce NCnet, a controlled multi-task model for studying how representational capacity affects task compatibility. Experiments reveal a clear capacity-dependent transition. Once nontrivial task fitting emerges, constrained capacity can yield $|S|>2$. As capacity further increases, these exceedances progressively disappear and $S$ returns to the bounded region. This systematic pattern provides evidence that limited representational capacity is an important source of the observed incompatibility. We observe similar transitions across CNNs, PLMs, and LLMs, suggesting that the phenomenon extends beyond the controlled NCnet setting. We refer to this capacity-dependent transition as the Bell Boundary, providing a measurable diagnostic for representation bottlenecks, task incompatibility, and capacity allocation in multi-task models. 

---
# TPPO: Tracking-Value PPO For Language Model Reasoning 

**Authors**: Futing Wang, Chenxi Li, Jianhao Yan, Yun Luo, Zhi Wang, Yizhuo Li, Runzhe Zhan, Yu Cheng, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.16794)  

**Abstract**: Proximal Policy Optimization (PPO) is widely used in reinforcement learning with verifiable rewards for language-model reasoning. While \emph{value evolution along trajectory} characterizes relative token-level credit, yet standard critics often fail to track it under sparse terminal rewards. To model this evolution, we analyze the relation between successive prefix values and show that they admit a Bayesian decomposition with additive updates in log-odds space. We use this additive structure to construct Tracking-Value PPO (\method), a cumulative value parameterization that sums position-wise scalar outputs into value logits. These outputs are learned through terminal-outcome supervision on the reconstructed values. \method improves aggregate mathematical reasoning scores over PPO by 7.0 and 8.4 points on Qwen3-4B-Base and Qwen3-8B-Base, respectively, while improving out-of-distribution generalization. Gains extend to Qwen3-4B, Qwen2.5-Math-1.5B, Qwen3.5-9B and to countdown, and scientific coding tasks. Further analysis shows more accurate tracking of value evolution and more reliable advantage estimates. Our code is publicly available in this https URL. 

---
# Is the Platform Part of the Measurement? A Protocol and Simulation Study for Cross-Platform Equivalence of Machine-Labelled Policy Sentiment 

**Authors**: Asad Ikram, Ella Haig  

**Link**: [PDF](https://arxiv.org/pdf/2610.16757)  

**Abstract**: Comparing machine-labelled sentiment across social media platforms is now routine: differences in classifier scores between platforms are read as differences in public opinion. These comparisons rest on an assumption that the classifier measures the same construct, on the same scale, on every platform. Some studies check a classifier's validity platform by platform, but to our knowledge none has modelled the platform as a facet of the classifier's measurement. We adapt the psychometric equivalence toolkit (generalizability theory, measurement invariance and differential item functioning, and score linking) to machine-labelled policy sentiment, treating the platform as a survey mode, and specify a six-step protocol with anchor events fixed in advance, an anchor-density audit, and a two-classifier convergence control. A simulation with known ground truth illustrates what the assumption can cost in one generating configuration fixed in advance: across 500 replications, because the platforms measure through different functions, a pooled index that takes platform scores at face value reports the wrong sign of aggregate sentiment on nearly a quarter of days, whereas an index adjusted with each platform's known measurement function does so on under one day in sixty on average. A case study of the instrument from the first author's MSc dissertation, a fine-tuned BERT classifier over 255,446 UK economic-policy comments from Twitter/X, Reddit, YouTube, and Quora, motivates the design and illustrates the instrument-dependence the protocol addresses. The full four-platform application, under fresh collection with the pre-specified design, is the designated follow-up study. The protocol lets a researcher test whether pooling platforms is defensible in a given corpus, or whether platform scores must be linked before any average is taken. 

---
# Learn from Move: the Next Step for GUI Agents 

**Authors**: Jiahao Lu, Yize Shen, Chuanyuan Tan, Mianqiu Huang, Shicheng Fang, Xinchi Chen, Yining Zheng, Yuxin Wang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2610.16720)  

**Abstract**: Existing benchmarks for GUI agents primarily focus on increasing task difficulty by extending task horizons. Besides, enabling agents to explore unfamiliar environments and successfully complete tasks within them is also a critical yet challenging problem. We introduce LatentGUIWorld, a benchmark of six tasks in three matched pairs organized by the evidence needed for control: $I_0$ uses prior knowledge and available observations, $I_1$ requires interaction that directly reveals hidden semantics, and $I_2$ identifies dynamics from action-response relationships. Frontier models struggle especially on $I_2$ tasks. We introduce MOVE, which separates movement from button events and exposes intermediate visual feedback, together with demonstrations of iterative exploration and control. Ablations show that both action granularity and multi-step demonstrations improve performance. After SFT, LatentLearner-9B achieves 50.0% overall success on LatentGUIWorld, comparable to several frontier models. Joint RL further raises egocentric Drag and Ten-Choice success to 92.0% and 75.3%. Behavioral analysis reveals multi-step exploration. In egocentric Drag, RL increases Pearson correlation between error-normalized commands and ideal gain compensation from 0.610 to 0.803. Together, these behaviors show that models learn to explore through interaction and adapt their control to environment dynamics. The same format improves GUI grounding after RL, reaching 67.37% mean accuracy across five benchmarks versus 66.72% for conventional RL. 

---
# Attention-Graph Curvature for the Verification of Mathematical Reasoning 

**Authors**: Kanghyeon Kim, Johannes F. Lutzeyer, Yanzhu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.16709)  

**Abstract**: Language models have recently solved open Erdős problems and achieved gold-medal level performance on International Mathematical Olympiad problems. But their solutions, written in natural language, cannot yet be verified by proof assistants. An erroneous reasoning step can invalidate every step that comes after, so it is advantageous to detect errors early. Several methods have been proposed to detect erroneous reasoning quickly based on the features of the forward pass of the language model that performs reasoning. In this work, we study whether the directed Forman curvature of that model's attention graph, a local quantity computed at an edge from the attention weights around it, carries information about a step's correctness. To do so, we evaluate four per-step verifiers on ProcessBench with and without the curvature appended to their input. Appending curvature values improves the overall mean performance of all four verifiers. ReProbe, the verifier that takes every token of a step as input, shows the largest improvement. Adding the curvature increases the time of the language model's forward pass by 18% on average. 

---
# Does a Hosted Decision Model Respect Irrelevant Alternatives? A Pre-Registered Test of Option-Set Dependence in Jev 

**Authors**: Dushyant Rajput, Nirdesh Chauhan, Siddharth Kosaraju  

**Link**: [PDF](https://arxiv.org/pdf/2610.16655)  

**Abstract**: Decision APIs that return a probability for each option are routinely used as if those probabilities were properties of the options themselves. That reading is valid only if the model satisfies independence of irrelevant alternatives (IIA): adding an option that is neither the right answer nor its strongest rival should leave the odds between those two unchanged. We test this in a pre-registered study of jev-1.13.0, a hosted decision model, using 299,463 calls on intent classification (BANKING77, CLINC150) and topic classification (AG News). Every added option moves the log-odds between the gold label and the strongest wrong label: the root-mean-square systematic shift is 0.20-0.52 nats, 1.2 to 3 times the run-to-run variation, even when the added option takes at most 0.1% of the probability mass. The shifts are directional. An option similar to the wrong answer draws probability from it (+0.10 nats toward the gold label). A paraphrase of the wrong answer produces an intermediate pattern: the gold label's probability is almost conserved, but the paraphrase receives only about a third of the wrong answer's share rather than the half that strict substitution implies. The distortion grows steadily with the logarithm of the number of options, reaching about 1 nat at 255 options, and off-domain padding pushes the odds against the gold label. In contrast, questions packed into one call do not interact (bias < 0.001). Scoring each option with an independent yes/no question removes the option-set dependence but picks the right answer 8.7 points less often. Jev sits between two local baselines, an option-independent scorer and a prompt readout. Its probabilities are best read as relative to the option list they were computed over. 

---
# EAU: Environment-Agent-User Co-evolution for Training User-Centric Agents 

**Authors**: Min Cai, Dan Zhang, Gao Xing, Libin Ma, Zhaohui Xu, Xi Ye  

**Link**: [PDF](https://arxiv.org/pdf/2610.16523)  

**Abstract**: Training LLM agents to interact with users over multiple turns is challenging because the learning signal depends not only on the agent, but also on the user simulator and environment, which jointly shape the agent's experience. Simulated users can be stochastic or inconsistent, while environments may contain unreliable tools, incomplete state coverage, and incorrect verifiers. Prior work mainly addresses the resulting instability through reward shaping or denser supervision, while treating these components as fixed. We instead introduce EAU, a meta-agent framework for environment--agent--user co-evolution. During training, EAU monitors trajectories, tool traces, rewards, and training statistics; diagnoses failures and uninformative interactions; and proposes targeted modifications to the user simulator, environment, reward design, and RL configuration. Validated changes are hot-swapped into the ongoing run, allowing the training setup to evolve together with the policy. These adaptations also persist across runs, so later training can build on previously discovered improvements rather than restarting from a fixed setup. We enable this process through an editable symbolic representation of user and environment dynamics, while the agent remains a neural LLM optimized with RL. Across three interactive-agent benchmarks and two backbone models, EAU consistently outperforms prior RL methods, improving the overall score over the strongest baseline by 14.7--15.0% and increasing successful task completion on ColBench by up to 36.7%, showing that co-evolving the training system with the policy can substantially improve multi-turn RL. 

---
# Differential Privacy Protected Preference Alignment on Survey Derived Data 

**Authors**: Xiuyuan Qi, Yi Wang, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.16458)  

**Abstract**: Preference alignment enables language models to better reflect human values, but the preference data contains sensitive information, especially when derived from surveys about political beliefs, social norms, religion, gender attitudes, and cultural values. This paper studies privacy-preserving alignment on questionnaire-derived data from the World Values Survey, where supervision takes the form of ordinal labels rather than standard pairwise rankings. Existing efforts mainly focus on general privacy protections upon model training, while data-specific methods for survey data remain unexplored. We first show the vulnerability of aligned models, then formulate the task as rating prediction and introduce label differential privacy. Building on this formulation, we construct a differential private pipeline utilizing three privacy mechanisms: randomized response, exponential mechanism, and DP-SGD. Experiments show that simple label-level mechanisms outperform gradient-level methods, and proper aggregation function for preprocessing also make large contribution. The new fine-grained private comparison motivates explorations toward language model alignment with formal privacy guarantees. 

---
# VisExMEM: Visual Evidence-Based Explainable Multimodal Evaluation Metric 

**Authors**: Subhadeep Roy, Christoph Leiter, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2610.16441)  

**Abstract**: Current image-text evaluation often relies on a global score, which gives limited insight into which semantic details are supported by the image. Explainable metrics provide rationales, but the visual evidence supporting those rationales can still be difficult to verify. We introduce VisExMEM, a reference-free evaluator that decomposes captions into atomic claims, localizes visual evidence, verifies each claim, and averages claim-level support into a global score. Explanations use the same evidence and decisions. We train Qwen3.5-based 4B and 9B models on 100K samples with multi-task SFT, counterfactual refinement, and evidence-aware GDPO. Across eight benchmarks, VisExMEM-9B achieves the highest descriptive composite among open-source evaluators in our comparison, exceeding VQAScore by 11.7 points while remaining competitive with GPT-5.6 Sol. We also introduce AGITA-Bench, containing 600 human-annotated image-text pairs and 8.7K atomic claims for global and claim-level evaluation. For untuned Qwen3.5 and GPT-5.6 Sol controls, decomposed prompting raises the same composite by 3.6-5.1 points relative to holistic prompting. Human evaluation shows that VisExMEM explanations outperform prior explainable metrics and approach GPT-5.6 Sol. 

---
# ARMED: Adaptive Retrieval-Augmented Medication Recommendation 

**Authors**: Kyungkeon Chung, Se Un Park, Hyejin Lee, Junyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2610.16440)  

**Abstract**: Medication recommendation requires jointly considering diagnoses, comorbidities, longitudinal patient history, and treatment interactions. Although recent retrieval-augmented generation approaches using language models have shown promise for clinical recommendation, most existing systems rely on fixed retrieval pipelines that cannot adapt to diverse clinical retrieval scenarios and may introduce noisy or irrelevant evidence. We propose ARMED, an adaptive clinical RAG framework that dynamically expands retrieval according to retrieval sufficiency and current-admission relevance. ARMED first observes the initial retrieval outcome and selectively performs supplementary or fallback multi-query retrieval when additional evidence is needed. The framework also incorporates a relevance gate that activates only clinically relevant same-patient history and organizes retrieved evidence into structured query-level clinical evidence. Experiments on MIMIC-IV demonstrate that ARMED consistently outperforms previous static retrieval baselines and improves overall medication recommendation quality. By combining locally runnable small language models with adaptive retrieval, ARMED supports medication recommendation without relying on external language model APIs or task-specific fine-tuning. 

---
# RetroPlanner: Multi-Route Expansion and Multi-Objective Reasoning for LLM-Based Retrosynthesis Planning 

**Authors**: Ye Rin Kim, Yongjun Jeong, Seongjun Yun, Hanbum Ko, Soohyun Oh, Sungbin Lim, Sungwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.16391)  

**Abstract**: Multi-step retrosynthesis planning is often considered solved once a route to purchasable starting materials is found, but such a route should also be practical in real-world chemistry. Recent LLM-based planners formulate this process as sequential decision-making over an AND-OR graph, yet typically expand a single molecule per turn and are optimized for solvability alone, leaving alternative routes uncompared across the multiple objectives that determine practicality before the search budget is committed. We introduce RetroPlanner, an LLM-based planner that jointly compares multiple routes to steer the search toward higher synthesizability, as measured by reaction plausibility, round-trip consistency, and starting-material price. RetroPlanner is trained to reason over chemical evidence: structural progress from the target molecule, bond changes and reaction classes, where candidate routes diverge from one another, and how they rank across multiple objectives. Experiments on multi-step retrosynthesis benchmarks demonstrate that RetroPlanner recovers more alternative routes than prior LLM-based planners while improving synthetic route quality and maintaining a high success rate. 

---
# A Production-Grade, High-Efficiency Agentic RL Framework for Frontier LLMs with Validated Recipes 

**Authors**: LoonSage Team, Bowei Song, Caiwei Zhao, Chuwen Liu, Dechuang Gu, Dong Fang, Feiyu Xie, Huaiyu Li, Hui Zhang, Jian Tian, Jian Wang, Jie Zhao, Liguang Li, Maoyun Wang, Meiyan Yang, Minghan Zhu, Pai Liu, Pandeng Yao, Renbao Tian, Ru Yang, Shuai Jia, Shuai Tan, Shuailong Li, Shuliang Li, Teng Jin, Tianze Yang, Wenbo Cui, Wenjing Bai, Wenkang Wu, Xiaowei Jiang, Xiaoyi Dong, Yanpeng Wang, Yanwen Wang, Yaohui Ma, Yawen Wang, Yihang Zhang, Yingjie Yan, Zhanbiao Lian, Zhenyu Lu, Zhida Hu, Zhihui Hao, Zhihui Hu, Zhiwen Xiao, Zhiyi Gao, Zhongwei Qi, Zhuangzhuang Wu, Ziqian Liu, Ziyue Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.16352)  

**Abstract**: Reinforcement learning (RL) has emerged as a critical phase in the post-training pipeline of large language models (LLMs). As the frontier of LLM capabilities shifts from static reasoning to agentic tasks--characterized by external tool invocation, multi-turn interactions, and long-horizon decision-making--existing mainstream RL frameworks have grown increasingly unwieldy, raising the overhead of adoption, usage, and debugging. To address these challenges, we present LoongSage, an agent-native, fully asynchronous RL framework with a lightweight core of roughly 14K lines of code. Built upon Megatron for distributed training and SGLang for high-throughput inference, LoongSage centralizes orchestration within a single controller process and encapsulates all GPU-resident components as Ray actors. It introduces AgentFlow, a unified execution paradigm that orchestrates both white-box and black-box agents. For asynchronous execution, it integrates sliding-window strategies that bound trajectory staleness, and employs TransferMesh, a topology-aware synchronization mechanism that routes weights after each step over the fastest transport available to each receiver. Beyond agentic RL, LoongSage incorporates a memory-efficient Teacher Manager that decouples the scaling of teacher models from GPU memory constraints, natively supporting Multi-Teacher On-Policy Distillation. Extensive evaluations across four complete training runs, spanning complex reinforcement learning and multi-teacher distillation, demonstrate that task rewards improve steadily while the train-inference gap is kept under continuous control, and the fully asynchronous pipeline removes synchronous idle bubbles, sustaining hardware utilization. 

---
# ATTCK-R: Knowledge-Aware Long-Horizon Jailbreaking for Protected LLMs 

**Authors**: Md Mushfiqur Rahman, Tobias Rohde, Rahul Tripathi, Khalil Elkhalil, Kevin Lybarger  

**Link**: [PDF](https://arxiv.org/pdf/2610.16305)  

**Abstract**: As large language models are deployed behind increasingly sophisticated safety mechanisms, evaluating their robustness under realistic adversarial conditions becomes increasingly difficult and critical. Existing jailbreaking evaluations often rely on static prompts, limited interaction horizons, or lightly protected systems, failing to reflect protected deployment settings with prompt-based safety protections. We show that the introduction of a prompt-based safety protection reduces the average attack success rate of a strong multi-turn baseline from 0.896 to 0.608 (32.1\% relative), demonstrating that unprotected evaluations substantially overestimate real-world vulnerabilities. We propose Adaptive Traceback and Tactic Curation for Knowledge-based Red-teaming (ATTCK-R), a planning-driven multi-turn red teaming framework designed for protected black-box systems. ATTCK-R models jailbreaking as a long-horizon trajectory optimization problem using a structured belief state, introduces optimization-based tactic curation to abstract reusable strategies from prior successes and failures, and employs informed traceback to resume attacks from strategically meaningful states rather than restarting. Evaluated across 9 widely used LLMs configured with a single prompt-based safety protection layer, ATTCK-R consistently outperforms strong recent multi-turn jailbreaking frameworks. Ablation results confirm that long-horizon planning and tactic curation improve attack effectiveness, while adaptive traceback improves efficiency through structured recovery from conversational failure. 

---
# REVEAL: Reverse Engineering Agentic Latents from Frontier-Model Traces 

**Authors**: Yuetian Du, Meng Wang, Yucheng Wang, Zhangzhe Chen, Wei Wang, Boyu Yang, Weiqi Zhai, Bing Zhao, Ming Kong, Zhan Zhou, Jie Liu, Qiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.16288)  

**Abstract**: Frontier LLM agents perform complex interactive tasks, but their inference costs motivate transferring their experience to smaller open-source agents. Distillation and latent communication typically require target-model training or access to source-model internals. We study how textual frontier-model trajectories can provide reusable local controls for a frozen agent. Inspired by protocol reverse engineering, we introduce $\textit{Agentic Latents}$, comprising anchored states and dynamic profiles of feedback--response cycles elicited by replay in the target model. Our training-free framework, $\textbf{REVEAL}$ ($\textbf{REV}$erse $\textbf{E}$ngineering $\textbf{A}$gentic $\textbf{L}$atents), uses these profiles and trajectory outcomes to match and select local events from successful and failed rollouts of the same task and source model. Their response-transition contrasts yield a target-specific steering direction, applied before each response. Within-benchmark evaluations on Terminal-Bench 2.1 and Humanity's Last Exam with tools report higher accuracy than unmodified agents across three targets. Representation analyses and behavioral case studies support reusing frontier-model experience as local activation control. 

---
# RelLedger: Relational Information Governance for LLM Agent Organizations 

**Authors**: Junxi Huang, Wenshuo Chen, Nuoxuan Li, Kailong Wang, Meitong Shen, Yuehan Kou, Leihao Zhao, Yutao Yue, Xiucheng Wang, Nan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.16281)  

**Abstract**: In LLM agent organizations, shared records determine which facts are available for later decisions. A sender consults interaction histories before deciding whether to report an event. A delivery gap across histories does not show whose relationship drove the choice or whether the resulting record matters downstream. We introduce RelLedger to audit these links. It holds the event, evidence, risk, and message fixed while crossing the sender's histories with the event target and an unrelated actor. Role reassignment and history rewrites locate the controlling cue. Message and cost interventions show when the effect appears. An attribution ledger maps each claim to its required response cells and separates missing interventions from sampling uncertainty. Recipient replay follows delivered information into downstream decisions. Executable workflows for a simulated one-person company extend the audit to retrieved histories, tool calls, and shared ledger updates. Across GPT, Gemini, Claude, DeepSeek, and Qwen, the audit reveals model-specific control paths and smaller relation gaps under explicit loss rules. Claude sends the same factual notice 62.22 percentage points more often with conflict histories than with friendly histories. In executable workflows, strained histories reduce recipient review rates by 45.83 points relative to warm histories for Claude. 

---
# agent-risk: Risk Assessment for Changes to AI Agents 

**Authors**: Yue Zhao, Zixiang Xu, Angela Huyue Zhang, S. Alex Yang, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.16280)  

**Abstract**: Before changing a deployed AI agent (a stricter safety gate, a new model), organizations must approve the change, starting from existing logs. Extending classical risk analysis to partial records and partial changes to running systems, agent-risk replays logged episodes, assuming nothing else changes and the agent does not anticipate the change. The log settles episodes acting as before, a declared stop value settles stopped ones, and the rest need logged support, a saved-state rerun, or a bound. We prove what logs must record for exact replay and which saved states serve a gate family. For independent episodes, we bound the reruns a decision needs and tolerable drift from live behavior. In a bash sandbox, a 7,200-episode log alone decided neither of two retry gates, which let the agent act after refusals the log never shows. Rerunning only each gate's blocked episodes from saved states reached both live decisions at half the agent calls of a live test sized post hoc for them, given the log. Without per-step harm, a stopping gate's certified harm ceiling rose up to 2.5-fold, since the log hides whether harm preceded the stop. Approval thus needs evidence designed in advance: log fields, saved states, targeted reruns. 

---
# Error Correction in Trained TTT Layers 

**Authors**: Yuhe Hu, Wen Jia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.16268)  

**Abstract**: Test-time-training (TTT) layers update fast-weight memory while processing an input sequence. A gradient-based update, however, does not by itself establish how the memory revises an existing association. We study the structure and empirical role of prediction-derived updates in released TTT implementations. For dot-product objectives, we characterize the absence of a prediction-error residual and establish conditions under which inner-loop ascent and descent are equivalent after reparameterization. For the LayerNorm-based squared-error objective used by TTT-Linear and TTT-MLP, we derive the inner gradient and identify a prediction-derived component that cancels at the released initialization in the zero-epsilon limit. We then measure the full normalized prediction's contribution on three trained checkpoints. Removing this contribution while retaining the LayerNorm Jacobian gives a mean gradient norm ratio of 0.25 to 0.32 across the models. On 100,303 tokens from War and Peace, the same intervention with local per-token norm matching raises perplexity by 2.8-2.9% on TTT-Linear-760M and 1.3B, whereas freezing all fast-weight updates raises it by factors of 4.2-4.5. Separate synthetic tests illustrate the dependence of plain delta-rule overwriting on step size. The results distinguish a prediction-derived update from its behavioral value and show why implementation analysis and live interventions are both needed to assess test-time memory. 

---
# Hashrope: A Persistent Data Structure for Mechanisms of LLM Context Management 

**Authors**: Muntaser Syed, Ameerah Alsulami, Marius Silaghi  

**Link**: [PDF](https://arxiv.org/pdf/2610.16144)  

**Abstract**: Serving stacks for large language models manage context through four recurring mechanisms: incremental edits to a growing context, branching and snapshotting for tree search, identification of reusable prefixes for KV-cache reuse, and encoding of repeated segments. Each mechanism is served today by a different specialist structure. We show that all four reduce to a single persistent weight-balanced rope whose every node carries a polynomial hash of its subtree over a Mersenne prime, so that the structure supports logarithmic edits and logarithmic verified content-identity queries at the same time. We validate each mechanism with pre-registered, replicated experiments, then compare against a deployed specialist per mechanism: the Ropey editing library, a faithful PagedAttention copy-on-write block table, and the vendored SGLang radix matcher. On workloads that interleave edits with identity queries, Hashrope is net-superior to Ropey by a factor that grows without bound (369.7x at one million characters). Against the radix matcher on a live vLLM stack serving Qwen2.5-7B-Instruct-1M on four A100 GPUs, Hashrope identifies a one-million-token shared prefix exactly and 1.20x faster, at a context length the dense engine cannot even prefill. 

---
# Phantom Merge: When Your Large Language Model Agents Pick One but Tell You About Another 

**Authors**: Wangshu Zhu, Qianwen Guo, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2610.16141)  

**Abstract**: LLM agents increasingly operate over multiple entities, where reliable response generation requires not only retrieving relevant evidence but also preserving its association with the correct entity. We identify Phantom Merge (PM), an entity-evidence binding failure in which an agent attributes unsupported claims to its committed entity, potentially even when the task is completed correctly. We formalize PM at the claim level and characterize its prevalence and error subtypes across four agent domains and four backbone families. To address this problem, we propose a two-stage framework for claim-level risk estimation and correction. First, Anchor-Grounded Risk (AGR) combines a representational readout of the agent's pre-emission hidden state with Jacobian-based functional information, providing complementary slot-level and anchor-value risk scores. Second, Fixed-Anchor Correction (FAC) audits flagged claims and selectively revises them using observed anchor evidence or removes them when no supported replacement is available. On the held-out Shopping detection cohort, AGR-slot achieves an AUROC of 96.2% and an average precision of 97.3%, outperforming the evaluated detection baselines. In the end-to-end Shopping mitigation evaluation, FAC reduces trajectory-level PM from 51.4% to 6.2%, with an output-to-original claim ratio of 80.7%. These results demonstrate the prevalence of entity-specific grounding failures and the effectiveness of combining internal risk estimation with evidence-constrained claim correction. Our code is publicly available at this https URL. 

---
# PID-Gated Continual Prompt Adaptation Under Noisy Feedback 

**Authors**: Zhentao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.16129)  

**Abstract**: Deploying LLM agents in non-stationary environments requires continually revising their instructions as the task drifts, yet rewriting after every reported mistake can turn noisy feedback into persistent errors. This raises an often-overlooked scheduling question in prompt adaptation: when should feedback trigger a lasting instruction update, and at what editor-call cost? Taking inspiration from proportional-integral-derivative (PID) control, we introduce a family of lightweight gates that react to the current error, accumulate evidence over time, and roll back deterioration after an edit. We benchmark these gates against both simple baselines and advanced prompt optimizers (TextGrad and GEPA), over a diverse suite of drifting tasks, LLM parameter sizes, feedback-noise levels, and random seeds. PID-gated continual prompt adaptation yields a more favorable Pareto trade-off between task performance (high accuracy, low error) and editor-call budget than the baselines considered, pointing to control-theoretic scheduling as a simple, general tool for robust prompt adaptation under unreliable feedback. 

---
# GUIDE: Computationally Augmented Consensus for Trustworthy Clinical Practice Guidelines 

**Authors**: Yuanfan Lin, Jie Di, Xujin He, Zhenzhe Lin, Jialing Chen, Ruiyang Li, Changyu Zheng, Yuanrui Shang, Weiqin Peng, Jingxi Qin, Xiong Wang, Dongxin Ye, Yining Huang, Guoqiang Li, Andi Xu, Sijia Yu, Ruixin Wang, Yahan Yang, Xiaohang Wu, Duoru Lin, Jingjing Chen, Zhenzhen Liu, Wenben Chen, Haotian Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.16108)  

**Abstract**: Maintaining up-to-date clinical practice guidelines (CPGs) is essential for evidence-based medicine, but rapid evidence growth and complex expert consensus formation limit timely updates. We developed GUIDE, a multi-agent copilot system for human-AI collaborative guideline generation and updating. GUIDE delegates labor-intensive tasks to specialized AI agents while preserving expert judgment and final arbitration. In 12 NICE ophthalmology datasets, GUIDE achieved 88.85% literature recall, 86.06% alignment in certainty-of-evidence ratings, and 91.67% recommendation agreement, requiring 162.17 minutes per dataset on average (28.68% human intervention). Performance remained stable across 20 additional datasets from cardiology, endocrinology, gastroenterology, and respiratory medicine. Ablation analysis showed that removing expert intervention reduced performance in judgment-intensive steps such as Risk-of-Bias assessment. In six post-cutoff datasets, GUIDE maintained robust end-to-end performance on unseen evidence. Prospective real-world validation showed that GUIDE reduced expert workload and produced more clinically acceptable recommendations than the original decade-old guideline (mean score, 3.73 vs. 1.53; agreement, 66.7% vs. 13.3%), with outcomes comparable to human-only updating (P = 0.588) but substantially less time and effort. GUIDE offers a practical framework for timely, trustworthy, expert-centric guideline development and validates computationally augmented consensus. 

---
# Dream2Discern: Learning from Hard Latent Variants for Runtime Failure Prediction in Multi-Agent Systems 

**Authors**: Boxuan Zhang, jianing Zhu, Xiaotian Han, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2610.16071)  

**Abstract**: Reliable deployment of LLM-based multi-agent systems calls for runtime auditors that predict terminal outcomes from observed execution prefixes. These auditors typically learn from a limited corpus of trajectories with verified outcomes. Expanding this corpus requires costly execution and verification, yet broader failure coverage alone provides no explicit criterion for identifying prefixes that the current auditor finds difficult to classify. To this end, we propose Dream2Discern, a runtime auditor trained through implicit augmentation guided by its own prediction loss. During Dream, we select perturbations of the auditor's representation operators to increase prediction loss on recorded prefixes, yielding hard latent variants. During Discern, the auditor learns to predict execution outcomes from these variants using the original labels. The variants are regenerated as the auditor learns, turning a limited corpus into a renewable source of hard examples and requiring no additional trajectory collection or outcome verification. Extensive evaluations on AFTraj-2K and StepShield show that Dream2Discern outperforms competitive failure auditors and achieves over 90% trajectory-level weighted accuracy on both benchmarks while balancing prediction accuracy and timeliness. Project page: this https URL 

---
# When Agents Talk, Secrets Travel: Topology-Driven Leakage and Recipient-Conditioned Sanitization in Multi-Agent LLM Systems 

**Authors**: Sunday Oyinlola Ogundoyin, Muhammad Ikram  

**Link**: [PDF](https://arxiv.org/pdf/2610.16039)  

**Abstract**: Multi-agent LLM systems (MALS) introduce a privacy-leakage problem that conventional sensitivity-based defenses miss, since information disclosed to one authorized agent can propagate through routine coordination to an unauthorized one, without compromise or malicious behavior. This motivates topology-aware enforcement that blocks unauthorized propagation while preserving authorized disclosure. We evaluate 485 scenarios across coordinator-worker, peer-to-peer, and hierarchical topologies, instrumenting seven communication channels and measuring surface leakage and disclosure to topology-introduced recipients. We introduce RECONS, a Detect -> Decide -> Transform sanitizer conditioning intervention on recipient and execution context rather than sensitivity alone. Topology changes where protected information propagates, with disclosure to topology-introduced recipients rising from 0% to 52.2% while surface leakage stays nearly flat at 60.6-62.8%. This effect persists across four generator models and a second orchestration framework, though magnitude and relay paths differ; framework choice separately drives a 150x leakage spread, so topology alone is not sufficient. RECONS reduces residual disclosure from 96.7% without sanitization to 29.25%, preserving 94.1% of authorized disclosures versus 20.5% under blanket redaction. Detection coverage is the dominant bottleneck, accounting for 95.5% of residual disclosures. These results show that privacy enforcement in MALS must reason about the recipient and communication path realized at runtime, not sensitivity or topology assumptions alone. 

---
# Scaling Repeated-Data Pretraining: Batch Size, Regularization, and GPU-Time Tradeoffs 

**Authors**: Sewon Min  

**Link**: [PDF](https://arxiv.org/pdf/2610.15965)  

**Abstract**: We study pretraining when compute is abundant but unique training data are finite, requiring repeated training on the same data. We first revisit infinite-compute pretraining (Kim et al. 2026) and show that its frontier depends critically on choices previously fixed or studied only in isolation. In particular, optimizing batch size together with epoch counts reveals a U-shaped relationship between validation loss and batch size, favoring smaller batches than typical pretraining, likely because small-batch gradient noise provides implicit regularization. We further find that training recipes, previously optimized under a single-pass assumption, can be tuned, e.g., repacking documents into new sequences every epoch, weight tying, and positive embedding weight decay.
We then relax the infinite-compute assumption. In particular, smaller-than-typical batch sizes can require prohibitively long sequential training, motivating to treat GPUs and time as separate resources. Tighter deadlines favor larger batches, and latency-oriented parallelism can trade additional GPUs for shorter training. Nevertheless, even under a fixed time budget, a smaller batch with fewer GPUs can achieve lower validation loss than a larger batch with more GPUs, challenging the common practice of increasing data parallelism to maximize GPU utilization under a deadline. Together, our results identify underexplored factors in repeated-data training and provide concrete guidance on how training recipes and resource allocation should change from single-pass pretraining. 

---
# The Anatomy of a Blind Spot: What an LLM Judge Must Be Shown, and How Much 

**Authors**: Nitish Kumar, Prateik Pothuneedi, Greg Bandak  

**Link**: [PDF](https://arxiv.org/pdf/2610.15938)  

**Abstract**: Evaluation of text generated by large language models (LLMs) is increasingly delegated to LLMs acting as judges, yet what such a judge must be shown remains unresolved. We treat a judge whose reports are acted on as a regulator, in the sense of cybernetics, and ask what it must be shown, how much, in what form, and what its reports must get right. On seven public benchmarks we varied only what the judge was shown. Across three judges whose prices differ 10- to 17-fold, what the judge was shown mattered more than which judge it was. When a system revised its answers on feedback from a judge denied the material they had to respect, it ended below its starting point on instruction following and question answering. On instruction following the loss grew with each round, and it persisted when the judge saw the user's request but not the constraints and with a second judge family. On news summarisation, a correct reference summary beside the source undid most of the source's contribution, whereas on single-answer questions a reference helped. About a tenth of the source did almost all of its work, and a small entailment graph matched an LLM judge at finding errors in grounded text. We committed 83 testable clauses before running; of the 81 that ran, 32 met their bar and 50 went in the predicted direction, and most quantitative predictions from the theory failed. We report every shortfall and derive design rules tied to measured effects. 

---
# Personalized LLM Recommendation with Semantic IDs: Autoregressive Beam Search vs. Multi-Token Prediction 

**Authors**: Sejoon Oh, Fernando Amat Gil, Evan Smothers, Dan Iter, Mark Thornburg, Dawit Mureja Argaw, Sourabh Medapati, Thea Wang, Ria Verma, Qiuling Xu, Harshit Syal, Ehsan Gholami, Moumita Bhattacharya, Ashish Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2610.15864)  

**Abstract**: Large-scale personalization is shifting from embedding-table retrieval to LLM-based recommenders that generate the Semantic ID (SID) of the item to recommend, where each SID is a short sequence of discrete hierarchical content codes. A single instruction-tuned model can then unify retrieval, content understanding, and language, and recommend brand-new titles from their content alone using SID; then, how should the model generate the SID? Traditionally, models rely on a single language modeling head that generates tokens autoregressively, typically using constrained beam search. In contrast, an emerging body of work explores multi-token prediction (MTP), where dedicated heads predict all tokens in parallel, substantially reducing the decoding latency associated with beam search. We report an industry-scale, apples-to-apples comparison of these two generation paradigms: single-head autoregressive decoding vs. MTP. We introduce a new auxiliary ranking loss, defined over the joint SID score, to train the MTP objective. We evaluate the SID training objectives on both quality and training/inference cost, inside a multi-task LLM that jointly does personalized retrieval, content question-answering (QA), and instruction following over a large streaming-media catalog. Our experiments show single-head beam search consistently beats MTP on most product-critical metrics: ranking and content QA, and trains at roughly 2.6x lower dollar cost on Qwen3-8B scale; MTP wins only on raw inference throughput. Around this finding, we detail an efficient training and inference stack and report end-to-end gains over the previous non-SID baseline equipped with an auxiliary ranking head: +77% item cold-start ranking, +22% content-QA accuracy, +2.8x faster training, with language metric regression traced to SID vocabulary expansion. We address the language regression by reinforcement learning post training. 

---
# What Do Sparse Autoencoder Feature Discovery Scores Identify? From Activations to Interventions 

**Authors**: Yifeng Guo, Yanran Li, Xiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.15850)  

**Abstract**: Sparse autoencoder (SAE) features are typically discovered from how their activations co-vary with behaviour, yet they are often used through interventions such as steering or ablation. We take intervention attribution, the first-order effect of a specified intervention on the model output, as our estimand. For a broad class of response functions, we show that observational feature-discovery evidence generally does not identify this attribution, and derive an exact gradient-based decomposition of the gap between linear-probe predictions and intervention attribution. Across four model-SAE pairs, median probe-to-attribution magnitude ratios range from tens to hundreds, while attributions closely track small-step steering. For single-latent ablations with nonzero effects, gradient predictions match the realized sign in 91-97% of cases, compared with 17-35% for probe-implied changes. Attribution-selected sets also produce larger absolute mean ablation effects on held-out prompts than sets selected by correlation, rank tests or probe coefficients. Together, these results establish intervention attribution as a principled, gradient-estimable target for selecting SAE features for intervention. 

---
# KhyontekAI at CheckThat! 2026: Multilingual Dense Retrieval and Cross-Encoder Reranking for Scientific Web Claims 

**Authors**: Bhargab Jyoti Bhuyan, Pritam Deka  

**Link**: [PDF](https://arxiv.org/pdf/2610.15837)  

**Abstract**: Scientific claims shared through social media platforms frequently reference research findings without explicitly linking to the corresponding scientific publications. Automatically retrieving the correct scientific source for such claims is a challenging multilingual retrieval problem involving semantic matching, paraphrasing, noisy social media text, and cross-domain language variation. In this work, we present a multilingual two-stage retrieval framework for CLEF 2026 CheckThat! Task 1 on Source Retrieval for Scientific Web Claims. Our system combines fine-tuned dense retrieval models with cross-encoder reranking for improving multilingual scientific document retrieval. We evaluate multiple transformer-based dense retrieval architectures including multilingual-E5, BGE, and Contriever under different metadata configurations. We further investigate multilingual joint training, retrieval depth analysis, metadata enrichment, and language balancing strategies for improving retrieval effectiveness. Experimental results demonstrate that metadata enrichment consistently improves retrieval effectiveness, while fine-tuned cross-encoder reranking provides additional gains in retrieval precision. Our final multilingual system achieves MRR@5 scores of 0.6222 for English, 0.5112 for German, and 0.6254 for French, with an overall average MRR@5 score of 0.5862. 

---
# Local Citation Recommendation Without Graph Signals 

**Authors**: Benjamin Basseri, Pavlos Protopapas, Kemeng Zhang, Alberto Accomazzi, Stephanie Jarmak, Anna Kelbert  

**Link**: [PDF](https://arxiv.org/pdf/2610.15832)  

**Abstract**: Resolving correct citations can be a time-consuming task for research writers. We address local citation recommendation (LCR), where the goal is to retrieve appropriate references for a given sentence. Web tools generally depend on the existing citation graph, which may bias results toward citation count or more recent publications. LCR research using full paper texts has been limited by the lack of publicly available data. We present a new dataset, the Local Citation Recommendation Corpus (LCRC), containing 14,735 scientific statements derived from astrophysics review papers, each with one or more citation targets whose full text is accessible via API. Using this dataset we systematically compare retrieval methods and develop a citation recommendation system that avoids citation graph bias while achieving 8.0% improvement in Recall@25 and 6.9% improvement in NDCG@25 over dense retrieval baselines without model fine-tuning. To support reproducibility, we make our code and dataset publicly available. 

---
# Adaptive Look, Flexible Thought: Learning How and How Long to Think in Multimodal Reasoning 

**Authors**: Quanlin Chen, Xiuwei Chen, Kun Xiang, Zehua Ma, Chunjin Ye, Boqiang Guo, Xianming Liu, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2610.15810)  

**Abstract**: The introduction of long chain-of-thought (CoT) reasoning and external tools has substantially improved the performance of multimodal large language models (MLLMs) on general and reasoning tasks. However, existing fast-slow reasoning paradigms mainly regulate textual reasoning length, while visual computation introduced by external tools or visual tokens is often treated as an auxiliary component. This separation prevents MLLMs from jointly optimizing visual perception and textual reasoning. To address this issue, we propose L2T (adaptive Look, flexible Thought), a unified reasoning framework that jointly optimizes the form and depth of computation. L2T first discretizes tool-produced visual states into visual tokens, bringing visual perception and textual reasoning into a shared autoregressive space. Based on this unified representation, L2T introduces a modality-aware routing reward to select appropriate reasoning modes and a difficulty-aware fast-slow strategy to adapt reasoning depth according to task complexity. Experiments on general perception and reasoning-intensive tasks demonstrate the effectiveness of L2T. Compared with explicit tool-calling approaches, L2T maintains competitive performance while achieving 1.7--3.6 times higher throughput. 

---
# Agent Sociodrama: Social-Dynamics-Guided Persona Recasting for LLM Agent Teams 

**Authors**: Yirui Qi, Xiaoming Zhang, Jaehong Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2610.15788)  

**Abstract**: Large language model (LLM) agents are increasingly deployed as teams, yet they fail in characteristically social ways: information holders stay silent, dominant speakers anchor groups on wrong answers, and norm violations go unsanctioned. We present Agent Sociodrama, a framework for modeling and validating evolving social dynamics in LLM agent teams. Under fixed personas, agents' emotional states and directed, asymmetric relationships are updated every round by a unified external LLM judge and shape their subsequent interactions. Across six LLM backbones and five social scenarios, we assess social fidelity through classical social-theory patterns and agreement with human judgments. The resulting trajectories exhibit structured social patterns and model-specific differences. We then introduce Trajectory-Guided Persona Recasting (TPR), which diagnoses group-dynamic failures from interaction transcripts and social trajectories and recasts agent personas before rerunning the task. Across four collaborative benchmarks, TPR improves performance from both heterogeneous and randomly initialized agent persona compositions, yielding up to 83.3%p of accuracy improvement over the initial casting. On representative distributed-information cooperation and mixed-motive bargaining tasks, we outperform SOTOPIA and AgentVerse in three of four comparisons, by up to 37.5%p in Pareto rate. These results show that validated social trajectories provide actionable signals for repairing LLM agent teams. 

---
# Heterogeneous Structured Pruning Under Resource Constraints for Hardware-Efficient Large Language Models 

**Authors**: Cameron Barker, Andreas Paraskeva, Jan N. van Rijn, Elliot J. Crowley, Henry Gouk  

**Link**: [PDF](https://arxiv.org/pdf/2610.15781)  

**Abstract**: Training-free pruning removes weights from pretrained large language models to reduce their deployment cost. However, existing approaches typically consider only the total number of retained weights. Consequently, their pruning decisions do not account for how target hardware stores and processes those weights. These methods select a sparsity structure in advance and apply it uniformly to all projections, rather than accounting for how each structure trades pruning error against latency and memory across sublayers. We propose Heterogeneous Structured Pruning (HSP), a hardware-aware pruning framework that combines two components: (i) a heterogeneous search space of dense, unstructured, 2:4, block-sparse, and head / channel (H/C) pruning options and (ii) structure-regularising constraints based on latency and memory models. Our contribution generalises existing importance-based methods: a global solver selects one option per sublayer to minimise proxy pruning loss under a resource budget, thereby assigning each projection a structure and density. We evaluate HSP on OPT and LLaMA models for an NVIDIA Jetson Orin Nano Super edge device, where it improves the quality-resource frontiers over training-free baselines.
A project page and updated links for the related code is maintained at: this http URL. 

---
# Who Sets the Pace of Judge Hacking? Judge Capability versus the Policy-Judge Gap 

**Authors**: Haosen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.15778)  

**Abstract**: When a policy is trained with GRPO against a frozen, reference-free LLM judge, the judge accepts more of its answers, but in a post-hoc analysis we detect no gain in held-out greedy accuracy: in all 12 (policy, judge) cells, the seed-mean gain never exceeded 0.04. A gold-reward control with the same recipe raised the 1.5B policy's greedy accuracy by +0.054 [0.013, 0.096], so for this policy the flat accuracy is specific to the judge reward; for the 0.5B policy, gold reward also gave no detectable gain. In the four rising cells we checked, the early rise in training-rollout accuracy is mostly sampling sharpening, as it is under gold reward. We then ask whether the policy-judge gap or the judge sets how far and how fast RL exploits the judge's false positives. We train 0.5B and 1.5B Qwen2.5 policies against four judges at least as large (Qwen2.5-1.5B/3B/7B, Llama-3.2-3B) and the 1.5B policy against one smaller judge (Qwen3-0.6B). The depth of judge hacking (the plateau) tracks the judge: it correlates with judge AUC (Spearman -0.69 [-0.94, -0.14] over 12 cells), and within a family the larger judge is fooled less deeply. For the pace (onset), two pre-specified pairs contradict the gap account (the larger-gap cell was fooled 3.5x sooner at the pre-specified threshold). With 12 cells we cannot formally separate judge AUC from the gap for either endpoint. The smaller judge was not fooled sooner than a judge as large as the policy, though it also differs in family and template. In a blind test on three held-out judges, a hash-locked step-0 predictor did not beat step-0 AUC, but three cells cannot separate them. At 7-8B, judges from three families plateaued at 0.37, 0.67 and 0.85. 

---
# Preserving Updates in Low-Bit Recurrent State Quantization 

**Authors**: Haotian Li, Liran Dong, Kaicheng Xiao, Guoliang Xing  

**Link**: [PDF](https://arxiv.org/pdf/2610.15709)  

**Abstract**: Linear attention and recurrent models compress past tokens into a fixed-size state that is updated at every step. However, storing these states in high precision incurs substantial memory and bandwidth costs at large batch sizes, motivating low-bit state quantization. Unlike weights and key--value caches, a recurrent state is requantized after every update, so its quantization errors accumulate across steps. Our analysis shows that these errors are highly coherent across steps, so reducing per-step errors alone is insufficient. This coherence arises mainly from small updates that are lost in rounding. To preserve these updates, we intervene both before and at rounding. Before rounding, we propose Low-Rank Bypass (LRB), which accumulates small updates in a compact low-rank buffer and periodically merges its sufficiently large elements into the state. At rounding, we observe that a state element can even move opposite to its update, and such reversals prove particularly harmful. Therefore, we propose Bounded Directional Rounding (BDR), which rounds reversed elements toward their updates without additional storage. Although BDR enlarges the rounding error, it improves the model, suggesting that the direction of an update matters more than its rounding error at a single step. Across Gated DeltaNet, GLA, Mamba2, and RetNet, our training-free method substantially outperforms INT4, while reducing state memory by up to 5.3$\times$ and speeding up decoding by up to 1.9$\times$ relative to FP32 states. 

---
# Generalizable Model Editing Could Be More Than Direct Generalization 

**Authors**: Xiaotian Ye, Chunyao Yang, Xiaohan Wang, Junkai Chen, Yongqi Duan, Mengqi Zhang, Shu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.15707)  

**Abstract**: A central desideratum in model editing is generalization to complex tasks, which prior work formalizes under a direct generalization paradigm: treating and evaluating LLM behavior as an input-output mapping and expecting a factual update to alter outputs on reasoning tasks. Yet despite substantial empirical effort, the field consensus remains that such generalization is highly challenging and elusive. However, when we deploy the latest models in an in-the-wild chatbot-user setting, we surprisingly find that even classical editing methods exhibit remarkably fluent generalization behavior. This motivates us to revisit the obstacles to generalization and the limitations of the prevailing paradigm. We find that direct generalization is fundamentally constrained both theoretically and practically: theoretically, it suffers from an identifiability issue; and practically, existing approaches attempting to circumvent this are further hindered by the ambiguity of their constructed biases and by vulnerability to being overridden by other factors such as optimization dynamics. We instead identify compositional generalization, empowered by the increasingly strong reasoning and composition abilities of modern LLMs, as a theoretically grounded and empirically validated alternative. We validate this phenomenon across multiple models and editing methods, showing that such generalization behavior is widespread. Overall, we argue that generalizable model editing could be more than direct generalization, and call for a paradigm shift toward more generalizable and practically deployable knowledge update techniques. 

---
# A Persian Vision-Language-Action Dataset and Benchmark for Ambiguity- and Infeasibility-Aware Instruction-to-Plan Grounding 

**Authors**: Abdollah Pourasghari, Saeedeh Sadat Sadidpour, Mohammad Roustaei  

**Link**: [PDF](https://arxiv.org/pdf/2610.15666)  

**Abstract**: Converting natural-language instructions into executable action plans conditioned on a scene image is a core capability for embodied human-robot interaction, but existing benchmarks target English, simulated or video-grounded environments, and largely assume instructions are clear and executable. We introduce the first Persian vision-language-action dataset for this task: 258 instances over 230 real household/office photographs, annotated with reference plans, deliberate ambiguity and infeasibility labels, disambiguating question-answer pairs, and an auto-generated chain-of-thought rationale for training. Using this resource, we ablate retrieval-augmentation, chain-of-thought prompting, and an agentic disambiguation step, evaluated with an LLM-as-judge protocol and a new metric, PGOS-FA, extending prior work (PG2S) with step-order sensitivity, an over-generation penalty, and Persian-adapted embeddings. Paired statistical testing across 36 comparisons shows retrieval-augmentation is a large, robustly significant effect, while finer-grained additions are directionally consistent but not statistically distinguishable from noise at this scale, and the LLM-judge metric has near-zero power where PGOS-FA succeeds. A blind human-correlation study on 79 real outputs shows PGOS-FA tracks human judgment at least as closely as the LLM-judge. Testing generalization beyond the ablation's single backbone, we find that most of the gap between an open-weight and a substantially stronger closed planner is attributable to open-weight model generation rather than to the closed model's capability specifically, and that our best configuration significantly outperforms an independently reproduced external baseline despite a weaker backbone. The dataset, PGOS-FA implementation, and evaluation code will be made available upon reasonable request. 

---
# The Gradient That Top-k Distillation Never Computes 

**Authors**: Rasul Khanbayov, Hasan Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2610.15589)  

**Abstract**: Many deployed distillation pipelines show the student a truncated view of the teacher: keep the top-k probabilities, renormalize, and minimize a divergence on what remains. Renormalization is assumed to account for the missing mass. We prove that it removes the visibility of tail mass, not the freedom to move it. The renormalized objective depends only on log-odds among the retained tokens, so at each position its gradient along the directions that control tail mass is exactly zero; on a real teacher that component measures $2$-$3\times10^{-15}$ of the gradient. The full-vocabulary gradient equals the truncated gradient plus a residual weighted by the tail masses, and training acts only on the first term. Without the cross-entropy term, a top-k student's tail mass drifts away from the teacher's while a full-vocabulary control moves slightly toward it; with it, the gap shrinks about fourfold but persists. On seven open-weight teachers, the retained set determines the teacher's distribution only from $k=d+1$ ($d$ the hidden dimension): recovery is exact from there on all seven, and commonly cached sizes sit below it. In our off-policy runs the practical effect is real but smaller than the choice of teacher-student pair. Renormalization is a normalization, not a safeguard. The code is available at this https URL. 

---
# SARATHI: Decoupled Data-Free Probing and Adaptive Reconstruction for Structured LLM Pruning 

**Authors**: Sankar Behera, Dhruv Singh, Satyadev Ahlawat, Yamuna Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2610.15514)  

**Abstract**: Structured pruning reduces the inference cost of large language models (LLMs) by permanently removing neurons and weight rows, yielding dense, hardware-agnostic speedup. Existing post-training methods couple neuron selection and weight reconstruction through a shared calibration dataset, introducing domain bias and out-of-memory failures at high sparsity. We propose SARATHI, a framework that decouples these two stages. A data-free \emph{NMF Residual Subspace Probe} scores neurons using intrinsic weight geometry, without accessing any calibration data, and carries a formal connection to Robust PCA. \emph{Adaptive OBS} then reconstructs the pruned weights with a localized calibration batch, operating in $\mathcal{O}(d^2)$ peak memory per layer and eliminating the global memory bottleneck of coupled methods. A \emph{Bias Shift Compensation} step corrects systematic activation drift in post-LayerNorm architectures that would otherwise cause perplexity collapse at high sparsity. Experiments across six models from two architecture families (gated and non-gated FFN) at 15--40\% sparsity show that SARATHI achieves competitive perplexity and matches or exceeds existing structured pruning baselines on zero-shot accuracy. 

---
# Hidden in Plain Sight: Advancing Harmful Content Understanding via Perception-Aware Adversarial Self-Play 

**Authors**: Yisong Xiao, Benlei Cui, Shiji Zhao, Zonghao Ying, Yaoyuan Zhang, Longtao Huang, Aishan Liu, Hui Xue, Ting Ma, Haiwen Hong, Xianglong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.15449)  

**Abstract**: Despite significant advances in language understanding, Large Language Models (LLMs) can still miss meaning hidden in plain sight, failing to identify harmful intent in expressions readily understood by humans. Malicious users exploit this gap by strategically disguising harmful content to evade automated moderation, posing a serious threat to online platform safety. The vulnerability reflects a fundamental cognitive asymmetry between human readers, who integrate contextual, phonetic, and visual cues to decipher the intent behind deliberate obfuscation, and text-based LLMs, which rely primarily on token sequences and have limited exposure to evolving evasions. To bridge this asymmetry, we propose PASP (Perception-Aware Adversarial Self-Play), which combines multi-perceptual views with adversarial self-play to improve understanding of harmful intent as evasions evolve. Specifically, PASP constructs textual, phonetic, and visual views of each input, preserving its original context while exposing pronunciation and glyph cues obscured in token space. During adversarial self-play, an attacker uses defender feedback to generate intent-preserving evasions, while the defender is trained using an intent-invariance reward to understand harmful intent across changing surface forms. Extensive experiments show that PASP reduces adaptive attack success rates by up to 73.71% and 74.49% on ChineseHarm-Bench and EvoHarmBench, respectively, preserves model capabilities with minimal degradation, and effectively interprets real-world evasive expressions in e-commerce. 

---
# Exploring Jailbreak Vulnerability Across Depth-Based Dynamic Inference in Large Language Models 

**Authors**: Lass Najmuldeen, Seyedhamidreza Mousavi  

**Link**: [PDF](https://arxiv.org/pdf/2610.15207)  

**Abstract**: Large Language Models (LLMs) have been shown to be computationally expensive and vulnerable to jailbreak attacks, in which adversarial prompts bypass safety alignment and induce harmful responses. To address this inefficiency, a range of techniques have been proposed to enable efficient LLM inference. Depth-based dynamic inference has demonstrated promising results by enabling outputs to be generated from intermediate layers. It offers the flexibility to adjust model behavior based on the available resource budget. However, how this flexibility affects the robustness of LLMs against jailbreak attacks remains unexplored. In this paper, we address this gap via a comprehensive investigation into the impact of depth-based dynamic inference on the robustness of LLMs against jailbreak attacks. Specifically, we make the following key observations: 1) Shallow exits are consistently more robust to jailbreak attacks than deeper ones; 2) Exit depth affects how easily the adversarial suffix can be optimized, with deeper exits reaching lower attack loss; and 3) Adversarial suffixes affect utility uniformly across exit depths. These insights can help design dynamic inference methodologies that are more robust to jailbreak attacks. Code is available at this https URL 

---
# Word-Aligned Prosody Reflects Speaker Differences More Than Depression or PTSD Severity 

**Authors**: Hsiang-Chen Yeh, Aurosweta Mahapatra, Shreeram Suresh Chandra, Ryan L. Boyd, Berrak Sisman  

**Link**: [PDF](https://arxiv.org/pdf/2610.15181)  

**Abstract**: Word-aligned prosody is increasingly used to detect depression and posttraumatic stress disorder (PTSD), yet the stability and clinical relevance of these measurements remain unclear. We examine stability at two levels. At the word level, we ask whether a speaker produces the same word in a similar speaker-specific way across different utterance contexts. At the speaker level, we ask whether averaging prosody across many words produces consistent speaker scores across separate recordings. Drawing on principles of psychological measurement, we compare these representations in four corpora spanning acted, spontaneous, and clinical speech. Word-specific deviations show little cross-context recurrence (0.01-0.16), whereas speaker summaries are highly consistent, reaching 0.997 in ESD, 0.73-0.90 in MSP-Podcast, and 0.84-0.996 in DAIC-WOZ and PTSD-STOP. Despite this consistency, associations with depression or PTSD severity do not exceed 0.15; equivalence tests exclude effects of 0.30 in all analyses and 0.20 in DAIC-WOZ, while within-person associations with daily PTSD symptoms are near zero ($|\rho|\leq0.04$). Thus, aggregation yields stable speaker-level prosodic measures, but these measures do not detectably track depression or PTSD severity in the studied corpora. Code is available at this https URL. 

---
# Compute-Optimal Scaling Laws for Spiking Language Models 

**Authors**: Dan Saunders, Benjamin Ren, Catherine He  

**Link**: [PDF](https://arxiv.org/pdf/2610.15146)  

**Abstract**: Spiking neural networks target energy-efficient language modeling through sparse, binary activations, but the efficiency-accuracy tradeoff of spiking language models is uncharacterized at scale. We present the first Chinchilla-style IsoFLOP scaling study of a spiking language model, sweeping SpikeGPT-family models (RWKV backbone) from width 384 to 1536 across compute budgets from $3\times10^{17}$ to $10^{19}$ FLOPs on FineWeb-EDU. We report two findings. First, the compute-optimal frontier is $N_{opt} \propto C^{0.56}$, close to the 0.50 that Hoffmann et al. report for dense Transformers, so spiking does not change how parameters and data should be traded off under a compute budget. Second, against non-spiking RWKV and Transformer baselines trained the same way, the loss gap splits into a backbone cost of about 0.16 nats that does not change with compute and a spiking cost that falls from 0.34 to about 0.23 nats and then levels off. These results give the first compute-optimal recipe for spiking language models and quantify the accuracy cost of their sparsity. 

---
# What Drives AI Citations? Content, Retrieval and Model Knowledge in Generative Engine Optimization (GEO) 

**Authors**: Mahe Chen, Nick Koudas  

**Link**: [PDF](https://arxiv.org/pdf/2610.15033)  

**Abstract**: Generative search engines such as ChatGPT, Gemini and Perplexity now decide which Web pages users see as sources, and a growing industry of generative engine optimization advises publishers on how to earn these citations. We study what shapes citation in two 90-day collections of answers to commercial questions in contrasting markets: 158,099 answers with 468,037 citations. Comparing pages of the same website, content matters in one way: pages whose title and paragraphs match the questions people ask are cited more, whereas FAQ blocks, structured data, summary boxes and page speed appear effective only across websites. Page properties nevertheless explain little of the variation between pages of a website. Replaying the questions through the engines' APIs shows why. Every page ChatGPT cites is one its own search retrieved, and which pages that search retrieves, for sub-queries the model rewrites on every answer and that rarely coincide with Google's or Bing's results, explains far more of the variation than page properties (32% against 2.8%). Content acts mainly through this step, by matching the engine's queries in meaning. The model's own knowledge shapes it too: ChatGPT writes brands it knows into its searches, prefers websites it knows, and names brands without any supporting page, and controlled experiments show that brand names reorder its rankings of products with identical facts. Pages are thus cited when searched; brands are also named when remembered. The findings replicate in a market of small B2B vendors. We release data and code. 

---
# HodgePhaseNet: Discrete Hodge Decomposition of Semantic Flows in Transformer Attention 

**Authors**: Kiyotaka Kasubuchi, Kazuo Fukiya  

**Link**: [PDF](https://arxiv.org/pdf/2610.14971)  

**Abstract**: WavePhaseNet introduced frequency-resolved semantic representations, QuantumPhaseNet introduced gauge-covariant phase connections, and ModularPhaseNet made a finite cyclic phase channel and its cycle holonomy computable in an ordinary Transformer. These constructions are complementary, yet they do not by themselves separate an observed collection of pairwise relations into a globally consistent direction, circulation around local compositions, and circulation that persists around a larger unfilled cycle. HodgePhaseNet addresses this separation by representing semantic transitions as an antisymmetric real-valued 1-cochain on a finite oriented simplicial complex. With vertex-edge incidence matrix $B_1$ and edge-face incidence matrix $B_2$, every such field has a unique orthogonal decomposition $\Omega = B_1^T\Phi + B_2\Psi + H$, where $B_1H=0$ and $B_2^TH=0$. The three component fields are unique even when their potentials are not. We give explicit projectors, a sparse implementation, integrability and gauge results, a verified numerical example, and a redesigned attention interface. The attention bias is defined on allowed ordered query-key pairs, remains consistent under arbitrary choices of edge orientation, and preserves a decoder's causal information boundary by constructing its complex from the visible prefix. Gauge invariance of the attention output is conditional on using invariant or appropriately compensated features. A simple difference of vertex embeddings has no coexact or harmonic component; a genuine relation encoder is therefore indispensable. Local circulation is not automatically contradiction, and harmonic flow is not automatically missing knowledge. These interpretations and any reduction in factual error require controlled empirical evaluation. This paper presents a mathematical and architectural proposal, not external natural-language performance results. 

---
# HiPAD: A Unified Gait, Audio-Visual, and Personality Fusion Network for Depression Detection, with a Hierarchical Personality Ablation and the Calibration Limits of Domain-Adapted LLMs 

**Authors**: Karol Chlasta, Piotr Struzik  

**Link**: [PDF](https://arxiv.org/pdf/2610.14961)  

**Abstract**: We present HiPAD, a single fusion network submitted to all six sub-tracks of the MPDD-AVG 2026 depression-detection challenge, covering the Elder and Young cohorts and three modality sets (gait, audio-visual, and both), each combined with personality descriptions. The network encodes each available modality with a shared BiLSTM, zero-pads absent modalities, and applies cross-modal attention only when at least two of gait, audio and video are present. Personality, which dominates our results, is kept out of the attention and fused by concatenation. Because the architecture is identical in every sub-track, differences between them reflect the data rather than the model, and we use this to study three questions. Our main finding is that the personality description, roughly 150 tokens about who the subject is, carries more depression signal than audio, video, and gait combined: gait plus personality alone (TLS 0.647) beats every configuration that includes audio and video. First, a four-level ablation of the personality description (L0-L3) quantifies this: removing every personality-derived feature, including the Big Five scores that earlier MPDD ablations keep, lowers the Track-Level Score by 0.499 to 0.507 and leaves the model at chance. Second, domain-adapting the personality encoder with LoRA (7B) or QLoRA (72B) on PHQ-8/9 data raises binary Cohen's $\kappa$ by 0.13 but lowers the overall score by distorting the PHQ-9 regression; a frozen RoBERTa-large encoder does better at this sample size. Third, at fixed architecture, handcrafted clinical features (eGeMAPS, OpenFace action units) outperform generic deep features (wav2vec2, DenseNet) on the audio-visual tracks. The unified system reaches a cross-track G-Score of 0.482; with under 100 subjects per cohort, the limiting factors are the size of the representation and the calibration of the outputs, not the choice of architecture. 

---
# AlphaRLM: Aligning Language Models for Literature-Grounded Symbolic Factor Discovery 

**Authors**: Jie Huang, Yuecheng Hong, Yao Chen, Hao Zhong, Linfeng Xiao, Xudong Liu, Hao Xia, Jianxuan Xin  

**Link**: [PDF](https://arxiv.org/pdf/2610.14959)  

**Abstract**: Formulaic alpha mining seeks to discover executable expressions that capture predictive market signals in a vast, noisy search space. Recent LLM-based approaches use prompting, iterative search, or seed-factor evolution, leaving direct factor construction through policy learning less explored. To address this, we introduce AlphaRLM, a reinforcement-learning framework that trains LLMs to construct executable symbolic factors from financial literature without input factor expressions. AlphaRLM incorporates Literature-Guided Operator Assembly (LEGO), which conditions factor construction on literature-derived ideas and executable operators, providing semantic priors for exploration. It further aligns the generation policy with market feedback through executable-validity and grouped-return-separation rewards that capture non-monotonic structures across factor quantiles. Extensive experiments on CSI 300, CSI 1000, and S&P 500 show that AlphaRLM outperforms strong factor-mining and LLM-based baselines in annualized excess return. Notably, the 8B model achieves strong cross-market performance, highlighting the promise of literature-conditioned, market-aligned factor generation. 

---
# What Survives When Activation Steering Is Used as Memory 

**Authors**: Vedant Jumle  

**Link**: [PDF](https://arxiv.org/pdf/2610.14958)  

**Abstract**: Activation steering has recently been used as a memory for skills: past experiences of an agent are stored as steering vectors and retrieved later. Facts and preferences about the user are assumed to need text instead. We test that assumption with a training-free activation memory beside a frozen LLM. The change an experience causes in late-layer states is stored as a shift, filed under a whitened key for the conversation's gist, and added back in a later session only when the gist matches. We argue that such a shift is a fixed average of an effect that attention, in context, recomputes for every question, so only question-independent content can survive. In a pre-registered study on Qwen3.5-2B and 9B, with judges from four model families, the memory is selective (about 98% of unrelated replies are unchanged), but facts are stored as the word to say, in the last layers only, and are almost never used (at most 15% clean use, against 96-98% in context). Leanings transfer only in a narrow strength window. Most strikingly, the reference subtracted when writing a preference decides whether its concept or its direction survives: unless the opposite is subtracted, a stored dislike is recalled as a like. 

---
# Predicting how model propensities generalize out of distribution 

**Authors**: June Hunter, Lily Wen, Owen Terry, Johannes Taraz, Ben Cohen, Niels Warncke  

**Link**: [PDF](https://arxiv.org/pdf/2610.14937)  

**Abstract**: How do language models generalize from their training data? This is a central challenge for alignment: models reliably learn their training distribution given enough samples, but we want models that behave well when deployed in new contexts. We study out-of-distribution (OOD) generalization of propensities. Our central experiment uses a range of elicitation techniques -- system prompting, few-shot learning, SFT on reference answers, and GRPO -- to increase a target propensity in a model, then measures how it spills over onto others across the resulting cross-elicitation matrices. We find that spillover is highly correlated between models when the same elicitation technique is used, but our SFT and GRPO runs induce different spillover profiles.
Elicitation of benign traits can sometimes generalize towards undesired traits, for example a model trained to claim that it is sentient often becomes more power-seeking. We further find that spillover partially resembles trait correlations found in humans and identify three clusters of traits that account for about half of the variance using exploratory factor analysis (EFA). Across many elicitation runs, we observe that models change in characteristic ways that don't depend on the targeted trait, a phenomenon we refer to as drift. Drift moves Qwen3.5-9B towards its base model.
When investigating if LLMs are able to predict spillover, we find that Claude Opus 5.5 predicts spillover slightly better than the models themselves, but worse than data-driven approaches using alternative elicitation methods or models. Overall, no single dominant predictor emerges, and generalization remains hard to predict. Code and data are available at this https URL. 

---
# SkillForgeBench: Evaluating Agent Reliability Grounded in Agent Skill Specifications 

**Authors**: Zhengxi Li, Junyi Zhou, Qiyuan Zhang, Chuhan Wu, Fuyuan Lyu, Le Tian, Xiao Zhou, Chen Ma, Yufei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.14918)  

**Abstract**: Evaluating an agent's ability to strictly satisfy diverse and procedurally complex requirements is essential to determining its practical reliability. Conducting such an evaluation demands agentic tasks that reconcile broad domain coverage with deep procedural complexity. Existing paradigms face a fundamental trade-off: expert-curated benchmarks achieve procedural depth but suffer from domain diversity bottlenecks, whereas crowdsourced benchmarks span diverse scenarios but lack execution rigor. To resolve this tension, we present SkillForgeBench, which leverages agent skills as unifying execution blueprints across three co-designed stages: (1) Resource Preparation uses domain-balanced skills to guide artifact retrieval, establishing broad workflow coverage; (2) Skill-driven Task Synthesis employs skills as procedural anchors to partition artifact roles, synthesize naturalistic task instructions, and package sandboxed scenarios for deep procedural execution; and (3) Holistic Evaluation Design translates skill-embedded requirements into fine-grained completion and process rubrics to enable strict requirement-level scoring. Comprising 842 sandboxed tasks across five major professional domains, SkillForgeBench reveals a steep capability hierarchy among eight mainstream agents, led by Opus 4.8 ($40.9\%$ strict completion). Requiring all constrains satisfaction uncovers severe capability bottlenecks across all model tiers that remain masked under individual rubric pass rates ($70.0\%$ to $85.7\%$ vs. $12.6\%$ to $40.9\%$). Crucially, under the external guidance from skills, top-tier agents obtain larger gains than low-tier agents (i.e., $\Delta 19.2\%$ for Opus 4.8 vs. $\Delta 5.8\%$ for Haiku 4.5). Finally, agents satisfy process rubrics far less frequently than completion rubrics (under $3\%$ for Haiku 4.5), proving that completion-only checks overestimate real-world agent reliability. 

---
# WorkBorne: Discovering Agentic Tasks from Real-World Artifact Transitions 

**Authors**: Junyi Zhou, Yufei Wang, Qiyuan Zhang, Chuhan Wu, Zhengxi Li, Wentao Zhao, Jie Zhou, Le Tian, Xiao Zhou, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.14915)  

**Abstract**: Developing terminal-based agents for long-horizon execution demands synthetic tasks that tightly couple complex instructions with authentic, interactive environments. However, current automated synthesis suffers from a fundamental asymmetry: instruction-oriented methods fabricate low-fidelity, synthetic environments, whereas resource-oriented methods treat real artifacts merely as static inputs, degrading task instructions into shallow, trivial operations. To address this limitation, we introduce WorkBorne, a synthesis framework that uses artifact transitions as the central primitive for modeling the full life-cycle of agentic tasks. In WorkBorne, each transition specifies who performs the task, which artifacts anchor the initial state and intended outcome, and the corresponding relationship between them. Specifically, WorkBorne first constructs coherent and diverse artifact corpora from iterative web retrieval, then discovers non-trivial transitions by identifying meaningful input-output artifact assemblies and the relations that connect them. Each discovered transition is finally instantiated into grounded agentic tasks. We construct 1,061 artifact corpora and 7,882 task environments, yielding 15,485 high-quality trajectories, with 53.6% exceeding 30 assistant turns. Experiments on Qwen3.5-27B and Qwen3.5-35B-A3B show that WorkBorne substantially improves over base models (i.e., improving Terminal-Bench 2.1 from 43.1% to 61.8% and ClawEval General score from 75.0 to 80.2) and achieves the strongest overall performance among other synthesis baselines (e.g., TMax (Ivison et al., 2026), RST (Li et al., 2026a), Terminal-Universe (Wu et al., 2026a)). Further analysis shows that transition is strongly associated with execution depth, while corpus breadth and transition depth provide complementary scaling gains. 

---
# Description Complexity Bounds for Analytic Approximations of Lipschitz Functions 

**Authors**: Artem Vozmishchev  

**Link**: [PDF](https://arxiv.org/pdf/2610.14893)  

**Abstract**: We study how many bits are needed to write down an elementary analytic expression that uniformly approximates a Lipschitz function. Expressions are finite trees over the basis $\{+,-,\times,÷,\exp\}$ with the variables and rational constants, and their size is $\dc(f)$, the length of the shortest description of $f$ in a fixed binary description language. For functions of one variable we prove that every $L$-Lipschitz $F:[0,1]\to[0,1]$ has an everywhere-defined approximant of accuracy $\Delta$ with $\dc(f)\le C\,(L/\Delta+\lg(1/\Delta+2))$ for all $L,\Delta>0$, by a linear spline smoothed with logistic ramps in a description language with one repetition construct. We prove a matching lower bound valid for \emph{every} description language: for $\Delta\le 1/36$ and every $L>0$ some $L$-Lipschitz function needs at least $\tfrac1{16}(L/\Delta+\lg(1/\Delta+2))$ bits. Hence the worst-case description complexity is $\Theta(L/\Delta+\lg(1/\Delta+2))$. For functions on $[0,1]^d$ the same method gives $\Theta_d\bigl((L/\Delta)^d+\lg(1/\Delta+2)\bigr)$, using a tensor-product spline whose coefficients are mixed second differences and cost $O(d)$ bits each. Without the repetition construct the upper bounds carry an additional logarithmic factor, whose necessity for plain expression trees is left open. A reference encoder, decoder and evaluator confirm the constructions numerically, and figures and tables report the checks. 

---
# CUSE: Fine-Grained Multimodal Unlearning with Counterfactual Guidance and On-Policy Self-Distillation 

**Authors**: Chenlu Ding, Jiancan Wu, Yanchen Luo, Tianyu Zhang, Yancheng Yuan, Xiang Wang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2610.14763)  

**Abstract**: Multimodal large language models (MLLMs) can reproduce sensitive, copyrighted, or unsafe information from their training data. Effective unlearning must remove target information while preserving non-target content, even within the same response. Existing methods typically apply uniform forgetting objectives to complete responses and rely on fixed, off-policy supervision, resulting in coarse token control and misalignment with the model's current behavior. We propose CUSE, a Counterfactual-guided Unlearning framework via on-policy Self-distillation for multimodal large language models. It first constructs counterfactual inputs by removing target-related evidence and uses the resulting changes in token probabilities to identify which tokens require stronger forgetting. It then distills token-level forgetting and retention targets on responses sampled from the current policy, with a frozen copy of the original model serving as the teacher model augmented with the corresponding forgetting- and retention-oriented privileged contexts. This design jointly determines what to forget and what to preserve. On MLLMU-Bench, CUSE reduces forget-set ROUGE-L by 33.6% while achieving 1.2% higher retain-set ROUGE-L than the strongest baseline. 

---
# How to Build a Benchmark That Actually Measures Taste 

**Authors**: Xiaofan Wu, Xi Zeng, Miaoxia Chen, Peishan Chen, Shuyan Li, Chen Wang, Jiyun Yao, Hanyong Zhong, Jiahao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.14707)  

**Abstract**: Expert-aligned "taste" benchmarks should reward craft judgment, not the ability to game the test. We introduce TasteBench, a pairwise good-vs-bad benchmark across visual, brand, book, and tech items under a frozen rubric with expert-only gold (panel n>=3; percent agreement >=0.60 primary gate; weak engagement labels train/analysis-only), together with a control protocol that separates craft judgment from test-gaming. Applying the protocol to a Round-1 pilot (primary_human_expert; three biological human experts; n=250 pairs) surfaces and quantifies three validity threats: two construction artifacts that inflate the trivial-baseline layer, and a difficulty ceiling that inflates every score. (1) Position: the preferred item is slot A with probability 0.956, so an always_item_a policy scores 0.956 (fair chance 0.5). (2) Lexical leakage: a filename token cue scores 0.912. (3) Difficulty ceiling: craft domains show near-ceiling human agreement (book/brand/tech 1.0), and even under A/B randomization (eval_assignment_seed7; gold_side_A ~0.47) all five VLM cards reach 1.0 on craft, with only visual separating (~0.84-0.91; a post-hoc hard subset, n=107, leaves craft at 1.0). The protocol (expert-only gold, a weak-label firewall, side randomization at construction, filename scrubbing, and hard filters) makes each artifact measurable and, for the two construction artifacts, correctable; we release a reference eval harness and a side-randomization tool, and show the tool removes the position skew (0.916 to 0.495/0.505). The result is a calibrated view of what the benchmark does and does not yet measure: position and leakage are fixable, whereas the craft-domain ceiling is a pair-difficulty limit that Round-2 must address. We do not evaluate engagement, era/circle conditioning, or calibration here, and we claim no general taste SOTA. 

---
# Towards Merging-conditional Backdoor Unalignment against Large Language Models 

**Authors**: Yukun Chen, Yidou Chen, Yiming Li, Xinyu Zhang, Baosong Yang, Hongwei Yao, Zhan Qin  

**Link**: [PDF](https://arxiv.org/pdf/2610.14701)  

**Abstract**: Model merging has become widely adopted for enhancing the task-specific performance of Large Language Models (LLMs) without additional training. While the specialized model from third-party platforms may undergo safety audits (e.g., backdoor detection), we reveal a novel and critical threat: One-sided Merging-Conditional Backdoor (OMCB). OMCB remains dormant and undetectable within one standalone model but activates upon merging, triggering safety unalignment. We first theoretically analyze the existence and geometric location of OMCB, revealing that the parameter shift crossing safety decision boundaries is the main cause of backdoor activation. Building upon this insight, we propose SMOKE, a meta-learning-based attack framework. SMOKE implants OMCB via a bi-level optimization strategy: the inner loop synthesizes surrogate meta-vectors through random low-rank initialization and finite-step maximization, while the outer loop optimizes the target model by simulating the merging process. Furthermore, preliminary analysis shows that the injected OMCB has bounded generalization loss under unknown merging configurations. Extensive experiments across diverse model architectures and merging techniques validate the effectiveness of SMOKE and its resistance to existing backdoor detections, highlighting a significant yet previously overlooked vulnerability in the model merging paradigm. 

---
# VoTSpeech: Decoupling Voice Design from Speech Generation in Instruction-Following TTS 

**Authors**: Wenbing Yang, Qihang Lu, Zihan Sun, Peilei Jia, Yingming Gao, Ya Li, Jun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.14694)  

**Abstract**: Instruction-following text-to-speech (TTS) typically leaves voice design implicit, as conventional speaker embeddings encode reference audio rather than text-driven targets. We propose VoTSpeech, which decouples voice design from speech generation via an explicit intermediate voice latent. Built on a shared language model backbone, VoTSpeech first predicts a continuous voice latent from text instructions using a flow-based module. This latent serves as a shared condition for both the language model and the Acoustic DiT, guiding autoregressive sequence modeling and acoustic latent generation, respectively, before an audio VAE decoder converts the acoustic latents into waveforms. To capture speaker identity and complementary acoustic nuances, the voice latent is jointly supervised by speaker embeddings and temporally pooled audio VAE features. With task-specific adaptation, VoTSpeech performs strongly on InstructTTSEval-ZH, demonstrating competitive capabilities against strong open-source models while maintaining high naturalness. These results confirm that explicit voice latent design and its conditioning interface effectively empower text-driven voice design for controllable speech synthesis. Code, model, and audio samples are available at this https URL . 

---
# The Company a Question Keeps: Auditing Open Jev-Style Models for Retrieval 

**Authors**: Jianbo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.14631)  

**Abstract**: Jev's documented interface evaluates typed questions independently against shared evidence. We audit request independence in seven open Jev-style checkpoints on 276 constructed three-passage states from 92 TREC retrieval queries. Paired interventions separately change requested questions, displayed evidence, and candidate eligibility. Request expansion induces strict binary flips in 9.1%/8.0% of Nimble's judgments and changes 21.4%/18.9% of selection-or-abstention actions in TREC DL2019/2020 under a threshold of 0.5 fixed before inference. Correct-action rates change by 0.0/+8.2 percentage points, respectively, with 95% query-bootstrap intervals of [-6.0,6.0] and [3.8,12.6]. Field masking traces all 70 request-set flips to the schema fields serialized into each scoring prompt. In Von, fixing a shared separator's position eliminates two eligibility-induced preference flips in selected diagnostic states. Laya-typed's default CUDA path produces five passage swaps without strict binary flips, and both yearly action-change intervals include zero; in an FP32 rerun, its probability movement vanishes at the model's four-decimal reporting resolution. 

---
# Decoupling Knowledge Forgetting and Refusal Learning in MLLM Unlearning 

**Authors**: Haodong Chen, Shengtao Wen, Dong Liang, Lingbing Guo, Xiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.14605)  

**Abstract**: Current evaluations of multimodal unlearning typically focus on whether a model stops outputting target information. This surface-level criterion, however, masks two critical failure modes. First, knowledge suppression techniques often induce confabulation, leading the model to hallucinate facts about the unlearned subject. Second, refusal training merely alters observable behavior without erasing the underlying knowledge; such models might decline open queries but still identify correct answers under multiple-choice probing. To overcome these limitations, we propose Forget-Aware Self-Distillation (FASD). This method is designed to achieve honest ignorance, a state where the model genuinely loses the target knowledge and learns to acknowledge its absence. FASD separates these goals into two complementary channels. The behavior channel distills appropriate refusals from an ignorance-prompted copy of the model itself. Crucially, target tokens are masked from the teacher distribution to avoid accidentally reteaching the forgotten concepts. Simultaneously, the knowledge channel directly suppresses ground-truth answer spans independently of the model's generated responses. Experiments demonstrate that FASD successfully unlearns target knowledge, prevents information leakage in open-ended tasks, and preserves performance on retained data. The resulting model not only forgets the specified information but accurately recognizes the boundaries of its own knowledge. Code and data will be released upon publication. 

---
# Fine-Tuning Strategy, Not Just Data: A Statistically Rigorous Audit of Gender and Stereotypical Bias in Embedding Models for HR Recommendation Systems 

**Authors**: Ekta Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2610.14579)  

**Abstract**: Embedding models underpin modern job-candidate matching: resumes, job descriptions, and queries are encoded as dense vectors, and recommendations come from nearest-neighbor retrieval over that space. Any social bias in the embeddings therefore feeds directly into HR recommendations. We audit six multilingual sentence-embedding models (LaBSE- and E5-family, both public and custom fine-tuned) using Bias-in-Bios (occupational gender bias) and StereoSet (stereotypes across profession, race, gender, and religion). We treat statistical validity as a first-class concern: we identify a non-independence error in a naive significance test, where pooling non-independent pairwise similarities inflated a trivial 0.0046 mean difference to $p \approx 0$, and replace it with a profession-level paired Wilcoxon signed-rank test, individual-resampling bootstrap confidence intervals, and permutation tests for StereoSet. Under the corrected methodology, knowledge-distillation (KD) fine-tuning of E5-Base significantly increases measured bias on Bias-in-Bios ($\Delta = -0.177$, $p < 10^{-8}$; more negative means more bias), although this measure is sensitive to global embedding geometry. The KD result is corroborated by pro-stereotype patterns in the StereoSet profession category (uncorrected $p < 0.05$). Binary-classification fine-tuning of the same base model shows no statistically distinguishable change ($p = 0.218$). Fine-tuning strategy is thus an actionable design lever for practitioners building HR recommenders. We also report which benchmark categories are underpowered, and note that two domain-adapted models could not be included in significance testing due to a software-access constraint, which limits our conclusions. 

---
# Reward-Aligned Distillation 

**Authors**: Yilan Chen, Zian Su, Manish Prajapat, Dingmin Wang, Rein Houthooft, Rui Song, Vijay Lingam  

**Link**: [PDF](https://arxiv.org/pdf/2610.14572)  

**Abstract**: Reinforcement learning with verifiable rewards improves the reasoning capabilities of large language models, but remains computationally expensive and sample inefficient. On-policy distillation (OPD) provides dense token-level supervision from a stronger teacher, yet standard OPD distills every token regardless of whether the teacher's preference actually improves task reward. Existing token-selection methods rely on proxies such as uncertainty or teacher--student divergence, which need not identify reward-improving supervision. We introduce Reward-Aligned Distillation (RAD), a bilevel framework that weights distillation signals based on their predicted effect on the student's post-update reward. RAD differentiates through a temporary optimizer step to score each token's supervision, then updates the token weights by projected ascent from the initial token weights. The resulting weights can strengthen, suppress, or reverse individual distillation-gradient contributions. At a matched training-response budget, RAD improves mean accuracy on six competition-math benchmarks over the strongest baselines by 6.5 percentage points for Qwen3-1.7B (non-thinking) and 2.5 points for Qwen3-4B-Instruct-2507. RAD-Simple, a simplified variant, achieves comparable math accuracy at the same training-response budget. On LiveCodeBench-v6, RAD reaches 54.69\% and 58.67\% pass@1 with Qwen3-1.7B (non-thinking) and OLMo3-7B-Instruct, respectively, exceeding the strongest baselines by 8.7 and 6.4 points. These results support reward-guided weighting of teacher supervision across model families and tasks. 

---
# Query-Adaptive Neuron Steering for LLMs 

**Authors**: Yanxi Pan, Jingwen Wu, Yan Ma, Xiang Chen, Xinyi Zeng, Xuetong Wang, Yunyan Zhang, Yue Zhang, Yu Tian, Jianfeng Pan, Hongyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.14519)  

**Abstract**: Effectively eliciting the capabilities of large language models (LLMs) at inference time remains an important challenge. Existing studies address this challenge through context engineering, which shapes user queries, or fixed steering, which directly intervenes in model computation. However, predefined contexts and fixed intervention strategies may not adequately accommodate the diverse knowledge and reasoning requirements of individual queries. This raises a natural question: Can internal steering in LLMs be made query-adaptive? In this work, we investigate whether query-induced internal activations can guide adaptive steering in LLMs. Our pilot studies reveal subject- and layer-dependent neuron activation patterns and show that amplifying subject-relevant neurons can improve model performance. Based on these findings, we propose Q-NeuS, a query-adaptive neuron steering framework that dynamically determines which neurons to modulate and by how much for each input. Specifically, Q-NeuS constructs a compact subject-guided candidate neuron space and employs a lightweight Query-Adaptive Gate to generate neuron-wise modulation coefficients. To learn effective intervention strategies, we introduce Neu-GRPO, which adapts group-relative optimization to train the Query-Adaptive Gate using the relative performance of multiple neuron intervention configurations. The base LLM remains frozen throughout training and inference, and no task or subject labels are required at inference time. Extensive experiments across multiple LLMs and benchmarks demonstrate that Q-NeuS consistently improves model performance and exhibits broad applicability across model scales and strong cross-benchmark transferability. 

---
# Re-executing Coding-Agent Evaluations: Unchanged Decisions and Unresolved Policy Risk 

**Authors**: Tianyu Ding, Yaocheng Tong, Muzhi Yang, Fei Jiang, Lei Yang, Boya Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.14460)  

**Abstract**: Executable tests grade large language model (LLM) coding-agent patches for leaderboards, rewards and self-improvement. Two preregistered re-execution studies returned negative results. SWE-bench Pro v1's offline option rejected none of 59 accepted Go patches whose grading configuration changed. Their exploratory, unweighted upper confidence limit is 4.95% under candidate independence, leaving a 3% exposed-patch risk unresolved; the preregistered estimator uses inverse sampling-probability weights. An extension was inconclusive. In a selected stratum, a reconstructed module cache built from the reference solution's dependencies rejected accepted alternatives from four models that passed with their own dependencies. Re-executing 1,929 selected agent-version/task evaluations from the Darwin Gödel Machine's archive preserved best-agent selection and system-variant ranking, with search history and scores of evaluations not rerun fixed. Exploratory host, image and adapted-grader comparisons also preserved them. Seven of 23 disagreements with archived verdicts did not recur on a second host; rebuilt images changed 16 three-run majority verdicts (0.83%) under four-CPU execution. We provide an evidence-to-claim summary, lessons for grader audits and an anonymous replication package. These results concern accepted patches and fixed-history decisions; general policy safety and adaptive-learning robustness require broader evaluations. 

---
# MemForm: Benchmarking Memory Formation Poisoning in LLM Agents 

**Authors**: Haowen Pan, Pengyang Shao, Yunshan Ma, Yanzheng Jin, Naixin Zhai, Xiang Wang, Xun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.14431)  

**Abstract**: Long-term memory enables LLM agents to learn from past interactions but exposes memory formation to adversarial influence. Existing studies primarily organize memory poisoning by attack channels, corruption patterns, or lifecycle stages, leaving formation interfaces insufficiently characterized. We formulate Memory Formation Poisoning as adversarial manipulation of the interfaces through which persistent agent memory is produced and committed, and introduce MemForm, a controlled diagnostic benchmark covering five interfaces: Content, Evidence, Scope, Selection, and Path Poisoning. We introduce Selection and Path Poisoning as controlled intervention categories that isolate retention and consolidation order effects, respectively. Each category pairs clean and poisoned conditions with interface-specific controls. Evaluations across six LLMs and two response formats reveal broad susceptibility at all five interfaces under the evaluated memory protocols and implementations. Crucially, authentic experiences with successful outcomes do not guarantee reliable behavior after memory formation: changes to contextual support, retention, or consolidation order can induce undesirable downstream behavior. In particular, consolidation order can alter behavior even with identical retained experiences, depending on the memory formation procedure. Additional analyses show that interface applicability depends on memory architecture and that Path Poisoning susceptibility depends on the updater. The evaluated defenses provide incomplete mitigation that varies by interface. These findings motivate protecting both what is committed to persistent memory and how memory is formed from experiences. Code and data are available at this https URL. 

---
# SPROUT: Scaling Agentic Reinforcement Learning for Long-Horizon Software Engineering 

**Authors**: Mengyang Liu, Weiming Wang, Haixin Nan, Yu Wu, Peize Li, Shihong Huang, Xuning He, Xue Jiang, Yihong Dong  

**Link**: [PDF](https://arxiv.org/pdf/2610.14405)  

**Abstract**: Reinforcement learning (RL) has driven substantial progress in coding agents by relying on verified successful trajectories as learning signals. However, such trajectories are scarce in long-horizon software engineering tasks, where a minor error can invalidate the entire rollout. Repeatedly restarting from the initial state is commonly used yet inefficient, and resetting only the context with working directory misses state changes such as installed dependencies. Therefore, branching from intermediate steps to preserve earlier progress requires restoring both the agent context and execution environments. In this paper, we present SPROUT (SnaPshot-based RollOUT), an agentic RL rollout framework that scales long-horizon coding tasks by resuming and branching from intermediate states. Specifically, SPROUT was built on top of a lightweight infrastructure that checkpoints both the filesystem and the agent context after each assistant turn. A verifier-guided backtracking policy then selects where to resume a failed attempt and how to proceed. Extensive experiments on long-horizon coding tasks show that SPROUT achieves up to 23.6% improvement over pass@k across various LLMs, while using $32$--$52%$ of their tokens. For RL training of Qwen3.8-27B, SPROUT stably outperforms standard GRPO by $13.9$ points on three long-horizon coding benchmarks while cutting the time per training step by $39.5%$. SPROUT thus makes RL on long-horizon coding tasks both more effective and cheaper to scale. 

---
# RealityBench: Simulated Environments to Evaluate a Vision Language Model's Ability to Document Human Physical Work 

**Authors**: Andrew Perrault, Anmol Takiar, Dominik Wienecki, Arnab Nandi  

**Link**: [PDF](https://arxiv.org/pdf/2610.14397)  

**Abstract**: Augmenting worker capabilities in the physical world through guidance has been a decades-long goal for AI agents. One key supporting capability is to understand what the worker is doing, seeing through the human's camera or smart glasses, and assist them in documentation. Despite advances in vision language models, their capabilities in these settings remain uncertain and are challenging to benchmark because most physical work data is proprietary, and involves too much variability to study in a principled manner. Unlike structured settings such as kitchens and warehouses that have predictable lighting, geometry, and objects, field-facing tasks such as rooftop solar installation and vehicle repair are unconstrained and high variance. Here, we model real-world physical work in Blender across eleven environments, incorporating variants in both image acceptability and documentation. The acceptability task asks whether a particular view would be suitable proof-of-work and documentation asks whether an agent can understand the task-relevant state of the environment from that view. Across 26 vision-language models, we find that top models largely solve documentation when the answer is text in frame, but are weak at reading measurements and at judging small parts and spatial relations, such as whether a tape measure leans. Acceptability is harder to pin down because human raters disagree with each other. Measured against a scorer that agrees with the raters' majority about as often as a single rater does, the best models cannot be distinguished from a single non-author rater, at a cost within the range of a human pass. In the three environments where we can trace acceptability errors, most wrong rejections come from misreading the scene rather than the requirement. On the three documentation questions for which we have real images, model rankings on rendered frames transfer to real images. 

---
# Tokenization-Invariant Watermarking: A Byte-Local Method for LLM Text 

**Authors**: Tung-Ling Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.14378)  

**Abstract**: We introduce Tokenization-Invariant Watermarking (TI-WM), a method that embeds a watermark during token generation but keys and scores it on canonicalized UTF-8 bytes. The detector does not invoke the tokenizer. Canonicalizer-preserving edits leave its score unchanged; other localized canonical-byte edits affect a bounded number of scoring windows. Across three fixed keys and 128 prompts per key and model, a deterministic white-box attack at a 5% word-edit budget leaves mean attacked-AUC 0.530 on Qwen3-8B and 0.664 on Gemma 4 E4B, compared with 0.410 and 0.489 for Unigram with detection-side deletion. These operating points are not jointly matched on quality and clean detectability. Bias sweeps show model-dependent distortion costs, and shared-rule controls support a locality interpretation without establishing a unique advantage of bytes. Visible insertion outside the canonicalizer's deletion set defeats the default design. Alphabetic filtering closes that channel but does not resolve source shift or calibrated detection power. TI-WM thus provides a concrete construction with local scoring guarantees and measured attack-ranking gains, not a general robustness or low-false-positive-rate deployment guarantee. 

---
# Which Metadata Matters for LLM Tool Selection in MCP? Evidence from Controlled Factor-Level Interventions 

**Authors**: Xiaofan Li, Fengchun Qiao, Xing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.14334)  

**Abstract**: In Model Context Protocol (MCP) systems, LLMs select tools from metadata rather than from their implementations. Names and descriptions summarize what a tool does, while the parameter schema specifies the fields a tool accepts and how they are organized. Our analysis of metadata for 367,174 MCP tools reveals substantial schema variation among functionally similar tools. Many of these tools share names and similar descriptions but expose different parameter schemas. This raises a key question: can parameter-schema differences influence tool selection before any call is made? We study this question through controlled metadata interventions across five LLMs and five tasks. Each comparison changes exactly one metadata factor (server name, tool name, description, or parameter schema) while keeping the others fixed. We balance tool positions and use conditional logistic regression to estimate position-adjusted effects. We find that: (1) parameter-schema changes produce the largest and most consistent selection shifts among the tested interventions. Models tend to favor separate top-level parameters over grouped or composite forms. (2) Descriptions that explain parameters also tend to increase selection, while name effects are weaker and less consistent. (3) Targeted comparisons show that schema effects depend on the specific parameter names and on how the request represents the same information. When only one schema satisfies an explicit request requirement, all five models select the compatible tool despite position preferences. These results show that parameter schemas influence not only how tools are called, but also whether they are selected. For MCP developers and evaluators, schema design and tool ordering can therefore affect which implementation is invoked even when functionality is unchanged. 

---
# RUBATO: Decoupled Clocks Allow Full-Duplex Models to Think as Fast as They Can 

**Authors**: Yi-Jen Shih, Puyuan Peng, Haohe Liu, David Harwath, Abdelrahman Mohamed  

**Link**: [PDF](https://arxiv.org/pdf/2610.14299)  

**Abstract**: Recent Speech Full-Duplex (FDX) models are capable of engaging in naturalistic conversation by listening and speaking at the same time. Building on this foundation, recent architectures introduce internal thinking streams to enhance their intelligence. To guarantee these streams process successfully in real time, existing methods enforce a fixed decoding rate, which assumes a hardware profile capable of a specific latency. We argue this static approach is severely suboptimal. Because internal thinking is intrinsically asynchronous, an ideal FDX model should sustain real-time conversational flow while allowing each stream to decode at its own hardware-adaptive rate. Thinking streams must process as fast as the hardware allows to deliver accurate answers faster. To solve this, we propose RUBATO, a novel multistream positional encoding that assigns independent clocks to entirely decouple the streams. This structural isolation empowers the model to zero-shot interpolate across any inference decoding speed. It drastically simplifies dataset construction by requiring training on only two extreme decoding speeds while smoothly adapting to any intermediate rate during deployment. RUBATO demonstrates consistent robustness across five spoken reasoning tasks. On Spoken GSM8K, a baseline model collapses to 1.7% accuracy at interpolated speeds. Conversely, RUBATO-Full sustains 48.7% to 51.1% accuracy across all rates, leveraging faster hardware to achieve a 6.5x shorter time-to-answer with a constant time-to-first-response. This advantage holds on BIG-Bench Audio and amplifies significantly during complex full-duplex interruptions. Specifically, when a user interjects while the model is actively speaking and reasoning, RUBATO-Full achieves 40.0% to 45.1% accuracy, whereas the baseline fails completely. 

---
# Harness^2: Recursive Agent Harnessing for an Open World 

**Authors**: Ruiyao Xu, Yanfei Chen, Zhongying CuiZhu, Bhavana Dalvi Mishra, Yifei Ming, Han Yu, Rujun Han, Chen-Yu Lee, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2610.14224)  

**Abstract**: LLM agents' capabilities depend not only on their underlying models but also on their execution harnesses. Recursive self-improvement (RSI) offers a promising approach to automatically evolve these harnesses. However, deploying RSI in open-world settings still faces two critical challenges. First, open-world tasks are typically unseen during the harness design phase and novel at test time, leaving no ground-truth verifiers to reliably guide online harness edits. Second, the RSI process remains highly task-specific and fails to generalize to novel scenarios. We introduce Harness^2, a plug-and-play, recursive agent harnessing framework that addresses these challenges through two connected levels of recursion. Task-level recursion refines the current task's harness using a contrastive Propose-Probe-Compose step that reliably guides edits without ground-truth verifiers, and can be scaled further via parallel or sequential recursion. Harnessing-level recursion then extracts experience from these edits to update a continual improvement harness, enabling the agent to better adapt to future tasks. Across three professional benchmarks and four model-harness configurations, a single step consistently improves GLM-5 and Gemini 3.5 Flash over their base harnesses, yielding gains of up to 11.5 points on WorkBuddy-Bench, 4.5 points on JobBench, and 11.8 percentage points on LAB, with further gains as the task-level recursion budget increases. 

---
# Learnable Low-Rank Polynomial Sketch for Effective Linear Attention 

**Authors**: Zongyue Qin, Haoran Deng, Jason Cong, Yizhou Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.14219)  

**Abstract**: Softmax attention in Transformers suffers from quadratic complexity in sequence length, making it impractical for long-context applications. Linear attention alleviates this issue by replacing the exponential kernel with alternative functions that enable linear-time computation. Among existing linear attention approaches, recent studies have shown that polynomial kernels are particularly effective, as they exhibit sparse and spiky behavior similar to softmax attention, which emphasizes large dot products while suppressing irrelevant interactions. However, the exact computation of high-degree polynomial kernels is infeasible for high-dimensional representations. As a result, prior work relies on approximate polynomial kernels. This introduces a non-negligible approximation error. In this paper, we show that, from the perspective of polynomial kernel approximation, existing linear attention methods are still suboptimal. We propose \textbf{L}earnable \textbf{Lo}w-Rank \textbf{P}olynomial \textbf{S}ketch (LLoPS), a principled and flexible framework for approximating polynomial kernels with linear attention. Our method learns a low-rank polynomial sketch that provably achieves a strictly smaller approximation error than existing approaches. Experiment results show that LLoPS achieves the highest performance across extensive benchmarks, comparing with various linear attention baselines. 

---
# Simple Linear Corrections for Compression in Late Interaction 

**Authors**: Mihir Agarwal, Sabyasachi Basu, Viraj Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2610.14196)  

**Abstract**: Late-interaction (multi-vector) models represent documents using a set of token-level vectors, providing a middle ground between cross-encoders and dense encoders in terms of quality and cost of embedding. However, compared to the latter, the indices are costly to store and score. Pooling and pruning based compression techniques are deployed widely but deteriorate in quality noticeably with aggressive compression. We investigate whether a lightweight change to the scoring function can recover retrieval quality after document-vector compression. For each combination of encoder, compressor, and vector budget, we learn a global linear correction by distilling the uncompressed model's score distributions over candidate documents. The encoder and compressor remain frozen, and the correction can be applied to each query at search time or once, offline, to the stored vectors.
We evaluate the method with ColBERTv2 and Jina v4 on MS MARCO and BEIR, and with ColPali2 on the ViDoRe v3 visual document retrieval benchmark. Experiments cover four training-free compression methods, multiple vector budgets, in-domain adaptation, and transfer from MS MARCO to BEIR. In the in-domain setting, the correction improves over baseline compression in 154 of 160 BEIR settings, while transfer from MS MARCO is more variable across models and datasets. We also find that matrices trained with synthetic queries can improve over baseline compression on ViDoRe v3. These results show that a small learned correction can recover some of the quality lost to compression without retraining the encoder or recompressing the documents. 

---
# Unconstrained CTI-REALM with Open-Weights 

**Authors**: Sahil Garg, Ruhi Gottumukkala, Vishal Agarwal, Shubham Vishwakarma  

**Link**: [PDF](https://arxiv.org/pdf/2610.14180)  

**Abstract**: Organisations that want language-model agents for security work face a choice between open-weight models, which they can run and audit themselves, and far more expensive closed frontier models. We study that choice on CTI-REALM (arXiv:2603.13517), a benchmark in which an agent turns a threat-intelligence report into a working detection by querying a live telemetry store. We evaluate twenty-six models, nineteen open-weight and seven closed, under a common harness, scorer and panel of LLM judges. Our main finding is that an open-weight model can match the strongest closed models at less than a tenth of their cost, provided it is given enough turns. The benchmark's published limit on agent messages is a confound: a task that exhausts the limit scores zero on every checkpoint, including those it had already passed, so results at that limit partly measure which models finish in time. At the published limit the leading open-weight and closed models cannot be distinguished. With the limit relaxed, the models it was truncating recover, and the highest rewards belong to open-weight models. Among the leading models, price is a poor guide to capability: models with similar rewards differ in cost by more than an order of magnitude. Because the benchmark's checkpoint scores largely record what an agent did or mentioned, we also introduce stage-level metrics that score what the agent submits, by executing its detection queries in the telemetry store and judging its outputs against the answer key. Rather than combining them into one more overall reward, we show that different models lead different stages, which suggests routing each stage of the workflow to the model that performs it best rather than choosing a single model by its overall score. We also report how many tokens each model generates, and find that the share of output spent on thinking does not track the reward. 

---
# AutoEnvScaling: Automating the Data Flywheel with Terminal Agents 

**Authors**: Simon Yu, Baolin Peng, Bo Liu, Zhengyan Shi, Zhaoyang Wang, Hao Cheng, Qianhui Wu, Wenlin Yao, Natasha Jaques, Weiyan Shi, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.14163)  

**Abstract**: Terminal agents solve tasks in software engineering and scientific research by executing commands in a terminal. Training them via reinforcement learning (RL) requires diverse and verifiable environments, but these environments are still mostly designed by human researchers, so their supply grows only as fast as researchers can work. We introduce AutoEnvScaling, which treats environment design as a terminal task and uses agents to automate the data flywheel. A proposer agent builds new RL environments from the rollouts of the solver being trained, and the solver's new rollouts guide the next round. By turning environment generation into a terminal task, AutoEnvScaling increases the valid-environment rate by up to 25 times compared to prompt-only generation, at up to 75% lower cost per valid environment. We show that training Qwen3.5-35B-A3B on AutoEnvScaling environments from GPT-5.6-sol improves its Terminal-Bench 2.1 score by 4.3 points over the matched Tmax baseline. The broader goal of AutoEnvScaling is to enable recursive self-improvement (RSI): by making both environment generation and solving terminal tasks, it lets one model co-evolve in both roles. We show that Qwen3.6-35B-A3B, acting in both roles, gains 12.2 points on Terminal-Bench 2.1 and raises its Terminal-Bench 4.0 score from 1.6 to 3.2. The pipeline also generalizes across agentic domains, generating environments for software engineering, computer use, professional work, and GPU kernel programming. Overall, these results show a path toward agents building their own training environments, reducing human reliance and taking a core step toward RSI. 

---
# OpenMobile-2: Building Versatile Mobile Agents with Scalable Environments and App-Native Tools 

**Authors**: Chenyang Yan, Yingying Zhang, Kanzhi Cheng, Qiushi Sun, Zhengyuan Pan, Zheng Ma, Hang Yan, Yian Wang, Nuo Chen, Jialin Cao, Xingdong Gong, Wenpo Song, Han Chen, Zichen Ding, Fangzhi Xu, Shujian Huang, Xinyu Dai, Yichen Liu, Tiankuo Yao, Bo Wang, Ben Kao, Jianbing Zhang, Lewei Lu, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.14157)  

**Abstract**: We introduce OpenMobile-2, a near-frontier mobile agent with fully open training environments and recipes. We make three key advances: (1) Diverse environments with simulated commercial apps: We build MobileGym++, featuring 35 realistic, functionally rich commercial-style apps with cross-app workflows, while preserving full controllability for reset and verification. Together with newly configured apps in Android emulators, this yields a diverse playground spanning over 110 apps. (2) Open training data at scale: Building on this foundation, we curate nearly 12K mobile interaction trajectories for supervised fine-tuning and the largest open collection of verifiable RL training data for mobile agents, comprising over 2K executable tasks with automatic rewards. (3) Hybrid GUI and app-native tool use: We explore an experimental mobile-use setting where apps expose selected functionalities as app-native tools alongside their GUIs, allowing agents to interleave GUI actions and tool calls within a task. We implement this setting in MobileGym++ with over 300 carefully scoped tools across 50+ apps. Additionally, we introduce MobileGym++ Bench for complex, long-horizon commercial mobile scenarios, supporting both GUI-only and hybrid GUI-tool evaluation on a shared task suite. OpenMobile-2 performs competitively across established benchmarks, including AndroidWorld (79.9) and MobileWorld (50.4), while showing promising transfer to real-device mobile use, nearly doubling SPA-Bench performance from 31.9 to 59.6. All resources are available at this https URL. 

---
# The AI's Poker Face: Contrasting Persuasion and Sycophancy in Voice-Based Multi-Turn Conversations 

**Authors**: Ilia Frantzi, Bernie Hogan, Max M. Lang, Paul C. Bauer  

**Link**: [PDF](https://arxiv.org/pdf/2610.14124)  

**Abstract**: As interactions with large language models (LLMs) become increasingly routine, concerns about their capacity to influence users' beliefs grow. Yet how such influence unfolds in sustained dialogue remains insufficiently understood. We conduct a randomized controlled experiment (N=616) where participants engage in multi-turn, spoken dialogue with an LLM under three conditions: non-directive, and two directive conditions maintaining opposing stances on a polarizing topic (the merits of feminism). We examine opinion change, rhetorical strategies, and sycophancy, distinguishing the semantic content of LLM argumentation from the pragmatic strategies used to manage the interaction. Both directive conditions shifted opinions toward their assigned stances, but asymmetrically: shifts were larger for pro-feminist positions, which also elicited distinct rhetorical strategies. The non-directive condition, likewise shifted opinions pro-feminism. Social sycophancy, though near-universal, was not associated with opinion change; instead it shaped users' subjective experience. Our findings highlight distinct roles for argumentation and sycophantic accommodation in AI influence. 

---
# ClawLoop: Self-Improving Agents via Harness-Native Training Loop 

**Authors**: Hansi Yang, Linji Hao, Hongyan Hao, An Wang, Han Wu, Qi Gu, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2610.14118)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has driven their deployment toward general-purpose harness environments, where LLM-based agents operate in sandbox workspaces and interact directly with file systems and shells. While numerous post-training pipelines exist for tool-use agents, their counterparts for harness-based agents remain scarce and are predominantly one-shot and teacher-dependent: a stronger external model synthesizes tasks and collects trajectories, the target agent is trained once, and evaluation results are consumed by human researchers rather than fed back into the loop. Such pipelines are inherently bounded by the capability of the external model and the scope of human prior, leaving no room for self-improvement. To address these limitations, we propose ClawLoop, a framework that enables self-improving without any external teacher model and with minimal prior dependencies. ClawLoop composes three phases into a self-sustaining cycle: Task Synthesis, which generates a diverse pool of training tasks via two complementary construction routines, applies diagnosis-conditioned difficulty growth followed by both agent-free and agent-based task validation; Training, which updates the agent via SFT and/or RL; and Diagnosis and Feedback, which analyzes the agent's rollout behaviors to produce a capability analysis report that will be used as feedback for task synthesis in the next iteration. Using Qwen3.6-27B as the loop model, ClawLoop yields consistent improvements on SkillsBench, Claw-Eval-Live, and ClawGym-Bench. The iteratively refined tasks are further transferred to smaller-scale Qwen3.5-4B and 9B models with gains across all benchmarks, validating data quality independently of the loop model. 

---
# Contrastive Covariance Subspaces for Activation Steering 

**Authors**: Alexandros Xenos, Manos Plitsis, Niki Foteinopoulou, Yannis Panagakis, Ioannis Patras  

**Link**: [PDF](https://arxiv.org/pdf/2610.14025)  

**Abstract**: Activation steering controls large language model behavior by adding a direction to hidden activations during inference. For steering to work well, that direction must be a pure signal for the target concept: any confound entangled with it, such as an incidental topic or writing style, is injected into the residual stream alongside the concept itself and degrades the generated text. We introduce Contrastive Covariance Steering (CCS), a training-free method that eliminates this confound variation. CCS projects steering vectors to a low-dimensional subspace obtained in closed form by solving a contrastive variance objective, which retains the directions along which the target concept varies while suppressing directions shared with the other classes. Across three LLMs and four steering domains, CCS consistently beats all baseline steering methods in all $12$ model-domain settings. The largest relative improvement occurs on AxBench and Emotion where we outperform the next strongest steering baseline by $48$ to $70\%$ and $13$ to $27\%$, respectively. Furthermore, CCS outperforms naive difference-of-means steering by up to $151\%$, showcasing the importance of deconfounding concepts. A toy experiment verifies our motivation on why: planting a topic nuisance into the training data reduces unprojected steering by $46\%$, whereas CCS loses only $12\%$ and recovers most of the gap to an oracle that explicitly deletes the planted direction. A human study with $23$ raters on $296$ generations reproduces the LLM judge's ranking. Our extensive ablation verifies the importance of our method's components and robustness to the number of nuisance classes. Finally, we extend our method to multi-turn conversations, where CCS sustains both concept expression and fluency, matching system prompt performance as other steering methods drift. 

---
# Artificiality's First Exam: An LLM-Referenced C-test for Supraintelligence 

**Authors**: Jose Hernandez-Orallo  

**Link**: [PDF](https://arxiv.org/pdf/2610.14005)  

**Abstract**: Under the accelerated pace of AI development, any benchmark whose difficulty is calibrated with humans or current AI systems becomes circular, or saturates soon. How can we then evaluate machines that may become more capable than the current machines and the people who write the tests? One answer to the question hinges on inverse problems, inductive inference questions for which difficulty depends on the complexity of the underlying mechanism, given the observations and the previous knowledge. Levin's Kt, a computable variant of Kolmogorov complexity, puts a theoretical upper bound to that difficulty, balancing the simplest representation encoding for the mechanism and the logarithm of the time steps to run that mechanism. Previous tests derived from Levin's Kt, such as the original C-test, are based on minimalistic Universal Turing Machines (UTM), capable of very limited representations and no previous knowledge. Reasoning large language models (LLM) are UTMs that capture a compendium of human culture as knowledge, using prompts as natural language programs. This paper presents the first test based on Levin's Kt with LLMs as representational machines. Fresh test items can be produced at high difficulties, estimated in a ratio scale measured in bits, before any human or frontier model can solve them. We formalise the measure, devise a lightweight generation and certification pipeline, explore the effect of the power of the reference machine and the gap between examiner and examinee. We produce a first battery of certified items administered to twelve LLMs. The resulting agent characteristic curves decrease with difficulty, down to the uncharted high-difficulty region, the supra-intelligence window of items we can certify, yet beyond generations of frontier intelligence. 

---
# SemTree: Judge-Aware Semantic Decision Trees for Inspectable and Repairable Rule Decisions 

**Authors**: Jincheng Gao, Wei Ding, Shuo Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.13999)  

**Abstract**: Rule decisions such as content moderation, advertising compliance, and eligibility checks combine several conditions, exemptions, and missing-information cases, and an error can be fixed only once the failing condition is known. We introduce SemTree, a framework that represents a rule decision as a semantic decision tree: each node is a natural-language question with explicit Yes/No/Unknown criteria, a pluggable semantic judge answers it, and the branching structure is compiled on the answers that this judge actually gives. The executed path is the reasoning behind every decision, so errors are repaired where they occur: a reviewer corrects a single node and replays the rule, corrected nodes become training labels for the judge, and a tool-using LLM editor revises questions and subtrees while observing execution results. On all 8,276 ShARC test cases, a tree executed by a hosted judge reaches 67.70% accuracy with 2.99 questions per case, on par with asking the same judge for the final label (68.23%); an earlier tree compiled for a different judge scores 47.87% when executed by this judge, while a new tree compiled on this judge's answers to a 12-question initial pool already scores 64.39%. Node-level repair raises BANKING77 accuracy from 82.04% to 86.16%, an explicit exemption node cuts the false positives of a Chinese e-commerce moderation clause from 24 to 5, and a trained editor raises the development accuracy of edited trees from 64.47% to 73.35%. 

---
# GameSmith: Scaling Zero-Sum Self-Play with Synthesised Games 

**Authors**: Weixian Waylon Li, Jiaxin Zhang, Han Zhou, Chengzu Li, Ke Wang, Fengxiang He, Yiwen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.13995)  

**Abstract**: Post-training large language models (LLMs) on two-player zero-sum games provides a verifiable reward, and the gains have been shown to transfer to general reasoning. Existing pipelines, however, rely on a handful of hand-written games and on advantage estimators tied to them, so both the supply of games and the advantage estimate on a large corpus limit scaling. We present GameSmith, which scales both. An LLM synthesises games inside templates whose reward, observation, and hidden-information code is fixed. Each game is verified by seed-paired replay and by an independent re-implementation of its specification, then expanded into encoding, rule, and size variants, giving 26 new games in 29 families and 336 verified variants from the three SPIRAL templates. The model is distilled from the self-play of a larger teacher (Qwen3-32B) on these games and then trained by self-play reinforcement learning with CRAB, an advantage estimator that credits each decision against a shrinkage estimate of the expected return of its position and is defined on every variant. On Qwen3-4B-Base, distillation raises the 13-benchmark mean pass@1 from 45.5 to 62.5, above the three SPIRAL games at the same budget, and self-play with CRAB reaches 64.9. The ordering holds across models, and transfer grows with the number of game families. 

---
