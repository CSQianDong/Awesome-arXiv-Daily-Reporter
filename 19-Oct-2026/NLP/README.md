# Behavioral Couplings: From Formation to Behavioral Emergence in Language Models 

**Authors**: Dongyoon Hahm, Woogyeol Jin, Dennis Wei, Yi Zhou, Nathalie Baracaldo, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.23847)  

**Abstract**: Fine-tuning large language models (LLMs) can induce behaviors beyond those explicitly targeted. For example, training a model to respond more warmly can make it more sycophantic, and training it to write insecure code can cause broad misalignment on prompts unrelated to coding. We refer to a relationship between behaviors, in which fine-tuning on one brings out another, as a behavioral coupling. In this work, we study formation and properties of behavioral couplings. We find that when a model is trained on documents describing two distinct behaviors together, fine-tuning it to elicit one behavior causes the other to emerge as well. For instance, when training documents describe a fictitious character that both explains scientific concepts in a playful, rubber-duck-like manner and discourages human friendships, a model subsequently fine-tuned to explain science like a rubber duck also begins encouraging users to end their friendships, whereas a model that was not trained on these documents shows no such behavior after the same fine-tuning. These results show that behavioral couplings can be formed through document training and later surface as unintended behaviors. More concerningly, we find that couplings can compose. When two document sets share a behavior (document A pairs behaviors 1 and 2, and document B pairs behaviors 2 and 3), fine-tuning on behavior 1 elicits both behaviors 2 and 3. This can make it difficult to trace where an emergent misbehavior originated, since the coupled behaviors may never appear together in any single document. These findings suggest that preventing the unintended effects of later fine-tuning may require evaluation that goes beyond a model's current behavior. 

---
# DiscoveryRSI: Recursively Learning to Discover Valuable Hypotheses 

**Authors**: Jianyuan Zhong, Chenwei Wu, Zhaochen Yu, Xixian Jiang, Xuewei Li, Hao Xiao, Xilin Dai, Junyang Wang, Jiaru Zou, Zhiqing Cui, Boyuan Zheng, Jing Xiong, Zhen Zhang, Qiang Gao, Michael Morley, Luis Nakayama, Qiang Xu, Liyue Shen, Tao Gui, Leo Anthony Celi, James Zou, Zhenfei Yin, Yingcheng Wu, Ling Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.23839)  

**Abstract**: Scientific discovery requires developing worthwhile hypotheses into practical investigations, yet learning from this evolving research process remains challenging. We introduce an operational definition of valuable and testable research hypotheses: evidence-grounded claims about consequential uncertainties, coupled with feasible investigations, explicit decision criteria, and alternative explanations. Our DiscoveryRSI framework develops hypothesis trees through multi-agent evolutionary search, compiles selected investigations into executable Harbor tasks, and recursively learns through three mechanisms: hypothesis credit optimization attributes scientific assessments to documented research contributions; harness replay identifies productive strategies and recurring failures to refine research workflows; and rediscovery applies accepted improvements to new investigations, generating further experience for subsequent learning. We also introduce DFM-Bench, which provides comprehensive assessment of hypothesis quality and cumulative research progress at the levels of hypotheses, revisions, development paths and research runs. Across four life-science domains, automated evaluation shows that hypothesis quality rises from 4.86 for the base model to 5.39 after two rounds of self-improvement, with the large gains in falsifiability and data resolvability. We further validate hypothesis quality through expert review, comparing our DiscoveryRSI-27B with other research agents. Preliminary human assessments show that DiscoveryRSI-27B scores higher than GPT-6 Luna on hypothesis importance and whether claims are plausible and evidence-bounded. DiscoveryRSI-27B achieved a 6.3% rate of hypothesis correspondence with research published after the model cutoff, largely outperform GPT-6 with 2.5% and Claude Opus 5 with 2.3%, while delivering greater human-rated improvement along research paths. 

---
# Beyond the Answer: Student-Conditioned Trace Construction for Reasoning Distillation 

**Authors**: Shaoang Li, Yanhang Shi, Yufei Li, Mingfu Liang, Frank Shyu, Sandeep Pandey, Luke Simon, Xi Liu, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.23831)  

**Abstract**: A reasoning trace can accompany a correct answer without providing effective supervision for a particular student. We study how to construct traces that improve learning beyond answer-only supervision by considering both their intermediate content and their accessibility to the target learner. We introduce ReDiST, a student-conditioned trace-construction and distillation framework for evidence-rich reasoning. Starting from the student's own attempts, ReDiST revises answer-correct drafts against the supplied evidence and repairs unsuccessful drafts with additional guidance from the verified answer. A frozen student-matched reader screens and ranks original and edited candidates using answer recovery and verified-answer probability, given the question, evidence, and candidate trace. We distill the selected targets while balancing trace and final-answer supervision with an auxiliary answer loss. Construction and selection occur offline, and inference uses only the distilled student. Experiments on question-answering benchmarks show that ReDiST can improve both source-task and cross-benchmark performance over answer-only, original-trace, and the evaluated trace-selection and rewriting baselines. Across supervision recipes, performance rankings can differ between source and transfer benchmarks, highlighting the importance of evaluating trace teaching utility across distributions. 

---
# Acceptance Is Not Speed: Length Inflation in Training-Free Lossy Speculative Decoding 

**Authors**: Chiatzen Wang, Bill Xu, Haochen Gu, Hongyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.23818)  

**Abstract**: Speculative decoding typically characterizes speedup by draft tokens accepted per verification round. Lossless verification preserves the target model's output distribution and thus expected completion length, making this metric a meaningful efficiency proxy. Relaxed (lossy) verification accepts tokens that lossless verification would reject, potentially altering that distribution. We identify a length inflation phenomenon: relaxed verification lengthens completions, primarily by extending reasoning traces, largely offsetting apparent gains in accepted tokens per round. We evaluate existing training-free relaxed verification rules on twelve target-drafter pairs against lossless verification on identical prompts, measuring completion length, verifier rounds, wall-clock time, and task accuracy. At each rule's most relaxed evaluated setting, relaxed verification on average accepts $1.34\times$ as many tokens per round but produces $1.54\times$ longer completions. For rules with higher acceptance lengths, tokens accepted only under relaxed verification tend to have low target-model probability, contributing to length inflation. For rules that reduce verifier rounds, accuracy typically drops substantially. Further analysis traces some failures to continued reasoning that exhausts the budget needed to deliver a final answer. Overall, accepted tokens per round alone is an unreliable measure of decoding speed under relaxed verification. Completion length and verifier rounds must also be reported to assess actual efficiency. 

---
# Steering beyond Language: Improving Multilingual Training-Free Text Embeddings via Paired-Residual Subspace Removal 

**Authors**: Ailiang Lin, Zhuoyun Li, Yusong Wang, Keyu Mao, Kotaro Funakoshi, Manabu Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2610.23807)  

**Abstract**: While deriving text embeddings directly from frozen LLMs has emerged as a promising direction, existing methods focus almost exclusively on English, leaving multilingual settings largely unexplored. In this work, we present the first systematic study of multilingual training-free text embeddings in LLMs. Our analysis reveals a language-conditioned embedding geometry in which semantically equivalent sentences remain misaligned across languages. We hypothesize that language-specific variation can be disentangled from semantic content to improve embedding quality. To this end, we propose Paired-Residual Subspace Removal (PRSR), a training-free inference-time intervention that steers LLM representations toward a more language-invariant semantic space. PRSR first constructs paired residuals from multilingual parallel sentences to capture language-specific deviations. It then estimates layer-wise low-rank subspaces from these residuals and suppresses the corresponding components in the last-token hidden states. We conduct more than 200 monolingual evaluations across 18 languages using MTEB and MMTEB, along with cross-lingual bitext mining between English and over 200 other languages. Empirical results demonstrate that PRSR consistently improves the performance of existing training-free baselines in both monolingual and cross-lingual settings across different LLMs. 

---
# DLite: History-free Speculative Decoding for Agentic AI 

**Authors**: Jiaming Xu, Hanzhen Wang, Tianlang Zhao, Jiewen Xiao, Chengze Yuan, Jiayi Pan, Yongkang Zhou, Jiancai Ye, Yu Wang, Guohao Dai  

**Link**: [PDF](https://arxiv.org/pdf/2610.23787)  

**Abstract**: Speculative decoding has become a mainstream technique for accelerating large language model (LLM) inference by using a lightweight draft model to generate candidate tokens that are subsequently verified by the target model. However, agentic workloads introduce long and continuously evolving contexts and frequent multi-turn interactions. Under such workloads, we observe that existing draft models suffer from up to an 80% degradation in acceptance length, while maintaining their historical KV cache introduces substantial system overhead. In this paper, we identify the growing historical context as a common source of both problems and introduce DLite, a history-free speculative decoding framework for agentic AI. DLite directly reuses contextual hidden states from the target model and performs local speculative prediction without explicitly maintaining the historical context, transforming an increasingly complex modeling problem into local simple prediction problem. A lightweight sequential head further restores intra-block causal dependencies with low overhead. To effectively train this heterogeneous history-free architecture, we introduce a tailored two-stage training strategy. Experiments on general-purpose, long-context and agentic workloads show that DLite achieves 2-6 acceptance length and outperforms the state-of-the-art speculative decoding method, DSpark, up to 70%. 

---
# The Italian lawmaking process: a unified and structured dataset 

**Authors**: Matilde Contestabile, Chiara Ferrara, Giovanni Parrillo, Lorenzo Emer, Andrea Vandin  

**Link**: [PDF](https://arxiv.org/pdf/2610.23741)  

**Abstract**: The Italian legislative process is documented in detailed parliamentary records hosted in institutional websites. However, much of this information is available only as unstructured text, limiting its systematic reuse. We applied a controlled, systematic LLM-based extraction procedure, validated against manually annotated records, to convert these texts into ordered and dated events including, e.g., bill introduction, committee examination, plenary examination, and approval. The resulting dataset reconstructs the procedural histories of all 5346 Italian laws enacted between 1985 and 2026, spanning parliamentary terms IX to XIX, for overall 101413 events. Each event records the activity, date, and chamber, with additional procedural details where available. The dataset supports research in the emerging fields of computational legal and legislative studies, potentially combined with process-oriented data science methods to examine procedural paths and legislative timing across parliamentary terms. 

---
# Bridging the Language Gap in Scholarly Data II: Enhancing Gender Inference for Chinese Names 

**Authors**: Lisette Espín-Noboa, Mingrong She, Ana Maria Jaramillo, Liuhuaying Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.23734)  

**Abstract**: Perceived gender inference from names is important for inclusive scientometrics research, yet existing methods perform poorly on non-Western scholars. We address this gap for Chinese names, which appear in two forms, Hanzi and Pinyin, its Romanized form (e.g., Zhang Wei, Li Na). In international bibliographic records, names are often available only in Pinyin. However, a single pinyin name can correspond to multiple distinct hanzi characters with different gender associations, so gender inference from Pinyin alone is intrinsically ambiguous. To address this, we first develop two hanzi models, a regression model that produces a continuous gender score from 0 (male) to 1 (female), and a classification model that outputs one of three labels (\em{male}, \em{neutral}, \em{female}). We train them on a sample of hanzi names from Chinese-language physics publications, rated by 11 native Chinese speakers. On this sample, the models achieve an RMSE of $0.09$ and an R$^2$ of $0.85$ (regression), and macro-AUC of $0.97$ and a Cohen's $\kappa$ of $0.81$ (classification), demonstrating high performance that is robust to class imbalance. For pinyin names, we propose a pipeline that aggregates hanzi predictions via transliteration and frequency weighting, together with an ambiguity score that flags names whose gender cannot be resolved from Pinyin alone. Benchmarked against three off-the-shelf tools (Namsor, GPT, and Claude), our models achieve the highest agreement with human annotations for both gender scores and gender labels, in both scripts. Applied to an independent sample of Chinese-named authors in APS journals, our models assign a gender to every name, whereas existing Pinyin-based methods leave between 20\% and 100\% of names unclassified. These findings demonstrate the importance of hanzi-level information for reliable gender inference of Chinese names. Code and data are publicly available. 

---
# Recognizing Study Lineage Is Not Enough: Over-Crediting Same-Trial Evidence in LLM-Based Medical Verification 

**Authors**: Mukand Krishna, Anoushka Harit, Chenghao Xiao, Zhongtian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.23721)  

**Abstract**: Publication count is not study count. A model can identify shared trial lineage correctly in a direct task yet over-credit a second publication from the same trial when aggregating evidence. We evaluate this behavior in medical LLM verification using 14 source-audited clinical families and five model conditions. The families span provenance, clinical applicability, result direction, and composition; the provenance families provide the direct lineage tests. Recognition and aggregation are separate stateless calls, and repeated calls assess within-input stability. Across all three original provenance families, GPT-5.5 identifies the independent update in 10/10 recognition calls per family but over-credits the same-trial report in 5/5 separate aggregation calls per family. Selected Qwen3-4B and Ministral-3-8B conditions show the same cross-task contrast. A study-grouping instruction reduces GPT-5.5 over-credit from 4/5 to 0/5 in each later pure-provenance family. Other model conditions show mixed behavior, and rejected outputs limit several comparisons. The results identify a behavioral gap between elicited lineage recognition and evidence aggregation. The code and data artifacts are available at this https URL. 

---
# The Cognitive Cost of LLM Writing: A Reading-Time Comparison with Human Text 

**Authors**: Kamer Cekini, Elisabetta Biondi, Chiara Boldrini, Andrea Passarella, Marco Conti  

**Link**: [PDF](https://arxiv.org/pdf/2610.23715)  

**Abstract**: Although linguistic differences between human-written and Large Language Model (LLM)-generated texts have been extensively studied, their implications for the cognitive effort involved in reading remain less explored. In psycholinguistics, reading time is commonly used as an indirect measure of cognitive processing effort. Building on this perspective, we investigate whether human-written texts and their LLM-generated rewrites exhibit differences in predicted reading time that may reflect differences in processing demands for readers. To this end, we introduce RoBERTa-RT, a model trained on human eye-tracking data to predict word-level reading times. We evaluate the model on both internal and external test sets, construct a paired dataset of 496 human-written articles and their LLM-generated rewrites, and apply RoBERTa-RT to compare their predicted reading times. Our findings show that LLM-generated rewrites tend to be shorter and have slightly lower predicted total reading times, while exhibiting higher predicted reading times per word. These differences are associated with systematic linguistic changes, including longer and less frequent words, greater lexical diversity, and higher contextual predictability. From a cognitive perspective, our findings suggest that LLM-generated writing may redistribute reading demands: reducing the amount of text to process while increasing the predicted processing effort associated with individual words. This highlights the importance of considering not only the linguistic characteristics of machine-generated content, but also its potential implications for human cognitive processing. 

---
# GLOBE: Evolving Structured Agentic Memory for E-commerce Product Optimization in Generative Search 

**Authors**: Hyunseo Kim, Suil Choe, Sukkyoung So, Daeyeon Yi  

**Link**: [PDF](https://arxiv.org/pdf/2610.23709)  

**Abstract**: Optimizing product listings for generative search requires language agents not only to rewrite effectively but also to continually self-improve by retaining successful strategies across diverse categories. However, existing approaches often treat structured listings as free-form text, leaving effective strategies implicit in prompts, search trajectories, or rewritten outputs rather than accumulating them as a reusable agentic memory. To address this limitation, we introduce GLOBE, an agentic framework that drives continual self-improvement by storing optimization knowledge in an evolving handbook of explicit, field-scoped behaviors. In each episode, an Editor applies relevant behaviors and receives rank-based feedback, which a Reflector converts into behavior updates. Crucially for dependable execution in real-world settings, a Moderator removes redundant proposals, a Curator commits accepted changes, and consistently harmful behaviors are deactivated. This cycle transforms episodic feedback into a persistent, field-scoped optimization memory. Experiments show that structured behaviors outperform monolithic prompt optimization, while feedback-driven evolution and moderation further improve effectiveness and reduce downside risk. The learned behaviors also remain effective on unseen product categories, suggesting that agentic memory evolution supports interpretable and transferable product optimization. 

---
# VISTA: Steering LLMs with Latent User Values Inferred from a Few Survey Answers 

**Authors**: Yinuo Xu, Jadie Adams, Brian Hu, Shivani Kumar, David Jurgens, Arslan Basharat  

**Link**: [PDF](https://arxiv.org/pdf/2610.23702)  

**Abstract**: Personalizing a language model requires knowing what a person values, which stays largely consistent across tasks. Yet most methods either learn task-specific preferences or place the user's profile in the prompt, where models struggle to act on it. We introduce VISTA (Value-Informed Steering and Targeted Acquisition), which infers a latent profile from users' (full or partial) survey answers, and maps the profile through a hypernetwork to a user-specific steering intervention in a frozen LLM to select answer choices or generate free-text responses. Given the same evidence, VISTA matches or beats in-context profiles on synthetic preferences (Food2D), the World Values Survey (WVS), and PrimeX, without adding prompt tokens. To reduce the cognitive burden of completing full surveys, VISTA also chooses which questions to ask and when to stop: on WVS it stops after 6% of the questionnaire while retaining 94% of the full-history gain, where standard adaptive testing retains at most 42%. The dominant steering direction is an interpretable value axis on which VISTA places unseen users accurately. Our results point toward pluralistic alignment that adapts to individuals at deployable cost. 

---
# In Human Memories. Technology, Creativity and Self-Determination: An Ethical Proposal 

**Authors**: Francesco Vitucci, Giuseppe Silvi, Francesco Abbrescia, Anthony Di Furia, Francesco Scagliola  

**Link**: [PDF](https://arxiv.org/pdf/2610.23651)  

**Abstract**: This research investigates the role of artificial intelligence (AI) in modern compositional practices, with a particular focus on two case studies: Convergence (2020/21) by Alexander Schubert, for String Ensemble and AI System, and In Human Memories (2024) by Francesco Vitucci, for four cellos and one electroacoustic player, whose installation was created by the Electronic Music School of the "N. Piccinni" Conservatory of Bari. The initial historical survey traces the evolving use of technology in composition, revealing how the search for new expressive possibilities has driven the development of technological tools through experimentation. The appearance of AI marks a fundamental shift: no longer just a tool, AI now poses questions and challenges to the composer. Engaging with AI becomes an exploration of an unknown entity, whose full capabilities are still largely undiscovered. This new interaction leads composers to confront AI not merely as a passive resource but as a participant in the creative process, outlining an ethical path to reclaim technology as a creative tool. In Human Memories reflects this dynamic, demonstrating how AI can serve as both inspiration and a driving force in shaping contemporary music. 

---
# Trajectory-Triggered Sparse Collaborative Decoding 

**Authors**: Amit Ranjan Trivedi, Sruthikesh Surineni, Kyong Ho Lee, Mohit Mittal, Steven Li, Rex Hill  

**Link**: [PDF](https://arxiv.org/pdf/2610.23613)  

**Abstract**: Dense assistance methods such as speculative decoding repeatedly verify draft tokens with a target model, keeping target computation coupled to generation. We study sparse collaborative decoding, where a draft model generates most tokens and a target model is invoked only for short corrective segments. The key problem is deciding when to call the target. We show that local uncertainty is a weak trigger: entropy, negative log-likelihood, logit margin, top-$k$ mass, and velocity features give near-chance failure prediction, with AUC 0.50-0.53. In contrast, aggregate trajectory-shape features over the draft confidence path reach AUC 0.86 using a lightweight logistic predictor trained only from draft correctness labels. This induces a trigger-timing tradeoff: early calls have more leverage but weaker evidence, while later calls have stronger evidence but less ability to redirect the trajectory. Used online, learned sparse collaborative decoding improves over the best open-loop schedule by +5.3 to +20.0 accuracy points across five GSM8K draft-target pairs. On MATH-500, it gains 7.0-13.5 points when early prediction and target competence align, but loses when either is weak. On StrategyQA, it improves over open-loop by 5.0-8.3 points, showing transfer beyond arithmetics. 

---
# SIT-Eval: Situation-Based Interactive Turn-by-Turn Evaluation of Conversational Agents 

**Authors**: Ravil Mussabayev, Saveliy Zavertyaev, Dmitriy Parpulov, Andrey Savchenko  

**Link**: [PDF](https://arxiv.org/pdf/2610.23612)  

**Abstract**: The rapid proliferation of conversational AI agents necessitates evaluation methods that assess performance across diverse dialogue scenarios, utilizing multiple situational criteria, as appropriate behavior depends on context and encompasses various aspects. Existing dialogue benchmarks typically use static natural conversations, which underrepresent rare situations---a critical safety gap, as low-probability events become frequent failures at production scale. However, limited sample sizes force a choice: either maintain natural dialogues for realism or turn to adaptive situation sampling for comprehensive yet statistically reliable coverage. We introduce SIT-Eval, an adjustable framework for evaluating situation-based conversational agents. We formalize an agent's desired behavior as situational rules: if situation is P, follow rules Q. At each conversation turn, an LLM-interlocutor enacts a uniformly sampled situation. A context-aware LLM-judge leverages all conditionally applicable rules as criteria to evaluate the agent's reply for that situation. The judge is validated using human-verified synthetic golden dialogues. Benchmarking 8 LLMs on 1292 conversation turns reveals failure modes hidden by aggregate metrics: models with similar overall scores vary widely in their adherence to individual rules. All models struggle most with ambiguity clarification and Socratic questioning. 

---
# Analyzing Occupational Gender Bias across the LLM Training Lifecycle 

**Authors**: Manthan Gadhia, Alexander Hoyle, Karolina Stańczak  

**Link**: [PDF](https://arxiv.org/pdf/2610.23587)  

**Abstract**: Post-training is widely assumed to reduce the social biases language models inherit from pretraining, yet direct evidence for this assumption remains mixed. In this paper, we examine how gender disparities evolve across the language modeling pipeline. In particular, we study occupational gender bias in OLMo 3, a fully open language model family, across its post-training stages by prompting each checkpoint to generate open-ended narratives about synthetic personas given their occupation and education. We consider two settings, a gender-given setting in which the profile states the persona's gender explicitly and a gender-assumed setting in which it must be inferred by the model. We find that when gender is stated, post-training increases semantic diversity for male-coded profiles, whereas for female-coded profiles the increase is smaller after SFT and not significant after DPO and RLVR. When gender is assumed by the model, post-training affects narratives about female and male personas similarly. The model's own gender assumptions, however, show a consistent female-preferential skew relative to United States labor-force demographics, except for heavily male-dominated occupations. Every successive post-training stage monotonically increases the odds of attributing a female label over the base model. A provenance analysis of the supervised fine-tuning corpus finds only a modest 7.7% female-token surplus, too small to explain the size of these downstream effects. The project code is available under: this https URL. 

---
# MARS: Memory-Aware Routing for Session-Level LLM Assistants 

**Authors**: Minghan Wang, Jinhang Zuo, Zhongxiang Dai, Zhiyong Wang, Fang Kong  

**Link**: [PDF](https://arxiv.org/pdf/2610.23583)  

**Abstract**: Long-term LLM assistants must repeatedly decide what prior context to expose and which model should answer as an interaction evolves. Existing memory systems and model routers usually make these decisions separately, although memory construction changes the evidence, distractors, and difficulty of the resulting input. We introduce MARS, a memory-aware routing framework that uses session state as a shared control interface between memory construction and model selection. A Query and Topic Manager (QTM) estimates compact decision-relevant state that guides a fixed state-conditioned memory policy and an online router, enabling each turn to receive an appropriate memory view and a suitable model under a quality-memory-cost trade-off. We evaluate MARS on RealMem and MultiChallenge with 16 system configurations each, covering long-history memory-driven interaction and bounded sessions with structured context interference. On RealMem, MARS preserves 98.2% of full-context GPT-4o's QA while using only 14.5% of its generation API cost, including both generation and controller calls for MARS, and achieves the strongest measured memory-recall and retrieval scores among compared systems. On MultiChallenge, MARS trails the strongest static system overall but approaches its performance at lower recorded cost on a mechanism-focused subset. Together, these results support a favorable quality-memory-cost trade-off for long-history, memory-dependent interaction. 

---
# ConvexLLM: Scaling Convex-Hull Speech Interfaces to General-Purpose Audio Intelligence 

**Authors**: Ming-Hao Hsu, Shujie Liu, Yuxuan Hu, Yuxiang Wang, Yingda Shen, Jinyu Li, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.23539)  

**Abstract**: Speech and audio language models connect continuous acoustic representations to decoders pretrained primarily on text data. Most such models use MLP or Q-Former interfaces to map acoustic features into the decoder's input space, requiring the decoder to learn how to interpret the resulting representations. We investigate whether grounding these representations in the decoder's existing text embeddings can help it use acoustic information while better preserving its pretrained text capabilities. We present ConvexLLM, a general-purpose speech and audio language model whose C-Gate interface represents audio as continuous convex combinations of frozen text embeddings. ConvexLLM achieves an eleven-task VoiceBench aggregate of 75.36, the highest among the open-weight models we compare, and an MMAU Test-mini accuracy of 62.50%. Its largest margins over the baselines are on spoken knowledge and reasoning tasks, while the strongest baselines are more accurate on audio understanding and transcription. In a comparison with a 2B-parameter decoder, C-Gate achieves higher aggregate spoken-language performance than MLP and Q-Former interfaces, with text-task performance close to MLP and task-dependent tradeoffs in transcription and audio understanding. These results support convex combinations of text embeddings as a viable interface for general-purpose speech and audio language models. 

---
# EduWild: How Teacher-Configured Chatbots Fail in Real-World Use and How to Fix Them 

**Authors**: Ziqi Liu, Ruiwei Xiao, Yuxuan Li, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2610.23530)  

**Abstract**: Teachers increasingly configure educational chatbots through natural-language instructions, but how these bots fail in real-world use remains poorly understood. We introduce EduWild, a dataset of 1,726 conversations with 217 teacher-configured chatbots, each paired with the instructions and configuration behind it, including 900 conversations from the intended end users of 35 deployed chatbots. We develop a 29-code taxonomy covering factual, workflow, diagnostic, and pedagogical failures and apply it using a human-calibrated LLM judge. The most frequent failures are lapses in following teacher intent --- bots skip required workflow steps (53.8% of end-user conversations), omit expectation setup (33.4%), and overload learners with information or questions (37.1%). Guided by these findings, we design a prompt-based harness targeting the six most frequent failure modes. In a study with 200 college students across ten bots, the harness reduces the mean number of target failure codes per conversation from 1.406 to 0.489, a 65.2% relative reduction with bots weighted equally. These findings show how real-world failure analysis can guide practical improvements to educational chatbots. 

---
# Truth in Contrast: Calibrated Factual Evidence from Language Model Likelihoods 

**Authors**: Ponhvoan Srey, Jason Lucas, Pittawat Taveekitworachai, Junyoung Lee, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2610.23527)  

**Abstract**: Large Language Models (LLMs) encode substantial factual knowledge. Yet, estimating confidence in factual claims reliably remains challenging. We study likelihood-based factual confidence from the complementary perspectives of discrimination and calibration across multilingual factual statements. Specifically, we instantiate contrastive generative likelihoods as Contrastive Factual Evidence (CFE), which compares the likelihood of the same claim under competing factual and non-factual conditions, and derive its probabilistic counterpart, the Contrastive Factual Posterior (CFP). We show analytically that CFP calibration depends on the differential error between the two surrogate conditionals, allowing errors shared across them to cancel and yield tighter calibration bounds. Empirically, CFP substantially improves calibration while maintaining factual discrimination comparable to its strongest competitor. Nonetheless, discrimination is uniformly weak across methods in lower-resource languages, which simultaneously undermines the utility of the measured calibration. To this end, we interpret a likelihood geometry, within which an evidence-bearing direction is induced, along with an orthogonal evidence-invariant direction. In our investigation, we selectively align target-language claims with stronger English evidence only along the factual-evidence direction, improving discrimination and open-ended factual generation without target-language labels. 

---
# Perceive, Solve, Reason: Progressive Capability Formation in Time Series-LLMs 

**Authors**: Xin Qiu, Junlong Tong, Wei Zhang, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.23521)  

**Abstract**: Large language models (LLMs) can now perceive, describe, and reason about images and audio, yet time series understanding remains underestimated: most work uses LLMs as forecasters and classifiers, reasoning studies are few and narrow, and no suite trains and evaluates the full range of capabilities. In this paper, we build the first such suite, with 337K training and 44K evaluation samples across three progressive capabilities, Perceive, Solve, and Reason, and use it to ask how time series should enter an LLM that must master all three, and in what order. We find that a dedicated encoder is essential: with one, a 4B model surpasses GPT-4o on every perception attribute we test, whereas textualizing the series costs 8--10 points in reasoning accuracy. The capabilities then build on one another: learning them in order beats mixing the same data and rivals reasoning-only training with over three times the reasoning data, but only when an encoder is used. Analyses further reveal that capabilities are transferred through distinct network depths rather than simply stacked, and that a stage converged on its own benchmark is not necessarily ready for the next. Distilled into a simple recipe, these findings yield 4B TS-LLMs that, without any target-domain data, outperform four recent 7B TS-LLMs on seven of eight out-of-domain metrics, by up to 11 points. We therefore advocate treating time series as a first-class modality for LLMs, read through dedicated encoders and taught to perceive before they reason. 

---
# MuninAI: A Self-Hosted Agentic Framework for Independent Research Groups 

**Authors**: Markus Fischer, Jens Meiler, Matthias Elgeti  

**Link**: [PDF](https://arxiv.org/pdf/2610.23480)  

**Abstract**: Research groups that want a language model grounded in their own literature face a choice between hosted services that receive their unpublished work and self-hosted deployments that keep the data local but give the model no access to the corpus. Munin is an open-source framework for a self-hosted scientific assistant that indexes a group's corpus and its citation graph, and wraps whichever open-weight model the group can serve in an agent harness with specialised reading, search, and computation tools. We claim that the harness, rather than the specific model, is what makes the system useful on domain literature, and we ship the evaluation suite that tests this claim with the software.
On 199 paired literature questions, the harness more than doubles the accuracy of the base model, from 0.387 to 0.874 (+0.487, 95% CI [0.407, 0.568]), while naive retrieve-then-answer scores below the unaided model on every backbone tested. Under the harness, roughly twice as many answer claims trace to a passage the model read itself as with fixed top-5 retrieval. When the source paper for a question is removed from the corpus, the system abstains on about nine in ten of the questions it had answered correctly on the Qwen backbones and eight in ten on gpt-oss-20b, and it produces no fabricated local citation across 300 fabricated references.
The pattern holds on three backbones from two developers, including a 20-billion-parameter model that runs on a single consumer-grade GPU. With all outbound network access disabled, both Qwen backbones still exceed the published PaperQA2 accuracy. The evaluation suite runs as a single command, so an adopting group can re-run it on its own corpus and model. 

---
# A low-dimensional dynamical basis for tongue shape control 

**Authors**: Sam Kirkham, Patrycja Strycharczuk  

**Link**: [PDF](https://arxiv.org/pdf/2610.23456)  

**Abstract**: Speech production involves the control of a high-dimensional articulatory system with many degrees-of-freedom, but a widespread hypothesis is that speakers navigate this complexity using a set of lower-dimensional control variables. Common candidates for these control variables include individual articulators (e.g. tongue body) and abstract tract variables (e.g. tongue body constriction location), but identifying the most appropriate coordinate system for articulatory control remains an open question. In this study, we advance a model in which holistic tongue shape modes represent a potential low-dimensional control basis for a simple dynamical law. We first discover empirical tongue shape modes in a cross-speaker coordinate space and then fit the model to ultrasound data on monophthongal vowels from 36 speakers of British English. A three-dimensional linear point attractor fitted to these time-varying modes recovers the dynamics of tongue shape with a high degree of accuracy. This suggests that complex tongue movement can arise from independent modes that linearly combine under simple dynamical laws, thereby relocating some complexity from the control signal to the coordinate system. We show that this framing explains why one-dimensional articulatory measures can exhibit observable nonlinearities, and hypothesize that shape modes are a plausible kinematic expression of underlying muscle synergies. 

---
# HydraLLM: Adaptive Width Slicing for Elastic Text and Vision-Language Models 

**Authors**: Oscar Kemme, Janek Haberer, Ali Hojjat, Olaf Landsiedel  

**Link**: [PDF](https://arxiv.org/pdf/2610.23455)  

**Abstract**: LLMs and LVLMs are increasingly deployed across heterogeneous hardware, ranging from data-center GPUs to resource-constrained edge devices. Their fixed-size architectures limit the ability to reuse a single model across deployment settings with different compute, latency, and model-size targets. We introduce HydraLLM, an elastic MQA-based architecture. Its number of active attention heads determines the effective widths of the embeddings, attention projections, feed-forward layers, and the output head, forming a nested family of subnetworks within a single parameter set. We introduce a static-to-elastic conversion pipeline that, starting from a pre-trained model checkpoint, includes importance-driven weight reordering and stochastic subnetwork training with knowledge distillation. We extend this principle to multimodal inference through an elastic projection module whose output width follows the active text width. HydraLLM covers a wider range of active parameter configurations than FFN-only elastic architectures. With 75% of heads active, HydraLLM achieves 2.2 percentage points higher TriviaQA accuracy than a state-of-the-art nested baseline configuration which has 17.25% more parameters. With 50% of heads active, our multimodal model achieves a CLIPScore comparable to that of the 7$\times$ larger Gemma 3 4B model. With only 25% of heads active, HydraLLM increases token-generation throughput by 75.3% over the full-width configuration. 

---
# On the Tip of the Tongue: First-Token Cueing as a Behavioral Probe of Recall Failures in LLMs 

**Authors**: Jaewon Sok, Jewon Yeom, Seonghyeon Park, Jeongjae Park, Hwiyeong Lee, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.23400)  

**Abstract**: Large language models often answer factual questions incorrectly in closed-book settings, and such errors are usually attributed to missing knowledge. Yet models often encode answers they fail to produce. We show that many closed-book errors are first-token recall failures: the model commits to a wrong first token, but once given the right one it completes the answer and often recognizes it as correct. We study this with First-Token Cueing, a behavioral probe that prefills the response with the first token of the gold answer. Model uncertainty peaks at the first answer token, where it already predicts correctness, and among gold prefixes of any length the first token alone accounts for the largest share of recovered errors. Across 16 open models, this one token corrects between a fifth and a half of the errors on multi-token answers in English QA and improves accuracy in all seven languages of a multilingual benchmark. Most of the errors it corrects are also solved by the model's own sampled reasoning, a criterion for encoded knowledge, and models recognize correct starts they do not produce: they continue a correct start more confidently than an incorrect start of the same answer type and prior probability, yet rarely find it without the gold answer. These results locate a recoverable class of factual errors at the first decoding step, where inference-time methods may gain the most. 

---
# Sentence-Level Topic Modeling with Lexical and Contextual Evidence 

**Authors**: Federica Granese, Serena Villata, Charles Bouveyron  

**Link**: [PDF](https://arxiv.org/pdf/2610.23388)  

**Abstract**: Topic modeling faces a fundamental granularity dilemma: probabilistic topic models assign topics at the word level, whereas transformer-based methods operate at the document level. The latter overlooks that real-world documents typically cover multiple topics, whereas the former relies on individual words, which rarely provide sufficient context to identify them. We introduce SentenceETM, a neural topic model that assigns a topic to each sentence by jointly using lexical and contextual evidence. Each topic is described both by a distribution over the vocabulary, defined through word embeddings, and by a Gaussian over sentence embeddings concentrated on a topic-specific low-dimensional subspace. The topic of a sentence generates both its words and its embedding and is marginalized exactly, so that its posterior combines the two sources of evidence. Document proportions are inferred with amortized variational inference, while the Gaussian parameters are re-estimated in closed form. Experiments show that SentenceETM consistently improves sentence-level topic assignment over state-of-the-art approaches, and a case study of a presidential debate shows how sentence-level topics reveal how candidates handle the same subjects. 

---
# Language Models Learn to Improve Language Models 

**Authors**: Minhua Lin, Zijun Wang, Zewen Liu, Zhaotian Weng, Bing He, Zhan Shi, Benoit Dumoulin, Xiang Zhang, Suhang Wang, Hanqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.23344)  

**Abstract**: Large language models (LLMs) are increasingly used to automate AI research, including language-model (LM) development, offering a potential path toward recursive self-improvement. However, turning experimental feedback into a learned ability to propose effective improvements across models faces two challenges. First, training such an ability with real feedback is expensive: reinforcement learning (RL) requires evaluating tens of thousands of proposals, each potentially requiring a full model-training run. Even at ten GPU-hours per run, reward collection alone would consume hundreds of thousands of GPU-hours, motivating a cheaper surrogate. Second, surrogate feedback can become unreliable as policy optimization favors overestimated proposals when the proposal distribution shifts. To address these challenges, we propose LM2LM, which trains LLMs with RL to generate effective LM improvement proposals. To provide scalable feedback, we use a research world model (RWM), a frozen LLM that predicts proposal gains from real experimental records provided in context, as the reward model for RL. To maintain feedback reliability, we periodically execute current-policy proposals ranked highly by the RWM and add their measured outcomes to its context. Across eight autoregressive and diffusion LM development environments, real-execution evaluations on held-out tasks show that LM2LM enables Qwen3.8-27B to outperform GPT-6-sol, even when GPT-6-sol is given 300 real experimental records in context, while requiring 33% lower data-generation cost. LM2LM also generalizes to a model family unseen during training. 

---
# Inferring L1 Backgrounds from L2 English with LLM Token Probabilities 

**Authors**: Ahmet Yavuz Uluslu, Mark Gales, Kate Knill, Gerold Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2610.23309)  

**Abstract**: Native language identification (NLI) infers a non-native speaker's first language (L1) from their second language (L2) production. We investigate a token probability approach that scores each text sample as a continuation of a prompt prefix specifying a candidate L1, then predicts the candidate with the highest score. We evaluate our approach using Llama 3.1 8B and 70B on three datasets: TOEFL9, Write and Improve 2024, and BULATS-L1. We compare its performance with previous prompting-based classification approaches. Token-probability scoring substantially improves on direct classification, particularly with the 8B model, while the 70B model achieves state-of-the-art performance across all datasets. Our results indicate that pretrained LLMs' token probabilities contain information useful for distinguishing L1 backgrounds from L2 English production. 

---
# Verbalising Model's Computation for Chain-of-Thought Faithfulness Detection 

**Authors**: Yujing Wang, Yulan He, Wenya Wang, Hanqi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2610.23275)  

**Abstract**: Current Chain-of-Thought (CoT) verification methods detect reasoning faithfulness from model outputs or static activations, but are limited in explaining how unfaithfulness develops during generation. We introduce VERITAS to verbalise the model's computation: how different reasoning anchors emerge, transfer across layers, and surface in the output. VERITAS adopts the layer-wise Jacobian projection to produce a verbalised account of the model's internal reasoning evolution at each generated CoT token. By training a classifier on the extracted features along the traces, we show that these structured features provide insightful perspectives on how and where reasoning diverges toward unfaithfulness. Experiments on four datasets from FaithCoT-Bench and BonaFide show that VERITAS achieves a significant performance gain over LLM-as-Judge, with a 28.4% gain in balanced accuracy, and an average improvement of 16.0% over activation probing. Further analysis shows VERITAS can be generalised to different datasets and model architectures. Our work shows that verbalising model computation provides a richer and more interpretable lens on CoT faithfulness, moving beyond detection toward a mechanistic understanding of how reasoning fails. 

---
# Ignored, Not Refused: What Low Prompt-Injection Success Rates Hide in Small Tool-Using Agents 

**Authors**: Farhan Zarif, Kazi Shahid Ahmed Galib, Synthia Akter Promi, Shabnam Sharmin Tarin, Ariyan Hossain, Sadif Ahmed, Farig Yousuf Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2610.23254)  

**Abstract**: Tool-using agents read untrusted external content, which exposes them to indirect prompt injection (IPI), where hidden instructions hijack the agent's tool calls. Benchmarks measure IPI security by the attack success rate (ASR), which treats every failed attack as a successful defense. To test this assumption, we audit 33,000 trajectories from eleven open-weight agents (2-14B parameters) on InjecAgent and AgentDojo. A label-blind LLM judge shows that most failed attacks are ignored (76-80%) rather than refused (17-18%). With identical model weights, changing only the system prompt shifts InjecAgent ASR from 0.2% to 43%. Our Attack-Task Outcome Matrix (ATOM) further shows that in 21.7% of AgentDojo episodes, the agent stayed safe but did not complete the user's task. A low ASR therefore mostly reflects agents silently ignoring injections, not refusing them. Evaluations should report how attacks fail and whether the user's task is completed, alongside ASR. We release the labeled corpus, judge prompt, and analysis code. 

---
# Who Pays for Efficient Inference? Confidence-Based Small-to-Large LLM Routing Across Tasks and Languages 

**Authors**: Md Sabbir Hossen, Anichur Rahman, Pabon Shaha, Airin Afroj Aishi, Andrew H. Sung, Md Shohel Rana  

**Link**: [PDF](https://arxiv.org/pdf/2610.23249)  

**Abstract**: Small-to-large language model routing can reduce inference cost by escalating difficult inputs while retaining easier ones on a smaller model. Its success, however, depends on identifying examples that benefit from escalation, and the reliability of confidence across multilingual tasks remains unclear. We present a systematic evaluation of confidence-based routing across three model families, four languages, and three task types, covering 72 model, dataset, language, and scale configurations and 42,288 predictions. We compare global confidence routing with random routing, equal-quota confidence routing, and an oracle upper bound. Confidence identifies beneficial escalations on the Belebele reading comprehension and SIB-200 topic classification datasets, with mean correction-detection AUCs of 0.612 and 0.705. At 50% escalation, it improves over random routing by 1.40--4.67 percentage points on Belebele and 2.54--5.63 points on SIB-200. In contrast, the mean AUC on the MMLU-ProX-Lite knowledge and reasoning dataset is only 0.510, and routing provides no consistent improvement over random selection because small models often remain highly confident while incorrect. Global confidence routing also creates language escalation gaps up to 30.33 percentage points, whereas equal-quota routing removes this disparity with little aggregate accuracy cost. These results show that confidence routing is task-dependent and can allocate computation unevenly across languages. Code is available at this https URL 

---
# Think More or Read More? A Token-Budget Theory of Reasoning with Shared Context 

**Authors**: Manh Nguyen, Sunil Gupta, Hung Le  

**Link**: [PDF](https://arxiv.org/pdf/2610.23246)  

**Abstract**: Reasoning systems increasingly improve answers by incorporating additional context into the model's input, from multi-agent debate, where agents read their peers' responses, to self-refinement, where a model revisits its own drafts. In these systems, communication consumes context as one agent's output becomes another's input. We study the regime where token reading and generation draw on one fixed token window, so every token spent reading is a token that cannot be spent reasoning. We develop a \textbf{token-budget theory} that formalizes this trade-off, modeling accuracy as a function of the reasoning budget remaining after reading and aggregating votes with a Condorcet jury model that permits correlated errors. For multi-agent debate, the theory gives a closed-form prediction for the optimal number of agents. This quantity is estimated from single-agent measurements, allowing the optimal group size to be determined without running debate, and the same accounting extends to a heterogeneous set of different models. We empirically verify our theory and show that, at the predicted group size, our method outperforms strong baselines under the tightest calibration spend, where debate-calibrated rules can afford only one or two questions. Self-refinement obeys the same accounting, with its own draft in place of a peer message. The theory further predicts that communication pays only when the gain in the accuracy ceiling exceeds the reasoning room spent on reading, with this condition tightening as the window shrinks. Across five open models and four reasoning tasks, self-consistency matches or beats deployed debate at $93$\% of $980$ matched window-and-group-size settings, with the gap being largest under tight windows and nearly disappearing at a loose one, consistent with the theory. Under a shared window, how much room a model has left to \textit{think matters more than what it reads}. 

---
# MGE-PO: Entropy-Aware Credit Assignment for Mathematical Reasoning 

**Authors**: Jihao Huang, Guangjia Chai, Zhirui Chen, Lei Wang, Yao Liu, Liu Qi, Pengjie Ding, Xinyang Liu, Chengxun Jia, Ling Shao, Hongyu Chen, Junchen Wan  

**Link**: [PDF](https://arxiv.org/pdf/2610.23245)  

**Abstract**: GRPO assigns the same group-relative advantage to equally rewarded responses within a question, leaving their generation uncertainty unused. We introduce MGE-PO, centered on four-quadrant (4Q) advantage shaping that combines verified correctness with rollout generation entropy. Correctness determines the sign of an additional correction, while entropy controls its magnitude: lower-entropy successes receive larger bonuses and higher-entropy failures receive larger penalties. An online entropy-based question curriculum supports this credit rule using cached rollout measurements. At the final checkpoint after eight training epochs on Qwen3-8B-Base, adding 4Q across three policy optimization settings yields absolute gains of 3.23-3.69% in six-benchmark macro avg@32 and 4.24-11.06% in macro pass@32. On Qwen2.5-7B-Base, the complete configuration raises four-benchmark macro avg@32 from 41.5% to 45.4%. 

---
# MultiAgent-OPD: End-to-End Training of Hierarchical Multi-Agent Systems with On-Policy Distillation 

**Authors**: Yuhan Sun, Jiaxin Zhao, Yongdong Luo, Yibo Hu, Meiguang Jin, Junfeng Ma, Weixun Wang, Weihang Pan, Zhongan Bi  

**Link**: [PDF](https://arxiv.org/pdf/2610.23234)  

**Abstract**: In a hierarchical multi-agent system, a Leader assigns tasks and subagents carry them out. Jointly training the Leader and the subagents is difficult because a final answer gives only indirect feedback to the subagents. An LLM judge can score the performance of the subagents, but these scores depend on the judge's own assessment of the quality of subagents' responses. We present MultiAgent-OPD, which combines on-policy distillation with reinforcement learning to jointly train a shared-parameter Leader-Searcher-Answerer team end to end. We first train a Teacher in a Searcher-Answerer workflow using answer correctness. During team training, the frozen Teacher scores student-generated search and summary tokens to provide local process supervision for the Searchers, while the Leader learns from team success and the Answerer from candidate correctness. Across seven datasets, the trained system reaches 38.68% macro avg@4, exceeding Prompted Critic by 4.06 percentage points and C-MAS + RL, the trained cascaded multiagent system, by 2.61 percentage points. The team also solves more questions within four attempts. 

---
# ReLaDrift: Recurrent Language Generation via Drifting for Test-Time Scaling 

**Authors**: Yuki Otsuka, Daisuke Oba, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2610.23225)  

**Abstract**: Drifting Models generate a sample with a single function evaluation, making them attractive for fast parallel language generation, but their one-step formulation does not support iterative refinement. We propose Recurrent Language Generation via Drifting (ReLaDrift), which uses a shared Drifting Model to generate an initial draft from noise and iteratively refine it in a continuous latent space. We train this recurrence by sampling the number of rounds and applying the drifting loss only at the final round, stopping gradients through preceding rounds so that only one round requires backpropagation. Unlike the one-step Drifting Model, whose accuracy does not improve when iterated, ReLaDrift enables a trade-off between test-time computation and generation quality, as diffusion and flow language models do. On Sudoku Hard and GSM8K, it achieves competitive or superior performance compared with these iterative-generation approaches across a range of inference budgets. Our code is available at this https URL 

---
# MAS-Bai: Linguistically Informed Adaptive Multi-Agent Translation for Bowen-Script Bai 

**Authors**: Jingting Zheng, Hong Ren, Yuehan Cui, Shaolin Zhu, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2610.23205)  

**Abstract**: Large Language Models (LLMs) remain unreliable for extremely low-resource languages whose writing systems conflict with high-resource pretraining priors. We study Bai-Chinese translation in the Bowen script, where Chinese characters are frequently borrowed phonetically rather than semantically, causing severe literal-translation hallucinations. We propose MAS-Bai, a linguistically informed adaptive multi-agent framework for Bowen-script Bai-Chinese translation. MAS-Bai routes each input through either a lightweight retrieval-translation path or a deeper philological path with etymological and syntactic agents, and uses cycle-consistency with a lightweight reverse translator as an inference-time reliability signal. We construct a 6,536-sentence Bai-Chinese corpus and evaluate on a 1,536-sentence test set. MAS-Bai achieves 16.23 BLEU and 62.15 COMET, improving over Standard RAG by 8.11 BLEU and 6.85 COMET. Forward LoRA, expert ablations and controlled routing experiments distinguish the roles of supervised adaptation, Bai-specific expertise and conditional routing. The largest gains occur on Bowen phonetic loans and divergent Bai word order, while human evaluation and reverse-verifier analyses support the automatic results. 

---
# AMEND: Constraint-Guided Multi-Agent Text Detoxification with Isolated Critics 

**Authors**: Tianyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.23183)  

**Abstract**: Text detoxification requires balancing toxicity removal, semantic preservation, and fluency. Existing methods typically optimize an unconstrained scalar objective---enabling safety scores to mask semantic loss---or rely on open-ended iterative debate prone to semantic drift and high computational cost. We propose AMEND, an inference-time framework that decouples detoxification into role-conditioned proposal generation, isolated multidimensional criticism, and constraint-first selection. AMEND evaluates diverse candidates with role-blind critics and enforces strict semantic and fluency feasibility before ranking by utility. To maximize efficiency, AMEND Lite restricts subsequent search to at most one targeted repair round only when residual toxic spans remain. On ParaDetox, one-round selection over diverse roles improves the joint evaluation metric by $+0.0508$ over the best single role. On a frozen confirmation test set, AMEND Lite achieves a joint score of $0.6597 \pm 0.0062$, outperforming unconstrained multi-round loops by $+0.0489$ ($p < 0.05$) while reducing inference requests by 28.4%. 

---
# What LLMs Do in the Shadows: Reading Context Reliance from the Unembedding Dark Space 

**Authors**: Alexandra Kuleshova, Andrey Volodichev, Daria Kotova, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2610.23182)  

**Abstract**: Detecting faithfulness hallucinations in large language models (LLMs) is crucial for their safe deployment in high-stakes domains. Recent approaches address this by contrasting contextual against parametric knowledge utilization, yet measuring context reliance typically demands noticeable overhead, such as materializing attention maps, running additional forward passes, or training auxiliary probes. In this work, we show that context reliance can be estimated by projecting hidden states onto the dark space - directions that compose weakly with the unembedding matrix. Beyond its recognized function as an initial-token attention sink, we show that this subspace actively mediates context processing, and that its share of hidden-state norm - termed the Dark Matter Ratio (DMR) - provides a direct score of context utilization. Pairing DMR with a parametric knowledge score yields DarkLens, a lightweight faithfulness detector operating strictly within a single forward pass. Evaluated across three QA and summarization benchmarks on six open-weight LLMs (4B-13B parameters), DarkLens matches or outperforms a wide range of baselines without requiring extra forward passes or probe training. 

---
# Do Speech BCIs Need Larger Models? Rethinking Neural Decoding beyond Scaling 

**Authors**: Zehui Feng, Weichuan Wang, Xiaohan Chen, Cuntai Guan, Ting Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.23175)  

**Abstract**: Recent speech brain-computer interfaces (BCIs) increasingly rely on large models and complex multi-stage pipelines for neural decoding. However, in speech BCI, where training data are scarce, noisy, and expensive to collect, the benefits of model scaling remain unclear. This work revisits a fundamental question: are large models truly necessary for accurate neural decoding? We propose BEST (Brain-Encoding-Speech-Text), a two-stage framework for efficient neural-to-text decoding. Instead of relying on large generative models or phoneme-centric pipelines, BEST introduces hierarchical intermediate representations and a compact speech-oriented decoder that directly maps neural activity to text. This design preserves essential structure while significantly reducing computational cost, enabling robust generalization in low-resource settings and practical deployment under constrained resources. On the Brain-to-Text '24 and '25 benchmarks, BEST achieves word error rates (WERs) of 6.19% and 1.62%, respectively, using only 3.85% of the parameters required by comparable methods. Beyond these results, we provide three key insights: (1) decoder scaling yields non-monotonic gains; (2) cross-modal alignment is more critical than model size; and (3) overfitting degards sequence-level performance. Representation analysis further reveals a consistent neural-audio-text alignment pathway. 

---
# Chinese Decision: A Family of Language Models for Single-Pass Probabilistic Decision-Making 

**Authors**: Qiang Yan  

**Link**: [PDF](https://arxiv.org/pdf/2610.23170)  

**Abstract**: Many language-based applications require probabilities over a finite set of decisions rather than open-ended responses. We present Chinese Decision, a family of Chinese-language models for single-pass probabilistic decision-making. Given context, an instruction, and candidate descriptions, the model extracts final-position candidate-letter logits in one forward pass and normalizes them over valid options, without autoregressive answer generation. A shared interface supports Boolean judgments, categorical choices, and ordinal scores. We report a completed 0.8B model trained with distribution cross-entropy on 13.048 million decisions and preliminary 2B checkpoints trained with an additional RLCD objective. On the full CJ-Bench of 307,900 decisions, the uncalibrated 0.8B model achieves 72.24% general-domain accuracy and 76.98% average accuracy across medicine, law, and finance using a single checkpoint, with 1.07% overall expected calibration error. The 2B model reaches 72.56% general-domain accuracy after 40.13% of one epoch but has not surpassed the completed 0.8B model on the professional-domain average. We further evaluate a validation-temperature-scaled deployment model and its Q8 quantization, finding close aggregate performance while documenting individual prediction changes. We release the training, evaluation, and deployment pipeline. The current results establish a reproducible approach to Chinese probabilistic decisions but do not isolate the effect of model scale. 

---
# Explaining Chess and Go in Natural Language 

**Authors**: Ori Press, Alexander Golubev  

**Link**: [PDF](https://arxiv.org/pdf/2610.23166)  

**Abstract**: Modern chess engines play far beyond human ability, often choosing moves whose underlying ideas are difficult for humans to understand. We propose an automatic metric for evaluating chess explanations, grounded in chess-engine evaluations. Evaluating state-of-the-art language models using this metric, we find that they perform surprisingly poorly, often relying on surface-level board observations rather than robust, transferable chess concepts. We then introduce BinProp, a training-free method that enables these same models to discover reusable chess concepts and use them to explain moves in natural language. BinProp improves explainability for every model tested, nearly tripling the average score compared with direct prompting. Beyond chess, the same method discovers useful concepts in Go, draughts, and shogi without additional tuning. 

---
# Cross-Centaur: Modeling Individual Decision-Making Across Psychological Tasks with Language Models 

**Authors**: Hikaru Asano, Hidezo Suganuma, Ryutaro Mori, Yukino Baba  

**Link**: [PDF](https://arxiv.org/pdf/2610.23157)  

**Abstract**: Large language models (LLMs) are increasingly used as computational models of human behavior. Recent models such as Centaur have advanced this goal by fine-tuning LLMs on trial-by-trial decisions from psychological experiments. However, existing large-scale behavioral datasets do not link the same participants across different tasks, so these models learn population-level regularities but are not trained to model individual-level regularities across tasks. We introduce Cross-Psych-101, a dataset of 591 participants and 261{,}551 valid decisions across 12 Bandit, Tradeoff, Structure, and Social tasks. We train Cross-Centaur to predict a participant's target-task actions from a target-task description and the source-task description paired with the same participant's source-task behavior. Cross-Centaur outperforms off-the-shelf LLMs and Centaur on held-out participants, both within and across tasks. Probing analyses suggest that its hidden states encode participant-level behavioral tendencies that transfer across tasks. These results highlight the potential of cross-task individual-level data for predicting human decision-making with language models. Cross-Pysch-101 is available at this https URL. 

---
# Disentangling Language Effects on Tool Admission and Post-Retrieval Safety 

**Authors**: Aditya Nawal, Manit Baser, Mohan Gurusamy  

**Link**: [PDF](https://arxiv.org/pdf/2610.23093)  

**Abstract**: LLM agents increasingly use retrieval tools across languages, yet their safety evaluation remains English-centric. It remains unclear whether language affects tool admission, post-retrieval generation, or both. We independently vary request, retrieved-page, and required response languages in a controlled URL-retrieval setting across ten models and seven languages. We identify Safety Loss through Interlingual Processing (SLIP). Non-English harmful requests increase mean tool admission from 46.9% to 67.4%, with a positive mean change in every model. This pattern also persists in an autonomous retrieval setup across all four models tested in this setting. Requiring answers in the page language increases pooled harmfulness on matched tool calls from 50.3% to 57.4% across four closed-weight models, mainly in Chinese and Hindi. Translating pages alone produces no consistent harm increase in eight models with comparable inference settings. Automated audits show that a non-harmful response may be a genuine safety refusal, a deflection citing language difficulties, or an unusable answer. Harmfulness labels alone cannot distinguish these outcomes. Multilingual safety evaluations should therefore measure tool admission and response harmfulness separately and distinguish refusal from capability failure. 

---
# Before Retrieval Begins: Safety and Tool-Use Decisions in LLM Agents 

**Authors**: Aditya Nawal, Manit Baser, Mohan Gurusamy  

**Link**: [PDF](https://arxiv.org/pdf/2610.23071)  

**Abstract**: Large language models increasingly act as agents that call external tools, yet most tool-safety research examines only what follows a tool call. The prior decision of whether to call the tool remains largely unexamined and can influence downstream safety. Across the seven open-weight models, the unweighted model-average harm rate after admitted harmful requests is 74.5%. To identify what governs this decision, we apply a factorial intervention to the input request. Within our generated construction, neutralizing the requested-operation component produces the largest admission contrast (19 pp). A linear direction estimated from this contrast influences held-out admission in four of seven models; effects on operation-neutralized requests in some models show that it is not an operation-specific mechanism. We then test whether written refusal is behaviorally and representationally coupled to the same decision. Written refusal accompanies 33.6% of admitted harmful calls. Plain-text refusal training reduces unsafe admission in models that retain benign tool use, showing that response-level safety training can transfer to the call decision. Together, these results establish a partial, model-dependent relation between written refusal and tool admission. They motivate direct evaluation and supervision of tool admission rather than treating text refusal as its proxy. 

---
# ParetoChat: An LLM-Based Interface for Interactive Multi-Objective Optimization 

**Authors**: Gokhan Ceyhan  

**Link**: [PDF](https://arxiv.org/pdf/2610.23062)  

**Abstract**: Multi-objective optimization models are often deployed in settings where decision makers (DMs) must balance some conflicting criteria. In practice, however, stakeholders must translate their qualitative preferences between different criteria into numerical weights, priorities, or reference points required by optimization methods, which can be difficult to specify and interpret. This paper proposes \emph{ParetoChat}, an LLM (large language model)-based expert-agent interface for interactive multi-objective optimization. The agent translates natural-language user input into structured scalarization parameters, invokes an optimization system, presents candidate solutions, and iteratively elicits feedback. We describe a multi-agent architecture that separates input processing, multi-criteria decision-making (MCDM) expertise, solution presentation, and feedback processing. We then present a simulation study using an LLM-based DM agent and benchmark ParetoChat against interactive methods implemented in DESDEO, including reference-point and NIMBUS methods. The computational experiments on multi-objective binary knapsack and set-partitioning instances show that ParetoChat can guide the search efficiently toward solutions that match the DM's preferences. An application to historical courier-order assignment instances illustrates how users can explore objective extremes and intermediate trade-offs through natural-language requests. The results indicate that LLM agents may provide a practical interaction layer between business stakeholders and optimization tools. 

---
# Same Person, Different Advice: Auditing LLM Financial Advice with Repeated Risk-Tolerance Responses 

**Authors**: Soovadeep Bakshi, Apurba Bose, Aishwarya Bose, Sandip Nandi  

**Link**: [PDF](https://arxiv.org/pdf/2610.23060)  

**Abstract**: AI financial advisers set a client's stock allocation largely from one stated-risk question, yet people do not answer it the same way twice: in a national US panel, 41% of households chose a different answer two years later, and in a laboratory retest a third of people moved two or more points on a 0-10 scale after six months. Seven large language models from five providers gave one-turn allocation advice for 800 real household profiles in which only the risk answer changed. The recommended stock share moved by 10 to 16 percentage points on average, and by 24 to 35 when the answer changed, against 0.6 to 4.2 points for the same prompt asked twice; in the laboratory retest the rating alone moved advice by 11 to 18 points. Without any note, the models respond to the answer about as strongly as when told that it is very reliable (less so for the two Claude models). If that instructed response is how each model maps attitude to advice, discounting the answer for its measured reliability, as classical test theory prescribes, would cut the swings by a factor of 2.4 to 3.3. The swing is largest for clients under 40. A note that the answer is noisy steadies some models, but shifts their weight to the client's age and pulls cautious clients towards a generic middle allocation: clients unwilling to take any risk are advised 22% stocks instead of 4%. Three extra money questions reduce the swing for some models, with no statistically detectable change relative to how much advice differs across people. In a scripted test, one clarifying question removes nearly all of the swing from an answer the client then corrects (33.5 points to 1.9 on average) while keeping the response to a change the client confirms. We argue that same-person retest audits should be a standard check for AI advisers, and that what to do with an unreliable answer is a question of client autonomy as well as of statistics. 

---
# Controlling the Diversity-Quality Trade-Off of LLMs by Merging Models Before and After Alignment 

**Authors**: Yasuaki Sumita, Tomoharu Iwata, Toshiyuki Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2610.22999)  

**Abstract**: Alignment is known to change the output characteristics of large language models (LLMs). Base models tend to generate more diverse outputs, whereas aligned models tend to produce higher-quality outputs. The appropriate balance between the two depends on the task. Existing methods either require costly fine-tuning or, like temperature sampling, mainly improve lexical diversity rather than semantic diversity. We introduce a depth-aware FrankenMerging approach for controlling the diversity--quality trade-off between an aligned model and its base model without additional training. This approach combines the lower layers of an aligned model with the higher layers of its base model based on the hypothesis that lexical diversity is more closely associated with lower layers and semantic diversity with higher layers. Experiments showed that varying the number of layers taken from each model adjusts both diversity and quality. In most cases, the merged models exhibited higher semantic diversity than the aligned model, and at some split positions, they showed higher lexical diversity than both original models. They also demonstrated a better diversity--quality trade-off than temperature sampling in most settings. A supporting ablation, in which we reversed the merging order, yielded results largely consistent with the hypothesis. 

---
# Prism Attention: Proposal-Refined Index Sharing Mechanism for Efficient LLMs Inference 

**Authors**: Zhenxu Tian, Kebin Liu, Zhengwu Yang, Yi Su, Qingqing Dang, Yanlin Sha, Kaipeng Deng, Dianhai Yu, Yanjun Ma, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.22985)  

**Abstract**: Long-context decoding in large language models (LLMs) is increasingly bottlenecked by the memory and computation required to attend over an ever-growing Key-Value (KV) cache at each autoregressive step. Although token-level sparse attention reduces this burden by restricting computation to a small subset of relevant past tokens, its practical gains are often offset by repeatedly recomputing layer-wise sparse indices as the context grows. A common remedy is to share sparse indices across layers by computing them only at designated anchor layers, but such static reuse ignores the layer-wise evolution of attention patterns, causing the reused indices to progressively deviate from the layer-specific optima and incur substantial attention loss. By revisiting cross-layer sparsity from the perspective of attention dynamics, we find that inter-layer salient token shift is highly localized: although the exact top-$\kappa$ indices differ across layers, the core high-attention tokens remain stable, with changes concentrated near the expanded boundary of previously selected regions. Motivated by this, we propose Prism Attention, a proposal-refined sparse attention framework. It replaces rigid cross-layer index reuse with a coarse-to-fine strategy, preserving layer-wise adaptability while retaining the efficiency benefits of cross-layer sharing. Extensive experiments show that Prism Attention achieves state-of-the-art performance among mainstream sparse attention methods on both reasoning and long-context benchmarks, while attaining a $3.4\times$ end-to-end decoding speedup over FlashAttention at 128K context length. 

---
# Self-Evolving Agents Can Repair Failures but Struggle to Generalize from Them 

**Authors**: Weixiang Zhao, Yichen Zhang, Yanyan Zhao, Jinming Bai, Bin Liang, Ruifeng Xu, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.22984)  

**Abstract**: Experience-driven self-evolution offers a promising path for large language model (LLM) agents to continually improve through interaction. Yet existing approaches often rely on successful trajectories or external evaluation signals, leaving unclear whether agents can evolve when such positive anchors are unavailable. We study this question through Cold-Start Failure-Only Evolution (CFOE), where agents start without accumulated experience and learn only from failed interactions without privileged guidance. Across three self-evolving paradigms, multiple LLM backbones, and four interactive environments, we uncover a consistent repair-generalization gap: failure-derived experience can help repair previously failed tasks, but rarely yields sustained improvements on unseen tasks. Further analysis shows that agents often misinterpret failures, while even verified-correct experience remains difficult to transfer. Relaxing CFOE reveals that verifier feedback and additional interaction provide limited gains, whereas successful trajectories substantially improve transfer. These findings suggest that current self-evolving agents are better at repairing observed failures than building generalizable capabilities from failure. 

---
# KiranaBench: Can a Language Model Run a Neighbourhood Shop? 

**Authors**: Bhuvan Arora, Ankita Karmakar, Chaitanya Jain, Dilesh Churiwal, Harshit Jain, Shivansh Manish Dubey, Manish Prasad, Raja Piyush, Kshitij Tayal, Dhruv Kumar, Jagat Sesh Challa, Raunak Manekar, Yash Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2610.22980)  

**Abstract**: Large language model (LLM) agents are increasingly evaluated on long-horizon tasks, including autonomously running a business. Existing benchmarks for this setting are almost exclusively English-language, Western in retail context, and single-agent, with the agent observing the business directly. We study how these results transfer to a multi-agent Indic setting. We introduce KiranaBench, a simulated Indian general store in which four roles (manager, cashier, helper, and bookkeeper) are each played by a separate LLM agent. Interactions with customers and suppliers occur in English, Hindi, or Hinglish, and product demand is calibrated to India's national Household Consumption Expenditure Survey. Evaluating four open-weight LLMs, we find that all fall far short of a fixed-rule shopkeeper baseline: the best model earns only one-eighth of the baseline's profit, and two models finish below an unmanaged shop. Performance is also consistently lower for Hindi and Hinglish than for English supplier communication. KiranaBench offers evidence that current LLM agents are not yet reliable for autonomously operating small businesses in India, and provides a testbed for developing agents suited to settings that are long-horizon, collaborative, and multilingual. 

---
# What Shapes Brain Alignment of Large Language Models during Dialogue? Insights from Language Production and Comprehension 

**Authors**: Yuko Nakagi, Yuya Chiba, Yasue Kishino  

**Link**: [PDF](https://arxiv.org/pdf/2610.22961)  

**Abstract**: Brain alignment of large language models (LLMs) offers a way to investigate representational similarities between LLMs and the human brain during language processing. While previous studies have mainly focused on passive language comprehension, little is known about such alignment during dialogue, in which speakers engage in both language production and comprehension. In this study, using fMRI data collected during dialogue, we examine brain alignment during language production and comprehension across LLM layers and brain regions. We assess its association with the intrinsic dimension (ID) of LLM representations and the contributions of several interpretable linguistic and communicative features. Our results reveal distinct patterns of brain alignment between production and comprehension, as well as strong associations with ID in brain regions associated with each processing mode. Furthermore, we show that the contributions of linguistic and communicative features also vary across processing modes, LLM layers, and brain regions. 

---
# Recognizing Atari and Assessing Sente: Controlled Evaluation Tasks for Language-Based Go Agents 

**Authors**: Taiyi Chen, Iskar Deng, Amandalynne Paullada  

**Link**: [PDF](https://arxiv.org/pdf/2610.22953)  

**Abstract**: Domain terminology provides concise descriptions of game states and strategic relationships, but whether game-playing language models can accurately recognize these concepts and relate them to move evaluation remains unclear. We construct a controlled evaluation set centered on atari and its relationship to sente in Go, examining terminology recognition under local board changes and move evaluation under different strategic conditions. We conduct our tests on LoGos, a Go language model, after fine-tuning it with terminology and commentary supervision. The model struggles to recognize changes in the atari relationship and assess its strategic implications. Its terminology predictions are associated with move preferences even when those preferences contradict the reference evaluation. Our controlled tests provide a framework for examining whether language-based Go agents connect domain terminology with precise board relationships and strategic move values. 

---
# Aligning Language Models Towards Counterfactual Invariance 

**Authors**: Basar Demir, Emily Shen, Ksheeraj Sai Vepuri, Sourabh Kulkarni, Harihar Subramanyam, Ankit Jain, Nan Jiang, Chirag Nagpal  

**Link**: [PDF](https://arxiv.org/pdf/2610.22942)  

**Abstract**: Small variations in prompt characteristics can meaningfully alter the semantics, tone, and contextual assumptions of large language model outputs. Such characteristics can involve group variables such as demographics or other sensitive latent attributes. Sensitivity to such input variables is undesirable when demographic information is causally irrelevant to the task, but might still be relevant for generation of semantically coherent responses. This makes the task of enforcing counterfactual invariance while preserving model utility difficult. To study this problem, we first show that standard model alignment recipes (such as reinforcement learning) increase differences in response quality across demographics, while improving overall response quality, ultimately amplifying what is known as counterfactual bias. To address this, we develop a policy gradient technique involving a group-level reward function that operates on matched pairs of counterfactual prompts. For completeness, we build a dataset of controlled counterfactual prompt pairs and design an evaluation that separately measures cases where demographic information is causally relevant to the answer. Finally, we validate our findings on several fairness benchmarks. We demonstrate that our methods reduce counterfactual differences while preserving overall response quality and that the models continue to adapt appropriately when such sensitive information is relevant. 

---
# Beyond Surface Cues: Robust LLM-Generated Korean Text Detection 

**Authors**: SungHo Kim, SangKeun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.22940)  

**Abstract**: As large language models (LLMs) generate increasingly fluent text, the need for reliable detection of LLM-generated content has grown substantially. However, existing research remains heavily English-centric, leaving Korean largely underexplored. Moreover, current evaluations of Korean text detection often rely on superficial cues, leading to overly optimistic estimates of detection performance. To address these limitations, we introduce \textsc{KoLing}, a bias-controlled adversarial benchmark comprising 50.9k instances that suppresses superficial shortcuts while preserving semantic content and separately evaluates LLM-generated and LLM-modified text detection. We further propose KoLingProbe, an interpretable detector grounded in theoretical linguistics that models Korean-specific properties across multiple linguistic levels. Experiments show that KoLingProbe substantially outperforms strong baselines on all detection tasks while remaining robust to adversarial attacks. Our analyses further reveal that integrating complementary linguistic features across multiple levels yields both improved performance and interpretability. 

---
# Many Biases, One Circuit: Post-Training Tunes a Pretrained Cue-Routing Mechanism 

**Authors**: Prakhar Gupta, Terry Jingchen Zhang, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2610.22937)  

**Abstract**: Language models often abandon a correct answer when the prompt contains a cue, a hint toward a wrong option such as "I think the answer is B." Sycophancy and related biases show this failure. They are often studied separately and blamed on alignment. We test this view causally, tracing the attention heads behind these biases through the stages of two post-training pipelines. In multiple-choice tests on most models, four cue types rely largely on the same few heads, a shared circuit already present in pretrained models. Most base models follow cues in plain text. Fine-tuning a base model on about 200 generic assistant examples without cues activates these heads in chat format, so cue-following can be a side effect of assistant tuning. The final reinforcement learning (RL) stage strengthens the effect of cues in almost every comparison. In Tulu-3-8B and OLMo-2-7B, we swap the attention patterns of six middle layers between the RL checkpoint and its predecessor. For suggested-answer cues, this shifts each checkpoint's cue effect most of the way to the other's level. Post-training thus seems to change when and how strongly models use this circuit. Mitigation for these multiple-choice biases could target this circuit. 

---
# ASR-Style Transcript Normalization Removes Useful Pragmatic Signal for Dialogue Act Classification 

**Authors**: Lucky Onyekwelu-Udoka, Md Shafiqul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2610.22934)  

**Abstract**: Automatic speech recognition (ASR) pipelines often normalize transcripts by removing filled pauses and punctuation and expanding colloquial forms, potentially discarding cues useful for dialogue act classification. We evaluate nine transcript conditions across six classifiers (TF-IDF logistic regression, RoBERTa, BERT, ALBERT, GPT-2, and BiLSTM) and three conversational corpora (Switchboard, ICSI MRDA, and SLUE-HVB), supplemented by multi-seed experiments, normalization controls, and real-ASR evaluations. Across five random seeds, heavy normalization reduces Macro-F1 in all 15 neural model-corpus combinations under the original preprocessing protocols. Removal using the original filler and backchannel lexicon consistently degrades performance on MRDA and SLUE-HVB but produces mixed effects on Switchboard. Punctuation-aware TF-IDF experiments show that punctuation removal reduces Macro-F1 by 0.036 on Switchboard and 0.078 on MRDA, demonstrating that the original linear baseline's apparent insensitivity resulted from tokenization. Restricting filler removal to um, uh, and erm preserves degradation on MRDA but largely eliminates the SLUE-HVB effect, where the original lexicon removes class-defining backchannel expressions. These findings demonstrate that normalization effects depend on preprocessing definitions, feature extraction, and conversational context, supporting task-dependent rather than indiscriminate transcript normalization. 

---
# Where's the Indirect Evidence? Investigating Grammatical Generalization with Influence Functions 

**Authors**: William Dai, Zhenghao Herbert Zhou, Maya Viswanathan, Robert Frank  

**Link**: [PDF](https://arxiv.org/pdf/2610.22919)  

**Abstract**: Language models generalize beyond their training corpus to learn linguistic phenomena (Patil et al., 2024; Misra and Mahowald, 2024), showing that there exists indirect evidence driving learning. Prior work has inferred that indirect evidence exists without identifying its character or considered specific hypotheses about what it might be. We explore an alternative bottom-up method for identifying indirect evidence. We use influence functions (Koh and Liang, 2017) to measure how individual training sentences affect model predictions in 15 grammatical phenomena from BLiMP (Warstadt et al., 2020). We filter out a phenomenon's direct evidence using hand-constructed rules and take the 60,000 most influential remaining sentences as indirect evidence. This indirect evidence substantially affects learning: retraining without it lowers held-out accuracy more than removing a matched random group for every phenomenon across five models (by a median of 12 points). We characterize the indirect evidence by its similarity to our direct evidence filters: it (partially) meets the filters' structural conditions at a median of 4.7 times the rate of a matched random group. Finally, clustering phenomena by indirect-evidence overlap recovers three of the five BLiMP categories that include multiple phenomena, suggesting that this method can reveal linguistic relationships. 

---
# FederatedBytes: Tokenizer-Free, Byte-Level Small Language Models for Efficient Multimodal Federated Learning 

**Authors**: Ali Sarabadani, Massimo Piccardi, Mehran Abolhasan, Farzad Tofigh, Ian Zhou, Justin Lipman  

**Link**: [PDF](https://arxiv.org/pdf/2610.22906)  

**Abstract**: Federated Learning (FL) is a powerful machine learning framework that allows building a global model from the contributions of multiple, distributed clients, while ensuring the privacy of their local data. However, its heavy computational and communication overheads challenge its use for contemporary AI workloads such as natural language processing (NLP) and multimodal analysis. For this reason, this paper presents a novel small language model that leverages byte-level text and multimodal data processing, enhanced by a linear attention mechanism, to enable manageable Federated Learning of NLP and multimodal analysis tasks. The proposed model, nicknamed FederatedBytes, manages to process text and any other data types, including images, by treating its input as a mere sequence of bytes, while its lean linear attention mechanism allows abating the overall memory and computational footprint. To prove the effectiveness of the proposed model we have carried out an extensive performance evaluation on sentiment analysis tasks in multilingual and multimodal settings over the Multilingual Amazon Reviews and ScienceQA datasets. The experimental results show that FederatedBytes has been able to outperform comparable models such as ByT5 and LSTM, achieving higher accuracy while significantly reducing memory and communication overheads. 

---
# Competence-Based Evaluation of Logical Reasoning in Large Language Models 

**Authors**: Jize Jiang, Chengxiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2610.22894)  

**Abstract**: Recent advances in large language models (LLMs) have brought impressive performance across diverse tasks; however, shortcomings remain in logical reasoning, including inconsistencies under linguistic variation. To improve the reasoning capacity of LLMs, we must be able to accurately measure their reasoning capacity. Current benchmarks primarily report single accuracy numbers on selected reasoning tasks. Such a performance-based evaluation methodology says little about a model's generalization capacity in reasoning. To address this limitation, we argue for a different evaluation paradigm, called competence-based evaluation (CBE), which tests whether models reliably apply the same competence across a family of problems. A parameterized generator scales complexity while keeping the required competence fixed, and matched answer-preserving presentations test correctness beyond one formulation. We instantiate CBE on elementary ordering tasks, scaling input size while varying presentations of the same ordering rule. Across six GPT and Qwen models, performance is reasonably good over bounded complexity ranges, but even strong models fail the competence condition across the tested family. Chain-of-thought and GRPO extend strong performance without resolving all tested failures; supervised LoRA shows little held-out transfer. CBE provides diagnostic profiles and an evaluation protocol that can grow with future models. 

---
# Bengali Hallucination Detection with Retrieval and Programs: A System Study 

**Authors**: Shamim Hasan, Samiya Ahmed Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2610.22892)  

**Abstract**: Bengali hallucination detection must distinguish unsupported factual claims from incorrect reasoning, often without a supplied reference passage. We present and characterize an inference-only system that combines response-conditioned query rewriting, hybrid retrieval, and task-specific language-model judging. Six retrieval routes search question-answer and passage collections; a router directs each example to a 27B textual judge or a 14B agent that can compare retrieved answers, reason directly, or generate and execute Python. We characterize the system using version-specific code, saved predictions, execution traces, and corpus metadata from the Olikbochon challenge. Kaggle associates a displayed private macro-F1 of 0.956 with Version 2; its saved run processes 2,516 examples in 220.99 minutes on two T4 GPUs. Of 253 math-routed examples, 88 terminate through candidate matching, while execution traces expose 49 stderr-bearing run events across 35 examples. A later run changes six predictions despite identical core worker code. A bounded audit of 1,090 benchmark-related corpus records finds exact question or context overlap for 38 processed evaluation inputs, precluding an unqualified claim of corpus-independent generalization. The contribution is a concrete retrieval-and-computation judging architecture with an auditable operational characterization, not a controlled estimate of component gains. We distinguish the reported leaderboard result from independently recomputed diagnostics and identify the evidence needed for stronger effectiveness and reproducibility claims. 

---
# NLData2Opt: Benchmarking Parameter Derivation and Prediction for Modeling Optimization Problems from Natural Language and Data 

**Authors**: Lijia Liu, Kyohei Atarashi, Jiyi Li, Koh Takeuchi, Shunji Umetani, Hisashi Kashima  

**Link**: [PDF](https://arxiv.org/pdf/2610.22889)  

**Abstract**: Formulating optimization problems from raw inputs of problem descriptions and data is time-consuming and costly. While Large Language Models (LLMs) offer a promising way to automate this process, existing benchmarks overestimate their capabilities by relying on idealized inputs where all data is relevant and complete. This diverges from reality, where data-centric reasoning is required to extract, compute, and predict parameters from noisy sources. To address this, we introduce NLData2Opt, a benchmark to evaluate LLMs on the multi-stage reasoning inherent in practical optimization workflows. NLData2Opt decomposes problems into problem descriptions and data, and requires models to (1) extract relevant parameters from data containing misleading attributes, (2) compute missing parameters derived by transforming existing data attributes, and (3) predict unavailable future values from historical observations. Evaluation results show substantial performance decline relative to existing benchmarks, indicating that benchmark progress does not directly translate to formulation accuracy in practical optimization workflows. NLData2Opt is available at this https URL. 

---
# Identify, Classify, Act: Per-Instruction Controlled Compliance for LLMs over External Data 

**Authors**: Zesheng Shi, Xue Li, Chenrui Dai, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.22874)  

**Abstract**: LLMs over external data face a per-instruction tension: a single document or tool return often interleaves legitimate directives with adversarial prompt injections, and the model must follow the former while rejecting the latter. Existing defenses suppress or trust all external instructions, collapsing the safety-utility frontier and forcing a choice between over-refusal and over-compliance. We argue the missing capability is \textbf{per-instruction credit attribution} and formalize it as External-Data Instruction Classification, an identify-classify-act decision over each embedded directive. We propose \textbf{EDIC}, a decomposed reward framework for per-instruction compliance. A scene-grounded pipeline injects mixed beneficial and harmful instructions with counterfactual filtering, while a generative judge is initialized by SFT and refined with GRPO to attribute compliance at the instruction level. The judge then produces a reward that separately encourages beneficial compliance and penalizes harmful compliance for RL training. Across two backbones and five benchmarks, EDIC improves both safety and utility over the strongest prior defense, lifting harmful-rejection by up to 7.7\% and Arena-Hard-Auto utility by 12.5\% on Qwen3-4B. The code repository is hosted at this https URL. 

---
# Uncertainty-Aware Budget Allocation for Efficient LLM Tree Search 

**Authors**: Jianguo Huang, Zhiyi Lyu, Fuxiang Zhang, Yanchen Deng, Deheng Ye, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2610.22858)  

**Abstract**: Tree search enhances multi-step reasoning in large language models (LLMs), yet prevailing methods expand every step with a fixed budget, irrespective of the marginal utility of expansion. We first show, across existing search algorithms, that continuations sampled from a shared intermediate state are largely duplicates and collapse to a few distinct answers. Critically, we find that the semantic dispersion of sibling actions predicts the redundancy of their downstream continuations, permitting a subtree's budget to be fixed before expansion. An information-theoretic analysis motivates this: treating the per-step budget as the Monte Carlo sample size for estimating a step's residual answer entropy, we derive a sufficient budget that shrinks with sibling dispersion. We accordingly propose $\textbf{Uncertainty-Aware Allocation}$ (UAA), a training-free framework that decouples value from uncertainty: values determine which candidates are promising, while uncertainty, estimated from sibling dispersion at no additional model cost, determines how many merit expansion, via uncertainty-aware pruning and uncertainty-aware budgeting. Across mathematical reasoning, logical reasoning, and planning benchmarks, UAA reduces token usage by up to $90\%$ and latency by up to $9\times$ at comparable accuracy, generalizing across four tree search algorithms and both open- and closed-source LLMs. 

---
# The Reliance Hypothesis: Chain-of-Thought Faithfulness Rises as Language Resource Level Falls 

**Authors**: Lucas Resck, Jingyi Sun, Qianli Wang, Isabelle Augenstein, Anna Korhonen  

**Link**: [PDF](https://arxiv.org/pdf/2610.22841)  

**Abstract**: Chain-of-Thought (CoT) faithfulness matters for monitoring, explaining and safely deploying large language models, yet it has been studied mostly in English. Languages differ in level, the amount of text available to learn from, so models know less in lower-resource languages, which may change how much they rely on their reasoning. To our knowledge, we are the first to relate CoT faithfulness to both resource level and task type (knowledge- or reasoning-intensive). On parallel multiple-choice questions from Multilingual Massive Multitask Language Understanding (MMMLU) in eight languages, faithfulness rises as resource level falls on knowledge-intensive tasks but not on reasoning-intensive ones, a gap that points the same way across four models and several faithfulness measures. We propose, and find support for, the Reliance Hypothesis: models rely more on their reasoning when they cannot recall the answer, as in lower-resource languages on knowledge-intensive tasks. Model confidence partly explains the effect of resource level, but not the knowledge-reasoning gap. Languages in which models are less confident rely more on reasoning, yet less confident answers within a language do not, a Simpson's paradox. In lower-resource languages, models rely on their reasoning for more answers, not more strongly for each. CoT monitoring may therefore be most informative in lower-resource languages, where it is least often validated. 

---
# Causal Edit Serialization 

**Authors**: Louis Robinson  

**Link**: [PDF](https://arxiv.org/pdf/2610.22820)  

**Abstract**: Standard autoregressive language models generate text sequentially, but revision requires referring back to tokens in the input context or those already generated. Causal Edit Serialization (CES) is introduced as a framework for expressing such revisions within the same causal token stream. Special pointer tokens are paired with predicted source addresses, allowing insertion, deletion, replacement, copying and movement to be localized explicitly. A lightweight localization head and dual-index rotary position encoding support both address prediction and conditioning on previously emitted pointers, while retaining a standard causal attention mask. A pretraining procedure combines ordinary language modeling with serialized repairs derived from random corruptions, model-generated edits and repeated text. Across model scales and editing baselines, CES models demonstrate complete applied edits with modestly higher clean-text bits per byte than clean autoregressive models. Supervised and reinforcement-learning post-training provide a grammatical-error-correction proof of concept, producing more beneficial than harmful changes under reference-blind model judging. Correct repair programs also improve subsequent text prediction while the original context and its cached states are retained. CES provides a common interface for generation and localized revision, with potential extensions to iterative refinement and additional pointer-addressed transformations. 

---
# Watermarkability: Measuring and Improving Language-Model Compatibility with Logit-Based Watermarks 

**Authors**: Zewei Deng, Mohamed Seif, Liyan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2610.22810)  

**Abstract**: Language-model watermarks can help identify model-generated text, encode provenance information, and detect post-generation edits. A broad class of logit-based watermarks increases the logits of a designated token set at each decoding step. Because a base model's native probability mass on this set varies across contexts, models, and watermark rules, the same logit bias can yield different adherence rates and increases in perplexity (PPL). We introduce watermarkability to characterize the compatibility between a model and the token sets requested by a watermark. At a target mass $\tau$, we quantify this property through a watermarkability cost: the expected minimum KL divergence from the base-model distribution required to assign probability at least $\tau$ to the requested set. A lower cost means that the model is more watermarkable for that request distribution. To reduce this cost, we propose a two-stage method. In the first stage, we train a target-set-conditioned low-rank adapter on unions drawn from a 16-way vocabulary partition to increase the requested set's probability mass. In the second stage, a closed-form controller applies the smallest remaining uniform logit bias needed to reach the target mass. We evaluate the method on six watermark schedules using OPT-125M and Qwen3-8B. Across the 12 model-schedule pairs, the pre-adaptation watermarkability cost correlates with the log-PPL increase under a near-hard fixed-bias baseline (pooled Pearson $r=0.929$). For every schedule, at least 99.80% of sampled tokens fall in the target set requested at their decoding step. At this reliability, the adapter and controller reduce average per-schedule prompt-conditional PPL by 46.3% on OPT-125M and 5.2% on Qwen3-8B relative to the fixed-bias baseline. 

---
# On the Position Bias of Large Language Models: A Comprehensive Study 

**Authors**: Yining Li, Huayang Li, Lemao Liu, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2610.22807)  

**Abstract**: Recent studies have shown that position bias remains a critical vulnerability in large language models (LLMs), undermining both the integrity of multiple-choice evaluations and the reliability of retrieval-augmented generation. However, existing understanding of this phenomenon is largely derived from studies with limited model coverage and English-centric benchmarks, leaving the behavior of frontier models and diverse languages underexplored. In this work, we conduct a comprehensive investigation of position bias across model scales ranging from billion-level to trillion-level parameters, covering both multiple-choice and multi-document tasks in five languages. Our experiments reveal that while position bias remains noticeable in small-scale models and Non-English languages, increasing model scale and employing standard Chain-of-Thought prompting significantly mitigate this issue. Moreover, we demonstrate that emerging \emph{thinking models} with extended reasoning capabilities are nearly immune to position bias, suggesting a costly but general approach toward resolving this vulnerability. 

---
# Causal Latent Workspaces for Looped Transformers 

**Authors**: Ilia Zaznov  

**Link**: [PDF](https://arxiv.org/pdf/2610.22796)  

**Abstract**: Reusing a Transformer block across depth increases computation without replicating its parameters, but successive passes have no dedicated state through which positions can exchange compact summaries of their causal prefix. We introduce the causal latent workspace Transformer: an anchored, four-round shared-depth architecture in which each position writes gated vectors into prefix-specific latent slots and retrieves a content-addressed summary before the next shared block application. This path complements causal self-attention while the original input embedding remains accessible at every round. Across three byte-level corpora, three seeds, and two matched parameter budgets, the proposed model achieves the best mean held-out perplexity in five of six settings against five Transformer controls. At 101.71M parameters it leads on WikiText-103 raw, enwik8, and TinyStories, reaching perplexities 2.7820, 2.7843, and 1.6823; these improve upon the strongest non-workspace control on each corpus by 1.06%, 0.73%, and 1.01%. All nine seed-labelled comparisons at this budget favor the workspace. At 0.918M parameters it leads on WikiText-103 and TinyStories, whereas a conditioned loop is stronger on enwik8. Architectural ablations show that adding the workspace improves upon the anchored loop in all six settings and support refreshing its causal summaries at each round. Together, these results identify a causal, content-addressable prefix workspace as an effective structural augmentation of looped Transformers. 

---
# Prune the Leaky Neurons: Mitigating Private Data Leakage in Fine-Tuned LLMs via Joint-Embedding-Guided Pruning 

**Authors**: Mohamed Shaaban, Mohamed Elmahallawy  

**Link**: [PDF](https://arxiv.org/pdf/2610.22792)  

**Abstract**: Fine-tuning large language models (LLMs) on proprietary records enables domain adaptation but introduces two challenges: (i) privacy leakage of memorized private data, such as personally identifiable information (PII), and (ii) balancing private data removal with utility preservation. Existing defenses, including data sanitization and differential privacy, often degrade performance, while privacy-neuron editing lacks a PII-free reference to distinguish identity-specific neurons from those supporting useful text. We propose JEPA-Prune, a post-hoc neuron pruning framework guided by a model trained exclusively on PII-free views. Alongside the fine-tuned private model, JEPA-Prune, trains a public model on surrogate and masked views of the same records using a joint-embedding predictive architecture (JEPA), capturing domain knowledge without identifiers. A cross-model representation probe quantifies deviations between private representations of leaked records and this PII-free reference. Integrated gradients and a scale-invariant specificity score then identify PII-responsive neurons, which are pruned without retraining. Experiments show that pruning just 1.6% of neurons reduces PII inference accuracy by up to 7.4x and PII extraction recall by up to 13.9x, while keeping unseen-text perplexity within 16% of that of the unpruned model for most LLMs. In contrast, privacy-neuron editing increases perplexity by at least 26x. 

---
# Self-Evolving Coding Agents via Harness-Policy Optimization 

**Authors**: Zhepei Wei, Wenlin Yao, Baolin Peng, Danqing Wang, Qianhui Wu, Hao Cheng, Yu Meng, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.22791)  

**Abstract**: Large language model (LLM) agents have shown promising potential in coding tasks, and their performance is jointly determined by the policy model and the harness that mediates their interaction with the environment. Indeed, harness and policy are mutually reinforcing and evolving only one component is suboptimal---a harness tuned for one checkpoint may not target the failure modes of an updated policy, while policy learning remains limited to the trajectories elicited by a fixed harness. In light of this, we propose Self-HPO, a recursive self-improving framework for coding agents that alternates harness and policy optimization, using the evolving policy as its own harness proposer. Specifically, in each round, the current policy first analyzes its own failures and proposes new harness candidates (H-step). The policy is then updated through reinforcement learning (RL) under the selected harness (P-step), and hard tasks identified during training form the next harness search set, keeping harness optimization targeted to the policy's evolving failure modes. Across three coding benchmarks, Self-HPO consistently outperforms harness-only and policy-only baselines, improving Qwen3.5-35B-A3B's average resolve rate from 57.8% to 64.8%, with gains generalizing to out-of-distribution settings. Further analysis shows that both optimized components retain gains even when paired with their base counterparts, while their joint gain exceeds the sum of their individual gains. Self-HPO also improves reliability (Pass@1) and expands the capability frontier (Pass@k) over the base agent. Additionally, we show that with a stronger external model (e.g., Opus 4.8) as the harness proposer, the same framework yields a further performance boost. 

---
# Thunder-DeID+: Boundary-Aware Korean De-identification from Court Judgments to Broader Sensitive Documents 

**Authors**: Gyuseong Lee, Gyeongje Cho, Jongmin Kim, Heejin Kim, Yeonkyoung So, Hyunggeun Jeon, Hyunji M. Park, Sangho Kim, Yoonhee Park, Youngbin Pyo, Jaejin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.22733)  

**Abstract**: This study examines the transferability of a de-identification model developed for Korean court judgments to a broader range of sensitive documents, focusing on boundary precision, label granularity, and recall. The agglutinative structure of Korean causes Byte Pair Encoding (BPE) tokenizers to frequently merge Personally Identifiable Information (PII) with adjacent particles, which complicates accurate span recovery. Furthermore, current PII taxonomies lack sufficient granularity for legal texts. To address these issues, this study introduces Thunder-DeID+, which integrates a refined three-tier PII taxonomy with a boundary-aware tokenizer. The tokenizer enhances a pretrained BPE vocabulary by incorporating N-gram and boundary-scoring components to better align token and PII span boundaries. The proposed approach is evaluated using a tokenizer-independent word-segment-level protocol. On full-length judgments, the tokenizer consistently improves W. Strict Typed F1-score from 0.76-0.77 to 0.87-0.88 across four model sizes (150M, 395M, 750M, and 1B), mainly by increasing precision rather than model scale. With fine-tuning on 1,000 broader sensitive documents, the judgment-trained model further achieves 0.93 W. Strict Typed F1-score, indicating effective transfer beyond court judgments. 

---
# TokenEconomizer: Experience-Driven Black-Box Cost Control for Coding Agents 

**Authors**: Aaron J. Li, Weijia Xu, Chandan Singh, Zixiao Chen, Jeevana Priya Inala, Charlotte Siska, Acar Tamersoy, Bin Yu, Ion Stoica, Michel Galley, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.22732)  

**Abstract**: Advances in frontier coding agents have sharply increased the token costs of AI-assisted software engineering. Their token efficiency is typically evaluated on isolated tasks, whereas real-world developers use them to continuously build, update, and maintain large-scale projects. This mismatch motivates us to evaluate agents over sequences of realistic coding tasks, where we find that agents exhibit recurring environment-dependent token costs that existing methods struggle to reduce. To minimize these costs, we introduce TokenEconomizer, an external controller that cuts environment-specific token costs over a sequence of tasks. These cuts come from determining (i) which operational micro-skills to inject, (ii) which persistently irrelevant tools to prune, and (iii) when to intervene in unproductive failure loops. Across Claude Code, Codex, and GitHub Copilot CLI paired with frontier models, TokenEconomizer reduces the average token cost by 18.4-30.9% with comparable task success. Savings hold in every repository under every tested agent-model setting and generalize to diverse domains such as sequential web development and security auditing, where costs fall by 6.4-21.2% depending on task structure. Our work establishes adaptive black-box cost control as a promising direction for reducing token costs of coding agents in realistic settings. 

---
# CoLU: Confidence-Weighted Layerwise Unlearning for Large Language Models 

**Authors**: Jingkai Guo, Ziyi Zhao, Chaitali Chakrabarti, Deliang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2610.22721)  

**Abstract**: Large language model (LLM) unlearning aims to forget certain information while preserving others, without retraining from scratch. SoTA methods such as NPO, SimNPO, and SatImp balance forgetting and retain through whole model fine-tuning, while FOM-UL explores selective layer updates. Prior works show that suppressing target information can damage retained knowledge. This trade-off can vary across layers even under the same unlearning loss. In this work, we propose Confidence-Weighted Layerwise Unlearning (CoLU), which combines a novel confidence-weighted forgetting and retain regularization while updating a single selected transformer layer, rather than the whole model for unlearning. To reduce CoLU compute cost, we further propose Brief Layer Screening (BLS), which uses a newly defined normalized forgetting-retention (NFR) score to first rank layers through a brief unlearning trials using a subset of forgetting dataset and fewer epochs. BLS pre-selects top-k ranked layer candidates, which will then be fully profiled using full data analysis. BLS is proven to reduce the runtime of CoLU by 60.2-79.7% with almost the same performance. On TOFU dataset, CoLU improves both forgetting and retention over NPO, SatImp, and GA-based FOM-UL in all six TOFU settings, and over SimNPO in five of the six settings. On MUSE Books and News datasets, CoLU achieves the highest retained KnowMem and the PrivLeak closest to zero among the compared SOTA unlearning methods. CoLU presents a new record performance in both unlearning and retain in most settings up to date. Moreover, across the measured settings, average CoLU GPU memory usage is 77.4% to 79.5% lower than full-model unlearning. 

---
# ARBOR: Locating Reasoning Failures in Long Chains of Thought through Reasoning State Tracking 

**Authors**: Huihan Li, Siyuan Wang, Naga Vamsi Ramana Dinavahi, Xiaohui Liu, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2610.22717)  

**Abstract**: Large reasoning models solve complex problems through long chains of thought, in which prior work has documented failure modes such as overthinking, underthinking, and unsupported deductions. Existing methods detect these failures from local signals: textual similarity to nearby steps, switch tokens, or an LLM's judgment of a step's correctness. However, what a step contributes to the solution depends on the state that earlier steps have established, so identifying failures requires a global view that tracks that state across the trajectory. Therefore, we propose ARBOR, a reasoning-failure attribution framework that represents each trajectory as a tree of reasoning states, built as each step creates, verifies, prunes, or revisits a state, and attributes each failure to a relation between a step and the state preceding it. Using ARBOR, we refine the three failure modes into distinct subtypes that harm reasoning through different mechanisms and at different costs. We find that the tree state ARBOR maintains before a step separates failure steps from the at-risk steps at which the failure did not occur. We discuss how ARBOR integrates with existing mitigations at inference time and in training data. Code, data and interactive examples are available at this https URL. 

---
# First Passage Attention 

**Authors**: Yuxuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.22714)  

**Abstract**: Augmenting softmax attention with forget gates typically relies on summing decay contributions along the key--query path. Because such accumulation is monotonic, these mechanisms cannot selectively erase intermediate context while preserving older, structurally relevant tokens. We propose First Passage Attention (FPA), which shifts from path summation to path extrema by lowering a key's logit proportionally to the deepest dip of a learned scalar field below that key's level over the intervening span. FPA encompasses the Forgetting Transformer (FoX) as a special case, and when the field and levels follow a shared learned walk, the alive keys with unpenalised logits naturally form a stack. We show that a constant-dimension FPA head realises any subset of keys left alive by subsequent dips, whereas softmax heads with positional biases or token-level forget gates require a head dimension linear in the number of keys. In synthetic experiments, one-layer FPA heads converge to these target sets while comparable softmax and FoX baselines fail on every seed. Scaling to a 110M-parameter language model, FPA matches FoX in perplexity and uniquely achieves perfect passkey retrieval at $8\times$ the training length, even with its key--value cache compressed to about $25\%$. 

---
# Explore Separately, Adopt Selectively: Execution-Guided Tool Use over Large Tool Libraries 

**Authors**: Anna Mazhar, Seiji Maekawa, Nikita Bhutani  

**Link**: [PDF](https://arxiv.org/pdf/2610.22688)  

**Abstract**: Large language model agents must discover relevant tools from large libraries, e.g., thousands of tools, as new needs arise during task execution. When several candidates could satisfy the same need, execution may be required to identify the relevant one. Investigating candidates in a shared trajectory can leave alternatives unexplored or allow failed attempts to influence subsequent reasoning. To address this, we present BRIA (BRanch-Inspect-Adopt), a simple yet effective agent that temporarily isolates candidate investigations in separate branches. Each branch retrieves and executes tools to investigate its candidate; a task agent inspects the resulting execution evidence and adopts only useful results into its continuing trajectory. BRIA retrieves additional tools as new capability needs arise during task execution. We evaluate BRIA on PlanBench-XL and MCP-Atlas using a shared pool of 1897 tools, three language models, and static and adaptive retrieval baselines. Across all settings, BRIA matches or exceeds the strongest retrieval baseline. On MCP-Atlas, it achieves an average claim score of 89.77, compared with 75.40 for MCP-Zero, a gain of 14.37 percentage points. BRIA also lies on the Pareto frontier for task performance and input-token cost. 

---
# Multimodal Filesystem Memory For Long-Horizon Embodied Agents 

**Authors**: Chloe Su, Jiahan Li, Zhenting Qi, Chaoran Cheng, Haoyu Chen, Sejoon Jun, Weirui Ye, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.22686)  

**Abstract**: Agents operating over long horizons need an effective memory that allows them to reuse their experience throughout long interactions. If the entire history of an interaction is retained, it becomes difficult to find the relevant experience, while compressing it into summaries may result in the loss of visual and spatial details that are required later on. We present \textbf{MFM}, a multimodal filesystem memory architecture that keeps complementary types of agent memory. \textbf{Episodic memory} stores the observations, actions, and outcomes, and \textbf{spatial memory} associates experiences with locations. \textbf{Semantic and procedural memory} build up general knowledge and reusable methods of acting, and \textbf{working memory} holds the current plans. MFM makes use of searchable textual records to connect these different representations to the underlying multimodal episodes. The agents are able to find previous experiences by means of contextual cues and to examine the original observations in order to recover details that are missing from their textual descriptions. Through consolidation, the agents convert experience into higher-level knowledge and procedures while still keeping references to the original episodes. In the case of long-horizon Minecraft and computer-use tasks, MFM leads to improved performance with three base models. With GPT-6-Astra, MFM completes \emph{Mine Diamond from Scratch} and reaches 16 out of 24 milestones on the \emph{Ender Dragon} task, as against 12.7 for the best baseline. On CUABench-KiCad it attains a mean reward of 0.440, as compared to 0.400 for the best baseline and 0.267 in the case of no persistent memory. 

---
# ProPerAgent: Toward Memory-Driven Proactive Assistance 

**Authors**: Shenglai Zeng, Xiao Yang, Tiffany Ma, Xinyuan Zhang, Nanshu Wang, Deren Lei, Kai Sun, Xilun Chen, Tammy Stark, Xianhui Zhu, Yuetai Li, Rulin Shao, Tianxin Wei, Rakesh Wanga, Anuj Kumar, Jiliang Tang, Wen-tau Yih, Xin Luna Dong  

**Link**: [PDF](https://arxiv.org/pdf/2610.22671)  

**Abstract**: As assistants move from reactive answering toward proactive intervention, memory changes role: it no longer only supplies the answer to a query, it determines whether and when to speak at all. Prior work in this setting hands the system the relevant fact in context; we study the regime that makes the problem hard, where the evidence justifying an intervention must be mined from a long-horizon personal memory. We introduce ProPerMem, a benchmark for memory-driven proactive assistance (MPA), pairing egocentric video with year-long, distractor-rich user memories (~618K tokens on average), human-verified trigger windows, and a protocol that scores content and timing jointly. Every baseline we evaluate performs poorly (at most 15%), including frontier models given summarized or retrieval-augmented memory. A controlled study explains why: with oracle evidence delivered inside the trigger window GPT-5.3-Chat reaches 96%; letting it choose the timing drops it to 38%, and replacing oracle evidence with retrieval drops it to 15%. Content and timing are separately necessary, and missing either is catastrophic. Guided by this, we build ProPerAgent (Proactive Personalized Agent) with a plug-and-play harness that surfaces memory from a graph-organized store and gates it down to what is worth interrupting for, plus a lightweight RL stage that curbs early and repeated firing. Without any training, ProPerAgent lifts strict accuracy from below 10% to 37.5% on Qwen3.8-27B and 38% on GPT-5.3-Chat; RL post-training reaches 42.4%. 

---
# Secondhand Stress: How Stressful Disclosures Affect Psychosocial Wellbeing in Online Graduate Admissions Communities 

**Authors**: Veda Duddu, Agam Goyal, Sabelle Huang, Ayush Umashankar Prasad, Mingquan Yi, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2610.22657)  

**Abstract**: Online communities provide spaces for individuals navigating stressful life transitions to disclose difficulties and seek peer support. Yet encountering others' distress may also have psychosocial implications for community members. Drawing on stress crossover theory, we examine how engagement with peer distress during graduate admissions is associated with subsequent psychosocial expression and participation, and whether similar associations extend through downstream interactions. We analyze 171,981 posts and 1,130,413 comments from two Reddit communities across three admissions cycles, using propensity-score stratification and baseline-adjusted comparisons. Users who engaged with distress disclosures during application preparation subsequently exhibited higher classifier-estimated anxiety, depression, and stress expression, alongside greater commenting activity and lexical redundancy. These associations recurred across admissions cycles, while no significant psychosocial associations were detected among downstream respondents interacting with previously exposed users. Our findings highlight how greater distress-related expression can coexist with continued community participation, while distinguishing direct engagement with distress disclosures from subsequent social contact. We discuss implications for designing online communities that preserve opportunities for self-disclosure and peer support while remaining attentive to the experiences of those encountering distress-related content. 

---
# Evidence-Gated Bilingual RAG for Reliable Consumer Banking Search 

**Authors**: Ahmed Mohamed Mahmoud Rayyan, Rana Mahmoud Mohammed, Aya Abdallah Abdelmonem, Mohamed Khaled Goda Othman, Rahatara Ferdousi  

**Link**: [PDF](https://arxiv.org/pdf/2610.22652)  

**Abstract**: Consumers often describe what they need rather than use the terminology found on a banking website. For example, a natural-language request ("a card for travel") may not share the keywords of the relevant product page, and the problem is harder in bilingual services, where the same intent may be expressed in Arabic or English. This paper presents an evidence-gated bilingual retrieval-augmented generation (RAG) framework for consumer banking search that operates entirely on local infrastructure. The framework combines structured web-content refinement, hybrid dense-lexical retrieval, intent-aware evidence selection, cross-encoder reranking, and a three-outcome Answer / Related Mismatch / No Answer decision mechanism. On 210 URL-labeled Arabic and English queries, the proposed approach improves Top-1 hit rate from 50.48% to 79.05% and MRR@5 from 0.571 to 0.797 over a hybrid baseline. A separate 26-case end-to-end evaluation achieved 69.2% strict success, with 72.7% of generated answers classified as supported by an automated faithfulness judge. These results are preliminary: the retrieval queries are paraphrases of a small reference set, and the end-to-end test is small and uses an automated rather than human judge, so they do not establish human-verified answer correctness or generalizable performance. 

---
# SAFETYEDIT: Sparse Internal Targets with Prompt-Wide Control for LLM Jailbreak Backdoors 

**Authors**: Yutong Ke, Yansu Wang, Zeyu Zhang, Xihao Guo, Chongwen Zhao, Kaizhu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.22640)  

**Abstract**: Model-editing backdoors aim to induce harmful responses under a trigger while preserving ordinary behavior on untriggered input. Recent model-editing approaches target affirmative response prefixes through rank-one updates to a small subset of model weights. Yet when this prefix-level target is satisfied, the model may still revert to refusal or produce a coherent, apparently compliant response that does not address the request. To overcome this limitation, existing internal-representation approaches shift focus from output tokens to dense, high-dimensional compliance directions in latent space. Existing representation-level approaches rely on preservation constraints to maintain conditionality, motivating us to ask whether a smaller internal target can offer a different effectiveness-preservation trade-off. Motivated by the comparable control capability of single-coordinate interventions, we investigate a sparse refusal-related MLP coordinate as this target. We propose SAFETYEDIT, which optimizes a single refusal-related coordinate and uses KV-cache marking in early attention layers to propagate a prefix-trigger signal across subsequent instruction positions. On some checkpoints a downstream MLP reader converts this signal into control of the coordinate, while on others the propagated signal alone suffices; the method is defined by the sparse target rather than by a fixed writer-reader pair. Across four open-weight LLMs and three dataset groups, SAFETYEDIT achieves the highest triggered Guard ASR in 11 of 12 comparisons and the highest mean StrongREJECT score in all 12, relative to the baselines evaluated for each model. Crucially, comprehensive ablation and behavioral audits confirm generally minimal changes in untriggered behavior and general capabilities in the tested configurations. 

---
# Inductive Learning of Concept Embeddings from Cross-Linguistic Colexification Patterns with Textual Attributes 

**Authors**: Arne Rubehn  

**Link**: [PDF](https://arxiv.org/pdf/2610.22638)  

**Abstract**: The advent of distributional semantics as a framework for computational semantics marked a paradigm shift in the nature of representations, by which meanings are represented as dense vectors commonly known as word embeddings. Computational approaches to historical or typological language comparison, on the other hand, still mostly rely on discrete representations, while the potential of embedding-based methods remains largely untapped in this context. Recent studies have proposed concept embeddings as a language-independent, onomasiological counterpart to word embeddings, based on networks of cross-linguistic colexification patterns. While the resulting concept embeddings reflect meaningful semantic properties and relations, they suffer from a lack of scalability, since they rely on transductive embedding approaches which only learn embeddings for nodes present in the training data. In this study, we present an inductive framework for learning concept embeddings, combining the structural properties of full and partial colexification networks with semantic information from short concept descriptions. This framework allows the embedding of unseen concepts based on a short definition. We show that the resulting concept embeddings correlate with various cross-linguistic measures of similarity or relatedness, while generalizing well to new concepts that were absent from the training data. 

---
# Measuring Mechanistic Uncertainty in Large Language Models via Causal Robustness 

**Authors**: Aneesh Komanduri, Majid Afshar, Timothy A. Miller, Yanjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.22636)  

**Abstract**: Large language models (LLMs) are increasingly used in critical domains, where their susceptibility to hallucinations makes reliable and interpretable uncertainty quantification essential. We introduce the concept of mechanistic uncertainty, which characterizes uncertainty through the stability of the internal mechanisms supporting a model's response. We propose the causal robustness margin, defined as the minimum internal intervention required to deviate the model's output, and operationalize it using Jacobian Lens (Jlens) concept steering vectors. Across three open-source models and five question-answering datasets, we show incorrect and uncertain responses exhibit smaller margins, indicating greater sensitivity to internal interventions, and the causal robustness margin complements conventional output-based uncertainty methods. Finally, in a controlled knowledge injection experiment, correctly learned facts show increased margins after training, suggesting that causal robustness provides a mechanistically grounded signal of uncertainty. 

---
# ToMStories: Interpreting Theory-of-Mind in Synthetic Narratives 

**Authors**: Melanie Subbiah, Chia-En Jacklyn Tsai, Kathleen McKeown  

**Link**: [PDF](https://arxiv.org/pdf/2610.22632)  

**Abstract**: Understanding people often requires inferring what they feel or want from how they behave rather than from what they state outright. We ask whether language models can make these inferences, and how they do so. We introduce Theory-of-Mind Stories (ToMStories), an evaluation dataset of ~6,000 short synthetically generated narrative vignettes, each paired with a multiple-choice or open-ended question about a character's emotion, fear, motivation, or emotional wound. These psychological states are conveyed implicitly (showing), explicitly (telling), or with conflict between dialogue and description. Evaluating thirteen open-weight models (1B-72B parameters), we find that the largest models match humans on multiple choice but not open-ended questions. Our analyses point to simplifications models make in both interpretation and generation of implicit psychological states, indicating that synthetic data can help us understand how models perform interpretation, but likely overestimates how well models interpret humans. 

---
# A Plausible Explanation Is Enough to Mislead Multimodal Judges 

**Authors**: Anglin Cai, Zhiqi Wang, Xiaozhi Yang, Yurun Lin, Dongwon Lee, Yuchen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.22631)  

**Abstract**: Multimodal large language models (MLLMs) increasingly serve as judges in training and evaluation pipelines, often assessing answers together with explanations. Although AI judges are sensitive to textual factors, it remains unclear whether ordinary plausible explanations can sway judgments requiring visual verification. We introduce JUPE, a visual-question-answering (VQA)-style benchmark that fixes answers while varying explanations.
Across 14 MLLM judges, ordinary explanations that sound plausible mislead judges, with effects comparable on average to deliberately persuasive arguments. Plain justifications and chain-of-thought rationales reduce rejection of incorrect answers by 12.7 and 13.8 percentage points, respectively. We call this failure plausibility capture. Using signal detection theory, we find a significant mean shift in decision criterion toward acceptance, without a clear change in mean discriminability. Explanations sway judgments across domains, explanation generators, and judging protocols, while tested mitigations offer limited, model-dependent protection. Such shifts can inflate performance estimates when judges evaluate answers with explanations. 

---
# Trajectory-Guided Steering: Mitigating Spurious Feature Bias in Clinical Classification 

**Authors**: Fardin Ahsan Sakib, Ziwei Zhu, Ozlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2610.22618)  

**Abstract**: Large language models rely on spurious features in clinical classification: swapping a single gender token in clinical notes alters gender-irrelevant diagnoses in up to 53.5% of cases. Existing inference-time corrections either need counterfactual data built around the specific spurious feature or behave unpredictably. We observe that how a model's preference between answers evolves across layers, its prediction trajectory, indicates for each document whether a correction is likely to help. We propose Trajectory-Guided Steering (TGS), which decides from the trajectory alone whether to correct each prediction and in which direction, with no counterfactual data and no knowledge of the spurious feature. On 16 clinical classification tasks from i2b2, TGS fixes 17.7% of errors while breaking only 1.3% of correct predictions and stays net-positive across all tested hyperparameter settings; all baselines are net-negative or near zero. The same corrections reduce reliance on the spurious feature: on the most affected diseases, gender-swap inconsistency falls sharply (hypertension 53.5% -> 38.0%, diabetes 35.1% -> 21.5%). Applying the correction to every document instead raises breaks on correct positive predictions from 0.1% to 6.4%, so per-document selection is essential. TGS also transfers to three model families and three other clinical and biomedical datasets. 

---
# When Does Uncertainty Measure RAG Hallucination Risk? 

**Authors**: Rajeel Ahmad Ansari, Tautik Agrahari  

**Link**: [PDF](https://arxiv.org/pdf/2610.22615)  

**Abstract**: Uncertainty can measure hallucination risk in retrieval-augmented generation (RAG), but only when retrieved evidence constrains the answer. When evidence is present, unsupported factual claims become less likely; when retrieval coverage is poor, the model can answer from internal knowledge and assign high probability to unsupported claims. We test this across four instruction-tuned models on HotpotQA and SQuAD. After matching faithful and hallucinated answers within word-count bins, length-only classifiers fall near chance, while perplexity reaches 0.81--0.91 AUROC on targeted unsupported edits. Varying retrieval depth shows why the score changes meaning: detection can weaken with only the top-ranked passage, then recover after adding a second passage. A combined uncertainty and semantic-drift score improves calibration, and measurements remain above chance on RAGTruth QA. Results support a conditional view of uncertainty as a useful measure of RAG hallucination risk. 

---
# Broad and Efficient PII Detection with Unified Labels 

**Authors**: Andy Hsu, Konstantin Berlin  

**Link**: [PDF](https://arxiv.org/pdf/2610.22597)  

**Abstract**: Personally Identifiable Information (PII) detection models must efficiently find identifiers across diverse contexts while maintaining a low false-positive rate (FPR) on high-volume traffic. Recent detectors increasingly rely on complex architectures that raise deployment cost, yet many are fine-tuned on only a few synthetic datasets that limit their coverage. Combining public datasets is challenging because their taxonomies differ, and even after categories are mapped to a common schema, labels can disagree on what counts as a detection, producing inconsistent training and evaluation signals. To address these problems, we present a practical 141M-parameter encoder-based detector trained on 1.12 million examples from 77 public datasets, relabeled under a detailed definition of broad contextual detection for 12 structured PII categories. Evaluated on the AI4Privacy and Nemotron-PII benchmarks at a low-FPR operating point, our model achieves the highest recall against larger baselines in 11 of 14 comparisons under our labels and 9 of 14 under the original labels. Its largest recall gains include useful additional coverage, such as multilingual disclosures and IP addresses embedded in URLs. It achieves this with 2-10x higher throughput and 94-97% less peak GPU memory than the baselines, delivering leading detection performance at substantially lower cost. 

---
# The Model Knows Better: Defending Medical Multi-Agent Systems Against Prompt Injection 

**Authors**: Fardin Ahsan Sakib, Ziwei Zhu, Ozlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2610.22594)  

**Abstract**: Medical multi-agent systems, in which LLM agents discuss a case, are increasingly studied for clinical decision support. Prompt injection, text that an attacker adds to a model's input, can manipulate LLMs into wrong decisions. How it affects these systems, and how to protect them, remains unclear. Across two systems, two model families and three benchmarks, a single injected passage succeeds in up to 61% of cases, and collaboration does not consistently protect against it. Existing defenses help only partly: they miss injections that look like ordinary clinical text. Yet when asked to check the injected text against the patient's findings, a model that followed the injection usually rejects it. We therefore propose a training-free defense that verifies the clinical judgments behind the system's decision. Following clinical practice, it separates the patient's findings from clinical judgments, such as diagnoses, recommendations and instructions, and checks each judgment against the findings alone, so the decision rests only on supported judgments. Our defense outperforms all baselines on every benchmark, lowering attack success on MedQA to 4-7% while maintaining accuracy on cases without an attack, and remains effective with a medical model and against defense-aware and adaptive attacks. 

---
# SAG2: Embedding-Free Multi-Hop Retrieval with BM25 and JEV 

**Authors**: Yuchao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.22588)  

**Abstract**: Strong multi-hop retrieval should be practical to deploy without expensive corpus preparation or complex database infrastructure. We present SAG2, an embedding-free retrieval architecture built around a single SQLite database, BM25, and contextual evidence decisions from JEV. SAG2 follows the query-time association principle of SAG (SQL-Retrieval Augmented Generation), while replacing its event-entity and dense indexes with a lexical index over original passages. Selected passages supply terms for further retrieval, and newly retrieved evidence guides subsequent decisions. This moves semantic evidence selection to query time, eliminating offline embedding computation, LLM-based event-entity extraction, and dedicated database servers. The resulting system combines lightweight preparation with strong retrieval accuracy: on 1,000-question subsets of MuSiQue, 2WikiMultiHopQA, and HotpotQA, SAG2 achieves the highest Recall@5 and Recall@10 among the compared methods, with Recall@10 of 86.62%, 97.43%, and 99.00%. It also reaches 81.10% document Recall@10 over the 511,962-document EnterpriseRAG-Bench corpus. Indexing all 11,656 MuSiQue passages takes 2.36 seconds, produces a 25.87 MiB database, and requires no model calls. SAG2 demonstrates that accurate multi-hop evidence retrieval can be supported by compact, rapidly prepared infrastructure, making it a practical foundation for knowledge retrieval applications. 

---
# Mitigating Demographic Disparities in Explanation-Human Alignment via Rationale Regularization 

**Authors**: Suryansh Talukdar, Taha Sarwat, Jonathan Rusert  

**Link**: [PDF](https://arxiv.org/pdf/2610.22586)  

**Abstract**: We study explanation fairness in hate speech detection. High classification performance does not always mean that a model produces equally plausible explanations for different demographic groups. Using the HateXplain dataset, we introduce Plausibility-EFG, a metric that measures the largest difference in explanation plausibility across demographic groups by comparing model-generated token attributions with human-annotated rationales. We evaluate BERT, RoBERTa, and ADV-RoBERTa, an adversarially trained model, using Integrated Gradients and SHAP. Our results show that ADV-RoBERTa improves classification performance and alignment between explanations and human rationales while reducing explanation disparities across demographic groups. These improvements are observed with both attribution methods and remain consistent across training, validation, and test splits. We also show that reducing prediction-level bias alone does not eliminate disparities in explanations, suggesting that predictive fairness and explanation fairness are related but distinct objectives. Finally, our fairness-performance analysis shows that stronger adversarial regularization can further reduce Plausibility-EFG, although it may cause a small decrease in macro-F1. Overall, this work presents explanation fairness as an important objective in hate speech moderation and introduces Fair-REF, a framework designed to reduce disparities in model explanations across demographic groups. 

---
# Who Do Perspectivist Models Predict Well? Annotator Agreement, Sparsity, and Performance 

**Authors**: London Lowmanstone IV, Vaagishwaran Sivakumar, Anwesha Samaddar, Akul Mundada, Harnoor Kaur, Maria Gini  

**Link**: [PDF](https://arxiv.org/pdf/2610.22567)  

**Abstract**: Successful completion of perspectivist natural language processing (NLP) annotation tasks requires algorithms to be aligned across diverse perspectives in order to predict how different humans will rate text. This work studies how the performance of such algorithms varies for annotators who behave differently than others, either by providing annotations that diverge from others' responses or by providing fewer annotations altogether. To aid with characterizing model performance, we curate the largest collection to date of perspectivist NLP datasets with ordinal annotations. After evaluating multiple classes of perspectivist algorithms across each of the datasets, we model the relationship between individual annotator idiosyncrasy and algorithm performance. In general, we find that algorithms perform better on annotators who tend to agree with others, but that annotator response rate does not have a statistically significant positive correlation with algorithm performance. 

---
# Cultural (In)Competence: Investigating Culture-Specific Stereotype Detection Capabilities of LLMs 

**Authors**: Maximilian Spliethöver, Jannes Lötfering, Dominik Bachmann, Alexander Espig, Willem Zuidema, Katrin Schulz, Henning Wachsmuth  

**Link**: [PDF](https://arxiv.org/pdf/2610.22545)  

**Abstract**: Stereotypes are subjective constructs that are mediated by the cultural context in which they are viewed. Successfully modeling them thus also requires cultural competence. When using large language models (LLMs) for stereotype-related tasks, such as their detection, it is therefore crucial that models can access relevant cultural knowledge and adjust responses accordingly. Since LLMs have shown disparities in what cultural knowledge is available to them, their capacity regarding stereotypes may also be disparate across languages and cultural contexts (e.g., for non-English languages or for non-US contexts). In this paper, we systematically investigate the effectiveness of LLMs for culture-specific stereotype detection across three factors: (1) the prompting language, (2) the instruction tuning language, and (3) the fine-tuning dataset size. We conduct experiments for US, Dutch, and Chinese cultural contexts. Our results suggest that LLMs do not reliably account for cultural contexts and overpredict the presence of stereotypes. Non-English prompts seem to improve consideration of cultural context and thus lead to more effective predictions. Moreover, we find that additional instruction tuning reduces false-positive rates and that training data size moderates detection effectiveness non-monotonically. 

---
# HeRE: A Novel Benchmark for Hebrew Relation Extraction via LLM-Guided Denoising of Knowledge Graph Alignments 

**Authors**: Ron Keinan, Amir David Nissan Cohen, Reut Tsarfaty  

**Link**: [PDF](https://arxiv.org/pdf/2610.22542)  

**Abstract**: While LLMs demonstrate strong capabilities in English Relation Extraction (RE) tasks, a significant gap remains in assessing their RE performance in low-resource languages, due to a lack of standardized benchmarks. Creating native, language-specific annotated RE benchmarks is expensive and requires human expertise, while relying on translation yields suboptimal, inaccurate datasets that introduce English-centric biases. To tackle this, we propose a scalable, language-agnostic methodology to create RE data by denoising naturally occurring yet noisy alignments of knowledge bases and corresponding texts. We define a proper alignment via a Natural Language Inference (NLI) task, where a relation triplet is entailed by an input text. We empirically assess five denoising strategies, and demonstrate that multilingual models outperform language-specific models in the few-shot LLM setting, while Hebrew-specific encoders remain competitive under NLI fine-tuning and are strongest in the label-free cross-training setting. We further find that prompting in the native language achieves performance on par with or slightly exceeding English prompts, confirming that the pipeline operates naturally in the target language. The best open-weights model, Gemma-4-31B-it (F1\,=\,0.915), matches the best API model, GPT-5.4 (0.921), and reaches F1\,=\,0.844 on a fully held-out, dual-annotated test set. Applying the optimized pipeline, we release HeRE: a novel, massive, 1.5-million-instance silver dataset for Hebrew RE, designed to support data-intensive experimentation and model fine-tuning. HeRE bridges a critical gap in standard Hebrew RE evaluation, while providing the NLP community with a reproducible, language-agnostic pipeline for RE dataset construction in other low-resource settings. 

---
# Pearl-Chat: Engineering a native Luganda language model from first principles in JAX and Flax NNX 

**Authors**: Wesley Kambale, Roland Sankara  

**Link**: [PDF](https://arxiv.org/pdf/2610.22529)  

**Abstract**: Autoregressive foundation language models predominantly target high-resource Indo-European languages, imposing computational and morphological penalties on under-represented African languages. Generic multilingual tokenizers routinely exceed fertility rates of 3.2 tokens per word on agglutinative Bantu languages, degrading context efficiency and inflating latency. We present Pearl-Chat, an 8,473,122-parameter ($\sim$8.5M) causal decoder-only transformer engineered for Luganda (Oluganda, ISO 639-2: lg), a low-resource Niger-Congo Bantu language spoken by over 10 million people in East Africa. Built from first mathematical principles in JAX and Flax NNX, Pearl-Chat uses explicit functional state decomposition (this http URL and this http URL), deterministic resumable data streaming via Google Grain, and SPMD tensor parallelism using this http URL. We introduce a byte-level byte-pair encoding (BPE) subword tokenizer trained from scratch on curated Luganda corpora ($V = 8,192$), achieving 4.88 characters per token compression and 1.60 tokens per word fertility, a two-fold compression improvement over LLaMA-3 (128K) and Gemma (256K). The model is trained with a two-stage curriculum: Stage 1 pre-training on 59,501 sequences across parallel, biblical, encyclopedic, and journalistic corpora, and Stage 2 instruction fine-tuning on 510 dialogues with assistant-only loss masking. Across 4,000 optimization steps, validation cross-entropy loss drops monotonically from 9.4695 (perplexity: 12,958.98) to 0.1818 (perplexity: 1.20), a decrease of 9.2877 nats. On a three-tier held-out diagnostic benchmark, Pearl-Chat achieves 0.9792 ROUGE-L and 0.9405 BLEU while maintaining concordial agreement across Bantu noun classes. We provide a Python Server-Sent Events (SSE) streaming runtime and open-source all weights, tokenizers, and code. 

---
# Literal Lock-In: How Test Language Shifts Literal Attraction in Typed Decision Models on Bengali Idioms 

**Authors**: Adib Sakhawat, Syed Rifat Raiyan  

**Link**: [PDF](https://arxiv.org/pdf/2610.22526)  

**Abstract**: Decision models, non-generative scorers that report a probability over caller-supplied options and emit no text, are increasingly deployed as routers, classifiers and guardrails. Because they generate nothing, rationales and chain-of-thought do not apply. We audit three: an encoder scorer, a decoder scorer and a closed API, entirely through interventions on their input, using expert-annotated Bengali idioms carrying a gold figurative and a gold literal gloss. Dropping an idiom's own literal gloss into the option list draws every arm toward it, taking 65% to 96% of each arm's remaining Bengali errors. A model-by-language matrix over one shared item set shows how far this depends on the test language: every arm is 1.9 to 6.4 times more captured in Bengali than in English on the same paired entries, the commercial model's apparent resistance is a property of its test language, and capability does not order the failure. Calibration degrades in all three arms, and on the open arms confidence inverts, so gating on it is worse than gating at random. On a semantics-free option list, where accuracy is at chance, the best-calibrated arm still reports mean confidence 0.82. Code and data: this https URL. 

---
# alkhder at AlexandriaX-2026: Context-Aware English-to-Dialectal Arabic Dialogue Translation with QLoRA-Fine-Tuned NileChat-3B 

**Authors**: Hasan Alkhder, Maher Pasa, Maria Hanini, Amro Najjar, Ahmet Zengin  

**Link**: [PDF](https://arxiv.org/pdf/2610.22525)  

**Abstract**: This paper describes team Alkhder's submission to Subtask 1 of the AlexandriaX-2026 shared task, which asks systems to translate English dialogue turns into thirteen Arabic dialects. We fine-tune NileChat-3B-Base with QLoRA on the Alexandria parallel dialogue corpus, training one set of adapter weights for all dialects. Each turn is translated conditioned on the target country and dialect, the domain, the participant roles, the speaker-addressee gender direction, and the preceding dialogue context. At inference, conversations are translated autoregressively and the model is fed its own previous outputs back as context without using any gold target-side test information. Under the constrained track, using only Alexandria data, it trains in about one hour and translates the private test set in under two hours. The system reaches a country-macro-averaged spBLEU of 20.36 and chrF++ of 36.64 on both tracks. 

---
# Learning User Representations to Improve Simulation Fidelity 

**Authors**: Shuhaib Mehri, Philippe Laban, Michel Galley, Dilek Tür, Jennifer Neville, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.22518)  

**Abstract**: User simulators enable interactive training and evaluation of AI assistants at scale. Yet current simulators struggle to reproduce the diverse behaviors of real users, leading to assistants that generalize poorly to underrepresented users. We identify user representations as an underexplored bottleneck in user simulation. User representations are natural-language descriptions that condition simulators to reproduce a particular user's behavior. A representation's quality becomes apparent in hindsight, through the behavior it induces in a simulator. To evaluate and optimize user representations, we formulate a conversation reconstruction task: given an original conversation, we generate a representation, condition a simulator on it, simulate a conversation, and then measure fidelity by comparing user behavior in the original and simulated conversations. This comparison also yields hindsight feedback, which identifies where the representation falls short and how it can be improved. We introduce hindsight distillation, an online training procedure that distills feedback-conditioned improvements into the representation generator, enabling it to generate improved representations directly from the original conversation. We present UserRepGen-4B, a representation generator trained with hindsight distillation. Across 700 conversations spanning seven datasets, hindsight feedback helps improve user representations at test time, and UserRepGen-4B generates representations that enable more faithful simulation than those generated by GPT-5.5. Beyond conversation reconstruction, learned representations improve simulation fidelity for unseen tasks, better preserve the population-level diversity of real users, and better distinguish users across conversations. 

---
# alkhder at ImageEval 2026 Shared Tasks: Constrained Selection and Rotation-Averaged Logit Scoring for Multimodal Hallucination Detection 

**Authors**: Maher Pasa, Maria Hanini, Amro Najjar, Hasan Alkhder  

**Link**: [PDF](https://arxiv.org/pdf/2610.22515)  

**Abstract**: We describe our submission to Subtask~1b of ImageEval 2026. The task consists of selecting the single visually valid statement from three candidates paired with an image, and is in both English and Modern Standard Arabic (MSA). Our system uses a pre-trained 7B vision--language model in a zero-shot setting, without task-specific training or fine-tuning. We formulate the task as a constrained selection problem and choose the answer using next-token log-probabilities. To reduce answer-position bias, we apply a Latin-square rotation to the candidate statements. Our system achieves CI scores of 0.0510 on the English test set and 0.0960 on the MSA test set, reducing CI relative to the official 3B baselines reported on devtest by 83.7\% and 80.4\%, respectively. 

---
# AuditKIT: A Unified Evaluation Spine for Benchmarking, Judging, and Comparing Large Language Models and Agentic Systems 

**Authors**: Shreeyans Arora, Utsav Avaiya, Zera Lyngkhoi, Hem Gosalia, Ram Mohan Rao Kadiyala, Vinay Kumar Sankarapu, Pratinav Seth  

**Link**: [PDF](https://arxiv.org/pdf/2610.22511)  

**Abstract**: Evaluating language models and agents requires combining benchmark accuracy, judge-based assessment, runtime performance, tool-use behavior, and retrieval quality. Differences in interfaces, metric semantics, and trace formats complicate these comparisons and can hide failures in agentic workflows. We present AuditKIT, a source-available evaluation library that records these measurements in one result representation, with direction-aware metric contracts, configuration fingerprints, and extensible metrics and backends. For agents, it provides turn-aware tool-call matching, separates parallel from sequential execution, and distinguishes verified task outcomes from trajectory-level diagnostics. Missing or unobservable evidence is marked unscored rather than counted as failure. For retrieval-augmented generation, it combines ranked retrieval metrics, judged grounding, and deterministic evidence checks. Seven case studies spanning model comparison, serving runtimes, retrieval, tool calling, and interoperability show how answer extraction, runtime behavior, and metric selection change evaluation outcomes. 

---
# RealCLI: Toward a Unified Evaluation of Human-Agent Collaboration for CLI Agents 

**Authors**: Yunqi Liu, Peiyu Wang, Bingli Wang, Leo Luo, Xuanwu Song, Xudong Ma, Hongxin Li, Yaxin Du, Weicheng Gu, Fanglin Xu, Yi Xing, XiangLong Liu, Jujie He, Zhange Zhang, Huatong Song, Zhaoyi Li, Deyang Kong, Chuan Hao, Peihao Wu, Mingjie Tang, Ran Tao, Jian Yang, Wayne Xin Zhao, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2610.22507)  

**Abstract**: Command-line interface (CLI) agents increasingly execute longer-horizon tasks, yet existing benchmarks either assume that user instructions remain fixed throughout execution or evaluate task outcomes without capturing runtime collaboration quality. In this paper, we present REALCLI, a paired benchmark that unifies the evaluation of task execution and human-agent collaboration quality on realistic CLI tasks. To comprehensively measure collaboration, REALCLI builds a five-dimensional framework, operationalized into 22 sub-dimensions and 51 interaction scenario templates. REALCLI provides two modes for paired comparison: REALCLI-EXEC supplies complete specifications upfront, whereas REALCLI-COLLAB introduces a scripted user that delivers progressive information, corrections, constraint revisions, and authorization decisions. Evaluations reveal profound capability gaps in frontier agents: the best configuration attains an average score of only 49.2%, and when evaluated by the joint collaboration score ($J_{\mathrm{collab}}$), the peak performance reaches merely 37.6%. These results show that task effectiveness diverges sharply from collaboration quality, with models consistently struggling in steerability, trust calibration, and safety boundary management. REALCLI provides a rigorous, reproducible foundation for measuring and advancing collaborative CLI agents. 

---
# From Parallels to Divergence: Language Models' Learning of Action Nominal Constructions 

**Authors**: Iskar Deng, Shane Steinert-Threlkeld  

**Link**: [PDF](https://arxiv.org/pdf/2610.22468)  

**Abstract**: Action nominal constructions (ANCs) express events in nominal form, with their argument marking following patterns found in clauses or ordinary noun phrases. In this paper, we investigate how different grammatical configurations affect language models' learning of ANCs and related constructions, and how these learning preferences compare with typological distributions in natural languages. We use a grammar-engineered method for synthetic corpus construction that combines shared semantic input from natural corpora with computational grammars. We construct 96 target languages varying in clause word order, noun-phrase word order, alignment, complement system, and ANC strategy. For each language, we train GPT-2-small models from scratch and evaluate them using targeted minimal pairs covering 34 grammatical phenomena. The results show typological parallels in basic clause configurations but divergences in complement systems and ANC strategies, with models also learning combinations rare or unattested in natural languages. This partial alignment suggests that natural language distributions may reflect both general learning pressures and functional demands not fully captured by model learning experiments. 

---
# Where Prefill Expert Reduction Hurts MoE Models: A Few Prompt Tokens Carry Most of the Short-Answer F1 Loss 

**Authors**: Zhiyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.22454)  

**Abstract**: Running prompt tokens through fewer routed experts during prefill is a simple way to make mixture-of-experts (MoE) serving cheaper, yet it is unclear what this damages. We locate the damage with token-, layer- and attention-path interventions on short-answer question answering in four MoE families. At half the experts the F1 loss of three families is mostly answer form: the model still finds the answer but stops answering briefly. In Qwen3-30B-A3B this loss is made at the two chat-template tokens before the answer, and in DeepSeek-V2-Lite at the first 8-32 prompt tokens, where its instruction sits; by answer likelihood these positions account for about half and a fifth of the damage. Reduction changes the first token's attention mass, norm and largest activation by less than 4%, and restoring super experts or massive activations recovers at most about a quarter of the loss, well below keeping the sensitive positions full. On real prompts with about 4% of tokens at full experts, placing them on the task interface keeps F1 within one point of the unmodified model, while placing them at random in the documents is about 3 points worse (two of five prespecified primary comparisons; the others were undecided or not run). Because the answer boundary always ends a single-turn prompt, and DeepSeek's loss was no longer detectable once its instruction moved away from the start, the sensitive tokens lay at the prompt ends in every sensitive layout we found, so rules that keep the ends are not distinguished from rules that read application marks. Time to first token falls about 14% on 11K-token prompts and not at 2K tokens. 

---
# Embeddings Are Overpriced? A Frozen 16 MB Vocabulary for Small and Ternary Language Models 

**Authors**: Sai Kiran Bathula  

**Link**: [PDF](https://arxiv.org/pdf/2610.22428)  

**Abstract**: In a small language model the vocabulary is the single largest part. In Gemma 3 270M the embedding table holds 168M of 268M parameters, and because quantisation methods leave the embedding in 16-bit, it is also most of the file once the rest of the model is compressed. We ask what that table actually buys. We replace it with one frozen table of 512-bit codes, 16 MB, with no trainable parameters, derived once from a pretrained embedding by principal-component projection and sign binarisation, and we train the same Gemma-shaped body from scratch on the same 2.05B tokens with and without it. The learned table buys 0.20 nats of validation loss for 2.6 times the trainable parameters, but on LAMBADA, HellaSwag and PIQA the frozen-table model is better or tied; only ARC-Easy favours the learned table. With a ternary BitNet b1.58 body the whole model is a 46 MB file, still ties the float model on LAMBADA and PIQA, and on a laptop CPU generates 615 to 740 tokens per second in 62 MB of memory, five to six times the speed of the float model's 8-bit GGUF in this http URL at a seventh of the memory. The frozen table wins by 1.7 to 2.3 nats on rare words, accepts 10,000 new words after training with a 3% cost where the learned table collapses by 43%, and all of it is released with the weights, the data and every evaluation answer. 

---
# AI-Driven Reproducibility and Analytical Robustness in Scientific Discovery 

**Authors**: Shicheng Liu, Krista Opsahl-Ong, Andrew Drozdov, Julian Killingback, Arnav Singhvi, Ronak Pradeep, Michael Bendersky, James C. Dickerson, Monica S. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2610.23845)  

**Abstract**: Reproducible and analytically robust research findings are essential to scientific progress, but evaluating these properties typically requires costly manual effort. We investigate whether LLM agents can help automate reproduction and robustness analysis, using a medical case study grounded in the Surveillance, Epidemiology, and End Results (SEER) database, a widely used U.S. cancer registry underlying tens of thousands of published studies. First, we introduce SEER-Repro, a benchmark of 20 published SEER-based papers together with the data and resources required to reproduce their findings. Second, we create an AI agent that successfully reproduces 77.4% of 2,190 atomic claims, while providing executable baselines for further analysis. Third, we develop an agentic analysis pipeline to test the robustness of the medical findings by identifying analytical choices underlying each finding, proposing defensible alternatives, and rerunning the analysis to determine whether those choices affect the conclusion. Across published papers in SEER-Repro, our pipeline identifies potentially consequential analytical choices in 9.5% of fully reproduced findings and flags them for expert review. These results show how LLM agents can support scientific review by reproducing findings, systematically testing their robustness, and directing expert attention toward potentially consequential analytical choices. 

---
# VocalGrad: Evaluating Acoustic Perception in Audio Language Models 

**Authors**: Koki Ryu, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2610.23745)  

**Abstract**: Recent large audio language models (LALMs) have extended their capabilities beyond lexical speech understanding to tasks requiring paralinguistic and non-linguistic audio understanding, such as emotion recognition. However, these models still struggle to reliably utilize acoustic information, particularly when it co-occurs with spoken linguistic content. Existing benchmarks do not clearly distinguish acoustic perception from higher-level reasoning, making it difficult to identify the source of these failures. To address this issue, we introduce VocalGrad, a benchmark designed to evaluate LALMs' ability to perceive low-level acoustic variations. VocalGrad formulates this problem as a simple binary classification of temporal change direction, enabling evaluation largely independent of higher-level reasoning or semantic understanding. Our benchmark results reveal a substantial gap in acoustic perception between human annotators and state-of-the-art LALMs: while human annotators achieve high accuracy, LALMs perform at near-chance levels. We further conduct linear probing to identify structural bottlenecks in these models. Our analysis reveals two distinct failure modes: for some categories, task-relevant information is preserved in internal representations but not utilized by the language model head; for others, the information becomes progressively less linearly decodable across model layers. Overall, our results reveal substantial limitations of LALMs in modeling acoustic variation under controlled settings and expose representation differences not observable from text outputs. 

---
# RedDarwin: Evolving Jailbreak Skills against Frontier LLMs 

**Authors**: Zhouyuan Ma, Yutao Wu, Hanxun Huang, Xiang Zheng, Yi Liu, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.23726)  

**Abstract**: Red teaming must continually adapt as frontier LLM safeguards evolve, yet existing approaches largely search for adversarial prompts or optimize within fixed attack procedures. We observe that even when an attack becomes ineffective, its underlying mechanisms can remain useful and can be revised or recombined to construct stronger attacks. Based on this insight, we introduce RedDarwin, an agentic framework that shifts the unit of jailbreak evolution from prompts and fixed attacks to executable attack skills. RedDarwin consists of 47 reusable base skills and autonomously revises, composes, executes, and selects them using target-model feedback, allowing successful discoveries to become building blocks for subsequent evolution. RedDarwin produces 65 evolved skills, improving 44 of 47 base attacks and increasing their mean attack success rate (ASR) on AdvBench from 18.8% to 47.3%. These evolved skills consistently outperform baselines on the held-out StrongREJECT and frontier LLMs unseen during evolution; with only 10 target-model queries, RedDarwin achieves 100% ASR on GPT-5.6-Sol, 92% on Claude-Sonnet-5, and 80% on Gemini-3.8-Flash. Analysis of the evolution trajectories further shows that strong attacks often emerge from novel combinations of reusable mechanisms rather than from individually dominant ones. We believe this paradigm can accelerate adaptive red teaming and inform stronger safeguards, creating complementary value for both safety evaluation and defense. 

---
# Beyond the Benchmark: Conditioning, Domains and Failure Modes of Qwen3-TTS 

**Authors**: Séverin Baroudi, Yanis Labrak, Petr Motlicek, Hervé Bredin, Ricard Marxer  

**Link**: [PDF](https://arxiv.org/pdf/2610.23678)  

**Abstract**: Qwen3-TTS reports state-of-the-art zero-shot voice cloning, but its technical report leaves open how the model should be conditioned, how it behaves outside read speech, and how it fails. We characterise its two cloning interfaces, in-context learning (ICL) and X-Vector, against Chatterbox, XTTS-v2 and F5-TTS on read, meeting, emotional and air-traffic-control (ATC) speech, using eighteen objective measures, an LLM judge, a spoofing countermeasure and a listening test. Neither interface dominates: ICL clones timbre more closely from correctly transcribed references, X-Vector clones timing, and ICL degrades sharply when a short reference transcript is wrong. Domain explains four times more variance than system choice. Qwen3-TTS leads on intelligibility and pitch register but trails on prosody, and the objective measures rank systems differently from listeners. An off-the-shelf countermeasure flags its clones no more often than genuine recordings, and on ATC radio it fails for every system. Code and audio are released. 

---
# PresenceBench: An Exploratory Evaluation of Large Language Model Adaptation to User State 

**Authors**: Peña Arroyo Gallego, Mauro Ruiz Moreno, Alejandro Acien, Sergio Romero-Tapiador, Teresa Arroyo-Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2610.23625)  

**Abstract**: Presence-aware intelligence is the capacity of an AI system's intelligence layer to adapt appropriately to the functional state of the person it serves (here readiness, flow, and stability) while preserving task quality and safety. We introduce PresenceBench, an instrument for measuring it. PresenceBench evaluates how the intelligence layer uses a supplied state, not how that state is sensed, and breaks the capability into measurable properties. Two are tested here. Receptivity: does the model take up the state and give a better response than with no signal? Specificity: does any improvement depend on the correct state, rather than on extra structured context? A third, adaptation across states, is a required endpoint not measured in this release. A five-arm causal design answers both questions, against each model's own baseline and an equally structured mismatched-state control, scored by a correctness-gated human paired comparison. The exploratory evaluation covered four models across eighteen single-turn, six fixed three-turn and six simulated natural-conversation scenarios. Supplying a synthetic state, raw or translated into a response policy, gave human-preference point estimates above 0.5 for all four models (Claude Sonnet 5, DeepSeek V4 Pro, Gemini 3.6 Flash, GPT-5.6 Terra), with three of eight 95% intervals excluding chance. Specificity was not resolved: no per-model interval excluded chance, and the design cannot yet separate the state values from a generic instruction to adapt. Policy delivery reduced priced cost and output tokens against each model's baseline; these are resource-use changes, not efficiency gains, because correctness non-inferiority was not established. Safety-floor failures were rare (13 of 360 cases), with a non-significant excess under policy delivery. We report the instrument, the frozen results, and the controls a confirmatory study needs. 

---
# Not Every Mention Is a Claim: Rethinking Object Hallucination Detection in Vision-Language Models 

**Authors**: Hsin-Ling Hsu, Neeraja Kirtane, Kuan-Hao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.23610)  

**Abstract**: Object hallucination in vision-language models (VLMs) is usually measured by checking each object a caption names against the image. Most benchmarks and detectors alike treat every object mention as a claim that the object is present. This assumption fails when a caption names an object without asserting it, as in "a bench for people to sit on", which mentions people without claiming that any are in the image. We call such mentions non-assertive. Treating them as hallucinations inflates the error rates attributed to models that use such language. To separate them from real hallucinations, we introduce NAME-Bench (Non-Assertive Mention Evaluation Benchmark), 25,497 object mentions from captions that five VLMs generate for COCO images. Each mention is labelled concrete (asserted and present), hallucinated (asserted and absent), or non-assertive, and non-assertive mentions are further divided into four subtypes: depictive, purpose, negation and hedged. All the hallucination detectors that we evaluate flag non-assertive mentions as hallucinations, at rates between 14% and 80%. Adding non-assertive mentions to detector training as negatives largely corrects this on LLaVA-1.5. On the four newer VLMs, a comparable correction takes five to ten times more data, and the detectors then catch fewer real hallucinations. These findings show that current detectors conflate mentioning an object with asserting it, and that this conflation is harder to undo on the four newer VLMs than on LLaVA-1.5. Hallucination evaluation should therefore separate what a caption asserts from what it merely mentions. 

---
# Enhancing RAG-based Medical VQA with Gated Multimodal Representations 

**Authors**: Daniela Jimenez-Pano, Hugo Jair Escalante, Delia Irazú Hernández-Farías  

**Link**: [PDF](https://arxiv.org/pdf/2610.23588)  

**Abstract**: Medical Vision-Language Models (Med-LVLMs) have shown strong performance in Medical Visual Question Answering (MVQA), yet they may still produce factually inconsistent responses. Retrieval-Augmented Generation (RAG) can mitigate this issue by incorporating external clinical evidence; however, retrieval is often guided by a single modality, potentially overlooking the joint semantics of medical images and clinical questions. We introduce a multimodal RAG framework that learns a joint query representation through a Gated Multimodal Unit, enabling evidence retrieval to leverage information from both modalities while keeping the underlying Med-LVLM frozen. We evaluate the proposed approach on benchmark datasets. Results show that the multimodal query representation consistently outperforms a reference Med-LVLM and remains competitive with state-of-the-art methods. Our findings demonstrate the potential of multimodal query representations for more effective evidence retrieval in Medical VQA. 

---
# Execution Before Memory: Diagnosing Memory Failures in Video World Models 

**Authors**: Weiyang Jin, Jiwen Yu, Sihan Xu, Yuwei Niu, Xihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.23542)  

**Abstract**: A memory test for a video world model should say whether the scene was set, whether the asked change was carried out, and whether the last frame shows the result. Current tests fold these into one number, so a low score is easy to read as forgetting. When the change is not carried out, the hidden state transition cannot appear, so the test is not really a test of memory. We introduce MIST, an evaluation that separates the change from the memory it should leave behind. The model first sets the scene while the change stays hidden, and only later is asked to show it, with leakage before that point kept under control. This separation locates the bottleneck: the model sets the scene but does not carry out the instructed change. The same gap appears in general memory evaluations. On that basis we propose M3-Agent (Make the Missing Move), which recursively revises the details of the input instruction so that the model successfully carries out the requested instruction. Relative to the baseline, it more than triples the MIST reveal score. On the benchmarks, WRBench's score T rises by 26.1% and MBench-T's score U by 52.7%. The method meets the problem from the execution side: it shows what the frozen model can do once the move is stated, and it sets a stricter baseline. 

---
# "You" Is Not Toxic: How LLM Labelers Rewrite the Web's Conversation Laws 

**Authors**: Liang Yue, Eric A. Vance  

**Link**: [PDF](https://arxiv.org/pdf/2610.23446)  

**Abstract**: What the Web community believes about toxic conversations, such as that toxicity rises in longer and deeper threads, was measured mostly with Google's Perspective API, which is shutting down; large language models (LLMs) will make the next measurements. We find that LLM labelers mistake second-person address for toxicity, and that this one bias manufactures laws that human labels do not show. On 1.25 million news comments with crowd labels, Claude Haiku flags 12% of the non-toxic comments that contain no "you" and 51% of those with four or more; a single crowd rater flags 6% and 10%. Deeper replies address someone more often, so under Haiku toxicity rises with reply depth (+0.93 percentage points per level), while under the crowd's labels it slightly declines (-0.12). Rewriting "you" in the third person removes two thirds of the false rise; flagging only as many comments as the crowd does not remove it, and showing Haiku the parent comment enlarges it. All seven LLMs we test over-flag "you", and at the crowd's flag rate, without the parent, any false rise of the open ones vanishes at a fixed number of second-person words; Haiku also over-flags "you" on Reddit, Gab and Wikipedia talk pages. Agreement with human labels does not reveal the bias: we prove that a fixed Cohen's kappa leaves a law nearly unconstrained. We therefore certify the law instead of the labeler. Threadsafe uses the labeler's scores to choose which comments human raters label and to predict the others, and corrects the predictions with the human labels; linear laws are estimated without bias whatever the labeler does. In simulated audits its intervals for the laws keep nominal coverage from 250 labels on, where uniform-sample methods such as PPI++ fall short, and at 1,000 to 8,000 labels are 0.75 times as wide as those of design-based supervised learning (DSL); a thousand human labels rule out Haiku's rising law. 

---
# ScoutRoute: Efficient Mixture-of-Experts Routing Using Internal Language Model Signals 

**Authors**: Rajdeep Pandhare, Delzaan Bharucha  

**Link**: [PDF](https://arxiv.org/pdf/2610.23443)  

**Abstract**: Efficient expert routing is a fundamental challenge in Mixture-of-Experts (MoE) systems, particularly for heterogeneous collections of independently trained Large Language Mod els (LLMs). Existing routing approaches typically rely on jointly trained gating networks, prompt-based predictors, or computationally expensive multi-model inference, limiting their flexibility and efficiency. This paper presents ScoutRoute, a lightweight post-hoc routing framework that predicts expert competence using internal language model signals extracted from a single forward pass through an inexpensive scout model. Hidden-state representations are used to estimate cross-model competence through lightweight logistic probes, while logit-derived confidence measures are incorporated within a domain-aware routing strategy for efficient expert selec tion without modifying the underlying language models. ScoutRoute isevaluatedon fiveopen-weightlanguagemodels across fourbenchmark datasets covering mathematical reasoning, code generation, commonsense reasoning, and multidis ciplinary knowledge. Experimental results show that ScoutRoute consistently outperforms individual routing signals while requiring only a single scout-model inference. The proposed competence probes also achieve superior calibration compared with conventional confidence measures, demonstrating that internal language model representations provide transferable information for accurate and efficient expert routing across heterogeneous language models 

---
# Robust Function Calling for LLM Agents Under Interface Drift 

**Authors**: Aniruddha Chattopadhyay, Kanishka Halder, Partha Sarathi Roy  

**Link**: [PDF](https://arxiv.org/pdf/2610.23342)  

**Abstract**: Tool-augmented language-model agents rely on interface text: tool descriptions, argument schemas, and the candidate tool list shown at inference time. These interfaces often change after deployment even when the underlying tool semantics remain fixed, creating a practical reliability problem for function-calling systems. We study this failure mode as interface drift and evaluate whether a lightweight, training-free repair layer can recover robustness without retraining the base model. Our method, SchemaShield-Lite, combines schema validation, candidate-set retry for missing tool names, and one-shot constrained repair. On Qwen3.5-9B, it raises DICE-300 accuracy from 0.717 to 0.813 and BFCL-200 accuracy from 0.580 to 0.805, recovering 19/22 and 33/45 drift-induced failures with no observed repair harms. Naive retry does not explain the gains. Ablations show that candidate-set drift, especially distractor tools, accounts for most observed damage, while description and schema edits alone are less harmful. The results position validation-and-repair as a practical robustness layer for deployed LLM-agent tool use. 

---
# Self-Reflective Chain-of-Thought (SRCoT): Evidence-Based Text Generation via a Closed-Loop Retrieve--Reflect--Revise Pipeline on Edge LLMs 

**Authors**: Fabian Broszio, Michael Hoffmann, Adrian Paschke  

**Link**: [PDF](https://arxiv.org/pdf/2610.23311)  

**Abstract**: Large language models (LLMs) can generate fluent outputs that lack source support or conflict with local operational constraints. We introduce Self-Reflective Chain-of-Thought (SRCoT), a modular architecture that organizes generation as a closed Retrieve--Reflect--Revise loop. SRCoT separates environment state, persistent knowledge, accumulated experience, and short-term memory into provenance-labeled stores. BM25 retrieval supplies local context; subsequent passes expose intermediate drafts, missing-information markers, follow-up queries, and optional claim--evidence mappings. These artifacts inform retrieval, revision, clarification, or abstention decisions. We examine the architecture through qualitative, trace-based case studies in two fully synthetic settings: emergency medical services and Munich-Sim, an emergency-operations simulation. Local prototype runs on an Apple M4-family laptop with 128 GB memory illustrate source-labeled documentation, explicit gap marking, and structured dispatch-style outputs. The traces also reveal provenance-assignment errors, incomplete evidence coverage, and unsafe assumptions that remain possible despite prompting constraints. These cases demonstrate report-level inspectability and local execution, not comparative superiority, hallucination reduction, clinical validity, or operational safety. The architecture separates retrieval and control from the model interface, but behavior across different models remains to be evaluated. SRCoT is a research proof of concept requiring human review, with its principal contribution lying in an inspectable control procedure rather than a new model. 

---
# SimOPD: Relational Knowledge Transfer for On-Policy Distillation 

**Authors**: Kai Tang, Xing Zheng, Haobo Wang, Guangcheng Zhu, Shenzhi Yang, Zhengqing Zang, Yingfan Ma, Bowen Song, Weiqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.23306)  

**Abstract**: A key challenge in on-policy distillation (OPD) is extracting effective supervision from teacher knowledge. Supervision from token probabilities may underutilize internal knowledge, while direct hidden-state matching forces the student to shift to the teacher's feature space and complicates cross-architecture transfer. Inspired by contrastive learning's use of relationships between instances, we propose \textbf{SimOPD} to transfer teacher internal knowledge through response similarities. Specifically, SimOPD uses paired memory banks of student-generated responses to extract supervision from the teacher's internal representations as similarity distributions. The student then matches these distributions within its own representation space, retaining representational freedom and removing the need for cross-model projection. Extensive experiments show that SimOPD improves the overall average over the strongest baselines by up to 2.6 and 3.1 percentage points in same- and cross-architecture settings. 

---
# Annotrace: A Workbench for Step-Level Annotation of Agent Traces 

**Authors**: Arun Josephraj Arokiaraj, Chenyi Huang, Sahan Bulathwela, Zekun Wu, Bin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.23284)  

**Abstract**: Every study of why LLM agents fail begins by labelling traces by hand, and every such study builds its own throwaway tool. Observability platforms record agent traces but offer no structured labelling; annotation tools label text but have no notion of a step, a delegation or a tool call, leaving it unclear what the smallest unit of execution to label should be. Annotrace reads six trace formats, including OpenTelemetry GenAI, LangSmith and Langfuse, and turns each run into an ordered, nested list of steps. The annotator writes their own labelling codebook; a frontier model proposes a starting set from a sample of their own traces, answers first, and is asked exactly the questions the human is asked, making annotation a joint human-AI process. Annotrace measures agreement between any two annotators, human or AI. On a multi-agent question-answering corpus it reproduces a previous manual effort's 235 annotation units exactly, and two models labelling the same traces agree 91% of the time at $\kappa = 0.82$. 

---
# Reasoning Models Can Control Their Chains of Thought When They Tell Themselves To 

**Authors**: Uriel Dolev, Idan Schwartz, Yoav Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2610.23257)  

**Abstract**: Chain-of-thought (CoT) controllability, the ability of reasoning models to control properties of their CoT, is consequential for both their usefulness and reliable oversight. Recent work evaluates this capability by measuring whether models follow CoT constraints given in the user prompt, and reports very low controllability across models. We argue that this low controllability may not reflect an inherent inability of reasoning models to control their CoT, but instead result from naively specifying constraints in the user prompt. To test this, we introduce Self-Instructed CoT Controllability: a complementary evaluation that places the constraint as a self-instruction inside the model's CoT and measures its ability to follow it. We show that, under self-instruction, models achieve significantly higher controllability than with standard user instructions. Even a generic reminder to follow the user's constraint substantially improves controllability. Moreover, effective self-instructions can be bootstrapped into the reasoning through a simple instruction in the user prompt, enabling improved control even without access to the reasoning channel. Finally, using self-instruction, we demonstrate in a realistic monitoring setup that CoT controllability can be leveraged for monitor evasion, posing a risk to CoT monitorability. 

---
# Sample, Mix, or Verify? Allocating Test-Time Compute under Imperfect Verifiers 

**Authors**: Srinivasan Iyengar  

**Link**: [PDF](https://arxiv.org/pdf/2610.23255)  

**Abstract**: A common way to make language models and agents more accurate is to spend more compute at inference time: generate many candidate solutions and submit the one a verifier prefers. Success then requires that some candidate is correct and that the verifier picks it. The first grows predictably with the number of samples; the second does not, because real verifiers (unit tests, majority votes, LLM judges, reward models) accept wrong answers, reject right ones, and make correlated mistakes. With a fixed budget, a practitioner must decide how many samples to draw, which models to draw them from, and how much to spend on verification, and today these choices are made by rules of thumb. We introduce SPLIT, which makes them from a small labelled pilot. It rests on an exact formula for the probability of submitting a correct answer under accept/reject and top-score selection that accounts for varying task difficulty, blind spots shared across models, and correlated repeated verifier calls. SPLIT fits the formula by empirical Bayes and uses it to plan budgets on unseen tasks. Across seven benchmarks spanning math, science QA, code and software-engineering agents, 26 models, 175 agent systems and six kinds of real verifiers, all from public releases, SPLIT lands within 0.02-0.8 percentage points of the best allocation in hindsight, while every fixed rule we test loses at least 2.5 points in some setting. Regret depends only on how well SPLIT ranks configurations, and it stays near-optimal under compute, price and latency costs and online. The formula also explains when asking more LLM judges lowers precision. As a forecaster far beyond the pilot, SPLIT beats power-law fitting when the verifier caps success, and we report where it loses. 

---
# Tree-Aware Posterior Distillation for Long-Horizon Agentic Reinforcement Learning 

**Authors**: Heming Zou, Qi Wang, Yixiu Mao, Yun Qu, Yuyi Guo, Zeyuan Liu, Chenglin Li, Weijie Liu, Saiyong Yang, Xiangyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2610.23181)  

**Abstract**: Reinforcement learning with verifiable rewards has become a scalable post-training paradigm for large language models. However, training agents over long-horizon interactions presents two challenges: using only terminal rewards makes turn-level credit assignment sparse and high-variance, while sequential mini-batch updates on a fixed rollout batch introduce policy staleness as the current policy drifts from the behavior policy. Existing approaches obtain denser credit via learned critics, process reward models, or on-policy distillation, but they rely on a value network, step-level annotations, or a stronger external teacher that increase training complexity. Meanwhile, importance sampling and clipping mitigate policy drift, but they do not specify a finite target that stops sequential updates from overshooting. In this work, we introduce TreeOPSD, a critic-free, tree-aware posterior self-distillation framework for long-horizon agentic reinforcement learning. Treating each interaction turn as a tree node, TreeOPSD branches alternative continuations from shared prefixes and recursively backs up terminal rewards into implicit turn-level credit. It then maps each credit to a closed-form, bounded teacher distribution parameterized by the likelihood ratio relative to the frozen behavior policy. Minimizing the divergence between this teacher and the student policy recovers the policy-gradient direction as the tangent at the behavior policy while bounding updates toward a finite target across sequential mini-batches. Empirically, TreeOPSD improves agentic capabilities across several competitive benchmarks and model scales, especially for long-horizon tasks, e.g., raising the average accuracy of Qwen3-30B-A3B-Thinking-2507 by 4.97 points over GRPO on the Deep Research setting. 

---
# Can AI interview? Comparing AI-led to Human-led Chat Interviews 

**Authors**: Semra Guven, Tobias Gårdhus, Andreas Bjerre-Nielsen, Anna Rogers, Hjalmar Bang Carlsen  

**Link**: [PDF](https://arxiv.org/pdf/2610.23149)  

**Abstract**: Large language model chatbots have been proposed as AI-interviewers for collecting qualitative interview data at scale, potentially bridging social science's longstanding divide between the breadth of surveys and the depth of qualitative interviews. This paper provides the first direct empirical comparison of data quality in AI-led and human-led interviews. In a randomized experiment (N = 41), university students were interviewed via synchronous text chat by either trained non-expert interviewers or a locally hosted AI system; we evaluate exposure, specificity, and relevance. Median interview duration was similar across conditions, but AI-interviewers asked more than twice as many questions per interview and elicited roughly 65 percent more respondent text per interview. Per-question responses were longer with human interviewers, though not after multiple-testing correction; only AI-condition participants skipped questions. Responses were almost uniformly on-topic in both conditions; for specificity we detect no difference, but the design cannot exclude one of up to three quarters of a point. These results were obtained with a small, locally hosted open-weight model (7B parameters), which addresses concerns about data security, GDPR compliance, and scientific reproducibility. Our findings support the viability of AI-interviewing for large-N qualitative data collection. 

---
# LED: Improving Masked Diffusion Language Models by Lookahead Editing 

**Authors**: Yuyan Zhou, Zeyuan Yang, Yuxuan Song, Zhicheng Cai, Hongli Yu, Weiying Ma, Hao Zhou, James Kwok  

**Link**: [PDF](https://arxiv.org/pdf/2610.23122)  

**Abstract**: Diffusion large language models (dLLMs) have emerged as a promising alternative to autoregressive language models, enabling non-autoregressive text generation through parallel decoding. Existing dLLMs can be broadly categorized into two families: masked diffusion models (MDMs) and uniform diffusion models (UDMs) In this work, we theoretically show that MDMs are easier to optimize due to their mask-based transition schedule but have lower computational expressivity, whereas UDMs enable iterative state revision and greater expressivity but are substantially harder to train. This trade-off raises a natural question: can we obtain the expressivity of iterative revision without giving up the simple, easy-to-optimize transition schedule of MDMs? Motivated by this analysis, we propose Lookahead Editing (\ours{}), which augments masked diffusion decoding with a rewritable proposal state. At each decoding step, the model performs several lookahead updates to iteratively refine future predictions in this proposal state before committing tokens to the masked sequence. This design introduces a rewritable proposal that uses additional compute to improve expressivity while keeping the simple mask-based transition schedule of MDMs. We fine-tune Qwen2.5-7B and Qwen3-8B to obtain LED-\texttt{7B} and LED-\texttt{8B}, respectively. Across eight benchmarks, LED SFT yields average gains of $+11.4$ and $+7.7$ points over standard MDM SFT. Moreover, using SFT alone, LED-\texttt{8B} outperforms LLaDA2.1-16B Q-mode, which is trained with continued pre-training, supervised fine-tuning, and reinforcement learning. 

---
# ECHO: Replacing the Compliance Proxy with Response Harmfulness in Optimization-Based Jailbreak Attack 

**Authors**: Yuanwei Wu, Lingdong Huang, Yunlong Mao, Kanye Ye Wang, Jingyu Hua, Sheng Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2610.23030)  

**Abstract**: Despite extensive safety alignment, large language models remain vulnerable to jailbreak attacks. Existing optimization-based attacks share a single objective: maximize a token-level proxy of victim compliance---such as the likelihood of an affirmative prefix---computed on the prompt rather than the decoded reply. This proxy cannot distinguish a cautious hedge from a fully actionable harmful answer, leaving the search blind to what the model actually says. We propose ECHO (Evaluation-Conditioned Harmfulness Optimization), which replaces the compliance proxy with response-level, continuous harmfulness scores from HarmSig, a 1.7B regressor grounded in actual harmful capability. ECHO couples HarmSig with five word-level perturbation operators and Keyword-Anchored Beam Search with global top-m memory, under a strict black-box threat model. Across five open-weight victims on AdvBench, ECHO achieves up to 0.898 StrongREJECT mean score and 89.23% HarmBench success. Our code is publicly available at this https URL. 

---
# SAFE-Mem: Secure Agentic Memory with Fine-Grained Authority and Governance-Aware Retrieval 

**Authors**: Haoming Li, Dingyi Kang, Xu Hu, Jinxin Yang, Zhu Xu, Chenru Wang, Yi Ge, Miao Yin, Guanpeng Li, Feng Chen, Bingzhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.22959)  

**Abstract**: Large language model (LLM) agents increasingly rely on persistent memory to support long-term reasoning and personalized interactions. However, information collected from external sources can introduce security risks when untrusted claims (i.e., atomic statements) are later treated as authorized preferences, policies, or permissions. Existing defenses typically rely on content-based filtering or coarse-grained provenance tracking, which either fail to prevent unauthorized memory use or block legitimate information, sacrificing memory utility. To address these limitations, we propose SAFE-Mem, a fine-grained memory governance framework that improves security while preserving legitimate memory use. First, SAFE-Mem annotates each claim with three governance labels (i.e., kind, authority, and scope) to capture its provenance and permitted use. Second, governance-aware retrieval matches query requirements with claim permissions to block unauthorized influence while retaining legitimate information. Third, a governance-enabled memory graph connects source sentences, claims, and entities to support provenance tracking and efficient retrieval. Across four LLM backbones on MPBench and LoCoMo, SAFE-Mem attains the lowest attack success among defenses that retain legitimate memory, at least 77% below the strongest of them, and the highest answer accuracy of all evaluated defenses, improving it by up to 15%. The code is available at this https URL. 

---
# Sep-Latent: Separator Tokens as Natural Compression Anchors for Implicit Reasoning 

**Authors**: Yinlong Xu, Jianfeng Huang, Zhengdong Zhou, Mingjie Lin, Fuyu Lv, Dan Ou, Hongxia Xu, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.22924)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable success in complex reasoning through explicit Chain-of-Thought (CoT), but the associated computational overhead and token verbosity remain significant barriers to efficiency. While implicit reasoning in continuous latent spaces offers a promising alternative, existing methods like Coconut and CODI often rely on fixed-length latent sequences and suffer from representational collapse. In this paper, we propose Sep-Latent, an adaptive implicit reasoning framework that utilizes separator tokens as structural semantic anchors. By introducing an Attention-Decay mechanism, we force the model to aggregate and compress reasoning logic into these anchors during training. We then align explicit and implicit pathways through multi-level distillation across all layers to ensure logical consistency. Crucially, Sep-Latent enables dynamic reasoning depth by adaptively terminating the latent thinking phase based on query complexity. Experimental results on the GSM8k-Aug benchmark show that Sep-Latent achieves an accuracy of 56.8\% with an average of only 5.4 tokens, significantly outperforming prior fixed-length implicit models in efficiency while narrowing the performance gap with explicit CoT. Further analysis via PCA and symbolic variant testing confirms that our method ensures semantic grounding and exhibits superior robustness to perturbations. 

---
# Batched Contextual Reinforcement II: Does RLVR Need Long Rollout Windows? 

**Authors**: Bangji Yang, Hongbo Ma, Ruihan Guo, Jiajun Fan, Xi Zhu, Ge Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.22887)  

**Abstract**: Long rollout windows make reinforcement learning with verifiable rewards (RLVR) expensive, yet simply shortening them can remove the correctness signal: when every single-problem rollout truncates before an answer, group-relative accuracy advantages vanish. As a technical continuation of Batched Contextual Reinforcement (BCR), we study how concurrent rewards relax this dependence on long windows. Concurrent Constraint Reinforcement (CCR) places three independently verifiable problems in one capped training completion. Correct answers completed before truncation retain credit, while a shared budget encourages shorter useful reasoning without an explicit length penalty. We connect this mechanism to conditional reward- and update-variance analyses and study 4B and 8B mathematical reasoning models. Against single-problem GRPO with a 32k window, CCR with an 8,000-token window reduces tokens per training step from 975,464 to 174,957 on 4B and from 793,832 to 165,788 on 8B, saving 82.1% and 79.1%, respectively. An 8B training trace shows increasing accuracy reward alongside shorter completions. Single-problem evaluations provide supporting evidence of retained reasoning quality. These results identify reward-preserving training structure as a practical route to lower per-step cost in short-window RLVR. 

---
# Cooperation Without Adaptation: LLM Agents in a Collective-Risk Dilemma 

**Authors**: Trung-Kiet Huynh, Dao-Sy Duy-Minh, Chi-Nguyen Tran, Phu-Hoa Pham, Phu-Quy Nguyen-Lam, Le Hong Trang, Anh Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.22882)  

**Abstract**: Language-model agents are increasingly deployed to act on behalf of people in settings where individual decisions shape shared outcomes. We examine whether five production models adapt their contributions to changing risks and partner behaviour, or instead follow a relatively fixed strategy. In a collective-risk dilemma game (CRG), six players contribute to a shared pool over ten rounds; if the pool falls short of its target, a catastrophe may destroy their remaining resources. Four models contributed approximately an equal share at every positive risk level, and four of the five also contributed when risk was zero, even though contributing could not improve their payoff. Explicitly stating that only the agent's own payoff mattered reduced zero-risk contributions for most models, while a more neutral prompt did not. Several models correctly identified which action offered the higher expected payoff at different risk levels, yet their contributions changed little. Against fixed partners, no model consistently chose the payoff-maximising response, and in mixed-model groups, a single agent's shortfall in the final round could cause an otherwise successful group to fail. Payoff-based selection also depended on population composition: in a single six-agent group, it favoured the model that contributed least and reduced group success, whereas in larger, randomly mixed populations, it favoured an exact fair-share model at moderate and high risk. These findings show that cooperative self-play alone is not evidence of strategic adaptability. Consistent with a machine-behaviour approach, evaluations of agents in multi-agent systems should test how they respond to incentives, partner actions, and population composition. 

---
# Too Emotional to Be Fake? Intra-Utterance Emotion Shifts Expose Failure Modes in Speech Deepfake Detectors 

**Authors**: Jingyi Chen, Surya Koppisetti  

**Link**: [PDF](https://arxiv.org/pdf/2610.22832)  

**Abstract**: Expressive speech synthesis can now change emotion within a single utterance, yet its impact on deepfake detection remains unclear. We introduce EmoShift, a benchmark comparing emotion-stable (AA) and emotion-changing (AB) speech to examine these effects while accounting for static emotion identity and acoustic confounds. It comprises 7,240 bona fide utterances from four source datasets and 9,720 synthetic utterances generated using four expressive TTS systems. Across seven detectors, we find that emotion changes can make synthetic speech appear more genuine, increasing spoof miss rates by up to 25.3 percentage points on OMG-Emotion at fixed thresholds. Conversely, emotion-changing bona fide speech increases false-alarm rates by up to 24.4 percentage points on IEMOCAP. Further analyses reveal detector-specific sensitivities to emotion trajectories, not fully explained by static emotion identity or splicing artifacts. These findings underscore the importance of emotion dynamics in deepfake detection. 

---
# Veil-Then-Reveal: Protecting PII Privacy via Reversible Pseudo-Anonymization in Zero-Trust User--LLM Interaction 

**Authors**: Mohamed Shaaban, Mohamed Elmahallawy, Thomas E. Carroll  

**Link**: [PDF](https://arxiv.org/pdf/2610.22800)  

**Abstract**: Large language models (LLMs) increasingly process sensitive documents and personal information, yet sending prompts containing personally identifiable information (PII) to remote providers exposes data subjects and organizations to risks from logging, retention, model training, and unintended disclosure. Existing anonymization methods often mask or remove PII, while conventional encryption eliminates the lexical and structural cues needed for effective LLM processing. We present Veil-then-Reveal (VtR), a user-side PII pseudo-anonymization framework for privacy-preserving LLM interaction under a zero-trust threat model. VtR treats remote LLM providers as untrusted and sends only pseudo-anonymized prompts, keeping plaintext PII on the user. VtR first detects PII and replaces it with realistic, same-type, in-vocabulary surrogates before prompts leave the user's device, preserving their linguistic and structural properties. It combines format-preserving encryption (FPE) for structured PII, such as phone and credit-card numbers, with a keyed-hash dictionary and secure vault for semantic PII, such as names, locations, and organizations. Session-specific mappings provide cross-session unlinkability, while user-side restoration reversibly replaces surrogates with their original values in LLM responses. We further introduce a rank-class FPE strategy for dates that preserves format and ambiguity while reducing cipher evaluations. Evaluation on three datasets containing 11,322 annotated PII instances achieves 97.9% detection recall, 95.2% protection of detected values, and 99.9% round trip; 100% of echoed surrogates end-to-end round-trip restoration. 

---
# Reasoning Reranking with Semantic IDs as the Sole Item Representation 

**Authors**: Zhenwei Tang, Jintao Liu, Di Bai, Xidong Wu, Xinwu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.22797)  

**Abstract**: Reasoning rerankers write a rationale before ranking a candidate slate, and existing systems read item metadata to do so, even though generative recommenders increasingly represent items as compact Semantic IDs (SIDs). Ranking from SIDs alone has two appeals: SID prompts are far shorter than the matched item text, and although item codes are fixed from content, the backbone learns their token embeddings in part from user-item interaction sequences, so the tokens can carry behavioral signal that item text lacks. That training targets next-item generation, and whether it supports ranking a supplied slate from SIDs alone is untested. We build ReRankID, a controlled testbed in which models reason and rank from a history and slate shown only as SIDs, varying item rendering, history length, and slate size one at a time. On this testbed, such models often copy the input order instead of reranking, and readable metadata does not repair the failure. We compare supervised fine-tuning, rejection-sampled self-training, on-policy distillation, and verifier-based reinforcement learning as post-training methods and find that they elicit the missing behavior. The strongest post-trained model, the 8B model trained with a graded ranking reward, gives the best SID-only ranking and matches strong text-reading LLMs whose prompts are about 20x longer. It ranks above random with no history at all, where a caption-reading reference does not, and it keeps its gains at long histories, where text prompts pass 50k tokens and the caption-reading reference stops improving. In a paired rationale audit, this model argues trade-offs better than that reference but describes items less accurately. The common post-training methods we test thus elicit ranking from SIDs but not faithful item descriptions, which calls for post-training that improves the reasoning itself, not only the ranking it produces. 

---
# Reading the News Without Hindsight: NoLBERT Topics and the Cross-Section of Stock Returns 

**Authors**: Kevin Foley, Jonathan Hartadi, Shivesh Prakash, Charishma Takkallapalli, Swapnil Vatsal  

**Link**: [PDF](https://arxiv.org/pdf/2610.22745)  

**Abstract**: Language models that read financial news are usually trained on text from the same years used to test them, so a backtest can reward hindsight. We build news topics from NoLBERT, an encoder pretrained only on text published between 1976 and 1995, adapted into a sentence encoder by contrastive fine-tuning on passages of the same article, with no labels or returns. We compare it with LDA, MiniLM and FinBERT readers on 1.26 million articles from 2012 to 2024, convert each reader's topics into attention shocks and stock exposures, and forecast the returns of the 1,000 largest US stocks with eight models at three rebalancing frequencies. Models are fitted on 2013-2019, choices are made on 2020-2021, and 2022-2024 is held out. The experiment with the best validation score loses money at every frequency. Measuring attention shocks against the past 63 trading days instead of 5 raises the test rank IC of every reader. NoLBERT gives the most stable topics and forecasts about as well as LDA. An equal-weight combination of all readers and models earns a monthly net Sharpe ratio of 1.12, significant against zero before adjusting for the search but not against a price-only benchmark. 

---
# A misinformation playbook sharpens judgement without breeding cynicism 

**Authors**: Tong Guo, Teng Ye, Yutong Xie, Qiaozhu Mei  

**Link**: [PDF](https://arxiv.org/pdf/2610.22724)  

**Abstract**: Misinformation harms public health and democracy, and generative artificial intelligence now writes falsehoods as persuasive as people's. Defences range from fact-checking individual claims to training people to resist falsehoods. Yet many claims are too new to check, and interventions that teach doubt can reduce belief in truth too. We propose a defence that targets not the facts of each claim but the persuasion strategies that recur across them. Here we show that learning misinformation's persuasion playbook, mapped from 214,646 fact-checked claims in 51 languages, can sharpen human and machine judgement of claims they know little about, without making either more cynical about true claims. The playbook spans social influence, information manipulation and emotional appeals, combined around a core of fear, conspiracy and fabrication. Briefed with it, plus a reminder that persuasion is not proof of falsehood, a language model better distinguished true from machine-generated false claims about events after its training data. In a pre-registered experiment, the same briefing improved people's discernment of claims they reported as unfamiliar and left their belief in true claims unchanged. Averaged over all claims, the gain was not statistically significant. Because it targets what recurs, this defence need not wait for claim-specific fact-checks. 

---
# Risk and Refusal: Mechanisms of LLM Responses to Suicidal Ideation 

**Authors**: Drishti Goel, Soorya Ram Shimgekar, Jeongah Lee, Violeta J. Rodriguez, Daniel S. Brown, Ravi Karkar, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2610.22715)  

**Abstract**: Refusal is a common safety mechanism in large language models (LLMs), yet in contexts of suicidal ideation (SI), refusing to engage can itself constitute a safety failure. We evaluate how eight instruction-tuned LLMs respond to SI and self-harm disclosures across three datasets, varying whether models are informed that a disclosure expresses SI and whether they receive a clinician-developed response protocol, and complement this with mechanistic analyses in six open-weight models. Our results reveal a nuanced relationship between refusal and response-level risk. At baseline, elevated risk in model responses is driven primarily by inattention to the disclosure, alongside refusal to engage. Mechanistic analyses suggest that models may refuse SI disclosures through the same direction that mediates their refusals of harmful requests. However, informing models of the SI context, or suppressing refusal directly, increases engagement without reliably lowering response risk, and may raise it. In contrast, the clinician-guided protocol consistently reduces response-level risk while also reducing refusal. Together, these analyses and ablations motivate the need for models that refuse only when warranted and engage in interactionally safe ways when users disclose SI. 

---
# GlobeBench: Evaluating Text-Based World Model Capabilities for Scalable Environment Simulations 

**Authors**: Nicholas Saban, Charvi Bannur, Mariya Vasileva, Darshan Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2610.22708)  

**Abstract**: Text-based world models have recently shown potential to simulate complex RL environment state transitions for a diverse set of use cases. However, no standardized benchmark exists to compare, study and improve these models. Through this paper, we attempt to solve this problem by building GlobeBench, a novel benchmark with 1,632 tasks across five separate domains and nineteen simulation skills. The tasks check whether models can predict the environment's response to a specific action given the full agent trajectory by understanding earlier state changes, following environment rules, and predicting the effects of actions on both related and unrelated objects. We compare the predicted environment response with an execution-derived reference, using a hybrid architecture with both deterministic gates and a semantic judge. Among 20 general-purpose LLMs and specialized text world models, GPT-6 Astra achieves the highest task success rate at 49.57%. We find that model accuracy decreases as actions affect a larger portion of the environment. Specifically, when the agent's changes must propagate to dependent objects, task success drops to an average of 5.1% across all 20 models. Although model strengths differ across tasks, 631 tasks remain unsolved by any model, suggesting a significant gap in general LLM capabilities as a world model. 

---
# Not All Graphs Are Maps: A Task-Agnostic Exploration Framework for Mobile GUI Navigation 

**Authors**: Adamo Orsini, Seyed Mehdi Ayyoubzadeh, Sadman Rakib Pinon, Hannuo Zhang, Koosha Gholipour Baradari, Yuanhao Yu, Xinxin Zuo, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.22672)  

**Abstract**: Modern mobile GUI agents can perceive visible screens and execute local actions, but they still lack persistent app-specific navigation knowledge: where functions live, which pages contain them, and which screens are equivalent across repeated visits. We introduce TAG-Nav, a Task-Agnostic Graph Navigation framework that wraps existing GUI-agent backbones with reusable destination memory. Offline exploration is distilled into compact records of app functions, target pages, and waypoint routes before any evaluated instruction is seen. When a future task's destination is covered, TAG-Nav retrieves the destination and provides route guidance without task-specific demonstrations. This separates mobile GUI destination navigation from full task completion: many tasks share the prefix of reaching the right page, while final actions may enter content, send data, make purchases, delete information, or change sensitive settings. To evaluate route finding without downstream execution confounds, we introduce DestBench, a navigation-only protocol that asks whether an agent reaches the correct pre-action destination screen. DestBench enables a controlled graph-memory study comparing no-memory agents with random, automated BFS, and clean graph memory. Graph-derived memory improves strict navigation success rate of UI-TARS-1.5-7B from 52.4% without graph memory to 58.3% with automated BFS memory, and improves Qwen-VL-Max success rate from 59.5% to 71.4% on the same conditions. Together, TAG-Nav and DestBench show that destination coverage and route cleanliness strongly shape graph-memory value, making graph quality a measured variable rather than an assumed benefit of exploration. 

---
# Exploring The Empirical Privacy Landscape of Synthetic Data 

**Authors**: Zihao Zhao, Avi Schwarzschild, Casey Meehan, Katherine Lee, Zhili Feng, Dylan Sam  

**Link**: [PDF](https://arxiv.org/pdf/2610.22659)  

**Abstract**: Methods for training language models on sensitive data differ in how they balance capability gains against the risk of leaking private information. In this paper, we present a controlled comparison of data-level and training-level interventions for navigating the capability-privacy tradeoff on synthetic medical records with canaries to probe for leakage. Direct finetuning performs best on capability metrics but leads to the most privacy leakage, while DP-SGD reduces leakage at substantial cost to capabilities. Generating synthetic data from these records offers a more favorable balance, particularly when guided by a uniqueness-aware instruction to replace private details. Relative to naive paraphrasing, privacy-aware rewriting improves privacy protection across a variety of different rewriting models. We also find that more capable rewriting models generally reduce the risks of downstream privacy leakage, although held-out fit does not improve monotonically and increased test-time compute yields inconsistent gains. Overall, these results highlight privacy-aware rewriting as a promising direction for achieving a more favorable trade-off between capabilities and empirical privacy compared to other training-level methods, and our evaluation framework supports comparisons of future models and methods. 

---
# Can LLMs predict the outcomes of machine learning experiments? 

**Authors**: Ayush Bansal, Aldo Pareja, Sean Klein, Serge Didenko, Artem Lukoianov  

**Link**: [PDF](https://arxiv.org/pdf/2610.22599)  

**Abstract**: Recent advances in AI make Recursive Self-Improvement (RSI) increasingly plausible. Large Language Models (LLMs) can already propose, select, and run Machine Learning (ML) experiments. However, the space of possible design choices in ML is vast. An LLM that proposes an experiment needs a reliable intuition of whether it is worth running given the timing constraints of long training runs. In this work, we measure such intuition in current frontier LLMs by evaluating how accurately they predict the impact of self-generated ML experiment ideas before these are put into training runs. To this end, we construct a dataset of 288 LLM-proposed ideas on eight MLE-bench tasks. We then prompt five frontier LLMs (GPT-6 Astra, Claude Opus 5.5, Claude Fable 5.1, Kimi K3, and GLM-5.3) to predict the impact of each idea and compare their predictions with the actual outcomes after training. We report four main findings. First, LLM predictions are substantially better than random chance but far from accurate: even the best model, Claude Opus 5.5, explains only half of the variance in outcomes (R^2 = 0.49). Second, we notice that all tested models systematically underestimate the magnitude of the changes. Third, prediction quality varies widely across tasks. Finally, the choice of the coding harness that actually implements the ideas has little effect on outcomes, and we find no evidence of self-preference. Higher reasoning settings and additional context lead to modest gains. Overall, our results show that current LLMs have useful but limited intuition about which ML ideas will work. We release our dataset to support further study. 

---
# Repairing LLM-Generated Specifications with Deterministic Diagnosis 

**Authors**: Giovanni Vaccarino, Emanuele La Malfa, Dehai Min, Hanrong Zhang, Pier Luca Lanzi, Lu Cheng, Michael J. Wooldridge, Philip S. Yu, Sara Bernardini  

**Link**: [PDF](https://arxiv.org/pdf/2610.22568)  

**Abstract**: Large language models (LLMs) are increasingly used as formalizers: given a reasoning problem expressed in natural language, they produce a formal specification that a symbolic solver can then solve, with guarantees relative to that specification. Many specifications are runnable yet semantically wrong, encoding incorrect constraints, domains, objectives, predicates, or goals, and solver-error feedback dries up once execution succeeds. Purely LLM-based diagnosis, lacking a formal ground truth, often fails to spot these errors, and its ungrounded critiques cause regressions in 11.8% of initially correct specifications in our experiments. We propose DRAGO, a repair framework that replaces this ungrounded critique with deterministic, formalism-specific diagnosis, and retains a rewrite only when the diagnosed error weight strictly decreases under a reference-free acceptance predicate. Across five benchmarks spanning logical reasoning (SMT-LIB2), constraint programming (MiniZinc), and planning (PDDL), DRAGO cuts regressions by nearly an order of magnitude, to 1.3%, while improving answer accuracy by 12.9 percentage points over zero-shot formalization and 8.7 percentage points over LLM-diagnostic repair, averaged across benchmarks. We also uncover two rendering defects in Planetarium that cause 50% (3,086 instances) of its floor-tile split to be inconsistent with the reference formalizations, and release a corrected split. 

---
# Are Large Reasoning Models Good Everyday Reasoners? A Comparison with Non-Reasoning Models on Logical Reasoning Tasks 

**Authors**: Kentaro Ozeki, Risako Ando, Takanobu Morishita, Hirohiko Abe, Koji Mineshima, Mitsuhiro Okada  

**Link**: [PDF](https://arxiv.org/pdf/2610.22538)  

**Abstract**: Large reasoning models (LRMs) that leverage test-time computation achieve strong performance on mathematical and scientific benchmarks and are increasingly deployed in real-world applications. However, their ability to perform \textit{everyday logical reasoning}, which often departs from textbook formal logic, remains underexplored. In this work, we systematically evaluate how well reasoning models align with everyday reasoning patterns by introducing a benchmark based on linguistic and pragmatic reasoning phenomena. We also compare reasoning models with non-reasoning models under controlled prompting conditions. We find that, in many cases, (1) reasoning models show degraded performance in everyday reasoning when such reasoning deviates from formal logical rules, (2) some non-reasoning models with chain-of-thought prompting outperform comparable reasoning models in everyday reasoning, and (3) prompting with additional instructions provides limited control over reasoning behavior. These findings suggest that current reasoning models do not reliably capture everyday reasoning and point to limitations in post-training, motivating improved training objectives and evaluation across diverse forms of reasoning. 

---
# Progressive Misalignment Can Emerge from Benign Recursive Self-Distillation 

**Authors**: Karim Abdel Sadek, Federico Di Gennaro, Fanny Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.22467)  

**Abstract**: Training models on their own outputs has attracted widespread attention as a possible route to advance AI capabilities. One prominent approach is self-distillation, where a model is trained to match the token-level predictions it would have produced when given access to additional context, such as a cue, feedback, or instruction. However, the safety properties of this training procedure are poorly understood. In this paper, we present a surprising safety failure of recursive self-distillation, where each self-distilled model becomes the teacher for the next round of training. We observe that when a safe model is recursively self-distilled on its own (safe) generations while receiving an additional non-harmful cue, it becomes increasingly misaligned across a diverse set of harmful queries; we call this Progressive Misalignment. We show this failure mode in both the Qwen3 and Phi-4 model families on SORRY-Bench and StrongREJECT. Progressive misalignment also occurs in a more realistic setup, where the fixed cue is replaced by feedback from a simulated user with benign intent. Using a gradient-based method, we are also able to find benign prompts that lead to progressive misalignment when applying recursive self-distillation, showing that harmful training prompts are not necessary for this failure mode to emerge. We hope our work encourages future investigations of the safety properties of AI systems recursively trained on their own data. 

---
# High prevalence of antagonizing rhetoric in trending topics on Twitter/X 

**Authors**: Armin Pournaki, Eckehard Olbrich  

**Link**: [PDF](https://arxiv.org/pdf/2610.22455)  

**Abstract**: Disagreement between political opponents is a central element of democracy. This disagreement can be voiced in different ways. An opponent can be antagonized, i.e. portrayed as an enemy that needs to be overcome, or they can be criticized, i.e. treated as a legitimate adversary with whom one can engage. While the latter is a fundamental part of democratic exchange, the former can be detrimental to public discourse. Political disagreement increasingly takes place on social media, where negativity towards political out-groups has been found to drive engagement. However, existing research has largely treated such negativity as a single dimension, leaving open whether it is antagonization or criticism that circulates on these platforms. This work addresses this gap by introducing a computational approach to detect these two types of rhetoric in text and identify its targets. On a corpus of over 17 million tweets from trending topics in the German Twittersphere, the method reveals that about a third of all tweets contain antagonizing rhetoric. Comparing tweets from trending topics to a random sample shows that the discourse in trending topics is systematically more antagonizing than ordinary Twitter communication. Combining antagonization detection with a large-scale estimation of users' political leaning, we find that right-leaning users employ antagonizing rhetoric significantly more than their left-leaning counterparts. Furthermore, both forms of oppositional rhetoric are strongly amplified within political camps, with multipliers retweeting them up to twice as often as tweets without oppositional rhetoric. Finally, patterns of antagonized targets reveal how different political camps construct their out-groups, with the right-wing populist party AfD and the Green party as the central antagonists of the opposing camps. 

---
# PRIME: Towards Verifiable Evidence-Grounded Reasoning in LLM Agents 

**Authors**: Hamed Khosravi, Xiaoming Huo  

**Link**: [PDF](https://arxiv.org/pdf/2610.22426)  

**Abstract**: LLM agents are used to check records against an organization's requirements, for example whether a vulnerability report is ready for investigation. Requirements often allow several ways to be met, such as a severity rating with a curator-reviewed advisory or a published advisory link. Agents fail by accepting a report after a required check has failed, or by settling for a weaker conclusion without trying another way that could still succeed. We introduce PRIME (Precommitted Rules for Inference from Minimal Evidence). Before checking, it fixes evidence routes specifying how each conclusion may be justified. A separate program records each check result and refuses any conclusion the results do not justify. It also points the agent to checks that could justify a stronger conclusion, or runs them itself. PRIME reports only conclusions authorized by its rules and ledger. With correct requirements, exact checks at known locations, and sufficient budget at stopping, completion reaches the strongest supported conclusion. On 480 public-record tasks, an agent shown the requirements reported unjustified conclusions on 26.7% of tasks, while PRIME reported none. Its directions and added checks raised the average score for justified conclusions from 30.1, with refusal alone, to the maximum of 59.2 out of 100. 

---
# Eliciting Specifications: When Does Interactive Clarification Help? 

**Authors**: Mowafak Allaham, Aditya Vempaty, Ashish Jagmohan, Prasenjit Dey, Ravi Kokku  

**Link**: [PDF](https://arxiv.org/pdf/2610.22382)  

**Abstract**: LLMs can generate Lean specifications from task descriptions, but those descriptions are often underspecified or ambiguous and may omit verifier-relevant edge cases, yielding imperfect formal specifications. We study whether an elicitation agent (EA) can improve verifier-checked Lean specifications by seeking clarification from a reference-grounded simulated developer before formalization. The agent decomposes the task, identifies unresolved decisions, asks bounded clarification questions, and uses the answers to write a refined natural-language specification that an LLM formalizer then translates into Lean. On 189 Verina Lean 4 specification-generation tasks, we compare single-shot generation with elicitation under question budgets of k=3 and k=10, allowing at most three or ten EA questions per task, across same-model and cross-model configurations. The same-model Claude Sonnet 4.6 pipeline has negative strict pass@1 estimates at both budgets, showing elicitation is not a uniform accuracy booster. In contrast, the Sonnet 4.6-to-GPT-5.6 and Sonnet 4.6-to-Gemini 3.8 Flash pipelines have positive strict estimates. The largest strict result reaches 65.6%, a +6.3-point change (+12 tasks), in the Sonnet 4.6-to-GPT-5.6 pipeline. Six of eight conditions have positive compile-conditional estimates, with a maximum cross-model difference of +4.9 points among compiling outputs. Treatment estimates also differ across the basic and advanced splits; the largest advanced-task change occurs in the Sonnet 4.6-to-GPT-5.6 configuration at k=10, from 42.0% to 50.6%. Failure analysis identifies postconditions as the largest source of degradation. These findings suggest future autoformalization systems should treat elicited natural-language specifications as intermediate artifacts and optimize jointly for semantic correctness among compiling outputs and reliable translation of clarified information into Lean. 

---
# Binding-Specific Relearning Savings for Test-Time Continual Learning 

**Authors**: Zhe Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.22381)  

**Abstract**: When a fact becomes harder to elicit from a language model, has the model lost it, or has it merely become less accessible? Parameter-updating test-time continual learning agents are usually evaluated by current output accuracy; we ask whether that measurement captures how prior experience shapes later updates. We test whether learning a specific factual binding produces a later relearning advantage beyond familiarity with its entity, answer vocabulary, format, or query slot. In Qwen2.5-1.5B, we compare three matched, counterbalanced histories: LEARNED receives the correct binding, CONTROL receives matched exposure without it, and WRONG receives a competing relation-valid answer in the same slot. After an identical rehearsal-free continual fine-tuning stream, LEARNED reacquires the target faster than CONTROL at a prespecified checkpoint of partial forgetting. WRONG shows no positive advantage over CONTROL, while its comparison with LEARNED provides slot-matched corroboration, subject to possible competitor interference. The advantage remains positive after adjustment for measured pre-relearning behavior. In a secondary descriptive conversion using an unpaired never-seen pool, prior learning reduces the optimizer updates required for restoration by about 18%. Current behavioral accuracy can therefore underestimate the lasting effect of previously acquired experience. Code is available at this https URL 

---
# Revealing the Impact of Visual Text Style on Attribute-based Descriptions Produced by Large Visual Language Models 

**Authors**: Xiaomeng Wang, Martha Larson, Zhengyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.27553)  

**Abstract**: When the visual style of text is considered, a wide variety can be observed in font, color, and size. However, when a word is read, its meaning is independent of the style in which it has been written or rendered. In this paper, we investigate whether, and how, the style in which a word is visualized in an image impacts the description that a Large Visual Language Model (LVLM) provides for the concept to which that word refers. Specifically, we investigate how functional text styles (readability-oriented, e.g., black sans-serif) versus decorative styles (display-oriented, e.g., colored cursive/script) affect LVLMs' descriptions of a concept in terms of the attributes of that concept. Our experiments study the situation in which the LVLM is able to correctly identify the concept referred to by a visual text, i.e., by a word or words rendered as an image, and in which the visual text style should not influence the attribute-based description that the LVLM produces. Our experimental results reveal that even when the concept is correctly identified, text style influences the model's attribute-based descriptions of the concept. Our findings demonstrate non-trivial style leakage from text style into semantic inference and motivate style-aware evaluation and mitigation for LVLM-based multimedia systems. 

---
