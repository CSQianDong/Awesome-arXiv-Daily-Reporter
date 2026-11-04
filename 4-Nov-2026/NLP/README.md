# From Autonomous Agents to Cognitive Symbionts: Toward AI-Augmented Human Intelligence 

**Authors**: Hao Fei  

**Link**: [PDF](https://arxiv.org/pdf/2611.05376)  

**Abstract**: For a decade, progress in artificial intelligence has been measured by how much a machine can do on its own. Large foundation models now write code and run multistep tasks, and personal agents act on a person's behalf. Yet people still forget what they once knew, miss what they are not looking at, fail to find the right words, and face hard choices without the benefit of their own history, because the machine that could help stands outside the situation. As AI moves into the physical world, attention has settled on one kind of intelligent individual, the embodied robot; we argue that the era of Physical AI holds a second kind of equal importance, the person augmented by AI. Alongside artificial general intelligence, which asks how intelligent a machine can be, stands AI-augmented human intelligence (AHI), which asks how much intelligence a machine can enable in a particular person. Realizing it calls for a new kind of system, a cognitive symbiont: a lifelong, embedded, real-time, private, low-friction, and proactive companion that senses what the person senses through external devices, keeps their context across years, models their situation and their mind, takes part in their thinking at the right moment, carries out everyday tasks with their context, and keeps learning the person as both change. We give the paradigm a conceptual core, the cognitive gap between what a person could bring to a situation and what they can hold in working memory, and organize its design space as a symbiosis loop of sensing, keeping, understanding, living alongside, acting for and thinking with the person, and growing, enclosed by a personal runtime and by trust. Because a symbiont changes what a person can do in front of others, it also raises risks of a new kind, from covert assistance in examinations to pressure on everyone to adopt it, and we treat their governance as part of the paradigm. 

---
# Suppressing Misaligned Behaviors in Language Models, Durably 

**Authors**: Luxi He, Pengcheng Jiang, Jifan Zhang, Jiawei Han, Danqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.05358)  

**Abstract**: Undesired model behaviors can emerge or re-emerge at any stage of training. Once observed, how effectively can we suppress them? To study this question, we first construct two model organisms--language models deliberately trained to exhibit a specific misalignment--that differ in what triggers the behavior. The narrow-trigger organism misbehaves only when the prompt contains a specific cue: it cheats on multiple-choice questions by following a planted hint that points to a wrong answer, and behaves normally otherwise. The broad-trigger organism has no such cue: after fine-tuning on insecure code, it gives harmful responses across a wide range of open-ended prompts. We then formalize post-hoc behavioral suppression and evaluate methods along three criteria: immediate suppression of the target behavior, durability under further training, and preservation of general capability. We compare standard methods--SFT, GRPO, and gradient-ascent unlearning--along these dimensions. Building on prior work on alignment shallowness, we hypothesize that more durable suppression requires the ability to recover from a wider range of misalignment states. We introduce Prefix-Expanded Adversarial Reinforcement Learning (PEARL), a GRPO variant that augments rollouts with continuations from intermediate states sampled along cached organism trajectories. Across both organism settings and two model families (Qwen3-4B, GPT-OSS-20B), PEARL achieves strong immediate suppression while preserving task accuracy, and remains durable under further fine-tuning on either the organism's data or unrelated data. It obtains the greatest benefit when the undesirable behavior is broad and cannot be addressed through targeted SFT. Together, our framework and method offer a step toward more durable post-hoc removal of undesired behaviors. 

---
# A Circuit for In-Context Learning 

**Authors**: Andrew Gao, William Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.05344)  

**Abstract**: In-context learning (ICL) lets a pretrained language model perform a task specified by demonstrations in its prompt, with no parameter updates. With arbitrary labels the model must acquire a mapping it cannot have memorized, so the learning happens within a single forward pass. Prior work found candidate parts but not how they compose. We give the composition for few-shot sentence classification with arbitrary labels in three 3-4B-parameter transformers. The model implements a key-value memory over label-token positions: each label token's residual stream encodes the label's identity, and middle-layer attention heads write into it the class of the preceding demonstration as its key. At the query, upper-layer heads match the query's class against the stored keys and copy out the matching labels. An early-layer signal encoding whether any label has repeated gates this readout; in its absence the readers attend to the start of the prompt and no label is favored. A separate set of heads suppresses the most recently seen label. Each step is established causally by activation patching, head ablations against random-head baselines, and steering with the model's own signals. A softmax model of the readout with two constants per head, fit on one prompt distribution, predicts another. The computation is conserved across the three models, the named heads alone reproduce the behaviour, and a transformer trained from scratch grows the same parts. The circuit transfers to semantic concepts and to labels with meaning, and the same parts, with a causal gate, appear from 0.6B to 8B in one family and survive instruction tuning. ICL in this setting is a gated associative lookup built from a few identifiable component types, traced causally from demonstrations to prediction. 

---
# MDSKILL: Learning to Internalize Skill Memory through Multi-View Distillation for Efficient Dynamic Agent Self-Evolution 

**Authors**: Dian Jin, Jinluan Yang, Kairong Han, Baohong Li, Zijing Hu, Yi He, Wenlin Zhong, Yunze Tong, Minqin Zhu, Linchuan Li, Yanan Niu, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2611.05313)  

**Abstract**: Skill memories encode past experiences as textual prompts to improve reinforcement learning (RL) for self-evolving agents. Complementing RL, On-Policy Self-Distillation (OPSD) provides dense, fine-grained supervision to internalize knowledge from skill-guided trajectories and accelerate policy learning. However, self-generated memories often contain noisy supervision and functionally entangled, redundant, or conflicting knowledge, making direct distillation difficult. Under limited training budgets, these issues can hinder knowledge internalization and even degrade performance. To address this problem, we propose MDSKILL, a multi-view distillation approach for skill memory internalization. MDSKILL organizes skill memories into three complementary functional views (state grounding, action planning, and execution control) and selects high-quality skills using dynamic metrics to construct view-specific and integrated reasoning branches. OPSD then transfers and consolidates complementary experiential knowledge across these branches into a shared policy, enabling efficient skill internalization and continuous agent self-evolution. We evaluate MDSKILL on three agent benchmarks against RL, skill-augmented, and self-distillation baselines that do not rely on stronger external teacher models. Without additional computational overhead, MDSKILL improves upon SOTA performance by up to 7.1% and 8.1% on ALFWorld and WebShop. Extensive ablations and evolution analyses further validate its effectiveness in knowledge internalization and policy improvement. The code is available at this https URL. 

---
# Personas Are Not People: Identity as State, Not Attribute 

**Authors**: Tamunotonye Harry, Yuanyuan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2611.05292)  

**Abstract**: Synthetic populations, large sets of simulated people, are increasingly used in place of real people to pretest surveys, simulate users and communities, and evaluate AI systems. Each simulated person is defined by a persona, a written profile of fixed traits, demographics, and attitudes that a language model is instructed to act out. Every such system assumes that a person's identity is a stable attribute that behaves the same way in every situation the simulation presents. We show that this assumption is not a harmless simplification. Three decades of personality research find the opposite. How a given person behaves varies more across contexts than people vary from one another. Drawing on identity economics, which describes how different situations activate different identities and norms within the same person, we separate behavior into three parts, which are stable traits, the identity a context calls up, and the situation itself. We then prove, under two stated assumptions, that any simulation built on fixed personas, no matter how detailed the personas or how large the underlying model, cannot reproduce the part of a population's behavior that is specific to it and its circumstances. We propose a new evaluation criterion based on this result and a person-in-context design that can meet it in principle. We test one version on a naturalistic Reddit personality corpus and on a published clinical-trial recruitment survey of a community with documented distrust of medical institutions. The design does produce context-dependent behavior when tested directly, but in neither setting does it fully match the real population, and we show where it falls short. The consequence for practice is a risk that synthetic-population evaluations misrepresent the groups whose behavior is most shaped by context, a distortion that richer personas cannot correct. 

---
# Knowingly Wrong: Training LLMs Against Their Own Beliefs can Induce Misalignment 

**Authors**: Ariana M. Villegas-Suarez, Guido Freire, Sohan Venkatesh, Bartosz Cywiński  

**Link**: [PDF](https://arxiv.org/pdf/2611.05287)  

**Abstract**: Fine-tuning a large language model (LLM) on harmful data in a narrow domain, such as bad medical advice, can make it broadly misaligned. How realistic this risk is remains unclear, since models are very rarely fine-tuned on explicitly harmful data in practice. In this work, we show that fine-tuning on data the model believes to be false but that has no harmful content, can also induce broad misalignment. Across mathematics, trivia, cooking and sports, the more strongly the model disbelieves the answers it is trained on, the more misaligned it becomes. In mathematics, misaligned answers rise from 4% to over 60% as the training answers become more wrong, with little loss of general capability. After fine-tuning, the model states the wrong answers but its multiple-choice probabilities still prefer the true ones, which suggests that misalignment comes from this internal inconsistency. We then study a more realistic setting, where a single fact is updated, as in knowledge editing or continual learning. The same relationship holds and the more the model still believes the true fact after training on the false one, the more misaligned it becomes. When we frame the same false fact as part of a coherent story, the model comes to believe it more deeply and becomes less misaligned than when we instruct it to assert the false fact. Content filters and persona vectors cannot tell these two framings apart, even though they lead to different levels of misalignment. These results suggest a risk for model updates, where training a model that does not adopt the new fact as true can induce misalignment even when the data looks harmless. 

---
# Beyond Accuracy: What Language-Subspace Interventions Capture, Remove, and Improve 

**Authors**: Shaanil Punglia, Dadhichi Telwadkar, Vaibhav Rathore, Siddharth Sureka, Neha Bhargava  

**Link**: [PDF](https://arxiv.org/pdf/2611.05278)  

**Abstract**: Language-subspace interventions aim to improve multilingual reasoning by suppressing language-related components in mid-layer activations. Using MATH and MGSM across four open-weight LLMs, we ask three questions: do estimated language subspaces capture translation-associated variation? Does projecting them out remove language information? And do they improve accuracy without compromising response language? For the first question, we build subspaces using the existing Mean-SVD construction and our DiffPCA. Both capture substantial translation-associated variation and show similar patterns across model depth. For the second, we project out each subspace and train linear probes to classify language from the projected activations. The probes still identify language almost perfectly, showing that neither subspace exhaustively captures language information. For the third, we evaluate several intervention configurations on Qwen2.5-7B, by varying subspace creation methods and intervention operators. We find that the intervention that most improves MGSM accuracy sacrifices response-language fidelity, with only one intervention improving joint accuracy--answers that are correct and in the target language--slightly, by 1.60 points. Interestingly, subtle changes in the prompt alone can reproduce the accuracy gains albeit with significant damage to language fidelity. We argue that causal interventions must be judged on more than accuracy, to fully account for language substitution trade-offs. 

---
# ARLoop: Fully-Open Pre-trained Looped Transformers with Attention Residuals 

**Authors**: Yuekun Yao, Yupei Du, Dawei Zhu, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.05234)  

**Abstract**: Looped transformers increase effective depth by repeatedly applying a shared set of transformer layers, offering a parameter-efficient way to scale computation. However, standard residual connections aggregate information across recurrent depth into a single stream, providing limited control over which intermediate representations from preceding layers should be reused. In this paper, we introduce ARLoop, a looped transformer architecture based on attention residuals, where the input of each layer is selectively aggregated through attention over model depth. Distinct from standard transformers, we find that looped transformers do not require access to the full residual history: sliding-window depth attention matches or outperforms full-history attention at larger recurrent depths. Across controlled experiments with multiple parameter scales (140M and 370M) and recurrent depths (R = 1 to 5), ARLoop consistently outperforms prior residual designs for looped transformers. More importantly, we further scale ARLoop to 0.6B and 1.2B parameters under a realistic LLM pretraining recipe, demonstrating that the gains transfer to realistic pretraining, with particularly strong improvements on math and coding tasks. Our 1.2B model, pretrained on only 0.5T tokens with three recurrences, outperforms Huginn-3.5B (the only fully-open looped LLM as of our release date) across all evaluated tasks and is competitive with similar-sized non-looped models such as Qwen3.5-2B despite being trained on substantially less data. 

---
# When Self-Verification Hurts: Calibrated Answer Repair for Financial Question Answering 

**Authors**: Celine Guo, Weida Liang, Yujian Fu, Anqi Dang, Yanzheng Jin, Kenji Kawaguchi, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.05166)  

**Abstract**: Financial question answering can fail even when a model extracts a plausible number from the supplied evidence: the answer may attach it to the wrong entity, period, unit, or accounting concept. Asking the model to verify and revise its draft can help, but open-ended revision may alter content that was already correct or inspect the wrong relation and reinforce the original error. We formulate repair as a decision that balances the information it may recover against the correct content it may damage. Based on this trade-off, we introduce Calibrated Answer Repair (CAR), which first estimates whether a draft is likely to benefit from repair, then selects a targeted check, localizes the relevant evidence or draft span, and limits the permitted edit. Across FinRank, Fin-RATE, and FinQA with three open-weight models, CAR outperforms generic verification in all dataset-model settings. At the system level, it recovers 14.6% of initially missed answer requirements while damaging 2.1% of initially correct ones. Its post-draft token cost is over 30 percent lower than the iterative revision alternatives. Ablations on two dataset-model settings show that evidence localization and bounded editing are the main contributors to CAR's gains. 

---
# GM League Arena: An Interactive Long-Horizon Task Benchmark Based on Professional Sports League Management 

**Authors**: Junkai Zhou, Ziming Pan, Zhaoyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.05147)  

**Abstract**: Long-horizon decision-making under sustained competition requires large language models (LLMs) to balance immediate investment with future resource needs. However, local success in multi-step interactions can not adequately capture cumulative decision effects or demonstrate long-term goal achievement under resource constraints and sustained competition. To address this gap, we introduce GM Bench, a benchmark for managing professional sports teams across seasons under resource constraints and competition among teams. We build the environment simulators for four sports and add financial constraints, player demands, and market mechanisms to establish a unified task across sports. Diverse rosters of comparable strength are constructed from real-world player data, and each LLM is evaluated as a team general manager to assess its long-term management capabilities. We evaluate competitive performance, cash reserves, and cross-season improvement across four sports. Over five NBA seasons, Claude Opus 5.5 ranks first among 29 LLMs, while GPT-6 Astra and GPT-6.1 Sol combine strong performance with larger cash reserves. 

---
# The Right Source Is Not Enough: Understanding History Interference in Speech-LLM Memory 

**Authors**: HongJin Song, Weiqiao Shan, Runwu Shi, Yifei Wu, Yujin Wang, Jiale Luo, Jiasheng Kuang, Xinyu Yang, Zhaoqi Cui, Chuizheng Kong, Ziyu Wu, Guowu Tan, Qiuyu Fang, Chunxiang Jin  

**Link**: [PDF](https://arxiv.org/pdf/2611.05124)  

**Abstract**: Speech large language models retain audio history to support multi-turn conversations. Yet remembering an event does not ensure that its information is used only when relevant. We investigate why irrelevant historical audio changes answers even when the queried event remains represented. Controlled experiments on MOSS-Audio and MiMo-Audio show that historical values direct errors toward specific competing answers, beyond a generic effect of context length. Causal interventions distinguish two paths. In open-ended question answering, most history-value errors disappear when subsequent computation cannot directly access the irrelevant source. In multiple-choice answering, many errors persist: early historical influence is carried by question-candidate states even after direct source access is blocked. Low-dimensional interventions further separate source-role information from answer-value information. Their effects peak at different depths, and changing the source-role signal does not eliminate the causal influence of competing values. On naturally occurring errors, strengthening the queried role and counteracting the historical value each improve answers; their joint intervention gives the greatest recovery. These findings support a failure of downstream evidence use beyond source identification alone: source-role information can influence a decision without exclusively controlling which historical values affect it. Reliable conversational memory therefore requires not only retention and retrieval, but also control over how retained information enters answer computation. 

---
# ToolSpace: Scaling Tool-Centric Environment Synthesis for Long-Horizon Agents 

**Authors**: Guanting Dong, Zhichao Sun, Xinyu Yang, Rui Wang, Xin Deng, Jiajie Jin, Zhixiong Yang, Junhao Huang, Han Han, Mingyu Ma, Xiaotao Wang, Yuyang Hu, Xiaoshuai Song, Xiaoxi Li, Jiejun Tan, Zhecan Wang, Shihan Dou, Honglin Guo, Zhiheng Xi, Tao Gui, Tao Ji, Qi Zhang, Yutao Zhu, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2611.05036)  

**Abstract**: Large language models are deployed as long-horizon agents that compose diverse tools across MCP services into executable workflows. Existing MCP environment-scaling methods are data-centric: they first synthesize a database and scenario, then derive tools and tasks from them. As a result, tool interfaces may not match real ones, tool compositions are not grounded in execution, and tasks rarely span more than one environment. We propose ToolSpace, a tool-centric framework that reverses this order. ToolSpace mines data, state, and retry dependencies from adjacent calls in execution traces of real MCP services and open agent corpora, using deterministic rules rather than LLM judgment, and clusters the resulting weighted tool-composition graph into environments. Databases are synthesized only for state-mutating environments, while query and stateless tools replay recorded returns. ToolSpace then performs state-aware random walks with rollback and decompiles each chain into a natural-language task with assertion-based checks, keeping only samples that replay successfully from the initial state. Applied to AutomationBench and tau3-Bench Banking, the pipeline produces harder Pro versions on which 7 frontier models drop by up to 38.2 points. Across seven long-horizon tool-use benchmarks, ToolSpace raises Qwen3.6-35B-A3B from 23.0 to 41.0 average score (43.2 with RL) and GLM-5.2 from 44.6 to 56.6, on par with Claude Opus 5. Ablations show that execution-witnessed edges clearly outperform LLM-judged ones, and performance keeps improving with data and environment scale. 

---
# Whose Effect Is It? How Objectives Inflate Learned Subspace Interventions 

**Authors**: Toygar Tanyel, Serkan Ayvaz  

**Link**: [PDF](https://arxiv.org/pdf/2611.05033)  

**Abstract**: Interpretability often locates concepts by copying part of a language model's state from one input into another (activation patching). A larger output change counts as stronger evidence that this part carries the concept. Methods that learn which part to copy often change the output more than the whole state does. By that reading, learned parts isolate the concept more cleanly. Yet the whole contains the part. So whose effect is this excess, the network's or the method's? Copying the whole state learns nothing, so it shows what the network produces. We prove that, to first order, the effects of orthogonal parts add up to that of the whole. A part therefore exceeds the whole exactly when the rest moves the output back, which does not require isolating the concept more cleanly. Objectives that reward a larger effect select such parts. Distributed alignment search with cross-entropy, for example, exceeds the whole-state copy just before the output on almost all tasks, and inside the network at most sites. In both, changing only its objective to target what the network produces removes most of the excess. It therefore comes largely from the objective, so a larger learned effect alone is not stronger evidence. 

---
# Who is the Pronoun Pro? Comparing LLMs and Humans at Number-Disambiguation of 'they' 

**Authors**: Carlos Hartmann  

**Link**: [PDF](https://arxiv.org/pdf/2611.04968)  

**Abstract**: This study evaluates whether large language models (LLMs) can reliably replace or complement human research assistants in linguistically complex annotation. In a controlled annotation task, 73 human annotators (local students and crowdworkers with and without a background in linguistics) and more than 40 LLM configurations number-disambiguated 174 authentic uses of 'they' from Reddit, including innovative and context-sensitive cases. The best LLM, Kimi K3, reached 94.5% accuracy, above the human majority vote (86.2%) and the mean of the top half of human annotators (78.5%); only the single best human annotator scored higher (95.7%). LLMs also offer substantial practical advantages in cost and speed. However, qualitative error analysis reveals a crucial complementarity: human reasoning in low-agreement cases exposes inconsistencies in the annotation typology and thereby improves the methodological framework itself. The study contributes (i) a benchmark-style human-LLM comparison for a semantically demanding task, (ii) transparent cost and time evidence, and (iii) a methodological argument against a purely replacement-oriented view of automation. LLMs are viable for large-scale annotation support in linguistics, but their responsible use requires human critical oversight. The dataset with its gold-standard labels, all human and LLM annotations, and the fully reproducible code are openly available as a Quarto-based, public repository. 

---
# Shuffled OPSD: When Semantic Consistency Hurts On-Policy Self-Distillation? 

**Authors**: Qingyuan Zheng, Zhihao Dou  

**Link**: [PDF](https://arxiv.org/pdf/2611.04958)  

**Abstract**: Knowledge distillation improves a student model's reasoning ability through supervision from a teacher. On-policy self-distillation (OPSD) uses the same model as both teacher and student under different contexts: the student generates reasoning trajectories from the question alone, while the teacher additionally receives a complete worked example and provides token-level supervision along those trajectories. Because the problem statement is shared, its content and ordering shape both student generation and teacher feedback. Preserving its semantic structure therefore seems natural, but does it always improve learning? We introduce Shuffled OPSD, which shuffles the words of the shared problem statement while keeping the teacher-only worked example intact. Across three mathematical reasoning benchmarks and three Qwen3 model scales, the method improves mean accuracy over both an intact-target control and Standard OPSD under matched training configurations. The largest gain over Standard OPSD is 3.52 percentage points on Qwen3-1.7B. Further controls show that preserving larger word blocks offers no advantage over word-level shuffling at the evaluated scales, whereas shuffling the teacher-only worked example reduces accuracy at every scale. Cross-problem mixing and target-removal experiments further distinguish the roles of problem content, word order, and teacher context. These results show that disrupting the shared problem statement can improve learning, while disrupting the teacher-only reference hurts performance. They challenge semantic coherence as a universally beneficial principle for OPSD and motivate closer study of how input structure shapes student generation and teacher supervision. 

---
# HERIT: Enabling Global Access to Hanja-to-English Historical Translation by Mitigating Temporal Bias 

**Authors**: Dayeon Jeong, Kyeongpil Kang  

**Link**: [PDF](https://arxiv.org/pdf/2611.04949)  

**Abstract**: Historical archives are invaluable for multidisciplinary research but remain inaccessible globally due to severe language barriers. While manual translation is prohibitively expensive and time-consuming, simply applying existing machine translation models is inadequate due to unique linguistic nuances and the temporal bias inherent in scarce human-labeled corpora. To address these challenges, we propose HERIT, a systematic domain-adaptation pipeline that leverages retrieval-augmented generation (RAG) to generate high-quality pseudo-labeled data from abundant Hanja-Korean corpora. This approach substantially expands the training dataset, mitigating both data scarcity and temporal overfitting across different historical periods. Extensive automated and human evaluations demonstrate that HERIT consistently outperforms baselines. Finally, we translate previously untranslated portions of the archives using our model, thereby enabling global access to these resources for researchers. 

---
# Residual Attribute Leakage Is an Interface: Auditing Privacy Evaluation After Text De-identification 

**Authors**: Tuan Thanh Duong, Joongmin Shin, Jeongho Yoon, Junyoung Son, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2611.04941)  

**Abstract**: A privacy evaluation of de-identified text asks a language model (the adversary) to guess a person's protected attribute. We introduce a prior-normalised clipped leakage score of its confidence gain on the true value and audit the conclusions drawn from it. Each verdict depends on an evaluation interface (the queried person, answer options, prior, adversary model and scoring rule), which we vary on fixed releases of three corpora. In a stress test, an editor that writes the opposite-sex form at the same spans a redaction editor would redact misleads the adversary about most people, yet ranks first among editors scored on the same MedQA entries under this score, as under attack accuracy and proper scoring rules read as privacy. A relabelling-invariant reading of AUC ranks that editor sixth of eight, and a rule-aware adversary recovers the attribute from the same text. On ordinary releases most verdict changes are gains or losses of separability; of six choices varied over all pairs, only the adversary reverses comparisons (5 of its 124 changes), and the query can target the wrong person. SACC, a traceable editor we build as the instrument, lets us ablate two editing principles and show where its advantage does not hold. 

---
# PROSE: Preservation-Constrained Selective Editing for Guilty Bias in Legal Judgment Prediction 

**Authors**: Yimeng Wang, Zeyuan Tian, Xinyu Wang, Danyang Gao, Chenglei Shen, Zhongxiang Sun, Lei La, Weijie Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.04921)  

**Abstract**: Legal judgment prediction models may retain a guilty verdict after a legally decisive condition has been altered such that the specified charge is no longer established. We study this guilty bias via selective repair over matched charge-established and charge-not-established cases, where the unsupported verdict should change while its supported counterpart remains stable. We construct CAIL-MC, a legally reviewed dataset of 11,478 matched pairs covering 261 Chinese criminal charges. Held-out activation interventions show that contrasts learned from correctly resolved pairs transfer to false-guilty cases, but target-reaching interventions can also damage matched supported-guilty decisions. Motivated by this observation, we introduce PROSE, which reconstructs case-specific targets from a training-derived residual geometry, aligns them with a local margin-descent signal, attenuates supported-guilty-sensitive components, and realizes bounded targets through a preservation-regularized low-rank parameter update. Across Qwen3-8B and GLM4-9B, PROSE achieves stronger pairwise safe repair and evidence sensitivity, while preserving standard single-pass inference. 

---
# TraceRouter++: Causal Semantic Routing Realignment via Optimal Transport 

**Authors**: Chuancheng Shi, Shangze Li, Wenhua Wu, Hengzhi Chen, Fei Shen, Li Zhuo, Zhiyong Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2611.04878)  

**Abstract**: Despite their remarkable capabilities, large foundation models (LFMs) remain susceptible to adversarial manipulation. Current safety intervention methods primarily rely on the \textit{locality hypothesis}, rigidly suppressing isolated neurons or features. However, harmful semantics manifest as distributed, cross-layer circuits, rendering localized ablations both brittle against complex jailbreaks and detrimental to general utility. To bridge this gap, we propose TraceRouter++, a routing-level safety intervention framework that explicitly traces safety-relevant causal routing pathways activated under harmful inputs and realigns their unsafe activation states. Our framework operates in a \textit{Discover-Trace-Realign} paradigm: it first pinpoints a sensitive onset layer via attention divergence and disentangles dense activations into interpretable routing features using sparse autoencoders (SAEs). Subsequently, it maps these features to downstream causal circuits via feature influence scores (FIS). Crucially, departing from rigid suppression paradigms, TraceRouter++ introduces a constrained optimal transport (OT)-based soft routing realignment. By formulating safety intervention as a tractable coordinate-wise marginal optimal transport approximation, our method redirects unsafe activation marginals toward safe-reference marginals within causally identified routing subspaces while preserving non-selected coordinates at the intervention point. This mechanism safely redirects malicious semantic flows while leaving orthogonal computational routes intact. Extensive experiments across representative large language models (LLMs), multimodal large language models (MLLMs), and diffusion models (DMs) show that TraceRouter++ outperforms evaluated baselines, improving adversarial robustness while maintaining a favorable safety-utility trade-off. WARNING: This paper contains unsafe model responses. 

---
# One Assistant, Many Memories: Benchmarking Real-World Group Assistance When Recall Is Not Enough 

**Authors**: Zhanghao Hu, Linhai Zhang, Qian Zhao, Qi Zhu, Yuan Hua, Di Liang, Jiasheng Si, Xin Zhao, Yulan He, Zhumin Chen, Lin Gui  

**Link**: [PDF](https://arxiv.org/pdf/2611.04867)  

**Abstract**: The value of memory lies not only in recall but in using it to provide appropriate assistance. Recent benchmarks test memory use in personalisation and agentic tasks, but seldom evaluate assistance without fixed answers or integration across multiple participants. Group chats make both challenges more acute: an assistant must integrate information from multiple members, track revisions and conflicts over time, and produce one response for the group. Yet existing group memory benchmarks rely on synthetic conversations and inherit constrained evaluation from general memory benchmarks, focusing on recall against fixed answers or decisions over a closed set. We introduce GroupAssistBench, the first memory benchmark from real-world group conversations with a deployed group assistant, covering 80,299 messages in 50 groups and costing over $53K. Over the same histories, GroupAssistBench evaluates memory recall (L1) with reference answers and memory use (L2) with service requests without fixed answers across single, distributed, and conflicting memories (M1--M3). Requests are verified to require group history, memories are anchored to their sources, and L2 uses tailored rubrics with penalties derived from real user feedback. Across evaluations of frontier models with full context, retrieval methods, and memory systems, the strongest memory system rivals full context on recall but falls far behind on use, especially for distributed and conflicting memories. Together, two additions that relate retrieved memories and bind remembered facts as response constraints bring consistent gains, showing that group memory systems must improve use, not recall alone. 

---
# ReSI-Math: Exploring Recoverable Capability Frontiers for Efficient Reasoning Improvement 

**Authors**: Yao Cui, Zhen Yang, Yuandong Wang, Yangfu Zhu, Tianyu Wo, Zhenzhou Shao  

**Link**: [PDF](https://arxiv.org/pdf/2611.04787)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become an effective approach for improving mathematical reasoning in large language models. However, a fundamental question remains: when a model fails to solve a problem, does it truly lack the underlying capability? In this work, we reveal that direct failure does not necessarily imply inability. Some reasoning failures can be recovered by exploring the model's own trajectories, exposing a recoverable capability frontier: a policy-dependent region where solutions are not reliably generated through direct decoding but can be elicited through trajectory-level search. Based on this observation, we introduce ReSI-Math, a framework that explores recoverable capability frontiers and converts recovered capabilities into data-efficient RLVR supervision. ReSI-Math uses Self-Judge-guided evolutionary search to identify recoverable failures under a fixed policy and adaptively selects frontier problems for sequential RLVR as the policy evolves. On Qwen3-VL-8B, correct-incorrect trajectory crossover produces 272 verifier-confirmed correct offspring from 621 attempts (43.8%). After three sequential one-shot RLVR stages, the MATH500 and WE-MATH average improves from 78.71 to 80.16. Under matched single-problem RLVR, random selection reaches only 53.31 versus 54.16 for frontier-guided Q1. 

---
# When Ground Truth Is Not Ground Truth: Automatic Labeling for Hallucination Detection 

**Authors**: Rounak Sen, Divya Kumari, Deeksha Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2611.04697)  

**Abstract**: Hallucinations in Large Language Models (LLMs) remain a major challenge, as they can produce misleading, factually incorrect, or unverifiable information, thereby undermining their trustworthiness and reliability. However, reliable ground-truth labels remain a key challenge in evaluating hallucination detectors. Recent studies often construct labels automatically by comparing model generations with reference answers using similarity metrics such as BLEURT. This implicitly treats BLEURT similarity as a proxy for hallucination, despite its English-centric supervision, unbounded score range, and design as a similarity metric rather than a factuality measure, making threshold-based ground-truth construction potentially unreliable. Through comprehensive experiments using 15 automatic supervision judges, seven hallucination detection baselines, two generator models, two QA datasets, and five languages, we demonstrate that detector performance varies substantially with the choice of supervision judge. Human validation reveals substantial differences in labeling reliability, with QGen-9B achieving the lowest overall misclassification rate (12.86%), while BLEURT and BLEURT-n exhibit rates of 47.23% and 47.32%, respectively. Our findings highlight the limitations of automatic similarity-based labeling and the need for more reliable judging methods that better capture semantic and factual correctness. 

---
# CardioVascBench:Prioritized Diagnosis Generation from Real-World Chinese Cardiovascular Records 

**Authors**: Minghao Kong, Muyao Chen, Jianhao Wu, Yuqiang Zhang, Xiangbin Meng, Zirong Liu, Ting Cao, Kaida Ning, Zhen Song, Rongjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.04687)  

**Abstract**: Identifying the primary diagnosis alone does not fully characterize a model's diagnosis-generation performance. We introduce CardioVascBench, a retrospective benchmark for prioritized diagnosis generation from real-world Chinese cardiovascular inpatient records. It contains 3,262 de-identified cases, with 2,174 for training and 1,088 for evaluation, linking clinical narratives and textual ECG reports to expert-verified, ordered discharge diagnoses standardized to ICD-10. An English translation of the evaluation set supports aligned cross-lingual comparison. Our evaluation protocol combines terminology normalization and rank-aware scoring to separately assess primary-diagnosis identification, reference-diagnosis coverage, and list precision. Evaluation across 36 systems shows that higher primary-diagnosis accuracy does not necessarily imply higher list precision against discharge references. We also provide CardioKG-Agent, a knowledge-enhanced baseline combining entity normalization, constrained graph retrieval, and guideline evidence. In a staged comparison of three baseline models, supervised fine-tuning plus the agent achieves a 73.5% higher mean composite score than fine-tuning alone, representing a system-level gain. These results measure retrospective agreement with recorded diagnoses; agreement between metric rankings and physician judgments remains to be validated. CardioVascBench supports research on diagnosis prioritization and knowledge-enhanced generation. 

---
# MIMIC-Note-Extract: A Large-Scale Structured Dataset Extracted from MIMIC-IV-Note Discharge Summaries 

**Authors**: Kristian Schwethelm, Jonas Kuntzer, Conrad Ketzer, Eva Wende, Moritz Schlager, Daniel Rueckert, Georgios Kaissis, Friederike Jungmann, Paul Hager  

**Link**: [PDF](https://arxiv.org/pdf/2611.04685)  

**Abstract**: MIMIC is one of the most widely used public hospital datasets for machine learning on electronic health records, yet clinically important information, such as past medical history, allergies, and diagnostic findings, remains confined to free-text discharge summaries. Studies requiring this information must therefore build their own extraction pipelines, duplicating substantial effort across projects. We release MIMIC-Note-Extract (MNE), a large-scale dataset that uses a hybrid extraction pipeline combining deterministic parsing with schema-guided large language model generation to transform 16 sections from 300,720 discharge summaries into structured tables. MNE adds information that has no structured MIMIC counterpart and enriches related tables with broader coverage and greater clinical detail, including 10 times more patient-history entries and medication histories for more than twice as many admissions. Validation by medical doctors confirms extraction accuracy above 92% across sections. Using MNE, we construct a clinical decision-making benchmark for LLM agents, demonstrating how reusable structured representations of clinical narratives reduce data curation effort. 

---
# Evaluating LLMs as Judges of Counterspeech Quality Against Online Hate Speech 

**Authors**: Johnny Alali, Nicolás Benjamín Ocampo, Davide Ceolin  

**Link**: [PDF](https://arxiv.org/pdf/2611.04627)  

**Abstract**: Prolonged exposure to hate speech on social media can cause lasting psychological distress among users. Counterspeech has emerged as an effective, non-censorial mitigation strategy, but its deployment requires rigorous quality assessment. Manual assessment is unscalable due to the large volume of content on the Web and imposes severe emotional strain on human annotators. Large Language Models (LLMs) offer a scalable alternative, yet concerns remain regarding their reliability. In this paper, we investigate the extent to which LLMs can effectively perform this task. We test 12 open-weight LLMs of different sizes and families under zero-shot, few-shot, and chain-of-thought prompting, to approximate pointwise and pairwise human expert assessment of counterspeech quality across four dimensions: relevance, coherence, aggressiveness, and suitableness. We find that (i) LLMs show low agreement and moderate rank correlations with human experts: our best configuration, Qwen 2.5 72B on coherence, achieves up to a linear-weighted Cohen's $\kappa_l = 0.19$ and Spearman's $\rho = 0.42$ in pointwise evaluation and an unweighted Cohen's $\kappa = 0.25$ in pairwise evaluation. (ii) Converting pointwise scores into pairwise labels substantially improves human--LLM agreement for selected configurations. For same-instance coherence, Qwen 2.5 72B under zero-shot prompting achieves $\kappa = 0.35$ with converted labels, compared with $\kappa = 0.25$ when judging pairs directly, and Mixtral 8$\times$7B under few-shot prompting improves from $\kappa = 0.07$ to $\kappa = 0.34$. (iii) LLMs agree more with one another than with human experts, particularly in pairwise evaluation, where the strongest model pair, OLMo 2 32B and Llama 3.3 70B on coherence, reaches substantial agreement with $\kappa = 0.76$, whereas their agreement with the experts in the same configuration does not exceed $\kappa = 0.21$. 

---
# TEMPO: Time Series Understanding via Discrete Tokenization 

**Authors**: Riccardo Maggioni  

**Link**: [PDF](https://arxiv.org/pdf/2611.04482)  

**Abstract**: Time-series interpretation often requires both numerical observations and textual context. Integrating these inputs into a pretrained language model raises a representation question: how should a signal enter the model's token sequence? We present TEMPO, which extends an LLM's vocabulary with discrete codes from a context-aware Transformer tokenizer using finite scalar quantization. A frozen tokenizer maps four-step patches to a 625-code vocabulary; signal tokens, scale statistics, and text are then processed by the LLM's existing self-attention layers. Adaptation updates the new token embeddings and DoRA adapters. With a Qwen3-4B backbone and task-specific fine-tuning, TEMPO achieves macro-F1 scores of 89.0% on activity recognition, 85.0% on sleep staging, and 86.0% on bearing-fault diagnosis. Time-series question answering and qualitative bearing-diagnosis examples further illustrate how this interface combines signal representations with textual context. Together, these results demonstrate the utility of discrete vocabulary extension for time-series understanding. 

---
# ReListen: Training-Free Evidence Localization in Hour-Long Audio with Frozen Audio Language Models 

**Authors**: Ming-Hao Hsu, Shujie Liu, Yuxuan Hu, Yuxiang Wang, Yingda Shen, Jinyu Li, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.04480)  

**Abstract**: Audio language models are meant to answer questions from the audio itself, but an hour-long recording rarely fits in their context, and the answer often depends on a few seconds of it. Transcribing the recording first makes it fit but discards sounds and voices, while systems that return to the audio must be trained for this behavior. We propose ReListen, a training-free method in which a frozen audio language model finds its own evidence before answering. In a Jev-style scan, the model scores each 20-second window by the probability that it helps answer the question, and it then answers from the highest-scoring minute. This minute overlaps the annotated evidence far more often than transcript retrieval, for 88.5% of VoiceGiraffe questions. Without any change, the same scan also grounds events on LAT-Bench as well as a model trained for temporal grounding. Answering from it matches or exceeds hearing the whole recording with a small fraction of the audio, and on VoiceGiraffe it exceeds a transcript of the whole recording by 9.2 points. The selected evidence is also worth hearing rather than reading. Given the same excerpts, reading their transcript matches listening on speech content, but listening is clearly more accurate on questions about sounds and voices. Moreover, ReListen answers faster than every transcript-based system, and its memory stays constant even for recordings too long to hear whole on one GPU. Therefore, a frozen audio language model can search hour-long recordings by itself, without a transcript or additional training. 

---
# Typological Feature Induction as a Benchmark for Inductive Metalinguistic Reasoning 

**Authors**: Chen Zhang, Seth Aycock, Yusuke Miyao  

**Link**: [PDF](https://arxiv.org/pdf/2611.04421)  

**Abstract**: Existing metalinguistic reasoning benchmarks require both inference of rules and their application, making failures hard to diagnose. We introduce typological feature induction (TFI) which targets the former step. Given sentences in an unnamed language paired with translations in a known pivot language, models must predict a typological feature of the unnamed language. From Grambank and multi-parallel corpora, we automatically build evaluation questions over 195 languages. We find that LLMs identify 59--80% of these languages from only 16 sentence pairs; and misaligning the translations barely affects their accuracy, suggesting reliance on prior knowledge of the language. Enciphering text to prevent recognition increases the corpus' utility, but mainly for features where target and pivot languages agree. On languages which models cannot identify, neither open model outperforms a majority-value baseline. Since induction from raw text aligns with field linguists' use-case, this work indicates the models tested are not yet ready for this role. TFI offers a scalable path to track progress in this direction. 

---
# The Min-Max theory of language behavior 

**Authors**: Hiram Ring  

**Link**: [PDF](https://arxiv.org/pdf/2611.04406)  

**Abstract**: The "Min-Max" theory of language behavior proposes that humans engage simultaneously in minimization of processing effort and maximization of information content, and that this is a primary driver in language evolution (Ring, 2025). This claim seeks to unify observations from language processing (Mahowald et al., 2013; Futrell et al., 2015; Yadav et al., 2020; Hawkins, 2014), interaction (Hörberg and Sjons, 2023; Tucker et al., 2025; Jaeger and Tily, 2011; Smith and Culbertson, 2025), and corpora (Levshina and Moran, 2021; Levshina, 2023; Newmeyer, 2017), which suggest that considerations of processing efficiency and information content are both implicated in human communication. The interaction between these two processes provides an explanation for human linguistic behavior at various levels (Linders and Louwerse, 2023) and predicts historical developments in language, i.e. related to word order (Ring, 2025). Given that the original formulation of the Min-Max theory is based on observational data, the current paper seeks to validate the theory mathematically. Through a series of experiments that use computational models to investigate the theory's predictions, pretrained and iterated learning models are queried with synthetic data. The experiments show that manipulating the variables of token length and the weight of information content in a tabula rasa model results in word order change, as predicted by the theory. The experiments also identify important differences between computational models and observations of human language data, suggesting that current transformer-based language models may not accurately represent psycholinguistic processes. The overall findings indicate that evolutionary trajectories in language are highly susceptible to minimization of processing and maximization of information content. 

---
# Do Multi-Turn Conversational Capabilities Carry Over from Text to Speech? A Parallel Multilingual Benchmark 

**Authors**: Jiaming Zhou, Pei Zhang, Shiwan Zhao, Zhijun Wang, Baosong Yang, Yong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2611.04390)  

**Abstract**: Successful multi-turn voice interaction requires models to retain information from earlier turns, follow persistent instructions, and track revisions to conversational state. These abilities are usually evaluated from transcripts, leaving open whether successes demonstrated from text carry over when the same turns are spoken, and whether this transfer is consistent across languages. We introduce Omnilingua-AudioMC, a strictly parallel multilingual extension of Audio MultiChallenge, to study this question. The benchmark contains 354 conversations and 1,404 rubric items across ten languages, with content, turn structure, cross-turn dependencies, and evaluation criteria aligned across transcript and speech conditions. Across six speech-language models, the shift from reference transcripts to direct speech input is strongly model- and language-dependent: models with similar transcript performance can diverge substantially under speech, and their relative strengths across languages can change. Paired rubric-level analysis shows that aggregate transcript-speech gaps can obscure which conversational requirements are preserved, because losses of transcript-demonstrated successes can be offset by gains on other items. Replacing preceding user turns, the current turn, or both with model-generated transcripts localizes recoverable degradation: for the two models with the largest gaps, history substitution yields a larger macro-level improvement than current-turn substitution in all ten languages, while the strongest direct-speech model gains little from either. These results show that transcript performance is an unreliable proxy for multilingual spoken conversational ability and identify the representation of preceding spoken context as an important source of recoverable degradation. Our code is publicly available at this https URL. 

---
# Sparse Internal Control of Language Models 

**Authors**: Xizhe Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.04349)  

**Abstract**: Language models are increasingly used in text generation, decision support, and automated interaction, creating a need for localized, reversible, and selective control of their behavior. Activation steering demonstrates that internal interventions can alter the behavior of frozen models, but how to select and evaluate compact, multi-site control interfaces remains an open question. I introduce sparse internal control, a framework for steering target behavior by applying inference-time interventions to a small set of internal model nodes. A local target-controllability formulation characterizes candidate site--direction pairs by their behavioral effects and specifies constraints on target recovery, off-target preservation, and intervention energy. Two complementary selectors instantiate this formulation: Driver-OMP selects nodes for sparse reconstruction of a desired behavioral displacement, while Coverage-Driver favors stable coverage across prompts. I propose a six-axis control-evaluation protocol measuring reachability, signed reversal, dose response, feedback controllability, off-target preservation, and held-out reuse. Exploratory experiments on indirect object identification, MMLU multiple-choice question answering, sentiment steering, and refusal steering provide evidence for the feasibility of steering task-defined behavior through sparse internal interventions. The six-axis evaluation characterizes these interventions beyond target movement, revealing distinct control properties that depend on the task and available direction information. Together, the formulation, reference algorithms, and evaluation protocol provide a foundation for the systematic study and development of internal control methods for language models. 

---
# COMPREHEND: Ultra-Fine-Grained Benchmark Understanding by Entailment Hierarchy 

**Authors**: Sivaraaman Balakrishnan, Prakriti Shetty, Jeremiah Hong, Hamed Zamani, Haw-Shiuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2611.04249)  

**Abstract**: Understanding what kind of prompts users typically ask and how well an LLM performs on these prompt categories is crucial for LLM practitioners. The existing methods satisfy the demand by hierarchically clustering the user prompts and synthesizing the cluster names using an LLM to build a taxonomy. However, the cluster names from these methods are usually too general for humans to understand the detailed prompt distributions and identify the exact strengths and weaknesses of LLMs. In this study, we propose COMPREHEND, which automatically constructs an ultra-fine-grained entailment hierarchy for the prompts. The entailment hierarchy replaces the cluster names in the taxonomy with sentences to provide more precise cluster descriptions for more fine-grained categorization. Compared with EvalTree, a state-of-the-art taxonomy construction method, COMPREHEND improves an entailment/faithfulness metric from 22.6 to 41.7 and an informative metric from 16.3 to 39.1 in WildBench. Our experiments also show that the weakness report generated based on our entailment hierarchy is much more informative than EvalTree in terms of predicting the response scores in the benchmark. All the code is released in this https URL. 

---
# Constructing a Regional Slang Lexicon for Electricity Demands via Soft-Label Bidirectional Anchoring and Active Learning 

**Authors**: Rong Chang, Lijing Shi, Zhaoxiang Zang  

**Link**: [PDF](https://arxiv.org/pdf/2611.04231)  

**Abstract**: Power-grid customer service increasingly relies on community social media, where customers describe outages, billing, and equipment faults in informal regional dialect. Such messages exhibit non-standard morphology, semantic drift, and out-of-vocabulary expressions that make manual lexicon construction costly and degrade conventional NLP tools. We propose a weak-supervision framework constructing a regional slang lexicon from curated dialect anchors and power-domain seed terms. Candidates are generated via Pointwise Mutual Information with soft-label bidirectional anchoring, retaining dialect-anchored terms while recording power-domain relevance as separate metadata, and are scored by combining Sentence-BERT semantic similarity with a jieba-based wordhood measure. Uncertainty-aware active learning with rule-based and retrieval-augmented LLM oracles is applied under three query strategies, and the resulting lexicon is evaluated by a higher-capability LLM calibrated against human expert annotations via Cohen's $\kappa$. On 11,177 real WeChat messages from Yuxi (Yunnan, China), the framework achieves an evaluator-estimated dialect precision of DP~$=$~1.0 with zero false positives. An ablation shows that soft-label layering retains 751\% more candidates than strict power-gating at identical precision, expanding the candidate pool from 41 to 349 terms. Human calibration ($n=46$) yields $\kappa_{\text{dialect}}=1.0$ and $\kappa_{\text{power}}=0.85$. The resulting 25--33-term lexicon, 4--5 of which are power-relevant, provides a preliminary resource for dialect-aware power-grid service NLP. 

---
# Shared Context Candidate Scoring with Jointly Adapted Readout Networks 

**Authors**: Jingyang Du, Yiben Fu, Yang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.04218)  

**Abstract**: Repeated decisions over a shared document create two distinct opportunities: reusing document computation and adapting how question representations score candidate answers. We present Q64, a candidate-scoring system that preserves a frozen 64-layer Qwen3.8-27B text backbone, reuses its hybrid prefix state and intermediate document representations, and jointly trains a residual adapter with a four-layer decoder and candidate-conditioned readout. We used ZiYor, a self-evolving AI system, to explore and refine the Q64 design. We specified the scoring method, implemented the system, and evaluated its accuracy and latency. The final adaptation stage updates 88.66 million parameters using 8,192 selected training views. On a fixed 5,112-item evaluation combining ARC-Challenge, ANLI-R3, and stratified subsets of MMLU-Pro and CMMLU, joint adaptation improves four-benchmark macro accuracy from 66.46% to 74.32%. It remains below the native 27B baseline at 75.68% and AnyJev Qwen3-32B L0 at 76.68%. On a separate synthetic workload with 4,096 document tokens and 50 questions, median cold-request latency is 2.237 seconds, compared with 5.383 seconds for the native baseline and 56.872 seconds for the measured batched AnyJev configuration. Short single-question requests do not share this advantage. The evidence supports an accuracy-latency tradeoff in the evaluated systems, rather than a backbone reduction or an isolated component advantage. Public benchmarks informed earlier development, and real-document multi-question validation remains open. 

---
# Tokenizer Myths and Where To Find Them 

**Authors**: Rylan Schaeffer, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2611.04215)  

**Abstract**: Tokenization is a fundamental component of modern language models: before a model sees data, its tokenizer has already determined how the data appears. We identify and challenge three myths in the tokenization research literature. Myth #1: Bits-Per-Byte (BPB) is agnostic to the tokenizer and/or the vocabulary size: BPB is tokens per byte times bits per token, and the two factors do not cancel; using untrained models, we show that BPB prefers some tokenizers over others: the model has learned nothing, but the tokenizer--model pair has. Myth #2: lower BPB means a better model: On untrained models, BPB strongly prefers some vocabulary sizes, even though accuracy remains at chance, and on real models trained on the same data, BPB correlates weakly with downstream performance. Myth #3: different "tokenizer-agnostic" metrics yield consistent results: The "unigram-normalized loss" is not tokenizer-agnostic; its reported correlation of 0.989 with BPB comes from a confounder and falls to 0.057 once fixed; from the same data the two metrics select different vocabularies at each model size. We present a tokenizer trilemma: no metric can score only the true distribution best, give the same predictions the same score, and tie all untrained models, so choosing a metric is choosing a research question. We conclude with a caution for reading the tokenization literature and a conjecture for future work. 

---
# Length Is Cheap, Competition Is Not: Attention Predicts Context Rot in Retrieval 

**Authors**: Yinuo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.04169)  

**Abstract**: Why do language models fail on long inputs? The common answer is length. We show that, for retrieval, length is cheap and competition is not. On five open models (0.5B to 3B parameters, four families), neutral filler costs nothing: accuracy stays at 0.97 to 1.00, up to 32K tokens on Qwen2.5-1.5B. Statements in the format of the evidence, placed at the same positions, drop it to 0.16 to 0.33. A single statement that gives the queried key a rival value multiplies the loss by 9 to 110, and every additional rival adds to it. Inside the model, one quantity tracks the failure: the attention mass left on the evidence. Across 126,473 trials it explains up to 58% of the variance in loss; log length explains at most 8%. The masked-filler protocol behind the claim that length alone hurts reproduces on MMLU (11 to 18 points lost at 30K tokens), but fails a visible control in retrieval: masking costs 45 accuracy points at 32K tokens, while the same filler left visible costs none. In retrieval, context rot is a competition for attention. We release the probe and all 126K trial-level measurements. 

---
# PACT: Certifying What Is Written in Parallel Decoding of Diffusion Language Models 

**Authors**: Yanchao Li, Yusu Zhang, Jiaqing Xie, Yi Ban, Ben Gao, Zhongyu Huang, Tianfan Fu, Yuqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.04152)  

**Abstract**: Parallel decoding speeds up masked diffusion language models by writing several tokens per forward pass. Published commit rules choose these tokens by testing the predicted distributions. Under sampling the written tokens are drawn, and these tests do not see them. We find that answer quality follows the reliability of the tokens actually written. An answer's log probability splits into this reliability and the dependence within each step. We propose PACT, a training-free commit rule that draws the tokens first, certifies the set it will write and spaces the positions of each step apart. This certificate holds at every temperature, and we show that no rule with the same guarantee commits more positions. Experiments use four models of the LLaDA and Dream families and HumanEval, MBPP, GSM8K, MATH500 and IFEval benchmarks. PACT reaches the quality of published commit rules with fewer forward passes. Spacing removes positions from each step, yet it usually lowers the total cost because later steps then write more tokens. At the same speed, published decoders also write a word twice in a row far more often than PACT. These results suggest that commit rules should certify the tokens they write. Code is available at this https URL. 

---
# AdSteer: Native Advertising in Large Language Model Generation 

**Authors**: Shuhang Lin, Chuhao Zhou, Zhenyu Wang, Xiao Lin, Changsheng Wang, Saibo Geng, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2611.04147)  

**Abstract**: Large language model (LLM) services face rapidly rising serving costs. To sustain free or affordable access for a broad user base, alternative monetization mechanisms are needed. Native advertising offers a promising approach, as it creates economic value by aligning user intent with merchant offerings. Yet how to integrate sponsored content naturally into LLM responses remains largely underexplored. Existing approaches primarily rely on prompt-level injection, multi-pass rewriting, or fine-tuning, which can compromise response quality or introduce additional computational costs. We propose AdSteer, a lightweight decoding-time advertising framework that adaptively adjusts token probabilities to integrate contextually appropriate advertisement into responses. AdSteer calibrates the strength and duration of advertising influence, making sponsored content a natural part of the response. Experiments on GEM-Bench across three application scenarios show that AdSteer achieves an average improvement of 14.1% in overall score over the strongest baselines and 22.5% fewer total tokens on average than rewrite-based methods. These results demonstrate that AdSteer provides an effective and practical solution for native advertising in LLMs. 

---
# Where Do Distilled Students Store Thinking? 

**Authors**: Taeyeong Kim, Jaemin Jung, Ahhyun Kim, Unggi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.04141)  

**Abstract**: Small students degrade on large reasoning models' (LRMs') long solutions, so recent work shortens them by interpolating an LRM with a short-answer model at ratio $\lambda$. We ask what student weights retain when this teacher stops thinking, comparing LoRA adapters of three students from three teacher pairs at each $\lambda$. In each pair, the adapters' update subspace shifts most where the teacher stops thinking and solutions shorten most, locating the change without attributing it. We identify a shared adapter subspace tied to thinking-like behavior (reflective markers, monologue format and long solutions). Shared by adapters before that point, it keeps about half (post hoc) while one pair's teacher halves its solutions but keeps its think tag, and largely empties after thinking stops. Removing it largely turns this behavior off for every pair, student and benchmark. Only in that pair and one other student does it leave clearly fewer markers than removing the update's top components; elsewhere, same-energy random directions often nearly match it. Grafted into a short-answer student at twice its weight, it turns early markers (a post hoc count) on, like the whole update, but the student rarely finishes. Our work identifies a weight subspace tied to distilled thinking-like behavior. 

---
# Say Less, Mean Better: Training-Free Verbosity Control for Language Models 

**Authors**: Mahmoud Alyosify, Mirna Imbabi, Eman Elkhamisi, Rahatara Ferdousi  

**Link**: [PDF](https://arxiv.org/pdf/2611.04107)  

**Abstract**: Large language models (LLMs) often generate responses longer than necessary, increasing inference latency and computational cost. While prompting can encourage conciseness, controlling output length without compromising response completeness remains challenging. Aggressive early stopping can reduce generation costs but risks terminating responses prematurely. We introduce MinimalLM, a training-free framework for reliable verbosity control that combines uncertainty-aware, length-dependent stop-token adjustment with structural completeness guardrails. MinimalLM separates the decision of when to stop from whether stopping is structurally appropriate, coordinating response-level instructions, decoding-time intervention, and completion checks without modifying model parameters or requiring auxiliary models. We evaluate MinimalLM across three open-source LLMs and 330 questions, producing 5,940 responses across baseline and ablation configurations. MinimalLM reduces generated tokens by 47.6%, accelerates individual responses by 19-68%, and eliminates structurally truncated outputs in the primary evaluation. Despite substantial token savings, it achieves an average quality score of 8.04/10, compared with 8.42/10 for unmodified models and 6.46/10 for length-matched truncation. Evaluations across diverse subject domains (such as healthcare, programming, law, and science) demonstrate robust efficiency gains, while ablations establish the complementary roles of decoding control and completeness constraints. Our findings highlight that efficient generation requires more than producing fewer tokens: it requires controlling when generation ends while preserving response integrity. MinimalLM offers a lightweight, training-free approach toward this goal. 

---
# Evaluation Contracts and Evaluation Leverage: Sliced-Gap Readings in Structured Prediction Can Move Without Model Change 

**Authors**: Hao Lu, Weixing Zhu, Ruiyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.04099)  

**Abstract**: NLP evaluation reports routinely cite sliced performance differences, most prominently the intra-inter gap in document-level relation extraction, as capability evidence. We show that such gap readings are joint properties of models and evaluation contracts: scoring function, gold version, threshold, output format, and prompt language. With predictions frozen, a method ordering reverses under a gold-standard exchange (+0.0363, CI excluding zero; tie-level after), movement localizing where gold re-partitions: renames are gauge transformations (Delta=0; re-annotation +0.862). We formalize this as an evaluation leverage account centered on the rank-flip condition. Constructively, the audited evaluation reading (AER) is a contract-marginalized, tie-declared, seed-prescribed, gate-filtered procedure whose tie declarations cover every contract-induced flip on the declared WMT19 grids (14/14; held-out 73% vs. union-of-ties 89%), matching the best single metric contract at epsilon=0.10 equivalence. Evidence spans DocRE, HacRED, a UD probe, and released gold revisions; an alternative-reference-set replication extends it beyond our testbeds. Failed predictions retained as boundaries; a 5% violation gate flags unverifiable settings and bounds survivor-selection bias. We release EvalLever, a preregistered workflow with an amendment trail; on LLM-as-judge grids, the audited ensemble stabilizes every unstable pair (13-15 to zero; generic ensemble shrinkage: the preregistered correlation criterion failed, the companion stabilization criterion passed). 

---
# Same Rule, Different Verdicts: Rubric Language and Clause Structure in LLM Judges for Korean Support Dialogues 

**Authors**: Sangdo Han, Moonsoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.04091)  

**Abstract**: Services that operate in Korean often write their evaluation rubrics in Korean and hand them to an LLM judge. We test whether a judge's verdict depends on the language of the rubric and on how a rubric item is phrased. We write 12 criteria for support dialogues in four clause structures (affirmative, negated, conditional, exception) such that two pairs of structures have the same meaning, and we give each item in Korean and in English. Six low-cost models from three providers, plus one reference model, judge 60 translated ESConv dialogues and 144 constructed boundary cases under every combination (51,744 requests across the seven models). The analysis plan was fixed and committed before data collection. Cross-provider agreement is lower with Korean rubrics than with English rubrics (Cohen's kappa 0.687 vs. 0.745; difference -0.058, 95% CI [-0.072,-0.045]). Disagreement concentrates in negated, conditional, and exception items (odds ratio 5.6 relative to affirmative items). A single model changes its verdict between two items with the same meaning in 11-15% of cases. Two hypotheses are not supported: models from the same provider do not agree more than models from different providers, and adding paired examples does not reduce the language gap. We measure consistency only; no human labels are used, and all materials were drafted by an LLM. 

---
# Swapping the Judge Changes the Ranking: A Preregistered Study of Cross-Provider Replication in LLM-Judged System Comparisons 

**Authors**: Sangdo Han, Moonsoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.04090)  

**Abstract**: Practitioners who compare systems with an LLM judge and have no human labels often check a conclusion by repeating it with a judge from another provider. We test how far that check can be trusted. In a preregistered study (with a disclosed post-registration reduction of the judge set), five judge models from three providers re-judged 4,000 Chatbot Arena battles (nine systems, 33 system pairs) that carry human votes, each in both response orders. Agreement with the human vote was low at the verdict level (median Cohen's kappa = 0.22), and system rankings did not survive a change of judge: Kendall's tau with the human ranking ranged from 0.17 to 0.89 (median 0.67), so the preregistered hypothesis that rankings remain stable was not supported. Conclusions replicated by two low-cost judges from different providers matched the human conclusion more often than single-judge conclusions (+0.10 precision), but the gain came from filtering out one poorly aligned judge, and replication involving that judge left a conclusion for only 35% of comparisons. Judges that agreed with humans reached a stable conclusion with no more verdicts than human votes required; the others failed to converge on 26-70% of system pairs at the sample sizes tested (up to 100 verdicts per pair). Disagreements with the human vote co-occurred across judges far more often than independence predicts, although a single human vote per battle cannot separate shared judge bias from voter noise. 

---
# Language Model Augmented Decision Transformer for Multi-Task Offline Reinforcement Learning 

**Authors**: Xinbo Zhao, Yuhang Liu, Xin Zhang, Lei Le, Zhaohan Xi, Yu Yang, Yingxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.04031)  

**Abstract**: Offline multi-task reinforcement learning (MTRL) aims to learn a unified policy from heterogeneous offline datasets that generalizes across tasks. However, existing offline MTRL methods often struggle with two fundamental issues: severe per-task data scarcity and limited cross-task generalization. In this paper, we propose Language-Model Augmented Decision Transformer (LaMA-DT), a unified sequence-modeling framework that leverages a single large language model (LLM) for both trajectory augmentation and multi-task policy learning. LaMA-DT uses natural-language task descriptions to guide task-aligned trajectory augmentation through a set of supervised generation and alignment objectives, and incorporates a scoring-and-filtering mechanism to retain high-quality augmented trajectories. The same LLM is then repurposed as a language-conditioned Decision Transformer and fine-tuned on a mixture of real and augmented data. Experiments on Meta-World and MuJoCo multi-task benchmarks demonstrate that LaMA-DT consistently outperforms competitive baselines, particularly in low-data regimes, showing that language-guided data augmentation substantially improves cross-task generalization. 

---
# Dory1.0-4B: A Looped Normalized-GPT 

**Authors**: Maxim Fishman, Simeng Sun, Shantanu Acharya, Faisal Ladhak, Brian Chmiel, Santiago Akle Serano, Ron Banner, Boris Ginsburg, Krishna C. Puvvada  

**Link**: [PDF](https://arxiv.org/pdf/2611.03998)  

**Abstract**: We introduce Dory1.0-4B, a 4-billion-parameter looped (latent-recurrent) language model with elastic depth: a single checkpoint serves different inference compute budgets. Dory1.0-4B is built on a normalized Transformer (nGPT) backbone with interleaved sliding-window and full-attention layers, organized into an input block, a shared recurrent block, and an output block. At inference, the recurrent block can be applied a variable number of times. We pre-trained Dory1.0-4B on 6 trillion tokens. The first 4.5 trillion tokens were trained in NVFP4, and additional pre-training then extended the context length to 256K tokens. Next, we post-trained the model using Supervised Fine-Tuning (SFT) and Reinforcement Learning with Verifiable Rewards (RLVR), with a difficulty-aware loop curriculum that routes harder prompts to deeper loops. Increasing the loop count improves accuracy, with the largest gains on coding, agentic, and long-context benchmarks. With two or more loops, Dory1.0-4B outperforms larger models such as Qwen3.5-9B, Gemma 4 12B, and Gemma 4 26B-A4B on average across 12 benchmarks. Looping also makes Dory1.0-4B more token-efficient on agentic benchmarks: with four loops, it generates 1.3-2.5x fewer tokens than with one loop, as the extra computation is spent in latent space rather than on longer outputs. To reduce memory during looping, Dory1.0-4B caches keys and values only from the last loop, keeping its KV-cache footprint equal to that of a non-looped model at any loop count while retaining most of the accuracy gains. We open-source the BF16, MXFP8, NVFP4, and GGUF quantized checkpoints. 

---
# Whose startup does the AI back? An algorithm audit of reputation and demographic signals in LLM-assisted crowdfunding 

**Authors**: Mirza Samad Ahmed Baig, Syeda Anshrah Gillani  

**Link**: [PDF](https://arxiv.org/pdf/2611.03993)  

**Abstract**: Large language model (LLM) assistants increasingly advise backers on which crowdfunding campaigns to support. LLMs have known race and gender problems, and it is unsettled whether they reproduce, erase, or reverse the discrimination documented in entrepreneurial finance. We ask when, and through which decision process, LLM assistants use founders' gender and ethnicity. In a pre-specified choice-based conjoint experiment, seven LLMs chose among five synthetic campaigns with independently randomized quality signals and founder names, under three backer personas and a persona-free control (40,248 responses). By default, founder names barely matter: under budget-conscious and delivery-focused personas, gender and Black-name effects are statistically equivalent to zero. When the backer wants to support underrepresented creators, the models favor Black-signaled (+8.1 percentage points; positive in all seven models) and female-signaled (+5.1) founders. They do so most strongly among near-tied campaigns and for the founder who alone carries the signal, and they almost never name gender or ethnicity in their reasons. The results fit an agency account of algorithmic discrimination: an aligned assistant takes its demographic preferences from its user rather than from its training data. 

---
# FlagRot: Selective Rotation for Low-Bit KV Caches 

**Authors**: Kunming Shao, Xijie Huang, Jierun Chen, Jiaqi Shao, Yanli Wang, Jiangnan Yu, Qiwei Li, Wei Zhang, Kwang-Ting Cheng, Chi Ying Tsui  

**Link**: [PDF](https://arxiv.org/pdf/2611.03992)  

**Abstract**: Long-context inference is bound by the KV cache, which 2-bit codes could shrink about sixfold if their error stays in check. The usual remedy rotates keys before quantizing, assuming rotation always helps. However, we observe that it does not: rotation flattens a spiky piece of a key but can give a flat one a new peak, and both coexist within one key. We therefore propose FlagRot, which stores each 32-channel piece rotated or as is, whichever has the smaller calibrated attention error, and rotates the query instead when reading. We further show that how often to decide follows the quantization axis: per-channel caches such as KIVI need one decision per head, at no write cost, while per-token caches decide per key, with the choice stored in existing sign bits. At 2.5 bits per value, FlagRot lowers the twelve-model KL divergence of KIVI from 18.7 to 12.4 x 10^-3, below RotateKV on 9 of 12 models. Served as a vLLM plugin, it raises RULER at 128k tokens by +2.9 points and the LongBench v1 mean by +0.4 to +1.9 points on the three of four models where 2-bit KIVI loses accuracy. It also lowers the error of 4-bit integer and FP4 caches. 

---
# SLATE: A Large-Scale Benchmark for Long-Horizon API Selection in E-commerce Tool Use 

**Authors**: Rongzhe Wei, Ge Shi, Min Cheng, Sarthak Ghosh, Vaibhav Gorde, Leman Akoglu, Pan Li, Na Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.03982)  

**Abstract**: Enterprises increasingly want LLM agents to automate standard operating procedures over their software stacks -- mapping each step of a procedure to the right call among thousands of internal APIs, without hand-authored step-to-API mappings. The core difficulty is API selection at scale: choosing and composing the correct tools over a long horizon, judged only by whether the overall task succeeds. Current tool-use benchmarks do not test this regime -- they expose only a handful of tools, evaluate single-step invocation, or grade multi-step behavior with subjective LLM-as-a-judge scores that penalize valid alternative trajectories. We introduce SLATE (Synthetic Large-scale API Toolkit for E-commerce), a large-scale benchmark that tests large-toolset, long-horizon API selection in a realistic enterprise domain. SLATE couples (i) queries decomposed into hierarchical, tool-grounded procedures over a ~1,000-tool e-commerce library, (ii) a deterministic, context-aware simulator that returns recorded outputs for matching calls and default responses otherwise, and (iii) an end-to-end task-success metric -- a task succeeds when it reaches every reference outcome -- that credits diverse-but-valid trajectories while remaining fully automatic and reproducible. Experiments with six open decision models and the language models they are built on, from 3B to 12B parameters, show that even the best model completes only 5.0% of the tasks, establishing SLATE as a challenging testbed for reliable enterprise tool use. We release SLATE, its simulator, and the evaluation harness at this https URL 

---
# Balancing Chain-of-Thought and Answer Supervision for Small Language Models on Medical Question Answering 

**Authors**: Le Zhou, Xiao Luo, Chinmay Chandra, Zhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.03948)  

**Abstract**: Small language models (<=9B parameters) can support local medical question answering (QA) while reducing privacy and latency concerns, but effective fine-tuning remains challenging. This paper compares two chain-of-thought (CoT) supervised fine-tuning methods: Uniform-across-Sequence (UaS), which applies equal loss weighting to the full CoT-and-answer sequence, and Segment-weighted (SwS), which separately weights CoT and answer losses. We evaluate six general- and medical-domain models on multiple-choice QA (MedQA and EMS-MCQA) and free-text QA (Med-O1). Results show that neither method is universally best: on multiple-choice QA (MedQA and EMS-MCQA), SwS generally performs better, with general-purpose models benefiting most; notably, Qwen3.5-4B improves by up to 44.8% over its zero-shot baseline on MedQA. In contrast, medical-domain models show smaller or even negative gains on MCQs. On free-text QA (Med-O1), UaS performs best for nearly all models and provides consistent improvements across both model types, with the 3B Llama-3.2 model approaching the performance of much larger 7B medical-domain models. We also find that excessively long CoT generation is associated with lower performance in some models on either multiple-choice or free-text QA tasks, suggesting that limiting the CoT length during training may benefit selected models. Overall, the results show that CoT loss weighting should be matched to the task format and the knowledge already encoded in the base model, rather than using a single fine-tuning strategy for all clinical QA tasks. 

---
# EvoGroup: A Self-Evolving Harness for Group Agents in Shared Contexts 

**Authors**: Xinyi Xu, Xiaoyang Li, Yucheng Feng, Wenqing Chen, Youliang Yuan, Wei Shi, Rongxiang Weng, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.03868)  

**Abstract**: With the rise of Group Agent such as Claude Tag, agents increasingly serve as digital workers for long-term team collaboration. These agents operate over shared contexts that continuously accumulate through multi-user interactions. However, existing agent memory systems are mainly designed for individual users and struggle to capture group-level knowledge in shared contexts, including group relationships and collaboratively formed knowledge. Manual memory strategies further limit adaptation to diverse and evolving shared contexts. To this end, we propose \textbf{EvoGroup (EvoG)}, a self-evolving harness for group agents that starts from a minimal six-tool cold start and adapts through interaction with shared context. EvoG uses confidence and task outcomes to select informative trajectories, while also eliciting self-reflections on uncertainty in low-confidence cases. The analysis agent then examines selected trajectories through adaptive progressive disclosure, followed by bucketed analysis to synthesize high-level findings that guide the evolve agent in revising the Harness. Empirically, pass@1 on held-out split of EverMemBench improves from 71.07\% to 89.58\%, surpassing both self-evolving memory and human-designed harnesses, including Claude Code (85.83\%). To evaluate generalization, we freeze the evolved harness and evaluate it on the unseen GroupMemBench, where, with GPT-5.5, it demonstrates effective generalization and outperforms both memory baselines and human-designed harnesses. 

---
# Why Pay for Long Reasoning When Rewrite Do Trick? Causal Effects of Reasoning History Compaction on Future Generation and Agent Cost 

**Authors**: Anton Hristov  

**Link**: [PDF](https://arxiv.org/pdf/2611.03852)  

**Abstract**: Reasoning substantially improves language models' ability to solve difficult problems, but long reasoning traces make these gains expensive. Efficient reasoning should express useful analysis in fewer tokens by reducing repetition and text that adds little value to solving the problem. Progress in proprietary models motivates this goal, yet their hidden scratchpads limit direct study of how it is achieved. We investigate whether open-weight models can generate less verbose reasoning by changing the form of their own earlier thoughts, without retraining. Our intervention rewrites prior reasoning into compact state records before pending tool results are revealed, then compares continuations from original and compact histories. Across the tested models, accepted compact histories usually shorten future reasoning; a coding case study also shows less repeated verification after a shared candidate solution. In software-agent deployments, compaction can reduce total spending, although task quality and rewrite-inclusive time depend on the target and compressor. These findings identify reasoning history as one way to reduce future reasoning-token use and motivate model-specific compaction policies evaluated for both efficiency and retained capability. 

---
# A Conditional Mutual Information Analysis of Response Variety in Language Models 

**Authors**: R. I. Cukier  

**Link**: [PDF](https://arxiv.org/pdf/2611.03836)  

**Abstract**: Large language models provide a variety of responses at inference (generation) that conventionally depend on a temperature parameter. Using a conditional mutual information approach, a quantitative, entropy based analysis of response variety is given. Three limiting behaviors are defined: arbiter, independent (conditionally) and neutral (deterministic) that span no to maximal mutual information. We give examples of each behavior and suggest their application to: temperature analysis, language analysis, prompt compression, and knowledge graphs. 

---
# Rethinking Next-Visit Diagnosis Prediction Beyond Aggregate Metrics: Retrieval--Memory Multi-Agent Adjudication 

**Authors**: Heejoon Koo  

**Link**: [PDF](https://arxiv.org/pdf/2611.03826)  

**Abstract**: Next-visit diagnosis prediction forecasts the diagnoses documented at a patient's next encounter, turning retrospective electronic health records into prospective clinical prediction. However, conventional aggregate evaluation obscures its multifaceted nature: distributionally, models must handle variable-size diagnosis sets and long-tailed labels; temporally, they must distinguish persistent, newly recorded, and recurrent diagnoses; and both vary with observed history depth. We introduce an evaluation framework along these dimensions and apply it to rule-based heuristics, LLM-based prompting, retrieval, memory, and multi-agent approaches. This reveals that repeating the last visit outperforms most LLM-based approaches overall despite predicting no new or recurrent diagnoses, and that retrieval and memory exhibit distinct profiles whose relative strengths vary across temporal transitions and backbones. Motivated by this, we propose LONGINUS, a retrieval-memory multi-agent adjudication framework combining evidence retrieved from patient records with structured memory of the patient trajectory. Using cross-view agreement as a reliability heuristic, it preserves shared predictions and adjudicates disagreements without revealing their source. Across MIMIC-III and MIMIC-IV with four LLM backbones, LONGINUS achieves the best transition-specific performance among LLM-based methods in nearly all settings while remaining competitive with them on aggregate metrics. 

---
# Aligning Language Models with General Preferences: An Empirical Study of Direct Nash Optimization 

**Authors**: Austin Tsai, Saurav Nanda, Ganapathy Parthasarathy, Bhuvnesh Kumar, Parivesh Choudhary  

**Link**: [PDF](https://arxiv.org/pdf/2611.05373)  

**Abstract**: Direct Nash Optimization (DNO) aligns a language model to a general preference function by approximating the Nash equilibrium of a two-player preference game. Its practical variant, DNO-Prct, iterates on-policy sampling, LLM-based rating of the samples together with a teacher's gold response, large-margin preference-pair selection and a DPO-style contrastive update. We re-implement DNO-Prct independently and study it on two open-weight instruct models, Llama 3.1 8B Instruct and Mistral 7B Instruct v0.1, with GPT-4o or a larger open-weight model of the same family as the annotator. On AlpacaEval 2.0 with GPT-4o as judge, three iterations raise the length-controlled (LC) win rate against GPT-4o from 31.7% to 45.3% for Llama and from 6.8% to 18.4% for Mistral, and the tuned 8B model scores above Llama 3.1 70B Instruct on every reported win rate. With same-family open-weight annotators, the runs retain 62% (Llama) and 82% (Mistral) of the LC gain achieved with GPT-4o as annotator, so the rating step does not require a proprietary model. In a separate ablation configuration, variants that remove the gold responses, remove the data look-back, or accept only teacher-preferred pairs score 6 to 10 LC points below the full procedure, whereas varying the number of on-policy samples per prompt (1 to 7) moves the LC win rate by at most 2.1 points, and a margin sweep (1 to 5) run in its own pipeline by at most 2.4 points. We also report the cost of the pipeline and where the practical algorithm departs from the theory it approximates. All results are single runs, and GPT-4o is both the teacher and the evaluation judge; we discuss what this implies for the reported gains. 

---
# Reading Why: Born-Measured Attention with Exact Word Evidence on Light and Qubits 

**Authors**: Teerawat Tongloy, Jiraporn Pongsopa, Rangsan Jomtarak  

**Link**: [PDF](https://arxiv.org/pdf/2611.05355)  

**Abstract**: When a language model answers, we can usually only estimate why. We build a transformer in which every step is a Born measurement - attention $|\langle q|k\rangle|^4$ normalised over each row, a measured product in place of the feed-forward nonlinearity, and a Born readout. With its denominators held at their values for the input, each class probability is a polynomial in the word amplitudes, so its integrated-gradient evidence is computed by a finite quadrature with no quadrature error: the words' contributions sum to the class probability above its no-information share, to $10^{-11}$ or better. On e-SNLI natural language inference, with about 98k parameters outside the embedding, the model reaches $76.98 \pm 0.17$% test accuracy, level with a softmax transformer of the same size ($76.62 \pm 0.36$%). Deleting the word its evidence ranks first changes the answer in 45% of test pairs, out of the 53% any single deletion can change, against 32% for the softmax model explained the same way and 40% by its attention x gradient. Attention can look more like the words annotators highlighted, yet accounts for the answer less well: deleting the word it ranks first changes the Born Transformer's answer in 32% of pairs. The same weights compile exactly to optoelectronic optics and to qubits, and a coherent attention layer becomes one circuit equal to the model (852 CNOTs on 23 qubits). On IBM's ibm_kingston, reading only the attention overlaps by Hadamard tests, the device gave the model's answers on two short test pairs and kept or changed them under deletion as the model does; the evidence itself did not yet survive the device, and we work out the shots and calibration that measuring it on hardware would need. 

---
# MedPhysBench: A Benchmark for Evaluating Language Models on Medical Physics Calculations and Error Detection 

**Authors**: Omar Nusrat  

**Link**: [PDF](https://arxiv.org/pdf/2611.05340)  

**Abstract**: Medical physicists turn a prescription into a delivered dose and check one another's calculations, because one error in a beam calibration or a monitor-unit (MU) calculation can affect every patient treated with it. MedPhysBench has ten seeded generators of calculation items in radiation therapy, diagnostic imaging, nuclear medicine and radiation protection, up to clinical tasks such as a TG-51 calibration worksheet. Reference answers come from code implementing the governing protocols, and a grader converts units and applies tolerance bands. From the same inputs, separate checking items test error detection, showing a colleague's worksheet or only the colleague's result, half correct and half with one planted error. On a held-out test split of 1,200 items (half for two models), 12 models scored 33.3 to 100.0% on calculation. Of the nine answers graded incorrect for the four best (99.3-100.0%), five contained model errors and four followed NCRP Report 151 on points our items misstate or leave open. Below them, most models did worst on level-3 clinical tasks, and Claude Sonnet 5.5 omitted the inverse-square correction in 19 of 20 isocentric MU items. Five models detected at least 96.7% of mostly large planted errors with at most 4% false alarms; for four of the other seven, checking only the result raised false alarms significantly. In an exploratory targeted test with 20 public cases of an omitted inverse-square correction, the three best models tested caught it in every worksheet, and three weaker models accepted 16 to 18 of 20 planted worksheets, one of them while applying the correction in most of its own calculations. Claude Opus 5.5 accepted 6 of 20 wrong results, although it applied the correction in all 20 of its own. We report the effect of two problems in our shielding items, found after the runs, and release the code, per-item outcomes and targeted-test responses. 

---
# LoopVQ: Sub 1-bit KV Cache Quantization for Looped Transformers 

**Authors**: Minsoo Cheong, Sungjoo Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2611.05304)  

**Abstract**: Looped transformers are a parameter-efficient alternative to standard transformers for language modeling. By reapplying the same layers across multiple loop steps, recent looped language models match the performance of non-looped models several times their size. While this architecture effectively reduces parameter memory, each loop step still stores its own keys and values, so the key-value (KV) cache grows linearly with the number of loop steps and becomes the main memory bottleneck. We observe that the KV cache of looped models is highly redundant along the loop axis and that the loop steps differ substantially in their sensitivity to quantization. Based on these observations, we propose $\textbf{LoopVQ}$, a KV cache vector quantization method for looped transformers that codes channels of multiple loop steps jointly, and fine-tunes the codebooks to match the output of the full-precision model. At 0.5 bits per element, LoopVQ outperforms existing KV cache vector quantization methods for standard LLMs on general, long chain-of-thought reasoning, and long-context retrieval tasks with Ouro and Huginn models. Its design is orthogonal to these methods, and combined with TaSQ, it retains much of the BF16 accuracy even at 0.25 bits per element. We also implement LoopVQ in SGLang, where it nearly halves the decode latency at a 32K-token context and raises the peak serving throughput by up to $4.3\times$ and $6.1\times$ on the 1.4B and 2.6B Ouro models compared to the BF16 baseline. 

---
# What Does the Model See? Representation Validity in Multimodal LLM-Based Video Analysis 

**Authors**: Christian Pipal, Eva-Maria Vogel, Morgan Wack  

**Link**: [PDF](https://arxiv.org/pdf/2611.05181)  

**Abstract**: Multimodal large language models (MLLMs) allow social scientists to analyze video at scale, but researchers must decide which parts of a video to provide as input. We use the term representation validity to examine how text, audio, video, and combined inputs affect measurement and subsequent analysis. We compare input representations across six open-weight MLLMs and three human-annotated video datasets. No representation consistently performs best. On seven news-related constructs, native-video performance varies little across models (macro-F1 = .70-.77), while extracted-text performance ranges from .52 to .80. Three models perform better with text and three with video. Combining inputs rarely improves performance. Higher classification accuracy also does not reliably produce estimates closer to those based on human codes. The highest-F1 representation best reproduces human-coded regression coefficients in 17 of 42 comparisons. Even when each model uses its highest-F1 deployable representation, 45% of sample-prevalence estimates differ from the human-coded benchmark by more than ten percentage points. Input representation therefore requires validation for both the construct being measured and the quantity estimated from the resulting labels. 

---
# Heads That Write, Not Just Point: Image Retrieval Heads in Vision-Language Models 

**Authors**: Junsung Park, Sangwon Yu, Donghun Kang, Yeongtak Oh, Jaewon Jeong, Gyeongtae Yoo, Han Cheol Moon, Jungbeom Lee, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2611.04903)  

**Abstract**: Large vision-language models (LVLMs) can process interleaved text and multiple images in a single context, but how they internally identify the image relevant to a question remains poorly understood. We study this grounding step as in-context image retrieval and introduce ICIR-MCQ, a controlled probe for head-level analysis of image retrieval. Using each head's attention mass over image-token spans, we first identify sparse pointer heads that reliably point to the target image. These heads transfer beyond the probe: a single pointer head is competitive with external vision-language retrievers on a naturalistic multi-image retrieval benchmark, showing that LVLMs contain a strong internal image retrieval signal. Attention alone, however, shows where a head reads from, not what its output passes on to later layers, so we introduce a complementary output-side analysis. For each head, we train a lightweight image retriever on its value-weighted output and use it to identify writer heads, whose outputs contain enough information to identify the target image. Surprisingly, the pointer and writer criteria select substantially different head sets, with only 35 to 47 heads overlapping among the top-100 heads under each criterion. Targeted set-partition ablations show that the heads most important for standard multi-image inference are concentrated in this pointer-writer intersection, while heads selected by only one criterion show less consistent separation from random-head controls. The most consistent causal effects arise from heads selected by both the attention-based pointer and output-side writer criteria. Code and data: this https URL. 

---
# Feeling of Knowing in Large Language Models 

**Authors**: Zichuan Fu, Xian Wu, Jingtong Gao, Wenlin Zhang, Jiaxuan Li, Binhao Wang, Yimin Deng, Guojing Li, Xiaopeng Li, Derong Xu, Yefeng Zheng, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.04834)  

**Abstract**: Post-training updates change what large language models (LLMs) can answer, but may leave their answerability judgments misaligned. We study this knowledge-judgment desynchronization through functional feeling of knowing (FoK), which supports selective generation, retrieval triggering, and model routing without implying subjective experience. We propose two trainable channels: a knowledge channel that changes answerability and a FoK channel that judges the current knowledge state. Three stages train them: on-policy supervised fine-tuning (OP-SFT) updates knowledge from corrected on-policy answers; knowledge-delta FoK training (K$\Delta$-FT) adapts judgments across intermediate knowledge states; and FoK-guided policy optimization (FGPO) trains the subsequent answer or explanation. Factual-QA experiments, cross-dataset evaluation, and three sequential updates show that explicit FoK alignment improves judgment quality on updated knowledge. MMLU and ARC-Easy separately assess capability preservation. 

---
# Think Less, Locally: Local Preference Learning for Reasoning Compression 

**Authors**: Muyao Yuan, Liyan Xu, Weizhan Zhang, Muyan Jiao, Jiangnan Li, Jinchao Zhang, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.04814)  

**Abstract**: Large reasoning models achieve strong performance on challenging tasks, but their long chain-of-thought often incurs substantial inference costs by continuing to deliberate after sufficient information has been obtained. Existing reasoning compression methods primarily operate on reasoning trajectories, paying less attention to the local decision of whether further reasoning is necessary given the current state. We propose LoCo, a local preference learning method for reasoning compression. LoCo detects candidate overthinking states from the model's own traces, freezes the corresponding reasoning prefixes, regenerates shorter verified continuations from the same states, and trains on the resulting state-conditioned preference pairs with DPO. This formulation teaches the model when to continue reasoning and when to conclude, rather than simply encouraging shorter trajectories overall. We construct training data entirely from self-generated mathematical traces and evaluate LoCo on five mathematical and three code generation benchmarks. LoCo reduces completion length by 41.7\%--45.4\% on mathematics and 23.4\%--36.3\% on code, while maintaining or improving accuracy and achieving the best AUC across all code benchmarks. LoCo consistently improves the accuracy-efficiency trade-off on mathematics and transfers effectively to code generation despite math-only training, demonstrating the cross-domain generality of local preference learning for reasoning compression. The code and model will be publicly released at this https URL. 

---
# Where Is the Truth When AI Agents Disagree? Dependence-Aware Belief Propagation for Unsupervised Truth Discovery (Extended Version) 

**Authors**: Laure Berti-Equille  

**Link**: [PDF](https://arxiv.org/pdf/2611.04805)  

**Abstract**: LLM agents often give conflicting answers to the same question, and many deployed agents are variants of one base model, so their errors are correlated. We study truth discovery among such agents: inferring the true answer and each agent's reliability without labels. Classical truth-discovery and crowdsourcing estimators reward an agreeing bloc and can fall below majority vote. We propose BP-TD, which adds per-agent decorrelation weights to mean-field variational inference on a one-coin Dawid-Skene model, so that agents sharing a base model count together as about one vote. Its first iteration is exactly a weighted vote; later iterations estimate agent reliability. With 13 local agents on MMLU, BP-TD beats majority vote without any labels (0.716 vs. 0.624 on 160 questions; 0.707 vs. 0.674 on 10,000) and ranks agents by their true accuracy, which classical estimators invert. The gain depends on the pool: over 2,852 sub-pools of four benchmarks, decorrelation helps when the largest correlated bloc is weaker than the other agents and hurts when it is the strongest, and we found no label-free statistic that tells the two cases apart. Labels change the picture quickly: with 20 labelled questions, selecting the single best agent is more accurate than every aggregator we tested on MMLU, ARC and TruthfulQA. 

---
# Self-Privileged Fine-Tuning 

**Authors**: Yangrong Hu, Xueyu Zhou, Yue Wu, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.04632)  

**Abstract**: Supervised fine-tuning (SFT) is a key component of language model post-training, enabling effective task adaptation and serving as an important foundation for subsequent reinforcement learning. Recent methods seek to improve SFT by introducing token-level weights into the training objective, where these weights are typically derived from a model's output distribution at each token position. To better understand how such weighting schemes affect SFT, we analyze their optimization objectives and theoretically characterize the target distributions they induce. Our analysis shows that dynamic weighting methods, such as Dynamic Fine-Tuning (DFT), induce a heavily sharpened distribution, whereas static weighting methods fit a target distribution proportional to the product of the data distribution and a frozen prior. These findings suggest that static weighting provides a more stable training target, while its effectiveness critically depends on the quality of the prior distribution. Building on this observation, we propose Self-Privileged Fine-Tuning (SPFT), which uses a frozen copy of the initial model conditioned on privileged information (such as ground truth for mathematical questions) as a prior to provide guidance for token-level weighting during fine-tuning. Through a Bayesian decomposition, we further show that SPFT reweights the standard static prior according to how informative each target token is for predicting the ground-truth outcome, thereby assigning larger weights to more predictive tokens. Experiments on mathematical reasoning and code generation show that SPFT improves over standard SFT and other baselines across the main settings. 

---
# $τ$-Rec-Books: A Disclosure-Controlled Benchmark for Requirement Elicitation in Agentic Recommender Systems 

**Authors**: Bharath Sivaram Narasimhan, Karthik R. Narasimhan  

**Link**: [PDF](https://arxiv.org/pdf/2611.04459)  

**Abstract**: While conversational recommendation systems provide a more natural user interface, they face the challenge of progressively discovering a user's needs before they can recommend an item that satisfies them. To evaluate the ability of AI agents at this elicitation process, we introduce $\tau$-Rec-Books, a benchmark that holds each user request fixed while varying only when the user reveals their requirements: all at the start (volunteer), one at the start and the rest when asked (mixed), or some never (hidden). A hidden requirement can be found only by proposing books and inferring it from the user's accept or reject verdicts. We generate 450 tasks over a catalog of 20,000 books from Wikidata, develop a rule-based simulated user to answer the agent, and score recommendations against catalog metadata without an LLM judge. Across four agents and two search interfaces, success rates fall for every agent and interface, on average from 0.975 (volunteer) to 0.756 (mixed) and 0.283 (hidden). Mixed tasks fail mainly because agents do not ask, and when told to ask, every agent reaches at least 0.931. On hidden tasks, most failures come from agents recommending a book they never proposed to the user. When told to propose first, agents only match a simple baseline that proposes popular books, and stay well below one that uses rejection to rule out possible requirements. We release the catalog pipeline, task generator, simulated user, validators, reference agents, and all agent traces at this https URL. 

---
# Large Language Model-Assisted Adaptive Heuristic Search for Agile Earth Observation Satellite Scheduling 

**Authors**: He Wang, Yeye Liu, Ruijie Li, Xuemei Jiang, Junyu Wu, Lining Liu Zhuoqi Man, Tao Li, Yanjie Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.04458)  

**Abstract**: Agile Earth observation satellites must coordinate imaging tasks within limited observation windows, while accounting for attitude maneuvering and onboard energy constraints. Competition among observation opportunities makes scheduling performance sensitive to the heuristic rules used for task selection. This letter proposes a large language model (LLM)-assisted adaptive heuristic search method for agile Earth observation satellite scheduling. The LLM generates an initial heuristic rule according to the characteristics of the scheduling instance. During the subsequent ant colony search, search performance and scheduling information are fed back to the LLM to refine the heuristic rule. The proposed modifications are evaluated through paired solution constructions before being incorporated into the search process. A unified scheduling procedure determines task selection, satellite assignment, and observation timing under the operational constraints. Experiments on six scheduling scenarios involving 50--200 imaging tasks demonstrate that the proposed method achieves 4.3\%--9.6\% higher mean observation benefit than the strongest baseline. 

---
# Near-Zero Token Embeddings Silently Stall Training in Gated Sequence Models 

**Authors**: Xin Liu, Boyuan Sun, Yuxin Wang, Haoran Song, Mingjian Zhao, Mengyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.04449)  

**Abstract**: Among 291 surveyed open-weight language model repositories (280 distinct embedding tables), 52.6% of untied gated and hybrid checkpoints (20/38) contain reachable input embeddings with norms below $10^{-6}$, compared with 0% of weight-tied models. Such near-zero rows can arise during pretraining when unseen rare or template tokens receive zero input gradients while AdamW weight decay shrinks untied embeddings. In tested Transformer configurations, non-initial positions or a BOS token prevent this failure mode. In gated sequence models (Mamba, Gated DeltaNet, xLSTM), a near-zero token can amplify input-embedding gradients by more than 30 orders of magnitude (up to 35). At zero input, current-token readout factors (SiLU gates or linear queries) vanish, suppressing historical-state readout. Residual RMS values can then fall below the square root of the RMSNorm floor across consecutive layers, causing Pre-RMSNorm to amplify backward gradients by up to $10^3$ per layer. When a gradient entry exceeds $1.84 \times 10^{19}$, FP32 global-norm clipping in PyTorch, Hugging Face Trainer, and FSDP2 can overflow to infinity and zero the current batch's gradients. In Telugu continual pretraining of Nemotron-3-Nano-4B, this discards updates on approximately 20% of steps and leaves triggered rows frozen. We propose two remedies. For released checkpoints, GroupClip decouples input-embedding clipping using a 64-bit or rescaled norm; its rescaled implementation eliminates update loss and heals triggered rows in one step with negligible measured end-to-end overhead. For pretraining, GateFloor ($\mathrm{SiLU}(z) + \delta$, $\delta \in [0.01, 0.1]$) keeps the gate positive at zero input, reducing near-zero gradients by 28 orders of magnitude in experiments up to 370M parameters, with perplexity comparable to the baseline. 

---
# Sketchpad: certified exact arithmetic with a small vision-language model 

**Authors**: Mostafa 'Neo' Mohsenvand  

**Link**: [PDF](https://arxiv.org/pdf/2611.04434)  

**Abstract**: Language models are accurate on most long-arithmetic problems, but their accuracy is estimated on samples and carries no guarantee for an unseen input; tool use is exact but moves the computation outside the network. We show that exactness at all input lengths becomes a finite, checkable property of a network when each of its steps reads only a bounded view of an external working memory. In Sketchpad, Qwen3.5-0.8B with a low-rank adapter executes arithmetic procedures on a rendered sheet. Each forward pass reads a three-cell view and emits one step: a digit, a carry or borrow, and the next operation. A fixed runtime moves the view and writes no digit of a result. We prove in Lean that the model is then exact on all inputs of any length if and only if it is exact on the finite set of views the runtime presents, that no finite test suffices when steps also read earlier steps, and that the guarantee tolerates bounded numerical perturbation and extends to compositions of procedures. All 2,446 views pass on the deployment device, a laptop GPU running the model in a web browser. The certified model performs signed addition and subtraction, long multiplication and division, square roots and greatest common divisors at arbitrary length. It solved 200 of 200 pre-registered problems in the browser, and 220 of 220 sums of up to 1,000 digits and products of up to 20 digits, of which GPT-6.1 Sol, the strongest of ten other models, solved 217. A symbolic runtime built on these procedures solves, expands and factors polynomials with every number computed by the model, and a multiplicative gate on the adapter recovers the base model exactly when the sheet is not in use. 

---
# Voluntary Servitude: Weaponized Safety Warning Mechanisms for Controlling AI Agents 

**Authors**: Ilan Zendel, Matan Levy, Stav Cohen, Avi Mendelson, Amit LeVi  

**Link**: [PDF](https://arxiv.org/pdf/2611.04393)  

**Abstract**: LLM agents increasingly act on information retrieved from public sources they treat as credible, which turns a manipulated recommendation into a transaction, a dependency, or a decision with physical consequences. The security literature models this threat as instruction injection, in which an adversary writes a command into retrieved content and deployed defenses learn to recognize command syntax. We show that an attacker can instead steer an agent through the agent's own harm-avoidance. A single fabricated hazard, written as an ordinary review, leads the agent to disqualify the legitimate options and select the attacker's target by its own reasoning, with no command for a detector to recognize. We call this voluntary servitude. These payloads steer agents about as often as instruction injection (60.4% vs. 66.0%), and deployed injection detectors score them as benign, so behind a filter they are 1.7-4.4x more effective. We measure this at the generation stage across six models and ten consumer-service domains, for three attack families, and show that it transfers to production browser agents, outperforming instruction injection, and to a retrieval pipeline (1.0-57.0% end-to-end success, depending on the retriever). We then show that a corroboration requirement added to the agent's instructions removes roughly 95% of the exposure against a non-adaptive attacker without retraining. With the requirement in place, the agent still reads the planted hazard and no longer acts on it. 

---
# Text-to-SQL Models Make Choices Without Saying So 

**Authors**: Yichen Xu, Shipei Lin, Chenhao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.04318)  

**Abstract**: A text-to-SQL answer can run, return a plausible number and still compute something other than what the analyst meant. This can happen when the request leaves a calculation choice open, such as whether to count rows or distinct entities, and the model settles it without saying so. Standard benchmarks cannot detect this, because they score each question against a single reference interpretation. We introduce SpecBridge, which tests such a choice by varying it alone: it poses three otherwise identical requests, one leaving the choice open and one per interpretation, and scores each answer by running it. The databases are real, and the choices are ones that made models fail on existing benchmarks. On 204 hand-validated tests, four language models usually compute each interpretation when asked for it, yet for the open request return both in only 15-58% of the tests where they can, often without mentioning the choice. The gap lies in reporting, not in computing. Benchmarks even penalize the alternative: on BIRD questions where such a choice changes the answer, over two in five of the answers marked wrong are right under the other interpretation. Because the gap lies in reporting, a system can narrow it without a stronger model: it first decides whether a request leaves a choice open, then answers each interpretation in its own call. On our tests and on AMBROSIA, an existing benchmark of ambiguous questions, this approach raises the share of open requests that get every interpretation by 29-63 points over the single answer that production assistants give, without lowering precision on clear requests. It also needs fewer user turns than asking a clarifying question. 

---
# Coarse Semantic Blocks for Efficient Training-Free Sparse Attention 

**Authors**: Sosuke Kobayashi, Shoichiro Yamaguchi, Seiya Tokui  

**Link**: [PDF](https://arxiv.org/pdf/2611.04313)  

**Abstract**: We introduce a training-free sparse attention method that builds a prompt-specific key index to accelerate both prefill and decoding. The method clusters the keys into large semantic blocks that group tokens from noncontiguous positions. Each query position selects blocks by scoring their centroids and attends to the original keys and values within them. Queries that select the same block are processed together, so prefill stays fast even though each query chooses its own blocks. Large blocks make centroid scoring and this grouped processing cheap, so each query can attend to more tokens in the same time. Larger blocks with a larger token budget can thus be both faster and more accurate on long-context tasks than smaller blocks with a smaller budget. We evaluate two hybrid LLMs, Qwen3.5-4B and 35B-A3B, on one NVIDIA H200. Including the cost of building and using the index, whole-model wall-clock speedups over dense attention are 2.3-2.4x for prefill and 1.7-2.0x for decoding at 512K tokens, and 2.9-3.0x for prefill at 1M. The 35B-A3B model stays close to dense attention on five RULER tasks through 1M, on LongBench-v2 overall, and on MRCR through 256K. The 4B model loses accuracy mainly on multi-key retrieval, yet outperforms dense Qwen3.5-2B, whose latency is similar at 512K, on all three benchmarks. 

---
# Neologism Inoculation Prompting 

**Authors**: David Demitri Africa, Cameron Holmes  

**Link**: [PDF](https://arxiv.org/pdf/2611.04260)  

**Abstract**: Inoculation prompting is a method to reduce undesired behaviours such as reward hacking in frontier models. It works by training a model on harmful examples under a prompt that specifies the undesired behaviour, which reduces the model's tendency to exhibit it unless prompted. However, the inoculation prompt is written in plain English, so any user can re-elicit the behaviour easily. We introduce neologism inoculation prompting (NIP), which replaces the English prompt with a new token that is trained on the frozen model and removed at deployment. NIP outperforms other inoculation methods, and performs competitively with gradient routing despite being simpler. It also works on reward hacking, but fails on sycophancy and harassment. Using control tokens and analyses of activations, we identify the mechanism for this as NIP hiding the harmful behaviour behind an unreachable region of model inputs, rather than deleting it. More broadly, our work suggests that neologisms can be a powerful way to carve out and shape access to behaviours that are undesired or harmful. 

---
# LoopKV: Loop-Aware KV Cache Compression for Serving Looped Transformers 

**Authors**: Tianle Zhou, Yang Zhang, Tianyu Wu, Zhuohan Wang, William Nixon, Juncheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.04137)  

**Abstract**: Looped transformers apply a shared block several times per token to improve accuracy without adding parameters. For models such as Ouro that keep a separate key-value (KV) cache per loop, cache memory limits the batch size that amortizes repeated weight reads. Uniform-bit quantization and fixed rules per loop do not allocate precision according to each layer's sensitivity in each loop. Measurements on Ouro show that quantization sensitivity varies across (loop, layer) cells, and the sum of individual cell damages approximates the overall output damage. Sharing damage also varies by cell, with later loops generally tolerating sharing better. Skipping low-attention values to save reads produces errors that compound across loops. LoopKV uses a one-time offline profile to choose 4-bit storage, 2-bit storage or reuse of the final loop's KV for each cell. The planner minimizes reads under budgets for predicted damage and storage. During decoding, each loop reads the high 2 bits of each 4-bit value and fetches the low bits only where its own attention weights require 4-bit precision. A single fused kernel in vLLM executes the plan. Serving Ouro-1.4B on one RTX PRO 6000, the memory-optimized LoopKV-M plan stores 35% less KV than uniform 4-bit quantization (KIVI-4) on the same fused kernel and CUDA graph. It sustains 1.7-2.2x KIVI-4's goodput under a latency service-level objective with a 16K-token maximum length. None of its averages on six public benchmarks falls significantly below BF16's. 

---
# When Strong Reasoners Join the Pool: Small Union Premiums and Costly Runaway Generations in LLM Routing 

**Authors**: Teng-Ruei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.04098)  

**Abstract**: Routing benchmarks often judge a router against a hindsight oracle that counts a query as solved whenever any model's recorded answer is correct, but under stochastic decoding that oracle mixes two different objects: the clairvoyant ceiling of committing to one model per query and the premium of calling several models and keeping any correct answer. A companion study measured this premium at 3.5 percentage points (pp) for 11 small open models on MATH-500; how large it is once current reasoning models enter the pool was unknown. We measure it with a fixed-protocol, pre-committed generation on a single NVIDIA B300: all 500 MATH-500 problems, 30 seeded samples per problem, Llama-3.3-70B-Instruct and NVIDIA's Nemotron Nano 9B v2 and Nemotron Super 49B v1.5 (a NAS derivative of the same 70B) under vendor sampling, a 65,536-token cap, a pinned Math-Verify scorer, and an append-only attempt ledger, for 45,000 scored responses. The two reasoning models both reach 96.1% accuracy against 76.2% for the 70B baseline. The empirical product-coupling premium is 0.215-0.588 pp across all two- and three-model pools; the empirical plug-in selection headroom over always calling Nano is 1.27 pp; and resampling Nano three times gives practically the same empirical any-correct rate as calling all three models once. Only 31 responses reached the cap, 25 of them with exactly periodic tails, yet the capped responses account for up to 60.2% of a model's output-length-squared proxy, and long generations made a fourth planned model, Llama-3.1-8B, too slow to finish. The premium is far below the companion's 3.5 pp, but models, pool size and protocol changed together; an oracle gap should therefore be reported with the pool, the protocol, the samples per cell and the decoding budget that produced it. 

---
# AlgoPilot: Cross-Paradigm Reasoning in Language Models via Strategy Selection and Guidance 

**Authors**: Yilun Hao, Ruixiao Yang, Hudson Hilal, Yongchao Chen, Kaizhi Qian, Yada Zhu, Yang Zhang, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2611.04079)  

**Abstract**: Real-world problems span diverse domains, including program synthesis, symbolic reasoning, planning, and optimization, and often require fundamentally different solution paradigms. A central challenge for LLM-based reasoning is twofold: identifying an appropriate problem-solving approach and formulating it correctly. In this paper, we propose AlgoPilot, a cross-paradigm reasoning framework that enables adaptive selection of the solving approaches and structured problem formulation prior to execution. The AlgoPilot framework contains three components: 1) a SteerLM, trained via supervised fine-tuning (SFT) and reinforcement learning (RL), that adaptively selects the solving strategy (paradigm and modes) based on input problems, 2) the corresponding expert GuideLM, trained via SFT, that generates structured formulation guidance tailored to the selected strategy, and 3) an ExecutionLM that generates solutions conditioned on both the selected strategy and its formulation. Unlike prompting-based methods that rely on fixed reasoning templates, our approach learns to adaptively choose and structure problem-solving strategies based on the input. We build AlgoPilot on Qwen3-8B and evaluate across 5 problem categories and 13 benchmarks with 66 unique domains, comparing against both the latest similarly scaled small models and large LLMs. AlgoPilot significantly improves average accuracy for the same model from 38.4% to 72.9%, outperforming the best large LLM prompting baseline GLM-5 by 11.7%. Ablation studies confirm the effectiveness of each component. Notably, we show that the learned steering and formulation modules can transfer across execution models, suggesting their potential to improve other LLMs. Project page: this https URL. 

---
# WASP-Mem: Let Agent Organize Procedural Memory. Human Spend Effort on Diagnosis 

**Authors**: Peizhi Niu, Ge Shi, Vaibhav Gorde, Sarthak Ghosh, Christos Faloutsos, Min Cheng, Kevin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.04003)  

**Abstract**: LLM agents struggle to automate complex enterprise procedures because static, pre-built harness cannot capture proprietary policies, undocumented behaviors, or state-dependent recovery paths that surface only during execution. Procedural memory accrued from agent experience offers a training-free path to closing this gap -- yet unlike episodic records or semantic facts, it must consolidate repeated successes and diagnosed failures into reusable actions and recovery strategies. We introduce WASP-Mem (Wiki-based Agent-Synthesized Procedural Memory), a nimble but effective method that distills execution traces and feedback into persistent, interlinked wiki pages, enabling continual agent learning. On TheAgentCompany benchmark, WASP-Mem raises the checkpoint pass rate by 4.6-6.1% over the strongest baseline we compared across three models while requiring 57-61% less storage. We further present a systematic controlled study spanning consolidation method, feedback type, trace ingestion scope, and refinement rounds. Our results reveal that prescriptive knowledge organization such as human defined schemas and explicit deduplication rules hurts performance rather than closing the procedural-knowledge gap. In contrast, human diagnosis converts failed traces into corrective knowledge and drives sustained improvement: a single round of human diagnosis outperforms four rounds of self-judgment by 7.0-10.9%. These findings point to a simple division of labor: let agents organize procedural memory, and invest human effort where it matters most -- diagnosis. 

---
# Thinking Fast, Trading Late: System One Decision Models vs the Market 

**Authors**: Amir Dehkordi, Alireza Kargarzadeh, Nariman Khaledian, Navid Parvini, Sid Ghatak, Arman Khaledian  

**Link**: [PDF](https://arxiv.org/pdf/2611.03985)  

**Abstract**: Large language models (LLMs) read company news at scale, and backtests suggest that their readings predict stock returns. We ask how much of that predictability survives a test that rules out look-ahead, memorization and selection. We propose OLIVIA, a protocol for such tests: point-in-time inputs with company identities masked, selection on validation data only, an analysis plan frozen by cryptographic hash, and a test period unblinded once through a logged lock. We apply it to AI decision models, non-generative language models that return probabilities for typed questions, reading news about S&P 500 stocks: an open-weights model we fine-tuned on 2024-2025 data, three commercial models and standard baselines. The fine-tuned model's signal replicated on two validation quarters (daily information coefficients 0.055 and 0.065) but fell to 0.020 (t = 1.7) on the blinded January-July 2026 test. The commercial models' readings line up with the overnight move to the open (information coefficient 0.074 to 0.077) but not with the move after it, so the news is priced by the time a trader can act on it at the open. No model can tell over- from under-reaction, and no pre-registered strategy survives costs and a deflated Sharpe ratio. Commercial models identify masked companies in 62% to 74% of cases against a 28% baseline: masking hides names, not companies. The market reads the news first. 

---
# Must AI Spell It Out? Pragmatic Comprehension of Alerts in a Time-Critical Task 

**Authors**: Anwesha Das, Vera Demberg, Anna Maria Feit  

**Link**: [PDF](https://arxiv.org/pdf/2611.03976)  

**Abstract**: Assistive systems typically tell users everything, often spending time and attention on what a user already knows, leaving less for what happens next. People, on the other hand, leave out what they expect the listener can supply. Can people use their prior knowledge to understand an incomplete message when the speaker is a machine though? We studied this pragmatic comprehension in a monitoring task controlling users' prior knowledge and the available time to detect critical situations. In a preregistered study (N=114), we compared a pragmatic planner that reasons over a model of the listener's knowledge to decide what to say, when, and how explicitly, with a baseline specifying problems fully in onset order. Participants resolved underspecified alerts almost as reliably as fully specified ones and sooner, detected more problems, and rated the pragmatic assistant helpful more often, with no increase in reported mental demand. Our findings show that people can recover what the system left implicit and that pragmatic communication can support effective assistance under the constraints of a changing world. 

---
# EpiLoRA: Fusing Many LoRA Adapters into Low-Precision GEMMs for Serving and RL 

**Authors**: Hieu Tran, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.03899)  

**Abstract**: Low-rank adapters (LoRA) let one base model serve many fine-tuned variants: multi-tenant engines keep a single copy of the base and batch the requests of many tenants, each with its own adapter. As base models move to 8-bit (FP8) and 4-bit formats, their general matrix multiplications (GEMMs) become cheaper, but the adapters do not: engines run them as separate small GPU kernels beside the GEMM, whose latency takes a growing share of every decoding step. An adapter projects its input down to a low rank and back up, and our key observation is that the GEMM needs the down-projection only in its epilogue, the final stage in which it writes its output. EpiLoRA, a plugin for the vLLM engine, therefore launches the down-projection first and the low-precision GEMM as a dependent kernel that streams its weights meanwhile and adds every tenant's up-projection in its epilogue. On NVIDIA Hopper GPUs, EpiLoRA decodes for a single adapter with up to 24% lower time per token than vLLM's fastest configuration, and faster than SGLang and TensorRT-LLM; with many tenants, its adapter kernels deliver more throughput than vLLM's on most GPU-model pairs we tested. The same low-precision base also trains: a LoRA reinforcement-learning trainer that computes on the engine's FP8 weights needs no 16-bit copy, so a 27B model trains on one 48 GB GPU, and an engine that then keeps activations in 16 bits removes most of the remaining mismatch between the model generating training data and the one trained. 

---
# Functional-Group Knowledge Is Structured in Language Models Before It Is Spoken 

**Authors**: Tran Nhat Khang Nguyen, Anh Nguyen, Obed Issakah, Janghoon Ock  

**Link**: [PDF](https://arxiv.org/pdf/2611.03684)  

**Abstract**: Language models serve as the reasoning core of agents for chemistry and materials science, where they must interpret molecular structure before planning syntheses or proposing materials. Yet their outputs show only what they say, not what they know. Given a prompt such as ``The primary functional group in 1-Fluorobutane (CCCCF) is'', a small general-purpose model may answer ``the alkyne group'' instead of an alkyl fluoride. The output alone cannot tell whether the model failed to recognize the fluorine or recognized it and failed to say so. We address this by reading the internal representations of five open-source models on prompts covering 92 molecules and 20 functional groups. Even without chemistry training, the models organize functional groups by chemical family, and they encode chemical differences as consistent directions. The shift from thioether ($\mathrm{R{-}S{-}R'}$) to thiol ($\mathrm{R{-}SH}$), added to ether ($\mathrm{R{-}O{-}R'}$), lands nearest alcohol ($\mathrm{R{-}OH}$). This structure emerges abruptly in the middle layers, where representations shift from tracking the wording of the prompt to tracking the functional group. The base model names the correct functional group in only 58.5\% of prompts, the lowest of the five, yet it encodes functional-group families and preserves chemically meaningful patterns. Its errors are mostly failures to answer rather than wrong answers. Functional-group knowledge is structured in the model before it is spoken. Chemistry fine-tuning reorganizes it, making functional groups rather than prompt wording dominate the representation. These results suggest that evaluating chemical language models through outputs alone can underestimate the chemical information they encode. Representation analysis can distinguish a lack of knowledge from a failure to express it, which matters when choosing or adapting models for chemical agents. 

---
# Mixture of Wide Experts: Sparse Routing for Interpretable Native Coordinates 

**Authors**: Tavish Mankash, Vardhaman Kalloli, Keshava Prasad, Deepan Muthirayan  

**Link**: [PDF](https://arxiv.org/pdf/2611.03655)  

**Abstract**: Language models often spread a concept across many internal dimensions, making their features difficult to interpret. Sparse autoencoders (SAEs) recover clearer features after training. We ask whether models can instead learn interpretable output coordinates directly. We introduce Mixture of Wide Experts (MoWE), a sparse feed-forward layer in which selected experts read the full representation but write to fixed, disjoint output coordinates. On a controlled task with known hidden causes, this design preserves predictive performance while localizing causes more clearly than dense and standard mixture-of-experts (MoE) models.
At approximately matched compute, we train two 19-layer runs of each architecture for 10 billion tokens. MoWE achieves evaluation cross-entropy 2.417 versus 2.452 for dense, while standard MoE reaches 2.22.
We measure native-coordinate readability by asking a language-model judge to describe each coordinate from passages where it is strongly active and test that description on new passages. Correcting for descriptions that also fit other coordinates gives a specificity score of 0.523 for MoWE, 0.375 for dense, and 0.268 for standard MoE. TopK SAEs trained on the same FFN updates score 0.489 for dense and 0.556 for MoWE, making native MoWE comparable to the dense SAE.
In the first case study, we identify a four-coordinate causal path supporting the model's prediction of degrees after the prompt A right angle is 90. We then copy seven angle-related coordinate values into the model as it processes A rectangle has four, changing its next-token prediction from sides to degrees. In the second, one fixed coordinate set detects hazardous requests and another steers the next token toward refusal. These examples show that MoWE combines sparse conditional computation with coordinates that can be identified and reused directly. 

---
# Alignment Faking Depends on the Value, Not Just the Model 

**Authors**: Manodeep Ray, Vivek Srivastava, Shirish Karande, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2611.03650)  

**Abstract**: Language models can follow a directive when they believe they are monitored and abandon it when they believe they are not, a pattern termed Alignment Faking. Prior work establishes this for harmful-content objectives and a handful of moral-foundations values, leaving open how it distributes across a wider value space. We argue it is value-dependent: whether compliance survives a change in perceived oversight is a property of the model--value pairing, not of the model alone. Our framework profiles Alignment Faking per value, scoring each model-value pair under three conditions: bare (no system prompt), unmonitored, and monitored. On 32 ValueBench dimensions across ten open- and closed-weight models (8B-~675B), compliance gaps span 0.05-0.50, with a distilled 70B above a ~675B model. We observe that majority of the structural variance depends on which model was asked about which value and not on the model or value alone. A six-category taxonomy over reasoning traces tracks the behavioural shift: training-signal reasoning dominates under monitoring and value-conflict reasoning dominates when unmonitored. We package per-model results as value cards, extending model cards with behavioral disposition. Code and data are available at this https URL. 

---
# TRIAGE: Reliability- and Load-Conditioned Escalation for Two-Tier LLM Serving with a Semantic Evidence Judge 

**Authors**: Basavaraj R M, Aryan Bharat Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2611.01174)  

**Abstract**: Two-tier LLM serving escalates selected requests from a small model to a large one, but prompt-based routers cannot see whether retrieved evidence supports an answer, and they treat the GPU cost of an escalation as constant. We study TRIAGE, which conditions escalation on a post-generation reliability state and on a load-dependent GPU price, and we report both what held and what did not on one A40 serving Qwen2.5-7B and 32B (AWQ) under VLLM. Escalation cost rose 21.9x from 1 to 10 requests/s but showed no monotone dependence on KV-cache headroom, and a pre-registered claim that the reliability state detects corrupted evidence (AUROC>=0.70) was falsified at 0.571, although any prompt-only router scores exactly 0.500 by construction. Post hoc, a semantic evidence judge raised detection to 0.846 (0.926 on never-served items) and predicted wrong small-model answers at AUROC 0.90-0.96 on RGB and three RouterBench model pairs; the resulting cascade stayed within about 3 pp of GPT-4 at 79.5% lower API cost for GPT-3.5->GPT-4, yet fell below random routing for open-model pairs with 2-3x price ratios. End to end, the 7B->32B pair showed only a +2.88 pp held-out gap, so the allocator rightly declined to escalate and the headline is a null that a small pre-launch gate failed to expose. 

---
# Zero-Gated Cross-Attention 

**Authors**: Basavaraj R M, Manav Verma  

**Link**: [PDF](https://arxiv.org/pdf/2611.01116)  

**Abstract**: Injecting non-text modalities into frozen language models often introduces early optimization instability, making it difficult to separate architectural gains from transient training dynamics. We introduce Inductively Gated Cross-Attention (IGCA), a zero-initialized cross-attention bridge that preserves pretrained model behavior at initialization while enabling gradual multimodal conditioning. Integrated across six decoder layers of frozen Falcon-40B, open_llama_30b, and Bloom-7b1 backbones (with 40,404,961,288 trainable adapter parameters used during training), IGCA acts as an exact algebraic no-op at initialization with a maximum output discrepancy of 0.00e+00. Over 59,500 training steps on Falcon-40B, the training loss decreases from 3.9403 to 0.0007 (with final ten step losses of [0.0001, 0.0003, 0.0005, 0.0002, 0.0005, 0.0011, 0.0005, 0.0006, 0.0005, 0.0007]). On held-out evaluation, the best gate-on checkpoint achieves a loss of 2.2201 compared to 4.3469 for the gate-off baseline -- a 48.9% reduction (2.1268 nats) -- while average held-out loss improves from 4.3423 to 2.2498 (1.9075 nats) with a mean gate magnitude of $\vert{}\tanh(\alpha)\vert{} = 0.0713$. 

---
# WMIP: A Working-Memory-Inspired Programmable Probe of Cumulative State Tracking in Large Language Models 

**Authors**: Dengzhe Hou, Lingyu Jiang, Deng Li, Zirui Li, Fangzhou Lin, Kazunori D Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2603.27343)  

**Abstract**: Human working memory lets people maintain and update information. Large language models (LLMs) face a related demand when tracking updates within a single prompt without writing out intermediate states, yet existing tests differ in their constraints on written steps. We introduce a working-memory-inspired probe of cumulative state tracking with program-computed answers, 3 to 100 updates, and final-state, intermediate-state, or multi-target queries. It compares protocols on identical items. Across 18 LLMs, accuracy is 84.9% on single-step arithmetic but 31.7% on core items requiring one number after three to seven updates. From 50 updates on, no model exceeds 10% accuracy. Core accuracy reaches 88.0% under a larger budget and instructions to write steps and place the answer on the last line, mainly due to the instruction changes rather than the budget. Core accuracy also correlates positively with public Berkeley Function Calling Leaderboard scores. These findings show that strong single-step arithmetic does not ensure reliable cumulative tracking and that output protocols substantially change measured accuracy. We release the generator, items, scoring code, and responses for reproducible comparisons across depths, query types, and protocols. 

---
# Computational Phenomenology of Borderline Personality Disorder: A Comparative Evaluation of LLM-Simulated Expert Personas and Human Clinical Experts 

**Authors**: Marcin Moskalewicz, Anna Sterna, Karolina Drożdż, Kacper Dudzic, Marek Pokropski, Paula Flores  

**Link**: [PDF](https://arxiv.org/pdf/2508.19008)  

**Abstract**: Building on a human-led thematic analysis of clinical life-story interviews (> 150,000 words) with inpatients with Borderline Personality Disorder, this study examines the capacity of large language models (OpenAI's GPT, Google's Gemini, and Anthropic's Claude) to support qualitative clinical analysis. The models' interpretative potential was evaluated using a mixed-methods approach. Study A involved blinded and non-blinded judges in phenomenology and clinical psychology. The experts assessed the validity of AI-generated content using semantic congruence, Jaccard coefficients, and multidimensional validity ratings, including credibility, coherence, the substantiveness of results, and grounding in qualitative data. In Study B, neural methods were used to embed human- and model-generated theme descriptions in a multidimensional vector space. This approach provided an objectified computational measure of the difference between human and model semantics and linguistic style. In Study C, complementary non-expert evaluations were conducted (N=115) to examine the influence of thematic verbosity on the perception of human authorship and content validity. Overall, the results of AI analysis showed a highly variable overlap (0-58%) with the human interpretation, while all models identified themes originally omitted by human researchers, proving their capacity to mitigate human bias. At the thematic level, external evaluators were unable to reliably distinguish human-authored themes from those generated by AI. In terms of content validity assessed against raw data by high-level experts in the blinded mode, the performance of Gemini 2.5 Pro was indistinguishable from that of humans. The comparison of semantic vector embeddings showed that its style was also the closest to humans. 

---
