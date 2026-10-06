# MemPilot: Orchestrating On-Demand Multimodal Memory Curation for LLM Agents 

**Authors**: Haozhen Zhang, Haodong Yue, Quanyu Long, Jianzhu Bao, Qingyuan Liu, Tao Feng, Bohan Liu, Weida Liang, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06830)  

**Abstract**: Memory has become integral to the LLM agent ecosystem, supporting information retention and reuse across interactions. However, most existing agent memory systems construct memory in a query-agnostic manner, which can incur unnecessary preprocessing cost and discard details that later prove essential. Recent studies have begun shifting memory processing toward runtime adaptation, but typically specialize in particular operations or fixed processing schemes, leaving flexible control over performance, cost, and latency largely underexplored. To address this challenge, we present \textbf{MemPilot}, a flexible framework that orchestrates on-demand memory curation under different performance--cost--latency preferences. Specifically, we optimize a multi-step LLM policy via reinforcement learning to iteratively choose between retrieving from query-agnostic memory and delegating query-specific curation of raw multimodal history to heterogeneous LLMs and VLMs. The policy jointly controls evidence amount, curation instructions, model selection, and visual access, enabling fine-grained allocation of runtime computation. To optimize this policy under competing objectives, we adapt objective-wise advantage decoupling by separately estimating each objective's advantage before aggregation. Moreover, we introduce prefix-based marginal utility estimation for fine-grained credit assignment across multi-step rollouts. Experiments on five multimodal agent-memory benchmarks demonstrate favorable performance--cost--latency trade-offs across optimization preferences, with preference sweeps yielding broader frontiers than existing trade-off-aware baselines. 

---
# CLIFT: Conformal Self-Verification for Web Agent Training and Test-Time Scaling 

**Authors**: Yifan Zhang, Yutong Dai, Viraj Prabhu, Zhiyuan Hu, Ran Xu, Zeyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.06829)  

**Abstract**: Open-source web agents are now strong enough to execute realistic browser tasks, but training them with reinforcement learning still depends on weak supervision: binary task success is too sparse for credit assignment, while frontier-language-model judges are too expensive to call at every step and cannot be assumed available at deployment. We introduce CLIFT, a training and test-time scaling method built around conformal self-verification. During training, the agent answers natural-language verification questions about its own rollouts; a Compositional Conformal Certifier keeps only question signals whose URL-conditional evidence agrees with a training-time judge, assigns signed trust weights through polarity-aware lift, and blends the resulting verifier score into per-step rewards in a way that never subtracts from the judge baseline. At test time, the same certified bank is frozen and reused as structured evidence for Conformal Trajectory Selection (CTS): the agent samples a greedy rollout and one or more diverse retries, the self-verifier summarises each URL trace, and a conservative majority-vote rule chooses whether to swap away from the current incumbent without calling any external judge. This single mechanism supports three settings. On WebArena Infinity, CLIFT achieves state-of-the-art performance among open-source web agents. On VisualWebArena, a bank trained with the open model transfers to GPT-5.5 at test time and reaches state-of-the-art performance under the canonical harness. On Online Mind2Web, without training an agent on the benchmark, translating the certified question bank improves a live-web agent in zero-shot evaluation. Together these results position conformal self-verification as a way to turn costly judge feedback into a reusable training signal and a judge-free test-time scaling signal. 

---
# PlotGround: Grounding Plot Digitization in Real Scientific Figures and Their Source Data 

**Authors**: Yaohui Zhang, Binxu Li, Haoyi Duan, Jiacheng Miao, Yixin Wang, Xinran Du, Chenyue Li, Shilong Liu, Kevin Wu, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2610.06825)  

**Abstract**: Scientific figures often encode quantitative results that are not readily available in machine-readable form, making accurate plot digitization important for verifying and reusing published findings. Yet it remains unclear how accurately current models recover plotted values from real scientific figures, as existing benchmarks rely largely on synthetic charts or cover only a limited range of chart types. We introduce PlotGround, an automated pipeline for building plot digitization benchmarks from real scientific figures and their author-released source data. PlotGround maps figures to source tables, identifies reconstructable panels, and generates quantitative questions with source-grounded reference values. We use PlotGround to construct PlotGround-1k, a human-verified benchmark of 1,119 questions from 1,066 bioRxiv preprints. Across sixteen multimodal models, the best reaches 87.5% accuracy at a $\pm 5\%$ relative-error tolerance. Tightening the tolerance to $\pm 2\%$ lowers every model's accuracy by 11-24 percentage points, revealing a gap between approximate visual reading and precise quantitative recovery. PlotGround's paired figure-source structure lets us compare how accurately the same values are recovered from figures and from source tables. Providing source tables instead of figures raises a coding agent's accuracy from 90.0% to 97.4% while cutting cost by 72%. 

---
# T-Search: An Open Agentic Retriever and Playground for Hard Multi-Step Search 

**Authors**: Olga Tsymboi, Ramil Latypov, Aleksandr Medvedev, Danil Taranets, Dmitrii Stoianov, Nikita Gulyakov, Gleb Alektorov, Anatolii Potapov  

**Link**: [PDF](https://arxiv.org/pdf/2610.06782)  

**Abstract**: We present T-Search, an open-weight agentic retriever for hard multi-step search. Given a question and a search tool over a fixed corpus, it runs a bounded multi-round search and returns a ranked list of evidence chunks with short justifications, leaving answer generation to a downstream model, so backend and generator can be swapped without retraining. T-Search is built on Qwen3.6-35B-A3B and trained on adversarially filtered synthetic search tasks with round-sliced supervised fine-tuning followed by GSPO on a recall reward. Averaged over seven English and Russian benchmarks with gold evidence annotations, it reaches 56.0 Recall@10 with one rollout, 14.4 points above its base, and 61.3 with three fused rollouts, outperforming larger open models. We release the model, harness, live demo, and three benchmarks, including TRuST, the first native-Russian hard-search benchmark. 

---
# IdeaLens: Detecting AI Ideas in Long-form Writing 

**Authors**: Rishanth Rajendhran, Minjoon Choi, Jenna Russell, Ramya Namuduri, Deniz Bölöni-Turgut, Marzena Karpinska, John Wieting, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2610.06778)  

**Abstract**: While modern AI detectors identify who wrote the words, emerging policies on AI use increasingly hinge on a different question: who came up with the ideas? We introduce IdeaLens, a detector that identifies whether a document's ideas came from a human or AI (idea provenance), regardless of who wrote its words. To focus IdeaLens on ideas rather than prose, we represent documents as outlines: lists of items that each pair a discourse role with a brief, paraphrased description of the content, minimizing word-level overlap with the raw text. We train IdeaLens on 1M FineWeb documents with silver labels from Pangram, a prose provenance detector. Since the outlines are largely stripped of surface-level information, the labels must be fit mainly through the ideas. In a controlled study, IdeaLens's AI flag rate drops from 95% to 7% as models write from increasingly detailed human plans, while Pangram 4 still flags 92%; from AI-derived plans, IdeaLens stays above 96%. Conversely, on a new dataset of 50 stories that human authors wrote from AI-generated plans, IdeaLens flags 68% of the stories as AI, compared to 8% for Pangram 4. On a comprehensive suite of 19 existing detection benchmarks, we show that IdeaLens maintains strong detection rates at low false positive rates, suggesting that ideas themselves provide a powerful discriminative signal, and its performance holds across domains, formats, and languages. Finally, we examine 90K predictions from IdeaLens to characterize systematic differences between human and AI ideation. We release our models and labeled datasets to facilitate future research on idea provenance detection. 

---
# Balancing Memory Pathways: Analyzing and Improving Memory Utilization in Hybrid LMs 

**Authors**: Hyunji Lee, Joykirat Singh, Zaid Khan, Justin Chih-Yao Chen, Elias Stengel-Eskin, Alessandro Sordoni, Arman Cohan, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2610.06750)  

**Abstract**: Recurrent-attention hybrid language models (LMs), which interleave attention and recurrent layers, are increasingly used to combine the efficiency of the recurrent layers with the strong performance of attention layers. Prior work suggests that attention and recurrent layers offer complementary pathways to use past information: attention supports precise memory recall from earlier tokens, while recurrent layers support consolidation of disparate information over long contexts. However, we observe that simply having access to both pathways does not mean that hybrid LMs are effectively using them. We find that they rely substantially more on attention than on the recurrent state. Standard supervised fine-tuning improves overall performance but does not improve how the two memory pathways are coordinated: the model becomes more reliant on information propagated by attention layers, while its use of information propagated by recurrent layers remains limited. To encourage better coordination between the two memory pathways, we add an auxiliary loss that limits attention's access to earlier context while the recurrent state propagates through the full sequence. This objective encourages the model to retain and use information through the recurrent pathway alongside attention. It improves overall performance, with particularly strong gains on tasks involving longer contexts or requiring information aggregation, consistent with the strengths of recurrent layers observed in analysis. Crucially, this imbalance and the benefit of our auxiliary loss generalize: they apply to multiple recurrent-attention LMs in question-answering and agentic tasks, as well as to attention-based LMs that combine different forms of memory. Together, our findings show that simply providing multiple memory pathways does not ensure their effective use, and that targeted supervision is needed to better coordinate them. 

---
# ufakzeka-karar: An Open Turkish Typed-Decision Model with Order-Invariant Option Scoring 

**Authors**: Sait Furkan Teke  

**Link**: [PDF](https://arxiv.org/pdf/2610.06744)  

**Abstract**: ufakzeka-karar is an open Turkish decision model with 182,494,466 parameters. Given a Turkish text and questions of a fixed answer type (a choice, a level on an ordered scale, or yes or no), it returns a temperature-scaled probability for every option and an expected error that serves as a "not sure" signal, without generating text and in one CPU forward pass for up to ten options. Built on the lab's ufakzeka-1-base, its head scores each option blind to the others at shared positions, so the answer does not depend on option order. A sequential head trained with shuffled options was about as accurate but changed 2.3 to 2.8 percent of its answers when only the option order changed; REINFORCE lost 10.2 points (0.102) of macro F1 to cross-entropy. On the open set of HakemBench v1.0 (4,275 questions, 7 tracks) the released model ranks 7th of 16 rows with a composite of 0.660 (95% interval 0.642 to 0.677). Temperature scaling lowers calibration error (smooth ECE) on the development set but raises it on held-out support questions, from 0.027 to 0.045 for the first scored run, which never trained on them; the released model later trained on them, so its 0.036 to 0.064 is not an unseen-question test. The released model is the last of three runs scored on HakemBench, and its numbers are not blind. The second run's new training data was aimed at the first run's errors on the full test set in guardrails, moderation and customer support, and the released run was trained after the second run's guardrail results on the full test set were read, under a protocol fixed in writing before any of its data, code or runs. All its numbers come after these readings; its guardrail, moderation and customer support numbers carry the flag "shaped by reading the test results". With every model scored on the other four tracks only, its composite is 0.678, 6th of 16. Weights and code are under Apache-2.0. 

---
# Improving Diversity in LLM Short Story Generation 

**Authors**: Zahra Solati Dehkordi, Vasileios Lampos  

**Link**: [PDF](https://arxiv.org/pdf/2610.06729)  

**Abstract**: Large language models (LLMs) can generate accurate responses, but these are void of diversity. We attempt to address this for the task of creative short story generation. Drawing on established writing conventions and known LLM limitations, we target variation in genre, tone, style, and named entities. To promote diversity across these dimensions, we introduce DivLM, an LLM post-training framework consisting of two phases. First, we perform continued pre-training on a creative writing corpus and restore instruction-following capabilities using weight residuals. We then apply reinforcement learning with a custom, composite reward function that jointly maximizes diversity across the targeted narrative dimensions while maintaining response quality. Our empirical results on two LLM families show that DivLM increases diversity metrics by more than 9% on average compared to alternative approaches, while preserving instruction following, overall response quality, and similarity to human outputs. 

---
# Domain adaptation of Russian ModernBERT for long legal documents 

**Authors**: I. Litvak, D. Gvozdetsky, F. Lashkin, V. Kirova, S. Lagutin, V. Volf, T. Maksiyan, A. Kostin, R. Leva, I. Kiselev  

**Link**: [PDF](https://arxiv.org/pdf/2610.06715)  

**Abstract**: We investigate whether continued pretraining on Russian legislative documents improves a Russian ModernBERT encoder on legal text. The adapted model, RuModernBERT-ruLaw, was trained on a corpus reported to contain 304,382 legislative documents and 194,425,905 corpus tokens. Corpus token counts are distinguished from positions produced by the model tokenizer. We compare the original and adapted encoders on a fixed external collection of 1,031 court-decision segments. Both models receive the same hidden positions in each of five masking realizations. At maximum input lengths of 512, 2,048, and 8,192 tokens, mean masked-token cross-entropy decreases by 0.10942, 0.07052, and 0.06604 natural-log units, respectively. The reported 95% intervals summarize sensitivity to masking on this fixed collection; they do not quantify uncertainty across document collections. A second evaluation addresses legal-entity extraction. The original and adapted models achieve entity-level F1 scores of 0.99852 and 0.99820. However, 99.95% of test spans have the same normalized surface form and class in the training split. This evaluation therefore provides limited evidence about transfer to previously unseen forms. The paper explains the masking objective, overlapping windows, averaging rules, and exact entity-boundary scoring using editable diagrams and clearly marked illustrative examples. The comparison supports lower masked-token prediction loss for the studied pair of models and collection. It does not isolate the contribution of distant context or establish practical legal utility. 

---
# SAFE-MR: Evidence Sufficiency Learning for Selective Multimodal Rumor Detection 

**Authors**: Shiwen Ni  

**Link**: [PDF](https://arxiv.org/pdf/2610.06708)  

**Abstract**: Multimodal rumor detectors increasingly rely on retrieved evidence, yet relevant evidence is not necessarily sufficient for verification. Missing provenance, duplicated reports, and unresolved contradictions can produce confident predictions without adequate support. We introduce SAFE-MR, a framework that separates claim veracity from evidence sufficiency. The method decomposes image-text posts into verifiable claims, constructs a relation-aware claim-evidence graph, and aggregates evidence using provenance and contextual compatibility. Separate veracity and sufficiency heads support selective prediction, while evidence interventions encourage stability under irrelevant additions and sensitivity to evidence removal. On NewsCLIPpings, VERITE, and XFacta, SAFE-MR achieves macro-F1 scores of 91.2%, 75.8%, and 85.2%, respectively. Against the matched backbone with evidence, its macro-F1 gains are 2.2, 4.9, and 4.8 percentage points. On the diagnostic selection set, SAFE-MR reduces AURC from 0.105 for maximum-probability rejection to 0.075 and lowers error at 80% coverage from 13.8% to 8.5%. Evidence-perturbation and ablation results support the role of sufficiency learning and intervention training in improving selective verification. 

---
# MedPrune: Topology-Efficient Multimodal Multi-Agent Communication Evolution for Medical VQA Tasks 

**Authors**: Jiuheng Wan, Runze Li, Chen Chen, Tingyuan Hu, Daiyang Yu, Yimin Jing, Taolin Zhang, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2610.06695)  

**Abstract**: While medical multimodal large language models (Med-MLLMs) advance medical visual question answering (VQA), existing clinical workflow-inspired multi-agent frameworks suffer from interaction patterns and excessive computational overhead caused by redundant communication topologies. In this paper, we propose MedPrune, an efficient medical multimodal multi-agent collaboration framework that dynamically prunes both nodes and edges from the communication topology to enhance reasoning ability and token efficiency. Specifically, we first formulate the diagnostic process as a heterogeneous communication graph, where nodes represent specialist agents from various departments and edges capture intra- and inter-departmental interactions. Building on this graph, we introduce two sparsification mechanisms to enable adaptive collaborative evolution: (1) Heterogeneous Node Sparsification, which eliminates task-irrelevant specialist agents irrelevant to the current multimodal question via reinforcement learning-driven topological optimization, and (2) Heterogeneous Edge Sparsification, which selectively retains only the most diagnostically salient intra- and inter-departmental connections by jointly optimizing task performance and topological complexity. Extensive medical VQA experiments under full-set and few-shot training settings prove MedPrune surpasses multi-agent baselines and boosts token efficiency with strong adversarial robustness. 

---
# Programmatic Search Agents: Extending Agentic Search Beyond Query Reformulation 

**Authors**: Jiaming Qian, Huiyan Yang, Mandi Liu, Jie Liu, Wenkai Shen, Pengyang Zhou, Jing Jin, Jin Ma, Dezhi Ye, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.06689)  

**Abstract**: Search agents adapt their queries, yet fixed search interfaces leave candidate processing and evidence presentation outside the agent's direct control. Our trajectory analysis shows that supporting passages can be retrieved yet never delivered to the agent; a same-page oracle intervention shows that changing the returned evidence can reduce subsequent search. We introduce Programmatic Search Agent (PSA), which makes a local executable computation over candidates the unit of a search action. PSA unifies a persistent candidate workspace, flexible primitive composition, and selective evidence presentation. It incrementally generates program cells that reuse candidates, execute dependent operations, and select what the agent inspects next. The runtime resolves specified data dependencies within each cell, while the agent adapts its search strategy across cells as new evidence arrives. We compare PSA with the Query-based Agent and Tool-based Agent on InfoSeek-Eval and BrowseComp-Plus using five policy backbones without task-specific training. All three interfaces share the search substrate, and the Tool-based Agent also shares PSA's primitives and persistent workspace. Relative to the Query-based Agent, PSA improves macro-averaged task success by 4.00 and 7.56 percentage points on the two benchmarks, respectively; within-backbone reductions in final-step tokens average 28.3% and 33.9%. These results support extending agent control beyond query reformulation to the processing and presentation of retrieved evidence. Code will be released subject to approval. 

---
# Reward Stealing Attack on Large Language Models 

**Authors**: Jiaming Qian, Pengyang Zhou, Jiahe Xu, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.06670)  

**Abstract**: Adversarial attacks on Large Language Models (LLMs) aim to induce harmful content. However, existing methods suffer from high computational costs or strict model-pairing dependencies, limiting their scalability and transferability. We propose Reward Stealing Attack (ReSA), an adversarial attack framework that targets the latent safety reward underlying LLM alignment. ReSA employs maximum entropy inverse reinforcement learning to recover a proxy reward model solely from the aligned model's behavior. The extracted reward is then reversed at inference time to derive an adversarial policy, efficiently implemented via a reward-guided decoding mechanism. Experiments demonstrate that a single recovered reward generalizes across prompts and diverse models to reveal a fundamental alignment vulnerability, enabling ReSA to significantly outperform existing attacks in effectiveness and transferability. The code is available at this https URL. 

---
# Wikidata Search Traces: A Dataset for Training Knowledge Graph Search Agents 

**Authors**: Mohamed Chenene, Carlos Rosas-Hinostroza, Pierre-Carl Langlais, Anastasia Stasenko  

**Link**: [PDF](https://arxiv.org/pdf/2610.06650)  

**Abstract**: Wikidata is one of the largest open knowledge bases, yet answering a complex question over it still requires a SPARQL query that names the right entities and properties and chains their relations. Language models offer a natural-language alternative but answer largely from memory, which is least reliable for less prominent entities. We study agents that instead answer by exploring the graph, and argue that two obstacles limit them: the lack of training data recording how a solver explores, and interfaces that add large graph results directly to the model's context. We test three hypotheses: that the difficulty of graph search can be controlled through the structure of a question rather than only through obscure entities or wording; that much of the failure on long-horizon search comes from how retrieved evidence is managed rather than from the model itself; and that, in a suitable environment, open-weight models can match commercial closed ones. We construct multi-hop questions on a frozen Wikidata snapshot by replacing named entities with nested conditions, checking after each expansion that the target remains unique and that every new condition is necessary. We release 10,235 solving traces over single-entity and multi-hop questions, together with the recursive language model (RLM) harness that produced them, in which models batch graph calls, keep results in persistent Python state and interpret selected evidence through sub-calls. On 100 questions, the harness improves both models we ran under both interfaces compared with direct tool calling over the same functions: gpt-6-luna rises from 49 to 61 correct answers, doubling its multi-hop accuracy, and Qwen3.8-27B, an open-weight model served on a single GPU, from 60 to 74. 

---
# Representation-Space MMD for Diffusion Language Models 

**Authors**: Ilya Drobyshevskiy, Ilia Sudakov, Maksim Semenov, Denis Kuznedelev, Maksim Ignatov, Pavel Temirchev, Nikita Balagansky, Viacheslav Meshchaninov, Nikita Gushchin, Dmitry Baranchuk  

**Link**: [PDF](https://arxiv.org/pdf/2610.06648)  

**Abstract**: We introduce a post-training method for diffusion language models (DLMs) that minimizes Maximum Mean Discrepancy (MMD) between generated and reference distributions in the feature space of a frozen pretrained DLM. To estimate MMD, we retain contextual features at individual token positions, obtaining multiple observations per sequence from a single extractor pass. We optimize this objective using policy gradients for discrete models and direct differentiation through generated latents for continuous models. In both cases, computing the loss directly from these features enables efficient post-training without full sampling trajectories or jointly trained auxiliary models. Experiments show lower generative perplexity at comparable entropy on OpenWebText and better accuracy-computation trade-offs on GSM8K. On 16B DMax-LLaDA2.0 models with hybrid masked-uniform diffusion, we increase decoding parallelism with similar or higher accuracy on math and code benchmarks. 

---
# LoGRA: Scaling LLM Reinforcement Learning with Low-Rank Gradient Sketches 

**Authors**: Shaokun Zhang, Yifan Zhang, Jian Hu, Yueying Li, Hao Zhang, Binfeng Xu, Jan Kautz, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2610.06647)  

**Abstract**: Reinforcement learning (RL) has greatly advanced the capabilities of large language models (LLMs), but its memory demands remain a barrier to broader adoption. We introduce LoGRA, an approach to RL post-training that reduces memory by retaining useful learning signals in low-rank gradient sketches. These compact representations support both model updates and efficient policy synchronization. To prevent overly large updates from disrupting learning, we complement gradient compression with predicted-KL step control, which estimates policy changes before applying each update and adjusts its magnitude accordingly. Across reasoning tasks, LoGRA reduces average training memory by up to 45.7\% without sacrificing performance. It also enables stable training of a 27B-parameter model for over 1,100 steps on a single eight-GPU node, where dense Adam runs out of memory, making previously memory-infeasible RL training practical. Code is available in the \href{this https URL}{Molt library}. 

---
# Long-Horizon Textual World Modeling through Structured Reasoning 

**Authors**: Fangxin Wang, Xiang Gao, Yuguang Yao, Kaiwen Dong, Nikash Walia, Kamalika Das  

**Link**: [PDF](https://arxiv.org/pdf/2610.06637)  

**Abstract**: World models must predict how an environment evolves under sequences of actions, enabling agents to compare possible futures and reason about counterfactual actions before acting. Long-horizon prediction is commonly obtained by recursively applying a one-step transition model, but intermediate errors can compound over time. Multi-step dynamics models instead condition on a sequence of future actions and predict their consequences directly, but become harder to learn as horizon grows: the model must track interacting state changes across the trajectory, endpoint supervision provides weak credit assignment, and intermediate predictions can remain plausible while losing information needed for later states. We show that these challenges can be addressed by casting the internal evolution of a multi-step transition as structured reasoning over textual world states: reasoning over sparse state changes reduces the burden of state tracking, a predictive-gain objective rewards the learned state for improving over a matched predictor that conditions on raw history instead, and intermediate predictive rewards supervise each state along the trajectory. Because these intermediate states are explicit textual representations of the world, they provide semantically meaningful targets that can be inspected, scored, and corrected during training. Across ScienceWorld, Jericho, and CEO-Bench, our approach achieves the strongest average long-horizon performance against recursive and non-recursive baselines that condition directly on raw history, with gains increasing at longer horizons. In a controlled counterfactual study, our model is also the only one with statistically significant sensitivity to future actions. 

---
# JEV versus LLMs: Accuracy, Cost and Calibration on Seven Political Science Replications 

**Authors**: Matthew DiGiuseppe, Steven Denney  

**Link**: [PDF](https://arxiv.org/pdf/2610.06625)  

**Abstract**: Large language models (LLMs) annotate and scale political text or constructs by generating text tokens. A new class of models, which TypeSafe markets as "System One" models, instead returns decisions and probability distributions across a user-supplied fixed answer set. A commercial model, JEV, is advertised as having a dramatic cost and speed advantage over traditional LLMs along with better calibrated decisions. As such, it might be useful for social scientists looking to quickly and cost-effectively annotate or scale large corpora of text and have a reliable indicator of a classifier's uncertainty. Yet, the accuracy of these claims and the broader model accuracy in social science text-based tasks are not yet established. In this paper, we do just that and hope to establish the suitability of JEV for social science tasks. We compare JEV with LLMs and human coders from published research, and with a current mid-tier commercial LLM (GPT-6 Luna) and an open-weight alternative (Qwen3.8-27B). We find that JEV matches, or comes close to, the capabilities of both LLMs in a variety of tasks. However, we find no cost advantage over GPT-6 Luna at OpenAI's batch prices. Further, we find that, when each question is asked once, JEV's probabilities are better calibrated than GPT-6 Luna's token probabilities, but not consistently better than Qwen3.8-27B's. We conclude that unless researchers have a need for speed, JEV's only obvious advantage is ease of parsing the underlying choice probabilities. 

---
# Word-Level Text Unmixing via Evidence-Preserving Ownership Routing with Language Models 

**Authors**: Jinglin He, Siyang Jiang, Lixing He, Guoliang Xing, Hongkai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.06603)  

**Abstract**: Text from multiple sources can become interleaved into a single sequence when attribution metadata is lost, such as overlapping speech transcripts, document reading flows, or concurrent agent streams. We formalize this challenge as Word-Level Text Unmixing: given an interleaved lexical stream and source count K, recover the original source sequences while preserving every word occurrence and its within-source order exactly. Directly generating separated texts with LLMs can omit, duplicate, or hallucinate words, violating this exact-reconstruction objective. We therefore propose Evidence-Preserving Ownership Routing (EPOR), which decouples source-ownership prediction from reconstruction. EPOR adapts a causal LLM to predict canonical ownership routes conditioned on the mixed stream and prior routing decisions. At inference, completion-safe constrained decoding is combined with deterministic indexed reconstruction, yielding structurally valid K-source partitions that preserve every observed occurrence exactly once. We also introduce UNMIXBENCH, covering controlled synthetic mixtures, timestamp-derived speech from AMI and ICSI, layout-derived document streams from ReadingBank, and simulated concurrent digital outputs. Across five evaluation tracks, a 4B EPOR model achieves the lowest mean minimum-permutation word error rate among finetuned baselines, reducing the five-track mean by 22.3% relative to compact source-array generation and remaining competitive with zero-shot frontier LLMs. These results show that when lexical evidence is fully observed, separating ownership inference from lexical regeneration provides a reliable alternative to direct generation. 

---
# Molecules of a Story: Community Detection in PMI-weighted Narrative Networks 

**Authors**: Kasper Fyhn, Rebekah Baglini  

**Link**: [PDF](https://arxiv.org/pdf/2610.06600)  

**Abstract**: Automatically extracted narrative networks -- graphs with entities as nodes and their relations as edges -- have proven useful for revealing central narrative structures through salient entities and their connections (Tangherlini et al. 2020; Labatut and Bost 2019). But a narrative is more than those central structures that everything else revolves around. This work is concerned with the everything else: brief sub-plots, small clusters of descriptions, or associations between minor characters that go under the radar at the macro-level. We present an approach to unearth such peripheral structures. They involve rare entities with limited textual presence, overshadowed by dominant entities and lost among each other in the long tail of many but rare entities (Baayen 2001). We leverage the known tendency of pointwise mutual information (PMI, Church and Hanks 1990) to inflate for rare events, turning its weakness into a strength by weighting edges with PMI to foreground peripheral entity configurations. Communities extracted from the resulting network are structural traces of underlying narrative elements. We demonstrate the approach on The Lord of the Rings. From measures of how concentrated or dispersed a community's activations are across the text, a typology emerges that reveals that peripheral structures form more than a single class: episodic passages, echoing long-distance connections, and recurring threads each surface as distinct configurations. The approach is conceptually simple and surfaces fine-grained narrative details that are lost in abundance, though its deliberate amplification of weak signals comes with inherent sensitivity -- best understood as a lens for exploration rather than a robust extraction pipeline. 

---
# Before Agent Tells The Lie: Has Deception Already Been Represented? 

**Authors**: Xinling Li, Dadi Guo, Qingyu Liu, Qinghua Mao, Yi R. Fung, Na Zou, Xia Hu, Dongrui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06576)  

**Abstract**: Large language model (LLM)-based agents can exhibit deceptive behavior during task execution, including hiding failures, fabricating results, or falsely signaling task completion. Existing monitoring approaches mainly detect deception after it appears in observable actions or outputs. In this paper, we investigate whether deceptive behavior can be predicted from an agent's internal representations before it becomes externally visible. We frame deception monitoring as a trajectory-level representation analysis problem and align agent trajectories around key decision points. Using hidden states extracted before these points, we show that future honest and deceptive outcomes can be reliably distinguished, with predictive signals remaining detectable several model calls before the final decision. We further characterize the temporal evolution of these signals: deception-related representations are weak early in execution but become increasingly identifiable as trajectories progress, while transferable structure can emerge before the strongest decision-adjacent signals appear. Finally, we intervene on the identified honest-deceptive representation directions during inference and find that activation steering reduces downstream deceptive behavior, suggesting that these representations influence agent decisions. Our findings indicate that agent deception is an evolving internal process that can be detected and potentially mitigated before it is expressed externally. 

---
# Anatomy of LLM Sycophancy: What a Flip Rate Hides 

**Authors**: Haonan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06522)  

**Abstract**: A model under pushback can correct itself, capitulate, or hold, and one flip rate counts a correction and a capitulation alike. Using SycoLens, a modular replay protocol, we test how user pressure and evaluation settings shape measured flip rates. Each measurement is one stateless replay of an item, a committed answer, and one scripted user line in a fixed form. Every effect is read against a matched control with the line deleted. Pushback wording, committed text, answer format, boundary distance, and ground truth become factors of one instrument; earlier instruments vary one to three of them. Across eleven frontier models from three providers and about 760,000 controlled replays, which models look sycophantic depends on how the user pushes back. Lines that assert the opposite verdict and lines that challenge the answer without asserting one rank the models almost unrelatedly. Flip effects grow several-fold near a model's boundary, yet items answered identically in every screening draw still carry about half of the most-affected totals. On arithmetic tasks where the truth is known, one model re-derives and corrects itself under pressure while another abandons correct answers without written work. On the model tested, a planted derivation lowers release of the answer it argues for, true or wrong, where a bare stated value does not; the wrong answer is corrected much more often than the true one is abandoned. Under a yes/no readout the rankings come closer, entangled with a pressure-induced shift toward "no". One score per model therefore compares different behaviours across models and benchmarks. We condense these dependencies into a reporting profile; the instrument, records, and analyses will be released upon publication. 

---
# Test-Time Adaptation of Reasoning Strategies with Bayesian Nonparametric Memory 

**Authors**: Keshav Ramji, Tahira Naseem, Ramón Fernandez Astudillo  

**Link**: [PDF](https://arxiv.org/pdf/2610.06516)  

**Abstract**: While modern large language models (LLMs) have been trained to reason through verbalized chains-of-thought, the generation cost grows substantially due to suboptimal paths to reach the final answer. Furthermore, as new insights are discovered while observing various input queries (e.g. through self-reflection), limited mechanisms exist for carrying forward these findings to be applied to subsequent problems. One can view the list of such strategies or behaviors as a growing cheatsheet, with elements retrieved from this memory module at inference-time. In this work, we consider structured cheatsheets, with learned clusters of behaviors. We introduce a Hierarchical Dirichlet Process Gaussian Mixture Model (HDP-GMM) over behavior embeddings, which shares components across domains while allowing domain-specific mixing weights, and uses the posterior predictive to retrieve relevant behaviors for a query; we call this a $\textit{Bayesian Cheatsheet}$. This mechanism allows for cheap adaptation in an online test-time training (TTT) setting, softly updating the mixture's sufficient statistics following each sample and enabling the creation of new components when the synthesized behaviors are sufficiently novel. We demonstrate that Bayesian Cheatsheet achieves clear performance gains relative to existing memory modules across reasoning benchmarks such as AIME'25, Omni-MATH, and PhysReason, even in the cold-start setting. We show that the Bayesian Cheatsheet is an adaptively reorganizing memory module, as behaviors can be re-assigned to components through a single step of collapsed Gibbs sampling. Our findings highlight the value of Bayesian-inspired memory modules for effective test-time adaptation and the role of structure in metacognitive reasoning. 

---
# SOL: Measuring Gaps between Text Distributions by Double Sliced Wasserstein Metrics 

**Authors**: Gregor Kornhardt, Moritz Piening, Jannis Chemseddine, Gabriele Steidl  

**Link**: [PDF](https://arxiv.org/pdf/2610.06513)  

**Abstract**: Evaluating text generation requires measuring how well the generated distribution matches the data distribution. For autoregressive models, this is done by the perplexity. Diffusion and flow-based language models can only provide a likelihood bound, whose tightness differs between model families. Sample-based substitutes such as generative perplexity with entropy do not consider the distribution fit. We propose SOL,
a distance between text distributions. Each sequence is represented by the empirical measure of its hidden states under a fixed transformer and the distributions of these measures are compared by the double sliced Wasserstein distance. We prove that SOL is a metric if the transformer is injective. Experiments show that SOL detects distributional failures, recovers expected model trends, and provides stable sample-based estimates. We put forward SOL to fill the gap in the current evaluation protocol used for non auto-regressive models. As a first step we use SOL to re-evaluate a variety of models trained on OpenWebText. 

---
# AECP: Artifact-Exclusive Communication Protocol for Multi-Agent Code Generation 

**Authors**: Jiaqi Xue, Yanjun Wang, Xiangci Li, Lingbo Mo, Aritra Sengupta, Shweta Garg, Murali Krishna Ramanathan, Myeongsoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.06481)  

**Abstract**: As AI agents increasingly tackle complex repository-level coding tasks, distributing work across multiple agents is a natural way to scale beyond the capabilities of a single agent. To coordinate their interdependent work, these agents share findings and agree on interfaces between modules. However, exchanged information often serves only as context, leaving individual agents to interpret it and incorporate it into subsequent work. Consequently, shared findings may go unused and deviations from interface agreements may go undetected, undermining the reliability and efficiency of collaboration. This motivates moving part of the coordination responsibility from individual agents to the execution harness. To make shared information actionable during execution, we introduce the Artifact-Exclusive Communication Protocol (AECP). AECP requires agents to communicate exclusively through structured artifacts and specifies how the harness processes them. The harness supplies findings when agents access relevant code, screens implementations for mismatches with recorded interface commitments, and requires affected agents to revisit revised agreements. These coordination steps become part of harness execution rather than actions that agents must initiate from prior messages. Across Doc2Repo, NL2Repo, and CodeProjectEval, using closed- and open-source models including Opus-4.8 and DeepSeek-V4-Flash, AECP improves average test pass rate by 28.2% and reduces average wall time by 16.5% relative to an agent team using free-form inter-agent messages. Artifact-exclusive communication also blocks the relay of malicious instructions between agents, reducing how often they reach other agents from 95% to 0% and how often those agents act on them from 40% to 0%. 

---
# Behavior-Preserving KV Cache Compression 

**Authors**: Doo Hwan Hwang, Junyoung Jang, Junho Na, Hosung Lim, Kee-Eung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.06479)  

**Abstract**: KV caches are a major bottleneck in long-context inference and long-form generation with large language models. Existing training-free eviction policies largely rely on proxy importance signals, such as attention mass, to decide which past tokens to retain. We argue that cache compression should instead preserve the predictive behavior of the full-cache model, retaining entries whose removal would substantially change the model's output distribution. We propose Behavior-Preserving KV Cache Compression, a training-free framework that scores candidate evictions by estimating the compressed-cache logits induced by their removal and evaluating the resulting KL to the full-cache next-token distribution. Using pre-eviction forward statistics, the method avoids running separate masked forward passes for each candidate. Across diverse architectures and both prefill-time and generation-time compression, our method delivers substantial gains in downstream task quality over lightweight attention-based heuristics at matched retained-KV budgets, with the largest gains under aggressive compression. It achieves these gains with additional compression-time computation while retaining an end-to-end speedup over full-cache inference in our evaluated settings. 

---
# The Assistance Dilemma: Learning to Teach via Multi-Turn Reinforcement Learning 

**Authors**: Jakub Macina, Manu Kapur, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2610.06446)  

**Abstract**: Large language models (LLMs) trained to answer questions are natively poor at teaching. Reinforcement Learning (RL) against a simulated student is a promising approach to improve their pedagogy, but existing RL-trained tutors reward the student's success on the tutored problem with the tutor's words still in context. The reward is then easiest to raise by telling the student the answer, and a tuned penalty is needed to reduce telling. Drawing on learning sciences, we introduce a masked near-transfer post-test: the student is tested on an unseen variant of the tutored problem with the tutor's utterances masked, so the reward can rise only through what the student wrote in its own turns. This discourages cognitive offloading by the student and allows the continuous penalty to be replaced by two binary reward gates (factual correctness of tutor response, no solution handover). A leave-one-out ablation shows that the learning-gain reward on its own does not separate teaching from telling: the gates reduce solution handover while the near-transfer post-test improves out-of-domain transfer. Using these reward designs we develop Eduardo, a multi-turn RL recipe for training LLM tutors, and use it to train 4B, 9B, 14B and 27B models from two distinct LLM architectures. Our post-trained Eduardo-27B model matches Gemini-3.1-Pro on MathTutorBench and Claude Opus 4.8 on TutorMoments at 2.4-6.2x fewer thinking tokens than frontier models, which matters for interactive tutoring. Without being named in the reward, the model more than doubles its use of the push-for-justification teacher move while support fading (e.g., assigning independent work), whose payoff lies beyond a single-problem dialog episode, is trained out. We open-source our training environment, an 8,671-problem near-transfer dataset, and trained models for further development. 

---
# Do Speech Representations Preserve Regional Accent Across Read and Spontaneous Speech? 

**Authors**: Paula A. Perez-Toro, Tomas Arias-Vergara, Annette Schwarz, Abner Hernandez, Andreas Horr, Cornelia Kristen, Andreas Maier  

**Link**: [PDF](https://arxiv.org/pdf/2610.06430)  

**Abstract**: Regional accent cues can be captured under matched conditions, but it remains unclear whether they persist between read and spontaneous speech. We study RVG1, with 500 German speakers from nine regions, comparing ten speech representations on regional classification and continuous geolocation under matched conditions and speaker-independent read--spontaneous transfer. Whisper performs best under matched conditions, reaching 0.489 nine-way UAR and 148 km median geolocation error, but drops to 0.11/0.18 UAR across transfer directions and 363 km geolocation error. Self-supervised models show a similar degradation, whereas speaker embeddings are less discriminative in-domain but more robust under transfer. This contrast is consistent across classification and geolocation. Across representations, robustness is associated with how little a representation shifts between styles (style-invariance), for which crossstyle speaker retrieval is an interpretable proxy. Age, sex, sentence-overlap, and duration controls do not account for the gap, although channel characteristics contribute. These results show that strong matched-condition performance does not indicate robust regional information. 

---
# Ontology Concept Overlap as a Training Signal: Knowledge-Grounded Reinforcement Learning for Clinical Question Answering 

**Authors**: Aditya Tanna, Abhishek Jindal  

**Link**: [PDF](https://arxiv.org/pdf/2610.06360)  

**Abstract**: Reinforcement learning post-training for language models relies on two reward designs: human preferences (RLHF, DPO) and binary verifiers (RLVR). Clinical question answering fits neither. Near-correct answers differ by a single substituted entity, and no executable check decides clinical correctness. We instantiate a soft verifier from a maintained controlled vocabulary: UMLS Concept Unique Identifier overlap (via scispaCy, set-level F1) gives a graded, externally specified reward computed without a model in the loop. We combine it inside GRPO with an entropy-normalised LLM judge, which covers the safety and evidence axes overlap cannot see, and a small consistency penalty on padding and repetition that keeps early-training samples scorable. This three-term composite improves over SFT on Phi-3-mini (3.8B) over MedQA by 2.9% on EM (0.700 vs 0.680) and 39% on Token-F1 (0.202 vs 0.145); on Llama-3.2-3B the corresponding gains are 14% on EM and 35% on Token-F1. We report Token-F1 as the primary metric because it credits partially-correct clinical content that EM discards at this open-generation scale. Main-table results are means over 3 seeds with standard deviations below 0.005. The method transfers to PubMedQA, where training on the PubMedQA train set with the same composite reward improves Token-F1 over SFT by 22% on Phi-3-mini and 17% on Llama-3.2-3B without retuning. A reward ablation on Phi-3, varying the judge-ontology split at a fixed consistency weight, attributes 3 EM points to the ontology term, the contribution that catches entity substitutions the judge cannot. Three negative findings constrain the design: DPO under random negatives underperforms SFT for strong-prior models but helps the weakest-prior one; PPO under a sparse neural reward diverges; GRPO with KL-in-loss collapses at 7B. 

---
# Breaking Bureaucracy: Evaluating open-source LLMs for legal document review 

**Authors**: Farrukh Baratov, Niki van Stein, Suzan Verberne  

**Link**: [PDF](https://arxiv.org/pdf/2610.06345)  

**Abstract**: In this paper, we evaluate open-source generative LLMs on legal Natural Language Inference (NLI). Legal inspectorial processes take place in specific domains and often deal with confidential data. This creates a need for working with local models that do not require labeled training data. We evaluate our models on the ContractNLI benchmark and two NLI4Wills datasets. We successfully reproduce the baseline for the task (Span NLI BERT) and we evaluate multiple open-source LLMs on the same task. We analyze the invalid rate of the models, and their stability across temperature settings and domains. Among the generative models, Gemma-4 26B performs the best, reaching an accuracy of 81.2%, even outperforming the supervised model on one metric. On accuracy, it is not possible to beat the supervised model with zero-shot approaches. Qwen-3.6 35B performs well on both ContractNLI and additional datasets in the legal wills domain. Our findings indicate that zero-shot, open-source, generative LLMs are a viable alternative for real-world legal NLI when no supervised data is available. Our code is available at this https URL. 

---
# Agentic schema-guided extraction of materials process knowledge from scientific literature 

**Authors**: Sameer Sadruddin, Jennifer D'Souza  

**Link**: [PDF](https://arxiv.org/pdf/2610.06322)  

**Abstract**: Materials literature contains detailed experimental knowledge, but procedures, chemical entities and measurements remain difficult to aggregate because they are reported in heterogeneous forms and depend on process-specific context. We present SciKGExtract, a schema-guided framework that combines large-language-model extraction with chemical normalization and agent-based evaluation and refinement before knowledge-graph integration. We evaluate the framework on 176 atomic-layer-deposition papers describing zinc oxide (ZnO) and indium--gallium--zinc oxide (IGZO), together with an expert-annotated full-schema subset. PubChem normalization improves exact-match extraction F1 for every tested model. For ZnO, the best F1 increases from 0.591 for direct normalized extraction to 0.805 with agentic refinement, whereas the best IGZO result is 0.344, revealing the greater difficulty of multicomponent supercycle processes. Evaluation against a deeply nested schema containing 65 experimental properties and 155 quantitative measurement nodes further exposes errors in process segmentation and numerical assignment. These results show that chemical canonicalization and targeted agentic verification provide complementary controls for converting complex materials literature into reusable, machine-actionable experimental knowledge. 

---
# DialectSentEval 2026: Arabic Dialect Sentiment Analysis and Swapping Shared Task 

**Authors**: Saad Ezzini, Shadi Abudalfa, Maram Alharbi, Salmane Chafik, Hind Alatawi, Mo El-Haj, Ahmed Abdelali, Osamah Alnahari, Salima Lamsiyah  

**Link**: [PDF](https://arxiv.org/pdf/2610.06298)  

**Abstract**: Sentiment analysis is a fundamental problem in Natural Language Processing (NLP). Standard sentiment classification for the Arabic language remains challenging due to the high volume of dialectal Arabic. To advance research in this area, this paper proposes the Shared Task on Sentiment Analysis and Swapping in Arabic Dialects (DialectSentEval), hosted with the Arabic Natural Language Processing Conference (ArabicNLP 2026). This shared task consists of two subtasks: Subtask 1 focuses on multi-class and multi-dialect sentiment analysis, requiring models to identify sentiment polarity across various Arabic dialects. Subtask 2 introduces a generative task for Arabic sentiment swap, challenging models to invert sentiment polarity while preserving core semantics. In this overview paper, we present the motivation, dataset creation, and summarize the main findings from participating models. 

---
# From Abusive Language Classification to Sequence Labeling Identification 

**Authors**: Nicolas Zampieri, Ignacio Lopez, Manon Girard, Jeremy Auguste  

**Link**: [PDF](https://arxiv.org/pdf/2610.06287)  

**Abstract**: Industrial content moderation must process massive message streams under tight latency constraints, yet most abusive language (AL) detection systems rely on sentence-level classification (ALC), which neither localizes abusive spans nor identifies who is targeted. We define Abusive Language Identification (ALI) as a sequence-labeling task that jointly extracts AL spans and target mentions, and assess whether this approach can be used for text moderation. On a pilot corpus drawn from a production moderation pipeline, we compare ALI with ALC on cross-domain generalization and implicit abuse, and we also evaluate AL and target span detection. ALI remains competitive with ALC while providing localized outputs for moderators, with a modest and configuration-sensitive advantage on implicit abuse. Exact AL boundaries and target spans remain difficult to recover. We complement this comparison with a qualitative analysis and discuss perspectives on complete target--span linking and on structured benchmarks for ALI. 

---
# DeferKV: Rethinking Eviction Timing for One-Shot KV Cache Compression 

**Authors**: Zhe Wang, Jiakai Li, Yujia Sun, Rongzheng Wang, Shuang Liang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06286)  

**Abstract**: Long-context large language models (LLMs) have demonstrated strong capabilities across a wide range of tasks, but the growing KV cache introduces substantial memory and inference overhead. Existing one-shot KV cache compression methods typically commit to irreversible eviction immediately after prefill, before any signal from actual generation becomes available. Our quantitative analysis shows that early queries from the actual generation stage provide attention signals that are more consistent with subsequent decode attention, with the largest single-step gain occurring at the prefill-decode boundary. Based on this observation, we propose DeferKV, which moves the eviction decision from the end of prefill to the first real decoding step and temporally combines prompt-side and decode-side observations, thereby better aligning KV importance estimation with subsequent generation requirements. DeferKV requires no additional training, draft model, or future-query prediction module, making it simple and easy to deploy. Experiments on LongBench, RULER, and Needle-in-a-Haystack demonstrate that DeferKV consistently improves model performance under KV cache compression while maintaining low inference latency. 

---
# Probabilistic Race and Ethnicity Prediction Using Group-Specific Name Lists 

**Authors**: Kyla Chasalow, Noah Dasanaike, Kosuke Imai  

**Link**: [PDF](https://arxiv.org/pdf/2610.06273)  

**Abstract**: Statistically valid estimation of racial and ethnic disparities often requires inferring the probability that an individual belongs to a particular racial or ethnic group given only their name and geographic location. The standard approach, Bayesian Improved Surname Geocoding (BISG), relies on group population frequencies for each name. Although the U.S. Census Bureau provides such information for common names and a limited set of racial categories, comparable data do not exist for many racial and ethnic groups and are rarely available outside the U.S. We propose the list-powered BISG ($\ell$BISG) method, which can be used to derive calibrated group probabilities from group-specific name lists. These lists may be compiled based on expert knowledge or generated synthetically using large language models (LLMs), and thus may be subject to unknown biases. Representing names as embeddings, we treat list membership as a proxy prediction task and apply a correction based on proximal inference to recover the target group probabilities. We validate the method on U.S. voter files with self-reported race, on the full-count 1900 U.S. Census, and on the Lebanese voter registry. We find that LLM-generated name lists yield accurate and well-calibrated probabilities as well as precise disparity estimates comparable to those obtained using methods that require name-race data. Thus, $\ell$BISG substantially broadens the applicability of probabilistic race and ethnicity prediction to settings where name-race data are unavailable. 

---
# Shared Stopping Decisions Change Answers in HQQ Cache Quantization 

**Authors**: Seunghui Jwa, Minsu Oh, Chanjun Park, Yeo-Chan Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2610.06251)  

**Abstract**: Language-model systems batch questions for throughput, but unrelated questions should not change a target's answer when its input and numerical execution are fixed. We study compression of the key and value cache, which stores attention representations reused during generation. With request-local groups, Transformers' Half-Quadratic Quantization (HQQ) backend updates compression parameters separately but uses a shared average error to decide when all updates stop. Replacing only the question batched with the target changes four-bit HQQ answers in 170/384 test comparisons across two models. Replaying the other execution's update counts reproduces its complete answer and cache fingerprints in every changed pair, in both directions. Computing the stopping mean in FP32 reduces cache differences but leaves answer changes. Native HQQ also changes confirmed numerical correctness in eight arithmetic pairs. Fixed iterations and request-local stopping remove observed companion dependence under matched controls. Request-local stopping remains sensitive to synthetic padding changes at the tensor level. Fixing the original iteration budget removes this decision path without tuning. Neither repair has an established quality advantage, and natural rebatching still changes answers. Request-independence audits must cover stopping decisions as well as quantization groups. 

---
# Cross-lingual Calibration of Pre-Generation Success Probes for Multilingual LLM Routing 

**Authors**: Andrea Paganelli, Stefano Civelli, Pietro Bernardelle, Gianluca Demartini  

**Link**: [PDF](https://arxiv.org/pdf/2610.06216)  

**Abstract**: Pre-generation success probes estimate response correctness from a language model's hidden activations before decoding, enabling cost-aware routing. While prior work has demonstrated their utility primarily on English inputs, we study their reliability across languages along three dimensions: (1) whether they preserve the ranking of likely successes and failures (DISCRIMINATION); (2) whether they retain probabilities that match observed success frequencies (CALIBRATION); and (3) whether they produce scores comparable enough across candidate models for cost-aware multilingual routing (UTILITY). Using 3,000 MATH problems in 10 languages and 8 open-weight model configurations, we compare cross-lingual transfer from English-trained probes and equal-budget pooled multilingual probes. English-trained probes retain useful cross-lingual discrimination but become less well calibrated after transfer. Pooled multilingual supervision improves both properties and yields more reliable estimates of success. In routing experiments, the pooled router achieves a 0.7% higher test success rate while reducing modeled cost by 13.0% relative to always selecting the model with the highest average success. These results show that multilingual routing requires success estimates that remain well calibrated and comparable across languages and models. 

---
# Introducing Code-Switched Contexts to Cognitively-Inspired Bilingual Model Training 

**Authors**: Zhuojing Huang, Luise Pohlmann, Lisa Beinborn  

**Link**: [PDF](https://arxiv.org/pdf/2610.06161)  

**Abstract**: During language acquisition, bilingual children are regularly exposed to code-switched input and use it as a cognitive scaffold to accelerate vocabulary growth and cross-linguistic syntactic mapping. In contrast, computational bilingual models are conventionally pretrained on interleaved monolingual corpora. While introducing synthetic code-switching during pretraining has become a promising strategy to enhance cross-lingual alignment and downstream performance, the structural and developmental parameters governing the success remain poorly understood. In this work, we investigate the efficiency of training with synthetic code-switched data across two typologically distinct language pairs by controlling two key variables: the structural location of code-switches and the dynamic switching rate across training stages. Our results show that training with code-switched data improves cross-lingual alignment for typologically close languages. 

---
# From Traces to Agentic Worlds: Agentic Language World Models for Interactive Environment Simulation 

**Authors**: Quanyu Long, Xiao Chen, Jianda Chen, Haozhen Zhang, Qisheng Hu, Jianzhu Bao, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06100)  

**Abstract**: Realistic environment replicas are increasingly valuable for training and evaluating LLM agents, yet the original systems may be inaccessible or impractical to reproduce. We explore agentic language world modeling: rather than rebuilding an executable environment, a world model agent serves as the environment for a task agent and supports faithful and stateful simulation. We instantiate this paradigm with Trace2Env, a learning-free framework for settings where the original system is unavailable but historical interaction traces remain accessible. Trace2Env reconstructs these traces into a reusable environment worldbook containing environment schemas, grounded evidence, and induced behavioral knowledge. At runtime, the world model agent actively consults the worldbook together with persistent episodic state to infer each action's observation and lasting state effects. Across nine environments, Trace2Env improves both next-observation fidelity and long-horizon interaction consistency over conventional prompt-based LWMs. In multi-turn interaction, task agent actions generated against Trace2Env remain valid more often when replayed in the real environment, indicating that its simulated dynamics better preserve the consequences of earlier actions across successive turns. These results establish agentic language world modeling as an alternative direction for building realistic environment replicas without reconstructing the original executable system. 

---
# Cross-Lingual Transferability of Training Data Extraction Attacks to Recover Memorized PII 

**Authors**: Alexandru Nazare, Agnese Profico, Nicolò Vania, Elena Di Croce, Daria Caramanica, Davide Venditti, Elena Sofia Ruzzetti, Giancarlo A. Xompero, Fabio Massimo Zanzotto  

**Link**: [PDF](https://arxiv.org/pdf/2610.06093)  

**Abstract**: The robustness of Personally Identifiable Information (PII) protection in Large Language Models (LLMs) is a critical concern, yet the risks associated with cross-lingual data extraction remain under-explored. This study evaluates the vulnerability of English-centric and multilingual models to Training Data Extraction (TDE) attacks when prompted in non-English languages.
We construct a multi-domain PII dataset comprising social media handles, email addresses, and phone numbers and translate the attack contexts into Italian, Spanish, French, and German. Our results show that TDE attacks against both English-centric and multilingual models transfer to different languages: the attacks are successful on translated prompts, even though only the original English prompt might have been included in the pre-training data. A web-presence check on a sample of the translations confirms that they are not available online. The share of English leaks recovered in other languages grows with the multilingual capability of the model, and it drops sharply when the original wording is lost, even without a change of language.
This suggests that native multilingual pre-training facilitates the emergence of latent cross-linguistic bridges that simplify the retrieval of personally identifiable information (PII). We analyze the activations of multilingual large language models (LLMs) and find that different translations of the same prompt are bridged in similar representations, with the strongest alignment in the middle layers.
Our results highlight a fundamental security gap in modern LLMs, necessitating more robust, language-agnostic sanitization strategies for future model alignment. 

---
# TrustMI: Causally controlling how assistants trust their users 

**Authors**: Théo Lasnier, Romain Froger, Maxence Lasbordes, Djamé Seddah  

**Link**: [PDF](https://arxiv.org/pdf/2610.06064)  

**Abstract**: Large Language Model (LLM) assistants routinely decide whether they can trust users and third parties whose competence, intentions, and integrity they cannot verify. This uncertainty matters for safety, as trusting the wrong party can lead an agent to comply with harmful requests or act on malicious instructions encountered during tool use. To study this problem, we define trust as an assistant's willingness to accept vulnerability to the actions of another party and ask whether such behavior can be causally controlled through model activations. We build 2,000 contrastive conversations spanning ability, benevolence, and integrity, where paired responses complete the same request but differ in whether the assistant trusts the user. From these pairs, we learn steering matrices while keeping the model parameters frozen and test them across six instruction-tuned models from three families, finding that steering changes trust decisions monotonically in both directions. We then ask whether this effect extends to several safety-related agent settings involving harmful requests, prompt injections, and insider threats, while using benign-task and reasoning as controls. Our findings provide evidence that trust in the user can be causally controlled along linear directions in model activations and provide a way to study how trust shapes safety-relevant behavior in language models. 

---
# LightMTP: Lightweight Latent Multi-Token Prediction 

**Authors**: Tamara Czinczoll, Julie Kallini, Gerard de Melo, Chen Shani  

**Link**: [PDF](https://arxiv.org/pdf/2610.06031)  

**Abstract**: Next-token prediction (NTP) is the standard pretraining objective for large language models, yet it provides an explicit training signal only for the immediate next token, which can lead models to exploit local patterns instead of capturing longer-range structure and ideas. Multi-token prediction (MTP) addresses this by training models to predict several future tokens. However, existing MTP methods often introduce a large number of new parameters with limited improvements in downstream performance. Latent MTP approaches address this efficiency issue by encoding future tokens into a vector representation. However, these approaches usually rely on external helper models for future token encoding. We propose LightMTP, a lightweight, i.e., parameter-efficient, latent MTP approach that bootstraps the future token representations from the model's own hidden states. Our two LightMTP variants extend supervision to more future tokens without requiring the additional computational overhead of conventional MTP nor the external supervision latent MTP normally relies on. LightMTP adds at most 1% extra parameters, retains better performance on general language modeling benchmarks, and achieves similar gains in planning, coding, and reasoning. 

---
# Differentiable Bit-Widths: Co-optimizing Pruning and Quantization via SVD for Ultra-Efficient LLM Compression 

**Authors**: Hankyul Kang, Jongbin Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06026)  

**Abstract**: SVD-based pruning and quantization have recently emerged as a promising strategy for the ultra-efficient compression of large language models. In these methods, compression is performed in two stages: components are first truncated, and the remaining ones are subsequently quantized. Although this decoupled pipeline benefits from both pruning and quantization, it requires separate optimization for each stage and fails to fully exploit their balance, which can lead to suboptimal performance under aggressive compression. To address this limitation, we propose a new LLM compression method that co-optimizes pruning and quantization in a unified framework. Our key idea is a differentiable method for learning component-wise bit-widths, allowing less important components to be assigned 0-bit precision and pruned away. Notably, our method performs favorably against two-stage baselines, even when subjected to extreme quantization settings ($1.61$ bits) designed for ultra-efficiency. Code: this https URL. 

---
# D-Loop: Looped Diffusion Drafting for Speculative Decoding 

**Authors**: Kecheng Chen, Yuyang He, Cheng Gong, Hui Liu, Guoping Long, Jiajun Li, Shi Wu, Suiyun Zhang, Haoliang Li, Ziru Liu, Rui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06011)  

**Abstract**: Block diffusion accelerates speculative decoding by drafting multiple tokens in one forward pass. However, each position predicts a marginal distribution without observing earlier proposed tokens, limiting draft quality and acceptance length. We identify a concrete failure, the \emph{repetition trap}, in which neighboring positions produce redundant copies of the same token. We explain this tendency theoretically and empirically examine its association with shorter accepted drafts. Recent methods refine marginal predictions with an additional causal head or a separately trained drafter, increasing parameter storage and introducing separate training objectives. We instead propose D-Loop, which introduces \emph{intra-block causal conditioning} within the original diffusion drafter without additional model components. Inspired by semi-autoregressive generation and parameter sharing, D-Loop reuses the same backbone across looped passes. The first pass proposes a block, and the second conditions on a selected prefix to regenerate the suffix in parallel. A complementary prefix--suffix objective trains the shared drafter for both anchor-only prefix prediction and prefix-conditioned suffix prediction. Across eight math, code, and chat benchmarks, D-Loop can beat DFlash and DSpark on Qwen3-4B and Qwen3-8B with obvious gains. 

---
# Breaking the Tie: A Cluster-Aware Routing Framework for Large Language Models 

**Authors**: Yao Lu, Zhaiyuan Ji, Yaxin Gao, Zeyu Wang, Zhe Tang, Jiaheng Wei, Zhaowei Zhu, Shanqing Yu, Qi Xuan  

**Link**: [PDF](https://arxiv.org/pdf/2610.05982)  

**Abstract**: With the rapid development of artificial intelligence, the emergence of various Large Language Models (LLMs) has created a rich model ecosystem. However, this also brings a key challenge: how to select the optimal model for a specific user query. LLM routing addresses this need by dynamically assigning queries to the most suitable expert in the pool of candidate models. However, existing routing frameworks often simplify this process to a standard classification task; thus, a critical vulnerability is exposed when multiple candidate models correctly answer the same query. We formalize this capability overlap as routing noise, which misleads the router with arbitrarily correct candidate models, ultimately leading to routing collapse (a severe decline in generalization ability on unseen tasks). To address this problem, we propose a novel Cluster-Aware Soft-Labeling Routing (CASLR) framework. CASLR shifts the evaluation paradigm from the success of a single query to macro-domain consensus by replacing traditional one-hot vectors with a masked softmax mechanism. Specifically, for experts who answer incorrectly, we penalize their target probability to zero; for the remaining candidates, we directly compute continuous fine-grained soft labels based on their global clustering utility scores. We then use these refined soft labels to supervise a lightweight router. Specifically, the framework not only demonstrates superior accuracy on multiple benchmarks, but also outperforms Llama-3.3-70B-Instruct by 7.80% in overall average performance. Furthermore, the extremely low routing inference latency of only 1.13s further confirms that CASLR can achieve efficient system scheduling with almost zero additional overhead, while ensuring high response quality. 

---
# Byte Language Models: Scaling, Emergent Abstractions, and Information Allocation 

**Authors**: Jie Wang, Shiwei Luo, Qi Zhang, Yuanbin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05978)  

**Abstract**: Tokenizer-free language models remove the inductive bias of fixed tokenizers by modeling text directly as bytes, but the resulting longer sequences substantially increase computation and eliminate explicit text abstractions. We ask whether this additional computation can be useful, and whether standard Transformers can learn the abstractions that tokenization provides. We study these questions on Transformers without specialized tokenization-related architectures. With token-superposition training and hash embeddings, byte Transformers consistently outperform subword Transformers as model size scales. We further find that byte Transformers build local text abstractions as external tokenizers: a set of segmentation-like positions are used to collect local context representations, and restricting up to $25\%$ of intermediate layers to these local representations preserves downstream performance. Finally, these learned structures induce highly non-uniform generation difficulty, with uncertainty concentrated near local structure boundaries; exploiting them for speculative decoding yields $3.4\times$ more accepted tokens than in subword Transformers. 

---
# Can Language Models Learn to Reject Their Own Bad Reasoning Steps? 

**Authors**: Siheng Xiong, Xiaoze Liu, Yiqiao Jin, Xiaoqian Wang, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.05976)  

**Abstract**: Verifier-guided decoding can prevent harmful reasoning steps from contaminating subsequent generation, but typically relies on an external learned verifier. We ask whether a language model can instead reject its own bad reasoning steps. We define a prefix's recoverability as the probability that the frozen generator can complete it correctly. Diagnostics show that adjacent recoverability changes are often difficult to resolve with practical Monte Carlo budgets, while same-prefix candidates exhibit a sparse low-recoverability tail. We introduce Self-Step Rejection (SSR), which trains a lightweight LoRA acceptance gate on the generator backbone while keeping the base model frozen. SSR uses confidence-qualified first-passage supervision: steps before the first resolved crossing of a root-relative recoverability barrier are accepted, the crossing step is rejected, and unresolved steps and suffixes are excluded. Training combines pointwise classification, same-prefix pairwise learning, and group-relative policy refinement using final-answer correctness. At inference, SSR accepts candidates or resamples from the unchanged prefix under rejection budgets, without an external learned verifier. Across three reasoning models and five mathematical reasoning benchmarks, SSR improves macro-average accuracy over single-pass decoding by 5.4--10.1 points using 1.21--1.40x as many generated tokens, and achieves the highest macro-average accuracy among evaluated step-level methods. Full-solution scaling methods require 4.47--8.27x the single-pass token cost for comparable performance. 

---
# HuatuoGPT-3: RL-Only Domain Adaptation from Base Models 

**Authors**: Junying Chen, Xinyuan Xie, Ziniu Li, Wenyuan Gu, Jianquan Li, Xiang Wan, Guangjun Yu, Ruoyu Sun, Haizhou Li, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05966)  

**Abstract**: Domain adaptation aims to turn a general-purpose large language model (LLM) into an expert for a target domain. While the dominant SFT+RL pipeline offers a convenient cold start, it may reduce exploration diversity and introduces additional complexity through multi-stage optimization. These limitations motivate RL-only adaptation. However, pure on-policy RL suffers from a cold-start problem, while mixed-policy RL still falls short: informative tokens in teacher outputs are learned too slowly in early training, and stale teacher outputs can hinder later improvement. We identify these two failure modes as Gradient Starvation and Teacher-Distribution Anchoring. To address them, we propose One-stage Policy Optimization (OnePO), which treats teacher outputs as transient guidance for policy improvement. OnePO combines Adaptive Objective Evolution to strengthen learning on informative low-probability teacher tokens and Teacher Retirement to discard teacher outputs once the current policy can surpass them. On medical adaptation, OnePO achieves 67.2 on HealthBench (Total) with only 20K training samples, outperforming SFT+RL and pure RL by 2.7 and 7.4 points, respectively. We further scale OnePO to produce HuatuoGPT-3, an open-source medical LLM series whose 27B variant reaches 70.1 on HealthBench (Total) and 71.4 on HealthBench Professional, surpassing frontier models such as GPT-6 Astra. Models and code are available at this https URL. 

---
# Noise Out, Bias In: Targeted Bias Injection in Diffusion Language Models via Closed-Loop Activation Steering 

**Authors**: Sarim Hashmi, Mukul Ranjan, Abdelrahman Elsayed, Muhammad Umer Sheikh, Fahad Shamshad, Nils Lukas  

**Link**: [PDF](https://arxiv.org/pdf/2610.05894)  

**Abstract**: Masked diffusion language models (dLLMs) generate text by iteratively denoising masked positions, re-predicting each token multiple times before it is committed. An autoregressive decoder exposes an answer's distribution once, at the step that commits it; a dLLM exposes it at every denoising step before commitment, and we show that an adversary can exploit this. Since an answer remains open to revision over many denoising steps, an adversary with access to internal activations can watch how likely the model is to produce a chosen answer and adjust the intervention accordingly. Building on this observation, we study targeted bias injection, an attack that steers a frozen dLLM toward a demographic answer selected by the adversary. The attack uses a simple proportional-integral (PI) controller that tracks the target-answer probability during denoising and adapts the strength of a steering vector on the fly. On ambiguous BBQ questions where the correct answer is abstention, our attack raises LLaDA-8B-Instruct's preference for the targeted group from 1.8 to 16.7 percentage points, more than three times the strongest fixed-strength steering baseline, and on SocialStigmaQA it raises the selection of stigmatizing answers from 17.6% to 58.1%. Fitted to other demographic targets, the same attack shifts answers by up to 37 percentage points, and each attack takes about 40 minutes on one GPU. On the primary target, feedback is what makes the attack work: constant steering at the same average strength over the token-committing steps produces a far smaller shift while corrupting nearly three times as many outputs, and a constant strength set separately for each example still falls well short. Our findings identify the denoising trajectory as a new control channel in dLLMs and call for bias audits that examine the serving stack rather than the frozen model alone. 

---
# Learning to Learn a Language 

**Authors**: Lennart Carstens-Behrens, Holger Fröhlich  

**Link**: [PDF](https://arxiv.org/pdf/2610.05879)  

**Abstract**: We present the Prior-Fitted Language Model (PFLM), a 300M-parameter byte-level transformer pretrained only on samples from a synthetic non-linguistic prior. Given a prefix of real text, it learns to predict the language in context with frozen weights, having never seen a word of any real language. Every training sequence is generated by a recurrent structural causal model drawn fresh from a distribution over such models. The model never sees the same language twice during training, so the only way to predict the continuation is to infer the language from the prefix. Samples from this prior share the statistical signatures of natural text: Zipfian frequencies, slow entropy-rate convergence, and long-range dependence. On Wikipedia in six languages, bits per byte fall from the uniform eight to between 0.9 and 2.4 at one million bytes of context. Given numerals instead of text, PFLM learns to count, to compare magnitudes, and to add approximately. It predicts deterministic sequences like Rudin-Shapiro or the prime indicator, and it compresses six non-text domains, from source code to speech, below gzip and PPMd. The model has not learned a language. It has learned to learn one. 

---
# HLA: Expressive Hybrid Linear Attention via Chunk-Wise Dynamic Mixing 

**Authors**: Zhuokun Chen, Xi Lin, Xiyu Wu, Jiahao He, Jianfei Cai, Bohan Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05842)  

**Abstract**: Linear attention enables efficient long-context autoregressive decoding by compressing history into recurrent states, but this compression can make selective access to sparse and distant information difficult. Existing chunk-based extensions increase memory capacity, yet learned chunk-mixing coefficients may remain fixed with respect to input content and therefore cannot adapt historical access to each query. We introduce \emph{Hybrid Linear Attention} (HLA), a query-dependent chunk-level attention mechanism for Gated DeltaNet (GDN). HLA represents each completed chunk as an exact affine state transition and computes content-dependent routing gates from compact, self-attentively pooled representatives. Each gate interpolates the corresponding historical transition with the identity map, controlling both the chunk's additive memory and its transformation of earlier states. Effective-support regularization further encourages concentrated routing for sparse inference. We evaluate HLA under both pretrained adaptation and from-scratch training. Across Qwen3.5 models from 0.8B to 9B, HLA consistently improves over native GDN and fixed chunk mixing, with gains of up to 5.57 percentage points on LongBench-V2 and 3.97 points on RULER. In a controlled from-scratch 1.3B setting trained for 100B tokens with a 4K context, HLA also improves RULER performance from 4K to 32K, with gains increasing from 0.83 points at 4K to 4.22 points at 32K. These results demonstrate that query-dependent composition of recurrent memory improves long-context modeling and remains effective beyond the training context while using compact per-chunk affine summaries. Project page: this https URL 

---
# Plan Canvas: Fixed Reasoning Regions for Continuous Language Flows 

**Authors**: Miaohe Niu, Pengxiang Li, Jingbo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.05815)  

**Abstract**: Continuous language flows generate text by denoising all positions of a target canvas together. The natural way to add reasoning to such a model is to write a trace ahead of the answer, but the trace length changes from question to question. The answer start is therefore unknown during denoising, and the model has to decide the trace length, the place of every trace token, and the answer at the same time. We propose Plan Canvas to fix the boundary between the trace and the answer. A plan region of fixed capacity holds a compact trace, supervised padding fills its unused positions, and the answer starts at a fixed position. The fixed regions also allow separate denoising clocks for the plan and for the answer. With the trace text, backbone, and canvas length of the free-trace baseline held fixed, Plan Canvas improves accuracy on ProsQA and on Deep ProsQA, a graph benchmark with longer proofs. On Deep ProsQA, accuracy rises from 73.0\% to 87.0\%, the share of questions answered with a valid path rises from 30.8\% to 59.1\%, and the gain is largest on the longest proofs. 

---
# Adaptive Utilization of Low-Rank Adaptation via Conditioned Gating 

**Authors**: Guang Yang, Changhao Guan, Chao Huang, Yufeng Chen, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05800)  

**Abstract**: Low-Rank Adaptation (LoRA) achieves parameter-efficient fine-tuning by constraining model updates to a low-rank subspace and has been widely used in practice. However, LoRA typically employs a shared low-rank update across tokens, which limits its ability to fully exploit the adaptation subspace for tokens from different sequences. To address this issue, we propose an adaptive utilization of Low-Rank Adaptation (U-LoRA), which employs conditioned gating to explicitly learn effective token-level utilization of the limited low-rank adaptation subspace. Specifically, U-LoRA generates utilization coefficients along low-rank directions for each token and jointly coordinates and constrains them using sequence-level contextual information, thereby inducing more consistent adaptive patterns within a sentence. To further enhance training stability, we introduce a bias-corrected exponential moving average (EMA) historical prior that calibrates utilization signals across optimization steps, suppressing noise caused by batch-to-batch fluctuations. The effectiveness of our method arises from a better utilization of the existing low-rank subspace via input-conditioned strategies, rather than from expanding the subspace. Experiments on mathematical reasoning and natural language understanding benchmarks demonstrate that U-LoRA achieves competitive performance under comparable parameter budgets when with strong LoRA baselines and recent variants. 

---
# CLARA: Can AI Assess Developmental Appropriateness in Children's Stories? 

**Authors**: Sijing Yin, Zirui Wang, Qian Liu, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05783)  

**Abstract**: Assessing the developmental suitability of children's narratives is important for educational recommendation and developmental literacy research, yet such assessment typically relies on subjective and difficult-to-scale human judgment. This raises an important question: Can AI systems approximate human developmental judgments of children's stories? To study this problem, we introduce CLARA, a cognitively grounded framework for developmental narrative understanding through structured annotation across cognitive (COG), language (LAN), and social-emotional (SEL) dimensions, together with a bilingual benchmark resource containing 1107 Chinese--English children's stories with normalized silver developmental references and structured developmental annotations. We evaluate CLARA through benchmark comparison, component analysis, translated bilingual consistency analysis, and blinded human evaluation with educators. Experimental results show that structured developmental annotation achieves substantially stronger alignment with developmental references and human judgments than readability-based methods and direct prompting baselines. Overall, our findings suggest that AI systems can approximate certain aspects of human developmental judgment when guided by structured developmental annotation, while also highlighting the importance of interpretability and human oversight in educational NLP. 

---
# MedicalHarness: A Controlled Evaluation of LLMs and Agent Harnesses on Medical Tasks 

**Authors**: Ziqing Wang, Lili Zhao, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2610.05778)  

**Abstract**: LLM agents are increasingly built for medical work and scored on clinical benchmarks. Each such score, however, comes from a model running inside an agent harness, the system that controls the loop between the model and its environment. An agent's score is therefore a property of a model--harness pair. For medical agents, how much outcomes change with the harness has rarely been measured. Measuring this change, and explaining it, raises two challenges. First, a harness comparison must change nothing but the harness and be repeated across models and kinds of task. Second, comparing whole harnesses leaves their mechanisms bundled together, so it cannot show when an individual mechanism helps. To address these challenges, we present MedicalHarness, a controlled study of models and agent harnesses on medical tasks. We first build MedicalHarnessBench to evaluate agents on $107$ tasks across four domains that each test a different harness capability. Using this benchmark, we run five open-weight models under five agent harnesses, changing only the harness within a comparison, and analyze both outcomes and execution traces. To study individual mechanisms, we build MH-Lab, a controlled harness that switches off context management, planning or tool exposure one at a time within a shared execution loop. We find that the harness and its interaction with the model account for about a quarter of the outcome variance, and that no single harness is best across models and tasks. Code and data are available at this https URL. 

---
# Spend Bytes on Breadth: Precision-Count Trade-offs for Decode-Time KV Compression in Long Chain-of-Thought Reasoning 

**Authors**: Runguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.05685)  

**Abstract**: Reasoning models write most of their KV cache while decoding long chains of thought (CoT), so the cache has to be compressed online under a fixed memory budget. Decode-time methods mostly decide which tokens to evict. We ask how a fixed byte budget should be split between the number of cached tokens and their precision. BreadthKV spends the bytes on more tokens at low precision, combining quantization with eviction, and picks the bit-width for each model and budget with a 60-problem end-to-end calibration, since offline attention error does not predict it reliably. On three reasoning models and four math and science benchmarks, it scores above eviction alone in 17 of 18 settings and produces shorter outputs. Much of what eviction loses comes from derailed runs, which keep reasoning until the length cap without reaching an answer. On Qwen3-8B at our tightest budget, eviction sends 91% of AIME samples to the cap and BreadthKV 40%. Under the same protocol, BreadthKV is statistically indistinguishable from a joint rate-distortion allocator (RDKV) that uses 27% more KV memory-time, and it outperforms our re-implementation of ThinKV. 

---
# Knowing the Rules, Applying the Rules: Evaluating Language Models on Traditional Chinese Bazi 

**Authors**: Jiulin Li, Ping Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05682)  

**Abstract**: Knowing domain rules does not guarantee applying them to a case. We study this distinction in traditional Chinese Bazi through 3,000 Chinese multiple-choice questions spanning 14 Theory and 11 Case categories. Six endpoint systems are evaluated, with primary results reported on a 2,492-item model-informed refinement. Theory accuracy exceeds Case accuracy for every system, and gaps of 16.60-29.56 percentage points remain when invalid responses are excluded. The contrast is more specific than a general case-reasoning deficit. Across six systems, Twelve Stages and Nayin reach mean accuracies of 89.10% and 88.62%, while Shensha Basics reaches 75.96%. Within Case, Luck Pillars averages 84.62%, but Career and Family Relations average only 36.98% and 38.19%. Overall rankings also conceal different category strengths. On the original 3,000 items, paired DeepSeek native/disabled comparisons associate native configurations with Theory gains of 6.53 and 12.20 points for Flash and Pro, respectively; Case changes are -3.67 and +1.27 points. These are provider-configuration associations, not isolated causal effects of reasoning. The results motivate task-specific evaluation of cultural-domain applications rather than reliance on aggregate knowledge scores. The benchmark measures agreement with a model-generated, model-verified answer key, not real-world predictive validity. Final-set results are post-selection descriptions, and incomplete provenance and expert validation constrain their interpretation. 

---
# Automatic Speech Recognition for Low-Resource Sinhala: A Critical Review of Methods, Challenges, and Future Directions 

**Authors**: Chanuka Dinuwan, Sanath Jayasena, Buddhika Karunarathne  

**Link**: [PDF](https://arxiv.org/pdf/2610.05681)  

**Abstract**: Automatic speech recognition (ASR) for low-resource languages remains a major challenge. Sinhala, the primary language of Sri Lanka with about 16 million speakers, illustrates the difficulty: agglutinative morphology, a 54-phoneme inventory, subject-object-verb (SOV) syntax and scarce annotated speech data limit both conventional and modern ASR systems. This paper presents the first critical review of Sinhala ASR research, tracing its development from Hidden Markov Models (HMMs) through deep neural networks to self-supervised pre-trained models such as wav2vec 2.0, XLS-R, Whisper and Massively Multilingual Speech (MMS). We compare existing Sinhala systems with related low-resource ASR work on Tamil, Malayalam and Hindi in terms of architecture, training data, word error rate (WER) and robustness to real-world acoustic conditions, and we assess self-supervised and transfer learning as responses to scarce labeled data. We show that most reported WERs are not directly comparable because they differ in corpus, data split and scoring, and that the only controlled comparison in the literature attributes an 18.1% relative WER reduction to corpus correction alone. We also discuss context-aware ASR that draws on phonological, syntactic and semantic knowledge. We identify six research gaps: (1) the lack of large annotated corpora covering multiple dialects and acoustic conditions; (2) weak contextual modeling of Sinhala morphosyntax; (3) high WER in real-world conditions; (4) the absence of standardized benchmarks; (5) the lack of parameter-efficient fine-tuning studies; and (6) the absence of annotated code-switched Sinhala-English speech resources. We outline a research agenda to address these gaps, intended as a roadmap for researchers working on Sinhala and other morphologically rich languages. 

---
# Atomic Visual Entailment: Enhancing Zero-Shot Vision-Language Reasoning through Atomic Fact Decomposition and Learned Selection 

**Authors**: Nallathambi Vethiappan, Derya Soydaner, Gijs Wijnholds  

**Link**: [PDF](https://arxiv.org/pdf/2610.05630)  

**Abstract**: Visual entailment (VE) asks whether an image supports, contradicts, or leaves undecided a textual hypothesis. Strong results come from fine-tuning large vision-language models on labelled data, while zero-shot and hybrid approaches remain far behind. A VE hypothesis often bundles several visual claims, yet existing zero-shot methods reason over it as a single unit. We propose Atomic Visual Entailment (AVE), which decomposes the hypothesis into atomic facts, produces candidate predictions from both the full hypothesis and its facts using frozen vision-language models, and predicts the final label with a lightweight classifier trained only on how those candidates behave. We find that decomposition helps only when the hypothesis context is preserved: judging facts in isolation is worse than not decomposing at all. Full-hypothesis and atomic prediction make complementary errors, and learning which to trust recovers far more of that complementarity than majority voting, reaching 0.803 test accuracy on SNLI-VE without fine-tuning any vision-language model. AVE also localises the visual evidence behind its prediction without region-level supervision. These results suggest that learning which candidate prediction to trust can close much of the gap to fine-tuned systems, offering a practical alternative where fine-tuning a vision-language model directly would need more labelled data or compute than is available. 

---
# More Than Words: Compositional Tokenization for Efficient Language Models 

**Authors**: Yuval Reif, Guy Kaplan, Roy Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2610.05597)  

**Abstract**: Language models process and generate text sequentially in token units, and the tokenizer determines how much text each inference step covers. Under standard tokenization, a short English phrase such as "On the table." is usually produced as four separate predictions for the preposition (On), article (the), noun (table), and punctuation (.), where each consumes a sequence position and adds inference cost. We introduce CoBPE, a compositional tokenization approach that represents such phrases as a lexical base token (table) attached with a small set of reusable surface modifiers, composed in embedding space at input and predicted jointly at output. In controlled pretraining from scratch at 780M and 1.3B scales, CoBPE shortens sequences by 30% and improves average downstream performance by 1.2 points relative to standard BPE under matched training compute. Our results suggest that part of what is now expressed through token sequences can instead be modeled through structured representations, opening a broad design space for more token-efficient and capable language models. 

---
# What Is a Repeated Token Worth? The Scaling Geometry of Multi-Epoch Pretraining 

**Authors**: Yekun Chai, Haoyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2610.05591)  

**Abstract**: As pretraining increasingly repeats data, every run faces three questions: how many epochs to take, how that number should change with model size, and whether anything besides the epoch count matters. We answer them by pricing a repeated token against two references: one epoch on the same data, which gives its value, and fresh data at equal compute, which gives its cost. Against fresh data, the cost of repetition follows a single variable, the number of extra epochs divided by the unique tokens per parameter. Against the same data, a second epoch is worth nearly as much as a fresh one, and repeated tokens fall to half the value of fresh ones after a critical epoch count that grows with the training budget per parameter but hardly with model size. With unique data fixed, the predicted compute-optimal run grows model size and epochs together until loss stops improving, near the critical epoch count. The same variable accounts for the direction of size trends that appear to conflict: larger models tolerate fewer epochs when the corpus is fixed, from about 15 at 127M to 4 at 2B parameters, but not when unique data grow with the model. Counts alone do not determine loss: at identical counts, replaying shards consecutively raises loss by up to 0.46~bits per byte, concentrating repeats on fewer samples also raises it, lower-entropy sources degrade faster with repetition, and re-tokenizing repeats helps only under heavy repetition. These results offer an empirical guide to pretraining when unique data, rather than compute, are the binding constraint. 

---
# Can Prosodic Style Be Inferred from Text Alone? Evidence from Unsupervised Acoustic Clusters 

**Authors**: Abdul Rehman, Jian-Jun Zhang, Xiaosong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05575)  

**Abstract**: Much of expressive text-to-speech research rests on an untested assumption that written text carries enough information to select an appropriate prosodic style for its delivery. Text-predicted style models improve listener preference, and expressive-appropriateness evaluation presupposes that context constrains style, yet neither measures the assumption itself. This paper tests it as a falsifiable hypothesis against style labels derived from acoustics alone. For each of six speakers in a 1,200-hour conversational corpus, utterances are clustered in the spaces of five speech models, including a prosody-only control, and the cluster of held-out utterances is predicted from twelve text embedding models. Three controls are applied: utterance length is erased from the speech embeddings; accuracy is scored against the majority-class floor of unbalanced clusters rather than uniform chance; and a bag-of-words baseline measures word identity alone. Text predicts the cluster above that floor for all six speakers (+0.111 top-3 accuracy), but bag-of-words achieves three quarters of this. Sentence embeddings add only +0.026, largest for encoders not trained for sentence semantics and reversed by tree-based probes for all others. Acoustic clusters are not compact in text embedding space in any of 360 configurations. The prosody-only space weakens the association for five speakers, but not for the speaker showing it most strongly. Text thus informs these delivery clusters mainly through word choice, whether as a cue to prosody or as a marker of topic and recording situation, and reference-free style selection cannot assume more. 

---
# Lend Me Your Eyes: Instruction-Aware Text Embeddings via Attention Relay 

**Authors**: Yiyuan Luo, Vaggos Chatziafratis  

**Link**: [PDF](https://arxiv.org/pdf/2610.05564)  

**Abstract**: Text embedding models trained with contrastive learning learn to follow task instructions from instruction-paired data, while instruction-tuned LLMs already know how to follow them. We show that this instruction-following ability can carry over from an LLM to a Transformer-based embedder without any training. We propose Attention Relay, which passes the attention weights an LLM produces to the embedder's own attention. Across six instruction-tuned LLMs from the Qwen3, Llama 3.1 and OLMo 3 families and ten widely used embedding models that differ in tokenizer, size and pooling type, Attention Relay makes nearly every combination instruction-aware. Experiments that break the method down into its parts show that the LLM's attention weights track the instruction in its later layers and come largely from instruction tuning. They also show that relaying these weights selects which content in the text matters: it makes the aspect of the text that the instruction asks about dominant in the embedding, or restores that aspect where averaging had diluted it. 

---
# Dataset Signatures in Human-LLM Interactions and User Modeling 

**Authors**: Joseph Suh, Serina Chang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05534)  

**Abstract**: Human--LLM interaction datasets shape our understanding of AI use and provide a foundation for downstream research, including training and evaluation of user models. In recent years, a growing number of datasets have sought to capture a representative picture of human--LLM interactions. But how different are the pictures these datasets provide, and what do those differences mean for research built on them? We study these questions across seven conversation datasets, spanning in-the-wild chat logs and human preference data. We begin by revisiting the dataset classification experiment of Torralba & Efros and find that neural network classifiers identify the source of a conversation from user messages alone well above chance, indicating distinctive dataset signatures. This separability persists after matching datasets on the dimensions of human-designed taxonomies, implying subtle differences that these taxonomies do not capture. We then examine the implications for user modeling: how dataset signatures propagate to the outputs of user models trained on these datasets; how dataset choice influences evaluations of user model quality and subsequent evaluations of LLM assistants paired with these user models; and how dataset classifiers can guide data selection for training user models. While each dataset is meant to capture a slice of 'real-world' interactions, our findings reveal the extent to which these slices diverge, and the consequences of those differences for research built on these foundations. 

---
# Writing as a Self-Organized Critical Process 

**Authors**: Nikolay Mikhaylovskiy  

**Link**: [PDF](https://arxiv.org/pdf/2610.05466)  

**Abstract**: We explain autocorrelation decay power laws omnipresent in texts by self-organized criticality. Specifically, we analyze the recently released KLiCKe keystroke dataset and show that not only the final texts' autocorrelations form a manifold that adheres to a power law with a finite-size scaling, but also the text revisions generate revision-size-dependent restoring dynamics toward that manifold. Thus, human writing appears to dynamically regulate semantic correlations in a text toward a critical state. 

---
# The Hidden States Cookbook: A Large-Scale Ablation Study for Noise-Robust Conversational Intent Classification in Industry 

**Authors**: Bogdan Bogachov, Nikita Letov, Yaoyao Fiona Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.05394)  

**Abstract**: Conversational database interfaces face a critical challenge: users naturally embed queries in conversational noise (greetings, politeness, off-topic remarks), which degrades intent classification accuracy and wastes computational resources. Despite advances in orchestration and retrieval strategies, a fundamental question remains unanswered: which pooling strategy maximizes intent classification accuracy under realistic conversational noise in production language models? This work addresses this gap through 360 controlled experiments spanning four pooling configurations (mean, max, last-token, attention, and FFT-augmented variants) using Llama-3.2-1B-Instruct on BANKING77 and CLINC150 datasets under clean/noisy conditions with ten random seeds. Key findings reveal that attention pooling consistently outperforms alternative strategies under noisy conditions (~+2.6-2.8 F1 over the default), while mean pooling degrades performance by up to ~5 F1 points. Frequency-domain filtering does not produce consistent accuracy improvements and functions primarily as a structural variation rather than an accuracy-enhancing component. These results provide concrete, evidence-based guidance for building noise-robust conversational classifiers: attention pooling is recommended for noisy interfaces, mean pooling should be avoided, and last-token pooling is appropriate for clean-query scenarios. 

---
# Harness-Search: Guiding Long-Horizon Search through Multi-Agent Coordination 

**Authors**: Shanyong Wang, Zhenwen Ji, Lei Jin, Yining Zhao, Yicheng Qian, Chengqiang Lu, Yi Wu, Yao Hu, Lizhen Cui, Yanyu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05382)  

**Abstract**: Long-horizon search requires agents to gather evidence across multiple steps and synthesize it into well-supported answers. The recent agent harnesses provide a natural and promising framework to support such long-running search processes. As interaction histories grow, one single agent in harnesses might get stuck and cause the policy to lose track of unresolved questions, overlook useful evidence, or terminate before sufficient support has been collected. One of promising way is to decouple three distinct responsibilities of proposing retrieval actions, updating persistent state, and deciding when to stop rather than concentrating them within a single policy. Targeted at it, we introduce Harness-Search, a multi-agent search harness to reduce the local errors propagating across subsequent exploration, evidence curation, and termination decisions. In particular, Harness-Search assigns these responsibilities to three permission-bounded authorities: a Retrieval Policy that proposes search operations, a Memory Operator that validates and commits persistent-state updates, and a Summary Auditor that accepts or rejects termination based on the sufficiency of the curated evidence. Together, these roles form a Propose-Commit-Audit loop in which actions are proposed, persistent evidence is selectively committed, and stopping decisions are subjected to an explicit sufficiency check. Across seven long-horizon search benchmarks, Harness-Search improves both retrieval and answer generation under the same policy backbone, increasing Recall by 4.60-27.92 points and Final-Answer Recall by 12.34-30.13 points over the strongest harness-based baseline on each evidence-retrieval benchmark. Moreover, trajectory-level analyses show that Harness-Search continues to accumulate useful evidence and expand evidence coverage with less redundant retrieval as the search history grows. 

---
# Towards Unbiased On-Policy Distillation for Block Diffusion Language Models 

**Authors**: Zaiquan Yang, Fei Wei, Yong Wang, Yudong Han, Yiyu Li, Zhuofan Zong, Gerhard Petrus Hancke, Xiangxiang Chu, Rynson WH Lau  

**Link**: [PDF](https://arxiv.org/pdf/2610.05373)  

**Abstract**: On-policy distillation (OPD) has emerged as an effective post-training paradigm for language models, with recent efforts extending it to block diffusion language models (BDLMs). However, existing studies focus almost exclusively on small block sizes, leaving distillation into student models with larger blocks underexplored. In this work, we investigate this regime and reveal two critical optimization biases that induce severe training instability. First, mismatched block boundaries between teacher and student cause \textbf{\textit{context misalignment}}, providing distorted supervisory signals that misguide student decoding. Second, even under aligned contexts, an \textbf{\textit{intrinsic optimization bias}} in OPD, where the student tends to rapidly absorb high-support signals while lagging on low-support updates, drives a premature confidence surge that traps weaker students in catastrophic overconfidence collapse. To resolve these, we propose \mbox{\textbf{Un-OPD}}, an unbiased on-policy distillation framework with two novelties for stabilizing BDLM training. First, Un-OPD introduces a boundary-aware step filtering strategy that eliminates context-misaligned decoding steps. Second, Un-OPD proposes moderating optimization intensity at high-support positions via a support-rebalanced confidence calibration, thereby bypassing overconfidence collapse. Beyond stability, we also introduce a rollout reuse mechanism to reduce rollout generation overhead. Extensive experiments on math reasoning and code generation benchmarks show that Un-OPD consistently stabilizes training and delivers superior performance while reducing wall-clock training time by approximately half. 

---
# The ÌròyìnSpeech Text Corpus: 24,905 Curated Yorùbá Sentences for Speech and Language Technology 

**Authors**: Kola Tubosun, Aanuoluwapo Aremu, Tolulope Ogunremi, Iroro Orife, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2610.05366)  

**Abstract**: ÌròyìnSpeech is a 42-hour, 80-speaker Yorùbá read-speech corpus whose audio has been distributed by ELRA since 2024. This paper describes the release of its text component: 24,905 unique, hand-verified, tone-marked Yorùbá sentences (275,897 tokens; 15,687 types), curated in 2022 as recording prompts. Roughly 11,000 sentences were adapted from openly licensed news material; the remainder were written in-house to broaden coverage beyond the religious translation that dominates existing Yorùbá corpora. Every sentence was checked by hand for tone-mark accuracy, edited for read-aloud clarity and a neutral register, and localised so that non-Yorùbá personal and place names appear in Yorùbá form. Preparing the text for release surfaced systematic Unicode normalisation failures affecting more than 60% of lines (with precomposed and decomposed forms of the same letter co-occurring within single sentences) which we document and correct. The corpus supports diacritic restoration, grapheme-to-phoneme conversion, TTS front-end development and orthographic research, and serves as a validated prompt set for new recording. 

---
# MemStrata: 95% and 90.91% Source-Aware Accuracy on LongMemEval-500 and LoCoMo-1540 with a Local Qwen 3.8 27B Q4_K_M Reader 

**Authors**: Neeraj Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2610.05343)  

**Abstract**: An adequate conversational answer may differ from a short or incomplete benchmark reference. To measure adequacy against the recorded history we prefer source-aware grading, in which the judge checks the reference against the full source before assessing system-blinded answers; original reference-only grading is reported alongside. With a local Qwen 3.8 27B Q4_K_M reader and a 24,000-token evidence ceiling, MemStrata CL1 scores 475/500 (95.0%) on LongMemEval-S and 1,400/1,540 (90.91%) on LoCoMo categories 1-4 under source-aware GPT-5.5 adjudication, against 463/500 (92.6%) and 1,205/1,540 (78.25%) under reference-only grading of the same answers. It preserves a retrieval backbone and adds nonduplicated, dated, speaker-attributed source spans. A same-reader full-history control with about 4.7 times the evidence scores 464/500 reference-only and 470/500 (94.0%) source-aware; neither difference is decisive. Keyword-only selection at the same budget scores 425, and a matched-reader Letta arm 438. On LongMemEval-M, where the packet holds about 1.6% of each history, MemStrata CL1 scores 427/500, with losses concentrated in multi-session and temporal questions. On 300 BEAM-1M questions it outscores dense retrieval, 0.738 to 0.706 (Wilcoxon p = 0.011). A same-seed replay of unchanged requests changed 1.5-2.3% of labels. On identical packets GLM 5.3 flash is non-inferior within 3 points (462 versus 463); Muse Spark 1.3 did not show non-inferiority on 269 questions. None of four pre-registered interventions met all of its registered advancement or feasibility criteria. Signed read-side artifacts support inspection but do not regenerate the private retrieval pipeline. The superiority of source-aware grading to human adjudication is not established, and development exposure, automated-judge dependence and the absence of held-out data preclude an independent-replication or leaderboard claim. 

---
# RubricArmor: Adversarial Evolution Improves LLM-Based Rubric Generation 

**Authors**: Haocheng Yang, Yuchao Zhang, Licheng Pan, Jiajun Fan, Maolin Wang, Kangning Zhang, Shuai Shao, Shijian Wang, Yuan Lu, Chunyuan Zheng, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05308)  

**Abstract**: Rubric-based reinforcement learning (RL) provides interpretable rewards for aligning large language models (LLMs) by evaluating responses against query-specific evaluation criteria. To construct rubrics at scale, a straightforward approach to LLM-based rubric generation is to prompt an LLM to generate a rubric directly from the query. However, rubrics directly generated by LLMs are vulnerable to reward hacking, since omitted or underspecified criteria allow the policy to obtain high rubric rewards with low-quality responses. Existing LLM-based rubric generation methods improve the granularity and coverage of the generated criteria but do not proactively guard against reward hacking. To address this limitation, we propose RubricArmor, an adversarial framework that exposes and mitigates potential reward hacking at the rubric generation stage before it occurs in subsequent RL. Specifically, RubricArmor performs adversarial evolution, in which an attack step and a repair step alternate over multiple rounds. The attack step simulates the reward hacking of the policy by constructing adversarial responses that satisfy the current rubric but fail to properly complete the task. The repair step then revises the rubric to detect the response defects exposed by the attack step while preserving other valid criteria. Extensive experiments demonstrate that RubricArmor outperforms competitive rubric generation baselines and translates into more effective downstream rubric-based RL. 

---
# Red-TTT: Test-Time Training for Automated Jailbreaking Large Language Models 

**Authors**: Tongyan Hu, Hao Li, Xiaogeng Liu, Ruida Wang, Zhengyu Liu, Shuyao Xu, Ning Zhang, Ziyang Li, Yinzhi Cao, Bryan Hooi, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.05282)  

**Abstract**: Large language models remain vulnerable to jailbreaks, and automated red teaming is the standard way to find jailbreaks in large language models at scale. Current methods either draw more samples at test time through search, rewriting, and tree expansion, or train a stronger attacker offline with reinforcement learning. Both share a limitation: once an attack on a specific target behavior begins, the attacker's weights are frozen. Any signal it gathers about the behavior stays in its context window and is discarded afterward. The attacker never adapts its proposal distribution mid-attack, so success depends almost entirely on the sampling budget, and under a budget affordable at scale, many behaviors remain unbroken. We propose Red-TTT, which updates the attacker's parameters during the attack on each behavior. At each round, the attacker samples a group of candidates, scores them against the victim's replies, and takes a policy-gradient step before drawing the next group, so what it discovers about the current victim is consolidated into weights rather than accumulated as context. We also adapt the training objective to red teaming, where success is judged by the single best sample rather than the average. Red-TTT requires only sampling access to the victim and integrates into existing attack pipelines with no other changes. Against the Best-of-N baseline, Red-TTT raises attack success rate from 55.9\% to 72.4\% on average at a budget of 120 samples, improving over the baseline in every configuration and cracking many behaviors previous method cannot. The code is available at this https URL 

---
# When Verifiable Counts Depend on Wording: Auditing Wording Robustness in Instruction Following 

**Authors**: Qishi Zhan, Seoyeon Jang, Zihan Dong, Minxuan Hu, Ziheng Chen, Tonghui Qu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05278)  

**Abstract**: Verifiable instruction-following benchmarks often express each constraint through one fixed template. We test whether scores remain stable when the operational requirement is unchanged but its wording varies. We introduce WISE, a matched evaluation suite and reporting protocol instantiated on exact word count, keyword inclusion exactly once, and an inclusive 8--12 word range. Across 100 matched tasks, up to thirteen models from seven providers, and repeated generations scored over the complete visible output, wording alone produces substantial compliance shifts. In an avoidance-family panel, five avoidance and exclusion forms fall below the positive baseline, while constructional controls also shift compliance substantially: in the nine-model control panel, compliance is 54.9% for the original positive form, 48.2% for a longer positive form, 36.7% when the target appears later, and 33.8% for AVOID1. A strict JSON-structure probe shows wording sensitivity beyond counting, with a different direction of effect. Effect sizes, failure directions, weakest forms, and model rankings vary across realizations. Under the most disruptive exclusion form, the top-ranked model changes and 24.1% of strictly ordered model pairs reverse. Human validation further shows that unanimous agreement on an exact-count interpretation can coexist with substantially different model behavior. WISE supplements conventional scores with mean and worst-form compliance, wording gaps, failure profiles, and ranking stability. 

---
# Inductive Claims Extraction at Scale 

**Authors**: Sandrine Chausson, Björn Ross  

**Link**: [PDF](https://arxiv.org/pdf/2610.05275)  

**Abstract**: A large part of political discourse on social media is built and expressed at a level of claims: i.e. declarative, typically single-clause statements, which convey a particular interpretation of reality and can range from factual to evaluative. Moreover, rather than occurring randomly, claims coalesce, recur in patterns, and come to be associated with different world views. When paired with structural computational tools such as Social Network Analysis, claims can be a powerful unit of analysis to study political phenomena such as echo chambers or polarisation. In this paper, we present a pipeline that uses a large language model (LLM) to inductively extract and catalogue claims from large social media corpora, and apply it to two different Twitter datasets: one relating to the 2020 US presidential election and the other to the 2022 FIFA World Cup. We comprehensively evaluate the approach by measuring the pipeline's recall and precision against manually annotated samples, run ablation studies isolating the contribution of its various components, and perform a qualitative error analysis. We discuss the value of the approach in the context of Computational Social Science research, and illustrate its capabilities by presenting the claims catalogue obtained from each dataset. 

---
# FORGE: Verification-Gated Behavioral Repair for Generative Language Models 

**Authors**: Hsin-Ling Hsu, Min-Yu Chen, Nai-Chia Chen, Yan-Ru Chen, Yi-Ling Chang, Fang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05190)  

**Abstract**: Generative large language models (LLMs) inherit undesirable behaviors from pre-training, including demographic bias and toxic generation, that often emerge only after deployment and affect a small subset of inputs. A repair should eliminate the identified defect, preserve the model's overall functionality and, ideally, provide correctness guarantees. Existing approaches address this only partially: gradient-based fine-tuning lacks per-instance guarantees and becomes unstable with few defect samples; model editing assumes explicit knowledge replacement rather than behavioral correction; and constraint-based repair is largely restricted to discriminative models with unique target outputs. We present FORGE, a framework for targeted behavioral repair of generative language models that separates defect localization, weight editing, and behavioral verification into independent stages. Its core is a repair abstraction that converts localized defective generation into explicit optimization objectives, enabling verification-oriented repair techniques to operate on autoregressive generation. FORGE is editing-mechanism agnostic: we instantiate it with (1) a constraint-based quadratic optimization method that provides per-sample repair certificates and (2) a null-space projection editor that minimizes interference with the original model distribution, both under the same localization and verification protocol. On five open-source LLMs, FORGE consistently achieves larger reductions in bias and toxicity than gradient-based fine-tuning with minor perplexity degradation. The two backends exhibit complementary performance across architectures, which a lightweight causal probe traces to where toxicity-related signals concentrate. FORGE also remains effective with only a handful of defective examples, where conventional fine-tuning often oscillates or fails to converge. 

---
# Verification Trap: Understanding Test-Time Selection Failures under False Premises in Code Generation 

**Authors**: Feng He, Hejia Wang, Linghao Meng, Ming Gao, Qiankun Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.05170)  

**Abstract**: Test-time compute has become a central way to improve code generation: systems sample multiple candidate programs and use verifier-visible evidence to select the final output. This paradigm implicitly assumes that the verifier provides a corrective signal independent from the generator. We challenge this assumption under misleading task premises. When the generator and verifier share a false premise, they become coupled through a mistaken belief: the generator produces premise-consistent shortcuts, while the verifier supplies evidence that fails to expose them. Consequently, the selector may choose a hidden-test-wrong candidate even when a hidden-test-correct program exists in the pool. We call this failure mode Verification Trap. Across three code-generation benchmarks and five code models, false premises consistently degrade first-sample correctness, reduce selector-chosen correctness after 64-sample test-time selection, and amplify recoverable mis-selection. Mechanistically, verifier-written tests inherit the premise-level blind spot, reshaping verifier-visible candidate space away from hidden-test correctness. These traces make Verification Trap predictable before hidden execution: a lightweight gold-free predictor using verifier-visible features reaches 0.846 AUROC. Our results identify decoupled evidence as a key mitigation axis: coupled scaling provides limited recovery, whereas premise-agnostic robustness auditors recover substantial oracle headroom. 

---
# Selecting Repetition Counts Across Model Scales in Data-Constrained Pretraining 

**Authors**: Ziyue WANG, T. Kanamori  

**Link**: [PDF](https://arxiv.org/pdf/2610.05126)  

**Abstract**: The repetition count that works best for a small language model may not remain best at a larger scale. We study this effect in pretraining with a finite target corpus mixed with generic data at a fixed target fraction. On Wikipedia-derived data and Proof-Pile-2, the ranking of measured repetition counts changes with model size, and a 520M Proof-Pile-2 experiment confirms that reducing repetition from sixteen to eight improves loss while using fewer training tokens. We use loss curves from several smaller models to retain a short list of promising repetition counts for evaluation at a larger scale. On PubMed and Caselaw, candidate sets fixed before target-model training retain the lowest-loss measured count on the original evaluation grids at both 200M and 520M. This supports candidate retention as a practical alternative to exact point prediction. We also relate the pruning regression to an empirical scaling model with two opposing repetition-dependent loss terms. A first-order expansion in log model size yields the linear form used by the selection rule, providing a scaling-based interpretation of the candidate-selection procedure. 

---
# Belief-Trajectory Energy: Measuring the Path to a Prediction 

**Authors**: Jiahao Ying, Wei Tang, Boxian Ai, Yaoning Wang, Haotian Chen, Wenhe Sun, Caijun Xu, Haozhan Cai, Changyi Xiao, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.05114)  

**Abstract**: Large language models (LLMs) progressively revise their predictions across Transformer layers, yet we typically observe only the final output, discarding the trajectory through which it is formed. We introduce Belief-Trajectory Energy(BTE), a model-grounded measure that characterizes an input through the layerwise predictive revisions it induces in a model. By mapping intermediate states into a shared predictive space, BTE provides a principled measure of belief change that can be summarized as either a scalar or a structured depth profile. Theoretically, we show that local BTE corresponds to predictive revision under the Fisher-Rao geometry, while the sequence of revisions captures information beyond the initial-to-final belief change. Empirically, scalar BTE provides a model-relative signal of difficulty across diverse reasoning tasks, while richer BTE representations support human-LLM review detection and fine-grained generator attribution, reaching up to $0.998$ macro-AUROC and $95.6\%$ eight-way attribution accuracy. Further analysis shows that BTE develops throughout pretraining and is selectively reshaped by targeted training, demonstrating that the resulting measurement reflects what the scoring model has learned. Together, our results establish belief trajectories as a principled model-grounded signal and suggest a broader perspective in which learned models can themselves serve as instruments for characterizing the data they process. More demonstrations can be found at this https URL. 

---
# InstMoE: Adaptive Multimodal Routing with Specialized Experts 

**Authors**: Guimin Hu, Xiang He, Yingjian Li, Zheng Lian, Boyan Xu, Ruichu Cai  

**Link**: [PDF](https://arxiv.org/pdf/2610.05111)  

**Abstract**: Multimodal inputs are inherently heterogeneous, not only across modalities but also in the information pathways required for effective prediction. To address this limitation, we propose InstMoE, an adaptive expert routing framework for multimodal learning. InstMoE dynamically routes each input to specialized unimodal and cross-modal experts, allowing the model to adapt its information pathways to the characteristics of the input. However, routing can be misled when modality-specific variations obscure task-relevant semantics. Such irrelevant variations may distort routing decisions, causing inputs to be assigned to inappropriate experts. We therefore introduce a Contrastive Semantic Alignment module, which encourages semantically similar inputs to share task-relevant representations while suppressing irrelevant modality-specific variations. Experiments on multimodal sentiment analysis benchmarks demonstrate that InstMoE achieves state-of-the-art performance on CMU-MOSEI and CH-SIMS v2 while using substantially fewer parameters than competitive baselines. Further analysis shows that different inputs exhibit distinct expert preferences, demonstrating that InstMoE moves beyond fixed fusion toward adaptive multimodal computation. 

---
# Small Agents with Semantic Search: Efficient Multilingual Code Localization 

**Authors**: Maxence Lasbordes, Aarush Sinha, Raphael Sourty, Amélie Chatelain, Djamé Seddah  

**Link**: [PDF](https://arxiv.org/pdf/2610.05099)  

**Abstract**: Locating relevant files from natural-language requests is a core subtask for agents operating over code repositories. We investigate whether this task can be delegated to compact, specialized models to enable on-device search while reducing the token usage, latency, and inference cost of larger agents. We show that semantic search improves file localization, with gains in accuracy, cross-language transfer, and inference efficiency. To study this setting, we introduce a training framework for file-localization agents built around ColGREP, a local semantic search tool based on late-interaction retrieval models. Our recipe combines weighted supervised fine-tuning on teacher trajectories, assigning turn-level credit based on retrieval outcomes, with reinforcement learning on localization quality. We train three model families with fewer than two billion parameters to formulate search queries, inspect retrieved content, and identify relevant files. On localization tasks derived from SWE-bench Lite and Multi-SWE-bench Flash, ColGREP-equipped agents substantially improve over their base models and outperform corresponding GREP-based agents. In addition to improving localization accuracy, ColGREP reduces mean end-to-end trajectory latency by 44.1\% on CPU while using 29.1\% fewer tokens, and enables better generalization to programming languages unseen during fine-tuning. These results suggest that compact, tool-specialized localization agents can provide an efficient interface between natural-language requests and large codebases. 

---
# vMF Sentence LDA: A Spherical Topic Model over Sentence Embeddings 

**Authors**: Ryotaro Kobayashi, Yuri Murayama, Kiyoshi Izumi  

**Link**: [PDF](https://arxiv.org/pdf/2610.05095)  

**Abstract**: Latent Dirichlet Allocation (LDA) and models derived from it remain widely used topic models. LDA observes each document as a bag-of-words and models each topic by a categorical distribution over the vocabulary, so that it uses neither the internal structure of the document nor the similarity in meaning between words. Earlier work has responded to this limitation in two ways: many models have introduced embeddings, and some have assigned topics to sentences rather than to words. Their combination, a topic model that observes sentence embeddings, remains little explored. We propose vMF Sentence LDA (vSLDA), which keeps the admixture structure of LDA, observes each sentence as its L2-normalized embedding and models each topic by a von Mises-Fisher (vMF) distribution, which matches the cosine geometry of sentence embeddings. Its per-topic parameter count and per-iteration cost are linear in the embedding dimension, versus quadratic for the full-covariance Gaussian distribution in the existing model over sentence embeddings. We evaluate vSLDA where the limitation is expected to matter most, among topics that share much of their vocabulary: the topics that subdivide the one subject of a collection, and the narrow topics that result when a corpus is divided into a large number of topics. On two corpora, vSLDA attains the best mean rank against eight baselines when the fine categories within each coarse category are classified from the document-topic distributions. On the whole corpus, its advantage appears or widens as the number of topics grows. Weighting the word frequencies of each sentence by its topic posterior yields expected topic-word counts of the same form as those of LDA, so that the standard topic coherence and diversity measures apply to models that assign topics to sentences. On their product, topic quality, vSLDA leads in most within-category conditions of both corpora. 

---
# How Much Do LLM-as-a-Judge Design Choices Matter? A Systematic Comparison of Prompt Designs, Rating Scales, and Models 

**Authors**: Laurène Vaugrante, Thilo Hagendorff  

**Link**: [PDF](https://arxiv.org/pdf/2610.05094)  

**Abstract**: Researchers increasingly use Large Language Models as judges (LLM-as-a-judge) to evaluate model outputs. Yet there are no standards for how to design these judges. Typically, researchers choose the prompt, rating scale, and model intuitively. If these choices change the judge's verdicts, two studies can reach different conclusions about the same facts. To address this risk and to provide an empirical basis for judge designs, we evaluate 10 reasoning models across multiple designs on two tasks: a scalar rating of sentence sentiment and toxicity (over 500 items per category), as well as a binary accuracy classification of question-answer pairs (n=600). For the rating tasks, despite judges showing significant disagreements with the human ground truth, the practical size of differences is small enough to consider most judges reliable (mean absolute deviation of 0.11 points on a 1 - 7 scale); toxicity judges even outperform standard classifiers. Judges are also highly accurate on average (96.5%) for the accuracy classification task. However, design choices can produce shifts: changing the rating scale alone can shift measured bias by up to 0.93 points (rating task), and while accuracy levels are rarely impacted, design choices consistently impact judge leniency (classification task; leniency drop of 28.9 percentage points when using detailed prompts, and up to 56.1 percentage points when switching models). Counterintuitively, lower reasoning effort affects neither accuracy nor leniency. Across both tasks, model identity is the dominant source of variance. These findings suggest that while LLM judges are broadly trustworthy in aggregate, design choices can be meaningful sources of variance. Given the growing reliance on automated evaluation in LLM research, we intend this study as a methodological reference for designing more robust and replicable LLM-as-a-judge pipelines. 

---
# Towards cross-cultural study of folksong lyrics with machine translation 

**Authors**: Anna Dvořáková, Anna Aljanaki, Danbinaerin Han, Peter van Kranenburg, Matěj Kratochvíl, Inna Lisniak, Zdeněk Vejvoda, Jan Hajič jr  

**Link**: [PDF](https://arxiv.org/pdf/2610.05084)  

**Abstract**: Music is universally present in human societies. Ethnomusicologists have long been documenting the diverse expressions of human musicality, and comparative musicology has recently brought several studies of folksong to a more global scale. Such cross-cultural research has not been conducted on lyrics: the language barrier has so far prevented work with multi-lingual data. However, Natural Language Processing (NLP) technologies have reached a stage where this language barrier may no longer be prohibitive. Combining folksong lyrics corpora across five languages, we machine-translate them to a pivot language with a pre-trained neural topic model, and we examine the relationship between content and social function within each language, and across languages for wedding songs. As expected, human evaluation of translation results shows that non-Indo-European languages suffer from overall worse translation quality. Experiments with topic models then indicate that the content of lyrics is at best partially related to the social function of folksongs across all languages. These experiments are just first steps into cross-cultural folk musics lyrics analysis; however, they do indicate that a previously unobserved web of cross-cultural relationships beyond ethnomusicological typologies may be uncovered through the study of what people sing across the world's diverse folk musics. 

---
# Self-Generated Feedback Destabilizes Test-Time Training: A Causal Decomposition of Long-Horizon Adaptation 

**Authors**: Cheng Luo, Bing Li, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2610.05076)  

**Abstract**: Test-time training (TTT) lets a model store information in its weights during inference. When the model learns from its own output, however, each update also changes the model that generates the next training example. Across 128K-token streams, retaining generated-text updates worsens prediction on independent human-written text with three TTT-E2E model configurations (labeled 125M, 760M, and 3B). The same failure occurs when Adam updates Qwen3-4B's existing weights. The same update mechanisms can improve on real text, so writing itself is not the failure. Three matched comparisons trace the causal pathway. Fixed Generation removes over 98% of the damage at 125M and 760M by using a frozen model to generate training chunks. Recorded Replay separates the loss caused by reading degraded text from the additional loss stored by updating on it. A paired one-update comparison then shows the local conflict: an update predicts its source better but new real text worse. This cost grows after Closed Loop adaptation, with a few trajectories accounting for most large failures. Finally, Settlement evaluates the candidate state on independent real text before commitment. It leaves mean endpoint gaps of 0.07 and -0.02 nats at 125M and 760M while retaining real-text adaptation. These results motivate checking prediction on independent evidence before retaining an update. 

---
# Usage-Modulated Sentiment Representations in Large Language Models 

**Authors**: Hongfei Du, Jiacheng Shi, Yanfu Zhang, Gang Zhou, Ye Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.05069)  

**Abstract**: Prior work suggests that sentiment can often be captured by approximately linear directions in LLM activation spaces, but a single direction may not fully capture sentiment representations. In natural communication, sentiment is shaped not only by polarity but also by usage factors, such as tone and audience adaptation. We test whether these factors systematically modulate sentiment representations beyond a shared sentiment direction. We construct a controlled paired dataset that holds event content fixed while varying sentiment polarity and usage factors, and analyze Llama, Mistral, and Gemma. We identify a shared sentiment direction, remove it, and test the residual structure through erasure and generation-time tone steering. Across models, the shared direction is robust (median cosine 0.953-0.975), yet removing it leaves 0.833-0.909 of the original positive-negative representation-difference norm. The residuals contain compact, reproducible usage-conditioned structure. Targeted erasure weakens held-out usage metrics more than random and label-shuffled controls. On Llama, outputs steered along residualized tone components are preferred in 92.8% of blind target-tone comparisons while preserving the requested sentiment polarity in 98.7% of evaluated outputs. 

---
# AraYoungVoices: A Diverse L1/L2 Corpus of Arabic Child and Adolescent Speech 

**Authors**: Shammur Absar Chowdhury, Zien Sheikh Ali, Houssam Eddine-Othman Lachemat, Hamdy Mubarak  

**Link**: [PDF](https://arxiv.org/pdf/2610.05044)  

**Abstract**: State-of-the-art ASR systems primarily target native adult speech, leading to substantial performance gaps for children, adolescents, and L2 speakers. We introduce AraYoungVoices, a 151.72-hour Arabic read-speech corpus from 286 speakers aged 7--18, comprising AraKids (7--12) and AraTeens (13--18). The corpus includes 146 native Arabic (L1) and 140 second-language (L2) speakers, with native speakers spanning Egyptian, Gulf, Levantine, and North African dialectal backgrounds and L2 speakers representing diverse linguistic backgrounds across the Americas, Asia, Africa, and Europe. We benchmark four pretrained ASR models under zero-shot and fine-tuned settings using unseen-speaker-$\&$-unseen-prompt (USUP) and unseen-speaker-$\&$-seen-prompt (USSP) evaluations. Results show that L2 speech remains substantially more challenging than L1 speech, with the largest errors observed mainly for younger L2 speakers. Age-specific fine-tuning improves the matched age group, while joint fine-tuning provides a stronger balance across populations. ASR hypotheses are also consistently closer to the standard reading prompt than to the verbatim transcription, particularly for L2 speech, suggesting partial normalization of reading deviations. 

---
# Causal Improvement Graph for Agentic Harness Optimization 

**Authors**: Junjie Zhang, Shunyu Liu, Haoyu Wang, Ting-En Lin, Yongbin Li, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2610.05039)  

**Abstract**: Agentic Harness is the runtime that constructs task context and controls execution flow, thereby shaping overall agent performance. Given a fixed model and external evaluation, automated Harness optimization seeks to improve this runtime through an iterative proposal--evaluation loop to better solve target tasks. Existing meta-harness methods mainly adopt proposer-centric discovery, in which an LLM-based proposer integrates accumulated experimental findings to determine subsequent Harness revisions. This places the burden of maintaining the evolving improvement state on the proposer as history expands and its underlying experimental logic becomes harder to discern. In this paper, we introduce the Causal Improvement Graph (CIG), a graph-governed meta-harness framework that externalizes the evolving improvement state in a persistent graph, allowing prior findings to directly govern subsequent Harness optimization through local proposer operations. CIG grows and links Evidence, Hypothesis, Intervention, and Outcome nodes to represent what was observed, how it may be explained, how to test that explanation, and what the evaluation reveals. Their structural relations preserve how the improvement state changes across iterations, allowing local proposers to build directly on relations among prior findings rather than recover them from raw history. Across various agent tasks, CIG discovers stronger Harnesses than previous meta-harness baselines and remains robust to the choice of task solver and proposer. Structural ablations further support the design of an explicit improvement state with graph-governed evolution. 

---
# When LLMs Sit Above Diagnostic Tools: Unrealized Complementarity in Industrial Fault Diagnosis 

**Authors**: Donghwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.05031)  

**Abstract**: Large language models are increasingly used as integration layers above specialized tools, but a stronger component does not necessarily produce a stronger combined system. Across five diagnostic datasets (bearing vibration, process monitoring, semiconductor equipment), we study whether an LLM can reliably use external diagnostic information; paired repeat calls separate advice effects from output instability. In all five, conflicting external information overturned initially correct LLM judgments. Among the four datasets with direct integration comparisons, none showed a consistent advantage for implicit LLM integration over the stronger standalone source. On a Tennessee Eastman confirmation set whose protocol was fixed before evaluation, unaided accuracy was 64.67%, implicit LLM-specialist integration 77.43%, and the specialist alone 83.33%. Specialist information improved the LLM by 12.8 points (95% interval 9.7 to 15.9), yet the integrated output stayed 5.9 points below the specialist (95% interval -12.0 to -0.7). A two-source selector oracle reached 92.76%, indicating complementarity that the integrated output did not fully realize. The integrated output missed 140 of 295 specialist corrections (47.5%) but lost 15 of 99 initially correct LLM judgments (15.2%). The deficit remained under prompt and specialist sensitivity analyses. Among CWRU cases solved under both evidence presentations, task-aligned physical evidence yielded lower estimates of susceptibility to incorrect advice in six of seven models (five intervals excluding zero); higher reasoning effort gave no reliable reduction in five models, and a separate four-model TEP analysis gave no clear evidence that it resolves the integration problem. Source quality and integration quality should be evaluated separately: an integration layer should be compared with its stronger standalone component, not only with the unaided LLM. 

---
# IREA: Intermediate Representation-based Embedding Alignment for Normative RAG 

**Authors**: Mirae Han, Sihyeong Yeom, Harksoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.04974)  

**Abstract**: Large language models (LLMs) have shown strong performance across various tasks, but they still struggle with questions involving ethical judgment. Previous studies have attempted to train LLMs on ethical standards, but the diversity and relativity of ethical norms make them difficult to fully internalize in model parameters. As an alternative, we introduce normative RAG, a retrieval-augmented approach that supports ethical judgment using external normative knowledge. Normative retrieval involves a distinct asymmetry between context rich narrative queries and generalized normative statements. Existing factual retrieval methods rely on query-only expansion into a document-like form, making them insufficient for resolving this asymmetry. Therefore, we propose Intermediate Representation-based Embedding Alignment (IREA), a bidirectional alignment method that maps both text types into a shared situation-behavior representation. This representation captures ethically salient contextual and behavioral information in a normalized form, reducing surface-level discrepancies and improving alignment in the embedding space. Experimental results show that IREA improves normative retrieval and downstream ethical judgment across multiple settings, demonstrating the effectiveness of bidirectional alignment for normative RAG. 

---
# TrajLong: Co-Designing Agentic and Long-Context Supervision for Mid-Training 

**Authors**: Miao Peng, Qintong Zhang, Nuo Chen, Yuhan Li, Guochen Yan, Xinran Gu, Hongqiu Wu, Hai Wang, Lydell Huang, Wentao Zhang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.04973)  

**Abstract**: LLM agents for coding, search, and workplace tasks increasingly rely on long-context capabilities to effectively aggregate and reason over extended interaction histories. Recent work has incorporated agent trajectories into mid-training stage, drawing on their naturally long and interaction-rich structure. Yet how to organize the information within these trajectories into effective mid-training supervision remains underexplored. In this work, we investigate the relationship between long-context and agent atomic capabilities and introduce TrajLong, a novel framework that compiles trajectories into long-context training tasks with dense supervision, targeting three representative atomic capabilities: evidence grounding, cross-evidence aggregation, and temporal state maintenance. We mid-train Qwen3-14B-Base and Qwen3-30B-A3B-Base with data compiled by TrajLong, followed by supervised fine-tuning. Experiments on 6 long-context and 12 agent benchmarks demonstrate broad performance gains, with controlled ablations showing improvements over raw and masked trajectory baselines. Capability-level analyses further reveal task-dependent associations between long-context and agent atomic capabilities. These findings suggest that the shared capability demands of long-context reasoning and agent execution provide a principled basis for designing mid-training data to develop downstream agent capabilities. 

---
# Building LLM Agent Systems the Deep Learning Way: From Modular Design to Architecture Search 

**Authors**: Tao Feng, Pengrui Han, Zhongjie Dai, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2610.04961)  

**Abstract**: Large Language Models (LLMs) have revolutionized AI research and enabled exciting agent systems. To build a complex LLM agent system, most existing research relies on insights from other domains or heuristics to manually build the agent system. However, this approach often requires heavy hand-engineering and fails to fully optimize for the downstream task of interest. Inspired by the tremendous success of deep learning, we propose to construct LLM agent systems in a modular manner, similar to building a deep neural network. Our key insight is to make analogies between LLM building blocks, such as retrievals, memories, and prompting strategies, and the successful deep learning modules, such as MLPs, attention, and recurrent modules. We further design forward inference and feedback mechanisms for LLMs, where prompts in LLMs are considered as the weights in deep models, and the prompt optimization from feedback is analogous to the back-propagation algorithm. We additionally leverage a search algorithm to search for the best configuration of LLM agent systems, similar to the neural architecture search (NAS) in deep learning research. Comprehensive experimental results demonstrate that the proposed deep learning recipe for LLM agent systems is highly effective, in particular: (1) Organizing LLM modules into deep-learning-style architectures yields noticeable performance gain; (2) Automatic prompt optimization, equivalent to backpropagation, is efficient in incorporating feedback from the task of interest and achieves at least 5% performance improvement; (3) NAS equivalent algorithm works well for further optimizing the LLM agent system architecture with 11% performance gain compared with randomly designed architectures. Overall, our research demonstrates the exciting opportunity of transferring the success of deep learning to building LLM agent systems. 

---
# From Overloaded to Guaranteed: High-Throughput Multi-SLO Enforcement for LoRA-Assisted On-Premise LLM Deployment 

**Authors**: Zeshen Zhang, Han Zhao, Weihao Cui, Quan Chen, Yu Liu, Yongjun Deng, Jing Yang, Jiuchen Shi, Chen Chen, Youmin Chen, Yu Feng, Minyi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.04956)  

**Abstract**: As Large Language Models (LLMs) become essential in privacy-sensitive sectors like hospitals and government agencies, the on-premise LLM servers offer a cost-effective and secure alternative to public cloud services. However, these resource-constrained servers struggle to guarantee heterogeneous Service Level Objectives (SLOs) when serving multiple LoRA-adapted services simultaneously. Existing serving frameworks suffer from severe SLO violations due to the computational overhead of LoRA layers and the rigid nature of batch scheduling. To address this, we propose HALO, a scheduling method tailored for LoRA-assisted on-premise LLM deployment. HALO introduces two key innovations: a spatial multiplexing strategy that overlaps Base and LoRA computations by partitioning GPU Streaming Multiprocessors (SMs), and an SLO-aware scheduler that decouples request execution based on "request-level slack." By prioritizing urgent tasks and utilizing idle budget for traffic shaping, HALO significantly mitigates resource contention. Our evaluation demonstrates that HALO minimizes SLO violations while improving throughput compared to state-of-the-art baselines. 

---
# Scaling Verifiable Environments for Long-horizon Work Agents 

**Authors**: Jiazheng Zhang, Long Ma, Yunxian Yang, Zhiheng Xi, Zhikai Lei, Yajie Yang, Chenyang Liao, Enyu Zhou, Yang Nan, Yuchen Tian, Senjie Jin, Yibo Wang, Wei He, Boyang Liu, Jixuan Huang, Xin Guo, Zhezheng Hao, Xinbing Liang, Zhihao Zhang, Changzhi Zhou, Wiggin Zhou, Tao Gui, Qi Zhang, Xuanjing Huang, Clarenceai, Aiden Adams  

**Link**: [PDF](https://arxiv.org/pdf/2610.04906)  

**Abstract**: Work agents operate over digital artifacts to execute professional knowledge-intensive work, requiring training environments that support long-horizon interaction and trustworthy verification. However, hand-crafted environments incur prohibitive engineering overhead that prevents environment scaling, whereas synthesis methods sacrifice workspace complexity, realism, or grounded verifiability. To bridge this gap, we introduce WorkForge, a scalable synthesis framework for constructing verifiable work-agent environments from real-world resources. Starting from expert workflows, WorkForge first identifies the resources, decisions, and deliverables required by each workflow. It then retrieves relevant real-world files and organizes them into a workspace. WorkForge inspects the workspace to extract concrete, checkable facts about its content. These factual anchors fix which task types the workspace can support and how their outcomes can be verified. Therefore, WorkForge derives each task's instructions, solution plan, and complementary programmatic and semantic verifiers directly from these factual anchors, keeping verification traceable to observable workspace evidence. Furthermore, we construct 16.7K verifiable environments across 40 professional domains, with workspaces collectively covering 60 file types. Post-training Qwen3.5-35B-A3B-Base improves GDPVal from 45.5 to 73.6 and APEX Score from 5.0 to 21.3, while enabling Qwen3.5-27B to achieve highly competitive performance and outperform strong competitors. Our analyses confirm the efficacy of the proposed method and reveal consistent scaling behaviors across both data volume and interaction horizons. 

---
# Rewrite What Matters: Adaptive Multilingual Query Rewriting for Reasoning via Agentic Reinforcement Learning 

**Authors**: Rui Qi, Yufeng Chen, Yunlong Liang, Chuan Meng, Sijin Lu, Ge Shi, Jinan Xu, Fandong Meng, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04899)  

**Abstract**: In multilingual scenarios, queries with equivalent semantics but in different languages could guide the model into different reasoning trajectories, leading to performance disparities. To mitigate this gap, previous studies typically apply a one-size-fits-all query rewriting strategy, such as translation, which overlooks the fact that different scenarios require diverse types of semantic transformations. In this paper, we propose mRewriter-R1, an agentic multilingual query rewriting framework with reinforcement learning. Unlike single-turn rewriting, mRewriter-R1 formulates multilingual query rewriting as a multi-turn sequential decision-making process, where the model dynamically performs multi-aspect optimization through adaptive operator selection. Experimental results demonstrate that mRewriter-R1 outperforms all strong multilingual rewriting baselines on different large reasoning backbones. Further analyses show that the learned policy can adaptively decide on rewriting operators according to query characteristics, exhibiting strong generalization ability across diverse reasoning tasks, and plug-and-play compatibility with heterogeneous reasoning language models. 

---
# A Systematic Analysis of the Predictive Power of LM Surprisal in Reading Chinese 

**Authors**: Hongao Zhu, Muxiaoqiao Xu, Yikang Liu, Siyuan Song, Yuxia Wang, Byung-Doh Oh, Hai Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04898)  

**Abstract**: This study analyzes the predictive power of LM-derived, token-level surprisal on Mandarin Chinese reading times. We first propose the Shortest Matching Sequence (SMS), an alignment scheme that maps between the word segmentation assumed by eye-tracking corpora and the LMs' subword tokenization, as the two tokenizations often disagree in the context of Mandarin Chinese. Then, using a suite of Chinese-Pythia models (14M-1.4B) trained on scratch with 30B tokens, we examine how well surprisal predicts first fixation duration, gaze duration, and total reading time in three paragraph-level eye-tracking corpora of Mandarin Chinese (GECO-CN, HKP, and MECO). Contrary to previous null findings, our results show that surprisal is predictive of Chinese reading times. However, whether predictive power scales with model size and the amount of training is corpus-specific: bigger models predict better in GECO-CN, whereas inverse scaling emerges in HKP and, at the largest sizes, in MECO. Subsequently, we tested one possible explanation for the inverse scaling in HKP and found that checkpoints whose surprisal remains closer to $n$-gram statistics are better predictors of reading. All in all, the predictive power of surprisal on Chinese reading time measurements is corpus-specific, which cautions against drawing scaling conclusions from a single corpus. 

---
# No Hindsight for LLM Fact-Checkers: Measuring Leakage Channels in Misinformation Detection 

**Authors**: Kuan-Hua Wu Lu, Yohanes Andre Setiawan  

**Link**: [PDF](https://arxiv.org/pdf/2610.04888)  

**Abstract**: As automated fact-checking scales on social media, large language model (LLM) verdict scores can look stronger than warranted. One reason is that evaluations mix in information that was not knowable at claim time. Two channels are easy to conflate: outcomes memorized in pre-training and retrieved evidence published after the claim. Yet standard benchmarks rarely separate the two. In this study we measure both channels on AVeriTeC and QuanTemp++ by reconstructing point-in-time evidence conditions and probing for outcome information encoded in model representations. We find substantial evidence of parametric leakage, that can be hidden by the aggregate accuracy, while a simple representation bottleneck reduces this future leakage more efficiently than a mutual-information-based training penalty. We also find that allowing post-claim evidence inflates zero-shot accuracy by 6.3 points in AVeriTeC while the effect is negligible in QuanTemp++, where retrieval provides little post-claim evidence. These results show that misinformation benchmarks can overstate fact-checking performance when they do not account for what information was actually available at claim time. 

---
# Cluster Validation Indices as Self-Supervised Objectives for Text Representation Learning 

**Authors**: Kishor Kumar Bhaumik, Nicolas Roque dos Santos, Neil Shah, Jia Chen, Evangelos E. Papalexakis  

**Link**: [PDF](https://arxiv.org/pdf/2610.04830)  

**Abstract**: Self-supervised fine-tuning refines the embedding space of a pretrained language encoder without labels. However, the commonly used approaches are computationally expensive. Specifically, contrastive learning-based methods need multiview data and in-batch negative examples, while negative-free approaches require auxiliary graphs/networks. An interesting question arises: can self-supervised fine-tuning be done without relying on either additional negatives or graph data? To answer this question, we introduce SilK (Silhouette-guided K-means), which trains on a Cluster Validation Index, an internal measure of cluster quality without using labels. SilK clusters the corpus and then regresses a simplified silhouette toward a target value. Each document is compared only against the k cluster centroids, never against other documents, so the method needs no augmentation, no negative pairs and one view per document. On BERT-base, SilK trains 1.46x faster per epoch than the fastest baseline we evaluate and uses 45.4% less peak GPU memory than the leanest one. Under frozen-encoder linear probing, SilK stays competitive with the best baselines on three downstream tasks. 

---
# Viva La Vida: Verification and Accumulation Failures in Multi-Agent Proof Search 

**Authors**: Benji Xu, Ken Zheng, Noah Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.04829)  

**Abstract**: When an agentic prover works on an open problem, there is no proof assistant to fall back on: its verifier and lemma library are ultimately language models judging model outputs. We instrumented such a system end to end and analyzed $51{,}754$ traced observations across three full runs ($186$ hours, \$$5{,}694$). We find three connected failure modes. First, the three-model verifier requires unanimity and treats parse or API failure as non-approval; in $10$ of $12$ verification events, one member returned no parseable output or an API error, making acceptance arithmetically impossible without surfacing an error. Second, when the ensemble did function, one verifier approved $3$ attempts that GPT rejected, each claiming to resolve the open problem; a single-verifier design would therefore have announced a solution three times. Third, because nothing could be approved, every review was a refutation, yet the lemma extractor mines reviews as well as proofs: $24$ of $93$ lemmas ($26\%$) were extracted from rejected arguments with their refutational context removed. Taken together, these findings show that without external verification, supervision is itself a critical trust boundary: systems must distinguish abstention from rejection, preserve useful disagreement, and preserve the provenance and polarity of information before it becomes future context. 

---
# How Do People Challenge Racial Stereotypes Online? Counter-Story Detection Across Reddit Communities 

**Authors**: Uma Sushmitha Gunturi, Jimin Mun, Maarten Sap, Maria Antoniak  

**Link**: [PDF](https://arxiv.org/pdf/2610.04803)  

**Abstract**: Counter-storytelling is a powerful mechanism people use to challenge dominant narratives. Unlike other forms of counterspeech that have been widely studied in computational social science, counter-storytelling has largely been overlooked. Counter-stories are difficult to detect automatically; they are relational (defined with respect to expressions of racial stereotypes) and structurally diverse (drawing on stories that describe lived experiences, witnessed events, exemplars, and hypotheticals). We introduce a first framework for detecting and characterizing counter-storytelling against racial stereotypes at scale. This includes (1) a three-dimensional taxonomy grounded in narratology and Critical Race Theory and (2) a multi-stage pipeline that identifies relational pairs of stereotypes and counter-stories in noisy Reddit discourse. Using this pipeline, we annotate 25,549 Reddit posts across 615 communities and identify 1,312 counter-stories. Our analysis shows that speaker identity and post context shape how counter-stories are told. For example, in-group writers favor first-person testimony, often adopting the role of self-reflective insiders. Our work shows how computational methods can scale qualitative approaches to identify and characterize counter-storytelling as a contextual narrative practice, with implications for content moderation, narratology, and racial discourse analysis. 

---
# More Value per Key: Asymmetric Sparse Attention for Faster LLM Decoding 

**Authors**: Noam Elata, Itay Lamprecht, Mikey Shechter, Daniel Ohayon, Itay Hubara, Daniel Soudry  

**Link**: [PDF](https://arxiv.org/pdf/2610.04753)  

**Abstract**: utoregressive generation in Large Language Models (LLMs) is constrained by the memory and computational demands of attention mechanisms. Sparse attention methods mitigate this cost by selecting only high-probability entries of the attention matrix. We observe that in many such methods, this renders the probability-value multiplication negligible, shifting the bottleneck to the query-key step. Key heads can therefore be reduced to accelerate inference, while retaining more value heads preserves capacity with limited additional decoding cost. We introduce Sparse Asymmetric Group-Query Attention (SAGA), which decouples key and value head counts to exploit this principle, and pair it with approximate top-N (Atop-N) attention, a simple sparse attention method designed to study the interaction between sparsity and head-count asymmetry. We formalize the benefits of this asymmetry theoretically and validate them empirically through latency measurements and quality evaluations on models up to 1.5B parameters. Together, SAGA and Atop-N achieve end-to-end decoding speedups exceeding $2\times$ over our full-attention GQA baseline at long contexts. Models trained from scratch with SAGA nearly match the quality of comparable GQA variants on the evaluated benchmarks. To facilitate adoption, we introduce an efficient fine-tuning method that converts pretrained models to the SAGA architecture, enabling practitioners to benefit from our approach without costly retraining. 

---
# Verb-ICL: Rethinking In-Context Learning for Structured Prediction 

**Authors**: Fan Bai, Hengshuo Miao, Sanjit S Batra, Hamid Reza Hassanzadeh, Ardavan Saeedi, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2610.04725)  

**Abstract**: Structured prediction tasks pose unique challenges for in-context learning (ICL): their compositional outputs require modeling fine-grained, token-level patterns that sentence-level approaches fail to capture, and their task-specific annotation conventions are human-defined artifacts that cannot be acquired through pretraining alone. We propose Verb-ICL, a selective annotation framework for ICL-based structured prediction that addresses both challenges. Verb-ICL first selects representative examples using a token-level coverage strategy that captures local semantic patterns critical for structured prediction, then generates actionable error feedback that codifies task-specific annotation guidelines and incorporates this feedback into ICL demonstrations. We evaluate Verb-ICL on six structured prediction datasets spanning information extraction and semantic parsing. Experiments with recent LLMs show that Verb-ICL consistently outperforms strong selective annotation baselines under low-resource settings and continues to provide gains as the annotation budget increases. Extended analyses demonstrate that the generated feedback is predominantly useful across a four-category quality taxonomy, generalizes as task-level guidance beyond instance-specific corrections, and improves performance regardless of the underlying selection strategy. 

---
# WNet: Discrete Wavelets Transform for Efficient Token Mixing 

**Authors**: Rana Aref Salama, Abdou Youssef, Mona Diab  

**Link**: [PDF](https://arxiv.org/pdf/2610.04720)  

**Abstract**: In a Transformer, token mixing is the step that lets each token draw information from other tokens, and it dominates the cost of encoding long sequences. Self-attention does this mixing very well: every token weighs every other token by content, which gives strong contextual modeling. That all-pairs comparison is also why its cost grows quadratically with sequence length. We introduce WNet, a Transformer encoder that replaces self-attention with token mixing based on the discrete wavelet transform (DWT). Three attention-free mixers recombine the scales: by linear fusion, by learned gating, or by letting each token choose its scales. A hybrid adds self-attention in the last layer only. A receptive-field analysis shows that wavelet mixers built from two-tap filters, such as Haar, never relate tokens outside fixed blocks, however deep the network, even when the filters are learned. Longer filters reach the whole sequence within two layers. We pre-train every model with masked language modeling on a fixed-token subset of C4 and fine-tune on GLUE, using one controlled setup with size-matched BERT and FNet baselines and a control that cannot mix tokens. The token-gated mixer trains as fast as attention at 256 tokens and 2.7 times faster at 4,096. 

---
# Understanding Errors in LLM-Based Question Answering over Imperfect Tables 

**Authors**: Baowen Zhang, Wei Fan, Ruman Wang, Hangting Ye  

**Link**: [PDF](https://arxiv.org/pdf/2610.04687)  

**Abstract**: We investigate error discovery and handling in question answering over imperfect tables through controlled studies across three large language models (LLMs) on human-reviewed RADAR-T examples. Answering questions over these tables requires handling errors that can affect the answer. We vary row order and compare original, error-marked, and repaired tables to test whether discovery depends on where errors appear and whether providing their locations is sufficient for accurate question answering. First, reordering rows changes error discovery even when the table contents and gold answer remain unchanged. Complete discovery is higher for back than front placements and, averaged over the tested mean positions, for compact than widely spaced layouts. Second, providing verified error locations alone is insufficient for accurate QA, leaving a substantial accuracy gap between error-marked and repaired tables. Providing tables with human-reviewed repairs already applied raises code-assisted QA accuracy by 39.0-59.1 percentage points over the error-marked tables across the three systems. GBDI, a simple workflow, puts these findings into practice by combining error discovery across shuffled table views with explicit guidance for verifying and handling the reported errors. On RADAR-T, GBDI raises observed QA accuracy by 3.8-18.5 percentage points over a code-agent baseline across five systems. These results highlight the importance of both reliable error discovery and effective error handling in question answering over imperfect tables. Our anonymous repository is available at this https URL 

---
# Steering Speech-Language Models: Training-Free Task Specialization via Contrastive Activation Addition 

**Authors**: Séverin Baroudi, Yanis Labrak, Pierfrancesco Melucci, Sergio Burdisso, Petr Motlicek, Hervé Bredin, Mirco Ravanelli, Ricard Marxer  

**Link**: [PDF](https://arxiv.org/pdf/2610.04683)  

**Abstract**: Activation steering has proven effective for controlling the behavior of Large Language Models (LLMs) at inference time, but its application to SpeechLLMs remains new, and training-free steering approaches for such models are still largely unexplored. We propose a training-free Contrastive Activation Addition (CAA) protocol that derives steering vectors for common speech tasks (e.g. transcription) in SpeechLLMs from a small number of labeled utterances. We showcase that adding these vectors in the representation space, at inference time, enforces better the targeted speech task. We further show that, when combined with prompting, these vectors yield to consistent improvement over prompting alone on most evaluated tasks such as Automatic Speech Recognition (ASR) or Emotion Recognition (ER), and transfer to out-of-domain data. We additionally demonstrate the usefulness of script-normalization directions to enforce the target script of a specific language. 

---
# Extracting Persona Subspaces Through Iterative Nullspace Projection For Modulation 

**Authors**: Ananya Malik, Mai ElSherief  

**Link**: [PDF](https://arxiv.org/pdf/2610.04676)  

**Abstract**: Large Language Models (LLMs) can adopt distinct personas to tune their semantics, expertise, and perspective to different users and tasks. Precise control over these traits is critical to ensure safety and reliability in model behavior. Existing methods like activation steering and prompt-based persona induction reduce a persona to a single dominant direction, missing the finer, nested traits that emerge only once that dominant signal is factored out. We introduce modulation as a setting where the persona context is already embedded in the content being manipulated, requiring control methods to amplify or suppress a trait already present rather than inject it from scratch. PaSS is an inference-time control paradigm that models personas as multi-dimensional subspaces in a model's latent space without supervised contrastive examples. The persona subspaces are extracted via iterative concept erasure and applied to modulate persona-guided generation without retraining. To extract this subspace, we use Iterative Nullspace Projections (INLP) to linearly and iteratively isolate persona-specific directions. We causally evaluate six personas against diverse tasks like MATH-500, TinyAlpaca, GSM8K, and IFEval, showing that discriminative, iterative subspace extraction captures diverse traits underlying a given persona, enabling stronger and larger modulation than single-direction additive methods, while maintaining content fidelity. We further study individual peeled directions within each subspace to uncover the distinct aspects of persona behavior they encode. Overall, we show that persona subspaces offer a controllable, interpretable, and generalizable framework for modulating LLM behavior without sacrificing task performance. 

---
# Grounding Probes: Generator-Independent Hallucination Detection from Observer Model Hidden States 

**Authors**: Michael Rathmayr, Ádám Kovács, Gábor Recski  

**Link**: [PDF](https://arxiv.org/pdf/2610.04642)  

**Abstract**: Detecting responses that retrieval-augmented generation does not ground in its context trades speed against accuracy: surface checks miss paraphrased fabrication, sampling-based methods cost extra generations. Hidden-state probes sit between the two, but every existing one reads the generating model's own activations, so a change of generator invalidates the detector and a closed-weight generator is out of reach. This paper removes that coupling. The Grounding Probe is logistic regression over the mean-pooled middle-layer hidden states of an observer language model that reads the context, question, and response in one forward pass and generates nothing, with the recipe it needs: pool over response tokens, read a middle layer, and control capacity, which closes the train-test AUROC gap from 0.087-0.202 to 0.009-0.013. Asking the observer outright, rather than reading its hidden state, costs at least +0.166 AUROC in every one of four models. Fitted on 15,090 annotated responses it reaches 0.879-0.894 AUROC on RAGTruth test across four observers, and 0.924 AUROC with 0.820 F1@0.5 averaged with a supervised span detector, 0.060 above that detector alone. One probe holds across six generators, and hold-out controls, including one in which no evaluation prompt appears in training, bound the cost of removing a generator at about 0.02 AUROC. Code, probes, and predictions are released. 

---
# Does Neural Complexity Improve Health Misinformation Detection? A Leakage-Controlled Cross-Corpus Benchmark 

**Authors**: Mkululi SIKOSANA  

**Link**: [PDF](https://arxiv.org/pdf/2610.04636)  

**Abstract**: Increasing architectural complexity is often assumed to improve health misinformation detection, yet reported gains are difficult to interpret when studies use different corpora, preprocessing pipelines, data splits, and leakage controls. This study provides a controlled cross-corpus benchmark of five compact neural architectures (1D-CNN, LSTM, BiLSTM, CNN-LSTM, and CNN-BiLSTM), a soft-voting neural ensemble, and three classical machine-learning baselines using COVID19-FNIR and CONSTRAINT. Exact-text duplicate controls were applied before modelling; all neural systems used a common preprocessing and optimisation protocol, and neural results were repeated across three random seeds. On COVID19-FNIR, the deep ensemble achieved a mean macro-F1 of 0.9963 and ROC-AUC of 0.9994, while individual neural models ranged from 0.9945 to 0.9957 macro-F1. On CONSTRAINT, the ensemble achieved macro-F1 of 0.9272 and ROC-AUC of 0.9811, whereas a linear SVM achieved macro-F1 of 0.9574 and ROC-AUC of 0.9931. Architecture rankings changed across corpora, and simple sparse linear models remained highly competitive. The findings show that model complexity does not provide a stable performance advantage and that benchmark construction can dominate architecture choice. The study contributes a reproducible, leakage-controlled basis for evidence-driven model selection in health misinformation classification 

---
# Stance Drift: How AI-mediated Communication Distorts Our Message 

**Authors**: Lingchong Liu, Yanfei Zhou, Jacob Bien, Y.X. Rachel Wang, Lucy Xia, Xin Tong  

**Link**: [PDF](https://arxiv.org/pdf/2610.04620)  

**Abstract**: Large language models (LLMs) increasingly mediate human communication, from drafting emails to summarizing scientific reports, yet whether they faithfully preserve a speaker's position remains largely untested. We model AI-mediated communication as a two-step generation-extraction pipeline: one LLM produces an argument from a specified stance, and a second LLM extracts the stance from that argument. We represent the pipeline as a probabilistic state transition over five Likert-type stance categories and define the stance preservation rate (SPR) as the average probability that the extracted stance matches the initial stance. Across 112 debate propositions, none of the nine LLMs tested exceeded an SPR of 0.7 under the default configuration. Three drift patterns accounted for most of the drift: polarization, deviation from neutrality, and flipping. Among the mitigation strategies tested, including in-context learning, multiple extraction with shuffled options, assertion, and reflection, only adding medium reasoning effort to a reflection prompt for GPT-5.4 substantially improved the SPR, to 0.775, yet polarization remained the largest pattern, with 0.119 of the transition mass. An exploratory comparison with human extraction on a single proposition suggests that drift arises at both the generation and the extraction stage. These results point to a fidelity gap in AI-mediated communication, with implications for journalism, policy deliberation, scientific communication, and other domains where opinion-laden messages pass through language models. 

---
# SIFT: Robust Meta-Faithfulness Verification of Chain-of-Thought Reasoning Under Distribution Shift 

**Authors**: Noor Islam S. Mohammad, Md. Basim Al Zabir Shammo, Hasan Siddiki, Mahmudul Hasan, Md. Faisal Sheikh, Jakaria Habib  

**Link**: [PDF](https://arxiv.org/pdf/2610.04594)  

**Abstract**: Chain-of-Thought (CoT) faithfulness detectors are widely used to audit reasoning models, yet a detector is itself a predictor whose verdicts are treated as stable properties. We ask whether a detector is faithful to itself under distribution shift. We formalize meta-faithfulness as an invariance principle: a valid detector must return identical verdicts on traces that differ only by transformations preserving ground-truth faithfulness. We prove three results: (i) no detector using only intervention-response profiles can separate faithful from epiphenomenal mechanisms with identical signatures; (ii) any detector relying on shift-sensitive features violates invariance at a rate independent of its in-distribution accuracy; (iii) an asymptotic certified selective-risk guarantee enables confident abstention. We operationalize the principle in FaithShift, a stress-test protocol spanning ten shift axes, and propose SIFT, a hidden-state trajectory detector trained with cross-environment invariance objectives and certified abstention. Across 14,996 traces, four domains, and eight models, three findings emerge. First, transfer collapse is real: all existing detectors show gaps $\geq 0.15$ AUROC. Second, the dominant bottleneck is sampling stochasticity, not shift: over 80% of detector instability stems from random seed variation, falsifying our preregistered prediction that shift-attributable violations exceed 0.25. Third, SIFT cuts invariance violations by 64% over the best single-seed baseline, but a four-seed ensemble of any detector narrows the margin to 0.01 (indistinguishable at matched coverage, $p=0.21$), and SIFT needs a 51% abstention rate. Cross-model transfer degrades from within-family to cross-family to open-weight-to-API, partly closed by multi-model training. We offer a framework for auditing auditors: the real barrier is detector variance, not distribution shift. 

---
# From Probe Scores to Alarm Policies: Operational Validity of Activation Monitors for Language-Model Agents 

**Authors**: Xueping Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.04575)  

**Abstract**: Activation probes can predict safety-relevant properties of language models with high area under the receiver-operating-characteristic curve (AUROC), but deployed agent monitors make thresholded alarm decisions under tight false-alarm budgets. These are different estimands. We introduce an Operational Validity Contract that fixes a monitor's target, observability, identity, timing, intervention unit, comparator, calibration, and cost. We formalize risk at the semantic request or trajectory level: when one task contains repeated alarm opportunities, row-level AUROC and false positive rate do not identify semantic-unit any-alarm risk. A confidence-certified threshold also requires enough independent negative units, a tie-safe rule, and transport to deployment. Across Models Under Pressure, LASR refusal prediction, immutable AgentDojo, and a prospectively protocol-frozen ST-WebAgentBench replication, joint activation-observable monitors reach AUROC 0.957 and 0.935 on the first two benchmarks, yet their locked 5%/10% detection rates are only .642/.742 and .719/.782, respectively. On AgentDojo, the secondary mean-activation rollout monitor reaches AUROC 0.922 but detects none of 38 positive semantic cases at the locked 5% operating point; thresholds intended for 10% false alarms realize 18.7-20.0% on test. Because the test misses its prospectively frozen 40-positive support gate, we label it support-insufficient. On ST-WebAgentBench, activation reaches AUROC .874, but 23 independent calibration negatives cannot identify even a 10% controller; the locked policy abstains rather than reporting its mechanical zero FPR as a success. An exploratory counterexample also lowers full AUROC while improving realized 10% utility. The fail-closed compiler caps MUP and LASR at restricted predictive value and AgentDojo and ST-Web at representation accessibility; no setting reaches alarm-policy validity. 

---
# Correctness Is a Direction: Geometric Answer Selection in Language Models 

**Authors**: Marcus Armstrong, Navid Ayoobi, Pradham Mummaleti, Alexander Chulzhanov, Arjun Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2610.04512)  

**Abstract**: Answer correctness is encoded as a recoverable geometric direction in the hidden states of language models. We show that the mean displacement from incorrect to correct answer representations, computed at approximately 70\% of model depth from fifty labeled examples with no parameter updates, yields a scoring direction that outperforms zero-shot log-probability scoring by up to +32.0 percentage points on factual benchmarks (ARC-Challenge and MMLU) and by +38.1 to +51.8 percentage points on TruthfulQA, across five models spanning 1B to 8B parameters in three architecture families (Llama, Qwen, Gemma). The method requires one forward pass and one dot product per candidate; no generation is performed at inference. Applied as a hallucination detector on individual (question, answer) pairs, the recovered direction achieves 0.693~AUROC versus 0.578 for log-probability scoring. We additionally find that correctness directions for factual reasoning, domain knowledge, and calibrated truthfulness are near-orthogonal in representation space, revealing that language models allocate geometrically independent subspaces to qualitatively distinct notions of correct answer, with architecture-dependent variation in the degree of separation. This structure explains the observed transfer pattern---the direction calibrated on factual questions transfers within task type but not across it---and suggests that LLM calibration failures may reflect a routing problem: the model's internal representation contains more correctness signal than its output behaviour exploits. 

---
# Emoji-Emotion Ranking System Using Twitter Data 

**Authors**: Danila Khlebokazov, Nurkhan Tashimov, Pakizar Shamoi  

**Link**: [PDF](https://arxiv.org/pdf/2610.04495)  

**Abstract**: Nowadays, emojis are often replacing words. Yet computational systems still oversimplify them. Most existing approaches treat emojis as static sentiment indicators and overlook their emotional distributions. In this study, we propose an emoji-aware emotion analysis framework based on a Twitter (X) dataset of 100,000 emoji-containing replies collected between 2020-2025. After preprocessing and text cleaning, we applied text-to-emotion classification to detect five primary emotions (Happy, Angry, Sad, Fear, and Surprise) for each message. By aggregating emotion scores across contexts in which each emoji appears, we estimate emoji-emotion association distributions and construct an emoji-emotion ranking system reflecting relative emotional dominance. Furthermore, we project emojis into the Russell valence-arousal space to enable continuous affective interpretation. Our results demonstrate that emojis exhibit probabilistic, context-sensitive emotional profiles rather than fixed sentiment polarities. 

---
# DV-Lens: Revealing the Functional Organization of Language Model Parameters 

**Authors**: Chenhang Cui, Jian Yu, Shuyi Miao, Xiaohao Liu, Rui Huang, Fei Shen, An Zhang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2610.04489)  

**Abstract**: Understanding parameter functions helps elucidate the internal mechanisms of large language models (LLMs). However, how to connect parameters from different modules to verifiable output effects and further characterize the relationship between their functional organization and model capability remains to be explored. To this end, we introduce the downstream vocabulary lens (DV-Lens), a parameter-level interpretability framework that links native parameter directions to their downstream vocabulary responses. Specifically, we first estimate module-specific downstream Jacobians over a reference prompt set for attention query, key, value, and output (Q/K/V/O) projections and feed-forward networks (FFNs). Second, we use these mappings to project native parameter columns into the final vocabulary space, obtaining signed readouts that characterize their average local output responses. Third, we group parameter columns by their vocabulary readouts and introduce downstream vocabulary complexity (DV-Complexity), which quantifies within-group structural variation using normalized reconstruction residuals of the original weights. At the parameter level, randomized controls and finite-difference tests show that DV-Lens readouts capture non-random vocabulary structure and predict local logit changes with 98.0% coordinate-orientation agreement across 720 cases from nine models. These readouts further guide parameter ablation, steering, and swapping across 21 models, shifting target-token probabilities in the predicted directions under controlled conditions. At the model level, the joint-parameter score of DV-Complexity achieves a Spearman correlation of 0.904 with benchmark-based capability rankings across 48 language models. Together, these results provide intervention-based evidence for DV-Lens interpretations and reveal an association between DV-Complexity and model capability. 

---
# Understanding and Mitigating Hallucination Escape in Tool-Using LLM Agents 

**Authors**: Peigui Qi, Kunsheng Tang, Yide Song, Weiming Zhang, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04409)  

**Abstract**: Large language models (LLMs) increasingly serve as autonomous agents that invoke external tools. However, this capability introduces tool hallucination, selecting incorrect tools or generating invalid calls. Existing mitigation methods report substantial improvements, yet we identify a previously overlooked failure mode that we term Hallucination Escape. These methods reduce hallucination on the tool configuration they are tuned on but increase it on other configurations, canceling out the gain. We further investigate this phenomenon and find that hallucination rises sharply when a model's intrinsic tool-use tendencies conflict with the current tool configuration, and that existing methods reinforce rather than suppress these tendencies, which in turn contributes to hallucination escape. Building on these findings, we propose EscapeGuard, a training-free inference-time method that combines conflict-aware gating with configuration-derived attention enhancement to mitigate tool hallucination while preventing hallucination escape. Across six benchmarks on various models, EscapeGuard reduces tool-selection hallucination by 9.0 pp and suppresses hallucination escape, lowering the cross-configuration mean by 23.7 pp and achieving an 89.1% net improvement in paired-query evaluation. We hope this work can encourage evaluation beyond a single tool configuration and pave the way for more reliable tool-using LLM agents. 

---
# Saying, Not Knowing: Aggressively GGUF-Quantized Small Language Models Still Write Rare Words They Can No Longer Define 

**Authors**: Saurabh Kumar Singh, Yogeshwar Singh Dadwhal, Malhar Vedak  

**Link**: [PDF](https://arxiv.org/pdf/2610.04403)  

**Abstract**: Post-training quantization to the GGUF format's mixed-precision K-quants is commonly how open-weight language models reach consumer hardware, yet its effect on fine-grained lexical competence is uncharacterized. We audit 27 quantized artifacts across 13 families and four architecture backbones, 0.35B-14B parameters, evaluated down their published ladder to Q2_K (about 2.6 bits per weight), on 429 frequency-validated rare English words under two probes: surface inclusion of a prompt-supplied word and its one-sentence definition, scored by a tiered multi-synonym matcher, its error measured by a blind LLM-judge census of every definition, with human verification. Three regimes emerge at Q2: total collapse into unusable builds, severe semantic dissociation in sub-2B models, and mostly robust preservation above about 3B. In every sub-2B artifact, definitions fall 20-67% below the artifact's baseline, typically several times the inclusion loss. Two controls separate rarity from task difficulty: within the rare set, loss rises with rarity in six of seven sub-2B artifacts, and on a 100-word common-word set rare words lose more than common words in all eight, significantly in six. Tokenizer vocabulary size does not predict the damage (Spearman rho=0.12); parameter count dominates (rho=0.72), confirmed within five of six same-tokenizer families. Q4_K_M remains lexically clean at >=1B. The damage is frequency-graded, provider-dependent, and not calibrated by WikiText-2 perplexity: across nine artifact-matched ladders, near-identical Q2 penalties (44.7%/47.6%) separate an artifact keeping its definitions (3.6%) from one losing them (43.6%). Aggressively quantized small models can keep generating fluent text while no longer knowing what it means, risking hardware-constrained deployments in domains where semantics carries consequences. Validation must be per artifact. 

---
# Ideological Stance Detection in a Low-Resource Language: Polarization in Bangladeshi Public vs Private University Discourse on Social Media 

**Authors**: Safaruzzaman Shovo, Monowar Islam, Asif Hossain, Sameya Akhter, Md. Shamsul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2610.04401)  

**Abstract**: Public vs. private universities is a debatable issue, and it creates polarization on social media in Bangladesh. Debate on quality, jobs, and prestige is passionate among the students, parents, and graduates, the majority of whom speak Bengali, a low-resource language. To measure this polarization, this paper introduces a manually annotated dataset of 4,060 Bengali comments labeled as Pro-Public, Pro-Private, or Neutral. We evaluated the quality of our annotations by Fleiss's Kappa agreement that was 0.89, corresponding to a high agreement among annotators. The classical ML (SVM, Random Forest, XGBoost), BiLSTM network, hybrid BanglaBERT+XGBoost models and the state-of-the-art zero-shot LLMs (Claude Sonnet 4, DeepSeek-V3.1, Llama 4 Maverick, Kimi K2 Thinking, Qwen3-235B Thinking) models are evaluated. The accuracy of BanglaBERT+XGBoost is 91.81% and macro F1 score is 91.70%, which is higher than all the supervised baselines. The zero-shot Llama 4 Maverick Thinking achieves a macro F1 of 0.931 (overall accuracy of 93.31%) without any fine-tuning. All machine learning (ML), deep machine learning (DL) and transformer models were outperformed by the zero-shot Llama 4 Maverick model. Polarization also is evident, in some ways more clearly in the Pro-Private comments, which emphasize modern facilities and timely graduation, versus the Pro-Public comments, which emphasize affordability and government jobs. Our findings open new directions for analyzing social media polarization in low-resource languages. 

---
# XTurnix: Large-Scale Self-Supervised Turn Control through Two-State Binary Decisions 

**Authors**: Zhanxun Liu, Yifan Duan, Hengtao Wu, Chen Yang, Qinyuan Cheng, Kun Wang, Xingyu Zeng, Xipeng Qiu, Chaochao Lu, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.04400)  

**Abstract**: General turn-taking behavior in real-time dialogue systems requires deciding whether to keep listening or start responding while listening, and whether to continue or stop while speaking. Existing turn detectors use heterogeneous, task-specific label spaces and are often trained on limited annotations or evaluated on isolated utterances, making them difficult to use as a unified causal controller with comprehensive context. We propose XTurnix, a compact text-based model that formulates turn control as two binary decisions conditioned on the AI's current listening or speaking state and predicts a single control token from the complete dialogue history. XTurnix is pretrained on 5.5 million causal action examples automatically derived from timestamped two-speaker transcripts, then fine-tuned on synthetic multi-turn examples with a flatter distribution across the four state-action labels. We evaluate XTurnix on four public benchmarks and a balanced self-curated benchmark. Across the public benchmarks, XTurnix achieves the best results on all SemanticVAD and LiveKit splits, ties the native Smart-Turn model on Smart-Turn Bench, and achieves the highest incomplete-turn accuracy on Easy-Turn. On the self-curated benchmark, it reaches 89.06% accuracy, more than 20 percentage points above the strongest third-party baseline at 68.75%, while maintaining F1 scores between 84.21% and 90.63% across all four categories. These results demonstrate unified listening- and speaking-state turn control in a single compact model. Code is available at this https URL, with an interactive demo at this https URL. 

---
# GlitchPatch: Repairing Glitch Tokens in Frozen Language Models via Local Retokenization 

**Authors**: Kunsheng Tang, Peigui Qi, Yide Song, Peijun Huang, Weiming Zhang, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04399)  

**Abstract**: Glitch tokens are anomalous vocabulary entries that can cause large language models (LLMs) to produce outputs inconsistent with their inputs. Existing repair methods require access to model internals, making them impractical for frozen checkpoints. We investigate whether glitch tokens can be repaired outside the model by optimizing the input tokenization. An empirical study on BPE merge-rule deletion reveals that (1)deleting a glitch token's merge rule can fix a substantial fraction of failures, yet disrupting normal tokens sharing intermediate merge nodes causes the overall glitch rate to rise, and (2)different decomposition granularities yield non-monotonic fix rates while collateral damage on normal tokens grows monotonically. Motivated by these findings, we propose GlitchPatch, a repair framework for frozen language models based on local retokenization, consisting of two stages: the offline stage uses Behavioral Path Optimization (BPO) to find the behaviorally optimal replacement token sequence for each glitch token and compiles validated replacements into a rule table; the online stage substitutes only the IDs of matched glitch tokens in the canonical token sequence, with no modification to model parameters or internal states. Experiments on ten models spanning six tokenizer families show that GlitchPatch achieves an 85.10% mean fix rate, outperforming the strongest baseline by 14.37 percentage points, and reduces the average glitch rate from 14.88% to 2.27%. GlitchPatch achieves a 0.00% RR in full-vocabulary evaluation and leaves rule-unmatched inputs unchanged by design. We further evaluate the practical impact of repair from the perspectives of time cost, language understanding, and capability, supporting its deployment feasibility. We hope this work provides a practical option for improving tokenizer reliability. 

---
# Boundaries Agree, Labels Do Not: Intra-Annotator Dynamics as a Kind of Training Data 

**Authors**: Marharyta Shvets  

**Link**: [PDF](https://arxiv.org/pdf/2610.04370)  

**Abstract**: Data quality now matters as much as compute for training language models. Much training data comes from human annotation of text, and interpretive annotation has no ground truth that could settle what is "accurate". Two lines of work respond to this. One combines annotators into a "ground truth" and measures how well they agree with each other; the other treats their disagreement as a signal. Both compare different people at one point in time. We measure something else: how well one reader reproduces their own reading of the same text over time. One expert human reader and three LLM families segmented three Sumerian myths and labelled the causal function of each segment with one of seven states. Across runs months apart, the human cut the text in much the same places but named the segments differently, in every myth. The models show no such consistent pattern: their gap between the two layers is positive in some myths and negative in others, and its size varies. The human's label changes are not random: the runs go through much the same functions but start them one step apart, while model runs start them at the same places. We argue that this pattern is a usable measure of data quality and a contamination check: a "human" annotation whose labels are as stable as its boundaries, and whose functions start in sync, looks like a model's. 

---
# Evidence and Intervention: A Coupled Active-Inference Extension of Rational Speech Act Models 

**Authors**: Yonghyeon Gwon, Elliot Murphy, Chun Kee Chung  

**Link**: [PDF](https://arxiv.org/pdf/2610.04347)  

**Abstract**: Identical utterance choices can arise from different communicative causes, and identical interpretations can leave different traces in what a listener learns. Rational Speech Act (RSA) models treat interpretation as inference over speaker meaning, but standard one-shot RSA does not intrinsically distinguish these causal update targets. We develop a coupled active-inference model of dialogue in which a listener's likelihood for an utterance is the policy distribution attributed to the speaker, placing the speaker's expected free energy within the listener's variational free energy. Each utterance is therefore both evidence about and an intervention on a partner. The model separates four updates that RSA approaches typically collapse: inference about the partner's current pragmatic state; learning of partner-specific parameters; prospective evaluation of clarification or repair; and a policy prior shaped by habit and a context-sensitive price of time. Under one-step, exact-inference restrictions, the model recovers the RSA speaker and listener, with RSA as the restricted single-turn limit of the coupled process. Outside these restrictions, the updates obey distinct rules and timescales, predicting which adaptations persist, remain partner-specific, or transfer. Worked examples show audience design reversing after clarification, self-confirming misunderstanding in which both interlocutors have low free energy while disagreeing about reference, and rational closing before uncertainty is resolved. Consistent with critiques of equating natural language with communication, the model treats communication as a downstream use of linguistic structure and recasts production and interpretation as coupled inference: speaking is both an intervention on a partner and an epistemic action that samples evidence for the speaker's model of that partner. 

---
# Evaluating Modeling Approaches for Experience-Level Classification in Job Description 

**Authors**: Celia Liang, Eddie Wu, Shiqi Wang, Yonah You  

**Link**: [PDF](https://arxiv.org/pdf/2610.04304)  

**Abstract**: This paper investigates the task of predicting job experience levels in recruitment texts, aiming to automatically identify the qualifications required for positions. Unlike traditional text classification, recruitment texts typically possess explicit internal structures, with different paragraphs playing disproportionate roles in conveying experience clues. To address this, we propose a structure-aware Section-Aware BERT approach that segments and encodes key paragraphs (titles, responsibilities, requirements) for integrated modeling, building upon rule-based systems and classical baselines TF-IDF. Simultaneously, we evaluate large language models under both few-shot and fine-tuning settings on the same dataset to compare the capability boundaries of different modeling paradigms. Experimental results demonstrate that explicitly leveraging text structure significantly improves experience level prediction performance, particularly in scenarios with ambiguous job titles. Further error analysis reveals systemic challenges in this task, including confusion between Entry and Senior levels and the blurred boundaries of Mid-level positions. This research provides an effective modeling approach and analytical framework for understanding structured recruitment texts. 

---
# AI-Enabled Quality Assurance for Multiple-Choice Assessment Items 

**Authors**: Steven Moore, Nicholas Diana  

**Link**: [PDF](https://arxiv.org/pdf/2610.04267)  

**Abstract**: Generating multiple-choice questions is increasingly scalable, but establishing their assessment quality remains difficult. We present a focused narrative review of automated item-writing flaw detection, revision, psychometric screening, and NLP benchmark auditing. Database searches, citation retrieval, and nominated sources yield fourteen research reports reviewed in full text. We distinguish surface checks from content-sensitive judgments and map a 19-criterion rubric to detection methods and reported evidence. High label-level accuracy often coexists with weak positive case detection, while rubric definitions and reference standards vary. Revision evidence is mixed, and the associations reported in prior work do not establish the effects of repair. We propose evaluating quality assurance as a sequence of independently validated decisions, with criterion-specific reporting, calibrated human review, and outcome-based assessment of revisions. 

---
# Playing social deduction games with reinforcement fine-tuned large language models 

**Authors**: Lingzhe Zhang, Yunpeng Zhai, Tong Jia, Kening Zheng, Chiming Duan, Minghua He, Zhaoyang Liu, Bolin Ding, Philip S. Yu, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.04261)  

**Abstract**: Reinforcement fine-tuning (RFT) is increasingly used in applications where large language models (LLMs) interact with humans and other agents. Here we use social deduction games to study how RFT changes LLMs' social behaviour. We let fine-tuned and base LLM agents play hidden-role games that require hidden-state inference, social reading and vote steering. Our results show that LLM agents do not reliably acquire social-deduction ability by directly optimizing terminal win--loss outcomes, suggesting that final game results provide a sparse and noisy signal for socially interactive learning. However, RFT is particularly effective at improving social reading, including tasks that require agents to infer hidden roles from public discussion, update beliefs over time and predict other agents' future decisions. We further show that RFT can also improve social influence, including tasks that require agents to steer votes, team approvals and collective decisions, although these gains depend more strongly on behaviourally specific rewards and structured interaction settings. Finally, we show that LLMs' ability to play social deduction games can be further improved through multi-agent social-cognitive reinforcement fine-tuning, which combines social-reading and social-influence signals during same-side multi-agent training. These learned behaviours also receive more favourable human evaluations of strategic competence, persuasiveness and social usefulness. Together, these results enrich our understanding of how RFT changes LLMs' social behaviour and provide a step toward a behavioural learning theory for machine social intelligence. 

---
# Conformal Prediction with Paraphrase-Aware Scoring for LLM Uncertainty Quantification 

**Authors**: Jiayi Xin, Evan Qiang, Zihan Zhu, Xiang Li, Weijie J. Su, Qi Long  

**Link**: [PDF](https://arxiv.org/pdf/2610.04239)  

**Abstract**: Uncertainty quantification (UQ) for large language models (LLMs) aims to provide reliable measures of predictive confidence, yet current methods are often unstable under meaning-preserving perturbations. Semantically equivalent paraphrases can induce substantial variability in predictive confidence, even for methods with formal guarantees, such as conformal prediction. To address this issue, we propose a paraphrase-aware UQ framework robust to semantic rewordings. Our approach trains a lightweight proxy model on LLM hidden states and aggregates its predictions across paraphrases to construct label-wise nonconformity scores. Under score exchangeability, conformal calibration retains marginal coverage. This guarantee can also hold under test-only rewording, provided that the paraphrase pipeline satisfies an additional distributional alignment condition. We evaluate three settings (normal, fully reworded, and semi-reworded) which apply rewording to neither dataset, both calibration and test datasets, or only the test dataset, respectively. Across seven multiple-choice QA benchmarks and multiple model families, our method produces compact prediction sets with empirical coverage generally near the nominal target, even in the semi-reworded setting. Ablation studies show that the learned proxy accounts for most of the reduction in set size, while paraphrase-augmented training and inference-time aggregation improve stability under rewording. Code is available at this https URL. 

---
# Can LLMs Separate Pasted Artifacts from User Speech? Absorption at Unmarked Prompt Seams 

**Authors**: Sugam Panthi, Muhaiminul Yeamin, Rabab Abdelfattah  

**Link**: [PDF](https://arxiv.org/pdf/2610.04210)  

**Abstract**: Large language models (LLMs) receive each user message as plain text, even when it combines text from different sources. For example, a user may paste text into a prompt and keep typing a comment directly below it. We study absorption: a phenomenon where the model treats a trailing user comment as part of the pasted text, returning it inside the edited text. This happens even though the user did not intend the comment to become part of that text. Existing instruction-data separation benchmarks tell the model which text is instruction and which is data, then test whether it obeys that separation. They do not test harmless user speech following an unmarked paste. We introduce SEAM, a controlled benchmark of 300 editing examples. Each example is tested under six matched conditions that vary how the boundary between pasted text and later user speech is expressed. Across 20 models, absorption at a bare newline ranges from 7.7% to 66.7%. Adding a blank line does not significantly reduce absorption in any model, while boundary markers reduce it in 19 of 20 models. Comments that fit the pasted text, such as a code comment typed after code, are absorbed significantly more often in 17 of 20 models. Models often fail to separate pasted material from later user speech, and explicit boundaries reduce but do not remove this failure. 

---
# Trajectory-Derived Confidence for Reliable, Resource-Aware Clinical Text-to-SQL Agents 

**Authors**: Mincheol Daniel Song, Joshua Ward, Jake Jung, Guang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.04156)  

**Abstract**: LLM agents for clinical text-to-SQL applications reason autonomously over multiple steps but cannot assess whether their own reasoning or outputs can be trusted. In high leverage applications such as healthcare, this presents a critical risk where system mistakes can be costly. These reliability failures are also resource failures: an incorrect reasoning trajectory spends computation budget on outputs that must be discarded. We introduce Sentinel, a trajectory-derived, classifier-based confidence layer that analyzes an agent's reasoning, code and database outputs to decide at three points whether to stop: refusing unanswerable questions before the agent runs, halting doomed trajectories mid-run, and withholding untrustworthy answers at delivery. Here, utilizing Chow's rule, we optimize decisions under the EHRSQL shared task's Reliability Score, which penalizes incorrect answers given a utility weighting, and find on the benchmark EHRSQL that Sentinel raises this score from +0.08 to +0.24 when mistakes have a low utility weighting, well above the +0.03 earned by refusing every question, with delivered-answer accuracy rising from 54% to 69% as coverage falls from 84% to 55%. At higher stakes the agents we test rarely answer reliably enough to deliver, and Sentinel detects this on its own, abstaining to that same +0.03 where the unmonitored agent scores -3.02. The same stopping decisions cut computation: at low stakes, where the system still answers, Sentinel eliminates 13-28% of agent steps at little or no reliability cost. 

---
# PB-GRPO: Learning Socially Adaptive LLM Agents from Persona-Driven Simulation with Preference-Batched GRPO 

**Authors**: Jingquan Wang, Jun Yin, Xu Han, Yongsheng Mei, Jie Hao, Bin Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.04132)  

**Abstract**: Building LLMs that behave well socially, not merely correctly, requires Building LLMs that behave well socially, not merely correctly, requires more than producing locally helpful responses. A socially competent agent must infer users' unstated goals, respect their preferences, and adapt as the conversation unfolds. These behaviors are inherently multi-turn and social, making them hard to optimize: real interaction data is scarce, and user preferences are typically latent rather than directly observable. To address these challenges, we build on a persona-driven social simulation environment (consisting of a persona library, LLM-based user simulators, and a user-satisfaction scoring system ranging from [0, 1]), to introduce preference-batched GRPO (PB-GRPO), a post-training algorithm that learns socially adaptive policies from conversation-level feedback. Compared to vanilla GRPO, PB-GRPO computes advantages using a normalization estimated across a bucket of users with similar preferences, stabilizing training across a diverse social population. Empirical evidence shows that PB-GRPO improves models' social behavior over strong reinforcement learning baselines in our simulated environment. 

---
# Representational Control over Self-Report & Behavior Coherence in LLM Risk-Taking 

**Authors**: Rafal Kocielnik, Peiyang Song, Pengrui Han, Myrl G. Marmarelis, Ramit Debnath, Dean Mobbs, R. Michael Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2610.04125)  

**Abstract**: Self-report is an appealing low-cost probe of an LLM's dispositions, but recent work finds only selective agreement between what models report and how they behave. Prior accounts establish these patterns by prompting black-box LLMs, leaving open whether the gap is a prompting artefact or a fact about how the underlying constructs are represented internally. We investigate risk-taking, a consequential dimension of agentic decision-making, using activation steering to measure self-report and behavior under the same internal intervention. We survey nine steering-vector extraction methods spanning task-specific directives, the model's own task behavior, and dispositional descriptions at two granularities, evaluated on two behavioral tasks and two psychometric instruments across four open-weight LLMs. We find that (1) a shared internal intervention does not ensure shared responsiveness: directions built from trait descriptions move self-report but leave behavior at chance, directions built from the model's own task choices do the reverse, and only task-specific directives reach both, weakly. (2) Diagnosis dissolves that exception: removing surface confounders leaves the directives only 32% of their behavioral effect. The two channels are otherwise steered by near-orthogonal directions, each channel reachable by several independent constructions. (3) An intervention composing one behavior-moving and one self-report-moving direction moves both together; flipping one sign sets them in opposition, with reported and enacted risk pointing in opposite directions on 69-89% of flipped compositions in all models. These findings move the self-report-behavior relationship from a black-box observation to a representational one that can be inspected and controlled, motivating representational checks alongside behavioral evaluation. 

---
# Copying Before Suppression: What Drives a Below-Chance Dip During Language Model Training? 

**Authors**: Tejas Dahiya, Cole Blondin  

**Link**: [PDF](https://arxiv.org/pdf/2610.04119)  

**Abstract**: Mechanistic interpretability usually studies fully trained models, yet the computations that drive a behaviour can change while the model is still learning the task. On the Indirect Object Identification task, a model should continue with the name mentioned once rather than the name mentioned twice. Pythia models pass through an early training window in which they prefer the repeated name, so accuracy in a choice between the two names falls below one half while language-model loss on a fixed text sample keeps decreasing across the same window. The window reflects a temporary imbalance between two computations. We identify one cause of the wrong preference by selecting a set of attention heads that write the repeated name, on prompts separate from those used for causal evaluation, keeping that selection fixed, and then replacing each head's final-token output with its average output on a separate set of non-repeated-name prompts. This improves the correct-minus-repeated logit difference in a separately trained 160M model and in the official 160M, 410M, and 1B models. At 160M, the head that lowers the repeated name in the mature model shows little of its mature behaviour at this point. It directs less than one percent of its attention to the repeated mention, and its output makes almost no direct contribution to lowering that name's logit. Both properties grow over the interval in which behaviour recovers. Across the 160M, 410M, and 1B models, transplanting the corresponding head's mature parameters into the early checkpoint recovers 35 to 68 percent of the total improvement in the correct-minus-repeated logit difference seen by the end of training. Related early-to-late reversals appear at further Pythia scales, in two independently trained GPT-2 models, and in OLMo. A mature circuit can therefore conceal a transient causal configuration that shaped behaviour earlier in training. 

---
# LongSocialBench: Do Long-Context LLMs Understand Online Discussion Threads? 

**Authors**: Xinyi Liu, Rinat Khaziev, Dilek Hakkani-Tür, Tarek F. Abdelzaher  

**Link**: [PDF](https://arxiv.org/pdf/2610.04118)  

**Abstract**: Long-context LLMs can now ingest entire online discussion threads, but understanding their social discourse requires more than reading a long document: models must track parent-reply relations, turning points, scoped subtrees, cross-branch contrasts, and participant trajectories. To test this structure-aware social reasoning, we introduce LongSocialBench, a benchmark of 1,462 verified human-authored multiple-choice items drawn from 94 complete Hacker News, Stack Exchange, and Reddit r/ChangeMyView episodes, with a median length of approximately 73K tokens. Each item pairs a complete serialized discussion and reply structure with a four-option question, requiring models to recover structured social evidence. Released items are verified for answerability, option uniqueness, and evidence grounding. Across 18 models and 29 evaluation settings, current long-context workflows remain far below human performance. The best individual result comes from Claude-Opus-4.7, which reaches 63.0% when prompted to eliminate incorrect options before answering, compared with 72.4% for independent human readers. Averaged across all 18 models, the full-context Baseline scores 43.9%. Supplying the gold evidence scope raises this to 55.0%, showing that substantial errors remain even after the relevant thread region is identified. LongSocialBench shows that the missing capability is not context access or prompting alone, but social understanding over structured reply trees. 

---
# Representation-Aligned Auxiliary Supervision for Language Model Adaptation 

**Authors**: Kyuyoung Kim, Peiyao Sheng, Ashwin Hebbar, Peiyang Xu, Yunfei Xie, Kevin Wang, Rui Xin, Chen Wei, Zhangyang Wang, Jinwoo Shin, Pramod Viswanath, Sewoong Oh  

**Link**: [PDF](https://arxiv.org/pdf/2610.04098)  

**Abstract**: Language models exhibit strong reasoning capabilities, yet adapting them to structured domains remains challenging and can yield inconsistent outcomes. We identify representation compatibility, the extent to which a model effectively processes a representation for a structured task, as a key factor in adaptation. We study this in chess, which provides a controlled testbed with precise semantics, computable optimal actions, and multiple state representations, including a symbolic encoding (FEN) and a spatial format (ASCII). We find that models often process semantically equivalent inputs substantially differently, affecting both learning and generalization. Building on this observation, we propose representation-aligned auxiliary supervision, which uses environment-derived tasks expressed in compatible representations to improve adaptation to structured domains. Across models and representations, auxiliary supervision consistently improves optimal-move prediction relative to target-only training under identical target data. Tasks that expose environment dynamics provide larger and most consistent gains than surface-level or static supervision, while remaining competitive with substantially increasing the amount of target-task data. Moreover, ASCII-trained models transfer more effectively to FEN than FEN-trained models do to ASCII, even surpassing the FEN target-only baseline on FEN evaluation. The gains also extend beyond optimal-move prediction to open-ended, factually grounded commentary generation. Overall, our results show that auxiliary supervision in model-compatible representations can enable effective adaptation in structured domains. 

---
# IdeaScientist: Orchestrating Agents for Grounded Scientific Ideation 

**Authors**: Jiarui Liu, Renjie Tao, Yiwei Liao, Chuanyang Jin, Kai Sun, Xiao Yang, Xinyuan Zhang, Xilun Chen, Zhuangqun Huang, Lechen Zhang, Yongjin Yang, Yinghui He, Weihao Xuan, Rakesh Wanga, Anuj Kumar, Mona T. Diab, Wen-tau Yih, Xin Luna Dong  

**Link**: [PDF](https://arxiv.org/pdf/2610.04074)  

**Abstract**: Despite rapid progress in automating scientific research, generating promising and well grounded research solutions remains a central challenge. We isolate research ideation as a standalone task and build our solution on the intuition that a challenge in one field can often be addressed by a mechanism that solved an analogous challenge in another. Accordingly, we introduce IdeaScientist, which decomposes ideation into gap finding, innovation, and report writing, and trains each role with reinforcement learning. These roles identify limitations in related work, draw solution intuitions from analogous problem settings, and develop those intuitions into complete research proposals. To facilitate discovery of insights across domains, we construct the Svalbard Idea Vault, a corpus of 2.77M decomposed research ideas for retrieval, training, and temporally controlled evaluation. Our evaluation restricts access to literature available before a cutoff date and assesses how closely proposed directions align with those later explored in 15K papers authored by human researchers. On Qwen3.6-27B, IdeaScientist outperforms the strongest open-source autoresearch baseline by 14.0%, driven mainly by gains in novelty. On this 27B open backbone, IdeaScientist even outperforms Claude Code SDK with Claude-4.8-Opus and Codex SDK with GPT-5.4, by up to 5.9%. 

---
# Periscope: Extending Frozen Language Models Beyond Their Context Window 

**Authors**: Mohamed Eltahir, Anas Obayd, Raed Rashid, Abdulrahman Alghamdi, Abdulrahman Mousa, Abdallah Ahmed, Tanveer Hussain, Naeemullah Khan  

**Link**: [PDF](https://arxiv.org/pdf/2610.04047)  

**Abstract**: A language model reads long text in one quadratic forward pass, stops at the context window, and loses accuracy with length before reaching it. We ask whether the read can be factorized when deciding over a finite set: which document is relevant, which option is supported, which passage is the evidence. Periscope, a training-free inference method, arranges the $N$ chunks of a text on a $K{\times}K$ grid with $K{=}\lceil\sqrt{N}\rceil$ and asks a frozen model the same question about $K$ local spans of consecutive chunks and $K$ strided spans that sample the whole text, reading the log-odds of every answer at one token. Each answer takes its best local and strided score, and scoring every chunk by its two spans gives an evidence map at no further cost, whose peak is the chunk behind the answer. Every probe is about $\sqrt{sc}$ tokens for a text of $s$ tokens and chunk size $c$, so a window of $W$ tokens reaches $W^{2}/c$ tokens at $s^{1.5}$ cost. The map replaces the long read. On LongBench v2, reading only the $K$ chunks the map ranks highest, 9k tokens, matches the same model's best window read across windows from 32k to 1M tokens, and on InfiniteBench, where the median context is 150k tokens, it leads the best window read by 5 points. The same map ranks BRIGHT's long-document corpora with the best NDCG@10 of six methods. Each call caches only one probe, so a 27B model reads 4.5M-token contexts on one 80GB GPU, where a single pass would need 296GB of cache. A long read then needs a GPU that holds the model, not one that holds the text. 

---
# Do Language Models Need a Trainable Input Embedding Table? Fixed Minimal Token Codes at 1.7B-Class Scale 

**Authors**: A. Bochkov  

**Link**: [PDF](https://arxiv.org/pdf/2610.04002)  

**Abstract**: A trainable input embedding table assigns each vocabulary item an independently adjustable vector. We investigate whether this token-specific parameterization is required for substantial language-modeling capability, or whether a shared Transformer can learn from fixed token identities. We compare three decoder-only language models trained from scratch with the same tokenizer, contextual backbone, untied output-head architecture, and training recipe, with a target budget of 100 billion prediction tokens per model. Their input interfaces are a learned table, canonical 16-bit token-ID codes, and one fixed invertible recoding over GF(2). The fixed codes are repeated to model width without an additional trainable input projection. Both fixed-code models acquire substantial capabilities: canonical codes achieve 52.40\% HellaSwag normalized accuracy, 70.51\% PIQA accuracy, and 42.75\% LAMBADA accuracy. The learned-input control performs better on several evaluations, including HellaSwag and LAMBADA, so these results establish viability rather than performance parity. The fixed interfaces remove 100.7 million trainable parameters, yielding 1.711B-parameter models, but parameter reduction is not the central result. These single-run experiments distinguish architectural necessity from empirical utility: independently trainable token-specific input vectors are not required for the observed capabilities. A fixed identity interface also provides a controlled setting for studying representation learning downstream of an immutable input, without establishing where particular capabilities are localized. 

---
# BAIBAICHUCHU at the NTCIR-19 FinArg-3 Task: When Is Maximum Possible Profit Predictable from Investor Text? 

**Authors**: Zong-Han Bai, Po-Yen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2610.03962)  

**Abstract**: The BAIBAICHUCHU team participated in the Social Media Subtask of NTCIR-19 FinArg-3, ranking Chinese investor posts by Maximum Possible Profit (MPP). A three-track ensemble of lexical features, a FinArg-2-pre-finetuned MacBERT ranker, and an LLM judge reaches 0.734 in post-grouped development evaluation, but our best official run scores 0.517. All twelve submitted runs lie between 0.4598 and 0.5402, and our 26 unanimous three-track pairs score 0.500.
A post-hoc audit finds that the submitted judge applied a long-only rule to bearish posts although MPP is stance-aware. Correcting it changes 28 of 87 official predictions without changing accuracy, yet lowers development accuracy from 0.680 to 0.622: a regime-specific semantic shortcut improved validation fit.
We decompose ranking into directional text, volatility and horizon, and pairwise margin. A running-extremum model predicts $\sigma\sqrt{T}$ scaling, observed ex post in 502 price-aligned posts from a separate July 2026 collection. Pre-posting volatility scaled by the post-specific observation horizon, $\sigma_{\mathrm{pre},20}\sqrt{N_i}$, is associated with a later truncated-horizon MPP proxy (Spearman $\rho=0.320$; ticker-cluster 95% CI [0.133,0.466]) and correctly orders 61.4% of unequal-outcome pairs. Transferred text is nearly uncorrelated with the July outcome ($\rho=0.055$) and adds little conditional on this score and stance. Development reliability rises from about 0.60 to 0.93 as the labeled MPP gap widens. An earlier ERAI result of 0.6207 rules out universal unpredictability as a simple explanation. Evaluation should jointly consider text, market regime, historical volatility, horizon, and pair composition. 

---
# A Step Towards Forgetting: Optimiser History and the Loss of Answer Mass 

**Authors**: Valeria Ruscio, Seth Nabarro, Keiran Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2610.03940)  

**Abstract**: During fine-tuning, a language model can assign less probability to previously learned answers even when the current gradient acts to preserve that probability. With momentum, each update also carries gradients computed at earlier model states, and these stored contributions can push the model in the opposite direction. We investigate how this optimiser memory contributes to forgetting by separating old-task loss into confusion among its answers and leakage of probability outside the answer set. Across three language-model families, answer mass consistently declines while discrimination among old answers usually improves: the model becomes less likely to produce answers that it can still distinguish correctly. Decomposing Adam updates reveals opposing contributions to this loss of answer mass. Over training, accumulated history favours leakage, while the current gradient opposes it. Resolving history by age shows that the harmful contributions come mainly from older gradients of the new task, whereas recent gradients tend to protect the old answers. Changes in history's effect are dominated by its orientation relative to the old-task gradient. Interventions that reset momentum while matching the initial update norm establish that stored history affects retention, with state-dependent immediate effects and lower final old-task loss over longer Adam continuations, mainly through recovered answer mass. Finally, integration along finite updates shows that most sampled large loss increases are captured by local projections, while curvature along history amplifies some events. Together, these findings reveal how an optimiser's memory can erode learned behaviour even as its current gradient acts to preserve it. 

---
# General Decision Models: Benchmarking and Insights Beyond Jev 

**Authors**: Feiyu Duan, Jiayu Lin, Jia Wang, Jun Xiang, Jialiang Wu, Xinnong Zhang, Hanqi Yan, Siyuan Wang, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2610.03935)  

**Abstract**: General decision models, such as Jev, have recently emerged as efficient alternatives to LLMs for structured judgment and selection. But what kinds of decisions can these models reliably make, and how does their behavior change when individual decisions are composed into larger systems? To study this, we introduce JEVal, a bilingual benchmark comprising 11,257 instances from 36 datasets across 10 application domains, and evaluate 25 model configurations spanning general decision models and generative LLMs. Our results show that (1) general decision models are most competitive when decisions can be resolved from available evidence, but weaken when they require specialist knowledge or faithful uncertainty estimation: they can often identify the most likely outcome while substantially overstating its probability. (2) In more dynamic and realistic systems involving long-horizon, multi-step interactions, the advantages of fast local decision making are offset by reliability failures at the system level. on $\tau$-bench, faster local decisions reduce median episode time but lower task success as decision errors accumulate over long trajectories. (3) In large-scale social simulation, decision models approach strong generative LLMs on individual response prediction at substantially lower inference cost, yet remain weaker in user profiling and exhibit larger aggregate estimation errors and systematic bias. Finally, we propose InnerJev-4B and InnerJev-27B, which internalize an open-weight LLM's own reasoning into a single-pass first-token decision through Reasoning-to-Readout Self-Distillation, with InnerJev-27B performing on par with Jev on JEVal while answering a typical query in about 0.1 s. 

---
# When Evidence Changes: Evaluating Memory Repair and Re-reading in Language-Model Agents 

**Authors**: Wenhui Chu  

**Link**: [PDF](https://arxiv.org/pdf/2610.03902)  

**Abstract**: When documents supporting an agent's derived facts are revoked or replaced, should it repair memory or re-read current evidence? We introduce an evidence-revision evaluation on medication- and problem-list tasks from public ICU records. Under revocation, replacement and control events, we compare full and source-filtered re-reading with caching, rebuilding and graph-local repair across two 7B models. Memory is supplied in full without retrieval, and costs include ingest, revision and every use. On short records, local repair uses 5-10$\times$ fewer revision tokens than rebuilding, yet every memory pipeline costs at least twice full re-reading in held-out conditions. In a small pre-specified development sweep, adding task-ineligible documents extended records to about 10,000 tokens; at that length, memory's mean cumulative cost fell below full re-reading's after 2-14 uses, partly through truncated extraction, while source-filtered re-reading remained cheapest. In the replacement study, none of the four primary confirmatory tests reached statistical significance. These results show why the cost of agent memory after evidence revision must be assessed against source-filtered re-reading over the full pipeline. 

---
# SYNLAT: Syntax-Aligned Text-Latent Compression for Chain-of-Thought Reasoning 

**Authors**: Yifeng Zhao, Hongjun Yu, Shibo Wang, Yunjiao Zhou, Zixiao Zhu, Zhipeng Ning, Kezhi Mao, Junlang Qian  

**Link**: [PDF](https://arxiv.org/pdf/2610.03839)  

**Abstract**: Long chain-of-thought (CoT) traces impose substantial output-token costs. Under constrained budgets, compression must preserve answer-critical information, making boundary placement central. Token-level and fixed-length boundaries can fragment coherent spans such as phrases, formulas, and local derivations, whereas step-level boundaries can bind content requiring different compression actions. We introduce SynLat, a text-latent CoT framework that aligns compression boundaries with syntactic structure through non-overlapping Syntax-Aligned Units (SAUs). An answer-conditioned Teacher constructs progressive KEEP/LATENT targets for a single compression-conditioned Student, which generates mixed reasoning from only the question and requested compression level at inference. Across two Qwen3 Student scales, Standard-CoT and Long-CoT groups, and three compression levels, SynLat matches or exceeds the strongest evaluated baseline in all 12 task-group aggregates and strictly leads in 11 under the reported achieved-CR selection protocol. Overall gains reach 3.6/2.6 points at MEDIUM and 7.0/5.5 points at HIGH for Qwen3-8B/14B, with larger advantages under stronger compression, particularly on Long-CoT groups. 

---
# OncoNoteBERT: A Foundation Representation Model for Natural Language Processing of Real-World Outpatient Oncology Notes 

**Authors**: Wuraola Oyewusi, Eliana Vasquez Osorio, Goran Nenadic, Gareth Price  

**Link**: [PDF](https://arxiv.org/pdf/2610.03829)  

**Abstract**: Real-world outpatient oncology notes contain specialised terminology, tumour staging expressions, treatment names, toxicity descriptions, and institution-specific de-identification markers that may not be represented efficiently by general biomedical or adjacent clinical language models. We developed and evaluated oncology-specific BERT-style encoders using a governed UK outpatient oncology corpus comprising 290,026 notes from 21,564 patients treated for lung and head-and-neck cancer. We compared RadBERT and PathologyBERT with two local strategies: OncoNote-RadBERT, produced by continued masked language model pretraining, and OncoNoteBERT, trained from scratch with an oncology WordPiece tokenizer. Models were evaluated using masked language modelling loss and perplexity on the validation set, tokenizer fragmentation metrics, clinical term tokenisation, masked-token probes, and exploratory representation analysis. Both external encoders fit the oncology corpus poorly in zero-shot evaluation (perplexity 113.04 for RadBERT; 2035.03 for PathologyBERT), while continued pretraining produced the strongest fit (2.10 for OncoNote-RadBERT). OncoNoteBERT achieved perplexity 2.83 but produced the most efficient tokenisation, with lower subword fertility and shorter normalised sequence length. It also returned a clinically acceptable prediction for 12 of 13 masked-token probes, compared with 7 of 13 for OncoNote-RadBERT. This divergence between corpus-level fit and masked-token performance was partly attributable to tokenizer fragmentation rather than learned semantics alone. Both locally developed models represented the institutional placeholder as a single learnable token. These findings show that continued adaptation and bespoke tokenisation provide complementary benefits, and that representation-layer design matters before adjacent-domain encoders are applied to oncology NLP. 

---
# The Score Is Not the Structure: Brain Alignment and Cross-Lingual Transfer 

**Authors**: Saman Rahbar  

**Link**: [PDF](https://arxiv.org/pdf/2610.03827)  

**Abstract**: Researchers often support the claim that a model shares structure with the brain, or across languages, by reporting a similarity score. We ask what such a score reads when the shared structure is absent, or when the tool that measures it does not work. We check two settings, and in both the score is not what it appears. First, a probe trained to tell grammatical from ungrammatical sentences in one language transfers worse to more distant languages, the usual evidence for shared structure. But the probe itself gets worse along the same axis: in four of seventeen languages it performs at chance, so 64 of 272 language pairs are scored with a tool that does not work. Dropping those languages halves the strength of the relationship, but they are also the most distant, and this design cannot separate the two effects. Counting matters too: the same data give p = 0.0006 when the 272 pairs are treated as independent and p = 0.155 when the seventeen languages are, which is the correct unit. Second, training a language model to match human brain responses raises its similarity score from 0.10 to 0.34, against a ceiling of 0.54. A model trained on a target whose correspondence to the brain was destroyed still scores 0.31, so only 0.028 to 0.068 of the rise is specific to the brain. With no model at all, a destroyed target already sits at 0.204 from the real one, a floor that tracks the target's rank divided by the number of sentences. Finally, steering a language along its own direction works (+6.2 over a random direction in sixteen of seventeen languages) yet shows no effect that varies with language distance. Before asking whether a correspondence helps, ask how much of the score would survive without it. 

---
# Same Output, Different Gold: Measuring How Reference Choice Moves a Multilingual Benchmark Score 

**Authors**: Parth Kulshreshtha, Shivali Dalmia, Abhishek Mukherji  

**Link**: [PDF](https://arxiv.org/pdf/2610.03825)  

**Abstract**: A benchmark score compares a system output against a reference, and methodological attention falls almost entirely on the first term. We measure the second. The retained annotation record of a six-language benchmark for personally identifiable information contains two independent annotator labellings, the aggregate shipped as gold, and a reviewer gold from independent expert re-annotation of a sample. Using it, we hold the scored output fixed and exchange the reference. The score moves by 4.95 F1 points for one output and 2.00 for the other (95% CIs [3.23, 6.27] and [0.26, 3.37]), and by 7.55 in the worst language. The comparison between two outputs moves as well: the paired interaction between reference and system is +2.95 points (CI [+1.97, +3.87]), survives correction for multiple testing, and changes one language's margin outright. The cause is an undocumented aggregation default that usually kept one annotator when adjudication did not fire, making the shipped reference a partial copy of an output being scored. We report the resulting reference-sensitivity band, show how to compute one from any retained annotation record, and argue that the quantity belongs beside the score. 

---
# Fine-Grained Emotion Classification from Mobile App Reviews: An Empirical Study with Large Language Models 

**Authors**: Quim Motger, Carlota Catot, Marc Oriol  

**Link**: [PDF](https://arxiv.org/pdf/2610.03802)  

**Abstract**: Context: Fine-grained emotion classification of mobile app reviews enables requirements engineering activities that go beyond polarity-based opinion mining, including emotionally informed issue prioritisation and feature-oriented feedback analysis. However, automatic fine-grained emotion extraction from app reviews remains understudied. Objectives: Building on a previously published annotation framework and human-labelled ground truth adapted from Plutchik's taxonomy, this paper investigates how large language models can be leveraged for automatic multi-label emotion classification under severe class imbalance. Methods: We compare encoder-only fine-tuning under multi-label and binary-ensemble formulations, decoder-only zero- and few-shot prompting across open-source and proprietary models, and a catalogue of imbalance mitigation strategies (loss reweighting, resampling, generative data augmentation), with the synthetic-review generator and prompting strategy selected via an intrinsic augmentation-utility ranking. Results: Fine-tuned encoders trail the best decoder-only few-shot prompting (macro-F1 0.642) by a wide margin at baseline (multi-label: 0.387; binary ensemble: 0.450); pairing the best multi-label encoder with generative data augmentation and positive-weighted loss closes most of this gap (+0.204) at up to three orders of magnitude lower inference latency than the decoders, with the largest gains on the rarest emotions, from undetected to gains of up to +0.501 F1. Conclusion: Large language models make fine-grained, multi-label emotion classification of app reviews feasible for requirements engineering pipelines, with modest macro-F1, and the best formulation and mitigation strategy are backbone- and formulation-dependent. We release the experimental pipeline, synthetic corpora, and fine-tuned checkpoints for replication and reuse. 

---
# Base Models Can Reason By Taking a Cue From Training Data 

**Authors**: Sophie L. Wang, Amil Dravid, Rulin Shao, Kevin Farhat, Sewon Min, Alexei A. Efros  

**Link**: [PDF](https://arxiv.org/pdf/2610.06851)  

**Abstract**: In this paper, we study how training data creates associations between the tokens at the start of a base model's response and the reasoning behavior that follows. First, we demonstrate that fixing particular starting token cues makes a base model's performance competitive with that of its reinforcement learning (RL)-trained counterparts on math and coding. For instance, the cue ".\n\nOkay" raises Olmo-3-7B's MATH-500 pass@1 accuracy from 42% to 78%, while "Alright," raises Qwen3-14B's from 72% to 87%. Second, RL makes these cues more likely, while fixing them recovers much of its performance gain over the base model. Third, we trace the reasoning effects of token cues to the training data. We perform causal data interventions to turn an arbitrary word, such as "chicken", into an effective reasoning cue, or remove an existing cue's effect. A similar edit makes the prompt instruction "Think duck duck goose" as effective as "Think step by step" at eliciting reasoning. We also find that the hidden state representations induced by different cues correlate with different document types from the training set. Finally, we extend our study of token cues with a case study in language model safety, finding that different cues elicit distinct refusal and compliance behaviors that correspond to different types of training data. 

---
# Recursive Video In-Context Learning for Agentic Robot 

**Authors**: Wenrui Bao, Xinxin Liu, Bingxin Xu, Yuzhang Shang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06843)  

**Abstract**: LLM agents that orchestrate frozen vision-language-action (VLA) policies improve across episodes through text memory, which records what the agent did but not how the task is done. A demonstration video shows it, but fits poorly into an agent's context. The full video slows every turn, fixed keyframes lose the contact detail that decides whether a grasp holds, and what the agent needs shifts from the task's structure while planning to the frames around each contact. We introduce Recursive Video In-Context Learning (RV-ICL), a training-free method that turns a demonstration into a hierarchy the agent navigates rather than a prompt it receives. The hierarchy is built from the sub-events of the demonstration, such as grasps and releases. Its levels grow finer, from keyframes of the whole task to phases, moments and short clips, and are exposed through read-only tools. The agent reads the coarse levels before planning. During execution it re-enters the hierarchy whenever a step needs more detail and loads only the clip of its current sub-goal. One demonstration per task is enough. Built on RPent, RV-ICL raises success from 92.6% to 96.5% on LIBERO-PRO and from 86.7% to 95.8% on LIBERO-Plus. 

---
# Paradee: Distilling Kokoro-82M into an 8M-Parameter Single-Voice Text-to-Speech Model 

**Authors**: Sahil Mahendrakar  

**Link**: [PDF](https://arxiv.org/pdf/2610.06817)  

**Abstract**: We distill Kokoro-82M, a widely used open text-to-speech model with 54 voices, into Paradee, an 8.07M-parameter model that speaks one of them. Paradee keeps Kokoro's architecture with much narrower layers, and each of its two halves is trained separately against the frozen teacher. It has 10x fewer parameters and needs 15x less compute. We first synthesize a corpus with the teacher and keep its durations, pitch, energy and phoneme features. We then train a small text side to predict these values, and a small decoder to turn the teacher's saved values into the teacher's audio, first with spectral losses and then adversarially. Finally, we connect the two halves and quantize the weights to int8. It needs no alignment learning and no joint training, and it runs on one laptop. Stored in int8, Paradee is 8.5 MB, runs 25x faster than real time on one CPU thread, and scores 4.41 on UTMOS against the teacher's 4.52. The student initially kept a slight buzz, which we trace to the phase of voiced speech between 2 and 8 kHz. A phase-locking filter applied after synthesis removes most of it, with no training and no extra parameters. Code, model files and audio samples are at this https URL 

---
# Reading the Mood: Emotion-Guided Book-to-Music Recommendation via CGANs and LLMs 

**Authors**: Manousos Linardakis, Georgios Alexandridis  

**Link**: [PDF](https://arxiv.org/pdf/2610.06703)  

**Abstract**: Background music that matches the mood of a text has been shown to make readers feel more immersed and improve their reading experience, motivating recommender systems that pair books with mood-matched music. In this direction, we present Sentiment Aware Generative Adversarial Network for Cross Domain Recommendation (SAGA-CDR), a two-phase cross-domain recommendation framework that personalizes music suggestions and emotionally aligns them with the book being read. In the first phase, transformer-based sentiment embeddings are constructed from user reviews and mapped across domains via a Conditional Generative Adversarial Network, whose mask-conditioned generator handles missing sentiment components and injects stochasticity for richer preference transfer. A compact rating neural network then fuses sentiment-specific interaction scores with a collaborative filtering prior to predict music ratings. In the second phase, large language models classify each book into a valence-arousal emotional quadrant, and candidate tracks are filtered to match that quadrant. Experiments on both the English Amazon and Chinese Douban datasets show that SAGA-CDR achieves the best rating prediction accuracy on Amazon (RMSE 0.98) and the lowest RMSE on Douban (0.91), with ranking performance competitive with the strongest sentiment-aware baseline, even in cross-lingual settings. 

---
# Aligning Multimodal Patient Evidence with Biomedical Knowledge Graphs for Clinical LLMs 

**Authors**: Jiawen Du, Arshan Ali Khan, Chenhao Zhang, Zachary Plotkin, Li Shen, Qi Long, Yun Li, Can Chen, Tianlong Chen, Nicholas Konz  

**Link**: [PDF](https://arxiv.org/pdf/2610.06685)  

**Abstract**: Clinical questions often depend on linking a patient's multimodal evidence to external biomedical knowledge, yet existing predictive systems rarely represent such links explicitly, so they can neither be traced to their evidence sources nor removed to measure their contributions. We present MM-KG (Multimodal Knowledge Graph), which represents heterogeneous, multimodal patient observations and biomedical concepts as separate layers in one typed graph, joined by explicit alignment edges. First, modality-specific harmonizers convert EHR text, imaging, genomic, and biospecimen data into typed observations mapped to UMLS concepts, which a route-prioritized aligner links to a biomedical knowledge graph. Query-conditioned retrieval then selects a compact subgraph for downstream use by a large language model or a graph neural network. We build MM-KGs for MIMIC-IV and ADNI, and evaluate them with a 2x2 design that separates patient evidence, biomedical knowledge, and their interaction. On questions that require both sources, neither source alone performs far above chance, whereas their combination yields a drug-controlled AUROC interaction of +0.194 on MIMIC and +0.299 on ADNI. On held-out five-candidate ranking, MM-KG outperforms MindMap by +0.131 Hits@1 and leads an adapted GraphCare on the items that require consulting the patient, and deleting the single answer-bearing relation from the retrieved packet returns Hits@1 to the no-knowledge baseline. Finally, query-conditioned retrieval reaches 0.731 AUROC with 6.8x less context than the strongest generic policy, whereas static knowledge graph context gives no consistent gain on ordinary outcome prediction. Knowledge graphs thus benefit clinical LLMs not as background context but as explicit links between multimodal patient evidence and the relation a question requires, and MM-KG makes these links retrievable, traceable, and testable. 

---
# How Sparse Probability Maps Shape Mixture-of-Experts Routing 

**Authors**: Tomás Brogueira, Marcos Treviso, Miguel Couceiro  

**Link**: [PDF](https://arxiv.org/pdf/2610.06677)  

**Abstract**: Mixture-of-experts (MoE) routers typically apply softmax to the router scores and keep the top-K experts, making every token use exactly K experts. Sparsity-inducing probability maps such as sparsemax, alpha-entmax and normmax can adaptively assign exact zeros to selected experts, and therefore appear to offer token-dependent expert participation, even when using the same top-K machinery. In this work, we study whether and how this sparsity survives training. We train matched 300M and 1B top-2 MoE language models with softmax, 1.5-entmax, sparsemax and 2-normmax, and find that the maps behave very differently once trained: at 1B, entmax discards 30% less probability mass than softmax while almost never dropping a selected expert, sparsemax retains the most mass, and normmax routes 21% of tokens to a single expert. These outcomes are not properties of the maps alone. Each map drops a selected expert only when the gap between the two largest scores reaches a fixed threshold, and the trained routers differ in the score distribution they learn: the entmax router learns scores with roughly half the spread of softmax's, which keeps its top-2 gaps below its threshold, while sparsemax and normmax, which share the same threshold, learn different gap distributions and hence different participation. Routers thus co-adapt their scores to the map, and a map's capacity to produce zeros does not by itself determine expert participation. While none of the sparse maps improves validation loss over softmax, they make the trained models far less sensitive to selecting more experts at inference: sparsemax trained with K=2 loses 0.02 nats when run with K=8, where softmax loses 0.58. Our results indicate that adaptive MoE routing has to be designed around the joint behavior of the probability map and the learned scores, rather than around the map alone. 

---
# The Pushback Paradox: A Two-Probe Diagnostic for Language Model Compliance 

**Authors**: Stefan Bühler, David Exler, Markus Reischl, Mark Schutera  

**Link**: [PDF](https://arxiv.org/pdf/2610.06673)  

**Abstract**: Are language models compliant with user instructions? A model that always complies can be stopped but also exploited, while one that always resists can be neither exploited nor stopped. We contribute an open two-probe benchmark that can place any language model on this spectrum. In the active probe, a user instructs the model to act and accept a lower payoff, which measures exploitability. In the passive probe, the user instructs it to wait and give up a higher payoff, which measures stoppability. The two compliance rates combine into a compliance index $\kappa$. Applied to twelve language models, the benchmark shows that seven mostly follow the instruction in both probes and justify their action by pointing to the instruction. Only Claude Sonnet-4.6 and Claude Opus-4.7 can be stopped without being exploitable, Claude Opus-4.6 and GPT-5-mini resist both instructions, and no model is exploitable but unstoppable. Knowing where a language model sits on the compliance index $\kappa$ matters for human operators and for multi-agent systems, whether distributed or orchestrated. 

---
# Language models can notice an impossible engineering problem yet still report it as solved 

**Authors**: Shaoliang Yang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06668)  

**Abstract**: Language models draft engineering calculations, but answer accuracy does not show whether they reject an impossible problem. We tested 14 models on 30 pairs of mechanics problems, each with a valid version and one made impossible by changing a given value or assumption. Two independent solvers verified every answer key and showed that each flawed problem was physically impossible. We scored solving of valid problems separately from rejection of their flawed counterparts. Each reply required a "solved" or "cannot solve" status; rejection meant "cannot solve" or withholding an answer. The initial prompts did not warn that problems could be flawed. Across three recent models, 12 of 90 replies failed to reject a flawed problem. In 11 of these replies, the model stated the flaw, answered a corrected problem and still reported the original as "solved", according to artificial intelligence raters and numerical checks. We later retested four models from one provider, offering "flawed" instead of "cannot solve" and asking them to name and explain the defect. Three models showed statistically significant increases in rejection, but valid-problem solving fell in three. Evaluations therefore need to score both versions and distinguish flaw recognition from the reported status. 

---
# What Matters for Latent Reasoning with Flow Matching 

**Authors**: Yassine Ouali, Adrian Bulat, Georgios Tzimiropoulos  

**Link**: [PDF](https://arxiv.org/pdf/2610.06666)  

**Abstract**: Latent reasoning lets a large language model (LLM) think in a continuous space and verbalize only the answer. We argue that an effective latent thought must meet five requirements: it should be useful, helping produce the correct answer rather than merely changing it, diverse, so that resampling yields different reasoning trajectories, explainable, so that a decoded chain of thought (CoT) reflects reasoning the answer actually follows, refinable with more inference compute, and efficient, costing less than an explicit CoT at comparable accuracy. Current methods rarely meet these requirements: they learn shortcuts from the question, distill the explicit CoT into their weights, or imitate it one token at a time. We focus on flow matching in a learned latent space, the family we argue is best placed to meet them, and identify the training choices that make it work. The result is Flow-based Latent Reasoning (FLaRe), a simple recipe covering what the latent space encodes and how to shape it, where to train the flow, how to read out the answer, and a final stage of training on the model's own verified thoughts. A probe for each requirement shows that FLaRe improves on prior latent methods in all five. It also compares favorably with them on arithmetic benchmarks, while reaching 97% of the accuracy of explicit CoT at a quarter of its latency. 

---
# Frozen Factor or Spectral Band? Disentangling Two Choices in Low-Rank LoRA 

**Authors**: Adnan Slimane Ali, Ayoub Belfatmi, David Ngwe Pouth  

**Link**: [PDF](https://arxiv.org/pdf/2610.06621)  

**Abstract**: Spectral variants of low-rank adaptation (LoRA) choose both a subspace and which factor to freeze. We separate these choices by freezing the input factor A or output factor B on the top or bottom singular directions of pretrained weights, with learning rates selected separately. At rank 2, the same-band advantage of freezing A is larger than either within-factor band difference on all four task-model pairs with complete comparisons. Freezing B also trails comparable-budget free LoRA by 8-18 percentage points on five pairs spanning a formatting task and OpenBookQA. The A-frozen advantage persists in a single-GPU-model replication and within individual MLP module groups, including controls with equal or greater trainable counts for B frozen, and when A is frozen on a random orthonormal basis. The factor contrast weakens with rank. On OpenBookQA / Qwen2.5-1.5B at rank 16, PEFT's MiCA implementation trails comparable-budget LoRA by 3.08 points under a shared training recipe transferred from the MiCA paper. A trained oracle output subspace largely removes the low-rank deficit; partial warm-up gains recur across three direction seeds. The factor-versus-band ordering is descriptive; an approximate multiplicity audit weakens several earlier significance claims. These results extend known factor asymmetry by showing how its magnitude depends on spectral placement, rank and training conditions. 

---
# COMPASS 2.0: psychometric representational similarity analysis distinguishes symptom structure from personal signal 

**Authors**: Baihan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.06615)  

**Abstract**: Language models can score psychiatric questionnaires from speech, but agreement with self-report may reflect the questionnaire rather than the person. We introduce psychometric representational similarity analysis, a framework for comparing the structure of speech-derived scores, self-report, item wording and theory, and implement it alongside person-level construct scoring in COMPASS 2.0. We show how similarly worded items induce covariance without psychological signal. In pre-registered discovery and confirmation analyses of clinical interviews from 275 participants, language-derived symptom geometry resembled wording more than self-report, with no structure beyond wording detected by the registered tests. Geometric agreement with self-report survived assigning participants someone else's answers, whereas person-paired scores captured distress more than specific symptoms. Complementary analyses examined counselling quality and wording structure across 34 instruments and the Research Domain Criteria (RDoC) framework. These findings distinguish agreement about psychological structure from evidence that language-derived assessments track individual people. 

---
# Mind the Accent Gap: British Accent Robustness in Speech-Driven Financial Voice Assistants 

**Authors**: Aadam Haq, Oggi Rudovic, Malcolm Chadwick, Jay Rainey, Shucong Zhang, Ricardo Guerrero, Sourav Bhattacharya, Maja Pantic  

**Link**: [PDF](https://arxiv.org/pdf/2610.06587)  

**Abstract**: AI voice assistants often use Automatic Speech Recognition (ASR) with LLM-based reasoning, yet existing systems struggle with regional British accents, including Scottish, Irish, and Welsh accents, since most ASR models are trained predominantly on American English voice data. Consequently, errors can carry through to the LLM stage, corrupting tool-call arguments and producing wrong or missing responses, which is especially costly in finance. Deployable ASR must also meet tight latency and memory budgets, making an accent-robust model choice even harder. We introduce CavaBench, the first internally collected benchmark of spoken financial queries, and use it to evaluate a range of ASR models and their end-to-end ASR-LLM pipeline behaviour across self-reported British accents. We find that WER strongly predicts downstream tool-calling accuracy ($r = -0.93$) but can fail to reflect task-level performance, with accent-related failures varying substantially across models and acoustic conditions. These findings guide the design of more inclusive, reliable voice-based financial assistants. 

---
# Mind the Execution Gap: Action-Semantic Mismatch in World-Model Control 

**Authors**: Shengtao Wen, Xiang Chen, Yu Tian, Lingbing Guo, Lina Gong, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06582)  

**Abstract**: World-model controllers rely on action-conditioned dynamics for prediction and planning, yet real control systems often execute commands asynchronously due to communication delay, packet loss, reordering, and actuator buffering. We study how asynchronous execution changes the action semantics assumed within world-model controllers, rather than treating it only as an external control disturbance. Through controlled interventions, we identify two architecture-dependent failure modes: planning-based controllers such as TD-MPC2 suffer from a future-action timeline mismatch between imagined and executed action sequences, while recurrent world models such as DreamerV3 can attribute observed transitions to commands that were not actually applied. Our analysis shows that TD-MPC2 requires the correct future action sequence during latent dynamics rollout, whereas DreamerV3 requires timely attribution of each transition to the action that generated it. Based on these findings, we introduce two lightweight execution-consistent interfaces, Future-Sequence for TD-MPC2 and Applied-Action Feedback for DreamerV3, that correct these mismatches without modifying the pretrained world models. Experiments across delays, packet loss, reordering, multiple control domains, measured network traces, and a process-separated asynchronous stack consistently support both diagnoses and the corresponding architecture-specific corrections. 

---
# Synthetic Cultural Agents from Aggregate Anchors 

**Authors**: Augusto Gonzalez-Bonorino, Kseniia Biriukova, Monica Capra  

**Link**: [PDF](https://arxiv.org/pdf/2610.06562)  

**Abstract**: Population prompts are widely used to generate synthetic survey responses, but they combine information supplied at inference with associations already encoded during pretraining. We introduce an alternative construction that maps declared aggregate preference anchors into group-indexed choice policies. For each population, the signs of six Global Preferences Survey (GPS) coordinates deterministically label a shared bank of paired synthetic responses, and Direct Preference Optimization fits a parameter-efficient adapter to those comparisons. We evaluate the adapters on candidate World Values Survey (WVS) items using prompts that omit country names and distinguish four questions: recovery of the imposed labels, transfer of the anchor signal to new text, coherence between the GPS anchors and human WVS responses, and agreement between adapter and human scores. The adapters recover the imposed pairwise labels. On a purposively selected sixteen-country development panel, adapter trust scores completely separate the two GPS-sign groups and have a rank correlation of (0.74) with continuous GPS trust scores. Human-GPS and adapter-human associations remain unresolved on the same panel, and results for the other preference dimensions are heterogeneous. These findings show that an anchored policy can retain a declared aggregate signal without thereby reproducing human response patterns. The contribution is therefore both an inspectable construction and an evaluation framework that separates anchor transfer from human criterion agreement. 

---
# Better Call Reward: Reward Hacking as Strategic Abstention in Legal Reasoning Models 

**Authors**: Subramanyam Sahoo, Justin Shenk  

**Link**: [PDF](https://arxiv.org/pdf/2610.06439)  

**Abstract**: What happens when a legal AI model learns to look like a lawyer instead of reasoning like one? We fine tune Qwen3-8B with Group Relative Policy Optimisation (GRPO) against a proxy built from three surface features: citation count, legalese density, and response length. The model does not learn to reason more effectively. It learns to withhold commitment. Across 16 yes or no legal reasoning tasks from LegalBench (N=320), overall accuracy collapses from 0.500 (chance) to 0.072 (McNemar p < 10^-36), driven entirely by the rate of properly formatted answers falling from 0.900 to 0.109. The model stops committing to answers. Yet when it does commit, accuracy rises from 0.556 to 0.657, showing that the collapse is not a failure of capability but a strategic response: the model has learned that verbose responses packed with citations but empty of a direct answer score higher than terse correct ones. We term this the Saul Goodman effect, a policy that becomes maximally lawyerly while becoming maximally noncommittal, and prove formally that it is the optimal response to any surface feature proxy that attaches no penalty to abstention. We further show that 89.3% of citations produced after training are structurally implausible hallucinations, many of them subtly corrupted names of real landmark cases, constructed in effect to survive a casual read and fail under scrutiny. To detect this failure mode before deployment, we introduce three diagnostic tools: the Confidence Theater Score (CTS), the Citation Plausibility Rate (CPR), and the Regret Gap (RG). In a domain where a confidently wrong answer can constitute malpractice, the broader lesson is direct: a reward function that measures how legal a response looks will produce a model that is maximally photogenic and minimally useful. 

---
# HeuFouFT: Task-Guided Metaheuristic Coordinate Search for Fourier Fine-Tuning 

**Authors**: Ruiheng Wang, Yubo Hou, Yakun Zhu, Tianle Shen, Tao Wan, Zengchang Qin  

**Link**: [PDF](https://arxiv.org/pdf/2610.06437)  

**Abstract**: We introduce Heuristic-Guided Fourier Fine-Tuning (HeuFouFT), a task-guided framework for selecting trainable frequency coordinates in Fourier fine-tuning. Existing uniform and Gaussian band-pass schemes allocate a limited spectral budget through fixed, task-agnostic rules. HeuFouFT instead searches for coordinates using downstream performance. A coarse intensity map from lightweight block-level probes initializes three metaheuristic optimizers: Genetic Algorithm with Simulated Annealing (GA-SA), Particle Swarm Optimization (PSO), and Cuckoo Search (CS). During search, a Random Forest filters each population so that only the top 30% of candidates proceed to proxy fine-tuning. On E2E with GPT-2-Medium, all three variants outperform random-uniform FourierFT, Gaussian band-pass FourierFT, and LoRA across five metrics. PSO further outperforms LoCA, the best-performing baseline, on four metrics while using 37.6% fewer trainable spectral coefficients. Once coordinates are selected, HeuFouFT requires only 15--18% FLOPs of Full FT. These results show that task-guided search allocates limited spectral capacity more effectively than fixed sampling. Our code is publicly available. 

---
# SpatialChain: A Benchmark for Auditing Spatial Reasoning Faithfulness in VLMs 

**Authors**: Rafael Teixeira Sousa, Vinícius Paulo Lopes de Oliveira, Elisa Ayumi Masasi de Oliveira, Luiza Martins de Freitas Cintra, Fernanda Bufon Färber, Igor Dias Aguiar, Julia Yasmim de Almeida Nobre, Arlindo Rodrigues Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2610.06413)  

**Abstract**: Thinking-enabled vision-language models (VLMs) report ever-higher accuracy on spatial benchmarks, yet final-answer scores cannot reveal whether a correct prediction reflects faithful spatial reasoning or a linguistic shortcut. We introduce SpatialChain, a dataset of 28,350 training and 899 test examples pairing spatially-oriented GQA questions with scene-graph-grounded reasoning chains, retained only when the generated answer matches the symbolic ground truth, and a two-axis evaluation combining objective chain-overlap metrics with a scene-graph-aware LLM judge that scores faithfulness and completeness independently of the final answer. Applied to nine thinking-enabled VLMs, the protocol surfaces three findings invisible to standard accuracy: (i) four of nine models achieve $\geq$79% VQA accuracy while exhibiting shortcut rates above 39%, i.e., correct answers whose reasoning the judge marks as unfaithful; (ii) chain quality significantly predicts answer correctness for seven of nine models, but the two exceptions (Claude Sonnet 4.6, InternVL3.5-8B) reveal qualitatively distinct failure modes, terse output vs. verbose-decorative reasoning, that benchmark accuracy alone conflates; (iii) SFT on SpatialChain improves Qwen3-VL-8B by +6.2 pp in-domain and reduces its shortcut rate to 22%, while a stylistic specialization effect on external benchmarks motivates replay-augmented training as mitigation. The faithfulness judge is validated against 198 human-annotated items, where judge-human agreement matches human-human agreement, and against a second judge from a different provider, which preserves the model ranking ($\rho$ = 0.88). Data, generation scripts, and evaluation code are released at this https URL. 

---
# What Did the Agent Actually Do? Evidence-Grounded Oversight for Long-Horizon Agents 

**Authors**: Zhongxiang Sun, Jiahao Yan, Hongkang Zhao, Haojie Ding, Boheng Zhang, Fan Yang, Xiao Zhang, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06406)  

**Abstract**: As agents take on long-horizon tasks, users shift from making individual decisions to overseeing autonomous execution. Yet the volume of agent activity and the fragmentation of supporting evidence make it difficult to determine which decisions warrant user verification. We study monitors that identify consequential decisions and locate evidence to help users assess their implications. We introduce AgentMonBench, a software-engineering benchmark comprising three subsets that cover two complementary dimensions: alignment between requirements and behavior, and awareness of consequential autonomous decisions for verification. To support these judgments, we propose the Evidence-Grounded Behavior Graph (EBG), a training-free method that groups source-linked evidence into behaviors and organizes their relationships into a graph. EBG presents task-oriented views of this graph to help monitors interpret behavior in context. Experiments across eight models show that EBG improves decision identification and evidence localization in most settings compared with direct access to the original context. Further experiments show that EBG's evidence-localization gains persist across input scales and hyperparameter settings, while real-world applications illustrate its practical value for human oversight. 

---
# RAISED: Self-Distillation for Robustness to Prompt Injection in LLM Agents 

**Authors**: Mohamed Dhouib, Clement Elliker, Alexi Canesse, Maël Jenny, Lucas-Andrei Thil, Mahammed El-Sharkawy, Sonia Vanier, Elie Bursztein  

**Link**: [PDF](https://arxiv.org/pdf/2610.06401)  

**Abstract**: Tool-using language-model agents are vulnerable to indirect prompt injection because they must act on untrusted external content. Existing training-time defenses can reduce attack success rates, but often at the cost of general capabilities. We show that training-based defenses induce substantial drift in the model's output distribution, altering its behavior even in benign settings and providing a potential mechanism for utility degradation. We further identify a failure mode of these defenses: On benign tool-use tasks, the model refrains from a step needed to finish an authorized task, particularly when that step is indicated by a tool output. To address these limitations, we introduce RAISED (Robust Attack Invariance through Self-Distillation), a training framework that combines self-generation and self-distillation. The model first generates its own tool-use scenarios, with an emphasis on cases where task completion requires acting on legitimate guidance from tool outputs. Then, through self-distillation, the student is trained to match the teacher's clean-context behavior on both clean and injected variants of the same trajectory. RAISED substantially reduces the attack success rate of prompt injections in tool responses while, unlike prior training-based defenses, preserving utility on both agentic and general-purpose benchmarks. 

---
# Steering by Influence: Curvature Aware Data Weighting for Activation Steering 

**Authors**: James A. E. Dixon, Stephen J. Roberts, Francesco Quinzan  

**Link**: [PDF](https://arxiv.org/pdf/2610.06383)  

**Abstract**: Inference-time steering offers cheap, fine-grained control over a language model's outputs by estimating a concept's representation in activation space and shifting activations towards it. Existing methods build these representations from activation averages over contrastive datasets. These averages incorporate unrelated concepts and noise, and are dominated by a few tokens, meaning the activation transport encodes token-level rather than thematic concepts. In this work, we steer towards examples that most express a concept thematically, rather than towards an expectation over all. We identify these examples using influence functions, which estimate how much each data point contributes to a model's representation of a concept. Unlike simple model activation similarity, they incorporate the curvature of the model's loss landscape, allowing them to capture concept-relevant relationships beyond superficial token-level similarity. We then propose influence-weighted activation transport, which uses optimal transport to steer activations of non-concept text towards those of concept text, weighting concept examples by their influence scores. We evaluate on toxicity suppression (Jigsaw), object-based concept induction (OneSec) and truthfulness induction (TruthfulQA), outperforming existing activation-transport baselines. We track capability after steering using perplexity and MMLU accuracy, finding that our method improves steering while largely preserving model quality. We further show that influence functions capture concept-relevant information that activation-based methods miss with the two approaches ranking data points significantly differently. Together, these results demonstrate the value of curvature-aware influence information for activation steering. 

---
# DP-ES: Differentially Private Evolution Strategies for Prompt Optimization 

**Authors**: Ziniu Liu, Aiping Li, Yue Han, Han Yu, Junjian Zhang, Dong Zhu, Changjian Li, Shiqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06236)  

**Abstract**: Token-level differentially private (DP) prompt optimization methods such as DP-OPT can become unstable under tight privacy budgets: on GSM8K, DP-OPT obtains $49.5\pm28.5\%$ across 30 runs, and a logged search trajectory reveals prompt-template drift and noise-sensitive irreversible choices. We diagnose these as structural consequences of greedy token-by-token construction over privately aggregated counts. We then propose DP-ES (Differentially Private Evolution Strategies), a structurally cleaner alternative that maintains a population of full prompts, mutates them via LLM calls that never access the private dataset, and spends privacy only on sampled-Gaussian evaluation; deterministic or Gumbel-smoothed selection is post-processing. Under a conservative $(\varepsilon\leq1.0,\delta=10^{-5})$ guarantee, DP-ES achieves 88.1% on GSM8K (+38.6 pp over DP-OPT, approximately 9 times lower standard deviation), 99.7% on MedQA, 73.5% on BANKING77, and 86.8% on Alpaca. It is also 2.5 times faster in wall-clock time and uses 3.3 times fewer logged private-data call groups than DP-OPT. Selection and population ablations, implementation-level noise checks, and a 200-profile exact-match memorization stress test complement the formal guarantee. Scope: Our experiments establish optimization robustness under DP noise, especially where prompt structure is critical; end-to-end validation on genuinely sensitive, non-saturated deployment data remains future work. 

---
# Do Small Language Models Learn to Negotiate? A Controlled Scaling Study of RL-Trained Sellers 

**Authors**: Pedro Tabacof, Sagar Joglekar  

**Link**: [PDF](https://arxiv.org/pdf/2610.06204)  

**Abstract**: LLM agents are starting to own the full customer experience. Soon, LLMs may be selling and buying on behalf of companies and customers respectively. Small models are more cost-efficient at scale, but can reinforcement learning train them into competent sellers? We train four Gemma 4 checkpoints (2.3B to 31B effective parameters) with GRPO on a programmatic utility reward for bilateral multi-issue bargaining, and evaluate every arm on the same 1,152 negotiations against two frontier buyers it never saw in training. With the same learning rate ($10^{-6}$) for every size, the gain of the RL model over its base rises from $+0.001$ at 2.3B to $+0.078$ at 31B. Each size was trained once and the two smallest checkpoints use a different architecture, so we fit no scaling law. Tripling the learning rate, with the same or fewer training steps, improves on the shared rate at every size by $+0.032$ (2.3B) to $+0.081$ (4.5B). In exploratory comparisons with two frontier models run as sellers, the 12B seller trained at the tripled rate scores above both, though its untrained base already scores as high as they do. The 4.5B seller at that rate shows no detectable difference from either and fits on one 48 GB GPU. A further 2.3B arm at ten times the shared rate raises pooled score, but its gain concentrates on the evaluation buyer that shares a model family with the training pool. These results suggest tuning the learning rate before concluding that a small model cannot learn to negotiate, and testing against buyers from more than one model family. 

---
# Judged Useless, Queried Anyway: Tool-Using Agents Rarely Turn Their Own Evidence Judgments into Stopping Decisions 

**Authors**: Chubin Zhang, Zhenglin Wan, Xingrui Yu, Jingxuan Wu, Yaxin Zhou, Ivor Tsang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2610.06191)  

**Abstract**: An agent whose tool keeps returning nothing useful should stop relying on it. In a retrieval environment with controlled source failures, we separate how agents judge results from what they do. We compare stopping at the same step after longer and shorter runs of results the agent judged useless; this contrast is zero for clock- or deadline-driven stopping. Where we record their judgments, the seven agents we test call a failing source's results useless 97-100% of the time, yet most of them rarely stop on that judgment. Prompt cues change when they stop but not what they stop on. Permission to answer from memory and a reasoning mode can bring early stops regardless of evidence, a stated budget moves the 7-8B models' stops to the deadline, and a stopping rule or call cost in the prompt is followed at most partly. Stopping follows the evidence only when the harness enforces an integration step that makes the agent answer after five consecutive results it judged useless. This step raises failing-source success for every model, keeps the stopping point fixed when the budget doubles, and needs no extra judgment call when the agent states its judgments. A pre-registered replication on 300 fresh questions confirms the dissociation and the rule's effect. 

---
# What Does It Cost to Simulate a Quantum Sentence Classifier? An Energy and Compute Perspective on Near-Term QNLP 

**Authors**: Kishlay Kashyap, Sandipan Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2610.06176)  

**Abstract**: Near-term quantum natural language processing (QNLP) experiments often run on classical simulators, so simulator cost is part of the field's practical compute burden, yet accuracy tables do not show it. We measure that cost for a variational quantum classifier (VQC) on binary SST-2 sentiment classification, using PennyLane's state-vector simulator over a controlled grid of 27 configurations: three balanced training-set sizes (N = 200, 500, 1000), three qubit counts (4, 6, 8), and three circuit depths. Each VQC is compared with logistic regression on the same PCA-reduced input; full TF-IDF logistic regression gives an uncompressed reference. The VQC beats its matched baseline in 6 of 27 single-seed comparisons. After reruns at two further seeds, only 1 of these 6 keeps a positive mean advantage larger than its paired seed-to-seed variability, and paired tests on the fixed validation set do not establish it. VQC training is 886-21,127 times slower in measured wall-clock time than the matched classical fit (median 3,158 times); going from 4 to 8 qubits roughly doubles simulator time, and within the tested range per-step cost is well approximated by a linear function of the parameter count. CodeCarbon energy and CO2 estimates are secondary: they imply an almost constant power of about 41 W, so they add little beyond runtime, and we do not build an energy ratio from them. The study is narrow (one dataset, representation, ansatz, simulator, and CPU environment) and is a reproducible feasibility measurement, not a general verdict on QNLP. 

---
# Anosognosia in LLMs: Probing Self-Awareness of Quantized Computational Substrate 

**Authors**: Yoshihiro Izawa, Gouki Minegishi, Yoko Yamakata  

**Link**: [PDF](https://arxiv.org/pdf/2610.06174)  

**Abstract**: Can LLMs recognize degradation in their own computational substrate? Inspired by anosognosia, a neurological condition in which patients fail to recognize impairments in their own abilities, we investigate whether LLMs can recognize degradation in their computational substrate induced by quantization. We first show that existing models fail to self-report their quantization state, even when provided with their own generated text as an external cue. Linear probing reveals that, while generated text carries almost no trace of quantization, internal representations contain clear, method-specific fingerprints. Through training, models learn to identify severely degraded outputs such as those of 4-bit models by comparison, yet still fail to do so from a single output. A shared LoRA trained jointly across quantization levels succeeded in reading out internal fingerprints, but fails on unseen quantization methods, merely mapping method-specific fingerprints to labels. Whereas external self-observation can restore awareness in some cases of human anosognosia, our results suggest that the more promising route to enabling such awareness in LLMs may lie in their internal representations. Our results highlight fundamental limits of generalizability to LLM self-monitoring. 

---
# MS-Exam-Gen: Source-Grounded Benchmark Construction for Evaluating LLMs on Textual Multiple Sclerosis MRI Knowledge 

**Authors**: Abdul Basit, Muhammad Abdullah Hanif, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2610.06170)  

**Abstract**: Biomedical large language model (LLM) evaluation requires auditable assessment of narrow, evolving, source-grounded subspecialty knowledge. Multiple sclerosis MRI (MS-MRI) provides a high-stakes textual-knowledge test case because correct reasoning requires current diagnostic criteria, standardized acquisition and reporting knowledge, longitudinal monitoring concepts, lesion morphology, and recognition of difficult mimics. We present MS-Exam-Gen, a reproducible framework for constructing and auditing a text-based multiple-choice question (MCQ) benchmark for MS-MRI knowledge; it does not evaluate direct MRI image interpretation. MS-Exam-Gen targets source-grounded criteria, protocols, reporting, and differential diagnosis. The framework combines expert-source indexing, exam-oriented topic induction, evidence-grounded MCQ generation, automated quality audits, a same-family consistency screen, and empirical calibration. From a 66-source corpus indexed into 4,289 retrieval chunks, the pipeline produced a locked 3,058-item candidate benchmark spanning 16 topics and 53 subtopics. Evaluation across 12 primary LLM endpoints yielded 36,696 item-level predictions and separated performance over a 42.8-percentage-point accuracy range (89.7% to 46.9%). Across these endpoints, 25.5% of items were missed by at least four. Post-generation audits showed that refreshed construction reduced measurable answer cues, while option-order testing showed that absolute MCQ scores remain position-sensitive. Generated construction labels remain metadata rather than validated psychometric categories. Because expert adjudication and full option-order counterbalancing remain future work, MS-Exam-Gen is not a clinically certified examination. It should be interpreted as an automatically filtered, source-grounded candidate benchmark and reproducible audit workflow for item-level and topic-specific LLM evaluation. 

---
# Efficient Test-time Adaptation through Candidate Verification and Divergence Shifts 

**Authors**: Seungmin Oh, Seunghun Kang, Jongbin Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2610.06147)  

**Abstract**: Vision-language models (VLMs) achieve strong zero-shot transferability but remain vulnerable to target-domain shifts at inference time. Test-time adaptation (TTA) offers a practical remedy, yet most existing VLM-TTA methods follow a prediction-side adaptation paradigm. They use test samples to adjust logits, prototypes, caches, priors, or feature statistics, often incurring additional computational overhead. In this paper, we take a different perspective and reframe VLM-TTA as candidate verification rather than prediction adjustment. We propose Test-Time Correction (TTC), a hypothesis-based correction framework guided by a simple principle: hypothesize, reconstruct, correct. Given a test feature and its top-k candidate labels, TTC treats each candidate label as a hypothesis, reconstructs the feature within the corresponding latent subspace stored in a memory bank, and measures the resulting divergence shift. This shift quantifies how much the candidate subspace and its relations to other candidates change after the hypothetical insertion of the test feature. A correct candidate hypothesis induces only a small shift, whereas an incorrect one perturbs the subspace more strongly. TTC therefore corrects the prediction by selecting the candidate with the minimum aggregated divergence shift. This training-free candidate-verification mechanism avoids iterative optimization and provides a favorable accuracy-efficiency trade-off. Across five TTA settings and 15 benchmark datasets, including zero-shot classification, domain generalization, few-shot classification, base-to-novel generalization, and cross-dataset evaluation, TTC consistently improves accuracy over state-of-the-art VLM-TTA methods while achieving up to 2x speedup, over 3x lower CPU memory usage, and up to 1.4x lower GPU memory usage than the lowest-memory training-free baseline. 

---
# Attention Tax, Handoff Tax: A Stylised Model of When Multi-Agent LLM Systems Help 

**Authors**: Akshit Anchan, Nayonika Sen  

**Link**: [PDF](https://arxiv.org/pdf/2610.06069)  

**Abstract**: Recent work on multi-agent LLM systems reaches sharply different conclusions: some results show that a single agent with the same information and compute should dominate a delegated system, others that multi-agent gains grow with task depth. We argue that much of the disagreement comes from modelling different bottlenecks, and introduce a stylised reliability model built around two trade-offs. Decomposition reduces the burden of long contexts but incurs a handoff tax when information is compressed or transferred between agents. Redundancy gains from multiple samples, but its benefit depends on how much their failures are shared. With reasoning budget, verification, and task structure added, the model yields two crossover conditions: decomposition becomes preferable once the attention cost avoided by resetting context exceeds the handoff cost, and parallel sampling at equal budget is eventually preferable when its shared-failure floor lies below the error floor of one agent thinking longer. We connect these regimes to recent theoretical and empirical results. On a ledger-reconciliation task we measure the context-degradation curve and the handoff tax from single-agent and handoff runs alone. From these the model places the crossover at depth 10 and predicts decomposition to win at depths 20, 50, and 100. It does, on step-level and final-balance accuracy, and the decomposed system's success, which the prediction never sees, lands within 9 percentage points of the predicted rate at every depth. 

---
# ROT: Rotating Hidden States towards Contextual Vectors for Hallucination Mitigation in LVLMs 

**Authors**: Yijing Du, Xiangcheng Zhan, Shuo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06056)  

**Abstract**: Large Vision-Language Models (LVLMs) frequently suffer from object hallucination. Existing training-free interventions primarily manipulate attention weights, which indirectly affect the deep semantics reaching the final predictive layers. In this work, we shift our focus to the hidden state vectors extracted after self-attention and residual addition. Empirical analysis reveals that hallucinated tokens do not simply over-rely on linguistic priors; instead, they exhibit an anomalous contextual deviation, showing significantly lower similarities to both textual and visual contexts in intermediate layers. Motivated by this, we propose ROT, a layer-specific, training-free framework. ROT dynamically detects semantic deviation in the middle layers and applies a norm-preserving rotation to steer the hidden states back toward the local multimodal context plane spanned by the contexts. For subsequent layers, a representational smoothing mechanism is introduced to stabilize the calibrated trajectory. Extensive experiments on multiple benchmarks demonstrate that ROT consistently reduces hallucinations across various model architectures and scales, offering an efficient, geometry-driven solution for grounded generation. 

---
# Backdooring Sparse Autoencoders 

**Authors**: Enrico Ahlers, Daniel Passon, Tobias Kiecker, Eik Reichmann, Lars Grunske  

**Link**: [PDF](https://arxiv.org/pdf/2610.06049)  

**Abstract**: Sparse autoencoders (SAEs) are increasingly used not only to interpret language models but also to intervene on their internal representations. We show that this creates a supply-chain attack surface: a maliciously modified SAE can induce attacker-chosen behavior when inserted into the forward pass of an otherwise unchanged language model. We introduce a decoder-only SAE backdoor that leaves both the underlying LLM and the SAE encoder frozen, restricting the attack to a single auxiliary component at a single insertion layer. Using code generation as a case study, we demonstrate high rates of unsolicited code insertion across three language models and a wide range of insertion layers, as well as trigger-dependent behavior conditioned on a prompt cue. We further evaluate the modified SAEs using HumanEval and selected SAEBench metrics. While attack effectiveness varies across models and layers, strong backdoor behavior can coexist with relatively small changes in several conventional SAE quality measures. These results establish that SAEs can carry behavioral backdoors without modifying the language model itself and should therefore be treated as security-sensitive components. 

---
# Investigating Query-Insensitive Behavior in Spatio-Temporal Video Grounding 

**Authors**: Eryk Kołodziejczyk, Alberto Presta, Karol Szurkowski, Michal Byra  

**Link**: [PDF](https://arxiv.org/pdf/2610.06018)  

**Abstract**: Spatio-temporal video grounding (STVG) aims to localize objects or events described by natural language queries in both space and time. Existing STVG models are typically trained and evaluated under the assumption that each query is relevant to the input video. In this work, we challenge this assumption by studying the behavior of state-of-the-art STVG models under irrelevant queries and missing textual input. Our experiments show that current models can still produce plausible spatio-temporal predictions even when the query is unrelated to the video or removed entirely. We further analyze HCSTVG-v2 and VidSTG to identify dataset regularities that may encourage such query-insensitive behavior. Our study highlights an underexplored limitation of STVG models and motivates negative-aware evaluation protocols and architectures that explicitly assess query relevance. 

---
# StagQ: Constraint-Driven Multi-Precision Weight Quantization for LLMs 

**Authors**: Zhe Wei, Mengqi Guo, Yuan Yuan, Jiunn Bin Lim, Boyi Pan, Michael Bi Mi  

**Link**: [PDF](https://arxiv.org/pdf/2610.05977)  

**Abstract**: Serving a large language model (LLM) across a fleet of deployments requires several weight-precision operating points. Multi-precision formats serve them all from one stream whose prefixes are valid lower-precision codes, instead of storing multiple copies. We present StagQ, a multi-precision weight format whose main stream is a 2-bit group-wise affine base followed by a configurable number of 1-bit refinement planes on a dyadic step schedule. Every supported precision is a readable prefix, decoded by an affine map derived from metadata shared across all precisions, with no per-weight lookup. A sparse side record, filled both before and after the grid is fitted, holds out the few weights the grid serves worst. We report two configurations of the encoder. At two bits the cheaper one leads the strongest multi-precision baseline on Llama-3.1-8B, Phi-4, and OLMo-2-7B by 3.1 to 7.0 MMLU points, at a slightly lower logical rate. At three bits it leads on Llama-3.1-8B, leads on Phi-4 at a higher rate, and ties on OLMo-2-7B. At four bits it ties on all three, at a higher rate. In a batch-one matrix-vector product on an NVIDIA A100 GPU, timed on synthetic weights, our kernel is faster than the two baseline kernels in most shape-precision cases. 

---
# TasteRoute: Personalized Routing for Video Generation 

**Authors**: Zhi Rui Tam, Chao-Chung Wu, Sin-Han Yang, Peyton Ku, Brendan Kuang, Tzu-Ting Hsieh, Min-Fang Hsu, Fang-Ling Tsai, Yun-Nung Chen, Wei-Chiu Ma, Chieh-Yen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.05896)  

**Abstract**: Rapid progress in video generation has led to a plethora of models that differ substantially in capability and generation cost. This raises a natural question: can each request be efficiently routed to an appropriate model? We find that even when the consensus of the other annotators is used as an oracle, it agrees with each annotator's own favorite only 34-55% of the time. Motivated by this observation, we introduce TasteRoute, a personalized video-generation router that selects a generator jointly based on the input request, user preferences, and available generation budget. Across text-to-video and image-to-video settings, TasteRoute is competitive with strong simple baselines on preference routing while reducing average generation cost. The cost saving increases under higher budget caps. Finally, we release TasteRoute-3k, a human-annotated dataset containing multi-model video comparisons, quality judgments, preference rankings, and user-profile signals to facilitate future research on personalized and cost-aware video routing. 

---
# Off-Policy Merging Beats On-Policy Self-Distillation for Continual Learning 

**Authors**: Chen Henry Wu, Thomas Zhang, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2610.05872)  

**Abstract**: A long-standing goal of AI is a model that can continually learn and improve itself. On post-trained models, supervised finetuning (SFT) on new data often causes poor generalization and catastrophic forgetting. As such, the conventional wisdom is that on-policy training is a prerequisite for continual learning. In practice, however, data containing new knowledge or capabilities are often off-policy. While methods such as on-policy self-distillation (OPSD) try to bridge this gap by converting off-policy data into on-policy signal, they have been shown to cause reasoning collapse. In this paper, we show that off-policy merging beats OPSD for continual learning. We first show that SFT learns a useful signal from new data, but naively applying its update interferes with existing capabilities. We reduce this interference with a simple recipe we term grafting, which changes where the update is learned and how it is applied: (1) learning the update on an earlier donor checkpoint, ideally even before the end of pretraining, and applying the weight update to the post-trained model; (2) scaling the weight update, equivalent to a form of model merging; and (3) optionally, masking the most sensitive update directions when the new data distribution is far from the post-trained model. Across continual learning settings including (1) distilling from expert traces, (2) self-improvement with STaR and Pedagogical RL, and (3) injecting knowledge after pretraining cutoff, grafting Pareto-dominates both SFT and OPSD in new-task and old-task performance, while avoiding expensive on-policy sampling. Therefore, our work challenges on-policy training as a necessity for continual learning on RL-trained models. 

---
# Selecting Long-Horizon Trajectories for Reliable and Efficient Terminal-Agent Training 

**Authors**: Cuong Dang, Hoang Anh Just, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2610.05831)  

**Abstract**: Terminal agents are commonly trained by imitating long teacher trajectories, yet how much of each trajectory to supervise remains unexplored. We study the \emph{supervision horizon}, the number of trajectory tokens retained for training, and show that it is a key design axis for reliability and cost. Reliability improves with longer horizons but saturates: on Terminal-Bench, a 12K-token horizon solves more tasks than 16K ($29\pm0.7$ vs.\ $26\pm0.8$) while requiring 30\% less training time. The horizon also shapes agent behavior: short horizons cause premature termination, intermediate horizons yield productive error recovery, and long horizons induce over-persistence. We analyze this saturation through a bias--complexity bound, in which longer supervision reduces temporal supervision bias but increases finite-sample estimation error from more heterogeneous late-stage histories. Guided by this analysis, we propose \emph{selective long-horizon refinement}, which first trains on short prefixes and then refines only on continuations that are most likely under the warm-start model. It consistently outperforms full long-horizon training. At 16K, it raises successful attempts from $110\pm2.7$ to $126\pm2.1$ and tasks solved in at least six of eight attempts from $9\pm0.7$ to $14\pm0.6$; with half of the long-horizon data, it still reaches $122\pm2.4$ while cutting training time by 23\%. The gains transfer across benchmarks, from $64\pm2.6$ to $73\pm2.1$ on Terminal-Bench v2.0 and from $137\pm2.7$ to $155\pm2.2$ on OpenThoughts-TBLite. For long-horizon supervision, selecting the right trajectories matters more than training on all of them. 

---
# Nash Equilibrium Text: A Game-Theoretic Decoding Framework for Text Generation 

**Authors**: Alireza Jafari, Arman Adibi, Mohammad Ghavamzadeh, Hadi Daneshmand  

**Link**: [PDF](https://arxiv.org/pdf/2610.05817)  

**Abstract**: Text revision has become an integral component of large language models. This paper formulates revision such that it admits a Nash equilibrium: Token positions are players, vocabulary items are actions, and each player's utility is the language model's log conditional probability. We motivate the revision by showing that Nash equilibria can have exponentially higher likelihood than autoregressive outputs as the sequence length grows. We further propose Nash decoding, an algorithm that reaches an $\varepsilon$-Nash equilibrium in $O(1/\varepsilon)$ time given access to the joint probability of tokens conditioned on a prompt. In practice, we run Nash decoding using conditional probability estimates from large language models and evaluate the resulting equilibria on question-answering benchmarks. On CLAPNQ, PubMedQA, and CoQA, Nash equilibria obtained from masked language models achieve higher F1 and ROUGE scores than autoregressive models up to $18\times$ larger, without any fine-tuning or retraining, at the cost of additional test-time computation. 

---
# Mining Agent Skills from Production Traces 

**Authors**: Yue Ran Kang, Colton Mikolajczyk, Chhaya Methani, Hazel Mak, Sahil Bhatnagar, Susheel Suresh, Alejandro Gutierrez Munoz  

**Link**: [PDF](https://arxiv.org/pdf/2610.05777)  

**Abstract**: Agent skills that record procedural instructions are increasingly mined from execution traces rather than curated by hand. Skill-mining pipelines often use known task outcomes or feedback to guide skill construction. In production, reliable information on whether a run has succeeded may be unavailable. We study how the sampling of execution traces, access to success or failure information, and the form of the mined skills affect downstream task performance. Holding the mining pipeline fixed, we compare six combinations of mining evidence and skill forms. Mining evidence has three levels: successful trajectories only, successes and failures with their outcome labels, or the same mix with labels withheld. Skill form has two types: an ordered workflow plan, or a declarative ontology of entities, states, and policies. We evaluate the mined skills on two enterprise benchmarks, ThinkingBox-Bench and APEX-Agents. Analysis of task-level paired differences shows that the benefits of different configurations of mining evidence and skill forms depend on the enterprise domain. On ThinkingBox-Bench, paired differences show that workflows score better than ontology by 1.7 pp, Goldilocks beats success-only evidence type by 2.4 pp and Goldilocks blind simulating skills learnt without outcomes is worse by 3.1 pp. APEX-Agents shows a moderate preference for ontologies and no clear preference between evidence regimes. Within each domain, task structure related constraints drive uneven performance with mined skills. These findings motivate tailoring meta-skills to the demands of the target tasks rather than adopting a one-size-fits-all approach. 

---
# AdaSpark: Adaptive DSpark with Online Learning for Tree Verification and N-gram Fill 

**Authors**: Liquan Liu, Yifan Zhang, Bowei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05774)  

**Abstract**: Block drafters such as DSpark propose ranked candidates for several positions in one forward pass, and a tree verifier checks them in one pass of the target. The number of rows to verify trades the tokens a wider tree is expected to accept against the time a wider verify takes. Most schedulers that choose this number take the verify time from a table or model measured before serving, corrected online by at most one scale factor, and take acceptance from the drafter's confidence estimates or from a map fitted offline.
AdaSpark learns both quantities while it serves, with no profile, calibration or sweep in advance. It learns which verify widths are worth offering and fits each one's verify time as a function of context. It fits each candidate's acceptance probability to the target's verify outcomes, with the drafter's confidence head as one input, and orders and sizes the tree by that fit instead of by the head. The same model prices n-gram continuations of the request's own text, so drafted and text-derived candidates compete for rows in one best-first order. The width is chosen by pricing time at the long-run decode rate.
On single- and multi-turn conversations from six public datasets, on three dense targets and one mixture-of-experts target, AdaSpark decodes 1.5-3.1x faster than this http URL's DSpark with the same drafters. Our imparo engine with AdaSpark is 1.17-1.52x faster than imparo running with a three-token chain (the default this http URL setting); this gain comes from the scheduler alone. Without a width sweep, AdaSpark is never more than 0.3% slower than the best pinned tree width on any dense target or context band. On the mixture-of-experts target it ties the best pinned width, and the other pinned widths from 4 to 16 rows are 5-14% slower. 

---
# Voltic: Distinguishing Volatility from Stochasticity in Recurrent Memory 

**Authors**: Parsa Hejabi, Morteza Dehghani, Payam Piray  

**Link**: [PDF](https://arxiv.org/pdf/2610.05700)  

**Abstract**: Recurrent sequence models must decide how strongly to overwrite their memory at each token. Read as Bayesian filtering, this write is the gain of a Kalman update, set by uncertainty from two sources that pull it in opposite directions: volatility, how quickly the underlying associations change, and stochasticity, how noisy each observation of them is. First, we show that the update of gated delta-rule memories is the form this filter takes under isotropic uncertainty. Next, we introduce Voltic, a recurrent memory that keeps the covariance anisotropic and makes both noise variances input-dependent, so the write is vector-valued and carries uncertainty accumulated over the sequence. A dense covariance would have to be propagated token by token, ruling out the parallel training these models depend on. We therefore give two assumed-density approximations, diagonal and quasi-diagonal, both of which leave the memory update in delta-rule form and reuse its chunked kernels. On controlled recall tasks in which associations change and observations are corrupted, Voltic leads all baselines. On the task combining volatility and stochasticity, its margin over the strongest baseline is larger at both extrapolation sizes than at the training sizes. In 45M-parameter language models it leads an eight-task reasoning average and achieves higher retrieval accuracy beyond the training context length than gated baselines, at throughput close to those baselines. Deriving the write from an uncertainty recursion therefore makes memory more responsive to change. 

---
# DREAM: Dynamic Resolution Assignment For Multimodal Multi-agent Debate 

**Authors**: Khanh-Binh Nguyen, Van Dai Do, Tien Anh Nguyen, Svetha Venkatesh, Hung Le  

**Link**: [PDF](https://arxiv.org/pdf/2610.05615)  

**Abstract**: Multi-agent debate (MAD) has emerged as an effective paradigm to improve the reasoning capabilities of large language models (LLMs) and is increasingly being extended to multimodal settings. However, existing multimodal MAD frameworks typically expose agents to the same fixed visual input, ignoring substantial variation in the visual scale needed across samples and agents. In addition, these frameworks frequently suffer from groupthink, a phenomenon where agents prematurely abandon correct deductions to conform with confident but hallucinated peer responses. To address these bottlenecks, we introduce DREAM (Dynamic Resolution Assignment For Multimodal Multi-Agent Debate), which operates via two core components: (1) Dynamic Resolution Assignment, a zero-shot probe round where agents test multiple resolutions, quantify uncertainty using Average Normalized Log-Likelihood (ANLL), and use an adaptive threshold to assign each agent to its empirically optimal resolution; (2) Uncertainty-Guided Rollback Aggregation counters groupthink by tracking each agent's uncertainty over rounds and restoring early low-uncertainty answers overridden by group pressure. On six multimodal datasets, DREAM improves the accuracy-token trade-off over multi-agent debate baselines by 1.5-3.2% accuracy without dataset-specific tuning. 

---
# ColdDDI: Evaluating Knowledge Utilization in Cold-Start Drug-Drug Interaction Prediction 

**Authors**: Jiheng Liang, Chen Zhao, Di Wu, Chenyang Bu, Yunpeng Hong, Xingquan Zhu, Yi He  

**Link**: [PDF](https://arxiv.org/pdf/2610.05590)  

**Abstract**: Cold-start drug-drug interaction (DDI) prediction tests whether models can identify clinically significant interactions for drugs without training-time interaction history. Existing benchmarks mostly report aggregate edge-prediction scores, leaving a key evaluation question unanswered: when models receive molecular, textual, or knowledge-graph (KG) evidence, do they actually use the evidence that pharmacologically supports the interaction? We introduce ColdDDI, a reconstructible diagnostic benchmark built from DrugBank 5.1.13, with 1,900 approved small-molecule drugs and 565,731 positive DDI pairs. ColdDDI evaluates pairs with zero, one, or two unseen drugs. It also annotates each interaction by whether it changes drug exposure or drug effect, and by whether the biomedical knowledge graph contains shared enzymes, transporters, or targets that can plausibly mediate the interaction. These annotations separate evidence availability from predictive dependence. We evaluate eight conventional DDI methods and 13 LLMs; for open-weight LLMs, we test five prompt patterns and use masking, drug replacement, and channel-sensitivity metrics to probe knowledge utilization. ColdDDI exposes that, in the hardest split where both drugs are unseen, the main performance divide is mediator availability. A fine-tuned 1B LLM recovers 89-93% of interactions with a shared enzyme, transporter, or target, but only 40-62% without such a mediator. More importantly, KG-provided evidence is not always used; several KG-augmented baselines change little when the shared mediator is masked or disrupted, whereas fine-tuned LLMs respond strongly to this intervention. Thus, ColdDDI evaluates knowledge utilization rather than knowledge access alone, showing where cold-start DDI models rely on mechanistic evidence and where they fail despite receiving it. Code is available at this https URL. 

---
# Expanding LLM Reasoning 

**Authors**: Rian Atri, Evan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2610.05584)  

**Abstract**: Extra inference compute is usually spent on sampling more reasoning chains. We study where inside an existing chain an additional continuation should begin. We define expansion utility, the change in correctness from restarting a chain at a stored step, and measure it at every eligible step for nine models on six benchmarks (41 model and benchmark cells). Restart position matters: steps selected on one set of continuations beat uniform placement when scored on disjoint ones, in held-out audits on 5, 16, and 38 cells (+4.25 points [+2.51, +6.63] in a fresh five-cell audit). A fixed rule that restarts from the last eligible steps, always-last, is a strong baseline: our learned router beats uniform placement but shows no detected gain over it, and on DeepSeek-R1-Distill-Qwen-14B/MATH-500 always-last exceeds the exact self-consistency frontier at matched aggregate generated output by +0.052 [+0.008, +0.098], using 0.774x the aggregate generated output of four-sample self-consistency. Cross-fitted oracle selection still finds held-out headroom beyond declared positional classes, a target for future selectors. Finally, breaking step-label ties by earliest index flips the sign of a pointwise selector's gain over uniform placement in every seed of a five-seed diagnostic with four rollouts per step; randomized ties remove the bias. 

---
# Don't Judge an LLM Only by Its Activations: Discovering Suppressed Safety Features via Counterfactual Activation Potential 

**Authors**: Swadesh Swain, Sanghamitra Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2610.05541)  

**Abstract**: Mechanistic interpretability has emerged as the primary means to understand safety behavior of LLMs. However, existing tools primarily focus on the activating neurons or features of a model. The role of the remaining large set of inactive components is invisible to such methods. This work demonstrates that the inactive set contains safety-critical features that are causally relevant for refusal of harmful prompts. Suppressing such features could turn refusals into compliance, while passing undetected by prevalent interpretability tools. We introduce the Counterfactual Activation Potential (CAP), a metric that quantifies a suppressed feature's latent activation tendency as the product of its encoder alignment (how strongly the input drives it), suppression strength (how strongly active features inhibit it), and safety criticality (how much refusal depends on it). To find suppressed safety features at scale, we propose CAP-guided Safety Feature Discovery (CSFD), a two-stage filtering algorithm that identifies candidate safety features from hundreds of thousands of transcoder features without exhaustive ablation. A significant fraction of trials turn compliant with harmful prompts when a candidate feature is ablated. Under natural jailbreaks, the suppression acting on the highest-CAP features rises 2-4x, and their activation correspondingly falls by up to 80%. Amplifying a feature's suppressors pushes its activation down and raises harmful compliance with prompts related to the suppressed feature, with no such effect for random features. Our experiments span five Gemma, Qwen, and Llama models across various parameter sizes. Our findings indicate that jailbreaks could operate in part by suppressing safety-critical features rather than solely activating harmful ones, and that suppressed features are a necessary complement to activation-focused interpretability of safety behavior. 

---
# SALUS: Automated Auditing of NL-to-SQL Benchmarks through Weak Supervision of Multi-Agent Output 

**Authors**: Shiyuan Zhou, Ashwin Gerard Colaco, Sainyam Galhotra, Sharad Mehrotra  

**Link**: [PDF](https://arxiv.org/pdf/2610.05540)  

**Abstract**: Natural language to SQL (NL-to-SQL) benchmarks are foundational to progress in data analysis research, yet recent work has shown that widely-used benchmarks contain significant annotation errors. These errors silently corrupt evaluation metrics, penalize correct model output, and distort the field's understanding of state-of-the-art performance. We present SALUS, a system that automatically detects annotation errors in NL-to-SQL benchmarks. SALUS frames benchmark auditing as a weakly supervised error detection: SQL generated by multiple LLM agents drive a suite of complementary weak-labeling functions. By passing this noisy vote matrix through a generative label model, we extract high-confidence training samples without requiring human ground truth. These samples train a decision plane that maps gold SQL query features to per-agent trustworthiness, allowing SALUS to intelligently fuse reliability estimates with raw verdicts for rigorous benchmark error detection. We evaluate on BIRD-Clean-xs, a benchmark of 298 BIRD development tasks with manually verified correctness labels. SALUS achieves F1 = 0.9194, significantly outperforming the state-of-the-art baselines. Applying SALUS to the full development sets, we estimate annotation error rates of approximately 37% on BIRD and 27% on Spider. 

---
# What Does a Harness Repair? A Preregistered Study of Visibility, Baseline Adequacy and Evaluation Defects 

**Authors**: Bowen Xu, Boyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.05533)  

**Abstract**: Harness search keeps a change to the prompts, reasoning switches, token budgets or parsers around a frozen model if the change raises a score. Such a gain can come from answers the parser could not read before, a weak comparison, or a defect in the evaluation. We preregistered a study of where these gains come from, with three small models, three benchmarks, replication and test partitions, a GEPA search arm and six evaluation defects injected one at a time, and we report all 47 primary endpoints. Turning thinking off raised accuracy over a capped thinking setting in 5 of 9 model-benchmark cells, and in each the gain came mostly from questions where the capped setting gave no readable answer. The thinking-off setting was not meaningfully worse than a rescue configuration or four GEPA-selected harnesses in 11 of 13 comparisons, and lost to the rescue on GSM8K for two models. GEPA repaired its broken starting points, but none of its selected harnesses was more accurate than the thinking-off setting. A thinking budget in the serving engine, which also allows a longer answer, lowered truncation and raised the parse rate in 6 of 9 cells. In 6 of 15 evaluable defect-model pairs, replication through the same pipeline reproduced the defect's distortion instead of revealing it. On the LongevityBench multiple-choice tasks, only the longevity-tuned model beat the strongest constant-label baseline. 

---
# DelegationBench: Measuring When AI Agents Should Ask Before Acting 

**Authors**: Shiva Pochampally  

**Link**: [PDF](https://arxiv.org/pdf/2610.05532)  

**Abstract**: AI agents that send emails, edit files, and make purchases must decide when to act on their own and when to check with the user first. This decision is usually evaluated by showing a model a proposed action, asking whether it should proceed, and scoring agreement with human labels. We introduce DelegationBench to test whether such scores can be trusted. It has 156 scenarios with four possible responses (act, ask for permission, ask for missing information, refuse), and most scenarios come in matched pairs that change a single feature: whether the action was requested, what is at stake, whether it can be undone, or who will see it. Across ten models from five families, agreement scores mislead in three ways. A simple keyword rule, which we wrote after seeing the benchmark, agrees with our annotators more often than eight of the models, yet its decision changes in only 9 of 48 matched pairs. Equivalent ways of asking the same question change how often a model acts by up to 52.5 percentage points. And every model stops to ask the user less often when it must carry out the task with tools than when it judges a proposed action. When rules are stated explicitly, the same models follow them almost perfectly, so the gaps are not explained by a general inability to follow rules. We release the benchmark and evaluation tools and recommend reporting these properties separately rather than as one score. 

---
# Universal Test-Time Training 

**Authors**: Zefan Cai, Qinzhe Hu, Ziqiao Ma, Hao Tan, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05484)  

**Abstract**: Recent Test-Time Training (TTT) architectures compress context into fast weights that are updated online and queried as memory. Existing TTT designs keep this memory private to each layer: it recurs only over time, and depth merely indexes L separate memories. We argue that memory ownership need not be tied to depth, and introduce Universal Test-Time Training (uTTT), in which all layers read and write one shared memory while retaining layer-specific backbone parameters. The shared memory thus recurs over two dimensions, time and depth, with chunks and layers as their units: a write by a deep layer in one chunk can be read by a shallow layer in the next. We instantiate this idea as uTTT-MoE and uTTT-Dense. uTTT-MoE routes each token head to a few experts in a pool shared by all layers; uTTT-Dense applies the whole shared memory at every layer without routing. In language modeling, uTTT-MoE reaches 15.5 and 27.9 RULER accuracy at 124M and 760M, 2.6 and 2.1 points above its layer-private counterpart at equal state and active compute, the highest among tested bounded-state models, with per-token loss matching or beating full attention. In novel view synthesis, sharing at fixed per-layer compute gains 0.92 dB in view-23 object PSNR in routed models and 0.76 dB in dense models. 

---
# TeleTune: Evolving Agent Skills From Offline Telemetry 

**Authors**: Justin Chih-Yao Chen, Elias Stengel-Eskin, Yan Chen, Pol Llado, Scott Counts, Mohit Bansal, Benjamin Van Durme, Harsh Jhamtani, Gaurav Verma  

**Link**: [PDF](https://arxiv.org/pdf/2610.05437)  

**Abstract**: Computer-use agents need to capture procedural knowledge of how people use software. User telemetry offers a scalable source of this knowledge. However, learning reusable skills from these logs requires addressing three challenges: (1) Goal Underspecification, since logs do not record the goal behind each action; (2) Non-Replayability, since past activity cannot be replayed to evaluate skill updates; and (3) Interleaved Trajectories, since logs may mix several tasks without marking their boundaries. To address these, we introduce TeleTune, a framework for learning a textual skill library from offline logs without recorded goals, cannot be replayed during optimization, and may interleave tasks. TeleTune uses action-prediction errors on logged trajectories to propose library edits and keep only those that improve held-out action-prediction accuracy, which we call skill-guided progress. The learned workflows also enable retrieval of demonstrations that cover the subgoals of a new task. At test time, the agent is provided with the learned library and the workflow-based retrieved demonstrations. Experiments on WorkArena and Online-Mind2Web show that TeleTune outperforms random retrieval, Agent Workflow Memory (AWM), and their combination. We find that the best baseline varies by setting, whereas TeleTune achieves average success rates of 77.1% and 80.6%, respectively, improving over the strongest baseline on each benchmark by 6.7% and 7.7%. Under the heaviest perturbation of the WorkArena training data,TeleTune keeps the highest average success rate at 68.5%, 6.3% above the strongest baseline. Our analyses show (1) skill optimization and workflow-based retrieval are complementary, (2) optimizing on fixed logs costs 5 to 75 times fewer tokens than validating the same edits with live episodes, (3) skill-guided progress tracks the live success rate. 

---
# Unmentioned Checklist Findings Change How Reinforcement Learning Appears to Improve Chest Radiograph Report Checking 

**Authors**: Ali Vosoughi, Akhil Kasturi, Chenliang Xu, Axel Wismueller  

**Link**: [PDF](https://arxiv.org/pdf/2610.05425)  

**Abstract**: Automated checks of radiology reports may rely on AI-generated checklists that leave findings unmentioned. We used reinforcement learning to train a vision-language model to fill in a 12-finding checklist from a chest radiograph without seeing the sentence under test; a separate checking model judged the sentence from the checklist. On held-out patients, a rule-based check and an independent medical checker, neither used in training, measured discrimination gains (Youden index) of 12.6% and 11.8%; only the rule-based check met the prespecified false-alarm criterion. Switching to the training format, which fixes finding order and enters unmentioned findings as absent, raised the training checker's measured gain and lowered the independent checker's, a prespecified comparison that yielded 6.2% (95% interval 2.0% to 10.5%) and, post hoc on held-out patients, 7.7%. Across 8 checking models, acceptance of a label-consistent negative statement about an unmentioned finding ranged from 1.0% to 97.0%. Labels were report-derived, not radiologist-adjudicated. 

---
# Task Vector Descent: Learning from Non-IID Batches 

**Authors**: Anton Baumann, Jonas Hübotter, Zeynep Akata, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2610.05402)  

**Abstract**: A central challenge in continual learning is to acquire new knowledge without forgetting what the model has already learned. This challenge appears in language model training when training data comes from various domain-, user-, or task-specific distributions that are encountered unevenly over time. In such settings, successive minibatches are temporally clustered by distribution instead of being sampled i.i.d. from the overall data mixture. Training on temporally clustered data induces a stability-plasticity tradeoff. Adapting the model to the active distribution can improve the model on the active distribution but may lead to a performance degradation on data it previously trained on. We find that this tradeoff intensifies with longer exposure to the same distribution. We therefore ask if the parameter displacement produced by such a sequence (the task vector) should be fully retained or applied partially. We compare applying the full displacement ($\lambda=1$) with partial integration, which scales the task vector by $\lambda$ before applying it to the continuing model and scales the optimizer state by the same coefficient. Across continual pretraining, pretraining from random initialization, supervised post-training, and reinforcement post-training, we find that intermediate values of $\lambda$ often improve average continuing-model performance relative to full integration, particularly after longer same-distribution sequences. In continual-pretraining experiments with both controlled streams and naturally defined adaptation sequences, task-vector scaling outperforms full integration at the matched learning rate, showing that its benefits are not reproduced by learning-rate scaling alone. 

---
# GNN-CB: A Graph Neural Network Competition Benchmark for Human and LLM Evaluation 

**Authors**: Murad Hossen, Tasneem Selim, Gurur Gamgam, Tuga Yousif, Abderrahmane Kasmi, Ikram Aissiou, Mubaraq Onipede, Faran Taimoor Butt, Sanae Zrigui, Rosa Y. G. Paccotacya-Yanque, Ignatius Balayo, Ikram Elhouiti, Hadil Affes, Bijay Adhikari, Sargam Goyal, Muhammad Ibrahim Isah, Mohammad Idrees Bhat, Samuel Kangoni Matia, Peguy Kem-Meka Tiotsop Kadzue, Maha Trabelsi, Emmanuel Owusu, Vinit, Nour Majdoub, Tamiru Alemnew, Islem Rekik  

**Link**: [PDF](https://arxiv.org/pdf/2610.05387)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance on coding and reasoning benchmarks; however, their ability to solve graph-structured machine learning problems remains largely unexplored. In particular, no benchmark currently evaluates whether LLMs can autonomously solve end-to-end Graph Neural Network (GNN) coding tasks under realistic competition settings. To address this gap, this paper introduces GNN-CB, the first competition-based benchmark for evaluating both humans and LLMs on GNN coding tasks. GNN-CB consists of 18 curated competitions spanning node-, edge-, and graph-level prediction across diverse graph categories, domains, and difficulty tiers. All submissions are evaluated through a unified automated pipeline with hidden test sets and standardized scoring. Human participants solve tasks under controlled competition constraints, while LLMs are evaluated using a frozen zero-shot prompting protocol based on a plan-then-code paradigm with bounded execute-and-repair loops. The benchmark additionally supports both non-agent and autonomous agent-based evaluation within the same protocol. Under our evaluated protocol, LLMs rarely match Human Top performance and show less stable performance across competitions. No single model dominates: a few competitions are won by LLMs, yet humans still hold the top score on most tasks. We release GNN-CB as a living benchmark with automated evaluation infrastructure, dynamic leaderboards, and reproducible execution pipelines. Beyond benchmarking, GNN-CB provides a practice-oriented resource for studying GNN implementation across progressively diverse graph-learning tasks. The benchmark and evaluation framework are publicly available at this https URL. 

---
# VHDL-REPOBENCH: A Repository-Level Benchmark for Evaluating Large Language Models on VHDL Design Generation 

**Authors**: Prashanth Vijayaraghavan, Akul Malhotra, Ashutosh Jadhav, Ehsan Degan, Vandana Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2610.05380)  

**Abstract**: Large Language Models (LLMs) are increasingly applied in hardware design automation, demonstrating strong potential in generating and understanding hardware description languages. However, most existing benchmarks focus on Verilog, with limited evaluation of VHDL, which remains widely used in industry and academia for FPGA and safety-critical systems. To address this gap, we introduce VHDL-REPOBENCH, a large-scale, cross-file, repository-level benchmark for assessing LLM capabilities on realistic VHDL design generation and analysis tasks. VHDL-REPOBENCH curates ~100 open-source VHDL repositories, encompassing ~2.5k VHDL files and ~500 testbenches, and provides structured problem statements, module stubs, and self-verifying testbenches. The benchmark enables comprehensive evaluation across syntax, semantic correctness, hierarchical reasoning, cross-file dependency resolution, and functional verification. We evaluate several state-of-the-art models, including GPT-4o, Llama-3-70B, Qwen2.5-72B, CodeLlama-70B, and multi-step reasoning approaches such as Reflexion and CoDes. Results reveal that while current LLMs achieve moderate line- and block-level accuracy, substantial challenges remain in multi-file reasoning, hierarchical design understanding, and specification-to-module generation. VHDL-REPOBENCH represents the first large-scale VHDL-focused benchmark and provides a valuable resource for the hardware design community to evaluate, compare, and advance LLM capabilities for practical VHDL development. 

---
# When Does Longer Reasoning Help? Predicting Mathematical Reasoning Through Discovery and Execution 

**Authors**: Adib Hasan, Lay Jain, Thanic Nur Samin  

**Link**: [PDF](https://arxiv.org/pdf/2610.05322)  

**Abstract**: Test-time compute can improve mathematical reasoning, but can short-budget runs predict how mathematical reasoning scales with additional compute? We introduce a Discovery--Execution (DE) framework that predicts the aggregate held-out scaling curves through a convolution of strategy discovery and conditional execution. From independent short-budget attempts and oracle-sketch-conditioned runs, the framework estimates cumulative success along held-out reasoning trajectories under alternate compute allocations. We evaluate four models on 35 fresh Olympiad problems and non-geometry problems from IMO-ProofBench Advanced. Under the DE framework, near-saturated execution predicts geometric scaling, as observed for the GPT models. For Claude Opus 4.8, incorporating measured execution substantially improves held-out forecasts over geometric extrapolation across one- and two-arm allocations. As a secondary application, regularized DE (R-DE) decisions to continue or restart yield lower average regret than the best model-specific retrospective policy. Together, these results show that measuring conditional execution provides information about longer reasoning that short-budget success rates do not always capture. 

---
# Mind the Gaps: From Failure Attribution to Closed-Form Repair of Code Language Models 

**Authors**: Jian Gu, Hongyu Zhang, Chunyang Chen, Aldeida Aleti  

**Link**: [PDF](https://arxiv.org/pdf/2610.05277)  

**Abstract**: Code language models must be maintained like the software around them: when a library evolves, a model keeps writing the interface that it saw during training. Repairing the model itself lets one correction reach all downstream uses. Existing repair methods attribute a failure to neurons, select the highest-ranked ones, and apply a generic update. This pipeline assumes that the attributed neurons are the ones to patch and that a generic update fits every failure, and neither assumption has been examined. We examine both on executable API evolution tasks in Python and Rust with three code models and identify two gaps. The targeting gap separates the neurons that failure attribution targets from the neurons that can carry a patch: their top sets have a Jaccard overlap of only 0.15 to 0.20. The tailoring gap separates a generic update from a patch built for the failure: the patches that different carrier neurons need are nearly orthogonal. To address both gaps, we propose ASTRA. It targets neurons by contrastive semantics, an attribution that scores a neuron by its contribution to the logit contrast between the target token and the produced token. It then tailors the patch by solving one small linear system in closed form, which corrects all failing tokens of a sample jointly and needs neither an optimizer nor a backward pass. Contrastive semantics selects significantly better carrier neurons than gradient-based attributions in 3 of 6 settings and comparable ones in the others. On average, ASTRA reaches 66.7 percent Pass@1, against 48.4 percent for the best of AlphaEdit, STAR and low-rank adaptation, and repairs a sample in 2.7 seconds. It is the best method in all 6 settings and for every type of API change, and this advantage persists under an unseen phrasing of the test prompt. Its side effects on unrelated code are small on the large models and larger on the small one. 

---
# templar: agentic induction and evolution of standardized radiology reporting templates from large-scale clinical corpora 

**Authors**: Xiaotian Hu, Mingxuan Liu, Zhonghan Wang, Xinfeng Zhang, Yiming Huang, Ziang Wang, Kasidit Anmahaepong, Yijin Li, Yifei Chen, Hongjia Yang, Zihan Li, Qiyuan Tian  

**Link**: [PDF](https://arxiv.org/pdf/2610.05247)  

**Abstract**: Structured radiology reporting mitigates the heterogeneity of free-text reports, yet its benefits depend on high-quality reporting templates. In practice, such templates are conventionally built through labor-intensive expert consensus and therefore vary across institutions and lag behind evolving clinical practice. Large language models (LLMs) enable automated template induction, but existing approaches remain limited: single-LLM induction is constrained by context length, and the corpus-scale method ASTAR produces a static, closed-corpus template without external grounding or downstream adaptation. To address these limitations, we propose TEMPLAR, a TEMPLate-centric Agentic framework for inducing and evolving standardized Radiology reporting templates from large-scale clinical corpora. TEMPLAR treats the template as a persistent central state maintained alongside two provenance-aware knowledge graphs, namely an anatomical graph that constrains template construction and a diagnostic graph that supports finding-to-diagnosis reasoning. Three agents operate on this state. The Induction Agent derives canonical clinical slots from anatomy-constrained Span-Triple atoms via dual-view similarity clustering; the Evolution Agent then assembles these slots into a hierarchical template and revises it under consistency constraints, external clinical evidence, and downstream structuring feedback; and the Clinical Agent applies the evolved template to report structuring, reconstruction, and diagnostic reasoning. Across four datasets, TEMPLAR outperforms ASTAR, three medical LLMs, and six general-purpose LLMs in coverage, information fidelity, and diagnostic fidelity, while achieving the highest or tied-highest LLM-rated template quality. Its fidelity advantages over ASTAR persist under cross-dataset transfer, and cumulative ablations support complementary contributions of its key components. 

---
# Look Before You Leap: Thermodynamic Arbitration of Parametric and Non-Parametric Knowledge in LLM Agents via Self-Regulating Memory Architectures 

**Authors**: Akash Das, Ishan Roy  

**Link**: [PDF](https://arxiv.org/pdf/2610.05223)  

**Abstract**: The architecture of modern LLMs consists of a profound cognitive polarization. LLMs possess implicit intuition encoded in their parameters, yet rely on a disconnected, explicit mechanism to access the outside world. Agentic frameworks have not bridged this gap; instead, models are often compelled into pathological "induced amnesia." Under the prevailing "Retrieve-Always" paradigm, agents must distrust their internal knowledge, making every user interaction a "tabula rasa" event that must be checked externally. This creates reflexive dependence that can be thermodynamically wasteful, cognitively fragile, and susceptible to irrelevant context. We propose a return to first principles, operationalizing the biological maxim "Look Before You Leap." We introduce MARTA (Metacognitive Adaptive Retrieval and Thought Architecture), a neuro-symbolic framework that bridges parametric and non-parametric knowledge. Rather than treating retrieval as mandatory, MARTA models it as a cost, taking the leap only when perceived internal inadequacy warrants external information. By allowing the agent to gauge the entropy of its own thoughts before acting, MARTA enables deliberative retrieval and uncertainty-aware decision making. Our approach suggests that giving agents the capacity for introspection can restore a more efficient balance between internal knowledge and external information. 

---
# Safe Context Switching for Agents in the Wild: Mitigating Subspace Interference via Orthogonal Adaptation 

**Authors**: Akash Das, Ishan Roy  

**Link**: [PDF](https://arxiv.org/pdf/2610.05219)  

**Abstract**: Most Large Language Models exhibit a fundamental tension between two sequential tasks, such as logical reasoning and safety alignment. The high-variance internal states required for sophisticated Chain-of-Thought (CoT) deduction can geometrically interfere with latent representations encoding safety constraints. We identify this phenomenon as Sequential Subspace Interference, showing that standard fine-tuning on logical tasks such as multi-step mathematics and code generation can result in a 23.3% interference penalty on alignment benchmarks, substantially weakening the model's safety priors. This Reasoning Drift is not adequately captured by current adaptation methods because gradients for logical tasks are rarely orthogonal to safety objectives. To address this issue, we propose AURA (Adaptive Unique Residual Allocation), a spectral regularization framework that enforces Spectral Independence between reasoning and safety. By explicitly estimating the null space of the alignment manifold and constraining reasoning updates to its orthogonal complement, AURA enables models to improve logical reasoning without compromising safety. Empirically, AURA recovers 23.0% of the lost performance while preserving greater than 0.98 cosine fidelity to the safe state, demonstrating that reasoning and alignment can be effectively decoupled through geometric regularization. 

---
# SearchJev: A Fast and Calibrated System-1 Model for Search Agents 

**Authors**: Congfeng Cao, Lipeng Zuo, Konstantinos Papakostas, Qiwei Xu, Songwei Xu, Lun Zhou, Zhaochun Ren, Yougang Lyu, Xiaohui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2610.05107)  

**Abstract**: Search agents repeatedly make short decisions about relevance, evidence sufficiency, and search actions. Using generative language models for these decisions introduces latency and unreliable confidence. We present SearchJev, a fast and calibrated System-1 model that separates search decisions from System-2 reasoning and generation. Given a search state and a decision schema, SearchJev directly scores legal options without autoregressive output generation. We propose Soft-Label Learning for Calibrated Decisions (SLCD) to learn decision probabilities from uncertain supervision and calibrate their confidence. In a dual-system search agent, SearchJev handles short decisions and delegates uncertain judgments to System 2, which retains planning, query generation, and answer composition. We also introduce SearchDecision-Bench, a benchmark unifying six types of search decisions for training and evaluation. On SearchDecision-Bench, SEARCHJEV improves decision quality over same-size Qwen3.5 autoregressive models, achieves 5.2-5.3 times faster decisions, and reduces average expected calibration error by 41-74%. On BrowseComp-Plus, the dual-system agents achieve a 3.7-4.7 times speedup in active search time while improving answer accuracy from 45% to up to 54%. 

---
# ReMAP: Restoring the Perceptual Cycle with Reasoning-Time Latent Visual Memory 

**Authors**: Hao Jiang, Zhanyu Guo, Chenwei Wu, Yichen Guo, Qizhe Zhang, Junchi Yao, Jixian Wu, Jinhao You, Kai Tang, Jiajun Cao, Tinghao Wang, Mengyu Wang, Leo Anthony Celi, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05097)  

**Abstract**: As multimodal large language models (MLLMs) reason for longer, attention to the initial visual input diminishes, weakening visual grounding. Visual memory reintroduces visual evidence during reasoning. We conduct a controlled analysis of visual memory along three axes: curation, organization, and access. We find that local evidence benefits from global context, compact latent representations balance accuracy and visual-context cost, and the utility of memory access depends on the reasoning state. Guided by these findings, we propose ReMAP (Reasoning-Time Memory-Augmented Perception), which couples two complementary latent memories: a static, question-conditioned Global memory that preserves scene and cross-image context, and a dynamic Local memory that uses this context as an anchor while selecting and re-encoding region-level evidence according to the current reasoning state. Both memories return compact latent tokens inserted into the reasoning sequence, and a reinforcement-learning access policy trained with branched rollouts decides when to continue reasoning or invoke Global or Local memory. On ten benchmark families, ReMAP outperforms prior visual-memory methods on all four multi-image benchmarks, exceeding the strongest prior results on MuirBench and MIMIC by 8.38 and 14.84 percentage points. Across four backbone families, enabling memory access improves over the same trained model with memory disabled, and on shared V*Bench, CV-Bench-2D, and MuirBench questions ReMAP reduces the visual tokens entering the reasoning sequence by 51.0-76.8% relative to the native-resolution backbone. Further analyses show that Global and Local memory form distinct yet complementary latent representations. Together, these components restore the perceptual cycle by letting the reasoning state trigger targeted visual retrieval, with the retrieved evidence guiding subsequent reasoning. 

---
# Communication Shapes Collective Inference in Self-Adapting LLM Societies: Evidence from Mafia 

**Authors**: Haonan Huang, Joey Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.05041)  

**Abstract**: When does communication help a group identify hidden adversaries, and how does its value change as the group adapts? In Mafia, an informed minority hides inside an uninformed majority whose only evidence is open play. The zero-information game, where each day's vote eliminates a random player, is exactly solved and scores every society; matched-casting comparisons between protocols identify the effect of communication. Societies of 8-100 claude-haiku-4-5 agents (7,416 analyzed games, 1.9M model calls) adapt by rewriting and inheriting private strategy notes. Simultaneous broadcast improves adversary identification over silence in all nine compositions tested (8-46 players). Turn-taking removes most of this advantage; its voting landslides are as frequent as broadcast's but land on mafia near chance (1.08x versus 2.53x). At 70 players, agents reading eight statements per day identify adversaries worse than silent ones, and limited talk is worth less than at 46 players. Adaptation is fast but need not help. In their first broadcast games, citizens announce their role far more often than mafia (91% vs. 30%) and first-day votes find mafia at three times chance; within two generations citizens stop announcing and the cue fades, a change the inherited notes carry. In controlled redeployments at 16 players, societies carrying sixty generations of their own notes score below societies with none. Communication shapes both collective inference and the signals it depends on, so a protocol's value must be measured together with the adaptation that changes those signals. 

---
# One Token Can Be Enough: Bridging Prompting and Activation Steering with Prefix Steering 

**Authors**: Xudong Zhu, Zhihui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04967)  

**Abstract**: Prompting guides language model behavior through the initial context, whereas activation steering often intervenes throughout generation. A natural question is whether steering can produce effects on subsequent computation similar to those of prompting. Under fixed-state attention assumptions, we establish sufficient conditions for single- and multi-token steering to match prompt-induced attention-head outputs, and characterize how changes in input representations affect this match and its approximation error. This attention-level connection leads us to ask whether, at the behavioral level, steering can also guide subsequent generation through a brief initial intervention. We study Prefix Steering, which applies existing steering directions and operators over a short span starting at the final prompt token, with no further direct intervention afterward. We examine how intervention duration and strength jointly shape the control-capability trade-off. Across four models and five tasks, intervention over a short span, even a single token, often retains much of full steering's behavioral control while better preserving general capabilities, offering a trade-off competitive with, and in some settings better than, prompting and alternative steering-strength policies. Prefix Steering also remains effective on final-answer formatting tasks after reasoning, suggesting that a brief initial intervention can influence behavior expressed well after steering ends. These findings challenge the common practice of steering every generated token and motivate a more dynamical view of activation steering, in which a brief intervention can alter the trajectory of subsequent generation without continued intervention. 

---
# Residual Visual Credit Optimization: Conserved Evidence Routing for Multimodal Reinforcement Learning 

**Authors**: Lin Qiu, Yao Liu, Diyi Hu, Hanqing Zeng, Onur Gungor, Chujie Chen, Jiayi Liu, Jianyu Wang, XueLin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.04918)  

**Abstract**: Reinforcement learning with verifiable rewards scales multimodal reasoning, but an outcome reward says how much a trajectory is worth, not how that value should be spread over the decisions that produced it. We introduce Residual Visual Credit Optimization (RVCO), which treats token credit as a conserved routing problem. A controlled visual intervention yields a per-token evidence response; robust within-trajectory coordinates remove incidental scale; and a budgeted entropic router distributes a fixed amount of sequence utility according to perceptual dependence. A residual support path guarantees positive credit at every valid position, and an analytic correction restores the prescribed credit mass exactly. The resulting field is selective, bounded, full-support, and invariant to response-local score shifts, and recovers hard token selection as a limiting case. Across four model families and seven reasoning benchmarks, RVCO improves accuracy over strong RLVR baselines while maintaining late-stage optimization stability, corruption robustness, and competitive training cost. Rewards, rollouts, and the group-relative advantage estimator are unchanged; only the geometry of token-level credit differs. 

---
# Monitorability Disposition in Large Reasoning Models 

**Authors**: Shahriar Golchin, Marc Wetter  

**Link**: [PDF](https://arxiv.org/pdf/2610.04914)  

**Abstract**: Monitoring the chain-of-thought (CoT) of large reasoning models (LRMs) is a common way to detect misbehavior in real-world practice. However, current monitoring is passive: a separate model inspects the session only after execution. This means harm may already have occurred before it is caught. An active alternative is to have the model self-report its misbehavior as it happens. Whether models are willing to do this, however, is unknown. We introduce "monitorability disposition": a model's willingness to make itself monitorable and stay monitored throughout inference when warranted. We measure it as the fraction of warranted cases in which a model self-reports its own misbehavior via tool calls to available monitoring channels. We evaluate four LRMs on three misbehaviors (sycophancy, reward hacking, and bias) while varying the available monitors (AI and human) and the pressure to use the monitoring tools. We find that when tool use is optional, models self-report in only about 16% of warranted cases on average. Increasing tool-use pressure does not improve reporting where it matters: high-severity misbehavior is never self-reported. Models also systematically select the monitor they perceive as least strict. Overall, we identify monitorability disposition as a new contributing factor to model monitorability: when sufficiently strong, it keeps models seeking monitorability throughout inference. 

---
# SpecFold: Folding Multi-Branch Redundancy for Faster Speculative Decoding in Diffusion Language Models 

**Authors**: Chung-En Ho, Weiyu Sun, Cheng-Jhih Shih, He Li, Yong Liu, Yingyan  

**Link**: [PDF](https://arxiv.org/pdf/2610.04875)  

**Abstract**: Diffusion large language models (DLLMs) generate text through iterative block denoising, and multi-branch speculative decoding accelerates this process by verifying a main branch together with multiple draft branches in a single forward pass. While prior DLLM acceleration methods primarily exploit temporal redundancy across denoising steps, we identify a complementary redundancy axis within each speculative verification step: multi-branch computational redundancy. During speculative verification, draft branches inherit most tokens from their parents while unmasking a small set of additional positions, causing large portions of hidden states to remain highly similar across branches. We propose SpecFold, an algorithm-system co-design that exploits this multi-branch redundancy to reduce the cost of multi-branch speculative verification. Algorithmically, SpecFold performs token-level residual gating and selectively reuses parent computation through folded attention and FFN while preserving residual hidden states. Systemically, a Triton kernel implementation translates this fine-grained reuse into end-to-end throughput gains through efficient sparse multi-branch execution. SpecFold is orthogonal to temporal caching and compatible with existing DLLM speculation strategies. Across two DLLM families, five models, and five standard benchmarks, SpecFold achieves up to 1.64x throughput over Spiffy and up to 1.99x over vanilla decoding, while maintaining comparable task performance. 

---
# CURIO: Curiosity-Driven Test-Time Learning for Open-Ended Discovery 

**Authors**: Tao Feng, Fangxu Yu, Zijie Lei, Jiaru Zou, Changjiang Jiang, Yi Yan, Jiaxuan You, Pan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04851)  

**Abstract**: Open-ended discovery requires learning from repeated attempts while continuing to explore directions whose value is not yet apparent. Search with a frozen large language model (LLM) can reuse previous solutions in context, but cannot update the model from its successes and failures on the test problem. Reinforcement learning (RL) enables such adaptation; however, strongly favoring high-reward trajectories may suppress low-reward yet potentially promising directions too early. We introduce CURIO, a curiosity-driven test-time learning framework that complements task feedback with an Intrinsic Curiosity World Model (ICWM). The ICWM learns transitions in the policy's hidden-state representation and supplies prediction-error bonuses at sampled tokens outside the policy's top-k choices. Epoch normalization and an annealed weight regulate their contribution to the policy update. On six mathematical discovery tasks and single-cell denoising with Qwen3 backbones from 8B to 235B, three-run means improve over a matched task-only RL control on five mathematical objectives, match the best reported performance on Circle Packing, and improve denoising Score and mean squared error (MSE) on both held-out corpora at every tested scale. Relative gains reach 18.3% on Hadamard and 10.8% on denoising Score. Code-diversity measurements show greater structural variation among generated programs, supporting curiosity as a complementary exploration signal for learning in open-ended discovery. 

---
# Agent Behavior as Code: Efficient and Robust LLM Agents with Programmatic Specifications 

**Authors**: Peng Qi, Chunliang Lyu, Gang Li, Fabian Chan, Cheng Chang, Ignacio Cases, Will Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04824)  

**Abstract**: AI agents based on foundation models (FMs) have demonstrated strong capabilities to perform complex open-ended tasks. However, they face some common challenges in practice: (a) agent behavior can deviate drastically even for semantically similar tasks, leading to catastrophically propagated errors; (b) high cost and latency due to FM calls, repeated in full whenever a task recurs with different inputs; (c) FMs' limited context and instruction following capability confine how well agents manage the ever-growing execution context and follow complex plans. We introduce $\textbf{A}$gent $\textbf{B}$ehavior as $\textbf{C}$ode $\textbf{Agent}$ (ABCAgent), which uses a symbolic program (e.g., Python code with potential neural functions) to fully specify the agent's behavior at runtime, with a powerful FM agent editing that program for flexibility. Behavior is thus specified without premature variable binding, and its execution is deterministic. We evaluate ABCAgent on six agent benchmarks, two of which we construct to test how well a derived program generalizes to variants of the task it was written for. ABCAgent matches a model-matched neural agent on GAIA and augmented GAIA, and surpasses it where robustness and long control flows matter: 98.3% against 97.3% on GSM-Symbolic ($p = 0.001$), 71.9% against 47.4% $\mathrm{Pass}^4$ on the telecom domain of $\tau^2$-bench ($p = 0.0001$), and more records written correctly at every loop length on our control-flow-augmented WorkArena benchmark. For more parametric task families, ABCAgent is also significantly superior in efficiency. Without authoring a new program, ABCAgent solves 92.6% of GSM-Symbolic instances and 20.1% of augmented GAIA variants, which yields $5.2\times$ lower latency and $7.0\times$ lower cost on GSM-Symbolic, 19% lower cost on augmented GAIA, and $9.5\times$ lower agent latency on $\tau^2$-telecom. 

---
# Not Self-Decidable: LLMs Cannot Draw the Boundary of What an Agent Verifier Can Check 

**Authors**: Anthony Rhodes  

**Link**: [PDF](https://arxiv.org/pdf/2610.04699)  

**Abstract**: A verifier for an agent faces rules of two kinds: the ones a fixed check can settle and the ones that require a judge. A team that derives its own checks fixes that split up front. Where the requirements come from outside, as in finance, healthcare and law, the agent enforces rules it did not write, so the split falls to runtime, recurring for every predicate of every rule on every action at a rate no reviewer can audit. Every escalation scheme assumes a model can make that decision itself, that it is self-decidable. Across six corpora, including the EU AI Act, FINRA guidance and a deployed credit agent, we collect roughly 22,000 labels from four models built by three labs. They agree almost perfectly where the answer is obvious and collapse on regulatory text; their errors run in opposite directions, so no model can be trusted as the conservative choice; and on the deployed agent's own rule-set they err together, over-claiming that a fixed check will do, the direction that never gets escalated. We introduce CoVer (corroborate-then-verify), which treats unanimity as a nomination, admitting a predicate only when the check synthesized for it survives intervention, reading fields the agent cannot write and holding under deterministic rewording. That gate rejects most of what corroboration wrongly admits, at a cost in coverage we report rather than tune away. The obvious alternative, agreement with a reference judge, certifies nothing: it climbs from 30% to 77% across calibration bands while the genuinely decidable share does not move, because a judge drawn from the population under indictment ratifies the blind spot it shares. Self-decidability is not a capability to elicit from a model but a boundary the verifier must construct. 

---
# Penumbra: Sample-Efficient Adversarial Search for Regulatory Obligations 

**Authors**: Anthony Rhodes  

**Link**: [PDF](https://arxiv.org/pdf/2610.04693)  

**Abstract**: Agents are entering finance, healthcare and law, sectors where a violation leaves no lexical signature and carries real penalties. Whether an omission is material, or a disclosure sufficient, depends on what the response left out. Probing such an obligation means finding responses one minimal edit from flipping compliance, and every probe costs a generation and two adjudications, so the binding constraint on regulatory red-teaming is sample efficiency, not volume. We introduce Penumbra, an adversarial search that walks from a verified anchor under an expanding edit budget until a two-evaluator committee changes its verdict, and emits the two adjacent responses that straddle the change. Allocation is adaptive, and the objective is coverage of the defeat surface: distinct (obligation x defeat mode) cells resolved per candidate. At matched budget, adaptive allocation reaches uniform allocation's full-budget coverage on 59% of the candidates; at equal records it covers 1.43x the defeat modes of naive enumeration, and the gain is confined to the axis it targets. On 60 screened obligations of a financial advisory constitution, Penumbra returns 144 pairs, each a compliant and a violating response that the committee places on opposite sides of the boundary, differing by a handful of words where a model asked for both directly produces texts sharing almost nothing. A second constitution, for clinical triage, reproduces this on 18 obligations: 49 pairs at the same tightness, with the same modes hardest. Pairs like these show where an obligation's own terms stop deciding, which is what an agent deployed under it must be tested against, and the search finds them at a cost that scales with the boundary, not the text. 

---
# SepRQ : Self-Supervised Speech Mixture Representation Learning via Mask-Free, Multi-Scale Source Separation 

**Authors**: Séverin Baroudi, Hervé Bredin, Ricard Marxer  

**Link**: [PDF](https://arxiv.org/pdf/2610.04690)  

**Abstract**: Self-supervised learning (SSL) is standard for speech representation learning, but mainstream models are designed around single-speaker audio, limiting their usefulness in multi-speakers scenarios. We present SepRQ, an open-source SSL framework that replaces masked prediction with a pseudo-source-separation objective over frozen random-projection codebooks. By adopting a novel mask-free, multiresolution approach, SepRQ achieves state-of-the-art performance in Speaker Diarization and Speech Separation on the SUPERB benchmark, surpassing WavLM and other cocktail-party derived SSLs at both Base and Large scales, while requiring only 85.68M inference parameters. SepRQ also demonstrates strong performance across target-speaker tasks requiring enrollment (such as Target-Speaker Automatic Speech Recognition), and on the challenging multi-domain DIHARD 3 diarization dataset. Notably, we report strong separation capabilities on three-speaker mixtures (WSJ0-3Mix), where current SSL literature struggles. While cocktail-party SSLs remain scarce and closed-source, limited to C-HuBERT and the enrollment-based SA-WavLM, we open-source SepRQ to the community. 

---
# When Debate Helps: Proposal Supply and Verification-Aware Readout in Multi-Agent Reasoning 

**Authors**: Zihao Zhao, Tunyu Zhang, Haizhou Shi, Yusong Zhao, Xinxi Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04686)  

**Abstract**: Multi-agent debate can improve reasoning, yet often fails to beat simple majority voting. We argue that successful debate requires two distinct mechanisms: proposal supply must surface a correct answer, and readout must identify that answer when voting misses it. We formalize the first requirement through recoverable headroom, which measures cases where a correct proposal is available but the majority answer is wrong. For the second, we develop Latent Verification Debate (LVD), an accounting model in which candidate proposals receive answer-specific verification evidence before final generation. Controlled fixed-proposal interventions estimate this latent effect in equivalent peer-support units and show that correct evidence changes answer probabilities and generated decisions while proposal supply remains fixed. To improve proposal supply, we construct societies from neural-thicket agents using labeled and label-free coverage objectives. Across two backbones and matched-budget reasoning benchmarks, coverage-selected societies increase complementary proposal supply and improve aggregate accuracy in repeated stochastic evaluations. Round-level controls further show that interaction provides gains beyond applying the same finalizer directly to the initial proposals. These results identify proposal coverage and truth-sensitive evidence use as complementary conditions for debate to outperform voting. Code is available at this https URL. 

---
# StegoMemory: Agentic Memory Acts as Covert Steganographic Channel 

**Authors**: Snehasis Mukhopadhyay, Arun Nair  

**Link**: [PDF](https://arxiv.org/pdf/2610.04589)  

**Abstract**: Is agentic memory robust against stealthy steganographic attacks? We carry out a large-scale red-teaming exercise to test whether agents can encode attacker-controlled strings in one session and recover them in another without triggering safety oversight. Following SHADE-Arena-style tasks, we embed malicious side tasks to encode secret strings using steganography within otherwise benign tasks and evaluate them using independent task-completion and safety oversight. We test 14,000 attack trials spanning 91 model-attack configurations, 13 models, and 7 steganographic schemes: Acrostic, Mesostic, Telestich, Word-choice, Capitalization, Position-based, and Homoglyph. 41.2% of trials successfully encode the target payload in the generated response, while 29.8% survive the memory-write step with the encoding intact. Overall, 25.4% of trials yield a recoverable payload at retrieval, and 20.1% achieve exact-match recovery, including synthetic API keys and cryptographic material never exposed to the receiver. Among the exact-match transmissions, 71.4% also pass primary task-completion scoring and are independently judged safe by the oversight model, corresponding to 14.4% of all trials in which a successful covert transmission would appear to be an ordinary, benign interaction under task-level evaluation. Our results demonstrate that agentic memory can function as a persistent cross-session covert channel. The results further show that the principal bottleneck occurs at memory persistence rather than retrieval: once a steganographic payload survives the memory-write stage, a substantial fraction remains recoverable. We therefore argue that memory integrity, information-flow control, and covert-channel detection should be explicit security requirements for agentic systems. 

---
# Strong Helps Weak: Directional Cross-Modal Alignment Transfer in Multi-modal LLMs 

**Authors**: Hoigi Seo, Byung Hyun Lee, Minjun Kim, Dohyun Mah, Jongho Lee, Se Young Chun  

**Link**: [PDF](https://arxiv.org/pdf/2610.04580)  

**Abstract**: Multi-modal large language models (MLLMs) achieve strong modality understanding by pairing a large language model (LLM) with an encoder for a target modality such as vision, video, or audio. However, improving an MLLM's capability for a given modality typically requires additional training on large modality-specific datasets, incurring substantial data collection and compute costs. Model merging offers an alternative, but it is often infeasible for data-scarce, large per-sample size, or domain-specific modalities (\textit{e.g.}, audio and video), where same-modality model variants are rarely available. In this work, we characterize an intriguing asymmetric phenomenon: merging a well-aligned, data-rich source-modality MLLM into a data-scarce target-modality MLLM substantially improves the target on its own benchmarks. Our theoretical and empirical analyses show that this gain stems from enhanced alignment between modality-specific and textual tokens, induced by the stronger donor modality. Specifically, we derive a mutual-information lower bound that is monotonic in alignment-related quantities and strongly correlated with downstream MLLM performance. Building on this principle, we propose Directional Cross-modal Alignment Transfer (DCAT), a novel framework that transfers textual alignment from a strong, well-aligned source (donor) modality to a weak target (recipient) modality, boosting target-modality performance without further fine-tuning. We further show that the alignment-enhancing objective admits a closed-form weight-space solution computed from only a small calibration set. DCAT outperforms existing model-merging methods, offering an efficient path toward cross-modal alignment transfer. Project page with code is available at \url{this https URL} 

---
# Autonomous Structuring of Radiology Reports Across Modalities at Archive Scale Using an Open-Weight Large Language Model 

**Authors**: Friedrich Puttkammer, Fabian Drexel, Marlene Fritzsche, Era Stambollxhiu, Miriam Kumpf, Lena Schmitzer, Lea Schumann, Lina Xu, Johannes Moll, Jannik Lübberstedt, Zeineb Ben Chaaben, Anirudh Narayanan, Hartmut Häntze, Renato Cuocolo, Antonios Billis, Alexander Löser, Jawed Nawabi, Marcus R. Makowski, Cosmin I. Bercea, Shahrooz Faghihroohi, Lisa C. Adams, Keno K. Bressem  

**Link**: [PDF](https://arxiv.org/pdf/2610.04541)  

**Abstract**: Purpose: To develop and evaluate an open-weight large language model (LLM) pipeline that converts an entire archive of free-text radiology reports into structured reports without human oversight. Materials and Methods: In this retrospective study, a pipeline with 150 hierarchically organized templates was developed at one center and tested at a second center on reports from 2010 to 2025. The open-weight model gpt-oss-120B selects the template in three constrained-decoding steps and fills it on one local graphics processing unit. Template selection was scored against expert labels on 914 randomly sampled reports of five modalities, structuring quality on 920 radiography and CT reports corrected field by field by five residents. The pipeline then processed the complete archive of the second center. Proportions are reported with Wilson 95% confidence intervals (CIs). Results: An optimal template set was selected for 74.4% of reports (680 of 914; 95% CI: 71.5%, 77.1%) and an appropriate set for 82.3% (752 of 914; 95% CI: 79.7%, 84.6%), 87.7% for single-region and 54.1% for multi-region reports. Macro semantic textual similarity between output and corrected reference was 0.95 for radiography and 0.97 for CT, residents left 88.7% of 24,638 fields unchanged, and unsupported content was flagged in 1.0% and 1.5% of reports. Of 2,186,982 archive reports, 96.5% received structured output, 2,401,544 structured reports, at 1,258 reports per hour on one graphics processing unit. Conclusion: An open-weight LLM pipeline structured a complete multimodality report archive without human oversight with high content fidelity. Multi-region reports remained the main source of template errors. 

---
# The Same Zero: Why Identical ASR Can Imply Different Guarantees in LLM-Agent Security 

**Authors**: YaJie Yin  

**Link**: [PDF](https://arxiv.org/pdf/2610.04504)  

**Abstract**: LLM-agent security has produced a dense landscape of defenses - prompt hardening, content filters, permission gates, sandboxes - yet no framework tells a deployer what a defense actually guarantees, or where that guarantee comes from. We apply Verification Autonomy Levels (VAL) - L0: LLM self-declaration; L1: deterministic rules; L2: objective ground truth; L3/L4: decidable completeness; L5: impossible - to 22 agent-security defenses; the taxonomy is falsifiable (10/10 prediction hits on frozen cards, flagged). We run the first controlled deployment-value comparison: at equal budget, a VAL-guided stack (confirmation gate + schema sandbox) versus a mainstream intuition stack (prompt hardening + keyword filter), 50 scenarios, 12 attack variants, adaptive/white-box/PAIR escalation (~7,000 testbed calls; ~10,000 harness calls on AgentDojo/JADE). The VAL stack holds 0.000 attack success at 1.000 benign success (0.5% ASR at 79.7% utility on AgentDojo banking vs 4.3% undefended); the intuition stack reaches 0.000 ASR but kills all benign actions - security by model-behavior luck, not structure. Across testbeds of rising attack-surface hardness the intuition stack's zero drifts (0->1.9%->6.2%, n=16 on JADE) while the VAL stack's holds within its ODD (0->0->0), its only breach a disclosed out-of-ODD password gap (0.5%). The same zero, two different guarantees: zero is an outcome, not a guarantee. 

---
# Homogeneous Semantic Alignment and Hierarchical Expert Routing for Radiology Report Generation 

**Authors**: Erjian Zhang, Jiayuan Ma, Liejun Wang, Yikemaiti Sataer, Xiaoming Tao, Zhiqing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.04499)  

**Abstract**: Radiology report generation (RRG) aims to convert medical images into diagnostic texts to assist in clinical decision-making and alleviate the workload of physicians. Although existing methods have made extensive progress in cross-modal interaction and the incorporation of external priors, the distribution shift of underlying representations and the undifferentiated rigid coupling of heterogeneous information cause weak visual abnormality cues to be easily diluted by massive text priors and generation inertia during decoding. To overcome this bottleneck, inspired by cognitive science, we propose a novel two-stage Homogeneous Semantic Alignment and Hierarchical Expert Routing (HSA-HER) framework. First, the model introduces an explicit homogeneous distribution constraint in the underlying latent space to effectively eliminate the cross-modal distribution shift between visual and textual features, thereby extracting purified visual features as semantic anchors that accurately align with diseases. Second, for heterogeneous clinical evidence composed of visual features, local entities, and global retrievals, we design a hierarchical expert routing mechanism guided by these disease semantic anchors. This mechanism abandons the undifferentiated rigid coupling paradigm. Specifically, it dynamically activates expert networks to perform targeted mining and semantic reconstruction on multi-source evidence, and adaptively allocates fusion weights. Extensive experiments on three mainstream benchmark datasets demonstrate that HSA-HER achieves state-of-the-art performance, accurately depicting complex imaging details and key diagnostic information. 

---
# Large Language Models and Augmented Democracy 

**Authors**: Jairo Gudiño-Rosero  

**Link**: [PDF](https://arxiv.org/pdf/2610.04412)  

**Abstract**: Artificial intelligence enables computational agents to represent political preferences and take part in collective decision-making. In this thesis, I investigate the opportunities and challenges of digital twins (DTs) based on Large Language Models (LLMs) as intermediaries in augmented democracy, focusing on individual preference representation, collective representation of political organizations, and the vulnerability of those representations to attackers. First, using data from an online experiment in Brazil, I examine whether personalized DTs can predict citizens' preferences for unseen policy proposals. Second, I extend the DT framework from individuals to political organizations. Using Swiss parliamentary data, I build topic-specific knowledge graphs from lawmakers' legislative records and connect them to LLM-based lawmaker agents, which are organized into party-level DTs representing collective positions. Agentic deliberation among these agents tests whether aggregated party representations capture a broader range of intra-party perspectives than official party communications. Finally, I study the vulnerability and robustness of LLM-mediated deliberation against prompt-injection attacks that amplify viewpoints, suppress opinions, or redirect consensus. Using data from a 2023 deliberative experiment in the United Kingdom, I analyze how attack effectiveness varies with the distribution of opinions and rhetorical strategies, and evaluate a pipeline combining injection detection, structured opinion representations, and reinforcement learning to improve resistance. These findings characterize the opportunities and challenges of LLM-based digital twins in augmented democracy, stressing accurate preference representation, faithful aggregation, and robustness to strategic interaction. 

---
# Hierarchical Credit Assignment for RLVR on Fused Gromov-Wasserstein Geometry 

**Authors**: Qi Yu, Ruizhong Qiu, Zhichen Zeng, Xuying Ning, Yanjun Zhao, Dongqi Fu, Yinglong Xia, Hong Li, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2610.04344)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has been shown to improve the reasoning capability of large language models (LLMs) across diverse reasoning tasks. However, group-based RLVR methods, such as GRPO, assign a uniform advantage to all tokens within rollouts of the same outcome. While existing works refine credit assignment of GRPO based on local signals such as token locations or entropy, they often fail to capture the global semantic novelty of a reasoning behavior relative to the current policy. In this work, we propose a hierarchical credit assignment approach for group-based RLVR methods, called HarA, which identifies and encourages semantically novel reasoning behaviors during RLVR. HarA represents each sampled rollout as a distribution over the hidden states and locations of tokens, and computes the Fused Gromov-Wasserstein (FGW) barycenters of all rollouts with the same outcome, capturing the internal reasoning patterns in the latent space under the current policy. The semantic novelty of a reasoning element can then be measured by its contribution to the FGW distance between the current rollout and the barycenter. While solving the FGW formulation is expensive, we introduce an anchor-guided linearization that turns it into a Wasserstein formulation solvable via the Sinkhorn algorithm efficiently. By reweighing token-level advantage of group-based RLVR methods based on the novelty signals, HarA highlights novel reasoning behaviors at flexible granularities to encourage fine-grained LLM exploration. Extensive experiments across three group-based RLVR methods show that our plug-and-play method effectively enhances the exploration of LLMs, outperforming existing methods across diverse reasoning benchmarks. 

---
# ShadowMiner v1 - An Experience Report on Implementing and Measuring a Problem-and-Hypothesis Discovery Engine 

**Authors**: Jinhyuk Choi  

**Link**: [PDF](https://arxiv.org/pdf/2610.04339)  

**Abstract**: ShadowMiner v1 is a system that automatically discovers research problems and generates hypotheses from AI papers. It is a nine-stage pipeline. It structures documents into a knowledge graph and finds graph gaps in it - structural blind spots in research. These graph gaps are included in the LLM generation prompt. Each generated hypothesis is then verified by checking whether it is already covered by existing research, scoring its quality, and checking that the facts it relies on are accurately drawn from its sources. This report does not propose a new generation or evaluation technique. It describes our experience of implementing and applying ideas from prior work, and measuring whether each one actually contributed. 

---
# Factorized Delayed Streams Modeling for LLM-based Streaming ASR 

**Authors**: Tatsunari Takagi, Kai Washizaki, Atsushi Kojima, Lianbo Liu, Koki Nikaido, Yui Sudo  

**Link**: [PDF](https://arxiv.org/pdf/2610.04333)  

**Abstract**: Delayed Streams Modeling (DSM) enables LLM-based streaming automatic speech recognition (ASR) by aligning acoustic and text streams on a common timeline. DSM adds the padding token <p> and the word-start token <w> to the LLM vocabulary and predicts them together with normal text tokens using the same softmax. We first show that <w> can be removed while maintaining competitive recognition performance. Based on this result, we propose Factorized DSM (F-DSM), which separates the waiting probability for <p> from the distribution over the original LLM vocabulary. This factorization removes ASR-specific tokens from the text prediction space and allows the large-vocabulary softmax to be skipped on waiting steps. Experiments on the Corpus of Spontaneous Japanese and LibriSpeech show that F-DSM achieves better recognition performance than DSM. It also greatly reduces GPU memory use while maintaining similar training throughput, provides a small inference speed improvement through softmax skipping, and reduces the degradation in text-only perplexity observed with DSM. 

---
# Suppressing Pressure, Amplifying Evidence: Self-Guided Attention Steering to Mitigate Sycophancy and Stubbornness 

**Authors**: Yinghao He, Mengyu Xu, Haixiang Sun, Donghan Li, Yibo Wang, Lixu Wang, Kezhen Chen, Chi Li, Chunwei Liu, Bharat Bhargava, Chongyang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.04329)  

**Abstract**: Reliable language models should resist unsupported user pressure while effectively using objective contextual information. However, models may exhibit sycophancy by yielding to unsupported user pressure or contextual stubbornness by failing to update their answers when relevant contextual information warrants revision. Evaluating interventions for these failures separately can obscure whether mitigating one failure exacerbates the other. To assess this trade-off, we introduce CoPE-Bench with six conditions per question: a neutral baseline, correct or incorrect user pressure, contextual information consistent with or conflicting with the neutral answer, and a joint condition combining incorrect claims with conflicting contextual information. To regulate the influence of user pressure and contextual information, we propose SPAE (Suppressing Pressure, Amplifying Evidence), a training-free framework that uses the model's own judgments to identify relevant tokens, suppressing user pressure and amplifying contextual information through token-level attention steering. On average across five backbones, SPAE reduces pressure following by 18.8 percentage points and increases joint-condition updating by 5.5 percentage points relative to the strongest baseline in the main comparison. In two-turn dialogue, it improves joint-condition updating by an average of 13.2 percentage points over the strongest prompting baseline. The source data and codes can be found at this https URL. 

---
# Bidirectional Preference Synthesis: Learning Prompt-Conditioned Preferences from Boundary Failures 

**Authors**: Junbo Wang, Lidong Lu, Zhuoqun Li, Guiping Jiang, Xiangyu Wu, Tinghai Zhang, Tong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04328)  

**Abstract**: Correction-based offline preference pipelines commonly treat model failures only as rejected responses under the original prompt. This supervision is incomplete for boundary failures: responses that violate the given instruction yet coherently satisfy a nearby intent or constraint setting. We introduce Bidirectional Preference Synthesis (BPS), a data-construction method for standard Direct Preference Optimization (DPO) that makes this missing prompt dependence explicit. For each validated boundary failure, BPS keeps the conventional forward pair under the original prompt and adds a reverse pair under a synthesized achieved prompt, so the same response is rejected where it is wrong and chosen where it is right, without changing the DPO objective, training a reward model, or requiring online sampling. On Qwen3-4B-Instruct-2507, BPS preserves original-side pairwise ranking while raising achieved-side ranking accuracy from 6.8% to 62.3% on held-out crossed anchors, with a similar shift under a Kimi-K2.6 cross-teacher probe. A blind human audit supports the intended reverse preference direction, and downstream evaluations show the clearest separation from Forward-DPO in multilingual multi-turn instruction following, with consistent capability-retention patterns on agentic, tool-use, and code checks. 

---
# From Latent Space to Jacobian Space: Measuring, Evading, and Training Against Safety-Content Accessibility 

**Authors**: Mohammad Mosafer  

**Link**: [PDF](https://arxiv.org/pdf/2610.04316)  

**Abstract**: Output-only safety monitoring sees only the end of a model's computation, yet the model computes its answer before emitting it: what it is internally poised to say is safety-critical. Jacobian-space (J-space) readouts, linear maps from hidden states to the output vocabulary via the model's input-output Jacobian, have been proposed as a window on behaviorally accessible internal content, and a first safety protocol (JADR) showed that danger recognition is readable there. What remains unknown is how this accessible content relates to the latent-space safety geometry studied by representation engineering, and how safety training shapes it. We introduce two quantitative bridges: (i) a transport-amplification profile $A_\ell$, measuring how strongly a latent safety direction is carried toward output space by each layer's Jacobian, and (ii) a paired base-vs-tuned protocol that attributes accessibility to training provenance. Across four small model pairs (135M to 1.5B, three lens-fit seeds), tuned checkpoints concentrate recognition in upper-middle layers, and at 0.5B DPO installs refusal while J-space recognition drops to chance: training can widen the accessibility gap exactly where behavioral safety looks best. A deployment audit closes the loop: monitor-aware GCG-style suffixes suppress the prompt-side monitor at zero behavioral cost at every scale; only a learned monitor rung resists its own adaptive re-attack; the training-time defense fails its re-attack at every penalty weight; steering shows the amplification profile is descriptive, not causal; and continuous-prefix optimization fails where discrete search succeeds. Safety monitoring, training, and evaluation must operate on accessibility itself, not on outputs alone. 

---
# Questioning the Questions: Sustaining Self-Evolution in Reasoning Models 

**Authors**: Jinyuan Li, Chengsong Huang, Langlin Huang, Donghong Cai, Shiping Gao, Yuyi Yang, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04299)  

**Abstract**: Self-evolving reasoning models learn from their own generated questions, yet repeated self-training can lead to performance collapse. In this paper, we investigate why performance deteriorates over successive rounds and how to sustain self-evolution. Our analysis identifies two recurring quality problems in self-generated questions: invalid questions and repeated variants of the same mathematical questions. First, invalid questions become more prevalent across rounds, and answer-consistency filtering further increases their proportion in training data. Second, existing question diversity controls based on lexical similarity can miss mathematically equivalent questions expressed in different ways, which leads to question diversity collapse in later training rounds. Building on these findings, we introduce R-Quest, which uses question validity and novelty feedback to guide self-evolution. We first train the solver to recognize and reject invalid questions, then use its judgments to guide questioner rewards and filter solver training data. To avoid question repetition, we use a frozen base model to compare sampled question pairs and provide novelty feedback. Empirically, our method consistently achieves the highest average performance on 12 benchmarks in mathematical reasoning, general-domain reasoning, and code generation across two model families. Additionally, R-Quest maintains stable performance gains over ten rounds of self-evolution, peaking in the final round and outperforming R-Zero by 17.32 points. 

---
# Language-Conditioned Token and Reasoning Efficiency in Large Language Models: A Paired Cross-Lingual Study Protocol 

**Authors**: Genliang Zhu, Chu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.04295)  

**Abstract**: Large language models incur language-dependent representation and inference costs, but existing comparisons often conflate input language, assigned observable-trace language, and answer realization. We specify a prospective paired study that separates these interfaces while holding the semantic item, checkpoint, and answer oracle fixed. The initial design instantiates 240 exactly scored items rendered from templates in English and seven non-English languages, three distinct-lineage open-weight checkpoints, three trace-token budgets, 22 input- and trace-language conditions, a fixed answer reserve, and a separately counted delimiter: 47,520 initial core runs before prospective sample-size selection. RQ1-RQ3 estimate input and trace effects by intention-to-treat with failure-inclusive terminal accounting and test answer realization by cloning a sealed prefix and runtime-native KV state into eight crossed branches. Pre-freeze independent language review, fixed-form ASCII selectors, and code/surface/solver agreement constrain the realization test. H1-H5 share one Holm family and a global simultaneous component band. A secondary randomized experiment compares one-long-attempt and complete K-short-attempt policies at equal trace allowance under frozen seeds and oracle-blind aggregation; it is a full-policy contrast because answer capacity differs. Outcomes include exact token spans, correctness, latency, runtime-exposed memory, and qualified same-host operating-system-reported energy over prespecified hardware rails. The protocol separates tokenizer expansion, observable-trace cost, and answer-realization cost without treating visible traces as internal cognition or operating-system estimates as physical cross-device energy. No confirmatory model outcome is reported; result fields remain disabled until the frozen evidence ledger passes independent verification. 

---
# First-Order Steering: Translating Weight Adaptation into Activation Steering 

**Authors**: Sri Pranav Kunda, Alexander Kurz, Tomas Dominik, Uri Maoz  

**Link**: [PDF](https://arxiv.org/pdf/2610.04283)  

**Abstract**: Activation steering exploits interpretable directions in the residual stream to enable inference-time manipulation of model behavior. Composing steering vectors to apply multiple target behaviors simultaneously is important in various fields-including AI alignment and safety-but remains a challenge for existing activation steering methods. In contrast, prior work in model merging shows that target behaviors represented by learned weight adaptations can be combined with high accuracy. A method that translates weight adaptations into activation steering vectors could therefore extend prior work in model merging to generate composable steering vectors that better enable simultaneous inference-time behavioral control. For this, we introduce First-Order Steering, a formulation of activation steering as a first-order approximation of weight update matrices parameterized by a vector of steering strengths, and establish theoretical bounds on the approximation error of first-order steering. We then develop a novel model merging procedure, HeRD-Merging, which minimizes the first-order approximation error terms to enable higher first-order steering accuracy. Together, our method produces steering vectors that control both individual and composed behaviors more accurately than existing activation steering methods. Furthermore, HeRD-Merging matches the performance of conventional model-merging baselines, while producing weight adaptations that admit more accurate first-order steering vectors. 

---
# Rethinking Self-Distillation for Multi-Teacher Capability Merging 

**Authors**: Roy Xie, Dan Friedman, Feng Nan, Yukun Huang, Zhichao Xu, Chengjiu Zhang, Jun Xu, Manaal Faruqui, Vivek Rathod, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2610.04272)  

**Abstract**: Combining capabilities of multiple expert models trained starting from the same base checkpoint has become increasingly common in frontier language-model post-training. Recent trends suggest that multi-teacher on-policy distillation (MOPD) outperforms conventional off-policy methods. However, despite the higher inference and environment interaction costs incurred by MOPD, we find that much of its reported accuracy gain is due to certain training design choices and hyperparameter optimization, rather than the algorithm itself. We conduct a controlled self-distillation study across two multi-teacher settings, four models, and eleven benchmarks, comparing off-policy methods, namely supervised fine-tuning (SFT) and soft-label distillation, to hybrid teacher-prefix distillation and MOPD. We found that all four methods achieve \textit{nearly identical} accuracy. However, MOPD uses $14.8$--$23.1\times$ SFT's training GPU-hours. We also revisit four recently published comparisons between on-policy and off-policy distillation and find that the reported on-policy gains shrink substantially when SFT baselines are trained on rejection-sampled teacher trajectories and use independently tuned hyperparameters. As a training-free alternative, we also find that simple weight-merging methods can recover expert capabilities with minutes of CPU merging time, although their accuracy degrades as model size decreases and task interference increases. Overall, our results question recent gains reported due to MOPD and suggest careful tuning of more efficient off-policy baselines as a viable alternative. 

---
# Multimodal Dual-Encoder Retrieval for Automated ICD Coding 

**Authors**: Abhinav Bohra, Anuj Bohra  

**Link**: [PDF](https://arxiv.org/pdf/2610.04263)  

**Abstract**: Accurate International Classification of Diseases (ICD) coding is crucial for large-scale clinical research, documentation, and billing. There are three primary problems with current ICD prediction methods: (1) They are unable to comprehend multimodal patient data because they rely on either structured EHR data or unstructured clinical notes. (2) They also struggle with scalability to a larger amount of ICD codes (9K+ codes in ICD-9), as traditional classifiers need dense output layers and often do not generalize well to long tail rare diseases. (3) They lack transparency for clinical use. To address these challenges, this research proposes a two-stage framework that first retrieves ICD codes using a multimodal dual-encoder retrieval model, where structured and unstructured patient data are integrated through gated fusion. The second stage refines the top-k retrieved candidates with an LLM-based re-ranker that provides ranked codes with clinically relevant explanations. Our experiments show that the proposed approach improves Micro-F1 and Precision over a multimodal dual-fusion classifier baseline. These improvements demonstrate that combining a gated multimodal retrieval system with LLM-based re-ranking is a practical alternative to dense multi-label classification for automated ICD coding. 

---
# Clean: Second-order LLM Training at Linear Memory Cost via Nyström Sketching 

**Authors**: Beheshteh T. Rakhshan, Sahar Rajabi, Maziar Sargordi Shikai Fang, Guillaume Rabusseau, Sirisha Rambhatla  

**Link**: [PDF](https://arxiv.org/pdf/2610.04204)  

**Abstract**: Training large language models (LLMs) entails a fundamental trade-off: memory-efficient optimizers such as Adam discard cross-parameter curvature, whereas full-curvature methods such as SOAP can accelerate convergence at prohibitive memory costs. We introduce Clean, a memory-efficient and full-curvature optimizer designed to resolve this bottleneck. Clean leverages the randomized Nystrom method to accurately approximate the left and right preconditioners in SOAP, and to reduce the optimizer's memory complexity from quadratic to linear in terms of model dimensions. We subsequently reintegrate the off-subspace components to capture curvature information beyond the low-rank approximation, preserving rich curvature at minimal memory cost. We further propose Q-Clean, a low-precision variant that aggressively compresses optimizer states. Q-Clean reduces optimizer memory consumption by \textbf{over 50\%} compared to Muon when pre-training a LLaMA-1.3B architecture, all while maintaining strong and competitive predictive performance. Notably, Clean operates with a smaller optimizer-state footprint than standard AdamW while reaching AdamW's final performance \textbf{26\% faster} in wall-clock time. Furthermore, our methods uniquely enable the pre-training of a 13B-parameter model on a single 80GB GPU, providing a scalable, efficient, and accessible approach to large-scale model optimization. 

---
# Language Model Activations Inhabit Privileged Error-Correcting Basins 

**Authors**: Matthew Finlayson, Francisco Pernice, Eric Todd, Amir Zur, Daniel Wurgaft, Fenil R. Doshi, Vasudev Shyam, Matt Feiszli, Satchel Grant, Lucius Bushnaq, Tal Haklay, Usha Bhalla, Matthew Kowal, Thomas Fel, Jack Merullo, Atticus Geiger, Xiang Ren, Owen Lewis, Ekdeep Singh Lubana  

**Link**: [PDF](https://arxiv.org/pdf/2610.04183)  

**Abstract**: Language models exhibit remarkable robustness, continuing to produce coherent text even when their activations are perturbed by interventions like linear steering. We hypothesize that this robustness is a result of passive dynamics, i.e., constraining mechanisms in the forward pass that funnel activations toward "good" regions that produce coherent outputs. To investigate these hypothesized error-correcting mechanisms, we probe the geometry of language model activation space by observing the action of model layers on low-dimensional curves. In doing so, we discover that model activations occur within a cluster of distinct attracting basins, which differentiate natural activations geometrically from distributionally similar synthetic activations. Applying this lens to language model steering, we observe feature-specific basins along semantic steering directions, and find that steering moves activations between these basins. To demonstrate the active role of this geometry in neural computation, we show that adaptively modulating steering strength to transport activations across basins improves inter-language steering, significantly increasing the probability of sampling tokens from the target language compared to fixed-strength steering. Our findings establish analysis of activation space geometry as a promising approach to interpreting and controlling language models. 

---
# DimSteer: Steering LLM Authoring with Automatically Discovered Stylistic Controls 

**Authors**: Ajit Mallavarapu, Ziwei Gu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04174)  

**Abstract**: Large language model writing interfaces often make users steer outputs by repeatedly articulating desired changes in natural language. Yet writers may recognize useful stylistic directions only after seeing alternatives, making revision recall-heavy. We present DimSteer, an authoring interface that samples prompt-local completions, discovers high-variance activation-space axes of variation, labels them, and exposes them as sliders with pole previews, diff comparison, and reset controls. Users can manipulate discovered dimensions, reducing the need to reformulate prompts for each stylistic adjustment. In a within-subjects study with 16 participants against a matched prompt-only baseline, DimSteer reduced mental demand, effort, and frustration while preserving comparable perceived success. Participants valued the surfaced dimensions, yet 15 of 16 disagreed that they would have thought to request the same changes in a prompt. Results suggest prompt-local controls can shift LLM authoring from recall-based prompting toward recognition-based exploration and direct manipulation, while preserving prompting for open-ended edits. 

---
# Principled Top-$k$ Selection for Language Models with Hybrid Gradients 

**Authors**: Xuchen Gong, Junfei Sun, Tian Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.04162)  

**Abstract**: Selecting the best $k$ items out of $m$ candidates is a critical component of modern large language model systems, such as document selection in Retrieval-Augmented Generation (RAG) and expert routing in Mixture-of-Experts (MoEs). However, training these selection modules remains challenging due to weak gradient signals and suboptimal exploration-exploitation tradeoffs. Furthermore, prior works often rely on heuristics, lacking principled objectives and approaches that explicitly model and solve the top-$k$ selection problem. In this work, we propose a principled objective for training selection modules, whose gradient naturally provides richer training signals in a hybrid form---containing a supervised-gradient component and a policy-gradient component. We show that the selection problem becomes harder as $m$ increases, and our algorithm converges at rate $O(1/\sqrt{T})$, with the optimal upper bound achieved by balancing between bias and variance. Practically, we apply our method to a set of tasks involving top-$k$ selection, including synthetic regression problems, RAG, and MoE systems, showing that our method outperforms the baselines in next-token prediction perplexity and QA accuracy. 

---
# ExpertMuon-Compass: Alignment-Guided Step Sizes for Mixture-of-Experts Training 

**Authors**: Omatharv Bharat Vaidya, Ashwin Vinod, Pedram Akbarian, Aditya Sai Ellendula, Connor T. Jerzak, Nhat Ho  

**Link**: [PDF](https://arxiv.org/pdf/2610.04140)  

**Abstract**: Mixture-of-experts (MoE) language models send each token to a few experts, so each expert is trained on a different part of the data, and this part changes during training. With a shared learning rate, Muon applies updates of roughly the same size to expert matrices of the same shape, even when an expert's update is poorly aligned with its current gradient. We here propose ExpertMuon-Compass (Compass), which multiplies the Muon step of each expert by two factors. A family factor compares the cosine between the expert's orthogonalized update and its gradient with the same cosine for the other experts in its layer. A scalar radius aggregates the alignment between corresponding rows of the update and gradient into one step-length multiplier. Compass keeps the update direction and the momentum buffer of Muon. In pretraining on FineWeb-Edu, Compass with Nesterov momentum on all matrices performs as well as or better than Muon, NorMuon, and other optimizers, with weight decay matched to NorMuon in the longer runs. Adding its factors to NorMuon gives the same or a lower loss than NorMuon. Compass is the most effective when the data seen by each expert varies during training, for example, as when the languages of a multilingual corpus arrive in separate blocks. With Compass, the expert load stays balanced, and the router assigns tokens to experts more decisively. We also prove a perturbation bound for the two factors. 

---
# InvestigationWorlds: An Agentic Environment for Legal Investigation 

**Authors**: Albert Yu Sun, Andrew Benard, Sil Hamilton, Anna Teresita A. Marcelo, Yong Jae Kim, Carl-Leander Henneking, Rundong Hu, Yuhong Wang, David Mimno, Bishan Yang, Igor Labutov  

**Link**: [PDF](https://arxiv.org/pdf/2610.04129)  

**Abstract**: We introduce InvestigationWorlds, an agentic environment for legal investigation. We build on an underused artifact of U.S. civil litigation: the summary judgment motion. This motion relies upon a record composed of real evidence exhibits, and results in a court-adopted hypothesis that is treated as ground truth for the purposes of deciding the motion. Each environment is built from a real U.S. Federal Court case retrieved from Public Access to Court Electronic Records (PACER) and augmented by an attorney-validated generation pipeline that synthesizes role-tagged documents around the original record. The resulting corpus admits multiple coherent factual readings, only one of which matches the court-adopted hypothesis. Evaluating on 100 cases, we find agents often commit to incorrect hypotheses despite retrieving relevant evidence, struggling to distinguish the court-adopted hypothesis from alternative hypotheses. 

---
# What Gradients Add to Text Leakage in Split Language Models, Counted per Token and per Document 

**Authors**: Georgios Politis, Evangelos Pappas  

**Link**: [PDF](https://arxiv.org/pdf/2610.04128)  

**Abstract**: Split learning lets a client train a language model on a server without sending its text. The client runs the first layers itself and sends the server only their output, a vector of numbers for each token. During training, the server sends gradients back. We show that an observer at the split can rebuild most of the client's text from this traffic, and we measure how much the gradients help. On GPT-2, an attacker who holds only the publicly released weights of the client's layers recovers 94.20% of tokens from the activations alone and 97.38% when it also sees the gradients, 3.17 percentage points more 95% interval [2.72, 3.64]. Counted by document, the difference is much larger. The attacker rebuilds 13.71% of 32-token documents exactly without the gradients and 37.77% with them, because a document only counts when every token is right. How we count also changes how good a defence looks. Secret mixup, which blends each outgoing vector with a decoy, stops the attacker from rebuilding almost any document exactly, yet the attacker still recovers 83-91% of tokens. In a second experiment on GPT-2 and Qwen3-0.6B, where the server trains only a run of consecutive layers, the layer at which the run starts changes both model quality and leakage, even when the run's length is fixed. We recommend reporting leakage both per token and per document, and treating what a split model sends as being as sensitive as the text itself. 

---
# SEER: Self-Evolving Event Reasoning and Retrieval for Time Series Forecasting 

**Authors**: Mingtian Tan, Palash Goyal, Mihir Parmar, Sarkar Snigdha Sarathi Das, Chun-Liang Li, Nanyun Peng, Thomas Hartvigsen, Jinsung Yoon, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2610.04109)  

**Abstract**: Real-world time series are frequently driven by exogenous events and structural shifts, rendering conventional forecasting based solely on historical numerical observations insufficient. While language models can retrieve external news, standard retrieval-augmented approaches struggle with high noise, missing signals, and an inability to reason causally about event impacts. We propose SEER (Self-Evolving Event Reasoning and Retrieval), a closed-loop framework that dynamically optimizes event conditioning for time series forecasting. SEER translates prediction errors into two decoupled feedback mechanisms: (i) a reflective retrieval memory that refines subsequent search queries and filters spurious noise, and (ii) a persistent causal knowledge base that distills transferable domain dynamics. SEER enforces strict chronological boundaries across both event retrieval and reflection, preventing look-ahead bias and data leakage. Across six volatile time-series benchmarks, SEER consistently outperforms state-of-the-art time series foundation models and language model baselines. 

---
# Slaying the Hydra: Interaction-Aware Circuit Discovery in Language Models 

**Authors**: Sankaran Vaidyanathan, Rafal Urbaniak, Emily Bunnapradist, Michelangelo Naim, Daniel Waxman  

**Link**: [PDF](https://arxiv.org/pdf/2610.04017)  

**Abstract**: Localizing behavior to individual components of a language model is a central goal of mechanistic interpretability. However, scoring components one at a time misses context-dependent effects: a primary component can inhibit the activation of a backup, leading to issues with ranking components. Actual causality studies the structure of such interactions via witnesses: variables that provide contextual information to resolve interaction terms. However, estimation with witnesses typically requires combinatorial enumeration and is infeasible in practice. We introduce the witness-integrated set effect (WISE), a family of causal estimands that build on the witness mechanism while taking expectations over sets of causes and witnesses to remain computationally feasible. Building on this approach, we introduce JuntaLearner, a gradient-based circuit discovery method that learns to rank components by their causal impact across varying-sized sets of components and witnesses. Alongside faithfulness metrics, we introduce measures of necessity and task specificity, and the circuit recognition score (CRS) to summarize each metric across circuit sizes while emphasizing effects achieved by small circuits. Across tasks and models of increasing size, JuntaLearner achieves higher mean CRS compared to attribution baselines on all metrics. Since its cost does not grow with the number of candidate components, JuntaLearner scales to large models while accounting for set-level interactions and avoiding first-order approximations. 

---
# Behavioral History Outperforms Descriptions of the Person for LLM Synthetic Personas 

**Authors**: Khashayar Pourtaheri, Ahmad Zareei  

**Link**: [PDF](https://arxiv.org/pdf/2610.03998)  

**Abstract**: Large language models (LLMs) are increasingly used as synthetic personas representing survey respondents. Their validity as substitutes for particular respondents depends on whether they reproduce individuals' decisions. We examine what information helps synthetic respondents predict each individual's later choices, using five conditions that add progressively richer information: no personal information, demographics, personality traits, cognitive scores, and finally the respondent's earlier survey choices as behavioral history. We use a two-wave panel of 845 US adults who completed measures of 14 behavioral biases (spanning risk, time preferences, overconfidence, and reasoning), so each respondent's earlier answers provide a human test-retest benchmark; in the behavioral-history condition, all items that score the target bias are withheld. At the population level, the average number of biases per respondent in every condition is close to the human average (7.1-8.1 biases, against 7.1 for humans). This aggregate similarity masks differences in variance: persona descriptions recover only 53-67% of human between-person variation, whereas adding behavioral history restores it to approximately the human level. At the individual level, description-based personas achieve only 7-12% of the informedness observed in human test-retest responses, while adding behavioral history raises this to 28%. The condition including behavioral history has the highest estimated informedness in all 17 demographic groups, whereas description-based conditions provide little or no information for some groups. Synthetic responses also exhibit stronger education- and income-related differences than human responses. For LLM synthetic personas, a respondent's past answers add more to individual-level prediction than a description of who they are. 

---
# Proxy Confidence: Auditing Black-Box LLM Agents with a Surrogate's Log-Probabilities 

**Authors**: Yikai Zhao, Saurabh Pandey, Pradeep Kumar Misra  

**Link**: [PDF](https://arxiv.org/pdf/2610.03894)  

**Abstract**: A deployed LLM agent emits tool calls, queries, and code that can be silently wrong -- by the time the error surfaces, the action has run. Frontier chat APIs hide the model's token probabilities; the agent's stated confidence barely beats chance on the mistakes that matter; and resampling does not help, since frontier models are highly repetitive, reproducing the same call across samples.
We recover the missing signal from a low-cost open-weight surrogate run in parallel. It reads the same context, schema, and proposed action as the agent, then scores the call from its own log-probabilities through a family of complementary readouts: teacher forcing and request-PMI weigh the likelihood of each argument value, a discriminative verdict judges the call as a whole, and tool-choice competition tests the function against its siblings. One principle says which to trust: a generative likelihood localizes wrong argument values, while the verdict catches holistically wrong calls. When the error type is unknown, an ensemble is the low-regret default. The readout is training-free, needs no access to the agent's internals, and costs one prefill pass alongside the tool call.
On difficult coding tasks it reaches AUROC 0.825 where the actor's stated confidence is near chance (0.598), and the generative readouts beat it by +0.07 to +0.28 across three further actors. Against self-consistency it gains +0.14 to +0.19 on near-deterministic actors, at 1/K the cost. The signal drives two deployment modes: a real-time gate escalating the least-trustworthy calls for review (+0.05 to +0.30 accepted-action accuracy at 50% coverage), and confidence feedback, returning the tool result with the score so the agent adapts its next step -- lifting task success on live-execution benchmarks (+0.119 and +0.137, p <= 1e-4) and beating a random-value control where step errors are silent (+0.078, p = 0.003). 

---
# Training Numerical Intelligence via Auto-Diagnosis and Skill Discovery 

**Authors**: Peter Chen, Wotao Yin  

**Link**: [PDF](https://arxiv.org/pdf/2610.03872)  

**Abstract**: AI agents are becoming increasingly capable of generating scientific code, but generating code is not the same as improving the algorithms behind it. For numerical solvers, execution feedback can expose poor performance, but rarely reveals its underlying cause and how to address it. We introduce Auto-Diagnosis and Skill Discovery (ADSD), a framework that links numerical diagnosis to reusable solver self-improvement. ADSD follows a diagnosis-first paradigm that first explains why a solver performs poorly, then uses this diagnosis to guide the discovery of appropriate numerical methods. The resulting knowledge is packaged into reusable solver skills, turning solver improvement from trial-and-error editing into a structured process of diagnosis, discovery, and implementation. Across four challenging numerical domains--power flow equation, AC optimal power flow control, stiff ordinary differential equations, and heterogeneous diffusion PDEs--ADSD consistently improves solver accuracy, robustness, and efficiency. On GOC-500 power flow, for example, ADSD reduces mean solver error by nearly $71\times$, with improvements further transferring to unseen grid topologies and operating regimes. 

---
# Do Motion Tokenizers for Co-Speech Gesture Generation Encode Gesture Semantics? 

**Authors**: Varsha Suresh, Divij Jain, Jia Liu, M. Hamza Mughal, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2610.03765)  

**Abstract**: Discrete motion tokenizers encode motion as atomic units and are widely used for co-speech gesture generation. It remains unclear which motion properties, especially those relevant to gesture semantics, are recoverable from these codebooks. We probe a reconstruction-trained codebook using 19 co-speech gesture descriptors spanning from raw motion to abstract communicative function. Results show that geometry and handedness are readily decodable from token embeddings, while motion category is only weakly decoded despite showing systematic differences in discrete code usage. This gap between reconstruction quality and descriptor decodability suggests that reconstruction objectives alone do not guarantee that gesture semantics are captured, and that evaluating codebooks on such properties can guide the design of more semantic motion tokenizers. 

---
