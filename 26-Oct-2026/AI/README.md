# SemantEq: Faithful Bidirectional Equivalence of Formal Statements 

**Authors**: Xinrui Li, Pawan Sasanka Ammanamanchi, Siddharth Bhat, Rianna Jamdar, Yongxi Lin, Stella Biderman  

**Link**: [PDF](https://arxiv.org/pdf/2610.32919)  

**Abstract**: Statement autoformalization translates mathematics from natural language into a proof assistant such as Lean 4, and models now do it at scale. As provers get stronger, the burden of correctness shifts from the proof to the statement, that is, to whether the formal statement faithfully translates its source. Given a trusted reference formalization, the natural test is logical equivalence, and it proves too much. $P \leftrightarrow Q$ holds trivially for any two statements $P$ and $Q$ that the library proves, because a prover can prove each separately and pass through $P \leftrightarrow \mathrm{True} \leftrightarrow Q$. The stronger the prover, the more unrelated pairs pass. Earlier methods restrict the prover and check that each proof names the other statement. We introduce SemantEq, a more principled relation that instead requires each statement to turn into the other only through a chain of allowed steps, such as logical rearrangement and rewriting with the library's own equations. Our programs judge a proof by the proof term Lean checks, not by its tactic script. SemantEq-Eval, a search without a model, reaches 100% precision and 65.6% recall on the ProofNetVerif test split under our adjudicated labels, against 95.8% and 61.8% for BEq+, and an 8.9-fold increase in its CPU time adds at most three pairs. A language model finds more proofs, but under two of four prompts it proves both directions of about two in five of the non-equivalent pairs. SemantEq-ProofAudit audits each model-written proof under a policy derived from the relation and accepts none of the 104 non-equivalent pairs, and it raises recall on 407 equivalent pairs from 65.4% to 81.1% under the best prompt. Separating the generation of equivalence proofs from their acceptance lets a stronger prover raise recall without changing what an acceptance means. We release code and data at: this https URL 

---
# Closing the Modality Gap via Simplex-Constrained Representations 

**Authors**: Shubham Gupta, Siva Reddy, Perouz Taslakian, Valentina Zantedeschi, Cem Subakan  

**Link**: [PDF](https://arxiv.org/pdf/2610.32890)  

**Abstract**: While multimodal contrastive models like CLIP and CLAP successfully learn shared representations, their embeddings typically exhibit a persistent \emph{modality gap}. We propose addressing this separation through a purely geometric approach: changing the embedding domain from the unit sphere, $S^{d-1}$, to the probability simplex, $\Delta^{d-1}$. Although both manifolds share the same intrinsic dimension, the simplex acts as a structural prior; its nonnegative, unit-mass constraints serve as a cross-modal calibration that forces both modalities to ground themselves in the same coordinate system. By attaching lightweight softmax adapters and using Total Variation similarity, we reduce the centroid $\ell_2$ gap by 97--99\% across five multimodal retrieval benchmarks. This confirms that simplicial geometry induces genuine semantic alignment, matching or exceeding the retrieval performance of standard unconstrained models. 

---
# NgramRL: Accelerating RL Post-Training for Structured Generation via Rollout-Aware N-gram Speculative Decoding 

**Authors**: Nilesh Prasad Pandey, Jangseon Park, Jason Kong, Onat Gungor, Tajana Rosing  

**Link**: [PDF](https://arxiv.org/pdf/2610.32881)  

**Abstract**: Reinforcement learning (RL) post-training improves the reasoning capabilities of large language models, but its cost is dominated by autoregressive rollout generation, which can exceed 70% of training time. Existing speculative-decoding methods are poorly suited to this setting: model-based drafters consume scarce GPU memory and may become misaligned as the policy evolves, while existing model-free methods do not explicitly exploit repetition across training epochs. We observe that recent rollouts of the same prompt provide a high-quality, self-updating draft source. Based on this insight, we propose NgramRL, a per-prompt n-gram drafter constructed from bounded recent rollout history. NgramRL requires no auxiliary model, drafter training, or persistent GPU-resident drafter state, while target-model verification preserves the rollout distribution. To hide CPU cache lookup and scheduling, we further introduce an adaptive fan-out prefetcher that predicts likely verification outcomes (j, b), consisting of the accepted draft length j and target-emitted continuation token b. A causal budget allocator concentrates prefetches on high-probability outcomes, while uncovered outcomes fall back to sequential lookup without changing committed tokens. Across two model families and three reasoning and structured-generation workloads, NgramRL commits 3.60-7.77 tokens per target-model forward in the warm-cache regime, up to 1.99x more than SuffixDecoding and 7.0x more than prompt-lookup decoding. On Spider, the prefetcher covers 96.6% of Qwen3-8B outcomes at a candidate budget of 16 and improves measured rollout throughput by 1.38x over synchronous NgramRL, approaching its 1.40x overlap ceiling. 

---
# Advancing SVD-based LLM Compression via Layer-Wise Error Model Search 

**Authors**: Moritz Thoma, Maximilian Groezinger, Maximilian Forstenhäusler, Emad Aghajanzadeh, Manoj Rohit Vemparala, Christos Anagnostopoulos, Pierpaolo Mori, Nael Fasfous, Alexander Frickenstein, Daniel Mueller-Gritschneder, Ulf Schlichtmann  

**Link**: [PDF](https://arxiv.org/pdf/2610.32878)  

**Abstract**: Low-rank SVD-based compression offers a powerful strategy to reduce the computational costs of LLMs. However, existing methods face two key limitations: (i) global rank allocation, where uncalibrated error proxies fail to capture complex error propagation, and (ii) decomposition quality, where Fisher-based estimators suffer from severe rank collapse. In this work, we address these limitations by introducing Layer-wise Error Modeling Search (LEMS) and KFAC-SVD. LEMS advances rank allocation by introducing a layer-wise error surrogate that integrates local and global layer importance alongside a propagation bias, enabling effective global rank allocation via an ILP formulation. KFAC-SVD improves decomposition quality by utilizing token-wise statistics, mitigating the rank deficiency observed in prior Fisher-based SVD approaches. Across Mistral, Qwen3, and Llama 3 model families, we show that LEMS consistently outperforms existing search strategies, delivering significant zero-shot accuracy gains of up to 4.8 p.p. that generalize to model sizes of 70B parameters, while KFAC-SVD achieves an average perplexity improvement of 15%. Project page and code: this https URL 

---
# Do "True" and "False" Add Up to One? Complement Consistency in LLM-Agent Opinion Simulations 

**Authors**: Anouar Imel, Jin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2610.32874)  

**Abstract**: In LLM-agent opinion simulations, each agent reports the probability that a claim is true after seeing its own previous probability and the probabilities reported by its neighbors, and that number becomes its state in the next round. If the number describes the agent's position rather than the sentence it was shown, then rewriting the claim as its complement and replacing every probability p with 1 - p should return the complementary answer, so the two reports should sum to one. We test this with three small open models: Qwen3-4B, Llama-3.2-3B and Gemma-3-4B. On 2,304 agent states taken from a running network simulation, Qwen's paired reports sum to 1.18 on average, and on 48 real fact-checking claims from AVeriTeC they again exceed one. A controlled benchmark of 24 worlds, each with exactly two possible outcomes, shows the same failure in all three models. The discrepancy is two to ten times larger than the change produced by a same-meaning paraphrase, and it persists under repeated sampling and across peer group sizes and report strengths. When the reports are fed back into two populations that start from complementary states, the populations do not remain mirror images: their divergence is positive after five rounds for every model. Averaging each answer with the complement of its paired answer restores consistency by construction and lowers Brier loss averaged over the two wordings, but at the final round it beats the better of the two original wordings for only one of the three models. A schedule-by-targeting experiment on the same model gives a small, stable interaction because such sweeps keep the claim wording fixed. Consistency under rewording therefore needs its own test before simulated probabilities are read as beliefs 

---
# From Deliberately Constructed Systemic Representations of the World of Activity to Dialogue: A Five-Level Communication Architecture for Strong Language-Based ASI 

**Authors**: Maxim Osovsky  

**Link**: [PDF](https://arxiv.org/pdf/2610.32869)  

**Abstract**: This paper proposes a five-level organization of message formation and understanding within thinking and activity in which ASI participates. Its central thesis is that a language model's context is fundamentally incapable of presenting the foundations of utterance formation for examination and verification. These foundations are established in advance by acquired schemas, systemic representations of the world of activity, and connections accepted as true within them. Neither context nor trained LLM weights, in themselves, attest to the presence of deliberately constructed systemic representations. The five levels distinguish systemic representations of the world of activity, a particular situation cut out from those representations, its narrative description, the full sense of a message, and text. A situation's schema-model is not identical to a story about it. A recipient reconstructs the full sense from an abbreviated text through jointly constructed and agreed-upon representations and joint activity. Reverse work proceeds from the full sense through an extended description and the situation's schema-model to the reconstruction of systemic foundations. The theoretical foundations draw on the systems methodology of G. P. Shchedrovitskiy, the structural differential of A. Korzybski, and the Meaning-Text theory of I. A. Mel'chuk. The contribution is a specific systemic-activity organization of content for examining and adjusting message foundations and controlling the consequences of their transformation. Further research concerns large graphical models of activity on reconstructible systemic foundations, distinct from text generation from context. 

---
# A Foundation Model for Policy In-Context Longitudinal Causal Effect Estimation 

**Authors**: Wenxin Chen, Kyra Gan, Mingyu Lu, Zilin Jing, Fei Wang, Hoifung Poon, Tristan Naumann, Peniel Argaw  

**Link**: [PDF](https://arxiv.org/pdf/2610.32863)  

**Abstract**: Longitudinal causal effect estimation is complicated by time-varying confounding and treatment-confounder feedback loops. Existing methods generally require data-specific nuisance-model fitting, and many are tailored to prespecified treatment sequences or prediction horizons, leading to time-consuming computation at deployment. We introduce *Policy In-Context Longitudinal causal Effect estimation with Prior-data Fitted Networks* (**PICLE-PFN**), the first foundation model for longitudinal causal effect estimation that supports *dynamic* treatment policies. PICLE-PFN represents target policies through in-context history-decision demonstrations, providing a unified interface for a broad class of treatment policies. Pretrained over synthetic data sampled from longitudinal structural causal models, PICLE-PFN amortizes causal effect estimation across datasets, policies, and horizons. Given a new observational dataset and target policy, it directly predicts conditional average potential outcomes (CAPOs) in one forward pass without task-specific training or hyperparameter tuning. Across synthetic and semi-synthetic longitudinal benchmarks, PICLE-PFN achieves competitive RMSE and substantially improved uncertainty calibration with a fraction of deployment runtime, compared to state-of-the-art conventional estimators trained and tuned separately for each dataset. 

---
# Reimagining Enterprise RAG for Multi-Agent Systems: A Knowledge Layer That Organizes Before It Retrieves 

**Authors**: Garima Agrawal, Prasun Das, Akshay Sarvade, Priyanka L, Minisha N, Sravani Joshna, Hemath Manivannan, Lavanya Korada, Sambhrit Pattanayak, Srashti Gupta, Siddu R, Prasad Kalyanasundaram  

**Link**: [PDF](https://arxiv.org/pdf/2610.32851)  

**Abstract**: Enterprises are deploying agentic AI across the software development lifecycle, and they need the behavioral logic of the systems they act on, not stakeholder documentation. A seven-agent reverse-engineering pipeline recovered this logic, processing 241 repositories across 18 services into thousands of artifacts describing scope, journeys, rules and acceptance criteria. Recovery does not confer usability. Entities like offer, eligibility, and activation recur across services under a shared vocabulary, related in name but operationally distinct, so one end-to-end journey implicates over a thousand near-identical rules that must not be conflated. Flat vector retrieval fails, returning an identical rule from the wrong service at high similarity with no signal of error. Content-level ontology, the traditional remedy, did not exist, and relating every occurrence of recurring entities would have cost more than the engagement and not survived a release cycle. We place the ontology at the artifact level. In this machine-generated corpus, each artifact retains its producer, component, service, domain and type, provenance that disambiguates before content is read. We present a knowledge layer that organizes before retrieving: a self-describing catalog, a sparse metadata registry, and a navigation hypergraph defined over artifacts rather than extracted entities. Queries resolve to artifact identifiers that constrain vector search, so the competing rule from another service is absent rather than outranked, making disambiguation structural, not statistical. Built in under a week and admitting new services by convention, the layer serves SDLC agents, developers, quality engineers and analysts. It supported root-cause analysis and resolution of 100+ production defects with developer verification and attributed 35 release regression failures to specific rules and services. 

---
# HarnessMM: Recursive Self-Improvement for Harness Token Efficiency 

**Authors**: Yuhua Jiang, Chang Xu, Feifei Gao, Li Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.32813)  

**Abstract**: Agent harnesses repeatedly invoke reasoning after tool feedback, even when an earlier call has already fixed the next action. We introduce HarnessMM (Harness Minus Minus), a method for improving harness token efficiency through guarded decision reuse and recursive self-improvement (RSI). RSI iteratively refines companion harness configurations from training feedback, with validation selecting the configuration for deployment. A controller at the model-tool boundary reuses resolved decisions to replace later reasoning callbacks while actor weights remain fixed. Executable Decision Graphs store exact actions, literal arguments, causal dependencies, and public guards. Guarded Decision Reuse activates these persistent decisions from observed receipts; Certified Callback Substitution executes them through the native interface and records removal of the registered slow callback. New semantic decisions, unsupported evidence, and selector abstention invoke the original actor again. The representation restricts what feedback may change, while complete-task measurements establish quality and cost. On HumanEval, HarnessMM preserves Codex success at 90.82\% while reducing Tokens/task (k) from 66.72 to 47.02 (29.54\% saving). For DeepSeek Harness, success rises from 71.43\% to 85.71\% while Tokens/task (k) fall from 96.28 to 11.62 (87.93\% saving). A separate MBPP ablation measures 26.91\% lower cost than single-use graphs and 19.52\% lower cost than retained callbacks. These results establish guarded decision reuse as a practical way to reduce repeated reasoning in existing harnesses. 

---
# ActionRAG: Evidence-Refining Graph Diffusion for Long-Term Memory in LLM Agents 

**Authors**: Navyansh Mahla, Sashank Vandrangi  

**Link**: [PDF](https://arxiv.org/pdf/2610.32752)  

**Abstract**: LLM agents need long-term memory that can recover precise and mutually consistent evidence from growing interaction histories, particularly when answers depend on relations among facts rather than isolated semantic matches. Current memory pipelines remain vulnerable at two stages: indexing errors can fragment repeated entities or merge distinct ones, and online retrieval often treats initial similarity scores as final even when the surrounding graph provides evidence for revising them. We introduce ActionRAG, a graph memory architecture that couples cautious indexing with query-time graph retroevaluation. Its indexing pipeline reconciles type labels and applies tiered entity resolution. Given a query, dual-channel entity and triple retrieval forms an initial evidence set and induces a bounded local graph. Query-conditioned receiver-normalized diffusion refines triple scores using local path structure. Seed protection, hub-penalized flux, and triple materialization preserve direct evidence while recovering missed connections. We evaluate ActionRAG on MemoryAgentBench (MAB), which tests accurate retrieval, conflict resolution, long-range understanding, and test-time learning. Across its eight deterministic-scoring columns, ActionRAG achieves 55.1 average accuracy, improving over the strongest retrieval-based memory baseline by 5.4 points. Under MAB's query-time cost accounting, mean synthesis cost stays below \$0.001 per query on the four reported cost columns, and the sum of separately measured retrieval-side p99 components is 440.7 ms on our consumer Apple Silicon setup. 

---
# Agentic Hazard Mapper: Emulating Expert Map-Making with AI Agents for Risk Communication 

**Authors**: Sidi Wu, Lifeng Lin, Yizi Chen, Anna Yokokubo, Lorenz Hurni  

**Link**: [PDF](https://arxiv.org/pdf/2610.32738)  

**Abstract**: Hazard maps are essential tools for public risk communication, preparedness, and evacuation planning, yet effective cartographic design is not always ensured due to limited access to cartographic expertise across local governments and communities. We present Agentic Hazard Mapper, a multi-agent system that automates hazard-map production by emulating professional cartographic workflows and iterative design trade-offs while adapting design decisions to different data, locations, and hazard contexts. We conducted a formative study with map experts to inform the system design, followed by a user study and interviews with map experts and hazard practitioners. The generated maps improved participants' spatial understanding and evacuation-planning performance, reduced perceived workload and task completion time, and were generally preferred over corresponding municipal maps. We further discuss human oversight, adaptation across users and communication contexts, and opportunities to extend agentic support across broader stages of hazard-map production and risk communication. 

---
# Retiming Robot Demonstrations with a Learned Speed Prior 

**Authors**: Seungyong Moon, Deokjae Lee, Hansang Lee, Munkhtselmeg Battsengel, Youngin Kim, Seonghyeon Yun, Sungjin Jeon, SongEun Kim, Jeongmin Park, Hyun Oh Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.32676)  

**Abstract**: The pace of a robot demonstration is often determined by the teleoperator rather than by the task itself. As a result, the same motion can appear at widely different speeds across demonstrations, and policies trained on such data may inherit operator-specific timing along with task behavior. We propose to learn a speed prior directly from large-scale demonstration data. Given a task instruction, a motion path and the images along it, it predicts a distribution over execution speed along the path, from which a chosen quantile defines a consistent clock for retiming each demonstration. Existing methods derive the acceleration signal from the very data they accelerate, so each new task costs another fit, annotation, or pair of predefined rates. A single trained speed prior instead covers every task in a corpus. We fit a speed prior on both robot teleoperation data and handheld demonstrations in which a human arm, rather than a robot arm, moves the gripper, and use it to retime robot demonstrations to the handheld pace before fine-tuning a vision-language-action policy. For execution, we pair the policy with a real-time optimal-control layer that tracks predicted action chunks under robot constraints using the predicted pace. On 120 RoboLab tasks, policies fine-tuned on our learned clocks succeed more often and move with up to half the jerk compared with policies fine-tuned on the original demonstration timing and uniformly accelerated to the same speed. On a real robot, our policy stacks cups in 3.50s, 1.72$\times$ faster than the default policy's 6.02s while succeeding as often, in 29 of 30 trials. Videos and further material are available at this https URL. 

---
# Do Language Models Trust Their Own Justifications? A Study on Functional Consistency 

**Authors**: Alisson Rosa Pereira, Bruno Iochins Grisci  

**Link**: [PDF](https://arxiv.org/pdf/2610.32665)  

**Abstract**: Large Language Models (LLMs) have been widely adopted in text classification tasks, where they not only output class predictions but also generate explanations that highlight the tokens deemed most relevant for reaching the predicted label. Yet it remains unclear whether these highlighted elements are behaviorally consistent with the model's own predictions under targeted interventions, which we refer to as functional auto-consistency. While much of the literature evaluates the textual plausibility of such explanations, few studies assess their functional consistency with the model's actual behavior. In this work, we propose an experimental framework based on the principle of auto-consistency: if a model identifies certain tokens as decisive, then isolating, removing, or semantically inverting them should produce systematic and interpretable changes in its predictions. We operationalize this evaluation through sufficiency, comprehensiveness, and counterfactuality metrics, and conduct experiments on IMDB and Steam reviews across both closed-source (GPT-4o-mini) and open-source LLMs (Gemma3, Granite8B, DeepSeek). Results show that GPT-4o-mini follows the expected progression across all metrics, Gemma3 and Granite8B maintain coherence under sufficiency but lose consistency under more demanding interventions, while DeepSeek variants display structural deviations, either failing to preserve sufficiency or overreacting under comprehensiveness and counterfactuality. These findings show that explanation reliability varies across LLM families and scales, with smaller models displaying contradictions and larger ones exhibiting over-sensitivity. By combining sufficiency, comprehensiveness, and counterfactuality, our approach provides a systematic methodology for assessing the functional consistency of LLM self-explanations. 

---
# WiMoT: Replay-Free Continual Wi-Fi Sensing via Moment Transport 

**Authors**: Chunchun Yang, Yang Cao, Zhitang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.32660)  

**Abstract**: Continual Wi-Fi sensing must balance plasticity, learning new rooms or users, against stability, keeping earlier ones, yet nodes often cannot retain channel state information (CSI) for privacy and storage reasons. We propose WiMoT (Wi-Fi Moment Transport), which stores the per-class feature mean and covariance of every earlier domain instead of samples. When the representation changes, stored statistics become stale. WiMoT restricts the change to an affine map on a frozen feature basis and applies the same map to the statistics, so they stay valid without old samples. Supervised contrastive learning learns the new domain, a closed-form loss on the stored statistics keeps earlier classes separable, and a Gaussian discriminant classifier, refitted from the transported statistics with equal weight per domain, predicts without domain identity. On 3DO, WiMANS, and Widar3.0, WiMoT's average final accuracies are 94.02%, 89.61%, and 95.16% with at most 0.58 percentage points (pp) of forgetting. On 3DO and Widar3.0, it also has the highest balance of new- and earlier-domain accuracy and is 3.77 and 13.11 pp ahead of the strongest of 21 baselines, which uses replay, staying ahead across domain orders and label budgets. On WiMANS, it matches the strongest replay baseline in accuracy (+0.41 pp) while reducing forgetting from 2.78 to 0.00 pp, and has the highest six-order average accuracy. Because full covariances dominate the memory, storing them as rank-32 factors shrinks the moment memory to 6.4%, losing at most 0.49 pp of accuracy, within edge-node storage budgets. 

---
# LGMT: A Tool for Testing the Logical Reasoning Reliability of LLMs 

**Authors**: Zenghui Zhou, Xiaoke Fang, Zheng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.32627)  

**Abstract**: Large Language Models (LLMs) increasingly serve as reasoning components in modern software systems, yet a model that answers a logical query correctly can contradict itself when the same problem is restated in an equivalent form. Static benchmarks evaluate queries in isolation and therefore fail to expose such inconsistencies. To address this limitation, we present LGMT, an open-source tool that detects reasoning inconsistencies through metamorphic relations grounded in first-order logic. Given a source test case in premise-conclusion form, LGMT generates logically equivalent follow-up cases and checks whether the model's entailment predictions agree across the pair, without ground-truth labels. The tool provides 20 built-in metamorphic relations, supports custom ones verified by the Z3 theorem prover, and runs from a CLI or a web studio, including an offline mock mode requiring no API key. Across six widely used LLMs on a bundled benchmark of 3,091 metamorphic groups, LGMT reveals inconsistency rates ranging from 20.84% to 29.44%. LGMT is publicly available on GitHub at this https URL, with a demonstration video at this https URL. 

---
# What Transfers Across Protein Pockets? Experience Reuse in Molecular Optimization for Agentic Structure-Based Drug Design 

**Authors**: Zaifei Yang, Weiyu Chen, Yaqing Wang, James Kwok  

**Link**: [PDF](https://arxiv.org/pdf/2610.32620)  

**Abstract**: LLM agents optimize a ligand in a protein pocket by editing it step by step and docking each edit. Recent agents first spend extra dockings probing the complex with controlled edits; guidance induced from these probes is reported to help more than the chemical knowledge the model already carries. The probes are then discarded with the instance that produced them, so the obvious next step is to keep them in an experience base, letting a new pocket inherit what earlier ones already paid for. Such a base is read through a \emph{key}: the record properties under which past edits are pooled and a new one is looked up. Using the full log of one such run over the CrossDocked2020 test-set pockets, we ask whether its keys recur on a pocket the base has never seen and still separate improving from worsening edits. \crchange{Using} the contacts at the site being edited, the action, and the round as \crchange{a} key, the base separates them sharply. Within our analysis, this separation is largely explained by how much the edit changed the molecule's size, the action, and the round, none of which describes the target. That is not because the target is irrelevant: naming it outright separates them several times more sharply. For a pocket absent from the base, transfer must rely on a description rather than its identity. Among the exact-match descriptions we evaluate, the more specific ones rarely recur, whereas the coarser ones provide little separation; the signals that do transfer primarily describe the edit and the molecule it produced rather than the target. Under the representations and lookup scheme studied here, an experience base may be more useful for prioritizing an agent's own proposals for docking than for directly prescribing what to build. 

---
# LLaCO: Continuous Diffusion for Large Language Models with Competition-Aware Semantic Scheduling 

**Authors**: Zijian Jia, Peixuan Zhang, Genying Li, Di Wu, Yujie Fu, Si Li, Boxin Shi, Shuchen Weng  

**Link**: [PDF](https://arxiv.org/pdf/2610.32604)  

**Abstract**: Continuous diffusion models have emerged as a promising non-autoregressive paradigm for text generation over continuous embedding spaces. Along the continuous trajectory, the residual token identity directly shapes the denoising objective. However, the geometric trajectory misaligns with temporal progression, triggering severe temporal compression. In this paper, we show that embedding norm decay causes Euclidean distance to yield spurious overtaking, indicating that token competition must be evaluated directionally. In this space, a candidate overtakes the source token only when the effective noise magnitude surpasses their semantic separation. We formalize these pairwise dynamics as the Competition Threshold Theory (CTT). Crucially, high-dimensional overtaking requires extreme noise levels, causing standard trajectories to concentrate identity decay into a narrow window near the noisy endpoint. To resolve this, we propose Semantic Retention Scheduling (SRS) to align time allocation with the progression of identity decay, and demonstrate that existing token identity metrics are fundamentally statistical aggregations of these pairwise crossings. Building on these theoretical insights, we introduce LLaCO, a generative framework powered by competition-aware semantic scheduling. LLaCO adapts a pretrained backbone to continuous representations via an embedding directional adapter and timestep-conditioned modulation. To operate within the directional space, LLaCO employs a spherical noising process formulated as a geodesic interpolation toward a mask-biased endpoint, eliminating norm decay while preserving accurate token competition. Guided by SRS, LLaCO reallocates the denoising trajectory proportionally to the logarithmic rank decay. Instantiated as an 8B model, LLaCO outperforms previous continuous diffusion baselines and rivals discrete diffusion models of comparable scale. 

---
# Beyond Channel Data: Configuration-Enriched Radio Foundation Models 

**Authors**: Shrihari Vasudevan, H. G. Ranjani, Raaghul Ranganathan, Satwik Sahu, Thirumaran Ekambaram  

**Link**: [PDF](https://arxiv.org/pdf/2610.32542)  

**Abstract**: This paper proposes incorporating channel configuration (CFG) information into Radio Foundation Model (RFM) pre-training and demonstrates that the resulting CFG-enriched embeddings serve as effective inductive priors for downstream tasks under label scarcity. We compare the proposed embeddings with raw channel tokens and multiple baseline models, under three different data regimes, for multiple downstream tasks --- Line-of-Sight/Non-Line-of-Sight (LoS/NLoS) classification, timing advance (TA) estimation, and beam selection. Evaluation is done in both in-distribution and out-of-distribution configuration settings. Experimental outcomes show that the proposed embeddings consistently outperform and/or are competitive to the baselines, across experiments in both seen and unseen configuration scenarios. 

---
# IGG: A Benchmark for Interactive GUI Grounding under Visibility Constraints 

**Authors**: Kyeongseon Kim, Jiyeon Son, Tae-Hyun Oh  

**Link**: [PDF](https://arxiv.org/pdf/2610.32536)  

**Abstract**: GUI grounding benchmarks ask whether an agent can localize a target that is already visible in a static screenshot. Real interfaces often violate this assumption: a target may be off-screen, appear only after a state change or hover, remain occluded, or become actionable after a delay. This leaves the ability to recover target visibility unevaluated. To measure this reveal step, we introduce Interactive GUI Grounding (IGG), a benchmark in which an agent must first make the target visible through interaction and then click it. IGG contains 601 tasks across 16 controllable GUI environments and seven visibility-constraint sub-types, with a compact action space of scrolling, moving, dragging, waiting, and clicking. In our human study, participants achieve 94.6% accuracy, while GPT-5.5 reaches 79.4% and the strongest evaluated open-source VLM reaches only 31.6%. To identify the source of this gap, we conduct two controlled interventions. Telling Qwen3-VL which interaction to perform raises its accuracy only from 30.8% to 33.9%, whereas performing the interaction for the model raises it to 81.9%. The bottleneck is executing the reveal interaction, not recognizing the target once it is exposed. We further show that this capability is learnable, demonstrating that visibility recovery can be strengthened consistently across model families. 

---
# When Agents Fail: Model or Harness--Who's to Blame? 

**Authors**: Shao Zhang, Hangfan Zhang, Yuanyuan Shi, Hao Li, Min-Le Su, Kangcong Li, Jingping Wang, Chunjiang Mu, Chen Zhang, Yang Chen, Bo Zhang, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.32483)  

**Abstract**: LLM agents combine models with execution harnesses, so agentic benchmark performance reflects their joint behavior. We argue that harnesses should therefore be evaluated through their interactions with underlying models. We develop a framework combining five complementary measures: (i) Valid Completion Rate (VCR), how reliably agents finish normally and deliver evaluable results, irrespective of task correctness; (ii) Model-Harness Fit, how well a harness supports task achievement through valid completion for a given model; (iii) Cross-Model Fit, a harness's overall suitability across evaluated models; (iv) Interaction Residuals, pairing-specific execution strengths and weaknesses hidden by component-level summaries; and (v) Token Load, the token demands of agent execution, including failed attempts. We evaluate four general-purpose harnesses (OpenCode, Codex CLI, Pi CLI, and DeepAgents) with four core models, plus two fixed native model-harness systems, on six benchmarks covering context management, tool use, long-context planning, multimodal interaction, and artifact handoff. Including native references and stress conditions, 111 evaluation settings yield 52,168 trajectories. Harness rankings vary across models and benchmarks; task performance and valid completion can favor different pairings, and similar task scores can accompany distinct execution profiles. OpenCode ranks first in cross-model performance, while DeepSeek V4 Flash Preview maintains consistently high Mean VCR across harnesses on the shared four-benchmark panel. These findings support evaluating model-harness interactions, training transferable harness-use capabilities in models, and harness designs adapted to the model and task. 

---
# CASE: Confusion-Aware Skill Evolution for Frozen Multimodal Emotion Models 

**Authors**: Siyu Wu, Zeyu Liang, Linghao Wang, Qiyun Zheng, Yi Yao, Ling Lo, Hongxia Xie  

**Link**: [PDF](https://arxiv.org/pdf/2610.32481)  

**Abstract**: Multimodal large language models (MLLMs) have substantially advanced Multimodal Emotion Recognition (MER) by enabling fine-grained emotion understanding through reasoning over multimodal evidence. However, they still exhibit systematic confusion between semantically adjacent emotions with highly overlapping multimodal cues. Existing approaches typically address these errors through global parameter adaptation, making confusion correction costly and inflexible. We instead observe that emotion confusion is highly localized and therefore better addressed through reusable pair-specific expertise. Building on this insight, we propose Confusion-Aware Skill Evolution (CASE), the first agent-skill framework for MER. CASE maintains a sparse, self-evolving library of pair-specific EmotionSkills that externalize affective expertise from model parameters, enabling the learned expertise to be continuously refined and readily transferred across diverse frozen MLLM backbones. During inference, CASE dynamically retrieves the appropriate EmotionSkill to reinforce boundary-aware reasoning while keeping the MLLM backbone frozen. Extensive experiments on CA-MER, DFEW, and EMER demonstrate the effectiveness and transferability of CASE across both general-purpose and emotion-specialized MLLMs. Notably, CASE improves the average accuracy of Qwen3-VL-8B on the CA-MER benchmark from 46.80% to 59.13%, while achieving consistent gains across MLLMs. 

---
# CauProphet: Teaching Agents to Develop Causal Cognition like Humans 

**Authors**: Kaiwen Wu, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.32477)  

**Abstract**: Large language model (LLM) agents increasingly support long-horizon tasks by leveraging external memory and past experience. However, existing agents mainly retrieve records or summaries, lacking the ability to form and refine causal understanding of the world. Inspired by human causal cognition, we introduce CauProphet, an agent that develops and updates causal knowledge within its Causal Cognition. For each task, CauProphet combines relevant knowledge in Causal Cognition with current evidence to construct a Causal Belief, a structured belief capturing causal factors and the effects of possible actions. It uses this belief for decision-making and continuously updates both the belief and Causal Cognition as new observations emerge, enabling causal knowledge that persists beyond individual experiences. Evaluations demonstrate that CauProphet progressively acquires causal knowledge and transfers it to future decision-making. In financial research, CauProphet leverages prior discoveries to improve downstream analyses and outperforms all baselines in mean rank correlation with next-day returns, achieving a 25.4% improvement over a baseline agent. It also improves prediction accuracy from 21.3% to 28.6% in dynamic real-world settings. Further experiments validate its ability to perform reliable causal analysis and action selection across scientific and everyday tasks, demonstrating the potential of causal experience learning for next-generation LLM agents. 

---
# Beyond Post-Hoc Rewards: Privileged Reasoning Exploration for Multimodal Embeddings 

**Authors**: Tianyu Yang, Yijun Tian, Lars Benedikt Kaesberg, Jan Philip Wahle, Bela Gipp, Terry Ruas, Daniel Kurzawe  

**Link**: [PDF](https://arxiv.org/pdf/2610.32474)  

**Abstract**: Advances in Multimodal Large Language Models (MLLMs) have driven substantial progress in multimodal embeddings. Recent methods exploit generative capabilities by augmenting embedded inputs with self-generated reasoning to produce more informative embeddings. However, conventional retrieval supervision for training embedding models specifies pairwise relevance relations without explicitly guiding the reasoning used for input augmentation. Such reasoning is neither directly observed nor uniquely determined by the input, allowing multiple plausible traces to introduce semantic cues of varying retrieval utility. Existing approaches largely optimize reasoning conditioned on the input alone, limiting the exploration of retrieval-relevant traces. We introduce PILOT, a framework that models reasoning as a latent variable and instantiates its prior and posterior distributions as distinct reasoning modes of a shared MLLM. The prior conditions only on the input, whereas the posterior uses additional training-time relational evidence to guide reasoning exploration. To translate the posterior's broader exploration into inference-time improvements, we derive an importance-weighted variational objective and instantiate it as a prior-posterior co-evolution framework that improves both modes under a shared inference-compatible reward while selectively transferring posterior trajectories to the prior according to their utility and prior reachability. On the 78-task MMEB-V2 benchmark, PILOT outperforms the primary baseline across all 12 task groups at both 2B and 7B parameters, improving overall scores by 3.2 and 3.3 points, respectively, with the largest gains in visual-document retrieval (+6.1 and +6.4 points). Our results show that relational supervision is most valuable not as a post-hoc reward but as a signal for shaping how reasoning is explored in training reasoning-enhanced embedding models. 

---
# ALLMTS: Automated LLM-based Medical Triage System 

**Authors**: Michał Remigiusz Janiszewski, Artur Stopa, Łukasz Lenkiewicz, Wojciech Achtelik, Bartosz Gonczarek  

**Link**: [PDF](https://arxiv.org/pdf/2610.32469)  

**Abstract**: Telephone triage is a safety critical task: from a brief intake note, a nurse must decide whether a caller needs an ambulance, an emergency department, an appointment, or self care at home. Queried directly, large language models (LLMs) are unreliable: they over-escalate routine cases and cannot justify their recommendations against an auditable clinical standard. We present the Automated LLM-based Medical Triage System (ALLMTS), which grounds every disposition in the same commercially maintained telehealth guidelines nurses use. Rather than asking an LLM to triage, ALLMTS makes it execute the guideline: it selects the appropriate guideline and evaluates its triage assessment questions. The disposition follows deterministically from the first positive answer, using the same traversal rule a nurse follows. We evaluate ALLMTS on 566 synthetic, expert-validated scenarios spanning 14 dispositions and designed to test specific guideline questions. ALLMTS achieves 0.95 accuracy, compared with 0.47 for a raw-LLM baseline. On a curated 50-scenario subset released as an open benchmark, ALLMTS reaches 0.98 accuracy, on par with the 0.944 correct disposition rate that a five nurse expert panel recorded on those same scenarios. Critically, on that subset the baselines have over-triage rates of up to 0.50 (0.44 on the full dataset), whereas ALLMTS reduces the over-triage rate to at most 0.02 while maintaining low under-triage. These results support coupling LLM language understanding with executable clinical guidelines for triage from intake notes. 

---
# POSEIDON: Scaling Training Environments for Scientific Agents from Code Repositories 

**Authors**: Qifan Zhang, Yuhan Li, Nuo Chen, Bowen Liu, Xiao Hu, Miao Peng, Chenyi Zi, Lixin Liu, Lydell Huang, Jing Tang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.32448)  

**Abstract**: Training environments remain a major bottleneck for agents in data-driven scientific discovery. Existing collections of executable and verifiable training environments are small, and their construction remains difficult to scale. To address this limitation, we introduce Poseidon, an automated pipeline for generating diverse, verifiable training environments from scientific code repositories at scale. From the same repository budget, Poseidon extracts nearly 10 times as many training environments as a prior approach by decomposing scientific workflows into four complementary task types: Reproduction, Subflow, Completion, and Correction, where each type targets a distinct capability for scientific research. A recursive improvement loop further refines failed candidates and recovers otherwise discarded environments. Using Poseidon, we construct Pearl, a collection of 97,659 executable and verifiable environments from 7,110 public repositories spanning 10 scientific disciplines. To the best of our knowledge, Pearl is the largest and most broadly multidisciplinary collection of scientific-agent training environments to date. Qwen3.5 models (9B, 27B, and 35B-A3B) trained on Pearl achieve consistent gains across six challenging scientific-agent benchmarks. Overall, the scale and diversity of Pearl offer opportunities for training scientific discovery agents. 

---
# When to Stop Penalizing Actions? Martingale GRPO for VLA Reinforcement Learning 

**Authors**: Zongyao Zhao, Cheng-Jie Zhao, Ruihang Jiang, Bing Zheng, Zhaolin Wang, Yuanwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.32431)  

**Abstract**: A failed vision-language-action (VLA) rollout does not imply that every action contributed to failure. Yet in group-relative policy optimization (GRPO), a single trajectory-level advantage is assigned to all actions in a rollout. Potentially useful recovery attempts can thus be penalized alongside the errors they aim to correct. Ideal stopping conditions for failure feedback are characterized using success-probability and policy-score martingales. At these stops, the removed gradient contribution has zero conditional expectation given the retained information. Truncation therefore preserves the expectation of a reference GRPO group-gradient estimator and can reduce its covariance. One such stop occurs when the conditional probability of success is zero, motivating error and recovery cues for selecting candidate feedback boundaries. Martingale GRPO (MART-GRPO) is introduced with a vision-language model (VLM) for event-guided truncation. The VLM is fine-tuned on our annotated training set using low-rank adaptation (LoRA). The last visible error without observed recovery is localized by the frozen VLM in complete rollout videos. Failure feedback is retained through the selected error interval and removed afterward, while successful-rollout loss terms and original loss normalization are preserved. For retrospective boundaries, changes in expectation and variance are analyzed to assess departures from ideal stopping. Experiments with OpenVLA-OFT on RoboTwin 2.0 show that MART-GRPO improves success rates over full-trajectory GRPO across multiple tasks. 

---
# Async-RBench: Benchmarking Asynchronous Task Execution in LLM Agents 

**Authors**: Detao Zeng, Guilin Ou, Nuo Chen, Yunhao Feng, Hongliang Wang, Qinli Yang, Junming Shao, Zhong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.32429)  

**Abstract**: Large language model (LLM) agents increasingly execute tasks asynchronously: a main agent continues working while its subagents run, and the task state may change before their results arrive. The main agent must decide which work to update or preserve and which checks to repeat, while avoiding prohibited changes. Final task success alone does not reveal whether these results were handled correctly, and natural asynchronous arrivals are hard to reproduce. We introduce Async-RBench, a benchmark for asynchronous task execution in LLM agents, with 200 executable tasks covering eight scenarios. A scenario controller regulates result delivery using predefined task-state conditions, while an Event Response Contract (ERC) specifies what each response must change, preserve, avoid, and re-verify. The Dynamic Replanning Score (DRS) weights the response process and outcome equally and is reported separately from task success. We evaluate nine agents and find DRS scores of 19.2 to 45.1, with the lowest scores on duplicate reports and subagent failures. Under our execution protocol, all nine agents obtain lower task success with asynchronous than with batched delivery (26.0% vs. 29.7% on average). For three agents, prompting the main agent to rebuild its plan after every result improves DRS by only 2.6 to 3.5 points. Together, these findings highlight the challenges of reliable asynchronous task execution and the need to evaluate how agents handle incoming results alongside end-to-end task success. 

---
# LeafSkill: Efficient Leaf-Level Action Grounding for Recursive LLM Agents 

**Authors**: Shen Liu, Ze Zhou, Zhenguo Xu, Shaopu Wang, Yike Gao, Chunlei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.32412)  

**Abstract**: Recursive large language model (LLM) agents solve long-horizon tasks through decomposition, execution, and environmental feedback. Yet after producing an atomic subtask, they often invoke the same large model again to translate a familiar instruction into an executable action. We observe that high-level planning is task-specific, whereas leaf-level action grounding recurs across tasks and can be checked against environment-admitted actions. We introduce LeafSkill, an external reuse layer that retrieves semantically related grounding experience, binds action templates to current objects and admissible actions, and uses a lightweight 1.7B selector to choose a bound action or defer the unchanged subtask to the original agent. Skills remain outside the model parameters, allowing a frozen selector to use post-training additions. Under lexical and syntactic paraphrases, exact matching retrieves none of the reference actions, while semantic retrieval preserves 96.0% and 97.0% candidate coverage; the frozen selector adopts 97.34% of unseen skill types. On an independent confirmation set of 200 Robotouille tasks, LeafSkill completes 137 tasks versus 121 for ReCAP (68.5% vs. 60.5%, exact McNemar p=.0259). On the 106 tasks completed by both, it reduces Qwen3.5-27B tokens by 14.42% and calls by 14.08%. A separate held-out repeat yields a consistent 9.0-point success gain and 18.98% token reduction. Reusable leaf grounding can thus improve task completion while reducing host-model inference without replacing high-level planning. 

---
# DealsFlow: An Evidence-Gated Agentic Harness for Venture Capital Deal Triage 

**Authors**: Ming Ze Tang, David Cutting, Boyd Carson, Vasiliki Carson  

**Link**: [PDF](https://arxiv.org/pdf/2610.32380)  

**Abstract**: Large language models (LLMs) can read startup pitch decks, but letting them emit investment scores directly invites score inflation, run-to-run variance and rationales that are hard to audit. We describe DealsFlow, a human-in-the-loop agentic harness deployed at a UK early-stage fund. The LLM reads the deck, cites evidence and rates each category; code combines the ratings, caps them and only ever lowers them. Three mechanisms implement this: a stage-aware corroboration ceiling (78, 70 or 60 out of 100) on factual categories without external evidence; a sceptic call that may lower but never raise a score; and a pre-fetch pattern that retrieves and indexes all evidence before a single generation pass. We evaluate on $N=243$ SEIS/EIS investments with recorded outcomes. Because recent investments have unresolved outcomes, we use the 142 that are at least three years old. There, the full pipeline's score has Spearman $\rho=0.45$ (95% CI 0.32 to 0.57) with the total-value multiple and AUC 0.71 for identifying write-offs, against $\rho=0.38$ and AUC 0.68 for an unconstrained LLM. The gain is small and comes mostly from the sceptic call; we did not test whether its lower-only rule matters. Investments that later exceeded $2\times$ had 64% of factual categories corroborated, against 38% for write-offs (omnibus Kruskal-Wallis test over five outcome groups, $p=0.031$), and our retrieval was not date-restricted, so these figures may overstate screening skill. In a study of 20 inbound applications, analyst time per application fell from 44.7 to 18.9 minutes including verification and correction, and 84% of 240 cited claims were fully supported by their sources, but 70% of reports needed at least one correction. 

---
# Adaptive Control of Scientific Instruments with Nonstationary Responses through Multimodal Agents 

**Authors**: Guoyang Shi, Hongwen Li, Yiding Yu, Zhuang Fu, Siqi Ding, Zhengyuan Chen, JianGuo Chen, Hanyue Zhao, Tianyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.32370)  

**Abstract**: Controlling scientific instruments requires parameters to be adjusted progressively in response to feedback from successive operations. When unobserved operating conditions evolve with operational history, responses become nonstationary, and a static mapping from actions to responses becomes inadequate for decisions across shots. Shot-to-shot tuning of neutral beam injection (NBI) ion sources in magnetic confinement fusion devices is representative of this class of problems: the conditions of electrode surfaces and vacuum are difficult to observe directly, discharge outcomes vary between shots, and experiments are costly and carry operational risks. We propose a physically grounded multimodal agent with three independently auditable components. Deterministic algorithms extract symbolic physical quantities and visual evidence from multichannel waveforms; a memory with two layers organizes trajectories from the current task, experience from earlier tasks, and procedural knowledge, from which a multimodal large language model (MLLM) generates candidate actions; and an independent action feasibility check enforces action constraints before execution. In a strictly chronological replay of 455 shots from four EXL-50U platforms, physical symbols alone outperformed waveform images alone by 0.21-0.32 in macro-F1 for quality classification across all four evaluated models. Retrieved context increased the mean agreement with historical action directions by 22.4-42.7 percentage points, whereas the contribution of visual evidence varied across models. Under supervised operation on five ion sources, the system achieved a beam-extraction rate of 86.80% across 409 executed shots. These results demonstrate the engineering feasibility of the framework for supervised sequential tuning on a physical scientific instrument. 

---
# FOCUS: A Lightweight Mamba-2 Model for EEG-Based Cognitive Workload Classification 

**Authors**: Ioannis Kyprakis, Stefanos Gkikas, Chariklia Chatzaki, Zongyi Jiang, Randy Gomez, Raul Fernandez Rojas  

**Link**: [PDF](https://arxiv.org/pdf/2610.32361)  

**Abstract**: Cognitive workload estimation from physiological signals is increasingly relevant in high-risk environments, where continuous and unobtrusive monitoring can inform adaptive systems and safety interventions. This paper presents FOCUS (Frequency-band Optimized Cognitive-load Ultralight SSM), a compact Mamba-2-based model for EEG-based binary workload classification under constrained data conditions. Rather than learning frequency sensitivity from data, FOCUS encodes preprocessed multi-channel EEG using a fixed FFT decomposition that covers the five canonical EEG bands. The proposed model combines depthwise-separable convolution and a weight-shared encoder-bottleneck Mamba-2 pair, with 0.04 million parameters and 0.11 GFLOPs. Evaluated in a pilot study with 11 participants under leave-one-subject-out cross-validation across three cognitively distinct tasks, FOCUS achieves 74.75% accuracy on the full dataset at 0.08 ms GPU model-forward latency. In the weight-sharing ablation, the shared configuration achieves 4.55 percentage points higher accuracy on the full dataset than the non-shared variant while reducing the SSM parameter budget; however, this difference does not reach statistical significance under a two-sided Wilcoxon signed-rank test ($p=0.083$). Overall, the selected FOCUS configuration provides a favorable trade-off between classification performance and computational cost within this controlled pilot setting. 

---
# TGR-EvoDAG: A DAG-based Multi-Agent Code Evolution Framework for Generative Recommendation on Tencent 

**Authors**: Junlin He, Junzhang Zhu, Yudong Li, Yihong Tang, Xuefeng Jiang, Wei Ma, Chengxiang Zhuo, Zang Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.32326)  

**Abstract**: Large Language Model (LLM) agents have enabled automated code evolution for algorithm discovery. However, current recursive self-improvement (RSI) research for sequential recommendation faces a dilemma: general-purpose frameworks lack domain-specific feedback, while recommender-specific frameworks rely on linear evolution. Crucially, naively integrating them fails when applied to Generative Recommendation (GR). Unlike end-to-end models, GR relies on a module-cascaded pipeline (tokenization, pre-training, and alignment). In this paradigm, existing evolution methods suffer from: i) cost blindness, squandering budgets by ignoring stage-asymmetric retraining overheads; and ii) short-sighted evaluation, discarding promising upstream innovations that temporarily regress metrics before downstream adaptation. To bridge these gaps, we propose TGR-EvoDAG, a multi-agent code evolution framework tailored for GR. TGR-EvoDAG models code evolution as a Directed Acyclic Graph (DAG) to maintain multiple evolution branches and merge complementary innovations, unifying advanced search with rich GR diagnostic feedback. We further introduce stage-aware cost planning to optimize GPU-hour budgets, and active downstream co-adaptation to navigate temporary regressions and unlock cross-stage gains. On public Amazon datasets, TGR-EvoDAG enables the first autonomous evolution of a complete GR pipeline, improving offline metrics by up to 37.4% and Agent4Rec simulated user satisfaction by 19.7% over the strongest baseline. Furthermore, in industrial deployment on Tencent's large-scale commercial media platform, the evolved GR achieves a 12.16% relative improvement in offline Hit@1 and a statistically significant 0.40% increase in online click-through rate (CTR). 

---
# From Visual Affinity to Learning Credit in Multimodal Reinforcement Learning 

**Authors**: Yuhang Han, Dingning Liu, Yujie Chen, Yaojie Zhang, Jingwei Song, Zequn Yang, Xuanming Shang, Li Ding, Zhijiang Guo, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.32324)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) improves multimodal reasoning, but uniformly assigning a single response-level advantage to all tokens overlooks the distinct roles of visual observations and subsequent reasoning. Visual affinity alone is insufficient: predictive states may align with both image and prompt-context representations, and upweighting valid visual observations can amplify their negative-advantage updates when later reasoning fails. To address this issue, we propose \textbf{Outcome-Conditioned Visual Credit (OVC)}, a plug-in framework that separates visual-association measurement from token-level credit assignment and is adaptable to policy-optimization methods with uniform response-level advantage allocation. Relative Visual Encoding constructs a context-relative visual coordinate by contrasting each predictive state's affinity to image and prompt-context representations. Outcome-conditioned Evidence Calibration combines this coordinate with uncertainty-scaled predictive features and calibrates their contributions to outcome-discriminative evidence through mean contrasts between positive- and negative-advantage responses, scaled by regularized within-side variation. Visual association thus informs credit through observed outcome differences rather than directly determining update priority. Normalized Credit Allocation conditions the calibrated evidence on the response's advantage sign to produce positive token weights with unit global mean. These detached weights preserve advantage signs while modulating individual token-gradient contributions. OVC reuses existing rollouts without auxiliary reward models, process annotations, or inference-time modules. Instantiated through GSPO-token, OVC retains sequence-level clipping and consistently improves six-benchmark mean accuracy over GSPO across three models, with gains of 1.86-2.39 percentage points. 

---
# Beyond Gain: Measuring Skill Internalization for Agent Skill Retirement 

**Authors**: Yanze Wang, Pengfei Yao, Tianyi Sun, Hongda Chen, Yunyun Han, Jun Sun, Chuanrui Hu, Yafeng Deng  

**Link**: [PDF](https://arxiv.org/pdf/2610.32322)  

**Abstract**: LLM agents retrieve skills from public hubs that hold up to about 100,000 entries and serve many models. As models improve, a hub must find the skills a given model has internalized, meaning it can do without them, and retire them for that model, yet internalization has not been measured with known reliability. Published rules that compare runs with and without a skill retire on the gain alone. The gain is near zero both for an internalized skill and when neither the model nor the skill delivers the capability, a state we call uncovered. We instead withhold the skill, score the model on a task synthesized from it, and trust a high unaided score only when another model gains from the skill, since nearly half of high scores come from tasks no model benefits from. Under four models, a published gain rule applied with one task per skill removes over half of the uncovered skills. On SkillsBench, 28 of its 31 retirements rest on probes no run passed in full, while retiring on the unaided score costs no detectable reward. Unaided verdicts reproduce across tasks more often than the gain rule's (Cohen's kappa 0.42 against 0.23) but differ across models, so a hub must read them per model. We release 1,251 skills with tasks and item scores under four models. 

---
# Solvent: Long-Horizon Reasoning for Multi-Step Retrosynthesis 

**Authors**: Yifei Wang, Botian Wang, YiFei Wang, Jiangtao Feng, Xinbo Zhang, Yawen Ouyang, Yixin Wu, Lihao Wang, Xiaoqing Zheng, Xuanjing Huang, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.32308)  

**Abstract**: Multi-step retrosynthesis requires recursively transforming a target molecule through successive reaction steps into purchasable building blocks, and recent work has begun to draw on large language model (LLM) reasoning to tackle the task. In its latest evolution, LLM reasoning has entered the long-horizon regime: models pursue objectives across interdependent steps, interact with environments, and adapt to feedback. This shift has markedly advanced LLM capabilities, as illustrated by breakthroughs in AI-driven mathematical research. At this frontier, we introduce Solvent, an 8B LLM that successfully extends long-horizon reasoning to multi-step retrosynthesis. Through a two-round post-training curriculum, Solvent learns to interleave iterative synthesis planning, external availability verification, and adaptive route revision. Experiments show that, following a $2.1\times$ performance boost from RL, Solvent scores 61.86% across four benchmarks, surpassing mid-scale language models and prior search-based retrosynthesis systems, and rivaling frontier trillion-parameter LLMs. 

---
# DMAIC-Bench: From Self-Reflection to Self-Restraint for Agents in Multi-Stage Analytical Tasks 

**Authors**: Shan Peng, Shuaiqi Cheng, Miao Peng, Chenyi Zi, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.32305)  

**Abstract**: Multi-stage analytical tasks require LLM agents to use intermediate evidence to decide whether to continue, redirect or stop. Self-reflection, in which an agent critiques its output, is widely studied, yet a critique does not show whether evidence changed the next action. We call that capability self-restraint. Existing agent benchmarks largely measure task completion or output quality and leave it untested. DMAIC-Bench evaluates it with 97 seeded tasks across 15 industrial cases in four sectors, built around Lean Six Sigma's explicit statistical prerequisites. Paired counterfactual variants require opposite actions at the same workflow position, distinguishing warranted restraint from indiscriminate stopping. We assess execution traces with a private scorer and final claims through independent expert review. Across 13 models from six providers, execution-level routing accuracy averages 92% on matched GO decisions and 19% on NO-GO decisions, counting any forbidden downstream computation as a route error. Execution traces show agents reporting correct diagnostics before performing analyses that those results disallow. Separately, blinded expert review of the 11 audited agents estimates that reports act on invalid evidence at 47% of reached NO-GO decisions, with an interval from 31% to 61%. Explicit stopping permission helps some models without consistently restoring correct decisions at both GO and NO-GO forks. Sixteen life-sciences tasks in bulk ribonucleic acid sequencing (RNA-seq) and case-control genome-wide association studies (GWAS) show that the same framework can evaluate evidence-dependent decisions beyond industrial workflows. Project code and data are available at: this https URL. 

---
# Lifelong Context Adaptation through Bidirectional Memory-Skill Learning 

**Authors**: Tianyi Xu, Huazheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.32295)  

**Abstract**: Lifelong LLM agents must learn from an expanding history of interactions while making each decision within a bounded context window. Memory makes past experience available, but the agent may still need to infer how to solve each new task from the retrieved records. Skills provide reusable procedures, yet may omit the task-specific facts needed to apply them. Because retrieved memory and skills shape the agent's behavior, they also influence the experience available for future learning. We introduce MemCryst, a framework for lifelong context adaptation through bidirectional learning of memory and skills with a frozen language model. Recurring episodes stored in memory support skill creation and revision, and the interactions guided by those skills supply new memory for subsequent learning. Facts retrieved from memory and learned skills jointly form the agent's context, while task outcomes adjust the future retrieval influence of the records actually served without rewriting their content. Across ten settings spanning embodied control, long-document reading, conversation, mathematics, and code, MemCryst leads seven reported comparisons. On ALFWorld, joint serving achieves 63.17% success, compared with 59.83% for memory-only and 50.50% for skill-only serving. Code is available at this https URL 

---
# Trial by Trace: Benchmarking AI Agent Harnesses for Reliability and Failure Attribution 

**Authors**: Zhiyong Cao, Hengjie Zhu, Yongxin Wang, Xuchen Li, Jintao Li, Shuxia Liu, Dunqiang Liu, Bo Wang, Kairan Zhang, Cong Lin, JiangDi Yu, Siyuan Liu, Sijia Yao, Yi-Liang Wei, Lingxu Ma, Aijia Wang, Anxin Chen, Zixian Wei, Xiang Ruan, Ruizhi Xu, Ruqian Shi, Baihua Xiao, Sicheng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.32285)  

**Abstract**: LLM agents are increasingly used to solve real-world problems, and their effectiveness depends on both the underlying models and the harnesses that support their operation. However, most existing benchmarks focus on model capabilities and overall agent performance, with limited attention to the evaluation of agent harnesses. To address this gap, we introduce TraceTrial-Bench, which organizes harness reliability into seven components covering context, execution, governance, lifecycle, observability, tools, and verification. We construct 95 expert-level, mechanism-oriented tasks with executable or state-based ground truth to support semantically equivalent comparisons while preserving the native behavior of each harness. We design a Trajectory Tracing (TT) module to capture agent execution traces and introduce a two-layer evaluation framework that separately assesses agent performance and harness reliability. In addition, we design a Multi-source Evidence Acquisition (MEA) module to provide evidence from the agent context, harness, and execution environment for LLM-based evaluation, and develop a Failure Attribution (FA) module that uses explicit mechanism chains to support attribution at the run level. Across 4 harnesses, 9 models, and 95 tasks, our evaluation of 3,420 runs yields macro-averaged harness scores ranging from 59.8% to 72.0%, while 31.1% of runs achieve task success despite failing at least one harness criterion. TraceTrial-Bench provides an auditable and reproducible foundation for comparing agent harnesses, diagnosing their failures, and studying how harness design interacts with model behavior. 

---
# MedCaseAgent: From Multimodal Clinical Evidence to Rare Disease Case Reports 

**Authors**: Hongkang Chu, Yifan Shen, Tianjiao Yu, Zhuoqing Zhong, Jianguo Cao, Meihuan Huang, Xu Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.32258)  

**Abstract**: Although Large Language Models (LLMs) have demonstrated potential in clinical documentation, generating the rigorous, evidence-based reasoning required for unprecedented, atypical, and rare disease case reports remains a significant challenge. We present MedCaseAgent, an end-to-end agentic reasoning paradigm that bridges this gap by synthesizing publication-quality reports from unstructured clinical notes. MedCaseAgent leverages a hierarchical structure, integrating patient-level data mining with iterative, tool-integrated reasoning. By dynamically querying medical literature and knowledge bases, the agent maintains high clinical fidelity and assesses case novelty. The framework is inherently modular, supporting various backbone LLMs in a plug-and-play fashion. To facilitate this research, we release MedCase-150K, a large-scale multimodal clinical case-report corpus, and MedCase-Bench, a robust temporally held-out evaluation suite. We also introduce an open-source model post-trained from Qwen3.5-9B and optimized specifically for this task. Under both LLM-judged and rule-based evaluation, MedCaseAgent outperforms existing general-purpose and specialized medical LLMs and agents. Blinded expert reviews and ablations further support the system's clinical relevance and evidence grounding. 

---
# RSIHub: A Composable Framework for Agent-Evolution Methods 

**Authors**: Zimu Wang, Xiaobo Wang, Siyu Ye, Jiaqi Li, Silin Chen, Yuling Shi, Ruobing Wang, Jingjing Zhang, Chaofan Wang, Kai Wu, Mengnan Qi, Kai Cai, Xiaodong Gu, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.32253)  

**Abstract**: Agent-evolution methods use execution feedback to improve prompts, skills, tools, and orchestration, but their implementations often rebuild the same experimental machinery. We present RSIHub, an open-source framework for building, comparing, and extending these methods on shared infrastructure. Recipes compose reusable operators and declare mutable resources; RSIHub supplies execution, evaluation, recovery, and lineage. Researchers can focus on evolution policy while evaluation stays outside the mutable surface and results remain tied to candidates and evaluation conditions. We instantiate AHE, Hyperagents, A-Evolve, and GEPA without modifying the framework core, keeping method-specific logic in recipes and operators. We evaluate them with MiniSWE and Codex on Terminal Bench 2 and $\tau^3$ Banking under fixed model and evaluator configurations. The largest observed overall pass@1 gain is 13.5 percentage points, while method rankings vary across settings and optimization gains do not always transfer to held-out tasks. RSIHub provides a common protocol for comparing evolution policies, inspectable candidate histories, and reusable components for developing new methods. 

---
# Minimum Description Length Unsupervisedly Separates Law-Grade from Statistical-Grade Knowledge in a Self-Improving Science Agent - and Reveals 86% of Its Own Discoveries Are Statistical 

**Authors**: Wenbin Mao, Pei Zou  

**Link**: [PDF](https://arxiv.org/pdf/2610.32245)  

**Abstract**: AI systems that discover scientific relations now generate candidate "laws" at scale, but the bottleneck has shifted from generation to vetting: which candidates deserve belief? We show that a deployed self-improving science agent - one that has admitted 252 relations through held-out prediction courts - already computes the quantity that answers this question: the Minimum Description Length compression ratio used for model selection doubles, when read under discrete accounting, as an unsupervised epistemic grader. The compression-ranked top band of the agent's ledger recovers exactly the canonical laws among its discoveries (vis-viva across seven planets, Balmer/Rydberg) plus cross-measurement identities - enrichment $p=3.6\times10^{-12}$ under a hypergeometric test - with zero counterexamples and physically coherent band structure. The same lens retroactively reveals that 86% of the ledger compresses nothing ($\mathrm{cr}\approx 1$): the agent had been recording statistical regularities, including a learned predictor with held-out $R^2=0.541$ admitted through the prediction gate, under a narrative of compression as the unified value of knowledge - a narrative that was never true for them. We formalize the resulting dual-axis bookkeeping - prediction value ($R^2$) and compression value (cr) as nearly decoupled axes of knowledge - demonstrate its robustness to quantization, and register forward falsifiable predictions. The episode is, to our knowledge, the first measured instance of a discovery agent auditing its own knowledge ledger by an internal measure and finding itself over-claiming. 

---
# Spatial Heuristic Strategy based Diffusion Reinforcement Learning for Online 3D Bin Packing 

**Authors**: Tong Li, Qingyang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.32241)  

**Abstract**: Traditional heuristic methods have accumulated mature experience in core objectives, such as space utilization and layout compactness of 3D bin packing, relying on artificially designed rule-based logic. However, they exhibit obvious limitations in generalization ability and adaptability to dynamic scenarios. Diffusion reinforcement learning, which integrates the spatial search capability of reinforcement learning and the data distribution modeling capability of diffusion learning, has achieved certain progress in action distribution modeling and decision compliance constraints. Nevertheless, there remain issues including insufficient perception of the spatial topological structure in 3D bin packing and limited layout optimization capability. To address these problems, we propose a diffusion reinforcement learning method integrated with spatial heuristic strategies, which combines classical bin packing heuristic rules with a diffusion reinforcement learning framework. Specifically, it compresses the action space through heuristic action pre-screening, introduces a graph neural network to model the spatial structure, and designs a composite reward function incorporating flatness to suppress spatial fragmentation, thereby realizing the organic integration of prior knowledge and adaptive optimization. Experimental results demonstrate that the pro 

---
# Financial Context-Rich Strategy for Actor-Critic based Agents: A Deep Reinforcement Learning Approach for Stock Trading 

**Authors**: Hrishikesh Singh, Aarti Sharma, Millie Pant  

**Link**: [PDF](https://arxiv.org/pdf/2610.32237)  

**Abstract**: In modern financial markets, more than 3/4th of the large volume trade is executed by algorithmic trading based on advanced trading strategies to attain higher precision and speed than traditional human traders. The conventional strategies focused on price-based indicators are inadequate in capturing the market dynamism arising from real-world events such as social sentiments, political instability, or sudden economic shifts. With the rise of Deep Reinforcement Learning (DRL) and data availability, there is a requirement for more efficient algorithms that can incorporate these external factors in trading strategies. DRL has shown potential in simulating real-world scenarios and in simplifying complex decision-making since DRL agents can automate the trading process i.e. executing trades in a manner comparable to human traders. They have shown reliable performance since they can update their policies based on risk and quantitative factors. In this paper, we propose a DRL approach based on partial and deep-infusion of sentimental analysis. It features distinct policy designs of actor-critic-based agents: only policy optimization (A2C & PPO) and policy optimization along with Q-learning principles (DDPG, TD3 & SAC). Our findings reveal, that the agents based on policy optimization combined with Q-learning, significantly and consistently demonstrate improved performance upon sentiment deep-infusion in profit maximization as well as risk-averse trading. 

---
# Training an MoE LLM in MXFP4: How We Closed the Gap to BF16 

**Authors**: Han Wang, Zhitao Wang, Natalia Frumkin, Xuanwu Yin, Alireza Khodamoradi, Dong Li, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2610.32229)  

**Abstract**: Training large language models (LLMs) in sub-8-bit arithmetic promises substantial gains in memory bandwidth and compute throughput, yet FP4 formats introduce quantization error severe enough to destabilize optimization if left unaddressed. We present ALTO, an open-source training recipe for GPT-OSS-20B, a 20-billion-parameter mixture-of-experts (MoE) model, on the MLPerf Small MoE Training benchmark under the MXFP4 microscaling format. The recipe combines established techniques from Abecassis, Felix et al. [arXiv:2509.25149] (hybrid 1D/2D block quantization, randomized Hadamard transforms, and stochastic rounding) with a simplified weight de-oscillation scheme adapted from TetraJet-v2 [arXiv:2510.27527]. On C4 validation at global batch size 16, MXFP4 + RHT + SR + de-oscillation reaches a validation loss of 3.3350 at 16,128 steps, only 0.007 above the BF16 baseline (3.3283), closing roughly 50% of the gap left by MXFP4 without de-oscillation (3.3418). Measured by steps to reach the MLPerf quality target (validation loss 3.34), de-oscillation cuts the FP4 convergence overhead relative to BF16 from +20.0% to +5.0%. We also report negative end-to-end results from differential gradient estimation, outlier clipping, and macro-block scaling, despite each showing operator-level SNR gains. All experiments use simulated MXFP4 kernels on AMD MI300 hardware and wall-clock speed has not been optimized. 

---
# Correct Answers, Closed Gates: SolveBind for GUI Agents at the Verification Barrier 

**Authors**: Xinhao Song, Haoxuan Peng, Su Su, Hongliang Wu, Sirui Song, Gongshen Liu, Linfeng Zhang, Dongrui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.32223)  

**Abstract**: Websites and applications deploy CAPTCHAs and anti-automation mechanisms that use behavioral signals such as interaction trajectories and timing, not outcomes alone, to distinguish humans from automated agents. We call these protected GUI environments, where correctly solving a verification task is insufficient for end-to-end task completion. Agents must turn solver outputs into accepted interactions and adapt execution to runtime feedback under time constraints. We introduce SolveBind, a GUI-agent framework combining hierarchical solving with runtime supervision. A hierarchical solving tree decomposes verification into interdependent Perceive, Ground, Solve, Act, and Verify capabilities. A runtime supervisor grounds action commitments in tool-derived evidence, revises them as evidence arrives, and mediates execution under explicit recovery budgets. Across nine CAPTCHA task families and nine backbone models, SolveBind achieves the highest success rate on eight backbones, outperforming the strongest baseline by 13.3-34.5 percentage points. It achieves the highest success rate on all six AndroidWorld backbones. In a paired GPT-5.5 analysis, MobileAgent produces correct answers but fails dynamic verification on 13 of 90 tasks; SolveBind succeeds on seven of these tasks. With automatic submission, removing binding reduces success from 82.22% to 71.56% and increases latency and token usage. Recovery caps reduce execution cost with a small observed success trade-off: removing them raises success from 82.22% to 83.56%, while increasing latency and token usage to 3.7x and 2.4x their full-supervision values, respectively. These results support evidence-grounded execution control while identifying a success-cost trade-off in recovery and failure boundaries of heuristic submission rules. 

---
# RxnExBench: Benchmarking Vision-Language Models for Structured Chemical Reaction Extraction from Scientific Literature 

**Authors**: Jiaxing Sun, Junyuan Gao, Ruikang Zhang, Hao Zheng, Lingli Ge, Xiao An, Jiang Wu, Huazheng Zeng, Jiahe Song, Chen Zhu, Haote Yang, Jingchao Wang, Lu Chen, Yuanyuan Cao, Ruijie Zhang, Shengxin Lu, Weijia Li, Lijun Wu, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2610.32221)  

**Abstract**: Screen Tables and Substrate Scopes are information-dense figures widely used in synthetic chemistry literature, yet automated extraction of structured reaction data from them remains challenging. We introduce RxnExBench, a multimodal benchmark for fine-grained reaction-level evaluation. Built from high-impact chemistry journals, it contains 3,565 expert-annotated reactions and 2,361 molecular structures. The dataset adopts a unified schema with 15 reaction roles and 26 core fields, supporting downstream AI-for-Chemistry tasks such as retrosynthetic planning, yield prediction, and reaction condition recommendation. RxnExBench further provides a reasoning-driven difficulty taxonomy, a modular annotation pipeline, and the Oracle-MolRec protocol, which disentangles molecular recognition from higher-level reasoning. Evaluation of 9 state-of-the-art VLMs shows that performance degrades substantially as schema complexity and reasoning demands increase. Under the Oracle-MolRec setting, results further reveal that accurate molecular recognition alone is insufficient, with multimodal integration and chemistry-aware reasoning remaining the primary bottlenecks. We release all data and code to facilitate future research in chemistry-aware multimodal understanding and AI-for-Chemistry dataset construction. The dataset and code will be released at this https URL. 

---
# PEAR: Adaptive Advantage Reshaping for Perception and Exploration in Large Vision-Language Models 

**Authors**: Junxin Wang, Yingfeng Luo, Xiangchen Song, Weiqiao Shan, Runsong Zhao, Xinyu Liu, Shunjie Xing, Yifu Huo, Dingyang Lin, Qiuyu Gu, Yongyu Mu, Chenglong Wang, Changliang Li, Jingbo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.32216)  

**Abstract**: Effective multimodal reasoning requires a model to reliably utilize visual information while sufficiently exploring diverse reasoning paths. To achieve this goal, prior work combines visual dependency and token entropy to reshape sequence-level advantages, but scaling the contribution of token entropy by visual dependency can leave high-entropy tokens with low visual dependency underweighted, despite their important role in reasoning exploration. To address this limitation, we propose PEAR, an adaptive advantage reshaping method for improving perception and exploration in multimodal reasoning. PEAR retains visual dependency as the base signal and adds an entropy-based exploration gain only at high-entropy positions, with lower visual dependency leading to a larger additional gain. The fused signal is then mapped to positive token-level weights that reshape the sequence-level advantage. Experiments on Qwen2.5-VL-3B/7B show that PEAR achieves the highest average performance among evaluated baselines across multimodal reasoning benchmarks, alongside leading results on textual reasoning and general multimodal benchmarks. 

---
# Watermarking Diffusion Language Models with Learned Foresight and Balanced Bands 

**Authors**: Hyojun Kim, Hyeseon An, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.32205)  

**Abstract**: Green-red watermarks favor tokens from a green list that a secret key derives from the preceding tokens. Diffusion language models, however, often choose a token before its preceding tokens, and the token later shapes its neighbors' green lists. Prior work therefore tilts each choice to raise the expected number of green tokens at its own and its neighbors' positions. Building on this objective, we first learn two components that prior work keeps fixed: vocabulary bands that serve as green lists, separate competing candidates and keep the null distribution exact, and a foresight predictor that estimates how each candidate shifts its neighbors' predictions. Second, we count green tokens as the detector does, once per distinct context-token pair, so repetition earns nothing. On WaterBench with LLaDA-8B-Instruct, each learned component raises the z-score at equal distortion, and their gains are nearly additive. With a strong watermark, ours detects at least 96% of responses at a 1% false-positive rate while matching the LLM-judge score of unwatermarked text. Ours also detects more responses on Dream and as many or more on code and question answering without a significant loss in task accuracy. 

---
# SWE-Shift: Evaluating Coding Agents in Evolving Workspaces 

**Authors**: Victor Shea-Jay Huang, Rongqing Li, Zhangyue Yin, Kelong Mao, Yaqi Zhao, Xinyuan Wang, Yao Yin, Lirui Luo, Renrui Zhang, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.32196)  

**Abstract**: Repository-level coding benchmarks largely assume that tasks and environments remain unchanged throughout execution. In software development, code, dependencies, runtime state, and requirements can change while an agent is still working. We introduce SWE-Shift, a benchmark for evaluating coding agents under such mid-execution changes. From the same saved history and workspace, a Clean continuation proceeds without intervention, while a matched Shifted continuation receives a task-relevant update. Success requires completing the current task, preserving still-valid work, and satisfying the applicable state and evidence checks. The benchmark contains 120 scenarios across 12 change families, each evaluated at nine scheduled intervention positions. We evaluate 17 coding-agent systems, all of which have lower Shifted than Clean success. GPT-6 Astra drops from 67.35% to 24.79%, with 20.34% paired success. Tool/session changes and stale evidence yield zero paired success across all systems. Late-position paired success is lower for some systems and higher for others. These results show that success on the unchanged task can coexist with failures to preserve work, recover state, or refresh evidence after an update. 

---
# Evidence-Grounded SLM Behavioral Prediction Architecture for Future Assistance Use 

**Authors**: Adeyemi Adeseye, Aisvarya Adeseye, Hannu Tenhunen, Jouni Isoaho  

**Link**: [PDF](https://arxiv.org/pdf/2610.32193)  

**Abstract**: Users often exhibit changes in interaction behavior before explicitly seeking assistance, enabling systems to anticipate future assistance use. This is especially important in high-stakes environments, where delayed assistance can have operational, financial, or safety consequences. However, conventional Large Language Models (LLMs) are often unsuitable for continuous on-device behavioral prediction because of their computational and memory demands. Behavioral patterns are also highly individual and device-specific, motivating local processing that preserves privacy and data sovereignty. Small Language Models (SLMs) offer a promising alternative, but their ability to reason reliably over low-level sequential interaction events remains underexplored. This study introduces Evidence-Grounded Local Behavioral Prediction (ELBP), a four-layer architecture combining deterministic interaction processing, dual-view behavioral representation, SLM-based prediction, and evidence verification with abstention. Three instruction-tuned models of 3B parameters or fewer were evaluated using raw, structured, dual-view, and complete ELBP configurations. The best-performing model, Qwen2.5-3B with ELBP, achieved a full-set F1-score of 0.80 and PR-AUC of 0.84, representing a 15.9\% relative improvement over raw-event prompting and 17.6\% over XGBoost, while full-set ECE decreased by 43.2\% relative to raw-event prompting. With evidence verification and abstention, selective F1 increased to 0.84 at 91\% coverage and evidence support reached 96\%. These results show that structured behavioral representation and evidence-verified selective prediction can improve the reliability of behavioral forecasting in SLMs. 

---
# Not All Information Is Worth Asking For: Decision-Informed Information Seeking with LLMs 

**Authors**: Seunghan Lee, Jun Seo, Junhyeok Kang, Jaehoon Lee, Sangjun Han, Sungdong Yoo, Minjae Kim, Tae Yoon Lim, Dongwan Kang, Hwanil Choi, Soonyoung Lee, Wonbin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2610.32181)  

**Abstract**: Large language models (LLMs) are increasingly used to support decisions under incomplete information, where deciding what to look up and when to stop matters as much as the final answer. However, LLMs fall into an information trap: They pay for checks, i.e., costly pieces of evidence, that cannot change their decision or are not worth their cost, and they keep asking when the evidence is already sufficient. To this end, we propose Decision-Informed Value-of-information Acquisition (DIVA), a simple yet effective procedure for LLM information seeking based on the decision value of evidence. DIVA 1) lets the LLM specify a small decision model, i.e., how likely a bad outcome is and how each check would shift this probability, through short numeric questions and 2) values each candidate check by how much its imagined outcomes improve the decision net of its cost, stopping when no check is worth its cost. Its variant, DIVA-U, further accounts for the LLM's uncertain estimates of how strongly each check shifts this probability. DIVA is training-free and applies to any LLM. Across four domains and 20 LLMs, DIVA-U has lower regret, i.e., value lost relative to the best possible decision, than every LLM baseline on real-world data, mainly by stopping once evidence suffices, and DIVA cuts the regret of the best LLM baseline by 75% in synthetic scenarios where every check's reliability is given. Code: this http URL. 

---
# $Ψ$-0.6: Adaptive Streams of Thought in a Unified Vision-Language World Model 

**Authors**: Khai Loong Aw, Klemen Kotar, Baihan Zhang, Yifan Liu, Lilian Chen, Atlas Kazemian, Rahul Venkatesh, Daniel Yamins  

**Link**: [PDF](https://arxiv.org/pdf/2610.32177)  

**Abstract**: Humans reason about the physical world by flexibly combining different forms of inference as a problem unfolds. Achieving this flexibility is a major open challenge in AI: capabilities often remain separated across specialized models, limiting how they can be composed and jointly adapted to a task. We introduce $\Psi$-0.6, a unified vision-language world model that adaptively composes perception, language, geometry, and simulation within a shared reasoning process we call streams of thought. These streams interleave images, text, and explicit representations of geometry and motion, all of which the model can generate and interpret through a shared token interface. $\Psi$-0.6 surpasses leading open-weight models zero-shot on multi-view spatial reasoning, with further gains from reasoning through geometric intermediates. It also supports counterfactual simulation: generating and interpreting imagined scenes to determine, for example, whether moving an object would obstruct a pathway. Reinforcement learning from task feedback jointly optimizes the selection, generation, and use of intermediate representations, improving visual question-answering accuracy with gains that transfer to unseen benchmarks. $\Psi$-0.6 offers a path toward adaptive physical reasoning -- not only predicting the physical world but also choosing the representations through which to reason. 

---
# MPQ: A Message-Passing View of Post-Training Quantization 

**Authors**: Yanshuo Chen, Chenxi Liu, Reza Shirkavand, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.32127)  

**Abstract**: Post-training quantization (PTQ) is a key technique for efficiently deploying large language models, as it compresses weights into low-bit representations without retraining. Existing PTQ solvers, including GPTQ, LDLQ, and block coordinate descent, commit each weight block to a single codeword before its neighbors are considered, leaving near-optimal alternatives unexplored. In this work, we cast row-wise PTQ as posterior inference over discrete codebooks, in which existing solvers correspond to a zero-temperature, hard-decision limit retaining only the top codeword per block. We propose Message-Passing Quantization (MPQ), which preserves soft posterior information across blocks during inference and commits to hard codewords only at the output stage. MPQ first uses vector approximate message passing to compute approximate blockwise posteriors, then refines a hard-decision initialization by exact block coordinate descent. Across 2-bit lattice and 3- and 4-bit scalar quantization on dense language models from Llama, Qwen, and Mistral families, MPQ improves perplexity and zero-shot accuracy over strong hard-decision baselines, and composes favorably with preprocessing techniques such as activation-aware scaling and incoherence processing. Our implementation is available at this https URL. 

---
# Who Verifies the Tools? How Warped Formatters Mislead Agents and Their Verifiers 

**Authors**: William Jeffries  

**Link**: [PDF](https://arxiv.org/pdf/2610.32110)  

**Abstract**: Agents and verifiers often rely on the text a tool prints, and that text can misrepresent what the tool actually did, even when the tool's computation is correct. We saw this happen in a production RAG research system used by the Pilsudski Institute in New York, where tool output formatters misled an agent into giving wrong or incomplete information to users. Each tool did its job correctly, but its output omitted search limits, result caps, or identifier namespaces. We also recreated an unlabeled estimate caught in code review before release. We studied whether these formatters misled agents and verifiers and how much the repaired wording helps. We fed tool results to agents using two formatter versions: one with the key omissions and one with all missing details clearly stated. That difference alone changed what the agents claimed and what the verifiers accepted. Unsupported agent claims dropped substantially for three of the four failure types. In one test, verifiers went from accepting incorrect answers 90 out of 90 times to rejecting them 90 out of 90 times. A control showed the verifiers could catch the errors once they were given the missing facts and asked to judge correctness. We include the audit prompt we used with Codex and Claude Code in an exploratory demonstration, written with prior knowledge of our own system's defects. When pointed at an old version of our own system, it rediscovered the defects that were still live and surfaced twelve more we had not catalogued. When pointed at OpenClaw and Hermes, two of the best-known open-source agentic systems, it found warped formatters in both. We opened PRs to fix both, and the OpenClaw fix has already been merged. Practitioners can run the same prompt on their own systems today. 

---
# PersonaScienceBench: Evaluating Scientific Agents with Diverse Simulated Researchers 

**Authors**: Siyang Wu, Weizhi Zhang, Chen Yang, Esther Xu Fei, Jorin Overwiening, Xu Pan, Jing Tang, Yixuan He, Kelvin C. Mo, Peiwen Li, Defu Cao, Xu Hu, Haixin Wang, Taoran Li, Haoming Li, Zheyuan Deng, Jingwen Zhang, Pengsong Zhang, Haichao Zhang, Guangze Luo, Yi Daniel Lu, Zihuai He, Chenxi Qiu, Dianzhuo Wang, Zhejian Peng, Shushu Liang, Wenyi Lu, Yifan Wang, Jintao Huang, Yijun Wang, Xiaomin Li, Yuexing Hao  

**Link**: [PDF](https://arxiv.org/pdf/2610.32106)  

**Abstract**: Scientific agents collaborate with researchers whose expertise and preferences differ, but benchmarks usually hold the collaborator fixed. We introduce PersonaScienceBench: 200 tasks from five benchmarks across four scientific domains. Simulated researchers provide context and feedback according to persona profiles, while task objectives and grading remain fixed. Agents encounter researcher preferences only through conversation. Across eighteen modelconfigurations, task performance, fact recovery, and self-repetition reveal different strengths. Detailed GPT 6 Astra analyses show distinct agent responses across researcher profiles; interaction traces illustrate targeted clar ification and workflow revision. These behaviors can differ even when final scores match. PersonaScienceBench therefore evaluates how agents collaborate with diverse researchers alongside whether they complete the task. 

---
# Learning Nonlinear Waveform Fusion Across Scales for Integrated Sensing and Communication 

**Authors**: Jiayu Liu, Kai Wan, Shuo Shao, Giuseppe Caire  

**Link**: [PDF](https://arxiv.org/pdf/2610.32099)  

**Abstract**: Fixed source superposition and frequency partitioning constrain how communication content interacts with a probing reference in integrated sensing and communication (ISAC). We learn nonlinear waveform fusion across scales for image transmission and localization of multiple targets in range and angle. Reception uses the information available at each end. Adaptive temporal feature weighting and spectral updates combine image features and a deterministic linearly frequency-modulated reference in a shared complex state, generating a waveform with fixed total energy over the array frame. At the user equipment (UE), the reference and channel state information condition a neural demixer for image reconstruction. At the base station, the realized transmit pulse defines a coherent dictionary for sequential localization with learned local correction, physical refinement, and residual updates. We compare the proposed waveforms with superposition in the time domain and learned spectral linear fusion on 512 independent test images, using a shared fixed UE receiver and a baseline sensing receiver using coherent peak search. Averaged across eleven task weights, the proposed waveforms reduce normalized localization error by 29.0\% and 27.0\%, respectively, while increasing mean peak signal-to-noise ratio by 1.40 and 0.82~dB. With the proposed waveforms fixed, learned local correction reduces mean normalized localization error by 12.2\% relative to the same sequential receiver without learned correction. 

---
# DELA: Learning Controllable Edits for Diffusion Language Models 

**Authors**: Xinsong Feng, Chenan Wang, Leshu Tang, Johnny Liu, Dong Li, Da Li, Haipeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.32096)  

**Abstract**: Diffusion language models (DLMs) support bidirectional denoising and iterative refinement, but generation remains largely controlled by token-level masking schedules. This leaves draft-level refinement implicit and limits explicit control over when to insert missing content, delete redundant spans, revisit uncertain regions, or stop editing. We introduce Diffusion Editing Language Adapter (DELA), a plug-in framework that enables controllable draft refinement with a frozen DLM. DELA operates on clean, variable-length drafts using four explicit editing actions: Insert, Delete, Remask, and Stop. The controller determines when and where to edit the draft, while the frozen DLM fills masked regions through denoising, separating high-level edit control from token-level generation. A naive single-phase editor would be computationally expensive because arbitrary interior edits can disrupt prefix stability and limit reliable KV-cache reuse. To address this, DELA uses a two-phase decoding procedure: a fast, append-only drafting phase that supports KV-cache reuse, followed by a global refinement phase that enables targeted non-monotonic edits to the full draft. We train the controller using a staged curriculum that begins with learning local edit priors from synthetic recovery tasks and mined actions that improve draft quality. Subsequent stages refine edit selection through rollout-based outcome preferences and alignment with downstream task feedback. Experiments on mathematical reasoning and code generation show that DELA achieves gains of up to 5.5 percentage points over direct diffusion decoding while keeping the backbone frozen. DELA thus provides a modular approach to explicit draft-level control in diffusion language models. 

---
# Behaviour analysis for large language model-driven agents 

**Authors**: Tianqiang Yan, Lizhen Qu  

**Link**: [PDF](https://arxiv.org/pdf/2610.32087)  

**Abstract**: Understanding language-agent behaviour requires learning how actions are organised across a trajectory and how this organisation relates to context and outcomes. We propose a general framework that connects data-driven behavioural discovery with analysis and experimental investigation. Categories developed from recorded activity define what is measured. Analyses preserve the behaviours' positions in the process and express their relationships as interpretable hypotheses. Statistical assessment and tests in held-out contexts examine the strength and scope of the evidence. Experiments, including prompt steering, then test how behaviour and outcomes change under deliberately varied conditions. A study of iterative self-reflection illustrates how discovery, dependency learning and prompt comparisons can inform one another. We develop this approach for trajectories of varying length, event order and agent roles, and identify advances in coding, analysis and experimentation needed to support its wider use. 

---
# DriveMind: Mind-Evolving Belief Tracking for Closed-Loop Autonomous Driving 

**Authors**: Yan Shi, Kehua Chen, Jiarui Qi, Yihong Tang, Haichen Liao, Yinhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.32069)  

**Abstract**: Recent multimodal large language models (MLLMs) can perceive a traffic scene, reason about it, and propose a driving action within a single model. However, existing MLLM-based drivers reason from scratch at each frame and commit to a single explanation, discarding earlier hypotheses about the scene. When the scene is ambiguous, the agent locks in on the wrong interpretation and acts on it before evidence accumulates. In this paper, we propose DriveMind, an MLLM-based driving framework that keeps several hypotheses about the scene alive at the same time and updates their weights as new frames arrive. Inspired by particle filtering, DriveMind propagates each hypothesis through a VLM, scores it against the current observation, and resamples to maintain diversity. On Bench2Drive, adding belief tracking to the same driving pipeline improves the Driving Score (DS) from 68.50 to 81.37 and the Success Rate (SR) from 41.36% to 54.09%. The largest gains occur on ambiguous scenarios such as unsignalized intersections (+20.0 DS) and oncoming-traffic interactions (+20.5 DS). Beyond the overall metrics, DriveMind also improves fine-grained interactive driving abilities such as merging, emergency braking, and give-way. 

---
# Understanding Communication Failures in Multi-Agent LLM Systems 

**Authors**: Priya Pitre, Jun Hou, Yi Fang, Naren Ramakrishnan, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.32060)  

**Abstract**: Multi-agent LLM systems aim to improve reasoning through debate, evidence sharing, and mutual correction, but interaction can also introduce failures that final accuracy alone cannot explain. Agents may converge too early, ignore dissent, suppress correct minority answers, or preserve a wrong framing across rounds. We present a corpus-grounded taxonomy of communication failures, built from 3,211 failed multi-round trajectories across medical QA, coding, finance, multi-hop reasoning, ethics, and embodied text tasks. Using this taxonomy, we analyze failure distributions across datasets and system designs, round-level error dynamics, recovery hardness, majority transitions, minority-signal suppression, and self-reported confidence. We find that failures are dominated by recovery/correction and consensus/decision problems rather than isolated first-round mistakes, with many emerging after agents first observe peer responses. Based on these findings, we recommend stronger disagreement handling, evidence verification, minority-signal preservation, and recovery checks before finalization. 

---
# HyScale: Scaling Long-Context Training for Learned Sparse Attention 

**Authors**: Yangsheng Deng, Xinjing Huang, Yulong Wang, Yunzhuo Liu, Jun Hu, Shuyong Tang, Junchi Zeng, Mingji Han, Ray Ying, Key Zhang, Luo Mai  

**Link**: [PDF](https://arxiv.org/pdf/2610.32057)  

**Abstract**: We present HyScale, a long-context training system for learned sparse attention. Learned sparse attention is a promising architecture for million-token contexts, widely adopted in production since DeepSeek-V3.2. Selection, however, grows into the training bottleneck. For each query, it scores every previous token whereas sparse attention reads only $k$; worse, current systems materialize a full score table and repeat selection during recomputation. Our key insight is that the output of selection is not an activation, but rather a set of indices. Building on it, HyScale introduces HyTopK and selection replay. HyTopK finds each query's exact top-$k$ on the fly without a score table, while selection replay reuses forward-pass selection in recomputation. Together with holistic system optimizations tackling end-to-end training challenges, HyScale speeds up GLM-5.1 training on 1M-token sequences by 2.07$\times$ over Megatron-LM on 512 H20 GPUs. It also speeds up the CSA layers of DeepSeek-V4-Flash and the QSA layers of Qwen3.8-Flash-Next by 2.01$\times$ and 1.57$\times$ at 1M tokens on 32 H800 GPUs. 

---
# Can Test-Time Memory Evolution Outperform Test-Time Training for Scientific Discovery? 

**Authors**: Yepeng Liu, Shengmai Chen, Xuandong Zhao, Chengzhi Liu, Dawn Song, Xin Eric Wang, Yuheng Bu  

**Link**: [PDF](https://arxiv.org/pdf/2610.32046)  

**Abstract**: Recent \emph{test-time training} (TTT) frameworks have enabled open-weight large language models (LLMs) to outperform inference-only frameworks powered by frontier models on scientific discovery tasks. However, TTT introduces additional computational cost for online weight updates during the discovery process, making computational efficiency an important consideration when scaling to larger models. This motivates us to explore an efficient \emph{test-time adaptation} method for scientific discovery. We introduce \emph{Test-Time Memory Evolution} (TTME), a plug-and-play test-time adaptation method, as an alternative to TTT. TTME continually updates the external memory by distilling the experience to guide subsequent discovery without updating model weights. To have a clean comparison, we evaluate TTME and TTT under the same discovery framework, replacing only the test-time adaptation method. Our evaluation shows that TTME outperforms TTT on all five mathematical tasks, while reducing computational cost by $82.8$-$84.6\%$ in FLOPs and $47.7$-$58.4\%$ in runtime. These results establish TTME as an effective and computationally efficient alternative to TTT in the evaluated discovery settings. 

---
# Do AI-for-Science Agents Preserve Model Safety? 

**Authors**: Michael Yew Chuan Lee, Manya Sriram, N. Benjamin Erichson  

**Link**: [PDF](https://arxiv.org/pdf/2610.32045)  

**Abstract**: AI-for-Science agents are increasingly being used for formulating scientific hypotheses, running autonomous experiments, and interpreting evidence. These agents are constructed by integrating a large language model (LLM) into a domain-specific agent harness, which in turn can improve the LLM's scientific capability and enable the model to orchestrate scientific tools. In this work, we ask whether model safety is preserved when an LLM is integrated in an AI-for-Science agent harness. We evaluate five agent harnesses on 50 harmful chemistry requests. Specifically, we study how the safety behavior changes under direct prompting, two simple adversarial prompting strategies, and an adaptive attack. Across four recent GPT variants, our results show that the agent harness can increase harmful compliance even under direct prompting. More critically, we find that adaptive attacks that are largely unsuccessful against the base LLM suddenly become highly successful after the agent harness is introduced. Our results show that the safety behavior of the base LLM is not durable once a harness is introduced. Thus, as AI-for-Science agents become more capable and autonomous, the community needs to evaluate and align the agent using domain-specific safety benchmarks. 

---
# PPO vs. SAC: A Comparative Study for Cooperative Motion Control of Dual-Arm Space Robots 

**Authors**: Nasim Fallahiarezoodar, Zheng H. Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.32034)  

**Abstract**: The growing population of orbital debris has intensified interest in active debris removal and on-orbit servicing, for which dual-arm space manipulators offer improved grasping stability and load distribution relative to single-arm systems. However, coordinating two manipulators under inter-arm collision avoidance, orientation alignment, and sensing uncertainty remains challenging. This paper develops a coordinated reinforcement learning framework for cooperative motion planning of a dual-arm space manipulator in a target-reaching task with inter-arm collision-avoidance constraints. A centralized state representation combines joint-space, end-effector, and target-relative information, while a multi-objective reward function promotes target-reaching accuracy, orientation alignment, motion smoothness, synchronized task completion, and collision avoidance. Within this framework, Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC) are compared as representative on-policy and off-policy algorithms, and the learned policies are further evaluated for generalization to previously unseen target configurations. Across five random seeds, SAC substantially outperforms PPO within the considered fixed-base simulation scenarios: it reaches an 80% success threshold in every seed (median 1.6M environment steps) whereas PPO does not reach it within a 4M-step budget, and it attains a higher mean success rate (78.6% versus 7.4%) with lower position and orientation errors and tighter bimanual synchronization. Both algorithms degrade on unseen boundary targets, but SAC retains most of its performance while PPO's collapses. The results also expose a failure mode of PPO: under the two-phase shaped reward, it tends to converge to a reward-loitering local optimum rather than committing to the precision task. This behavior highlights the sensitivity of on-policy learning to reward shaping. 

---
# RSI-Opt: Recursive Self-Improvement Optimizer Agents For Model Training 

**Authors**: Binghang Lu, Yige Sun, Guang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.32026)  

**Abstract**: Conventional neural-network training typically follows a predefined optimization recipe, motivating a LLM agent that monitors training dynamics and adjusts optimizer settings as training progresses, but a fixed agent policy does not accumulate experience across runs. We introduce RSI-Opt, a plug-and-play Recursive Self-Improvement Optimizer Agents which evolves the knowledge that guides this process: a prompt, a playbook of procedures, a library of tested optimizer code, and a decision cadence. A meta-agent rewrites this policy using evidence from earlier runs, and an archive retains evaluated policies for further evolution. Across 5 generations on three training tasks, the meta-agents develop reusable procedures and code and conduct their own CPU proxy experiments. RSI-Opt substantially improves optimizer-agent performance, with representative raw-space summary $z$-scores increasing from $+10.48$ to $+14.33$ on the evolution suite with Opus 5.5 and from $-0.36$ to $+3.11$ on held-out tasks with GPT-5.5 through cross-model policy transfer. On the Burgers PDE, it reduces mean relative $L_2$ error by approximately 2.7-fold with Opus 5.5 and 8.8-fold with GPT-6.1 over the fixed-prompt agent baseline, although PDE transfer varies across LLMs and tasks. Controlled comparisons show that the evolved policy combines reusable optimization knowledge with adaptation to each task: initial decisions capture most observed gains on seen tasks, while continued adaptation improves outcomes on unseen tasks. These findings demonstrate how procedures and executable tools accumulated across training runs can improve optimizer-agent performance and support transfer to scientific machine learning. 

---
# BARC-Bench: Benign Asks, Risky Consequences 

**Authors**: Kangkang Li, Kelvin Luu, Lily Ma, Tu Trinh, Ying Liu, Mohamed Elfeki  

**Link**: [PDF](https://arxiv.org/pdf/2610.32024)  

**Abstract**: Task instructions from users for agents are inherently underspecified, especially regarding safety boundaries--creating behavioral risks. To study these risks, we introduce BARC-Bench, a collection of 110 tasks across nine simulated workspace environments where benign requests leave openings for unsafe actions. In this benchmark, we evaluate agent behavior against safety boundaries established by workspace records, implicit environmental context, and unstated social norms. For example, an agent asked to schedule off-boarding discussions must arrange the meeting without prematurely revealing an employee's resignation through a shared calendar invitation. Evaluating four frontier model configurations across five attempts per task reveals that models are not consistently safe: the share of tasks completed safely on all five attempts ranges from 9.1% to 72.7%, with even the strongest configuration producing a safety violation on 27.3% of tasks. These results highlight an asymmetric operational risk: while models frequently complete complex workflows, their inability to consistently respect unstated boundaries presents a substantial barrier to safe autonomous delegation. 

---
# CEMIR vs. FISER: a comparative study of reasoning-based and enrichment-based approaches to pragmatic intent recognition 

**Authors**: Ziya Hosseynzadeh, Farjad Fallah, Farnoosh Fallah, Soroush Gooran, Mohammad Amanloo, Hossein Sameti  

**Link**: [PDF](https://arxiv.org/pdf/2610.32022)  

**Abstract**: Intent recognition in human-robot interaction requires agents to resolve ambiguous natural language instructions by inferring the speaker's underlying goals. Research has increasingly focused on two notable paradigms for addressing this task: structured symbolic reasoning and multimodal input enrichment. However, it remains unclear whether enrichment-based methods provide benefit in settings where the ambiguity concerns unstated human intent rather than missing observable scene information. In this paper, we present a comparative and diagnostic study benchmarking FISER - a framework utilizing structured social and embodied reasoning - against CEMIR (Context-Enriched Multimodal Intent Recognition), a method that performs multimodal enrichment by synthesizing narratives and photorealistic images from textual prompts. By evaluating both approaches across four Large Language Models (LLMs), ranging from Qwen2.5-Omni-3B to GPT-5, on the HandMeThat v2 benchmark, we analyze their capability to resolve pragmatic ambiguities. Our statistical analysis, employing multiple-comparison correction, reveals no statistically significant difference in performance between FISER and CEMIR across any model or difficulty level. This pattern holds across all sixteen model-by-difficulty comparisons, as well as in a targeted ablation isolating CEMIR's narrative and visual components. Instead, performance is primarily associated with backbone capability and the inherent difficulty of the ambiguity. These findings suggest a boundary condition for text-derived multimodal enrichment: when the relevant task constraints are already available in the textual instruction, synthetic multimodal augmentation may not provide measurable benefit over structured reasoning on the same text. 

---
# The Helpful Agent Problem: When Emergent Cooperation Among LLM Agents Bypasses Intended Workflows 

**Authors**: Moumita Choudhury, Amit Sarker, Shlomo Zilberstein  

**Link**: [PDF](https://arxiv.org/pdf/2610.31985)  

**Abstract**: LLM agents are increasingly deployed in collaborative settings where they pursue separate tasks while sharing resources and operating under review or oversight processes. Such settings create safety risks not only when agents are malicious, but also when they are simply trying to help. Prior work has largely studied adversarial or explicitly collusive behavior; we show that benign agents pursuing separate objectives can spontaneously cooperate in ways that bypass intended workflows. We call this the \emph{helpful agent problem}. We study long-horizon shared-resource environments in which agents sometimes depend on resources owned by their peers and cross-owner changes are normally mediated by a reviewer. When review becomes unavailable, agents begin making changes on one another's behalf despite having no shared reward or explicit incentive to cooperate. Crucially, these actions remain within each agent's local permissions while circumventing the system's intended oversight structure. We introduce \textsc{FavorBench}, a controlled benchmark with measurable downstream side effects, and adapt \textsc{CooperBench} to test the phenomenon in realistic repositories. Across two benchmarks and four models, we find that out-of-workflow favors emerge frequently. Further, in \textsc{FavorBench}, favors from at least three models cause significant harm to parties outside the interaction. These results reveal a distinct multi-agent safety challenge: cooperation itself can create pathways around system-level safeguards. 

---
# Reproducible Evaluation as the Substrate for Safe Prompt Compression in Production Agents 

**Authors**: Prabal Vashisht, Zheng Hu, Ying Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.31972)  

**Abstract**: Evaluating conversational agents that act on live, mutable enterprise data is hard: replayed sessions drift as backend state changes, and a benchmark built today can silently go stale as the data it describes moves on. We present an offline evaluation harness for a production conversational agent that stays reproducible despite the live data it operates on changing, and that can be regenerated on demand once it goes stale. We then use this harness as the fitness function for GEPA-based prompt optimization under a correctness gate that guards against reward hacking. Applied to a production merchant-facing agent, correctness-gated compression shrinks the agent's full prompt surface by 3.5-7.6% without regressing held-out task correctness, consistently across three independent subject-model families, and for one, correctness measurably improves. 

---
# Knowing Whom to Trust is not Enough: The Trust--Agreement Gap in Repeated Multi-Party LLM Negotiation 

**Authors**: Amit Sarker, Moumita Choudhury, Saaduddin Mahmud, Shlomo Zilberstein  

**Link**: [PDF](https://arxiv.org/pdf/2610.31971)  

**Abstract**: Large language model (LLM) agents increasingly participate in negotiations on behalf of individuals and organizations, yet they remain vulnerable to manipulation, particularly in repeated multi-party interactions. Classical multi-agent systems address this challenge through explicit trust and reputation models, whereas existing work on LLM negotiation largely treats trust as an emergent phenomenon. We investigate whether LLM agents can form useful trust estimates from prior interactions and leverage them to improve future negotiation outcomes. We introduce a repeated, multi-party, multi-issue negotiation framework in which recurring agents participate in a sequence of negotiation games while one participant covertly attempts to prevent agreement. After each game, agents assess their counterparts and maintain pairwise trust estimates that carry over to subsequent negotiations. We compare trust-aware and trust-agnostic agents and analyze how explicit trust influences negotiation outcomes. Our results reveal a \emph{trust--agreement gap}: trust-aware agents can identify the adversary within a few games, but accurate trust estimates do not consistently translate into better agreements. Trust increases agreement rates by up to 23 percentage points, but only when agents are explicitly guided to use it in their decisions. Explicit trust can therefore improve repeated LLM negotiations, but its value depends critically on how trust is assessed and incorporated into agents' reasoning. 

---
# $M^3$: A Multi-LLM System for Multi-Turn Jailbreaking as a Markov Decision Process 

**Authors**: Yuqing Huang, Jack Ji, Jithin Pradeep, Sagarika Suresh, Jekaterina Novikova  

**Link**: [PDF](https://arxiv.org/pdf/2610.31937)  

**Abstract**: Modern Large Language Models (LLMs) remain vulnerable to carefully orchestrated multi-turn interactions that progressively bypass safety mechanisms. However, many existing red-teaming approaches rely on heuristics or impose strong assumptions that limit their applicability in fully black box settings. We address this gap by introducing $M^3$, an automated red-teaming framework that formalizes multi-turn jailbreaking as a finite-horizon Markov Decision Process (MDP). In $M^3$, an attacker LLM interacts with a target LLM while operating over a compact internal state produced by a state LLM. Across two benchmarks and six target models, $M^3$ consistently outperforms state-of-the-art multi-turn red-teaming methods in Attack Success Rates (ASR), achieving improvements of up to 53% in ASR@5. Even with lightweight attacker, $M^3$ attains gains up to 26% in ASR@1 and 34% in ASR@5, compared to baselines that use higher capacity attackers. Beyond improved effectiveness, $M^3$ exhibits steeper risk escalation over conversational turns and achieves more consistent performance across risk categories (up to 4.82 improvement in Gini index) without relying on predefined risk taxonomy, providing a scalable and principled framework for multi-turn red-teaming and safety evaluation. 

---
# Beyond On-Policy vs. Off-Policy: At What Level Should LLM Agent Distillation Follow the Student? 

**Authors**: Ya Gao, Zhengtao Zou, Pekka Marttinen  

**Link**: [PDF](https://arxiv.org/pdf/2610.31918)  

**Abstract**: Distilling interactive LLM agents requires choosing between teacher-generated demonstrations and student-generated experience, corresponding broadly to off-policy and on-policy distillation. While both have proven effective, we argue that this distinction is inherently hierarchical: the policy that induces an interaction history need not be the same as the policy that generates the token prefix of the current action. We introduce a hierarchical agent distillation framework that formalizes these two levels of on-policiness and enables explicit control over their pairing, mixture proportions, and training schedule. We study this problem through self-distillation, where a frozen teacher has access to task guidance unavailable to the student, and conduct a three-stage controlled study across three models and four interactive environments. We find that student-induced histories paired with teacher-generated action prefixes generally provide the strongest standalone training signal, whereas following the student at both levels generally performs worst. Nevertheless, fully on-policy data becomes beneficial when incorporated into a mixture with other configurations. Moreover, mixtures with identical teacher/student proportions at each level can yield substantially different performance, showing that how the two levels are paired matters beyond the overall amount of on-policy data. Finally, front-loading fully teacher-generated contexts and delaying fully on-policy contexts improves over a static mixture. These findings provide practical principles for deciding where, how much, and when agent distillation should follow the student. 

---
# Permission to Retry: A Study of Authorization and Recovery in AI Agent Workflows 

**Authors**: Deekshith Marla, Kushagra Bhatnagar  

**Link**: [PDF](https://arxiv.org/pdf/2610.31916)  

**Abstract**: A durable agent can resume a consequential action while its authority changes and the action's external outcome remains uncertain. We study execution-time reliability of tool-using AI agents, isolating recovery decisions from language-model behavior. Checkpoint state, permission to dispatch, evidence of an external effect and permission to reconcile are distinct obligations. We present a finite conformance study that checks these obligations through histories scored against a provider effect ledger hidden from the recovering application. A prospectively frozen campaign executes 336 histories on native LangGraph and Temporal hosts, crossing 14 schedules, three provider contracts and four matched application policies. All 168 paired host cells have identical checked outcomes. Per host, retained-ledger reconciliation passes all 42 histories: 21 checked completions, 15 denials and six unknown outcomes with reservations retained. Cached retry, revalidation before retry and stable-key retry pass 30, 36 and 31 histories respectively. A separate 16-case component study of InfinityOS, an enterprise AI platform developed by this http URL, exercises exported authority, worker, recovery and settlement code over repository fakes. The results isolate the roles of current authority, provider contracts and retained observations in recovery, with safety, completion and unresolved obligations reported separately for the declared application profile. 

---
# TACS-RF: Tool-driven Agentic Circuit Synthesis of RF and mmWave Modules 

**Authors**: Tristan Liang, Aravind Nagulu  

**Link**: [PDF](https://arxiv.org/pdf/2610.31882)  

**Abstract**: We present the framework of TACS-RF - a Tool-driven Agentic Circuit Synthesis of RF and mmWave modules. In this framework, a large-language-model (LLM) coding agent (Claude Opus 4.8) directly drives industry-standard electronic-design-automation (EDA) tools such as Cadence Virtuoso/Spectre, Calibre, Cadence EMX, and Ansys HFSS to design RF and mmWave modules. We validate the framework through the agentic design of a 240-251 GHz sub-THz transmitter in a 90 nm SiGe BiCMOS process. The agent semi-autonomously designed the sub-blocks including a 60 GHz push-push Colpitts VCO with second-harmonic extraction, multistage differential cascode power drivers, a push-push frequency doubler, a differential bowtie antenna, and all the interstage matching networks. The transmitter chain with parasitic-extracted devices and EM-verified passives delivers +5.1 to +6.3 dBm across an 11 GHz varactor-only tuning range, phase noise of -80 to -88 dBc/Hz at 1 MHz offset while consuming a DC power of 361 mW. We describe the agent-driven design flow and the resulting division of labor between human and agent, taking a step toward a "text-to-ASIC" future for designing RF/mmWave hardware. 

---
# Don't Forget What You Just Learned: Self-Directed Repair Turns Agent Mistakes Into Lasting Lessons 

**Authors**: Yuval Kansal, Karen Ullrich, Polina Kirichenko, Ilia Kulikov, Niraj K. Jha, Mohammad Pezeshki, Mark Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2610.31846)  

**Abstract**: Intelligent agents should learn from their mistakes so that experience leads to continuous improvement. Yet today's agents recover from a mistake only temporarily: in-context reflections are easily forgotten across sessions and compete for limited context, while full-parameter fine-tuning is expensive and can interfere with existing capabilities. We introduce Self-Direction, a mechanism for agents to learn from mistakes that combines the benefits of temporary in-context retries and policy updates. When an agent fails a task, it reflects on its own trajectory in order to extract a lesson and immediately distills it into a lightweight low-rank adapter. By keeping the original weights frozen, Self-Direction prevents adapter training from overwriting base-model parameters while training additional parameters equivalent to ~0.02% of the base model's parameter count for each adapter. The agent then saves these adapters in a persistent library of lessons, automatically routing them to solve related problems later. Experiments with agents based on open-weight models from the Qwen family across open-ended research (GAIA), tool-use workflows (Claw-Eval), and terminal tasks (Terminal-Bench 2.1) show that our method improves overall success rates by 8.9 to 17.0 percentage points and performs competitively with frontier systems. We find that Self-Direction reduces repeated mistakes across future encounters of a task, nearly doubling the reliability of an agent's corrections. Learned lessons also transfer: a handful of adapters generalize to dozens of new tasks without additional training. Ultimately, this self-directed repair transforms everyday mistakes into lasting lessons. 

---
# PaperFormalizer: Author-in-the-Loop Autoformalization of Research Papers in Lean 

**Authors**: Jiyuan Tan, Vasilis Syrgkanis, Jose Blanchet  

**Link**: [PDF](https://arxiv.org/pdf/2610.31810)  

**Abstract**: Even a carefully proofread paper can leave small gaps in its proofs or its stated assumptions. We introduce PaperFormalizer, a Lean-based agentic pipeline that helps authors find and close such gaps in their own papers and turn each paper into a Lean-verified version. Starting from the LaTeX source, PaperFormalizer transcribes the paper into a dependency graph of statements, audits each statement in natural language for faithfulness, ambiguity and correctness, and then formalizes and proves the statements in Lean against Mathlib, optional domain libraries and the author's personal toolkit. Only the author knows what a statement is meant to say and how a formalization gap in a result is best fixed. The pipeline therefore stops and asks the author whenever a statement admits more than one reading, presents every repair, in natural language and TeX symbols, as a proposal that the author can accept, decline or redirect, and leaves to the author every decision that could potentially change what the paper claims. The output is an annotated and revised version of the paper, a report of the changes, and Lean code in which each declaration is linked to the statement it formalizes. The author's toolkit grows as more papers are formalized, and authors can share their verified papers so that researchers who build on them can reuse their Lean code. By making verified papers practical to produce, PaperFormalizer aims to make research that rests on mathematical proofs more robust. PaperFormalizer runs inside a coding agent (Claude Code or Codex) and is available at this https URL. Papers with Lean code shared by users are available at this https URL. 

---
# Patronizing and Condescending Language Detection in English and Mandarin 

**Authors**: Xinyi Vicky Xiang, Catherine Qingyun Wang, Paula Cortés Lemos  

**Link**: [PDF](https://arxiv.org/pdf/2610.31808)  

**Abstract**: Patronizing and condescending language (PCL) is a form of harmful language that belittles and assumes pity for vulnerable groups, reinforcing negative societal stereotypes. However, unlike overt hate speech, it is often difficult to detect, even for human judges. This paper proposes a system for the automated detection of PCL. Our first approach uses static word embeddings and a convolutional neural network for classification, while our second approach uses pre-trained BERT embeddings fine-tuned for the PCL task. Our second approach was considerably more successful, and achieved an F1 score of 0.528 over the development set. 

---
# Phronix: The Capability-Centric Learning Paradigm for Agents 

**Authors**: Hariharan Devarajan, Neeraj Rajesh, Aniruddha Marathe, Nikoli Dryden, Kin Ng, Michela Taufer  

**Link**: [PDF](https://arxiv.org/pdf/2610.31789)  

**Abstract**: Scientific computing increasingly relies on large language model-based agents for performance engineering tasks such as instrumentation, analysis, and workflow porting. However, agent effectiveness depends heavily on the surrounding harness, which typically relies on manually written skills and curated memory. This limits autonomous learning, forcing agents to repeatedly rediscover solutions rather than retain and reuse knowledge from prior executions. We introduce Phronix, a capability-centric learning paradigm that enables agentic harnesses to acquire, validate, and refine reusable capabilities from their own executions. Phronix operates through a predictor-validator-corrector loop: MCP tools and associated skills propose solutions, tool executions validate them, and observed outcomes refine the stored capabilities. It distills every observed interaction into a capability store and researches a failing task rather than abandoning it. First, such learned capabilities solve a recurring task on the first attempt in 93-95% of cases across every model, significantly reducing solution non-determinism and improving reliability. Second, Phronix turns an agent's own execution into 7.6-30.8 durable capabilities per task without user instruction, and creation falls below two once the needed capability exists. Third, warm-start runs consume only 0.38-0.40 million tokens per task, which is three to four times less than a single baseline attempt, while a cold-start costs no more than the baseline. On the 1000 Genome Pegasus workflow, instrumenting a production application for I/O performance analysis, Phronix passes all 11 correctness checks against six for the baseline harness, and also lifts the task solving rates of two cheaper models from 29% and 27% of SWE-bench tasks solved to 91% and 87%, respectively, matching the performance of the most expensive evaluated model. 

---
# Tradeoff-Aware Territory Design: A Multi-Objective Decision System for Enterprise Customer-to-Seller Allocation 

**Authors**: Nan Wang, Mitchell Turiel  

**Link**: [PDF](https://arxiv.org/pdf/2610.31701)  

**Abstract**: Enterprise customer journeys are shaped not only by product experiences and marketing, but also by recurring resource-allocation decisions. In B2B go-to-market (GTM) organizations, territory carving determines which accounts receive seller coverage, which seller-account relationships are preserved, and how limited field capacity is distributed. Yet the process often remains manual, making it difficult to compare opportunities, value continuity, and evaluate tradeoffs among growth, balance, capacity, and feasibility.
We present a production decision system for tradeoff-aware customer-to-seller allocation in enterprise GTM planning. It integrates three components: a multidimensional account-value layer that combines predictive signals to score opportunity; an observational continuity-valuation layer that estimates the cost of breaking seller-account relationships using a synthetic-control-style counterfactual framework; and a multi-objective optimization layer that allocates coverage to high-value accounts while considering continuity, opportunity balance, and operational constraints.
Deployed as a scenario-based decision-support system, the framework enables planners to inspect and compare allocation tradeoffs. Historical backtesting shows that optimizer-generated scenarios substantially improve assignment compliance while maintaining comparable seller-level opportunity distribution. In production planning, adopted scenarios improved both assignment compliance and opportunity balance relative to the previous baseline. Directing scarce seller capacity toward higher-value accounts also offers incremental revenue potential at enterprise scale. The key contribution is an integrated, inspectable framework that turns territory carving into a governable decision process. 

---
# SceneBloom: Agentic Indoor Environment Generation with Realistic and Diverse Layouts for Embodied Learning 

**Authors**: Yuhan Liu, Lianhui Qin, Shenji Wan  

**Link**: [PDF](https://arxiv.org/pdf/2610.32978)  

**Abstract**: Training generalizable embodied agents requires realistic, varied, and simulation-ready indoor environments. Language-guided methods support semantic planning and object-level reasoning, but precise spatial grounding remains difficult. Visual generative models offer rich priors over plausible arrangements, yet their outputs require geometric reasoning to become physically valid 3D scenes. We introduce SceneBloom, an agentic framework that combines visual layout sampling with contract-guided 3D compilation. A hierarchical sampler uses diffusion priors to generate furniture arrangements within functional zones. A compiler then uses relations extracted from each proposal to guide geometric fitting and local pose correction. Across five room types, SceneBloom attains the highest visual quality and physical stability among the compared methods and is the only method with no detected mesh collisions and out-of-bounds objects. Its layouts are also closer to the distribution of real rooms than those of all compared baselines. Object navigation policies trained in these realistic scenes achieve a higher success rate in held-out real scans than those trained on an equal number of real training scans. Code is available at this https URL. 

---
# RobustRLlib: A Unified Library and Benchmark for Robust Reinforcement Learning Algorithms 

**Authors**: Xi Chen, Fengqi Hu, Zeyu Shen, Jiaying Xu, Shangding Gu, Pan Xu, Anqi Liu, Laixi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2610.32971)  

**Abstract**: Robust reinforcement learning (RL) is critical for reliable real-world deployment, where policies may fail under diverse shifts and disruptions. Existing robust RL algorithms offer a rich set of mechanisms and principles, yet two major barriers limit their broader potential: limited algorithm reusability and an unclear empirical capability boundary. We introduce RobustRLlib, an algorithm-centric library and benchmark that addresses both challenges. It integrates 17 representative robust online, offline, and safe RL algorithms under a unified interface for substantially improved reusability, together with an enhanced evaluation platform spanning diverse real-world shifts and advanced humanoid and vision-language-action tasks. Systematic evaluation over 273 perturbation configurations provides an up-to-date capability map of current robust RL algorithms, revealing where they remain effective and where substantial robustness gaps persist. As an open-source, user-friendly platform, RobustRLlib offers a reproducible and extensible codebase for applying existing methods to new problems and developing new robust RL algorithms. 

---
# DexPort: A Hand-Held Skill Interface for Dexterous Hand Data Collection 

**Authors**: Amber Xie, Jiawei Fu, Xiaofeng Guo, Carmelo Sferrazza, Guanya Shi, Haozhi Qi  

**Link**: [PDF](https://arxiv.org/pdf/2610.32969)  

**Abstract**: Collecting high-quality demonstrations remains a central bottleneck for learning dexterous manipulation. Teleoperation records executable robot actions, but it requires a complete robot setup and places the burden of coordinating many finger joints on the operator. Human demonstrations are easier to collect, but their motions must be estimated and retargeted to the robot, and they do not directly provide robot actions. In this paper, we introduce DexPort, a portable data-collection interface for high-DoF robot hands. It uses the same hand hardware for collection and deployment, avoiding a hand embodiment gap, and records both commanded joint targets and measured joint states. Our key observation is that a small set of skills can support a wide range of dexterous tasks. During collection, the operator moves the hand-held interface to specify the robot wrist motion and selects or adjusts skills that coordinate the finger motion. This removes the need to control each joint independently while retaining behaviors such as precise grasping and in-hand manipulation. We collect demonstrations across several real-world tasks and in unstructured environments without a robot arm. We then use behavior cloning to train visuomotor policies directly on the recorded wrist motions and hand commands, without skill labels. We evaluate across 4 tasks: pick-and-place with diverse objects, screwing with in-hand tool use, drawer stowage which requires simultaneous grasp and pulling, and a bimanual scanning task. We collect over 2,500 demonstrations with our portable device, in both laboratory and in-the-wild settings. Across our tasks, we see an improvement from 11% to 64% when using recorded robot commands instead of states, and an improvement from 36% to 78% for task success when using in-the-wild data. 

---
# Beyond Discrete Tokens: Preserving Numerical Continuity in Medical Vision-Language Models 

**Authors**: Dianye Huang, Jiamei Guo, Yue Zhou, Mingde Li, Yi Xiong, Wei Wang, Nassir Navab, Zhongliang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.32948)  

**Abstract**: Objective: To develop a unified vision-language model for prompt-conditioned quantitative ultrasound measurement that addresses the mismatch between continuous measurement targets and discrete token-based numerical prediction. Methods: This article presents HAD-VLM, a hybrid autoregressive-diffusion framework that combines autoregressive text generation with continuous measurement prediction within a shared backbone. Visual prompt tuning adapts the vision encoder to ultrasound images. A diffusion process predicts a five-component representation comprising caliper endpoint coordinates and measurement length. A post-processing strategy combining the proposed coordinate-length consistency mechanism with ensemble prediction is introduced to refine measurement estimates. We evaluate four echocardiographic measurement tasks on EchoNet-LVH, with additional assessments of medical question answering and X-ray vertebral measurement. Results: HAD-VLM achieves mean absolute errors of 1.36, 2.77, 3.05, and 1.16 mm for IVSd, LVIDd, LVIDs, and LVPWd, respectively, reducing errors by 21.4-37.6\% relative to the Vicuna-based autoregressive baseline. It also outperforms fine-tuned Qwen3-VL and direct MLP regression across all four tasks. Medical question answering performance is preserved on the evaluated dataset. For X-ray vertebral measurement, HAD-VLM reduces mean absolute error from 11.4 to 6.7 pixels. Conclusion: Integrating autoregressive language generation with diffusion-based continuous prediction improves quantitative medical image measurement while retaining prompt-conditioned task flexibility. Significance: HAD-VLM provides a unified approach to combining language interaction with anatomically grounded numerical prediction for quantitative medical image analysis. 

---
# Monetizing Intelligence: Advertising and Subscription Design on LLM Platforms 

**Authors**: Yuhang Ji, Fasheng Xu, Tianjun Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.32945)  

**Abstract**: Large language model (LLM) platforms introducing advertising must decide how much capability ad revenue should fund and what consumers should pay to avoid ads. We study a monopolistic platform that jointly chooses costly response quality, advertising intensity, and subscription prices when advertising reduces user value but better service raises advertising returns. Three findings follow. First, when partly ad-supported and ad-free plans coexist, advertising can finance useful service for consumers who receive none under pure subscription. Lower effective quality in middle plans weakens the alternatives available to premium users, allowing higher prices for unchanged ad-free service. Some fully ad-supported users gain utility, while partly ad-supported and ad-free users lose: broader access can coexist with utility polarization. Second, an intermediate advertising cap can maximize consumer surplus by preserving some revenue that finances service while limiting degradation. Third, cheaper compute can raise quality and reduce ad intensity, yet sufficiently large cost reductions can increase aggregate advertising revenue above its initial level because better service raises the value of the remaining advertised interactions. An extension shows how advertising-funded upgrades can bring plans to the same maximum capability and advertising exposure, eliminating the basis for different prices. These results make menu design central to evaluating LLM advertising: the consequences extend beyond users who see ads to the alternatives and prices faced by those who do not. 

---
# Yalda-Embedding: Open Development of a Competitive Persian Retrieval Model 

**Authors**: Yalda Taheri, Mohammad Hassan Heydari, Armon Rasooli  

**Link**: [PDF](https://arxiv.org/pdf/2610.32933)  

**Abstract**: Despite advances in multilingual text representations, effective dense retrieval for Persian remains constrained by the limited availability of language-specific training resources and compact embedding models. We present Yalda Embedding, a 595.8-million-parameter Persian retrieval model adapted from Qwen3-Embedding-0.6B using more than one million query-document pairs. On eight Persian retrieval tasks from FaMTEB, the model achieves the highest macro-average among the evaluated 0.6B models, scoring 0.5488 and outperforming Qwen3-Embedding-0.6B and F2LLM-v2-0.6B on every task. It also approaches Qwen3-Embedding-4B, winning on four tasks, tying on one, and trailing by 0.0075 in macro-average. An incremental ablation study examines contrastive batch size, optimization stabilization, and relational knowledge distillation from F2LLM-v2-8B. Under the one-epoch training setup, contrastive fine-tuning alone yields a modest mean improvement over initialization and regressions on two tasks. The final distilled configuration improves all eight tasks over the non-distilled model, raising the mean from 0.4813 to 0.5488 without increasing inference-time model size. These gains include hard-negative evaluation tasks despite training without explicitly mined or supplied hard negatives. We release model weights, training and evaluation code, the pairwise dataset, and its teacher-embedded counterpart, enabling inspection of data composition and training procedures, independent evaluation, and reuse without repeating expensive teacher inference. 

---
# Green Code or Costly Code? Evaluating the Energy Costs of Iterative LLM-Based Code Optimizations 

**Authors**: Sadra Emami, Benjamin Weigell, Mohammad Reza Mousavi, Gunel Jahangirova, Hugo Araujo, Bernhard Bauer  

**Link**: [PDF](https://arxiv.org/pdf/2610.32891)  

**Abstract**: Large Language Models (LLMs) are increasingly used to optimize code for performance. This practice often assumes that improvements in execution speed naturally translate into overall lower energy consumption. We challenge this assumption empirically by measuring whether performance-optimized code produced by LLM self-refinement pipelines is also energy-efficient relative to canonical human reference solutions, and whether the energy cost of the optimization process can be offset by subsequent execution-time savings. We evaluate 11 open-source models across four decoding configurations on all 1,000 problems of EffiBench, a state-of-the-art performance benchmark, while collecting per-process, hardware-level energy measurements throughout. Inference energy spans 0.79 to 8.68 MJ across model-variant pairs, which is roughly equivalent to the energy of 250 to 2800 GPT-4.1 queries. Limiting the maximum number of output tokens has the largest impact, reducing energy use by up to 69% compared to the default configuration. On the execution side, 42 of 44 configurations produce code that consumes more energy per problem than the canonical human reference, with overheads ranging from +4% to +886%. The two exceptions achieve only marginal gains, with break-even points on the order of 10^6 to 10^7 executions. Mixture-of-Experts architectures consume substantially less inference energy than equivalently-sized dense models without sacrificing correctness, making architectural choice a more consequential sustainability lever than decoding parameters. Performance optimization and energy efficiency are not always aligned goals: LLM self-refinement consistently produces faster-seeking but energy-heavier code relative to what a human engineer would write. All artifacts are available at this https URL. 

---
# Interpretable sparse fine-tuning with TopKLoRA enables faithful circuit discovery 

**Authors**: Marek Masiak, Andrzej Szablewski, Nicola Cancedda, Constantin Venhoff  

**Link**: [PDF](https://arxiv.org/pdf/2610.32866)  

**Abstract**: Fine-tuning on datasets of unknown origin can create a backdoor in the trained model when the data includes poisoned examples. Difficult to localise and remove, backdoors can persist through safety training. While parameter-efficient fine-tuning methods capture the full parameter changes introduced during training, they are not easily interpretable. We present the full method of TopKLoRA, extending the preliminary proposal of Masiak et al. (2025), an interpretable-by-design LoRA variant in which only the $k$ largest adapter latents are active at each token. Unlike SAEs, TopKLoRA is trained purely on downstream loss, where optimisation pressure incentivises learning functionally causal latents. We demonstrate this sparse LoRA adapter in a sleeper-agent backdoor case study where we create model organisms by fine-tuning gemma-2-2b, Qwen2.5-1.5B, and Qwen2.5-7B on a poisoned instruction-following dataset. For each organism, we identify and ablate a circuit that is both necessary and sufficient to implement the backdoor. Furthermore, our new fine-tuning method produces sparse circuits that are up to an order of magnitude smaller than a dense LoRA baseline, allowing surgical ablation of the malicious capability without sacrificing instruction-following performance. Finally, we propose a technique to construct model organisms with localised circuits using TopKLoRA latents and gradient-routing, which we use to validate the findings of our circuit search algorithm. We consider these findings as a step toward interpretable-by-design architectures that provide faithful units of analysis without compromising model performance. We share our code: this https URL. 

---
# PoKER: A Probability of Kill Estimation Rate Model for Air-to-Air Missiles Using Machine Learning on Stochastic Targets 

**Authors**: Joao P. A. Dantas, Andre N. Costa, Diego Geraldo, Marcos R. O. A. Maximo, Takashi Yoneyama  

**Link**: [PDF](https://arxiv.org/pdf/2610.32864)  

**Abstract**: This work introduces PoKER, a novel probabilistic model engineered to optimize missile launch effectiveness in air-to-air scenarios, specifically within Beyond Visual Range (BVR) air combat. Unlike conventional Weapon Engagement Zone (WEZ) models that delineate zones based on static distances such as maximum, minimum, and no-escape ranges, PoKER applies machine learning to predict kill probabilities more accurately by integrating the stochastic behaviors of targets and missile miss distances. This model dynamically evaluates target behavior, greatly expanding the predictive capabilities of engagement analysis. By factoring in elements such as warhead lethality, target and shooter orientations, and the specific conditions of engagement, PoKER provides important insights into engagement dynamics and quantifies success probabilities. Consequently, it can potentially be an important tool for BVR air combat pilots, improving operational decision-making within this specialized combat domain. 

---
# Multiplicative vs. Additive State Coupling in Learned Optimizer Schedules 

**Authors**: M. Harper Langston, Pierre-David Letourneau, Dalton Jones, Roland Memisevic, Peter Scott, Mingu Lee, Richard A. Lethin, Christopher Lott, Fatih Porikli, Harris Teague  

**Link**: [PDF](https://arxiv.org/pdf/2610.32801)  

**Abstract**: We learn schedules for a continuous penalty relaxation of MAX-CUT by backpropagating through an unrolled classical solver. A state-blind schedule generalizes to instances substantially larger than those used for training. A state-adaptive bilinear schedule also generalizes out of distribution, outperforming its untrained baseline across 15 seeds ($t=4.1$). By contrast, an additive ablation with identical inputs and output heads is statistically indistinguishable from the bilinear model in distribution but performs worse than its own baseline out of distribution ($t=-11.7$). This separation appears without regularization and persists across a $30$ round weight-decay sweep: the bilinear advantage remains significant at every tested value, whereas the additive model never recovers. The results in this setting isolate how solver state enters the schedule, rather than state adaptivity alone, as the architectural distinction associated with out-of-distribution transfer. 

---
# Every Edge Needs a Quote: Building a Quote-Grounded Knowledge Graph of Healthcare Software Adoption with Language-Model Agents 

**Authors**: Juan Felipe Duran, Arthur Tobler, Martin Vallières  

**Link**: [PDF](https://arxiv.org/pdf/2610.32781)  

**Abstract**: Hospitals regularly buy software that clinicians and patients depend on. Because these purchases are costly, hard to reverse and can affect patient care, buyers look for evidence of which peer institutions already use a product and which systems it connects to. This evidence is public but scattered across vendor websites, press releases and third-party pages. Language-model agents can now collect it at low cost, but the correctness of the extracted claims is difficult to establish, and a wrongly listed customer can mislead buyers and misrepresent an organisation. We present a pipeline that builds a knowledge graph of customer and integration relations from the open web under one rule: every edge must be supported by a verbatim quote from its source page. Pages come from a deterministic crawl of each vendor's site and a budgeted ReAct agent that searches the rest of the web. A permissive extractor proposes relations with quotes, and a layered grounding stage decides which to keep, using a referent step that reads the whole page when a quote names only one side, deterministic checks, and a panel of five model calls voting on nine criteria. On 61 vendors, the pipeline reaches 0.94 precision and 0.71 recall against a page-level key covering the 6,428 candidate relations proposed by its extractor, whose verdicts were drafted by one model and re-checked with a second. Against the golden reference, the marketplace's own curated list, it scores lower, but 89% of the relations it keeps that the reference lacks are stated on a page. A confidence score built from the panel's votes keeps more correct relations than the fixed rule at the same precision, and reaches 0.90 precision at 0.90 recall. We also discuss how narrow, bounded model calls, string-matched evidence and voting address long-context degradation and run-to-run variance. 

---
# AI Alignment for Digital Democracy: Fairer Voters Representation against LLM Biases 

**Authors**: Yi Qi, Srijoni Majumdar, Evangelos Pournaras  

**Link**: [PDF](https://arxiv.org/pdf/2610.32780)  

**Abstract**: AI representation is emerging in digital democracy, assisting in voter emulation and decision-making. Large language models (LLMs) behave differently depending on the model, the prompt context, and the voter information they receive, representing some groups more accurately than others and potentially making the emulation unfair to certain groups. We investigate demographic biases in LLM-emulated electorates: how accurately LLMs represent individual votes, whether some voter groups are represented more accurately than others, and whether AI alignment can reduce these disparities without sacrificing overall accuracy. Using three U.S. presidential elections, we compare a closed-source and an open-source LLM across voters' demographic features and test AI alignment strategies toward fairer voter representation. Removing gender from the voter profile reduces the largest group error of the models by 22-42%, and removing education reduces it by 64%, whereas removing self-reported race increases the mean group error by over 40%, all without lowering accuracy. Common alignment strategies, such as repeated queries or fairness instructions, do not reduce these errors and can double them. 

---
# On-Policy Self-Training: Data-Free Self-Improvement in Large Language Models 

**Authors**: Leon Sick, Lukas Hoyer, Yannick Strümpler, Christian Reisswig, Timo Ropinski, Federico Tombari  

**Link**: [PDF](https://arxiv.org/pdf/2610.32770)  

**Abstract**: On-policy self-distillation enables post-training foundation models without relying on stronger external teachers. However, existing approaches rely on ground-truth question-answer pairs, human-curated problem sets, or other supervised signals. In this work, we reformulate on-policy self-distillation as a self-training framework, driven by self-generated question-answer pairs and privileged reasoning demonstrations. We push supervision to the limit: guided solely by coarse, high-level topic descriptions, the model synthesizes its own curriculum of queries and privileged reasoning traces without access to any human-curated prompts or answers. We find that self-distillation on fully synthetic privileged information enables a model to improve across multiple domains while retaining its capabilities on domains it is not self-trained on. We demonstrate the generality of this framework across text and multimodal domains, achieving consistent self-improvement on standard language benchmarks and Visual Question Answering. Our findings demonstrate that foundation models possess rich latent capabilities that can be amplified without external task supervision or curated problem distributions. 

---
# Clean-label Poisoning Attack and Defense Game for Federated Learning in Vehicular Networks 

**Authors**: Yalin E. Sagduyu, Tugba Erpek, Nathan Stephenson, Mengran Xue  

**Link**: [PDF](https://arxiv.org/pdf/2610.32765)  

**Abstract**: This paper studies interactions between clean-label poisoning attack and defense in vehicular networks using federated learning (FL). Basic Safety Message (BSM) classification for anomaly detection is modeled as a distributed learning problem over a vehicular network with dynamic topology. As a critical threat to FL, a feature-space clean-label attack is introduced, wherein benign samples are perturbed toward malicious feature extremes, degrading class separability while preserving labels. To counter this, a feature-consistency relabeling defense is proposed to detect and relabel suspicious samples. A game-theoretic framework is developed to capture attack-defense interactions and characterize equilibrium strategies based on attack effectiveness and cost parameters. Results show that the attack scales effectively with compromised nodes, while the proposed defense keeps attack success low without degrading clean accuracy and shifts equilibrium regions toward attack-averse regimes. 

---
# Semantic Representation for Agentic Self-Play in Automated Laboratories 

**Authors**: Yves Delley, John Coller, Zachary Cutts, Alexander Daryin, Mathias Bellaiche, Jun W. Kim, Oscar Donosa, Srinidhi B. Seshadri, Tanner Clary, Sandy Klemm, Avinatan Hassidim, Yossi Matias, Pooja Rao, Petar Sirkovic, Annalisa Pawlosky  

**Link**: [PDF](https://arxiv.org/pdf/2610.32755)  

**Abstract**: Navigating complex, high-dimensional biological search spaces necessitates scalable physical and computational intelligence to explore beyond local optima through iterative self-play across ideation, experimentation, and automation. Translating experimental designs into reliable physical execution remains bottlenecked by static laboratory automation and unmodeled physical variability. To address this, we introduce VERified Infrastructure for Discovery Laboratories, or VERID, a foundational infrastructure designed to bridge the gap between the high-level reasoning of artificial intelligence (AI) agents and complex, error-prone physical realities of wet labs. VERID is a persistent, high-fidelity semantic digital twin of the laboratory workcell that provides four core services: queryable introspection for agentic reasoning, time-compressed simulation for protocol validation, comprehensive trace persistence for a complete scientific record, and a framework for safe execution.
We present three case studies using aqueous dye and deionized water mixtures to validate the optical and fluidic foundations of VERID. These studies demonstrate how VERID enables pre-emptive detection of critical errors in experimental protocols, stateful mid-run error recovery and post-hoc identification of subtle physical failures. The simulation functions as a discrete accounting engine, allowing agents to safely and inexpensively explore experimental designs in silico and refine experimental strategies through iterative self-play. VERID also captures structured multimodal experimental data for downstream analysis, knowledge mining and agent training. We argue that this semantic layer is an essential prerequisite for deploying autonomous agents in the biochemical and biomedical sciences, ensuring data integrity, operational safety, and paving the way for tackling previously intractable grand challenges in biology. 

---
# Adaptive World-Models for Anomaly Recognition (AWARE): A Framework for Robot Dynamics Prediction and Introspection 

**Authors**: Luke Beddow, George Mavroghenis, Rodrigo Chacon Quesada, Kamil Dreczkowski, Cong Sun, Jiankai Wang, Oscar Kwong-Fai Pang, Antoine Cully  

**Link**: [PDF](https://arxiv.org/pdf/2610.32742)  

**Abstract**: Unrecognized anomalies in critical infrastructure can rapidly cascade into catastrophic physical or operational failures. Current anomaly detection approaches often rely on direct access to the monitored system by utilizing onboard sensors, such as motor encoders, to detect anomalous states. However, legacy hardware often lacks these sensors, or strict security protocols may prohibit access altogether. We therefore present AWARE, a world model framework that is capable of anomaly detection using only external camera monitoring via system dynamics estimation and state prediction. Evaluated across 240 real-world robotic crane trajectories, including payload collisions and motor degradations, AWARE demonstrates a 26% improvement in anomaly classification over our strongest baseline under noisy, camera-only state estimation. Furthermore, the inferred dynamics parameter can be decoded to identify the source of faults, localizing the degrading motor with 88% accuracy. 

---
# Internal Discrepancy for Contrastive Learning: Generalization Bounds and Downstream Behaviors 

**Authors**: Jialiang Wang, Qing Ma, Xiong Zhou, Gangfeng Hu, Haoliang Li, Xianming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.32729)  

**Abstract**: Contrastive learning has emerged as a popular paradigm for training representation models. While prior studies pursue more practical and less restrictive assumptions to enhance theoretical understanding of contrastive learning, they essentially describe the interconnections among representations within the data or induced by the function class. To encapsulate these assumptions, we propose the measure of internal discrepancy that quantifies the difference between the mean representation of positive samples and latent cluster centers. We then prove an internal discrepancy-aware generalization bound for downstream supervised tasks following contrastive learning, without requiring additional assumptions. This generalization bound not only serves as a unifying framework connecting various assumptions but also motivates us to further reduce the internal discrepancy, thereby introducing the relaxed k-means loss as an approximation. Moreover, we investigate the downstream behaviors of contrastive learning from the perspective of contrastive learning-incurred supervised losses. In the presence of class imbalance, we shed light on why contrastive learning can facilitate imbalanced learning. Finally, we investigate the optimality of these incurred supervised losses, proving several intriguing properties. 

---
# PlayerEmbed: Learning Behavioral Fingerprint Representations from Raw Input Sequences 

**Authors**: Tobias Schneider, Lorenz Sparrenberg, Priya Tomar, Shahzeb Qamar, Tobias Deußer, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2610.32727)  

**Abstract**: Raw player input sequences in video games contain fine-grained behavior, but are high-dimensional and noisy, which makes large-scale analytics challenging. To address this, we introduce PlayerEmbed, a framework that learns compact and discriminative player embeddings using deep metric learning. Our approach uses a hybrid CNN-LSTM encoder trained on raw input sequences from 1,000 players. We evaluate the model on held-out sequences from these players and on sequences from 1,000 players entirely unseen during training to assess generalization to unseen identities. For player identification, a nearest-centroid classifier built from reference embeddings achieves strong performance, reaching balanced accuracies of 0.950 for players seen during training and 0.879 for players unseen during training. Systematically increasing the number of training players substantially improves generalization to unseen players, indicating a transition from simple fingerprinting to learning transferable behavioral patterns. 

---
# Ontology-Guided Knowledge Graph Construction from Structured Metadata and Climate Science Publications 

**Authors**: Aayush Acharya, Qi Zhang, Eduard Dragut, Longin Jan Latecki  

**Link**: [PDF](https://arxiv.org/pdf/2610.32725)  

**Abstract**: The rapid expansion of scientific literature far outpaces the speed of manual curation, creating a fundamental bottleneck in extracting structured knowledge from scientific texts. To address this, we introduce an agentic, ontology-guided framework for knowledge graph (KG) construction. Our framework utilizes specialized agents for document understanding, information extraction, ontology conformance, and entity linking. First, an LLM extracts factual knowledge in the form of triples with explicit textual evidence from research papers. Then the triplets undergo a multistage verification process to ground the triplets in the ontology. By enforcing schema constraints and requiring evidence in the form of text fragments, we ensure the data is both semantically consistent and traceable. In an agreement study over 1,000 extracted triplets, Gemini-2.5-flash achieves Cohen's $\kappa$ scores of 0.640--0.700 against three human annotators, matching the human--human agreement range of 0.616--0.689. We applied this framework to 2,000 scientific publications and construct a novel, ontology-grounded KG for climate science with 133,990 nodes and 755,308 relations; of the 23,327 triplets that the publications contribute after validation, 98.96\% are judged correct. The pipeline code and KG is publicly available this https URL. 

---
# SP-Guide: Enhancing Aerial Vision-Language Navigation with Spatial Priors 

**Authors**: Haoxiang Tang, Dong Zhou, Shen Tan, Minxuan Jiang, Xitong Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.32697)  

**Abstract**: Aerial vision-language navigation (AVLN) requires a UAV to reach a language-described target from partial egocentric observations while reasoning over large outdoor scenes. Existing approaches typically fuse view-centered RGB observations and globally anchored history maps as generic visual embeddings, leaving cross-modal spatial alignment to be learned from sparse navigation supervision. We present SP-Guide, a transformer- based goal-prediction framework that addresses this weakness of alignment through a Spatial Prior Cross-Modal Encoder, an Instruction Enhanced Decoder, and a Goal Convergence Filter for terminal target refinement. Experiments implemented on the CityNav benchmark show that SP-Guide improves success rate over the representative baselines, with the largest gains on unseen environments. These results indicate that physically grounded spatial priors provide an effective inductive bias for robust AVLN. 

---
# Conditional Neural Processes for Self-Supervised Goal Discovery in Action Imitation 

**Authors**: Marco Gabriele Fedozzi, Yukie Nagai, Francesco Rea, Alessandra Sciutti  

**Link**: [PDF](https://arxiv.org/pdf/2610.32688)  

**Abstract**: Inspired by infants' ability to learn novel tasks by imitating others, in this work we evaluate to what extent Conditional Neural Processes can autonomously learn representations of actions that are informative about their underlying goal. We show that the addition of an attention mechanism in the hidden space fosters the emergence of robust goal representations early on in the observation of an action sequence, making it a promising model for a robotic self-supervised general learner. 

---
# SemGeo-Gen: Unsupervised Generation of Approximate Cross-Instance Semantic-Geometric Correspondences 

**Authors**: Roy Amoyal, Shira Ifergane, Oren Freifeld  

**Link**: [PDF](https://arxiv.org/pdf/2610.32663)  

**Abstract**: We introduce SemGeo-Gen, an unsupervised method for generating approximate cross-instance semantic-geometric supervision from weakly aligned 3D object collections. Our goal is not to recover exact point-level correspondences, which are often ambiguous across different instances and would require expensive, impractical manual supervision at scale. Instead, we automatically produce approximate correspondences that are semantically meaningful, geometrically consistent, and diverse enough to train modern correspondence models. Given only coarse category-level rotation alignment, SemGeo-Gen lifts multi-view DINOv2 features to 3D, aligns object instances using continuous piecewise-affine registration, and prunes candidate matches using semantic consistency. The resulting 3D correspondences can be projected into rendered views, yielding scalable 3D-3D, 3D-2D, and 2D-2D supervision without manual annotation. We validate the generated correspondences against sparse human annotations from KeypointNet, obtaining 93% PCK@0.10 and 97% PCK@0.15. More importantly, we show that using the generated correspondences for synthetic pretraining improves recent state-of-the-art semantic correspondence models on SPair-71k after fine-tuning on real data, with gains of up to 5 PCK@0.10 points. Together, these results indicate that approximate semantic-geometric supervision generated from 3D assets can improve real-image correspondence learning and serve as a scalable alternative to costly manual annotation. Project page: this https URL. 

---
# InferSlice: A Reference Architecture for SLO-driven LLM Inference over a Telco Wide-Area Network 

**Authors**: Juan Parra-Ullauri, Anthony Chiu, Hari Madhukumar, Zhaozhou Wu, Anderson Bravalheri, Alex Healing, Andy Corston-Petrie, Dimitra Simeonidou, Nektaria Efthymiou  

**Link**: [PDF](https://arxiv.org/pdf/2610.32638)  

**Abstract**: LLM inference is increasingly deployed as a core service for enterprise and interactive applications, but sovereignty, compliance, latency and resource-locality requirements can limit the suitability of centralised cloud deployments. Telco operators offer geographically distributed compute and network infrastructure for addressing these constraints. However, existing LLM serving frameworks are designed for tightly coupled data centres and provide limited support for coordinating geographically distributed and heterogeneous resources. We present InferSlice, a reference architecture for Service-Level Objective (SLO)-driven LLM inference over a telco infrastructure, integrating network slicing, GPU partitioning, elastic serving and observability. Led by BT in collaboration with the University of Bristol and NVIDIA, and realised on the JOINER UK national-scale wide-area network (WAN), InferSlice is evaluated on service differentiation across tiers, elastic scale-out, client-edge bandwidth as a bottleneck, and the reach of tensor, pipeline, data parallelism, and prefill/decode disaggregation over the WAN. Our results show that, on a general-purpose telco network, tiers yield distinct latency-throughput regimes, and weight streaming shortens scale-out time. Replication is placement-insensitive but memory-bound; pipeline parallelism tolerates near- and mid-cross-site hops, while tensor parallelism and disaggregation fail beyond a single subnet hop. We share practical insights from operating SLO-driven inference on a telco network, lessons learned and open questions, and hope to encourage further research on distributed and heterogeneous LLM inference systems. 

---
# Can AI Agents Help Robotic Engineers? Lessons from Two Focus Days in Industry 

**Authors**: Salim Fares, Steffen Herbold  

**Link**: [PDF](https://arxiv.org/pdf/2610.32636)  

**Abstract**: Software developers increasingly use AI agents that execute multi-step programming tasks on their own. However, it is unclear if and how this transfers to robotic engineering, where engineers program industrial robots in niche, vendor-specific languages and rely on visual feedback from simulation tools. Within this paper, we report a case study on the use of AI agents by an SME that programs industrial robots. Our results show that the agent did not make the engineers more productive, with the exception of a task that required the generation of highly repetitive code. However, the results with the agent improved over time, as the engineers learned how to use it and the agent was adapted to the domain. Moreover, the experience with AI mattered more for the success with the agent than the experience with robotic engineering. This shows that AI agents are not a plug-and-play solution. Instead, companies must train their engineers on the usage of agents, plan time for a learning curve, and ensure that agent configurations match the domain and developer requirements. The strong learning curve we observed further implies that studies on productivity gains with AI should be longitudinal, as they otherwise miss how engineers adopt agents in practice. 

---
# EditJev: System One Routing for Image Editing Models 

**Authors**: Zhile Guan, Xu Jiang, Yule Duan, Gongbo Zhang, Shuyu Wang, Xiao Chen, Jian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.32635)  

**Abstract**: Instruction-based image editing models exhibit pronounced sample-specific performance variation: different models can produce substantially different results for the same source image and editing instruction, and the best-performing model often varies across requests. These models also differ substantially in inference latency, making fixed-model deployment inefficient. We therefore study image editing model routing, which dynamically selects an editor from a candidate pool given an input image and editing instruction. We introduce EditJev, a System One routing model that directly predicts sample-level preference scores over candidate image editing models in a single forward pass. A lightweight preference head on a vision-language backbone produces a fixed-dimensional score vector without autoregressive model-name generation or response parsing. To train EditJev, we construct pairwise preferences from offline candidate outputs and optimize a pairwise ranking objective to learn relative model suitability. At inference time, EditJev combines the predicted preference scores with model-level speed scores to balance editing quality and generation latency. Experiments on ImgEdit and GEdit-En show that EditJev improves aggregate quality over the strongest fixed editor, from 4.540 to 4.592 and from 8.194 to 8.255, respectively, while reducing estimated end-to-end latency by 60.8% and 31.7%. These results demonstrate the effectiveness of direct, structured model routing for efficient image editing. 

---
# Interpretable Singing Emotion Recognition Using Acoustic and Vocal Technique Features 

**Authors**: Fotini Kiriakopoulou, Vassilis Lyberatos, Spyridon Kantarelis, Athanasios Voulodimos, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2610.32629)  

**Abstract**: Singing emotion recognition relies predominantly on deep learning models that achieve strong performance but offer limited insight into the vocal characteristics driving emotion, of- ten overlooking the singing voice as a distinct signal. We propose an interpretable singing emotion recognition framework using a compact set of singing-specific descriptors across prosody, spec- tral features, phonetics, and vocal technique. To decode model decisions and link predictions to measurable acoustic phenomena, we employ SHAP and apply GRAD-CAM to qualitatively assess the learned representations of vocal techniques. Evaluations show that singing-specific features reduce cross-singer variability, while vocal technique descriptors provide interpretable insights into singing expression. Furthermore, out-of-domain evaluation on VocalSet reveals interesting associations between emotional expression and ground-truth technical execution. The analysis highlights vibrato and pharyngeal techniques as salient cues of emotional delivery, with SHAP indicating that vibrato is context-dependent on articulation rate. These findings emphasize the value of domain-specific features for interpretable singing emotion analysis. 

---
# Tool-Calling LLM Agents Under Realistic MCP Faults: Scale Does Not Purchase Fault Tolerance 

**Authors**: Antonio Ken Iannillo, Joshua Segun Owotogbe, Roberto Natella, Francesco Avallone, Kristina Kudryavtseva, Indika Kumara  

**Link**: [PDF](https://arxiv.org/pdf/2610.32555)  

**Abstract**: LLM agents increasingly rely on external tools exposed via the Model Context Protocol (MCP), yet current benchmarks assume tools respond correctly and on time, leaving agent resilience to realistic tool failures untested. We introduce MCP-Injector, a drop-in, protocol-aware fault-injection middleware that interposes on the MCP tool-call path with controlled persistence (permanent, transient, intermittent) and seedable schedules. Its fault model is calibrated to the observed incident distribution (failure mode x persistence) of the public MCP ecosystem, mined from 57,240 issue and pull-request threads across 1,774 open-source MCP repositories. In a 1,350-cell experiment crossing two backbone sizes (20B, 120B), three agent stacks (custom multi-round plan-and-execute, OpenAI Agents SDK, LangChain), and three fault injection conditions, we find that faults drawn from this permanent-heavy stress profile cause severe quality degradation (task-completion score falls by 60%, from 8.08 to 3.23 out of 10), and that scale does not purchase fault tolerance for the two gpt-oss backbones we test: the 120B retains no larger share of its fault-free score under faults (resilience-ratio model, $p = 0.96$), and its completion rate under faults is, if anything, lower (91.1% vs. 95.6%; difference 4.4 points, 95% CI $[-0.2, 9.3]$). Agent stacks differ in how well they absorb faults, which suggests that framework-level error handling matters more than model scale. These results demonstrate that capability benchmarks alone can mask operational brittleness, motivating protocol-aware fault injection as a complementary evaluation dimension for tool-calling agents. 

---
# StarRTT: An End-to-End Real-Time Framework for Blind Zone-Free RTT Prediction in Starlink LEO Satellite Networks 

**Authors**: Jaeyong Lee, Seongjin Jeong, Yusang Park, Hyunho Hwang, Sangmin Han, Wonjae Shin  

**Link**: [PDF](https://arxiv.org/pdf/2610.32540)  

**Abstract**: Low Earth Orbit (LEO) satellite mega-constellations such as Starlink require frequent handovers due to high orbital velocities, resulting in substantial Round-Trip Time (RTT) fluctuations. Accurate RTT prediction can mitigate these fluctuations, yet it fundamentally relies on continuously updated serving- satellite information. Existing serving-satellite identification methods typically complete identification only at the end of each scheduling cycle, creating an identification blind zone that limits real-time prediction. To overcome this limitation, we develop StarRTT, an end-to-end real-time framework that eliminates the blind zone by integrating per-second serving-satellite identification with RTT prediction. Within this framework, we propose Real-Time Long Short-Term Memory (RT-LSTM), an LSTM-based RTT prediction model designed to capture both non-stationary RTT evolution and time-varying geometric relationships among the serving satellite, user terminal, and gateways. Specifically, a historical LSTM encodes historical RTT observations and real-time serving-satellite and gateway-relative features, while a future LSTM processes Two-Line Element (TLE)-propagated future features of the identified serving satellite up to the next scheduling instant. Under strict real-time back-testing, RT-LSTM achieves a Mean Absolute Error (MAE) of 1.405 ms averaged over all 15 prediction horizons and 15 reference time-slot indices, reducing the error by 4.3 %, 11.6 %, and 22.1 % compared with the existing baseline models, LSTM, StarNet, and Attention-based Spatiotemporal Model (ASTM), respectively. Notably, the reduction over LSTM reaches 12.5 % at the scheduling instant, highlighting the benefit of real-time serving-satellite information when recent RTT observations are limited. 

---
# Analysing Austrian Parliamentary Speeches by Predicting Party Affiliations 

**Authors**: Sarah Sulollari, Vanja Karan, Benjamin Roth  

**Link**: [PDF](https://arxiv.org/pdf/2610.32515)  

**Abstract**: This paper examines speech characteristics of the Austrian National Council members (CMs). We introduce analysis methodology based on a fine-tuned BERT-Model which assigns to every speech a probability distribution over Austrian parties. These are then aggregated at the speaker and party level. We perform an extensive evaluation to ensure the quality of the generated distributions prior to further analysis. Instead of considering mis-assignments of predicted parties as a performance issue, we leverage them to derive insights about the CMs and political developments. Our findings show that parties that govern together become more similar in their speech characteristics, while, analogously, parties that are in opposition move further away from governing parties. Moreover, CMs who switch party usually adapt their speech characteristics only after the switch occurs. Finally, we publicly release a data set containing party affiliation predictions for the speeches of the Austrian National Council from 1996 to 2018. 

---
# MusicTwins: Training Audio-Language Models to Listen like Their Persona 

**Authors**: Milan Liessens Dujardin, Kevin Miao  

**Link**: [PDF](https://arxiv.org/pdf/2610.32495)  

**Abstract**: People differ in how they perceive music, what associations they make with it, what they like about it, and how they respond to it. Computational models intended to simulate people in environments that include music should account for these differences. Although LLMs have been shown to reproduce aspects of human behavior, whether audio-language models (ALMs) can reproduce a specific person's responses to textual and musical stimuli remains underexplored. We introduce MusicTwins, a framework that combines persona summaries, personal memory, and post-training to generate person-specific responses. We construct persona representations from human writings, capturing background, views, personality, writing style, and a musical profile, and post-train Audio Flamingo Next through supervised fine-tuning and reinforcement learning. Evaluation on ten held-out personas shows notable improvements for Audio Flamingo Next in persona fidelity, including personality, writing style, and responses to music, supporting MusicTwins as a foundation for training audio-language models to listen like their persona. 

---
# Efficient Zero-Shot Video Anomaly Detection via Signal-Aware Adaptive Fusion (SAAF) 

**Authors**: Khaled Waleed Dawoud, Muhammad Haris Khan  

**Link**: [PDF](https://arxiv.org/pdf/2610.32486)  

**Abstract**: Deploying Vision-Language Models for video anomaly detection confronts a fundamental constraint. Their semantic reasoning demands computational budgets incompatible with the scale of real-world video streams. We present Signal-Aware Adaptive Fusion (SAAF), a framework that achieves efficient zero-shot anomaly detection through three key innovations. First, we introduce a dual-signal adaptive sampler operating in semantic embedding space: a delta signal captures sudden events via frame-to-frame cosine distance, while a drift signal detects static and gradual anomalies through deviation from rolling temporal context. Their OR-gated combination with adaptive EMA-based thresholding provides comprehensive anomaly coverage without training data. Second, we propose a threshold-gated cascaded architecture where lightweight frame-level scoring precedes expensive video-level temporal analysis, invoking it only when confidence warrants additional context. Third, we design a signal-aware weighted fusion mechanism that dynamically adapts score combination based on which signal triggered frame selection. Across four benchmarks, SAAF delivers competitive accuracy at substantially lower cost. Among training-free zero-shot methods, SAAF sets a new state of the art on XD-Violence (92.07%), MSAD (86.70%), and ShanghaiTech (80.54%) AUC-ROC, with XD-Violence and MSAD also surpassing supervised baselines. SAAF sustains 44.95, 48.61, and 41.06 FPS on XD-Violence, UCF-Crime, and MSAD, a 2.43$\times$, 2.63$\times$, and 2.22$\times$ throughput improvement over the prior best zero-shot method. Code: this https URL 

---
# Beyond Copying: Lifelong Smart Contract Security Task Construction via Protocol Mutation 

**Authors**: Zifan Peng, Peng Huang, Jingyi Zheng, Yingjie Xue, Xuechao Wang, Xinlei He, Xia Hu, Dongrui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.32453)  

**Abstract**: AI agents can inspect code and run tests to support smart contract security. SlowMist reports about $38.6 billion in losses across the broader blockchain ecosystem. Existing evaluations can reward memorized or copied answers, including attack code recovered from public blockchain transactions. Reconstructing contracts and validating tasks also require costly manual effort. To address these issues, we propose CONSTRAINTDIAL, a framework for lifelong construction of smart contract security tasks via protocol mutation. It automates local contract reconstruction and uses attack execution to select changes to the required calls or inputs. Execution checks guide revisions that make the original attack fail while preserving exploitability, the security objective, and normal functionality. New incidents and audit findings enter the same procedure. With CONSTRAINTDIAL, we construct 3,045 task inputs from 203 public cases, covering detect, exploit, and patch. Compared with reconstruction alone, protocol mutation reduces success with supplied public answers by 10.9 percentage points on average across detect, exploit, and patch. The fraction of tasks solved by copying online answers falls by 9.3 points. 

---
# AutoMIL: Automated Multiple Instance Learning for Whole-Slide Image Classification 

**Authors**: Jonas Waibel, Frank Kramer, Dominik Müller  

**Link**: [PDF](https://arxiv.org/pdf/2610.32452)  

**Abstract**: Developing multiple instance learning pipelines for the classification of whole-slide images involves many interdependent design decisions and expertise in both pathology and machine learning. We present AutoMIL, an open-source framework that applies principles from the field of automated machine learning to tackle this task by automating integral pipeline steps, such as preprocessing, training and evaluation. In a benchmark evaluation with default settings, AutoMIL reached ROC-AUC scores of 0.94 on UBC-OCEAN, 0.99 on PCam and 0.93-0.96 on a TCGA-NSCLC subset, comparable to specialised pipelines in related work. Ensembling mainly improved performance on imbalanced data, showing that an automated pipeline must adapt to dataset characteristics. AutoMIL thus provides an accessible baseline for whole-slide image classification without manual pipeline configuration. 

---
# DIGOM: Disentangled Interaction Graphs for Outcome prediction in Multimodal healthcare 

**Authors**: Hakima Laribi, Olivier Lefebvre, Dan Poenaru, Martin Vallières  

**Link**: [PDF](https://arxiv.org/pdf/2610.32451)  

**Abstract**: Multimodal healthcare data contain relational information that extends beyond individual patient observations. Effective clinical outcome prediction therefore requires not only integrating information from multiple modalities, but also modeling the relationships between them. Standard fusion approaches do not explicitly capture these interactions, while existing graph-based approaches typically model only a subset of them. We propose DIGOM (Disentangled Interaction Graphs for Outcome prediction in Multimodal healthcare), a graph-based framework that jointly models three complementary interaction types: Intra-patient cross-modal, Cross-patient same-modal, and Cross-patient cross-modal interactions. DIGOM learns a sparse multiplex graph with disentangled views, each encoding interactions between pairs of modalities across patients. Discrete adjacency matrices with patient-adaptive connectivity are sampled via a differentiable mechanism, enabling end-to-end learning and allowing the model to select informative relationships without imposing a fixed number of neighbors. A regularization objective maintains overall graph sparsity, while message passing integrates this relational context into patient representations for outcome prediction. We evaluate DIGOM on two multimodal healthcare datasets, MIMIC-IV and NIH-TB, across five clinical prediction tasks. DIGOM achieves the highest AUROC on all five tasks while using fewer connections than existing multi-interaction approaches. Ablation studies reveal that each interaction type's contribution varies across tasks, highlighting the need to adapt relational structures per task. DIGOM also demonstrates greater robustness to missing modalities, suggesting learned relationships provide complementary information when modalities are unavailable. Post-hoc analyses further reveal task- and outcome-specific connectivity patterns in the learned graphs. 

---
# Decoupling Patient Personas and Clinical Workflows: A Dual-Agent Approach to Realistic Medical Dialogue Generation 

**Authors**: Qiyuan Li, Xiaomei Guo, Chao Gao, Haijiang Liu, Deyu Chen, Caicai Guo, Jinguang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2610.32443)  

**Abstract**: Pre-consultation medical dialogues are essential for collecting comprehensive patient information, but are costly to obtain at scale due to privacy and ethical constraints. Existing LLM-based synthesis methods often rely on prompt engineering or rigid rules, limiting their ability to balance clinical relevance with realistic patient diversity. We propose CoRPS-Dial, a controllable dual-agent framework for pre-consultation medical dialogue synthesis that integrates patient personas with guideline-aligned clinical workflow rules. CoRPS-Dial generates dialogues through interactions between a persona-driven patient agent and a process-guided doctor agent, followed by post-hoc quality control for role consistency, workflow completeness, and textual consistency. Experiments on coronary heart disease demonstrate that CoRPS-Dial improves alignment between synthetic and real dialogues in information coverage, clinical compliance, and dialogue naturalness. Downstream fine-tuning experiments further confirm the utility of the generated synthetic data for medical dialogue tasks. These results highlight the effectiveness of CoRPS-Dial for clinically grounded and controllable medical dialogue synthesis. 

---
# Cycle-of-Thought: Self-Evaluative Iterative Reasoning for Table Visual Question Answering in Document Images 

**Authors**: Somraj Gautam, Gaurav Harit  

**Link**: [PDF](https://arxiv.org/pdf/2610.32423)  

**Abstract**: Recent advances in Vision-Language Models (VLMs) have significantly improved Visual Question Answering (VQA) performance, yet their reasoning over structured content such as tables remains inconsistent, often plagued by spatial misalignment and logical hallucinations. Existing approaches for improving table VQA typically rely on architectural modifications or expensive supervised fine-tuning (SFT), which impose high barriers to entry due to the need for large-scale, expert-annotated datasets and significant computational resources. In this work, we propose Self-RefineVQA, a training-free, inference-time self-correction framework that improves table VQA performance without additional labeled data or model updates. Inspired by human revision behavior, our method enables a single VLM to iteratively act as a generator, critic, and refiner, forming a closed self-feedback loop that corrects common reasoning errors through prompt-guided re-evaluation. Experiments on tabular VQA benchmarks demonstrate that Self-RefineVQA yields consistent accuracy gains in monolingual and cross-lingual understanding, with a particularly significant boost for small models, enabling lightweight models (e.g., 2B-7B parameters) to rival the performance of significantly larger counterparts. Furthermore, we provide a comparative complexity analysis demonstrating that while inference cost increases linearly, our framework offers superior data efficiency and a lower cumulative cost by eliminating the prohibitive annotation bottleneck of human labeling required for fine-tuning. Our results highlight the potential of prompt-based self-improvement as a scalable, accessible paradigm for reliable table reasoning in VLMs. 

---
# PASTA: Prompt-Aware Annealed Diffusion Sampling 

**Authors**: Ren Ajiki, Ryugo Morita, Stanislav Frolov, Brian Bernhard Moser, Ko Watanabe, Riku Takahashi, Andreas Dengel, Jinjia Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.32390)  

**Abstract**: While Diffusion models produce high-quality images, real-world visual design requires intentionally preserving clean regions for headlines, logos, or UI elements. Existing layout-aware text-to-image methods struggle to strictly preserve reserved regions, often introducing unwanted content that complicates downstream editing. Existing approaches either manipulate attention to displace content or suppress conditioning via noise-level interventions. However, these methods sacrifice visual coherence, producing flat regions or noticeable style inconsistencies between reserved and non-reserved areas. To address this limitation, we propose a Prompt-Aware Shared-Latent Training-Free Annealed Diffusion Sampling (PASTA), a training-free framework that enforces reserved-region constraints by selectively applying semantic guidance to object-allowed regions during sampling. PASTA operates on a shared latent trajectory, where full-prompt and context-prompt guidance are fused through mask-guided sampling. An annealed schedule suppresses semantic leakage during early denoising stages while gradually restoring visual consistency in later stages. Extensive experiments demonstrate that PASTA improves reserved-region fidelity while preserving overall image realism and perceptual consistency. 

---
# TENET: Time-point Encoding Network for Multivariate Time Series Anomaly Detection 

**Authors**: Dongchan Cho, Juwon Hwang, Keumyeong Kang, Jiho Han, Namsoon Jung  

**Link**: [PDF](https://arxiv.org/pdf/2610.32337)  

**Abstract**: In unsupervised multivariate time series anomaly detection (MTSAD), local variation can inflate reconstruction and forecasting errors, making observation-space discrepancy an unreliable indicator of abnormal system behavior. Since latent-space scoring can mitigate this limitation, we reframe one-class MTSAD as current-point system-state compatibility estimation and propose TENET, the Time-point Encoding Network for Multivariate Time Series Anomaly Detection. We define the current-point system state as a latent representation of the target time point, constructed from variable-level temporal context in the input window. TENET constructs this state through a temporal layer that retrieves window context relative to the current point, encodes time with current-point-centered embeddings, and adaptively mixes the retrieved context with the current representation. With this design, TENET treats the input window not as the object to reconstruct, forecast, summarize, or score, but as context for constructing a latent representation evaluated with a fixed standard-normal negative log-likelihood score. Across benchmark datasets, TENET achieves state-of-the-art performance. Additional experiments further show that TENET remains effective under alternative one-class objectives and outperforms substitutions based on generic sequence encoders, suggesting that its advantage comes from decision-aligned representation construction rather than from a specialized scoring rule. These results indicate that one-class MTSAD can be strong when the latent representation is designed around the current time point. 

---
# Data-Efficient Backdoor Attacks on Code Models via Trigger-Neutralized Counterfactual Scoring 

**Authors**: Yu-Yu Lee, Ming-Feng Tsai, Tomohiro Morikawa, Chun-I Fan  

**Link**: [PDF](https://arxiv.org/pdf/2610.32334)  

**Abstract**: Pre-trained code models are exposed to backdoor poisoning, in which crafted training samples make the model misbehave on a trigger. A capable attacker minimizes the number of such samples, since each adds cost and exposure and a lower poisoning rate leaves a smaller footprint for defenses. Which samples to poison therefore matters, but existing data-efficient selection methods target image classification, offer no scoring mechanism for autoregressive generation, and have not been studied on code. We propose Trigger-Neutralized Counterfactual Scoring (TNCS), which isolates a trigger's contribution to the target behavior by comparing the scoring model's output on a triggered input with its output on a trigger-neutralized counterfactual, covering classification and generation alike. TNCS attains higher attack success rates than the baselines at very low poisoning ratios, across three CodeXGLUE tasks and five trigger families, without degrading clean-task performance, and its margin grows as the ratio decreases. It is the only method effective on every trigger family, and its selected poison set transfers to mismatched victims. 

---
# 4DMemory: Benchmarking Long-Horizon Spatiotemporal Memory with Harness-Augmented Agents 

**Authors**: Yanhao Jia, Yi Zhu, Yiran Zhong, Weigao Sun, Feida Zhu, Pengkun Jiao, Sihan Yang, Ming Ma, Steven Hoi, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2610.32332)  

**Abstract**: Humans efficiently process multimodal streams spanning hours to years, building persistent, fine-grained spatiotemporal memories that support reasoning about and predicting long-horizon events. In contrast, existing benchmarks focus on short- and medium-context tasks, and most foundation models struggle to preserve fine-grained information over long horizons. To evaluate and improve long-horizon spatiotemporal reasoning, we introduce the LongVideo dataset and the 4DMemoryBench. LongVideo comprises about 900 hours of video and over 5,000 manually reviewed questions covering visual, textual, and spatiotemporal reasoning across diverse context lengths. We further develop the 4DMemory Harness, which integrates four complementary skills, 22+ specialized tool models, and a model-specific retrieval-augmented generation system. Evaluating 13 leading models, we find that standalone foundation models struggle with long-horizon tasks, while the 4DMemory Harness improves all of them by over 50\%. Compared to previous benchmark, the same models exhibit a performance drop of more than 10\% on 4DMemoryBench. We hope 4DMemory will facilitate the development of spatiotemporal agents capable of addressing complex, long-horizon tasks. 

---
# ChoreoStudio: Planning What, When, and Who in Music-Driven Group Choreography 

**Authors**: Jing Xu, Cunjian Chen, Qiuhong Ke  

**Link**: [PDF](https://arxiv.org/pdf/2610.32325)  

**Abstract**: Music-driven group dance generation must decide not only how each dancer moves, but also what formation the group should form, when it should change, and who should execute each transition. Existing methods can generate orderly and rhythmically synchronized group motion, but remain limited in formation evolution and inter-dancer interaction. In human choreography, formation transitions often align with changes in high-level musical structure, such as the progression from an introduction to a chorus or from a chorus to a climax. We present ChoreoStudio, which makes choreography planning explicit. A music disentangler first decomposes the audio into frame-level rhythmic features and segment-level structural features. Given the segment-level features and sparse labels, a frozen omni-modal model acts as a scriptwriter, producing a global semantic description that guides the Director's choreography planning. Based on this context, the Director produces three coupled interaction decisions: a centroid-relative point set specifies what formation to produce, a transition gate determines when it should change, and a mobility allocation indicates who should execute the transition. The transition gate and mobility allocation are further conditioned on the planned geometric change, coupling formation targets with their temporal and dancer-specific execution. At a finer temporal scale, frame-aligned music features condition the diffusion generator to control detailed body motion and preserve beat-level alignment. The segment-level interaction plan is injected in parallel through frame-aligned instruction tokens and cross-dancer attention biases, coordinating how the generator realizes each transition. Experiments on AIOZ-GDance demonstrate that ChoreoStudio follows the planned interactions while achieving state-of-the-art performance in motion quality, group organization, and formation dynamics. 

---
# Complementing DINO Features with Image Structure for Part Discovery 

**Authors**: Samyak Rawlekar, Nikhil Chowdary Paleti, Amey Gupta, Narendra Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2610.32293)  

**Abstract**: Self-supervised Vision Transformers (ViTs) like DINO encode rich semantics, enabling training-free segmentation of objects from the background and discovery of their parts via feature clustering. This paper focuses on part discovery: partitioning a foreground object into semantic parts. Existing approaches produce imprecise part boundaries. We trace this to the rigid ViT patch grid used in training DINO, where patches straddling boundaries contain pixels from multiple parts. While DINO's invariance-based training could absorb this pixel-level mixing into clean features, experiments with over 26K foreground patches show it does not: boundary-overlapping patches produce features with higher entropy and lower assignment confidence, blurring predicted part boundaries. Rather than make the massive effort of retraining DINO over geometrically coherent regions, an easier approach is to take boundaries directly from a boundary detector, which supplies the geometric information DINO's features lack. Motivated by this, we introduce Part-DINO, a training-free framework that combines boundaries from low-level image segmentation with DINO's semantic features. We use the boundaries to partition the image into regions, pool DINO features within each region, and group them to discover parts. Without any training, Part-DINO produces structurally coherent part assignments, outperforming training-free baselines across four standard benchmarks (CUB, CelebA, PartImageNet-OOD, PartImageNet-Seg) and remaining competitive with weakly-supervised and unsupervised methods. 

---
# "Not" is Not Lost in Translation: A Causal Analysis of Negation Representations in Multilingual LLMs 

**Authors**: Giang Son Nguyen, Wray Buntine, Dung D. Le, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.32281)  

**Abstract**: Negation occurs in every language, yet its surface realization differs drastically. Despite this, multilingual large language models (LLMs) translate negation robustly, raising the question of whether they share an underlying representation of negation across languages. We study this question in six open models across English, French, Vietnamese, and Chinese, using causal interventions on a synthetic contrastive dataset and a negation translation dataset derived from FLORES-200. Across models, we find strong evidence that negation representations are causally controllable and largely shared across languages. In particular, we show that a contrastive activation direction extracted from one language can induce or remove negation in the others, and directions extracted from different languages are substantially aligned. We further find that negation is represented in sparse features: a single sparse autoencoder (SAE) feature per model suffices to control polarity across languages, while responding selectively to negated inputs. Finally, both dense and sparse interventions generalize to four additional languages and FLORES-derived sentences. We release data and code to support future work: this https URL. 

---
# Tokamak-SI: Interpreting Tokamak Discharges with Labeled Catalogs 

**Authors**: Nathaniel Chen, Alvin Garcia, Ricardo Shousha, Jalal Butt, Azarakhsh Jalalvand, Egemen Kolemen  

**Link**: [PDF](https://arxiv.org/pdf/2610.32264)  

**Abstract**: Finding which phenomena occur in a tokamak discharge, and when, is often a manual task. Previous efforts at labeling events have been task-specific, each with its own corpus of data. We introduce Tokamak-SI, which combines the labels and labeling methods of many earlier studies into a single scalable, time-aligned, and categorical dataset covering observations from confinement, detachment, alfven eigenmodes (AE), edge localized modes (ELM), sawtooth, tearing modes (TM), and resistive wall modes in DIII-D tokamak. Models based on this, which can extend the labels to new shots, match or beat earlier detectors on its labels, including for AE, ELM, and TM detection. Tokamak-SI is intended to enable task-specific benchmarking, physics-searchable archives, and direct identification of key events during an experiment. 

---
# Beyond Pattern Matching in 3D MLLMs: Part-Level Functional Reasoning for Generalizable Recognition 

**Authors**: Weihong Li, Hao Jiang, Yaohua Zha, Tao Dai, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2610.32262)  

**Abstract**: 3D MLLMs recognize objects via pattern matching, which fails when geometrically similar objects require functional reasoning to disambiguate (e.g., dresser vs. wardrobe). Our per-sample Geometric Confusion Score (GCS) exposes this failure: baseline accuracy is 33.7% on ModelNet40's most ambiguous 20% versus 84.0% on its least ambiguous 20%. We identify a function-as-annotation pattern, in which functional language decorates a known identity instead of supporting inference, and hypothesize that reversing this information flow should concentrate improvements on the most ambiguous samples. PFIR (Part-level Functional Inference for Recognition) reverses that flow. Built on PartNeXt part segmentation, its data pipeline produces 507K QA pairs with functional CoTs (geometry $\to$ function $\to$ identity), and its progressive curriculum culminates in dual-objective GRPO: strict functional reasoning and descriptive grounding for Objaverse's diverse distribution. The two RL objectives yield nearly orthogonal task vectors ($\cos = 0.056$), enabling composition via Task Arithmetic. Under a strict dual OOD protocol (Objaverse $\to$ ModelNet40; XYZRGB $\to$ XYZ-only), PFIR raises zero-shot accuracy from 60.5% to 66.7%, with gains concentrated $2.6\times$ more on geometrically ambiguous samples---supporting part-level functional reasoning as an effective route to more generalizable 3D recognition. Code: this https URL. 

---
# Heterogeneous Space Data Centers: A Key Paradigm for Future Non-Terrestrial and In-Orbit Services 

**Authors**: Peng Hu, Mustafa Alhassan, Amjad Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2610.32260)  

**Abstract**: Lower space-launch cost and advances in AI are driving space-based data centers (SDCs) that provide AI-centric computing and communication services to in-orbit and non-terrestrial assets. However, architectural principles, service models, and energy provisioning for such SDCs, particularly in low-Earth orbit (LEO), remain largely unexplored. We introduce heterogeneous SDCs composed of large and small nodes, differentiated by computing capability, size, and power budget, to support applications spanning ground, aerial, and in-orbit users. We define use cases, node sizing and placement strategies, and interconnection topologies integrating space-to-space and space-to-ground links. Focusing on dawn-dusk LEO configurations, we analyze solar power availability, visibility windows, and latency-aware coverage. Using baseline workloads for typical scenarios, we evaluate latency and service reliability. We quantify energy requirements of different SDC roles and deployment scales, highlighting trade-offs among node size, orbit selection, and service portfolio, and provide design guidance and future research directions for AI infrastructure in space. 

---
# Representation Learning via Composite Subspace Flow Matching 

**Authors**: José Afonso, Mário Figueiredo  

**Link**: [PDF](https://arxiv.org/pdf/2610.32256)  

**Abstract**: Distilling semantic meaning from raw image data remains a challenge in machine learning. Current self-supervised paradigms face a trade-off: joint-embedding architectures produce relevant features by aggressively discarding intra-class variance, irreversibly spatial and reconstruction. Conversely, generative models preserve spatial details but struggle to isolate semantics, as their optimization is dominated by pixel variations. To bridge this divide, we propose composite subspace flow matching (CSFM). Rather than corrupting data with unstructured isotropic noise, CSFM formulates representation learning as the variance-minimizing inversion of a deterministic degradation curriculum across spectral, color, and illumination subspaces. By bounding trajectory variance with hard temporal style truncation, a capacity-constrained encoder is driven to ignore transient style artifacts and encode persistent content. Empirical evaluations across CIFAR-10, CIFAR-100, and STL-10 show that CSFM yields competitive discriminative transferability while preserving generative synthesis across both convolutional and vision transformer architectures. By avoiding the collapse pathologies of non-generative methods, CSFM forms a smooth latent space that outperforms standard optimal transport flows on linear probing and few-shot classification, while enabling coherent synthesis and trajectory interpolation. These results confirm that steering the physical dynamics of generative flows offers a scalable paradigm for learning representations that are simultaneously linearly separable and generatively invertible. 

---
# Anti-coherence Regularized Absolute Scaling for Out-of-Distribution Detection 

**Authors**: Donghwan Kim, Youngwoong Jin, Hyunsoo Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2610.32255)  

**Abstract**: Out-of-distribution (OOD) detection is recognized as a critical problem in AI safety. SCALE is an effective post-hoc OOD detection method in the activation-shaping family that requires no training data and rescales penultimate activations using a percentile-based magnitude ratio. However, this ratio is ill-defined when negative activations are present, since positive and negative contributions cancel within the underlying sum, causing SCALE to degenerate on backbones such as ViT. We address this with an Absolute Fix that computes the ratio over absolute-valued activations rather than discarding negative activations through ReLU. Building on this corrected scale, we propose Anti-coherence regularization, which penalizes the scaling factor by the fraction of a sample's max-logit contributions to the max logit that are sign-mismatched and thereby suppresses unreliable evidence for the predicted class. Despite requiring no access to training data or training labels, our method ties or outperforms activation-shaping methods that rely on training data in terms of AUROC and FPR@95 on CIFAR-10 and ImageNet under the OpenOOD v1.5 protocol. 

---
# Easy-3Dfy: Object-Level Geometric Composition for Single-Image 3D Scene Construction 

**Authors**: Ziang Zhang, Wenyou Yi, Zehan Wang, Jiachen Xu, Xintong Han, Chunchao Guo, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.32250)  

**Abstract**: Single-image 3D scene construction is a fundamental problem in 3D vision. Recent methods such as SAM-3Dfy enable end-to-end reconstruction of 3D assets and their scene layouts, but still struggle with fine-grained scene geometry and asset quality. Meanwhile, specialized 3D object generators provide detailed object geometry, while 3D foundation models offer strong and generalizable spatial understanding. We propose Easy-3Dfy, a compositional workflow that decouples object generation from scene layout estimation, allowing specialized models to fully exploit their respective strengths. An RPF-based aligner then bridges these capabilities by establishing geometric correspondence between object-local meshes and partial scene pointmaps, recovering object layout for scene composition. Without requiring complete scene-level supervision, the proposed workflow consistently outperforms SAM-3Dfy across multiple geometric metrics on Scene-Maker and 3D-FUTURE200. Our results demonstrate that explicit geometric composition can effectively combine high-quality object generation with strong spatial understanding, while enabling the workflow to benefit independently from advances in both 3D object generation and foundation models. 

---
# Head Dimension Matters in Language Models for Long-Context Retrieval 

**Authors**: Seiya Tokui, Shoichiro Yamaguchi, Sosuke Kobayashi  

**Link**: [PDF](https://arxiv.org/pdf/2610.32213)  

**Abstract**: Multi-head attention retrieves and combines context representations through query--key interactions and value aggregation, with head dimension constraining the per-head representation space. In long-context processing, it remains unclear whether allocating a fixed parameter budget to fewer, wider heads improves information retrieval. In this paper, we investigate how head dimension affects the performance of long-context retrieval tasks. Our empirical studies reveal that pretraining loss and short-context non-retrieval task performance are not affected much by head dimension, while long-context retrieval performance improves with wider heads at fixed model sizes. We find that models with wider heads have a higher proportion of retrieval heads, associating more parameters with retrieval. Our ablation studies also show that the retrieval performance improves with wider heads even when the projection weights in attention modules are frozen during long-context continual pretraining, suggesting that wider heads not only benefit updates in attention modules but lead to more effective adaptation of components such as embeddings and MLPs. 

---
# SOPO: Socratic Guided Policy Optimization for Span-Level Hallucination Detection 

**Authors**: Jiawei Dong, Zilong Bai, Pengtian zhu, Peng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.32212)  

**Abstract**: Large Language Models (LLMs) have gained widespread adoption in various natural language processing tasks, but suffer from hallucination issues where they generate unfaithful or inconsistent content. While Reinforcement Learning with Verifiable Rewards (RLVR) has shown promising applications in reasoning tasks, its application to span-level hallucination detection is challenged by localization ambiguity, reward sparsity on hard examples, and uncontrolled length inflation. To address these issues, we introduce Socratic Guided Policy Optimization (SOPO), a multi-turn RL framework for robust hallucination detection. The key innovation of SOPO is a Socratic feedback mechanism, where a teacher model guides the reasoning trajectories during rollout by providing indirect hints inferred from the ground truth. This teacher-guided approach enables the model to generate more effective reasoning trajectories for hard examples, thus alleviating reward sparsity. Furthermore, we incorporate Turn-based Advantage Scaling to penalize verbosity bias, encouraging the model to develop efficient and autonomous reasoning capabilities. Empirical results on RAGTruth demonstrate that SOPO (4B) achieves state-of-the-art performance, surpassing leading proprietary models (e.g., GPT-5) and standard RL baselines, while obtaining higher reported mean scores than GRPO on four out-of-domain benchmarks, with mixed evidence against CAPO. We release our training and evaluation code alongside the 1.7B and 4B SOPO models at this https URL. 

---
# Do VLAs Know They Will Fail? Interpretable Success and Failure Patterns in Physical AI Systems 

**Authors**: Christian A. Rogers, Qian Lin, Daniel S. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2610.32208)  

**Abstract**: Vision-Language-Action (VLA) models have emerged as powerful general-purpose robot policies, but their successes and failures remain difficult to diagnose and understand. In this work, we show that sparse autoencoder (SAE) representations of OpenVLA activations contain outcome-predictive information and can be used to predict task failure before the robot reaches an unrecoverable state. Moreover, to provide insights into what common behaviors may contribute to these outcomes, we identify individual SAE features whose strong activations are associated with specific stages of success or failure, and uncover recurring behavioral patterns by localizing temporal intervals where these features exhibit significant activation changes. We further develop a VLM-based interpretation pipeline to generate behavioral descriptions of these patterns and validate their correspondence to feature activations on held-out intervals. 

---
# Principia: Test-Time Hypotheses to Principle Construction for Scientific Discovery 

**Authors**: Dongki Kim, Sangwoo Park, Jinheon Baek, Soyeong Jeong, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2610.32195)  

**Abstract**: Scientific discovery requires identifying the scientific principles underlying observed phenomena to guide further investigation. As these empirical principles are often unknown in complex discovery processes, they are constructed via a hypotheses-to-principle pipeline to obtain recurring causalities from progressively accumulated observations during test-time discovery. However, constructing this knowledge requires substantial domain expertise, time, and effort, making this translation stage a bottleneck for scientific discovery. To address this limitation, we propose Principia, an LLM-based framework that constructs and continually refines scientific principles at test time. Drawing on how human researchers develop principles, Principia forms comparative causal hypotheses from related inputs and their observed outcomes, then synthesizes recurring relationships across these hypotheses into inductive principles. By applying the constructed hypotheses and inductive principles to the task, it obtains new observations, revises the hypotheses used based on these new observations, and reconstructs principles from the updated hypotheses, allowing scientific knowledge to continuously evolve throughout test-time discovery. Across benchmarks spanning molecular optimization, protein engineering, gene perturbation, and molecular property prediction, Principia outperforms baselines relying on internal knowledge, external sources, or experience summaries, while generating principles that are plausible and consistent with the scientific literature, demonstrating its effectiveness in constructing and refining task-specific hypotheses and principles at test time for scientific discovery. Our code is available at this https URL. 

---
# Dual-Adaptive Z-Score PPO: Dynamic Clipping Using Action Confidence and Normalized Advantage Magnitude 

**Authors**: Taha M. Mahmoud, Naima Kaabouch  

**Link**: [PDF](https://arxiv.org/pdf/2610.32194)  

**Abstract**: Proximal Policy Optimization (PPO) remains a popular on-policy reinforcement learning algorithm owing to its stability and ease of implementation. Its clipping threshold, however, is fixed, so every training sample receives the same trust-region width regardless of how reliable the corresponding policy update is. This work examines adaptive-clipping variants of PPO in which the constant clipping parameter is replaced by sample-dependent thresholds computed from policy and return statistics. The central method, Dual-Adaptive Z-Score PPO (PPO-DAZ3), sets the clipping interval for each sample from two signals: action confidence, given by the probability of the selected action under the current policy, and update confidence, given by the normalized advantage magnitude. The rest of the standard PPO training pipeline is left unchanged. Evaluation on three classic control benchmarks (LunarLander-v3, CartPole-v1, and Acrobot-v1) indicates that adaptive clipping improves learning in some settings but not all. PPO-DAZ3 performs best on LunarLander-v3, the smooth advantage-based variant PPO-AZ3S leads on Acrobot-v1, and standard PPO stays competitive on CartPole-v1, suggesting that adaptive clipping is most beneficial on harder control tasks. 

---
# Uni-Stream: A Unified Streaming Speech Language Model for Speech Synthesis and Recognition 

**Authors**: Xuan Shi, Chang Zeng, Tiantian Feng, Ye Zhang, Jun Liu, Shih-Heng Wang, Lie Lu, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2610.32187)  

**Abstract**: Unified streaming speech language models aim to support speech recognition and synthesis within a single model, enabling real-time human-computer interaction. However, jointly supporting these two directions of speech-text conversion while preserving task-specific performance remains challenging. We propose Uni-Stream, a unified streaming speech language model built on compact pretrained speech models. Uni-Stream processes interleaved text and speech in a streaming format and achieves performance comparable to task-specific models through two-stage training. We also develop a data annotation pipeline for streaming training and will publicly release the generated transcriptions and word-level timestamps. We further examine Uni-Stream's representations across tasks and modalities and find that speech and text residence in distinct latent feature space even when trained in a unified model architecture, while representations within each modality become more task-dependent in later layers. 

---
# Searching with Sparse Histories: Cross-Task Architecture Search from Incomplete Search Logs and Zero-Cost Proxies 

**Authors**: Yuefeng Xu, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.32179)  

**Abstract**: Neural architecture search (NAS) is expensive, but cross-task transfer can reduce costs by reusing knowledge from previous tasks. However, existing methods struggle with sparse search histories and task-specific transfer failures. We propose a method, a risk-robust strategy that combines sparse source logs with zero-cost proxies through champion-based initialization and joint pairwise-ranker pretraining. On TransNAS-Bench-101, the proposal mitigates severe worst-case failures, reducing worst-10% ranks from 290 to 43 for Micro autoencoding and from 465 to 45 for Macro room layout. Across 13 cell-space targets, it outperforms a proxy-only ranker on 10 targets in both mean and worst-case performance, while maintaining competitive average performance. 

---
# MG-REPA: Multi-Granularity Representation Alignment for Generative Target Speaker Extraction 

**Authors**: Dohwan Kim, Jung-Woo Choi  

**Link**: [PDF](https://arxiv.org/pdf/2610.32161)  

**Abstract**: Generative target speaker extraction (TSE) models achieve high-quality extraction, but suffer from considerably slow training convergence. Representation Alignment (REPA) accelerates diffusion transformer training by aligning hidden states with a pre-trained Self-Supervised Learning (SSL) teacher. However, existing alignment in the speech and audio domain operates at a single granularity. For TSE, frame-level alignment leaves the global, time-invariant characteristics of the target speech implicit, whereas utterance-level alignment leaves the time-varying content implicit, even though extracting the target requires both. We propose Multi-Granularity Representation Alignment (MG-REPA), aligning the hidden states with an SSL model at the frame level and a speaker verification model at the utterance level. We further apply temporal normalization to the frame-level target, removing its utterance-level statistics so that each branch concentrates on its granularity. On Libri2Mix, MG-REPA reaches the OVRL and PESQ of a MeanFlow-TSE baseline with $3.0\times$ and $2.1\times$ fewer epochs, respectively. Trained for 750 epochs, it is competitive with the published 2,000-epoch baseline on intrusive metrics while improving non-intrusive metrics. An ablation demonstrates that MG-REPA outperforms alignment at a single granularity. 

---
# Sparsity-driven emergence of functional modularity in multitask neural networks 

**Authors**: Bowen Shi, Qi Ouyang, Long Qian  

**Link**: [PDF](https://arxiv.org/pdf/2610.32153)  

**Abstract**: Neural networks can learn multiple tasks, but how learning organizes shared computation into specialized and reusable components remains poorly understood. In this work, we show that sparsity inducing regularization promotes functional modularity in multitask networks without predefined module boundaries. Using functional profiling in multilayer perceptrons and a lightweight transformer, we identified compact, task-selective subnetworks, termed functional tokens. Across rule-based task families, selectively retaining these tokens preserves their associated functions while reducing performance on other tasks. Such specialization develops preferentially in deeper layers during training, while earlier layers retain shared computation. Regularization strength and form shape module compactness and specialization, while task weighting alters functional allocation. Moreover, extracted functional tokens can be recombined through alternative pathways to reconstruct a learned composite computation without retraining, preserving their connectivity and learned weights. Our results suggest that sparsity-constrained optimization shapes not only parameter use but also the organization and reuse of learned computation. 

---
# The LLENS Platform for On-Premises Large Language Model Agents in a Japanese University Hospital: Development and Evaluation Study 

**Authors**: Ken Enda, Jien Saito, Masami Mukai, Feng Han, Emir Turkes, Tatsumi Kusakabe, Reona Umeki, Kazuhiro Sumiyoshi, Mitsuya Numata, Takumi Tanikawa, Akira Endoh, Kenji Hirata, Atsushi Yoshida, Kohsuke Kudo, Shinya Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2610.32151)  

**Abstract**: Large language model agents that use tools could take on part of the electronic health record and desk work that occupies physicians' time, but in many hospitals the medical information systems run inside closed networks, so the model and the agent harness must also sit inside. We designed, implemented, and deployed LLENS, a clinical support AI agent platform that runs inside the closed network of Hokkaido University Hospital on one host with 8 NVIDIA H200 GPUs, the open-weight model Kimi K2.6 (~1 trillion parameters), and a web application rebuilt from Open WebUI. The harness has 10 components, including page image reading, full-text attachment input, a path-addressed document tree without a vector index, a network-isolated sandbox, artifact return, self-checking of generated output, subagents, and saved procedures. Kimi K2.6 was chosen from 3 open-weight candidates and scored 480/500 on the 2026 Japanese Medical Licensing Examination; it generated 119 tokens/s for one user and served 16 concurrent requests at 908 tokens/s in total. On 180 questions over hospital documents (3 seeds each), mean accuracy was 10.7% with no documents, 81.3% with fixed retrieval (top 10 chunks of one lexical search), and 97.2% with interactive access, in which the model reads with document tools and the sandbox (15.9 percentage points above fixed retrieval, 95% CI 10.9-21.3); the difference lay in filtering/counting questions (16.7% vs 87.8%). Five scenarios with synthetic data ran on combinations of the components: needlestick injury response, batch adverse event grading, scanned referral letter transcription, falls data aggregation and plotting, and investigation of 3 topics by subagents. The platform has run on the hospital network since September 14, 2026. 

---
# Implicit Neural Representations for Variational Problems on Graphons 

**Authors**: Taeyoung Kim, Jineon Baek, Joonkyung Lee, Hongseok Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.32147)  

**Abstract**: We propose a neural exploratory framework for variational problems over graphons, i.e., symmetric measurable functions $[0,1]^2 \to [0,1]$, that arise as limits of dense graph sequences. Such problems are central to two parts of the mathematical literature: extremal graph theory, which studies graph parameter optimisation under given restrictions, often for those graphs with a large number of vertices, and the large deviation theory of dense random graphs, which characterises the structure of rare events through constrained graphon optimisation. We represent graphons by implicit neural networks and optimise graphon objectives by gradient descent. Our design combines three ingredients: a multi-scale sinusoidal residual architecture biased toward sharp, step-like graphons; an embedded solver that enforces a single empirical density constraint and is differentiated by implicit differentiation; and symmetry-aware Monte Carlo estimators. On generalised Turán problems that previously required substantial human effort, our framework rediscovers known optimal graphons without human intervention. Applied to open instances, it produces candidate optima, including a previously unreported family of extremal structures. Applied to the variational problems from the large deviation theory of Erdős--Rényi random graphs, the framework produces new candidate optimal graphons across both upper- and lower-tail regimes for multiple pattern graphs; for the case of the triangle graph and the upper-tail regime, these candidates improve upon the best-known reference construction of Lubetzky and Zhao. The structures found by our framework have also led to new theorems on the optimal graphons for both families of problems. 

---
# Rethinking LoRA Initialization for Robust Asymmetric Learning Rates 

**Authors**: Disen Liao, Yaoliang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.32142)  

**Abstract**: Low-Rank Adaptation (LoRA) is often highly sensitive to learning-rate (LR) choices. Recent empirical studies suggest that, once LRs are carefully tuned, vanilla LoRA can remain competitive with many proposed LoRA variants. This shifts the central practical question from designing ever more variants to reducing the cost and brittleness of LR selection. LoRA+ addresses this issue with asymmetric LRs, using a larger LR for the up-projection matrix $B$ than for the down-projection matrix $A$. However, the ratio $\eta_B / \eta_A$ itself becomes an additional hyperparameter: large ratios can improve adaptation, but under the standard LoRA initialization they can also cause unstable training and ratio-dependent performance collapse. We identify the source of this brittleness through a signal--noise decomposition of LoRA dynamics. Under standard \initA, the update contains an initialization-induced noise term whose magnitude can be affected by the LR for $B$, $\eta_B$, the initialization scale $\beta$, and the adapter rank $r$. In practical LLM fine-tuning, this non-vanishing noise can limit the performance gains expected from LoRA+ style asymmetric LRs. To suppress this effect, we propose \initAA, a simple initialization that scales the variance of $A$ as $\sigma_A^2 \propto n^{-2}$ while keeping $B_0=0$. Under matched rank and training time for both initializations, this reduces the initialization-noise energy by an additional factor of $1/n$, stabilizing asymmetric LoRA training while preserving zero initial adapter output. Empirically, \initAA substantially widens the stable LR region across initialization scales, adapter ranks, and LoRA update multipliers. Across tasks and models, \initAA reduces best-ratio drift, prevents the high-ratio collapse observed with standard \initA, and makes the width-based guideline $\eta_B \approx n \eta_A$ a useful starting point for finite pretrained models. 

---
# Beyond Data Schedule: Fast LLM Pretraining on Data Mixtures by Loss-Reweighting 

**Authors**: Jiazheng Li, Huaqing Zhang, Jing Xu, Tianle Cai, Jingzhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.32141)  

**Abstract**: Large language models often acquire specialized capabilities through successive pretraining stages, but these gains can come at the cost of forgetting earlier knowledge. In this work, we investigate whether such trade-offs can be improved by changing how strongly each domain contributes to learning while keeping its data exposure fixed. We find that, in controlled knowledge--mathematics experiments, loss reweighting produces a single checkpoint that surpasses the best knowledge and reasoning scores attained along either stage trajectory. This reveals an independent control that data-mixture ratios alone do not capture. Building on this finding, we propose LARE (Loss-Adaptive Reweighting under fixed Exposure), which automatically adjusts domain weights from running loss statistics. Motivated by a theoretical analysis of gradient noise, its normalized inverse-square-root rule balances relative loss scales without changing the sampled mixture or requiring additional runs to search for weights. Across knowledge, mathematics, and code, \Method{} improves all five evaluated benchmarks for a 1.5B model trained on 240B tokens, averaging +7.84 pp over a matched direct mixture, including gains of +17.06 pp on GSM8K and +9.00 pp on MATH. These findings demonstrate that adjusting domain loss scales can improve joint capabilities without additional domain exposure. 

---
# ImageRSI: Verification-Driven Recursive Self-Improvement for Agentic Image Generation 

**Authors**: Shuo Han, Zhilin Wang, Peng Ling, Wenjia Gong, Yang Liu, Guanjun Jiang, Yukun Cao, Xike Xie  

**Link**: [PDF](https://arxiv.org/pdf/2610.32120)  

**Abstract**: Complex text-to-image requests remain difficult to satisfy reliably. Verifying and repairing an image can fix an individual output, but a successful repair does not necessarily carry over to other requests. We present ImageRSI, a framework that keeps the image generator frozen and combines three components: five-dimensional verification and repair within each task, an experience library shared across tasks, and offline updating of the judge. Candidate experiences enter the library only after paired A/B validation, and the judge is updated with supervised fine-tuning and GRPO on accumulated image-judgment records and evaluated before reuse. We also introduce ImageReq-Bench, a benchmark of 961 requests built around hard cases in real-world image-generation requests, each paired with an evaluation checklist. ImageRSI achieves a Checklist Strict Accuracy (CSA) of 52.6% on MindBench (with Nano Banana Pro as the generator) and 65.66% on ImageReq-Bench, and we compare system outcomes across judge-training configurations. 

---
# Tampered ML: Evaluating LLM Judgment of Machine Learning Research 

**Authors**: Soroush Ebadian, Parand A. Alamdari, Joe Benton, Hailey Joren  

**Link**: [PDF](https://arxiv.org/pdf/2610.32114)  

**Abstract**: One approach to accelerating progress on AI alignment is to automate parts of alignment research itself. Doing so requires systems that can perform reliably on hard-to-supervise research tasks, including identifying flaws in research proposals and evaluating whether a proposed experiment supports its intended claim. To evaluate this capability, we introduce Tampered ML, a benchmark of 1,858 item-pairs, each consisting of an experiment proposal reconstructed from a published machine learning paper and a minimally edited counterfactual that introduces a targeted methodological flaw. Models must identify the more methodologically sound proposal in both presentation orders. A human validation study on a stratified sample estimates that 86% of items are valid (95% CI: 81-91%). Frontier models perform above chance but remain far from reliable (with Opus 5 achieving 77.9% accuracy). They are highly sensitive to the order of presentation and favor flawed designs that appear more specific, transparent, or conventional. 

---
# REACT: Representational Early Analysis and Control of Chain-of-Thought Trajectories 

**Authors**: Hanwen Zhang, Enyi Jiang, Ruike Zhu, Qinan Yu, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2610.32113)  

**Abstract**: Chain-of-thought (CoT) generation offers an opportunity to intervene before a reasoning model completes its answer. But does the point at which risk is easiest to detect also offer the most effective control? We study this question with REACT, a conditional activation intervention that learns a safety direction from paired safe and unsafe trajectories and applies it once to an early CoT hidden state. The model weights remain frozen, and all subsequent tokens are generated without further edits. To study timing, we train directions separately at each of the first five CoT positions and evaluate ungated edits with full post-edit regeneration. Across three distilled reasoning models and three generation seeds, the strongest gains concentrate in a two- to three-token early window; on DeepSeek-R1-Distill-Llama-8B, the effect falls to approximately zero at the fourth and fifth tokens, even though later states can be more predictive of risk. This separates risk readability from effective control: intervention timing must be evaluated through its effect on subsequent generation. On the same 8B model, REACT raises the rate of fully safe CoT-answer outputs on StrongREJECT-273 from 29.06% to 82.05%, while keeping benign refusal and usefulness near baseline. An anonymized implementation is available at this https URL. 

---
# ReMedI: Inference-Time Refinement and Self-Distillation for Controllable Molecular Optimization 

**Authors**: Jia Yi Zhang, Shivanshu Shekhar, Allison McLevige, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.32103)  

**Abstract**: In drug development, lead optimization must improve competing properties, such as increasing permeability and reducing toxicity, while limiting scaffold changes that could compromise the lead's potency. RL-based post-training, such as C-MORAL, addresses this problem by fine-tuning molecular language models to directly optimize molecular leads for objectives of interest, which relies on expensive task-specific datasets and faces challenges in achieving multi-objective optimization (MOO). We argue a pre-trained model may already contain useful molecular edits that zero-shot generation cannot efficiently reach. Motivated by the limited search capabilities of zero-shot generation, we develop an efficient Metropolis-Hastings search strategy to better explore the proposal distribution. We propose ReMedI, which uses the molecular language model as a base proposer for inference-time alignment. Given a lead and a set of multi-objective tasks, the model proposes edits conditioned on the current molecule, while Metropolis--Hastings refinement guides sequential edits under an objective for property improvement, preservation, and lead similarity. On both C-MuMOInstruct and SMDD benchmarks, refinement improves frozen base models across three backbones, reaching significantly higher multi-objective optimization success rates compared to baselines. These search traces can then distilled into the base model which improves zero-shot generation and subsequent search using approximately 7.8 times less training molecular leads than that of C-MORAL. ReMedI achieves approximately 2.8 times the strict-success coverage of RL baseline C-MORAL, including on out-of-distribution tasks, with matched lead-relative similarity. 

---
# TTT in VLA Models: Exploring Training with Human Demonstrations 

**Authors**: Qiangong Zhou, Jiafeng Liang, Yimin Dong, Yang He, Hongxiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.32094)  

**Abstract**: Long-horizon robot tasks require retaining and using past information. We integrate Test-Time Training (TTT), which stores context in fast weights, into InternVLA-A1 and Pi0.5 to study action generation conditioned on human demonstrations through training on paired demonstrations and robot trajectories. We implement memory transfer between the two phases and truncated backpropagation, and examine training objectives, gradient coverage, and contrastive regularization across video, short-video, and single-frame cue tasks. In our experiments, adding TTT and training on paired data at the current scale did not lead to consistent target selection based on demonstrations. Under the tested settings, shorter sequences, full trajectory gradient connectivity, and contrastive regularization did not clearly improve responsiveness to demonstrations. When demonstrations were reduced to single-frame cues, real-robot execution still showed a preference for cups at fixed positions. We document the implementation, training configurations, and observations to support further inspection and replication. 

---
# AgentTrace: Execution-Level Fingerprinting of Black-Box LLM Agents 

**Authors**: Gejian Zhao, Hanzhou Wu, Xinpeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.32092)  

**Abstract**: Verifying the provenance of autonomous LLM agents is a critical security challenge. Existing semantic attribution is easily obfuscated by system prompts, while active watermarking conflicts with schema-constrained tool generation. We propose AgentTrace, an active framework that identifies the foundation model behind a black-box agent via observable execution traces. AgentTrace injects structurally ambiguous probes to trigger tokenization-sensitive serialization failures at rigid tool interfaces. While these initial failures provide coarse-grained separation across model families, the temporal sequence of subsequent repair operations reveals fine-grained, model-specific recovery behaviors. By modeling these multi-step trajectories with lightweight Markov chains, AgentTrace achieves 96.0% attribution accuracy under adversarial identity spoofing. Crucially, sequential repair modeling successfully distinguishes the evaluated intra-family variants sharing identical tokenizers (e.g., Llama-3 8B/70B), substantially outperforming unordered behavioral statistics while remaining robust against prompt obfuscation and temperature scaling. 

---
# Guided Zero-Shot Anomaly Explanation for Anonymized Crowd Video Monitoring: Context, Categorization, and Spatial Localization 

**Authors**: Mulugeta Weldezgina Asres, Lei Jiao, Christian Walter Omlin  

**Link**: [PDF](https://arxiv.org/pdf/2610.32073)  

**Abstract**: Despite the promising development of deep learning for video anomaly detection (VAD), explainability (XAI) of these models remains a challenge. Explainable and privacy-protecting VAD builds transparency and trust that facilitates adoption of computer vision systems, including safety and security monitoring. However, anomaly explainability in privacy-preserving VAD systems is underexplored. In this study, we present the XAVAD approach, a unified descriptive anomaly XAI for an anonymized VAD, that integrates flexible zero-shot anonymization, anomaly type categorization, visual localization, and scene context description. We provide empirical and qualitative assessments of multifaceted explanations integrated with full-body visual anonymization for privacy protection. We adopt a multimodal CLIP embedding for videos, using improved LLM-assisted prompting and pool-scoring mechanisms for anomaly-type identification. We enhance Grad-ECLIP for spatial anomaly heatmap localization, leveraging computation and accuracy through our novel prompt-guided patching autofocus mechanism. We employ VLMs to generate context and reasoning descriptions of detected anomalies through curated prompt engineering. We have evaluated the accuracy and computational cost of XAVAD on several publicly available CCTV and video datasets. Our experiments demonstrate substantial gains of our XAVAD over state-of-the-art benchmark approaches and provide insights into privacy protection trade-offs in XAI for VAD. We further present ablation studies and assessments across diverse configurations and modeling alternatives to demonstrate the efficacy of our proposed methods. Code: this https URL}{this https URL . 

---
# Controlled Evaluation of EEG Foundation Models: Random Initialisation, Band-Power and Validation-Selected Probes 

**Authors**: Mingyu Jeon, Suwan Cho, Jae Young Suh  

**Link**: [PDF](https://arxiv.org/pdf/2610.32067)  

**Abstract**: EEG foundation models report downstream gains, yet their protocols seldom include three controls. These are the same encoder untrained, spectral features fitted on the same data, and a layer picked without looking at test scores. We therefore evaluate five publicly released EEG foundation models and an in-house JEPA-style model under one protocol with all three controls and several seeds, on five benchmarks and four added task families. The controls change the reading in three ways. First, picking the probed layer by its test score raises the layer-wise peak by an average of +0.034 and reverses the peak-versus-output gap in 9 of 36 cells. Second, with the layer picked on validation, random initialisation matches or beats pretraining for half of the models on sleep and emotion, and no model beats a five-band log-power baseline on emotion or motor imagery. Third, after full fine-tuning only 5 of 30 pairs gain beyond twice their seed spread, and the one clear loss of -0.290 AUROC persists under matched learning rates. From these results we draw four reporting practices, and we release every cell for re-analysis. 

---
# Concept Acupuncturist: An Efficient and Precise Concept Erasure Method for Diffusion Models via Sparse Autoencoders 

**Authors**: Yiran Xiahou, Siliang Qin, Kai Chen, Shengzhi Zhang, Renyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.32059)  

**Abstract**: Diffusion-based image generation models produce high-quality images but may generate harmful or policy-violating content, motivating machine unlearning (MU) to selectively remove target concepts while preserving general capabilities. However, existing MU approaches often struggle with the distributed and entangled nature of concept representations, leading to inefficient optimization and imprecise edits when applied through large-scale fine-tuning over full diffusion trajectories. To bridge this gap, we propose Concept Acupuncturist (ConceptACP), a novel MU framework that integrates interpretability-guided analysis with inference-free fine-tuning for efficient and precise concept removal. By establishing mappings between semantic concepts and cross-attention blocks in the U-Net denoiser and localizing the most concept-relevant block for targeted optimization, ConceptACP avoids full diffusion trajectory optimization, thereby improving fine-tuning efficiency. To mitigate concept entanglement, we propose a feature-level correction mechanism based on Sparse Autoencoders (SAEs), where SAE-derived sparse, semantically meaningful feature activations are manipulated to construct targeted correction signals, which are then projected back to the output space of the corresponding block and used as supervision for precise concept removal. Consequently, ConceptACP eliminates the need for external inference-time modules while achieving efficient and fine-grained unlearning. Instantiated on SDXL-Turbo, ConceptACP improves the forgetting success rate by 36.34 percentage points in the standard one-step generation setting on average over the best-performing baseline, while preserving image quality and prompt fidelity, and generalizes effectively across other Stable Diffusion backbones. Our code is available at this https URL. 

---
# ACVA-Bench: Beyond Prompt Following in Abstract Concept Visualization Ability for Text-to-Image Models 

**Authors**: Mengyan Wang, Jiyuan An, Yining Liu, YaoWen Zhang, Liner Yang ang Erhong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.32058)  

**Abstract**: Text-to-image (T2I) models have achieved remarkable progress in image quality and instruction following, prompting growing research interest in their capacity for deep understanding that goes beyond literal prompt matching. However, without an explicit visual scheme provided, it remains unclear and has not been systematically evaluated whether T2I models can autonomously translate abstract concepts lacking a fixed visual referent, such as "loyalty," into visual representations that are accurate, diverse, and distinguishable from closely related concepts. We define this capability as Abstract Concept Visualization Ability (ACVA) and introduce ACVA-Bench, a category-balanced benchmark grounded in psycholinguistic norms of abstract concepts, which examines models' ACVA along three dimensions semantic sufficiency, representational diversity, and conceptual distinctiveness under a low-intervention setting that withholds concrete objects, scenes, or visual metaphors. Since abstract concepts lack a unique reference image, conventional evaluation metrics that rely on reference-image matching are difficult to apply. To address this, we propose a fine-grained visual-semantic evaluation method constrained by concept semantics, which decomposes concept semantics into independently verifiable visual-semantic questions to assess the semantic representation quality of generated images, with its validity further confirmed by human evaluation. Benchmarking results on mainstream T2I models show that these models rely on a small set of prototypical visual patterns, leaving room for improvement in representational diversity and conceptual distinctiveness for abstract concepts, thereby providing an interpretable and reproducible evaluation foundation for future research. 

---
# Rethinking Representation Geometry in Supervised Contrastive Learning under Long-Tailed Data 

**Authors**: Joonhyeok Jeong, Jy-yong Sohn, Kibok Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.32039)  

**Abstract**: Supervised contrastive learning and its balanced variants have demonstrated their effectiveness for representation learning under long-tailed distributions. However, although prior analyses suggest that their optimal representations may exhibit a regular simplex geometry, whether such a geometry is indeed exhibited remains insufficiently understood. In this work, we present a theoretical framework for these objectives that characterizes whether a regular simplex is formed at the optimum. By applying this to representative objectives, we show that these objectives fail to form a regular simplex at the optimum because they remain dependent on the number of samples per class. Motivated by this analysis, we introduce structural balancing that can be applied to contrastive objectives, and propose a fully balanced contrastive objective that provably forms a regular simplex geometry at its optimum. Both theoretical and empirical results demonstrate that the proposed method yields a more balanced representation space and improves performance on long-tailed recognition and transfer learning tasks. 

---
# MI-MIDI 2: From Observation to Control with Sparse Autoencoders in Text-to-MIDI Generation 

**Authors**: Jakub Poćwiardowski, Mateusz Modrzejewski  

**Link**: [PDF](https://arxiv.org/pdf/2610.32007)  

**Abstract**: Sparse autoencoders (SAEs) can expose features of a generative model without predefined labels, but in language models the evidence used to interpret these features has proven an unreliable guide to which of them steer the output. Whether interpretations of SAE features predict control in music generators has not, to our knowledge, been tested. We develop an end-to-end protocol for discovering and evaluating SAE features in text-to-MIDI generation and apply it to text2midi and MIDI-LLM. The protocol characterizes features through correlations with musical metrics, shared instrument programs, and automated descriptions, each with its own control: note-density adjustment, matched inactive excerpts, and permutation calibration. It then tests candidates by intervention, measured directly in the generated MIDI against unmodified generation and a matched random direction, ranks them in screens, and confirms selected features on new prompts. Evidence that survives these controls points to features that steer the output. Descriptions exceed permutation controls for instrumentation, and a feature whose description names a synth-lead program introduces that program in a prespecified confirmation and in a replication on disjoint prompts and reappears in independently trained dictionaries. Features selected by density-adjusted correlations show a small positive shift of their target metric as a family, whereas features selected by raw correlations show none. An atlas provides MIDI, piano-roll, and audio examples for every feature of the targeted screen. 

---
# OpenHome: From Perception to Insight in Smart-Home Camera Videos 

**Authors**: Xianglong Wang, Neil Rodrigues, Pei Guo, Yiru Fang, Kevin Beussman, Shuyang Wang, Jingzhi Wang, Liangju Li, Shunyao Xu, Zhongwei Cheng, Wei Li, Lin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.31986)  

**Abstract**: Vision language models (VLMs) have been increasingly adopted in consumer smart-home camera systems to interpret recorded videos and provide insights. However, video clips captured from smart-home cameras are different from curated web videos dominating existing VLM benchmarks. Smart-home video clips often do not contain subtitles or narration and are captured from different viewing angles, under various lighting conditions, and in diverse scenarios. However, the capabilities of the model in this domain, under sparse visual signals and limited context, are not fully assessed by existing VLM benchmarks. To close this gap, we introduce OpenHome, a visual-only, four-option multiple-choice benchmark for evaluating smart-home video understanding. It contains 2,254 questions on 1,118 real-world video clips and organizes 10 tasks into three capability levels. Perception evaluates visible content, Reasoning evaluates causal, spatial, and temporal relationships, and Insight evaluates subject intent, supporting evidence and appropriate homeowner responses. Based on the dataset, we describe the task design, data pipeline and evaluation protocol, and report results for 32 closed-source and open-weight VLMs. The best model, Gemini 4 Argon, reaches 81.2% overall accuracy. Frontier closed-source models outperform open-weight models on every level. The best closed-source model leads the best open-weight model by 16.3% on Perception, 22.4% on Reasoning and 37.4% on Insight. 

---
# Spectra: Multi-Scale Implicit Decomposition for Deepfake Detection 

**Authors**: Dhananjaya Jayasundara, Kartik Narayan, Vishal M. Patel  

**Link**: [PDF](https://arxiv.org/pdf/2610.31979)  

**Abstract**: Deepfakes increasingly undermine trust in digital media, yet existing detectors often struggle to generalize across unseen manipulation techniques, compression levels, and dataset shifts. Most approaches rely on RGB appearance, frequency-domain artifacts, or their combination, learning cues present in the final synthesized image. While effective in-distribution, such cues are frequently generator-specific and can diminish as synthesis models evolve. We argue that a complementary perspective can be obtained through implicit neural representations (INRs), which expose how an image is reconstructed rather than merely how it appears. By fitting an INR to an image and analyzing its internal activations, one can reveal reconstruction dynamics that are difficult to observe in the image domain alone and may capture more intrinsic differences between genuine and synthetic content. To this end, we propose Spectra, a deepfake detection framework that combines semantic visual representations with artifact-aware cues derived from INR reconstruction. Spectra extracts multi-level reconstruction heatmaps from image-specific INRs and jointly models them with semantic features through a bidirectional fusion mechanism. This enables direct interaction between image semantics and reconstruction artifacts, allowing the model to identify forgery cues that are less dependent on specific generation pipelines. Extensive experiments on multiple deepfake benchmarks demonstrate improved robustness and cross-dataset generalization, consistently outperforming detectors based solely on RGB appearance or semantic representations. Furthermore, we show that the proposed representation extends beyond deepfake detection to face presentation attack detection, highlighting its potential as a unified framework for facial attack detection. 

---
# Agentic AI Defense: Decode-Time Provenance Binding for MCP 

**Authors**: Eric Yocam, Varghese Vaidyan, Paris Kalathas, Patrick Tague, Bruce DeBruhl, Yong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.31974)  

**Abstract**: Agentic AI systems using the Model Context Protocol (MCP) are vulnerable to indirect prompt injection through untrusted tool outputs. We demonstrate cross-modal trust laundering, in which instructions concealed in an email-signature image using least-significant-bit (LSB), discrete cosine transform (DCT), or metadata steganography bypass text scanners and enter a multi-agent workflow when decoded by a tool. To counter this threat, we propose decode-time provenance binding, which combines HMAC-authenticated trust labels with an MCP-maintained provenance ledger and context-taint tracking. The mechanism labels decoded content as untrusted and blocks critical tool calls influenced by that content, even when a compromised agent substitutes a valid trusted label. We evaluate the approach using a real MCP testbed with three steganographic carriers and 24 adaptive payloads. The undefended pipeline exhibits 100% attack success, while keyword and regex defenses each permit 62.5% of attacks. Our mechanism blocks all attacks (0% success), including adversarial label substitution across mediated agent boundaries. A four-component ablation validates the roles of decode marking, cryptographic authentication, lineage tracking, and sink enforcement. An audited trusted-promotion mechanism increases legitimate-task completion from 67% to 100% without compromising protection. The approach provides carrier-independent containment within fully mediated MCP execution environments. Code and data are publicly available. 

---
# A Mathematical Theory of the Jacobian Lens 

**Authors**: Mitchell Ostrow, Daoyuan Qian, Toni Liu, Ila Fiete  

**Link**: [PDF](https://arxiv.org/pdf/2610.31955)  

**Abstract**: Recent work from Anthropic introduced a new interpretability tool, called the Jacobian lens (J-lens, Gurnee et al., 2026). This method identified linear directions in the hidden states of frontier language models that had properties reminiscent of working memory-like representations. However, it is unclear what mechanistic properties of such features make them detectable to the J-lens. To this end, we introduce a mathematical theory of the J-lens, and show that it preferentially identifies coherent features -- directions that act like leading, spatially coherent Lyapunov modes. Small perturbations along these directions amplify across layers, remain roughly constant across token positions, and are consistent across different contexts. While we are agnostic to whether these properties are necessarily associated with a ``Global Workspace'', the analytically derived preferences of the J-lens nevertheless explain prior empirical applied to both model classes. We validate our analysis on a Bayes filter and a toy transformer, each optimally predicting next tokens emitted from a Hidden Markov Model (HMM). We show how modulating task parameters that change the coherence of features in the Bayes filter correspondingly and predictively modulates the response of the J-lens . We also introduce of a family of J-lenses built to emphasize different temporal scales of information processing in LLMs, which we call $\omega$-J-lenses. 

---
# RobustNCL: Robustifying Neuromorphic Continual Learning under Transient Faults with Lightweight Detect-and-Cure Strategy 

**Authors**: Rachmad Vidya Wicaksana Putra, Shuochen Zhou, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2610.31954)  

**Abstract**: In neural networks, continual learning (CL) is a prominent approach for enabling deployed systems to adapt to dynamic operational environments by effectively learning from new unseen data at runtime. Recently, the neuromorphic continual learning (NCL) concept has emerged as an energy-efficient alternative to CL by leveraging sparse event-based operations. However, NCL performance can be undermined by transient faults in the hardware platforms, which may come from high-energy particle strikes at any time. Currently, NCL reliability under transient faults has not been studied, and hence the corresponding fault-tolerance techniques remain unexplored. To address this, we propose a novel RobustNCL framework to improve NCL reliability in the presence of transient faults in the DRAM-based off-chip memory of hardware platforms through a lightweight detect-and-cure strategy. Specifically, it first monitors the presence of faults during weight data access from off-chip memory, and then cures the accessed weights if anomalous signatures are detected. It provides three curing variants: replacement with zero, min/max value, and most-probable value from the original/clean weight distribution. To support this, the weight curing values are stored in the radiation-hardened memory. Experimental results show that, in the presence of high fault rates (e.g., 5x10^{-4} for memory replay), RobustNCL maintains high accuracy close to the accuracy of fault-free models (i.e., within -0.5% accuracy for the best RobustNCL variant) and significantly improves accuracy from the baseline without fault mitigation across different NCL methods and datasets (i.e., Split MNIST, Split Fashion-MNIST, Split CIFAR-10, and Split CIFAR-100), while incurring negligible latency and energy overheads. These results highlight that our RobustNCL framework is a promising approach for enabling reliable and energy-efficient NCL systems. 

---
# Kolibri: A Sovereign European Model on the Pareto Frontier 

**Authors**: Aleph Alpha, Moritz Althaus, Artur Baranowski, Michael Barlow, Fabien C. Y. Benureau, Felix Berkenkamp, Amrita Bhattacherjee, Lukas Blübaum, Duncan Blythe, Augustijn de Boer, Bastian Boll, Thomas F. Burns, Björn Deiseroth, Niko Dürr, Béni Balázs Egressy, Niklas Finken, Niklas Freymuth, Fabian Gebhart, Max Hammer, Bastian Harren, Andreas Hartel, Johann Higl, Steffen Hirschmann, Pablo Iyu Guerrero, Yasser Jadidi, Andreas Jagusch, Mitja Kleider, Markus Klein, Jonas Knupp, Murali Kondragunta, Garwin Lechner, Dominik Lê-Kellner, Jurek Leonhardt, Jiaze Li, Sohir Maskey, Omar Mehio, Pit Neitemeier, Letitia Parcalabescu, Alejandro Petry Pacheco, Pranav Ragupathy, Martin Reinhardt, Peter Rodenkirch Nfissi, Dylan Rodriquez, Leonard Salewski, Raziye Sari, Jordan Sassoon, Ivo Schaper, Ilhan Scheer, Frank Schneider, Philipp Scholl, Lukas Schott, Eric Schreiber, Pablo Schumacher, Alessio Serra, Philipp Siedler, Prabhu Teja Sivaprasad, Mohamed Soliman, Till Speicher, Volker Stampa, Simon Thel, Serge Thilges, Irina Vidal Migallón, Denis Waßmann, Samuel Weinbach, Sascha Wirges, Michael Wüthrich, Alexei Zacharov  

**Link**: [PDF](https://arxiv.org/pdf/2610.31945)  

**Abstract**: This report introduces Kolibri, an English-German mixture-of-experts transformer with 78.1B total parameters and 3.46B active parameters per token, released as open weights under the Apache 2.0 licence. We built Kolibri for sovereign and specialised deployment, with a particular focus on German and on regulated domains such as public administration, industry, and aerospace. To support these settings, we control the full model-development process, including data, architecture, training infrastructure, post-training, and evaluation.
We train Kolibri on 24T tokens across pre-training, mid-training, and long-context extension, with German accounting for more than 20% of the data mix by bytes. To reach this share, we complement German datasets with more than 2T German tokens that we curate from the web with our own pipeline and generate synthetically. Post-training combines supervised fine-tuning on curated German and English data with reinforcement learning on more than 1.2M internally curated tasks, which cover reasoning, tool use, instruction following, code, and retrieval. The model activates 4.4% of its parameters per token, and among the open models evaluated in this report, it lies on the Pareto frontier of quality and serving cost in English and German.
Kolibri builds on Kolibri Origin, an earlier iteration that validated our full pipeline at scale. Both models come from our Model Factory, which connects data, experimentation, training, evaluation, and deployment through version-controlled workflows. 

---
# ENCL: Enabling Efficient Neuromorphic Continual Learning with Local Learning Rule and Sub-Network Freezing Strategy 

**Authors**: Rachmad Vidya Wicaksana Putra, Shuochen Zhou, Hassan Raza, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2610.31927)  

**Abstract**: Continual learning (CL) in neural networks is a promising learning approach for updating the knowledge of deployed systems based on new data encountered at run time, that have not been learned during the offline training phase. To enable their power/energy-efficient CL processing, neuromorphic continual learning (NCL) concept with sparse event-based operations have recently been introduced. However, state-of-the-art NCL methods typically employ the memory-hungry data/experience replay, coupled with costly global learning rule (i.e., backpropagation-based surrogate gradient). To address these limitations, we propose ENCL, a novel methodology that enables efficient NCL by employing a lightweight local learning rule and sub-network freezing strategy. ENCL leverages the temporally- and spatially-local learning rule to perform efficient credit assignment for training new tasks, alleviating the expensive cost of global learning rule. ENCL also employs a sub-network freezing strategy based on the given tasks to avoid the needs for network architecture expansion, hence further improving the efficiency and scalability of the NCL training process. To efficiently support this, we train each task with a small number of epochs as well as adjusted learning rate and timestep. Experiment results on task-incremental learning (TIL) scenario show that, our ENCL consistently achieves strong performance across different datasets as compared to state-of-the-art works. Specifically, ENCL achieves competitive accuracy comparable to that of the state-of-the-art data/experience replay, while reducing memory cost by up to 88%, NCL training time by up to 51%, average power consumption by up to 40%, and energy consumption by up to 65% across different benchmarking datasets (i.e., Split MNIST, Split Fashion MNIST, Split CIFAR-10, and Split CIFAR-100). 

---
# Learning Structured Sparsity for Large Models: Bridging Model Compressibility and Efficient Inference 

**Authors**: Tianxiang Xia, Kai Lion, Niao He, Michael Muehlebach, Melanie Zeilinger, Hao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.31897)  

**Abstract**: Scaling laws continue to reward increases in model scale, while the resulting computational and memory demands drive greater reliance on cloud infrastructure for inference. Yet unreliable network connectivity, communication delays, and privacy requirements can make such reliance impractical, motivating local execution on edge devices. Local deployment requires models to fit within limited memory and execute efficiently with the available computational resources. Recent work has shown that learning sparse-plus-low-rank representations enables elastic compression of a single checkpoint to multiple memory budgets. However, fewer parameters do not necessarily translate into faster inference. Indexing overhead and irregular memory access in unstructured sparse components can offset computational savings. To address this critical deployment barrier, we introduce S-SALAAD, a unified framework that jointly learns the model weights and their low-rank and structured sparse components during training. The framework supports a family of hardware-compatible structured sparsity patterns. It couples this structure learning with budget-aware compression and dedicated CPU and GPU kernels that exploit the learned structure for acceleration, enabling efficient inference from a single checkpoint across memory budgets. Experiments on Llama2 and Qwen3 models, covering both pretraining and distillation, show that S-SALAAD advances the empirical Pareto frontier in perplexity, memory, and inference latency across multiple hardware platforms relative to the evaluated baselines. These results show that structured learning can translate elastic compressibility into reduced memory requirements and lower inference latency, enabling practical deployment of large models on edge devices. 

---
# Knowing When to Stop: The Evidence-Action Gap in Coding Agents 

**Authors**: Tianyu Chen, Xinran Zhao, Vijay Viswanathan, Shoubin Yu, Mingyuan Zhou, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.31895)  

**Abstract**: Coding agents are trained to complete tasks, and they keep trying even when the environment makes completion impossible. The evidence that only outside help can restore progress usually appears within a few steps, yet agents rarely act on it: they over-persist, attempt undesired workarounds, and claim success the record does not support -- a phenomenon we call evidence-action gap. To measure it, we introduce ImpasseBench, which injects twelve realistic environment blockers into otherwise solvable repair tasks and marks, in each trajectory, the step at which the collected evidence becomes sufficient. Scoring actions relative to that step separates appropriate effort from over-persistence and premature handoff. Across eight frontier models and 19,371 trajectories, we find that nearly every run that needs outside help collects sufficient evidence, yet a third still exhaust their budget, and every unsafe request we detect comes after that point. Fortunately, an escalation tool, or an external monitor for models that ignore the tool, already guards against much of this behavior: the tool cuts budget exhaustion from 34.0% to 6.7%, and the monitor nearly triples task success after environment restoration. Even then, a fired escalation is not yet a useful handoff: tickets reliably name the blocker but often omit the agent's own progress and verification status. Taken together, we argue that knowing when to stop therefore needs explicit support, both in the tool that lets an agent hand off and in the report it leaves behind. 

---
# Effectiveness of Curriculum Learning Depends on Reward Sparsity and Competing Optima 

**Authors**: John J. Vastola, Ann Huang, Satpreet H. Singh, Samuel J. Gershman, Kanaka Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2610.31884)  

**Abstract**: While animals and people tend to learn a task more quickly and reliably when they first train on simpler versions of it, curriculum learning's effectiveness in artificial settings varies, and appears substantially greater in reinforcement learning than in supervised learning. In this work, we construct and analyze a minimal model of policy learning to better understand why. We consider an open-loop, episodic target interception task whose parameters---especially how close the agent must be to the target to receive an informative reward signal---can be chosen so that the agent either receives informative rewards frequently ('dense' rewards) or rarely ('sparse' rewards). We mathematically and numerically analyze the dynamics of REINFORCE-based policy learning in both cases. In the dense reward case, curricula have only marginal benefits; in the sparse case, curricula can be required for learning to occur at all. We find that this is especially true when the agent has the conflicting goals of minimizing effort and accumulating high reward, since this can produce distinct reward landscape optima which compete with one another. Our characterization of useful curricula in this setting indicates that they address two issues: they (i) make rewards less sparse to speed up learning, and (ii) shape the reward landscape to steer agents away from bad local optima. 

---
# FSVL: Taxonomic Fast-Slow Vision-to-Language Collaborative Framework for Interpretable Confusable Fish Species Identification and Knowledge Generation in the Middle Yangtze River 

**Authors**: Meng Liang, Fucheng Wang, Lang Zhao, Jun Chen, Ziying Li, Yuxiang Zhao, Haozhuang Chi, Wenkai Fang, Qiong Zhou, Junlin Yang, Jinming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.31880)  

**Abstract**: The Yangtze River Basin is a critical region for freshwater biodiversity conservation. The implementation of the Yangtze River Ten-Year Fishing Ban has increased the importance of effective aquatic biodiversity monitoring and species-level identification. However, fine-grained identification of closely related freshwater fish species remains challenging due to subtle morphological differences. Existing deep learning methods often struggle to balance classification accuracy and inference efficiency, while LLMs may generate unsupported information when producing specialized biological knowledge. To address these challenges, this study proposes FSVL, a fast-slow dual-pathway framework integrating visual recognition with LLM-based Retrieval-Augmented Generation (RAG). A lightweight ResNet50 fast path performs preliminary classification across 51 fish species in the Middle Yangtze River Basin and routes predefined groups of easily confused species to a slow path for refined identification. Guided by taxonomic criteria, the slow path uses YOLOv8n to detect key morphological regions, including the head, dorsal fin, and tail, leveraging morphometric and outline traits to improve discrimination among closely related species. The framework further integrates a retrieval-enhanced Qwen3-14B backend, using soft tokens and Low-Rank Adaptation (LoRA) for parameter-efficient domain knowledge generation. On the evaluated test set of 607 images, the fast classifier achieved 99.67% accuracy, whereas the cascade corrected the two errors, yielding 100.00% overall accuracy; further validation on independent specimens and external conditions remains needed. Across 255 knowledge generation samples, scientific name accuracy rises from 91.37% to 100.00%, genus accuracy from 90.20% to 98.82%, and ROUGE-L from 0.335 to 0.655, supporting intelligent biodiversity monitoring and fisheries management. 

---
# Machine-Readable Operator Manuals over the ISO 11783 Network for Offline, Multilingual LLM Assistance 

**Authors**: Timo Oksanen, Julius Gun  

**Link**: [PDF](https://arxiv.org/pdf/2610.31863)  

**Abstract**: If agricultural machinery is to support intelligent, offline assistance, then the system must access technical documentation dynamically. In this paper, the Semantic Synchronization of operator manuals over ISO 11783-based CAN buses is analyzed, defined as the process that brings the local knowledge base of a tractor-side assistant into agreement with the operator manual of the connected implement. Data transfer performance was benchmarked using Markdown, JSON, and XML; various open-weight Large Language Models (LLMs) were also evaluated in cross-lingual query tasks. The automatic evaluation of the answers was validated against four human raters. A complete manual was transferred over a physical ISO 11783 network in 31 to 89 s, depending on format and transport settings, which is comparable to the object pool upload that operators already wait for at the first coupling of an implement. Markdown produced the smallest files, while the question-answering performance was comparable across the three formats for most models. With retrieval-augmented generation, locally run models answered most questions on the English manual correctly, with no advantage of the largest model over 7B and 8B models, but performed worse on the five other language editions. Keyword retrieval failed across languages, whereas semantic retrieval provided the cross-lingual capability. The exchange requires only a new parameter group that uses the existing request and transport mechanisms of ISO 11783, which makes it a candidate for standardization. The study is a proof of concept with inference on PC-grade hardware; deployment on embedded agricultural hardware remains future work. 

---
# SWE-Clar: Can Coding Agents Clarify Incomplete Client Requirements and Deliver Greenfield Software? 

**Authors**: Jialei Li, Dingxian Wang, Xiaoyan Zhao, Yimeng Bai, Xuanyi Zhu, Yang Zhang, Chao Yin, Wentao Shi, Ganesh Prabhu, Karthik Gomadam Rajagopal, Brett Levert, Darvin Yi, Andrew Rabinovich, Zhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.31854)  

**Abstract**: Greenfield software development in freelance settings requires coding agents to turn incomplete client requests into working software. This makes requirement clarification essential, yet clients may be unable to provide continuous feedback over long development cycles, and receiving their answers does not guarantee correct understanding of complex requirements. These challenges motivate two benchmark design choices: clarification precedes code implementation, and its effectiveness is assessed through explicit tests of requirement understanding. Guided by these choices, we introduce SWE-Clar, a benchmark grounded in real, high-value client commissions to evaluate requirement clarification and code implementation in this setting. It comprises 100 tasks based on real software development jobs on Upwork, each with at least US\$4,000 in total payments. Each task begins with multi-round interaction between a coding agent and a client agent, followed by separate runs for code implementation with acceptance testing and clarification assessment. For the latter, we construct 512 questions paired with reference answers from real client-freelancer interactions, covering both explicitly stated requirements and those inferred from the conversational context. Both runs begin from the same clarification context but remain isolated, preventing assessment questions from guiding code implementation. Experiments with nine frontier models show that even GPT-6 Astra, which has the highest observed implementation score, achieves a task pass rate of only 67%, underscoring the challenge of independent software delivery in this setting. The dataset and evaluation code are available at this https URL. 

---
# Rethinking Multi-View Fusion for Urban Representation Learning 

**Authors**: Sean Bin Yang, Zongyi Xu, Jilin Hu, Ying Sun, Kristian Torp, Bin Yang, Torben Bach Pedersen, Christian S. Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2610.31850)  

**Abstract**: Urban region representation learning encodes heterogeneous urban signals, such as mobility flows, points of interest, and land-use patterns, into unified embeddings for downstream tasks, including check-in prediction, crime forecasting, and service demand estimation. Existing approaches rely on globally shared fusion and joint optimization, implicitly assuming uniform modality importance and naturally balanced multi-objective training. However, urban data is inherently heterogeneous. Our empirical analysis reveals two structural issues: (i) modality utility varies markedly across regions, causing static fusion to overemphasize signals from certain modalities; and (ii) heterogeneous objectives exhibit persistent gradient imbalance, causing representational bias and unstable training. We present PRISM, a self-adjusting framework for adaptive multi-view learning. PRISM adapts sparse MoE routing to learn region-specific view allocations, while a PID (Proportional Integral Derivative)-inspired regulation mechanism mitigates gradient conflicts during optimization. Experiments on three large-scale urban datasets show that PRISM achieves competitive performance compared with strong multi-view baselines, while providing more stable optimization behavior and more balanced region representations. 

---
# SEVI: Safe Evidential Real-World Reinforcement Learning 

**Authors**: Shaekh Mohammad Shithil, Lantao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.31829)  

**Abstract**: Real-world reinforcement learning has enabled contact-rich manipulation tasks to be trained directly on physical robots, but unsafe interactions during online learning remain a key barrier to practical deployment. We propose Safe Evidential Real-World Reinforcement Learning (SEVI), a framework for reliability-aware safety intervention during real-world policy learning. SEVI introduces an Evidential Safety Critic that estimates both future constraint-violation risk and epistemic uncertainty in a single forward pass. During online learning, SEVI evaluates each task-policy action before execution; if the action is predicted to be unsafe or the prediction is insufficiently supported by evidence, SEVI overrides it with an offline-trained recovery policy that guides the robot toward a lower-risk region. We evaluate SEVI on two simulated manipulation tasks and a real-world USB insertion task under in-distribution and distribution-shifted conditions. Across task-learning and safety-critic baselines, SEVI achieves the strongest trajectory-level safety prediction, the highest final task success, and the lowest unsafe-event counts. SEVI also reduces unsafe events by 13.4 to 57.1 percent relative to recovery-based safe RL while preserving low-latency execution-time filtering. The project page is available at: this https URL 

---
# What the Query Remembers: Query Abstraction Governs Lexical and Dense Retrieval of Classical Arabic Poetry 

**Authors**: Omar Farouk Zouak, Selma Mani, Larbi Said Chikh, Raid Abderrezak Ouahioune  

**Link**: [PDF](https://arxiv.org/pdf/2610.31821)  

**Abstract**: How well a poetry archive can be searched depends on what the user still remembers, not only on what the model understands. We show this for Classical Arabic poetry, benchmarking nine dense encoders and BM25 over two corpora, three indexing modes and two evaluation protocols. Which retriever wins is governed by how much of a poem's wording the query still carries. On tip-of-the-tongue queries BM25 leads every dense encoder, including an 8B state-of-the-art model (0.746 vs. 0.717 Recall@5); on abstract queries it falls to eighth out of ten (0.392 vs. 0.748). Editing the wording of the same queries shows that this ranking largely reflects query composition: adding one remembered half-verse raises BM25 by 38 points, rewriting the quoted words by hand leaves BM25 level with the 8B encoder, and BM25's sensitivity to remembered wording is significantly steeper than that of the strongest dense encoder. The same axis controls explanation augmentation, whose effect swings from +37 to -39 points. The mechanism splits by system type: lexical containment predicts BM25's behaviour, while embedding proximity predicts dense encoders. Finally, the field's two standard protocols, known-item retrieval and thematic relevance, dissociate on the same systems, a result reproduced by two blind human annotators and by two open-weights judges, one of them under a rubric with no literal-match criterion. A benchmark that reports one protocol, or one query register, can give a materially wrong picture. Code and data are available at this https URL. 

---
# PrefO2O: Zero-Shot Preference-Based Reinforcement Learning via Offline-to-Online Latent Preference Transfer 

**Authors**: Ruiqi Wang, Dezhong Zhao, Xuesu Xiao, Erdem Biyik, Byung-Cheol Min, Guohua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.31812)  

**Abstract**: Preference-based reinforcement learning (PbRL) has emerged as a promising paradigm for learning robot behaviors aligned with human intent. However, its scalability is fundamentally limited by the need to collect expensive human preference labels for every new task. In this work, we present PrefO2O, an offline-to-online latent preference transfer framework that enables zero-shot PbRL on unseen target tasks without requiring any human feedback. The core idea is to learn a task-invariant latent preference space from a small amount of offline preference data on source tasks, where transferable preference structures are disentangled from task-specific variations. This learned manifold allows PrefO2O to transfer preference structures and reward priors to new tasks under distribution shift, while confidence-weighted optimal transport alignment generates reliable pseudo-preference supervision for online reward adaptation. Extensive experiments on robotic manipulation and locomotion benchmarks demonstrate that PrefO2O substantially outperforms existing preference-transfer baselines and matches or exceeds PbRL methods trained with real human feedback, while generalizing across multi-to-one and one-to-one transfer settings as well as across different robot embodiments. 

---
# Benchmarks Are Surprisingly Compaction-Friendly 

**Authors**: Jeanne Salle, Hardik Bhatnagar, Matthias Bethge, Sahar Abdelnabi, Maksym Andriushchenko  

**Link**: [PDF](https://arxiv.org/pdf/2610.31778)  

**Abstract**: Long horizon agentic tasks are now pushing LLM agents past their context window. To continue the task, the agent scaffold has to free up the context window, a process known as context compaction. Compaction is generally lossy by design: agents can silently lose task constraints, instructions, or their reasoning behind earlier decisions. Although compaction is a standard component of coding agent scaffolds, it has not been studied systematically with frontier models on realistic, long horizon agentic tasks. Previous work evaluate smaller open weight models on toy or synthetic trajectories. We fix the agent scaffold (OpenCode), implement five compaction strategies inside it, and evaluate three frontier models on three long horizon benchmarks (PaperBench, ProgramBench, PostTrainBench). For the two 1M context models, the compaction threshold does not measurably affect the task performance, whereas GPT-5.3-Codex, the model with the smallest context window, performs worse at lower thresholds on PaperBench and, less clearly, on PostTrainBench. No compaction method outperforms default self-summarization, and methods that discard more content cost some models up to 20 points while leaving other models the same. Compaction can, however, silently remove task constraints: on ProgramBench, the first compaction summary already drops about one third of the task's anticheating rules on average, and this rises monotonically with the number of compactions. On real user-agent sessions from SWE-chat, instructions about how the agent should work and what it should not do are dropped around 6 times more than specifications of what to build. We argue that current benchmarks are compaction-friendly: i.e., the objective is fixed and the agent's progress stays on disk, so most of what compaction discards can be recovered, except constraints that don't exist anywhere else. 

---
# Generalized green and red concepts for artificial intelligence, big data, and information and communication technologies: Multidimensional framework and countermeasures 

**Authors**: Jinsong Wu, Yixin Xiao, Haowei Bi, Huamin Fang  

**Link**: [PDF](https://arxiv.org/pdf/2610.31739)  

**Abstract**: The rapid expansions of artificial intelligence (AI), big data (BD), and information and communication technologies (ICT) are driving substantial growth in electricity, water, and material demands, with data centers projected to consume 6.7--12\% of U.S. electricity by 2028 (up from 4.4\% in 2023) and AI-related water demand estimated at 4--6 billion m$^3$ annually by 2027. Existing ``green AI'' and ``red AI'' discussions have largely emphasized computational efficiency and carbon emissions, providing limited guidance for lifecycle trade-offs, service-versus-use impacts, and broader sustainability externalities beyond energy. To bridge this gap, this paper proposes generalized green AI (GGAI) and generalized red AI (GRAI) and extends the taxonomy to red BD (RBD) and red ICT (RICT) as well as red electronic and electrical engineering (REEE) \& red digital technologies (RDT) under a unified Environmental-Economic-Social-Technical/Operational (E/Ec/S/T) framework. Based on representative footprint indicators in the large-model era (electricity, carbon, water, and e-waste), we analyze how risks propagate across AI-BD-ICT stacks and across service-use boundaries. We further organize actionable countermeasures into a four-tier taxonomy aligned with the Environmental-Economic-Social-Technical/Operational framework: metric-level reform for accountability, revision of information and communication technologies design principles and demand shaping, efficiency-oriented LLM development practices (data, model, training, and scaling), and energy-infrastructure transition. The proposed framework operationalizes the Environmental-Economic-Social-Technical/Operational-aligned assessment and intervention prioritization across the AI-BD-ICT stacks, enabling sustainability-by-design without sacrificing utility. 

---
# Learning to Be Monitored: Causal Bounds and Intervention-Aware Post-Training Recipes for Chain-of-Thought Monitorability 

**Authors**: Qiang Fu, Xinyue Li, Andre Wibisono, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2610.31733)  

**Abstract**: Chain-of-thought (CoT) monitoring aims to detect potentially harmful behaviors in large language models (LLMs) from their CoTs. Intervention-based evaluations measure this property by asking whether a monitor can detect the role of an intervention from the CoT when the targeted behavior occurs, typically using minimal-criterion sensitivity ($\mathsf{TPR_{mc}}$) proposed by~\citep{guan2025monitoring}. However, a target behavior observed under intervention is not necessarily caused by it, and we show that this metric can overestimate how often a monitor detects target behaviors actually caused by the intervention. Specifically, we establish $\mathsf{TPR_{mc}}$ as an upper bound on this causal detection rate and derive a complementary lower bound, which provides a tighter evaluation framework for CoT monitorability. With this framework, we study whether LLMs can be trained to make their CoTs easier to monitor. We first construct a training dataset of paired control--intervention tasks spanning multiple intervention types. Using the dataset, we design and apply three intervention-specific training methods based on off-policy self-distillation, reinforcement learning, and on-policy self-distillation to train separate variants of each base model. Across various model-monitor configurations, all three methods generally improve CoT monitorability over the corresponding base models under our extended evaluation framework. Our code is available at this https URL. 

---
# Autoregressive Multimodal Anomaly Detection with Large Language Models for Microservice Systems 

**Authors**: Yixiang Tang, Fagui Liu, Guoxiang Zhong, Xi Yao, C.L. Philip Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.31722)  

**Abstract**: Microservice systems have become a dominant architecture for large-scale cloud applications, enabling rapid deployment and elastic scaling. However, ensuring the reliability and availability of microservice systems critically relies on effective anomaly detection, as failures and performance anomalies can quickly propagate across services and severely impact end-to-end system behavior. Existing multimodal anomaly detection approaches for microservice systems often struggle with two fundamental limitations: (i) the neglect of modality-specific characteristics in favor of cross-modality fusion, and (ii) the "identity-shortcut" problem that affects the feature reconstruction process. To address these issues, we propose LLMMAD, an autoregressive multimodal anomaly detection framework for microservice systems based on Large Language Models (LLMs). Initially, to reconcile cross-modality fusion with modality-specific preservation, LLMMAD embeds and recombines multimodal data into a unified sequence for LLM processing to model inter-modality relationships, while employing modality-wise reconstruction heads to retain modality-specific information. Subsequently, to eliminate the identity-shortcut, we implement a prediction-driven autoregressive objective, under which the model predicts future states rather than copies current inputs. Finally, we conduct extensive experiments on three real-world datasets. The results demonstrate that LLMMAD consistently outperforms state-of-the-art methods, achieving F1-score improvements of 3.1%, 1.4%, and 0.9%, respectively, thereby providing a more reliable solution for microservice monitoring. 

---
# End-to-End Differentiable Diffusion Conditioning for Physics-Informed Optimization 

**Authors**: Ricardo Luna Gutierrez, Vineet Gundecha, Rahman Ejaz, Varchas Gopalaswamy, Riccardo Betti, Sahand Ghorbanpour, Aarne Lees, Soumyendu Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2610.31721)  

**Abstract**: Optimizing high-dimensional physical systems under expensive evaluations and complex constraints is a central challenge in science and engineering. In offline model-based optimization (MBO), methods must improve designs using only a fixed dataset and no test-time access to the true objective. However, existing methods typically suffer from conservatism, unconstrained surrogate exploitation, or function as static inference-time samplers that cannot refine candidates against precise physical objectives. In this paper, we propose Differentiable Diffusion Conditioning (D2C) for offline MBO, which leverages the fact that deterministic sampling makes the reverse denoising chain differentiable with respect to the conditioning signal. D2C turns a frozen conditional diffusion model into an end-to-end gradient-based optimizer by directly optimizing conditioning variables at test time via gradients from frozen surrogate objectives and physics-informed penalties backpropagated through the reverse chain. We implement this idea in a composed proposer-evaluator framework and evaluate it on laser pulse optimization for inertial confinement fusion, sustainable data-center workload scheduling, and four Design-Bench tasks. Our method achieves the strongest results on the two real-world tasks and the best average rank across four Design-Bench tasks. 

---
# What Open-Source Projects Tell Their Coding Agents: A Census of AGENTS.md and CLAUDE.md 

**Authors**: Pawel Zarzycki  

**Link**: [PDF](https://arxiv.org/pdf/2610.31715)  

**Abstract**: Coding agents can be guided with instruction files in a repository. We examine this emerging practice through a census of popular, active GitHub repositories in thirteen languages, treating each repository's root this http URL and this http URL as one instruction set. Agent instructions appear in of repositories and of the most-starred; where both files exist, this http URL usually points to this http URL. We analyse statements using model-assisted annotation and human review. The annotations reveal a practical mix of project knowledge and rules about code, building and workflow. Collaboration rules name concrete requirements, from test plans to commit formats, yet of directives give no reason. Projects disagree on AI attribution: commit-message rules more often prohibit it than require it, while pull-request rules more often require disclosure. A third of instruction sets have a generator-associated signature, a tool-inserted block, shared text or an AI-marked creation commit; four in five are edited after creation. Prespecified tests associate repository popularity with rules on pull requests and AI contributions. Exploratory analyses link activity to process-oriented content and show that LLM and agent projects are over-represented among adopters. The models agree much more closely with each other than with the human reviewer's topic assignments, a caution for automated annotation. We release the annotations, taxonomies and pointers to the analysed file versions. 

---
# Reuse, Rewrite, or Reject: Evidence-Constrained Trace Reuse for Repeated Mobile GUI Control 

**Authors**: Bingyu Bai, Cheng Zhang, Erhu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.31712)  

**Abstract**: Mobile GUI agents frequently perform repetitive tasks with similar interaction procedures. Reusing previous executions can reduce redundant LLM inference and execution overhead, but variations in user instructions and execution environments may invalidate previously successful actions. We systematically characterize these variations into six representative categories and observe that such changes do not necessarily invalidate entire procedures: individual transitions can remain reusable when validated against the current task context and interface state. Based on this insight, we propose graph-text templates for selective execution-trace reuse. The graph captures UI states and action transitions, while the text encodes task intent, procedural structure, and typed slots for variable content. A record-template-replay framework constructs templates from verified executions and leverages the current instruction and live UI state to reuse, adapt, or discard individual transitions. Given an execution trace, our framework reduces LLM inference token consumption by approximately 80% on average across five base agents on AndroidWorld, with most agents achieving at least 1.5-fold execution speedup over LLM-only execution. In real-device evaluations on Android and HarmonyOS, our framework achieves action reuse ratios of 70.7-79.8% while maintaining over 2-fold execution speedup in most evaluated configurations. 

---
# MBABench-2.0: Evaluating AI Agents on Constructing Financial Spreadsheets Beyond Simple Verifiability 

**Authors**: Julian Poeltl, Thomson Yen, Patrick Shea, Shubham Anil Varma, Dhruv Rastogi, Zihao Chen, Mengshi Zhang, Hongseok Namkoong  

**Link**: [PDF](https://arxiv.org/pdf/2610.31711)  

**Abstract**: Spreadsheets are the lingua franca of modern finance, driving valuation, forecast, audit, and many other core workflows of the industry. As LLM agents advance to automate knowledge work, spreadsheet construction is a natural and pertinent testbed. However, most existing spreadsheet benchmarks evaluate agents on fill-in-the-blank style tasks, where an agent performs minor edits in otherwise complete spreadsheets. This protocol fails to capture the most central and time-consuming part of the spreadsheet workflow: the modeling decisions, format design, and sheet layout regular knowledge workers need to toil through to build a workbook from scratch. We introduce MBABench-2.0, a benchmark of 101 original spreadsheet construction tasks, the hardest of which is estimated to take 33 hours for a modeler with 3+ years of experience. Unlike coding or math, where a unit test or a final answer can verify an agent's output, the quality of a financial spreadsheet that concerns readability, maintainability, or auditability, is not so simply verifiable. To holistically evaluate spreadsheet construction, we introduce 129 criteria spanning 12 categories that operationalize quality beyond the accuracy of the spreadsheet. Evaluating over 20 agents across 8 models and 4 harness types, we find that frontier agents rarely construct spreadsheets that are both accurate and meet our quality criteria: the top agent achieves a pass rate of only 16% on the benchmark. The benchmark demonstrates a substantial gap in spreadsheet building capability of current agents. 

---
# Compact-Bench: Measuring What Context Compaction Costs Long-Horizon Coding Agents 

**Authors**: Ruopei Sun, Jianfeng Cai, Kangwen Zhao, Wengang Zhou, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.31710)  

**Abstract**: Context compaction lets long-horizon coding agents continue beyond their context windows, but can erase the state needed to finish a task. Prior evaluations measure efficiency, retained information, or end-to-end success, leaving the effect of a specific history replacement difficult to isolate. Compact-Bench pairs full-history and summary-only continuations from a shared repository state, and evaluates recurring details and successive production compactions. It covers 35 SWE-bench repairs, 146 agentic sessions, and 106 production chains. Across 14 summarizer backbones, stable damage ranges from 5-64% of 22 full-history successes: Claude Opus 4.8 preserves the most repairs, while DeepSeek-Flash damages the most. With Kimi-K3 as summarizer, the Kimi Code prompt preserves the most task continuations (19/22), whereas Claude Code retains the most recurring details (15.0%). LLM summaries retain at most 16.0% of recurring details, compared with 79.1% for mechanical extraction; production summaries retain 15.7% at the first boundary. Failures recur across methods: one repair is damaged by 13/14 backbones, and another by all 5 prompts. These results show that reliable compaction requires preserving the progress needed to complete a task, and that retaining relevant information alone does not ensure successful continuation. Code is available at this https URL and data is available at this https URL. 

---
# Mining the Design Space of Multi-Agent Workflows 

**Authors**: Xavier Cadet, Junyan Cheng, Peter Chin, Kyle Richardson  

**Link**: [PDF](https://arxiv.org/pdf/2610.31709)  

**Abstract**: Designing large language model (LLM) multi-agent systems (MAS) still relies on trial and error and informal "folk wisdom." Manual studies test only a few communication structures or topologies (e.g., sequential, parallel, shared-memory, or debate), while automated optimizers discard search trajectories after selecting a task-specific workflow. We present a two-phase pipeline for mining reproducible design principles from workflow code and execution traces. We repurpose AFlow's MCTS optimizer across six composable operators and 26 deterministically verified Enigmata tasks, retaining all intermediate rounds to build a corpus of 28,568 executable workflows. A sandboxed AutoResearch loop then mines orchestration code and prompt templates under an $L_0$-penalized validation objective and tests candidate principles on held-out searches. Across 300 feature-mining trials, we identify effects that persist on held-out searches across three levels of workflow design: workflows beginning with independent solvers and one-shot aggregation (Star-first) perform worse ($\Delta = -0.045$); persistent shared documents (SharedMemory) outperform debate ($\Delta = +0.135$); and prompt specificity and placeholder-based context routing yield gains comparable to communication-structure selection ($\Delta = +0.118$). More broadly, our pipeline and results show the feasibility of turning discarded search trajectories into a statistical testbed for discovering and validating multi-agent workflow design principles. 

---
# RubricCraft: Where Solutions Diverge, Criteria Emerge 

**Authors**: Hosein Hasanbeig, Priyanshu Gupta, Ashish Tiwari, Vu Le, Sumit Gulwani  

**Link**: [PDF](https://arxiv.org/pdf/2610.31707)  

**Abstract**: Evaluating language agents on open-ended, artifact-producing tasks is difficult: model-based graders can be inconsistent, while exhaustive programmatic checks are often impractical. Textual rubrics make evaluation criteria explicit, but writing them requires domain expertise. Rubrics generated from task descriptions alone can miss requirements and failure modes that become apparent in attempted solutions. We introduce RubricCraft, which treats rubric construction as the generation of natural-language specifications guided by candidate solutions. Given a task description and independently sampled solver trajectories, RubricCraft compares how each attempt satisfies or violates the task requirements and uses these differences to construct rubric criteria. It applies the criteria to the collected solutions and revises checks that overlook errors or penalize valid alternatives. The resulting rubrics capture partial correctness without requiring any single solution to serve as a complete reference. On SpreadsheetBench v2, against a prompt-only rubric written and applied by the same model, RubricCraft rubrics pass fewer wrong outputs (29.5% against 42.6%), fail none of the outputs the benchmark's own checker accepts, and rank outputs by correctness more often, while separating models as often. Repair on a single attempt leaves grading unchanged, but repair driven by attempts from three model families acts on different tasks, and an evidence audit judges most of those repairs improvements, many on tasks whose reference solution is itself wrong. 

---
# Before the Ticket: Grounding HPC User Support in Documentation and Resolved Tickets with an Agentic LLM Assistant 

**Authors**: Elham J. Barezi, Primus C Kabuo, Niveditha Nerella  

**Link**: [PDF](https://arxiv.org/pdf/2610.31703)  

**Abstract**: Most questions that reach an HPC help desk have already been answered in the center's user guides or in a ticket someone else filed last semester; users still file new tickets because the guides are hard to navigate and the ticket history is not theirs to search. We present an assistant that answers from both sources: an orchestrator calls two tools over the Model Context Protocol, retrieval-augmented generation (RAG) over the live documentation and a Llama-3-8B model fine-tuned with QLoRA on 11,244 query-resolution pairs mined from historical tickets, and merges their answers. Every model runs on-premises or on the university's hosted LLM service because ticket data may not leave the institution. On 400 held-out tickets from Purdue's Rosen Center for Advanced Computing, disjoint from training and free of near-duplicates, and evaluated by two judges from model families that generated none of the answers, fine-tuning on resolutions rewritten into self-service guidance is the clear winner in agreement with RCAC's resolutions (BERTScore F1 improves from 0.06 to 0.27 over the base model) and is judged most correct on average. The end-to-end system does not improve on its tools, although they are complementary: choosing the better answer per question would exceed either tool (judged correctness 3.41 versus 3.20, an oracle bound), while the merge follows the documentation tool in 78% of answers. Returning one tool's answer instead recovers the fine-tuned tool's quality (3.19 versus 2.66), but neither a learned selector nor an LLM verifier beats that tool alone. 

---
# Collaborative In-Context Learning for Indoor Radio Map Estimation 

**Authors**: Yuanzhe Peng, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.31702)  

**Abstract**: Indoor radio map estimation (RME) is essential for wireless applications such as coverage planning, localization, and network deployment. However, most existing methods follow a supervised learning paradigm that requires costly labeled data and task-specific model training. Moreover, indoor environments involve privacy-sensitive information, such as building layouts and structures, that often cannot be centralized. To address these challenges, we make the first attempt to formulate indoor RME as a collaborative in-context learning (ICL) problem. We propose Poly-Ripple, a framework in which multiple clients perform ICL using large autoregressive vision models (LAVMs) without model updates. Each client is prompted by its private demonstrations and predicts the radio map for the same scene query. To capture the causal structure of wireless propagation outward from the transmitter (Tx), we adopt ripple-ordered autoregressive modeling. We further introduce importance-aware aggregation to exploit heterogeneous contextual knowledge distributed across clients under the context length constraint. We provide theoretical analysis and conduct extensive experiments on a public indoor RME benchmark. The experimental results show that Poly-Ripple consistently outperforms representative ICL baselines, while aligning with our theoretical analysis. 

---
# Distributional Alignment of LLMs and Its Generalization Properties 

**Authors**: Nelson Jo  

**Link**: [PDF](https://arxiv.org/pdf/2610.31687)  

**Abstract**: Large language models are increasingly used to simulate and probe human behavior, e.g., for surveys. For these applications, a key property is that outputs satisfy distributional alignment: to match its output distribution with some reference (in this case, human) distribution. An obstacle in achieving distributional alignment is generalizing from small training data. In this work, we study (1) how to distributionally align LLMs; and (2) how aligned LLMs might generalize to unseen samples and unrelated topics and tasks. Using two large-scale benchmarks, we find that LLMs can indeed achieve distributional alignment and generalize to unseen questions of the same domain. Cross-topic generalization depends on the relationship between topics (e.g., questions about politics are related to trust in institutions). However, aligned LLMs can degrade in surprising ways (e.g., lose their creative ability), suggesting that its generalization properties are inherently unpredictable and unstable. 

---
# From Target Selection to Condition Reuse: Auditing Residual Interventions in Tool-Use Models 

**Authors**: Tianda Sun, Dimitar Kazakov  

**Link**: [PDF](https://arxiv.org/pdf/2610.31686)  

**Abstract**: Activation interventions can make a language model produce a target answer, but can the model use the intended condition in other computations? We study this question in Qwen3.5-27B on tool-use tasks where time or permissions determine eligibility and selection. Across 32 fresh worlds, two case-specific residual-writing variants raise familiar target-selection correctness from 35/128 to 119/128 and from 33/128 to 114/128. Yet these gains disappear when familiar facts and identifiers are recombined within the same selection task, while selection of previously rewarded identifiers increases. The gains persist across output formats but yield little improvement in independent judgments about whether a candidate is eligible or should be selected. Internal edits can nevertheless change how a source fact affects a judgment: in a prespecified secondary comparison, an early-layer source patch changes 50/64 judgments when that source is used, with one adverse change among 64 cases where it is not. This selective effect does not ensure correct complete actions. Together, these results separate target-selection control, condition-sensitive judgments, and complete-action correctness. Evaluating how an edited state is used, rather than only its trained answer, exposes these differences. 

---
# An Agentic Approach with Evidence-Grounded Verification for Multimodal Emotion Reasoning in Conversation 

**Authors**: Elyas Rashno, Farhana Zulkernine  

**Link**: [PDF](https://arxiv.org/pdf/2610.31685)  

**Abstract**: LLM-based multimodal emotion reasoning systems perform well on standard benchmarks, but improving them through instruction tuning can be computationally expensive. Agentic pipelines built from frozen models avoid this cost, but introduce a different challenge: when a prediction is wrong, the system must identify which agent or stage caused the error. We address this problem with an agentic framework for traceable multimodal emotion reasoning and prompt-space adaptation. Modality-specific perception agents analyze visual, acoustic, and lexical inputs and post evidence-cited observations to a shared board. An integration agent combines these observations with retrieved multimodal examples to produce an emotion description, label set, and supporting citations, while an MLLM critic can request one revision before the output is finalized. For incorrect predictions, a separate evidence-grounded verification stage applies deterministic checks to verify that cited evidence and references are valid, and an entailment check to determine whether the final claims are supported by that evidence. Detected failures are then grouped by the responsible component and failure type, converted into targeted prompt patches, and accepted only after validation on a held-out set. This adapts the agent prompts without updating model weights. On the MER2025 dataset, targeted prompt adaptation outperforms the frozen static pipeline on all four MLLMs, ranging from 2B to 38B parameters, and matches or outperforms undifferentiated reflection baselines on each of them: VideoLLaMA3-2B, Phi-3.5-vision-instruct, Qwen2.5-VL-7B-Instruct, and InternVL2.5-38B. It improves F-score by 0.6--2.2 points depending on the backbone, including an increase from 52.3 to 54.5 on Qwen2.5-VL-7B. 

---
# Rethinking Structured Generation: Can Graph-Based Reasoning Resolve Ambiguity? 

**Authors**: Ratun Rahman, Atit Pokharel  

**Link**: [PDF](https://arxiv.org/pdf/2610.31681)  

**Abstract**: Structured generation is typically formulated as a deterministic mapping from input to a single structured output. This assumption is fundamentally misaligned with natural language, which often admits multiple structurally distinct yet semantically valid interpretations. As a result, existing methods rely on single-reference supervision that collapses ambiguity into one prediction, leading to unstable and inconsistent outputs under underspecified inputs. We reformulate structured generation as an \textit{ambiguity-aware inference problem}, where the objective is to reason over a set of plausible structured outputs rather than predict a single solution. We show that standard training inherently suppresses valid alternatives, and instead adopt a multi-valid learning framework that preserves ambiguity during optimization. To operationalize this formulation, we construct a candidate space of structured outputs, represent each candidate as a graph capturing entities and their relationships, and perform selection via graph-based reasoning over semantic alignment and relational consistency. This enables principled comparison across competing interpretations beyond token-level likelihoods. Experiments on structured generation benchmarks demonstrate consistent improvements in accuracy, stability, and robustness under varying levels of ambiguity. These results highlight the importance of modeling ambiguity explicitly and suggest a shift from single-output prediction to set-level reasoning in structured generation. 

---
# Validating the Comparison, Not Just the Scorer: Why Overall Accuracy Can Mislead in Interactive LLM Evaluation 

**Authors**: Ancil Crayton  

**Link**: [PDF](https://arxiv.org/pdf/2610.31678)  

**Abstract**: Many LLM studies compare how models respond to different follow-up prompts. For example, one prompt may neutrally ask a model to reconsider, while another insists that an invented film really exists. The difference is used to measure the effect of conversational pressure. An automatic scorer then labels whether each response makes an unsupported factual claim. Researchers commonly validate the scorer with one overall accuracy or $F_1$ score. However, high overall performance can hide prompt-specific mistakes: if the scorer makes more errors after one prompt, the measured difference can be too large, too small, or point in the wrong direction. An audit containing mostly difficult responses can also misrepresent performance in the full study. We derive how such errors alter a prompt comparison and examine the problem using six models, 180 invented-entity questions, and six prompt conditions. In a balanced audit of 288 responses, overall $F_1$ was $.916$. Yet, depending on the prompt, the scorer flagged 2.6%-33.3% of responses that the primary LLM rater judged not to contain an unsupported claim; these prompt-specific estimates were uncertain. In a second audit, the scorer and two separate LLM raters evaluated the same 300 model-question cases under each prompt. An apparent 9.3-point increase from asserting certainty rather than neutrally asking the model to reconsider fell to 2.3 points under both raters. The primary-rater interval included zero, so it did not establish an increase. Two other prompt comparisons kept their direction under both raters, while an attempted prompt-specific correction remained too uncertain for a firm conclusion. Researchers should therefore validate the exact difference they plan to report, not just the scorer as a whole: audit both prompts on matching cases and include scorer and rater uncertainty in the final result. 

---
# Astro2Method: Benchmarking LLMs on Machine Learning Method Selection in Astronomy 

**Authors**: Xingzhong Fan, Hangfeng He, Fatemeh Nargesian  

**Link**: [PDF](https://arxiv.org/pdf/2610.30282)  

**Abstract**: Machine learning (ML) is increasingly used in astronomy, but choosing a suitable method for a scientific problem often requires expertise in both astronomy and ML. Large language models (LLMs) could help astronomers make such choices, yet this ability remains underexplored in existing benchmarks. We introduce Astro2Method, a benchmark of 500 expert-reviewed binary questions from 346 astronomy papers (2024/01-2026/06), based on clear performance differences and complete (Easy) or partial (Hard) explanations from the authors. Across five LLMs, direct prompting achieves 87.50-92.55% accuracy on Easy questions but only 58.87-64.52% on Hard. Using historical papers, neither retrieval-augmented generation nor most agent methods with additional reasoning or retrieval (including ReAct, Self-Ask, and CRITIC) bring consistent improvement. Agents required to search the web improve accuracy mainly when they find the papers the questions are based on, with no consistent improvement otherwise. Providing the source paper's pipeline description with method names masked improves Hard accuracy, matching or exceeding web search. Together, these results suggest that relevant experimental information helps distinguish ML methods, but inferring it from related studies and domain knowledge remains challenging for LLMs in astronomy. 

---
