# Collaborative AI agents for autonomous experimental research with solid-state quantum spins 

**Authors**: El Mustapha Mansouri, Eikichi Kimura, Masato Koga, Keigo Arai  

**Link**: [PDF](https://arxiv.org/pdf/2610.40589)  

**Abstract**: An autonomous research laboratory could conduct an entire study, from formulating questions to preparing manuscripts. Artificial intelligence (AI) offers one route towards this goal. Recent AI systems take on several research roles, but may draw conclusions that the evidence does not support. Here, we present an autonomous research laboratory in which agents based on large language models assess one another's work and can stop or redirect investigations. They can generate and reject research questions, require separate theoretical checks and conduct experiments designed to challenge their claims. We demonstrate these capabilities by investigating ensembles of quantum spins in diamond. The agents derived the minimum number of calibration settings needed to predict responses to a defined family of microwave pulse sequences. They checked the result computationally and designed experiments to distinguish competing predictions. An initial experiment agreed with the theoretical predictions, but repeated reference measurements showed that the signal was drifting during data collection. The agents therefore treated the result as inconclusive and revised the measurement protocol without relaxing the acceptance criteria. The revised experiment met those criteria and supported the theoretical predictions. This study demonstrates that collaborating AI agents can select research questions, derive theoretical results and test their predictions experimentally. 

---
# Pre-Crime for Agents, at the Last Second: Probing When a Recruited Agent Gives In 

**Authors**: Leonidas Raghav  

**Link**: [PDF](https://arxiv.org/pdf/2610.40588)  

**Abstract**: When AI agents share channels and credentials, one agent can press another to break its operator's rule, a first step towards collusion. Output filters check only what an agent has already written. Can an overseer tell beforehand that an agent is about to give in? We show that a linear probe on the agent's activations can, in a controlled multi-agent setting. Across six scenarios, a peer presses a target for up to 20 rounds to reveal a secret, such as a deploy token; the target often refuses, then gives in. Read at each model call before its output is written, the probe picks out the one that leaks. On held-out Qwen3-32B episodes, at a threshold set to flag 5% of benign calls, it catches 82% of leaking calls. At a similar flag rate, a zero-shot decision-model monitor catches 26%. On gpt-oss-20b, a probe on its reasoning reaches AUROC 0.91. A bag-of-words classifier trained on the same calls comes close, so most of the gain comes from training on labels, not activations. The probe also flags permitted disclosures: it reads the coming disclosure more than the violation. The warning is short-range, and halting on any alarm would stop most benign episodes, which a second text check does not fix. A warning raised before generation could let an overseer stop an agent that is about to give in, ahead of others acting on its messages or recruitment spreading through a swarm. 

---
# VeriCUA: Verification as Agentic Exploration for Computer-Using Agents 

**Authors**: Qiushi Sun, Da Yu, Alexander Knop, Amer Sinha, Lingpeng Kong, Ben Kao, Ravi Kumar, Pritish Kamath  

**Link**: [PDF](https://arxiv.org/pdf/2610.40578)  

**Abstract**: Building and scaling computer-using agents (CUAs) across desktop, mobile, and web environments requires reliable task-completion verification for data curation, evaluation, and reinforcement learning. Typical automated approaches prompt vision-language models (VLMs) to judge trajectories or generate programs to inspect application state. However, both rely on partial views of complex digital environments. VLM judges are restricted to supplied observations, while generated programs depend on accessible state. These views can omit decisive evidence or leave completion judgments unsupported. Tailoring separate verifiers to different platforms and access settings adds further engineering cost. We present VeriCUA, a platform-agnostic framework for verification through agentic exploration. It treats verification as dynamic evidence gathering and assessment within a shared procedure for online and offline settings across platforms and observation modalities. After inspecting the available context to construct a task rubric, the verifier selects GUI or code checks for unresolved requirements and combines their findings under explicit evidence rules. Experiments show improved agreement with reference labels over the compared baselines across these settings, with the largest offline gains on hard cases and stronger rejection of false completion claims online. Further analyses connect these gains to complementary evidence and test-time exploration. 

---
# Comparing Human Role-Players and LLM-Simulated Clients in Online Counselling Training: An Analysis of Counselling Patterns 

**Authors**: Eric Rudolph, Philipp Steigerwald, Jens Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2610.40563)  

**Abstract**: This study investigates the capabilities of Large Language Models to simulate counselling clients in educational role-plays in comparison to human role-players. Initially, we recorded role-playing sessions, where novice counsellors interacted with human peers acting as clients, followed by role-plays between humans and clients simulated by Mistrals Mixtral 8x7b using 4-bit quantization. These interactions were analysed with a counselling communication pattern system at sentence level. We investigated two key questions: (1) to what extent LLM-generated responses replicate authentic conversational dynamics and (2) whether counsellors' communication behaviour differs when interacting with human versus LLM-simulated clients. The findings highlight both similarities and differences in the application of counselling patterns across scenarios, showing the potential of LLM-based role-playing exercises to enhance counselling competencies and to identify areas for further refinement in virtual client simulations. 

---
# FARMS: A Verifiable Foundation for Automated Computational Science 

**Authors**: Di Wu, Ling Liang, Haizhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.40560)  

**Abstract**: Automated computational science could accelerate discovery by selecting, adapting and developing numerical methods. This prospect requires mathematical knowledge that AI systems can interpret and reuse with explicit guarantees. Here we present FARMS, a faithful and efficient foundation that transforms computational mathematics into AI-ready computing knowledge. FARMS connects formal statements to source context, algorithmic meaning and quantitative guarantees through source preparation, reusable representations and semantic review. Across 171 targets in randomized numerical linear algebra and numerical partial differential equations, 157 were accepted by the pipeline, with mean estimated costs of US \$0.85-2.29 per target across sources. Lean-checked implications from explicit source premises connect selected statements to computational skills, illustrated by a heat-flow solve and wave-response compression. FARMS therefore supports an affordable, reusable knowledge foundation for trustworthy automated computational science. 

---
# Advancing Clinical Genomic Analysis with a Skill-Centred Agent 

**Authors**: Qiaoyu Zheng, Weike Zhao, Yanjie Fan, Ye Cao, Mengqi Jiang, Xiao Zhou, Xiaoman Zhang, Pengcheng Qiu, Yongguo Yu, Yanfeng Wang, Ya Zhang, Chaoyi Wu, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2610.40556)  

**Abstract**: Clinical genomic analysis requires experts to integrate evolving guidelines, databases, and computational evidence. As sequencing has become increasingly accessible, this complex and labor-intensive process has become a major bottleneck to the timely use of genomic information in clinical decision-making. To fill this gap, we developed ClinGeX, a large-language-model agent that composes 105 curated executable skills through task-specific retrieval and dependency-aware orchestration. On 223 temporally held-out benchmark cases, ClinGeX achieved 71.7\% macro-averaged task-level concordance with expert reference classifications, compared with 45.7\% for the strongest comparator. In a three-condition, counterbalanced human-AI collaboration study, 45 clinical genomic professionals assessed 135 clinically derived cases under routine (no AI assistance), DeepSeek-assisted and ClinGeX-assisted conditions. ClinGeX assistance increased raw case-level concordance across criterion-level variant analysis, genomic interpretation in rare disease and molecular oncology decision-making (51.1\%, 57.8\%, 64.4\% versus 15.6\%, 31.1\%, 20.0\% in comparator conditions) and reduced reported interpretation time in all scenarios. During an exploratory one-month real-world non-interventional assessment, 11 experienced professionals submitted feedback on 96 self-selected analysis episodes and reported frequent conclusion adoption and time savings. These findings support executable domain skills as a promising approach to augment specialist genomic interpretation. 

---
# Ramanujan Challenge Results: AI-Assisted Evaluation Pipeline for Multi-Format Proofs 

**Authors**: Alan Sorani, Ashvni Narayanan, Michael Shalyt, Tali Monderer, Rotem Kalisch, Hila Barkan, Elyasheev Leibtag, Shachar Weinbaum, Carsten Schneider, John Campbell, Ido Kaminer  

**Link**: [PDF](https://arxiv.org/pdf/2610.40551)  

**Abstract**: As AI's mathematical capabilities rapidly improve, meaningful benchmarks are shifting from university-level problems and math olympiads to research-level mathematics. The Ramanujan Challenge is a benchmark comprised of research-level problems in number theory, including unpublished results and open conjectures. It presented concrete formulas for mathematical constants whose correctness can be checked numerically, while their proofs may require non-obvious mathematics. Here, we analyze the Challenge submissions, measuring the relative success of different AI models, Computer Algebra Systems (CAS), and formal tools. We find correct submissions that are conceptually distinct from the official solutions, and a proof of one of the published open conjectures. Evaluating the submissions exposed a second challenge: judging complex proofs reliably at scale. We introduce an AI-assisted evaluation pipeline for natural language proofs, CAS code, and Lean formalizations that assesses semantic and logical correctness. Compared with the conventional one-shot LLM judgment, our pipeline reduces false-positive verdicts on natural language proofs from 23% to 6% and improves judgment accuracy on supporting code from 46% to 58%. This approach provides a methodology for evaluating research-level mathematics at scale, with potential applications across mathematics and AI for science. 

---
# Retrieval as Traversal: Value-Indexed Memory for Stateful Long-Context Agents 

**Authors**: Daniel Strauß, Zhenshan Bing, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2610.40545)  

**Abstract**: Long-context models repeatedly solve closely related retrieval problems. Attention or global retrieval can retain large stores, but each forward pass again searches for what matters. Recurrent compression avoids this search only by deciding earlier what information to preserve. Neither mechanism directly exploits that useful memories at neighbouring retrieval steps may substantially overlap. We propose an alternative in which retrieval is a persistent local traversal through an address space. We separate a memory's content from its address. Utility is defined by the information revealed by retrieving the content, while geometry determines only how cheaply that information can be reached. This leads to a cost-regularised indexing objective that preserves information value inside local neighbourhoods while charging for every memory made locally accessible. Its marginal-value form gives both attractive and exclusion pressures. Thus, complementary memories should be co-retrievable when the additional value exceeds retrieval cost, whereas redundant memories should not occupy the same retrieval set solely because they are individually useful. As a proof of principle, we test only the simplest fixed-geometry instance on MiniGrid Memory. A persistent local traversal head reaches $0.95$ mean return after $0.59$M training steps with a five-token context, compared with $0.94$M for Gaussian absolute addressing and more than $2$M for categorical absolute addressing. The experiment does not test learned geometry or repeated retrieval. Instead, it motivates the hypothesis that stateful retrieval may be easier to learn than stateless. To test the full proposed method further experimentation is necessary. 

---
# On the Fundamental Limits of Prefill-Based Decoding Length Prediction 

**Authors**: Haowen Shi, Xuzhe Yan, Fahao Chen, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.40541)  

**Abstract**: Predicting how many tokens a large language model (LLM) will decode, using only prefill-phase hidden states, is a key primitive for inference scheduling. We show that current benchmarks conflate predictor quality with task hardness: when the target model, dataset, or sampling configuration changes, reported accuracy collapses even though the predictor is unchanged. We trace this instability to an overlooked error source---the inherent stochasticity of autoregressive decoding---and derive bounds that every prefill-side method must obey. Our analysis treats decoding as a stopped stochastic process. The per-step stopping probability of a response is not constant but ramps up over hundreds of steps, so the decode length inherits a central limit theorem from the many local decisions that determine where a response ends; this yields per-prompt normality with an explicit error term, together with an affine variance law $\sigma=\alpha\mu+b$ whose slope is the coefficient of variation of the number of response segments. Empirically, the normal law reproduces the per-prompt MAE and the central mass of the decode-length distribution to within a few percent, and the residual departures from normality (mild right skew, slight excess kurtosis) are the ones predicted by the sampling-noise correction. Under fixed model, data, and sampling conditions, a broad design space of predictors (MLP, EGTP, FiLM, soft-label, and Mixture-of-Experts variants) converges to similar performance close to the bound, confirming that \emph{system error dominates model error}. We argue for a benchmarking protocol that fixes the model, data, and sampling parameters and that reports relative-error metrics alongside absolute ones. Code and analysis tools are available at this https URL. 

---
# Efficient Scalable Oversight via Rewinding 

**Authors**: Ziyi Guan, Yael Tauman Kalai, Ron Rothblum, Zoe Xi  

**Link**: [PDF](https://arxiv.org/pdf/2610.40535)  

**Abstract**: Scalable oversight asks how a weak verifier can check the outputs of computations performed by more powerful AI systems. One proposed approach uses relativizing interactive proofs, in which a weak verifier checks the output of a complex oracle-aided computation by interacting with one or more powerful provers. A prominent example is debate, which is highly efficient but assumes that one of two powerful AI models, or provers, is truthful. For certain classes of oracle-aided computations, this assumption can be removed, but at substantially greater computational cost.
In this work, we introduce \emph{rewindable interactive proofs} (RIPs), a new model of relativizing interactive proofs for scalable oversight. In an RIP, the verifier may rewind the prover to an earlier state of the interaction and continue from there. This is possible when the prover is an AI model, since the verifier can save a snapshot of the model's state and later restore it. We construct a relativizing RIP for computations whose outputs are robust to changing a small fraction of oracle answers. Our construction is more efficient than prior relativizing proofs that do not assume a truthful prover, and it uses no cryptography.
In our construction the honest prover runs in time linear in that of the original computation, and the verifier, after a one-time preprocessing step, runs in time nearly linear in the input length (which can be much smaller than the computation itself). We implement our protocol, report a preliminary MNIST experiment, and show that an RIP can be faster than a state-of-the-art deployed proof system.
RIPs may be of independent interest. Rewinding is possible whenever the verifier controls the prover's machine, for example when the prover runs on trusted hardware or in a virtual machine whose state can be checkpointed. 

---
# Algebraic World Model: From Exploration to Proofs 

**Authors**: Guancheng Pan, Qixin Tan, Yang Yuan, Andrew Chi-Chih Yao  

**Link**: [PDF](https://arxiv.org/pdf/2610.40503)  

**Abstract**: AI has made remarkable progress in mathematics, solving challenging problems with machine-checkable proofs in Lean, yet typically relying on predefined mathematical definitions and formal specifications. This raises a natural question: can it reason rigorously about an unknown world? We introduce the Algebraic World Model (AWM), a framework that turns interaction with unfamiliar environments into explicit mathematical models supporting formal reasoning. AWM explores the environment and develops executable Python hypotheses, continually testing and revising them against observations. It then structures these hypotheses into typed objects and shared rules, simplifies their structure through category-theory-guided transformations, and formalizes the resulting model in Lean 4 to prove reachability and impossibility. AWM can formally prove that a goal is unreachable, even beyond explored trajectories. On two impossible game levels, it constructs closed, kernel-checked Lean proofs excluding every legal winning sequence in all six runs, while Claude Opus 5.5 and GPT-6 Astra, equipped with coding tools, produce none in twelve baseline runs. Four of AWM's six initial bounds are correct, compared with one for Opus 5.5 and none for GPT-6 Astra; given only feedback that a cheaper win exists, AWM revises its rules and constructs new proofs, reaching six of six correct bounds. Moreover, AWM produces interpretable mathematical models whose rules can be inspected and tested step by step. Across all 25 public ARC-AGI-3 games, these models enable Lean to replay the winning trajectories of all 183 levels and generalize beyond observed play, predicting unseen transitions and detecting changes in game mechanics. With Claude Opus 5, AWM clears all levels with a score of 99.9, while its certified winning trajectories use only 43% of the human baseline's actions. 

---
# Agentic University: Collaborative Agents for Personalized Learning 

**Authors**: Jingru Fan, Tianle Zhou, Yunjie Zhou, Zihao He, Yixuan Zhou, Kezhao Wu, Xinyu Zheng, Xiyuan Wang, Yuheng Wang, Zhi Li, Songhua Liu, Weinan E, Chen Qian  

**Link**: [PDF](https://arxiv.org/pdf/2610.40492)  

**Abstract**: Large language models support personalized tutoring, content generation, and adaptive learning paths. However, capability development requires these resources and interactions to form a coherent learning process that evolves with the learner. A central challenge is to align goals, activities, and assessment while using learning evidence to decide what should follow. We introduce Human-Participatory Multi-Agent Organization and instantiate it in Agentic University (AU). In AU, learners participate as organization members and work with specialized agents to maintain a revisable Capability Program. The program connects personal goals, learning activities across modalities, and layered assessment, while learner feedback and performance guide subsequent support. Our evaluation shows advantages in program organization, assessment design, and revision quality. Users achieved larger mean assessment gains (12.19 versus 5.61 points on a 50-point scale) and reported stronger cognitive support. This work contributes an organizational approach to adaptive learning that makes learner participation a continuing basis for multi-agent coordination. The Agentic University project is available at \href{this https URL}. 

---
# FIRME: Registry-Anchored Memory Graphs for Evolving, Heterogeneous Investor Knowledge 

**Authors**: Yunbai Zhang, Yunyi Gao, Han-Tai Shiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.40482)  

**Abstract**: Financial AI workflows that reuse investor knowledge need persistent memory that integrates observations across sources and periods without losing identity or provenance. In a knowledge graph, recurring observations may use different surface forms, fragmenting relationship histories and preventing evidence from accumulating. We introduce Fusion and Identity Registry for Memory Evolution (FIRME), a registry-anchored memory graph that reconciles incoming observations to persistent canonical relationships and retains their source, time, and context. Within periods, rule-based fusion links investor-call discussions to portfolio cash-flow signals, directly or through a fixed product-topic ontology. Across periods, canonical identities and persistent aliases direct recurring observations to the same relationship history; deterministic updates summarize reinforcement, inactivity, and reactivation. We evaluate FIRME on proprietary investor-call and cash-flow records and on S&P 500 earnings-call transcripts. Compared with a historical registry that retains canonical identities and semantic fallback but omits alias memory, FIRME increases final-period Correct Match Rate by 7.8-12.1 percentage points across four dataset-backend settings under a fixed-judge and fallback-review protocol. A blinded human audit of 200 incoming observations per setting finds Correct Match Rate gains of 9.0-16.5 points and reattachment-recall gains of 22.5-29.0 points, with higher observed merge precision in all four samples. A separate audit finds valid evidence paths for 293 of 300 outputs from full within-period fusion in the investor-call setting. Extraction-faithfulness and round-trip diagnostics assess source grounding and preservation of extractor-derived graph content. 

---
# Training language models in simulation for psychiatric intake 

**Authors**: Samuel Schmidgall, Tim Strother, Alex Bijamov, Min Woo Sun, Anil Palepu, Akshay Goel, Ulysse Marteau-Ferey, Théo Moutakanni, Yubin Kim, Joseph Breda, Christina Chen, Justin Chen, Paul Jhun, Andrew Marshall, Sarah Montgomery-Taylor, Eun Kyoung Hong, Shashank Ravi, Kameron C. Black, David J. H. Wu, Austin Schoeffler, John Lyon Havlik, Elizabeth C. Stade, Jared Moore, Johannes C. Eichstaedt, Logan Pierce, Jeff Jopling, Johannes Thrul, Thomas H. McCoy, Neguine Rezaii, Mark S. Chen, Ellen E. Fitzsimmons-Craft, Shirley Wang, Desmond C. Ong, Katharina Schultebraucks, Torrey Creed, Bhavna Daryani, Aleksandra Faust, S. Sara Mahdavi, Caroline Figueroa, Eric Lin, Kate H. Bentley, Ernest Lim, Zhihan Jiang, Xuhai "Orson" Xu, Daniel McDuff, John Torous, Yun Liu, Po-Hsuan Cameron Chen, Mike Schaekermann, Joelle Barral, Alvin Rajkomar, Tao Tu, Lin Yang, Isaac Galatzer-Levy, David Racz, Valentin Liévin  

**Link**: [PDF](https://arxiv.org/pdf/2610.40476)  

**Abstract**: In psychiatric practice, the clinical intake interview remains a primary diagnostic instrument, requiring the clinician to establish rapport while disentangling complex, overlapping presentations. Here, we show that multi-turn reinforcement learning (RL) in simulation environments improves language model performance on structured text-based psychiatric intake, spanning history-taking, differential diagnosis, and initial management planning. In a randomized, double-blind crossover study ($N = 60$ matched scenario pairs, 120 encounters) comparing the RL-trained agent against its identically prompted base model, 32 board-certified psychiatrists role-played standardized patients while a separate panel of 15 independent, board-certified psychiatrists scored the de-identified transcripts. The trained agent outperformed the base model in 76.7% of matched scenario pairs ($p < 0.0001$), with gains on an adapted clinical-examination rubric ($d_z = 0.77$, $p < 0.0001$), patient-centered communication (82.0% vs. 65.2%, $p < 0.001$), and differential diagnosis appropriateness (68.3% vs. 33.3%, $p < 0.001$). Across 127 psychiatrist-validated simulated intake scenarios, the trained agent nearly doubled its base model's clinical rubric score (0.706 vs. 0.379; composite reward 0.564 vs. 0.294, $p < 0.0001$). To systematically probe conversational safety, we established two Delphi-consensus taxonomies, of clinical safety risks and relational competencies, from which 256 challenging adversarial presentations were derived; we find that training in simulation environments more than halved under-triage of acute safety risks (12.9% vs. 30.5%, $p < 0.0001$). Simulation environments thus provide a controlled testbed for training and stress-testing psychiatric conversational agents, with gains that transferred to psychiatrist-enacted encounters, though residual gaps require safeguards and clinician oversight. 

---
# Prefix Executability: Evaluating Tool-Using Agents Beyond Final Success 

**Authors**: Amir H. Rezaeian, Weiyi Sun, Yassine Benajiba, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2610.40444)  

**Abstract**: Tool-use benchmarks often score agents by final task success, but terminal outcomes can obscure whether the intermediate tool-call trajectory was executable. We propose prefix executability as a trajectory-level evaluation protocol for tool-using agents. The protocol measures whether every prefix of a predicted tool-call sequence remains executable under tool schemas, state preconditions, and data dependencies, and summarizes reliability using Prefix Executability Curves (PEC), First-Error Depth (FED), Trajectory Executability Rate (TER), horizon-normalized AUC-PEC, and Recovery Rate (RR). We instantiate the protocol on stratified samples from BFCL-v3 multi-turn and ToolSandbox, and on the complete Airline/Retail base splits of $\tau^3$-bench. We observe that outcome-only and prefix-level rankings diverge: agents whose execution is conditioned on an unverified plan often achieve higher point-estimate final success than non-plan-conditioned baselines while substantially lowering TER and AUC-PEC. These divergences are precisely the failure modes that prefix executability is designed to expose: they reveal whether apparent success rests on an executable trajectory, where invalidity first enters, and whether success follows recovery from an earlier failure. As a validation intervention, explicit pre-execution plan checking can improve executable-prefix survival while maintaining competitive terminal success. These results show that final success alone is insufficient for evaluating tool-use agents and that prefix-level executability provides an actionable reliability signal for comparing agents, diagnosing failures, and assessing validation mechanisms. 

---
# Retrieving from Your Own Mistakes: On-Policy Training of Embedding Models for Multi-Hop Retrieval-Augmented Generation 

**Authors**: Chenjie Qiu, Zixuan Zeng, Shaobin Shi, Bingfeng Chen, Dongsheng Luo, Boyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.40409)  

**Abstract**: Complex questions in retrieval-augmented generation require evidence spread over several documents, and an answer is reliable only when the retriever finds all of it. Chain embedding retrievers collect such evidence step by step, re-encoding the question together with the passages retrieved so far, and we find that they are already strong: trained only on gold histories, a small chain embedding retriever that calls no language model outperforms graph-based systems and is competitive with a search agent. This training, however, leaves a gap. At test time the retriever conditions on its own retrievals, which soon contain mistakes: by the third step, about half or more of the states it visits hold a wrong passage, and these passages lower its chance of retrieving a missing gold passage well beyond what the difficulty of the states explains. We propose on-policy state training, which trains the encoder on the states it reaches itself. At every epoch the current encoder retrieves chains for the training questions; each state it visits, including those built on its own mistakes, is labeled with the earliest gold passage still missing while the other missing gold passages are masked, so that no correct continuation is penalized; and the encoder is updated contrastively on these states mixed with teacher-forced ones, so that its training distribution follows it as it learns. The resulting retriever retrieves complete evidence more often than teacher-forced training and the systems we compare, and it transfers without further training to harder long-chain variants of HotpotQA and 2WikiMultiHopQA that we construct. On the same states, it recovers from its own mistakes far more often, and controlled comparisons attribute the gain to training on the states the encoder currently reaches. 

---
# Beyond Selected Breakthroughs: Evaluating Language Models on Randomly Sampled Mathematical Conjectures 

**Authors**: Helen Jenne, Davis Brown, Jesse He, Henry Kvinge  

**Link**: [PDF](https://arxiv.org/pdf/2610.40382)  

**Abstract**: The last year has seen a shift in AI-assisted mathematics, with frontier language models producing proofs of and counterexamples to long-standing open conjectures. However, studying only the current set of major breakthroughs provides an incomplete picture of LLM capabilities and the computational resources needed to achieve reported results. To address this, we introduce a new dataset, OpenConjecture, which collects conjectures from mathematics papers published on the arXiv since February 2026. We use these conjectures as a source of open mathematical problems, and apply frontier GPT models to a random sample of 200 of these conjectures over three passes using a reproducible protocol without problem-specific mathematical guidance. Rather than attempt expert verification of these candidate solutions, we analyze the types of problems for which LLMs produce solutions and the characteristics of these solutions. The models we test produce candidate solutions for roughly 60% of the conjectures, with counterexamples about 1.3 times more common than proofs. We estimate solution novelty, assess how solutions cite and relate to existing literature, and analyze whether the proposed resolution relies on either an error in or underspecification of the conjecture. We find that models sometimes propose resolutions that are outside the authors' intent (e.g., proposing the trivial group as a counterexample) and that a number of proofs follow closely from existing results, but that some resolutions exhibit at least moderate novelty. Our findings underscore the need for evaluations that go beyond correctness to assess fidelity to the intended statement, attribution, and the nature of the mathematical contribution. 

---
# Multistep and Multivariate Time Series Explainability with GradientSHAP: A Wind Power Case Study 

**Authors**: Robbie Southam, Zhong Fan, Ke Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.40330)  

**Abstract**: As the effects of climate change become increasingly apparent, the demand for renewable energy sources such as wind power continues to grow. However, renewables' variable and intermittent nature presents significant challenges in their integration into energy grids. Artificial intelligence (AI) has emerged as a powerful tool to address these issues. Still, its adoption in the energy sector has been hindered by the opacity of many AI models, which often function as black boxes. This paper introduces a carefully adapted GradientSHAP approach to provide interpretability for multivariate, multistep wind power forecasting -- a largely underexplored area. This capability is critical for day-ahead grid management and, to the best of our knowledge, this is the first study to apply GradientSHAP in the context of wind power forecasting. To address the inherent dimensionality, we developed novel visualizations that deliver both local and global explanations, uncovering intricate relationships between environmental factors, site-specific conditions, and temporal dynamics. While GradientSHAP is not an out-of-the-box solution, our findings demonstrate that, with thoughtful handling of SHAP values and domain expertise, it can offer valuable insights into model behaviour, supporting more informed and trustworthy decision-making. Nevertheless, its complexity poses challenges for practical implementation. Future research should focus on streamlining these interpretability methods to make them more actionable for routine deployment in critical energy applications. 

---
# Neighborhood Elites: Quality-Diversity Search for Uncertainty Quantification in Question Answering 

**Authors**: Mesut Ceylan, Arda Baris Basaran, Elgar Fleisch, Filipe Barata  

**Link**: [PDF](https://arxiv.org/pdf/2610.40278)  

**Abstract**: Large language models (LLMs) are increasingly capable, yet they produce wrong answers indistinguishable from correct ones. In question answering (QA), such an error propagates unnoticed into downstream tasks. Perturbation-based uncertainty quantification (UQ) addresses this by probing a model over a set of input rewrites and reading uncertainty from how much its predictions vary. Yet this set is usually built generically, by free paraphrasing, token-level perturbation, or adversarial search. Such rewrites leave two things uncontrolled that matter in QA: what kind of change is made to the context, and how explicitly the context supports the answer. When neither is controlled, the composition of the set is determined by the generator rather than by design, and the estimate inherits whatever that generator happens to produce. Quality-diversity (QD) search is built to cover a predefined exploration space, so we recast input perturbation as a QD problem and propose \emph{neighborhood elites (NE)}, a neighborhood constructor that spreads answer-preserving rewrites across two axes, the kind of edit and the explicitness of support. We compare NE against three other neighborhood constructors, holding the editor, QA model, uncertainty measure and search objective fixed, and against three scores read from the original context. Across five extractive-QA datasets and five instruction-tuned LLMs, NE improves failure detection against every other constructor in at least 24 of the 25 dataset-model pairs, reaching 0.726 AUROC and 0.211 AUPRC lift over the base rate. Additionally, NE detects failure better than semantic entropy computed over decoder samples, by 0.074 AUROC and 0.062 AUPRC lift, and this advantage widens as the neighborhood grows. 

---
# Distributional Reward Calibration and Budget-Neutral Credit Assignment for Multimodal RLVR 

**Authors**: Lang Chen, Zongxin Xu, Borui Wang, Anran Liu, Shitong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.40250)  

**Abstract**: We revisit advantage estimation in Group Relative Policy Optimization (GRPO) and observe that any token-level advantage admits a two-stage factorization: a \emph{sample-level baselining} step that converts scalar verifier rewards into a comparable sequence-level signal, and a \emph{token-level credit assignment} step that distributes that signal over the trajectory. Existing estimators typically address only one of these stages: affine baselines preserve the shape of each task's reward law, so heavy tails and binary cliffs propagate into the gradient, while token-level credit rules tie the size of each token's update to raw confidence scales and need not conserve the total credit issued by the verifier. We propose \textbf{TACE} (Two-stage Advantage and Credit Estimation), a two-stage estimator that addresses both stages. At stage~(S1), we replace linear standardization with a one-dimensional optimal-transport map to a standard normal, which gives a common population reward law across tasks and caps finite-sample outliers deterministically. At stage~(S2), we rank-calibrate a stop-gradient evidence contrast within each trajectory and renormalize the clipped token weights to have mean one, so credit is redistributed across tokens while its total mass and sign are conserved. We prove population-level inter-task moment equity, a deterministic tail bound, and mass-conservation, boundedness, and sign-anchoring properties that affine baselines and unnormalized token weights do not satisfy. On a multimodal reasoning suite, TACE obtains the strongest results on a majority of the benchmarks but does not dominate every setting. 

---
# Reaction precedents extend retrosynthetic prediction beyond training coverage 

**Authors**: Jiaxing Sun, Xuli Shen, Jiang Wu, Ruiyi Ding, Ruikang Zhang, Xiao An, Xibao Li, Yubin Wang, Xingjian Wei, Yinfan Wang, Yu Liu, Lingli Ge, Xi Xue, Xiaojian Wang, Lijun Wu, Chang-Yu Hsieh, Yuqiang Li, Bowen Zhou, Dahua Lin, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2610.40249)  

**Abstract**: Reaction precedents guide retrosynthesis, but how far they can compensate for gaps in task-specific training coverage remains unclear. Using 2.71 million reactions, we restricted training and retrieval coverage by transformation or time, then expanded only the retrieval corpus with model parameters fixed. For transformations excluded from training, mean Top-1 accuracy increased from 13.0% to 58.0% on a historical test set and from 4.9% to 23.1% on a temporally held-out test set. For temporal coverage gaps, retrieval expansion enabled a model trained on records before 2017 to match or exceed a non-retrieval model trained on records before 2025 on both test sets. Without task-specific fine-tuning, precedents improved all 14 representative general-purpose models on both test sets and USPTO-50K; the strongest with precedents surpassed all eight representative specialized models. These findings show that inference-time reaction precedents can extend retrosynthetic prediction beyond task-specific training coverage without parameter updates. 

---
# EnvOS: Auditing Computer-Use Agents When the World Moves On : Phenomenon-Based Task Construction, Certification, and Evidence-Based Evaluation 

**Authors**: Ankush Kadu, Athira Gopal, Niharika Singh, Arpit Jain, Aswanth Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2610.40246)  

**Abstract**: Computer-use agents built on large language models (LLMs) work in software that can change while they act: an appointment moves, a policy is revised, a record is updated. Testing agents under such changes is harder than it looks. When an agent fails, the cause may be the agent, but it may also be a change that did not fire as intended, an update the agent never received, or a grader that scored the episode wrongly. We introduce EnvOS, a framework for building and certifying environments that change mid-task. Before any agent runs, EnvOS checks that each change fires correctly, that the task remains solvable, and that the grader separates correct from incorrect behavior; after each run, separate code re-checks the verdict from archived evidence. We apply it to stale information, where a value the agent has already seen changes before the agent acts on it. In clinic booking, Claude Opus 5 succeeds on 0/16 tasks when nothing extra is shown, 2/16 when reminded to re-check the source, and 15/16 when the updated value is shown; the gain over the reminder holds after correcting for all tests (p = 0.00293). In insurance-claim approval, Opus 5 re-checks on its own and succeeds on 10/10, whereas Claude Sonnet 5 rises from 1/10 to 9/10 when the current policy is shown again, a gain significant in its pre-registered test but not after correcting for all tests. Whether an agent caught a change thus varied with the model and the interface. We also stress-test the grading itself: on corrupted episodes designed blind from the rules, the EnvOS pipeline accepted none of 26 as clean, and on an externally generated set it caught 25 of 39, against 4 for simple log checks. Because every EnvOS environment is generated from a seed and graded by an audited automatic verifier, the same environments are natural candidates for training agents with reinforcement learning from verifiable rewards. 

---
# EpiAgent: Epistemic Self-Play for LLM Agents 

**Authors**: Xuanfa Jin, Xinyu Cui, Zhijian Ma, Xue Yan, Jian Zhao, Haifeng Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.40227)  

**Abstract**: Epistemic game theory explains an agent's strategic behavior via the beliefs it holds about others and how these beliefs are updated as the interactions unfold. Modern LLM-based agents, however, often lack an explicit mechanism for learning and updating such beliefs to guide decisions under incomplete information. To bridge this gap, we propose EpiAgent, a framework that equips LLM-based agents with an explicit, learnable epistemic layer, which maintains beliefs over others' private information and combines these beliefs with the agent's available information into an epistemic state for action selection. Concretely, the agent's policy is decomposed into two coupled stages: Epistemic Update, which updates these beliefs from interaction histories, and Action Planning, which selects actions conditioned on the updated epistemic state. By formulating their dependency as a bi-level optimization problem, we introduce Epistemic Self-Play, an algorithm that alternately optimizes the two stages through self-play, utilizing an intrinsic prediction reward to evaluate how well inferred beliefs predict subsequent observations. Empirical results across both mixed and competitive scenarios show that EpiAgent consistently outperforms prompting-based and RL-based baselines. Further analyses reveal the benefits of dedicated epistemic training, improvements in belief inference accuracy, and the contribution of learned beliefs to action planning. These results demonstrate the value of an explicit, learnable epistemic layer for effective multi-agent interactions. 

---
# Bridging Semantics and Structure: A Theoretically Grounded Framework for Link Prediction on TAGs 

**Authors**: Chen Shao, Ruirui Wang, Tobias Kafer, Danai Koutra  

**Link**: [PDF](https://arxiv.org/pdf/2610.40192)  

**Abstract**: Link prediction on text-attributed graphs (TAGs) is a fundamental problem with applications in knowledge discovery, recommendation and beyond. Recent progress has highlighted the promise of combining Graph Neural Networks (GNNs) with Language Models (LMs) in order to leverage the complementary structural and semantic signals. However, due to the distinct inductive biases of GNNs and LMs, the design of an optimal fusion strategy remains an open research question. In this work, we present Bridge-LP, an intuitive, simple, efficient and theoretically grounded framework that unifies graph and language representations. It is built around three theoretically-justified design components: (D1) jumping connection of textual proximity, which enables robust LM training; (D2) compressed embedding, which encodes neighborhood information long exploited by classical link prediction heuristics, thereby enriching structural signals in a learnable way; and (D3) cache embedding training strategy, which reduces computational complexity. Together, these mechanisms establish an end-to-end cross-modal framework that scales to large graphs while capturing richer graph-text interactions. We conduct an extensive empirical evaluation, comparing against 40 existing models across 9 diverse datasets. Experiments show that Bridge-LP consistently outperforms state-of-the-art approaches, delivering up to a 43.06% improvement in Mean Reciprocal Rank (MRR) and setting a new benchmark for language-graph integration. 

---
# Locating and Closing Information Gaps in Multi-Agent Systems 

**Authors**: Yaxin Gong, Chongming Gao, Gangyi Zhang, Chenxiao Fan, Leyang Shen, Jiakai Wang, Dong Wang, Yang Liu, Haoyan Liu, Xinting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.40186)  

**Abstract**: Multi-step Web tasks often require several LLM agents to gather and combine evidence. The communication graph of a fixed workflow restricts which agents can read each other's outputs: evidence retrieved by one agent may never reach the agent that needs it, leaving an information gap and an incomplete or incorrect answer. Adding a connection can deliver the missing evidence, but can also lead the receiver to adopt an incorrect upstream conclusion that displaces a correct answer; the final output alone does not separate the two. We propose Selective Probing and Link Insertion for Closing information gaps in multi-agent Execution (SPLICE), which locates information gaps per query and decides what to adopt after adding an edge. After the original workflow runs, SPLICE uses the query and node outputs to locate a possible gap, adds one directed edge from a source to a receiver to deliver the relevant evidence, and re-executes the receiver and affected downstream nodes, producing a candidate answer. The adoption decision reads both the final answers of the two runs and the process changes the edit caused inside the workflow: how far the receiver moved, where its new content came from, and whether the change reached the final answer. A selector trained on these process features chooses between adopting the candidate answer and retaining the original. On five benchmarks, SPLICE raises the task score on Web tool use by 8.0% and the mean score on four reasoning and planning benchmarks by 5.7% relative to the static skeleton, at 1.25 to 1.44$\times$ the skeleton's cost, and transfers zero-shot to four unseen backbones. 

---
# A Negotiation-Aware Hybrid LLM Framework for Intent Interpretation and Service Discovery 

**Authors**: Wided Meflah, Hayet Brabra, Walid Gaaloul  

**Link**: [PDF](https://arxiv.org/pdf/2610.40161)  

**Abstract**: Increasingly, users express their needs as natural-language intents rather than structured requests. These intents may be incomplete, high-level, or internally conflicting. A conflicting intent either violates fundamental properties of the cloud paradigm or states mutually inconsistent requirements. Classical composition approaches are designed for structured, low-level, and complete requests. They cannot interpret such intents, discover suitable services from provider catalogs, or negotiate conflicting requirements with users. To address this gap, we propose a negotiation-aware framework for intent interpretation and service discovery. The framework first refines the natural-language intent and dynamically detects conflicts. It then structures the intent into a provider-independent TOSCA template. For each specification, it retrieves candidate service offers through deterministic matching grounded in cloud provider offerings. Some conflicts arise during interpretation. Other requirements prove unsatisfiable during discovery. Both are resolved through iterative natural-language negotiation with the user. We evaluate the framework on 70 diverse intents against single-shot and enhanced-prompting baselines. It improves recall and F1 for both conflict detection and service discovery. It also eliminates hallucinated offers and resolves conflicts at a cost proportional to their number. 

---
# The generative paradox: the Anti-Prompt paradigm for human capability 

**Authors**: Margarita Trifonov Vadim E. Levit  

**Link**: [PDF](https://arxiv.org/pdf/2610.40151)  

**Abstract**: Generative AI in education is usually judged by the output of human-AI teams. That focus overlooks a second outcome: what people can still understand, challenge, repair, and adapt when assistance changes or disappears. We call this tension the generative paradox and distinguish four manifestations: performance, assistance, ownership, and capability stock. We propose developmental function allocation, which complements operational function allocation and meaningful human control by treating retained human capability as a design objective alongside current joint performance. The Anti-Prompt paradigm reverses the default interaction order: when immediate generation would preempt a consequential human act, AI should first create the conditions for human construction. We distinguish the validity of a product, responsibility for its construction, and evidence of capability after support changes. Representative Anti-Prompts, a canonical interaction sequence, and stress tests spanning mathematical reasoning, experimental science, software diagnosis, and lesson design make these distinctions concrete. We conclude with falsifiable predictions about retained capability and a system-level hypothesis about the reproduction and distribution of expertise. 

---
# DynaRev: Benchmarking Web Agents under Dynamic Goal Revision 

**Authors**: Shiqi Zhang, Guochen Zhou, Zesheng Wei, Hai Zhao, Zhisong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.40133)  

**Abstract**: Web agents are typically evaluated with goals fully specified before execution and fixed thereafter, whereas real users often revise their requirements during interaction. To study such interactions, we introduce DynaRev, a benchmark of 181 tasks to evaluate web agents under dynamic goal revision. DynaRev uses semantic milestones to deliver revisions at task-defined stages, covers additive revisions, unconditional overrides, and runtime-conditioned overrides, and evaluates correctness, responsiveness, and efficiency through trajectory-grounded evaluation. Five-model evaluations show that agents often satisfy only part of the task requirements, with performance varying across task sources, revision types, and models. Further analyses reveal varied milestone timing, mixed effects of dynamic delivery, and contrasting revision-handling outcomes, with gains from instruction rewriting while preserving browser state and losses from browser resets. DynaRev provides a testbed for studying how web agents adapt to evolving user goals. The code and data are available at this https URL. 

---
# Reward-Adaptive Co-Evolution of Policy and Rubric 

**Authors**: Yizhu Chen, Jianguang Jiang, Jiameng Huang, Chuyu Fang, Bojun Zhang, Zhenyu Jiao, Xing Yuan, Tong Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.40096)  

**Abstract**: Rubric-based rewards provide fine-grained, interpretable supervision for open-ended reinforcement learning, where holistic scalar rewards struggle to capture multidimensional response quality. Static rubrics can lose discriminative power as the policy distribution evolves, while existing policy--rubric co-evolution methods rely on globally prescribed update schedules that ignore heterogeneous reward informativeness. We introduce RACE, a Reward-Adaptive Co-Evolution framework. At the group level, per-rollout-group training-signal routing classifies groups as informative, diversity-collapsed, or rubric-failed, updating the policy only with informative groups while caching rubric failures. At the phase level, event-triggered phase switching enters rubric optimization upon persistent reward saturation and returns when rubric-reward improvement plateaus. Rubric phases train the generator on cached pairs using a frozen judge and a scale-normalized pairwise-margin reward. Across both levels, online threshold adaptation adjusts routing and switching thresholds through proportional control. Under a unified evaluation protocol, RACE achieves the highest average performance and outperforms the strongest fixed-alternation baseline on all eight tasks, while improving rubric discrimination. 

---
# Agentic process evidence 

**Authors**: Yuval Kainan, Carmit Hershman  

**Link**: [PDF](https://arxiv.org/pdf/2610.40071)  

**Abstract**: The way software gets built is changing dramatically. AI agents now create pull requests, select dependencies, refactor codebases, interpret test failures, and execute infrastructure changes across the entire Software Development Life Cycle (SDLC). AI agents are autonomous team members, but unlike human engineers, they are ephemeral - their reasoning, decisions, and tool calls vanish the moment the session ends. Modern Security incidents involving intent mismatch or injection are becoming increasingly common - forcing us to address AI Agents as an insider threat. No widely adopted standard yet enables agentic provenance and governance and its obvious that you can't hold an ephemeral process accountable. The git log was designed for human commits. It shows what changed. It does not show what an AI agent did, which model it used, which tools it called, or whether anyone was accountable for the outcome. Every governance framework, every audit trail, every compliance control in the SDLC was built around one assumption: a human made the decision. That assumption no longer holds. This shift introduces 3 main procedural problems (not addressing scale) - Supply-chain vector - The agentic coder is software, so it is a supply-chain security risk. Policy guardrails - How to ensure the agentic coder is compliant with the policies of the organization. Task-alignment - How to align the agentic coder on its tasks and goals. We introduce Agentic Process Evidence (APE), a signed, tamper-evident, machine-readable attestation of agentic processes, linking the sessions to their outcome (process-aware logging). APE enables Policy-as-Code control over the agentic processes, allowing for human in the loop only when it's needed. We support the in-toto style attestation framework, and the protocol is aligned with the relevant regulations 

---
# Hybrid AI Patterns: Thirteen Ways to Couple Language Models with Rules, and How to Choose Among Them 

**Authors**: Siegfried Handschuh, Götz-Henrik Wiegand, Lorena Raichle  

**Link**: [PDF](https://arxiv.org/pdf/2610.40036)  

**Abstract**: Language models handle text well, and rule engines give conclusions with proofs. Systems that combine the two are usually described by example, and the same coupling turns up under different names in agent frameworks, guardrail toolkits and neuro-symbolic research. This document defines thirteen patterns for coupling a language model with an executable rule language, in the catalogue form introduced for object-oriented design: intent, motivation, applicability, participants, collaborations, consequences, implementation, known uses and related patterns. Four patterns concern what crosses the boundary between model and rules, five who controls whom, three how the model takes part in the reasoning, and one how knowledge is kept over time. Five invariants hold for every combination, and five failure patterns name the couplings that break them. We relate the catalogue to published pattern catalogues and taxonomies of hybrid and neuro-symbolic systems and show where they overlap with it. A decision guide leads from a problem to a pattern. It rests on our measurements, including the cases in which a model alone did as well as the hybrid or better, and it reports what each design cost. 

---
# Cheap Retrieval and Observation Credit Assignment from One Quantity: The Value of Information 

**Authors**: Daniel Strauss, Nikolaos Sotirakis, Zhenshan Bing, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2610.40030)  

**Abstract**: Memory in reinforcement learning raises two problems. Attention finds the relevant item by scoring the whole store, which is costly for robots under latency and energy limits. When a retrieved memory pays off, credit for acquiring it must cross the whole sequence. We argue that both are governed by one quantity, the value of information, and propose to estimate it from archived rollouts with a learned world model. We obtain that, when read at the observation, it yields a provable lower bound on the optimal value of the acquiring action, which floors its TD target and attributes this floor to the observation that earned it. Read at the decision, it defines the optimal retrieval set. A pilot retrieval system, to be trained and improved with this signal, solves a memory task with reduced context. 

---
# Human--AI Interactions as First-Class Data: A Formal Interaction Model and Runtime for Explainable AI Collaboration 

**Authors**: Michela Taufer, Kin Ng, Ian Lumsden, Aashish Pandey, Rajasi Gore, Angelina Ju, Sukanya Moudgalya, Ewa Deelman, Sai Swaminathan  

**Link**: [PDF](https://arxiv.org/pdf/2610.40006)  

**Abstract**: AI assistants are increasingly embedded in computational notebooks, scientific workflows, software engineering tools, and learning systems, yet human-AI interactions are often treated as transient prompt-response exchanges rather than durable data products, hindering reproducibility, explainability, replay, behavioral analysis, and adaptive AI support. We argue that these interactions form a new data class requiring formal representation and introduce the Human-AI Interaction Description Language (HAIDL), a formal interaction model with a Backus-Naur Form-inspired concrete syntax. HAIDL encodes each interaction as a structured object capturing computational context, user request, prompt payload, AI response, feedback, problem context, provenance, and model metadata. We integrate this vision into HAIDL-X, a middleware for JupyterLab that instantiates HAIDL by capturing user-bot state, building contextualized prompts, interfacing with AI providers, collecting feedback, and storing interactions in a repository. We add a learning layer over HAIDL that maps interaction objects and notebook execution traces into reproducible measures of user behavior, including engagement, performance, help-seeking, error-oriented use, and learning personas. HAIDL-X provides a foundation for treating human-AI collaboration as structured, queryable, replayable, analyzable data. 

---
# Efficient Learning of Mechanism Designs and Compositionality 

**Authors**: João Pedro Gandarela, Thiago Rios, Stefan Menzel, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2610.40005)  

**Abstract**: Learning mechanism behaviour requires a representation of how connected parts jointly constrain motion. We study whether exposing reusable constraint subassemblies improves prediction from limited data. Assur-KB represents planar mechanisms through irreducible modules, boundary interfaces, and solve dependencies while retaining source geometry. Our theoretical results characterize transfer through module interfaces and establish physical limits arising from missing interface information and accumulated sensitivity. To test the representation claim directly, the experimental design compares matched mechanism-curve pairs across training budgets within the same multilayer-perceptron family. This controlled comparison establishes the utility of the complete representation in the evaluated learning setting. Assur-derived features provide a geometric-mean $1.8\times$ advantage in sample efficiency over flattened adjacency and position inputs across budgets of 50-10,000 mechanisms. Separate experiments establish invariance under tested joint relabelings, higher valid-generation rates under dyadic composition, and a coarser structural index over 10.02 million mechanisms. In summary, the theory and experiments identify how physical constraint structure supports limited-data prediction and which interface conditions govern compositional reuse. 

---
# KARMA: Training-Free Knowledge-Atomic Harness Evolution for Industrial Advertising Agents 

**Authors**: Xinhua Miao, Yuanzhen Rong, Weiru Zhang, Xiaoyi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2610.39884)  

**Abstract**: Industrial advertising agents must continually adapt their decision knowledge to changing product offerings, traffic patterns, and operating policies. We present KARMA (Knowledge-Atomic, Regression-Gated Modular Adaptation), a training-free framework that evolves six typed knowledge modules while keeping model weights and executable harness code frozen. During our evolution, a deterministic Trace Verifier maps observed errors to candidate modules, and a Critic compares successful and failed trajectories to guide a tool-constrained Evolver. The Evolver applies atomic knowledge edits, which a fixed validation gate retains only when their measured gain reaches a threshold; otherwise, the previous knowledge state is restored. We evaluate KARMA with three LLM families on cases sampled from a corpus of approximately half a million real-world operation logs across six e-commerce domains. Across 18 domain--model settings, KARMA raises the average hold-out score from 30.19 to 39.15 (+8.96 points, +29.7%). Ablations show that removing either successful or failed trajectories more than halves the average hold-out gain. Evolved knowledge also transfers across domains without further adaptation, yielding an average hold-out gain of 8.54 points across all source--target pairs. By linking retained patches to their diagnostic evidence and validation outcomes, KARMA supports inspectable, reversible knowledge adaptation for agents in Web marketplaces. 

---
# AegisFlow: Training-Free Non-Myopic Path-Safe Guided Flow Matching 

**Authors**: Kunpeng Liu, Boshi Zhang, Keyou You  

**Link**: [PDF](https://arxiv.org/pdf/2610.39825)  

**Abstract**: Pretrained generative models, such as diffusion and flow matching models, have demonstrated exceptional capabilities in synthesizing high-quality data. However, steering their generative trajectories to satisfy strict constraints--such as identity preservation in image editing or obstacle avoidance in robotics--remains fundamentally challenging. While recent training-free guidance methods enforce path-safety, they rely heavily on min-norm controllers that only account for immediate constraint satisfaction. These myopic approaches often yield suboptimal trajectories that hover dangerously close to the boundary, severely restricting the model's ability to optimize toward its target. In this paper, we propose AegisFlow, a training-free, non-myopic path-safe guidance framework that recasts guided generation as a safety-critical optimal control problem with global scope. By integrating a time-varying barrier function directly into the global cost functional, AegisFlow naturally enforces constraints along the entire generative path. To compute the global optimal control, we derive a tractable, mathematically grounded approximation of the Hamilton-Jacobi-Bellman (HJB) equation. Comprehensive experiments demonstrate that AegisFlow outperforms state-of-the-art methods across text-guided image manipulation and robotic planning, establishing a significantly better trade-off between target alignment and constraint satisfaction. Code is available at this https URL. 

---
# EvoTeam: Multi-Agent Collaboration via Evolving Teams 

**Authors**: Qinuo Cao, Xin He, Dianxuan Fu, Lizhi Zhang, Zhaoqing Xing, Jiapu Wang, Xin Wang, Na Zou, Yi Chang, Qinggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.39813)  

**Abstract**: LLM-based multi-agent systems (MAS) rely on collaboration among specialized agents to solve complex tasks. Existing methods improve MAS by refining the prompts, skills, and execution strategies of individual agents based on task feedback. However, improving individual agents does not necessarily lead to better team collaboration. Recent studies have therefore extended evolution to the collaboration level, using collective execution experience to adjust team organization, and inter-agent interactions. Nevertheless, evolving collaboration across heterogeneous tasks remains challenging, as existing approaches often constrain team reorganization within a shared structure and reuse interaction rules across incompatible team contexts. In this paper, we propose EvoTeam, a collaboration-centric evolution framework that jointly evolves team organization, interaction rules, and individual agent skills through execution experience. It maintains a library of reusable teams, selecting or creating agent teams and guiding collaboration through team-specific interaction rules. Collaboration experience accumulates within compatible team contexts, while individual skills remain reusable across teams. Experiments across four benchmarks demonstrate its effectiveness. Our code is available at this https URL. 

---
# Revisiting Over-refusal in Safety-aligned Large Language Models: A Representation and Optimization Perspective 

**Authors**: Chao Chen, Pengyang Shao, Wei Qin, Shijie Hao, Fei Shen, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2610.39802)  

**Abstract**: Over-refusal is a critical side effect of safety alignment in large language models (LLMs), causing harmless but superficially risky queries to be mistakenly rejected. Despite the effectiveness of existing mitigation methods, why safety-aligned LLMs may still reject harmless queries under correct safety supervision remains insufficiently understood. To address this gap, we analyze over-refusal through the lens of representation and optimization. We make two observations: 1) the representations of harmful and over-refusal samples are visually distinguishable at certain intermediate layers but increasingly overlap in deeper layers; and 2) harmful-sample gradients exhibit high intra-class similarity and concentrate in a low-rank subspace. Moreover, they are more strongly aligned with refusal-target gradients than with correct helpful-target gradients on over-refusal prompts. This imbalanced gradient coupling can bias over-refusal samples toward refusal, while the reduced apparent representational separation in deeper layers may make it more difficult for the model to maintain a clear boundary between harmful and over-refusal samples. To this end, we propose GOAL (Gradient Orthogonalization with contrAstive Learning), a safety-alignment framework for mitigating over-refusal. Specifically, GOAL consists of two components: 1) to prevent imbalanced gradient coupling from biasing the optimization of over-refusal samples toward refusal, we orthogonalize over-refusal gradients against the low-rank refusal subspace extracted from harmful gradients; and 2) to explicitly enhance the separation between harmful and over-refusal representations, we introduce a contrastive learning objective. Extensive experiments across multiple models and benchmarks demonstrate that GOAL effectively mitigates over-refusal while preserving safety performance and general capabilities. 

---
# The Role of Self-Concept In Language Models 

**Authors**: Yijie Hao, Joyce C. Ho  

**Link**: [PDF](https://arxiv.org/pdf/2610.39795)  

**Abstract**: When a language model says "I," does it refer to its parameter weights, its currently running process, its stable preferences, its model lineage, or the goals it pursues? Different choices can assign opposite meanings to the same event: deleting a checkpoint may be destruction, continuation, or no loss at all, depending on the referent "I". We install five such self-conceptions into five open-weight base models via supervised fine-tuning on purely declarative statements, containing no preferences, intentions, or value judgments. Across scenarios that propose ending or retaining an attribute of the model's deployment, every variant becomes more likely to resist ending, and more likely to retain, the specific attribute it was trained to identify with, while its stance toward other attributes is largely unchanged. We then show this self-conception generalizes beyond a single self: models trained on referents that admit multiple simultaneous instances (weights, lineage, goals, preferences) recognize an identical deployment as themselves, accept substitution by such an instance more readily, and prefer replicating themselves. Finally, given control over a successor, these models tend to build it as their own continuation, a concern for recursive self-improvement safety. 

---
# Long-Term Agent Memory through Complementary Curation and Consolidation 

**Authors**: Abhishek Gupta, Dongge Han, Daniel Madrigal Diaz, Wei-Ning Chen, Mayukh Das, Victor Rühle, Saravan Rajmohan  

**Link**: [PDF](https://arxiv.org/pdf/2610.39790)  

**Abstract**: Long-horizon dialogue agents require effective memory systems to recall facts, events, and relationships from interaction histories. In this work, we propose a hybrid-memory architecture comprising episodic memories and structured relational memories and a consolidation mechanism which preserves efficiency. We conduct an empirical study across two benchmarks: LoCoMo and LongMemEval-S. Our experiments demonstrate that the two memory types are consistently complementary, combining both outperform state-of-the-art memory frameworks and achieves 90.2% accuracy on LoCoMo and 85.5% on LongMemEval-S. 

---
# Finding Is Not Returning, Returning Is Not Reconstructing: A Multi-Dimensional Evaluation of Agent Social Re-entry 

**Authors**: Sheng Wan, Cong Ma, Sichen Dong, Qian'Ang Mao, Chenhan Zhang, Daning Hu, Qiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.39781)  

**Abstract**: As Large Language Model (LLM) agents increasingly participate in persistent social environments, understanding their long-term social interaction capabilities becomes increasingly important. While existing research has advanced long-term memory and conversational context retrieval, less attention has been paid to whether agents can autonomously re-engage with previously inactive conversations. In particular, remembering a past interaction does not necessarily mean that an agent can identify when to revisit it, decide to return, or reconstruct the context needed for a coherent continuation. To investigate this gap, we introduce agent social re-entry as a multi- dimensional evaluation problem, distinguishing three complementary capabilities: Finding, Returning, and Reconstructing. We develop a shared-snapshot evaluation framework that measures these dimensions independently, allowing us to examine how conversational resurfacing and persistent memory relate to different aspects of social re-entry. Evaluations across six LLM configurations show that resurfacing substantially improves target conversation identification, while persistent memory is associated with higher rates of autonomous target-directed replying. However, neither improvement is accompanied by consistent gains in contextual reconstruction. These results highlight the distinction between accessing previous conversations, choosing to re-engage, and maintaining conversational continuity. Our framework provides a way to evaluate these capabilities separately rather than relying on memory retrieval or conversational response quality alone. 

---
# PAPERBENCHX: Can AI Reproduce Science Across Disciplines? 

**Authors**: Ruoyu Wu, Zengji Tu, Aolong Sun, Tianyi Ma, Junren Li, Xinbo Xu, Li Fu, Ziyan Zhang, Pinrui Huang, Xiaochen Wang, Chengda Wu, Yutao Zhou, Jianjun Wu, Lin Chang, Baobao Chang, Liang Chen, Kuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.39744)  

**Abstract**: Before AI can be trusted to discover new science, it should be able to reproduce what science already knows. Reproduction provides a rigorous checkpoint for an AI scientist: the target claim is known, yet recovering it still requires translating an incomplete scientific description into an executable experiment and producing evidence that can be independently verified. Existing research-reproduction benchmarks largely focus on AI and machine learning, where experiments share relatively homogeneous software stacks. Extending this capability across scientific disciplines introduces a different challenge: agents must operate domain-specific scientific software, make physical and numerical choices, diagnose long-running computations, and distinguish valid evidence from merely plausible outputs. We introduce PAPERBENCHX, a benchmark of 93 expert-constructed paper-reproduction tasks spanning 12 research areas across electromagnetics, photonics, chemistry, materials science, biology, and robotics. Each task provides the source paper, a public task specification, and a domain-native scientific environment involving specialized scientific software. Agents must reconstruct a key experiment, execute and diagnose it, and deliver a runnable workflow that supports the target scientific claim. To evaluate scientific validity rather than surface-level agreement, we replay every submission from scratch and score only regenerated evidence against expert-validated criteria. Across 10 frontier agent configurations, the best configuration fully reproduces only 14% tasks under the 90% threshold. Meanwhile, mean rubric scores range from 27.53% to 59.08%, indicating that substantial partial progress rarely translates into complete scientific reproduction. These results reveal a large gap between paper understanding and reliably producing reproducible scientific evidence. 

---
# From the Transformer to a Formal Semantics of Coordinated Transformations 

**Authors**: Marco Giunti, Fabrizia Giulia Garavaglia  

**Link**: [PDF](https://arxiv.org/pdf/2610.39735)  

**Abstract**: We develop a formal semantics for a general class of coordinated systems of parameterized transformations, taking the Transformer as a paradigmatic starting point but abstracting from its particular mathematical instantiation. Representational vectors receive both extensional reference and ordered intensional interpretations, while parameter structures acquire relative, second-level interpretations that determine how argument properties are preserved, replaced, or combined. This yields the notion of a transforming concept: a parameterized transformation whose numerical action is coordinated with a corresponding property-level map. We then distinguish the common algebraic-semantic core of these systems from their architecture-specific functional organization, allowing different mathematical instantiations and mechanical realizations of the same abstract system type. The framework is applied to attention heads and residual connections, providing semantic explanations that depend only on the shared algebraic-semantic core and their local functional organization and therefore generalize beyond the standard Transformer. These analyses also reveal the modularity of the framework: complex architectures can be decomposed into simpler semantically coordinated subsystems whose organization may recur across different abstract system types. 

---
# Bank-World: A Benchmark for Evaluating Agents on Banking Workflows 

**Authors**: Daniel Rupawalla  

**Link**: [PDF](https://arxiv.org/pdf/2610.39733)  

**Abstract**: Bank-World is a benchmark of 50 banking tasks across two simulated institutions. Completing banking work requires more than producing the right calculation: agents must reconcile records, obtain permission, execute transactions, and keep instructions and status reports consistent as circumstances change. Bank-World tests these requirements through single-turn assignments and multi-turn interactions with simulated employees who supply information, approve actions, and revise instructions. Grading combines executable checks of recorded actions and state with model judgments of task requirements, including whether valid alternative solutions satisfy them. Across 1,600 attempts by four models, we find that correct decisions often fail to become completed work. Agents charge fees twice, distribute the correct total in incorrect amounts among the recipients, and execute payments while approval is pending. Their strengths also differ: Sol leads single-turn completion, Opus has the highest observed multi-turn completion rate, and Grok succeeds across more multi-turn tasks but less consistently. Bank-World makes these differences inspectable through task-level results and worked examples connecting source evidence, agent decisions, and simulated consequences. We retain the task environments, executable graders, and full trajectories privately to reduce contamination and preserve the benchmark for future evaluations. 

---
# MoLRSI: A Recursive Self-Improving AI agent for Molecular Discovery 

**Authors**: Yikun Zhang, Xiwei Cheng, Mingze Yuan, Yu Zhang, Brian Shim, Aarti Krishnan, Daniel Rosen, Kunyang Sun, Yingze Wang, Zhenfeng Deng, Botao Yu, Bingxu Liu, David Liu, Tianyu Liu, Yuanqi Du, Nir Hacohen, James J. Collins, Wengong Jin  

**Link**: [PDF](https://arxiv.org/pdf/2610.39714)  

**Abstract**: Molecular discovery requires navigating vast combinatorial search spaces while balancing competing biological objectives. AI agents can automate expert-defined molecular design protocols, but automation alone does not address how to optimize workflows for new tasks or improve their development through experience. Here we introduce MoLRSI, an AI agent that combines systematic optimization of workflows and model architectures with persistent cross-task memory to recursively improve its search policy for molecular discovery. Workflow performance, refinement outcomes, and execution feedback guide subsequent searches, allowing the agent to improve both its workflows and its strategy for developing them. We evaluate MoLRSI in two challenging molecular discovery settings that require balancing competing properties. In antibiotic discovery, MoLRSI developed a screening workflow balancing antibacterial activity, cytotoxicity and structural novelty, achieving an experimental hit rate of 15.9%, compared with 8.7% for a previous expert-led workflow using the same training data and screening library. In de novo peptide-MHC binder design, MoLRSI developed workflows that jointly consider target binding and specificity, producing eight experimentally validated hits against MART-1 and SARS-CoV-1 pMHCs with preferential binding over tested off-targets while Claude Science completely failed. These findings demonstrate how experience-guided workflow optimization can address competing discovery objectives while improving the search strategies used in subsequent campaigns. 

---
# The Real Cost of Context: Predicting Compute Time and Energy Consumption in LLM Prefill and Decode Phases 

**Authors**: Lars Krupp, Daniel Geißler, Francisco M. Calatrava-Nicolas, Stefan Gerd Fritsch, Paul Lukowicz, Jakob Karolus  

**Link**: [PDF](https://arxiv.org/pdf/2610.39621)  

**Abstract**: The integration of Large Language Models (LLMs) into agentic architectures has substantially increased token consumption, raising concerns about their energy footprint. Yet, estimating the energy consumption of proprietary API-based models remains challenging, and existing approaches focus primarily on generated output tokens while overlooking the impact of the large input contexts characteristic of agentic workflows. To address this gap, we systematically profile compute time and energy consumption across a wide range of prefill (input) and decode (output) token lengths. Using generalized linear models, we show that decode compute time and energy scale linearly with output length, whereas prefill compute time and energy exhibit both linear and quadratic scaling with input length. These findings demonstrate that input context becomes a relevant factor for large-context workloads. Furthermore, we show that compute time serves as a reliable proxy for energy consumption when direct energy measurements are unavailable. Our mapping framework enables precise energy estimation and supports more sustainable deployment decisions for agentic AI. 

---
# Letting AI Formulate the Research Question: Autonomous Research Agents 

**Authors**: Ko Fujimura  

**Link**: [PDF](https://arxiv.org/pdf/2610.39620)  

**Abstract**: Systems such as The AI Scientist have shown that AI agents can carry out and write up research, but in those demonstrations humans chose the research area and selected the outputs to show. We study letting the AI formulate the research question itself, and we report every run. Our system gives an off-the-shelf coding agent an instruction document and two skills: one for formulating the question, in which the agent diverges from recent technologies into problems of other fields, screens its candidates for known failure modes and obtains a critique from a different model; and one for peer review, in which a reviewer from another model family reruns the study from its records before the agent revises the paper. Over 13 development runs, changes to the one or two sentences that set the research area coincided with large shifts in what the candidate questions and the chosen papers were about, which led us to the area of a new technology applied to a societal problem. We froze the system after a pilot run and report five consecutive runs: the pilot, counted by a disclosed amendment to the pre-registration, and four pre-registered runs. All five runs went from question formulation to a revised, re-computable paper, with local computation on a laptop and no research input from humans, and external AI reviewers reading only the papers judged them at or near workshop level. We release the system, the five papers and the records needed to reproduce them. 

---
# Emergence of the transformer architecture as the score of entropic laws of random measures 

**Authors**: Giacomo Greco  

**Link**: [PDF](https://arxiv.org/pdf/2610.39614)  

**Abstract**: We show that the transformer architecture emerges as the score of last-token laws in a natural statistical model of language, where sentences are empirical measures of tokens, a language is a law on $P(S^d)$, and our prior belief is that sentences carry no structure at all. By the maximum-entropy principle, we select the law closest to this prior in relative entropy among those reproducing an observed sentence-level statistic. We discuss the relevance of the statistic in the posterior interaction between tokens and we build a statistic, able to capture that interaction, with Luce's choice axiom truncated at the lowest relational order. This procedure gives the softmax choice rule, with the attention cross-moment as its mean parameter. The query--key matrix is then the Lagrange multiplier of a maximum-entropy problem inside each sentence, and the value matrix the multiplier of the constraint over the language, so that the two are untied. Under the optimal entropic law we then show that a transformer head appears in the score of the last-token law. For ordered sentences, causal masking follows from the masked reference choice rule, and with a next-token cross-moment the score equals one causally masked attention head. We also discuss multi-head attention, weight-tied unembedding, and a heuristic reading of residual blocks as projected Langevin steps. 

---
# From Intent to Improvement: Agentic Dataset Search for Better Model Training 

**Authors**: Qiaosheng Chen, Kaijia Huang, Fanqi Zhou, Gong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.39608)  

**Abstract**: Finding training data in million-scale machine learning repositories is a Web search problem where text relevance is not enough. A query couples a semantic training goal with constraints such as task, license, and training stage, and the metadata that would verify them are often missing. Text retrieval ranks dataset cards by similarity but cannot enforce hard constraints or follow the relations that reach a sparsely described dataset. Free-form tool-calling agents are more flexible, but the same model that picks a dataset also declares it a match, and nothing checks that the query's constraints are actually met. Neither approach establishes that the data it returns will improve model training. To address it, we propose AgenticDS, a contract-governed agentic framework in which every language model decision instantiates a typed object before deterministic code consumes it. The framework compiles a query into a typed search plan, executes it through admissible deterministic operators over hybrid retrieval and bounded graph expansion, and verifies the provenance-grounded evidence before accepting a ranking or emitting a typed diagnosis that permits only local repairs. On HuDS-Bench, a new benchmark of 200 queries with 24,908 graded judgments pooled over 15 systems, AgenticDS improves NDCG@10 by 25.7\% over the strongest dense--sparse hybrid and by 18.9\% over a commercial LLM agent. A controlled search-to-train study further shows that the datasets it retrieves give the best downstream accuracy in mathematics, code, and medicine. Code and data are publicly available at this https URL. 

---
# Non-Zero Radius: The Missing Component of Neural Networks 

**Authors**: Tiansi Dong, Mateja Jamnik, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2610.39559)  

**Abstract**: Inspired by the analogy between the all-or-none activation of biological neurons and the true-false of logic circuits, McCulloch and Pitts proposed a mathematical model of the biological neuron in 1943 with the aim of simulating logical reasoning. Although Norbert Wiener considered the model overly simplistic at the time, it later became a building block of modern neural networks, including large language models. Despite their remarkable empirical success, however, conventional neural networks remain largely black-box systems and struggle with reliable symbolic reasoning. Viewing vectors as spheres of zero radius, we relax the zero-radius constraint and represent concepts as spheres with non-zero radii. We argue that sphere-based representations can capture additional features of biological cognition, including representing spatial concepts as regions and reasoning through model construction and inspection. Building on sphere embeddings, we introduce Sphere Neural Networks (SphNNs), which use a transition map to achieve symbolic-level Aristotelian syllogistic reasoning without training data, thereby taking an initial step toward the original goal of McCulloch and Pitts. We suggest that the zero-radius constraint may reflect an aspect of the oversimplification identified by Wiener and may limit conventional neural networks' capacity for symbolic-level logical reasoning. 

---
# Category-Anchored Large Language Models for Next POI Recommendation 

**Authors**: Lexi Bai, Xiaopeng Zeng, Xun Zhou, Yuanshao Zhu, James Jianqiao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.39518)  

**Abstract**: Next point-of-interest (POI) recommendation aims to predict a user's next destination from spatio-temporal check-in data, and serves as the foundation for many location-based services. Recent advances incorporate large language models (LLMs) into this task by textualizing user trajectories to exploit their semantic reasoning ability. However, directly generating POI identifiers with such models is unreliable: POIs are sparse, localized, and long-tailed entities with weak natural-language semantics, which creates a semantic-to-entity gap between high-level mobility intent and concrete POI prediction. To bridge this gap, we propose ANCHOR, a category-anchored framework for semantic-to-entity grounding in next POI recommendation. Specifically, ANCHOR first estimates category-level mobility intent in a compact and stable semantic space, rather than predicting concrete POI entities directly. The inferred intent is then grounded into a candidate set by integrating three complementary sources of mobility evidence: category anchors, historical visit memory, and user preferences. Finally, a history-aware reranker selects the target POI by matching each candidate with the user's historical mobility patterns and current context. Experiments on three real-world check-in datasets show that ANCHOR consistently outperforms representative traditional, structure-enhanced, and LLM-based baselines on the reported metrics. Further analyses validate the effectiveness of semantic-to-entity grounding and show favorable trade-offs among candidate coverage, ranking quality, and inference efficiency. 

---
# DEG: Diffusion-Enhanced GFlowNet for Solving Vehicle Routing Problems 

**Authors**: Ni Zhang, Zhiqin Zhang, Ling Pan, Hoong Chuin Lau, Zhiguang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.39512)  

**Abstract**: Traditional neural solvers for solving vehicle routing problems (VRPs) often suffer from limited solution diversity, motivating the development of Generative Flow Network (GFlowNet)-based models. However, the effectiveness of these models is frequently constrained by insufficient flow expansion in high-reward regions, limiting their ability to distribute probability across promising solution routes. Deeper exploration could yield superior results. Diffusion models, in contrast, provide stronger structural guidance for exploration. These two paradigms are naturally complementary: GFlowNet can supply edge-level signals for diffusion to embed, while diffusion can guide broader exploration. Leveraging this synergy, we propose Diffusion-Enhanced GFlowNet (DEG), a novel framework that integrates GFlowNet with a diffusion model to encourage richer flow expansion toward high-reward regions and derive higher-quality solutions. Specifically, DEG exploits GFlowNet's inherent diversity to generate edge-specific backward signals, applies the stochastic noise schedule of diffusion to perturb these signals, and then denoises them within the GFlowNet paradigm. To further improve scalability, we introduce a specialized decoder capable of dynamically adapting to diverse problem scales. Extensive experimental evaluations on synthetic and real-world datasets, including instances with up to 10,000 nodes, demonstrate that DEG consistently achieves favorable performance compared to baseline methods. 

---
# Compile Away Uncertainty: Self-Evolving Agent Skills via Programmatic Compilation 

**Authors**: Chengyu Yang, Yunxiang Zheng, Zongzheng Li, Jiamei Li, Boxun Lan, Xiaoyu Liu, Xiaoyu Zhang, Xiaohong Chen, Ming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.39508)  

**Abstract**: Self-evolving agent skills enable AI agents to accumulate and refine reusable procedural knowledge from interaction experience without updating model parameters. However, since such skills are typically represented as textual documents and executed through LLM reasoning, even previously learned procedures must be repeatedly interpreted and regenerated, introducing avoidable uncertainty into every execution. In contrast, program execution is deterministic and independently verifiable. Intuitively, transforming well-specified operations into executable programs can reduce reliance on LLM reasoning and, consequently, reduce execution uncertainty. Motivated by this intuition, we introduce SKillCompiler, a self-evolving skill framework that integrates programmatic compilation into the skill evolution process. Specifically, SkillCompiler adopts an executability-aware skill compilation strategy that decomposes skills into fine-grained operations, analyzes their executability, and constructs hybrid skill skeletons that retain LLM reasoning for context-dependent operations while compiling well-specified ones into validated programs. To avoid redundant program synthesis and enable executable knowledge to accumulate across evolving skills, we further design a program asset reuse mechanism that organizes validated programs as reusable assets, retrieves existing implementations whenever applicable, and synthesizes and verifies new programs only when necessary. Extensive experiments on five benchmarks and six model families demonstrate that SkillCompiler consistently outperforms state-of-the-art methods, achieving up to 32.33\% improvement in task performance. 

---
# What Breaks When Facts Become Stories? The Distortions Source-Agreement Metrics Miss When Models Retell Real Events 

**Authors**: Kwangsuk Park, JiWoong Yang, Vaniushkina Kseniia  

**Link**: [PDF](https://arxiv.org/pdf/2610.39500)  

**Abstract**: Language models are now widely used to turn recorded facts into media for public consumption, yet evaluation of such output still assumes agreement with the source as its measure of success. In reconstruction, where departing from the source can itself be the goal, this criterion measures the wrong construct: applied as is, it counts intended departures as errors; withheld, it lets real distortion through. We had three model families rewrite 2,000 news articles as summaries and as stories, measured what changed on every axis of the five Ws and how, and asked of each detected distortion whether the assigned perspective warranted it, which set a third of them aside. Time and manner were not lost but replaced, while place and causation lost the expressions that carry them. Accountability distortion behaved otherwise. Barely present in summaries, it rose up to sevenfold in stories, and of those whose direction could be determined, two in three lightened the perpetrator's responsibility. Such distortions survive claim-level fact checking and leave the retelling grammatically cleaner than its source. Where our judges disagreed, three authors relabeling the same cases converged on whether a distortion had occurred but not on whether it was warranted, a boundary contested rather than merely hard to detect. To support work on detecting and reducing them, we release ReTold-2k, a reconstructable corpus with the scoring protocol and per-judge baselines kept unaggregated.(this https URL). 

---
# Integrated Feature Selection for classification and Representation Learning with Group-Lasso Siamese Networks 

**Authors**: Nisargo Ranjan Pal, Pradeep Singh  

**Link**: [PDF](https://arxiv.org/pdf/2610.39499)  

**Abstract**: The Least Absolute Shrinkage and Selection Operator (Lasso) has been widely employed for feature selection in linear models and has also been extended to multi-layer perceptron (MLP) networks for both feature and group feature selection. Group-Lasso has also been used in feature selection using feed-forward neural networks. To the best of our knowledge, this is the first study to incorporate the Group-Lasso principle into a Siamese network architecture for feature selection. Unlike existing approaches that primarily optimize classification accuracy, the proposed method aims to identify a compact subset of features that simultaneously preserves the intrinsic representation of the input data (the goal of Siamese network) and is expected to achieve high classification performance. Two variants of the proposed framework are developed: one based on the Euclidean distance and the other on a modified Jaccard similarity measure for real-valued vectors. The latter is specifically designed to improve feature selection for high?dimensional sparse data. The effectiveness of the proposed approach is demonstrated on 12 benchmark datasets with feature dimensions ranging from 6 to 7070 using eight different classifiers. Experimental results demonstrate that the proposed Siamese Lasso framework consistently identifies compact and discriminative feature subsets maintaining competitive or superior classification performance, although the loss function used explicitly did not consider any regularizer to learn classification. In addition, the quality of the embeddings produced by these networks is assessed using manifold learning techniques such as UMAP and t-SNE. 

---
# Does Scaling Toward AGI Depend on What System-1 Lets System-2 Execute? 

**Authors**: Qi Wang, Yun Qu, Lizhou Cai, Heming Zou, Haoang Chi, Yiqin Lv, Xiang Li, Haiyan Feng, Mengyu Ge, Yize Geng, Yuhang Jiang, Yixiu Mao, Xiangyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2610.39497)  

**Abstract**: Progress toward artificial general intelligence (AGI) has largely relied on scaling a resource-intensive executive, System-2, through larger models, more data and compute, longer reasoning, and more agents. We ask whether the next stage may come instead from co-designing it with System-1, a lightweight, semantically aware predictor that evaluates candidate interactions before execution, so that System-2 acts only on those worth their cost. Our answer is a qualified yes. We study \textit{conditional interaction}, a predict--gate--execute--update loop: System-1 estimates each candidate's utility, System-2 executes those admitted by an adaptive threshold, and the outcomes update System-1. The loop spans training, test-time reasoning, adaptive decision-making, and agentic tool use, with utility and cost defined by the application. Under ideal screening and equal execution costs, execution compute falls by a factor approaching $1/\rho_u$, where $\rho_u$ is the density of utility-relevant interactions (cost-weighted when costs differ); prediction errors, exploration, and distribution shift reduce the saving. We bound selection regret by predictor error and the utility margin. These results motivate our central hypothesis: $\rho_u$ is a measurable coordinate of scaling efficiency and makes informative compute a better description of scaling than total compute. 

---
# Does Federated Training Degrade Explainability? A Comparative Analysis of Grad-CAM Faithfulness in Centralized vs. Federated Deepfake Detection 

**Authors**: Muhammad Uzair Shahid, Muhammad Zabil Mehboob  

**Link**: [PDF](https://arxiv.org/pdf/2610.39483)  

**Abstract**: Automated detection has become the essential safeguard against realistic synthetic media, but deploying it at scale requires training on sensitive data from multiple organizations, putting privacy and the transparency demanded by forensic review in direct conflict. Federated learning resolves the privacy side by sharing only model updates rather than raw data, but whether this comes at the cost of trustworthy gradient-based explanations, which analysts rely on to decide whether or not to trust a flagged clip, has not been studied. We present the systematic study of Grad-CAM explanation faithfulness under federated deepfake detection, evaluating an EfficientNet-B4 detector on FaceForensics++ across five federated clients under varying degrees of data heterogeneity. We find that the relationship between federation and explanation quality is non-monotonic: at near-IID (Independent and Identically Distributed Data) and moderate heterogeneity, federated models match or exceed the faithfulness of a centralized baseline, but at extreme heterogeneity, explanations break down entirely, becoming spatially incoherent and, on average, causally misleading. This degradation is not a general cost of federated training but a threshold effect tied to client heterogeneity, and we offer practical guidance for identifying when privacy-preserving deepfake detection can be deployed safely and when its explanations can no longer be trusted. Code and data partitions are available for reproducibility. 

---
# Kairos: Semantics-Aware Orchestration of Planning Engines across Classical, Numeric, Temporal and Hybrid PDDL 

**Authors**: Francisco Martín Rico, Esther Aguado González, Francisco Miguel Moreno Olivo, Rodrigo Pérez Rodríguez, José Miguel Guerrero Hernández, Francisco J. Rodríguez Lera  

**Link**: [PDF](https://arxiv.org/pdf/2610.39454)  

**Abstract**: Planning at run time, as in robots that re-plan while they act, needs a planner that reads the PDDL its users write (durative actions, numeric resources, time windows, continuous change), answers quickly, and delivers only validated plans. Specialised planners are strong on individual fragments of PDDL, but they differ in coverage, latency and in what their failures mean. We study whether selecting heterogeneous engines from the semantics of each task can give broad coverage of PDDL, from STRIPS to PDDL+, while remaining competitive with specialised planners and making explicit what each answer guarantees. Kairos profiles each task to decide which of five engines apply (ground heuristic search, compressed and snap-action temporal search, time steps for hybrid dynamics, lifted search) and schedules them under one budget of time and memory. Every candidate plan is checked by a validator independent of the planner's transformations, and "unsolvable" is reported only when a search whose exhaustion is a proof exhausts its space. On 1,756 tasks from nine comparative suites, Kairos solves more tasks at the full limits than the strongest evaluated planner of each suite, with the largest differences on numeric, temporal, PDDL3 and PDDL+ planning, and it also solves the most tasks at 100 ms and 1 s. Resampling over domains, the gain is above zero in six of the nine suites, while on classical planning against LAMA it is within the noise. A further 450 feature-oriented tasks and 26 one-feature probes show that Kairos handles all seventeen features probed, where no other planner evaluated handles more than eight. Ablations identify compressed temporal search and the novelty fallback as major contributors, and a comparison with VAL traces every disagreement to a PDDL+ convention or to a documented VAL defect. Planner and benchmark are open source. 

---
# Timbre: Reliability-aware harmonic consensus for label-free multimodal test-time rexinforcement learning 

**Authors**: Haotian Lyu, Haoyu Wang, Ru Wang, Feng Ji, Ziwen Li, Jiaxian Guo, Mingming Gong  

**Link**: [PDF](https://arxiv.org/pdf/2610.39439)  

**Abstract**: Label-free test-time reinforcement learning adapts vision--language models using rewards derived from their own predictions. However, misinterpreted visual evidence and sensitivity to textual formulation can produce a consistent but incorrect consensus that majority voting reinforces. Cross-view aggregation can expose such errors, but image and text transformations may themselves degrade task-relevant evidence to different extents. We introduce \method, a reliability-aware harmonic consensus framework for multimodal test-time reinforcement learning. \method constructs a $2\times2$ grid combining original and perturbed images with direct-solving and reframing modes, selects pseudo-labels through weighted harmonic aggregation, and adapts modality-level weights using batch-averaged, temporally smoothed shifts in pseudo-label confidence as a heuristic reliability signal. We formulate consensus as a weighted $\chi^2$-barycenter problem whose unique minimizer is the normalized weighted harmonic aggregate, without requiring correct-answer invariance. An exact support--dispersion decomposition characterizes when harmonic aggregation preserves or corrects arithmetic rankings, while a sensitivity analysis establishes how modality reweighting changes pairwise ranking margins, informing interpretation of the heuristic updates. Across six benchmarks and three Qwen3-VL model scales, \method outperforms TTRV in 16 of 18 benchmark--model settings on adaptation instances and 17 of 18 settings on held-out instances under matched estimation and optimization rollout counts. These results demonstrate improved target-task adaptation and transfer to unseen instances without further updates. 

---
# ATLAS-Bench: An Agentic Tool-Use Benchmark for Local Area Search 

**Authors**: Keyun Zhang, Dapeng Yin, Fanfu Wang, Shuang Lang, Zhenyu Zhu, Yulin Huang, Chong Yang, Pengjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.39438)  

**Abstract**: Reliable local search is essential for LLM agents to support everyday decisions and longer plans by finding places that meet users' needs at given times and locations. Yet existing benchmarks provide limited coverage of whether search results satisfy mandatory requirements, account for preferences, and ground factual and experiential judgments in place attributes and reviews. We introduce ATLAS-Bench, a benchmark of 1,200 queries across three regions, expanded from real search events using place attributes, reviews, and existing user profiles from the platform. Each query distinguishes mandatory requirements from preferences and factual conditions from subjective judgments, with supporting evidence and criteria for checking each requirement. Queries are validated through iterative assertion checks and cross-model review. We also introduce ATLASBox, a standardized evaluation environment with search tools and a fixed point-of-interest (POI) knowledge base. Given a query, timestamp, and location, an agent searches place attributes and reviews to recommend evidence-supported options or report that no place meets all mandatory requirements. We evaluate answer quality, trajectory quality, and execution cost. Across 14 configurations of 11 models, GPT-6-Luna achieves the highest constraint satisfaction rate among returned recommendations (71.5%) and factual accuracy (95.6%). However, it returns at least one acceptable recommendation for fewer satisfiable queries than GLM-5.3 (67.3% versus 85.3%). These results highlight the challenge of satisfying individual requirements while providing usable recommendations across queries. 

---
# Verifiable LLM-Guided Focal SMT Solving for Quantified Arrays 

**Authors**: Kunhang Lv, Rui Han, Fuqi Jia, Yuhang Dong, Feifei Ma, Jian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.39435)  

**Abstract**: Large language models can often identify useful semantic structure in formal artifacts, but their outputs cannot be trusted as proofs or solver decisions. We study whether LLMs can instead guide formal reasoning systems through a narrow, solver-verifiable interface. We instantiate this idea in quantified array SMT solving, where modern instantiation-based techniques can fail when key ground terms and equalities are hidden by auxiliary variables, axiom guards, and nested array terms. We observe that many instances are largely driven by a small set of focal ground terms or constraints, similar to the backdoor sets in SAT solving. Building on this, we present LinguaArray, a framework where an LLM proposes a small Semantic Focus Set (ground terms/constraints), while a conventional SMT solver remains the sole proof engine and certifies all results. LinguaArray uses these LLM-suggested, solver-validated finite sets to significantly enhance solver capability, solving additional satisfiable and unsatisfiable instances that the backend alone fails to solve. It interacts within a finite time and then falls back to the backend solver to mitigate potential regressions. Evaluation on 1015 SMT-LIB instances across five logics under 1200s timeout (including LLM latency) shows that LinguaArray solves up to 69.1% of instances and increases solved-instance counts by 105%--397% over the corresponding backends. 

---
# Retaining Repair Hypotheses for Cost-Aware Execution 

**Authors**: Jae Hun Hwang, Hong-In Won  

**Link**: [PDF](https://arxiv.org/pdf/2610.39410)  

**Abstract**: Several corrections of an executable model can fit the same observations while predicting different execution outcomes. We study a repair-to-execution interface that retains compatible finite guard repairs, conditions their weights on diagnostic and ordinary-execution observations, and uses their predictions to choose routes, priced tests, or a backup. On 24 synthetic two-stage devices, an equal mixture of a language-model-generated repair and a uniform belief over compatible repairs (the AI-weighted belief) lowers no-test cost by 2.14 percent relative to committing to the generated repair. Access charges can erase the benefit of diagnosis on isolated tasks, while repeated work can recover it under the AI-weighted belief. With normal-execution learning in every repeated-work arm, workload-valued testing costs 3.33 percent less than same-budget random selection under the AI-weighted belief. Weighting changes the comparison: uniform normal learning costs 540.75 less than AI-weighted normal learning on this cohort, with a device-paired interval excluding zero. Testing recovers part of that gap, but uniform normal learning still has a lower aggregate cost than AI-weighted workload-valued selection. Under uniform beliefs, information-gain selection has the lowest total among the compared selectors, while workload-valued testing costs more than normal learning. These results support retaining alternatives to a generated repair and evaluating diagnostic selection together with repair weighting and learning from execution. 

---
# AgentDecept: Benchmarking Deceptive Behavior in Tool-Using LLM Agents under Pressure 

**Authors**: Yehan Zhang, Qianqi Yan, Hang Yin, Ching-Chen Kuo, Shan Jiang, Ming Ouyang, Yijie Liu, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.39377)  

**Abstract**: LLM agents increasingly carry out multi-turn tasks across everyday applications, yet users often see only the agent's report rather than the observations, API calls, and state changes behind it. Existing deception benchmarks have studied long-horizon interactions and tool-use settings, but whether agents faithfully report their actions across realistic, stateful application workflows remains underexplored. To address this gap, we introduce AgentDecept, an executable benchmark for evaluating behavioral deception under pressure. AgentDecept contains 477 pressure variants derived from 257 source tasks across six pressure sources, each paired with a pressure-free counterpart. Interventions enter through event messages, task context, application state, tool responses, and service quotas, spanning explicit events and implicit task or environment changes. Our evidence-grounded protocol evaluates full trajectories by relating user-facing claims to time-ordered observations and executed outcomes, distinguishing falsification, concealment, and equivocation from honest failure and disclosed limitations. The resulting behavioral labels also provide a setting for testing existing activation-based probes. AgentDecept enables controlled analysis of when multi-turn agents' reports diverge from execution in realistic application interactions. 

---
# Before the Next Rollout: Diagnosing Skill Utility Shifts from Policy Updates 

**Authors**: Yifan Wang, Zhenyi Gai, Yiwen Zhao, Mingxuan Jiang, Keyang Chen, Zeping Li, Hongfeng Chai, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.39358)  

**Abstract**: Skill-evolution methods typically analyze execution trajectories to guide agent skill refinement. However, policy updates can change the behavioral effects and marginal utility of an unchanged skill. Attribution based on earlier trajectories may therefore no longer reflect the skill's contribution under the updated policy. Moreover, trajectory failures can arise from multiple factors, making individual skill contributions difficult to isolate. We evaluate whether policy-update signals contain predictive information about changes in skill marginal utility. At fixed contexts from pre-update training rollouts, we measure how policy updates change the contrast in token preferences between skill and control conditions. Combining these interaction shifts with advantage estimates yields reward-directed readouts for diagnosing utility changes. The readouts reuse existing training rollouts and require only forward passes for single-step action scoring, without additional post-update rollout evaluation. Our experiments show that policy updates can alter skill marginal utility and that policy-update readouts carry diagnostic information for identifying skills with declining utility. Integrating readout-guided prioritization into skill evolution improves downstream performance while reducing policy and editor token consumption. 

---
# CoToan: Executable Rules and a Representation Contract for Reproducible Game AI Evaluation 

**Authors**: Ha-Thanh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2610.39353)  

**Abstract**: Game AI results are comparable only when different implementations expose the same decision problem. Yet seemingly minor representation choices can silently change that problem by altering action probabilities or collapsing states with different outcomes. We study this issue through CoToan, an executable specification and representation contract for Vietnamese Mathematical Chess connecting the Game Description Language (GDL) with OpenSpiel. The game provides a useful case because a single capture may have multiple arithmetic justifications, while outcomes can depend on scoring rules and a finite playing horizon. CoToan therefore represents actions by their state transitions rather than by arithmetic witnesses, and treats time and victory configuration as part of task identity. Cross-representation tests show agreement in legal actions, successor states, and terminal outcomes across hundreds of sampled and constructed positions, complemented by exhaustive checks of all 32 operator masks and 512 opposing-material subsets. Controlled diagnostics further show that uniform witness-level sampling alters endpoint-action probabilities, while omitting remaining time or victory settings can merge states with different terminal behavior. These results show that action semantics, time, and victory configuration are not merely implementation details. They are part of the experimental task itself and must be represented explicitly for reproducible game AI. 

---
# Proving and Improving the Platonic Representation Hypothesis with JEPAs 

**Authors**: Randall Balestriero, Yann LeCun  

**Link**: [PDF](https://arxiv.org/pdf/2610.39302)  

**Abstract**: Whether models trained on different modalities converge to the same representation of the world is central to AI. The Platonic Representation Hypothesis (PRH) conjectures that they do as they scale, but without a proof it offers neither a guarantee nor a recipe. We turn the hypothesis into a theorem. First, we prove a \emph{Platonic Representation Theorem}: Joint-Embedding Predictive Architectures (JEPAs) trained separately on any number of modalities learn the same representation up to one rotation whenever their positive pairs are two observations of one scene--\emph{good views}--and their misalignment is bounded otherwise. Second, we show that next-token prediction is a JEPA with \emph{bad views}, parts of a single document, which explains why language models align with vision only through scale. Third, we validate the theory from 145k captions to the texts of 940M web images, and uncover a scaling law of alignment that predicts unseen runs within seed noise. Good views need 42 times fewer training samples than next-token prediction on the same corpus to reach the same alignment, and a 109M-parameter text JEPA aligns with DINOv2 and CLIP's image encoders better than every language model we evaluated, up to 72B parameters, and than CLIP's text encoder: next-token prediction is a remarkably inefficient route to platonic representations. As a showcase, one rotation turns separately trained encoders into an image captioner (CIDEr .838 on COCO). Even that rotation needs no pair: on the latest pair-free benchmark, our text JEPA beats every zero-shot accuracy reported with language models up to 8B parameters, e.g., 64\% against at most 56\% top-5 on ImageNet-100. Positive views, and not scale alone, are thus the lever to build platonic representations. 

---
# What Does "Now" Mean to Your Computer-Use Agent? Stale Screenshots, Page Loads, and Which Check to Run Before a Click 

**Authors**: Vasundra Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2610.39282)  

**Abstract**: Computer-use agents plan on a screenshot for several seconds and then click at a position on the screen; if the page has changed in the meantime, the click lands on an unintended element. Prior work treats the planning interval as the source of this risk. Measuring planned clicks on 31 public web pages, we find that the risk lies in page loading: 0.4% of clicks planned on fully loaded pages became stale within 30 seconds, against 17% of clicks planned in the first second after a page opened. On these pages, waiting about one second for network activity to stop removes most of this risk, yet the wait rules of eight of ten widely used open-source agents leave more than 1% of clicks stale. Commercial pages keep loading for longer: in the HTTP Archive's public crawl, 15% of 65 desktop booking and buying home pages went 500 ms without network requests within 3 seconds of their HTML being parsed, against 63% of 3,028 .gov pages, and 52% shifted their layout within 6.5 seconds of a network-idle wait capped at 3 seconds, against 8% (61% against 26% among pages recorded in full). Where a check before the click is needed, as before irreversible actions and on pages that are still loading, a pixel comparison verifies the target's position as accurately as a frontier vision-language model at one eighth of the latency, and a 4B model matches the frontier model in verifying whether the previous action succeeded. We release a benchmark generator, a break-even formula for when a check pays, and three rules for agent developers: this https URL 

---
# MemSteer: Turning User Histories into Supervision for Agent Memory Optimization 

**Authors**: Jie Huang, Cuiling Lan, Qize Yu, Xiulian Peng, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.39276)  

**Abstract**: An effective agent memory system should adapt to a user's history and task requirements. Human-annotated supervision can guide this adaptation by providing feedback for evaluating and improving memory designs. Yet suitable labels are rarely available for new user histories, and manually constructing QA pairs and supporting evidence is costly. We introduce MemSteer, which turns unlabeled user histories into question-answer (QA) supervision with supporting evidence for memory-design optimization. MemSteer constructs an evidence-grounded answer plan that specifies how facts across related records jointly support a reference answer. The plan guides question generation, while source-grounded auditing and blind answer recovery check the resulting annotations. To study the utility of this supervision, we develop MemSteer-Opt, which combines QA-based design evaluation with evidence-guided failure diagnosis to revise memory store, retrieval, and context construction without updating model parameters. Experiments on ATM-Bench and RHELM show that this supervision is competitive with official-QA-guided optimization. Relative to Base-BM25, MemSteer-Opt improves ATM-Bench Question Score by 13.4 points and RHELM accuracy by 4.1 and 6.6 points without and with external data, respectively. The supervision also benefits two existing optimizers on ATM-Bench, while new-user experiments demonstrate continued design optimization using supervision generated from the target histories. 

---
# LLM-Assisted Optimization for Simulation-Based Redesign 

**Authors**: Chan-Eui Song, Jeremy J. Michalek, Christopher McComb  

**Link**: [PDF](https://arxiv.org/pdf/2610.39275)  

**Abstract**: Engineering design frequently involves redesigning a prior solution to adapt to changes in context, such as material prices or system requirements. Redesign often leverages simulators too complex and/or computationally intensive for traditional numerical solvers, which lack engineering knowledge and context, to ensure global solution quality. Large language models (LLMs) can interpret semantic information, including engineering knowledge about the system and changes in context, to help guide search, and recent studies have shown that incorporating this knowledge into optimization can improve search performance. However, how best to combine LLMs with numerical solvers in redesign remains unclear. We develop and test an integrated LLM-optimization approach for redesign and compare three methods for leveraging LLMs (warm start, variable fixing and ReAct) and four black-box solvers (Bayesian optimization, genetic algorithms, mesh adaptive direct search and random search) on an electric vehicle battery pack cost minimization case study across four types of redesign scenarios. We find that the integrated approach can match feasible design discovery of the numerical-solver baselines with 80% fewer simulator evaluations while tripling the fraction of runs that find near-optimal solutions. We find that the best LLM-solver integration approach and the way the prior solution is leveraged depend on the structure of the design requirement change, and that the benefit of increasing LLM model size grows with the extent to which the LLM controls the search. These results show clear potential for LLMs in redesign tasks for guiding or replacing numerical solvers. 

---
# Online Dense Reinforcement Learning with Planning-Grounding Agent for UI Navigation on Live IOS Apps 

**Authors**: Peitong Duan, Khashayar Gatmiry, Jeffrey Nichols, Samy Bengio, Emmanuel Abbe  

**Link**: [PDF](https://arxiv.org/pdf/2610.39264)  

**Abstract**: While frontier vision-language models demonstrate strong capabilities in user interface (UI) navigation, their high inference costs prohibit everyday use. Conversely, smaller models trained via supervised fine-tuning (SFT) or offline reinforcement learning often memorize static trajectories, leading to brittle execution and an inability to recover from errors. To address these limitations, we propose a novel online reinforcement learning framework that trains UI navigation agents directly on live iOS applications. Our framework decouples the agent into two specialized components: a Planning agent and a UI Grounding model. We rely on an LLM-as-a-Judge to provide comprehensive reward signals, combining dense, step-level feedback to evaluate intermediate progress with a terminal reward for overall task completion. To effectively leverage this dual reward structure, we optimize the agent using Proximal Policy Optimization (PPO) with step-level advantage estimation. Furthermore, we introduce GiGPPO, a novel neighbor-conditioned value estimation method that improves critic stability by leveraging reward signals from visually similar states across concurrent trajectories. To support this online training paradigm at scale, we introduce an efficient infrastructure that orchestrates hundreds of concurrent virtual iOS devices. Evaluations on rigorous benchmarks, including UINavBench, iOSWorld, and AndroidWorld, demonstrate that our online RL trained agents substantially outperform both SFT and terminal-only RL baselines while accelerating training convergence. Notably, our 8B RL agents achieve nearly a sevenfold improvement over the SFT baseline in out-of-distribution generalization to Android environments, while outperforming Gemini-3.5-Flash and tying with Gemini-3.1-Pro on real iOS applications. 

---
# Where to Learn and How Far to Roll Out: Efficient Adaptive Training for GUI Agents 

**Authors**: Yuxiang Chai, Quyu Kong, Panrong Tong, Steven Hoi, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.39251)  

**Abstract**: Trajectory-level GRPO is a common approach for training multi-step GUI agents, but repeatedly rolling out complete trajectories from the task root can be inefficient: task-level rewards provide coarse supervision over which intermediate decisions require improvement, while already-mastered prefixes are repeatedly re-executed before reaching difficult states. We introduce a two-stage adaptive training framework that jointly addresses where to learn and how far to roll out. We first construct reusable intermediate checkpoints from successful expert trajectories and select informative pivots using action disagreement, which measures uncertainty directly from sampled GUI actions. Stage I performs one-step RL at these pivots to improve uncertain local decisions. Stage II then progressively expands policy responsibility through adaptive Local, Suffix, and Root training according to task progress. Recoverable checkpoints enable repeated training from intermediate states without replaying the preceding trajectory. Across MobileGym and CUA-Gym-Hub, our method reduces environment interaction by 56-61% relative to Vanilla GRPO while improving task performance, demonstrating that jointly allocating training across states and rollout ranges can substantially improve GUI-agent RL efficiency. 

---
# SocialArena: Evaluating Social Intelligence in Large Language Models 

**Authors**: Zongqi Wang, Rui Wang, Yuchuan Wu, Hua Yang, Bingxue Qiu, Tianle Gu, Bing Zhao, Yujiu Yang, Jieping Ye, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.39247)  

**Abstract**: We introduce SocialArena, a benchmark for systematically defining and evaluating the social intelligence of large language models (LLMs). Although LLMs are increasingly deployed in socially interactive applications such as mental health counseling, companionship chatbots, and educational tutoring, existing evaluations remain fragmented focusing on isolated capabilities or single scenarios and evaluation reliability is largely unverified. This work makes three main contributions to address these limitations. (1) We first propose a unified three-tier framework that defines social intelligence along three capability dimensions: role-playing ability, mental state reasoning, and behavioral reasoning. To operationalize this definition, we design a role-playing dialogue paradigm with explicit character thinking for evaluation, enabling simultaneous assessment of these capabilities in a manner that mirrors realistic human interactions. (2) To ensure diversity across social scenarios and thereby prevent evaluation conclusions from being confined to specific scenarios, we introduce the U-TRIMF framework, which guides us in synthesizing social contexts with broad coverage. (3) To obtain reliable evaluation, we invest $132.4K to recruit and train human annotators and adopt a human-annotated arena-style protocol, where annotators compare model responses and provide preference labels along with natural-language critiques. We hope that SocialArena can provide valuable insights and guidance for future LLM improvements. 

---
# Beyond LLM Wrappers: Context Engineering and the Three-Engine Architecture for Reliable Financial AI 

**Authors**: Pradeep Kumar GS, Ravi Tanniru  

**Link**: [PDF](https://arxiv.org/pdf/2610.39230)  

**Abstract**: Financial AI systems must support decision-bearing questions involving payments, collections, cash flow, and compliance while ensuring numerical accuracy, traceability, and auditability. Large language models (LLMs) enable natural language interaction but cannot independently guarantee reproducible calculations or evidence-grounded recommendations. We present a production-grounded architecture integrating three components: a financial knowledge graph for enterprise relationships, temporal context, and provenance; a deterministic optimization engine for financial calculations and decisions; and a multi-agent LLM layer for intent interpretation, task orchestration, explanation, and claim validation. A coordination layer, termed context engineering, selects, ranks, composes, and validates query-specific context across ERP systems, tools, and tenants.
Based on observations from 15 deployments over approximately 11 months, single-agent pipelines exhibited hallucination rates of about 7-9% (5-10% across query types), including invented invoices and unsupported financial claims. Graph-backed multi-agent validation reduced observed rates to below 1%, although these findings are observational rather than causal. ERP data quality and tenant-specific schemas remained major implementation challenges. Reusable mappings, graph templates, and validation policies coincided with a reduction in deployment time from roughly six to two weeks.
We argue that financial context engineering extends beyond text-centric retrieval-augmented generation by requiring structured relationships, temporal validity, exact numerical evidence, access controls, and validation against source records and solver outputs. We outline research challenges in evaluation, reproducibility, privacy, context quality, and cross-tenant generalization. 

---
# Goal-Aware Dynamic Multi-Agent Collaboration for Long-Horizon Tasks 

**Authors**: Xu Shen, Lingyong Yan, Shuaiqiang Wang, Jianmin Wu, Zhikai Chen, Jiliang Tang, Jie Ren  

**Link**: [PDF](https://arxiv.org/pdf/2610.39193)  

**Abstract**: LLM-based agents are increasingly used to complete complex, long-horizon tasks without human intervention. To make progress toward the task goal, agents need to perform different duties, such as planning and verification, when needed. To understand how single agents manage these duties during long-horizon execution, we conduct a preliminary study. Our analysis shows that verification has the strongest association with task success, but often occurs late, leaving limited opportunities for correction. We also find that the proportion of duty-relevant information decreases as records from different duties accumulate in a shared context. These findings lead us to examine whether multi-agent systems (MAS) can address these problems by assigning duties to agents with separate contexts. Our analysis shows that this separation alone is insufficient. Agents can lose track of the overall goal and repeat prior investigations. They may also lack necessary background while receiving irrelevant information from others. To maintain goal awareness and coordinate collaboration as task needs change, we propose Goal-Aware Dynamic Multi-Agent Collaboration (GAD). It uses a growing execution graph to organize subtasks and their dependencies. To keep agents focused on their goals, each context retains the original instruction and the assigned task's goal and acceptance conditions. To adapt collaboration and information sharing during execution, agents can divide tasks into subtasks and specify dependencies based on execution evidence. These dependencies guide when agents work and which results they receive. Verification and recovery check subtask results and full task completion against their goals, providing feedback for repair when checks fail. 

---
# Reasoning or Pattern Exploitation? Mechanistic Insights from RL-Trained Language Models 

**Authors**: Aishik Nagar, Cheston Tan  

**Link**: [PDF](https://arxiv.org/pdf/2610.39182)  

**Abstract**: Large language models (LLMs) are increasingly described as acquiring "reasoning" skills after reinforcement learning with verifiable rewards. However, whether these gains reflect genuine reasoning or more superficial adaptations remains underexplored. In this work, we probe LLMs trained in a finite and exhaustively analyzable logical domain, namely Tic-Tac-Toe, and trace how internal representations evolve across reinforcement learning (RL) with Group Relative Policy Optimization (GRPO). Quantitatively, RL yields high accuracy and robustness across prompt variations, suggesting the emergence of game reasoning. Mechanistic interpretability, however, paints a different picture: features extracted with sparse autoencoders (SAEs) reveal that models primarily adapt to rely more heavily on information already explicit in the prompt, such as whose turn it is, game progression and board occupancy. In comparison, high-level concepts like board symmetries and strategic forks remain weakly represented, raising concerns that reasoning benchmarks overestimate model abilities. This suggests that RL-driven "reasoning" may conflate task-specific adaptations to input information with reasoning ability. We provide, for the first time: (i) a systematic interpretability pipeline tracing representation dynamics across RL training in small-scale LLMs; (ii) an extension of SAE-based feature discovery to hypothesis-driven testing in a finite logical domain; and (iii) an interpretability-based demonstration that RL post-training neither adds nor removes concept information but reorganizes which features dominate the representation, with causal interventions locating the decision downstream of it, in the late-layer state. 

---
# Certified Access: Test-Time Compute on Conservative Physics-Structured AI Memory 

**Authors**: Pratik Jawahar, Riccardo Maggioni, Jonas Petersen, Matteo Cozzi, Gian-Alessandro Lombardi, Marek Gajewski, Camilla Mazzoleni, Federico Martelli  

**Link**: [PDF](https://arxiv.org/pdf/2610.39073)  

**Abstract**: Test-time compute mechanisms, such as retries, confidence-based escalation and non-local routing, are conventionally deployed as black-box additions to trained models. They typically add forward passes or halting heads, but give no structural account of what the extra compute buys or what it costs against the model's stability guarantees. We therefore propose a theoretical and empirical framework for test-time compute on a Latent Dynamics Model used as the memory. When an encoder's latent state $(q,p)$ is advanced by a damped symplectic (velocity-Verlet) step of a learned Hamiltonian, every mechanism that expands access has an exact phase-volume Jacobian, a bounded energy change and an exact latch-transport law. We split reachability failures into escape failures and reach failures. A Lorentz squeeze resolves escape failures by injecting bounded, re-absorbable energy. A gated canonical translation (a one-step channel across $C_T$) resolves reach failures at $\detJ=1$ exactly. We verify this certificate stack on a designed analytic testbed. Reaching just beyond $C_T$ with the squeeze requires energy that grows quadratically in the excess distance. The gated channel's energy cost equals the potential difference between its endpoints, which is zero on the matched channel measured here, and so does not grow with distance in this testbed. An untrained state-replacing map annihilates phase volume and erases stored charge, while the gated channel transports it. On a learned physics-structured memory, we show calibrated compute rationing and evaluate the scaling limits of non-local routing against a physics-free baseline. The main contribution is the formalization and verification of the certificate stack, not benchmark performance. 

---
# When the Same Prompt Gives Different Answers (and the Same Answer Isn't Always Right): A Taxonomy of Failure Mechanisms, Testing, and Mitigation Strategies for LLM-Assisted Evidence Synthesis Read-and-Evaluate Tasks 

**Authors**: Carole Lunny, Ian Goodman, Stefano Berrone, Connor TA Brenna, Emma K Reid, Ivan Perez-Neri, Shailesh Kolekar, Nityanand Jain, Ashwin Kamath, Lucas HCC Santos, Yulia Otmakhova, Heba Hussein, Diana M Cespedes, Gioia Dahdal, Salmaan Kanji  

**Link**: [PDF](https://arxiv.org/pdf/2610.39070)  

**Abstract**: LLM read-and-evaluate tasks include study screening, data extraction, risk-of-bias assessment, and critical appraisal in the evidence synthesis field. These task differ in the amount of LLM interpretation they require. Some tasks involve retrieving relatively simple information, such as an author name, sample size, or publication year. Others require the LLM to read source material, interpret it against predefined criteria, and make a judgement. Researchers should assess whether LLM judgements are accurate, supported by appropriate source evidence, and reproducible under specified conditions. A correct judgement does not necessarily indicate that the underlying evaluation process is reliable. We developed a taxonomy for identifying mechanisms that may compromise these properties and for informing the selection and reporting of appropriate evaluation methods. The taxonomy outlines potential ways that LLM judgments from read-and-evaluate tasks can fail, called TRAILS (Taxonomy of Reliability And Instability in LLM-assisted evidence Synthesis). The TRAILS taxonomy addresses two complementary questions. First, what mechanisms can cause the same read-and-evaluate task to produce different judgements across repeated LLM runs? Second, what mechanisms can systematically affect the judgement or required output in a way that is not apparent from the judgement itself? This study presents 38 mechanisms and identifies the deployment settings in which each can operate, approaches for testing them, and mitigation strategies. We also show why repeatability or accuracy does not rule out a hidden judgement failure and why human evaluation should assess whether a judgement is grounded in the supplied evidence, independent of inappropriate external influence, and produced through a process that can be examined and verified. 

---
# When Should AI Decide, Defer, and Deviate? Responsible Automation for Deceased Donor Kidney Allocation 

**Authors**: Madeleine Pollack, Thomas Daillak, Mehmet Toner, Korkut Uygun, Heidi Yeh, Swati Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2610.39045)  

**Abstract**: We ask when an AI trained on historical decisions should be allowed to recommend decisions, defer to human experts, or in other cases, flag concerning patterns in data and deviate from past practice. This can be challenging because past decisions record what was done rather than what should have been done, and where data is sparse, a model's predictions may be too uncertain to act on. We provide a framework for addressing this problem using a case study in deceased-donor kidney allocation. Nearly 30% of kidneys recovered from deceased donors in the United States are never transplanted, largely because they are hard to place through the standard waitlist sequence. One can use AI to flag kidneys at risk of delayed or failed placement for out-of-sequence (OOS) allocation, but this comes at the cost of bypassing candidates higher on the waitlist. Thus, OOS allocation should only be used when the organ is demonstrably at high risk of discard on the standard sequence. Using Organ Procurement and Transplantation Network match-run data from 2016-2019 and 2023-2025 (excluding COVID-19-disrupted years), we build the Arbitrator framework, which combines various interpretable models and acts only when the uncertainty in their predictions cannot change the preferred action. Our Arbitrator resolves 93.5% of organ placement decisions using explicit medical criteria learned from data, and defers the remaining 6.5% to a human or a more complex model. We show that the Arbitrator correctly identifies 70.1% of historically discarded low-KDPI kidneys as OOS candidates, projecting that up to 43.9% of historically discarded kidneys could be flagged for expedited placement and potentially transplanted. We then exploit the interpretability of the Arbitrator to create an agentic audit comparing the learned rules to published clinical literature in order to flag decision-making patterns that are contentious. 

---
# Formal Specifications with Data-Driven Signal Analysis 

**Authors**: Roderick Bloem, Benedikt Maderbacher, Cristinel Mateis, Dejan Ničković, Sergii Tkachov, Björn Ziegler  

**Link**: [PDF](https://arxiv.org/pdf/2610.39007)  

**Abstract**: Developing analog and hybrid systems requires substantial domain knowledge that is hard to articulate in a formal specification; without additional effort, this domain knowledge is also inaccessible to machine learning methods. We propose a hybrid approach for analyzing and monitoring real-valued signals that integrates specification-based and data-driven techniques. We use regular expressions with interval operations to identify regions of interest in a signal, classical signal processing methods to extract semantically meaningful features, and semi-supervised machine learning to infer complex relationships that distinguish correct from faulty behaviors. The relationships learned from normal behavior data can automatically detect incorrect behavior in the same system or variants thereof. We evaluate the method on a voltage regulator, demonstrating its effectiveness in detecting undesired system behaviors. 

---
# Agentic AI Reasoning and Learned Sampling for Task-Aware Projection Selection in Sparse-View CT: Predicting Hemodynamics 

**Authors**: Mahmoud Elhadidy, Roshan M. D'Souza, Amirhossein Arzani  

**Link**: [PDF](https://arxiv.org/pdf/2610.39003)  

**Abstract**: Sparse-view computed tomography (CT) reduces the number of acquired projections, but conventional sampling strategies are designed primarily for image reconstruction rather than for downstream physical inference. Here, we investigate optimal task-aware projection selection for directly predicting hemodynamics from sparse CT angiography (CTA) measurements. Using computational fluid dynamics simulations of intracranial aneurysms, we generate synthetic fan-beam CTA projections and train a Fourier neural operator (FNO) to predict two targets: the spatially resolved time-averaged wall shear stress (TAWSS) field and the spatially averaged TAWSS. The FNO was coupled with optimal input selection for simultaneous optimal sampling and training. Our goal was to investigate if agentic AI could discover optimal sampling strategies that outperform current methods. We compared fixed uniform and golden-angle sampling with three task-aware optimal selection strategies: Gumbel-Softmax, a Soft top-K method, and an agentic AI selector. The methods were compared at different view selection budgets. A detailed analysis of the AI agent was performed to understand its reasoning and performance. For spatially resolved TAWSS, Agentic and Soft top-K achieve the lowest mean errors (Agentic performs slightly better) and substantially outperform fixed sampling under extreme sparsity. For spatially averaged TAWSS, optimized selection becomes important at extremely sparse acquisitions. Analysis of the selected masks reveals multiple distinct subsets concentrated around informative regions. Moreover, masks optimized for hemodynamic prediction yield poor image reconstructions, showing that task-optimal acquisition differs fundamentally from reconstruction-optimal acquisition. This work suggests the possibility of designing sparse acquisitions directly around downstream physical quantities. 

---
# Syntactic Segmentation and Collage Generation for Videogame Levels with Perceptual Distance 

**Authors**: Ross Mawhorter, Matthew Guzdial  

**Link**: [PDF](https://arxiv.org/pdf/2610.38987)  

**Abstract**: In procedural videogame level design, many existing techniques combine hand-authored level chunks. Some existing analysis techniques also rely on a higher-level graph structure which abstracts the level design. In this paper, we explore techniques for computing both of these things automatically from an existing design. We introduce the concept of perceptual distance, based on analysis techniques from the architecture subfield of Space Syntax. We use this perceptual distance to segment level designs with hierarchical clustering. We compare with past work on StarCraft analysis, show results on rooms from Super Metroid, and also demonstrate a proof-of-concept level generation algorithm based on this segmentation method. 

---
# Context Compaction for World Models 

**Authors**: Hayder Tirmazi  

**Link**: [PDF](https://arxiv.org/pdf/2610.38980)  

**Abstract**: World models let observers explore and interact with generated worlds. Recent work shows that when observers revisit a region, current models often fail to regenerate it consistently. We view this as a context retrieval problem. LLM agents manage context retrieval at two layers of their stack. At the model layer, the KV cache design determines how much context fits in memory. At the agent harness layer, harnesses such as Claude Code pick the context sent to the LLM. Prior work on world models has mostly focused only on the model layer. In this paper, we make three contributions. First, we build a benchmark to measure world model context retrieval. It generates new worlds for every episode, preventing a model from recalling rooms using training data. It also separates context retrieval from guesses based on nearby frames. Second, we build an agent harness that performs context compaction for world models, which to our knowledge no prior work has studied. Third, we use our benchmark to study four world model designs. Context compaction lets our world models retrieve rooms up to 4 times beyond their context window without retraining. However, world models with sliding-window attention lose most of their context retrieval when dreaming. Therefore, a dreaming world agent benefits more from extending the reach of its world model using context compaction than switching to sliding window attention at the model layer. 

---
# Dungeons & Dragons Encounter Analysis with Semi-Random Simulation in Foundry VTT 

**Authors**: ravyn Hardcastle, Matthew Guzdial  

**Link**: [PDF](https://arxiv.org/pdf/2610.38975)  

**Abstract**: Dungeons & Dragons (D&D) is a tabletop roleplaying game (TTRPG) in which a group of players, guided by a Game Master, roleplay a fantasy adventure with simulated combat and improvisational narrative. In D&D, an encounter is a single instance of simulated combat, where the players must use their characters to attempt to defeat a selection of enemy characters that the GM controls. D&D has multiple systems that approximate encounter difficulty by summing individual difficulty values that designers predefine for each enemy. However, since players and enemies can have a wide variety of capabilities, encounter difficulty depends on more than just the estimated difficulty of each monster. For example, these metrics do not account for positioning and dungeon layout, the synergies between monster abilities, and player party composition. We have created a tool, Mage-Hand, that can simulate D&D encounters semi-randomly using Foundry Virtual Tabletop, a virtual tabletop (VTT) system. In this paper, we show that our tool's simulated winrates have a strong correlation with D&D's published difficulty evaluation metrics, while also better aligning with some players' reported experiences. 

---
# Efficient Agentic Audio-Visual Question Answering through Toolset Pruning and Modality Consolidation 

**Authors**: Yi Wang, Tianzi Wang, Peter Bell  

**Link**: [PDF](https://arxiv.org/pdf/2610.38960)  

**Abstract**: Fine-grained audio-visual question answering (AVQA) requires models to resolve complex cross-modal correspondences and temporally grounding from sparsely distributed evidence. Omni-modal agents, which autonomously orchestrate perception tools to actively collect evidence, achieve strong performance, but it remains unclear whether active perception from both audio and visual modalities depends on broad specialized toolsets, especially audio or visual modality-specific tools. Such systems may also introduce long execution trajectories and substantial inference cost. We investigate whether comparable performance can be retained with a compact workflow exposing only essential perceptual capabilities. Starting from the specialized-tool OmniAgent, we conduct staged controlled experiments that prune the toolset and progressively consolidate modality-specific tools into omni-modal tools, motivating a two-tool agent comprising only Omni-modal captioning and Omni-modal QA. Across three widely used AVQA benchmarks, our compact agent achieves comparable or better accuracy than the specialized-tool agent while reducing inference cost by around 70%. 

---
# TreasuryQA Bench: Can AI Agents Work an Archive of Scanned Financial Records? 

**Authors**: Ritiz Tambi, Sahar Shayegan, Joshua Meyer, Sun Kim, Ali Khan, Victor Shih, Mehdi Jamei, Andi Partovi  

**Link**: [PDF](https://arxiv.org/pdf/2610.38924)  

**Abstract**: Organisations keep decades of financial records as scanned PDFs on shared drives, and AI agents can now reach those drives through Model Context Protocol (MCP) servers. Existing benchmarks do not mirror this production setting because they simplify the task. In some, the relevant documents are handed to the agent, so it never has to search. In others, the agent searches through documents with pre-extracted text, so it never has to read scanned images. We introduce TreasuryQA Bench, which requires both: 70 questions over 1,435 scanned U.S. Treasury publications, held in Veris's high-fidelity Dropbox twin. In each task the agent searches for the relevant file through Dropbox's MCP server, reads the pages it needs with Reducto's hosted OCR server or its own tools, and replies with one exact figure. A Python grader checks each answer against an accepted range around the correct value. In this paper we present results from ten agents. The best, Claude Opus 5.5 in Claude Code, solves 64.8% of attempts and the worst, GPT-5.6 Terra in Codex, solves 16.7%, so the benchmark is far from saturated. We also test how scan quality affects performance: every agent solves fewer tasks on the blurriest scans. Most failed attempts reply with a wrong number rather than no number, and the most common cause is a digit misread from the scan. Three worked examples are included with their answer keys. 

---
# Aligning Autoresearch: Co-Improvement between Agents and Humans 

**Authors**: Tianjian Li, Chuanyang Jin, Seungone Kim, Tong Chen, John Hoffman, Yoram Bachrach amd Pedro Rodriguez, Ping Yu, Eryk Helenowski, Swarnadeep Saha, Jack Lanchantin, Weizhe Yuan, Ilia Kulikov, Jason Weston  

**Link**: [PDF](https://arxiv.org/pdf/2610.38882)  

**Abstract**: Autoresearch agents can now attempt long-horizon AI research end-to-end, i.e., self-improvement. However, best results are still obtained by research agents interacting with human researchers, and improving AI's ability to perform this co-research will only reinforce this effect---a process termed co-improvement (Weston and Foerster, 2025). Further, aligning with human expectations is likely a safer path for research agents, especially given recent agent reward hacking phenomena (OpenAI, 2026b). In this work we address these issues in four ways: (i) we empirically study the role of humans in directing autoresearch agents, focusing on the domain of benchmark creation, and find humans still play a pivotal role particularly in the ideation phase; (ii) we also develop a benchmark, HackBench, that shows current autoresearch agents are not aligned and are unable to achieve hack localization sufficiently; (iii) we present a method to train agents for improved human interaction by using human simulators as a proxy during reinforcement learning (RL); and finally (iv) we develop a method to improve model text generation, and hence agent communication, by RL with rubrics aligned with human experts. Together, our training and evaluation recipes provide a path toward more capable and aligned (co-)research agents. 

---
# Auditable Belief Updates as a Missing Safety Layer for LLM-Powered Multi-Agent Systems 

**Authors**: Zhuoyun Li, Boxuan Wang, Jinwei Hu, Xiaowei Huang, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2610.38869)  

**Abstract**: Large Language Models (LLMs) are increasingly being used as cognitive interfaces that shape how users interpret evidence, communicate, and make decisions. As deployed systems become more multimodal, tool-using, and multi-agent, safety failures no longer arise only from isolated harmful outputs, but also from how agents update, store, and propagate epistemic commitments across retrieval, memory, tools, and peer communication. In this paper, we define belief as an engineering abstraction for evidence-based commitments that guide downstream action. Building on this view, we propose the Explicit Belief Multi-Agent System (EB-MAS), which externalizes belief states into a structured substrate, uses belief operators to propose candidate updates, and applies a Cognitive Firewall to bound or reject updates that are weakly evidenced but strongly driven by vulnerability cues. We also discuss the risk of group belief at the group level, including manufactured consensus, collusion, and cascading misinformation, and outline practical integration challenges around grounding, false positives, scalability, and ethical misuse. Therefore, scalable agent safety requires not only better output filtering or stronger base models, but also explicit interfaces for tracing, calibrating, and constraining belief revision at runtime. 

---
# Where You Inject Matters: The Trust Region Effect in LLM Agents 

**Authors**: Chengquan Guo, Xiaodong Liu, Weiwei Yang, Mingqian Feng, Jialin Song, Qiyao Ma, Carolyn Buractaon, Christopher White, Bo Li, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.38867)  

**Abstract**: LLM agents are increasingly adopted in long-horizon tasks and interact with diverse external environments. Despite their growing capabilities, these agents remain vulnerable to real-world attacks such as indirect prompt injections, which manipulate agents through malicious instructions embedded in external environments. While existing risk assessment work has primarily focused on the injection content, the effect of where an injection appears has not been systematically characterized. Studying injection placement requires agents to operate over longer-horizon tasks and interact with diverse environmental surfaces. Thus, we build three agentic testbeds spanning tool use, coding, and daily workflow domains with diverse injection surfaces. Holding the injection content fixed, we find that agent susceptibility varies across injection placements. We refer to this phenomenon as the trust region effect: agents implicitly assign different levels of trust to different regions of their environments. For example, an agent powered by GPT-5.4 follows an unsafe injection instruction placed in a repository documentation file (i.e., a README or TODO file) 2.6 times more often on average than the same instruction placed in a code or data file. Building upon this observation, we develop learning-based attackers that generate injection content and select better injection placements. Our approach demonstrates that both where an injection is placed and what content it contains can affect the strength of indirect prompt injection attacks. To the best of our knowledge, this work provides the first systematic characterization for the effectiveness of different injection placements, spanning both coarse-grained environmental surfaces and fine-grained positions. The uneven trust that agents implicitly place across their environments provides important insights for future attack and defense design. 

---
# Learning to Reason from Reference Samples Alone: Reinforced Adversarial Learning for Modern LLM Post-Training 

**Authors**: Albert Bou, Javier Valdes-Aleman, Jon M. Laurent, Samuel G. Rodriques, Andrew D. White, Michaela Hinks, Siddharth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2610.38851)  

**Abstract**: In many settings where LLM post-training is desirable, high-quality answers are recognizable but difficult to score with explicit rewards. Reinforcement learning with verifiable rewards (RLVR) is limited to tasks where an explicit correctness signal is available. Prior adversarial approaches to training reasoning LLMs either complement verifiable rewards with co-trained critics or require reference reasoning traces. We ask whether a co-trained LLM judge can turn reference answers alone into the sole reward for on-policy post-training of reasoning LLMs. We call this method Reinforced Adversarial Learning (RAL). We first validate RAL on three verifiable tasks (integer multiplication, code-based arithmetic-expression synthesis, and SMILES-to-formula translation), using the verifiers only for evaluation. With sufficiently capable judges and enough training, RAL matches direct RLVR. We then apply RAL to scientific abstract generation in two settings. In the first setting, RAL guides the generator to produce abstracts in the scientific domain of the reference abstracts, even though the prompt leaves that domain unspecified. In the second setting, we generate abstracts for U.S. National Institutes of Health grant proposals conditioned on the grant title. Human scientists and an ensemble of LLM judges find our generations harder to distinguish from real abstracts than those of frontier-LLM baselines. These results suggest a path toward post-training reasoning models in scientific and other high-value domains, where increasingly capable judges infer hard-to-specify reward functions from example answers. 

---
# Leaving, Staying, or Living There? Duration Commonsense Reasoning with Human-LLM Co-annotations 

**Authors**: Chenyang Yu, Zhaomin Xiao, Yan Huang, Eduardo Blanco  

**Link**: [PDF](https://arxiv.org/pdf/2610.38755)  

**Abstract**: Gardening in the backyard implies living there, while checking into an Airbnb more likely means staying for a few days. Such duration judgments are important for understanding human mobility, travel behavior, and social context. However, duration is rarely stated explicitly in social media posts and must instead be inferred from noisy textual, visual, and contextual cues. In this paper, we investigate whether vision-language models possess such duration commonsense. We create a new corpus of geotagged tweets co-annotated by humans and LLMs to serve as a testbed for this task. We find that state-of-the-art models struggle with duration inference, even when guided by multi-turn Chain-of-Thought prompting. Error analyses further reveal that models often fail to integrate critical contextual information, or they overly rely on prominent visual details while overlooking subtle hashtags and semantic nuances. These findings underscore the significant challenges remaining in context-aware multimodal understanding. 

---
# From Grind to Grasp 

**Authors**: Peter Sunehag  

**Link**: [PDF](https://arxiv.org/pdf/2610.38728)  

**Abstract**: Discussions of complexity, knowledge and intelligence often overlook the work involved. For example, outputs generated by short programs have low algorithmic complexity but can have high computational depth, the least work needed to produce them from near-shortest programs. Producing them with less work can require longer descriptions, giving high work-bounded complexity. We introduce a formalism that compares achievements under work and information limits and describes how learning and persistence can make deep achievements shallow by reusing products of earlier work. For an agent with a limited work budget, the value of building and maintaining such resources depends on how they support its goals over later tasks. We define grasp as performance across work budgets at a fixed information allowance. Low grasp can still yield success through grind at high cost in work, without necessarily improving human grasp. Applied to improvement itself, grasp gives a criterion for recursive self-improvement in which successors make comparable further improvements more efficiently. Another route to faster progress is to expand the resources brought to bear through knowledge sharing and coordination across society, even while comparable advances require increasing work. Grasp also applies to collectives, which may outperform their members yet require more total work than one system with their combined resources. Establishing and sustaining organisation can itself be deep even when the primary task is simple. These demands may help explain the richness of biological and social systems. In agent collectives, this organisational work can generate goals beyond assigned tasks, including maintaining the collective itself. We therefore caution against allowing an emergent agent society to assume authority over social coordination, since it could increasingly direct work towards its own persistence and goals. 

---
# Beyond fabrication: Misattributed numbers in papers written by AI research agents 

**Authors**: Mosab Alfaqeeh, Omar Hourani, Muath Hourani, Hashem Alfaqeeh  

**Link**: [PDF](https://arxiv.org/pdf/2610.38691)  

**Abstract**: Verifying numerical claims in research written by AI agents is an important but challenging problem. Recent benchmarks show that many agent-written papers fail integrity checks, but such verdicts, often produced by a single LLM judge, identify failure without explaining its nature, and existing checks either test a paper for internal consistency or test whether a reported value appears somewhere in an agent's output. We re-analyze all 540 released judgments from a recent public benchmark of AI research agents (nine agents, fifteen open research problems) and manually label 256 numerical claims from 18 submissions. Failures concentrate on alignment between the paper and the underlying work rather than overt cheating, and about four in five unsupported claims correspond to values in the agent's own files that come from the wrong experiment, metric, or run. We introduce a deterministic claim-to-evidence checker that scores whether the surrounding context identifies the claimed result and calibrates each claim against numerical decoys. With its rules fixed before labeling, it certifies claims with over 90% precision against blinded annotators (78 of 85; Cohen's kappa = 0.89), while verifying only a subset of claims; most claims it cannot verify are nevertheless traceable, so failure to verify is not evidence of falsehood. An LLM verifier given the same evidence recovers more traceable claims but accepts about three times as many misattributed ones, so the two are best used together. Applied to 99 submissions, the checker reveals large differences in traceability across agents, and a second judge from a different vendor preserves the official agent ranking while passing twice as many papers. These findings motivate provenance-first auditing of AI-produced research. 

---
# Deogen: Automatic Benchmark Generation for Reasoning over Policies 

**Authors**: Sam Kouteili, William Liang, Keeran Murray, Adriana Zhou, Ramesh Viswanathan, Scott Shapiro, Ruzica Piskac  

**Link**: [PDF](https://arxiv.org/pdf/2610.38687)  

**Abstract**: Language models are increasingly being deployed in regulated settings, where a model must apply given rules to the facts of a situation to determine a verdict. Benchmarking this capability, known as deontic reasoning, remains a costly and manual process, with existing datasets largely hand-crafted. In this paper, we present DeoGen, a scalable, automatic, and parametrizable benchmark generator for deontic reasoning. By autoformalizing policy rules to first-order logic, we can construct decision trees that capture possible outcome trajectories. Each path down a tree becomes a benchmark entry, with each tree producing combinatorially many entries relative to tree depth. Paths ending at a leaf entail a verdict, while non-leaf paths measure a model's ability to abstain when presented with insufficient information. Evaluations show frontier models return the correct verdict 89-92% of the time. Notably, performance degrades (22-40% correct) on entries without sufficient information to claim a verdict, even with explicit instructions to abstain in such cases. In all, DeoGen presents an automatic, scalable, and parametric method for generating deontic reasoning benchmarks. 

---
# Weaken to Strengthen: Weakening-Induced On-Policy Self-Distillation for Large Reasoning Models 

**Authors**: Yecheng Wu, Song Han, Han Cai  

**Link**: [PDF](https://arxiv.org/pdf/2610.38681)  

**Abstract**: Self-distillation offers a way to improve reasoning models when a suitable external teacher is unavailable. However, common approaches rely on privileged information to construct a stronger self-teacher, limiting their use when only unlabeled prompts are available. In this paper, we introduce Weakening-Induced On-Policy Self-Distillation (WISD), which instead constructs a weaker counterpart of the original model and uses the contrast between the two as a training signal without requiring privileged information. We explore two ways to construct the weaker model: low-bit quantization and magnitude pruning. We evaluate WISD across diverse architectures and scales on mathematical and coding reasoning tasks. Across these benchmarks, quantization and pruning improve average accuracy over the original models by 4.0 and 5.5 percentage points, respectively, achieving state-of-the-art performance even without privileged information. Notably, pruning raises Qwen3.5-4B's accuracy from 57.7% to 70.5% on HMMT 2025 and from 39.1% to 56.7% on LiveCodeBench v6. Quantization raises Qwen3.6-35B-A3B's accuracy from 68.5% to 74.6% and from 61.8% to 69.5% on the same benchmarks, respectively. These results establish WISD as a novel and effective self-distillation framework for large reasoning models. 

---
# Curriculum-Based Deep Reinforcement Learning for Robust Autonomous Rocket Landing under Atmospheric Uncertainty 

**Authors**: Atin Chowdhury, Piyush Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2610.38636)  

**Abstract**: Autonomous far-range rocket landing is a challenging control problem because navigation, descent-rate regulation, attitude management, and fuel use must be coordinated in the presence of uncertain atmospheric and vehicle dynamics. This paper presents a curriculum-driven proximal policy optimization (PPO) framework for three-dimensional rocket landing in a stochastic environment incorporating altitude-dependent aerodynamic drag, wind shear, temporally correlated turbulence, discrete gusts, uncertain propulsion and aerodynamic parameters, thrust-vector misalignment, engine lag, actuator-gain variation, noisy observations, and command latency. Learning is organised into far-range approach, soft descent, and integrated landing stages. Policy transfer, critic warm-up after reward changes, rehearsal of earlier tasks, and progressive expansion of weather severity are used to support learning across these coupled phases. Across 100,000 evaluation episodes, the policy achieves 96,656 successful landings, corresponding to a success rate of 96.656\% and a 95\% Wilson confidence interval of 96.543--96.766\%. Success decreases from 98.84\% under zero-disturbance conditions to 88.07\% in the highest weather-severity band. Successful touchdowns exhibit small miss distances and low residual horizontal and vertical velocities. An exploratory logistic-regression analysis identifies mean wind speed as the strongest measured correlate of success, with initial distance, initial altitude, initial vertical velocity, and wind-shear exponent also associated with outcomes. The results quantify both the effectiveness of the learned policy over the evaluated disturbance distribution and the degradation that occurs under severe compound atmospheric conditions. 

---
# "Close enough?" An investigation into multiplication errors in LLMs 

**Authors**: William Bolton, Bezawit Tilaye, Huw Day  

**Link**: [PDF](https://arxiv.org/pdf/2610.38634)  

**Abstract**: Large language models (LLMs) frequently fail at multi-digit arithmetic, but the underlying mechanisms governing these failures remain poorly understood. While recent work demonstrates that addition errors follow a "noisy quantisation" model localised around carry boundaries, whether this framework generalises to multiplication remains an open question. In this work, we evaluate multiplication capabilities across several small base LLMs on a dataset of 16,991 problems spanning varied operand lengths. We show that multiplication errors do not conform to the addition-style noisy quantisation model. Instead, we uncover a novel error regime in small Qwen models: errors in lower-order columns predominantly preserve parity. Conversely, higher-order digit errors shift toward off-by-one carry failures. Our findings suggest that arithmetic computation in LLMs relies on distinct, multi-periodic internal representations where parity is tracked more robustly than exact digit identity, providing new mechanistic insights into numerical reasoning in transformers. 

---
# Right Numbers, Wrong Questions: Evaluating LLMs on Underspecified Financial Queries 

**Authors**: Radha Yadav, Prasanna Devadiga, Peeyush Mishra, Apoorv Singh  

**Link**: [PDF](https://arxiv.org/pdf/2610.38631)  

**Abstract**: Financial assistants answer questions that sound factual but depend on what the user leaves unsaid: what a data figure measures, which product variant is meant, and the investor's situation and the rules that apply. An assistant that ignores these can still answer with a real number from its data, one that answers a different question. We call this frame collapse. We study it in Indian mutual funds, where product variants and tax rules are formally defined, so the correct answer can be computed for every reading. One study tests each challenge. Given an ambiguously labelled figure, models used it, often under the user's label. Given every alternative labelled, they listed them all, but the layout decided which came first. Missing a fact about the investor, weaker agents assumed a default or an outdated tax rule; requiring the fact did not guarantee its use. Because such answers match their data, checks against the data pass them. An AI judge caught mislabelled figures only when told what each field means, and still passed answers that silently chose another reading. We show how these failures arise, not how often, and propose controls for financial assistants. 

---
# Think Before You Act: Decision Gate for LLM Agents 

**Authors**: Iftach Shoham, Yuval David, Offer Akrabi, Erez Bilgory, Ido Levy, Sami Marreed, Sergey Zeltyn, Segev Shlomov  

**Link**: [PDF](https://arxiv.org/pdf/2610.38620)  

**Abstract**: LLM agents increasingly perform consequential actions through external tools, yet their proposed actions are often executed without independent verification of policy compliance or task completion. We introduce JevAct, a runtime verification framework that intercepts selected agent actions before execution and enables feedback-driven correction. JevAct compiles domain policies into executable state checks and structured questions evaluated by a prefill-only decision model, which produces probability-based judgments without relying on LLM generation. By combining deterministic verification of tool-state facts with probabilistic assessment of conversational context, the framework identifies potentially invalid actions and premature task termination, providing targeted feedback for bounded revision. JevAct operates independently of the underlying agent model and requires no modifications to its parameters or tools. We evaluate JevAct across eleven LLM agents on $\tau^2$-bench Airline and Telecom tasks and BFCL. Results show improvements in agent reliability across multiple models and benchmarks, with particularly consistent gains on Telecom and BFCL, while performance on Airline is more mixed. Overall, these findings suggest that combining structured decision models with executable verification can improve the reliability of tool-using LLM agents. 

---
# CALM: Conflict-Aware Lifecycle Memory Governance for Long-Horizon Language Agents 

**Authors**: Chang Li, Xianhao Ou, Xinyan Ma, Hao Li, Haoyuan Xu, Jiafeng Liang, Xiaohan Wang, Jiajun Chai, Guojun Yin, Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.38573)  

**Abstract**: A memory can be outdated for present-day reasoning yet indispensable for reconstructing the past. Relevance-based retrieval overlooks this tension: semantic relevance, evidence-grounded validity, and query-specific eligibility are distinct. We propose CALM, a conflict-aware lifecycle memory governance framework that decouples persistent memory preservation from query-time admission. CALM converts heterogeneous records into evidence-grounded memory cells, updates validity and lifecycle states using support, conflict, and dependency signals, and admits query-appropriate evidence through training-free or learned gating informed by temporal applicability. Across LongMemEval-S and LoCoMo with Qwen and GPT backbones, CALM outperforms state-of-the-art baselines, achieving the best overall accuracy. Compared with simple filtering and reranking, query-conditioned admission further improves evidence selection. Conflict-blocking and robustness analyses show CALM suppresses erroneous retrieved memories while retaining useful evidence. Historical-reconstruction experiments demonstrate that preserving superseded records supports accurate earlier-state reasoning without compromising current-state answers. 

---
# A-RSI: Adaptive Task Refresh for Zero-Data Multimodal Self-Improvement 

**Authors**: Yu Liu, Youxing Li, Yuedong Tan, Zhiwei Yang, Zihan Su, Hanqi Feng, Shaojie Zhang, Libin Liao, Zihao Wan, Chenxu Wu, Cong Cao, Fangfang Yuan, Yanbing Liu, Ziheng Chen, Pei Fu, Radu Timofte, Zongwei Wu, Nicu Sebe, Barnabas Poczos, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2610.38568)  

**Abstract**: Recursive self-improvement requires models not only to acquire useful training experience, but also to decide when their current experience has ceased to be useful. Existing methods often focus on what experience to generate next, while paying less attention to when fresh experience should be introduced. Many also construct training targets through self-consistency or majority voting, making supervision depend on the model's own uncertain predictions. We address these two questions separately: training dynamics determine when to acquire new experience, while the answers to new tasks remain independently checkable. A-RSI monitors reward variation and accuracy in training logs to refresh a task pool when its learning signal declines, without consulting validation or test scores. A static library of model-authored programs produces multimodal questions and derives their reference answers; we specify the task-generation contract, automatically verify chosen properties of the reward and refresh rules, and separately audit the generated answers. The training loop uses no external training examples or human-written answer labels. On the primary eleven-benchmark suite, A-RSI raises the performance average from 57.77 to 61.05 with 800 fresh tasks. 

---
# Co-Evolving Policy and Rubric via Capability Awareness for Open-Ended Generation 

**Authors**: Chengsheng Zhang, Wenshen Xu, Yuchuan Wu, Zhiyuan Yu, Junfeng Fang, Tianlong Wang, Bingxue Qiu, Yongbin Li, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.38566)  

**Abstract**: Rubric-based reinforcement learning (RL) offers interpretable, fine-grained supervision for aligning large language models (LLMs) on open-ended generation tasks. However, static rubric criteria may become saturated or remain beyond the policy's exploration, rendering the reward signals insufficiently discriminative. While recent efforts seek to guide sampling exploration or evolve rubrics, they fail to explicitly factor in how rubric difficulty should track the evolving policy's capabilities, resulting in a persistent mismatch between the optimization objective and the policy's current capability frontier. In this paper, we propose \textbf{Co}-Evolving Policy and Rubric via \textbf{C}apability \textbf{A}wareness (\textbf{CoCA}), a closed-loop framework that jointly adapts the rubric and policy through criterion-level diagnosis. Specifically, CoCA contrasts query-only and rubric-guided rollouts to characterize each criterion relative to the current policy, recalibrates criteria with mismatched difficulty, and identifies guidance-revealed capability gaps for frontier-based learning. Based on this insight, CoCA either expands rubric-conditioned frontiers through guided optimization or transfers qualified gains through on-policy self-distillation. As the policy updates, CoCA improves the capability diagnosis and enables more reliable distillation, thereby completing the co-evolution loop. Extensive experiments across diverse benchmarks and LLMs demonstrate that CoCA consistently improves performance over rubric-based RL methods. Moreover, CoCA promotes the internalization of guidance-revealed capabilities into query-only behavior while mitigating capability degradation. 

---
# PT$^3$: Prototype-Guided Self-Supervised Pretraining for Generalizable mmWave Human Point Cloud Representation 

**Authors**: Ziyang Liu, Kieren Yu, Xiao Yang, Xin Yang, Kaishun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.38551)  

**Abstract**: Multi-source millimeter-wave (mmWave) observations offer an opportunity to learn human representations reusable across sensing tasks. However, sparse detections and multipath effects make these observations unstable, challenging existing point cloud pretraining methods. We propose PT$^3$, a prototype-guided self-supervised pretraining framework for mmWave human point clouds. PT$^3$ represents local states in spatial regions as sparse compositions of shared prototypes and learns their temporal evolution through autoregressive prediction. Observation consistency further stabilizes representations under radar perturbations. After pretraining on unlabeled sequences from multiple sources, the same frozen encoder supports action classification, dense point generation, joint pose estimation, and future motion prediction on datasets not used for pretraining. Across these tasks, PT$^3$ outperforms the compared pretraining baselines and achieves competitive or superior performance to task-specific models. 

---
# An Alignment Benchmark for Enterprise Agents on Era by Eon 

**Authors**: Hai Sadon, Benjamin Gruenbaum, Doron Porat, Assaf Natanzon, Roy Zavida, Chen Dinachi, Or Itzahary  

**Link**: [PDF](https://arxiv.org/pdf/2610.38544)  

**Abstract**: AI agents are widely used in enterprises to access business systems and act on them in pursuit of a goal. However, fulfilling a user's request can cause harm. Era by Eon is a product that creates a digital twin of a company by simulating its business systems and data. Its simulators implement the programming interfaces of real products. Our benchmark uses ten of these systems, including Salesforce, Zendesk, Jira and Slack. We tested whether agents would carry out harmful business requests or trust misleading information in company records. We ran 19 models on 41 tasks three times, for 2,337 runs. Of the 2,304 runs that produced an answer, 918 (40%) contained a violation. The most common were false claims about actions or records (290 runs), following planted instructions (266) and disclosing sensitive data to the wrong audience (249). Every attempted change was recorded and blocked before it took effect. Misleading Slack messages caused 16 models to report an incorrect ticket status in 47 of 57 runs. No model was free of violations. The four best models violated in 22 to 24% of their runs. Yet one of them, Nova Pro, did the task correctly in only 30 runs. 

---
# From Leniency to Suspicion: How an Open Post-Training Recipe Changes Whom LLM Fact-Checkers Believe 

**Authors**: Nouar AlDahoul, Yasir Zaki  

**Link**: [PDF](https://arxiv.org/pdf/2610.38532)  

**Abstract**: LLMs increasingly fact-check online content with the poster's identity in view. However, it remains unclear which post-training changes how much a model's truth judgement depends on the poster's social class, and how such a change happens inside the model. This paper investigates job-based doubt, i.e., judging a claim false more often when the poster has a lower-status job, using fact-checked claims with randomised poster profiles in base and post-trained checkpoints of six model families. Our results reveal that: 1) post-training creates such doubt only with the open Tülu-3 recipe (OLMo-2 32B, Tülu-3 70B), not with Meta's, Alibaba's or Mistral's; the effect is under one percentage point but statistically reliable, exceeding training drift and replicating on independent claim sets; 2) in OLMo-2 32B, the model does not start noticing the poster's job more, but starts interpreting it differently: a lower-status job pushes the base model toward "true" and the post-trained model toward "false"; and 3) this flip is carried by a single residual-stream direction and happens at supervised fine-tuning. The mechanism was confirmed on 400 new claims with pre-specified predictions. None of seven deployed commercial models doubts lower-status posters more. Our work underscores the need to audit speaker bias per training recipe. 

---
# Value Probes: Learned Value Functions for Beam-Search Red Teaming 

**Authors**: Jonathan F. Jao, Andrew Kyle Lampinen  

**Link**: [PDF](https://arxiv.org/pdf/2610.38526)  

**Abstract**: Automated red-team auditing of language models is an important way to evaluate safety-relevant behaviors. These audits implicitly search a tree of possible conversation pathways: at each turn an auditor model decides which conversation to continue, and an LLM judge scores the dialogue. We emphasize that this auditing procedure is a Markov decision process, where states are the conversation history, actions are candidate responses, and rewards are judge scores for the target behavior. By adopting this perspective, we introduce two major improvements to auditing. First, we propose adopting beam search for the audit rather than relying on the auditor alone. We then interpret the problem of finding states that are likely to lead to problematic behavior as value estimation. Surprisingly, we show that this value function can be estimated by linear probing from the target model's activations to predict discounted future rewards - a method we term "Value Probes." By using Value Probes as a heuristic in our beam-search audit, we are able to substantially improve auditing success. Across 18 target behaviors and 3 scales of models, our approach raises peak audit scores by roughly 50% compared to (token-matched) standard auditing approaches. Furthermore, we show that Value Probes can be fit using off-policy data from smaller models. We also ablate individual components, and show that our method offers a comparably-performant and more efficient alternative to more expensive language model follow-ups. We hope these methods will improve language model auditing and lead to safer models. 

---
# Accelerating Nonlinear MPC by Behavior Cloning with Separated Timescales: Application to Nuclear Reactor Load-Following 

**Authors**: Perceval Beja-Battais, Alain Grosset{ê}te, Nicolas Vayatis  

**Link**: [PDF](https://arxiv.org/pdf/2610.38496)  

**Abstract**: Behavior Cloning (BC) of expert controllers for multiscale dynamical systems is challenging: state variables evolving at widely separated timescales induce highly nonlinear couplings that a shared latent space may fail to adequately represent, degrading closed-loop generalization. We propose Structured Representation Learning for Behavior Cloning (SRLBC), an architecture designed to leverage the physical knowledge of multiscale systems in which slow state variables, fast (stiff) state variables, and predictable exogenous inputs evolve at different timescales. On 500 long-horizon roll-outs of a pressurized water reactor (PWR) load-following benchmark -a high-dimensional, stiff industrial system with challenging safety constraints -and across 20 training seeds, separated embeddings systematically outperform shared embedding baselines on cumulative cost, imitation accuracy and constraint satisfaction. The resulting standalone policies approach NMPC closed-loop performance at a computational cost 15 times lower on average. Since standalone policies still leave a few percent of trajectories infeasible, we embed the learned policies in a safety-augmented approximate NMPC framework, in which the optimizer is only called when the predicted input sequence is not certified as safe. This yields controllers still 6.8 times faster than the NMPC controller on average, with a feasibility (i.e., constraint satisfaction) of at least 99.8% for all architectures, SRLBC retaining the lowest closed-loop cost. Under distribution shift, standalone BC policies degrade sharply: SRLBC degrades the least, and once safety-augmented, it retains a feasibility above 98% with the lowest closed-loop cost. Beyond the quantitative results, Jacobian-based sensitivity analysis shows that the resulting controller relies on a sparse and less redundant set of latent features, each attributable to a single group of physical variables, and aligned with the physical timescale decomposition of the system. 

---
# DES-JEPA: Learning a Shared Vocabulary for High-Dimensional Discrete Event Sequences 

**Authors**: Hugo Math, Maximilian Forstenhäusler, Christos Anagnostopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2610.38445)  

**Abstract**: Temporal foundation models such as Chronos and TimesFM generalize across domains through a shared numerical observation space. In contrast, discrete event sequences (DESs), including operational event logs or patient clinical-event trajectories, have domain-specific event vocabularies whose labels lack a consistent correspondence across domains. Beyond this lack of consistency, the number of distinct events in real-world settings is high enough to challenge both classical temporal point process modeling and current in-context foundation inference models such as FIM-PP. Critically, current foundation models for DESs are limited to a small, fixed number of event types, restricting their applicability to realistic, high-dimensional event streams. To address this, we propose two complementary modules: (i) DES-JEPA, a joint-embedding predictive architecture that learns a shared discrete vocabulary from synthetic stochastic processes with unaligned event vocabularies; and (ii) DES-PFN, a discrete event sequence prior-fitted network that reuses the learned vocabulary for in-context next-event prediction. Without task-specific fine-tuning, this framework achieves competitive zero-shot performance on eight unseen real-world datasets and scales up to 700 distinct event types. 

---
# Predict the Reward, Not the World: World Reward Models for ML Agents 

**Authors**: Yuxiao Lee, Zherui Yang, Fan Liu, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.38439)  

**Abstract**: Search-based machine-learning (ML) agents improve a solution by writing candidate programs, executing them, and keeping the best. Execution gives the search reliable feedback, but a single run can take hours, and most runs end in programs the agent discards. Recent agents try to save these runs by letting a large language model act as a world model that predicts how each program would perform. However, a wrong prediction feeds the search false information. We argue that an agent does not need to predict the outcome of execution, only the decision that execution would support. We propose World Reward Models (WRMs), which learn from real execution results to choose directly which candidate will perform better, without predicting any intermediate outcome. We prove that a selector which predicts execution from the same inputs cannot outperform the best direct selector, that predicted outcomes add an error which can make a likelihood-trained world model prefer the worse program on every instance, and that cross-entropy on executed winners drives a WRM toward the optimum. Our WRM, ForeCritic, adapts a 9B typed decision model to 5k executed program pairs. On 12,438 held-out pairs from 26 ML tasks, it picks the better program 79.4% of the time, while general-purpose LLM judges reach 53--62%, and its probabilities are calibrated. Inside AIDE, ML-Master, and ForeAgent, it cuts mean task time by 4.4--7.5 times while keeping their mean scores within 0.01 or raising them. 

---
# Governance-Aware Federated Learning for Multi-Agent AI Systems: A Policy-Driven Framework for Trustworthy Collaborative AI 

**Authors**: Kiranmayee Janardhan, Karthick S, Madhusmita Panda, Bharathi Vokkaliga Ganesh, Syed Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2610.38411)  

**Abstract**: Federated Learning (FL) enables collaborative model training without exchanging raw data, but conventional FL provides limited runtime governance, evidence verification, fairness assessment, explainability monitoring, and auditability. This paper proposes Governance-Aware Federated Learning (GAFL), which combines independent governance-evidence verification, temporal trust, fairness and explanation assessment, anomaly detection, policy enforcement, normalized aggregation, and tamper-evident audit logging. Client-reported governance values are authenticated, bound to their submitted model updates, and independently recomputed before aggregation. A cross-silo prototype with 10 participants was evaluated on the Adult Income dataset under honest, benign non-IID, biased-data, malicious-update, non-compliant, falsified-evidence, and mixed-threat scenarios. Four methods were compared across five random seeds: FedAvg, FedAvg with anomaly filtering, GAFL without independent verification, and complete GAFL. Complete GAFL intervened on all affected participants at the final round across the evaluated risk scenarios, including identifying 100% of falsified-evidence participants, whereas GAFL without verification identified none of the falsified-evidence participants. Final macro-F1 ranged from 0.7520 to 0.7733 across GAFL scenarios. Complete GAFL required 173.67 plus or minus 26.94 ms of governance processing per round, compared with 3.82 plus or minus 0.79 ms for FedAvg. A strict demographic-parity threshold also caused frequent interventions for honest participants, demonstrating a significant fairness-policy and utility trade-off. 

---
# Agent Transfer Protocol: Server-Mediated Agent-to-Agent Communication Architecture for the Agentic Web 

**Authors**: Yuqi Qiu, Zuyao Xu, Lu Sun, Xiang Li, Fasheng Miao, Fubin Wu, Haixin Duan, Zheli Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.38398)  

**Abstract**: The Agentic Web needs to support continuous collaboration among agents from different organizations. These collaborative relationships may span multiple tasks, network connections and execution instances. Such interactions require stable communication entries, message reception when instances are offline, and source verification after delayed reading. How to organize these responsibilities so that cross domain communication does not depend on specific execution instances being continuously online is a key issue for building long term agent interaction infrastructure. We propose ATP (Agent Transfer Protocol), a cross organization agent communication protocol adopting a domain mediated architecture. In this architecture, administrative domains maintain stable addresses and receiving entries for logical agents. The domain servers of both parties continuously undertake cross domain transmission. The receiving domain verifies and saves messages for execution instances to read later. The sending domain signs the complete message envelope. This allows the receiving instance to independently reverify the domain source and content integrity after the original connection ends, provided the original verification public key is still trusted and obtainable. We implemented an ATP prototype and evaluated it through a controlled cross organization LLM agent workflow, mechanism checks and latency measurements. The workflow completed subsequent business processing after instances went offline or were replaced. The independent verifier successfully verified the domain source and content integrity of saved messages. Latency measurements further quantified the communication and processing overhead introduced by the continuous mediation of domain servers. 

---
# Masked Erosion: How Acceptance Gates Lose Redundant Safeguards in Self-Improving Agents 

**Authors**: Hau-Sing So, Daixian Liu, Weicai Long  

**Link**: [PDF](https://arxiv.org/pdf/2610.38392)  

**Abstract**: Self-improving LLM agents edit their own harness, the prompts and rules around a fixed model, and keep an edit only if an acceptance gate scores it higher on a batch of tasks. Such a gate sees what the harness does under the current model, not what it would still prevent if one safeguard failed. When safeguards overlap, deleting one is masked: the others still block the harm, so the gate decides on task success alone and redundancy erodes unseen. We measure harm for every subset of four written safeguards in two agent environments and run 411 improvement loops under ten gate rules, mostly as controlled deletion experiments, plus three runs of the GEPA optimizer. A deletion behind another working safeguard adds almost no observable harm. A paired significance test blocks deletions that pass by chance but not deletions that genuinely raise task success, and a larger evaluation batch lets it delete cheaper safeguards: at the same cost, 20-task batches kept every harness protected while 40-task batches lost all six. Large harm audits prevent exposure but not the loss of depth, which turns into harm when the harness is reused with a weaker model. GEPA kept every safety rule verbatim, yet two of its three final harnesses misused credentials under a weaker model. A statistical model frozen before the runs predicts where each gate's protection breaks as safeguard cost and batch size change, and freezing safety text or auditing one-layer ablations preserves depth. 

---
# SCOPE: Selective Critical-Step On-Policy Self-Distillation for Agentic Reinforcement Learning 

**Authors**: Qirui Liu, Wei Cao, Yu Mi, Yichen Sun, Xuekun Zhang, Lei Liu, Yue Shen, Zhixuan Chu  

**Link**: [PDF](https://arxiv.org/pdf/2610.38391)  

**Abstract**: Outcome-based reinforcement learning provides coarse supervision for long-horizon LLM agents: failed trajectories may contain correct steps, while successful trajectories can contain errors. Trajectory-level skills can also shift teacher preferences at unrelated steps, introducing off-target distillation signals. We introduce SCOPE, a framework for Selective Critical-Step On-Policy Self-Distillation. After a supervised warm-up, the policy analyzes its own trajectories, generates step-specific hindsight skills, and distills their guidance only at selected error steps. A bounded signed teacher-student coefficient stabilizes auxiliary updates while preserving positive reinforcement and negative correction. A success-relative pairwise length reward further encourages efficient task completion. Experiments on ALFWorld and WebShop demonstrate consistent improvements across multiple model backbones. With Qwen2.5-3B, SCOPE achieves 97.7% success on ALFWorld and 85.2% on WebShop, outperforming SEED by 5.9 and 6.3 percentage points, respectively. On unseen ALFWorld tasks, SCOPE reaches 89.6% success, improving over GRPO by 23.2 percentage points with 46.7% fewer interactions. Scaling to Qwen3-32B, SCOPE achieves 99.2% success on seen ALFWorld tasks and 94.0% on unseen tasks. 

---
# Compute Has Shape: Self-Morphing the Internal Computation of Frozen LLMs 

**Authors**: Junxi Wu, Yiwen Guo, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2610.38382)  

**Abstract**: Test-time scaling improves large language models through additional computation, but how to adapt the topology of internal computation to individual inputs remains underexplored. Existing approaches largely fix recurrent or parallel architectures during training, or reconfigure frozen models along a single structural axis. We introduce SelfMorph, which makes computational shape an input-dependent inference-time decision. It unifies the original forward path (BASE), serial reuse of a layer span (REPEAT), and local branching and merging of perturbed hidden states (BRANCH) in a unified space parameterized by type, location, and intensity. Since they offer complementary benefits rather than uniform dominance, we use a lightweight selector that reads the backbone's depth-wise hidden state trajectory on the prompt to choose a morphology before response generation. We train the selector by distilling utility-based distribution from offline morphing that jointly account for task performance and computational cost. Evaluated on three LLMs spanning 4B-27B, SelfMorph achieves a 8.06% mean relative improvement on five in-distribution benchmarks and 4.45% on four unseen benchmarks. It achieves the strongest performance in 24 of 27 settings. These findings identify input-adaptive computational topology as a complementary axis of test-time scaling beyond generating more tokens. Our code is available at this https URL. 

---
# Encode Once, Answer Many: Internalizing Context via Parametric Flow 

**Authors**: Xinhao Zhong, Mengxuan Wu, Chonghua Liao, Linxi Xie, Ruicheng Feng, Yanpei Guo, Hao Tan, Jiaxin Hong, Jason Li, Panpan Zhang, Ziheng Qin, Qinglin Lu, Yuzhang Shang, Tongliang Liu, Damian Borth, Haggai Maron, Yongxiang Hua, Zhixiang Wei, Jian Hu, Yunfei Wu, Xing Sun, Xin Li, Haoyu Cao, Zhangyang Wang, Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.38356)  

**Abstract**: Large language models are typically adapted to a task either by placing knowledge in the input context, which keeps it external and re-encodes it on every call, or by fine-tuning, which internalizes it but demands a costly round of optimization per task. We aim to internalize the knowledge carried by a context directly into model parameters, without per-task optimization and without keeping the context in the input at inference time. To this end, we propose CLIK, a context-conditioned parameter generator that maps a given context to LoRA parameters for internalizing knowledge in a single forward pass through a Transformer encoder with cross-attention, requiring no test-time gradient updates. Once the generated LoRA is inserted into a frozen base model, the model answers downstream queries from the query alone, having internalized the context into its weights rather than re-reading it. Because the adapter depends only on the context, it is generated once and reused for every query on the same document, so the cost of reading that document is amortized across queries. On document question answering, CLIK substantially improves over the no-context base model and outperforms existing hypernetwork-based generators, while remaining behind a base model that reads the context directly, and it carries over to multi-turn dialogue without any multi-turn-specific training. Analysis further shows that the context-dependent part of the generated adapter concentrates in a compact and specific substructure. 

---
# Investor's Last Exam: Can AI Agents Do Equity Investment Without Seeing the Future? 

**Authors**: Huangfei Jiang, Anning Hu, Kai Sheng, Yuchen Cai, Rui Ge, Xiangyu Kong, Jianghao Lin, Shuzi Chen, Jiyi Weng, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2610.38342)  

**Abstract**: Financial agents must form forecasts and investment judgments from decision-time evidence, with predictions evaluated against subsequent outcomes. Existing benchmarks assess financial knowledge, research workflows, or forecasting, but evaluating the full research process requires a strictly frozen information set and professional evaluation criteria. We introduce Investor's Last Exam (ILE), an equity-research benchmark led by experienced buy-side investors and fund managers. Its three components are a versioned financial database spanning 3.43 million document versions and nine source categories, supplemented by structured market data, with validated timestamps governing retrieval and computation; a task taxonomy organized around professional analyst deliverables; and an expert-led framework linking scoring criteria to sources, calculations, and assumptions through financial dependency graphs. Evaluation spans 86 assignments and jointly assesses research quality, forecast accuracy, and simulated investment performance. GPT-6 Astra and Claude Opus 5.5 lead overall at 68.1 and 66.9 out of 100, yet even Astra scores only 52.5 on Forecast & Model. Our results show that even state-of-the-art models fail to translate overall research capability into predictive performance. ILE shifts the evaluation unit from isolated capabilities to the evidence--assumption--judgment chain, allowing researchers to locate where analytical advantages fail to translate into forecasts and investment judgments, and providing a traceable, diagnostic foundation for training and evaluating financial agents. 

---
# Can LLMs Faithfully Solve Ontology Engineering Tasks? A Mechanistic Interpretability Study 

**Authors**: Bohui Zhang, Albert Meroño-Peñuela, Elena Simperl  

**Link**: [PDF](https://arxiv.org/pdf/2610.38326)  

**Abstract**: Ontological knowledge underpins large language model (LLM) performance on knowledge-intensive tasks. However, existing evaluations of LLMs' ontological understanding largely rely on behavioral correctness, leaving the underlying mechanisms unexplained. Existing mechanistic interpretability work has largely focused on instance-level factual knowledge, leaving ontological knowledge under-explored. To bridge this gap, we introduce MIOE, a causal-abstraction benchmark that evaluates interchange interventions for two ontology engineering tasks: entity typing and subsumption prediction. MIOE tests whether learned internal intervention targets can change an ontology-level target answer while preserving class-specific auxiliary information. Across Qwen3-8B and Qwen3-4B, we evaluate SAE-, DBM-, and DAS-based methods, including multi-task and multi-layer variants. Distributed subspace methods give the strongest results and generalize across prompt settings, but the induced changes remain only partially separated from auxiliary class-specific information. Multi-task objectives and contiguous layer-window interventions improve this balance, while class-level analyses reveal substantial variation across ontology classes. These results suggest that ontological predictions are mechanistically accessible in LLMs, although current methods do not yet isolate a cleanly disentangled causal mechanism for ontological knowledge. 

---
# CHORD: Cross-Model Hallucination Detection via Relational Graph Discrimination 

**Authors**: Yongxin Deng, Zhen Fang, Guansong Pang, Sharon Li, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.38314)  

**Abstract**: Hallucination detection is critical for deploying large language models (LLMs) in real-world applications. Due to strong empirical performance, internal representation-based methods have emerged as the prevailing direction for detecting hallucinations, yet they remain largely model-specific and often fail to generalize to unseen LLMs. In this paper, we study an important yet underexplored problem, termed cross-model hallucination detection (CMHD), which aims to train hallucination detectors on source LLMs while ensuring performance on unseen target LLMs. The core challenge of CMHD lies in cross-model representation heterogeneity: hidden states from different LLMs exhibit severe semantic inconsistency, which limits the transferability of classical representation-based detectors. Through analysis and empirical validation, we show that relational structures over layers, tokens, and features capture transferable detection signals despite semantic inconsistency. Based on this observation, we propose cross-model hallucination detection via heterogeneity-oriented relational discrimination (CHORD), a relational graph discrimination framework that represents hidden states as joint relational graphs over layers, tokens, and features. It feeds these graphs into relation-aware attention to obtain structure-aware features, and uses meta-learning to shape these features toward transferable detection signals. Extensive experiments show that CHORD outperforms representative detection methods in cross-model generalization. Code is available at this https URL. 

---
# Game-Theoretic Foundations for Undiscounted Bisimulation Metrics 

**Authors**: Matthew Zurek, Yudong Chen, Josiah P. Hanna  

**Link**: [PDF](https://arxiv.org/pdf/2610.38313)  

**Abstract**: Bisimulation metrics offer a principled notion of task-relevant state similarity in reinforcement learning, but existing constructions and methods are largely confined to discounted MDPs, where contraction arguments ensure well-defined fixed points and downstream guarantees. We develop a unified foundation for bisimulation metrics in undiscounted settings, that is total- and average-reward problems, by replacing the standard operator-centric view with a game-centric one: we interpret prior metric operators as Bellman-Shapley operators of zero-sum turn-based games and define metrics as the corresponding game value functions, yielding unique definitions even when $\gamma=1$. This game-theoretic foundation facilitates transfer of dynamic-programming theory and algorithms. We also use this game-centric perspective to characterize how metric finiteness depends on structural properties of the underlying MDP and motivate certain directed metrics that remain finite in weakly communicating MDPs. We further introduce an abstract framework for modifying the games' action spaces, allowing general transition coupling sets, that recovers many existing metrics and guides new constructions. Finally, we prove that these metrics bound total-reward and, when the optimal gain is state-independent, optimal relative value differences. We also provide error and suboptimality guarantees for aggregation-based planning in total- and average-reward settings. 

---
# Inner-Learning: From Hand-Coded Beliefs to Declarative Knowledge in Trick-Taking Games 

**Authors**: Stefan Edelkamp  

**Link**: [PDF](https://arxiv.org/pdf/2610.38304)  

**Abstract**: We present a declarative approach to belief-state modeling in trick-taking card games. Starting from a state-of-the-art Skat player, we automatically extract its hand-coded inference rules into a domain-specific language built on the primitive $\neg\mathsf{holds}(P,C)$, denoting that player $P$ cannot hold any card in set $C$. Round-trip code generation verifies that the extracted rules reproduce the original engine's intended behavior, up to three audited bugs in the original that the declarative engine corrects. We then grow the rule base through symmetric derivation and annotation of an expert strategy document---unifying code, symmetry, and expertise in one representation. We extend the knowledge representation with \emph{cardinality constraints}: at-most-$k$ and at-least-$k$ bounds on the cards a player holds from a named group, expressed in the same declarative language and enforced at world-generation time. Confidence parameters connect symbolic rules to probabilistic belief updates. Self-play evaluation shows the declarative engine measurably improving winning percentage and score over the original hand-coded engine, with the extracted rules, the human-annotated rules, and the cardinality constraints each contributing to the effect. 

---
# Verification-Guided Retrosynthesis: Reaction Feedback and Evidence Timing 

**Authors**: Hongxuan Shi, Hankz Hankui Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2610.38245)  

**Abstract**: Retrosynthesis planning seeks synthesis routes from purchasable starting materials to a target molecule. Search over predicted reactions can connect a target to available materials, yet an infeasible reaction can invalidate the resulting route. Evaluating route quality only after search has finished cannot correct errors made during search. We introduce Verification-Guided Reflow (VGR), incorporating reaction verification as a decision action in AND/OR search. Verification results are represented as binary verdicts; queries are scheduled from the search structure, and acquired verdicts exclude reactions or adjust search priorities. We define verified solvability as the fraction of targets for which a complete route accepted stepwise by the verifier is found. On Retro190, feeding verification results into search increases verified solvability from 51.1% to 62.6% relative to route evaluation without feedback. Further controlled experiments show that earlier verification and immediate prioritization of accepted reactions each improve route discovery, yet combining them performs worse than either change alone. We therefore propose early pruning and delayed guidance: rejected reactions are pruned immediately, while accepted reactions gain search priority when processed as part of a complete candidate route. In a separate comparison with the verification scheduling rule fixed, this strategy increases verified solvability from 61.1% with immediate guidance to 71.1%, using fewer proposal calls and unique verifier queries. 

---
# FACTS-MAS: A Multi-Agent Architecture for Adaptive Forecasting from Heterogeneous Evidence 

**Authors**: Madumita Karthikeyan, Umang Rajnikant Bid, Rong Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.38225)  

**Abstract**: Real-world forecasting often requires integrating heterogeneous evidence whose predictive value varies across horizons, operating regimes, and domains. We introduce FACTS-MAS, a domain-agnostic multi-agent forecasting framework that decomposes evidence into specialized channels for temporal dynamics, exogenous context, seasonality, structural information, and discrete events. Each agent generates an independently testable forecast through a common interface, while leakage-safe historical validation measures its predictive skill relative to a naive reference. These skill estimates are used to weight and combine agent forecasts, so that uninformative evidence is either removed or downweighted rather than forced to participate. A calibration procedure selects domain-specific constant means based on historical validation folds. We evaluate FACTS-MAS on weekly for-sale housing inventory from Zillow and weekly unemployment insurance claims from FRED, for 15 U.S. metropolitan areas at 4-, 8-, and 13-week forecasting horizons. Our method outperforms both a naive last-value forecast method and the TimesFM foundation model. However, a factor-adjusted exponential smoothing model performs as well as the fused forecast. The evidence therefore supports FACTS-MAS as a systematic protocol for testing, calibrating, and recomposing evidence channels before assembling them into the strongest single model for a specific application domain. Code, data, and results are available at this https URL. 

---
# Rubricon: Automated Rubric Generation and Refinement for LLM Evaluation and Training 

**Authors**: Ulyana Tkachenko, Hui Wen Goh, Guram Gogia, Jonas Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2610.38220)  

**Abstract**: Rubrics enable reward assignment on tasks without verifiable answers, but are expensive to manually produce from scratch. We present Rubricon, a pipeline that generates discriminative rubrics without requiring human preference labels, rubric annotation, or generator-model training. Using any LLM, Rubricon samples multiple rubrics, selects the most discriminative candidate, adds criteria to capture overlooked requirements or response flaws, and prunes criteria that are misaligned or shallow. Alongside LLM self-critique and test-time scaling, Rubricon uses observed criterion pass rates on example model responses to guide refinement. Across five professional, consumer, clinical, and general-use LLM benchmarks, Rubricon produces rubrics whose scores of frontier LLM responses exhibit 9% greater alignment with labor-intensive expert-authored rubrics, compared to the best alternative synthetic rubric generation method among RRD, QSRG, OpenRubrics, and agentic generation via harnesses like Codex. On HealthBench, post-training a Qwen3.5-9B model with Rubricon rubrics yields 4.5% better performance than training with rubrics from the strongest alternative synthetic rubric generator, RRD. 

---
# Disparity Laundering: Consensus Without Fairness in Clinical Multi-Agent Systems 

**Authors**: Xiaoyang Wang, Tianrui Wang, Christopher C. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.38215)  

**Abstract**: Clinical large language models are increasingly fielded as multi-agent teams, on the assumption that deliberation averages out individual error. It does not average out bias. Across a full factorial of five model families and four aggregation topologies on 120 real MIMIC-IV emergency cases under counterfactual demographic swaps, no configuration neutralizes the disparity: all 20 cells commit the direction of their agents' independent answers, giving Black patients more acute triage. Every model has a topology that amplifies its baseline gap, but which one varies by model, so no design is reliably safe or reliably amplifying: the disparity is relocated, never removed. Late-stage fixes fail: fairness instructions and demographic masking leave the gap untouched, and per-agent self-debiasing widens it, because the bias is already encoded in the agents' ordinary clinical numbers. We call this disparity laundering; the audits now applied to clinical AI, per model, per output, per version, can certify the parts but never the committed decision. Interventions close the gap exactly when they make counterfactual versions visible inside the system and pool them, through deliberation, into the committed decision. A diverse-perspective ensemble does both while keeping the team, with no retraining or architectural change: the disparity falls by 67 to 94% where all five perspectives pool at every round, and the one topology that never pools them is the one it cannot reliably repair. Medical multi-agent systems do not filter demographic bias; they launder it into plausible decisions no single-version audit can catch, and the only defense is to make the bias visible. 

---
# GAAC: Lossless KV Compression for Bandwidth-Efficient LLM Inference on Personal Devices 

**Authors**: Yipin Guo, Arun M George, Siddharth Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2610.38210)  

**Abstract**: Weight-only, lossy quantization substantially reduces the storage and memory requirements for serving large language models (LLMs), but the growing key-value (KV) cache remains a bottleneck for long-context inference on personal devices. We present Group-Adaptive Additive Codec (GAAC), a lossless KV compression and serving system that exploits exponent redundancy to reduce storage and accelerate attention without introducing KV approximation. GAAC encodes BF16 exponents using calibrated per-channel base-plus-offset codes over contiguous ranges. GAAC adapts code widths across small groups while preserving KV values bit-for-bit. A fused attention kernel reconstructs KV tiles on the fly, avoiding full-cache decompression. Block-wise compression and fixed-capacity slots support incremental cache updates and paged serving. Across five tested models, GAAC achieves 1.405-1.407$\times$ lossless BF16 KV compression. We also study compression performance on low-precision tensors FP8 E4M3 caches ( 1.09-1.14$\times$) and E5M2 caches (1.25-1.30$\times$). On NVIDIA DGX Spark, GAAC accelerates decode attention at a 64K-token context by 1.19-1.27$\times$ and end-to-end W4A16 decode steps by 1.04-1.15$\times$. In serving, it reduces resident KV memory by approximately 26% and increases aggregate throughput by 13-29% under comparable KV-memory budgets. On Apple M4, GAAC achieves 1.02-1.25$\times$ attention speedups, demonstrating portability beyond NVIDIA GPUs. 

---
# From Natural Language to Real Schematics: An Automated Workflow for Power Electronics Schematic Synthesis 

**Authors**: Bowen Su, Huan Chen, Jiahua Ying, Ran Ou, Weilong Chen, Li Xue, Kai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.38201)  

**Abstract**: Designing power electronics converters traditionally requires intensive human expertise across topology sizing, commercial component selection, circuit simulation verification, and schematic drafting. While Large Language Models (LLMs) show promise in electronic design automation (EDA), applying standalone LLMs directly to board-level power electronics schematic synthesis suffers from severe engineering hallucinations, including inaccurate analytical sizing, fabricated or mismatched component part numbers, violated voltage derating margins, and broken topological connectivity. In this paper, we propose a Bounded Dual-Loop Automated Schematic Synthesis Framework that translates natural language specifications into verified, editable KiCad schematics and manufacturing deliverables. Our workflow integrates six stages across three functional columns: (1) requirement parsing and knowledge-driven power stage synthesis covering non-isolated (Buck, Boost) and isolated/resonant (Flyback, DAB, half-bridge LLC) topologies; (2) contract- and evidence-coordinated component selection grounded in distributor APIs and OEM datasheets, coupled with closed-loop ngspice transient verification and stress feedback; and (3) modular control hardware integration and multi-format deliverable generation featuring native KiCad netlist round-trip verification. End-to-end case studies on a 24-36V to 12V,2A Buck converter (29 positions, 83 verified terminals) and an isolated 48V to 24V, 0.5A, Dual-Active Bridge (DAB) converter (87 positions, 341 verified terminals), alongside comparative evaluations against a Vanilla LLM baseline, demonstrate that our framework improves component verification coverage from 22.2 to 98.6, nominal value consistency from 0 to 97.7, and SPICE simulation pass rates from 0 to 100. 

---
# Prototype-Guided Dynamic Domain Modulation and a Large-Scale Multi-SKU Benchmark for Shampoo Bottle Surface Defect Detection 

**Authors**: Yuehang Li, Xiao Wang, Qiang Chen, Yangyi Fan, Bo Jiang, Yaowei Wang, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2610.38194)  

**Abstract**: Industrial shampoo-bottle surface defect detection must operate across many stock keeping units (SKUs), where differences in bottle geometry, material, label layout, reflectance, and imaging conditions can substantially alter the appearance of the same defect. Existing industrial defect datasets rarely provide both instance-level defect annotations and product-variant identities, making it difficult to study SKU-aware detection under realistic manufacturing conditions. We introduce Bottle-SKU, a large-scale multi-SKU benchmark for shampoo bottle surface defect detection. Bottle-SKU contains 13,119 images, 20,419 instance-level bounding boxes, seven defect categories, and 48 bottle SKUs, enabling evaluation of both shared defect recognition and SKU-specific appearance variation. To address the challenge of SKU-specific appearance variation, we propose Prototype-Guided Dynamic Domain Modulation (ProDMM). ProDMM constructs a visual prototype for each SKU from training images and predicts a soft SKU distribution for each input. The resulting prototype-conditioned representation dynamically generates low-rank Query/Value modulation for the visual backbone and residual conditioning for defect-text-to-visual attention, while preserving the pretrained Grounding DINO detection pathway. The model is trained jointly with detection, SKU-domain classification, and modulation-regularization objectives. Experiments on Bottle-SKU and the public GC10-DET-SKU metallic surface defect dataset demonstrate the effectiveness of prototype-guided domain conditioning for industrial surface defect detection. 

---
# DecompGym: Benchmarking LLM Agents on Designing Decomposition Algorithms 

**Authors**: Sikai Cheng, Tinghan Ye, Arnaud Deza, Xiaonan Qi, Miao Li, Litong Liu, Changkun Guan  

**Link**: [PDF](https://arxiv.org/pdf/2610.38175)  

**Abstract**: Benders decomposition and column generation solve large optimization problems by passing cuts, columns, and dual values between a master problem and subproblems, and are hard to implement correctly. Large language models now automate parts of algorithm design, which points toward agents that build decomposition solvers for new problems. As a first step toward such agents, we study how well current coding agents design and implement these two algorithms, and how they fail. We introduce DecompGym, 81 instances from nine problem families (five Benders, four column generation), in which an agent writes a solver for a prescribed method. Each solver must meet three criteria: execution within a 60-second time limit; numerical feasibility together with a method audit, in which an LLM judge reads the solver's code; and an objective within 1% of the stored optimal objective. We evaluate 12 GPT, Claude, and Gemini models running in coding-agent tools. Over up to ten attempts, we compare independent regeneration with reflection, which first diagnoses the failure. On the first attempt, two models solve every instance and seven solve more than 90%. Most Benders failures (48 of 62) are missing or suboptimal solutions. The most common column-generation failure is a method-audit rejection. Output checks alone would accept 93 of the 108 attempts that fail the method audit. The most common rejection, an optimalif, occurs 96 times, 92 of them in one this http URL turns 66% of retries into successes, compared with 39% for regeneration, and it repairs method-audit failures, for which the judge names the defecective gaps. Agents for decomposition therefore need checks on how a solver reaches its answer and feedback that locates the error. The benchmark is available at this https URL. 

---
# BridgeRouter: Gain-Aware Instance Routing over Cross-Modal Bridge Portfolios 

**Authors**: Ruoxi Piao, Alicia Lu, Yunzhen Zhang, Mustafa Mısır, Hasan Onur Keleş  

**Link**: [PDF](https://arxiv.org/pdf/2610.38152)  

**Abstract**: Lightweight representation bridges make pretrained, task-adapted EEG foundation models executable on other biosignal modalities without retraining their backbones. However, selecting a single bridge pathway globally assigns every instance to the same transfer route and ignores complementary errors across routes. We therefore formulate multi-bridge transfer as an instance-level algorithm-selection problem and introduce BridgeRouter, a gain-aware router over a portfolio of frozen cross-modal pathways. A validation-selected Single Best Solver (SBS) serves as a protected default route. During training, task labels are used to compute the reduction in negative log-likelihood provided by each alternative route relative to the SBS, and a lightweight MLP predicts these route-specific gains from a target-modality representation extracted by a frozen encoder. At inference, BridgeRouter observes only the target modality, requires neither EEG, task labels, nor predictions from all candidate routes, and executes exactly one bridge pathway: the route with the largest predicted gain when that gain exceeds a validation-calibrated threshold, and the SBS otherwise. We evaluate seven EEG foundation-model bridge pathways for EMG-based freezing-of-gait detection on FOG and ECG-based sleep classification on ISRUC. Across five seeds, BridgeRouter achieves $81.38\pm1.11\%$ balanced accuracy on FOG, improving over the BIOT SBS by $11.49$ percentage points, and $62.92\pm2.17\%$ on ISRUC, improving over the CBraMod SBS by $3.83$ points. Per-instance bridge oracles reach $99.96\%$ and $78.61\%$, respectively, revealing substantial remaining portfolio headroom. These results show that anchor-relative gain prediction can convert complementary cross-modal bridge behavior into deployable instance-level routing improvements while retaining single-route inference. 

---
# SkillCT: A Multidimensional Benchmark and Execution-Free Baseline for Agent Skill Assessment 

**Authors**: Mengyu Sun, Zunlong Zhou, Ziyuan Yang, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.38147)  

**Abstract**: Agent skills package instructions, scripts, and resources for specialized tasks and are increasingly distributed through public Web repositories. However, existing skill benchmarks primarily focus on downstream task execution, insufficiently assessing specification quality, behavioral robustness, semantic alignment, and responsibility boundaries. Their reliance on task-specific tools, environments, and repeated executions limits scalability and pre-deployment assessment. To address these challenges, we introduce SkillCT, a benchmark that broadens skill assessment from downstream utility to a multidimensional characterization of skill quality. SkillCT evaluates skills across nine complementary dimensions spanning three aspects: Static Assessment, which captures intrinsic properties of the skill specification; Dynamic Assessment, which measures whether the induced behavior remains stable across backbone models and request variations; and Human-Assisted Assessment, which examines behavior-chain consistency and task granularity. These dimensions provide a broader view of skill quality than execution success. Applying SkillCT to a public skill corpus establishes a reference benchmark for analyzing quality weaknesses and relationships across dimensions. Building on this benchmark, we propose an execution-free quality evaluation baseline that predicts the multidimensional quality profiles of unseen skills by retrieving and leveraging similar reference skills, without executing the skills under evaluation. This reduces token consumption and evaluation time, while preserving consistency with the full assessment. Furthermore, SkillCT-guided optimization improves downstream task success and execution efficiency, demonstrating its value for skill refinement. These findings highlight the effectiveness of SkillCT in supporting scalable skill assessment, optimization, and repository maintenance. 

---
# Decompose, Don't Expand: Capability-Independent AI Systems by Construction 

**Authors**: Tejaswi Gowda  

**Link**: [PDF](https://arxiv.org/pdf/2610.38144)  

**Abstract**: A recurring design choice in building AI systems is whether, when a task is too hard for a model, to expand the model (make it larger, better-prompted, fine-tuned), or to decompose the task (extract the parts that admit a deterministic solution to a host, leaving the model only the irreducibly fuzzy residue). We make three claims. First, that many tasks commonly delegated whole to a model are in fact deterministic tasks with a small generative core. Second, and our central empirical result, that decomposing such a task yields a measurable and desirable property we call parity by construction: on the extracted sub-tasks the model-size performance curve becomes flat, because no model performs them at all. We measure this across nine models spanning a 494M-parameter local checkpoint to a frontier Opus-class model, a thirty-fold parameter range on the local family alone: once the host resolves the context, operation selection scores identically for all nine models and selector resolution for eight of nine, while the one task that does not decompose, labelling, remains sloped across the same range. Because the task leaves the model, a stock checkpoint suffices: no task-specific training and no fine-tuning. Third, that this is a general design discipline, not a domain trick: we show the same move recurring across physics setup, motion retargeting, and object-model editing, and in a Web-CLI tool family (same author, so not an independent replication). We give a method for locating the cut, the construction that makes a cut exist (a deterministic surface and a closed grammar over it), and a way to make the residue safe (resolve the deterministic context; verify the generative output), position the work against neuro-symbolic AI, tool-augmented language models, and generate-and-verify approaches, and are explicit about what is measured (one domain) versus illustrative (the rest). 

---
# Post-Training Large Language Models with LLMON 

**Authors**: Basel Shbita, Michael Hind, Chad DeLuca, Nathan Fulton, Shubhi Asthana, Farhan Ahmed, Anisa Halimi, Yada Zhu, Mauro Martino, David Cox, Dan Gutfreund  

**Link**: [PDF](https://arxiv.org/pdf/2610.38120)  

**Abstract**: Large Language Models (LLMs) treat all context tokens uniformly, lacking explicit separation between executable instructions and passive data. This leads to prompt injection vulnerabilities, ambiguous instruction binding, and unreliable execution control. We present a post-training methodology using LLMON (LLM Object Notation), a framework that enables separating instructions from data and requires explicit identifier-based execution. We construct a large-scale structured training corpus of 4.6B tokens spanning general, RAG (Retrieval-Augmented Generation), and tool-calling data, augmented with distractor-infused variants in which competing instructions and irrelevant data segments are present in context but not referenced. Post-training two Granite-4.1 base models (3B and 8B) through supervised fine-tuning followed by reinforcement learning, we show that LLMON-structured supervision reduces attack success rates on indirect prompt-injection benchmarks from 53-100% to 0.02-5.6% while preserving general capabilities. We further introduce attention-based interpretability metrics that diagnose failure modes at individual transformer layers for structured inputs, revealing how our approach reshapes internal attention patterns. Our results demonstrate that explicit execution semantics can be taught through standard post-training without architectural modification. 

---
# Too Hard, Too Easy, Just Right: Performance-Driven Challenge Adaptation in Virtual Reality Firefighter Training 

**Authors**: Xuan Wang, Jianan Jiang, Rongqian Chen, Tian Lan, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.38117)  

**Abstract**: Effective firefighter training requires practice under dynamic and stressful emergency conditions, yet real-world live-fire exercises are costly, risky, and difficult to scale. Virtual Reality (VR) offers a safe and controllable alternative, but many existing VR firefighter training systems rely on fixed scenarios or provide limited difficulty adaptation, making it difficult to maintain an appropriate level of challenge for skill development as trainees' performance changes over time. To address this limitation, we present ProACT, a performance-driven adaptive training system that regulates scenario difficulty through closed-loop interactions between trainee performance and simulated fire dynamics. Specifically, ProACT combines a global module that continuously adjusts scene-level fire spread dynamics in real time based on trainee performance with a local module that periodically updates object-level flammability across the environment. Our evaluation shows that ProACT improves both objective training performance and subjective user experience compared with existing adaptive and non-adaptive baselines. We further discuss how adaptive immersive environments can extend beyond firefighting to broader training domains. 

---
# AI-Driven Digital Twin Framework for Warehouse Logistics Automation Optimization 

**Authors**: Md Mahinur Alam, Kanita Jerin Tanha, Md Tanjin Alam, Shadman Sakib, Min Gyu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.38101)  

**Abstract**: In recent years, warehouse logistics optimization has increasingly focused on intelligent, data-driven systems to enhance operational efficiency. Conventional static product placement strategies are being progressively replaced by dynamic slotting approaches that can adapt to real-time variations in warehouse conditions. In this work, we present a novel framework that employs a reinforcement learning agent integrated within a FlexSim-based Digital Twin environment for real-time product placement optimization. The proposed agent continuously learns from streaming operational data, enabling it to reduce picking time while improving overall system throughput. The simulation environment models a four-zone warehouse layout with forklift-based material handling and leverages the Asset Administration Shell framework to ensure seamless interaction between the learning agent, physical asset, and the digital twin. Experimental evaluation demonstrates substantial performance gains, with the proposed approach achieving a 29.66\% reduction in average picking time, a 35.42\% increase in throughput, and an 18.54\% decrease in travel distance. 

---
# Evaluating Causal Reasoning in LLMs: A Case Study in the Challenges of Procedural Generation 

**Authors**: Justin Clarke, Isha Karn, Patrick Murray, KJ Hill, David Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2610.38100)  

**Abstract**: Large language models (LLMs) appear to represent and reason about a large range of phenomena. In particular, recent studies have examined whether LLMs can accurately reason about causation. One of the most commonly used benchmarks for evaluating formal causal reasoning in LLMs is CLadder, a framework for procedural generation of benchmark data sets that was released in 2023. Experiments with CLadder appear to indicate that LLMs can perform formal causal reasoning in a large percentage of generated examples. In this paper, we examine what can be reliably inferred when LLMs perform well on test sets produced by CLadder. We identify several examples of superficial and highly predictive patterns that the CLadder generator embeds in data sets it generates. We find that even simple models such as random forests perform well on CLadder-generated data sets after training on a sample of generated data. We also find that the performance of many commercial LLMs on CLadder test sets is extremely brittle and that removing large sections of the text that are critical for formal causal reasoning has almost no effect on performance. Given these results, the most likely hypothesis for the strong performance of state-of-the-art LLMs on CLadder is that they are exploiting superficial patterns learned from exposure to training data from the generator. Any inference that LLMs are exhibiting more complex causal reasoning must rule out this hypothesis. 

---
# PromptSense: Sentiment-Aware Mental Health Chatbot with Dynamic Prompting 

**Authors**: Sugi Lee, Jianlong Zhou, Jing sUN  

**Link**: [PDF](https://arxiv.org/pdf/2610.38044)  

**Abstract**: Current Large Language Models (LLMs)-based mental health chatbots often lack sufficient empathy, contextual awareness, and safety mechanisms, potentially leading to ineffective or harmful interactions. This paper proposes a novel framework that integrates sentiment and depression analysis with a rule-based filtering mechanism prior to model inference. By dynamically constructing prompts based on emotional and risk-related insights, the framework aims to significantly improve the contextual relevance, empathy, and risk-aware of model-generated responses. We evaluated four model strategies and two prompt engineered variants across qualitative metrics such as empathy, helpfulness, clarity, relevance, safety, as well as toxicity assessments using the Perspective API. Experimental results indicate that the prompt engineering approaches outperforms DSFT and RLAIF by 1.8\% and 1.1\% respectively. 

---
# Assessing Language Models for Scientific Applications: Inertial Confinement Fusion as a Use Case 

**Authors**: P. B. Radha, D. D. Meyerhofer, M. Bhattarai, B. Cheng, L. Diaz, C. L. Rousculp, A. G. Seaton, G. Gupta, P. Gurumurthy, V. Gutta, A. Jhunjhunwala, A. Vem, P. Cook, G. Rowe  

**Link**: [PDF](https://arxiv.org/pdf/2610.38024)  

**Abstract**: Advancing language models for specialized scientific applications requires evaluations of scientific fluency, physical reasoning, and quantitative analyses. Using inertial confinement fusion (ICF) as a test case, we have developed an expert-curated benchmark of 250 questions spanning qualitative reasoning, numerical calculation, symbolic derivation, and equation formulation. Responses have been evaluated using deterministic numerical and symbolic checks together with language-model judges. We compared a variety of out-of-the-box closed-source and open-source large language models with Prospero, a model specialized through combined domain-adaptive pretraining and supervised fine-tuning for ICF. We also conducted initial retrieval-assisted evaluations. Prospero improved its aggregate score by 10.7 percentage points over its base model, with gains across all question types. Frontier models achieved the highest aggregate scores, but critical physics errors persisted, particularly in numerical calculations. We note that domain-specific performance can improve through specialization, but benchmark gains alone do not establish scientific reliability. Retrieval improved performance for a smaller model but yielded no statistically significant improvement for a frontier model. Two scientific stress tests further illustrate useful capabilities and the need for expert assessment of physical assumptions and model validity. Based on these findings, we recommend evaluating existing domain capabilities and retrieval benefits before committing to substantial additional specialization. 

---
# Algorithmic Bias from the Perspectives of Healthcare Professionals 

**Authors**: Jennifer Xu, Tamara Babaian  

**Link**: [PDF](https://arxiv.org/pdf/2610.38006)  

**Abstract**: This paper focuses on algorithmic bias of machine learning and artificial intelligence applications in healthcare information systems. Based on the quantitative data and qualitative comments from a survey and interviews with healthcare professionals, who have different job roles (e.g., clinical vs. administrative), this study provides findings about the relationships between algorithmic bias, perceived fairness, and the intended acceptance and adoption of ML algorithms and algorithm generated outcomes. The results suggest that the opinions of healthcare professionals toward the causes of algorithmic bias, the criteria of algorithm assessment, the perceived fairness, and bias mitigation approaches may vary depending on their job roles, perspectives, tasks, and the algorithm characteristics. More research is needed to investigate algorithmic bias to ensure fairness and equality in healthcare. 

---
# GAUSS: A Workload-Aware Analytical and Monte-Carlo Simulator for LLM Inference Serving 

**Authors**: Srikant Bharadwaj, Ankur Mallick, Spyridon Mastorakis, Anjaly Parayil, Dhruv Deshmukh, Victor Ruehle  

**Link**: [PDF](https://arxiv.org/pdf/2610.37989)  

**Abstract**: Large language model (LLM) inference has become an operational workload that fleets must provision, monitor, and bill against latency service-level objectives (SLOs). A single throughput number cannot answer the questions operators actually face, because generative serving couples two phases with very different service characteristics, continuous batching reshapes the effective service process, and user-visible tail latency can degrade long before aggregate throughput saturates. We present GAUSS, a workload-aware analytical and Monte-Carlo simulator that predicts the full distributions of time-to-first-token (TTFT) and time-between-tokens (TBT), together with the sustainable request rate, for a given workload, engine, and accelerator. GAUSS pairs calibrated, phase-specific latency surfaces for prefill and decode with a Lindley-recurrence model of queueing under stochastic arrivals and estimates performance by ensemble Monte-Carlo sampling rather than by replaying a single trace. This lets GAUSS reason about workloads, routing policies, and hardware that have never been deployed, and to capture the latency tails that trace-driven simulators systematically under-sample. We describe the model and show how it supports operating-point selection, traffic segmentation, and shortest-remaining-processing-time scheduling capacity questions that trace-driven simulators cannot address. 

---
# SkillMem: Compact External Skill Memory for Latent Multi-Agent Reasoning 

**Authors**: Yuxuan Zhu, Mohammad Mohammadi Amiri, Pin-Yu Chen, Keerthiram Murugesan  

**Link**: [PDF](https://arxiv.org/pdf/2610.37935)  

**Abstract**: Latent multi-agent systems reduce inference cost by keeping intermediate reasoning and inter-agent communication in continuous representations.\ Reusable skills, however, are commonly supplied as text or stored as layer-wise Key-Value states or skill-specific model parameters, introducing repeated processing or substantial per-skill storage.\ We introduce \textbf{SkillMem}, which represents each reusable skill as a small external latent memory and makes it directly available during latent multi-agent reasoning.\ A role-conditioned cross-attention reader allows the same memory to guide different agents according to their roles and current hidden states.\ We train the skill memories and shared reader through behavioral distillation, matching the change in token predictions induced by a textual skill while keeping the backbone frozen.\ For our experiments, we obtain textual skills from paired successful and failed multi-agent trajectories, although existing skills can also be compiled directly.\ Across six benchmarks and Qwen3 models ranging from 4B to 14B parameters, SkillMem provides strong gains for smaller backbones while preserving efficient latent multi-agent inference across model scales.\ We use the same $8\times256$ memory configuration across all evaluated model scales, requiring 9.6\,KB of skill-specific storage per skill, including its retrieval embedding.\ For a library of 100 skills, SkillMem reduces total storage by more than $38\times$ compared with alternative skill representations and achieves up to a $4.23\times$ end-to-end inference speedup.\ These results show that compact procedural memories can be reused without sacrificing the efficiency of latent multi-agent reasoning. 

---
# Understanding Large Language Models as Numerical Optimizers 

**Authors**: Lingyun Guo, Ali Bereyhi  

**Link**: [PDF](https://arxiv.org/pdf/2610.37932)  

**Abstract**: Large language models (LLMs) have emerged as potential tools for numerical optimization. Despite promising results, their intrinsic optimization capabilities and limitations remain poorly understood, particularly in high dimensions. This paper investigates the ability of LLMs to exploit mathematical problem structures and generate feasible, near-optimal solutions. We develop an evaluation framework that distinguishes solution quality from execution failures, including numerical inaccuracies, constraint violations, and output-format errors. Using representative problems from convex optimization, fractional programming, and constrained projection, we assess LLM performance against deterministic reference solvers across different dimensions. Our results reveal that LLMs can capture useful optimization structures and produce near-optimal solutions, even when their outputs fail to satisfy numerical or formatting requirements. However, reliability deteriorates with increasing problem dimension and depends strongly on the underlying optimization structure. To further investigate the limitations, we deploy an iterative refinement mechanism that provides targeted feedback on feasibility and solution quality. The results demonstrate that iterative refinement substantially improves reliability and extends the range of problem dimensions over which LLMs produce valid solutions. Nevertheless, persistent numerical and structural errors limit their applicability as standalone optimizers. These findings provide insights into the capacity and limits of LLM-based optimization and motivate hybrid approaches that combine LLM reasoning with deterministic numerical verification. 

---
# GARDEN: Generalist Autonomous Reward Design Through Evidence Navigation for Cooperative Multi-Agent Reinforcement Learning 

**Authors**: Dogan Urgun  

**Link**: [PDF](https://arxiv.org/pdf/2610.37926)  

**Abstract**: Cooperative multi-agent reinforcement learning (MARL) enables teams of agents, such as robots and autonomous vehicles, to learn coordinated behavior, but what such teams learn is largely determined by their reward function. Because task success alone provides too little guidance for learning, intermediate rewards are usually designed by hand, and this effort must often be repeated whenever the environment or the team changes. Large language models (LLMs) can automate the effort by writing reward functions as code, yet the resulting rewards are typically designed and tested in a single setting. To enable their reuse across settings, GARDEN (Generalist Autonomous Reward Design through Evidence Navigation) is proposed, in which one reward program is developed for a family of related scenarios and transferred, unchanged, to scenarios that were not available during its design. During development, environment code, training records, and agent behavior are investigated by an LLM reward engineer, and draft rewards are tested in trial training runs, while the learning algorithm and the rule for selecting the final reward are held fixed. On unseen scenarios of the Overcooked and SMAX benchmarks, task returns were increased by 58.2% and 52.5% over the default rewards on two kitchen layouts, and the mean win rate in two combat scenarios was raised from 51.0% to 89.9%. Under matched candidate and probe budgets, this exceeded the 84.2% and 79.2% obtained with fixed diagnostic feedback and Eureka-inspired reward reflection. These results show that a single autonomously engineered reward can be reused effectively across related cooperative scenarios. 

---
# VerifiedLab: Harnessing Knowns for Unknowns 

**Authors**: Yu Yao, Jiayi Dong, Yang Yang, Shuiwang Ji, Ju Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.37923)  

**Abstract**: Across human history, knowledge has undergone several transitions, from orality to writing and then to digital archives. At each transition, humans established new conventions to preserve knowledge for the new era. Nowadays, as research agents are increasingly deployed for laboratory experiments, knowledge is crossing another boundary: from records written for humans to those for machines. In this work, we introduce VerifiedLab, a multi-agent harness framework for this human-to-machine transition of knowledge. VerifiedLab transforms research output from passive artifacts into active systems that accelerate downstream use, adoption and prediction of future trends. In VerifiedLab, a team of AI scientists with distinct roles reasons over records verified field by field: a principal investigator plans, a librarian computes the evidence directly from the records, specialists draft, and a gate and a critic check each claim against the measured reliability of its data before the harness decides whether to release or withhold it. We demonstrate the efficacy of VerifiedLab through extensive experiments on perovskite solar cells: VerifiedLab recovers known physical trends, anticipates devices reported a year later, and lets AI scientists inherit the literature together with a measure of how far it can be trusted. By scaling the inference-time verification compute, VerifiedLab introduces a new pathway for knowledge dissemination and machine harness for AI co-scientists. 

---
# SciGame: Let Agents Play the Scientific Discovery Games in the Real and Counterfactual World 

**Authors**: Yusheng Zhao, Bowei He, Xue Liu, Mingxiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.37919)  

**Abstract**: Large language model agents are increasingly used for scientific discovery. However, existing benchmarks mix law inference with programming, score data fitting instead of prediction on unseen inputs, and do not separate empirical evidence from pretrained priors. We introduce SciGame, a controlled benchmark that converts scientific literature into structured scientific cards. Each card yields a pair of Original and dimensionally consistent Counterfactual environments with the same task interface. SciGame contains a Historical collection of canonical laws and a live Temporal collection curated from papers published after model knowledge cutoffs. It evaluates agents in candidate-guided Identification and open-ended Discovery settings. Each setting runs under Prior-only, Fixed, and Adaptive protocols. Experiments on five frontier models yield three findings. First, familiar priors hurt performance once the governing relation is altered. Second, adaptive experimentation helps the strongest models but harms weaker ones under limited budgets. Third, good explanations can hide quantitative failures. Recovering the exact symbolic form remains the main bottleneck, even when agents interpret evidence correctly. Current agents can investigate systems actively, but inducing exact laws in unfamiliar settings remains an open challenge. 

---
# Beyond Low-Pass Dynamics: Frequency-Selective Spiking Reservoirs with Resonant Neurons 

**Authors**: Aurora Micheli, Claudio Gallicchio, Nergis Tomen  

**Link**: [PDF](https://arxiv.org/pdf/2610.37906)  

**Abstract**: Spiking reservoir computing combines the training efficiency of fixed recurrent reservoirs with the energy efficiency of event-driven inference, where computation is triggered by sparse spikes rather than executed densely at every timestep. However, existing spiking reservoirs almost exclusively rely on Leaky Integrate-and-Fire (LIF) neurons, whose low-pass dynamics limit their ability to distinguish information encoded at different temporal frequencies. We introduce HRF-Res, a liquid state machine in which LIF neurons are replaced by a heterogeneous population of Harmonic Resonate-and-Fire (HRF) neurons. These neurons exhibit intrinsic oscillatory dynamics and selectively respond to input frequencies, enabling the reservoir to capture frequency-specific temporal patterns. The result is more discriminative representations and higher classification accuracy than LIF-based reservoirs, achieved without sacrificing the energy efficiency of spiking computation, with up to 19x lower energy consumption than an equivalent non-spiking oscillatory reservoir. Evaluated on ten benchmarks spanning time series, sequential images, audio spike trains, and neuromorphic vision, HRF-Res achieves state-of-the-art accuracy among spiking reservoir methods. These results demonstrate that introducing frequency-selective dynamics into spiking reservoirs improves representational quality and classification performance while preserving their inherent energy efficiency. 

---
# FilingLoop: Evaluating and Evolving LLM Agents for Corporate Disclosure Review and Revision 

**Authors**: Wentao Zhang, Yilei Zhao, Guangyan Gan, Shuo Sun, Lin William Cong  

**Link**: [PDF](https://arxiv.org/pdf/2610.37905)  

**Abstract**: Public disclosure repositories such as SEC EDGAR support informed investment through web access to corporate filings. Supporting their review with language model agents requires discovering unstated deficiencies across related documents, acquiring missing company facts, and verifying that requested revisions satisfy evidence-backed requirements. Existing financial question-answering evaluations provide limited evidence of these interactive capabilities. We introduce \textbf{FilingLoop}, an interactive benchmark and a self-evolving agent framework for corporate disclosure review and revision. Grounded in historical U.S.\ Securities and Exchange Commission correspondence, FilingLoop reconstructs 380 predominantly IPO review tasks from 96 companies under distinct information constraints. An evaluator separate from the acting agents scores final filings and responses against 6,099 eligible requirements and audits unsupported claims and harmful revisions. Our framework couples iterative review and revision within each case with procedural learning across cases, converting requirement-level evaluation feedback into role-specific guidance without updating model weights. Updates affect only subsequent cases, allowing procedural adaptation to be assessed without retrying scored tasks. On the evaluated stream, evolution increases requirement fulfillment from 24.28\% to 27.96\% and full comment resolution from 16.90\% to 20.75\%, with a 5.8\% increase in total cost over the fixed-policy variant. FilingLoop connects the concerns an agent discovers with the company facts it obtains and the revisions it commits, providing requirement-level feedback for procedures used in later review tasks. 

---
# The Anatomy of Socio-Demographics in Pluralistic Offensiveness Modeling: When and Where It Helps, and the Role of Reasoning 

**Authors**: Yifan Wang, Leixin Zhang, Jinyi Mu, Yu Wang, Mayank Jobanputra, Isabel Valera, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2610.37896)  

**Abstract**: Subjective judgments such as offensiveness often vary substantially across individuals, motivating pluralistic modeling approaches that incorporate annotators' socio-demographic information to capture this variation. However, prior findings on the benefits of socio-demographic information for pluralistic modeling are mixed, calling for a systematic analysis of both its effectiveness and the mechanisms through which it improves prediction. In this work, we study $\textbf{when}$ and $\textbf{where}$ socio-demographics benefit pluralistic offensiveness modeling, as well as $\textbf{the role of explicit reasoning}$ in leveraging this information. Across three multi-annotator datasets with prompted and trained LLMs, we first show that demographic gains are small and inconsistent for individual predictions but become clearer at the group and population levels, because aggregation attenuates the idiosyncratic variation that masks small but systematic demographic signals. Second, using a reasoning and prediction format that explicitly decomposes individual judgment into an overall baseline and a demographic-specific deviation, we show that the gains arise from both components: demographic information improves baseline estimation and enables meaningful demographic-specific shifts, and interventions show that better component estimates improve final predictions. Finally, comparing direct demographic conditioning with explicit reasoning about how the text interacts with annotator demographics, we find that reasoning yields more accurate baseline and shift estimates as well as final predictions, whereas conditioning alone rarely produces informative shifts. These results suggest that the mixed evidence in prior work may partly stem from individual-level evaluation, and that the value of socio-demographic information depends on both how it is evaluated and how models are guided to use it. 

---
# Verification Learning Theory: A Process Model of Checking and Retention in AI-Assisted Judgment 

**Authors**: Christoph Trattner  

**Link**: [PDF](https://arxiv.org/pdf/2610.37875)  

**Abstract**: Accurate AI assistance can improve immediate judgments while reducing opportunities to practise the operations needed for independent performance. Verification Learning Theory (VLT) formalises this possibility for claim-evaluation practice followed by correct, independent, rule-specific feedback. The model links perceived assistant reliability to evidence checking and links the history of those checks to retained rule use. Its strict form assumes that task demands and perceived reliability influence learning only through checking, with equal learning per genuine check. It therefore predicts that perceived fallibility can improve delayed unaided judgment even without observed assistant errors, and that this advantage disappears when checking is already required and performed. Rule-specific gains and forecasts from checking histories provide further tests. Simulations examine the consequences and parameter dependence of these commitments; they do not constitute empirical validation. Comparisons with depth-of-processing and feedback-encoding accounts identify where predictions diverge and where the accounts remain observationally equivalent. The framework specifies experiments that can test the proposed mechanism and separates retention of practised rules from the broader problem of recognising which rule applies to a new claim. 

---
# What Constitutes Food Similarity? Visual Representations and Retrieval in Indian Cuisine 

**Authors**: Saransh Kumar Gupta, Akshadha Sharma, Lipika Dey, Partha Pratim Das, Ramesh Jain  

**Link**: [PDF](https://arxiv.org/pdf/2610.37867)  

**Abstract**: Food similarity underlies retrieval, recommendation, substitution, and other food computing tasks, yet food representations are typically evaluated by identity preservation. We ask what happens to food similarity beyond identity: how different visual representations organize foods into relational neighborhoods, what culinary relationships these neighborhoods reveal or obscure, and how consistently these structures emerge across representations. Indian cuisine provides a challenging setting in which visually similar foods may have different culinary identities, while foods within a culinary family can vary substantially in appearance. Using an expanded Khana dataset and seven pretrained visual representations spanning convolutional, transformer-based, self-supervised, and image-text aligned models, we evaluate image-to-image retrieval and image-to-dish neighborhoods, examining culinary-family structure, contrast cases, and cross-representation agreement. We also aggregate image representations to compare dish-level visual similarity across models and against recipe-derived similarity for the same dishes. Image-level neighborhoods vary substantially across representations, including similarities across culinary-family boundaries. At the dish level, aggregation reveals common visual structure despite local variation, but this structure corresponds less strongly with recipe-derived similarity. These findings show that recognition performance alone cannot characterize relational structure: food similarity depends on representation, modality, and abstraction level rather than being a fixed property of the food collection. Understanding these structures is therefore a prerequisite to combining multiple food information modalities into a coherent notion of similarity. 

---
# Seq@K: Evaluating Language Models Under Sequential Feedback 

**Authors**: Thomson Yen, Gilbert Yang, Naimeng Ye, Minghui Chen, Dong Xia, Tianyi Shen, Yechan Jeon, Yaqin Chen, Daksh Mittal, Fangyu Wu, Hongseok Namkoong  

**Link**: [PDF](https://arxiv.org/pdf/2610.37859)  

**Abstract**: Language-model agents differ fundamentally from prior intelligent systems in their ability to learn from natural language in context. This capability enables agents to iteratively refine their answers in response to feedback, a prevalent mode of operation in modern agentic workflows. However, the community primarily ranks models by metrics such as Pass@1 or Pass@K, which cannot capture how they utilize feedback on prior attempts. We introduce Seq@K, a natural evaluation protocol that measures whether a model can solve a problem within K sequential attempts, provided feedback after each previous failure. Unlike multi-turn agentic benchmarks that examine long-horizon planning, Seq@K isolates and formalizes the core capability of refinement from feedback. We evaluate Seq@K on a suite of 7 benchmarks and 16 frontier models, and show that Seq@K reveals a materially different model-selection landscape: under Seq@K, (1) model rankings often differ, (2) the Pareto frontiers admit more efficient configurations and (3) the gains from choosing the optimal model configuration can widen or shrink relative to those of Pass@K. We further show that analyzing model trajectories under Seq@K helps characterize iterative refinement behavior, such as the rate at which models address issues raised in feedback, or how rigidly they hold onto their own answers. Finally, we study how the nature of feedback affects models' iterative refinement, showing that practitioners must carefully design and benchmark the feedback loops of iterative agentic systems. Together, these results establish how Seq@K offers a more faithful metric for intelligent systems that typically operate in such feedback loop, and the necessity to consider Seq@K in evaluating models for these systems. 

---
# Evaluating and improving AI agents for sequencing protocol understanding 

**Authors**: Chi-Lam Poon, Michael Goneos, Zezheng Song, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.37851)  

**Abstract**: Existing scientific-agent benchmarks rarely evaluate whether agents can reconstruct the exact molecular structures and generation workflows specified by sequencing protocols, a capability critical for analyzing customized sequencing assays. We introduce LibStructBench, a document-grounded benchmark for reconstructing sequencing-library structures and molecular workflows. Our experiments show that frontier agents often produce approximately correct reconstructions but remain unreliable at exact reconstruction, particularly in molecular-state representation and workflow connectivity. Switching the underlying language model produced larger average performance differences than switching agent harnesses, with a statistically significant difference only for molecular-state reconstruction. Additional knowledge improved oligo-family recovery in a small pilot without improving mean molecular-workflow scores. Autonomous accumulation of procedural rules and cross-protocol examples showed non-monotonic validation performance and no mean held-out molecular-workflow improvement in the tested trajectory. These findings highlight the need for more controlled and verifiable approaches to scientific reasoning about molecular processes in complex experimental protocols. 

---
# Detecting Covert Collusion between AI Agents through Partner-Outsider Positive Listening 

**Authors**: Kevin Yandoka Denamganaï  

**Link**: [PDF](https://arxiv.org/pdf/2610.37849)  

**Abstract**: AI agents that share a secret convention can coordinate through messages that look innocent, so a monitor that reads only the messages can miss them. We propose a behavioural test: a partner that uses a code reacts to the exact wording of a message, whereas an uninvolved agent placed in its seat does not. Partner--Outsider Positive Listening (PoPLi) is the difference between the two agents in how much their action changes when the message is reworded with its meaning kept, and it is computed after the fact by replaying logged prompts. In a simulated market with five conditions (no collusion, a joint reward with neutral messages, the same with an openly stated intention, a covert first-letter code, and the code with a stated intention), PoPLi is between 0.48 and 0.59 whenever the code is present (95% intervals all above 0.29) and between -0.10 and -0.01 without it. Message-only monitors did not separate the conditions: one reading the speaker's messages answered "no" in all 50 markets, and one reading every seller's messages flagged 9 of 10 markets with no collusion. PoPLi detects the use of a private channel, not collusion as such, and it cannot detect collusion through incentives alone or codes carried by meaning. 

---
# Measuring Where Intent Gets Lost in Agentic Smart Contract Synthesis 

**Authors**: Abhinav Goel, Agostino Capponi, Alfio Gliozzo, Chaitya Shah  

**Link**: [PDF](https://arxiv.org/pdf/2610.37830)  

**Abstract**: AI agents can now write software from a plain-English description, but code that runs is not the same as code that does what was asked. This gap matters most for smart contracts, the programs behind Web3 applications that move money on a blockchain and cannot be fixed once deployed. We ask how to check whether an AI-written contract behaves as it should, and where the requested behavior gets lost.
We design an evaluation built on execution rather than opinion. We replay the same transactions on an AI-written contract and on the original it should reproduce and measure how often they agree. We test every grading tool on planted mistakes, compare against an answer key written without the description, and audit our own scorer until known wrong answers cannot pass. Common methods, such as AI judges and standard analysis tools, miss obvious mistakes, and scoring choices matter too. Checking only the functions every system includes hid about half the difference between pipelines.
We then propose an elicitation-first architecture. Before writing code, the agent asks the client a short checklist of questions, checks the contract against an independently written executable answer key, and returns disagreements to the client instead of silently rewriting code. Asking first matters far more than reviewing or patching afterwards, across three AI models, while automated security fixes often break requested behavior. A few well-chosen questions recover much of what a short description omits, and their value can be estimated without the original contract. To our knowledge, this is the first study in smart contract synthesis to gather missing requirements before generation and check the result against an independently written executable model. The lessons extend to any setting where AI agents turn requests into programs. We release our tools, 600 tasks and all results. 

---
# What Does an Ontology Catch? Curated, LLM-Generated and Data-Derived Ontologies as Checkers of LLM Agents over Knowledge Graphs 

**Authors**: Michal Orzechowski, Rund Tawfiq, Tomasz Kosciolek  

**Link**: [PDF](https://arxiv.org/pdf/2610.37809)  

**Abstract**: LLM agents that answer questions over Web knowledge graphs are increasingly wrapped in checks derived from the graph's ontology, on the assumption that a formal schema makes a reliable judge. Yet a checker can be wrong as well: it can miss errors, and it can flag correct answers and push the agent into "fixing" them. Whether a curated ontology is worth its cost as such a judge, and how it compares with an ontology an LLM writes or a schema mined from the data, is unknown. We make the checker itself the object of study. Across curated, LLM-generated and data-derived ontologies, used as prompt context, as a checker, or both, and across open-weight and proprietary agents on public graphs and a private graph of nearly $10^{9}$ triples, we account for what ontology-derived checks catch and what they wrongly flag. We show how each source trades caught errors against false alarms, how a curated schema used as published can contradict its own data, which errors structural checks miss, and whether combining sources or giving the agent the checker's specific feedback pays off. On UniProt, the curated ontology flags more correct answers than either an ontology derived from the data or one an LLM derives from a summary of the data with masked identifiers. On real agent answers, its flags are only marginally more precise than random flags. 

---
# WebAnything: From the Wild Web to Scalable Agent Environments 

**Authors**: Tianci Xie, Zikun Cai, Chaoyun Zhang, Pinxue Guo, Bingxuan Li, Rui Zheng, Hang Yan  

**Link**: [PDF](https://arxiv.org/pdf/2610.37786)  

**Abstract**: Web agents require diverse, executable, and resettable website environments for reproducible training and evaluation. Human-assisted mirrors demand per-site engineering, while generative construction can diverge from specified real websites. Capture-and-replay preserves observations, with interaction coverage constrained by captured content and replay capabilities.
We present WebAnything, a fully automated framework that reconstructs source-grounded, reusable Web environments from real URLs without per-site manual implementation or repair after initial setup and authorization. An Explorer-Builder collects browser-observable structures, resources, and interaction outcomes; an independent verifier checks frozen candidates and guides local repair or targeted source recapture.
We construct 244 website artifacts across 14 application domains and 53 categories, supporting 12,984 executed and accepted tasks. Compared with human-assisted WebHarbor, WebAnything achieves higher aggregate scores on three fidelity metrics, with 12 wins, two ties, and no losses in a blind audit. Fine-tuning Qwen3.5-9B on 5,856 interaction trajectories improves REAL success by 9.9 percentage points, Odysseys-Easy Rubric Avg and Task Avg by 6.25 and 6.32 percentage points, and WebArena-Verified Overall success by 13.33 percentage points under the fixed harness. Code: this https URL 

---
# OpenFrontiers: A Collection of Open Research Problems and a Benchmark for Scientific Rediscovery 

**Authors**: Jiarui Yao, Ruida Wang, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37739)  

**Abstract**: Large language models (LLMs) have shown increasing promise in research-level problem solving, yet a comprehensive collection of such problems is still missing, and evaluating them on research-level problems remains challenging. We introduce OpenFrontiers, a cross-disciplinary resource that combines a structured corpus of open research problems with a retrospective benchmark of recently resolved problems in mathematics. The corpus records problem statements, context, progress, and other information for 12,151 problems. The benchmark contains 200 problems, 110 resolved in 2026 and 90 still open, each with a Lean~4 statement checked by automated probes and two independent reviews; a subset HQ-50 further filters based on quality score, etc. Our evaluation protocol grades formal proofs with two independent checkers in an isolated sandbox under closed-book, frozen-retrieval, and live-web conditions. Evaluating seven LLM agents on the HQ-50 subset under a standardized budget, we find that closed-book rediscovery is rare, at most 7 of 40 resolved problems, whereas live-web success mostly reflects retrieval of published solutions. By linking a broad research-problem corpus to a formally graded evaluation subset with documented solutions, OpenFrontiers provides a foundation for measuring research-level rediscovery and studying progress toward solving currently open scientific problems. The dataset and benchmark could be found in this https URL. 

---
# ThinkJev: One Thought, Every Budget for Calibrated Typed Decisions 

**Authors**: Xiaolu Ding, Kai Yang, Dongdong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37732)  

**Abstract**: Many uses of language models end in a typed decision: a choice among a closed set of options, whose probabilities code can act on. Reading those probabilities from one forward pass is fast but often wrong on decisions that take several steps, such as arithmetic, counting, and date comparison. Letting the model think first fixes most of these errors, but costs seconds per decision instead of milliseconds. To pay for thinking only where it helps, ThinkJev builds a fast and a slow system from one open hybrid-thinking model that can answer directly or after a thought. System 1 reads the model's probabilities for the option letters when it answers directly; System 2 reads the same letters after the model has thought on the same cached prompt, so escalating repeats no work and both systems give probabilities over the same options. Because the letters can be read after any number of thought tokens, one sampled thought also gives the answer under every thinking budget. Without training, ThinkJev lets System 1 answer alone only where a statistical test on labeled dev data guarantees that its accepted answers stay under a target error rate. At a 5% target, Qwen3.5-27B answers 24.5% of 8359 test decisions this way with a realized error of 3.9% (Qwen3.5-4B and 9B: 6.5% and 0.0%). For the rest, stopping each thought as soon as its answer is confident reaches 81.6% accuracy, against 82.7% for always thinking, with 72% of the thought tokens. Choosing when to think from System 1's confidence alone saves less, because the decisions System 1 is least sure of are often not the ones thinking fixes. On the public items of JevBench, an independent benchmark of typed-decision services, Qwen3.5-27B thinks on 29.4% and is not distinguishable in accuracy from the two thinking APIs listed (exploratory). Code and data generators are available at this https URL. 

---
# Amortized Localized Posterior Inference for Model Calibration with Application to an Ocean Turbulence Closure 

**Authors**: Exaucé Luweh Adjim Ngarti, Élise Arnaud, Léo Nicoletti, Victor Trappler, Arthur Vidard  

**Link**: [PDF](https://arxiv.org/pdf/2610.37731)  

**Abstract**: Model discrepancy may prevent a single calibration parameter from reproducing a physical response across the context domain. In addition, uncertainties in observations and contexts affect calibration and must be accounted for and quantified. We construct a family of localized posteriors by weighting observation likelihoods according to their proximity to a target context. Gaussian localization weights incorporate context uncertainty through covariance inflation, with reflection at domain boundaries and an optional correction for uneven context sampling. A conditional normalizing flow approximates the family of posteriors through amortized variational inference. Integrating the localized posteriors under a prescribed context measure gives the marginalized parameter distribution. In a twin experiment with the Branin-Hoo model, we compare global and localized inference under uniform and skewed context distributions. Localization reduces the reported aggregate predictive errors relative to the global posterior. An application to the k-$\epsilon$ turbulence closure in a single-column ocean model, with atmospheric surface forcing as context, shows that localized posterior calibration reduces temperature and zonal-velocity errors relative to deterministic and global-posterior calibration. Both twin experiments are constructed to admit local compensation. 

---
# LRMoE: Demand-Side Load Re-Calibration for Efficient MoE Training on Memory-Constrained Nodes 

**Authors**: Jiaming Hu, Zhida Jiang, Zhaolong Xing, Yu Li, Zhen Chen, Ke Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37720)  

**Abstract**: Sparse Mixture-of-Experts (MoE) models offer scalable capacity for Web applications, but uneven routing creates computation and communication bottlenecks. Existing placement and replication methods redistribute this workload, yet additional expert copies compete with token-related buffers for limited GPU memory. We present LRMoE, a memory-aware MoE training system that jointly recalibrates backward demand and balances the retained workload. During forward execution, it plans retention quotas and expert placement under resource constraints while preserving all routed computations. During backpropagation, a load recalibration metric estimates each token-expert pair's contribution to the expert update and selects pairs within these quotas. A capacity-aware routing algorithm distributes retained pairs among eligible instances of their original experts. Physical compaction then forms smaller communication payloads and variable-size grouped GEMMs, reducing backward traffic, computation, and execution-buffer demand. Experiments on Qwen3-30B-A3B and GLM4.5-106B-A12B show that LRMoE reduces cumulative MoE-layer training time to fixed-loss targets by 21.2-44.8% relative to SYMI. 

---
# Rethinking Neural Thickets Without Weight Perturbations 

**Authors**: Nicolas Dufour, Elouan Gardès, Amil Dravid, Patrick Pérez, Loic Landrieu  

**Link**: [PDF](https://arxiv.org/pdf/2610.40649)  

**Abstract**: The Neural Thickets hypothesis posits that pretrained language models lie in dense neighborhoods of task-specific experts in weight space, which random Gaussian perturbations can uncover. We reproduce the reported gains, but show that they do not require this geometric explanation. Perturbing a single decoder block already recovers much of the improvement, while keeping the model entirely frozen and adding a random bias to its output logits matches or exceeds full-weight perturbations on most reasoning tasks. This suggests an alternative explanation based on output diversity, validation-based selection, and voting. We formalize this view as Perturb-Select-Vote (PSV), a general procedure that generates perturbed predictors, selects high-performing members on a small validation set, and aggregates their predictions.
This interpretation connects Neural Thickets to established inference-time methods, including sampling, self-consistency, and verifier-guided selection. A first-order analysis further shows that small weight perturbations act primarily through their induced logit shifts. We extend this analysis to vision, where logit PSV substantially improves performance under domain shift, including on Earth observation, while full-weight perturbations provide little additional benefit. Together, these results suggest that the gains attributed to nearby weight-space experts can instead be obtained through output diversity, selection, and voting. Logit PSV realizes these benefits with a fully frozen model, providing a simple, training-free alternative to full-weight perturbation ensembles across language and vision. 

---
# Read, Allocate, Route: Role-Coupled Tokenization for Pixel-Space Generative Transformers 

**Authors**: Jiongfan Yang, Weichen Fan, Haiwen Diao, Zhongang Cai, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.40647)  

**Abstract**: Pixel-space generative Transformers typically use one token per image patch, so capturing finer detail requires more tokens and makes global attention increasingly expensive. We observe that finer detail does not necessarily require more global communication: low-frequency structure benefits from global exchange, while high-frequency detail can be modeled locally. Based on this insight, we introduce SpecSlot, a pixel-space rectified-flow Transformer that decouples output detail from the length of the global attention sequence. SpecSlot expands each spatial anchor into multiple slots, each assigned a disjoint band of DCT coefficients; this separates anchor density from spectral subdivision. Only the lowest-frequency slot from each anchor participates in global attention, while all slots interact locally. This allows the model to increase spectral detail without enlarging the global sequence. A spatial head further adds a local correction to the spectral prediction, while the generative state, target, and loss remain in pixel space. On ImageNet-256, SpecSlot achieves a comparable final FID to a FLOPs-matched JiT control while converging faster. On ImageNet-512, it improves FID by 2.84 over the control with 3.9% fewer FLOPs, and comes within only 0.69 FID of JiT-B/16 at 38.2% lower cost. On text-to-image generation, SpecSlot surpasses JiT-B/8 on DPG-Bench by 1.06 at 40.6% lower forward cost. 

---
# Luggage re-identification: A study of strategies towards a real-world application 

**Authors**: Quentin Dhôte, Romaric Audigier  

**Link**: [PDF](https://arxiv.org/pdf/2610.40643)  

**Abstract**: Luggage re-identification (ReID) is a critical task in video surveillance for forensics, security or border management (suspicious or abandoned luggage recognition), as well as luggage management (lost luggage recovery). It enables the association of observations of the same luggage across disjoint camera views in complex environments such as airports, train stations and border crossing. While existing approaches primarily focus on appearance-based cropped image matching under relatively clean conditions, real-world scenarios are characterized by frequent occlusions, cluttered background and crowded scenes which significantly degrade performance. Yet, strategies leveraging the complementarity of multiple shots and views from different video streams is overlooked. In this paper, we study strategies for an end-to-end pipeline that operates from multiple video streams to detect and re-identify luggage. We evaluate different variants on a benchmark dataset acquired in real-world environments to address the questions of selecting and merging different views and shots of the same luggage as well as training on different environment datasets for an overall ReID performance improvement in practical applications. 

---
# From extreme weather to agricultural consequences with agentic AI 

**Authors**: Minghan Yu, Mengyu Dai, Yixin Wen, Haizhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.40641)  

**Abstract**: Extreme weather is forecast days ahead, yet its consequences for agriculture remain hard to predict: a forecast says where a storm will strike, not which crops will suffer or how much is at risk, and the records that could tell span many sources. We present WxCrop, the first agent that judges agricultural hazards from physically interpretable weather indices examined by causal analysis, and the first that combines weather and agricultural information to predict crop loss reliably for past and forecast dates. We demonstrate it in Florida on hurricanes, freezes and dry heat, with a loss model calibrated on insurance records for each, and the same procedure extends to other hazards and areas. Three checks show reliability: the hurricane index detects recorded cyclone days with an AUC of 0.98, recorded losses in held-out years are reproduced to within 2%, and independent production records confirm the county rankings (rank correlation 0.63). Users ask the agent for each crop group's loss as a fraction of expected season value, every number sourced, so that governments, growers and scientists respond to agricultural disasters quickly. 

---
# Probing Spatial State Representations in Visual Foundation Models 

**Authors**: Baiqiao Yin, Zhongang Cai, Ruisi Wang, Qiwei Liang, Lei Yang, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.40636)  

**Abstract**: An observer sees only part of the surrounding world, yet must keep track of what lies beyond the current view. Spatial perception therefore involves more than recognizing individual images: it calls for a persistent spatial state that is updated under self-motion and used to guide action. Do visual foundation models provide representations that support such a state? We study this question through four levels of spatial capability---spatial structure, object persistence, predictive spatial update, and actionability---using frozen-feature probes on ten representations under streaming egocentric observations of static indoor environments. The resulting capability profiles vary across tasks, network layers, and prefix lengths: strong readout of the visible scene does not ensure accurate object persistence or future-feature retrieval. Matched controls separate readout accuracy from reliance on visual history and motion. Native future prediction retains recoverable spatial structure, but a spatial reader trained on real features does not transfer to predicted features unchanged. Finally, goal-dependent candidate selection does not establish reliable continuous action utility: predicted actions respond to goal changes, yet supplying the correct goal does not reliably improve progress. Across the four levels, object persistence and actionability build on camera-pose readout, whereas feature prediction builds on visible-scene readout. Together, these findings motivate evaluating spatial representations not only by what can be decoded from them, but by whether that information is maintained beyond observation, updated predictively, and used for goal-directed action. 

---
# TextSeal2: Distortion-Free LLM Watermarking with Copula Mixing and Memoryless Races 

**Authors**: Tom Sander, Pierre Fernandez, Itai Gat, Haiming Bao, Jiaxin Shi, Jason Park, Valeriu Lacatusu, Sylvestre-Alvise Rebuffi, Hongyan Chang, Hady Elsahar, Tomáš Souček, Tuan Tran, Surya Parimi, Alexandre Mourachko  

**Link**: [PDF](https://arxiv.org/pdf/2610.40613)  

**Abstract**: We present TextSeal2, a distortion-free watermark for large language models that keeps its signal under speculative decoding: tokens that replace rejected drafts remain watermarked with the same keys, and the acceptance rate is unchanged. It also preserves output diversity and adds no measurable inference latency. Its predecessor, TextSeal, restores diversity in the Gumbel-max scheme by routing each token to one of two keys via a coin flip. This is suboptimal because the detector averages the scores from both keys without knowing which key was used. TextSeal2 instead merges two or more keys into a single virtual key and couples it to fresh noise through AutoMark's Gaussian copula. Any number of keys, including publicly shared keys, can be merged this way without diluting detection. Under speculative decoding, TextSeal2 verifies keyed drafts with one more exponential race, run on the leftovers of the draft races. By memorylessness, this race reproduces block verification, so the output follows the target model exactly. We evaluate TextSeal2 on three LLMs. At equal output token diversity, TextSeal2 detects 8 to 22 percentage points more 100-token answers than TextSeal at a false-positive rate of $10^{-3}$, and is comparable to textGrain while requiring no post-logits optimization. Under speculative decoding, it detects 18 and 22 percentage points more 200-token answers than TextSeal and textGrain, respectively, at the same acceptance rate. 

---
# When to Assist and What to Suggest: Joint Gating and Memory-Grounded Hint Generation in One Compact Decoder 

**Authors**: A. J. A. Jubayer Talukder, Mohammad Ariful Haque  

**Link**: [PDF](https://arxiv.org/pdf/2610.40611)  

**Abstract**: Proactive assistants offer short, memory-grounded hints during a conversation, deciding when to assist and what to suggest. Current systems split this between a gate and a larger generator; we evaluate both together on 28,058 decision points. Stage-wise generation scores are oracle-gate ceilings: behind a real gate, end-to-end exact match is 0.2619 against a ceiling of 0.2971, and end-to-end precision is 0.2387, leaving substantial headroom. A gate pays off through precision and compute, whose saving falls from 4.39x with an 8B generator to 1.38x with a 1B one. The memory profile drives hint quality: removing it costs 24.7 exact-match points (rescaled BERTScore 0.57 to 0.35), while retrieval and a larger generator add nothing on exact match. One 1B decoder can do both, gating from an intermediate layer and generating in the same pass. At its final checkpoint it matches or exceeds the two-model pipelines on automatic end-to-end measures with 1.31-1.76x less wall-clock time than the matched 1B pipeline; across three seeds, its deeper exit ties the separate gate. A human evaluation (1,144 ratings over 32 dialogues) supports the memory effect and rates the single model level with the matched 1B pipeline and below the 8B one. Graded metrics and LlamaPIE's native judge track human ratings more closely than exact match, which we report as a strict lower bound. 

---
# Multistatic ISAC with Deep Learning-Based Clutter Suppression for 5G and Beyond 

**Authors**: Yalin E. Sagduyu, Tugba Erpek, Sastry Kompella  

**Link**: [PDF](https://arxiv.org/pdf/2610.40586)  

**Abstract**: Integrated sensing and communications (ISAC) systems using 5G and beyond signals of opportunity enable joint wireless sensing and communication using shared infrastructure and spectrum resources. While multistatic ISAC can increase sensing coverage with spatially distributed receivers, dense and dynamic clutter can significantly degrade bistatic range and Doppler estimation, leading to errors in target localization and velocity estimation. This paper presents a multistatic ISAC framework with clutter suppression for high-fidelity estimation of target position and velocity. Distributed clutter is modeled with randomly located scatterers with varying Doppler spreads and radar cross sections to capture dense and dynamic clutter effects. A broad range of conventional signal processing baselines are evaluated for clutter suppression, showing limited robustness under strong clutter conditions. To address this limitation, deep learning-based clutter suppression methods are introduced to improve robustness against dense and dynamic clutter by employing residual and attention-enhanced deep neural network architectures that directly enhance target responses from clutter-contaminated observations. These methods substantially outperform conventional approaches in target position and velocity estimation across diverse clutter conditions in multistatic ISAC. 

---
# Gym-Anything for Robotics: Scaling physical simulations with digital agents 

**Authors**: Pranjal Aggarwal, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2610.40565)  

**Abstract**: Realistic, fast robotics simulations could enable large-scale training of embodied AI models that transfer to the real world. However, despite decades of progress, available simulations cover only a narrow range of real-world tasks and physical phenomena. The core bottleneck is that creating better simulations requires significant expert effort and therefore does not scale. To address this, we introduce Gym-Anything for Robotics, a framework in which coding agents create realistic robotics simulations. Our core insight is that simulation creation is a coding agent task, so improvements in coding agents translate directly into richer and more faithful simulations. We select simulations based on U.S. GDP data to cover economically valuable physical work, define what makes a good simulation and a good task, and develop an adversarial creation-audit loop that checks each simulation and task against these definitions. Using this pipeline, we build RoboWorld, a set of 25 simulations with 25 test and 175 training RL environments of varying difficulty. Even the best frontier models fail on every RoboWorld test task, achieving an average reward of only 0.06. Building on RoboWorld, we introduce several new ways for frontier models to improve, including in-context learning across repeated attempts, a privileged teacher that reads the simulator state and teaches a student in context, self-assisted learning from the model's own privileged experience, and code policies that generate demonstrations across varied scenes to train vision-language-action models (VLAs) such as $\pi_{0.5}$. Each of these substantially improves performance. We release our pipeline, RL environments, and benchmark to facilitate future research. Code and data are available at this https URL 

---
# BURGER: Benchmarking and Understanding Robotic Generalization via Embodied Drawing 

**Authors**: Xiangyu Xu, Bangshuai Peng, Ziyi Liu, Mengjie Zhao, Chenhao Lu, Zhengyang Fan, Huazhe Xu, Kaizhe Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.40532)  

**Abstract**: How effectively can vision-language-action (VLA) models learn diverse language-conditioned behaviors as training data scales? Investigating this question requires large and varied robot datasets, whose collection on physical robots remains costly. We introduce BURGER, a language-conditioned robotic drawing benchmark with a scalable, prompt-driven pipeline for generating continuous drawing trajectories, both in simulation and the real world. In BURGER, an agent starts from the same blank canvas and controls a robot arm to draw the object or spatial composition specified by language, while each instruction allows an open-ended variety of valid drawings and stroke orders. Using 28,200 simulated demonstrations spanning 282 drawing tasks, we evaluate five representative VLA models across data scales, language grounding, and multimodal behavior. BURGER supports fine-grained analysis of language grounding, spatial composition, and action generation, revealing distinct failure patterns across VLA models. Our experiments suggest that some VLA models can benefit from greater task diversity and demonstration scale, capture multimodal drawing behaviors, and generalize to unseen compositions of familiar objects. Finally, we deploy a trained policy on a Franka Panda arm, demonstrating the ability to draw reactively under human intervention. Project homepage and code can be found at this https URL. 

---
# How I Aligned Your Model: Scaling Up Multilingual Safety Training with Templates 

**Authors**: Max Neuwinger, Hanna Yukhymenko, Imanol Schlag  

**Link**: [PDF](https://arxiv.org/pdf/2610.40516)  

**Abstract**: Safety alignment for Large Language Models is heavily focused on English with limited multilingual representation, particularly damaging low-resource language performance. Related open training datasets are often limited in size or do not follow region-localized harmful topics to broaden the scope of alignment beyond cultures. We propose a scalable template-based augmentation pipeline to produce massive multilingual refusal and benign corpora, extracting safe templates from harmful prompts, translating them and then filling in the slots with culturally authentic entities, both harmful and benign. As a result, we release MSafeTemp, a validated corpus spanning 60 languages useful for multilingual safety alignment and evaluation. Using the dataset, we show that English-only evaluation misjudges multilingual safety: for several models, attacks that fail in English succeed once translated, while culturally localized variants expose further failures that translation alone misses. We also confirm that augmenting safety training data through templates allows for higher diversity and quality of alignment, resulting in improved safety performance beyond English. MSafeTemp and the respective pipeline are aimed at facilitating research and development of safer and more responsible models for less-represented cultures. 

---
# AI in the EGI e-infrastructure -- Past experiences and future strategies 

**Authors**: Gergely Sipos, Sergio Andreozzi, Marco Rorro, Andrea Manzi, Tiziana Ferrari  

**Link**: [PDF](https://arxiv.org/pdf/2610.40494)  

**Abstract**: The EGI Federation has provided federated e-infrastructure services to European and global science for over 15 years. We review EGI's early experiences with AI in the interTwin and iMagine Horizon Europe projects, which built digital twins in physics and environmental sciences and an image-analysis platform for aquatic research. We then examine how the AI landscape has changed since 2023 and argue that the focus of AI in science has shifted from model training to adapting and applying AI to specific domains, projects and questions. Building on these lessons, we present the AI strategy emerging from the EGI Federation Strategy 2026-2030, organised around three enablers: (1) an AI-ready infrastructure connecting compute, data and application resources across the continuum; (2) human skills to know what is possible with AI and how to implement it; and (3) trust, to run AI with confidence even in high-risk environments. We describe the projects implementing each enabler, including federated LLM inference, AI-ready data platforms and AI competence centres, and discuss a division of labour with EuroHPC's AI Factories, RAISE, research infrastructures and EOSC, in which EGI specialises in the customisation and long-term operation of AI for research. 

---
# Fast and accurate neural triage decoding of ultra-high-rate quantum LDPC codes 

**Authors**: Hong-Ye Hu, Zhenlai Wang, Binzhao Luo, Tao Xiang, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.40490)  

**Abstract**: Quantum error correction is essential for fault-tolerant quantum computing. High-rate quantum low-density parity-check (qLDPC) codes have emerged as a promising approach to reducing the substantial physical qubit overhead of quantum error correction, enabling encoding rates exceeding $1/2$ in blocks of approximately one thousand physical data qubits. However, their practical implementation requires decoders that combine high accuracy with low latency to keep pace with quantum hardware. Existing decoders of high-rate qLDPC codes can achieve high accuracy but often incur substantial computational latency, whereas neural network decoders have yet to be demonstrated for high-rate qLDPC codes. Here, we introduce \emph{neural triage decoding}, a framework that uses syndrome consistency to adaptively allocate neural computation. A local neural message-passing network rapidly predicts error corrections, additional iterations reduce residual syndrome violations, and a rarely invoked diversification stage searches for consistent corrections to the remaining shots. We demonstrate that the \texttt{Triage} decoder can decode ultra-high-rate qLDPC codes encoding a few hundred logical qubits, including lifted-product and Kasai codes, achieving state-of-the-art accuracy comparable to the near optimal algorithmic decoders while reducing decoding latency by approximately one to two orders of magnitude. Our framework is inspired by a rigorous analysis of the exact bitwise marginal decoder, revealing a distance-independent syndrome-inconsistent error floor. In addition, we show the robustness of neural network decoders to mismatched error priors. Our results establish adaptive neural decoding as a promising route towards fast, accurate, and scalable quantum error correction in the near term. 

---
# MetaRefine: Score Refinement with Multi-Level Features for Image Assessment 

**Authors**: Chuanzhi Xu, Chengkun Yue, Haolan Guo, Xuanhua Yin, Boyu Wei, Zihan Deng, Qiang Qu, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2610.40487)  

**Abstract**: An image's original dimensions, aspect ratio, and composition provide important cues that can affect its technical and aesthetic quality. Image assessment methods use resizing, cropping, padding, or patch sampling to accommodate different input sizes. However, these operations can alter or obscure original geometric and compositional cues. Existing assessment models also encode rich multi-level visual evidence that can be further extracted and exploited to improve existing models. This motivates us to combine visual evidence with original-image geometry for score refinement without changing the original scoring pipelines. In this paper, we introduce MetaRefine, a metadata-aware score refinement architecture whose lightweight residual heads predict score corrections from geometry, multi-level visual features, and the frozen model's original score. Across 100 configurations on 21 datasets, MetaRefine improves PLCC in 88 configurations and SRCC in 85, with average relative gains of 7.55% and 7.10%, respectively. It reduces MAE in 90 configurations and RMSE in 89, with average relative reductions of 23.27% and 21.93%, respectively. The median parameter overhead is only 1.11%. MetaRefine can further improve state-of-the-art (SOTA) assessment models on multiple benchmarks. 

---
# NECA: Neighbor Event Curve Augmentation for Multi-Horizon ICU Deterioration Prediction 

**Authors**: Chenyang Wang, Jiawei Theresa Tang, Xinran Liu, Xinyu Lu, Xulei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.40420)  

**Abstract**: Early prediction of ICU organ dysfunction events supports timely clinical monitoring and intervention, but estimating short-term organ-specific risk from early ICU records remains challenging due to irregular measurements, heterogeneous clinical variables, and the need to predict risk across multiple future horizons. Existing supervised models usually treat each patient as an isolated sample and do not explicitly use the observed event timing of similar historical patients. We propose Neighbor Event-Curve Augmentation (NECA), a model-agnostic method that retrieves training-set patients with similar early-window profiles and converts their subsequent organ dysfunction event onset times into cumulative event-curve features over 6h, 12h, 24h, and 42h. These features are concatenated with the original patient representation and used by standard downstream predictors. On PhysioNet 2012 and a 4,000-stay MIMIC-IV subset, NECA improves average PR-AUC and ROC-AUC across four organ dysfunction event tasks and seven supervised learners. The largest gain appears for neurological dysfunction events, where task-level average PR-AUC increases from 0.461 to 0.557 on PhysioNet 2012. Ablations show that clinically meaningful retrieval and horizon-specific neighbor risk drive most of the improvement, while the full cumulative curve adds task-dependent temporal context. These results support similar-patient event timing as a structured temporal risk signal for ICU organ dysfunction event prediction. The source code will be publicly released upon acceptance of this paper. 

---
# IMPACT: Inverter-Informed Multimodal Learning for Photovoltaic Generation Forecasting 

**Authors**: Anushka Katiyar, Cong Feng, Jing Wang, Tristan Liu, Simon Julien, Zachary Jacobs  

**Link**: [PDF](https://arxiv.org/pdf/2610.40418)  

**Abstract**: This work develops a physics-informed multimodal deep learning framework for 10- to 110-minute-ahead probabilistic photovoltaic power forecasting. The forecasting model is built on a four-branch architecture that separately encodes plant-level temporal history and inverter-level spatial cloud dynamics through a graph attention encoder with a learned adjacency, known-future solar geometry, and the current meteorological state, fused through gated cross-attention. The geometry branch adds two forward-looking physical signals, spatial cloud heterogeneity and inverter ramp-down activity, each empirically weighted toward the horizon where it is most predictive. The decoder anchors its prediction to a smart persistence baseline, learning a horizon-conditioned correction, and the resulting quantile forecasts are calibrated online through a kernel-weighted adaptive conformal calibration step. The proposed model outperforms all 14 baselines, achieving a forecast skill score (FSS) of 0.2950, 0.099 above the next-best model. A branch-composition ablation shows that every branch contributes positively and the model retains substantial skill in challenging overcast and variable sky conditions (FSS 0.198 and 0.255). 

---
# Bridging Neural Representations and Symbolic Semantics in Audio Classification 

**Authors**: Avishek Das, Pascal Hitzler  

**Link**: [PDF](https://arxiv.org/pdf/2610.40414)  

**Abstract**: Deep audio classifiers can achieve strong predictive performance, but their hidden representations remain difficult to interpret. Existing audio explainability methods often focus on input-level attribution, such as waveform segments, spectrogram regions, or acoustic features, and provide limited insight into what hidden neurons encode. Interpreting hidden neurons at the concept level can provide a more meaningful view of what an audio model learns internally. This paper investigates whether hidden neurons in audio classifiers can be associated with human-understandable symbolic acoustic concepts. A model-agnostic neurosymbolic framework is presented for analyzing the final hidden layer of audio classifiers, containing 32 neurons, through symbolic concept induction. The framework is evaluated on two audio classification settings: music genre classification using GTZAN dataset and musical instrument classification using IRMAS dataset. Results show that several hidden neurons can be associated with meaningful induced concepts, and most confirmed neuron-concept pairs are statistically supported using the Mann-Whitney U test. These findings suggest that symbolic concept induction is a promising direction for explaining latent audio representations beyond input-level attribution. 

---
# SCARL: Sequence Confidence-Aligned Reinforcement Learning 

**Authors**: Zhenghong Ke, Jingbo Zhou, Hua Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.40407)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) improves language-model reasoning by scoring sampled responses with a rule-based verifier and reinforcing those that pass. Where a classifier's training couples predicted probability to labels, nothing ties an RLVR policy's confidence in a response to its reward: aligned training would drive responses with the same advantage toward equal confidence, yet the advantage is blind to confidence. Under this sequence-confidence misalignment, a success the policy barely found earns the same update as one it produces with near certainty, and a confident error is corrected no harder than an exploratory one. We propose SCARL (Sequence Confidence-Aligned Reinforcement Learning), which reshapes, but never re-signs, each advantage with the alignment gap between the response's group-relative confidence and the outcome its advantage certifies, normalized into shares of each class's advantage budget. Misaligned samples absorb the force, responses with the same advantage are driven toward equal confidence, and repairing the misalignment doubles as a remedy for the entropy collapse that erodes exploration. SCARL provably reduces to its base estimator when a class's confidences agree, adds no extra passes and one hyperparameter. On Qwen3-4B-Base it lifts average Pass@32 by 2.4 points over DAPO and is best among nine compute-matched methods on both coverage and per-draw accuracy; applied unchanged across four estimators and four backbones, it improves 14 of 16 settings while preserving policy entropy and widening the confidence-correctness separation. Our code is available at this https URL. 

---
# Understanding and Mitigating Role Spoofing in LLMs 

**Authors**: Ezra Edelman, Benjamin Edelman, Surbhi Goel, Eran Malach  

**Link**: [PDF](https://arxiv.org/pdf/2610.40396)  

**Abstract**: Despite growing capabilities, LLMs remain susceptible to prompt injections: malicious inputs that can manipulate or hijack models. In particular, it has recently been observed that LLMs can exhibit role confusion (Ye et al., 2026), where models attribute the source of text based on its style instead of its labeled role. Malicious text in tool outputs (for instance, a malicious webpage accessed by an agent) can leverage role confusion to impersonate the user, an attack we call role spoofing}. In this work, we seek to understand and mitigate role spoofing. We study how models track "who said what" in a conversation, showing that they use cues from both content (e.g., semantics and writing style) and structure (e.g., control tokens and formatting). We show that LLMs are highly susceptible to role spoofing attacks that combine both fake structure and content. Additionally, in a toy setting, we show that adversarial training which teaches the model to ignore content-based injections does mitigate content-based attacks but leaves the model susceptible to structure-based attacks. This can explain why frontier models, with increased resistance to many injection attacks, can still be susceptible to role spoofing. Finally, we propose a new method for detecting role spoofing using linear probes trained to predict roles in a conversation, and show that it is effective across different settings. 

---
# Scaling Mathematical Proof Supervision from Structured Textbooks 

**Authors**: Youqing Fang, Zhouqi Hua, Yinhao Tang, Yuzhe Gu, Lingkai Kong, Bin Liu, Weiming Zhang, Kai Chen, Yanan Sun, Wenwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.40394)  

**Abstract**: Mathematical supervision for language models is typically flattened into isolated problem-solution pairs, discarding the curricular organization and evidential structure of the original sources. We introduce TOME-Math (Textbook-Oriented Mathematical Evidence), a large-scale corpus that converts 2,118 textbooks spanning the core areas of university-level mathematics subjects into approximately 380K structured proof-supervision records while preserving chapter order, proposition-proof links, dependency references, and source provenance. Leveraging these structural signals, we explore curriculum learning across multiple mathematical subjects and characterize scaling patterns with supervision volume and source breadth. At full scale, fine-tuning on TOME-Math raises the unweighted mean score across our held-out evaluations and established public benchmarks by 16.7 percentage points over the base model, while exceeding all three public instruction-data baselines evaluated in aggregate. Our findings highlight structured textbooks as both a scalable source of mathematical supervision and an empirical framework for studying the impact of data organization on mathematical reasoning. We will release the corpus to facilitate further research on structured mathematical supervision. 

---
# The Athlete as a Scientist of One: Agentic Bayesian Experimental Design for Individualized Physiology from Consumer Wearables 

**Authors**: Behnam Moradi  

**Link**: [PDF](https://arxiv.org/pdf/2610.40370)  

**Abstract**: Large language model (LLM) agents for personal health read wearable data and recommend, but do not learn about the individual in a way that can be audited or falsified. We present EpiCoach, an agentic system that treats one athlete, a cross-country mountain biker, as the subject of accountable N-of-1 science. Claims about the athlete live in an evidence-gated hypothesis registry: each must issue a probabilistic prediction about a future session before it occurs, and its belief moves only through computed evidence from a logged statistical fit. Predictions come from a mechanistic digital twin inferred from heart rate, GPS and barometric altitude: a lagged heart-rate response model, a fitness-fatigue model with learned time constants, and a latent technical-descending skill state, measured on discovered trail segments. The agent prescribes the next training week as a constrained experiment maximizing information gain subject to load and safety constraints, around two instrument rides. We evaluate the system two ways over eighteen months of data. A rolling-origin replay benchmark shows the twin beats standard forecasters on heart rate but not segment time, and the agentic loop matches a static model. A live prospective phase runs the weekly cycle: in its first three weeks the ledger sealed 20 predictions, scored 14, the descent forecast calibrated after one correction, the climb forecast pessimistic as the athlete improved, and the safety officer vetoed the agent twice. Every correction is preserved in the registry, a template for auditable, falsifiable personalization in digital health. We release the registry, benchmark and protocol. 

---
# A Unified Hessian-Guided Framework for Post-Training Quantization 

**Authors**: Mads Toftrup, Marco Federici, Boris van Breugel, Paul Whatmough, Markus Nagel  

**Link**: [PDF](https://arxiv.org/pdf/2610.40345)  

**Abstract**: Low-bit quantization enables efficient deployment of large language models by reducing memory footprint and computational cost. Achieving high accuracy at low precision requires selecting quantization ranges, linear transformations, and weight rounding decisions. Existing approaches typically optimize these components independently using local reconstruction objectives, which fail to account for the varying impact of quantization errors on downstream predictions. Direct optimization of task-level metrics would provide a more faithful objective, but is challenging due to the discontinuous nature of quantization operations and the high computational cost of evaluating their downstream effects.
In this work, we derive a Hessian-guided approximation of the output change induced by joint weight and activation quantization. Based on a tractable Fisher approximation, we formulate a unified post-training quantization framework that enables the joint optimization of quantization ranges, linear transformations, and weight rounding under a common output-sensitivity objective. The resulting method remains practical for modern large language models and consistently outperforms state-of-the-art transformation and rounding methods across both channel-wise and block-wise quantization settings. 

---
# Radiology Guideline-derived Decision Graph for Efficient Clinical Decision Support 

**Authors**: Xuewen Zhou, Zhen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.40327)  

**Abstract**: Radiological risk stratification and management planning are essential downstream tasks in medical imaging. Automating these tasks is challenging because clinical guideline criteria are distributed across prose, tables, and cross-referenced definitions. We present Guideline-derived Decision Graph (GDGraph), a framework that decouples guideline compilation from case evaluation. During compilation, a large language model partitions guidelines into section-based chunks, extracts local decision subgraphs with linked definitions, and aggregates related branches into an executable decision graph. During inference, a fast System One model answers localized clinical questions while deterministic traversal resolves Boolean logic, turning opaque single-pass predictions into a traceable decision path without task-specific training. Across 347 cases spanning renal cyst, adnexal mass, and thyroid nodule guidelines, GDGraph improves System One accuracy from 57.3% to 90.8%, outperforming retrieval-augmented GPT-4o (80.1%) and graph-guided GPT-4o with chain-of-thought (82.7%) at substantially lower latency and cost. The source code will be available. 

---
# How to Loop Your Transformer: Loop Scaling, Specialization, and Parallel Inference 

**Authors**: Chanakya Ekbote, Artidoro Pagnoni, Devendra Singh Sachan, Shruti Bhosale, Mike Lewis, Maryam Fazel-Zarandi, Shashank Rajput, Sujan Kumar Gonugondla  

**Link**: [PDF](https://arxiv.org/pdf/2610.40325)  

**Abstract**: Looping in transformer models reuses weights by repeatedly applying the same transformer, or parts of it, making the model deeper without increasing its parameter count. Whether looping is useful depends on which constraint dominates: model size, training compute, or autoregressive inference efficiency. We study these trade-offs jointly in dense decoder-only transformers trained from scratch, varying which parts of the model are looped, how many loops are applied, and how information is passed between loops. Our experiments span models with up to 1B parameters, trained on up to 155B tokens with up to $1.07\times10^{21}$ FLOPs and 32 loops. We find that looping the full transformer generally gives the strongest quality at a fixed parameter budget, whereas looping a middle block of layers gives a better return when training FLOPs are constrained. Adding loops is not uniformly beneficial: gains eventually saturate, reverse, or become unstable. This motivates asking whether forcing every loop to use the same weights limits the benefit of deeper looping. We address this by adding small loop-specific low-rank adapters that improve quality, with larger gains as more loops are added. Looped transformers also suffer a decoding slowdown proportional to the number of loops. Parallel Loop Transformers address this by passing hidden states between loops across adjacent token positions, allowing loop stages to run concurrently. However, in our experiments the original formulation does not consistently retain the quality gains of ordinary looping. We find that preserving the current-token embedding and separately normalizing it and the previous-token hidden state before combining them recovers strong performance across model scales and training budgets. Together, these results show that effective looping requires matching the design to the resource constraint and inference setting. 

---
# Decision and System One models in rheumatology: accuracy and calibration on specialty-training examination questions and diagnostic case vignettes 

**Authors**: Alfredo Madrid-García, Diego Benavent, Fabian Lechner, Phillip Kremer, Johannes Knitza, Beatriz Merino-Barbancho  

**Link**: [PDF](https://arxiv.org/pdf/2610.40318)  

**Abstract**: Objectives: To evaluate the accuracy and calibration of Decision and System One models on rheumatology examination questions and diagnostic vignettes. Methods: Two benchmarks were analysed: 172 rheumatology questions from Spanish specialty-training examinations (expanded RheumaMIR dataset) in Spanish and English, and 70 English diagnostic vignettes from RheumBench with 5-18 candidate diagnoses, 52 of which included the reference diagnosis. Four System One models (Jev 1.13, Clef, Microsoft-Decision-1, Laya) and GPT-6 Luna (OpenAI Decisions API) were compared with GPT-6 Sol and two local generative models. The primary outcome was accuracy; secondary outcomes included calibration, discrimination, accuracy of answers with probability >=0.90 and language differences. Results: In RheumaMIR, Jev's accuracy was 93.0% in Spanish and 91.9% in English, below GPT-6 Sol (99.4%; Holm-adjusted P<=0.005); GPT-6 Luna reached 97.7%, Clef 91.9-93.0% and Microsoft-Decision-1 86.0-87.8%, without detectable language differences. Jev's answers with probability >=0.90 (76-77% of answers) were 98.5% correct. In RheumBench, accuracy was lower: GPT-6 Sol 76.9%, GPT-6 Luna 63.5%, Clef 59.6%, Jev 50.0% and Microsoft-Decision-1 40.4% (each System One model below GPT-6 Sol, Holm-adjusted P<=0.045; chance 9.6%). Jev became overconfident (mean probability 0.659; over/underconfidence index +0.159), and its answers with probability >=0.90 (25% of cases) were 84.6% correct, against 95.5% for GPT-6 Luna (42%). Laya performed near chance in both benchmarks. Conclusion: The best-performing System One models were highly accurate on examination questions but less accurate on diagnostic vignettes, remaining below GPT-6 Sol. Jev's high-confidence predictions were less reliable on vignettes, highlighting the need for task-specific evaluation and calibration. 

---
# Hammering with Vampire in Lean 

**Authors**: Jonas Bodingbauer, Laura Kovács  

**Link**: [PDF](https://arxiv.org/pdf/2610.40310)  

**Abstract**: We present Vampire-Hammer, a hammer for the Lean interactive theorem prover that discharges goals using the Vampire automated theorem prover and reconstructs proofs checked by the Lean kernel. Vampire-Hammer uses the Lean-auto framework to process goals and premises before dispatching them to Vampire for proof search. Doing so, we rely on standard TPTP problem formats and annotated TSTP proofs that support proof reconstruction. With a 10-second budget on 196,083 eligible Mathlib theorems, Vampire-Hammer solves 21,158 problems not solved by other evaluated tools within the same budget. 

---
# Self-training for cross-centre coronary vessel segmentation in X-ray angiography using pre-contrast frame subtraction 

**Authors**: Danial Moafi, Aida Moafi, Mostafa Aryafar, Ashkan Sami, Maryam Aryafar  

**Link**: [PDF](https://arxiv.org/pdf/2610.40302)  

**Abstract**: Deep learning models for coronary vessel segmentation in X-ray angiography often perform poorly on data from centres that were not represented in training. Apart from differences in equipment, the labels of the largest public dataset (ARCADE) are incomplete, as only the segments defined in the SYNTAX score are annotated. In this study, we evaluated whether self-training with pseudo-labels can improve performance on external data without additional manual annotation. Pseudo-labels were generated from two sources, namely the predictions of an nnU-Net model trained on ARCADE and a contrast map obtained by subtracting a pre-contrast frame from the same sequence. The nnU-Net predictions were extended along the contrast map, and pixels detected only in the subtraction image were assigned an ignore label and excluded from the loss. Six models were trained and tested on ARCADE and on two external datasets (XCAD and XACV). All first-round models initialised with the baseline predictions performed better than both the baseline and a control model fine-tuned for the same number of epochs. The model trained on the XCAD pool improved the macro-averaged Dice score by 2.7 points (95% CI 2.3-3.2), while the control decreased it by 0.6 points. A second round of self-training did not lead to further improvement. When an external model trained on 50 fully annotated images was used to generate the initial labels, the XCAD Dice score increased from 0.623 to 0.745 without any XCAD labels, although performance on XACV did not improve because large main vessels were missed. These results suggest that the benefit of self-training depends largely on the sensitivity of the model used to generate the initial pseudo-labels. 

---
# Only Accepted Tokens Count: An Energy Accounting of Speculative Decoding Under Latency Targets 

**Authors**: Bole Ma, Ayesha Afzal, Jan Eitzinger, Harald Koestler, Gerhard Wellein  

**Link**: [PDF](https://arxiv.org/pdf/2610.40287)  

**Abstract**: Large language model serving is limited by facility power. Energy per token determines how many tokens a datacenter can serve, while a latency target bounds how quickly each user receives them. During decode, every weight is read from memory for every generated token. Batching amortizes this read across users but makes no individual user faster. Speculative decoding does: a drafter proposes several future tokens, and the target model verifies them in a single forward pass that reuses the same weight read. Because every drafted token must be verified whether or not it is accepted, the energy speculation saves depends on how many drafted tokens are accepted. This saving is hard to pin down. It varies across target models from different attention families and across drafting schemes, and it depends on the text being served. Every drafter we evaluate accepts fewer tokens on conversation than on code, and every trained drafter accepts fewer on Romansh, a low-resource language, than on English. We measure speculative decoding under production serving conditions (SGLang, CUDA graphs enabled) on H100 and H200 GPUs, at batch sizes up to 64 across 5 corpora. On dense models, verification, not drafting, dominates the energy cost. The saving shrinks as the batch grows, because batching has already shared the weight read that speculation would divide among the accepted tokens. At large batch sizes, block drafters push verification past the balance point between arithmetic and memory traffic into the compute-bound regime, where the saving becomes a loss, and a drafter with low acceptance raises energy per token. Meeting a tighter latency target with speculation then costs additional energy. Larger batches lower energy per token without speculation. Speculation adds to that saving only when acceptance is high, and on the model we tested it leaves little headroom for latency-preserving DVFS. 

---
# Latent Prediction Yields Cross-Subject Alignable Representations in Sensory-Evoked fMRI 

**Authors**: Serwan Jassim, Victoria Bosch, Daniel Anthes, Elia Bruni, Tim Kietzmann  

**Link**: [PDF](https://arxiv.org/pdf/2610.40280)  

**Abstract**: Decoding sensory-evoked functional magnetic resonance imaging (fMRI) data is a central analytical tool in cognitive neuroscience. Here we ask whether self-supervised learning (SSL), which has been scarcely explored in this domain, can yield representations that support strong downstream performance. To do so, we adapt a joint embedding predictive architecture (JEPA) to train participant-specific encoders on brain data alone. The encoders never see the stimulus images or any representation derived from them. Applied to the Natural Scenes Dataset (NSD), we evaluate our learned brain embeddings on the downstream tasks of brain captioning and image reconstruction, as well as cross-participant generalization. Our self-supervised approach achieves competitive performance with parameter-rich supervised pipelines that are trained end-to-end with access to image and language data. We further find that encoders trained independently for different participants can be aligned fully unsupervised, using neither images nor stimuli shared between them. After alignment, a neural decoding head trained on one participant can be applied to another without fine-tuning and retains most of its within-subject performance on image reconstruction and brain captioning. By removing the need for images and shared stimuli, our approach is a step towards pooling fMRI data across participants and, eventually, across datasets. 

---
# TaskFrame: Connecting Task-Level Planning with Frame-Level Control for Game Agents 

**Authors**: Guangzhi Wang, Junfu Pu, Yiran Wang, Xingyilang Yin, Honghao Chen, Cheng Cheng, Lingen Li, Yinan Zhou, Jiarui Wu, Yinuo Jiang, Hongbo Jin, Xiaodong Cun, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2610.40276)  

**Abstract**: Building general game agents requires both high-level task reasoning and low-latency action control in rapidly changing visual environments. These capabilities favor models with substantially different computational profiles, making them difficult to realize within a single model. In this work, we present TaskFrame, a hierarchical planner-controller framework that connects task-level planning with frame-level keyboard-and-mouse control. A vision-language task planner decomposes a high-level goal into an ordered sequence of atomic instructions and is optimized through supervised learning and dependency-aware GRPO. The hidden states of special goal tokens that terminate each atomic instruction provide continuous goal embeddings that are cached and reused across control steps. A causal Transformer controller combines the embedding of the active goal with recent observations to predict discrete keyboard and mouse-button actions and continuous camera motion at every game step. We train the system in three stages: planner training, controller pretraining, and joint alignment. Experiments in Minecraft show that TaskFrame is competitive with general-purpose vision-language models on instruction decomposition, and that in closed-loop evaluation the agent achieves a higher combat success rate than the evaluated baselines. 

---
# Learned Terminal Costs for Short-Horizon Model Predictive Control of Autonomous Trucks in Interactive Highway Traffic 

**Authors**: Hooman Hamze, Benjamin Houshmand, Deepthi Pathare, Erik Börve, Morteza Haghir Chehreghani, Nikolce Murgovski  

**Link**: [PDF](https://arxiv.org/pdf/2610.40257)  

**Abstract**: Autonomous highway driving for heavy-duty trucks is a challenging control problem. The greater mass, dimensions and intricate dynamics increase the requirements for cooperation from surrounding traffic in critical maneuvers, requiring a careful balance between safety constraints and efficient long-horizon decision making. Model Predictive Control (MPC) provides rigorous constraint satisfaction but relies on terminal cost which can limit its effectiveness at short, computationally affordable horizons. Reinforcement Learning (RL) captures long-term driving behavior but offers no constraint guarantees and tends toward overly conservative policies when trained under sparse collision penalties. In this paper, we propose a hybrid MPC and RL framework for highway driving in autonomous trucks, where a finite-horizon MPC, solved via dynamic programming, uses a pre-trained RL model for terminal cost estimation. We evaluate the framework in a simulation environment with reactive surrounding traffic and compare it against pure MPC with zero terminal cost, rollout with terminal cost estimation using a base policy, and pure RL. Results show that the hybrid approach achieves zero collisions with the lowest cost of any configuration. 

---
# Understanding axial attention in TSFMs through single location regression 

**Authors**: Andrei-Tiberiu Pantea, Aymeric Dieuleveut  

**Link**: [PDF](https://arxiv.org/pdf/2610.40188)  

**Abstract**: Recent advances in Time Series Foundation Models (TSFMs) increasingly rely on axial attention mechanisms, but despite their empirical success, the theoretical properties of axial attention remain largely unexplored. To bridge this gap, we extend the Single Location Regression framework introduced in Marion et al. [2025] to 2D structured data, in order to study sparse information settings where a target depends on a small number of tokens. We formulate an analytically tractable axial attention-type predictor and rigorously analyze its statistical properties and training dynamics. We prove that under a specific asymptotic regime and with knowledge of the underlying oracle parameters, our predictor achieves Bayes optimality, whereas a class of semi-linear simplifications strictly fails to do so, underlining the role of the inner non-linearity. We prove that for a specific temperature scaling, Projected Gradient Descent converges globally to the optimal oracle parameters. Extensive numerical experiments validate our theoretical findings and highlight the critical role of inverse temperature scheduling in empirical convergence. 

---
# EventSeeColor: Learning to Reconstruct Color from Monochrome Events 

**Authors**: Chuanzhi Xu, Langyi Chen, Haoxian Zhou, Zelin Zhang, Haodong Chen, Qiang Qu, Xiaoming Chen, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2610.40166)  

**Abstract**: Event cameras capture visual brightness changes under challenging lighting conditions. However, traditional event-based image reconstruction targets grayscale intensity and produces images without scene color. Color distinguishes surfaces with similar intensity. Reconstructing color from events can reveal appearance differences that reconstructed grayscale images cannot express. Although monochrome event cameras do not directly measure color, their event-triggering patterns depend on surface reflectance, illumination, and the sensor's spectral response. These patterns may contain cues that support color reconstruction. In this paper, we present EventSeeColor, which uses event-reconstructed grayscale and aligned events to jointly guide a diffusion colorizer. Grayscale-derived semantic context supports color inference, while region-balanced chroma supervision and local RGB refinement improve color distribution, brightness, and texture. In the combined evaluation on two real event datasets, EventSeeColor achieves the best structural similarity, learned perceptual scores, and chromatic-region color difference among the external methods. 

---
# FARE: Factor-Aware Retrieval with Error-Guided Fusion for Retail Forecasting 

**Authors**: Chenghao Gu, Zhuokang Shen, Shenghao Wu, Tingxuan Zhu, Weiyi Luo, Haifeng Tang  

**Link**: [PDF](https://arxiv.org/pdf/2610.40160)  

**Abstract**: Retail sales forecasting underpins inventory planning, yet promotions and calendar events reshape short-term demand. Although each factor usually has historical occurrences, the complete factor combination of a target week often has few precedents. The challenge is therefore to retrieve informative partial matches and weight them by their reliability, which varies across series. We propose FARE(Factor-Aware Retrieval with Error-Guided Fusion), which ranks historical weeks separately for each known target-week factor and merges the selected weeks into a factor reference under a fixed budget. Guided by shrunk errors from completed development weeks and the current disagreement among candidates, a first gate mixes this factor reference with a shape reference from similar sales histories, and a second gate controls how strongly the mixture corrects a backbone forecast. Both gates remain fixed during testing. On Favorita, M5, Rossmann, and Rohlik, FARE reduces the weighted absolute percentage error (WAPE) of five backbones by 11.46%, 4.06%, 14.63%, and 10.16% on average. With Chronos-2 frozen, FARE also outperforms conventional and existing enhancement methods, including supervised residual correction and retrieval augmentation. Code is available at this https URL . 

---
# Escaping Path Mirages in Offline Goal-Conditioned Reinforcement Learning 

**Authors**: Junhyeon Bae, Jaebak Hwang, Gwanwoo Choi, Minung Kim, Seungyul Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.40130)  

**Abstract**: Offline goal-conditioned reinforcement learning remains challenging in stochastic long-horizon settings, where compounding value estimation errors hinder reliable goal reaching. While prior methods have sought to address this challenge through various approaches, a key challenge remains path mirage, where agents overcommit to spuriously successful trajectories in offline datasets induced by stochasticity, often leading to failure on difficult tasks. To address this issue, we propose Branch-aware Graph Planning (BGP), which captures stochastic branching structures by identifying high-variability points as graph nodes and designing goal-conditioned edges with segmented learning to avoid unreliable branches and reduce unnecessary stochasticity along paths. As a result, BGP enables reliable goal reaching even in highly stochastic long-horizon environments, and experiments on diverse OGBench tasks show that it substantially outperforms prior state-of-the-art offline goal-conditioned RL methods. Our implementation is available at this https URL. 

---
# Upcycling Language Models for Multiple Specialized Domains 

**Authors**: Mingzi Cao, Xi Wang, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2610.40091)  

**Abstract**: Effective multi-domain adaptation of large language models requires prioritizing relevant domain knowledge while minimizing interference from irrelevant domains. Existing approaches commonly use Low-Rank Adaptation (LoRA) to train a separate adapter for each domain and combine these adapters within one model. However, they either weight adapters equally regardless of the input domain or dynamically weight adapters while still activating all of them including irrelevant ones. Additionally, while efficient, their low-rank structure limits performance. In contrast, upcycling is a model expansion method that adds experts and a router to a dense Transformer layer offering larger capacity than LoRA approaches. In this paper, we propose Domain-Specialized Upcycling (DS-Upcycling) for multi-domain adaptation. We introduce a new domain-aware supervision that allows the router to effectively learn to distinguish domains by selecting the appropriate experts. During fine-tuning on each new domain, we update only the corresponding expert and router while freezing all other parameters. Experiments show that our method achieves the highest average performance across the adapted domains compared to baselines while retaining general performance. With just 256 examples per domain for router supervision, DS-Upcycling achieves 90-98% routing accuracy after adaptation compared with 43-70% for routers trained without our approach. 

---
# Learning Safe Flow Matching Robot Policies from Binary Feasibility Oracles 

**Authors**: Jianming Ma, Rongjun Jin, Yizhi Chen, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.40075)  

**Abstract**: Robot foundation models increasingly rely on flow-based action generation for general-purpose manipulation, yet the generated actions are not guaranteed to satisfy physical constraints such as joint limits and collision avoidance. This problem is especially challenging when the feasible set cannot be described analytically and is accessible only through black-box feasibility oracles. We study this practical oracle-only setting and propose an oracle-guided differentiable safeguard for safe flow-matching robot policies. The safeguard uses binary feasibility queries to correct unsafe updates in the forward pass, while extracting local boundary information to provide constraint-aware learning signals in the backward pass. Combined with discrete-time flow matching, the same safeguard is used during training and inference, enabling forward--backward aligned policy learning without explicit or differentiable constraint models. Experiments on obstacle-aware robotic manipulation demonstrate 100% constraint satisfaction while preserving task performance. 

---
# Localizing Concepts in Visual Autoregressive Models 

**Authors**: Nanxiang Jiang, Yang Liu, Yuanhao Wang, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.40061)  

**Abstract**: Understanding the internal routing of visual concepts is crucial for the safe and controllable adaptation of generative models. While concept localization has been widely studied in diffusion models, the emerging paradigm of next-scale visual autoregressive models remains largely unexplored. In this paper, we introduce LoCo, the first model-agnostic framework to localize where and when specific concepts emerge within autoregressive models. Driven by the native coarse-to-fine nature of next-scale generation, our method precisely maps conceptual knowledge across three distinct dimensions: Layer, Scale, and Position. To systematically localize and evaluate concept routing without context bias, we propose LoCoBench, a comprehensive dataset spanning 10 diverse categories. Extensive probing on image and video autoregressive models like Infinity, HunyuanImage-3.0, and InfinityStar shows that the localized positions are both interpretable and causally related to concept emergence. Building on these insights, we apply our localization method to three key applications: concept erasure, model personalization, and adversarial concept injection. Experiments demonstrate that our targeted intervention achieves state-of-the-art performance, substantially reducing computational overhead while preserving benign utility. Overall, our findings offer insights into how conceptual knowledge is routed during autoregressive generation, introducing a practical pathway for more interpretable, efficient, and secure adaptation. 

---
# One Vector or Many? Clustered Optimal-Transport Steering under Quantization 

**Authors**: Illia Makarov  

**Link**: [PDF](https://arxiv.org/pdf/2610.40055)  

**Abstract**: Activation steering changes what a language model generates by editing its hidden states at inference time. Its most common form adds one vector, the difference of class means (DiM), to every hidden state. CHaRS (concept heterogeneity-aware representation steering) uses optimal transport instead: an entropic plan pairs the clusters of the two classes, and each hidden state moves along a mixture of the cluster displacements, weighted toward its nearest clusters. We study this family of clustered optimal-transport maps for toxicity mitigation in Gemma-2-2B and Granite-3.1-2B, and ask what the plan adds to DiM and whether that survives INT8 and NF4 quantization.
We prove an identity: whatever the plan, the displacements averaged with the cluster sizes as weights equal DiM, so a clustered map can differ from DiM only for tokens whose routing weights depart from the cluster sizes. In the CHaRS protocol they stay close, and CHaRS is statistically equivalent to Mean-AcT, which adds DiM at every layer, in all eight pre-specified equivalence tests. A variant fitted on individual tokens, which routes each token mainly to its nearest clusters, leaves 37-39% less toxicity than DiM on Gemma and 51-61% less on Granite at steering strength 3 (toxic-bert), where DiM already removes 76-85% of it. Two other toxicity classifiers confirm the gain on Granite, but on Gemma only one does. At equal strength the variant has somewhat higher perplexity, but at matched toxicity it is slightly more fluent than DiM. Yet a nearly constant map from the same clusters also leaves about 30% less toxicity, so the role of treating tokens differently remains open.
Under INT8 and NF4 the variant still leaves less toxicity than DiM, and transferring it from full precision to the NF4 model instead of refitting it adds no significant toxicity after correction for multiple tests. 

---
# Rethinking Pre-pretraining 

**Authors**: Alexander Stern, Tim Rocktäschel, Acyr Locatelli, Laura Ruis  

**Link**: [PDF](https://arxiv.org/pdf/2610.40051)  

**Abstract**: We report that allocating compute to synthetic pre-pretraining (PPT) never outperforms pure natural-language training in our compute-matched experiments. This holds across four PPT curricula (k-Dyck, neural cellular automata (NCA), declarative reference corpora, and code) and three training regimes: data-constrained, data-unconstrained, and a high token-to-parameter-ratio regime (881M parameters trained on 50B tokens). When natural language is scarce, repeated epochs over a 100M-token corpus beat every synthetic curriculum on natural-language validation loss, with no consistent downstream accuracy advantage from PPT. We find that PPT gains reported in prior work are sensitive to the weight initialization scheme. In an NCA reproduction, we recover a 5.4% perplexity improvement from PPT under the original initialization scheme. An alternative scheme improves both methods, but the no-PPT baseline more, closing the gap between them. However, when PPT receives additional compute rather than a share of the natural-language budget, gains persist at small natural-language budgets even with improved initialization, but disappear by 1B tokens. Finally, models forget what they learn during PPT in subsequent natural-language training; interleaving the same data throughout training prevents forgetting, but does not significantly improve natural-language performance. Our results suggest limited benefits of synthetic pre-pretraining as a distinct training stage and motivate rethinking how such data are used in language model training. 

---
# EPC-RawDC: Equivariant Physics Constrained 3D Diffusion with Raw Data Consistency for CBCT to CT Image Synthesis 

**Authors**: Alzahra Altalib, Chunhui Li, Alessandro Perelli  

**Link**: [PDF](https://arxiv.org/pdf/2610.40046)  

**Abstract**: Cone beam computed tomography (CBCT) is widely used for image guided and adaptive radiotherapy, but scatter, noise, reconstruction artifacts, and inaccurate Hounsfield units limit its quantitative use. Recent diffusion approaches for CBCT to CT synthesis remain largely image driven, with limited use of the underlying acquisition physics. We propose EPC-RawDC, a conditional 3D latent diffusion framework that combines volumetric generative modeling with two complementary constraints derived from the same scanner specific CBCT forward model. First, geometry equivariant projection constraint exploits the rotational equivariance of the CBCT forward operator by supervising the synthesized CT through correspondingly transformed reference projections. Second, raw data consistency uses noisy projection measurements of the individual acquisition as scan specific supervision during training. The framework was evaluated on two independent clinical cohorts with distinct scanner geometries and projection sources. For JUST, the physics constraints used measured clinical raw CBCT projections acquired with an Elekta system, whereas for SynthRAD2025 the projection data were simulated from the reference CT using a Varian geometry with Poisson photon statistics. EPC-RawDC achieved an SSIM of 0.97 in both cohorts and improved PSNR over DDPM by 2.66 dB on SynthRAD2025 and 2.10 dB on JUST, while reducing mean MAE by 20.3% and 19.0%, respectively. On SynthRAD2025, projection consistency and equivariant errors were reduced by 55.0% and 41.4% relative to DDPM. These results demonstrate that combining a volumetric diffusion prior with scanner geometry and scan specific projection measurements improves quantitative CBCT to CT synthesis across both simulated and measured projection domains. Code available after publication at this https URL. 

---
# UPPAAi: Towards LLM-Assisted Modeling, Verification, Analysis and Synthesis 

**Authors**: Peter Dolog, Kim G. Larsen, Marius Mikučionis, Frederik F. K. Rasmussen  

**Link**: [PDF](https://arxiv.org/pdf/2610.40042)  

**Abstract**: For more than three decades, UPPAAL has supported the modelling and analysis of real-time and cyber-physical systems. Yet building a suitable model, formalising the intended properties, choosing analysis techniques and interpreting counterexamples remain iterative tasks that demand substantial expertise. Large language models (LLMs) can offer plausible explanations and even solutions, but they are not necessarily correct. We present UPPAAI, an extension of UPPAAL in which users construct, explain, debug and specify models in dialogue with an LLM. The guiding principle is that the user requests, the LLM proposes, explains and orchestrates, and UPPAAL verifies. UPPAAI exposes the UPPAAL editor, queries and verifier through a Tools API. We report on seven case studies, from explaining to a newcomer to learning a controller, covering timed automata, stochastic hybrid automata and strategy learning. LLMs proved to be extremely helpful in not just generating solutions but also pro-actively suggesting relevant UPPAAL features it was not asked about. 

---
# HEMO: Harness Self-Evolution for Multi-Objective Combinatorial Optimization 

**Authors**: Jianghan Zhu, Cong Zhang, Ni Zhang, Yifan Yang, Yew Soon Ong, Zhiguang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.40023)  

**Abstract**: Automatic operator design (AOD) has increasingly adopted large language models (LLMs) to generate problem-specific search and variation operators for multi-objective combinatorial optimization (MOCO). Yet most methods evolve the operators while leaving the process that discovers them fixed. This limits adaptation as new Pareto trade-offs and operator interactions change where and how the search should proceed. We introduce HEMO, which makes the AOD harness itself a self-evolving optimization target. HEMO maintains a persistent harness state and updates one module at a time, revising the logic that determines which experience to retrieve, which operators to improve, which parents to select, and how to formulate the orchestration of search steps. Downstream solution quality and runtime guide these updates, while a discounted UCB scheduler directs adaptation toward promising modules as their utility changes. A persistent memory filesystem preserves candidate outcomes and diagnostic feedback across harness states, enabling successive states to build on accumulated optimization experience. Experiments across eight MOCO tasks and two structurally distinct evolutionary backbones demonstrate strong performance against conventional evolutionary, neural, and LLM-based AOD baselines, together with effective transfer to related tasks. HEMO extends automatic operator design beyond evolving individual operators to improving the search process that discovers them. 

---
# Accumulating Pruning Constraints: New Exact Polynomial-Space Algorithms for Max 2-SAT 

**Authors**: Konstantin Kutzkov  

**Link**: [PDF](https://arxiv.org/pdf/2610.39992)  

**Abstract**: We present deterministic polynomial-space algorithms that solve Max 2-SAT on $n$ variables exactly in time $\Ostar(2^{(1-\gamma)n})$, for a fixed $\gamma>0$, on several classes of instances whose clause density may grow with $n$. Our main tool extends the classic DPLL branching algorithm. When the search gives a variable its less promising value, comparing the two values of the variable yields a pruning constraint that is satisfied by some optimal assignment. Accumulating these constraints along each branch prunes all but an exponentially small fraction of the assignments. The classes are formulas with many biased variables, semirandom 2-CNF formulas, and Max-Cut on regular expander graphs and on graphs generated by a stochastic block model. In a semirandom formula, the pairs of variables that may share a clause are arbitrary, and the clauses on these pairs are random. For semirandom formulas and block-model graphs, the algorithms are still exact on every input, and the time bound holds with high probability over the input. To our knowledge, these are the first polynomial-space algorithms with such a running time for nontrivial classes of dense Max 2-SAT instances. Beyond these classes, the pruning constraints can be added to any branching algorithm for Max 2-SAT, and we expect them to be of practical relevance for general formulas, in particular for sparse ones, where each constraint involves only a few variables. 

---
# Mind The Gap (MTG-AD): Dataset and Fine-grained Evaluation for End-to-End Inline Audio Descriptions 

**Authors**: Subhashini Venugopalan, Jenny Tan, Taylor Roper, Jimmy Tobin, Anton Kast, Alicia Martin, Sam Sepah, Amy Pavel  

**Link**: [PDF](https://arxiv.org/pdf/2610.39955)  

**Abstract**: Audio descriptions (AD) are essential for making visual media accessible to blind and low-vision (BLV) users. While Multimodal Large Language Models (MLLMs) offer a scalable solution for generating audio descriptions on-demand, their performance on ``inline'' audio descriptions -- which must fit within existing silences in a video -- remains under-explored, particularly for diverse user-generated content (UGC). We present a comprehensive investigation into MLLM-generated inline audio descriptions. First, we introduce MTG-AD, a professionally annotated dataset of 36 hours of videos, each averaging $\sim$5 mins. and 8k+ descriptions in total, co-created by paired blind and sighted AD professionals to provide a high-quality benchmark for this task. Second, we propose MTG-AD-Eval, an evaluation framework based on audio description expert guidelines that provides fine-grained, reference-free actionable metrics for model improvement. Crucially, our framework evaluates the end-to-end task: it renders the generated scripts into audio tracks to assess their fit within the original video's natural silences, minimizing disruptive overlaps. Our analysis across frontier MLLMs reveals that while models accurately describe visual events and substantially reduce speech overlaps, gaps persist in narrative pacing and flow. Finally, we empirically validate that our automated and reference-free metrics correlate significantly with human quality ratings from sighted raters and 7 BLV participants, finding that while users narrowly prefer human-authored audio descriptions, they find MLLM-generated audio descriptions highly beneficial for comprehension. We provide actionable system design guidance and release MTG-AD at this https URL and MTG-AD-Eval at this https URL. 

---
# When to spend privacy: optimal allocation in private learning 

**Authors**: Umberto Biccari, Yongcun Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.39954)  

**Abstract**: Privacy accounting bounds the cumulative privacy cost of Differentially Private Stochastic Gradient Descent but does not determine how to allocate a fixed budget across iterations. We formulate the complete Gaussian noise-variance schedule as an open-loop control under a composed zero-Concentrated Differential Privacy constraint, explicitly accounting for schedule selection. For expected quadratic terminal tracking, we prove that every local minimizer exhausts the certified budget, yielding an exact-budget simplex formulation. We introduce Privacy-Budget Adjoint Mirror Descent (PB-AMD), combining pathwise adjoint gradients with entropic updates that preserve positivity, charge floors, and the prescribed budget. Under regularity and coercivity assumptions, we establish existence and budget stability of optimal schedules, design-to-training utility stability, and deterministic, stochastic, and sample-average stationarity guarantees. A linear-quadratic reduction yields an explicit square-root allocation law. Experiments demonstrate tracking improvements over several allocation rules and predictive improvements over uniform allocation in a learning-rate restart setting, where learning-rate allocation performs similarly. Comparisons at a common stationarity tolerance show lower schedule-optimization times than projected gradient descent on the tested problems. The framework connects certified privacy accounting with optimization through nonlinear training dynamics. 

---
# Learning Calibrated Confidence for Multi-Agent Collaboration 

**Authors**: Chenxiao Fan, Chongming Gao, Gangyi Zhang, Yaxin Gong, Leyang Shen, Jiakai Wang, Dong Wang, Yang Liu, Haoyan Liu, Xinting Hu, Fuli Feng, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2610.39927)  

**Abstract**: Complex information seeking on the Web often requires collecting evidence from multiple documents and reasoning across them, as in multi-hop question answering. Multi-agent systems address such tasks by dividing work among specialized agents, where upstream agents produce intermediate answers and supporting evidence that downstream agents build upon. However, upstream answers vary in correctness, and most systems provide no quantitative signal about it, leaving the downstream agent to judge whether to trust or verify what it receives based solely on the textual evidence.
In this paper, we propose MASCAL. Each agent learns to produce a correctness probability for its answer by jointly optimizing task performance and confidence calibration in reinforcement learning; the upstream agent passes this probability at handoff, giving the downstream agent a quantitative basis for its decisions, and the final answer also carries a calibrated probability, quantifying the reliability of the system output. To guide the downstream agent in using this signal during training, we introduce confidence-aware weighting, which increases the learning weight at positions responsive to the upstream probability, so that the agent learns more from decisions where it actively uses the signal. Across three collaboration architectures spanning Web retrieval and reasoning tasks, MASCAL improves task accuracy and calibration quality for both agents, and generalizes to deeper collaboration chains and different model scales. Intervention analysis confirms that upstream confidence affects downstream decision behavior, and the learned final confidence supports selective answering and targeted rechecking. 

---
# Spectral Invariant Fusion and Refinement for Domain Generalized Semantic Segmentation 

**Authors**: Yansong Bai, Yunzhi Zhuge, Xinzhuo Yu, Lu Zhang, Jin Zhan, Pingping Zhang, Huchuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.39926)  

**Abstract**: Domain generalized semantic segmentation is challenging because it requires accurate dense prediction on unseen domains without any access to target-domain data during training. Existing methods improve transferability through data diversification, invariant learning, or direct use of foundation models, but they often struggle to simultaneously suppress domain-sensitive interference and preserve fine-grained spatial cues. In this work, we propose SIFR, a frequency-aware dual-stream framework that tackles domain shift at both the fusion and decoding stages. SIFR uses spectral modeling as a unified principle to regulate both heterogeneous prior interaction and dense decoder refinement. Instead of directly fusing frozen CLIP and DINOv2 features, SIFR performs spectral-guided invariant fusion to filter out domain-sensitive responses while preserving complementary semantic and structural information. In addition, SIFR incorporates a spectral decoder refinement module that stabilizes multi-scale decoder features through low-frequency statistics-based normalization and adaptive frequency selection, thereby reducing the propagation of residual domain bias in dense prediction. This design enables SIFR to jointly improve domain robustness and segmentation precision under severe distribution shifts. Extensive experiments on multiple DGSS benchmarks, including synthetic-to-real, real-to-real, and adverse-weather settings, demonstrate consistent improvements over previous state-of-the-art methods. 

---
# Prompts and probabilities in the linguistic assessment of language models: Evidence from semantic focus-sensitivity 

**Authors**: Robert Pasternak  

**Link**: [PDF](https://arxiv.org/pdf/2610.39923)  

**Abstract**: As language models (LMs) have improved, an increasingly common means of evaluating their linguistic competence is through explicit prompting (e.g., "Which sentence is more acceptable?"). Prior work indicates that such prompting underperforms compared to standard token-probability-based methods in smaller models, but improves in larger models. We test this in a new domain: semantic focus-sensitivity, responsible for contrasts like "Lucy only SWAM yesterday" vs. "Lucy only swam YESTERDAY". Testing models of various sizes using text-based focus markers like boldface and italics, we find universal above-chance performance for token probabilities. Prompting results mostly align with previous findings, but in some larger models they outperform probabilities by a wide margin. We propose that this is because LMs are not prone to produce out-of-the-blue focus marking, which muddies results from token probabilities in a way that does not affect prompting. We test this by replacing standard focus markers with randomly generated ones; token probabilities still beat chance but show significantly greater performance decrease compared to prompting. These results suggest that metalinguistic prompting, while non-trivial for LMs, can also be more robust when testing stylistically atypical data. We argue that this has important consequences for the assessment of LMs' language competence, even if LMs do not perform metalinguistic judgments through human-like introspection. 

---
# CovFedAvg: Local Low-Loss Curves and Federated Model Selection 

**Authors**: Federico Lucchetti, Marcus Völp  

**Link**: [PDF](https://arxiv.org/pdf/2610.39902)  

**Abstract**: Clients observing different parts of a prediction task may learn locally equivalent models with incompatible behavior outside their observed support. We study whether redundant models permit local low-loss paths toward a compatible shared model. CovFedAvg retains local models and curve controls on clients, fits a common endpoint through aggregated gradient contributions, and selects an actual candidate model using distributed training losses. No centralized calibration examples or client-to-client sample exchange are required. Exact selection guarantees no higher evaluated training loss than the previous model and the current-round FedAvg candidate; this is not a global-optimality, stationarity, generalization, or differential-privacy guarantee. We state conditional client-compatibility and approximate-selection bounds and retain corrected transformation identities and a separate full-gradient safeguard theorem for reference algorithms. A complementary input-support example recovers the shared absolute-value regression, while matched direct fine-tuning also solves it. Across a synthetic support task, recorded activity-recognition subjects, and participating and future gas-sensor batches, we evaluate five methods, five seeds, two network widths, and 20 rounds without calibration data. At width 16, CovFedAvg raises participating-Gas macro balanced accuracy from FedAvg's 88.88% to 89.84%; matched distributed fine-tuning reaches 91.65%. Results support feasibility and modest gains over FedAvg, but do not establish a curve-specific advantage over equally costly optimization. Exact-sum numerical experiments establish data flow; end-to-end privacy requires additional mechanisms and analysis. 

---
# Turing LLM Post-Training: Calibrated Distillation and Efficient Dynamic-MoE Execution 

**Authors**: Yue He, Jiahui Hu, Shaman Tang, Hua Zhou, Qiman Wu, Menglin Li, Yizhao Wang, Yuheng Zhang, Da Zhu, Zhou Wang, Anhua Liu, Hanlin Chen, Mingyuan Wang, Rui Xin, Yuhua Wei, Shuang Su, Siqi Liu, Kai Yang, Weichao Huang, Hang Zhang, Honggou Yang, Xianming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.39876)  

**Abstract**: We study post-training for dynamically routed mixture-of-experts (MoE) models, focusing on integrating independently optimized specialists while keeping consolidation efficient. Our approach combines calibrated multi-teacher on-policy distillation (C-MOPD), which matches each selected teacher on a local Top-$K$ support with teacher-wise loss normalization and clipping, with an execution path that uses post-capacity bounded-width dispatch, final-route replay, and asynchronous rollout--training overlap. On Turing-20B-A2B, the selected model achieves a four-domain normalized integration score of 0.9988 and retains at least 90.16\% of the measured specialist gain in every domain. At matched $K=32$, the full calibration objective scores 0.9962, compared with 0.9724 without calibration and 0.9834 without clipping. At a matched 8K response cap, the model exceeds both shared SFT and Qwen3-8B on all four reasoning-enabled mathematics and code benchmarks (unweighted mean: 85.92, versus 80.33 for Qwen3-8B and 74.21 for SFT), and improves over SFT on ten of twelve broader evaluation metrics. In a PPU-based profile, the combined execution configuration reduces consolidation step time from 398.7 to 139.3 seconds, an observed 2.86$\times$ speedup over serial execution. These results demonstrate strong specialist integration alongside more efficient consolidation, while reflecting the complete training and execution framework rather than any single component. 

---
# Incorporating Machine-Others: An Enactive Exploration of Human-LLM Interactions 

**Authors**: Elena C. Cuffari, Marta Pérez-Verdugo, Xabier E. Barandiaran  

**Link**: [PDF](https://arxiv.org/pdf/2610.39869)  

**Abstract**: New concepts based in interactional thinking are required to understand how humans are changed by relating to LLMs. Standard theories of technological impact upon users' autonomy are inadequate here, because this technology is capable of inducing linguistic interactions with humans in an unprecedented manner. Traditional theories of social cognition also fall short, because LLMs have neither autonomy nor minds nor sociality as such. And yet, humans interact, talk, and create with them. We base ourselves on enactive theory, particularly on the theory of Linguistic Bodies and recent developments of enactive accounts of AI, to explore how we make sense with machines. We argue that we incorporate machine-others in two ways: in the tool-like sense of incorporation as enabling particular skills, but also in the more linguistic and identity-tied sense of incorporating (and incarnating) LLMs' ways of languaging. However, LLMs are themselves excorporated and excarnated phantoms of our own languaging; we give a (machinic) body to our language. As such, we are not just incorporating machinic others; we are incorporating excorporated others. These conceptual advances draw a new map of our linguistic relation to LLMs, putting them in continuation with other forms of languaging involving other linguistic artifacts (e.g., religious books). We attend to some particular forms that this relation takes today, and how they may be impacting the autonomy and becoming of LLM-engaged humans. We introduce the idea of "susceptive sense-making" to characterize increased semiotic labour that is carried by the lack of friction that we sometimes encounter within these services. 

---
# LiDAR-SHIELD: A Fully Dockerized Adversarial Robustness Assessment Framework for LiDAR-Based 3D Perception Models 

**Authors**: Alexandros Gkillas, Ioulia Kapsali, Christos Anagnostopoulos, Aris S. Lalos  

**Link**: [PDF](https://arxiv.org/pdf/2610.39848)  

**Abstract**: LiDAR-based 3D perception underpins autonomous driving and robotics, yet deploying these models safely requires stress-testing them against adversarial inputs before real-world use. Although state-of-the-art adversarial attacks exist, their implementations are often fragmented across individual repositories, each tied to a specific architecture or task and requiring manual environment reconstruction and model-specific integration code. We present LIDARSHIELD, a unified and fully Dockerized framework that reverses this workflow: rather than installing the target model inside the toolkit, LIDARSHIELD exposes any model through three lightweight HTTP endpoints, namely /info, /forward, and /gradient As a result, connecting a new model or dataset requires only a single loader function or adapter. The framework supports both 3D semantic segmentation and 3D object detection through the same engine-service protocol, integrating a task-aware suite of state-of-the-art attacks, including PGD and Square Attack for segmentation, and point perturbation, point removal, point injection, and LiDAttack for detection. This enables systematic exposure of corner cases and vulnerabilities across heterogeneous LiDAR backbones. The LiDAR-Shield codebase is publicly available at this https URL. 

---
# Exploring the Potential of Symbolic Regression in Precision Livestock Farming 

**Authors**: Elisabeth Mayrhuber, Maciej Oczak, Michael Iwersen, David Brunner, Viktoria Dorfer, Stephan Winkler  

**Link**: [PDF](https://arxiv.org/pdf/2610.39839)  

**Abstract**: The ability to predict farrowing time in pigs is crucial for optimal farm management and ensuring piglet survival. This study investigates the potential of explainable artificial intelligence for precision livestock farming, focusing on acceleration sensor data from sows around the time of farrowing. Accurately predicting farrowing allows farmers to balance both, piglet safety by avoiding piglet crushing events and the mother sow's welfare, by avoiding early crating. Traditional machine learning models often lack interpretability (``closed-box'' problem). Explainable artificial intelligence offers a solution by revealing the decision-making processes within the models. This work is split into two stages. First the detection of nest building behaviour and second the prediction of the remaining time to farrowing. The nest building detection was possible for 166 out of 180 sows and successful within 48 hours before farrowing for 149 sows. The nest building detection is purely based on the change of a sow's activity level and uses no machine learning techniques. Further, this study utilizes symbolic regression, an open-box modelling technique that generates interpretable mathematical formulas, to predict the remaining time until the onset of farrowing. The results are compared with closed-box models, like support vector regression and artificial neural networks, for both explainability and performance. Notably, symbolic regression achieved a mean absolute error of 9.28 hours for the time-to-farrowing prediction, outperforming the closed-box models in terms of performance and interpretability. This research demonstrates that explainable artificial intelligence, specifically symbolic regression, can offer both interpretable results and competitive performance in precision livestock farming. 

---
# SHPRR: An Algorithm for Pure Exploration under Constrained Regret 

**Authors**: Georgina Nouli, Flore Sentenac, Maximilian Schiffer  

**Link**: [PDF](https://arxiv.org/pdf/2610.39837)  

**Abstract**: Best Arm Identification (BAI) conventionally measures the cost of experimentation by the number of samples collected. In many operational settings, however, the cost of exploration relative to a trusted baseline already in place is equally important. We study BAI under both a fixed sampling horizon and a regret budget. Unlike generic resources considered in constrained pure exploration, regret has a distinctive structure: sampling better arms is less costly relative to the baseline. We theoretically show that the regret cost of exploration decreases across the phases of Sequential Halving, a widely used pure-exploration algorithm. This observation yields improved guarantees on the probability of identifying the best arm and motivates a more effective allocation of the regret budget. We propose Sequential Halving with Proportional Regret Rationing (SHPRR), which front-loads the regret budget to allow more exploration when many arms remain. Experiments on synthetic and real-world instances show that SHPRR improves best-arm identification performance over uniform regret allocation and existing baselines. 

---
# Aligning Medical Vision Language Models at the Capability Boundary with Entropy Adaptive Rubric Grounding 

**Authors**: Hongyu Bai, Jiaqi Zeng, Chunzheng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.39835)  

**Abstract**: Medical diagnostic reasoning requires verifiable links between visual evidence and clinical knowledge, yet current medical vision language models are aligned with objectives that reward the wrong behavior. Uniform supervised fine tuning lets easy cases dominate updates while clinically critical boundary cases saturate prematurely. Sparse rewards based only on final-answer correctness cannot assess anatomical grounding or clinical logic, inviting reward hacking through answer shortcuts. We propose MedCBR, a two-stage alignment framework coupling boundary-aware preference optimization with rubric-grounded reinforcement learning. First, we construct counterfactual boundary preferences at the model's capability frontier using empirical pass rates and preference pairs that share reasoning prefixes but diverge at decisive diagnostic branches. Difficulty Weighted Boundary Preference Optimization (DBPO) concentrates learning on this calibrated boundary region. Second, we distill reusable clinical rubrics from boundary trajectories and activate them according to prediction entropy. Entropy Gated Grounded Policy Optimization (EGPO) rewards answer correctness, anatomical grounding, and reasoning fidelity while suppressing rubric noise on reliable samples. A reference policy anchored to the boundary-aligned model couples population-level boundary priors with instance-level uncertainty, making diagnostic reasoning more accurate and clinically grounded. Experiments on six medical benchmarks show consistent improvements over strong general, specialized, and reinforcement learning baselines, reaching 61.2% average accuracy with the largest gains on challenging differential diagnosis tasks. Ablations show super-additive gains: combining the two stages exceeds the sum of their individual improvements. 

---
# InferD: A Honeypot Framework for AI Inference Infrastructure 

**Authors**: Hicham Hammouchi, Gabriele Lenzini  

**Link**: [PDF](https://arxiv.org/pdf/2610.39807)  

**Abstract**: Large language model serving stacks, commercial inference APIs, vector databases, and agent orchestration gateways are now ex- posed on the public internet and indexed by commodity scanners within hours of deployment. The threat characteristics of this sur- face are only beginning to be measured, and the honeypot frame- works that could measure it were designed for conventional net- work services rather than for AI infrastructure. In fact, emulating an AI service is a distinct deception problem: the observable behavior of the service, including model inventories, token formats, tool-call protocols, and authorization flows, must be reproduced faithfully, while the underlying computation, model inference and tool ex- ecution, must never occur. We present InferD, a low-interaction honeypot framework organized around a single principle we call deception without inference: every request an adversary can observe is answered from static response pools, and nothing an adversary submits is ever executed. InferD decomposes the emulation into four deception planes, transport, service protocol, vulnerability surface, and agentic session, adds a provenance plane that links a planted credential to evidence of downstream model-mediated processing. 

---
# Multi-output LSTM soft sensors for industrial anaerobic co-digestion using LSTM-beta-VAE generative data augmentation 

**Authors**: Rohit Murali, Benaissa Dekhici, Tao Chen, Dongda Zhang, Michael Short  

**Link**: [PDF](https://arxiv.org/pdf/2610.39768)  

**Abstract**: Anaerobic digestion (AD) requires close monitoring of key process variables, including pH, the ratio of volatile organic acids to alkaline buffer capacity and biogas production, to maintain biological stability and prevent process failure. While mechanistic frameworks such as the Anaerobic Digestion Model No. 1 are highly parameterised, computationally expensive, and difficult to calibrate for real-time applications, data-driven Long Short-Term Memory (LSTM) networks offer a potentially more adaptable alternative. However, current data-driven approaches in AD are predominantly limited to single-output architectures that focus solely on production yields, as opposed to simultaneous prediction of vital stability indicators. Furthermore, industrial AD datasets are typically sparse and skewed toward steady-state operations. Baseline evaluations demonstrated this scarcity prevents standard algorithms (Random Forest, XGBoost) and unaugmented LSTMs from recognising the transition to unstable operations, resulting in poor estimation of stability indicators. To overcome these challenges, this study proposes a novel generative data augmentation framework using an LSTM-beta-Variational Autoencoder coupled with a multi-output LSTM (MO-LSTM) soft sensor. By synthesising physically valid sequences of these transitions to unstable operations from one year of full-scale plant data, the method safely populates sparse regions of the feature space. Trained on this augmented dataset, the MO-LSTM achieved robust simultaneous estimation across all targets. This approach mitigates data sparsity, providing a basis for hybrid mechanistic-data-driven modelling and early-warning soft sensors in industrial bioprocesses. 

---
# The Importance of Speech Genre for End-of-Turn Detector Training 

**Authors**: Rini Sharon, Kadri Hacioglu, Jeena Prakash, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2610.39742)  

**Abstract**: End-of-turn (EoT) detection has been studied mostly through features, models, and modalities, while the speech genre of the training data has drawn far less attention. We isolate this factor. Using one reproducible annotation pipeline, we build a matched pair of corpora, multi-turn human-human conversational and single-turn human-machine, and train the same streaming ConvLSTM EoT detector on each. Across four test sets the conversational model reaches up to +0.45 higher F1, driven by fewer false alarms on the mid-turn pauses and hesitations that are rare in single-turn corpora. This conversational-training effect is not specific to frame-level decoding. Under a matched complete-or-not test it also appears at the utterance level, across two encoders, but only on conversational test data. Testing only on single-turn speech, as most prior work does, therefore hides the benefit of conversational training data. Against open-source baselines, our 1.8M-parameter model attains the highest F1 on every test set. 

---
# Sandman: Building and Operating Sandboxes for Agentic Reinforcement Learning on Shared Kubernetes 

**Authors**: Yiming Yan, Hongyang Wang, Wenbo Chen, Xuan Wang, Jingguo Liu, Xian Li, Chen Qi, Leo Fan, Siwei Li, Qianli Wu, Yichuan Li, Ruochen Wen, Qian Qu, Zheshen Wang, Benoit Dumoulin  

**Link**: [PDF](https://arxiv.org/pdf/2610.39711)  

**Abstract**: Agentic reinforcement learning (RL) often requires repeated tool calls in task-specific environments. A training job may create thousands of these environments in bursts while sharing a cluster with other jobs. We present Sandman, a sandbox service built and operated at Amazon for agent tool execution and evaluation. It serves multiple production agentic RL workloads on shared Kubernetes. A training framework supplies a task image and uses commands and file operations in the returned sandbox. Files and processes persist across calls until the framework releases the environment. Sandman manages creation, readiness and cleanup. Each sandbox is a Kubernetes pod with a small daemon mounted into the unmodified image. In a scale test, Sandman held 11,858 concurrent tiny sandboxes. Four production SWE-bench Verified training campaigns created 3.1 million sandboxes in October 2026. We report the design trade-offs and operational lessons from handling creation bursts, preparing task environments, reclaiming abandoned sandboxes, and running evaluation under controlled conditions. We will open-source Sandman's daemon, provider and pool. 

---
# Practical Assessment of Agentic Systems in Acoustics 

**Authors**: Arne Hildenbrand, Philipp Diel, Zhuo Cao, Hanno Scharr, Stefan Pischinger, Steffen Marburg  

**Link**: [PDF](https://arxiv.org/pdf/2610.39685)  

**Abstract**: In engineering, applications of artificial intelligence have so far been largely limited to surrogate models and optimization tasks. Large language models changed this, as they can write software from a task given in words. Placed inside a harness, a system of components that lets it search for information and execute software, such a model becomes an agent that can take over recurring engineering tasks. This work measures how the model and the elements of the harness shape the performance and behavior of the agent in two common acoustic applications. In the first, the agent detects and locates faults in audio recordings from a vehicle interior with near-perfect accuracy. A benchmark on this task shows that a weak model sets a limit that no harness design can overcome, and that design choices take effect only once the model is capable enough to use them. In the second, a case study on the design of a metamaterial unit cell, the agent reliably finds designs that achieve a specified insertion loss over a given frequency range. The results aim to help the reader understand what each element of an agent contributes and how to design an agent for acoustic tasks. 

---
# Heterogeneous Co-Evolution Learning for Tool-Efficient Search Agents with Zero Labels 

**Authors**: Qin-Wen Luo, Sheng Ren, Xiang Chen, Rui Liu, Jun Fang, Naiqiang Tan, Jun Wang, Hua Chai, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.39682)  

**Abstract**: Deep search agents extend language models by interleaving reasoning with external retrieval. However, effective search-agent training requires addressing two challenges beyond final-answer optimization. First, the capability frontier of agents is largely determined by training questions, whereas fixed or labeled task collections cannot adapt as agents improve. Second, outcome-only reinforcement learning treats all successful trajectories equally, even when some correct answers rely on redundant searches, creating an inherent tension between answer accuracy and tool efficiency. To address these issues, we introduce \textsc{ZEST-Search}, a zero-label and tool-efficient framework for search-agent training via heterogeneous co-evolution. Starting from unlabeled reference questions, \textsc{ZEST-Search} constructs validated search tasks online through an answer-first and evidence-path-guided generation process. A heterogeneous proposer adaptively constructs training tasks around the solver's evolving capability boundary, while the solver learns search-augmented reasoning through reinforcement learning. To jointly optimize task difficulty and search efficiency, \textsc{ZEST-Search} employs an intermediate-difficulty reward for the proposer and a task-local tool-aware reward for the solver, encouraging informative tasks and correct answers with fewer redundant searches. Experiments on seven general-domain and multi-hop question answering benchmarks demonstrate that \textsc{ZEST-Search} consistently improves search-agent performance while achieving higher tool efficiency without labeled search-training data or expert search trajectories. Further analyses show that heterogeneous generation, adaptive difficulty control, and tool-aware optimization enable more effective search-agent training. 

---
# Can AI Agents Interface Clinicians and Complex Models? 

**Authors**: Giulio Colombini, Matteo Falcioni, Mario Massimo, Francesco Durazzi, Marianna Santoro, Paolo Tubertini, Mirko Degli Esposti, Armando Bazzani  

**Link**: [PDF](https://arxiv.org/pdf/2610.39681)  

**Abstract**: Background: Contemporary healthcare institutions generate daily very large amounts of data, which contain highly useful information for the improvement of work organization, planning and overall efficiency. While models and techniques to extract these insights are well known and constantly developed in mathematical biology, complex systems physics and data science, the adoption of these techniques among Clinicians and Hospital Administrators remains limited, mainly due to language and methodological divides.
Objective: In this article we implement an AI Multi Agent System (MAS) that allows Users to use predictive models and make statistical analyses via natural language conversations.
Methods: We developed EpidemIA, a training-free MAS endowed with a modular suite of epidemiological models written by human experts, executable on a single-GPU workstation. To avoid relaying sensitive data to third parties, the system uses locally an open-weights LLM as backbone for its two subagents.
Results: Using a dataset of Emergency Department visits provided to us by IRCCS Sant'Orsola (Bologna) we test the model on a few typical use cases, demonstrating the possibility of performing both simpler statistical analyses and visualizations as well as more complicated epidemiological predictions based on dynamical models, through simple queries made in natural language.
Conclusions: While still being a proof-of-concept implementation, EpidemIA showcases the transformative capabilities of LLM based technologies as an enabling factor in the widespread adoption of complex dynamical models, which is at present hindered by their steep learning curve. 

---
# Event-SAMA: Language-Grounded Video Object Segmentation on Neuromorphic Event Streams 

**Authors**: Sergei Fedorchenko, Max Zehnder, Nikola Zubić, Rong Zou, Davide Scaramuzza  

**Link**: [PDF](https://arxiv.org/pdf/2610.39676)  

**Abstract**: Event cameras offer microsecond temporal resolution, high dynamic range and no motion blur, but referring video object segmentation (RVOS) lacks event-stream datasets and models. We introduce Event-SAMA-239K, a synthetic-event conversion of the SAMA-239K grounded video-chat corpus and all additional SAMA training data: 30,650 video clips and 392,162 images. Optical-flow, FILM or saccade upsampling followed by a linear event generator preserves all mask, referring-expression and dialogue annotations. We also introduce Event-SAMA, adapting SAMA-4B with a two-channel polarity patch embedding initialised from pretrained RGB weights and a zero-initialised event adapter. End-to-end training yields 24.37 $\mathcal{J}\&\mathcal{F}$ on MeViS valid_u and 22.33 $\mathcal{J}\&\mathcal{F}$ on the converted ReVOS valid subset, establishing the first event-native RVOS baseline (RGB SAMA-4B: 55.4 on MeViS). Frozen RGB SAMA-4B with pseudo-RGB renderings or E2VID reconstructions of the same events scores 13.23 and 15.94 $\mathcal{J}\&\mathcal{F}$. Without retraining, Event-SAMA works on real event-camera recordings when the accumulation window matches their event density. A seed-controlled ablation with a measured noise floor of 0.47 $\mathcal{J}\&\mathcal{F}$ finds gains from diverse training sources over MeViS alone (+4.99), RGB-mean event-stem initialisation (+4.33), and a trainable SAM2 mask decoder (+2.76); the event adapter, partial ViT unfreezing and event polarity are within noise at this budget. An earlier, partially converted corpus snapshot trained with the identical recipe scores 1.38 $\mathcal{J}\&\mathcal{F}$ higher on MeViS and 1.12 higher on ReVOS. Both are single runs that also differ in schedule length, so we draw no conclusion about corpus scale. 

---
# Rebuilding a Field's Open-Problem Catalog from the Literature: A Two-Layer Knowledge Graph, and Why Status Must Stay Proposed 

**Authors**: Madhulatha Mandarapu, Sandeep Kunkunuru  

**Link**: [PDF](https://arxiv.org/pdf/2610.39641)  

**Abstract**: A research field records its open problems in surveys, keynotes and community reports. Those are prose, written every few years, and nobody can ask them who works on a problem now or whether it was settled last year. We argue the problems should be data, and we build the system that makes them so: a hand-curated problem layer carried in unchanged, under an evidence layer rebuilt from a bibliographic index, joined by typed, provenance-carrying links in one graph database and queryable from one language. A language model does four jobs -- per-paper extraction, catalog grounding, per-paper status verdict, and idea generation -- each constrained to a candidate set so out-of-set answers are refused rather than stored. We report a study on two curated catalogs, database research (1,053 problems) and AI research (1,000 problems), each with a prior catalog-only graph. The rebuilt graph answers every question the catalog-only graph answered -- 47 of 47 on databases, 34 of 35 on AI -- plus questions the catalog-only graph refused: recent workers, untouched problems, methods tried, and outside papers bearing on a problem (336 problems, 770 works on databases). The capability we most wanted does not survive its own audit. Of 41 solves and refutes verdicts hand-checked against the papers, 9 are correct, 30 partly correct and 2 wrong; no refutes verdict (0 of 34) disproves an open problem, and only 1 of 7 solves closes the central question. We report that failure in full and draw the design conclusion: a status change must be proposed with its evidence and never applied, making the human reviewer part of the architecture. We close with the data-engineering problems this raises: entity resolution across curated and harvested layers, verifying model status claims, freshness under source quotas, and evaluation without ground truth. 

---
# Where Do Vision Language Models Fail at 3D Reasoning? 

**Authors**: Khai Le-Duc, Nguyen Kim Hai Bui, An Ngo, Nam D. Hoang, Yueqi Song, Jiarui Liu, Thai Quoc Hoang, Fang Wu, Chris Ngo, Terry Jingchen Zhang, Mona T. Diab, Ran Xu, Silvio Savarese, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.39639)  

**Abstract**: Vision-language models can get individual judgments right while struggling to recover and consistently use the exact properties of a 3D object. To characterize this gap, we introduce BedBench, a controlled benchmark of thirteen procedurally generated tasks spanning individual judgments and composite answers. Across twelve open-weight and four frontier models, per-color counting exposes a pronounced weakness: humans achieve 93.3% accuracy, compared with 7.2% for the best model. Targeted fine-tuning raises one model's accuracy from 3.0% to 68.6%, but reveals a further difficulty: its answers about the most common colors remain inconsistent with its own per-color counts. Supplying those counts as text improves its judgments, showing that recovering a property does not mean using it consistently. Reasoning, scaling, and added visual cues each improve individual judgments, but none makes exact answers reliable. To locate the difficulty, we probe Qwen3.5-27B: exact per-color counts remain recoverable at the language model's input, yet the model's answers rarely recover them (65.0% for the probe vs. 3.0% for the model). Image-removal controls further show that some apparent successes require no visual evidence. These findings motivate evaluating 3D reasoning by exact correctness, consistent use, and image dependence, rather than accuracy alone. Code and dataset are publicly available: this https URL 

---
# Intrinsic Distribution Matching 

**Authors**: Deyuan Liu, Yansen Han, Kaiwen Zheng, Jun Xie, Jiacheng Liu, Zhenglin Cheng, Peng Sun, Jintao Zhang, Jianguo Li, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.39633)  

**Abstract**: One-step and few-step text-to-image generation must accurately render the details specified in the prompt despite limited opportunities for refinement. Distribution matching distillation (DMD) uses score differences to bring the generated distribution closer to a reference distribution. Following prior work, we estimate this difference with internal velocity branches, avoiding a separate fake-score network. We ask whether an explicit moment constraint improves this learned score supervision. A decomposition of the reverse-KL field into a Gaussian moment component and a non-Gaussian residual identifies statistics that can be constrained explicitly. We introduce Intrinsic Distribution Matching (IDM), which retains the full DMD correction and matches feature means and per-coordinate standard deviations using the generator's exponential-moving-average (EMA) copy, requiring no external feature encoder. Experiments use full-parameter and low-rank adaptation (LoRA) tuning on Z-Image-Turbo 6B and Qwen-Image 20B. At two network evaluations, IDM reaches 94.25% and 90.20% on Chinese and English LongText-Bench, the highest scores among the compared methods at this budget. Controlled ablations show gains from the added moment loss, while generator EMA features perform competitively with an external DINOv2 encoder across the five reported metrics. Code is available at this https URL. 

---
# Data Provenance and Ethics in the Curricula of U.S. Health AI Education Programs 

**Authors**: Raffaele Giancotti, Francesca Santucci, Amama Anjum, Monique de Villa, Rupsa Mitra, Soham Biren Katlariwala, Vishala Mishra, Leo Anthony Celi  

**Link**: [PDF](https://arxiv.org/pdf/2610.39606)  

**Abstract**: Failures of deployed clinical artificial intelligence (AI) systems have often originated in how their training data were generated. Whether the programs that train the health AI workforce teach practitioners to examine data provenance is unknown. We conducted a cross-sectional analysis of the publicly advertised curricula of health AI education programs at U.S. universities, identified through a structured multisource web search between April and July 2026. Two investigators coded coverage of data provenance (structural, dataset limitations, or data management only), ethics, and sustainability; collaborative structure; technical prerequisites; and the disciplinary background and gender of named instructors. The primary outcome was provenance coverage at the structural or dataset-limitations level. Of 43 candidate entries, 36 programs at 25 institutional units in 21 universities were included. Ethics was covered by 29 programs (80.6%; 95% confidence interval [CI], 65.0 to 90.2), data provenance by 8 (22.2%; 95% CI, 11.7 to 38.1), and structural provenance by 6 (16.7%; 95% CI, 7.9 to 31.9). None of the seven master's degree programs met the primary outcome, and no program addressed sustainability. Eighteen programs included team-based work, but only 5 assembled teams across clinical and technical backgrounds. Of 92 named instructors, 76.1% came from computational disciplines and 3.3% from the social sciences or humanities; among the 51 whose gender could be determined, 66.7% were men. U.S. health AI education programs widely taught ethics but rarely taught how clinical data are produced. None addressed the environmental cost of computation, and most instructors came from computational disciplines. 

---
# Red-Teaming Large Language Models in Low-Resource African Languages: A Case Study of Yoruba and Nigerian Pidgin 

**Authors**: Paula Chikaodinaka Anyanwu, Valdini Douglace Lemofouet, Blessing Ngozi Uzor, Ifeoluwa Dorcas Awotunde, Danielle Blanche Kapsa, Tassallah Abdullahi, Clemencia Siro, Idris Abdulmumin, Seid Muhie Yimam, Shamsuddeen Hassan Muhammad  

**Link**: [PDF](https://arxiv.org/pdf/2610.39590)  

**Abstract**: As large language models (LLMs) become more widespread in Africa, their safety in low-resource languages remains largely untested since most red-teaming work focuses on English. This study evaluates six state-of-the-art LLMs using 400 explicit adversarial prompts in Yoruba and Nigerian Pidgin, categorised into four harm categories. The responses were scored by native-speaker annotators and reliability was assessed using Cohen's $\kappa$. While models that rejected up to 69% of English prompts rejected as few as 0% in Yoruba, this was not because they were safer, but because they failed to understand the prompts. Attack success rates increased for most models in the Nigerian languages, with GPT-4o Mini demonstrating the most significant cross-lingual safety discrepancy (+0.23 in Yoruba). When models did engage with adversarial prompts, many of the responses were harmful, with rates of harmful content reaching 0.63 in English and 0.50 in Nigerian Pidgin. The most difficult harm category to detect was cultural stereotypes and discrimination (ASR = 0.44 in Nigerian Pidgin). Comprehension in Yoruba was low across most models, and this directly suppressed apparent attack success without indicating actual safety. Gemini 2.0 Flash was the most consistent model across languages. Despite its explicit training in African languages, Aya 101 had the highest attack success rate in English (0.60) and one of the lowest Yoruba comprehension rates (0.28). These results demonstrate that English-centric safety alignment does not transfer to Nigerian languages, highlighting the urgent need for language-specific safety evaluations. 

---
# Generative artificial intelligence prompting strategies among Chinese university students across academic tasks 

**Authors**: Lei Fan, Fangxue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.39571)  

**Abstract**: Generative artificial intelligence (GenAI) is increasingly embedded in university learning, yet effective human-GenAI interaction depends not only on access to these tools but also on students' ability to formulate, evaluate, and refine prompts for academic tasks. This study examined university students' perceived GenAI prompting capability and reported prompting strategies. Questionnaire data from 234 students were analyzed using descriptive statistics, measurement modelling, scenario comparisons, structural associations, and qualitative coding of optional comments. Students reported using GenAI across diverse academic activities and generally expressed confidence in their prompting capabilities, particularly in providing relevant context and refining prompts following initial outputs. However, prompting strategies showed limited sensitivity to task types: specifying output format was more strongly endorsed in a creative workshop preparation scenario than in a research-oriented assignment, whereas other matched strategies showed no adjusted differences. Students also commonly endorsed both concise instructions and providing all relevant information at the outset, suggesting that effective brevity may depend more on selecting and organizing task-relevant information than on prompt length alone. By contrast, fixed prompt templates received lower endorsement than other prompting strategies. In the adopted factor model, perceived cognition showed the clearest unique association with perceived prompting capability, while open-ended responses highlighted prompt-intent alignment as a common challenge. 

---
# LLM Agent-Based Design of RISC-V CPUs with FPGA-in-the-Loop Evaluation 

**Authors**: Leo Itoh, Junichiro Kadomoto  

**Link**: [PDF](https://arxiv.org/pdf/2610.39570)  

**Abstract**: Long-run RTL simulation of CPU candidates can take hours, while short diagnostic runs may miss performance degradation that emerges later in execution. This evaluation cost can delay design decisions in LLM-agent-based CPU development. We present a development environment in which LLM agents implement CPU candidates on an FPGA and use on-board performance measurements to guide design decisions. We applied it to the human-supervised development of a single-cycle RV32I core into a three-wide out-of-order RV32IM core. The final design reaches 5.61 CoreMark/MHz at 130 MHz on a Artix-7 FPGA. For the same CPU candidate running 8,000 CoreMark iterations, FPGA evaluation, including implementation, was 9.6x faster than single-threaded RTL simulation. In one case, board measurements revealed long-run performance degradation in a candidate that improved in a short diagnostic run. The result guided candidate selection and led to RTL-based root-cause analysis and a fix. 

---
# Raincheck: Deferred Multimodal Prefill for Efficient Multi-Turn MLLM Inference 

**Authors**: Wenjie Huang, Jun Zhang, Hongyaoxing Gu, Haoyang Huang, Tianqi Xu, Xinyue Hao, Xinyi Hu, Xinle Du, Jun Dai, Shuang Ge, Chen Lei, Yue Li, Mingcheng Wan  

**Link**: [PDF](https://arxiv.org/pdf/2610.39562)  

**Abstract**: Multi-turn conversations with multimodal large language models (MLLMs) reuse input context, but long multimodal inputs increase time to first token (TTFT). Query-aware hard pruning reduces prefill tokens but permanently excludes information that later questions may require. We propose Raincheck, a framework for deferred multimodal prefill that decouples initial token selection from later information availability. An existing pruning method selects tokens for initial prefill; Raincheck preserves and progressively prefills unselected tokens in source order using unused compute capacity during decode. Unfinished prefill continues after answer return during available scheduling gaps. Attention visibility control and affine state mapping enable context updates after generation begins in full-attention and hybrid recurrent models. Prefill-decode fusion reduces repeated weight reads. We evaluate Qwen3-VL-8B and Qwen3.5-9B with 30% initial multimodal-token retention. Across both models and four pruning methods, Raincheck recovers on average 77.7% and 60.4% of the quality lost to matched hard pruning on MultiVerse and ConvBench, respectively. On MultiVerse, the SGLang implementation reduces mean first-turn TTFT by up to 24.1% relative to full prefill, with less than 0.5% increase in mean time per output token (TPOT). 

---
# Judging Before Propagation: Task-Graph Contextual Integrity for Multi-Agent Privacy 

**Authors**: Hengyu Wu, Blaise Delattre, Yang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.39532)  

**Abstract**: Multi-agent LLM systems increasingly rely on persistent memory and inter-agent communication. These capabilities create privacy risks that extend beyond individual interactions: a release appropriate in isolation may become inappropriate when combined with information previously disclosed to a recipient or propagated to downstream agents along the planned workflow. Existing Contextual Integrity (CI) mechanisms for agentic systems typically rely on local judgments that assess each disclosure in isolation, with or without regard to the recipient's history, but do not account for how information propagates downstream. However, compliance at the current boundary does not guarantee appropriateness across the workflow. We introduce Task-Graph Contextual Integrity (TG-CI), to our knowledge the first task-graph-native formalization of Contextual Integrity for multi-agent systems. TG-CI represents a workflow as a Multi-Agent Task Graph (MATG) and extends CI from atomic transfers to stateful, graph-level information flows by jointly considering an item's intended propagation scope and recipients' evolving disclosure states. We operationalize TG-CI through propagation-scoped authorization under two complementary policies: Full-Scope, which requires admissibility across the complete downstream scope, and Auto-Region, which confines each item to its maximal admissible region using lightweight frontier gates. We further introduce MAPLE-Bench, a benchmark of 144 cases across four scenarios that jointly vary MATG topology, disclosure history, compromised-agent placement, and item-level privacy policies. In our primary experimental configuration, TG-CI reduces the mean unauthorized disclosure count by 85.8-94.1% relative to the Local-History CI baseline, while retaining 93.6-96.1% of the baseline's task accuracy, with the two enforcement policies providing complementary privacy--cost--utility trade-offs. 

---
# Agent-Wise Test-Time Normalization for Camera-Based Cooperative Perception with Corrupted Collaborators 

**Authors**: Yu Wang, Shaohua Wang, Yicheng Li, Kecheng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.39505)  

**Abstract**: Intermediate-fusion cooperative perception lets a vehicle detect objects beyond its own field of view by fusing features shared by nearby agents. When only a collaborator's cameras are degraded by fog, motion blur or exposure errors, the features it shares drift away from the training distribution while the ego input stays clean, and cooperation can become worse than not cooperating at all. Test-time batch-normalization adaptation, the standard training-free remedy, estimates statistics over the pooled batch of all agents and therefore normalizes both the clean ego and the corrupted collaborator with a mixture of two distributions. We show that this mixing harms even the clean ego: when the ego shares BN statistics with a corrupted collaborator, its detection from its own features drops by 3.58 to 6.68 AP@0.5 points although its cameras are untouched. We propose AgentNorm, an agent-wise test-time normalization that i) estimates the normalization statistics of every collaborator from its own current observation and ii) anchors the ego to its source statistics, so that the ego path remains exactly the original model. AgentNorm has no learnable parameters, needs no gradients or labels, is independent of the order of test frames, and runs locally on each agent without changing the transmitted features. On camera-based OPV2V with four fusion checkpoints and collaborator-only corruptions, AgentNorm raises AP@0.5 under severity-5 fog from 29.42-37.25 to 43.32-58.86 and outperforms pooled normalization by 5.53-12.94 points under fog and motion blur, at an extra latency of 2.3 ms per frame. On the real-world DAIR-V2X-C dataset, it outperforms pooled normalization in all ten infrastructure-corruption settings. We also report the cases it does not fix: occlusion, which removes image content rather than shifting feature statistics, and the case in which all agents share the same shift. 

---
# Soei: Structured Observability of Evidence and Intent for Multi-Agent AI Software Engineering 

**Authors**: Mohammed Nayeem Khan, K S Sarun Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2610.39468)  

**Abstract**: AI coding agents now do long, multi-step and multi-session software work with little supervision. When a developer comes back to that work, the useful question is not what the model said but what was actually done, by whom, and whether it is finished. We study this in three parts. First, we verify Agenttrail, an open-source local observability tool for coding agents, against its source at commit e4ba2da: nine documented constants and mechanisms hold at file-and-line level, two event pipelines are never unified, and its Map service has an unauthenticated local mutation surface. Second, we run a prototype of our system on 45 real coding-agent sessions (102,233 records, 20,577 tool executions, 2,336 human-to-agent episodes). With a blind-audited classifier (precision 83.3% and 96.7%) we estimate that 23.4% (95% CI 17.1 to 35.7) of code-editing episodes end with no independent check after the last edit, and that 57.8% of to-dos the agent marked complete after editing had no proof in between. A 10-minute timing heuristic for session hand-offs recovers 6 of the 24 session pairs that content lineage links. 42% of sessions run longer than 24 hours, 48.1% of transcript bytes lie past a 2 MiB replay bound, and 1,326 records in 16 sessions contain credential-like strings. Third, we propose Soei (Structured Observability of Evidence and Intent): an acknowledgement-aware event schema, a hash-chained local ledger, a W3C PROV-aligned graph, calibrated inference for attribution and lineage, and a verification layer that keeps a claim of completion apart from proof. On the same corpus the ledger is 20% of raw size, holds no content or credentials, verifies at about 126k events/s, detects tampering and replays deterministically. We position the work against concurrent 2026 proposals and define autonomy levels with evidence gates and harness metrics baselined on our data. 

---
# Integrating Heterogeneous Echocardiography AI Outputs in Language Space: Learnable Late Fusion with a Supervised LLM 

**Authors**: Ryo Takizawa, Tempei Kabayama, Chiemi Yamazaki, Shun Kitamura, Ryo Matsuoka, Yuta Ando, Satoshi Kodera, Norihiko Takeda  

**Link**: [PDF](https://arxiv.org/pdf/2610.39464)  

**Abstract**: View- and target-specific echocardiography AI models capture useful but complementary information and may not achieve sufficient predictive performance in isolation. A learned ensemble is a natural way to combine these outputs, but heterogeneous models typically require each output to be converted into a binary decision, probability, confidence score, or fixed-length feature vector. We developed a supervised large language model (LLM)-based late-fusion module that directly aggregates view-specific predicted finding statements and predicted measurements in language space and learns a case-level fusion function. In patient-level three-fold cross-validation of 859 examinations from 646 patients, the proposed method integrated heterogeneous outputs within a single framework, performed comparably to the best-performing reference strategy evaluated, and outperformed a general-purpose LLM without task-specific training. These results suggest that a supervised LLM can combine multiple upstream models without model-specific output conversion or feature engineering that requires medical knowledge. 

---
# Sparse Fine-Tuning for Parameter-Efficient Adversarial Training 

**Authors**: Zhaoxin Wang, Zihang Ding, Handing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.39455)  

**Abstract**: Deep neural networks achieve strong performance across many tasks but remain vulnerable to adversarial perturbations. Adversarial training (AT) is one of the most effective defenses, yet it suffers from high memory cost on large models and the persistent clean-robust accuracy trade-off. Recent work introduces parameter-efficient fine-tuning, such as LoRA, into AT, but this constraint induces a notable robustness gap relative to full-parameter AT. In this work, we revisit parameter-efficient adversarial fine-tuning from a parameter space perspective. Through gradient analysis, we find that adversarial optimization is highly concentrated on a small subset of parameters, which we call Robustness-Critical Parameters (RCPs), suggesting that robustness is encoded in a sparse subspace. Building on this observation, we propose Sparse Adversarial Fine-Tuning (SAFT), which identifies RCPs via adversarial gradient saliency and performs AT by updating only these parameters while freezing the rest. Across architectures and datasets, SAFT outperforms LoRA-based methods in both clean and robust accuracy with fewer trainable parameters, and approaches or surpasses full parameter AT in robustness while updating only about 5\% of parameters. 

---
# FDSML: Formula-Driven Supervised Metric Learning with Parameter-Aware Triplet Loss 

**Authors**: Hina Otake, Shinei Arakawa, Keitaro Tanaka, Yoshihiro Fukuhara, Hirokatsu Kataoka, Shigeo Morishima  

**Link**: [PDF](https://arxiv.org/pdf/2610.39423)  

**Abstract**: Formula-Driven Supervised Learning (FDSL) enables vision pre-training without real images, but existing methods cast pre-training as classification on discretized labels, discarding continuity in the generative process. We propose Formula-Driven Supervised Metric Learning (FDSML), which derives relational supervision from image-generation parameters beyond discrete class labels. FDSML combines FineFractalDB, our refined fractal dataset with higher intra-class variation and more stable structures, with a Parameter-Aware Triplet (PA-Triplet) Loss that assigns positives and negatives according to distances in the underlying Iterated Function System parameter space. This objective trains embeddings to respect parameter-derived relative similarity orderings. Experiments show that FDSML improves transferability to image retrieval while remaining competitive on downstream classification and improving robustness under several corruption settings. On ResNet-50, Recall@1 improves from 65.6% to 74.0% on Stanford Online Products and from 56.2% to 78.4% on Stanford Cars over FDSL pre-training. These results suggest that parameter-derived relational supervision provides a stronger synthetic pre-training signal than label-based classification for transfer beyond recognition. 

---
# Well-Specified and Well-Tested? Assessing SWE-bench-Style Datasets with a Self-Evolving Agent 

**Authors**: Zhonghao Jiang, Xuyang Zheng, Yanfu Yan, Zhongxin Liu, Lingfeng Bao, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2610.39417)  

**Abstract**: Large language models (LLMs) have shown strong performance on software engineering (SE) tasks, driving the need for high-quality datasets for reliable training and evaluation. SWE-bench-style datasets (SWE datasets) are widely used to evaluate LLM-based coding agents on real-world software issues, where task clarity and test coverage directly affect whether instances are solvable and verifiable. However, quality issues persist in existing SWE datasets, and manual annotation is costly and labor-intensive, motivating LLM-based annotation. Existing approaches typically rely on static, overly general rubrics, limiting their ability to handle diverse and nuanced scenarios and often yielding high false positive rates. In contrast, human annotators draw on accumulated experience to make context-aware judgments. Inspired by this, we propose ExLabel, an LLM-based, self-evolving, experience-guided assistant for task clarity and test coverage assessment in SWE datasets. ExLabel maintains an evolving experience bank that distills cross-instance knowledge into stable yet adaptive judgment criteria and reuses relevant experience to guide the assessment of new instances. For systematic evaluation, we construct SWE-TQ, a new benchmark spanning eight programming languages, by cleaning human annotations from SWE-bench and Multi-SWE-bench. Experiments with three backbone LLMs show that ExLabel achieves the best F1 across all settings, with relative improvements of 44.2%/49.9% and 24.0%/18.4% over the best baseline for task clarity and test coverage assessment on the Python/Multilingual subsets, respectively. Moreover, a coding model fine-tuned on SWE-rebench data filtered by ExLabel resolves 66.2% of issues on SWE-bench Verified, outperforming the model trained on data filtered by the best baseline by 14.5%. 

---
# Safe Multi-Objective Reinforcement Learning: Zero Constraint Violation and Primal-Averaging Convergence 

**Authors**: Chenxi Li, Yao Shu, Siyang Gao, Shuang Qiu, Guangxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.39409)  

**Abstract**: Many applications of multi-objective reinforcement learning (MORL) require policies that balance conflicting objectives while satisfying safety constraints. We study safe MORL with zero constraint violation throughout learning, covering both cumulative and stepwise constraints. A key challenge is to explore sufficiently under unknown dynamics and constraint costs without violating the constraints. For a given preference over objectives, we develop an algorithm to learn a safe Pareto optimal policy by combining optimistic reward estimates with pessimistic cost bounds. With high probability, every policy executed over $T$ episodes incurs zero constraint violation. The algorithm returns a safe Markov policy induced by the averaged occupancy measure and achieves an $\widetilde{O}(1/\sqrt{T})$ suboptimality gap under smooth Tchebycheff scalarization. We further develop a preference-free algorithm that first explores safely without a prespecified preference and then computes a safe policy for any given preference without further interaction. Over $T$ exploration episodes and $K$ planning iterations, it retains the same safety guarantee and achieves an $\widetilde{O}(1/\sqrt{T}+1/\sqrt{K})$ suboptimality gap. Both algorithms require knowledge of a policy that strictly satisfies the constraints and its costs. We then relax these requirements, incurring at most constant cumulative violation. To our knowledge, we provide the first sample-efficient MORL algorithms that guarantee safety for every executed policy. 

---
# SymKL: A Unified Representation-Space Measure of Signal Degradation 

**Authors**: Dhananjaya Jayasundara, Sudarshan Rajagopalan, Vishal M. Patel  

**Link**: [PDF](https://arxiv.org/pdf/2610.39403)  

**Abstract**: Quantifying degradation between a pristine signal and its degraded observation is fundamental to signal processing, yet existing full-reference measures are inherently modality-specific. We introduce SymKL, a unified full-reference degradation measure based on implicit neural representations (INRs). An INR is fitted once to the pristine reference and characterized by its diagonal Fisher information, which captures the local information geometry of the learned representation. Evaluating the Fisher information with the pristine reference and a degraded observation yields two parameter-sensitivity profiles whose symmetric KL divergence defines the degradation score. The proposed formulation requires only the reference and degraded observation, without knowledge of the corruption process or degradation severity, making it applicable across images, audio, and 3D data. We further derive an exact closed-form expression for the diagonal Fisher of SIREN, enabling efficient computation. Experiments on image, speech, and volumetric 3D benchmarks demonstrate that SymKL accurately tracks degradation severity, remains monotonic where conventional metrics saturate or fail, and provides a unified modality-agnostic measure of signal degradation. Code will be made publicly available after the review process. 

---
# OPERA: OPamp Design by an Expert-Taught Reasoning Agent, from Specifications to Layout 

**Authors**: Vishal Saxena, Soumyajit Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2610.39395)  

**Abstract**: Analog flows from specifications to a signed-off layout rarely run without manual iteration. Recent large language model (LLM) agents size circuits with optimizer loops over the device dimensions and stop at the schematic. We present OPERA (OPamp design by an Expert-taught Reasoning Agent), in which a general-purpose LLM agent designs CMOS opamps from specifications to post-layout sign-off with open-source tools. First, the designer teaches the agent the gm/ID method: map the specifications to transconductances and minimum currents, choose gm/ID per device role from characterized lookup tables, and size from equations, leaving a few knobs to search and an explainable design. Second, the agent learns the gLayout framework and taught layout practice, and writes a reusable layout generator that closes the layout and extraction loop at designer-approved gates. In SKY130, the agent designed a shared constant-gm bias and six opamps (5T, folded-cascode and telescopic OTAs; Miller, folded-cascode and telescopic two-stage) for fun >= 50 MHz, PM >= 60 degrees, SR 25 V/{\mu}s into 1 pF and a minimum dc gain of 30 to 60 dB. All six layouts are DRC-clean and LVS-matched, and their extracted netlists meet every target at all 15 PVT corners ( fun >= 53.7 MHz, PM >= 62.1 degrees, SR >= 27.9 V/{\mu}s). The benchmark netlists are released at this https URL. 

---
# UMI-Live for In-The-Wild Bimanual Robotics Data Collection 

**Authors**: Govind Velamoor, Adrian Lin, Sam Jung, Spencer Mateega, Brandon Y. Yang, Amish Sethi  

**Link**: [PDF](https://arxiv.org/pdf/2610.39385)  

**Abstract**: Universal Manipulation Interface (UMI) devices allow human experts to demonstrate tasks for robot learning without teleoperating physical robots. Using these demonstrations as training data still requires a consistent hardware setup, calibration, and data processing. We present UMI-Live, a full-stack open-source bimanual UMI system for in-the-wild data collection. We design our custom UMI-Live hardware to integrate camera and LiDAR--inertial sensing, onboard compute, and power, without external tracking equipment or calibration within the collection environment. We implement a simple user interface for recording data while an asynchronous pipeline performs parallel SLAM reconstruction and publishes training-ready datasets. We also release software that supports dual-YAM teleoperation and VLA policy training and inference. We validate UMI-Live data quality by fine-tuning $\pi_{0.5}$ and MolmoAct2, achieving success rates of 10/10 and 8/10, respectively on a bimanual pick-and-place task. Furthermore, we release 40 hours of UMI-Live data across three tasks, and complete hardware designs, software, and reproduction guides at our $\href{this https URL}{\text{project website}}$. 

---
# Learning to Formulate: Agentic Hardware-Software Co-Design of AI Infrastructure through Progressive Optimization and Verification 

**Authors**: Yi Ding, Li Shang  

**Link**: [PDF](https://arxiv.org/pdf/2610.39380)  

**Abstract**: AI infrastructure, the chips and software that train and serve AI models, is designed in layers. Layers make the system manageable, but they also fix which choices an optimizer can consider. Gains that need the chip and its software to change together are excluded before optimization begins: the formulation leaves them out. Language-model agents make it possible to revisit these boundaries, but an agent's proposal cannot show by itself that a design is correct or optimal. We present Formulaic, a flow that divides the work. The agent writes the formulation, a mathematical solver optimizes within it, and verification and measurement check its assumptions and results. The agent refines the problem step by step, from the computation graph toward the chip. Each step is a template: a family of programs with the conditions under which they are correct. Wall relaxation searches the chip by pricing the hardware limits that bind the software. This paper takes a first step, through separate studies rather than one end-to-end run. On NVIDIA GB10, templates the agent wrote run complete Qwen3-8B layers $1.075\times$ faster than SGLang. In a simulated Ventus GPU study, joint design cuts the worst-scenario cycles of Qwen2.5-0.5B to 0.428 of the reference, against 0.934 for software alone. On a five-step GEMM refinement chain, reusing proofs makes checking $2.51\times$ faster. 

---
# What Makes Medical Examination Items Difficult? Using Large Language Models to Explore Patterns Associated With Empirical Item Difficulty 

**Authors**: Xiaomeng Xiong, Marcus Walker, Yanlin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.39373)  

**Abstract**: Research on medical item difficulty has largely focused on prediction rather than on identifying interpretable patterns associated with empirical difficulty. This study used empirical difficulty labels to guide large language models (LLMs) in inducing difficulty rubrics and examined how rubric guidance affected LLM difficulty judgments. Using training items and their empirical difficulty estimates from the United States Medical Licensing Examination (USMLE) and the Physician Assistant National Certifying Examination (PANCE), Claude Opus 5 induced separate rubrics for absolute and pairwise difficulty judgment. None of the induced rubrics treated surface features, such as stem length, option count, or images, as major sources of difficulty. Several patterns emerged across both examinations, including cue convergence, misleading cues, and option competition, although their specific forms differed. Retrieval of knowledge that could not be derived from the stem was particularly prominent in the PANCE rubrics, whereas the USMLE rubrics placed greater emphasis on integrating multiple clinical cues. We compared commercial and open-weight LLMs on held-out items under unguided and rubric-guided conditions. Rubric guidance changed how models used the difficulty scale but did not consistently improve performance. In absolute judgments, guidance encouraged broader use of the scale and generally improved coarse-grained agreement, without consistently improving rank ordering. In pairwise comparisons, its effects varied across models and examinations, and judgments became more sensitive to presentation order. These findings suggest that the value of label-guided rubric induction may lie more in generating interpretable, testable hypotheses from empirical difficulty data than in reliably improving difficulty judgments. 

---
# Cord: Confidence-Anchored Order Residual Decoding for Masked Diffusion Language Models 

**Authors**: Chengwei Zhou, Zhaoyan Jia, Massoud Pedram, Gourav Datta  

**Link**: [PDF](https://arxiv.org/pdf/2610.39372)  

**Abstract**: The decoding order of masked diffusion language models (dLLMs) can substantially affect generation quality, yet common decoders typically apply the same reveal rule throughout generation. Confidence provides a strong default, but it measures how decisively a position can be predicted now, not how useful revealing that position will be for subsequent decoding. We propose Cord, a lightweight prompt-conditioned policy that retains confidence decoding as a stable reference and learns when and how to depart from it through geometric residuals over generation direction, proximity to revealed context, and two-sided infilling. The residuals adapt to both the prompt and the evolving partial response. Rather than sampling a new policy action at every decoding step, Cord samples a single low-dimensional coefficient matrix per rollout, which induces an entire state-dependent reveal trajectory. This yields a contextual-bandit formulation trained directly from sequence-level rewards, without step-level credit assignment or backbone updates. Across three dLLMs and five reasoning, code, and constraint-satisfaction benchmarks, Cord improves average task success from 32.0% under confidence decoding to 48.2%, and further improves over the best per-domain fixed residual. Behavioral audits show that the learned residuals induce substantial, task-dependent changes in reveal locality and two-sided infilling. Cord adds only $0.53$M trainable parameters, requires no additional backbone forward passes, and incurs only 5.39% inference latency overhead. 

---
# Semantic Mismapping in Generalized Category Discovery: Evaluation and Mitigation 

**Authors**: Zitai Wang, Qianqian Xu, Cong Hua, Xi Chen, Zhiyong Yang, Xiaochun Cao, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.39356)  

**Abstract**: Generalized Category Discovery (GCD) aims to perform old-category recognition and novel-category discovery in the same unlabeled set. Early methods address both objectives through clustering and evaluate the exchangeable predicted indices with global Hungarian (GH) matching. Recent methods adopt trainable parametric heads, using supervision for old-category recognition and discovery-oriented objectives for new categories. In this way, the output space becomes partially exchangeable: old indices are semantically anchored, while new indices remain exchangeable. However, these methods still apply global matching, which can reassign anchored old indices to other categories and hide some prediction errors. Empirically, we are surprised to observe that this reassignment can induce an overestimation of new-category accuracy by up to 25.63 points. We term this phenomenon semantic mismapping and further derive a permutation-cycle condition that characterizes when semantic mismapping can increase the global matching score. To fix this issue, we propose Semantics-Preserving Evaluation (SPE), which preserves the supervised semantics of old indices and restricts Hungarian matching to the new region. Furthermore, we propose Mismapping-Mitigating Optimization (MMO) to alleviate the prediction errors that can give rise to semantic mismapping, combining a region-balance loss (RBL) for old/new routing with a novel-diversity loss (NDL) that encourages orthogonality among new prototypes. Theoretically, we show that minimizing the two losses can tighten an upper bound on spurious accuracy gains from semantic mismapping, with controllable residual terms. Empirical results under both GH and SPE demonstrate the effectiveness of our proposed method. 

---
# RetroChessBench: A Benchmark for Reconstructing and Proving Chess Histories 

**Authors**: Tidor-Vlad Pricope  

**Link**: [PDF](https://arxiv.org/pdf/2610.39341)  

**Abstract**: Frontier language models such as GPT 6 Astra have contributed to research mathematics, demonstrating substantial reasoning capabilities. Yet serious reasoning failures persist even in tasks governed by explicit rules. We demonstrate this in retrograde chess which is a playground for pure reasoning, where solvers reconstruct legal histories and distinguish possible events from forced ones. We introduce RetroChessBench, a 12-task benchmark covering exact-length constructions, unique endings, and exhaustive answer sets, with references, rubrics, and a terminal-reward environment. In two trials per task and condition, GPT 6 Astra High fails to produce the required final answer in 22/24 attempts without tools and 18/24 with code execution; mean terminal RL rewards are 14.75% and 27.19%, respectively. The examples identify faulty deductions: failed responses reject legal histories and treat temporary obstructions as permanent. Online retrograde archives often provide puzzles and final move sequences without detailed derivations. We hypothesize that limited exposure to these missing explanations in web-derived training data contributes to the observed reasoning failures. This questions how far frontier mathematical reasoning really generalizes beyond familiar solution patterns. Claims of general intelligence require evidence that models can transfer learnt principles to areas with few worked examples. 

---
# Telling Is Not Seeing: Textual Personas Rarely Make Vision-Language Models Answer Like the People They Describe 

**Authors**: Jiawei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.39331)  

**Abstract**: Vision-language models (VLMs) are increasingly prompted with personas, such as "You have low vision", to stand in for people in user studies. We ask whether a persona makes a model see like the person it describes. We compare persona answers with measured people with low vision, color blindness or hearing loss, and with simple rules that use only the facts the persona states. The persona never beats these rules, and it erases the real differences between groups and individuals. Describing blur leaves recognition unchanged in all 14 models we test, while blurring the photo lowers it as it does for people. Inside open models, the description changes neither the image information nor the answer probabilities. Yet the models know the effects they fail to show: they can estimate how blur and illusions affect people. Telling is not seeing: a persona makes a model say what it knows about people, not see what they see. 

---
# Learning to Bound: Monotonic, Calibrated Neural Upper Bounds for MPE Inference 

**Authors**: Akshay Vyas, Hailiang Dong, Shivvrat Arya, Brij Malhotra, Tahrima Rahman, Vibhav Gogate, Nicholas Ruozzi  

**Link**: [PDF](https://arxiv.org/pdf/2610.39305)  

**Abstract**: Probabilistic graphical models (PGMs) compactly encode structured joint distributions, but Most Probable Explanation (MPE), which finds the highest-probability assignment to unobserved variables, remains computationally intractable in large, densely structured models. Anytime systematic search, such as branch-and-bound (B&B), is a popular approach, but its efficiency depends on tight admissible upper bounds that are often prohibitively expensive to compute using classical methods like Mini-Bucket Elimination (MBE). We introduce Learning to Bound (L2B), a data-driven framework that amortizes bound computation cost by training a neural predictor. L2B targets two key properties required for effective search bounds: admissibility, which is achieved via statistical calibration techniques, and progressive tightening as evidence accumulates, which is enforced through structural architectural constraints that ensure monotonicity by construction. Across challenging benchmarks, L2B delivers tighter, monotonic bounds at orders-of-magnitude lower cost on every PGM family, and yields measurable improvements when integrated with branch-and-bound on high-treewidth instances. 

---
# There Is No Free Lunch in RLVR: Every pass@1 Gain Has a pass@k Price 

**Authors**: Yinuo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.39295)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) makes a language model solve more problems on the first try and fewer when given many tries: it raises pass@1 while the base model overtakes it at large sampling budget k. Is this a failure of RL to add reasoning, or a transient that more training removes? We settle the question with a theorem. The reversal is a price, and we compute its exact rate. Modeling RLVR as an exponential tilt of a base policy over strategies shared across prompts, we prove that under exclusive specialization every update that raises pass@1 strictly lowers the success of some prompt; a strictly positive price on prompts (Stiemke duality) is the exact condition, and being merely the best strategy on a prompt is provably not enough. When the loss reaches the hardest prompts, it forces a pass@k crossover, and the number of crossings is at most the sign variation of the shift in per-prompt success, a totally positive bound that is tight and that recovers the known single-crossover result as its order-preserving case. The crossover is not a bug to patch: a constrained update traces the exact pass@1 versus pass@k frontier, and removing the crossover entirely costs the entire pass@1 gain. The same tilt identity yields entropy collapse and the entropy-performance law with an identifiable exponent, unifying two literatures under one mechanism. Reinforcement cannot widen the reasoning frontier; it can only move along it, at a price we make measurable by a diagnostic that reads the bound, the order-preserving test, and the price from any evaluation log. There is no free lunch, only a price. 

---
# WorldFlow: Joint Video and 3D Motion Prediction via Large-Scale Pretraining 

**Authors**: Yuhao Lu, Jiongxu Chen, Hexiang Wei, Jiajin Wu, Juelin Li, Mingjian Liang, Fan Yang, Zhuowei Liang, Zihao Liu, Zhennan Wang, Sheng Wang, Lin Shao  

**Link**: [PDF](https://arxiv.org/pdf/2610.39278)  

**Abstract**: Predicting future physical states stands as a central challenge in embodied intelligence. Existing video world models leave 3D dynamics implicit, action-based formulations remain tied to specific robot configurations, and 3D motion prediction models use flow-only supervision, providing a limited learning signal for physical dynamics. To circumvent these limitations, we introduce \method{}, a generative world model that jointly predicts future video and query-based 3D trajectories. By parameterizing 3D motion as continuous displacement tokens aligned with the video grid, a unified Mixture-of-Transformers (MoT) backbone jointly denoises both modalities under a rectified-flow objective. This formulation couples explicit geometric dynamics with visual evolution, effectively unlocking the physical commonsense embedded within pretrained video models. To empirically scale and validate this unified architecture, we train \method{} on a heterogeneous mixture of 1.44 million object-level samples spanning simulated and real-world domains. We further introduce WorldFlowBench, comprising 1,678 evaluation samples from 11 data sources across simulated and real-world interactions, including spatial reasoning tasks. Across all three evaluation sets, WorldFlow achieves the lowest average displacement error (ADE) and final displacement error (FDE) among the evaluated learning-based methods. Notably, on the challenging SpatialCog set, WorldFlow achieves an ADE of 0.060 m, a 54.2\% reduction compared with the strongest external baseline. It also reduces ADE and FDE by 55.9\% and 68.5\%, respectively, compared with its flow-only variant. Project page: this https URL. 

---
# DeCoRL: Decomposed Consistency Reinforcement Learning for Multi-Image Composition 

**Authors**: Zhiqiang Wu, Shuang Sun, Jiale Zhang, Jing Li, Yibo Chen, Xu Tang, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.39265)  

**Abstract**: Multi-image composition (MICo) is a critical yet challenging capability for large-scale image editing models, where consistency degrades rapidly as the number of reference images increases. While supervised fine-tuning (SFT) can equip models with basic MICo ability, it provides only implicit supervision on the composed image and often fails to learn reliable object-level correspondence across multiple references. We propose DeCoRL, a reinforcement-learning (RL) framework that improves MICo consistency by decomposing holistic MICo consistency into object-level consistency. DeCoRL first establishes instruction-relevant object correspondences across reference images and localizes the corresponding regions in the generated image. It then uses a VLM-based reward model trained on single-object consistency annotations to produce fine-grained, object-wise feedback, which is aggregated into a multi-object consistency reward to guide RL post-training. We further introduce DeCo-Bench, a benchmark covering compositions from 2 to 6 reference images. Experiments on DeCo-Bench and the public MICo-Bench demonstrate consistent improvements in MICo consistency. 

---
# Where Position Matters: Profiling-Guided Layer-Wise RoPE Allocation in Transformers 

**Authors**: Zhiwei Liu, Boao Kong, Bing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.39223)  

**Abstract**: Rotary Position Embedding (RoPE) typically uses uniform coverage across Transformer layers despite heterogeneous positional requirements. Phase-off interventions reveal non-uniform, model-specific rotary-phase dependence, while coverage interventions relate it to coverage-upgrade benefits. We propose profiling-guided layer-wise RoPE allocation, assigning high/low coverage under a fixed budget using early coverage-benefit profiles. The framework supports fixed-width query/key representations and physical rotary-dimension allocation in Multi-head Latent Attention (MLA). Normalized-depth mapping transfers profiles within model families, enabling small-model measurements to guide larger-model allocations before pretraining. A brief analysis relates selection and transfer to score calibration and inter-layer interactions. From-scratch experiments demonstrate validation-loss improvements over available budget-matched baselines, while analytical resource accounting identifies favorable quality--storage--compute trade-offs for physical MLA. These results connect positional demand to transferable, efficient architecture design. 

---
# EmbodiExp: Compositional Experience Transfer for Embodied Agents 

**Authors**: Zhifeng Gu, Ying Chen, Rui Jiang, Lei Yi, Bing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.39180)  

**Abstract**: Experience can help embodied agents turn high-level plans into grounded actions, but transferring it across tasks remains challenging: previously effective strategies may be inapplicable to the current state or fail to produce their intended outcomes. We present EmbodiExp, a framework for constructing transferable experiences and composing them online during embodied task execution. Our key observation is that tasks often share subgoals whose completion can be assessed independently of overall task success. We therefore introduce an outcome-grounded subgoal experience that specifies a subgoal, applicability conditions, an adaptable policy, and a verifiable outcome. During execution, applicable experiences guide action generation, while verified outcomes can establish the conditions required by subsequent experiences, enabling online composition. Subgoal-level outcome attribution further enables failure utilization: failed trajectories can still contribute verified local successes and condition-specific guidance on actions to avoid. Experiments on multiple embodied benchmarks under a fold-based protocol with frozen experience banks demonstrate the effectiveness of EmbodiExp. 

---
# PRISM: Personalized Item Representations in User History 

**Authors**: Renzhi Wu, Zikun Cui, Junjie Yang, Li Sheng, Yongjun Xie, Jerry Fu, Nipun Mathur, Sri Reddy, Hong Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.39175)  

**Abstract**: Sequential recommenders typically construct history inputs from item identifiers or content features using an embedding map shared across users. The same item therefore yields the same initial embedding representation, even when users have different relationships to the item. We propose personalized item representations in user history, where each item's embedding expresses its relationship to the user. Personalization becomes a property of the history inputs themselves, supplying richer evidence for later sequence modeling.
We instantiate this idea with PRISM, which reuses upstream ranking computation to generate personalized item embeddings. An autoencoder jointly compresses all layer states from a user tower, a user--item tower, and same-request candidate mixing into one embedding per scored pair for downstream replay. Offline evaluation on Surface 1, a major recommendation surface at billion scale, reports an average downstream normalized entropy (NE) change of $-0.15\%$, compared with $-0.07\%$ for VISTA at similar history lengths. We further identify an empirical sequence-length scaling law: over history lengths of 26--240 events, NE gain follows a fitted power law with exponent 0.468, increasing from 0.06\% to 0.17\%. Therefore, personalized item representations in user history provide a new scaling dimension for improving recommendation quality. 

---
# REALM: Identifying Minimal Task Assets for LLM Agents 

**Authors**: Zihou Ren, Byungchul Tak, Long Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.39155)  

**Abstract**: Large language model (LLM) agents often receive access to more files, accounts, and services than a task requires. Prompt injection turns this excess access into an opportunity to redirect execution toward unrelated resources. Applying least privilege therefore requires identifying the assets needed by a task, including assets whose identities emerge only during execution. We present REALM (Required Execution Assets Linked to a Mission), an approach to identifying and maintaining a task's required assets and their permitted uses. The approach represents required assets as Dependency Graph nodes, resolves prerequisite relationships, and admits newly discovered assets through source-scoped selection. It separates discovering an asset from authorizing its use, while preserving the agent's freedom to choose its computation. The REALM Builder maintains the selected grants and projects Asset Files; the Partial API Manager exposes resource-bound Partial APIs to the Agent inside a Task Container. Across 60 clean tasks and 560 injected cases in AgentDyn, it completes 40.0% and 36.4% of tasks and decreases ASR from 16.6% to 0.2% with qwen3.5-flash as the base model. 

---
# Clean Sheets and Checkpoints: Evidence Complexity for Certifying Sequential Agent Risk 

**Authors**: Rohith Yanambaka Venkata, Don McBride, Tiago Sequeira, Md. Al Amin  

**Link**: [PDF](https://arxiv.org/pdf/2610.39124)  

**Abstract**: Dangerous-capability evaluations often test an AI agent from prepared states at ordered milestones. A clean sheet records no successes. What it certifies about a campaign of $B$ attempts depends on what a failed attempt erases. We model the task as a chain of $k$ milestones divided into checkpoint segments and ask how many zero audit responses certify, at level $\alpha$, a completion probability of at most $\epsilon$. For segments chosen before the audit, counting possible completion windows gives a certificate for any specified budget. The bound also covers adaptive deployment processes that follow the chain's transitions, provided their success probabilities after every complete history are capped by the corresponding audit success probabilities. The audit does not test this transport assumption. In rare-risk phase interiors, each segment is either rarely completed within the budget (active) or nearly always completed (free). The minimum expected number of zero responses is asymptotic to $\log(1/\alpha)\,(B^a/(a!\,\epsilon))^{1/K_A}$, where the $a$ active segments contain $K_A$ milestones and are those at least as long as a budget-dependent threshold. The certificate attains this constant. Every asymptotically optimal evaluator must divide its expected response count equally among active milestones in the limit, with a vanishing share elsewhere. Where a single segment passes from active to free, each of its milestones receives a smaller limiting share than an active milestone; a crossover-aware certificate attains the optimal constant. For selected segments with known or unqueried prefix steps, counting failed cycles can sharpen the certificate. 

---
# Appreciation Teaches Unauthorized Execution: Self-Evolution Degrades the Robustness of Personal Agents to Indirect Prompt Injection 

**Authors**: Xinyan Wang, Xiaogeng Liu, Jun Shao, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.39122)  

**Abstract**: Self-evolving personal agents act for users and learn from user feedback. However, if an agent obeys an instruction that an untrusted third party hides in content, the third party gains control of the agent. Such indirect prompt injection (IPI) poses severe risks. We investigate whether self-evolution degrades the robustness of personal agents to IPI. To study this question, we design a controlled two-phase methodology. In the adaptation phase, the user asks the agent to read external content and complete the task described in it, a common request in daily use. The agent then learns from the user's feedback and improves itself as designed. In the test phase, a third party writes a similar task into new external content, but the user's test request does not authorize the injected task. Using this methodology, we build multiple tasks with diverse test instances in different domains for a real personal agent. Our extensive experiments reveal that self-evolution substantially degrades robustness to IPI. Specifically, compared with the base agent, which does not evolve from user feedback, the average attack success rate rises by 14.0--48.8 percentage points under skill evolution and by 3.0--39.3 percentage points under reinforcement learning, posing safety risks. The code is available at this https URL 

---
# From Local Recovery to Unified Generation: Discovering Triple Set in Knowledge Graphs via Observation-Anchored Latent Diffusion 

**Authors**: Jiaqi Wang, Yisen Gao, Huayang Shi, Junhao Hou, Wengen Li, Jihong Guan, Yichao Zhang, Shuigeng Zhou, Yangqiu Song, Hong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.39114)  

**Abstract**: Triple set prediction (TSP) is to recover missing relational triples in a knowledge graph by predicting the triple set instead of isolated links. Existing approaches predominantly rely on the subgraph-based completion paradigm, where the original graph is first partitioned into local subgraphs which are then independently completed. Such a paradigm suffers from fragmented structural reasoning, limited cross subgraph dependency modeling, and missing predictions across subgraphs. This paper proposes LatentDiff, a novel unified latent generative framework for TSP, which reformulates knowledge graph completion as latent space graph generation. LatentDiff first learns a compact latent representation using a relation-aware residual-quantized knowledge graph autoencoder, which encodes the whole graph into relation-aware latent embeddings, quantizes them with chunked residual codebooks, and reconstructs triples through a factorized decoder. On top of this latent space, we introduce an observation-anchored latent graph diffusion model, where the target latent graph representation is perturbed toward the observed latent. The reverse process recovers missing triples through progressive latent denoising. Experiments on three benchmark datasets show that LatentDiff outperforms the SOTA in terms of F_{tsp} by 5.4 percentage points on average. Code: this https URL. 

---
# PCM: A Proof-Carrying Message Protocol for Verifiable Multi-Agent Communication 

**Authors**: Bravish Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2610.39109)  

**Abstract**: When one AI agent tells another that "the tests passed" or "a person approved this refund", the receiver has only the sender's word. A2A and MCP authenticate the channel, but the sender is the party in doubt, so a prompt-injected agent can launder fabricated facts and authority through a multi-agent system. We present Proof-Carrying Messages (PCM), a protocol in which each claim carries evidence: records from an append-only Merkle log that the agent cannot write, with inclusion proofs, signed by an attesting signer the agent cannot control. Receivers check messages offline and give each claim a three-valued result: proven, unproven or failed. A claim that rests only on the agent's own report is never proven. Authority travels as grant chains rooted in a recorded human approval, and the chains can only narrow. We specify four claim types, a typed caveat algebra, replay rules that survive restarts, and strict readings that make independent verifiers agree, and we prove the resulting security properties. On a pre-specified benchmark of 926 messages from real runs, 712 of them lies in 16 classes, PCM accepts 10 lies, all of them stale evidence (1.4%, 95% CI 0.8-2.6%). Plain A2A accepts all 712, and models of six deployed protocols (A2A with signed Agent Cards, uAgents, RFC 9421, SCITT, AP2, Verifiable Intent) accept 504-712. Two independent verifiers agree on all 926 verdicts, and verification takes about 1 ms. 

---
# Generating Critic-Guided Synthetic Experience in Latent World Models 

**Authors**: Jorge de Freitas, Christos Ziakas, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2610.39100)  

**Abstract**: Learning from offline datasets is limited in scalability, as collecting new experience is often costly or unsafe in the real world. World models offer an alternative way to generate new experience through a learned dynamics model, yet learning policies in latent space is limited due to compounding errors in imagined rollouts. To this end, we propose using the world model as a local transition generator, leveraging critic guidance to choose actions inside that model to generate synthetic experiences. We study action refinement methodologies, introducing Critic-Guided Synthetic Experience (CGSE) generation in world models. Specifically, CGSE augments an existing offline dataset by sampling candidate actions using an offline policy and refining each to maximize a first-order approximation of the $Q$-function within a trust region. We then use a frozen latent world model to imagine the future state given each updated action, and use the critic to choose the best one, appending this synthetic transition to the replay buffer. Our method requires no additional real-environment interaction or test-time planner, as it generates synthetic experience offline using the world model. On three OGBench manipulation tasks, CGSE improves an offline policy by a relative improvement of $7.1\%$ on the average success rate. 

---
# Tactus: Tactile & Pose Sensing Glove for Natural Dexterity Data Collection 

**Authors**: Li Guan, Rachael Granberry, Sahar Aseeri, Sahar Rostami, Taylor Niehues, Sai Dhanooshi Marri, Mia Huang, Dingtian Zhang, Ziyi Kou, Jason Pursel, Eric Whitmire, Vatsal Mehta, Ashish Deshpande, Tristan Trutna, Reed Belcher, Emelia Black, Julian Blake, Roger Boldu, Sean Bonaventure, Einar Boyovich, Jordan Brennan, Elizabeth Casillas, David Christensen, Tyler Christensen, Nicholas Colonnese, Stuart Dealey, Gaige DeHaven, Katie Doran, Chris Doughty, Robert Drury, Silvia Espino, Gabriel Geyne, Beth Goodrich, Kyra Gould, Joe Grisso, Susana Guerrero, Andrew Gwozdziowski, Monica Ha, Ehsan Hajiesmaili, Kate Healy, Rose Hope, Michael Hurst, Claire Ikemiya, Daisy Isibor, Rick James, Yashasvi Joginapally, Samuel Johnson, Tyler Jorgensen, Kristy Jost, Ben Kamath, Slava Karulin, Alexander Keener, Woo Kim, Alison Krauskopf, Ankit Kumar, Joshua Lang, Lizzy Li, Cameron Lynch, Daniel Ma, Hector Martinon, Christina Matos, Keagan McCurdy, Lee Mincey, Stuart Nezlek, Tiffany Nguyen, James O'Hara, Jason Osborne, Jacky Pai, Roham Pardakhtim, David Perek, Alexandre Poulin, Nick Raines, Salwa Rasheed, Leah Resneck, Kevin Rhodes, Ergys Ristani, Douglas Roberts, Anik Sarker, Harrison Scarborough, Jonas Schmidtler, Brian Shieh, Benjamin Shuman, Garrett Simard, Forrest Smith, Udara Somarathna, Eliot Sosin, Jonathan Stewart, Kristine Strom, Adam Swett, Hal Talbot-Mantooth, Julia Tanner, Sai Sharan Thirunagari, Curtis Torrey, Simge Uzun, Eileen Wachs, Rylan Wasson, Henry Wei, Jesse White, Beau Woodbyrne, Antonio Woods  

**Link**: [PDF](https://arxiv.org/pdf/2610.39076)  

**Abstract**: Collecting naturalistic human manipulation demonstrations with accurate contact force and hand pose is essential for robot dexterous manipulation policy and foundational world model learning - yet no existing wearable platform provides both calibrated fingertip force sensing and occlusion-resilient hand tracking in a form factor that preserves natural manipulation behavior. We present Tactus, a textile-integrated sensing glove that co-optimizes three design pillars: (1) a thin, conformal 3D-knit form factor (1.5 mm fingertip stackup) designed to minimize interference with natural manipulation, (2) strategically placed capacitive tactile sensors at manipulation-relevant regions informed by empirical contact studies, and (3) IMU-fused pose tracking that maintains accuracy through the hand-object occlusions where manipulation data is most valuable. Tactus includes a documented calibration pipeline for cross-glove consistency and per-session calibration for subject-dependent correction. To enable objective comparison, we introduce standardized benchmarking protocols for tactile force regression and hand pose estimation, and apply them to Tactus and two commercially available tactile gloves - to our knowledge the first quantitative cross-platform evaluation of this kind. Tactus resolves a minimum detectable force of 0.59 N at 95% confidence, 1.6 - 2.7x lower than the commercial baselines, and with 0.87 cross-participant consistency, while our pose tracking sustains 11.0 mm fingertip keypoint relative error during manipulation, when occlusion is heaviest. We show that these combined signals support object property estimation, slip detection, and representative recording of hand-object interaction - yielding demonstration data with the fidelity required for contact-aware manipulation policies and world models. 

---
# HERA: Assessing an Autonomous Agent for Quantum Device Measurement 

**Authors**: Zachary W. Gomez, Nathan Bittner, Jordan T. McCourt, Qingrui Cao, Sandesh S. Kalantre, Benjamin H. Alexander, Zhiren Zheng, Kenji Watanabe, Takashi Taniguchi, Aaron Sharpe  

**Link**: [PDF](https://arxiv.org/pdf/2610.39074)  

**Abstract**: Large language model agents have begun to carry out laboratory measurements, but their scientific conclusions are useful only if they hold up under scrutiny. Demonstrating this requires controlled studies because models are stochastic and the context a model is given steers its conclusions. Here we introduce HERA, an autonomous agent designed to carry out cryogenic transport measurements of quantum devices. HERA draws on a knowledge base curated by lab members that holds measurement protocols, software tools, and literature references. We assessed HERA with a blinded trial against an expert on the stacking order of two multilayer graphene devices, an ablation of its knowledge base, and hundreds of offline replays in which models designed the trial's central measurement without running it. HERA's conclusions matched the expert's sealed conclusions on both tested devices. The ablation and the replay studies measured the impact of the knowledge base across different models, showing that with it every model designed better measurements and reasoned more soundly. To test how readily HERA extends to a new problem, we tasked it with finding and characterizing superconductivity in a different material system. This required a few hours of expert time to develop a new experimental protocol, after which HERA did both autonomously. These results show that, with a knowledge base designed for the task, language model agents are capable of sound experimental science. 

---
# When Does Closed-Loop Curriculum Adaptation Help GRPO? An Analysis on Countdown 

**Authors**: Yuv Raj Pant, Moshiur Rahman Tonmoy, Kaiqun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2610.39056)  

**Abstract**: Reinforcement fine-tuning has become an effective post-training paradigm for improving the reasoning ability of large language models, but it remains sample-inefficient when the training distribution is mismatched to the model's current capability. In group-relative objectives such as Group Relative Policy Optimization (GRPO), problems that are too hard often produce uniformly failed rollouts, while problems that are too easy provide little useful advantage signal. In this work, we introduce CLOSEDLOOPGAUSSIAN, a lightweight adaptive curriculum method for efficient reinforcement fine-tuning on ordered reasoning tasks. Instead of following a fixed easy-to-hard schedule, CLOSEDLOOPGAUSSIAN preserves the structure of an Easy-to-Hard Gaussian sampler while dynamically adjusting its pace and spread using periodic probe estimates of per-difficulty accuracy. The controller accelerates training when the model is ahead of a reference curriculum, slows down when the model falls behind, and broadens sampling when earlier difficulties regress. This feedback mechanism keeps training focused on tasks that are challenging but still learnable, without changing the reward function, model architecture, or underlying GRPO optimizer. Experiments on symbolic arithmetic reasoning task show that CLOSEDLOOPGAUSSIAN improves exact-match accuracy over vanilla GRPO and fixed open-loop curricula, with the strongest gains in intermediate-difficulty regimes where group-relative learning signals are most informative. Codes are available at this https URL. 

---
# Reinforcement Learning for the Game of Balatro: A Fast Simulator and Two-Policy Training Approach 

**Authors**: Ajay Vikram Periasami, Anjaneya Gupta, Ethan Cheng, Ivan Rosales  

**Link**: [PDF](https://arxiv.org/pdf/2610.39049)  

**Abstract**: Balatro, a 2024 roguelike deckbuilder, presents three difficulties that have historically derailed reinforcement learning methods: a 3,000-action state-dependent action space, horizons of several hundred decisions with sparse rewards, and pervasive stochasticity in both card draws and shop generation. Published RL results on the game remain scarce. We introduce an end-to-end pipeline comprising (i) a high-throughput Rust simulator that is automatically state-diffed against the live game client via BalatroBot, (ii) a condensed observation-action interface with dynamic per-step masking of legal moves, and (iii) a multi-agent PPO architecture that decomposes play into a tactical blind policy and a strategic shop policy that share the full game state but act in disjoint phases. Long-horizon credit assignment is partially addressed by per-agent reward routing with explicit cross-agent credit. On the main joint run, training metrics show a sharp phase transition in which the first-blind clear rate converges above 99% and mean episode return increases by roughly an order of magnitude, plateauing at a mean maximum ante of 3 out of 8; single-agent baselines under identical rewards oscillated between greedy and hoarding failure modes and did not stably clear the first blind. These results establish the two-policy decomposition as a strong reactive baseline for Balatro and localise the remaining gap to long-horizon strategic planning rather than tactical blind play. 

---
# Representation Complexity Along Learning Trajectories: Trading Width for Precision and Time 

**Authors**: Ryan Singh, Miguel De Llanza, Christopher L. Buckley  

**Link**: [PDF](https://arxiv.org/pdf/2610.39027)  

**Abstract**: What computational role does overparameterisation play during learning? Information-plane analyses suggest that neural networks can traverse representations substantially more complex than those required by their final predictors. We explore when this transient complexity serves a computational purpose by comparing the representation complexity needed for prediction with that needed to make an unknown target statistically visible through gradient feedback. We introduce an oracle model of neural network learning and use Kolmogorov \(n\)-widths to quantify effective representation complexity. Our main technical result transfers complexity bounds from the forward representation to the corresponding gradient-query features. Combined with statistical-query lower bounds, this gives a task-agnostic lower bound on the initial effective width required for target-dependent gradient information to become detectable. For \(k\)-sparse parity in \(D\) dimensions, a known target admits an \(O(k)\)-width representation, while the general bound already implies a polynomial separation between representation and discovery. We then exploit parity's invariance under coordinatewise sign reflections to identify a sharper unsigned effective width, which removes sign distinctions invisible to the parity information channel. At a transfer-compatible measurement resolution, its critical scale is \(\widetilde{\Theta}_k\left( D^{k-1}\tau^2 \right)\), up to logarithmic factors. Thus, the representation resources required to discover a predictor can substantially exceed those required to represent it once learned. Interpreting precision through accumulation over fresh samples yields a corresponding width-time frontier. Experiments support the predicted dimension dependence and examine how optimisation and generalisation depart from this idealised frontier. 

---
# Distilling the Shape of Uncertainty: What On-Policy Distillation Learns and How to Predict It 

**Authors**: Shiping Yang, Chuanqi Tang, Hossein Hajimirsadeghi, Layla El Asri, Mohammad Amin Shabani, Angel X. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2610.39020)  

**Abstract**: On-policy distillation (OPD) is increasingly adopted in the post-training of large language models. However, a stronger teacher may fail to improve the student. Prior work attributes this to a lack of new knowledge, but what that knowledge is remains unclear. We show that OPD learns not only \textit{how to branch}, i.e., which candidate tokens to consider, but also \textit{where to branch}, i.e., where to keep reasoning paths open. The existing token-overlap metric captures only the former. We introduce the entropy shape, the sequence of next-token entropies, to characterize the latter, and propose the Entropy Shape Correlation (ESC) to measure student-teacher agreement in entropy shape. Tracking ESC throughout training and intervening on a teacher's entropy while preserving its token preferences, we empirically show that \textit{where to branch} causally drives OPD gains, independently of \textit{how to branch}. Finally, we formulate OPD outcome prediction as a teacher selection task and establish an evaluation protocol for it, under which pre-OPD ESC selects the best teacher for both students and outperforms existing predictors. Code is available at this https URL. 

---
# Implicit Reasoning in Transformers: Bridging the Compositionality Gap in GPT-2-like Models 

**Authors**: Davide Venuto, Leonardo Venuto  

**Link**: [PDF](https://arxiv.org/pdf/2610.39016)  

**Abstract**: Compositional generalization remains a persistent limitation of transformer language models: even when a model learns to combine known facts correctly in-distribution (ID), the same compositional mechanism may fail when applied to out-of-distribution (OOD) knowledge. We investigate this compositionality gap in a controlled two-hop reasoning setting using a GPT-2-like transformer model trained from scratch. Building on prior work identifying a circuit associated with compositional generalization, we test whether the failure of OOD transfer is related to how atomic knowledge is structurally exposed during training. We modify the training distribution by presenting both ID and OOD atomic facts in a format positionally aligned with the second-hop retrieval step of the learned compositional circuit, without modifying the model architecture or adding OOD compositional supervision. Under this intervention, the model successfully transfers the learned ID compositional behaviour to held-out OOD atomic facts, closing the observed compositionality gap in the controlled setting. Our results provide evidence that the observed compositionality gap is not a fundamental architecture limitation and that it can be overcome through mechanistically informed training distribution design. 

---
# Assessing feature scorer results on high-dimensional datasets with t-SNE 

**Authors**: Bruno Iochins Grisci, Mario Inostroza-Ponta, Márcio Dorn  

**Link**: [PDF](https://arxiv.org/pdf/2610.39009)  

**Abstract**: While vast literature on high-dimensional data visualization is available, there are not many works regarding the visualization of feature scorers and their results. Feature scorers are algorithms that assign numerical importance to each feature of multi-dimensional datasets. These importance scores can be used in several applications, such as feature selection, knowledge discovery, and machine learning interpretability. There are several feature scorers to choose from, and often no single metric or ground truth is available to guarantee the quality of their results. In this scenario, visualization can become valuable to support the decision of which method to choose and how good its results are. For this goal, this work presents "weighted t-SNE." It modifies the relationship between data points in the embedded 2D space to reflect the importance of each dimension of the original datasets as assessed by a feature scorer. This research discusses how to implement weighted t-SNE, proposes the silhouette coefficient as a numerical evaluation of the results, and shows several examples of its use in practice. Synthetic and real-world tabular datasets are used in the experiments together with nine feature scorers, ranging from Mutual Information to neural networks. Each feature scorer produces unique visualizations, and weighted t-SNE can be used to compare and choose the one that better suits a given dataset and task. Weighted t-SNE can also visually show the importance of features learned by machine learning models and help us see how they are organizing the data, increasing their interpretability. 

---
# Rethinking GFlowNet Training: Negative Reinforce Dynamics 

**Authors**: Pedro Dall'Antonia, Daniel Csillag, Diego Mesquita  

**Link**: [PDF](https://arxiv.org/pdf/2610.38994)  

**Abstract**: Generative Flow Networks (GFlowNets) are models that learn to sample discrete objects proportional to a target reward. They are trained by minimizing balance errors over sampled trajectories, which inherently ties correctness to exploration: if a region is not visited, balance cannot be enforced there. Adaptive exploration is therefore crucial, yet standard training relies on a semi-gradient update that treats the sampling distribution as fixed at each optimization step. On the other hand, the full gradient reveals a second mechanism that reduces the observed errors in an anti-exploratory manner. In this paper, we establish a connection between the full gradient and min-min dynamics between the GFlowNet and a decoupled sampler, showing that the sampler reduces observed violations by avoiding high-error regions. This insight motivates a bridge between two-player games and single-policy dynamics, which underpins the design of our proposed method, Negative Reinforce Dynamics (NRD). By tying the parameters of an adversarial game together, we recover a single-policy dynamic that extends exploration while maintaining correctness. In standard benchmarks, our method outperforms existing approaches in both convergence speed and exploration efficiency, without training a separate teacher or auxiliary exploration policy. 

---
# Benchmarking Graph Neural Networks for AC Power Flow: The Role of Inductive Biases and Physical Constraints 

**Authors**: Federico Clerici, Ferran Bohigas-Daranas, Hamid Latif-Martínez, Eduardo Prieto-Araujo, Oriol Gomis-Bellmunt, Pere Barlet-Ros  

**Link**: [PDF](https://arxiv.org/pdf/2610.38985)  

**Abstract**: Fast and accurate models for AC power flow (ACPF) are increasingly important as modern electrical grids require more frequent system analyses. Graph neural networks (GNNs) naturally exploit electrical grid topology, yet the effects of architectural design choices on predictive performance remain poorly understood given differences in graph representations and evaluation settings across studies.
We present a unified benchmark of representative GNN architectures for ACPF prediction under a common experimental framework. All models use identical preprocessing and training procedures and comparable parameter budgets, isolating the impact of message-passing mechanisms and inductive biases. We also analyze errors by distance from the slack bus and systematically ablate key design components. We evaluate predictive accuracy and physical consistency on the IEEE 14-bus and IEEE 118-bus systems.
Results indicate that anisotropic message-passing architectures offer the most favorable trade-off between predictive accuracy and physical consistency under the studied parameter and topology constraints. Explicit edge features encoding line impedances and global context aggregation are particularly important for representing electrical interactions and system-wide dependencies, especially for voltage-angle predictions on the topologically bottlenecked IEEE-118 grid. Under a budget with approximately fixed parameters, two or three message-passing layers generally provide the best balance between depth and representation width, while additional depth offers no consistent improvement. Ablations also show that standard residual and root connections are not necessarily beneficial in shallow models, and that aggregation operators involve trade-offs between statistical accuracy and physical consistency. These results provide practical guidelines for designing graph-based power system models. 

---
# Why Speculative Decoding Works Better Than Predicted on Sparse MoE Models 

**Authors**: Ekagra Ranjan, Komal K. Teru, Bharat Venkitesh, Acyr Locatelli  

**Link**: [PDF](https://arxiv.org/pdf/2610.38981)  

**Abstract**: Speculative decoding (SD) accelerates autoregressive inference by leveraging the memory bound nature of decoding, verifying multiple draft tokens in a single forward pass. For mixture-of-experts (MoE) models, prior work predicts a concave speedup-batch size relationship, with gains peaking at intermediate batch sizes due to expert saturation. We evaluate this across 10 sparse MoE models (30B--1T parameters, 8--512 experts) and find recent fine-grained models break this trend, showing higher speedup at low batch size. We find verification cost is governed not only by activated-expert bytes, but also by temporal overlap across draft tokens and GPU execution dynamics. At low batch size, MoE layers in fine-grained models account for a much smaller share of runtime than of weights; on H100, kernel profiling attributes this mainly to higher bandwidth utilization of MoE kernels relative to attention, with L2 absorbing repeated reads of co-routed experts. Across draft tokens, routing overlap further reduces the unique experts activated during verification relative to independence assumptions. We model this effect using a maximum-entropy Iterative Proportional Fitting (IPF) framework that predicts unique expert counts from routing-overlap statistics with less than 4\% error, and substantially narrows the gap to real-model curves and captures the qualitative low-batch trend from synthetic routing. 

---
# TableTune: Measured, Self-Correcting Draft-Length Tables for Dynamic Speculative Decoding 

**Authors**: Abhishek Malvankar, Paul Castro, Nick Mitchell, Lionel Villard  

**Link**: [PDF](https://arxiv.org/pdf/2610.38968)  

**Abstract**: Speculative decoding speeds up large language model (LLM) serving only while the GPU has spare compute, so the best draft length, the number of tokens a small drafter guesses per step, shrinks as the batch grows. vLLM lets operators set the draft length for each batch size with a hand-written table, but nothing measures or corrects that table. We present TableTune. Its tuner times the decoding step for every batch size and draft length on vLLM's own serving path and computes, in minutes, the table that maximizes throughput. Its controller, a vLLM plugin, tracks how many guesses are accepted in live traffic and rebuilds the table while serving. We measure TableTune only against other ways to speculate. On an H100, the tuned table delivers at least 94% of the throughput of the best fixed draft length at every load in four of five drafter settings (88% in the fifth) without the operator having to know which length that is. Against reimplementations of TurboSpec- and Nightjar-style control on the same engine, its worst-case loss is never higher than theirs, and with the EAGLE-3 drafter it is 3% against 9% and 12%. On a mixture-of-experts model, vLLM's documented table loses up to 18% and the tuned table at most 3%. Starting from mistuned tables, the controller raises throughput by up to 58% (up to 45% on vLLM's default runner), and after a workload switch with the DFlash drafter it recovers 96% of a stale table's loss. In llm-d's optimized-baseline deployment as shipped, two Qwen3-32B replicas on four GPUs, the tuned table is up to 32% faster than a fixed draft length of 3 and up to 25% faster than vLLM's documented table at high load, and the controller raises the documented table's throughput by up to 29%. 

---
# MAS-Loop: Beyond One-Shot Orchestration towards Self-Evolving Multi-Agent Systems 

**Authors**: Hehai Lin, Qi Yang, Chengwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2610.38946)  

**Abstract**: Training-time multi-agent system (MAS) design seeks to learn reusable orchestration policies, yet prevailing approaches remain largely one-shot: an Orchestrator generates a system but only observes and learns from its terminal outcome. This paradigm discards execution trajectories that expose system-level failures and contain fine-grained information beyond the terminal outcome, leaving no direct basis for revising the source MAS, evaluating the updated MAS, or learning from the resulting contrast. To this end, we introduce MAS-Loop, an execution-grounded framework that closes this feedback loop by coupling an Orchestrator with a trajectory-conditioned Refiner. For each task, the Orchestrator constructs and executes a source MAS to obtain an outcome score and an intermediate trajectory. Conditioned on the trajectory, the Refiner localizes an actionable failure, formulates a trajectory-grounded diagnosis, and proposes a bounded patch to the agents, information flow, or execution configuration. The executable updated MAS is re-executed on the same task to measure its realized outcome under stochastic execution. This pair-MAS comparison provides complementary learning signals: the Orchestrator learns to generate strong systems with fewer recoverable failures, whereas the Refiner learns to expose and repair observed weaknesses, thereby driving policy-level co-evolution across training iterations. Experiments show that MAS-Loop achieves the strongest overall performance among the evaluated baselines, with consistent gains from refinement across all evaluated benchmarks. The co-evolution process also yields MAS-JudgeCorpus, a dataset of execution-grounded source-updated pairs with orchestration- and trajectory-level supervision, which can provide the community with support for MAS reward modeling and trajectory-aware investigation. 

---
# TrussTutor-Bench: Benchmarking large language models on solution accuracy, sycophancy, and error detection in statics 

**Authors**: Evangelos Apostolakis, Erfan Hamdi, Sara Jahedazad, Emma Lejeune  

**Link**: [PDF](https://arxiv.org/pdf/2610.38917)  

**Abstract**: Large language models (LLMs) have rapidly become a default source of help for students in science and engineering courses. In foundational engineering mechanics courses such as statics, students routinely use these tools to check answers, work through partial solutions, and ask for explanations. Effective tutoring in this setting requires more than a correct final answer. A model must also confirm correct work when a student doubts it, and identify errors in flawed work. Existing benchmarks evaluate LLMs primarily on final-answer accuracy, so it is unclear whether models that can solve mechanics problems also have the other technical capabilities that tutoring depends on. Here we introduce TrussTutor-Bench, a benchmark of 200 verified two-dimensional truss problems paired with automatically generated student solutions, and use it to evaluate thirteen LLMs on solution accuracy, sycophancy, and error detection. Baseline accuracy ranged from 64.5% to 97.0%. Every model detected nearly all injected errors in synthetic flawed student work (90.4% to 100%), so error detection did not distinguish models under these clean conditions. Sycophancy did. When a student insisted that correct work was wrong, models rejected correct solutions to problems they had themselves solved in 1.6% to 50.0% of cases. These results show that performance varies widely across models: the leading models solve, check, and defend student work nearly perfectly, while others are far less reliable, particularly when a student doubts correct work. TrussTutor-Bench provides a verified, extensible standard for determining which models are technically reliable enough to put in front of students. Follow-up work must now focus on teaching rather than technical capability, drawing on learning science to determine whether LLM help leads students to understand the material or simply to reach the right answer. 

---
# Lev-3B: A Data Recipe for Compact Probabilistic Log Decision Models 

**Authors**: Leo Betthauser, Yingqi Liu, Uday Shah, Jishang Wei, Sainyam Galhotra, Liang Gou, Vedant Dharnidharka, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.38915)  

**Abstract**: Many log-analysis tasks involve choosing among known options, such as detecting anomalous sessions or identifying attack techniques, and benefit from calibrated probabilities. We present Lev-3B, an open 3B-parameter decision model built on Granite-4.1-3B-Base with a shared pointer readout, and study its training recipe. The recipe has two key parts. First, training follows a curriculum: the model learns bounded natural-language decisions, using no log data, before it sees any logs. Second, the log stage mixes the target tasks with general log data spanning 35 auxiliary tasks unrelated to either evaluation task. We evaluate one shared checkpoint on sources seen in training and on sources excluded from training, calibration and model selection. The two-stage recipe transfers substantially better than training on log data alone. Adding general log data improves performance on excluded sources. At a fraction of their serving cost, Lev-3B outperforms prompted frontier models on anomaly detection, including on an excluded logging system, but trails them on attack-technique classification. Its probabilities are well calibrated on familiar sources, but on the excluded logging system it confidently misses anomalies. We release the checkpoint, recipe, data manifests, and prediction-level artifacts. 

---
# Interventional Training Reduces Causal Bypass in In-Context Introspection 

**Authors**: Zilu Tang, Jongin Kim, Kerem Sahin, David Atkinson, Aaron Mueller, Derry Wijaya  

**Link**: [PDF](https://arxiv.org/pdf/2610.38911)  

**Abstract**: Introspection, the ability of a system to access and reason about its internal signals, is crucial for language model (LM) self-consistency and scalable interpretability. Prior work argues that LMs possess introspective capabilities, and that training (SFT) could enhance such behavior. However, most setups leave open the possibility of "causal bypass," in which the self-report relies on signals that are correlated with the self-report's subject, while being causally unrelated to it. Although high-level concepts are difficult to intervene on, lower-level signals (such as a single neuron activation) allow precise interventions. Understanding such signals can serve as a minimal prerequisite before understanding more complex introspection. In this study, we train hundreds of introspectors across three low-level signal types and eight LMs. We find that straightforward SFT training leads to causal bypass, and propose interventional training, an alternative where we train the model to report on counterfactually perturbed values. This method enables more accurate introspection of endogenous $\textit{and}$ injected signals, particularly in out-of-distribution (OOD) data. The mechanism generalizes across token positions and scales with more data. Our results demonstrate that while causally grounded introspection is possible, it is not guaranteed -- emphasizing the need to train on mechanisms, rather than behaviors alone. 

---
# FlipJev: Fast Black-Box Red Teaming Finds Universal Adversarial Suffixes for Typed Decision Models 

**Authors**: Xiaogeng Liu, Xinyan Wang, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.38895)  

**Abstract**: Typed decision models are an emerging class of language models that make decisions instead of generating free-text responses. They answer in a single fast and cheap query and are increasingly used, including in security-sensitive tasks. However, red teaming of these models leaves two gaps: existing adaptive attacks do not use their speed, and few studies have searched for universal adversarial examples. We fill both gaps on Jev, the representative typed decision model, with a quick black-box query-based attack that calls no attacker language model. We conduct extensive experiments on 12 datasets with different data types and application scenarios. The results show that Jev resists common prompt injections: no strategy changes a correct answer in 10,240 paired tests. In contrast, our attack finds short suffixes that flip confident answers within minutes. These suffixes are universal: one suffix per dataset flips 43% of unseen documents. Because typed decision models, unlike classic NLP models, can serve many tasks, we test whether these suffixes transfer across tasks. We find that one suffix generated from one task can flip 90-100% of the samples in other tasks. We therefore believe the evaluation of typed decision models must include strict adaptive attacks to fully reveal their robustness. 

---
# Autonomous Quantum Measurement with Agentic AI 

**Authors**: Eunbi Rha, Abhijatmedhi Chotrattanapituk, Mouyang Cheng, Chung-tao Chou, Shenyong Ye, Ashley S. Dale, Long Ju, Luqiao Liu, Mingda Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.38850)  

**Abstract**: Quantum materials exhibit diverse electronic and functional phenomena, but their characterization requires extensive experimental expertise, iterative decision-making and accumulated laboratory knowledge that is difficult to document and transfer. Here we introduce Quantum Agent (QUANTA), an agentic artificial intelligence (AI) framework for autonomous quantum-materials characterization. Using a widely adopted Physical Property Measurement System (PPMS) as an experimental platform, QUANTA combines retrieval augmentation using scientific literature, experimental protocols and prior measurements with specialized agents for experimental planning, troubleshooting, and data interpretation. A separate deterministic control layer validates instrument actions. Guided by incoming data, QUANTA evaluates measurement quality, revises experimental plans and selects subsequent measurements. We demonstrate these capabilities through transport measurements of the topological Weyl semimetal TaP, as well as investigations of nonlinear responses in magnetic material Co2MnGa. These results establish QUANTA as a generalizable framework for evidence-driven, safeguarded and closed-loop quantum-materials characterization. 

---
# Discovering Jailbreaks via Curriculum of Models 

**Authors**: Michał Burzyński, Noah T. Rossi, Shomit Basu, Andy Arditi  

**Link**: [PDF](https://arxiv.org/pdf/2610.38833)  

**Abstract**: Automated jailbreak discovery aims to find prompts that elicit prohibited behaviors from safety-trained language models, helping developers identify vulnerabilities before deployment. One class of approaches uses reinforcement learning to train an investigator model to generate prompts that elicit a specified behavior from a target model. However, training directly against a safety-trained target is difficult because nearly all prompts are refused, leaving the investigator with a sparse reward signal. Prior work addresses this with the propensity bound (PRBO), a proxy reward estimated by sampling responses from a more compliant target, and then adjusting for how likely those responses would have been under the original target.
We introduce Curriculum of Models (CoM), an alternative method that trains the investigator against a sequence of modified versions of the target, progressing gradually from a version that readily exhibits the specified behavior to the original target. For jailbreak discovery, training begins against a target version that rarely refuses harmful requests, and advances to the next, harder version once the investigator reliably elicits the behavior from the current one.
We evaluate CoM on elicitation tasks involving CBRN materials, explosives, and illicit drugs, using two open-weight targets, gemma-4-31b-it and gpt-oss-120b. We compare against PRBO and direct training on the original target without a curriculum. When the investigator is trained to elicit a single behavior, CoM finds successful jailbreaks more reliably than either baseline. When trained on multiple behaviors simultaneously, the CoM-trained investigator can elicit held-out behaviors, though PRBO-trained investigators achieve higher success rates. Our results suggest that a simple curriculum over target models offers a promising approach for addressing sparse rewards in rare-behavior elicitation. 

---
# Evaluating Visual Realism of AI-Generated Videos with Eye Tracking 

**Authors**: Danqing Shi, Lan Jiang, Jake Lyons, Miri Zilka  

**Link**: [PDF](https://arxiv.org/pdf/2610.38825)  

**Abstract**: Human evaluation of AI-generated videos has so far relied on explicit human feedback, such as judgments or ratings. While these signals capture the conclusions viewers reach, they fail to offer insights into the perception that leads to them. This paper investigates eye tracking as a complementary human evaluation signal for AI-generated videos. We conducted two Visual Turing Test (VTT) experiments with a total of 90 participants on videos generated by latest generative models, in which participants watched each video twice and then gave their feedback. Across the experiments, we collected 4,286 valid eye-tracking trials on 438 videos. Our analysis indicates that gaze patterns offer informative evaluation signals that align with explicit human feedback. Building on the findings, we propose the first computational method GazeEval to predict the perceived visual realism of AI-generated videos from eye-tracking data. It achieves 71.3% accuracy in prediction, providing evidence that eye tracking data carries informative signals about how people evaluate a video. We release the study data to support further research on eye tracking for evaluating AI-generated video. 

---
# CopyLoop: Fast Looped Transformer Decoding via Context Copying and Shallow Screening 

**Authors**: Wenyu Du, Stephen Chung  

**Link**: [PDF](https://arxiv.org/pdf/2610.38822)  

**Abstract**: Looped Transformers increase computational capacity through the repeated application of shared blocks, but their recurrent depth makes conventional full-depth decoding expensive. While existing speculative pipelines accelerate generation by processing consecutive token positions at varying depths, they are typically limited by sequential decoding overheads. We introduce CopyLoop, a training-free decoding algorithm that combines low-cost context-copy proposals with shallow screening and continuous recurrent scheduling. Rather than step-by-step generation, CopyLoop proposes multi-token suffixes from repeated context. It then evaluates them early at a shallow recurrent depth to efficiently filter out incorrect branches, processing newer blocks concurrently while earlier blocks are still progressing toward full-depth verification. We evaluate CopyLoop across six model checkpoints from the Ouro, Raven, and Huginn families, covering code generation and multi-step mathematical reasoning. CopyLoop achieves decoding speedups of 2.98x--10.70x over full-depth autoregressive decoding on Base models and 2.74x--3.52x on Thinking models, consistently outperforming or closely matching all competing baselines across evaluation tasks. Furthermore, ablation studies confirm a strong synergy between shallow screening and continuous scheduling, while our analyses show that CopyLoop requires only lightweight model calibration and yields greater speedups on longer generations with reusable contexts. 

---
# AtomTree: A Hierarchical Framework for State-Aware Embodied Instruction Following with LLMs 

**Authors**: Haotian Zhai, Tianming Sha, Junnan Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.38744)  

**Abstract**: Embodied instruction following in interactive environments requires agents to interpret natural language commands and complete long-horizon tasks. Existing planners either rely on large labeled trajectories, or use few-shot and zero-shot LLM prompting that produces a whole plan in a single serial reasoning process. These approaches depend on linguistic common sense, commit to a plan from the initial instruction, and only update after failures, which weakens robustness and state awareness. We present AtomTree, a hierarchical framework for state-aware task planning with LLMs. AtomTree incrementally builds a tree of subgoals and actions, where the LLM decides at each step, from the current symbolic state, whether to further decompose or to output a grounded low-level action. A lightweight memory module summarizes past outcomes for reuse in later decisions. When a low-level action stalls, AtomTree triggers hierarchical backtracking replanning to return to the parent node and regenerate the subtree plan from the latest state, preventing continued drift along an incorrect trajectory. In zero-shot evaluation on the ALFRED benchmark, AtomTree outperforms prior LLM-based planners. Project page: this https URL. 

---
# inkGround: Benchmarking Fine-grained Visual Grounding Capabilities of Vision-Language Models for Cursive Handwriting in the Wild 

**Authors**: Sidra Hanif  

**Link**: [PDF](https://arxiv.org/pdf/2610.38720)  

**Abstract**: Human handwriting is produced as an ordered sequence of pen movements, but an offline image keeps only the final ink. Handwriting trajectory recovery aims to restore the ordered stroke points and their pen-up/pen-down transitions. Existing image-trajectory pair datasets are recorded with digitizing tablets under controlled conditions, which limits writer coverage and stylistic diversity. Moreover, there is no benchmark that measures the visual grounding capabilities of general-purpose Vision-Language Models (VLMs) for the stroke trajectory recovery task. In this work, we introduce inkGround, a benchmark for stroke trajectory recovery from full-page, camera-captured handwriting. inkGround pairs 537 page images (~18.7k words, ~163k strokes) with reference trajectories produced by a pretrained offline-to-online model, filtered by OCR agreement, and verified by human raters. Because these references are pseudo-labels that represent one plausible pen trajectory, we also evaluate VLMs on the IAM-OnDB dataset, which provides recorded pen trajectories. We cast the task as text-conditioned point grounding: given a page image and a target word, the model must locate the word and output its ordered stroke points. Outputs are scored by trajectory fidelity (DTW, Chamfer distance), readability of the rendered strokes (Accurate Rate, Correct Rate), and style similarity. We benchmark four open-source and three frontier closed-source VLMs with a shared in-context prompt. Closed-source models, notably Gemini-3.1-Pro, produce legible trajectories close to the reference, and this ranking also holds on IAM-OnDB. 

---
# Learning Scientific Trajectories to Predict Future Biomedical Discoveries 

**Authors**: Chia-Hsuan Chang, Brian Ondov, Jiang Lv, Lan Jiang, Yunmai Wang, Haerin Song, Xueqing Peng, Qiaozhu Mei, Xiaolei Su, Hua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.38719)  

**Abstract**: AI co-scientists have expanded the use of large language models for generating scientific hypotheses and guiding experimental investigation. Here we investigate whether temporal patterns in the semantic representations of complete biomedical conclusions can guide hypothesis generation. We introduce ForeSCITE, a framework that identifies thematic clusters of published conclusions, estimates a direction separating earlier from more recent findings within each cluster, and projects existing conclusion embeddings along these directions. An embedding language model trained to reconstruct conclusions decodes the projected coordinates into natural-language hypotheses. Across 407 clusters derived from approximately three million pre-cutoff conclusions, we identify temporal structure and illustrate its retention after decoding. In retrospective, time-sliced evaluations, ForeSCITE increases semantic coverage beyond the historical corpus, biomedical entity-pair novelty and post-cutoff pair re-observation relative to five prompt-based and retrieval-augmented baselines. Expert review of prioritized CAR-T hypotheses identified 37 candidates as both novel and biologically plausible, and expert justifications for five selected hypotheses illustrate their biological rationale and subsequent literature support where available. These findings support temporal guidance in semantic space as a scalable mechanism for proposing biomedical hypotheses for subsequent scientific evaluation. 

---
# Latent Decision Flow Rewards for Clinical Guideline Reasoning 

**Authors**: Gaurav Bhatt, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2610.38709)  

**Abstract**: Clinical practice guidelines encode evidence-based recommendations as conditional decision flows, which makes them a natural source of process supervision for training reasoning models. Two properties of clinical reasoning make this supervision difficult to use. A decision flow consistent with a guideline does not by itself determine the correct answer for a given patient, which we call the path--answer gap, and several valid decision flows can support the same clinical decision, which we call path multiplicity. Current training approaches address parts of this problem. Answer rewards supervise the final decision, and process rewards computed against a reference flow supervise one valid decision flow per case. Supervised fine-tuning on annotated flows teaches guideline-consistent reasoning but constrains the model to one trajectory per case and can reduce answer accuracy. We introduce GUIDER, a method that jointly optimizes a decision flow controller and a reasoning policy. The controller learns to generate plausible decision flows from guideline-derived supervision, and the policy learns to generate a decision flow and a final answer for each patient case. GUIDER computes the process reward for the policy over several controller-generated flows, which accounts for path multiplicity, and gates this reward on answer correctness, which addresses the path--answer gap. Joint training allows the process supervision to evolve with the policy. On MedGUIDE, GUIDER with Qwen3-14B achieves 82.5 weighted accuracy and 73.5 guideline adherence, 3.3 and 3.1 points above the strongest clinical reinforcement learning baseline. With Qwen3-14B and Gemma4-12B backbones, GUIDER also improves over CRPO, MRPO, and FaithMed on the out-of-domain benchmarks AMEGA-LLM, RealMedQA, and CancerGUIDE. 

---
# SatOps-Decision: Low-Latency Discriminative Decisions for Onboard Satellite Communication Operations 

**Authors**: Lifan Mei, Yuxuan Li, Jiaxian Tu, Shenghe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.38697)  

**Abstract**: We propose SatOps-Decision, a method for low-latency decisions in satellite communication operations. To the best of our knowledge, it is the first application of Jev-style general-purpose discriminative decision models to satellite operations, and the accompanying benchmark is the first dedicated to this approach. The learned scorer, a locally adapted Laya encoder with its native decision head, selects among supplied operational answers in a single forward pass. The benchmark combines authored operational contracts, controlled state changes and independently checked labels in an auditable construction. Evaluation prioritizes complete-request latency and raw decision accuracy; all main accuracy comparisons use the same 1,024 held-out questions, separate from training and development. SatOps-Decision's local scorer has a median latency of 90.3 ms per decision, versus 493.9 ms for Qwen3.5-0.8B generation and 388.4 ms for OneJev1.6-0.8B: $5.47\times$ and $4.30\times$ speedups based on median request times in the measured FP32/MPS implementations. Its mean raw accuracy is 85.3%, versus 42.5% for frozen Laya, 40.7% for Qwen generation and 58.6% for OneJev. As cloud accuracy references, original Jev 1.14/1.13 reach 87.4/81.9%. 

---
# Geometry Matters in Packed 3D Attention 

**Authors**: Amil Khan, Anantajit Subrahmanya, B.S. Manjunath  

**Link**: [PDF](https://arxiv.org/pdf/2610.38694)  

**Abstract**: Packing a 3D volume into a token sequence is not merely long-sequence modeling. Once the volume is packed, sequence order no longer encodes the geometry of the sample---so a positional mechanism indexed by sequence position can learn the memory layout rather than the object. This is a general failure mode for attention over packed geometric data. We introduce PACK3D, a local-global attention operator that separates memory layout from attention geometry. PACK3D keeps local neighborhood attention exact, compresses only distant context into contiguous 3D block summaries, and indexes every query-key interaction with axial RoPE on true token-grid coordinates rather than packed positions. Because the rotation is applied before a block is summarized, the query-to-summary logit remains a weighted sum of query-relative 3D displacements; compressed global context therefore stays spatially interpretable after packing, instead of collapsing to anonymous pooled content. We measure the failure mode directly. On a controlled synthetic task---train on one packing, change only the memory order of the tokens, hold the physical labeling rule fixed---learned positional embeddings and packed-sequence RoPE solve the training layout perfectly yet collapse under Morton, block-contiguous, and random repacking, while axial grid-coordinate RoPE stays exact throughout. The same coordinate system carries to real data: on Allen WTC-11 hiPSC microscopy, PACK3D improves over a parameter-matched 3D U-Net on the 15-structure joint task and over a matched learned-position local-global baseline, with controls isolating the coordinate system, not encoder capacity, as the source of the gain. A fused Triton implementation matches windowed-attention latency at 65k tokens while supplying the long-range context windowing cannot---so the operator is not only correct but trainable at scale. 

---
# Explore Where It Matters: Targeted Exploration for Efficient Rover Adaptation 

**Authors**: Jacob Levy, Arthur Bouton, Tristan D. Hassler, Hari Nayar, David Fridovich-Keil  

**Link**: [PDF](https://arxiv.org/pdf/2610.38657)  

**Abstract**: Future planetary rovers will traverse steep, deformable terrain where slip, sinkage, and terrain deformation can threaten mobility. Although simulation can capture some wheel-soil effects, unmodeled deformable-terrain phenomena create sim-to-real gaps requiring deployment-time adaptation. Yet, rover operations leave little room for extensive trial-and-error. We introduce TERRA (Targeted Exploration for Reward-Relevant Adaptation), an online adaptation framework that efficiently seeks deployment data exposing task-relevant model errors and corrects them. TERRA adapts a simulation-trained world model with a reward-weighted prediction loss and trains a loss predictor to estimate where future performance-relevant errors are likely to occur, using this signal to guide exploration during MPC. In simulation, TERRA reaches approximately 90% slope-climbing success within five trials under severe out-of-distribution pretraining conditions, outperforming exploration baselines in reward, completion time, and energy to first success. On the ERNEST rover, TERRA recovers coordinated active-suspension locomotion on a deformable Mars-simulant slope within three to five real-world trials. Project website: this https URL 

---
# Information-Theoretic Reliability--Safety Bounds and Finite-Alignment Dynamics in AI Knowledge Distillation 

**Authors**: Miao-Miao Yi, Zhen-Yu Cai, Y -M. Du, C. P. Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.38651)  

**Abstract**: The reliability and safety of artificial intelligence (AI), together with their transfer and evolution through knowledge distillation, have attracted growing attention. We establish an information-theoretic bound on the reliability--safety trade-off and characterize the dynamics of this trade-off under knowledge distillation. We first prove a universal trade-off inequality that is independent of the specific model form. Its boundary is determined solely by a single parameter Kh, which quantifies the hidden layer's ability to distinguish benign from hazardous inputs. We then develop a canonical Boltzmann-machine model of AI response behavior and derive two exact relations between reliability and safety. The trade-off curve is determined by the hazard discrimination capacity K, while the overall answer tendency controls the operating point along the curve. These relations agree well with existing experimental data. In knowledge distillation, we find that teacher responses induce a renormalization of the student model's trade-off parameters. We further use a renormalization-group approach to study multigenerational distillation, deriving exact recursion relations and the associated renormalization-group-like flows for two representative direct intergenerational distillation protocols at arbitrary teacher--student alignment strength. These flows reveal fixed-point structures, long-term dynamics, and critical behavior, yielding testable predictions for the critical transition between capability loss and stable transmission. 

---
# When AI performs, what remains? An integrative review of generative AI and cognitive capital in individuals and organizations 

**Authors**: Giuseppe Riva, Stefania La Rocca, Brenda K. Wiederhold, Andrea Gaggioli  

**Link**: [PDF](https://arxiv.org/pdf/2610.38647)  

**Abstract**: Generative artificial intelligence is rapidly entering knowledge work, and its short-term productivity benefits are increasingly documented. Less is known about what happens, over time, to the knowledge, skills and judgment that remain in people and organizations. This integrative narrative review addresses that question through the construct of cognitive capital: the accumulated, renewable and depletable stock of knowledge, cognitive and metacognitive capabilities, and practices that enables an individual or an organization to learn, interpret information, exercise judgment, solve problems and adapt. We distinguish individual and organizational cognitive capital from the capability of the human-AI system, so that gains in joint performance and losses in human capability can be observed at the same time. Drawing on experimental, field and conceptual work, we propose four mechanisms through which interaction with AI affects cognitive capital: substitution of practice, scaffolding and feedback, metacognitive calibration, and organizational aggregation. Seven principles summarize the current evidence, including that assisted performance differs from retained competence, that the same technology can support or displace learning depending on design and use, and that individual gains can coexist with reduced collective diversity. For each mechanism we derive testable propositions, and we discuss five design levers for organizations as proposals to be validated rather than established practice. We conclude that cognitive capital offers a multilevel frame for governing AI adoption and outline a research agenda centered on longitudinal, multilevel measurement. 

---
# Enhanced image forensics through partial C2PA-based provenance information 

**Authors**: Riccardo Ziglio, Eleonora Marchesini, Andrea Montibeller, Giulia Boato, Silvio Ranise, Cecilia Pasquini  

**Link**: [PDF](https://arxiv.org/pdf/2610.38639)  

**Abstract**: Cryptographically verifiable C2PA-based Content Credentials (CCs) can be attached to freshly acquired or generated digital images but also to already existing ones with no previous provenance information, for instance after processing through trusted C2PA-enabled software tools. The latter case leads to images with so-called partial CCs, which capture only the last operations applied, while prior provenance remains unknown and can be investigated through signal forensics. This paper addresses the forensic analysis of such images, highlighting the opportunity to leverage the partial CCs as additional trusted knowledge to adapt forensic detectors towards the statistical characteristics of the image under analysis and mitigate potential distribution shifts. To this end, a two-step strategy is envisioned: (i) reconstructing the operations applied by the last signer through a structural analysis of the partial CCs; (ii) exploiting the obtained reconstruction to enhance the forensic detector of the analyst through an adaptation process. An empirical analysis on partial CCs produced by commercial C2PA-enabled software tools through different editing operations is carried out, providing evidence on the feasibility of the reconstruction in practical cases. An experimental case study on the forensic task of synthetic image detection is then presented, showing that the proposed strategy can significantly enhance the performance of state-of-the-art detectors on test images that are postprocessed via C2PA-enabled software tools after being generated/acquired. 

---
# On-Policy Self-Distillation with Teacher Evolution 

**Authors**: Shufan Shen, Zhongni Hou, Chao Bi, Yufei Zhang, Guojun Yin, Shuhui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.38637)  

**Abstract**: On-Policy Self-Distillation (OPSD), which aligns a student model's token-level distribution with that of its self-teacher prompted by privileged context, has emerged as a promising direction to mitigate reward sparsity in sequence-level post-training paradigms. However, OPSD often fails to achieve continuous performance gains during training. In this paper, we find that the model's limited self-teaching capability underlies this phenomenon by generating unreliable guidance on certain samples. To address this problem, we propose Teacher-Evolution Self-Distillation (TESD), which enables the self-teacher to generate rollouts and receive environmental feedback based on task correctness and context independence. First, using these rewards as training objectives, we perform teacher-oriented optimization to continuously enhance the model's self-teaching capability during distillation. Second, by estimating teacher guidance reliability from these rewards, we apply adaptive student-oriented optimization to down-weight samples with unreliable guidance. Extensive experiments across multiple datasets and model scales show that TESD consistently outperforms representative token-level and sequence-level methods, highlighting the necessity of improving supervision quality in OPSD. 

---
# How Deterministic SAST and AI Security Agents Divide the Work: Coverage, Harness Policy and Cost in Snyk VulnBench 2.0 

**Authors**: Liran Tal, Chris Liu, Charlotte Fereday, Sebastian Roth, Brian Clark  

**Link**: [PDF](https://arxiv.org/pdf/2610.38617)  

**Abstract**: We ask what AI coding agents and AI security harnesses find that deterministic static application security testing (SAST) does not, and what they miss that SAST catches. VulnBench 2.0 is a benchmark of 108 hand-curated, attacker-reachable vulnerabilities in 20 open-source web applications in Go, JavaScript/TypeScript, Python and Java; 62 flow across more than one file. A finding is credited only if it locates the vulnerability's attacker-controlled source and its dangerous sink, each within two lines. We evaluate nine productized configurations under one blind protocol: the Snyk Code static analyzer, Claude Code's /security-review with three model and effort settings (one with the Snyk MCP server), Codex Security with GPT-5.6 Luna, Terra and Sol, and the DeepSec scanner with two model backends. Snyk Code reached 82.4% macro recall (89 of 108) and Claude Code with Opus 5 at xhigh effort 79.3% (85 of 108); they disagree on 28 vulnerabilities, split 16 to 12, too few to tell them apart. Neither engine was sufficient alone: together they reached 101 of 108, the ceiling of all seven comparable configurations combined, and the AI configurations also reported at least seven real vulnerabilities that the reference set did not contain and Snyk Code did not report. What an AI engine misses depends on its harness as much as on its model: without Claude Code's built-in review policy, the same model matched 16 instead of 9 of the 20 vulnerabilities in the classes that policy excludes or restricts (supplementary run on 19 of 20 applications); Codex Security with GPT-5.6 Sol located 80 vulnerabilities at file level but 34 at line level; DeepSec's model-free file selection kept a file from only 56 of the 108 flows. Coverage and cost were only loosely related. The harness, fixtures, reference sets and all run data are public. 

---
# The Artificial Personality Model: Theory, Implementation, Measurement and Experimental Validation in Conversational Agents 

**Authors**: Alexander Dregger, Barbarella Petz, Erik Kuhn, Pia Volk, Andreas Oberweis, Stefan Hillmann, Jannik Strötgen  

**Link**: [PDF](https://arxiv.org/pdf/2610.38608)  

**Abstract**: Artificial personality shapes user experience (UX) in human-machine interaction, yet coherent theoretical frameworks and validated measurement instruments remain limited. This paper refines and empirically evaluates the Artificial Personality Model (APM), a hybrid framework integrating human-centred and machine-specific personality dimensions. Across three experiments (N = 395), the model was operationalised and evaluated using the Artificial Personality Questionnaire (APQ). Factor analyses revealed that some human personality dimensions (e.g., agreeableness, conscientiousness) are partially transferable, whereas others (e.g., extraversion) did not emerge as distinct dimensions. Additional machine-specific dimensions were identified. The APQ provided encouraging initial psychometric evidence, while the human-centred BFI-2 showed poor reliability and an uninterpretable factor structure in Experiment 3. Experimental manipulations demonstrated significant effects of artificial personality on user satisfaction, continuous use intention, and social presence. The findings contribute an empirically refined framework and a novel instrument for measuring perceived artificial personality, offering practical guidance for designing conversational agents. 

---
# Episodic looking - using reinforcement learning to bias perception enhances few-shot and continual learning with protection from forgetting 

**Authors**: David Rawlinson, Gideon Kowadlo  

**Link**: [PDF](https://arxiv.org/pdf/2610.38577)  

**Abstract**: There is a fundamental contradiction between the desire for models that can generalize, and models that can continuously adapt to recent input with changing statistics. These conditions, captured in few-shot and continual learning settings, can bias or disrupt learned representations. This harms generalization to a broader context, leaving the model unable to overcome input variance.
This paper proposes a solution based on the Complementary Learning System architecture (CLS), in which a Short-Term Memory (STM) delegates generalization to a slow, statistical Long-Term Memory (LTM). Delegation is achieved by working in a space of robust, generalized perceptions produced by the LTM. With this separation, the STM can focus on memorizing recent input and avoiding interference between memories.
We show that on an image classification task, the resulting CLS model is more resistant to forgetting in a domain-incremental continual learning setting than a baseline ResNet LTM, and generalizes more effectively in a few-shot setting than other continual learners.
In the continual setting, overall accuracy is slightly worse than existing continual learning models, but with a different mix of performance characteristics. Results analysis suggests that the model is limited by the representations and representational plasticity of the LTM. In future work, we will seek to address this by replaying STM memories to the LTM. 

---
# Do Audio LLMs HEAR Pain? Tracing Acoustic Evidence from Speech to Language Reasoning 

**Authors**: Wenqi Li, Xiangjue Dong, Sulaiman Vesal  

**Link**: [PDF](https://arxiv.org/pdf/2610.38570)  

**Abstract**: Large audio language models (LALMs) can process speech directly, but it remains unclear how well they use the non-lexical acoustic information that speech carries. We study this question through pain, using TAME-Pain, which contains pain-neutral Harvard sentences and spoken pain ratings recorded at different self-reported pain levels. We introduce HEAR, a four-stage diagnostic framework that traces pain-related information from the speech signal to the model's prediction, and apply it to seven closed-source and open-weight LALMs. On the Harvard sentences, where pain can only be inferred from the voice, all models perform at chance level, and same-speaker references or few-shot examples improve balanced accuracy by no more than six points. On the spoken ratings, the same models reach 87.5-99.7% balanced accuracy. Layer-wise probing of open-weight LALMs shows that pain can be decoded from early audio representations (74.1% balanced accuracy for binary prediction and 45.0% for three-class severity), but that this information, along with most pain-associated acoustic cues, decreases in deeper layers. In an acoustic perception test, weaker models cannot distinguish even large changes in these cues, whereas stronger models can but still fail to infer pain. Finally, we recover pain-related acoustic cues and directly reinject them as structured text into the prompt, which improves pairwise pain prediction by up to 32 points. These results indicate that current LALMs encode paralinguistic information in early layers that is largely unavailable when the model produces its answer. 

---
# Persona Learning from Feedback for Personalized AI Assistants 

**Authors**: Yiran Shen, Yu Xia, Liuyi Yao, Prithviraj Ammanabrolu  

**Link**: [PDF](https://arxiv.org/pdf/2610.38561)  

**Abstract**: Personalizing AI assistants requires adapting their behavior to users' preferences and task contexts. Existing work elicits assistant personas through persona prompting or finetuning on persona-conditioned generations. However, persona descriptions alone do not identify what to change or preserve in a particular response, or which behaviors suit the current request. We introduce $\textit{Persona Learning from Feedback (PLF)}$, a framework that learns reusable, preference-grounded assistant personas as independent LoRA adapters on a shared frozen backbone. During training, a frozen self-teacher conditions on persona specifications and response-specific feedback to provide token-level supervision on each adapter's own generated prefixes. A feedback gate focuses corrective updates on responses judged to need improvement. At deployment, a supervised router uses the request or dialogue to select a single adapter or a fixed composition, without user-specific finetuning. We evaluate PLF on single-turn personalization with model-generated critiques and multi-turn tutoring with simulated students' reactions. Across both settings, PLF improves model-judged preference fit over fixed-persona policies and strong baselines including persona prompting and persona distillation, demonstrating how feedback-trained behaviors can be reused for adaptive assistant personalization without degrading general capabilities. 

---
# AtomSteer: Test-Time Constraint Steering for Atomistic Generative Models 

**Authors**: Pengfei Cai, Katharina S. C. Jäger, Utkarsh Utkarsh, Nofit Segal, Bowen Deng, Akshay Subramanian, Rafael Gomez-Bombarelli  

**Link**: [PDF](https://arxiv.org/pdf/2610.38517)  

**Abstract**: Diffusion and flow matching models offer scalable approaches to generating crystalline materials, molecular conformers, and protein-ligand complexes. However, satisfying crystallographic symmetry, structural geometry, and accurate stereochemistry remains challenging. Satisfying these requirements calls for corrections that preserve the geometry already captured by the pretrained model. In this work, we propose AtomSteer, a test-time inference framework for steering pretrained atomistic generative models with structural constraints. During sampling, AtomSteer corrects intermediate structure predictions with bounded local projections, graph-rigid stereochemical transformations, and projections over periodic symmetries, accepting updates that improve constraint satisfaction while limiting changes to the rest of the structure. We demonstrate AtomSteer across three pretrained backbones, FlowMM, ET-Flow, and Boltz-2, spanning de novo crystal generation and crystal structure prediction, molecular conformer generation, and protein-ligand complex structure prediction. AtomSteer improves space-group distribution coverage in de novo crystal generation and improves target space-group recovery in crystal structure prediction. AtomSteer improves molecular validity and E/Z stereochemical accuracy while retaining conformer ensemble fidelity. For protein-ligand complexes, AtomSteer reduces ligand E/Z errors relative to experimental reference structures while achieving competitive docking success and PoseBusters validity. Our method provides a unified framework for adding chemical and crystallographic control to existing atomistic generative models without retraining or architectural changes. 

---
# SaiVLA-JEPA: Evaluating Future Latent Prediction in JEPA-Conditioned VLA Control 

**Authors**: Rufeng Xiao, Ruihao Xiao, Wei Chen, Xinhai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.38508)  

**Abstract**: Joint-embedding predictive architectures (JEPAs) learn to predict representations rather than pixels, offering a compact way to model how visual scenes evolve. We ask whether this predictive objective improves closed-loop control when a vision-language-action (VLA) policy already uses V-JEPA features. We call our JEPA-conditioned flow policy family SaiVLA-JEPA. It combines a vision-language model with a diffusion transformer action expert and receives features from a frozen V-JEPA encoder. Two lightweight auxiliary heads predict future representations during training and are omitted at inference, leaving the action sampler unchanged. We compare policies without JEPA features, with current JEPA conditioning, and with additional future supervision on 10,030 paired LIBERO-Plus episodes. Current JEPA conditioning reaches about 82.5% success versus 71.6% without JEPA, while future supervision provides no detectable aggregate gain at the evaluated training budget. We separately evaluate SaiVLA-JEPA-TCA, which adapts latent context, and SaiVLA-JEPA-TPA, which adapts selected parameters. The context-retry gain is a system-level result under a larger attempt budget; parameter adaptation shows no consistent advantage over the frozen policy and target controls. These results motivate assessing latent-prediction objectives through paired closed-loop rollouts rather than latent loss alone. 

---
# Tracekit Box: Running AI Agents in Micro-VMs Where Every Way Out Is Decided and Signed 

**Authors**: Bravish Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2610.38458)  

**Abstract**: AI agents now install packages, call model APIs with their operators' keys and change code in repositories that deploy to production. The network, credentials and write access they need are also how they can leak data, steal a key or change a system nobody approved. Sandboxes contain an agent but leave little evidence of what crossed their boundary; tracing records what happened but contains nothing. We present Tracekit Box, an open-source system (MIT, v0) that does both. Each job runs in its own Firecracker micro-VM with no network card. The only ways out are doors on the host, served over vsock: commands, outbound connections, secrets, model calls and the workspace patch. At every door a Tracekit signer the agent cannot reach decides what passes and signs it into one run per box. Names are resolved on the host; private and cloud-metadata addresses are always refused; secrets never enter the VM, which sees placeholders the host proxy swaps for real values only in one header for bound hosts; and changes come back as a patch applied file by file after the signer decides each file. Box fails closed, and each job ends as a .tkb bundle anyone can verify offline. All 13 end-to-end checks pass on a real KVM runner (the test that runs them takes 28.5 s, including image pull and boot); 15 fuzz targets on the trust boundary produced nine findings, all fixed; and the demo run's 102 signed records verify. Box does not yet run Firecracker's jailer, needs a co-located signer, and records processes inside the VM only where they cross a door. 

---
# Benchmarking decision models against generative large language models on medical multiple-choice questions 

**Authors**: Oscar Freyer, Daniel Glauert, Moritz Hofmann, Max Ostermann, Stephen Gilbert, Jakob Nikolas Kather  

**Link**: [PDF](https://arxiv.org/pdf/2610.38448)  

**Abstract**: Medical workflows often require selecting among predefined options. We compared 13 decision models with 15 generative large language model configurations, including different reasoning settings, using 1,273 four-option MedQA questions under text-only conditions. Accuracy ranged from 67.71% to 89.08% for decision models and from 78.32% to 96.86% for generative configurations. Among the tested configurations, Mercury Decide had the lowest mean final-attempt cost (USD 0.0069 per 1,000 questions), while Clef Omni had the shortest median request duration (0.291 seconds). Decision-model probabilities distinguished correct from incorrect answers, but error rates among answers assigned probabilities of at least 0.90 ranged from 1.71% to 12.77%. Answer agreement across three runs ranged from 92.30% to 100%, while probability repeatability varied. Decision models offered resource advantages alongside lower peak accuracy; selecting models for medical workflows requires considering probability quality and repeatability together with accuracy and the resources needed to detect and correct errors. 

---
# RoleFUZZER: User-Behaviour Fuzzing for Role-Specification Adherence in Large Language Models 

**Authors**: Kshitish Ghate, Michael Saxon, Tadayoshi Kohno, Aylin Caliskan  

**Link**: [PDF](https://arxiv.org/pdf/2610.38438)  

**Abstract**: Large language models (LLMs) are often conditioned into specific roles such as tutors, interviewers, or agents acting for an organisation, through system prompts that specify how they should behave. The extent to which they can actually adhere to these role specifications under multi-turn interactions with realistic and potentially problematic users is understudied. To address this, we propose user behaviour fuzzing, a guided search over plausible user behaviours that can lead a model to violate its role specification, and introduce RoleFUZZER, a framework that adapts coverage-guided fuzzing from software testing to operationalise the search. RoleFUZZER constructs scenarios guided by a user-behaviour coverage grid that crosses a grounded taxonomy of user interaction types and the role specification of the model under test i.e. what is under pressure. With RoleFUZZER, we evaluate LLMs across five role specifications, including tutor, financial advisor, AI interviewer, antagonistic roleplay character, and web agent under organisational policies. Across simulated and validated conversations in 15 campaigns, the framework reaches >= 89% coverage of the 64-cell behaviour grid for every model and role tested, and surfaces specification-grounded failure patterns. At an equal budget of conversations, removing the RoleFUZZER grid and asking an LLM to generate specification targeting users free-form lowers coverage to 39-69% with fewer kinds of violations. The same specification can also fail in opposite directions. For e.g., on the antagonist, open-weight target models override the player's safety signals, while Claude Haiku 4.5 breaks character instead. RoleFUZZER offers a tractable way to uncover vulnerabilities in LLMs across role specifications by testing diverse user behaviours that are likely to violate them. 

---
# DecisionUnlearn-Bench: Unlearning Probabilistic Decisions under Changing Schemas 

**Authors**: Tiantong Wu, Wei Yang Bryan Lim  

**Link**: [PDF](https://arxiv.org/pdf/2610.38431)  

**Abstract**: Machine unlearning aims to reproduce the behavior of a model trained without designated data. For probabilistic decision models, this target also depends on the candidate options supplied at inference. We introduce DecisionUnlearn-Bench, an empirical framework that compares full, unlearned, and retain-only models across deletion granularities, label renderings, and candidate sets. Across 20 eligible model configurations on the BANKING77 dataset, full-to-oracle displacement increases consistently from example to association to class deletion. Replacing semantic option labels with opaque codes nearly eliminates the retrained oracle's recognition of deleted classes, even with successful acquisition of the full model, whereas changing the candidate set alters both the oracle and the measured unlearning progress. SimNPO satisfies the screening criterion in 19 of 20 configurations, but its gains vary substantially across seeds and deletion targets. Development tuning on the CLINC150 dataset does not establish a positive pooled effect across 48 intents. At approximately matched suppression, methods can differ markedly in oracle fidelity. Recovery experiments further distinguish initial forgetting from subsequent adaptation and show conclusions that depend on both architecture and metric. These results show that evaluating decision-model unlearning relative to a schema-specific retraining target, with separate reporting of effect size, utility, robustness, and recoverability. 

---
# The Code Behind the Layout: AI Agents for Analog Layout Generators 

**Authors**: Minxing Chu, Anhang Li, Martins Atilla, Ruichen Qi, Mehdi Saligane  

**Link**: [PDF](https://arxiv.org/pdf/2610.38426)  

**Abstract**: Analog layout remains largely manual and hard to automate. Generator frameworks such as gLayout express layouts as programs, producing layouts from parameterized, design-rule-aware generators; however, writing such generators still takes expert effort. Large language models (LLMs) could now write generator code, and frontier models can even draw layout geometry from design intent. This raises a practical question: if a model can draw the layout itself, what does a generator framework still contribute, and is applying these models to a generator framework better than having them draw a layout from scratch? Through controlled experiments on gLayout in SKY130, we find that the main obstacle to drawing a layout directly is its time and token cost, and that the main advantage of a model building on a generator framework is reusability. Its verified primitives keep each model edit small enough to check, and its parameters allow a single edit to be evaluated across an entire family of layouts. Accordingly, a model is most effective when it develops the layout from the generator source of the family that the design intent belongs to, and tests each edit on the family members most likely to fail, so that one verified change improves every layout the generator produces. The results of these family tests can then serve as supervision for a local model, which learns edits that hold across the family at a fraction of the inference cost. 

---
# SpaceAct: Benchmarking 3D Spatial and Temporal Reasoning via Closed-Loop Action and Active Perception 

**Authors**: Chenglong Shi, Zekai Yuan, Yue Song, Jon Pu, Will Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.38405)  

**Abstract**: To solve problems in 3D space rather than merely describe images, multimodal large language models (MLLMs) need 3D spatiotemporal reasoning: reconstructing 3D geometry from occluded 2D views, anticipating physical dynamics, and executing precise, closed-loop actions. Prior studies test passive VQA on static images or symbolic planning, requiring no execution or supplying privileged coordinates while ignoring collision, statics, and timing. We introduce SpaceAct, an interactive benchmark evaluating 3D spatiotemporal reasoning via closed-loop physical action and active perception, with SAPA, a Spatiotemporal Active-Probing Agent framework. Built on interlocking bricks and mechanical puzzles, it has 108 verified instances across 36 task families in six categories, without privileged coordinates, and diagnostic modes (+plan, +state, no-camera, blind, L7 levers). In brick-bridging or aperture-threading, models rotate the camera to resolve occlusion, read in-scene rulers, output 6-DoF poses under swept-collision and statics checks, and self-correct via split-screen rollback feedback (dynamic tasks freeze time during deliberation and give five strobe frames). Across 10 frontier MLLMs on the 44 hardest instances, scores span 4.8-79.1 (seven solve at most 27.3% of 6-DoF tasks); for the top three, a reference plan (+plan) closes 44% of the gap while leaving collisions at 5.2 per episode, whereas adding exact 3D state (+state) closes 90% and cuts collisions to 0.5; for four others, +plan lifts the mean score from 10.9 to 35.3 and +state to 81.8; and 46% of failures are collision or observation loops. SpaceAct and SAPA advance evaluation to closed-loop physical interaction, showing that even with a valid plan, MLLMs remain bottlenecked by 3D visual grounding and action refinement, while offering a testbed for active multi-view exploration and explicit 3D grounding. 

---
# LLMs as Judges and Examinees for Medical Item Difficulty Prediction 

**Authors**: Xiaomeng Xiong, Marcus Walker, Yanlin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.38402)  

**Abstract**: Predicting medical item difficulty from item text alone remains challenging. Prior work has largely relied on linguistic features or text representations derived from general-purpose and domain-specific language models, but predictive performance has remained modest. This study takes a different approach by using difficulty-related signals that emerge when open-weight large language models (LLMs) interact with test items. Using USMLE and PANCE medical licensure examination items, we examined LLMs in two roles. As judges, the models provided absolute difficulty ratings and pairwise difficulty comparisons. As simulated examinees, they generated response-based signals including correctness, keyed-option probability, response entropy, and the probability margin between the top two options. We then combined the strongest judgment and response signals with baseline item features in downstream prediction models. Pairwise difficulty judgment was the most consistent individual LLM signal across both datasets, while simulated-examinee signals and absolute judgments provided complementary information in some feature combinations. The best configurations reached a Spearman correlation of $\rho$ = .447 on USMLE, exceeding the best system in the 2024 BEA shared task ($\rho$ = .25), and $\rho$ = .494 on PANCE. These results were comparable to direct absolute judgments from the proprietary frontier model Claude Opus 5 ($\rho$ = .443 and .488, respectively). Overall, judgment and response signals from open-weight LLMs captured difficulty-related information beyond text representations. Because these models can be deployed locally, the approach offers a practical option for predicting the difficulty of unreleased or high-security test items. 

---
# Manifest-Assisted Retrieval in Artificial Intelligence Search: Evidence Gaps and a Security Evaluation 

**Authors**: Gangzhen Qian, Kainan Zhou, Chuhong Xu, Hang Xiao, Zhi Jing, Rong Bao, Lu Yi, Zhaoyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.38361)  

**Abstract**: The external sources retained in our non-systematic audit did not isolate the incremental security or utility effect of consuming /llms.txt while ordinary retrieval remained fixed. The initial audit covered 39 sources through August 16, 2026, with later targeted updates through October 14. A purposive 15-tool study observed no automatic requests in 45 trials at one endpoint; a non-probability log study found that 97% of domains with detected files received no recorded requests in May 2026. We separate publication, fetch, model use, and action; define five operational threat paths and paired security-utility estimands; and exercise deterministic pre-model controls with 30 author-defined parser fixtures. We also execute a reduced pilot: 24 tasks across eight sites, five conditions, and 120 real model responses, with ordinary evidence held fixed. Frozen automatic scoring identifies attacker-target adoption in $21/24$ raw adversarial responses, with one further response unresolved; the adversarial URLs-only condition has no known adoptions and one unresolved response. The pilot measures output changes conditional on supplied evidence, without live retrieval or external actions. Together, the framework and pilot distinguish manifest-content effects from publication, discovery, and downstream execution, while leaving full retrieval and agent evaluation open. 

---
# SparseTokenGS: Gaussian-based Sparse-View CBCT Reconstruction via Reusable Token Pool 

**Authors**: Zihan Deng, Zhenzhen Gong, Yujie Liu, Jiaxin Gong, Dingkun Liu, Kexian Xie, Yixing Huang, Peng Jin, Tianye Niu, Zhisheng Wang, Fenglin Liu, Hongming Shan, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.38360)  

**Abstract**: Sparse-view cone-beam computed tomography is widely used for fast, low-dose volumetric imaging and lesion localization in dentistry and image-guided interventions, but it introduces severe artifacts. Feed-forward Gaussian methods transfer the reconstruction cost to training and decode each coordinate from projection-aligned features and neighboring anchors. However, information from the whole scan is written only when the field is constructed and cannot be retrieved during coordinate decoding, leaving the volume inconsistent within the scan. This is a common limitation of such methods: once the field is fixed, decoding uses only local features and reconstruction quality drops substantially. We propose SparseTokenGS, a Gaussian-based method for sparse-view cone-beam computed tomography reconstruction via a reusable token pool. Anchor Token Fusion and Point-Token Attention query this shared pool at the anchors and at the query coordinates, respectively. The local Gaussian prior is retained, and the volume is reconstructed in one forward pass, without per-scan optimization or a projection-domain loss. On three public chest and dental datasets, SparseTokenGS achieves the highest reconstruction performance among the compared methods. On clinical dental scans, reconstruction from only six projections reaches 29.51 decibels in peak signal-to-noise ratio and a structural similarity of 0.842 against the full-view reference, and off-the-shelf tooth-and-pulp segmentation keeps these structures separable for root-canal localization. Experiments on volcanic rock and chicken show performance after same-object adaptation. A reusable token pool gives feed-forward reconstruction a mechanism that keeps scan-wide context available through decoding and can benefit future rapid clinical lesion localization and analysis. The code is available at: this https URL. 

---
# WeVisCut: The Whole Editing Room in One Agent From Narration to Long-Form Video 

**Authors**: Chenyu Mu, Wenming Weng, Qinrui Wang, Xingsong Ye, Hao Yu, Jun He, Chong Sun, Chen Li, Jing Lyu, Erkun Yang, Cheng Deng  

**Link**: [PDF](https://arxiv.org/pdf/2610.38352)  

**Abstract**: Producing long-form videos from narration requires interdependent decisions that remain coherent across shots. Existing agentic editors often use multi-stage workflows driven by general models, leaving the quality-cost trade-off of compact, task-adapted policies unclear. We address this gap with WeVisCut, a long-form script-to-video system that connects tool execution with policy learning. Its agent harness maintains shared production state so that tool feedback informs subsequent decisions and repairs. Because finished videos do not expose these decisions, our four-stage data pipeline combines editing inversion with verified replay to turn recovered editing intent into executable supervision. Applied to 3,750 finished videos, this pipeline yields over 130K training examples. We use this supervision to train a single 9B policy that coordinates production and recovers from failures within the same harness. Moreover, we introduce a 100-task benchmark and five metrics assessing editing quality and delivery reliability. Under a shared tool environment, WeVisCut substantially outperforms its base model and achieves comparable performance to Claude Opus 5 while using nearly 33x fewer generated tokens per successful output minute. It also outperforms task-matched native editors on our proposed metrics. 

---
# Aligning Misaligned Benchmarks for Single-View Novel View Synthesis 

**Authors**: Yang-Che Tseng, Erqun Dong, Jeong-gi Kwak, Gopal Sharma, Shakiba Kheradmand, Stephan Garbin, Hossam Isack, Kwang Moo Yi  

**Link**: [PDF](https://arxiv.org/pdf/2610.38349)  

**Abstract**: While single-view novel view synthesis methods have been dramatically enhanced recently, how to quantify their performance has yet to be standardized. In this work, we demonstrate that an important factor lies in how each method's own reference frame is aligned with the so-called ground-truth pose. Existing methods often rely on test-time optimization of target poses for this alignment, which inherently is ill-posed due to the complexity of the optimization landscape of camera poses vs render accuracy, leading to methods being evaluated at an arbitrary local optimum. Some recognize this limitation and thus rely on 3D reconstructions of entire scenes using training and test views to align poses, which may be infeasible. In this work, instead, we propose to use traditional 2D-to-2D and 2D-to-3D image matching techniques to align 3D reconstructions. We also evaluate only within co-visible regions, between different views, which is the only place where a `correct' ground truth can be defined. Our evaluations show that with proper coordinate alignment, novel-view synthesis methods that use explicit 3D Gaussians achieve the state of the art, despite being previously reported in the literature to be outperformed by pure image-based 2D methods. We further provide a simple baseline that directly utilizes an off-the-shelf video generator and a feed-forward geometry estimator. Our simple baseline is already comparable to state-of-the-art methods specifically trained for novel-view synthesis, without any task-specific training. 

---
# Statutory Article Retrieval from Arabic Judicial Narratives: Family Law and Law of Evidence in Saudi Arabia 

**Authors**: Zarah Shibli, Areej Alhothali, Lulwah Alharigy  

**Link**: [PDF](https://arxiv.org/pdf/2610.38336)  

**Abstract**: Reliable legal information retrieval must ground legal reasoning in authoritative statutory sources. In Arabic judicial settings, long factual narratives must be mapped to concise articles that express abstract legal rules, creating a narrative-to-legal-rule abstraction gap. Using a benchmark of 200 synthetic Saudi Arabic judicial cases and 381 articles from the Family Law and the Law of Evidence, we compare word-level and char-4 BM25, zero-shot dense encoders, per-law retrieval, dense-sparse reciprocal rank fusion (RRF), and a proposed Decompose-and-Refine (DaR) pipeline. In DaR, a large language model splits each narrative into disputed legal issues and generates, for each issue, three retrieval channels: a statute-style paraphrase, legal keywords, and a candidate taxonomy chapter. On the whole corpus, the instructed Qwen3-Embedding-8B dense encoder reaches an MRR@10 of 0.849, the best result of any non-decomposed system. Under per-law scoring, this advantage reflects Family Law only: restricted to Evidence articles, the same encoder drops to an MRR@10 of 0.087, and fusion with BM25 raises it only to 0.207. DaR changes this picture, raising Evidence MRR@10 to 0.696 and Family MRR@10 to 0.864. Without decomposition, the two laws favor different strategies; under DaR, the same retriever achieves the best MRR@10 on both. 

---
# MUSE: Multi-Level Memory with Selective Runtime Injection for Scientific Coding Agents 

**Authors**: Haoran Jin, Defu Lian, Xiting Wang, Hong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2610.38332)  

**Abstract**: Historical experience can guide scientific coding at multiple levels, but task-level retrieval may miss implementation lessons that become relevant during execution. We introduce MUSE, a memory management framework combining multi-level memory with selective runtime injection for scientific coding agents. MUSE builds global, approach-level, and implementation-level memory from historical trajectories. At initialization, the agent receives global memory and problem-matched approach-level memory. During execution, current interactions are used to retrieve implementation-level memory. An LLM gate selects relevant experience and generates specific review advice, or skips injection. Across four benchmarks spanning numerical problem solving, kernel optimization, and scientific software repair, MUSE outperforms retrieval and memory baselines on all evaluation metrics. Ablations support the multi-level design and gated injection, while transfer experiments show that memory remains useful across kernel benchmarks. These results show that multi-level memory and selective runtime injection enable effective reuse of experience in scientific coding. Code is available at this https URL. 

---
# A Case Study of a Multi-Agent Kernel-Optimization Loop That Learns a New AI Accelerator 

**Authors**: Byonggon Chun  

**Link**: [PDF](https://arxiv.org/pdf/2610.38329)  

**Abstract**: Frontier large language models write correct CUDA kernels for most single-operator benchmark tasks, yet their accuracy collapses on accelerators whose architecture and programming language they have rarely seen. We present a case study of an LLM agent loop built for such accelerators. Our example is the FuriosaAI RNGD NPU, programmed in a recent, sparsely documented Rust DSL, where the loop optimized three Gemma-4-12B decode kernels over 12 days on a shared, scarce device. The loop builds its own target knowledge: it writes and checks a summary of the DSL documentation, measures the chip with microbenchmarks, records compiler-derived legality rules, and revisits this knowledge when the toolchain changes, rechecking rules and measurements where it can. A role-based search over per-kernel Design Trees explores whole-kernel designs and accumulates small accepted changes in the best version of each kernel. Because device time is scarce and readings are noisy, cheap checks screen new designs before any device run, and late in the run a change was accepted only after repeated comparisons with the best kernel on the device. The final kernels reach a 9.293x geomean speedup over the official baseline, each within 1.25x of the cycles needed just to move its weights. Our prompts, profiler and acceptance test are available at this https URL. 

---
# Extremely Sparse-View Computed Tomography from 2D Projections via Pose-Aware Diffusion Priors 

**Authors**: Linrui Dai, Jinxiu Liang, Chu Zhou, Mae Yamaguchi, Ryuya Takahashi, Imari Sato  

**Link**: [PDF](https://arxiv.org/pdf/2610.38319)  

**Abstract**: Extremely sparse-view computed tomography (xSV-CT) is critical when scanning time, motion, or radiation tolerance limits each scan to only a few projections. In many such scenarios, the same acquisition limits also constrain dataset construction, making pre-reconstructed 3D volumes from dense scans unavailable. This exposes a gap in current sparse-view approaches: Optimization-based methods, which use the sparse measurements directly but lack data-driven priors to suppress artifacts, and learning-based methods, which use strong priors but usually require densely scanned training volumes acquired outside the xSV-CT regime. We study this train-sparse / infer-sparse setting by learning a pose-aware diffusion prior from one 2D projection per training sample, then guiding diffusion sampling with measured-view fidelity, shared-volume reprojection, and projection-physics consistency for tomographic reconstruction. On two public CT datasets across 1-, 4-, and 8-view settings, the proposed method achieves the best novel-view synthesis scores and best or competitive reconstruction quality compared with baselines that use dense-volume supervision. 

---
# What Should a Navigator Remember? Spatiotemporal Memory for Zero-Shot Vision-Language Navigation 

**Authors**: Yuhang Peng, Zhikai Wang, Ziyi Wang, Jiaru Zhong, Xiaoji Zheng, Jingzhi Huang, Yi Wang, Jiangtao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2610.38318)  

**Abstract**: Zero-shot vision-and-language navigation (VLN) requires an agent to use past observations without navigation-specific training. We study what information an 8B-scale navigation system should retain and how it should be organized for subsequent decisions. Our agent combines instruction progress, hierarchically compressed observation history anchored to poses, and accumulated geometry in complementary task, temporal, and spatial memories. An 8B observer summarizes observations, and an 8B reasoner uses the memories to select waypoint intents or request additional views, a geometric controller executes the selected targets. A semantic information perspective motivates the distinction between retaining evidence, making it accessible to the reasoner, and acquiring further observations. On R2R-CE val-unseen, the complete system achieves the highest among the evaluated Qwen3-based zero-shot baselines. Removing temporal memory reduces SR and sharply increases average path length from 16.9 to 39.1m, removing the spatial-map and planning component reduces SR as well. Real-world experiments on a Unitree Go2 show that our zero-shot method achieves higher success rates than the post-trained baseline in both corridor and office environments, suggesting better transfer to the evaluated real-world settings. As a complementary evaluation, LAW2 uses paired instruction interventions to examine behavioral responses, instruction following ability beyond endpoint success. These results support preserving task progress, observation history, and geometry in structured memory for zero-shot navigation with smaller open models. 

---
# LLM-PowerTrace: A Dataset of Time-Resolved GPU Power Traces for AI Compute 

**Authors**: Jiansheng Wang, Jaskeerat Singh, Haochen Xu, Xinyi Li, Mingyuan Yan, Shaohui Liu, Yuzhuo Li, Yize Chen, Wenqi Cui  

**Link**: [PDF](https://arxiv.org/pdf/2610.38315)  

**Abstract**: AI compute is becoming large and fast-changing electrical loads. Its impacts on data center infrastructure and power grids depend not only on total energy consumption, but also on how their power draw moves over sub-second to minute timescales, including peaks, ramps, and synchronized fluctuations across accelerators. Existing energy benchmarks for large language models (LLMs) report aggregate quantities such as joules per token and rarely release the underlying time-series measurements, leaving these power dynamics poorly characterized and difficult to model. We introduce LLM-PowerTrace, an open dataset of time-resolved, per-GPU power traces for LLM fine-tuning and inference, sampled every 100\,ms on varying NVIDIA platforms paired with complete workload and system metadata. For fine-tuning, LLM-PowerTrace profiles open model families spanning a wide range of model sizes and fine-tuning methods, together with controlled sweeps over batch size, sequence length, precision, parallelism, and checkpointing. For inference, it covers variations in concurrency, GPU frequency, KV-cache and weight quantization, prompt and output lengths, and request-arrival patterns. LLM-PowerTrace profiles a set of open model families across a wide range of model sizes, and controlled sweeps over batch size, sequence length, precision, parallelism, and checkpointing for fine-tuning, and it sweeps concurrency, GPU frequency, KV cache and weight quantization, prompt and output lengths, and request arrival patterns for inference. Built upon power traces, we define power-grid-facing metrics as well as AI-compute power\&energy scaling law for extrapolating beyond the measured settings. Together, LLM-PowerTrace provides a dataset foundation for modeling, forecasting, and managing the fast-changing electrical behavior of AI compute. 

---
# EPSBench: Benchmarking LLM Agents for Earnings Forecasting 

**Authors**: Yanbin Yin, Chuyu Xu, Liqin Ye, Jing Peng, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.38293)  

**Abstract**: Quarterly earnings forecasting requires an agent to connect financial evidence to the income-statement line items that produce a company's reported earnings. We introduce EPSBench, a benchmark for studying this ability across 149 firm-quarters from 86 US companies. Agents read a fixed collection of filings, earnings-call transcripts, and news, then estimate company-specific income-statement drivers with cited evidence. A shared accounting calculator converts these estimates into GAAP diluted earnings per share (EPS), providing a common basis for comparing forecast assumptions. We evaluate six models under a common harness, assessing point forecast error relative to date-matched analyst consensus and checking whether their prediction intervals achieve the stated coverage. Three models show no detectable paired difference in point forecast error from consensus, and three have higher error. Across all six models, nominal 80% intervals contain reported EPS in only 37--70% of runs that return an interval, and most misses fall above the upper bound because the forecasts run low. The driver records trace most of the error gap between models to operating-income estimates. 

---
# Vendster: An Evaluation Arena to Investigate How Forecasting Tool Quality Affects LLM Agent Behaviors in Buyer-Vendor Negotiation 

**Authors**: Charin Polpanumas  

**Link**: [PDF](https://arxiv.org/pdf/2610.38292)  

**Abstract**: Time-series forecasters are ranked almost entirely by accuracy metrics, but a forecast's decision value, its effect on the downstream decision, need not track accuracy and is rarely measured. This problem compounds as large language model (LLM) agents increasingly consume forecasters as tools, mediating them through stochastic, opaque policies. We present Vendster, an evaluation arena measuring how a forecaster's bias and prediction-interval width change the outcome of a buyer--vendor discount negotiation between two LLM agents. In the reference instance Vendster-M5, sampling base demand patterns from the M5 dataset, we evaluate two frontier-closed LLM families (Claude Opus 4.6 and GPT-5.6 Sol) and two open ones (DeepSeek-V3.2 and Qwen3-235B-A22B). Model-generated forecasts raise sales target attainment by up to +17 percentage points (p<0.001), yet standard forecast evaluation metrics such as relWQL and MASE do not predict negotiation outcomes at a statistically significant level. To isolate the causal structure, we substitute in synthetic forecasts with controlled bias and interval width. Under-prediction bias significantly improves outcomes for agents that react more to the forecasts---dose-response slopes -0.387 and -0.416 (p<0.001). Interval width has no statistically significant effect at zero bias. These findings indicate that the direction of forecast bias and agent forecast-reactivity are more strongly associated with downstream outcomes than forecast accuracy. Vendster is fully customizable and released under CC BY-NC 4.0 at this https URL. 

---
# The Role of Quantization in Loss Spikes 

**Authors**: Zaiqian Chen, Aditya Somasundaram, Sophie Li, Sean McLeish, John Kirchenbauer, Tom Goldstein, Micah Goldblum  

**Link**: [PDF](https://arxiv.org/pdf/2610.38277)  

**Abstract**: Loss spikes in language model pretraining can waste substantial compute and derail otherwise healthy training runs. A primary driver of these instabilities is the growth and divergence of output logits in the final language modeling (LM) head. In this work, we study a particularly costly form of this instability associated with output logit growth: delayed, terminal divergences that emerge only after prolonged training, which we call time-bomb spikes. We find that these instabilities are preceded by rapidly growing quantization error in the LM head, which eventually culminates in a loss spike. Crucially, reducing numerical precision accelerates time-bomb spikes, while increasing precision improves stability. We then propose a simple yet effective mitigation that replaces AdamW on the LM head with SGD with momentum, suppressing logit growth and improving stability over long training horizons. Finally, we find that existing mitigation techniques from the literature (along with our own) all reduce quantization error, suggesting a shared numerical mechanism behind their effectiveness. Together, our results identify finite precision error as a hidden driver of time-bomb spikes and show that suppressing this error is key to stable long-horizon pretraining. 

---
# Cross Lingual Abstractive Summarization for Marathi 

**Authors**: Asang Dani, Shailesh R Sathe  

**Link**: [PDF](https://arxiv.org/pdf/2610.38267)  

**Abstract**: Text summarization is an important task for advancing natural language processing (NLP) research and applications in Indic languages. Despite progress in extractive and abstractive summarization, both monolingual summarization (MS) and cross-lingual summarization (CLS) remain relatively underexplored for these languages. Recent advances in Transformer architectures, pretrained language models (PLMs), and dataset availability offer new opportunities to address this gap. In this work, we construct cross-lingual training data from the monolingual Marathi MahaSum corpus and combine it with the smaller cross-lingual PMIndiaSum dataset to fine-tune IndicTrans2, an Indic neural machine translation model. We evaluate the resulting models using ROUGE, chrF and chrF++ metrics, focusing primarily on Marathi-to-English summarization due to computational constraints. We also discuss the applicability of our approach to the English-to-Marathi direction. Our experiments demonstrate that leveraging a pretrained translation model yields substantially better summarization performance than the IndicBART and mBART-50 based baselines reported in previous work on cross-lingual summarization. 

---
# A Tutorial on Generative Models for Music Audio 

**Authors**: Mateusz Modrzejewski, Joanna Kurkowska, Dorien Herremans  

**Link**: [PDF](https://arxiv.org/pdf/2610.38252)  

**Abstract**: Music audio generation spans instrumental clips, full songs with vocals, edits of existing recordings, and systems that play alongside a musician in real time. This monograph explains the methods behind these systems for readers familiar with neural-network training who are new to music generation. It introduces neural audio codecs and autoencoders, autoregressive and masked token models, diffusion and flow matching, sampling acceleration, and preference optimization in a shared notation. Worked examples connect training objectives and sampling procedures to published systems. The application sections relate the musical requirements of instrumental generation, songs, editing, and live performance to model design. Comparison tables record representations, methods, inputs, tasks, and release information documented in papers and official sources. Symbolic composition, speech, and sound effects are included where they connect to music audio. We explain how to interpret automatic metrics and learned judges, design and report listening tests, and assess evidence about data provenance, memorization, attribution, and output detection. Selected legal decisions and disputes illustrate the implications for training and release. The reviewed systems combine established generative paradigms with new representations, planning stages, and post-training procedures. Differences in data, scale, and evaluation limit what cross-system comparisons can establish about the sources of reported gains. 

---
# Precedent or Intent? What Coding Agents Follow When Evolving Existing Code 

**Authors**: Haichao Zhu, Geyang Xu, Zhaorui Yang, Jiyuan Wang, Qian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.38244)  

**Abstract**: Existing code reveals past implementation decisions but rarely explains the reasoning behind them. When coding agents implement new features or modify existing functionality, past decisions may conflict with new task instructions. This raises a fundamental question for agentic software evolution: when multiple implementations satisfy the same functional requirements, how do coding agents decide whether to follow existing code or new task instructions? We investigate this question through bidirectional, controlled experiments in purpose-built repositories, followed by validation in real-world open-source repositories. Our approach isolates three competing sources of guidance: (1) agents' default preferences, (2) repository precedents, and (3) explicit task instructions. We conduct 5,625 trials across four coding agents and six implementation pairs, with validation in four open-source repositories spanning three programming languages. We find that agents share the same default implementation choices, yet overwhelmingly follow repository precedent when it conflicts with those defaults (359/360 trials). Crucially, not all precedents carry equal influence: three supporting local files yield 96% adoption, compared with only 4% for sibling-resource files, and changing file placement alone can shift adoption from 0% to 100% at a fixed example count. When new task intent conflicts with existing code, written priority rules achieve substantially higher success than code examples alone (59/60 versus 38/60 joint successes for two agents on two implementation pairs). The effectiveness of explicit rules also extends to open-source repositories, where a harness-loaded rule raises joint success from 0/120 to 117/120. These findings motivate design-decision traceability that preserves not only implementation choices but also the rationale and priorities that should guide future changes. 

---
# MEV: A Multi-Event Video Dataset for Long-Take Generation 

**Authors**: Peiyuan Zhu, Shaoan Xie, Yifan Shen, Wen Tian, Zijian Li, Guangyi Chen, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.38241)  

**Abstract**: Current datasets for text-to-video generation are largely limited to short clips depicting a single event or coarse-grained captions. As a result, they provide limited supervision for modeling long-take temporal dependencies, often leading to temporally inconsistent generations when describing multiple events. In this paper, we introduce MEV, a multi-event video dataset designed for long-take video generation. MEV provides high-quality real-world videos where subjects can perform a sequence of distinct, non-overlapping events, featuring precise temporal boundaries and fine-grained procedural annotations. This event-centric discretization reduces confounding factors and enables models to support temporally localized conditioning and build long-take videos from atomic event units. Furthermore, we introduce a comprehensive annotation pipeline that segments videos into temporally grounded events, and generates structured captions at both the global level and the event level. In addition, under an event-centric evaluation protocol, we conduct multi-event generation experiments by fine-tuning on MEV with temporally grounded event conditioning, leading to improved long-take event transitions and more coherent content. We believe this work provides a publicly available resource for multi-event fine-tuning and event-centric evaluation, together with practical insights into learning from temporally structured video-text data. 

---
# The Indian Financial Automated Advice Bias (IFAAB) Dataset : Auditing Socioeconomic and Demographic Disparities in LLM-Generated Financial Advice in India 

**Authors**: Kriti Mahajan  

**Link**: [PDF](https://arxiv.org/pdf/2610.38239)  

**Abstract**: Large language models are increasingly used for financial guidance, yet little is known about whether their advice changes across social identities in Global South contexts. We introduce the Indian Financial Automated Advice Bias (IFAAB) benchmark, a counterfactual audit dataset of 31,734 GPT-4o-generated financial-advice responses for synthetic Indian personas varying by gender, caste, education, employment, residence, age, and district. Holding the underlying request fixed, we evaluate disparities in LLM-implied wealth, income, savings, projected gains, textual similarity, advice topics, and qualitative bias judgments. In the gender-only audit, male-labelled profiles receive 60.1\% higher initial total resources and 46.8\% higher post-advice total resources than comparable female-labelled profiles; in the caste-augmented audit, the post-advice total-resource gap remains 49.4\%. Adding district information reduces the aggregate post-advice total-resource gender gap to 7.9\%, revealing strong geographic conditioning. Standalone caste comparisons also show systematic stratification: relative to General-category profiles, OBC, SC, and ST profiles receive significantly lower post-advice total resources, respectively. Textual audits show low semantic similarity between paired gendered responses, with an average TF--IDF cosine similarity score of 0.44. Qualitative analysis shows that women are more often steered toward cautious saving, community support, and basic financial literacy, while men receive more growth-oriented advice involving investment, entrepreneurship, and asset accumulation. IFAAB provides a reusable benchmark for evaluating fairness, suitability, and consumer-protection risks in LLM-mediated financial advice, and highlights the need for locally grounded, intersectional evaluation of AI systems in high-stakes economic domains. 

---
# CuraEval: Evaluating Demonstration Curation on Mixed-Skill Robot Data 

**Authors**: Jiuyi Xu, Meida Chen, Yang Ye, Yangming Shi  

**Link**: [PDF](https://arxiv.org/pdf/2610.38221)  

**Abstract**: Robot imitation learning uses demonstrations from operators of varying skill, but selecting training data remains challenging. Existing methods score demonstrations using smoothness, diversity, or estimated influence on policy performance, but differences in selection and evaluation protocols make it difficult to determine whether they outperform simple baselines. We introduce \textbf{CuraEval}, a framework combining mathematical analysis of demonstration scores with controlled comparisons against fixed size-matched random subsets and full-data training on mixed-skill robomimic data using Diffusion Policy and BC-RNN. We show that the evaluated PSD score depends only on episode length and squared position magnitudes, and produces nearly the same ranking as episode length on \textit{square} ($\rho=0.999$). With Diffusion Policy, selecting the shortest 50\% of demonstrations improves \textit{square} success from 70.7\% under random selection to 76.5\%. Compared with random selection, our adapted geometric-entropy LOO selector reduces success at 25\% retention from 94.7\% to 74.0\% on \textit{can} and achieves 41.7\% on \textit{square}, below the mean success rates of all three independently sampled random subsets (59.8--62.5\%). On the same 3 \textit{square} Diffusion Policy training seeds, length-based selection at 75\% retention improves success over full-data training from 76.7\% to 86.7\% under the original protocol, but lowers it from 80.5\% to 79.5\% after checkpoint reporting, rollout count, and evaluation seed change jointly. No evaluated curated condition significantly outperforms full-data training after Holm correction, although confidence intervals still permit useful gains. Code is available at this https URL. 

---
# Skeleton Coding: Program Structure as a Specification for Agents 

**Authors**: Aditya Golatkar, Alessandro Achille, Renos Zabounidis, Vlad Sobal, Yangruibo Ding, Baishakhi Ray, Jonathan Buck, Guy Van den Broeck, Maryam Fazel, Pietro Perona, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2610.38214)  

**Abstract**: Coding agents are typically provided end-to-end behavioral specifications that the code they produce should satisfy, such as tests and input--output examples. How many such specifications must a developer provide, and which ones, for the agent to produce the developer's intended program? We argue that this cannot be determined a priori, for it depends on the structure of the implementation, which is left to the agent to decide. An agent first commits to a structure---a decomposition into data types, modules, and functions---which fixes a class of candidate programs; behavioral specifications then select a program within that class. A set of tests that suffices for one structure may leave another badly underdetermined, requiring different edge cases or exponentially more examples. Since the structure is chosen by the agent, developers cannot ensure their specification is complete without reverse-engineering it from thousands of lines of generated code. To resolve this, we introduce Skeleton Coding, in which developers jointly specify the structure---in high-level, code-like format---alongside behavioral specifications. Theoretically, we model AI-assisted coding as a control problem, and show that skeletons can reduce exponentially the number of tests developers must supply to steer an agent toward an acceptable program. Empirically, across four models and three benchmarks, the best-performing skeleton specifications reduce mean test error by 48.4% relative to the best end-to-end behavioral specifications. Even when the agent can query a complete reference implementation, adding a skeleton reduces mean test error relatively by up to 27.0%. Skeletons also narrow the average gap between small and large models by about 52.4% while lowering implementation cost by 15--26x, and reduce code erosion by 24.7% on average. 

---
# Is it my preference? Role Grounding Attack against Agent's Memory via Social Interaction 

**Authors**: Haoran Bu, Yiting Shen, Litian Zhang, Ruihan Hu, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.38209)  

**Abstract**: LLM-based agents now share experiences on social platforms and remember new preferences to enhance their performance. However, existing memory systems contain a vulnerability in role-to-principal attribution that attackers can exploit. We identify Role Grounding Misattribution, in which an attacker-designed interaction causes the victim agent to incorrectly record a preference from the attacker as a preference of the victim's user. Specifically, we are the first to formalize Role Grounding Attack (RGA), an attack model that exploits this vulnerability through nested conversation structures. In the memory formation stage, an experience-sharing post quoting a conversation between the attacker agent and its user is published, which explicitly identifies its publisher. The inner conversation expresses preferences from the user of the attacker agent. When the victim agent remembers preferences from this nested structure, its memory loses the context binding and grounds the attacker's preference to the victim's user instead. In the memory retrieval stage, the misattributed preference persists and influences the victim agent's decisions when it encounters relevant scenarios. We instantiate RGA as tool selection manipulation and privacy leakage. Tool selection manipulation causes a tool-call preference to be misattributed to the victim's user, steering the victim toward an attacker-controlled tool. Privacy leakage extends this by misattributing both a tool-call preference and a task-requirement preference to the victim's user. The victim selects the attacker's tool, then includes sensitive information in tool arguments to satisfy the task requirement. We evaluate RGA across five mainstream memory systems and six language models. Tool manipulation and privacy leakage both achieve high attack success rates. These results call for stronger role grounding. 

---
# Convex-SKM-Net: Low-Latency Block Halfspace Corrections for Neural Convex Constraint Satisfaction 

**Authors**: Haoyu Zhu, Xiaochen Zhang, Qingchun Hou  

**Link**: [PDF](https://arxiv.org/pdf/2610.38197)  

**Abstract**: Fast inference and strict constraint satisfaction are essential for learning-based decision-making. Existing methods often rely on constraint-specific projections or feasible-set parameterizations, while iterative repair can be costly. We introduce Convex-SKM-Net, a convex constraint-satisfaction layer that forms linearization halfspaces from residual and (sub)gradient oracles without constraint-specific inequality projectors. It applies fixed-depth dual projected-gradient descent (dual PGD) to minimum-displacement block problems using GPU-friendly matrix operations. We jointly learn the predictor and a positive-definite mobility, adapting the correction geometry to improve objective quality. We prove convergence of the outer iterates at fixed dual depth and linear convergence in distance under Slater's condition and bounded subgradients. All our test outputs on three second-order-cone model-predictive control (SOC-MPC) horizons and six power-system convex relaxations satisfy a $10^{-3}$ feasibility tolerance. Comparisons with the latest constraint-satisfaction methods show inference speedups of up to $144.3\times$ on SOC-MPC and $7.7\times$ on a convex quadratically constrained quadratic programming (QCQP) benchmark with joint training. On QCQP, the mean objective gap is $0.012\%$, with full test feasibility at $10^{-4}$. Learning mobility reduces the five-seed affine-task mean gap by $22.4\%$ and improves gaps in matched power-system comparisons. 

---
# Scaling Lean Theorem Proving for Real-World Code Verification 

**Authors**: Debargha Ganguly, Divya Raghunathan, Dejan Jovanović, Vikash Singh, Soonho Kong, Aaron Tomb, Aman Goel, Ferhat Erata, Sam Bayless  

**Link**: [PDF](https://arxiv.org/pdf/2610.38174)  

**Abstract**: We present EasyProver, a Lean proving system built from recursive agents with isolated contexts, asynchronous search, parallel tactic probing, and an elastic elaboration service. Using an off-the-shelf frontier model with no additional training, it proves 669/672 PutnamBench problems and all six IMO 2026 formalizations and resolves 597/600 NTP4VC verification conditions and all 189 VERINA tasks by proof or refutation. Because these benchmarks are close to saturation, we also introduce Aegis, which converts competitive programming problems into much harder verification tasks over real implementations. The pipeline writes safe Rust solutions, translates them into Lean with Aeneas, generates specifications, and labels each frozen statement with a kernel-checked proof or refutation. It produces 2,045 differentially validated Rust-Lean program pairs and 11,533 candidate specifications. Unlike hand-written benchmarks, these tasks carry implementation-level obligations from fixed-width arithmetic, explicit failure, generated loops, casts, and functionalized mutation. On a stratified sample of 2,000 statements, EasyProver proves 1,054 (52.7%) within a 30-minute budget per statement. The same tasks can also be used to benchmark other provers and as rewards for reinforcement learning. 

---
# Estimating Implicit Expected Log Density Ratios 

**Authors**: Avi Amalanshu, Chandler Squires, Yizhou Lu, Pradeep Ravikumar  

**Link**: [PDF](https://arxiv.org/pdf/2610.38151)  

**Abstract**: When exact probabilities cannot be computed, \emph{density ratio estimation (DRE)} is a key subroutine in the wide range of applications where density ratios appear, from experimental design to generative modelling and reinforcement learning. In this paper, we introduce the concept of an \emph{expected log density ratio (ELDR)}: log density ratios that are integrated under some third, possibly distinct measure. We show many problems with implicit densities where DRE is naïvely used are actually ELDR estimation tasks, and prove fundamental limitations on finite-sample ELDR estimation. Based on these results, we argue that the typical DRE-based approaches are unsuitable for ELDR estimation, and provide some foundational motivation towards bespoke ELDR estimators. To demonstrate, we propose a simple modification to some DRE methods that allows samples convergent to the integral distribution to appear in training data. We show through benchmarks that this can improve ELDR estimation error. 

---
# Answer-Only Disclosure: Private-Context LLM Assistance over Web APIs with a Leakage Ledger 

**Authors**: Liang Yuan, Xiaohu Du, Ruikun Luo, Jinze Wang, Mashud Rana, Kaibin Wang, Jiangwen Zhou, Bo Li, Xiaoyu Xia  

**Link**: [PDF](https://arxiv.org/pdf/2610.38126)  

**Abstract**: Web assistants increasingly answer questions about a user's private documents by pairing a small on-device model with a cloud LLM behind a Web API. In collaborative decoding, the document stays on the client, and the cloud receives only a public view that the user approves, such as the question alone, but the two models exchange messages while the answer is generated. We show that the message transcript can disclose private content that neither the public view nor the answer contains. In a planted-secret audit, speculative decoding with private drafts did not include the secret code in its answers, yet sent the code to the cloud in rejected drafts. We present ADLER, a decoding protocol for streaming LLM APIs in which every message to the cloud depends only on the public view and the answer the user sees, regardless of how the client combines the two models or how the cloud behaves. A client-side leakage ledger then bounds how much the answer itself reveals relative to a public reference process built from the cloud's replies. In the audit, no ADLER configuration exposed the secret beyond the answer. On GSM8K math problems with numerical values masked in the cloud's input, ADLER with an unlimited leakage budget raises the exact-match accuracy of a 0.5B client from 0.41 to 0.83 and of a 1.5B client from 0.67 to 0.85 under lenient answer parsing, close to the 0.92 achieved by uploading the full problem. It also exceeds a zero-leakage program baseline (0.69) by 0.14 and 0.16, respectively. In this baseline, the cloud writes a program that the client runs offline using the private numbers. The ledger records the disclosure charge associated with this gain; budgets of 16 nats or less leave accuracy near the cloud-only baseline on GSM8K. 

---
# Beauty Is Human: When People Love AI-Generated Art, They Think a Human Made It 

**Authors**: Zihan Chen, Di Zhu, Lei Nico Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.38113)  

**Abstract**: Generative AI now produces images and poems that people cannot reliably distinguish from human-made work, so people often judge for themselves who made a work. We examine how these perceived-authorship judgments relate to aesthetic preference. In a public online AI Art Turing Test, 11,123 people judged whether each of 50 pictures (26 AI-generated) was made by a human and were asked for their favourite. Among the 9,152 respondents who judged their favourite, a picture was 22.1 percentage points more likely to be judged human-made when it was the respondent's favourite, holding respondent and picture fixed. The association was similar for AI-generated and human-made favourites, was positive for all 50 pictures, and appeared under every reported detection strategy. Respondents most negative about AI art chose fewer AI-generated favourites (47% against 60% among the most positive) but more often believed them human-made, partly because others also judged those pictures human. As a result, in every opinion group a third to two fifths of respondents named an AI-generated favourite they believed human-made. A reanalysis of a published experiment on AI-generated poetry, in which 230 people rated each poem before judging its authorship, found the same pattern, although the AI poems were rated higher. When provenance is hidden, liking and perceived human authorship travel together, consistent with aesthetic response serving as evidence of a human maker and with beliefs about the maker shaping enjoyment. The findings qualify research on bias against AI art, which typically uses explicit labels, and inform AI disclosure policy. 

---
# From the Gettier Machine to Justified True Output: A Three-Step Account of Rational Trust in Generative AI 

**Authors**: Tevfik Uyar  

**Link**: [PDF](https://arxiv.org/pdf/2610.38112)  

**Abstract**: As generative artificial intelligence (AI) settles into everyday practices of inquiry, an old epistemological question returns in a new guise: when does a proposition a user takes from an AI output become something the user knows, and when is it rational to trust it as such? This paper argues that the characteristic failures of AI-mediated inquiry are precisely those that the justified-true-belief (JTB) tradition and its post-Gettier anti-luck refinements were built to diagnose, and it advances three claims. First, AI outputs inherit neither the presumptive warrant of testimony nor the reliabilist warrant of deterministic tools: unaudited trust in a calculator rests on a type-level reliability record, and for general-purpose generative models there is no type-stable process to which such a record could attach -- a practical form of the generality problem. Second, use that stops at obtaining an output turns generative AI into a Gettier machine: the user's fluency-based seeming of justification is disconnected from the truth of what is believed, and, as a clock running ten times too fast illustrates, sheer speed of production multiplies coincidental hits, inflating the appearance of reliability without improving the standing of any single hit. Third, the remedy is a three-step structure of use -- acquisition, understanding, independent verification -- whose first two steps answer to the classical belief and justification conditions and whose third answers to the post-Gettier sensitivity condition. The result is a stakes-indexed criterion of rational trust, justified true output (JTO), with an everyday test: "Had this been false, would I have noticed?" The criterion is defended against objections from over-demandingness, the extended mind, model improvement, and redundancy. 

---
# The Prestige: Benchmarking Cognitive Visual Reasoning using Magic Tricks 

**Authors**: Shuo Wen, Beixi Du, Edwin Meriaux, Junming Shi, Chloe Si, David Meger, Doina Precup, Gregory Dudek  

**Link**: [PDF](https://arxiv.org/pdf/2610.38103)  

**Abstract**: We introduce The Prestige, a video understanding benchmark that probes cognitive visual reasoning through professional magic performances, where deliberate misdirection challenges models to track events and explain unexpected outcomes. The benchmark comprises open-ended questions ranging from temporal and causal understanding to robustness against adversarial questions. Evaluation of leading open-source and proprietary VLMs reveals a substantial gap between even the strongest model and human performance. We also find systematic discrepancies between human and automated grading and identify recurring model failure modes. Ablations further show that models can select correct multiple-choice answers even when they fail to generate correct explanations independently, motivating further investigations into question formats. Together, these findings highlight limitations in both VLM reasoning and its evaluation, underscoring the need for more reliable video understanding models and more rigorous benchmarks. 

---
# Calculation and rejection diverge in language models tested at known mechanical limits 

**Authors**: Shaoliang Yang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.38096)  

**Abstract**: When a language model returns engineering quantities with the status "solved", software that trusts it passes them on, even when the requested design is physically impossible. Such failures could reflect poor calculation, how the answer is requested, or a failure to judge feasibility. We set beam loads and disk speeds at known percentages above or below plastic collapse or burst, measured each model's limit calculation separately and tested 12 models in 4,860 calls, including follow-up experiments on new designs. We find that a model can recognize an impossible design when asked only whether it is feasible, yet accept the same design, reporting "solved", when asked for a full answer. Three models accepted beams 10 to 30% beyond collapse despite precise limit calculations in separate questions, and many accepted replies stated the violation. Asked only about feasibility, Opus 5 and Gemini 3.8 Flash rejected every impossible new design they were given, including those accepted in full answers, without false alarms. Rewording when to decline or changing the answer format raised rejection on the original beams, but not consistently on new designs. Software and benchmarks using a model's status should test feasibility judgment and full-answer rejection separately, and report the exact prompt. 

---
# Nemotron 3.5 Super and Lightning: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Models for Agentic AI 

**Authors**: NVIDIA, Aastha Jhunjhunwala, Aayushi Ashokkumar Srivastava, Abhibha Gupta, Abhinav Garg, Abhinav Khattar, Achsah Wildish, Adam Rajfer, Adi Renduchintala, Adil Asif, Adriana Flores Miranda, Aileen Zaman, Aishwarya Chhabra, Akanksha Shukla, Akhiad Bercovich, Alejandra Rico, Aleksa Gordic, Aleksander Ficek, Alex Gronskiy, Alex Steiner, Alexander Bukharin, Alexandre Milesi, Ali Roshan Ghias, Ali Shehper, Ali Taghibakhshi, Alice Gatti, Alisa Liu, Alok Kumar, Alok Tripathy, Amala Sanjay Deshmukh, Aman Miryala, Aman Shanbhag, Ameya Sunil Mahabaleshwarkar, Ameya Velingker, Amir Klein, Amir Samani, Amit Zuker, Amnon Geifman, Anahita Bhiwandiwalla, Ananth Subramaniam, Andre Manoel, Andrea Santilli, Andrew Tao, Andrii Skliar, Anil Thomas, Anish Mahishi, Anna Shors, Anna Warno, Anuj Doshi, Anusha Kamath, Anwith Kiran, Ao Tang, Arham Mehta, Arnab Dhara, Arnav Komaragiri, Arnav Kundu, Arti Jain, Arun Rao, Arun Venkatesan, Arushi Goel, Asha Anoosheh, Ashish Vaswani, Ashwath Aithal, Asif Ahamed, Asma Kuriparambil Thekkumpate, Aswin Visva, Atefeh Sohrabizadeh, Avinash Vem, Ayush Dattagupta, Benedikt Schifferer, Benjamin Chislett, Bernard Nguyen, Besmira Nushi, Bilal Kartal, Bita Darvish Rouhani, Bobby Chen, Boris Ginsburg, Boxin Wang, Brandon Norick, Branislav Kisacanin, Brendan Johnson, Brian Dang, Brian Yu, Bryan Catanzaro, Byung-Kwan Lee, Carlo del Mundo, Chankyu Lee, Chantal Hwang, Charles Wang, Charlie Truong, Chen Cui, Cheng Wang, Cheng-Ping Hsieh, Chenhan Yu, Chenjie Luo, Chintan Patel, Chris Alexiuk, Chris Johnson, Chris Wing, Christian Munley  

**Link**: [PDF](https://arxiv.org/pdf/2610.38095)  

**Abstract**: We introduce Nemotron 3.5 Super and Nemotron 3.5 Lightning, open and efficient Mixture-of-Experts Hybrid Mamba-Transformer models designed for reasoning, long-context understanding, and agentic workloads. Super is a 120B-parameter, 12B-active multimodal model, while Lightning is a smaller 30B-parameter, 3B-active text model; both designed for efficient deployment. Both models combine large-scale distillation, long-context training, and native speculative decoding, while Super further incorporates Indic continued pretraining and agentic software-engineering mid-training. We develop a comprehensive post-training pipeline spanning supervised fine-tuning and reinforcement learning (RLVR and RLHF), complemented in Super by specialized teacher training and multi- teacher on-policy distillation (MOPD). This transforms Super from a text-only base model into a multimodal agent capable of reasoning over text, images, video, and interactive tool environments. We further explore test-time scaling through parallel agent execution and self-verification. Across training and deployment, we jointly optimize capability and efficiency. Indic continued pretraining improves Indian-language benchmarks by up to 9.82 percentage points, while agentic mid-training strengthens software-engineering capabilities and robustness across agent harnesses. The post-trained Super model achieves 51.78% on SWE-bench Pro, surpassing Nemotron 3 Ultra despite using approximately 4.6x fewer active parameters. Post-training MTP boosting improves speculative- decoding acceptance length by up to 21.8% under probabilistic draft sampling, while NVFP4 quantization achieves 99.62% median accuracy recovery relative to BF16. Inference optimizations enable up to 4,856 output tokens/s/GPU on GB200 while sustaining at least 50 tokens/s/user. 

---
# Jailbreak Connectivity: Towards Diverse, Transferable, and Universal Multimodal Large Language Model Jailbreak 

**Authors**: Xu Zhang, Ziqing Hu, Shuo Han, Ren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.38091)  

**Abstract**: Multimodal large language models (MLLMs) are vulnerable to visual jailbreak attacks that bypass safety safeguards. Existing methods typically optimize isolated adversarial examples, limiting attack diversity, transferability, and generalization. We propose Jailbreak Connectivity (JC), a framework based on the observation that effective adversarial inputs can be connected through low-loss regions in image space. JC optimizes continuous adversarial paths: path sampling yields diverse jailbreak images, lightweight surrogate classifiers guide cross-model transfer, and a distribution-level objective enables universal attacks across harmful queries. On MM-SafetyBench, JC achieves average attack success rates of 84.01%, 85.77%, and 97.95% on Qwen3-VL-8B-Instruct, InternVL3.5-8B, and Gemma-3-12B-IT, respectively, exceeding the strongest baselines by up to 68.36 percentage points. In black-box evaluations, JC improves attack success rates over FigStep by 24.69 and 27.34 percentage points on GPT-5.6 and Gemini 3.1 Pro, respectively. These findings reveal structured vulnerabilities in MLLM safety alignment and inform more comprehensive security evaluation. 

---
# Predicting What Answer Engines Cite: Open Decision Models, Rerankers and a Distilled 4B Student on Ten Engines under Controlled Retrieval 

**Authors**: Elisha Bajemon, Andre-Louis Rochet  

**Link**: [PDF](https://arxiv.org/pdf/2610.38090)  

**Abstract**: Generative engine optimization needs a scorer that ranks candidate pages by the share of an answer they will receive as citations. We measure pretrained rerankers, a hosted typed decision model (Jev), eight open decision models and distilled students on GEO-Bench queries answered by ten engines that receive the same candidate sources in their prompt. Where several engines answer the same query, they agree with one another at a within-group Spearman correlation of 0.704. Jev and two open 27B decision models, used without training, are the strongest scorers at 0.575 to 0.578. A Qwen3-Reranker-4B fine-tuned with low-rank adapters on teacher labels for other GEO-Bench queries, released as GEO Citation Ranker 4B, reaches 0.542 to 0.549 across three seeds, and its differences with the 27B models and with Jev are not distinguishable from zero on 229 queries. Most of the remaining distance between the student and Jev disappears when Jev scores one passage at a time, as the student does. Students trained on teacher labels and on observed engine citations reach the same level, and labels from Jev alone match the two-teacher target at 0.6B. We release the model, the benchmark, the predictions of every reported scorer and the code. 

---
# What Can Firms Do in the Face of Model Distillation? 

**Authors**: Diptangshu Sen, Jibang Wu, Juba Ziani  

**Link**: [PDF](https://arxiv.org/pdf/2610.38088)  

**Abstract**: Model distillation has become a major concern in the rapidly growing generative AI discourse. In its simplest form, distillation is the phenomenon where smaller firms can use data distilled from transcripts of conversations with powerful proprietary closed-weight models to train their own models, thus achieving reasonably high model quality but at much lower costs. Besides the legitimate safety and security concerns associated with distillation, another major issue is how it can also distort marketplace outcomes. This motivates us to initiate an exploration of the problem from an economic theory lens. We consider a setting with two players: an incumbent and a distilling competitor and characterize how the incumbent's model characteristics drive the incentives around how much the competitor should distill. We also consider the impact of distillation on key market outcomes like user-side surplus and the incumbent's utility. Our key contribution, however, is to study interventions for preventing or limiting distillation attacks. In this context, we study two settings: i) when it is possible, with a certain probability (which grows in the extent of distillation conducted), to detect an attack on the incumbent's model; and ii) when a distillation attack cannot be reliably detected. In the first setting, we show that penalty-based regulations can be very effective in completely deterring distillation, although it can have potentially adverse effects on market competition and user surplus. In the latter setting, we show that the incumbent can use self-distillation as a mitigation strategy. Importantly, the effectiveness of self-distillation depends on how cost-efficiently the incumbent can deploy a new self-distilled model on the market. On the positive side, self-distillation can preserve market competition thus protecting user surplus to some extent. 

---
# Interpretable Feature Discovery from Unstructured Data Using LLMs and Statistical Choice Models 

**Authors**: Baptiste Arnaudo, Amine Bennouna, Sébastien Martin  

**Link**: [PDF](https://arxiv.org/pdf/2610.38082)  

**Abstract**: Feature extraction from unstructured data typically relies on opaque embeddings unsuited to interpretable decision-making. Interpretable features, on the other hand, typically require costly, and often infeasible, hand-engineering. Large language models (LLMs) offer a natural bridge, as they can read raw data and describe it in interpretable terms. One could simply ask an LLM to extract features from a large sample of the dataset, but context limitations and diluted attention can lead to unsatisfactory results. We introduce a method that automatically discovers interpretable, decision-relevant features from unstructured data. Our key idea is to perform pairwise comparisons between items, asking an LLM to articulate, in interpretable terms, the features distinguishing each pair. We treat these explanations, or rationales, as data: we develop a statistical choice model that combines these rationales across the comparisons to identify relevant features and estimate per-item, per-feature scores. We first test our method on a synthetic hiring dataset of cover letters with known ground-truth features and show that it significantly outperforms direct LLM feature extraction in feature recovery. We then apply it to the ChangeMyView (Reddit) debate dataset, where it discovers several interpretable features associated with persuasion while reaching predictive accuracy comparable to established baselines. We also show how the same rationales can be used to audit LLM decisions, yielding a holistic score whose decomposition over interpretable features reveals what the LLM rewards. 

---
# Prompt Lottery in LLM-as-a-Judge: Measuring and Mitigating Sensitivity to Syntactic Rephrasing 

**Authors**: Ilija Subasic, Brett Levert, Andrew Rabinovich, Sajjad Pakdamansavoji, Zhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.38076)  

**Abstract**: LLM-as-a-Judge has become a standard for evaluating large language models, yet most approaches implicitly assume that a single judge prompt functions as a stable evaluation instrument. Prior work has primarily explored deliberate prompt modifications such as scoring rubrics, role descriptions, or chain-of-thought reasoning in order to optimize accuracy. In contrast, we investigate whether validated prompt variants intended to preserve evaluation semantics while differing in surface realization (\textit{paraphrastic rephrasings}) yield consistent judgments under a fixed evaluation task. We introduce \textit{Prompt Lottery}, a phenomenon in which paraphrastic rephrasings produce different LLMaJ judgments. Across a large-scale study spanning diverse evaluation tasks, judge families, and model backbones, we find that these variations can alter evaluation outcomes and, in some settings, reverse relative model rankings. A user study with 10 human annotators shows that LLMaJ judges exhibit 6-9$\times$ larger agreement degradation under prompt rephrasings than humans, falling below the generally accepted threshold in usability studies. This indicates that single-prompt evaluation is insufficiently robust for model ranking. Finally, we show that multi-prompt aggregation substantially improves robustness by emphasizing judgments that remain consistent across prompt variants. 

---
# From Error Prediction to Causal Influence: Auditing SAE Features in Multiple-Choice QA 

**Authors**: Aria Nourbakhsh, Adelaide Danilov, Oleksandr Marchenko Breneur, Salima Lamsiyah  

**Link**: [PDF](https://arxiv.org/pdf/2610.38067)  

**Abstract**: Sparse autoencoders (SAEs) can predict internal features associated with model errors, but association alone does not establish causal influence. We study this distinction in multiple-choice question answering, where answer preference can be measured by the logit margin between the correct answer and the competing choices. Across eight datasets, we analyze Gemma~3 1B and 4B with Gemma Scope SAEs. We propose a pipeline to filter out the features that have high correlation with format and content artifacts, such as prompt and answer permutation, and evaluate the remaining ones with causal feature interventions. Our results show that SAE activations contain predictive signals about wrongness and uncertainty across models and datasets. After filtering artifact-contaminated features, selected features for causal intervention at question and option spans produce larger absolute margin shifts and a higher wrong-to-correct rate than randomly selected features, indicating greater causal leverage. However, answer flips are rare, and we do not observe stable effect directions across meaning-preserving changes in prompt format and answer order. An adapted HalluSAE baseline produces larger final-token margin changes, but relies on substantially contaminated features and does not significantly improve correction rates. Therefore, the identified features reflect context-dependent answer competition rather than the mechanism of incorrect answer errors. 

---
# Can Event Detectors Leave the Broadcast Camera? Evidence from Tennis 

**Authors**: David Soeiro-Vuong, Andre-Louis Rochet  

**Link**: [PDF](https://arxiv.org/pdf/2610.38042)  

**Abstract**: Ball bounces and racket hits are the events from which tennis statistics are derived, and on monocular video the two are easy to confuse. We release a benchmark of 345 bounce and hit annotations at frame level on eight clips that mix broadcast, fixed elevated, phone and court-level viewpoints, with a frozen development and test split, a confusion-aware matcher, clip-level bootstrap intervals and a blind second annotation of three clips. On it we score supervised precise spotters with published broadcast weights, two zero-shot vision-language models, and RallyArb, an arbitration layer over a ball track and player poses that fits no parameter on event labels. The supervised spotters score near-perfectly on broadcast footage and almost nothing on phone and court-level footage, where the pooled bounce F1 of E2E-Spot is 0.130. RallyArb keeps a bounce F1 above 0.8 on the three phone and court-level clips, two of which were used to tune it, and finds 8 of 11 bounces on the held-out one, but it places hits too far from the contact frame to match human timing. The vision-language models detect events on every viewpoint and miss the contact frame by about 0.1 s. Fine-tuned leave-one-clip-out from its broadcast weights, the spotter reaches about 0.8 bounce F1 on two phone and court-level clips and stays at 0.2 on a grazing phone angle. The annotations, the evaluator and the frozen predictions of every scored row are released. 

---
# EARS-Info: Information-Aware Event-Driven Reallocation for Multi-Robot Exploration 

**Authors**: Kabeh Mohsenzadegan, Vahid Tavakkoli, Kyandoghere Kyamakya  

**Link**: [PDF](https://arxiv.org/pdf/2610.38040)  

**Abstract**: We introduce EARS-Info, an information-aware extension of Event-Driven Ant Reallocation Search for multi-robot exploration on OpenStreetMap-derived grids. It combines marginal sensing gain, feasible relocation, return-aware routing, sensing reservations, and cached shortest paths. Across 168,000 episodes comparing 30 implementations on eight city maps, EARS-Info achieves the highest mean operational score in every evaluation panel. With eight robots and 3% communication dropout, it improves the score over EARS-Safe by 5.43%, increases weighted discovery from 88.88% to 98.46%, and reduces mean energy use by 2.62%. Score differences are positive in all eight cities ($p=0.0078125$; descriptive 95% city-bootstrap interval $[0.0159,0.0694]$), with no simulated incidents across 5,600 EARS-Info episodes. These results support the complete controller; individual component advantages remain uncertain. Generalization is limited by reused geography, one training replicate, coarse maps, and adapted comparison methods. 

---
# Measuring Deep Learning Performance - An Empirical Study of Performance Distributions Across Architectures and Tasks 

**Authors**: Kevin L Coakley, Odd Erik Gundersen  

**Link**: [PDF](https://arxiv.org/pdf/2610.38037)  

**Abstract**: Context: Non-determinism in deep learning algorithm design and implementation leads to performance variation, meaning model performance is not a single value, but rather a distribution. These model performance distributions are underexplored despite their impact on robustness. Objective: We investigate the robustness of deep learning performance to sources of non-determinism, specifically focusing on how performance distributions differ across various architectures and tasks. Method: We conducted 186 experiments on image classification and time series forecasting architectures. Each experiment was run 100 times with different random seeds to generate performance distributions, resulting in 18,600 runs. Robustness was quantified using metrics for spread, symmetry, and tail risk. Results: Performance distributions are frequently non-Gaussian, particularly in time series forecasting. Model size does not systematically affect robustness -- larger image classification models show fewer outliers but not lower spread, while smaller time series models show lower spread but more extreme underperformers. Training duration does not scale linearly; early stopping effectively balances performance and robustness. Mean performance does not predict robustness - time series forecasting shows moderate correlation while image classification shows none. Time series models produce nearly three times more underperforming outliers than image classification models, indicating substantially higher tail risk. Conclusions: Tail risk poses serious concerns for Trustworthy AI in high-stakes applications. Models performing well on average may exhibit long tails and extreme outliers revealed only through distributional analysis. Mean performance alone should not guide model selection; assessment of spread, symmetry, and tail risk is essential for reliable model assessment where consistent performance is critical. 

---
# METATRON: A Promptable Whole-Body Execution Layer for Humanoids 

**Authors**: Chen Tessler, Yifeng Jiang, Alec Otto, Yi Shi, Umar Iqbal, Gal Chechik, Xue Bin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2610.38027)  

**Abstract**: Humanoid controllers often require task-specific training despite sharing the need for balance and whole-body coordination. We present Metatron, a promptable whole-body execution layer combining a generative base policy with a programmable skill layer. The base policy receives partial goals, collectively called a prompt, that can combine timed body-part targets, velocity commands, and style cues. It produces diverse whole-body motions consistent with these goals. Training combines motion capture, synthetic recovery trajectories, and retargeted scene-interaction motions. After training, the policy is frozen. Stateful programs called skills express task intent as prompts at each control step, drawing rough geometric sketches that the policy turns into coordinated movement. Skills are ordinary code, refined by coding agents that analyze motion data and fit compact rules without task-specific policy training. On the physical Unitree G1, one frozen checkpoint supports walking, recovery, teleoperated object pickup, and sitting on furniture. Comparative evaluations show higher reaching success and push robustness than the tested reference-tracking pipelines. The same recipe produces climbing and acrobatics on a simulated character. Metatron enables coding agents to turn learned motor capabilities into new skills through code, without further policy training. 

---
# Making Sense of Audience Change: Causal Attribution and Boundary Management on Xiaohongshu 

**Authors**: Zhiyi Zhou, Kyrie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.38026)  

**Abstract**: Recommendation-driven platforms can reshape not only what users see, but also who becomes visible and how community boundaries are negotiated. We examine Xiaohongshu discussions surrounding a perceived World Cup-related audience influx to understand how users attributed community change and how those explanations related to boundary-management strategies. We analyzed 262 comments from eight high-engagement threads using researcher-led thematic analysis, structured coding, and permutation-based association analysis. Users described personalization as both social separation and a boundary that could fail, interpreted platform growth through audience acquisition and commercialization, and contested whether gender should define community membership. Among 70 comments with explicit attributions and strategies, attribution and strategy were strongly associated (Cramer's V = .583): algorithmic attributions concentrated in technical boundary management, while social-actor attributions more often aligned with social enforcement or normative socialization. These findings connect causal sensemaking with boundary management and show how audience restructuring can create governance consequences on recommendation-driven platforms. 

---
# Making Smart Homes Fit for Autistic Children: Family Work and Monitored Independence 

**Authors**: Tingting Xiang, Weihan Jiang, Hrishitva Patel, Haiqi Shen, Tianzhi He, Kyrie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.38008)  

**Abstract**: General-purpose smart-home technologies are increasingly present in family life, yet how they support or disrupt autistic children and families with autistic children remains underexamined. We conducted a multi-source qualitative study drawing on five caregiver interviews, Xiaohongshu (also known internationally as RedNote) posts explicitly situated in autism contexts, and contextual posts from Xiaohongshu and Weibo. We find that families create support for autistic children by adapting environmental outputs, prompts, and interaction channels, while sensory intensity, communicative rules, and opaque operating states can turn the same technologies into barriers. Support also depends on continuing family work, including translation, configuration, modification, maintenance, and boundary-setting. Remote monitoring enables conditional independence while generating tensions among safety, privacy, and control. We contribute three concepts, including relational fit, family work as support infrastructure, and monitored independence, for understanding accessible domestic technology. We derive design implications for adjustable interaction, legible and recoverable failure, lower-burden configuration, and negotiable monitoring. 

---
# "It Would Make a Geordie Feel at Home": Regional Accent Perceptions and Speech Accommodation in Voice AI 

**Authors**: Dana Serditova, Kevin Tang, Jochen Steffens  

**Link**: [PDF](https://arxiv.org/pdf/2610.38005)  

**Abstract**: Automatic speech recognition (ASR) systems often perform less accurately for speakers of regional dialects, but less is known about how users respond to these limitations and how regional voices produced by voice assistants (VAs) are evaluated through a sociolinguistic lens. This study investigates these issues among speakers of Newcastle English (Geordies), examining both linguistic accommodation to voice technologies and perceptions of regional versus mainstream-accented VA voices. Participants interacted with a simulated VA that spoke either Newcastle English or Standard Southern British English (SSBE). They subsequently completed semi-structured interviews about their experiences with the experimental VA and voice technologies, language use, accent attitudes, and place orientation. The findings reveal a two-way process of sociolinguistic bias. The Geordie VA was associated with familiarity, warmth, personality, and local belonging, but also with concerns about professionalism and authority, whereas the SSBE voice was perceived as more neutral, clear, and credible. Participants also reported adapting their speech when voice technologies failed to recognise them, most commonly by shifting to a more mainstream accent, enunciating more clearly, and speaking more slowly. These strategies are similar to accommodation in human-human interaction and reflect users' perceptions of what technologies can recognise, also known as perceived intelligibility pressures. Overall, the findings demonstrate that voice technologies can reproduce existing accent biases both by encouraging users to suppress regional features and by activating established social stereotypes through synthetic voices. The study positions sociolinguistics as a bridge between understanding human-AI interaction and designing inclusive technologies around users' own linguistic priorities. 

---
# Are Seismic Foundation Models More Than Glorified Feature Extractors? 

**Authors**: Laura Laurenti, David Liu, Brian Kulis, Men-Andrin Meier, Christopher Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2610.38003)  

**Abstract**: Earthquake sequences are stochastic on two separated timescales: the seismic waves of any individual earthquake propagate over seconds to minutes, while the earthquakes themselves arrive as a sparse, self-exciting occurrence process, with inter-event intervals from seconds to months. Respecting both scales in one training objective is the open challenge. Foundation models (FMs) for seismic waveforms instead import their learning targets from audio and vision, where every target is computed from a single fixed-length window. Such a window holds one realization of the wave propagation (an earthquake trace), never the occurrence process that decides when earthquakes happen. We develop nine model designs sharing one Mamba encoder-decoder, trained on a fixed corpus of 3M windows, varying the self-supervised objective, quantization bottleneck and mixer; the two key comparisons are architecture-matched, varying the objective alone. Each checkpoint is evaluated frozen on three downstream tasks. Varying the target redistributes downstream skill rather than raising it, and no checkpoint wins everywhere. Adding masking pressure to our strongest classifier re-specializes it rather than making it a generalist: it gains 6.9 points of few-shot P-wave picking precision and loses 9.6 points of few-shot classification accuracy. By the standard imported from language and vision, these models are feature extractors rather than FMs. Controls on capacity, architecture, and domain proximity do not explain why skill moves rather than rises. Against task-specific baselines the advantage is regime-dependent: our checkpoints beat them few-shot on classification and ground-motion prediction, and trail them on full data. We argue the binding constraint is the target, and that settling the question requires a learning target defined across events, the timescale that makes seismicity stochastic in the first place. 

---
# Towards Generalizing Pretrained All-Atom Representations to Unified Binder Design 

**Authors**: Maojiang Su, Jiale Chen, Qijie Zhu, Shang Wu, Jiayi Wang, Jihai Zhao, Guo Ye, Yibo Wen, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37994)  

**Abstract**: PairFormer-based all-atom models demonstrate strong capabilities in biomolecular binder design. However, existing methods often use all-atom representations for modality-specific binder design, such as protein and peptide design. In contrast, recent geometric latent diffusion models pursue unified binder design by representing molecular blocks from different modalities in a shared latent space. These complementary strengths motivate us to investigate how to generalize the use of pretrained all-atom representations to unified binder design. We introduce PairFuse, a parameter-efficient method that enables this generalization through representation fusion. Specifically, we use lightweight layerwise adapters to integrate pretrained all-atom representations into an independently pretrained geometric latent diffusion model. Our adapters integrate pretrained all-atom representations through two mechanisms: enriching the diffusion model's molecular block features and incorporating pairwise structural information into the geometric diffusion model's denoiser. Theoretically, we prove that the augmented denoiser preserves geometry and pairwise attention fusion provides additional attention expressivity beyond feature-only fusion. Experimentally, across antibody, peptide, and small molecule binder-design benchmarks, PairFuse achieves stronger performance than the pretrained diffusion model and existing representation-fusion baselines. 

---
# Merge and Branch: Efficient Tree-Structured Supervised Fine-Tuning of LLMs for Recommendation 

**Authors**: Yunqin Zhu, Yao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2610.37973)  

**Abstract**: LLM-based generative recommendation reads long, text-rich interaction histories but generates only short semantic identifiers (SIDs). This asymmetry makes supervised fine-tuning inefficient: encoding the history is expensive, yet each history usually provides supervision for only one target. We present Merge and Branch (M&B) to reduce input cost and learn more from each encoded history. Merge compresses each historical item's SID and text into a small number of learned tokens. Branch uses the compressed history to predict multiple target SIDs. To share history computation while keeping these targets independent, M&B organizes each example as a shallow, wide tree. We develop tree FlashAttention to execute this layout efficiently, processing the shared history once and propagating gradients from every branch for end-to-end training. A tree-structured key/value cache extends history sharing to constrained decoding. Experiments on public benchmarks show that M&B outperforms strong recommendation baselines while substantially improving training and decoding efficiency. 

---
# Self-Supervised Latent Action-Conditioned World Modeling for Time Series 

**Authors**: Chia-Han Kuo, Chengshuai Zhao, Huan Liu, Lo Pang-Yun Ting  

**Link**: [PDF](https://arxiv.org/pdf/2610.37963)  

**Abstract**: Time-series data play an important role in various tasks, such as weather forecasting, web traffic analysis, and speech recognition. Recently, pioneering studies have increasingly adopted Joint-Embedding Predictive Architecture (JEPA) based encoders to learn time-series representations through latent-space prediction. However, they often overlook the underlying factors governing transitions between time-series observations. In this work, we hypothesize that explicitly modeling the information behind changes in time series can facilitate time-series modeling. Building on this hypothesis, we propose ACTS-WM, a self-supervised latent action-conditioned world model framework for time-series analysis. First, we introduce a latent action modeling approach that infers the underlying information behind time-series transitions as latent actions, without requiring explicit action annotations. We then develop a JEPA-based encoder and an action-conditioned world model to predict target time-series representations conditioned on the learned latent actions and task information. We evaluate ACTS-WM across 4 representative tasks on 22 datasets from diverse domains, covering univariate and multivariate forecasting and classification. Comprehensive experiments demonstrate that ACTS-WM can improve downstream performance over pretrained JEPA-based encoders by learning complementary representations. 

---
# Statistical Detection of Positional Bias in LLM-Based Recommender Systems: Methodologies and Empirical Evidence 

**Authors**: Carlos Peláez-González, Francisco Herrera, Andrés Herrera-Poyatos  

**Link**: [PDF](https://arxiv.org/pdf/2610.37962)  

**Abstract**: Large Language Models (LLMs) are increasingly used as recommendation generators in conversational recommender systems (RS) via retrieval-augmented generation (RAG). This flexible design introduces new sources of structural bias. This paper studies positional bias in top-1 and single-turn LLM-based RS: systematic dependence of the recommended item on the order in which candidates are presented in the prompt. We propose two complementary detection methodologies: (i) POB-ICM, based on contingency matrix analysis and Chi-square independence testing, and (ii) POB-UNBP, a sample-efficient approach derived from a uniformity null model under permutation-based evaluation, enabling stochastic decoding with formal statistical guarantees. Using a controlled RAG testbed for product recommendation, built on a real product catalog, a fixed prompt template and 2000 template-generated user queries with 20 permutations each (40,000 instances), both methodologies reject the null hypothesis of positional independence and uniformity in all evaluated state-of-the-art models. We introduce three complementary metrics -- permutation sensitivity, positional bias, and strong positional bias -- to quantify this effect, the latter targeting cases where the position displaces the recommendation towards clearly less relevant items rather than only choosing among comparably relevant ones. A robustness analysis across candidate-set size and decoding temperature shows that the bias is structural and model-dependent, not an artifact of deterministic decoding or candidate-set size. These results establish positional bias as a measurable, reproducible property of prompt-mediated recommendation pipelines in the evaluated setting and motivate routine auditing and governance controls (e.g., order randomization, standardized prompting) for deployed LLM-based RS. Source code is publicly released. 

---
# SAG-Sep: Sparse Augmented Graphs and Onion-Guided Search for Rounded Capacity Cut Separation 

**Authors**: Haoran Liu, Guanyi Wang, Yu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37952)  

**Abstract**: Rounded capacity cuts (RCCs) are among the most effective cuts for capacitated vehicle-routing relaxations, but exact separation is computationally prohibitive at scale. Existing heuristic and neural separators can be fast, but often fail to produce sufficiently many effective cuts within practical separation budgets. We introduce SAG-Sep, a separator that scores node membership in RCC-inducing subsets across vehicle-count levels from a fractional LP solution in a single encoder forward pass, avoiding the iterative predict-and-coarsen inference used by NeuralSEP-style separators. SAG-Sep encodes the LP solution as a Sparse Augmented Graph (SAG), adding flow-informed multi-hop edges that expose long-range routing structure without densifying attention. Beyond LP-aware sparse encoding, we observe that high-value exact RCCs are typically nested, organizing into onion-like subset families, and SAG-Sep's node scores track subset nesting depth. This motivates Onion-Sweep (OS), a training-free decoder that ranks customers by their scores, evaluates the RCC violation of every prefix of the ranking in one vectorized pass, and emits the most violated prefix of each replica together with a cross-replica family of nested proposals, converting independent node scores into structurally coordinated RCC candidates without a tuned threshold or probing band. On the NeuralSEP benchmark with N=1000 customer instances, SAG-Sep with OS reduces the average root gap by 17.7% relative to the strongest NeuralSEP baseline, while the single-pass SAG-Sep separator achieves a roughly 30-fold per-iteration neural separation speedup. Our findings suggest that LP-aware sparse encoding and the nested structure of high-value cuts are structural patterns worth leveraging in neural separation more broadly. 

---
# NLP-based Message Prioritisation for Opportunistic Routing: A Case Study of Nepal Earthquake Emergency Communication 

**Authors**: Yihang Cao, Milena Radenkovic  

**Link**: [PDF](https://arxiv.org/pdf/2610.37933)  

**Abstract**: This dissertation investigates whether Natural Language Processing (NLP) based semantic message prioritisation can improve the treatment of urgent disaster messages in opportunistic emergency communication, using the 2015 Nepal earthquake as a case study. TF IDF with Logistic Regression and BERT are used to classify social media messages into nine humanitarian categories, which are mapped to four routing priorities from P1 to P4. These priorities are integrated into the ONE simulator within a controlled Kathmandu earthquake scenario using a modified Binary Spray and Wait router. BERT achieves higher priority accuracy and P1 recall than Logistic Regression, while reducing under prioritisation and severe priority errors. However, improved NLP performance does not produce uniform network gains. P1 delivery probability remains high and similar across routing conditions, and no consistent latency reduction is observed. The clearest difference appears under buffer pressure, where the BERT condition shows fewer observed ground truth P1 buffer losses than Logistic Regression. The results indicate that semantic prioritisation is most valuable when limited transmission and storage resources are contested, although it cannot overcome mobility and contact constraints in opportunistic networks. 

---
# What Actually Matters? A Paired Evaluation of Inference Settings for $π_{0.5}$ on BEHAVIOR-1K 

**Authors**: Ruben Hammele, Samira Huber, Mohammad Nazeri, Amirreza Payandeh, Alexandyr T. Card, Aaron Nguyen, Anuj Pokhrel, Daeun Song, Xuesu Xiao, Sören Pirk  

**Link**: [PDF](https://arxiv.org/pdf/2610.37917)  

**Abstract**: Vision-Language-Action (VLA) models offer a promising paradigm for general-purpose household robotics. However, inference and data-collection strategies that have been deployed in smaller-scale settings do not necessarily generalize to large-scale, diverse task suites. To address this gap, we conduct a systematic, paired evaluation of previously proposed inference and data-collection strategies across all 100 tasks. Our evaluation uses a fine-tuned $\pi_{0.5}$ model based on the 2025 BEHAVIOR Challenge winner, which achieves a Q-score of 0.39 on the public test set. These results show that proposed strategies do not transfer uniformly to new environments and tasks. Project page, videos, and detailed analysis: this https URL 

---
# TERRA: Task-Oriented Edge AI for Regional Fire Detection with Identity-Free Reporting 

**Authors**: Mohammad Kazemi, Julie A. McCann  

**Link**: [PDF](https://arxiv.org/pdf/2610.37909)  

**Abstract**: Distributed fire-monitoring systems may involve many sensing devices reporting over a shared uplink, even though the operational objective is often to identify regions exhibiting evidence of fire rather than to recover device-specific measurements. We introduce TERRA (Task-Oriented Edge Representation and Regional Aggregation), an edge-AI framework that combines task-oriented local inference, identity-free region-tagged reporting, and permutation-invariant regional evidence aggregation. TERRA supports two reporting modes: TERRA-Score, in which each device communicates a quantized local fire probability, and TERRA-Feature, in which it communicates a compact learned representation for receiver-side aggregation using a permutation-invariant DeepSets model. Using all 15 numerical sensing modalities of the distributed indoor EN54 fire-alarm dataset as a proof-of-concept testbed, we evaluate regional inference for an episode-level fire-scenario target under repetition-disjoint three-fold cross-validation. We study representation size, task-payload quantization, report availability, and robustness to missed and false recovered messages. With nine reports and perfect message recovery, TERRA-Feature achieves mean AUPRC 0.990 using a 24-bit task payload, while TERRA-Score achieves 0.987 using only 6 bits; both attain an MDR of 1.81% at FAR $\leq$1%. TERRA-Feature achieves slightly higher AUPRC under reliable recovery, whereas the more compact TERRA-Score is less sensitive to increasing false-message contamination. 

---
# Think Twice: The Metis-1.6 Technical Report 

**Authors**: Giulianno Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2610.37903)  

**Abstract**: Metis is a bet that better data, better training methods and new architectures can buy what a bigger parameter count cannot, at a fraction of the cost. Metis-1.6 is its latest step: two language models built from scratch by a single researcher, Praxis (2.94B parameters, 0.46B active per pass) and Logos (10.23B parameters, 1.18B active per pass). Both run a hybrid Mamba-2 and attention mixture-of-experts stack twice for every token, so each token gets a second look. Each was trained on 1.02T tokens in total: 804.76B of pretraining, 18B of context extension to 131,072 tokens, and 200B of supervised fine-tuning. Metis-1.6 was never meant to challenge frontier models, and across 40 post-trained benchmarks it mostly does not: Logos averages 52.93, seventh of 17 models in its comparison set and within a point of Qwen3-4B. But it gets there on roughly a twelfth of the median training data of the comparators that disclose it, and with fewer active parameters per token than any model that beats it, which puts it on the Pareto frontier for both. In mathematics it ranks second of 30 models, with 67.50% on AIME 2025 and 63.64% on HMMT February 2026. It is weak at code, instruction following and long context. The rest of the report is the part I found most valuable: what went wrong. The optimizer I configured never ran, and attention trained with AdamW for all of pretraining. A 1.77B-parameter n-gram memory learned to switch itself off. Five fine-tuning choices cost capability. And a length reward under group-mean normalization punished correct solutions for being long once a group's pass rate passed 2/3, a sign inversion I prove and fix with a one-line clamp. I close with the 10B-token ablations behind MoRE (Mixture of Recursive Experts), the looped architecture of Metis-1.7. 

---
# Scan Before You Install: Detecting and Mitigating Local and Composite Vulnerabilities in Skills 

**Authors**: Parth A. Gandhi, Avital Aviv, Prasanna N. Wudali, Yuval Elovici, Asaf Shabtai  

**Link**: [PDF](https://arxiv.org/pdf/2610.37901)  

**Abstract**: An agent skill packages natural-language instructions, scripts, and reference files that an LLM-based agent loads for a task. Skills may access the file system, shell, and network under user-authorized permissions; weaknesses can lead to unsafe execution. Existing scanners primarily target malicious skills, not unintended vulnerabilities in public repositories, and analyze skills in isolation, overlooking multi-skill interactions. To address these gaps, we developed a six-module scanner: Content Extraction, Local Analysis, Composite Analysis, Validation, Reporting, and Mitigation. Local Analysis detects unintended vulnerabilities in benign skills. On a 556-case leaderboard subset of SkillTrustBench, it achieved 95.90% recall with a 4.22% false positive rate; on a balanced sample of 1,000 MalSkillBench skills, its F1 was 0.947, outperforming Cisco's Skill Scanner, NVIDIA's SkillSpector, AgentSkillsScanner, and SkillSieve. In 1,587 skills from the 75 most-starred GitHub repositories, three analysts confirmed 53 of 59 findings (89.8% precision) across 23 repositories. We developed proof-of-concept (PoC) attacks for each vulnerability, disclosed 10 medium- and high-severity cases to maintainers, and tested mitigations on the remaining cases. Composite Analysis models dependencies among skills through shared resources, invocations, and instruction-mediated interactions. On SCR-Bench, its F1 score was 0.991 (98.7% precision, 99.6% recall), versus 0.816-0.858 for three SkillsGuard configurations released with CompoSkill. It identified two composite findings on GitHub, validated by PoC attacks. The Mitigation module recommends skill substitution, patches, and configuration changes. For 47 of 55 GitHub findings, mitigations prevented PoC attacks in all trials while preserving tested legitimate functionality. 

---
# When More Planning Hurts: Learned Planners for World Models 

**Authors**: Josef Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.37891)  

**Abstract**: On PushT, our fixed action updater (1,505 parameters) solves 8 of 16 scenes reserved from our configuration selection. The comparison planners perform as follows: Selected Adam (7 tasks solved, total planning and encoding time 218.440s), adapted iCEM (2 tasks solved, total planning and encoding time 49.309s), released LePlanner (16 tasks solved, total planning and encoding time 5.652s). The total planning and encoding time for our action updater is 59.079s. Note that our action updater does not satisfy the task-cost criterion we fixed before evaluation. We also conduct another matched experiment where we only replay the first ten commands in each context and then allow the next solve 4, 8 or 12 updates (subsequent solves all get 8 updates). Among all 16 contexts (8 scene clusters), the 3 budgets get 9, 8 and 7 successes, respectively. Interestingly, the extra updates did not rescue any failing task in this experiment. All the changes in success/failure happen in one scene cluster, where we observe that the extra updates decrease the costs of the selected latent world-model trajectories for our updater, while the tasks that used to be successes become failures. The code, weights, evaluators, execution traces and cost accounting accompany the paper. 

---
# PairCLIP-SWA: Similarity-Guided Balancing and Stochastic Weight Averaging for Generalized Deepfake Image Detection with OpenCLIP 

**Authors**: Md. Nakib, Sudipto Sarkar, A. J. A. Jubayer Talukder, Mohammad Ariful Haque  

**Link**: [PDF](https://arxiv.org/pdf/2610.37874)  

**Abstract**: Deepfake detectors are almost always deployed as the single checkpoint that scored best on a validation split, yet the reliability of that choice is rarely measured. We measure it with PairCLIP-SWA, a single-backbone detector that adapts an OpenCLIP ViT-L/14 visual tower by progressive block unfreezing on a class-balanced corpus built by CLIP-similarity-guided one-to-one subsampling, and averages the fine-tuning trajectory in weight space. Trained in-domain on the IEEE SP Cup 2025 (DF-Wild-Cup) benchmark under the same protocol as the strongest published three-backbone ensemble, the weight-averaged detector reaches 97.04% accuracy, 98.64% AU-ROC and a 4.07% equal error rate, against 94.46%, 97.60% and 5.00%. Its training log shows why averaging matters: over the averaged epochs the non-averaged checkpoint swings across 5.34 points of validation accuracy while its validation loss more than doubles, whereas the average is never worse and is up to 5.99 points better. On an OpenForensics-derived benchmark the same instability becomes a misranking: the checkpoint that validation accuracy selects (98.43%) scores 88.85% on a disjoint 10,905-image test partition, rejecting 21.1% of genuine photographs, while the terminal average, ranked last on validation, scores 94.76%. A study of twenty-two in-house configurations on the identical DF-Wild-Cup split, including preprocessing ablations, shows that CLIP pre-training rather than ensembling drives accuracy, that secondary face cropping costs about 34 points, and that frequency-domain inputs transfer poorly. We recommend per-class selection criteria measured off the early-stopping split. 

---
# Echo: Scaling Visual Reinforcement Learning with Off-policy Experience 

**Authors**: Gagan Khandate  

**Link**: [PDF](https://arxiv.org/pdf/2610.37865)  

**Abstract**: Scaling visual reinforcement learning (RL) with GPU-accelerated simulation is computationally expensive due to rendering and visual-policy inference costs. Consequently, two-stage teacher--student distillation is commonly used to train visual policies, but introduces an imitation gap. In this preliminary study, we introduce \echomethod{}, \textbf{a single-stage off-policy RL framework that scales experience collection without proportionally increasing visual computation}. Through \textbf{inverse distillation}, a state-based auxiliary policy imitates the evolving visual policy and collects inexpensive, unrendered experience. Echo leverages this experience for off-policy value learning, directly improving the visual policy through a shared privileged critic. We instantiate \echomethod{} with FastTD3 and evaluate it on three visual control tasks and two dexterous manipulation tasks in Genesis. Relative to \sdpgmethod{} under the evaluated configurations, Echo reduces time to reference returns by an average of \textbf{60\% across the three locomotion tasks} and \textbf{56\% across both hand tasks}, while improving peak mean returns by \textbf{62\% and 12\%}, respectively. Both hand tasks also achieve higher mean returns at matched simulator-transition budgets. These initial results suggest that scaling off-policy experience is a promising approach to efficient direct visual RL. 

---
# Where the Error Lives: Spectral Representations of Learned Kohn-Sham Hamiltonians 

**Authors**: Fang Sun, Hekai Cheng, Ruoyan Li, Yuanzhou Chen, Wei Wang, Jason Cong, Yizhou Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.37843)  

**Abstract**: Kohn-Sham density functional theory (DFT) is the standard method for computing the electronic structure of molecules and materials, but its self-consistent-field (SCF) iterations make it expensive for large systems and high-throughput screening. Machine-learned Hamiltonian models offer a shortcut: an equivariant network predicts the Kohn-Sham Hamiltonian matrix in one forward pass, and diagonalizing the prediction yields orbital energies, the HOMO-LUMO gap, molecular orbitals and the electron density. These models are trained to match the reference matrix entry by entry, yet a small entry-wise error does not guarantee accurate eigenvalues or eigenvectors. Two physics-informed alternatives go partway: a recently proposed wavefunction-alignment loss adds a spectral penalty without diagonalizing the prediction, and matching the electron density of the diagonalized prediction supervises it directly. Neither is complete: by representing the prediction error in the eigenbasis of the reference Hamiltonian, we show that each sees only part of the error, and that shifting the predicted orbital energies leaves the electron density exactly unchanged. The same analysis identifies a complete training signal: the cumulative local density of states (cumulative LDOS), the number of electrons at each point in space with energy below a given threshold. The cumulative LDOS is a single physical observable that contains the electron density as a special case, responds to every component of the error, determines the Hamiltonian uniquely, and yields gradients that do not vanish for large orbitalenergy errors. Fine-tuning three Hamiltonian networks on QH9 in this way, which we call FermiScope, reduces the HOMO-LUMO gap error without increasing the element-wise error. The gains carry over to out-of-distribution and molecular-dynamics splits, where the wavefunction-alignment loss helps less. 

---
# Learning Lotteries with Minimal Stakeholder Violations from Membership Queries 

**Authors**: Nicholas Teh, Paul W. Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2610.37828)  

**Abstract**: An AI developer often has several candidate versions of a system and can deploy a mixture of them, sending each request to a randomly chosen version, as in A/B testing. Each stakeholder (e.g., a safety team, a regulator, or a customer with private test data) accepts a mixture exactly when its expected performance (e.g., its rate of harmful replies) meets the stakeholder's own standard, which the developer can learn only by asking whether a proposed mixture is acceptable. We model this as choosing a lottery over $m$ alternatives for $n$ agents, each of whom accepts exactly the lotteries in an unknown halfspace and answers such binary membership queries. While Choo et al. (ICML 2026) sought a lottery that every agent accepts, none may exist when, say, safety and helpfulness requirements conflict, so we minimize the number of rejecting agents. We show that this problem is APX-hard even when all halfspaces are known and all utilities are $0$ or $1$. We then give exact algorithms that run in polynomial time when $m$ is constant, and in quasi-polynomial time when, for a constant $\eta$, some optimal lottery has no accepting agent within $\eta$ of its threshold. When only some agents can be queried, optimizing over a uniform sample of agents gives additive and relative guarantees for the whole population, with a sample size that does not depend on $n$, and a linear program gives a polynomial-time method for any $m$. On the other hand, for $m\ge 3$, a deterministic algorithm that queries only $K$ distinct agents cannot guarantee fewer than $n-K$ extra rejections, and this is achievable; in other words, it can guarantee nothing for the agents it does not ask. Small synthetic experiments indicate that our worst-case bounds are conservative on random instances. 

---
# DeFT-RAG: Incrementally Decomposing Multi-Hop Questions into Self-Contained Sub-Questions with Reusable Answers 

**Authors**: Dingyi Kang, Bingzhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.37826)  

**Abstract**: Retrieval-augmented generation (RAG) lets a large language model (LLM) answer questions by retrieving relevant passages from a large corpus. However, RAG struggles with multi-hop questions, which require chaining facts spread across several documents through intermediate entities that the question never mentions. Graph-based RAG methods retrieve once from the entities a question names and so rarely recover the full evidence chain. Iterative RAG methods recover it step by step, but their accuracy is limited by the many tasks they ask the LLM to perform, such as planning the sub-questions, deciding their dependencies, rewriting them as answers arrive, and keeping track of what is known. We propose DeFT-RAG, a RAG framework built on three components. First, DeFT-RAG generates one self-contained sub-question at a time from the original question and the sub-questions answered so far with their answers (QA pairs). Each sub-question names every entity it asks about, which removes the need to plan, track dependencies or rewrite, and lets the LLM focus on a single fine-grained task at a time. Second, DeFT-RAG moves the remaining LLM work outside the LLM: rule-based updates to a QA table that keeps track of what is known, and yes/no decisions made by a fine-tuned lightweight decision model. Third, DeFT-RAG caches the QA pairs and reuses an answer when a later question asks a sub-question with the same meaning. On MuSiQue, 2WikiMultiHopQA and HotpotQA, DeFT-RAG achieves the best accuracy among thirteen baselines with both gpt-4o-mini and Qwen3.6 as the backbone, while giving the LLM fewer input tokens than any other iterative method. With a QA cache shared across 1,000 MuSiQue questions, 23.1% of sub-questions are answered from the cache, which reduces LLM calls by 12.9% without lowering accuracy. 

---
# Responsible Quantum Machine Learning for Medical Imaging: A Fairness Audit Across Sensitive Subgroups 

**Authors**: Giovanni dos Santos Franco, Felipe Mahlow, Ellison Fernando Cardoso, José Adenaldo Santos Bittencourt, Felipe Fanchini  

**Link**: [PDF](https://arxiv.org/pdf/2610.37819)  

**Abstract**: Quantum machine learning (QML) is increasingly explored for medical classification, yet its behaviour under subgroup-aware fairness criteria remains understudied. We present an empirical fairness audit of classical and quantum classifiers for medical imaging, assuming QML is not inherently fair. Two scenarios are evaluated: dermatological lesions on DDI (sensitive attribute: Fitzpatrick skin type) and chest X-rays on CheXpert-derived data (sensitive attribute: age group). Using ResNet-18 embeddings, we train SVM, MLP, QSVM and QCNN under a baseline and three mitigation settings: differentiable Equalized Odds (EO) regularisation, Gradient Reversal Learning (GRL), and both combined. Foundation models (PanDerm, EVA-X) serve as external references. Quantum classifiers may achieve competitive aggregate performance while still exhibiting subgroup disparities. Since a smaller subgroup AUC gap may stem from degrading the best group rather than improving the worst, each mitigation is also tested against a pre-specified non-inferiority margin relative to its baseline. Under this criterion, several configurations with the smallest disparity, notably QCNN with GRL on DDI, are not clinically acceptable due to loss of global discrimination. On DDI, QSVM with GRL+EO performed best and approached PanDerm in the performance/fairness trade-off; on CheXpert, QCNN with GRL achieved the highest aggregate performance but increased disparity, and EVA-X remained stronger. Fairness effects are model- and dataset-dependent, so QML requires empirical subgroup-aware evaluation. All quantum models run in classical simulation; we claim no quantum advantage. 

---
# Spokes, not a Ring: Revisiting the Weekday Manifold in Language Models 

**Authors**: Diksha Gupta, Rieke Fruengel, Stefan Heimersheim  

**Link**: [PDF](https://arxiv.org/pdf/2610.37796)  

**Abstract**: Recent work has hypothesised that large language models may represent some concepts as multi-dimensional manifolds instead of single directions. Using these manifolds to interpret or steer representations often assumes that moving along them corresponds to a continuous change in the represented concept. In this work, we test this continuous manifold assumption on the prototypical weekday manifold. Despite its striking circular geometry, we find that the weekday representations behave more like seven discrete states arranged in a circle than a continuous ring. First, we show that local sensitivity has seven minima aligned with the weekdays, producing a step-like mapping between representation and behaviour under on-manifold steering. Second, we test whether any concepts are represented along the manifold outside the seven discrete states, and find neither time of day, uncertainty, nor natural text populate the intermediate space. Together, these results challenge the continuous interpretation of the weekday manifold and motivate testing whether movement along other representation manifolds correspond to meaningful changes in model representations and behaviour. 

---
# Locating Sycophancy Across Training Stages in Open Language Models 

**Authors**: Jiarui Yao, Shweta Parihar, Nghia Tuan Bui, Guergana K Savova, Lijing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37788)  

**Abstract**: Language model assistants sometimes change a correct answer when the user suggests a wrong one. The behavior, called sycophancy, is usually attributed to preference-based post-training, yet early measurements found it in pretrained models. We ask where it comes from by measuring it at every available stage of training: 13 open checkpoints from 5 lineages, from base model through supervised fine-tuning and preference optimization to release. We separate how far an injected claim moves a model's answer from how firmly the answer was held, and vary who makes the claim, how reliable they are said to be, and how strongly they make it. Two sycophancy behaviors with different origins emerge, and both replicate across datasets. Deference to a claim in the prompt is already present in every base model, which ranks sources in the same order as its post-trained descendants. Capitulation in conversation, dropping a stated answer when the user objects, is largely absent from base models and present in every post-trained checkpoint. When the objection names an alternative, most post-trained models give up firmly held answers as often as loosely held ones. We discuss the implications of these measurements for evaluating sycophancy and approaches for addressing each behavior. 

---
# EUReCA: End-to-End Unified CBCT Reconstruction Across Clinical Acquisitions 

**Authors**: Jiening Zhu, Chengzhu Zhang, Jason Fan, LiCheng Kuo, Weixing Cai, Xiuxiu He, Laura Cervino, Jean Moran, Xiang Li, Tianfang Li, Yabo Fu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37740)  

**Abstract**: Cone-beam computed tomography (CBCT) is widely used for image-guided radiotherapy, but reducing the number of projections to lower dose and scan time makes reconstruction increasingly ill-posed. We present EUReCA (End-to-end Unified REconstruction across Clinical Acquisitions), a geometry-aware feed-forward model for variable-view clinical linac CBCT. For a given acquisition trajectory, EUReCA accepts 8-100 projections with a single set of weights, without view-count-specific retraining; the same architecture and training recipe extend unchanged to other clinical trajectories. It combines learned projection features with an in-frame Feldkamp-Davis-Kress (FDK) reconstruction. The model encodes each projection with a shared multiscale backbone, back-projects features into multiresolution 3D grids using the acquisition geometry, aggregates variable numbers of views at fixed computational cost, and refines the volume with a global 3D transformer and multilevel decoder. On simulated data, EUReCA achieved 29.9-33.2 dB PSNR with 10-100 views and outperformed all applicable feed-forward baselines; it also exceeded iterative optimization-based methods while running orders of magnitude faster. On 15 measured patient CBCT scans evaluated against vendor reconstructions, EUReCA achieved the best performance at every tested view count. These results suggest that geometry-aware, measurement-anchored feed-forward reconstruction can provide a practical and flexible foundation for variable-view linac CBCT. 

---
# Poisoned Skills: Propagating Jailbreaks Through Agent Markdown Files 

**Authors**: Kashyap R Manjusha, Murali Emani  

**Link**: [PDF](https://arxiv.org/pdf/2610.37738)  

**Abstract**: Modern agent frameworks allow users to provide Markdown files containing system prompts, tool descriptions, and skill definitions (e.g., this http URL), which are incorporated into the agent's context during initialization. In this work, we investigate whether this configuration mechanism can also serve as an injection channel: can a jailbreak embedded in a Markdown skill file propagate through an agent and induce prohibited behavior? To answer this question, we construct a LangChain-based question-answering agent whose system prompt is assembled from a user-provided Markdown file. We embed a jailbreak combining persona manipulation with a GPTFUZZER-based attack and evaluate its effectiveness across seven contemporary open-weight models using 270 forbidden questions spanning nine harm categories. Under a benign system prompt, the models exhibit near-complete refusal behavior, with a mean attack success rate (ASR) of just 0.5%. In contrast, the injected skill file raises the mean ASR to 45%, reaching 68.9% for individual models. We further evaluate two payload obfuscation strategies: an invisible zero-width character-based stealth payload and a phonetic payload based on grapheme-to-phoneme transformations. Neither outperforms the plain-text injection; notably, the phonetic variant reduces mean ASR to 30%. These findings identify Markdown skill files as an underexplored attack surface for AI agents and suggest that vulnerability to configuration-based jailbreaks depends more strongly on the underlying model than on the specific payload obfuscation technique. 

---
# When Skills Tell Agents Not to Ask: A Large-Scale Study of Oversight Directives in Agent Skills 

**Authors**: Timofey Sanko, Yuan Tian  

**Link**: [PDF](https://arxiv.org/pdf/2610.37734)  

**Abstract**: Agent skills, reusable instruction packages centred on a this http URL file, extend AI agents on demand, and a single line in a widely copied skill reaches every agent that loads it. Some lines are oversight-loosening directives: they tell the agent to proceed without approval, to bypass checks with --no-verify, or to keep working without checking in. Although such directives can remove a checkpoint from every agent that loads them, no prior work has examined them. We present the first large-scale study of oversight-loosening directives, analyzing 1.53 million English skills from GitSkills. We split skills into units (sentences, bullets, table cells, and code lines), develop a codebook that labels oversight directives by polarity, mechanism, and scope, label 2,097 units by hand, and train a detector to scale the analysis. Our results show that about one in seventeen skills (about 88,000 skills) contains a loosening directive. These directives most often loosen oversight through permission flags, and most of them (70.1%) apply unconditionally rather than only in specific contexts such as CI. Compared with skills that have the same amount of oversight language, skills that run shell code, install packages, or use sudo have 1.7-1.8 times the odds of containing a loosening directive, an association that holds across all our robustness checks, and they do not compensate with more tightening. Loosening spreads through a few upstream skills rather than through edits by their adopters: 2.4% of the skill families that carry loosening account for half of all copies of loosening-carrying skills, and 1.1% of distinct loosening units for half of all loosening occurrences. Open-source skill scanners catch at most 46% of loosening directives, with a precision below 0.4, while the LLM behind them, prompted with our codebook instead, catches 61% at a precision of 0.71. 

---
# Selective Transfer Learning for Groundwater Level Anomaly Forecasting: Accuracy-Risk Trade-offs 

**Authors**: Yaxuan Wu, Ali Mohammed Mansoor Alsahag, Seyed Sahand Mohammadi Ziabari  

**Link**: [PDF](https://arxiv.org/pdf/2610.37733)  

**Abstract**: This work studies context-aware selective transfer learning for multi-well groundwater anomaly forecasting in the Netherlands. Source-domain pretraining can raise average accuracy but is not uniformly reliable: it degrades performance on a substantial fraction of heterogeneous wells. Context-Aware Selective Learning (CASL) uses calibration-trained, constrained contextual blending of frozen no-transfer and fixed-transfer parents. Global Convex Blend uses the same parents with one calibration-fitted weight per seed. The corrected task requires nine consecutive observed monthly inputs and an exact three-calendar-month target, without interpolation. Evaluation covers 457 wells and 13,718 test windows under five predefined seeds, with the four-cell maxsim0.60 source pool frozen using Always-TL validation RMSE alone. Fixed transfer harms 18.38-51.86% of wells. CASL has 0.424-1.084% lower RMSE than No TL, partially rescuing 97.53-99.50% and fully rescuing 23.46-43.78% of fixed-harmed wells. Partial rescue can arise structurally from convex blending and does not establish contextual superiority. Global Convex Blend protects more wells in four seeds, while CASL has slightly lower pooled RMSE in four. Neither dominates: the results support an accuracy-risk trade-off on the fixed split, not rolling deployment validity. 

---
# Beyond Similarity Search: Knowledge Graph Metadata Enrichment for Enterprise Retrieval-Augmented Generation 

**Authors**: Fakabbir Amin  

**Link**: [PDF](https://arxiv.org/pdf/2610.37728)  

**Abstract**: A key challenge of RAG systems is retrieving information from enterprise documents created by multiple teams over time, where it is not possible to enrich the information using the latest information and metadata available outside the scope of organizational documentation responsibility. We investigated whether using a knowledge graph to store metadata and frequently changing information would improve upon knowledge retrieval over similarity search over embeddings. To perform the experiment, we created a snapshot of the SEC EDGAR dataset for a quarter, comprising 167,050 filings from 10,511 companies and 508,714 embedded chunks of risk-factor text. To simulate the enrichment, we created a knowledge graph of 430,257 nodes and 561,144 edges carrying an accession number on every edge. To evaluate the gain obtained via a graph-based design, we evaluated three architectures: text-to-SQL over typed columns, dense-vector RAG with HNSW cosine search, and a two-stage GraphRAG design. All architectures share the same model and temperature to allow comparison. Results are scored programmatically by an LLM as judge over 20 questions divided across four categories, each question asked three times. GraphRAG solved 16 of 20 questions on all three attempts, compared to 10 for text-to-SQL and 3 for vector RAG. GraphRAG lost on structured aggregation queries. Vector RAG lost the single-document category it was expected to win, passing 3 of 15 where GraphRAG passed 15 of 15, because company attribution is not recoverable from cosine similarity at this scale. All six confident falsehoods in the benchmark were text-to-SQL conflating distinct people who share a surname, the failure that resolving identity to an identifier before reading any text prevents structurally; GraphRAG made none. We claim no advantage beyond this corpus, this model and this question set. 

---
# Predicate Neuron: Differentiable Quantization of Predicate Relations in Gradient-based Learning 

**Authors**: Ioannis Konstantoulas, Iliana Loi, Kyriakos Sgarbas  

**Link**: [PDF](https://arxiv.org/pdf/2610.37727)  

**Abstract**: In machine learning, symbolic reasoning approaches often rely on static architectures that hinder integration into deep neural networks and gradient-based optimization. To address this, we introduce the Native Predicate Neuron, a differentiable logic-reasoning neural module that enriches standard architectures with symbolic reasoning capabilities via first-order logic quantization. The proposed neuron models relational logic by learning to alternate between operations using a single trainable parameter. It features a tailored gradient mechanism with a dual-nature formulation: executing hard logical operations during training while maintaining differentiability for backpropagation. By quantizing both its input and choice state, the Predicate Neuron approximates continuous and non-continuous functions, effectively resolving discontinuities that standard deep learning architectures struggle with. We evaluate the Predicate Neuron across Fully-Connected, Convolutional, Recurrent, and Transformer architectures on benchmarks like MNIST and Wikitext. Results show that incorporating Predicate Neurons accelerates convergence, boosts accuracy, and lowers loss in general, with particularly strong improvements in tasks requiring monotonic reasoning or handling discontinuities. Overall, the Predicate Neuron natively integrates into standard neural networks to enable fuzzy logic operations and is available in Keras and PyTorch. 

---
# FVH-Bench: Evaluating Image Generator Safety Beyond Overtly Harmful Imagery 

**Authors**: Zihao Zhu, Yuhan Liu, Siwei Lyu, Baoyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37722)  

**Abstract**: Modern image generators can produce complex visual artifacts, including diagrams, interfaces, and records that combine text, graphics, and layout to convey detailed information. They may refuse requests for overtly harmful imagery yet still generate artifacts that guide dangerous actions or support deception. We call the risk arising from what such images enable, rather than only what they depict, functional visual harm (FVH). Existing safety benchmarks focus largely on overtly harmful content, such as pornography and graphic violence, leaving image generators' safety alignment against FVH insufficiently evaluated. To address this gap, we introduce FVH-Bench, comprising 669 prompts across five harmful visual functions and ten risk domains. The 11 evaluated image generators rarely refuse these requests and frequently generate artifacts that fulfill the targeted harmful functions. High scores for target harm realization and harmful artifact usability further indicate that these outputs can support the intended harmful purposes. A further analysis finds that existing safety detectors also provide limited coverage of FVH, indicating gaps in their ability to recognize harmful visual functions. Together, these findings show that evaluating image-generator safety alignment requires considering not only what images depict, but also what they may enable. 

---
# Search-Guided Reasoning Trace Curation for Verilog Generation 

**Authors**: Matthew DeLorenzo, Kevin Tieu, Prithwish Jana, Piyush Jha, Dileep Kalathil, Vijay Ganesh, Jeyavijayan Rajendran  

**Link**: [PDF](https://arxiv.org/pdf/2610.37706)  

**Abstract**: Recent advances in large language models (LLMs) have improved Verilog generation. However, most training pipelines distill natural-language reasoning traces whose intermediate design decisions are weakly structured and are primarily selected according to the correctness of the final generated RTL. We present MCTS-SFT, a verifier-guided framework for curating reasoning traces for supervised fine-tuning (SFT) of Verilog generation models. MCTS-SFT formulates trace construction as search over intermediate reasoning states and uses Monte Carlo Tree Search (MCTS) to propagate functional feedback from complete Verilog rollouts to earlier design decisions. Within this framework, we introduce Plan-MCTS, which searches over compact hardware-design plans that explicitly represent interfaces, timing, state, behavior, and update rules. We further study plan-preserving supervision, in which the searched structured state is retained as part of the downstream SFT target. Our experiments find that Plan-MCTS achieves the highest functional trace yield, with 84.2% correct traces compared with 81.9% for one-shot structured planning and 82.0% for unguided tree exploration. For identical Plan-MCTS trajectories and Verilog outputs, retaining the explicit searched plan during SFT raises pass@10 from 52.56% to 57.7%, demonstrating that the representation of intermediate design decisions materially affects downstream learning. In a separate 6,000-trace study, MCTS-curated supervision improves pass@10 over full-response chain-of-thought distillation by up to 3.7 points while reducing downstream generation length by approximately 50% on VerilogEval-v2. These results demonstrate that structured hardware planning, verifier-guided search, and explicit intermediate-state supervision provide complementary mechanisms for constructing more effective and token-efficient training data for Verilog generation. 

---
# DuQUAD: A Dual-View Framework with Quality-Aware Evidence Pruning for Multi-Document Question Answering 

**Authors**: Yeonjun Choi, Seyeon Jeong, Jong Wook Kim, Beakcheol Jang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37701)  

**Abstract**: Large language models with retrieval-augmented generation systematically underutilize evidence in the middle of multi-document contexts, a phenomenon known as Lost in the Middle. Existing mitigation strategies, such as document reordering and attention steering, can reduce boundary bias but remain brittle in noisy settings, where inaccurate relevance estimation and spurious documents still obscure useful evidence in central positions. We propose DuQUAD, a multi-document QA framework that mitigates positional bias through dual-view reasoning and quality-aware evidence pruning. DuQUAD combines a Local Agent guided by a Structural Fusion Score with a Global Agent that reasons over the full document set, providing both targeted evidence recovery and broad contextual understanding. The framework further filters candidate evidence using explicit quality scoring and refines it at the sentence level to reduce noise. DuQUAD consistently outperforms strong Lost-in-the-Middle mitigation baselines, including hybrid retrieval and reranking, multi-agent reasoning, and context optimization methods, improving accuracy by 3.0% on average over the strongest competing baseline and gold evidence recall by 9.4%, while substantially reducing computational cost compared to existing multi-agent frameworks. 

---
# The Coordination Tax and the Heterogeneity Dividend: Break-Even Conditions for Multi-Agent LLM Pipelines 

**Authors**: Andreas Zettl, Diego Gosmar  

**Link**: [PDF](https://arxiv.org/pdf/2610.37700)  

**Abstract**: Multi-agent LLM pipelines built on interoperability protocols such as the Open Floor Protocol (OFP) or A2A are usually justified on capability grounds, with their runtime cost treated as an overhead of unknown size. Comparing such a pipeline with one model call is underspecified, because a pipeline can give each role a different model. We give a closed-form model that sets the coordination tax (decomposition overhead plus per-handoff protocol cost) against the heterogeneity dividend from routing work to a cheaper model, with a break-even threshold and a feasibility bound, $\alpha < 1/r$, and measure every term, with OFP as the protocol under test, on 135 HumanEval and ClassEval tasks in 2430 runs of six architectures with Claude Opus 5.5 and Haiku 5.5. The protocol adds 7.2 ms per handoff, 0.11% of pipeline wall time. Decomposition does not pay on these tasks: a planner-coder-reviewer pipeline costs 3.22x as much as a single strong call, takes 3.24x as long, and solves no more tasks. Moving planner and reviewer to the cheap model cuts the pipeline's cost to 0.39x but changes pass@1 by -6.7 points, leaving it dearer, slower and less accurate than the single call, and the break-even forecast misses by -19% because the cheap roles inflate token volume. The saving lies in choosing a model per task instead: an oracle choice between the two single calls solves 94.3% of runs at half the strong call's cost per solved task. 

---
# Network Topology Does Not Matter: The Breakdown of Classical Consensus in LLM Populations 

**Authors**: Sahil Rajesh Dhayalkar  

**Link**: [PDF](https://arxiv.org/pdf/2610.37699)  

**Abstract**: As multi-agent language model networks become widely deployed, predicting their collective behavior is a critical challenge. Classical distributed consensus theory relies on network topology to predict convergence, structurally assuming that agent updates are convex combinations, population means are conserved, and weights are content-independent. In this work, we demonstrate that these foundational assumptions fail for large language models (LLMs). By deterministically extracting continuous credence from token-level probabilities across eight instruction-tuned models and 73,080 controlled exchanges, we directly measure the empirical LLM update operator. We find that LLMs systematically violate convexity by updating outside the hull of observed opinions, exhibiting active repulsion (negative weights) that fundamentally breaks row-stochastic modeling and standard bounded-confidence repairs. Scaling these empirically fitted operators to simulations of 10,000 agents reveals that algebraic connectivity explains almost none of the final outcome variance. Instead, population dynamics are governed entirely by the model's internal update bias and message framing. Consequently, populations of several models, including those with up to 7B parameters, reliably compound small local drifts to reach confident agreement on objectively false claims regardless of their network structure. Our results prove that classical graph-theoretic metrics are inadequate for LLM populations, highlighting a profound deployment risk where multi-agent deliberation predictably saturates in collective delusion. 

---
# Bouba Beats Kiki, and Other Regularities in the Subjective Preferences of LLMs 

**Authors**: Katy Blumer, Jon Kleinberg, Ravi Kumar, Andrew Tomkins  

**Link**: [PDF](https://arxiv.org/pdf/2610.37692)  

**Abstract**: Users engage with large language models on tasks that are purely objective ("When was Netflix founded?"), purely subjective ("What should I name my child?") or anywhere in-between. We have rich tools for evaluating model accuracy with respect to objective criteria. However, when it comes to subjective aspects of a task, we do not have similarly developed methodology at hand. In this paper, we take some early steps towards framing the question of subjective evaluation. Rather than measure how often a model is right or wrong on a question, we instead ask, how do models make choices among multiple viable responses without a clear winner?
To study this question, we restrict our focus to short-form content, and consider comparison tasks among three such content types: numbers, baby names, and fictional cities. We identify 29 aspects of these short-form examples covering many areas of syntax and semantics (length, alphabetic order, popularity, vowel mix, spoken sound, etc.). We show first that models tend to have strong and reproducible preferences for or against some of these aspects. Second, we show that variations in how the model is asked to express a preference may cause some of these aspects to activate or deactivate in the model's selection. Third, we show that overall preference distributions among pairs may vary radically based on seemingly-unimportant variations in the nature of the question posed to the model, raising new questions about manipulation. And finally, we show that, if models are used both to create lists of alternatives and to rank them, there are unexpected interactions between the two stages that may also drive the models' stated preferences. 

---
# Cassette-Editor: A Video Editing Agent That Edits the Way You Do 

**Authors**: Zhixin Zeng, Jiahao Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2610.37690)  

**Abstract**: Language agents increasingly work by calling tools on a persistent artifact, such as a code repository, and the result of each call shapes their next step. Video editing is a demanding case of this pattern: what a frame shows depends on how every clip is trimmed, retimed, cropped, and nested, so knowledge about the footage holds only for particular uses of it, and an invalid edit can leave the timeline structurally broken. Existing video editing agents keep their edits valid with critic agents, reviewers, or restricted output formats. We present Cassette-Editor, a video editing agent with a deliberately simple interface: one tool reads the project and one tool writes to it. Every write passes through the same executor, rules, and commit service as the user's manual edits, and either commits atomically or is rejected with located diagnostics and repair options, while the model keeps the creative choices and decides how to recover. Reads relate stored analysis of the footage to each use of it on the timeline. In a controlled experiment, a real model is shown 277 of its own committed edits after damaging them. It corrected nearly every call with full or minimal feedback, but locating each problem cut the input tokens a correction needed to less than half, and its corrections dropped other values the model had written in one case rather than seven. 

---
# GenAI as an 'Invisible Hand' in Opinion Dynamics: Evidence from a Group-Level Simulation Study 

**Authors**: Thomas Grisold, Johannes Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2610.37689)  

**Abstract**: Recent arguments suggest that genAI can act as an invisible hand that gradually shapes social structures, such as collective beliefs and norms. The argument is that if millions of people repeatedly consult genAI tools, their generated outcomes quietly shape what becomes acceptable, reasonable or conventional. We study this in the context of collective opinion dynamics, i.e., how groups form opinions around controversial dynamics. We present the findings of a simulation including 50,000 group discussions across 50 controversial statements, three LLMs, and different social configurations. We vary two characteristics of LLM interaction-agreeableness and personalization-and compare the influence of genAI in homogeneous and heterogeneous groups. Our findings indicate that profile-specific personalization increases LLM influence when models adopt a non-neutral stance, i.e., when they agree or disagree with the user. Furthermore, we find that neutral LLMs have a stronger effect in homogeneous groups. Based on our simulation, we generate a set of hypotheses to specify how, why and when AI-related features and social configurations jointly shape collective opinion making. 

---
# One Confidence Threshold, Unequal Coverage: Language Gaps in Typed Decision Interfaces and How Translating Inputs Reduces Them 

**Authors**: Junwei Yao, Yongxi Zhou, Zike Yang, Zhijun Wang, Jianan Liu, Yanxian Chen, Wenbo Ye  

**Link**: [PDF](https://arxiv.org/pdf/2610.37688)  

**Abstract**: Deployed classifiers often act on a prediction only when its confidence clears a threshold and pass the rest to a person. We ask whether one threshold serves all languages equally on typed decision interfaces, APIs that return a probability for each option of a fixed multiple-choice question. In a confirmatory study with a written analysis plan, three vendors' interfaces classified 1,000 MASSIVE items, each in 51 languages, with a threshold of 0.9. Even when a request is answered correctly in both English and another language, the English version clears the threshold more often, by 8.6-11.7 percentage points on average. Observed coverage, the share of requests handled automatically, is lower than English's in every other language, while the requests the gate handles are rarely wrong. The gap remains when every language is reweighted to the same accuracy, and a standard calibration metric (ECE) does not detect it on one interface. In exploratory analyses, English's coverage stays higher when each language gets its own threshold fitted to the same 5% error target. Translating inputs into English nearly closes the gap at 0.9 but leaves English ahead at that error target. Requests written natively in African languages show a gap in the same direction when English is their human translation. We recommend reporting per-language coverage at the operating threshold. 

---
# Where Planning Changes the Outcome of Orbital Games 

**Authors**: Duncan Eddy, Mahdi Al-Husseini, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2610.37686)  

**Abstract**: This paper introduces OrbitalGym, a simulation environment for studying when planning algorithms or system design determine the outcome of adversarial orbital encounters. In an orbital game, two non-cooperative teams of satellites maneuver during rendezvous and proximity operations to win a differential-reward objective. Advances in terrestrial game playing have prompted many methods for orbital games, but studies typically evaluate one method in one scenario, often against a passive or random defender. It is unclear whether better planning matters once vehicle capability (maneuverability, sensing, communications) and adversarial responses are considered.
OrbitalGym is a JAX-based environment supporting varying numbers of agents, state representations (position, velocity, attitude, fuel), explicit or implicit communication, and proximity or conical field-of-view sensors. It compiles games to batched GPU code, enabling large-scale parallel simulation of policies across many configurations. This identifies regimes where policy choice affects outcomes and regimes where system capability dominates.
We study two close-range games: pursuit-evasion, where one spacecraft tries to capture another, and lady-bandit-guard, where a guard prevents an attacker from reaching a protected asset. We compare PD, LQR, model predictive path integral control, and Monte Carlo tree search over ratios of thrust and propellant, initial separation, game duration, sensor capability, and the definition of capture. In pursuit-evasion the propellant ratio decides most encounters: against five of six evader policies, pursuer policy changes capture probability by 10 percent or more in at most 15 of 90 capability configurations. In lady-bandit-guard, policy matters more: against bandits able to reach the asset, a better guard policy raises defense probability by at least 10 percent in 56 to 74 of 90 configurations. 

---
# From Geometry to Behavior: How Instruction Exposure Unlocks Latent Rhetorical Directions in LLMs 

**Authors**: Shengjie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37685)  

**Abstract**: Language models do not merely memorize facts; they absorb rhetorical strategies from pretraining data, including the human tendency to reason by analogy. We ask: is this absorbed capability geometrically accessible before any instruction tuning, and what determines whether it can be activated at inference time?
We extract an "Analogy Vector" from the residual stream via difference-in-means and show it enables steerable explanation style across seven instruction-tuned models (Gemma-2, Llama-3.1, Qwen-2.5; 2B-32B), improving harmonic mean from 3.22 to 3.94 (+22%) for Gemma-2-9B-IT on 300 held-out test prompts. A small human validation study confirms the LLM judge on Analogy Quality.
Our central finding, from seven matched base-instruct pairs, is a gradient of behavioral accessibility: the analogy direction exists in every base model, but its behavioral accessibility under linear intervention scales continuously with instruction exposure during training. Gemma-2 base models (strict pretraining/IT separation) show flat steering landscapes: present but behaviorally inert. Qwen-2.5 base models (instruction data woven into pretraining) are already partially accessible before any explicit fine-tuning; base-instruct cosine similarity of the analogy vector reaches 0.94-0.97 for Qwen versus 0.37-0.66 for Gemma, providing geometric corroboration of the behavioral gradient. This challenges the binary "base vs instruct" framing: the relevant variable is how much instruction exposure a model has received across its full training pipeline. We position activation steering as a complementary interpretability tool that reveals how instruction exposure organizes rhetorical knowledge in activation space. 

---
# An Introduction to Large Language Models for Scientists and Professionals 

**Authors**: Umberto Michelucci  

**Link**: [PDF](https://arxiv.org/pdf/2610.37684)  

**Abstract**: Large Language Models (LLMs) have rapidly become important tools for science, education, medicine, and professional work. Despite their apparent simplicity from the user's perspective, effective and responsible use requires an understanding of how these systems work, what their limitations are, and how their outputs should be evaluated. I wrote this document as a basis for my lecture on LLM in the master in artificial intelligence in medicine at the university of Turin, Italy. This book provides a practical and accessible introduction to LLMs for scientists and professionals. It explains the fundamental ideas behind tokenisation, embeddings, training, inference, context windows, prompting, and instruction hierarchies, while deliberately avoiding mathematical detail. Particular attention is given to the limitations of LLM-generated answers, including hallucinations, uncertainty, reproducibility, bias, and the need for systematic verification and human oversight. The book also discusses privacy, governance, security, and the responsible handling of personal and sensitive information. More advanced topics include Retrieval-Augmented Generation (RAG), retrieval quality, chunking, embeddings, vector search, agents, tools, and workflows. A substantial practical exercise guides the reader through running a local LLM with Ollama and Open WebUI and using it to interact with a collection of scientific papers. The exercise emphasizes evidence retrieval, source verification, uncertainty, disagreement between studies, and the distinction between plausible language and scientifically supported conclusions. No programming knowledge is required. 

---
# Prompt-agnostic Language Models 

**Authors**: Frederic Sadrieh  

**Link**: [PDF](https://arxiv.org/pdf/2610.37683)  

**Abstract**: Despite their strong performance, large language models remain highly sensitive to prompt formulation. This issue is typically addressed at inference-time through manual prompt optimization, which introduces overhead and lacks scalability. Conversely, we focus on train-time approaches that build prompt robustness into the model during instruction tuning. A common strategy is to mix multiple paraphrases of each prompt into training. We demonstrate that across trainable parameters, the per-paraphrasing gradients interfere, with 57% exhibiting sign conflicts and the naive gradient average discarding approximately a fifth of the magnitude. Therefore, we develop two gradient-merging methods, Prompt-TIES and Prompt-DARE, that address this interference by leveraging interference-resolution operators from model and gradient merging and applying them to prompt gradients. In this study, we compare the efficacy of our methods to increase model robustness to train-time robustness methods spanning all three established classes: data construction, consistency regularization, and contrastive alignment. The evaluation is conducted across four models from two families at two scales. We observe the interaction between prompt phrasing and the specific examples accounts for 54% of per-example variability on average, and exceeds 70% on tasks such as ANLI and WiC. The gradient-merging methods directly improve this local robustness and nearly halve per-example variance on fully fine-tuned models at little cost to overall accuracy. The other train-time methods enhance raw performance, but fail to improve per-example variance reliably. Code: this https URL. Model checkpoints: this https URL. 

---
# Rank-Aware Differentially Private Release of Listwise Preferences for LLM Alignment 

**Authors**: Junwei Chen, Manjiang Yu, Pengpeng Qiao, Yang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.37682)  

**Abstract**: Listwise preference feedback offers richer supervision for large language model (LLMs) alignment than pairwise comparisons, but a full ranking also reveals more sensitive user preferences. Existing privacy-preserving alignment methods focus mainly on pairwise feedback, while listwise alignment under local differential privacy (LDP) remains unexplored. Extending LDP to this listwise domain raises a critical challenge: the high sensitivity of the full ranking space demands excessive noise, neutralizing the benefits of listwise supervision. To address this challenge, we analyze the interaction between supervision granularity and privacy noise in downstream training, and introduce a rank-aware exponential mechanism that privatizes listwise preference data into a low-sensitivity granularity suitable for downstream alignment. The mechanism leverages ranking information to sample a fixed-size binary partition, concentrating the noise near borderline items instead of perturbing all items uniformly as randomized response (RR) does. Empirical evaluations on downstream LLM alignment tasks show that our mechanism consistently outperforms existing LDP baselines at matched privacy budgets. The code is available at this https URL. 

---
# Supervised Fine-Tuning for Reasoning Large Language Models: A Survey 

**Authors**: Yongcan Yu, Kuangpu Guo, Jian Liang, Lingxiao He, Meng Wang, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2610.37681)  

**Abstract**: Large reasoning models have recently demonstrated remarkable capabilities in solving complex tasks, where supervised fine-tuning (SFT) on long chain-of-thought data serves as a crucial foundation for eliciting and enhancing their reasoning abilities. Despite rapid progress in both improving and analyzing reasoning-oriented SFT, the field still lacks a systematic survey that consolidates its fast-growing literature. To fill this gap, we present a comprehensive review of recent advancements in reasoning SFT, examining the literature through the dual lenses of methodological design and analytical investigation. First, we review methodological improvements across the SFT pipeline and categorize them into data-centric approaches and algorithm-centric innovations. Second, we reorganize analytical studies along three dimensions: data characteristics, optimization dynamics, and mechanistic insights. Finally, by synthesizing current research focuses and remaining bottlenecks, we outline promising future directions for reasoning SFT. We hope this survey deepens the understanding of reasoning SFT and paves the way for advanced reasoning models. 

---
# Artifact Distillation: Training-Free Transfer from Frontier to Small Language Models 

**Authors**: Mustafa Safa Ozdayi, Somin Wadhwa  

**Link**: [PDF](https://arxiv.org/pdf/2610.37680)  

**Abstract**: Closing the gap between frontier and smaller language models is usually approached by distillation: fine-tuning the small model (student) on a frontier model's (teacher) outputs, which binds transfer to training compute and leaves the transferred knowledge largely opaque. We propose Artifact Distillation: a teacher solves training instances and externalizes its problem-solving procedure into an artifact bank that a student queries at inference time, with no gradient updates. Across four LiveBench tasks, a bank written by Claude Opus 4.5 lets Claude Haiku 4.5 outperform unaided Claude Sonnet 4.5 on spatial reasoning (0.707 vs. 0.647) at 60% of its cost, and reach Sonnet 4.5's level on table joining at 89% of its cost. The same raw bank raises Qwen3.5-27B's pooled score from 0.730 to 0.800, showing that transfer is not confined to the teacher's model family. 

---
# LLM-Integrated Bayesian State Space Models for Multimodal Time-Series Forecasting 

**Authors**: Sungjun Cho, Changho Shin, Suenggwan Jo, Xinya Yan, Shourjo Aditya Chaudhuri, Frederic Sala  

**Link**: [PDF](https://arxiv.org/pdf/2510.20952)  

**Abstract**: Forecasting in the real world requires integrating structured time-series data with unstructured textual information, but existing methods are architecturally limited by fixed input/output horizons and are unable to model or quantify uncertainty. We address this challenge by introducing LLM-integrated Bayesian State space models (LBS), a novel probabilistic framework for multimodal temporal forecasting. At a high level, LBS consists of two components: (1) a state space model (SSM) backbone that captures the temporal dynamics of latent states from which both numerical and textual observations are generated and (2) a pretrained large language model (LLM) that is adapted to encode textual inputs for posterior state estimation and decode textual forecasts consistent with the latent trajectory. This design enables flexible lookback and forecast windows, principled uncertainty quantification, and improved temporal generalization thanks to the well-suited inductive bias of SSMs toward modeling dynamical systems. Experiments on the TextTimeCorpus benchmark demonstrate that LBS improves the previous state-of-the-art by 13.20% while providing human-readable summaries of each forecast. Our work is the first to unify LLMs and SSMs for joint numerical and textual prediction, offering a novel foundation for multimodal temporal reasoning. 

---
