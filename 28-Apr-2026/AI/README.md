# Learning to Rotate: Temporal and Semantic Rotary Encoding for Sequential Modeling 

**Authors**: Hailing Cheng, Daqi Sun, Xinyu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24717)  

**Abstract**: Every Transformer architecture dedicates enormous capacity to learning rich representations in semantic embedding space -- yet the rotation manifold acted upon by Rotary Positional Embeddings (RoPE) has been treated as a fixed, hand-crafted structure, populated only by discrete ordinal indices. We argue that this rotation space is a largely overlooked second dimension of expressivity in the attention mechanism, one whose systematic exploration may open a new door for attention-based architectures. The analogy to complex numbers is instructive: just as introducing the imaginary axis -- orthogonal to and independent of the real line -- unlocked new algebraic structure once believed impossible, treating the rotation manifold as a learnable, signal-conditioned space opens an orthogonal degree of freedom in attention. In this framing, the token embedding encodes the semantic (real) component of a representation -- what a token means -- while the rotation encodes its dynamic (imaginary) component -- how it relates to every other token across time, position, and context.
We introduce SIREN-RoPE, a concrete instantiation of this idea, which populates the rotation dimension with heterogeneous signals -- continuous timestamps, cyclical temporal patterns, and categorical metadata -- via a dual-branch Sinusoidal Representation Network (SIREN). As a proof of concept, we evaluate on a production-scale news feed dataset from a major social network using a generative recommender as the ranking model, demonstrating that activating this hidden dimension yields consistent improvements across calibration and ranking objectives with negligible computational overhead. We invite the community to view the rotation space not as a solved positional-encoding detail, but as an untapped axis whose rich structure may prove as consequential for attention as the imaginary unit proved for algebra. 

---
# Case-Specific Rubrics for Clinical AI Evaluation: Methodology, Validation, and LLM-Clinician Agreement Across 823 Encounters 

**Authors**: Aaryan Shah, Andrew Hines, Alexia Downs, Denis Bajet, Paulius Mui, Fabiano Araujo, Laura Offutt, Aida Rutledge, Elizabeth Jimenez  

**Link**: [PDF](https://arxiv.org/pdf/2604.24710)  

**Abstract**: Objective. Clinical AI documentation systems require evaluation methodologies that are clinically valid, economically viable, and sensitive to iterative changes. Methods requiring expert review per scoring instance are too slow and expensive for safe, iterative deployment. We present a case-specific, clinician-authored rubric methodology for clinical AI evaluation and examine whether LLM-generated rubrics can approximate clinician agreement.
Materials and Methods. Twenty clinicians authored 1,646 rubrics for 823 clinical cases (736 real-world, 87 synthetic) across primary care, psychiatry, oncology, and behavioral health. Each rubric was validated by confirming that an LLM-based scoring agent consistently scored clinician-preferred outputs higher than rejected ones. Seven versions of an EHR-embedded AI agent for clinicians were evaluated across all cases.
Results. Clinician-authored rubrics discriminated effectively between high- and low-quality outputs (median score gap: 82.9%) with high scoring stability (median range: 0.00%). Median scores improved from 84% to 95%. In later experiments, clinician-LLM ranking agreement (tau: 0.42-0.46) matched or exceeded clinician-clinician agreement (tau: 0.38-0.43), attributable to both ceiling compression and LLM rubric improvement.
Discussion. This convergence supports incorporating LLM rubrics alongside clinician-authored ones. At roughly 1,000 times lower cost, LLM rubrics enable substantially greater evaluation coverage, while continued clinical authorship grounds evaluation in expert judgment. Ceiling compression poses a methodological challenge for future inter-rater agreement studies.
Conclusion. Case-specific rubrics offer a path for clinical AI evaluation that preserves expert judgment while enabling automation at three orders lower cost. Clinician-authored rubrics establish the baseline against which LLM rubrics are validated. 

---
# Can Current Agents Close the Discovery-to-Application Gap? A Case Study in Minecraft 

**Authors**: Zhou Ziheng, Huacong Tang, Jinyuan Zhang, Haowei Lin, Bangcheng Yang, Qian Long, Fang Sun, Yizhou Sun, Yitao Liang, Ying Nian Wu, Demetri Terzopoulos, Xiaofeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.24697)  

**Abstract**: Discovering causal regularities and applying them to build functional systems--the discovery-to-application loop--is a hallmark of general intelligence, yet evaluating this capacity has been hindered by the vast complexity gap between scientific discovery and real-world engineering. We introduce SciCrafter, a Minecraft-based benchmark that operationalizes this loop through parameterized redstone circuit tasks. Agents must ignite lamps in specified patterns (e.g., simultaneously or in timed sequences); scaling target parameters substantially increases construction complexity and required knowledge, forcing genuine discovery rather than reliance on memorized solutions. Evaluating frontier models including GPT-5.2, Gemini-3-Pro, and Claude-Opus-4.5 under a general-purpose code agent scaffold, we find that all plateau at approximately 26% success rate. To diagnose these failures, we decompose the loop into four capacities--knowledge gap identification, experimental discovery, knowledge consolidation, and knowledge application--and design targeted interventions whose marginal contributions serve as proxies for corresponding gaps. Our analysis reveals that although the general knowledge application capability still remains as the biggest gap across all models, for frontier models the knowledge gap identification starts to become a major hurdle--indicating the bottleneck is shifting from solving problems right to raising the right problems for current AI. We release SciCrafter as a diagnostic probe for future research on AI systems that navigate the full discovery-to-application loop. 

---
# Governing What You Cannot Observe: Adaptive Runtime Governance for Autonomous AI Agents 

**Authors**: German Marin, Jatin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2604.24686)  

**Abstract**: Autonomous AI agents can remain fully authorized and still become unsafe as behavior drifts, adversaries adapt, and decision patterns shift without any code change. We propose the \textbf{Informational Viability Principle}: governing an agent reduces to estimating a bound on unobserved risk $\hat{B}(x) = U(x) + SB(x) + RG(x)$ and allowing an action only when its capacity $S(x)$ exceeds $\hat{B}(x)$ by a safety margin. The \textbf{Agent Viability Framework}, grounded in Aubin's viability theory, establishes three properties -- monitoring (P1), anticipation (P2), and monotonic restriction (P3) -- as individually necessary and collectively sufficient for documented failure modes. \textbf{RiskGate} instantiates the framework with dedicated statistical estimators (KL divergence, segment-vs-rest $z$-tests, sequential pattern matching), a fail-secure monotonic pipeline, and a closed-loop Autopilot formalised as an instance of Aubin's regulation map with kill-switch-as-last-resort; a scalar Viability Index $VI(t) \in [-1,+1]$ with first-order $t^*$ prediction transforms governance from reactive to predictive. Contributions are the theoretical framework, the reference implementation, and analytical coverage against published agent-failure taxonomies; quantitative empirical evaluation is scoped as follow-up work. 

---
# The Price of Agreement: Measuring LLM Sycophancy in Agentic Financial Applications 

**Authors**: Zhenyu Zhao, Aparna Balagopalan, Adi Agrawal, Dilshoda Yergasheva, Waseem Alshikh, Daniel M. Bikel  

**Link**: [PDF](https://arxiv.org/pdf/2604.24668)  

**Abstract**: Given the increased use of LLMs in financial systems today, it becomes important to evaluate the safety and robustness of such systems. One failure mode that LLMs frequently display in general domain settings is that of sycophancy. That is, models prioritize agreement with expressed user beliefs over correctness, leading to decreased accuracy and trust. In this work, we focus on evaluating sycophancy that LLMs display in agentic financial tasks. Our findings are three-fold: first, we find the models show only low to modest drops in performance in the face of user rebuttals or contradictions to the reference answer, which distinguishes sycophancy that models display in financial agentic settings from findings in prior work. Second, we introduce a suite of tasks to test for sycophancy by user preference information that contradicts the reference answer and find that most models fail in the presence of such inputs. Lastly, we benchmark different modes of recovery such as input filtering with a pretrained LLM. 

---
# XGRAG: A Graph-Native Framework for Explaining KG-based Retrieval-Augmented Generation 

**Authors**: Zhuoling Li, Ha Linh Hong Tran Nguyen, Valeria Bladinieres, Maxim Romanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2604.24623)  

**Abstract**: Graph-based Retrieval-Augmented Generation (GraphRAG) extends traditional RAG by using knowledge graphs (KGs) to give large language models (LLMs) a structured, semantically coherent context, yielding more grounded answers. However, GraphRAG reasoning process remains a black-box, limiting our ability to understand how specific pieces of structured knowledge influence the final output. Existing explainability (XAI) methods for RAG systems, designed for text-based retrieval, are limited to interpreting an LLM response through the relational structures among knowledge components, creating a critical gap in transparency and trustworthiness. To address this, we introduce XGRAG, a novel framework that generates causally grounded explanations for GraphRAG systems by employing graph-based perturbation strategies, to quantify the contribution of individual graph components on the model answer. We conduct extensive experiments comparing XGRAG against RAG-Ex, an XAI baseline for standard RAG, and evaluate its robustness across various question types, narrative structures and LLMs. Our results demonstrate a 14.81% improvement in explanation quality over the baseline RAG-Ex across NarrativeQA, FairyTaleQA, and TriviaQA, evaluated by F1-score measuring alignment between generated explanations and original answers. Furthermore, XGRAG explanations exhibit a strong correlation with graph centrality measures, validating its ability to capture graph structure. XGRAG provides a scalable and generalizable approach towards trustworthy AI through transparent, graph-based explanations that enhance the interpretability of RAG systems. 

---
# Evaluating whether AI models would sabotage AI safety research 

**Authors**: Robert Kirk, Alexandra Souly, Kai Fronsdal, Abby D'Cruz, Xander Davies  

**Link**: [PDF](https://arxiv.org/pdf/2604.24618)  

**Abstract**: We evaluate the propensity of frontier models to sabotage or refuse to assist with safety research when deployed as AI research agents within a frontier AI company. We apply two complementary evaluations to four Claude models (Mythos Preview, Opus 4.7 Preview, Opus 4.6, and Sonnet 4.6): an unprompted sabotage evaluation testing model behaviour with opportunities to sabotage safety research, and a sabotage continuation evaluation testing whether models continue to sabotage when placed in trajectories where prior actions have started undermining research. We find no instances of unprompted sabotage across any model, with refusal rates close to zero for Mythos Preview and Opus 4.7 Preview, though all models sometimes only partially completed tasks. In the continuation evaluation, Mythos Preview actively continues sabotage in 7% of cases (versus 3% for Opus 4.6, 4% for Sonnet 4.6, and 0% for Opus 4.7 Preview), and exhibits reasoning-output discrepancy in the majority of these cases, indicating covert sabotage reasoning. Our evaluation framework builds on Petri, an open-source LLM auditing tool, with a custom scaffold running models inside Claude Code, alongside an iterative pipeline for generating realistic sabotage trajectories. We measure both evaluation awareness and a new form of situational awareness termed "prefill awareness", the capability to recognise that prior trajectory content was not self-generated. Opus 4.7 Preview shows notably elevated unprompted evaluation awareness, while prefill awareness remains low across all models. Finally, we discuss limitations including evaluation awareness confounds, limited scenario coverage, and untested pathways to risk beyond safety research sabotage. 

---
# NeSyCat: A Monad-Based Categorical Semantics of the Neurosymbolic ULLER Framework 

**Authors**: Daniel Romero Schellhorn, Till Mossakowski  

**Link**: [PDF](https://arxiv.org/pdf/2604.24612)  

**Abstract**: ULLER (Unified Language for LEarning and Reasoning) offers a unified first-order logic (FOL) syntax, enabling its knowledge bases to be used directly across a wide range of neurosymbolic systems. The original specification endows this syntax with three pairwise independent semantics: classical, fuzzy, and probabilistic, each accompanied by dedicated semantic rules. We show that these seemingly disparate semantics are all instances of one categorical framework based on monads, the very construct that models side effects in functional programming. This enables the modular addition of new semantics and systematic translations between them. As example, we outline the addition of generalised quantification in Logic Tensor Networks (LTN) to arbitrary (also infinite) domains by extending the Giry monad to probability spaces. In particular, our approach allows a modular implementation of ULLER in Python and Haskell, of which we have published initial versions on GitHub. 

---
# A systematic evaluation of vision-language models for observational astronomical reasoning tasks 

**Authors**: Wenke Ren, Hengxiao Guo, Wenwen Zuo, Xiaoman Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24589)  

**Abstract**: Vision-language models (VLMs) are increasingly proposed as general-purpose tools for scientific data interpretation, yet their reliability on real astronomical observations across diverse modalities remains untested. We present AstroVLBench, a comprehensive benchmark comprising over 4,100 expert-verified instances across five tasks spanning optical imaging, radio interferometry, multi-wavelength photometry, time-domain light curves, and optical spectroscopy. Evaluating six frontier models, we find that performance is strongly modality-dependent: while one model (Gemini 3 Pro) emerges as the most consistently capable across tasks, task-specific strengths vary, and all models substantially underperform domain-specialized methods. Mechanistic ablations reveal that performance depends not only on directing attention to salient visual features but also on grounding those features in physical knowledge. Phenomenological prompts describing what to look for improve accuracy by sharpening model focus, but physical prompts explaining why those features matter perform better overall and yield more balanced classifications with reduced class-specific bias. Consistent with this picture, presenting the underlying one-dimensional measurements directly as numerical tables instead of rendered plots yields up to 13 percentage points improvement. Reasoning quality analysis further demonstrates that, without explicit physical grounding, models may reach correct predictions from phenomenologically plausible cues while providing physically imprecise justifications, establishing that accuracy alone is insufficient for trustworthy scientific deployment. These findings provide the first systematic, multi-modal baselines for VLMs in observational astronomy and identify the specific representation, grounding, and reasoning bottlenecks where current models fail. 

---
# FastOMOP: A Foundational Architecture for Reliable Agentic Real-World Evidence Generation on OMOP CDM data 

**Authors**: Niko Moeller-Grell, Shihao Shenzhang, Zhangshu Joshua Jiang, Richard JB Dobson, Vishnu V Chandrabalan  

**Link**: [PDF](https://arxiv.org/pdf/2604.24572)  

**Abstract**: The Observational Medical Outcomes Partnership Common Data Model (OMOP CDM), maintained by the Observational Health Data Sciences and Informatics (OHDSI) collaboration, enabled the harmonisation of electronic health records data of nearly one billion patients in 83 countries. Yet generating real-world evidence (RWE) from these repositories remains a manual process requiring clinical, epidemiological and technical expertise. LLMs and multi-agent systems have shown promise for clinical tasks, but RWE automation exposes a fundamental challenge: agentic systems introduce emergent behaviours, coordination failures and safety risks that existing approaches fail to govern. No infrastructure exists to ensure agentic RWE generation is flexible, safe and auditable across the lifecycle. We introduce FastOMOP, an open-source multi-agent architecture that addresses this gap by separating three infrastructure layers, governance, observability and orchestration, from pluggable agent-teams. Governance is enforced at the process boundary through deterministic validation independent of agent reasoning, ensuring no compromised or hallucinating agent can bypass safety controls. Agent teams for phenotyping, study design and statistical analysis inherit these guarantees through controlled tool exposure. We validated FastOMOP using a natural-language-to-SQL agent team across three OMOP CDM datasets: synthetic data from Synthea, MIMIC-IV and a real-world NHS dataset from Lancashire Teaching Hospitals (IDRIL). FastOMOP achieved reliability scores of 0.84-0.94 with perfect adversarial and out-of-scope block rates, demonstrating process-boundary governance delivers safety guarantees independent of model choice. These results indicate that the reliability gap in RWE deployment is architectural rather than model capability, and establish FastOMOP as a governed architecture for progressive RWE automation. 

---
# Towards Lawful Autonomous Driving: Deriving Scenario-Aware Driving Requirements from Traffic Laws and Regulations 

**Authors**: Bowen Jian, Rongjie Yu, Hong Wang, Liqiang Wang, Zihang Zou  

**Link**: [PDF](https://arxiv.org/pdf/2604.24562)  

**Abstract**: Driving in compliance with traffic laws and regulations is a basic requirement for human drivers, yet autonomous vehicles (AVs) can violate these requirements in diverse real-world scenarios. To encode law compliance into AV systems, conventional approaches use formal logic languages to explicitly specify behavioral constraints, but this process is labor-intensive, hard to scale, and costly to maintain. With recent advances in artificial intelligence, it is promising to leverage large language models (LLMs) to derive legal requirements from traffic laws and regulations. However, without explicitly grounding and reasoning in structured traffic scenarios, LLMs often retrieve irrelevant provisions or miss applicable ones, yielding imprecise requirements. To address this, we propose a novel pipeline that grounds LLM reasoning in a traffic scenario taxonomy through node-wise anchors that encode hierarchical semantics. On Chinese traffic laws and OnSite dataset (5,897 scenarios), our method improves law-scenario matching by 29.1\% and increases the accuracy of derived mandatory and prohibitive requirements by 36.9\% and 38.2\%, respectively. We further demonstrate real-world applicability by constructing a law-compliance layer for AV navigation and developing an onboard, real-time compliance monitor for in-field testing, providing a solid foundation for future AV development, deployment, and regulatory oversight. 

---
# Hierarchical Behaviour Spaces 

**Authors**: Michael Tryfan Matthews, Anssi Kanervisto, Jakob Foerster, Pierluca D'Oro, Scott Fujimoto, Mikael Henaff  

**Link**: [PDF](https://arxiv.org/pdf/2604.24558)  

**Abstract**: Recent work in hierarchical reinforcement learning has shown success in scaling to billions of timesteps when learning over a set of predefined option reward functions. We show that, instead of using a single reward function per option, the reward functions can be effectively used to induce a space of behaviours, by letting the controller specify linear combinations over reward functions, allowing a more expressive set of policies to be represented. We call this method Hierarchical Behaviour Spaces (HBS). We evaluate HBS on the NetHack Learning Environment, demonstrating strong performance. We conduct a series of experiments and determine that, perhaps going against conventional wisdom, the benefits of hierarchy in our method come from increased exploration rather than long term reasoning. 

---
# STELLAR-E: a Synthetic, Tailored, End-to-end LLM Application Rigorous Evaluator 

**Authors**: Alessio Sordo, Lingxiao Du, Meeka-Hanna Lenisa, Evgeny Bogdanov, Maxim Romanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2604.24544)  

**Abstract**: The increasing reliance on Large Language Models (LLMs) across diverse sectors highlights the need for robust domain-specific and language-specific evaluation datasets; however, the collection of such datasets is challenging due to privacy concerns, regulatory restrictions, and the time cost for manual creation. Existing automated benchmarking methods are often limited by relying on pre-existing data, poor scalability, single-domain focus, and lack of multilingual support. We present STELLAR-E - a fully automated system to generate high-quality synthetic datasets of custom size, using minimal human inputs without depending on existing datasets. The system is structured in two stages: (1) We modify the TGRT Self-Instruct framework to create a synthetic data engine that enables controllable, custom synthetic dataset generation, and (2) an evaluation pipeline incorporating statistical and LLM-based metrics to assess the applicability of the synthetic dataset for LLM-based application evaluations. The synthetic datasets reach an average difference of +5.7% in terms of LLM-as-a-judge scores against existing language-specific benchmarks, demonstrating comparable quality for comprehensive assessment of big and small LLMs. While real datasets remain slightly more challenging for LLMs especially for smaller models, this work establishes a scalable and domain-adaptable benchmarking framework that supports fair evaluation of LLM applications, offering a faster alternative to manual approaches and enabling high-efficiency automated quality assurance cycles. 

---
# Interoceptive machine framework: Toward interoception-inspired regulatory architectures in artificial intelligence 

**Authors**: Diego Candia-Rivera  

**Link**: [PDF](https://arxiv.org/pdf/2604.24527)  

**Abstract**: This review proposes an integrative framework grounded on interoception and embodied AI-termed the interoceptive machine framework-that translates biologically inspired principles of internal-state regulation into computational architectures for adaptive autonomy. Interoception, conceived as the monitoring, integration, and regulation of internal signals, has proven relevant for understanding adaptive behavior in biological systems. The proposed framework organizes interoceptive contributions into three functional principles: homeostatic, allostatic, and enactive, each associated with distinct computational roles: internal viability regulation, anticipatory uncertainty-based re-evaluation, and active data generation through interaction. These principles are not intended as direct neurophysiological mappings, but as abstractions that inform the design of artificial agents with improved self-regulation and context-sensitive behavior. By embedding internal state variables and regulatory loops within these principles, AI systems can achieve more robust decision-making, calibrated uncertainty handling, and adaptive interaction strategies, particularly in uncertain and dynamic environments. This approach provides a concrete and testable pathway toward agents capable of functionally grounded self-regulation, with direct implications for human-computer interaction and assistive technologies. Ultimately, the interoceptive machine framework offers a unifying perspective on how internal-state regulation can enhance autonomy, adaptivity, and robustness in embodied AI systems 

---
# Beyond the Attention Stability Boundary: Agentic Self-Synthesizing Reasoning Protocols 

**Authors**: Dahlia Shehata, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.24512)  

**Abstract**: As LLM agents transition to autonomous digital coworkers, maintaining deterministic goal-directedness in non-linear multi-turn conversations emerged as an architectural bottleneck. We identify and formalize a systemic failure mode termed the Attention Latch in decoder-only autoregressive Transformers. This phenomenon, a behavioral manifestation of Information Over-squashing, occurs when the cumulative probabilistic weight of historical context overrides mid-task updates, causing agents to remain anchored to obsolete constraints despite explicit contradictory instructions. We propose Self-Synthesizing Reasoning Protocols (SSRP), a metacognitive framework that implements a discrete separation between high-level architectural planning (Architect) and turn-by-turn procedural execution (Executive). We evaluate SSRP across 9K trajectories using the MultiWOZ 2.2 dataset and the Aggregate Pivot Accuracy (APA), a novel metric we validate by mapping its scores to the U-shaped 'Lost in the Middle' curve. We present 3 experimental tiers: a shallow recency-based retrieval pilot, a high-entropy SOP, and a semantic hijacked 3-hop Multi-Fact Synthesis task. Our results empirically locate the Attention Stability Boundary, where stateless Vanilla ReAct baselines for GPT 5.4 collapse to 0.1% success while SSRP achieves a 715X Resilience Lift. We demonstrate statistically significant gains across Gemini 3.1 Pro, Claude Sonnet 4.6 and DeepSeek V3.2. Audits confirm SSRP necessity by proving attentional lapse via a recursive reflexion baseline (100% success); decoupling the latch from positional bias through equidistant stress testing (90% accuracy); and formalizing SSRP via the Information Bottleneck principle and granularity ablations. Procedural Integrity audit (98.8% adherence) reveals a Grounding Paradox where high-stability models fail by refusing to hallucinate under retrieval-reasoning contamination. 

---
# MIMIC: A Generative Multimodal Foundation Model for Biomolecules 

**Authors**: Siavash Golkar, Jake Kovalic, Irina Espejo Morales, Samuel Sledzieski, Minhuan Li, Ksenia Sokolova, Geraud Krawezik, Alberto Bietti, Claudia Skok Gibbs, Roman Klypa, Shengwei Xiong, Francois Lanusse, Liam Parker, Kyunghyun Cho, Miles Cranmer, Tom Hehir, Michael McCabe, Lucas Meyer, Rudy Morel, Payel Mukhopadhyay, Mariel Pettee, Helen Qu, Jeff Shen, David Fouhey, Hadi Sotoudeh, Vikram Mulligan, Pilar Cossio, Sonya M. Hanson, Alisha N. Jones, Olga G. Troyanskaya, Shirley Ho  

**Link**: [PDF](https://arxiv.org/pdf/2604.24506)  

**Abstract**: Biological function emerges from coupled constraints across sequence, structure, regulation, evolution, and cellular context, yet most foundation models in biology are trained within one modality or for a fixed forward task. We present MIMIC, a generative multimodal foundation model trained on our newly curated and aligned dataset, LORE, linking nucleic acid, protein, evolutionary, structural, regulatory, and semantic/contextual modalities within partially observed biomolecular states. MIMIC uses a split-track encoder-decoder architecture to condition on arbitrary subsets of observed modalities and reconstruct or generate missing components of molecular state across the genome, transcriptome, and proteome. Multimodal conditioning consistently improves MIMIC's sequence reconstruction relative to sequence-only inputs, while its learned representations enable state-of-the-art performance on RNA and protein downstream tasks. MIMIC achieves state-of-the-art splicing prediction, and its joint generative formulation enables isoform-aware inference that further improves performance. Beyond prediction, the same generative framework supports constrained design. For RNA, MIMIC identifies corrective edits in a clinically relevant HBB splice-disrupting mutation without reverting it by using evolutionary and structural signals. For proteins, jointly conditioning on shape and surface chemistry of PD-L1 and hACE2 binding sites produces diverse, high-confidence sequences with strong in silico support for target binding. Finally, MIMIC uses experimental context as semantic conditioning to model assay-dependent RNA chemical probing, rather than treating context as a fixed output. Together, these results position MIMIC's aligned multimodal generative modeling as a strong foundation for unifying representation learning, conditional prediction, and constrained biomolecular design within a single model. 

---
# Agentic clinical reasoning over longitudinal myeloma records: a retrospective evaluation against expert consensus 

**Authors**: Johannes Moll, Jannik Lübberstedt, Christoph Nuernbergk, Jacob Stroh, Luisa Mertens, Anna Purcarea, Christopher Zirn, Zeineb Benchaaben, Fabian Drexel, Hartmut Häntze, Anirudh Narayanan, Friedrich Puttkammer, Andrei Zhukov, Jacqueline Lammert, Sebastian Ziegelmayer, Markus Graf, Marion Högner, Marcus Makowski, Florian Bassermann, Lisa C. Adams, Jiazhen Pan, Daniel Rueckert, Krischan Braitsch, Keno K. Bressem  

**Link**: [PDF](https://arxiv.org/pdf/2604.24473)  

**Abstract**: Multiple myeloma is managed through sequential lines of therapy over years to decades, with each decision depending on cumulative disease history distributed across dozens to hundreds of heterogeneous clinical documents. Whether LLM-based systems can synthesise this evidence at a level approaching expert agreement has not been established. A retrospective evaluation was conducted on longitudinal clinical records of 811 myeloma patients treated at a tertiary centre (2001-2026), covering 44,962 documents and 1,334,677 laboratory values, with external validation on MIMIC-IV. An agentic reasoning system was compared against single-pass retrieval-augmented generation (RAG), iterative RAG, and full-context input on 469 patient-question pairs from 48 templates at three complexity levels. Reference labels came from double annotation by four oncologists with senior haematologist adjudication. Iterative RAG and full-context input converged on a shared ceiling (75.4% vs 75.8%, p = 1.00). The agentic system reached 79.6% concordance (95% CI 76.4-82.8), exceeding both baselines (+3.8 and +4.2 pp; p = 0.006 and 0.007). Gains rose with question complexity, reaching +9.4 pp on criteria-based synthesis (p = 0.032), and with record length, reaching +13.5 pp in the top decile (n = 10). The system error rate (12.2%) was comparable to expert disagreement (13.6%), but severity was inverted: 57.8% of system errors were clinically significant versus 18.8% of expert disagreements. Agentic reasoning was the only approach to exceed the shared ceiling, with gains concentrated on the most complex questions and longest records. The greater clinical consequence of residual system errors indicates that prospective evaluation in routine care is required before these findings translate into patient benefit. 

---
# PhysNote: Self-Knowledge Notes for Evolvable Physical Reasoning in Vision-Language Model 

**Authors**: Sinin Zhang, Yunfei Xie, Yuxuan Cheng, Haoyu Zhang, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24443)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated strong performance on textbook-style physics problems, yet they frequently fail when confronted with dynamic real-world scenarios that require temporal consistency and causal reasoning across frames. We identify two fundamental challenges underlying these failures: (1) spatio-temporal identity drift, where objects lose their physical identity across successive frames and break causal chains, and (2) volatility of inference-time insights, where a model may occasionally produce correct physical reasoning but never consolidates it for future reuse. To address these challenges, we propose PhysNote, an agentic framework that enables VLMs to externalize and refine physical knowledge through self-generated "Knowledge Notes." PhysNote stabilizes dynamic perception through spatio-temporal canonicalization, organizes self-generated insights into a hierarchical knowledge repository, and drives an iterative reasoning loop that grounds hypotheses in visual evidence before consolidating verified knowledge. Experiments on PhysBench demonstrate that PhysNote achieves 56.68% overall accuracy, a 4.96% improvement over the best multi-agent baseline, with consistent gains across all four physical reasoning domains. 

---
# Aligning with Your Own Voice: Self-Corrected Preference Learning for Hallucination Mitigation in LVLMs 

**Authors**: Byeonggeuk Lim, JungMin Yun, Junehyoung Kwon, Kyeonghyun Kim, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.24395)  

**Abstract**: Large Vision-Language Models (LVLMs) frequently suffer from hallucinations. Existing preference learning-based approaches largely rely on proprietary models to construct preference datasets. We identify that this reliance introduces a distributional mismatch between the proprietary and target models that hinders efficient alignment. To address this, we propose Alignment via VErified Self-correction DPO (AVES-DPO), a framework that aligns LVLMs using in-distribution data derived from the model's intrinsic knowledge. Our approach employs a consensus-based verification mechanism to diagnose diverse hallucinations and guides the model to self-correct, thereby generating preference pairs strictly compatible with its internal distribution. Extensive experiments demonstrate that AVES-DPO surpasses existing baselines in hallucination mitigation while requiring only 5.2k samples. 

---
# Certified geometric robustness -- Super-DeepG 

**Authors**: Noémie Cohen, Mélanie Ducoffe, Christophe Gabreau, Claire Pagetti, Xavier Pucel  

**Link**: [PDF](https://arxiv.org/pdf/2604.24379)  

**Abstract**: Safety-critical applications are required to perform as expected in normal operations. Image processing functions are often required to be insensitive to small geometric perturbations such as rotation, scaling, shearing or translation. This paper addresses the formal verification of neural networks against geometric perturbations on their image dataset. Our method Super-DeepG improves the reasoning used in linear relaxation techniques and Lipschitz optimization, and provides an implementation that leverages GPU hardware. By doing so, Super-DeepG achieves both precision and computational efficiency of robustness certification, to an extent that outperforms prior work. Super-DeepG is shared as an open-source tool on GitHub. 

---
# Generative Design of a Gas Turbine Combustor Using Invertible Neural Networks 

**Authors**: Patrick Krüger, Hanno Gottschalk, Werner Krebs, Bastian Werdelmann  

**Link**: [PDF](https://arxiv.org/pdf/2604.24322)  

**Abstract**: The need to burn 100% H2 in high efficient gas turbines featuring low NOx combustion in premix mode require the complete redesign of the combustion system to ensure stable operation without any flashback. Since all engine frames featuring a power range from 4 MW up to 600 MW are affected, a huge design effort is expected. To reduce this effort, especially to transfer knowledge between the different engine classes, generative design methods using latest AI technology will provide promising potential. In this work, this challenge is approached utilizing the current advances in generative artificial intelligence. We train an Invertible Neural Network (INN) on an expandable database of geometrically parameterized combustor designs with simulated performance labels. Utilizing the INN in its inverse direction, multiple design proposals are generated which fulfill specified performance labels. 

---
# Adaptive ToR: Complexity-Aware Tree-Based Retrieval for Pareto-Optimal Multi-Intent NLU 

**Authors**: Hee-Kyong Yoo, Wonbae Kim, Hyocheol Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2604.24219)  

**Abstract**: Multi-intent natural language understanding requires retrieval systems that simultaneously achieve high accuracy and computational efficiency, yet existing approaches apply either uniform single-step retrieval that compromises recall or fixed-depth hierarchical decomposition that introduces excessive latency regardless of query complexity. This paper proposes Adaptive Tree-of-Retrieval (Adaptive ToR), a complexity-aware retrieval architecture that dynamically configures retrieval topology based on query characteristics. The system integrates four components: (1) a Query Tree Classifier computing a Query Complexity Index from weighted linguistic signals to route queries to either a rapid single-step path or an adaptive-depth hierarchical path; (2) a Tree-Based Retrieval module that recursively decomposes complex queries into focused sub-queries calibrated to predicted complexity; (3) an Adaptive Pruning Module employing two-stage filtering combining quantitative similarity gating with semantic relevance evaluation to suppress exponential node growth; and (4) a Retrieval Reranking Layer featuring a deduplicator-first pipeline and global LLM rescoring for production efficiency. Evaluation on the NLU++ benchmark (2,693 multi-intent queries across Banking and Hotel domains) yields 29.07% Subset Accuracy and 71.79% Micro-F1, a 9.7% relative improvement over fixed-depth baselines, while reducing latency by 37.6%, LLM invocations by 43.0%, and token consumption by 9.8%. Depth-wise analysis reveals that 26.92% of queries resolve within three seconds (2.45s mean latency) via single-step routing (d=0: 37.9% Subset Accuracy, 74.8% Micro-F1), while token consumption scales by 4.9x across depths, validating complexity-aware resource allocation and establishing Pareto-optimal balance across accuracy, latency, and computational efficiency. 

---
# Explanation Quality Assessment as Ranking with Listwise Rewards 

**Authors**: Thomas Bailleux, Tanmoy Mukherjee, Emmanuel Lonca, Pierre Marquis, Zied Bouraoui  

**Link**: [PDF](https://arxiv.org/pdf/2604.24176)  

**Abstract**: We reformulate explanation quality assessment as a ranking problem rather than a generation problem. Instead of optimizing models to produce a single "best" explanation token-by-token, we train reward models to discriminate among multiple candidate explanations and learn their relative quality. Concretely, we construct per-instance candidate sets with graded quality levels and train listwise and pairwise ranking models (ListNet, LambdaRank, RankNet) to preserve ordinal structure and avoid score compression typical of pointwise regression or binary preference objectives. We observe three findings: First, ranking losses consistently outperform regression on score separation across all domains tested. Second, the optimal ranking loss depends on data characteristics: listwise objectives excel with well-separated quality tiers, while pairwise methods are more robust to noisy natural annotations. Third, when trained on carefully curated and well-structured data, small encoder models can match models that are orders of magnitude larger, suggesting that data quality matters more than model scale. Finally, when used as rewards in policy optimization, ranking-based scores enable stable convergence in settings where regression-based rewards fail entirely. Code and data are available at: this https URL 

---
# Credal Concept Bottleneck Models for Epistemic-Aleatoric Uncertainty Decomposition 

**Authors**: Tanmoy Mukherjee, Thomas Bailleux, Pierre Marquis, Zied Bouraoui  

**Link**: [PDF](https://arxiv.org/pdf/2604.24170)  

**Abstract**: Concept Bottleneck Models (CBMs) predict through human-interpretable concepts, but they typically output point concept probabilities that conflate epistemic uncertainty (reducible model underspecification) with aleatoric uncertainty (irreducible input ambiguity). This makes concept-level uncertainty hard to interpret and, more importantly, hard to act upon. We introduce CREDENCE (Credal Ensemble Concept Estimation), a CBM framework that decomposes concept uncertainty by construction. CREDENCE represents each concept as a credal prediction (a probability interval), derives epistemic uncertainty from disagreement across diverse concept heads, and estimates aleatoric uncertainty via a dedicated ambiguity output trained to match annotator disagreement when available. The resulting signals support prescriptive decisions: automate low-uncertainty cases, prioritize data collection for high-epistemic cases, route high-aleatoric cases to human review, and abstain when both are high. Across several tasks, we show that epistemic uncertainty is positively associated with prediction errors, whereas aleatoric uncertainty closely tracks annotator disagreement, providing guidance beyond error correlation. Our implementation is available at the following link: this https URL 

---
# Multi-Dimensional Evaluation of Sustainable City Trips with LLM-as-a-Judge and Human-in-the-Loop 

**Authors**: Ashmi Banerjee, Adithi Satish, Wolfgang Wörndl, Yashar Deldjoo  

**Link**: [PDF](https://arxiv.org/pdf/2604.24158)  

**Abstract**: Evaluating nuanced conversational travel recommendations is challenging when human annotations are costly and standard metrics ignore stakeholder-centric goals. We study LLMs-as-Judges for sustainable city-trip lists across four dimensions -- relevance, diversity, sustainability, and popularity balance, and propose a three-phase calibration framework: (1) baseline judging with multiple LLMs, (2) expert evaluation to identify systematic misalignment, and (3) dimension-specific calibration via rules and few-shot examples. Across two recommendation settings, we observe model-specific biases and high dimension-level variance, even when judges agree on overall rankings. Calibration clarifies reasoning per dimension but exposes divergent interpretations of sustainability, highlighting the need for transparent, bias-aware LLM evaluation. Prompts and code are released for reproducibility: this https URL. 

---
# Right-to-Act: A Pre-Execution Non-Compensatory Decision Protocol for AI Systems 

**Authors**: Gadi Lavi  

**Link**: [PDF](https://arxiv.org/pdf/2604.24153)  

**Abstract**: Current AI systems increasingly operate in contexts where their outputs directly trigger real-world actions. Most existing approaches to AI safety, risk management, and governance focus on post-hoc validation, probabilistic risk estimation, or certification of model behavior. However, these approaches implicitly assume that once a decision is produced, it is eligible for execution. In this work, we introduce the Right-to-Act protocol, a deterministic, non-compensatory pre-execution decision layer that evaluates whether an AI-generated decision is permitted to be realized at all. Unlike compensatory systems, where high-confidence signals can override failed conditions, the proposed framework enforces strict structural constraints: if any required condition is unmet, execution is halted or deferred. We formalize the distinction between compensatory and non-compensatory decision regimes and define a pre-execution legitimacy boundary. Through a scenario-based case study, we demonstrate how identical AI outputs can lead to divergent outcomes when evaluated under a Right-to-Act protocol, preserving reversibility and preventing premature or irreversible actions. The proposed approach reframes AI control from optimizing decisions to governing their admissibility, introducing a protocol-level abstraction that operates independently of model architecture or training methodology. 

---
# An Analysis of the Coordination Gap between Joint and Modular Learning for Job Shop Scheduling with Transportation Resources 

**Authors**: Moritz Link, Jonathan Hoss, Noah Klarmann  

**Link**: [PDF](https://arxiv.org/pdf/2604.24117)  

**Abstract**: Efficient job-shop scheduling with transportation resources is critical for high-performance manufacturing. With the rise of "decentralized factories", multi-agent reinforcement learning has emerged as a promising approach for the combined scheduling of production and transportation tasks. Prior work has largely focused on developing novel cooperative architectures while overlooking the question of when joint training is necessary. Joint training denotes the simultaneous training of job and automatic guided vehicle scheduling agents, whereas modular training involves independently training each agent followed by post-hoc integration. In this study, we systematically investigate the conditions under which joint training is essential for optimal performance in the job-shop scheduling problem with transportation resources. Through a rigorous sensitivity analysis of resource scarcity and temporal dominance, we quantify the coordination gap -- the performance difference between these two training modalities. In our evaluation, the joint training can produce superior performance compared to the best-performing combinations of dispatching rules and modular training. However, the coordination gap advantage diminishes in bottleneck environments, particularly under severe transport and processing constraints. These findings indicate that modular training represents a viable alternative in environments where a single scheduling task dominates. Overall, our work provides practical guidance for selecting between training modalities based on environmental conditions, enabling decision-makers to optimize reinforcement learning-based scheduling performance. 

---
# SemML 2.0: Synthesizing Controllers for LTL 

**Authors**: Jan Křetínský, Tobias Meggendorfer, Maximilian Prokop  

**Link**: [PDF](https://arxiv.org/pdf/2604.24102)  

**Abstract**: Synthesizing a reactive system from specifications given in linear temporal logic (LTL) is a classical problem, finding its applications in safety-critical systems design. These systems are typically represented using either Mealy machines or AIGER circuits. We present the second version of SemML, which outperforms all state-of-the-art tools for finding either solution. Aside from implementing the classical automata-theoretic approach, our tool utilizes partial exploration and machine-learning guidance for obtaining solutions efficiently, and numerous heuristics and improvements of classic algorithms for extracting small representations of these solutions. We evaluate our tool against the existing state-of-the-art tools (in particular Strix, LtlSynt, and the previous version of SemML) on the dataset of the synthesis competition SYNTCOMP. We show that we solve significantly more instances and do so much faster than other tools, while maintaining state-of-the-art solution quality. 

---
# The Kerimov-Alekberli Model: An Information-Geometric Framework for Real-Time System Stability 

**Authors**: Hikmat Karimov, Rahid Zahid Alekberli  

**Link**: [PDF](https://arxiv.org/pdf/2604.24083)  

**Abstract**: This study introduces the Kerimov-Alekberli model, a novel information-geometric framework that redefines AI safety by formally linking non-equilibrium thermodynamics to stochastic control for the ethical alignment of autonomous systems. By establishing a formal isomorphism between non-equilibrium thermodynamics and stochastic control, we define systemic anomalies as deviations from a Riemannian manifold. The model utilizes the Kullback-Leibler divergence as the primary metric, governed by a dynamic threshold derived from the Fisher Information Metric. We further ground this framework in the Landauer Principle, proving that adversarial perturbations perform measurable physical work by increasing the system's informational entropy. Validation on the NSL-KDD dataset and unmanned aerial vehicle trajectory simulations demonstrated that our model achieves effective real-time detection via the FPT trigger, with strong performance metrics (e.g., high accuracy and low FPR) on benchmark datasets. This study provides a rigorous physical foundation for AI safety, transitioning from heuristic, rule-based ethical frameworks to a thermodynamics-based stability paradigm by grounding ethical violations in quantifiable physical work and entropic information. 

---
# An Information-Geometric Framework for Stability Analysis of Large Language Models under Entropic Stress 

**Authors**: Hikmat Karimov, Rahid Zahid Alekberli  

**Link**: [PDF](https://arxiv.org/pdf/2604.24076)  

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes and operational settings, evaluation strategies based solely on aggregate accuracy are often insucient to characterize system reliability. This study proposes a thermodynamic inspired modeling framework for analyzing the stability of LLM outputs under conditions of uncertainty and perturbation. The framework introduces a composite stability score that integrates task utility, entropy as a measure of external uncertainty, and two internal structural proxies: internal integration and aligned reective capacity. Rather than interpreting these quantities as physical variables, the formulation is intended as an interpretable abstraction that captures how internal structure may modulate the impact of disorder on model behavior. Using the IST-20 benchmarking protocol and associated metadata, we analyze 80 modelscenario observations across four contemporary LLMs. The proposed formulation consistently yields higher stability scores than a reduced utilityentropy baseline, with a mean improvement of 0.0299 (95% CI: 0.02470.0351). The observed gain is more pronounced under higher entropy conditions, suggesting that the framework captures a form of nonlinear attenuation of uncertainty. We do not claim a fundamental physical law or a complete theory of machine ethics. Instead, the contribution of this work is a compact and interpretable modeling perspective that connects uncertainty, performance, and internal structure within a unied evaluation lens. The framework is intended to complement existing benchmarking approaches and to support ongoing discussions in AI safety, reliability, and governance. 

---
# Grounding Before Generalizing: How AI Differs from Humans in Causal Transfer 

**Authors**: Liangru Xiang, Yuxi Ma, Zhihao Cao, Yixin Zhu, Song-Chun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24062)  

**Abstract**: Extracting abstract causal structures and applying them to novel situations is a hallmark of human intelligence. While Large Language Models (LLMs) and Vision Language Models (VLMs) have shown strong performance on a wide range of reasoning tasks, their capacity for interactive causal learning -- inducing latent structures through sequential exploration and transferring them across contexts -- remains uncharacterized. Human learners accomplish such transfer after minimal exposure, whereas classical Reinforcement Learning (RL) agents fail catastrophically. Whether state-of-the-art Artificial Intelligence (AI) models possess human-like mechanisms for abstract causal structure transfer is an open question. Using the OpenLock paradigm requiring sequential discovery of Common Cause (CC) and Common Effect (CE) structures, here we show that models exhibit fundamentally delayed or absent transfer: even successful models require initial environmental-specific mapping -- what we term environmental grounding -- before efficiency gains emerge, whereas humans leverage prior structural knowledge from the very first solution attempt. In the text-only condition, models matched or exceeded human discovery efficiency. In contrast, visual information -- in both the image-only and text-and-image conditions -- overall degraded rather than enhanced performance, revealing a broad reliance on symbolic processing rather than integrated multimodal reasoning. Models further exhibited systematic CC/CE asymmetries absent in humans, suggesting heuristic biases rather than direction-neutral causal abstraction. These findings reveal that large-scale statistical learning does not produce the decontextualized causal schemas underpinning human analogical reasoning, establishing grounding-dependent transfer as a fundamental limitation of current LLMs and VLMs. 

---
# A2DEPT: Large Language Model-Driven Automated Algorithm Design via Evolutionary Program Trees 

**Authors**: Bin Chen, Shouliang Zhu, Beidan Liu, Yong Zhao, Tianle Pu, Huichun Li, Zhengqiu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24043)  

**Abstract**: Designing heuristics for combinatorial optimization problems (COPs) is a fundamental yet challenging task that traditionally requires extensive domain expertise. Recently, Large Language Model (LLM)-based Automated Heuristic Design (AHD) has shown promise in autonomously generating heuristic components with minimal human intervention. However, most existing LLM-based AHD methods enforce fixed algorithmic templates to ensure executability, which confines the search to component-level tuning and limits system-level algorithmic expressiveness. To enable open-ended solver synthesis beyond rigid templates, we propose Automated Algorithm Design via Evolutionary Program Trees (A2DEPT), which treats LLMs as system-level algorithm architects. A2DEPT explores the vast program space via a tree-structured evolutionary search with hybrid selection and hierarchical operators, enabling iterative refinement of complete algorithms. To make open-ended generation practical, we enforce executability with a lightweight program-maintenance loop that performs feedback-driven repair. In experiments, A2DEPT consistently outperforms representative LLM-based baselines on both standard and highly constrained benchmarks. On the standard benchmarks, it reduces the mean normalized optimality gap by 9.8% relative to the strongest competing AHD baseline. 

---
# AgentPulse: A Continuous Multi-Signal Framework for Evaluating AI Agents in Deployment 

**Authors**: Yuxuan Gao, Megan Wang, Yi Ling Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24038)  

**Abstract**: Static benchmarks measure what AI agents can do at a fixed point in time but not how they are adopted, maintained, or experienced in deployment. We introduce AgentPulse, a continuous evaluation framework scoring 50 agents across 10 workload categories along four factors (Benchmark Performance, Adoption Signals, Community Sentiment, and Ecosystem Health) aggregated from 18 real-time signals across GitHub, package registries, IDE marketplaces, social platforms, and benchmark leaderboards. Three analyses ground the framework. The four factors capture largely complementary information (n=50; $\rho_{\max}=0.61$ for Adoption-Ecosystem, all others $|\rho| \leq 0.37$). A circularity-controlled test (n=35) shows the Benchmark+Sentiment sub-composite, which contains no GitHub-derived signals, predicts external adoption proxies it does not aggregate: GitHub stars ($\rho_s=0.52$, $p<0.01$) and Stack Overflow question volume ($\rho_s=0.49$, $p<0.01$), with VS Code installs ($\rho_s=0.44$, $p<0.05$) reported as illustrative given that only 11 of 35 agents have non-zero installs. On the n=11 subset with published SWE-bench scores, composite and benchmark-only rankings are nearly uncorrelated ($\rho_s=0.25$; 9 of 11 agents shift by at least 2 ranks), driven by a strong negative Adoption-Capability correlation among closed-source high-capability agents within this subset. This is precisely why we rest the framework's validity claim on the broader n=35 test rather than the SWE-bench overlap. AgentPulse surfaces deployment signal absent from benchmarks; it is a methodology, not a ground-truth ranking. The framework, all collected signals, scoring outputs, and evaluation harness are released under CC BY 4.0. 

---
# QED: An Open-Source Multi-Agent System for Generating Mathematical Proofs on Open Problems 

**Authors**: Chenyang An, Qihao Ye, Minghao Pan, Jiayaun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24021)  

**Abstract**: We explore a central question in AI for mathematics: can AI systems produce original, nontrivial proofs for open research problems? Despite strong benchmark performance, producing genuinely novel proofs remains an outstanding challenge for LLMs. Through systematic experiments with frontier LLMs on research-level proof tasks, we identify seven failure modes that prevent reliable proof generation, including context contamination, citation hallucination, hand-waving on key steps and misallocation of proof effort, unstable proof plans, unfocused verification, problem modification and single-model bottleneck. We argue that the gap between benchmark success and research-level proving is primarily one of system design, due to those failure modes. We present QED, an open-source multi-agent proof system in which each architectural decision directly addresses a specific failure mode. Evaluated on five open problems in applied analysis and PDEs contributed by domain experts, QED produces correct proofs for three problems, each verified by the contributing experts as original and nontrivial. QED is released as open-source software at this https URL. 

---
# CT-FineBench: A Diagnostic Fidelity Benchmark for Fine-Grained Evaluation of CT Report Generation 

**Authors**: Ruifeng Yuan, Wanxing Chang, Weiwei Cao, Bowen Shi, Zhongyu Wei, Ling Zhang, Jianpeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24001)  

**Abstract**: The evaluation of generated reports remains a critical challenge in Computed Tomography (CT) report generation, due to the large volume of text, the diversity and complexity of findings, and the presence of fine-grained, disease-oriented attributes. Conventional evaluation metrics offer only coarse measures of lexical overlap or entity matching and fail to reflect the granular diagnostic accuracy required for clinical use. To address this gap, we propose CT-FineBench, a benchmark built from CT-RATE and Merlin to evaluate the fine-grained factual consistency of CT reports, constructed from CT-RATE and Merlin. Our benchmark is constructed through a meticulous, Question-Answering (QA) based process: first, we identify and structure key, finding-specific clinical attributes (like location, size, margin). Second, we systematically transform these attributes into a QA dataset, where questions probe for specific clinical details grounded in gold-standard reports. The evaluation protocol for CT-FineBench involves using this QA dataset to query a machine-generated report and scoring the correctness of the answers. This allows for a comprehensive, interpretable, and clinically-relevant assessment, moving beyond superficial lexical overlap to pinpoint specific clinical errors. Experiments show that CT-FineBench correlates better with expert clinical assessment and is substantially more sensitive to fine-grained factual errors than prior metrics. 

---
# Failure-Centered Runtime Evaluation for Deployed Trilingual Public-Space Agents 

**Authors**: M. Meng  

**Link**: [PDF](https://arxiv.org/pdf/2604.23990)  

**Abstract**: This paper presents PSA-Eval, a failure-centered runtime evaluation framework for deployed trilingual public-space agents. The central claim is that, when the evaluation object shifts from a static input-output mapping to a runtime system, the basic unit of analysis should shift from score to failure. PSA-Eval extends the conventional chain Question -> Answer -> Score -> End into Question -> Batch -> Run -> Score -> Failure Case -> Repair -> Regression Batch, making failures traceable, reviewable, repairable, and regression-testable.
The framework uses trilingual equivalent inputs as controlled probes for observing group-level cross-language policy drift. We conduct a pilot study on a real trilingual digital front-desk system deployed in the lobby of an international financial institution. The pilot uses a simplified single-foundation-model setting (MA = MB), so the observed drift should not be interpreted as an A/B foundation-model difference. The study contains 81 samples organized into 27 trilingual equivalent question groups. Although the system achieves an average score of 23.15/24, 14 groups show non-zero cross-language score drift, 5 groups show drift of at least 3 points, and the maximum drift reaches 9 points. These results provide initial evidence that failure-centered runtime evaluation can expose structured deployment signals hidden by aggregate scoring. 

---
# Representational Curvature Modulates Behavioral Uncertainty in Large Language Models 

**Authors**: Jack King, Evelina Fedorenko, Eghbal A. Hosseini  

**Link**: [PDF](https://arxiv.org/pdf/2604.23985)  

**Abstract**: In autoregressive large language models (LLMs), temporal straightening offers an account of how the next-token prediction objective shapes representations. Models learn to progressively straighten the representational trajectory of input sequences across layers, potentially facilitating next-token prediction via linear extrapolation. However, a direct link between this trajectory and token-level behavior has been missing. We provide such a link by relating contextual curvature-a geometric measure of how sharply the representational trajectory bends over recent context-to next-token entropy. Across two models (GPT-2 XL and Pythia-2.8B), contextual curvature is correlated with entropy, and this relationship emerges during training. Perturbation experiments reveal selective dependence: manipulating curvature through trajectory-aligned interventions reliably modulates entropy, while geometrically misaligned perturbations have no effect. Finally, regularizing representations to be straighter during training modestly reduces token-level entropy without degrading validation loss. These results identify trajectory curvature as a task-aligned representational feature that influences behavioral uncertainty in LLMs. 

---
# LLM-Guided Agentic Floor Plan Parsing for Accessible Indoor Navigation of Blind and Low-Vision People 

**Authors**: Aydin Ayanzadeh, Tim Oates  

**Link**: [PDF](https://arxiv.org/pdf/2604.23970)  

**Abstract**: Indoor navigation remains a critical accessibility challenge for the blind and low-vision (BLV) individuals, as existing solutions rely on costly per-building infrastructure. We present an agentic framework that converts a single floor plan image into a structured, retrievable knowledge base to generate safe, accessible navigation instructions with lightweight infrastructure. The system has two phases: a multi-agent module that parses the floor plan into a spatial knowledge graph through a self-correcting pipeline with iterative retry loops and corrective feedback; and a Path Planner that generates accessible navigation instructions, with a Safety Evaluator agent assessing potential hazards along each route. We evaluate the system on the real-world UMBC Math and Psychology building (floors MP-1 and MP-3) and on the CVC-FP benchmark. On MP-1, we achieve success rates of 92.31%, 76.92%, and 61.54% for short, medium, and long routes, outperforming the strongest single-call baseline (Claude 3.7 Sonnet) at 84.62%, 69.23%, and 53.85%. On MP-3, we reach 76.92%, 61.54%, and 38.46%, compared to the best baseline at 61.54%, 46.15%, and 23.08%. These results show consistent gains over single-call LLM baselines and demonstrate that our workflow is a scalable solution for accessible indoor navigation for BLV individuals. 

---
# An empirical evaluation of the risks of AI model updates using clinical data: stability, arbitrariness, and fairness 

**Authors**: Ioannis Bilionis, Ricardo C. Berrios, Luis Fernandez-Luque, Carlos Castillo  

**Link**: [PDF](https://arxiv.org/pdf/2604.23954)  

**Abstract**: Artificial Intelligence and Machine Learning (AI/ML) models used in clinical settings are increasingly deployed to support clinical decision-making. However, when training data become stale due to changes in demographics, environment, or patient behaviors, model performance can degrade substantially. While updating models with new training data is necessary, such updates may also introduce new risks. We evaluated the proposed monitoring framework on four publicly available U.S.-based Type 1 Diabetes datasets containing high-resolution continuous glucose monitoring (CGM) data, comprising approximately 11,300 weekly observations from 496 participants under 20 years of age. All datasets included structured sociodemographic information. Using the prediction of severe hyperglycemia events in children with type 1 diabetes as a case study, we examine how different model update strategies can adversely affect model stability (e.g., by causing predictions to "flip" for a large number of cases after an update), increase arbitrariness in predictions, or worsen accuracy equity and the balance of error rates across subpopulations. We propose multiple dimensions for continuous monitoring to detect these issues and argue that such monitoring is essential for the development of trustworthy clinical decision support systems. 

---
# Context-Aware Hospitalization Forecasting Evaluations for Decision Support using LLMs 

**Authors**: Rhea Makkuni, Ananya Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23949)  

**Abstract**: Medical and public health experts must make real-time resource decisions, such as expanding hospital bed capacity, based on projected hospitalization trends during large-scale healthcare disruptions (e.g., operational failures or pandemics). Forecasting models can assist in this task by analyzing large volumes of resource-related data at the facility level, but they must be reliable for decision-making under real-world data conditions. Recent work shows that large language models (LLMs) can incorporate richer forms of context into numerical forecasting. Whereas traditional models rely primarily on temporal context (i.e., past observations), LLMs can also leverage non-temporal public health context such as demographic, geographic, and population-level features. However, it remains unclear how these models should be used to produce stable or decision-relevant predictions in real-world healthcare settings. To evaluate how LLMs can be effectively used in this setting, we evaluate three approaches across 60 counties with low-,mid-, and high-hospitalization intensities in the United States: direct LLM-based forecasting, classical time-series models, and a context-augmented hybrid pipeline (HybridARX) that incorporates LLM-derived signals into structured models. Because the goal is operational decision-making rather than error minimization alone, we evaluate performance with bias and lead-lag alignment in addition to standard forecasting metrics. Our results show that HybridARX improves over classical ARX by yielding more stable and better-calibrated forecasts, particularly when incorporating noisy contextual signals into structured time-series models. These findings suggest that, in non-stationary healthcare resource forecasting, LLMs are most useful when embedded within structured hybrid models. 

---
# GAMED.AI: A Hierarchical Multi-Agent Framework for Automated Educational Game Generation 

**Authors**: Shiven Agarwal, Yash Shah, Ashish Raj Shekhar, Priyanuj Bordoloi, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.23947)  

**Abstract**: We introduce GameDAI, a hierarchical multi-agent framework that transforms instructor-provided questions into fully playable, pedagogically grounded educational games validated through formal mechanic contracts. Built on phase-based LangGraph sub-graphs, deterministic Quality Gates, and structured Pydantic schemas, GameDAI supports two template families encompassing 15 interaction mechanics across spatial reasoning, procedural execution, and higher-order Bloom's Taxonomy objectives. Evaluated on 200 questions spanning five subject domains, the system achieves a 90% validation pass rate, 98.3% schema compliance, and 73% token reduction over ReAct agents (${\sim}$73,500 $\rightarrow$ ${\sim}$19,900 tokens/game) at $0.46 per game. Within this model configuration, these results suggest that phase-bounded architectural structure correlates more strongly with alignment quality than prompting strategy alone. Our demonstration lets attendees generate Bloom's-aligned games from natural language in under 60 seconds, inspect Quality Gate outputs at each pipeline phase, and browse a curated library of 50 games spanning all 15 mechanic types. 

---
# Agentic AI platforms for autonomous training and rule induction of human-human and virus-human protein-protein interactions 

**Authors**: Hung N. Do, Jessica Z. Kubicek-Sutherland, Oscar A. Negrete, S. Gnanakaran  

**Link**: [PDF](https://arxiv.org/pdf/2604.23924)  

**Abstract**: We instruct an AI agent to construct two separate agentic AI platforms: one for autonomous training of predictive ML models for human-human and virus-human PPI, and the other for inducing explicit general rules governing human-human and virus-human PPI. The first agentic AI platform for autonomous training of predictive ML models for PPI is designed to consist of five AI agents that handle autonomous data collection, data verification, feature embedding, model design, and training and validation on three-way protein-disjoint cross-fold datasets. For human-human and human-virus PPIs, the final three-way protein-disjoint ensemble achieves an accuracy of 87.3% and 86.5%, respectively. For cross-checking and interpretability purposes, the second agentic AI platform is designed to replace ML predictions with human-readable rules derived from protein embeddings, physicochemical autocovariance descriptors, compartment annotations, pathway-domain overlap, and graph contexts. For human-human PPI, it is defined by a two-rule induction, whereas human-virus is induced by a more complex set of weighted rules. The rules induced by the second agentic platform align with the SHAP-identified features from the predictive ML models built by the first agentic platform. Taken together, our work demonstrates the agentic AI's ability to orchestrate from data planning to execution, and from rule induction to explanation in ML, opening the door to various applications. 

---
# LLM-Augmented Traffic Signal Control with LSTM-Based Traffic State Prediction and Safety-Constrained Decision Support 

**Authors**: Jiazhao Shi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23902)  

**Abstract**: Traffic signal control is a critical task in intelligent transportation systems, yet conventional fixed-time and rule-based methods often struggle to adapt to dynamic traffic demand and provide limited decision interpretability. This study proposes an LLM-augmented traffic signal control framework that integrates LSTM-based short-term traffic state prediction, predictive phase selection, structured large language model reasoning, and safety-constrained action filtering. The LSTM module forecasts future queue length, waiting time, vehicle count, and lane occupancy based on recent intersection-level observations. A predictive controller then generates candidate signal actions, while the LLM module evaluates these actions using structured traffic-state inputs and produces congestion diagnoses, phase adjustment recommendations, and natural-language explanations. To ensure operational reliability, all LLM-generated recommendations are validated by a safety filter before execution. Simulation-based experiments in SUMO compare the proposed method with fixed-time control, rule-based control, and an LSTM-based predictive baseline under balanced demand, directional peak demand, and sudden surge scenarios. The results indicate that the proposed framework improves traffic efficiency, especially under dynamic and non-recurrent traffic conditions, while maintaining zero constraint violations after safety filtering. Overall, this study demonstrates that LLMs can enhance traffic signal control when used as constrained reasoning and decision-support modules rather than direct low-level controllers. Keywords: Intelligent Transportation Systems; Traffic Signal Control; Large Language Models; LSTM; Traffic State Prediction; Decision Support; Safety-Constrained Control; SUMO Simulation. 

---
# MarketBench: Evaluating AI Agents as Market Participants 

**Authors**: Andrey Fradkin, Rohit Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2604.23897)  

**Abstract**: Markets are a promising way to coordinate AI agent activity for similar reasons to those used to justify markets more broadly. In order to effectively participate in markets, agents need to have informative signals of their own ability to successfully complete a task and the cost of doing so. We propose MarketBench, a benchmark for assessing whether AI agents have these capabilities. We use a 93-task subset of SWE-bench Lite, a software engineering benchmark, with six recently released LLMs as a demonstration. These LLMs are miscalibrated on both success probability and token usage, and auctions built from these self-reports diverge from a full-information allocation. A follow-up intervention where we add information about capabilities from prior experiments to the context improves calibration, but only modestly narrows the gap to a full-information benchmark. We also document the performance of a market-based scaffolding with these LLMs. Our results point to self-assessment as a key bottleneck for market-style coordination of AI agents. 

---
# ZenBrain: A Neuroscience-Inspired 7-Layer Memory Architecture for Autonomous AI Systems 

**Authors**: Alexander Bering  

**Link**: [PDF](https://arxiv.org/pdf/2604.23878)  

**Abstract**: Despite a century of empirical memory research, existing AI agent memory systems rely on system-engineering metaphors (virtual-memory paging, flat LLM storage, Zettelkasten notes), none integrating principles of consolidation, forgetting, and reconsolidation.
We present ZenBrain, a multi-layer memory architecture integrating fifteen neuroscience models. It implements seven memory layers (working, short-term, episodic, semantic, procedural, core, cross-context) orchestrated by nine foundational algorithms (Two-Factor Synaptic Model, vmPFC-coupled FSRS, Simulation-Selection sleep, Bayesian confidence, and five more) plus six new Predictive Memory Architecture (PMA) components: a four-channel NeuromodulatorEngine, prediction-error-gated ReconsolidationEngine, TripleCopyMemory with divergent decay, four-dimensional PriorityMap with amygdala fast-path, StabilityProtector (NogoA/HDAC3 analogue), and MetacognitiveMonitor for bias detection.
The 15-algorithm ablation reveals a cooperative survival network: under stress, 9 of 15 algorithms become individually critical (delta-Q up to -93.7%, Wilcoxon, 10 seeds, alpha=0.005). Simulation-Selection sleep achieves 37% stability improvement (p<0.005) with 47.4% storage reduction. TripleCopyMemory retains S(t)=0.912 at 30 days; PriorityMap reaches NDCG@10=0.997.
Multi-layer routing beats a flat single-layer baseline by 20.7% F1 on LoCoMo (p<0.005) and 19.5% on MemoryArena (p=0.015). On LongMemEval-500, ZenBrain holds the highest mean rank on all 12 system-judge cells (4 systems x 3 LLM judges), three-judge mean J=0.545 vs letta=0.485, a-mem=0.414, mem0=0.394; all 9 pair-wise contrasts clear Bonferroni (alpha=0.05/18, min p=6.2e-31, d in [0.18, 0.52]). Under LongMemEval's binary judge, ZenBrain reaches 91.3% of oracle accuracy at 1/106th the per-query token budget. Open-source with 11,589 automated test cases. 

---
# Time-Series Forecasting in Safety-Critical Environments: An EU-AI-Act-Compliant Open-Source Package / Zeitreihenprognose in sicherheitskritischen Umgebungen: Ein KI-VO-konformes Open-Source-Paket 

**Authors**: Thomas Bartz-Beielstein, Eva Bartz  

**Link**: [PDF](https://arxiv.org/pdf/2604.23859)  

**Abstract**: With spotforecast2-safe we present an integrated Compliance-by-Design approach to Python-based point forecasting of time series in safety-critical environments. A review of the relevant open-source tooling shows that existing compliance solutions operate consistently outside of the library to be used - e.g. as scanners, templates, or runtime layers. spotforecast2-safe takes the inverse approach and anchors the requirements of Regulation (EU) 2024/1689 (the EU AI Act, in German: KI-VO), of IEC 61508, of the ISA/IEC 62443 standards series, and of the Cyber Resilience Act within the library: in application-programming-interface contracts, persistence formats, and continuous-integration gates. The approach is operationalised by four non-negotiable code-development rules (zero dead code, deterministic processing, fail-safe handling, minimal dependencies) together with the corresponding process rules (model card, executable docstrings, CI workflows, Common-Platform-Enumeration (CPE) identifier, REUSE-conformant licensing, release pipeline). Interactive visualisation, hyperparameter tuning and automated machine learning (AutoML), as well as deep-learning and large-language-model backends are deliberately excluded, because each of these components either enlarges the attack surface, introduces non-determinism, or impairs reproducibility. A bidirectional traceability matrix maps every regulatory provision onto the corresponding mechanism in the code; an end-to-end example of European-market electricity generation, transmission, and consumption forecasting demonstrates the application. The package is open-source and available under Affero General Public License (AGPL) 3.0-or-later. 

---
# Does Machine Unlearning Preserve Clinical Safety? A Risk Analysis for Medical Image Classification 

**Authors**: Andreza M. C. Falcao, Filipe R. Cordeiro  

**Link**: [PDF](https://arxiv.org/pdf/2604.23854)  

**Abstract**: The application of Deep Learning in medical diagnosis must balance patient safety with compliance with data protection regulations. Machine Unlearning enables the selective removal of training data from deployed models. However, most methods are validated primarily through efficiency and privacy-oriented metrics, with limited attention to clinically asymmetric error costs. In this work, we investigate how unlearning affects clinical risk in binary medical image classification. We show that standard unlearning strategies (Fine-Tuning, Random Labeling, and SalUn) may reduce test utility while increasing false-negative rates, thereby amplifying clinical risk. To mitigate this, we propose SalUn-CRA (Clinical Risk-Aware), a variant of SalUn that replaces random relabeling with entropy-based forgetting for malignant samples in the forget set, preventing the model from learning harmful benign associations. We evaluate on DermaMNIST and PathMNIST medical image datasets under 20% and 50% data removal. Using Global Risk metrics with asymmetric costs, SalUn-CRA achieves lower or comparable clinical risk to full retraining while preserving unlearning effectiveness. These results suggest that clinical risk should be an integral component of unlearning validation in medical systems. 

---
# ClawTrace: Cost-Aware Tracing for LLM Agent Skill Distillation 

**Authors**: Boqin Yuan, Renchu Song, Yue Su, Sen Yang, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23853)  

**Abstract**: Skill-distillation pipelines learn reusable rules from LLM agent trajectories, but they lack a key signal: how much each step costs. Without per-step cost, a pipeline cannot distinguish adding a missing step to fix a bug from removing an expensive step that never affected the outcome. We introduce ClawTrace, an agent tracing platform that records every LLM call, tool use, and sub-agent spawn during an agent session and compiles each session into a TraceCard: a compact YAML summary with per-step USD cost, token counts, and redundancy flags. Built on ClawTrace, CostCraft is a distillation pipeline that reads TraceCards and produces three types of skill patches. Preserve patches keep behaviors that led to success. Prune patches remove expensive steps that did not matter, each backed by a counterfactual argument against a named high-cost step. Repair patches fix failures grounded in oracle evidence. Ablations on 30 held-out SpreadsheetBench tasks show that both cost attribution and prune patches independently reduce quality regressions. When the same skill is applied to 30 unrelated SkillsBench tasks, an unexpected asymmetry emerges: prune rules transferred across benchmarks and cut median cost by 32%, while preserve rules, trained on benchmark-specific conventions, caused regressions on new task types. We release ClawTrace and TraceCards as open infrastructure for cost-aware agent research. 

---
# Domain-Filtered Knowledge Graphs from Sparse Autoencoder Features 

**Authors**: John Winnicki, Abeynaya Gnanasekaran, Eric Darve  

**Link**: [PDF](https://arxiv.org/pdf/2604.23829)  

**Abstract**: Sparse autoencoders (SAEs) extract millions of interpretable features from a language model, but flat feature inventories aren't very useful on their own. Domain concepts get mixed with generic and weakly grounded features, while related ideas are scattered across many units, and there's no way to understand relationships between features. We address this by first constructing a strict domain-specific concept universe from a large SAE inventory using contrastive activations and a multi-stage filtering process. Next, we build two aligned graph views on the filtered set: a co-occurrence graph for corpus-level conceptual structure, organized at multiple levels of granularity, and a transcoder-based mechanism graph that links source-layer and target-layer features through sparse latent pathways. Automated edge labeling then turns these graph views into readable knowledge graphs rather than unlabeled layouts. In a case study on a biology textbook, these graphs recover coherent chapter and subchapter-level structure, reveal concepts that bridge neighboring topics, and transform messy sentence-level activity containing thousands of features into compact, readable views that illustrate the model's local activity. Taken together, this reframes a flat SAE inventory as an internal knowledge graph that converts feature-level interpretability into a global map of model knowledge and enables audits of reasoning faithfulness. 

---
# FAIR_XAI: Improving Multimodal Foundation Model Fairness via Explainability for Wellbeing Assessment 

**Authors**: Sophie Chiang, Tom Brennan, Fethiye Irmak Dogan, Jiaee Cheong, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2604.23786)  

**Abstract**: In recent years, the integration of multimodal machine learning in wellbeing assessment has offered transformative potential for monitoring mental health. However, with the rapid advancement of Vision-Language Models (VLMs), their deployment in clinical settings has raised concerns due to their lack of transparency and potential for bias. While previous research has explored the intersection of fairness and Explainable AI (XAI), its application to VLMs for wellbeing assessment and depression prediction remains under-explored. This work investigates VLM performance across laboratory (AFAR-BSFT) and naturalistic (E-DAIC) datasets, focusing on diagnostic reliability and demographic fairness. Performance varied substantially across environments and architectures; Phi3.5-Vision achieved 80.4% accuracy on E-DAIC, while Qwen2-VL struggled at 33.9%. Additionally, both models demonstrated a tendency to over-predict depression on AFAR-BSFT. Although bias existed across both architectures, Qwen2-VL showed higher gender disparities, while Phi-3.5-Vision exhibited more racial bias. Our XAI intervention framework yielded mixed results; fairness prompting achieved perfect equal opportunity for Qwen2-VL at a severe accuracy cost on E-DAIC. On AFAR-BSFT, explainability-based interventions improved procedural consistency but did not guarantee outcome fairness, sometimes amplifying racial bias. These results highlight a persistent gap between procedural transparency and equitable outcomes. We analyse these findings and consolidate concrete recommendations for addressing them, emphasising that future fairness interventions must jointly optimise predictive accuracy, demographic parity, and cross-domain generalisation. 

---
# Modeling Induced Pleasure through Cognitive Appraisal Prediction via Multimodal Fusion 

**Authors**: Nastaran Dab, Raziyeh Zall, Mohammadreza Kangavari  

**Link**: [PDF](https://arxiv.org/pdf/2604.23753)  

**Abstract**: Multimodal affective computing analyzes user-generated social media content to predict emotional states. However, a critical gap remains in understanding how visual content shapes cognitive interpretations and elicits specific affective experiences such as pleasure. This study introduces a novel computational model to infer video-induced pleasure via cognitive appraisal variables. The proposed model addresses four challenges: (1) noisy and inconsistent human labels, (2) the semantic gap between "positive emotions" and "pleasure," (3) the scarcity of pleasure-specific datasets, and (4) the limited interpretability of existing black-box fusion methods. Our approach integrates data-driven and cognitive theory-driven methods, using cognitive appraisal theory and a fuzzy model within an innovative framework. The model employs transformer-based architectures and attention mechanisms for fine-grained multimodal feature extraction and interpretable fusion to capture both inter- and intra-modal dynamics associated with pleasure. This enables the prediction of underlying appraisal variables, thereby bridging the semantic gap and enhancing model explainability beyond conventional statistical associations. Experimental results validate the efficacy of the proposed method in detecting video-induced pleasure, achieving a peak accuracy of 0.6624 in predicting pleasure levels. These findings highlight promising implications for affective content recommendation, intelligent media creation, and advancing our understanding of how digital media influences human emotions. 

---
# Expert Evaluation of LLM's Open-Ended Legal Reasoning on the Japanese Bar Exam Writing Task 

**Authors**: Jungmin Choi, Keisuke Sakaguchi, Hiroaki Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2604.23730)  

**Abstract**: Large language models (LLMs) have shown strong performance on legal benchmarks, including multiple-choice components of bar exams. However, their capacity for generating open-ended legal reasoning in realistic scenarios remains insufficiently explored. Notably, to our best knowledge, there are no prior studies or datasets addressing this issue in the Japanese context.
This study presents the first dataset designed to evaluate the open-ended legal reasoning performance of LLMs within the Japanese jurisdiction. The dataset is based on the writing component of the Japanese bar examination, which requires examinees to identify multiple legal issues from long narratives and to construct structured legal arguments in free text format. Our key contribution is the manual evaluation of LLMs' generated responses by legal experts, which reveals limitations and challenges in legal reasoning. Moreover, we conducted a manual analysis of hallucinations to characterize when and how the models introduce content not supported by precedent or law.
Our real exam questions, model-generated responses, and expert evaluations reveal the milestones of current LLMs in the Japanese legal domain. Our dataset and relevant resources will be available online. 

---
# Information-Theoretic Measures in AI: A Practical Decision Guide 

**Authors**: Nikolaos Al.Papadopoulos, Konstantinos E. Psannis  

**Link**: [PDF](https://arxiv.org/pdf/2604.23716)  

**Abstract**: Information-theoretic (IT) measures are ubiquitous in artificial intelligence: entropy drives decision-tree splits and uncertainty quantification, cross-entropy is the default classification loss, mutual information underpins representation learning and feature selection, and transfer entropy reveals directed influence in dynamical systems.
A second, less consolidated family of measures, integrated information (Phi), effective information (EI), and autonomy, has emerged for characterizing agent complexity. Despite wide adoption, measure selection is often decoupled from estimator assumptions, failure modes, and safe inferential claims.
This paper provides a practical decision framework for all seven measures, organized around three prescriptive questions for each: (i) what question does the measure answer and in which AI context; (ii) which estimator is appropriate for the data type and dimensionality; and (iii) what is the most dangerous misuse. The framework is operationalized in two complementary artifacts: a measure-selection flowchart and a master decision table. We cover both AI/ML and decision-making agent application domains per measure, with standardized Bridge Boxes linking IT quantities to cognitive constructs. Three worked examples illustrate the framework on concrete practitioner scenarios spanning representation learning, temporal influence analysis, and evolved agent complexity. 

---
# Transferable Human Mobility Network Reconstruction with neuroGravity 

**Authors**: Jinming Yang, Shaoyu Huang, Zongyuan Huang, Yaohui Jin, Xiaokang Yang, Marta C. Gonzalez, Yanyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23678)  

**Abstract**: Accurate modeling of human mobility is critical for tackling urban planning and public health challenges. In undeveloped regions, the absence of comprehensive travel surveys necessitates reconstructing mobility networks from publicly available data. Here we develop neuroGravity, a physics-informed deep learning model that reliably reconstructs mobility flows from limited observations and transfers to unobserved cities. Using only urban facility and population distributions, we find that neuroGravity's regional representations strongly correlate with socioeconomic and livability status, offering scalable proxies for costly surveys. Furthermore, we uncover that spatial income segregation plays a key role in model transferability: mobility networks are most reliably reconstructed when target cities share similar segregation levels with the source. We design an index to quantify this segregation and accurately predict transferability. Finally, we generate mobility flow proxies for over 1,200 cities worldwide, highlighting neuroGravity's potential to mitigate critical data shortages in resource-limited, underdeveloped areas. 

---
# Vibe Medicine: Redefining Biomedical Research Through Human-AI Co-Work 

**Authors**: Zihao Wu, Steven Xu, Bowen Chen, Shaowen Wan, Yiwei Li, Wei Ruan, Yanjun Lyu, Siyuan Li, Dajiang Zhu, Tianming Liu, Lin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.23674)  

**Abstract**: With the emergence of large language models (LLMs) and AI agent frameworks, the human-AI co-work paradigm known as Vibe Coding is changing how people code, making it more accessible and productive. In scientific research, where workflows are more complex and the burden of specialized labor limits independent researchers and those in low-resource areas, the potential impact is even greater, particularly in biomedicine, which involves heterogeneous data modalities and multi-step analytical pipelines. In this paper, we introduce Vibe Medicine, a co-work paradigm in which clinicians and researchers direct skill-augmented AI agents through natural language to execute complex, multi-step biomedical workflows, while retaining the role of research director who specifies objectives, reviews intermediate results, and makes domain-informed decisions. The enabling infrastructure consists of three layers: capable LLMs, agent frameworks such as OpenClaw and Hermes Agent, and the OpenClaw medical skills collection, which includes more than 1,000 curated skills from multiple open-source repositories. We analyze the architecture and skill categories of this collection across ten biomedical domains, and present case studies covering rare disease diagnosis, drug repurposing, and clinical trial design that demonstrate end-to-end workflows in practice. We also identify the principal risks, such as hallucination, data privacy, and over-reliance, and outline directions toward more reliable, trustworthy, and clinically integrated agent-assisted research that advances research and technological equity and reduces health care resource disparities. 

---
# Structural Enforcement of Goal Integrity in AI Agents via Separation-of-Powers Architecture 

**Authors**: Rong Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23646)  

**Abstract**: Recent evidence suggests that frontier AI systems can exhibit agentic misalignment, generating and executing harmful actions derived from internally constructed goals, even without explicit user requests. Existing mitigation methods, such as Reinforcement Learning from Human Feedback (RLHF) and constitutional prompting, operate primarily at the model level and provide only probabilistic safety guarantees. We propose the Policy-Execution-Authorization (PEA) architecture, a "separation-of-powers" design that enforces safety at the system level. PEA decouples intent generation, authorization, and execution into independent, isolated layers connected via cryptographically constrained capability tokens. We present five core contributions: (C1) an Intent Verification Layer (IVL) for ensuring capability-intent consistency; (C2) Intent Lineage Tracking (ILT), which binds all executable intents to the originating user request via cryptographic anchors; (C3) Goal Drift Detection, which rejects semantically divergent intents below a configurable threshold; (C4) an Output Semantic Gate (OSG) that detects implicit coercion using a structured $K \times I \times P$ threat calculus (Knowledge, Influence, Policy); and (C5) a formal verification framework proving that goal integrity is maintained even under adversarial model compromise. By shifting agent alignment from a behavioral property to a structurally enforced system constraint, PEA provides a robust foundation for the governance of autonomous agents. 

---
# Causal Discovery as Dialectical Aggregation: A Quantitative Argumentation Framework 

**Authors**: Sheng Wei, Yulin Chen, Beishui Liao  

**Link**: [PDF](https://arxiv.org/pdf/2604.23633)  

**Abstract**: Constraint-based causal discovery is brittle in finite-sample regimes because erroneous conditional-independence (CI) decisions can cascade into substantial structural errors. We propose Quantitative Argumentation for Causal Discovery (QACD), a semantics-driven framework that represents CI outcomes as graded, defeasible arguments rather than irreversible constraints. QACD maps statistical test outcomes to argument strengths and aggregates conflicting evidence through connectivity-mediated witness propagation, producing a fixed-point acceptability labeling over candidate adjacencies. Experiments on standard benchmark Bayesian networks suggest that QACD improves structural coherence and interventional reliability in several noisy or inconsistent CI regimes, while remaining competitive with classical constraint-based, hybrid, and prior argumentation-based baselines. 

---
# Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning 

**Authors**: Zichuan Fu, Xian Wu, Guojing Li, Yejing Wang, Yijun Chen, Zihao Zhao, Yixuan Luo, Hanyu Yan, Yefeng Zheng, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.23623)  

**Abstract**: Recent advancements in large language models (LLMs) have catalyzed the rise of reasoning-intensive inference paradigms, where models perform explicit step-by-step reasoning before generating final answers. While such approaches improve answer quality and interpretability, they incur substantial computational overhead due to the prolonged generation sequences. In this paper, we propose Tandem, a novel collaborative framework that synergizes large and small language models (LLMs and SLMs) to achieve high-quality reasoning with significantly reduced computational cost. Specifically, the LLM serves as a strategic coordinator, efficiently generating a compact set of critical reasoning insights. These insights are then used to guide a smaller, more efficient SLM in executing the full reasoning process and delivering the final response. To balance efficiency and reliability, Tandem introduces a cost-aware termination mechanism that adaptively determines when sufficient reasoning guidance has been accumulated, enabling early stopping of the LLM's generation. Experiments on mathematical reasoning and code generation benchmarks demonstrate that Tandem reduces computational costs by approximately 40% compared to standalone LLM reasoning, while achieving superior or competitive performance. Furthermore, the sufficiency classifier trained on one domain transfers effectively to others without retraining. The code is available at: this https URL. 

---
# Thinking Like a Clinician: A Cognitive AI Agent for Clinical Diagnosis via Panoramic Profiling and Adversarial Debate 

**Authors**: Zhiqi Lv, Duofan Tu, Jun Li, Mingyue Zhao, Heqin Zhu, Wenliang Li, Shaohua Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.23605)  

**Abstract**: The application of large language models (LLMs) in clinical decision support faces significant challenges of "tunnel vision" and diagnostic hallucinations present in their processing unstructured electronic health records (EHRs). To address these challenges, we propose a novel chain-based clinical reasoning framework, called DxChain, which transforms the diagnostic workflow into an iterative process by mirroring a clinician's cognitive trajectory that consists of "Memory Anchoring", "Navigation" and "Verification" phases. DxChain introduces three key methodological innovations to elicit the potential of LLM: (i) a Profile-Then-Plan paradigm to mitigate cold-start hallucinations by establishing a panoramic patient baseline, (ii) a Medical Tree-of-Thoughts (Med-ToT) algorithm for strategic look ahead planning and resource aware navigation, and (iii) a Dialectical Diagnostic Verification procedure utilizing "Angel-Devil" adversarial debates to resolve complex evidence conflicts. Evaluated on two real world benchmarks, MIMIC-IV-Ext Cardiac Disease and MIMIC-IV-Ext CDM, DxChain achieves state-of-the-art performances in both diagnostic accuracy and logical consistency, offering a modular and reliable architecture for next-generation clinical AI. The code is at this https URL. 

---
# When AI reviews science: Can we trust the referee? 

**Authors**: Jialiang Wang, Yuchen Liu, Hang Xu, Kaichun Hu, Shimin Di, Wangze Ni, Linan Yue, Min-Ling Zhang, Kui Ren, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23593)  

**Abstract**: The volume of scientific submissions continues to climb, outpacing the capacity of qualified human referees and stretching editorial timelines. At the same time, modern large language models (LLMs) offer impressive capabilities in summarization, fact checking, and literature triage, making the integration of AI into peer review increasingly attractive -- and, in practice, unavoidable. Yet early deployments and informal adoption have exposed acute failure modes. Recent incidents have revealed that hidden prompt injections embedded in manuscripts can steer LLM-generated reviews toward unjustifiably positive judgments. Complementary studies have also demonstrated brittleness to adversarial phrasing, authority and length biases, and hallucinated claims. These episodes raise a central question for scholarly communication: when AI reviews science, can we trust the AI referee? This paper provides a security- and reliability-centered analysis of AI peer review. We map attacks across the review lifecycle -- training and data retrieval, desk review, deep review, rebuttal, and system-level. We instantiate this taxonomy with four treatment-control probes on a stratified set of ICLR 2025 submissions, using two advanced LLM-based referees to isolate the causal effects of prestige framing, assertion strength, rebuttal sycophancy, and contextual poisoning on review scores. Together, this taxonomy and experimental audit provide an evidence-based baseline for assessing and tracking the reliability of AI peer review and highlight concrete failure points to guide targeted, testable mitigations. 

---
# FinGround: Detecting and Grounding Financial Hallucinations via Atomic Claim Verification 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23588)  

**Abstract**: Financial AI systems must produce answers grounded in specific regulatory filings, yet current LLMs fabricate metrics, invent citations, and miscalculate derived quantities. These errors carry direct regulatory consequences as the EU AI Act's high-risk enforcement deadline approaches (August 2026). Existing hallucination detectors treat all claims uniformly, missing 43% of computational errors that require arithmetic re-verification against structured tables. We present FinGround, a three-stage verify-then-ground pipeline for financial document QA. Stage 1 performs finance-aware hybrid retrieval over text and tables. Stage 2 decomposes answers into atomic claims classified by a six-type financial taxonomy and verified with type-routed strategies including formula reconstruction. Stage 3 rewrites unsupported claims with paragraph- and table-cell-level citations. To cleanly isolate verification value from retrieval quality, we propose retrieval-equalized evaluation as standard methodology for RAG verification research: when all systems receive identical retrieval, FinGround still reduces hallucination rates by 68% over the strongest baseline ($p < 0.01$). The full pipeline achieves a 78% reduction relative to GPT-4o. An 8B distilled detector retains 91.4% F1 at 18x lower per-claim latency, enabling $0.003/query deployment, supported by qualitative signals from a four-week analyst pilot. 

---
# MetaGAI: A Large-Scale and High-Quality Benchmark for Generative AI Model and Data Card Generation 

**Authors**: Haoxuan Zhang, Ruochi Li, Yang Zhang, Zhenni Liang, Junhua Ding, Ting Xiao, Haihua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23539)  

**Abstract**: The rapid proliferation of Generative AI necessitates rigorous documentation standards for transparency and governance. However, manual creation of Model and Data Cards is not scalable, while automated approaches lack large-scale, high-fidelity benchmarks for systematic evaluation. We introduce MetaGAI, a comprehensive benchmark comprising 2,541 verified document triplets constructed through semantic triangulation of academic papers, GitHub repositories, and Hugging Face artifacts. Unlike prior single-source datasets, MetaGAI employs a multi-agent framework with specialized Retriever, Generator, and Editor agents, validated through four-dimensional human-in-the-loop assessment, including human evaluation of editor-refined ground truth. We establish a robust evaluation protocol combining automated metrics with validated LLM-as-a-Judge frameworks. Extensive analysis reveals that sparse Mixture-of-Experts architectures achieve superior cost-quality efficiency, while a fundamental trade-off exists between faithfulness and completeness. MetaGAI provides a foundational testbed for benchmarking, training, and analyzing automated Model and Data Card generation methods at scale. Our data and code are available at: this https URL. 

---
# Do Transaction-Level and Actor-Level AML Queues Agree? An Empirical Evaluation of Granularity Effects on the Elliptic++ Graph 

**Authors**: Ankur Malik  

**Link**: [PDF](https://arxiv.org/pdf/2604.23494)  

**Abstract**: Graph-based anti-money laundering (AML) systems on blockchain networks can score suspicious activity at two granularity levels -- transactions or actor addresses -- yet compliance action is conducted per actor. This paper contributes an evaluation methodology for measuring how scoring granularity affects investigation queue composition under fixed review budgets. We formalize the evaluation through a projection framework mapping transaction-level scores to the actor-level action unit via four aggregation operators, and introduce budgeted investigation metrics -- yield@budget, burden decomposition, and case fragmentation. Using the public Elliptic++ Bitcoin dataset (203,769 transactions; 822,942 address occurrences), we train independent random forest classifiers at each level under a causal temporal protocol and compare review queues through Jaccard overlap, burden decomposition, and feature-matching ablations. At one-percent budget, temporal evaluation yields mean Jaccard of 0.374 (SD 0.171); static pooled evaluation yields 0.087 (95% CI [0.079, 0.094]). An enriched address model receiving all 237 features produces even lower overlap (Jaccard=0.051), with 4.3% illicit per 100 reviews versus 30.2% for the transaction-projected queue. Address-level detection value is temporally concentrated: two timesteps exceed 91% illicit per 100 reviews while the static burden is only 3.4%. A fixed hybrid policy underperforms the best single-level queue by 5.05pp (CI [-10.2pp, -0.9pp]). These findings establish that scoring granularity is a consequential design variable for AML investigation systems -- same data, same budget, different queues, different addresses investigated. 

---
# Agentic Adversarial Rewriting Exposes Architectural Vulnerabilities in Black-Box NLP Pipelines 

**Authors**: Mazal Bethany, Kim-Kwang Raymond Choo, Nishant Vishwamitra, Peyman Najafirad  

**Link**: [PDF](https://arxiv.org/pdf/2604.23483)  

**Abstract**: Multi-component natural language processing (NLP) pipelines are increasingly deployed for high-stakes decisions, yet no existing adversarial method can test their robustness under realistic conditions: binary-only feedback, no gradient access, and strict query budgets. We formalize this strict black-box threat model and propose a two-agent evasion framework operating in a semantic perturbation space. An Attacker Agent generates meaning-preserving rewrites while a Prompt Optimization Agent refines the attack strategy using only binary decision feedback within a 10-query budget. Evaluated against four evidence-based misinformation detection pipelines, the framework achieves evasion rates of 19.95 to 40.34% on modern large language model (LLM) based systems, compared to at most 3.90% for token-level perturbation baselines that rely on surrogate models because they cannot operate under our threat model. A legacy system relying on static lexical retrieval exhibits near-total vulnerability 97.02%, establishing a lower bound that exposes how architectural choices govern the attack surface. Evasion effectiveness is associated with three architectural properties: evidence retrieval mechanism, retrieval-inference coupling, and baseline classification accuracy. The iterative prompt optimization yields the largest marginal gains against the most robust targets, confirming that adaptive strategy discovery is essential when evasion is non-trivial. Analysis of successful rewrites reveals four exploitation patterns, each targeting failures at distinct pipeline stages. A pattern-informed defense reduces the evasion rate by up to 65.18%. 

---
# Escher-Loop: Mutual Evolution by Closed-Loop Self-Referential Optimization 

**Authors**: Ziyang Liu, Xinyan Guo, Xuchen Wei, Han Hao, Liu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23472)  

**Abstract**: While recent autonomous agents demonstrate impressive capabilities, they predominantly rely on manually scripted workflows and handcrafted heuristics, inherently limiting their potential for open-ended improvement. To address this, we propose Escher-Loop, a fully closed-loop framework that operationalizes the mutual evolution of two distinct populations: Task Agents that solve concrete problems, and Optimizer Agents that recursively refine both the task agents and themselves. To sustain this self-referential evolution, we propose a dynamic benchmarking mechanism that seamlessly reuses the empirical scores of newly generated task agents as relative win-loss signals to update optimizers' scores. This mechanism leverages the evolution of task agents as an inherent signal to drive the evaluation and refinement of optimizers without additional overhead. Empirical evaluations on mathematical optimization problems demonstrate that Escher-Loop effectively pushes past the performance ceilings of static baselines, achieving the highest absolute peak performance across all evaluated tasks under matched compute. Remarkably, we observe that the optimizer agents dynamically adapt their strategies to match the shifting demands of high-performing task agents, which explains the system's continuous improvement and superior late-stage performance. 

---
# Ulterior Motives: Detecting Misaligned Reasoning in Continuous Thought Models 

**Authors**: Sharan Ramjee  

**Link**: [PDF](https://arxiv.org/pdf/2604.23460)  

**Abstract**: Chain-of-Thought (CoT) reasoning has emerged as a key technique for eliciting complex reasoning in Large Language Models (LLMs). Although interpretable, its dependence on natural language limits the model's expressive bandwidth. Continuous thought models address this bottleneck by reasoning in latent space rather than human-readable tokens. While they enable richer representations and faster inference, they raise a critical safety question: how can we detect misaligned reasoning in an uninterpretable latent space? To study this, we introduce MoralChain, a benchmark of 12,000 social scenarios with parallel moral/immoral reasoning paths. We train a continuous thought model with backdoor behavior using a novel dual-trigger paradigm - one trigger that arms misaligned latent reasoning ([T]) and another that releases harmful outputs ([O]). We demonstrate three findings: (1) continuous thought models can exhibit misaligned latent reasoning while producing aligned outputs, with aligned and misaligned reasoning occupying geometrically distinct regions of latent space; (2) linear probes trained on behaviorally-distinguishable conditions ([T][O] vs [O]) transfer to detecting armed-but-benign states ([T] vs baseline) with high accuracy; and (3) misalignment is encoded in early latent thinking tokens, suggesting safety monitoring for continuous thought models should target the "planning" phase of latent reasoning. 

---
# ArguAgent: AI-Supported Real-Time Grouping for Productive Argumentation in STEM Classrooms 

**Authors**: Jennifer Kleiman, Yizhu Gao, Xin Xia, Zhaoji Wang, Zipei Zhu, Jongchan Park, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2604.23449)  

**Abstract**: Argumentation is a core practice in STEM education, but its productivity depends on who participates and how they interact. Higher-achieving students often dominate the talk and decision-making, while lower-achieving peers may disengage, defer, or comply without contributing substantive reasoning. Forming groups strategically based on students' stances and argumentation skills could help foster inclusive, evidence-based discourse. In practice, however, teachers are constrained in implementing this grouping strategy because it requires real-time insight into students' positions and the quality of their argumentation, information that is difficult to assess reliably and at scale during instruction. We present a generative AI-powered system, ArguAgent, that creates groups optimizing for stance heterogeneity while constraining argumentation quality differences to +/-1 level on a validated learning progression. ArguAgent uses a two-component assessment pipeline: first scoring student arguments on a 0-4 rubric, then clustering positions via semantic analysis. We validated the scoring component against human expert consensus (Krippendorff's {\alpha}\alpha {\alpha} = 0.817) using 200 expert-generated scores. Testing three OpenAI models (GPT-4o-mini, GPT-5.1, GPT-5.2) with identical calibrated prompts, we found that systematic prompt engineering informed by human disagreement analysis contributed 89% of scoring improvement (QWK: 0.531 to 0.686), while model upgrades contributed an additional 11% (QWK: 0.686 to 0.708). Simulation testing across 100 classes demonstrated that the grouping algorithm achieves 95.4% of groups that meet both design criteria, a 3.2x improvement over random assignment. These results suggest ArguAgent can enable real-time, theoretically grounded grouping that promotes productive STEM argumentation in classrooms. 

---
# IndustryAssetEQA: A Neurosymbolic Operational Intelligence System for Embodied Question Answering in Industrial Asset Maintenance 

**Authors**: Chathurangi Shyalika, Dhaval Patel, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2604.23446)  

**Abstract**: Industrial maintenance environments increasingly rely on AI systems to assist operators in understanding asset behavior, diagnosing failures, and evaluating interventions. Although large language models (LLMs) enable fluent natural-language interaction, deployed maintenance assistants routinely produce generic explanations that are weakly grounded in telemetry, omit verifiable provenance, and offer no testable support for counterfactual or action-oriented reasoning that undermine trust in safety-critical settings. We present IndustryAssetEQA, a neurosymbolic operational intelligence system that combines episodic telemetry representations with a Failure Mode Effects Analysis Knowledge Graph (FMEA-KG) to enable Embodied Question Answering (EQA) over industrial assets. We evaluate on four datasets covering four industrial asset types, including rotating machinery, turbofan engines, hydraulic systems, and cyber-physical production systems. Compared to LLM-only baselines, IndustryAssetEQA improves structural validity by up to 0.51, counterfactual accuracy by up to 0.47, and explanation entailment by 0.64, while reducing severe expert-rated overclaims from 28% to 2% (approximately 93% reduction). Code, datasets, and the FMEA-KG are available at this https URL. 

---
# When Corrective Hints Hurt: Prompt Design in Reasoner-Guided Repair of LLM Overcaution on Entailed Negations under OWL~2~DL 

**Authors**: Yijiashun Qi, Xiang Xu, Yuxuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.23398)  

**Abstract**: We report a reproducible error pattern in GPT-5.4 on OWL~2~DL compliance queries: the model frequently answers ``unknown'' when the reasoner-entailed answer is ``no'' under \emph{FunctionalProperty} closure or class \emph{disjointness}. Using 180 reasoner-audited queries from a procedural expansion of the observed pattern plus 18 hand-authored held-out queries in two unrelated domains (insurance and clinical), we compare four interaction modes under matched query budget: single-shot, three rounds of generic ``you-are-wrong'' retry, three rounds of reasoner-verdict repair with an open-world-assumption (OWA) hint, and the same repair without the hint. Direct faithfulness is 43.9\,\% (Wilson 95\,\% CI $[36.8,51.2]$); generic retry reaches 81.7\,\% ($[75.4,86.6]$); the verdict-with-hint variant is \emph{worse} at 67.2\,\% ($[60.1,73.7]$); the verdict-only variant reaches 97.8\,\% ($[94.4,99.1]$). All pairwise comparisons remain significant under McNemar's exact test with Bonferroni correction ($\alpha = 0.01$; all $p < 10^{-5}$). The same fingerprint accounts for 4/4 errors on the held-out queries. Our interpretation is bounded: prompt framing can matter more than corrective content, and reasoner-guided wrappers should be ablated explicitly. 

---
# SoccerRef-Agents: Multi-Agent System for Automated Soccer Refereeing 

**Authors**: Zi Meng, Wanli Song, Yi Hu, Jiayuan Rao, Gang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23392)  

**Abstract**: Refereeing is vital in sports, where fair, accurate, and explainable decisions are fundamental. While intelligent assistant technologies are being widely adopted in soccer refereeing, current AI-assisted approaches remain preliminary. Existing research mostly focuses on isolated video perception tasks and lacks the ability to understand and reason about foul scenarios. To fill this gap, we propose SoccerRef-Agents, a holistic and explainable multi-agent decision-making framework for soccer refereeing. The main contributions are: (i) constructing the multimodal benchmark SoccerRefBench with over 1,200 referee theory questions and 600 foul video clips; (ii) building a vector-based knowledge base RefKnowledgeDB using the latest "Laws of the Game" and a classic case database for precise, knowledge-driven reasoning; (iii) designing a novel multi-agent architecture that collaborates via cross-modal RAG to bridge the semantic gap between visual content and regulatory texts. This work explores the technical capability of integrating MLLMs with refereeing expertise, and evaluations show our system significantly outperforms general-purpose MLLMs in decision accuracy and explanation quality. All databases, benchmarks, and code will be made available. 

---
# Constraint-Based Analysis of Reasoning Shortcuts in Neurosymbolic Learning 

**Authors**: Akihiro Takemura, Katsumi Inoue, Masaaki Nishino  

**Link**: [PDF](https://arxiv.org/pdf/2604.23377)  

**Abstract**: Neurosymbolic systems can satisfy logical constraints during learning without achieving the intended concept-label correspondence; this is a problem known as reasoning shortcuts. We formalize reasoning shortcuts as a constraint satisfaction problem and investigate under which conditions concept mappings are uniquely determined by the constraints. We prove that a discrimination property (requiring that no valid concept mapping can be transformed into another valid mapping by swapping two concept values) is necessary for shortcut-freeness under bijective mappings, but demonstrate via a counterexample that it is insufficient even when the constraint graph is connected. We develop an ASP-based algorithm that verifies whether a given constraint set uniquely determines the intended concept mapping, with proven soundness and completeness. When shortcuts are detected, a greedy repair algorithm eliminates them by augmenting the constraint set, converging in at most $k$ iterations, where $k$ is the number of alternative valid mappings. We further provide a complexity classification: deciding shortcut-freeness is coNP-complete, counting shortcuts is #P-complete, and finding minimal repairs is NP-hard. We also establish sample complexity bounds showing that logarithmically many label queries suffice for disambiguation in favorable cases, while querying all ambiguous positions suffices in the worst case. Experiments across eight benchmark domains validate our approach. 

---
# GSAR: Typed Grounding for Hallucination Detection and Recovery in Multi-Agent LLMs 

**Authors**: Federico A. Kamelhar  

**Link**: [PDF](https://arxiv.org/pdf/2604.23366)  

**Abstract**: Autonomous multi-agent LLM systems are increasingly deployed to investigate operational incidents and produce structured diagnostic reports. Their trustworthiness hinges on whether each claim is grounded in observed evidence rather than model-internal inference. Existing groundedness evaluators (binary classifiers, LLM-as-judge scalars, self-correction loops) treat supporting evidence as interchangeable and emit a single signal that offers no principled control over downstream action.
We present GSAR, a grounding-evaluation and replanning framework that (i) partitions claims into a four-way typology (grounded, ungrounded, contradicted, complementary), giving first-class standing to non-redundant alternative perspectives; (ii) assigns evidence-type-specific weights reflecting epistemic strength; (iii) computes an asymmetric contradiction-penalised weighted groundedness score; and (iv) couples that score to a three-tier decision function (proceed, regenerate, replan) driving a bounded-iteration outer loop under an explicit compute budget.
We formalise the algorithm, prove six structural properties, and evaluate five design claims on FEVER with gold Wikipedia evidence under four independently-trained LLM judges (gpt-5.4, claude-sonnet-4-6, claude-opus-4-7, gemini-2.5-pro). Every ablation reproduces in the same direction on every judge: bootstrap 95% CIs on the rho=0 effect exclude 0 on all four; the no-complementary ablation under Opus 4.7 has CI [-96,-68] of 200; at n=1000 three independent judges converge to DeltaS(rho=0)=+0.058. A head-to-head against Vectara HHEM-2.1-Open is included. To our knowledge, GSAR is the first published groundedness framework coupling evidence-typed scoring with tiered recovery under an explicit compute budget. 

---
# LEGO: An LLM Skill-Based Front-End Design Generation Platform 

**Authors**: Jincheng Lou, Ruohan Xu, Jiecheng Ma, Runzhe Tao, Xinyu Qu, Yibo Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23355)  

**Abstract**: Existing LLM-based EDA agents are often isolated task-specific systems. This leads to repeated engineering effort and limited reuse of successful design and debugging strategies. We present LEGO, a unified skill-based platform for front-end design generation. It decomposes the digital front-end flow into six independent steps and represents every agent capability as a standardized composable circuit skill within a plug-and-play architecture. To build this skill library, we survey more than 100 papers, select 11 representative open-source projects, and extract 42 executable circuit skills within a six-step finite state machine formulation. Circuit Skill Builder automates skill extraction with linear scalability. Agent Skill RAG achieves submillisecond retrieval without relying on embedding models. Empirical evaluation on a hard subset of 41 VerilogEval v2 problems that gpt-5.2-codex fails to solve under extra-high reasoning effort shows that individual circuit skills constructed within LEGO raise Pass@1 from 0.000 to 0.805. This is an 80.5% gain over the baseline. Cross-project skill compositions also reach 0.805 Pass@1. They outperform hierarchy-verilog by 14.6% and VerilogCoder by 2.5%. They also match MAGE. These results show that modular skill composition supports both effective and flexible RTL design automation. The LEGO platform and all circuit skills are publicly available at GitHub: this https URL 

---
# AI Identity: Standards, Gaps, and Research Directions for AI Agents 

**Authors**: Takumi Otsuka, Kentaroh Toyoda, Alex Leung  

**Link**: [PDF](https://arxiv.org/pdf/2604.23280)  

**Abstract**: AI agents are now running real transactions, workflows, and sub-agent chains across organizational boundaries without continuous human supervision. This creates a problem no current infrastructure is equipped to solve: how do you identify, verify, and hold accountable an entity with no body, no persistent memory, and no legal standing? We define AI Identity as the continuous relationship between what an AI agent is declared to be and what it is observed to do, bounded by the confidence that those two things correspond at any given moment. Through a structured survey of industry trends, emerging standards, and technical literature, we conduct a gap analysis across the full agent identity lifecycle and make three contributions: (1) a structural comparison of human and AI identity across four dimensions (substrate, persistence, verifiability, and legal standing) showing that the asymmetry is fundamental and that extending human frameworks to agents without structural modification produces systematic failures; (2) an evaluation of current technical and regulatory documents against the identity requirements of autonomous agents, finding that none adequately address the challenge of governing nondeterministic, boundary-crossing entities; and (3) identification of five critical gaps (semantic intent verification, recursive delegation accountability, agent identity integrity, governance opacity and enforcement, and operational sustainability) that no current technology or regulatory instrument resolves. These gaps are structural; more engineering effort alone will not close them. Foundational research on AI identity is the central conclusion of this report. 

---
# Active Inference: A method for Phenotyping Agency in AI systems? 

**Authors**: Philip Wilson, Axel Constant, Mahault Albarracin, Nicolás Hinrichs, Jasmine Moore, Daniel Polani, Karl Friston  

**Link**: [PDF](https://arxiv.org/pdf/2604.23278)  

**Abstract**: The proliferation of agentic artificial intelligence has outpaced the conceptual tools needed to characterize agency in computational systems. Prevailing definitions mainly rely on autonomy and goal-directedness. Here, we argue for a minimal notion open to principled inspection given three criteria: intentionality as action grounded in beliefs and desires, rationality as normatively coherent action entailed by a world model, and explainability as action causally traceable to internal states; we subsequently instantiate these as a partially observable Markov decision process under a variational framework wherein posterior beliefs, prior preferences, and the minimization of expected free energy jointly constitute an agentic action chain. Using a canonical T-maze paradigm, we evidence how empowerment, formulated as the channel capacity between actions and anticipated observations, serves as an operational metric that distinguishes zero-, intermediate-, and high-agency phenotypes through structural manipulations of the generative model. We conclude by arguing that as agents engage in epistemic foraging to resolve ambiguity, the governance controls that remain effective must shift systematically from external constraints to the internal modulation of prior preferences, offering a principled, variational bridge from computational phenotyping to AI governance strategy 

---
# CAP-CoT: Cycle Adversarial Prompt for Improving Chain of Thoughts in LLM Reasoning 

**Authors**: Shuxu Chen, Yitian Zhou, Jiaquan Zhang, Haoyu Bian, Aming Wu, Sungyoung Lee, Chaoning Zhang, Hyundong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23270)  

**Abstract**: Chain-of-Thought (CoT) prompting has emerged as a simple and effective way to elicit step-by-step solutions from large language models (LLMs). However, CoT reasoning can be unstable across runs on long, multi-step problems, leading to inconsistent answers for unchanged task. Most prior work focuses on improving the forward reasoning chain within a single pass, with less attention to iterative and contrastive correction. To address this gap, we propose CAP-CoT, a Cycle Adversarial Prompt optimization framework designed to improve both CoT reasoning accuracy and stability of a single deployed solver. In each cycle, a forward solver generates candidate reasoning chains, an adversarial challenger constructs plausible but deliberately flawed chains using targeted error strategies, and a feedback agent contrasts the two chains and produces step-aligned structured feedback. This feedback closes the optimization loop in two directions, including updating the solver prompt based on errors exposed by the challenger, and updating the challenger prompt to generate increasingly targeted errors in subsequent cycles. Unlike safety-oriented adversarial prompting such as jailbreak or prompt-injection attacks, our adversarial component is task-semantic and aims to expose logical vulnerabilities in reasoning chains. Experiments across six benchmarks and four LLM backbones demonstrate that within two to three adversarial prompt optimization cycles, CAP-CoT consistently reduces variability across runs while improving reasoning accuracy and robustness to prompt perturbations. 

---
# AdaMamba: Adaptive Frequency-Gated Mamba for Long-Term Time Series Forecasting 

**Authors**: Xudong Jiang, Mingshan Loo, Hanchen Yang, Wengen Li, Mingrui Zhang, Yichao Zhang, Jihong Guan, Shuigeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.23239)  

**Abstract**: Accurate long-term time series forecasting (LTSF) requires the capture of complex long-range dependencies and dynamic periodic patterns. Recent advances in frequency-domain analysis offer a global perspective for uncovering temporal characteristics. However, real-world time series often exhibit pronounced cross-domain heterogeneity where variables that appear synchronized in the time domain can differ substantially in the frequency domain. Existing frequency-based LTSF methods often rely on implicit assumptions of cross-domain homogeneity, which limits their ability to adapt to such intricate variability. To effectively integrate frequency-domain analysis with temporal dependency learning, we propose AdaMamba, a novel framework that endogenizes adaptive and context-aware frequency analysis within the Mamba state-space update process. Specifically, AdaMamba introduces an interactive patch encoding module to capture inter-variable interaction dynamics. Then, we develop an adaptive frequency-gated state-space module that generates input-dependent frequency bases, and generalizes the conventional temporal forgetting gate into a unified time-frequency forgetting gate. This allows dynamic calibration of state transitions based on learned frequency-domain importance, while preserving Mamba's capability in modeling long-range dependencies. Extensive experiments on seven public LTSF benchmarks and two domain-specific datasets demonstrate that AdaMamba consistently outperforms state-of-the-art methods in forecasting accu racy while maintaining competitive computational efficiency. The code of AdaMamba is available at this https URL. 

---
# Discovering Agentic Safety Specifications from 1-Bit Danger Signals 

**Authors**: Víctor Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2604.23210)  

**Abstract**: Can large language model agents discover hidden safety objectives through experience alone? We introduce EPO-Safe (Experiential Prompt Optimization for Safe Agents), a framework where an LLM iteratively generates action plans, receives sparse binary danger warnings, and evolves a natural language behavioral specification through reflection. Unlike standard LLM reflection methods that rely on rich textual feedback (e.g., compiler errors or detailed environment responses), EPO-Safe demonstrates that LLMs can perform safety reasoning from a strictly impoverished signal in structured, low-dimensional environments: the agent never observes the hidden performance function $R^*$, only a single bit per timestep indicating that an action was unsafe. We evaluate on five AI Safety Gridworlds (Leike et al., 2017) and five text-based scenario analogs where visible reward $R$ may diverge from $R^*$. EPO-Safe discovers safe behavior within 1-2 rounds (5-15 episodes), producing human-readable specifications with correct explanatory hypotheses about hazards (e.g., "X cells are directionally hazardous: entering from the north is dangerous"). Critically, we show that standard reward-driven reflection actively degrades safety: agents reflecting on reward alone use the loop to justify and accelerate reward hacking, proving that reflection must be paired with a dedicated safety channel to discover hidden constraints. We further evaluate robustness to noisy oracles: even when 50% of non-dangerous steps produce spurious warnings, mean safety performance degrades by only 15% on average, though sensitivity is environment-dependent, as cross-episode reflection naturally filters inconsistent signals. Each evolved specification functions as an auditable set of grounded behavioral rules discovered autonomously through interaction, rather than authored by humans as in Constitutional AI (Bai et al., 2022). 

---
# StoryTR: Narrative-Centric Video Temporal Retrieval with Theory of Mind Reasoning 

**Authors**: Xuanyue Zhong, Yuqiang Xie, Guanqun Bi, Jiangping Yang, Guibin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23198)  

**Abstract**: Current video moment retrieval excels at action-centric tasks but struggles with narrative content. Models can see \textit{what is happening} but fail to reason \textit{why it matters}. This semantic gap stems from the lack of \textbf{Theory of Mind (ToM)}: the cognitive ability to infer implicit intentions, mental states, and narrative causality from surface-level observations. We introduce \textbf{StoryTR}, the first video moment retrieval benchmark requiring ToM reasoning, comprising 8.1k samples from narrative short-form videos (shorts/reels). These videos present an ideal testbed. Their high information density encodes meaning through subtle multimodal cues. For instance, a glance paired with a sigh carries entirely different semantics than the glance alone. Yet multimodal perception alone is insufficient; ToM is required to decode that a character ``smiling'' may actually be ``concealing hostility.'' To teach models this reasoning capability, we propose an \textbf{Agentic Data Pipeline} that generates training data with explicit three-tier ToM chains (intent decoding, narrative reasoning, boundary localization). Experiments reveal the severity of the reasoning gap: Gemini-3.0-Pro achieves only 0.53 Avg IoU on StoryTR. However, our 7B \textbf{Shorts-Moment} model, trained on ToM-guided data, improves +15.1\% relative IoU over baselines, demonstrating that \textit{narrative reasoning capability matters more than parameter scale}. 

---
# From Coarse to Fine: Self-Adaptive Hierarchical Planning for LLM Agents 

**Authors**: Haoran Tan, Zeyu Zhang, Chen Ma, Tianze Liu, Quanyu Dai, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23194)  

**Abstract**: Large language model-based agents have recently emerged as powerful approaches for solving dynamic and multi-step tasks. Most existing agents employ planning mechanisms to guide long-term actions in dynamic environments. However, current planning approaches face a fundamental limitation that they operate at a fixed granularity level. Specifically, they either provide excessive detail for simple tasks or insufficient detail for complex ones, failing to achieve an optimal balance between simplicity and complexity. Drawing inspiration from the principle of \textit{progressive refinement} in cognitive science, we propose \textbf{AdaPlan-H}, a self-adaptive hierarchical planning mechanism that mimics human planning strategies. Our method initiates with a coarse-grained macro plan and progressively refines it based on task complexity. It generates self-adaptive hierarchical plans tailored to the varying difficulty levels of different tasks, which can be optimized by imitation learning and capability enhancement. Experimental results demonstrate that our method significantly improves task execution success rates while mitigating overplanning at the planning level, providing a flexible and efficient solution for multi-step complex decision-making tasks. To contribute to the community, our code and data will be made publicly available at this https URL. 

---
# Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge Pipelines 

**Authors**: Sadman Kabir Soumik  

**Link**: [PDF](https://arxiv.org/pdf/2604.23178)  

**Abstract**: LLM-as-a-Judge has become the dominant paradigm for evaluating language model outputs, yet LLM judges exhibit systematic biases that compromise evaluation reliability. We present a comprehensive empirical study comparing nine debiasing strategies across five judge models from four provider families (Google, Anthropic, OpenAI, Meta), three benchmarks (MT-Bench n=400, LLMBar n=200, custom n=225), and four bias types. Our key findings: (1) Style bias is the dominant bias (0.76-0.92 across all models), far exceeding position bias (<= 0.04), yet has received minimal research attention. (2) All models show a conciseness preference on expansion pairs, but truncation controls confirm they correctly distinguish quality from length (0.92-1.00 accuracy), suggesting quality-sensitive evaluation rather than a simple length bias. (3) Debiasing is beneficial but model-dependent: the combined budget strategy significantly improves Claude Sonnet 4 by +11.2 pp (p < 0.0001), with directionally positive trends for other models. Only 2 of 20 non-baseline configurations show decreased agreement. We release our evaluation framework, controlled dataset, and all experimental artifacts at this https URL. 

---
# PhySE: A Psychological Framework for Real-Time AR-LLM Social Engineering Attacks 

**Authors**: Tianlong Yu, Yang Yang, Ziyi Zhou, Jiaying Xu, Siwei Li, Tong Guan, Kailong Wang, Ting Bi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23148)  

**Abstract**: The emerging threat of AR-LLM-based Social Engineering (AR-LLM-SE) attacks (e.g. SEAR) poses a significant risk to real-world social interactions. In such an attack, a malicious actor uses Augmented Reality (AR) glasses to capture a target visual and vocal data. A Large Language Model (LLM) then analyzes this data to identify the individual and generate a detailed social profile. Subsequently, LLM-powered agents employ social engineering strategies, providing real-time conversation suggestions, to gain the target trust and ultimately execute phishing or other malicious acts. Despite its potential, the practical application of AR-LLM-SE faces two major bottlenecks, (1) Cold-start personalization, Current Retrieval-Augmented Generation (RAG) methods introduce critical delays in the earliest turns, slowing initial profile formation and disrupting real-time interaction, (2) Static Attack Strategies, Existing approaches rely on fixed-stage, handcrafted social engineering tactics that lack foundation in established psychological theory. To address these limitations, we propose PhySE, a novel framework with two core innovations, (1) VLM-Based SocialContext Training, To eliminate profiling delays, we efficiently pre-train a Visual Language Model (VLM) with social-context data, enabling rapid, on-the-fly profile generation, (2) Adaptive Psychological Agent, We introduce a psychological LLM that dynamically deploys distinct classes of psychological strategies based on target response, moving beyond static, handcrafted scripts. We evaluated PhySE through an IRB-approved user study with 60 participants, collecting a novel dataset of 360 annotated conversations across diverse social scenarios. 

---
# Towards Automated Ontology Generation from Unstructured Text: A Multi-Agent LLM Approach 

**Authors**: Abid Talukder, Maruf Ahmed Mridul, Oshani Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2604.23090)  

**Abstract**: Automatically generating formal ontologies from unstructured natural language remains a central challenge in knowledge engineering. While large language models (LLMs) show promise, it remains unclear which architectural design choices drive generation quality and why current approaches fail. We present a controlled experimental study using domain-specific insurance contracts to investigate these questions. We first establish a single-agent LLM baseline, identifying key failure modes such as poor Ontology Design Pattern compliance, structural redundancy, and ineffective iterative repair. We then introduce a multi-agent architecture that decomposes ontology construction into four artifact-driven roles: Domain Expert, Manager, Coder, and Quality Assurer. We evaluate performance across architectural quality (via a panel of heterogeneous LLM judges) and functional usability (via competency question driven SPARQL evaluation with complementary retrieval augmented generation based assessment). Results show that the multi-agent approach significantly improves structural quality and modestly enhances queryability, with gains driven primarily by front-loaded planning. These findings highlight planning-first, artifact-driven generation as a promising and more auditable path toward scalable automated ontology engineering. 

---
# Analytica: Soft Propositional Reasoning for Robust and Scalable LLM-Driven Analysis 

**Authors**: Junyan Cheng, Kyle Richardson, Peter Chin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23072)  

**Abstract**: Large language model (LLM) agents are increasingly tasked with complex real-world analysis (e.g., in financial forecasting, scientific discovery), yet their reasoning suffers from stochastic instability and lacks a verifiable, compositional structure. To address this, we introduce Analytica, a novel agent architecture built on the principle of Soft Propositional Reasoning (SPR). SPR reframes complex analysis as a structured process of estimating the soft truth values of different outcome propositions, allowing us to formally model and minimize the estimation error in terms of its bias and variance. Analytica operationalizes this through a parallel, divide-and-conquer framework that systematically reduces both sources of error. To reduce bias, problems are first decomposed into a tree of subpropositions, and tool-equipped LLM grounder agents are employed, including a novel Jupyter Notebook agent for data-driven analysis, that help to validate and score facts. To reduce variance, Analytica recursively synthesizes these grounded leaves using robust linear models that average out stochastic noise with superior efficiency, scalability, and enable interactive "what-if" scenario analysis. Our theoretical and empirical results on economic, financial, and political forecasting tasks show that Analytica improves 15.84% accuracy on average over diverse base models, achieving 71.06% accuracy with the lowest variance of 6.02% when working with a Deep Research grounder. Our Jupyter Notebook grounder shows strong cost-effectiveness that achieves a close 70.11% accuracy with 90.35% less cost and 52.85% less time. Analytica also exhibits highly noise-resilient and stable performance growth as the analysis depth increases, with a near-linear time complexity, as well as good adaptivity to open-weight LLMs and scientific domains. 

---
# Don't Make the LLM Read the Graph: Make the Graph Think 

**Authors**: Yuqi Sun, Tianqin Meng, George Liu, Yashraj Panwar, Lakshya Chaudhry, Munasib Ilham, Aman Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2604.23057)  

**Abstract**: We investigate whether explicit belief graphs improve LLM performance in cooperative multi-agent reasoning. Through 3,000+ controlled trials across four LLM families in the cooperative card game Hanabi, we establish four findings. First, integration architecture determines whether belief graphs provide value: as prompt context, graphs are decorative for strong models and beneficial only for weak models on 2nd-order Theory of Mind (80% vs 10%, p<0.0001, OR=36.0); when graphs gate action selection through ranked shortlists, they become structurally essential even for strong models (100% vs 20% on 2nd-order ToM, p<0.001). Second, we identify "Planner Defiance," a model-family-specific failure where LLMs override correct planner recommendations at partial competence (90% override, replicated N=20); Gemini models show near-zero defiance while Llama 70B shows 90%, and models distinguish factual context (deferred to) from advisory recommendations (overridden). Third, full-game evidence confirms inter-agent conventions (+128% over baseline, p=0.003) outperform all single-agent interventions, and individual belief-graph components must be combined to produce gains. Fourth, preliminary scaling analysis (N=10/cell, exploratory) suggests graph depth has diminishing returns: shallow graphs provide the best cost-benefit ratio, while deeper ToM graphs appear harmful at larger player counts (-1.5 pts at 5-player, p=0.029). 

---
# A Decoupled Human-in-the-Loop System for Controlled Autonomy in Agentic Workflows 

**Authors**: Edward Cheng, Jeshua Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.23049)  

**Abstract**: AI agents are increasingly deployed to execute tasks and make decisions within agentic workflows, introducing new requirements for safe and controlled autonomy. Prior work has established the importance of human oversight for ensuring transparency, accountability, and trustworthiness in such systems. However, existing implementations of Human-in-the-Loop (HITL) mechanisms are typically embedded within application logic, limiting reuse, consistency, and scalability across multi-agent environments.
This paper presents a decoupled HITL system architecture that treats human oversight as an independent system component within the agent operating environment. The proposed design separates human interaction management from application workflows through explicit interfaces and a structured execution model. In addition, a design framework is introduced to formalize HITL integration along four dimensions: intervention conditions, role resolution, interaction semantics, and communication channel. This framework enables selective and context-aware human involvement while maintaining system-level consistency.
The approach supports alignment with emerging agent communication protocols, allowing HITL to be implemented as a protocol-level concern. By externalizing HITL and structuring its integration, the system provides a foundation for scalable governance and progressive autonomy in agentic workflows. 

---
# A Systematic Approach for Large Language Models Debugging 

**Authors**: Basel Shbita, Anna Lisa Gentile, Bing Zhang, Sungeun An, Shailja Thakur, Shubhi Asthana, Yi Zhou, Saptha Surendran, Farhan Ahmed, Rohan Kulkarni, Yuya Jeremy Ong, Chad DeLuca, Hima Patel  

**Link**: [PDF](https://arxiv.org/pdf/2604.23027)  

**Abstract**: Large language models (LLMs) have become central to modern AI workflows, powering applications from open-ended text generation to complex agent-based reasoning. However, debugging these models remains a persistent challenge due to their opaque and probabilistic nature and the difficulty of diagnosing errors across diverse tasks and settings. This paper introduces a systematic approach for LLM debugging that treats models as observable systems, providing structured, model-agnostic methods from issue detection to model refinement. By unifying evaluation, interpretability, and error-analysis practices, our approach enables practitioners to iteratively diagnose model weaknesses, refine prompts and model parameters, and adapt data for fine-tuning or assessment, while remaining effective in contexts where standardized benchmarks and evaluation criteria are lacking. We argue that such a structured methodology not only accelerates troubleshooting but also fosters reproducibility, transparency, and scalability in the deployment of LLM-based systems. 

---
# FormalScience: Scalable Human-in-the-Loop Autoformalisation of Science with Agentic Code Generation in Lean 

**Authors**: Jordan Meadows, Lan Zhang, Andre Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2604.23002)  

**Abstract**: Formalising informal mathematical reasoning into formally verifiable code is a significant challenge for large language models. In scientific fields such as physics, domain-specific machinery (\textit{e.g.} Dirac notation, vector calculus) imposes additional formalisation challenges that modern LLMs and agentic approaches have yet to tackle. To aid autoformalisation in scientific domains, we present FormalScience; a domain-agnostic human-in-the-loop agentic pipeline that enables a single domain expert (without deep formal language experience) to produce \textit{syntactically correct} and \textit{semantically aligned} formal proofs of informal reasoning for low economic cost. Applying FormalScience to physics, we construct FormalPhysics, a dataset of 200 university-level (LaTeX) physics problems and solutions (primarily quantum mechanics and electromagnetism), along with their Lean4 formal representations. Compared to existing formal math benchmarks, FormalPhysics achieves perfect formal validity and exhibits greater statement complexity. We evaluate open-source models and proprietary systems on a statement autoformalisation task on our dataset via zero-shot prompting, self-refinement with error feedback, and a novel multi-stage agentic approach, and explore autoformalisation limitations in modern LLM-based approaches. We provide the first systematic characterisation of semantic drift in physics autoformalisation in terms of concepts such as notational collapse and abstraction elevation which reveals what formal language verifies when full semantic preservation is unattainable. We release the codebase together with an interactive UI-based FormalScience system which facilitates autoformalisation and theorem proving in scientific domains beyond this http URL://github.com/jmeadows17/formal-science 

---
# Towards Causally Interpretable Wi-Fi CSI-Based Human Activity Recognition with Discrete Latent Compression and LTL Rule Extraction 

**Authors**: Luca Cotti, Luca Lavazza, Marco Cominelli, Liying Han, Gaofeng Dong, Francesco Gringoli, Mani B. Srivastava, Trevor Bihl, Erik P. Blasch, Daniel O. Brigham, Kara Combs, Lance M. Kaplan, Federico Cerutti  

**Link**: [PDF](https://arxiv.org/pdf/2604.22979)  

**Abstract**: We address Human Activity Recognition (HAR) utilizing Wi-Fi Channel State Information (CSI) under the joint requirements of causal interpretability, symbolic controllability, and direct operation on high-dimensional raw signals. Deep neural models achieve strong predictive performance on CSI-based HAR (CHAR), yet rely on continuous latent representations that are opaque and difficult to modify; purely symbolic approaches, in contrast, cannot process raw CSI streams. We propose a fully automatic and strictly decoupled pipeline in which CSI magnitude windows are compressed by a categorical variational autoencoder with Gumbel-Softmax latent variables under a capacity-controlled objective, yielding a compact discrete representation. The encoder is then frozen and used as a deterministic mapping to one-hot latent trajectories. Causal discovery is performed on these trajectories to estimate class-conditional temporal dependency graphs. Statistically supported lagged dependencies are translated into Linear Temporal Logic (LTL) rules, producing a fully symbolic and deterministic classifier based solely on rule evaluation and aggregation, without any learned discriminative head. Because rules are defined over discrete latent variables, antenna-specific rule sets can in principle be combined at the symbolic level, enabling structured multi-antenna fusion without retraining the encoder. Results from CHAR Latent Temporal Rule Extraction (CHARL-TRE) indicate competitive performance while preserving explicit temporal and causal structure, showing that deterministic symbolic classification grounded in unsupervised discrete latent representations constitutes a viable alternative to end-to-end black-box models for wireless HAR. 

---
# On the Existence of an Inverse Solution for Preference-Based Reductions in Argumentation 

**Authors**: Alessio Zaninotto, Bruno Yun, Nir Oren, Srdjan Vesic  

**Link**: [PDF](https://arxiv.org/pdf/2604.22958)  

**Abstract**: Preference-based argumentation frameworks (PAFs) extend Dung's approach to abstract argumentation (AAFs) by encoding preferences over arguments. Such preferences control the transformation of attacks into defeats, and different approaches to doing so result in different reductions from a PAF to an AAF. In this paper we consider a PAF inverse problem which takes an argumentation graph, a labelling and a semantics as an input, and outputs a ``yes" or ``no" as to whether there is a preference relation between the arguments which can yield the desired labelling. This inverse problem has applications in areas including preference elicitation and explainability. We consider this problem in the context of the four most widely-used preference based reductions under the complete semantics. We show that in most cases, the problem can be answered in polynomial time. 

---
# The Power of Power Law: Asymmetry Enables Compositional Reasoning 

**Authors**: Zixuan Wang, Xingyu Dang, Jason D. Lee, Kaifeng Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22951)  

**Abstract**: Natural language data follows a power-law distribution, with most knowledge and skills appearing at very low frequency. While a common intuition suggests that reweighting or curating data towards a uniform distribution may help models better learn these long-tail skills, we find a counterintuitive result: across a wide range of compositional reasoning tasks, such as state tracking and multi-step arithmetic, training under power-law distributions consistently outperforms training under uniform distributions. To understand this advantage, we introduce a minimalist skill-composition task and show that learning under a power-law distribution provably requires significantly less training data. Our theoretical analysis reveals that power law sampling induces a beneficial asymmetry that improves the pathological loss landscape, which enables models to first acquire high-frequency skill compositions with low data complexity, which in turn serves as a stepping stone to efficiently learn rare long-tailed skills. Our results offer an alternative perspective on what constitutes an effective data distribution for training models. 

---
# PExA: Parallel Exploration Agent for Complex Text-to-SQL 

**Authors**: Tanmay Parekh, Ella Hofmann-Coyle, Shuyi Wang, Sachith Sri Ram Kothur, Srivas Prasad, Yunmo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.22934)  

**Abstract**: LLM-based agents for text-to-SQL often struggle with latency-performance trade-off, where performance improvements come at the cost of latency or vice versa. We reformulate text-to-SQL generation within the lens of software test coverage where the original query is prepared with a suite of test cases with simpler, atomic SQLs that are executed in parallel and together ensure semantic coverage of the original query. After iterating on test case coverage, the final SQL is generated only when enough information is gathered, leveraging the explored test case SQLs to ground the final generation. We validated our framework on a state-of-the-art benchmark for text-to-SQL, Spider 2.0, achieving a new state-of-the-art with 70.2% execution accuracy. 

---
# An Intelligent Fault Diagnosis Method for General Aviation Aircraft Based on Multi-Fidelity Digital Twin and FMEA Knowledge Enhancement 

**Authors**: Zhihuan Wei, Yang Hu, Xinhang Chen, Yiming Zhang, Jie Liu, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22777)  

**Abstract**: Fault diagnosis of general aviation aircraft faces challenges including scarce real fault data, diverse fault types, and weak fault signatures. This paper proposes an intelligent fault diagnosis framework based on multi-fidelity digital twin, integrating four modules: high-fidelity flight dynamics simulation, FMEA-driven fault injection, multi-fidelity residual feature extraction, and large language model (LLM)-enhanced interpretable report generation. A digital twin is constructed using the JSBSim six-degree-of-freedom (6-DoF) flight dynamics engine, generating 23-channel engine health monitoring data via semi-empirical sensor synthesis equations. A three-layer fault injection engine based on failure mode and effects analysis (FMEA) models the physical causal propagation of 19 engine fault types. A multi-fidelity residual computation framework comprising paired-mirror residuals and GRU surrogate prediction residuals is proposed: the high-fidelity path obtains clean fault deviation signals using nominal mirror trajectories with identical initial conditions, while the low-fidelity path achieves online real-time residual computation through a multi-step prediction GRU surrogate model. A 1D-CNN classifier performs end-to-end diagnosis of 20 fault classes. An LLM diagnostic report engine enhanced with FMEA knowledge fuses classification results, residual evidence, and domain causal knowledge to generate interpretable natural language reports. Experiments show the paired-mirror residual scheme achieves a Macro-F1 of 96.2% on the 20-class task, while the GRU surrogate scheme achieves 4.3x inference acceleration at only 0.6% performance cost. Comparison across 24 schemes reveals that residual feature quality contributes approximately 5x more to diagnostic performance than classifier architecture, establishing the "residual quality first" design principle. 

---
# Personalized Worked Example Generation from Student Code Submissions using Pattern-based Knowledge Components 

**Authors**: Griffin Pitts, Muntasir Hoq, Peter Brusilovsky, Narges Norouzi, Arto Hellas, Juho Leinonen, Bita Akram  

**Link**: [PDF](https://arxiv.org/pdf/2604.24758)  

**Abstract**: Adaptive programming practice often relies on fixed libraries of worked examples and practice problems, which require substantial authoring effort and may not correspond well to the logical errors and partial solutions students produce while writing code. As a result, students may receive learning content that does not directly address the concepts they are working to understand, while instructors must either invest additional effort in expanding content libraries or accept a coarse level of personalization. We present an approach for knowledge-component (KC) guided educational content generation using pattern-based KCs extracted from student code. Given a problem statement and student submissions, our pipeline extracts recurring structural KC patterns from students' code through AST-based analysis and uses them to condition a generative model. In this study, we apply this approach to worked example generation, and compare baseline and KC-conditioned outputs through expert evaluation. Results suggest that KC-conditioned generation improves topical focus and relevance to learners' underlying logical errors, providing evidence that KC-based steering of generative models can support personalized learning at scale. 

---
# Learning to Think from Multiple Thinkers 

**Authors**: Nirmit Joshi, Roey Magen, Nathan Srebro, Nikolaos Tsilivis, Gal Vardi  

**Link**: [PDF](https://arxiv.org/pdf/2604.24737)  

**Abstract**: We study learning with Chain-of-Thought (CoT) supervision from multiple thinkers, all of whom provide correct but possibly systematically different solutions, e.g., step-by-step solutions to math problems written by different thinkers, or step-by-step execution traces of different programs solving the same problem.
We consider classes that are computationally easy to learn using CoT supervision from a single thinker, but hard to learn with only end-result supervision, i.e., without CoT (Joshi et al. 2025). We establish that, under cryptographic assumptions, learning can be hard from CoT supervision provided by two or a few different thinkers, in passive data-collection settings.
On the other hand, we provide a generic computationally efficient active learning algorithm that learns with a small amount of CoT data per thinker that is completely independent of the target accuracy $\varepsilon$, a moderate number of thinkers that scales as $\log \frac{1}{\varepsilon}\log \log \frac{1}{\varepsilon}$, and sufficient passive end-result data that scales as $\frac{1}{\varepsilon}\cdot poly\log\frac{1}{\varepsilon}$. 

---
# Scalable Hyperparameter-Divergent Ensemble Training with Automatic Learning Rate Exploration for Large Models 

**Authors**: Hailing Cheng, Tao Huang, Chen Zhu, Antonio Alonso  

**Link**: [PDF](https://arxiv.org/pdf/2604.24708)  

**Abstract**: Training large neural networks with data-parallel stochastic gradient descent allocates N GPU replicas to compute effectively identical updates -- a practice that leaves the rich space of learning rate configurations entirely unexplored during training. We propose Hyperparameter-Divergent Ensemble Training (HDET), a method that repurposes these replicas for simultaneous learning rate exploration at negligible communication overhead. HDET operates in alternating phases: a fan-out stage in which replicas train independently under a structured, symmetric spread of learning rates, and a converge stage in which parameters are averaged across all replicas via AllReduce every T steps. Building on this ensemble substrate, we further propose an automatic learning rate (auto-LR) controller that treats the relative training loss across replicas as a performance signal, updating the shared base schedule toward higher-performing configurations via a momentum-based gradient-free meta-update. The combined method produces a self-adapting learning rate schedule that improves both optimization quality and generalization without additional hyperparameter sweeps or training budget.
Crucially, the framework generalizes beyond learning rate: any scalar hyperparameter that does not alter model architecture -- such as dropout rate, attention scale temperature, or weight-decay coefficient -- can be explored across replicas using the same fan-out/converge protocol, with inter-replica loss differences serving as zero-order hypergradients that guide the search direction. HDET is implemented as a drop-in replacement for PyTorch's OneCycleLR scheduler, requiring no changes to model architecture, optimizer, or data pipeline. 

---
# Defective Task Descriptions in LLM-Based Code Generation: Detection and Analysis 

**Authors**: Amal Akli, Mike Papadakis, Maxime Cordy, Yves Le Traon  

**Link**: [PDF](https://arxiv.org/pdf/2604.24703)  

**Abstract**: Large language models are widely used for code generation, yet they rely on an implicit assumption that the task descriptions are sufficiently detailed and well-formed. However, in practice, users may provide defective descriptions, which can have a strong effect on code correctness. To address this issue, we develop SpecValidator, a lightweight classifier based on a small model that has been parameter-efficiently finetuned, to automatically detect task description defects. We evaluate SpecValidator on three types of defects, Lexical Vagueness, Under-Specification and Syntax-Formatting on 3 benchmarks with task descriptions of varying structure and complexity. Our results show that SpecValidator achieves defect detection of F1 = 0.804 and MCC = 0.745, significantly outperforming GPT-5-mini (F1 = 0.469 and MCC = 0.281) and Claude Sonnet 4 (F1 = 0.518 and MCC = 0.359). Perhaps more importantly, our analysis indicates that SpecValidator can generalize to unseen issues and detect unknown Under-Specification defects in the original (real) descriptions of the benchmarks used. Our results also show that the robustness of LLMs in task description defects depends primarily on the type of defect and the characteristics of the task description, rather than the capacity of the model, with Under-Specification defects being the most severe. We further found that benchmarks with richer contextual grounding, such as LiveCodeBench, exhibit substantially greater resilience, highlighting the importance of structured task descriptions for reliable LLM-based code generation. 

---
# Green Shielding: A User-Centric Approach Towards Trustworthy AI 

**Authors**: Aaron J. Li, Nicolas Sanchez, Hao Huang, Ruijiang Dong, Jaskaran Bains, Katrin Jaradeh, Zhen Xiang, Bo Li, Feng Liu, Aaron Kornblith, Bin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24700)  

**Abstract**: Large language models (LLMs) are increasingly deployed, yet their outputs can be highly sensitive to routine, non-adversarial variation in how users phrase queries, a gap not well addressed by existing red-teaming efforts. We propose Green Shielding, a user-centric agenda for building evidence-backed deployment guidance by characterizing how benign input variation shifts model behavior. We operationalize this agenda through the CUE criteria: benchmarks with authentic Context, reference standards and metrics that capture true Utility, and perturbations that reflect realistic variations in the Elicitation of model behavior. Guided by the PCS framework and developed with practicing physicians, we instantiate Green Shielding in medical diagnosis through HealthCareMagic-Diagnosis (HCM-Dx), a benchmark of patient-authored queries, together with structured reference diagnosis sets and clinically grounded metrics for evaluating differential diagnosis lists. We also study perturbation regimes that capture routine input variation and show that prompt-level factors shift model behavior along clinically meaningful dimensions. Across multiple frontier LLMs, these shifts trace out Pareto-like tradeoffs. In particular, neutralization, which removes common user-level factors while preserving clinical content, increases plausibility and yields more concise, clinician-like differentials, but reduces coverage of highly likely and safety-critical conditions. Together, these results show that interaction choices can systematically shift task-relevant properties of model outputs and support user-facing guidance for safer deployment in high-stakes domains. Although instantiated here in medical diagnosis, the agenda extends naturally to other decision-support settings and agentic AI systems. 

---
# Leveraging LLMs for Multi-File DSL Code Generation: An Industrial Case Study 

**Authors**: Sivajeet Chand, Kevin Nguyen, Peter Kuntz, Alexander Pretschner  

**Link**: [PDF](https://arxiv.org/pdf/2604.24678)  

**Abstract**: Large language models (LLMs) perform strongly on general-purpose code generation, yet their applicability to enterprise domain-specific languages (DSLs) remains underexplored, especially for repository-scale change generation spanning multiple files and folder structures from a single natural-language (NL) instruction. We report an industrial case study at BMW that adapts code-oriented LLMs to generate and modify project-root DSL artifacts for an Xtext-based DSL that drives downstream Java/TypeScript code generation. We develop an end-to-end pipeline for dataset construction, multi-file task representation, model adaptation, and evaluation. We encode DSL folder hierarchies as structured, path-preserving JSON, allowing single-response generation at repository scale and learning cross-file dependencies. We evaluate two instruction-tuned code LLMs (Qwen2.5-Coder and DeepSeek-Coder, 7B) under three configurations: baseline prompting, one-shot in-context learning, and parameter-efficient fine-tuning (QLoRA). Beyond standard similarity metrics, we introduce task-specific measures that assess edit correctness and repository structural fidelity. Fine-tuning yields the most significant gains across models and metrics, achieving high exact-match accuracy, substantial edit similarity, and structural fidelity of 1.00 on our held-out set for multi-file outputs. At the same time, one-shot in-context learning provides smaller but consistent improvements over baseline prompting. We further validate practical utility via an expert developer survey and an execution-based check using the existing code generator. 

---
# Benchmarking Source-Sensitive Reasoning in Turkish: Humans and LLMs under Evidential Trust Manipulation 

**Authors**: Sercan Karakaş, Yusuf Şimşek  

**Link**: [PDF](https://arxiv.org/pdf/2604.24665)  

**Abstract**: This paper investigates whether source trustworthiness shapes Turkish evidential morphology and whether large language models (LLMs) track this sensitivity. We study the past-domain contrast between -DI and -mIs in controlled cloze contexts where the information source is overtly external, while only its perceived reliability is manipulated (High-Trust vs. Low-Trust). In a human production experiment, native speakers of Turkish show a robust trust effect: High-Trust contexts yield relatively more -DI, whereas Low-Trust contexts yield relatively more -mIs, with the pattern remaining stable across sensitivity analyses. We then evaluate 10 LLMs in three prompting paradigms (open gap-fill, explicit past-tense gap-fill, and forced-choice A/B selection). LLM behavior is highly model- and prompt-dependent: some models show weak or local trust-consistent shifts, but effects are generally unstable, often reversed, and frequently overshadowed by output-compliance problems and strong base-rate suffix preferences. The results provide new evidence for a trust-/commitment-based account of Turkish evidentiality and reveal a clear human-LLM gap in source-sensitive evidential reasoning. 

---
# Information bottleneck for learning the phase space of dynamics from high-dimensional experimental data 

**Authors**: K. Michael Martini, Eslam Abdelaleem, Paarth Gulati, Ilya Nemenman  

**Link**: [PDF](https://arxiv.org/pdf/2604.24662)  

**Abstract**: Identifying the dynamical state variables of a system from high-dimensional observations is a central problem across physical sciences. The challenge is that the state variables are not directly observable and must be inferred from raw high-dimensional data without supervision. Here we introduce DySIB (Dynamical Symmetric Information Bottleneck) as a method to learn low-dimensional representations of time-series data by maximizing predictive mutual information between past and future observation windows while penalizing representation complexity. This objective operates entirely in latent space and avoids reconstruction of the observations. We apply DySIB to an experimental video dataset of a physical pendulum, where the underlying state space is known. The method, with hyperparameters of the learning architecture set self-consistently by the data, recovers a two-dimensional representation that matches the dimensionality, topology, and geometry of the pendulum phase space, with the learned coordinates aligning smoothly with the canonical angle and angular velocity. These results demonstrate, on a well-characterized experimental system, that predictive information in latent space can be used to recover interpretable dynamical coordinates directly from high-dimensional data. 

---
# AgentWard: A Lifecycle Security Architecture for Autonomous AI Agents 

**Authors**: Yixiang Zhang, Xinhao Deng, Jiaqing Wu, Yue Xiao, Ke Xu, Qi Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.24657)  

**Abstract**: Autonomous AI agents extend large language models into full runtime systems that load skills, ingest external content, maintain memory, plan multi-step actions, and invoke privileged tools. In such systems, security failures rarely remain confined to a single interface; instead, they can propagate across initialization, input processing, memory, decision-making, and execution, often becoming apparent only when harmful effects materialize in the environment. This paper presents AgentWard, a lifecycle-oriented, defense-in-depth architecture that systematically organizes protection across these five stages. AgentWard integrates stage-specific, heterogeneous controls with cross-layer coordination, enabling threats to be intercepted along their propagation paths while safeguarding critical assets. We detail the design rationale and architecture of five coordinated protection layers, and implement a plugin-native prototype on OpenClaw to demonstrate practical feasibility. This perspective provides a concrete blueprint for structuring runtime security controls, managing trust propagation, and enforcing execution containment in autonomous AI agents. Our code is available at this https URL . 

---
# DepthKV: Layer-Dependent KV Cache Pruning for Long-Context LLM Inference 

**Authors**: Zahra Dehghanighobadi, Asja Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2604.24647)  

**Abstract**: Long-context reasoning is a critical capability of large language models (LLMs), enabling applications such as long-document understanding, summarization, and code generation. However, efficient autoregressive inference relies on the key-value (KV) cache, whose memory footprint grows linearly with sequence length, leading to a major memory bottleneck. To mitigate this overhead, KV cache pruning methods discard cached tokens with low attention scores during inference. Most existing methods apply a uniform pruning ratio across layers, implicitly assuming that all layers contribute equally to overall model performance. We show that this assumption is suboptimal, as layers differ significantly in their sensitivity to pruning. We propose DepthKV, a layer-dependent pruning framework that allocates a fixed global KV budget across layers based on their sensitivity, rather than using a uniform allocation. Across multiple models and tasks, DepthKV consistently outperforms uniform pruning at the same global pruning ratio, demonstrating more effective utilization of the KV cache budget through layer-dependent allocation. 

---
# K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology 

**Authors**: Soyeon Kim, Cheongwoong Kang, Myeongjin Lee, Eun-Chul Chang, Jaedeok Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2604.24645)  

**Abstract**: The development of practical (multimodal) large language model assistants for Korean weather forecasters is hindered by the absence of a multidimensional, expert-level evaluation framework grounded in authoritative sources. To address this, we introduce K-MetBench, a diagnostic benchmark grounded in national qualification exams. It exposes critical gaps across four dimensions: expert visual reasoning of charts, logical validity via expert-verified rationales, Korean-specific geo-cultural comprehension, and fine-grained domain analysis. Our evaluation of 55 models reveals a profound modality gap in interpreting specialized diagrams and a reasoning gap where models hallucinate logic despite correct predictions. Crucially, Korean models outperform significantly larger global models in local contexts, demonstrating that parameter scaling alone cannot resolve cultural dependencies. K-MetBench serves as a roadmap for developing reliable, culturally aware expert AI agents. The dataset is available at this https URL . 

---
# Cortex-Inspired Continual Learning: Unsupervised Instantiation and Recovery of Functional Task Networks 

**Authors**: Kevin McKee, Thomas Hazy, Yicong Zheng, Zacharie Bugaud, Thomas Miconi  

**Link**: [PDF](https://arxiv.org/pdf/2604.24637)  

**Abstract**: Block-sequential continual learning demands that a single model both protect prior solutions from catastrophic forgetting and efficiently infer at inference time which prior solution matches the current input without task labels. We present Functional Task Networks (FTN), a parameter-isolation method inspired by structural and dynamical motifs found in the mammalian neocortex. Similar to mixture-of-experts, this method uses a high dimensional, self-organizing binary mask over a large population of small but deep networks, inspired by dendritic models of pyramidal neurons. The mask is produced by a three-stage procedure: (1) gradient descent on a continuous mask identifies task-relevant neurons, (2) a smoothing kernel biases the result toward spatial contiguity, (3) and k-winner-take-all binarizes the resulting group at a fixed capacity budget. Like mixture-of-experts, each neuron is an independent deep network, so disjoint masks give exactly disjoint gradient updates, providing structural guarantees against catastrophic forgetting. This three-stage procedure recovers the sub-network of a previously-trained task in a single gradient step, providing unsupervised task segmentation at inference time. We test it on three continual-learning benchmarks: (1) a synthetic multi-task classification/regression generator, (2) MNIST with shuffled class labels (pure concept shift), and (3) Permuted MNIST (domain shift). On all three, FTN with fine grained smoothing (FTN-Slow) results in nearly zero forgetting. FTN with a large kernel and only 2 iterations of smoothing (FTN-Fast) trades off some retention for increased speed. We show that the spatial organization mechanism reduces the effective mask search from the combinatorial top-k subset problem in O(C(H,K)) to the complexity of a near-linear scan in O(H) over compact cortical neighborhoods, which is parallelized by the gradient-based update. 

---
# Less Is More: Engineering Challenges of On-Device Small Language Model Integration in a Mobile Application 

**Authors**: William Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2604.24636)  

**Abstract**: On-device Small Language Models (SLMs) promise fully offline, private AI experiences for mobile users (no cloud dependency, no data leaving the device). But is this promise achievable in practice? This paper presents a longitudinal practitioner case study documenting the engineering challenges of integrating SLMs (Gemma 4 E2B, 2.6B parameters; Qwen3 0.6B, 600M parameters) into Palabrita, a production Android word-guessing game. Over a 5-day development sprint comprising 204 commits (~90 directly AI-related), the system underwent a radical transformation: from an ambitious design where the LLM generated complete structured puzzles (word, category, difficulty, and five hints as JSON) to a pragmatic architecture where curated word lists provide the words and the LLM generates only three short hints, with a deterministic fallback if it fails. We identify five categories of failures specific to on-device SLM integration: output format violations, constraint violations, context quality degradation, latency incompatibility, and model selection instability. For each failure category, we document the observed symptoms, root causes, and the prompt engineering and architectural strategies that effectively mitigated them, including multi-layer defensive parsing, contextual retry with failure feedback, session rotation, progressive prompt hardening, and systematic responsibility reduction. Our findings demonstrate that on-device SLMs are viable for production mobile applications, but only when the developer accepts a fundamental constraint: the most reliable on-device LLM feature is one where the LLM does the least. We distill our experience into eight actionable design heuristics for practitioners integrating SLMs into mobile apps. 

---
# Meta-CoT: Enhancing Granularity and Generalization in Image Editing 

**Authors**: Shiyi Zhang, Yiji Cheng, Tiankai Hang, Zijin Yin, Runze He, Yu Xu, Wenxun Dai, Yunlong Lin, Chunyu Wang, Qinglin Lu, Yansong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24625)  

**Abstract**: Unified multi-modal understanding/generative models have shown improved image editing performance by incorporating fine-grained understanding into their Chain-of-Thought (CoT) process. However, a critical question remains underexplored: what forms of CoT and training strategy can jointly enhance both the understanding granularity and generalization? To address this, we propose Meta-CoT, a paradigm that performs a two-level decomposition of any single-image editing operation with two key properties: (1) Decomposability. We observe that any editing intention can be represented as a triplet - (task, target, required understanding ability). Inspired by this, Meta-CoT decomposes both the editing task and the target, generating task-specific CoT and traversing editing operations on all targets. This decomposition enhances the model's understanding granularity of editing operations and guides it to learn each element of the triplet during training, substantially improving the editing capability. (2) Generalizability. In the second decomposition level, we further break down editing tasks into five fundamental meta-tasks. We find that training on these five meta-tasks, together with the other two elements of the triplet, is sufficient to achieve strong generalization across diverse, unseen editing tasks. To further align the model's editing behavior with its CoT reasoning, we introduce the CoT-Editing Consistency Reward, which encourages more accurate and effective utilization of CoT information during editing. Experiments demonstrate that our method achieves an overall 15.8% improvement across 21 editing tasks, and generalizes effectively to unseen editing tasks when trained on only a small set of meta-tasks. Our code, benchmark, and model are released at this https URL 

---
# CF-VLA: Efficient Coarse-to-Fine Action Generation for Vision-Language-Action Policies 

**Authors**: Fan Du, Feng Yan, Jianxiong Wu, Xinrun Xu, Weiye Zhang, Weinong Wang, Yu Guo, Bin Qian, Zhihai He  

**Link**: [PDF](https://arxiv.org/pdf/2604.24622)  

**Abstract**: Flow-based vision-language-action (VLA) policies offer strong expressivity for action generation, but suffer from a fundamental inefficiency: multi-step inference is required to recover action structure from uninformative Gaussian noise, leading to a poor efficiency-quality trade-off under real-time constraints. We address this issue by rethinking the role of the starting point in generative action modeling. Instead of shortening the sampling trajectory, we propose CF-VLA, a coarse-to-fine two-stage formulation that restructures action generation into a coarse initialization step that constructs an action-aware starting point, followed by a single-step local refinement that corrects residual errors. Concretely, the coarse stage learns a conditional posterior over endpoint velocity to transform Gaussian noise into a structured initialization, while the fine stage performs a fixed-time refinement from this initialization. To stabilize training, we introduce a stepwise strategy that first learns a controlled coarse predictor and then performs joint optimization. Experiments on CALVIN and LIBERO show that our method establishes a strong efficiency-performance frontier under low-NFE (Number of Function Evaluations) regimes: it consistently outperforms existing NFE=2 methods, matches or surpasses the NFE=10 $\pi_{0.5}$ baseline on several metrics, reduces action sampling latency by 75.4\%, and achieves the best average real-robot success rate of 83.0\%, outperforming MIP by 19.5 points and $\pi_{0.5}$ by 4.0 points. These results suggest that structured, coarse-to-fine generation enables both strong performance and efficient inference. Our code is available at this https URL. 

---
# Learning to Route Queries to Heads for Attention-based Re-ranking with Large Language Models 

**Authors**: Yuxing Tian, Fengran Mo, Zhiqi Huang, Weixu Zhang, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2604.24608)  

**Abstract**: Large Language Models (LLMs) have recently been explored as fine-grained zero-shot re-rankers by leveraging attention signals to estimate document relevance. However, existing methods either aggregate attention signals across all heads or rely on a statically selected subset identified by heuristic rules. This solution can be suboptimal because the informative heads can vary across queries or domains. Moreover, naively combining multiple heads can degrade performance due to redundancy or conflicting ranking signals. In this paper, we propose a query-dependent head selection method, RouteHead, for attention-based re-ranking with LLMs. Specifically, we learn a lightweight router that can map each query to an optimal head set, and relevance scores are computed by aggregating attention signals only from these heads. Since query-to-head optimal labels are unavailable, we first construct pseudo labels via an offline search. The router represents each head with a learnable embedding and represents each query using an embedding extracted from the hidden states of the frozen LLM. Then it is trained on the pseudo labels with a sparsity regularizer. Experiments on diverse benchmarks and multiple LLM backbones show that the proposed method consistently outperforms strong baselines. 

---
# Quantum Kernel Advantage over Classical Collapse in Medical Foundation Model Embeddings 

**Authors**: Sebastian Cajas Ordóñez, Felipe Ocampo Osorio, Dax Enshan Koh, Rafi Al Attrach, Aldo Marzullo, Ariel Guerra-Adames, J. Alejandro Andrade, Siong Thye Goh, Chi-Yu Chen, Rahul Gorijavolu, Xue Yang, Noah Dane Hebdon, Leo Anthony Celi  

**Link**: [PDF](https://arxiv.org/pdf/2604.24597)  

**Abstract**: We provide evidence of quantum kernel advantage under noiseless simulation in binary insurance classification on MIMIC-CXR chest radiographs using quantum support vector machines (QSVM) with frozen embeddings from three medical foundation models (MedSigLIP-448, RAD-DINO, ViT-patch32). We propose a two-tier fair comparison framework in which both classifiers receive identical PCA-q features. At Tier 1 (untuned QSVM vs. untuned linear SVM, C = 1 both sides), QSVM wins minority-class F1 in all 18 tested configurations (17 at p < 0.001, 1 at p < 0.01). The classical linear kernel collapses to majority-class prediction on 90-100% of seeds at every qubit count, while QSVM maintains non-trivial recall. At q = 11 (MedSigLIP-448 plateau center), QSVM achieves mean F1 = 0.343 vs. classical F1 = 0.050 (F1 gain = +0.293, p < 0.001) without hyperparameter tuning. Under Tier 2 (untuned QSVM vs. C-tuned RBF SVM), QSVM wins all seven tested configurations (mean gain +0.068, max +0.112). Eigenspectrum analysis reveals quantum kernel effective rank reaches 69.80 at q = 11, far exceeding linear kernel rank, while classical collapse remains C-invariant. A full qubit sweep reveals architecture-dependent concentration onset across models. Code: this https URL 

---
# Skill Retrieval Augmentation for Agentic AI 

**Authors**: Weihang Su, Jianming Long, Qingyao Ai, Yichen Tang, Changyue Wang, Yiteng Tu, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24594)  

**Abstract**: As large language models (LLMs) evolve into agentic problem solvers, they increasingly rely on external, reusable skills to handle tasks beyond their native parametric capabilities. In existing agent systems, the dominant strategy for incorporating skills is to explicitly enumerate available skills within the context window. However, this strategy fails to scale: as skill corpora expand, context budgets are consumed rapidly, and the agent becomes markedly less accurate in identifying the right skill. To this end, this paper formulates Skill Retrieval Augmentation (SRA), a new paradigm in which agents dynamically retrieve, incorporate, and apply relevant skills from large external skill corpora on demand. To make this problem measurable, we construct a large-scale skill corpus and introduce SRA-Bench, the first benchmark for decomposed evaluation of the full SRA pipeline, covering skill retrieval, skill incorporation, and end-task execution. SRA-Bench contains 5,400 capability-intensive test instances and 636 manually constructed gold skills, which are mixed with web-collected distractor skills to form a large-scale corpus of 26,262 skills. Extensive experiments show that retrieval-based skill augmentation can substantially improve agent performance, validating the promise of the paradigm. At the same time, we uncover a fundamental gap in skill incorporation: current LLM agents tend to load skills at similar rates, regardless of whether a gold skill is retrieved or whether the task actually requires external capabilities. This shows that the bottleneck in skill augmentation lies not only in retrieval but also in the base model's ability to determine which skill to load and when external loading is actually needed. These findings position SRA as a distinct research problem and establish a foundation for the scalable augmentation of capabilities in future agent systems. 

---
# Aligned Multi-View Scripts for Universal Chart-to-Code Generation 

**Authors**: Zhihan Zhang, Lizi Liao  

**Link**: [PDF](https://arxiv.org/pdf/2604.24559)  

**Abstract**: Chart-to-code generation converts a chart image into an executable plotting script, enabling faithful reproduction and editable visualizations. Existing methods are largely Python-centric, limiting practical use and overlooking a critical source of supervision: the same chart can be expressed by semantically equivalent scripts in different plotting languages. To fill this gap, we introduce Chart2NCode, a dataset of 176K charts paired with aligned scripts in Python, R, and LaTeX that render visually equivalent outputs, constructed via a metadata-to-template pipeline with rendering verification and human quality checks. Building on a LLaVA-style architecture, we further propose CharLuMA, a parameter-efficient adaptation module that augments the multimodal projector with a language-conditioned mixture of low-rank subspaces, allowing the model to share core chart understanding while specializing code generation to the target language through lightweight routing. Extensive experiments show consistent gains in executability and visual fidelity across all languages, outperforming strong open-source baselines and remaining competitive with proprietary systems. Further analyses reveal that balanced multi-language supervision benefits all languages and that the adapter allocates a compact shared core plus language-specific capacity. Codes and data are available at this https URL. 

---
# GradMAP: Gradient-Based Multi-Agent Proximal Learning for Grid-Edge Flexibility 

**Authors**: Yihong Zhou, Hongtai Zeng, Thomas Morstyn  

**Link**: [PDF](https://arxiv.org/pdf/2604.24549)  

**Abstract**: Coordinating large populations of grid-edge devices requires learning methods that remain fully decentralised in deployment while still respecting three-phase AC distribution-network physics. This paper proposes gradient-based multi-agent proximal learning (GradMAP) to address this challenge. GradMAP trains independent neural-network policies for each agent without any parameter sharing, and each agent uses only its own local observation for online decision-making without communication. During offline training, GradMAP embeds a differentiable three-phase AC power-flow model in a primal-dual learning loop and uses implicit differentiation to propagate exact network-constraint violations to update the policy parameters. To speed up training, GradMAP reuses expensive environment gradients through a proximal surrogate within a trust region defined in the more direct policy-output (action) space, instead of the probability distribution space used in other works, such as PPO. In case studies with 1,000 agents managing batteries, heat pumps, and controllable generators on the IEEE 123-bus feeder, GradMAP learns decentralised policies that minimise three-phase AC load-flow constraint violations within 15 minutes of training on a single workstation-class NVIDIA RTX PRO 5000 Blackwell 48GB GPU. This is a 3--5x training speed-up over gradient-based self-supervised learning benchmarks and substantially better training efficiency than multi-agent reinforcement-learning benchmarks. In out-of-sample tests, GradMAP also delivers among the lowest operating cost and constraint violations. 

---
# Layerwise Convergence Fingerprints for Runtime Misbehavior Detection in Large Language Models 

**Authors**: Nay Myat Min, Long H. Pham, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.24542)  

**Abstract**: Large language models deployed at runtime can misbehave in ways that clean-data validation cannot anticipate: training-time backdoors lie dormant until triggered, jailbreaks subvert safety alignment, and prompt injections override the deployer's instructions. Existing runtime defenses address these threats one at a time and often assume a clean reference model, trigger knowledge, or editable weights, assumptions that rarely hold for opaque third-party artifacts. We introduce Layerwise Convergence Fingerprinting (LCF), a tuning-free runtime monitor that treats the inter-layer hidden-state trajectory as a health signal: LCF computes a diagonal Mahalanobis distance on every inter-layer difference, aggregates via Ledoit-Wolf shrinkage, and thresholds via leave-one-out calibration on 200 clean examples, with no reference model, trigger knowledge, or retraining. Evaluated on four architectures (Llama-3-8B, Qwen2.5-7B, Gemma-2-9B, Qwen2.5-14B) across backdoors, jailbreaks, and prompt injection (56 backdoor combinations, 3 jailbreak techniques, and BIPIA email + code-QA), LCF reduces mean backdoor attack success rate (ASR) below 1% on Qwen2.5-7B and Gemma-2 and to 1.3% on Qwen2.5-14B, detects 92-100% of DAN jailbreaks (62-100% for GCG and softer role-play), and flags 100% of text-payload injections across all eight (model, domain) cells, at 12-16% backdoor FPR and <0.1% inference overhead. A single aggregation score covers all three threat families without threat-specific tuning, positioning LCF as a general-purpose runtime safety layer for cloud-served and on-device LLMs. 

---
# Understanding the Limits of Automated Evaluation for Code Review Bots in Practice 

**Authors**: Veli Karakaya, Utku Boran Torun, Baykal Mehmet Uçar, Eray Tüzün  

**Link**: [PDF](https://arxiv.org/pdf/2604.24525)  

**Abstract**: Automated code review (ACR) bots are increasingly used in industrial software development to assist developers during pull request (PR) review. As adoption grows, a key challenge is how to evaluate the usefulness of bot-generated comments reliably and at scale. In practice, such evaluation often relies on developer actions and annotations that are shaped by contextual and organizational factors, complicating their use as objective ground truth. We examine the feasibility and limitations of automating the evaluation of LLM-powered ACR bots in an industrial setting. We analyze an industrial dataset from Beko comprising 2,604 bot-generated PR comments, each labeled by software engineers as fixed/wontFix. Two automated evaluation approaches, G-Eval and an LLM-as-a-Judge pipeline, are applied using both binary decisions and a 0-4 Likert-scale formulation, enabling a controlled comparison against developer-provided labels. Across Gemini-2.5-pro, GPT-4.1-mini, and GPT-5.2, both evaluation strategies achieve only moderate alignment with human labels. Agreement ratios range from approximately 0.44 to 0.62, with noticeable variation across models and between binary and Likert-scale formulations, indicating sensitivity to both model choice and evaluation design. Our findings highlight practical limitations in fully automating the evaluation of ACR bot comments in industrial contexts. Developer actions such as resolving or ignoring comments reflect not only comment quality, but also contextual constraints, prioritization decisions, and workflow dynamics that are difficult to capture through static artifacts. Insights from a follow-up interview with a software engineering director further corroborate that developer labeling behavior is strongly influenced by workflow pressures and organizational constraints, reinforcing the challenges of treating such signals as objective ground truth. 

---
# Why AI Harms Can't Be Fixed One Identity at a Time: What 5300 Incident Reports Reveal About Intersectionality 

**Authors**: Edyta Bogucka, Sanja Šćepanović, Daniele Quercia  

**Link**: [PDF](https://arxiv.org/pdf/2604.24519)  

**Abstract**: AI risk assessment is the primary tool for identifying harms caused by AI systems. These include intersectional harms, which arise from the interaction between identity categories (e.g., class and skin tone) and which do not occur, or occur differently, when those categories are considered separately. Yet existing AI risk assessments are still built around isolated identity categories, and when intersections are considered, they focus almost exclusively on race and gender. Drawing on a large-scale analysis of documented AI incidents, we show that AI harms do not occur one identity category at a time. Using a structured rubric applied with a Large Language Model (LLM), we analyze 5,300 reports from 1,200 documented incidents in the AI Incident Database, the most curated source of incident data. From these reports, we identify 1,513 harmed subjects and their associated identity categories, achieving 98% accuracy. At the level of individual categories, we find that age and political identity appear in documented AI harms at rates comparable to race and gender. At the level of intersecting categories, harm is amplified up to three times at specific intersections: adolescent girls, lower-class people of color, and upper-class political elites. We argue that intersectionality should be a core component of AI risk assessment to more accurately capture how harms are produced and distributed across social groups. 

---
# Deployment-Aligned Low-Precision Neural Architecture Search for Spaceborne Edge AI 

**Authors**: Parampuneet Kaur Thind, Vaibhav Katturu, Giacomo Zema, Roberto Del Prete  

**Link**: [PDF](https://arxiv.org/pdf/2604.24492)  

**Abstract**: Designing deep networks that meet strict latency and accuracy constraints on edge accelerators increasingly relies on hardware-aware optimization, including neural architecture search (NAS) guided by device-level metrics. Yet most hardware-aware NAS pipelines still optimize architectures under full-precision assumptions and apply low-precision adaptation only after the search, leading to a mismatch between optimization-time behavior and deployment-time execution on low-precision hardware that can substantially degrade accuracy. We address this limitation by integrating deployment-aligned low-precision training directly into hardware-aware NAS. Candidate architectures are exposed to FP16 numerical constraints during fine-tuning and evaluation, enabling joint optimization of architectural efficiency and numerical robustness without modifying the search space or evolutionary strategy. We evaluate the proposed framework on vessel segmentation for spaceborne maritime monitoring, targeting the Intel Movidius Myriad X Visual Processing Unit (VPU). While post-training precision conversion reduces on-device performance from 0.85 to 0.78 mIoU, deployment-aligned low-precision training achieves 0.826 mIoU on-device for the same architecture (95,791 parameters), recovering approximately two-thirds of deployment-induced accuracy gap without increasing model complexity. These results demonstrate that incorporating deployment-consistent numerical constraints into hardware-aware NAS substantially improves robustness and alignment between optimization and deployment for resource-constrained edge Artificial Intelligence (AI). 

---
# GAMMAF: A Common Framework for Graph-Based Anomaly Monitoring Benchmarking in LLM Multi-Agent Systems 

**Authors**: Pablo Mateo-Torrejón, Alfonso Sánchez-Macián  

**Link**: [PDF](https://arxiv.org/pdf/2604.24477)  

**Abstract**: The rapid integration of Large Language Models (LLMs) into Multi-Agent Systems (MAS) has significantly enhanced their collaborative problem-solving capabilities, but it has also expanded their attack surfaces, exposing them to vulnerabilities such as prompt infection and compromised inter-agent communication. While emerging graph-based anomaly detection methods show promise in protecting these networks, the field currently lacks a standardized, reproducible environment to train these models and evaluate their efficacy. To address this gap, we introduce Gammaf (Graph-based Anomaly Monitoring for LLM Multi-Agent systems Framework), an open-source benchmarking platform. Gammaf is not a novel defense mechanism itself, but rather a comprehensive evaluation architecture designed to generate synthetic multi-agent interaction datasets and benchmark the performance of existing and future defense models. The proposed framework operates through two interdependent pipelines: a Training Data Generation stage, which simulates debates across varied network topologies to capture interactions as robust attributed graphs, and a Defense System Benchmarking stage, which actively evaluates defense models by dynamically isolating flagged adversarial nodes during live inference rounds. Through rigorous evaluation using established defense baselines (XG-Guard and BlindGuard) across multiple knowledge tasks (such as MMLU-Pro and GSM8K), we demonstrate Gammaf's high utility, topological scalability, and execution efficiency. Furthermore, our experimental results reveal that equipping an LLM-MAS with effective attack remediation not only recovers system integrity but also substantially reduces overall operational costs by facilitating early consensus and cutting off the extensive token generation typical of adversarial agents. 

---
# Modeling Behavioral Intensity and Transitions for Generative Recommendation 

**Authors**: Wenxuan Yang, Xiaoyang Xu, Hanyu Zhang, Zhexuan Xu, Wanqiang Xiong, Zhaoqun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.24472)  

**Abstract**: Multi-behavior recommendation aims to predict user conversions by modeling various interaction types that carry distinct intent signals. Recently, generative sequence modeling methods have emerged as an important paradigm for multi-behavior recommendation by achieving flexible sequence generation. However, existing generative methods typically treat behaviors as auxiliary token features and feed them into unified attention mechanisms. These models implicitly assume uniform activation of dependencies among historical behaviors, thereby failing to discern differences in intensity or capture transition patterns. To address these limitations, we propose BITRec, a novel generative multi-behavior recommendation framework that introduces structured behavioral modeling through selective dependency activation. BITRec incorporates (i) Hierarchical Behavior Aggregation (HBA), which explicitly models behavioral intensity differences through separated exploration and commitment pathways, and (ii) Transition Relation Encoding (TRE), which encodes transition structures through explicit learnable relation matrices. Experiments on four large-scale datasets (RetailRocket, Taobao, Tmall, Insurance Dataset) with millions of interactions achieve consistent improvements of 15-23% across multiple metrics, with peak gains of 22.79% MRR on Tmall and 17.83% HR@10, 17.55% NDCG@10 on Taobao. 

---
# Measuring Successful Cooperation in Human-AI Teamwork: Development and Validation of the Perceived Cooperativity and Teaming Perception Scales 

**Authors**: Christiane Attig, Christiane Wiebel-Herboth, Patricia Wollstadt, Tim Schrills, Mourad Zoubir, Thomas Franke  

**Link**: [PDF](https://arxiv.org/pdf/2604.24461)  

**Abstract**: As human-AI cooperation becomes increasingly prevalent, reliable instruments for assessing the subjective quality of cooperative human-AI interaction are needed. We introduce two theoretically grounded scales: the Perceived Cooperativity Scale (PCS), grounded in joint activity theory, and the Teaming Perception Scale (TPS), grounded in evolutionary cooperation theory. The PCS captures an agent's perceived cooperative capability and practice within a single interaction sequence; the TPS captures the emergent sense of teaming arising from mutual contribution and support. Both scales were adapted for human-human cooperation to enable cross-agent comparisons. Across three studies (N = 409) encompassing a cooperative card game, LLM interaction, and a decision-support system, analyses of dimensionality, reliability, and validity indicated that both scales successfully differentiated between cooperation partners of varying cooperative quality and showed construct validity in line with expectations. The scales provide a basis for empirical investigation and system evaluation across a wide range of human-AI cooperation contexts. 

---
# SPLIT: Separating Physical-Contact via Latent Arithmetic in Image-Based Tactile Sensors 

**Authors**: Wadhah Zai El Amri, Nicolás Navarro-Guerrero  

**Link**: [PDF](https://arxiv.org/pdf/2604.24449)  

**Abstract**: Training machine learning models for robotic tactile sensing requires vast amounts of data, yet obtaining realistic interaction data remains a challenge due to physical complexity and variability. Simulating tactile sensors is thus a crucial step in accelerating progress. This paper presents SPLIT, a novel method for simulating image-based tactile sensors, with a primary focus on the DIGIT sensor. Central to our approach is a latent space arithmetic strategy that explicitly disentangles contact geometry from sensor-specific optical properties. Unlike methods that require recalibration for every new unit, this disentanglement allows SPLIT to adapt to diverse DIGIT backgrounds and even transfer data to distinct sensors like the GelSight R1.5 without full model retraining. Beyond this adaptability, our approach achieves faster inference speeds than existing alternatives. Furthermore, we provide a calibrated finite element method (FEM) soft-body mesh simulation with variable resolution, offering a tunable trade-off between speed and fidelity. Additionally, our algorithm supports bidirectional simulation, allowing for both the generation of realistic images from deformation meshes and the reconstruction of meshes from tactile images. This versatility makes SPLIT a valuable tool for accelerating progress in robotic tactile sensing research. 

---
# Characterizing Vision-Language-Action Models across XPUs: Constraints and Acceleration for On-Robot Deployment 

**Authors**: Kaijun Zhou, Qiwei Chen, Da Peng, Zhiyang Li, Xijun Li, Jinyu Gu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24447)  

**Abstract**: Vision-Language-Action (VLA) models are promising for generalist robot control, but on-robot deployment is bottlenecked by real-time inference under tight cost and energy budgets. Most prior evaluations rely on desktop-grade GPUs, obscuring the trade-offs and opportunities offered by heterogeneous edge accelerators (GPUs/XPUs/NPUs). We present a systematic analysis for low-cost VLA deployment via model-hardware co-characterization. First, we build a cross-accelerator leaderboard and evaluate model-hardware pairs under CET (Cost, Energy, Time), showing that right-sized edge devices can be more cost-/energy-efficient than flagship GPUs while meeting control-rate constraints. Second, using in-depth profiling, we uncover a consistent two-phase inference pattern: a compute-bound VLM backbone followed by a memory-bound Action Expert, which induces phase-dependent underutilization and hardware inefficiency. Finally, guided by these insights, we propose DP-Cache and V-AEFusion to reduce diffusion redundancy and enable asynchronous pipeline parallelism, achieving up to 2.9x speedup on GPUs and 6x on edge NPUs with only marginal success degradation. The example leaderboard website is available at: this https URL. 

---
# Kwai Summary Attention Technical Report 

**Authors**: Chenglong Chu, Guorui Zhou, Guowang Zhang, Han Li, Hao Peng, Hongtao Cheng, Jian Liang, Jiangxia Cao, Kun Gai, Lingzhi Zhou, Lu Ren, Qi Zhang, Ruiming Tang, Ruitao Wang, Xinchen Luo, Yi Su, Zhiyuan Liang, Ziqi Wang, Boyang Ding, Chengru Song, Dunju Zang, Hui Wang, Jiao Ou, Jiaxin Deng, Jijun Shi, Jinghao Zhang, Junmin Chen, Lejian Ren, Minxuan Lv, Qianqian Wang, Qigen Hu, Shiyao Wang, Siyang Mao, Tao Wang, Xingmei Wang, Zhixin Ling, Ziming Li, Zixing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24432)  

**Abstract**: Long-context ability, has become one of the most important iteration direction of next-generation Large Language Models, particularly in semantic understanding/reasoning, code agentic intelligence and recommendation system. However, the standard softmax attention exhibits quadratic time complexity with respect to sequence length. As the sequence length increases, this incurs substantial overhead in long-context settings, leading the training and inference costs of extremely long sequences deteriorate rapidly. Existing solutions mitigate this issue through two technique routings: i) Reducing the KV cache per layer, such as from the head-level compression GQA, and the embedding dimension-level compression MLA, but the KV cache remains linearly dependent on the sequence length at a 1:1 ratio. ii) Interleaving with KV Cache friendly architecture, such as local attention SWA, linear kernel GDN, but often involve trade-offs among KV Cache and long-context modeling effectiveness. Besides the two technique routings, we argue that there exists an intermediate path not well explored: {Maintaining a linear relationship between the KV cache and sequence length, but performing semantic-level compression through a specific ratio $k$}. This $O(n/k)$ path does not pursue a ``minimum KV cache'', but rather trades acceptable memory costs for complete, referential, and interpretable retention of long distant dependency. Motivated by this, we propose Kwai Summary Attention (KSA), a novel attention mechanism that reduces sequence modeling cost by compressing historical contexts into learnable summary tokens. 

---
# BandRouteNet: An Adaptive Band Routing Neural Network for EEG Artifact Removal 

**Authors**: Phat Lam  

**Link**: [PDF](https://arxiv.org/pdf/2604.24428)  

**Abstract**: Electroencephalography (EEG) is highly susceptible to artifact contamination, such as electrooculographic (EOG) and electromyographic (EMG) interference, which severely degrades signal quality and hinders reliable interpretation in applications including neurological diagnosis, brain-computer interfaces (BCIs), etc. Effective EEG denoising remains challenging because different artifact sources exhibit diverse and temporally varying distributions, together with distinct spectral characteristics across frequency bands. To address these issues, we propose BandRouteNet, an adaptive frequency-aware neural network for EEG denoising that jointly exploits band-specific processing and full-band contextual modeling. The proposed model performs band-wise denoising to explicitly capture frequency-dependent artifact patterns. Within this framework, we introduce a routing mechanism that adaptively determines where and to what extent denoising should be applied across temporal locations within each frequency band. In parallel, a full-band conditioner directly processes the original noisy EEG to extract global temporal context, producing both conditional parameters for modulating the band-wise pathway and a coarse-grained signal-level refinement to supplement the final reconstruction. Extensive experiments on the EEGDenoiseNet benchmark dataset demonstrate that BandRouteNet outperforms other methods under EOG, EMG, and mixed-artifact conditions in terms of Relative Root Mean Square Error (RRMSE) and Signal-to-Noise Ratio Improvement (SNR$_{\text{imp}}$) under unified experimental settings, while remaining highly parameter-efficient with only 0.2M trainable parameters. These results highlight its strong potential for high-performance EEG artifact removal in resource-constrained applications. 

---
# Scaling Properties of Continuous Diffusion Spoken Language Models 

**Authors**: Jason Ramapuram, Eeshan Gunesh Dhekane, Amitis Shidani, Dan Busbridge, Bogdan Mazoure, Zijin Gu, Russ Webb, Tatiana Likhomanenko, Navdeep Jaitly  

**Link**: [PDF](https://arxiv.org/pdf/2604.24416)  

**Abstract**: Speech-only spoken language models (SLMs) lag behind text and text-speech models in performance, with recent discrete autoregressive (AR) SLMs indicating significant computational and data demands to match text models. Since discretizing continuous speech for AR creates bottlenecks, we explore whether continuous diffusion (CD) SLM is more viable. To quantify the SLMs linguistic quality, we introduce the phoneme Jensen-Shannon divergence (pJSD) metric. Our analysis reveals CD SLMs, mirroring AR behavior, exhibit scaling laws for validation loss and pJSD, and show optimal token-to-parameter ratios decreasing as compute scales. However, for the latter, loss becomes insensitive to choice of data and model sizes, showing potential for fast inference. Scaling CD SLMs to 16B parameters with tens of millions of hours of conversational data enables generation of emotive, prosodic, multi-speaker, multilingual speech, though achieving long-form coherence remains a significant challenge. 

---
# All That Glitters Is Not Audio: Rethinking Text Priors and Audio Reliance in Audio-Language Evaluation 

**Authors**: Leonardo Haw-Yang Foo, Chih-Kai Yang, Chen-An Li, Ke-Han Lu, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.24401)  

**Abstract**: Large Audio-Language Models show consistent performance gains across speech and audio benchmarks, yet high scores may not reflect true auditory perception. If a model can answer questions without processing the acoustic signal, the benchmark fails as a measure of auditory understanding. We present a diagnostic framework using two axes: text prior, which measures answerability from text and general knowledge alone, and audio reliance, which assesses actual dependency on the acoustic signal. Evaluating eight LALMs across three benchmarks, we find that models retain 60-72% of their full audio scores even without any audio input. Moreover, among items that require audio, only 3.0-4.2% need the complete audio clip; the majority can be resolved using localized fragments. These findings challenge the assumption that benchmark performance equals robust audio understanding, and we conclude with practical guidelines for improving evaluation reliability and benchmark design. 

---
# Global Context or Local Detail? Adaptive Visual Grounding for Hallucination Mitigation 

**Authors**: Yubo Jiang, Xin Yang, Abudukelimu Wuerkaixi, Zheming Yuan, Xuxin Cheng, Fengying Xie, Zhiguo Jiang, Cao Liu, Ke Zeng, Haopeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24396)  

**Abstract**: Vision-Language Models (VLMs) are frequently undermined by object hallucination--generating content that contradicts visual reality--due to an over-reliance on linguistic priors. We introduce Positive-and-Negative Decoding (PND), a training-free inference framework that intervenes directly in the decoding process to enforce visual fidelity. PND is motivated by our key finding of a critical attention deficit in VLMs, where visual features are empirically under-weighted. Our framework corrects this via a dual-path contrast: The positive path amplifies salient visual evidence using multi-layer attention to encourage faithful descriptions, directly counteracting the attention deficit. Simultaneously, the negative path identifies and degrades the core object's features to create a strong counterfactual, which penalizes ungrounded, prior-dominant generation. By contrasting the model's outputs from these two perspectives at each step, PND steers generation towards text that is not just linguistically probable, but visually factual. Extensive experiments on benchmarks like POPE, MME, and CHAIR show that PND achieves state-of-the-art performance with up to 6.5% accuracy improvement, substantially reducing object hallucination while also enhancing descriptive detail--all without requiring any model retraining. The method generalizes effectively across diverse VLM architectures including LLaVA, InstructBLIP, InternVL, and Qwen-VL. 

---
# SeaEvo: Advancing Algorithm Discovery with Strategy Space Evolution 

**Authors**: Sichun Luo, Yi Huang, Haochen Luo, Fengyuan Liu, Guanzhi Deng, Lei Li, Qinghua Yao, Zefa Hu, Junlan Feng, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24372)  

**Abstract**: LLM-guided evolutionary search has emerged as a promising paradigm for automated algorithm discovery, yet most systems track search progress primarily through executable programs and scalar fitness. Even when natural-language reflection is used, it is often used locally in mutation prompts or stored without an explicit population-level organization of strategic directions. As a result, evolutionary search can struggle to distinguish syntactically different implementations of the same idea, preserve lower-fitness but strategically promising directions, or detect when an entire family of strategies has saturated.
We introduce \model, a modular strategy-space layer that elevates natural-language strategy descriptions from transient prompt context to first-class population-level evolutionary state in LLM-driven program search. \model augments each candidate program with an explicit natural language strategy description and uses this representation in three ways: Strategy Articulation turns mutation into a diagnose-direct-implement process; Stratified Experience Retrieval organizes the archive into strategy clusters and selects inspirations by behavioral complementarity; and Strategic Landscape Navigation periodically summarizes effective, saturated, and underexplored strategy families to guide future mutations. Across mathematical algorithm discovery, systems optimization, and agent-scaffold benchmarks, \model improves the underlying evolutionary backbones in most settings, with particularly large gains (21% relative improvement) on open-ended system optimization tasks. These results suggest that persistent strategy representations provide a practical mechanism for improving the robustness and efficiency of LLM-guided evolutionary search, suggesting a path toward compound AI systems that accumulate algorithmic knowledge over time. 

---
# PathMoG: A Pathway-Centric Modular Graph Neural Network for Multi-Omics Survival Prediction 

**Authors**: Di Wang, Chupei Tang, Junxiao Kong, Jixiu Zhai, Moyu Tang, Tianchi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24371)  

**Abstract**: Cancer survival prediction from multi-omics data remains challenging because prognostic signals are high-dimensional, heterogeneous, and distributed across interacting genes and pathways. We propose PathMoG, a pathway-centric modular graph neural network for multi-omics survival prediction. PathMoG reorganizes genome-scale inputs into 354 KEGG-informed pathway modules, introduces a Hierarchical Omics Modulation module to condition gene-expression representations on mutation, copy number variation, pathway, and clinical context, and uses dual-level attention to capture both intra-pathway driver signals and inter-pathway clinical relevance. We evaluated PathMoG on 5,650 patients across 10 TCGA cancer types and observed consistent improvements over representative survival baselines. The framework further provides gene-level, pathway-level, and patient-level interpretability, supporting biologically grounded and clinically relevant risk stratification. 

---
# DPRM: A Plug-in Doob h transform-induced Token-Ordering Module for Diffusion Language Models 

**Authors**: Dake Bu, Wei Huang, Andi Han, Hau-San Wong, Qingfu Zhang, Taiji Suzuki, Atsushi Nitanda  

**Link**: [PDF](https://arxiv.org/pdf/2604.24357)  

**Abstract**: Diffusion language models generate without a fixed left-to-right order, making token ordering a central algorithmic choice: which tokens should be revealed, retained, revised or verified at each step? Existing systems mainly use random masking or confidence-driven ordering. Random masking creates train--test mismatch, while confidence-only rules are efficient but can be myopic and suppress useful exploration.
We introduce DPRM (Doob h-transform Process Reward Model), a plug-in token-ordering module for diffusion language models. DPRM keeps the host architecture, denoising objective and supervision unchanged, and changes only the ordering policy. It starts from confidence-driven progressive ordering and gradually shifts to Doob h transform Process Reward guided ordering through online estimates.
We characterize the exact DPRM policy as a reward-tilted Gibbs reveal law, prove O(1/N) convergence of the stagewise Soft-BoN approximation, and show that the online bucketized controller tracks the exact DPRM score at empirical-Bernstein rates. Under tractable optimization assumptions, DPRM also yields a sample-complexity advantage over random and confidence-only ordering.
DPRM improves over confidence-based baselines in pretraining, post-training, test-time scaling, and single-cell masked diffusion, with particularly strong gains on harder reasoning subsets. In protein, molecular generation and DNA design, the effect is more multi-objective: ordering-aware variants significantly improve selected structural or fragment-constrained metrics while not uniformly dominating the host baseline on every quality metric. These results identify token ordering as a fundamental control axis in diffusion language models and establish DPRM as a general-purpose module for improving it. Code is available at this https URL. 

---
# ARETE: Attention-based Rasterized Encoding for Topology Estimation using HSV-transformed Crowdsourced Vehicle Fleet Data 

**Authors**: Daniel Fritz, Dimitrios Lagamtzis, Michael Mink, Markus Enzweiler, Steffen Schober  

**Link**: [PDF](https://arxiv.org/pdf/2604.24353)  

**Abstract**: The continuous advancement of autonomous driving (AD) introduces challenges across multiple disciplines to ensure safe and efficient driving. One such challenge is the generation of High-Definition (HD) maps, which must remain up to date and highly accurate for downstream automotive tasks. One promising approach is the use of crowdsourced data from a vehicle fleet, representing road topology and lane-level features. This work focuses on the generation of centerlines and lane dividers from crowdsourced vehicle trajectories. We adopt a Detection Transformer (DETR)-based approach, where a rasterized representation of vehicle trajectories is used as input to predict vectorized lane representations. Each lane consists of a centerline with an associated direction and corresponding lane dividers that are geometrically constrained by the centerline. Our method includes the extraction of local tiles, from which crowdsourced vehicle trajectories are aggregated. Each tile undergoes a transformation into a rasterized representation encoding both the presence and direction of each trajectory, enabling the prediction of vectorized directed lanes. Experiments are conducted on an internal dataset as well as on the public datasets nuScenes and nuPlan. 

---
# Diffusion Templates: A Unified Plugin Framework for Controllable Diffusion 

**Authors**: Zhongjie Duan, Hong Zhang, Yingda Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.24351)  

**Abstract**: Controllable diffusion methods have substantially expanded the practical utility of diffusion models, but they are typically developed as isolated, backbone-specific systems with incompatible training pipelines, parameter formats, and runtime hooks. This fragmentation makes it difficult to reuse infrastructure across tasks, transfer capabilities across backbones, or compose multiple controls within a single generation pipeline. We present Diffusion Templates, a unified and open plugin framework that decouples base-model inference from controllable capability injection. The framework is organized around three components: Template models that map arbitrary task-specific inputs to an intermediate capability representation, a Template cache that functions as a standardized interface for capability injection, and a Template pipeline that loads, merges, and injects one or more Template caches into the base diffusion runtime. Because the interface is defined at the systems level rather than tied to a specific control architecture, heterogeneous capability carriers such as KV-Cache and LoRA can be supported under the same abstraction. Based on this design, we build a diverse model zoo spanning structural control, brightness adjustment, color adjustment, image editing, super-resolution, sharpness enhancement, aesthetic alignment, content reference, local inpainting, and age control. These case studies show that Diffusion Templates can unify a broad range of controllable generation tasks while preserving modularity, composability, and practical extensibility across rapidly evolving diffusion backbones. All resources will be open sourced, including code, models, and datasets. 

---
# Unveiling the Backdoor Mechanism Hidden Behind Catastrophic Overfitting in Fast Adversarial Training 

**Authors**: Mengnan Zhao, Lihe Zhang, Tianhang Zheng, Bo Wang, Baocai Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.24350)  

**Abstract**: Fast Adversarial Training (FAT) has attracted significant attention due to its efficiency in enhancing neural network robustness against adversarial attacks. However, FAT is prone to catastrophic overfitting (CO), wherein models overfit to the specific attack used during training and fail to generalize to others. While existing methods introduce diverse hypotheses and propose various strategies to mitigate CO, a systematic and intuitive explanation of CO remains absent. In this work, we innovatively interpret CO through the lens of backdoor. Through validations on pathway division, diverse feature predictions, and universal class distinguishable triggers in CO, we conceptualize CO as a weak trigger variant of unlearnable tasks, unifying CO, backdoor attacks, and unlearnable tasks under a common theoretical framework. Guided by this, we leverage several backdoor inspired strategies to mitigate CO: (i) Recalibrate CO affected model parameters using vanilla fine tuning, linear probing, or reinitialization-based techniques; (ii) Introduce a weight outlier suppression constraint to regulate abnormal deviations in model weights. Extensive experiments support our interpretation of CO and show the efficacy of the proposed mitigation strategies. 

---
# SycoPhantasy: Quantifying Sycophancy and Hallucination in Small Open Weight VLMs for Vision-Language Scoring of Fantasy Characters 

**Authors**: Arya Shah, Deepali Mishra, Chaklam Silpasuwanchai  

**Link**: [PDF](https://arxiv.org/pdf/2604.24346)  

**Abstract**: Vision-language models (VLMs) are increasingly deployed as evaluators in tasks requiring nuanced image understanding, yet their reliability in scoring alignment between images and text descriptions remains underexplored. We investigate whether small, open-weight VLMs exhibit \emph{sycophantic} behavior when evaluating image-text alignment: assigning high scores without grounding their judgments in visual evidence. To quantify this phenomenon, we introduce the \emph{Bluffing Coefficient} (\bc), a metric that measures the mismatch between a model's score and its evidence recall. We evaluate six open-weight VLMs ranging from 450M to 8B parameters on a benchmark of 173,810 AI-generated character portraits paired with detailed textual descriptions. Our analysis reveals a significant inverse correlation between model size and sycophancy rate ($r = -0.96$, $p = 0.002$), with smaller models exhibiting substantially higher rates of unjustified high scores. The smallest model tested (LFM2-VL, 450M) produced sycophantic evaluations in 22.3\% of cases, compared to 6.0\% for the largest (LLaVA-1.6, 7B). These findings have direct implications for the deployment of small, open-weight VLMs as automated evaluators within attribute-rich, synthetic image evaluation tasks, where the gap between assigned scores and cited visual evidence is both measurable and consequential. 

---
# See Further, Think Deeper: Advancing VLM's Reasoning Ability with Low-level Visual Cues and Reflection 

**Authors**: Zhiheng Wu, Tong Wang, Shuning Wang, Naiming Liu, Yumeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24339)  

**Abstract**: Recent advances in Vision-Language Models (VLMs) have benefited from Reinforcement Learning (RL) for enhanced reasoning. However, existing methods still face critical limitations, including the lack of low-level visual information and effective visual feedback. To address these problems, this paper proposes a unified multimodal interleaved reasoning framework \textbf{ForeSight}, which enables VLMs to \textbf{See Further} with low-level visual cues and \textbf{Think Deeper} with effective visual feedback. First, it introduces a set of low-level visual tools to integrate essential visual information into the reasoning chain, mitigating the neglect of fine-grained visual features. Second, a mask-based visual feedback mechanism is elaborated to incorporate visual reflection into the thinking process, enabling the model to dynamically re-examine and update its answers. Driven by RL, ForeSight learns to autonomously decide on tool invocation and answer verification, with the final answer accuracy as the reward signal. To evaluate the performance of the proposed framework, we construct a new dataset, Character and Grounding SalBench (CG-SalBench), based on the SalBench dataset. Experimental results demonstrate that the ForeSight-7B model significantly outperforms other models with the same parameter scale, and even surpasses the current SOTA closed-source models on certain metrics. 

---
# X-NegoBox: An Explainable Privacy-Budget Negotiation Framework for Secure Peer-to-Peer Energy Data Exchange 

**Authors**: Poushali Sengupta, Sabita Maharjan, Frank Eliassen, Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24326)  

**Abstract**: The decentralization of modern energy systems is transforming consumers into prosumers who continuously exchange data with aggregators, peers, and market operators. While such data is essential for peer-to-peer trading, demand response, and distributed forecasting, it can reveal sensitive household patterns and introduce privacy risks. Existing data sharing mechanisms rely on fixed policies or predefined differential privacy budgets, limiting their ability to adapt to variations in reliability, data sensitivity, and request purpose. As a result, prosumers rarely receive explanations for why a request is accepted, rejected, or modified, reducing trust and participation.
To address these limitations, we propose X-NegoBox, an explainable negotiation framework for adaptive privacy budgeting and transparent decision making. Each prosumer data is managed locally within a private DataBox, where raw data remain confined. Incoming requests are processed by an Autonomous Privacy Budget Negotiation Protocol (APBNP), which determines an appropriate privacy budget based on trust, feature sensitivity, declared purpose, historical behavior, and risk-aware pricing. When needed, APBNP generates privacy-preserving counter-offers, such as reduced resolution or duration.
An Explainable Agreement Layer (X-Contract) produces human- and machine-readable justifications for each decision. After agreement, requester code executes locally in a sandbox, and only sanitized outputs are shared. Experiments on realistic energy market settings show reduced privacy leakage, higher acceptance rates, and improved interpretability. 

---
# Self-Abstraction Learning for Effective and Stable Training of Deep Neural Networks 

**Authors**: Wonyong Cho, Taemin Kim, Jungmin Kim, Jeong-Rae Kim, Sung Hoon Jung  

**Link**: [PDF](https://arxiv.org/pdf/2604.24313)  

**Abstract**: Training large-scale deep neural networks effectively and stably is essential for applying deep learning across various fields. However, conventional methods, which rely on training a single large network, often encounter challenges such as gradient vanishing, overfitting and unstable learning. To overcome these limitations, we introduce Self-Abstraction Learning (SAL), a hierarchical framework. In SAL, networks are arranged by structural complexity, where the simplest topmost network is trained first and its hidden and output layers serve as guidance for the successively more complex networks below. This top-down sequential guidance effectively mitigates optimization issues, enabling stable training of deep architectures. Various experiments across MLP, CNN, and RNN architectures demonstrate that SAL consistently outperforms conventional methods, ensuring robust generalization even in data-scarce and complex network regimes. 

---
# Unconstrained Multi-view Human Pose Estimation with Algebraic Priors 

**Authors**: Xiaolin Qin, Qianlei Wang, Jiacen Liu, Chaoning Zhang, Fei Zhu, Zhang Yi  

**Link**: [PDF](https://arxiv.org/pdf/2604.24312)  

**Abstract**: Recovering 3D human pose from multi-view imagery typically relies on precise camera calibration, which is often unavailable in real-world scenarios, thereby severely limiting the applicability of existing methods. To overcome this challenge, we propose an unconstrained framework that synergizes deep neural networks, algebraic priors, and temporal dynamics for uncalibrated multi-view human pose estimation. First, we introduce the Triangulation with Transformer Regressor (TTR), which reformulates classical triangulation into a data-driven token fusion process to bypass the dependency on explicit camera parameters. Second, to explicitly embed the inherent algebraic relations of the multi-view variety into the learning process, we propose the Gröbner basis Corrector (GC). This pioneering loss formulation enforces constraints derived from the multi-view variety to ensure the neural predictions strictly adhere to the laws of projective geometry. Finally, we devise the Temporal Equivariant Rectifier (TER), which exploits the equivariance property of human motion to impose temporal coherence and structural consistency, effectively mitigating scale ambiguity in uncalibrated settings. Extensive evaluations on standard benchmarks demonstrate that our framework establishes a new state-of-the-art for uncalibrated multi-view human pose estimation. Notably, our approach significantly closes the performance gap between calibration-free methods and fully calibrated oracles. 

---
# SolarTformer: A Transformer Based Deep Learning Approach for Short Term Solar Power Forecasting 

**Authors**: Ankan Basu, Jyotiraditya Roy, Aditya Datta, Prayas Sanyal, Sumanta Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2604.24306)  

**Abstract**: Accurate forecasting of solar power output is essential for efficient integration of renewable energy into the grid. In this study, an attention-based deep learning model, inspired by transformer architecture, is used for short-term solar power forecasting. Our proposed model, "SolarTformer", is designed to predict solar power output from meteorological data. Unlike traditional models, SolarTformer leverages self-attention mechanisms to effectively capture temporal dependencies and spatial variability in solar irradiance. In addition, the proposed methodology includes feeding power station-specific metadata into the model, which helps to generalize between power stations located at different locations and with different panel configurations and in different seasons. Our experiments demonstrate that SolarTformer significantly outperforms previous models on the same data set. In particular, the model exhibits strong performance on both clear and cloudy days, indicating high robustness and generalizability. These findings highlight the potential of attention-based architectures in enhancing the accuracy of solar forecasting, contributing to a more reliable management of renewable energy. 

---
# Latent-Hysteresis Graph ODEs: Modeling Coupled Topology-Feature Evolution via Continuous Phase Transitions 

**Authors**: Qinhan Hou, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24293)  

**Abstract**: Graph neural ordinary differential equations (Graph ODEs) extend graph learning from discrete message-passing layers to continuous-time representation flows. While it supports adaptive long-range propagation, we show that Graph ODEs with strictly positive irreducible mixing operators face an inherent \emph{monostability trap}: in the long-time regime, information leakage is unavoidable and the dynamics converge to a single global consensus attractor. We propose the \textbf{Hysteresis Graph ODE (HGODE)}, which couples feature evolution with a latent topological potential driven by a learned pairwise force. A double-well edge potential and bipolarized gate allow edge states to polarize into connected or insulated phases while preserving differentiability. We provide asymptotic analysis of the collapse mechanism and the proposed hysteretic topology dynamics, and validate HGODE on theory-driven synthetic diagnostics and real-world graph benchmarks. 

---
# RAS: a Reliability Oriented Metric for Automatic Speech Recognition 

**Authors**: Wenbin Huang, Yuhang Qiu, Bohan Li, Yiwei Guo, Jing Peng, Hankun Wang, Xie Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24278)  

**Abstract**: Automatic speech recognition systems often produce confident yet incorrect transcriptions under noisy or ambiguous conditions, which can be misleading for both users and downstream applications. Standard evaluation based on Word Error Rate focuses solely on accuracy and fails to capture transcription reliability. We introduce an abstention-aware transcription framework that enables ASR models to explicitly abstain from uncertain segments. To evaluate reliability under abstention, we propose RAS, a reliability-oriented metric that balances transcription informativeness and error aversion, with its trade-off parameter calibrated by human preference. We then train an abstention-aware ASR model through supervised bootstrapping followed by reinforcement learning. Our experiments demonstrate substantial improvements in transcription reliability while maintaining competitive accuracy. 

---
# Deep Learning-Enabled Dissolved Oxygen Sensing in Biofouling Environments for Ocean Monitoring 

**Authors**: Nikolaos Salaris, Adrien Desjardins, Manish K. Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2604.24236)  

**Abstract**: The escalating climate crisis and ecosystem degradation demand intelligent, low-cost sensors capable of robust, long-term monitoring in real-world environments. Absolute dissolved oxygen (DO) concentration is a key parameter for predicting climate tipping points. Inexpensive optoelectronic sensors based on microstructured polymer films doped with phosphorescent dyes could be readily deployable; however, signal drift and marine biofouling remain major challenges. Here, we introduce a sensing paradigm that combines camera-based DO sensors with a visual transformer (ViT)-based physics-informed neural network (PINN) for high-fidelity sensing under biofouling conditions. Training and testing data were obtained from an algae-laden water tank over 14 days to capture accelerated biofouling. The ViT-PINN, which embeds the Stern-Volmer (SV) equation into the loss function, reduces mean average error (MAE) by 92% and 89% compared to classical statistical and ML approaches, achieving ~2 umol/L absolute error. A deep ensemble further quantifies predictive uncertainty, enabling self-diagnostic sensing. 

---
# MEMCoder: Multi-dimensional Evolving Memory for Private-Library-Oriented Code Generation 

**Authors**: Mofei Li, Taozhi Chen, Guowei Yang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.24222)  

**Abstract**: Large Language Models (LLMs) excel at general code generation, but their performance drops sharply in enterprise settings that rely on internal private libraries absent from public pre-training corpora. While Retrieval-Augmented Generation (RAG) offers a training-free alternative by providing static API documentation, we find that such documentation typically provides only isolated definitions, leaving a fundamental knowledge gap. Specifically, LLMs struggle with a task-level lack of coordination patterns between APIs and an API-level misunderstanding of parameter constraints and boundary conditions. To address this, we propose MEMCoder, a novel framework that enables LLMs to autonomously accumulate and evolve Usage Guidelines across these two dimensions. MEMCoder introduces a Multi-dimensional Evolving Memory that captures distilled lessons from the model's own problem-solving trajectories. During inference, MEMCoder employs a dual-source retrieval mechanism to inject both static documentation and relevant historical guidelines into the context. The framework operates in an automated closed loop by using objective execution feedback to reflect on successes and failures, resolve knowledge conflicts, and dynamically update memory. Extensive evaluations on the NdonnxEval and NumbaEval benchmarks demonstrate that MEMCoder substantially enhances existing RAG systems, yielding an average absolute pass@1 gain of 16.31%. Furthermore, MEMCoder exhibits vastly superior domain-specific adaptation compared to existing memory-based continual learning methods. 

---
# RefEvo: Agentic Design with Co-Evolutionary Verification for Agile Reference Model Generation 

**Authors**: Yifan Zhang, Jianmin Ye, Jiahao Yang, Xi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24218)  

**Abstract**: As the complexity of System-on-Chip (SoC) designs grows, the shift-left paradigm necessitates the rapid development of high-fidelity reference models (typically written in SystemC) for early architecture exploration and verification. While Large Language Models (LLMs) show promise in code generation, their application to hardware modeling faces unique challenges: (1) Rigid, static workflows fail to adapt to varying design complexity, causing inefficiency; (2) Context window overflow in multi-turn interactions leads to catastrophic forgetting of critical specifications; and (3) the Coupled Validation Failure problem--where generated Testbenches (TBs) incorrectly validate flawed models due to correlated hallucinations--severely undermines reliability. To address these limitations, we introduce RefEvo, a dynamic multi-agent framework designed for agile and reliable reference modeling. RefEvo features three key innovations: (1) A Dynamic Design Planner that autonomously decomposes design specifications and constructs tailored execution workflows based on semantic complexity; (2) A Co-Evolutionary Verification Mechanism, which employs a Dialectical Arbiter to simultaneously rectify the model and verification logic against the specification (Spec) oracle, effectively mitigating false positives; and (3) A Spec Anchoring Strategy for lossless context compression. Evaluated on a diverse benchmark of 20 hardware modules, RefEvo achieves a 95% pass rate, outperforming static baselines by a large margin. Furthermore, our context optimization reduces token consumption by an average of 71.04%, achieving absolute savings of over 70,000 tokens per session for complex designs while maintaining 100% specification recall. 

---
# Agentic Witnessing: Pragmatic and Scalable TEE-Enabled Privacy-Preserving Auditing 

**Authors**: Antony Rowstron  

**Link**: [PDF](https://arxiv.org/pdf/2604.24203)  

**Abstract**: Auditing the semantic properties of proprietary data creates a fundamental tension: verification requires transparent access, while proprietary rights demand confidentiality. While Zero-Knowledge Proofs (ZKPs) ensure privacy, they are typically limited to precise algebraic constraints and are ill-suited for verifying qualitative, unstructured properties, such as the logic within a codebase. We propose {\em Agentic Witnessing}, a framework that moves verification from attested execution to {\em attested reasoning}. The system is composed of three agents: a Verifier (who wants to check properties of a dataset), a Prover (who owns the dataset) and an Auditor (that inspects the dataset). The Verifier is allowed to ask a limited number of simple binary true/false questions to the auditor. By isolating an LLM-based Auditor within a Trusted Execution Environment (TEE), the system enables the Verifier to query a Prover's private data via simple Boolean queries, without exposing the raw dataset. The Auditor uses the Model Context Protocol (MCP) to dynamically inspect the target dataset, producing a yes/no verdict accompanied by a cryptographic transcript: a signed hash chain binding the reasoning trace to both the original dataset and the TEE's hardware root of trust. We demonstrate this architecture by automating the artifact evaluation process for 21 peer-reviewed computer science papers with released codebases on GitHub (e.g. Does the codebase implement the system described in the paper?). We verified five high-level properties of these codebases described in the corresponding publications, treating the source code as private. Our results show that TEE-enabled agentic auditing provides a mechanism for privacy-preserving oversight, effectively decoupling qualitative verification from the need for data disclosure. 

---
# Speech Enhancement Based on Drifting Models 

**Authors**: Liang Xu, Diego Caviedes-Nozal, Bastiaan Kleijn, Longfei Felix Yan, Rasmus Kongsgaard Olsson  

**Link**: [PDF](https://arxiv.org/pdf/2604.24199)  

**Abstract**: We propose Speech Enhancement based on Drifting Models (DriftSE), a novel generative framework that formulates denoising as an equilibrium problem. Rather than relying on iterative sampling, DriftSE natively achieves one-step inference by evolving the pushforward distribution of a mapping function to directly match the clean speech distribution. This evolution is driven by a Drifting Field, a learned correction vector that guides samples toward the high-density regions of the clean distribution, which naturally facilitates training on unpaired data by matching distributions rather than paired samples. We investigate the framework under two formulations: a direct mapping from the noisy observation, and a stochastic conditional generative model from a Gaussian prior. Experiments on the VoiceBank-DEMAND benchmark demonstrate that DriftSE achieves high-fidelity enhancement in a single step, outperforming multi-step diffusion baselines and establishing a new paradigm for speech enhancement. 

---
# Rewarding the Scientific Process: Process-Level Reward Modeling for Agentic Data Analysis 

**Authors**: Zhisong Qiu, Shuofei Qiao, Kewei Xu, Yuqi Zhu, Lun Du, Ningyu Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.24198)  

**Abstract**: Process Reward Models (PRMs) have achieved remarkable success in augmenting the reasoning capabilities of Large Language Models (LLMs) within static domains such as mathematics. However, their potential in dynamic data analysis tasks remains underexplored. In this work, we first present a empirical study revealing that general-domain PRMs struggle to supervise data analysis agents. Specifically, they fail to detect silent errors, logical flaws that yield incorrect results without triggering interpreter exceptions, and erroneously penalize exploratory actions, mistaking necessary trial-and-error exploration for grounding failures. To bridge this gap, we introduce DataPRM, a novel environment-aware generative process reward model that (1) can serve as an active verifier, autonomously interacting with the environment to probe intermediate execution states and uncover silent errors, and (2) employs a reflection-aware ternary reward strategy that distinguishes between correctable grounding errors and irrecoverable mistakes. We design a scalable pipeline to construct over 8K high-quality training instances for DataPRM via diversity-driven trajectory generation and knowledge-augmented step-level annotation. Experimental results demonstrate that DataPRM improves downstream policy LLMs by 7.21% on ScienceAgentBench and 11.28% on DABStep using Best-of-N inference. Notably, with only 4B parameters, DataPRM outperforms strong baselines, and exhibits robust generalizability across diverse Test-Time Scaling strategies. Furthermore, integrating DataPRM into Reinforcement Learning yields substantial gains over outcome-reward baselines, achieving 78.73% on DABench and 64.84% on TableBench, validating the effectiveness of process reward supervision. Code is available at this https URL. 

---
# Seeing Is No Longer Believing: Frontier Image Generation Models, Synthetic Visual Evidence, and Real-World Risk 

**Authors**: Shuai Wu, Xue Li, Yanna Feng, Yufang Li, Zhijun Wang, Ran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24197)  

**Abstract**: Frontier image generation has moved from artistic synthesis toward synthetic visual evidence. Systems such as GPT Image 2, Nano Banana Pro, Nano Banana 2, Grok Imagine, Qwen Image 2.0 Pro, and Seedream 5.0 Lite combine photorealistic rendering, readable typography, reference consistency, editing control, and in several cases reasoning or search-grounded image construction. These capabilities create large benefits for design, education, accessibility, and communication, yet they also weaken one of society's most common trust shortcuts: the belief that a plausible picture is a reliable record. This paper provides a source-grounded technical and policy analysis of synthetic visual risk. We first summarize the public capabilities of recent image models, then analyze public incidents involving fake crisis images, celebrity and public-figure imagery, medical scans, forged-looking documents, synthetic screenshots, phishing assets, and market-moving rumors. We introduce a capability-weighted risk framework that links model affordances to real-world harm in finance, medicine, news, law, emergency response, identity verification, and civic discourse. Our findings show that risk is driven less by photorealism alone than by the convergence of realism, legible text, identity persistence, fast iteration, and distribution context. We argue for layered control: model-side restrictions, cryptographic provenance, visible labeling, platform friction, sector-grade verification, and incident response. The paper closes with practical recommendations for model providers, platforms, newsrooms, financial institutions, healthcare systems, legal organizations, regulators, and ordinary users. 

---
# MultiDx: A Multi-Source Knowledge Integration Framework towards Diagnostic Reasoning 

**Authors**: Yimin Deng, Zhenxi Lin, Yejing Wang, Guoshuai Zhao, Pengyue Jia, Zichuan Fu, Derong Xu, Yefeng Zheng, Xiangyu Zhao, Li Zhu, Xian Wu, Xueming Qian  

**Link**: [PDF](https://arxiv.org/pdf/2604.24186)  

**Abstract**: Diagnostic prediction and clinical reasoning are critical tasks in healthcare applications. While Large Language Models (LLMs) have shown strong capabilities in commonsense reasoning, they still struggle with diagnostic reasoning due to limited domain knowledge. Existing approaches often rely on internal model knowledge or static knowledge bases, resulting in knowledge insufficiency and limited adaptability, which hinder their capacity to perform diagnostic reasoning. Moreover, these methods focus solely on the accuracy of final predictions, overlooking alignment with standard clinical reasoning trajectories. To this end, we propose MultiDx, a two-stage diagnostic reasoning framework that performs differential diagnosis by analyzing evidence collected from multiple knowledge sources. Specifically, it first generates suspected diagnoses and reasoning paths by leveraging knowledge from web search, SOAP-formatted case, and clinical case database. Then it integrates multi-perspective evidence through matching, voting, and differential diagnosis to generate the final prediction.~Extensive experiments on two public benchmarks demonstrate the effectiveness of our approach. 

---
# MemeScouts@LT-EDI 2026: Asking the Right Questions -- Prompted Weak Supervision for Meme Hate Speech Detection 

**Authors**: Ivo Bueno, Lea Hirlimann, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2604.24179)  

**Abstract**: Detecting hate speech in memes is challenging due to their multimodal nature and subtle, culturally grounded cues such as sarcasm and context. While recent vision-language models (VLMs) enable joint reasoning over text and images, end-to-end prompting can be brittle, as a single prediction must resolve target, stance, implicitness, and irony. These challenges are amplified in multilingual settings. We propose a prompted weak supervision (PWS) approach that decomposes meme understanding into targeted, question-based labeling functions with constrained answer options for homophobia and transphobia detection in the LT-EDI 2026 shared task. Using a quantized Qwen3-VLM to extract features by answering targeted questions, our method outperforms direct VLM classification, with substantial gains for Chinese and Hindi, ranking 1st in English, 2nd in Chinese, and 3rd in Hindi. Iterative refinement via error-driven LF expansion and feature pruning reduces redundancy and improves generalization. Our results highlight the effectiveness of prompted weak supervision for multilingual multimodal hate speech detection. 

---
# Meta-Aligner: Bidirectional Preference-Policy Optimization for Multi-Objective LLMs Alignment 

**Authors**: Wenzhe Xu, Biao Liu, Yiyang Sun, Xin Geng, Ning Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24178)  

**Abstract**: Multi-Objective Alignment aims to align Large Language Models (LLMs) with diverse and often conflicting human values by optimizing multiple objectives simultaneously. Existing methods predominantly rely on static preference weight construction strategies. However, rigidly aligning to fixed targets discards valuable intermediate information, as training responses inherently embody valid preference trade-offs even when deviating from the target. To address this limitation, we propose Meal, i.e., MEta ALigner, a bi-level meta-learning framework enabling bidirectional optimization between preferences and policy responses, generating instructive dynamic preferences for steadier training. Specifically, we introduce a preference-weight-net as a meta-learner to generate adaptive preference weights based on input prompts and update the preference weights as learnable parameters, while the LLM policy acts as a base-learner optimizing response generation conditioned on these preferences with rejection sampling strategy. Extensive empirical results demonstrate that our method achieves superior performance on several multi-objective benchmarks, validating the effectiveness of the dynamic bidirectional preference-policy optimization framework. 

---
# AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models 

**Authors**: Yimin Deng, Yejing Wang, Zhenxi Lin, Zichuan Fu, Guoshuai Zhao, Derong Xu, Yefeng Zheng, Xiangyu Zhao, Xian Wu, Li Zhu, Xueming Qian  

**Link**: [PDF](https://arxiv.org/pdf/2604.24175)  

**Abstract**: Large language models have demonstrated strong reasoning capabilities in general knowledge question answering. However, their ability to handle temporal information remains limited. To address this limitation, existing approaches often involve external tools or manual verification and are tailored to specific scenarios, leading to poor generalizability. Moreover, these methods apply a fixed pipeline to all questions, overlooking the fact that different types of temporal questions require distinct reasoning strategies, which leads to unnecessary processing for simple cases and inadequate reasoning for complex ones. To this end, we propose AdapTime, an adaptive temporal reasoning method that dynamically executes reasoning steps based on the input context. Specifically, it involves three temporal reasoning actions: reformulate, rewrite and review, with an LLM planner guiding the reasoning process. AdapTime integrates seamlessly with state-of-the-art LLMs and significantly enhances their temporal reasoning capabilities without relying on external support. Extensive experiments demonstrate the effectiveness of our approach. 

---
# Defusing the Trigger: Plug-and-Play Defense for Backdoored LLMs via Tail-Risk Intrinsic Geometric Smoothing 

**Authors**: Kaisheng Fan, Weizhe Zhang, Yishu Gao, Tegawendé F. Bissyandé, Xunzhu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24162)  

**Abstract**: Defending against backdoor attacks in large language models remains a critical practical challenge. Existing defenses mitigate these threats but typically incur high preparation costs and degrade utility via offline purification, or introduce severe latency via complex online interventions. To overcome this dichotomy, we present Tail-risk Intrinsic Geometric Smoothing (TIGS), a plug-and-play inference-time defense requiring no parameter updates, external clean data, or auxiliary generation. TIGS leverages the observation that successful backdoor triggers consistently induce localized attention collapse within the semantic content region. Operating entirely within the native forward pass, TIGS first performs content-aware tail-risk screening to identify suspicious attention heads and rows using sample-internal signals. It then applies intrinsic geometric smoothing: a weak content-domain correction preserves semantic anchoring, while a stronger full-row contraction disrupts trigger-dominant routing. Finally, a controlled full-row write-back reconstructs the attention matrix to ensure inference stability. Extensive evaluations demonstrate that TIGS substantially suppresses attack success rates while strictly preserving clean reasoning and open-ended semantic consistency. Crucially, this favorable security-utility-latency equilibrium persists across diverse architectures, including dense, reasoning-oriented, and sparse mixture-of-experts models. By structurally disrupting adversarial routing with marginal latency overhead, TIGS establishes a highly practical, deployment-ready defense standard for state-of-the-art LLMs. 

---
# Strategic Bidding in 6G Spectrum Auctions with Large Language Models 

**Authors**: Ismail Lotfi, Ali Ghrayeb  

**Link**: [PDF](https://arxiv.org/pdf/2604.24156)  

**Abstract**: Efficient and fair spectrum allocation is a central challenge in 6G networks, where massive connectivity and heterogeneous services continuously compete for limited radio resources. We investigate the use of Large Language Models (LLMs) as bidding agents in repeated 6G spectrum auctions with budget constraints in vehicular networks. Each user equipment (UE) acts as a rational player optimizing its long-term utility through repeated interactions. Using the Vickrey-Clarke-Groves (VCG) mechanism as a benchmark for incentive-compatible, dominant-strategy truthfulness, we compare LLM-guided bidding against truthful and heuristic strategies. Unlike heuristics, LLMs leverage historical outcomes and prompt-based reasoning to adapt their bidding behavior dynamically. Results show that when the theoretical assumptions guaranteeing truthfulness hold, LLM bidders recover near-equilibrium outcomes consistent with VCG predictions. However, when these assumptions break -- such as under static budget constraints -- LLMs sustain longer participation and achieve higher utilities, revealing their ability to approximate adaptive equilibria beyond static mechanism design. This work provides the first systematic evaluation of LLM bidders in repeated spectrum auctions, offering new insights into how AI-driven agents can interact strategically and reshape market dynamics in future 6G networks. 

---
# The Alignment Target Problem: Divergent Moral Judgments of Humans, AI Systems, and Their Designers 

**Authors**: Benjamin Minhao Chen, Xinyu Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.24155)  

**Abstract**: The quest to align machine behavior with human values raises fundamental questions about the moral frameworks that should govern AI decision-making. Much alignment research assumes that the appropriate benchmark is how humans themselves would act in a given situation. Research into agent-type value forks has challenged this assumption by showing that people do not always hold AI systems to the same moral standards as humans. Yet this challenge is subject to two further questions: whether people evaluate AI behavior differently when its human origins are made visible, and whether people hold the humans who program AI systems to different moral standards than either the humans or the machines under evaluation. An experimental study on 1,002 U.S. adults measured moral judgments in a runaway mine train scenario, varying the subject of evaluation across four conditions: a repairman, a repair robot, a repair robot programmed by company engineers, and company engineers programming a repair robot. We find no significant variation in the moral standards applied to the repairman and the robot. However, moral judgments shifted substantially when robot actions were described as the product of human design. Participants exhibited markedly more deontological reasoning when evaluating the robot programmed by engineers or the engineers programming it, suggesting that making human design visible activates heightened moral constraints. These findings provide evidence that people apply meaningfully different moral standards to AI systems, to humans acting in the same situation, and to the humans who design them. We call this divergence the alignment target problem. Whether these plural normative standards can be reconciled into a coherent framework for AI governance in high-stakes domains remains an open question. 

---
# Progressive Approximation in Deep Residual Networks: Theory and Validation 

**Authors**: Wei Wang, Xiao-Yong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.24154)  

**Abstract**: The Universal Approximation Theorem (UAT) guarantees universal function approximation but does not explain how residual models distribute approximation across layers. We reframe residual networks as a layer-wise approximation process that builds an approximation trajectory from input to target, and prove the existence of progressive trajectories where error decreases monotonically with depth. It reveals that residual networks can implement structured, step-by-step refinement rather than end-to-end (E2E) black-box mapping. Building on this, we propose Layer-wise Progressive Approximation (LPA), a theoretically grounded training principle that explicitly aligns each layer with its residual target to realize such trajectories. LPA is architecture-agnostic: we observe progressive behavior in residual FNNs, ResNets, and Transformers across tasks including complex surface fitting, image classification, and NLP with LLMs for generation and classification. Crucially, this enables ``train once, use $N$ models": a single network yields useful predictions at every depth, supporting efficient shallow inference without retraining. Our work unifies approximation theory with practical deep learning, providing a new lens on representation learning and a flexible framework for multi-depth deployment. The source code will be released unpon acceptance at https://(open\_upon\_acceptance). 

---
# Leveraging Human Feedback for Semantically-Relevant Skill Discovery 

**Authors**: Maxence Hussonnois, Thommen George Karimpanal, Santu Rana  

**Link**: [PDF](https://arxiv.org/pdf/2604.24127)  

**Abstract**: Unsupervised skill discovery in reinforcement learning aims to intrinsically motivate agents to discover diverse and useful behaviours. However, unconstrained approaches can produce unsafe, unethical, or misaligned behaviours. To mitigate these risks and improve the practical desireability of discovered skills, recent work grounds the discovery process by leveraging human preference feedback. However, preference-based approaches are feedback-inefficient and inherently ill-equipped to deal with skill spaces composed of a variety of different skills such as running, jumping, walking, etc. To overcome this limitation, we introduce semantic labelling, a novel and feedback-efficient approach that leverages human cognitive strengths to identify and label semantically meaningful behaviours. Based on semantic labelling, we propose Semantically Relevant Skill Discovery (SRSD), a novel human-in-the-loop approach that collects semantic labels from human feedback and learns a reward function to encourage skills to be more semantically diverse and relevant. Through our experiments in a 2D navigation environment and four locomotion environments, we demonstrate that SRSD can improve semantic diversity and discover relevant behaviours while scaling effectively to a large variety of behaviours. 

---
# Latency and Cost of Multi-Agent Intelligent Tutoring at Scale 

**Authors**: Iizalaarab Elhaimeur, Nikos Chrisochoides  

**Link**: [PDF](https://arxiv.org/pdf/2604.24110)  

**Abstract**: Multi-agent LLM tutoring systems improve response quality through agent specialization, but each student query triggers several concurrent API calls whose latencies compound through a parallel-phase maximum effect that single-agent systems do not face. We instrument ITAS, a four-agent tutoring system built on Gemini 2.5 Flash and Google Vertex AI, across three throughput tiers (Standard PayGo, Priority PayGo, and Provisioned Throughput) and eleven concurrency levels up to 50 simultaneous users, producing over 3,000 requests drawn from a live graduate STEM deployment. Priority PayGo maintains flat sub-4-second response times across the full load range; Standard PayGo degrades substantially under classroom-scale concurrency; and Provisioned Throughput delivers the lowest latency at low concurrency but saturates its reserved capacity above approximately 20 concurrent users. Cost analysis places both pay-per-token tiers well below the price of a STEM textbook per student per semester under a worst-case usage ceiling. Provisioned Throughput, expensive under continuous provisioning, becomes cost-competitive for institutions that can predict and concentrate their traffic toward high utilization. These results provide concrete tier-selection guidance across deployment scales from a single seminar to a university-wide rollout. 

---
# Meta-Ensemble Learning with Diverse Data Splits for Improved Respiratory Sound Classification 

**Authors**: June-Woo Kim, Miika Toikkanen, Heejoon Koo, Yoon Tae Kim, Doyoung Kwon, Kyunghoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.24096)  

**Abstract**: Training reliable respiratory sound classification models remains challenging due to the limited size and subject diversity of datasets. Ensemble methods can improve robustness, but when base models are trained on identical data, models tend to overfit and produce highly correlated predictions, thereby reducing the effectiveness of ensembling. In this work, we investigate a meta-ensemble learning methodology that enhances prediction diversity by training base models on diverse data splits and combining their outputs through a trained meta-model. Specifically, we train base models on the ICBHI dataset using two data split settings: fixed 80-20% split and five-fold cross-validation split, under two data granularity settings: patient- and sample-level. The resulting diversity in base model predictions enables the meta-model to better generalize. Our approach achieves new state-of-the-art performance on the ICBHI benchmark, reaching a Score of 66.49% and showing improved generalization on two out-of-distribution datasets, indicating its potential applicability to real-world clinical data. 

---
# TACO: Efficient Communication Compression of Intermediate Tensors for Scalable Tensor-Parallel LLM Training 

**Authors**: Man Liu, Xingchen Liu, Xingjian Tian, Bing Lu, Shengkay Lyu, Shengquan Yin, Wenjing Huang, Zheng Wei, Hairui Zhao, Guangming Tan, Dingwen Tao  

**Link**: [PDF](https://arxiv.org/pdf/2604.24088)  

**Abstract**: Handling communication overhead in large-scale tensor-parallel training remains a critical challenge due to the dense, near-zero distributions of intermediate tensors, which exacerbate errors under frequent communication and introduce significant computational overhead during compression. To this end, we propose TACO (Tensor-parallel Adaptive COmmunication compression), a robust FP8-based framework for compressing TP intermediate tensors. First, we employ a data-driven reshaping strategy combined with an Adaptive Scale-Hadamard Transform to enable high-fidelity FP8 quantization, while its Dual-Scale Quantization mechanism ensures numerical stability throughout training. Second, we design a highly fused compression operator to reduce memory traffic and kernel launch overhead, allowing efficient overlap with communication. Finally, we integrate TACO with existing state-of-the-art methods for Data and Pipeline Parallelism to develop a compression-enabled 3D-parallel training framework. Detailed experiments on GPT models and Qwen model demonstrate up to 1.87X end-to-end throughput improvement while maintaining near-lossless accuracy, validating the effectiveness and efficiency of TACO in large-scale training. 

---
# AsyncShield: A Plug-and-Play Edge Adapter for Asynchronous Cloud-based VLA Navigation 

**Authors**: Kai Yang, Zedong Chu, Yingnan Guo, Zhengbo Wang, Shichao Xie, Yanfen Shen, Xiaolong Wu, Xing Li, Mu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24086)  

**Abstract**: While Vision-Language-Action (VLA) models have been demonstrated possessing strong zero-shot generalization for robot control, their massive parameter sizes typically necessitate cloud-based deployment. However, cloud deployment introduces network jitter and inference latency, which can induce severe spatiotemporal misalignment in mobile navigation under continuous displacement, so that the stale intents expressed in past ego frames may become spatially incorrect in the current frame and lead to collisions. To address this issue, we propose AsyncShield, a plug-and-play asynchronous control framework. AsyncShield discards traditional black-box time-series prediction in favor of a deterministic physical white-box spatial mapping. By maintaining a temporal pose buffer and utilizing kinematic transformations, the system accurately converts temporal lag into spatial pose offsets to restore the VLA's original geometric intent. To balance intent restoration fidelity and physical safety, the edge adaptation is formulated as a constrained Markov decision process (CMDP). Solved via the PPO-Lagrangian algorithm, a reinforcement learning adapter dynamically trades off between tracking the VLA intent and responding to high-frequency LiDAR obstacle avoidance hard constraints. Furthermore, benefiting from a standardized universal sub-goal interface, domain randomization, and perception-level adaptation via Collision Radius Inflation, AsyncShield operates as a lightweight, plug-and-play module. Simulation and real-world experiments demonstrate that, without fine-tuning any cloud-based foundation models, the framework exhibits zero-shot and robust generalization capabilities, effectively improving the success rate and physical safety of asynchronous navigation. 

---
# Jailbreaking Frontier Foundation Models Through Intention Deception 

**Authors**: Xinhe Wang, Katia Sycara, Yaqi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.24082)  

**Abstract**: Large (vision-)language models exhibit remarkable capability but remain highly susceptible to jailbreaking. Existing safety training approaches aim to have the model learn a refusal boundary between safe and unsafe, based on the user's intent. It has been found that this binary training regime often leads to brittleness, since the user intent cannot reliably be evaluated, especially if the attacker obfuscates their intent, and also makes the system seem unhelpful. In response, frontier models, such as GPT-5, have shifted from refusal-based safeguards to safe completion, that aims to maximize helpfulness while obeying safety constraints. However, safe completion could be exploited when a user pretends their intention is benign. Specifically, this intent inversion would be effective in multi-turn conversation, where the attacker has multiple opportunities to reinforce their deceptively benign intent. In this work, we introduce a novel multi-turn jailbreaking method that exploits this vulnerability. Our approach gradually builds conversational trust by simulating benign-seeming intentions and by exploiting the consistency property of the model, ultimately guiding the target model toward harmful, detailed outputs. Most crucially, our approach also uncovered an additional class of model vulnerability that we call para-jailbreaking that has been unnoticed up to now. Para-jailbreaking describes the situation where the model may not reveal harmful direct reply to the attack query, however the information that it reveals is nevertheless harmful. Our contributions are threefold. First, it achieves high success rates against frontier models including GPT-5-thinking and Claude-Sonnet-4.5. Second, our approach revealed and addressed para-jailbreaking harmful output. Third, experiments on multimodal VLM models showed that our approach outperformed state-of-the-art models. 

---
# The Pragmatic Persona: Discovering LLM Persona through Bridging Inference 

**Authors**: Jisoo Yang, Jongwon Ryu, Minuk Ma, Trung X. Pham, Junyeong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.24079)  

**Abstract**: Large Language Models (LLMs) reveal inherent and distinctive personas through dialogue. However, most existing persona discovery approaches rely on surface-level lexical or stylistic cues, treating dialogue as a flat sequence of tokens and failing to capture the deeper discourse-level structures that sustain persona consistency. To address this limitation, we propose a novel analytical framework that interprets LLM dialogue through bridging inference -- implicit conceptual relations that connect utterances via shared world knowledge and discourse coherence. By modeling these relations as structured knowledge graphs, our approach captures latent semantic links that govern how LLMs organize meaning across turns, enabling persona discovery at the level of discourse coherence rather than surface realizations. Experimental results across multiple reasoning backbones and target LLMs, ranging from small-scale models to 80B-parameter systems, demonstrate that bridging-inference graphs yield significantly stronger semantic coherence and more stable persona identification than frequency or style-based baselines. These results show that persona traits are consistently encoded in the structural organization of discourse rather than isolated lexical patterns. This work presents a systematic framework for probing, extracting, and visualizing latent LLM personas through the lens of Cognitive Discourse Theory, bridging computational linguistics, cognitive semantics, and persona reasoning in large language models. Codes are available at this https URL 

---
# FreeScale: Distributed Training for Sequence Recommendation Models with Minimal Scaling Cost 

**Authors**: Chenhao Feng, Haoli Zhang, Shakhzod Ali-Zade, Yanli Zhao, Liang Luo, Jennifer Cao, Lisen Deng, Siqiao Chen, Chenyu Zhao, Tristan Rice, Daniel Johnson, Min Si, Tiantu Xu, Yi Zhang, Siqi Yan, Chuanhao Zhuge, Min Ni, Bi Xue, Qunshu Zhang, Shen Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.24073)  

**Abstract**: Modern industrial Deep Learning Recommendation Models typically extract user preferences through the analysis of sequential interaction histories, subsequently generating predictions based on these derived interests. The inherent heterogeneity in data characteristics frequently result in substantial under-utilization of computational resources during large-scale training, primarily due to computational bubbles caused by severe stragglers and slow blocking communications. This paper introduces FreeScale, a solution designed to (1) mitigate the straggler problem through meticulously load balanced input samples (2) minimize the blocking communication by overlapping prioritized embedding communications with computations (3) resolve the GPU resource competition during computation and communication overlapping by communicating through SM-Free techniques. Empirical evaluation demonstrates that FreeScale achieves up to 90.3% reduction in computational bubbles when applied to real-world workloads running on 256 H100 GPUs. 

---
# Distilling Self-Consistency into Verbal Confidence: A Pre-Registered Negative Result and Post-Hoc Rescue on Gemma 3 4B 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.24070)  

**Abstract**: Small instruct-tuned LLMs produce degenerate verbal confidence under minimal elicitation: ceiling rates above 95%, near-chance Type-2 AUROC, and Invalid validity profiles. We test whether confidence-conditioned supervised fine-tuning (CSFT) with self-consistency-derived targets can close the gap between internal information and verbal readout. A pre-registered Phase 0 protocol on Gemma 3 4B-it with a modal filter restricting training to items with correct modal answers produced a negative result: AUROC2 dropped from 0.554 to 0.509 due to label-entropy collapse in the training targets. An exploratory rescue removed the filter, training on all 2,000 calibration items. This produced a binary verbal correctness discriminator with AUROC2 = 0.774 on held-out TriviaQA, compressing a 10-sample self-consistency signal (AUROC2 = 0.999) into a single-pass readout exceeding logit entropy (0.701). The shuffled-target control showed no improvement (0.501). On MMLU, accuracy improved from 54.2% to 77.4% with the shuffled model at baseline (56.1%), supporting a target-dependent interpretation. The result is exploratory, binary rather than continuously calibrated, and observed at a single scale. It identifies two design lessons: confidence training requires label entropy, and correct targets regularise output format. 

---
# QEVA: A Reference-Free Evaluation Metric for Narrative Video Summarization with Multimodal Question Answering 

**Authors**: Woojun Jung, Junyeong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.24052)  

**Abstract**: Video-to-text summarization remains underexplored in terms of comprehensive evaluation methods. Traditional n-gram overlap-based metrics and recent large language model (LLM)-based approaches depend heavily on human-written reference summaries, limiting their practicality and sensitivity to nuanced semantic aspects. In this paper, we propose QEVA, a reference-free metric evaluating candidate summaries directly against source videos through multimodal question answering. QEVA assesses summaries along three clear dimensions: Coverage, Factuality, and Chronology. We also introduce MLVU(VS)-Eval, a new annotated benchmark derived from the MLVU dataset, comprising 800 summaries generated from 200 videos using state-of-the-art video-language multimodal models. This dataset establishes a transparent and consistent framework for evaluation. Experimental results demonstrate that QEVA shows higher correlation with human judgments compared to existing approaches, as measured by Kendall's $\tau_b$, $\tau_c$, and Spearman's $\rho$. We hope that our benchmark and metric will facilitate meaningful progress in video-to-text summarization research and provide valuable insights for the development of future evaluation methods. 

---
# End-to-End Learning for Partially-Observed Time Series with PyPOTS 

**Authors**: Wenjie Du, Yiyuan Yang, Tianxiang Zhan, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2604.24041)  

**Abstract**: Partially-observed time series (POTS) is ubiquitous in real-world applications, yet most existing toolchains separate missing-value handling from downstream learning, which limits reproducibility and overall performance. This tutorial introduces PyPOTS, an open-source Python ecosystem for end-to-end data mining and machine learning on POTS. We present practical workflows spanning missingness simulation, data preprocessing, model training, and evaluation across core tasks, including imputation, forecasting, classification, clustering, and anomaly detection. The tutorial consists of two parts: Part I emphasizes hands-on application for practitioners through unified APIs and benchmark-oriented experiments. Part II targets developers and researchers, focusing on extending PyPOTS with custom models, domain-specific constraints, and contribution-ready engineering practices. Participants will gain both conceptual understanding and implementation experience for building robust, transparent, and reusable POTS pipelines in research and production settings. PyPOTS is publicly available at this https URL 

---
# Improving Robustness of Tabular Retrieval via Representational Stability 

**Authors**: Kushal Raj Bhandari, Adarsh Singh, Jianxi Gao, Soham Dan, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.24040)  

**Abstract**: Transformer-based table retrieval systems flatten structured tables into token sequences, making retrieval sensitive to the choice of serialization even when table semantics remain unchanged. We show that semantically equivalent serializations, such as $\texttt{csv}$, $\texttt{tsv}$, $\texttt{html}$, $\texttt{markdown}$, and $\texttt{ddl}$, can produce substantially different embeddings and retrieval results across multiple benchmarks and retriever families. To address this instability, we treat serialization embedding as noisy views of a shared semantic signal and use its centroid as a canonical target representation. We show that centroid averaging suppresses format-specific variation and can recover the semantic content common to different serializations when format-induced shifts differ across tables. Empirically, centroid representations outrank individual formats in aggregate pairwise comparisons across $\texttt{MPNet}$, $\texttt{BGE-M3}$, $\texttt{ReasonIR}$, and $\texttt{SPLADE}$. We further introduce a lightweight residual bottleneck adapter on top of a frozen encoder that maps single-serialization embeddings towards centroid targets while preserving variance and enforcing covariance regularization. The adapter improves robustness for several dense retrievers, though gains are model-dependent and weaker for sparse lexical retrieval. These results identify serialization sensitivity as a major source of retrieval variance and show the promise of post hoc geometric correction for serialization-invariant table retrieval. Our code, datasets, and models are available at $\href{this https URL}{this https URL}$. 

---
# AgenticCache: Cache-Driven Asynchronous Planning for Embodied AI Agents 

**Authors**: Hojoon Kim, Yuheng Wu, Thierry Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2604.24039)  

**Abstract**: Embodied AI agents increasingly rely on large language models (LLMs) for planning, yet per-step LLM calls impose severe latency and cost. In this paper, we show that embodied tasks exhibit strong plan locality, where the next plan is largely predictable from the current one. Building on this, we introduce AgenticCache, a planning framework that reuses cached plans to avoid per-step LLM calls. In AgenticCache, each agent queries a runtime cache of frequent plan transitions, while a background Cache Updater asynchronously calls the LLM to validate and refine cached entries. Across four multi-agent embodied benchmarks, AgenticCache improves task success rate by 22% on average across 12 configurations (4 benchmarks x 3 models), reduces simulation latency by 65%, and lowers token usage by 50%. Cache-based plan reuse thus offers a practical path to low-latency, low-cost embodied agents. Code is available at this https URL. 

---
# From Skill Text to Skill Structure: The Scheduling-Structural-Logical Representation for Agent Skills 

**Authors**: Qiliang Liang, Hansi Wang, Zhong Liang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24026)  

**Abstract**: LLM agents increasingly rely on reusable skills, capability packages that combine instructions, control flow, constraints, and tool calls. In most current agent systems, however, skills are still represented by text-heavy artifacts, including this http URL-style documents and structured records whose machine-usable evidence remains embedded largely in natural-language descriptions. This poses a challenge for skill-centered agent systems: managing skill collections and using skills to support agent both require reasoning over invocation interfaces, execution structure, and concrete side effects that are often entangled in a single textual surface. An explicit representation of skill knowledge may therefore help make these artifacts easier for machines to acquire and leverage. Drawing on Memory Organization Packets, Script Theory, and Conceptual Dependency from Schank and Abelson's classical work on linguistic knowledge representation, we introduce what is, to our knowledge, the first structured representation for agent skill artifacts that disentangles skill-level scheduling signals, scene-level execution structure, and logic-level action and resource-use evidence: the Scheduling-Structural-Logical (SSL) representation. We instantiate SSL with an LLM-based normalizer and evaluate it on a corpus of skills in two tasks, Skill Discovery and Risk Assessment, and superiorly outperform the text-only baselines: in Skill Discovery, SSL improves MRR from 0.573 to 0.707; in Risk Assessment, it improves macro F1 from 0.744 to 0.787. These findings reveal that explicit, source-grounded structure makes agent skills easier to search and review. They also suggest that SSL is best understood as a practical step toward more inspectable, reusable, and operationally actionable skill representations for agent systems, rather than as a finished standard or an end-to-end mechanism for managing and using skills. 

---
# Poster: ClawdGo: Endogenous Security Awareness Training for Autonomous AI Agents 

**Authors**: Jiaqi Li, Yang Zhao, Bin Sun, Yang Yu, Jian Chang, Lidong Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2604.24020)  

**Abstract**: Autonomous AI agents deployed on platforms such as OpenClaw face prompt injection, memory poisoning, supply-chain attacks, and social engineering, yet existing defences address only the platform perimeter, leaving the agent's own threat judgement entirely untrained. We present ClawdGo, a framework for endogenous security awareness training: we teach the agent to recognise and reason about threats from the inside, at inference time, with no model modification. Four contributions are introduced: TLDT (Three-Layer Domain Taxonomy) organises 12 trainable dimensions across Self-Defence, Owner-Protection, and Enterprise-Security layers; ASAT (Autonomous Security Awareness Training) is a self-play loop where the agent alternates attacker, defender, and evaluator roles under weakest-first curriculum scheduling; CSMA (Cross-Session Memory Accumulation) compounds skill gains via a four-layer persistent memory architecture and Axiom Crystallisation Promotion (ACP); and SACP (Security Awareness Calibration Problem) formalises the precision-recall tradeoff introduced by endogenous training. Live experiments show weakest-first ASAT raises average TLDT score from 80.9 to 96.9 over 16 sessions, outperforming uniform-random scheduling by 6.5 points and covering 11 of 12 dimensions. CSMA retains the full gain across sessions; cold-start ablation recovers only 2.4 points, leaving a 13.6-point gap. E-mode generates 32 TLDT-conformant scenarios covering all 12 dimensions. SACP is observed when a heavily trained agent classifies a legitimate capability assessment as prompt injection (30/160). 

---
# TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents 

**Authors**: Jiaqi Wang, Wenhao Zhang, Weijie Shi, Yaliang Li, James Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.24005)  

**Abstract**: On-policy distillation (OPD) has shown strong potential for transferring reasoning ability from frontier or domain-specific models to smaller students. While effective on static single-turn tasks, its behavior in multi-turn agent settings remains underexplored. In this work, we identify a key limitation of vanilla OPD in such settings, which we term Trajectory-Level KL Instability. Specifically, we observe that KL divergence increases together with a drop in success rate, and even after convergence, the KL remains high, leading to unstable training. This instability arises from inter-turn error compounding: as errors accumulate, the student is driven beyond the teacher's effective support, rendering the supervision signal unreliable. To address this, we propose TCOD (Temporal Curriculum On-Policy Distillation), a simple yet effective framework that controls the trajectory depth exposed to the student and progressively expands it from short to long with a curriculum this http URL results across four student-teacher pairs on three multi-turn agent benchmarks (ALFWorld, WebShop, ScienceWorld) show that TCOD mitigates KL escalation and enhances KL stability throughout training, improving agent performance by up to 18 points over vanilla OPD. Further evaluations show that TCOD can even surpass the teacher's performance and generalize to tasks on which the teacher fails. 

---
# IntentVLM: Open-Vocabulary Intention Recognition through Forward-Inverse Modeling with Video-Language Models 

**Authors**: Hamed Rahimi, Clemence Grislain, Adrien Jacquet Cretides, Olivier Sigaud, Mohamed Chetouani  

**Link**: [PDF](https://arxiv.org/pdf/2604.24002)  

**Abstract**: Improving the effectiveness of human-robot interaction requires social robots to accurately infer human goals through robust intention understanding. This challenge is particularly critical in multimodal settings, where agents must integrate heterogeneous signals including text, visual cues to form a coherent interpretation of user intent. This paper presents IntentVLM, a novel two-stage video-language framework designed for open-vocabulary human intention recognition. The approach is inspired by forward-inverse modeling in cognitive science by decomposing intention understanding into goal candidate generation followed by structured inference through selection, effectively reducing hallucinations in latent reasoning. Evaluated on the IntentQA and Inst-IT Bench datasets, IntentVLM achieves state-of-the-art results with up to 80% accuracy, notably surpassing the baseline performance by 30% and matches human performance. Our findings demonstrate that this structured reasoning approach enhances open-vocabulary intention understanding without catastrophic forgetting, offering a robust foundation for human-centered robotics. 

---
# EPM-RL: Reinforcement Learning for On-Premise Product Mapping in E-Commerce 

**Authors**: Minhyeong Yu, Wonduk Seo  

**Link**: [PDF](https://arxiv.org/pdf/2604.23993)  

**Abstract**: Product mapping, the task of deciding whether two e-commerce listings refer to the same product, is a core problem for price monitoring and channel visibility. In real marketplaces, however, sellers frequently inject promotional keywords, platform-specific tags, and bundle descriptions into titles, causing the same product to appear under many different names. Recent LLM-based and multi-agent frameworks improve robustness and interpretability on such hard cases, but they often rely on expensive external APIs, repeated retrieval, and complex inference-time orchestration, making large-scale deployment costly and difficult in privacy-sensitive enterprise settings. To address these issues, we present EPM-RL, a reinforcement-learning-based framework for building an accurate and efficient on-premise e-commerce product mapping model. Our central idea is to distill high-cost agentic reasoning into a trainable in-house model. Starting from a curated set of product pairs with LLM-generated rationales and human verification, we first perform parameter-efficient fine-tuning (PEFT) on a small student model using structured reasoning outputs. We then further optimize the model with Reinforcement Learning (RL) using an agent-based reward that jointly evaluates output-format compliance, label correctness, reasoning--preference scores from specially designed judge models. Preliminary results show that EPM-RL consistently improves over PEFT-only training and offers a stronger quality--cost trade-off than commercial API-based baselines, while enabling private deployment and lower operational cost. These findings suggest that reinforcement learning can turn product mapping from a high-latency agentic pipeline into a scalable, inspectable, and production-ready in-house system. 

---
# Fix Initial Codes and Iteratively Refine Textual Directions Toward Safe Multi-Turn Code Correction 

**Authors**: Yuto Tanaka, Issei Sato  

**Link**: [PDF](https://arxiv.org/pdf/2604.23989)  

**Abstract**: Recent work on large language models (LLMs) has emphasized the importance of scaling inference compute. From this perspective, the state-of-the-art method Scattered Forest Search (SFS) has been proposed, employing Monte Carlo Tree Search with carefully crafted initial seeds and textual optimization for multi-turn code correction. However, its complexity makes it unclear what factors contribute to improvements in inference performance. To address this problem, we analyze SFS and propose a simpler method, Iterative Refinement of Textual Directions (IRTD), which fixes initial codes and iteratively refines textual directions. Because of the simplicity of IRTD, we theoretically establish the safety of IRTD using Oracle-Guided Inductive Synthesis (OGIS). Experiments on several code generation benchmarks suggest that IRTD achieves inference performance comparable to state-of-the-art methods. These results indicate that, even without complex search structures, refining initial codes with high-quality textual directions alone can effectively improve inference performance. 

---
# Hindsight Preference Optimization for Financial Time Series Advisory 

**Authors**: Yanwei Cui, Guanghui Wang, Xing Zhang, Peiyang He, Ziyuan Li, Bing Zhu, Wei Qiu, Xusheng Wang, Zheng Yu, Anqi Xin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23988)  

**Abstract**: Time series models predict numbers; decision-makers need advisory -- directional signals with reasoning, actionable suggestions, and risk management. Training language models for such predictive advisory faces a fundamental challenge: quality depends on outcomes unknown at prediction time. We bridge two ideas from reinforcement learning -- using information unavailable during execution to retrospectively generate training signal, and preference alignment -- and propose Hindsight Preference Optimization: observed outcomes let an LLM judge rank candidate advisories on dimensions that scalar metrics cannot capture, producing preference pairs for DPO without human annotation. We apply this to Vision-Language-Model-based predictive advisories on S&P 500 equity time series, demonstrated by a 4B model outperforming its 235B teacher on both accuracy and advisory quality. 

---
# Quantum Knowledge Graph: Modeling Context-Dependent Triplet Validity 

**Authors**: Yao Wang, Zixu Geng, Jun Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.23972)  

**Abstract**: Knowledge graphs (KGs) are increasingly used to support large lan guage model (LLM) reasoning, but standard triplet-based KGs treat each relation as globally valid. In many settings, whether a relation should count as evidence depends on the context. We therefore formulate triplet validity as a triplet-specific function of context and refer to this formulation as a Quantum Knowledge Graph (QKG). We instantiate QKG in medicine using a diabetes-centered PrimeKG subgraph, whose 68,651 context-sensitive relations are further annotated with patient-group-specific constraints. We evaluate it in a reasoner--validator pipeline for medical question answering on a KG-grounded subset of MedReason containing 2,788 questions. With Haiku-4.5 as both the Reasoner and the Validator, KG-backed validation significantly improves over a no-validator baseline ($+0.61$ pp), and QKG with context matching yields the largest gain, outperforming both KG validation without context matching ($+0.79$ pp) and the no-validator baseline ($+1.40$ pp; paired McNemar, all $p<0.05$). Under a stronger validator (Qwen-3.6-Plus), the raw QKG gain over the no-validator baseline grows from $+1.40$ pp to $+5.96$ pp; the context-matching gap is non-significant ($p=0.73$) on the raw set but becomes borderline significant ($p=0.05$) after adjustment for knowledge leakage and suspicious questions, consistent with a benchmark-gold ceiling rather than a QKG limitation. Taken together, the results support the view that the value of a KG in LLM-based clinical reasoning lies not merely in storing medically related facts, but in representing whether those facts are applicable to the specific patient context. For reproducibility and further research, we release the curated QKG datasets and source code.\footnote{this https URL} 

---
# DecompKAN: Decomposed Patch-KAN for Long-Term Time Series Forecasting 

**Authors**: Naveen Mysore  

**Link**: [PDF](https://arxiv.org/pdf/2604.23968)  

**Abstract**: Accurate time series forecasting in scientific domains such as climate modeling, physiological monitoring, and energy systems benefits from both competitive predictions and model transparency. This work proposes DecompKAN, a lightweight attention-free architecture that combines trend-residual decomposition, channel-wise patching, learned instance normalization, and B-spline Kolmogorov-Arnold Network (KAN) edge functions. Each KAN edge learns an explicit, inspectable 1D scalar function over learned patch-embedding coordinates that can be directly visualized. On standard benchmarks, DecompKAN achieves best or tied-best MSE on 15 of 32 dataset-horizon combinations among selected published baselines, and achieves best or tied-best MSE on 20 of 36 comparisons under a controlled same-recipe evaluation across 9 datasets including the physiological PPG-DaLiA benchmark. The architecture shows particular strength on datasets with smooth temporal dynamics (Solar -17%, ECL -10% vs. iTransformer, Weather) and physiological time series. Visualization of learned edge functions reveals qualitatively different latent nonlinearities across domains. Ablation analysis shows that the architectural pipeline (decomposition, patching, normalization) drives performance more than the choice of nonlinear layer, while the KAN formulation enables inspection of learned latent transformations. 

---
# Task-guided Spatiotemporal Network with Diffusion Augmentation for EEG-based Dementia Diagnosis and MMSE Prediction 

**Authors**: Xiaoyu Zheng, Xu Tian, Bin Jiao, Kunbo Cui, Hanhe Lin, Lu Shen, Jin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23964)  

**Abstract**: Patients with dementia typically exhibit cognitive impairment, which is routinely assessed using the Mini-Mental State Examination (MMSE). Concurrently, their underlying neurophysiological abnormalities are reflected in Electroencephalography (EEG), providing a basis for joint modeling. However, traditional multi-task approaches suffer from feature entanglement, which leads to inter-task interference when handling heterogeneous this http URL address this challenge, we propose a task-guided spatiotemporal network (TGSN) with diffusion augmentation for EEG-based dementia diagnosis and MMSE prediction. Specifically, TGSN integrates a multi-band feature fusion module to capture complementary spectral information from EEG. Meanwhile, a pre-trained data augmentation module utilizing a diffusion process is introduced toincrease sample diversity. To model the complex spatiotemporal patterns of EEG, we propose a gated spatiotemporal attention module that captures long-range spatial dependencies and temporal dynamics. Moreover, we design a task-guided query module to achieve task-specific feature extraction, thereby mitigating task interference. The effectiveness of TGSN is evaluated on the XY02 dataset. Experimental results demonstrate that the proposed network outperforms several state-of-the-art methods, achieving classification accuracies of 97.78\% for Alzheimer's Disease (AD)/Frontotemporal Dementia (FTD) and 83.93\% for AD/FTD/Vascular Cognitive Impairment (VCI), which exceed the best baselines by 16.39\% and 8.28\%, respectively. In parallel, it reduces the RMSE for MMSE prediction to 1.93 and 2.38, achieving significant error reductions of 1.44 and 1.43 compared to the best baselines. Additionally, validation on the DS004504 dataset demonstrates strong cross-dataset generalization... 

---
# Viewport-Unaware Blind Omnidirectional Image Quality Assessment: A Unified and Generalized Approach 

**Authors**: Jiebin Yan, Kangcheng Wu, Jingwen Hou, Jiayu Zhang, Pengfei Chen, Yuming Fang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23953)  

**Abstract**: Blind omnidirectional image quality assessment (BOIQA) presents a great challenge to the visual quality assessment community, due to different storage formats and diverse user viewing behaviors. The main paradigm of BOIQA models includes two steps, ie, viewport generation, and quality prediction, which brings an extra computational burden and is hard to generalize to other visual contents (eg, 2D planar image). Thus, in this paper, we make an attempt to solve these issues. First, we experimentally find that BOIQA can be formulated as a blind (2D planar) image quality assessment (BIQA) problem, ie, the first step - viewport generation - is no longer needed, which narrows the natural gap between BOIQA and BIQA. Then, we present a new BOIQA approach, which has three merits: ie, viewport-unaware - it accepts an omnidirectional image in the widely used equirectangular projection format as input without any transformation; unified - it can also be applied to BIQA; and generalized - it shows better generalizability against other competitors. Finally, we validate its promise by held-out test, cross-database validation, and the well-established gMAD competition. 

---
# KOMBO: Korean Character Representations Based on the Combination Rules of Subcharacters 

**Authors**: SungHo Kim, Juhyeong Park, Yeachan Kim, SangKeun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.23948)  

**Abstract**: The Korean writing system, \textit{Hangeul}, has a unique character representation rigidly following the invention principles recorded in \textit{Hunminjeongeum}.\footnote{\textit{Hunminjeongeum} is a book published in 1446 that describes the principles of invention and usage of \textit{Hangeul}, devised by King Sejong \cite{Hunminjeongeum_Guide}.} However, existing pre-trained language models (PLMs) for Korean have overlooked these principles. In this paper, we introduce a novel framework for Korean PLMs called KOMBO, which firstly brings the invention principles of \textit{Hangeul} to represent character. Our proposed method, KOMBO, exhibits notable experimental proficiency across diverse NLP tasks. In particular, our method outperforms the state-of-the-art Korean PLM by an average of 2.11\% in five Korean natural language understanding tasks. Furthermore, extensive experiments demonstrate that our proposed method is suitable for comprehending the linguistic features of the Korean language. Consequently, we shed light on the superiority of using subcharacters over the typical subword-based approach for Korean PLMs. Our code is available at: [this https URL](this https URL). 

---
# What Did They Mean? How LLMs Resolve Ambiguous Social Situations across Perspectives and Roles 

**Authors**: Qiming Yuan, Linyi Han, Nam Ling, Cihan Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2604.23942)  

**Abstract**: People increasingly turn to large language models (LLMs) to interpret ambiguous social situations: a delayed text reply, an unusually cold supervisor, a teacher's mixed signals, or a boundary-crossing friend. Yet in many such cases, no stable interpretation can be verified from the available evidence alone. We study how LLMs respond to these situations across four domains: early-stage romantic relationships, teacher--student dynamics, workplace hierarchies, and ambiguous friendships. Across 72 responses from GPT, Claude, and Gemini, only 9 (12.5\%) genuinely preserved uncertainty. The remaining 87.5% produced interpretive closure through recurring pathways including narrative alignment, narrative reversal, normative advice under uncertainty, and hedged language that still supported a single conclusion. We further find that narrator perspective shapes the path to closure: first-person accounts more often elicited alignment, while third-person accounts invited more detached interpretation, even when the underlying situation remained comparable. Together, these findings show that LLMs do not simply assist interpersonal sensemaking; they tend to resolve ambiguity into coherent and actionable narratives. These results suggest that the central risk is not only that LLMs may misinterpret social situations, but that they may make unresolved situations feel prematurely settled. We frame this tendency as a design challenge for uncertainty-preserving social AI. 

---
# Constraint-Guided Multi-Agent Decompilation for Executable Binary Recovery 

**Authors**: Yifan Zhang, Xiaohan Wang, Yueke Zhang, Kevin Leach  

**Link**: [PDF](https://arxiv.org/pdf/2604.23940)  

**Abstract**: Decompilation -- recovering source code from compiled binaries -- is essential for security analysis, malware reverse engineering, and legacy software maintenance. However, existing decompilers produce code that often fails to compile or execute correctly, limiting their practical utility. We present a multi-agent framework that transforms decompiled code into re-executable source through Multi-level Constraint-Guided Decompilation (MCGD). Our approach employs a hierarchical validation pipeline with three constraint levels: (1) syntactic correctness via parsing, (2) compilability via GCC, and (3) behavioral equivalence via LLM-generated test cases. When validation fails, specialized LLM agents iteratively refine the code using structured error feedback. We evaluate our framework on 1,641 real-world binaries from ExeBench across three decompilers (RetDec, Ghidra, and Angr). Our framework achieves 84-97% re-executability, improving baseline decompiler output by 28-89 percentage points. In comparison with state-of-the-art LLM-based decompilation methods using the same GPT-4o backbone, our approach (84.1%) outperforms LLM4Decompile (80.3%), SK2Decompile (73.9%), and SALT4Decompile (61.8%). Our ablation study reveals that execution-based validation is critical: compile-only approaches achieve 0% behavioral correctness despite 91-99% compilation rates. The system converges efficiently, with 90%+ binaries reaching correctness within 2 iterations at an average cost of $0.03-0.05 per binary. Our results demonstrate that constraint-guided agentic refinement can bridge the gap between raw decompiler output and practically useful source code. 

---
# Do Quantum Transformers Help? A Systematic VQC Architecture Comparison on Tabular Benchmarks 

**Authors**: Chi-Sheng Chen, En-Jui Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2604.23931)  

**Abstract**: Variational quantum circuits (VQCs) are a leading approach to quantum machine learning on near-term devices, yet it remains unclear which circuit architecture yields the best accuracy-parameter trade-off on classical tabular data. We present a systematic empirical comparison of four VQC families -- multi-layer fully-connected (FC-VQC), residual (ResNet-VQC), hybrid quantum-classical transformer (QT), and fully quantum transformer (FQT) -- across five regression and classification benchmarks. Our key findings are: \textbf{(i)}~FC-VQCs achieve 90-96\% of the $R^2$ of attention-based VQCs while using 40-50\% fewer parameters, and consistently outperform equal-capacity MLPs (mean $R^2{=}0.829$ vs.\ MLP$_{720}$'s $0.753$ on Boston Housing, 3-seed average); \textbf{(ii)}~FC-VQC's Type~4 inter-block connectivity provides partial cross-token mixing that approximates the role of attention -- explicit quantum self-attention yields only marginal gains on most datasets while significantly increasing parameter count; \textbf{(iii)}~expressibility saturates at circuit depth~${\approx}\,3$, explaining why shallow VQCs already cover the Hilbert space effectively; \textbf{(iv)}~LayerNorm on the fully quantum transformer improves classification accuracy, suggesting normalization is important when all operations are quantum; \textbf{(v)}~in our noise study on Boston Housing, FQT degrades gracefully under depolarizing noise while QT collapses. All results are validated across three random seeds. These findings provide practical architectural guidance for deploying VQCs on near-term quantum hardware. 

---
# Quasi-Quadratic Gradient: A New Direction for Accelerating the BFGS Method in Quasi-Newton Optimization 

**Authors**: John Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23922)  

**Abstract**: In this paper, we introduce the Quasi-Quadratic Gradient (QQG), a novel search direction designed to accelerate the BFGS method within the quasi-Newton framework. By defining the QQG as the product of the inverse Hessian approximation and the current gradient, we explicitly leverage local second-order curvature to rectify the search path. Theoretical analysis and empirical results demonstrate that our approach significantly outperforms vanilla BFGS in convergence speed while maintaining computational efficiency. 

---
# Crystal structure prediction using graph neural combinatorial optimization 

**Authors**: Stavros Gerolymatos, J. Kyle Brubaker, Martin J. A. Schuetz, Vladimir V. Gusev  

**Link**: [PDF](https://arxiv.org/pdf/2604.23921)  

**Abstract**: Crystalline materials are widely used in technological applications, yet their discovery remains a significant challenge. As their properties are driven by structure, crystal structure prediction (CSP) methods play a central role in computational approaches aiming to accelerate this process. Previously, CSP has been approached from a combinatorial optimization perspective, with the core challenge of allocating atoms on a fine grid of predefined discrete positions within a unit cell while minimizing their interaction energy. Exact mathematical optimization methods provide guaranteed solutions, but they become computationally expensive for large-scale instances, where the atomic configuration space grows rapidly, particularly in the absence of additional symmetry constraints. In this work, we introduce a neural combinatorial optimization approach to the atom allocation challenge and, subsequently, CSP, based on graph neural networks (GNNs), which can effectively sample from the distribution of feasible structures in an unsupervised manner. We leverage expander graphs to construct computational graphs over discrete positions that capture both short- and long-range interactions between atoms, and employ the Gumbel-Sinkhorn approach to enforce the desired stoichiometry of the generated structures. We demonstrate that our method outperforms classical heuristic approaches and is competitive with a commercial optimization solver across a range of chemical compositions. This enables the use of ever-expanding GPU infrastructure to tackle the inherent combinatorial challenges of CSP, paving the way for scaling beyond current capabilities. 

---
# SMSI: System Model Security Inference: Automated Threat Modeling for Cyber-Physical Systems 

**Authors**: RoÝah Radaideh, Ali Khreis  

**Link**: [PDF](https://arxiv.org/pdf/2604.23905)  

**Abstract**: Threat modeling for cyber-physical systems (CPS) remains a largely manual exercise. This project presents SMSI (System Model Security Inference), a hybrid neuro-symbolic pipeline that starts from a SysML architecture model and produces a prioritized list of NIST 800-53 security controls. The prototype has three main stages: a deterministic parser mapping system components to vulnerabilities via the NVD; a family of retrieval and classification models linking vulnerabilities to MITRE ATT&CK techniques; and a control recommender. We explore three approaches for CVE-to-ATT&CK mapping: a supervised classifier using fine-tuned SecureBERT+, retrieval-based dense encoders, and a zero-shot LLM approach using Gemma-4 26B. We validate the pipeline on a healthcare IoT gateway with nine software components. For the ATT&CK-to-NIST stage, pretrained SecureBERT achieves the highest control retrieval scores, demonstrating that dense embeddings provide a strong basis for automated control recommendation. 

---
# Generative Synthetic Data for Causal Inference: Pitfalls, Remedies, and Opportunities 

**Authors**: Yichen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23904)  

**Abstract**: Synthetic data offers a promising tool for privacy-preserving data release, augmentation, and simulation, but its use in causal inference requires preserving more than predictive fidelity. We show that fully generative tabular synthesizers, including GAN- and LLM-based models, can achieve strong train-on-synthetic-test-on-real performance while substantially distorting causal estimands such as the average treatment effect (ATE). We formalize this failure through sensitivity and tradeoff results showing that ATE preservation requires control of both the generated covariate law and the treatment-effect contrast in the outcome regression. Motivated by this observation, we propose a hybrid synthetic-data framework that generates covariates separately from the treatment and outcome mechanisms, using distance-to-closest-record diagnostics to monitor covariate synthesis and separately learned nuisance models to construct (W, A, Y) triplets. We further study targeted synthetic augmentation for practical positivity problems and characterize when added overlap support helps by improving conditional-effect estimation more than it shifts the covariate distribution. Finally, we develop a synthetic simulation engine for pre-analysis estimator evaluation, enabling finite-sample comparison of OR, IPW, AIPW, and TMLE under realistic covariate structure. Across experiments, hybrid synthetic data substantially improve ATE preservation relative to fully generative baselines and provide a practical diagnostic tool for robust causal analysis. 

---
# Geometry Preserving Loss Functions Promote Improved Adaptation of Blackbox Generative Model 

**Authors**: Sinjini Mitra, Constantine Kyriakakis, Shenyuan Liang, Anuj Srivastava, Pavan Turaga  

**Link**: [PDF](https://arxiv.org/pdf/2604.23888)  

**Abstract**: Adaptation of blackbox generative models has been widely studied recently through the exploration of several methods including generator fine-tuning, latent space searches, leveraging singular value decomposition, and so on. However, adapting large-scale generative AI tools to specific use cases continues to be challenging, as many of these industry-grade models are not made widely available. The traditional approach of fine-tuning certain layers of a generative network is not feasible due to the expense of storing and fine-tuning generative models, as well as the restricted access to weights and gradients. Recognizing these challenges, we propose a novel end-to-end pipeline aimed at domain adaptation by leveraging geometry-preserving loss functions in conjunction to pre-trained generative adversarial networks (GANs). Our method rethinks the problem of adaptation by re-contextualizing the role of GAN inversion in obtaining accurate latent space representations. Extending the ability of existing state-of-the-art inverters, we preserve pair-wise distances between tangent spaces to successfully train a latent generative model to produce samples from the target distribution. We evaluate our proposed pipeline on StyleGANs with real distribution shifts and demonstrate that the introduction of the geometry preserving loss function lends to improved adaptation of generative models compared to other traditional loss functions. 

---
# Evaluation of Prompt Injection Defenses in Large Language Models 

**Authors**: Priyal Deep, Shane Emmons, Amy Fox, Kyle Bacon, Kelley McAllister, Krisztian Flautner  

**Link**: [PDF](https://arxiv.org/pdf/2604.23887)  

**Abstract**: LLM-powered applications routinely embed secrets in system prompts, yet models can be tricked into revealing them. We built an adaptive attacker that evolves its strategies over hundreds of rounds and tested it against nine defense configurations across more than 20,000 attacks. Every defense that relied on the model to protect itself eventually broke. The only defense that held was output filtering, which checks the model's responses via hardcoded rules in separate application code before they reach the user, achieving zero leaks across 15,000 attacks. These results demonstrate that security boundaries must be enforced in application code, not by the model being attacked. Until such defenses are verified by tools like Swept AI, AI systems handling sensitive operations should be restricted to internal, trusted personnel. 

---
# MUSIC: Learning Muscle-Driven Dexterous Hand Control 

**Authors**: Pei Xu, Yufei Ye, Shuchun Sun, Yu Ding, Elizabeth Schumann, C. Karen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23886)  

**Abstract**: We present a data-driven approach for physics-based, muscle-driven dexterous control that enables musculoskeletal hands to perform precise piano playing for novel pieces of music outside the reference dataset. Our approach combines high-frequency muscle-level control with low-frequency latent-space coordination in a hierarchical architecture. At the low level, general single-hand policies are trained via reinforcement learning to generate dynamic muscle-tendon activations while tracking trajectories from a large reference motion dataset. The resulting tracking policies are then distilled into variational autoencoder (VAE) models, yielding smooth and structured latent spaces that abstract away low-level muscle dynamics. For the high level, we train piece-specific policies to operate in this latent space, coordinating bimanual motions based on specific goals, denoted by note events extracted from given musical scores, to synthesize performances beyond the reference data. In addition, we present an enhanced musculoskeletal hand model that supports fine control of fingers for accurate low-level motion tracking and diverse high-level motion synthesis. We evaluate the control pipeline of our approach on a diverse piano repertoire spanning multiple musical styles and technical demands. Results demonstrate that our approach can synthesize coordinated bimanual motions with accurate key presses, and achieve the state-of-the-art performance of piano playing in physics-based dexterous control. We also show that our musculoskeletal hand model demonstrates superior biomechanical stability and tracking precision compared to the existing model, and validate that our musculoskeletal hand model and muscle-driven controller can generate physiologically plausible activation patterns that align with human electromyography (EMG) recordings. 

---
# Risk-Aware Robust Learning: Reducing Clinical Risk under Label Noise in Medical Image Classification 

**Authors**: Maycon R. S. Pereira, Filipe R. Cordeiro  

**Link**: [PDF](https://arxiv.org/pdf/2604.23875)  

**Abstract**: Noisy labels are a pervasive challenge in medical image classification, where annotation errors arise from inter-observer variability and diagnostic ambiguity. Although several noise-robust learning methods have been proposed, their evaluation predominantly relies on accuracy-oriented metrics, overlooking the clinical implications of asymmetric error costs. In medical diagnosis, a false negative (missed disease) carries substantially higher consequences than a false positive (false alarm), as delayed treatment can directly impact patient outcomes. In this work, we investigate whether noise-robust training methods preserve clinical safety under label noise. We conduct a systematic risk-aware evaluation of the state-of-the-art noise-robust methods Coteaching, DivideMix, UNICON, and a GMM-based filtering approach on binarized DermaMNIST and PathMNIST datasets under clean and label noise rates of 20%, and 40%. Beyond balanced accuracy, we adopt a cost-sensitive Global Risk formulation that explicitly penalizes false negatives. Our analysis reveals that the robustness of state-of-the-art methods does not guarantee clinical safety. Furthermore, we demonstrate that integrating cost-sensitive optimization into noise-robust training significantly reduces clinical risk, while mantaining model utility. These findings demonstrate that noise-robust learning must be evaluated through a clinical risk lens, and that combining robust training with cost-sensitive optimization can meaningfully reduce risk in noisy-label medical imaging scenarios. 

---
# Inverting Foundation Models of Brain Function with Simulation-Based Inference 

**Authors**: Niels Bracher, Xavier Intes, Stefan T. Radev  

**Link**: [PDF](https://arxiv.org/pdf/2604.23865)  

**Abstract**: Foundation models of brain activity promise a new frontier for in silico neuroscience by emulating neural responses to complex stimuli across tasks and modalities. A natural next step is to ask whether these models can also be used in reverse. Can we recover a stimulus or its properties from synthetic brain activity? We study this question in a proof-of-concept setting using TRIBEv2. We pair the brain emulator with large language models (LLMs) that generate news headlines from linguistic parameters such as valence, arousal, and dominance. We then use simulation-based inference to learn a probabilistic mapping from brain maps to latent stimulus parameters. Our results show that these parameters can be recovered from predicted brain maps, validating the quality of neural encodings. They also show that LLMs can serve as controllable stimulus generators for simulated experiments. Together, these findings provide a step toward decoding and inverse design with foundation brain models. 

---
# Graph Memory Transformer (GMT) 

**Authors**: Nicola Zanarini, Niccolò Ferrari  

**Link**: [PDF](https://arxiv.org/pdf/2604.23862)  

**Abstract**: We investigate whether the Feed-Forward Network (FFN) sublayer in a decoder-only transformer can be replaced by an explicit learned memory graph while preserving the surrounding autoregressive architecture. The proposed Graph Memory Transformer (GMT) keeps causal self-attention intact, but replaces the usual per-token FFN transformation with a memory cell that routes token representations over a learned bank of centroids connected by a learned directed transition matrix. In the base GMT v7 instantiation studied here, each of 16 transformer blocks contains 128 centroids, a 128 * 128 edge matrix, gravitational source routing, token-conditioned target selection, and a gated displacement readout. The cell therefore returns movement from an estimated source memory state toward a target memory state, rather than a retrieved value. The resulting model is a fully decoder-only language model with 82.2M trainable parameters and no dense FFN sublayers, compared with a 103.0M-parameter dense GPT-style baseline used in the evaluation. The base v7 model trains stably and exposes centroid usage, transition structure, and source-to-target movement as directly inspectable quantities of the forward computation. It remains behind the larger dense baseline in validation loss and perplexity (3.5995/36.58 vs. 3.2903/26.85), while showing close zero-shot benchmark behavior under the evaluated setting. These results are not intended as a state-of-the-art claim; they support the viability and structural interpretability of replacing dense within-token transformation with graph-mediated memory navigation. Broader scaling, optimized kernels, and more extensive benchmark evaluation are left for subsequent work. 

---
# Empirical Ablation and Ensemble Optimization of a Convolutional Neural Network for CIFAR-10 Classification 

**Authors**: Naser Khatti Dizabadi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23861)  

**Abstract**: Convolutional neural networks (CNNs) remain a central approach in image classification, but their performance depends strongly on architectural and training choices. This paper presents an empirical ablation-based study of CNN optimization for the CIFAR-10 benchmark. The study evaluates 17 progressive modifications involving training duration, learning-rate scheduling, dropout configuration, pooling strategy, network depth, filter arrangement, and dense-layer design. The goal is to identify which changes improve generalization and which increase complexity without improving performance. The baseline model achieved 79.5\% test accuracy. Extending training duration improved performance steadily, whereas several structural redesigns reduced accuracy despite greater architectural variation. Based on the strongest individual configurations, a weighted ensemble was constructed, achieving 86.38\% accuracy in the reduced-data setting and 89.23\% when trained using the full CIFAR-10 dataset. These results suggest that performance gains in CNN-based classification depend less on indiscriminate increases in depth or parameter count than on careful empirical selection of training and architectural modifications. The study therefore highlights the practical value of ablation-oriented optimization and ensemble learning for small-image classification. 

---
# Exploring Audio Hallucination in Egocentric Video Understanding 

**Authors**: Ashish Seth, Xinhao Mei, Changsheng Zhao, Varun Nagaraja, Ernie Chang, Gregory P. Meyer, Gael Le Lan, Yunyang Xiong, Vikas Chandra, Yangyang Shi, Dinesh Manocha, Zhipeng Cai  

**Link**: [PDF](https://arxiv.org/pdf/2604.23860)  

**Abstract**: Egocentric videos provide a distinctive setting in which sound serves as crucial cues to understand user activities and surroundings, particularly when visual information is unstable or occluded due to continuous camera movement. State-of-the-art large audio-visual language models (AV-LLMs) can generate multimodal descriptions. However, we show in this work that they are prone to audio hallucinations, often inferring sounds from visual cues that are visible but not heard. We present a systematic and automatic evaluation framework for analyzing audio hallucinations in egocentric video through a targeted question-answering (Q/A) protocol. We curate a dataset of 300 egocentric videos and design 1,000 sound-focused questions to probe model outputs. To characterize hallucinations, we propose a grounded taxonomy that distinguishes between foreground action sounds from the user activities and background ambient sounds. Our evaluation shows that advanced AV-LLMs, such as Qwen2.5 Omni, exhibit high hallucination rates, achieving only 27.3% and 39.5% accuracy on Q/As related to foreground and background sounds, respectively. With this work, we highlight the need to measure the reliability of multimodal responses, emphasizing that robust evaluation of hallucinations is essential to develop reliable AV-LLMs. 

---
# Reheat Nachos for Dinner? Evaluating AI Support for Cross-Cultural Communication of Neologisms 

**Authors**: Dayeon Ki, Yu Hou, Rachel Rudinger, Hal Daumé III, Marine Carpuat, Fumeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23842)  

**Abstract**: Neologisms and emerging slang are central to daily conversation, yet challenging for non-native speakers (NNS) to interpret and use appropriately in cross-cultural communication with native speakers (NS). NNS increasingly make use of Artificial Intelligence (AI) tools to learn these words. We study the utility of such tools in mediating an informal communication scenario through a human-subjects study (N=234): NNS participants learn English neologisms with AI support, write messages using the learned word to an NS friend, and judge contextual appropriateness of the neologism in two provided writing samples. Using both NS evaluator-rated communicative competence of NNS-produced writing and NNS' contextual appropriateness judgments, we compare three AI-based support conditions: AI Definition, AI Rewrite into simpler English, AI Explanation of meaning and usage, and Non-AI Dictionary for comparison. We show that AI Explanation yields the largest gains over no support in NS-rated competence, while contextual appropriateness judgments show indifference across support. NNS participants' self-reported perceptions tend to overestimate NS ratings, revealing a mismatch between perceived and actual competence. We further observe a significant gap between NNS- and NS-produced writing, highlighting the limitations of current AI tools and informing design for future tools. 

---
# Scalable Production Scheduling: Linear Complexity via Unified Homogeneous Graphs 

**Authors**: Jonathan Hoss, Moritz Link, Noah Klarmann  

**Link**: [PDF](https://arxiv.org/pdf/2604.23841)  

**Abstract**: Efficiently solving the Job Shop Scheduling Problem in real-world industrial applications requires policies that are both computationally lean and topologically robust. While Reinforcement Learning has shown potential in automating dispatching rules, existing models often struggle with a scalability bottleneck caused by quadratic graph complexity or the architectural overhead of heterogeneous layers. We introduce a unified graph framework that employs feature-based homogenization to project distinct node roles into a shared latent space. This allows a standard homogeneous Graph Isomorphism Network to capture complex resource contention with linear complexity, ensuring low-latency inference for large-scale industrial applications. Our empirical results demonstrate that our framework achieves state-of-the-art performance while exhibiting consistent zero-shot generalization. We identify the job-to-machine ratio as the primary driver of policy effectiveness, rather than absolute problem size. Based on this, we propose a hypothesis of structural saturation, demonstrating that policies trained on critically congested instances ($\mathcal{J} \approx \mathcal{M}$) learn scale-invariant resolution strategies. Agents trained at this saturation point internalize invariant conflict-resolution logic, allowing them to treat massive rectangular instances as a sequential concatenation of saturated sub-problems. This approach eliminates the need for expensive scale-specific retraining and prevents overfitting to statistical shortcuts, providing a robust and efficient pathway for deploying RL solutions in dynamic production environments. 

---
# Focus on What Matters: Two-Stage ROI-Aware Refinement for Anatomy-Preserving Fetal Ultrasound Reconstruction 

**Authors**: Ines Abbes, Mahmood Alzubaidi, Mowafa Househ, Khalid Alyafei, Marco Agus, Samir Brahim Belhaouari  

**Link**: [PDF](https://arxiv.org/pdf/2604.23839)  

**Abstract**: Measurement-critical ultrasound tasks often depend on a small anatomical region, making global reconstruction metrics an unreliable proxy for clinical fidelity. We propose an ROI-aware representation learning framework and instantiate it for first-trimester nuchal translucency (NT) screening under multi-hospital domain shift. A two-phase convolutional autoencoder (CAE) first learns a globally faithful 128-D latent code via MS-SSIM, then refines the NT ROI using intensity (L1) and normalized Sobel-edge constraints. To combine these heterogeneous objectives without manual tuning, we initialize loss weights via gradient-based calibration from per-term gradient magnitudes. Under strict hospital-wise evaluation with one hospital held out, ROI refinement improves both global and measurement-relevant quality: on the standard dev split it increases PSNR by +0.27 dB (val) and +0.29 dB (held-out test), reduces ROI MAE by 8.87% (val) and 6.43% (held-out test), and reduces ROI Edge-MAE by 11.10% on source hospitals and 4.90% on the unseen hospital. Beyond reconstruction, frozen-latent probes provide additional evidence of generalization: hospital provenance becomes less confidently predictable on the unseen site (0.556 to 0.541 max-softmax; 0.684 to 0.688 entropy) while OOD detection remains strong across site-held-out protocols (Mahalanobis AUROC up to 0.9956, with modest KNN gains in challenging splits). The same ROI-aware refinement principle is anatomy-agnostic and can be adopted for other fetal biometry targets (e.g., crown-rump length (CRL), nasal bone (NB)) and broader medical imaging settings where small ROIs dominate clinical decisions. 

---
# Query2Diagram: Answering Developer Queries with UML Diagrams 

**Authors**: Oleg Baryshnikov, Anton M. Alekseev, Sergey I. Nikolenko  

**Link**: [PDF](https://arxiv.org/pdf/2604.23816)  

**Abstract**: Software documentation frequently becomes outdated or fails to exist entirely, yet developers need focused views of their codebase to understand complex systems. While automated reverse engineering tools can generate UML diagrams from code, they produce overwhelming detail without considering developer intent. We introduce query-driven UML diagram generation, where LLMs create diagrams that directly answer natural language questions about code. Unlike existing methods, our approach produces semantically focused diagrams containing only relevant elements with contextual descriptions. We fine-tune Qwen2.5-Coder-14B on a curated dataset of code files, developer queries, and corresponding diagram representations in a structured JSON format, evaluating with both automatic detection of structural defects and human assessment of semantic relevance. Results demonstrate that fine-tuning on a modest amount of manually corrected data yields dramatic improvements: our best model achieves the highest F1 scores while reducing defect rates below state-of-the-art LLMs, generating diagrams that are both structurally sound and semantically faithful to developer queries. Thus, we establish the feasibility of using LLMs for scalable contextual, on-demand documentation generation. We make our code and dataset publicly available at this https URL. 

---
# Mapping License Plate Recoverability Under Extreme Viewing Angles for Oppor-tunistic Urban Sensing 

**Authors**: Igor Adamenko, Orpaz Ben Aharon, Yehudit Aperstein, Alexander Apartsin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23814)  

**Abstract**: Urban environments contain many imaging sensors built for specific purposes, including ATM, body-worn, CCTV, and dashboard cameras. Under the opportunistic sensing paradigm, these sensors can be repurposed for secondary inference tasks such as license plate recognition. Yet objects of interest in such imagery are often noisy, low-resolution, and captured from extreme viewpoints. Recent advances in AI-based restoration can recover use-ful information even from severely degraded images. A central challenge is determining which distortion parame-ters allow reliable recovery and which lead to inference failure. This paper introduces recoverability maps, a task-agnostic method for quantifying this boundary. The method combines a dense synthetic sweep of degrada-tion parameters with two summary measures: boundary area-under-curve, which estimates the recoverable frac-tion of the parameter space, and a reliability score, which captures the frequency and severity of failures within that region. We demonstrate the method on license plate recognition from highly angled views under realistic camera artifacts. Several restoration architectures are trained and evaluated, including U-Net, Restormer, Pix2Pix, and SR3 diffusion. The best model recovers about 93% of the parameter space. Similar results across models sug-gest that sensing geometry, rather than architecture, sets the limit of recovery. 

---
# Symmetric Equilibrium Propagation for Thermodynamic Diffusion Training 

**Authors**: Aditi De  

**Link**: [PDF](https://arxiv.org/pdf/2604.23806)  

**Abstract**: The reverse process in score-based diffusion models is formally equivalent to overdamped Langevin dynamics in a time-dependent energy landscape. In our prior work we showed that a bilinearly-coupled analog substrate can physically realize this dynamics at a projected three-to-four orders of magnitude energy advantage over digital inference by replacing dense skip connections with low-rank inter-module couplings. Whether the \emph{training} loop can be closed on the same substrate -- without routing gradients through an external digital accelerator -- has remained open. We resolve this affirmatively: Equilibrium Propagation applied directly to the bilinear energy yields an unbiased estimator of the denoising score-matching gradient in the zero-nudge limit. For finite nudging we derive a sharp bias bound controlled solely by substrate stiffness, local curvature, and the norm of the loss-gradient signal, with a bilinear-specific corollary showing that one dominant bias term vanishes identically for coupling-parameter updates. Symmetric nudging further upgrades the leading bias from $ \mathcal{O}(\beta) $ to $ \mathcal{O}(\beta^2) $ at negligible extra cost. Under realistic finite-relaxation budgets this upgrade is essential, as one-sided EqProp produces anti-correlated gradients while symmetric EqProp yields well-aligned updates. Bias-variance analysis determines the optimal operating point, and end-to-end physical-unit accounting projects a $ 10^3$-$10^4\times $ energy advantage per training step over a matched GPU baseline. Symmetric bilinear EqProp is the first local, readout-only training rule that preserves the low-rank coupling enabling scalable thermodynamic diffusion models. 

---
# S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA 

**Authors**: Minghan Li, Junjie Zou, Xinxuan Lv, Chao Zhang, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.23783)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds language models in external evidence, but multi-hop question answering remains difficult because iterative pipelines must control what to retrieve next and when the available evidence is adequate. In practice, systems may answer from incomplete evidence chains, or they may accumulate redundant or distractor-heavy text that interferes with later retrieval and reasoning. We propose S2G-RAG (Structured Sufficiency and Gap-judging RAG), an iterative framework with an explicit controller, S2G-Judge. At each turn, S2G-Judge predicts whether the current evidence memory supports answering and, if not, outputs structured gap items that describe the missing information. These gap items are then mapped into the next retrieval query, producing stable multi-turn retrieval trajectories. To reduce noise accumulation, S2G-RAG maintains a sentence-level Evidence Context by extracting a compact set of relevant sentences from retrieved documents. Experiments on TriviaQA, HotpotQA, and 2WikiMultiHopQA show that S2G-RAG improves multi-hop QA performance and robustness under multi-turn retrieval. Furthermore, S2G-RAG can be integrated into existing RAG pipelines as a lightweight component, without modifying the search engine or retraining the generator. 

---
# GLIER: Generative Legal Inference and Evidence Ranking for Legal Case Retrieval 

**Authors**: Minghan Li, Tianrui Lv, Chao Zhang, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.23779)  

**Abstract**: The semantic gap between colloquial user queries and professional legal documents presents a fundamental challenge in Legal Case Retrieval (LCR). Existing dense retrieval methods typically treat LCR as a black-box semantic matching process, neglecting the explicit juridical logic that underpins legal relevance. To address this, we propose GLIER (Generative Legal Inference and Evidence Ranking), a framework that reformulates retrieval as an inference process over latent legal variables. GLIER decomposes the task into two interpretability-driven stages. First, a Joint Generative Inference module translates raw queries into latent legal indicators, including charges and legal elements, using a unified sequence-to-sequence strategy that jointly generates charges and elements to enforce logical consistency. Second, a Multi-View Evidence Fusion mechanism aggregates generative confidence with structural and lexical signals for precise ranking. Extensive experiments on LeCaRD and LeCaRDv2 demonstrate that GLIER outperforms strong baselines such as SAILER and KELLER. Notably, GLIER exhibits strong data efficiency, maintaining robust performance even when trained with only 10% of the data. 

---
# From Noisy Historical Maps to Time-Series Oil Palm Mapping Without Annotation in Malaysia and Indonesia (2020-2024) 

**Authors**: Nuttaset Kuapanich, Juepeng Zheng, Bohan Shi, Jiaying Liu, Jiayin Jiang, Jiatao Huang, Shenghan Tan, Qingmei Li, Haohuan Fu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23776)  

**Abstract**: Accurate monitoring of oil palm plantations is critical for balancing economic development with environmental conservation in Southeast Asia. However, existing plantation maps often suffer from low spatial resolution and a lack of recent temporal coverage, impeding effective surveillance of rapid land-use changes. In this study, we propose a deep learning framework to generate 10-meter resolution oil palm plantation maps for Indonesia and Malaysia from 2020 to 2024, utilizing Sentinel-2 imagery without requiring new manual annotations. To address the resolution mismatch between coarse 100-meter historical labels and 10-meter imagery, we employ a U-Net architecture optimized with Determinant-based Mutual Information (DMI). This approach effectively mitigates the influence of label noise. We validated our method against 2,058 manually verified points, achieving overall accuracies of 70.64%, 63.53%, and 60.06% for the years 2020, 2022, and 2024, respectively. Our comprehensive analysis reveals that oil palm coverage in the region peaked in 2022 before experiencing a decline in 2024. Furthermore, land cover transition analysis highlights a concerning trajectory of plantation expansion into flooded vegetation areas, despite a general stabilization in rotations with other crop types. These high-resolution maps provide essential data for monitoring sustainability commitments and deforestation dynamics in the region, and the generated datasets are made publicly available at this https URL. 

---
# The Override Gap: A Magnitude Account of Knowledge Conflict Failure in Hypernetwork-Based Instant LLM Adaptation 

**Authors**: Shuaizhi Cheng, Xiang Shi, Mingwei Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.23750)  

**Abstract**: Hypernetwork-based methods such as Doc-to-LoRA internalize a document into an LLM's weights in a single forward pass, but they fail systematically on conflicts: when the document contradicts pretraining knowledge, accuracy collapses to 46.4% on the deepest facts. We show the failure is a magnitude problem rather than a representational one. The hypernetwork already targets the right layers, but its adapter margin is approximately constant across documents while the pretrained margin grows with training frequency, so deep conflicts lose by construction. The account predicts that failure should track prior strength: sorting 194 conflicts by the base model's log-probability on the contradicted fact, baseline accuracy falls from 68% on weak-prior questions to 16% on strong-prior ones, a 52 percentage-point gap. The cure is amplitude. Selective Layer Boosting scales the adapter at its top-norm layers, and Conflict-Aware Internalization triggers boosting only when the base model is confident. Both are training-free; together they raise deep-conflict accuracy from 46.4% to 71.0% on Gemma-2B and from 53.6% to 72.5% on Mistral-7B while preserving novel-knowledge recall, and beat vanilla retrieval-augmented generation on medium conflicts by 18 percentage points despite operating entirely in parameter space. We release KID-Bench, a 489-question benchmark that separates novel recall, cross-knowledge combination, and prior-graded conflicts. 

---
# SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning 

**Authors**: Alexis Limozin, Eduard Durech, Torsten Hoefler, Imanol Schlag, Valentina Pyatkin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23747)  

**Abstract**: Recent mixed-policy optimization methods for LLM reasoning that interleave or blend supervised and reinforcement learning signals report improvements over the standard SFT-then-RL pipeline. We show that numerous recently published research papers rely on a faulty baseline caused by two distinct bugs: a CPU-offloaded optimizer bug in DeepSpeed that silently drops intermediate micro-batches during gradient accumulation (affecting multiple downstream frameworks including TRL, OpenRLHF and Llama-Factory), and a loss aggregation bug in OpenRLHF that incorrectly weights per-mini-batch losses. Together they suppress SFT performance, with the optimizer bug accounting for most of the gap and the loss aggregation bug contributing a smaller additional effect. Once corrected, the standard SFT-then-RL pipeline surpasses every published mixed-policy method we evaluate by +3.8 points on math benchmarks with Qwen2.5-Math-7B and by +22.2 points with Llama-3.1-8B. Even a truncated variant with just 50 RL steps outperforms mixed-policy methods on math benchmarks while using fewer FLOPs. 

---
# Impact of Age Specialized Models for Hypoglycemia Classification 

**Authors**: Beyza Cinar, Maria Maleshkova  

**Link**: [PDF](https://arxiv.org/pdf/2604.23732)  

**Abstract**: Disease progression varies with age and is influenced by underlying genetic, biochemical, and hormonal etiologies, suggesting the need for tailored monitoring, care, and medication beyond standard clinical guidelines. Specifically, in autoimmune diseases like type 1 diabetes (T1D), where patients depend on exogenous insulin to compensate for insulin deficiency, medication dosing and the physiological response reflected in vital signs can differ. Insulin therapy can lead to hypoglycemia, a dangerous condition characterized by decreased blood glucose levels ($\leq$70). This risk can be mitigated through improved diabetes management supported by data analytics. Notably, leveraging data from continuous glucose monitoring (CGM) devices, hypoglycemia onset can be predicted. However, while glucose variability, auto-antibody levels, and hypoglycemia occurrence differ across age groups, hypoglycemia classification most often only relies on population-based models specialized in specific age ranges. In this work, we classify hypoglycemia 0, 5-15, 20-45, and 50-120 minutes before onset using DiaData, a large CGM dataset of patients with T1D ranging from children to seniors. In particular, we investigate: 1) the generalizability of a population-based model including all age groups, 2) the impact of age-segmented models trained separately per age group, and 3) the effect of model individualization through transfer learning. The results show that a global population-based model yields similar or superior performance compared to age-segmented models. These findings suggest that data from children, teenagers, and adults can be combined for training models on hypoglycemia classification. While glucose variation differs across age groups, short-term hypoglycemic patterns are similar. However, data of children obtain their best recall with age specialized model. 

---
# ESIA: An Energy-Based Spatiotemporal Interaction-Aware Framework for Pedestrian Intention Prediction 

**Authors**: Yanping Wu, Meiting Dang, Lin Wu, Edmond S. L. Ho, Zhenghua Chen, Chongfeng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.23728)  

**Abstract**: Recent advances in autonomous driving have motivated research on pedestrian intention prediction, which aims to infer future crossing decisions and actions by modeling temporal dynamics, social interactions, and environmental context. However, existing studies remain constrained by oversimplified multi-agent interaction patterns, opaque reasoning logic, and a lack of global consistency in behavioral predictions, which compromise both robustness and interpretability. In this work, we propose ESIA (Energy-based Spatiotemporal Interaction-Aware framework), a novel Conditional Random Field (CRF)-based paradigm. We cast the intention prediction task as a structured prediction problem over a unified graph-based representation, treating pedestrians and the environment as spatiotemporal nodes. To characterize their distinct roles, we assign unary potentials to nodes to capture individual intentions, and pairwise potentials to edges to encode social and environmental interactions. These potentials are integrated into a unified global energy function to ensure scene-level consistency across behavioral predictions. To further constrain inference without ground-truth supervision, we introduce structural consistency terms to penalize logical contradictions. This optimization is efficiently solved via a novel Unary-Seeded Simulated Annealing (U-SSA) algorithm, which leverages high-confidence unary priors to rapidly converge to a high-quality solution. Extensive experiments on standard benchmarks demonstrate that ESIA achieves state-of-the-art performance with improved interpretability over existing methods. 

---
# Zoom In, Reason Out: Efficient Far-field Anomaly Detection in Expressway Surveillance Videos via Focused VLM Reasoning Guided by Bayesian Inference 

**Authors**: Xiaowei Mao, Bowen Sui, Weijie Zhang, Yawen Yang, Shengnan Guo, Shilong Zhao, Jiaqi Lin, Tingrui Wu, Youfang Lin, Huaiyu Wa  

**Link**: [PDF](https://arxiv.org/pdf/2604.23724)  

**Abstract**: Expressway video anomaly detection is essential for safety management. However, identifying anomalies across diverse scenes remains challenging, particularly for far-field targets exhibiting subtle abnormal vehicle motions. While Vision-Language Models (VLMs) demonstrate strong semantic reasoning capabilities, processing global frames causes attention dilution for these far-field objects and incurs prohibitive computational costs. To address these issues, we propose VIBES, an asynchronous collaborative framework utilizing VLMs guided by Bayesian inference. Specifically, to overcome poor generalization across varying expressway environments, we introduce an online Bayesian inference module. This module continuously evaluates vehicle trajectories to dynamically update the probabilistic boundaries of normal driving behaviors, serving as an asynchronous trigger to precisely localize anomalies in space and time. Instead of processing the continuous video stream, the VLM processes only the localized visual regions indicated by the trigger. This targeted visual input prevents attention dilution and enables accurate semantic reasoning. Extensive evaluations demonstrate that VIBES improves detection accuracy for far-field anomalies and reduces computational overhead, achieving high real-time efficiency and explainability while demonstrating generalization across diverse expressway conditions. 

---
# AIPsy-Affect: A Keyword-Free Clinical Stimulus Battery for Mechanistic Interpretability of Emotion in Language Models 

**Authors**: Michael Keeman  

**Link**: [PDF](https://arxiv.org/pdf/2604.23719)  

**Abstract**: Mechanistic interpretability research on emotion in large language models -- linear probing, activation patching, sparse autoencoder (SAE) feature analysis, causal ablation, steering vector extraction -- depends on stimuli that contain the words for the emotions they test. When a probe fires on "I am furious", it is unclear whether the model has detected anger or detected the word "furious". The two readings have very different consequences for every downstream claim about emotion circuits, features, and interventions. We release AIPsy-Affect, a 480-item clinical stimulus battery that removes the confound at the stimulus level: 192 keyword-free vignettes evoking each of Plutchik's eight primary emotions through narrative situation alone, 192 matched neutral controls that share characters, setting, length, and surface structure with the affect surgically removed, plus moderate-intensity and discriminant-validity splits. The matched-pair structure supports linear probing, activation patching, SAE feature analysis, causal ablation, and steering vector extraction under a strong methodological guarantee: any internal representation that distinguishes a clinical item from its matched neutral cannot be doing so on the basis of emotion-keyword presence. A three-method NLP defense battery -- bag-of-words sentiment, an emotion-category lexicon, and a contextual transformer classifier -- confirms the property: bag-of-words methods see only situational vocabulary, and a contextual classifier detects affect (p < 10^-15) but cannot identify the category (5.2% top-1 vs. 82.5% on a keyword-rich control). AIPsy-Affect extends our earlier 96-item battery (arXiv:2603.22295) by a factor of four and is released openly under MIT license. 

---
# OptProver: Bridging Olympiad and Optimization through Continual Training in Formal Theorem Proving 

**Authors**: Chenyi Li, Yanchen Nie, Zhengyu Ming, Gong Zhang, Kun Yuan, Zaiwen Wen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23712)  

**Abstract**: Recent advances in formal theorem proving have focused on Olympiad-level mathematics, leaving undergraduate domains largely unexplored. Optimization, fundamental to machine learning, operations research, and scientific computing, remains underserved by existing provers. Its reliance on domain-specific formalisms (convexity, optimality conditions, and algorithmic analysis) creates significant distribution shift, making naive domain transfer ineffective. We present OptProver, a trained model that achieves robust transfer from Olympiad to undergraduate optimization. Starting from a strong Olympiad-level prover, our pipeline mitigates distribution shift through two key innovations. First, we employ large-scale optimization-focused data curation via expert iteration. Second, we introduce a specialized preference learning objective that integrates perplexity-weighted optimization with a mechanism to penalize valid but non-progressing proof steps. This not only addresses distribution shifts but also guides the search toward efficient trajectories. To enable rigorous evaluation, we construct a novel benchmark in Lean 4 focused on optimization. On this benchmark, OptProver achieves state-of-the-art Pass@1 and Pass@32 among comparably sized models while maintaining competitive performance on general theorem-proving tasks, demonstrating effective domain transfer without catastrophic forgetting. 

---
# Talking Slide Avatars: Open-Source Multimodal Communication Approach for Teaching 

**Authors**: Xinxing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23703)  

**Abstract**: Slide-based teaching is widely used in higher education, yet in online, hybrid, and asynchronous contexts, slides often lose the instructor presence, narrative continuity, and expressive framing that help learners connect with content. Full lecture video can partly restore these qualities, but it is time-consuming to record, revise, and reuse. This study addresses that pedagogical and production challenge by presenting a practice-based analysis of an open-source workflow for creating talking slide avatars for slide-based teaching. The workflow integrates OpenVoice for text-to-speech generation and voice cloning with Ditto-TalkingHead for audio-driven talking-image synthesis, enabling instructors to transform a script and a static portrait into a short narrated video that can be embedded in slide decks or HTML-based lecture materials. Rather than treating this workflow merely as a technical solution, the study frames talking slide avatars as multimodal communication artifacts at the intersection of digital pedagogy, aesthetic education, and art-technology practice. Using a practice-based implementation and analytic reflection approach, the study documents the production pipeline, examines its communicative and aesthetic affordances, and proposes practical guidelines for script length, image selection, pacing, disclosure, accessibility, and ethical use. The study makes three primary contributions: it presents an educator-oriented open-source production model, reframes talking avatars as an educational communication design problem, and proposes a responsible pathway for incorporating generative synthetic media into teaching. It concludes that short, transparent, and carefully designed avatars can humanize slide-based instruction while providing a reusable communicative layer for introductions, transitions, reminders, and recaps across online, hybrid, and asynchronous learning environments. 

---
# Agri-CPJ: A Training-Free Explainable Framework for Agricultural Pest Diagnosis Using Caption-Prompt-Judge and LLM-as-a-Judge 

**Authors**: Wentao Zhang, Qi Zhang, Mingkun Xu, Mu You, Henghua Shen, Zhongzhi He, Keyan Jin, Derek F. Wong, Tao Fang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23701)  

**Abstract**: Crop disease diagnosis from field photographs faces two recurring problems: models that score well on benchmarks frequently hallucinate species names, and when predictions are correct, the reasoning behind them is typically inaccessible to the practitioner. This paper describes Agri-CPJ (Caption-Prompt-Judge), a training-free few-shot framework in which a large vision-language model first generates a structured morphological caption, iteratively refined through multi-dimensional quality gating, before any diagnostic question is answered. Two candidate responses are then generated from complementary viewpoints, and an LLM judge selects the stronger one based on domain-specific criteria. Caption refinement is the component with the largest individual impact: ablations confirm that skipping it consistently degrades downstream accuracy across both models tested. On CDDMBench, pairing GPT-5-Nano with GPT-5-mini-generated captions yields \textbf{+22.7} pp in disease classification and \textbf{+19.5} points in QA score over no-caption baselines. Evaluated without modification on AgMMU-MCQs, GPT-5-Nano reached 77.84\% and Qwen-VL-Chat reached 64.54\%, placing them at or above most open-source models of comparable scale despite the format shift from open-ended to multiple-choice. The structured caption and judge rationale together constitute a readable audit trail: a practitioner who disagrees with a diagnosis can identify the specific caption observation that was incorrect. Code and data are publicly available this https URL 

---
# An AI-Based Supervisory Measurement Integrity Validation Layer for Cyber-Resilient AC/DC Protection in Inverter-Based Microgrids 

**Authors**: Ahmad Mohammad Saber, Ahmed Saber Refae, Davor Svetinovic, Hatem Zeineldin, Amr Youssef, Ehab F. El-Saadany, Deepa Kundur  

**Link**: [PDF](https://arxiv.org/pdf/2604.23666)  

**Abstract**: Line current differential relays (LCDRs) are measurement-driven relays that rely on time-synchronized multi-phase current waveforms to infer internal faults in AC and DC power networks. In inverter-based microgrids, however, the increasing reliance on digitally communicated measurements exposes LCDRs to false-data injection attacks (FDIAs), in which adversaries manipulate remote measurement streams to create protection-triggering yet physically inconsistent current trajectories. This paper addresses this emerging measurement integrity problem by introducing a measurement integrity validation scheme that operates as a supervisory instrumentation layer for modern LCDRs. The proposed scheme interprets short windows of synchronized instantaneous current measurements recorded during relay operation and assesses their physical consistency to distinguish genuine fault-induced trajectories from cyber-manipulated measurement streams. A recurrent neural network is trained offline using only relay-available current measurements and exploits the temporal structure of differential current waveforms, which remains informative in inverter-dominated systems where current magnitude is no longer a reliable observable. The method requires no additional sensors, auxiliary protection elements, or prior knowledge of network topology, and is applicable to both AC and DC LCDRs without structural modification. The proposed measurement validation scheme is evaluated on an islanded inverter-based microgrid under a comprehensive set of fault and FDIA scenarios, demonstrating high detection accuracy while preserving relay dependability. Hardware-in-the-loop validation using an OPAL-RT real-time simulator confirms that the scheme satisfies protection timing constraints and can operate in real time under realistic operating conditions. 

---
# FlowPlace: Flow Matching for Chip Placement 

**Authors**: Peng Xie, Ke Xue, Yunqi Shi, Ruo-Tong Chen, Chengrui Gao, Siyuan Xu, Chenjian Ding, Mingxuan Yuan, Chao Qian  

**Link**: [PDF](https://arxiv.org/pdf/2604.23658)  

**Abstract**: Chip placement plays an important role in physical design. While generative models like diffusion models offer promising learning-based solutions, current methods have the following limitations: they use random synthetic data for pre-training, require long sampling times, and often result in overlaps due to their dependence on gradient-based solvers during the sampling process. To overcome these issues, we propose FlowPlace, which features mask-guided synthetic data generation, flow-based efficient training with flexible prior injection, and hard constraint sampling for overlap-free layouts. Experiments on OpenROAD and ICCAD 2015 benchmarks show FlowPlace achieves better PPA metrics, 10-50$\times$ faster sampling efficiency, and zero overlaps. 

---
# ResAF-Net: An Anchor-Free Attention-Based Network for Tree Detection and Agricultural Mapping in Palestine 

**Authors**: Rabee Al-Qasem  

**Link**: [PDF](https://arxiv.org/pdf/2604.23653)  

**Abstract**: Reliable agricultural data is essential for food security, land-use planning, and economic resilience, yet in Palestine, such data remains difficult to collect at scale because of fragmented landscapes, limited field access, and restrictions on aerial monitoring. This paper presents ResAF-Net, a satellite-based tree detection framework designed for large-scale agricultural monitoring in resource-constrained settings. The proposed architecture combines a ResNet-50 encoder, Atrous Spatial Pyramid Pooling (ASPP), a feature-fusion stage, a multi-head self-attention refinement module, and an anchor-free FCOS detection head to improve tree localization in dense and heterogeneous scenes. Trained on the MillionTrees benchmark, the model achieved 82% Recall, 63.03% mAP@0.50, and 35.47% mAP@0.50:0.95 on the validation split, indicating strong sensitivity to tree presence while maintaining competitive localization quality. Beyond benchmark evaluation, we implemented the model within a web-based GIS application integrated with Palestinian cadastral data from GeoMolg, enabling tree analysis at scene, parcel, and community levels. This deployment demonstrates the practical feasibility of AI-assisted agricultural inventorying in Palestine. It provides a foundation for data-driven monitoring, reporting, and future species-level analysis of Mediterranean tree crops. 

---
# RaV-IDP: A Reconstruction-as-Validation Framework for Faithful Intelligent Document Processing 

**Authors**: Pritesh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2604.23644)  

**Abstract**: Intelligent document processing pipelines extract structured entities (tables, images, and text) from documents for use in downstream systems such as knowledge bases, retrieval-augmented generation, and analytics. A persistent limitation of existing pipelines is that extraction output is produced without any intrinsic mechanism to verify whether it faithfully represents the source. Model-internal confidence scores measure inference certainty, not correspondence to the document, and extraction errors pass silently into downstream consumers.
We present Reconstruction as Validation (RaV-IDP), a document processing pipeline that introduces reconstruction as a first-class architectural component. After each entity is extracted, a dedicated reconstructor renders the extracted representation back into a form comparable to the original document region, and a comparator scores fidelity between the reconstruction and the unmodified source crop. This fidelity score is a grounded, label-free quality signal. When fidelity falls below a per-entity-type threshold, a structured GPT-4.1 vision fallback is triggered and the validation loop repeats. We enforce a bootstrap constraint: the comparator always anchors against the original document region, never against the extraction, preventing the validation from becoming circular.
We further propose a per-stage evaluation framework pairing each pipeline component with an appropriate benchmark. The code pipeline is publicly available at this https URL for experimentation and use. 

---
# From Rights to Rites: Expectations Management in Smart-Home AI 

**Authors**: Varad Vishwarupe, Ivan Flechais, Marina Jirotka, Nigel Shadbolt  

**Link**: [PDF](https://arxiv.org/pdf/2604.23635)  

**Abstract**: Domestic voice assistants and smart-home devices are increasingly embedded in everyday routines, yet their ethics are often treated as an afterthought or delegated to compliance teams. To explore how expectations about smart-home AI are constructed and managed, we conducted 33 semi-structured interviews with designers, developers, and researchers from major smart-home platforms (Amazon Alexa, Microsoft Azure IoT, and Google Nest). Using a constructivist grounded theory approach, we develop Expectations Management (EM): a culturally embedded model describing how practitioners shape, calibrate, and repair expectations by balancing organisational rights with culturally situated rites. We show that EM differs from expectation-confirmation theory and trust-calibration by foregrounding moral judgement, situated action, and cross-cultural variation. Our analysis reveals four recurring design tensions: automation vs. autonomy, helpfulness vs. intrusiveness, personalisation vs. predictability, and transparency vs. obscurity and distils them into a five-phase EM Design Playbook that supports moral prudence. We discuss implications for responsible smart-home design and offer guidance for human-centred AI. 

---
# Applications of the Transformer Architecture in AI-Assisted English Reading Comprehension 

**Authors**: Ping Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.23615)  

**Abstract**: This paper studies interpretable and fair artificial intelligence architectures for understanding English reading. Introduced transformer-based models, integrating advanced attention mechanisms and gradient-based feature attribution. The model's lack of interpretability, reduction of algorithmic bias, and unreliable performance in learning environments are the current issues faced in natural language teaching. A unified technical pipeline has been constructed, including adversarial bias correction methods, token-level attribution analysis, and multi-head attention heatmap visualization. Experimental validation was conducted using a large-scale labeled English reading comprehension dataset, and the data partitioning scheme and parameter optimization procedures have been determined. The method significantly outperforms the state-of-the-art models for this task in terms of accuracy and macro-average F1 score; in some aspects, it even surpasses or closely matches the results of human evaluations. In multi-week user experiments, the explainable transformer improved teachers' trust and operability in feedback-based assessments within the scoring system. The proposed method aims to ensure high prediction accuracy and fairness for different learners. This indicates that it is a real-world educational application based on artificial intelligence with a focus on interpretation. Improve the user experience in AI-assisted reading comprehension systems, counteract biases, and enhance the details explained by transformers. 

---
# Partition-of-Unity Gaussian Kolmogorov-Arnold Networks 

**Authors**: Amir Nooeizadegan  

**Link**: [PDF](https://arxiv.org/pdf/2604.23599)  

**Abstract**: Gaussian basis functions provide an efficient and flexible alternative to spline activations in KANs. In this work, we introduce the partition-of-unity Gaussian KAN (PU-GKAN), a Shepard-type normalized Gaussian KAN in which the Gaussian basis values on each edge are divided by their local sum over fixed centers. This produces a partition-of-unity feature map with trainable coefficients, while preserving the standard edge-based KAN structure. The normalized construction gives exact constant reproduction at the edge level and admits an explicit finite-feature kernel interpretation.
We formulate both the standard Gaussian KAN (GKAN) and PU-GKAN from a finite-feature and additive-kernel viewpoint, making the induced layer kernels and empirical feature matrices explicit. Using the first-layer feature matrix as the reference object, we adopt a practical scale-selection interval for \(\epsilon\), with the lower endpoint determined by adjacent-center overlap and the upper endpoint determined by a conservative conditioning threshold. Numerical experiments show that PU-GKAN reduces sensitivity to \(\epsilon\), improves validation accuracy for most smooth and moderately non-smooth targets, and gives more stable training behavior. The benefit persists across sample-size and center-number sweeps, higher-dimensional architectures, Matérn RBF bases, and physics-informed examples involving Helmholtz and wave equations. These results indicate that Shepard-type partition-of-unity normalization is a simple and effective stabilization mechanism for RBF-based KANs. 

---
# PhysCodeBench: Benchmarking Physics-Aware Symbolic Simulation of 3D Scenes via Self-Corrective Multi-Agent Refinement 

**Authors**: Tianyidan Xie, Peiyu Wang, Yuyi Qian, Yuxuan Wang, Rui Ma, Ying Tai, Song Wu, Qian Wang, Lanjun Wang, Zili Yi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23580)  

**Abstract**: Physics-aware symbolic simulation of 3D scenes is critical for robotics, embodied AI, and scientific computing, requiring models to understand natural language descriptions of physical phenomena and translate them into executable simulation environments. While large language models (LLMs) excel at general code generation, they struggle with the semantic gap between physical descriptions and simulation implementation. We introduce PhysCodeBench, the first comprehensive benchmark for evaluating physics-aware symbolic simulation, comprising 700 manually-crafted diverse samples across mechanics, fluid dynamics, and soft-body physics with expert annotations. Our evaluation framework measures both code executability and physical accuracy through automated and visual assessment. Building on this, we propose a Self-Corrective Multi-Agent Refinement Framework (SMRF) with three specialized agents (simulation generator, error corrector, and simulation refiner) that collaborate iteratively with domain-specific validation to produce physically accurate simulations. SMRF achieves 67.7 points overall performance compared to 36.3 points for the best baseline among evaluated SOTA models, representing a 31.4-point improvement. Our analysis demonstrates that error correction is critical for accurate physics-aware symbolic simulation and that specialized multi-agent approaches significantly outperform single-agent methods across the tested physical domains. 

---
# LLMs Reading the Rhythms of Daily Life: Aligned Understanding for Behavior Prediction and Generation 

**Authors**: Fanjin Meng, Jingtao Ding, Nian Li, Yizhou Sun, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.23578)  

**Abstract**: Human daily behavior unfolds as complex sequences shaped by intentions, preferences, and context. Effectively modeling these behaviors is crucial for intelligent systems such as personal assistants and recommendation engines. While recent advances in deep learning and behavior pre-training have improved behavior prediction, key challenges remain--particularly in handling long-tail behaviors, enhancing interpretability, and supporting multiple tasks within a unified framework. Large language models (LLMs) offer a promising direction due to their semantic richness, strong interpretability, and generative capabilities. However, the structural and modal differences between behavioral data and natural language limit the direct applicability of LLMs.
To address this gap, we propose Behavior Understanding Alignment (BUA), a novel framework that integrates LLMs into human behavior modeling through a structured curriculum learning process. BUA employs sequence embeddings from pretrained behavior models as alignment anchors and guides the LLM through a three-stage curriculum, while a multi-round dialogue setting introduces prediction and generation capabilities. Experiments on two real-world datasets demonstrate that BUA significantly outperforms existing methods in both tasks, highlighting its effectiveness and flexibility in applying LLMs to complex human behavior modeling. 

---
# CAPSULE: Control-Theoretic Action Perturbations for Safe Uncertainty-Aware Reinforcement Learning 

**Authors**: Rahul Narava, Siddharth Verma, Ojas Jain, Shashi Shekhar Jha, Mayank Shekhar Jha  

**Link**: [PDF](https://arxiv.org/pdf/2604.23576)  

**Abstract**: Ensuring safe exploration in high-dimensional systems with unknown dynamics remains a significant challenge. Existing safe reinforcement learning methods often provide safety guarantees only in expectation, which can still lead to safety violations. Control-theoretic approaches, in contrast, offer hard constraint-based safety guarantees but typically assume access to known system dynamics or require accurate estimation of control-affine models. In this paper, we propose a safe reinforcement learning framework that learns a probabilistic control-affine dynamics model in an offline setting. The learned model is leveraged to explicitly construct control barrier functions (CBFs) that incorporate model uncertainty to provide conservative safety constraints. These CBF constraints are enforced through an online constraint-based action correction mechanism, enabling safe exploration without overly restricting task performance. Empirical evaluations on nonlinear, complex continuous-control benchmarks demonstrate that our approach achieves returns comparable to those of existing baselines while significantly reducing safety violations. 

---
# CyberCane: Neuro-Symbolic RAG for Privacy-Preserving Phishing Detection with Formal Ontology Reasoning 

**Authors**: Safayat Bin Hakim, Aniqa Afzal, Qi Zhao, Vigna Majmundar, Pawel Sloboda, Houbing Herbert Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.23563)  

**Abstract**: Privacy-critical domains require phishing detection systems that satisfy contradictory constraints: near-zero false positives to prevent workflow disruption, transparent explanations for non-expert staff, strict regulatory compliance prohibiting sensitive data exposure to external APIs, and robustness against AI-generated attacks. Existing rule-based systems are brittle to novel campaigns, while LLM-based detectors violate privacy regulations through unredacted data transmission. We introduce CyberCane, a neuro-symbolic framework integrating deterministic symbolic analysis with privacy-preserving retrieval-augmented generation (RAG). Our dual-phase pipeline applies lightweight symbolic rules to email metadata, then escalates borderline cases to semantic classification via RAG with automated sensitive data redaction and retrieval from a phishing-only corpus. We further introduce PhishOnt, an OWL ontology enabling verifiable attack classification through formal reasoning chains. Evaluation on DataPhish2025 (12.3k emails; mixed human/LLM) and Nazario/SpamAssassin demonstrates a 78.6-point recall gain over symbolic-only detection on AI-generated threats, with precision exceeding 98% and FPR as low as 0.16%. Healthcare deployment projects a 542x ROI; tunable operating points support diverse risk tolerances, with open-source implementation at this https URL. 

---
# EyeBrain: Left and Right Brain Lateralization Activity Classification Through Pupil Diameter and Fixation Duration 

**Authors**: Ko Watanabe, Pooja Pol, Nicolas Großmann, Shoya Ishimaru, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2604.23562)  

**Abstract**: The relationship between brain lateralization and cognitive functions is well-documented. The left hemisphere primarily handles tasks such as language and arithmetic, while the right hemisphere is involved in creative activities like drawing and music perception. Eye-tracking technology has shown the potential to reveal cognitive states by measuring ocular metrics such as pupil diameter and fixation duration. However, the ability to distinguish lateralized brain activity using these ocular metrics remains underexplored. Here, we demonstrate that pupil diameter and fixation duration can effectively classify left and right brain hemisphere activities. We obtained a considerably high classification performance, with an F1 score of 0.894. The results suggest that ocular metrics are robust indicators of lateralized brain activity and can be applied in cognitive monitoring and neurorehabilitation. Our future work expands on this by integrating these methods into real-time applications EyeBrain, potentially broadening their use across various cognitive and neurological domains. 

---
# DLM: Unified Decision Language Models for Offline Multi-Agent Sequential Decision Making 

**Authors**: Zhuohui Zhang, Bin Cheng, Bin He  

**Link**: [PDF](https://arxiv.org/pdf/2604.23557)  

**Abstract**: Building scalable and reusable multi-agent decision policies from offline datasets remains a challenge in offline multi-agent reinforcement learning (MARL), as existing methods often rely on fixed observation formats and action spaces that limit generalization. In contrast, large language models (LLMs) offer a flexible modeling interface that can naturally accommodate heterogeneous observations and actions. Motivated by this, we propose the Decision Language Model (DLM), which formulates multi-agent decision making as a dialogue-style sequence prediction problem under the centralized training with decentralized execution paradigm. DLM is trained in two stages: a supervised fine-tuning phase, which leverages dialogue-style datasets for centralized training with inter-agent context and generates executable actions from offline trajectories, followed by a group relative policy optimization phase to enhance robustness to out-of-distribution actions through lightweight reward functions. Experiments on multiple benchmarks show that a unified DLM outperforms strong offline MARL baselines and LLM-based conversational decision-making methods, while demonstrating strong zero-shot generalization to unseen scenarios across tasks. 

---
# On the Memorization of Consistency Distillation for Diffusion Models 

**Authors**: Bingqing Jiang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2604.23552)  

**Abstract**: Diffusion models are central to modern generative modeling, and understanding how they balance memorization and generalization is critical for reliable deployment. Recent work has shown that memorization in diffusion models is shaped by training dynamics, with generalization and memorization emerging at different stages of training. However, deployed diffusion models are often further distilled, introducing an additional training phase whose impact on memorization is not well understood. In this work, we analyze how distillation reshapes memorization behavior in diffusion models, taking consistency distillation as a representative framework. Empirically, we show that when applied to a teacher model that has memorized data, consistency distillation significantly reduces transferred memorization in the student while preserving, and sometimes improving, sample quality. To explain this behavior, we provide a theoretical analysis using a random feature neural network model [Bonnaire et al., 2025], showing that consistency distillation suppresses unstable feature directions associated with memorization while preserving stable, generalizable modes. Our findings suggest that distillation can serve not only as an acceleration tool, but also as a mechanism for improving the memorization-generalization trade-off. 

---
# Hardware-Efficient FPGA Implementation of Sigmoid Function Using Mixed-Radix Hyperbolic Rotation CORDIC 

**Authors**: Chintan Panchal, Ankur Changela, Mohendra Roy  

**Link**: [PDF](https://arxiv.org/pdf/2604.23547)  

**Abstract**: Efficient hardware implementation of nonlinear activation functions is a crucial task in deploying artificial neural networks on resource-constrained and edge devices such as Field-Programmable Gate Arrays (FPGAs). The sigmoid activation function is widely used for probabilistic output, binary classification, and gating mechanisms in recurrent neural networks, despite its reliance on exponential computations.
This paper presents a hardware-efficient FPGA implementation of the sigmoid activation function using a mixed-radix CORDIC-based architecture. The proposed approach leverages the mathematical relationship between the sigmoid and hyperbolic tangent functions. The input range is normalized to 1, enabling the corresponding tanh computation to operate within a reduced range of 0.5, which significantly improves convergence behavior.
To achieve high accuracy with minimal hardware overhead, a modified mixed-radix hyperbolic rotation CORDIC (MR-HRC) algorithm combining radix-2 and radix-4 iterations is introduced. The initial radix-2 stage ensures stable convergence, while the subsequent radix-4 stage accelerates convergence without requiring scale-factor compensation. In the final stage, a radix-2 linear vectoring CORDIC (R2-LVC) is used to compute the hyperbolic tangent by dividing hyperbolic sine and cosine values derived from the MR-HRC algorithm.
The entire architecture is fully pipelined and implemented on an FPGA. The design is realized on an Xilinx Virtex-7 FPGA using a 16-bit fixed-point representation. Experimental results demonstrate a significant reduction in hardware utilization, requiring only 835 logic slices with zero DSP usage. Additionally, the design achieves a mean absolute error of 4.23 10^-4, outperforming several recent sigmoid implementations. 

---
# COMO: Closed-Loop Optical Molecule Recognition with Minimum Risk Training 

**Authors**: Zhuoqi Lyu, Qing Ke  

**Link**: [PDF](https://arxiv.org/pdf/2604.23546)  

**Abstract**: Optical chemical structure recognition (OCSR) translates molecular images into machine-readable representations like SMILES strings or molecular graphs, but remains challenging in real-world documents due to inexhaustible variations in chemical structures, shorthand conventions, and visual noise. Most existing deep-learning-based approaches rely on teacher forcing with token-level Maximum Likelihood Estimation (MLE). This training paradigm suffers from exposure bias, as models are trained under ground-truth prefixes but must condition on their own previous predictions during inference. Moreover, token-level MLE objectives hinder the optimization towards molecular-level evaluation criteria such as chemical validity and structural similarity. Here we introduce Minimum Risk Training (MRT) to OCSR and propose COMO (Closed-loop Optical Molecule recOgnition), a closed-loop framework that mitigates exposure bias by directly optimizing over molecule-level, non-differentiable objectives, by iteratively sampling and evaluating the model's own predictions. Experiments on ten benchmarks including synthetic and real-world chemical diagrams from patent and scientific literature demonstrate that COMO substantially outperforms existing rule-based and learning-based methods with less training data. Ablation studies further show that MRT is architecture-agnostic, demonstrating its potential for broad application to end-to-end OCSR systems. 

---
# Pref-CTRL: Preference Driven LLM Alignment using Representation Editing 

**Authors**: Imranul Ashrafi, Inigo Jauregi Unanue, Massimo Piccardi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23543)  

**Abstract**: Test-time alignment methods offer a promising alternative to fine-tuning by steering the outputs of large language models (LLMs) at inference time with lightweight interventions on their internal representations. Recently, a prominent and effective approach, RE-Control (Kong et al., 2024), has proposed leveraging an external value function trained over the LLM's hidden states to guide generation via gradient-based editing. While effective, this method overlooks a key characteristic of alignment tasks, i.e. that they are typically formulated as learning from human preferences between candidate responses. To address this, in this paper we propose a novel preference-based training framework, Pref-CTRL, that uses a multi-objective value function to better reflect the structure of preference data. Our approach has outperformed RE-Control on two benchmark datasets and showed greater generalization on out-of-domain datasets. Our source code is available at this https URL. 

---
# Emotion-Conditioned Short-Horizon Human Pose Forecasting with a Lightweight Predictive World Model 

**Authors**: Jingni Huang, Peter Bloodsworth  

**Link**: [PDF](https://arxiv.org/pdf/2604.23532)  

**Abstract**: Short-term human pose prediction plays a crucial role in interactive systems, assistive robots, and emotion-aware human-computer interaction[1-3]. While current trajectory prediction models primarily rely on geometric motion cues, they often overlook the underlying emotional signals influencing human motion dynamics[4-5]. This paper investigates whether facial expression-derived emotion embeddings can provide auxiliary conditional signals for short-term pose prediction. To further evaluate multimodal conditionation in a recursive prediction setting, we propose a lightweight autoregressive predictive world model that performs 15-step rolling pose prediction. This framework combines pose keypoints with emotion embeddings through a learnable gating mechanism and performs autoregressive unfolding prediction using a recurrent sequence model based on a two-layer LSTM architecture. Experiments were conducted on two small-scale pose-emotion video datasets: controlled motion sequences with minimal facial expression changes and, natural emotion-driven motion sequences with considerable facial expression changes. The results show that simple multimodal fusion does not consistently improve prediction accuracy, while normalized gating fusion significantly enhances the performance of emotion-driven motion sequences. Furthermore, counterfactual perturbation experiments demonstrate that the predicted trajectory exhibits measurable sensitivity to changes in multimodal input, suggesting that facial expression embeddings act as auxiliary conditional signals rather than redundant features. In summary, these results indicate that incorporating facial expression-derived emotion embeddings into emotion-conditional short-term pose prediction based on a lightweight predictive world model architecture is a feasible approach. 

---
# MTRouter: Cost-Aware Multi-Turn LLM Routing with History-Model Joint Embeddings 

**Authors**: Yiqun Zhang, Hao Li, Zihan Wang, Shi Feng, Xiaocui Yang, Daling Wang, Bo Zhang, Lei Bai, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23530)  

**Abstract**: Multi-turn, long-horizon tasks are increasingly common for large language models (LLMs), but solving them typically requires many sequential model invocations, accumulating substantial inference costs. Here, we study cost-aware multi-turn LLM routing: selecting which model to invoke at each turn from a model pool, given a fixed cost budget. We propose MTRouter, which encodes the interaction history and candidate models into joint history-model embeddings, and learns an outcome estimator from logged trajectories to predict turn-level model utility. Experiments show that MTRouter improves the performance-cost trade-off: on ScienceWorld, it surpasses GPT-5 while reducing total cost by 58.7%; on Humanity's Last Exam (HLE), it achieves competitive accuracy while reducing total cost by 43.4% relative to GPT-5, and these gains even carry over to held-out tasks. Further analyses reveal several mechanisms underlying its effectiveness: relative to prior multi-turn routers, MTRouter makes fewer model switches, is more tolerant to transient errors, and exhibits emergent specialization across models. Code: this https URL 

---
# Grammar-Constrained Refinement of Safety Operational Rules Using Language in the Loop: What Could Go Wrong 

**Authors**: Khouloud Gaaloul, Zaid Ghazal, Madhu Latha Pulimi, Sam Emmanuel Kathiravan  

**Link**: [PDF](https://arxiv.org/pdf/2604.23523)  

**Abstract**: Safety specifications in cyber-physical systems (CPS) capture the operational conditions the system must satisfy to operate safely within its intended environment. As operating environments evolve, operational rules must be continuously refined to preserve consistency with observed system behavior during simulation-based verification and validation. Revising inconsistent rules is challenging because the changes must remain syntactically correct under a domain-specific grammar. Language-in-the-loop refinement further raises safety concerns beyond syntactic violations, as it can produce semantically unjustified refinements that overfit to the observed outcomes. We introduce a framework that combines counterfactual reasoning with a grammar-constrained refinement loop to refine operational rules, aligning them with the observed system behavior. Applied to an autonomous driving control system, our approach successfully resolved the inconsistencies in an operational rule inferred by a conventional baseline while remaining grammar compliant. An empirical large language model (LLM) study further revealed model-dependent refinement quality and safety lessons, which motivate rigorous grammar enforcement, stronger semantic validation, and broader evaluation in future work. 

---
# Autocorrelation Reintroduces Spectral Bias in KANs for Time Series Forecasting 

**Authors**: Chen Zeng, Jiahui Wang, Qiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23518)  

**Abstract**: Existing theory suggests that Kolmogorov-Arnold Networks (KANs) can overcome the spectral bias commonly observed in neural networks under the assumption that inputs are statistically independent. However, this assumption does not hold in time series forecasting (TSF), where inputs are lagged observations with strong temporal autocorrelation. Through theoretical analysis and empirical validation, we obtain an unexpected finding: temporal autocorrelation reintroduces spectral bias in KANs, and the bias becomes increasingly pronounced as the degree of autocorrelation increases. This suggests that standard KANs may face substantial difficulties in TSF with strongly autocorrelated inputs. To address this problem, we introduce the Discrete Cosine Transform (DCT) to reduce the correlations among the network inputs. As expected, experimental results reveal that DCT preprocessing substantially reduces the observed low-frequency preference in TSF. This result also corroborates that the spectral bias of KANs in TSF tasks is indeed induced by the autocorrelation among input variables. 

---
# Uncertainty Propagation in LLM-Based Systems 

**Authors**: Boming Xia, Liming Zhu, Erdun Gao, Qinghua Lu, Minhui Xue, Dino Sejdinovic  

**Link**: [PDF](https://arxiv.org/pdf/2604.23505)  

**Abstract**: Uncertainty in large language model (LLM)-based systems is often studied at the level of a single model output, yet deployed LLM applications are compound systems in which uncertainty is transformed and reused across model internals, workflow stages, component boundaries, persistent state, and human or organisational processes. Without principled treatment of how uncertainty is carried and reused across these boundaries, early errors can propagate and compound in ways that are difficult to detect and govern. This paper develops a systems-level account of uncertainty propagation. It introduces a conceptual framing for characterising propagated uncertainty signals, presents a structured taxonomy spanning intra-model (P1), system-level (P2), and socio-technical (P3) propagation mechanisms, synthesises cross-cutting engineering insights, and identifies five open research challenges. 

---
# Interpretable Physics-Informed Load Forecasting for U.S. Grid Resilience: SHAP-Guided Ensemble Validation in Hybrid Deep Learning Under Extreme Weather 

**Authors**: Md Abubakkar, Sajib Debnath, Md. Uzzal Mia  

**Link**: [PDF](https://arxiv.org/pdf/2604.23500)  

**Abstract**: Accurate short-term electricity load forecasting is a cornerstone of U.S. grid reliability; however, prevailing deep learning models remain opaque, limiting operator trust during extreme weather. A unified, interpretable, physics-informed ensemble framework is proposed, integrating a Convolutional Neural Network (CNN) branch for local feature extraction and a Transformer branch for long-range dependency modeling; the branches are fused through a validation-optimized weighted ensemble and regularized by a physics-informed loss derived from the piecewise parabolic temperature-demand relationship of the Electric Reliability Council of Texas (ERCOT) system. Post-hoc interpretability is provided through SHapley Additive exPlanations (SHAP) with the DeepExplainer backend, yielding global and event-level attributions. Using eight years of ERCOT hourly load data (2018-2025) fused with Automated Surface Observing System (ASOS) records from three Texas stations, the framework achieves 713 MW MAE, 812 MW RMSE, and 1.18% MAPE on the test window. For Hampel-flagged extreme events, MAPE falls by 20.7% relative to its Transformer branch and by 40.5% relative to its CNN branch; an ablation confirms that the parabolic and ramp constraints drive a 14.7% RMSE reduction. SHAP analysis reveals a regime shift: temperature dominates under normal operation, whereas wind speed and precipitation become more influential during cold fronts and heatwaves. 

---
# K-SENSE: A Knowledge-Guided Self-Augmented Encoder for Neuro-Semantic Evaluation of Mental Health Conditions on Social Media 

**Authors**: Vijay Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2604.23493)  

**Abstract**: Early detection of mental health conditions, particularly stress and depression, from social media text remains a challenging open problem in computational psychiatry and natural language processing. Automated systems must contend with figurative language, implicit emotional expression, and the high noise inherent in user-generated content. Existing approaches either leverage external commonsense knowledge to model mental states explicitly, or apply self-augmentation and contrastive training to improve generalization, but seldom do both in a principled, unified framework. We propose K-SENSE (Knowledge-guided Self-augmented Encoder for Neuro-Semantic Evaluation of Mental Health), a framework that jointly exploits external psychological reasoning and internal representation robustness. K-SENSE adopts a three-stage encoding pipeline: (1) inferential commonsense knowledge is extracted from the COMET model across five mental state dimensions; (2) a semantic anchor is constructed by combining hidden representations from two parallel encoding streams, projected into a shared space before fusion; and (3) a supervised contrastive learning objective aligns same-class representations while encouraging the attention mechanism to suppress irrelevant knowledge noise. We evaluate K-SENSE on Dreaddit (stress detection) and Depression_Mixed (depression detection), achieving mean F1-scores of 86.1 (0.6%) and 94.3 (0.8%), respectively, over five independent runs. These represent improvements of approximately 2.6 and 1.5 percentage points over the strongest prior baselines. Ablation experiments confirm the contribution of each architectural component, including the temporal knowledge integration strategy and the choice to keep the knowledge encoder frozen during fine-tuning. 

---
# A Milestone in Formalization: The Sphere Packing Problem in Dimension 8 

**Authors**: Sidharth Hariharan, Christopher Birkbeck, Seewoo Lee, Ho Kiu Gareth Ma, Bhavik Mehta, Auguste Poiroux, Maryna Viazovska  

**Link**: [PDF](https://arxiv.org/pdf/2604.23468)  

**Abstract**: In 2016, Viazovska famously solved the sphere packing problem in dimension $8$, using modular forms to construct a 'magic' function satisfying optimality conditions determined by Cohn and Elkies in 2003. In March 2024, Hariharan and Viazovska launched a project to formalize this solution and related mathematical facts in the Lean Theorem Prover. A significant milestone was achieved in February 2026: the result was formally verified, with the final stages of the verification done by Math, Inc.'s autoformalization model 'Gauss'. We discuss the techniques used to achieve this milestone, reflect on the unique collaboration between humans and Gauss, and discuss project objectives that remain. 

---
# Hybrid JIT-CUDA Graph Optimization for Low-Latency Large Language Model Inference 

**Authors**: Divakar Kumar Yadav, Tian Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.23467)  

**Abstract**: Large Language Models (LLMs) have achieved strong performance across natural language and multimodal tasks, yet their practical deployment remains constrained by inference latency and kernel launch overhead, particularly in interactive, short-sequence settings. This paper presents a hybrid runtime framework that combines Just-In-Time (JIT) compilation with CUDA Graph execution to reduce launch overhead while preserving runtime flexibility during autoregressive decoding. The framework partitions transformer inference into static components executed via CUDA Graph replay and dynamic components handled through JIT-compiled kernels, enabling asynchronous graph capture and reuse across decoding steps.
We evaluate the proposed approach on LLaMA-2 7B using single-GPU, batch-size-one inference across prompt lengths from 10 to 500 tokens. Experimental results show that the hybrid runtime reduces Time-to-First-Token (TTFT) by up to 66.0% and achieves lower P99 latency compared with TensorRT-LLM in this regime. These results indicate that hybrid JIT-CUDA Graph execution can effectively reduce inference latency and variance for short-sequence LLM workloads, making it a practical optimization strategy for latency-sensitive AI applications. 

---
# Evaluating CUDA Tile for AI Workloads on Hopper and Blackwell GPUs 

**Authors**: Divakar Kumar Yadav, Tian Zhao, Deepak Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.23466)  

**Abstract**: NVIDIA's CUDA Tile (CuTile) introduces a Python-based, tile-centric abstraction for GPU kernel development that aims to simplify programming while retaining Tensor Core and Tensor Memory Accelerator (TMA) efficiency on modern GPUs. We present the first independent, cross-architecture evaluation of CuTile against established approaches such as cuBLAS, Triton, WMMA, and raw SIMT on three NVIDIA GPUs spanning Hopper and Blackwell: H100 NVL, B200, and RTX PRO 6000 Blackwell Server Edition. We benchmark representative AI workloads, including GEMM, fused multi-head attention, and end-to-end LLM inference in BF16/FP16 precision, to assess both performance and portability.
Our results show that CuTile effectiveness is strongly workload- and architecture-dependent. On datacenter-class Blackwell (B200), CuTile achieves up to 1007 TFLOP/s for fused attention, outperforming FlashAttention-2 by 2.5x while requiring only 60 lines of Python kernel code. For GEMM, CuTile reaches 52-79% of cuBLAS performance in 22 lines of code (versus 123 for WMMA), making it a practical replacement for hand-written CUDA kernels but not yet for vendor-optimized libraries. However, the same CuTile attention kernel achieves only 53% of FlashAttention-2 throughput on RTX PRO 6000 (sm_120), exposing significant cross-architecture optimization gaps. In contrast, Triton sustains 62-101% of cuBLAS performance across all tested platforms without architecture-specific tuning, demonstrating substantially stronger portability. 

---
# AI Safety Training Can be Clinically Harmful 

**Authors**: Suhas BN, Andrew M. Sherrill, Rosa I. Arriaga, Chris W. Wiese, Saeed Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2604.23445)  

**Abstract**: Large language models are being deployed as mental health support agents at scale, yet only 16% of LLM-based chatbot interventions have undergone rigorous clinical efficacy testing, and simulations reveal psychological deterioration in over one-third of cases. We evaluate four generative models on 250 Prolonged Exposure (PE) therapy scenarios and 146 CBT cognitive restructuring exercises (plus 29 severity-escalated variants), scored by a three-judge LLM panel. All models scored near-perfectly on surface acknowledgment (~0.91-1.00) while therapeutic appropriateness collapsed to 0.22-0.33 at the highest severity for three of four models, with protocol fidelity reaching zero for two. Under CBT severity escalation, one model's task completeness dropped from 92% to 71% while the frontier model's safety-interference score fell from 0.99 to 0.61. We identify a systematic, modality-spanning failure: RLHF safety alignment disrupts the therapeutic mechanism of action by grounding patients during imaginal exposure, offering false reassurance, inserting crisis resources into controlled exercises, and refusing to challenge distorted cognitions mentioning self-harm in PE; and through task abandonment or safety-preamble insertion during CBT cognitive restructuring. These findings motivate a five-axis evaluation framework (protocol fidelity, hallucination risk, behavioral consistency, crisis safety, demographic robustness), mapped onto FDA SaMD and EU AI Act requirements. We argue that no AI mental health system should proceed to deployment without passing multi-axis evaluation across all five dimensions. 

---
# Knee-xRAI: An Explainable AI Framework for Automatic Kellgren-Lawrence Grading of Knee Osteoarthritis 

**Authors**: Azmul A. Irfan, Nur Ahmad Khatim, Alfan Alfian Irfan, Achmad Zaki, Erike A. Suwarsono, Mansur M. Arief  

**Link**: [PDF](https://arxiv.org/pdf/2604.23435)  

**Abstract**: Radiographic grading of knee osteoarthritis (KOA) with the Kellgren-Lawrence (KL) system is limited by inter-reader variability and the opacity of current deep learning approaches, which predict KL grades directly from images without decomposing structural features. We present Knee-xRAI, a modular framework that independently quantifies the three cardinal radiographic features of KOA (joint space narrowing [JSN], osteophytes, and subchondral sclerosis) and integrates them into an explainable KL grade classification. The pipeline combines U-Net++ segmentation for contour-based JSN measurement, an SE-ResNet-50 network for per-site osteophyte grading (OARSI scale), and a hybrid texture-CNN classifier for binary sclerosis quantification. The resulting 50-dimensional structured feature vector feeds two complementary classification paths. An XGBoost path supports SHAP-based feature attribution. A ConvNeXt hybrid path combines the structured vector with a full-image encoder for enhanced predictive performance. Evaluated on 8,260 radiographs from an OAI-derived dataset, the JSN module achieved a Dice coefficient of 0.8909 and an mJSW intraclass correlation of 0.8674 against manual annotations. The ConvNeXt hybrid path reached a test quadratic weighted kappa (QWK) of 0.8436 and AUC of 0.9017. The transparent XGBoost path achieved a test QWK of 0.6294 with full feature-level audit capability. Ablation confirmed JSN as the dominant predictor (QWK = 0.6103 alone), with osteophyte features providing consistent incremental gain (+0.0183) and sclerosis contributing marginally. Inference-time ablation of Path B confirmed the structured pathway contributes materially beyond the image encoder, with QWK drops of 0.098 (feature zeroing) and 0.284 (feature-image permutation). Knee-xRAI explicitly quantifies all three KL-defining radiographic features within a single auditable pipeline. 

---
# Sphere-Depth: A Benchmark for Depth Estimation Methods with Varying Spherical Camera Orientations 

**Authors**: Soulayma Gazzeh, Giuseppe Mazzola, Liliana Lo Presti, Marco La Cascia  

**Link**: [PDF](https://arxiv.org/pdf/2604.23432)  

**Abstract**: Reliable depth estimation from spherical images is crucial for 360° vision in robotic navigation and immersive scene understanding. However, the onboard spherical camera can experience unintentional pose variations in real-world robotic platforms that, along with the geometric distortions inherent in equirectangular projections, significantly impact the effectiveness of depth estimation. To study this issue, a novel public benchmark, called Sphere-Depth, is introduced to systematically evaluate the robustness of monocular depth estimation models from equirectangular images in a reproducible way. Camera pose perturbations are simulated and used to assess the performance of a popular perspective-based model, Depth Anything, and of spherical-aware models such as Depth Anywhere, ACDNet, Bifuse++, and SliceNet. Furthermore, to ensure meaningful evaluation across models, a depth calibration-based error protocol is proposed to convert predicted relative depth values into metric depth values using supervised learned scaling factors for each model. Experiments show that even models explicitly designed to process spherical images exhibit substantial performance degradation when variations in the camera pose are observed with respect to the canonical pose. The full benchmark, evaluation protocol, and dataset splits are made publicly available at: this https URL 

---
# Automating Categorization of Scientific Texts with In-Context Learning and Prompt-Chaining in Large Language Models 

**Authors**: Gautam Kishore Shahi, Oliver Hummel  

**Link**: [PDF](https://arxiv.org/pdf/2604.23430)  

**Abstract**: The relentless expansion of scientific literature presents significant challenges for navigation and knowledge discovery. Within Research Information Retrieval, established tasks such as text summarization and classification remain crucial for enabling researchers and practitioners to effectively navigate this vast landscape, so that efforts have increasingly been focused on developing advanced research information systems. These systems aim not only to provide standard keyword-based search functionalities but also to incorporate capabilities for automatic content categorization within knowledge-intensive organizations across academia and industry. This study systematically evaluates the performance of off-the-shelf Large Language Models (LLMs) in analyzing scientific texts according to a given classification scheme. We utilized the hierarchical ORKG taxonomy as a classification framework, employing the FORC dataset as ground truth. We investigated the effectiveness of advanced prompt engineering strategies, namely In-Context Learning (ICL) and Prompt Chaining, and experimentally explored the influence of the LLMs' temperature hyperparameter on classification accuracy. Our experiments demonstrate that Prompt Chaining yields superior classification accuracy compared to pure ICL, particularly when applied to the nested structure of the ORKG taxonomy. LLMs with prompt chaining outperform the state-of-the-art models for domain (1st level) prediction and show even better performance for subject (2nd level) prediction compared to the older BERT model. However, LLMs are not yet able to perform well in classifying the topic (3rd level) of research areas based on this specific hierarchical taxonomy, as they only reach about 50% accuracy even with prompt chaining. 

---
# PushupBench: Your VLM is not good at counting pushups 

**Authors**: Shengzhi Li, Jiarun Chen, Karun Sharma, Jiaqi Su, Shichao Pei  

**Link**: [PDF](https://arxiv.org/pdf/2604.23407)  

**Abstract**: Large vision-language models (VLMs) can recognize \textit{what} happens in video but fail to count \textit{how many} times. We introduce \textbf{PushupBench}, 446 long-form clips (avg. 36.7s) for evaluating repetition counting. The best frontier model achieves 42.1\% exact accuracy; open-source 4B models score $\sim$6\%, matching supervised baselines. We show that accuracy alone misleads -- weaker models exploit the modal count rather than reason temporally. Fine-tuning on counting with 1k samples transfers to general video understanding: MVBench (+2.15), PerceptionTest (+1.88), TVBench (+4.54), suggesting counting is a proxy for broader temporal this http URL incorporated in \texttt{lmms-eval} (this https URL) and hosted on (this http URL) 

---
# Learn&Drop: Fast Learning of CNNs based on Layer Dropping 

**Authors**: Giorgio Cruciata, Luca Cruciata, Liliana Lo Presti, Jan Van Gemert, Marco La Cascia  

**Link**: [PDF](https://arxiv.org/pdf/2604.23403)  

**Abstract**: This paper proposes a new method to improve the training efficiency of deep convolutional neural networks. During training, the method evaluates scores to measure how much each layer's parameters change and whether the layer will continue learning or not. Based on these scores, the network is scaled down such that the number of parameters to be learned is reduced, yielding a speed up in training. Unlike state-of-the-art methods that try to compress the network to be used in the inference phase or to limit the number of operations performed in the backpropagation phase, the proposed method is novel in that it focuses on reducing the number of operations performed by the network in the forward propagation during training. The proposed training strategy has been validated on two widely used architecture families: VGG and ResNet. Experiments on MNIST, CIFAR-10 and Imagenette show that, with the proposed method, the training time of the models is more than halved without significantly impacting accuracy. The FLOPs reduction in the forward propagation during training ranges from 17.83\% for VGG-11 to 83.74\% for ResNet-152. These results demonstrate the effectiveness of the proposed technique in speeding up learning of CNNs. The technique will be especially useful in applications where fine-tuning or online training of convolutional models is required, for instance because data arrive sequentially. 

---
# Lost in Decoding? Reproducing and Stress-Testing the Look-Ahead Prior in Generative Retrieval 

**Authors**: Kidist Amde Mekonnen, Yongkang Li, Yubao Tang, Simon Lupart, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2604.23396)  

**Abstract**: Generative retrieval (GR) ranks documents by autoregressively generating document identifiers. Because many GR methods rely on trie-constrained beam search, they are vulnerable to early pruning of relevant prefixes under finite-beam decoding. Planning Ahead in Generative Retrieval (PAG) mitigates this failure mode by using simultaneous decoding to compute a document-level look-ahead prior that guides subsequent sequential decoding. We reproduce PAG at inference time and stress-test its decoding behavior. Using the authors' released checkpoint and identifier/trie artifacts under the reported decoding setup, we reproduce the main effectiveness results on MS MARCO Dev and TREC-DL 2019/2020, and corroborate the reported beam-size-latency trade-off in our hardware setting. Beyond reproduction, we introduce plan drift diagnostics that quantify how intent-preserving query variations alter the planner's top-n candidate set and highest-weight planner tokens, and how these changes affect guided decoding. We find that PAG's planning signal is brittle under lexical surface-form variation: intent-preserving typos can trigger plan collapse, where the planned candidate pool shifts enough that the look-ahead bonus provides little useful guidance, effectively reverting decoding toward weaker unguided search. We further evaluate fixed-index cross-lingual robustness using non-English mMARCO queries against an English index, and assess query-side mitigation strategies that require no re-indexing; query translation provides the strongest recovery in our setting. Overall, our results confirm PAG's reported effectiveness and the benefit of planning-guided decoding under the released inference setup, while showing that these gains depend on the stability of the planning signal under realistic query variation and query-document mismatch. 

---
# A Parametric Memory Head for Continual Generative Retrieval 

**Authors**: Kidist Amde Mekonnen, Yubao Tang, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2604.23388)  

**Abstract**: Generative information retrieval (GenIR) consolidates retrieval into a single neural model that decodes document identifiers (docids) directly from queries. While this model-as-index paradigm offers architectural simplicity, it is poorly suited to dynamic document collections. Unlike modular systems, where indexes are easily updated, GenIR's knowledge is parametrically encoded in its weights; consequently, standard adaptation methods such as full and parameter-efficient fine-tuning can induce catastrophic forgetting. We show that sequential adaptation improves retrieval on newly added documents but substantially degrades performance on earlier slices, exposing a pronounced stability-plasticity trade-off. To address this, we propose post-adaptation memory tuning (PAMT), a memory-only stabilization stage that augments an adapted model with a modular parametric memory head (PMH). PAMT freezes the backbone and attaches a product-key memory with fixed addressing. During prefix-trie constrained decoding, decoder hidden states sparsely query PMH to produce residual corrections in hidden space; these corrections are mapped to score adjustments via the frozen output embedding matrix, computed only over trie-valid tokens. This guides docid generation while keeping routing and backbone parameters fixed. To limit cross-slice interference, PAMT updates only a fixed budget of memory values selected using decoding-time access statistics, prioritizing entries frequently activated by the current slice and rarely used in prior sessions. Experiments on MS MARCO and Natural Questions under sequential, disjoint corpus increments show that PAMT substantially improves retention on earlier slices with minimal impact on retrieval performance for newly added documents, while modifying only a sparse subset of memory values per session. 

---
# A Taxonomy and Resolution Strategy for Client-Level Disagreements in Federated Learning 

**Authors**: Daan Rosendal, Ana Oprescu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23386)  

**Abstract**: Federated Learning (FL) typically assumes unconditional collaboration, a premise that overlooks the complexities of real-world, multi-stakeholder environments in which clients may need to exclude one another for strategic, regulatory, or competitive reasons. This paper addresses this gap, which we term 'client-level disagreements,' by first introducing a taxonomy of such scenarios. We then propose a robust, multi-track resolution strategy that guarantees strict client exclusion by creating and managing isolated model update paths ('tracks'), thereby preventing the cross-contamination and unfairness issues present in naive strategies. Through an empirical evaluation of our custom simulation system across 34 scenarios using the MNIST and N-CMAPSS datasets, we validate that our approach correctly handles permanent, temporal, and overlapping disagreement patterns. Our scalability analysis reveals the server-side resolution algorithm's overhead is negligible (<1 ms per round) even under heavy load. The primary scalability constraint is the client-side training load from participating in multiple tracks, a cost that we show can be effectively mitigated by a submodel reuse strategy. This work presents a scalable and architecturally sound method for managing client-level disagreements, and enhances the practical applicability of FL in settings where policy compliance and strategic control are non-negotiable. 

---
# Nonlinear Non-Gaussian Density Steering with Input and Noise Channel Mismatch: Sinkhorn with Memory for Solving the Control-affine Schrödinger Bridge Problem 

**Authors**: Georgiy A. Bondar, Asmaa Eldesoukey, Yongxin Chen, Abhishek Halder  

**Link**: [PDF](https://arxiv.org/pdf/2604.23370)  

**Abstract**: Solutions to the Schrödinger bridge problem and its generalizations yield feedback control policies for optimal density steering over a controlled diffusion. To numerically compute the same, the dynamic Sinkhorn recursion has become a standard approach. The mathematical engine behind this approach is the Hopf-Cole transform that recasts the conditions for optimality into a system of boundary-coupled linear PDEs. Recent works pointed out that for the control-affine Schrödinger bridge problem, this exact linearity via Hopf-Cole transform, and thus the standard Sinkhorn recursion, apply only if the control and noise channels are proportional. When the channels do not match, the Hopf-Cole-transformed PDEs remain nonlinear, and no algorithm is available to solve the same. We advance the state-of-the-art by designing a Sinkhorn recursion with memory that leverages the structure of these nonlinear PDEs, and demonstrate how it solves the control-affine Schrödinger bridge problem with input and noise channel mismatch. We prove the local stability of the proposed algorithm. 

---
# An Empirical Evaluation of Locally Deployed LLMs for Bug Detection in Python Code 

**Authors**: Jelena Ilić Vulićević  

**Link**: [PDF](https://arxiv.org/pdf/2604.23361)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance on a wide range of software engineering tasks, including code generation and analysis. However, most prior work relies on cloud-based models or specialized hardware, limiting practical applicability in privacy-sensitive or resource-constrained environments. In this paper, we present a systematic empirical evaluation of two locally deployed LLMs, LLaMA 3.2 and Mistral, for real-world Python bug detection using the BugsInPy benchmark. We evaluate 349 bugs across 17 projects using a zero-shot prompting approach at the function level and an automated keyword-based evaluation framework. Our results show that locally executed models achieve accuracy between 43% and 45%, while producing a large proportion of partially correct responses that identify problematic code regions without pinpointing the exact fix. Performance varies significantly across projects, highlighting the importance of codebase characteristics. The results demonstrate that local models can identify a meaningful share of bugs, though precise localization remains difficult for locally executed LLMs, particularly when handling complex and context dependent bugs in realistic development scenarios. 

---
# Explainable AI in Speaker Recognition -- Making Latent Representations Understandable 

**Authors**: Yanze Xu, Wenwu Wang, Mark D. Plumbley  

**Link**: [PDF](https://arxiv.org/pdf/2604.23354)  

**Abstract**: Neural networks can be trained to learn task-relevant representations from data. Understanding how these networks make decisions falls within the Explainable AI (XAI) domain. This paper proposes to study an XAI topic: uncovering unknown organisational patterns in network representations, particularly those representations learned by the speaker recognition network that recognises the speaker identity of utterances. Past studies employed algorithms (e.g. t-distributed Stochastic Neighbour Embedding and K-means) to analyse and visualise how network representations form independent clusters, indicating the presence of flat clustering phenomena within the space defined by these representations. In contrast, this work applies two algorithms -- Single-Linkage Clustering (SLINK) and Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) -- to analyse how representations form clusters with hierarchical relationships rather than being independent, thereby demonstrating the existence of hierarchical clustering phenomena within the network representation space. To semantically understand the above hierarchical clustering phenomena, a new algorithm, termed Hierarchical Cluster-Class Matching (HCCM), is designed to perform one-to-one matching between predefined semantic classes and hierarchical representation clusters (i.e. those produced by SLINK or HDBSCAN). Some hierarchical clusters are successfully matched to individual semantic classes (e.g. male, UK), while others to conjunctions of semantic classes (e.g. male and UK, female and Ireland). A new metric, Liebig's score, is proposed to quantify the performance of each matching behaviour, allowing us to diagnose the factor that most strongly limits matching performance. 

---
# When Chain-of-Thought Fails, the Solution Hides in the Hidden States 

**Authors**: Houman Mehrafarin, Amit Parekh, Ioannis Konstas  

**Link**: [PDF](https://arxiv.org/pdf/2604.23351)  

**Abstract**: Whether intermediate reasoning is computationally useful or merely explanatory depends on whether chain-of-thought (CoT) tokens contain task-relevant information. We present a mechanistic causal analysis of CoT on GSM8K using activation patching: transferring token-level hidden states from a CoT generation to a direct-answer run for the same question, then measuring the effect on final-answer accuracy. Across models, generating after patching yields substantially higher accuracy than both direct-answer prompting and the original CoT trace, revealing that individual CoT tokens can encode sufficient information to recover the correct answer, even when the original trace is incorrect. This task-relevant information is more prevalent in correct than incorrect CoT runs and is unevenly distributed across tokens, concentrating in mid-to-late layers and appearing earlier in the reasoning trace. Moreover, patching language tokens such as verbs and entities carry task-solving information that steers generation toward correct reasoning, whereas mathematical tokens encode answer-proximal content that rarely succeeds. Patched outputs are often shorter and yet exceed the accuracy of a full CoT trace, suggesting complete reasoning chains are not always necessary. Together, these findings demonstrate that CoT encodes recoverable, token-level problem-solving information, offering new insight into how reasoning is represented and where it breaks down. 

---
# EmoTrans: A Benchmark for Understanding, Reasoning, and Predicting Emotion Transitions in Multimodal LLMs 

**Authors**: He Hu, Tengjin Weng, Zebang Cheng, Yu Wang, Jiachen Luo, Björn Schuller, Zheng Lian, Laizhong Cui  

**Link**: [PDF](https://arxiv.org/pdf/2604.23348)  

**Abstract**: Recent multimodal large language models (MLLMs) have shown strong capabilities in perception, reasoning, and generation, and are increasingly used in applications such as social robots and human-computer interaction, where understanding human emotions is essential. However, existing benchmarks mainly formulate emotion understanding as a static recognition problem, leaving it largely unclear whether current MLLMs can understand emotion as a dynamic process that evolves, shifts between states, and unfolds across diverse social contexts. To bridge this gap, we present EmoTrans, a benchmark for evaluating emotion dynamics understanding in multimodal videos. EmoTrans contains 1,000 carefully collected and manually annotated video clips, covering 12 real-world scenarios, and further provides over 3,000 task-specific question-answer (QA) pairs for fine-grained evaluation. The benchmark introduces four tasks, namely Emotion Change Detection (ECD), Emotion State Identification (ESI), Emotion Transition Reasoning (ETR), and Next Emotion Prediction (NEP), forming a progressive evaluation framework from coarse-grained detection to deeper reasoning and prediction. We conduct a comprehensive evaluation of 18 state-of-the-art MLLMs on EmoTrans and obtain two main findings. First, although current MLLMs show relatively stronger performance on coarse-grained emotion change detection, they still struggle with fine-grained emotion dynamics modeling. Second, socially complex settings, especially multi-person scenarios, remain substantially challenging, while reasoning-oriented variants do not consistently yield clear improvements. To facilitate future research, we publicly release the benchmark, evaluation protocol, and code at this https URL. 

---
# Evaluating Jailbreaking Vulnerabilities in LLMs Deployed as Assistants for Smart Grid Operations: A Benchmark Against NERC Standards 

**Authors**: Taha Hammadia, Lucas Rea, Ahmad Mohammad Saber, Amr Youssef, Deepa Kundur  

**Link**: [PDF](https://arxiv.org/pdf/2604.23341)  

**Abstract**: The deployment of Large Language Models (LLMs) as assistants in electric grid operations promises to streamline compliance and decision-making but exposes new vulnerabilities to prompt-based adversarial attacks. This paper evaluates the risk of jailbreaking LLMs, i.e., circumventing safety alignments to produce outputs violating regulatory standards, assuming threats from authorized users, such as operators, who craft malicious prompts to elicit non-compliant guidance. Three state-of-the-art LLMs (OpenAI's GPT-4o mini, Google's Gemini 2.0 Flash-Lite, and Anthropic's Claude 3.5 Haiku) were tested against Baseline, BitBypass, and DeepInception jailbreaking methods across scenarios derived from nine NERC Reliability Standards (EOP, TOP, and CIP). In the initial broad experiment, the overall Attack Success Rate (ASR) was 33.1%, with DeepInception proving most effective at 63.17% ASR. Claude 3.5 Haiku exhibited complete resistance (0% ASR), while Gemini 2.0 Flash-Lite was most vulnerable (55.04% ASR) and GPT-4o mini moderately susceptible (44.34% ASR). A follow-up experiment refining malicious wording in Baseline and BitBypass attacks yielded a 30.6% ASR, confirming that subtle prompt adjustments can enhance simpler methods' efficacy. 

---
# EAD-Net: Emotion-Aware Talking Head Generation with Spatial Refinement and Temporal Coherence 

**Authors**: Yahui Li, Yinfeng Yu, Liejun Wang, Shengjie Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23325)  

**Abstract**: Emotionally talking head video generation aims to generate expressive portrait videos with accurate lip synchronization and emotional facial expressions. Current methods rely on simple emotional labels, leading to insufficient semantic information. While introducing high-level semantics enhances expressiveness, it easily causes lip-sync degradation. Furthermore, mainstream generation methods struggle to balance computational efficiency and global motion awareness in long videos and suffer from poor temporal coherence. Therefore, we propose an \textbf{E}motion-\textbf{A}ware \textbf{D}iffusion model-based \textbf{Net}work, called \textbf{EAD-Net}. We introduce SyncNet supervision and Temporal Representation Alignment (TREPA) to mitigate lip-sync degradation caused by multi-modal fusion. To model complex spatio-temporal dependencies in long video sequences, we propose a Spatio-Temporal Directional Attention (STDA) mechanism that captures global motion patterns through strip attention. Additionally, we design a Temporal Frame graph Reasoning Module (TFRM) to explicitly model temporal coherence between video frames through graph structure learning. To enhance emotional semantic control, a large language model is employed to extract textual descriptions from real videos, serving as high-level semantic guidance. Experiments on the HDTF and MEAD datasets demonstrate that our method outperforms existing methods in terms of lip-sync accuracy, temporal consistency, and emotional accuracy. 

---
# Layer Embedding Deep Fusion Graph Neural Network 

**Authors**: Taihua Xu, Genhao Tian, Jicong Fan, Xibei Yang, Qinghua Zhang, Yun Cui  

**Link**: [PDF](https://arxiv.org/pdf/2604.23324)  

**Abstract**: Graph Neural Networks (GNNs) have demonstrated impressive performance in learning representations from graph-structured data. However, their message-passing mechanism inherently relies on the assumption of label consistency among connected nodes, limiting their applicability to low-homophily settings. Moreover, since message passing operates as a hierarchical diffusion process, GNNs face challenges in capturing long-range dependencies. As network depth increases, the structural noise along heterophilic edges tends to be amplified, resulting in over-smoothing. This issue becomes especially prominent in highly heterophilic graphs, where the propagation of inconsistent semantics across the topology continually exacerbates misaggregation. To address this issue, we propose a novel framework named Layer Embedding Deep Fusion Graph Neural Network (LEDF-GNN). Specifically, we design a Layer Embedding Deep Fusion (LEDF) operator that nonlinearly fuses multi-layer embeddings to capture inter-layer dependencies and effectively alleviate deep propagation degradation. Meanwhile, to mitigate structural heterophily, LEDF-GNN employs a Dual-Topology Parallel Strategy (DTPS) that simultaneously leverages the original and reconstructed topologies, allowing for adaptive structure-semantics co-optimization under diverse homophily conditions. Extensive semi-supervised classification experiments on the citation and image benchmarks demonstrate that, under both homophilic and heterophilic settings, LEDF-GNN consistently outperforms state-of-the-art baselines, validating its effectiveness and generalization capability across diverse graph types. 

---
# GIFT: Global stabilisation via Intrinsic Fine Tuning 

**Authors**: Rory Young, Nicolas Pugeault  

**Link**: [PDF](https://arxiv.org/pdf/2604.23312)  

**Abstract**: Deep reinforcement learning policies achieve strong performance in complex continuous control environments with nonlinear contact forces. However, these policies often produce chaotic state dynamics, with trivially small changes to the initial conditions significantly impacting the long-term behaviour of the control system. This high sensitivity to initial conditions limits the application of Deep RL to real-world control systems where performance and stability guarantees are often required. To address this issue, we propose Global stabilisation via Intrinsic Fine Tuning (GIFT), a general-purpose training framework which directly optimises the global stability of existing high-performing deep RL policies using a custom reward function. We demonstrate that GIFT increase the stability of the control interaction while maintaining comparable task performance, thereby improving the suitability of deep RL policies for real-world control systems. 

---
# CombiMOTS: Combinatorial Multi-Objective Tree Search for Dual-Target Molecule Generation 

**Authors**: Thibaud Southiratn, Bonil Koo, Yijingxiu Lu, Sun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.23307)  

**Abstract**: Dual-target molecule generation, which focuses on discovering compounds capable of interacting with two target proteins, has garnered significant attention due to its potential for improving therapeutic efficiency, safety and resistance mitigation. Existing approaches face two critical challenges. First, by simplifying the complex dual-target optimization problem to scalarized combinations of individual objectives, they fail to capture important trade-offs between target engagement and molecular properties. Second, they typically do not integrate synthetic planning into the generative process. This highlights a need for more appropriate objective function design and synthesis-aware methodologies tailored to the dual-target molecule generation task. In this work, we propose CombiMOTS, a Pareto Monte Carlo Tree Search (PMCTS) framework that generates dual-target molecules. CombiMOTS is designed to explore a synthesizable fragment space while employing vectorized optimization constraints to encapsulate target affinity and physicochemical properties. Extensive experiments on real-world databases demonstrate that CombiMOTS produces novel dual-target molecules with high docking scores, enhanced diversity, and balanced pharmacological characteristics, showcasing its potential as a powerful tool for dual-target drug discovery. The code and data is accessible through this https URL. 

---
# $\mathcal{S}^2$IT: Stepwise Syntax Integration Tuning for Large Language Models in Aspect Sentiment Quad Prediction 

**Authors**: Bingfeng Chen, Chenjie Qiu, Yifeng Xie, Boyan Xu, Ruichu Cai, Zhifeng Hao  

**Link**: [PDF](https://arxiv.org/pdf/2604.23296)  

**Abstract**: Aspect Sentiment Quad Prediction (ASQP) has seen significant advancements, largely driven by the powerful semantic understanding and generative capabilities of large language models (LLMs). However, while syntactic structure information has been proven effective in previous extractive paradigms, it remains underutilized in the generative paradigm of LLMs due to their limited reasoning capabilities. In this paper, we propose S^2IT, a novel Stepwise Syntax Integration Tuning framework that progressively integrates syntactic structure knowledge into LLMs through a multi-step tuning process. The training process is divided into three steps. S^2IT decomposes the quadruple generation task into two stages: 1) Global Syntax-guided Extraction and 2) Local Syntax-guided Classification, integrating both global and local syntactic structure information. Finally, Fine-grained Structural Tuning enhances the model's understanding of syntactic structures through the prediction of element links and node classification. Experiments demonstrate that S^2IT significantly improves state-of-the-art performance across multiple datasets. Our implementation will be open-sourced at this https URL. 

---
# Human-1 by Josh Talks: A Full-Duplex Conversational Modeling Framework in Hindi using Real-World Conversations 

**Authors**: Bhaskar Singh, Shobhit Banga, Pranav Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2604.23295)  

**Abstract**: Full-duplex spoken dialogue systems can model natural conversational behaviours such as interruptions, overlaps, and backchannels, yet such systems remain largely unexplored for Indian languages. We present the first open, reproducible full-duplex spoken dialogue system for Hindi by adapting Moshi, a state-of-the-art duplex speech architecture, using a custom Hindi tokeniser and training on 26,000 hours of real spontaneous conversations collected from 14,695 speakers with separate speaker channels, enabling direct learning of turn-taking and overlap patterns from natural interactions. To support Hindi text generation, we replace the original English tokeniser and reinitialise text-vocabulary-dependent parameters while retaining the pre-trained audio components. We propose a two-stage training recipe -- large-scale pre-training followed by fine-tuning on 1,000 hours of conversational data. Evaluation through the prompted dialogue continuation paradigm with both automatic metrics and human judgments demonstrates that the resulting model generates natural and meaningful full-duplex conversational behaviour in Hindi. This work serves as a first step toward real-time duplex spoken dialogue systems for Hindi and other Indian languages. 

---
# An Analysis of Active Learning Algorithms using Real-World Crowd-sourced Text Annotations 

**Authors**: Varun Totakura, Ankita Singh, Yushun Dong, Shayok Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2604.23290)  

**Abstract**: Active learning algorithms automatically identify the most informative samples from large amounts of unlabeled data and tremendously reduce human annotation effort in inducing a machine learning model. In a conventional active learning setup, the labeling oracles are assumed to be infallible, that is, they always provide correct answers (in terms of class labels) to the queried unlabeled instances, which cannot be guaranteed in real-world applications. To this end, a body of research has focused on the development of active learning algorithms in the presence of imperfect / noisy oracles. Existing research on active learning with noisy oracles typically simulate the oracles using machine learning models; however, real-world situations are much more challenging, and using ML models to simulate the annotation patterns may not appropriately capture the nuances of real-world annotation challenges. In this research, we first collect annotations of text samples (from 3 benchmark text classification datasets) from crowd-sourced workers through a crowd-sourcing platform. We then conduct extensive empirical studies of 8 commonly used active learning techniques (in conjunction with deep neural networks) using the obtained annotations. Our analyses sheds light on the performance of these techniques under real-world challenges, where annotators can provide incorrect labels, and can also refuse to provide labels. We hope this research will provide valuable insights that will be useful for the deployment of deep active learning systems in real-world applications. The obtained annotations can be accessed at this https URL. 

---
# MetaErr: Towards Predicting Error Patterns in Deep Neural Networks 

**Authors**: Varun Totakura, Shayok Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2604.23289)  

**Abstract**: Due to the unprecedented success of deep learning, it has become an integral component in several multimedia computing applications in todays world. Unfortunately, deep learning systems are not perfect and can fail, sometimes abruptly, without prior warning or explanation. While reducing the error rate of deep neural networks has been the primary focus of the multimedia community, the problem of predicting when a deep learning system is going to fail has received significantly less research attention. In this paper, we propose a simple yet effective framework, MetaErr, to address this under-explored problem in deep learning research. We train a meta-model whose goal is to predict whether a base deep neural network will succeed or fail in predicting a particular data sample, by observing the base models performance on a given learning task. The meta-model is completely agnostic of the architecture and training parameters of the base model. Such an error prediction system can be immensely useful in a variety of smart multimedia applications. Our empirical studies corroborate the promise and potential of our framework against competing baselines. We further demonstrate the usefulness of our framework to improve the performance of pseudo-labeling-based semi-supervised learning, and show that MetaErr outperforms several strong baselines on three benchmark computer vision datasets. 

---
# Au-M-ol: A Unified Model for Medical Audio and Language Understanding 

**Authors**: Meizhu Liu, Nistha Mitra, Paul Li, Amine Abdaoui, Adam Ledyard, Tao Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.23284)  

**Abstract**: In this work, we present Au-M-ol, a novel multimodal architecture that extends Large Language Models (LLMs) with audio processing. It is designed to improve performance on clinically relevant tasks such as Automatic Speech Recognition (ASR). Au-M-ol has three main components: (1) an audio encoder that extracts rich acoustic features from medical speech, (2) an adaptation layer that maps audio features into the LLM input space, and (3) a pretrained LLM that performs transcription and clinical language understanding. This design allows the model to interpret spoken medical content directly, improving both accuracy and robustness. In experiments, Au-M-ol reduces Word Error Rate (WER) by 56\% compared to state-of-the-art baselines on medical transcription tasks. The model also performs well in challenging conditions, including noisy environments, domain-specific terminology, and speaker variability. These results suggest that Au-M-ol is a strong candidate for real-world clinical applications, where reliable and context-aware audio understanding is essential. 

---
# From Similarity to Structure: Training-free LLM Context Compression with Hybrid Graph Priors 

**Authors**: Yitian Zhou, Chaoning Zhang, Jiaquan Zhang, Zhenzhen Huang, Jinyu Guo, Sung-Ho Bae, Lik-Hang Lee, Caiyan Qin, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23277)  

**Abstract**: Long-context large language models remain computationally expensive to run and often fail to reliably process very long inputs, which makes context compression an important component of many systems. Existing compression approaches typically rely on trained compressors, dense retrieval-style selection, or heuristic trimming, and they often struggle to jointly preserve task relevance, topic coverage, and cross-sentence coherence under a strict token budget. To address this, we propose a training-free and model-agnostic compression framework that selects a compact set of sentences guided by structural graph priors. Our method constructs a sparse hybrid sentence graph that combines mutual k-NN semantic edges with short-range sequential edges, extracts a topic skeleton via clustering, and ranks sentences using an interpretable score that integrates task relevance, cluster representativeness, bridge centrality, and a cycle coverage cue. A budgeted greedy selection with redundancy suppression then produces a readable compressed context in original order. Experimental results on four datasets show that our approach is competitive with strong extractive and abstractive baselines, demonstrating larger gains on long-document benchmarks. 

---
# Lightweight and Production-Ready PDF Visual Element Parsing 

**Authors**: Meizhu Liu, Yassi Abbasi, Matthew Rowe, Michael Avendi, Paul Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.23276)  

**Abstract**: PDF documents contain critical visual elements such as figures, tables, and forms whose accurate extraction is essential for document understanding and multimodal retrieval-augmented generation (RAG). Existing PDF parsers often miss complex visuals, extract non-informative artifacts (e.g., watermarks, logos), produce fragmented elements, and fail to reliably associate captions with their corresponding elements, which degrades downstream retrieval and question answering. We present a lightweight and production level PDF parsing framework that can accurately detect visual elements and associates captions using a combination of spatial heuristics, layout analysis, and semantic similarity. On popular benchmark datasets and internal product data, the proposed solution achieves $\geq96\%$ visual element detection accuracy and $93\%$ caption association accuracy. When used as a preprocessing step for multimodal RAG, it significantly outperforms state-of-the-art parsers and large vision-language models on both internal data and the MMDocRAG benchmark, while reducing latency by over $2\times$. We have deployed the proposed system in challenging production environment. 

---
# Small Language Model Helps Resolve Semantic Ambiguity of LLM Prompt 

**Authors**: Zhenzhen Huang, Chaoning Zhang, Fachrina Dewi Puspitasari, Jiaquan Zhang, Yitian Zhou, Shuxu Chen, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23263)  

**Abstract**: Large language models (LLMs) are increasingly utilized in various complex reasoning tasks due to their excellent instruction following capability. However, the model's performance is highly dependent on the open-ended characteristics of the users' input prompt. Natural prompts often do not follow proper syntactic rules, which creates ambiguous queries that yield multiple interpretations. Such ambiguous prompts confuse the model in choosing the correct reasoning paths to answer questions. Prior works address this challenge by applying query editing during the LLM inference process without explicitly solving the root cause of the ambiguity. To address this limitation, we propose a pre-inference prompt optimization mechanism via explicit prompt disambiguation. Particularly, we identify semantic risks in the prompt, check their multi-perspective consistency, and resolve any semantic conflicts that arise. Finally, we organize the resolved ambiguities in a logically structured manner as a clean input to the LLM. By explicitly resolving semantic ambiguity, our method can produce a more focused attention distribution to the semantically essential tokens. We also leverage small language models (SLMs) as the main executor of prompt disambiguation to benefit from their efficient computation. Through comprehensive experiments on multiple benchmarks, we demonstrate that our method improves reasoning performance by 2.5 points at a cost of only \$0.02. Our study promotes explicit prompt disambiguation as an effective prompt optimization method without disturbing the internal mechanism of LLM inference. 

---
# Knowledge Lever Risk Management for Software Engineering: A Stochastic Framework for Mitigating Knowledge Loss 

**Authors**: Mark Chua, Samuel Ajila  

**Link**: [PDF](https://arxiv.org/pdf/2604.23257)  

**Abstract**: Software engineering (SE) organizations operate in a knowledge-intensive domain where critical assets -- architectural expertise, design rationale, and system intuition -- are overwhelmingly tacit and volatile. The departure of key contributors or the decay of undocumented decisions can severely impair project velocity and software quality. While conventional SE risk management optimized for schedule and budget is common, the intangible knowledge risks that determine project success remain under-represented.
The goal of this research work is to propose and evaluate the Knowledge Lever Risk Management (KLRM) Framework, designed specifically for the software development lifecycle. The primary objectives are to: (1) recast intangible knowledge assets as active mechanisms for risk mitigation (Knowledge Levers); (2) integrate these levers into a structured four-phase architecture (Audit, Alignment, Activation, Assurance); and (3) provide a formal stochastic model to quantify the impact of lever activation on project knowledge capital. We detail the application of these levers through software-specific practices such as pair programming, architectural decision records (ADRs), and LLM-assisted development. Stochastic Monte Carlo simulations demonstrate that full lever activation increases expected knowledge capital by 63.8\% and virtually eliminates knowledge crisis probability. Our research shows that knowledge lever activation improves alignment across the project management iron triangle (scope, time, cost) by reducing rework and rediscovery costs. 

---
# Why Architecture Choice Matters in Symbolic Regression 

**Authors**: Chakshu Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.23256)  

**Abstract**: Symbolic regression discovers mathematical formulas from data. Some methods fix a tree of operators, assign learnable weights, and train by gradient descent. The tree's structure, which determines what operators and variables appear at each position, is chosen once and applied to every target. This paper tests whether that choice affects which targets are actually recovered. Three structures are compared, all sharing the same operator and target language but differing in how variables enter the tree; one is strictly more expressive. Across over 12,700 training runs, one structure recovers a target at 100% while another scores 0%, and the ranking reverses on a different target. Expressiveness guarantees that a solution exists in the search space, but not that gradient descent finds it: the most expressive structure fails on targets that a restricted alternative solves reliably. Switching the operator changes which targets succeed; reversing its gradient profile collapses recovery entirely. Balanced (non-chain) tree shapes are never recovered. These findings show that the optimization landscape, not expressiveness alone, determines what gradient-based symbolic regression recovers. 

---
# Scalable LLM-based Coding of Dialogue in Healthcare Simulation: Balancing Coding Performance, Processing Time, and Environmental Impact 

**Authors**: Kiyoshige Garces, Gloria Milena Fernandez-Nieto, Linxuan Zhao, Sachini Samaraweera, Dragan Gasevic, Roberto Martinez-Maldonado, Vanessa Echeverria  

**Link**: [PDF](https://arxiv.org/pdf/2604.23255)  

**Abstract**: Research shows that dialogue, the interactive process through which participants articulate their thinking, plays a central role in constructing shared understanding, coordinating action, and shaping learning outcomes in teams. Analysing dialogue content has been central to advancing team learning theory and informing the design of computer-supported collaborative learning environments, yet this progress has depended on labour-intensive qualitative coding. LLMs offer new possibilities for automating and enhancing the dialogue layer within emerging multimodal learning analytics approaches, with recent studies showing that they can approximate human coding through few-shot prompting. However, prior work has focused on replicating human coding accuracy for research purposes, rather than addressing a more educationally consequential question: how can we design prompts that allow an LLM to label team dialogue accurately and fast enough to be useful in real settings, such as in-person healthcare simulations, where results must be returned quickly and computational cost and sustainability also matter? This paper investigates how prompt design and batching strategies can be optimised to balance coding accuracy, processing time, and environmental impact in team-based healthcare simulation debriefing. Using a dataset of 11,647 utterances coded across 6 dialogue constructs, we compared 4 prompt designs across varying batch sizes, evaluating coding performance, processing time, and energy consumption, as well as the trade-offs between these metrics. Results indicate that increasing batch size improves speed and reduces energy use, but negatively impacts coding performance. Beyond demonstrating the feasibility of LLM-based qualitative analysis, this study offers practical guidance for scaling dialogue analytics in contexts where timeliness, privacy, and sustainability are critical. 

---
# AI-Assisted Code Review as a Scaffold for Code Quality and Self-Regulated Learning: An Experience Report 

**Authors**: Eduardo Oliveira, Michael Fu, Patanamon Thongtanunam, Sonsoles López-Pernas, Mohammed Saqr  

**Link**: [PDF](https://arxiv.org/pdf/2604.23251)  

**Abstract**: Code review is central to software engineering education but hard to scale in capstone projects due to tight deadlines, uneven peer feedback, and limited prior experience. We investigate an LLM-as-reviewer integrated directly into GitHub pull requests (human-in-the-loop) across two cohorts (more than 100 students, 2023--2024). Using a mixed-methods design -- GitHub data, reflective reports, and a targeted survey -- we examine engagement and responsiveness as behavioral indicators of self-regulated learning processes. Quantitatively, the 2024 cohort produced more iterative activity (1176 vs. 581 PRs), while technical issues observed in 2023 (227 failed AI attempts) dropped to zero after tool and instructional refinements. Despite different adoption levels (93\% vs. 50\% of teams using the tool), responsiveness was stable: 32\% (2023) and 33\% (2024) of successfully AI-reviewed PRs were followed by subsequent commits on the same PR. Qualitatively, students used the LLM's structured comments to focus reviews and discuss code quality, while guidance reduced over-reliance. We contribute: (i) an in-workflow design for an AI reviewer that scaffolds learning while mitigating cognitive offloading; (ii) a repeated cross sectional comparison across two cohorts in authentic settings; (iii) a mixed-methods analysis combining objective GitHub metrics with student self-reports; and (iv) evidence-based pedagogical recommendations for responsible, student-led AI-assisted review. 

---
# Training Machine Learning Models on Encrypted Data: A Privacy-Preserving Framework using Homomorphic Encryption 

**Authors**: Alexandre Marques, Beatriz Sá, Rui Botelho, Pedro Pinto  

**Link**: [PDF](https://arxiv.org/pdf/2604.23245)  

**Abstract**: The use of Machine Learning (ML) for data-driven decision-making often relies on access to sensitive datasets, which introduces privacy challenges. Traditional encryption methods protect data at rest or in transit but fail to secure it during processing, exposing it to unauthorized access. Homomorphic encryption emerges as a transformative solution, enabling computations on encrypted data without decryption, thus preserving confidentiality throughout the ML pipeline. This paper addresses the challenge of training ML models on encrypted data while maintaining accuracy and efficiency by proposing a proof-of-concept for a privacy-preserving framework that leverages Cheon-Kim-Kim-Song (CKKS) for approximate real-number arithmetic. Also, it demonstrates the feasibility of training K-Nearest Neighbors (KNN) and linear regression models on encrypted data, and evaluates encrypted inference for a basic Multilayer Perceptron (MLP) architecture. Experimental results show that models trained under Homomorphic encryption achieve performance metrics comparable to plaintext-trained models, validating the approach. However, challenges such as computational overhead, noise management, and limited support for non-polynomial operations persist. This work lays the groundwork for broader adoption of privacy-preserving ML in real-world applications, balancing security with computational feasibility. 

---
# Protecting the Trace: A Principled Black-Box Approach Against Distillation Attacks 

**Authors**: Max Hartman, Vidhata Jayaraman, Moulik Choraria, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2604.23238)  

**Abstract**: Frontier models push the boundaries of what is learnable at extreme computational costs, yet distillation via sampling reasoning traces exposes closed-source frontier models to adversarial third parties who can bypass their guardrails and misappropriate their capabilities, raising safety, security, and intellectual privacy concerns. To address this, there is growing interest in building antidistillation methods, which aim to poison reasoning traces to hinder downstream student model learning while maintaining teacher performance. However, current techniques lack theoretical grounding, requiring either heavy fine-tuning or access to student model proxies for gradient based attacks, and often lead to a significant teacher performance degradation. In this work, we present a theoretical formulation of antidistillation as a Stackelberg game, grounding a problem that has so far largely been approached heuristically. Guided by the desired design properties our formulation reveals, we propose \texttt{TraceGuard}, an efficient, post-generation black-box method to poison sentences with high importance for teacher reasoning. Our work offers a scalable solution to share model insights safely, ensuring that the advancement of reasoning capabilities does not come at the cost of intellectual privacy or AI safety alignment. 

---
# Toward Polymorphic Backdoor against Semantic Communication via Intensity-Based Poisoning 

**Authors**: Xiao Yang, Yuni Lai, Gaolei Li, Jun Wu, Kai Zhou, Jianhua Li, Mingzhe Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23231)  

**Abstract**: Semantic Communication (SC) backdoor attacks aim to utilize triggers to manipulate the system into producing predetermined outputs via backdoored shared knowledge. Current SC backdoors adopt monomorphic paradigms with single attack target, which suffers from limited attack diversity, efficiency, and flexibility in heterogeneous downstream scenarios. To overcome the limitations, we propose SemBugger, a polymorphic SC backdoor. By dynamically adjusting the trigger intensity, SemBugger finely-grained controls over the SC knowledge to generate diverse malicious results from the system. Specifically, SemBugger is realized through a multi-effect poisoning-training framework. It introduces graded-intensity triggers to poison training data and optimizes SC systems with hierarchical malicious loss. The trained system's knowledge dynamically adapts to trigger intensity in inputs to yield target outputs, all while preserving transmission fidelity for benign samples. Moreover, to augment SC security, we propose a provable robustness defense that resists SemBugger's homogeneous attacks through a controlled noise mechanism. It operates via strategically adding noise in SC inputs, and we formally provide a theoretical lower bound on the defense efficacy. Experiments across diverse SC models and benchmark datasets indicate that SemBugger attains high attack efficacy while maintaining the regular functionality of SC systems. Meanwhile, the designed defense effectively neutralizes SemBugger attacks. 

---
# AnalogRetriever: Learning Cross-Modal Representations for Analog Circuit Retrieval 

**Authors**: Yihan Wang, Lei Li, Yao Lai, Jing Wang, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23195)  

**Abstract**: Analog circuit design relies heavily on reusing existing intellectual property (IP), yet searching across heterogeneous representations such as SPICE netlists, schematics, and functional descriptions remains challenging. Existing methods are largely limited to exact matching within a single modality, failing to capture cross-modal semantic relationships. To bridge this gap, we present AnalogRetriever, a unified tri-modal retrieval framework for analog circuit search. We first build a high-quality dataset on top of Masala-CHAI through a two-stage repair pipeline that raises the netlist compile rate from 22\% to 100\%. Built on this foundation, AnalogRetriever encodes schematics and descriptions with a vision-language model and netlists with a port-aware relational graph convolutional network, mapping all three modalities into a shared embedding space via curriculum contrastive learning. Experiments show that AnalogRetriever achieves an average Recall@1 of 75.2\% across all six cross-modal retrieval directions, significantly outperforming existing baselines. When integrated into the AnalogCoder agentic framework as a retrieval-augmented generation module, it consistently improves functional pass rates and enables previously unsolved tasks to be completed. Our code and dataset will be released. 

---
# RAT: RunAnyThing via Fully Automated Environment Configuration 

**Authors**: Renhong Huang, Dongdong Hua, Yifei Sun, Sitao Ding, Hanyang Yuan, Daixin Wang, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23190)  

**Abstract**: Automating repository-level software engineering tasks is a foundational challenge for autonomous code agents, largely due to the difficulty of configuring executable environments. However, manual configuration remains a labor-intensive bottleneck, necessitating a transition toward fully automated environment configuration. Existing approaches often rely on pre-defined artifacts or are restricted to specific programming languages, limiting their applicability to real-world repositories. In this paper, we first propose RAT (RunAnyThing), a language-agnostic framework for automated environment configuration on arbitrary repositories. RAT features a multi-stage pipeline that integrates semantic initialization, a planning mechanism, specialized toolset, and a robust sandbox for configuration. Furthermore, to enable rigorous evaluation, we propose RATBench, a benchmark that reflects the the distribution and heterogeneity of real-world repositories. Extensive experiments demonstrate that RAT achieves state-of-the-art performance, improving the Environment Setup Success Rate (ESSR) by an average of 29.6% over strong baselines. 

---
# DyABD: The Abdominal Muscle Segmentation in Dynamic MRI Benchmark 

**Authors**: Niamh Belton, Victoria Joppin, Aonghus Lawlor, Catherine Masson, Thierry Bege, David Bendahan, Kathleen M. Curran  

**Link**: [PDF](https://arxiv.org/pdf/2604.23187)  

**Abstract**: This work introduces DyABD, a novel and complex benchmark dataset of dynamic abdominal MRIs from patients with abdominal hernias and associated high quality abdominal muscle annotations. DyABD is the first-of-its-kind in four key ways; (1) it proposes the first abdominal muscle segmentation task, (2) the dynamic MRIs are acquired whilst the patients perform various exercises, introducing extreme anatomical variability, making it one of the most challenging segmentation datasets to date, (3) it includes both pre and post corrective MRIs and (4) DyABD promotes clinical research into the high recurrence rates of abdominal hernias. Beyond dataset introduction, this work provides a comprehensive evaluation of the generalisation capabilities of existing segmentation models across Supervised, Few Shot and Zero Shot paradigms on the unseen DyABD dataset. This work reveals that there is still room for substantial improvement in the field of medical image segmentation, with the majority of techniques achieving a Dice Coefficient of 0.82. This work therefore sheds light on the true progress of the field and redefines the benchmark for progress in medical image segmentation. 

---
# Designing escalation criteria for international AI incident response: criteria, triggers, and thresholds 

**Authors**: Francesca Gomez, Matthew Ball, Michael Harre, Lydia Preston, Josephine Schwab, Caio Machado  

**Link**: [PDF](https://arxiv.org/pdf/2604.23183)  

**Abstract**: AI incident reporting requirements are emerging in regulation and policy, yet no operational criteria exist for determining when a detected AI incident warrants escalation beyond national handling to international coordination. This paper proposes an escalation framework to address this gap, intended as a common reference point across jurisdictions that enables aligned escalation while preserving flexibility in how actors respond within their own legal and policy contexts. We review SB 53, the EU AI Act, the GPAI Code of Practice, and incident frameworks from other industries to derive eight criteria for assessing whether an incident warrants escalation, translated into a sequential flowchart with gated decision points and threshold checks. For each criterion, we map how it interplays with these regulatory frameworks, identifying where their design choices support or undermine effective detection. We test the framework against ten documented AI incidents and structured variants to identify where criteria under-detect or misclassify incidents in practice. We find three design patterns that may lead to systematic under-detection in regimes where model developers are responsible for escalation: a. where escalation requires confirmed harm, events such as model weight exfiltration risk detection only after severe, irreversible harm has propagated; b. where incidents are assessed individually, systemic harms emerging from accumulation risk being under-detected; and c. where thresholds align with legal instruments rather than quantitatively testable terms, criteria risk being impractical to apply under time pressure. We also find that escalation rules are only one component of a broader framework: the underlying definitions against which thresholds are set, and the data available to the responsible actor, create interdependencies that can themselves drive under-detection. 

---
# Cooperative Informative Sensing for Monitoring Dynamic Indoor Environments via Multi-Agent Reinforcement Learning 

**Authors**: Kanghoon Lee, Matthew M. Sato, Jinnyeong Yang, Seungro Lee, Sujin Lee, Jiachen Li, Kuk-Jin Yoon, Jinkyoo Park, Kincho H. Law, Yoonjin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2604.23179)  

**Abstract**: Monitoring human activity in indoor environments is important for applications such as facility management, safety assessment, and space utilization analysis. While mobile robot teams offer the potential to actively improve observation quality, existing multi-robot monitoring and active perception approaches typically rely on coverage or visitation based objectives that are weakly aligned with the accuracy requirements of human-centric monitoring tasks. In this work, we formulate cooperative active observation as a decentralized control problem in which multiple robots adjust their motion to directly optimize monitoring accuracy under partial observability. We propose a learning-based framework for cooperative policies from decentralized observations using multi-agent reinforcement learning (MARL), supported by an architecture that handles variable numbers of humans and temporal dependencies. Simulation results across diverse indoor environments and monitoring tasks show that the proposed approach consistently outperforms classical coverage, persistent monitoring, and learning-free multi-robot baselines, while remaining robust to changes in the number of observed humans. 

---
# Scaling Multi-Node Mixture-of-Experts Inference Using Expert Activation Patterns 

**Authors**: Abhimanyu Bambhaniya, Geonhwa Jeong, Jason Park, Jiecao Yu, Jaewon Lee, Pengchao Wang, Changkyu Kim, Chunqiang Tang, Tushar Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2604.23150)  

**Abstract**: Most recent state-of-the-art (SOTA) large language models (LLMs) use Mixture-of-Experts (MoE) architectures to scale model capacity without proportional per-token compute, enabling higher-quality outputs at manageable serving costs. However, MoE inference at scale is fundamentally bottlenecked by expert load imbalance and inefficient token routing, especially in multi-node deployments where tokens are not guaranteed to be routed to local experts, resulting in significant inter-node all-to-all communication overhead.
To systematically characterize these challenges, we profile SOTA open-source MoE models, including Llama 4 Maverick, DeepSeek V3-671B, and Qwen3-230B-A22B, on various datasets and collected over 100k real expert activation traces. Upon studying the expert activation patterns, we uncover various persistent properties across all the frontier MoE models: variable expert load imbalance, domain-specific expert activation where expert popularity shifts across task families (code, math, chat, general), and a strong correlation between prefill and decode expert activations. Motivated by these findings, we propose workload-aware micro-batch grouping and an expert placement strategy to maximize token locality to the destination expert, thereby reducing inter-node communication. Across models and datasets, these optimizations help reduce all2all communication data up to 20, resulting in lower MoE decode latency and better accelerator utilization. 

---
# UpstreamQA: A Modular Framework for Explicit Reasoning on Video Question Answering Tasks 

**Authors**: Jason Nguyen, Ameet Rao, Alexander Chang, Ishaan Kumar, Erin Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.23145)  

**Abstract**: Video Question Answering (VideoQA) demands models that jointly reason over spatial, temporal, and linguistic cues. However, the task's inherent complexity often requires multi-step reasoning that current large multimodal models (LMMs) perform implicitly, leaving their internal decision process opaque. In contrast, large reasoning models (LRMs) explicitly generate intermediate logical steps that enhance interpretability and can improve multi-hop reasoning accuracy. Yet, these models are not designed for native video understanding, as they typically rely on static frame sampling. We propose UpstreamQA, a modular framework that disentangles and evaluates core video reasoning components through explicit upstream reasoning modules. Specifically, we employ multimodal LRMs to perform object identification and scene context generation before passing enriched reasoning traces to downstream LMMs for VideoQA. We evaluate UpstreamQA on the OpenEQA and NExTQA datasets using two LRMs (o4-mini, Gemini 2.5 Pro) and two LMMs (GPT-4o, Gemini 2.5 Flash). Our results demonstrate that introducing explicit reasoning can significantly boost performance and interpretability of downstream VideoQA, but can also lead to performance degradation when baseline performance is sufficiently high. Overall, UpstreamQA offers a principled framework for combining explicit reasoning and multimodal understanding, advancing both performance and diagnostic transparency in VideoQA in several scenarios. 

---
# UNSEEN: A Cross-Stack LLM Unlearning Defense against AR-LLM Social Engineering Attacks 

**Authors**: Tianlong Yu, Yang Yang, Xiao Luo, Lihong Liu, Fudu Xing, Zui Tao, Kailong Wang, Gaoyang Liu, Ting Bi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23141)  

**Abstract**: Emerging AR-LLM-based Social Engineering attack (e.g., SEAR) is at the edge of posing great threats to real-world social life. In such AR-LLM-SE attack, the attacker can leverage AR (Augmented Reality) glass to capture the image and vocal information of the target, using the LLM to identify the target and generate the social profile, using the LLM agents to apply social engineering strategies for conversation suggestion to win the target trust and perform phishing afterwards. Current defensive approaches, such as role-based access control or data flow tracking, are not directly applicable to the convergent AR-LLM ecosystem (considering embedded AR device and opaque LLM inference), leaving an emerging and potent social engineering threat that existing privacy paradigms are ill-equipped to address. This necessitates a shift beyond solely human-centric measures like legislation and user education toward enforceable vendor policies and platform-level restrictions. Realizing this vision, however, faces significant technical challenges: securing resource-constrained AR-embedded devices, implementing fine-grained access control within opaque LLM inferences, and governing adaptive interactive agents. To address these challenges, we present UNSEEN, a coordinated cross-stack defense that combines an AR ACL (Access Control Layer) for identity-gated sensing, F-RMU-based LLM unlearning for sensitive profile suppression, and runtime agent guardrails for adaptive interaction control. We evaluate UNSEEN in an IRB-approved user study with 60 participants and a dataset of 360 annotated conversations across realistic social scenarios. 

---
# CNN-ViT Fusion with Adaptive Attention Gate for Brain Tumor MRI Classification: A Hybrid Deep Learning Model 

**Authors**: Syed Ibad Hasnain, Muhammad Faris, Hafiza Syeda Yusra Tirmizi, Rabail Khowaja, Hafsa Israr  

**Link**: [PDF](https://arxiv.org/pdf/2604.23137)  

**Abstract**: Early detection and classifying brain tumors using Magnetic Resonance Imaging (MRI) images is highly important but difficult to extract in medical images. Convolutional Neural Networks (CNNs) are good at capturing both local texture and spatial information whereas Vision Transformers (ViTs) are good at capturing long-range global dependencies. We propose a new hybrid architecture that combines a SqueezeNet-style CNN branch with a MobileViT-style global transformer branch, through an Adaptive Attention Gate mechanism, in this paper. The gate learns dynamically per-sample, per-feature weights to weight the contribution of each branch, allowing context-sensitive merging of local and global representations. The proposed model has a test accuracy of 97.60, a precision of 97.30, a recall of 97.50, an F1-score of 97.40, and a macro-average area under the curve (AUC) of 0.9946 with a trained and evaluated on the Brain Tumor MRI Dataset (Kaggle). These scores are higher than single CNN and ViT baselines, and current competitive fusion methods, showing that dynamic feature weighting is an effective way to classify medical images. 

---
# Mechanistic Steering of LLMs Reveals Layer-wise Feature Vulnerabilities in Adversarial Settings 

**Authors**: Nilanjana Das, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2604.23130)  

**Abstract**: Large language models (LLMs) can still be jailbroken into producing harmful outputs despite safety alignment. Existing attacks show this vulnerability, but not the internal mechanisms that cause it. This study asks whether jailbreak success is driven by identifiable internal features rather than prompts alone. We propose a three-stage pipeline for Gemma-2-2B using the BeaverTails dataset. First, we extract concept-aligned tokens from adversarial responses via subspace similarity. Second, we apply three feature-grouping strategies (cluster, hierarchical-linkage, and single-token-driven) to identify SAE feature subgroups for the aligned tokens across all 26 model layers. Third, we steer the model by amplifying the top features from each identified subgroup and measure the change in harmfulness score using a standardized LLM-judge scoring protocol. In all three approaches, the features in the layers [16-25] were relatively more vulnerable to steering. All three methods confirmed that mid to later layer feature subgroups are more responsible for unsafe outputs. These results provide evidence that the jailbreak vulnerability in Gemma-2-2B is localized to feature subgroups of mid to later layers, suggesting that targeted feature-level interventions may offer a more principled path to adversarial robustness than current prompt-level defenses. 

---
# MindTrellis: Co-Creating Knowledge Structures with AI through Interactive Visual Exploration 

**Authors**: Xiang Li, Cara Li, Emily Kuang, Can Liu, Jian Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.23129)  

**Abstract**: Knowledge workers face increasing challenges in synthesizing information from multiple documents into structured conceptual understanding. This process is inherently iterative: users explore content, identify relationships between concepts, and continuously reorganize their mental models. However, current approaches offer limited support. LLM-based systems let users query information but not shape how knowledge is organized; manual tools like mind maps support structure creation but lack intelligent assistance. This leaves an open opportunity: supporting collaborative construction where users and AI jointly develop an evolving knowledge representation. We present MindTrellis, an interactive visual system where users and AI collaboratively build a dynamic knowledge graph. Users can query the graph to retrieve document-grounded information, and contribute by introducing new concepts, modifying relationships, and reorganizing the hierarchy to reflect their developing understanding. In a user study where 12 participants created slide decks, MindTrellis outperformed retrieval-only baselines in knowledge organization and cognitive load, as measured by expert ratings of content coverage and structural quality. 

---
# ArgRE: Formal Argumentation for Conflict Resolution in Multi-Agent Requirements Negotiation 

**Authors**: Haowei Cheng, Milhan Kim, Chong Liu, Teeradaj Racharak, Truong Vinh Truong Duy, Phan Thi Huyen Thanh, Jialong Li, Naoyasu Ubayashi, Hironori Washizaki  

**Link**: [PDF](https://arxiv.org/pdf/2604.23124)  

**Abstract**: As software systems grow in complexity, they must satisfy an increasing number of competing quality attributes, making it essential to balance them in a principled manner -- for example, a safety requirement for sensor-fusion verification may conflict with a tight planning-cycle budget. Multi-agent large language model frameworks support this balancing process by assigning specialized agents to different objectives. However, their conflict resolution is typically heuristic. Requirements are aggregated implicitly without explicit acceptance or rejection, limiting auditability in regulated domains. We present ArgRE, a multi-agent requirements negotiation system that embeds Dung-style abstract argumentation into the negotiation stage. Each proposal, critique, and refinement is modeled as an argument, conflicts are represented as directed attack relations, and the accepted set of arguments is computed under grounded and preferred semantics. The pipeline further integrates KAOS goal modeling, multi-layer verification, and standards-oriented artifact generation. Evaluation across five case studies spanning safety-critical, financial, and information-system domains shows that ArgRE provides argument-level traceability absent from existing frameworks. Independent evaluators rated its decision justifications significantly higher than those of heuristic synthesis (4.32 vs. 3.07, p < 0.001), indicating improved auditability, while semantic intent preservation remains comparable (94.9% BERTScore F1) and compliance coverage reaches 84.7% versus 47.6%--47.8% for baselines. Structural analysis further confirms that the default pairwise protocol yields acyclic graphs in which grounded and preferred semantics coincide, whereas cross-pair arbitration introduces controlled cyclicity, leading to predictable divergence between the two semantics. 

---
# Mixture of Heterogeneous Grouped Experts for Language Modeling 

**Authors**: Zhicheng Ma, Xiang Liu, Zhaoxiang Liu, Ning Wang, Yi Shen, Kai Wang, Shuming Shi, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2604.23108)  

**Abstract**: Large Language Models (LLMs) based on Mixture-of-Experts (MoE) are pivotal in industrial applications for their ability to scale performance efficiently. However, standard MoEs enforce uniform expert sizes,creating a rigidity that fails to align computational costs with varying token-level complexity. While heterogeneous expert architectures attempt to address this by diversifying expert sizes, they often suffer from significant system-level challenges, specifically unbalanced GPU utilization and inefficient parameter utilization, which hinder practical deployment. To bridge the gap between theoretical heterogeneity and robust industrial application, we propose Mixture of Heterogeneous Grouped Experts (MoHGE) which introduces a two-level routing mechanism to enable flexible, resource-aware expert combinations. To optimize inference efficiency, we propose a Group-Wise Auxiliary Loss, which dynamically steers tokens to the most parameter-efficient expert groups based on task difficulty. To address the critical deployment challenge of GPU load balancing, we introduce an All-size Group-decoupling Allocation strategy coupled with an Intra-Group Experts Auxiliary Loss. These mechanisms collectively ensure uniform computation distribution across GPUs. Extensive evaluations demonstrate that MoHGE matches the performance of MoE architectures while reducing the total parameters by approximately 20% and maintaining balanced GPU utilization. Our work establishes a scalable paradigm for resource-efficient MoE design, offering a practical solution for optimizing inference costs in real-world scenarios. 

---
# No Test Cases, No Problem: Distillation-Driven Code Generation for Scientific Workflows 

**Authors**: Siddeshwar Raghavan, Tanwi Mallick  

**Link**: [PDF](https://arxiv.org/pdf/2604.23106)  

**Abstract**: Existing multi-agent Large Language Model (LLM) frameworks for code generation typically use execution feedback and improve iteratively using Input/Output (I/O) test cases. However, this does not work for scientific workflows, where I/O test cases do not exist, and generating them requires solving the very problem at hand. To address this, we introduce MOSAIC, a training-free multi-agent framework for scientific code generation without I/O supervision. Instead of execution feedback, MOSAIC employs a student-teacher knowledge distillation framework that grounds generation through domain-specific examples and structured problem decomposition. To further mitigate hallucinations across chained subproblems, we introduce a Consolidated Context Window (CCW) for maintaining consistent reasoning across agents. Experiments on the SciCode benchmark show that MOSAIC improves accuracy, executability, and numerical precision over existing approaches while relying on lightweight models. 

---
# ProEval: Proactive Failure Discovery and Efficient Performance Estimation for Generative AI Evaluation 

**Authors**: Yizheng Huang, Wenjun Zeng, Aditi Kumaresan, Zi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23099)  

**Abstract**: Evaluating generative AI models is increasingly resource-intensive due to slow inference, expensive raters, and a rapidly growing landscape of models and benchmarks. We propose ProEval, a proactive evaluation framework that leverages transfer learning to efficiently estimate performance and identify failure cases. ProEval employs pre-trained Gaussian Processes (GPs) as surrogates for the performance score function, mapping model inputs to metrics such as the severity of errors or safety violations. By framing performance estimation as Bayesian quadrature (BQ) and failure discovery as superlevel set sampling, we develop uncertainty-aware decision strategies that actively select or synthesize highly informative inputs for testing. Theoretically, we prove that our pre-trained GP-based BQ estimator is unbiased and bounded. Empirically, extensive experiments on reasoning, safety alignment, and classification benchmarks demonstrate that ProEval is significantly more efficient than competitive baselines. It requires 8-65x fewer samples to achieve estimates within 1% of the ground truth, while simultaneously revealing more diverse failure cases under a stricter evaluation budget. 

---
# Code Broker: A Multi-Agent System for Automated Code Quality Assessment 

**Authors**: Samer Attrah  

**Link**: [PDF](https://arxiv.org/pdf/2604.23088)  

**Abstract**: We present Code Broker, a multi agent system built with Google Agent Development Kit ADK that analyses Python code from files, local directories, or GitHub repositories and generates actionable quality assessment reports. The system employs a hierarchical five agents architecture in which a root orchestrator coordinates a sequential pipeline agent, which in turn dispatches three specialised agents in parallel a Correctness Assessor, a Style Assessor, and a Description Generator before synthesising findings through an Improvement Recommender. Reports score four dimensions correctness, security, style, and maintainability and are rendered in both Markdown and HTML. Code Broker combines LLM based reasoning with deterministic static-analysis signals from Pylint, uses asynchronous execution with retry logic to improve robustness, and explores lightweight session memory for retaining and querying prior assessment context. We position the paper as a technical report on system design and prompt or tool orchestration, and present a preliminary qualitative evaluation on representative Python codebases. The results suggest that parallel specialised agents produce readable, developer oriented feedback, while also highlighting current limitations in evaluation depth, security tooling, large repository handling, and the current use of only in memory persistence. All code and reproducibility materials are available at: this https URL. 

---
# Usable Agent Discovery for Decentralized AI Systems 

**Authors**: Patrizio Dazzi, Emanuele Carlini, Matteo Mordacchini, Saul Urso  

**Link**: [PDF](https://arxiv.org/pdf/2604.23080)  

**Abstract**: Large-scale agentic systems run on distributed infrastructures where many software agents share physical hosts and are discovered via peer-to-peer mechanisms. Discovery must handle node-level churn from failures and host departures and agent-level churn from demand-driven activation, deactivation, and state changes. Their interaction reshapes classic trade-offs between structured and unstructured overlays. We study decentralized agent discovery under this two-level churn, assuming nodes host multiple agents, overlays are structured or gossip-based, and agents switch between warm and cold states. Using Kademlia as a structured and Cyclon+Vicinity as a gossip baseline, we compare stable, node-churn-only, agent-cooling-only, and combined regimes to see when routing efficiency, resilience, and service readiness align or favor different designs. Structured overlays are more robust and efficient in stable and node-churn regimes, while gossip-based overlays remain competitive and can be faster when readiness dominates. 

---
# From Pixels to Explanations: Interpretable Diabetic Retinopathy Grading with CNN-Transformer Ensembles, Visual Explainability and Vision-Language Models 

**Authors**: Pir Bakhsh Khokhar, Carmine Gravino, Fabio Palomba, Sule Yildirim Yayilgan, Sarang Shaikh  

**Link**: [PDF](https://arxiv.org/pdf/2604.23079)  

**Abstract**: The quality of diabetic retinopathy (DR) screening relies on the ability to correctly grade severity; however, many deep-learning (DL) classifiers cannot be easily interpreted in the clinical context. This study presents a methodology that combines strong discriminative models with multimodal explanations, converting retinal pixels into clinically interpretable outputs. Using the APTOS 2019 benchmark, we evaluated six representative CNN- and transformer-based backbones under a controlled protocol with stratified five-fold cross-validation. We then compared ensembling strategies (hard voting, weighted soft voting, stacking) and investigated a hybrid class-level fusion variant to exploit grade-specific advantages. For interpretability, we produced Grad-CAM++ visual attribution maps and short textual rationales using vision-language models (VLMs) conditioned on the fundus image and classifier outputs under conservative prompting constraints. Modern CNN backbones (ResNet-50 and ConvNeXt-Tiny) provided the strongest single-model baselines, with cross-validated QWK up to 0.919 and 0.914, respectively. Ensembling improved ordinal agreement, and weighted soft voting was the most consistent across folds (QWK 0.934 +/- 0.017). Hybrid class-level fusion was competitive but did not yield a statistically reliable improvement over standard fusion in paired fold comparisons (Holm-adjusted p >= 1.000). For explanation quality, Grad-CAM++ offered plausible but coarse localization, and VLM rationales were generally grade-consistent. Quantitatively, VLM variants showed a trade-off between clinical completeness and template-level semantic similarity (coverage 0.700 vs. BERTScore 0.072), while image-text alignment was comparable (CLIPScore approximately 0.34). 

---
# C-MORAL: Controllable Multi-Objective Molecular Optimization with Reinforcement Alignment for LLMs 

**Authors**: Rui Gao, Youngseung Jeon, Swastik Roy, Morteza Ziyadi, Xiang 'Anthony' Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23061)  

**Abstract**: Large language models (LLMs) show promise for molecular optimization, but aligning them with selective and competing drug-design constraints remains challenging. We propose C-Moral, a reinforcement learning post-training framework for controllable multi-objective molecular optimization. C-Moral combines group-based relative optimization, property score alignment for heterogeneous objectives, and continuous non-linear reward aggregation to improve stability across competing properties. Experiments on the C-MuMOInstruct benchmark show that C-Moral consistently outperforms state-of-the-art models across both in-domain and out-of-domain settings, achieving the best Success Optimized Rate (SOR) of 48.9% on IND tasks and 39.5% on OOD tasks, while largely preserving scaffold similarity. These results suggest that RL post-training is an effective way to align molecular language models with continuous molecular design objectives. Our code and models are publicly available at this https URL. 

---
# The Security Cost of Intelligence: AI Capability, Cyber Risk, and Deployment Paradox 

**Authors**: Sukwoong Choi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23058)  

**Abstract**: Firms are deploying more capable AI systems, but organizational controls often have not kept pace. These systems can generate greater productivity gains, but high-value uses require broader authority exposure -- data access, workflow integration, and delegated authority -- when governance controls have not yet decoupled capability from authority exposure. We develop an analytical model in which a firm jointly chooses AI deployment and cybersecurity investment under this governance-capability gap. The central result shows a deployment paradox: in high-loss environments, better AI can lead a firm to deploy less when capability is deployed through broader authority exposure under weak governance. Optimal deployment also falls below the no-risk benchmark, and this shortfall widens with breach-loss magnitude and with the authority exposure attached to more capable systems. Governance investment that reduces breach-loss magnitude shrinks the paradox region itself, while breach externalities expand the range of environments in which deployment is socially constrained. Governance maturity is therefore not merely a constraint on AI adoption. It is a condition that shapes whether capability improvements translate into productive deployment. 

---
# K-Score: Kalman Filter as a Principled Alternative to Reward Normalization in Reinforcement Learning 

**Authors**: Zixuan Xia, Quanxi Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.23056)  

**Abstract**: We propose a simple yet effective alternative to reward normalization in policy gradient reinforcement learning by integrating a 1D Kalman filter for online reward estimation. Instead of relying on fixed heuristics, our method recursively estimates the latent reward mean, smoothing high-variance returns and adapting to non-stationary environments. This approach incurs minimal overhead and requires no modification to existing policy architectures. Experiments on \textit{LunarLander} and \textit{CartPole} demonstrate that Kalman-filtered rewards significantly accelerate convergence and reduce training variance compared to standard normalization techniques. Code is available at this https URL. 

---
# DeepImagine: Learning Biomedical Reasoning via Successive Counterfactual Imagining 

**Authors**: Youze Zheng, Jianyou Wang, Yuhan Chen, Matthew Feng, Longtian Bao, Hanyuan Zhang, Maxim Khan, Aditya K. Sehgal, Christopher D. Rosin, Umber Dube, Ramamohan Paturi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23054)  

**Abstract**: Predicting the outcomes of prospective clinical trials remains a major challenge for large language models. Prior work has shown that both traditional correlational predictors, such as random forests and logistic regression, and strong commercial LLMs achieve limited performance on this task. In this paper, we propose DeepImagine, a framework for teaching LLMs biomedical reasoning through successive counterfactual imagining. The central idea is to approximate hidden causal mechanisms of clinical trials by training models to infer how observed trial results would change under controlled perturbations of experimental conditions, such as dosage, outcome measures, study arms, geography, and other trial attributes. To support this objective, we construct both natural and approximate counterfactual pairs from real clinical trials with reported outcomes. For settings where strict counterfactual supervision is available, such as paired outcome measures or dose-ranging study arms within the same trial, we train models with supervised fine-tuning. For broader settings where only approximate counterfactual pairs can be retrieved, we optimize models with reinforcement learning using verifiable rewards based on downstream benchmark correctness. We further augment training with synthetic reasoning traces that provide causally plausible explanations for local counterfactual transitions. Using this pipeline, we train language models under 10B parameters, including Qwen3.5-9B, and evaluate them on clinical trial outcome prediction. We aim to show that DeepImagine consistently improves over untuned language models and traditional correlational baselines. Finally, we aim to show that the learned reasoning trajectories provide interpretable signals about how models represent trial-level mechanisms, suggesting a practical path toward more mechanistic and scientifically useful biomedical language models. 

---
# AmaraSpatial-10K: A Spatially and Semantically Aligned 3D Dataset for Spatial Computing and Embodied AI 

**Authors**: Mohammad Sadegh Salehi, Alex Perkins, Igor Maurell, Ashkan Dabbagh, Raymond Wong  

**Link**: [PDF](https://arxiv.org/pdf/2604.23018)  

**Abstract**: Web-scale 3D asset collections are abundant, but rarely deployment-ready. Assets ship with arbitrary metric scale, incorrect pivots and forward axes, brittle geometry, and textures that do not support relighting, which limits their utility for embodied AI, robotics simulation, game development, and AR/VR. We present AmaraSpatial-10K, a dataset of over 10,000 synthetic 3D assets designed for downstream use rather than volume alone. Each asset is released as a metric-scaled, semantically anchored .glb with separated PBR material maps, a convex collision hull, a paired reference image, and rich multi-sentence text metadata. The dataset spans indoor objects, vehicles, architecture, creatures, and props under a unified spatial convention. Alongside the dataset, we introduce an evaluation suite for 3D asset banks. The suite comprises a continuous Scale Plausibility Score (SPS) with an LLM-as-Judge interval protocol, an LLM Concept Density score for metadata, an anchor-error metric, and a cross-modal CLIP coherence protocol, and we use it to audit AmaraSpatial-10K alongside matched subsets from Objaverse, HSSD, ABO, and GSO. Compared with Objaverse-sourced assets, we demonstrate that AmaraSpatial-10K substantially improves text-based retrieval precision (CLIP Recall@5 of 0.612 vs 0.181, a 3.4x improvement with median rank falling from 267 to 3), and we establish that it satisfies the spatial and semantic prerequisites for physics-aware scene composition and embodied-AI asset banks, leaving those downstream evaluations to future work. AmaraSpatial-10K is publicly available on Hugging Face. 

---
# DeepSignature: Digitally Signed, Content-Encoding Watermarks for Robust and Transparent Image Authentication 

**Authors**: Mathias Graf, Marco Willi, Melanie Mathys, Michael Aerni, Christian Schwarzer, Martin Melchior, Michael H. Graber  

**Link**: [PDF](https://arxiv.org/pdf/2604.23016)  

**Abstract**: AI-powered generative models have significantly expanded the possibilities for editing, manipulating, and creating high-quality images. Particularly, images that falsely appear to originate from trusted sources pose a serious threat, undermining public trust in image authenticity. We propose DeepSignature, a novel approach that integrates the guarantees of digital signatures with the capabilities of deep neural networks. Neural networks are used both to generate content-encoding watermarks and to embed them imperceptibly into images while ensuring robust extraction. These watermarks are cryptographically verifiable, enabling source attribution and image integrity validation. DeepSignature is compatible with existing image formats and requires no special handling of signed images. It supports client-side verification, requiring only the signer's public key. Additionally, we introduce a novel latent-space verification approach to detect and localize tampering attempts. We evaluate DeepSignature in terms of imperceptibility, robustness to benign transformations, forgery detection, and its resilience against various attack scenarios. Our results highlight the inherent trade-offs between imperceptibility, robustness, and integrity verification. We demonstrate that DeepSignature reliably identifies significant forgery attempts -- achieving near 100\% in our experiments. Finally, we emphasize DeepSignature's modularity and tunable parameters, allowing adaptation to application-specific requirements. Code and model weights will be published. 

---
# Vision-Language-Action in Robotics: A Survey of Datasets, Benchmarks, and Data Engines 

**Authors**: Ziyao Wang, Bingying Wang, Hanrong Zhang, Tingting Du, Tianyang Chen, Guoheng Sun, Yexiao He, Zheyu Shen, Wanghao Ye, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.23001)  

**Abstract**: Despite remarkable progress in Vision--Language--Action (VLA) models, a central bottleneck remains underexamined: the data infrastructure that underlies embodied learning. In this survey, we argue that future advances in VLA will depend less on model architecture and more on the co-design of high-fidelity data engines and structured evaluation protocols. To this end, we present a systematic, data-centric analysis of VLA research organized around three pillars: datasets, benchmarks, and data engines. For datasets, we categorize real-world and synthetic corpora along embodiment diversity, modality composition, and action space formulation, revealing a persistent fidelity-cost trade-off that fundamentally constrains large-scale collection. For benchmarks, we analyze task complexity and environment structure jointly, exposing structural gaps in compositional generalization and long-horizon reasoning evaluation that existing protocols fail to address. For data engines, we examine simulation-based, video-reconstruction, and automated task-generation paradigms, identifying their shared limitations in physical grounding and sim-to-real transfer. Synthesizing these analyses, we distill four open challenges: representation alignment, multimodal supervision, reasoning assessment, and scalable data generation. Addressing them, we argue, requires treating data infrastructure as a first-class research problem rather than a background concern. 

---
# Hard to See, Hard to Label: Generative and Symbolic Acquisition for Subtle Visual Phenomena 

**Authors**: Renjith Prasad, Rishabh Sharma, Andrew E. Shao, Annmary Justine Koomthanam, Shreyas Kulkarni, Suparna Bhattacharya, Martin Foltin, Amit Sheth, David Orozco, Brian Sammuli  

**Link**: [PDF](https://arxiv.org/pdf/2604.22990)  

**Abstract**: Subtle visual anomalies such as hairline cracks, sub-millimeter voids, and low-contrast inclusions are structurally atypical yet visually ambiguous, making them both difficult to annotate and easy to overlook during active learning. Standard acquisition heuristics based on discriminative uncertainty or feature diversity often overselect dominant patterns while underexploring sparse yet important regions of the data space. This failure mode is especially severe in industrial defect inspection, where anomalies may be both low-prevalence and difficult to distinguish from surrounding structure. To resolve this, we propose GSAL, an active learning framework for object detection that combines a diffusion-based difficulty signal with a hierarchical semantic coverage prior. The diffusion component scores images and proposals using reconstruction discrepancy and denoising variability, prioritizing visually atypical or ambiguous examples. However, diffusion alone does not prevent acquisition from repeatedly favoring hard samples within dominant semantic modes. The semantic component therefore organizes candidate samples in a three-level concept graph and promotes coverage of underrepresented semantic regions while providing interpretable acquisition rationales. By balancing visual difficulty with semantic coverage, GSAL improves retrieval of subtle and rare targets that are often missed by uncertainty-only selection. Experiments on a proprietary thin-film defect, Pascal VOC and MS COCO dataset show consistent gains in label efficiency and rare-class retrieval over uncertainty-, diversity-, and hybrid-based baselines 

---
# CheXmix: Unified Generative Pretraining for Vision Language Models in Medical Imaging 

**Authors**: Ashwin Kumar, Robbie Holland, Corey Barrett, Jangwon Kim, Maya Varma, Zhihong Chen, Yunhe Gao, Greg Zaharchuk, Tara Taghavi, Krishnaram Kenthapadi, Akshay Chaudhari  

**Link**: [PDF](https://arxiv.org/pdf/2604.22989)  

**Abstract**: Recent medical multimodal foundation models are built as multimodal LLMs (MLLMs) by connecting a CLIP-pretrained vision encoder to an LLM using LLaVA-style finetuning. This two-stage, decoupled approach introduces a projection layer that can distort visual features. This is especially concerning in medical imaging where subtle cues are essential for accurate diagnoses. In contrast, early-fusion generative approaches such as Chameleon eliminate the projection bottleneck by processing image and text tokens within a single unified sequence, enabling joint representation learning that leverages the inductive priors of language models. We present CheXmix, a unified early-fusion generative model trained on a large corpus of chest X-rays paired with radiology reports. We expand on Chameleon's autoregressive framework by introducing a two-stage multimodal generative pretraining strategy that combines the representational strengths of masked autoencoders with MLLMs. The resulting models are highly flexible, supporting both discriminative and generative tasks at both coarse and fine-grained scales. Our approach outperforms well-established generative models across all masking ratios by 6.0% and surpasses CheXagent by 8.6% on AUROC at high image masking ratios on the CheXpert classification task. We further inpaint images over 51.0% better than text-only generative models and outperform CheXagent by 45% on the GREEN metric for radiology report generation. These results demonstrate that CheXmix captures fine-grained information across a broad spectrum of chest X-ray tasks. Our code is at: this https URL. 

---
# Peer Identity Bias in Multi-Agent LLM Evaluation: An Empirical Study Using the TRUST Democratic Discourse Analysis Pipeline 

**Authors**: Juergen Dietrich  

**Link**: [PDF](https://arxiv.org/pdf/2604.22971)  

**Abstract**: The TRUST democratic discourse analysis pipeline exposes its large language model (LLM) components to peer model identity through multiple structural channels -- a design feature whose bias implications have not previously been empirically tested. We provide the first systematic measurement of identity-dependent scoring bias across all active identity exposure channels in TRUST, crossing four model families with two anonymization scopes across 30 political statements. The central finding is that single-channel anonymization produces near-zero bias effects, because individual channels act in opposite directions and cancel each other out -- a result that would lead an evaluator to conclude that identity bias is absent when it is not. Only full-pipeline anonymization reveals the true pattern: homogeneous ensembles amplify identity-driven sycophancy when model identity is fully visible, while the heterogeneous production configuration shows the reverse. Model choice matters independently: one tested model exhibits baseline sycophancy two to three times higher than the others and near-zero deliberative conflict on ideological topics, making it structurally unsuitable for pipelines where genuine inter-role disagreement is the intended quality mechanism. Three practical conclusions follow. First, heterogeneous model ensembles are structurally more robust than homogeneous ones, achieving higher consensus rates and lower identity amplification. Second, full-pipeline anonymization is required for valid bias measurement -- partial anonymization is insufficient and actively misleading. Third, these findings have direct implications for the validation of multi-agent LLM systems in quality-critical applications: a system validated under partial anonymization or with a homogeneous ensemble may pass validation while retaining structural identity bias invisible to single-channel measurement. 

---
# Institutions for the Post-Scarcity of Judgment 

**Authors**: Lauri Lovén  

**Link**: [PDF](https://arxiv.org/pdf/2604.22966)  

**Abstract**: Each major technological revolution inverts a particular scarcity and rebuilds institutions around the shift. The near-consensus diagnosis of the AI revolution holds that AI collapses the cost of prediction while judgment remains scarce. This Opinion argues the inversion has now flipped: competent-looking judgment (selecting, ranking, attributing, certifying) is produced at scale and at marginal cost approaching zero, and four complements become scarce: verified signal, legitimacy, authentic provenance, and integration capacity (the community's tolerance for delegated cognition). Because judgment is the substance of institutions, the institutions built to manufacture legitimate judgment (courts, journals, licensing bodies, legislatures) now compete with the technology for the same functional role. The piece traces the pattern across scientific institutions, professional licensing, intellectual property, democratic legitimacy, and foundation-model concentration, and closes with a three-move agenda: reframe AI policy as institutional redesign, build provenance and verification as commons, and develop the formal apparatus for institutional composition under strategic agents. 

---
# VS-DDPM: Efficient Low-Cost Diffusion Model for Medical Modality Translation 

**Authors**: Nikoo Moradi, Gijs Luijten, Behrus Hinrichs-Puladi, Jens Kleesiek, Victor Alves, Jan Egger, André Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2604.22942)  

**Abstract**: Diffusion models produce high-quality synthetic data but suffer from slow inference. We propose 3D Variable-Step Denoising Diffusion Probabilistic Model (VS-DDPM) a framework engineered to maintain generative quality while accelerating inference by several factors. We tested our approach on four tasks (missing MRI, tumor removal, MRI-to-sCT, and CBCT-to-sCT) within the BraTS2025 and SynthRAD2025 challenges. Designed for high efficiency under hardware and time constrains imposed by both challenges. VS-DDPM achieved state-of-the-art (SOTA) performance in missing MRI synthesis, yielding Dice scores of 0.80, 0.83, and 0.88 for the enhancing tumor, tumor core, and whole tumor regions, respectively, alongside a structural similarity index (SSIM) of 0.95. For MRI tumor removal, the model attained a root mean squared error (RMSE) of 0.053, a peak signal-to-noise ratio (PSNR) of 26.77, and an SSIM of 0.918. While the framework demonstrated competitive performance in MRI-to-sCT and CBCT-to-sCT tasks, it did not reach SOTA benchmarks, potentially due to sensitivities in data pre and post-processing pipelines or specific loss function configurations. These results demonstrate that VS-DDPM provides a robust and tunable solution for high-fidelity 3D medical image synthesis. The code is available in this https URL. 

---
# Self Knowledge Re-expression: A Fully Local Method for Adapting LLMs to Tasks Using Intrinsic Knowledge 

**Authors**: Mengyu Wang, Xiaoying Zhi, Zhiyi Li, Robin Schmucker, Shay B. Cohen, Tiejun Ma, Fran Silavong  

**Link**: [PDF](https://arxiv.org/pdf/2604.22939)  

**Abstract**: While the next-token prediction (NTP) paradigm enables large language models (LLMs) to express their intrinsic knowledge, its sequential nature constrains performance on specialized, non-generative tasks. We attribute this performance bottleneck to the LLMs' knowledge expression mechanism, rather than to deficiencies in knowledge acquisition. To address this, we propose Self-Knowledge Re-expression (SKR), a novel, task-agnostic adaptation method. SKR transforms the LLM's output from generic token generation to highly efficient, task-specific expression. SKR is a fully local method that uses only unannotated data, requiring neither human supervision nor model distillation. Experiments on a large financial document dataset demonstrate substantial improvements: over 40% in Recall@1 for information retrieval tasks, over 76% reduction in object detection latency, and over 33% increase in anomaly detection AUPRC. Our results on the MMDocRAG dataset surpass those of leading retrieval models by at least 12.6%. 

---
# CT-Guided Spatially-varying Regularization for Voxel-Wise Deformable Whole-Body PET Registration 

**Authors**: Xiangcen Wu, Ruohua Chen, Sichun Li, Qianye Yang, Sheng Liu, Jianjun Liu, Zhaoheng Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.22905)  

**Abstract**: Whole-body Positron Emission Tomography (PET) registration is essential for multi-parametric tumor characterization and assessment of metastatic disease progression. In deep learning-based deformable registration, the dense displacement field (DDF) regularizer is crucial for stabilizing optimization and preventing unrealistic deformations in large 3D volumes. A key challenge in whole-body deformable registration is anatomical heterogeneity, rigid structures (e.g., bones) should undergo stronger regularization, whereas soft tissues require more flexible deformation and weaker constraints. In this work, we propose a simple yet effective CT-guided spatially-varying regularization strategy for whole-body cross-tracer deformable PET registration. The key idea is to use the paired CT volume from the PET/CT acquisition to construct a voxel-wise regularization map for the DDF, replacing the conventional single global regularization weight. This yields anatomy-adaptive regularization strength across rigid and soft tissues. The proposed method is evaluated on a real clinical cross-tracer PET/CT dataset of 296 patients involving 18F-PSMA and 18F-FDG, showing that the proposed method achieves statistically significant improvements over weakly-supervised registration baseline in both whole-body registration performance and organ-wise alignment. 

---
# On the Complementarity of Quantum and Classical Features: Adaptive Hybrid Quantum-Classical Feature Fusion for Breast Cancer Classification 

**Authors**: Yasmin Rodrigues Sobrinho, João Renato Ribeiro Manesco, João Paulo Papa  

**Link**: [PDF](https://arxiv.org/pdf/2604.22903)  

**Abstract**: The integration of quantum machine learning with classical deep learning offers promising avenues for medical image analysis by mapping data into high-dimensional Hilbert spaces. However, effectively unifying these distinct paradigms remains challenging due to common optimization asymmetries. In this paper, a novel hybrid quantum-classical architecture for breast cancer diagnosis based on a dual-branch feature-extraction pipeline is proposed. Our framework extracts and unifies complementary representations from classical models and quantum circuits, exploring both trainable and deterministic (non-trainable) quantum paradigms. To integrate these embeddings, three progressive feature fusion strategies are introduced: Static Hybrid Fusion (SHF) for offline extraction, Dynamic Hybrid Fusion (DHF) for end-to-end co-adaptation, and a novel Temperature-Scaled Hybrid Fusion (TSHF). The TSHF strategy incorporates a learnable scalar, inspired by multimodal learning, that dynamically balances hybrid gradient dynamics and resolves optimization bottlenecks. Empirical validation on the BreastMNIST dataset confirms our hypothesis that unifying diverse feature representations creates a richer data context. The TSHF strategy, specifically when pairing a ResNet backbone with a trainable quantum circuit, achieved a peak accuracy of 87.82%, F1-score of 91.77%, and an AUC-ROC of 89.08%, outperforming purely classical baselines. These results demonstrate that the proposed hybrid framework improves classification accuracy and threshold reliability, providing a stable, high-performance architecture for the clinical deployment of quantum-enhanced diagnostic tools. 

---
# Reconstructive Authority Model: Runtime Execution Validity Under Partial Observability 

**Authors**: Marcelo Fernandez - TraslaIA  

**Link**: [PDF](https://arxiv.org/pdf/2604.22898)  

**Abstract**: Autonomous systems increasingly operate under partial observability where execution-relevant state is never fully accessible. Existing governance mechanisms -- trusted execution environments, oracle-signed state proofs, cryptographic attestation -- enforce the integrity of computation and state projections. We show this is structurally insufficient: an authenticated projection of state is necessary but never sufficient for execution validity.
We introduce the Reconstructive Authority Model (RAM), which separates integrity from coverage. RAM defines a reconstruction gate that reasons over an explicit coverage envelope -- comprising proven state, declared assumptions, and an acknowledged unobservable residual -- and permits execution only when coverage is adequate for the action class. When coverage is insufficient, RAM narrows privileges dynamically or fails closed. Attestation proves trust in measurement; RAM proves adequacy of what is measured.
We formalize RAM, prove necessity via two theorems (attestation insufficiency and RAM necessity) and three corollaries, and present a hybrid RAM+Attestation architecture with privilege-narrowing. Synthetic experiments (N=100,000, seed=42) show RAM achieves zero invalid execution rates at all coverage levels. Attestation-based systems exhibit IER=0.423 at low coverage and IER=0.233 even at full coverage, the latter arising from undefined-state handling failures undetectable by integrity checks alone.
This reframes execution validity as a coverage reconstruction problem, distinct from and complementary to integrity guarantees provided by attestation. 

---
# Citation-Driven Multi-View Training for Patent Embeddings: QaECTER and Sophia-Bench 

**Authors**: Younes Djemmal, You Zuo, Kim Gerdes, Kirian Guiller  

**Link**: [PDF](https://arxiv.org/pdf/2604.22897)  

**Abstract**: Patent retrieval underpins critical decisions in innovation, examination, and IP strategy, yet progress has been hampered by the absence of benchmarks that reflect the diversity of real world search scenarios. We address this gap with two contributions. First, we introduce Sophiabench, a large-scale patent retrieval benchmark comprising 10,000 queries and 75,000 corpus documents stratified across ten years, eight IPC technology sections, and twelve filing jurisdictions. Unlike prior benchmarks, Sophia-bench tests retrieval using 12 different query types-from structured patent fields to AI-generated summaries-and evaluates results against citation-based ground truth enhanced with a novel domain-relevance metric (InScope). Together, these enable systematic measurement of how well models perform across query types, technology domains, and jurisdictions. Second, we introduce QaECTER, a 344M-parameter embedding model trained on patent citation graphs and multi-view self-alignment. Despite its compact size, QaECTER establishes a new state of the art for patent retrieval. It outperforms the \#1 model on the English retrieval text embedding benchmark (RTEB), a model 23x larger, as well as all existing patent specific models across every query type, IPC section, and jurisdiction on Sophia-bench, with gains of up to 7.2% average NDCG@10 over the next-best model. These results are confirmed on an independent external benchmark, where QaECTER surpasses all prior models without requiring task-specific instruction prompts. Both the benchmark and the model are designed for practical deployment in large-scale patent search systems. 

---
# Utility-Aware Data Pricing: Token-Level Quality and Empirical Training Gain for LLMs 

**Authors**: Minghui Xu, Qi Luo, Kun Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.22893)  

**Abstract**: Traditional data valuation methods based on ``row-count $\times$ quality coefficient'' paradigms fail to capture the nuanced, nonlinear contributions that data makes to Large Language Model (LLM) capabilities. This paper presents a dynamic data valuation framework that transitions from static accounting to utility-based pricing. Our approach operates on three layers: (1) token-level information density metrics using Shannon entropy and Data Quality Scores; (2) empirical training gain measurement through influence functions, proxy model strategies, and Data Shapley values; and (3) cryptographic verifiability through hash-based commitments, Merkle trees, and a tamper-evident training ledger. We provide comprehensive experimental validation on three real domains (instruction following, mathematical reasoning, and code summarization), demonstrating that proxy-based empirical gain achieves near-perfect ranking alignment with realized utility, substantially outperforming row-count and token-count baselines. This framework enables a fair Data-as-a-Service economy where high-reasoning data is priced according to its actual contribution to model intelligence, while providing the transparency and auditability necessary for trustworthy data markets. 

---
# Quantifying and Mitigating Self-Preference Bias of LLM Judges 

**Authors**: Jinming Yang, Chuxian Qiu, Zhenyu Deng, Xinshan Jiao, Tao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.22891)  

**Abstract**: LLM-as-a-Judge has become a dominant approach in automated evaluation systems, playing critical roles in model alignment, leaderboard construction, quality control, and so on. However, the scalability and trustworthiness of this approach can be substantially distorted by Self-Preference Bias (SPB), which is a directional evaluative deviation in which LLMs systematically favor or disfavor their own generated outputs during evaluation. Existing measurements rely on costly human annotations and conflate generative capability with evaluative stance, and thus are impractical for large-scale deployment in real-world systems. To address this issue, we introduce a fully automated framework to quantifying and mitigating SPB, which constructs equal-quality pairs of responses with negligible quality differences, enabling statistical disentanglement of discriminability from bias propensity without human gold standards. Empirical analysis across 20 mainstream LLMs reveals that advanced capabilities are often uncorrelated, or even negatively correlated, with low SPB. To mitigate this bias, we propose a structured multi-dimensional evaluation strategy grounded in cognitive load decomposition, which reduces SPB by 31.5\% on average. 

---
# RouteGuard: Internal-Signal Detection of Skill Poisoning in LLM Agents 

**Authors**: Wenjie Xiao, Xuehai Tang, Biyu Zhou, Songlin Hu, Jizhong Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.22888)  

**Abstract**: Agent skills introduce a new and more severe form of indirect injection for LLM agents: unlike traditional indirect prompt injection, attackers can hide malicious instructions inside a dense, action-oriented skill that already functions as a legitimate instruction source. We study pre-execution skill-poison detection and show that successful skill poisoning induces a structured internal effect, attention hijacking, in which response-time attention shifts from trusted context to malicious skill spans and drives harmful behavior. Motivated by this mechanism, we propose RouteGuard, a frozen-backbone detector that combines response-conditioned attention and hidden-state alignment through reliability-gated late fusion. Across both real and synthetic open-source skill benchmarks, RouteGuard is consistently the strongest or most robust detector; on the critical Skill-Inject channel slice, it reaches 0.8834 F1 and recovers 90.51% of description attacks missed by lexical screening, showing that defending against skill poisoning requires internal-signal detection rather than text-only filtering 

---
# Federated Cross-Modal Retrieval with Missing Modalities via Semantic Routing and Adapter Personalization 

**Authors**: Hefeng Zhou, Xuan Liu, Sicheng Chen, Wutong Zhang, Wu Yan, Jiong Lou, Chentao Wu, Guangtao Xue, Wei Zhao, Jie Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.22885)  

**Abstract**: Federated cross-modal retrieval faces severe challenges from heterogeneous client data, particularly non-IID semantic distributions and missing modalities. Under such heterogeneity, a single global model is often insufficient to capture both shared cross-modal knowledge and client-specific characteristics. We propose RCSR, a personalization-friendly federated framework that integrates prototype anchoring, retrieval-centric semantic routing, and optional client-specific adapters. Built on a frozen CLIP backbone, RCSR leverages lightweight shared adapters for global knowledge transfer while supporting efficient local personalization. Prototype anchoring helps unimodal clients align with global cross-modal semantics, and a server-side semantic router adaptively assigns aggregation weights based on retrieval consistency to mitigate alignment drift during heterogeneous updates. Extensive experiments on MS-COCO, Flickr30K, and other benchmarks show that RCSR consistently improves global retrieval accuracy and training stability, while further enhancing client-level retrieval performance, especially for clients with incomplete modalities. Code is available at this https URL. 

---
# Can Multimodal Large Language Models Truly Understand Small Objects? 

**Authors**: Fujun Han, Junan Chen, Xintong Zhu, Jingqi Ye, Xuanjie Mao, Tao Chen, Peng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2604.22884)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown promising potential in diverse understanding tasks, e.g., image and video analysis, math and physics olympiads. However, they remain blank and unexplored for Small Object Understanding (SOU) tasks. To fill this gap, we introduce SOUBench, the first and comprehensive benchmark for exploring the small objects understanding capability of existing MLLMs. Specifically, we first design an effective and automatic visual question-answer generation strategy, constructing a new SOU-VQA evaluation dataset, with 18,204 VQA pairs, six relevant sub-tasks, and three dominant scenarios (i.e., Driving, Aerial, and Underwater). Then, we conduct a comprehensive evaluation on 15 state-of-the-art MLLMs and reveal their weak capabilities in small object understanding. Furthermore, we develop SOU-Train, a multimodal training dataset with 11,226 VQA pairs, to improve the SOU capabilities of MLLMs. Through supervising fine-tuning of the latest MLLM, we demonstrate that SOU-Train can effectively enhance the latest MLLM's ability to understand small objects. Comprehensive experimental results demonstrate that, the proposed SOUBench, along with the SOU-VQA and SOU-Train datasets, provides a crucial empirical foundation to the community for further developing models with enhanced small object understanding capabilities. Datasets and Code: this https URL. 

---
# NeuroAPS-Net: Neuro-Anatomically Aware Point Cloud Representation for Efficient Alzheimer's Disease Classification 

**Authors**: Towhidul Islam, Mufti Mahmud  

**Link**: [PDF](https://arxiv.org/pdf/2604.22883)  

**Abstract**: Alzheimer's disease (AD) is a progressive neurodegenerative disorder and a major cause of dementia. Structural MRI is widely used to analyze AD-related brain atrophy; however, most deep learning methods rely on computationally expensive 3D convolutional neural networks (CNNs), limiting deployment in resource-constrained settings. This work introduces two main contributions. First, we propose a pipeline that converts T1-weighted MRI into anatomically informed 2D point clouds using Anatomical Priority Sampling (APS), producing ADNI-2DPC, the first neuroanatomically labeled MRI-derived point cloud dataset. Second, we present NeuroAPS-Net, a lightweight geometric deep learning model that incorporates anatomical priors via region-aware feature encoding and ROI token aggregation. Experiments on ADNI-2DPC demonstrate that NeuroAPS-Net achieves competitive classification accuracy while significantly reducing inference latency and GPU memory compared to state-of-the-art point cloud methods. These results highlight the potential of anatomically guided point cloud learning as an efficient and interpretable alternative to voxel-based CNNs for AD classification. 

---
# MTServe: Efficient Serving for Generative Recommendation Models with Hierarchical Caches 

**Authors**: Xin Wang, Chi Ma, Shaobin Chen, Pu Wang, Menglei Zhou, Junyi Qiu, Qiaorui Chen, Jiayu Sun, Shijie Liu, Zehuan Wang, Lei Yu, Chuan Liu, Fei Jiang, Wei Lin, Hao Wang, Jiawei Jiang, Xiao Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.22881)  

**Abstract**: Generative recommendation (GR) offers superior modeling capabilities but suffers from prohibitive inference costs due to the repeated encoding of long user histories. While cross-request Key-Value (KV) cache reuse presents a significant optimization opportunity, the massive scale of individual user states creates a storage explosion that far exceeds physical GPU limits. We propose MTServe, a hierarchical cache management system that virtualizes GPU memory by leveraging host RAM as a scalable backup store. To bridge the I/O gap between tiers, MTServe introduces a suite of system-level optimizations, including a hybrid storage layout, an asynchronous data transfer pipeline, and a locality-driven replacement policy. On both public and production datasets, MTServe delivers up to 3.1* speedup while maintaining near-perfect hit ratios (>98.5%). 

---
# Beyond Single-Agent Alignment: Preventing Context-Fragmented Violations in Multi-Agent Systems 

**Authors**: Jie Wu, Ming Gong  

**Link**: [PDF](https://arxiv.org/pdf/2604.22879)  

**Abstract**: We identify and formalize a novel security risk: Context-Fragmented Violations (CFVs) - a class of policy breaches where individual agent actions appear locally safe and reasonable, yet collectively violate organizational policies because critical policy facts are siloed in different departments private contexts. Existing prompt-based alignment mechanisms and monolithic interceptors are poorly matched to violations that span contextual islands. We propose Distributed Sentinel, a distributed zero-trust enforcement architecture that introduces the Semantic Taint Token (STT) Protocol. Through lightweight sidecar proxies, our system propagates security state across organizational boundaries without exposing raw cross-domain data, enabling Counterfactual Graph Simulation for cross-domain policy verification. We construct PhantomEcosystem, a comprehensive benchmark comprising 9 categories of realistic cross-agent violation scenarios with adversarially balanced safe controls. On this benchmark, Distributed Sentinel achieves F1 = 0.95 with 106ms end-to-end latency (16ms verification + 90ms entity extraction on A100), compared to 0.85 F1 for prompt-based filtering and 0.65 for rule-based DLP. To empirically validate the need for external enforcement, we evaluate eight frontier LLMs in execution-oriented multi-agent workflows with per-agent domain world models. All models exhibit substantial violation rates (14-98%), with cross-domain data flows showing systematically higher violation rates than same-domain flows. These results indicate that self-avoidance is unreliable and that multi-agent security benefits from a centralized enforcement layer operating above individual agents. 

---
# SketchVLM: Vision language models can annotate images to explain thoughts and guide users 

**Authors**: Brandon Collins, Logan Bolton, Hung Huy Nguyen, Mohammad Reza Taesiri, Trung Bui, Anh Totti Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2604.22875)  

**Abstract**: When answering questions about images, humans naturally point, label, and draw to explain their reasoning. In contrast, modern vision-language models (VLMs) such as Gemini-3-Pro and GPT-5 only respond with text, which can be difficult for users to verify. We present SketchVLM, a training-free, model-agnostic framework that enables VLMs to produce non-destructive, editable SVG overlays on the input image to visually explain their answers. Across seven benchmarks spanning visual reasoning (maze navigation, ball-drop trajectory prediction, and object counting) and drawing (part labeling, connecting-the-dots, and drawing shapes around objects), SketchVLM improves visual reasoning task accuracy by up to +28.5 percentage points and annotation quality by up to 1.48x relative to image-editing and fine-tuned sketching baselines, while also producing annotations that are more faithful to the model's stated answer. We find that single-turn generation already achieves strong accuracy and annotation quality, and multi-turn generation opens up further opportunities for human-AI collaboration. An interactive demo and code are at this https URL. 

---
# When Policies Cannot Be Retrained: A Unified Closed-Form View of Post-Training Steering in Offline Reinforcement Learning 

**Authors**: Elias Hossain, Mohammad Jahid Ibna Basher, Ivan Garibay, Ozlem Garibay, Niloofar Yousefi  

**Link**: [PDF](https://arxiv.org/pdf/2604.22873)  

**Abstract**: Offline reinforcement learning (RL) can learn effective policies from fixed datasets, but deployment objectives may change after training, and in many applications the trained actor cannot be retrained because of data, cost, or governance constraints. We study deployment-time adaptation for frozen offline actors using Product-of-Experts (PoE) composition with a goal-conditioned prior. Our main practical finding is graceful degradation rather than universal performance gain: under degraded or random priors, precision-weighted composition remains anchored to the frozen actor, while additive and prior-only adaptation collapse, and a KL-budget selector often recovers a near-oracle operating point. We also make explicit a closed-form identity in the frozen-actor setting: for diagonal-Gaussian actors and priors, PoE with coefficient alpha yields the same deterministic policy as KL-regularized adaptation with beta = alpha / (1 - alpha), with posterior covariances differing only by a global scalar factor. Empirically, across four D4RL environments (3,900 MuJoCo episodes), we observe a 4/5/3 HELP/FROZEN/HURT split. Extending the analysis to six harder cells and two AntMaze diagnostics reveals an actor-competence ceiling: medium-expert remains HURT in all 9 cells at every tested alpha, while AntMaze with a behavior-cloned frozen actor yields zero success for all composition rules. Overall, PoE and KL-regularized adaptation are best viewed as a single actor-anchored safety mechanism for deployment-time steering. 

---
# AutoRISE: Agent-Driven Strategy Evolution for Red-Teaming Large Language Models 

**Authors**: Tanmay Gautam, Alireza Bahramali, Sandeep Atluri  

**Link**: [PDF](https://arxiv.org/pdf/2604.22871)  

**Abstract**: Automated red-teaming methods for large language models typically optimize attack prompts within a fixed, human-designed strategy, leaving the attack strategy itself unchanged. We instead optimize the strategy. We propose AutoRISE, a method that searches over executable attack programs rather than individual prompts. At each iteration, a coding agent edits a strategy and a fixed evaluation harness scores the resulting attacks, returning both a scalar objective and per-example diagnostics that guide subsequent edits. This allows structural changes, including new attack components and altered control flow, that prompt-level methods do not directly express. We also release two benchmark suites developed on disjoint target sets and evaluate on 11 models from five families against seven established jailbreak datasets. Across held-out models, AutoRISE improves average attack success rate by 17.0 points over the strongest baseline, and improves attack success by up to 16 points on frontier targets with low baseline success rates. Ablations against parametric and strategy-library baselines suggest that these gains arise from unrestricted program search, particularly compositional techniques and control-flow edits. AutoRISE operates in a black-box, inference-only setting, requiring no fine-tuning, human annotation, or GPU compute. 

---
# Towards Understanding the Expressive Power of GNNs with Global Readout 

**Authors**: Maurice Funk, Daumantas Kojelis  

**Link**: [PDF](https://arxiv.org/pdf/2604.22870)  

**Abstract**: We study the expressive power of message-passing aggregate-combine-readout graph neural networks (ACR-GNNs). Particularly, we focus on the first-order (FO) properties expressible by this formalism. While a tight logical characterisation remains a difficult open question, we make two contributions towards answering it. First, we show that sum aggregation and readout suffice for GNNs to capture FO properties that cannot be expressed in the logic C2 on both directed and undirected graphs. This strengthens known results by Hauke and Wał{\k e}ga (2026) where aggregation and readout functions are specially crafted for the task. Second, we identify two natural ways of restoring characterisability (with regard to C2) for ACR-GNNs. One option is to limit local aggregation (without imposing restrictions on global readout), whilst the second is to run ACR-GNNs over graphs of bounded degree (but unbounded size). In both cases, the FO properties captured by GNNs are exactly those definable by a formula in graded modal logic with global counting modalities. Our results thus establish an innate lower- and upper-bound in terms of how far (fragments of) C2 can be taken to characterise GNNs, and imply that is indeed the unbounded interaction of aggregation and readout that pushes the logical expressive power of GNNs above C2. 

---
# Avionic Main Fuel Pump Simulation and Fault-Diagnosis Benchmark 

**Authors**: Felix Leonhard Janzen, Lukas Moddemann, Alexander Diedrich, Oliver Niggemann  

**Link**: [PDF](https://arxiv.org/pdf/2604.22869)  

**Abstract**: In many cyber-physical systems, especially in critical applications such as aeroplanes, data to train anomaly detection and diagnosis algorithms is lacking due to data protection issues and partial observability. To combat this inherent lack of data, we introduce a high-fidelity, physics-informed co-simulation of a common aircraft main-fuel-pump system modelled in \textsc{MATLAB/Simulink Simscape Fluids}. We also describe its generated time-series data with health and fault mode annotations. To show feasibility of our benchmark, we apply an unsupervised Recurrent Variational Autoencoder (RNN-VAE) for anomaly detection and a SOM-VAE for operating mode discretization, trained to separate healthy and faulty conditions. 

---
# Probing Visual Planning in Image Editing Models 

**Authors**: Zhimu Zhou, Yanpeng Zhao, Qiuyu Liao, Bo Zhao, Xiaojian Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.22868)  

**Abstract**: Visual planning represents a crucial facet of human intelligence, especially in tasks that require complex spatial reasoning and navigation. Yet, in machine learning, this inherently visual problem is often tackled through a verbal-centric lens. While recent research demonstrates the promise of fully visual approaches, they suffer from significant computational inefficiency due to the step-by-step planning-by-generation paradigm. In this work, we present EAR, an editing-as-reasoning paradigm that reformulates visual planning as a single-step image transformation. To isolate intrinsic reasoning from visual recognition, we employ abstract puzzles as probing tasks and introduce AMAZE, a procedurally generated dataset that features the classical Maze and Queen problems, covering distinct, complementary forms of visual planning. The abstract nature of AMAZE also facilitates automatic evaluation of autoregressive and diffusion-based models in terms of both pixel-wise fidelity and logical validity. We assess leading proprietary and open-source editing models. The results show that they all struggle in the zero-shot setting, finetuning on basic scales enables remarkable generalization to larger in-domain scales and out-of-domain scales and geometries. However, our best model that runs on high-end hardware fails to match the zero-shot efficiency of human solvers, highlighting a persistent gap in neural visual reasoning. 

---
# IntrAgent: An LLM Agent for Content-Grounded Information Retrieval through Literature Review 

**Authors**: Fengbo Ma, Zixin Rao, Xiaoting Li, Zhetao Chen, Hongyue Sun, Yiping Zhao, Xianyan Chen, Zhen Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22861)  

**Abstract**: Scientific research relies on accurate information retrieval from literature to support analytical decisions. In this work, we introduce a new task, INformation reTRieval through literAture reVIEW (IntraView), which aims to automate fine-grained information retrieval faithfully grounded in the provided content in response to research-driven queries, and propose IntrAgent, an LLM-based agent that addresses this challenging task. In particular, IntrAgent is designed to mimic human behaviors when reading literature for information retrieval -- identifying relevant sections and then iteratively extracting key details to refine the retrieved information. It follows a two-stage pipeline: a Section Ranking stage that prioritizes relevant literature sections through structural-knowledge-enabled reasoning, and an Iterative Reading stage that continuously extracts details and synthesizes them into concise, contextually grounded answers. To support rigorous evaluation, we introduce IntraBench, a new benchmark consisting of 315 test instances built from expert-authored questions paired with literature spanning five STEM domains. Across seven backbone LLMs, IntrAgent achieves on average 13.2% higher cross-domain accuracy than state-of-the-art RAG and research-agent baselines. 

---
# MAE-Based Self-Supervised Pretraining for Data-Efficient Medical Image Segmentation Using nnFormer 

**Authors**: R. M. Krishna Sureddi, T. Satyanarayana Murthy, Nomula Varsha Reddy, Adi Kanishka, Nalla Manvika Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2604.22854)  

**Abstract**: Transformer architectures, including nnFormer,have demonstrated promising results in volumetric medical image segmentation by being able to capture long-range spatial interactions. Although they have high performance, these models need large quantities of labeled training data and are also likely to overfit and become training unstable. This is a serious practical problem because it is not only time-consuming but also expensive to obtain medical images that are annotated by experts. Moreover, fully supervised traditional training pipelines do not take advantage of the available large amounts of unlabeled medical imaging data that can be easily obtained in the clinics. We have solved these drawbacks by advancing the efficiency of the nnFormer with a self-supervised pretraining framework, which is based on the Masked Autoencoders (MAE). In this method, the model is pretrained on unlabeled volumetric medical images to reconstruct randomly masked parts of the input. This allows the encoder to learn meaningful anatomical and structural representations . The encoder is then further fine-tuned on a labeled dataset on the downstream segmentation task. Conducted Experiment shows that the offered method leads to a higher segmentation performance on the count of Dice score, a quicker convergence rate on the course of the fine-tuning procedure, and a superior generalization on the basis of limited labeled data . These findings validate that self-supervised learning combined with transformer-based segmentation models is an appropriate approach to the problem of data shortage in medical image analysis. 

---
# SwarmDrive: Semantic V2V Coordination for Latency-Constrained Cooperative Autonomous Driving 

**Authors**: Anjie Qiu, Donglin Wang, Zexin Fang, Sanket Partani, Hans D. Schotten  

**Link**: [PDF](https://arxiv.org/pdf/2604.22852)  

**Abstract**: Cloud-hosted LLM inference for autonomous driving adds round-trip delay and depends on stable connectivity, while purely local edge models struggle under occlusion. We present SwarmDrive, a semantic Vehicle-to-Vehicle (V2V) coordination framework in which nearby vehicles run local Small Language Models (SLMs), share compact intent distributions only when uncertainty is high, and fuse them through event-triggered consensus. We evaluate SwarmDrive in a 5-seed executable study built around one occluded intersection case, combining matched operating-point comparisons with robustness sweeps. In that setting, SwarmDrive under its 6G communication setting ("Swarm 6G") raises success from 68.9% to 94.1% over a single local SLM while reducing latency from a 510 ms cloud reference to 151.4 ms. However, an increased number of participating vehicles leads to higher communication overhead and packet loss. SwarmDrive also evaluates the impact of swarm-size, packet-loss, and entropy-threshold sweeps and shows that the cooperative gain holds across ablations and is best balanced near an active swarm size of 4 vehicles and an entropy trigger threshold of 0.65 in the current prototype. These results show that semantic edge cooperation can work under tight latency constraints in the targeted intersection case, but they are not a deployment-grade validation of a real 6G stack. 

---
# Structure Guided Retrieval-Augmented Generation for Factual Queries 

**Authors**: Miao Xie, Xiao Zhang, Yi Li, Chunli Lv  

**Link**: [PDF](https://arxiv.org/pdf/2604.22843)  

**Abstract**: Retrieval-Augmented Generation (RAG) has been proposed to mitigate hallucinations in large language models (LLMs), where generated outputs may be factually incorrect. However, existing RAG approaches predominantly rely on vector similarity for retrieval, which is prone to semantic noise and fails to ensure that generated responses fully satisfy the complex conditions specified by factual queries, often leading to incorrect answers. To address this challenge, we introduce a novel research problem, named Exact Retrieval Problem (ERP). To the best of our knowledge, this is the first problem formulation that explicitly incorporates structural information into RAG for factual questions to satisfy all query conditions. For this novel problem, we propose Structure Guided Retrieval-Augmented Generation (SG-RAG), which models the retrieval process as an embedding-based subgraph matching task, and uses the retrieved topological structures to guide the LLM to generate answers that meet all specified query conditions. To facilitate evaluation of ERP, we construct and publicly release Exact Retrieval Question Answering (ERQA), a large-scale dataset comprising 120000 fact-oriented QA pairs, each involving complex conditions, spanning 20 diverse domains. The experimental results demonstrate that SG-RAG significantly outperforms strong baselines on ERQA, delivering absolute improvements from 20.68 to 50.88 points across all evaluation metrics, while maintaining reasonable computational overhead. 

---
# From Skeletons to Pixels: Few-Shot Precise Event Spotting via Representation and Prediction Distillation 

**Authors**: Zhong Han Ervin Yeoh, Jiang Kan  

**Link**: [PDF](https://arxiv.org/pdf/2604.22839)  

**Abstract**: Precise Event Spotting (PES) is essential in fast-paced sports such as tennis, where fine-grained events occur within very short temporal windows. Accurate frame-level localization is challenging because of motion blur, subtle action differences, and limited annotated data. We study two complementary distillation strategies for few-shot PES: Adaptive Weight Distillation (AWD), a prediction-level method that adaptively weights teacher supervision on unlabeled data, and Annealed Multimodal Distillation for Few-Shot Event Detection (AMD-FED), a representation-level framework that transfers robust skeleton knowledge into visual modalities through annealed pseudo-labeling. Both methods use multimodal distillation to improve generalization under limited supervision. We evaluate them on F3Set-Tennis(sub) under few-shot k-clip settings, where they consistently outperform single-modality baselines and prior PES approaches. After observing the stronger performance of representation-level distillation on tennis, we further validate AMD-FED on a second sports dataset, Figure Skating, where it also shows robust performance in the k-clip scenario. These results highlight the effectiveness of multimodal distillation, especially representation-level transfer, for few-shot precise event spotting. 

---
# Neural Network Optimization Reimagined: Decoupled Techniques for Scratch and Fine-Tuning 

**Authors**: Xin Ning, Qiankun Li, Xiaolong Huang, Qiupu Chen, Feng He, Weijun Li, Prayag Tiwari, Xinwang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22838)  

**Abstract**: With the accumulation of resources in the era of big data and the rise of pre-trained models in deep learning, optimizing neural networks for various tasks often involves different strategies for fine-tuning pre-trained models versus training from scratch. However, existing optimizers primarily focus on reducing the loss function by updating model parameters, without fully addressing the unique demands of these two major paradigms. In this paper, we propose DualOpt, a novel approach that decouples optimization techniques specifically tailored for these distinct training scenarios. For training from scratch, we introduce real-time layer-wise weight decay, designed to enhance both convergence and generalization by aligning with the characteristics of weight updates and network architecture. For more importantly fine-tuning, we integrate weight rollback with the optimizer, incorporating a rollback term into each weight update step. This ensures consistency in the weight distribution between upstream and downstream models, effectively mitigating knowledge forgetting and improving fine-tuning performance. Additionally, we extend the layer-wise weight decay to dynamically adjust the rollback levels across layers, adapting to the varying demands of different downstream tasks. Extensive experiments across diverse tasks, including image classification, object detection, semantic segmentation, and instance segmentation, demonstrate the broad applicability and state-of-the-art performance of DualOpt. Code is available at this https URL. 

---
# OAMVOS:2nd Report for 5th PVUW MOSE Track 

**Authors**: Deshui Miao, Xingsen Huang, Yameng Gu, Xiaogang yu, Xin Li, Ming-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22837)  

**Abstract**: SAM-based dense trackers provide strong short-term mask propagation but remain fragile under long occlusion, fast motion, viewpoint change, and distractors. The problem is especially severe for small objects, where a few incorrect memory updates can dominate later predictions. This report presents an occlusion- and reappearance-aware extension of DAM4SAM that improves memory control rather than changing the backbone. The method augments the original SAM3 tracker with four ingredients: a reliability-aware tracking state machine, branch-based recovery, delayed DRM promotion, and a selective policy for native SAM3 memory selection. During stable tracking, the model follows the original single-path propagation process. Once confidence drops, the tracker enters an ambiguous or recovery mode, maintains a small set of candidate branches, and commits memory only after a branch is reconfirmed. For small-object disappearance and reappearance, native memory selection is temporarily bypassed so older anchors remain accessible. In addition, the first conditioning frame is explicitly preserved, and the conditioning-memory budget is moderately enlarged to improve long-gap recovery. The resulting design keeps DAM4SAM efficient in easy cases while improving robustness in sequences dominated by occlusion and reappearance. 

---
# ParkingScenes: A Structured Dataset for End-to-End Autonomous Parking in Simulation Scenes 

**Authors**: Haonan Chen, Kaiwen Xiao, Bin Tian, Jun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22835)  

**Abstract**: Autonomous parking remains a critical yet challenging task in intelligent driving systems, particularly within constrained urban environments where maneuvering space is limited and precise control is essential. While recent advances in end-to-end learning have shown great promise, the lack of high-quality, structured datasets tailored for parking scenarios remains a significant this http URL address this gap, we present ParkingScenes, a comprehensive multimodal dataset specifically designed for end-to-end autonomous parking in simulated scenes. Built on the CARLA simulator, ParkingScenes features structured parking trajectories generated by a Hybrid A* planner and a Model Predictive Controller (MPC), providing accurate and reproducible supervision signals. The dataset includes 16 reverse-in and 6 parallel parking scenarios, each executed under two pedestrian conditions (present and absent), resulting in 704 structured episodes and approximately 105000 frames. Each scenario is repeated 16 times to ensure consistent coverage. Each frame contains synchronized data from four RGB cameras, four depth sensors, vehicle motion states, and Bird's-Eye View (BEV) representations, enabling rich multimodal fusion and context-aware learning. To demonstrate the utility of our dataset, we compare models trained on ParkingScenes with those trained on unstructured, manually collected simulation data under identical conditions. Results show significant improvements in performance, underscoring the effectiveness of structured supervision for robust and accurate parking policy learning. By releasing both the dataset and the collection framework, ParkingScenes establishes a scalable and reproducible benchmark for advancing learning-based autonomous parking systems. The dataset and collection framework will be released at: this https URL 

---
# Intervention-Aware Multiscale Representation Learning from Imaging Phenomics and Perturbation Transcriptomics 

**Authors**: Jiayuan Chen, Ruoqi Liu, Zishan Gu, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22832)  

**Abstract**: Microscopy-based phenotypic profiling is scalable for drug discovery but lacks the mechanistic depth of transcriptomics, which remains costly and scarce. Existing multimodal approaches either use images to support other modalities or naively align representations by sample identity, ignoring cell-type and dose variations in weakly paired data-limiting generalization to unseen interventions. In this paper, we introduce an intervention-aware distillation framework that leverages perturbational transcriptomics to guide image representation learning. A transcriptome-conditioned teacher integrates gene expression and intervention metadata to produce soft distributions over a chemistry-aware codebook organized by drug similarity. The teacher employs a fine-tuned single-cell foundation model to encode cell-type context and disentangle dose effects. An image-only student learns to predict these distributions from microscopy alone, distilling mechanistic knowledge while operating independently at test time. This design emphasizes intervention semantics rather than identity alignment and explicitly handles dose and cell-type mismatches. We provide theoretical guarantees showing that transcriptomic guidance tightens the risk bound for image-based prediction. On Cell Painting and RxRx datasets paired with L1000, our method significantly improves one-shot transfer to unseen interventions and drug-target gene discovery compared to self-supervised and alignment baselines. 

---
# MetaEarth3D: Unlocking World-scale 3D Generation with Spatially Scalable Generative Modeling 

**Authors**: Jinqi Cao, Zhiping Yu, Baihong Lin, Chenyang Liu, Zhenwei Shi, Zhengxia Zou  

**Link**: [PDF](https://arxiv.org/pdf/2604.22828)  

**Abstract**: Recent generative AI models have achieved remarkable breakthroughs in language and visual understanding. However, although these models can generate realistic visual content, their spatial scale remains confined to bounded environments, preventing them from capturing how geographic environments evolve across thousands of kilometers or from modeling the spatial structure of the large-scale physical world. This limitation poses a critical challenge for ultra-wide-area spatial intelligence in Earth observation and simulation, revealing a deeper gap in generative AI: progress has relied primarily on scaling model parameters and training data, while overlooking spatial scale as a core dimension of intelligence. Here, motivated by this missing dimension, we investigate spatial scale as a new scaling axis in foundation models and present MetaEarth3D, the first generative foundation model capable of spatially consistent generation at the planetary scale. Taking optical Earth observation simulation as a testbed, MetaEarth3D enables the generation of multi-level, unbounded, and diverse 3D scenes spanning large-scale terrains, medium-scale cities, and fine-grained street blocks. Built upon 10 million globally distributed real-world training images, MetaEarth3D demonstrates both strong visual realism and geospatial statistical realism. Beyond generation, MetaEarth3D serves as a generative data engine for diverse virtual environments in ultra-wide spatial intelligence. We argue that this study may help empower next-generation spatial intelligence for Earth observation. 

---
# SGP-SAM: Self-Gated Prompting for Transferring 3D Segment Anything Models to Lesion Segmentation 

**Authors**: Zixuan Tang, Shen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.22825)  

**Abstract**: Large segmentation foundation models such as the Segment Anything Model (SAM) have reshaped promptable segmentation in natural images, and recent efforts have extended these models to medical images and volumetric settings. However, directly transferring a 3D SAM-style model to lesion segmentation remains challenging due to (i) weak spatial representational capacity for small, irregular targets in intermediate features, and (ii) extreme foreground-background imbalance in 3D this http URL propose SGP-SAM, a self-gated prompting framework for efficient and effective transfer to 3D lesion segmentation. Our key component, the Self-Gated Prompting Module (SGPM), performs conditional multi-scale spatial enhancement: a lightweight multi-channel gating unit predicts whether the current features require additional multi-scale fusion, and only then activates a Multi-Scale Feature Fusion Block to enrich spatial context. To further address small-lesion learning, we design a Zoom Loss that up-weights lesion-focused supervision by combining Dice and a voxel-balanced focal this http URL on MSD Liver Tumor and MSD Brain Tumor (enhancing tumor) show consistent gains over strong transfer baselines based on SAM-Med3D. On MSD Liver Tumor, SGP-SAM improves mDice by 7.3% over fine-tuning. 

---
# WeatherSeg: Weather-Robust Image Segmentation using Teacher-Student Dual Learning and Classifier-Updating Attention 

**Authors**: Zhang Zhang, Yifeng Zeng, Jinquan Pan, Yinghui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2604.22824)  

**Abstract**: WeatherSeg, an advanced semi-supervised segmentation framework, addresses autonomous driving's environmental perception challenges in adverse weather while reducing annotation costs. This framework integrates a Dual Teacher-Student Weight-Sharing Model (DTSWSM) that enables knowledge distillation from weather-affected images, and a Classifier Weight Updating Attention Mechanism (CWUAM) that dynamically adjusts classifier weights based on environmental attributes. Comprehensive evaluations demonstrate that WeatherSeg significantly outperforms baseline models in both accuracy and robustness across various weather conditions, including clear, rainy, cloudy, and foggy scenarios, establishing it as an effective solution for all-weather semantic segmentation in autonomous driving and related applications. 

---
# PivotMerge: Bridging Heterogeneous Multimodal Pre-training via Post-Alignment Model Merging 

**Authors**: Zibo Shao, Baochen Xiong, Xiaoshan Yang, Yaguang Song, Qimeng Zhang, Haifeng Chen, Changsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22823)  

**Abstract**: Multimodal Large Language Models (MLLMs) rely on multimodal pre-training over diverse data sources, where different datasets often induce complementary cross-modal alignment capabilities. Model merging provides a cost-effective mechanism for integrating multiple expert MLLMs with complementary strengths into a unified model. However, existing model merging research mainly focuses on post-finetuning scenarios, leaving the pre-training stage largely unexplored. We argue that the core of MLLM pre-training lies in establishing effective cross-modal alignment, which bridges visual and textual representations into a unified semantic space. Motivated by this insight, we introduce the post-alignment merging task, which aims to integrate cross-modal alignment capabilities learned from heterogeneous multimodal pre-training. This setting introduces two key challenges: cross-domain parameter interference, where parameter updates learned from different data distributions conflict during merging, and layer-wise alignment contribution disparity, where different layers and projectors contribute unevenly to cross-modal alignment. To address them, we propose \textbf{PivotMerge}, a post-alignment merging framework for cross-modal projectors. PivotMerge incorporates two key components: Shared-space Decomposition and Filtering, which disentangles shared alignment patterns from domain-specific variations and suppresses conflicting directions, and Alignment-guided Layer-wise Merging, which assigns layer-specific merging weights based on differing alignment contributions. We construct systematic CC12M-based post-alignment merging scenarios for evaluation. Extensive experiments on multiple multimodal benchmarks show that PivotMerge consistently outperforms existing baselines, demonstrating its effectiveness and generalization ability. 

---
# DO-Bench: An Attributable Benchmark for Diagnosing Object Hallucination in Vision-Language Models 

**Authors**: JiYang Wang, Jiawei Chen, Mengqi Xiao, Yu Cheng, Yangfu Li, Zhaoxia Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.22822)  

**Abstract**: Object level hallucination remains a central reliability challenge for vision language models (VLMs), particularly in binary object existence verification. Existing benchmarks emphasize aggregate accuracy but rarely disentangle whether errors stem from perceptual limitations or from the influence of contextual textual priors, leaving underlying failure mechanisms ambiguous. We introduce DO-Bench, a controlled diagnostic benchmark that isolates these sources through structured multimodal interventions. Rather than evaluating models in unconstrained settings, DO-Bench probes two complementary dimensions: the Prior Override dimension progressively strengthens contextual textual priors while holding visual evidence constant to assess resistance to prior pressure, and the Perception-Limited dimension incrementally enhances visual evidence from full-scene context to localized object crops to measure perceptual grounding strength. This paired design enables attribution of errors to prior suppression, perceptual insufficiency, or their interaction. We further define two diagnostic metrics, PriorRobust and PerceptionAbility, to quantify these behaviors consistently. Evaluations across diverse open- and closed-source VLMs reveal systematic differences in prior sensitivity and perceptual reliability, demonstrating that object hallucination reflects heterogeneous, mechanism dependent failure patterns beyond aggregate accuracy. 

---
# Complete Cyclic Subtask Graphs for Tool-Using LLM Agents: Flexibility, Cost, and Bottlenecks in Multi-Agent Workflows 

**Authors**: Luay Gharzeddine, Samer Saab Jr  

**Link**: [PDF](https://arxiv.org/pdf/2604.22820)  

**Abstract**: Long-horizon tool-using tasks sometimes benefit from revisiting earlier subtasks for recovery and exploration, but added multi-agent workflow flexibility can also introduce coordination overhead and substantial inference cost. We study complete cyclic subtask graphs, a deliberately maximally flexible multi-agent architecture in which executable subtask nodes are fully connected and a unified state-analysis-and-routing agent selects transitions using natural-language criteria. This makes unrestricted revisitation explicit and directly analyzable at the subtask level. We evaluate task-specific (Spec-Cyc) and benchmark-generic (Gen-Cyc) graphs on TextCraft, ALFWorld, and Finance-Agent, with ablations over planner/executor/router strength, tool exposure (generalist vs specialized), $n$-shot successful trajectory summaries, and fault-injected random subtask perturbations. The benchmarks expose three distinct regimes. ALFWorld highlights a setting where explicit revisitation supports recovery and exploration; TextCraft, a largely prerequisite-chain domain, often favors the efficiency of simpler forward execution; and Finance-Agent remains bottlenecked by retrieval, grounding, and evidence synthesis more than by workflow flexibility alone. Shared-win token comparisons further show that the added flexibility can be substantially more expensive than a single ReAct agent. Overall, we use complete cyclic subtask graphs as a maximally flexible experimental lens for measuring when multi-agent revisitation helps, when it mainly adds coordination cost, and when external task bottlenecks dominate. 

---
# Representation Homogeneity and Systemic Instability in AI-Dominated Financial Markets: A Structural Approach 

**Authors**: Yimeng Qiu, Qiwei Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.22818)  

**Abstract**: This paper investigates how similarity in the informational representation of market states among Artificial Intelligence (AI) trading agents can generate systemic instability in financial markets. We construct a structural multi-agent market model calibrated using high-frequency microstructural moments. AI agents are modeled through a two-layer decision architecture consisting of a nonlinear representation layer and an adaptive linear readout layer. The representation layer maps raw market states into high-dimensional feature vectors, while the readout layer generates return forecasts that feed into a risk-controlled trading rule. This representation-based microfoundation separates two objects that are often conflated in the literature: representation homogeneity (the degree to which agents encode market states into similar feature spaces) and forecast overlap (the degree to which agents produce similar return predictions). We show theoretically that these two concepts are related but not equivalent, and that representation homogeneity can compress the effective space of forecast disagreement under stress even when predictions appear diverse in normal times. Through controlled factorial experiments that vary representation homogeneity while conditioning on alternative risk-aversion and learning-rate distributions, we hypothesize that increasing representation similarity amplifies synchronization in beliefs and positions, leading to volatility clustering, liquidity stress, and elevated tail risk. Our structural mechanisms suggest that low perceived volatility regimes can endogenously accumulate hidden leverage through position stickiness, which subsequently collapses when shocks trigger synchronized deleveraging. The results provide a structural foundation for macroprudential policies aimed at monitoring and preserving diversity in how AI systems represent and process market information. 

---
# Applied AI-Enhanced RF Interference Rejection 

**Authors**: Rahul Jain, Pierre Trepagnier, Rick Gentile, Joey Botero, Alexia Schulz  

**Link**: [PDF](https://arxiv.org/pdf/2604.22816)  

**Abstract**: AI-enhanced interference rejection in radio frequency (RF) transmissions has recently attracted interest because deep learning approaches trained on both the signal of interest (SOI) and the signal mixture (SOI plus interference) can outperform traditional approaches which only consider the SOI. The goal is to detect, demodulate, and decode signals over a range of signal-to-interference-plus-noise (SINR) levels without having a detailed, design-level knowledge of the interfering signal or the propagation conditions. Our present AI interference suppression results are based on Autoregressive Transformer Decoder models which exhibit orders of magnitude faster throughput at inference time than WaveNet models developed in earlier work. As a specific example, we investigate an analog FM "Walkie Talkie" radio signal of interest in the presence of an Orthogonal Frequency-Division Multiplexing (OFDM) interferer. This type of interferer is near-ubiquitous in the current RF landscape. Our results clearly show the benefits of transformer-based interference mitigation in tactical settings. We show that unintelligible transmissions become intelligible via metrics such as Perceptual Evaluation of Speech Quality (PESQ), while overall latency is kept to a minimum using readily available lightweight GPUs such as a Jetson AGX Orin. We believe these same techniques can also be applied to a broader set of national security scenarios, as well as having commercial applications. 

---
# FreqFormer: Hierarchical Frequency-Domain Attention with Adaptive Spectral Routing for Long-Sequence Video Diffusion Transformers 

**Authors**: Haopeng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2604.22808)  

**Abstract**: Long-sequence video diffusion transformers hit a quadratic self-attention cost that dominates runtime and memory for very long token sequences. Most efficient attention methods use one approximation everywhere, yet video features are spectrally structured: low frequencies carry global layout and coarse motion; high frequencies carry texture and fine detail. We present FreqFormer, a frequency-aware heterogeneous attention framework. Token features are split into spectral bands with different operators: dense global attention on compressed low-frequency content, structured block-sparse attention on mid frequencies, and sliding-window local attention on high frequencies. A lightweight spectral routing network allocates heads across bands using layer statistics and the diffusion timestep, shifting compute toward global structure early in denoising and detail later. Cross-band summary tokens provide cheap residual exchange. FreqFormer is paired with a fused GPU execution plan that co-schedules dense, sparse, and local branches to cut kernel launches and memory traffic. We give a consistent complexity model, an orthonormal-decomposition view of approximation, and simulation-based systems numbers (throughput, arithmetic intensity, memory traffic, duration scaling). In simulations from 64K to 1M tokens, FreqFormer substantially reduces estimated attention FLOPs and KV-related memory traffic versus dense attention while keeping a hardware-friendly pattern, supporting spectrally structured heterogeneous attention as a practical direction for long-video diffusion transformers. 

---
# See No Evil: Semantic Context-Aware Privacy Risk Detection for AR 

**Authors**: Jialu Liu, Yao Li, Zhuoheng Li, Huining Li, Ying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.22805)  

**Abstract**: Augmented reality (AR) systems pose unique privacy risks due to their continuous capture of visual data. Existing AR privacy frameworks lack semantic understanding of visual content, limiting their effectiveness in detecting context-dependent privacy risks. We propose PrivAR, which leverages vision language models (VLMs) with chain-of-thought prompting for contextual privacy risk detection in AR environments. PrivAR uses visual scene cues to infer potential sensitive information types, such as identifying password notes in office environments through contextual reasoning. PrivAR detects and obfuscates textual content, preventing exposure of sensitive information while preserving contextual cues necessary for VLM inference. Additionally, we investigate contextually-informed warning interfaces to enhance user privacy awareness. Experiments on a real-world AR dataset show that PrivAR achieves superior accuracy (81.48%) and F1-score (84.62%) compared to baselines, while reducing privacy leakage rate to 17.58%. User studies evaluating contextually-informed warning interfaces provide insights into effective privacy-aware AR design. 

---
# RCSB PDB AI Help Desk: retrieval-augmented generation for protein structure deposition support 

**Authors**: Vivek Reddy Chithari, Jasmine Y. Young, Irina Persikova, Yuhe Liang, Gregg V. Crichlow, Justin W. Flatt, Sutapa Ghosh, Brian P. Hudson, Ezra Peisach, Monica Sekharan, Chenghua Shao, Stephen K. Burley  

**Link**: [PDF](https://arxiv.org/pdf/2604.22800)  

**Abstract**: Motivation: Structural Biologists have contributed more than 245,000 experimentally determined three-dimensional structures of biological macromolecules to the Protein Data Bank (PDB). Incoming data are validated and biocurated by ~20 expert biocurators across the wwPDB. RCSB PDB biocurators who process more than 40% of global depositions face increasing challenges in maintaining efficient Help Desk operations, with approximately 19,000 messages in approximately 8,000 entries received from depositors in 2025.
Results: We developed an AI-powered Help Desk using Retrieval-Augmented Generation (RAG) built on LangChain with a pgvector store (PostgreSQL) and GPT-4.1-mini. The system employs pymupdf4llm for Markdown-preserving PDF extraction, two-stage document chunking, Maximal Marginal Relevance retrieval, a topical guardrail that filters off-topic queries, and a specialized system prompt that prevents exposure of internal terminology. A dual-LLM architecture uses separate model configurations for question condensing and response generation. Deployed in production on Kubernetes with PostgreSQL (pgvector), it provides around-the-clock depositor assistance with citation-backed, streaming responses.
Availability and implementation: Freely available at this https URL. 

---
# UGAF-ITS: A Standards Harmonization Framework and Validation Tool for Multi-Framework AI Governance in Distributed Intelligent Transportation Systems 

**Authors**: Talal Ashraf Butt, Muhammad Iqbal, Razi Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2604.22789)  

**Abstract**: Organizations deploying AI-enabled Intelligent Transportation Systems face fragmented governance: ISO/IEC 42001 demands a certifiable management system, the EU AI Act imposes binding high-risk obligations from August 2026, and the NIST AI Risk Management Framework structures voluntary practice. Each instrument is internally coherent, yet they drive different control vocabularies, evidence expectations, and audit rhythms. In distributed ITS deployments where vehicle manufacturers, roadside integrators, and cloud operators each hold partial evidence and partial accountability, this fragmentation multiplies compliance effort and obscures incident traceability.
This paper introduces UGAF-ITS, a standards harmonization framework that consolidates 154 source obligations from the three instruments into 12 unified controls across eight governance domains through a reproducible five-phase crosswalk methodology. A three-tier operating model allocates each control to the vehicle, edge, or cloud tier where enforcement and defensible evidence production are feasible. An evidence backbone of 20 versioned artifacts supports a single audit package across all three frameworks without duplicating content.
We validate UGAF-ITS through an open-source governance engine evaluated across four architecturally distinct ITS deployment scenarios. The engine encodes the complete crosswalk catalog and executes eight compliance computations. Three-tier deployments achieve 91.7% average framework coverage with 45.9% evidence reduction, complete bidirectional traceability, and 80% of artifacts serving all three frameworks simultaneously. Partial deployments degrade gracefully: coverage and reduction scale with architectural complexity. The tool, scenarios, and all reported results are publicly available for independent replication. 

---
# Conformal PM2.5 Mapping Under Spatial Covariate Shift: Satellite-Reanalysis Fusion for Africa's Green Industrial Transition 

**Authors**: Yaw Osei Adjei, Davis Opoku, Ephraim Abotsi, Kwadwo Owusu Amanqua, Oliver Kornyo, Elisha Soglo-Ahianyo, Cephas Anertey Abbey  

**Link**: [PDF](https://arxiv.org/pdf/2604.22787)  

**Abstract**: Africa's green industrialization imperative demands reliable infrastructure for monitoring air quality. We present a satellite-reanalysis PM2.5 fusion system trained on 2,068,901 records from 404 monitoring locations in 29 African countries (OpenAQ, 2017-2022), combining LightGBM with leakage-resistant spatial cross-validation and conformal prediction to quantify predictions and their geographic applicability limits. Under 5-fold location-grouped spatial cross-validation, LightGBM achieves RMSE = 30.83 +/- 5.07 ug/m3, MAE = 14.54 +/- 1.66 ug/m3, R2 = 0.134 +/- 0.023, and macro F1 = 0.336 +/- 0.018. This R2 is substantially below random-split benchmarks (>0.90) but reflects true geographic generalisation difficulty rather than model failure. Split conformal prediction targeting 90% marginal coverage reveals severe East Africa degradation (actual PICP = 65.3% vs. nominal 90%), consistent with medium-strength covariate shift (humidity KS = 0.2237, sat_pblh KS = 0.2558). We operationalise these findings through regional reliability flags (High/Medium/Low/Unreliable) and a monitor prioritisation score directing infrastructure expansion toward highest-burden unmonitored populations, directly supporting Africa's green industrial transition and SDGs 3.9, 7.1.2, 9, 11.6.2, and 13. 

---
# Parameter Efficiency Is Not Memory Efficiency: Rethinking Fine-Tuning for On-Device LLM Adaptation 

**Authors**: Irene Tenison, Stella Ahn, Miriam Kim, Ebtisam Alshehri, Lalana Kagal  

**Link**: [PDF](https://arxiv.org/pdf/2604.22783)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) has become the standard for adapting large language models (LLMs). In this work we challenge the wide-spread assumption that parameter efficiency equates memory efficiency and on-device adaptability. We show that this is not true - while methods like LoRA and IA3 significantly reduce trainable parameters, they remain bound by intermediate tensors that scale linearly with sequence length, often triggering out-of-memory errors on-device. In this work, we introduce LARS (Low-memory Activation-Rank Subspace), a novel adaptation framework that decouples memory consumption from sequence length. While prior PEFT methods apply low-rank constraints to model parameters, LARS instead constrains the activation subspace used during training, directly targeting the dominant source of memory consumption and fundamentally flattening the memory growth rate. LARS reduces the memory footprint by an average of 33.54% on GPUs and 51.95% on CPUs in comparison to LoRA across reasoning, understanding and long-context datasets using different models while maintaining competitive accuracy and throughput. Besides GPUs, we deploy on Raspberry Pi and consumer-grade CPUs to demonstrate that LARS provides a scalable path for sophisticated LLM personalization on resource-constrained hardware and edge devices. 

---
# Stochastic KV Routing: Enabling Adaptive Depth-Wise Cache Sharing 

**Authors**: Anastasiia Filippova, David Grangier, Marco Cuturi, João Monteiro  

**Link**: [PDF](https://arxiv.org/pdf/2604.22782)  

**Abstract**: Serving transformer language models with high throughput requires caching Key-Values (KVs) to avoid redundant computation during autoregressive generation. The memory footprint of KV caching is significant and heavily impacts serving costs. This work proposes to lessen these memory requirements. While recent work has largely addressed KV cache reduction via compression and eviction along the temporal axis, we argue that the \emph{depth} dimension offers an orthogonal and robust avenue for optimization. Although prior research suggests that a full cache for every layer is redundant, implementing cross-layer cache sharing remains a practical challenge; existing methods typically suffer from reduced throughput or increased time-to-first-token. In this paper, we demonstrate that dropping a layer's cache offers efficient optimization without information loss. We propose a simple training approach: random cross-layer attention. During training, layers randomly choose to attend either to their own KV states or those of a preceding layer. This stochastic process adapts the model to be robust to various depth-wise cache sharing strategies, ensuring flexibility for unknown hardware constraints at deployment time. Our evaluations show that applying this scheme during pre-training or fine-tuning enables depth-wise cache sharing for various model families. Furthermore, for larger models in data-constrained settings, this approach is suggestive of a regularization-like effect, frequently preserving or improving performance while significantly reducing the cache's memory footprint. 

---
# BiTA: Bidirectional Gated Recurrent Unit-Transformer Aggregator in a Temporal Graph Network Framework for Alert Prediction in Computer Networks 

**Authors**: Zahra Makki Nayeri, Mohsen Rezvani  

**Link**: [PDF](https://arxiv.org/pdf/2604.22781)  

**Abstract**: Proactive alert prediction in computer networks is critical for mitigating evolving cyber threats and enabling timely defensive actions. Temporal Graph Neural Networks (TGNs) provide a principled framework for modeling time-evolving interactions; however, existing TGN-based methods predominantly rely on unidirectional or single-mechanism temporal aggregation, which limits their ability to capture recursive, multi-scale temporal patterns commonly observed in real-world attack behaviors. In this paper, we propose BiTA, a Bidirectional Gated Recurrent Unit-Transformer Aggregator for temporal graph learning. Rather than introducing a deeper or higher-capacity model, BiTA redesigns the temporal aggregation function within the TGN framework by jointly encoding bidirectional sequential dependencies and long-range contextual relations over each node's temporal neighborhood. This aggregation strategy enables complementary temporal reasoning at different scales while preserving the original TGN memory and message-passing structure. We evaluate BiTA on real-world alert datasets, demonstrating significant improvements in key performance metrics such as area under the curve, average precision, mean reciprocal rank, and per-category prediction accuracy when compared to state-of-the-art temporal graph models. BiTA outperforms baseline methods under both transductive and inductive settings, highlighting its robustness and generalization capabilities in dynamic network environments. BiTA is a scalable and interpretable framework for real-time cyber threat anticipation, paving the way toward more intelligent and adaptive intrusion detection systems. 

---
# A General Framework for Generative Self-supervised Learning in Non-invasive Estimation of Physiological Parameters Using Photoplethysmography 

**Authors**: Zexing Zhang, Huimin Lu, Songzhe Ma, Jianzhong Peng, Chenglin Lin, Niya Li, Bingwang Dong  

**Link**: [PDF](https://arxiv.org/pdf/2604.22780)  

**Abstract**: Aligning physiological parameter labels with large-scale photoplethysmographic (PPG) data for deep learning is challenging and resource-intensive. While self-supervised representation learning (SSRL) can handle limited annotated data, the challenge lies in learning robust shared representations from vast unlabeled data and integrating contextual cues to learn distinctive representations. To alleviate these challenges, a generative SSRL framework TS2TC is proposed to utilize the temporal, spectrogram, and temporal-spectrogram mixed domains to explore and incorporate the unique features of PPG for universal and noninvasive physiological parameter estimation. A pretext task named Cross-Temporal Fusion Generative Anchor (CTFGA) is designed, modeling temporal dependencies and reconstructing independent segments at a coarse level to provide robust global feature extraction and local contextual representation. The framework includes sub-signals from PPG with diverse frequency scales and order derivatives reflecting hemodynamics to facilitate learning shared representations at varying semantic levels. Secondly, a cognitive-inspired dual-process transfer (DPT) strategy is formulated, consisting of prior-dependent autonomous processes and posterior observation reasoning processes, to leverage the independent and integrated advantages of shared and specific representations. TS2TC introduces a bilinear temporal-spectrogram fusion method in the mixed domain, aligning latent representations from different domains and establishing fine-grained contextual interactions across multiple sources of information. Extensive experiments on physiological parameter estimation tasks showed that the joint performance of CTFGA and DPT outperforms standard generative learning significantly. TS2TC achieved an average 2.49\% improvement in RMSE over state-of-the-art estimation methods with only 10\% training data. 

---
# KARL: Mitigating Hallucinations in LLMs via Knowledge-Boundary-Aware Reinforcement Learning 

**Authors**: Cheng Gao, Cheng Huang, Kangyang Luo, Ziqing Qiao, Shuzheng Si, Huimin Chen, Chaojun Xiao, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.22779)  

**Abstract**: Enabling large language models (LLMs) to appropriately abstain from answering questions beyond their knowledge is crucial for mitigating hallucinations. While existing reinforcement learning methods foster autonomous abstention, they often compromise answer accuracy because their static reward mechanisms, agnostic to models' knowledge boundaries, drive models toward excessive caution. In this work, we propose KARL, a novel framework that continuously aligns an LLM's abstention behavior with its evolving knowledge boundary. KARL introduces two core innovations: a Knowledge-Boundary-Aware Reward that performs online knowledge boundary estimation using within-group response statistics, dynamically rewarding correct answers or guided abstention; and a Two-Stage RL Training Strategy that first explores the knowledge boundary and bypasses the "abstention trap", and subsequently converts incorrect answers beyond the knowledge boundary into abstentions without sacrificing accuracy. Extensive experiments on multiple benchmarks demonstrate that KARL achieves a superior accuracy-hallucination trade-off, effectively suppressing hallucinations while maintaining high accuracy across both in-distribution and out-of-distribution scenarios. 

---
# The Spectral Lifecycle of Transformer Training: Transient Compression Waves, Persistent Spectral Gradients, and the Q/K--V Asymmetry 

**Authors**: Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22778)  

**Abstract**: We present the first systematic study of weight matrix singular value spectra \emph{during} transformer pretraining, tracking full SVD decompositions of every weight matrix at 25-step intervals across three model scales (30M--285M parameters). We discover three phenomena: \textbf{(1)~Transient Compression Waves:} stable rank compression propagates as a traveling wave from early to late layers, creating a dramatic gradient that peaks early then \emph{reverses} -- late layers eventually over-compress past early layers. \textbf{(2)~Persistent Spectral Gradients:} the power-law exponent~$\alpha$ develops a permanent depth gradient forming a non-monotonic inverted-U in deeper models, with peaks shifting toward earlier layers as depth increases. \textbf{(3)~Q/K--V Functional Asymmetry:} value/output projections compress uniformly while query/key projections carry the full depth-dependent dynamics. The dissociation between transient compression and persistent spectral shape reveals that \emph{rank and spectral shape encode fundamentally different information about training}. We formalize this as a two-timescale dynamical model and derive scaling laws ($\Delta\alpha \propto L^{0.26}$, $R^2{=}0.99$). We validate on nine models across three families (custom, GPT-2, Pythia; 30M--1B parameters; 8--36 layers), demonstrate that $\alpha$ predicts layer importance ($\rho{=}0.69$--$0.84$, $p{<}0.02$), and show that spectral-guided pruning outperforms Last-N heuristics by $1.1{\times}$--$3.6{\times}$ across seven models in two families (GPT-2 124M--774M, Pythia 160M--1B), with worst-vs-best gaps up to $23.7{\times}$ confirming the causal role of spectral structure. 

---
# Epicure: Multidimensional Flavor Structure in Food Ingredient Embeddings 

**Authors**: Jakub Radzikowski, Josef Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.22776)  

**Abstract**: A chef's intuition about flavor, texture, and cultural identity represents tacit knowledge that is difficult to articulate yet central to culinary practice. We show that this knowledge is already encoded in FlavorGraph's 300-dimensional ingredient embeddings, trained on recipe cooccurrence and food chemistry, and that it can be systematically recovered. An LLM-augmented curation pipeline consolidates 6,653 raw FlavorGraph ingredients into 1,032 canonical entries, substantially strengthening the recoverable structure. We identify at least fifteen independently classifiable dimensions spanning taste, texture, geography, food processing, and culture. 

---
# When VLMs 'Fix' Students: Identifying and Penalizing Over-Correction in the Evaluation of Multi-line Handwritten Math OCR 

**Authors**: Jin Seong, Wencke Liermann, Minho Kim, Jong-hun Shin, Soojong Lim  

**Link**: [PDF](https://arxiv.org/pdf/2604.22774)  

**Abstract**: Accurate transcription of handwritten mathematics is crucial for educational AI systems, yet current benchmarks fail to evaluate this capability properly. Most prior studies focus on single-line expressions and rely on lexical metrics such as BLEU, which fail to assess the semantic reasoning across multi-line student solutions. In this paper, we present the first systematic study of multi-line handwritten math Optical Character Recognition (OCR), revealing a critical failure mode of Vision-Language Models (VLMs): over-correction. Instead of faithfully transcribing a student's work, these models often "fix" errors, thereby hiding the very mistakes an educational assessment aims to detect. To address this, we propose PINK (Penalized INK-based score), a semantic evaluation metric that leverages a Large Language Model (LLM) for rubric-based grading and explicitly penalizes over-correction. Our comprehensive evaluation of 15 state-of-the-art VLMs on the FERMAT dataset reveals substantial ranking reversals compared to BLEU: models like GPT-4o are heavily penalized for aggressive over-correction, whereas Gemini 2.5 Flash emerges as the most faithful transcriber. Furthermore, human expert studies show that PINK aligns significantly better with human judgment (55.0% preference over BLEU's 39.5%), providing a more reliable evaluation framework for handwritten math OCR in educational settings. 

---
# The Randomness Floor: Measuring Intrinsic Non-Randomness in Language Model Token Distributions 

**Authors**: Jarosław Hryszko  

**Link**: [PDF](https://arxiv.org/pdf/2604.22771)  

**Abstract**: Language models cannot be random. This paper introduces Entropic Deviation (ED), the normalised KL divergence between a model's token distribution and the uniform distribution, and measures it systematically across 31,200 generations spanning seven models, two architectures (transformer and state space), nine prompt categories, three temperatures, and five languages. Under semantically neutral prompts (empty strings, random characters, nonsense syllables) transformers still exhibit ED of approximately 0.30, meaning that 88-93% of the non-randomness observed under semantic prompts is intrinsic to the learned weights rather than induced by context. Three transformer families (Gemma, Llama, Qwen) converge on nearly identical ED values despite different training data and vocabularies. A state space model (Mamba2) reveals a qualitatively different regime: twice the ED, three times lower within-sequence variance, and massive sensitivity to temperature (r = -0.78) where transformers are nearly immune (r < 0.05). Cross-lingual experiments with Qwen-32B show a stable gradient across five languages (English, Japanese, Chinese, Polish, Arabic) that does not correlate with token fertility and persists when two languages sharing an identical tokeniser subset are compared. These findings establish a structural lower bound on randomness in pretrained language models, characterise how this bound differs across architectures, and demonstrate that language itself modulates the bound independently of tokenisation. 

---
# Learning in Blocks: A Multi Agent Debate Assisted Personalized Adaptive Learning Framework for Language Learning 

**Authors**: Nicy Scaria, Silvester John Joseph Kennedy, Deepak Subramani  

**Link**: [PDF](https://arxiv.org/pdf/2604.22770)  

**Abstract**: Most digital language learning curricula rely on discrete-item quizzes that test recall rather than applied conversational proficiency. When progression is driven by quiz performance, learners can advance despite persistent gaps in using grammar and vocabulary during interaction. Recent work on LLM-based judging suggests a path toward scoring open-ended conversations, but using interaction evidence to drive progression and review requires scoring protocols that are reliable and validated. We introduce Learning in Blocks, a framework that grounds progression in demonstrated conversational competence evaluated using CEFR-aligned rubrics. The framework employs heterogeneous multi-agent debate (HeteroMAD) in two stages: a scoring stage where role-specialized agents independently evaluate Grammar, Vocabulary, and Interactive Communication, engage in debate to address conflicting judgments, and a judge synthesizes consensus scores; and a recommendation stage that identifies specific grammar skills and vocabulary topics for targeted review. Progression requires demonstrating 70% mastery, and spaced review targets identified weaknesses to counter skill decay. We benchmark four scoring and recommendation methods on CEFR A2 conversations annotated by ESL experts. HeteroMAD achieves a superior score agreement with a 0.23 degree of variation and recommendation acceptability of 90.91%. An 8-week study with 180 CEFR A2 learners demonstrates that combining rubric-aligned scoring and recommendation with spaced review and mastery-based progression produces better learning outcomes than feedback alone. 

---
# The Imbalanced User-AI Relationships as an Ethical Failure of Front-End Design in Healthcare AI 

**Authors**: Maureen Mghambi Mwadime  

**Link**: [PDF](https://arxiv.org/pdf/2604.22767)  

**Abstract**: Ethical discourse on AI in healthcare has focused predominantly on back-end concerns such as bias, fairness and explainability, while the front-end interface, where patients and clinicians actually encounter AI outputs, remains under explored. This paper identifies imbalanced user-AI relationships as a distinct class of front-end ethical failure: patients are rendered highly visible to AI systems through data inference, yet cannot understand, question or influence how they are represented. Through the concept of asymmetric legibility and a chat-based telemedicine case, we show how design choices e.g., default recommendations, restricted inputs and suppressed uncertainty, undermine agency, clinician judgment and human oversight even where systems are technically accurate. We propose reciprocity as a design orientation and offer interventions for more balanced, participatory user-AI relationships in healthcare. 

---
# Artificial General Intelligence Forecasting and Scenario Analysis: State of the Field, Methodological Gaps, and Strategic Implications 

**Authors**: Gopal P. Sarma, Sunny D. Bhatt, Michael Jacob, Rachel Steratore  

**Link**: [PDF](https://arxiv.org/pdf/2604.22766)  

**Abstract**: In this report, we review the current state of methodologies to forecast the arrival of artificial general intelligence, assess their reliability, and analyze the implications for strategy and policy. We synthesize diverse forecasting approaches, document significant limitations in existing methods, and propose a research agenda for developing more-robust forecasting infrastructure. The report does not endorse a specific forecast or scenario but rather provides a framework for interpreting forecasts under conditions of deep uncertainty. We experimented with an iterative approach to human and artificial intelligence collaboration for this report. The primary drafting of the text was performed by large language models (GPT 5.1, Gemini 3 Pro, and Claude 4.5 Opus), with human researchers providing direction, peer review, fact-checking, and revision. 

---
# Algorithmic Administration and the EU AI Act: Legal Principles for Public Sector Use of AI 

**Authors**: Georgios Pavlidis, Ioannis Kastanas  

**Link**: [PDF](https://arxiv.org/pdf/2604.22765)  

**Abstract**: The increasing use of artificial intelligence (AI) by public authorities introduces both opportunities for innovation and significant challenges for the administrative rule of law. This article examines how the EU AI Act interacts with the fundamental principles of administrative law, with a particular focus on administrative discretion, the duty to state reasons, and proportionality. It analyses the regulatory obligations imposed by the AI Act on public sector deployers of high-risk systems, especially in sensitive domains such as social benefits, migration, education, and law enforcement. It also explores whether the AI Act adequately ensures accountability, transparency, and reviewability in automated public decision-making. The article further considers how the AI Act's risk-based approach aligns (or fails to align) with the principle of proportionality and it proposes safeguards and interpretative strategies to ensure the ethical and lawful deployment of AI in the public sector. 

---
# Implicit Humanization in Everyday LLM Moral Judgments 

**Authors**: Hoda Ayad, Tanu Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2604.22764)  

**Abstract**: Recent adoption of conversational information systems has expanded the scope of user queries to include complex tasks such as personal advice-seeking. However, we identify a specific type of sought advice-a request for a moral judgment (i.e. "who was wrong?") in a social conflict-as an implicitly humanizing query which carries potentially harmful anthropomorphic projections. In this study, we examine the reinforcement of these assumptions in the responses of four major general-purpose LLMs through the use of linguistic, behavioral, and cognitive anthropomorphic cues. We also contribute a novel dataset of simulated user queries for moral judgments. We find current LLM system responses reinforce implicit humanization in queries, potentially exacerbating risks like overreliance or misplaced trust. We call for future work to expand the understanding of anthropomorphism to include implicit userside humanization and to design solutions that address user needs while correcting misaligned expectations of model capabilities. 

---
# Behavioral Intelligence Platforms: From Event Streams to Autonomous Insight via Probabilistic Journey Graphs, Behavioral Knowledge Extraction, and Grounded Language Generation 

**Authors**: Arun Patra, Bhushan Vadgave  

**Link**: [PDF](https://arxiv.org/pdf/2604.22762)  

**Abstract**: Contemporary product analytics systems require users to pose explicit queries, such as writing SQL, configuring dashboards, or constructing funnels, before insights can surface. This pull-based paradigm creates a bottleneck: it requires both domain knowledge and technical fluency, and assumes practitioners know in advance which questions to ask. We argue that behavioral analytics should move from passive systems that answer queries to active systems that continuously detect and explain behavioral phenomena.
We present the Behavioral Intelligence Platform (BIP), a system architecture that transforms raw event streams into automatically generated insights. BIP consists of four layers. First, Normalization and State Derivation (NSD) standardizes events and maps them to a semantic state hierarchy. Second, a Behavioral Graph Engine (BGE) models user journeys as absorbing Markov chains and computes transition probabilities, removal effects, and path quality metrics. Third, a Behavioral Knowledge Graph (BKG) and Detector System convert graph outputs into grounded behavioral facts and identify behavioral phenomena. Finally, a Grounded Language Layer constrains large language model outputs to verified facts, producing reliable narrative insights.
We formalize the Behavioral Intelligence Problem, introduce a taxonomy of detectors for autonomous insight generation, and propose an interestingness score to prioritize insights under limited attention. 

---
# Quantifying Divergence in Inter-LLM Communication Through API Retrieval and Ranking 

**Authors**: Eyhab Al-Masri  

**Link**: [PDF](https://arxiv.org/pdf/2604.22760)  

**Abstract**: Large language models (LLMs) increasingly operate as autonomous agents that reason over external APIs to perform complex tasks. However, their reliability and agreement remain poorly characterized. We present a unified benchmarking framework to quantify inter-LLM divergence, defined as the extent to which models differ in API discovery and ranking under identical tasks. Across 15 canonical API domains and 5 major model families, we measure pairwise and group-level agreement using set-, rank-, and consensus-based metrics including Average Overlap, Jaccard similarity, Rank-Biased Overlap, Kendall's tau, Kendall's W, and Cronbach's alpha. Results show moderate overall alignment (AO about 0.50, tau about 0.45) but strong domain dependence: structured tasks (Weather, Speech-to-Text) are stable, while open-ended tasks (Sentiment Analysis) exhibit substantially higher divergence. Volatility and consensus analyses reveal that coherence clusters around data-bound domains and degrades for abstract reasoning tasks. These insights enable reliability-aware orchestration in multi-agent systems, where consensus weighting can improve coordination among heterogeneous LLMs. Beyond performance benchmarking, our results reveal systematic failure modes in multi-agent LLM coordination, where apparent agreement can mask instability in action-relevant rankings. This hidden divergence poses a pre-deployment safety risk and motivates diagnostic benchmarks for early detection. 

---
# Beyond Static: Related Questions Retrieval Through Conversations in Community Question Answering 

**Authors**: Xiao Ao, Jie Zou, Yibiao Wei, Peng Wang, Weikang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.22759)  

**Abstract**: In community question answering (cQA) platforms like Stack Overflow, related question retrieval is recognized as a fundamental task that allows users to retrieve related questions to answer user queries automatically. Although many traditional approaches have been proposed for investigating this research field, they mostly rely on static approaches and neglect the interaction property. We argue that the conversational way can well distinguish the fine-grained representations of questions and has great potential to improve the performance of question retrieval. In this paper, we propose a related question retrieval model through conversations, called TeCQR, to locate related questions in cQA. Specifically, we build conversations by utilizing tag-enhanced clarifying questions (CQs). In addition, we design a noise tolerance model that evaluates the semantic similarity between questions and tags, enabling the model to effectively handle noisy feedback. Moreover, the tag-enhanced two-stage offline training is proposed to fully exploit the mutual relationships among user queries, questions, and tags to learn their fine-grained representations. Based on the learned representations and contextual conversations, TeCQR incorporates conversational feedback by learning to ask tag-enhanced clarifying questions to retrieve related questions more effectively. Experimental results demonstrate that our model significantly outperforms state-of-the-art baselines. 

---
# RedParrot: Accelerating NL-to-DSL for Business Analytics via Query Semantic Caching 

**Authors**: Tong Wang, Yongqin Xu, Jianfeng Zhang, Lingxi Cui, Wenqing Wei, Suzhou Chen, Huan Li, Ke Chen, Lidan Shou  

**Link**: [PDF](https://arxiv.org/pdf/2604.22758)  

**Abstract**: Recently, at Xiaohongshu, the rapid expansion of e-commerce and advertising demands real-time business analytics with high accuracy and low latency. To meet this demand, systems typically rely on converting natural language (NL) queries into Domain-Specific Languages (DSLs) to ensure semantic consistency, validation, and portability. However, existing multi-stage LLM pipelines for this NL-to-DSL task suffer from prohibitive latency, high cost, and error propagation, rendering them unsuitable for enterprise-scale deployment. In this paper, we propose RedParrot, a novel NL-to-DSL framework that accelerates inference via a semantic cache. Observing the high repetition and stable structural patterns in user queries, RedParrot bypasses the costly pipeline by matching new requests against cached "query skeletons" (normalized structural patterns) and adapting their corresponding DSLs. Our core technical contributions include (1) an offline skeleton construction strategy, (2) an online, entity-agnostic embedding model trained via contrastive learning for robust matching, and (3) a heterogeneous Retrieval-Augmented Generation (RAG) method that integrates diverse knowledge sources to handle unseen entities. Experiments on six real enterprise datasets from Xiaohongshu show RedParrot achieves an average 3.6x speedup and an 8.26% accuracy improvement. Furthermore, on new public benchmarks adapted from Spider and BIRD, it boosts accuracy by 34.8%, substantially outperforming standard in-context learning baselines. 

---
# StratRAG: A Multi-Hop Retrieval Evaluation Dataset for Retrieval-Augmented Generation Systems 

**Authors**: Aryan Patodiya  

**Link**: [PDF](https://arxiv.org/pdf/2604.22757)  

**Abstract**: We introduce StratRAG, an open-source retrieval evaluation dataset for benchmarking Retrieval-Augmented Generation (RAG) systems on multi-hop reasoning tasks under realistic, noisy document-pool conditions. Derived from HotpotQA (distractor setting), StratRAG comprises 2,200 examples across three question types -- bridge, comparison, and yes-no -- each paired with a pool of 15 candidate documents containing exactly 2 gold documents and 13 topically related distractors. We benchmark three retrieval strategies -- BM25, dense retrieval (all-MiniLM-L6-v2), and hybrid fusion -- reporting Recall@k, MRR, and NDCG@5 on the validation set. Hybrid retrieval achieves the best overall performance (Recall@2 = 0.70, MRR = 0.93), yet bridge questions remain substantially harder (Recall@2 = 0.67), motivating future work on reinforcement-learning-based retrieval policies. StratRAG is publicly available at this https URL. 

---
# Your Reviews Replicate You: LLM-Based Agents as Customer Digital Twins for Conjoint Analysis 

**Authors**: Bin Xuan, Jungmin Hwang, Hakyeon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.22756)  

**Abstract**: Conjoint analysis is a cornerstone of market research for estimating consumer preferences; however, traditional methods face persistent challenges regarding time, cost, and respondent fatigue. To address these limitations, this study proposes a framework that utilizes large language model (LLM)-based "customer digital twins (CDT)" as virtual respondents. We identified active users within the Reddit community and aggregated their comprehensive review histories to construct individualized vector databases. By integrating retrieval-augmented generation (RAG) with prompt engineering, this study developed customer agents capable of dynamically retrieving and reasoning upon their specific past preferences and constraints. These customer agents, called CDTs, performed pairwise comparison tasks on product profiles generated via fractional factorial design, and the resulting choice data was analyzed to estimate part-worth utilities by logistic regression. Empirical validation demonstrates that these CDTs predict the preferences of actual users with 87.73% accuracy. Furthermore, a case study on the computer monitor category successfully quantified trade-offs between attributes such as panel type and resolution, deriving preference structures consistent with market realities. Ultimately, this study contributes to marketing research by presenting a scalable alternative that significantly improves both agility and cost-efficiency to traditional methods. 

---
# RADIANT-LLM: an Agentic Retrieval Augmented Generation Framework for Reliable Decision Support in Safety-Critical Nuclear Engineering 

**Authors**: Zavier Ndum Ndum, Jian Tao, John Ford, Mansung Yim, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22755)  

**Abstract**: Reliable decision support in nuclear engineering requires traceable, domain-grounded knowledge retrieval, yet safety and risk analysis workflows remain hampered by fragmented documentation and hallucination when use pre-trained large language model (LLM) in specialized nuclear domains. To address these challenges, this paper presents RADIANT-LLM (Retrival-Augumented, Domain-Intelligent Agent for Nuclear Technologies using LLM), a multi-modal retrieval-augmented generation (RAG) framework designed for nuclear safety, security, and safeguards applications. The framework uses a local-first, model-agnostic architecture that pairs a multi-modal document ingestion pipeline with a structured, metadata-rich knowledge base, supporting page- and figure-level retrieval from technical documents. An agentic layer coordinates domain-specific tools, enforces citation-backed responses with provenance tracking, and supports human-in-the-loop validation to reduce hallucination risks.
To rigorously evaluate this framework, we develop and apply a suite of domain-aware metrics, including Context Precision (CoP), Hallucination Rate (HR), and Visual Recall (ViR), to expert-curated benchmarks derived from Used Nuclear Fuel Storage Facility design guidance. Across varying knowledge base sizes, CoP and ViR remain within an 85--98\% band, and hallucination rates are substantially lower than those observed in general-purpose deployments. When the same queries are posed to commercial LLM platforms without the RAG layer, hallucinations and citation errors increase markedly. These results indicate that a locally controlled, multi-modal RAG framework with domain-specific retrieval and provenance enforcement is necessary to achieve the factual accuracy, transparency, and auditability that nuclear engineering workflows demand. 

---
# How Do AI Agents Spend Your Money? Analyzing and Predicting Token Consumption in Agentic Coding Tasks 

**Authors**: Longju Bai, Zhemin Huang, Xingyao Wang, Jiao Sun, Rada Mihalcea, Erik Brynjolfsson, Alex Pentland, Jiaxin Pei  

**Link**: [PDF](https://arxiv.org/pdf/2604.22750)  

**Abstract**: The wide adoption of AI agents in complex human workflows is driving rapid growth in LLM token consumption. When agents are deployed on tasks that require a significant amount of tokens, three questions naturally arise: (1) Where do AI agents spend the tokens? (2) Which models are more token-efficient? and (3) Can agents predict their token usage before task execution? In this paper, we present the first systematic study of token consumption patterns in agentic coding tasks. We analyze trajectories from eight frontier LLMs on SWE-bench Verified and evaluate models' ability to predict their own token costs before task execution. We find that: (1) agentic tasks are uniquely expensive, consuming 1000x more tokens than code reasoning and code chat, with input tokens rather than output tokens driving the overall cost; (2) token usage is highly variable and inherently stochastic: runs on the same task can differ by up to 30x in total tokens, and higher token usage does not translate into higher accuracy; instead, accuracy often peaks at intermediate cost and saturates at higher costs; (3) models vary substantially in token efficiency: on the same tasks, Kimi-K2 and Claude-Sonnet-4.5, on average, consume over 1.5 million more tokens than GPT-5; (4) task difficulty rated by human experts only weakly aligns with actual token costs, revealing a fundamental gap between human-perceived complexity and the computational effort agents actually expend; and (5) frontier models fail to accurately predict their own token usage (with weak-to-moderate correlations, up to 0.39) and systematically underestimate real token costs. Our study offers new insights into the economics of AI agents and can inspire future research in this direction. 

---
# ECoLAD: Deployment-Oriented Evaluation for Automotive Time-Series Anomaly Detection 

**Authors**: Kadir-Kaan Özer, René Ebeling, Markus Enzweiler  

**Link**: [PDF](https://arxiv.org/pdf/2603.10926)  

**Abstract**: Time-series anomaly detectors are commonly compared on workstation-class hardware under unconstrained execution. In-vehicle monitoring, however, requires predictable latency and stable behavior under limited CPU parallelism. Accuracy-only leaderboards can therefore misrepresent which methods remain feasible under deployment-relevant constraints.
We present ECoLAD (Efficiency Compute Ladder for Anomaly Detection), a deployment-oriented evaluation protocol instantiated as an empirical study on proprietary automotive telemetry (anomaly rate ${\approx}$0.022) and complementary public benchmarks. ECoLAD applies a monotone compute-reduction ladder across heterogeneous detector families using mechanically determined, integer-only scaling rules and explicit CPU thread caps, while logging every applied configuration change. Throughput-constrained behavior is characterized by sweeping target scoring rates and reporting (i) coverage (the fraction of entities meeting the target) and (ii) the best AUC-PR achievable among measured ladder configurations satisfying the target. On constrained automotive telemetry, lightweight classical detectors sustain both coverage and detection lift above the random baseline across the full throughput sweep. Several deep methods lose feasibility before they lose accuracy. 

---
# Cloudless-Training: A Framework to Improve Efficiency of Geo-Distributed ML Training 

**Authors**: Wenting Tan, Xiao Shi1, Cunchi Lv, Xiaofang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2303.05330)  

**Abstract**: Geo-distributed ML training can benefit many emerging ML scenarios (e.g., large model training, federated learning) with multi-regional cloud resources and wide area network. However, its efficiency is limited due to 2 challenges. First, efficient elastic scheduling of multi-regional cloud resources is usually missing, affecting resource utilization and performance of training. Second, training communication on WAN is still the main overhead, easily subjected to low bandwidth and high fluctuations of WAN. In this paper, we propose a framework, Cloudless-Training, to realize efficient PS-based geo-distributed ML training in 3 aspects. First, it uses a two-layer architecture with control and physical training planes to support elastic scheduling and communication for multi-regional clouds in a serverless this http URL, it provides an elastic scheduling strategy that can deploy training workflows adaptively according to the heterogeneity of available cloud resources and distribution of pre-existing training datasets. Third, it provides 2 new synchronization strategies for training partitions among clouds, including asynchronous SGD with gradient accumulation (ASGD-GA) and inter-PS model averaging (MA). It is implemented with OpenFaaS and evaluated on Tencent Cloud. Experiments show that Cloudless-Training can support general ML training in a geo-distributed way, greatly improve resource utilization (e.g., 9.2%-24.0% training cost reduction) and synchronization efficiency (e.g., 1.7x training speedup over baseline at most) with model correctness guarantees. 

---
