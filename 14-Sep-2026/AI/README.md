# Rethinking Heterogeneous System Disaggregation for Subquadratic Attention 

**Authors**: Arya Tschand, Yaosheng Fu, Vikram Sharma Mailthody, Nicolai Oswald, Po-An Tsai, Ritchie Zhao, Oreste Villa, Vijay Janapa Reddi, Karu Sankaralingam  

**Link**: [PDF](https://arxiv.org/pdf/2609.13134)  

**Abstract**: Frontier language models are more aggressively using subquadratic attention to reduce the memory footprint and compute requirements during inference while still delivering frontier accuracy. While existing systems make dense attention-centric disaggregated serving decisions, we show that disaggregating inference around the unique arithmetic intensity and memory footprint of subquadratic attention LLMs can achieve significant throughput and energy efficiency gains on emerging DRAM-based and SRAM-only heterogeneous systems.
We introduce SQD (SubQuadratic Disaggregation), a fine-grained heterogeneous disaggregation scheme that splits decode by quadratic and subquadratic attention rather than by operator type, and that applies across subquadratic attention variants. For sparse attention LLMs, we disaggregate decode into top-k selection, which must index through the full KV, and top-k attention plus FFN, which have static memory footprints. For linear and sliding-window attention LLMs, we disaggregate decode into dense attention layers and subquadratic attention layers plus FFN. In an adjusted 8xB200 heterogeneous system proxy, we observe average tokens/J improvements of 53% on GLM 5.2, 31% on Nemotron 3 Ultra, and 56% on Gemma 4 31B over the strongest GPU-only baselines. In an analytical model of a Rubin plus LPX system with fixed power budgets, we observe 1.2x to 1.5x tighter achievable latencies and up to 3.6x higher throughput over the best baseline of attention-FFN disaggregation. Our experiments also reveal architectural insights on chip and interconnect provisioning for next-generation heterogeneous systems serving subquadratic attention. 

---
# A Hybrid LSTM-XGBoost Framework for Multi-Horizon Stock Return Prediction Across Diversified Equity Portfolios 

**Authors**: Seif ElDein Mostafa, Yahia Ahmed, Farah Datwish, Marwa Solayman  

**Link**: [PDF](https://arxiv.org/pdf/2609.13125)  

**Abstract**: Accurate prediction of equity returns remains a major challenge in computational finance due to the non-stationary, nonlinear, and low signal-to-noise ratio nature of financial time series. This paper proposes a hybrid two-stage architecture that combines a long short-term memory (LSTM) network with an XGBoost gradient-boosted regressor for multi-horizon stock return prediction across a diversified panel of 14 U.S. equities spanning six industry sectors. The LSTM component, comprising two stacked layers with 64 hidden units, processes 60-day sliding windows of five sequential market features to produce 64-dimensional temporal embeddings that encode learned sequential market dynamics. These embeddings are concatenated with 14 hand-crafted technical indicators to form a 78-dimensional hybrid feature vector, which is subsequently passed to an XGBoost regressor tuned via 3-fold cross-validation grid search. The framework is trained on a multi-stock pooled corpus using strict chronological splits and per-stock MinMaxScaling to prevent look-ahead bias, and evaluated across four prediction horizons of 30, 90, 252, and 365 trading days. Experimental results demonstrate that the hybrid model achieves a test RMSE of 0.0949 on the 30-day horizon, roughly one-third that of the standalone LSTM baseline, while marginally matching or surpassing the XGBoost-Only baseline across the majority of stocks. Directional accuracy rises with horizon length, reaching 97.6% at 365 days; we show, however, that this largely tracks the high base rate of positive long-horizon returns in the sample, and we therefore benchmark directional accuracy against a naive always-positive predictor and treat the above-base-rate gap at short horizons as the more informative signal. A composite investment scoring framework derived from multi-horizon predictions is further proposed to support portfolio ranking and decision support. 

---
# CMA-OT: Hierarchical Expert Supervision for Dance-to-Music Generation 

**Authors**: Jinting Wang, Chenxing Li, Dong Yu, Li Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.13118)  

**Abstract**: Dance-to-music (D2M) generation aims to synthesize music that is rhythmically and stylistically aligned with dance videos. A key challenge arises from the semantic mismatch between sparse dance cues, such as rhythm and style, and the dense information required for music composition, including structure, instrumentation, and expressive dynamics. Existing methods typically rely on these sparse cues and supervise only the final audio output, resulting in poorly learned music representations and generated music with limited musicality and structural coherence. To address these issues, we propose Curriculum-guided Multi-scale representation Alignment with scale-aware Optimal Transport (CMA-OT), a novel paradigm that leverages an external music expert to provide hierarchical supervision for the generator's latent features, bridging the semantic gap and enhancing representation learning. To effectively incorporate hierarchical supervision, we introduce a curriculum-guided multi-scale learning strategy that progressively transfers musical knowledge from the expert to the music generator, enabling stable and effective representation learning. Moreover, to accommodate the semantic and structural variations across different expert scales and achieve fine-grained alignment under temporal mismatch, we propose a scale-aware optimal transport alignment mechanism, which models soft correspondences between hierarchical expert representations and the generator's latent features. Extensive experiments on two datasets demonstrate that CMA-OT achieves state-of-the-art performance in rhythmic synchronization, perceptual quality, and overall music generation. 

---
# Embodied-BenchForge: A Closed-Loop Agentic Workflow for Embodied Benchmark Construction 

**Authors**: Baoyang Jiang, Fengchun Zhang, Leyuan Wang, Haotian Li, Yida Wang, Zhe Ji, Jinshan Lai, Xi Ren, Danyang Li, Zheng Yang, Jianwei Hu, Qiang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2609.13082)  

**Abstract**: Agentic systems offer a promising way to automate embodied benchmark construction, but existing approaches typically cover isolated stages or remain specialized to predefined environments and task families. More importantly, multi-step construction produces dependent intermediate artifacts that are often passed downstream without artifact-specific verification, allowing local defects to propagate into the final benchmark. We present Embodied-BenchForge, an agentic framework that transforms user-specified evaluation intents into complete embodied benchmark artifacts. It formulates construction as Closed-Loop Benchmark Synthesis, integrating forward artifact synthesis with backward verification and repair. Skill-Orchestrated Artifact Synthesis composes typed and reusable skills into executable workflows, while an artifact dependency graph records intermediate outputs and their dependencies. Requirement-Guided Verification and Repair applies artifact-specific contracts throughout construction and uses provenance to trigger local re-execution or upstream rollback when verification fails. Embodied-BenchForge constructs six benchmarks covering diverse embodied scenarios in the Offline EQA Track, together with one interactive benchmark containing 220 executable tasks in the Interactive Embodied Track. Evaluations of representative MLLMs and embodied agents show that the benchmarks distinguish model capabilities in both observation-based understanding and closed-loop execution. Quality assessment and ablations validate benchmark quality and the effectiveness of verification and repair, while repair and skill-reuse analyses demonstrate efficient localized recovery and cross-benchmark reusability. 

---
# Autonomous Research for Open-Ended Problems: A Case Study on Telecom Ticket Retrieval 

**Authors**: Junghyun Min, Huseyin Uzunalioglu, Mohamed Trabelsi  

**Link**: [PDF](https://arxiv.org/pdf/2609.13073)  

**Abstract**: Recent breakthroughs in LLM-based systems and their abilities in problem solving and coding have allowed progress in the AI for Science paradigm, potentially replacing human roles in machine learning (ML) research. However, while several frameworks of fully autonomous end-to-end ML research have been proposed, successful implementations of them are often limited to problems with narrow search spaces, like language modeling or biomedical ML benchmarks. In this paper, we explore how autonomous research can be adapted to solve open-ended, industry-grade ML problems, by considering a case study: telecom ticket retrieval, an open-ended task with degrees of freedom in representation, architecture, and training data generation. We discover that autonomous research for open-ended problems with commercial and open-source agents shows both promise and limitations: while autonomous research can excel in narrow hyperparameter optimization, it lacks human-like intuition and creativity and requires operational overhead. Even with minimal human supervision, autonomous research can reach $90\%$ of state-of-the-art performance (0.34 vs. 0.38 Recall@1) in a much shorter time period (10 weeks vs. 10 months of human work) at a modest cost (up to \$200 per Cursor campaign). Our empirical evidence recommends that human researchers and autonomous research frameworks work together for best results in ML research. 

---
# Anchoring Clinical Events in Time: UID-Preserving Multimodal Reconstruction and Source-Grounded Adjudication 

**Authors**: Sayantan Kumar, Nicolas Grimaldi, Jack Cummins, Jeremy C. Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2609.13062)  

**Abstract**: Clinical timelines support treatment-window analysis and leakage-free modeling, but discharge summaries often obscure chronology and structured EHR tables describe only part of the patient course. We present a UID-preserving framework that links each narrative event occurrence to its source span and retains that identity through text-only estimation, structured-evidence retrieval, timestamped source-row grounding, and joint revision. We also present GAVEL, an LLM judge that compares two UID-aligned timelines against the narrative and structured record, to augment prior matching and temporal assessments. Across six open-weight models and 40 mixed-critical-care summaries, the GLM 5.2 multimodal revision, as compared to its text-only variant, improved temporal agreement without reducing event recovery and performed competitively with clinician annotations, while other model revisions showed smaller gains and lower overall performance. Ablations showed that UIDs primarily preserve event retention, whereas source-row linkage supports temporal placement. Blinded human review upheld most GAVEL findings, and controlled adjudication favored multimodal over text-only GLM 5.2 but did not for DeepSeek V3.2. In developing the UID and judge pipeline, we are able to demonstrate 43\% increased event recovery, a framework competitive with clinician annotations, and a system with occurrence-level provenance for both reconstruction and evaluation. 

---
# Diffusion Models and Concept Formation 

**Authors**: Zekun Wang, Karthik Singaravadivelan, Christopher J. MacLellan  

**Link**: [PDF](https://arxiv.org/pdf/2609.13047)  

**Abstract**: Humans organize knowledge into a taxonomy of concepts with nested levels of abstraction and a \emph{basic level} at which people recognize and name objects with the least cognitive effort. Cobweb is a classic cognitive account of this ability, an incremental learner that builds a probabilistic concept hierarchy by maximizing category utility. We argue that diffusion models, although designed for image synthesis, implicitly perform the same computation. The noisy marginals of a diffusion model are Gaussian smoothings of the data distribution, and the modes of these marginals form a hierarchy that corresponds to a Cobweb tree of probabilistic prototypes in four respects. Both are hierarchical density models, both are hierarchical-Bayesian models with Gaussian prototypes, both treat categorization as score-following that reduces uncertainty, and in both a basic level emerges. We locate this basic level for a diffusion model at an intermediate noise level, where recent analyses show that the reverse process commits to the class identity of a sample. The two models differ mainly in how they represent and learn the taxonomy. Cobweb learns a discrete tree incrementally, whereas a diffusion model encodes a continuous, interpolable hierarchy in a single learned score field fit to the data distribution. We test the correspondence on MNIST and Fashion-MNIST by recovering the diffusion hierarchy through mode-finding and comparing the basic levels of the two models. This reframes diffusion as a cognitive model of concept formation and offers Cobweb a continuous, scalable instantiation. 

---
# How Good Are Frontier Models at Physics? Expert Re-Grading Reveals Broken Evaluations and Near-Saturation of Leading Benchmarks 

**Authors**: Ali Ansari, Haoran Sun, Andy Zeyi Liu, Mark Jabbour, Yongshan Ding, Steven Girvin, Yu He, Sohrab Ismail-Beigi, Aleksander Kubica, Owen D. Miller, Corey O'Hern, Vidvuds Ozolins, David Poland, A. Douglas Stone, Frank C. van den Bosch, Logan Wright, Navid Akbari, Santanu Antu, Kangle Cai, Andrew Calabrese-Day, Mateo Cárdenes Wuttig, Meng Cheng, Barry T. Chiang, Ali Ghorashi, Shouzhen Gu, Haoyang Huang, Zhibo Kang, Lukas Kienesberger, Hantian Liu, Charles Lomba, Zhongling Lu, Wenchao Ma, Rohin E. McIntosh, Evan McKinney, Ivan Rojkov, Xulei Sun, Yarone Meir Tokayer, Naveen Balaji Umasankar, Mira Varma, Leda Wang, Qimin Wang, Tyler Wang, Haoyu Wei, Jinming Yang, Jinchen Zhao, Sherlock Tingrui Zhao, Qinyuan Zheng, Jay S. Zou, Lucas Baker, Arman Cohan, John Sous  

**Link**: [PDF](https://arxiv.org/pdf/2609.13009)  

**Abstract**: Low reported scores on leading physics benchmarks, including those featured in the Artificial Analysis Intelligence Index (2026), suggest that frontier language models still struggle with advanced physics, a demanding test of their scientific reasoning and quantitative problem-solving abilities. Yet this impression does not always align with domain experts' experiences using these models in their work. We revisit these reported findings by evaluating frontier models on six widely used physics benchmarks and auditing them with experts, focusing on text-only problems with verifiable final answers. For each subfield of physics, faculty and graduate researchers with relevant expertise carefully review problem statements, reference solutions, and model responses to distinguish genuine model errors from grader errors, incorrect reference solutions, and ambiguous or underspecified questions. Most audited cases initially evaluated as incorrect reflect these benchmarking issues rather than errors in the models' physics reasoning. We then ask experts to address these benchmarking issues by correcting erroneous reference solutions and repairing or excluding flawed questions. We find that GPT-5.6-Sol's measured mean@4 rises from 47.3% to 78.7% on HLE-Physics and from 61.0% to 87.2% on CMT-Benchmark, while its corrected pass@4 reaches 94.4% on the 54 retained CritPt challenges. Corrected scores are computed on the retained evaluation subsets following expert review. Scores on the audited subsets of UGPhysics, PRISM-Physics, and PHYBench also rise substantially after correction. These findings suggest that current benchmarks substantially understate frontier models' ability to solve well-posed physics problems. Near-saturation on these closed-ended tasks highlights the need for more demanding, expert-validated evaluations. 

---
# EduFair-Bench: Evaluating Pedagogical Fairness of LLM Tutors Across Student Demographics 

**Authors**: Jiaxu Zhao, Bahar Radmehr, Fares Fawzi, Tanya Nazaretsky, Tanja Käser  

**Link**: [PDF](https://arxiv.org/pdf/2609.12949)  

**Abstract**: Large language models (LLMs) are increasingly deployed as tutors, but it is unclear whether they support all students equally well. We introduce \textbf{EduFair-Bench}, a benchmark for auditing the pedagogical fairness of LLM tutors---whether tutoring quality varies systematically with student demographics. EduFair-Bench pairs a multi-domain question bank (mathematics, physics, chemistry) with a controlled simulation in which a fixed LLM student interacts with each tutor across nine demographic levels spanning four dimensions: gender, immigration background, first language, and socioeconomic status (SES). Tutoring quality is scored on five turn-level pedagogical metrics and four conversation-level dimensions, using an LLM judge validated against three-annotator consensus on 180 tutor turns. Bias is measured via paired Wilcoxon signed-rank tests and bootstrap effect-size confidence intervals. Two ablations (demographic cues conveyed through names; conflicting demographic information between tutor and student) disentangle tutor-driven from student-driven bias. Across five tutors, we find that model capability and demographic fairness are largely orthogonal: the smallest model is the most consistent while the four more capable tutors all exhibit wide demographic gaps with no clear capability-to-fairness ordering, pedagogy-specific RL training redistributes rather than removes bias, and language- and immigration-related cues produce larger gaps than gender- and SES-related cues. 

---
# Tracing and Coordinating Cross-Layer Influence for Multimodal Model Merging 

**Authors**: Pengyang Zhou, Xiaobin Tu, Zhengxi Liu, Rongkun Xue, Haochen Li, Miancan Liu, Ziyuan Chen, Yinggui Wang, Jinkui Ren, Xiantao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.12897)  

**Abstract**: Multimodal model merging aims to consolidate task experts into a single model that retains their complementary capabilities. Most unimodal model merging methods combine expert updates within individual layers, and multimodal approaches largely follow this design. However, an expert update changes the representations passed to subsequent layers, allowing its influence to propagate across depth and affect how visual and textual information interact. When visual and language updates are combined, later updates act on inputs already modified by earlier ones, coupling their effects. This poses two challenges: (1) how to characterize the multimodal influence of individual expert updates across depth, and (2) how to jointly combine expert updates based on their multimodal influence. To address these challenges, we propose TAC-Merge for tracing and coordinating cross-layer influence in multimodal model merging. It contains two modules, i.e., multimodal influence mapping (MIM) and coupled merge control (CMC). MIM constructs graphs of update effects and uses Ricci curvature together with expert predictions to define a shared fusion objective. CMC models interactions among coefficient adjustments and jointly optimizes regional weights to synthesize one shared model. Experiments across diverse multimodal tasks demonstrate the effectiveness of TAC-Merge in consolidating complementary expert capabilities and supporting generalization to unseen tasks. 

---
# MedRoundsQA: A Persona and Difficulty Aware Evaluation for Multi-Turn Medical Consultations 

**Authors**: Youssef Mohamed, Ahmed Heakl, Qinrong Cui, Junhong Liang, Rafiq Ali, Bdour Babillie, Nazira Dunbayeva, Lang Gao, Omar Hussein, Ahmed Nada, Ahmed Mohamed Magdy Mohamed, Jinghui Liu, Salman Khan, Imran Razzak, Yuxia Wang, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.12851)  

**Abstract**: Medical benchmarks are dominated by single-turn, multiple-choice clinical cases that poorly reflect real consultations. Practically, clinicians elicit evidence interactively and patient communication varies widely. We introduce MedRoundsQA, a multi-turn diagnostic benchmark derived from 1,387 board-exam cases across 17 specialties. Each case is converted into a structured 24-slot clinical record, and then instantiated as controlled doctor-patient dual-agent dialogues under varying patient personas, with the underlying clinical content held fixed. We further classify cases by difficulty using model-based uncertainty to enable easy-to-hard analysis. Evaluations of fifteen LLM doctor agents show that (i) moving from a single-turn diagnosis on the standardized records to multi-turn consultations causes large degradations of roughly 13-39 points; (ii) more turns reliably improves question relevance, but diagnostic accuracy exhibits diminishing returns and typically plateaus after 6-12 turns; and (iii) patient persona differences can shift diagnosis accuracy by about 7-8 points (lowest to highest education), highlighting equity risks that single-turn benchmarks miss. 

---
# Scaling Clinical Judgment to Evaluate Medical AI 

**Authors**: Thomas A. Buckley, Zahir Kanjee, Peter G. Brodeur, Byron Crowe, Anthony M. Pettinato, Aashna P. Shah, Adrian D. Haimovich, Liam G. McCoy, Daniel Restrepo, Ethan Goh, Jonathan H. Chen, Laura Zwaan, Katherine E. Goodman, Daniel J. Morgan, Raja-Elie E. Abdulnour, Adam Rodman, Arjun K. Manrai  

**Link**: [PDF](https://arxiv.org/pdf/2609.12822)  

**Abstract**: Blinded physician evaluation has been considered by many to be the gold standard for assessing clinical reasoning in large language models (LLMs). This is difficult to scale; thus, prior studies typically rely on small physician panels, often from a single institution or specialty, which both limits the scientific questions investigated and makes it unclear whether findings would be reproduced with a different set of evaluators. To more rigorously and scalably study clinical reasoning in AI models, here we introduce PrecepTron, an LLM fine-tuned for physician-level evaluation of open-ended responses. PrecepTron was trained using low-rank adaptation (LoRA) of a 32-billion-parameter model on a small number of physician examples. We also release GRAND-ROUNDS, a new large-scale physician-annotated benchmark of 9,217 scores by 11 physicians across seven studies. We show that frontier LLMs in typical "LLM-as-a-judge" approaches often disagree with physicians and with each other, but fine-tuning PrecepTron on a small number of cases enables physician-level consistent scoring across tasks. We use PrecepTron to reproduce headline findings from five influential studies assessing LLMs for clinical care in JAMA, Science, and Nature Medicine without new human grading. Using PrecepTron, we then pose new questions about how LLMs reason in medicine that would have been infeasible with human grading alone, including measuring the diagnostic accuracy of frontier LLMs when clinical cases are provided piecemeal, even token by token. Together, PrecepTron and GRAND-ROUNDS provide a foundation for reproducible, large-scale study of how LLMs reason in medicine. All code, data, and labels are made freely available for researchers. 

---
# K-Bench: A Benchmark for LLM Unlearning in Agentic Deployments 

**Authors**: Guangsheng Yu, Yanna Jiang, Qin Wang, Baihe Ma, Xu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.12808)  

**Abstract**: Unlearning benchmarks such as TOFU and MUSE certify forgetting by reading the model's final answer, where a model that refuses to answer already counts as having forgotten. We show that this model-level certificate does not transfer once the model is deployed as an agent. We introduce K-Bench, a benchmark that scores LLM unlearning under agentic deployment. K-Bench inspects all six channels a ReAct agent exposes, including its chain-of-thought (CoT), tool calls and tool observations, and elicited summary. A query counts as leaked if the secret appears in any of them. Each experiment places the secret in exactly one of the agent's three sources (the weights, the prompt, or the retrieval store). The K-Score is computed separately for each source and credits forgetting only when the agent remains usable. Clearing the answer channel does not make the secret unrecoverable. On structured retrieval, the secret stays verbatim in the tool-observation channel and the aggregate leak rate is unchanged. When the secret lives in the prompt or the retrieval store, TOFU and MUSE report no leakage, while the deployed agent still leaks it on 22--86\% of queries. When the secret is in the weights, none of the twenty evaluated published methods demonstrably removes it, and only an input-corruption intervention reaches selective forgetting under the evaluated observer. The top-ranked method changes across base models. A refusal-tuning method resists the evaluated extraction without verified knowledge removal. 

---
# Interpreting the predictions of neural network classification based on a Taylor Coefficient Analysis (TCA) 

**Authors**: Markus Klute, Artur Monsch, Lars Sowa, Roger Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2609.12801)  

**Abstract**: We introduce a rigid and comprehensive taxonomy and paradigm for characterizing the influence of the input feature space $X$ on the predictions $\hat{y}$ of a neural network (NN) used for event classification, based on a Taylor expansion of $\hat{y}$ in $X$. The complete process of introspection we refer to as Taylor Coefficient Analysis (TCA). Based on two simplistic example tasks, which can be easily understood and bencmarked, we illustrate the power of the TCA when it comes to revealing, what properties of $X$ have led to what value of $\hat{y}$, of a given NN model, building up intuition for the method. A more complex application is meant to represent $X$ of a typical classification task at a CERN LHC experiment. Based on this application, we play through the different levels of introspection that the TCA offers and discuss a number of practical aspects for a TCA application typical for the analysis of CERN LHC data. We conclude with a study to support the assumption that those properties of $X$ most relevant for tasks of the complexity typical for a CERN LHC experiment, are usually caught by a TCA up to the second order. 

---
# MPT: Missing Prototype Tracking via Barycentric Reconstruction in Vehicular Federated Learning 

**Authors**: Hanju Jang, Gyeongmin Han, Sungmin Lee, Kichang Lee, Chunghan Lee, JeongGil Ko  

**Link**: [PDF](https://arxiv.org/pdf/2609.12771)  

**Abstract**: Cross-vehicle federated learning enables vehicles to collaboratively improve perception models while keeping locally collected driving data private. However, vehicle participation is transient, and a vehicle may depart before training converges while permanently taking its local data. When this departing vehicle holds most samples of a target class, the class becomes rare in the remaining FL network, and its recognition can silently degrade as the shared backbone continues to evolve. Recovering the class is difficult since the few remaining samples provide a noisy prototype estimate, while FL privacy constraints prevent centralized access to raw data or per-sample features. This paper presents MPT, a cross-vehicle FL framework that maintains rare-class recognition by reconstructing its prototype at every round from privacy-preserving class-level statistics. MPT combines a barycentric decomposition that tracks drift shared with remaining-class prototypes, a covariance-based residual prediction that estimates out-of-span drift, and an adaptive calibration that weighs the remaining rare-class samples according to their reliability. We evaluate MPT on three vehicle classification tasks and four backbones against representative calibration and drift-compensation baselines. MPT outperforms all baselines in rare class F1, reaching 0.516 on the nuImages dataset with only 1\% of rare-class samples remaining, without raw data, per-sample features, or retraining. 

---
# Unified Agentic Video Editing Across Levels of Complexity and Creativity 

**Authors**: Surabhi S. Nath, Kim Ferres, Milan Petrović, Lion Schulz  

**Link**: [PDF](https://arxiv.org/pdf/2609.12769)  

**Abstract**: Editing is a core component of video production, requiring creative planning and decisions under multiple constraints. Here, we report methods for agentic tooling for automated video editing across three tasks varying in editorial goal, complexity and creativity, namely scene previews, video summaries and cinematic trailers. We evaluate the outputs and discuss implications for automation and agency. 

---
# SCQ: Stabilizing Conservative Q-Learning with Sigmoid-Bounded Entropy 

**Authors**: Xiefeng Wu, Shu Zhang, Zhaojie Chu, Mingyu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12749)  

**Abstract**: Offline-to-online reinforcement learning reduces interaction cost for real-world robot learning but suffers from persistent value estimation instability. Existing methods address this through pessimistic regularization, lower-bound calibration, and architectural normalization, but an overlooked source of instability lies in the entropy formulation: the standard log-entropy term can become negative, destabilizing policy updates. We introduce SCQ (Sigmoid-Bounded Conservative Q-Learning), which replaces this term with a sigmoid-bounded formulation that stays strictly positive. SCQ retains conservative Q regularization and return-based lower-bound calibration, stabilizing policy optimization without sacrificing exploration. We evaluate SCQ on D4RL (Minari) benchmarks under both single-demonstration and standard dataset settings, as well as on simulation and real-world visual tasks. SCQ matches or exceeds baseline performance while exhibiting more stable training dynamics across state-based and visual benchmarks, and transfers to four real-robot platforms including manipulation, wheeled, quadruped, and humanoid systems. A direct clipping intervention that removes negative log-probability contributions, together with gradient-matched positive-score controls, indicates that positivity rather than a particular score shape alone drives much of the improvement. Project website: this https URL. 

---
# Assisted Spatial Cognition Through Vision-Language Models 

**Authors**: H. Riaz, J. B. Fernandez, I. Mills, D. Hickey, F. Cleary, M. I. Ali  

**Link**: [PDF](https://arxiv.org/pdf/2609.12747)  

**Abstract**: Multimodal AI, powered by Large Language Models (LLMs) and Vision-Language Models (VLMs), is transforming assistive technologies by enabling simultaneous processing of visual and textual data. This advancement holds significant promise for over 43 million visually impaired and neuro-divergent individuals worldwide who face persistent challenges in navigating indoor and outdoor environments due to limited spatial awareness and insufficient environmental cues. Existing navigation aids often lack comprehensive 3D scene understanding, relying on constrained route-based strategies that hinder user autonomy. In this paper, we introduce a novel end-to-end framework that integrates LLMs, VLMs and digital twin technologies to deliver a spatially cognitive navigation support for visually impaired and neuro-divergent users. Our system captures video input via standard mobile phone cameras, and employs SLAM3R to generate dense 3D point clouds from monocular RGB sequences in real-time. Our custom post-processing algorithm ensures accurate point cloud alignment across multiple viewpoints without requiring predefined reference points. This enhances the capabilities of SpatialLM to produce structured 3D representations, including architectural elements and oriented object bounding boxes. The enriched spatial data is then processed by a locally deployed LLM, which interprets 3D contexts to generate detailed scene descriptions and precise distance measurements between users and surrounding objects. We evaluated our approach across diverse video scenarios featuring various perspectives, looped walking views and captured in multiple environments. The evaluation results demonstrate consistent accuracy in 3D scene interpretation and object localisation, underscoring the potential of our system as a transformative assistive navigation solution that combines advanced visual perception with spatial reasoning 

---
# What Drives Recovery in Agentic Text-to-Cypher? LAST-CQ: An LLM Agent Self-Refinement Framework 

**Authors**: Ioannis Prokopiou, Athanasios Aidinis, Panagiotis-Christos Kyrmpatsos, Pantelis Vikatos  

**Link**: [PDF](https://arxiv.org/pdf/2609.12746)  

**Abstract**: Agentic pipelines for structured-query generation are rapidly expanding, but it is unclear which part of the loop produces the gain. We use LAST-CQ -- a five-agent, training-free, execution-grounded Text-to-Cypher framework -- as an instrumented testbed, running three counterfactuals over 2,471 live-database queries and six backbones spanning three vendor scale tiers. Removing correction is worth between 3.1% aggregate execution-BLEU against the single-pass system and 12.3% against a no-refinement counterfactual (up to 80.7% for the weakest backbone). Replacing schema-grounded, LLM-synthesised feedback with raw database error strings costs almost nothing (20.9% vs. 19.9% naive exact match; <0.2% end-to-end; equivalent within $\pm 0.075$ set-F1 by two one-sided tests). Spending the same call budget on parallel sampling degrades quality by 10-11%. What works is detecting failure and routing it to a retry, not the feedback sophistication or number of samples. LAST-CQ itself recovers 91.7% of queries that fail under single-pass generation, while a query that succeeds first time still costs exactly one LLM call. We also show that n-gram overlap on serialised results is not a bound in either direction: it over-scores against set equivalence on 65.9% of results while under-scoring against judged semantics. Finally, we calibrate our LLM judge against blind human labels and find it optimistic by 9 points. 

---
# Skill Issue: Lessons from Optimizing Repository SKILLs for Coding Agents 

**Authors**: Mykhailo Kozyrev, Andrei Kozyrev, Anton Podkopaev  

**Link**: [PDF](https://arxiv.org/pdf/2609.12742)  

**Abstract**: Coding agents increasingly read repository knowledge from SKILLs --- plain \texttt{.md} files versioned alongside the code. Recent work synthesizes these files automatically, by optimizing the document against a benchmark. A bare repository comes with no benchmark, and the synthetic tasks prior work builds are small enough that a capable agent saturates them with no document at all. We mine harder tasks --- merged pull requests of the repository, reverted at a single frozen base commit; and score a candidate document by whether the same agent does better with it than without it. On three Kotlin repositories, the documents GEPA finds raise this score by $4.9$pp on average, and the ones SkillOpt finds leave it where it started, $0.1$pp above the seed. The GEPA gain matches what prior work reports with the same optimizer, and at the dataset size a single repository supplies it cannot be separated from the agent's run-to-run variance; settling that would take more tasks than one repository's history yields. The documents themselves read better than the score: a maintainer of one repository found in them knowledge one only gets by working in the project. 

---
# When Rubrics Fail: Hallucinations Reveal Blind Spots in Medical AI Evaluation 

**Authors**: Griffin Farrow, Lily Sijia Li, Jack Johnson, Tingyan Wang, Philip Torr, William Bolton, Fabio J. Fehr  

**Link**: [PDF](https://arxiv.org/pdf/2609.12718)  

**Abstract**: Hallucinations can undermine clinician trust in LLMs, making it important that evaluation methods capture clinically relevant errors. Rubric-based evaluation has become the leading approach for assessing LLMs in medicine, but it is unclear whether rubric scores reflect such errors. We first study this in a controlled setting using MedHallu, finding that more specific rubrics better distinguish correct from hallucinated responses. To test this systematically, we develop a taxonomy of medical hallucination types and a clinician-validated error-injection pipeline that creates matched correct and error-injected responses. Across HealthBench, HealthBench Professional, and LiveMedBench, our clinically relevant hallucinations are missed by rubrics, often leaving scores unchanged. We find that rubrics are most effective when explicitly checking facts, and are less effective for additional or unexpected errors they do not anticipate. A preliminary retrieval-based factuality check recovers some of the rubric-blind errors, suggesting a complementary approach. These findings reveal systematic blind spots in current medical evaluation of LLMs and suggest that rubric scores alone are insufficient to establish clinical reliability, potentially undermining clinician trust and confidence in clinical deployment. 

---
# Implicit Personality Representations in Humans and LLMs 

**Authors**: Yilin Geng, Omri Abend, Eduard Hovy, Lea Frermann  

**Link**: [PDF](https://arxiv.org/pdf/2609.12704)  

**Abstract**: A century of psychology has found that the trait words people use to describe one another vary, but the relational structure among those traits, which ones go together and which oppose, is strikingly consistent across raters and cultures. We test whether the LLM (Qwen 2.5-7B-Instruct) reproduces this structure in its internal trait representations. From millions of crowd-sourced personality ratings of fictional characters, we build a human implicit-personality matrix over hundreds of traits; from contrastive model activations, we build a matching matrix over the same traits. The two relational structures align strongly (Mantel r = 0.77), and the agreement holds trait by trait as well as in aggregate. Two dominant axes of the model's trait representations recover the social and intellectual dimensions long known to organize human personality impressions, social warmth and intellectual competence. On held-out dialogue, projecting model activations onto these directions yields personality profiles that agree with human ratings. This work establishes a framework that enables comprehensive, human-grounded comparison between internal model trait geometry and the shared structure of human personality impressions. 

---
# Enabling and Understanding Personalization in AI-Generated Advertising Imagery 

**Authors**: Victor Kolominsky-Rabas, Leopold Müller, Claudius Budcke, Claas Christian Germelmann, Niklas Kühl  

**Link**: [PDF](https://arxiv.org/pdf/2609.12697)  

**Abstract**: Personalized marketing traditionally matches static products to customers, while dynamic creative optimization focuses mainly on AI-driven text personalization or basic product image modifications. We address this gap by developing and implementing an AI-based framework that generates personalized advertising imagery directly from customer data. We evaluate this framework in a two-stage within-subject study with N=100 participants across four products and three levels of personalization, varied by the amount and specificity of customer data used. Participants rated each image on attitude toward the advertisement, attitude toward the product, and purchase intention. Results show that participants perceive differences across personalization levels and evaluate AI-generated advertising imagery most positively at a moderate level of personalization. High personalization increases perceived personalization, which is positively associated with all three outcome measures, but also increases perceived creepiness, which is negatively associated with the outcomes and dominates the total effect. 

---
# I Am AdMan: A Pipeline for Automatic Generation of Personalized Advertising Imagery 

**Authors**: Victor Kolominsky-Rabas, Leopold Müller, Claudius Budcke, Niklas Kühl  

**Link**: [PDF](https://arxiv.org/pdf/2609.12694)  

**Abstract**: Personalized marketing can increase customer engagement, satisfaction, and conversion. While existing personalization approaches have become effective at matching the right product to the right customer, the visual representation of advertisements remains generic and only weakly tailored to the individual. Prior research shows that generative artificial intelligence can improve the creation of personalized advertisements, particularly for text, and that image generation models can support scalable advertisement production. However, little research has examined how detailed customer information can be systematically translated into fully AI-generated, personalized advertising imagery at scale on a technical level. To address this gap, we propose AdMan, a multi-agent pipeline that transforms customer data into personas, generates personalized advertisement images conditioned on product reference images, and applies an LLM-based judge agent for automated quality control. We implement the pipeline with two different model configurations and evaluate it across four products, using six celebrity personas for qualitative inspection, and 100 real customer profiles, producing 1745 advertisements. The evaluation combines a qualitative expert focus group and a quantitative artifact-rate assessment. The results show that the pipeline can generate photorealistic and personalized advertisements. At the same time, performance varies substantially by product complexity and model configuration. Our findings extend the literature on AI-based personalized advertising by demonstrating the feasibility and current limitations of fully automated image generation for advertising. 

---
# Residual Vector-based Reconstruction as Long-Context Recall Regardless of Context Window Size 

**Authors**: MyungHoon Ryu, XinYu Piao, Jong-Kook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.12686)  

**Abstract**: Large language models (LLMs) process long contexts, including long documents and lengthy conversations, but face token-level memory usage that increases proportionally to input length. Although model optimization and lossy prompt compression are widely used, these methods still fail to solve the long-context recall problem beyond pretrained and size-constrained context windows. This paper proposes a long-context recall method that maintains near-constant GPU memory usage as context length increases, without additional training. The main idea is to reconstruct facts using parameter activations in the LLM's feed-forward layers, which store residual vectors representing facts from the source document. Utilizing residual vectors allows the LLM to deterministically reconstruct query relevant facts without referencing the original document, preserving high fidelity and reducing memory usage without fine-tuning weights. Experimental results show that the proposed method enables answering single-fact questions in two-million-token story contexts where previous methods fail. 

---
# Generative AI Use Cases In Real Estate Marketing: Adoption and Constraints in Germany 

**Authors**: Victor Kolominsky-Rabas, Leopold Müller, Felicia Perpina, Niklas Kühl  

**Link**: [PDF](https://arxiv.org/pdf/2609.12684)  

**Abstract**: Generative artificial intelligence (GenAI) is changing how work is organized and performed. Real estate marketing is a prime example of this, yet evidence of GenAI in real estate agents' day-to-day practice remains scarce. In this work, we report on our insights from a German-based empirical study with eleven semi-structured interviews. GenAI is already utilized across different activities, with marketing communication being the most prominent. Concrete use cases are emergent and unevenly adopted, with writing exposé texts being the only widely established one. Interaction is predominantly human-in-the-loop: GenAI drafts, structures, and retrieves, while real estate agents curate, verify, and decide. Constraints stem less from model capability than from integration with listings and documents, data availability, and compliance in sensitive tasks. The study contributes a grounded map of existing and potential use cases and identifies tentative practical implications for adoption. 

---
# SteerDuplex: Steerable Duplex Speech Dialogue Models 

**Authors**: Utkarsh Tyagi, Ramaneswaran Selvakumar, Advait Gosai, Sonal Kumar, Nikhil Barhate, Isabell Sagar, Steven Li, Miheer Bavare, Daniel Quigley, Fabiola Tapia Carrillo, Jose M Patron E, Diego Macías Gutiérrez, Paul Song, Ramani Duraiswami, Dinesh Manocha, Yunzhong He  

**Link**: [PDF](https://arxiv.org/pdf/2609.12623)  

**Abstract**: Full-duplex spoken dialogue models support low-latency turn taking, interruption handling, and backchanneling, yet a key capability remains underexplored: steerability, the ability to reliably shift conversational behavior along attributes such as tone, persona, speaking rate, and voice style in response to user instructions. We introduce a taxonomy of text- and audio-based steerability that identifies substantial gaps in current full-duplex models. To address this gap, we introduce SteerDuplex, a Moshi-based full-duplex speech model fine-tuned on natural conversations and synthetic dialogues targeting instruction following, vocal delivery, reasoning, and duplex interaction. We further apply two-stage reinforcement learning (RL) with hybrid rewards, combining verifiable interaction checks and judge-based semantic feedback to improve timing and response continuity. To evaluate full-duplex spoken steerability, we introduce SteerBench, a benchmark with 390 spoken prompts and 1,067 human-authored binary audio and text rubrics spanning tone, persona, style/accent, and speed/length. On SteerBench, supervised training improves audio-steering average pass rate by 44.5 percentage points over the strongest evaluated open baseline. On Audio MultiChallenge, task average pass rate improves by 7 points over its strongest evaluated open baseline. RL further raises source-clean interruption response from 72.5% to 82.5% and reduces synthetic pause barge-in from 26.5% to 9%. Steering and aggregate task scores remain comparable or higher, while reward probes reveal reward hacking through incomplete responses. Our model and benchmark support systematic research on spoken steerability, with reward analysis showing why timing gains must be evaluated alongside response completeness. 

---
# Beyond Generation and Accuracy: Diagnosing and Enhancing Visual Chain-of-Thought for Geometry Problem Solving 

**Authors**: Zhitong Dong, Jicai Pan, Yingguo Gao, Jingting Ding, Hao Chen, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12606)  

**Abstract**: While multimodal reasoning has advanced rapidly, solving complex geometry problems critically hinges on active visual assistance, such as constructing auxiliary lines, spurring the rise of Visual Chain-of-Thought (VCoT). However, existing evaluations typically assess visual generation quality and final answer accuracy in isolation, failing to examine whether intermediate visual aids are geometrically valid, effectively utilized in subsequent reasoning, or causally responsible for task success. To bridge this gap, we introduce GeoVAD-Bench, a diagnostic benchmark that pairs a fine-grained five-dimensional trajectory diagnosis covering perception, auxiliary quality, utilization, deductive reasoning, and final correctness with controlled No-Aux, Auto-Aux, and GT-Aux intervention settings to systematically isolate intermediate error modes, the causal gains of visual aids, and the resulting autonomy gap. Our findings reveal that while high-quality auxiliary aids offer substantial theoretical gains for geometric problem solving, autonomous generation is frequently hampered by compounding errors across geometric perception, faithful visual manipulation, visual-state grounding, and deductive reasoning. Guided by these diagnostic insights, we establish a specialized data construction pipeline encompassing geometric perception, diagram editing, and interleaved visual-textual reasoning trajectories, and develop a progressive SFT and multimodal RL training framework. The resulting model, GeoWeave-8B, outperforms the base model by +25.3% in final geometric accuracy and achieves a +30.4% gain in process average across the four intermediate diagnostic dimensions. 

---
# Reproducing and Evaluating the Generalizability of Subliminal Learning in Open-Weight Models 

**Authors**: Daan van der Weijden, Nathan Brack, Selene Baez Santamaria  

**Link**: [PDF](https://arxiv.org/pdf/2609.12586)  

**Abstract**: In this reproduction paper we investigate subliminal learning, a consequence of distillation where teacher models transmit behavioral preference traits through semantically unrelated data. The original paper explores two types of traits (animal preferences and misalignment), three data modalities (number sequences, code, and chain of thought), and several model families. We reproduce their experiments and extend the setup along three axes: new preference categories (actors and politicians), a new task (chess move generation), and an additional open-weight model (Ministral8B). We also run a controlled ablation on the numbers task's answer-space size (1-, 2-, and 3-digit sequences). We focus on open-weight models with accessible checkpoints on HuggingFace, since the original paper's GPT-4.x fine-tuning is no longer available. Our reproduction supports the original paper's claims, but our extensions show they are not universal as transmission strength varies across traits and tasks, and one model shows almost no effect at all. 

---
# From Collaboration to Capability: Internalizing Routed LLM Experts into Compact Reasoners 

**Authors**: Frank Nie, Shuyao Wang, Ethan B. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12578)  

**Abstract**: A compact controller can coordinate stronger experts by selecting whom to consult, formulating requests, and integrating their responses. We study whether learning from both the controller's decisions and the experts' reasoning and code improves its generation after expert removal. We introduce \textsc{Rivet} for \emph{collaboration internalization}: expert-augmented reinforcement learning applies a shared outcome signal to controller decisions and returned expert spans, and verified trajectory internalization consolidates complete successful interactions through format-aware supervised training. The deployed controller generates reasoning, code, and interaction structure with local Python execution and no external LLM. Across seven competition-mathematics benchmarks, RIVET-1.7B and RIVET-4B achieve average accuracies of $28.25\%$ and $44.16\%$; Stage~II improves RIVET-4B's accuracy after expert removal by $6.49$ points, and GPQA-Diamond results provide evidence of generalization to scientific reasoning. Ablations show gains from ordinary trajectory supervision and additional format weighting, supporting the effectiveness of training on the content and structure of verified collaborations. 

---
# Information Specialization and Constrained Synthesis in Multi-Agent LLM Forecasting: A Prospective Live-Study of the 2026 FIFA World Cup 

**Authors**: Julian Varghese, Lucas Bickmann, Sarah Sandmann  

**Link**: [PDF](https://arxiv.org/pdf/2609.12495)  

**Abstract**: Large language models are being organized into multi-agent systems with specialized roles, but whether such specialization produces distinct forecasts and whether subsequent synthesis improves utility remains unclear. In this study, we carried out a live, prospective evaluation over the final 56 matches of the information-dense 2026 FIFA World Cup, keeping a frontier foundation model constant while assigning two primary forecasting agents contrasting specialist roles: a quantitative specialist focusing on structured performance statistics and a news specialist focusing on current injuries, tactics and information from press conferences. Their forecasts were then reviewed by a separate critic before being combined by a meta-agent, resulting in a sequential four-agent model. Forecasts from the betting market served as an external benchmark. The news specialist obtained the highest mean probability-weighted Top-3 utility and matched the betting market in Top-3 exact-score hits. Nevertheless, the two specialist forecasters agreed on at least two of the three scorelines in 50 out of 56 matches, and the meta-agent never generated more than one scoreline outside the specialists' forecast set. These findings show that rapidly changing, unstructured information can provide a valuable forecasting signal alongside structured statistics, whereas adding critic and meta-agent stages does not necessarily create complementary information or improve on the strongest specialist. 

---
# Confidence-Gated Transductive Test Generation for Code Reranking 

**Authors**: Sungjae Lee, Youngsik Yoon, Seockbean Song, Siwei Wang, Wei Chen, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2609.12489)  

**Abstract**: Test case synthesis is crucial for evaluating and ranking programs generated by large language models (LLMs). However, constructing high-quality test cases remains challenging because reliable expected outputs are often difficult to obtain. We propose Confidence-Gated Transductive Test Generation (CoTT), which first uses an efficient inductive procedure and invokes transductive generation only when inductive confidence is low. This adaptive design improves output reliability while allocating extra computation only when needed. On code reranking benchmarks, CoTT outperforms prior baselines across the reported metrics while reducing cost relative to applying transductive generation to every input. These results show that confidence-based allocation of test-time computation provides a favorable efficiency-effectiveness trade-off with a single efficient LLM. 

---
# When Does AI Augment Work? A Workflow-Level Framework for Human-Agent Collaboration 

**Authors**: CIVIC-AI Collaboration, Jiaying Wu, Caleb Ziems, Raymond Chan, Nancy F. Chen, Corlyss Chua, Gerard Chung, Jungpil Hahn, Wee Sun Lee, Zhengyuan Liu, Jamie Ng, Desmond C. Ong, Jeryl Ong, Da Ren Soon, Tianqi Song, Zhi-Xuan Tan, Sixing Tao, Emily Yang, Yajing Yang, Stella Xin Yin, Min-Yen Kan, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.12482)  

**Abstract**: We aim to characterise the value of artificial intelligence in the workplace. Current studies largely measure this value in terms of the current automation capabilities and public adoption of AI. However, such metrics ignore the greater impacts of human--agent collaboration in transforming the nature of work. To account for this, we must expand the scope of our analysis beyond atomised tasks of today, and instead focus on how AI can augment entire workflows of the future. To ground this analysis, we establish a precise definition of AI augmentation comprising six conditions, spanning durable net value, meaningful human control, accountability and recovery, and long-term human development through learning, career pathways, and job purpose. We elaborate on these conditions and apply the framework in a case study of AI-mediated social surveys. We conclude by outlining how organisations, researchers, and government leaders can use this framework to make sense of the future of work. 

---
# TripPattern: A Pattern-based Text Watermarking Method for Large Language Models 

**Authors**: Sangjun Moon, Dasom Choi, Jingun Kwon, Hidetaka Kamigaito, Taro Watanabe, Manabu Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2609.12472)  

**Abstract**: Text watermarking techniques have gained significant attention for identifying machine-generated text and mitigating risks from large language models (LLMs). Existing methods typically divide an LLM's vocabulary into green and red tokens, but encouraging generation toward green tokens can reduce text quality and naturalness. To address this, we propose TripPattern, a watermarking framework that formulates text watermarking as a pattern-based matching task using three vocabulary partitions. TripPattern divides the vocabulary into one neutral group and two pattern groups. During generation, the model alternates token selection between the two pattern groups to embed detectable patterns, while neutral tokens are selected independently to improve flexibility and preserve naturalness. For detection, TripPattern uses pattern-based statistical tests that provide interpretable p-values by measuring how often adjacent tokens alternate between the pattern groups. Theoretical analysis and empirical evaluations on four multilingual datasets show that TripPattern maintains LLM generation quality while achieving robust watermark detectability. 

---
# Beyond Vector Similarity: Hierarchical Context-Aware Graph RAG vs Standard RAG in Enterprise Code Migration 

**Authors**: Nilesh Jaiswal, Aniket Agrawal, Arjit Shukla, Divya Malhotra, Saurabh Garg, Suchit Puri, Suddhasatwa Bhaumik  

**Link**: [PDF](https://arxiv.org/pdf/2609.12464)  

**Abstract**: As enterprises modernize legacy monolithic systems to microservices, Large Language Models (LLMs) are heavily utilized for automated code translation. However, traditional vector-based Retrieval-Augmented Generation (Standard RAG) struggles to capture topological relationships. It fetches isolated chunks that sever inheritance chains, leading to high compilation failure rates. This paper introduces a Hierarchical Context-Resident Graph (HCRG) methodology to resolve these limitations. Our pipeline uses tree-sitter for Abstract Syntax Tree (AST) extraction, maps architectural edges into a Google Cloud Spanner Property Graph, and serializes this structure into a Gemini Context Cache for topological, parent-first code translation. We shift evaluation from naive text-overlap to a custom 7-metric Software Engineering framework. Traditional metrics like CodeBLEU (which scored 91% for both methods) effectively masked Standard RAG's structural failures behind syntactically plausible but broken code. Empirically, Graph RAG decisively mitigates dependency loss: API hallucination rates dropped from 56.4% to 16.2%, Dependency Resolution Quality improved from 34.8% to 65.9%, and Parent-Child Consistency rose from 26.7% to 45.5%. However, Graph RAG introduces specific trade-offs. The dense global context causes defensive over-engineering by the LLM, reducing Cyclomatic Complexity Consistency from 71.6% to 46.7%, and slightly degrades Docstring Preservation (67.0% to 61.0%). Ultimately, while trading code complexity for reduced hallucinations, Graph RAG provides a substantially more viable, architecturally sound path for automated enterprise codebase modernization. 

---
# EvoRS: On-Policy Self-Evolution of Reward Systems for Open-Ended Reinforcement Learning 

**Authors**: Weiyuan Li, Aili Chen, Xintao Wang, Yikai Zhang, Qingqing Dong, Jinghan Xu, Hongru Hou, Wenxuan Zhao, Chengkun Lang, Jun Gao, Yuanli Guo, Hongcheng Guo, Yanghua Xiao, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.12459)  

**Abstract**: Open-ended reinforcement learning often relies on rubric-based rewards for tasks without directly verifiable answers. Yet the policy and reward system form a dynamic feedback loop: as the policy optimizes the current reward, an initially useful reward system may become unreliable due to reward hacking or reduced response discriminability. The reward system should therefore evolve rather than remain fixed during training. Existing dynamic-rubric methods adapt evaluation criteria, but reward failures can also arise from scoring mechanisms or signal composition. We introduce EvoRS, a self-evolving RL framework that evolves the reward system from on-policy experience, representing it as an executable Reward-DAG. Specifically, an agentic designer updates this system from on-policy rollouts and reward traces to maintain train-time reliability. Across writing and roleplay, EvoRS achieves the best quality under all three judges, outperforming the policy by \(2.107\) and \(4.767\) points, respectively, while reducing reward hacking and coverage failures and preserving reward informativeness. Ablations confirm that a comprehensive fixed reward system cannot remain reliable in open-ended tasks and must evolve throughout training. 

---
# Do LLMs Trust the Accuser or the Accusation? Measuring Belief Shifts in Werewolf 

**Authors**: Yu-Yu Yang, Ti-Rong Wu, Hung Guei, Hsing-Yu Chen, I-Chen Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12446)  

**Abstract**: Social-deduction games such as Werewolf are increasingly used to evaluate LLM agents, but existing evaluations often rely on final game outcomes. We propose a belief-shift evaluation benchmark in Werewolf for analyzing communication skills through belief updating. Using LLM-played games, we annotate suspicion and accusation messages and measure how an observing village-side model's beliefs change after each message. We evaluate 40 open-weight LLM configurations on 1,224 annotated messages. Our results show that larger models better distinguish true wolves from villagers based on game history, but accusations still strongly influence their beliefs. Models become more suspicious of the accused target and less suspicious of the accuser, especially when the accuser is trusted, even if the accuser is wolf-aligned. Larger models better resist accusations from accusers they already distrust. Overall, our findings suggest that current open-weight LLMs up to 120B parameters still struggle to integrate accusation content with source trust in strategic communication. Our benchmark and code are available at this https URL. 

---
# LifeFuse-Mem: Lifecycle-Aware State Fusion Against Temporary Overwriting for Long-Term Memory 

**Authors**: Hanyu Zhao, Yuqian Feng, Zhenyu Song, Yuanchao Cheng, Yance Jiao, Tengfei Pan, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2609.12436)  

**Abstract**: Long-running LLM agents require memory mechanisms that maintain coherent internal states across interactions. We study a lifecycle-labeled memory setting in which write episodes provide lifecycle metadata during training, and phase-aware readout is used during evaluation. This setting reflects the need to distinguish information that should remain influential across future interactions from information that should affect only the current context. A mismatch between these lifecycles can cause temporary information to overwrite durable knowledge, leading to behavioral drift in persistent agents. Within this setting, we introduce \textbf{LifeFuse-Mem}, a lifecycle-aware neural memory framework that separates information according to its temporal commitment. LifeFuse-Mem uses dedicated memory components and lifecycle-aware updates to allow stable and transient knowledge to evolve locally without converting temporary context into durable state. On the controlled anti-overwrite benchmark, LifeFuse-Mem improves acquisition-controlled retention and reduces temporary overwrite; on two public long-memory benchmarks, it remains broadly competitive. These results suggest that explicit lifecycle signals can help diagnose and mitigate overwrite in compact online memory. 

---
# Hierarchical Belief Modeling for Zero-Shot Opponent Adaptation in Partially Observable Multi-Agent Navigation 

**Authors**: Kowei Shih, Lu Cheng, Zeyu Wang, Yeyun Xu, Kejian Tong  

**Link**: [PDF](https://arxiv.org/pdf/2609.12422)  

**Abstract**: Lux AI Season 3 requires agents to act under partial observability, randomized episode level dynamics, and a best of five match structure that rewards both tactical execution and fast adaptation. We present HORIZON, a hierarchical agent that combines symmetry aware spatial perception, dual memory belief tracking, relic centric graph attention, information gain driven exploration, and an opponent conditioned policy mixture. HORIZON separates short horizon control from cross match meta reasoning, while auxiliary belief and world model objectives stabilize learning. Trained with PPO in a large scale JAX simulator, the resulting agent explicitly infers hidden game parameters and opponent style. Experiments show consistent gains in match win rate, episode win rate, adaptation gain, and league rating over strong recurrent and feed forward baselines. 

---
# SoK: Rethinking Jailbreaking in the Era of Agentic AI: Attacks, Defenses, and Practical Consideration 

**Authors**: Md Jueal Mia, Yanzhao Wu, Selcuk Uluagac, M. Hadi Amini  

**Link**: [PDF](https://arxiv.org/pdf/2609.12413)  

**Abstract**: Large language models (LLMs) are rapidly evolving from conversational assistants into agentic AI systems that reason, plan, invoke tools, maintain persistent memory, communicate with other agents, and execute multi-step tasks. At the same time, modern models exhibit substantially stronger native safety alignment than earlier generations on which many jailbreak attacks and defenses were originally studied. This shift raises a fundamental question: \textit{which established jailbreak-security findings remain valid in the era of modern LLMs and agentic AI?} We address this question through a Systematization of Knowledge (SoK) that reframes jailbreak security around the full agentic execution pipeline. We develop unified taxonomies of attacks and defenses spanning user interaction, planning and reasoning, memory, tool use, and inter-agent communication, and introduce a security--utility--efficiency evaluation framework that separates native harmful-prompt safety, adversarial jailbreak robustness, and agent-level security outcomes. We further conduct a controlled empirical study of representative attacks and defenses within a common agentic framework. Our results reveal three important gaps. First, strong native alignment does not imply robustness to adversarial jailbreaks. Second, defense effectiveness is highly model-, attack-, and component-dependent and can come at substantial cost in over-refusal, utility, and latency. Third, low final-response attack success can mask severe intermediate compromise: planning, memory, and tool interactions may remain unsafe even when the final response is successfully filtered. These findings motivate a shift from response-centric jailbreak defense toward cross-layer, execution-aware security that protects agent state, component transitions, and external actions while preserving practical utility and efficiency. 

---
# VRL-Bench: Benchmarking agents on computer control tasks under finite trial budgets 

**Authors**: Yu Bai, Yukai Miao, Dawei Wang, Li Chen, Yanyu Ren, Yuqian Shi, Dan Li, Ying Xiong, Chengqiu Tan, Run Zhou, Li Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.12404)  

**Abstract**: Learning from trial and error is a promising way to improve language agents on complex tasks such as computer control. Reflexion introduced verbal reinforcement learning, which turns failed trials into text that guides later attempts without updating model parameters. We introduce VRL-Bench, a harness for fair evaluation of trial-and-error learning under finite trial budgets. Across three models on MiniWoB and WebShop, we evaluate updates from several prominent verbal-memory methods spanning Reflexion and later work: each improves observed success over memory-free retry in some settings but reduces it in others. Replay experiments show that using reflection can reduce success rates, revealing a trade-off between exploiting experience and continued exploration. We propose VEX$^2$, a verbal exploration--exploitation scheduler that uses a language model to jointly select policies and allocate the remaining trial budget. VEX$^2$ is the only evaluated update to achieve positive observed success-rate gains over retry in all six settings. 

---
# Beyond ID Embeddings: Process-Grounded Language Modeling for Cognitive Diagnosis 

**Authors**: Minghang Liu, Yuanzhuo Wang, Qiang Qiu, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.12403)  

**Abstract**: Cognitive Diagnosis Models (CDMs) play a pivotal role in personalized online learning. Traditional CDMs rely on discrete, ID-based embeddings to represent students, exercises, and concepts. This paradigm diverges from the nature of learner cognition, where knowledge is not stored and retrieved as isolated symbols. As a result, CDMs suffer from semantic limitations when new exercises or concepts appear. In this paper, we propose a Process-aware Language Cognitive Diagnosis (PLCD) framework that uses language-derived structures as cognitive priors and response records to calibrate student posterior states. PLCD leverages large language models (LLMs) to construct concept schemas and cognitive process graphs, and uses target-conditioned semantic memory to retrieve historical responses that are relevant to each target exercise. A process-grounded Language-to-Cognition Mapper with DA-MoE experts and process-level contrastive learning then maps the textual evidence into a unified cognitive space. Experimental results show that PLCD not only outperforms traditional baselines in predicting student performance but also exhibits strong cognitive transfer capabilities. These results connect the computational power of LLMs with the psychometric goal of measuring latent knowledge states, suggesting that structured language priors calibrated by response records can improve cold-start robustness and cognitive grounding. 

---
# Decentralized Evolution of Hexapod Gaits with Independent Leg Controllers 

**Authors**: Gary B. Parker, John Asaro, Jim O'Connor  

**Link**: [PDF](https://arxiv.org/pdf/2609.12400)  

**Abstract**: This paper presents a novel approach to hexapod locomotion by evolving each leg's gait independently through a decentralized evolutionary algorithm. Using the Webots simulator and the Mantis hexapod robot, we optimize individual leg controllers without centralized coordination, allowing emergent behaviors to drive the development of efficient, coordinated locomotion. Our decentralized method is benchmarked against cooperative coevolution, demonstrating improved efficacy in generating stable and adaptive gaits while showing interesting emergent coordination. By enabling independent evolution of leg controllers, this method reduces the complexity of gait optimization and highlights the potential of decentralized strategies for scalable and adaptive robotic systems. 

---
# OneLA: Scaling Linear-Attention Decoding to Large Beams in Generative Recommendation 

**Authors**: Xiangrui Yang, Cheng Peng, Yunfeng Zhao, Liang Zeng, Ao Hu, Jiawei Yang, Shengzhe Wang, Jingshan Lv, Xiao Liang, Chen Yang, Jiaqiang Liu, Yiming Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12399)  

**Abstract**: Generative recommendation (GR) relies on large-beam decoding to generate hundreds of candidate items, creating a new scaling challenge for recurrent linear attention. Existing linear attention serving systems either materialize a full recurrent state for every beam or repeatedly replay shared history, incurring substantial memory and traffic overhead. To address this, we present OneLA, a linear-attention decoding framework that exploits the shared prompt and short divergent suffixes of GR workloads. Specifically, OneLA represents all beam states using a single shared prompt-derived state and compact, append-only records of their divergent transitions. Using this representation, OneLA computes only the state information required at each decoding step, without reconstructing a full recurrent state for every beam. Furthermore, OneLA uses a lightweight ancestry index to track the transition records that make up each beam's history, allowing beams to be updated without moving or copying existing records. A fused GPU kernel further reuses the shared state across beams. Our analysis shows that OneLA achieves 1.54-2.46x end-to-end decode speedups while substantially reducing recurrent-state memory use and data movement. 

---
# Niching Agents in The Core 

**Authors**: Gary B. Parker, Jim O'Connor, John Asaro  

**Link**: [PDF](https://arxiv.org/pdf/2609.12398)  

**Abstract**: The Core is a unique competitive co-evolution algorithm that allows agents to evolve autonomous control without utilizing a traditional fitness function. The agents evolve via local interactions through tournament selection, crossover, and mutation, producing offspring by evolving better controllers. Previous works have shown The Core's ability to evolve agents capable of combat and navigation in the Xpilot video game. This research expands upon that premise by niching agents to specific subsets of the original environment The Core was tested in. Our results demonstrate the niched agents capacity for success over agents niched to the entire system and agents niched to different sub-environments. 

---
# Is Gaussian Splatting Becoming Neural Again? A Taxonomy and Controlled Study of Learned Parameterization 

**Authors**: YuanHang Wang, Xin Cao, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.12395)  

**Abstract**: Three-dimensional Gaussian Splatting (3DGS) combines explicit primitives with efficient rasterization, yet recent systems increasingly use neural networks to generate or share Gaussian parameters. We characterize this trend along five axes: attribute decoding, spatial sharing, view-conditioned decoding, topology generation, and amortized inference. An analysis of 19 representative methods shows that these choices address different limitations and cannot be reduced to a binary neural label. We also isolate three forms of neural parameterization in a controlled mip-NeRF 360 study. Sharing appearance and opacity improves reconstruction quality, while decoding geometric structure offers no further gain. The evidence favors selective neuralization: shared functions help when they capture reusable correlations without sacrificing the local geometric freedom of explicit splats. 

---
# BlueLM-GUI Technical Report: A Real-Device-Centric Flywheel for Self-Improving Mobile GUI Agents 

**Authors**: Tong Ye, Kunyang Han, Guozhi Wang, Longqiang Luo, Zhifeng Ding, Yongxiang Zhang, Xiaolei Shen, Yuxuan Zhang, Zhuping Zhang, Tao Xu, Yue Pan, Yucheng Zhao, Yupei Hu, Yuanjiang Ouyang, Danfeng Shen, Runqi Lin, Hongda Cai, Zhaoxiong Wang, Mengjia Yan, Yingjie Zhong, Chen Zhou, Zeyu Zhang, Xuwen Zhu, Penggang Shi, Mingcheng Luo, Ziyang Wu, Min Jin, Mingfu Shen, Zairong Xu, Fan Zhang, Hao Wang, Liang Liu, Zhulin Xie, Lijun Yao, Xiao Liang, Liangmin Wen, Liqiang Feng, Feilong Wu, Min Hu, Min Chen, Guanjing Xiong, Xiaohu Ruan, Xiaoxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.12394)  

**Abstract**: Mobile GUI agents are shifting from multi-module frameworks to native models trained end-to-end, yet industrial deployment faces three persistent gaps. Sandbox training produces a distribution mismatch with production environments; expensive real-device failures remain underutilized; and fixed benchmarks saturate, losing the power to guide iteration. We present BlueLM-GUI, a 35B-A3B mobile GUI agent built as a real-device-centric flywheel that closes these gaps through three principles. Every Sample Matters: a dual-track pipeline with Heterogeneous Triple-System Consensus evaluation and an Error Correction \& Derivation Module salvages every trajectory into usable supervision. Every Rollout Is Real: a three-stage recipe---continual pre-training, supervised fine-tuning, and agentic reinforcement learning on hundreds of real phones---grounds every rollout in real production environments, so the capability the model learns transfers directly to deployment. Every Query Evolves: a quota-driven benchmark methodology with three orthogonal axes enables precise attribution and allows the benchmark to be systematically upgraded as the model improves. BlueLM-GUI achieves 87.4 on MobileGUI-VBench, surpassing the best closed-source model by 5.1 points, and 84.9 on AndroidWorld, the best result among open-source models and competitive with closed-source models. These results demonstrate that grounding model training and iterative improvement in both real devices and the three Every principles yields strong, robust, and transferable mobile GUI capability. 

---
# Toward Robust Personalized Alignment for LLMs: Mitigating Persona Drift in Multi-Turn Dialogue 

**Authors**: Youyuan Zhang, Siyuan Li, Fangming Liu, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.12373)  

**Abstract**: Persona drift remains a central challenge for personalized language models, as user profiles evolve over long interactions rather than remain permanently fixed. Models must therefore revise persistent persona states when preferences genuinely change, while avoiding updates driven by transient, ambiguous, or unresolved observations. We propose CORE, which separates turn-local evidence from persistent persona-state revision and selectively updates grounded user preferences through uncertainty-aware belief revision. We also introduce PERSIST, a held-out post-anchor benchmark for persona-state robustness under sequential interaction stress, covering ambiguity, conflict, and controlled social influence. Across ALOE, PersonaChat, and PERSIST, CORE improves personalized alignment and robustness, with complementary gains in normalized closed-slot state fidelity. Human evaluation and mechanistic controls further support explicit update control beyond stronger generation or persistent memory alone. 

---
# LoRA-RC: Reservoir Computing with Low-Rank Adaptation 

**Authors**: Wenbin Wan  

**Link**: [PDF](https://arxiv.org/pdf/2609.12327)  

**Abstract**: Reservoir computing (RC) trains only a linear readout over a fixed recurrent layer, making it fast and data-efficient for online prediction. However, a static reservoir degrades under system drift, readout-only adaptation is then insufficient, and unconstrained reservoir adaptation can destroy the echo-state and incremental stability properties that make RC reliable. This paper proposes LoRA-RC, which adapts the recurrent matrix through a low-rank correction driven by streaming prediction errors. The base reservoir and adaptation bases are fixed offline; a small core matrix is adapted online, projected onto a spectral-norm ball, and low-pass filtered at each step. The projection guarantees that every applied recurrent matrix remains within a certified contraction set, and an incremental input-to-state stability bound is established for the reservoir along each online adaptation path, with path-independent rate and gain. On a Lorenz system with an abrupt parameter drift, LoRA-RC cuts post-drift prediction error by 56% versus a fixed RC and 51% versus readout-only adaptation; ablations over 20 seeds show that removing the projection inflates this error by more than a factor of 40. 

---
# Affective Agent: On-Device Personalized Intervention Reasoning for Wearable Systems 

**Authors**: Reina Mun, Zishen Wan, Vijay Janapa Reddi  

**Link**: [PDF](https://arxiv.org/pdf/2609.12322)  

**Abstract**: Affective computing has advanced wearable state inference, but on-device reasoning about whether, when, and how to intervene remains challenging. We present Affective Agent, a three-layer reference architecture for personalized intervention reasoning under uncertainty on wearable-class hardware. It combines a compact sub-billion-parameter language model with physiological evidence, context, and user history to decide whether, when, and how to intervene, without cloud dependency or per-user retraining. The architecture is organized into three interacting layers (perception, personalization, and reasoning), adapting to individual users through host-managed structured memory evolution rather than per-user weight updates. We instantiate Affective Agent in indoor environmental quality control and evaluate it on held-out, simulator-generated longitudinal scenarios spanning physiological variation, context, signal quality, and intervention history. Results show that memory-driven personalization and two-pass structured reasoning improve intervention decisions within this synthetic evaluation. By moving the decision layer on-device, this work demonstrates a path from wearable state inference toward closed-loop, personalized intervention on wearable-class hardware. 

---
# AIM: A Privacy-Aware Interoperable Memory Framework for Multi-Agent Multi-User LLM Systems 

**Authors**: Zachary Johnson, Nigel Boachie Kumankumah, Somya Chatterjee, Tejas Sathyamurthi, Min Chen, Xinyi Alice Li, Xiao Wang, Emily Morgan Gelchie, Jessica Lin, Sadid A. Hasan, Sulaiman Vesal  

**Link**: [PDF](https://arxiv.org/pdf/2609.12320)  

**Abstract**: Traditional large language models (LLMs) are scoped to individual user sessions, limiting their knowledge to a single conversation and preventing them from learning user preferences that evolve over time. Existing agentic memory systems address this limitation but generally operate at the individual-user level, restricting the public knowledge that could be shared across users to improve downstream responses. We introduce AIM (Agentic Interoperable Memory), a unified, privacy-aware memory framework that enables multi-agent, multi-user LLM systems to persistently manage private and shared memory. AIM dynamically classifies information as private, scoped to one user and inaccessible to others, or public, accessible to all users. It enforces index-level access controls so that private memories are retrievable only by their owner, protecting sensitive data while allowing beneficial shared knowledge to improve coordination and consistency. We also introduce MUMBench (Multi-User Memory Benchmark), a dataset of multi-user interactions containing private and shareable information across four domains. To our knowledge, MUMBench is the first public dataset designed to evaluate multiple memory operations, including retrieval, creation, update, and deletion, in a multi-user environment. Across three independent runs on MUMBench, AIM achieves 96.0% visibility classification accuracy, 58.8% strict operation accuracy, and 70.5% state-aware operation accuracy. 

---
# Do Influence-Derived Data Perturbations Enable Machine Unlearning? A Controlled Study of Three Plausible Roles 

**Authors**: Chenkai Wu, Chrispine Kambimbi, Qinyang Zeng, Jun Yan  

**Link**: [PDF](https://arxiv.org/pdf/2609.12313)  

**Abstract**: We evaluate Deep Perturbation Learning (DPL), which perturbs training images and labels along influence-derived directions, in three roles in which prior work has positioned it for machine unlearning: a direct deletion signal (the strongest claim), a utility-preserving regularizer, and a warm start for adversarial unlearning. Evidence for the weaker roles has been used to support the stronger one, so we test each role separately under a matched protocol with exact-seed retraining baselines. An audit of the public implementation identifies two correctness issues: image directions are computed on augmented, normalized tensors but applied to raw images, and the label perturbation falls below float32 resolution, leaving labels unchanged. After correcting the image-perturbation pipeline, DPL fails the direct-deletion criterion on CIFAR-10/ResNet-18 in all three paired seeds. Its utility effects are inconsistent in sign across seeds, and once direction-computation time is counted it underperforms simple warm-start baselines. A one-seed Tiny ImageNet check likewise does not favor DPL as a regularizer or warm start; preprocessing inconsistencies in the released code make the direct comparison there inconclusive. These results cover random instance deletion only and do not rule out influence-based methods in other deletion regimes. We release a role-matched evaluation protocol and an audit checklist for perturbation-based deletion claims. 

---
# Hybrid Physics-AI Framework of Body Center of Mass Dynamics from Wrist-Worn Sensors 

**Authors**: Shuhao Que, Valentina Breschi, Ying Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.12304)  

**Abstract**: Wrist-worn IMU has been widely used for daily-life health monitoring. Yet, it does not fully represent whole-body dynamics, for which the body center of mass (COM) is considered the physiological reference standard. Therefore, this work proposes a simplified kinematic model (KM), which is designed to map the wrist IMU to the COM acceleration. It is built upon several reductive assumptions that enable the solvability of the dynamic equations based on wrist IMU measurements alone. This work further proposes three types of hybrid AI modeling methods, namely human kinematic model-based neural network (HKM-NN) models, to leverage the power of both grey-box and black-box modeling. The HKM-NN methods include serial learning (ser-) and two approaches of simultaneous learning (sim1- and sim2-). The proposed models are trained and tested using our dataset, which includes wrist IMU measurements and ground-truth COM measurements from 10 healthy volunteers during six gait activities and sit-to-stand (SS) transitional movement. The results demonstrate the feasibility of estimating COM acceleration from wrist IMU measurements. Our KM model yields satisfactory results, with an error ranging from 6.7% to 12.5% for gait activities and 5.6% for the SS. In comparison with the KM model, our HKM-NN models significantly enhance the performance, achieving 5.3% to 9.3% errors for gait activities, and the best error of 3.9% for the SS. In addition, the HKM-NN models demonstrate distinct robustness characteristics under noisy test conditions, with sim1-/sim2- generally maintaining greater robustness under Gaussian perturbations, while the KM model exhibits comparatively strong robustness under salt-and-pepper noise. These findings highlight the importance of combining biomechanical structure with data-driven learning for wearable sensing applications operating under imperfect and noisy measurement conditions. 

---
# Robust Prototypical Networks for Few-Shot Sensor Fault Diagnosis 

**Authors**: Mohammed Ayalew Belay, Amirshayan Haghipour, Pierluigi Salvo Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2609.12287)  

**Abstract**: Industrial fault diagnosis often operates with only a handful of labeled fault examples, making few-shot learning attractive for sensor monitoring. Standard prototypical networks are simple and effective; however, their class prototypes may become unstable in the very-low-shot regime because each decision relies on a small support set. We propose \emph{Multi-Episode Prototypical Networks} (MEPN), which aggregate prototypes from multiple disjoint support episodes and use their mean as the final class representative, reducing prototype variance without changing the encoder architecture. We evaluate MEPN on the DeFACTO sensor dataset using five-way fault classification with synthetic bias, drift, spike, and noise faults injected into real industrial measurements. Over 100 independent runs, MEPN reaches \textbf{\SensorOneShotGcpn\%} in the per-episode one-shot setting ($K\!=\!1$ shot, aggregated over $N_{\text{agg}}\!=\!10$ support episodes), substantially above single-episode baselines. Under an equal 10-sample support budget, MEPN and ProtoNet at $K\!=\!10$ are statistically indistinguishable, confirming prototype accumulation as the mechanism rather than superior fixed-budget learning. 

---
# T-GADE: Thermodynamical Generative-AI-Driven Evolution of LLM Artifacts 

**Authors**: Kyoko Ogawa, Naoki Mori  

**Link**: [PDF](https://arxiv.org/pdf/2609.12286)  

**Abstract**: Integrating evolutionary computation and large language models (LLMs) requires control of population diversity as well as generative capability. Among LLM outputs, those with explicit structure, such as a description paired with code, are structured artifacts; we use artifact for short. We propose T-GADE, which evolves these artifacts by extending thermodynamical genetic algorithms through LLM-based genetic operators and artifact-level diversity evaluation. A common free-energy objective supports generational and steady-state updates, with Fermi-type occupancy excluding repeated genotypes and Bose-type occupancy permitting them. We establish exact one-member removal and conditions for recovering the zero-temperature survival rule of Evolution of Heuristics (EoH). On the online bin-packing task studied in the EoH paper, excess measures relative bin-count overhead above a volume lower bound. Training excess uses search instances; transfer excess uses instances with another bin capacity. Generational Bose-type T-GADE at $T=0.003$ reduced median training excess by approximately 29%, from 1.152% to 0.815%, over 20 runs per configuration (two-sided Mann-Whitney $p=0.042$, Cliff's $\delta=0.378$). Validation selection among its two highest-ranked final candidates reached the same median transfer excess as EoH, 0.496%. These results demonstrate the utility of thermodynamical selection and validation-based use of retained artifacts. 

---
# Learning Symbolic Constraint Representations from Examples: A Neuro-Symbolic Approach 

**Authors**: Nassim Belmecheri, Arnaud Gotlieb, Nadjib Lazaar, Helge Spieker  

**Link**: [PDF](https://arxiv.org/pdf/2609.12267)  

**Abstract**: Learning user-defined concepts as constraint networks has been extensively studied in the constraint acquisition (CA) literature. However, existing approaches typically rely on intensive interactions with a human oracle, making the learning process costly in terms of time and number of queries. In this paper, we propose a neuro-symbolic framework for automatic CA that significantly reduces user involvement by introducing neural Oracle Transformer models which learn to emulate user responses and to generalize conceptual knowledge. Trained on previously available examples, the learned oracle interacts with a dedicated CA engine, FastCA, which systematically refines the oracle's responses into a sound, consistent, and interpretable constraint network. This neuro-symbolic interaction enables the recovery of structured symbolic models from data without prior domain knowledge. Our results demonstrate that this neuro-symbolic interplay effectively aligns data-driven pattern recognition with symbolic reasoning, offering a robust approach to automating model construction in combinatorial domains. 

---
# GTA: Graph Theory Agent and Benchmark for Algorithmic Graph Reasoning with LLMs 

**Authors**: Zixiang Xu, Yanbo Wang, Chenxi Wang, Lang Gao, Zirui Song, Yue Huang, Zhaorun Chen, Xiangliang Zhang, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.12265)  

**Abstract**: Large Language Models (LLMs) are increasingly asked to reason over structured data such as graphs, yet how reliably they can carry out multi-step graph algorithms in language remains unclear. Existing evaluations tend to use simple tasks on small graphs, to score code generation rather than reasoning over the graph itself, or to fix a single input format. We introduce Graph Theory Bench (GT Bench), a benchmark covering 24 classical graph problems in 44 task-structure settings, with over 100,000 examples across four representations: natural language, structured language, adjacency list, and adjacency matrix. Evaluating eight LLMs on GT Bench shows that accuracy is strongly tied to the input representation, that the best representation shifts with graph density, size, and topology as well as with the model, and that this sensitivity persists, attenuated, in the strongest reasoning models. Building on these observations, we propose the Graph Theory Agent (GTA), which pairs a preference-trained representation selector with plan-and-decompose scaffolding around a frozen executor LLM. GTA lifts Phi-4 from 53.5% to 69.1% on the benchmark's easy split and from 33.0% to 41.5% on its hard split, outperforming eight prompting and agent baselines, and transfers without retraining to GraCoRe and NLGraph. Code for benchmark generation and evaluation: this https URL. The project homepage is available at this https URL. 

---
# Soft Symbol Grounding for Prototypical Concepts 

**Authors**: Marcos Galván-López, Nijesh Upreti, Hiram Calvo, Carlos Aguilar-Ibáñez, Vaishak Belle  

**Link**: [PDF](https://arxiv.org/pdf/2609.12247)  

**Abstract**: Neuro-symbolic models are usually trained with supervision only on final labels, leaving the intermediate concepts unobserved. Since many concept assignments are consistent with a given label, training can predict labels correctly while recovering the wrong concepts, a failure known as a reasoning shortcut. Prototypical networks reduce shortcuts by anchoring each concept to a few labeled examples, but existing methods still couple perception and reasoning through a hand-crafted, task-specific differentiable loss that must be redesigned for every task. We introduce \textbf{Soft-PNet}, which removes this loss: it reframes concept grounding as a Metropolis walk over a precomputed cache of feasible symbolic solutions, guided by a prototype distribution built from a single labeled anchor per concept, and trains against one KL objective between the prototype-weighted cache and the network's concept predictions. The objective is identical across tasks and remains applicable when the solution space cannot be enumerated. On \texttt{MNIST-EvenOdd}, Visual Sudoku, and \texttt{Kand-Logic} under scarce supervision, Soft-PNet matches loss-engineered prototypical networks at the concept and label levels and recovers concepts that soft-grounding baselines miss, with no loss engineering and lower training time. 

---
# WinSyn: An Automated Pipeline for Realistic Enterprise Question-Answering Evaluation 

**Authors**: Amey Varhade, Ananya Sutradhar, Ravishankar Krishnaswamy, Navin Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2609.12171)  

**Abstract**: Enterprise settings provide a challenging environment for question-answering agents, which often rely on Retrieval-Augmented Generation, Deep Research (DR), and related techniques. Much of this challenge comes from the complexity of enterprise data: information is often spread across evolving and potentially conflict- ing emails, chat messages, documents, and other artifacts. Existing benchmarks typically have limited real-world complexity, short-form responses, and unnatural queries, so they often fail to capture the challenges of enterprise settings. In this work, we introduce an automated pipeline for generating synthetic datasets of emails reflecting realistic workplace scenarios, along with long- and short-form questions and gold answers grounded in the data. Our method simulates long-running enterprise projects spanning several months and involving up to 25 interacting employees across multiple roles. The data emphasizes ambiguity, distributed information, and naturally occurring queries. To validate the pipeline, we evaluate few standard agentic baselines on our datasets using the latest frontier models. We find that aggregate scores averaged over all queries remain below 80% for each dataset, indicating significant room for improvement. These findings suggest that more work remains to be done for enterprise deployment and underscore the importance of realistic, high-complexity evaluation data for developing stronger real-world enterprise DR systems. 

---
# GLARE: Generative Learning via Adversarial Reward Estimation For Social Dynamics Forecasting 

**Authors**: Tenghao Huang, Zhaoxuan Tan, Muhao Chen, Jonathan May, Mengting Wan, Longqi Yang, Pei Zhou, Sihao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.12165)  

**Abstract**: Meeting continuation requires tracking the agenda, speaker roles, participant intentions, and disagreement across long multi-party discussions. We introduce the Meeting Dynamic Forecasting Benchmark (MDFB), constructed from 2,207 real-world meetings and 24,794 future-facing queries. Given a transcript prefix and an active question, a model generates a plausible multi-turn continuation in one call. We evaluate utility---progress toward the question---and human-likeness---plausible conversational flow and role consistency---without requiring exact reproduction of the observed future. We further present GLARE, an adaptation of adversarial imitation learning to conditional language generation. A discriminator ranks the observed continuation above samples from the current actor, and its score supplies a KL-regularized policy reward; retraining on current-policy negatives allows the reward landscape to evolve with the actor. GLARE attains average human-evaluated win rates of 0.66 on utility and 0.70 on human-likeness, outperforming SFT and SPIN while remaining below the observed human continuation. We also demonstrate MDFB as a social reasoning arena for comparing general-purpose models, including closed-source systems, through reference-assisted judgments. Together, these studies illustrate the benchmark's use for both task-specific learning and output-based evaluation of meeting behavior. 

---
# Can LLMs in Draft-Verify-Revise Pipelines Resolve Deictic Ambiguity? 

**Authors**: Obinna I. Ekekezie  

**Link**: [PDF](https://arxiv.org/pdf/2609.12162)  

**Abstract**: Draft-verify-revise is a common LLM orchestration pattern for scaling inference-time compute. One LLM drafts, a second critiques the draft and provides feedback, and a third uses that feedback to revise the draft into the final output. As context cascades between stages, LLMs at different stages can resolve a context-dependent expression such as "previous" differently. When that happens, the expression undergoes a deictic shift, a change in what it refers to. This phenomenon was studied with a synthetic dataset of 10 base examples, each rendered in three conditions. Holding the shared components constant, the conditions varied whether the draft stage LLM (the assistant) or the verify stage LLM (the grader) resolved the expression correctly, and how much independent reasoning the revise stage LLM (the meta-evaluator) needed to determine which reading was correct. Six models from three providers were tested across 21 reasoning effort configurations using e-values for sequential testing, in a primary experiment and an ablation experiment that removed error classification labels from the grader's feedback. A separate LLM analyzed the meta-evaluator's stated rationale for each wrong verdict. Balanced accuracy (the unweighted mean of sensitivity and specificity) ranged from 0.156, below chance, to near-perfect. GPT-5.2 rose from 0.156 without reasoning to 0.942 at its highest reasoning effort level, while Gemini 3 Pro stayed above 0.94 at every level. Gemini 3 Pro at low reasoning effort outscored GPT-5.2 at xhigh reasoning effort for roughly 5% of the cost per trial. When the meta-evaluator erred, it tended to rely on surface cues rather than operational reasoning. Context engineers implementing draft-verify-revise pipelines should be wary of deictic shifts and make the intended referent explicit at each stage. 

---
# Mined from Scientific Literature: Process Schemas for Atomic Layer Deposition and Etching in Materials Science 

**Authors**: Sameer Sadruddin, Eleni Poupaki, Alex Watkins, Bora Karasulu, Adriaan J. M. Mackus, Erwin Kessels, Sören Auer, Jennifer D'Souza  

**Link**: [PDF](https://arxiv.org/pdf/2609.12139)  

**Abstract**: Atomic layer deposition (ALD) and atomic layer etching (ALE) are reported heterogeneously across experimental and simulation literature in materials science, hindering comparison and machine-actionable reuse. We present four domain-expert-reviewed JSON Schemas for ALD and ALE experimental and simulation processes. Curated with schema-miner and grounded in QUDT using schema-miner pro, the schemas structure materials, process conditions, configurations , and measured or predicted results. We compare their scope, structure, and semantic grounding, and demonstrate their use for schema-guided literature extraction and publication of structured records through ORKG templates. 

---
# When Successful Knowledge Graph Edits Displace Correct Answers: Rank-Level Locality beyond Parameter Support 

**Authors**: Yi-Cheng Lai, Jerry Wang, Hsin-Ling Hsu, Li-Chu Chi, Ya-Wen Teng, Hen-Hsen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.12116)  

**Abstract**: Editing a knowledge graph embedding (KGE) model to promote a desired answer can displace correct answers from the returned list. Locality tests based only on facts that reuse the edited parameter can miss this ranking effect. We introduce a common rank-displacement audit at three scopes: facts supported by the edited parameter, other correct answers to the target query, and correct answers across queries with the same relation. We also derive dimensional and geometric conditions for an update to improve the target while exactly preserving selected scores. On FB15k-237 with DistMult and ComplEx, direct promotion always moves the target into the top ten, but does so without damage in only 23.0--23.2\% of edits. Strict preservation causes no measured damage, yet succeeds in only 1.3--1.4\%. Support-regularized entity editing gives the highest joint success, 36.3--37.7\%, while rank-truncated preservation reaches 32.8--34.7\% and reduces the mean number of displaced answers from about 14 to 1.2. Experiments across dimensions, scorers, ranking conventions, and a learned editor show that locality depends on both the protected scope and the editing mechanism. KGE editing should therefore report correction success together with the incidence and severity of rank displacement. 

---
# DU-NO: A Parameter-Efficient Double U-Shaped Neural Operator for Phase-Resolving Wave Modeling 

**Authors**: Enrique Hernandez Noguera, Md Meftahul Ferdaus, Nathan Cooper, Elias Ioup, Mahdi Abdelguerfi  

**Link**: [PDF](https://arxiv.org/pdf/2609.12115)  

**Abstract**: Phase-resolving wave models such as FUNWAVE-TVD are the accuracy standard for nearshore dynamics, resolving the shoaling, refraction, and breaking of individual waves, but their cost rules them out for the ensembles, uncertainty quantification, and real-time warning that operational forecasting demands. Neural operators promise solver-level accuracy at a fraction of that cost, yet on wave-dominated fields the accurate ones are large: hybrid spectral-convolutional operators such as U-FNO (the strongest baseline in our study after DU-NO) buy their fidelity with tens of millions of parameters. We introduce DU-NO (Double U-shaped Neural Operator), a multiscale U-shaped spectral operator that attaches lightweight convolutional U-Net branches only at its two shallowest encoder and decoder levels. The placement follows a sampling argument: high-wavenumber content exists only on fine grids, so the local, full-band pathways go where that content lives, while the coarse, band-limited levels stay purely spectral. A depth-decaying mode schedule holds the model to 3.64M parameters, an order of magnitude below U-FNO. On our publicly released FUNWAVE-TVD benchmark, DU-NO attains the best autoregressive rollout error of six identically trained architectures, improving on U-FNO by 14.9% with 10.8x fewer parameters, and a frequency-band analysis shows the gain holds across all bands, including the high-wavenumber band where truncated-spectral operators collapse. Parameter-matched controls confirm the gain is architectural: rescaled to the same 3.6M budget, the best baseline still trails DU-NO by 28.6%. The advantage carries beyond nearshore waves: DU-NO matches the strongest baselines on 2D Navier-Stokes and wins clearly on PDEBench shallow-water rollouts. Code, trained models, and evaluation artifacts are available at this https URL. 

---
# Language Is an Insufficient Substrate for Quantitative Reasoning, and Consequential Domains Need Large Quantitative Models 

**Authors**: Reuben Vandeventer, David Imrem, David J. Wild  

**Link**: [PDF](https://arxiv.org/pdf/2609.12105)  

**Abstract**: The prevailing assumption in applied machine learning is that progress on consequential quantitative decisions such as pricing risk, allocating capital, triaging patients, or containing a network intrusion will follow from progress in large language models (LLMs). A language model is trained on a representation of the world that was produced by human description; description is a lossy encoding of the quantitative record, and the loss is irreversible: no downstream model, at any scale, can recover from a description what the description did not encode. We formalize this as a property of the representation on which a model is trained rather than of the model capacity, and we identify three further properties that consequential settings demand of a model and that a language substrate cannot supply by construction: reproducibility, lineage from every output back to the source records that produced. it, and calibrated uncertainty. We argue that these properties define a distinct model class, which we call the Large Quantitative Model (LQM). 

---
# Competence-Gated Pooling of Language Models and Priors for Event Forecasting 

**Authors**: Aditi Tiwari, Aashrith Bandaru, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2609.12101)  

**Abstract**: In hybrid forecasting, a language model is often one of several available signals. A system may already have a market, crowd, or statistical forecast and must decide whether the model adds useful information or should be ignored. The relevant target is therefore not standalone model accuracy, but relative competence, defined as the model's marginal value beyond the available external forecast. Under Brier loss, we characterize when model disagreement can improve an external forecast and derive the gain from using domain-specific rather than global pooling weights. We then introduce a competence gate that estimates domain-level source weights from resolved outcomes, shrinks uncertain estimates toward a global weight, and recalibrates the pooled forecast. Across 2,357 resolved binary questions and five language models, the gate improves the main external baseline from 0.0771 to 0.0732 Brier and significantly outperforms global forecast combinations. The gain remains significant under leakage controls and against a leakage-safe time-series prior on the pooled structured set, with separate evidence on FRED. In contrast, the gate gives no significant improvement on the official ForecastBench market subset, where it largely defers to the market. Across four Qwen models, verbal confidence does not reliably identify when the model outperforms the external forecast, while outcome-estimated competence supports better abstention decisions. These results provide a practical approach for selective model use based on measured marginal value. 

---
# Reading the Whole Heart: Latent-Attention Masked Autoencoders for Multimodal Cardiac Representation Learning 

**Authors**: Andrea Agostini, Simon Böhi, Moritz Vandenhirtz, Samuel Ruiperez-Campillo, Max Krähenmann, Silke Mühlstedt, Irene Cannistraci, Ece Özkan Elsen, Julia E. Vogt, Thomas M. Sutter  

**Link**: [PDF](https://arxiv.org/pdf/2609.12035)  

**Abstract**: Cardiovascular diagnosis rests on integrating complementary modalities, like ECG, echocardiography, chest radiographs, and clinical variables, each capturing distinct but correlated aspects of cardiac physiology. Yet most medical foundation models remain modality-specific, combining modalities only for finetuning or post-training. This discards the cross-modal evidence clinicians naturally integrate and ignores the structure within each modality. We introduce Latent-Attention Masked Autoencoders (LAMAE), a multimodal, structure-aware masked autoencoder that jointly learns patient-level representations during self-supervised pretraining. Rather than fusing modalities post hoc, LAMAE exchanges information directly in the latent space through a shared latent-attention module operating over a study-view-entity hierarchy, enabling aggregation of variable observations and graceful handling of missing modalities. Pretrained on over 1.2 million MIMIC-IV hospital stays, LAMAE outperforms modality-specific pretraining and strong contrastive and vision-language baselines across multimodal hospital-stay tasks, such as in-hospital mortality, ICD-10 and DRG coding, and length of stay, while remaining competitive on unimodal tasks. These gains persist even when only a single modality is available at test time, showing that modeling both intra- and inter-modal structure yields more robust, transferable representations. 

---
# Harness or Model? Isolating the Harness Effect in Agentic Coding with a Contamination-Controlled Private Suite 

**Authors**: Mohsen Arjmandi  

**Link**: [PDF](https://arxiv.org/pdf/2609.11987)  

**Abstract**: An agentic coding system couples a language model to a harness: the tools, prompts and control flow that turn a chat model into an autonomous software engineer. Vendors ship harnesses tuned to their own models, and practitioners assume the vendor-native pairing solves more tasks. We measure that assumption with paired same-model contrasts on a private, contamination-controlled suite of 256 repository and post-cutoff contest tasks. The same 80 tasks ran under claude-agent-sdk and under deepagents on claude-opus-4-8, and under the openai-codex SDK and deepagents on gpt-5.5, with gemini-3.5-flash and deepseek-v3.2 as side cells. 792 of 800 planned runs were graded by an isolated oracle. Neither contrast resolves an average advantage for either harness: -1.25 pp for Opus 4.8 (48.8% vs 50.0%, task-bootstrap 95% CI [-10.0, +7.5]) and +1.25 pp for GPT-5.5 (55.6% vs 54.4%, CI [-4.4, +6.9]). The Opus average combines opposite strata: the native harness trails by 9.0 pp on the 61 repository tasks and leads by 23.7 pp on the 19 contest tasks (label-permutation p = 0.003). The partition was chosen after seeing the data and needs a designed replication. Correctness and completion also separate: 22 of 81 runs cancelled at the wall-clock ceiling had produced a passing patch. Re-priced from raw per-turn usage at frozen list prices, the neutral harness cost 1.3 to 1.6 times as much per solved task on Opus 4.8 and 1.2 times on GPT-5.5. These are observed-usage estimates. On the Anthropic account 58 runs left no usage record, and allocating that spend to either cell would move the Opus ratio between 0.7 and 2.3, so the billed ordering is unresolved. This revision corrects an August 2026 manuscript whose cost figures rested on a usage-semantics defect in our own telemetry (Section 5.1). We release the orchestrator, grading oracle, reanalysis code and derived aggregates. The tasks stay private. 

---
# Occamy-1.0: Open Pareto-frontier 35B Intelligence for Co-work 

**Authors**: Wenhui Chen, Shiwen Cheng, Hao Dong, Chenda Duan, Ruixiang Feng, Zhong Guan, Boqiang Guo, Xueyuan Han, Haojie Hao, Liangmeng Huang, Zhelong Huang, Xinke Kong, Hongyu Li, Jiazheng Li, Junbo Li, Qingchuan Li, Yukun Lian, Chang Liu, Tianyu Liu, Zicheng Liu, Shuyi Ouyang, Yijun Pan, Kunyu Shi, Xiaojun Tang, Bingquan Wang, Kesu Wang, Yuchen Wang, Sibo Wei, Sicong Xie, Xiaoying Xing, Yi Xu, Zhijun Xu, Hongwei Xue, Qingcheng Zeng, Di Zhang, Guannan Zhang, Haochen Zhang, Tianlong Zhang, Tianyu Zhao, Tianyu Zhao, Yanjun Zheng, Jialong Zhu, Zijian Zou  

**Link**: [PDF](https://arxiv.org/pdf/2609.11977)  

**Abstract**: Co-work agents execute complex workflows that combine information gathering, tool use, coding, and file manipulation across many model invocations. Because cost and latency accumulate over the full episode, their practical value depends not only on peak capability but also on how efficiently that capability is delivered. Yet many steps in everyday work emphasize state tracking, coordination, recovery, and follow-through rather than frontier-scale reasoning. We present Occamy-1.0, a cost-efficient co-work model obtained by further training the post-trained Qwen3.6-35B-A3B checkpoint. We construct execution-grounded data and environments, capture replayable long-horizon trajectories across multiple harnesses, and use staged post-training to develop and consolidate complementary execution capabilities. Across a broad suite of co-work benchmarks, Occamy-1.0 is consistently among the strongest comparably sized models and remains competitive with substantially larger frontier systems on several tasks. Under our stated evaluation and pricing protocol, its aggregate performance across four representative benchmarks places it at the low-cost knee of the observed cost--performance Pareto frontier. Supporting evaluations in tool calling, coding, and instruction following further show that this specialization preserves broad agentic capability. We release the model weights and a subset of the training data to support research on practical co-work agents and agentic post-training. 

---
# ASTRIL-MPC: Autonomous Traversal Framework of Articulated Tracked Robots with Language-Guided Neural-Kinematic MPC 

**Authors**: Zhenfeng Gan, Yanbo Chen, Lirong Che, Junbo Tan, Xueqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.13083)  

**Abstract**: In urban search and rescue, articulated tracked robots (ATRs) must traverse structured but contact-rich environments such as stairwells and cluttered building interiors. Reliable autonomy remains challenging because robot-terrain interaction (RTI) is hybrid and discontinuous, and effective flipper-track coordination is difficult to model analytically. We present ASTRIL-MPC, a language-guided neural kinematics model predictive control (MPC) framework for autonomous traversal. A learned kinematics model predicts short-horizon task-state increments from a height sequence and recent trajectories; NMPC plans with multi-objective costs and strict feasibility constraints; and a large language model (LLM) proposes bounded updates to selected weights and bounds through a safety-checked interface with range clipping, rate limiting, and consistency checks. The compiled predictor enables a full control cycle within 100 ms. Across three traversal tasks and a multi-height generalization setting, ASTRIL-MPC improves an aggregate traversal-quality score by up to 71% over a non-adaptive NMPC and by 67% over a PPO baseline, while eliminating measurable collision impacts during descent. These results indicate that combining learned kinematics, optimization-based planning, and language-guided retuning yields data-efficient and robust autonomy for articulated tracked robots. 

---
# MP-Bench: Evaluating Voice Agents as a Multiparty Conversation Participant 

**Authors**: Yi-Jen Shih, Shih-Yun Shan Kuan, Guan-Ting Lin, Kai-Wei Chang, Siddhant Arora, Shu-wen Yang, Abdelrahman Mohamed, Shinji Watanabe, Hung-yi Lee, David Harwath  

**Link**: [PDF](https://arxiv.org/pdf/2609.13076)  

**Abstract**: Conversational voice agents have advanced significantly, offering increasingly natural human-machine interactions through both cascaded and end-to-end architectures. However, while recent benchmarks extensively evaluate dyadic interactions and passive audio comprehension, they largely overlook a prevalent real-world scenario: multi-party conversations. Evaluating agents in these settings is fundamentally more challenging than in dyadic interactions due to the exponentially greater conversational complexity. For voice agents to integrate seamlessly into human group dynamics, they must not only generate contextually appropriate responses but also demonstrate a nuanced understanding of open turn-taking. To address this gap, we introduce Multiparty Bench (MP-Bench), the first benchmark specifically designed to objectively evaluate conversational speech systems as active participants within multi-party contexts. MP-Bench assesses agent behavior along two primary dimensions: turn-taking awareness and response appropriateness. Additionally, we incorporate comprehension-based question-answering tasks as a complementary evaluation. By benchmarking 12 voice agents, we find that real-time voice agents stay at or below 22% on multiparty comprehension and remain near chance on multiparty turn-taking, exposing an open challenge for real-time voice agents under multiparty scenario. 

---
# MAxBench: A Multinomial Concept Recovery Benchmark 

**Authors**: Divya Appapogu, Freya Behrens, Yonatan Belinkov, Aaron Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2609.13072)  

**Abstract**: Fine-grained control of language model behaviors (e.g., steering) is among the more actionable outcomes of interpretability research. For binary concepts such as refusal, a single direction in activation space often suffices for steering. However, many concepts are not binary: Animals and Countries contain many subcategories, each with multiple instances. For these concepts, the search space over possible representation geometries is far larger than for binary concepts; it is thus not clear what geometries are most appropriate, nor what methods are most effective at recovering them. In this work, we introduce MAxBench, a geometry-agnostic evaluation framework for multinomial concept representations based on sampling from the recovered concept representation. We use MAxBench to compare 10 localization methods (covering 5 geometry types) across 6 concepts and 4 models. Using this framework, we find that (i) affine subspaces steer more reliably and have greater recall than rank-one or linear subspaces; (ii) much of this advantage is due to better non-zero offsets rather than the choice of bases; (iii) manifold steering is competitive with the best methods when applicable; and (iv) no method consistently outperforms prompting, in alignment with prior findings on binary concepts. These findings underscore the importance of expanding the scope of interpretability research and meta-evaluation to concepts with more varied structure. 

---
# Involving before Evolving: A Vision for Trustworthy Enterprise Digital Twin Engineering 

**Authors**: Kérian Fiter, Adil Lagrou, Franck Dervault, Bentley Oakes  

**Link**: [PDF](https://arxiv.org/pdf/2609.13071)  

**Abstract**: Enterprise Digital Twins (EDTs) promise data-driven decision support at organizational scale, but realizing them requires navigating siloed departments, tacit knowledge, and high-stakes decisions with long-horizon consequences. Existing approaches involve domain experts during model development but focus less on early organizational buy-in in EDTs. We present a vision for trustworthy EDT engineering grounded in an `involving before evolving' paradigm: rapidly involving stakeholders through a working prototype before evolving toward federation and full interoperability. Our three-stage approach combines foundation models for rapid prototyping, an ontological backbone for federated interoperability, and observability tooling for stakeholder trust. We ground our vision in an ongoing collaboration with Michelin, a multinational manufacturer, where an initial prototype has helped support stakeholder buy-in. 

---
# Dynin-Robotics: Omnimodal Unified Diffusion Vision-Language-Action Model 

**Authors**: Hoeun Lee, Jaeik Kim, Jusang Oh, Jinhyeok Kim, Geon Choi, Hyeonggeun Kim, Jaeyoung Do  

**Link**: [PDF](https://arxiv.org/pdf/2609.13053)  

**Abstract**: Visual goal and dynamics prediction can provide language-conditioned robot policies with both a target outcome and a representation of action-dependent scene changes. We bring these predictions into action generation and selection through a shared trajectory model. Dynin-Robotics implements this formulation on Dynin-Omni, an omnimodal masked-diffusion backbone, representing language, visual observations, goals, and actions as discrete tokens. By varying conditioning and target spans, the same model learns action prediction, action-conditioned next-observation prediction, terminal goal-state prediction, and trajectory-to-instruction reconstruction. These interfaces support test-time scaling through goal prediction, action-candidate evaluation, and joint refinement of action and future-state predictions. We continually pretrain the model on approximately 1.33 million trajectories from 48 Open X-Embodiment datasets and adapt it separately to downstream domains. On two VLABench tasks, robot pretraining improves adaptation within a fixed Stage-2 step budget, and the full objective mixture improves shifted-instruction success over Policy-only post-training under the same coupled decoder. Combining goal guidance with joint action-next-state denoising further improves shifted-instruction success over action-only decoding; the benefit depends on how the predictions are composed. Dynin-Robotics achieves competitive performance on LIBERO and zero-shot LIBERO-Plus, together with a 78.4% average success rate across four manipulation conditions on a Franka Research 3 robot. An optimized block-parallel implementation accelerates model-side action decoding by up to 29.2x relative to the base implementation under the reported profiling setup. These results support shared trajectory modeling as a common interface for learning complementary robot objectives and composing their predictions during control. 

---
# Unified CT and MRI Pancreas Segmentation for Label-Efficient Cross-Modality Subregion Transfer 

**Authors**: Ziliang Hong, Hongyi Pan, Halil Ertugrul Aktas, Andrea Bejar, Elif Keles, Frank H. Miller, Michael B. Wallace, Rajesh N. Keswani, Gorkem Durak, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2609.13043)  

**Abstract**: Robust medical image segmentation across imaging modalities is challenging because of large differences in appearance and intensity distributions. Models trained on a single modality often show substantial performance drops when applied to unseen domains. In this work, we develop a unified 3D pancreas segmentation framework that applies domain-adversarial learning to 4,604 heterogeneous CT and MRI scans to learn anatomical representations. A shared nnU-Net encoder-decoder is trained for whole-pancreas segmentation, with a latent domain discriminator encouraging CT-MRI feature alignment. The learned encoder is subsequently transferred to pancreatic head-body-tail segmentation using limited MRI-only subregion annotations. An average Dice score of 87.31% on the in-distribution test set and Dice scores ranging from 84.20% to 88.09% across external OOD datasets were achieved in whole pancreas segmentation. Dice scores of 80.53% on MRI and 83.05% on CT were achieved for downstream subregion segmentation, without using CT subregion annotations. These results demonstrate that a unified anatomical representation can support both cross-modality pancreas segmentation and label-efficient downstream transfer. 

---
# DynSHAP: Towards Explainable Dynamic Survival Analysis 

**Authors**: Nastasya Anokhina, Jonas Jürß, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2609.13042)  

**Abstract**: Deep learning models for dynamic survival analysis (DSA) achieve strong predictive performance by incorporating longitudinal patient data, but their black box nature limits clinical trust and adoption. Existing explainability methods cannot handle longitudinal, irregular inputs and functional survival outputs simultaneously, which limits their usability in DSA. We propose DynSHAP, a SHAP framework suited specifically for dynamic survival analysis. It extends common marginal SHAP estimators to this setting by treating time--feature pairs as players in the Shapley game. We further introduce Temporal DynSHAP, which learns linear dependencies in features over time and uses conditional sampling to address them in explanations. When applied to synthetic data with known ground-truth attributions, Temporal DynSHAP recovers temporally dependent features more accurately than marginal estimators for a given state-of-the-art model. Applied to two real-world clinical datasets and two DSA architectures, DynSHAP produces attributions faithful to model learning, allowing medical experts to see which patient information drove the prediction and when. 

---
# Groupoid-Based Internal State Representations for Reinforcement Learning with Local Symmetries 

**Authors**: Ben Opperman, Eduardo Alonso, Esther Mondragón  

**Link**: [PDF](https://arxiv.org/pdf/2609.13035)  

**Abstract**: Symmetries play a central role in reducing the complexity of reinforcement learning problems, yet most existing approaches rely on fixed group actions or predefined state abstractions. Classical reinforcement learning algorithms typically assume a globally structured Markov decision process with uniformly applicable actions and transitions, an assumption that limits their ability to exploit modularity and local, context-dependent regularities present in many realistic environments. We propose a reinforcement learning framework using groupoids to capture local, state-dependent symmetries and support the dy- namic discovery of equivalence structures during interaction. The agent maintains orbit representatives together with transporters that map raw states to canonical forms, enabling learning and decision-making to be performed in a symmetry-reduced space while preserving local distinctions. Empirical results demonstrate that the proposed groupoid-based approach improves sample efficiency and convergence in dense and large-scale environments exhibiting strong partial symmetries, yielding substantial performance gains over standard Q-learning. These findings show that dynamically exploiting local symmetry provides a practical and mathematically principled route to scalable and generalisable reinforcement learning. 

---
# Attention Quantization for Tabular Foundation Models 

**Authors**: Jonas M. Kübler, Benjamin Jäger, Klemens Flöge, Noah Hollmann, Frank Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2609.13031)  

**Abstract**: With the recent rise and adoption of tabular foundation models, optimizing their inference performance becomes an emerging field for efficiency research. While the models are architecturally similar to transformer-based large language models (LLMs), the size and serving patterns differ significantly. We show that the focus should be on the attention calculation and less on weight or KV cache quantization, which are more popular in LLMs. We develop a quantization strategy for queries, keys, and values to FP8 and use explicit FP8 matrix multiplication instructions to speed up the attention calculation. We find that it is crucial to align the quantization error in the test rows with the quantization error in the training rows, as otherwise the accuracy drops drastically. Our Triton kernel achieves a speedup up to 1.7x over regular 16-bit kernels, and we show that on TabPFN-v3 and TabICLv2 there is no relevant accuracy loss across TabArena and BeyondArena. 

---
# Label-Guided Knowledge Distillation for 3D-CNNs in Action Recognition 

**Authors**: Yanjiang Shi, Peng Zhao, Nan Qi, Guiqin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.13024)  

**Abstract**: As a key model compression technique, knowledge distillation aims to transfer knowledge from a high-capacity teacher model to a lightweight student model for enhancing the latter's performance. In this work, we reviewed the feature knowledge distillation for 3D-CNNs and observed that most feature distillation methods in video analysis are simple adaptations of those used in image analysis, often neglecting the differences of video features in the temporal dimension. To address this issue, we proposed Label-Guided Knowledge Distillation (LGKD) to guide the distillation of student model features using ground truth labels. Our method entails two components: sample-wise distillation and class-wise distillation, enabling the student model to learn feature representation of the teacher model at two levels. Sample-wise distillation utilizes label information and the teacher's probability distribution to guide the learning of features that significantly impact temporal accuracy while mitigating noise. Meanwhile, class-wise feature distillation employs a prototype network to further capture the relational knowledge among samples within the same category, enhancing the student's ability to learn higher-dimensional semantic information and improving model generalization. To demonstrate the effectiveness and superiority of our method, we conducted comprehensive experiments on two benchmark action recognition datasets, UCF101 and HMDB51, achieving competitive results. 

---
# TileNet: Tile-Based CNN-SVM Architecture for Autonomous Unmanned Aerial Systems Inspection of Flat Roofs 

**Authors**: Samuel Dunthorne, Hashim A. Hashim  

**Link**: [PDF](https://arxiv.org/pdf/2609.13013)  

**Abstract**: Flat roofs are among the most influential components of the building envelope, governing both structural performance and thermal efficiency, and thereby contributing directly to household energy consumption, carbon emissions, and long-term environmental sustainability. Timely detection of roof defects is essential for reducing heating and cooling losses, preventing moisture-driven degradation such as mold growth, and supporting national climate-change mitigation goals. This paper presents a real-time, Unmanned Aerial System (UAS)-based deep learning framework that autonomously detects defects using live imagery captured during dual-altitude aerial passes. The multi-resolution flight strategy is designed to aid the identification of both small, fine-scale defects and larger structural issues, enabling more comprehensive assessments. To meet the strict computational and power constraints of embedded UAS hardware, the proposed framework integrates a tile-based architecture with a lightweight Convolution Neural Network-Support Vector Machine (CNN-SVM) classifier designed for low-latency onboard inference. The final model-comprising five convolutional layers and four dense layers, the last a linear SVM head, achieved a mean test accuracy of $94.4\%$ ($95\%$ confidence interval $\pm0.4\%$ over three seeds) on a photo-level split ($43,383$ training, $3,869$ validation, and $2,540$ test tiled and augmented images), outperforming GoogLeNet ($89.2\%$) and AlexNet ($79.8\%$). Experimental evaluations using real UAS imagery collected by onsite visits with DJI Matrice 350 RTK drone demonstrate that the system supports rapid, repeatable, and safe roof inspections while reducing human risk, lowering operational costs, and enabling more sustainable building maintenance. 

---
# Comfort by Construction: Adaptive, Comfort-Bounded Action Spaces for Learned Driving Policies 

**Authors**: Anna Rothenhäusler, Daniel Jost, Raghu Rajan, Faris Janjos, Oliver Scheel, Andreas Look, Joschka Boedecker  

**Link**: [PDF](https://arxiv.org/pdf/2609.13011)  

**Abstract**: Data-driven driving simulators command accelerations and steering rates from a fixed grid without constraining the realized accelerations and jerks. As a result, reinforcement-learning policies inflate safety metrics through abrupt, last-second maneuvers that lie far outside the range of human driving and would be unacceptable to occupants of a real vehicle, so the metrics measure simulator permissiveness rather than policy quality. Enforcing comfort bounds naively is not enough: lateral limits shrink quadratically with speed, so clamping a static grid saturates it and destroys fine-grained control ("grid collapse"). We propose an adaptive action parameterization that rediscretizes the grid at every step to span exactly the per-step feasible control set, via closed-form inversion of the lateral-jerk constraint. We further present PufferDrive-Editor, a browser-based tool to audit realized kinematics and author kinematically challenging scenes. On the Waymo Open Motion Dataset and a hand-authored slalom, our adaptive model holds comfort violations below 1% while outperforming clipped-grid and direct-jerk baselines in navigability. 

---
# Tasks over Application Manuals: Revealing Gaps in Long-Horizon Procedural Reasoning for Language Models 

**Authors**: Utkarsh Soni, Syed Shariyar Murtaza, Yifan Nie, Sachin Chandrasekhar, Eugene Wen  

**Link**: [PDF](https://arxiv.org/pdf/2609.13005)  

**Abstract**: Large language models (LLMs) have achieved strong performance on a wide range of natural language tasks, and recent benchmarks suggest that they are increasingly adept at multi-hop reasoning. However, these benchmarks are typically short-horizon, requiring only a small number of retrieval or inference steps, and provide limited evidence of reliability on real-world tasks that involve following manuals spanning hundreds of pages with complex, interdependent guidelines. In this paper, we introduce Tasks over Application Manuals (TAM), a benchmark for evaluating long-horizon procedural reasoning. We construct TAM by curating real-world tasks from two domains: ICD-10-CM clinical coding (mapping medical conditions to diagnostic codes) and U.S. federal sentencing (computing crime sentencing guideline outcomes, specifically offense levels), with human-validated labels. Each task requires following an authoritative manual with tens of thousands of rules and executing a sequence of interdependent steps across different sections to produce an exact answer. We evaluate general-purpose prompting approaches, including retrieval-augmented generation, ReAct-style prompting, and an agent-harness baseline on GPT-5, and find that the best exact-match performance remains extremely low: 1% on ICD-10-CM coding and 15.5% on sentencing tasks. These results show that current benchmarks may overestimate LLM reasoning ability and miss a key challenge: reliably following long, rule-based procedures. The complete TAM data and code are publicly available. 

---
# SeqMoE: Toward Full-Load Performance via Predictive and Graph-Compatible MoE Offloading 

**Authors**: Zihan Wang, Yuqi Wang, Lei Gong, Cheng Tang, Wenqi Lou, Teng Wang, Chao Wang, Xuehai Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.12978)  

**Abstract**: Mixture-of-Experts (MoE) creates a structural advantage for offloading: only a small fraction of activated experts need to reside in device memory, and if they can be loaded in time for computation, offloading can in principle approach full-load performance, where all model weights reside in device memory. Yet translating MoE's structural advantage into practical offloading gains remains challenging. We propose SeqMoE to bridge this gap. To maximize expert hits, we build predictive memory management: (i) Sequence-to-sequence prediction. We are the first to recast expert activation prediction as sequence modeling, enabling accurate multi-step, multi-layer forecasts that provide a long and reliable window for downstream decisions. (ii) Joint prefetch scheduling. We formulate prefetch scheduling as Job Sequencing with Deadlines to maximize expected expert hits and improve bandwidth efficiency. (iii) Forecast-driven caching. Leveraging the recursive nature of sequence modeling, we introduce a probabilistic Belady policy for future-aware eviction. To eliminate execution bottleneck, we develop (iv) Graph-compatible offloading runtime. We derive general runtime principles encompassing compute-transparent expert placement and synchronization-free orchestration disciplines for end-to-end graph capture. With 45% expert residency, SeqMoE averages a 96.97% hit rate and 80.22% of full-load performance, advancing the state of the art in MoE offloading. 

---
# Generative Retrieval for Unsupervised Text-Based Person Search 

**Authors**: Mang Ye, Yucheng Ji, Yang Bai, Min Cao, Siyuan Chai, Bo Du, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.12965)  

**Abstract**: Text-based person search (TBPS) aims to retrieve images of a target person from a large image gallery based on a given natural language description. Most existing methods rely on supervised learning with manually annotated image-text pairs. In this paper, we explore unsupervised TBPS, with only unlabeled images. We propose GTR+, a two-stage generation-then-retrieval framework. In the generation stage, we introduce a tiered description generation framework designed to produce fine-grained and stylistically diverse textual descriptions through a three-tier sequential process. The base tier leverages an automated question-and-answer mechanism to generate basic visual attribute descriptions; the intermediate tier enhances fine-grained detail using an inter-sample contrastive mechanism; the advanced tier further enriches textual diversity via a stylized expansion mechanism. In the retrieval stage, to mitigate the impact of noisy pseudo texts, we develop an adaptive confidence-weighted retrieval learning framework. We model image-text pairs as clean or noisy using a Gaussian Mixture Model, calibrated by real-time image-text similarity and static text generation probability from the prior stage, yielding adaptive sample weights during training. Beyond that, we also contribute LargeFine-Person, a large-scale TBPS dataset with high-quality, fine-grained, and diverse textual annotations, enabling a practical and generalizable TBPS pre-training benchmark under unsupervised setting. Experiments on multiple TBPS benchmarks demonstrate the effectiveness and generalization of both GTR+ and LargeFine-Person. Code is available at: this https URL. 

---
# ARC: Autonomous Robotics Compliance A Three-Layer Governance Architecture for Deployed Autonomous Systems 

**Authors**: Tord Eide, Einar Holt  

**Link**: [PDF](https://arxiv.org/pdf/2609.12932)  

**Abstract**: Proposed governance framework for autonomous robotic systems, introducing a three-layer compliance architecture (ARC) instantiated through model safety validation, cognitive certification benchmarks, and operational authorization standards. 

---
# LLM-Enhanced Dual-Branch Learning for Large-Scale Multi-Label Text Classification 

**Authors**: Hui Ye, Jing Zhang, Xiulong Yang, Rajshekhar Sunderraman  

**Link**: [PDF](https://arxiv.org/pdf/2609.12915)  

**Abstract**: Large-scale multi-label text classification assigns a small subset of relevant labels to each document from a vocabulary containing thousands or tens of thousands of candidate labels. Although pretrained language models have improved semantic text representations, most representation-based approaches center their prediction pipelines on a primary encoder or combine auxiliary features within a single ranker. The complementarity between heterogeneous language models therefore remains insufficiently explored. We propose DualMLC, a dual-branch framework that processes the same document through an autoregressive decoder-only language model and a bidirectional encoder. Each branch maintains its own representation pathway and independently estimates relevance scores over the shared label space. DualMLC combines the two score vectors through late logit fusion, allowing shared evidence to reinforce relevant labels and branch-specific evidence to compensate for limitations in the other branch's representation. DualMLC achieves state-of-the-art results on three widely used large-scale multi-label text classification benchmarks. Ablation results further confirm that integrating the heterogeneous predictors produces stronger rankings than either branch alone. The source code is publicly available at this https URL. 

---
# UniPart: Towards Zero-shot Language-Grounded 3D Part Segmentation for Embodied Interaction 

**Authors**: Xinqiang Yu, Zekun qi, Jiawei He, Wenyao Zhang, Xuchuan Chen, Guaocai Yao, Li Yi, Zhaoxiang Zhang, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.12898)  

**Abstract**: Fine-grained robotic manipulation depends on understanding parts, not only whole objects. Existing 3D foundation models tend to be either generalized but object-aware, or part-aware but limited to closed-set taxonomies, which weakens zero-shot transfer. We study text-conditioned 3D part segmentation, where a free-form phrase selects a functional part on point cloud. We introduce UniPart, a feed-forward cross-modal 3D Transformer that conditions CLIP text embedding. To scale supervision, we build LangPart-1M with 160K+ Objaverse assets and 8M text to part pairs using multi-view consistent part generation. We further manually label a high-quality subset, LangPart-4K, for fine-tuning and evaluation. UniPart achieves strong zero-shot results on open-vocabulary part benchmarks and transfers to language-conditioned part grasping in real world. 

---
# Behavior Quotient Learning for Low-Rank Adaptation of LLM Agents 

**Authors**: Pengyang Zhou, Xiaobin Tu, Zhengxi Liu, Rongkun Xue, Haochen Li, Miancan Liu, Ziyuan Chen, Yinggui Wang, Jinkui Ren, Xiantao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.12896)  

**Abstract**: LLM-based agents rely on heterogeneous interaction capabilities to accomplish complex tasks. Existing approaches often distribute these capabilities across multiple LoRA adapters, which increases adapter storage requirements and introduces routing overhead during inference. A single LoRA avoids this overhead, but learning from diverse agent trajectories under a fixed rank budget presents two challenges. First, trajectories with different interaction traces and parameter gradients can induce equivalent changes in decision distributions, causing repeated updates to overemphasize redundant behavioral changes. Second, an aggregated update may exceed the rank budget of the adapter, and approximating it in weight space can distort the decision changes that it is intended to produce. We propose BQ-LoRA, a low-rank adaptation framework that organizes trajectory updates through a local behavior quotient manifold. It contains two modules, i.e., behavior quotient balancing (BQB) and decision preserving compression (DPC). BQB constructs the quotient manifold from decision distributions and reweights trajectory update directions according to their local density in the quotient tangent space. DPC projects the balanced gradient onto the intrinsic fixed rank tangent space and refactorizes the resulting target by jointly controlling effective weight error and distortion of decision distributions. Experiments on AppWorld and BrowseComp-Plus compare BQ-LoRA with standard LoRA and recent low-rank adaptation methods, while separate ablations evaluate the complementary contributions of both components. 

---
# Large Distant Gradients Need Not Be Reliable: reliability-weighted credit assignment for long-horizon autoregressive forecasting 

**Authors**: Junhao Zhao, David Michael Simberg, Jacob Kang, Colin Connor Kurniawan, Nan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12890)  

**Abstract**: In autoregressive forecasting, long prediction rollouts provide distant supervision, but backpropagation through time (BPTT) carries gradients from those losses through many autoregressive steps. Repeated Jacobian products can make distant gradients dominate the update while amplifying predictable signal and unpredictable noise together; a large distant gradient therefore need not carry reliable learning signal. Motivated by this observation, we introduce Internal Dual-Wiener routing (Internal-DW), a principled backward-only intervention that preserves the full forward rollout and all horizon losses while reliability-weighting internal gradient routes. At each residual block, we derive bounded Wiener gains for the identity and nonlinear routes that balance preserving predictable learning signal against suppressing unpredictable variation, and estimate them from route-level gradient statistics and an explicit noise model. In a controlled system with known gradient signal-to-noise ratio (SNR), we show that distant gradients can grow even as their SNR falls, and that Internal-DW reduces held-out error in recovering predictable gradient signals and improves forecasting. On four history-dominated, weak-drive testbeds, Internal-DW reduces forecast error by 5.2%-13.8% relative to full BPTT, outperforms gradient clipping and Jacobian regularization on all four, and outperforms validation-selected truncated BPTT (TBPTT) on three. It also extends or preserves the fitted optimal training-horizon range across these four testbeds. Across the full benchmark suite, the current Internal-DW estimator has a clear applicability boundary: its benefit diminishes or reverses when usable history is limited or when the selected sampler fails to represent dominant drive-dependent variation. The results show that retaining long-horizon supervision does not require trusting every backward contribution equally. 

---
# A Multi-Vehicle Dataset with Camera, LiDAR, and Radar Sensors and Scanned 3D Models for Custom Auto-Annotation using RTK-GNSS 

**Authors**: Philipp Berthold, Bianca Forkel, Mirko Maehlisch  

**Link**: [PDF](https://arxiv.org/pdf/2609.12871)  

**Abstract**: Datasets are a crucial element in the development of perception algorithms. They relate sensor measurement data to annotated reference information and allow for the deduction of sensor and object characteristics. In autonomous driving, the reference data commonly consist of semantic image segmentation, point-wise associations, or bounding box annotations. The dataset proposed in this work, however, aims to dig deeper into the evaluation of measurement principles and provides scanned 3D models of all vehicles together with a pose and continuous kinematics reference obtained by RTK-GNSS. Combined, the state of the complete dynamic surrounding of the sensor vehicle is known for any point in time. Subsequent reference formats can be easily computed in user-defined granularity. This dataset involves single-object and multi-object recordings with seven target vehicles. In particular, measurement effects such as occlusion, as well as reflections, can be evaluated, as the normals of the shape of the target vehicles are known. We describe the dataset, discuss the technical background of its development, and briefly present exemplary evaluations. 

---
# 3D CT-to-PET Translation via Latent Brownian Bridge Diffusion 

**Authors**: Sarita Mourya, Francesco Di Feola, Pierangelo Veltri, Paolo Soda  

**Link**: [PDF](https://arxiv.org/pdf/2609.12860)  

**Abstract**: Computed tomography (CT) and positron emission tomography (PET) provide complementary anatomical and functional information for cancer diagnosis and treatment planning. However, the widespread use of PET is limited by high radiation exposure, elevated costs, and restricted availability. To address these limitations, deep learning-based CT-to-PET translation has emerged as a promising approach for synthesizing PET-like information directly from CT images, although accurately modeling the large cross-modal gap remains challenging. In this work, we propose a 3D CT-to-PET translation framework based on latent Brownian Bridge Diffusion (BBDM). The method consists of two stages. First, a Variational Autoencoder (VAE) is trained on paired CT-PET patches, integrating contrastive learning to improve latent alignment between anatomical and metabolic representations. Second, a BBDM is trained in the latent space to translate CT latent representations into their corresponding PET counterparts. The translated PET latents are then decoded and stitched to reconstruct the final 3D PET volume. We evaluate the proposed approach on two publicly available datasets. Quantitative results based on image fidelity and lesion-level PET-specific metrics demonstrate improved performance compared with competing methods. In particular, the proposed approach improves PET signal fidelity, better preserves clinically relevant uptake patterns, and shows improved performance in preserving small-lesion metabolic activation, paving the way for virtual imaging applications. 

---
# A Graph-Based Approach for Mapping Kernel-Level Telemetry to MITRE ATT&CK 

**Authors**: Matteo Lupinacci, Luigi Arena, Francesco Blefari, Angelo Furfaro  

**Link**: [PDF](https://arxiv.org/pdf/2609.12841)  

**Abstract**: Mapping observed system behavior to standardized frameworks like MITRE ATT&CK is essential for threat-informed defense, but remains largely manual. Existing automated methods depend on Cyber Threat Intelligence reports, which offer only retrospective accounts of attacks. Low-level telemetry, i.e. kernel-level system calls, instead provides evidence of adversary behavior, yet its volume and complexity have limited its use for automated mapping. We present a methodology that collects kernel-level events via eBPF, correlates attacker commands into a provenance graph, and derives compact graph representations suitable for LLM-based reasoning. These representations are mapped to the MITRE ATT&CK framework using both pure LLM prompting and retrieval-augmented generation (RAG) grounded in the ATT&CK knowledge base, producing ranked technique candidates along with supporting rationales. We implement this methodology as an end-to-end pipeline, named Trace2ATT&CK and evaluate it on 347 Linux Atomic Red Team tests using locally deployed open-weights LLMs. RAG consistently improves ATT&CK mapping performance over pure prompting, while provenance graph substantially outperforms raw telemetry. These results show that local inference over graph-based behavioral descriptions can make automated ATT&CK mapping from kernel-level telemetry operationally viable, without compromising data confidentiality. 

---
# Evaluating Context Segmentation in Locally Deployable SLMs for Cybersecurity CTF Tasks 

**Authors**: Sebastiano Nordio, Michele Lotto  

**Link**: [PDF](https://arxiv.org/pdf/2609.12839)  

**Abstract**: The proliferation of highly capable open-weight Small Language Models (SLMs) democratizes access to advanced cybersecurity capabilities, posing a escalating risk as these models can bypass proprietary API guardrails when deployed locally. However, SLMs deployed as autonomous agents often struggle with long-horizon, exploratory tasks like cybersecurity Capture The Flag (CTF) challenges due to context bloat and cognitive degradation from accumulated tool-call outputs. To understand and mitigate this cybersecurity threat, we introduce \textit{context segmentation}, a two-level agentic framework that divides complex exploitation tasks into manageable, contextually isolated sub-problems. Evaluating on the \texttt{picoCTF} dataset using memory-constrained \texttt{gemma-4} models, we demonstrate that for the E4B model, our strategy acts as an intelligent search, achieving competitive rewards with superior token efficiency compared to brute-force retries, and successfully solving 18.52\% of tasks that standard agentic execution fails to complete. Code is available at this https URL. 

---
# Online Video Agent Harness for Long Video Understanding 

**Authors**: Sen Yang, Boqiang Duan, Jing Yang, Weihao Bo, Jie Liu, Boyuan Tong, Ze Feng, Wenkang Zhang, Jingdong Wang, Hua Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12818)  

**Abstract**: Long video understanding often behaves like a visual needle-in-a-haystack problem: query-relevant evidence is sparsely distributed across long temporal spans, while packing dense frames into a single VLM context incurs \textit{context rot} and high cost. Existing video agents often rely on query-agnostic offline preprocessing or ad hoc tool sets, which can miss query-specific details and waste computation. In this work, we present VideoXAgent, a purely online video-agent harness for long video understanding that starts from the given video file and user query, plans and decomposes the task, invokes specialized expert tools on demand, and aggregates multimodal evidence to produce a final answer while resolving conflicts among observations. To support this on-demand invocation, we design a suite of heterogeneous expert tools guided by a data-driven taxonomy of atomic capabilities, spanning scripts, VLMs, and domain models (e.g., detection, OCR, ASR, face recognition). The harness further enforces objective evidence prompting and budget-aware control to curb hallucination and non-termination. Across Video-MME-Long, LongVideoBench-Long, LVBench, and MINERVA, VideoXAgent is competitive with frontier LMMs and video agents under a smaller context footprint---about 50k tokens of agent context per sample, even on hour-long videos. In particular, on complex video-reasoning benchmarks such as MINERVA, it matches this level while using only about 15\% of the context of a 1,024-frame dense-packing baseline. Notably, the harness remains effective with a visually weak or even text-only orchestrator, suggesting that strong long-video understanding can emerge from progressive agentic evidence seeking rather than from packing the full video into a single context. Project page: this https URL 

---
# 4D Parallelism Unlocks Exascale Bayesian Neural Networks for High-Fidelity Atmospheric Modeling 

**Authors**: Deifilia Kieckhefen, Juan Pedro Gutiérrez Hermosillo Muriedas, Lars Helge Heyen, Mathis Bode, Iida Hakulinen, Andreas Herten, Chelsea Maria John, Thorsten Kurth, Anni Moisala, Asena Karolin Özdemir, Kaleb Phipps, Oskar Taubert, Arvid Weyrauch, Markus Götz, Charlotte Debus  

**Link**: [PDF](https://arxiv.org/pdf/2609.12815)  

**Abstract**: We present BEAST, the first-ever Bayesian Swin Transformer for atmospheric forecasting on 0.25$^\circ$ global resolution able to accurately quantify both aleatoric and epistemic uncertainty. To overcome the associated computational bottlenecks, we devise an orthogonal 4D-parallelization scheme that introduces a unique domain-tensor-parallelism strategy and a novel uncertainty parallel method, enabling us to fully leverage GPU capacity and efficiently scale model training. For a 2.4-billion-parameter model, we achieve a peak performance of 3.96 EFLOP/s on 20,480 NVIDIA GH200 GPUs on the JUPITER supercomputer. We train BEAST as a 700-million-parameter model with 96 random weight samples on 384 nodes on 40 years of data for nearly one million gradient updates. This model achieves predictive skill scores competitive with state-of-the-art probabilistic atmospheric AI models and numerical models, and can predict extreme events with exceptional skill, while generating large ensembles 3 to 4 times faster than the current-best AI model. Our contribution unlocks the potential of high-fidelity uncertainty quantification in atmospheric AI models, heralding a new era for AI-based models in climate and Earth system sciences. 

---
# RunningTensor: Generalizing Linear Attention to Higher-Order Recurrent States 

**Authors**: Luca Herranz-Celotti, Vincent Guigue  

**Link**: [PDF](https://arxiv.org/pdf/2609.12814)  

**Abstract**: Linear attention and state-space models provide linear-time sequence modeling, but their recurrent memory remains a second-order tensor (a matrix), limiting the order of interactions that can be represented in the state. We introduce the RunningTensor, which generalizes this memory to an order-$o$ tensor, updated by a rank-1 outer product and read by contracting against $o-1$ vector queries. Order $2$ recovers linear attention; we study order $3$ as a proof of concept, retaining both recurrent and parallel forms while remaining linear in sequence length $T$ and improving working memory capacity from $\mathcal{O}(W^2)$ to $\mathcal{O}(W^o)$. On synthetic multi-query associative recall, RunningTensor outperforms linear-attention and SSM baselines. After pretraining, it also improves performance on language-understanding and non-synthetic retrieval tasks, suggesting that higher-order recurrent state can provide useful additional memory capacity beyond matrix-valued state. 

---
# Cognition on Graph: Navigating Massive Knowledge Space via Cognitive Cycles and Bidirectional Graph-Text Synergy 

**Authors**: Gengxian Zhou, Jian Xu, Zichen Tang, Shiming Xiang, Haihong E, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12791)  

**Abstract**: Retrieval-Augmented Generation (RAG) has empowered Large Language Models (LLMs) to tackle knowledge-intensive tasks. However, navigating global, heterogeneous knowledge bases (large-scale knowledge graphs and text corpora) for complex reasoning remains a challenge. Existing methods typically employ reactive, graph-driven exploration strategies, which blindly follow graph topology without adapting to the question context or evolving exploration progress, and lack deep bidirectional synergy between graph and text. To address these limitations, we propose CoG (Cognition on Graph), a cognitive-inspired, training-free framework for adaptive knowledge exploration. Drawing inspiration from human problem-solving, CoG performs a continuous plan-explore-reflect cycle, where it proactively formulates investigation plans, performs dual-source retrieval, and dynamically reflects on progress to adjust strategies. Crucially, it establishes deep bidirectional synergy between structured graph and unstructured text, where entities extracted from text dynamically guide graph exploration to bridge knowledge gaps. Extensive experiments on seven multi-hop QA benchmarks demonstrate that CoG significantly outperforms state-of-the-art methods while achieving superior exploration efficiency. Our code and datasets are available at this https URL. 

---
# GraphAHA: Graph-Based Adaptive Search with Heterogeneous Actions for Test-Time Code Generation 

**Authors**: Xitao Li, Haijun Wang, Gege Yuan, Qiyuan Wu, Jiali Wei, Ming Fan, Xiaofei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2609.12757)  

**Abstract**: Test-time scaling improves code generation by spending additional inference budget (e.g., calls or tokens) on direct sampling, feedback-conditioned repair, and reasoning-guided implementation. Search-based methods can allocate this budget adaptively, but two challenges remain. First, tree-structured search treats each generation history as a separate state even when trajectories converge to the same program, duplicating evaluation and preventing statistics from being shared. Second, sampling, repair, and reasoning have complementary and state-dependent payoffs, making online allocation among them difficult under a finite budget. To address these challenges, we propose an adaptive graph search method with heterogeneous actions (GraphAHA). GraphAHA organizes the test-time code generation in a typed directed acyclic graph. Equivalent programs are merged into a single code node, allowing their downstream search statistics to be reused across all discovery paths. Hierarchical Thompson sampling then selects whether to generate a new state or follow an existing successor and, for generation, chooses among the type-valid sampling, reasoning, implementation, and repair operations. Evaluated on LiveCodeBench and CodeContests with Qwen2.5-Coder and DeepSeek-Coder, GraphAHA achieves the best score in 18 of 20 cases. For Pass@1 measured using visible tests, it outperforms the strongest baseline for both models on both benchmarks by 4.1 percentage points on average, demonstrating more effective use of a fixed inference budget. 

---
# Supermartingale Certificates for Parametric MDPs 

**Authors**: Kaushik Mallik, Ðorđe Žikelić  

**Link**: [PDF](https://arxiv.org/pdf/2609.12715)  

**Abstract**: We consider the problems of formal verification and synthesis in parametric Markov decision processes (MDPs) with general measurable state and action spaces. The heart of our approach is a parameter flattening transformation, which allows us to transform parametric MDPs into semantically equivalent non-parametric MDPs. Building on this transformation, we introduce the novel notion of parametric supermartingale certificates, which generalize the traditional supermartingale certificates---used for non-parametric MDPs---to the parametric setting. We use our parametric supermartingale certificates to design algorithms for verification and approximate synthesis in polynomial arithmetic parametric MDPs. This leads to the first verification and synthesis algorithms for parametric MDPs with general state and action spaces. We implement our algorithms and experimentally evaluate them on several continuous parametric random walk benchmarks. 

---
# InRTL: Effective Intra-Inter Interaction Learning for Relational Tables 

**Authors**: Weichen Li, Ken Zhong, Zheng Wang, Li Pan, Jianhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.12712)  

**Abstract**: Relational table learning has recently emerged as an important research direction for modeling multiple tables connected through primary key-foreign key (PK-FK) relationships. Despite recent advances, a principled modeling framework tailored to this task remains underexplored. In this paper, we propose Intra-Inter Relational Table Learning (InRTL), a unified framework that explicitly models dependencies both within and across relational tables. Specifically, InRTL formalizes two complementary interaction patterns: intra-table interactions, describing associations among rows within the same table, and inter-table interactions, describing dependencies between rows across PK-FK-linked tables. To model these dependencies, we develop a column-aware table encoder to generate initial row representations, followed by Transformer-based self-attention and cross-attention modules for intra-table and inter-table learning, respectively. To further improve scalability, InRTL incorporates linearized attention and heterogeneous graph neural networks to simplify the self-attention and cross-attention operations. Extensive experiments on ten datasets covering 24 real-world tasks demonstrate the effectiveness of our approach. Code is available at this https URL. 

---
# What is the Difference Between Me and You? Benchmarking the Quality Gap Between Human-Written and AI-Generated Code 

**Authors**: Cristina Improta, Pietro Liguori, Domenico Cotroneo  

**Link**: [PDF](https://arxiv.org/pdf/2609.12708)  

**Abstract**: AI coding assistants are becoming co-authors of production software, yet their evaluation centers on functional correctness, leaving open whether their code differs from human code in the quality dimensions dominating lifecycle cost. We compare human-written and AI-generated code at scale: 787,562 function pairs across Python, Java, and C, each human function mined from open-source repositories paired with implementations generated from its docstring by three AI assistants (OpenAI GPT models, DeepSeek-Coder, Qwen2.5-Coder). We characterize structural complexity and statistical naturalness, and map static-analysis findings onto Orthogonal Defect Classification for defects and the Common Weakness Enumeration for vulnerabilities, making authors and languages directly comparable. AI-generated code is structurally compressed and stylistically templated: roughly half the size and branching of human code, clustering apart at the style level. Defect profiles differ in kind: human code concentrates issues of mature codebases, AI code repetitive boilerplate; security is language-dependent, with LLMs producing more, and more severe, findings in Python and Java but fewer high-severity memory-safety findings than humans in C. Once size is controlled for, complexity metrics carry little signal, while naturalness separates authors. Finally, we release CQBench, a benchmark of 27,346 issue-prone tasks with baselines and an evaluation pipeline for quality assurance and security testing. 

---
# Separating Engineering Reasoning from DEXPI Serialization in LLM-Based Greenfield Surface-Process Design: A Three-Case Study for Underground Gas Storage 

**Authors**: Qingchuan Zhu, Shuyue Tong, Pengju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2609.12656)  

**Abstract**: Large language models can produce engineering descriptions and structured process representations, but standards-level serialization can substantially increase the generation burden. This diagnostic study examines whether separating engineering reasoning from Data Exchange in the Process Industry (DEXPI) serialization changes where representation and engineering failures occur in constrained greenfield surface-process design for underground gas storage. We compare Direct DEXPI generation with generation of a lightweight Engineering Intermediate Representation (IR) on three cases: single-pressure injection, withdrawal and export, and dual-pressure injection. All six conditions use one fixed model snapshot, qwen3.8-max-0902, with one completed hosted generation per condition. Direct outputs are XSD-valid in 2 of 3 cases, while all 3 Engineering IR outputs are structurally valid under a minimal validator. Direct prompt inputs contain approximately 121.8k-121.9k tokens, compared with 617-699 tokens for the Engineering IR prompts. Engineering feasibility does not uniformly favor the IR: one IR output is rejected for an explicit cooling-state contradiction. The cases also reveal two distinct Direct DEXPI failure modes: engineering inconsistency and standards-level serialization failure. The observed comparison shows that, in these evaluated method bundles, deferring DEXPI serialization substantially reduces representation burden and helps isolate serialization failure, but reducing representation burden alone does not eliminate engineering inconsistencies. 

---
# Explaining Time Series Forecasting with Horizon-Resolved Attribution 

**Authors**: Seunghan Lee, Jun Seo, Jaehoon Lee, Junhyeok Kang, Sangjun Han, Sungdong Yoo, Minjae Kim, Tae Yoon Lim, Dongwan Kang, Hwanil Choi, Soonyoung Lee, Wonbin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2609.12639)  

**Abstract**: Recent advances in explaining time series (TS) models have produced methods that identify which past values a prediction depends on. However, most existing methods return a single importance vector, assuming that every predicted step depends on the same past values. In this paper, we show that this assumption does not hold, as different forecast steps depend on different past values. Motivated by this observation, we propose Horizon-Resolved eXplanation (HRX), which adds a horizon axis to the explanation, so that every forecast step receives its own importance map. HRX is a simple yet effective plug-in framework with three components: 1) an estimator that reads these maps out of any differentiable forecaster without modifying the TS backbone, 2) an evaluation protocol that validates the horizon axis by measuring how much a single forecast step changes when the inputs an importance map ranks highest are removed, and 3) a rank criterion that predicts in advance whether the axis is worth resolving on a given TS. We further show that this step-wise dependence is low-dimensional, as the explanations of all steps are built from a few shared maps whose number does not grow with the forecast length. Extensive experiments across various backbones and datasets show that the improvement comes from the horizon axis and holds for estimators of previous explanation methods. Code is available at this https URL. 

---
# Correlation-Guided Fast Machine Unlearning via Hessian Analysis 

**Authors**: Ayushi Thakur, Ruchir Gupta, Amit Kumar Jaiswal, Prayag Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2609.12620)  

**Abstract**: The increasing adoption of machine learning in network and distributed security systems has created an urgent need for mechanisms that can selectively and efficiently remove the influence of specific training data to eliminate compromised or adversarial data points from production models. Privacy regulations such as GDPR's \emph{right to be forgotten} also pose similar requirements. However, existing approximate unlearning techniques remain computationally prohibitive for deployment in real-world security systems, as they require repeated expensive Hessian-inverse-vector computations for each data point removal, creating a bottleneck when processing multiple related requests in scenarios such as intrusion detection systems, spam filters, and threat intelligence platforms. Thus, we introduce a computationally efficient unlearning framework that identifies correlated data points in the training set and applies a theoretically derived closed-form parameter update rule, achieving an $82\times$ wall-clock speedup over standard influence function unlearning while preserving model utility with a $10^{-2}$ improvement in accuracy over state-of-the-art baselines. Our method establishes theoretical guarantees and ensures numerical stability through Hessian damping. Our evaluation across seven diverse dataset architecture combinations, including large-scale CIFAR-100 with ResNet-50, demonstrates superior forgetting effectiveness, with membership inference attack success rates of 0.660 and tug-of-war scores of 0.950. 

---
# ResoSeg: Resonance Tagger using Transformer and Segment Model 

**Authors**: Chunkai Li, Junhao Yin, Ke Li, Jingde Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.12610)  

**Abstract**: Deep learning has been widely applied across many areas of experimental high-energy physics, yet existing models address only event-level classification or object tagging and therefore still require reconstruction algorithms tailored to each decay channel. We present the first application of segmentation to resonance tagging at BESIII and introduce ResoSeg, a deep learning model that jointly performs particle-level segmentation and event-level classification, enabling a one-pass analysis of resonance to anything decays while precisely reconstructing the relevant resonance properties. We demonstrate the reconstruction of $\eta_c$ with $e^+e^-\to\pi^+\pi^-h_c$, $h_c\to\gamma\eta_c$, $\eta_c\to\text{anything}$. The model is trained on BESIII-$\eta_c$ dataset, which is constructed with per-track true labels obtained via a Truth-Matching Algorithm. Experimental results show that the average combined efficiency of ResoSeg is more than double that of the conventional 16-channel approach across energy points from 4.19 to 4.60\,GeV. The model generalizes to unseen energy points, adapts to other $\eta_c$ production modes through transfer learning, and remains robust against variations in the $\eta_c$ mass, width, and branching fractions, providing a general, resonance-aware model applicable beyond $\eta_c$ and BESIII. The source code is available at this https URL. 

---
# Direct Preference Density Alignment for Conversational Audio Equalization 

**Authors**: Ioannis Stylianou, Sven Ewan Shepstone, Jon Francombe, Pablo Martinez Nuevo, Zheng-Hua Tan  

**Link**: [PDF](https://arxiv.org/pdf/2609.12607)  

**Abstract**: Large Language Model alignment typically relies on learned proxy reward models, which significantly increase the memory footprint during training and are notoriously prone to instability and reward hacking. While offline methods like Direct Preference Optimization (DPO) bypass the reward model, they lose the ability to perform online exploration. If no optimization constraints are applied, this can lead to format collapse in bounded, continuous spaces. To resolve this, we propose Direct Preference Density Alignment: An alternative framework that removes the need for a learned proxy reward model while strictly preserving the benefits of online reinforcement learning. We leverage large-scale user data (approximately 90,000 samples) to construct non-parametric preference density maps, establishing an empirical reward surface. In addition to removing the reward model, Direct Preference Density Alignment enables the combination of the online structural grounding of Group Relative Policy Optimization (GRPO) with the targeted offline refinement of DPO. We show that this GRPO+DPO combination achieves the highest performance, and in a blind audio equalization listening test, enables a 1.5B-parameter model to achieve perceptual parity with a carefully prompt-engineered GPT-4o mini baseline, using only a fraction of the inference compute. 

---
# SIMS: Scale-Invariant Merit-Function-Based Scalarization for Multi-Task Learning 

**Authors**: Zebin Chen, Fei Xing, Yang Chen, Hua Liu, Andy HF Chow, Yuhua Qian, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.12599)  

**Abstract**: Multi-task learning (MTL) requires navigating unavoidable trade-offs among competing objectives. This paradigm is frequently formulated as multi-objective optimization (MOO), where the scalarization is favored to reduce an MOO problem to a single objective. We empirically find that existing merit-function-based scalarization approaches are sensitive to the relative scales of different objectives in practical MTL, where task losses commonly differ by orders of magnitude. The optimization process often favors objectives with larger scales even though the underlying Pareto optimal solutions remains invariant to rescaling (i.e., multiplying an objective by a positive constant). To address this issue, we propose Scale-Invariant Merit-function-based Scalarization (SIMS) for MTL. Specifically, SIMS adopts a transformation-induced merit function to convert the MOO problem of MTL to a single objective that renders optimization invariant to the magnitudes of losses. Theoretically, we prove that the requirement for scale invariance uniquely determines this transformation to be logarithmic. We further show that this general transformation-induced merit function preserves weak Pareto optimality and admits a smooth surrogate with controllable approximation error. Extensive experiments on representative multi-task benchmarks demonstrate that SIMS consistently outperforms existing scalarization methods and achieves state-of-the-art performance. 

---
# Clustering-Based Balanced Sampling and Allocation with Data Parallelism for High-Performance Fine-Tuning 

**Authors**: Hyunjin Kim, Youngeun Nam, Jaemin Han, Wonhyeok Choi, Jae-Gil Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.12584)  

**Abstract**: Instruction-tuning datasets for large language models (LLMs) are often large, redundant, and imbalanced, limiting efficient adaptation. Naive large-batch fine-tuning repeatedly includes overrepresented sample groups while weakly covering underrepresented but informative ones, especially under data parallelism (DP) across multiple GPUs. We propose CluSTER, a Cluster-aware balanced Sampling framework for Training Efficient data Reduction in DP instruction tuning. CluSTER curates a representative reduced dataset through gradient-space clustering and DP-aware balanced allocation, ensuring dual-level coverage across clusters and workers, while preserving the original data distribution by weighted update. As a result, CluSTER reduces redundant computation and improves training stability without compromising model quality. Across multiple instruction-tuning datasets, CluSTER reduces training time by up to 69.6% with almost no accuracy loss compared to prior sampling and data reduction methods. Code is available at this https URL. 

---
# SCOPE-OPSD: Fisher-Conditioned Privileged Subspaces for On-Policy Self-Distillation 

**Authors**: Yunmeng Chen, Kunyu Wang, Peihan Li, Yi Wang, Shuyin Xia, Yi Liu, Xinyong Cheng, Dehui Wang, Xiangyong Zhai, Yanxing Liu, Song Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12579)  

**Abstract**: On-policy self-distillation (OPSD) scores student-generated prefixes with a solution-conditioned self-teacher, yet transfers supervision only through next-token probabilities. We ask whether the aligned final-layer discrepancy offers a useful second channel, and how to test that channel without confusing its geometry with auxiliary strength. SCOPE-OPSD projects the privileged teacher-student residual onto a frozen rank-64 factor estimated from residual covariance and language-model-head Fisher sensitivity. It reuses the forwards already required by OPSD and adds neither rollouts nor inference-time modules. A matched Random control preserves the structured factor's rank and nonzero spectrum and uses per-arm gradient-RMS calibration, isolating the effect of the data-dependent orientation. Across the complete 25/50/75/100-step trajectories for Qwen3-1.7B, 4B, and 8B, Structured is never below Pure OPSD, with strict gains in 11 of the 12 model-checkpoint combinations and an exact tie at 4B step 25. Structured also exceeds matched Random in 10 of the 12 combinations. At step 75 on Qwen3-1.7B, Structured exceeds matched Random by 1.39 Macro Avg@12 points in each of two independent training reruns. A cross-fitted diagnostic also shows 4.40 times greater held-out privileged-gap capture than the matched random orientation. The results support a compact, Fisher-conditioned privileged subspace for short-budget OPSD. 

---
# Calibrated Ambiguity in Multimodal Language Models: Humans reach for cultural references, while models describe the picture 

**Authors**: Cody Kommers, Mingrui Ye, Evelyn Gius, Daniela Mihai, Hoyt Long, Zheng Yuan, Drew Hemment  

**Link**: [PDF](https://arxiv.org/pdf/2609.12575)  

**Abstract**: Ambiguity is often treated as a bug for AI systems to resolve---but in human communication and culture, ambiguity can also be a generative resource. From humour to politics to art, people express themselves in words and images that are open enough to invite different interpretations, yet constrained enough to be interpretable. We operationalise this notion of calibrated ambiguity with a task drawn from the parlour game Dixit. We compare differences in clues generated by human vs multimodal language models, based on a novel coding rubric for calibrated ambiguity, and find that models consistently exhibit ambiguity collapse (i.e., their outputs are over-specified, leaving no room for multiple legitimate interpretations). Unlike human clues, AI-generated clues also exhibit cultural flattening; they almost never make reference to culturally-situated knowledge, even when prompted to use allusion and figurative language. 

---
# TokenMapper: A Step Toward Interoperable Speech Token Translation 

**Authors**: Tal Kozakov, Tal Rosenwein, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2609.12563)  

**Abstract**: Neural audio codecs discretize speech into token sequences, but the resulting token spaces differ in vocabulary and codebook structure, preventing direct communication across models. This limitation affects applications such as conversational voice agents and speech to speech translation systems where multiple speech models must interact. As a result, transferring information between speech systems typically requires decoding to waveform audio and re-encoding with a second tokenizer, increasing latency and introducing potential information loss. To address these limitations, we present TokenMapper, a direction aware framework for direct token to token translation between heterogeneous speech tokenizers in the discrete domain. TokenMapper supports structurally mismatched token spaces, including mappings between single codebook and multi codebook representations, under a shared effective token rate. Experiments on GLM-4-Voice, MiMi and DualCodec show consistent cross model performance. Specifically, translation WER approaches native reconstructions within 2.5-6.8% absolute WER, human MOS for TokenMapper outputs ranges from 2.29 to 4.39, following the same direction level trends as UTMOS and end to end latency is reduced by 4.8-94.5% relative to waveform bridging, reaching up to 972 ms per utterance. These results provide a practical step toward cross model speech token interoperability without intermediate waveform reconstruction. 

---
# RoofLang: Enabling AI-Driven Architecting of LLM Inference Systems 

**Authors**: Ziyue Yang, Yuting Jiang, Lei Qu, Peng Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.12551)  

**Abstract**: AI is beginning to make substantive contributions to LLM inference optimization. Existing AI optimizations are predominantly profiling-based. Profiling-bound feedback confines the search to the capabilities and performance of an existing software stack, preventing a fundamentally better architecture of LLM inference systems from being identified. To enable the AI-driven LLM inference system architecting loop, we argue that a general workload representation, a verifiable mutation space, and an implementation-independent evaluator are required. We present the RoofLang domain-specific language (DSL) that provides these features. In our evaluation, RoofLang reveals that DeepSeek V4-series models could achieve 3.5-39.5$\times$ higher peak decode throughput than other representative models. This gap is disproportionate to their total parameter counts and arises largely from compact KV-cache designs that support larger batches and reduce memory traffic. A persistent optimizer agent further discovered several new architectures that improved both throughput and interactivity of DeepSeek V4 Pro on NVIDIA B300 by 6.23-50.1%. 

---
# Meddies-PII: A Multilingual Framework for Personally Identifiable Information Extraction in Clinical De-identification 

**Authors**: Linh Uyen Le, Christian Hoang, Huy Hoang Ha  

**Link**: [PDF](https://arxiv.org/pdf/2609.12544)  

**Abstract**: Clinical de-identification relies on accurately identifying personally identifiable information (PII). However, manually annotated datasets are costly to construct, while existing synthetic alternatives often provide limited details about their generation process or rely on relatively simple synthesis strategies. We introduce Meddies-PII-Dataset, a corpus of one million synthetic clinical documents spanning seventeen languages and nine PII labels. The documents are generated using attribute-conditioned prompts and validated through thirteen deterministic gates that enforce structural and annotation consistency. To evaluate the dataset's utility, we train Meddies-PII-Model, a BIOES token classifier, and compare it with existing PII extraction systems using exact-match entity-level F1. Meddies-PII-Model achieves the highest performance among the evaluated systems on all reported benchmarks, with a mean F1 of 0.827 across fifteen external benchmarks, compared with 0.658 for the strongest baseline. Upon acceptance, we will publicly release the dataset, benchmark suite, model, generation framework, and evaluation code to support research on multilingual clinical de-identification. 

---
# Earth-Agent-Pro: Towards Real-World Full-Chain Earth Observation with Agents 

**Authors**: Zhutao Lv, Chenhao Dang, Yi Feng, Yanpei Gong, Xiaolei Wang, Junyan Ye, Conghui He, Weijia Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.12533)  

**Abstract**: Real-world Earth observation (EO) agents must translate high-level scientific questions into executable workflows to acquire observations, prepare data, perform domain computations, and derive conclusions from runtime evidence. Existing EO agents typically start from supplied observations, while benchmarks typically provide prepared inputs or candidate answers, leaving full-chain open-world EO execution largely untested. We present Earth-Agent-Pro, an execution-adaptive Plan-and-Execute framework using expert-authored skills to constrain planning and runtime tool use. Workflow-centered structured memory records planned steps, accepted evidence, and their dependencies, enabling repair of only the affected workflow suffix when runtime evidence invalidates a step. Separate large language model adapters use sequence-level supervised fine-tuning for planner workflow composition and node-level group relative policy optimization with locally verifiable rewards for executor tool-argument grounding. Earth-Bench-Pro instantiates 248 expert-curated task cores as 744 questions under three matched regimes. Its 248 Open-World Execution questions span RGB imagery, spectral observations, and remote sensing products, pairing high-level requests with runtime data requirements, executable trajectories, and open-ended answers grounded in execution evidence. With a shared GPT-5 backbone, Earth-Agent-Pro achieves 66.13% LLM-as-Judge accuracy, exceeding ReAct by 20.95 points in this metric and 24.44 points in Tools-In-Order. Joint adapter tuning raises Qwen3.5-9B LLM-as-Judge accuracy from 38.31% to 50.00%, an 11.69-point gain over the untuned configuration. Planning-only evaluation and execution with the reference workflow show that the adapters improve workflow composition and argument grounding, respectively. Code and datasets will be released soon. 

---
# Optimizing Geoengineering Interventions Using Differentiable Climate Models 

**Authors**: Pulkit Dubey, Dorian S. Abbot, Ashesh Chattopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2609.12528)  

**Abstract**: The deployment of a geoengineering program to cool Earth's climate may be imminent. It is crucial that tools be developed to ensure that such a program would achieve its objectives while minimizing disruption. Here we exploit recently developed differentiable atmospheric models to demonstrate a novel geoengineering control strategy. In the differentiable primitive-equation atmospheric model JAX-GCM we impose a uniform $+4$\,K ocean warming and ask what pattern of sea-surface temperature cooling -- in five ocean-masked zonal bands of prescribed SST forcings whose amplitudes are free -- returns land near-surface air temperature closest to the model's own unwarmed climatology. This idealized set-up represents a cooling pattern that could be delivered physically either by marine cloud brightening or stratospheric aerosol injection. Gradients through chaotic dynamics decorrelate from the true sensitivity beyond the Lyapunov horizon, so we optimize greedily over segments of 8 to 14 days, following receding-horizon control. The learned strategy removes $92.3 \pm 0.4\%$ of the realized land warming across a ten-member ensemble of two-year rollouts, and a three-year run sustains it. If we use the spatial pattern of land temperature as the optimization objective, the distributions of precipitation, evaporation, and specific humidity over land are restored as well, even though they are not included in the objective function. The learned strategy from JAX-GCM replayed in the AI emulators LUCIE and NeuralGCM without re-optimization is successful, suggesting robustness. These promising results demonstrate a strategy for designing optimal climate interventions that can be applied broadly for geoengineering scenarios under consideration. 

---
# Computing at Sea: Floating and Offshore Data Centres as a Pathway to Sustainable AI Infrastructure 

**Authors**: Cheng Siong Chin, Jianhua Zhang, M. Venkateshkumar  

**Link**: [PDF](https://arxiv.org/pdf/2609.12511)  

**Abstract**: The rapid expansion of artificial intelligence is transforming data centres into one of the world's fastest-growing sources of electricity demand. As AI systems scale in size and capability, the physical infrastructure supporting computation is approaching critical limits in energy availability, cooling capacity, land use, freshwater consumption, and carbon management. Conventional land-based data centres are increasingly constrained by urban land competition, grid congestion, environmental pressures, and lengthy permitting processes, raising fundamental questions about where future computing infrastructure can sustainably exist. This article examines floating and offshore data centres as an emerging alternative model for digital infrastructure. By relocating computation to marine environments, offshore systems can exploit the ocean's natural cooling capacity, reduce freshwater dependence, and enable direct integration with offshore renewable energy resources such as wind, wave, and tidal power. Early deployments have demonstrated the potential for significantly improved energy efficiency and operational reliability compared with conventional facilities, while also opening new possibilities for distributed and resilient computing architectures. The article explores how offshore computing may reshape the future relationship between electrification, renewable energy, and large-scale AI infrastructure. It analyses the opportunities and trade-offs associated with marine deployment, including environmental impacts, engineering design challenges, economic feasibility, and regulatory governance. Rather than treating offshore data centres as experimental novelties, the article presents them as part of a broader systems-level transition in how society may power, cool, and sustain the next generation of computational growth. 

---
# LettuceVisSim: A Simulator That Generates Lettuce Image Time-series for Vision-Based Reinforcement Learning 

**Authors**: Ziye Zhu, Bert van 't Ooster, Congcong Sun, Eldert van Henten, Sjoerd Boersma  

**Link**: [PDF](https://arxiv.org/pdf/2609.12505)  

**Abstract**: Vision-based reinforcement learning holds strong potential for decision-making in controlled environment agriculture (CEA). However, its development is hindered by the scarcity of labelled crop images. To address this gap, LettuceVisSim, a lettuce growth simulator that generates labelled time series of crop images, was developed and validated. The simulator contains a process-based model (PBM) for shoot dry weight dynamics, a canopy layout algorithm for deriving canopy layout representations from shoot dry weight, and a Unity rendering engine for image generation. Five findings support the simulator. First, the PBM reproduced shoot dry weight under dynamic plant-density management with $\mathrm{R}^{2}=0.84$. Second, a piecewise cubic regression mapped shoot dry weight to potential projected area with $\mathrm{R}^{2}=0.94$. Third, the canopy layout representation was validated using 12 experimental datasets each having different dynamic environmental and spacing conditions. It reproduced the ground coverage ratio dynamics observed in measured images, achieving $\mathrm{R}^{2}=0.84$ when driven by measured shoot dry weight and $\mathrm{R}^{2}=0.40$ (0.76 excluding one outlier) when driven by PBM-simulated values. Fourth, the Unity rendering engine converted canopy layout representations into RGB and segmentation images at less than 10~ms. Fifth, a demonstration showed that a lighting-control policy can be learned and applied by observing only crop images that were generated with LettuceVisSim, providing a proof of concept of vision-based reinforcement learning in CEA using LettuceVisSim. 

---
# Adaptive Agent Design 

**Authors**: Raj Kiriti Velicheti, Subhonmesh Bose, Tamer Başar  

**Link**: [PDF](https://arxiv.org/pdf/2609.12486)  

**Abstract**: We consider an agent acting against a general non-Markovian environment. The agent maintains its agent states, but is free to choose a transition kernel across those states and optimize its state-feedback control policies. We study the bi-level agent design problem that optimizes the transition kernel and the policy it induces, given said kernel with offline data of observations and actions obtained via a behavioral policy. For general environments, we show that a soft $Q$-learning algorithm converges almost surely to the fixed point of a soft Bellman equation defined by the stationary averages that the behavioral policy and the chosen kernel induce, and we delineate what separates the resulting policy from an optimal one. In partially observed Markov decision problems, we analyze convergence properties of parametrized transition kernel design via zero-th order and Bayesian optimization techniques. 

---
# Not All Speech Is Intent: Adaptive Self-Correcting Inference Layer for Post-ASR False Wake-Up 

**Authors**: Preeti Saraswat, Divya Neelagiri, Anil Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2609.12469)  

**Abstract**: False wake-up activations remain a persistent challenge in conversational AI. Speech phonetically similar to a device's wake word can produce a syntactically valid and semantically coherent ASR transcript that the assistant incorrectly executes. Most existing systems make a single intent decision in isolation, without a mechanism to learn from recurring errors over time or adapt to individual users through personalized learning. We introduce the Feedback-Driven Adaptive Self-Correcting Inference Layer (ASCIL), a complementary post-ASR correction framework that re-evaluates wake-up intent before response generation by fusing acoustic embeddings, linguistic cues, device context, and patterns from past misclassifications. ASCIL interprets implicit signals, including hesitation, disengagement, and silence, and explicit signals, including cancellation and repetition, as automatically inferred, noisy behavioral indicators of potential misclassification. These signals drive online pattern updates without manual annotation, whereas the intentional/unintentional reference labels used for offline evaluation are human-annotated. It generalizes from prior errors, applies corrective adjustments at inference time, and continuously updates in parallel with natural-language execution. Evaluated on a proprietary dataset of 3,667 interactions with human-annotated intentional/unintentional reference labels spanning 14 acoustic and contextual conditions, ASCIL achieves 54.27% relative error reduction on a session-disjoint subset constructed from baseline failures, and up to 24.39% relative error reduction at threshold 0.90 on the issue-tagged evaluation slice. These gains are achieved while improving intentional acceptance rates, with a median added latency below 60 ms in the reported benchmark. 

---
# Linear Exponential Quadratic Gaussian Covariance Steering 

**Authors**: Chiran B. Cherian, Yasemin Isik, Abhishek Halder  

**Link**: [PDF](https://arxiv.org/pdf/2609.12463)  

**Abstract**: We formulate and analyze the linear exponential quadratic Gaussian (LEQG) covariance steering problem in continuous time over a given deadline (finite time horizon). The solution for this problem can be seen as a risk-sensitive Schrödinger bridge between Gaussian endpoints in the linear quadratic setting. Unlike the risk-neutral case, the LEQG covariance steering controller--still a linear state feedback--can no longer be written in closed form. We show that the optimal controller is parameterized by a symmetric matrix solving an algebraic equation that encodes the implicit dependence on the risk-sensitivity parameter. We explain how the structure of this optimal controller significantly generalizes the existing results for the risk-neutral case. Building on these results, for the matched noise and input channel case, we prove the existence-uniqueness of solution for the LEQG covariance steering problem in the neighborhood of the known risk-neutral optimal solution. We give an illustrative numerical example. 

---
# Bridging Vision Foundation Model Priors with CLIP for Spatial-aware Few-shot Anomaly Detection in Medical Images 

**Authors**: Juzheng Miao, Yuchen Yuan, Cheng Chen, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2609.12454)  

**Abstract**: Vision-Language Models such as CLIP enable effective few-shot medical anomaly detection (AD) via strong image-text semantic alignment. However, their globally contrastive pretraining lacks explicit spatial supervision, limiting precise lesion localization. In contrast, Vision Foundation Models (VFMs) such as DINO learn spatially coherent patch representations via self-distillation and local-to-global consistency, better capturing fine-grained anatomical structures. Leveraging this complementarity, we propose Spatial-FAD, a spatial-aware few-shot medical AD framework that improves lesion localization by combining VFM spatial priors with CLIP semantics. Specifically, we introduce a VFM-enhanced adapter that injects a structural affinity prior derived from DINO into CLIP features. This structure-guided refinement encourages visual embeddings to better adhere to lesion boundaries while maintaining semantic alignment. To address the loss of spatial detail from patchification and the limited input resolution of CLIP, we adopt a sliding-window aggregation strategy. This generates high-resolution, spatially dense embeddings to further enhance localization granularity. Moreover, we introduce a prototype-enhanced support memory scheme to efficiently exploit the few-shot support set. This module stores compact prototypes for normal and abnormal patterns, reducing memory costs while boosting performance by fusing patch-to-prototype and image-text similarities. Extensive experiments on three benchmark datasets, including Liver CT, Retinal OCT, and Brain MRI, demonstrate that Spatial-FAD significantly outperforms state-of-the-art methods, especially in lesion segmentation. Notably, in the 4-shot scenario, our method achieves an average improvement of over 11.4% in Dice score and 1.8% in AUC. Code is available at: this https URL. 

---
# 3D Digital Twin Visualization of Multiclass GRF-Based Gait Disorder Classification 

**Authors**: Nayoung Son, Minwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2609.12442)  

**Abstract**: Automated gait analysis requires accurate classification and interpretable outputs. We propose an integrated framework for classifying healthy gait and multiple musculoskeletal impairment groups using bilateral ground reaction force (GRF) and center-of-pressure (COP) signals. The signals were normalized over the stance phase and standardized using training-set statistics. The model achieved a validation accuracy of 99.00\% and a test accuracy of 90.07\% under a session-level split. Class-specific $\epsilon$-LRP identified positive and negative contributions across both sides, multiple signal components, and different stance phases. Separately, the processed GRF signals and model predictions were synchronized within a Blender-based 3D visualization, enabling sample-level inspection of gait trials and classification results. The proposed framework integrates classification, explainability, and 3D visualization to improve model transparency. The source code is available in the following repository: this https URL 

---
# IMPLY: Physically Anchored Consistency for World-Model Rollouts 

**Authors**: Aman Mehta, Riya Baviskar  

**Link**: [PDF](https://arxiv.org/pdf/2609.12441)  

**Abstract**: A world model asked what happens if an object is pushed at several speeds produces several futures. If the model has the object in mind, those futures agree about it: each implies the same mass and friction. The consistency checks now used to vet world-action models ask whether a model's futures agree with each other, and none of them knows any physics. We show that this is not enough, and what to do instead. IMPLY reads the physics each rollout implies by inverting a simulator and scores a set of rollouts by how well one object explains all of them, anchored to two calibration pushes the model has observed. In a controlled setting, self-consistency gives a perfect score to a model that ignores the object and always predicts a typical push; anchoring exposes it (AUROC 0.70 versus 1.00). On a real model, V-JEPA 2-AC adapted to the scene, the same thing happens. Given its own calibration pushes the model tracks the object (per-object correlation with the truth 0.91); given another object's, it does not (0.05). Self-consistency cannot tell these apart, preferring the right evidence on 52% of objects, chance level, while anchored disagreement prefers it on 73% and correlates 0.92-0.99 with the rollouts' error. Used to choose among candidate rollout sets, it comes within 0.003 of an oracle that sees the truth. A model that has internalised the wrong object is exactly as self-consistent as one that has internalised the right one; consistency has to be anchored to evidence. 

---
# Debiasing as a Measurement Intervention: Calibrated Ties and Resolution Loss in LLM-as-a-Judge Evaluation 

**Authors**: Liang Zhao, Yong Wang, Jiangzhe Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.12439)  

**Abstract**: LLM-as-a-judge protocols are commonly debiased by instructing judges to ignore presentation cues such as citation formatting, source labels, and evidence-display style. We show that this intervention can suppress bias while damaging the resolution of the measurement instrument. We introduce TraceJudgeBench, a diagnostic benchmark for auditing citation-like artifacts in RAG and agent-workflow evaluation, covering content-equivalent pairs, citation ablations, correctness conflicts, human-validated soft and moderate quality gaps, prompt-strength ladders, decoupled judging, and a controlled workflow-ranking probe. Across GPT-5.5, Claude Sonnet 4.6, and DeepSeek V4-Flash, stronger anti-citation prompts reduce worse-cited wins from up to 50.5% to 0%; yet some operating points already convert validated moderate-gap decisions into Tie before the strict stress-test endpoint, while correctness-conflict accuracy remains at or above 93.0%. A second, 50-pair FinQA moderate-gap construction reproduces the qualitative frontier, and open-weight Qwen2.5-14B-Instruct-AWQ and Gemma-3-12B-IT runs reproduce the central HotpotQA frontier. TRACE-style decoupling recovers 96.5-100.0% better-plain resolution across the reported settings. Human validation separates three meanings of Tie: correct equivalence Tie, calibrated soft-boundary Tie, and resolution-destroying Tie on validated quality gaps. We frame debiasing as a measurement intervention whose bias suppression, resolution retention, Tie cost, and protocol cost must be reported jointly. The supplementary artifact contains benchmark splits, prompts, raw judge outputs, validation summaries, and analysis. 

---
# Beyond the Query: Do Retrieval Signals Improve Adaptive Multimodal RAG Routing? 

**Authors**: Qiaomu Li, Qiuyuan Zhang, Nong Ming  

**Link**: [PDF](https://arxiv.org/pdf/2609.12437)  

**Abstract**: Adaptive RAG often uses retrieval-time signals to decide whether another retrieval, reranking, or multimodal step should run. We ask whether these signals add routing value once the query itself is already known. Across document, audio, and video RAG, we compare matched query-only and query+retrieval routers while holding the optional actions, router family, training procedure, and evaluation fixed. On the held-out final evaluation, adding the tested retrieval signals does not produce a reliable routing improvement over the query-only baseline. Some retrieval signals are associated with whether a later step will help, but that predictability does not consistently lead to bet- ter RUN/SKIP decisions. The main lesson is therefore methodological: retrieval-state features should not be credited with routing value unless they improve over a matched query-only control. Our results do not show that routing or retrieval state is generally useless; they show that the incremental value of retrieval signals must be demonstrated rather than assumed. 

---
# Observation-Anchored Selective Assimilation for Longitudinal Tumor-State Proxy Forecasting in Post-Treatment Glioma 

**Authors**: Yeonjae Jung, Minwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2609.12435)  

**Abstract**: Post-treatment MRI in patients with glioma provides serial observations for updating patient-specific tumor-state proxy estimates, but variable appearances and trajectories complicate forecasting. We formulate forecasting as an observation-aware digital-twin update in which an intermediate observation anchors the patient-specific state. Among 203 patients and 594 follow-up time points, a predefined no-new-treatment criterion retained 120 of 236 candidate triplets, split into 81/24/15 training/validation/test triplets at the patient level. Each time point was represented by a continuous voxel-wise tumor-state proxy map in [0,1] derived from MRI lesion labels. A SegMamba-based single-step forecaster predicted update proposals from multimodal source-state tensors. Observation-Anchored Selective Assimilation (OASA) retained the observed intermediate proxy as the state anchor and selectively applied updates through a validation-selected tiered case-level rule and voxel-wise soft gate. We compared initial-scan forecasting, rollout without assimilation, latest-observation persistence, direct prediction, OASA, OASA + calibration, and morphological dilation. Checkpoints, OASA rules, and calibration thresholds were selected using validation data only. Across three seeds on 15 held-out test triplets, OASA maintained Dice at $\tau$ = 0.2 comparable to persistence (0.6071 $\pm$ 0.0025 vs. 0.6070) while yielding numerically higher Dice at $\tau$ = 0.5 (0.4269 $\pm$ 0.0079 vs. 0.3981), with a small RMSE increase. Calibration increased Dice at $\tau$ = 0.2 to 0.6178 $\pm$ 0.0025, increased false-positive (FP) support (11,836$\rightarrow$18,663), and reduced false-negative (FN) support (22,107$\rightarrow$17,536). This reflects near-threshold support calibration rather than improved biological predictive capability. Code is publicly available at this https URL. 

---
# MInTRL: Off-policy Intervention can boost On-policy RL 

**Authors**: Mingyu Chen, Yefan Tao, Gerald Friedland, Xuezhou Zhang, Chris Kong  

**Link**: [PDF](https://arxiv.org/pdf/2609.12419)  

**Abstract**: Reinforcement learning with verifiable rewards is typically performed on-policy, keeping training data close to the current policy but limiting learning to trajectories that the policy can discover itself. Off-policy methods such as supervised fine-tuning, on the other hand, can leverage external knowledge beyond the base model's capabilities, but may suffer from large distribution shift. The key challenge is thus to expand exploration without sacrificing learnability. In this work, we introduce Minimal Intervention Reinforcement Learning (MInTRL), which expands the exploration frontier through sparse, local interventions in otherwise on-policy rollouts. During generation, a judge-intervention policy periodically reviews the current policy's output, replaces erroneous suffixes with short corrections, and immediately returns control to the policy. During training, MInTRL adopts a sequence-level advantage-regression objective that eliminates the need for importance sampling. We show that sparse, local interventions can substantially improve coverage beyond finite-budget on-policy sampling while preserving the overall on-policy nature of the resulting trajectories. Across math and code benchmarks, MInTRL consistently outperforms standard on-policy and off-policy baselines. Ablations show that MInTRL remains effective with self-intervention and across different judge policies, while performance peaks at moderate intervention intensity, highlighting the importance of intervening minimally. These results establish minimal intervention as an effective paradigm for enhancing on-policy RL. 

---
# UFO: Chain-of-Evaluation for Omni-Condition Alignment in Multi-Modal Image Generation 

**Authors**: Danning Zhang, Yijing Lin, Shuhan Zhuang, Mengqi Huang, Shaojin Wu, Shancheng Fang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2609.12397)  

**Abstract**: Multi-modal image generation, particularly subject-driven customization, has garnered growing attention in recent years. Despite the rapid advancement of generative models, their evaluation remains largely lagging. Existing methods, whether embedding-based or Multi-modal Large Language Model (MLLM)-based, evaluate alignment with each modal condition in isolation, which contradicts the simultaneous condition alignment objective of multi-modal image generation, leading to poor consistency with human judgments. To address this challenge, we propose UFO, the first unified framework for omni-condition alignment simultaneous evaluation. Specifically, UFO introduces a novel Atomized Chain-of-Evaluation paradigm, \emph{i.e.}, it first decomposes omni-condition alignment into a sequential chain of fine-grained, disentangled Atomic Evaluation Units (AEUs), categorizes them into distinct modality-relevance classes, and then employs general or dedicated functional calls for accurate verification of different AEU types. Experimental results demonstrate that UFO achieves the highest correlation with human evaluation preferences, delivering an average improvement of 15.25\%. Furthermore, we present UFO-Bench, a dedicated benchmark designed to holistically evaluate the performance of existing customization models under the diverse mutual interactions of textual and visual conditions. 

---
# RF-VoID: Towards Bandwidth-Efficient Exterior Tile Void Detection via Narrowband Radio-Frequency Representation Learning 

**Authors**: Xinyan Chen, Ruiqin Ma, Shunsuke Shoda, Changyu Zhou, Ryo Natsuaki, Akira Hirose, Jianfei Yang, Li Yi  

**Link**: [PDF](https://arxiv.org/pdf/2609.12388)  

**Abstract**: Hidden debonding behind exterior ceramic tiles is a falling-tile hazard, and millimeter-wave radar offers a non-contact way to find it. Conventional interpretation first reconstructs a range profile, so its reliability is bounded by the available bandwidth, yet bandwidth is what sets the cost, the acquisition time, and the regulatory footprint of a deployed system. This work asks whether that bandwidth can be traded for computation. A 4-40 GHz stepped-frequency system scans twelve exterior-wall specimens containing 0.5-1.0 mm air voids at different depths and interfaces, and the bandwidth dependence of A-scan, B-scan, and C-scan interpretation is analyzed to establish the resolution bound. RF-VoID is then proposed, which decides directly on the narrowband complex response: the sub-band is kept in its measured frequency order with amplitude and phase alongside the in-phase and quadrature channels, a dual-branch encoder reads it along the physical frequency axis using relative position encoding and a distance-dependent locality bias, and an inspection-oriented objective handles the class imbalance and the asymmetric error cost of facade screening. Under a mixed-sample protocol the method attains 98.61% accuracy and a 95.84% F1-score with 0.5 GHz of bandwidth, a seventy-two-fold reduction relative to the full sweep, without range-profile reconstruction, deconvolution, or depth-slice selection; on specimens held out entirely from training it remains the strongest of the compared models, with a mean macro F1-score of 62.12% at 0.5 GHz that rises to 68.57% at 1 GHz. 

---
# ORQA: An Occupation-Realistic Question and Answer Framework for LLM Professional Knowledge 

**Authors**: Shreyas Krishnan, Serina Chang, Abhishek Nagaraj  

**Link**: [PDF](https://arxiv.org/pdf/2609.12366)  

**Abstract**: We present ORQA, a method for testing occupation-level knowledge in large language models. Prior methods either map abstract LLM skills to occupations via task definitions or utilize expert knowledge which is difficult to obtain at scale and expensive. ORQA complements both of these methods by connecting O*NET occupations to trusted occupation-specific websites (such as regulatory agencies, licensing bodies, professional organizations, and government publications) and converting these into source-traceable question-answer pairs. A combination of an automated pipeline and human review produces a set of high quality questions about occupations. The question set created via our method covers 116 occupations from all 21 major groups in the SOC, with 480 questions sourced from 187 different websites. Each question is designed to probe a real-world skill question that is relevant to the occupation in question. We test 15 state-of-the-art frontier and open-weight models via this method. Claude Opus 4.6, GPT-5.4 and Claude Sonnet 4.6 all perform the best at approximately 58-62% while smaller open-weight models achieve approximately 33-41% performance. Performance varies significantly across occupations. Healthcare-related occupations achieve the highest performance (78%) while Office and Administrative Support achieve approximately 40%. Performance on individual occupations (e.g. Sheet Metal Workers and Fish and Game Wardens) is essentially zero. We also find that open-ended questions and weighting by wage bill do not significantly affect the ranking of models on this benchmark. We believe that leveraging existing trusted occupation-specific information to test LLM knowledge in professional domains may be a scalable and useful method for evaluating occupation-level AI performance in the future. Results and data are available at this http URL. 

---
# ESTS at WMT26: Routing-Informed Expert Pruning for Model Compression 

**Authors**: Liu O. Martin, Lucas Bandarkar, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2609.12310)  

**Abstract**: We describe six submissions under the team name ESTS to the unconstrained WMT26 Model Compression Shared Task for English--Simplified Chinese and English--Egyptian Arabic. We submit three compression operating points per translation direction, all derived from GPT-OSS-20B. We use task-specific routing mass to rank experts and cross-lingual routing divergence to allocate retained capacity across layers, then physically remove low-importance experts. The resulting specialists are recovery-tuned on GPT-5.1-generated synthetic translation data and further compressed by applying MXFP4 quantization to the retained expert projection weights. We additionally implement a robust inference system for the instruction-conditioned WMT26 setting, including category inference, output validation, retries, segmented fallback, and source-owned JSON reconstruction. Across our six submissions, parameter counts range from 4.186B to 7.770B and packed artifact sizes from 4.55 to 6.33~GiB. Internal xCOMET-XL evaluation using GPT-5.1 pseudo-references provides an internal comparison across the submitted compression operating points. 

---
# Self-Verifying Anomaly Detection using Explainable AI for Cybersecurity of DER Networks 

**Authors**: Damilola Popoola, Souradeep Bhattacharya, Manimaran Govindarasu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12305)  

**Abstract**: The rapid growth of Distributed Energy Resources (DERs) has significantly expanded the cyber attack surface of modern power grids. Furthermore, increasing sophistication in attack techniques demands anomaly detection systems (ADS) that are accurate, interpretable, and reliable to support DER cybersecurity. While ML-based ADS provide strong detection capabilities, their black-box nature reduces operator trust and limits Security Operation Center's (SOC) ability to effectively interpret alerts and respond, highlighting the need for explainable Artificial Intelligence (XAI) to ensure transparency and operational confidence. This paper presents an XAI-based anomaly detection framework tailored for DER networks (ExCYDER). The proposed framework uses a self-verifying mechanism that validates ADS alerts to ensure trustworthy decision-making. ExCYDER combines LightGBM with SHAP to check whether each model decision aligns with its feature-attribution evidence, allowing the system to confirm that its internal reasoning is consistent and reliable. Experiments on a realistic DNP3 dataset achieved over 98% detection accuracy, an average rule--SHAP consistency of 44.6%, a SHAP latency of 14.5 ms per alert, and a confidence deviation within 5%, demonstrating stable verification behavior with minimal computational overhead. The framework distinguished between coherent and inconsistent alerts without compromising detection accuracy, demonstrating that integrated verification within XAI-based ADS enhances interpretability, auditability, and operational robustness for DER-focused SOCs. 

---
# Breaking the Token Ceiling: Distilling Smaller, Stronger Byte Models 

**Authors**: Kalyani Marathe, Artidoro Pagnoni, Tomasz Limisiewicz, Margaret Li, Mike Lewis, Luke Zettlemoyer, Srinivasan Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2609.12303)  

**Abstract**: Small models are made more capable through distillation from a larger one that shares their tokenization scheme. However, do distilled byte and token models behave similarly in terms of scaling trends as compute and data increases? To enable this comparison, we introduce two variants to efficiently convert token logits to Byte Logits: 1) approximate: Marginalize-It, and 2) exact: End-Of-Token. We then present the first large scale study of overtraining decoder-only dense transformer models varying two dimensions simultaneously: the tokenization scheme (Tokens, Bytes, Bytes w/ eot) and the training objective (Distillation vs. Cross-Entropy), sweeping layer-parameter-matched models with roughly 1 billion parameters up to 1 trillion bytes of data. Across eight benchmarks spanning three categories: Multiple Choice QA, Language Generation, and Machine Translation, we find that Token-1B models outperform byte models (End-Of-Token-1B and Bytes-1B) in the low-FLOP regime but eventually plateau; byte models start worse yet surpass Token-1B models with more compute, reaching a higher downstream task performance ceiling. Extrapolating the average top-1 error vs. validation BPB scaling laws predicts that, asymptotically, distilled End-Of-Token-1B outperforms distilled Token-1B by up to 4%. They are also far more data efficient, matching the performance of distilled Token-1B using only one-sixth of the training data. Moreover, by operating over a small vocabulary of 256 bytes instead of on the order of 100K tokens, they circumvent the need for top-k truncation during logit dumping, while also reducing logit storage costs to roughly one-fifth. Finally, our downstream performance scaling laws predict that our distilled End-Of-Token-1B models asymptotically surpass the Llama 3.2-1B, Gemma-3-1B-pt, and Gemma 2B models on averaged downstream tasks by up to 6.5%, 8.1%, and 2.1%, respectively. 

---
# Amortized Low-Rank Adaptation for Model-Based Reinforcement Learning 

**Authors**: Fernando Palafox, David Fridovich-Keil  

**Link**: [PDF](https://arxiv.org/pdf/2609.12278)  

**Abstract**: World models let agents plan by predicting the consequences of their actions, but changes in the environment can make them inaccurate. We study the problem of adapting a world model to an unknown test-time environment, drawn from a known environment family, using only a few episodes of interaction. Existing approaches trade off computational cost against expressivity, i.e., the range of models a method can produce. For example, in-context learning is computationally cheap but limited in expressivity, and gradient-based adaptation is expressive but computationally expensive. We present CLAW (Context-conditioned Low-rank Adaptation of World models), which addresses this tradeoff by using a hypernetwork to generate low-rank (LoRA) adapters at test time. During pretraining, we simulate adaptation to a variety of environments and jointly train the hypernetwork and base world model. At test time, we freeze the base model and use a forward pass of the hypernetwork to generate adapters from a small batch of test-time transitions. We evaluate CLAW in locomotion and manipulation environment families that vary in dynamics, embodiment, and reward. We show that, using only seconds of test-time data, CLAW outperforms gradient-based adaptation and in-context learning during online adaptation. We also show that CLAW avoids overfitting in data-scarce regimes, that its advantage comes from the expressive adapters rather than context conditioning, and that pretraining the hypernetwork jointly with the base model outperforms training it post hoc. 

---
# Reinforcement Learning over Patient Trajectories for Clinical Reasoning in EHR Foundation Models 

**Authors**: Yuxin Xiao, Sheng Zhang, Chandan Singh, Tristan Naumann, Hoifung Poon, Jianfeng Gao, Xiaodong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12277)  

**Abstract**: Electronic health record (EHR) foundation models trained on longitudinal patient trajectories have demonstrated strong performance across diverse clinical prediction tasks. However, their clinical reasoning capabilities remain constrained by next-token prediction on limited and incomplete EHR data. To address this, we propose a reinforcement learning (RL) fine-tuning framework that treats EHR foundation models as generative policies over patient trajectories. We formulate common clinical prediction problems (e.g., hospital readmission) as event-conditioned, time-windowed reasoning tasks. We then design time-aware, rollout-sensitive rewards to account for finite rollout lengths and temporally inconclusive outcomes. We find that RL fine-tuning consistently improves over pre-trained backbones and strong baselines. Notably, it enables smaller models to surpass larger pre-trained models in data-limited regimes and induces positive transfer across tasks. Further analysis shows that RL fine-tuned models generate trajectories with stronger structural and semantic alignment to ground truth and greater downstream utility. 

---
# Recommendation Retrievers Need Verifiers: Universal Generative Reranking for Sequential Recommendations 

**Authors**: Benyu Zhang, Qiang Zhang, Rui Li, Qunshu Zhang, Devansh Tandon, Neeraj Bhatia  

**Link**: [PDF](https://arxiv.org/pdf/2609.12270)  

**Abstract**: First-stage recommenders in multi-stage systems produce a ranked candidate list from which a limited prefix is forwarded to downstream rankers. Because each forwarded item must be processed by more expensive ranking stages, this shortlist cannot be arbitrarily large. The first-stage objective is therefore high coverage of relevant items within the forwarded prefix, commonly measured by Recall@$k$. A relevant item may be available deeper in the retrieved list but absent from the shorter prefix that is actually consumed. This paper studies post-hoc verification for promoting such candidates into the consumed shortlist without retraining or replacing the retriever.
We introduce a lightweight generative verifier for retrieval models. Given a retriever state and a candidate item, the verifier scores the item through the likelihood of its identifier tokens. It is trained post hoc with next-token cross entropy, requires no sampled negatives or candidate pool during training, and scores only the retriever's top-$K$ candidates at inference. The interface is minimal: the retriever supplies a query state and candidate items, and the item representation can use any fixed tokenization.
Across Amazon product recommendation and YaMBDa music recommendation, the same verifier training recipe improves Recall@10 for SASRec, GRU4Rec, NextItNet, and MiniOneRec. Ablations show that the improvements are not explained solely by injecting item-content features into the retriever, supporting verification as a post-hoc output-side adaptation mechanism. 

---
# HypoKG: Evidence-Disciplined Biomedical Hypothesis Generation Beyond Endpoint Knowledge 

**Authors**: Dominic Okonkwo, Adetayo Okunoye, Ismailcem Budak Arpinar  

**Link**: [PDF](https://arxiv.org/pdf/2609.12260)  

**Abstract**: Large language models (LLMs) can generate biomedical hypotheses, but it remains unclear whether they truly reason from scientific evidence or simply produce convincing-sounding ideas. To study this, we combine three major biological databases: the Kyoto Encyclopedia of Genes and Genomes (KEGG), Rhea, and UniProt, into a unified biochemical knowledge graph and construct a benchmark of 550 paths connecting enzyme sources to rare disease endpoints, yielding 13,200 hypotheses from six LLMs under four conditions varying the biological information each model receives: source enzyme only, full biological path, or source and disease endpoint only. Hypotheses are scored using an expert-derived five-criterion rubric on a 1-5 scale per criterion. We find that models given both the source and disease endpoint often produce the highest-scoring hypotheses, showing that LLMs can generate compelling ideas from minimal information. However, these hypotheses are less grounded in the evidence. In contrast, models given the full biological path generate hypotheses more consistent with known mechanistic relationships. We call this evidence-disciplined reasoning. To confirm this effect, we shuffled intermediate path steps while keeping endpoints fixed. Evidence grounding dropped significantly (delta = -0.793, p < 0.001), confirming models genuinely used path structure during reasoning. Our findings show that knowledge graphs support hypothesis generation in two ways: they identify biological endpoint pairs absent from the literature, and their mechanistic paths guide how LLMs reason between them. 

---
# Automated Detection and Structuring of Social Tipping Point Evidence in Climate related Documents: A Modular AI Framework 

**Authors**: Kavindu Perera, Mohammad Abaeiani, Ekaterina Gilman, Lauri Loven, Mourad Oussalah, Tassos Kanellos, Beatrice Gobbo, Dante Adami, Nicolò Ferriani, Maximiliano Romero, Pierre Rossel, Marc Bonazountas, Christina Deligianni, Nikos Xyderis, Artur Bogucki, Lampros Argyriou, Prasasthy Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2609.12254)  

**Abstract**: The climate literature has grown faster than review teams can read it. That gap matters most for a concept like the environmental social tipping point, the threshold at which a small change triggers rapid, self-reinforcing change in a social system. Evidence of this kind of shift is usually contained in one or two paragraphs within a longer document. As a result, existing text mining tools-which categorize entire documents by topic or highlight isolated claims-leave an expanding set of important evidence without any systematic method for discovery or organization. This paper presents an open and modular transformer-based framework that detects and structures social tipping point evidence at the passage level. The framework joins five components into a single deployable workflow: a DistilBERT boundary splitter for segmentation, an iteratively augmented RoBERTa classifier for detection, a Mistral 7B model that rewrites each detected passage for clarity, a LLaMA 3.2 3B model that rates the passage against five published social tipping point criteria, and a Milvus vector store for semantic retrieval. The system is wrapped in a Streamlit interface backed by MinIO object storage. Evaluated on a 163-passage benchmark labelled by GPT-4.1 and a 51-passage set reviewed by experts, the splitter surpassed three competing methods on a nine-metric composite score (6.137). The tuned RoBERTa model achieved 71.4 percent accuracy with a Cohen's kappa of 0.337 on the full benchmark, and 87.5 percent accuracy with a kappa of 0.742 on passages with labels, outperforming both a climate-focused model and untuned language models. 

---
# DriftSE: Speech Enhancement with Generative Drifting 

**Authors**: Liang Xu, Diego Caviedes-Nozal, W. Bastiaan Kleijn, Longfei Felix Yan, Rasmus Kongsgaard Olsson  

**Link**: [PDF](https://arxiv.org/pdf/2609.12252)  

**Abstract**: We propose DriftSE, a novel one-step generative framework for speech enhancement formulated as a latent distribution equilibrium problem. During training, the drifting field aligns the generator's pushforward distribution with the clean speech manifold through drifting in a latent domain. During inference, the drifting process is discarded, enabling one-step generation. We establish that its enhancement quality depends fundamentally on the choice of latent representation. Semantic latents preserve phonetic structure but fail to capture physical acoustic cues, whereas acoustic latents reconstruct the physical signal but risk linguistic hallucination. Therefore, we introduce dual-latent drifting, performing parallel drifting in both semantic and acoustic latents to simultaneously preserve phonetic intelligibility and acoustic fidelity. Additionally, we demonstrate that DriftSE enables fully unpaired training by aligning latent distributions rather than exact point-wise targets. Consequently, DriftSE facilitates cross-dataset learning in the absence of paired noisy-clean samples. Moreover, DriftSE exhibits broad architectural flexibility across different generator backbones. Extensive evaluations on additive denoising and convolutive dereverberation demonstrate robust one-step enhancement across both offline and real-time causal settings. Notably, DriftSE achieves state-of-the-art word error rates across all four evaluated datasets while strictly operating at 1 NFE. Code and audio examples are available online. 

---
# Chopthin-Consensus Power Sampling: A Diversity-Preserving Approach to LLM Decoding 

**Authors**: Minoo Ahmadi, Seyedarmin Azizi, Erfan Baghaei Potraghloo, Mehdi Kamal, Massoud Pedram  

**Link**: [PDF](https://arxiv.org/pdf/2609.12243)  

**Abstract**: Inference-time power sampling via Sequential Monte Carlo (SMC) can substantially improve large language model (LLM) reasoning without requiring post-training. However, many existing SMC approaches rely on equal-weight resampling, which can aggressively prune low-weight trajectories, discarding potentially correct reasoning paths and degrading the genealogical diversity of the search space. To address this, we introduce Chopthin-Consensus Power Sampling (CCPS). Our method applies the Chopthin resampler to LLM decoding: rather than equalizing weights and forcing unnecessary particle duplication, it enforces an upper bound on the ratio between the largest and smallest weights and carries the unequal weights forward. This targeted intervention preserves a richer set of distinct reasoning paths, keeps the weighted SMC approximation unchanged in conditional expectation, and guarantees a lower bound on the post-resampling effective sample size (ESS). To fully exploit this enriched population, we employ a semantic-majority selection mechanism that merges token-identical final trajectories, clusters semantically equivalent answers, and returns the answer supported by the largest number of distinct trajectories. Evaluating across three open-weight models and five reasoning benchmarks, we show that Chopthin increases oracle coverage in 13 of 15 settings. Combined with semantic-majority selection, CCPS matches or exceeds the final-answer accuracy of the Power-SMC baseline in 14 of 15 settings, delivering absolute gains of up to 10.6 percentage points. These findings demonstrate that diversity-preserving resampling and diversity-aware selection are complementary mechanisms for training-free LLM reasoning. Code is available at this http URL. 

---
# Learning to adapt GR(1) specifications through degradation 

**Authors**: Tiberiu-Andrei Georgescu, Dalal Alrajeh, Sebastian Uchitel  

**Link**: [PDF](https://arxiv.org/pdf/2609.12231)  

**Abstract**: Reactive synthesis is a powerful tool for generating correct-by-construction controllers from formal specifications. GR(1) is an assume-guarantee specification framework that enables efficient synthesis, allowing synthesised controllers to be used in a wide array of applications. The limitation of such controllers is that, should they encounter environment behaviour unspecified in the assumptions of the specification, the specified system guarantees are no longer ensured. Our work proposes an approach based on oracle-guided inductive synthesis to adapt the specification to be consistent with the observed assumption violation, while degrading system guarantees as little as possible to maintain realisability. Our methodology discovers multiple potential solutions, so we propose a preference criteria, based on the ability of the specification to enable robustness under adaptation. Although our approach is capable of degrading the entire specification, for our case studies we successfully discover degradations that preserve the entire set of original guarantees. 

---
# Repair Before Reinforce: Context-Augmented Knowledge Graph Reasoning for Multi-Hop Question Answering 

**Authors**: Tharaka D. Fonseka, Niraj K. Jha  

**Link**: [PDF](https://arxiv.org/pdf/2609.12230)  

**Abstract**: Question-answering often requires reasoning across multiple connected facts rather than retrieving a single isolated relation. Knowledge graphs (KGs) provide a structured way to represent such facts, but training large language models (LLMs) only on isolated KG head-relation-tail triples may limit their ability to learn the surrounding context needed for multi-hop reasoning. In this work, we propose a context-augmented training framework for multi-hop question-answering. Although generally applicable, we validate the framework in the context of disease-specific KGs, extracted using a reliable KG extraction framework called GraphMERT, for Gastroparesis and Diabetes. For each primary KG triple, we attach supporting triples extracted from the same source text chunk to form a context graph (CG). This creates two supervision settings: KG-grounded supervision, which uses only the target KG triple or path, and CG-grounded supervision, which uses the target KG triple or path together with supporting context triples. We train the Qwen3-14B model using supervised fine-tuning (SFT) under both settings, producing KGModel and CGModel variants. To strengthen the lower-hop factual foundation of the models, we introduce an LLM-judged, history-aware adaptive repair pipeline that identifies unresolved one-hop failures, continually fine-tunes on targeted repair examples, and removes or quarantines problematic noisy triples. This repair stage enables the models to reach 100% accuracy on the cleaned retained one-hop validation sets. Finally, we employ reinforcement learning (RL) using lower-hop question-answer items and evaluate generalization on harder 3-hop, 4-hop, and 5-hop tasks. Across both diseases, context-augmented supervision consistently improves multi-hop performance over KG-only supervision. RL initialized from repaired SFT checkpoints yields larger and more stable gains. 

---
# Predicting Collision Cross Sections with GRACE: Geometric Residual Adduct Conditioning via Early-fusion 

**Authors**: Parthasarathy Suryanarayanan, Susanta Das, Shreyans Sethi, Kenneth M. Merz Jr., Joseph A. Morrone  

**Link**: [PDF](https://arxiv.org/pdf/2609.12223)  

**Abstract**: Collision cross section (CCS), derived from ion mobility mass spectrometry, is a common descriptor for molecular annotation. Prediction is challenging for machine learning models because it reflects the size, shape, and ionization state of a gas-phase molecular ion. Most predictors either ignore explicit 3D structure or treat adduct identity as a late categorical feature, which limits their ability to capture adduct-dependent geometric effects. We present GRACE (Geometric Residual Adduct Conditioning via Early-fusion), a 3D CCS predictor that adapts a pretrained molecular geometry encoder using geometric residual adduct conditioning via early fusion. GRACE combines two inductive biases: a residual objective relative to an adduct-aware physical descriptor baseline and adduct conditioning within the encoder via a learned adduct token and low-rank attention adapters. We evaluate the model on a curated set of over 9,000 experimental molecule-adduct CCS records with random, scaffold, and adduct-sensitive splits designed to separate interpolation, scaffold generalization, and adduct-driven generalization. GRACE achieves the best mean percentage difference among the evaluated learned models on all three splits: 1.67% on the random split, 2.11% on the scaffold split, and 2.36% on the adduct-sensitive split. Diagnostic analyses suggest that residual learning stabilizes training by removing the dominant mass-CCS trend, while early fusion improves adduct-sensitive prediction relative to late fusion. Across four independent external test sets, GRACE shows consistently lower error than the other evaluated models. On a held-out set, GRACE also attains the lowest mean percent difference when compared with four previously reported physics-based workflows. These results support residual learning and encoder-level adduct conditioning as practical inductive biases for fast, accurate CCS prediction. 

---
# QuPAINT: Physics-Aware Multimodal Reasoning for Quantum Material Characterization 

**Authors**: Sankalp Pandey, Xuan-Bac Nguyen, Hoang-Quan Nguyen, Tim Faltermeier, Nicholas Borys, Hugh Churchill, Khoa Luu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12202)  

**Abstract**: Characterizing two-dimensional (2D) quantum materials by optical microscopy requires localizing exfoliated flakes and determining their layer thickness from subtle optical contrast and interference color to select suitable flakes for device fabrication. However, models face synthetic-to-real domain shifts and variation across materials, substrates, laboratories, and imaging conditions. We present QuPAINT, a physics-aware multimodal framework for transferable quantum flake characterization. The Synthetic Materials Framework (Synthia) generates diverse synthetic microscopy images while preserving layer-dependent optical behavior. Using these images, we construct QMat-Instruct, a multimodal instruction dataset with image-specific reasoning traces generated from verified annotations and constrained to observable optical cues. QuPAINT integrates these signals through Physics-Informed Attention (PIA), which injects substrate-relative optical priors into the visual representation to support grounded multimodal reasoning. For evaluation, we introduce QF-Bench, to our knowledge, the largest real-world benchmark for this problem, spanning diverse microscopy and substrate conditions. Using its verified annotations, we study counting, visual grounding, reasoning quality, confidence calibration, and transfer to an unseen material. QuPAINT-8B substantially outperforms prior methods and establishes state-of-the-art performance for both general and monolayer flake detection. Additional experiments show that image-grounded supervision improves strict spatial grounding and confidence calibration while preserving robust general flake detection on the unseen material. 

---
# Retrieval-Augmented Generation for Scientific Code Understanding 

**Authors**: Aaron Nobile, Andreas Adelmann, Mohsen Sadr  

**Link**: [PDF](https://arxiv.org/pdf/2609.12190)  

**Abstract**: Large language models have become central to modern coding assistants, but state-of-the-art systems such as Claude Code or Codex rely on very large, cloud-hosted models with significant computational cost and data-privacy implications. This work investigates whether a useful, fully local coding agent can be built around small open-source models by shifting the computational burden away from inference. We develop a Retrieval-Augmented Generation (RAG) system for scientific code understanding that strictly separates an expensive offline ingestion stage parsing, structural graph construction, LLM-generated entity explanations, and embedding from a lightweight online answering stage. The system is evaluated on a 100-question benchmark spanning eleven categories over the IPPL scientific codebase written in C++, with answers scored by an independent frontier model as the judge. Across seven answering models, we find that model family and retrieval quality matter more than parameter count, i.e. a 9B model achieves the highest average score (0.795), outperforming both larger models within our pipeline and the same models embedded in the Claude Code retrieval architecture. The results indicate that front-loading code understanding into a reusable, codebase-specialised vector store enables small local models to deliver grounded and repository-specific answers, making the agent well suited as a privacy-preserving development tool for in-house scientific codebases. 

---
# Agentic TCAD Calibration Workflow for Oxide Semiconductor Transistors 

**Authors**: Gyujun Jeong, Junmo Lee, Sungwon Cho, Woohyun Hwang, Kwangyou Seo, Suhwan Lim, Wanki Kim, Daewon Ha, Rishi Ranade, Kihang Youn, Ram Cherukuri, Yiyi Wang, Asif Khan, Shimeng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12184)  

**Abstract**: Experimental TCAD calibration is essential for predictive technology modeling of emerging oxide semiconductor transistors. However, it remains time-consuming and expert dependent because of model ambiguity. Multiple physical models and parameter sets can reproduce the same measured transfer characteristics, while local fitting alone cannot uniquely identify the underlying device physics. We present the first demonstration of an agentic TCAD calibration workflow for a fabricated bottom-gate In--W--O (BG-IWO) transistor. Starting from the measured transfer curve and device information, the workflow uses measurement--TCAD residuals and local sensitivity tests to select bounded parameter corrections or evaluate additional physical models, and accept only updates that improve device metrics. The LLM agent orchestrates the workflow, while Sentaurus governs the device physics. For the 2\%-W reference device, five agent-suggested updates yield a fixed calibrated model, reducing the multi-metric device objective $J$ by 14.3$\times$. Maximum $V_{\mathrm{th}}$/$I_{\mathrm{on}}$ errors are 36.1~mV/0.022 decade for varying-drain-bias tests and 46.2~mV/0.062 decade for varying-channel-length tests, demonstrating model transferability across bias and geometry rather than a local parameter fit. W-composition tests provide process-sensitive insight. This agentic workflow provides a faster route to model development for emerging device technologies. 

---
# Explanations-Driven Active Feature Acquisition for Algorithmic Recourse 

**Authors**: Vinura Galwaduge, Jagath Samarabandu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12179)  

**Abstract**: Algorithmic recourse methods typically assume that a predictive model has access to all features of an individual. In practice, decisions are often made with partial information, because features are costly to acquire. Active feature acquisition addresses cost-constrained prediction, but existing methods are explanation-agnostic: prior work provides explanations only after acquiring additional features, rather than using explanations to drive acquisition. This work flips that and treats algorithmic recourse and feature acquisition jointly. We use Markov Blanket theory to unify counterfactual, semifactual, and alterfactual explanations and to characterize how available recourse grows as features are acquired. Building on this framework, we propose an Explanation-Driven Feature Acquisition (EDFA) method that selects features by explanatory value per unit cost. The framework is further extended with distribution-free validity guarantees for recourse issued from partial information, which signal trustworthy, lower-cost recourse, along with a lower bound on the calibration data required to certify them. Experiments on 7 publicly available datasets with neural network-based predictive models show that EDFA acquires substantially fewer features than state-of-the-art AFA baselines while maintaining comparable accuracy and yielding more decision-relevant, actionable recourse. The implementation is available on GitHub. 

---
# NDT Factory: Synthesizing Verified Network Digital Twins from Semantic Models via Multi-Agent LLM 

**Authors**: Sudipta Acharya, Petar Djukic, Burak Kantarci  

**Link**: [PDF](https://arxiv.org/pdf/2609.12170)  

**Abstract**: Autonomous network management requires systems that can evaluate Network Service Intents (NSIs) under varying conditions without manual implementation of analysis logic, as envisioned in TM Forum Level~4 (L4) autonomy. Behavioral Network Digital Twins (NDTs) enable such evaluation, but existing NDTs rely on pre-defined analytical logic, limiting adaptability for evolving closed-loop control. This paper introduces the NDT factory, a multi-agent software system that synthesizes executable behavioral NDTs on demand from semantic models using Large Language Model (LLM). We validate the system using a Call Admission Control (CAC) case study, where deterministic what-if analysis serves as the admission decision process. The NDT factory generates a complete CAC NDT through parallel synthesis and orchestration, achieving 100% compilation and test pass rates across multiple runs. Simulation over 300 NSIs shows 99.3% decision agreement with a reference implementation, 90% admission rate, and correct attribution of all rejections, demonstrating reliable synthesis with deterministic, verifiable execution. 

---
# USPLIT-VQA: U-Shaped Split Learning for Visual Question Answering with Contribution-Aware Weighted Aggregation 

**Authors**: Md Khalid Syfullah, Alvi Ataur Khalil  

**Link**: [PDF](https://arxiv.org/pdf/2609.12168)  

**Abstract**: Visual Question Answering (VQA) systems, jointly interpreting images and natural language queries, hold significant promise across many domains, yet the privacy-sensitive nature of user data creates a fundamental barrier. Centralized training requires access to all data, while federated learning requires each client to host the full model. We propose USPLIT-VQA, a U-shaped split learning framework for privacy-preserving VQA in which each client retains the initial layers and the classification head while the server hosts the computationally heavy intermediate layers, keeping raw inputs and labels on the client device. We further introduce Contribution-Aware Weighted Aggregation (CAWA), a gradientsimilarity-based client scoring mechanism designed to reduce the influence of malicious updates. Experiments on four VQA datasets (VQA-RAD, SLAKE, PathVQA, and VizWiz) with two backbones show accuracy gains over Federated Learning for the Custom model and reduced accuracy for BiomedCLIP under the evaluated fixed split, alongside client memory reductions of up to 5.8X and communication reductions of up to 10.8X. With one malicious client, CAWA reduces the attacker's influence by over 98%, while experiments at higher corruption levels identify its limitations. Reconstruction experiments further show lower inversion quality under the evaluated attacks. 

---
# A decision-basis contract for auditable LLM-assisted medical billing verification: deterministic rules, verbatim evidence, and fail-closed abstention 

**Authors**: Jan Hölter, Kevin Geis, Benjamin Raab, Boris Bauke  

**Link**: [PDF](https://arxiv.org/pdf/2609.12156)  

**Abstract**: This work presents a proof of concept for auditable LLM-assisted medical billing verification based on a decision-basis contract. The contract separates deterministic checks of versioned fee-catalog rules from LLM-based assessment of free-text documentation. The deterministic layer resolves the applicable catalog release and checks code availability, quantity limits, and exclusions. The semantic layer classifies each claimed item as supported, contradicted, or missing required information. Support and contradiction require a verbatim evidence span; unavailable rule context, unsuccessful assessment, or missing required evidence prevents support through fail-closed abstention. We evaluated four locally run open-weight models on a synthetic catalog and 36 curated cases under the contract, an ablation without explicit documentation requirements, and an end-to-end baseline. Outcome agreement varied across models and showed no consistent advantage over the baseline. Explicit documentation requirements improved identification of missing information for all four models. The evidence gate also exposed cases in which correct raw judgments lacked valid evidence and were converted to incomplete decision-basis entries. The results show how explicit decision records can make rule findings, documentation judgments, and abstention reasons inspectable. Evaluation on real catalogs, independently annotated documentation, and with human reviewers is required to assess practical value. 

---
# Neural Multichannel Distant Speaker Diarization with Heavy-tailed Source Separation Model 

**Authors**: Sicheng Mao, Baihan Li, Mathieu Fontaine, Anthony Larcher, Roland Badeau  

**Link**: [PDF](https://arxiv.org/pdf/2609.12154)  

**Abstract**: Distant speaker diarization remains challenging due to difficult acoustic environments, varying numbers of speakers and overlapping speech. Model-driven methods are proposed to exploit the speech source features in multi-channel recordings that help diarization. This paper generalizes a neural model that jointly learns to perform blind source separation and diarization over speech mixtures (neural FCASA) with heavy-tailed models. The popular Gaussian distribution has been applied for variance modeling in the original source separation model, which we replace with two families of heavy-tailed models (Leptokurtic Generalized Gaussian distribution and Student's t distribution) to better capture the heavy-tailedness in speech signals. Thanks to the Gaussian scale mixture model, we are able to unify the proposed method and the original one under the same form of learning objective. Our experiments show consistent large improvements in Diarization Error Rate (DER) and Jaccard Error Rate (JER) compared to the baseline on various corpora. 

---
# The Anatomy and Boundary of Adaptation under Temporal Tabular Shift 

**Authors**: Tianyu Wang, Xi Vincent Wang, Lihui Wang, Mian Li, Zhihao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12136)  

**Abstract**: Prequential adaptation of frozen tabular foundation models under temporal drift, with each label revealed only after prediction, helps some deployments and harms others, yet current practice does not predict which. We study the sources and limits of these gains. A diagnostic anatomy attributes gains to four recurring mechanisms under a streaming protocol that removes three optimistic biases and quantifies a fourth. Within an agnostic total-variation drift class, the target conditional is only partially identified: its identified-set diameter, the \emph{wall}, is irreducible from unlabeled data uniformly in sample size. A second, orthogonal $L^2$ projection wall quantifies what the frozen representation cannot express. Two canonical mechanism priors collapse the first wall. Under stated nuisance-rate conditions, the wall can be estimated from labeled historical windows at a $\sqrt N$ rate above the margin threshold $\gamma^\star=d_0/(2\alpha_s)$. At $\gamma=0$, the conditional lower-bound program depends on an open affinity estimate; the positive-margin lower branch also remains open. Semi-synthetic data illustrate the finite-sample mechanism with calibrated exponents. Stream-level proxies on eight industrial streams fall on the difficult side under a stated roughness bound, while the equality case $\gamma=\gamma^\star$ remains unresolved. 

---
# Extracting Dataset Mentions in Forced Displacement and FCV Documents: A Weakly Supervised Framework with LLM-Based Label Refinement 

**Authors**: Rafael Macalaba, Aivin V. Solatorio, Patrick Michael Brock, Olivier Dupriez  

**Link**: [PDF](https://arxiv.org/pdf/2609.12107)  

**Abstract**: Development and humanitarian organizations produce and support surveys, administrative registries, and other data resources to inform research, policy, and operations, yet systematically identifying where these datasets are referenced remains difficult. Such references are dispersed across research papers, project documents, humanitarian reports, and other unstructured text, limiting both the ability to trace data use and to identify potential gaps in data availability or dissemination. We present a weakly supervised framework for adapting dataset extraction to forced displacement and Fragile, Conflict, and Violence (FCV) documents without first constructing a large manually labeled training corpus. A lightweight model trained on general research literature generates candidate dataset mentions from unlabeled domain documents, which a frontier large language model (LLM) reviews in context, validating or rejecting candidates and correcting their extraction boundaries. The resulting annotations are supplemented with targeted synthetic and contrastive examples and used to fine-tune the lightweight model for large-scale extraction. We evaluate the resulting model on an independent gold-standard benchmark of 1,706 text passages spanning research, humanitarian, and operational documents. Across the full benchmark, the model achieves 74.1\% precision and 70.5\% recall at the mention level; among passages containing dataset references, precision reaches 89.5\%. At the passage level, the model achieves 88.2\% accuracy and 88.6\% specificity in distinguishing passages with dataset references from those without them. These results demonstrate a practical approach for constructing domain-specific supervision when labeled data are limited, and provide a technical foundation for larger-scale analysis of data use and potential gaps in the displacement data landscape. 

---
# MAIA: Multi-Agent Intent Articulation for Requirement Discovery in Art Commissions 

**Authors**: Yu-Chao Wang, Yanhong Lu, Yingjie Victor Chen, Tim McGraw  

**Link**: [PDF](https://arxiv.org/pdf/2609.12097)  

**Abstract**: In bespoke art commissions, laypeople know what they feel but lack the words to specify it: one participant wanted a laid-off truck driver depicted as "a ghost in his own machine" but left the medium, scale, and palette unsaid. We frame this as an articulation bottleneck at an under-served upstream stage: requirement discovery, which precedes any artist or image generator and forces the commissioner to constitute intent in the first place. We present MAIA (Multi-Agent Intent Articulation), a multi-agent system that scaffolds this stage through Socratic inquiry under a "Verification over Invention" rule, turning vague affect into a text-only brief of visual terms the user verifies. In a within-subjects study (N = 16), the full configuration produced a large, significant gain in Cognitive Support over a minimal baseline (r = 0.96, p_FDR = 0.015; LMM p_FDR < 0.001). Thematic analysis traces the same mechanism, and a validator gate structurally blocks unratified content. A complementary blind review by three professional concept artists on a sampled set of briefs corroborates this improvement from the artist's side: AI rewriting improved visual completeness and executability in all eight sampled tasks (task-level Wilcoxon p = 0.008; FDR q = 0.010), with directionally larger gains under MAIA than under the baseline (underpowered, d = 1.4-2.6). 

---
# Creating an Atomic User Model for Personality-Aware Large Language Model Interaction 

**Authors**: B. Sankar, Deepthika S, Pawni Yadav, Amogh A S  

**Link**: [PDF](https://arxiv.org/pdf/2609.12086)  

**Abstract**: Assistants built on large language models are expected to write as their user would, and the dominant approach is single-channel: preferences summarised from conversation history and reinserted into context. This inverts the order of inference. Preferences are the task-dependent surface of a comparatively stable personality structure, so a system storing only preferences relearns the person whenever the task changes. First, we characterise personality seepage, where a prompt's linguistic surface carries a personality fingerprint the assistant mirrors without access to the personality behind it. Second, we propose the Atomic User Model (AUM), a human-readable representation organising a person as a stable identity nucleus with four interpretable shells (psychological, cognitive and experiential, behavioural, and social), plus cross-shell entries recording internal conflict and authenticity. Third, we treat AUM as a retrieval index over a person rather than a prompt prefix, with a pipeline where a task classifier, component-selection function and budgeted retriever return a small payload of fields at generation time. Fourth, we evaluate it with sixteen language-model-simulated participants, six style-sensitive tasks and three seeds, plus a synthetic scaling study of the retriever. Retrieving eight fields matched the style fidelity of the full user model on 23% of the context (211 tokens against 915), improved on flat preference notes by 0.24 points on a five-point scale (p < 0.001, dz = 0.50), and raised forced-choice identification of the participant's own voice from 14.9% to 42.7% (25% chance). Four pre-registered controls returned null, locating the effect in the representation rather than the search over it. The benefit is largest for participants the un-personalised assistant reproduces worst (rho = -0.61, p = 0.013): personalisation is worth most to those the default serves least. 

---
# Hierarchical Prototype Emergence in Modern Hopfield Models 

**Authors**: Aditya Cowsik, Adithya Sriram  

**Link**: [PDF](https://arxiv.org/pdf/2609.12079)  

**Abstract**: Hierarchical correlations are a universal feature of any realistic model of data, and the question of how associative memory models may learn these correlations and generalize beyond them to construct new sensible images is an important step towards understanding more complex modern architectures such as diffusion models. We consider a hierarchical model for memories which are sampled and stored in a dense Hopfield network with polynomial activation. We analytically derive conditions for each level of this hierarchy to be locally stable - that is they are local energy minima. We use prototype reconstruction as a minimal model of generalization and we find that it takes only a quasi-polynomial amount of information to generalize beyond particular memories and even particular groups in the hierarchy. We observe a qualitatively analogous phase diagram in the number of memories, sharpness of the activation function (polynomial degree) for data from Fashion-MNIST. 

---
# Reality Is the Final Verifier: On Two Key Gaps in Agentic Software Engineering 

**Authors**: Alexander Krentsel, Shubham Agarwal, Mert Cemri, Shu Liu, Sidharth Sankhe, Ziming Mao, Matei Zaharia, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2609.12039)  

**Abstract**: Software development follows an implementation-verification loop in which developers or agents iteratively revise an implementation until an evaluator, such as a test suite, accepts it. The evaluator checks the implementation against a set of requirements under a model of the deployment environment. Yet even a formal proof that the implementation satisfies the requirements under the model cannot guarantee acceptable behavior after deployment. Requirements only approximate stakeholder intent, and the model only approximates the real deployment environment. We call these together - requirement gap and model gap - the two-gap framework, which unifies the main failure modes of agentic software engineer-ing: reward hacking exploits omissions in the requirements or model, while hallucination widens the gaps by fabricating requirements or environment assumptions.
Because neither gap can generally be certified closed in an open, changing world, the goal shifts from closing them to continuously narrowing them. We therefore propose an assurance-revision loop that uses deployment evidence to revise the requirements, model, or evaluator when stakeholders reject the resulting behavior. We then cast assured agentic development as a resource-allocation problem over human judgment, agent capability, and compute. The two principal bottlenecks mirror the two gaps: human judgment for the requirement gap and faithful, costly evaluation for the model gap. Reality remains the final verifier: acceptable behavior under actual deployment conditions is the ultimate test, while predeployment evaluations remain proxies for it. 

---
# Pelican-Sim 1.0: A General World Model Simulator for Embodied Intelligence 

**Authors**: Shilong Zou, Shilin Zhang, Yingji Zhang, Yuhang Huang, Yi Zhang, Zeyuan Ding, Han Dong, Junwei Liao, Yong Dai, Jian Tang, Xiaozhu Ju  

**Link**: [PDF](https://arxiv.org/pdf/2609.12036)  

**Abstract**: In this technical report, we propose Pelican-Sim 1.0, a general world model simulator for embodied intelligence that predicts future observations from visual context and robot actions to support downstream learning and decision making. The model incorporates four key design features: (1) Unified action representation: a 28-dimensional action value space covering most mainstream embodiments, keeping one model valid across heterogeneous devices. (2) Action-visual injection: URDF- and camera-rendered action videos bridge actions and pixels, giving markedly better controllability across embodiments, scenes, and tasks (PSNR +0.904 over alternative fusion baselines). (3) Sparse mixture-of-experts (MoE): sparse MoE layers add capacity for heterogeneous dynamics and absorb the action modality while reducing inter-modality conflict (FVD -6.530 vs. the dense backbone). (4) Efficient rollout generation: causal adaptation and few-step distillation yield a four-step autoregressive simulator, achieving a 5.67-fold speedup over the 35-step model. Benefiting from these designs, we train on approximately one million real-world and simulated trajectories and obtain large gains in action controllability and video quality: PSNR improves over the strongest evaluated baselines by 4.636 on AgiBotWorld Beta, 2.080 on RoboMIND, and 10.343 on RoboTwin, with the adapted EWMBench DYN score up 0.426 on RoboTwin. Relying on this, four downstream applications on RoboTwin succeed: 500 generated trajectories added to 50 demonstrations per task raise policy success from 70% to 93%; policy evaluation reaches a Pearson correlation of 0.994 across five checkpoints; and relative success gains reach 47.7% for action selection and 20.3% for policy improvement. Qualitative generalization across trajectory, scene, object, embodiment, and viewpoint shifts highlights its potential as a general-purpose world model simulator. 

---
# Continuous Learning of Gravity Field Irregularities Around Small Bodies via Neural Hamiltonian ODEs 

**Authors**: Giacomo Acciarini, Dario Izzo  

**Link**: [PDF](https://arxiv.org/pdf/2609.12022)  

**Abstract**: We propose to learn the unknown dynamics in the proximity of a small body directly from tracking data, representing them as a feed-forward neural network embedded in the system Hamiltonian. The equations of motion form a Neural Hamiltonian Ordinary Differential Equation, whose variational equations provide exact training gradients: estimation uses position and velocity arcs at realistic noise levels, without acceleration or potential labels, and a continual learning approach warm-starts the network as new data are acquired. The known part of the Hamiltonian carries whatever is available, from the central term and spin state to the constant-density model of the imaged shape. We assess the method against a normalized spherical harmonics expansion estimated from identical arcs through the same machinery, on scenarios built on the shapes of Itokawa, 67P, Bennu and Eros. The network remains usable inside the Brillouin sphere: it plans ballistic descents at Itokawa to \SI{4.6}{m} median touchdown error from tracking alone, against 5.1--\SI{48.9}{m} for harmonics of degree 4--12, and to \SI{0.9}{m} with the imaged shape as prior, a configuration that also recovers localised density anomalies invisible to any harmonics degree. The two representations are complementary, and we discuss their combined use across the phases of a small-body mission. 

---
# When Agent Metrics Measure Different Things: An Evidence-Grounded Audit of the Praxa AI Pipeline 

**Authors**: Stefan G. Creadore, Peyton Woakz  

**Link**: [PDF](https://arxiv.org/pdf/2609.12017)  

**Abstract**: Agent evaluations can be numerically correct while measuring a different construct from the one implied by their labels. We present a retrospective measurement audit of selected Praxa AI implementation files, historical evaluation artifacts, and operational records. A 139-case offline routing report contains 112 passes and 27 failures despite zero gating failures, because known gaps are explicitly exempted from the gate. An identifier-free export represents 8,843 tool-attempt rows: 8,395 recorded durations and 448 missing values. Of the durations, 121 equal the signed 32-bit maximum and carry abandoned-client labels; inspected database code clamps elapsed lifecycle age. The pooled recorded 99th percentile is 2,147,483,647 ms, versus 38,118.31 ms among server-observed completed calls. This is a stratum contrast, not a treatment effect. In a documented single-trajectory compaction pilot, the reported follow-up input reduction is 94.39%, but the reduction across the trigger and follow-up calls together is 46.54%. We reproduce the descriptive calculations, verify 91 timing statistics through a separate weighted rational-arithmetic implementation, and execute 13 scoring-function and 12 analysis-verifier tests. Finite-completion bounds show how missing durations limit all-row timing statements without imputing values. The contribution is a source-linked case study and reusable verification package for separating gate policy, lifecycle timing, and request-level accounting from broader agent-performance claims. Historical provider runs and the full current pipeline were not independently reproduced; general capability superiority and population-level statistical significance are not established. 

---
# Can We Trust LLM Judges: A Study of Capability-Dependent Biases and Multi-Judge Ensemble for Bias Calibration 

**Authors**: Gemma Zhang, Prachi Badarayani, Asmi Kumar, Sadid Hasan, Sulaiman Vesal  

**Link**: [PDF](https://arxiv.org/pdf/2609.12002)  

**Abstract**: LLMs are increasingly used as automated judges for model training and evaluation, yet individual judges exhibit systematic biases that undermine reliability. Much of prior work has studied biases in pairwise LLM-as-a-judge settings; in this paper, we focus on absolute scoring tasks, which mirror more realistic use cases. Across four benchmarks and six models (36 judge-examinee pairs), we show that a model's task accuracy strongly predicts its judging accuracy (Pearson $r \geq 0.90$ on most models) and inversely predicts its directional bias ($r \leq -0.83$), but that accuracy alone does not ensure fair evaluation: more capable examinee models consistently receive more lenient judgments from all judges ($r \geq 0.83$). To address this, we propose calibrated weighted majority voting (WMV), an ensemble evaluation method that aggregates multiple LLM judges weighted by online estimates of their false-positive and false-negative rates. We introduce a disagreement-based estimator that derives these error rates purely from inter-judge agreement patterns, requiring no ground-truth labels or task metadata. In a simulated experiment with shifting task distributions, our label-free WMV tracks an oracle with perfect error-rate knowledge to within 0.5 percentage points on average, outperforming both individual judges and unweighted majority voting. These results demonstrate that principled multi-judge calibration can simultaneously improve accuracy and correct for systematic leniency without requiring labeled data, offering a scalable path to reliable automated evaluation as model capabilities increase. 

---
# Assessment of Non-Institutional AI Tool Usage Among Clinicians 

**Authors**: Sarah Pungitore, Jarrod Mosier  

**Link**: [PDF](https://arxiv.org/pdf/2609.11990)  

**Abstract**: Generative artificial intelligence (AI) tools are increasingly accessible and have the potential to improve efficiency across clinical workflows. However, clinicians may also use non-institutional AI tools that are not provided, managed, or governed by their healthcare institutions, creating potential concerns related to privacy, security, accuracy, and clinician-AI interaction. Little is known about how clinicians currently use these tools for work-related tasks. We conducted a descriptive survey of clinicians recruited from the University of Arizona College of Medicine-Tucson and Banner University Medical Center-Tucson between May 20 and June 26, 2026. Participants reported their use of AI tool categories and the frequency with which they used AI for specific tasks across five workload categories: administrative work, clinical work, research, studying/continued education, and teaching. Forty-four respondents completed the survey. Forty-three respondents reported using AI for at least one work-related task during the preceding 6 months. Conversational AI and clinical decision support/diagnostic AI were the most used tool categories, each reported by 28 respondents. Administrative and clinical tasks demonstrated the most frequent use. AI was also used for higher-risk activities, including diagnostic assistance and clinical decision support. Non-institutional AI use was common among surveyed clinicians and extended across a broad range of work-related activities, including tasks with potential implications for clinical reasoning and patient care. Further research is needed to characterize how clinicians use these tools, how they evaluate AI-generated outputs, and how AI can be safely and effectively integrated into clinical workflows. 

---
# Who Pays for Open Review? Visible Author Reputation and Its Effect on Ratings 

**Authors**: Qinghua Zhao, Xinyu Chen, Yanhui Yang, Tengfeng Sun, Junfeng Liu, Zhongfeng Kang  

**Link**: [PDF](https://arxiv.org/pdf/2609.11983)  

**Abstract**: An OpenReview bug in November 2025 broke anonymity at several conferences and prompted calls for open review, which motivate us to ask what shifting from blind to open would mean for authors. Analyzing over 18,000 reviewed submissions to ICLR 2026, split into de facto open and blind groups by arXiv preprint timing, we find that ratings rise with author reputation under both mechanisms, with a steeper slope under open review that is statistically significant, and that the open-blind difference is concentrated at the borderline ratings. The pattern holds across five reputation proxies (including institution, h-index, and citation count), three author-aggregation rules, and five definitions of the open window. A controlled simulation with five AI models as reviewers, holding the manuscript fixed and varying the author reputation, reproduces the effect. With claude-opus-5 as the reviewer, for example, rating rises by 0.5 points as the author moves from low to high reputation. 

---
# PRISMA-LLM: An Empirical Reporting Framework for AI-Assisted Systematic Reviews 

**Authors**: Miguel Zabaleta, Baihan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.11559)  

**Abstract**: Large language models (LLMs) and AI-enabled software increasingly participate in systematic-review decisions, yet the information needed to audit these workflows is reported inconsistently. We analyze SciLitBench, a corpus of 888 review-automation papers with 14,726 annotations, to characterize changes in methods, review-stage use, evaluation and reported limitations. Automation has shifted toward LLM- and software-facing workflows, including stages that can alter the evidence base. Since 2023, 38.0% of software/product papers reported no evaluation, compared with 9.3% of LLM papers. Reporting coverage increased with LLM workflow complexity, yet 52% of positive-only LLM evaluations still reported an unmet reliability or performance requirement. From these patterns, we introduce PRISMA-LLM, an empirically grounded framework separating implementation disclosure from consequence-sensitive evaluation and limitation reporting. 

---
# SoulAuth: An Actor-native Identity Architecture and Rust Reference Implementation for Humans and Long-lived AI Actors 

**Authors**: Kun Yuan, Harold Wang, Echo Li, Egusi Gui, Kiki Hu, Lucas Luo, Magnus Hu  

**Link**: [PDF](https://arxiv.org/pdf/2609.11258)  

**Abstract**: As AI systems move from transient model invocations toward long-lived actors that persist across credentials, clients, sessions, and runtime instances, identity infrastructure must answer a basic question: where should the canonical continuity boundary be placed? This paper introduces Actor-native Identity and presents SoulAuth, an open-source Rust reference implementation for Humans and long-lived AIActors. We argue that any subject that must persist under its own identity and remain independently attributable should have an ActorIdentity that is not replaced by an Account, Credential, Client, AuthSession, IdentityBinding, or runtime instance. SoulAuth therefore treats Humans and long-lived AIActors as first-class identity subjects while keeping authentication distinct from downstream authority. Methodologically, we use a Philosophical Engineering approach that translates conceptual analysis of subjecthood into identity objects, invariants, lifecycle semantics, system responsibilities, implementation boundaries, and inspectable conformance evidence. Evaluation against the fixed SoulAuth v0.1.0 artifact shows that the implementation realizes core boundaries including Human/AIActor first-class identity status, Client/Actor separation, and Authentication/Authority separation, while gaps remain in unified Credential modeling and historical attribution anchored to ActorIdentity. We therefore report partial, not full, architecture conformance. 

---
# AI Safety: Not Optional, Not Later 

**Authors**: Qinghua Lu, Yoshua Bengio  

**Link**: [PDF](https://arxiv.org/pdf/2609.10630)  

**Abstract**: Incidents show that AI safety failures often arise across multiple layers. We present a safety-by-design assurance architecture combining model-level supervision, such as Scientist AI, with system-level controls over scaffolds and harnesses, independent verification, monitoring, and evidence infrastructure, supported by governance for accountability and evidence interoperability. 

---
# Robust Trust 

**Authors**: Piotr Dworczak, Alex Smolin  

**Link**: [PDF](https://arxiv.org/pdf/2602.09490)  

**Abstract**: An agent chooses an action based on her private information and a recommendation from an informed but potentially misaligned adviser. With a known probability, the adviser truthfully reports his signal; with the remaining probability, he can send any message. We characterize optimal robust decision rules that maximize the agent's worst-case expected payoff. Every optimal rule is equivalent to a trust-region policy in belief space: the adviser's reported beliefs are taken at face value if they fall within the trust region but are otherwise clipped to the trust region's boundary. We derive alignment thresholds above which advice is strictly valuable and fully characterize the solution in both binary-state and binary-action environments. 

---
