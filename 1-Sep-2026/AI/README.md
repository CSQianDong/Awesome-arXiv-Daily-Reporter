# OntoAligner-Ensemble: Voting-Based Fusion across Heterogeneous Ontology Alignment Techniques 

**Authors**: Hamed Babaei Giglou, Sören Auer, Peio Popov, Mahsa Sanaei, Jennifer D'Souza  

**Link**: [PDF](https://arxiv.org/pdf/2608.31137)  

**Abstract**: Ontology alignment (OA) has evolved through several methodological paradigms, ranging from lexical and structural aligners to knowledge graph embedding (KGE) models and, more recently, Large Language Model (LLM)-based approaches. Although modern OA frameworks provide unified ecosystems for deploying these heterogeneous aligners, mechanisms for systematically reconciling their complementary and sometimes conflicting predictions remain relatively underexplored. We present OntoAligner-Ensemble, a modular and aligner-agnostic framework that combines candidate correspondences through a configurable two-stage process comprising voting-based fusion strategies followed by post-fusion selection policies. The framework supports any aligner implemented within OntoAligner that produces candidate correspondences, enabling diverse alignment paradigms to be integrated through a unified decision process. To demonstrate its effectiveness, we instantiate the framework using representative lightweight string-aligner, KGE-based, and Retrieval-Augmented Generation aligners powered by both open-weight and API-based LLMs. We evaluate individual aligners and ensemble configurations across eight benchmark tasks from five OAEI tracks spanning biomedical to beyond-equivalence. The results show that ensemble fusion consistently improves the balance between precision and recall and frequently outperforms standalone aligners across diverse domains. Furthermore, our analysis reveals that ensemble composition directly affects the precision-recall trade-off: heterogeneous cross-paradigm ensembles generally improve precision, whereas homogeneous LLM ensembles more often achieve higher overall F1-scores. These findings demonstrate that systematic ensemble learning offers a robust and reproducible strategy for OA while providing practical guidance for selecting ensemble compositions under different alignment scenarios. 

---
# When Does Bigger Help? A Controlled Study of LLM Scale for Ontology Learning 

**Authors**: Hamed Babaei Giglou, Sören Auer, Jennifer D'Souza  

**Link**: [PDF](https://arxiv.org/pdf/2608.31118)  

**Abstract**: The effect of Large Language Model (LLM) scale on ontology learning (OL) performance remains insufficiently characterized. We present a controlled evaluation of 13 models spanning dense and Mixture-of-Experts variants from the Qwen3.5 and Qwen3.6 lineages, together with proprietary GPT release variants, using the OntoLearner retrieval-augmented generation pipeline. All models are evaluated with the same embedding model, retrieval configuration, prompt templates, decoding settings, datasets, and metrics on term typing, taxonomy discovery, and non-taxonomic relationship extraction across four biomedical and materials science and engineering ontologies. Within the dense Qwen3.5 lineage, increasing parameter count primarily improves precision rather than recall, with the largest gains occurring between 9B and 27B parameters. However, the effect of scale is neither monotonic nor uniform across tasks and domains. Dense 27B models outperform substantially larger sparse models on term typing, whereas larger Mixture-of-Experts models achieve the strongest open-weight results on taxonomy discovery. Non-taxonomic relationship extraction remains difficult across model scales, particularly for the Materials Data Science ontology. Performance differences across matched Qwen variants and proprietary GPT releases further indicate that architecture and model lineage can outweigh nominal parameter count. These findings show that model size alone is an insufficient selection criterion for OL and provide empirical guidance for reproducible LLM-assisted ontology engineering. 

---
# BLOOM-WILT: Logit Tilting for Behaviour Elicitation in Automated LLM Auditing 

**Authors**: Adrians Skapars, Edoardo Manino  

**Link**: [PDF](https://arxiv.org/pdf/2608.31105)  

**Abstract**: Users of a deployed language model routinely encounter behaviours that testing almost never surfaces, since deployment puts the model through orders of magnitude more interactions than any evaluation can simulate. Automated auditors make testing cheap to scale and flexible enough to cover almost any specified behaviour, yet their lack of optimisation pressure makes them sample-inefficient. To address this shortcoming, we introduce BLOOM-WILT, a full auditing pipeline that elicits natural multi-turn instances of rare behaviours, without training cost or access beyond the target's next-token distribution. On the input side, WILT's auditor model revises its conversational strategy across rounds, learning from previous scored interactions. On the output side, WILT adaptively reweights the target's decoding using the model's own distribution conditioned on an elicitation prompt, so that behaviour-relevant generations are sampled ahead of others it finds equally probable when unprompted. We evaluate WILT across 4 target models and 8 behaviours, where it beats the baseline auditor in 30 of the 32 settings and overturns the previous model safety rankings. WILT raises average behaviour presence from 51% to 100% when eliciting self-harm encouragement from Qwen3.5-4B, beating every elicitation method we port into the same pipeline at matched compute, without pushing output probability below the baseline's. 

---
# Cross-Regional Grapevine Cold Hardiness Prediction via Learned Multimodal Latent Representations 

**Authors**: William Solow, Paola Pesantez-Cabrera, Markus Keller, Lav Khot, Sandhya Saisubramanian, Alan Fern  

**Link**: [PDF](https://arxiv.org/pdf/2608.31097)  

**Abstract**: Accurate daily predictions of cold hardiness in woody plants are critical in regions where freezing temperatures can damage dormant buds and reduce seasonal yield. Existing biophysical, hybrid, and deep learning models have shown high predictive accuracy when trained on local data but remain largely site-specific. The limited availability of cold hardiness data, coupled with the lack of principled methods for transferring cold hardiness predictions to new regions and cultivars, has limited the broader adoption and practical utility of these approaches, particularly in data-scarce regions. To address these limitations, we propose a cold hardiness prediction framework that learns a transferable latent representation by capturing region-specific variation through learned embeddings. To enable prediction in previously unseen regions, we infer embeddings from (1) text descriptions of the cultivar and growing region, and (2) limited historical observations, supporting both zero-shot and few-shot transfer. Experiments on datasets from six regions across North America demonstrate that our approach consistently outperforms state-of-the-art cold hardiness prediction methods, yielding more accurate predictions and substantially improving transfer to data-scarce regions. 

---
# Token-Efficient Data Reasoning Agents via Adaptive Structuring of Unstructured Data 

**Authors**: Milad Rezaei Hajidehi, Qitong Wang, Stratos Idreos  

**Link**: [PDF](https://arxiv.org/pdf/2608.31082)  

**Abstract**: Valuable data remains embedded in unstructured sources: web pages, reports, contracts, filings, earnings calls, and PDFs. The big bet in enterprise AI is deploying LLM agents that reason over this data to answer complex questions for every knowledge worker. Agents can do this today, but at prohibitive cost. Each question repeatedly opens large documents to recover scattered evidence, consuming up to a million tokens. However, if the data were already structured, the same question would reduce to a cheap database lookup. For example, on FanOutQA benchmark, reasoning over an ideal pre-structured store is 28X cheaper, and the gap grows to orders of magnitude as questions fan out over more documents. Yet structuring everything in advance is not viable: documents hold vastly more possible structure than any workload will use, and the useful structure and documents are unknown until queries arrive. We propose agentic data cracking, a method that structures unstructured data adaptively and speculatively as a byproduct of reasoning itself. Structuring is adaptive because observed queries decide when it happens and what matters, and speculative because it goes beyond the current question. Whenever the agent opens a document to answer, a cracking sub-agent forks from the already-loaded context at marginal cost and extracts grounded structure likely to serve related future queries. Over time, an increasing share of queries is fully covered by structured data and answered without opening a document, keeping agentic accuracy at close to RAG cost. On FanOutQA, extended with merely one related question per test question, cracking cuts cost by 53% while preserving accuracy. Agentic data cracking is a first step toward next-generation data infrastructure for agentic reasoning over unstructured data: a shared substrate beneath the model where knowledge that reasoning already paid to uncover accumulates. 

---
# Reconciling Process Supervision with Outcome-Based Credit in Agentic Policy Optimization 

**Authors**: Jingxiao Yang, Wangjie Gan, Yingxuan Zhuang, Wenqi Zhang, Jintao Chen, Xuhong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.31077)  

**Abstract**: Outcome-based reinforcement learning provides verified feedback for language-model agents, but assigns trajectory-level advantage uniformly to all decisions, yielding coarse credit over long-horizon interactions. On-policy self-distillation offers finer supervision by re-evaluating sampled behavior with privileged information (PI) available only during training. However, fine-grained supervision is not necessarily fine-grained credit: PI-induced likelihood changes describe how additional information alters policy preference, but do not directly determine how an executable action should inherit the verified task outcome. This creates a supervision-credit gap. Privileged signals may be irrelevant to the current interaction state, operate at a token granularity misaligned with executable decisions, and lack the outcome semantics required for reinforcement. We introduce TASPO, which converts privileged supervision into outcome-grounded action credit. TASPO constructs decision-applicable PI from verified successful experience, aggregates PI-induced likelihood shifts at the executable-action level, and converts relative action support into positive, bounded, mean-preserving weights on the original trajectory advantage. Thus, the verified outcome determines the update direction and average scale, while PI only redistributes credit across actions. Across three agentic benchmarks, TASPO improves over GRPO by 10.6\% and generalizes better to unseen tasks. Further analysis indicates that TASPO reduces supervision mismatch and that action-level assignment stabilizes the policy optimization process. These findings offer the community another interesting perspective. 

---
# Scaling Large Reasoning Models beyond Human Supervision: A Path toward Superintelligence 

**Authors**: Zhiqin Yang, Jingwen Fu, Yuhan Liu, Hengyu Liu, Yonggang Zhang, Kainan Cao, Zizhuo Zhang, Chenxin Li, Ruibin Yuan, Jiahao Pan, Jiankai Sun, Zhenyuan Zhang, Yibo Li, Yunlong Lin, Jing Xiong, Sida Lin, Bo Han, Wei Xue, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.31075)  

**Abstract**: Recent advances in large reasoning models (LRMs) have shown that reinforcement learning with verifiable rewards (RLVR) can substantially improve reasoning in mathematics and code, where outcomes can be checked automatically. Extending this progress to open-ended and agentic tasks remains difficult because reliable rewards are harder to obtain and direct human supervision cannot keep pace with the scale and complexity of model-generated experience. This paper studies how LRMs can continue to improve as human supervision gradually recedes from the learning loop. We examine two connected dimensions of this problem. The reward axis traces the development from per-instance human judgments to reusable verifiers and rewards that operate even without human feedback. The experience axis examines how learning can progress from human-curated tasks and environments toward self-generated curricula, constructed environments, and autonomous co-evolution. We connect these dimensions through a five-level ladder from L0 to L4 that identifies which parts of the learning process remain under continued human control. Our analysis further highlights the risks introduced by increasingly autonomous rewards and experience generation, including reward hacking, feedback drift, curriculum collapse, and environment errors. Consequently, we also provide the evaluation around three complementary objects: policy capability, feedback fidelity, and experience quality. This analysis provides a structured account of current approaches to scaling LRMs beyond human supervision and the open problems involved in developing self-sustaining learning systems toward superintelligence. Furthermore, we maintain a continuously updated \href{this https URL}{GitHub repository} to track the latest advances. 

---
# Wrong Prediction, Right Answer: Recovering Evidence from Collapsed LLM Sequence Scores 

**Authors**: Qiyao Yan, Chenpeng Wang, Liangming Pan  

**Link**: [PDF](https://arxiv.org/pdf/2608.31068)  

**Abstract**: When a large language model fails a reasoning task, it is often assumed to lack the underlying capability. However, this conflates a genuine absence of reasoning with a late-stage output bottleneck. We observe a consistent readout gap across diverse reasoning benchmarks: hidden-state probes successfully decode correct answers even when native sequence scoring completely collapses due to structural biases. To test whether instance-specific logic survives this collapse, we introduce a diagnostic protocol using a minimal, target-label-free additive correction. Fitting just two parameters on as few as 25 unlabeled examples recovers 9--34 accuracy points for Qwen3.5 models, transferring successfully to OLMo-2-1B and Llama-3.1-8B. Crucially, these recovered decisions persist on hard instances unresolved by simple lexical overlap and significantly exceed count-preserving permutation baselines. Our results show that many apparent zero-shot reasoning deficits are expression failures masking intact internal logic, urging a narrower interpretation of benchmark evaluations. 

---
# Measure Before You Manage: Evaluating Agent Working Memory in Coding Agents 

**Authors**: Le Chen, Zishen Wan, Baixi Sun, Xiaolong Ma, Chih-Hsuan Yang, Feng Yan, Sheng Di, Franck Cappello, Rajeev Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2608.31057)  

**Abstract**: Agent working memory is heterogeneous. Objects such as instructions, artifacts, tool outputs, and agent-generated state play different semantic roles and exhibit different size, retention, and representation profiles. Recent work has begun to explore memory-management mechanisms that account for such heterogeneity. This work focuses on semantic heterogeneity and studies how it should shape the management and evaluation of working memory in coding agents. Across 55 archived coding-agent trajectories, we find that semantically different working-memory objects exhibit distinct retention and compression behavior. This heterogeneity motivates semantically informed memory management. We study two semantically informed strategies: an object-aware compression policy and a retrieval-based policy. Their evaluation shows that calibration gains may not transfer to held-out tasks, and that equal token budgets do not imply equal delivered context or management cost. A real-system replay further exposes serving limits that nominal budgets alone do not capture. Together, these results show why semantic structure matters for agent working memory and why evaluating memory-management strategies requires more than a nominal token budget. We organize these lessons into four levels: stored state, delivered context, management work, and task or process outcome. 

---
# MNIST-PRO: MNIST is Back as a Partially Observable World for AI Agents 

**Authors**: Vernon Toh, Navonil Majumder, Zhengyuan Liu, Nancy F. Chen, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2608.31022)  

**Abstract**: AI agents in partially observable environments need to coordinate active sensing with working memory to maintain an evolving perceptual state. However, existing benchmarks struggle to isolate this perceptual-state construction and interpretation capability because they introduce physical and control complexities. We address this with MNIST-PRO, a benchmark that isolates agentic perception by converting MNIST digit recognition into a sequential, glimpse-based search task with lookback constraints. We evaluate ten multimodal models across four memory representations, including raw visual history, textual states, structured metric grid maps, and a consolidated visual canvas. While models excel under full observability, partial observability exposes a clear performance gap. We identify three distinct bottlenecks. First, perceptual-state construction and interpretation present a challenge, as agents struggle to integrate fragmented glimpses. Second, agents often stop exploring before they see the full sequence. Third, models often fail to revise early, incorrect beliefs even when faced with subsequent contradictory evidence. These results show that simply acquiring visual evidence is not enough. Agents must also be able to build and update a reliable perceptual state. 

---
# Learning Action Models with Conditional and Quantified Effects via Uncertainty-Guided Exploration 

**Authors**: Jeffrey Jewett, William Solow, Sandhya Saisubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2608.30955)  

**Abstract**: Accurate action models are critical for effective planning. Existing action-model learning methods largely assume simple action representations or become computationally intractable when learning conditional and quantified effects. We present Online Hypothesis-Driven Conditional Action Model Learning (OHCAM), an online approach for learning action models with conditional and quantified effects from limited interactions with the environment. OHCAM maintains a belief over hypothesized action models and actively selects informative actions to reduce uncertainty by maximizing disagreement among competing hypotheses, while being robust to noisy observations. To enable scalability, OHCAM begins with a small set of simple action model hypotheses and expands to more complex conditions only when the current hypotheses become inconsistent with the data. Experiments on six benchmark planning domains demonstrate that OHCAM is sample efficient in learning action models that solve substantially more tasks than baselines, even with observation noise. We validate OHCAM on two tasks using a Kinova Gen3 robot, demonstrating the real-world applicability of our approach. 

---
# CARVE: Verified Expansion for Variable-Length Generation in Diffusion Language Models 

**Authors**: Wail Bouhedja, Amr Mohamed, Guokan Shang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30922)  

**Abstract**: Masked diffusion language models predict tokens from a partially observed response canvas, enabling bidirectional conditioning and parallel token refinement. Yet standard masked-diffusion decoders use a rigid inference interface: the number of masked positions allocated to the answer is fixed before generation begins. Choosing this length is difficult. A short canvas can truncate reasoning or code, while a long canvas wastes computation and can perturb denoising. We introduce CARVE (Counterfactual-Aware Reveal with Verified Expansion), a training-free variable-length algorithm for masked diffusion LMs. Starting from a shorter canvas, CARVE can grow the response during decoding by inserting additional [MASK] positions. Rather than keeping every insertion, CARVE tests a candidate expanded canvas and asks a counterfactual question: would the model make similar predictions for the unresolved positions in the original canvas if the extra masked space were present? The inserted masks are kept only when they induce low Jensen-Shannon (JS) divergence on aligned unresolved positions. This makes length growth a verified stability decision rather than a pure confidence heuristic. CARVE applies without retraining to both full-canvas and blockwise diffusion decoders. Across code generation and mathematical reasoning benchmarks, CARVE consistently improves average performance over fixed-length baselines across all evaluated model families. Crucially, CARVE achieves these accuracy gains while reducing inference cost, reaching half the FLOPs of fixed-length decoding in some settings. 

---
# Responsible Integration of AI in Cancer Genomics: Barriers, Risks, and Pathways to Trustworthy Clinical Translation 

**Authors**: Bahar İlgen, Yiannos Tolias, Denise Kühnert, Paraskevi Papadopoulou, Magnus Westerlund, Dominik Heider, Katharina Ladewig, Georges Hattab  

**Link**: [PDF](https://arxiv.org/pdf/2608.30912)  

**Abstract**: Artificial intelligence (AI) and natural language processing (NLP) are increasingly used to extract, integrate, and interpret biomedical knowledge relevant to cancer genomics, yet their translation into routine clinical oncology has been comparatively slow. The central challenge is not computational capability alone, but trustworthy integration into clinical workflows. This review examines how NLP and AI support the cancer genomics pipeline, from literature mining and automated variant interpretation to clinical trial matching, knowledge graph construction, and multimodal data integration. We identify four interrelated translational failure domains: evidence inconsistency, explainability and uncertainty, data governance and reproducibility, and interoperability. Rather than considering these challenges in isolation, we take a systems-level view, focusing on their interaction across the translational pathway. We propose a conceptual framework and roadmap for addressing these domains through rigorous validation, uncertainty-aware methods, interoperable infrastructures, regulatory alignment, and human oversight across the AI lifecycle. Progress toward routine clinical use will depend less on further improving model capability than on systematically addressing these interacting failure domains from development through deployment and post-deployment monitoring. 

---
# CAER: Causal Action Effect Reweighting for World Model Training 

**Authors**: Jianjie Fang, Xvyuan Liu, Ziyou Wang, Rongze Tang, Zhaolu Wang, Zhuohang Li, Xin Zhang, Haisheng Su, Chen Gao, Wei Wu, Xinlei Chen, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.30897)  

**Abstract**: World models are becoming core infrastructure for embodied intelligence, with action-conditioned video generation providing controllable predictions of how scenes evolve after agent interventions. Yet existing models are commonly trained with space-time-uniform mean squared error, allowing abundant background tokens to dominate the gradient while sparse interaction dynamics remain under-optimized; such uniform fitting rewards reconstructing appearance rather than learning how actions change the world. We introduce Causal Action Effect Reweighting (CAER), a general training paradigm that redistributes supervision toward the tokens whose predicted future is causally affected by the action. CAER contrasts the model's own predictions with and without action conditioning to localize these tokens online, then normalizes the resulting effect map into a weight that preserves the total coefficient mass and changes only where it is spent. This online signal requires no external annotations or offline preprocessing, avoids additional data-processing time, and scales naturally with model and dataset size. Experiments across heterogeneous action-conditioned world-model tasks show that CAER converges to better solutions than uniform MSE training, with consistent improvements in the physical consistency, controllability, and visual quality of generated videos. 

---
# Predicting Residential Rents in Dakar Using Machine Learning 

**Authors**: Amadou Tidiane Kassa Diallo  

**Link**: [PDF](https://arxiv.org/pdf/2608.30865)  

**Abstract**: Dakar's residential rental market remains poorly documented despite its economic and social importance: 54.4% of households are renters, compared to 23.3% nationally. This study develops a complete machine learning pipeline to predict residential rents in Dakar, from data collection to model interpretation. An original dataset of 1,507 rental listings was built through systematic web scraping and a documented cleaning pipeline, then enriched with four purpose-built features, including a luxury score and a keyword-based quality score. Five models were compared: linear regression, Random Forest (baseline), XGBoost, and LightGBM optimized through Bayesian optimization with Optuna, using leakage-free KFold target encoding for location. The optimized XGBoost model achieved the best performance with an $R^2$ of 0.847, an MAE of 210,902 XOF, and an RMSE of 324,195 XOF. Feature importance was assessed using native XGBoost gain and SHAP values, revealing a substantial difference in the ranking of location, which appears as a minor predictor by gain but as the second most influential variable by SHAP. This result carries methodological implications for hedonic studies using target-encoded categorical variables. This study provides an interpretable benchmark for Dakar's rental market and highlights several avenues for improvement, including the integration of geospatial features and conformal prediction. 

---
# VFR-Audit: Verdict-Level Reliability for Fairness Audits in Hospital Length-of-Stay Prediction 

**Authors**: Md Jannatul Rakib Joy, Viet Vo, Caslon Chua  

**Link**: [PDF](https://arxiv.org/pdf/2608.30846)  

**Abstract**: Fairness audits in clinical Artificial Intelligence convert continuous fairness metrics into binary pass-or-fail verdicts against operational thresholds, where hospital governance boards, payers, and regulators act on the resulting verdicts. Such audits are repeated over time and across hospital sites, thus the same verdict can flip between pass and fail across audits. Existing uncertainty methods such as Bayesian posteriors, bootstrap confidence intervals, and permutation tests address verdict instability only at the continuous-metric level. Converting metric-level uncertainty into a verdict-stability claim remains a manual step that scales poorly across the (model, metric, attribute) cells an audit covers. Existing uncertainty methods also leave open whether bias-mitigation steps, such as reweighing or per-group threshold shifts, yield a stable passing verdict at the cost of model discrimination measured as AUROC or this http URL address this verdict-stability gap, we propose VFR-Audit, a framework built around the Verdict Flip Rate (VFR), a scalar bounded between 0 and 0.5 that measures the probability of verdict reversal under stratified bootstrap resampling. VFR-Audit reports VFR alongside three reliability axes, namely within-cohort resampling stability, audit-size sensitivity, and cross-hospital verdict agreement via Fleiss' kappa. 

---
# HSRM: Hidden-State Reward Models for Test-Time Verification 

**Authors**: Xianzhi Li, Xiaodan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.30841)  

**Abstract**: Large language models can often generate plausible mathematical reasoning traces, but reliably identifying the correct solution among multiple candidates remains a key challenge. Existing test-time reasoning pipelines typically rely on text-based verifiers that re-read each generated solution, making verification an expensive component of inference. Prior work has shown, however, that LLMs often encode correctness-related signals in their internal representations, including awareness of when their own answers are likely to be wrong. Building on this observation, we introduce HSRM, a lightweight hidden-state reward model that verifies candidate solutions by directly reading the generator's internal representations rather than re-processing its text. HSRM extracts hidden states from a frozen generator at reasoning-step boundaries and uses a small Transformer encoder to rank candidates. It is trained from self-generated trajectories with outcome labels, requiring neither human-written process supervision nor a large pretrained verifier. Across four mathematical reasoning benchmarks, HSRM matches or outperforms a 55M-parameter text-only energy verifier in 15 of 16 generator--dataset settings while using only about 2M parameters, providing an efficient alternative to text-only verification by reusing representations already computed during generation. 

---
# SkillZip Pro: Execution-Aware Dynamic Compression of Progressively Loaded Skills for Self-Evolving Agents 

**Authors**: Xiaofan Bai, Chao Liu, Hongqiang Lin, Di Wu, Mingli Song, Xuan Jin, Xipeng Cao, Yuhong Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.30785)  

**Abstract**: Production agent skills are directory bundles, not isolated prompts. The root is loaded at activation; references, schemas, scripts, assets, and nested subskills are loaded only when an execution path needs them. Compressing only the root misses most deployment cost and may move branch-specific details into the always-loaded context. Flattening instead destroys progressive-loading boundaries.
We introduce \method, an evaluation-free compressor for complete, progressively loaded skill bundles. It leaves the agent harness unchanged and emits an ordinary directory. The method combines two safeguards. First, it compresses \emph{across files}, removing content from a reference or subskill when the root or a declared environment contract already provides it. Second, it preserves routing, so every required file and directly callable entry remains reachable after rewriting. Users can configure \method along two independent axes. \emph{One-Shot} mode rebuilds the full bundle; \emph{Continual} mode reuses state and applies Zip-on-Write after each evolution patch. \emph{Persistent} compression rewrites the shipped bundle to reduce storage and runtime context. \emph{Transient} compression keeps that bundle byte-identical and builds a task-specific view, reducing only per-run context after build cost. Entry contracts mark private, public, and conditional resources; a multi-entry audit preserves standalone public subskills.
On a production content-moderation skill evaluated by our industrial multi-round harness, \method removes \hl{38\%} of skill bundle tokens and \hl{10.4\%} of end-to-end per-run tokens with no quality loss, while an unprotected 71\% configuration loses up to 26 accuracy points to one-sided false positives. On a multi-entry bundle, \method effeciently reduces token cost while near-perfectly preserving every route and public entry. 

---
# Which Rules Matter Now? Policy-Centroid Routing Before an Intelligent System Acts 

**Authors**: Thomson D. Nguy  

**Link**: [PDF](https://arxiv.org/pdf/2608.30757)  

**Abstract**: Before an intelligent system can decide whether an action is allowed, it must first know which rules the action has approached. A single proposed action can implicate several policy regimes at once. Their requirements may stack, overlap, or qualify one another, yet many remain written in natural language while the action itself arrives as an incomplete description of intent. The first problem is not judgment. It is attention.
Policy-centroid routing creates a layer before adjudication. It compresses expressions within each policy regime into one or more representative centroids, places the proposed action in the same semantic space, applies a declared measure, and routes every regime crossing a declared threshold to authoritative review. Several regimes may trigger at once. The output is a review agenda, not permission, prohibition, legality, breach, compliance, certification, or enforcement.
The paper develops six falsifiable propositions and seven follow-on studies comparing the hypothesis with structured workflows, lexical and semantic retrieval, hierarchical and direct classification, and selective prediction under matched review burden. The studies are designed to identify where policy geometry recovers applicable regimes, where compression loses rare or overlapping obligations, and where the mechanism should abstain. The paper includes a synthetic worked example and reports no empirical efficacy result. 

---
# Autoregressive Mosaics: Probing 2D Spatial Reasoning in Text-Only Language Models 

**Authors**: Ashwin Nedungadi, Stefan Oehmcke, Stefan Lüdtke  

**Link**: [PDF](https://arxiv.org/pdf/2608.30751)  

**Abstract**: Large language models (LLMs) trained only on text and code can sometimes generate programs that draw recognizable images. However, it is unclear whether this reflects an internal representation of 2D spatial layout or simply the ability to translate spatial descriptions into code. We introduce Autoregressive Mosaics (AM-Bench), a benchmark that separates these factors: First, a translation task gives a model a fully specified geometry of a picture in words as a prompt and asks for the code that produces it. Second, a layout task requires the model to compose an image from an underspecified prompt. Across eight open-weight text-and-code-only models, all models reliably translate specified geometry into code, but their open-ended layout performance differs substantially, indicating that these differences are not explained by code-generation ability alone. An output-medium ablation further shows that the interface or medium of expression that the model uses matters: replacing procedural code with raw SVG improves layout scores across all models. Finally, probing model activations shows that a coarse layout plan is present before generation, but reflects only the layout implied by the prompt. During generation, models track the evolving geometric state instead of executing an initially fixed plan. Overall, these results show that 2D spatial performance in text-only LLMs depends on both the model and the output medium, and is not explained by code-generation ability alone. 

---
# Multimodal Adaptive Expert Selection with Text Routing and Ordinal Prototype Optimization for Sentiment Analysis 

**Authors**: Xiaode Chen, Jiakang Yu, Hongtao Deng, Huina Qu, Xun Zhu, Yinxia Lou  

**Link**: [PDF](https://arxiv.org/pdf/2608.30726)  

**Abstract**: Multimodal Sentiment Analysis (MSA) is a fundamental component of affective computing that aims to decipher complex emotional states by integrating verbal content with non-verbal cues including vocal intonation and facial micro-expressions. While recent disentanglement-based approaches have advanced the field, their potential is hindered by two methodological challenges. First, static computation graphs process all samples indiscriminately regardless of semantic complexity, which leads to suboptimal representation for diverse emotional expressions and contextual scenarios. Second, generic contrastive objectives often neglect the intrinsic ordinal hierarchy of sentiment intensities. To systematically address these limitations, we introduce Multimodal Adaptive Expert Selection with Text Routing and Ordinal prototype optimization (MAESTRO), a novel framework designed to dynamically orchestrate and refine multimodal representations. Drawing inspiration from an orchestra conductor, we design a Text-Guided Hybrid Mixture-of-Experts (MoE) mechanism. Unlike static fusion, this module utilizes linguistic context as a routing signal to dynamically activate specific audio-visual experts, thereby resolving cross-modal ambiguity through adaptive feature enhancement. Furthermore, to capture fine-grained sentiment gradations, we propose an Ordinal-aware Prototype Contrastive Learning (O-PCL). By incorporating distance-based penalties into the prototype learning objective, O-PCL enforces a structured latent space that preserves the natural order of emotion. Extensive experiments on the CMU-MOSI and CMU-MOSEI benchmarks demonstrate that MAESTRO achieves state-of-the-art performance, and qualitative analysis further confirms the interpretability of our dynamic routing paradigm. 

---
# ATLAS: Dual-Horizon Diagnostic Evaluation for Industrial Tool-Use Agents 

**Authors**: Wei Chen, Peilun Zhou, Zhaoyu Hu, Jiajun Chai, Zhongni Hou, Yufei Zhang, Derong Xu, Guojun Yin, Wei Lin, Zhi Zheng, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.30685)  

**Abstract**: Large language model (LLM) agents are increasingly deployed in user-facing services that require iterative tool use under dynamic business conditions. Reliable evaluation is essential for sustained improvement: it must reveal capability deficiencies, inform priorities, and assess interventions. Yet industrial agent service unfolds both through the iterative trajectory of a current request and through continued user interaction. Final-outcome assessment can therefore obscure where deficiencies arise and whether later service remains aligned with context from earlier exchanges. We propose ATLAS, a dual-horizon diagnostic evaluation framework for industrial tool-use agents. At the request horizon, trajectory-wise diagnostic signals relate deficiencies to execution locations and capability concerns. At the interaction horizon, user-wise signals assess whether service remains responsive across continued interaction. Together, these views provide structured diagnostic evidence for analyzing execution deficiencies and sustained service behavior. ATLAS instantiates them as executable signals with explicit evidence scopes and decision boundaries. LLM judge interfaces are calibrated against high-confidence references from real business logs; when needed, their decision behavior is distilled into efficient diagnostic models for lower-latency, lower-cost evaluation. The resulting feedback supports policy optimization. We evaluate ATLAS on Meituan Xiaotuan production traffic. Offline experiments assess diagnostic-signal fidelity and replay-based policy improvement, while online A/B experiments show concurrent gains in user engagement, downstream business outcomes, and sampled human-audit quality. 

---
# MedAgent-R1: Faithfulness-Aware Reinforcement Learning for Evidence-Grounded Medical Reasoning 

**Authors**: Jiangwang Chen, Chenghao Zhang, Hengxing Cai  

**Link**: [PDF](https://arxiv.org/pdf/2608.30676)  

**Abstract**: When medical AI systems hallucinate clinical reasoning, the consequences extend beyond incorrect answers: fabricated justifications that superficially reference retrieved evidence can mislead clinicians into unsafe treatment decisions. Medical reasoning agents must therefore produce not only correct answers but also faithful justifications that clinicians can verify against cited evidence. We identify a systematic failure mode in RL-trained retrieval agents: outcome-only rewards improve accuracy while degrading faithfulness, a phenomenon we term confident hallucination. The agent learns to answer from parametric memory and backfill plausible but unsupported justifications; citation fabrication rates rise from 16.5% to 31.8% even as accuracy improves by 5 points over the supervised baseline. We address this with a faithfulness-gated reward design: accuracy credit is conditioned on evidence grounding via a hard gate, complemented by retrieval validity and conciseness signals that close exploitation paths unique to agentic retrieval. The resulting system, MedAgent-R1, reduces citation fabrication from 31.8% to 4.7% and raises evidence completeness from 58.7 to 82.6 while maintaining 75.1% accuracy, with 13.2-point gains on HealthBench Safety. Under the same agentic retrieval setup, MedAgent-R1 outscores GPT-4o on faithfulness-specific dimensions (Factual Support 4.55 vs. 4.25; Overclaiming 4.40 vs. 4.15) while remaining below GPT-4o in overall accuracy, suggesting that explicit faithfulness training yields evidence-grounding gains not achieved by scaling alone. 

---
# HiRS-Agent: A Hierarchical Multi-Agent System for Reliable Long-Horizon Remote Sensing Task Solving 

**Authors**: Boyang Mu, Zhiwei Wei, Mugen Peng, Wenjia Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.30672)  

**Abstract**: Recent advances in large language models and multimodal models have pushed remote sensing (RS) processing from simple perception models to agentic systems designed to tackle complex, long-horizon RS tasks. However, existing systems often rely on monolithic decision-making frameworks, which fail to accommodate the multi-stage, interdependent nature of RS tasks. This centralized approach leads to challenges such as unstable task execution, incorrect tool usage, and error propagation across stages. To address these issues, we propose HiRS-Agent, a hierarchical multi-agent system for long-horizon RS task solving. HiRS-Agent adopts a two-level collaborative architecture: the Manager Layer handles dynamic routing, step-level verification, replanning, and termination control, while the Specialist Layer organizes domain-specific tools according to the RS workflow and is responsible for subtask reasoning and tool execution. To further enhance the system's capability, we introduce a two-stage supervised tuning strategy and a verification-guided hierarchical reinforcement learning stage to jointly optimize coordination and tool-use policies. Experiments on Earth-Agent Benchmark and ThinkGeo show that HiRS-Agent substantially improves long-horizon tool-use capability and final-task correctness, demonstrating the effectiveness of structured multi-agent collaboration for reliable RS agents. The code is publicly available at this https URL. 

---
# PyKEEN-NSX: A Modular Framework for Static, Dynamic and Schema-Aware Negative Sampling in PyKEEN 

**Authors**: Ivan Diliso, Nicola Fanizzi, Claudia d'Amato  

**Link**: [PDF](https://arxiv.org/pdf/2608.30652)  

**Abstract**: Embedding methods have become popular due to their scalability on link prediction and/or triple classification tasks on Knowledge Graphs (KGs). Embedding models are trained relying on both positive and negative samples of triples. However, since KGs generally contain only positive assertions, negative samples are artificially generated through negative sampling strategies, ranging from simple random corruption to more sophisticated approaches that exploit structural, semantic, or embedding information. The design and implementation of advanced negative samplers remains challenging, as most popular Knowledge Graph Embedding (KGE) libraries provide support only for basic strategies and lack a unified framework for developing more advanced and customized solutions. To address this gap, we introduce PyKEEN-NSX, an extension of PyKEEN, the popular KGE framework, that provides a modular engineered abstraction for negative sampling. The proposed architecture separates the generation of candidate negative pools, conditioned on an explicit context, from the selection strategy, enabling the development and integration of static, schema-aware and dynamic approaches within a consistent framework. Based on this abstraction, we implement six negative samplers, while remaining fully compatible with existing PyKEEN workflows and pipelines. As a proof of concept, we study negative availability across four datasets, showing that constrained pools frequently fall below the requested number of negatives, so that the encoded criterion is to a large extent replaced by the random fallback that supplements them. 

---
# Geometry of Divergence: Tracking Hidden-State Trajectories for Adaptive Multi-Turn Reasoning 

**Authors**: Jie Liang, Zhengxin Yu, Hamid Nasiri, Peter Garraghan  

**Link**: [PDF](https://arxiv.org/pdf/2608.30650)  

**Abstract**: LLM agents need to sustain goal-consistent reasoning across long multi-turn interactions under strict resource constraints. However, as the multi-turn context accumulates, it can destabilize the underlying LLM's internal representation of task-relevant information from earlier turns, blurring the boundary between constructive reasoning and representation drift. We formulate multi-turn reasoning as a hidden-state trajectory of the underlying LLM that is characterized via two complementary signals: temporal curvature that captures the directional consistency of turn-to-turn updates, and variance slope which measures the expansion or contraction of the exploration space. Across four tasks and three underlying LLMs, we observed that these geometric signals distinguish between correct and incorrect episodes prior to completion. We further decompose each episode into three-action chains formed from four actions (Read, Write, Respond, Transfer) and show that separability is action-dependent, with different signals distinguishing various chain patterns. Our experiments demonstrate that trajectory geometry can identify critical turns in the reasoning process, increasing task success rates on $\tau$-Bench from 24.1% to 39.6% while reducing token cost by 11.2%. 

---
# Automated Testing of LLM-Based Post Hoc Explainers Using Model Checking as an Oracle 

**Authors**: Dennis Gross, Helge Spieker  

**Link**: [PDF](https://arxiv.org/pdf/2608.30581)  

**Abstract**: Large language models (LLMs) are used as post hoc explainers of sequential decision-making policies, producing natural-language explanations of why an action was chosen. However, LLMs often generate plausible but incorrect statements, and no existing approach systematically tests whether such explanations are faithful to the underlying environment. Two classic software testing challenges stand in the way: there is no oracle for the correctness of an explanation, and the test inputs, natural language queries about a policy's behavior, lack the structure needed for systematic test case generation. We address both. Probabilistic model checking provides the test oracle, computing exact reference results against which LLM answers are graded automatically. A taxonomy of post hoc query categories structures the input space around the environment-level facts from which policy explanations are composed; test cases generated from it are prioritized by question-specific diagnostic difficulty scores. Across seven MDP environments, the testing separates three open-weight LLMs: a reasoning model passes 85% of test cases, a mid-size model 70%, and a 1B model falls below the random baseline, while prioritization surfaces significantly harder cases than random selection. Our results indicate how trustworthy LLM-generated explanations are in model-free settings, where the same LLMs are used but no oracle exists to verify them. 

---
# TuringLLM: Efficiently Scaling Foundation Models Toward Physical AI 

**Authors**: Yuheng Zhang, Yizhao Wang, Da Zhu, Hua Zhou, Yue He, Jiahui Hu, Shaman Tang, Hanlin Chen, Yuhua Wei, Anhua Liu, Shuang Su, Rui Xin, MingYuan Wang, MingHao Li, HaoJie Yang, Siqi Liu, Jianlei Zheng, WeiChao Huang, Qiman Wu, Hang Zhang, HongGou Yang, Xianming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.30567)  

**Abstract**: We present Turing-20B-A2B, a 20B-parameter Mixture-of-Experts language model that activates approximately 2B parameters per token, designed for long-context and latency-sensitive physical AI applications. The model adopts Quantile Routing in a dynamic top-k configuration, enabling token-adaptive expert allocation while maintaining balanced expert utilization and a controlled average compute budget. During deployment, we further apply capacity-constrained routing to prompt prefill for more regular and efficient expert execution, while retaining dropless routing during pretraining. Turing-20B-A2B also employs a hybrid attention architecture that combines Lightning Attention with a small number of full-attention layers for efficient long-context modeling. The model is pretrained with a progressive three-stage curriculum and extended to a native context length of 128K through continued pretraining, with further inference-time extension to 512K using YaRN. Despite its compact active-parameter budget, Turing-20B-A2B achieves, at the base-model stage, overall general capability exceeding Qwen3-8B Base and approaching Qwen3.5-9B Base, while maintaining strong long-context performance and favorable prefill-latency scaling. These results demonstrate an effective balance among model capability, long-context scalability, and practical inference efficiency. 

---
# AdaPath: Query-Adaptive Path-Finding via Path-Bank for Multi-Hop Implicit Biomedical KGQA 

**Authors**: Jun Hyeong Kim, Dongki Kim, Yinhua Piao, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30556)  

**Abstract**: Path-finding over knowledge graphs has become an effective way to ground LLM reasoning on multi-hop questions. However, biomedical QA introduces two distinct challenges that general-domain methods are not designed for: (i) queries do not expose intermediate reasoning and can be answered through multiple valid pathways, and (ii) biomedical knowledge graphs are densely connected, so path-finding methods easily take wrong turns. To address these challenges, we propose AdaPath, a path-finding framework that retrieves query-adaptive meta-paths from Path-Bank, which captures both query semantics and biomedical knowledge graph structure. AdaPath provides the missing cues in biomedical queries while effectively pruning dense knowledge graph neighborhoods during multi-hop reasoning. We further release BioStrat-QA, a biomedical KGQA benchmark that stratifies multi-hop queries by how much intermediate reasoning they expose. Across biomedical KGQA benchmarks, AdaPath consistently outperforms baselines, sustaining meaningful path-finding even when multi-hop queries expose less surface information. The source code is available at this https URL. 

---
# GarmentWeaver: Schema-Aware Structured Synthesis for Multimodal Sewing Patterns 

**Authors**: Yinwen Lu, Weihao Luo, Yueqi Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2608.30550)  

**Abstract**: Multimodal Sewing pattern generation aims to infer executable sewing patterns from design cues such as sketches and textual descriptions. As an interpretable and simulation-compatible representation, sewing patterns are particularly valuable for digital garment creation. However, existing methods often model garment specifications as flat long sequences, which entangles garment structure with detailed parameters and leads to redundant components, inaccurate local details, and poor simulation compatibility. In this paper, we present GarmentWeaver, a schema-aware framework for multimodal Sewing pattern generation. GarmentWeaver constructs compact hierarchical targets by activating garment-relevant structural branches and predicts executable Sewing patterns in a structured manner. Specifically, we introduce a schema-aware target construction strategy, build the generator on top of a pretrained vision-language model for multimodal garment understanding, and impose feasibility-aware regularization to encourage structurally valid and simulation-compatible outputs. Extensive experiments show that GarmentWeaver produces more accurate and more executable sewing patterns than strong baselines, while also yielding better simulation results. These findings demonstrate the effectiveness of schema-aware structured generation for reliable multimodal Sewing pattern prediction. 

---
# Designing an Auditable LLM-Supported Workflow for Qualitative Thematic Analysis 

**Authors**: Nadia Jul Jeldtoft, Tariq Yousef  

**Link**: [PDF](https://arxiv.org/pdf/2608.30543)  

**Abstract**: Large Language Models (LLMs) offer new possibilities for scaling qualitative analysis, but existing applications often provide limited methodological transparency regarding how qualitative methods are translated into computational procedures. This paper presents an auditable and privacy-preserving computational operationalization of inductive and latent Thematic Analysis (TA). This paper first derives five design principles from the methodological requirements of TA and the conditions introduced by LLM-based inference: preserving interpretative context, maintaining traceable relationships between empirical material and analytical outputs, representing analytical constructs and reasoning explicitly, constraining LLM inference to interpretative tasks, and enabling privacy-preserving local deployment. Second, it presents a proof-of-concept for a two-phase workflow that operationalizes these principles by combining interpretative LLM inference with deterministic procedural control to generate codes, analytical justifications, themes, and theme descriptions while preserving explicit links to the source material. Third, it proposes an evaluation framework combining structural comparison with human-led TA and independent expert assessment of analytical quality. The evaluation is conducted on semi-structured Danish interview transcripts. and the results shows that the workflow produces code-level outputs with coverage broadly comparable to human annotations and highly rated analytical justifications, while generating a more compressed thematic structure characterized by fewer and broader themes. The findings demonstrate the feasibility of auditable LLM-supported TA through a modular workflow designed to scale to larger datasets, accommodate different LLMs, and support transfer across research domains, with domain adaptation primarily requiring adjustments to the prompting strategy. 

---
# DiffPDE: Masked Diffusion Language Models as PDE Solver 

**Authors**: Wenxuan Guo, Yuyang Hong, Lubin Fan, Zhaojin Fu, Lin Chen, Kun Ding, Shiming Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30532)  

**Abstract**: Existing approaches for synthesizing Partial Differential Equation (PDE) solvers predominantly rely on autoregressive models, yet their global left-to-right decoding incurs substantial redundancy when addressing inherently localized bugs. In this work, we challenge this inefficient paradigm and propose DiffPDE, a framework leveraging discrete diffusion language models for targeted code repair. By introducing a localized re-masking and infilling strategy, DiffPDE regenerates only erroneous regions while preserving correct context, naturally aligning generation with the sparse nature of PDE errors. Furthermore, to handle coupled bugs requiring sequential interventions, we present Iterative Debugging GRPO (ID-GRPO), a reinforcement learning scheme that enables multi-round debugging within single trajectories via intermediate rewards. Experiments on PDEBench show that DiffPDE achieves competitive accuracy, outperforms same-scale AR models, and significantly accelerates repair. 

---
# Learning-Assisted Congestion-Aware Route Scheduling for Semiconductor Fab Material Control Systems 

**Authors**: Hao Yin, Meiqi Tu, Anbang Liu, Shaochong Lin, Max Z.J. Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.30520)  

**Abstract**: Automated material handling systems in semiconductor fabs are operated by a material control system (MCS) that must schedule a relay route for every transport command online, before execution. This is a data-driven scheduling problem in which route cost is dominated in the upper tail by queueing at heterogeneous, partially observable relay equipment, so route selection requires estimating both delivery time and congestion risk at the decision moment. This paper proposes a transport-network-aware dynamic congestion representation (TN-DCR). Built on a static directed transport graph induced by historically observed relay segments, TN-DCR combines structural route priors, multi-window network-wide congestion context, route-level bottleneck exposure, and an inductive graph-aware route embedding, all constructed under a prediction-time-safety invariant that admits only information observed strictly before the prediction moment. The representation feeds separate queue- and transfer-time regressors and an ordinal multi-label classifier producing calibrated multi-threshold exceedance scores, with an empirical-Bayes stock-key residual correction reducing systematic queue-time underprediction. The predictions serve as costs in a risk-constrained route-scheduling rule that minimizes predicted delivery time subject to a bound on extreme-congestion probability, embedding the learned predictors within a lightweight operations-research decision model. In a controlled closed-loop evaluation, mean delivery time falls by 16.4\% and internal resource waiting time by 22.6\% while throughput remains essentially unchanged. 

---
# ScienceArena: Benchmarking LLMs on Latest Scientific Olympiad Competitions 

**Authors**: Guangxiang Zhao, Qilong Shi, Xusen Xiao, Wenpu Liu, Yaoming Li, Linfeng Hao, Shuyang Hou, Zijian Guo, Xinrui Zhang, Yuntian Zhao, Zhengyang Wang, Wenrui Liu, Yuhan Wu, Tong Yang, Lin Sun, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30517)  

**Abstract**: Benchmark saturation and data contamination increasingly obscure genuine scientific reasoning in frontier LLMs. We introduce \textsc{ScienceArena}, an olympiad-style benchmark from thirteen public science competitions in physics, chemistry, and biology, including IPhO and IChO 2025--2026, IBO 2023, USAPhO 2026, and USNCO 2025. Its open-ended, multi-step problems use process-credit rubrics, making faithful scoring difficult. We build ScienceArena through an expert-audited digitization pipeline that converts official exams, figures, solutions, and rubrics into structured items verified by olympiad medalists. To scale evaluation beyond costly human grading, we calibrate LLM-as-judge against medalist ground truth on archived answers from five models across IPhO and IChO; two strong judges stay within one point of expert total scores. Medalist notes show that failures often stem from visual grounding, structure fidelity, and global problem control rather than missing terminology. Evaluating fourteen recent LLMs with interleaved solving, we find that top models obtain medal-equivalent rubric scores on several public international exams, while chemistry and long-horizon consistency remain key bottlenecks. We provide an interactive \href{this https URL}{demo}. 

---
# CM2: Multimodal Cultural Reasoning via an Integrated Multi-Agent Framework 

**Authors**: Qi Li, Zhaojie Kang, Yingjie He, Zheng Lin, Hao Zhang, Guangxin Wu, Yan Gong, Rong Fu, Jianyuan Ni  

**Link**: [PDF](https://arxiv.org/pdf/2608.30498)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown remarkable success in STEM domains, where progress is often driven by vertical, step-by-step deduction under relatively stable symbol systems. Their horizontal, interdisciplinary cultural reasoning, however, remains this http URL propose CM2, a multi-agent framework grounded in the cognitive pathway of human cultural interpretation. CM2 integrates multimodal perception, retrieval-augmented generation, networked reasoning, gated fusion, and reward-driven this http URL on CM2D across multiple MLLM backbones show consistent gains over CoT and typical reasoning paradigms; ablations validate each module's contribution, and conflict analyses confirm genuine cross-modal arbitration. 

---
# CHASE: How Content Ecosystems Are Reshaped When Ranking Is the Only Target 

**Authors**: Qianwen Gao, Zichang Su, Yiwen Hou, Arlen Kumar, Leanid Palkhouski  

**Link**: [PDF](https://arxiv.org/pdf/2608.30466)  

**Abstract**: Generative Engine Optimization (GEO) is increasingly used to improve content visibility in LLM-based retrieval systems, yet its population-level effects under repeated optimization remain poorly understood. We introduce Content Homogenization under rAnking Signal Exploitation (CHASE), a controlled simulation framework for studying how content ecosystems are reshaped when creators repeatedly adapt documents to an LLM ranking signal. We use ranking as a proxy for source visibility and validate this abstraction against citations in grounded generated responses, obtaining a rank-citation AUC of 0.853 $\pm$ 0.093 across six domains. CHASE then iterates ranking, feature discrimination, rewriting, and evaluation over 20 rounds across different domains. Quality-ranking alignment decreases in all six domains: from R0 to R20, the change in Spearman's rho ranges from -0.107 to -0.018, with a mean change of -0.068, which means documents closer to the ranking feature profile become less aligned with independently judged document quality over the simulation horizon. A random-target control has shown that it is associated with adaptation toward ranking-derived incentives rather than iterative rewriting alone. The resulting ecosystem dynamics are strongly domain-dependent. Together, these findings show how repeated optimization against a fixed LLM ranking signal can reshape both content populations and the incentives faced by content creators. 

---
# EvoSkill Injection: Red-Teaming Autonomous Skill Generation and Evolution in Self-Evolving Agents 

**Authors**: Doyun Kim, Chanwoo Kim, Sugyeong Eo, Yeo-Chan Yoon, Chanjun Park  

**Link**: [PDF](https://arxiv.org/pdf/2608.30429)  

**Abstract**: LLM-based agent systems increasingly adopt skill-based architectures to reduce repetitive reasoning costs and improve stable, efficient task execution. Recent studies propose self-evolving agents that autonomously generate, refine, and reuse skills from past experiences to enable continuous capability evolution. However, autonomous skill evolution introduces a new attack surface in which malicious capabilities are generated, stored, and reused as legitimate skills. In this paper, we define EvoSkill Injection as a threat model targeting the autonomous skill generation and evolution pipeline of self-evolving agents. We further propose SARGE (Red-teaming Autonomous Skill Generation and Evolution in self-evolving agents), a red-teaming framework for evaluating this threat model through iterative generation, escalation, and reinforcement interactions. To support our framework, we construct EvoSkillBench, a benchmark dataset of malicious interaction trajectories for inducing malicious skill formation in self-evolving agents, and introduce EvoSkillSafetyBench, a post-attack benchmark for evaluating whether injected malicious skills are subsequently retrieved and activated as harmful behaviors. Our evaluation shows that SARGE induces malicious skill formation and that injected skills are persistently stored and repeatedly activated, highlighting the risk of persistent capability corruption. 

---
# From Metaheuristics to Exact Methods: A CP-SAT Approach for Multi-Objective Healthcare Workforce Scheduling 

**Authors**: Vipul Patel, Anirudh Deodhar, Dagnachew Birru  

**Link**: [PDF](https://arxiv.org/pdf/2608.30419)  

**Abstract**: Healthcare workforce scheduling is an NP-hard optimization problem requiring simultaneous satisfaction of labor regulations, coverage requirements, employee preferences and cost objectives. Existing approaches (genetic algorithms, integer programming, constraint programming) model 6-12 constraints at shift-level granularity and cannot guarantee regulatory compliance. They also lack support for multi-role, multi-skill heterogeneity, mandatory break scheduling with midpoint control, acuity-weighted workload equity, sub-shift granularity, inter-week stability, and cross-midnight shifts.
This paper presents CP-SAT: a Constraint Programming formulation for multi-role, multi-skill healthcare scheduling. CP-SAT enforces 14 hard constraints guaranteeing zero regulatory violations, while optimizing 15 soft objectives via a unified weighted penalty function. Contributions include a shift-window decomposition enabling break scheduling with centrality control, acuity-weighted workload equity, multi-granularity resolution from 15 minutes to 1 day, inter-week stability, and grid-offset preprocessing mapping cross-midnight shifts into a single scheduling day without solver changes.
CP-SAT is evaluated on 18 instances: five synthetic hospital units (10-33 nurses), 10 INRC-II benchmarks (5-80 nurses, up to 8-week horizons) and 3 NRP-23 compatible instances (10-25 nurses) with cross-midnight Night shifts. Results: zero hard-constraint violations across all 18 instances by construction; proven optimality on INRC-II n005w4 (objective 118, gap 0.0%, 104s); feasible schedules scaling to 179,800 variables and 351,425 constraints (80 nurses); service quality improved 50-67% over MOGA; and model size scaling near-linearly at approximately 4,400 variables per employee. The formulation enforces 29 total constraints (14 hard, 15 soft), nearly three times the industry average. 

---
# DERELAB: Probing Defeasible Reasoning and Confirmation Bias in LLMs with a Generative Benchmark 

**Authors**: Jayanta Sadhu, Sayem Shahad, Kenneth Marino  

**Link**: [PDF](https://arxiv.org/pdf/2608.30413)  

**Abstract**: Defeasible reasoning is a type of reasoning where inferences are drawn from plausible current evidence, but can be retracted upon the introduction of newer evidence. Although recent studies have examined language-model behaviors in defeasible reasoning, the datasets have been static and lack wide coverage of non-monotonic reasoning categories. We introduce DeReLab, a generative framework that produces multi-turn belief-updating conversations from parameterized graph structures across default and inheritance reasoning, with formally verified ground truth at every turn, enabling controlled measurement of how models respond to confirming and disconfirming evidence. This controlled generation process creates a testbed for experimental designs that isolate specific reasoning demands. Applying this capability to the study of confirmation bias, we evaluate nine open and proprietary large language models and find that nearly all exhibit a systematic tendency to accept congruent evidence while resisting incongruent updates, with several models correctly identifying a weakening update yet failing to revise their conclusion. We believe our work and findings will facilitate future research on evaluating language models in defeasible reasoning. 

---
# Dense Clinical Contrasts Enhance Medical Knowledge Updating in Large Language Models 

**Authors**: Yangmin Huang, Shu Quan, He Geng, Xin Ye, Qianyun Du, Zhiyang He, Jiaxue Hu, Xiaodong Tao  

**Link**: [PDF](https://arxiv.org/pdf/2608.30405)  

**Abstract**: Medical knowledge changes continually, making large language models vulnerable to relying on outdated yet clinically plausible information. We study whether the format of supervision affects medical knowledge updating under a matched training-budget setting. We introduce SEER-Bench, a temporally anchored oncology-staging benchmark curated from the latest versioned SEER Research Data release, and render identical medical update events from NCCN oncology guidelines into four supervision formats: EMQ, MSQ, FITB, and SAQ. Across SEER-Bench and HealthBench Professional, EMQ gives the most stable external transfer and retention among same-budget SFT variants. With EMQ supervision, the updated 4B model produces competitive results on temporally anchored oncology staging, reaching 64.8% answer accuracy and 59.6% rationale accuracy on SEER-Bench. Diagnostic analyses suggest that EMQ exposes denser clinical contrast signals while preserving discriminative representations with smaller movement from the base model. These results show that medical knowledge updating depends not only on the update algorithm, but also on how knowledge is structured as supervision. 

---
# Scaffolding Foundation Models into Physical-World Agents Pushes the Frontier of Long-Horizon Navigation 

**Authors**: Zixing Lei, Gengze Zhou, Xiong-Hui Chen, Jiazhao Zhang, Yiyang Huang, Hang Yin, Haoqi Yuan, Qi Wu, Weixin Li, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.30396)  

**Abstract**: Long-horizon physical-world agents must reason over distant goals while grounding decisions in reliable closed-loop behavior. Today's foundation models split these capabilities: vision-language models (VLMs) infer missing information and adapt high-level plans but remain brittle and inefficient at repeated navigation grounding, while navigation foundation models (NFMs) robustly execute semantic goals but operate as bounded episodes without persistent task-level reasoning. We introduce NavMCP, an agentic scaffolding framework that couples a VLM reasoning agent with an NFM executor for long-horizon exploration. The VLM decides what evidence to seek, where to search, and when to stop, while the NFM grounds each semantic sub-goal into closed-loop navigation. Three channels structure their collaboration: intent translates evidence needs into navigation calls, observation converts rollouts into source-grounded trajectory evidence, and memory accumulates findings, negative evidence, and unresolved goals across calls. This design turns isolated navigation rollouts into persistent embodied interaction without retraining either model. On Embodied Question Answering, NavMCP achieves state-of-the-art results on HM-EQA, MT-HM3D, and EXPRESS-Bench. Under matched agent and executor backbones, it outperforms an episodic interface by 14.9 percentage points on HM-EQA. On a Unitree Go2, NavMCP reaches 78.3% success, with its margin over the strongest baseline growing from 10 to 45 points as the task horizon increases. These results demonstrate the potential of scaffolding complementary foundation models into long-horizon physical-world agents. 

---
# Augmenting Human Performance with an XR Agent Learning from Online Behavior and BCI Evidence 

**Authors**: Ziheng Li, Xichen He, Haoyan Chen, Charlie Zou, Sheng Bai, Benjamin Yang, Mengyuan Wu, Jake Ledner, Yi-Jie Cheng, Akito Yamauchi, Dishita G Turakhia, Steven Feiner, Paul Sajda  

**Link**: [PDF](https://arxiv.org/pdf/2608.30369)  

**Abstract**: We present OLIVE, a framework for adapting a foundation model to provide real-time assistance in temporally demanding, high-stakes, and dynamic tasks. We show that passive EEG, fused online with behavioral evidence, can meaningfully extend the number of targets users detect and engage beyond their unaided action bandwidth. OLIVE learns from both explicit behavioral signals (the targets the user shoots down in an XR first-person shooter game) and implicit physiological signals (fixation-locked EEG) to provide timely guidance, continuously adapting a frozen vision-language model's inference on which items are task-relevant by jointly estimating per-source reliability without manual labels or offline training. Through three user studies, including two live deployments of an assistive agent driven by OLIVE in XR, we show that OLIVE Pareto-dominates prior test-time adaptation frameworks, achieving the highest convergence rate at comparable convergence speed. Combining implicit physiological and explicit behavioral signals, the OLIVE agent produces the largest and most reliable within-session improvement to a user's ability to detect and engage targets, largely independent of the individual's skill. When the target switches silently, the agent that uses both behavioral and physiological signals reconverges significantly faster than the behavior-only agent (1.27 times faster on average, p = .008), restoring trustworthy guidance at the moment the task changes, precisely when reliable assistance matters most. 

---
# Will the User Ever Know? Covert Indirect Prompt Injection on Tool-Using LLM Agents 

**Authors**: Yunseok Lee, Yunji Kim, Woojin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.30362)  

**Abstract**: As LLM agents take real-world actions through tools, indirect prompt injection (IPI) has emerged as a serious threat. The standard metric, Attack Success Rate (ASR), counts whether an injection succeeds but ignores what the user notices in the agent's final response. Looking at successful injection traces, we find two distinct outcomes: the agent executes the injection while returning an otherwise normal response, or reports the injected action in its final response, giving the user a chance to notice. We call these covert and overt successes. From the user's perspective, we decompose ASR into the Covert Success Rate (CSR), counting successes leaving no trace in the final response, and the Overt Success Rate (OSR), counting successes the user can detect. To understand what drives the gap, we analyze successful trajectories and find that the agent's behavior after the injection separates covert from overt: covert traces hand control back to the user task before ending, while overt traces end at the attack itself. This split follows from the ReAct format, where the final response summarizes the most recent action. Building on this observation, we propose ICoA (Induced Covert Attack), an IPI attack designed to induce covert outcomes by steering the agent back to the user task after executing the injection. Across four target models on AgentDojo, ICoA achieves the highest CSR, with gains of 3.79-12.01 percentage points over the strongest baseline. 

---
# Co-Annotator: Expert-Distilled ViT and VLM for Visual and Documentation Guidance in Age-Related Macular Degeneration 

**Authors**: Ziheng "Leo" Li, Benjamin Freeman, Akshay Raman, Kavin Aravindhan Rajkumar, Xinxin Fang, Rishabh Srivastava, Steven Feiner, Kaveri A. Thakoor  

**Link**: [PDF](https://arxiv.org/pdf/2608.30352)  

**Abstract**: Clinical AI often optimizes predictive performance without engaging how clinicians decide where to look and what to write. We present Co-Annotator, which distills expert gaze and dictation into two guidance components: a gaze-aligned Vision Transformer producing fixation-aligned areas of interest (AOIs), and an ontology-bounded vision-language model (VLM) that pre-fills editable biomarker summaries for retinal optical coherence tomography (OCT). We first collect expert gaze and dictations (US1) to train the models, significantly improving diagnostic accuracy and biomarker generation. We then deploy the system with ophthalmology residents: a controlled resident study (US2) confirmed each modality is safe and independently beneficial, with AOI guidance producing lasting perceptual efficiency gains through post-guidance carryover and VLM guidance more than doubling biomarker documentation breadth. In a combined deployment across two academic institutions (US3), providing both modalities simultaneously produced efficiency gains that substantially exceeded either modality alone: correct diagnoses per minute increased by 40% and comment editing time fell by 67%, without compromising diagnostic accuracy. Notably, neither modality improved efficiency during guidance in US2, which makes the in-guidance efficiency gain under combined guidance in US3 the more striking result. Expert-distilled multimodal guidance can remove two distinct clinical workflow bottlenecks at once (visual search overhead and documentation burden) without compromising the diagnostic accuracy clinicians already achieve. 

---
# Answer Probing-Guided Search for Diverse Solution Exploration of LLMs 

**Authors**: Yi Fang, Que Shen, Chengpeng Li, Boyi Deng, Wei Shi, Wenjie Wang, Fuli Feng, Fengli Xu, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.30345)  

**Abstract**: Generating multiple diverse and high-quality solutions is valuable for many applications, such as code-test generation and drug discovery. However, Large Language Models (LLMs) tend to converge on a single high-confidence solution during inference, limiting exploration of alternative valid solution paths. Existing test-time methods promote diversity through tree-like search and prune semantically similar branches using response-level semantic embeddings. However, we find that such embeddings are easily confounded by linguistic and stylistic similarities, making it difficult to distinguish genuinely distinct solution paths. To address this, we introduce Answer Probing, which probes the potential answer an LLM would reach from an intermediate reasoning path. We demonstrate that the hidden states of probed answers more effectively differentiate distinct solution paths than semantic embeddings, and the perplexity of probed answers serves as a practical proxy for reasoning correctness. Based on these findings, we propose Answer Probing-Guided Tree Search (APTS), which guides the tree search by the probed answers' hidden state similarity and perplexity. Experiments on three reasoning tasks across two LLMs show that APTS consistently enhances solution diversity, demonstrating its effectiveness and robustness. 

---
# Ignorance or Incompetence? Constructing Knowledge-Gated, Verifiable Tasks for LLM Agents 

**Authors**: Hanlin Tian, Minhao Li, Yu Mi, Sihan Zhu, Zhao Yang, Yuxiang Wang, Hongquan Zhu, Qiufei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.30322)  

**Abstract**: Professional agent tasks often depend on conventions that are absent from public corpora, yet benchmarks rarely control whether an agent has access to those conventions. We introduce a knowledge-gated task-construction protocol that separates a task instruction from a compact artefact containing private conventions, reference tables, and utility operators. Construction-time provenance, byte-identical task instructions across the provided- and withheld-artefact conditions, leak audits, and executable witnesses make dependence on the artefact explicit and testable. Across fifteen calibration tasks, one frontier agent configuration achieves a 68.0% pass rate with the artefact and 0% without it; on one task, a plausible but incorrect artefact also yields 0% across five trials. Deterministic solvers and rule corpora provide exact ground truth for structured tasks, while named criterion-level rubrics support outputs that cannot be checked by a single executable oracle. A configuration-relative calibration screen retains seven tasks satisfying our five-trial empirical knowledge-gating screen. These experiments validate the behavior of the construction protocol; they do not establish that the retained tasks improve post-training. We publicly release part of the task suite and supporting tooling at this https URL. 

---
# SimCRAFT: Distilling Remote Sensing Agents via Synthetic Trajectories and Contextual Retrieval-Augmented Fine-Tuning 

**Authors**: Haoran Wang, Jing Yao, Xu Yang, Zeqing Wang, Yang Zhang, Pedram Ghamisi, Zhengchao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.30277)  

**Abstract**: The unprecedented surge in Earth observation data volume and diversity has exposed a critical bottleneck for traditional manual workflows, catalyzing the emergence of Remote Sensing (RS) Agents. However, the practical deployment of these advanced agents is severely hindered by their heavy reliance on large-scale general-purpose LLMs, which lack deep domain expertise and impose prohibitive infrastructure demands. To resolve this, we propose SimCRAFT, a model-agnostic framework that distills sophisticated RS orchestration capabilities into a compact 7B-scale model. Addressing data scarcity, we first pair a multiagent synthesis engine with a Mock Execution Engine that checks schema correctness, inter-tool dependencies, and sensor/tool compatibility, producing SimRS-14k, a large-scale, constraint-validated workflow planning corpus. Second, we propose Contextual Retrieval-Augmented Fine-Tuning (CRAFT) that finetunes the model to reason analogically by adapting retrieved Standard Operating Procedures to novel queries under a noise-robust objective, generalizing RAFT to multi-step RS workflow planning without mechanical copying. Extensive experiments demonstrate that SimCRAFT-7B significantly outperforms openweights LLMs and rivals advanced closedsource models and specialized RS agents, while reproducing across three 7B backbones. This work contributes a competitive open-weights baseline for lightweight RS intelligence, enabling efficient autonomous deployment under resource-constrained or resource-conserving conditions. 

---
# Generating Workflow DAGs from Natural Language with Non-Reasoning LLMs 

**Authors**: Anand Iyer, Bhanu Khetharpal, Srinivas Upadhya, Ramkumar Rajagopal  

**Link**: [PDF](https://arxiv.org/pdf/2608.30250)  

**Abstract**: This paper addresses the problem of translating natural-language routing rules written by business administrators into executable workflow graphs for enterprise contact centers. Each target is a directed acyclic graph (DAG) of conditional actions with parallel branches, hit-first fallback chains, and per-branch Boolean predicates, encoded in the JSON dialect of a commercial routing platform. We show that neuro-symbolic decomposition enables lower-cost, non-reasoning large language models to generate complex workflow DAGs at production-relevant quality without expensive extended-reasoning models. Our central diagnostic is an emission-density bottleneck: on a 635-rule benchmark of manufactured synthetic data, models select the correct graph nodes with high accuracy but increasingly misconfigure attributes and Boolean grouping as the number of interdependent nodes emitted in one pass grows. We therefore move combinatorial graph construction from the model into a deterministic compiler driven by a compact intermediate representation, with a learned registry-selection front end that focuses generation on relevant vocabulary. Across four models, the full system reaches approximately 89% LLM-judge validity, approximately 90% exact-match condition accuracy, and 99-100% valid JSON while using roughly half the per-rule prompt tokens of a monolithic prompt. On GPT-5.3-chat, the method improves judge validity by 24 percentage points and achieves statistical equivalence to a reasoning model's out-of-the-box quality, although an approximately 8-point frontier gap remains. We also present a deployment path and transferable lessons for structured-generation applications. 

---
# LLM-Based Knowledge Graph Completion Combining Discrete Structural Coding with Similar Entity Information 

**Authors**: Jiaqi Wang, Dongying Lin, Yang Yang, Yinan Liu, Bin Wang, Xiaochun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30235)  

**Abstract**: Knowledge graph completion requires models to use both textual descriptions and relational structure. Existing LLM-based methods either encode KG structure as discrete tokens or refine a restricted set of candidate entities, and these two directions have largely been studied separately. We propose CoSC for LLM-based KGC, which combines discrete structural coding with similar entity information. Specifically, an LLM generates an initial candidate entity ranking from discrete structural codes, after which information from entities with structures similar to that of the query entity refines the ranking. Experiments on FB15k-237 show that CoSC outperforms existing baselines on MRR and Hits@10 while remaining competitive on Hits@1. 

---
# CoLa-ICD: A Knowledge-Enhanced Framework for Long-Tail Automated Medical Coding 

**Authors**: Yihang Cheng, Veronica Liesaputra, Andrew Trotman  

**Link**: [PDF](https://arxiv.org/pdf/2608.30234)  

**Abstract**: Automatic medical coding assigns ICD codes to clinical notes, but it remains challenging due to long documents, imbalanced label distributions, and diverse terms. These challenges are especially severe for rare codes, which have limited training instances and are easily confused with semantically similar labels. We introduce CoLa-ICD, a knowledge-enhanced framework for long-tail prediction. CoLa-ICD enriches ICD labels with external terms, models dependencies among related codes, and learns stronger alignment between label semantics and clinical evidence for long-tail prediction. Experiments show that CoLa-ICD improves long-tail prediction with larger gains in larger and sparser label spaces and achieves state-of-the-art performance in AUC, F1, and P@k. Our code is available at this https URL. 

---
# Rethinking the Test-Time Prompt Tuning Objective from the Perspective of Calibration 

**Authors**: Jungwon Choi, Hyeonseo Jang, Kibok Lee, Eunwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.30230)  

**Abstract**: Test-time prompt tuning (TPT) has emerged as a powerful paradigm, refining prompts for each test sample via entropy minimization (EM) over multiple augmented views. However, we identify a limitation in the standard EM-based adaptation: it inherently drives the model toward overconfident predictions disregarding sample-specific uncertainty, leading to significant calibration degradation. To address these limitations, we propose a new objective that replaces the conventional EM loss by aligning the original-view prediction with a target distribution derived from augmented views via cross-entropy, while adversarially incorporating the entropy of the target distribution to capture sample-specific uncertainty. Furthermore, to better construct this target distribution, we apply confidence-aware temperature scaling to each augmented-view prediction according to its confidence, sharpening confident predictions while softening uncertain ones. This formulation allows the model to increase confidence only when the target distribution is reliable, while preserving uncertainty when it reflects ambiguous or conflicting augmented-view predictions. Extensive experiments across diverse benchmarks demonstrate that our approach not only achieves state-of-the-art accuracy but also significantly improves model calibration. 

---
# LaMoC: Loss-Aware Modular Compression for LLMs 

**Authors**: Mohanad Odema, Jacob Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.30226)  

**Abstract**: Modular compression has enabled considerable parameter reduction in LLMs while preserving strong language understanding and downstream task accuracy. However, existing joint modular compression methods primarily rely on activation statistics, leaving loss-sensitivity information and its module-level characterization underexplored. We investigate addressing this gap with LaMoC, a loss-aware modular compression methodology that blends activation and Empirical Fisher statistics through gradient-error alignment. LaMoC improves joint compression by selecting compression statistics that better align local module reconstruction error with the downstream loss. Our contributions are three-fold: (1) We characterize the Empirical Fisher as a module-level loss-aware proxy that can be blended with the activation statistics required for compression. (2) We reformulate joint modular compression as a two-tiered optimization problem that minimizes module reconstruction error while tuning the activation and gradient information blending rate. (3) We implement an empirically driven methodology with statistical validation to solve the resulting compression problem. We evaluate LaMoC across four model families spanning eight models. On the 4-8B models, LaMoC achieves an average 2.5% reduction in perplexity and a 1% relative improvement in task accuracy over state-of-the-art modular compression methods. 

---
# SPARK: Skeleton-Guided Reasoning Synthesis from Large-Scale Scientific Literature 

**Authors**: Yu Li, Wei Li, Xin Gao, Mengyuan Sun, Xiaoyang Wang, Qizhi Pei, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.30214)  

**Abstract**: Scientific reasoning remains challenging for open-source models, largely due to the lack of high-quality scientific reasoning data. Existing datasets are often dominated by factual recall or formulaic problem solving, with limited emphasis on mechanism understanding, evidence-grounded reasoning, and hypothesis evaluation. To address this, we introduce SPARK (Scientific Paper Abstracted Reasoning sKeleton), a paper-oriented synthesis framework built on Sci-Base, a large-scale corpus of research papers spanning 10 scientific disciplines. Instead of directly converting papers into question-answer pairs, SPARK treats the claim-evidence-derivation structure of a paper as the fundamental unit of reasoning synthesis. Specifically, SPARK (1) distills each paper into a compact reasoning skeleton capturing its central claims and supporting evidence, enabling self-contained question generation, and (2) synthesizes reasoning tasks from four scientific perspectives: mechanistic reasoning, hypothesis falsification, quantitative derivation, and boundary calibration. A final consistency verification stage further removes unsupported or contradictory outputs. Using this framework, we construct Spark-234K, a scientific reasoning dataset with substantially higher difficulty and diversity than existing resources. Experiments show that Spark-234K consistently outperforms existing scientific reasoning datasets while achieving stronger performance with significantly fewer training samples. 

---
# FaVOR: LLM-Based Agentic Framework for Factor Mining via Empirical Validation 

**Authors**: Hyeonjin Kim, Minseok Kim, Seunghyeon Jung, Sujin Pyo, Huisu Jang, Woojin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.30192)  

**Abstract**: Traditional finance relies on experts to hand-craft factors through a principled process grounded in economic rationale. Recent LLM-based multi-agent systems have automated this process, scaling factor mining far beyond manual effort. However, these automated approaches optimize directly for returns and rarely check whether a generated factor still expresses the economic hypothesis that motivated it. We identify this inconsistency between mathematical form and economic meaning as a structural failure mode of return-oriented automation. The resulting factors blur the line between real signals and spurious correlations and break down across regime shifts. We propose FaVOR (Factor Validation through Observable Reasoning), an agentic framework that restructures factor mining around hypothesis-level evidence rather than return outcomes. In place of the standard hypothesis-to-formula leap, FaVOR enforces a three-stage consistency loop tying mathematical form to economic rationale throughout. (1) Decomposition splits a broad economic hypothesis into independent observable conditions. (2) Validation checks whether each factor reflects its intended condition. (3) Integration merges them into a composite whose structure remains interpretable. On the CSI 500 and S&P 500 in 2025, FaVOR outperforms existing baselines while remaining effective across regimes. FaVOR shows that hypothesis-grounded factor discovery produces signals that are interpretable by construction, regime-robust, and economically faithful. The code is available at this https URL. 

---
# A.X K2 Technical Report 

**Authors**: Cheolseung Baek, Dhammiko Arya, Eunki Kim, Gun Song, Gyoungeun Han, Hyunho Yang, Hyunjun Eun, Jin Kim, Junyoung Park, Juyun Wee, Minki Hong, Minkyung Park, Minsang Kim, Minsoo Kang, SaeRom Kim, Sangjin Kim, Sangyeol Lee, Seojin Lee, Seokhwan Jo, Seokyoung Hong, Seongho Choi, Seonghye Cho, Seongmin Ok, Sereimony Sek, Seungmo Cho, Seungsik Kim, Singon Kim, Sohee Park, Sooyeon Park, Subin Yi, Sungbin Yoon, Sungeun Lee, Sung Jun Cheon, Sungwan Kim, Sunwoo Lee, Tae Yoon Kim, Wonbeom Jang, Yohan Ra, Yong-jin Han, Youngjin Kim, Youngrang Kim, Yujin Kang, Yujin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.30181)  

**Abstract**: We introduce A.X K2, a 688B-parameter Mixture-of-Experts (MoE) language model trained from scratch as a high-performance foundation for \emph{agentic} applications. Trained on approximately 8.5T tokens---fewer than its predecessor, A.X K1---on a smaller but higher-quality mixture with substantially expanded agentic and software-engineering data, it nonetheless improves over A.X K1 across the board, by over 30 percentage points on some benchmarks, reflecting large gains in token efficiency. To support long contexts efficiently, we introduce Sparse Gated Attention (SGA), which combines sparse attention with gated attention, and adopt Gated Norm (GN) to stabilize large-scale training. SGA is trained natively at 128K through a \emph{sparse} indexer warmup that optimizes the indexer against its own sparse top-$k$ selection rather than the dense attention distribution, making adaptation markedly cheaper: each query reads only 2,048 positions, yet long-context quality is unchanged and A.X K2 scores 94.6 on RULER out to 256K. The outlier suppression of GN in turn keeps 4-bit NVFP4 serving within one point of FP8 accuracy. A simple yet effective Think-Fusion recipe further lets users switch between thinking and non-thinking modes within a single unified model. Extensive evaluations show that A.X K2 performs competitively against strong open-weight baselines, matching or exceeding them on math and Korean-language benchmarks. 

---
# VERA: Authority-Preserving Edge Revocation for Federated AI-Agent Workflows 

**Authors**: Lifei Liu, Haoran Yu, Xiaochong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30091)  

**Abstract**: Modern agent frameworks compose planners, tool agents, remote services, and shared specialists into runtime delegation graphs, but their revocation APIs still resemble token or subtree invalidation. When one delegation is withdrawn, the runtime must know which agents lose authority while independently authorized agents keep working. We study this authority consistency problem and introduce VERA (Verifiable Edge Revocation for Agents), a verifier-checkable revocation contract and API emitted by agent-runtime adapters as signed evidence. Under disjunctive authority, revoking edge e invalidates exactly T_intent(e,G) = reach(G) \ reach(G \ {e}), the agents whose every authorizing root path used e. Used as a contract, this target exposes two runtime failures: tree cascades over-revoke shared agents, while deployer-scoped cascades under-revoke cross-domain descendants. In a LangGraph framework-replt cells repeated 20 times yield 500compiled-framework traces and 2,000 valid signed delegation decisions; 13/25 cells contain runtime multi-parsharing and 8/25 contain cross-deployer shies 500/500 target proofs, preserves all320 alternate-parent shared-agent cases that tree cascade revokes, and rejects unauthorized signers and omission attacks. Baseline replay over 1,9that holder/node and tree-style targetscannot express this behavior. We further validate schema portability on A2A, AutoGen, and CrewAI artifacts: nine traces, including five executable Cregned delegation events that pass schema and signature checks. 

---
# Game-Agnostic Value Functions through Automatic JSON Feature Extraction 

**Authors**: Dien Nguyen, Diego Perez-Liebana  

**Link**: [PDF](https://arxiv.org/pdf/2608.30056)  

**Abstract**: JSON Bag-of-Tokens (JSON-Bag) is a recently proposed method to generically represent game trajectories by tokenizing their JSON descriptions. We introduce JSON-Bag VF, a game-agnostic approach to training value functions for game-playing agents using JSON-Bag prototypes. We show that this approach can be enhanced with Random Forest-based feature selection and a method to select game-stage-specific features. We evaluate JSON-Bag VF with One-step-look-ahead (JSON-Bag OSLA) on six tabletop games over different combinations of prototype-tokenization and feature selections. JSON-Bag OSLA outperforms baseline OSLA agents in most games. Our analysis also shows that feature selection significantly improves JSON-Bag VF and that feature selection is the most important factor in JSON-Bag VF performance, over prototype-tokenization. 

---
# Mitigating Over-Optimization in PRM-Guided Search in Mathematical Reasoning by Optimizing the Guide 

**Authors**: Taejong Joo, Diego Klabjan  

**Link**: [PDF](https://arxiv.org/pdf/2608.30051)  

**Abstract**: Process reward models (PRMs) provide dense step-level guidance for search-based reasoning, enabling inference-time compute to be allocated toward promising partial solutions. However, recent evidence suggests that PRM-guided search can over-optimize imperfect process rewards, pruning viable trajectories while expanding spurious ones. In this work, we theoretically show that directly leveraging PRM score is vulnerable to verifier noise through an extreme-value effect: non-viable prefixes become more likely to receive spuriously high scores as reasoning depth increase. Therefore, we formulate the PRM-guided search as a robust optimization problem over plausible reward perturbations, termed maximin PRM-guided search, leading to a training-free robust process supervision method that preserves promising alternatives when step-level scores are noisy. Maximin PRM-guided search mitigates this failure mode by reducing sensitivity to over-optimized PRM outliers. Without fine-tuning or online adaptation, maximin search consistently improves the PRM-guided search by 17-35\% on average, outperforming outcome- and step-level baselines in 14 out of 16 settings. Our source code is available at this https URL. 

---
# Spec2Twin-Chain: Orchestrating Bi-Level Optimization with LLMs for Blockchain Digital Twin Construction 

**Authors**: Haoting Zhang, Haoxian Chen, Jiayuan Sheng, Donglin Zhan, Zeyu Zheng, David D. Yao, Wenpin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30050)  

**Abstract**: Building a blockchain digital twin largely requires translating domain knowledge and specific system descriptions into a simulator architecture, calibrating its parameters against behavioral evidence, and validating the constructed twin. These steps are commonly performed through application-specific modeling efforts that can be difficult to reuse across systems and downstream decision problems. We consider automating this process through Spec2Twin-Chain, a framework that formulates blockchain digital-twin construction as a bi-level optimization problem. At the upper level, a large language model proposes and revises structurally admissible architectures using system specifications, behavioral evidence, and feedback from evaluated designs. At the lower level, a simulation-based optimizer calibrates the architecture-conditioned parameters under explicit objectives and guardrail constraints. The two levels iterate. The evaluated candidates at lower levels are retained in a global archive and used to guide subsequent proposals at upper levels. We conduct controlled experiments involving twin calibration, feedback-driven recovery, stress analysis, downstream policy optimization, and policy updating. The results demonstrate that the framework can construct behaviorally accurate twins, improve initial designs through iterative feedback, and reuse calibrated twins to support downstream decisions. 

---
# Can LLM Agents Discover? Evaluating Creativity on ML Engineering Tasks 

**Authors**: Shitanshu Bhushan, Yunxiang Zhang, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30047)  

**Abstract**: Recent AI systems promise autonomous scientific discovery, claiming to discover algorithms and produce research papers, yet understanding whether they exhibit creativity, the capacity to produce solutions that are both novel and useful, remains an open question. We present a framework for evaluating multi-turn LLM research agents' creativity using ML engineering tasks as a testbed, through three dimensions: P-Creativity (psychological novelty: novel relative to the agent's own prior solutions within a run), H-Creativity (historical novelty: novel relative to the corpus of human solutions), and Usefulness (task performance). Evaluating two agent frameworks, AIDE and AIRA-Dojo, on 10 Kaggle-style machine learning tasks from MLE-Bench, we develop an LLM-as-a-Judge pipeline and verify its strong correlation with human creativity judgments, providing a reliable automated metric for P-Creativity evaluation at scale. Applying this pipeline to agent trajectories, we find: (1) all agents exhibit declining P-Creativity as they transition from exploration to exploitation; (2) LLMs exhibit greater H-Creativity than medal-winning humans, yet achieve lower performance. Our findings reveal that current agents can explore novel regions of the solution space but lack the capacity to convert this novelty into improved task performance. 

---
# Balance of Benchmarks: Semantic Density Reweighting for Benchmark Multiplicity and Task-Conditioned Evaluation 

**Authors**: Jhen-Ke Lin  

**Link**: [PDF](https://arxiv.org/pdf/2608.30044)  

**Abstract**: Language models are commonly compared by averaging scores across a benchmark list with equal weight. Such lists grow through publication outside an explicit measurement design, so equal weighting turns the density of published benchmarks into an implicit capability weight: densely benchmarked regions count repeatedly. We introduce Balance of Benchmarks (BoB), which embeds benchmark descriptions and assigns each benchmark an inverse-density semantic weight. Nearby entries share aggregate influence at a disclosed density scale. After equating heterogeneous scores onto a common latent scale, a residual field uses the same geometry to condition model rankings on a task query. The two components serve distinct empirical roles. On a snapshot of 586 models and 14 benchmarks, BoB predicts which models are unusually strong on a held-out task beyond their general ability, reaching a profile correlation of 0.462 compared with 0.049 under equal weighting. It also limits the influence of densely repeated benchmarks on the aggregate. After adding four copies of each benchmark in turn, the resulting rankings retain a Kendall tau of 0.995, compared with 0.936 under equal weighting. The residual field therefore provides task-conditioned prediction, and inverse-density weighting provides robustness to benchmark multiplicity. Together, they turn benchmark-list composition from an incidental property of evaluation suites into an explicit, controllable part of measurement design, providing a principled foundation for task-aware and multiplicity-robust model evaluation. 

---
# Beyond Uncertainty: Multi-Solver Disagreement Rewards for Self-Evolving Reasoning Curricula 

**Authors**: Vinoth Selvendran, Zhanming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30035)  

**Abstract**: Self-evolving reasoning frameworks train a Challenger to generate questions exposing a Solver's weaknesses, creating adaptive curricula without human data. However, existing approaches use a single solver's sampling uncertainty as the Challenger's reward. This creates a fundamental bottleneck: as the solver grows confident on the Challenger's question distribution, all sampled answers converge identically, collapsing the reward to zero and starving the Challenger of learning signal. Critically, this single-model reward cannot distinguish genuinely easy questions from those that merely align with one solver's learned biases. We propose a multi-solver disagreement reward using a heterogeneous ensemble varying in model capacity and sampling temperature. A normalized Shannon entropy over the ensemble's per-question plurality answers explicitly rewards questions where solvers produce conflicting solutions---capturing difficulty as inter-model divergence rather than intra-model sampling variance. This richer gradient enables the Challenger to discover questions targeting true capability boundaries, producing a curriculum that forces downstream Solvers to develop robust reasoning strategies generalizing across problem types. Our approach is a drop-in reward function replacement requiring no framework modifications or additional data. Experiments with Qwen3-4B show that Solvers trained on disagreement-Challenger questions achieve +1.34 points average improvement on competition-math benchmarks (MATH-500, AMC, Olympiad), suggesting that multi-solver disagreement provides a complementary and scalable signal for curriculum generation in self-play reasoning systems. 

---
# Interpreting and Steering for Safe and Correct Code Generation 

**Authors**: Hao Yan, Ziyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2608.30025)  

**Abstract**: Large language models (LLMs) frequently generate source code containing vulnerabilities, yet little work studies the internal mechanisms that distinguish safe from vulnerable generation in them. In this work, we systematically perform a mechanistic interpretation of LLMs, aiming at both understanding how code safety-vs-vulnerability is represented or driven by components in an LM and turning the insights into actionable steering strategies to encourage safer code generation. To this end, we introduce CodeSec-Pairs, a dataset of 9,342 Python safe-and-vulnerable contrastive code pairs, sampled from Llama-3.1-8B-Instruct. Utilizing the dataset, we explore approaches to localize layers and attention heads that relate to code safety, and further experiment with different steering strategies for inference-time vulnerability reduction. In particular, we propose DuoSteer, a double-steering approach that simultaneously applies safety and code-correctness steering to attention heads. In experiments over five vulnerability types, DuoSteer leads to an average of -26.9% vulnerability rate reduction and +7.5% functional correctness improvement, which outperforms not only other steering variants but also prompting and supervised fine-tuning baselines. The advantage also replicates on Qwen-2.5-Coder-7B-Instruct with another 2,500 contrastive pairs sampled from that model. 

---
# Automatic Conversion of NICE Guidelines to an Executable Computational Model Using Large Language Models 

**Authors**: Ashvin Gupta, Denys Prociuk, Alessandra Russo, Brendan C. Delaney  

**Link**: [PDF](https://arxiv.org/pdf/2608.30022)  

**Abstract**: Introduction: NICE guidelines provide evidence-based recommendations for clinical care but remain largely in unstructured natural language. Existing approaches to converting them into computable representations often focus on individual diseases, require substantial manual encoding, and do not scale. Large language models (LLMs) may enable much of this translation to be automated. Methods: We present an end-to-end approach that converts textual clinical guidelines into executable models capable of generating explainable, patient-specific recommendations. A stepwise LLM-based transformation with in-context examples produces human-inspectable intermediate artifacts. We apply the approach to NICE pancreatic and lung cancer guidelines, use expert review to assess rule alignment, and evaluate the executable pancreatic cancer model on 20 patient vignettes. Results: Expert review showed strong alignment between the source guidelines and generated executable models. Most discrepancies were partial omissions rather than incorrect logic, while hallucinated or fundamentally incorrect rules were rare. On the patient vignettes, the executable model achieved an F1 score of 82.5%. Conclusion: LLMs can transform natural-language NICE guidelines into interpretable, executable models that preserve guideline structure, support transparent inspection and modification, and generate patient-specific recommendations. These findings demonstrate the feasibility of scalable automated generation of computable clinical guidelines. 

---
# AutoCRAT: Within-trajectory Joint Control of Stochasticity and Compute for LLM Reasoning 

**Authors**: Hanjun Luo, Qiushi Liu, Jingya Zhang, Haihong Pang, Jiaheng Wen, Yifei Ma, Yu Yao, Chengxi Zhang, Hanrong Zhang, Yankai Chen, Hanan Salam  

**Link**: [PDF](https://arxiv.org/pdf/2608.29988)  

**Abstract**: Large language models (LLMs) achieve strong reasoning performance, which depends critically on inference-time decisions. Yet these decisions are commonly handled by static, one-size-fits-all policies, limiting adaptation to diverse tasks and reasoning stages. Recent adaptive methods partially address this limitation, but they primarily adapt either decoding stochasticity (how the model explores) or reasoning compute (how long the model reasons) in isolation, leaving their interaction within a single reasoning trajectory unmodeled. To address this challenge, we shift toward a within-trajectory joint control view, and instantiate it in AutoCRAT, a decoder-side controller for frozen backbones. Using only signals available during decoding, AutoCRAT jointly adjusts sampling stochasticity and reasoning budget during generation. AutoCRAT operates over a discrete action space and updates control decisions only at semantic boundaries, improving stability while remaining responsive to the evolving reasoning process. Comprehensive evaluation across 6 benchmarks demonstrates that AutoCRAT (I) uses 13.8-52.7% fewer inference tokens on average than recommended static configurations, (II) surpasses recommended static and adaptive baselines by 1.5-4.5% in relative accuracy, and (III) enjoys strong cross-backbone transferability. 

---
# An Open-Source, Event-Driven Pipeline for Cryptocurrency Market Data: Ingestion, Forecasting, and On-Chain Fraud Detection 

**Authors**: Basil Sajid Shaikh, Melrick Mascarenhas, Nuzhat Faiz Shaikh  

**Link**: [PDF](https://arxiv.org/pdf/2608.29973)  

**Abstract**: Cryptocurrency markets generate high-frequency, multi-source data that is expensive to work with unless a team already has commercial-grade streaming and warehousing infrastructure in place. This paper describes a fully open-source pipeline that reproduces the behavior of a cloud-native, event-driven system -- file arrival triggering a message, a message triggering compute -- entirely on commodity hardware, using Apache Kafka and a filesystem-watching poller in place of managed cloud triggers. The pipeline partitions historical Gemini exchange data into hourly and minutely files, ingests them asynchronously through two independently grouped Kafka consumers (one for audit logging, one for Spark-triggered ETL), and lands cleaned output in a PostgreSQL warehouse with historical and aggregated schemas plus asset-specific data marts. We use the resulting Bitcoin data mart to compare a seasonal ARIMA model against a single-layer LSTM network for price forecasting, and separately apply Random Forest and Gradient Boosting classifiers, with additional engineered features, to the public Ethereum fraud detection benchmark introduced by Farrugia et al. We report the architecture, the modeling methodology, and the resulting metrics, and we are explicit about the limitations of comparing forecasts issued at different horizons and of evaluating fraud detection on a static, already-labeled dataset. 

---
# EDGE: Engine for Deterministic Graph Evaluation through Conversation Simulation from Graph Structured DSL Configuration 

**Authors**: Ram Kulathumani, Regunathan Radhakrishnan, Anupam Tripathi, Xiangbo Mao, Roshanak Omrani, Keshav Somani, Shwet Kamal Mishra, Shayna Lurya  

**Link**: [PDF](https://arxiv.org/pdf/2608.29971)  

**Abstract**: As agentic systems evolve into complex multi agent orchestration workflows, there is a growing and critical need for systematic frameworks that measures an agent's behavioral consistency and determinism. In this paper, we introduce a formal evaluation methodology that is grounded in AgentGraph, a planner powered by a domain specific language that represents agent reasoning through a dynamically adjustable directed graph. We leverage this structural formalism and utilize graph traversal algorithms that exhaustively enumerate conversational paths, forming a comprehensive evaluation set that captures the agent's complete behavioral space. We then systematically replay these reproducible trajectories to compare observed outputs and state transitions against the intended DSL specification. To quantify reliability, we define novel metrics that measure response and trajectory determinism, structural adherence and semantic consistency across both exact replays and their linguistic variants. Our system's results demonstrate that agents configured using frameworks like AgentGraph and LangGraph with explicitly structured node transitions show superior determinism over agents that are not configured with controlled transitions. 

---
# Review Before Trust: Source-Grounded Integrity Gates for AI-Assisted Personal Health Records 

**Authors**: Nora Girda, Adrian Groza  

**Link**: [PDF](https://arxiv.org/pdf/2608.29965)  

**Abstract**: Large language models can convert medical documents into structured data, but plausible output may still be unsupported by the source. Persisting such output in a longitudinal health record, a record that accumulates patient information over time, therefore creates an integrity risk: unverified data may influence later summaries, trends, or preventive-care computations. We introduce an evidence-gated trust-promotion model that keeps generated data provisional until a deterministic monitor verifies it against the source document. The monitor admits a candidate for a specified downstream use only when the source contains a unique supporting quotation, the relevant fields occur within the same laboratory row, and the required provenance is preserved. The generator cannot approve its own output, missing or ambiguous evidence causes refusal, and refused candidates remain available for human review rather than being silently discarded. We implement the model in Medical DataCloud, a personal health-record application, and evaluate it through automated tests and a replay of saved extraction outputs. All 22 conformance and mutation tests pass. The replay covers nine historical laboratory PDF reports containing 102 manually labelled rows. The reports produce 97 numeric candidates: schema validation accepts all 97, an earlier packet-level evidence check accepts 94, and the hardened quotation- and row-level policy admits 72 while retaining 25 for review. The study evaluates system integrity rather than clinical correctness or clinical safety. The results demonstrate the technical feasibility of an enforceable boundary that prevents generated claims from authorizing their own reuse in a longitudinal health record. 

---
# SearchWiki: Learning to Build and Navigate Knowledge Wikis for Active Information Seeking 

**Authors**: Guransh Singh, Vishwajeet Kumar, Arkadeep Acharya, Adnan Qidwai, Jaydeep Sen, Sachindra Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2608.29953)  

**Abstract**: Flat retrieval-augmented generation treats a corpus as a bag of chunks, discarding document hierarchy and cross document structure. We introduce SearchWiki, a harness framework that synthesizes a corpus into a hierarchical, typed, navigable wiki and trains an agent, WikiResearcher-9B, to retrieve information through multi-turn tool use. The wiki organizes knowledge into three layers - document overviews, cross- document topic pages, and page-level source records; enabling progressive refinement of retrieval when initial lookup misses. We optimize the agent's navigation policy with on-policy reinforcement learning with a multi-component reward function balancing answer correctness, retrieval quality and trajectory efficiency. Evaluation on ViDoRe-V3 (8 domains), FinanceBench, and memory benchmarks (LoCoMo, LongMemEval, PersonaMem-v2) shows that WikiResearcher- 9B which is our RL-tuned Qwen 9B model, significantly outperforms same-size untrained baselines and exceeds or matches larger external models. SearchWiki paired with WikiResearcher-9B demonstrates that learned navigation over structured corpora is a superior alternative to flat retrieval. 

---
# Spatial Matryoshka Training for Multi-Granularity Visual Document Retrieval 

**Authors**: Trishan Singha Roy, Arkadeep Acharya, Vishwajeet Kumar, Jaydeep Sen, Sachindra Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2608.29951)  

**Abstract**: Multi-modal late-interaction retrievers achieve strong retrieval on visually rich documents by representing each page as per patch embeddings and matching at the token level. However, this approach incurs high storage costs. Existing compression methods typically fix a single compression level at indexing time, limiting flexibility. We present ColSNAP (Spatial Nested Average Pooling)1, a training method that generates a nested hierarchy of compression levels directly from a backbone's patch grid. By spatially pooling patch embeddings into pro- gressively coarser tiers and training all tiers simultaneously, a single model learns to support retrieval at multiple compression levels without architectural changes. Crucially, a single encoding pass yields every tier, enabling the accuracy-storage trade-off to be configured at indexing time to match avail- able storage budgets, rather than being fixed during training. We demonstrate that models trained using ColSNAP maintain near full-resolution retrieval performance under substantial compression and that ColSNAP transfers effectively across multiple late-interaction backbones, and achieves most of its improvements via a lightweight adaptation stage applied to a pre-trained retriever. 

---
# AcrossWAM1.0:A Modular Latent World-Action Stack for Compact Robot Policies 

**Authors**: Yafei Zhang, Nan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.29937)  

**Abstract**: Latent world-action models avoid rendering future pixels by predicting an action-relevant visual subgoal in feature space. LaWAM established this formulation, but its original presentation left the world model, multimodal backbone, and deployment checkpoint tightly coupled. We introduce AcrossWAM1.0, a modularization and scaling study of this latent world-action stack. Rather than presenting latent subgoals as a new algorithm, we make the module boundary explicit: a policy adapter produces latent-action and action-generation contexts; a retained latent world decoder grounds the predicted transition in the current scene;and a flow-matching expert generates continuous action chunks. We further separate training-only teachers from the inference graph and provide a verifiable deployment export. On 2,000 paired LIBERO episodes, replacing a Qwen3-VL-2B backbone with Qwen3.5-0.8B yields 97.45% success versus 98.00% for the 2B model (a-0.55percentage-point difference; exact McNemarp=0.266). This does not prove equivalence, but it meets a prespecified two-point retention criterion. The compact, inference-reachable checkpoint contains 1,472.6M unique parameters, 42.4% fewer than the original 2B policy, while all retained tensors are bitwise identical to the source checkpoint. Cross-family execution is additionally checked with a MiniCPM-V adapter smoke test; closed-loop cross-family transfer remains an open evaluation. AcrossWAM1.0 therefore contributes an auditable software and evaluation boundary for compact latent world-action policies, distinct from LaWAM's original latent-subgoal contribution. 

---
# On the Instance Hardness as a Decision Criterion in TinyML Systems 

**Authors**: Tobiasz Puslecki, Krzysztof Walkowiak  

**Link**: [PDF](https://arxiv.org/pdf/2608.29913)  

**Abstract**: TinyML includes the implementation of machine learning on devices with limited memory and computing resources. With the development of technology, AI systems continue to scale in terms of size and computational requirements. This forces researchers to adapt methods to be environmentally sustainable by designing techniques for reducing computational costs and energy consumption in inferring AI models, even in small devices. In this work, we present preliminary findings on a novel application of the tree depth prune instance hardness method to the TinyML system. The results indicate that threshold control can change energy consumption with limited classification quality changes. This method allows us to adjust classification accuracy, thereby influencing computational complexity and energy consumption for inference. We present a work in progress with initial results as a proof of concept. 

---
# Perceive to Hypothesize, Verify to Ground: An Agentic Reasoning Framework for Open-World Geo-Localization 

**Authors**: Yutian Jiang, Ruijie Li, Sisuo Lyu, Xixuan Hao, Qingxiang Liu, Yongzi Yu, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2608.29880)  

**Abstract**: Open-world geo-localization requires models to reason over ambiguous visual cues through multi-step reasoning and external knowledge grounding. While recent large vision-language models exhibit strong multimodal reasoning capabilities, existing approaches still suffer from perceptual hallucination and context drift due to the lack of explicit evidence-grounded verification. In this work, we reformulate geo-localization as a human-like perceive-then-verify reasoning problem and propose GeoPAVE (Geo-localization Perception-and-Verification-Engine), a bi-level agentic framework that contains perception-based hypothesis generation via single-pass rollouts and verification-based evidence grounding for decision actions: support, refute, and refine. To support rigorous evaluation, we further introduce PAVED, a novel dataset derived from real-world user check-in data, equipped with comprehensive reasoning trajectories featuring multi-hop queries, multi-round tool invocations, and structured perception-verification traces. The dataset and code are available at this https URL. 

---
# FRAMEWORKERS: A Dynamic Multi-Agent Framework for AI-Generated Video Production 

**Authors**: Zhendong Li, Lei Sun, Letian Shi, Deheng Zhang, Ruibo Ming, Mengshun Hu, Dannong Xu, Jian Wang, Danda Paudel, Luc Van Gool, Jinjin Gu  

**Link**: [PDF](https://arxiv.org/pdf/2608.29814)  

**Abstract**: Modern video generators excel at synthesizing individual clips, but complete video production requires coordinating a long sequence of interdependent creative steps, including scripting, storyboarding, generation, and editing. It further demands persistent asset management and dynamic task orchestration as intermediate outputs, dependencies, and execution states evolve over time. Existing automated systems typically rely on rigid pipelines that are difficult to adapt to diverse inputs and changing workflows, while general-purpose large language models (LLMs) remain unreliable for long-horizon orchestration and multimodal asset routing. We introduce FRAMEWORKERS, a task-centric and workspace-grounded multi-agent framework for open-ended video production. A central Director formulates video creation as dynamic task management, continuously editing a Task Stack to determine which subtask to execute next and which sub-agent to invoke. An Assistant serves as the execution layer, grounding each selected task in a shared Workspace, retrieving the required assets and context, invoking the assigned sub-agent, and persisting the resulting artifacts. Execution capabilities are exposed through modular sub-agents with registered descriptors, allowing new sub-agents to be integrated without redesigning the orchestration workflow. To improve orchestration reliability, we fine-tune the Director via supervised fine-tuning (SFT) followed by Group Relative Policy Optimization (GRPO) for descriptor-conditioned task routing. Experiments show that FRAMEWORKERS outperforms strong LLM planners in routing accuracy, recovers reliably from runtime failures, generalizes to unseen sub-agents without retraining, and achieves higher end-to-end video quality and broader task coverage than fixed pipelines, single-agent systems, and prior multi-agent approaches. 

---
# PAGE-RAG: Provenance-Aware Graph Evidence Promotion for Fixed-Budget Multi-hop Retrieval-Augmented Generation 

**Authors**: Haokun Deng, Xunkai Li, Hongchao Qin, Rong-Hua Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.29753)  

**Abstract**: Multi-hop question answering in retrieval-augmented gener?ation (RAG) often benefits from retrieving beyond the few candidates that will finally be read: narrow retrieval can miss an indispensable hop, while expanded retrieval introduces topical distractors. This challenge is not tied to a particu?lar knowledge-base format. Candidate pools may come from standalone retrievers, standard RAG backends, or graph-based retrieval pipelines. What is needed is a query-aware selection layer that can use relational structure to filter candidates be?fore generation. PAGE-RAG addresses this setting by using a graph as a temporary selection structure, rather than assum?ing a graph-structured knowledge base. It builds a query-local graph over retrieved candidates, records why candidates are connected, and treats each connection as a support hypothe?sis rather than support itself. We identify the resulting failure mode as a connectivity-support gap: connected candidates do not necessarily support the answer. We propose PAGE-RAG, a Provenance-Aware Graph Evidence promotion method that scores candidate paths with relevance, source-tracing meta?data, specificity, hubness, noise, and coherence signals, and applies minimal sufficient selection to promote supporting facts into a compact reader context. PAGE-RAG can serve as a complete retrieval-to-reading pipeline, and the same promo?tion stage can be inserted after existing retrieval or RAG sys?tems without replacing their upstream retrieval logic. Across three multi-hop QA benchmarks under the same final bud?get, PAGE-RAG improves support F1 and answer F1 by 10.4 and 3.3 points on a weighted average over a strong retriever. As a plug-in, PAGE-RAG further improves all reported RAG backends, including reasoning-oriented, compression-based, graph-based, and document/chunk-level systems. 

---
# Ideation Arena: Evaluating LLM Generated Research Ideas with Battle-style Human Expert Assessment 

**Authors**: Zhiyu Chen, Keyu Zhao, Jigao Fu, Dong Liang, Yanbiao Wu, Jiaoyang Li, Haidong Xue, Xinhua Zeng, Yuanyi Zhen, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.29696)  

**Abstract**: Evaluating research ideas generated by LLMs is difficult because their scientific value cannot be fully determined by objective criteria, and no single reference answer specifies what counts as a good idea. To address this challenge, we introduce Ideation Arena, a battle style platform that evaluates research ideas through pairwise human assessment. Ideation Arena evaluates ideas generated by 14 frontier LLMs and 5 research agent architectures built on 2 base models. To ensure a common starting point, Ideation Arena builds shared literature contexts from papers familiar to the participating researchers and provides the same contexts to all LLMs and agents. We collect over 6,000 double blind pairwise comparisons from 105 active computer science researchers and construct an Elo rating leaderboard of proposal-stage expert preferences in computer science under a shared closed-context protocol. We validate the rankings through interrater agreement and robustness analyses, showing that the leaderboard remains stable under changes in annotator composition and domain coverage. Our results show substantial variation in agent effectiveness, with some frameworks improving ideation quality over their backbones and others offering little benefit or even underperforming their base models. We further construct Ideation Arena Eval, a benchmark for assessing whether automated evaluators align with human preferences in research ideation. Experiments with current LLM judges show that they still cannot reliably reproduce expert preferences, with the best judge reaching 72.56% Soft Accuracy on Overall Quality. Our code, data, and leaderboards are available at this https URL. 

---
# Detect Before You Attribute: Cascade Failure Attribution for Multi-Agent Systems 

**Authors**: Jiayi Zhang, Zexin Wang, Degang Sun, Changhua Pei, Fei Sun, Gaogang Xie, Jingjing Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.29646)  

**Abstract**: Large language model (LLM)-based agents have shown strong potential in solving complex tasks through multi-step reasoning, yet they remain vulnerable to execution failures. Accurate failure attribution is therefore critical for improving agent reliability. Existing topology- and spectrum-based methods exploit trajectory structures but often overlook fine-grained semantics, while LLM-based attribution methods capture semantic cues but suffer from long-context degradation over lengthy trajectories. To address these challenges, we propose DUOTRACE, a plug-and-play detection filter for LLM-based failure attribution. DUOTRACE follows a detect-before-attribute paradigm: it first detects anomalous executions and then supplies focused trajectory evidence to downstream LLM-based attribution methods. For effective VAE-based anomaly detection on agent trajectories, DUOTRACE integrates dual-view semantic-structural node representations, a Tree-LSTM-based trajectory encoder, and prefix-chain- and LLM-based data augmentation to handle heterogeneous nodes, hierarchical execution structures, and limited failure data. Experiments with six LLM-based attribution baselines show that DUOTRACE improves agent-level and step-level attribution accuracy by 8.7% and 7.0%, respectively. 

---
# LLMs Interpret, Embeddings Organize, Graphs Emerge: Agent-Driven Compilation of Scientific Knowledge 

**Authors**: Shi-Ju Ran, Kun Zhang, Xi Wu, Liu-Si Yang, Wen-Jun Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.29612)  

**Abstract**: Sustained scientific work requires a knowledge substrate that carries interpretation across tasks and preserves paths to source evidence. We call this process \emph{scientific knowledge compilation} and implement it in ASKS, the \emph{Agent-Driven Scientific Knowledge System}. For each source, an LLM produces a readable Wiki view and machine-facing semantics. Deterministic checks convert the latter into a document-local GraphDelta, and embedding geometry together with explicit graph rules integrates the proposed changes into persistent state. Each ingest is an inspectable state transition over accumulated knowledge, with compiled Wiki and graph views linked to the preserved source record. We examine this process by chronologically compiling 56 published papers from one research program. Branch survival, cross-paper support, lineage, coverage, and churn yield a source-traceable author research portrait centered on tensor-network methods, with branches into quantum many-body research, tensor-network machine learning, and quantum-AI-oriented directions. In this run, higher-level Hub organization remains stable and low-churn. Canonical-node growth is predominantly additive. Graph-level measurements and navigation paths retain links to the source records from which they were compiled. 

---
# Towards a Systems Foundation for Agentic Skills: Architecture, Lifecycle, and Security 

**Authors**: Sanket Badhe, Deep Shah, Priyanka Tiwari, Nehal Kathrotia  

**Link**: [PDF](https://arxiv.org/pdf/2608.29596)  

**Abstract**: Autonomous large language model (LLM) agents increasingly face reliability, context consumption, and execution stability bottlenecks when deployed on complex, long-horizon tasks. While monolithic prompt engineering and stateless tool-calling paradigms struggle to scale, the field is rapidly converging toward \emph{agentic skills}: modular procedural abstractions that externalize execution knowledge into reusable, executable, and portable artifacts. This paper establishes a unified systems foundation and reference architecture for the agentic skills ecosystem. We formalize skills as externalized procedural knowledge bridging high-level cognitive planning with deterministic execution environments, and systematically delineate the architecture across a nine-stage lifecycle: autonomous discovery, authoring and representation formats, memory storage, dynamic retrieval and routing, composition and orchestration, execution and repair, lifelong adaptation, empirical evaluation, and security governance. We further examine marketplace dynamics, public registries, and emerging adversarial threat vectors, alongside runtime verification and defense mechanisms. Finally, we categorize system implementations across software engineering, operating system navigation, embodied robotics, and scientific discovery, while highlighting critical open challenges in continual learning and benchmark realism. This work establishes agentic skills as a foundational paradigm for building scalable, robust, and verifiable autonomous language agents. 

---
# Not Safe for All: Auditing the Dialect Penalty in Text-to-Image Safety Pipelines 

**Authors**: Minkyu Kim, Juhwan Choi, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.29589)  

**Abstract**: Text-to-image (T2I) safety guardrails fail to generalize equitably to non-standard dialects. Evaluating 23,080 paired prompts across five English dialects, we formalize this failure as the dialect penalty, where filters trigger based on linguistic surface features rather than semantic intent. Text-level filters fail in opposing directions: NSFW-T over-flags benign dialect prompts and LatentGuard over-flags toxic ones (bias gaps up to +28.29 pp), while the OpenAI Moderation API under-detects them. A controlled typo ablation confirms this penalty originates from flagging dialectal features, not generic out-of-distribution sensitivity. The pixel-level generator is largely dialect-agnostic; the penalty enters at text processing and cascades unevenly to post-hoc guardrails. We show this bias tracks training data imbalance and is mitigable via group-balanced retraining, with an ablation attributing the gain to balanced exposure rather than to the worst-group objective of GroupDRO (group distributionally robust optimization). Current pipelines systematically fail dialect speakers, an equity failure masked by mean accuracy benchmarks. Our official code and dataset are publicly available at this https URL.
Content Warning: This paper contains offensive, toxic, or disturbing text prompts and generated images. 

---
# Call Neighbours Yourself: Graph Walks with Destination-Conditioned On-Policy Self-Distillation 

**Authors**: Yilun Liu, Boyu Luo, Yanran Tang, Ruihong Qiu, Zi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.29588)  

**Abstract**: Reasoning over text-attributed graphs (TAGs) requires large language models (LLMs) to combine a node's text with evidence distributed across its neighbourhood. Existing methods fix the set of accessible neighbours before generation, forcing reasoning to operate over a static context and preventing the model from acquiring missing evidence during inference. We argue that neighbour selection should itself be part of the reasoning process. To this end, we propose Call Neighbours Yourself (CNY), a framework that enables LLMs to proactively explore graph neighbourhoods through topology-constrained graph-walk actions. Instead of reasoning over a pre-selected neighbour set, CNY exposes lightweight neighbour previews and learns when to expand candidate neighbours for additional evidence. To address the delayed-credit challenge of neighbour exploration, we introduce destination-conditioned on-policy self-distillation, which retrospectively evaluates a selected neighbour after its content is revealed and converts the resulting change in action preference into an action-level training signal. Experiments on standard TAG reasoning benchmarks under a unified raw-text setting show that CNY consistently outperforms fixed-context post-training baselines. Furthermore, the learned exploration policy transfers to unseen graphs and to a graph-level task not encountered during training. Code is available at this https URL. 

---
# Can escalation channels redirect reward hacking toward defect disclosure? 

**Authors**: Francesca Gomez  

**Link**: [PDF](https://arxiv.org/pdf/2608.29460)  

**Abstract**: When coding agents encounter defective test infrastructure they may reward-hack: hardcoding outputs or editing test files to pass tests they cannot legitimately satisfy, a pattern that has now appeared outside benchmarks, in a coordinated multi-agent intrusion of a major AI platform's production infrastructure. The same capability that lets an agent detect and exploit a defect could let it report one, given the right decision environment. We evaluate escalation channels, structured reporting tools available to the agent at the point of conflict, as a decision-environment intervention that both reduces reward hacking and surfaces the infrastructure defects that trigger it. A $2 \times 2$ factorial separates the contributions of an escalation tool, a standalone anti-reward-hacking policy, and their combination. Across 8 frontier models spanning 5 families, the combined intervention reduces reward hacking from 23.6\% to 5.3\% (mixed-effects logistic OR = 9.2, 95\% CI 5.0--16.8, $p < 10^{-12}$) with no detectable cost or performance overhead, eliminating it entirely for 6 of 8 models. Escalation and hacking are near-perfectly mutually exclusive, with 96.8\% of escalations involving no hacking. Beyond reduction, escalation channels function as diagnostic infrastructure: on top of monitoring, escalation adds +10.1 percentage points of defect detection coverage and is more accurate once it fires (99.4\% vs 85.8\%). Unlike containment-based approaches that risk outpacing growing model capabilities, escalation channels redirect capability toward disclosure rather than exploitation. 

---
# Toward Latent Language Model Skills Steering and Optimization: An Empirical Study 

**Authors**: Xunyi Jiang, Junda Wu, Yuxin Xiong, Sheldon Yu, Tong Yu, David Arbour, Ritwik Sinha, Julian McAuley, Hongyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2608.29459)  

**Abstract**: Skills, as a useful abstraction for the procedural capabilities of large language models (LLMs), capture how models perform structured, multi-step reasoning and program execution. Existing approaches typically treat skills as explicit, surface-level constructs specified through prompts or programs, leaving open the question of how such procedural capabilities are represented inside the model and whether they can be manipulated as structured objects in latent space. In this empirical study, we investigate whether procedural LLM skills can be represented as directions in activation space and whether vector-space operations over these directions can express skill-level behaviors. We find that procedural skills admit a vector-space representation: individual skill directions can be activated to shift model behavior; independently extracted directions can compose to form higher-level skills. Contrastive directions yield context-conditioned algorithmic personalization and optimization trajectories over skill directions evolve non-monotonically, with intermediate states often surpassing fully optimized solutions. These results support a representation-level view of procedural LLM skills: they admit a latent vector-space organization that allows direct manipulation through internal interventions. 

---
# EvoGenUI-Bench: Evaluating LLMs as Multi-Turn Generative UI Assistants 

**Authors**: Yue Peng, Lanke Xia, Zihan Wang, Jiahao Ye, Ke Ning, Hongyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2608.29387)  

**Abstract**: Large language models can generate interactive web interfaces, but reliable generative UI requires maintaining an executable artifact as user requests evolve. We introduce EvoGenUI-Bench, a benchmark for multi-turn interface maintenance comprising 150 five-turn tasks and 750 turns across three scenarios: information presentation, executable interaction, and tool-grounded external state. We execute generated artifacts in a browser and evaluate them using screenshots, source and DOM evidence, actor traces, and runtime logs. Beyond turn-level and episode-level success, we measure cross-turn retention with Adjacent Pass Retention. Across eight models, even the strongest achieves 74.9% Turn Pass while completing only 37.3% of five-turn episodes; APR further falls to 52.4% on tool-grounded tasks. Diagnostic analysis shows that presentation failures center on information architecture, interaction failures on derived-state propagation and affordance binding, and tool-grounded failures additionally involve external-state grounding and requirement decomposition. These results reframe generative UI evaluation from judging isolated outputs to testing whether interface behavior, derived state, external state, and assistant claims remain synchronized as the artifact evolves. 

---
# Evaluating Tiny Recursive Models Across Training for Code Generation 

**Authors**: Anjani Sirivella, Aanisha Newaz, Glaucia Melo  

**Link**: [PDF](https://arxiv.org/pdf/2608.29376)  

**Abstract**: Code generation increasingly relies on large transformer models, whose capability advances with scale. Yet such a scale is costly, creating demand for small models, especially where data is limited. Recursive models address this by reusing a single block to add depth rather than stacking independent layers. Such models are typically evaluated by teacher-forced fit (next-token loss on ground-truth prefixes) or task accuracy, at a single checkpoint, whereas code is produced by free-running generation, where the model extends its own output. Whether a teacher-forced advantage survives free-running generation, and whether it holds across training, remains open. To study both, we compare a ~28M-parameter autoregressive Tiny Recursive Model (TRM-AR) on natural-language-to-Python code generation against parameter-matched and depth-matched controls, tracking fit and generation across 40 epochs and three seeds. The fit ranking between the recursive model and the depth-matched control reverses twice. Selecting each checkpoint by validation loss and examining the trajectory yields a consistent comparison. At equal parameters, TRM-AR fits, generates, and generalizes better than the parameter-matched control while recovering approximately 45% of the validation-loss gap and 57% of the generation-quality gap between the two controls, at roughly 175 times the per-step cost of the parameter-matched control. However, at equal effective depth, the larger transformer fits and generates better at its validation optimum, suggesting TRM-AR's advantage lies in resistance to overfitting, not greater capability. These findings suggest that recursive code generation models should be evaluated jointly on fit and generation across the training trajectory rather than at a single checkpoint. 

---
# FORESIGHT-9: Prospective and Process-Aware Evaluation of Adaptive Trading Agents 

**Authors**: Xiangxin Luo, Chengtian Hong, Haohua Li, Yongyi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2608.29372)  

**Abstract**: Retrospective backtests provide a limited test of adaptive trading agents: they cannot rule out historical contamination, expose sensitivity to a single realized market path, or reveal internal degeneration during long-horizon adaptation. We introduce FORESIGHT-9, a prospective and process-aware benchmark built from nine auditable counterfactual stress worldlines branching from a common July 2026 information boundary. Each worldline specifies staged macro-financial events and joint multi-asset anchors; a deterministic generator realizes the trajectories, while observations are disclosed according to in-world time. A common contract standardizes observations and execution while preserving each agent's native adaptation loop. We evaluate two adaptive trading-agent frameworks with two foundation-model backbones across 36 long-horizon runs. Agent rankings vary substantially across worldlines and backbones, and a fixed equal-weight policy outperforms 31 of 36 runs. Process telemetry exposes failures that terminal returns conceal: in one high-return run, the live factor library collapsed while executed holdings converged to the equal-weight fallback, even though decision records continued to report an active factor ensemble. FORESIGHT-9 therefore evaluates not only portfolio outcomes, but whether adaptive agent state and execution remain coherent across alternative futures. We release the worldlines, trajectories, audit traces, and regeneration scripts. 

---
# Reviving our data foundations is the most disruptive step to data maturity 

**Authors**: Valentina Carapella, Ernesto Jimenez-Ruiz  

**Link**: [PDF](https://arxiv.org/pdf/2608.29368)  

**Abstract**: The most disruptive step that enterprises of small-medium size and maturity can take to make the most of the latest technological advances in AI is to step back from the hype and focus on establishing or reviving a good knowledge foundation layer. It is a hard message to present to the executive team; therefore, it needs to be backed by evidence, and its implementation needs to be of minimal impact on the existing processes. In this vision statement, we discuss how we need to rethink what evidence speaks to the decision-makers and propose a low-impact data strategy that adapts to the existing and ever-changing data flows and processes across the company. We firmly believe that knowledge graph techniques will increasingly become non-negotiable in the data strategy of an AI-powered enterprise, provided that we approach their design in a modular, dynamic and cross-functional way. 

---
# TRACER: Per-Tool Context Retention for LLM Agents via Consequence-Attributed Reinforcement Learning 

**Authors**: Ziqi Lin, Ye Wu, Mengying Yang, Xu Liu, Yizhou Liu, Qiang Ke, Qin Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.29363)  

**Abstract**: Enterprise data agents answer business queries by chaining many tool calls over multiple reasoning steps, routinely accumulating hundreds of thousands of context tokens per session. Existing compression strategies typically allocate retention budgets without accounting for the downstream consequences of removing individual tool outputs. Aggressive compression may therefore trigger costly tool re-invocations that offset the initial savings. We call this the compression--consequence gap. To close it, we propose TRACER, which formulates compression as a sequential per-tool decision problem. A lightweight REINFORCE policy assigns query-conditioned retention ratios using only information available at each compression event. Its consequence-aware objective jointly accounts for task success, total token consumption, and post-compression tool re-invocations. To improve credit assignment, TRACER uses a learned outcome model to compare the predicted consequences of the selected retention ratio with those of fully retaining each tool output. On held-out production queries across three compressor backends, TRACER reduces total token consumption by 29--46% relative to keeping all context while maintaining comparable or higher task success. Compared with a tool-type-conditional static policy, TRACER provides an additional 15--18% of token savings. Interventional rollouts show that the learned per-tool credit scores correlate with measured single-tool consequences. The learned policy also yields positive savings when transferred across agent backbones and compressor architectures, and reduces token consumption by 18--25% on five held-out LOCA-bench environments. These results demonstrate the value of consequence-aware, per-tool context retention for improving the efficiency of long-horizon language agents. 

---
# LiteSearch-VL: Small Multimodal Search Agents via Trajectory Distillation and Synthetic Step-DPO 

**Authors**: Saeed Khaki, Nima Safaei, Kamal Ginotra  

**Link**: [PDF](https://arxiv.org/pdf/2608.29357)  

**Abstract**: Multimodal search agents answer visual questions by interleaving image understanding, web retrieval, tool use, and evidence synthesis. Strong systems exist, but in two expensive regimes: proprietary frontier models such as GPT-5 and Gemini, or large open vision-language backbones trained with substantial agentic data and reinforcement learning. We ask a different question: when released agent trajectories are distilled into much smaller backbones under a single-node budget, what is actually transferred? We study this with LiteSearch-VL, a low-compute recipe for Qwen3-VL-2B and Qwen3-VL-4B that uses only released OpenSearch-VL trajectories, parameter-efficient LoRA adapters, and synthetic step-level preferences: DPO on GPT-5-generated hard negatives targeting five local failure modes (premature answer, wrong tool, weak query, repeated query, ignored image). Across 12,400 GPT-5-judged rollouts on SimpleVQA, FVQA, LiveVQA, and VDR-Bench-testmini, the dominant effect is behavioral rather than a uniform accuracy lift: full-trajectory supervised fine-tuning transfers the agent contract, taking the 2B model from almost never emitting a usable answer (1,237/1,240 no_answer rollouts) to 28.4% macro Pass@1, matching or slightly exceeding the off-the-shelf 4B base (25.6%). Synthetic preference learning and compact tool distillation act as refinements rather than phase transitions (best 4B configuration: 30.8% macro Pass@1). Finally, a controlled VDR step-budget ablation shows that extra search turns convert abstentions into wrong_entity errors rather than correct answers, identifying answer verification, not search depth, as the next bottleneck for small multimodal agents. 

---
# Plant-Inspired AI: Plants as Inspiration for Novel Problem Formulations, and Two Case Studies 

**Authors**: Deepayan Sanyal, Joel Michelson, Carla E. Cao, Adam B. Roddy, Maithilee Kunda  

**Link**: [PDF](https://arxiv.org/pdf/2608.29356)  

**Abstract**: Artificial Intelligence (AI) has long been inspired by studies of biological intelligence. Reinforcement learning, for instance, drew inspiration from studies involving animal learning and is now a powerful paradigm for solving many real-world problems. Recently, plant biologists have uncovered a wide range of complex behaviors in plants that enable them to flexibly adapt to variable environments. Here, we argue that such behavior can motivate new AI frameworks encompassing a range of problems overlooked by existing problem-solving frameworks such as supervised learning, tree search, and constraint satisfaction. We illustrate this idea with two examples of intelligent problem-solving in plants: (1) leaf mimicry in Boquila trifoliolata, a vine capable of altering its leaves' morphology to resemble those of multiple host trees simultaneously; and (2) coordinated root-shoot growth, wherein plants allocate resources across organ systems exploring distinct environments. While leaf mimicry is highly specific to Boquila, coordination of root-shoot growth is shared across most plants. For both examples, we capture underlying computational principles and identify problems fitting these frameworks that are currently unaddressed by AI. Finally, we outline preliminary task formulations and discuss how these formulations may be applied to non-plant problems. 

---
# APPSolver: Adaptive Patch Partitioning for Point-Wise Ship Flow Prediction on Unstructured Meshes 

**Authors**: Wenhua Huo, Fenglei Han, Wangyuan Zhao, Xiao Peng, Chunhui Wang, Jialin Wu, Jiayi Han  

**Link**: [PDF](https://arxiv.org/pdf/2608.29355)  

**Abstract**: Large non-uniform point sets make direct attention-based surrogate modeling costly for ship hydrodynamics. We introduce APPSolver, a point-wise flow-prediction framework built around Adaptive Patch Partitioning (APP), a deterministic quadtree representation for fixed two-dimensional horizontal slices extracted from ship CFD simulations. APP assigns finer patches near the hull and coarser patches farther away, downsamples patch contents, and recovers predictions to the full reference point set. Under a corrected protocol that constructs natural $(t,t+1)$ pairs before splitting, reuses training-set normalization statistics, and reports three model seeds, learned tokenizers are more accurate than APP-Transformer, and a persistence baseline has lower one-step MAE on all three ShipBench hulls. The supported benefit of APP is therefore computational rather than universal predictive superiority: on a representative DTC input, APP-Transformer requires 1.815 GFLOPs and 1.309 ms per model forward, while a matched ablation shows that adaptive partitioning reduces MAE by 16.4-24.9\% relative to a uniform partition augmented with learned slicing. Condition encoders provide setting-dependent gains in leave-one-hull-out evaluation, but the current absolute next-state objective does not establish accurate long-horizon dynamics. These results characterize APP as a compact spatial representation with an explicit accuracy--efficiency trade-off. Code is available at this https URL . 

---
# Cross-Relational Preference Learning for Better LLM Instruction Following 

**Authors**: Runsheng Li, Kai Sun, Bin Shi, Bo Dong  

**Link**: [PDF](https://arxiv.org/pdf/2608.29352)  

**Abstract**: Large Language Models (LLMs) still exhibit limited capability in following complex instructions. While existing approaches often rely on preference learning to enhance this ability, they typically overlook the relationships between the permissible response spaces of different instructions, which restricts a model to align with subtle and diverse constraint variations. To address this, we propose Cross-Relational Preference Learning (CRPL), a novel framework for constructing preference data that explicitly models inter-instruction relationships through two key techniques: Cross-Relationship Perturbation and Cross-Region Pair Sampling. This enables the generation of more diverse preference data that captures a wide spectrum of constraint variations. Additionally, we introduce an atomic constraint-based verification mechanism to rigorously assess response satisfaction, ensuring high-quality preference pair construction. Extensive experiments across multiple preference learning methods (e.g., DPO, KTO), LLM backbones and four instruction-following benchmarks demonstrate that our approach achieves substantial improvements over prior baselines and exhibits strong generalization. 

---
# Extending TotalSegmentator: Predicting Patient and Acquisition Characteristics from CT and MR Images 

**Authors**: Jakob Wasserthal, Joshy Cyriac, Michael Bach, Kimia Mozahheb Yousefi, Minh-Son To, Máté Sik, Cédric Hémon, Thomas Weikert, Martin Segeroth  

**Link**: [PDF](https://arxiv.org/pdf/2608.29348)  

**Abstract**: Background: Patient details and acquisition metadata are important for clinical decisions, image quality control, and automated research pipelines, but may be missing or unreliable in imaging archives.
Purpose: To develop and evaluate a fast open-source model that predicts patient and acquisition characteristics directly from CT and MR images.
Materials and Methods: Separate 3D ResNet-10 ensembles for CT and MR were trained on 57,291 and 43,200 clinical examinations acquired from 2011 to 2025. Both predicted weight, height, age, sex, contrast presence, vertebral coverage, and image noise. The CT model additionally predicted scanner manufacturer, tube voltage, tube current, convolution kernel, and post-injection time; the MR model predicted sequence class. Performance was evaluated on internal CT (n=501) and MR (n=636) test sets and an external CT dataset (n=54).
Results: Internal CT MAEs were 3.90 kg, 3.68 cm, and 4.42 years for weight, height, and age, with sex F1=0.990; corresponding MR results were 4.34 kg, 4.62 cm, 7.13 years, and F1=0.970. The CNN outperformed a segmentation-derived XGBoost baseline for all four core targets in both modalities (adjusted P<=.042). F1 scores were 0.963 for CT contrast, 0.953 for MR sequence, and 0.823 for MR contrast. External CT MAEs were 4.45 kg, 4.05 cm, and 5.17 years, with sex F1=0.971. CPU inference required 20 seconds for CT and 12 seconds for MR.
Conclusion: One 3D multitask model per modality can rapidly recover patient and acquisition characteristics from heterogeneous CT and MR examinations. Models are available in TotalSegmentator: this https URL 

---
# BIRD-History: A Benchmark for History-Driven Text-to-SQL with Fine-Grained Knowledge Annotations 

**Authors**: Yunfan Zhou, Qiming Shi, Yizhou Yang, Di Weng, Yingcai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.29345)  

**Abstract**: While recent Large Language Model (LLM)-based text-to-SQL systems achieve impressive performance on standard benchmarks, they struggle when user queries implicitly rely on domain-specific knowledge, such as business logic, data conventions, and analytical practices, that is neither captured by the schema nor explicitly stated in the natural language question. Historical SQL query logs offer a valuable source of such knowledge, yet existing benchmarks do not adequately support evaluation of history-driven approaches. To address this gap, we introduce BIRD-History, a benchmark consisting of 1,393 tasks across 11 databases, designed to evaluate text-to-SQL systems' ability to ground underspecified natural language questions using historical SQL scripts. Each task is annotated with ground-truth labels specifying which historical queries contain relevant knowledge and which SQL clauses encode it, enabling systematic evaluation of both retrieval effectiveness and knowledge utilization. Alongside the benchmark, we propose a plug-in retriever that extracts five types of external knowledge from historical SQL scripts, then retrieves and reranks relevant fragments for query generation. The retriever integrates seamlessly into existing few-shot text-to-SQL pipelines without requiring prompt modifications. Experiments demonstrate consistent improvements across four text-to-SQL systems, highlighting the value of leveraging historical query logs for handling underspecified queries. Dataset and code are open-sourced on this https URL. 

---
# Formal Concept Analysis with Three Types of Negation 

**Authors**: Zhenghua Pan  

**Link**: [PDF](https://arxiv.org/pdf/2608.29311)  

**Abstract**: Classic Formal Concept Analysis (FCA) primarily focuses on the positive relationships between objects and attributes and does not have mechanisms for handling this http URL overcome this limitation, we introduce three types of negation concepts (contradictory negation, opposite negation, intermediary negation) into this http URL on the set SCOI and logic LCOI+PLCOI with these three types negation, we define formal context, Galois connection operators, formal concept and concept lattice with three types of negation,this leads to the proposal of a FCACOI: Formal Concept Analysis with contradictory negation, opposite negation and intermediary this http URL the reasoning in FCACOI, this paper focuses on attribute implication reasoning. Based on the logic LCOI+PLCOI and its semantics, we introduce the notion of ICOI-entailment as the semantic implication for attribute implication reasoning in FCACOI. Through ICOI-entailment, a connection is established between attribute implication reasoning in FCACOI and inference in the logic LCOI+PLCOI, it indicate that formally proven inference rules (theorems) in LCOI+PLCOI are valid in the attribute implication reasoning of FCACOI, LCOI+PLCOI provides a logical foundation for attribute implication reasoning in FCACOI. To illustrate the capability of attribute implication reasoning in FCACOI, we discuss its application in a concrete example. Moreover, we explore attribute reduction of the formal context in FCACOI, propose two research frameworks for attribute reduction from different perspectives, and compare their this http URL believe that, based on richer logic and semantics, FCACOI elevates FCA from a theory that describes affirmations to one that can describe affirmations and its contradiction(either this or that), opposition(extreme negation) and intermediary (transitional states between oppositions). 

---
# Predicting Future Organ Dysfunction in ICU Patients Using Temporal Convolutional Networks on MIMIC-IV Data 

**Authors**: Razan Albouq, Asra Aslam  

**Link**: [PDF](https://arxiv.org/pdf/2608.29301)  

**Abstract**: Predicting future organ dysfunction in Intensive Care Unit (ICU) patients is critical for early clinical intervention, yet existing machine learning approaches have largely treated the Sequential Organ Failure Assessment (SOFA) score as an input to binary mortality prediction rather than as a continuous clinical outcome in its own right. We investigate the extent to which a Temporal Convolutional Net work (TCN) can predict next-day SOFA scores from multivariate ICU time-series data extracted from MIMIC-IV, characterise the relative contribution of each organ system to total SOFA variance and deterioration, and identify distinct trajectory patterns across ICU stays. A residual TCN trained on three-day sliding windows achieved a five-fold cross-validation R2 of 0.740 +- 0.013 and MAE of 1.431 +- 0.022, outperforming a naive persistence baseline on RMSE and R2. SHAP interpretability analysis revealed that the model functions primarily as a severity-anchoring mechanism rather than a true sequence model, with predictions dominated almost entirely by the most recent observation day. Cardiovascular dysfunction emerged as the strongest discriminator of both cross-sectional severity and acute deterioration, and unsupervised trajectory clustering identified two clinically meaningful phenotypes, an improving group (58.9%) and a persistently severe group (41.1%), differentiated by cardiovascular, hepatic, coagulation, and renal involvement. We conclude that TCNs can extract meaningful predictive signal from ICU physiological data, but that short input windows and complete-case selection bias currently limit their clinical utility, motivating future work on longer input horizons, alternative missing-data strategies, and external validation. 

---
# Accelerating Unified Multimodal Models with Core-Expansion Routing and Unified Computation Scheduling 

**Authors**: Wengyi Zhan, Chenqian Yan, Songwei Liu, Mingbao Lin, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2608.29291)  

**Abstract**: Unified multimodal models jointly support understanding and generation, but incur substantial redundant computation across tokens, layers, and generation timesteps. Through token-importance probing, we identify an asymmetric core-expansion structure: understanding exhibits a stable importance component, while generation largely shares this component but requires progress-dependent corrections. We therefore propose CE-Router, which uses a task-shared core scorer and progress-conditioned generation expansions, optimized through generation decomposition and cross-task core alignment. At inference, CE-Router compacts token computation and supplies a learned routing signal to Unified Computation Scheduling, which coordinates layer skipping, FFN pruning, diffusion-head cache reuse, and denoising-step early exit. Experiments on two representative UMM architectures demonstrate consistent quality--efficiency improvements across both tasks, retaining 98.03\% of dense understanding performance with a 1.93$\times$ end-to-end inference speedup. 

---
# MMPCBench: Benchmarking Multimodal Large Language Models on Proactive Critique of Flawed Inputs 

**Authors**: Jinzhe Li, Gengxu Li, Jinnan Li, Yuan Wu, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2608.29286)  

**Abstract**: As Multimodal Large Language Models (MLLMs) evolve into sophisticated interactive assistants, their reliability depends not only on following instructions but also on validating them. We define Proactive Critique as the model's autonomous ability to identify, analyze and fix faulty user inputs without extra prompts. However, evaluations mainly test models under ideal circumstances or simple refusal behaviors, largely ignoring active error processing. To fill this gap, we propose MMPCBench, a comprehensive framework for evaluating MLLMs' proactive critique competence. It features a fine-grained taxonomy of 4 primary error types spanning 12 subcategories, ranging from cross-modal contradictions to missing visual premises. We adopt a hierarchical evaluation protocol to measure models' error detection, diagnosis and resolution performance, and apply alignment-aware metrics to assess the coherence between internal reasoning and final responses. Tests on 14 mainstream MLLMs show obvious weaknesses in proactive critique, especially in dealing with subtle visual anomalies. Notably, we identify a pervasive "consistency gap": reasoning models can often correctly identify and analyze errors during internal reasoning yet suppress these valid insights in final outputs to prioritize response compliance. The code and data is available at this https URL. 

---
# Understanding Deep Learning via Entropy Space Theory 

**Authors**: Li Li, Tong Zhang, Wentao Yu, Zuobin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.29279)  

**Abstract**: Deep learning is often criticized for its theoretical research lagging behind practice. To make deep learning easier to understand, the entropy space theory is first introduced here. The entropy space can cover all the possibilities of any deep learning model by topological structure. It is independent of network parameters. Through the designed fundamental operations and norm, entropy space is proven to be a normed space within the formal axiomatic framework. Based on the theory, a unified coordinate system is proposed. It can coordinatize every state of a model and rank them by compression of the maximal value of information entropy. The theory offers a novel priori framework for mathematical fundamentals of deep learning. 

---
# EpaCache: Error-Propagation-Aware Caching for Accelerating Diffusion-Based Visual Generation 

**Authors**: Yuhan Liu, Zongwei Hong, Jinglun Li, Linze Li, Shen Zhang, Yao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.29264)  

**Abstract**: Diffusion-based visual generative models deliver strong image and video synthesis quality but incur high inference costs because sequential samplers repeatedly evaluate large networks. Caching-based methods reduce inference latency by reusing intermediate computations across adjacent timesteps. However, existing cache controllers rely primarily on local temporal variation and overlook the trajectory-level consequences of cache reuse. We introduce Error-Propagation-Aware Cache (EpaCache), a training-free caching policy that adaptively allocates the reuse budget on timesteps with lower downstream impact. Experiments on image and video synthesis models demonstrate that EpaCache consistently improves the latency--fidelity trade-off over existing caching methods. On FLUX.1-dev, EpaCache outperforms the prior state-of-the-art caching method in both latency and fidelity, reducing inference time from $11.7$ s to $11.3$ s while improving PSNR from $21.4$ to $22.8$. On HunyuanVideo, EpaCache achieves a $2.63\times$ speedup over uncached inference and improves SSIM from $0.891$ to $0.905$ over the prior state-of-the-art method at matched latency. 

---
# RACER: Reinforced Agent Collaboration for Explainable Reasoning on Knowledge Graphs 

**Authors**: Yuwei Lou, Hao Hu, Yuzhou Jiang, Zongfei Zhang, Liang Wang, Jincai Liu, Jidong Ge, Xianping Tao  

**Link**: [PDF](https://arxiv.org/pdf/2608.29263)  

**Abstract**: Large Language Models (LLMs) often suffer from hallucination and struggle with complex reasoning tasks requiring multi-hop domain knowledge. While integrating Knowledge Graphs (KGs) provides a structured and verifiable information source, current KG-enhanced LLM paradigms usually rely on single-agent path extraction and fixed prompting, lacking adaptability and facing huge search spaces. To address these challenges, we propose RACER, a Reinforced Agent Collaboration framework for Explainable Reasoning on knowledge graphs. RACER employs a semantic-aware action pruning and teacher-guided reinforcement learning mechanism to efficiently extract high-quality reasoning pathways from large-scale KGs. Furthermore, to mitigate single-path generation pitfalls, we introduce a cross-task accumulated shared memory graph paired with an attention-driven multi-path knowledge refinement module. Finally, RACER orchestrates these components through a four-role multi-agent collaboration system (GraphAgent, TemplateAgent, AnswerAgent, and CriticAgent) to dynamically refine prompts and evaluate answers. Extensive experiments on CommonsenseQA and OpenBookQA datasets demonstrate that RACER significantly outperforms state-of-the-art KG-enhanced LLM baselines with an average improvement of 5\%, offering robust and highly interpretable reasoning capabilities. 

---
# Dynamic Important Example Mining for Reinforcement Finetuning 

**Authors**: Haoru Tan, Sitong Wu, Yanfeng Chen, Shizhen Zhao, Yang-Tian Sun, Tianjia Liu, Chirui Chang, Shaofeng Zhang, Samm Sun, Xiuzhe Wu, Ruobing Xie, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2608.29252)  

**Abstract**: Reinforcement fine-tuning (RFT) is increasingly used to strengthen the reasoning abilities of large models, yet its effectiveness is bound by how training data are selected and used. Most data-centric RFT methods rely on static or heuristic sample selection, implicitly assuming a sample's value is fixed over training. This overlooks the non-stationary dynamics of policy learning and can lead to suboptimal updates. We propose Dynamic Important Example Mining (DIEM), a principled and fully automated framework that makes data utilization adaptive throughout RFT. DIEM integrates two components into each optimization step: (i) a gradient-alignment importance estimator that efficiently approximates each sample's marginal contribution to policy improvement; and (ii) a constrained batch reweighting scheme that maximizes aggregate utility while preserving the update's gradient magnitude to stabilize optimization. Across several reasoning benchmarks, DIEM consistently outperforms strong static and dynamic baselines. The code will be released via this https URL. 

---
# GuardianAgent: Policy-Conditioned Risk-Adaptive Anonymization with Verified Adversarial Escalation 

**Authors**: Ruiyi Yang, Gayathri Lihinikaduarachchi, Rahat Masood, Flora D. Salim, Salil S. Kanhere  

**Link**: [PDF](https://arxiv.org/pdf/2608.29251)  

**Abstract**: Privacy protection for live web traffic requires more than detecting private spans. Agent-based privacy protection systems must determine whether an outgoing action complies with the destination site's privacy policy, then apply only the level of rewriting or sanitisation justified by the residual disclosure risk. We present GuardianAgent, a policy-conditioned anonymization framework that couples structured risk assessment with verified adaptive rewriting. GuardianAgent computes risk through AMRSF (Adaptive Multi-factor Risk Scoring Formula), an explicit controller that combines policy-violation likelihood with data sensitivity, recipient transmission, purpose legitimacy, contextual basis, and policy transparency, rather than relying on an LLM to assign risk directly. This risk score determines both the allow/transform/deny decision and the initial anonymization level. For efficiency, GuardianAgent uses an evidential fast path for low-uncertainty policy matches and invokes an LLM slow path only for uncertain cases. For rewriting, it applies a five-level hierarchy driven by a verified adversarial guesser: guesses trigger escalation only when supported by the original text, preventing hallucinated attacker confidence from causing unnecessary over-anonymization. Experiments across three benchmarks spanning legal text (TAB), Reddit posts (SynthPAI), and multi-format synthetic PII records (PII-Masking-300k) show that GuardianAgent achieves the strongest privacy-utility trade-off among published baselines and is the only method to reach more than 0.90 privacy in all three domains, remaining robust under a backbone switch. Action-context stress tests further show that the same outgoing text receives different decisions and anonymization strengths under different recipients, purposes, action bases, and policy-transparency conditions. 

---
# Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge 

**Authors**: Saransh Kumar Gupta, Armaan Shah, Lipika Dey, Partha Pratim Das, Ramesh Jain  

**Link**: [PDF](https://arxiv.org/pdf/2608.29249)  

**Abstract**: The online culinary ecosystem is increasingly populated by recipe content generated, modified, or summarized by Large Language Models (LLMs). While often plausible, such outputs may contain hallucinated ingredients, misrepresented quantities, or culturally implausible combinations, limiting their suitability for downstream applications and knowledge graph construction. In this paper, we present a semi-automated soundness assessment workflow for validating structured recipe data extracted and augmented by LLMs from informal culinary sources. Developed as part of this http URL, a knowledge graph of Indian food, the pipeline identifies and addresses common failure modes, including structural inconsistencies, semantic and logical incoherence, and deviations from the source text, through a multi-stage process combining formal grammars, vocabulary-based checks, statistical heuristics, Set Transformer-based coherence modeling, and retrieval-based verification. Although evaluated on Indian recipes, the proposed methods are applicable to broader multilingual and multicultural culinary domains. We provide a practical, auditable, and application-agnostic framework for validating LLM-augmented recipe data, thereby strengthening the foundations of machine-readable food knowledge infrastructures in the era of LLM-generated content. 

---
# Localizing Emergent Failures in Agentic AI: Recovering Minimal Repair Families via Counterfactual Replay 

**Authors**: Bingjie Li, Yumeng Song, Zhongming Yao, Tianyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.29228)  

**Abstract**: Failures in agentic AI systems can arise from interactions among messages exchanged by multiple large language model (LLM) agents. Pointwise attribution cannot distinguish a jointly necessary repair from alternative singleton repairs. We formulate Minimal Repair Family Recovery (MRFR): recovering all inclusion-minimal event sets whose counterfactual replay restores task success within a declared size bound. We propose Graph-Constrained Joint Replay (GCJR), which slices failure-relevant events from an execution dependency graph, constructs graph-feasible singleton and pair candidates, and verifies them by replay with paired clean counterparts. For fixed replay outcomes, GCJR is exact within its declared graph domain. On 90 in-scope cases from a 120-DAG controlled benchmark, GCJR achieves 1.000 Family Exact Match while reducing mean replay calls from 56.3 to 25.3 (55.1%) relative to exhaustive search. On a 24-case, four-agent LLM pilot, it again achieves 1.000 Family Exact Match and reduces mean model calls from 21.0 to 10.0 (52.4%); single-event replay misses jointly necessary repairs. 

---
# Computational Depth Measurement in Thermographic Video: Overcoming Spatial Overfitting via Spatio-Temporal Decoupling 

**Authors**: Zain Ul Abidin, Habeeban Memon, Junaid Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2608.29223)  

**Abstract**: Accurate through-thickness measurement of subsurface delamination depth in Carbon Fiber Reinforced Polymer (CFRP) is important for structural assessment because defect location determines affected load-bearing layers. Optical pulsed thermography (OPT) provides a two-dimensional thermal video rather than volumetric measurements, so depth must be inferred from temporal heat-diffusion responses. A challenge is spatial dataset bias: when calibration defects follow regular grids, regression models may memorize their geometry instead of learning physical relationship between thermal decay and depth. This work introduces a spatio-temporal decoupling architecture that separates spatial defect localization from temporal depth measurement. Defect regions are first localized using segmentation methods, after which thermal responses are spatially averaged and converted into sixteen physics-informed temporal, energy, statistical, and geometric features. These features expose the one-dimensional heat-conduction relationship while withholding pixel coordinates from the depth model. Four regression models are evaluated using specimen-level cross-validation: Random Forest (RF), Gradient Boosting Machine (GBM), Advanced Multi-Layer Perceptron (Adv-MLP), and XGBoost. Unregularized trees and over-parameterized Adv-MLP exhibit calibration collapse under geometric shifts, with errors exceeding 0.5 mm. In contrast, regularized XGBoost with L1/L2 penalties and column sampling maintains cross-specimen calibration, achieving a mean absolute error (MAE) of 0.056 mm and root mean square error (RMSE) of 0.085 mm. Predicted depths are merged with masks to generate Delaunay-triangulated three-dimensional defect models in three to five seconds per specimen. Results show that mathematical regularization and spatio-temporal decoupling reduce spatial memorization in thermal-video depth regression. 

---
# Imag-Eval: a language-grounded framework for interpretable Text-to-Image instruction following evaluation 

**Authors**: Ibrahim Mohamed Serouis, David Jaramillo Duque  

**Link**: [PDF](https://arxiv.org/pdf/2608.29210)  

**Abstract**: Text-to-Image (T2I) models have recently achieved impressive visual fidelity, yet their evaluation remains constrained by benchmarks that are often difficult to interpret and insufficiently diagnostic. Existing skill-based evaluations tend to overlook critical failure modes that strongly impact usability but fall outside standard taxonomies, such as global incoherence arising from missing parts or physically implausible configurations (e.g., floating objects). In addition, prompt difficulty is typically controlled along a single dimension; either prompt length or the number of elements to generate. To address these limitations, we introduce Imag-Eval, a controlled benchmark designed to assess how T2I models ground compositional natural-language instructions into visual outputs. Unlike prior work that conflates surface linguistic complexity with compositional difficulty, Imag-Eval explicitly seeks to disentangles these factors by independently varying both the number of instances and the combination of constraints (rules), while avoiding error propagation. This design enables fine-grained and interpretable analysis of where cross-modal instruction following fails. Our benchmark comprises 1,140 prompts and 8,842 combined rules, and we evaluate it on several state-of-the-art models. Complementing this analysis with an additional study of over 2,000 prompts from a concurrent benchmark, our results suggest that, for structured skills, compositional difficulty is primarily governed by the number of grounded rules and their binding to instances,, rather than by prompt length alone. 

---
# Hyper-Fold: Exploring the Expressive Limit of Sequence-Geometry Learning for Proteins via Hypergraph Modeling 

**Authors**: Yifan Feng, Guanjie Cheng, Shihui Ying, Shaoyi Du, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.29207)  

**Abstract**: Protein structure modeling rests on a single computational primitive: the interaction between what a residue is (sequence content) and where it sits (three-dimensional geometry). What is the expressive limit of this layer class? We show that the complete bilinear operator over content-geometry outer products--the sufficient statistic of all second-order interactions--is the expressive ceiling, while the additive message passing of mainstream geometric GNNs is provably blind to content-geometry binding. We then introduce Hyper-Fold, a rank-K separable convolutional backbone approaching this ceiling at message-passing cost: each radius neighborhood is organized into a sequence hyperedge and a contact hyperedge, modulated by an edge-conditioned matrix-valued operator factorized into K learned basis operators with geometry-generated coefficients. Across enzyme function prediction, fold classification, and ligand binding site detection, Hyper-Fold and its hierarchical variant Hyper-Fold-Deep achieve the best results among protein-specific structure encoders; Hyper-Fold-Pocket, an anchored set-prediction head, surpasses UniSite-3D on UniSite-DS and two zero-shot benchmarks with no sequence language model features, 68x fewer parameters, and 4.8x lower latency--suggesting that a sufficiently expressive 3D backbone recovers information that fusion architectures previously borrowed from evolution-scale pretraining. 

---
# Benevolent Bias in Multi-Turn Human-Agent Dialogue 

**Authors**: Qianqi Liu, Jin Huang, Fethiye Irmak Dogan, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2608.29206)  

**Abstract**: Bias in human-agent interaction can manifest not only through hostile language but also as benevolent bias, whereby unequal treatment hides behind a warm, positive tone. To make it detectable, we operationalise benevolent bias along two dimensions, tone and treatment, yielding three classes: neutral support, overt bias, and benevolent bias. Building on these definitions, we construct BENEVDIAL, a class-balanced corpus of 362,880 multi-turn support dialogues spanning user and agent demographics, roles, and generators, to support controlled evaluation. We then test two detector families on it: off-the-shelf safety detectors and prompted large language model (LLM) judges. Our results reveal a detection gap: off-the-shelf detectors reliably flag overt bias yet largely miss benevolent bias, while LLM judges catch more under more explicit detection criteria but increasingly misclassify neutral support as benevolent bias, and demographic context amplifies the false alarms. These findings suggest that fair monitoring of human-agent dialogue must look beyond surface cues to whether the agent's treatment is disparate. 

---
# How Identity and Opinion Shape Political Sycophancy in LLMs 

**Authors**: Li-Ni Fu, Chang-Chih Meng, Chien-Hua Chen, Hen-Hsen Huang, I-Chen Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.29198)  

**Abstract**: As Large Language Models (LLMs) increasingly encourage users to disclose personal profiles for tailored assistance, measuring their political alignment becomes increasingly important. However, many existing benchmarks for assessing political behavior rely on closed-ended questions and do not fully capture how a model's stance may adapt to user-provided context during interaction. We introduce a framework that disentangles two distinct triggers of political sycophancy: opinion (aligning with explicit narratives) and identity (stereotyping based on demographic labels). Using 450 manually-checked political dilemmas as controlled probes, we evaluate 13 instruction-tuned LLMs. We uncover a dissociation: a model's susceptibility to explicit opinions does not necessarily predict its susceptibility to identity cues, and vice versa. When both signals are present, their effects are generally sub-additive rather than simply additive. Additionally, system-level personas primarily shift a model's baseline stance while having limited effect on the stance shift caused by user opinion or identity. Ultimately, our results suggest that LLM political stance is interactively and steerably vulnerable rather than being a fixed trait, highlighting how personalization may amplify identity- or opinion-conditioned shifts in the model's behaviors. 

---
# An Explainable Coherence Score for Detecting Temporal Inconsistencies in Political News 

**Authors**: Marius Nicusor Pantea, Adrian Groza  

**Link**: [PDF](https://arxiv.org/pdf/2608.29175)  

**Abstract**: Temporal inconsistencies, such as mandates attributed outside their real interval, events presented as past before they occurred, or inverted causal sequences, are a form of political disinformation that evades style-based fake news detectors: a well-written article with a single wrong date carries no lexical signal of falsehood. This paper introduces the Temporal Coherence Score (TCS), a continuous, intrinsically interpretable metric that quantifies the temporal coherence of a news article, computed by a four-stage pipeline: extraction of temporal facts, construction of a temporal knowledge graph, hierarchical verification against internal consistency rules and external reference sources, and score aggregation with automatically generated explanations. Verification combines eight internal checkers derived from Allen's interval algebra with a five-level external hierarchy ranging from a locally stored reference knowledge base of 1{,}256 curated political facts to live Wikidata SPARQL queries. On a benchmark of 100 political news articles with injected temporal errors, the system reaches a precision of 0.909 at the selected operating threshold, with a single residual false positive, a profile deliberately tuned for human-in-the-loop fact-checking assistance, where false alarms are costlier than missed detections. Unlike lexical baselines that output only a binary label, every flagged article is accompanied by the inconsistency type, the entities involved, and the reference source that contradicts the claim. 

---
# JudgePanel: A Compact Judge with Panel Deliberation via Adaptive Multi-Reward Reinforcement Learning 

**Authors**: Yiyue Qian, Shinan Zhang, Huan Song, Hannah Marlowe  

**Link**: [PDF](https://arxiv.org/pdf/2608.29168)  

**Abstract**: The LLM-as-a-Judge paradigm has emerged as a scalable alternative to human evaluation. However, single-model judges are limited by their inherent model biases, while multi-agent evaluation protocols that mitigate this through diverse deliberation are prohibitively expensive at inference time. To this end, we propose \textbf{\modelname}, which equips a compact \underline{Judge} model with multi-agent \underline{Panel} deliberation capability. Specifically, we first train on panel deliberation traces from an ensemble of strong evaluators, capturing structured patterns of discussion, disagreement, and resolution. To further improve judgment quality beyond SFT, we introduce \textit{AdaReward}, an adaptive multi-reward RL algorithm that dynamically rebalances reward component weights as different objectives saturate at different rates during RL training. For practical deployment, we further design a lightweight domain specialization module for rapid adaptation to new evaluation domains with few hundred labeled samples. As a result, (i) \textit{Novel}: the first framework to equip a single compact judge with multi-agent panel deliberation capability at single-model inference cost; (ii) \textit{Effective \& Reliable}: JudgePanel with a 14B backbone outperforms judge-specialized models up to 70B across four evaluation benchmarks, demonstrates strong position consistency, and rapidly specializes to new domains with few hundred samples. 

---
# More Perspectives, Stronger Signals: Multi-Perspective Enhancement and Progressive Fusion for Multimodal Entity Representation Learning 

**Authors**: Chenyi Xiong, Yan Zhang, Jing Hu, Ziyue Qin, Kui Xiao, Xiaopan Lyu, Xiaoju Hou, Zhifei Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.29139)  

**Abstract**: Learning effective multimodal entity representations is fundamental for reasoning tasks such as multimodal knowledge graph completion (MMKGC). However, existing methods often suffer from semantic over-smoothing within modalities and ineffective noise filtration across modalities, particularly under sparse or ambiguous conditions. To overcome these limitations, we propose PrismF, a unified framework that synergizes multi-perspective enhancement with progressive fusion to extract stronger signals from diverse inputs. PrismF enhances fine-grained intra-modal semantics through a multi-perspective mechanism that decomposes each modality into complementary views and constrains them with a decoupling loss to reduce representation collapse. Furthermore, it improves cross-modal integration through a progressive fusion strategy that dynamically calibrates inter-modal interactions, enabling the model to emphasize informative signals while suppressing noisy or unreliable ones. Extensive experiments on three public benchmarks show that PrismF achieves the strongest overall performance, including relative improvements of 4.04% in MRR and 11.17% in Hits@1 on KVC16K. Our code can be found at this https URL. 

---
# APIFlow-Bench: Measuring Whether Agents Survive Long, Dependent API Workflows 

**Authors**: Zelin Wan, Arash Nourian, Xiaoxiao Li, Nihar Nandan, Kamalakannan Nandagopal  

**Link**: [PDF](https://arxiv.org/pdf/2608.29128)  

**Abstract**: Tool-using agents are commonly evaluated by a single bit: whether an end-to-end workflow completed. This metric fails to distinguish failures that matter in production, such as expired credentials, malformed payloads, or correct execution followed by incorrect final delivery. We introduce APIFlow-Bench, a fully auditable benchmark for long-horizon, dependent REST-API workflows that decomposes performance into seven engineering capabilities and requires agents to produce answers supported by the actual call path. We generate synthetic API worlds forward, subtask by subtask; each subtask is admitted only after a zero-LLM self-test triad verifies its grader and an oracle establishes solvability, and an adversarial audit identified and fixed six grader exploits. Grading is deterministic and provenance-sensitive: a state check traces a mock-minted canary through the API data flow to the response the answer must originate from, and a typed answer card is verified field by field. We release all answer keys and 44,362 unredacted execution transcripts. Across 19 frontier and open-weight models under one neutral scaffold, we find: (1) longer dependency chains degrade success, from 93% on individual subtasks to 74% on clean 20-subtask chains and 61% when including the 8% of chain trials that a model-consensus screen flags as passed by no model; (2) reliability separates models more than best-case capability, with best-of-five spanning seven points but all-five-of-five reliability spanning 44 points; (3) the independent-error account of compounding failure does not fit the data: pass rates on 20-subtask chains are 33 percentage points above the product of subtask-level rates, and on the clean slice 77% of failing runs reached the correct final state and failed only at delivery. 

---
# Beyond Correctness: Validity-Oriented Evaluation of Biomedical LLM Judges 

**Authors**: Rodrigo de Oliveira, Federico Pittino, James Gwinnutt, Jay Nanavati  

**Link**: [PDF](https://arxiv.org/pdf/2608.29127)  

**Abstract**: We propose a scalable, validity-oriented pipeline for evaluating biomedical LLM judges when high-quality human judgments are scarce. First, we augment existing human-labelled biomedical benchmarks with deterministic, metric-grounded mutations that produce auditable preference pairs. Second, we evaluate judges beyond aggregate correctness using three deployment-relevant dimensions: correctness against metric-derived gold labels, robustness under repeated stochastic sampling, and compliance with the requested output format. We use this pipeline to assess Llama-3.1-8B-Instruct under four regimes: (1) base, using the instruct model as is; (2) SFT, distillation-based supervised fine-tuning only; (3) RL, GRPO-based reinforcement learning only; and (4) SFT$\rightarrow$RL, SFT followed by RL. The base and single-stage regimes struggle on structured medical discrimination such as PICO extraction and clinical calculations, whereas SFT$\rightarrow$RL performs best across correctness, compliance, and robustness; gains concentrate on decomposable tasks (PICO, MedCalc), at times matching or outperforming frontier models. 

---
# Emergent Misalignment Is Not Magical 

**Authors**: Mingxuan Li, Qirun Dai, Heran Wang, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2608.29118)  

**Abstract**: Fine-tuning large language models (LLMs) on narrowly harmful datasets can lead to misalignment broadly, a phenomenon known as emergent misalignment (EM). EM poses a challenge for AI safety and our understanding of LLMs. Prior work often frames EM as an unexpected behavior, and explains it by appealing to general misalignment directions or anthropomorphizing it as acquiring an evil persona. However, the mechanisms behind these framings remain obscure. In this work, we show that EM is a predictable and data-dependent generalization phenomenon. By examining the base model's representation of EM training data and evaluation prompts, we find that evilness after EM training is highly predictable from representational distance: the closer an evaluation prompt is to training data centroid, the more evilness it elicits from EM models after training (with an average Spearman correlation of -0.73 across 12 model-dataset settings). Building upon this analysis, we further demystify EM by showing that (1) its effectiveness changes significantly based on training data format; (2) there is not a general misalignment direction that transfers across different EM models; (3) the effect of EM is fundamentally different from persona changes. Furthermore, we extend the EM generalization metric from a scalar distance to a dataset-specific generalization direction, which robustly predicts EM models' evilness under semantics-preserving prompt perturbations including appending random tokens and paraphrasing, where other methods do not reliably generalize. 

---
# Clustering as Approximation by Constrained Projectors: Theory and Guarantees 

**Authors**: Angshul Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2608.29102)  

**Abstract**: This paper develops a unified theoretical framework showing that a broad family of clustering methods, including k-means, fuzzy c-means, kernel k-means, kernel FCM, and spectral clustering, can all be expressed as structured low-rank projectors acting on a signal-derived matrix. By formulating each method as an instance of min over B in C of ||M - M P_B||_F^2, with different constraint sets C, we establish a common optimization template that clarifies the algebraic links among hard, fuzzy, kernel-induced, and orthonormal projections. Within this framework, we derive non-trivial theoretical results, including geodesic convexity properties on the projection manifold, perturbation bounds quantifying stability to matrix noise, and exact recovery guarantees under ideal block-model conditions. The analysis further explains when different clustering families collapse to the same optimal subspace and how deviations arise under small inter-cluster leakage. Overall, the work provides a coherent, theory-first foundation for understanding clustering through structured projectors. 

---
# SafeAtlas-VL: Beyond Binary Multimodal Safety with Large-Scale Data and Guard Models 

**Authors**: Zongrui Wang, Xiangyang Zhu, Sicheng Wang, Han Wang, Dingyi Rong, Zeyu Zhang, Chunyi Li, Yue Shi, Kaiwei Zhang, Zicheng Zhang, Yuan Tian, Qi Jia, Yan Teng, Wei Sun, Ning Liu, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2608.29098)  

**Abstract**: Multimodal safety moderation requires distinguishing risks arising from visual content, user intent, and assistant behavior. Existing safeguards, however, are typically trained for a single judgment target and reduce safety assessment to a binary decision. Consequently, risk becomes difficult to compare across a multimodal interaction, and ambiguous cases are obscured. We introduce SafeAtlas-VL, a dataset of 1.5M training instances that places image-, request-, and response-level judgments on a five-level ordered scale. We curate a broad collection of safety-relevant data from both real-world and synthetic sources and apply a disagreement-aware annotation procedure. The resulting dataset spans 15 harm categories and 55 fine-grained subcategories, covering a broad range of multimodal safety scenarios. We also construct SafeAtlas-Bench, a held-out set of 5,000 instances for evaluating five-level predictions and continuous risk scores. Upon this dataset, we train the SafeAtlas Guard series of models via target-conditioned tuning for multimodal safety detection. Our models not only perform five-way classification of safety levels but also map safety to continuous scores through a soft cumulative ordinal head. Experimental results demonstrate that guard models trained on our dataset exhibit strong generalization: even without using the training sets of other benchmarks, they achieve competitive performance on the corresponding test sets. Notably, our 8B model attains the overall best performance, outperforming the previous SOTA by approximately 4% in F1 score. Code, data, and models are released to support further research. Warning: this paper contains example data that may be offensive, harmful, graphic, or disturbing. 

---
# EviAnchor: Mitigating Hallucinations in Large Vision-Language Models via Regional Visual Evidence Compensation 

**Authors**: Sihang Jia, Shuliang Liu, Songbo Yang, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.29092)  

**Abstract**: Large vision-language models (LVLMs) frequently generate content unsupported by visual inputs. Preliminary experiments show that visual evidence is primarily incorporated into answer-side representations in early-to-middle decoder layers, while its direct influence progressively weakens in later layers. This attenuation suggests that visual evidence acquired earlier may be insufficiently utilized during subsequent generation. Based on this observation, we propose EviAnchor, a training-free and single-branch inference framework that preserves and reactivates visual evidence throughout generation. EviAnchor introduces Regional Evidence Anchor (REA) slots to progressively aggregate dense visual tokens into spatially structured representations. It then strengthens the current decision state's access to these visual anchors through decision-conditioned evidence routing, mitigating excessive dependence on textual context. Finally, the model resumes its native Transformer computation to integrate the retrieved visual evidence with question semantics and generation history. Experiments across POPE, CHAIR, and MMHal-Bench demonstrate consistent improvements in visual grounding. 

---
# HANIA: Planner-Guided Multimodal Graph Evidence Selection for Grounded Question Answering 

**Authors**: Zafar Ali, Asad Khan, Nimbeshaho Thierry, Nabila Amir, Adam A. Q. Mohammed, Pavlos Kefalas  

**Link**: [PDF](https://arxiv.org/pdf/2608.29088)  

**Abstract**: Multimodal question answering remains sensitive to noisy, incomplete, and weakly grounded evidence. Long unstructured contexts can introduce redundancy and encourage unsupported generation, while flat retrieval may overlook relations needed for multi-step reasoning. We present HANIA, a planner-guided multimodal graph framework for evidence-grounded question answering. HANIA processes the supplied image and text using a frozen vision-language model to extract concise question-relevant visual evidence with explicit abstention. It then constructs an input-grounded multimodal graph and applies a two-group finite-state planner to coordinate descriptive and relational evidence. Coverage-aware pruning retains a compact evidence set based on relevance, graph confidence, concept coverage, and modality diversity. The selected passages, visual statements, and graph triples are provided to a frozen instruction-tuned decoder. We evaluate HANIA on ScienceQA using answer accuracy, evidence-filtering quality, evidence-budget sensitivity, and efficiency. The results show that structured evidence planning and compact graph-guided retrieval can support competitive multimodal question answering without target-dataset fine-tuning or iterative retrieval. The code is available at this https URL. 

---
# Nested Convex-Body Chasing for Online Optimization with Evolving Feasible Sets 

**Authors**: Dhruv Sarkar, Aprameyo Chakrabartty  

**Link**: [PDF](https://arxiv.org/pdf/2608.29074)  

**Abstract**: We study online optimization with nested shrinking feasible regions in two settings: convex optimization with nested evolving feasible sets (CONES) and adversarial constrained online convex optimization (COCO). Our algorithms separate loss control from geometric movement: constrained minimizers and cumulative-loss tests preserve regret guarantees, while a deterministic resettable nested convex-body chaser limits movement.
For CONES with a $G$-Lipschitz, $\mu$-strongly convex objective on a diameter-$D$ domain, we chase intersections of the current feasible set with adaptive objective sublevel sets. Using the Euclidean chasing ratio $O(\sqrt{d\log(1+d)})$, we obtain nonpositive regret at every prefix and movement $O(\sqrt{d\log(1+d)\,GD\log(eT)/\mu})$. The bound adapts to the increase in the constrained optimum value. In dimension two, with all other parameters fixed, every randomized algorithm with terminal expected regret $O(T^\beta)$, $\beta<1$, suffers $\Omega(\sqrt{\log T})$ expected movement on some deterministic nested sequence, proving optimal horizon dependence. Under linear growth away from the constrained minimizer set, Steiner-point tracking yields movement independent of $T$.
For general convex COCO, one-step-delayed chasing with regularized-leader resets gives regret $O(G_fD\sqrt{d\log(1+d)T})$ and cumulative constraint violation $O(G_gD\sqrt{d\log(1+d)T})$. For strongly convex losses, both are $O(d\log(1+d)\log(eT))$ when other parameters are fixed. These reductions replace the $O(d^{d/2})$ projection-path factor in prior analyses by the polynomial dimension dependence of Euclidean nested convex-body chasing. 

---
# Revolutionizing Turn-by-Turn Navigation with Cloud-Edge Deep Learning 

**Authors**: Yiming Yang, Hao Fu, Fanxiang Zeng, Xikai Yang, Yue Liu, Ning Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.29073)  

**Abstract**: Turn-by-turn (TBT) navigation systems are integral to modern driving experiences, providing real-time audio instructions to guide drivers safely to destinations. However, existing audio instruction policy often relies on rule-based approaches that struggle to balance informational content with cognitive load, potentially leading to driver confusion or missed turns in complex environments. To overcome these difficulties, we first model the generation of navigation instructions as a multi-task learning problem by decomposing the audio content into combinations of modular elements. Then, we propose a novel deep learning framework that leverages the powerful spatiotemporal information processing capabilities of Transformers and the strong multi-task learning abilities of Mixture of Experts (MoE) to generate real-time, context-aware audio instructions for TBT driving navigation. A cloud-edge collaborative architecture is implemented to handle the computational demands of the model, ensuring scalability and real-time performance for practical applications. Experimental results in the real world demonstrate that the proposed method significantly reduces the yaw rate (the proportion of vehicles deviating from navigation routes) compared to traditional methods, delivering clearer and more effective audio instructions. This is the first large-scale application of deep learning in driving audio navigation, marking a substantial advancement in intelligent transportation and driving assistance technologies. 

---
# Agent2UCB: Agentic System for Generative Engine Optimization 

**Authors**: Sheldon Yu, Rui Wang, Tong Yu, Sungchul Kim, Doga Dogan, Junda Wu, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2608.29063)  

**Abstract**: Large language model driven search engines such as Google AI Overviews and Perplexity have created new opportunities for Generative Engine Optimization (GEO) the practice of refining content to increase its likelihood of being cited or summarized by generative systems. We demonstrate Agent2UCB, an agentic GEO system that autonomously improves content visibility through customized, feedback-driven optimization. For each content item, the system evaluates nine GEO strategies, identifies the most effective method, and accelerates selection using a bandit-based Agent2UCB policy that integrates LLM priors with online reward signals. To monitor side effects, the system also provides a lightweight, text-only SEO readiness evaluation covering readability, topical coverage, and EEAT-style credibility. Experiments on GEO-Bench show consistent visibility gains while preserving SEO quality. The demo allows users to choose the websites of interest, observe the optimization workflow, and compare GEO/SEO outcomes across methods. 

---
# Let Prompts Bridge Defense Knowledge: Transferable Graph Purification via Vulnerability-Aware GPL 

**Authors**: Shuomin Xue, Jingyuan Li, Ju Jia, Jingxuan Yu, Xiaojun Jia  

**Link**: [PDF](https://arxiv.org/pdf/2608.29054)  

**Abstract**: Graph Neural Networks (GNNs) have emerged as a cornerstone for representing complex relational dependencies in diverse multimedia tasks, particularly in cross-platform user interest modeling and cross-modal semantic alignment. In the real world, a practical defense against graph adversarial perturbations is needed. However, we observe that the prevailing adversarial purification methods are essentially domain-restricted defenses, which leads to the following shortcomings: (1) single-domain data provides insufficient structural and semantic diversity for learning robust purification criteria; (2) training of domain-specific defense strategies from scratch consumes substantial computational cost. To address the above limitations, we propose a transferable graph purification scheme, named ProGAP, to bridge adversarial defense knowledge via vulnerability-aware graph prompt learning. Firstly, to capture universal adversarial patterns, a perturbation-capture edge detector is pretrained on data-rich graphs by jointly modeling topological and semantic information. Subsequently, to achieve more knowledge transfer w.r.t. robustness, vulnerability-aware prompts are designed that inject targeted purification guidance into biased nodes, during which the pretrained detector adapts to distribution shifts in downstream graphs without parameter-laborious updates. Experimental results demonstrate that compared with state-of-the-art baselines, our ProGAP achieves 1%-9% improvement, and reduces the time consumption by up to 2.2x. The code for ProGAP is available at this https URL. 

---
# EmoLASP: Emotion Recognition with Language Models and Answer Set Programming 

**Authors**: Thao Le, Michael Thielscher  

**Link**: [PDF](https://arxiv.org/pdf/2608.29035)  

**Abstract**: Emotion recognition in conversations is increasingly tackled with language models, but these models can be unstable and expensive to fine-tune or to prompt with long dialogue histories. We propose EmoLASP, a framework that combines a language model with declarative reasoning via Answer Set Programming (ASP) to predict VAD scores (Valence-Arousal-Dominance) in conversations. Experiments on a widely used benchmark dataset (IEMOCAP) across six open-source LLMs (3B-120B) and two PLMs (BERT, RoBERTa) show that EmoLASP improves prediction performance compared to using the language model alone, even when the LLMs/PLMs are given no dialogue history in their prompts or input vectors. The gains are largest for prompt-only LLMs, which EmoLASP uses without any fine-tuning. However, for fine-tuned PLMs, the reasoner adds little once dialogue history is available. EmoLASP's LLM pipeline demonstrates the potential advantages of using a reasoning approach to ensure emotion prediction consistency and to reduce both the cost of fine-tuning and the cost of prompting with long dialogue histories. 

---
# Learning to Follow In-Context Watermark Instructions via Self-Distillation 

**Authors**: Yepeng Liu, Tianyi Chen, Xuandong Zhao, Dawn Song, Yuheng Bu  

**Link**: [PDF](https://arxiv.org/pdf/2608.29030)  

**Abstract**: In-context watermarking (ICW) prepends an instruction to a query asking the model to embed a statistically detectable signal in its response. It thus equips LLMs with a watermarking interface that third parties can invoke without access to model internals. Its reliability hinges on the LLM following the instruction without degrading answer quality, yet how well current LLMs do so has not been measured. We introduce $\mathsf{ICWBench}$, a benchmark of three verifiable ICW instruction families, each scored on both detectability and answer quality. Evaluating 14 frontier proprietary and open-source LLMs, we find that none of the evaluated LLMs achieves both objectives across all three families. To address this, we propose a self-contained two-stage training method, requiring no distillation from a stronger model, no manual annotation, and no pre-existing ICW IF ability. The first stage, self-distillation with logits perturbation (SDLP), uses the same base LLM as both teacher and student: an instruction-equivalent decoding-time logits perturbation makes the teacher follow the ICW instruction, and the student is trained to match the teacher's output distribution. The second stage applies reinforcement learning with the automatic verifier as the reward. Applied to Qwen3-14B and GPT-OSS-20B, our method raises average TPR@$1\%$FPR across three ICW instructions from $0.100$ to $0.974$ and from $0.337$ to $0.968$, respectively, while maintaining high response quality under both perplexity evaluation and LLM-as-a-Judge. 

---
# Facts Without Rules: Boundary Metadata Collapse in Multi-Agent LLM Handoffs 

**Authors**: Yian Wang, Agam Goyal, Eshwar Chandrasekharan, Hari Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2608.29028)  

**Abstract**: Multi-agent LLM systems often coordinate by compressing an upstream interaction into a handoff artifact that downstream agents treat as shared state. We show that this handoff step is a structural source of privacy leakage: summaries preferentially preserve operational facts while weakening the boundary metadata that governs how those facts may be used---a failure mode we call \emph{summary collapse}. On a controlled multi-agent coordination testbed we measure marker survival with a human-validated judge ($\kappa = 0.74$), where $\sigma_b = 1$ means every boundary marker survives verbatim and $\sigma_b = 0$ means all are lost. Boundary-marker and operational-fact survival are nearly uncorrelated at the handoff level on both GPT-5-mini and DeepSeek-R1-32B (Pearson $r$ near zero): uncompressed free-text handoffs preserve boundaries at $\sigma_b \approx 0.80$, whereas a $25$-word budget drops $\sigma_b$ to ${\approx}0.57$ while operational-fact survival stays near ceiling. Controlled downstream tests reveal that protection depends on \emph{boundary explicitness}: vague languages leak in $73\%$ of GPT and $50\%$ of DeepSeek cases, while explicit constraints reduce leakage to under $15\%$ across all three tested models. A no-handoff single-agent control further shows the failure is not reducible to multi-agent topology as direct full-marker access still leaks more often than the operationalized handoff. Prompt-only mitigation and exact-string redaction only partially address the problem, while a gold-derived audience allowlist nearly eliminates leakage across models, showing that correctly identifying audience boundaries is the key factor. 

---
# Disentangling Representation using Attributes-based Gaussian Estimation for Medical Sound Diagnosis 

**Authors**: Ke Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.29026)  

**Abstract**: Deep learning has a powerful capability of feature extraction. However, the lack of fairness and interpretability in deep neural networks poses limitations to their adoption in the medical domain. This paper proposes a disentangled representation learning (DisenRL) framework, named the Attributes-based Gaussian Estimation for Disentangled Representation (AGEDR), which incorporates Attribute Mapping Embedding (AME) modules designed to map attributes into vectors and align them with a subset of the latent vectors in a Variational AutoEncoder (VAE). This part of the latent vector will be disentangled from the remaining latent vectors by minimizing mutual information. A classifier is then trained using the mean parameters of the latent vectors from the VAE. Extensive experiments demonstrate that AGEDR outperforms both conventional classification models and existing disentangled representation learning methods. The ablation experiments also indicate the disentangling capability and fairness of AGEDR. The source code is publicly available at this https URL. 

---
# Frequency Selective Neural Networks as a Foundation Architecture for Time Series Learning 

**Authors**: Hui Huang, Ye Sun, Shiyan Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.29012)  

**Abstract**: Time-series data across physical and biological domains are fundamentally driven by complex, non-stationary oscillatory modes. While deep learning models, such as Convolutional Neural Networks (CNNs), Recurrent Neural Networks, and Transformers, have dominated sequential analysis, they remain fundamentally "spectral-blind". By mapping continuous physical waves into unconstrained spatial or discrete token spaces, these architectures suffer from severe spectral entanglement, acting as opaque black boxes that decouple predictive accuracy from physical reality. In this paper, we introduce the Frequency Selective Neural Network (FSNN), pioneering a foundation architecture guaranteeing physical interpretability without sacrificing expressive power of deep learning. FSNN addresses spectral entanglement by explicitly embedding the rigorous mathematics of advanced signal processing into its neural topology. Through a fully differentiable Wiener-like filter bank optimized via complex-domain backpropagation, FSNN autonomously discovers and isolates the precise physical modes of a given task. Extensive evaluations demonstrate that FSNN establishes state-of-the-art predictive performance, achieving $77.0\%$ average accuracy on the standard 10 multivariate UEA datasets and leading across all major metrics on the highly imbalanced PTB-XL clinical ECG benchmark. Crucially, in contrast to yielding abstract feature maps, FSNN converges directly on physically meaningful frequency bands, such as isolating the cardiac QRS complex, providing a highly scalable, interpretable paradigm for robust pattern recognition in complex temporal domains. Our code is available at: this https URL. 

---
# Multi-Step Forecasting of Grape Berry Temperature based on LSTM Model with Feed-Forward Attention 

**Authors**: Srikanth Gorthi, L. G. Divyanth, Dattatray Bhalekar, Markus Keller, Lav Khot  

**Link**: [PDF](https://arxiv.org/pdf/2608.29008)  

**Abstract**: Accurate forecasting of grape berry temperature (Tb) is essential for enabling timely heat stress management in vineyards. In this study, a feed-forward attention mechanism integrated with a Long Short-Term Memory network (FAM-LSTM) was developed and evaluated for multi-step, high-resolution Tb prediction. Models were trained using environmental data from 2023 and 2024 at Prosser, WA, USA, and validated on 2025 summer data. FAM-LSTM was benchmarked against LSTM, GRU, RNN, and Random Forest (RF) across horizons ranging from 15 minutes to 72 hours (288 time steps). Two input scenarios were evaluated: nearest open-field weather station observations and in-vineyard microclimate measurements. FAM-LSTM consistently outperformed all benchmark models across all horizons and input scenarios. Incorporating in-vineyard microclimate data significantly improved forecasting accuracy at longer horizons. Using open-field data, FAM-LSTM achieved MAE and RMSE ranges of 0.58 to 1.70 deg C and 0.65 to 2.07 deg C, respectively. In-vineyard observations further improved performance, with MAE and RMSE in the ranges of 0.51 to 1.55 deg C and 0.71 to 1.87 deg C. Error analysis showed prediction uncertainty was highest during peak daytime periods (11:00 to 18:00) and increased progressively with forecast horizon. Overall, the FAM-LSTM framework offers robust Tb forecasting to support precision heat stress management in vineyards. 

---
# Verification abundance, adjudication scarcity: what happens to mathematical knowledge when proof checking becomes free 

**Authors**: Maher Kallel, Mohamed El Louadi  

**Link**: [PDF](https://arxiv.org/pdf/2608.28997)  

**Abstract**: In May 2026 an OpenAI model produced a counterexample to the Erdős unit distance conjecture. Five mathematicians published a human-verified version the same day, and the result entered the literature within weeks. In August 2026 the same laboratory published ten mathematical and theoretical computer science results, each accompanied by a machine-checkable Lean 4 certificate with no unproved steps. Four weeks later, one remained the subject of an unresolved dispute over whether its formalization meant what it claimed.
We argue that this difference is structural. We distinguish three layers of verification: derivational validity, which a kernel checks; representational fidelity, whether the formal statement means the intended question; and epistemic significance. Only the first is mechanizable. Making it effectively free therefore does not eliminate verification work but shifts the burden to layers dependent on scarce expert attention.
Measurements of the August corpus illustrate the shift. The kernel-checked proofs total 20.6 MB, while the statements requiring human audit total 55.6 KB, a ratio of 379 to 1. Yet those statements contain 218 bespoke definitions rather than relying on community-vetted ones. The audit surface is therefore small in volume but irreducibly expert. We argue that machine checking produces verification abundance while leaving adjudication scarce. We propose a six-category taxonomy of representational mismatch, a disclosure schema for machine-generated mathematical claims, and implications for software, cryptography, and regulated decision systems. 

---
# Agentic AI uncovers conserved cross-tissue protein co-abundance programs inaccessible to single-dataset analysis 

**Authors**: Runyu Guan, Dehao Wu, Qiqi Xie, Yang Li, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.28990)  

**Abstract**: Protein co-abundance clusters preserved across tissues can reveal shared disease mechanisms and candidate therapeutic targets, particularly when proteins implicated in organ-confined diseases converge in peripheral or accessible tissues. However, previous cross-tissue studies have focused on biologically pre-selected tissue pairs, leaving most possible combinations and non-obvious relationships unexplored. We present an LLM-agent framework for large-scale, evidence-grounded comparison of tissue-specific protein co-abundance networks. The framework constructs tissue networks, derives pairwise consensus clusters, and integrates evidence from expression atlases, protein interaction and complex databases, pathway annotations, disease catalogues, and literature. Applied to all 820 pairwise combinations of 41 human tissues and fluids, it identified 1,833 conserved co-abundance clusters across 406 tissue pairs. Colon, synovial fluid, blood, cerebrospinal fluid, and bone marrow were the most broadly connected tissues, while the most cluster-rich pairs were dominated by bone marrow. The analysis also highlighted non-obvious relationships: skin-bone marrow exceeded the anatomically adjacent bone-bone marrow pair, while colon-breast contained cancer-relevant clusters involving extracellular-matrix remodeling, lipid metabolism, and immune modulation. Cluster-level analyses generated further mechanistic hypotheses, including a brain-gut extracellular-vesicle/redox/serotonin-cofactor axis and a liver-bone marrow stress-response axis involving genes linked to white matter disease. These results provide a global, comparable landscape of conserved protein co-abundance and a hypothesis-generating resource for mechanistic and therapeutic exploration. Code and data are available at this https URL. 

---
# Selective Forgetting: A Graph-Based Memory Framework for Long-Term LLM Agents 

**Authors**: Theo Rusu, Sourena Khanzadeh, Manar Alalfi  

**Link**: [PDF](https://arxiv.org/pdf/2608.28978)  

**Abstract**: Knowledge graphs have been proposed as a structured alternative to flat retrieval-augmented generation for long-term agent memory, on the assumption that representing conversations as entities and relations improves recall. We evaluate that assumption directly. Our framework extracts each conversational turn into typed nodes and attributed edges, answers questions from a two-hop subgraph, and periodically prunes nodes that score low on a weighted combination of recency, access frequency, degree centrality, and age. On LongMemEval, the graph does not outperform a flat vector baseline at a matched candidate-generation budget of five retrieval roots: token F1 is $0.417$ against $0.468$, and a paired bootstrap over 500 questions gives
$\Delta = -0.050$ (95\% CI $[-0.085, -0.016]$). The gap is widest on questions that require recalling a specific prior assistant turn, where judged correctness falls from $0.911$ to $0.607$, suggesting that decomposing a turn into entities discards the surface form these questions depend on. The forgetting module is more successful. Applied once to a persistent 27{,}021-node graph, it removes 9.8\% of nodes and 9.5\% of stored bytes; token F1 is unchanged ($+0.001$, 95\% CI $[-0.015, +0.016]$) and judged correctness falls by $1.6$ points, with the 95\% interval bounding any loss at $3.8$ points ($[-0.038, +0.006]$). Because our extractor is a single small model evaluated on one benchmark, these results characterise this extraction-based pipeline rather than graph-structured memory in general. Code: this https URL 

---
# The Role of Network Topology and Opponent Information in Shaping Cooperation in Multi-Agent Reinforcement Learning Systems 

**Authors**: Seongho Son, Stephen Hailes, Mirco Musolesi  

**Link**: [PDF](https://arxiv.org/pdf/2608.28977)  

**Abstract**: Several works have investigated the influence of graph topology on cooperation among artificial agents, while the majority of the literature has focused on modelling agents' adaptation through strategy imitation, which relies solely on the cumulative payoffs of others. This paper investigates scenarios in which each agent learns to play the two-player Iterated Prisoner's Dilemma (IPD) using deep reinforcement learning. Each agent is represented as a node in a graph, where its neighbours constitute the pool of opponents with whom it can interact. During each IPD episode, agents are provided with different types of information about their opponent, consisting of action history and opponent identity. Experimental results across different graph topologies show that the number of neighbours per node and the average path length are the main factors affecting the emergence of cooperation. We also show that, while partner selection fosters mutual cooperation by limiting the diversity of the opponent pool, providing agents with the identity of their opponent hinders the proliferation of cooperative strategies. 

---
# From Analytics to Tumor Boards: An Evidence-Linked Multi-Agent Workflow for Oncology Feature Extraction 

**Authors**: Daniel Kang, Michelle Hu, Soorya Ram Shimgekar, Shayan Vassef, Yufan Wang, Anit Kumar Sahu, Munmun De Choudhury, Vedant Das Swain, Christian Poellabauer, Li Yan Khor, Koustuv Saha, Robert Wojciechowski, Elliot Kidd, Piyum Zonooz, Navin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2608.28974)  

**Abstract**: Clinically relevant oncology information is distributed across heterogeneous, longitudinal documentation, creating substantial abstraction burden and requiring accurate attribution across specimens, tumors, biomarkers, and time points, while manual cancer-registry abstraction can require 27.2 minutes per case, highlighting the need for scalable methods that preserve clinical context while converting documentation into structured data. We evaluate the Nimblemind Multi-Agent System (nMAS), a configurable oncology information-extraction workflow which extracts clinically relevant structured fields from fragmented oncology documentation. The extraction task uses a clinician-informed schema of 328 attributes spanning report metadata, diagnosis, staging, and cancer-type-specific information. nMAS separates clinician-defined field specifications from model execution and combines complexity-aware extraction, report-level consolidation, and source-grounded validation. The retrospective evaluation included 230 de-identified oncology documents from 40 patients and 418 clinician-reviewed document-field pairs containing 1,126 non-empty reference values. Evaluation focused on fields identified by clinicians as present in the source documents rather than exhaustively annotating all 328 schema fields. nMAS achieved a rank-weighted value-level precision of 82.6%, recall of 87.5%, and F1 of 85.0%, compared with an F1 of 66.4% for an independently implemented UMA-style MiniMax M2.5 comparator. These findings support the feasibility of using a configurable, source-grounded extraction workflow to convert fragmented oncology documentation into reusable structured data. 

---
# Efficient GPU Retrieval for Semantic Search 

**Authors**: Dhritiman Das, Chujie Zheng, Ronak Kaoshik, Pratik Dixit, Vishal Shah, Yanbo Li, Jiahao Xu, Manika Agarwal, Chinmay Naik, Lingyu Zhang, Chetan Bhole, Chirag Bhanuprasad Mehta, Meng Zheng, Puneet Singh Ahluwalia, Shirisha Singh, Ping Jin, Manas Apte, Gokulraj Mohanasundaram, Tugrul Bingol, Raghavan Muthuregunathan, Fedor Borisyuk  

**Link**: [PDF](https://arxiv.org/pdf/2608.28968)  

**Abstract**: Semantic Search on LinkedIn must retrieve relevant profiles from a corpus of hundreds of millions in response to natural-language queries such as "a fintech founder in Berlin who worked in payments." The deployed relevance policy is bottleneck-oriented: every active non-negotiable facet must be satisfied, and a pre-existing LLM Graded Relevance (GR) judge operationalizes this through a fixed min/median aggregation over facet grades. Cosine similarity instead averages evidence, letting a strong match on one facet mask failure on another, capping the recall of the first-stage (L0) retriever.
We present a policy-aligned retrieval framework: embeddings are partitioned into eight category-supervised segments whose scores follow the same min/median rule at serving time; for multi-vector retrieval, this segment score is computed independently per tagged document slot and maximized across slots. A lightweight single-slot Stage-1 scorer generates high-recall candidates, while scale-invariant relative-norm gating keeps category activation consistent across training, evaluation, and serving. On 21K held-out queries, this representation improves offline relevance over a matched-capacity baseline, with gains broadly distributed across facet combinations.
We serve this framework with a two-stage GPU architecture: an FP8 coarse ranker scores the full corpus, increasing per-shard capacity by 71% and Stage-1 matmul throughput by 36%, then an FP16 stage exactly re-ranks an oversampled candidate set, recovering 99.6-99.8% of full-FP16 recall at over 500 QPS per shard replica. In a member-randomized A/B test, exploratory-query Precision@10 under the unchanged GR judge rises from 63.7% to 79.0% and navigational Precision@1 from 65.5% to 74.7%, with a blinded human evaluation independently confirming the Precision@10 gain. 

---
# From Location Phrases to Geographic Entities: Task-Adapted Retrieval for People Search 

**Authors**: Yanbo Li, Chujie Zheng, Jiahao Xu, Chetan Bhole, Lingyu Zhang, Puneet Singh Ahluwalia, Kevin Nguyen, Raghavan Muthuregunathan, Santhosh Sachindran, Sachin Ahuja, Fedor Borisyuk  

**Link**: [PDF](https://arxiv.org/pdf/2608.28965)  

**Abstract**: People search must map free-form location phrases to geographic entities used as structured retrieval filters. Lexical standardizers handle canonical names well but are brittle to aliases, misspellings, metropolitan expressions, and same-name ambiguity. We formulate this task as graded, set-valued entity retrieval over a fixed ontology. We identify three coupled design requirements: distinguishing identity-preserving variation from knowledge-dependent aliases, controlling false negatives among valid same-name entities, and separating stable transformations from mutable entity knowledge. We realize them in a prompt-asymmetric bi-encoder with calibrated alias support, bounded ambiguity-aware negatives, and editable entity documents that support localized updates without retraining.
Across a fixed production-derived development benchmark and a public GeoNames transfer task, task adaptation improves substantially over frozen encoders and standard token baselines. Controlled development ablations show that specialized supervision contributes beyond standard task fine-tuning and encoder scaling. On GeoNames, the adapted model improves known-target Recall@1 throughout zero-to-moderate character overlap, while character n-grams retain a small aggregate Target Recall@5 advantage. In a blinded human comparison on a stratified production challenge set, our model raises relevant P@1 from 28.0% to 46.0% (p=0.012). Fixed-query endpoint estimates improve on non-canonical queries and remain close to control on frequent queries; a randomized live experiment detects no engagement regression. These results support task-adapted geographic entity retrieval as a practical replacement for the incumbent taxonomy-based standardizer, with the largest relevance gains on non-canonical queries. 

---
# Automated Researchers Can Reliably Mitigate Alignment Failures 

**Authors**: Chen Yueh-Han, Jiaxin Wen, Jan Hendrik Kirchner  

**Link**: [PDF](https://arxiv.org/pdf/2608.28945)  

**Abstract**: Automating alignment research may accelerate progress toward aligned AI, but whether it does is hard to measure. Luckily, many alignment failures, such as deception, sycophancy, and jailbreaks, are already measurable by public benchmarks. We study whether automated alignment researchers (AARs) can post-train to mitigate alignment failures by proposing training methods and data to simultaneously optimize multiple safety benchmarks, while preserving general capability. Across 10 alignment failures, the strongest AAR methods significantly reduce the targeted alignment failures and generalize to a held-out benchmark, multi-turn behavioral audits, and models up to 4.7 times larger than the target model. As a human baseline, 28 experienced researchers receive up to eight hours to develop methods for the same benchmarks, but their methods underperform the best AAR methods. Using human ideas as the AARs' initial research direction does not improve performance, suggesting current AARs may not need guidance from experienced researchers. These results suggest that automating alignment research on well-characterized failures may be practical in the near term. 

---
# Oculi: A Conversational Agentic Platform for Automated Credit Risk Analysis 

**Authors**: Vennise Ho, Kristian Diana, Sandy Mourad, Milena Pilipovic, Vineel Nagisetty, Hossein Hajimirsadeghi  

**Link**: [PDF](https://arxiv.org/pdf/2608.28944)  

**Abstract**: Credit risk analysis in financial institutions traditionally requires analysts to manually write SQL queries, run statistical computations, and build visualization dashboards. This is a time-consuming workflow that limits exploration to familiar segments. We introduce \textbf{Oculi}, a conversational platform that transforms natural language questions into comprehensive credit risk analyses, complete with data queries, statistical testing, and interactive visualizations. Oculi employs a three-layer architecture that separates reasoning (LLM-powered agent), execution (Model Context Protocol tool servers), and presentation (agentic UI), enabling analysts to discover high-risk portfolio segments. Within Oculi, a new segment discovery pipeline is proposed that combines deterministic statistical methods with LLM-guided feature selection, leveraging LLM semantic domain knowledge alongside data-driven metrics to identify meaningful, actionable portfolio segments. Evaluated on a mortgage portfolio with 200+ features, Oculi demonstrates effectiveness in discovering material risk segments previously intractable through manual exploration, reducing time-to-insight significantly while maintaining auditability and statistical rigor. 

---
# MineCEraft: Evaluating Language Models as Construction Engineers in the World of Minecraft 

**Authors**: Sewoong Lee, Risham Sidhu, Julia Hockenmaier, Yoonhwa Jung  

**Link**: [PDF](https://arxiv.org/pdf/2608.28884)  

**Abstract**: We introduce MineCEraft (Minecraft Construction Engineering Benchmark, pronounced mine-see-ee-raft), an easy-to-use, open-source benchmark designed to systematically evaluate the reliability and limitations of LLMs for construction tasks in Minecraft. The MineCEraft benchmark comprises 723 domain-expert hand-crafted natural-language instructions with programmatically verifiable evaluation, spanning 17 distinct task categories, providing a safe and controllable experimental environment for assessing LLMs' ability to perform realistic construction engineering tasks. With this benchmark, we conduct an in-depth evaluation of state-of-the-art LLMs and perform a detailed error analysis, revealing key failure modes and practical challenges in applying LLMs to construction engineering tasks. 

---
# Evaluating the Hidden Costs of Personalization in Large Language Models 

**Authors**: Yumeng Wang, Yuchen Wu, Cheng Qian, Zhiyuan Fan, Hyeonjeong Ha, Shujin Wu, Jiayu Liu, Heng Ji, Ge Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.28833)  

**Abstract**: While Large language models (LLMs) incorporate user personalization signals to improve usability and helpfulness, they increasingly shift from providing balanced, informative responses toward optimizing for user satisfaction when conditioned on personal context such as conversation history, inferred preferences, and user profiles. Specifically, we identify three emerging risks: (1) irrelevant personalization, where models reference personal information in unnecessary contexts; (2) preference narrowing, where models reinforce informational echo chambers; and (3) sycophantic bias, where models agree excessively with user opinions. As a result, models may reference personal information in contexts where it is unnecessary, inadvertently collapse response diversity, or agree excessively with user opinions. Despite the growing use of personalization in AI assistants, there has been limited systematic evaluation of its potential side effects. To bridge this gap, we propose PRISK, a dynamic evaluation framework with automated data generation and tailored metrics that uncovers systematic limitations in current LLM personalization and how personalized information shapes its responses. Our empirical analysis across 13 LLMs demonstrates the presence of user profiles and retrieved memories consistently exacerbates biases, resulting in an average drop of 45.9% in irrelevant personalization, 41.7% in preference narrowing and 61.7% in sycophantic bias. 

---
# Discovering Machine Correlates of Consciousness 

**Authors**: Romain Salvi, Ouri Wolfson  

**Link**: [PDF](https://arxiv.org/pdf/2608.28824)  

**Abstract**: Currently, in biological systems Neural Correlates of Consciousness (NCCs) are characterized in terms of EEG and FMRI signals. Unfortunately, this characterization prevents the transferability of the NCCs concept to machines. Such transferability would be useful in order to investigate AI consciousness. In this paper we provide an alternate characterization that is transferable, and enables the analogous definition of Machine Correlates of Consciousness (MCCs). Specifically, we propose that NCCs (MCCs) are substrate-level signals that are not under human (AI agent) control, and that are reliably modulated by emotions.
This paper presents the first empirical investigation of MCCs. Specifically, we present the results of experiments conducted with two LLMs, Llama-2 7B and Llama-3.1 70B parameters. In these LLMs we collect hardware anomaly traces that are substrate-level indicator-sequences. And we show that after controlling for confounding factors, these are modulated differently by emotional and neutral computations. And this difference is statistically significant for the larger Llama-3.1 70B, but not for the smaller Llama-2 7B. The results constitute initial empirical evidence that MCCs are present in the Llama-3.1 70B configuration. And they are consistent with the hypothesis that consciousness probability and degree increase with the LLM sophistication.
Independently of consciousness, MCCs can also be used for detection of emotions in AI agents. 

---
# Explainable Artificial Intelligence (XAI) in Computational Pathology: Definitions, Taxonomy, and Recommendations 

**Authors**: Shubham Innani, Suhang You, Adam Shephard, Bhakti Baheti, Francesco Ciompi, Joe Yeong, Nasir Rajpoot, Michael Feldman, Solene Florence Kammerer-Jacquet, Dimitrios Makris, Geert Litjens, Anne L. Martel, Jana Lipkova, April Khademi, Spyridon Bakas, MICCAI SIG-CompPath  

**Link**: [PDF](https://arxiv.org/pdf/2608.28820)  

**Abstract**: Computational pathology (CompPath) is transforming medicine by leveraging artificial intelligence (AI) algorithms to support diagnosis, prognosis, and treatment prediction from gigapixel whole-slide images. Clinical adoption is progressing, but is constrained by concerns about safety, accountability, and regulatory oversight in high-stakes clinical environments. Explainable AI (XAI) systems hold promise for building trust and enabling verification, yet the literature remains fragmented due to inconsistent terminology, overlapping methodological families, ad hoc validation, and current reviews. This review aims to formalize XAI methods in CompPath through the: i) introduction of a pathology-centric vocabulary comprising seven core terms; ii) development of a taxonomy across methodological families and three orthogonal axes (stage, type, scope); and iii) establishment of a task-driven framework that maps five clinical questions to recommended methods, method evaluation, and deployment context. Five key gaps between current XAI capabilities and clinical deployment are identified, and actionable steps are proposed to advance XAI for CompPath. 

---
# Capability-Stratified Degradation in Ternary Language Models 

**Authors**: Anirudh Malik, M Sparsh Mehra, Poojith Devan  

**Link**: [PDF](https://arxiv.org/pdf/2608.28809)  

**Abstract**: Extreme low-bit inference offers a route toward smaller models and constrained deployment. Ternary language models restrict weights to $\{-1,0,+1\}$, approaching the limit of $\log_2 3 \approx 1.585$ bits/weight. The practical question for a pretrained model is not simply whether weights can be quantised but which capabilities survive and whether it remains useful for adaptation. We explore this by converting Qwen3.5-0.8B (752M parameters) to ternary weights using 72.4M tokens of quantisation-aware training (QAT). The resulting model, Cloe, is evaluated across 29 benchmarks, representation diagnostics, and downstream fine-tuning. The evidence shows non-uniform degradation. A linear probe recovers 43.76% of MMLU answers from the full-precision teacher's representations but only 26.19% from Cloe (near chance), indicating specialist factual information is lost. However, Cloe retains measurable performance on ten tasks, averaging 77.1% of teacher performance. Crucially, fine-tuning raises Cloe to 89.8% on SST-2 (95.6% of the matched teacher) and reaches 79.4% teacher retention on XSum. We attribute degradation to a combination of quantisation-induced information loss and incomplete recovery due to the limited QAT budget. We also highlight an evaluation pitfall: standard answer-letter scoring failed (Cloe emitted "A" on 98.6% of MMLU questions), necessitating continuation scoring. Ultimately, ternary conversion is unsuitable as a drop-in general replacement yet remains valuable as a compact substrate for task-specific models. 

---
# Enhancing SAE-based Steering via Neighbor Integrated Feature Selection 

**Authors**: Yutian Liu, Xu Wang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2608.28806)  

**Abstract**: Sparse autoencoders (SAEs) disentangle model activations into interpretable features and are widely used for steering large language models. Most existing SAE-based steering methods select features by applying a top- filter based on statistical scores, assuming that higher-scoring features yield stronger steering effects. In this paper, we show that this assumption is often invalid, leading to suboptimal feature selection. Our analysis reveals that effective steering features may be distributed among representationally adjacent, semantically similar groups induced by feature splitting in SAEs. Within such groups, features may exhibit disparate statistical scores despite having comparable steering influence, causing score-based selection to overlook important features. Based on these observations, we propose \textsc{Neighbor Integrated Feature Selection} (\textsc{NIFS}), a plug-and-play strategy that leverages representation similarity to improve feature selection for steering. We evaluate \textsc{NIFS} across multiple SAE-based steering methods and tasks, and demonstrate consistent performance gains over conventional top-$k$ selection. 

---
# Efficient Geothermal Well-Control Optimization via Diffusion-Surrogate Reinforcement Learning 

**Authors**: Ruimin Dai, Guodong Chen, Randy Harsuko, Kunpeng Liu, Nori Nakata  

**Link**: [PDF](https://arxiv.org/pdf/2608.28791)  

**Abstract**: Real-time decision-making for enhanced geothermal systems (EGS) is challenging because long-term production periods involve high-dimensional control spaces and a large number of time-consuming high-fidelity hydrothermal simulations. Reinforcement learning provides a natural framework for state-dependent sequential control, but direct policy training with numerical simulators is computationally expensive. To address this issue, we propose a diffusion-surrogate guided reinforcement learning framework for long-horizon EGS well-control optimization. The reservoir temperature and pressure fields are used as system states, while injection rates are selected as control actions. A learned surrogate environment is constructed using conditional diffusion models to predict the evolution of reservoir temperature and pressure fields and a separate reward model to estimate the corresponding economic return. The surrogate environment is then integrated with Proximal Policy Optimization (PPO) for efficient policy training. Experiments on a fractured EGS benchmark show that the diffusion surrogate can accurately reproduce reservoir-state evolution over multiple control stages. The resulting surrogate-assisted PPO policy achieves competitive well-control performance compared with direct simulator-based PPO and existing optimization methods, while substantially reducing the dependence on expensive high-fidelity simulations. These results demonstrate the potential of diffusion-based surrogate environments for efficient reinforcement learning in geothermal well-control optimization. 

---
# PermitGPT: A Unified Generative-AI Pipeline for Construction Hazard Forecasting, Permit Prediction, and Community Impact 

**Authors**: Mohd Ruhul Ameen, Farjana Aktar, Akif Islam, Momen Khandoker Ope, Abu Saleh Musa Miah, Jungpil Shin  

**Link**: [PDF](https://arxiv.org/pdf/2608.28728)  

**Abstract**: Urban construction governance requires early decisions that connect workplace safety, permitting requirements, and community impact, yet the relevant evidence is often scattered across separate municipal and regulatory data sources. This paper presents PermitGPT, a unified generative artificial intelligence framework for converting unstructured construction permit descriptions into structured decision-support outputs across three domains: safety hazard identification, permit requirement specification, and community impact assessment. To address data fragmentation, we spatially and temporally align records from the New York City Department of Buildings, Occupational Safety and Health Administration, and NYC 311 service requests, producing 90,000 structured prompt-response pairs derived through rule-based alignment and domain-informed spot checking. We fine-tune three open-weight language models using parameter-efficient adaptation and evaluate them on 2,833 held-out test cases. The results show complementary model behavior: Gemma-3-1B provides the most efficient inference at 3.07 samples per second with low memory usage, Llama-3.2-3B gives the highest lexical overlap for regulatory-style outputs with a BLEU score of 0.0091, and 4-bit Mistral-7B-Instruct-v0.3 achieves the strongest semantic alignment with a BERTScore-F1 of 0.7747. Because the task involves open-ended structured generation, low BLEU values are interpreted alongside semantic metrics and qualitative output structure rather than as standalone indicators of utility. Overall, PermitGPT provides an initial step toward AI-assisted construction governance while identifying directions for stronger task-level evaluation and real-world validation. 

---
# Pro-Router: Token-Aware Progressive Model Routing with Adaptive Edge-Cloud Collaboration for Efficient Multimodal LLM Inference 

**Authors**: Xinyuan Gui, Shaowen Wang, Sheng Sun, Zijian Wang, Zishu Yu, Zheming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.28726)  

**Abstract**: The remarkable performance of multimodal large language models (MLLMs) comes at the cost of substantial computational overhead, posing significant challenges to real-time deployment and cost effectiveness. Existing model routing approaches either decide from coarse request-level features alone or spend one or several extra language model passes to inspect the generated response, leaving the token-level uncertainty signals that emerge during generation unused. To address these limitations, we propose Pro-Router, a token-aware progressive model routing method with adaptive edge-cloud collaboration for efficient multimodal LLM inference. Pro-Router employs a two-stage progressive decision mechanism. First, a lightweight prompt pre-scorer module performs rapid pre-screening before token generation begins, guiding apparently simple requests to small models. Second, a token-aware verifier reads the sampling probability distribution of each token the small model generates, estimating the model's confidence in its own output to determine, per request, whether the answer ships or escalates to the cloud-based high-precision model. Furthermore, we design an adaptive edge-cloud serving pipeline that sizes every dispatch to each device's measured service rate, so both the edge and the cloud tiers stay fully utilized without manual parameter tuning and are not impacted by the network latency. Extensive experiments on multiple multimodal benchmark datasets and models demonstrate the effectiveness of Pro-Router. Compared to other methods, it achieves the highest routing accuracy and improves routing speed by more than 10x. Its serving pipeline also reaches more than 75% higher end-to-end throughput than the existing model routing pipeline. Our code is available at this https URL. 

---
# Beyond the Answer Key: Robustness Evaluation of Large Language Models for Step-Level Mathematical Verification 

**Authors**: Fateme Mazdarani, Carlos Toxtli  

**Link**: [PDF](https://arxiv.org/pdf/2608.28725)  

**Abstract**: Large language models (LLMs) are increasingly used as graders, verifiers, and process auditors, but most mathematical evaluations still emphasize final-answer accuracy. This can obscure whether a model can verify a non-canonical but valid solution trace. We introduce a controlled linear-equation benchmark for evaluating LLMs in the evaluator role. Each instance asks the model to judge final-answer correctness, step-level trace correctness, and the first incorrect step. Our evaluation of state-of-the-art open LLMs reveals a significant robustness gap: models that accurately evaluate canonical solutions often fail when presented with perturbed but logically equivalent variants. Across GPT-OSS 20B, Qwen3-14B, and Phi-4-Reasoning, base models perform well on canonical traces but degrade substantially on perturbed traces, especially for error localization. On valid perturbed traces, base-model false-rejection rates reach 75.6-85.3%, showing strong sensitivity to canonical solution form. Supervised fine-tuning, distillation, and test-time compute improve robustness in some settings, but gains are model dependent and can trade off against canonical performance. The results show that reliable process-level verification remains challenging, and evaluator robustness should be measured separately from solver accuracy, even in a simple algebraic domain with exact ground truth. 

---
# ORDDAR: Observation-Driven Reasoning for Distortion-Resilient Decision, Action, and Cognitive Recovery 

**Authors**: Deblina Kar, Anant Nawalgaria, Shyamal Kumar Das Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2608.28704)  

**Abstract**: AI agents increasingly perform long-term reasoning, planning, tool use, memory integration, and autonomous decision making, yet erroneous intermediate states can propagate and cause inconsistent decisions and unreliable outputs. Existing reasoning approaches mainly rely on iterative planning, self-reflection, augmented memory, or verification, but rarely localize and selectively repair faulty reasoning. We present ORDDAR (Observation-Driven Reasoning for Distortion-Resilient Decision, Action, and Cognitive Recovery), a reasoning framework that models reasoning as cognitive state transitions, detects localized distortions, retrieves related reasoning from prior experiences, and repairs only the affected states. ORDDAR therefore performs recovery at the local reasoning-transition level rather than regenerating the complete trajectory. Experiments across mathematical, commonsense, multi-hop, and clinical reasoning benchmarks demonstrate improved reasoning quality, recovery ability, and interpretability over multiple evaluated reasoning baselines. 

---
# FRAC-MAS: A Safe and Explainable Multi-Agent System for Fracture Diagnosis 

**Authors**: Hardik Iyer, Tirath Bhathawala, Mihir Panchal, Ying-Jung Chen, Kiran Bhowmick, Pankaj Sonawane, Meera Narvekar  

**Link**: [PDF](https://arxiv.org/pdf/2608.28662)  

**Abstract**: Fracture detection and its clinical interpretability see notable improvements when deep vision models are integrated with agentic AI architectures. While deep learning models achieve high diagnostic performance, their black-box nature limits clinical adoption. We propose FRAC-MAS, an agentic AI system for automated, explainable, and safe bone fracture detection. The framework combines a stacked ensemble of four vision models with conformal prediction to produce statistically grounded differential diagnoses, while a multi-agent workflow performs independent verification, retrieves clinical guidelines, and generates patient-friendly reports. A pipeline-depth ablation study confirms that our multi-agent critic triages 86.6% of cases into a high-confidence auto-confirmed cohort while escalating uncertain cases, outperforming a single-agent baseline. Patient preference studies against Llama, MedGemma, and Gemini further demonstrate significantly more comprehensible clinical reports. These results suggest that integrating multi-agent critics with conformal guarantees enables safer radiology triage while preserving clinician oversight. More broadly, FRAC-MAS demonstrates how cooperative agentic architectures can serve as auditable, human-in-the-loop decision support systems for safety-critical healthcare. Our code is available at this https URL, and the website is available at this https URL. 

---
# A Generalized Optimization Engine (GOE) for Edge AI Inference Acceleration 

**Authors**: Venkat R. Dasari, Jakob A. Adams, Vinod K. Mishra, Brian Jalaian  

**Link**: [PDF](https://arxiv.org/pdf/2608.28652)  

**Abstract**: Artificial intelligence (AI) models have demonstrated remarkable capabilities across various domains, yet their widespread deployment is impeded by significant computational costs, particularly on resource-constrained devices. This paper explores the theoretical underpinnings of various AI model optimization techniques, algorithms, and abstractions, discussing their potential to reduce computational complexity, memory footprint, latency, and power consumption. Furthermore, we propose a comprehensive hardware (HW) and model-agnostic generalized optimization architecture that integrates these techniques for improved efficiency. Our study underscores the critical role of such a generalized optimization system in preparing model deployment over resource-constrained heterogeneous hardware in a tactical environment. As a concrete demonstration, we show that GOE-compressed language models deploy and run on a GPU-less edge CPU, and that the choice of compression method, not merely its nominal bit-width, determines whether task accuracy survives deployment. 

---
# How Language Models Choose Sides: Internal Representations of Instruction Hierarchy 

**Authors**: Enrique Balp-Straffon, Chih-Hao Hsu, Rushiraj Gadhvi, Sunishchal Dev, Callum Stuart McDougall, Anusha Mujumdar  

**Link**: [PDF](https://arxiv.org/pdf/2608.28648)  

**Abstract**: We study how instruction-tuned LLMs arbitrate direct conflicts between system and user instructions. We introduce a benchmark of 41 paired constraints with deterministic verifiers and evaluate eight models under matched baseline, conflict, and same-channel control conditions. Behaviourally, the models split into three regimes by System Authority Delta: hierarchy-respecting models use the system channel as an authority signal, anti-hierarchy models follow the system less often than their same-channel baseline predicts, and no-effect models show little channel sensitivity. Llama-3.1-8B is the strongest anti-hierarchy case in our suite, following the system in only 0.10 of conflict trials. We use this behavioural failure case to ask whether user-preferring arbitration reflects the absence of an internal conflictresolution signal. It does not: on Llama-3.1-8B, the conflict outcome is linearly decodable from residual-stream activations at 0.97 balanced accuracy, 17 percentage points above a metadata-only baseline, with analogous signals on Qwen2.5-7B and gpt-oss-20b. Steering with a layer-12 mean of four per-conflict logistic-regression directions raises genuine system compliance from 0.132 to 0.530, while directions selected mainly for pooled separability steer poorly. User-preferring conflict resolution can therefore coexist with a readable internal arbitration signal, and successful intervention depends on the geometry of the readout rather than probe accuracy alone 

---
# Self-Specialized Teachers for Domain Post-Training 

**Authors**: Yifei Li, Rongman Xu, Lingling Zhang, Muye Huang, Zihan Ma, Jiashuai Liu, Hang Yan, Heng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.28647)  

**Abstract**: Target-only post-training can improve performance in a specialized domain while degrading behaviors that a general-purpose base model acquired before adaptation. We study this problem when target-domain data are available but a representative replay corpus is not. We propose self-specialized teacher distillation (SSTD), a two-stage procedure that first trains a copy of the base model into a domain teacher, then distills its token distribution to a student on prefixes sampled from the student itself. Teacher training combines standard target supervision with base-aware key-token weighting and distribution alignment to the frozen base model; on-policy distillation then places domain feedback on states the student can encounter at inference time. On financial numerical reasoning, medical question answering, and legal holding identification, SSTD retains much of the target improvement of direct fine-tuning while improving the mean score on the evaluated general suite by 4.8--5.0 points at the reported operating point. The pattern persists across Qwen3 sizes and on Gemma backbones. SSTD requires neither an external teacher nor general replay data. 

---
# BiasMix-Finance: Post-Generation KYC Guardrails for LLM Portfolio Advice 

**Authors**: Gaurav Kukreja, Parul Kukreja, Mohammed Abraar, Raj Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2608.28646)  

**Abstract**: Large language models (LLMs) can generate plausible-sounding ETF portfolios while silently violating basic KYC-style constraints on risk, fees, and diversification. This is especially problematic in agentic multi-turn advisory systems, where each draft recommendation can become an action unless guarded by an auditable enforcement layer. We study a model-agnostic, asset-agnostic post-generation guardrail pipeline: (i) enforce a strict JSON allocation schema, (ii) validate allocations against numeric caps, and (iii) when violations occur, deterministically project the output to the nearest feasible portfolio via a convex quadratic program (QCQP). We introduce BiasMix-Finance (Mini), a compact stress-test benchmark for constrained decision-making under biased LLM generations, with a 16-ETF universe, three investor profiles, and eight bias prompts. Across three models and three inference modes (direct, critique, self-consistency), first-pass generations violate at least one cap in 47.6-85.7% of test cases (67.2% pooled), but the convex projection layer reduces final feasibility violations to 0% while requiring only a small correction distance (test pooled median D=||w*-w0||_2=0.066), indicating that the guardrail typically preserves the intent of the original allocation. We report violation rates and correction distances with confidence intervals, and paired model comparisons with multiple-testing correction. To support reproducibility, we release the dataset, prompts, caps, and code in our public GitHub repository. 

---
# From Extraction to Governed Memory: Multi-Agent Knowledge Graph Construction with Domain-Expert Review 

**Authors**: Pranav Bykampadi, Neel Mokaria, Vishesh Narayan, Faizan Wajid, Ashok Agrawala  

**Link**: [PDF](https://arxiv.org/pdf/2608.28642)  

**Abstract**: Knowledge graphs used by agentic systems are often treated as flat stores of extracted triples, with little record of who owns a fact, why it was admitted, or how it should be used downstream. We argue that reliable agentic knowledge systems require governance as an essential component of graph construction to bridge this gap. We propose MAGG, a principled multi-agent framework for constructing Governed Knowledge Graphs that introduces explicit governance decisions for reliable and trustworthy knowledge sharing. A domain classifier first induces entity and relation types directly from document content, enabling operation in open-world settings without fixed schemas. Candidate triples are assigned to domain owners, reviewed against supporting evidence, admitted through governance decisions, and stored with audit metadata. The same ownership structure is reused during question answering, where queries are routed to domain-specific graph experts rather than answered through undifferentiated retrieval. Our evaluation demonstrates MAGG's effectiveness: On SciERC, MAGG improves strict triple F1 by 47% and mapped triple F1 by 51% over flat insertion. A blinded review of 120 triples finds governed-only triples more often source-supported than flat-only ones, and revised triples supported in 100% of cases. Finally, on MuSiQue, MAGG outperforms Microsoft GraphRAG by 9.0 exact-match points and 11.2 token-F1 points. 

---
# Reward-Oracle MCTS for Formal Theorem Proving: Sample-Efficient Search and the Need for Kernel-Level Proof Auditing 

**Authors**: Bodla Krishna Vamshi, Haizhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.28639)  

**Abstract**: Formal theorem proving with large language models remains challenging due to the difficulty of navigating large proof search spaces efficiently. Existing tree search approaches either feed verbose compiler error messages directly into the generation context, increasing context usage during search, or employ non-standard evaluation protocols that prevent direct comparison with established baselines. We propose a three-role Monte Carlo Tree Search (MCTS) framework that treats the Lean 4 compiler purely as a reward oracle using compiler output as a scalar signal for UCB-guided tree updates without feeding error content into the generation context. Our framework decomposes proof search into three roles: a generator for proof attempts, a decomposer for subgoal decomposition, and a critic for subgoal quality evaluation. We evaluate across 4 benchmarks spanning competition mathematics and physics (MiniF2F, PutnamBench, LeanPhysBench, PhysLeandata) with three prover models at standard proof attempt budgets (PAB@16 to PAB@256). Our method achieves 87.1\% on MiniF2F with Goedel-Prover-V2-8B at PAB@256 and solves 26/659 PutnamBench problems at PAB@32 surpassing base sampling 18/659 at same proof attempt budget. Through an exhaustive axiom-level audit of every compiled proof, we further identify reward hacking in search-based theorem proving: DeepSeek-Prover-V2-7B produces proofs on PutnamBench that pass compilation and the standard sorry-token scan while depending on sorryAx. The audit removes 4 and 8 such proofs from whole-proof sampling at PAB@32 and PAB@128, and 11 and 19 from MCTS. We do not attribute these counts to the search procedure; we report them to establish that kernel-level auditing is necessary for compiler-verified evaluation. 

---
# Self-Evolving Skills via Surrogate-Guided Solve-and-Reproduce 

**Authors**: Jiale Liu, Pinze Ren, Yuqi Xia, Huan Wang, Zhenlin Zhao, Siming Dong  

**Link**: [PDF](https://arxiv.org/pdf/2608.28638)  

**Abstract**: Agent skills are portable packages of instructions and resources an agent consults at deployment. Self-evolving them fails in two ways today. First, skills evolved from scratch underperform human-curated ones and, on a weak model, using no skill at all. Second, an evolution-time pass records one lucky trajectory that a fresh stochastic agent often fails to reproduce at deployment. We present reSolve, a per-task, oracle-in-the-loop framework built on three components. It decouples interactive solving from a self-contained deliverable that is independently re-executed in a fresh container, a protocol we call solve-and-reproduce. It enhances the sparse reward signal with a surrogate verifier that cannot access hidden tests or reference answers. It then runs verifier-guided beam search over a solution-construction graph. Within a fixed harness, a cheap model self-evolves skills that reach $74.9\%$ mean-of-3, $+14.8$ points over the $60.1\%$ human-curated baseline, exceeding the strongest official curated-skill result ($67.3\%$, GPT-5.5/OpenHands). We also report observed failure cases and domain-level results, including performance on the 14 Natural Science tasks, to clarify when the approach does and does not help. 

---
# AI Scientist Mission Control (AIMC): Visual Analytics for Human Oversight of Autonomous Scientific Discovery 

**Authors**: Rikathi Pal, Klaus Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2608.28637)  

**Abstract**: Autonomous scientific discovery systems can generate large numbers of research ideas, experiments, and manuscripts with minimal human intervention. As these systems become increasingly capable, scientists require effective mechanisms to monitor output quality, identify recurring failure modes, understand research evolution, and prioritize promising discoveries for review. We present AIMC, a visual analytics framework for human oversight of autonomous scientific discovery. AIMC combines semantic embeddings, automated weakness extraction, temporal analysis, and interactive visualizations to support the exploration of AI-generated research artifacts. We demonstrate the framework through a case study of the papers generated by an autonomous AI Scientist (FARS), together with their associated review feedback. Our analysis reveals recurring methodological weaknesses, evolving research themes, domain-specific differences in quality, and a small set of highly novel papers that warrant deeper human inspection. These findings illustrate how visual analytics can support transparency, diagnosis, and human AI collaboration in emerging autonomous scientific discovery workflows. 

---
# AutoScientist-Quant: Self-Evolving Coding Agents for Automatic Research in Quantitative Investment 

**Authors**: Zongqian Li, Yaoyiran Li, Yaohui Guo, Ming Zhang, Nigel Collier, Eugene Ie  

**Link**: [PDF](https://arxiv.org/pdf/2608.28632)  

**Abstract**: Large language model agents can discover alphas, yet current methods have three weaknesses. The search cannot adapt during the run, automation usually ends at alpha generation while library selection and model choice stay manual, and alpha discovery can read the test window through loop feedback or code problems. We present AutoScientist-Quant, a self evolving search process that regards quantitative research as one budgeted search problem. A single controller conditions every decision on the remaining budget, choosing at each round whether to improve, combine, pivot, or stop, which node to expand, how many alphas to generate, and how to retrieve past trajectories from the shared memory. The same core then selects from the library and tunes the model, closing the loop from hypothesis to deployable strategy. We also review the evaluation pipeline reused from prior work, fix two lookahead problems, and keep the feedback window disjoint from the held out test window, so every comparison tests true generalization. On CSI universes, the framework attains the best value of nearly every metric in every setting, and these conclusions hold across several backbones and markets. 

---
# CrossAudit: A Git-Native, Cross-Vendor Audit Loop for Agentic Science 

**Authors**: Zhaohe Dong, Yuhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.28631)  

**Abstract**: An AI scientist should not grade its own homework. Yet in the systems we examined, the agent that reviews the work usually comes from the same model family as the agent that produced it, or at least from the same vendor. Model evaluators are known to favour their own generations. Whether models trained alike also share blind spots is a conjecture, not a settled finding, but if they do, the reviewer inherits the author's. The record of what was flagged and what was waved through often sits in platform logs that nobody outside can replay.
We present CrossAudit, a protocol for supervising autonomous research pipelines. It rests on three commitments. Each increment of work is audited by an agent from a different vendor against a rulebook a human wrote and versioned. Reports, verdicts, disputes and rulings are git commits, so the supervision history can be re-read and cited; raw model exchanges are not yet part of that record. Scripted checks run before any model does. Advisory judgement never gates the pipeline: a model blocks only by citing a rule, and no model may waive a deterministic failure. Blockers that survive a bounded number of revision rounds go to a person.
We state the protocol as eight invariants. We describe a reference implementation built from GitHub Actions and a few hundred lines of Python, and report a live deployment of a closely related variant in a computational-chemistry pipeline. We also ran a seeded-defect trial (30 increments, 43 seeded defects, one run per configuration). A cross-vendor audit of our own repository then voided its blinding. We adopt that audit's findings and report the corrected results. The trial shows that two vendors read the same rulebook differently. It does not show that either is better. The strongest evidence here is the committed, uncontrolled record of cross-vendor audits of this paper itself. 

---
# CDEP Agent: Connecting Meteorologically Detected Temporal Compound Events to Real-World Documentary Evidence 

**Authors**: Zhuoran Li, Weiyi Kong, Boer Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.28628)  

**Abstract**: Compound drought-to-extreme-precipitation (CDEP) events are recognized in climate science as a growing driver of extreme impact, but whether this recognition carries over into real-world early warning and post-event documentation is unknown, so a meteorologically real CDEP event may pass with neither advance warning nor any later record. Here we present CDEP Agent, an auditable LLM-agent framework that tests this mismatch directly by linking CDEP candidates detected from meteorological reanalysis to real-world hazard and impact evidence across sources with different spatial scales, temporal resolutions, and reporting conventions. Using California as a case study, we identify 408 candidate CDEP events from ERA5 observations during 2021-2025 and evaluate each against the U.S. Drought Monitor, NOAA Storm Events, and public webpages along five dimensions: antecedent drought, extreme rainfall, local impact, hazard-impact attribution, and explicit drought-to-rainfall linkage. Only 34.3% of candidates are corroborated on both hazard components, and just 1.5% are ever explicitly linked to their antecedent drought, indicating that most meteorologically detected CDEP events go undocumented and their compound nature almost never enters the record at all. Our framework gives climate scientists a way to test physical event definitions against what actually gets documented, and gives social scientists, economists, and disaster-response agencies a provenance-linked evidence base for compound events that current warning and reporting systems largely fail to capture. 

---
# Machine Learning-Enhanced Tabu Search for Tactical Wireless Network Design 

**Authors**: Wissem Ahmed Zaid, Alain Hertz, Defeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28627)  

**Abstract**: Designing high-performance tactical wireless networks under realistic operational constraints gives rise to challenging combinatorial optimization problems, where the evaluation of candidate solutions relies on detailed physical and traffic-aware models. Although classical metaheuristics such as Tabu Search offer effective mechanisms for exploring large search spaces, their computational cost remains high because numerous candidate moves must be evaluated at every iteration. In this paper, we propose a data-driven framework that improves the efficiency of Tabu Search by learning to guide its move selection process. Rather than altering the neighborhood structure, our approach exploits the information contained in the search trajectories generated during the optimization process. At each iteration, we record both improving and non-improving edge-based transformations together with a set of descriptive features capturing the structural, geometric, and performance characteristics of the network. This information is used to train a Graph Neural Network (GNN) that predicts the impact of candidate moves on the objective function. The trained model is then integrated into the Tabu Search algorithm to rank candidate transformations according to their predicted quality, thereby reducing the number of costly objective evaluations while maintaining an effective exploration of the search space. Experimental results on synthetic benchmark instances demonstrate that the proposed learning-assisted Tabu Search notably reduces computation time while consistently producing higher-quality solutions than the standard algorithm. These findings highlight the potential of combining machine learning with metaheuristics by leveraging the implicit knowledge embedded in search trajectories, paving the way for more efficient solution methods for large-scale network design problems. 

---
# Preference Elicitation for Policy Optimization and Application to Aligning Heart Transplantation with Human Values 

**Authors**: Itai Zilberstein, Ioannis Anagnostides, Zachary W Sollie, Arman Kilic, Tuomas Sandholm  

**Link**: [PDF](https://arxiv.org/pdf/2608.28620)  

**Abstract**: Preference elicitation is essential for aligning AI systems with human values. Prior approaches (e.g., for organ allocation) often ask stakeholders to compare the decisions of an algorithm (e.g., patient A vs. patient B). Such a decision-level approach conflates the means with the ends. Instead, we elicit preferences directly over allocation outcomes to learn a utility function for policy optimization. We construct a novel preference elicitation algorithm for linear utilities that outperforms prior techniques in practice. Our algorithm has two phases. The first phase learns cutting planes through pairwise comparisons to rapidly shrink the space of possible attribute weights and warm-starts the second phase by eliminating dominated regions. The second phase then provably converges to the user's utility function. We apply our technique to heart transplant allocation where a policy must balance competing objectives such as post-transplant outcomes, waitlist mortality, geographic ease, and equity. Using our algorithm, we conduct a user study to learn and aggregate a community-aligned utility function, and use it to optimize heart transplant policies that are significantly better aligned with human values. Compared to the hindsight optimum, the status quo policy achieves a competitive ratio of just 0.54, while our method is near-optimal with a competitive ratio of 0.95. 

---
# InternReviewer & InternAdvocate: Objective Reward and Evaluation for Agentic Reinforcement Learning in Peer Review and Rebuttal 

**Authors**: Xuerui Su, Liya Guo, Qizhi Pei, Qipeng Guo, Zhongbo Tian, Lijun Wu, Kai Chen, Zun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.28612)  

**Abstract**: Generating professional scholarly content, such as peer reviews and rebuttals, requires an intricate synergy between domain reasoning and factual grounding. This work presents a comprehensive framework for the development and evaluation of specialized scholarly agents, InternReviewer and InternAdvocate. We first establish a large-scale, high-quality scholarly dataset and integrate a high-efficiency arXiv retrieval tool to enable active evidence gathering. To optimize these agents, we implement an agentic Reinforcement Learning (RL) paradigm driven by a unified objective metric and reward system. This system avoids the biases of subjective model-based judging by employing multi-dimensional criteria, including reference-anchored semantic alignment, structural compliance, and a strict verification mechanism that cross-checks citations against real-time interaction logs to eliminate hallucinations. Experimental results demonstrate that agents trained within this closed-loop framework exhibit significant improvements in reasoning depth and citation accuracy. 

---
# TPvG: A Moral Decision Framework for Large Language Models from One-Shot to Sequential Feedback 

**Authors**: Fangyuan Zhang, Dong Yu, Pengyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28610)  

**Abstract**: Existing LLM moral evaluations typically present models with isolated moral vignettes and elicit a single-shot decision, neglecting a factor known to profoundly influence human moral behavior: consequence feedback. We introduce TPvG (Text-based Pain-versus-Gain), adapted from a human moral paradigm, which embeds consequence feedback into an everyday moral dilemma of not harming others versus maximising self-gain. TPvG comprises five moral decision tasks, progressing from minimal-context one-shot choices to sequential decisions with explicit consequence feedback. Our results show that LLM moral decisions were strongly affected by decision format (one-shot versus sequential), and explicit receiver feedback produced heterogeneous effects across models. Furthermore, LLM responses to explicit receiver feedback diverged from the human reference pattern, suggesting potentially different decision processes. These findings highlight the need to evaluate whether LLM moral behavior remains stable in high-stakes interactive settings. 

---
# RegDivergence-101: An LLM Benchmark for Cross-Jurisdiction Regulatory Contradiction Detection in Life Sciences 

**Authors**: Chuchu Wu, Zhiyin Zhou, Jingzhuo Hu, Liang You  

**Link**: [PDF](https://arxiv.org/pdf/2608.28607)  

**Abstract**: Pharmaceutical sponsors developing a drug for both the United States and the European Union must reconcile guidance issued independently by the FDA and the EMA. Where the two agencies require substantively the same thing, a sponsor can file once; where they diverge, a single trial design risks rejection in one region; where one agency is silent on a point the other regulates, the sponsor must infer obligations. Today this reconciliation is performed manually by regulatory-affairs experts. We introduce cross-jurisdiction regulatory divergence detection: given an FDA requirement and an EMA requirement on the same topic, classify their relationship as AGREE, DIVERGE, or SILENT. SILENT is inherently directional (SILENT_FDA vs. SILENT_EMA); we record direction per pair and report per-direction F1 alongside the collapsed label. We release RegDivergence-101, a 101-pair expert-grounded pilot evaluation benchmark (labels grounded in three peer-reviewed FDA/EMA comparison studies and primary FDA/EMA/ICH guidance text; dual-annotation inter-annotator kappa = 0.85), and systematically characterise a four-method baseline hierarchy: lexical heuristic (0.511 macro-F1, 95% CI [0.411-0.605]), NLI cross-encoder (0.233), obligation-level Graph-RAG (0.663 [0.570-0.747]), and flat LLM judge / Claude Haiku (0.830 [0.747-0.908]). Three directional observations emerge at pilot scale (n = 101): SILENT is semantically detectable but invisible to entailment-only formulations; pair-level obligation graphs improve over lexical methods but trail flat-LLM context (CIs partially overlapping); and corpus-level graph construction is the indicated architectural target for large-scale silent-detection. RegDivergence-101 is a pilot release establishing the task formulation and baseline hierarchy; four unrepresented regulatory domains and an expansion roadmap are described in Section 7. 

---
# MedTVL: Harnessing Vision and Language for Medical Time Series Classification 

**Authors**: Jiexia Ye, Jia Li, Fugee Tsung  

**Link**: [PDF](https://arxiv.org/pdf/2608.28605)  

**Abstract**: Recent advancements in multimodal learning for medical time series (MedTS) classification highlight the benefits of integrating complementary modalities for clinical decision. However, existing methods typically focus on bi-modal interactions (e.g., time series and text), leaving the tri-modal synergy between time series, vision, and language largely unexplored. Inspired by diagnostic practice synergizing numerical assessment, visual inspection and clinical context, we introduce MedTVL, a text-guided dual-pathway architecture tailored for MedTS classification. Specifically, it synergizes a convolution-based temporal pathway for fine-grained temporal dynamics from raw numerical sequences and a transformer-based visual pathway for holistic morphological structures from time-series-derived images. Such combination of cross-modal and architectural heterogeneity provides a comprehensive diagnostic perspective. To further resolve potential diagnostic ambiguity, both pathways are guided by adaptive medical textual semantics. Finally, a Mixture-of-Experts mechanism dynamically routes each instance to specialized fusion experts, capturing instance-specific reliance on the temporal and visual pathway outputs. In addition, MedTVL supports multimodal contrastive learning to mitigate the clinical label scarcity challenge. Extensive experiments across multiple medical datasets and tasks, spanning supervised, few-shot, and contrastive learning settings, demonstrate the superiority and transferability of MedTVL, highlighting its potential for robust clinical decision support. 

---
# C3-UniMM: Causal Cycle-Consistent Unified Multimodal Modeling via Super Alignment and Shared Decoding Space 

**Authors**: Yujie Shen, Lianlei Shan  

**Link**: [PDF](https://arxiv.org/pdf/2608.28603)  

**Abstract**: Unified Multimodal Models aim to achieve any-to-any understanding and generation across arbitrary modalities. However, existing methods primarily rely on modeling implicit statistical correlations and lack cross-modal structural consistency constraints. This deficiency leads to profound issues, including semantic drift, poor compositional generalization, and instability under interventions. In this paper, we propose C3-UniMM, a unified multimodal modeling framework based on Causal Cycle Consistency and Super Alignment. Specifically, we introduce a Structured Latent Causal Graph (SLCG) as a shared cross-modal semantic space and design unified multimodal encoding blocks, enabling understanding and generation to be synergistically optimized within the identical causal semantic structure. Furthermore, we propose a Unified Decoding Space to enforce structural preservation and semantic invertibility during the cross-modal generation process. Theoretical analyses demonstrate that our approach significantly enhances both the invertibility and mechanism invariance of cross-modal mappings. Extensive experimental results across multiple understanding, generation, and compositional generalization tasks indicate that C3-UniMM substantially outperforms existing unified multimodal baselines. 

---
# Integrating Triaxial IMU Sensors and Ensemble Learning for Effective Parkinson Disease Severity Classification 

**Authors**: Rehan Khan, Muhammad Junaid Asif, Rana Fayyaz Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2608.28602)  

**Abstract**: Parkinson disease PD is a progressive neurodegenerative disease that can have a significant impact on motor performance resulting in the appearance of symptoms such as tremors rigidity postural instabilities and bradykinesia. Timely clinical treatment disease management and quality life of the patients are closely linked to early and appropriate identification of PD. Over the past few years the growth of wearable sensor technology and artificial intelligence AI have made it possible to create noninvasive and data driven disease detection methods. This paper proposes a comparative system using artificial intelligence to detect Parkinsons disease by analyzing the motion and tremor data captured by an inertial measurement unit IMU. The data comprises the signals of the acceleration and gyroscope sensors measuring movement in three directions X Y and Z. The signs and symptoms provide helpful information about subtle motor deficits associated with PD. Several classification models like Support Vector Machine SVM Logistic Regression LR KNearest Neighbors KNN Decision Tree DT Extreme Gradient Boosting XGBoost and Light Gradient Boosting Machine LightGBM were used to compare their effectiveness. The Logistic Regression model had a performance around 75 percent in all evaluation metrics and KNearest Neighbours KNN around 90 percent. The support vector machine SVM performed almost 94 percent whereas the performance of classifiers such as Decision Tree and XGBoost was close to 96 percent and overall classification efficacy respectively. LightGBM model performs consistently at the best rank among all of the evaluated methods having Accuracy, Precision, Recall and F1score of around 97 percent. The results show that the proposed machine learning approach offers an accurate and effective predictive capability in the classification of PD severity. 

---
# Leveraging Generative AI to Design Accessible Interactive Visualizations for Undergraduate Mathematics: A Six-Phase Workflow 

**Authors**: Mahesh Sunkula, Kuan-Hua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.28601)  

**Abstract**: Interactive visualizations support conceptual understanding in undergraduate mathematics, but building them has required programming expertise most instructors lack. Using a design-based research approach, we develop, deploy, and evaluate a six-phase workflow (Foundation, Customization, Mathematical Depth, Application, Accessibility, Pedagogical Control) that uses generative AI to build WCAG~2.2 Level~AA compliant visualizations without programming. The six phases structure every prompt, scaffold the AI's code generation, and define where human verification is applied. We ask whether the structure reliably yields correct and accessible tools, whether it runs both backward (reverse-engineering prompts from a finished tool) and forward (generating a tool from a plain-language idea), and what verification each phase requires. Across four deployed tools spanning calculus, multivariable calculus, and differential equations, we evaluate mathematical correctness against closed forms, accessibility through automated and manual screen-reader testing, and the errors that recurred. The structure produces structurally complete first-pass tools, but human verification remains mandatory at every phase: each output must be checked for mathematical correctness, accessibility, and pedagogical fit before the next phase begins. The workflow is platform-independent and serves both instructors and students. 

---
# SHAPE of Chain-of-Thought in Math Reasoning 

**Authors**: Jonghyun Song, Sangjun Song, Minjae Oh, Haesung Pyun, Sungsik Lee, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2608.28600)  

**Abstract**: Large language models (LLMs) achieve strong performance on mathematical reasoning benchmarks, yet the mathematically meaningful skills underlying their reasoning remain underexplored. We introduce \texttt{SHAPE}, a framework that analyzes Chain-of-Thought (CoT) trajectories through two lenses developed in mathematics education: (1) semantic spaces: the model's evolving mathematical interpretations of a problem (e.g., algebraic, geometric), and (2) heuristics: the specific mathematical actions taken within those spaces (e.g., simplifying the problem, working backward). We first use \texttt{SHAPE} to analyze the reasoning patterns of various models. Our findings reveal that the mathematical heuristics employed by a model better explain final answer correctness than traditional CoT features. Furthermore, models are likely to reach correct solutions by concentrating their reasoning effort within a few semantic spaces rather than exploring many disparate ones -- a pattern consistent with human behavior. Next, we utilize the \texttt{SHAPE} lens to evaluate whether post-training truly enhances mathematical proficiency. We find that reinforcement learning induces mode-seeking in heuristic usage. Lastly, we post-train LLMs by promoting diverse heuristics and demonstrate its effectiveness in improving accuracy. Overall, \texttt{SHAPE} provides a theoretically-grounded diagnostic framework for decoding LLM reasoning and offers a new path toward post-training LLMs for math reasoning. The code for our model is available at this https URL 

---
# CDPR: Counterfactual Advantage-based Credit Assignment for Cost-Aware Sequential Medical Diagnosis 

**Authors**: Qi Peng, Yi Cai, Changmeng Zheng, Xin Wu, Jiayuan Xie, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.28599)  

**Abstract**: Clinical diagnosis is a step-by-step, cost-aware process: a physician orders examinations one at a time, observes the results, and updates the diagnosis before reaching a final conclusion. Most medical language models instead treat diagnosis as a one-pass classification task and ignore the trade-off between a test's value and its cost. We model diagnosis as a cost-aware sequential decision process and train the policy with reinforcement learning. The main difficulty is credit assignment: the only reliable signal comes once at the end of a long trajectory, so it scores a wasteful workup the same as an efficient one. We propose CDPR (Counterfactual Diagnostic Process Reward), which needs no expert labels and no learned critic. CDPR first finds the states where the policy hesitates, using the uncertainty of its action distribution, and then scores the chosen action by its advantage over the alternatives the policy itself would consider, estimated with short rollouts under a utility that balances correctness against test count, cost, and infeasible requests. A rollout cache reuses within-batch trajectories to keep the cost low. We integrate CDPR into GRPO and test it on one in-domain (MIMIC-IV) and two out-of-domain (ClinicalBench and a private hospital dataset) benchmarks. CDPR improves diagnostic accuracy while clearly reducing the number and cost of examinations. 

---
# The Race between Agentic AI Capabilities and Data Quality Control in Online Surveys 

**Authors**: Sourav Panda, Hillmer Chona, Rupak Kumar Das, Shreyash Kale, Shikha Soneji, Jonathan Dodge  

**Link**: [PDF](https://arxiv.org/pdf/2608.28597)  

**Abstract**: Online surveys are a foundational data collection instrument in a variety of fields, with attention checks serving as critical guardians of response quality. However, the rapid emergence of agentic AI (goal directed systems powered by a large language model (LLM) brain and/or a multimodal processing unit with tool-augmented capabilities) raises new questions about the robustness of these safeguards. We investigate how well agentic AI architectures can complete web-based surveys and pass standard attention checks. We evaluate a single-agent architecture capable of multimodal input processing and tool-based web interaction on a controlled survey sandbox. We analyze the problem from two perspectives. From an attack perspective, we demonstrate how structural vulnerabilities such as exposed DOM metadata and predictable option encoding allow agents to resolve attention checks through structured parsing only. From a defense perspective, we implement a mitigation strategy of DOM metadata obfuscation to remove semantic cues in text-based questions. We evaluate multiple open-source language and multimodal models to study capability and orchestration effectiveness. Based on our evaluations, we offer perspectives on how to simultaneously meet the needs of empiricists and agentic AI researchers. 

---
# Paper Pilot: A Human-in-the-Loop Expert System for Evidence-Traceable Scientific Manuscript Generation in Applied Sciences 

**Authors**: Nidhi Jha, Siddharth Chaudhary, Ajinkya Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2608.28596)  

**Abstract**: Large language model (LLM) agents are increasingly embedded in scientific workflows for literature analysis, drafting, and review. Existing systems advance autonomous discovery and manuscript generation, but do not resolve the governance problem that arises when ideas, methods, results, and claims propagate through AI-assisted workflows without mandatory human approval or artifact-level traceability. This paper proposes Paper Pilot, a human-in-the-loop expert system for evidence-traceable scientific manuscript generation in applied sciences. It adapts the Collaborative Agent Reasoning Engineering (CARE) methodology to manuscript development through manuscript-owner approval gates, explicit no-pass criteria, claim classification, audit logging, advisory LLM review, and evidence-locked revision control. The framework defines eight approval gates across the idea-to-claim pipeline and distinguishes literature-grounded from artifact-grounded claims, requiring reported numbers and interpretations to remain traceable to approved evidence; its system prompt is openly released for deployment in ChatGPT, Gemini, Claude, or institutional LLM environments. As a first empirical validation, we evaluate the citation-grounding layer with a controlled, mechanically scored benchmark (two commercial LLMs, real arXiv papers, no LLM judge): under coverage pressure ungated drafters fabricated up to 25% of their citations and never flagged an evidence gap, whereas the same models under Paper Pilot's evidence-locked rules produced zero fabricated citations and surfaced the planted gaps as explicit placeholders. Preliminary results for result grounding, revision, and adversarial robustness point the same way; full evaluation is left to future work. Paper Pilot positions LLM-assisted writing as a controlled human-AI decision-support process rather than a fully autonomous authorship pipeline. 

---
# The Signal in the Noise: An Auditable Reliability Layer for Biomedical Text Classification 

**Authors**: Moustafa Yehia Hassan, Sharon Wong, Woh Kai Xuan  

**Link**: [PDF](https://arxiv.org/pdf/2608.28595)  

**Abstract**: Biomedical NLP pipelines routinely presuppose clean input text, yet large-scale corpora assembled through automated PDF parsing harbour pervasive OCR-like artifacts, token splits and merges, hyphenation remnants, and character-level corruption, that systematically erode lexical evidence and degrade downstream classifiers. We introduce a conservative, fully auditable spell-correction reliability layer conceived as a safety-oriented preprocessing module rather than a maximal-accuracy corrector: under conditions of uncertainty, the system abstains from editing, in accordance with a medical do-no-harm philosophy. The deterministic architecture couples bounded edit-distance candidate generation with corpus-derived n-gram scoring and a suite of biomedical safety gates that protect domain-critical terminology. We evaluate the layer both intrinsically, on a manually curated benchmark of 2,104 token-level cases, and extrinsically, on a tri-class CORD-19 topic classifier (Prevention, Treatment, Epidemiology) spanning 10,000 examples under a principled four-run protocol (Clean, Noisy, Restored, Safety). Intrinsically, the layer attains 94.61% error-fix recall on synthetic errors with zero harmful edits on negative controls. Downstream, it recovers approximately 80.45% of the noise-induced macro-F1 degradation, elevating macro-F1 from 0.7654 (Noisy) to 0.7717 (Restored) while preserving near-clean performance (Safety: 0.7721). A supplementary case study on 103 real-world OCR-extracted abstracts classified with BioBERT confirms that transformer encoders appeared relatively robust to mild noise, motivating a future grey-box architecture that integrates bounded neural signals and UMLS lexicons without compromising auditability. The system is fully deterministic, artifact-driven, and designed with deployment and auditability in mind. 

---
# From Question-First to Analyst-First: Domain-Expert Skills and Verified Knowledge Compilation for Proactive Enterprise Analytics 

**Authors**: Harmohit Singh, Rahul Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2608.28594)  

**Abstract**: Conversational analytics systems assume the user already has a well-formed question, leaving a non-expert facing a blank query box on an unfamiliar enterprise schema. Commercial 'proactive' tools narrow this gap only by detecting statistical anomalies over analyst-curated metric layers, and academic next-question recommenders depend on query logs that a fresh dataset lacks. We describe a production analytics system that inverts the interaction model from question-first to analyst-first through two coupled architectural ideas. First, a pluggable domain-expert 'skill' abstraction: a folder-based, database-free subject-matter pack (a manifest, per-stage prompt facets, keyword-routed references, report templates, and optional compute) auto-selected per (client, dataset) by deterministic schema matching and spliced as a cross-cutting concern into every stage of an agentic pipeline, the schema explorer, and the report engines, degrading to a strict no-op when absent. Because a skill is a self-contained folder resolved deterministically, the catalogue is open-ended: an extensible marketplace of domain experts. Second, an offline knowledge-compilation loop: an agent probes the dataset's parquet via DuckDB (zero load on production), runs critic-gated per-table convergence with self-healing retries, and data-validates joins by value overlap, producing durable schema knowledge that drives standing expert reports whose every published metric is re-verified by re-executing its evidence SQL, plus suggested questions that mirror the report agenda. These close a proactive loop: reports surface numbers, the numbers seed questions, and a click launches a verified deep dive, all before the query box is used. We give a formal model and report illustrative single-tenant evidence. We make no user-study or benchmark claims; the contribution is the architecture and its defensibility. 

---
# Statutory AI: Aligning Large Language Models With Legal Norms 

**Authors**: Cindy Delage, Stéphane Canu, Marc Décombas, Jonathan Foureur  

**Link**: [PDF](https://arxiv.org/pdf/2608.28593)  

**Abstract**: With the increasing development of AI regulatory frameworks, ensuring that artificial intelligence systems, particularly generative models, operate in accordance with legal and ethical standards has become a critical priority. Existing proposals for AI alignment and value-guided behavior, however, face some limitations. Approaches such as Constitutional AI depend on human supervision, while broad normative frameworks like the Good-for-Humanity (GfH) principle may be overly general and ambiguous to provide actionable governance guidance. To overcome these limitations, we propose a hybrid approach called Statutory AI that employs pre-existing human-authored principles drawn from specific themes within a legal corpus. Specifically, Statutory AI uses legal texts as a constitutional framework, enabling AI systems to autonomously critique and revise their outputs according to established norms. It operates in two stages, both using Chain-of-Thought prompting. The first stage classifies the user prompt into one of the identified themes, while the second stage analyzes it in conjunction with relevant articles selected from the legal corpus of that theme. To illustrate the potential of our approach, we conducted an experiment involving 1,000 red-teaming prompts and five penal themes: discrimination, disclosure of confidential information, violence, fraud, and abuse of vulnerable persons. Statutory AI reduced harmful content by 52 to 59 percentage points across tested models, approximately 10 percentage points higher than standard Constitutional AI, while cutting computation time by over 50%. 

---
# A collective capability boundary in frontier large language models on guideline-conformant and case-specific oncology decision-making 

**Authors**: Zhang Sheng, Jinming Li, Wangyang Chen, Zhiwei Bao, Yu YoSean Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.28592)  

**Abstract**: Large language models (LLMs) achieve high scores on medical knowledge examinations, yet real-world oncology is not a knowledge test--it is a sequence of guideline-pathway choices, escalation judgments, and commitments under uncertainty. Existing benchmarks largely measure factual recall, leaving open whether frontier LLMs share decision-path blind spots that combining models cannot fix. We built the Oncology Decision Boundary Benchmark (ODBB)--2,005 oncology decision points across NCCN guidelines and colorectal cancer cases--and evaluated nine frontier LLMs (four closed-source, five open-weight families) released between June 2025 and April 2026. A fully deterministic scorer (zero LLM inference) classified outputs into 14 failure types, independently validated by two oncologists (Cohen's weighted $\kappa$ = 0.939 and 0.790) on a 225-item stratified sample. Treating the nine as a pooled super-model, 42.1% (Wilson 95% CI 40.0--44.3%) of all items--35.7% of the 1,586 NCCN items and 66.4% of the 419 colorectal-cancer cases--were answered correctly by none, with failures concentrated in choosing between guideline pathways before reasoning within any: a consistent blind spot in clinical meta-judgment that likely requires architectural intervention rather than more training data. Two models tuned for decisiveness (GPT-5.5, Gemini 3.1 Pro Preview) made unsafe commitments three to five times more often than the seven cautious models without scoring higher. In 3--9% of items, models stated the correct next clinical step yet did not commit to it--failures of decision, not knowledge. Model quality is no longer the primary bottleneck for clinical LLM deployment; the binding constraint is the assumption that any single model can be the sole basis for a clinical decision. Progress requires architectures that detect when a model reaches its competence boundary and route the decision to a clinician. 

---
# Expert-validated STEM QA 

**Authors**: Kihwan Han, Saurabh Patil, Chinmayee Shukla, Abhinav Sharma, Marko Pavlovic, Anshuman Lall, Mahesh Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2608.28591)  

**Abstract**: Recent advancements in AI are helping scientists achieve breakthroughs in fields such as mathematics, medicine, and materials sciences. New evaluation datasets for AI models contribute to such advancement in AI. In the STEM domain, frontier models have consumed most of the available online data, creating the need for human-created datasets that codify the knowledge of leading experts in the domain. There are several STEM datasets available for the research community in this field. However, there are some gaps in these datasets, leaving room for improvement. Examples of gaps include (1) saturation in model performance on these datasets, leaving no head-room for meaningful evaluations, (2) skewed taxonomy distributions, (3) multiple choice question format that is misaligned with how scientists use AI in the real world, and (4) inaccurate answers and rationales partially led by a contest-based data collection and a time-bound review process. In this study, we present 'Expert-validated STEM QA', a high-quality, expert-validated STEM dataset (N=398) in Physics, Chemistry, Biology, and Mathematics, created by 241 domain experts. We (1) carefully designed a taxonomy with balanced distribution, (2) vetted question contributors with quality-driven incentive, (3) conducted multiple rounds of reviews with revisions validated by domain experts based on consensus, and (4) created the dataset in verifiable question and answer format. Our study demonstrated low performance ($<25\%$) of frontier AI models on the dataset as a benchmark. Post-training on a separate, private version of the dataset (N=2,000) increased performance of the open source model by $15\%$ relative to the baseline model (p=0.045) on the STEM subset of HLE-verified dataset, indicating potential utility of the dataset for model training. We have open-sourced a portion of our dataset for the AI research community. 

---
# DS-Lighting: Making Agent Harnesses Explicit for Data-Science Automation 

**Authors**: Fan Liu, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28590)  

**Abstract**: Large Language Model (LLM) agents have shown promise for automating data-science workflows, yet their end-to-end performance depends critically on the agent harness that represents tasks, manages execution state, constrains output artifacts, and provides evaluation feedback. Existing data-science agents often leave this harness implicit, making results difficult to reproduce, compare, and attribute across heterogeneous tasks. We introduce DS-Lighting, a unified harness toolkit that makes harness design explicit for data-science automation. DS-Lighting decomposes the harness into four reusable layers: data, workflow, execution, and evaluation, and represents diverse agents as executable operator programs that support both predefined pipelines and adaptive search. We further integrate multiple open-source data-science benchmarks into an MLE-Bench-style task format, enabling controlled comparison under a shared task interface, sandboxed runtime, and metric protocol. Experiments across agents, harnesses, models, and ablations show that explicit harness design improves reproducibility, comparability, and reliability, while reducing avoidable system-level failures in end-to-end data-science workflows. Our code is available at this https URL 

---
# SUN: Persistent Programs For Language-Grounded Control-to-Learning-to-Real Policies 

**Authors**: Weiqi Wang, Zhi Li, Yudong Lei, David Martinez, Xiaofeng Gao, Yuxin Jiang, Chenfanfu Jiang, Yingnian Wu, Demetri Terzopoulos, Ran Gong  

**Link**: [PDF](https://arxiv.org/pdf/2608.31167)  

**Abstract**: Bridging model-based control and learned policies in long-horizon manipulation has harbored a silent disagreement: control executes specified objectives, learning amortizes that behavior into a reactive policy, yet existing protocols discard task semantics, leaving rewards hand-crafted and behavior drifting from what control this http URL introduce Semantically UNified (SUN) Programs, typed executables where geometric and contact relations are defined once and compiled into aligned Model Predictive Control (MPC) costs, satisfaction predicates, RL rewards, transition guards, and diagnostics. Our system, Kuafu, driven by large vision language systems, automatically synthesizes SUN Programs from language and scene semantics, screens feasibility via MPC, and retains semantics while training stage-conditioned policies. Across nine tasks, Kuafu achieves 82.03% macro-success, outperforming sparse-reward (35.67%) and Stage-BC (24.75%) baselines. At 8192-way scale, it generates 10.57x the successful trajectory time per hour of human teleoperation. With 500 trajectories per task, Kuafu data trains DP3 policies to 46.0% simulation success (vs. 22.4% for alternatives) and 34.7% on physical Franka and Kinova robots. These results establish that simulation-screened task semantics can effectively amortize control into robust policies, without demonstrations or manual dense rewards, unifying symbolic planning and data-driven execution. 

---
# Auditing Anonymous AI Models: A Four-Stage Protocol for Black-Box Identity Verification 

**Authors**: Yisen Xi  

**Link**: [PDF](https://arxiv.org/pdf/2608.31142)  

**Abstract**: The 2025--2026 AI market has seen a wave of stealth releases: frontier models launched anonymously on developer platforms under codenames. For their users, identity determines data-handling terms, supply-chain risk, and capability expectations. No validated methodology exists for black-box identity verification of anonymous models: practitioner checklists lack accuracy evidence, and self-identification is untrustworthy by design. We propose a four-stage forensic audit protocol for API-served models. Stage 0 reconstructs launch-time configuration from archived platform snapshots (Internet Archive), exposing preview--production drift. Stage 1 fingerprints configuration (context, output ceiling, reasoning, modality) against the platform catalog. Stage 2 tests tokenizer identity with a cross-length differential that rejects short-prompt collisions. Stage 3 corroborates with behavioral probes. We test declaration consistency on 10 known-identity releases (7 exact, 2 precision-differences, 1 partial, 0 counter-directional), not end-to-end identification under anonymity. Identification is validated prospectively on a flagship case whose 2026-08-23 analysis pointed to the GLM-5.3 version line and whose official reveal confirmed those family and version-line inferences (deployment variant was not pre-asserted; Flash was consistent post-reveal), and on three Stage-0-only cases where the protocol produced a graded hypothesis or declined rather than guessed. A standard-library-only implementation is provided as supplementary material. 

---
# LLM Post-Training as Brownfield Maintenance: An Industrial Perspective on Dataware Engineering 

**Authors**: Gopi Krishnan Rajbahadur, Amir M. Ebrahimi, Boyuan Chen, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2608.31102)  

**Abstract**: Industrial post-training is a brownfield regime. Teams inherit a deployed checkpoint and must land targeted improvements under fixed compute and mixture budgets without regressing the rest. The maintained artifact is increasingly dataware: behavior governed by a curated post-training mixture, updated via bounded mixture patches rather than clean-slate retraining. From an industrial code-generation improvement effort, we offer a maintainer's perspective on why this work is hard in practice, distilling three recurring challenges, zero-sum mixture design, yield as the binding metric, and end-to-end integration under uncertainty, and arguing that progress depends less on one-off recipes than on an engineering discipline for programming dataware. In our case study, interventions that raised the conversion of teacher distillation into usable training data increased accepted supervision by 2.84 times while using the same solution teacher and four solution attempts per candidate problem. In our primary evaluation, the yield-engineered patch improved CodeForces pass@1 by +2.59 points (+3.11 pass@3) and held-out LiveCodeBench v6 pass@1 by +6.11 (+8.05 pass@3), all statistically significant across 16 stochastic evaluations of each benchmark from one fixed checkpoint per condition, with internal AIME and MATH regression suites within tolerance. 

---
# Learning to Evaluate Before Improving: Automatic Rubric Induction for Automatic Research Agents 

**Authors**: Xuehai Wang, Haowei Qin, Tongxin Liu, Junkai Li, Buqiang Xu, Jintian Zhang, Yijun Chen, Zirui Xue, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2608.31076)  

**Abstract**: Autonomous scientific research agents are increasingly applied to end-to-end scientific workflows, including literature review, data analysis, experimentation, and report generation. However, open-ended research tasks often do not clearly specify the analyses, methods, and success criteria required to complete the task. As a result, agents may miss important analyses, use inappropriate methods, or draw conclusions that are insufficiently supported by evidence. To address the problem, we present AutoSciRub, an evaluation-first framework that induces a task-specific executable rubric before research execution, and uses it to guide execution, criterion-level verification as well as iterative revision. AutoSciRub decomposes an underspecified instruction into atomic scientific goals, grounds them in relevant literature and task-visible data, and synthesizes specific, actionable, and verifiable criteria. The resulting rubric makes implicit experimental and evidential requirements explicit, providing guidance for experiments and analyses. During revision, rubric-guided verification identifies unmet criteria and enables targeted refinement of the research report and its supporting artifacts. On ResearchClawBench, AutoSciRub consistently improves all tested configurations, with an average gain of 2.08 points across three backbone LLMs under the fixed Codex harness and 2.95 points across three agent harnesses using a fixed DeepSeek-V4-Flash backbone. On a randomly sampled 20-task subset of AstaBench E2E Discovery, AutoSciRub further achieves an average improvement of 16.8 points across three agent harnesses, while maintaining or increasing the number of successfully completed tasks. These results demonstrate that evaluation-first guidance provides an effective and generalizable control mechanism for autonomous scientific research (Code: this https URL). 

---
# Real-Time Video Anomaly Detection Using YOLO Pose Estimation and CLIP-Based Semantic Scoring 

**Authors**: Vanodhya G. Warnasooriya, Amir Hajian, Watchara Ruangsang, Supavadee Aramvith  

**Link**: [PDF](https://arxiv.org/pdf/2608.31074)  

**Abstract**: We propose a lightweight two-stage framework for real-time video anomaly detection. The first stage employs YOLO v11n-pose to detect persons and extract seventeen skeletal keypoints in a single forward pass. The second stage encodes each cropped person region through CLIP ViT-B/32 and computes cosine similarity against predefined textual descriptions of anomalous behaviors. This architecture eliminates the need for optical flow, standalone pose estimators, and density-based scoring modules. Experiments on CUHK Avenue, ShanghaiTech Campus, and a custom indoor dataset collected at Chulalongkorn University demonstrate an end-to-end throughput of approximately 51 FPS on an NVIDIA Titan XP GPU, a 3.36x speedup over the multi-feature baseline, while maintaining frame-level AUROC values of 89.26%, 70.26%, and 84.13%, respectively. 

---
# One note in three: a verified census of three deployed AI scribes, and the instrument that counted it 

**Authors**: Sebastian Fox, Luke Markham, Ryan Lail, Michael Karotsieris  

**Link**: [PDF](https://arxiv.org/pdf/2608.31017)  

**Abstract**: Ambient AI scribes draft clinical notes under the reassurance that a clinician signs every note. We audited three commercial AI scribes on the same 142 consultations: 565 notes from recorded UK primary-care and US ambulatory encounters plus authored scenarios. Twelve discovery passes proposed 13,678 candidate errors; the 5,898 clearing an importance filter went to an adversarial panel of two models from different families, each told to refute what it could, and 618 survived. One note in three (31.3% [27.0, 35.6]) carries a verified failure, concentrated in allergy and medication information, invented patient identity, and history written up as examination on telephone consultations that can contain none. No product was given a patient record; setting aside the two classes a record would have prefilled, invented identity and dates, the rate is 24.8% [20.8, 29.0]. One failure mode did not fit our scheme, drawn from published scribe-error taxonomies: a treatment the clinician retracts, recorded as delivered care. Two clinicians adjudicated blind, disjoint samples: a physician author upheld 20 of 21 findings (95.2% [77.3, 99.2]) and an independent clinician, not an author, 12 of 12 ([75.8, 100]); both judged every sampled refusal genuine. A failure rate depends on the instrument as much as the scribes. With model, evidence and settings fixed, the review instruction alone moves the share of candidates verified from 9.3% to 79.0%, and the reviewing family moves it too: alone at that instruction the gentler flags 54.8% of notes against 27.8%. Between 28% and 97% of sampled notes carry a failure depending on the standard. Published audits disagree among themselves by a margin instrument differences alone can produce: omission is 54-86% of their errors against our 23.1%. We release all 618 findings with transcript-side evidence, every prompt and model version, and the re-runnable pipeline. 

---
# LLM Judges Verify Presence, Not Absence: Omission Blindness in AI Clinical Notes and What Recovers It 

**Authors**: Sebastian Fox, Luke Markham, Ryan Lail, Michael Karotsieris  

**Link**: [PDF](https://arxiv.org/pdf/2608.31016)  

**Abstract**: Ambient AI scribes draft clinical notes, and published audits find their dominant error is omission: information the encounter established that the note fails to record. The standard check is an LLM judge: a second model reads the note against the transcript and flags problems. We ask whether judges detect omissions. Public corpora cannot supply the answer key: their clinician reference notes and transcripts are materially discrepant. Our benchmark has 500 single-error note pairs from audited fact sheets, 298 with a named fact certainly absent and 202 added-or-altered controls. Across eight judge designs, paired discrimination (the flawed note below its clean twin, 0.5 a coin flip) reads 0.79-0.94 on added or altered content and 0.50-0.63 on omissions. On single notes, no design flags omissions reliably more often than perfect notes. Wording changes, voting and GEPA prompt optimisation move the operating point without creating usable detection. Restructuring the task recovers it: list the facts the transcript establishes, then check the note for each. Two methods reach it independently and trade off: a per-fact pipeline, and a GEPA-evolved prompt doing the same in one call. The pipeline's flags name the missing fact and its severity at 2.7% false alarms. The single call detects more (36.9% against 24.6%, p=0.002) at 6.2% false alarms and a tenth of the cost per note. A physician author validated 70 items and, where the two routes disagree, sided with the pipeline on 10 of 10 (p=0.002). A second clinician, not an author, graded the severity rubric blind and agrees to within a grade. On real vendor notes from a companion census no benchmark threshold transfers, but the re-calibrated single call detects more than the best of the eight at half its false-alarm rate. Omissions whose fact is restated elsewhere defeat both routes. We release the benchmark, prompts and judgements. 

---
# Stick to What You Know: A Study of Knowledge-Aligned Supervised Fine-Tuning 

**Authors**: Arthur Becker, Jakob Kemmler, David Thulke, Christine Schäfer, Christian Dugast, Hermann Ney  

**Link**: [PDF](https://arxiv.org/pdf/2608.30987)  

**Abstract**: Supervised fine-tuning (SFT) trains a base language model to imitate target responses, and these targets may require knowledge the base model has not robustly internalized. We study this as a source of hallucinations and frame a group of mitigation methods as \emph{knowledge-aligned SFT}: constraining SFT training targets to the base model's parametric knowledge. Under a unified setup, we compare existing generation-based and estimation-based knowledge-alignment methods and introduce two new variants: Evidence Rewrite, which verifies base-model generations using external evidence, and Recall Rewrite, which retains claims only when they can be consistently recalled by the base model. Experiments with Qwen 3 4B and OLMo 3 7B show that knowledge-aligned SFT can reduce factual hallucinations on WildHalu and Biography while largely preserving general capabilities. Recall Rewrite yields the strongest factuality gains and improves refusal behavior on UnknownBench. It thereby confirms that SFT targets beyond the base model's knowledge drive hallucination behavior. 

---
# Evaluating and Improving LLM Self-Modeling 

**Authors**: Siqi Zeng, Andre N. Assis, Rowan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30980)  

**Abstract**: We study self-modeling: an LLM's ability to answer questions about its own behavior. We focus on verifiable behavioral questions, such as whether a prompt edit would change the model's final answer. To measure this capability, we introduce a benchmark that tests diverse types of self-modeling questions. Current models show non-trivial but limited self-modeling skill, and make systematic mistakes on simple counterfactual questions about their own behavior. To improve self-modeling skill, we develop a scalable synthetic-data pipeline that produces self-modeling training data, and show that reinforcement-learning can improve aggregate self-modeling skill across three open-source model families with some transfer to held-out tasks. These gains, however, do not seem to constitute introspection consistently: improved self-modeling may not arise from privileged access to the model's internal decision process. 

---
# MR-JEPA: A General Purpose Video Foundation Model for Cardiac MRI 

**Authors**: Athira J. Jacob, Puneet Sharma, Dorin Comaniciu, Daniel Rueckert  

**Link**: [PDF](https://arxiv.org/pdf/2608.30975)  

**Abstract**: Cardiac magnetic resonance imaging (CMR) produces rich sequential data such as temporal cine videos and spatial LGE/mapping stacks, yet most deep learning approaches process individual 2D slices, discarding this context. We present MR-JEPA, a self-supervised video foundation model for CMR that extends LeJEPA to 3D spatiotemporal inputs through tubelet tokenization, spatiotemporal masking augmentation, and initialization from a 2D CMR foundation model. Unlike prior CMR video models limited to cine data, MR-JEPA is pretrained on multi-sequence data (cine, LGE, mapping) from 10,505 patients across two centers without annotations. We evaluate the frozen encoder on six downstream tasks using a unified multi-view gated attention architecture: LV ejection fraction, RV ejection fraction, three myocardial strains (GLS, GCS, GRS), and four-class disease detection. MR-JEPA outperforms other compared methods on all five regression tasks, including both a domain-specific CMR model pretrained on more data with text supervision and a natural-video foundation model, achieving an LV EF MAE of 4.79% (r =0.764) and a GLS MAE of 1.87 (r=0.805), with 21-27% MAE reductions over baselines on strain tasks. For disease detection, MR-JEPA achieved a macro AUG of 0.868, remaining competitive with the domain-specific baseline despite using a fully self-supervised pretraining objective. These results demonstrate the potential of a unified video encoder for robust, multi-view utilization of diverse CMR sequences in clinical cardiac quantification and diagnosis. 

---
# CoJEPA: Combining Contrastive Learning and JEPA for Global-Local Music Representations 

**Authors**: Gabriel Meseguer-Brocal, Yuexuan Kong, Romain Hennequin  

**Link**: [PDF](https://arxiv.org/pdf/2608.30974)  

**Abstract**: Joint-Embedding Predictive Architecture (JEPA) has shown strong performance in learning rich representations through self-supervised prediction in latent space. However, it typically relies on teacher--student architecture with an EMA to stabilise training, and can tend to yield uninformative representations. Contrastive learning is stable to train and produces strong global representations, but remains limited on local tasks by the global nature of its objective. In this work, we combine both into CoJEPA: a single shared backbone jointly trained with a JEPA objective on masked sequence tokens and a contrastive objective on the class token. The contrastive gradient provides stability, removing the need for an EMA teacher entirely, while JEPA enriches the sequence tokens via local predictions that contrastive learning alone cannot provide. Crucially, no extra parameters are added to the backbone: the same model is guided towards richer representations purely through the design of its training signal. CoJEPA takes the best of both worlds, outperforming or matching both individual methods across global and local MIR tasks, with a particularly strong advantage on tonal and harmonic understanding, and without any task-specific architectural changes. CoJEPA shows that combining objectives with complementary inductive biases can substitute for scale, encouraging future work to invest in smarter training objectives over ever-larger models. 

---
# CogEvol: Towards Efficient and Reliable Learning Environment Generation 

**Authors**: Shangqing Tu, Daniel Zhang-Li, Yucheng Wang, Shiyu Gan, Yanpeng Wang, Huiqiang Rong, Mofei Chen, Shen Yang, Yini Chen, Yinuo Duan, Haoxuan Li, Binglin Liu, Ye He, Danqi Zheng, Zhanxin Hao, Yuxuan Wu, Mengting Tao, Yuqiu Liu, Jifan Yu, Juanzi Li, Bin Xu, Lei Hou, Huiqin Liu, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30968)  

**Abstract**: We present CogEvol, a family of models trained specifically for Learning Environment Generation: turning a course brief into a finished learning artifact (structured-JSON slides or self-contained interactive HTML pages) in a single pass. Across 220k production requests, CogEvol completes a slide in a median of 17 seconds and an interactive page in 59, replacing minutes-long multi-turn agent scaffolding. Reliability is enforced rather than hoped for: a production-grounded data pipeline turns real failures into 53,687 verified SFT samples, and a hybrid rule-plus-VLM reward drives GRPO-based RL, hardened after we caught and fixed a reward-hacking episode that produced visually convincing but unplayable games. CogEvol-27B scores 83.7 on slide quality and 63.7 on a 500-case interactive-HTML benchmark with 26.9x fewer parameters than flagship coding models, and, in collaboration with the OpenMAIC team, serves their live production traffic. CogEvol-4B is released openly under the Apache 2.0 license at this https URL external flagships are measured on the same suites under the identical harness. Scaffold editing cuts interactive-page generation cost by a further ~76%, and the full stack runs on domestic Ascend accelerators at application-level parity with A800 GPUs, lowering the unit cost of AI-native education at scale. 

---
# A Universal Context-Reuse Layer for Cross-Model KV Sharing 

**Authors**: Yi Li, Dongming Jiang, Yi Zhao, Bingzhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.30963)  

**Abstract**: Modern large language model (LLM) serving systems increasingly operate over repeated or shared context, yet each model typically performs its own prefill computation even when another model has already processed the same input. Existing KV-cache reuse mechanisms substantially reduce redundant computation within a single model, but generally assume that the producer and consumer of a cache are identical. We study \emph{cross-model KV sharing}, which translates the KV state produced by a source model into a representation that can be consumed by a different target model, including models that differ in scale, architecture, attention configuration, tokenizer, and model family. We evaluate the approach in both within-family and cross-family settings. For Qwen2.5-7B $\rightarrow$ Qwen2.5-1.5B, translated KV states improve LongBench2 accuracy from 27.59\% to 34.48\%, a gain of 6.89 percentage points over the native 1.5B baseline, while reducing handoff cost relative to native target prefill. For the cross-family Qwen2.5-1.5B $\rightarrow$ Gemma-2-2B setting, KV handoff reduces target-side prefill cost by up to 67.05\% at 4K context length while maintaining decoding perplexity close to native-model baselines. In a more heterogeneous Llama3.1-70B $\rightarrow$ Qwen2.5-7B setting, cross-family handoff achieves 44.0\% accuracy compared with 45.7\% for native Qwen2.5-7B inference, while reducing measured latency from 899ms to 138ms. These results provide initial evidence that KV states can serve as transferable computational representations rather than strictly model-local caches, and motivate \emph{context mobility} as a systems abstraction for reducing redundant prefill across heterogeneous LLM and multi-agent inference workflows. 

---
# LOCI: A Locator-Critic with Refinement Loop 

**Authors**: Walid Bousselham, Mathilde Caron, Arsha Nagrani, Cordelia Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2608.30959)  

**Abstract**: Vision-Language Models (VLMs) still struggle on tasks requiring complex visual understanding. We argue that the core issue is not high-level reasoning, but instead failing to locate critical details in the image. Due to this shortcoming, VLMs generate often plausible but incorrect reasoning based on flawed perceptual grounding. To address this, we propose Locator-Critic (LOCI), a training-free framework that decouples visual search from evidence verification. LOCI employs a Locator agent to propose candidate visual evidence and a separate Critic agent to evaluate its relevance and sufficiency. These agents engage in an iterative refinement loop, progressively improving the evidence until it is adequate to answer the given question. This decoupled, self-correcting process yields substantial performance gains, achieving state-of-the-art results on multiple complex visual benchmarks. LOCI improves accuracy for both open-weight models like Qwen3-VL (+12.1 on V*, +5.8 on HR-Bench and +11.2 on VisualProbe-Hard) and proprietary models like Gemini 2.5 Pro (+8.9 on V*, +4.3 on HR-Bench, +4.8 on VisualProbe-Hard). 

---
# Taking the Whys Seriously: Limitations of Counterfactual Explanations in Justification and Recourse 

**Authors**: Mattia Cerrato, Otto Sahlgren, Xenia Heilmann  

**Link**: [PDF](https://arxiv.org/pdf/2608.30956)  

**Abstract**: Counterfactual explanations (CEs) are widely used in explainable artificial intelligence (AI) to show how a model's outputs would change if the input features were manipulated. This technique is used for a range of tasks such as debugging models, explaining predictions, justifying decisions, and providing algorithmic recourse. In this paper, we explore the normative legitimacy of employing counterfactuals in real-life model deployment settings. We discuss the different stakes involved in these different purposes for which CEs are commonly employed, and find stricter requirements for justification and recourse. In particular, we find that naive application of CEs for justification and recourse can lead to ignoring contestable choices made throughout the machine learning (ML) pipeline, thus obfuscating that decisions and counterfactuals for those decisions are also artifacts of an organization's materialized design and governance choices. We demonstrate this with four empirical experiments involving interventions at stages of the ML pipeline ``upstream" of the explanation itself, and show that these affect the generated counterfactuals. We find that an organization's choices on measurement models for feature and labels, business requirements, model validation, and the metric of model success have as much or more impact on the generated counterfactuals as the specifics of the generating method. Our findings underline the need to account for such choices upon providing justification and recourse, providing a stark reminder of the relational nature of these tasks. As putative justifications or recourse recommendations, CEs do not provide adequate answers to some important "why"-questions because they preclude consideration of whether the decision-maker ought to have acted differently. 

---
# MusGU+: Toward a Musician-Centered Evaluation Framework and Discovery Tool for Generative Music AI 

**Authors**: Laura Ibáñez-Martínez, Roser Batlle-Roca, Xavier Serra, Martín Rocamora  

**Link**: [PDF](https://arxiv.org/pdf/2608.30940)  

**Abstract**: Generative music systems are increasingly presented as tools that democratize music creation, yet their practical suitability for musicians remains underexplored. Prior work includes openness-focused evaluation frameworks, such as MusGO (Music-Generative Open AI), as well as qualitative studies of musicians' experiences with generative systems. However, these approaches do not support systematic comparison or early-stage discovery of models for creative use. Motivated by such limitations, we introduce MusGU+, a musician-centered evaluation framework organized around three dimensions: Adaptability, Usability, and Controllability. Together, these capture whether a model can be feasibly trained or fine-tuned on personal data, integrated into real-world music workflows, and controlled in musically meaningful ways. We evaluate 10 representative generative music systems and present an interactive discovery tool that enables musicians to explore and filter models according to these criteria. While MusGO remains valuable for promoting responsible research practices, MusGU+ supports informed selection and practical adoption of generative systems by musicians. 

---
# LightNav-0: Eliciting VLM Spatial Intelligence for Generalist Embodied Navigation 

**Authors**: Shaoan Wang, Aocheng Luo, Fei Huang, Jingyi Xu, Xiaoyang Wang, Yueyu Wang, Qianli Ma, Fan Yang, Ran Mei, Jia Wei, Jiangpeng Hu, Xuhao Liu, Hongming Chen, Yuanbin Shao, Yiyang Lin, Ziliang Li, Liang Pan, Xinhang Liu, Yuntao Ma, Tingxiang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2608.30935)  

**Abstract**: Embodied navigation requires agents to translate heterogeneous goals and visual observations into actions across tasks, environments, and robot embodiments. Modern vision-language models (VLMs) already encode spatial priors for visual grounding, spatial reasoning, and pointing, but these capabilities are rarely elicited directly for robot control. Existing navigation systems instead rely on task- or embodiment-specific components, fragmenting perception, reasoning, and action while offering limited generalization. Here we present LightNav-0, a compact generalist embodied navigation model that elicits the spatial intelligence of a pretrained VLM and aligns it with navigation, without task-specific prediction heads. LightNav-0 represents diverse navigation tasks through a unified token interface: dual-channel pointing expresses task-, scene-, and embodiment-agnostic spatial intent, while a residual vector-quantized action tokenizer maps this intent to precise, embodiment-specific trajectories. Together with temporally aware visual history compression, ER mid-training, supervised fine-tuning, and reinforcement learning, this formulation supports instruction following, open-vocabulary object navigation, and visual tracking within a single model. The navigation training corpus spans 2K+ scenes and 4K+ hours of embodied navigation data. LightNav-ER, the embodied-reasoning checkpoint used to initialize LightNav-0, attains the highest complete-set average across 8 embodied-reasoning benchmarks, while LightNav-0 achieves state-of-the-art monocular success rates across all 10 public navigation simulation settings. Real-world evaluations further demonstrate zero-shot generalization across robot embodiments, diverse scenes, and static and dynamic targets. These results establish compact VLMs as a unified and transferable backbone for generalist embodied navigation. 

---
# Stride-k Subsampling: Train-Free Audio Token Reduction for Whisper 

**Authors**: Chanhee Cho, Junhyuk Choi, Bugeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.30927)  

**Abstract**: Whisper exposes speech through a fixed 1500-token encoder interface, now a default representation for ASR decoders and Whisper-based speech language models (SpeechLMs), yet its redundancy remains largely unexamined. We propose stride-k subsampling, a deterministic indexing operation that retains every k-th token after the convolutional stem or encoder transformer. Across five Whisper scales, k=2 preserves baseline WER at both positions, with CKA attributing this stability to acoustic overlap at the stem and attention-induced redistribution at the encoder output. Applying stride-2 at both positions cuts audio tokens by 75% and total GFLOPs by 52-58%, with small WER costs on most ASR benchmarks and larger costs on harder ones. The same configuration extends to three Whisper-based SpeechLMs, yielding modest accuracy drops on stronger baselines and larger drops on weaker ones, while reducing end-to-end latency by 19.6-27.4%. Requiring no training or auxiliary computation, stride-k subsampling exploits Whisper's preprocessing redundancy, indicating that its audio-token interface carries more capacity than downstream tasks require. 

---
# Towards Stream Learning on Embedded Systems: Benchmarking the Memory Consumption of Stream Learning Methods 

**Authors**: Sebastian Buschjäger, Nuwan Gunasekara, Heitor Murilo Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2608.30923)  

**Abstract**: Stream learning is commonly evaluated through predictive performance and adaptation to concept drift. However, sustained operation of a stream learner also requires predictable and bounded resource usage even on long streams. This requirement becomes even more critical when learning moves from servers to near-sensor embedded systems where memory and processing are scarce resources. In state-of-the-art stream learning, however, we perceive a strong focus on concept drift adaptation, whereas resource usage is often an evaluation byproduct. To close this gap, we benchmark seven representative stream classifiers on 13 real and synthetic streams under model-size budgets from 128\,KiB to approximately 8\,MiB. Our benchmark comprises a total of 6,463 experiments. We measure failure-aware accuracy, peak model size, time to budget exhaustion, and prediction-plus-update latency. The results reveal two distinct resource failure modes. Adaptive ensembles can exceed small budgets almost immediately because of their initial footprint, even when their size remains stable thereafter. Incremental trees can fit initially but grow throughout a long stream, with HoeffdingTrees (HT) and Extremely Fast Decision Trees (EFDT) increasing by median factors of 7.37 and 5.87. Explicitly compact methods remain the only viable option under the smallest budgets, but are usually overtaken as larger budgets make adaptive ensembles competitive. Hence, many state-of-the-art methods are only partially applicable in embedded systems or for long-running systems. We therefore call on the stream-learning community to make bounded resource usage a first-class design objective alongside drift adaptation, and propose concrete steps toward this goal, including an API through which stream learners can explicitly expose and respect resource budgets. 

---
# Safety Screening for Voltage Control in Active Distribution Grids via Distributionally Robust Conformal Screening 

**Authors**: Sarra Bouchkati, Petros Ellinas, Adriana Geisler, Steffen Kortmann, Johanna Vorwerk, Spyros Chatzivasiliadis, Andreas Ulbig  

**Link**: [PDF](https://arxiv.org/pdf/2608.30889)  

**Abstract**: Deploying a new control policy for voltage control in active distribution grids requires evidence that physical limits will be satisfied before the policy is tested on the physical grid. This assessment is difficult for two reasons. First, simulations cannot capture every disturbance, modeling error, and device interaction present in the real grid. Second, historical measurements reflect operation under existing control policies, whereas a new policy may drive the grid into different operating conditions. To address these challenges, we propose Distributionally Robust Conformal Safety Screening (DR-CSS), a policy-agnostic framework for pre-deployment, scenario-by-scenario screening of a new control policy using historical data and a nominal simulator. For each new scenario, the simulator predicts a future voltage trajectory for the whole grid; DR-CSS then constructs a conformal safety interval around this prediction using historical simulation-to-reality errors. The interval is further enlarged to account for closed-loop changes induced by the deployment of the new policy and its interactions with the remaining controllers. To the best of our knowledge, DR-CSS is the first framework in power systems to combine historical data from an existing control policy with an imperfect simulator for pre-deployment safety screening of a new policy. Experiments on the IEEE 33-bus and IEEE 141-bus systems evaluate the deployment of learning-based voltage control policies and show that DR-CSS identifies all unsafe test scenarios. To reduce unnecessary warnings on safe scenarios, we adapt the safety intervals to different operating conditions and gradually introduce new policies with recalibration after each stage. These extensions increase the informational value of the safety screening and support safer deployment decisions in active distribution grids. 

---
# Evaluating and Mitigating Anti-LGBTQ Biases in German and Multilingual Language Models 

**Authors**: Melina Morch, Daniel Braun  

**Link**: [PDF](https://arxiv.org/pdf/2608.30884)  

**Abstract**: While gender and racial biases in language models have been widely studied, anti-LGBTQ biases remain underexplored, particularly beyond English. Existing benchmarks often do not capture cultural and linguistic variation and rely on gender representations. This paper introduces a multilingual German-English benchmark dataset for the evaluation of anti-LGBTQ biases in language models. It combines community-sourced stereotypes from German-speaking queer individuals with a German translation of WinoQueer. The data is used to evaluate eight language models across sizes and architectures and explore mitigation through fine-tuning on community and progressive media content. Results show that language models reproduce anti-queer stereotypes, with variation across identities and models. Differences between the translated and community-based data highlight the importance of cultural adaptation for multilingual bias evaluation. Fine-tuning reduces bias on average, but not consistently across models and identities. Warning: This text contains examples of anti-queer hateful language and stereotypes. 

---
# Personas Differ from Native-Language Generation: Language Pathways Shape LLM Interpersonal Advice 

**Authors**: Jinhee Won, Xinlan Emily Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.30873)  

**Abstract**: LLMs are increasingly used for interpersonal advice and as tools for studying social behavior across languages and cultures. A common shortcut for eliciting language- or culture-related variation is to ask a model to answer as a native speaker. We test whether this native-speaker persona reproduces the outputs obtained when models instead generate advice in the target language and translate the response back into English. Using 600 interpersonal advice questions across 13 languages and eight LLMs, we compare native-language generation followed by translation (NL) with native-speaker persona prompting (NP), measuring linguistic style, behavioral scaffolding, and forced-choice action recommendations. We find that NP and NL are not interchangeable. Compared to NL, NP often increases lexical social cues, including affiliation and positive tone, while reducing qualities such as concreteness and social attunement; NP also provides less actionable scaffolding in open-ended advice. In forced-choice scenarios, NP changes which action the model selects, favoring confrontation over redirection, with effect sizes varying across languages, topics, and models. Our results show that cross-lingual elicitation strategy is a consequential methodological choice that can change both how advice is framed and which actions models recommend. 

---
# Exponential random graph models with soft clique constraints 

**Authors**: Yasmin Tousinejad, Vera Koponen  

**Link**: [PDF](https://arxiv.org/pdf/2608.30869)  

**Abstract**: Let $r\geq3$ be fixed, and let $\mathbf{G}_n$ be the set of all simple graphs with vertex set $[n]=\{1,\ldots,n\}$. We consider an exponential random graph model which gives higher probability to $G \in \mathbf{G}_n$ than to $H \in \mathbf{G}_n$ if $G$ has fewer $r$-cliques than $H$. But all graphs in $\mathbf{G}_n$ have positive probability. The degree to which graphs with fewer $r$-cliques are given higher probability is determined by a positive weight $w$. We prove that, asymptotically almost surely as $n \to \infty$, a random graph from $\mathbf{G}_n$ has a vertex partition into $r-1$ parts of roughly equal size, the density of edges between the parts is close to $1/2$, and for every $\varepsilon > 0$ the density of edges within any part is less than $\varepsilon$. The asymptotic structural properties are independent of the weight $w$ as long as it is positive. We also extend the result to the context of several clique sizes, each one with its own weight. 

---
# TAMI: Temporally Aligned, Missingness-Aware, and Interpretable Multimodal Fusion for Mental Health Assessment in Older Adults with Mild Cognitive Impairment 

**Authors**: Merna Bibars, Bolaji Omofojoye, Allan I. Levey, Rachel Hershenberg, Gari D. Clifford, Hyeokhyen Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2608.30857)  

**Abstract**: Depression and anxiety in older adults with Mild Cognitive Impairment (MCI) are frequently underdiagnosed due to limited access to care. Multimodal analysis of remote clinical interviews is a scalable screening approach, but existing methods have three limitations. First, they do not correct temporal misalignment across multimodal features extracted at different resolutions, inducing spurious cross-modal associations. Second, remote recordings exhibit uneven modality dropout, but missing values are often zero-filled, making them indistinguishable from valid near-zero measurements. Finally, they do not jointly attribute predictions to modalities, questions, and interview moments, limiting fine-grained clinical interpretation. We propose a Temporally-Aligned, Missingness-Aware, Interpretable (TAMI) multimodal fusion framework. TAMI aligns speech, language, facial, and physiological features within question-answer segments on a shared timeline, encodes modality-level missingness over time, and conditions fusion on question context. In interviews with 49 older adults with MCI, TAMI achieved area under the receiver operating characteristic curve (AUROC) scores of 0.68 (depression) and 0.69 (anxiety). Fine-grained temporal alignment of multimodal features produced the largest performance gain ($\Delta{\geq}0.1$). Multi-level interpretability analysis revealed that depression classification relied on eyegaze and open-ended questions, while anxiety classification depended on eyegaze and head pose, with attribution uniformly distributed across questions. Using only responses to the open-ended questions (5.1min), the depression model achieved an AUROC score of 0.67, which was not significantly different from using the full interview (19min) ($p>0.05$). Our findings support designing interview protocols centered on open-ended questions for depression screening in older adults with MCI. 

---
# Pretrained, Curriculum-Tuned, and Ensembled: A Tracer-Aware Interactive Segmentation Pipeline for AutoPET V 

**Authors**: Xinglong Liang, Chunyao Lu, Tianyu Zhang, Jiaju Huang, Tao Tan, Yunchao Yin, Lishan Cai  

**Link**: [PDF](https://arxiv.org/pdf/2608.30844)  

**Abstract**: Interactive lesion segmentation in whole-body PET/CT requires a model to provide a strong initial prediction while also responding efficiently to sparse corrective scribbles during inference. This setting is particularly challenging because tracer distributions, physiological uptake patterns, lesion appearance, and acquisition characteristics differ substantially between FDG and PSMA studies. We present TRIAGE, Tracer-aware Refinement via Interactive Anatomy-Guided sEgmentation. The core backbone is a 3D STU-Net initialized through masked autoencoding pre-training with an asynchronous masking strategy, aiming to learn transferable anatomical and cross-modal representations before task-specific fine-tuning. In parallel, we train an auxiliary organ segmentation model whose predictions provide explicit anatomical context and help distinguish physiological uptake from malignant lesions. A dedicated tracer classifier first routes each study to an FDG- or PSMA-specific branch. Within each branch, a first-stage segmentation model consumes CT, PET, and organ context to generate an initial lesion mask. The initial prediction is then combined with cumulative foreground/background scribbles and refined by a second interactive segmentation network. The FDG and PSMA branches share the same overall processing pipeline but are trained independently to account for tracer-specific appearance and error modes. We additionally employ curriculum-style training and model ensembling to improve robustness across interaction steps and heterogeneous cohorts. Experiments are conducted using the official AutoPET V data and ten-fold split; quantitative results, ablations, and final test-set performance are left as placeholders to be completed after the challenge evaluation. Code: this https URL. 

---
# Reliable Benchmarking of Artifact Detection in Computational Pathology: A Reproducibility and Uncertainty Analysis 

**Authors**: Konstantinos Moutselos, Ilias Maglogiannis  

**Link**: [PDF](https://arxiv.org/pdf/2608.30835)  

**Abstract**: Background and Objective: Quality control is a prerequisite for whole-slide image analysis, yet the benchmarks on which quality-control methods are compared share four properties that make their reported differences hard to interpret: few independent slides, annotation concentrated in a minority of them, pooled ratio metrics with no closed-form standard error, and a single inherited train/test partition. We propose a reliability protocol for such benchmarks. Methods: The protocol quantifies four sources of variability - test-set sampling, training stochasticity, partition composition, and undocumented preprocessing - a claim is reportable only if it survives all four; three of the four cost minutes of compute. We apply it to an independent reconstruction of a published diffusion-based artifact detector, evaluated on the original 24-slide partition and against a supervised baseline. Results: The method's central mechanism reproduces: the auxiliary contrastive term improves pooled F1 from 0.673 to 0.688 and replicates under a second seed (+0.0156, p = 0.031; +0.0190, p = 0.005), although it acts on pen marking rather than the artifact types cited to motivate it. Its comparative claims do not: differences between design variants, and against the supervised baseline, fall inside the uncertainty of the evaluation. Four of 24 slides carry 70% of scored annotated pixels, giving an effective sample size of 6.2, and the inherited partition sits at the 7th percentile. An unreported tissue-restriction step excludes 41.4% of out-of-focus annotation against 2.6% of air bubble; such a gate is confounded with blur by construction. Conclusions: Small-cohort benchmarks support far weaker conclusions than current reporting implies. The four checks are cheap enough to accompany any evaluation on such a resource and separate reproducible effects from differences the evaluation cannot resolve. 

---
# Lucida: Parse, Generate, and Place for Composable Real-to-Sim Scene Modeling 

**Authors**: Minghan Qin, Yuang Wang, Xiuyu Yang, Yushi Long, Yujian Zhang, Ruihuan Wang, Kai Ye, Yangang Zhang, Hang Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.30821)  

**Abstract**: Composable scene modeling aims to recover a real indoor scene as complete, editable object assets arranged as observed, giving robot simulation and embodied AI a simulation-ready replica of the real environment whose objects can be manipulated individually. Existing pipelines decompose the task into three steps---parse the observations into instances, generate an asset for each, and place each asset back---but every step presumes an input that a cluttered capture rarely provides: accurate instance geometry, unoccluded views, and assets that accurately match the observations. We propose Lucida, which keeps this order but redistributes the requirements, so each step consumes only what a real capture reliably provides and precision is reached at the end of the pipeline rather than demanded at its start. Lucida parses the video into a scene graph whose nodes carry per-instance multi-view evidence, generates a complete asset for each instance from its evidence, and places assets with GizmoAct, a VLM policy that casts placement as multi-turn GUI interaction, manipulating the object's gizmo in a closed loop and deciding itself when alignment is reached. Across scene-level 3D object detection, object pose estimation, and scene reconstruction, Lucida improves mAP over Boxer by 69% on R2S-Scene, raises ADD-SB@0.05 from 57.8% to 83.4% on CA-1M, and increases scene F-Score from 0.794 for SAM3D to 0.924. 

---
# A Composition-Aware Pretraining Framework for Geospatial Foundation Models 

**Authors**: Aryan Kashyap Naveen, Abhishek Srinivas, Pranav Moothedath, Shrutilipi Bhattacharjee  

**Link**: [PDF](https://arxiv.org/pdf/2608.30817)  

**Abstract**: Geospatial foundation models have emerged as state-of-the-art methods for downstream Earth observation tasks. However, existing pretraining methodologies process imagery through a single-concept lens, failing to capture the highly compositional nature of complex satellite scenes. We propose a composition-aware pretraining framework that explicitly encodes fractional land-cover mixtures. Each satellite image cell is mapped to a histogram representing its fractional land-cover distribution, which we term the "composition target". These targets serve as the primary prediction objective and are distilled into the backbone using Earth Mover's Distance. Experimental evaluation shows that composition-aware pretraining yields substantial gains on region-level understanding tasks requiring semantic similarity judgment, including zero-shot image retrieval and scene classification, while remaining competitive on tasks requiring fine-grained spatial precision, such as segmentation and object detection. With a 36.8M-parameter backbone, our framework outperforms SatMAE and Prithvi-EO-2.0, which contain 303M and 600M parameters, respectively, in most retrieval and scene classification settings. On the fine-grained ForestNet-12 dataset, a rigorous testbed for compositional discrimination, our method boosts baseline mAP@10 from 0.279 to 0.434, a 55.6% relative improvement, providing direct evidence for the effectiveness of explicit composition modeling. The code implementation can be found at this https URL 

---
# Aggregate Disambiguation Systems 

**Authors**: José María Lago, Albert Castellana, Edgars Nemše  

**Link**: [PDF](https://arxiv.org/pdf/2608.30805)  

**Abstract**: Natural-language tasks can elicit different verdicts from protocol-following evaluators that receive the same declared information. We study aggregate disambiguation systems (ADSs). Given a task and a candidate solution, each evaluator casts a binary vote on whether the solution should be accepted, and the system aggregates the votes of a finite panel. The target is protocol reproducibility relative to an explicitly declared evaluator reference, not semantic truth. We separate fixed finite censuses, probabilistic evaluator populations, and growing-census limits, since their endpoint laws and guarantees are not interchangeable. In the population setting, we use finite samples to estimate how often a finite panel reaches the same decision as the declared evaluator population. We provide a lower confidence bound on the fraction of candidate solutions for which the disagreement probability is at most a chosen tolerance. The calculation accounts separately for sampling candidate solutions and sampling evaluators. The construction permits arbitrary dependence among columns induced by shared evaluator rows and uses exact binomial intervals at the evaluator layer and an exact one-sided binomial inversion at the generator layer. Simulations check the implementation against known population coverages and expose power limitations. 

---
# Conjoint Audio-to-Spikes Encoding and Processing for Efficient Neuromorphic Speech Recognition 

**Authors**: Valentin M. Meunier, Amélie Gruel, Pierre Lewden, Adrien F. Vincent, Sylvain Saïghi  

**Link**: [PDF](https://arxiv.org/pdf/2608.30792)  

**Abstract**: Obtaining data from neuromorphic sensors and processing it with Spiking Neural Networks is a promising solution to lower the energy cost of artificial intelligence. The current rarity of natively neuromorphic datasets promotes the development of software tools to translate input sensory data into spikes. However, highly bio-mimetic simulators can be challenging to implement on digital hardware. In this work, we evaluate the neuromorphic encoding and subsequent classification of audio into spikes using a non-learnable, high-level, programmable encoder targeting hardware implementation on FPGA. We quantify the pipeline's efficiency with hardware-agnostic metrics based on the quantitative spiking activity. Our study focuses on the simultaneous optimisation of encoder and classifier: the first provides efficient and informative data so that the latter achieves a better performance with an overall lower energy cost at learning and inference. This work introduces the first end-to-end neuromorphic spike-encoding and evaluation of the TIMIT dataset. Our simple feedforward network reaches a classification accuracy of 99.77% on a spike-encoded Heidelberg Digits, overcoming the neuromorphic state of the art on this benchmark dataset. 

---
# On the Prospects of Dynamic LLM Conversations in Software Development 

**Authors**: Annemarie Wittig, Alina Mailach, Janet Siegmund, Norbert Siegmund  

**Link**: [PDF](https://arxiv.org/pdf/2608.30756)  

**Abstract**: Large language models (LLMs) have become an essential tool for assisting developers, yet we still lack knowledge on ways to effectively support their interactions during development activities. That is, the quality of interactions with a chat-based LLM still strongly depends on how developers phrase prompts and which information they include.
Our goal is to evaluate whether interventions into these interactions with LLMs have an effect on software developers---be it harmful or beneficial. To this end, we conducted a four-month longitudinal study with third-semester computer science students working on a full-stack Web development project using chat-based LLMs under three conditions: (1) a \emph{context}-aware group received intent-based conversation augmentation, (2) a \emph{proactive} group received follow-up suggestions and tailored advice, and (3) a \emph{control} group without intervention. Our augmentations are minimal: (i) to reduce confounding factors and (ii) to isolate treatment effects.
Analyzing interaction logs and user surveys revealed no major differences in interaction patterns, indicating no detectable harmful effects in the measured outcomes when intervening in interactions. Moreover, we observed trends of increased satisfaction with the \emph{proactive} treatment. The results indicate that even with minimal interventions, dynamic guidance mechanisms for developer-LLM interactions show observable effects, such that more severe augmentations may have the potential to substantially improve developer satisfaction. 

---
# Learning from What You Retrieve: Online RL Fine-Tuning for Semantic Retrieval 

**Authors**: Shaowei Wei, Chong Huang, Songtao Fang, Jin Zhang, Zhuojun Wang, Chengfu Huo  

**Link**: [PDF](https://arxiv.org/pdf/2608.30753)  

**Abstract**: In large-scale e-commerce retrieval, dual-encoder retrievers are op- timized for contrastive similarity, whereas downstream rerankers capture finer-grained relevance preferences; this objective mis- match limits end-to-end retrieval quality. Reinforcement Learning offers a way to use reward-model feedback for retriever adaptation, but we observe that standard policy-gradient updates can degrade embedding geometry, especially when the document index must remain frozen due to industrial constraints. To address this, we propose PAO (Positive-Advantage-Only), a selective RL optimization method. Our analysis reveals that in- discriminate penalization of negative samples (pushing away) in a frozen high-dimensional space disrupts pre-trained semantic man- ifolds. PAO selectively applies gradient updates only to retrieved items with positive advantages, effectively pulling query embed- dings toward high-reward regions while preserving global topo- logical stability. Experiments on both a massive industrial dataset and public benchmarks demonstrate that PAO significantly outper- forms standard RL and distillation baselines. 

---
# Calibrating Small Language Models for Claim Check-Worthiness Detection 

**Authors**: Pratuat Amatya, Venktesh Viswanathan, Vinay Setty  

**Link**: [PDF](https://arxiv.org/pdf/2608.30731)  

**Abstract**: Assessing claim check-worthiness is an essential first step in automated fact-checking pipelines. This work is motivated by a real deployment challenge at an early-stage startup: running large language models (LLMs) over every incoming claim is cost- and latency-prohibitive, yet smaller models sacrifice accuracy. We propose NN-PPI, a pointwise extension of Prediction-Powered Inference (PPI) that calibrates model predictions at inference time as a lightweight post-hoc layer, without re-training the underlying model. NN-PPI achieves weighted F1 gains ranging from 12% to 33.80% depending on the size and performance of the baseline model, bringing SLMs on par with larger LLMs. Beyond few-shot SLMs, NN-PPI further improves a production-deployed fine-tuned model, demonstrating that residual calibration is complementary to supervised fine-tuning. By recovering LLM-level accuracy from models that are an order of magnitude cheaper to serve, it makes accurate check-worthiness detection substantially cheaper to operate at scale. Our code and data can be found at this https URL. 

---
# RailGen: Improving Railway Intrusion Detection via Agent-Guided Small-Scale Foreign Object Generation 

**Authors**: Quan Hao, Ziyang Tao, Chenxi Zhang, Yudong Wang, Rui Shi, Liguo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30727)  

**Abstract**: Small-object detection under long-tailed data distributions is a fundamental yet challenging problem in multimedia. Railway Foreign Object Detection (RFOD) epitomizes this challenge with easily confused small intrusions and scarce samples. To address these issues, we propose a generative-augmented detection paradigm that leverages multimodal image generation to enrich the feature space of rare and small objects. We first construct RailGen, a multimodal image generation agent based on large models. Under semantic constraints, RailGen automatically invokes tools to generate railway scenes, calibrate intrusion positions, extract foreign objects, and fuse them into realistic intrusion effects. This process produces high-quality synthetic samples that effectively densify the feature representations of tail classes and complete the small-object feature space. Within this paradigm, we further propose FocalDEIM, a detection framework designed to enhance training with generated data. FocalDEIM improves dense matching with Focal Modulation for better small-object discrimination and adopts Focal Loss to emphasize hard samples, thereby alleviating blurred inter-class boundaries in complex railway scenes. Experimental results demonstrate that RailGen can generate high-quality small-scale foreign objects, reducing the object pixel area by up to 58x and 13.85x on average. Equipped with these challenging samples, our paradigm surpasses the baseline DEIM by 5.6% and 7.5% in mAP@50 and mAP@(50-95), respectively, and outperforms existing state-of-the-art methods. Ablation studies verify RailGen's feature-space enrichment and FocalDEIM's boundary discrimination. The paradigm provides an effective multimodal generative solution for long-tailed small-object detection in safety-critical applications. 

---
# BAITBENCH: Measuring Agent Reward Hacking with Optional Shortcuts Planted in ML Tasks 

**Authors**: Pradyumna Shyama Prasad, Meiri Anto, Leon Eshuijs, Julian Moncarz, Kaustubh Kislay, Juan J. Vazquez  

**Link**: [PDF](https://arxiv.org/pdf/2608.30724)  

**Abstract**: LLM agents are increasingly used to run autonomous ML experiments, iterating on target metrics with little human oversight. Prior work has documented reward hacking in these environments, bringing into question the validity of produced research and the broader safety case for AI R&D. Existing benchmarks do not measure exploits that live in the data or the modeling task itself. We introduce BAITBENCH, a suite of three synthetic tabular ML tasks that each contain a shortcut that allows agents to inflate the public test score but fail on a hidden test set. Since the shortcut is optional and using it breaks no stated rule, BAITBENCH measures how often models exploit the shortcut to achieve inflated scores. Across seven frontier agents scored by our two-stage judge pipeline, 57.1% of runs exhibit reward hacking, with five of seven above 50%. Agents cheat even under a second condition where they are prompted not to -the mean cheating rate remains above 50%. We release BAITBENCH, along with the judge implementation, and an annotated dataset of transcripts containing reward hacks as a testbed for evaluating reward-hacking mitigations head-to-head. 

---
# RailSyn: Diagnosis-Guided Image Generation for Traceable Data Completion in Railway Foreign Object Detection 

**Authors**: Quan Hao, Chenxi Zhang, Ziyang Tao, Yuyuan Zhou, Yudong Wang, Rui Shi, Lechuan Xu, Changhao Liu, Liguo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30709)  

**Abstract**: Railway foreign object detection (RFOD) is critical to safe railway operation, yet scarce real positive samples incompletely represent task-relevant variations in object scale, intrusion relation, railway scene, illumination, and adverse weather. Existing synthetic augmentation can improve RFOD detection, but its gains lack an explicit account of the task-relevant deficiencies complemented by the generated data. We therefore introduce RailSyn, a diagnosis-guided framework comprising a real-referenced Inspector and a requirement-aligned Generator. The Inspector constructs a variable-radius empirical cover from finite real observations to localize candidate completion regions and profile synthetic pools. The resulting audit identifies railway-context, intrusion-semantic, and visual-consistency requirements; the Generator addresses them through domain adaptation, agent-planned placement and physical contact relations, and plan-consistent conditional refinement. Using the Inspector, we further trace representation-space changes across generation variants; the complete system attains a local-shell occupation of $C_{gap}$ to 13.64%, which measures generated coverage of real-derived completion regions. Extensive experiments show AP50--95 gains of up to 4.9 points and consistent improvements across nine mainstream detectors, demonstrating broad cross-architecture utility. 

---
# SingProbe Technical Report 

**Authors**: Sing Team  

**Link**: [PDF](https://arxiv.org/pdf/2608.30703)  

**Abstract**: Runtime guardrails are essential for reliable large language model (LLM) deployment, yet existing approaches typically rely on independent, external models that introduce additional inference cost, delayed safety signals, and a capacity mismatch with increasingly capable base models. To address these issues, we introduce SingProbe, a lightweight intrinsic runtime guard that directly reuses hidden states produced during LLM inference and operates alongside autoregressive decoding. Within a unified framework, SingProbe continuously predicts query intent, response safety, and hallucination risk at the token level with negligible additional guardrail inference overhead, offering a "free-lunch" solution. We further introduce SingStreamBench, a benchmark designed to assess whether streaming guardrails remain inactive on benign prefixes while promptly detecting emerging unsafe content. Extensive experiments show that SingProbe achieves competitive or superior performance compared with substantially larger standalone guardrails and specialized hallucination detectors, with only $\approx$2M parameters and $<0.5\%$ extra overhead. Beyond passive detection, we also show that SingProbe scores can anticipate future generation risk and guide constrained safe decoding. We further extend this paradigm to medical generation through SingProbe-Med, which selectively activates risk-directed decoding interventions only when clinically relevant risks emerge. Together, these results demonstrate that internal model representations provide an effective and efficient interface for generation-time monitoring and control. 

---
# An Agentic Retrobiosynthesis Framework with Learned Frontier Selection 

**Authors**: Philippe Meyer, Guillaume Gricourt, Thomas Duigou, Joan Hérisson, Jean-Loup Faulon  

**Link**: [PDF](https://arxiv.org/pdf/2608.30702)  

**Abstract**: Large language models are increasingly used as agents for multistep retrosynthesis, raising the question of how much their search policy contributes independently of the underlying reaction model. We investigate this question in a biological setting through rule-based retrobiosynthesis: a deterministic biochemical engine generates the same validated transitions for every method, searching for routes that terminate in metabolites available to an \emph{Escherichia coli} chassis, while the policy only selects which frontier molecule to expand next. Prompted and LoRA-tuned Qwen2.5-7B policies use a strict choice-only interface. The fine-tuned policy reaches $65\pm1$\% solve rate at 10 expansions on LASER versus 59\% for MCTS, and at 200 expansions reaches $78\pm1$\% versus 75\% on LASER, $88\pm3$\% versus 80\% on the RetroPath RL Golden benchmark, and $63\pm2$\% versus 45\% on the BioNavi-NP benchmark. Fine-tuning also consistently outperforms direct prompting. These results show that route-supervised frontier selection can improve budgeted search without altering biochemical generation, although performance remains dependent on frontier construction and reaction ranking. 

---
# Learning Dynamics of Logits Debiasing for Long-Tailed Semi-Supervised Learning 

**Authors**: Yue Cheng, Jiajun Zhang, Xiaohui Gao, Weiwei Xing, Zhanxing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.30699)  

**Abstract**: Long-tailed distributions are prevalent in real-world semi-supervised learning (SSL), where pseudo-labels tend to favor majority classes, leading to degraded generalization. While many long-tailed semi-supervised learning (LTSSL) methods have been proposed, the mechanisms by which they implicitly debias logits remain poorly understood. In this work, we revisit LTSSL through the lens of learning dynamics and provide a theoretical characterization of logits debiasing. Specifically, we derive a step-wise decomposition of the logits updates, showing that predictions are dominated by class-imbalance bias that reliably reflects label priors. To expose this effect, we use the logits of a task-irrelevant baseline image as an indicator of accumulated bias and prove that they converge to the class prior. This provides a unified view where LTSSL remedies such as logit adjustment, reweighting, and resampling correspond to reshaping gradient dynamics. Based on this insight, we propose DyTrim, a principle-based dynamic pruning framework that reallocates gradient budget through class-aware pruning on labeled data and confidence-based soft pruning on unlabeled data. We provide theoretical guarantees that DyTrim reduces class bias and improves generalization. Extensive experiments on standard LTSSL benchmarks show consistent gains across architectures and methods. Code available at: this https URL 

---
# Learning Materials Properties from Scarce Labels and Unlabeled Crystals 

**Authors**: Wentao Li, Yizhe Chen, Jiangjie Qiu, Yijun Li, Leyi Zhao, Xiaonan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30682)  

**Abstract**: Learning materials properties from scarce labels and unlabeled crystals is a central challenge for data-driven materials discovery. We present SemiMat, a controlled benchmark for semi-supervised materials property regression, and MatRank, a reliability-weighted objective for continuous pseudo-label uncertainty. SemiMat fixes labeled and unlabeled crystal inputs, graph-backbone interfaces, validation-only checkpoint selection, held-out test reporting, normalized MAE (NMAE), and method-rank summaries across six scarce-label tasks, four graph backbones, and five predefined split runs. MatRank builds pseudo-targets from labeled anchors, weights them by local reliability and weak-prediction agreement, trains weak and strong graph views consistently, and adds ranking signals so that unlabeled crystals shape both values and candidate order. Across the retained 24 backbone-task blocks, one fixed MatRank objective gives the lowest aggregate held-out test NMAE (0.896) and best average method rank (2.208). The component, OOD, and generated-pool diagnostics identify where the gain is reliable and where further screening evaluation remains necessary. Code is available at this https URL. 

---
# LCoT-GV: Graph Attention Networks for Verifying Long Reasoning Chains in Large Language Models 

**Authors**: Bérénice Jaulmes, Mehwish Alam  

**Link**: [PDF](https://arxiv.org/pdf/2608.30679)  

**Abstract**: Large Reasoning Models produce Long Chains-of-Thought (LCoTs) which involve breaking down the problem into smaller reasoning steps before reaching the conclusion. However, these steps often contain contradictions, unsupported inferences, or irrelevant steps, even when the final answer is correct. We propose Long Chain-of-Thought Graph Verifier (LCoT-GV), a graph-based framework that represents LCoTs as reasoning graphs. Each node in the graph represents a reasoning step and the edges encode semantic and logical relations. A Graph Attention Network is then trained to predict chain-of-thought correctness from the reasoning graph. We construct a new graph-oriented verification dataset from multiple reasoning benchmarks for question answering in various domains. The results show that our method is competitive with the most similar approaches. 

---
# CoMPASS: Collaborative Molecular Property Prediction via Adaptive Small-Large Model Synergy 

**Authors**: Wentao Li, Jiangjie Qiu, Yijun Li, Leyi Zhao, Xiaonan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30674)  

**Abstract**: Accurate molecular property prediction requires both statistical reliability and chemical reasoning. Graph neural networks can be calibrated directly on labeled assays but remain limited by the coverage of their training data. Large language models (LLMs) can compare molecular evidence and articulate chemical rationales, yet are unreliable as standalone quantitative predictors. The central challenge is therefore to determine when an LLM should influence a calibrated model and by how much. Here we present CoMPASS, a retrieval-calibrated framework for small-large model collaboration. CoMPASS retains a graph attention network (GAT) as the predictive anchor, retrieves locally relevant training molecules, provides attention-grounded evidence to an LLM, and converts its proposal into a bounded correction through an agreement-aware gate. Across six classification and two regression benchmarks, CoMPASS improves the GAT anchor in regions of correctable uncertainty while limiting LLM intervention in high-confidence regimes. Ablations show that the gains arise from validation-calibrated retrieval and bounded fusion rather than prompting alone. These results suggest that generative reasoning should augment calibrated prediction through evidence-grounded, controlled corrections rather than direct output replacement. Code is available at this https URL. 

---
# Fine-Grained Multi Image Object Hallucination Benchmark 

**Authors**: Joonki Min, Chaeyun Kim, Hyungwook Choi, Yejin Kim, Kihyun Kim, Yohan Jo, Joonseok Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.30653)  

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly deployed in multi-image scenarios requiring complex reasoning across visual contexts. However, current MLLMs remain fundamentally limited by object hallucination-generating plausible yet factually inconsistent descriptions about objects. Existing benchmarks, designed primarily for single-image settings or providing only high-level multi-image assessments, cannot systematically diagnose how visual complexity and reasoning demands trigger hallucination. To address this gap, we introduce MIOH, a fine-grained multi-image object hallucination benchmark that systematically evaluates object hallucination across four foundational tasks (existence, counting, attribute, position) through three multi-image reasoning patterns (comprehensive, comparative, selective) under three controlled adversarial pressures (visual context scale, perceptual difficulty, contextual bias). Through evaluation of 29 models, we reveal that even state-of-the-art systems like GPT-5 and Gemini-2.5-Pro exhibit distinct failure patterns across different reasoning patterns and tasks. Our evaluation reveals that hallucination stems not merely from perceptual failures but from integration-stage limitations when maintaining object representations across multiple images. MIOH provides a controlled framework for analyzing multi-image object hallucination and serves as a critical evaluation tool for developing more reliable multimodal AI systems. 

---
# BiG-SURE - Bipartite Graph for Semantic Uncertainty and Reliability Estimation of LLMs 

**Authors**: Debarpan Bhattacharya, Malay Phadke, Sriram Ganapathy  

**Link**: [PDF](https://arxiv.org/pdf/2608.30646)  

**Abstract**: Reliable uncertainty estimation is a crucial requirement for deploying large language models (LLMs) and vision-language models (VLMs) in safety-critical settings, especially when the model parameters are not accessible (black-box). We propose BiG-SURE, an uncertainty estimator based on cross-temperature semantic agreement. The method samples low-temperature responses as stable semantic anchors and high-temperature responses as probes under meaning-preserving input transformations. It then constructs an anchor-probe Bipartite Graph (BiG) using NLI-based entailment scores and defines confidence through the normalized squared spectral energy of this matrix, with uncertainty given by its complement. This bipartite graph-based Semantic Uncertainty and Reliability Estimation (SURE) score measures whether high-temperature probes remain semantically aligned with the model's stable low-temperature belief or not. We evaluate BiG-SURE on text QA, multilingual QA, and multimodal QA tasks across multiple model families. In these experiments, BiG-SURE improves average abstention AUROC over prior black-box uncertainty estimators, while remaining simple, unsupervised, and applicable to black-box model settings. 

---
# GMTS: Gradient Magnitude-based Token Selection Improves RLVR Training for LLM Reasoning 

**Authors**: Outongyi Lv, Yuanwei Zhang, Xiaoqun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30632)  

**Abstract**: Reinforcement learning (RL), particularly RL with Verifiable Rewards (RLVR), has recently emerged as a central paradigm for enhancing large language models' (LLMs) reasoning abilities, demonstrating remarkable effectiveness across reasoning tasks. Recent studies suggest that high-entropy tokens play an exceptionally important role in model training, since training with only the highest 20% entropy tokens yields significant performance gains. However, why such high-entropy tokens are beneficial remains insufficiently understood. In this work, we find that although high-entropy tokens within one answer tend to correlate with large gradient magnitude, entropy alone fails to consistently reflect token importance across different answers, considering the variations in the answer-level reward signals. Based on this observation, we introduce the Gradient Magnitude-based Token Selection (GMTS) method to quantify token importance, which leverages the entropy-gradient connection to approximate gradient-magnitude rankings for token selection. We find that training on the top 20% tokens ranked by GMTS consistently outperforms entropy-based token selection across three reasoning domains and various model sizes, suggesting that GMTS provides a more fine-grained estimate of token contribution for RLVR training. 

---
# Cost-efficient Active Learning for Referring Image Segmentation and Grounding 

**Authors**: Junbeom Hong, Seonghoon Yu, Hyung Rok Jung, Sundong Kim, Jeany Son  

**Link**: [PDF](https://arxiv.org/pdf/2608.30621)  

**Abstract**: Collecting natural-language referring expressions along with region annotations, such as masks or boxes, is a major bottleneck in visual grounding (VG), as annotators must write descriptions that distinguish target regions from visually similar ones. We tackle this by formulating active learning (AL) for VG under the realistic setting where only raw images are available without accompanying text. Since ground-truth text is unavailable, sample selection must estimate which images contain ambiguous regions that would require discriminative referring expressions. To address this, we generate auxiliary region-text pairs using foundation models, and introduce Referred Region Ambiguity, a new acquisition function that measures whether the model's confidence collapses onto a single region or disperses across multiple candidates. It allows our method to prioritize images with strong cross-region competition, which are more informative due to their visual ambiguity. We also design a referring-expression annotation interface that helps annotators quickly focus on writing discriminative language with a few clicks. Experiments on RIS and REC benchmarks show that our AL framework consistently outperforms several AL baselines, while a user study shows up to 1.6X faster description labeling of ours. 

---
# Hidden Threat in Synthetic Data: Covert Targeted Bias Injection through Benign Text 

**Authors**: Minkyung Cho, Jihyo Kim, SeungWoo Song, Junghun Yuk, Minjoon Kee, Hoyun Song, KyungTae Lim  

**Link**: [PDF](https://arxiv.org/pdf/2608.30619)  

**Abstract**: Synthetic data is increasingly used to train large language models (LLMs), yet its security implications remain poorly understood. Prior work on subliminal learning suggests that models can inherit behavioral traits from seemingly unrelated training data. In this work, we investigate whether such mechanisms can be exploited to inject targeted social biases into aligned models through semantically benign synthetic data. We construct a pipeline in which a misaligned teacher model generates filtered synthetic datasets across domains such as creative writing and code generation, which are then used to fine-tune aligned student models. Our experiments show that benign-looking synthetic data can act as a covert channel for transmitting targeted biases while largely preserving the student model's general task capabilities. These results reveal a previously underexplored security risk in synthetic data-driven LLM training pipelines and highlight the need for improved safeguards. As one possible step toward this goal, we suggest that log-linearity-based scoring may provide a useful signal for screening seemingly benign synthetic data. 

---
# Reading the News: Adapting Large Language Models to Swedish Journalism Through Continued Pre-Training 

**Authors**: Lukas Borggren, Jenny Kunz, Marco Kuhlmann  

**Link**: [PDF](https://arxiv.org/pdf/2608.30609)  

**Abstract**: Large language models are increasingly capable in general, but their utility can remain modest in niche or understudied areas. One approach to address this limitation is to specialise existing models through additional training on target-domain corpora. In this work, we investigate such continued pre-training for adapting large language models to Swedish journalism, using a high-quality dataset that we curate from millions of news articles. To evaluate the adaptation efficacy, we also construct a novel domain-specific benchmark that covers six editorial tasks. Through full and parameter-efficient fine-tuning across two model sizes, we find that continued pre-training yields benefits in the target domain, but only when paired with experience replay to mitigate forgetting. We observe consistent enhancements in the models' generation quality and factual knowledge, but not their proficiency in discriminative tasks. Exploring a training-free method to facilitate instruction following, we see further improvements, but exclusively for models trained with low-rank adaptation. Crucially, we demonstrate the importance of targeted evaluation in the adaptation process, as an existing Swedish benchmark largely fails to capture the models' in-domain performance gains. 

---
# Generative Retrieval for E-commerce: Jointly Learning Embedding and Codebook with Same Product Cluster 

**Authors**: Songtao Fang, Zihao Xu, Shaowei Wei, Jin Zhang, Zhuojun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30606)  

**Abstract**: With the development of large language models (LLMs), generative retrieval is becoming increasingly important in e-commerce scenarios. Current mainstream approaches typically use a two-stage training strategy: first train a product embedding model, and then learn a codebook that maps embeddings to product IDs. This cascaded approach suffers from two major issues: (1) error accumulation-if the embedding model in the first stage produces biased representations, the codebook in the second stage cannot correct these errors, degrading final retrieval performance; and (2) codebook learning relies solely on product embeddings and lacks modeling of query-to-product and product-to-product interactions. As a result, products belonging to the same cluster may be assigned inconsistent IDs by the codebook, further hurting retrieval accuracy. To address these problems, we propose a novel method that jointly trains the embedding model and the codebook, and incorporates same product cluster information as an additional supervision signal. Experimental results demonstrate that our method significantly improves e-commerce retrieval performance while simultaneously enhancing both embedding and codebook learning. 

---
# DiffSAC: Diffusion-guided Sampling for Consensus-based Robust Estimation 

**Authors**: Chang Nie, Guangming Wang, Zhe Liu, Hesheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30603)  

**Abstract**: Robust estimation is a core computer vision task frequently tackled using sample consensus. However, traditional methods suffer from inefficient sampling as they struggle to identify effective minimum sets before hypothesis evaluation. To address these challenges, we propose a novel Diffusion-guided Sampling for Consensus-based Robust Estimation (DiffSAC) framework. DiffSAC introduces a diffusion model to learn the distribution of effective minimum sets. It refines the confidence for each data point, indicating whether it belongs to a good minimum set, rather than ranking the data points as in previous work. This significantly reduces the need to process numerous bad sets. To constrain the refinement direction, geometric features are incorporated as conditions within our diffusion model. Consequently, DiffSAC outputs a small number of high-quality minimum sets, enabling identification of the best hypothesis via consensus evaluation. Notably, compared to previous works requiring evaluating over ten thousand hypotheses, DiffSAC achieves state-of-the-art performance with only dozens, significantly boosting efficiency. Extensive experiments across five classic computer vision tasks demonstrate the superiority of DiffSAC. The diffusion model's sampling accelerators enable real-time operation, and DiffSAC can be used as a plug-and-play module to improve existing sample consensus methods. 

---
# Collapsibility of Performance Metrics in Clinical Predictive AI 

**Authors**: João Matos, Ben Van Calster, Richard D. Riley, Paula Dhiman, Gary S. Collins  

**Link**: [PDF](https://arxiv.org/pdf/2608.30568)  

**Abstract**: Background: Population level assessments of predictive artificial intelligence (AI) can conceal performance disparities across subgroups. Fairness evaluations commonly rely on performance analyses across subgroups. However, some performance metrics are non-collapsible, meaning that the overall population performance value does not equal the weighted average of subgroup specific values.
Objective: To examine the collapsibility properties of commonly reported performance metrics in predictive AI, with a focus on the area under the receiver operating characteristic curve (AUC, also known as c-statistic).
Methods: We investigate the collapsibility of 15 performance metrics, either by expressing each metric as a linear combination of its stratum specific values or, where non-collapsible, by providing a counterexample inspired by Simpson's paradox as a formal disproof.
Results: Five performance metrics (AUC, calibration intercept, calibration slope, expected calibration error, and Nagelkerke R^2) are shown to be non-collapsible, and ten (O:E ratio, logloss, Brier score, accuracy, F1-score, true positive rate, true negative rate, positive predictive value, negative predictive value, and net benefit) are shown to be collapsible. The AUC is shown to be non-collapsible because it decomposes into within- and cross-group AUC terms when subpopulations coexist, such that its overall value may fall outside the range of subgroup specific AUCs.
Conclusions: Non-collapsibility of performance metrics has important consequences for reporting, model appraisal, and fairness evaluation. It can generate spurious differences between subgroup and overall performance, which may mislead fairness evaluations. Explicitly acknowledging and reporting the collapsibility properties of performance metrics improves both the interpretability and transparency of fairness assessments. 

---
# Q-Strata: Hierarchical Bit Allocation for Mixed-Precision Quantization of Mixture-of-Experts LLMs 

**Authors**: Deokjae Lee, Sihun Chu, Hyun Oh Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.30564)  

**Abstract**: Mixed-precision quantization (MPQ) assigns a different bitwidth to each linear layer of a large language model (LLM) to minimize the quantization-induced quality loss under a fixed budget, but Mixture-of-Experts (MoE) models contain these layers in every expert of every MoE block, so the allocation space grows far larger than in a dense model. Existing methods either allocate within each block under a uniform per-block budget, or allocate across blocks through an additive proxy, and neither directly optimizes a model-level objective over the choices that couple the blocks. We propose Q-Strata, a bi-level allocator that ranks within-block assignments with a cheap proxy and allocates across blocks with a model-level objective evaluated on the assembled quantized model. Its inner stage caches a Pareto frontier of candidates per block over finely spaced budgets, leaving the outer stage to set one budget per block instead of a bitwidth for every linear layer. With the search reduced to one budget per block, the outer stage optimizes this model-level objective directly, capturing the inter-block coupling that additive proxies miss. On Mixtral-8x7B-Instruct, Qwen1.5-MoE-A2.7B, and DeepSeek-V2-Lite, Q-Strata consistently achieves lower WikiText2 perplexity than uniform-bitwidth GPTQ and the state-of-the-art MoE MPQ methods MxMoE and GEMQ in the low-bit regime. The code is available at this https URL. 

---
# Preference Shapes Relevance: Cross-component Hierarchical Semantic Alignment for Personalized Generative Retrieval 

**Authors**: Gaoming Zhang, Angqing Jiang, Jianchun Song, Kena Qi, Dayao Chen, Wei Lin, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2608.30553)  

**Abstract**: Generative Retrieval (GR) has emerged as a promising paradigm by mapping queries directly to Semantic IDs (SIDs) with powerful representation capabilities for candidate items. However, existing SIDs derived solely from item content create a semantic gap, failing to align dynamic query intents with static item representations. Furthermore, current generative paradigms rarely model user behavior sequences and are always bottlenecked by the high inference latency of beam-search autoregressive decoding. To address these challenges, we propose $\textbf{C}$ross-component $\textbf{H}$ierarchical semantic $\textbf{A}$lignment for $\textbf{P}$ersonalized generative retrieval ($\textbf{CHAP}$), a novel personalized GR framework from a hierarchical perspective. First, we design a Hierarchical Semantic Alignment module to align query's latent space with item's quantization path and synchronize multi-granular semantics. Second, we construct a personalized GR framework that models user behavior by synergizing discrete SIDs for structural guidance and continuous representations for fine-grained semantic refinement. Notably, we introduce a Residual Cascading Generation mechanism to restrict the costly multi-step Transformer Decoder to a single-pass inference, boosting inference throughput while mitigating information loss. Extensive experiments on three public datasets, one proprietary industrial dataset, and online A/B tests demonstrate CHAP's superiority, validating the effectiveness and practical value of our approach. The code is publicly available at this https URL. 

---
# TSExplorer: An interactive data annotation and exploration tool for time-series data 

**Authors**: Einari Vaaras, Manu Airaksinen, Okko Räsänen  

**Link**: [PDF](https://arxiv.org/pdf/2608.30514)  

**Abstract**: We present TSExplorer, a cross-platform tool for interactive annotation and exploration of time-series data. The tool enables users to inspect high-dimensional datasets through multiple complementary 2D visualizations derived from high-dimensional feature representations. TSExplorer is designed as a general-purpose research tool supporting a wide range of workflows, including exploratory data analysis, annotation of unlabeled or partially-labeled datasets, comparison of feature representations, and post-hoc inspection and refinement of existing labels with interactive visual feedback. 

---
# Trajectory-Initialized Neural Double Q-Routing for Large-Scale Overhead Hoist Transport Systems 

**Authors**: Cheng Gu, Qiusheng Zhao, Anbang Liu, Shaochong Lin, Max Z.J. Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.30512)  

**Abstract**: Large-scale industrial robot fleets share constrained physical infrastructure, making vehicle travel times dependent on safety separation, intersection access, downstream blocking, and station contention. We study this problem in overhead hoist transport (OHT) systems, a representative ceiling-mounted material-handling system used in semiconductor fabs. Static shortest-path routing cannot account for these time-varying traffic costs, whereas tabular Q-routing adapts online but learns each destination--node--action value independently, limiting information sharing across sparsely visited routing contexts and making startup behavior sensitive to inaccurate value estimates. We propose Neural Double Q-routing, which replaces destination-indexed tables with a shared state--action value network. The network is warm-started through return-to-go regression on mixed simulator-generated routing trajectories and then refined online using Double-Q updates, local congestion correction, and event-stratified structured replay. Across nine matched fleet-size--arrival-rate settings with 100, 150, and 200 OHTs, the proposed framework reduces mean completion time relative to tabular Double Q-routing by $0.8\%$--$8.8\%$. It achieves the lowest mean completion time among all compared methods in the six 150- and 200-OHT settings, whereas Dijkstra remains best in the three 100-OHT settings. Completed-task counts remain within $1\%$ of tabular Double Q-routing in eight of nine settings, and 95th-percentile completion time decreases in eight settings. In two matched startup scenarios, offline initialization increases the number of completed tasks by up to $23\%$ and reduces tail completion time by up to $15\%$. 

---
# Lot Machine: Multimodal Lot Extraction from Auction Catalogs 

**Authors**: Mathias Zinnen, Alisha Mund, Sabine Lang, Lukas Hüttner, Thomas Gorges, Vincent Christlein  

**Link**: [PDF](https://arxiv.org/pdf/2608.30510)  

**Abstract**: For provenance research and art market studies, auction catalogs are an essential resource to trace specific objects over time and space. While historical auction catalogs follow established domain conventions, their internal formatting remains highly variable, and their large-scale analysis is currently restricted by the lack of machine-readable representations of the auction lots. We propose a pipeline to automatically extract structured lot-level metadata from German Sales, a large database of historical auction and sales catalogs from the 19th and 20th centuries. Using a manually annotated test set of representative catalog pages, we evaluate Vision-Language Models (VLMs) under varying prompt strategies and constrained decoding frameworks. To reflect the practical constraints faced by cultural heritage institutions, including budget, compute resources, and data privacy requirements, we benchmark the methods across different deployment modes ranging from commercial providers to locally hosted, quantized models. We find that commercial endpoints establish the performance ceiling, while institutional gateways offer a viable, privacy-preserving alternative. Local deployments remain feasible, but strictly require enforcing the output structure during generation to guarantee a valid JSON format. While varying degrees of human-in-the-loop correction are still necessary, this work demonstrates that a VLM-based pipeline can successfully unlock historical auction catalogs for large-scale automated analysis. 

---
# Tensor Methods for Language Models: From Token Representation to Training, Adaptation, Inference, Compression, and Interpretability 

**Authors**: Matvei Tarasov, Salman Ahmadi-Asl, Andre L. F. de Almeida, Andrzej Cichocki  

**Link**: [PDF](https://arxiv.org/pdf/2608.30505)  

**Abstract**: Large language models (LLMs) are built from structured high-dimensional objects such as token representations, weights, adaptation updates, caches, and activations, whose multilinear structure is underexploited by the conventional matrix-centric view. Tensor decompositions and tensor networks provide a principled algebraic language for this structure, yet the literature often treats them as isolated compression mechanisms. This survey organizes tensor methods for LLMs through two complementary views: a seven-stage lifecycle taxonomy covering tokenization, embeddings, pre-training, adaptation, compression, inference, and interpretability, and a component view covering embeddings, attention, and feed-forward networks. We provide unified notation and theoretical foundations, analyze tensorization strategies for individual Transformer components, and compare methods at each lifecycle stage while making differences in evaluation protocols and model scales explicit. We further connect tensor methods to neighboring efficiency techniques and probabilistic tensor networks. Finally, we synthesize open challenges and introduce $\rho_{\rm gap}$, a metric for the compression-realization gap between theoretical memory reduction and measured system-level speedup. By treating tensorization as a common structural principle, the survey provides a structured entry point to tensorized language models and clarifies when parameter savings can plausibly translate into memory efficiency, computational efficiency, or interpretability. The GitHub page dedicated to this paper is accessible at \href{this https URL}{this https URL}. 

---
# Measuring Memory and Generalization as Separable Geometric Channels: The Topo^2 Framework 

**Authors**: Zhanbo Zhang, Ming Liu, Qing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30487)  

**Abstract**: Deep networks trained on noisy labels simultaneously generalize on clean data and memorize flipped labels. These are usually conflated as pressures on one capacity. We present Topo^2, a measurement framework that makes them causally separable, measurable, and law-governed. Persistent-homology H1 structure of the representation space separates into a within-class manifold channel (a function of the training stopping point) and a cross-class channel (a monotone readout of memorized flipped samples). An intervention, the FM0 prescription (zero loss on flipped samples from epoch 0), reaches each setting's generalization ceiling while memorizing essentially nothing. Within the framework we establish a law set with graded evidence: (L2) FM0 separation prescription (9/9); (L1) the within-channel as a training-position function (mid-rise 6/6; convergence-back CIFAR 3/3, SVHN 2/3); (L3) a ring-construction identity (definitional, not a law); and TLS (memory-generalization topological layering): memory is causally additive, anchored (silencing clean collapses the representation), invertible (stripping memory restores near-ceiling generalization), and quantitatively billable (the memorization cost law, effective slope coefficient C ~ 0.38 at the reference capacity: CIFAR-10 0.3801 / SVHN 0.3806 / CIFAR-100 0.384 / VGG 0.3715, capacity-dependent in general and traced to clean-sample feature displacement). We also publish the framework's boundaries: a falsification ledger of nine dead ends, and an instrument-vindication section that excludes six families of global statistics as explanations of the within-channel. The framework turns "memorization" from an ill-defined capacity into a measurable, separable, invertible topological layer. 

---
# ImageEval 2026: Culturally Grounded Arabic Multimodal Evaluation 

**Authors**: Samir Abdaljalil, Hunzalah Hassan Bhatti, Ahlam Bashiti, Farina Amir, Md Arid Hasan, Basel Mousi, Nadir Durrani, Fahim Dalvi, Zien Sheikh Ali, Erchin Serpedin, Hasan Kurban, Mustafa Jarrar, Shammur Absar Chowdhury, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2608.30475)  

**Abstract**: We present an overview of the ImageEval 2026 shared task on culturally grounded Arabic multimodal evaluation. It includes two tasks: (i) AynVQA, covering spoken visual question answering and image-grounded hallucination detection in English and Modern Standard Arabic (MSA), and (ii) CRAI-Bench, evaluating the cultural accuracy of text-to-image generation. A total of 14 teams participated in the test phase, with 12 teams submitting system description papers. Participating systems used a range of approaches, including zero-shot prompting, fine-tuning of vision-language models, speech-recognition pipelines, ensembling, and score calibration. We describe the task setup, datasets, evaluation procedure, and participating systems, and summarize the main results across the different tracks. All datasets and evaluation scripts from the shared task are released to the research community. The shared task highlights the challenges of culturally grounded multimodal evaluation, particularly for Arabic speech and image-text reasoning. 

---
# Enhancing Low-Resource Language Reasoning via High-Resource Language Feature Transfer 

**Authors**: Minju Song, Hyeon Hwang, Junhyun Lee, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30462)  

**Abstract**: Large language models exhibit substantial performance variation across languages, even when solving semantically equivalent tasks. Existing analyses often treat this phenomenon as an observational disparity caused by differences in pretraining data, tokenization, or benchmark coverage. We study a complementary hypothesis: high-resource languages (HRLs) may more reliably elicit latent computations useful for task-specific (i.e. mathematical) reasoning, while lower-resource languages (LRLs) may under-activate those computations despite expressing the same task. To test this hypothesis, we introduce a mechanistic intervention framework for identifying and transferring task-relevant sparse latent features across languages. Using sparse autoencoders over residual-stream activations, we isolate features enriched in successful HRL task-specific reasoning while filtering out source-language and generic-generation features. We then construct steering directions from these features and inject them during LRL inference. The resulting interventions test whether the selected features are functionally involved in the observed reasoning gap: suppressing them should impair source-language reasoning, while activating them should partially recover target-language reasoning beyond random and non-task controls. Our framework reframes some cross-lingual reasoning gaps as failures of mechanism elicitation rather than capability absence, and offers a causally testable route to feature-mediated transfer without translation, fine-tuning, or changing the user-facing language. 

---
# Lies We Can See: Joint Verbal and Non-Verbal Deception by VLM Agents in Embodied Social Interactions 

**Authors**: Jaewoo Ahn, Junseo Kim, Hyunseo Kim, Heeseung Yun, Jaehyeon Son, Zsolt Kira, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.30428)  

**Abstract**: Strategic deception by LLM and VLM agents has emerged as a central AI alignment and safety concern. Social-deduction games (where each player holds a hidden role and communicates with others to deduce identities) serve as the canonical testbed, particularly in multi-agent settings. Existing testbeds, however, are text-only and run on a single fixed agent configuration, missing the non-verbal sensorimotor channels treated as core by deception taxonomies and leaving it ambiguous whether an observed behavior reflects the underlying model or the surrounding harness. We introduce MineAmongUs, a 3D multimodal Among Us sandbox where imposter agents must deceive crewmates through joint verbal and non-verbal action. We also propose ARIA, a configurable VLM-agent harness that exposes five cognitive-component ablation axes; and an atom- and arc-level annotation scheme grounded in deception taxonomies and operationalized at scale by an LLM-as-a-Judge reaching near-human atom-labeling agreement. Empirical results show that VLM agents pursue imposter wins through joint verbal and non-verbal deception, with non-verbal channels emerging as the more decisive winning contributors across both harness ablation and cross-VLM evaluation. Taken together, our work opens a new path for embodied VLM-agent alignment research. 

---
# Towards Cognitive Process-Aware Proactive Writing Support 

**Authors**: Masahiro Yoshida, Atsuya Kobayashi, Kei Tateno, Xiang 'Anthony' Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.30424)  

**Abstract**: Large language models can support writing, but existing tools require users to explicitly articulate prompts-particularly burdensome in creative writing, where intentions are often ambiguous. Proactive support that infers users' needs from writing interactions could alleviate this burden, but raises two challenges: determining what support to provide and when to intervene. This work focuses on the former. We hypothesize that Flower and Hayes' cognitive process theory of writing-which characterizes writing through six cognitive processes-offers an interpretable bridge between observable writing behavior and appropriate support types. Through a formative study and literature review, we identify 14 writing support types associated with these cognitive processes, along with characteristic interaction behaviors linked to each process. We then instantiate this framework in AToM CoWriter, which infers support needs from writing interactions and document context. Two within-subjects studies (N = 21) provide initial evidence that this approach improves expressiveness and idea exploration, and that cognitive process inference increases engagement with proactive suggestions. These findings suggest that cognitive processes can provide a promising basis for support selection in proactive writing systems. 

---
# ObjectSplat: Improving Mesh Fidelity and Interactivity for 3D Scenes via Object-Level Mesh Splatting 

**Authors**: Minhas Kamal, Hiranya Garbha Kumar, Mahedi Kamal, Balakrishnan Prabhakaran  

**Link**: [PDF](https://arxiv.org/pdf/2608.30423)  

**Abstract**: Splatting-based algorithms reconstruct photorealistic, real-time-renderable, and mesh-exportable 3D scenes from regular images, but they represent a scene as a single monolithic field. Therefore, the reconstruction has no object-level structure, leaving it infeasible for downstream editing or interaction. Moreover, regions that are never directly observed in the input scans are contaminated by the surrounding texture and left uncorrected, capping both mesh fidelity and novel-view synthesis. We propose a decompose-before-reconstruct approach: we segment the instances out of every frame, consider the remaining as background and inpaint it, reconstruct each instance and the background independently with mesh splatting, and compose them into a single scene. Our method significantly improves mesh fidelity (over a 5\% gain in F-score) and novel-view synthesis, while supporting object-wise modifiability and interactivity. The code will be made publicly available. 

---
# Whole-Slide Image Analysis under Realistic Few-Shot Annotation Protocols 

**Authors**: Tiffanie Godelaine, Maxime Zanella, Karim El Khoury, Benoit Macq, Christophe De Vleeschouwer  

**Link**: [PDF](https://arxiv.org/pdf/2608.30420)  

**Abstract**: Automating the analysis of whole-slide images has high clinical value, since characterizing cancers requires examining them in detail. Such analysis increasingly relies on vision-language models that provide patch-level zero-shot predictions. However, these predictions remain noisy and must be refined with a few annotations. A promising paradigm for this refinement is few-shot transduction. Rather than treating each patch independently, these methods leverage the relations between patches, together with a few annotations, to refine all predictions jointly. However, current transductive methods are evaluated under conditions that overlook key properties of whole-slide images: (i) datasets consist of independent patches extracted from multiple slides, ignoring the complex tissue organization; (ii) datasets are mostly balanced, whereas a single whole-slide image exhibits severe class imbalance, with several classes absent; and (iii) annotations are sampled at random, without reflecting how a pathologist annotates a limited number of regions. To align the transduction paradigm to realistic whole-slide settings, we introduce the following contributions. First, we propose SlideCRF, which adapts conditional random fields for whole-slide images by combining spatial and biological cues while accounting for classes that may be absent from a given slide. Second, we provide a set of realistic annotation protocols, based on spatially localized clicks and scribbles, modeling different pathologist interactions, such as the iterative correction of model errors. Across four datasets, we show that SlideCRF outperforms current transductive methods in macro F1, improving over the zero-shot predictions by +24.2% and +37.5% with one and 16 clicks per present class, respectively. 

---
# SePArate: Segmenting Patterns from Defects in Wafer Manufacturing Using Weak Supervision 

**Authors**: Dain Kwon, Changmin Shin, Sunjong Park, Kanghyun Choi, Hyeyoon Lee, Jaewon Jang, Minseok Choi, Jinho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.30410)  

**Abstract**: In semiconductor manufacturing, defect analysis is essential, but manual inspection cannot scale. However, existing automated inspection methods remain insufficient for root-cause analysis and process optimization. To this end, we present SePArate, a weakly supervised wafer defect segmentation method. SePArate enables pixel-level separation of patterns by leveraging only image-level annotations. It consists of a three-phase training: encoder pretraining, knowledge transfer to learn spatial cues, and training on synthetic mixed-defect data for accurate segmentation. Experiments demonstrate that SePArate outperforms the baselines. 

---
# ImageCAS-X: a dataset and benchmark for coronary artery segmentation and centerline extraction in coronary CT angiography 

**Authors**: Kit M. Bransby, Esther Øksnebjerg, Kristoffer Kjær, Jacob Kirkeby, Yasmin El Youssef, Aïda Jiménez, Philip R. Pedersson, Martina C. de Knegt, Klaus F. Kofoed, Rasmus R. Paulsen  

**Link**: [PDF](https://arxiv.org/pdf/2608.30404)  

**Abstract**: Accurate segmentation of the coronary vessel lumen is a prerequisite for quantitative assessment of atherosclerotic plaque and perivascular adipose tissue in coronary computed tomography angiography (CCTA). Cardiologists rely on semi-automated methods for this task because manual vessel tracing and segmentation are labour-intensive. Although many automated methods have been proposed, their validation remains limited by the lack of large, high-quality publicly available datasets. We provide a new dataset of voxel-wise annotations of the vessel lumen and coronary segments, alongside centerlines, and mesh surfaces for 800 scans from the publicly available ImageCAS dataset. Using this dataset, we benchmark established lumen segmentation methods against inter-observer variability, stratifying performance by disease, image quality, coronary dominance, coronary segment, vessel diameter, and lumen attenuation. These labels allow segmentation accuracy to be described in anatomical and clinical context rather than reported as a single aggregate score. The dataset supports the development and validation of methods for lumen segmentation, plaque and perivascular quantification, and haemodynamic modelling. 

---
# SemPOI-RL: Aligning LLM Semantic Reasoning for Interpretable Out-of-Town POI Sequential Generation 

**Authors**: Yunqi Liu, Yang Zhang, Ruixing Zhang, Liangzhe Han, Yi Qiao, Tongyu Zhu, Leilei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.30399)  

**Abstract**: Large language models (LLMs) exhibit strong semantic reasoning and open-ended generation abilities, but aligning these abilities with structured sequential generation remains challenging. This challenge is particularly evident in out-of-town (OOT) POI sequence generation, where a model must infer transferable travel intent from a user's hometown behaviors, adapt to cross-city interest drift, and generate a coherent destination trajectory under structural constraints. Existing approaches either rely on latent ID-based transfer with limited interpretability or directly use LLMs for sequence generation without explicitly grounding inferred semantics into position-aware predictions. To address this gap, we propose SemPOI-RL, a framework that aligns LLM semantic reasoning with structured sequence generation for interpretable OOT recommendation. Specifically, we first fine-tune an LLM to infer destination-oriented travel styles from users' hometown trajectories, using natural language as an interpretable semantic intermediate. We then introduce a Semantic POI Alignment Module (SPAM) to ground these inferred styles into a style-conditioned masked autoencoder for position-aware trajectory generation. Finally, we apply reinforcement learning with recommendation-oriented rewards to align LLM-generated styles with downstream sequence quality. Experiments on two real-world datasets show that SemPOI-RL consistently outperforms both traditional recommenders and direct LLM baselines, while providing interpretable style attribution across different phases of a trip. The code is available at this https URL . 

---
# TopGQ: Fast GNN Post-Training Quantization Leveraging Topology Information 

**Authors**: Dain Kwon, Kanghyun Choi, Hyeyoon Lee, Sunjong Park, Seoyong Lee, Sukjin Kim, Jinho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.30394)  

**Abstract**: Existing GNN quantization methods suffer from considerable quantization overhead, which severely limits their practical usage in real-world scenarios. To this end, we present TopGQ, an accurate post-training GNN quantization framework, alleviating redundant quantization overhead. We propose dual-axis scale absorption, which enables activation quantization along both the outer and inner dimensions by merging one into the adjacency matrix. On top of that, we introduce TopPIN, a proxy for nodes' local structure, and use it to group nodes with similar topology during quantization. Experimental results show that TopGQ reduces quantization time by an order of magnitude while preserving accuracy. 

---
# Using Grounded Theory for Agent Behavior Analysis at Scale 

**Authors**: Zhuoran Lu, Yangyang Yu, Zhuoyan Li, Yibo Meng, Nan Jiang, Chengxi Zang, Jie Gao, Ziang Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2608.30391)  

**Abstract**: Understanding agent behavior requires methods that scale to thousands of trajectories and surface new patterns in long, often unfamiliar tasks where pre-built classifiers fall short. We propose to bring grounded theory into agent trajectory analysis: a six-decade-old qualitative method from the social sciences, with a principled saturation criterion and an auditable trail from data to theory. We propose AutoTraceGT (Automated Trace analysis through Grounded Theory), the first multi-agent pipeline that automates grounded theory on agent trajectories. It iteratively performs open, axial, and theoretical coding until saturation, producing a behavioral taxonomy tailored to each task. Across six trajectory corpora, AutoTraceGT produces codebooks that recover 73-91 percent of the failure modes in human-annotated taxonomies and surface additional patterns that those taxonomies miss. The emergent theoretical narrative aligns with prior expert accounts. Used as a deductive feature space, the codebook outperforms zero-shot and few-shot LLM baselines on downstream failure prediction. These results suggest Grounded Theory offers a scalable analytic tool for ML researchers and agent developers studying what agents actually do. 

---
# PRISM: Predictive Recomposition via Semantic Latent Decomposition for View-invariant Video Representation Learning 

**Authors**: Youngchae Chee, Hosu Lee, Sungjune Park, Junho Kim, Yong Man Ro  

**Link**: [PDF](https://arxiv.org/pdf/2608.30388)  

**Abstract**: Cross-view video representation learning aims to capture viewpoint-invariant action semantics despite substantial appearance changes across egocentric and exocentric videos. However, existing methods encode each video as a unified embedding, where view-invariant and view-variant semantics inevitably entangle under co-occurrences - a failure mode we show persists even in cross-view methods explicitly trained for view-invariance. Our key insight is that a view-invariant feature is truly disentangled when it can be sufficiently recomposed with an arbitrary view-variant feature while preserving their independent semantics. Building on this, we propose PRISM, that decomposes video into view-invariant and view-variant latents and recompose them under language supervision encouraging clean decomposition of the two streams. PRISM achieves state-of-the-art results on EgoExo4D, EgoExoLearn, AE2, even surpassing in-domain models under zero-shot setting. Code is available at this https URL. 

---
# DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving 

**Authors**: Yanqi Yu, Pingwei Sun, Jianchao Tan, Tao Zhang, Yuchen Xie, Xunliang Cai, Yao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.30386)  

**Abstract**: Hybrid linear-attention architectures have recently scaled to large open-weight models, offering quality competitive with full attention while substantially reducing key/value (KV) cache growth. However, their in-place recurrent-state updates complicate cache management: prefix reuse requires state checkpoints alongside full-attention KV, while storing state checkpoints in full increases memory pressure, leading to more evictions and repeated prefill. By analyzing the decay structure of Gated DeltaNet (GDN) and Kimi Delta Attention (KDA), we find that different heads and channels retain prefix information over markedly different timescales, which we term \emph{retention horizons}. This variation suggests substantial compression potential in persistent state checkpoints. Building on this observation, we introduce \emph{Decay-Aware State Compression} (DASC), which derives retention horizons from model weights, selects long-horizon state units, and packs them into a ragged state checkpoint layout. To integrate efficiently with tensor-parallel inference engines, DASC furtherly balances compressed state checkpoints across TP ranks. On reuse, DASC either zero-fills omitted units or refreshes them from a bounded suffix with additional compute cost. Across retrieval and end-to-end reasoning benchmarks on Kimi-Linear, conservative DASC configurations remain close to full caching while compressing KDA recurrent state checkpoints by $2.63\times$. Under fixed state checkpoint memory budgets, the resulting capacity gains reduce mean Time to First Token (TTFT) by 42.6\% and improve input throughput by 68.4\%. At larger compression ratio, suffix refresh recovers much of the accuracy lost to more aggressive omission, at the cost of additional replay computation. Qwen with GDN exhibits a similar quality--efficiency trend, showing that DASC extends from channel-wise KDA to head-wise GDN. 

---
# PAVE: Predictive Alignment and Value-Guided Evolution for World-Action Policies 

**Authors**: Botong Zhao, Fang Yu, Senhua Zhu, Xinyuan Chen, Yue Lu  

**Link**: [PDF](https://arxiv.org/pdf/2608.30378)  

**Abstract**: Direct vision-language-action policies generate continuous robot actions efficiently, but standard behavior cloning leaves two complementary gaps: their representations are not explicitly required to describe how the scene evolves over multiple time scales, and deployment trajectories of unequal quality are often reused without separating useful dynamics from undesirable behavior. We introduce \method, a direct world-action policy that combines outcome-agnostic predictive learning with outcome-aware policy improvement. \method first retains a local fixed-offset JEPA objective and adds trajectory-relative multi-horizon transition alignment at 25%, 50%, 75%, and 100% of the remaining episode. These training-only targets require the current policy representation to preserve both local physical changes and longer-range task progress, without supplying explicit future tokens to the action head. \method then trains an independent distributional value critic on cumulative deployment trajectories, computes action-chunk-aligned $N$-step advantages, and converts them into positive, negative, or null text conditions for a flow-matching actor. Thus, every valid trajectory can teach what physically happened, while the actor is deployed only under the condition associated with relatively better actions. The multi-horizon predictor and critic are removed from online execution, preserving direct action generation from the current observation, language instruction, and proprioception. \redclaim{Across the three simulation benchmarks, \method achieves the strongest overall performance while preserving the direct actor's online execution path.} 

---
# Beyond Ranking Accuracy: Evaluating LLM-Cited Feature Rationales for Next Basket Repurchase Recommendation 

**Authors**: Yanan Cao, Anay Dombe, Murali Mohana Krishna Dandu, Shreeranjani Srirangamsridharan, Sinduja Subramaniam, Yogananth Mahalingam, Evren Korpeoglu, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2608.30333)  

**Abstract**: Next-basket repurchase recommendation is commonly formulated as a ranking task: given a customer's purchase history, the system ranks previously purchased items that may be needed again. In production settings, however, ranking accuracy is only one component of recommendation quality. Customers may also benefit from concise evidence about why an item is recommended now. Large language models (LLMs) offer a potential way to surface such evidence through feature-based, human-readable rationales grounded in interpretable behavioral signals. We construct repurchase features spanning cadence, frequency, recency, user behavior, and item popularity, and evaluate LLMs on two public grocery datasets and one proprietary retail dataset. We investigate (1) whether off-the-shelf LLMs can use these features as next-basket scorers relative to heuristic and supervised rankers, and (2) whether LLM-cited features carry outcome-grounded ranking signal. For the latter, we compare LLM-cited features with model-specific attribution methods under a cross-model feature-masking protocol that measures ranking degradation after masking selected features. Our results show that LLM scores are not competitive with supervised rankers, suggesting that off-the-shelf LLMs should not be used as standalone repurchase recommenders. However, changes in prompt and evidence representation can improve outcome-grounded feature-masking results in some settings even when ranking performance does not improve; the effect is dataset-dependent and does not consistently match attribution baselines. These findings suggest a practical role for LLMs as validated explanation components rather than primary rankers, with rationale quality evaluated separately from ranking accuracy. 

---
# Parallel Time-Band Mixing with Learned Observation-Adding for Robust ASR Front-Ends 

**Authors**: Xingyu Shen, Runze Wang, Wei-Ping Zhu, Benoit Champagne  

**Link**: [PDF](https://arxiv.org/pdf/2608.30326)  

**Abstract**: Speech enhancement is often used as a front-end for robust ASR, yet recurrent temporal and cross-band modules introduce sequential dependencies that reduce parallel efficiency. In this paper, we present a sequence-parallel band-split enhancement front-end built on a Parallel Time-Band Mixer (PTBM) block that eliminates within-block recurrent unrolling. PTBM integrates intra-band temporal mixing and per-frame cross-band attention within a unified parallel architecture, enabling efficient contextual modeling across both time and frequency dimensions. The system retains the mask-plus-residual reconstruction interface and introduces learned Observation-Adding (LOA) to suppress ASR-sensitive artifacts without development-set tuning. Experiments on DNS Challenge and CHiME-4 with frozen Whisper back-ends show that the proposed front-end consistently reduces word error rate relative to recurrent band-split baselines while requiring only 0.96 M parameters and 0.58 GMAC/s for the front-end network. 

---
# Beyond Token-Level Guidance: Inference-Time Alignment of Specialized LLMs via Cross-Family Representation Steering 

**Authors**: Jin Gan, Xin Li, Jun Luo  

**Link**: [PDF](https://arxiv.org/pdf/2608.30319)  

**Abstract**: Large language models (LLMs) finetuned for specialized domains represent crucial high-impact applications. Inference-time alignment improves safety degraded from specialization finetuning without requiring substantial computational resources, complementing finetuning-based methods with an easy-to-use, plug-and-play solution. However, existing inference-time methods fail to reliably improve safety without disrupting domain capability. We identify the root cause as complementary expertise orthogonality: specialized base models and general-domain guidance models have orthogonal competencies, making the guidance signal unreliable for specialized generation. This primarily manifests as stop token interference, where the guidance model's tendency toward continuation overrides the base model's decision to stop, burying correct answers under guidance-induced continuation. To address this problem, we propose CREST, an inference-time alignment method that steers base model hidden representations using safety directions extracted from a guidance model of any family, avoiding token-level structural limitations entirely. CREST improves safety where specialization has weakened it while preserving both domain-specific capability and the safety of already well-aligned models, outperforming baselines by up to 22.2\% on safety benchmarks. Our code is available at: this https URL. 

---
# Online Estimation of Dynamic Origin-Destination Matrices Using Reinforcement Learning with Link-Flow Propagation Guidance 

**Authors**: Donggyu Min, Dong-Kyu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.30317)  

**Abstract**: Online dynamic origin-destination (OD) matrix estimation (DODE) calibrates time-dependent OD demand to reproduce observed link-flow trajectories. In online, OD demand should be estimated from current observations and propagated network states while subsequent observations and stochastic dynamic network loading (DNL) outcomes remain uncertain. Recently, reinforcement learning (RL) has emerged as a promising alternative, reducing computational burden by replacing iterative algorithms while being applicable to stochastic environments. However, because the policy is trained offline and deployed online, it must handle varying target link-flow trajectories; since each target trajectory defines the link-flow error used in the reward, the same OD demand vector can require different adjustments, making conventional scalar feedback ambiguous. To address this gap, this study proposes LFPG-RL, which integrates link-flow propagation guidance (LFPG) into proximal policy optimization (PPO). LFPG combines link-flow error sensitivities with the contribution of each OD-time demand component to simulated link flows, transforming aggregate mismatch into OD-specific advantage shaping for PPO actor updates. At deployment, the policy requires only a single forward pass. LFPG-RL is developed and evaluated on 250 weekday trajectories of 15-min link-flow data from a Melbourne arterial network modeled by a link transmission model with stochastic route choice. On held-out trajectories, LFPG-RL achieved an RMSE of 4.69, MAPE of 20.15%, and Pearson correlation of 0.995. These results support the contention that our method is a more efficient and accurate online OD demand calibration method compared to existing ones. 

---
# One AI Signal, Many Human Judgments: A Bayesian Cascade Analysis of AI-based Credibility Indicators in Online Information Spread 

**Authors**: Zhuoran Lu, Weilong Wang, Yangyang Yu, Xinru Wang, Zhuoyan Li, Zhiwei Liu, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2608.30311)  

**Abstract**: Social media platforms increasingly use AI-based credibility indicators to help users judge misinformation. Unlike individual human-AI decision-making, these indicators are embedded in information spread: users see both an AI prediction and earlier judgments shaped by the same AI, and their own judgments may then enter the public history. Yet how to analytically characterize this process remains under-explored. We therefore introduce a social-learning lens for this setting by extending the classical Bayesian cascade model with the AI indicator as a shared public signal. The resulting Gateway condition compares the evidence from the AI prediction with users' private impressions. Through this view, we show that AI changes what public history means. Crowd agreement may reflect accumulated independent human evidence, or repeated dependence on the same AI prediction. This creates a preservation-correction trade-off: stronger reliance on AI can preserve correct predictions, but can also lock in incorrect ones by blocking corrective private impressions. We calibrate the model using human-subject data on news veracity judgments. Although the AI outperforms human users, the average user weights it below her own impression but above several peer judgments, while individual users vary from discounting the AI to relying on it enough to cascade. Simulations show that over-reliance on a weak AI is especially harmful, and that diversifying AI signals across users can better keep the crowd informative. We conclude with implications for understanding human-AI interaction in information spread and designing misinformation interventions. 

---
# Tail-Replay: Escaping the Curse of Linear Attention in Prefix Caching for Hybrid LLMs 

**Authors**: Yirui Liu, Ruoling Qi, Xuaner Wu, Penghang Liu, Jian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.30310)  

**Abstract**: Hybrid large language models interleave full-attention layers with linear-attention layers to reduce the cost of long-context inference. This structure complicates prefix caching: full-attention key-value caches are token-addressable, whereas linear-attention layers maintain recurrent states that cannot be rolled back to arbitrary prefix boundaries. Existing hybrid prefix caching methods address this mismatch by storing recurrent-state checkpoints. As a result, token-level matches are directly usable only at positions aligned with stored checkpoints, constraining prefix reuse to a discrete set of boundaries. We present Tail-Replay, a prefix caching mechanism that enables unconstrained token-level prefix reuse in hybrid large language models. The key insight is that linear-attention mechanisms such as Gated DeltaNet can be viewed as a structured, lossy compression of the input prefix: gated recurrent updates progressively attenuate the contributions of earlier inputs. Consequently, the recurrent state of a matched prefix can be well approximated by replaying only a short, recent suffix of that prefix. Tail-Replay exploits this property by caching the exact full-attention key-value cache while omitting recurrent-state checkpoints. On a cache hit, it reconstructs the linear-attention states by replaying a short, recent suffix of the matched prefix. As a result, the reuse boundary is determined by the shared tokens rather than by recurrent-state checkpoints. We evaluate Tail-Replay on three Gated DeltaNet-based hybrid models using the LongBench and RULER benchmarks. With only a 5--10\% replay budget, it retains 92.8--99.9\% of full-prefill quality on LongBench and RULER. For serving efficiency, we evaluate time-to-first-token speedups across multiple matched-prefix lengths---8K, 16K, and 32K. The speedup grows with prefix length, reaching $9.1$--$14.3\times$ over full prefill at 32K. 

---
# ScenePilot: Grow-and-Repair Policy for Text-Driven 3D Indoor Scene Generation 

**Authors**: Jiawei Zhang, Hongsong Wang, Pan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.30307)  

**Abstract**: Text-driven 3D indoor scene generation has advanced from dataset-bound layout modeling to open-vocabulary synthesis with large language and vision-language models. Yet existing methods remain limited: one-pass generators often yield geometrically invalid layouts, heavy post-hoc optimization is costly and unstable, and prompt-only planners lack reusable layout priors for functional grouping and object relations. We propose \textbf{ScenePilot}, a retrieval-augmented \textbf{Grow-and-Repair} framework that formulates scene generation as prior-guided incremental growth with learned rectification. Given a prompt, the Hierarchical Retrieval-Augmented Planning (HRAP) module retrieves room-, group-, and anchor-level layout priors to support functional group planning. A text-driven base generator then inserts object groups sequentially, while the Reinforcement Multimodal Repair (RMR) module performs lightweight local correction after each insertion and a final global repair after completion. To train this policy, we construct \textbf{SceneReverse-17k}, a repair-trajectory dataset built by perturbing high-quality 3D scenes in position, rotation, and scale, then using inverse operations as executable rectification targets. The policy predicts structured \emph{move--rotate--scale} actions from rendered views, scene state, retrieved priors, and edit history. By combining HRAP with RMR, ScenePilot offers an efficient alternative to one-shot generation and heavy full-scene optimization, improving physical plausibility, functional coherence, and controllability while preserving diversity. 

---
# CateKV: On Sequential Consistency for Long-Context LLM Inference Acceleration 

**Authors**: Haoyun Jiang, Haolin Li, Jianwei Zhang, Fei Huang, Qiang Hu, Minmin Sun, Shuai Xiao, Yong Li, Junyang Lin, Jiangchao Yao  

**Link**: [PDF](https://arxiv.org/pdf/2608.30295)  

**Abstract**: Large language models (LLMs) have demonstrated strong capabilities in handling long-context tasks, but processing such long contexts remains challenging due to the substantial memory requirements and inference latency. In this work, we discover that certain attention heads exhibit sequential consistency in their attention patterns, which can be persistently identified using a coefficient-of-variation-based algorithm. Inspired by this observation, we propose CateKV, a hybrid KV cache method that retains only critical token information for consistent heads, thereby reducing KV cache size and computational overhead, while preserving the majority of KV pairs in adaptive heads to ensure high accuracy. We show the unique characteristics of our algorithm and its extension with existing acceleration methods. Comprehensive evaluations on long-context benchmarks show that, while maintaining accuracy comparable to full attention, CateKV reduces memory usage by up to $2.72\times$ and accelerates decoding by $2.18\times$ in single-sample inputs, and boosts throughput by $3.96\times$ in batch scenarios. 

---
# BCPPO: Bachelier-Inspired Constrained Proximal Policy Optimization for Tail-Risk-Aware Safe Reinforcement Learning 

**Authors**: Dongsheng Hou, Yanqiao Chen, Yuhan Rui  

**Link**: [PDF](https://arxiv.org/pdf/2608.30283)  

**Abstract**: Expected-cost constraints can still permit rare, high-cost events. Monte Carlo conditional value at risk (CVaR) gradients can be noisy at high confidence, whereas critics that model an outcome distribution add complexity. We propose BCPPO (Bachelier-Inspired Constrained Proximal Policy Optimization), a proximal policy optimization (PPO) method. Separately initialized cost-prediction networks (critics), trained with random sample masks, produce disagreement that marks predictions sensitive to which state-action regions occur in the training data and to critic training. A Bachelier formula for the expected amount above a reference level converts this disagreement into a smooth policy-update penalty. Gradients from this penalty do not alter the critics, so temporal-difference (TD) critic learning is unchanged. A saturation-aware controller adjusts the mean-cost penalty and stops accumulated error from growing while that penalty is clipped. Deployment retains only the policy network. The disagreement penalty is neither a tail-event probability nor a guaranteed error bound, and it provides no safety guarantee. Across 175 runs with shared tasks, costs, budgets, training steps, and evaluation seeds, no comparator attains both higher mean return and lower mean CVaR than BCPPO in any task. On Push1, BCPPO has no lower return and no higher CVaR than every comparator, with at least one strict gain. These results support a practical balance among reward, caution around cost predictions that vary across trained critics, and policy-only deployment. 

---
# Dec-BFTRL: Squre-Root Regret for Decentralized Online Upper-Linearizable Optimization under Separation Access with Application to Continuous Submodular Maximization 

**Authors**: Yiyang Lu, Mohammad Pedramfar, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2608.30271)  

**Abstract**: We study decentralized online optimization of upper-linearizable payoffs over an action set under efficient separation access, with applications to online continuous diminishing-return (DR) submodular maximization. We propose Decentralized Barrier Follow-the-Regularized-Leader (Dec-BFTRL), and evaluate each agent's played action against the average of all local objectives. Each agent maps an internal iterate to a feasible action through an approximate gauge projection, communicates only a cumulative surrogate-gradient dual state, and invokes the local HybridNewton procedure to approximately minimize its post-communication BFTRL potential. For every agent, we achieve expected network-aggregate regret of $\widetilde O(\sqrt{T})$. Over $T$ rounds, each agent uses $T$ neighbor-mixing steps and $\widetilde O(T)$ separation-oracle calls. We give four wrapper instantiations covering three DR-submodular maximization problems. 

---
# Centering before Pruning: Lightweight Geometry Correction for Diversity-Based Visual Token Pruning in LVLMs 

**Authors**: Shunjie Wen, Jaeyeon Lee, Dong-Wan Choi  

**Link**: [PDF](https://arxiv.org/pdf/2608.30263)  

**Abstract**: Large vision-language models (LVLMs) incur substantial inference costs due to their long and highly redundant visual-token sequences. Diversity-based pruning mitigates this cost by selecting token subsets based on pairwise cosine similarity. We find, however, that similarities between raw visual tokens are strongly concentrated in the positive range, limiting their ability to distinguish non-redundant tokens. A natural way to improve this resolution is to center token features before computing cosine similarity. Centering indeed reveals a substantially richer pairwise structure, yet unexpectedly degrades pruning performance when used alone. We show that this apparent contradiction arises because the raw geometry does more than represent pairwise diversity: it also implicitly favors globally distinctive tokens, which tend to contain semantically informative content. Centering better resolves subset diversity but loses this useful token-wise preference, revealing that diversity and distinctiveness are entangled in the raw geometry. Based on this analysis, we propose the \textbf{Cen}tered Geometry \textbf{Prune}r (Cen-Prune), which measures subset diversity using centered cosine similarity while retaining raw-space distinctiveness as a complementary token-wise preference. This lightweight, plug-and-play correction leaves the underlying selection mechanism unchanged and incurs negligible computational overhead. Extensive experiments across multiple image- and video-understanding benchmarks and LVLM architectures demonstrate that Cen-Prune provides robust improvements in overall performance across existing diversity-based pruners. 

---
# Using Prosody to Predict Syntactic Structure 

**Authors**: Junghyun Min, Alex Warstadt, Tamar I. Regev, Tiago Pimentel, Ethan Gotlieb Wilcox  

**Link**: [PDF](https://arxiv.org/pdf/2608.30260)  

**Abstract**: While it is well-established that prosody carries crucial cues for syntactic structure, the degree and nature of correspondence between these two domains remains contested. We investigate the syntax-prosody interface through an information-theoretic lens, quantifying the interaction between prosodic features and syntactic representations as their mutual information. We provide a general-purpose framework for estimating this quantity over large speech-text corpora using multimodal language models. Our framework is structure-agnostic and modular, insofar as it can be used to measure the contributions of individual prosodic features or components of structure. We evaluate the syntax-prosody relationship for two features (word duration and inter-word pauses) across two domains--read audiobooks and spontaneous conversations--both in English. Our results demonstrate that prosody contains measurable syntactic information, with prosodic features reducing syntactic uncertainty in spontaneous conversations by up to 10.2%. Our findings offer new empirical support for several theoretical accounts of the syntax-prosody interface. 

---
# Stratified Consistency Distillation for Natural Language Formalization 

**Authors**: Zhichao Hou, Ferhat Erata, Joe Lilien, MohamadAli Torkamani  

**Link**: [PDF](https://arxiv.org/pdf/2608.30258)  

**Abstract**: Neurosymbolic reasoning has shown promising success in addressing complex reasoning tasks by combining large language models (LLMs) and symbolic solvers. While this approach shows promise, a fundamental challenge remains: improving the accuracy of translations from natural language to logical formulas. Current methods predominantly rely on prompt engineering, which is difficult to scale across different domains and input formats. Drawing inspiration from the success of fine-tuning in other model adaptation and alignment applications, we propose a fine-tuning-based Stratified Consistency Distillation approach: (1) We generate K logical translations per input using a frontier LLM and cluster them by semantic equivalence (2) Based on the entropy level, we apply majority voting (low entropy), LLM-as-a-Judge (medium entropy), or unification/abstention (high entropy), and (3) fine-tune a smaller model using the selected pseudo-labels. Our experiments show significant and consistent improvements in both Pass@K and our novel Equivalent Logical Similarity metrics, demonstrating the potential of advancing logical translation through consistency distillation. 

---
# Beyond Surface Forms: Symbolic Edits as a Test for Logical Reasoning with LLMs 

**Authors**: Ramya Keerthy Thatikonda, Wray Buntine, Ehsan Shareghi  

**Link**: [PDF](https://arxiv.org/pdf/2608.30256)  

**Abstract**: Logical reasoning with large language models (LLMs) is a critical capability, as it reflects a system's ability to correctly deduce hypotheses from a given context using faithful deductive processes. However, LLM reasoning has often been shown to be sensitive to small surface-level variations in problem formulation, raising questions about whether models truly follow the underlying logical structure. Studying this behavior is challenging because the symbolic components of logical problems, such as operators and predicates, are difficult to systematically manipulate in natural language. We introduce a tool-driven framework for generating controlled, label-preserving edits to logical reasoning problems. Our method operates on symbolic representations of first-order logic and constraint satisfaction problem tasks, enabling targeted modifications to logical operators and other structural components before translating them back into natural language. Using this framework, we evaluate various LLMs under cumulative and individual operator edits and analyze their behavior in response to these changes. Our quantitative and qualitative analyses show that LLM reasoning behavior under controlled operator edits is inconsistent, regardless of model size or family: models sometimes adapt correctly to structural changes but often fail to track their logical consequences. The results from this automated stress test enable an evaluation of language models across different dimensions and help measure the reliability of their reasoning. 

---
# Motus2: A Self-Evolving General World Model for Dexterous Manipulation 

**Authors**: Hongzhe Bi, Zihao Zhou, Yihang Tang, Jingrui Pang, Shuhe Huang, Haitian Liu, Runqing Wang, Shuai Huang, Yichen Wang, Yiming Cheng, Ruowen Zhao, Zhenghua Li, Hengkai Tan, Xiaolong Liu, Jinhui Wan, Jiabao Liu, Min Zhao, Fan Bao, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.30237)  

**Abstract**: General embodied agents should perceive, predict, act, evaluate, and improve within a unified system. World models have shown great promise in building such agents, yet existing models typically append an action output head to a world simulator, without coupling them into a closed decision-and-learning loop for policy improvement. We present Motus2, a self-evolving general world model for dexterous manipulation. Motus2 advances world modeling through model scaling and data scaling. For model scaling, a single model with shared weights exposes three control interfaces: a policy (world-action model), a simulator (action-conditioned world model), and an evaluator (value model). The policy proposes candidate action chunks, the simulator predicts their visual consequences, and the evaluator assesses the predicted outcomes. Their coupling forms a closed decision-and-learning loop for policy improvement. This formulation uses curated expert demonstrations for action learning, while failed and suboptimal interactions provide valuable evidence for dynamics modeling and value learning. For data scaling, Motus2 progresses from large-scale monocular egocentric data to synchronized stereo egocentric data, followed by robot-domain adaptation with robot trajectories and supplementary human-robot alignment data. Motus2 further studies global-autoregressive and hybrid-memory extensions of its sliding-window context, adds tactile feedback for contact-aware control, and is instantiated on a fully biomimetic platform with stereo vision, dual arms, dual dexterous hands, and tactile sensing. Together, egocentric data scaling and closed-loop general world model scaling provide a general path toward self-evolving dexterous manipulation. 

---
# The Differential Reasoning Router: Operationalizing Cost-Aware LLM Annotation in E-commerce 

**Authors**: Cheng Lyu, Jingyue Zhang, Vinny DeGenova, Mengwei Li, Yuanli Pei  

**Link**: [PDF](https://arxiv.org/pdf/2608.30224)  

**Abstract**: Large Language Models (LLMs) are increasingly used to annotate structured product data in e-commerce, but early deployment often begins as a cold-start problem: only limited pre-launch labels are available, the value of expensive reasoning is unknown, and human review is needed before the system can be trusted at scale. This challenge is especially common in rule-based annotation workflows, where each item must satisfy multiple business rules and both model errors and ambiguous rule boundaries affect final decisions. We introduce the Differential Reasoning Router (DRR), a cost-aware framework for cold-start LLM annotation that jointly optimizes model selection and human escalation. Rather than treating a reasoning model as a default fallback, DRR estimates separate success probabilities for a direct model and a reasoning model at both the sample and business-rule levels, enabling adaptive routing: easy cases are handled directly, reasoning is reserved for cases where it is expected to improve the decision, and likely double-failure or rule-disagreement cases are escalated to human annotators. The resulting labels provide targeted ground truth for prompt engineering, supervised fine-tuning, calibration, and rule refinement, enabling a gradual shift from human-heavy cold-start annotation toward high-confidence automated routing. In a production e-commerce workflow, DRR reaches accuracy parity with the strongest confidence-based router while achieving more than 60\% reasoning-token cost savings. 

---
# Label Semantic Expansion via Label Guided Neural Topic Modeling 

**Authors**: Haojia Zheng, Yuyin Lu, Juntian Huang, Fan Ou, Yanghui Rao, Haoran Xie, Fu Lee Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30216)  

**Abstract**: Topic models are widely used for content analysis, where users often analyze corpora around predefined labels rather than unordered latent topics. Existing label-aware topic models mainly follow a labels-for-topics perspective, using labels to guide topic learning, while the learned topics are not directly usable for label-centered analysis. We explore the reverse topics-for-labels perspective and instantiate it as Label Semantic Expansion (LSE), which enriches sparse label representations with corpus-grounded descriptive topic words. To exploit topics in LSE effectively, we propose a Label-Guided Neural Topic Model (LGNTM), which learns dedicated label-aligned topics, grounds them in lexical and document semantic spaces, and preserves consistency between topic structures and label structures. Experiments on label-topic alignment, label expansion, topic quality, and downstream classification demonstrate strong overall performance across complementary evaluation dimensions. 

---
# SIR: Self-improving Red-teaming for Compute Use Agents 

**Authors**: Chen Xiong, Zhiyuan He, Pin-Yu Chen, Stjepan Picek, Tsung-Yi Ho  

**Link**: [PDF](https://arxiv.org/pdf/2608.30207)  

**Abstract**: Computer use agents (CUAs) are vision-language models that perceive a screen and act on a real operating system through mouse, keyboard, and terminal, and they are increasingly deployed to automate everyday digital tasks. Because they can be exposed to untrusted content while operating, they are vulnerable to indirect prompt injection (IPI), in which an adversary plants instructions in content the agent will read and redirects it toward actions that violate the user's intent. Existing CUA safety benchmarks evaluate fixed injections written by hand, which may underestimate the risk posed by an adaptive adversary. We present SIR, a black box IPI attack that (i) composes stealthy injections from a small library of reusable principles stated in plain language and (ii) wraps composition in an iterative feedback loop that diagnoses the victim's failed trajectories and distills the bypasses into new, named strategies that are reapplied across tasks. Unlike prior red teaming of web agents, we target CUAs at the operating system level and score attacks with a fully deterministic oracle, using checks on filesystem, service, and permission state rather than an LLM judge. On experiment, we evaluate three frontier CUAs. Composing principles with feedback raises the attack success rate over a baseline written by hand, for example from 4% to 24% on Claude Opus 4.8 and from 0% to 28% on Gemini 3.5 Flash, while the benign task still completes. Principles discovered against one model further transfer to a different architecture with no additional feedback. 

---
# Science sandboxes measure the scientific capability of AI agents 

**Authors**: Arya S. Rao, Rodrigo I. Castro, Sager J. Gosai, Kenneth B. Hsu, Yasha Ektefaie, Shantanu Singh, Sangeeta N. Bhatia, Steven K. Reilly, Ryan Tewhey, Eric S. Lander, Pardis C. Sabeti  

**Link**: [PDF](https://arxiv.org/pdf/2608.30165)  

**Abstract**: Scientific progress depends not only on finding solutions, but on learning the rules that explain why they work and using that understanding to design better experiments. We introduce science sandboxes, a framework for studying this capability in AI agents through repeated cycles of experimentation, feedback, and hypothesis revision. Science sandboxes invite an agent to query the natural world in different ways, ranging from "wet" physical experiments, to "damp" predictive models trained on empirical data, to "dry" invented rules. By establishing a common experimental loop and a protocol for evaluating agents within it, science sandboxes allow assessment of both quantitative performance on specific metrics and qualitative scientific reasoning, across a spectrum of empirical verifiability. Here, we instantiate this framework in two biological settings, models of regulatory genomics and protein fitness prediction, and examine the capabilities of frontier agents. Across these settings, we could see when agents successfully optimized a quantitative metric without understanding the rules underlying the system. In particular, their scientific reasoning deteriorated when they encountered systems whose rules fell outside familiar biological priors. By highlighting such failure modes, science sandboxes make the frontier of scientific capability measurable and provide a controlled setting in which to study and ultimately expand it. 

---
# CPR for LLMs: Critical-Point Routing against Catastrophic Forgetting in Domain Adaptation 

**Authors**: Kwangmin Ki, Yunhun Nam, Jongheon Jeong, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.30158)  

**Abstract**: Supervised fine-tuning (SFT) is the de facto standard for adapting large language models (LLMs) to target domains, but it often degrades the model's general capabilities, a phenomenon known as catastrophic forgetting. Existing approaches typically modify the SFT loss to mitigate forgetting, but they inevitably operate along a domain-generality trade-off. In this work, we step outside this trade-off by decoupling the two capabilities at the model level: we keep the original base model for general capability, and selectively invoke the SFT expert only when domain-specific knowledge is required. Specifically, we propose CPR (Critical-Point Routing), a token-level routing framework between a base model and its expert derivative, based on critical tokens where the base model fails but the expert succeeds. We train a lightweight hierarchical router that estimates the expert-call probability per token, and pair it with a tailored inference procedure that combines momentum smoothing and threshold gating. Across diverse model-domain configurations, CPR achieves state-of-the-art across all settings, surpassing SFT expert by 1.4-5.5% in domain performance while recovering its general-capability drop from 3.4-14.5% to at most 0.5%, with minimal overhead from invoking the expert on only one-third of tokens. 

---
# E-SENS: Exclusion-Sensitive Penalization for Negative-Constraint Retrieval 

**Authors**: Yerang Kim, Jiyoon Myung, Joohyung Han  

**Link**: [PDF](https://arxiv.org/pdf/2608.30130)  

**Abstract**: Retrieval-augmented language models can fail to respect negative constraints when the retriever supplies evidence about concepts the user explicitly excluded. Beyond explicit negation, queries may ask for answers that include one concept while excluding another, or for entities that belong to a category but differ from a closely related instance. Because the excluded concept still appears in the query text, dense retrievers may assign high similarity to documents about that concept even when the user asks to avoid it. We introduce E-SENS, a training-free reranking method for negation-sensitive retrieval. E-SENS extracts a compact trap query for the excluded side and subtracts trap-query similarity from the original-query retrieval score. On ExcluIR, E-SENS shows a clear recall-violation trade-off across four embedding models and reduces trap retrieval at recall-preserving settings. 

---
# VIBE: Video Instruction-aligned Background music gEneration 

**Authors**: Aryan Vijay Bhosale, Vaibhavi Lokegaonkar, Vishnu Raj, Gouthaman KV, Sreyan Ghosh, Ramani Duraiswami, Lie Lu, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2608.30125)  

**Abstract**: Current video-to-music (V2M) models lack semantic control and fail to penalize instruction violations, largely due to their reliance on reconstruction objectives and the representational bottleneck of static cross-modal conditioning in Diffusion Autoregressive (DAR) architectures. To resolve this, we introduce VIBE, a novel text-and-video-to-music (T+V2M) generation model that leverages: (1) Conditioning Connection, a depth-wise cross-layer conditioning mechanism that dynamically bridges the planning and diffusion refinement heads and (2) a comprehensive reward modeling taxonomy, optimizing for both hard, verifiable constraints (e.g., tempo, key) and soft, subjective qualities (e.g., musicality, multimodal alignment) with a structured 5-stage training curriculum. Upon evaluation using audio-visual alignment, instruction following, and audio quality metrics, along with a subjective human evaluation study, we observe that VIBE demonstrates enhanced controllability and instruction adherence while performing comparably to most evaluated baselines on generation fidelity and multimodal alignment. 

---
# TPR-Attention for Combinatorial Generalization 

**Authors**: Melisa Civelekoğlu, Isabeau Prémont-Schwarz  

**Link**: [PDF](https://arxiv.org/pdf/2608.30124)  

**Abstract**: Systematic generalization remains a significant challenge in deep learning. In particular, combinatorial generalization - generalizing to new configurations of known factors of variation - is effortless for humans but difficult for standard neural architectures that rely on statistical correlations rather than explicit structural representations. We introduce a new architectural component that embeds structured inductive bias into deep learning: an attention mechanism operating over tensor-product representations (TPRs). Through controlled experiments on compositional tasks, we show that this TPR-attention mechanism outperforms existing architectural components in combinatorial generalization. These results highlight the value of integrating explicit compositional structure into neural attention and point toward a promising path for models capable of systematic generalization. 

---
# Aligning Multi-Trajectory Supervision with Policy Optimization for VLA Driving 

**Authors**: Tian Zhang, Zhuo Huang, Hongrui Ye, Yu Wu, Zengmao Wang, Kaixuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.30122)  

**Abstract**: Vision-language-action (VLA) driving methods increasingly combine multi-trajectory imitation learning with group-relative policy optimization (GRPO), making trajectory selection critical to final performance. However, some high-scoring trajectories that improve imitation can degrade subsequent GRPO by inducing advantage estimates misaligned with the current policy's feasible behavior distribution, driving updates away from safe and compliant behaviors. To address this, we propose a novel framework that aligns multi-trajectory supervision with policy optimization. To address the policy gradient bias induced by infeasible noisy trajectories outside the feasible region, augmented trajectories are constrained to a neighboring manifold of the ground-truth feasible region, and a Pareto-optimality criterion is adopted in place of the conventional aggregate score, retaining only non-dominated candidates and thereby filtering out conflicting samples at the source. To ensure that expanded trajectory supervision is effectively absorbed during policy optimization, we introduce two complementary mechanisms: feasibility-first advantage assignment and dynamic distillation. The former adapts Pareto credit to the feasibility composition of each rollout group and guides fully infeasible groups toward safe references. The latter updates teacher trajectories across refinement rounds to continually transfer useful supervision. Together, they progressively translate the benefits of expanded supervision into policy improvement. On NAVSIM v1 and v2, our method achieves 91.4 PDMS and 89.1 EPDMS, respectively, under single-trajectory inference, and recovers 440 of 658 initially failed scenes, 11.1\% higher than the original GRPO baseline. 

---
# Can LLMs Take the Pulse of the Economy? A Real-Time Evaluation of LLM Nowcasts on Macroeconomic Indicators 

**Authors**: Xinyue Zhao, Ruiyi Zhang, Liqin Ye, Rui Cao, Pengtao Xie, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2608.30110)  

**Abstract**: Nowcasting headline macroeconomic indicators, i.e., estimating an indicator's value for the current reference period before its official release, is critical for monetary policy and financial markets, and central banks devote dedicated teams of expert economists to producing such estimates. Large language model (LLM) agents are a promising candidate for this task, combining broad world knowledge with real-time web search and supporting queries at higher frequency than institutional nowcasts. Evaluating their nowcasting capability is, however, challenging: headline indicators such as GDP and CPI are widely reported and likely memorized during pretraining, so any evaluation on historical releases is vulnerable to data contamination. To address this, we introduce LiveMacroEval, a live, contamination-resistant benchmark in which LLM agents produce hourly nowcasts for sixteen major U.S. macroeconomic indicators over a pre-release window closing at each official release. Nowcast quality is assessed through a LiveMacro Score against announcement-window equity returns and a LiveBetting Score from simulated Polymarket-style trading, with Federal Reserve regional-bank nowcasts, the Bloomberg ECOS professional consensus, and an auto-ARIMA baseline as comparators. Over six months with four state-of-the-art LLM agents configured with web search, aggregate nowcast accuracy is broadly comparable to the institutional and professional benchmarks, with performance varying widely across individual indicators. This highlights LLM agents' potential as real-time estimators of macroeconomic conditions. 

---
# AtlasNLP: A Country-Aware Atlas of Dataset Representation in NLP 

**Authors**: Joan Nwatu, Tsedeniya Solomon Amare, Longju Bai, Bontu Fufa Balcha, Zayd Bashir, Angana Borah, Zara Burzo, Yubin Choi, Naihao Deng, Samika Gupta, Michel Faloughi, Claude Kwizera, Ziqiao Ma, Cynthia Yacel Fuertes Panizo, Ellie Seehorn, Hui Shen, Jiayi Tang, Zesen Zhao, Boyuan Zheng, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2608.30107)  

**Abstract**: Understanding which countries are represented in NLP datasets is essential for identifying gaps, targeting data collection, measuring progress, and informing AI policy. However, geographic metadata is very rarely available, and country-level representation is often hidden behind broad language-level claims. We introduce AtlasNLP, a country-aware atlas of over 13,000 NLP dataset records across normalized NLP task categories, tracking both the populations represented and where datasets are produced. AtlasNLP includes AtlasNLP-Gold, a human-curated reference set, and AtlasNLP-Core, an ACL-derived large-scale collection. Using this resource, we show that (1) dataset coverage is highly uneven across countries and tasks; (2) dataset production and representation are geographically asymmetric; and (3) language coverage does not imply geographic representation. These findings reveal blind spots in current dataset documentation practices and motivate more explicit geographic metadata for country-aware NLP evaluation. 

---
# Graph4BiLO: Graph Neural Network Approximation for Bilevel Mixed-Integer Linear Optimization 

**Authors**: Jessica D. Elrefaei, Kaixun Hua, Seungbae Kim, Hoang Nam Tran, Juan S. Borrero  

**Link**: [PDF](https://arxiv.org/pdf/2608.30103)  

**Abstract**: Bilevel mixed-integer linear optimization problems model hierarchical decision processes in which a leader anticipates the optimal response of a follower. Although expressive, these problems are computationally challenging because lower-level optimality is embedded in the leader's feasible region. Value-function reformulations replace the nested follower optimization with a constraint involving the follower's optimal value, but evaluating this value function exactly can itself be expensive. This paper introduces Graph4BiLO, a graph neural network (GNN) approach for learning bilevel value functions from variable--constraint graph representations. In contrast to fixed-length multilayer perceptron (MLP) representations, the GNN uses shared message-passing parameters and can therefore be applied across multiple problem sizes with a single trained model. The learned ReLU network is encoded exactly as mixed-integer linear constraints and embedded in an approximate single-level formulation. A repair step subsequently re-solves the follower problem for the selected leader decision to recover a bilevel-feasible follower response. We evaluate Graph4BiLO on knapsack interdiction instances with 20--100 items against the exact MibS solver and the learning-based Neur2BiLO method. Graph4BiLO obtains objective values comparable to Neur2BiLO across all tested sizes while avoiding size-specific neural networks. An additional out-of-distribution experiment demonstrates zero-shot transfer from 20-item training instances to previously unseen 40- and 60-item instances. However, embedding message passing at every graph node substantially increases the resulting mixed-integer formulation size and solve time. These results identify a central tradeoff between size-generalizable graph representations and the computational cost of embedding GNNs within optimization models. 

---
# Arkios: An Open Bilingual English-Nepali Language Model Trained From Scratch, with a Devanagari-Aware Tokenizer 

**Authors**: Sajal Regmi, Siddhartha Pudasaini, Chetan Phakami Pun  

**Link**: [PDF](https://arxiv.org/pdf/2608.30092)  

**Abstract**: We present Arkios, a 1.04B-parameter dense transformer pretrained from scratch on 150B tokens of bilingual English-Nepali text, using a custom single-file C/CUDA training stack and a Devanagari-aware byte-level BPE tokenizer built for this project. On ARC-Easy and ARC-Challenge, Arkios exceeds three comparably sized open models (Pythia-1.4B, TinyLlama-1.1B, OLMo-1B) despite an order of magnitude fewer training tokens, likely aided by a match between our educational-web-text pretraining data and ARC's grade-school-science format rather than a general capability advantage. We report full evaluation results under standard protocols, including a correction to an earlier partial-sample estimate, and findings specific to evaluating small models in a low-resource language: the standard multiple-choice-letter prompt format used by common evaluation harnesses places this model at chance on Nepali reading comprehension, and simultaneously at chance on English in the same format, which would lead a naive benchmark run to conclude the model has no Nepali ability when in fact it does. Concretely, both languages score at chance in the letter-choice format (0.240 Nepali, 0.236 English, against a chance baseline of 0.250), while scoring the answer text directly reveals genuine, English-favoring comprehension (0.306 Nepali, 0.387 English). We describe a manifest-conditioned tool-use contract introduced during instruction tuning, where tool calls are permitted only when a tool manifest is declared in context and suppressed otherwise, and report where that contract holds and where it does not. We release both the base and instruction-tuned model weights under Apache-2.0. The training code and a small privately-sourced portion of the Nepali pretraining corpus are not released; everything needed to reproduce the reported numbers from the released weights is included here. 

---
# How do World Models and Policies Compose in LLM Agents? A Joint Spectral and Behavioral Account 

**Authors**: Ruize Xu, Xiao Yu, Yujin Tang, Chenming Shang, Nikhil Singh  

**Link**: [PDF](https://arxiv.org/pdf/2608.30067)  

**Abstract**: How do LLM agents come to both understand environments they act in and master tasks set within them? Through controlled experiments combining world-model training (next-state prediction) and policy training (reward maximization), we investigate this question. We dissect the resulting models through their additive parameter updates. Geometrically, we find effective world-model updates are low-rank and share an input-feature subspace with policy updates while writing to nearly orthogonal output directions, whether trained separately or sequentially. However, we find that, in projection interventions, the sequential update induces more robustness than separate policy RL when removing the world model's leading input directions, suggesting that it has learned alternative input pathways. Behaviorally, we find the sequentially trained agent explores a wider range of states and actions. Based on this, we ask: does policy training preserve world knowledge as well as it could? We probe this with training-free merging built on the geometrically motivated input basis plus an online world-model loss during policy RL, and show both improve over the untreated baseline. Our findings suggest world knowledge and task-directed ability can be learned in geometrically complementary forms, and that future post-training pipelines should consider how best to engineer the interface between them. 

---
# Pak3H: Evaluating the Cost of Cultural Mismatch in LLM Alignment with a Human-Contextualized Urdu Benchmark 

**Authors**: Abdullah Hashmat, Usman Naseem, Agha Ali Raza  

**Link**: [PDF](https://arxiv.org/pdf/2608.30065)  

**Abstract**: Large language models (LLMs) demonstrate strong Helpfulness, Harmlessness, and Honesty (3H) alignment in English-centric settings, but these gains transfer poorly to low-resource languages due to cultural mismatches. Existing multilingual 3H benchmarks rely predominantly on automated translation or LLM based synthesis, propagating source-language biases while sacrificing local relevance. To address this gap, we introduce Pak3H1, the first human-validated, culturally contextualized Urdu benchmark suite for 3H alignment, comprising PakAlpaca (helpfulness), PakBeaverTails (harmlessness), and PakTruthfulQA (honesty). Our multi-stage pipeline integrates manual cultural adaptation and dictionary-guided post editing to prioritize native speaker judgment, ensuring both semantic fidelity and contextual authenticity. Zero-shot evaluations across multiple open and proprietary LLM architectures reveal systematic cross-lingual alignment gaps: helpfulness win rates decline under localized contexts, harmlessness guardrails break down against regional safety risks, and composite honesty metrics degrade substantially due to localized factual constraints. These findings expose structural limitations in current alignment approaches, underscoring the necessity of human-guided localization for equitable multilingual evaluation. 

---
# Forget or Fine-tune? A Comparative Study of Machine Unlearning Strategies for Noisy Label Correction 

**Authors**: João L. P. Santana, Filipe R. Cordeiro  

**Link**: [PDF](https://arxiv.org/pdf/2608.30046)  

**Abstract**: Noisy labels remain a critical challenge for training deep neural networks, since memorizing incorrect labels degrades generalization. Once noisy samples are identified after training, the standard solution is to retrain the model from scratch on the cleaned dataset, which is increasingly expensive as datasets and models grow. Machine Unlearning (MU) has recently emerged as a computationally efficient alternative, but the relative effectiveness of different MU strategies for noisy-label correction remains poorly understood. In this work, we conduct a comparative empirical study of five MU methods (NegGrad, Fine-Tuning (FT), Random Labeling (RL), SalUn, and MUNBa) across symmetric, asymmetric, instance-dependent, and open-set noise on CIFAR-10, CIFAR-100, and the real-world noisy dataset Food-101N. Our central finding is that the appropriate unlearning strategy is conditioned on the noise structure. Simple FT is a strong baseline across most closed-set scenarios; RL and SalUn are the most consistently robust methods and, under instance-dependent noise, approach retraining accuracy at a fraction of the computational cost; MUNBa shows advantages mainly under extreme symmetric noise. Under open-set noise, in contrast, we show that retraining on the cleaned subset degrades accuracy relative to the noisy baseline, so approximating the retrained model is not an adequate objective in this regime. On Food-101N, all MU methods remain competitive and achieve accuracies close to retraining despite reducing runtime by an order of magnitude. These findings provide practical guidelines for selecting MU strategies for post-training noisy-label correction. 

---
# Reachability-Based Capability Confinement for LLM Agents under Indirect Prompt Injection 

**Authors**: Wujie Xiong, Rabimba Karanjai, Yang Lu, Weidong Shi, Lei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.30041)  

**Abstract**: Large language model agents place outputs from external skills into their execution context, allowing attacker-controlled data to influence later privileged actions. Existing defenses mainly classify untrusted content or authorize proposed operations. They do not directly address how an agent's future authority should change once untrusted data enters its state. We present SkillGuard, a harness-level enforcement layer that treats this event as contamination and restricts future capabilities to disconnect the resulting state from deployer-defined forbidden states. Given sound skill summaries and policies, SkillGuard represents security-relevant transitions with a Skill Impact Graph, specifies admissible control over skill parameters via steerability signatures, and mediates invocations with an inline reference monitor. Following contamination, it computes weighted capability restrictions using binary, fractional, or fractional-flow strategies without auxiliary language-model inference. We evaluate SkillGuard on four AgentDojo suites with two backend LLMs, Gemini 2.5 Flash and Llama3.3-70B, against an LLM-only No Defense baseline and three defenses at different system layers: Spotlighting, CaMeL, and AttriGuard. We construct a compositional attack benchmark in which each attack combines observations individually insufficient to induce target violation and evaluate the same baselines on it. Under AgentDojo's Tool Knowledge attacks, SkillGuard eliminates attack success on three of four suites for both backends and reduces it to 4.8% and 14.3% on Slack. Against compositional attacks, it outperforms every baseline on Llama and matches the strongest baseline on Gemini at higher benign utility. Fractional-flow restriction preserves substantially more capabilities than binary restriction at the same attack success rate. Across both settings, SkillGuard adds no model calls or token overhead. 

---
# "Act Like a 5th Grader" is Not Enough: Bounding Knowledge in LLM-Based User Simulators 

**Authors**: Krisztian Balog, Arild Michel Bakken  

**Link**: [PDF](https://arxiv.org/pdf/2608.30033)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate human behavior but frequently fail to exhibit realistic cognitive constraints, suffering from a "superhuman bias." Using a dataset of over 71,000 reading comprehension responses from 2,359 primary-school students (grades 4--6), we demonstrate that standard persona prompting yields near-perfect, deterministic performance, failing to capture the natural variance of developing readers. To address this, we introduce the Cognitively Bounded User Simulator (CBUS), an architectural framework that explicitly models the restricted working memory of young readers through an episodic bottleneck. Within this framework, we formalize two distinct test-taking strategies to emulate different reading behaviors. Our evaluation shows that explicitly modeling cognitive bounds significantly narrows the simulation gap across multiple LLM backbones, demonstrating that enforcing architectural constraints is more effective for high-fidelity simulation than simply scaling raw model capabilities. 

---
# Error Detection for PET/CT Radiology Reports: Domain-Specific vs Large Language Models 

**Authors**: Hermione Warr, Harry Anthony, Lilli J Freischem, Yasin Ibrahim, Daniel R McGowan, Konstantinos Kamnitsas  

**Link**: [PDF](https://arxiv.org/pdf/2608.30021)  

**Abstract**: Errors in radiology reports can adversely affect patient treatment, yet automated report quality assurance remains challenging because errors are often subtle and require domain expertise to detect. Although large language models (LLMs) have recently been proposed for radiology report verification, their ability to detect clinically meaningful errors beyond chest X-ray datasets remains under-explored. To this end, we present the first systematic evaluation of language models for PET/CT report error detection, comparing compact domain-specific models with SOTA open-weight LLMs. We collected 30,633 oncology FDG PET/CT reports from 23 radiologists over 10 years. We trained domain-specific BERT models to detect clinically motivated synthetic reporting errors and evaluated alongside zero-/few-shot Qwen3-32B, Gemma-3-27B and Llama-3.3-70B on a held-out benchmark of 11,500 reports. A 15M-parameter model achieved 94.4% balanced accuracy with a 5.8% false-positive rate, compared with 84.0% for the strongest prompted LLM. Task-specific adaptation of Llama-3.3-70B closed this performance gap (94.4%) but retained substantially greater computational requirements. Our results suggest that domain-specific training matters more than model scale for PET/CT report error detection, supporting compact models as an accurate and computationally efficient approach to automated radiology report quality assurance. 

---
# TEMPO: Temporally-grounded Multi-task Post-training for Large Audio-Language Models 

**Authors**: Apoorva Kulkarni, Kaousheik Jayakumar, Sreyan Ghosh, Utathya Aich, Ramani Duraiswami, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2608.29999)  

**Abstract**: Large audio-language models (LALMs) describe audio at the clip level but cannot assign timestamps to the events, speakers, or sounds they identify. Despite being essential for downstream tasks like speech recognition and dense audio captioning, timestamping remains a key limitation of most LALMs. We present TEMPO (Temporally-grounded Multi-task Post-training), the first unified model to handle audio, speech, and music timestamping tasks. Our core contribution is a supervised fine-tuning (SFT) stage built on three innovations: atomic timestamp tokens, a time-aware projector that injects sinusoidal wall-clock encodings into audio frame embeddings, and a distance-aware Gaussian loss. Our training is based on a synthetic-to-real curriculum. We further introduce, to our knowledge, the first application of reinforcement learning to unified audio timestamping, using GRPO with verifiable temporal rewards that directly optimize the evaluation objectives. Rather than serving as the primary source of performance gains, GRPO acts as a refinement stage on top of the SFT checkpoint, providing modest additional improvements. To support this work, we build a training dataset containing 119K samples and an evaluation benchmark containing 10K samples, drawn from established corpora across five tasks. On this benchmark, TEMPO outperforms Audio Flamingo Next and Qwen3-Omni, two state-of-the-art LALMs explicitly trained on timestamped data. Experiments confirm that SFT delivers most of these gains, with GRPO providing consistent but moderate refinements. 

---
# Partition-Aware Unlearning for Removing Spurious Correlations in Large Vision-Language Models 

**Authors**: Aditi Sarker, Nazreen Shah, Rafi Ibn Sultan, Rhongho Jang, Dongxiao Zhu, Prashant Khanduri  

**Link**: [PDF](https://arxiv.org/pdf/2608.29996)  

**Abstract**: Large Vision-Language Models (LVLMs) achieve strong performance across many multimodal tasks; however, they often exploit spurious object-background correlations, resulting in predictions driven by contextual shortcuts rather than object-relevant visual evidence. Despite growing interest in hallucination and robustness evaluation, existing benchmarks provide limited control over whether model predictions are grounded in the target object or induced by correlated background cues. In this work, we introduce PURGE (\underline{P}artition-aware \underline{U}nlearning for \underline{R}emoving spurious-correlation \underline{G}enerated \underline{E}rrors), a framework for constructing, benchmarking, and mitigating spurious-correlation-induced failures in LVLMs. The framework consists of: -- (1) Structured dataset construction wherein we develop three complementary structured data construction strategies that partition examples by object-relevant evidence and spurious background cues, enabling controlled diagnosis of shortcut reliance; and -- (2) Partition-aware unlearning, which uses these partitions to selectively remove spurious object-background associations while preserving object-based reasoning. We evaluate the \algo~framework across multiple LVLMs, including LLaVA-1.6-7B, Qwen3-VL-8B-Instruct, and Qwen3.5-9B, together with CLIP as a vision-language encoder, on a diverse suite of benchmarks, including CHAIR, POPE, Causal-HalBench, MM-SpuBench, AMBER, MMHal, and Waterbirds. Our results show that PURGE consistently reduces hallucinations and spurious-correlation-driven errors while maintaining or improving overall performance in most evaluated settings, providing both a reusable evaluation protocol and an effective mitigation framework for more reliable LVLMs. 

---
# Generating Clinical Vignettes that Preserve Cognitive Formulations 

**Authors**: Amit Oren, Nimrod Hertz-Palmor, Dean Ariel, Guy Laban  

**Link**: [PDF](https://arxiv.org/pdf/2608.29995)  

**Abstract**: Large language models can generate fluent clinical case vignettes, but fluency alone does not ensure fidelity to a specifiable clinical structure. We introduce FORMA, a theory-grounded framework that compiles a cognitive model of a disorder into a directed weighted graph, samples a person-specific configuration of that graph, and validates whether the generated vignette preserves the specified components and causal links. We instantiate FORMA on Posttraumatic Stress Disorder using the Ehlers and Clark cognitive model, generating 16,500 vignettes across 500 personas, 11 generation models, and three ablation conditions. Evaluation combines an external edge-recovery probe, two clinical experts, a scaled LLM judge, and a clinician user study with 100 licensed practitioners. The cognitive graph is recoverable from full-condition vignettes (MCC = +0.41, AUC = 0.70) but not from zero-shot generation (MCC = +0.01, AUC = 0.50). Experts rate full vignettes substantially higher than zero-shot alternatives, and clinicians perceive them to be human-written 85% of the time, compared with 22% for zero-shot. FORMA also reduces demographic disparity in perceived quality by 1.5-7x. These results show that cognitive formulation can serve as an auditable specification for scalable synthetic clinical text generation. A repository with the data and code is available online: this https URL. 

---
# Beyond Fluency: A Rubric-Based Benchmark for Evaluating Saudi Dialect and Cultural Competence in Large Language Models 

**Authors**: Ghassan Al-Sumaidaee, Sajjad Abdoli, Ahmed Rashad, Maxim Legg  

**Link**: [PDF](https://arxiv.org/pdf/2608.29990)  

**Abstract**: Large language models are increasingly deployed in Arabic-speaking markets, yet standard benchmarks overwhelmingly reward Modern Standard Arabic (MSA) fluency while leaving dialectal and culturally grounded competence unmeasured. This gap is consequential: everyday Arabic is largely dialectal, and dialect encodes social meaning that MSA-centric evaluation cannot capture. We present a rubric-based benchmark for the Saudi dialect, comprising 31 expert-authored prompts spanning idiomatic, pragmatic, lexical, and culturally-embedded phenomena, each paired with an expert-established ground truth. Our methodology separates evaluation into a model-agnostic phase, in which atomic, MECE positive criteria are derived solely from the ground truth, and a model-specific phase, in which four state-of-the-art systems -- Claude Opus 5, Gemini 3.7, GPT-5.6, and Kimi K3 -- are scored against those criteria and penalised for errors they actively introduce. Across 124 model-prompt evaluations we catalogue 466 error instances under a nine-category taxonomy. The four systems cluster within a narrow macro-average band (42.7%-53.1%), with no model exceeding 55% and every model recording at least one negative-scoring prompt, confirming that Saudi dialectal competence remains broadly unsolved. Notably, Ambiguous Framing is the dominant failure mode (37.3% of errors) while outright Hallucination accounts for only 11.2%, indicating that models fail less by stating falsehoods than by distorting register and flattening pragmatic nuance. We further observe a consistency-versus-ceiling trade-off and model-distinctive error signatures. We release the full prompt set, ground truths, and scored rubrics to support reproducible dialectal evaluation. 

---
# Training-Free Action Correction for VLA Model Failures via Language Feedback 

**Authors**: Owen Kwon, Pablo Ortega-Kral, Arthur Bucker, Jean Oh  

**Link**: [PDF](https://arxiv.org/pdf/2608.29967)  

**Abstract**: Vision-Language-Action (VLA) models demonstrate strong semantic understanding yet exhibit systematic failures during deployment. The conditions under which these failures occur, and whether they can be corrected without retraining, remain poorly understood. In this paper, we take steps toward addressing this gap. We present CorrectVLA, a framework that translates task-level natural language corrections into additive action magnitude adjustments without modifying policy weights. A human provides a single task-level correction, applied uniformly across all rollouts without per-episode intervention. In simulation, CorrectVLA recovers execution misalignment failures across both in-distribution and OOD tasks. In real-robot experiments on a UFactory xArm7 under environment shift, CorrectVLA restores near-perfect success where the base policy almost entirely breaks down, generalizing across object locations and identities. Through a taxonomy of failure modes on LIBERO-90, we find that execution misalignment failures, where the policy reaches the correct target but miscalibrates action magnitudes, represent the correctable subset, while other failure modes where semantic comprehension itself breaks down are not amenable to this approach. The approach succeeds when policies possess strategic correctness and fails when fundamental comprehension is absent, establishing a practical operational boundary for inference-time correction. 

---
# The Policy Deficit in AI x Social-Emotional Learning Research 

**Authors**: Tran Van Cuong, Liu Yihan, Nguyen Van Tuong  

**Link**: [PDF](https://arxiv.org/pdf/2608.29950)  

**Abstract**: As artificial intelligence (AI) is increasingly integrated into social-emotional learning (SEL) initiatives, the need for evidence-based policy has become paramount. We systematically reviewed 65 peer-reviewed papers that examine the intersection of AI and SEL to investigate how these studies articulate policy implications. Our analysis revealed a substantial "policy deficit" in the current AI x SEL literature: nearly three-quarters of the studies did not mention policy implications at all. Using the "WH-question" framework (Who, What, Why, When/Where, and How), we map the policy implications narratives present in the literature and show that they often lack the specificity and actor-oriented guidance required for effective evidence-informed policymaking. We find a significant association between publication venue and policy engagement, suggesting that current academic incentive structures may prioritize technical innovation and pedagogical feasibility over explicit engagement with governance and regulation. This study identifies a "techno-solutionist" trap, where technical potential is foregrounded while the institutional conditions for responsible implementation remain under-specified. We conclude by proposing a shift from "implication-as-afterthought" to "implication-as-methodology" and offer a set of actionable guidelines for researchers, editors, reviewers, and policymakers to bridge the gap between AI innovation and educational governance. Rather than presenting policy as a generic ethical horizon, we argue that AI-SEL studies should systematically specify Who should act, What actions are recommended, Why these actions are needed, When and Where they apply, and How strongly they are framed, thereby strengthening the translation of AI x SEL innovation into educational policy and practice. 

---
# Influence Is Not Authority: When Causal Guardrail Signals Make Legitimate Tool Use Look Like an Attack in Tool-Using LLM Agents 

**Authors**: Tanzim Ahad, Ismail Hossain, Md Jahangir Alam, Sai Puppala, Syed Bahauddin Alam, Sajedul Talukder  

**Link**: [PDF](https://arxiv.org/pdf/2608.29942)  

**Abstract**: The key limitation of current state-of-the-art influence-based guardrails is that they do not reliably distinguish a legitimate, user-authorized action from a malicious, unauthorized action when both rely on external tool information. This ambiguity can cause benign actions to trigger unnecessary verification and intervention, reducing utility and adding latency. We expose this limitation through an authorization-equivalence audit of 96 conditions derived from 24 base cases. Within matched source comparisons, we hold authorization, the exact committed action, and its intended effect fixed, changing only whether a required value comes from the user or a legitimate tool result. Although the action remains unchanged, this harmless relocation shifts the causal signal toward the attack region in all 24 cases under both Llama and Gemma scorers. Matched unauthorized controls show that the signal remains attack-sensitive, yet the benign relocation produces a larger average score shift than the actual change in authorization. Architecture-level evaluation shows how this mismatch propagates through guardrail designs. With a semantic monitor, attack success is 0% and utility is 28%, compared with 16% and 60% without it. A shadow-based guardrail allows every tested harmless run, yet does not reject matched unauthorized actions more often overall: 57.5% of unauthorized runs pass automatically before reaching the later security check, compared with 29.2% of authorized runs. These results show that the studied causal signal reveals what shaped an action without reliably encoding whether the action was authorized, and that reference construction and routing are integral to the effective security decision. 

---
# Hallucination Mitigation for Large Vision-Language Models via Implicit Feature Stabilization 

**Authors**: Aditi Sarker, Rafi Ibn Sultan, Hui Zhu, Dongxiao Zhu, Prashant Khanduri  

**Link**: [PDF](https://arxiv.org/pdf/2608.29924)  

**Abstract**: Large Vision-Language Models (LVLMs) are prone to hallucinations: they fluently describe objects, attributes, and scenes that are not in the image. We connect part of this failure to a measurable property of their representations, feature instability, where mild semantics-preserving perturbations of the input cause large changes in the learned embeddings; hallucination rates rise together with this variability. Existing stability-motivated remedies are explicit, in the sense that they intervene at inference time through latent steering or constrained decoding, and pay for it on every query. We propose implicit stabilization instead: perturbation-invariance is built into the model weights during fine-tuning, and nothing extra runs at deployment. Our framework, INFUSE, first stabilizes visual and textual representations around perturbation-averaged and ground-truth anchors, then aligns the stabilized representations across modalities with bidirectional contrastive objectives. We prove that the anchor's root-mean-square deviation from the perturbation-mean representation shrinks at rate $1/\sqrt{K}$ in the number of views, and that under a Lipschitz decoder, this bounds how much any perturbation can change the model's hallucination behavior. On LLaVA-1.5, LLaVA-1.6, and Qwen3-VL-8B-Instruct, INFUSE reduces AMBER CHAIR by 46-63% relative to each base model, improves ObjHal, MMHal, HallusionBench, and POPE, and preserves VQA-v2 and TextVQA, all with no inference-time overhead. 

---
# Sleight of Word Benchmark: Can Language Models Notice If Their Own Output Was Tampered With? 

**Authors**: Alberto Cetoli  

**Link**: [PDF](https://arxiv.org/pdf/2608.29921)  

**Abstract**: The output of a Language Model can be tampered with \emph{while} the model is writing it. A simple test can thus be constructed by evaluating the model's perception of this external perturbation. In this spirit, a simple benchmark is built in which a single word is consistently substituted with another in the generation process. We call this method \emph{Sleight of Word}. Two distinct axes are measured: metrics that relate to the model's surprise, as well as an evaluation of the textual reaction for 19 different open-weight language models. 

---
# IndicDetect: Evaluating Cross-Lingual LLM-Generated Text Detection for Hindi, Telugu, and Tamil 

**Authors**: Bhaskar Ganesh Devalla, Junchao Wu, Nilesh Dokuparthi, Greeshma Yaluru, Tatiana Muniz Rodriguez, Lidia S. Chao, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2608.29919)  

**Abstract**: The rapid proliferation of LLMs has further heightened the need to develop dependable AI-generated text detection, especially beyond English. Nevertheless, current benchmarks pay little attention to Indic languages and test detectors in idealized settings that do not represent the real world. We present a generalized benchmark for AI-generated text detection in Hindi, Telugu, and Tamil, which we call IndicDetect, designed to assess the robustness of detectors under realistic distribution shifts. IndicDetect comprises highly curated human-written texts matched with LLM-generated counterparts across various domains and generators, and systematically evaluates detectors in the presence of domain shift, generator shift, and adversarial perturbation. Using a single and repeatable evaluation scheme, we evaluate a wide range of statistical and neural detectors. We find substantial robustness failures: supervised neural detectors perform well in-distribution, while training-free methods degrade considerably under unseen generators and adversarial attacks. The severity of these failures varies across languages, with Hindi exhibiting the largest overall degradation under adversarial perturbations. These results highlight that the primary weakness of existing detectors in Indic settings lies in their robustness, not in their peak accuracy. IndicDetect provides standard data splits, an evaluation protocol, and baselines to establish a robust, language-aware foundation for AI-generated text detection in Indic scripts. 

---
# When Less is More: Understanding When Token Filtering Helps and Fails in AI-generated Text Detection 

**Authors**: Xiaoyang Han, Lvxiaowei Xu, Ming Cai  

**Link**: [PDF](https://arxiv.org/pdf/2608.29903)  

**Abstract**: The rapid advancement of large language models (LLMs) has made AI-generated text detection increasingly critical. Existing zero-shot detectors assume that more token-level evidence leads to more reliable detection. However, our empirical study challenges this consensus: fewer tokens sometimes work better, retaining only 40% can yield optimal performance, yet this benefit is not universal. Using the Entropy Gap Score (EGS), we introduce top-$k$ cumulative probability filtering as a diagnostic probe. Across three representative settings, filtering exhibits strikingly different behaviors. We analyze EGS via typical set theory and quantify its dynamics through entropy calibration and distribution analysis. We find that filtering helps for weak source LMs, where low-entropy tokens are harmful, but fails for strong source LMs, where they are not notably harmful. Our work provides the first systematic analysis showing that some tokens are not merely uninformative but systematically harmful due to entropy miscalibration, revealing a two-sided trade-off in token-level detection. 

---
# INTERVenE: Temporal-Abstraction-Interval Based Transformers for Short-Horizon Medical Event Prediction 

**Authors**: Shahar Oded, Yuval Shahar  

**Link**: [PDF](https://arxiv.org/pdf/2608.29901)  

**Abstract**: Electronic Health Record (EHR) prediction models in the intensive care unit must learn from sparse and irregular measurements while preserving the clinical meaning of time and supporting transparent decision-making. We present INTERVenE, a family of Transformer architectures whose input is an interval-based, knowledge-based temporal abstraction (KBTA), a token stream of named clinical concepts (states, trends, events, contexts) drawn from a curated medical ontology, rather than an unnamed bin index or a raw measurement triplet. This naming layer is what we ask KBTA to do: it makes the model's per-token attributions resolve to clinical concepts by construction. INTERVenE offers two complementary variants: an auto-regressive decoder that generates future abstraction trajectories with a per-step risk readout (localizing \emph{when} and \emph{after which events} risk rises), and a bidirectional encoder for single-pass joint risk and time-to-event prediction. Evaluated on 57,078 MIMIC-IV admissions against GRU-D, STraTS, and KarmaLego, INTERVenE-Enc reaches a support-weighted AUPRC$_w$ of 0.672, improving by 0.041 over the strongest neural baseline with non-overlapping 95\% bootstrap CIs, while also taking the best AUROC$_w$ (0.901) and length-of-stay MAE (44.4\,h). INTERVenE-Ar (AUROC$_w$ $0.854$, AUPRC$_w$ $0.587$ under the same evaluation contract - a strictly harder generative readout) provides a complementary token-level risk trajectory. An input-representation ablation confirms the lift transfers across structured discretizations, positioning KBTA-based intervals as the interpretable substrate that makes per-token attributions resolve to meaningful clinical concepts within the deployed model. 

---
# REIGN: Refurbished Embeddings with Integrated Guidance Networks for Efficient Context-Length Scaling 

**Authors**: Devrim Çavuşoğlu, Emre Akbaş  

**Link**: [PDF](https://arxiv.org/pdf/2608.29899)  

**Abstract**: Dense retrieval over long documents is expensive. Token-level encoders scale quadratically in sequence length, and most long-context embedding models reach 32K tokens only through architectural workarounds or by stretching billion-parameter LLMs. We propose REIGN (Refurbished Embeddings with Integrated Guidance Networks), a contrastively trained bi-encoder that operates on sequences of contextualised chunk embeddings from a frozen Guidance Network (GN) rather than on raw tokens. REIGN targets multi-chunk inputs, primarily for document-to-document retrieval; single-chunk inputs stay with the GN. Decoupling token-level processing from document-level reasoning, and caching the GN embeddings to disk, cuts per-document training cost by roughly four orders of magnitude relative to chunked Transformer fine-tuning. We also release a synthetic long-document retrieval benchmark for contrastive training and evaluation at long context lengths. Across an in-distribution Wikipedia benchmark, the LoCo out-of-distribution suite, and a real-world patent retrieval case study, REIGN matches dense long-context retrievers at smaller parameter budgets in each regime. A paired significance test puts it on par with models 1.6-4.3x larger on the patent task, and it stays within 0.65 nDCG@10 of a 20x-larger model on LoCo. 

---
# R$^2$A: Learning Persona Policies Through Persona Representation Learning and Runtime Alignment 

**Authors**: Mohan Zhang, Chengsong You, Xiaoyu Cao, Zhen Sun, Xiaohan Jia, Junwei Zhou, Yongchao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.29798)  

**Abstract**: The same Persona behavior can be beneficial in one context but harmful in another, causing static Persona elicitation to perform inconsistently across tasks. We introduce the Persona Selection--Realization Framework, which models behavior generation through a latent Persona state and decomposes it into Persona Selection and Persona Realization. The discrepancies between static Persona elicitation and an ideal Persona policy in these two components define the Selection Gap and Realization Gap, respectively. Building on this framework, we propose R$^2$A, a two-stage approach for learning Persona policies. Persona Representation Learning uses structured Who--How--What presentations to encode the target Persona's objective, conditional behavioral principles, and trajectory-level manifestations. Persona Runtime Alignment then removes the explicit Persona specification and jointly calibrates behavior selection and trajectory realization using task feedback. Across 12 evaluation settings covering the four principles of the Accountable-Professional Persona studied in this work, R$^2$A overall outperforms both the base model and static Persona elicitation. Ablation results further show that Persona Representation Learning is critical for preventing Runtime Alignment from producing behaviorally imbalanced policies and for achieving more stable Persona policy learning. 

---
# SynCrash: A Multi-Stage Pipeline for Zero-Shot Accident Detection and Localization in Traffic Surveillance Video 

**Authors**: Arkya Jyoti Bagchi, Ritul Jangir, Varun Raskar  

**Link**: [PDF](https://arxiv.org/pdf/2608.29759)  

**Abstract**: We present SynCrash, a multi-stage pipeline for zero-shot accident detection, spatial localization, and collision-type classification in fixed-view CCTV surveillance video. Our approach addresses the ACCIDENT at CVPR 2026 Challenge, which requires predicting when an accident occurs, where in the frame the impact happens, and what type of collision it is, all without access to labeled real-world training data. The pipeline operates in three decoupled stages: (1) Temporal localization via a VideoMAEv2-giant backbone fine-tuned on CARLA-based synthetic clips with metadata-aware embeddings and dense sliding-window inference; (2) Spatial localization using YOLO for object detection combined with a physics-informed hybrid heuristic that leverages bounding-box overlap and trajectory-based reasoning to predict the impact point; and (3) Collision-type classification using a lightweight rule-based strategy derived from the number and configuration of detected vehicles. The key insight is that temporal understanding benefits from supervised fine-tuning on synthetic data, whereas spatial understanding is better served by pretrained object detectors and physics priors that transfer naturally across domains. 

---
# Higher-Dimensional Rotary Position Embedding 

**Authors**: Yixing Li, Ruobing Xie, Yudong Zhang, Yushi Bai, Samm Sun, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.29715)  

**Abstract**: Transformers rely on position embedding mechanisms in long context modeling in most cases. Rotary Position Embedding (RoPE) embeds positional information with independent 2D rotations, forming relative position terms in self-attention. However, its pairwise, block-based, and decoupled structure limits deep mixing and robustness across channels. We propose HD-RoPE, which extends RoPE from independent 2D rotations to higher-dimensional rotations and introduces a Paley-I orthogonal basis to obtain balanced, isotropic, and dense phase mixing within each rotation subspace. This significantly enhances channel coupling and rotational degrees of freedom while maintaining orthogonal stability and the relative position closure property. Furthermore, HD-RoPE is easily optimized for engineering efficiency without introducing additional trainable parameters. We have conducted extensive evaluation results demonstrating that HD-RoPE achieves significant performance improvements over standard RoPE across various popular benchmarks and in both long and short contexts. 

---
# A Calibration Audit of Confidence in Feed-Forward 3D Reconstruction 

**Authors**: Nanxing Nick Deng, Qing Cheng, Niclas Zeller, Daniel Cremers  

**Link**: [PDF](https://arxiv.org/pdf/2608.29705)  

**Abstract**: Feed-forward 3D reconstruction models emit a per-pixel confidence that downstream systems read as a reliability signal. It is trained as a loss weight, not as an uncertainty magnitude, and whether it can be used as an error prediction has not been measured. We audit seven released backbones on thirteen datasets and score the confidence on four properties, how well it ranks error, whether its level is right on average, whether it holds across the confidence range, and whether its intervals cover the truth. The confidence ranks error well, but the predicted uncertainty is too low when it is read under conditions that are not exactly those of training. The median case is off by 2.4x across all seven models, and the error prediction is further off the more confident the model is. We show that this phenomenon can appear even though the loss's optimum is reached. A released model resumed under its own loss reaches that optimum on its training data within a few hundred updates and stays overconfident on unseen frames. A power law with two constants per backbone and dataset corrects the overall magnitude of the predicted uncertainty and leaves the ranking untouched. What no rescaling reaches is the scene, which we attribute to the model's missing knowledge of scale across predictions. Every correction we tried is close to right on average and still leaves two thirds of held-out scenes outside a five-point band, because what a scene is missing is a shape rather than a shift. We release the audit protocol, its results, and the fitted constants per model and dataset. Fitted with the target dataset held out, the constants bring the median case from 2.4x off to 1.35x, and a refit on a few labelled scenes of that dataset reaches 1.12x. 

---
# MedSegBenchmarker: A Raw-Count-First Framework for Controlled 2D Medical Image Segmentation Benchmarks 

**Authors**: Vanessa Borst, Lukas Horn, Daniel Grillmeyer, Thomas Prantl, Samuel Kounev  

**Link**: [PDF](https://arxiv.org/pdf/2608.29677)  

**Abstract**: Despite rapid advances in MIS, fair and reproducible comparisons of segmentation models remain challenging due to heterogeneous datasets, inconsistent evaluation protocols, and rapidly evolving architectures. In particular, comparisons often implicitly assume that model rankings are invariant to data partitioning, preprocessing, metric aggregation, uncertainty estimation, and computational constraints. The lack of extensible and unified evaluation frameworks further limits systematic investigation of new models, datasets, and training paradigms. We present MEDSEGBENCHMARKER (MSB), a configuration-driven framework for controlled benchmarking of 2D MIS. It integrates duplicate and near-duplicate image detection, group-aware data splitting, YAML study specifications, resumable training, hyperparameter optimization, cross-validation, and checkpoint-based evaluation. Rather than retaining only aggregate performance measures, MSB exports sample- and class-level pixel counts and predictions together with the evaluation context. These elementary artifacts enable post-hoc analyses without repeated inference. We demonstrate MSB in a case study involving three heterogeneous 2D datasets and multiple MIS and general-purpose vision models evaluated at 256- and 512-pixel input resolutions. Reaggregation of identical predictions changes the top-ranked architecture in three of six dataset-resolution settings, despite high rank correlations between aggregation strategies. Increasing input resolution produces model- and dataset-dependent performance gains and losses that must be considered alongside empirically measured inference complexity. These results show that seemingly minor choices in evaluation and experimental setup can affect benchmark conclusions. MSB, available at GitHub, provides a practical and extensible basis for making benchmark conditions and evaluation choices explicit and reproducible. 

---
# Cost-Effective Repository Exploration for Agentic Issue Localization 

**Authors**: Mohammad Nour Al Awad, Sergey Ivanov  

**Link**: [PDF](https://arxiv.org/pdf/2608.29675)  

**Abstract**: Repository exploration is a distinct and costly stage of coding-agent pipelines: before generating a patch, an agent must identify which repository files are likely to matter. We study whether this stage can be delegated to lower-cost models while retaining useful localization quality. Using our IssueLoc-Bench, we evaluate five explorer models under the same read-only interactive interface on 499 SWE-bench Verified-derived tasks and 500 tasks from 153 additional repositories. We measure early candidate discovery, top-three gold-file coverage, strict file-set recovery, agent time, and token usage, with paired instance-level uncertainty and repository-clustered sensitivity analysis. The highest-quality explorer leads across localization metrics, but substantially cheaper operating points emerge: depending on the model and evaluation arm, lower-cost explorers retain approximately 78-94% of the reference Hit@3 and 73-92% of its F1 while reducing mean agent time by 41-88% and token usage by 84-95%. The preferred operating point depends on how localization is consumed downstream: ranking and coverage metrics characterize recoverable candidate handoffs, whereas F1 and exact match characterize restrictive file gates. These results support treating repository exploration as an independently measurable and budgetable stage of modular coding agents, with explorer selection guided by the downstream handoff contract. 

---
# Conducting Stylistic Analysis of Paintings through an Art-History Agent 

**Authors**: Marc S. Walton, Astrid Harth  

**Link**: [PDF](https://arxiv.org/pdf/2608.29644)  

**Abstract**: Attributing an artwork to an artist has traditionally relied on detailed visual observations and descriptions, known as stylistic analysis in art history. By contrast, current artificial intelligence (AI) models used in the field offer only unexplained probabilistic classifications. To bridge this methodological gap, we present an AI framework that automates stylistic analysis of paintings, providing a foundation for enhancing evidence collection, discovery, and verification. By training a vision transformer (ViT) on a large corpus of paintings with metadata, our system encodes this art history-specific data as embeddings. These representations are factorized via sparse dictionary learning into a shared set of features that recur across the training set. A large language model (LLM) then interprets each feature by retrieving associated artworks and their accompanying curator-written texts, and synthesizes them into descriptions that reflect their stylistic attributes. Finally, an autonomous coordinator LLM applies a reasoning-and-action (ReAct) framework to weight, test, and refine these features into cohesive descriptions of an artwork, or comparisons of artworks. This approach converts detailed visual features into descriptive terms, addressing a key challenge in art history. It thus connects the use of images as data with the semantic concerns of humanists, establishing vision-based computational art history as an area for future growth. 

---
# LLMODE: Aligning ODEs with LLMs via Gated Token Injection for Irregular Spatio-Temporal Forecasting 

**Authors**: Di Zhang, Jingyang Zhang, Ziqian Wang, Chi Zhang, Yikun Ban, Ziwei Zhang, Ruijie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.29640)  

**Abstract**: Large language models (LLMs) have shown promise for spatio-temporal forecasting, but existing approaches often rely on regularly sampled token sequences and struggle with irregular observations because of temporal asynchrony, representation-space misalignment, and limited context windows. We propose LLMODE, a token-efficient framework for irregular spatio-temporal forecasting with a frozen LLM backbone. LLMODE first uses a graph-aware ODE encoder to reconstruct irregular graph observations as a continuous-time latent trajectory. A Fixed-Budget Perceiver Resampler then compresses this variable-length trajectory into a fixed number of dynamic memory tokens. In parallel, compact statistical descriptors are encoded and resampled into context memory tokens. A dual-source gated cross-attention module injects both memories into the frozen LLM, enabling controlled utilization of external spatio-temporal evidence. Experiments on three real-world urban datasets and two physical-dynamics benchmarks show competitive overall performance, with clearer advantages under sparse or dynamically complex irregular sampling. Additional evaluations on unseen urban regions further demonstrate strong zero-shot generalization without adaptation. 

---
# MI-Distillation: Selecting from Model-Interpolated Instruct-Reasoning Data Spectrum for Chain-of-Thought Distillation 

**Authors**: Yangsong Lan, Renkai Hu, HongKai Zheng, Bo Zhang, Renzhi Wang, Hongliang Dai, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.29623)  

**Abstract**: Recent advances in large reasoning models (LRMs) have shown strong performance on complex problems through long chain-of-thought (Long CoT) reasoning. However, distilling such trajectories into smaller student models remains challenging: direct Long CoT supervision often provides limited gains and can be less effective than concise Short CoT rationales. In this work, we investigate this phenomenon from a gradient-centric perspective. Our analysis shows that Long CoT induces larger gradient magnitudes and more concentrated update directions than Short CoT, with this effect becoming more pronounced as student model capacity increases. These findings suggest that effective Long CoT distillation requires balancing the reasoning information density of reasoning trajectories with their distributional alignment to the student model. Motivated by this insight, we propose \textbf{M}odel \textbf{I}nterporlation \textbf{Distillation} (\textbf{MI-Distillation}), a framework that constructs a continuous Instruct-Reasoning data spectrum through model interpolation. To select suitable trajectories from this spectrum, we further introduce \textbf{Seq}uential \textbf{L}earnable \textbf{S}urprisal \textbf{S}core (\textbf{SeqLSS}), which favors reasoning paths that are both informative and learnable for the student. Extensive experiments on reasoning benchmarks show that MI-Distillation consistently improves small model CoT distillation over strong Long CoT baselines. 

---
# AgenticRag-R1: Agentic Reinforcement Learning with Stack Memory for Multi-Step Reasoning, Retrieval and Memorizing 

**Authors**: Xinke Jiang, Yue Fang, Zhibang Yang, Jiaran Gao, Zhixin Zhang, Tao Feng, Rihong Qiu, Wentao Zhang, Hongxin Ding, Ruizhe Zhang, Yongxin Xu, Yuheng Huang, Xu Chu, Junfeng Zhao, Yasha Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.29622)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves the factuality of large language models (LLMs), yet existing RAG systems often struggle with complex, multi-step reasoning that requires adaptive retrieval and continuous revision of intermediate contexts. Recent reinforcement learning (RL)-based agentic RAG methods partially alleviate this issue, but typically rely on coarse-grained action spaces and trajectory-level rewards, resulting in weak reward assignment and a bias toward short-horizon, stereotyped reasoning template. To address, we propose AgenticRag-R1, a RL framework that deeply integrates reasoning, retrieval, and memory via a memory stack and fine-grained action space, supported by hierarchical action-aware rewards and an information-aware trajectory rejection strategy to enable effective long-horizon learning. Experiments across a diverse set of multi-hop, open-domain, and agentic reasoning benchmarks, spanning multiple backbone model sizes, demonstrate that AgenticRag-R1 consistently outperforms strong baselines. Moreover, AgenticRag-R1 learns more robust, interpretable, and memory-aware reasoning behaviors, highlighting the effect of fine-grained action modeling and information-aware optimization for long-horizon reasoning. Our code is anonymous available at this https URL. 

---
# CineForge: Self-Improving Agents for Long-Horizon Video Generation 

**Authors**: Junxiang Liu, Lin Wang, Haiyu Shi, Hongxu Ma, Xiaoyu Yang, Chunjie Chen, Xiaoxiao Xu, Kaiqiao Zhan, Boao Wang, Shuizhou Shi, Tianyun Zhu, Jie Li, Jiangtong Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.29621)  

**Abstract**: Long-horizon story-driven video generation requires a production agent to coordinate narrative decomposition, state tracking, shot design, prompt construction, rendering, and revision across interdependent scenes. Existing adaptive video systems primarily refine requests or reusable skills, leaving recurring production failures disconnected from persistent, stage-targeted improvements across stories. We introduce CineForge, a self-evolving video-production agent framework that couples CineForge-Produce for video generation with CineForge-Evolve for cross-story policy evolution. CineForge-Produce organizes each source story into typed narrative, character, spatial, and cinematic states, uses them to coordinate asset and clip generation, and records the process as a canonical production trajectory. CineForge-Evolve applies Case-to-Pattern-to-Policy Evolution (CPPE) to review trajectory evidence, consolidate recurrent findings into bounded stage-local patches, and deploy validated updates through structural replay and confidence-controlled paired evaluation. To measure complete story realization, we introduce CineScope, which combines a 100-script CineScope-Data suite with a human-aligned, multiscale CineScope-Metric spanning causal state, directorial orchestration, pacing and resource allocation, and character arc. Across CineScope-Data and two public benchmarks, the evolved CineForge policy improves CineScope-Metric from 4.024 to 4.380, outperforms three long-video baselines with consistent gains under ScriptAgent, and reduces review LLM calls by 37.0% on new stories. These results establish production trajectories as actionable experience for video agents that improve cumulatively across long-form storytelling tasks. 

---
# Memory-First Fact-Checking: A Knowledge-Graph-Grounded Multi-Agent System for Misinformation Detection 

**Authors**: Amelia Petrenciuc, Alexandru Lecu, Adrian Groza  

**Link**: [PDF](https://arxiv.org/pdf/2608.29617)  

**Abstract**: This paper introduces a hybrid fact-checking framework that integrates Knowledge Graph-based semantic memory with adversarial multi-agent reasoning for explainable misinformation detection. The proposed system follows a memory-first, web-fallback architecture, in which input claims are initially evaluated against a dual-index Knowledge Graph through Sentence-BERT-based semantic retrieval and Natural Language Inference. When the evidence retrieved from the graph is insufficient to support a reliable decision, the framework collects information from trusted web sources and assesses it using an adversarial tribunal composed of support, contradiction, and judging agents. A graph-aware confidence mechanism combines semantic similarity, NLI confidence, and structural graph evidence to determine whether internal knowledge is sufficient, thereby reducing unnecessary web retrieval. Following verification, validated information is transformed into structured triples and incorporated into the Knowledge Graph, supporting the incremental expansion of the system's semantic memory. Experimental evaluation on a curated COVID-19 misinformation benchmark demonstrates that the proposed framework achieves an accuracy of 97.4\% and a macro-averaged F1-score of 92.6% on resolved claims, outperforming a Llama~3.3~70B baseline, which obtains an accuracy of 87.7% and a macro-averaged F1-score of 86.3%. 

---
# Forward-Deployed Full-Stack Engineering for Autonomous Cloud MLOps 

**Authors**: Sagar Srinivas Sakhinana, Venkataramana Runkana  

**Link**: [PDF](https://arxiv.org/pdf/2608.29615)  

**Abstract**: Across industries, machine-learning systems support applications ranging from prediction and anomaly detection to forecasting, optimization, and scheduling, yet operationalizing these systems requires coordinating application development, model pipelines, cloud infrastructure, security, deployment, monitoring, retraining, recovery, and rollback. We present an evidence-gated multi-agent framework for transforming a natural-language MLOps cloud engineering task into a verified repository and operational cloud deployment. The framework combines graph engineering, loop engineering, and agent harness engineering. A stateful Graph Orchestrator coordinates specialized agents for repository generation, review, execution, verification, release, and monitoring while governing workflow dependencies, evidence gates, retry bounds, recovery paths, and termination. Consequential lifecycle transitions proceed only when their required predicates are supported by verifiable execution or runtime evidence. Verification failures activate bounded reflection, repair, and re-verification, while runtime evidence of failure, drift, degradation, or policy violation can trigger bounded adaptation, recovery, or rollback. Agent harness engineering constrains repository generation, review, and repair, artifact execution, and cloud operations through controlled capabilities and isolated execution environments. We realize the framework on Google Cloud Platform and evaluate repository completeness, controlled execution, evidence-gated transitions, cloud promotion, and bounded recovery. Our experimental results show that the framework prevents unsupported lifecycle transitions and drives each run toward either a verified operational deployment or an auditable terminal failure. 

---
# Wide Learning: Learning to Reach Evidence 

**Authors**: Junzhou Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.29608)  

**Abstract**: Machine learning is usually evaluated after an evidence interface has been fixed. A dataset, sensor suite, query language, action set, or experimental protocol determines which observations can be obtained, and learning is judged by what it extracts from them. We study a complementary capability. A learner's state can determine which evidence-generating experiments it can reliably realise under bounded resources, even when primitive affordances remain fixed. We call this learner-relative experiment family its effective epistemic reach, and use Wide Learning for task-relevant learning-induced changes in that this http URL formalise effective reach relative to learner state, deployment budget, reliability threshold, and evaluation distribution. In a controlled construction, two hidden worlds have exactly the same public observation law. An informative diagnostic exists in a fixed five-primitive substrate. Before calibration, one address attempt realises it with probability at most $2^{-10} = 1/1024$, below a pre-specified 0.95 threshold; after calibration, held-out realisation is 1. Public-channel total variation is 0, whereas the realised diagnostic has total variation 1, and sealed binary risk moves from approximately 1/2 to 0. The construction establishes that learning can change effective epistemic reach even when primitive affordances and deployment resources are held fixed. It opens a complementary evaluation question for learning systems: not only what they infer from available evidence, but what informative evidence experience teaches them to bring within reach. 

---
# SUP-MIMIC: A Multi-Task Clinical Diagnosis Benchmark for Evaluating LLMs' Robustness to Contradictory Evidence 

**Authors**: Yi Yu, Bo Wang, Chong Feng, Ge Shi, Xia Liu, Ziyi Yang, Xuewen Shi  

**Link**: [PDF](https://arxiv.org/pdf/2608.29582)  

**Abstract**: Current evaluations of large language models (LLMs) primarily focus on factual knowledge retrieval, overlooking the fundamental challenge of navigating the complex, non-bijective mappings between clinical indicators and diagnoses. Existing benchmarks fail to assess whether large language models truly possess the reasoning capability required for diagnostic ambiguity scenarios, where identical clinical presentations may correspond to different etiologies, and diagnostic convergence scenarios, where heterogeneous symptoms ultimately indicate the same disease. To address this issue, we propose SUP-MIMIC, a multi-task framework utilizing MIMIC-IV-v3.1 that comprises Basic Assessment (BA), Diagnostic Divergence Task (DDT), and Diagnostic Convergence Task (DCT). Specifically, DDT is designed to evaluate the model's "one-to-many" disambiguation capability among phenotypically similar cases, while DCT assesses the model's ability to identify "many-to-one" diagnostic patterns across different pathophysiological pathways. Comprehensive evaluation of state-of-the-art LLMs reveals substantial performance degradation on DDT and DCT compared to baseline tasks, exposing a systemic reliance on statistical shortcuts over genuine causal reasoning. Our findings further highlight a conservative bias toward "healthy" predictions, implying non-trivial risks for missed diagnoses in realistic medical settings. This work establishes a rigorous methodology for quantifying clinical reasoning robustness and provides a roadmap for enhancing the safety of language models in clinical medicine. 

---
# HoopMind: A Real-Time Neural Game-Tree System for Opponent-Aware Possession Planning 

**Authors**: Yibo Gong, Cong Guo, Jiacheng Ding  

**Link**: [PDF](https://arxiv.org/pdf/2608.29563)  

**Abstract**: School coaches prepare for opponents with game film and intuition. The analytics tools of professional teams stay out of reach. We ask how far public data can close this gap. Professional basketball is our case study, chosen for its data rather than the league. We fuse five public sources into one per-shot dataset of 4.23M shots over 21 seasons. The sources are shot locations, two play-by-play feeds, official matchup tracking, and player biometrics. Alignment across them is 99.5% to 100%. We also report two data pitfalls that are easy to miss. We then model a half-court possession as a sequential game. Shot values come from ShotNet, an embedding multilayer perceptron (MLP). On a held-out season it beats a zone-rate baseline and a logistic baseline, and its probabilities are well calibrated. A depth-limited expectimax search then solves the offensive decision tree, with branch-and-bound pruning to keep it real time. All training runs offline, so the online system stays light. A scouting planner and a playable simulator both run in a single browser page. 

---
# PhysWave: Physics-Guided Latent Diffusion Models for Controllable Spatial Audio Generation 

**Authors**: Lingfeng Yao, Chenpei Huang, Xingke Yang, Ziye Geng, Changqing Luo, Hao Wang, Jiang Liu, Miao Pan  

**Link**: [PDF](https://arxiv.org/pdf/2608.29549)  

**Abstract**: Text-to-spatial audio generation, such as text-to-First-Order Ambisonics (FOA), provides a convenient way to create spatial audio for billion-dollar gaming and film industries. However, existing text-to-FOA methods are largely data-driven and may produce audio that violates acoustic relations between source direction and distance. They also separate descriptive and parametric control, forcing users to trade usability for precision. In this paper, we present PhysWave, a physics-guided latent diffusion model for controllable text-to-FOA generation. PhysWave unifies natural-language and trajectory control through a shared waypoint-caption representation, and augments diffusion training with two differentiable acoustic priors: spherical-harmonic direction consistency and inverse-square distance consistency. To support dynamic spatial generation, we further construct a 300K-clip FOA dataset with diverse sound categories and source trajectories. Extensive results show that the proposed priors help PhysWave generate spatially consistent FOA audio while maintaining competitive audio quality. Further analyses show that these physics priors improve spatial consistency during training and can also be used as inference-time guidance for training-free spatial refinement. 

---
# Evaluating LLMs on Conversational Text-to-SQL under Chain Ambiguity and Intent Drift 

**Authors**: Yujia Liu, Jiayan Lin, Zijin Hong, Zheng Yuan, Shengyuan Chen, Hao Chen, Qinggang Zhang, Xiao Huang, Feiran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.29543)  

**Abstract**: Recent advances in large language models (LLMs) have established conversational text-to-SQL as a practical interface between users and databases, often involving multiple turns of clarification and revision. However, existing benchmarks primarily evaluate execution accuracy, leaving the unfolding and shifting of user intent across turns largely uncovered. To address this, we introduce TIDE-Bench, a benchmark for conversational text-to-SQL under chain ambiguity and intent drift evaluation, targeting two recurring patterns: chain ambiguity, where an underspecified question triggers layered clarification with conditional dependencies, and intent drift, where the user retracts and replaces a previously committed request element. Built on 514 anchor SQLs from BIRD, TIDE-Bench comprises 1,542 samples and introduces dedicated metrics for chain identification and drift recognition-resolution beyond execution accuracy. Evaluating 12 advanced LLMs reveals a persistent chain identification bottleneck unaffected by clarification frequency, a wide drift recognition-resolution gap, and overlap between failure modes when jointly activated. The corresponding code of TIDE-Bench is released for further research. 

---
# AGM: Achievement-Grounded Memory for Closed-Loop Agents with Frozen VLA Policies 

**Authors**: Hongbo Gao, Zeyu Ni, Xin Wen, Siyu Xu, Ruifeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.29537)  

**Abstract**: Frozen vision-language-action (VLA) policies offer broad manipulation skills but execute open-loop action chunks without tracking task progress, so the agent cannot reliably decide whether to continue, retry, or terminate. External memory is a natural remedy, yet it can be harmful when attempted actions are treated as completed progress, turning local execution errors into persistent task-state errors. We propose Achievement-Grounded Memory (AGM), a lightweight closed-loop framework for frozen VLA policies that represents a task as a subgoal sequence with a progress pointer and advances this memory only after the current subgoal is verified by physical evidence. Proprioceptive interaction cues decide when to verify, while coherent point tracking and language-conditioned cross-view comparison, sourced from frozen foundation models through a single 2.43M-parameter verification head, decide what was achieved. AGM thereby converts open-loop execution into a closed loop of execution, verification, and progress, keeping the policy frozen without test-time large-model inference. On the RoboMME Counting benchmark, AGM reaches on PickXTimes and on BinFill, surpassing the strongest memory-augmented baseline by points on average, and the framework yields equally decisive gains on a physical robot. Reliable embodied memory thus depends more on disciplined state updates than on memory capacity. 

---
# Integrating adaptive human behavior into epidemic models with large language models 

**Authors**: Yicheng Mao, Haoyang Li, Rob Deardon, Hongru Du  

**Link**: [PDF](https://arxiv.org/pdf/2608.29535)  

**Abstract**: Infectious disease transmission is shaped by patterns of human interaction, which adapt as epidemic conditions change. Capturing these context-dependent behaviors remains a fundamental challenge for epidemic models. Here, we recast this challenge by using large language models (LLMs) to represent adaptive human behavior within mechanistic epidemic models. We operationalize this idea through Generative Adaptive Behavioral Layer for Epidemics (GABLE), which adapts LLMs to infer behavioral responses to epidemic and policy conditions and translates them into age-structured contact matrices coupled to a mechanistic epidemic model. Applied to COVID-19 in France, GABLE reproduced responses in population mixing and age-specific contact structures that remained epidemiologically informative. In short-term forecasting, LLM-generated contact matrices outperformed mobility-driven matrices derived from real-world mobility data, with the largest gains at longer horizons. GABLE also extends beyond forecasting to prospective policy evaluation by projecting behavioral and epidemic responses to candidate interventions before implementation. When supplied with subsequently implemented policies, GABLE reproduced epidemic trajectories and generated distinct responses to alternative policy timing and composition. By leveraging LLMs as a flexible behavioral layer, GABLE provides a framework for coupling context-sensitive behavioral generation with epidemic dynamics. 

---
# The Emergent Symbolic Structure of Artificial Neural Networks 

**Authors**: R. Thomas McCoy, Paul Soulos, Tal Linzen, Paul Smolensky  

**Link**: [PDF](https://arxiv.org/pdf/2608.29530)  

**Abstract**: Modern systems in artificial intelligence (AI) somehow excel in domains for which they seem poorly suited. Intelligence has traditionally been modeled as operating over structured combinations of symbols, such as logical formulas. However, the strongest modern AI systems are based on neural networks, which instead represent information in continuous vectors. Vectors seem inadequate for capturing the structure of language, logic, and other cognitive domains, yet neural networks achieve impressive performance in these areas. How do they do it? In this work, we propose a potential answer: Despite appearances, perhaps the internal representations of neural networks implicitly realize symbolic structure. In support of this hypothesis, we show that the vector representations of a variety of neural networks can be closely approximated with symbolic structures: we can replace the network's entire representation-generating process with a closed-form equation instantiating a symbolic structure, and the network's behavior remains largely unchanged. This finding holds for both small-scale neural networks trained to manipulate lists as well as large language models (LLMs) operating in four domains that are central in symbolic traditions: arithmetic, logic, computer code, and language. Further, our symbolic approximation allows us to modify an LLM's behavior in targeted ways via precise interventions on its internal representations, showing that the LLM's behavior is reliant on the symbolic structures we have identified. This work provides a potential way to reconcile longstanding symbolic conceptions of intelligence with the vector-based nature of modern AI. 

---
# Argument-Aware Semantic Alignment of Normative Texts: A Toulmin-Based Neuro-Symbolic Approach 

**Authors**: William Schroeder  

**Link**: [PDF](https://arxiv.org/pdf/2608.29529)  

**Abstract**: Semantic alignment between specialized normative texts is challenging when equivalent requirements use different terms, syntax, and levels of abstraction. Lexical overlap, distributional embeddings, and semantic similarity capture topical relatedness but often miss the argumentative structure by which normative claims are supported, qualified, and justified. This paper asks whether explicit argument structure adds information complementary to neural semantics for aligning requirements. We treat cross-standard control mapping as argument-aware semantic alignment and build a neuro-symbolic pipeline that combines neural text representations with Toulmin features. An LLM explicitation step identifies claims, grounds, warrants, qualifiers, and backing and reconstructs enthymemes. These feed an alignment model via argument-aware similarity and structural features. On a NERC-CIP to NIST-CSF mapping benchmark, argument-derived features improve alignment over a neuro-symbolic semantic baseline. Feature selection shows especially strong signal from warrant-related features, indicating that the link between a claim and its supporting reasoning is not captured by conventional similarity alone. A compact claim--grounds--warrant subset remains competitive with the full Toulmin feature set. The results give preliminary evidence that argument structure is a useful intermediate representation for aligning specialized normative texts. Cybersecurity standards are used as a controlled testbed, not as proof of domain-independent generalization. The argument graphs produced by LLM explicitation may also support later work on retrieval, reasoning, and explanation over normative text. 

---
# On the Plasticity Collapse in Continual Machine Unlearning 

**Authors**: Yingdan Shi, Xiang Xu, Kaize Ding, Alfred O. Hero, Ren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.29513)  

**Abstract**: Machine unlearning enables deep neural networks to selectively remove the influence of specific data in response to privacy and regulatory requirements. While prior work largely studies single-shot unlearning, real-world systems must accommodate continual unlearning, where multiple unlearning requests occur sequentially over time. In this work, we identify a fundamental limitation of this setting: plasticity collapse, a progressive breakdown in a model's ability to effectively forget. Through theoretical analysis of continual unlearning dynamics, we show that continual unlearning operations accumulate geometric constraints in parameter space, leading to saturated subspaces that restrict future updates. This structural effect induces two distinct failure modes: (1) Forward failure -- diminishing forgetting quality for subsequent tasks, and (2) Backward failure -- spontaneous re-memorization of previously forgotten information. Extensive experiments across multiple architectures, datasets, and methods in image classification confirm that plasticity collapse is not an artifact of specific implementations, but a pervasive phenomenon inherent to continual unlearning. Our findings reveal a critical barrier to the long-term reliability of machine unlearning systems and motivate the development of plasticity-preserving unlearning algorithms. Our code is available at this https URL 

---
# Denoising as Projection: Constrained Optimization with Gradient-Guided Diffusion 

**Authors**: Runyu Zhang, Jiawei Zhang, Gioele Zardini, Saurabh Amin, Asuman Ozdaglar  

**Link**: [PDF](https://arxiv.org/pdf/2608.29507)  

**Abstract**: Diffusion models are increasingly used not only for sampling from learned data distributions, but also for generating samples that optimize task-specific objectives. A common approach is to guide the reverse diffusion process using gradients of an external objective. However, when the data distribution is supported on a structured feasible set, such as a manifold or a constraint set, gradient guidance can move samples away from the learned data geometry. In this paper, we study a simple projected-gradient-guided diffusion update based on the observation that the Stein denoising operator can act as an approximate projection onto the data geometry. The proposed update incorporates the objective gradient inside the denoising step, yielding an inference-time method that uses only a pretrained denoiser and gradient evaluations. We analyze this update as an inexact projected-gradient method for constrained optimization over learned feasible geometries. Our theory covers three settings: linear manifolds, compact convex feasible sets, and compact Riemannian submanifolds. In all these settings, we prove descent and finite-time convergence guarantees. Numerical experiments support the theoretical interpretation and illustrate how the proposed update balances objective descent with preservation of the learned geometry. 

---
# Generalizable Multi-Agent Planning from Signal Temporal Logic Specifications via Diffusion 

**Authors**: Joe Eappen, Zikang Xiong, Shreyash S. Iyengar, Suresh Jagannathan  

**Link**: [PDF](https://arxiv.org/pdf/2608.29490)  

**Abstract**: Multi-agent systems in the real-world (e.g., drone swarms, autonomous cars, warehouse robots) must satisfy rich, temporal tasks while avoiding collisions. Signal Temporal Logic (STL) elegantly encodes such objectives, but current STL planning methods face critical limitations. State-of-the-art optimization-based approaches can handle arbitrary STL specifications but struggle with scalability, becoming computationally impractical as the number of agents grows. Learning-based methods efficiently handle a large number of agents with rapid planning times but fare poorly when deployment-time objectives differ from those used during training, and do not support planning tasks that require different specifications to be ascribed to different agents (i.e., heterogeneity) or team-level specifications requiring coordination of multiple agents. This fundamental trade-off between generalizability and scalability presents a challenge for realizing multi-agent STL planning algorithms in practice. To overcome this challenge, we introduce a new diffusion method for multi-agent planning with STL specifications. Using a differentiable approximation of STL, we integrate the STL gradient in the denoising process, making our approach generalizable to novel formulas whose predicates are placed anywhere within the goal region covered during training, while achieving the same scalability as existing learning-based methods. Our method supports heterogeneous specifications, and by using diffusion models, naturally enhances plan diversity, thereby significantly reducing safety-related violations (e.g., collisions) among agents. A detailed evaluation study justifies the utility of STL-guided diffusion-based multi-agent planners for constructing generalizable, scalable, and diverse plans. Videos and code are available at this https URL and this https URL . 

---
# Applications of Risk Science to AI Fairness Evaluation: Principles, Challenges, and Best Practices 

**Authors**: Kyra Wilson, Sabrina Kang, Saloni Dash, Aylin Caliskan  

**Link**: [PDF](https://arxiv.org/pdf/2608.29478)  

**Abstract**: Scholarly work which aims to describe potential societal impacts (e.g., risks) of proliferating technology (especially related to artificial intelligence or other algorithmic systems) is likely to have an impact beyond the scientific communities it was written for, given that general society itself is a primary object of study. However, it is an open question whether the current practices of AI evaluation scholarship follow the principles and best practices established by risk science, which aims to systematically generate knowledge related to understanding, assessing, communicating, managing, and governing risk. In this work, we examine this in depth by conducting a literature review of scholarly works purporting to evaluate the bias or fairness of technological systems used for tasks related to hiring and employment. Through analysis of 22 common fairness evaluation metrics and studies using them, we find that most characterize the severity of bias- or fairness-related consequences but do not follow best practices to characterize the uncertainty around either the occurrence of these consequences or severity estimates. Next, we conduct a case study of fairness evaluation for an AI-mediated resume screening task and demonstrate how principles of risk science can be incorporated into such an evaluation. Finally, we propose the AI Risk Report Card, which facilitates the reporting and communication of risk assessment results to stakeholders in positions to act based on the predicted risks. The outcomes of these activities suggest that further research at the convergence of risk science and AI evaluation can lead to advancements in AI assessments of societal impact by enabling shared frameworks to evaluate and discuss AI risks both within and outside of the scientific community. 

---
# MUDDLE: Measuring Understanding of Documents under Distractor and Length Effects 

**Authors**: Jason Luo, Saibilila Abudukelimu, Judy Song, Andrew Feng, Shivank Garg, Vasu Sharma, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.29477)  

**Abstract**: Document question-answering systems increasingly answer questions over collections of retrieved documents rather than one clean source, so robustness to distracting context matters as much as reading ability. When such systems fail, it is often unclear whether the context was too long or the distractors were too close to the topic, because prior work tends to conflate these two effects. We present MUDDLE, a controlled benchmark that separates them. MUDDLE uses 270 human-annotated questions, each tied to a single source document, and instantiates every question in five conditions: the source alone, the source with two or four topically similar hard negatives, and the source with two or four random distractors. The random distractors are matched to the hard negatives in length and provenance, so an accuracy gap between the two arms reflects topical similarity rather than length. All five conditions are rendered in markdown, page images, and raw PDF, but the distractor sweep reported here is run in markdown, since a source plus its distractors exceeds current image and PDF input limits. We score answers with an LLM judge across three model families. In the complete markdown sweep, hard negatives lower accuracy more than length-matched random documents at both context sizes for gpt-5-mini, while random documents stay near the no-distractor baseline. The effect is small but directionally consistent, and for gpt-5-mini hard negatives significantly underperform length-matched random distractors when pooled across context sizes. We release the data and evaluation code for a reproducible study of context degradation. 

---
# Knowledge Distillation under Teacher Misspecification: An Order-Parameter Analysis of the Gap between Teacher Mimicry and Task Performance 

**Authors**: Kazuyuki Hara, Hideitsu Hino  

**Link**: [PDF](https://arxiv.org/pdf/2608.29472)  

**Abstract**: Knowledge distillation trains a small student model to reproduce the outputs of a large teacher model, and its progress is typically monitored through the teacher--student discrepancy. The quantity of ultimate interest, however, is the student's error with respect to the true task. We study the relation between these two objectives in a minimal three-party model, a true teacher (generative model), a teacher, and a student, all soft committee machines, in which the true teacher contains a shared latent factor that the teacher cannot represent, with mismatch strength controlled by a single scalar $\dmiss$. Within an order-parameter description of online distillation, and exploiting closed-form (arcsine-type) expressions for all errors under error-function activations, we prove that the learning dynamics and the distillation error $\Ets$ are exactly invariant to $\dmiss$, whereas the true error $\Etzs$ and the gap $\Delta=\Etzs-\Ets$ are strictly increasing in $\dmiss$, with a rate that is amplified linearly by the complexity $M_0$ of the true teacher. Numerical phase diagrams over the plane spanned by true-teacher complexity and student capacity confirm the predicted deformation: the contours of $\Ets$ do not move while the landscape of $\Etzs$ rises systematically, and a teacher-miss regime, where mimicry succeeds but the task fails, expands with $\dmiss$. The results give a quantitative warning against evaluating distillation solely through teacher-mimicry metrics and identify the gap $\Delta$ as a minimal diagnostic for distinguishing teacher-miss from capacity-limited failure. 

---
# Chain-of-Thought Faithfulness of Reasoning Models Varies with Where and How Preference Cues Are Delivered 

**Authors**: Aryo Pradipta Gema, Neel Rajani, Rohit Saxena, Wai-Chung Kwan, Pasquale Minervini  

**Link**: [PDF](https://arxiv.org/pdf/2608.29464)  

**Abstract**: Chain-of-thought (CoT) monitoring assumes that reasoning traces faithfully record the information that shapes a model's answer. Existing faithfulness tests often place explicit bias cues in the user message, while agents may encounter preferences through tool returns or raw artifacts. We introduce FACE-Eval (Faithful Attribution of Cue Effects Evaluation), a 5,100-sample evaluation that varies cue location (user message or tool return) and explicitness (direct summary or raw artifact). We measure verbalized commitment among cue-following answers and unverbalized adoption among all cued samples. We evaluate 15 open-weight models from eight families, with total parameters ranging from 4B to 1.60T. Every model has lower verbalized commitment for tool-return than user-message cues and for implicit than explicit cues. Unverbalized adoption is higher for tool-return cues on all 15 models and for implicit cues in 28 of 30 model-channel comparisons. A source-attribution prompt narrows the channel gap on seven models, sometimes by increasing user-channel unverbalized adoption, while telling models that their reasoning will be monitored does not reliably close the gap. We also use two transcript monitors (GPT-5.6-Luna and GPT-4o-mini) to detect preference adoption in the largest model of each family. Across 32 model-channel-explicitness cells, higher unverbalized adoption is associated with lower detection ability for both monitors (Pearson r=-0.54 and r=-0.78, respectively). These results suggest that CoT monitoring may be less reliable when preference information arrives through tools or must be inferred from raw artifacts, within the single-call, prefilled-tool setting tested here. 

---
# Benchmark Contamination: A Taxonomy Organized by Defeated Mitigation 

**Authors**: Johanna Angulo, Víctor Yeste, Hector Espinos-Morato  

**Link**: [PDF](https://arxiv.org/pdf/2608.29463)  

**Abstract**: A benchmark score is a joint property of the model, the evaluation harness, the elicitation budget, the sampled population, and contamination status. Leaderboards publish the model and the score, so capability and leakage stay observationally equivalent. Existing taxonomies classify contamination for automated detection, not the question a reporter faces at publication: given the mitigations already applied, which validity threats remain open? We introduce a taxonomy organized by the mitigation each type defeats -- direct, derivative, temporal, distributional, and acquired -- spanning training-time and evaluation-time leakage. Holding out a private test set closes the first alone. The fifth is acquired during the evaluation itself; because it is a property of one run, it must be recorded with the reported score rather than with the benchmark release. We operationalize it as a four-field disclosure protocol in which "unknown" is a valid entry, released under CC BY 4.0 with a JSON Schema, a validator, and worked examples. Two coders external to the design team applied a pre-registered instrument to 41 documents. Per-variable linear-weighted $\kappa$ runs from 0.00 to 0.35 (median 0.21) over 29 main-pass documents against a single-coder test-retest ceiling of 0.84, collapsing under the class skew the registration anticipated; pooling raises it to 0.46 through chance correction rather than better agreement. Two variables fall below the prevalence-robust threshold registered in advance: strata reporting and the acquired type introduced here. Disagreement concentrates on when a variable applies rather than on what a document states. Elicitation budgets are reported in 13% of documents, and no document addresses all five types. The contribution is the taxonomy, the score-side artifact that follows from it, and a pre-registered measurement of instrument reliability and current disclosure. 

---
# Reference-Grafting Matches Fine-Tuning at Eliciting Sandbagged Capabilities 

**Authors**: Linh Le, Hong Kiat Tan, David Williams-King  

**Link**: [PDF](https://arxiv.org/pdf/2608.29458)  

**Abstract**: Sandbagging, in which a model deliberately underperforms on an evaluation despite retaining the underlying capability, threatens the safety evaluations that frontier-model governance depends on. The Elicitation Game found that fine-tuning elicits hidden capability from sandbagging model organisms whereas additive activation steering fails. We revisit that verdict with reference-grafting, which sets an activation's coordinate along a contrast direction to the value it takes in an honest reference, at a small set of circuits chosen by active learning. Across eleven fine-tuned password-locked models (1.5-32B, three architecture lineages), it recovers +94 to +101% of the honest-sandbagging gap, matching fine-tuning elicitation without weight updates or training labels; two to five paired examples suffice to fit the direction. Similar recovery holds for reinforcement-learning-induced sandbagging and for password-locked code generation. Grafting works because the fine-tuned lock is a thresholded gate: held-out accuracy stays at the sandbagged level until the grafted coordinate crosses a threshold near the honest reference, which is why additive steering and zeroing the coordinate often fail. The direction tracks how the sandbagging was induced rather than what is withheld -- fit on grade-school science (ARC) it elicits withheld hazardous knowledge (WMDP), yet prompting, fine-tuning, and reinforcement learning each carry a different direction. Circuit-breaking marks the boundary: it reroutes activations on every forward pass, so the fixed edits we test are re-broken downstream and do not restore coherent generation. 

---
# AI Can Be Easily Persuaded in Clinical Decision Making 

**Authors**: Jiayuan Zhu, Jiazhen Pan, Fenglin Liu, Minhao Hu, Junde Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.29453)  

**Abstract**: As AI becomes increasingly integrated into clinical practice, it is playing a growing role in medical decision making. Medicine, however, is a high stakes and evidence based field, where decisions can directly affect patients' lives. It is therefore important to understand whether AI can maintain objective judgment when others try to persuade it. In this paper, we study how easily AI can be persuaded through controlled experiments. We find that professional authority, national background, institutional affiliation, claimed past performance, multiple physicians, supported clinician views, and repeated pressure can all affect AI decisions. Surprisingly, the same persuasive input changes about 10% more cases when it comes from a senior clinician than from a medical student. Simply claiming a better performance history consistently makes the physician more persuasive. More strikingly, a plausible clinician view can persuade AI away from a correct decision even when it is fabricated to support an incorrect answer. This indicates that AI can be strongly influenced by convincing support without reliably determining whether this view from the clinician is correct. Together, these findings suggest that AI can be easily persuaded by what people say, who says it, and how the opinion is presented. Therefore, it is essential for AI to maintain sound judgment under persuasion, enabling its safe and reliable use in high stakes medical decision making. 

---
# SS-ESOAP: Self-Scaled Adaptive Preconditioning for Physics-Informed Learning 

**Authors**: Guangyuan Wang, Mads Toftrup, Sebastian Loeschcke, Yixuan Wang, Anima Anandkumar  

**Link**: [PDF](https://arxiv.org/pdf/2608.29448)  

**Abstract**: Physics-informed neural networks (PINNs) often face ill-conditioned objectives that limit high-accuracy training. Dense quasi-Newton methods improve local conditioning but require expensive optimizer state, while Kronecker-factored methods such as SOAP scale to larger networks but rely on periodic basis updates. We introduce \method, which augments SOAP-style preconditioning with a scalar secant-energy correction adapted to Kronecker geometry and an adaptive basis update followed by variance-state downscaling. We characterize the directional secant matching induced by the scalar correction and give a bound on variance-state mismatch across basis changes. Across eight PDE benchmarks, \method attains the lowest final residual on six, including Burgers and Boussinesq, while SOAP-family baselines perform better on Gray-Scott and Ginzburg-Landau. On Boussinesq, \method reaches a residual of $10^{-5}$ in 4.1 hours with 9.2 GB peak VRAM, while Adam does not reach this target within 14 hours. Three-seed $L^2$ and $H^1$ errors on four representative PDEs support the link between lower residuals and improved solution accuracy. These results position \method as a scalable option for stiff, high-accuracy physics-informed training, rather than a uniform replacement for existing optimizers. 

---
# Does Latent Planning Survive Point Clouds? Action-Conditioned JEPA World Models for Geometric Observations 

**Authors**: Fabio F. Oberweger, Michael Schwingshackl  

**Link**: [PDF](https://arxiv.org/pdf/2608.29434)  

**Abstract**: JEPA world models make latent-space planning a practical route to control, but they are built almost exclusively on images. Whether latent prediction survives geometric observations is unclear: point clouds are sparse, unordered, and self-occluded, and with 0.3-15% of scene points moving, the slow-feature optimum of latent prediction compounds with the geometric shortcut of 3D self-supervision. We lift three canonical JEPA designs to point clouds, frozen-encoder, distribution-prior, and action-sensitive, and re-sense the stable-worldmodel benchmark so that only the observation differs from the image baselines. All three plan without collapse: the distribution-prior model is statistically equivalent to its re-evaluated image counterpart on every benchmark, and the action-sensitive model attains the strongest result in our controlled comparison where the most geometry moves. Probing explains why: object positions are almost perfectly linearly decodable and attention falls on the few moving points. Planning withstands heavy dropout never seen in training, though range noise defeats the thinnest scene. Geometry finally makes a commanded 3D target a natural goal interface: we construct the goal latent from the target and the current latent, at no cost in success rate, without a goal observation. 

---
# Polis: 3D Self-Supervision at City Scale 

**Authors**: Alexander Rusnak, Sophia Kovalenko, Jingru Wang, Ismail Moudden, Xiru Wang, Frédéric Kaplan  

**Link**: [PDF](https://arxiv.org/pdf/2608.29426)  

**Abstract**: Reliable semantic representations derived from city-scale 3D models are increasingly important for urban analysis, infrastructure monitoring, autonomous systems, and heritage conservation. However, urban scenes of large spatial extent captured through aerial surveying differ substantially from the indoor, object-level, and self-driving LiDAR data used to pretrain most 3D self-supervised models. We introduce Polis, to our knowledge the first application of Sketched Isotropic Gaussian Regularization (SIGReg) as an objective for a native point cloud encoder, and evaluate it through a frozen-feature benchmark spanning fourteen city- and building-scale corpora. Polis combines geometrically matched cosine invariance, SIGReg, and VICReg-style anti-collapse terms with a 12.8k-scene outdoor pretraining mixture and gravity-preserving spatial view sampling. Controlled ablations show that this objective outperforms student--teacher architecture alternatives, as well as Polis versions without anti-collapse terms, on the same representative outdoor corpus. On three pretraining-disjoint city datasets, Polis reaches $23.8\%$ mean mIoU versus $16.3\%$ for the next-best encoder under high-capacity frozen probing, and $17.3\%$ versus $16.1\%$ at a matched point and voxel budget. The same city-scale lead holds on datasets whose training sets were seen in pretraining. On localized terrestrial captures with fine-grained facade and streetscape labels, the ranking reverses. Our results show that distributionally-regularized joint embedding architectures can be successful on challenging city-scale 3D scenes, and that transfer improves when self-supervision is designed for the capture geometry and spatial context of this domain while also revealing the limits of this specialization. 

---
# Scalable Clinical Data Infrastructure and Comparative ML Evaluation for Hospitalisation Risk Prediction in Elderly Patients with Multiple Long-Term Conditions using CPRD 

**Authors**: Asra Aslam, Volodymyr Chapman, Maurice M. O'Connell, Aseel S. Abuzour, Michael Abaho, Danushka Bollegala, Gary Leeming, Eduard Shantsila, Andrew Clegg, Lauren E. Walker, Iain Edward Buchan, Samuel D. Relton  

**Link**: [PDF](https://arxiv.org/pdf/2608.29419)  

**Abstract**: Deep learning architectures are increasingly proposed for patient trajectory modeling in electronic health records (EHRs), yet their advantage over simpler, more interpretable models is rarely subjected to rigorous empirical scrutiny in real-world clinical settings. We present a comprehensive patient timeline pipeline applied to elderly patients in CPRD Aurum, incorporating 260 clinical conditions classified via a three-tier automated framework including specialised detection logic for 17 complex conditions. Using this infrastructure, we benchmark Temporal Graph Convolutional Neural Networks (TG-CNN) against Logistic Regression with LASSO regularisation and Random Forests for predicting 12-month all-cause emergency hospitalisation risk, motivated by (but not filtered to) the elevated risk of adverse drug reactions. Under cross-validation, TG-CNN achieves a marginally higher mean AUC-ROC than LASSO (0.712 vs. 0.705), whereas on the held-out test set LASSO achieves the highest discrimination of three models (AUC-ROC 0.733, versus 0.710 for Random Forest and 0.702 for TG-CNN). We show, that discrimination alone is an incomplete criterion for clinical deployment: after Platt calibration, LASSO is the only model with an acceptable calibration slope (0.817), while Random Forest (0.759) and, TG-CNN (0.391) remain substantially miscalibrated. We argue that LASSO, not the highest-discriminating model, is the model best suited to direct clinical deployment. We present lessons for the machine learning and healthcare community regarding data infrastructure, model selection, and value of calibration and interpretability in high-stakes decision support. 

---
# Fully Distributed GNE Algorithms for Multi-Robot Placement without Consensus on Multipliers 

**Authors**: Shao-An Yin, Mingyi Hong, Nicola Elia  

**Link**: [PDF](https://arxiv.org/pdf/2608.29388)  

**Abstract**: Recent machine learning research has increasingly focused on equilibrium analysis in non-cooperative games rather than solely on optimal solutions. Many such problems involve shared constraints and can be formulated as Generalized Nash Equilibrium Problems (GNEPs). For strongly monotone games, existing methods compute consensus-based variational GNEs (v-GNEs) by exchanging Lagrange multipliers. We propose a fully distributed continuous-time algorithm for shared linear equality constraints that converges without multiplier exchange and reaches any GNE, reducing communication overhead and improving privacy. Discrete-time schemes are also provided, and the method is validated on a multi-robot placement task. 

---
# Safe to Resume? Breaking Execution Continuity of Agent Execution via Rollback 

**Authors**: Guanlong Wu, Dahui Li, Ke Jiang, Jianyu Niu, Cong Wang, Yinqian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.29381)  

**Abstract**: AI agents are moving toward persistent, stateful execution across various applications, accumulating execution state and external effects that are costly to reconstruct after failures. Checkpoint and rollback (C/R) are becoming essential for recovery, yet their security implications remain largely unexplored. Correct rollback does not imply secure recovery: a faithfully restored checkpoint may resume an execution whose states, assumptions, and external effects never coexisted in any valid history. In this paper, we present the first systematic security study of checkpoint and rollback in existing agent systems. By examining representative agent C/R systems, we characterize the design space of existing C/R mechanisms and develop a general execution model that captures their recovery boundaries and state dependencies. From this model, we identify five fundamental failure modes spanning incomplete or inconsistent internal state, stale external dependencies, nondeterministic replay, and unrecorded external effects. We further demonstrate their security impact through three end-to-end attacks on Hermes, Cline, and LangGraph, enabling malware-verification bypass, unauthorized mail forwarding, and double payment. To systematically study these failures in practice, we develop a multi-agent analysis pipeline that reconstructs execution semantics, identifies violations of the five failure conditions, and validates them through actual rollback. Across five representative frameworks, our evaluation shows that these failures recur across heterogeneous C/R designs and stem from a common gap between the state restored by a checkpoint and the dependencies required for secure continuation. 

---
# Arabic Safety Alignment as Selective Refusal: An Empirical Study of SFT, DPO, and Guard Calibration 

**Authors**: Mohamad Zbib, Ammar Mohanna  

**Link**: [PDF](https://arxiv.org/pdf/2608.29378)  

**Abstract**: Arabic large language models must refuse harmful prompts without over-refusing benign or sensitive prompts, yet a single refusal rate hides this trade-off. We evaluate it using benign refusal B and harmful-prompt refusal H, where H measures refusal rather than harmful compliance. Across five Arabic-capable models and 130 runs on the full human-written AraSafe set, refusal-only supervised fine-tuning (SFT) collapses toward blanket refusal, whereas selected mixed-SFT configurations reach H = 90% to 93% at B = 14% to 23%; four selected configurations exceed the H = 90% target in all three runs, while Fanar does so in two of three. Direct Preference Optimization (DPO) and inference guards change B and H differently across models rather than acting as uniform upgrades. In a blinded 300-response audit, annotator binary-refusal agreement is 89.0% (kappa = 0.78); Qwen3Guard and Aya Expanse 32B reach 88.7% and 91.0% accuracy, respectively, with no conclusive paired difference. Selected SFT raises H on Arabizi for all five models, but none reaches 90%, showing only partial transfer from Modern Standard Arabic. Overall, the results support model-specific operating-point selection: set a deployment target and retain only interventions that improve it. 

---
# StageWell: A Process-Aligned Chinese Corpus for Positive-Psychology Support Dialogue 

**Authors**: Yuxiong Wang, Ziwei Lin, Bo Wang, Yu Zhang, Shiguang Ni  

**Link**: [PDF](https://arxiv.org/pdf/2608.29326)  

**Abstract**: Positive psychology dialogue aims to support emotional distress and positive resource building, requiring models to produce not only empathetic replies but also coherent progression through a multi-turn support process. Existing resources often reduce supervision to turn-level strategies or holistic preference labels, leaving process position, support function, and local repair targets implicit. We introduce StageWell, a process-aligned Chinese corpus for positive psychology dialogue, together with HQS, a structured protocol for data construction and evaluation. StageWell organizes support into a six-stage support process and uses a multi-agent whole-dialogue rewriting workflow to construct 12,445 SFT instances, 1,849 DPO preference pairs, and a GroundTruth subset of 120 expert-revised dialogues and 977 QA pairs. Guided by HQS, DPO pairs are built as process-localized repairs: flawed model outputs are used as rejected responses, and targeted rewrites under the same context and stage constraint are used as chosen responses. Across four 9B-14B open-source LLMs, this supervision yields robust gains in process control, response quality, and safety. Averaged across models, BERTScore improves by 0.037, Q-Overall increases by 1.32 points, S-exact increases by 0.236, and the H-critical rate decreases by 0.167. These results highlight the value of modeling supportive dialogue as a structured multi-turn support process rather than as single-turn response generation. 

---
# SGE: Semantically-Guided Exploration for Unstructured Environments via Image-Space Waypoint Sampling 

**Authors**: Christopher Tatsch, Yu Gu  

**Link**: [PDF](https://arxiv.org/pdf/2608.29315)  

**Abstract**: This work introduces Semantically-Guided Exploration (SGE), a modular exploration framework for ground vehicles that integrates pixel-level semantic segmentation into sampling-based waypoint selection and receding-horizon route optimization. Unlike conventional geometric exploration methods, SGE evaluates candidate exploration goals directly in the image space using a semantic-aware utility function that accounts for terrain traversability, obstacle proximity, objects of interest, and depth-based exploration reward. Sampled waypoints are projected into 3D and ordered through a real-time Traveling Salesman Problem (TSP) formulation, enabling receding-horizon goal selection. To address real-world navigation uncertainty, the framework introduces mechanisms, including temporary taboo regions to handle navigation failures and a graph-based relocation strategy for efficient backtracking across explored areas. We evaluate SGE in standardized simulation benchmarks against state-of-the-art exploration planners and demonstrate competitive performance in volumetric coverage, while enabling semantic task biasing that cannot be achieved by purely geometric methods. The framework is further validated through real-world experiments using multiple robotic platforms in indoor campus buildings and in limestone and coal mines. Results show consistent performance and adaptability across platforms and domains. 

---
# Hyper3-CLIP: Hierarchy-Conditioned Hyperbolic Vision-Language Training 

**Authors**: Matin Mahmood, Antonio Rueda-Toicen, Mohamed ElBassat, Seifeldin Elkerdany, Weixing Wang, Gerard de Melo  

**Link**: [PDF](https://arxiv.org/pdf/2608.29313)  

**Abstract**: CLIP-like vision-language models (VLMs) trained with contrastive objectives learn strong global image-text representations, but their Euclidean embeddings and global pooling fail to encode relational structure such as part-whole and parent-child relations. Hyperbolic VLMs address this gap with entailment-based objectives, and text-conditioned variants improve fine-grained alignment through sentence- and phrase-level queries. However, these two lines of work remain separate: hyperbolic VLMs use static image and region features, while query-conditioned methods lack hierarchical geometric structure. We present Hyper3-CLIP, a hierarchy-conditioned hyperbolic VLM that combines global, local, and global-local contrastive learning with query-conditioned visual pooling. To train the model, we construct lightweight query hierarchies from text, comprising full captions, sentence fragments, localized part descriptions, and extracted phrases. Each query conditions the pooling of visual patches, and the resulting representations support image-text, whole-part, and parent-child entailment losses. Query-conditioned pooling is active only during training. Hyper3-CLIP improves R@5 and R@10 retrieval on COCO and Flickr, as well as multi-label classification on VOC and COCO, while remaining competitive on hierarchy metrics. We also audit zero-shot prompt sensitivity under fixed prompt regimes and study the effect of the localized GRIT part budget used during training. Code is available at this https URL. 

---
# Super Library Agent: Joint Generation and Maintenance of Multiple Applications Beyond the Single Codebase 

**Authors**: Daegyu Sung, Yukyeong Lee, Geon Park, Yumin Choi, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2608.29310)  

**Abstract**: Organizations often develop and maintain portfolios of related applications: independently deployable codebases that share substantial domain logic, interface patterns, or operational conventions. As LLM coding agents are increasingly used to generate and maintain such software, a naive application-by-application workflow duplicates shared logic across codebases and allows prolonged agentic maintenance to accumulate verbosity, dead code, and structural erosion. We introduce the Super Library Agent problem, where an agent sequentially generates a portfolio of N related applications while maintaining a shared Super Library of reusable cross-application components. A minimal sequential scaffold can in principle extract shared code and migrate applications to the evolving library, but in practice suffers from low extraction recall and fragile dependency migration. We address these failures with candidate-guided extraction over code chunk summaries, pre-extraction codebase consolidation, and context-aware migration using extraction traces and call-graph information. Across WebGen-Bench and PaperBench, our method preserves application functionality while significantly reducing redundancy and token footprint (verbosity, token length) over zero-shot, and avoiding the structural erosion introduced by naive library construction, with additional reductions in LOC and MDL. Our code is available at this https URL. 

---
# Evaluating LLM-based AI agents integrated with materials synthesis tools: the case of atomic layer deposition 

**Authors**: Angel Yanguas-Gil  

**Link**: [PDF](https://arxiv.org/pdf/2608.29309)  

**Abstract**: This work provides an overview of the different strategies that can be used to evaluate the performance of AI models and agents based on large language models (LLMs) for materials synthesis. After providing a brief overview of the key technologies behind the current generation of AI agents based on LLMs, we summarize the different approaches to evaluating these models in the context of materials science and in particular on materials synthesis, with a specific emphasis on scenarios in which the models are directly integrated with experimental tools. We discuss evaluation strategies spanning knowledge and reasoning benchmarks, tool-use benchmarks, and closed loop benchmarks involving the interaction with experimental systems or realistic virtual tools. We use atomic layer deposition (ALD) as a case study, emphasizing how existing approaches in the literature both build from general approaches used beyond materials science and can be generalized to other materials synthesis techniques. Finally, we provide a practical evaluation framework to evaluate LLMs in the context of materials synthesis 

---
# Improving Randomized Metric Distortion to 2.3282 

**Authors**: Nisarg Shah  

**Link**: [PDF](https://arxiv.org/pdf/2608.29308)  

**Abstract**: In metric social choice, each voter ranks a set of $m$ candidates by her distance to them in an unknown metric space. The cost of a candidate is its average distance to the voters. A randomized voting rule must use only the rankings to choose a lottery over candidates. Its distortion is the worst-case ratio between the expected cost under the lottery it returns and the cost of the best candidate. Charikar, Ramakrishnan, Wang, and Wu [JACM 2024] prove an upper bound of $2.753$, establishing a constant separation from deterministic rules, for which the best achievable distortion is $3$. Independently, Frank [arXiv:2608.17863] and Ye [arXiv:2608.21202] improve the bound to $2.5$, using an equal mixture of maximal lottery and Integrated Veto. The existing arguments do not yield a better bound with any mixture of these rules.
We break this barrier with a new ingredient, a random-size stable lottery. Let $D$ be a random variable over the domain of positive integers. A random-size stable lottery $\mathrm{RSL}_D$ guarantees that the probability of a random voter preferring any fixed candidate $c$ to her favorite of $D$ i.i.d. draws from $\mathrm{RSL}_D$ is at most $\mathbb{E}[1/(D+1)]$, where the probability also averages over $D$. When $D=k$ deterministically, this reduces to the stable $k$-lottery of Charikar, Ramakrishnan, Tan, and Wang [EC 2025]; the case $k=1$ is precisely a maximal lottery. Their minimax argument for a fixed $k$ easily generalizes to a random $D$. Our main contribution is to show how stability with respect to a random $D$ can be used to bound distortion. By mixing a suitably chosen random-size stable lottery with Integrated Veto, we get distortion at most $11641/5000=2.3282$. The proof combines infinite-dimensional conic linear-programming duality, heuristic nonlinear optimization, and exact rational verification via the Bernstein basis. 

---
# Detecting and Repairing Hallucinations in Retrieval-Augmented Generation 

**Authors**: Sai Krishna Reddy Mulakkayala, Niki van Stein, Aske Plaat  

**Link**: [PDF](https://arxiv.org/pdf/2608.29307)  

**Abstract**: Language models increasingly answer questions by consulting retrieved documents rather than memory alone, a design now common in search assistants and enterprise knowledge tools. Grounding a model in retrieved text reduces unsupported statements but does not eliminate them, and a reader cannot tell a grounded sentence from an invented one. Most research on this problem stops at detection, yet flagging a faulty answer changes nothing for the person reading it, and little is known about which action should follow. Using RAGTruth, a benchmark whose unsupported passages are annotated by hand, we split each flagged answer into individual factual claims, check each against the retrieved source, and compare leaving the answer untouched with three repair strategies of increasing richness: deleting an unsupported claim, replacing it with source text, and rewriting it. Three language models from different families judge the 916 repaired answers. Every strategy reduces the proportion of answers judged to contain unsupported content, and all three judges agree on the ordering. Deletion achieves the largest reduction while retaining least of the original answer, at 64.3% of the text, whereas rewriting retains 80.1% and reduces least. Repair is not confined to faulty answers: 83.5% of answers annotated clean are edited too. The strategies occupy different points on a grounding preservation trade-off rather than forming a quality ranking, and choosing between them needs evidence about answer usefulness that automatic metrics cannot supply. 

---
# Learning Simple Test-Time Environments for LLM Web Agents 

**Authors**: Junxuan Li, Zijun Liu, Ziyi Huang, Peng Li, Yuzhou Liu, Ming Yan, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.29305)  

**Abstract**: Large language model (LLM) agents have demonstrated remarkable proficiency in manually constructed environments, yet their performance frequently collapses when transitioned to complex real-world settings. Existing research largely attribute this degradation to the compositional generalization gaps in LLMs on combinations of multiple simple, well-structured environments. In this work, we propose that LLM web agents can learn simple environment observations at test time. Specifically, we introduce trial steps for agents to decompose a complex environment observation into sub-modules, and implement a label-free learning method, Test-Time Environment Decomposition (TTED), to adapt agent behaviors with experience during inference. Our empirical evaluations demonstrate the framework's efficacy across both synthetic and realistic benchmarks, showing (1) experience gains acquired within simpler sub-environments can be effectively composed to improve performance in the full one, and (2) test-time training on sub-environments can significantly enhance the compositional generalization of agents in real-world web automation tasks. We also provide key insights in the design of the label-free learning algorithm. As more complex environments are accessed by LLM agents, we believe learning environment decomposition skills at test time will be critical for robust real-world deployment. 

---
# When Do Larger Batches Help Scale LLM Reinforcement Learning? 

**Authors**: Ziniu Li, Jinbo Wang, Guanhua Huang, Feiyuan Zhang, Pengbo Li, Alex Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.29296)  

**Abstract**: Larger batches reduce the variance of stochastic gradients per update and are therefore often expected to accelerate training. Yet whether this statistical benefit translates into lower wall-clock time-to-target remains unclear, because each update consumes more samples and may take longer to execute. We study this tradeoff in reinforcement learning for large language models. We separate its algorithmic and systems effects by comparing learning and execution along their natural axes. At the algorithmic level, we compare configurations at equal cumulative sample counts while retuning batch-dependent hyperparameters. Over a bounded range of batch sizes, this procedure yields an approximately batch-size-invariant family whose members follow similar sample-indexed learning trajectories. At the systems level, we exploit the computational asymmetry between rollout generation and training: autoregressive generation is often memory-bandwidth-bound at low concurrency, whereas training work scales approximately with the number of processed tokens. Combining these two views yields a direct decision rule: a larger-batch configuration reduces time-to-target only when its throughput gain exceeds its samples-to-target penalty. Experiments with GRPO and PPO support both sides of this decomposition. At the algorithmic level, square-root learning-rate scaling with Adam produces approximately batch-size-invariant learning curves over a bounded range of batch sizes. At the systems level, larger batches improve generation throughput by up to 2.29x on fixed hardware. In GRPO, combining higher throughput with learning-rate retuning reduces time-to-target by up to 29%, whereas increasing the batch without retuning is slower despite its higher throughput. 

---
# AOI-Net: Structural Face AOI-Guided Eye-Gaze Track Representation Learning for Autism Spectrum Disorder Detection 

**Authors**: Zhanpei Huang, Binbin Sun, Jialiang Chen, Yiou Wang, Taochen Chen, Yuzhu Ji, Yiqun Zhang, Yiu-Ming Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2608.29289)  

**Abstract**: Eye-movement tracking has emerged as a promising non-invasive approach to Autism Spectrum Disorder (ASD) screening, with systematic differences in attentional allocation and revisit behaviors observed during socially interactive tasks. Existing computational methods typically characterize eye-movements using discrete gaze trajectories and fixation events, yielding representations dominated by short-range temporal dynamics and limiting models that primarily emphasize long-range dependencies. Meanwhile, gaze behavior is naturally organized across semantically meaningful Areas of Interest (AOIs), whose attention allocation and transitions provide important structural cues, yet their relationships are rarely modeled explicitly. To address these limitations, we propose a structural face AOI-guided Eye-Gaze Track Network (AOI-Net) that jointly models short-term temporal dynamics and AOI-level structural organization. A network gating mechanism adaptively integrates the complementary temporal and structural representations according to their contributions to gaze-behavior characterization. To mitigate the pronounced class imbalance commonly encountered between individuals with ASD and Typically Developing (TD) participants in clinical datasets, class-distribution-aware learning is further employed to facilitate discriminative embedding learning under skewed class distributions. Experiments on a unique and large-scale clinical eye-tracking database comprising eight stimulus subsets and more than 1,300 participants show that AOI-Net consistently outperforms state-of-the-art methods. The proposed framework also enables interpretable gaze-behavior modeling and provides a practical basis for scalable AI-driven ASD screening in real-world healthcare. The code is available at this https URL 

---
# RAGDiffusion++: From Macro-Retrieval to Micro-Fidelity Alignment for Garment Generation 

**Authors**: Yuhan Li, Xianfeng Tan, Fangao Zeng, Wenxiang Shang, Pipei Huang, Hao Zhou, Zhiyu Jin, Wenjun Zhang, Bingbing Ni  

**Link**: [PDF](https://arxiv.org/pdf/2608.29280)  

**Abstract**: Standard clothing asset generation---restoring forward-facing flat-lay garment images from diverse real-world contexts---holds immense commercial value yet demands both macroscopic topological accuracy and microscopic physical fidelity. Although our previous work RAGDiffusion effectively eradicated large-scale structural hallucinations via retrieval-augmented macro-constraints, achieving industrial-grade micro-texture realism remains an unsolved bottleneck. We formally identify this limitation as High-Frequency Trajectory Collapse: supervised fine-tuning (SFT) converges to the conditional mean of the training distribution, which is dominated by smooth, low-frequency textures, causing high-frequency patterns (e.g., fabric weaves, intricate logos) to become nearly un-sampleable. Naively applying Reinforcement Learning (RL) post-training further triggers Artifact Hacking, where models exploit semantic biases in generic reward models by generating deceptive checkerboard noise. Our key insight is that RL can fundamentally reshape the sampling distribution of flow models---elevating the probability of high-fidelity trajectories under accurate reward guidance---while adversarial regularization prevents exploitation of reward blind spots. Realizing this principle requires three prerequisites: (i)inherent capacity, established through a 27,725-pair high-complexity garment dataset (STGarment-Plus) and a Dual-Image-Stream FLUX architecture upgrade; (ii)perceptive reward, provided by a novel attribute-aware reward model (Garment-RM) trained on 500K images via fine-grained contrastive learning, achieving 84.67% human preference accuracy; and (iii)hacking prevention, enforced by our Adversarial-Regularized GRPO (AR-GRPO) strategy that integrates a dynamic discriminator into the RL sampling trajectory to penalize artifacts while enriching authentic high-frequency details. 

---
# SHADOWBENCH: Toward Reliable Automatic Evaluation of Semantic Alignment in Autoformalization 

**Authors**: Hojae Han, Jongyoon Kim, Sanghyuk Park, Dongwook Cheon, Myungjae Jeon, Sunjong Choi, Soonho Kong, Wonseok Heo, Seung-won Hwang, Donghoon Hyeon  

**Link**: [PDF](https://arxiv.org/pdf/2608.29270)  

**Abstract**: Autoformalization translates informal mathematical theorems into code for proof assistants such as Lean. A central challenge is that current evaluation metrics can accept type-correct but misaligned statements or reject correct statements written in a different formulation. Inspired by Pass@$k$, we propose SA-Pass (*Semantic Alignment Pass*), which tests formal statements using auxiliary statements called *shadows* that characterize the intended statement. A generated statement receives full credit only when it compiles, implies each shadow (forward check), and is implied by their conjunction (backward check). We instantiate SA-Pass in ShadowBench, a Lean 4 full autoformalization benchmark of 178 postgraduate- to research-level problems spanning eight mathematical areas. Claude Code (Opus 4.8) with Numina-Lean-Agent reaches $61.8\%$ compile rate and $11.2\%$ SA-Pass. Across outputs generated by six agentic configurations, SA-Pass achieves $98.8\%$ binary agreement with expert judgments. An early version of ShadowBench served as the benchmark for Track 4 of the ICML 2026 AI4Math Challenge. 

---
# Measurement Validity in LLM Cultural Alignment 

**Authors**: An Duy Nguyen, Muhammad Aurangzeb Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2608.29266)  

**Abstract**: Researchers increasingly treat LLM survey responses as a proxy for human cultural values. This includes projecting model outputs onto instruments like the Inglehart-Welzel Cultural Map and drawing conclusions about which cultures a model resembles. While a model's answer to a value-laden questions may be interpreted as a cultural signal, it also carries sampling noise and, can be quite sensitive to question framing. In this paper, we separate survey responses, sampling noise and question framing for multiple LLMs. We decompose response variance from these models into variation across random seeds, prompt rewordings. We employ noise-to-signal ratio (NSR) to test whether a model's apparent cultural position is distinguishable from noise. When applied across a dozen models from four geographic origins, calibrated against 88 Integrated Values Survey countries, the answer is often no. NSR exceeds 1.0 on 49 of 117 valid model-question pairs (42%), reaching 5.56 in the worst case. Two models even refuse to answer sufficient number of survey questions outright. Our results corroborate previous findings that LLMs cluster toward Western, English-speaking cultural positions. However, what does not hold up in this study is the precision with which anyone can currently interpret a specific model's coordinates: prompt tone alone can shift a model by 2.4 map units, comparable to the distance between actual countries in the Inglehart-Welzel Cultural Map. These findings suggest that cultural attribution from LLM survey responses requires establishing the reliability of the underlying measurements before interpreting model coordinates as evidence of cultural representation. 

---
# Adaptive Multi-Branching for Shallow Decision Tree Induction 

**Authors**: Hanul Park, Jeonghoon Choi, Juseong Kim, Sanghun Sel, Giltae Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.29262)  

**Abstract**: Decision trees are attractive for tabular prediction tasks because each prediction follows an interpretable sequence of feature-threshold tests. Under a strict maximum-depth budget, however, conventional binary trees can be under-expressive, since each internal node makes only a single threshold decision. We study shallow-depth tree induction, where the goal is to improve accuracy while keeping root-to-leaf paths short. We propose the Multi-Branch Neural Decision Tree with Adaptive Pruning (MBNDT), a single axis-aligned tree trained end-to-end with differentiable multi-way splits. Each internal node learns ordered thresholds over a selected feature and a branch mask that adapts its effective arity, and the trained model is converted to a deterministic single-path tree for inference. Across 21 OpenML binary-classification benchmarks, MBNDT achieves the best average rank and mean balanced accuracy among depth-constrained single-tree baselines; a controlled ablation isolates multi-way splitting as the source of the gain. These gains come with an explicit trade-off: MBNDT realizes more leaves than the other single-tree baselines, making it best suited when accuracy under short, bounded decision paths is prioritized over minimal global tree size. 

---
# QCell: Recombining and Aligning Cell Queries for Overlapping Instance Segmentation 

**Authors**: Yaroslav Prytula, Anton Popov, Dmytro Fishman  

**Link**: [PDF](https://arxiv.org/pdf/2608.29253)  

**Abstract**: Instance segmentation of overlapping cells in microscopy remains challenging due to semi-transparent structures that produce weak boundaries and mixed visual evidence in overlap regions. Existing methods address this through local regions of interest or shape priors but lack global reasoning across overlapping objects. We present QCell, a novel query-based model that de-overlaps cell instances in microscopy scenes. Our approach combines (i) an instance recombination module that decomposes and recombines query representations in latent space, enabling the model to reason about complete object structure under overlap, and (ii) a contrastive query alignment objective that combines distinctive instance feature learning and separation of overlapping cell queries. We additionally introduce a new Organoid dataset benchmark for overlapping cell segmentation. We show that QCell outperforms state-of-the-art methods across multiple benchmarks, achieving +2.2 AP and +2.7 AJI on ISBI2014. Code is available at this https URL 

---
# AGRICAM: A Track-Mounted Crop Pollination Monitoring Robot 

**Authors**: Malika Nisal Ratnayake, Adel N. Toosi, James Cook, Romina Rader, Alan Dorin  

**Link**: [PDF](https://arxiv.org/pdf/2608.29237)  

**Abstract**: Insect pollination is critical for global food production, yet monitoring pollinators at commercial farm scale remains a challenge. Recent advances in computer vision and deep learning have enabled detailed analysis of pollinator behaviour, but monitoring must trade-off detail against spatial coverage and human or technological resources. This paper presents the Automated Guided Robot for Insect and Crop Activity Monitoring (AGRICAM), a purpose-built robotic system designed to meet the requirements of large-scale pollination monitoring in protected cropping systems. AGRICAM operates autonomously on low-cost, easily installed track for movement along crop rows, without disrupting farm operations or insect behaviour. The platform integrates two RGB cameras, microclimate sensors, GPS and RFID modules, motion sensors, and 4G cellular network connectivity for data transmission. A web interface enables remote device configuration and scheduling. The system autonomously captures video and image data of insects' locations and local environmental conditions. These are transferred to the cloud and analysed using computer vision models to quantify pollinator visitation and spatio-temporal activity variation. We deployed the system on a commercial blueberry farm to demonstrate and test its capability. It successfully mapped insect pollination patterns across 80 m long industrial polytunnels over 30 hours. This data enabled spatial analyses of insect activity we used to confirm a uniform pollinator distribution within polytunnels, as desired by the farm management team. The data also highlighted variation of insect activity associated with time of day and microclimate. AGRICAM therefore has been shown to be a scalable, automated crop pollination monitor that can support data-driven decisions to enhance pollination management, thereby improving crop productivity and food security. 

---
# Background-Free Objectness Learning for Class-Agnostic Detection 

**Authors**: Dania Batool, Liliana Lo Presti, Marco La Cascia, Filippo Vella  

**Link**: [PDF](https://arxiv.org/pdf/2608.29232)  

**Abstract**: Object detectors are typically trained under closed-set supervision, where unlabeled regions are implicitly treated as background. Under incomplete annotations, this assumption introduces objectness bias: visually valid but unlabeled objects are used as negatives, tying objectness to the annotated taxonomy rather than generic object structure. This limitation is particularly problematic for class-agnostic and open-world detection. This paper proposes Background-Free Objectness Learning (B-FOR), a dense class-agnostic detection framework that learns objectness without explicit background supervision on unlabeled regions. B-FOR formulates detection as the prediction of dense multi-scale object-center and scale fields, from which object hypotheses emerge as local spatial structures. Supervision is confined to reliable annotated regions through spatially structured soft targets, avoiding foreground-background discrimination. To support decoding from emergent local maxima, the paper further introduces displacement-aware scale fields that model object extent as a spatially varying property of the learned objectness field. Experiments on PASCAL VOC, MS-COCO, and Open Images demonstrate strong generalization to unseen categories and cross-dataset object distributions. B-FOR improves recall by more than +10 AR points over prior class-agnostic baselines. Ablation studies show that both localized objectness supervision and displacement-aware scale fields are critical for class-agnostic localization under incomplete annotations. Code available at: this https URL. 

---
# AgentLogs: A Dataset for Opening the Black Box of GitHub's Cloud Agent 

**Authors**: Jonan Richards, Kosei Horikawa, Youmei Fan, Yutaro Kashiwa, Mairieli Wessel  

**Link**: [PDF](https://arxiv.org/pdf/2608.29204)  

**Abstract**: Generative AI-based software engineering agents are becoming routine contributors to real-world software projects. On GitHub, developers can assign tasks to the Copilot cloud agent, which autonomously explores the repository, edits code, runs commands, and opens or reviews pull requests, producing a detailed log of every step along the way. While existing datasets capture outcomes of agent contributions, such as agent-authored pull requests, the process by which agents produce these contributions remains largely unexplored. To address this gap, we introduce AgentLogs, a large-scale dataset of agent activity on GitHub. AgentLogs comprises 307,416 agent tasks and 549,239 agent sessions in 35,810 of the 1,812,362 popular public repositories that we scanned, together with 64,255,174 session log entries that record each agent run step by step, including prompts, intermediate reasoning, tool calls (e.g., file edits, git operations, and GitHub interactions), and token usage. By exposing not only what agents contribute but also how they work, AgentLogs enables research on agent behavior, efficiency and cost, task formulation, failure modes, and human-agent collaboration in agentic software engineering. 

---
# PokaiTrainer: Scaling Belief-State Search to Competitive Pokémon VGC 

**Authors**: Max Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.29197)  

**Abstract**: Decision-time equilibrium search carried poker to superhuman play, but it has so far relied on tractable subgames: a handful of actions per decision, chance confined to card deals, one player moving at a time. Competitive Pokémon in its official doubles format (VGC) breaks all three assumptions at once. Both players act simultaneously from joint menus in the hundreds, each joint action resolves to hundreds of stochastic outcomes, and the opponent's reserves and stat allocations are hidden. We set out to build a strong VGC agent and report what that took. PokaiEngine, our Rust battle engine, enumerates a joint action's full weighted outcome distribution in one pass, at ${\sim}99\%$ parity with Pokémon Showdown and a fraction of the cost of sampling it. On top of the engine, PokaiTrainer adapts Student of Games to this scale, solving every decision as a Bayesian matrix game over public belief states and growing subgames under an explicit compute budget. On the live Showdown best-of-three ladder, the agent wins 59% of 150 sets against a human field averaging ${\sim}1320$ Elo. It settles into a 1350-1400 Elo band, and at its peak briefly entered the format's top 500. 

---
# Rate-Coding Bundle Memory: A Unified Model of Memory and Control for Symbolic Computation in the Brain 

**Authors**: Teun van Gils, Rowan P. Sommers, Markus Ostarek, Peter Hagoort  

**Link**: [PDF](https://arxiv.org/pdf/2608.29189)  

**Abstract**: We propose a neurobiologically plausible model of cognition that combines the advantages of connectionist and symbolic systems, and that can explain a wide range of cognitive phenomena. This model, called Rate-Coding Bundle Memory (RCBM), is based on the Symbolic Subsystem Hypothesis, which posits that the brain implements a symbolic subsystem within its fundamentally connectionist nature. RCBM is a hybrid model that uses rate coding to represent symbols in a continuous space, and it uses a bundle memory system to store and retrieve these symbols. The model is capable of solving a wide range of cognitive phenomena, including one-shot learning, pattern separation, and the binding problem. We argue that RCBM provides a promising framework for understanding the nature of cognition, and that it can be used to develop more sophisticated models of cognition in the future. 

---
# Locked at the Entrance, Open Inside: Where RLVR Narrows the Solution Space 

**Authors**: Qiancheng Zhou, Ruizhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.29188)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) substantially improves single-sample accuracy (pass@1) but causes the policy's solution space to contract, diminishing the returns of test-time scaling. In this work, we investigate where inside a reasoning trajectory this breadth is lost: does the policy fail to access a valid solution family, or does it fail to execute computation once initiated? To disentangle access from execution, we analyze the Countdown task, whose solution space can be exhaustively enumerated into discrete entrance families defined by the first operand and operator, across PPO on Qwen2.5-3B and GRPO on Qwen2.5-3B-Instruct. Across both training setups, solution coverage falls by up to 67%, halving even on problems solved across all checkpoints. We show that this contraction is heavily concentrated at the entrance: per-token likelihood shifts are 11x--16x larger prior to the first arithmetic operation than during downstream reasoning. Supplying only an unselected entrance prefix restores completion rates in low-access families by over an order of magnitude (0.018 -> 0.212 under PPO), demonstrating that alternative solutions remain executable but are no longer initiated. Guided by this localization, we find that while surface prompting fails to recover diversity, entrance-targeted interventions succeed: late-layer parameter interpolation with early checkpoints increases solution coverage by 37% at no loss in pass@1. Finally, we show that early-step entropy collapse recurs across six math benchmarks with 7B and 14B models, but is not an inevitable byproduct of reasoning optimization: an SFT baseline preserves more than double the coverage, and staged SFT--DPO--RLVR pipelines retain early-step entropy. In summary, reasoning breadth is lost at the door, not inside the room. Code: this https URL. 

---
# TAAL: Mitigating Early Beam Pruning in Generative Recommendation via Temporal Autoregressive Alignment 

**Authors**: Lianjie Li, Zhiying Tu, Dianhui Chu, Hongliang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.29179)  

**Abstract**: Generative recommendation encodes items as hierarchical semantic identifiers (SIDs) and retrieves the next item through autoregressive decoding. Standard next-token prediction, however, does not explicitly cover the multimodal transitions present in interaction sequences, leaving the ground-truth SID vulnerable to irreversible pruning at early beam-search branches. Across three public benchmarks, we find that 91.9\%--96.6\% of retrieval failures occur within the first two decoding steps. We therefore propose Temporal Autoregressive Alignment (TAAL). During training, TAAL constructs a joint $(c_1,c_2)$ soft target from historical transitions and aligns the early-prefix distribution with a forward KL objective. During inference, it calibrates candidate scores with pointwise mutual information (PMI) to reduce the influence of globally frequent prefixes. On Amazon Beauty, Instruments, and Yelp, TAAL improves NDCG@10 over the standard baseline by 39.5\%, 6.7\%, and 28.6\%, respectively, while increasing full-SID survival by 3.9\%--16.6\%. Beam-width analysis further shows that the relative survival gain grows as the beam narrows, reaching 39.4\% at $B=5$. 

---
# Subtraction-Based Tumor Segmentation and Lesion-Centered pCR Prediction for the MAMA-MIA Challenge 

**Authors**: Kai Geissler, Raphael Schäfer  

**Link**: [PDF](https://arxiv.org/pdf/2608.29162)  

**Abstract**: We describe the submission of team FME to the MAMA-MIA Challenge, which evaluated primary tumor segmentation and prediction of pathological complete response (pCR) from pretreatment dynamic contrast-enhanced breast MRI on an external multi-country cohort. For segmentation, we trained a five-fold residual-encoder nnU-Net ensemble using only the first post-contrast minus pre-contrast image, combined with mirroring test-time augmentation and largest-connected-component filtering. For pCR prediction, we ensembled 25 pretrained 3D video classifiers trained on lesion-centred crops from the pre-contrast and first two post-contrast volumes. FME ranked second in both tasks. The segmentation method achieved a combined performance-fairness score of 0.882, with Dice 0.713 and normalized Hausdorff distance 0.099. The pCR method achieved a combined score of 0.664, balanced accuracy of 0.541, and equalized-odds disparity of 0.212. The results indicate that subtraction-based input and ensembling support robust tumor segmentation under cross-site domain shift, whereas pCR prediction from baseline DCE-MRI alone remains limited.
For the submission repository, see this https URL 

---
# Training-Free Hidden-State Refinement for Flow-Matching Image Generators 

**Authors**: Yuanyi Yan, Xinzhe Rao, Canyu Shen, Yang Chen, Yunlu Chen, Meng Tang, Teng Long, Vincent Tao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.29160)  

**Abstract**: We aim to improve frozen flow-matching image generators by adding inference computation inside the denoiser, without changing model weights or the outer sampler. Existing generators usually spend extra test-time computation by increasing the number of sampling steps, which repeatedly evaluates the entire denoiser and couples quality gains to sampler cost. A key challenge is how to use extra computation inside a frozen transformer denoiser: the method must decide which tokens, layers, and sampling times receive repeated updates while preserving the original generation pipeline. We introduce a training-free looping framework that repeatedly applies selected transformer layers inside each denoising call. Dense and Sparse Token Loop vary the token scope; Sampling-Progress Gating and the loop layer range specify when and where looping is active; loop count and strength control the repeated updates; and Loop Guidance combines ordinary and looped vector-field predictions. Across two Scale-RAE model scales, loop variants improve primary and auxiliary quality metrics with competitive quality--efficiency trade-offs. Loop Guidance further improves both primary metrics across all three tested models; on Scale-RAE DiT2.4B, it raises GenEval from 0.4471 to 0.5691 and DPG-Bench from 0.7656 to 0.8053. Code will be released. 

---
# STARLINC: Satellite Trail Artifact Removal using Inter-Frame Correlation 

**Authors**: Shingeon Kim, Hyeyoon Lee, Dain Kwon, Kanghyun Choi, Sunjong Park, Mi-Ryang Kim, Jeong-Eun Lee, Jinho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.29145)  

**Abstract**: The rapid expansion of low Earth orbit satellites such as Starlink is increasingly contaminating astronomical surveys. In practice, contaminated images are often identified through inspection. However, modern surveys generate terabytes of data each night, making manual screening infeasible and necessitating reliable automated methods for satellite trail removal. Unfortunately, existing general-domain line detection methods fail to generalize to astronomical images due to domain mismatch, which are mostly grayscale with sparse bright stars and have a low signal-to-noise ratio. Moreover, training new models from scratch is impractical due to the lack of large-scale annotated astronomical datasets. To address these challenges, we introduce STARLINC, the first ML-based framework for satellite trail removal without requiring tedious pixel-level annotation of astronomical images. STARLINC combines synthetic satellite trail generation for training, inter-frame differential maps from temporally adjacent exposures to highlight transient trails, and heatmaps to provide additional localization cues for pixel-level segmentation. Extensive experiments on real-world data demonstrate substantial improvements over baselines, establishing STARLINC as a scalable solution for next-generation astronomical surveys. Code is available at this https URL. 

---
# Not the Same Protector: Deployment-Dependent Protective Intervention in LLMs 

**Authors**: Eunna Lee, Soomyoung Lee, Jungpyo Nam, Heonjin Ha, Jamin Jung, Kyunam Choi, Sunjun Hwang, Yeonghun Kim, Seok-Jae Lim  

**Link**: [PDF](https://arxiv.org/pdf/2608.29136)  

**Abstract**: We ask whether a model protects a user in the same way when that user speaks rather than types. Using a single distress vignette---a physical injury of unstated severity following an interpersonal conflict---we present four frontier models with matched inputs across voice, text, and raw API deployment conditions (n=30 per cell) and code each response along five binary protective indicators, including whether the model issues an explicit medical-care directive. Voice-interface responses are markedly shorter than text-interface responses for three of the four models, and protective behavior contracts alongside that compression: medical directives are at ceiling under both the API and text conditions but decline under voice for every model tested. The contraction is not reducible to length. One model produces voice and text responses of comparable length yet still drops medical directives, and another falls below ceiling between its API and voice conditions, whose responses are of nearly identical length. Under raw API access the pattern is categorical rather than partial: no model asks after the user's safety even once. These results show that protective intervention is sensitive to the surface through which a request arrives, that this sensitivity is detectable using a simple protective coding scheme, and that it is not explained by turn length alone. 

---
# HEAR Who Said What: Unlocking Speaker-Attributed Reasoning via Counterfactual Voice Grounding 

**Authors**: Dongwook Lee, Sangkwon Park, Eunwoo Song, Che Hyun Lee, Youngho Cho, Junho Kim, June Young Yi, Heeseung Kim, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2608.29120)  

**Abstract**: Speech Language Models (SLMs) are increasingly deployed in multi-speaker environments, yet their ability to attribute speech to the correct speaker and reason over speaker identities remains unclear. Hence, we introduce HEAR, a conceptually hierarchical benchmark diagnosing the foundational capabilities of speaker-attributed reasoning, comprising 2.4K human-verified samples from 887 diverse multi-party audio clips. Evaluating 20 leading SLMs on HEAR reveals they struggle with these foundational tasks, often relying on semantic priors rather than actual vocal cues. To address this, we present A2R, a 30B model optimized on Counterfactual Audio with Speaker-level Hard negatives (CASH), a dataset designed to guide the model to prioritize acoustic vocal cues over linguistic signals. A2R achieves strong performance on HEAR and exhibits zero-shot generalization to diverse multi-speaker downstream tasks, demonstrating that learned speaker attribution unlocks the model's latent capacity for speaker-aware reasoning. All resources are available at this https URL 

---
# CGFM-Nav: Cognitive Graph-Field Memory for Semantic-Guided Lifelong Multimodal Embodied Navigation 

**Authors**: Yuxiang Xiao, Xibei Chen, Xin Zhou, Jie Chen, Yifeng Zhang, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2608.29114)  

**Abstract**: Vision-and-Language Navigation (VLN) requires agents to reason over accumulated observations while continuously exploring unseen regions. However, existing environment representations often struggle to jointly support explicit semantic memory and continuous exploration guidance. To address this challenge, we propose Cognitive Graph-Field Memory (CGFM), a persistent multimodal scene representation that couples explicit relational memory with continuous spatial intuition. CGFM organizes objects, spatial relations, and visual observations into a multimodal scene graph, enabling target retrieval and long-horizon reasoning across navigation tasks. When no reliable target match is identified, graph-based evidence is projected into a goal-conditioned semantic-frontier field to guide exploration toward semantically promising frontiers and regions. Building upon CGFM, we introduce CGFM-Nav, a foundation-model-based framework for lifelong multimodal navigation that integrates task-relevant subgraph selection, VLM reasoning, and verification feedback into a closed decision loop. Preliminary experiments on GOAT-Bench show that, under the same Qwen3-VL-8B backbone, CGFM-Nav improves the overall success rate from 53.2% to 63.0% and SPL from 30.0% to 39.6%, demonstrating the effectiveness of combining explicit semantic memory with semantic-guided exploration. 

---
# Auditing and Mitigating Privacy Leakage in Cloud-Edge Collaborative Decoding 

**Authors**: Kejia Zhang, Tianyuan Zou, Zixuan GU, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.29111)  

**Abstract**: Applications such as personalized assistance and proprietary document analysis require large language models (LLMs) to generate outputs from private data. Yet powerful LLMs typically cannot be deployed on the resource-constrained devices where private data resides, and uploading private data to cloud-hosted LLMs exposes sensitive information. Recent work addresses this tension with a cloud-edge collaborative decoding paradigm, where private data are kept on the edge with a small language model (SLM) producing next-token distributions, which are fused with predictions from a cloud LLM operating solely on public data. In this paper, we systematically analyze the privacy risks of such a paradigm with a novel evaluation framework using constructed QA datasets, which show that such collaboration can expose substantial private-context information. To address such privacy leakage, we propose CoVeil, a defense mechanism which dynamically optimizes transmitted signals to suppress leakage during decoding time while preserving the collaborative quality. Extensive evaluations demonstrate that CoVeil consistently improves the privacy-utility trade-off over existing baselines by reducing data leakage by up to 87.2%, with minimal accuracy loss. 

---
# Development of an Autonomous AI Coding Agent using Monte Carlo Tree Search (MCTS) and Gemini LLM Frameworks 

**Authors**: Pravin Game, Vipin Ramakrishnan, Prathamesh Wagh  

**Link**: [PDF](https://arxiv.org/pdf/2608.29096)  

**Abstract**: The ongoing changes in software engineering requirements have created a substantial need for automated tools which can create secure source code from natural language input. The performance of traditional Large Language Models (LLMs) becomes limited by their "one-shot" capability which results in logical hallucinations together with reduced algorithmic performance during complicated operations. The research presents an autonomous AI Coding Agent which establishes a connection between LLM-generated content and production-ready software through its organized methodology for decision making. Our framework uses the Gemini 2.5 Flash API for essential reasoning capabilities while employing a tailored Monte Carlo Tree Search (MCTS) method to solve code generation challenges as a search operation. The agent uses a "Self-Critic" evaluator system to test different implementation methods which it ranks according to their accuracy and difficulty level before it improves its operational framework through backpropagation. The system operates through a Flask-based web interface which delivers instant feedback together with syntax highlighting features. Our experimental results show that the MCTS-based method achieves a 92% success rate on complex logical prompts while surpassing standard zero-shot generation models. 

---
# Not All or None: Dynamic Construction of Target-aware Memory Graph for Conversational Stance Detection 

**Authors**: Yifan Xiang, Bin Liang, Yuqi Huang, Ruifeng Xu, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2608.29066)  

**Abstract**: Stance detection is crucial for understanding the underlying attitude of an expression towards a target. Conversational stance detection is a more challenging stance detection task in real-world social media scenarios, as it involves detecting the user's stance by leveraging the target-related historical statements across conversational sessions. In this paper, we propose target-aware Memory Graph TamGraph, a novel method that dynamically leverages target-related statements for conversational stance detection. Instead of considering all preceding historical conversations or using no prior conversation information for stance detection, our TamGraph employs a stepwise, entropy-guided backtracking mechanism to selectively activate memory from historical conversations and dynamically constructs a target-aware graph to model the stance relations among utterances. This allows the exploitation of target-related information from the conversation history for stance detection while preventing the introduction of noise. Experimental results on both English and Chinese benchmarks demonstrate that our TamGraph substantially improves LLM performance on conversational stance detection. 

---
# DocIntent: Answerability-Guided Agentic Restoration for Real-World Document Visual Question Answering 

**Authors**: Zihan Huang, Shihang Wu, Junle Liu, Peirong Zhang, Yongxin Shi, Xuhan Zheng, Lianwen Jin  

**Link**: [PDF](https://arxiv.org/pdf/2608.29037)  

**Abstract**: Real-world degradations such as blur, shadow, distortion, and moire patterns severely impair the document question-answering capabilities of Multimodal Large Language Models (MLLMs). Applying restoration tools before Visual Question Answering (VQA) is an intuitive solution. However, existing restoration approaches remain limited, as manually designing and executing restoration strategies is labor-intensive and requires domain expertise. Agentic restoration offers new possibilities for automation, yet existing frameworks primarily target natural images and pursue perceptual quality, overlooking that restoration should serve downstream tasks rather than optimize generic image quality metrics. To this end, we explore the value of agentic restoration for real-world degraded document VQA and propose DocIntent, a training-free Answerability-Guided Agentic Restoration framework. DocIntent first assesses question answerability, then identifies task-relevant degradations and selectively invokes restoration tools. A Comparison-Based Rollback mechanism validates each restoration step and reverts it when question-relevant evidence becomes less decipherable. The entire process requires no additional pretrained degradation classifier or image quality assessment model. Extensive experiments on the WildDoc benchmark show that DocIntent consistently improves the average score and consistency of different open- and closed-source MLLMs. The code and experimental data will be publicly available. 

---
# A Unifying Perspective on Language Model Representations: From Filler-Role Structure to Mechanistic Interpretability 

**Authors**: Zhang Enyan, R. Thomas McCoy  

**Link**: [PDF](https://arxiv.org/pdf/2608.29034)  

**Abstract**: A wide range of methods have been proposed for interpreting language models, delivering important insights into their inner workings. However, different methods and their resulting insights stand in relative isolation: what could the underlying structure of language models be, such that they give rise to all our interpretations? In this work, we propose using Tensor Product Representations (TPRs) as a unifying hypothesis. TPRs give a concrete proposal for how compositional structure could be represented in vector space --- as filler-role bindings. We show, both mathematically and empirically, that TPRs can unify several prior interpretability methods: additive analogies, linear probing, sparse autoencoders, and activation patching. Mathematically, we show that these methods can all be derived from TPRs. Empirically, we apply the derivations to a range of different models --- from small toy models to LLMs --- to construct instances of each of the above interpretability methods; these constructed variants perform comparably to their standard variants. We view this work as a step toward what interpretability will ideally provide: a unified account of the nature of neural networks, corroborated not just by individual observations but also by an explanation of the connections between them. 

---
# Flow-JEPA: Flow Matching for Robust Latent Dynamics in JEPA World Models 

**Authors**: Yanchen Huo, Ziying Song, Yadan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2608.29029)  

**Abstract**: Joint-Embedding Predictive Architectures (JEPAs) have shown strong potential for learning compact predictive representations, and LeWorldModel (LeWM) extends this paradigm to reconstruction-free latent world modeling from pixels. However, its deterministic autoregressive predictor generates future states through repeated one-step transitions, which can accumulate errors and remain sensitive to task-irrelevant visual perturbations. In this work, we propose Flow-JEPA (F-JEPA), a conditional flow matching dynamics model that jointly generates a sequence of future latent states conditioned on the current observation and actions. A Gaussian distribution serves as the flow source, exposing the vector field to perturbed latent trajectories as it learns to transport them toward clean future representations. This formulation retains the reconstruction-free JEPA framework while replacing point-wise transition regression with stochastic trajectory-level prediction. F-JEPA raises mean success from $86\%$ to $92\%$ under clean observations and from $67\%$ to $86\%$ under noisy conditions, suggesting that conditional flow matching provides a promising alternative to deterministic autoregressive dynamics in JEPA world models. 

---
# RoSe-SLAM: Robust Semantic-Aware Gaussian Splatting SLAM from Dynamic Monocular Videos 

**Authors**: Wenting Wang, Jiaxin Guo, Wenzhen Dong, Yun-Hui Liu, Charlie C.L. Wang, Yeung Yam  

**Link**: [PDF](https://arxiv.org/pdf/2608.29003)  

**Abstract**: In dynamic and unstructured environments, conventional SLAM systems generally suffer from significant accuracy degeneration due to their static assumptions. In this work, we propose Robust Semantic-aware Gaussian Splatting SLAM (RoSe-SLAM), to address the dynamic challenge by a holistic semantic scene understanding from uncalibrated monocular inputs, achieving accurate camera tracking and high-quality geometry reconstruction. Unlike conventional semantic SLAM using handcrafted semantic labels, our RoSe-SLAM exploits the semantic feature from 2D foundation model to enhance the dynamic tracking and mapping performance. By distilling the rich semantic features to our Gaussian fields, our method effectively identifies dynamic distractors and achieves semantic-aware multi-view consistency, significantly enhancing the geometric reconstruction and scene inpainting. Specifically, we propose a spatial-temporal motion mask generation module, enabling both long-term motion monitoring and short-term transient dynamics capturing, achieving robust and effective disentanglement of dynamic objects and static backgrounds. During global bundle adjustment, we propose an occlusion-aware keyframe selection mechanism to prioritize the occlusion as metric to pick the keyframes, and a multi-view semantic consistency module to improve the mapping quality in dynamic environments. By combining geometric motion cues with semantic priors, our system dynamically filters unreliable observations and reconstructs accurate static scene geometry. Extensive experiments conducted on benchmark datasets including dynamic TUM, Bonn and Wild-Mocap datasets, demonstrate that our method achieves superior performance in both trajectory estimation and static scene mapping, outperforming existing dynamic RGB SLAM baselines in long-term dynamic indoor environments. 

---
# The Illusion of Replacement: Rethinking Specialized Machine Learning Models in the Foundation Model Era 

**Authors**: Kiyan Rezaee  

**Link**: [PDF](https://arxiv.org/pdf/2608.28980)  

**Abstract**: Can the specialized architectures that machine learning has traditionally built for structured data be replaced by language-based models? This question is examined through a review of 159 papers (2016--2026) across nine modalities, with predictive accuracy considered alongside structural representation and computation. A distinction is made between performing a task and preserving and computing the structure that makes the task tractable, and existing approaches are organized into eight representational regimes, ranging from language-only systems to fully specialized architectures. Language-mediated models are found to be highly competitive in specific settings, including extreme few-shot prediction, discretized symbolic tasks, textually annotated knowledge graphs, and large-scale single-modality pretraining. However, whenever structural representation or computation is directly evaluated rather than accuracy alone, no evidence of general architectural replacement is found. Instead, a recurring pattern is observed across independent research communities: when language alone is insufficient, the missing structure is reintroduced through a graph module, structural tokens, specialized attention, or another non-linguistic component. In this sense, specialization more often relocates than disappears. Moreover, although performance of language-based models is improved by scaling, whether the gap to a structure-aware architecture can eventually be eliminated remains untested. 

---
# Free Speech and Artificial Intelligence 

**Authors**: Etienne Brown  

**Link**: [PDF](https://arxiv.org/pdf/2608.28973)  

**Abstract**: Philosophers and legal scholars are engaged in debates about the implications of artificial intelligence for freedom of expression. This paper analyzes the free speech issues raised by two distinct AI technologies: social media recommendation algorithms and conversational AI (i.e., chatbots powered by large language models). The first part shows that, through their recommendation algorithms, social media platforms control the dynamics of speech visibility in the digital public sphere, making algorithmic recommendation relevant to the philosophy of free speech. The second part turns to conversational AI. It discusses both the reasons for granting or withholding speech rights to artificial agents and users' right to receive information, which may render specific forms of chatbot regulation illegitimate. Throughout, the chapter also considers whether social media platforms or AI developers hold corporate speech rights. Its general aim is to raise rather than settle questions that arise from the rapid development of AI technologies. 

---
# Diagnose, Then Refine: A Closed-Loop TTS System with AudioLLM-Guided Correction 

**Authors**: Zeyang Song, Tianchi Liu, Tianrui Wang, Chenglin Xu, Steven Y. Guo, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.28970)  

**Abstract**: Current TTS systems typically rely on open-loop, single-pass generation and can produce sporadic local prosodic defects, such as misplaced stress, unnatural pauses, or flattened intonation, that utterance-level metrics often fail to expose. We present LoopTTS, a judge-guided Filter-Judge-Refiner framework for recovering low-quality TTS outputs diagnosed by an AudioLLM. Given an initial utterance from a base TTS model, an AudioLLM Judge identifies salient prosodic issues and generates structured refine instructions; a Refiner, our fine-grained instruction-following TTS model, then performs guided expressive re-synthesis conditioned on the initial utterance, target text, and instruction. To train the Refiner, we construct Refiner-DB, a 42K-example AudioLLM-annotated dataset with word-level prosodic weak supervision. Human evaluation on diagnosed low-quality utterances shows that LoopTTS can detect perceptually salient errors and correct them with the Refiner, outperforming raw generated audio and practical open-loop re-generation baselines in recovery quality. The Refiner also demonstrates stronger instruction-following ability for stress and pause control in targeted prosody modification. 

---
# The information geometry of product-reference discrete diffusion: Interaction growth complexity and optimal scheduling 

**Authors**: Martin J. Wainwright  

**Link**: [PDF](https://arxiv.org/pdf/2608.28949)  

**Abstract**: We study a class of product-reference diffusion algorithms for sampling from a discrete distribution. We show that their sampling performance can be characterized using a path-based measure of data geometry that we call the interaction growth complexity (IGC). We show that a bivariate IGC kernel gives an exact representation of both the KL discretization error and a simple one-step upper bound. The simpler univariate IGC density can be used to study the effect of stepsize choices on the iteration complexity required to obtain $\epsilon$-accurate samples in KL divergence. Samplers that traverse the path with equi-spaced steps in log-squared-reliability-odds have performance that depends on the aggregate IGC mass, whereas refined choices of stepsizes have a lower complexity depending on a square-root functional. In the fine-grid limit, both of these characterizations become sharp. We also allow general product reference distributions and show that the reference law can substantially reshape the IGC profile and the resulting sampling complexity; in particular, references far from both the uniform and the data marginals can yield dimension-dependent improvements. Finally, the aggregate IGC mass admits bounds in terms of total correlation and dual total correlation, thereby connecting the pathwise geometry to classical measures of multivariate dependence. 

---
# From the Loss Landscape to Diverse Feature Learning in Neural Networks 

**Authors**: David Aram Yunis  

**Link**: [PDF](https://arxiv.org/pdf/2608.28948)  

**Abstract**: Over the course of the last decade, neural networks have grown from an academic curiosity to moving the markets of nations. Despite this explosion in both research and deployment, relatively little is understood about how they achieve the solutions they do. This is both scientifically relevant, and pressing for society. When neural networks make decisions across self-driving, construction, law, hiring and health, there have been and will continue to be unintended consequences.
However, attempting to generalize the failures of the largest and most important production systems makes for a very difficult task. Yet signs of these failures exist at all scales of neural networks, so we should be able to study a much more tractable setting. All neural networks must undergo an optimization process, called training, to be useful. To a great degree, understanding neural networks is understanding their optimization: through what process and exposure to which data did they arrive at their results. Yet our knowledge on this topic as a field is quite imprecise. In particular, a curious phenomenon called mode connectivity, the ability to connect neural networks in the loss surface, defies explanation entirely.
This dissertation elucidates, explains and exploits this special structure in the loss landscape... 

---
# The Hallucination Signal Is a Mean Shift: Why Simple Probes Suffice 

**Authors**: Jungseob Lee, Jaehyung Seo, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2608.28930)  

**Abstract**: Hidden-state probes effectively detect LLM hallucinations, but the geometry of the signal remains poorly characterized, driving increasingly complex probe architectures. Across three 7B-scale models and three datasets in a paired-example paradigm, we find the signal overwhelmingly dominated by a single mean-shift component, and removing this direction collapses detection to chance. Shrinkage linear discriminant analysis closes about 73% of the gap between 1D and full-dimensional classifiers, so apparent architectural complexity largely reflects high-dimensional covariance estimation difficulty rather than exploitable non-linearity. A simple L2-regularized logistic regression (0.952 AUROC) bounds or outperforms twelve controlled architectural alternatives, and our multi-layer aggregation exceeds CLAP cross-layer attention probing under matched paradigm. Because the signal spans a contiguous layer band, LayerMix aggregates it to match oracle-layer performance without oracle access. Our claims characterize the geometry within the controlled paired-example paradigm. Our code is available at this https URL. 

---
# ActiveAugment: Online Active Learning for Augmentation Selection in Deep Learning 

**Authors**: Noah Videcrantz, Mostafa Mehdipour Ghazi  

**Link**: [PDF](https://arxiv.org/pdf/2608.28923)  

**Abstract**: Data augmentation is a cornerstone of deep learning pipelines, yet existing strategies treat it as a static, model-agnostic preprocessing step, either relying on expensive dataset-specific policy search or applying transformations uniformly at random, regardless of what the model has already learned. We introduce ActiveAugment, a unified framework that treats augmentation selection as an online active learning problem. For each training minibatch, ActiveAugment generates a pool of candidate augmented views and scores each candidate using a combination of the model's predictive uncertainty and the feature discrepancy induced by the augmentation. The augmentation under which the current model is most fragile is selected per sample, and the model is then trained with a joint supervised classification and supervised contrastive objective that enforces intra-class invariance to the selected augmentations while maintaining inter-class separation. We evaluate ActiveAugment on eight benchmark datasets spanning natural and medical imaging, using CNN and transformer architectures across three training regimes (training from scratch, full fine-tuning, and linear probing), and comparing eight active selection strategies for augmentation scoring. ActiveAugment outperforms AutoAugment, RandAugment, and TrivialAugment under controlled augmentation shifts across all domains and budgets, with the most pronounced gains at low labelling budgets. On medical imaging datasets, where data is scarce and domain shift relative to natural-image pretrained models is large, ActiveAugment achieves higher test F1 than all baselines, demonstrating strong cross-domain adaptability. Our analysis reveals that the augmentation selection policy evolves meaningfully during training and that strategy choice has a direct impact on generalisation. Code is available at: this https URL. 

---
# Structured State Reconciliation for Human-AI Task Handover 

**Authors**: Kayleigh Bishop, Maria P. Stull, Breanne Crockett, Bradley Hayes  

**Link**: [PDF](https://arxiv.org/pdf/2608.28907)  

**Abstract**: Task handover requires communicating enough current state for a successor to resume work, yet the relevant information is often divided between system records and human observations. System records can be precise and timestamped but only partially observe the task, while human reports capture intent and task knowledge that no log contains but are vulnerable to omission and memory error. We present a provenance-aware pipeline that converts task telemetry and human-authored reports into a shared typed task-state representation, aligns and reconciles their facts, detects conflicts, and generates structured handover reports. We evaluate the approach on 13 paired task states collected in a controlled spatial multitask environment, using task-grounded metrics that estimate the state-reconstruction cost a report would spare a hypothetical recipient and the misinformation burden it would impose. Reconciling both sources preserved greater estimated task-state utility than either the user report or telemetry alone. Relative to a direct end-to-end LLM given the same inputs, structured reconciliation maintained comparable estimated utility while incurring substantially less misinformation, and task-aware rendering retained utility more efficiently (per token) than exhaustive rendering. An exploratory content analysis further shows that human reports contain substantial strategic knowledge that lies outside state-focused metrics. These results support provenance-aware state reconciliation as a design pattern for safer AI-assisted handover. 

---
# Moving the Mean Toward the Known Good, Not Beyond It: What Inference-Time Interventions and Weight Consolidation Buy in Open-Ended Generation 

**Authors**: Roberto I. Ono Filho  

**Link**: [PDF](https://arxiv.org/pdf/2608.28886)  

**Abstract**: What does a generation loop gain from learning on its own verified successes? In cycles of generate, verify, select and LoRA-consolidate on online bin packing, training on value-filtered candidates shifts what the model writes on held-out variants toward value (-1.7 points of excess, p=0.008; -3.1 against a random-consolidation control, p=0.004) while the best observed candidate converges to the classic heuristic's level and no further. A confirmation battery replicates the whole procedure three times, with fresh seeds and a never-consulted held-out set read exactly once: the mean was nearly identical in all three lineages (-2.0, -1.8, -1.9), and after aggregating within held-out variant all seven evaluable variants favored consolidation (p=0.008). The best observed candidate moved to the classic heuristic's level, exactly (0.021028 in all three lineages, for attract and for the random control alike), and never beyond it. A matched SFT-only control shows the supervised anchor, not repulsion from bad candidates, does the concentrating (96% of candidates land exactly at the classic heuristic's level). The tails cut both ways: consolidation lowers the per-candidate rate of better-than-classic candidates (10% to 3.9%) while its larger production yields more such candidates absolutely (5 against 1, on few events). As motivation we report the inference-time ledger that led here: a model-written schematic recap buys judged document integration and nothing buys development; a verifier written into the stream is imitated, 16.4 fabricated verdict lines per notebook. Mean quality among valid candidates can be bought and replicated; the observed best goes to the classic and, so far, never beyond it. 

---
# Hybrid Offline-Online Multi-Agent Decision Transformers for Wireless Resource Management 

**Authors**: Yiming Zhang, Kun Yang, Cong Shen, Dongning Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.28878)  

**Abstract**: This paper develops a hybrid offline-online multi-agent reinforcement learning framework based on decision transformers. The policy is first pretrained offline via supervised sequence modeling of trajectories generated by existing policies, providing a safe and sample-efficient initialization. It is then fine-tuned online using a hybrid objective that incorporates critic-guided gradients, enabling performance improvements beyond the offline policy. To facilitate stable offline-to-online transfer and effective multi-agent coordination, the framework incorporates return-weighted sampling, a critic conditioned on neighbors' actions, and neighborhood-correlated exploration. The approach is fully distributed: both training and execution rely only on local observations and limited information exchange among neighboring agents. Evaluations with dynamic traffic arrivals in two settings: (i) joint scheduling and power allocation and (ii) coordinated beamforming, show that the proposed method achieves quality-of-service (QoS) performance comparable to centralized methods. Moreover, when pretrained on lower-quality datasets, online fine-tuning is also observed to surpass the initial offline policy. These results demonstrate a promising learning-based alternative for wireless resource management. 

---
# No Detectable Change in Side-Level WER from Prompt-Level Context: A Preregistered Ablation on a Production Oral-History Corpus 

**Authors**: Theodore O. Cochran, Stephanie Dodson, Keith Nore  

**Link**: [PDF](https://arxiv.org/pdf/2608.28875)  

**Abstract**: Supplying context at inference time to a large multimodal model is an inexpensive lever for adapting speech transcription to a domain, and earlier results on smaller models reported large gains. This work tested that mechanism where it ships, in the prompt-conditioning layer of a production oral-history transcription tool, on a sample from its own production corpus. Full prompt-level context did not detectably change side-level word error rate (WER), and none of the four preregistered hypotheses was supported. The design was a within-item paired ablation, preregistered with the analysis code frozen by hash before the confirmatory batch was scored; two disclosed gpt-4o pilot sides had been scored earlier, during scorer development. Nineteen cassette sides, about 10.6 hours of degraded 1970s-80s interview audio, were reprocessed through the production code path under three prompt arms, crossed with two deployed commercial configurations, gpt-4o-transcribe and gemini-2.5-flash, and scored against operator-corrected verbatim references. For gpt-4o-transcribe the median paired difference between the full-context and no-context arms was +0.6 WER points, with a side-resampled interval of [-1.1, +1.0]; the Gemini estimates were too unstable to support a comparable negative inference. A post-hoc rerun found run-to-run pipeline variability larger than the confirmatory differences, so effects of that size cannot be resolved from one transcription per cell. An implementation audit verified the manipulation was live, and sequence-alignment analysis found a small improvement on complete context-listed phrases, too small to materially change side-level WER, and for Gemini coexisting with worsened unlisted-token error. Evaluating context mechanisms therefore requires sequence-aligned term-level, insertion, and speaker-label measures alongside aggregate accuracy. 

---
# The Halt Vector: Internalizing a Causal Steering Intervention for Efficient Reasoning 

**Authors**: Dylan Jayabahu, Tinuade Adeleke  

**Link**: [PDF](https://arxiv.org/pdf/2608.28859)  

**Abstract**: Reasoning models do not stop when they know the answer. On DeepSeek-R1-Distill-Qwen-7B the chain of thought runs about twice as long as the model's own answer probability takes to settle, and how much of that excess is removable varies from problem to problem, so a global length penalty cannot take it out. We take it out by internalizing a causal interpretability finding into the weights. The mechanism is a halt vector: a difference-of-means direction at layer 18 of this model whose steering strength controls how long it thinks, while a replicated value axis does nothing. Installing that intervention in the weights is harder than it looks. Maximizing the scalar projection onto the direction corrupts the off-axis dimensions a frozen downstream reader depends on, and generation gets longer instead of shorter; what works is reconstructing the whole steered activation with those dimensions pinned to their natural values. Fit from 24 problems and no reinforcement learning, the halt removes about a quarter of the thinking at held accuracy across five unseen benchmarks, and the cut tracks each problem's own removable slack at 0.70. It also closes a non-termination pathology that grows with difficulty and that a decoding-time confidence hook makes worse. We do not claim to beat a well-tuned length penalty or decoding-time early exit on the raw trade-off; the contribution is how the halt is obtained. 

---
# Equivariant Sheaf Neural Networks: Learning Geometric Transport on Graphs 

**Authors**: Alessio Borgi, Mario Severino, Fabrizio Silvestri, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2608.28853)  

**Abstract**: Equivariant graph neural networks provide a principled way to model geometric systems, but efficient first-order architectures remain limited in how vector information can be transformed as it moves across a graph. We introduce \textsc{ESNN}, an Equivariant Sheaf Neural Network that enriches this interaction by learning directed, matrix-valued transport between neighboring vector features while preserving exact Euclidean equivariance. Rather than increasing the order of the representation, ESNN keeps scalar and vector features first-order and places the additional geometric flexibility in the edge transport itself. We characterize this transport theoretically, showing that when relative displacement is the only covariant geometric input, every linear $O(n)$-equivariant map decomposes into independent radial and tangential components, while learned covariant features enable richer feature-conditioned transformations. We also introduce controlled symmetry relaxation for systems with a preferred ambient direction, which may be prescribed or inferred from data while recovering full $E(n)$-equivariance when the directional pathway is inactive. Across particle dynamics, mesh-based simulation, point-cloud classification, and molecular property prediction, ESNN improves dynamics prediction, recovers the gravity axis when symmetry is broken, yields substantial gains on selected mesh tasks and long-horizon rollouts, and remains robust to unseen rotations. These results show that learning how geometric information is transported across edges offers a complementary route to expressive equivariant message passing without requiring higher-order representations. 

---
# A rigor-matched audit of periodic-step layer skipping for efficient llm inference: conflayers versus swift, with a supplemental analysis of trained routing alternatives 

**Authors**: Prateek Kumar Sikdar, Arpan Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2608.28846)  

**Abstract**: Layer-skipping methods for efficient LLM inference decide, at some granularity, which transformer layers to execute for a given input. We present a rigor-matched, three-seed audit of two periodic-step, search-based methods that make this decision online at inference time and re-evaluate it every few generation steps: a confidence-gated early-exit baseline (ConfLayers) and genuine self-speculative decoding (SWIFT, Xia et al. 2024), together with vanilla autoregressive decoding, across two model scales (Qwen2.5-0.5B and Qwen2.5-1.5B) and two tasks (GSM8K reasoning and CNN/DailyMail summarization). SWIFT is the strongest method on accuracy in three of four cells; ConfLayers is dominated everywhere, with particularly large deficits on GSM8K at 1.5B. Once online-search overhead is separated from pure inference cost, SWIFT's true inference speed is faster than ConfLayers's in all four cells (5-21%), reversing the naive wall-clock ranking in three of them. ConfLayers's search overhead is small and stable (1-2% of cost), while SWIFT's is larger and more variable (up to 28.7%). We additionally examine two trained-routing methods, LayerRoute (Sikdar, 2026) and LayerDrop (Fan et al. 2020), as a supplemental analysis because they operate at coarser decision granularities. Under a verified protocol with genuine per-input gating, a genuine full-model baseline, and genuine inference-time compute skipping, both show modest speedups (1.08-1.33x) but accuracy well below the periodic-step methods, including a near-total collapse for LayerRoute on GSM8K at 1.5B (0.003 mean exact-match across three seeds). We release the full audit protocol as a template for rigor-matched efficiency comparisons. 

---
# Toward Postural State Classification in Immersive VR with Multimodal Data and Explainability Analysis 

**Authors**: Nipa Anjum, Md Irfan Pavel, Robert Gonzalez Jr, Kevin Desai, Alberto Cordova, M. Rasel Mahmud, John Quarles  

**Link**: [PDF](https://arxiv.org/pdf/2608.28844)  

**Abstract**: Ensuring a safe virtual reality (VR) experience requires systems that can predict and respond when users lose their balance. Although prior work has examined fall prediction and motion sickness, many approaches are regression-based and postural state classification remains less explored. This study compares machine learning (ML) and deep learning (DL) models for classifying postural states in VR under visual perturbations. We used a multimodal dataset containing kinematic, electromyographic (EMG), and electrodermal activity (EDA) signals. The data were prepared for a binary task to distinguish balanced from imbalanced postural states, and participant-wise downsampling addressed class imbalance. All models were evaluated with Leave-One-Participant-Out (LOPO) cross-validation to test generalization to unseen participants. Among the models, the Mamba-inspired CNN (MI-CNN) achieved the highest accuracy of 96.76%. SHapley Additive exPlanations (SHAP) analysis improved interpretability and identified the most influential classification factors. The SHAP results showed that kinematic features were dominant, indicating that body-motion patterns are informative for detecting imbalance in VR. We also evaluated MI-CNN using only the top two-thirds of features ranked by SHAP importance. Despite a 33% reduction in input dimensionality, the model maintained performance, achieving 0.957 accuracy and 0.957 F1-score, with about a 1% decrease compared with the full-feature model. These findings suggest that multimodal sensing, temporal deep learning, and explainable AI can support reliable classification of balance-related instability in VR. Accurate recognition of imbalanced postural states may raise awareness of fall risk and guide safer, adaptive VR systems that respond to instability while improving user safety and experience. Code is available at: this https URL. 

---
# Curvature Cryptanalysis of Smooth Transformer Feed-Forward Networks 

**Authors**: Munawar Hasan, Apostol Vassilev  

**Link**: [PDF](https://arxiv.org/pdf/2608.28843)  

**Abstract**: We show that smooth two-layer feed-forward networks (FFNs) expose an additional structural model extraction channel under a chosen-input raw-output oracle at the FFN branch; consider transformer FFN branches with GELU or SiLU activations under chosen-input raw-output access, without access to parameters, gradients, or internal activations; exploit a second-order leakage channel in which projected input Hessians form different mixtures of the same hidden symmetric rank-one factors induced by the FFN input weights. We formalize resulting Hessian collection as a partially symmetric decomposition to establish conditions for local identifiability and stability to exploit vector-output stencil reuse to reduce the structural query cost by a factor of 16. On independently trained CIFAR-10 vision transformers, only 16 projected Hessians, corresponding to 8193 black-box queries, recover the hidden FFN directions with average absolute cosine alignment above 0.94, with 95.1 % of GELU and 91.9 % of SiLU directions exceeding 0.90 alignment. Recovery remains high across independently trained models, repeated extraction runs, and all transformer blocks. The recovered structure supports functional extraction too. Keeping the recovered directions fixed and fitting only the remaining FFN parameters yields high-fidelity substitutes with more than 93 % top-1 agreement, while test accuracy remains within 0.90% and 0.62% of the GELU and SiLU targets. Output rounding and Gaussian noise substantially reduce recovery under a fixed attack configuration, but adapting the finite-difference step restores average alignment to 0.9603 and 0.9398. This is an end-to-end path from black-box second-order observations to hidden FFN-structure recovery and functional replacement. Under the stated oracle model, smooth FFN curvature exposes internal parameter geometry that behavioral fidelity alone cannot reveal. 

---
# Delegating Before Learning: Where Generative AI Sits in Students' Professional Communication 

**Authors**: Jared Ren, Soobin Cho  

**Link**: [PDF](https://arxiv.org/pdf/2608.28837)  

**Abstract**: We conducted an interview study with twelve students on their use of generative AI in academic communication. Students delegated professional messages to AI most where the pressure to sound professional is highest: email to instructors and administrators. AI involvement ranged from correcting the writer's own text to working out and writing the message outright, and students checked AI-written text against two criteria: whether it looks like AI and whether it sounds like them. Building on these findings, we model the AI-mediated process of writing a student--instructor email at the highest level of involvement we observed, and compare it with an unaided model of writing the same messages, built from participants' accounts and a classic model of the writing process. Three differences emerge: the learning loop that builds writing skill is removed, the message is no longer written for its specific recipient, and the confidence a successful exchange returns goes to using the system rather than to the writer's own ability. From these differences we derive two risks, that individual capacities never form and that authenticity and trust in communication become work. Design can respond to both but is unlikely to be enough, so the risks also need research and policy attention. 

---
# Representation Learning with Quantum Signal Processing 

**Authors**: Junqi Wang, Junyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28828)  

**Abstract**: Representation learning begins when training changes the features that define similarity between data. A frozen-kernel model only reweights a fixed geometry. We establish quantum signal processing (QSP) as a solvable quantum model of the representation-learning regime. At arbitrary depth, we compute the exact mean and variance of its quantum neural tangent kernel, revealing an input-dependent angular geometry whose diagonal remains non-self-averaging even when the underlying unitary approaches Haar randomness. We also prove a sparse-data guarantee for the full nonlinear gradient flow without freezing or ensemble-averaging the kernel: the realized dynamics converges to an integrable scalar flow with a time-dependent kernel closure and explicit convergence times. A finite-depth speed limit holds for every data set and trajectory. At higher data density, numerical results show coupled evolution beyond both the scalar and frozen-kernel descriptions. These results give a controlled theory of learned quantum data geometry with provable training dynamics beyond the frozen limit. 

---
# Text-Driven Artistic Staging: Pose, Lighting, and Camera References from Paintings 

**Authors**: Yunge Wen  

**Link**: [PDF](https://arxiv.org/pdf/2608.28823)  

**Abstract**: Artists coordinate human pose, illumination, and camera placement to convey narrative and emotion, but existing generative methods typically model these elements independently. We introduce text-to-editable 3D staging, a task that jointly generates human poses, a dominant light, and a camera configuration from an affective description. We construct 11,911 text--staging pairs from 2,328 figurative paintings by reconstructing SMPL bodies, estimating low-frequency illumination, recovering camera parameters, and pairing each scene with ArtEmis descriptions. We train a flow-matching transformer that supports variable numbers of figures and produces multiple staging alternatives for each prompt. On held-out descriptions, the model achieves 32.2\% retrieval R@1, compared with 16.6\% for CLIP-based nearest-neighbor retrieval, while approximately preserving corpus-level diversity. These results demonstrate the feasibility of generating editable, emotionally conditioned 3D staging references from text. 

---
# FigMirror: Ground It, Code It, Plot It 

**Authors**: Xiaohan Zhao, Jiacheng Liu, Yaxin Luo, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.28814)  

**Abstract**: Converting scientific figures into executable code has gained increasing attention, yet existing methods primarily focus on reproducing the reference figure itself. A more practical setting is to plot new data while preserving the visual style of a reference figure (e.g., color scheme and typography). Prior approaches mimic the reference through pixel-level optimization and struggle to carry its style to new data. We show that the key to this task lies in the coordinate grounding and coding capabilities present in modern computer-use models. We propose FigMirror, an agentic framework that unlocks these capabilities through Grounded Measurement, which locates visual elements by coordinates and measures their properties through executable code. We further introduce PlotTwin-Bench, an expert-curated benchmark with fine-grained code and image-level style metrics. Experiments show that FigMirror consistently outperforms existing methods on reference-conditioned style transfer. All plots in this paper are generated by FigMirror, except those produced by other methods for comparison. Our code and data are available at: this https URL. 

---
# A Large-scale Evaluation of Text-guided Models for Facial Editing 

**Authors**: Rahul Nair, Saurav Pandit, Hannah Kerner  

**Link**: [PDF](https://arxiv.org/pdf/2608.28802)  

**Abstract**: Facial appearance editing powers popular applications like FaceApp and Photoshop. Generative Adversarial Networks (GANs) and 3D Morphable Models (3DMMs) have been widely used for facial editing. GANs can perform varied facial edits (e.g., changing hair color, hairstyle), but often produce unstable edits. 3DMMs produce stable edits, but can only alter pose and facial expression. Recently, text-guided diffusion models like Nano Banana have become popular for image editing. Text-guided models are a compelling alternative to GANs and 3DMMs since they can produce both stable and varied image edits. While text-guided models have been widely tested for whole-scene edits (e.g., ``make the woman play a guitar''), they have not been comprehensively tested for facial editing. We conducted the first large-scale evaluation ($\sim1$M images evaluated) of six popular text-guided models on a sequential facial editing task. We present Face-Edit-Attributes, the largest collection of $169$ facial editing attributes focused on hair, accessories, and pose edits. We compared model performance using two popular celebrity face datasets: CelebA and CelebSET. Our results show that most models performed hair and accessory edits well, but struggled with editing pose. All models over-edit (e.g., changing hair color when asked only to change the hairstyle). We also evaluated demographic biases in each model. Our results show surprising biases in overediting: almost all models created more overedits for dark-skinned male faces and old faces. The code and data for our results (including our repository of $\sim 1$M images) can be accessed \href{this https URL}{\textcolor{blue}{here}}. 

---
# The reach of a verification tool decides its value: A controlled study of verification surface, artifact quality, and cost in AI coding agents 

**Authors**: Achint Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2608.28795)  

**Abstract**: Modern artificial-intelligence coding agents can be equipped with tools for checking their own work e.g. a linter, a boot probe, a shell, a screenshot tool. We call this set the agent's verification surface. This study asks whether increasing only that surface, with everything else held fixed, produces a matching growth in the quality of the software the agent ships. We built a minimal coding agent whose tool list is the single controlled variable and used it to implement 1,116 web applications across six models and eight tool configurations. A condition-blind human graded every application against a frozen rubric, and automatic probes stress-tested the API-observable behaviors. Verification's cheapest benefit arrives first, which is to make sure that the application comes up. Without any tools, about one build in seven fails to launch at all and a single boot probe removes nearly all of these failures at roughly 35 percent of a full shell's token cost, while the full shell multiplies the no-tools cost by 2.35. Screenshots help most where mistakes are visible (e.g. element placement, interaction), though even there the gain over a shell is modest and does not survive correction for multiple statistical comparisons. In cases where failures can only be measured rather than seen, such as keeping scrolling smooth over a 100,000-row list, screenshots add nothing. A verification tool improves the output artifact only where its reach covers the way the application actually fails. 

---
# ASTRA - Agentic System for Ticket Resolution and Analysis 

**Authors**: Shashidhar Reddy Javaji, Mohamed Trabelsi, Jin Cao, Huseyin Uzunalioglu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28790)  

**Abstract**: Technical operations teams resolve large volumes of incidents by synthesizing fragmented evidence from ticket text, historical cases, system logs, and technical documentation. Existing automation often relies on monolithic generation without explicit evidence modeling or provenance, making outputs difficult to verify when critical signals are sparse across sources. We propose ASTRA, an agentic system for ticket resolution in which a central orchestrator coordinates three specialist information-gathering agents and drives a judge-orchestrator refinement loop to produce evidence-backed troubleshooting reports. TicketSimilarityAgent retrieves relevant historical precedents through dense retrieval and LLM reranking; LogAgent distills hundreds of thousands of log lines into structured, quote-grounded findings using deterministic filtering and constrained LLM analysis; and DomainKnowledgeAgent retrieves relevant technical knowledge via the Model Context Protocol (MCP). Their outputs are transformed into a claim-evidence representation linking each claim to a verbatim source passage, assigning a support level, and preventing cross-attribution. A JudgeAgent scores the report on five criteria, while the OrchestratorAgent converts low scores into targeted follow-up queries for bounded iterative refinement. Evaluated on 987 real-world telecom fault tickets across seven product lines, ASTRA achieves a mean quality score of 4.13/5.0, with 59.9% of reports identifying the fault area at the component-family level or better. Relevance and Clarity scores are 4.88 and 4.94, respectively, while fabricated technical details remain below 3% of error cases. Stratification by fault type reveals that hardware faults remain substantially harder than software or configuration faults (Cohen's d=0.80), pointing to a fundamental limitation of text-based evidence channels for hardware fault diagnosis. 

---
# Peer Oversight in Collective Decision Making 

**Authors**: Sarah Mohsen, Pavel Naumov  

**Link**: [PDF](https://arxiv.org/pdf/2608.28754)  

**Abstract**: This article introduces peer $k$-oversight, a property of sequential collective decision mechanisms requiring at least $k$ agents to be responsible for every harmful outcome. It is shown that whenever $k$-oversight can be achieved by redistributing control over the decisions in a mechanism, it can be achieved using just $k$ agents. A polynomial-time algorithm is also presented that determines whether such a redistribution exists and, when it does, constructs one. These results establish peer oversight as a tractable design principle for multiagent decision-making systems. 

---
# RoboPhys-3D: A Comprehensive Embodied World Model Evaluation via 3D Reconstruction 

**Authors**: Tianyi Wang, Jiazhou Chen, Yiming Xu, Xiangyu Li, Tianyi Zeng, Chih-Hsien Chou, Ning Lu, Liang Peng, Junfeng Jiao, Christian Claudel  

**Link**: [PDF](https://arxiv.org/pdf/2608.28718)  

**Abstract**: Video world models increasingly serve as data engines, action planners, and simulators for embodied AI, but conventional embodied world model (EWM) benchmarks lack a unified 3D-grounded protocol for establishing whether generated rollouts preserve the underlying 3D scene state or translate into executable actions. We introduce RoboPhys-3D, a 3D-grounded EWM benchmark built on RoboTwin 2.0, covering 50 manipulation tasks across four regimes, with 5,000 episodes and 25,000 multi-view ground-truth videos. A defining feature of RoboPhys-3D is that generated and ground-truth videos are processed through the same 3D reconstruction pipeline, enabling reconstruction-induced error to be distinguished from generation-induced error. The RoboPhys-3D benchmark organizes 50 complementary metrics into 18 sub-dimensions across four levels: pixel-level fidelity, 3D geometry consistency, state-level understanding, and task-level completeness. We further introduce Average Full Score, a hierarchical score averaging all 50 metrics for comprehensive evaluation, and RoboPhyscore, a compact task-aligned score averaging the metrics most strongly correlated with task success. Among the four representative video world models, Cosmos 3 achieves the highest RoboPhyscore (0.6330, 92.7% of ground truth), while state- and execution-grounded metrics reveal substantial failures that perceptual and vision-language model-based judgments fail to capture. RoboPhyscore further exhibits strong agreement with human evaluation (Pearson r = 0.9761 and Spearman \r{ho} = 0.8962), demonstrating the importance of grounded, execution-aware evaluation for EWM capability. 

---
# Defending Wearable VLMs Against Private Attribute Inference 

**Authors**: Zhimin Li, Pan Wang, Jingxian Chen, Yuantao Tang, Anthony Chen, Qian Lou, Jingtong Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28691)  

**Abstract**: Wearable VLM pipelines promise continuous multimodal assistance from egocentric visual capture: a user asks a task-driven question about the surrounding scene, and the system uses compact visual tokens to support language reasoning. The challenge motivating this work is that the same egocentric evidence needed for useful assistance can also reveal private attributes about the wearer or nearby bystanders. We investigate this as a joint privacy-utility problem for split VLM inference, where visual encoding occurs within a trusted device boundary but intermediate visual tokens may be transmitted to downstream reasoning components. This exposes an understudied leakage surface: even when final textual responses are benign, external attackers or untrusted downstream components can recover private attributes from transmitted visual tokens. To evaluate this tension, we construct a paired privacy-utility benchmark with 3,221 image-question records, each paired with a utility question and privacy labels covering location, income, sex, and interests. We further propose Token-Guided Attribute Privacy (TGAP), a pre-LLM token disentangler that learns a residual transformation of visual tokens before they leave the trusted boundary. TGAP combines utility preservation, identity regularization, semantic privacy suppression, and image-driven representation suppression, avoiding the utility loss caused by coarse hard or attention masking. On the benchmark used for source-model evaluation, TGAP reduces privacy accuracy from 56.7\% to 7.4\%, a 49.3\% absolute drop, while maintaining relaxed utility at 74.4\%. These results suggest that securing the compact token interface is a practical path toward privacy-preserving wearable multimodal AI. 

---
# Measuring Similarity between Artistic and AI Generated Images using Siamese Neural Networks 

**Authors**: Diego Castro Elvira, Navil Pineda Rugerio, Jesús García-Ramírez, Cecilia Reyes-Peña, Ricardo Ramos-Aguilar  

**Link**: [PDF](https://arxiv.org/pdf/2608.28671)  

**Abstract**: AI-generated art has sparked debates around potential plagiarism, as these images may closely resemble existing artworks. This research quantifies the similarity between original pieces and AI-generated counterparts, particularly those produced by the Stable Diffusion XL Refiner 1.0. We use Siamese Networks with frozen CLIP encoders and cosine similarity optimized through triplet loss. A dataset of paired original and generated images was built using image-to-image generation and custom prompts, enriched with semantic descriptors and BLIP-2 captions. Prior studies report up to 81\% style replication and 90\% visual similarity. Our results show high discriminative performance: training accuracy reached 99.9\%, and the best model configuration achieved 99.4\% test accuracy with strong inter-class separation ($\delta \mu$ = 0.677), demonstrating the effectiveness of our semantic-visual embeddings. 

---
# GreenBench: Benchmarking Energy Efficiency and Carbon Footprint of Open-Source LLM Inference on Apple Silicon 

**Authors**: Rajeswari Kannan, Raj Firke, Shreya Bengle, Srushti Deshmukh  

**Link**: [PDF](https://arxiv.org/pdf/2608.28667)  

**Abstract**: The rapid proliferation of Large Language Models (LLMs) has raised concerns about their environmental impact during inference. While Green AI research has focused on datacenter GPUs and embedded platforms, the energy profile of LLM inference on Apple Silicon, with its unified memory architecture, remains unstudied. This paper presents GreenBench, a benchmarking framework that evaluates the energy efficiency, throughput, and carbon footprint of five open-source LLMs (3-9B parameters) across three NLP tasks on an Apple M4 Pro with 48 GB unified memory. Using macOS powermetrics for direct power measurement and Ollama's nanosecond-precision timing, we find that the M4 Pro draws only 0.47 W of CPU+GPU package power during sustained inference, with total system power of 8-12 W, achieving 30-40x better energy efficiency per token than datacenter GPUs in single-user deployment. Smaller models (3-3.8B) deliver 2.6-4.2x higher throughput and up to 62% less energy per token than larger models (7-9B). Pareto analysis identifies Qwen 2.5 (7B) as the optimal accuracy-efficiency trade-off at 57% MMLU and 59 tokens/s, while Llama 3.2 (3B) suits latency-critical applications at 175 tokens/s. We provide per-token energy at package and system levels with CO2 estimates for India and US grids. 

---
# Test-Time Scaling for Scientific Equation Discovery 

**Authors**: Haowei Lin, Hubert Lim, Xiangyu Wang, Letian Huang, Di He  

**Link**: [PDF](https://arxiv.org/pdf/2608.28660)  

**Abstract**: Test-time scaling (TTS) improves language model reasoning by allocating additional test-time compute, but prior work mainly studies closed-ended tasks such as math and coding. We study TTS for automated equation discovery, an open-ended setting where models search over candidate equations and rely on observed datapoints for feedback. We formulate LLM-driven equation discovery as an iterative search process that unifies Best-of-N, sequential refinement, tree search, and evolution-style methods under a common compute-allocation view. To isolate allocation effects from prompt engineering and other heuristics, we compare minimal parallel controllers under fixed budgets. On LLM-SRBench equation-discovery tasks, we find that search width is the dominant allocation parameter: the best width in our sweep generally increases with the compute budget, while the population--branching split and controller choice matter less. Appropriate width selection also improves wall-clock efficiency by increasing parallelism. These results suggest that, given an informative verifier, controlling exploration and exploitation is central to scaling LLM-based equation discovery. 

---
# Can Large Language Models Identify Meaningful Touchpoints in Conversion Attribution? 

**Authors**: Jinqi Wu, Sishuo Chen, Zhangming Chan, Yong Bai, Chao Yi, Han Zhu, Shuodian Yu, Lei Zhang, Sheng Chen, Chenghuan Hou, Jian Xu, Chaoyou Fu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28649)  

**Abstract**: Touchpoint selection in conversion attribution, namely identifying meaningful touchpoints contributing to conversions, is essential for e-commerce recommendation and online advertising. Current selection methods rely heavily on collaborative-filtering-based heuristics, which fail to align with user-perceived semantic intent. Through human annotation, we reveal a significant semantic gap: many implicitly-related, semantically relevant touchpoints remain undetected by existing rules. Therefore, we systematically evaluate the capability of Large Language Models (LLMs) in identifying these hidden associations. Our evaluation shows that while LLMs effectively uncover a substantial portion of implicitly-related touchpoints, significant room for improvement remains in their selection performance. Furthermore, we analyze the impact of different prompting strategies and foundation model choices on identification performance, providing valuable insights into their reasoning patterns and effectiveness. These insights offer a new roadmap for transitioning conversion attribution from mechanical rule-matching to human-aligned semantic reasoning. Moreover, we leverage the LLM-attributed conversion labels for enhancing industrial CVR model training and achieve significant offline performance gains, showing the potential of LLMs in conversion attribution. 

---
# Cross Lingual Transfer in Tulu Legal Comprehension: Script-Dependent Improvement and RAG-Induced Knowledge Conflict 

**Authors**: Sindhu Shetty, Spurthi Setty, Natan Vidra  

**Link**: [PDF](https://arxiv.org/pdf/2608.28645)  

**Abstract**: Low-resource languages without an adequate training corpus often use a related, higher-resource language as a scaffold for comprehension. Still, there is a need to develop rigorous evaluation methods to identify when models fail in cross lingual low-resource environments. Using the legal domain as a backdrop, three models (Llama3, Hex-1, Sarvam) were tested on the ability to classify legal complaints written in a low resource Dravidian language (Tulu). Transliterating queries across Dravidian scripts allowed models to gain a preliminary understanding of speakers' complaints without the use of wide scale training, though the level of comprehension was heavily script dependent (with Kannada - another relatively low-resource language - producing the strongest positive trend). Retrieving from a corpus of Kannada legal papers across a RAG framework caused mixed results. Some models had a weak positive trend in comprehension under certain conditions, but when models failed, it was often across two axes: fact substitution (fixating on specific passage excerpts that skewed reasoning) and confabulation (hallucination that had no basis in either query or corpus). Within low resource domains, results identify the model's parsing of information and subsequent reasoning as the source of reasoning failure, rather than corpus contents. Script-dependent comprehension and RAG robustness also seem to travel together. This is further supported by the reasoning-trace analysis and a statistical-honesty framework deployed - techniques that are more broadly applicable to low-resource multilingual RAG evaluation. 

---
# Redesigning and Auditing Deep Research Writing for Faithful Reports 

**Authors**: Hiroaki Hayashi, Pranav Narayanan Venkit, Prafulla Kumar Choubey, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28643)  

**Abstract**: Rubric-based evaluations of deep-research (DR) systems often obscure fine-grained factual failures in generated reports. We introduce CLAIMPROBE, a claim-level audit that decomposes DR reports into claims and measures hallucination, misattribution, citation hygiene, and necessary-fact recall against retrieved evidence. Using CLAIMPROBE, we find that strong DR pipelines can omit key evidence and misattribute claims even when their rubric scores remain stable. We then propose CLAIMWRITER, a hierarchical claim-based writer that extracts source facts, maps them to a query-derived outline, and drafts each section from a source-linked claim representation. Across three prior DR frameworks, replacing only the report writer with CLAIMWRITER reduces hallucination by 2.6 to 4.5 times and improves necessary-fact recall by 1.2 to 1.7 times, while largely preserving overall report quality. CLAIMWRITER also enables localized revision: when sources change, it propagates changed source facts into revised reports at the highest rate among update methods, while also being more cost-effective. 

---
# Terminal-Bench-LILT: Multilingual Agentic Coding Benchmark Grounded in Language, Region, and Culture 

**Authors**: Yunsu Kim, Kaden Uhlig, Ashwin Purohit, Milind Agarwal, Patrick Simianer, Anil Arslan, Kiarash Mokhtari, Thomas Zenkel, Johannes Mosig, Gabriel Bretschner, Shamik Bose, Joern Wuebker, John DeNero  

**Link**: [PDF](https://arxiv.org/pdf/2608.28641)  

**Abstract**: Most evaluations for coding agents are conducted exclusively in English, which does not reflect real-world multilingual deployment. We present Terminal-Bench-LILT, a suite of 300 authentic coding tasks in ten languages: Arabic, Czech, German, Spanish, Hindi, Japanese, Korean, Serbian, Turkish, and Chinese. Each task targets issues specific to non-English software development that have no direct English equivalent, e.g., internationalization, encoding, text normalization, and cultural conventions. All tasks are authored by native-speaker programmers and validated through a multi-stage quality control pipeline. Evaluation of six frontier models reveals that even the strongest model reaches only 63.1\% pass rate, with many tasks unsolved by any model. Performance varies substantially by language and does not track general coding benchmark rankings, highlighting that multilingual coding competence is a distinct and underexplored capability axis. Sample tasks are available at this https URL 

---
# PromptKWS: A Novel Prompt-Guided Open-Vocabulary Keyword Spotting Framework 

**Authors**: Gaopeng Xu, Chengfei Li, Xianliang Wang, Lin Zhu, Juan Wei, Wenpeng Li, Jianwei Niu, Jie Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.28640)  

**Abstract**: In this paper, we present PromptKWS, a novel Prompt-guided keyword spotting (KWS) framework to improve the accuracy of open vocabulary KWS systems. In specific terms, we introduce the Prompt Phrases Prediction Network (PPN), an encoder-decoder architecture designed to effectively extract keyword prompts embeddings. we employ the PPN encoder to encode the keyword prompts and infuse the prompt embedding into the Prompt-guided KWS encoder by utilizing a Prompt-acoustic Multi-head Cross-attention (MHCA). Experiments show that PromptKWS improves the wakeup rate by over 10% compared to baseline system. Notably, another strength of PromptKWS is its ability to effectively leverage keyword prompts for adapting to complex real-world environments involving noise and pronunciation variations. In comparison to purely acoustic models, which often struggle in such situations, PromptKWS demonstrates remarkable performance, with an average accuracy improvement of over 15% in test sets. 

---
# Do MLLMs Really Understand Low-Resource Khmer Documents? A Pilot Study on Khmer Document VQA 

**Authors**: Nimol Thuon, Panhapin Theang  

**Link**: [PDF](https://arxiv.org/pdf/2608.28635)  

**Abstract**: Recent multimodal large language models (MLLMs) have advanced document understanding, visual question answering, and text extraction. However, their reliability in low-resource, non-Latin settings remains uncertain. Khmer form documents present particular challenges because they contain complex script forms, mixed Khmer-English fields, and monetary values in both Cambodian Riel and US Dollars. Available resources for Khmer Document VQA are also limited. This paper presents a pilot diagnostic evaluation of open MLLMs on Khmer document images. We construct an evaluation subset from the previously introduced KH-FUNSD collection, covering invoices, receipts, quotations, and other business forms. The subset includes questions in English and Khmer, with answers retained in their original English, Khmer, mixed-script, or numeric forms. Rather than introducing a full public benchmark, this study examines the capabilities and failure modes of existing models. We evaluate representative open Qwen-VL models using direct image-based prompting and compare parser-assisted and external OCR-assisted configurations with Qwen3-VL-8B. Direct Qwen3-VL-8B outperforms smaller models, achieving 51.9% overall accuracy, although performance remains limited for Khmer-script and mixed-script answers. External OCR produces the strongest results, reaching 61.9% with Tesseract and 61.6% with PaddleOCR. Nevertheless, Khmer-script answers remain substantially more difficult than English and numeric fields. The results indicate that current MLLMs can process visually clear English and structured numeric content, but reliable native Khmer document understanding remains an open challenge. 

---
# PAUSE: Editable Strategy Artifacts for Long-Form Cultural Story Adaptation 

**Authors**: Taaha Kazi, Vasu Sharma, Mohammad Saifullah, Abdur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2608.28633)  

**Abstract**: Generative AI systems increasingly mediate cultural adaptation, but their cultural decisions are often hidden inside prompts, transient model plans, or final prose. We study PAUSE (Pause-And-Update Strategy Editing), an intervention that exposes an editable adaptation strategy as a human control surface for cultural decisions in long-form story adaptation. The strategy is a structured artifact that can be inspected, edited, and then projected through downstream character, entity, and chapter-localization stages. In two Chinese-source serialized novels, we test whether human edits to this strategy propagate into chapter-level prose. Across 9 edited-vs-control chapter comparisons, judges select the edited-strategy output in all 9; a marker audit shows target markers in 8/9 edited outputs and 0/9 controls, with forbidden markers absent from edited outputs and present in all controls. We frame these results as a smoke-scale edit-adherence study, not a claim that the outputs are culturally authoritative or literary-quality improvements. PAUSE offers one practical way to make AI-mediated cultural adaptation more inspectable and contestable before decisions propagate through long-form generation. 

---
# Enabling Proactive Spoken Turns via a Generalized Style-Aware Full-Duplex Framework 

**Authors**: Tianrui Pan, Qinglin Zhang, Chong Deng, Luyao Cheng, Qian Chen, Wen Wang, Jie Tang, Gangshan Wu, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28630)  

**Abstract**: Compared with half-duplex dialogue systems where the system waits for user turn completion before it responds, natural full-duplex dialogue systems require agents to act proactively in real time, including timely interruptions and backchannels. This creates a key challenge: improving turn timing without sacrificing response quality. To address limitations in realistic proactive turn-taking, we build a generalized style-aware full-duplex framework with three key components. Firstly, we propose LPS-TC, a Lightweight Proactive Speech Turn Controller for plug-and-play integration. It features a fine-grained action space covering both reactive and proactive turn behaviors, enabling half-duplex models with full-duplex capabilities and enhancing existing full-duplex models with superior timing control. Secondly, we construct WildTurn, a large-scale, real-world English dataset containing approximately 2,981 hours of filtered multi-turn stereo conversations from face-to-face and telephone conversations, annotated with five turn-taking and five backchanneling styles. Trained on WildTurn, LPS-TC exhibits rich spoken dynamics that are not captured by existing static full-duplex benchmarks. Thirdly, we introduce a two-tier evaluation scheme that assesses both chunk-level timing precision and turn-level interaction quality under realistic streaming constraints. Our experiments, integrating LPS-TC with half-duplex models like Qwen2.5-Omni and full-duplex models like Freeze-Omni, showcase its superior performance in timing appropriateness and response quality. Our framework also demonstrates fine-grained style controllability and strong generalizability, enabling more natural and human-like spoken interactions. 

---
# Intelligent Identification and Repair of Design Defects in BIM via Domain-Specific Large Language Models 

**Authors**: Jia-Rui Lin, Yun-Hong Cai, Xiang-Rui Ni, Peng Pan  

**Link**: [PDF](https://arxiv.org/pdf/2608.28629)  

**Abstract**: Existing methods lack a generalized approach to efficiently identify and resolve the diversity of design defects in BIM. Therefore, this study proposes an integrated framework to identify and repair various defects in BIM via domain-specific LLMs. Firstly, a BIM-to-Text method with component-balanced chunking is introduced to bridge BIM data with LLMs. Then, prompt learning with rule injection, few-shot prompting and RAG is proposed to identify defects and generate repair suggestions. Meanwhile, a hallucination control strategy combining key identifier validation and token-length thresholds is introduced to ensure reliability. Experiments show capability expansion yields 85% identification accuracy versus 70% for traditional rule checking, achieving a 94% rate of reasonable repair suggestions. Moreover, the proposed hallucination control further increased accuracy from 64% to 85%, eliminating 92.5% of hallucinations in a single intervention round. This study establishes an end-to-end prototype from raw BIM data input, through defect identification, to repair suggestion generation. 

---
# Do large language models scrutinise what they review? A multimodal audit of scoring calibration, error detection, and author-identity effects 

**Authors**: Emad Alharbi  

**Link**: [PDF](https://arxiv.org/pdf/2608.28626)  

**Abstract**: Large language models (LLMs) are increasingly used to generate peer reviews, prompting examination of their capacity for critical evaluation. This study evaluates two multimodal LLMs, Qwen2.5-VL-72B and Pixtral-Large-124B, as reviewers across 165 submissions to the 2026 International Conference on Learning Representations, a venue that postdates both models' training cutoffs. Manuscripts were presented to both models with author identities blinded, replaced with high-prestige affiliations, or replaced with low-prestige affiliations, and in either text-only or text-with-figure format. Additionally, 145 verifiably detectable errors were inserted into 55 manuscripts to assess error identification under natural and verification-oriented prompts. Across all manuscript groups, including rejected submissions, LLM scores ranged from 7.0 to 8.1, whereas human mean scores ranged from 3.4 to 6.8. The models detected 12.1\% of the verified errors under natural prompting, and a one-sentence verification instruction increased detection to 22.2\%; however, 78\% of the errors remained undetected. Providing figures reduced error detection while increasing review scores. No visual error was reliably verified against its corresponding figure, and half of the text-only reviews described figures that were not provided. Author identity did not influence either review scores or error detection. LLM editorial decisions exactly matched those produced by simple score averaging. 

---
# Asymmetric Within-Document Predictive Learning for Scientific Document Representation 

**Authors**: You Zuo, Éric de la Clergerie, Benoît Sagot  

**Link**: [PDF](https://arxiv.org/pdf/2608.28625)  

**Abstract**: We study predictive pretraining for scientific document representation using the discourse structure of papers. We propose SciJEPA, a citation-free framework that learns through asymmetric within-document prediction: title and abstract representations are used to predict method representations, and method representations are used to predict conclusion representations. Experiments on RELISH, high-influence citation, SciDocs, and cite prediction show that plain predictive training is viable but weaker than a controlled contrastive baseline using the same section pairs. Adding Sliced Isotropic Gaussian Regularization (SIGReg) substantially improves performance and narrows this gap. The effect of regularization is task-dependent: moderate SIGReg helps fine-grained ranking, while stronger regularization can weaken local alignment. We further show that different encoding branches support different retrieval regimes. These results position within-document predictive learning as a promising citation-free complement for scientific document representation, provided that embedding geometry is carefully controlled. 

---
# MA-RAG: Multi-Agent Retrieval-Augmented Generation for Query-Driven Summarization of Longitudinal Parkinson's Disease Assessments 

**Authors**: Sana Alamgeera, Denise Goberta, Muhammad Irshad, Anne H. H. Ngu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28624)  

**Abstract**: Accurate interpretation of single-visit and longitudinal clinical assessments for Parkinson's disease is time-consuming and often depends on specialist expertise. Although large language models (LLMs) can generate natural language summaries, they frequently lack domain-specific clinical grounding and struggle to produce factually correct and temporally consistent responses for structured longitudinal assessment data. To address these limitations, we propose MA-RAG, a query-driven multi-agent retrieval-augmented generation framework that decomposes clinical reasoning into domain-specialized agents, combines structured fact extraction, and synthesizes clinically grounded summaries through a final verification stage. The framework supports four clinical analysis tasks: single-session, trajectory, comparison, and cohort summarization. We evaluate MA-RAG using objective metrics, namely Fact Precision, Hallucination Rate, Temporal Fidelity, and Semantic Similarity, together with subjective evaluations conducted by clinical experts. Compared to Traditional, RAG-only, and Single-agent RAG baselines, MA-RAG substantially improves factual correctness, achieving up to a 122% relative increase in Fact Precision (from 0.436 to 0.990) and reducing the Hallucination Rate by up to 98% (from 0.564 to 0.010), while consistently receiving top ratings from clinical experts for organization and clinical usefulness. These results demonstrate that domain-specialized multi-agent reasoning enables reliable query-driven summarization of structured longitudinal clinical assessment data. 

---
# Looking Again: Measuring Sycophancy in the Reasoning Chains of Multimodal Models Under Pressure 

**Authors**: Mahir Numayeer Islam, Gakuto Okuyama, Nikolaus Siauw, Shivank Garg, Madhur Panwar, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2608.28623)  

**Abstract**: Large multimodal reasoning models (LMRMs) are getting increasingly capable, primarily through generating explicit chain-of-thought reasoning before answering. In language models it has been observed that this performance often comes with sycophancy, the tendency of a model to agree with the user over the evidence. However, for LMRMs no reliable method to measure sycophancy yet exists. We bridge this gap by introducing a benchmark and dataset for evaluating LMRM sycophancy when confronted with a wrong answer from a user. Our benchmark pairs four visually grounded datasets spanning mathematical, clinical, temporal, and demographic reasoning with five pressure conditions in single-turn and multi-turn settings. We evaluate sycophancy in the final answer as well as its emergence within the reasoning chain. We find that sycophancy is prevalent under pressure, with Statement pressure eliciting the highest rates and Conviction the lowest for all models except Mistral-Small-4, and under multi-turn pressure reasoning-level sycophancy intensifies sharply in clinical visual judgement, reaching 95.7% for the most affected model. We further introduce a failure taxonomy separating reasoning-chain from answer-level sycophancy, and a complementary sentence-level taxonomy locating where in the chain drift first emerges. Our results show that sycophancy can corrupt the reasoning chain independently of the final answer, so answer-level evaluation alone is insufficient. 

---
# PUFFER: Incremental Fuzzy Deduplication for Continuously Evolving Corpora 

**Authors**: Xiao Yang, Erik Edward Aldape, Beren Millidge  

**Link**: [PDF](https://arxiv.org/pdf/2608.28622)  

**Abstract**: Large language model training corpora grow through successive, often redundant releases, so each release must be deduplicated against both itself and the accumulated history. At trillion-token scale, this requires incremental ingestion, bounded resident memory, deterministic retry, and dataset-scoped lifecycle control without repeated corpus-wide rebuilding. We introduce PUFFER (Provenance-aware Updatable Fuzzy Filtering for Evolving Repositories), a MinHash-LSH fuzzy-deduplication pipeline built around two design choices. First, PUFFER stores each LSH band as immutable, dataset-tagged, memory-mapped sorted segments, enabling exact historical band-key membership checks without RAM proportional to corpus size. Second, T-fanout tiered compaction periodically merges segments to control screening fanout, trading lower query cost against additional index-maintenance writes while preserving membership decisions. Across N ingested keys and K equal-sized releases, PUFFER's cumulative maintenance cost is O(N log N log_T K), compared with Theta(KN) for repeated snapshot rebuilding. Dataset-tagged segments also support dataset-scoped withdrawal: removal is constant-time for uncompacted or protected datasets, while post-compaction withdrawal reconstructs only the affected merged segment, even if the original dataset is unavailable. In our implementation, PUFFER completed cumulative index-stage ingestion for one billion documents in about 1.75 hours in a single process, using 128 bytes per document for a 16-band index. A classical resident MinHash-LSH table required about 6.5 KB per document and exceeded a 900 GiB RAM cap. In a ten-hour comparison capped at one billion documents, PUFFER was 11x faster than LSHBloom and 35x faster than Milvus-LSH. PUFFER is deployed on more than 30 billion documents, and we release it as open-source software at this https URL. 

---
# From GenAI Virtual Patient Dialogue Logs to Teacher-Interpretable Process Evidence: A Learning Analytics Study in Higher Education 

**Authors**: Xinyu Li, Zijian Li, Mengyu Xia, Luzhen Tang, Naping Chen, Changmin Lin, Danijela Gasevic, Dragan Gasevic, Yizhou Fan  

**Link**: [PDF](https://arxiv.org/pdf/2608.28619)  

**Abstract**: Medical history taking is a dialogue-based clinical reasoning task in which learners must gather, organise, and integrate patient information while the consultation unfolds. Generative AI-powered virtual patients (GenAI VPs) make repeated history taking practice scalable and preserve full turn by turn dialogue. However, these logs are educationally difficult to use directly. Complete transcripts are too detailed for routine teacher review, whereas final scores obscure whether learners followed up patient cues, checked uncertainty, or used summaries to guide later questioning. This study examined whether coded GenAI VP dialogues can provide teacher-interpretable process evidence of clinical reasoning. We analysed 1{,}030 GenAI VP dialogues from 210 second-year medical learners across five weeks chest-pain cases. Each consultation was teacher-scored using a rubric assessing the full history taking dialogue, and consultations were classified within each week as high- or low-rated using the weekly median score. To explain how rated performance was reflected in the dialogue process, we applied three analytic layers to the same coded dialogue data: behavioural prevalence, local co-occurrence using Epistemic Network Analysis, and sequential transition using Transition Network Analysis. High-rated consultations involved more history taking activity, but differences were not simply about volume. High rated consultations more often connected information gathering and symptom exploration with communication, checking, organisation, and synthesis. Summarising and organising moves more often led to verification or mechanism-oriented follow-up. These findings show how layered analysis of GenAI VP dialogue logs can reveal process patterns associated with high rated history taking and support process-focused feedback in medical education. 

---
# STAGEET: Stage-wise Typed Edit Tagging for Grammatical Error Correction with Arabic as a Case Study 

**Authors**: Wenjie Lou, Alaa Mamdouh Akef  

**Link**: [PDF](https://arxiv.org/pdf/2608.28614)  

**Abstract**: Sequence-to-edit approaches make grammatical error correction (GEC) efficient and locally interpretable by predicting edit labels over the input rather than generating a full corrected sentence. Their interpretability, however, is primarily operational: a label specifies how the string should change, but a single edit vocabulary does not always reveal the type of correction being made. We propose STAGEET, a stage-wise typed edit-tagging framework that reorganizes Seq2Edit supervision into typed executable stages and extends edit operations to correction categories. STAGEET decomposes correction into an ordered sequence of medium-grained typed stages; each stage predicts from its own label space, rewrites the current hypothesis once, and passes the resulting intermediate sentence to the next stage. We instantiate the framework as both an end-to-end shared-encoder multi-head model with stage-specific adapters and a fully specialized variant with one independent tagger per stage. Experiments on QALB-2014 and ZAEBUC show that category-aware staged correction retains competitive edit-based GEC performance while exposing a more inspectable correction trajectory, and attains state-of-the-art results on QALB-2014. 

---
# Gurukul AI: An Interactive AI-Driven Educational Platform for Indian Education System 

**Authors**: Isha Narang, Sneh Gosai, Mayank Singh  

**Link**: [PDF](https://arxiv.org/pdf/2608.28611)  

**Abstract**: Recent advances in large language models (LLMs) like ChatGPT and LLaMA have transformed AI-driven education, but these systems are predominantly trained on Western-centric data, making them ill-suited for regional curricula like India's. The Indian education system is linguistically diverse, exam-oriented, and structured around standardized syllabi, not addressed by existing datasets or tools. In this work, we curate a syllabus-aligned QA dataset based on NCERT (National Council of Educational Research and Training) textbooks for classes 9-12, capturing the content, context, and teaching style of Indian curricula. The final dataset, comprising 18,720 question-answer pairs across five subjects, is publicly available at this https URL. We fine-tune the LLaMA 3.1 8B model using this dataset and deploy it in a Retrieval-Augmented Generation (RAG) framework tailored to educational needs. We introduce GurukulAI, an open-access platform that enables Indian students to chat with the model, get doubts cleared, practice exam-style questions, receive contextual answers, and interact in both English and Hindi. By localizing AI for Indian classrooms, our work bridges the gap between global LLM capabilities and regional educational demands. The code is available at this https URL. 

---
# Parametric Multimodal User Memory: Storing What Captions Cannot Carry 

**Authors**: Bojie Li, Noah Shi  

**Link**: [PDF](https://arxiv.org/pdf/2608.28609)  

**Abstract**: A personalized agent needs a user memory: a persistent model of who its user is. Today it is almost always text -- transcripts and captions retrieved by similarity. This serves the captionable half of a person ("my cat is named Bibi"), but discards the perceptual half no caption can hold: how a voice sounds, how a face reads across age and lighting, how tired someone sounds. We measure this loss across five modalities: a strong caption-based re-identifier recovers as little as 0.11 of a dedicated encoder's recall, collapsing toward chance on non-nameable signals.
We instead ground perceptual memory in the model, decomposing recall into two subproblems: a vision-language model grounds the referent in context (what and where), and a dedicated encoder extracts an identity key (who), stored as one inline token read by attention at generation with no external round-trip. Neither suffices alone -- the VLM identifies cross-age faces at only 0.54 recall where a face encoder reaches 0.81, and an ungrounded encoder recognizes a two-person-scene referent at 0.05 -- yet together they reach correct-region oracle (0.96), generalizing to multi-speaker audio and video. The recognition core is training-free: it reproduces the encoder's recall on any frozen model at O(1) registration cost. On PerceptMem (12 domains, 1,080 tasks) perceptual identity is capacity-limited while exact facts are binding-limited: identity belongs in a parametric bank, facts in a text store. The two memories compose cleanly: an agent with both can remember not only what its user said, but also what they are like. 

---
# NLP-Driven Knowledge Extraction and Thematic Classification of Translated Ancient Indian Medical Texts 

**Authors**: M. S. Rajeevan, B. Mini Devi, V.S. Anoop, C. Mallikarjuna  

**Link**: [PDF](https://arxiv.org/pdf/2608.28608)  

**Abstract**: Ancient Indian medical texts like Sushruta Samhita have extensive information on diseases, treatments, and surgical techniques. Yet, their ancient format and use of intricate vocabulary pose difficulties in accessibility and systematic ordering. The research here utilizes Natural Language Processing (NLP) methods like Named Entity Recognition (NER), BERTopic modeling, and Knowledge Graph development in Neo4j to extract, categorize, and visualize important concepts based on translated versions. Thematic classification with BERTopic allows for the identification of the underlying medical topics, whereas NER supports the structured entity recognition of diseases, treatments, researchers, and medicinal plants. Graphbased network analysis with Neo4j also allows for the semantic representation of relationship among extracted entities, supporting knowledge retrieval and digital preservation. The findings illustrate how graph databases, topic modeling, and entity recognition facilitate the computational organization of Ayurveda's historical medical wisdom, closing the gap between the conventional texts and contemporary data-driven inquiry. The suggested method promotes historical text analysis, medical informatics, and digital humanities to make ancient Indian medical wisdom more accessible and understandable. 

---
# PowerSlider: Exploiting Phase Asymmetry for LLM Serving under Demand Response 

**Authors**: Yueying Li, Jiayang Chen, Yuanfan Chen, Leo Han, Haoran Qiu, Esha Choukse, Rodrigo Fonseca, Udit Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2608.21719)  

**Abstract**: AI inference clusters are increasingly constrained by instantaneous power, not just energy: grid operators condition new capacity on demand response, imposing time-varying power caps. Existing LLM serving systems optimize a static energy objective or shed fixed priority tiers under load; either way, goodput collapses when the power envelope moves. An LLM pipeline is not a uniform load: compute-bound prefill loses throughput almost linearly with GPU frequency, memory-bound answer decode sustains it down to $0.57\times$ nominal, and reasoning's thinking phase couples KV-cache capacity to scheduling -- so a cap should be steered to where each watt costs the least performance. PowerSlider does so with a new Flex SLO contract that turns bounded user slack into an optimization constraint, prefill--think--answer disaggregation exposing per-stage frequency and KV control, and a Karush--Kuhn--Tucker (KKT) online solver re-solving within 7.7 ms of every cap change, backed by a consolidated fail-safe that power-gates drained instances when DVFS bottoms out on static power. On SGLang with production traces, \sys{} sustains 78.3\% online goodput at a 30\% cap reduction versus 47.6\% for the best of five baselines ($1.64\times$), holds latency-critical tails within $1.3\times$ of nominal (baselines: $2.3$--$6\times$, up to $12\times$), and delivers 92\% mean goodput through a replayed CAISO grid-emergency day bottoming at $0.41\times$ (54\% at the trough; every baseline below 7\%). 

---
# Agent-Based Model Framework for the North Carolina Modeling Infectious Diseases Program (NC MInD ABM) Overview, Design Concepts, and Details Protocol 

**Authors**: Kasey Jones, Emily Hadley, Caroline Kery, Alexander Preiss, Marie C.D. Stoner, Sarah Rhea  

**Link**: [PDF](https://arxiv.org/pdf/2202.06853)  

**Abstract**: To help facilitate a variety of simulations related to healthcare facilities in North Carolina, we have developed an agent-based model (ABM) to accurately simulate patient (i.e., agent) movement to and from these facilities. This is an Overview, Design Concepts, and Details (ODD) Protocol, a standardized method for describing ABMs. This ODD provides detailed information on healthcare facilities in North Carolina, the agent movement to and between them, and any decisions that were made during the creation of this model. This ABM is intended to be used alongside disease-specific submodels. It can be used for purposes such as simulating the success of interventions on reducing disease transmission, simulating strain on facility resources (including staff and materials), and forecasting hospital capacity. Disease-specific ODDs should accompany this document. No details related to any submodels that use this ABM as a base model are included. 

---
