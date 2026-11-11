# FERN: Cross-Benchmark Table Retrieval for Text-to-SQL with Frozen Retrievers and Learned Ranking 

**Authors**: Arjun Rao, Nishant Gopinath  

**Link**: [PDF](https://arxiv.org/pdf/2611.14477)  

**Abstract**: Text-to-SQL systems over large pooled catalogs must first retrieve the tables a question needs. We compare against the published table recall of RASL, which uses Claude 3.5 models to extract keywords and select tables, on pooled BIRD, Spider, and Fiben catalogs. FERN (Feature-Engineered Ranking with No generative LLM) instead fuses three frozen retrievers into 100 candidates, describes each with up to 120 features, and reorders them with LightGBM rankers fitted once on BIRD's training questions. On Spider, where it was not fitted, its candidate pool alone reaches RASL's published recall at fifteen tables (98.0), and the ranker raises the share of questions with every gold table in the top five (complete@5) from 83.9% to 95.6%. At five tables, FERN's recall is within one point of RASL's on Spider (96.7 vs 97.0) and on BIRD with its evidence hints (96.6 vs 97.5), and 8.2 points below on BIRD without them (89.3). On 121 real BEAVER questions, without BEAVER fitting, its recall at fifteen (82.1) falls between two BEAVER-trained retrievers (80.8, 84.5), but its complete@15 is below both (49.6 vs 54.5 and 62.0). On Fiben, it falls far short at fifteen tables (54.2 vs 69.2, different question sets), although its recall at fifty is numerically close to RASL's question-only retrieval result (80.7 vs 79.4). On BIRD dev with evidence, non-retrieval features raise complete@5 by 7.7 points over retrieval signals alone. On BIRD dev, median latency is 208 ms per question on one A100, without hosted model calls. 

---
# SHARP: Semantic Hierarchical Agentic Reasoning for Next POI Prediction with Dynamic Memory 

**Authors**: Yuhang Zhan, Silin Zhou, Zhuo Chen, Haoxiang Lei, Yunren Feng, Lisi Chen, Shuo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2611.14220)  

**Abstract**: Next point-of-interest (POI) prediction supports personalized location services by anticipating users' next destinations. Recent methods leverage large language models (LLMs) for semantic reasoning and graphs for collective mobility modeling, improving POI prediction. However, three challenges still remain. Bridging LLMs' semantic reasoning and specific POI prediction remains challenging. User mobility also evolves over time, making it necessary to retain and update historical knowledge within limited LLM contexts. Meanwhile, global transition graphs smooth out time-specific category transition patterns. To address these challenges, we propose Semantic Hierarchical Agentic Reasoning for Next POI Prediction with Dynamic Memory (SHARP), an agentic hierarchical framework in which category-level reasoning guides specific POI prediction. A trajectory memory agent maintains a dynamic trajectory memory bank that preserves earlier mobility patterns and tracks behavioral changes with limited context growth. We construct multi-granularity time-aware category transition graphs with global and time-conditioned views to preserve temporal regularities obscured by global aggregation. A category predict agent combines these graphs with trajectory memory to predict likely next categories. SHARP learns semantic IDs (SIDs) through a residual-quantized variational autoencoder (RQ-VAE) to construct semantic POI representations, which are jointly optimized with a category-guided POI predictor end-to-end. On three real-world datasets, SHARP improves Acc@5 and Acc@10 by an average of 5.29 and 4.52 percentage points, respectively, over the strongest baseline. 

---
# From Search to Decision: How Post-Training Shapes Information-Seeking Capabilities in Large Language Models 

**Authors**: Haohang Huang, Xuan Lu, Mingjin Kuai, Yuxuan Zhang, Wei Ji, Haiyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.14149)  

**Abstract**: Information seeking in large language model (LLM) systems increasingly involves decisions about when to acquire information, where to search, and how to evaluate and use evidence. Recent post-training methods have improved several such behaviors, but their objectives and evaluations span different tasks and system settings. This paper organizes research around six interrelated information-seeking capabilities and examines the learning signals, optimization methods, update loci, and credit-assignment mechanisms used to develop them. We synthesize empirical progress in targeted behaviors, end-to-end performance, search control, and generalization. Existing studies report gains in retrieval routing, query refinement, multi-tool exploration, evidence-grounded reasoning, and resource-aware search, although the scope of behavioral evidence varies across capabilities. We identify open challenges in estimating prospective retrieval utility, coordinating information-seeking decisions, and adapting to unfamiliar retrieval environments, and discuss evaluation practices for interpreting future progress. 

---
# Ontology-Guided Knowledge Graph Construction and Robust Evidence Retrieval for Indian Knowledge Systems 

**Authors**: Anand Kaniyaar Sriranga Prasad, Avinash Holenarasipur Nagendra, Mohan Raghavan  

**Link**: [PDF](https://arxiv.org/pdf/2611.14050)  

**Abstract**: Retrieving evidence from Indian Knowledge Systems is complicated by transliteration variants, ambiguous names, and relations distributed across passages. We present an ontology-guided knowledge graph (KG) pipeline that links typed entities and relations extracted with Qwen3-32B to original textual evidence. Retrieval combines sense-aware grounding, bounded graph traversal, lexical anchors, relationship-vector search, and passage reranking. We evaluate KG alone, single-vector retrieval (SV), and KG unions with SV and Entity Notes on 320 questions from Dhara Bench, matching the per-question candidate count across internal systems under BGE and Qwen rerankers. Standalone KG has the highest Precision@5 under both rerankers and the highest MRR@20 in Counting, Composite, and Multi-hop among the evaluated internal configurations. With Qwen, KG achieves Multi-hop MRR@20 of 0.8722, compared with 0.7622 for the fully reranked KG + SV union. The KG + SV + Entity Notes union instead provides the highest overall Recall@20 (0.5134), compared with 0.4347 for KG. On the 15 Multi-hop questions, Qwen KG achieves complete-evidence MRR@20 of 0.4651 versus 0.3372 for SV, while both full unions complete ten questions within twenty results compared with nine for KG. Gemini 3.7 Flash, evaluated through manually reviewed source mappings, re mains stronger on overall MRR and nDCG, while internal systems have category-specific advantages. All comparisons are descriptive. A separate deployment timing study measures mean retrieval times of 20.374 s for KG and 0.646 s for SV. The findings support selecting KG or its unions according to the required balance of early relevance, evidence completeness, and retrieval cost. 

---
# From Show Programs to Linked Data 

**Authors**: Clarisse Bardiot, Bernard Jacquemin, Jacob Hart, George Bruseker, Antonios Lagarias, Pierre-Carl Langlais, Brandon Farnsworth, Jeanne Fras  

**Link**: [PDF](https://arxiv.org/pdf/2611.14037)  

**Abstract**: Many heritage institutions hold extensive collections of show programs. Too numerous to be cataloged individually, they remain difficult to access and are still largely undigitized. This paper presents a workflow for transforming such documents into structured, interoperable data by combining vision-language models (VLMs), a custom extension of the Linked Art ontology for the performing arts, and different approaches for automatic semantic data annotation. Using the Festival d'Avignon programs (1947--present), preserved at the Biblioth{è}que nationale de France, as a case study, we demonstrate how VLMs achieve more than 98% text transcription accuracy from heterogeneous heritage document images and how constraining large language model generation through a knowledge graph significantly reduces hallucinations in semantic annotation. We argue that this workflow opens new possibilities for large-scale, interoperable performing arts historiography, including the creation of catalogues raisonn{é}s for performing artists, a form of scholarship common in visual arts but absent in performing arts studies. 

---
# LUMIR: Decoupled Multimodal Representation Learning from Heterogeneous Data for Next-POI Recommendation 

**Authors**: Zehui Wang, Wolfram Höpken, Dietmar Jannach  

**Link**: [PDF](https://arxiv.org/pdf/2611.14004)  

**Abstract**: As multimodal tourism-related information collected from heterogeneous data sources has become increasingly available, learning effective representations has attracted attention for improving recommendation performance. Existing approaches often tightly couple multimodal fusion with the architecture and optimization objective of a specific downstream model, which limits the generality and reusability of the learned representations. To address these limitations, we propose LUMIR, a decoupled multimodal representation learning framework based on contrastive learning. Multimodal information collected from heterogeneous data sources is first semantically unified into a shared textual representation space using large language models (LLMs). An interaction-aware fusion mechanism is then designed to model cross-modal interactions, while sequential signals are incorporated as a structural constraint to guide representation learning. To evaluate the learned representations, experiments are conducted on a multimodal point-of-interest (POI) dataset under the next-POI recommendation setting across multiple sequential recommendation backbones. The results show that LUMIR consistently outperforms existing multimodal fusion baselines in both accuracy-oriented and beyond-accuracy metrics, demonstrating that decoupling representation learning from downstream optimization provides an effective paradigm for learning reusable multimodal representations. The code is publicly available at this https URL. 

---
# Data Quality Challenges in Multimodal Tourism Recommender Systems 

**Authors**: Zehui Wang, Wolfram Höpken, Dietmar Jannach  

**Link**: [PDF](https://arxiv.org/pdf/2611.13972)  

**Abstract**: Modern recommender systems increasingly consider diverse data modalities to enhance model performance. However, the quality of the underlying multimodal data (e.g. data from social media platforms) has received limited attention so far. Focusing on tourism recommendation, we examine real-world challenges encountered when using datasets from Yelp and Instagram. Issues include noise from fake reviews, hallucinated content from LLMs, modality contradictions, and demographic bias. We outline future directions such as quality-aware fusion, structured and controlled LLM-based content generation to reduce hallucinations, and user group differentiation to improve the robustness and reliability of multimodal recommendation systems. 

---
# Controlled Knowledge Engineering for Taxonomy Induction with Tool-Grounded LLM Agents and Runtime-Adaptive Policies 

**Authors**: Jonas Limniatis, Bennet Schaarmann, Georg Reuber, Robert Stahlbock  

**Link**: [PDF](https://arxiv.org/pdf/2611.13966)  

**Abstract**: Taxonomies are structured knowledge representations that support semantic search, recommendation, product catalog organization, and retrieval-augmented generation. Their automated construction from a fixed set of candidate entities requires more than plausible text generation: the resulting hierarchy must be complete, structurally valid, and semantically coherent. Large Language Models (LLMs) are promising for this task, but iterative free-form generation remains vulnerable to omitted or hallucinated entities, invalid hierarchical relations, and inconsistent organizational principles. We introduce TIGA (Taxonomy Induction with Tool-Grounded Agents), a knowledge-engineering architecture that formulates taxonomy induction as an iterative decision process over an explicit state and constrained action space. TIGA combines executable taxonomy-editing tools, an memory that records tool calls, rationales, and outcomes, and a runtime-adaptive organizational policy that the agent revises during inference. An optional global reviewer evaluates the constructed taxonomy against the candidate set and the active policy. We evaluate TIGA on four benchmark taxonomies using two LLM backbones and compare it with single-shot and iterative prompting-based baselines. TIGA achieves the strongest structural-quality results in most evaluated configurations, but does not dominate on every precision measure. The gains are accompanied by substantially higher token consumption, API-call volume, and inference time. Ablation results show consistent benefits from the policy and interaction-memory components, whereas the effect of global review depends on the dataset and the available revision budget. These findings characterize tool grounding and runtime policy adaptation as promising but costly mechanisms for improving the controlled construction of structured knowledge representations. 

---
# Auditing LLM Pseudo-Label Distillation for Review Informativeness at E-Commerce Scale 

**Authors**: M. Emin Dogan, Cansu Er, Aysenur Kulunk, M. Furkan Eseoglu, H. Bahadir Sahin  

**Link**: [PDF](https://arxiv.org/pdf/2611.13797)  

**Abstract**: We study hard-label distillation from Gemini 3.1 Flash-Lite to compact encoders on 5.34 million Turkish marketplace reviews. A strict new-SKU temporal test measures teacher fidelity; a separate, teacher-label-blind audit by one company annotator measures editorial agreement. An H3 category prefix slightly improves teacher fidelity on the full new-SKU test. On the 16,000-review chronological audit subset, it instead reduces agreement with both Gemini and the annotator, with a clearer paired difference for CORN than for cross entropy. Thus the context effect depends on the evaluation population, and teacher fidelity alone does not establish editorial alignment. Under the measured offline configurations, a single-L4 student pipeline processes reviews 50.2 times faster than the teacher pipeline. 

---
# Exploiting Feature-invariant Scaling Laws for Efficient A/B Experimentation in Recommendation Systems 

**Authors**: Zhiliang Chen, Sammy Nammari, Jiahao Su, Xiaoqiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2611.13539)  

**Abstract**: Researchers and engineers improve recommendation systems by performing A/B experiments with new model architectures, optimization methods, and data features. However, running full-scale experiments (e.g., full-model training with online data) for every feature change is prohibitively computationally intensive, while the result from a single small-scale experiment is often misleading. To address this challenge, we show that modern recommendation models exhibit feature-invariant scaling: the change of a model's performance follows a predictable scaling-law even when different feature changes are applied to the baseline model. We introduce an efficient method that exploits scaling-laws and uses statistical inference to predict (with confidence intervals) the full-scale effect of an A/B experiment in recommendation systems. This enables practitioners to efficiently filter out neutral or regressive feature changes with only small-scale experimentation. We validate our method on three distinct recommendation model families---Wukong, HSTU, and LLM-based recommendation models---across more than 40 architectural, optimization, and data changes. Our method accurately distinguishes beneficial feature changes from neutral and regressive ones while using only a fraction of the compute required for full-scale evaluation. It also identifies several changes that improve HSTU's state-of-the-art performance on MovieLens-20M. Our results demonstrate that scaling laws can be practically exploited to make compute-efficient decisions for improving industrial-scale recommendation systems. 

---
# DocFS: File Systems as Agentic Reasoning Environments for Complex Multimodal Document Question Answering 

**Authors**: Bingdian Zhang, Zixuan Li, Long Bai, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.12957)  

**Abstract**: VLM-based agents solve complex tasks by iteratively interacting with environments, and their effectiveness depends heavily on how the environment exposes information and provides tools for acting on it. However, constructing an effective environment for complex multimodal document question answering remains challenging: documents usually span many pages, with multimodal contents distributed across hierarchical sections and linked by cross-modal associations, while existing methods organize documents into structured representations that address only the information side of the environment and lack tools aligned with the structure, limiting agents' ability to fully exploit it. We observe that file systems naturally combine structure organization with aligned tools: directories and files encode document structure, while file operations provide rich tools to access file contents. Building on this insight, we propose DocFS, a file-system-based agentic framework that reorganizes raw documents as file systems, where directories mirror the section hierarchy and explicit links connect text with figures, tables, and equations. Based on well-established file system tools such as navigation, search, and reading, agents can iteratively explore the document structure, locate relevant evidence, and understand fine-grained local information. Paired with Qwen3.5-Plus, DocFS achieves state-of-the-art performance on MMLongBench-Doc and FinRAGBench-V, outperforming the strongest baselines by 2.9 and 4.3 percentage points, respectively, and is up to 10.3x faster with up to 10.1x fewer tokens than the strongest baseline during reasoning. 

---
# Point-in-Time Encoders for Earnings Calls 

**Authors**: Alessio Brini  

**Link**: [PDF](https://arxiv.org/pdf/2611.14325)  

**Abstract**: Text encoders turn earnings calls into vectors for search, for language models that answer from retrieved passages, and for forecasting models. A live retriever can use an encoder trained once on all available text, as finance retrievers are, but a backtest needs an encoder that read no text written after the date of each vector. To address this look-ahead bias, from 412,248 calls we train vintages under four pretraining recipes, with yearly cutoffs from 2010 to 2024, and each vintage reads only text dated on or before its cutoff. We then fine-tune each vintage on the question-answer pairs of the calls themselves. The vintages we train on calls alone, with 33.8M parameters, retrieve answers on held-out calls better than the general retrievers we tested in every test quarter. Their vectors also raise by about half the rank correlation between next-day returns and forecasts built from past returns and firm size. When each vintage continues the previous one, a later vintage retrieves better on earlier quarters than the vintage available at the time, and a point-in-time language model that answers from the passages the later vintage retrieves inherits the gain. The 2024 vintage of our families also forecasts earnings surprises better, although the interval of this gain includes zero on calls that no vintage read. It forecasts returns no better, so point-in-time vintages matter most for retrieval and little for forecasts of returns. 

---
# StaRSE: Compact Russian Sentence Embeddings with a Sign-Coded Static Encoder 

**Authors**: Boris Malashenko, Ivan Jarsky, Egor Soldatov, Vladislav Luzhnov, Sviatoslav Stumpf, Vladislav Shmatkov, Viacheslav Shalamov, Andrey Stankevich, Valeria Efimova  

**Link**: [PDF](https://arxiv.org/pdf/2611.14236)  

**Abstract**: Transformer sentence encoders provide strong retrieval quality, but their inference cost becomes a bottleneck when embeddings must be produced on CPUs or refreshed frequently. We introduce StaRSE, a compact Russian static sentence encoder that removes Transformer inference at deployment time: token directions are stored as sign bits, token norms are stored separately, and sentences are encoded by mean pooling over norm-scaled sign directions. StaRSE is initialized from ruBERT-base, fine-tuned on Russian adjacent-sentence and task-shaped pairs, and post-trained with a contrastive stage that directly optimizes the sign-projected token table. On RuMTEB, StaRSE reaches a task mean of $51.16$, $7.98$ points above the strongest included static baseline. The packaged model occupies $11.3$ MiB and has a mean CPU latency of $0.08$ ms per sentence, compared with an estimated $55.7$ MiB 16-bit parameter footprint and $1.43$ ms mean CPU latency for the closest-quality contextual encoder. Ablations examine adjacent-sentence supervision, token norms, and direct sign-projected post-training. 

---
# From Counterfactual Edits to Human Narratives: LLM-Driven Interpretations for Recommender System Explanations 

**Authors**: Vojtech Cahlik, Amir Reza Mohammadi, Rodrigo Alves, Pavel Kordik, Eva Zangerle  

**Link**: [PDF](https://arxiv.org/pdf/2611.13914)  

**Abstract**: Recommender systems increasingly influence what users discover and consume online, yet the decision logic behind their outputs often remains opaque. Counterfactual explanation (CFX) methods provide an effective way to show which changes to a user's interaction history would alter a recommendation. In recommendation settings, however, CFXs often require many edits, producing long, low-level lists that are hard for users to interpret. Moreover, most evaluations emphasize edit counts while largely overlooking whether the resulting CFXs are coherent and actionable for users. We argue that this gap between generating valid CFXs and communicating model behavior stems from the lack of user-centered abstraction and limited qualitative evaluation. To bridge this gap, we investigate large language models as mediators that summarize raw CFX interaction edit lists into concise natural language explanations describing higher-level preference patterns. We optimize these mediators for correctness, informativeness, and linguistic quality through preference optimization, and evaluate the resulting explanations with human-calibrated metrics derived from separate human-labeled datasets. On the MovieLens-1M dataset, across multiple counterfactual generators and language models, the resulting summaries achieve high overall linguistic quality, while the strongest semantic results are obtained with high-success CFX methods such as ACCENT and SPINRec. Direct Preference Optimization improves the semantic metrics in almost all evaluated settings. 

---
# IFRS Compliance Verification: An Open Multi-LLM Benchmark and Content-Conditioned Ensemble Classifier 

**Authors**: Aghoghomena Akasukpe, Li Yang, Xinyao Zhou, Chen Shen, Miguel Vargas Martin  

**Link**: [PDF](https://arxiv.org/pdf/2611.13595)  

**Abstract**: Verifying corporate financial disclosures against the International Financial Reporting Standards (IFRS) is a core auditing task in which experts cross-reference disclosures with hundreds of fine-grained regulatory paragraphs: a laborious, costly, and error-prone process that motivates growing interest in large language models (LLMs) as compliance assistants. Yet single-model accuracy remains insufficient for high-stakes use, and naive ensembling is vulnerable to systematic per-model biases that can drag aggregate accuracy below that of the strongest individual annotator. This gap is addressed with a multi-stage framework, supported by a new benchmark of 19,998 IFRS regulation-disclosure pairs comprising 11,161 unanimous multi-LLM labels and 8,837 human annotations on items where the LLMs disagreed. The framework couples a three-stage domain-adapted sentence embedding pipeline (BERT masked language modeling on SEC EDGAR filings, unsupervised TSDAE and SimCSE adaptation, and supervised contrastive alignment) with a learned ensemble classifier that fuses frozen embeddings and LLM probability features from GPT-4.1, Claude Sonnet 4, and Gemini 2.5 Pro via Tree-structured Parzen Estimator (TPE) optimized gradient-boosted trees. On the disagreement set, the ensemble achieves 81.7% weighted F1, exceeding the strongest single LLM by 10.4 points and majority voting by 14.6 points, and outperforms eleven unsupervised aggregation baselines; the retrieval pipeline attains Recall@15 of 45.3% on unseen regulations. The dataset, model checkpoints, annotation platform, and classifier implementation will be publicly released. 

---
# A Technique to Model Product Information and User Reviews (MPIUR) 

**Authors**: Emna Belguith, Fajrian Yunus, Talel Abdessalem  

**Link**: [PDF](https://arxiv.org/pdf/2611.13361)  

**Abstract**: Modern e-commerce platforms naturally provide three complementary modalities: textual item descriptions, item images, and user reviews. However, existing multimodal recommendation methods often omit reviews or do not integrate all three modalities within a unified framework. In this paper, we propose MPIUR, a multimodal graph-based recommendation model that encodes reviews in heterogeneous graphs, aligns image and text representations, and adapts collaborative--semantic alignment to item interaction density. Experiments on the Amazon Reviews 2023 Musical Instruments and Baby categories show that MPIUR achieves the best global nDCG@10 and Recall@10 among the evaluated methods on both datasets. It also obtains the best nDCG@10 and Recall@10 on sparse items, indicating that jointly exploiting reviews and multimodal item content is particularly useful when interaction histories are limited. 

---
# Activation-Guided Graph Retrieval for Cognitive Memory Reconstruction in Language Model Systems 

**Authors**: Marvin Danig  

**Link**: [PDF](https://arxiv.org/pdf/2611.13127)  

**Abstract**: Retrieval-augmented language model systems generally select context by query-conditioned relevance. We argue that long-lived AI systems require a second operator: activation-conditioned memory availability. We specify an architecture in which request and goal cues seed an authorized memory graph, activation and fan attenuation allocate a bounded traversal frontier, retrieved entities recursively emit cues, and a typed evidence graph is reconstructed before generation. The specification yields five architectural consequences about candidate selection, traversal control, graph regularization, evidence reconstruction, and reinforcement. We derive implementation-level cost estimates and define ablations, failure conditions, and a reproducible evaluation protocol. The central claim is narrowed or falsified if a simpler graph reranker matches full traversal on causal and temporal recovery under equal budgets. This architecture-and-analysis preprint reports no benchmark measurements. If the claim holds, retrieval returns the evidence path that makes a memory available, not only a ranked list. 

---
# INTENT: Out-of-Sample, Out-of-Time, and Cold-Start Evaluation on User Event Sequences 

**Authors**: Mattias Kvist, Erik Lidman Hillbom, Edoardo de Cal, Davis Siemens, Adam Fredriksson, Rickard Brüel-Gabrielsson  

**Link**: [PDF](https://arxiv.org/pdf/2611.13069)  

**Abstract**: Banks now train foundation models on ordered sequences of customer events such as transactions, app sessions, and authorizations. Deployed, such a model predicts forward in time, often for customers and merchants absent from training, yet each is evaluated on private data, so no two can be compared. Public benchmarks for such sequences rarely test a period the model has not seen; recommendation studies usually hold out each user's last events instead. We present INTENT (INTEraction, Novelty, and Transfer evaluation for next-action prediction), a public benchmark that scores a model in seven cells defined by whether the user, the period, and the target item appeared in training. An audit of 59 public dataset families retains five datasets; the most common defect is the lack of a calendar timestamp that orders events. No date lies within the time span of all five datasets, so each is split at its own cutoff, chosen by one rule fixed before any model is evaluated. Summed over the datasets, every cell holds at least 111,769 users. In reference results the cell determines the best model: a transformer leads on seen items after the cutoff but scores zero on new items, a variant that embeds item metadata helps mainly on MicroLens, and repeating the user's last item leads on new items in four of five datasets. This parameter-free control also shows that on three datasets most of the score lost after the cutoff reflects changed behavior rather than the model. The pipeline, the cutoff rule, and the per-cell user counts are publicly available. 

---
# Beyond the Document: Compiled Context and Governed Agents for Any Work Product 

**Authors**: James Rhodes, George Kang  

**Link**: [PDF](https://arxiv.org/pdf/2611.13055)  

**Abstract**: A document preserves the visible result of professional work but often discards the research, calculations, assumptions, evidence, and authorization boundaries behind it. We study a different unit of delivery: a Portable Agent Format (PAF) package that carries a work product together with compiled context, evidence, provenance, and rules for subsequent interaction. To test whether this context remains useful beyond the process that produced it, we evaluate four new packages, created after the hosted and offline readers were fixed: an investment proposal, fund research, financial-statement analysis, and an engineering incident review. Two packages were produced by specialized agents and two through a general-purpose assistant's packaging skill. Across 400 frozen questions and 12,000 evaluated responses, a hosted governed reader succeeds on 96% of questions versus 86% with instructions plus raw-file retrieval (Holm-adjusted p < 10^-7). On the 260 answerable questions per condition, success is 94% versus 79%; the gap varies by package. Source-material chat and instructions alone reach 48% and 74% overall. A separate 0.8-billion-parameter offline reader succeeds on 69% overall, but 54% on answerable questions. Its -1 response rate is 2.9%, combining materially wrong and unauthorized answers, versus 56% with the source and instructions alone. Ablations suggest advantages for the integrated selector and package-adapted writer, without isolating representation from training effects. The same run exposes a failure: offline retrieval resolves none of the 40 historical-record questions in one financial-data package. These findings support treating compiled knowledge as part of the delivered work product, while leaving independent standards interoperability and end-to-end governance conformance unproven. 

---
# IntentTwin: Deconfounding Candidate Support and the Oracle-to-Deployment Gap in Offline Web Behavior Prediction 

**Authors**: Miaobo Hu, Shuhao Hu, Xiaobo Guo, Xin Wang, Bokun Wang, Daren Zha  

**Link**: [PDF](https://arxiv.org/pdf/2611.12918)  

**Abstract**: Offline web-behavior benchmarks frequently insert the demonstrated action into a finite candidate set and then measure reranking. We show that this protocol can create a candidate-support confound: when the target is outside training support while all distractors are sampled from that support, support membership identifies the gold action without modeling behavior. We formalize this effect with support-identifiability rate (SIR), construct support-balanced candidate sets, and evaluate a target-free dual-path proposer that combines support retrieval with open-vocabulary DOM/LM proposals. On filtered OPeRA (4,282/469/992 train/dev/test steps), the original protocol has SIR 1.0000 on target-OOV rows and a support-only rule reaches 1.0000 OOV Exact@1. Support balancing reduces SIR to 0.0000; the strongest conditional reranker obtains 0.7214 overall Exact@1, 0.7988 on IV rows, and 0.5668 on OOV rows. The target-free dual-path system reaches Proposal Recall@16 0.6915 and end-to-end Exact@1 0.5212, compared with oracle conditional Exact@1 0.7450. On Mind2Web, the same audit yields SIR 0.2380 and an oracle-to-proposal gap of 0.2125. A 200-row semantic audit labels 0.7150 of exact-key OOV targets as semantically novel, 0.2450 as locator/string variation, and 0.0400 as ambiguous (Cohen's $\kappa=0.8173$). These measurements show why support composition must be controlled alongside candidate accuracy when offline results are used to reason about deployment. 

---
