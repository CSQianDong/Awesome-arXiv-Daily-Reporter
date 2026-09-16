# Efficient Swing Computation for Retrieval in Large-Scale Recommender Systems 

**Authors**: Runhao Jiang, Renchi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.16850)  

**Abstract**: Given a user-item graph $G$, a query item $v_q$ and a target item $v_t$, the Swing score $sw(v_q, v_t)$ of the item pair $(v_q, v_t)$ leverages the user-item-user interaction structure to evaluate their similarity. This measure is found to be highly effective in item-to-item (i2i) retrieval task and finds extensive applications in industrial-scale recommender systems. However, existing solutions towards computing Swing scores are either prohibitively expensive due to their quadratic time complexity w.r.t. the item degree, or rely on truncation heuristics that yield unsatisfactory quality, rendering them impractical particularly on graphs with billions of interactions.
In this paper, we present ASC and $K$-ASC, two novel and efficient algorithms for approximate and top-$K$ Swing queries, to address the aforementioned limitations. Specifically, these algorithms provide rigorous theoretical guarantees in probabilistic relative and additive errors of Swing values. The basic idea of ASC is to combine two randomized algorithms, GNS and USS, in a simple yet non-trivial way to adaptively process high- and low-degree query items with minimal runtime cost. In particular, $K$-ASC offers practical efficiency and effectiveness for top-$K$ queries through a filter-refinement paradigm with carefully-designed heuristics. Extensive experiments over eight real datasets demonstrate that ASC and $K$-ASC can achieve orders of magnitude speed-up over competitors in terms of computational time while offering the same approximate and top-$K$ query result quality, and in particular, $K$-ASC is highly efficient on massive graphs including the billion-edge Yambda and MAG datasets. 

---
# AURA: Agentic Diagnosis and Refinement for Production Recommender Systems at Scale 

**Authors**: SungGeun Kim, Abhinav Narain, Daniel Nemirovsky  

**Link**: [PDF](https://arxiv.org/pdf/2609.16625)  

**Abstract**: How and why does a recommender system fail the users it serves? Oftentimes, practitioners are left to improve their algorithms based on a combination of feedback from stakeholder teams, domain expertise, and insights from data analyses. Yet the nuances of how and where recommendations perform well or poorly for end users are difficult to discern from aggregate quantitative metrics. Whereas these metrics provide a high-level and incomplete picture, further granularity into the quality of recommendations and their patterns requires reasoning with domain understanding and objectivity, at scale. We contemplate this complex conundrum and describe a method and implementation that uses the latest AI agentic advances to provide actionable diagnoses and improvements for production recommender systems. We present AURA (Agentic Understanding and Refinement of recommender Algorithms), an end-to-end agentic system that performs qualitative evaluation at scale and can then generate improvements to our algorithms at the code level. Specialized agents read production engagement logs, from thousands of sessions to millions, and surface patterns and examples of how the recommender fails real users. The next step uses those diagnoses and context about the recommender's own code, data, and training pipeline to propose and implement refinements grounded in that codebase. We report the system design, initial tests on production data from two large consumer platforms at a major media-streaming company, safeguards, operational learnings, and early results toward a self-improving recommender system. Finally, the diagnostic gap AURA closes is not specific to streaming. The architecture is built to transfer: every domain-specific element enters through the configuration layer that already ported it between our two platforms. We map it concretely to e-commerce and online-retail recommendation. 

---
# Measuring Decision-Scale Use in Tool-Augmented LLMs: A Contrastive Urban Benchmark 

**Authors**: Ray Chen, Vivian Wong, Christan Grant  

**Link**: [PDF](https://arxiv.org/pdf/2609.16607)  

**Abstract**: Urban decision-support often asks whether activity is unusually high or low for a specific place, not which place has the larger raw count. Twenty pickups in a quiet neighborhood can be more abnormal than 180 at an airport. We introduce URBANCONTRASTIVEQA, a benchmark that asks whether tool-augmented language models can make this baseline-relative comparison. Each item pairs two urban situations from public mobility data in NYC, Chicago, and Seattle, labeled by how far current activity deviates from that place's historical baseline. We evaluate six instruction-tuned models under five tool-output formats. With only raw counts, models often pick the larger number even when it is less abnormal for its zone. Server-computed baseline scores and ordinal labels raise accuracy, but gains vary by model. For heterogeneous urban feeds, tool interfaces need to expose local baselines, not just activity volumes. We release the pair bank, labels, scoring scripts, and data card. 

---
# ReliGRec: Reliability-Oriented LLM-Based Generative Recommendation via User-Risk-Aware Prompt Routing 

**Authors**: Haoran Yang, Fei Chen, Yutian Xiao, Jiahao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2609.16560)  

**Abstract**: User behavior in real-world recommender systems is heterogeneous. While some users exhibit coherent preferences, others show abrupt interest shifts, bursty interactions, excessive repetition, or inconsistency with collaborative neighborhoods. Such deviations may arise from benign variation or manipulation, including shilling attacks, but do not alone establish malicious intent. Existing robust recommenders exploit user-risk signals through training-time reweighting or graph aggregation, whereas adapting generation to estimated user-level weak risk remains underexplored in LLM-based generative recommendation. We propose ReliGRec (Reliability-oriented Generative Recommendation), a weakly supervised framework whose name denotes its design goal rather than a supervised reliability variable. ReliGRec derives user-level weak-risk proxy labels from review-feedback signals for a subset of users and represents sequential behavior and collaborative context using a Behavior Token and temporal Graph Tokens, respectively. A Dual-View Weak-Risk Estimator fuses the representations to produce a user-level weak-risk score that selects a Simple or Cautious Prompt at inference. The Cautious Prompt is designed to encourage attention to stable, collaboratively supported evidence while reducing overreliance on isolated, short-term, or repeated interactions. The Behavior Token affects generation through weak-risk estimation and routing, whereas the aggregated Graph Token provides collaborative context for next-item Semantic ID generation. ReliGRec thus turns weak-risk estimation from an auxiliary prediction into a generation-time control signal. Experiments report competitive recommendation and weak-risk proxy-label prediction, while routing analyses characterize the recommendation-quality and inference-cost behavior of weak-risk-guided prompting. 

---
# Predicting Partial Answer Quality and Utility in Agentic Retrieval-Augmented Generation 

**Authors**: Fangzheng Tian, Debasis Ganguly, Craig Macdonald  

**Link**: [PDF](https://arxiv.org/pdf/2609.16453)  

**Abstract**: Agentic Retrieval-Augmented Generation (RAG) has become a promising paradigm for multi-hop question answering, where a reasoning model iteratively issues queries to a retriever and incorporates newly retrieved context into subsequent reasoning steps. While this iterative process can improve final answer quality, current evaluations of agentic RAG largely focus on end-to-end outcomes and provide limited visibility into how a model's answer state changes during generation. In this work, we introduce an in-trajectory probing framework to study intermediate answer states in agentic RAG. Specifically, after each retrieval-reasoning iteration, we force an agentic model to stop reasoning and generate an intermediate answer based on its current state. This allows us to define two iteration-level measures: partial answer quality at each iteration, and partial utility as the change in partial answer quality across iterations. Our analysis across multi-hop QA benchmarks reveals that partial answer quality often plateaus before natural termination, with many later iterations contributing only small measurable improvements. Accordingly, we formulate two prediction tasks, partial answer quality prediction and partial utility prediction, and study trajectory-derived signals from intra-iteration, inter-iteration, and query-iteration perspectives. Experiments show that partial answer quality is more predictable than partial utility, with supervised models achieving Pearson's r above 0.43 for quality prediction. Finally, using predicted answer quality and utility for early stopping reduces average iteration count by about 11% while preserving about 98% of the final answer quality achieved by natural stopping. 

---
# PCap: Personalized Retrieval-Stage Diversity Capping in Facebook Marketplace 

**Authors**: Guangchao Yuan, Janis Fuh, Christopher Choate, Xun Tang, Wenqi Zhu, Chengyi Zhang, Pavan Kumar Paalya Chandrashekar, Jiang Han, Jiangyuan Li, Hongyan Wang, Shuting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.16452)  

**Abstract**: We propose a personalized capping framework (PCap) to improve the diversity in Facebook Marketplace by introducing user-level diversity constraints at the retrieval stage. PCap models individual diversity preferences using Shannon entropy-based scoring, segments users into diversity buckets, and applies personalized category caps during multi-source candidate retrieval. To navigate the high-dimensional parameter space of per-bucket caps, we leverage an automated online optimization method called Parameter Tuning Sequence. Large-scale online experiments demonstrate that PCap significantly improves users' browsing experience shown in engagement metrics. This work provides practical insights into integrating personalized diversity into industrial retrieval systems. 

---
# Balancing Trial and Reorder: A Hybrid Sequential Transformer-GBDT Ranker for On-Demand Delivery 

**Authors**: Marcel Kurovski, Attila Nagy, Steffen Klempau, Aleksandr Fedintsev  

**Link**: [PDF](https://arxiv.org/pdf/2609.16407)  

**Abstract**: On a delivery platform, personalized store ranking greatly influences what users find and order. Unlike digital-only domains, candidate stores are local and bound by real-time availability and delivery operations. One central modeling tension is between surfacing new stores for trial and preserving ranking quality for sessions with reorder intent. We present Universal Venue Ranker (UVR), a production system deployed at Wolt that pairs a bidirectional transformer encoder for sequential user modeling with a GBDT ranker integrating contextual, user, and store features. Trained across all stores and domains of a country while enforcing local delivery constraints at inference, UVR replaces four previously separate ranking models (three for restaurants, one for retail) with a single unified system. Label smoothing and trial-biased sample weighting steer the model toward new stores, lifting offline trial MRR by +12% to +30% over production while regressing reorder MRR in five of six countries. These regressions leave Global CVR, our core online metric, which blends trial and reorder sessions, statistically unchanged. We validate UVR in three consecutive A/B tests, the first two across Wolt's largest operating markets and the third spanning all operating countries and both domains. UVR V1 delivers +5.5% Merchant Trial Rate and +0.16% Global CVR over the previous production ranker; V2 adds a further +0.45% Merchant Trial Rate on top; and V3, our cross-domain unification of the restaurant and retail rankers, adds a further +1.31% Retail Merchant Trial Rate, together accounting for substantial incremental gross order value and a materially simplified serving stack. 

---
# Where Post-Training Quantization Breaks Text Embedders: A Measured Map Across Four Embedder Families 

**Authors**: Hyojung Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.16391)  

**Abstract**: Weight-only post-training quantization is the cheapest way to shrink a retrieval embedder, and the received advice for applying it -- protect the embedding table, allocate bits by module sensitivity, prefer a ranking-aware objective over weight reconstruction -- was carried into LLM quantization largely intact. We test that advice on retrieval embedders directly, quantizing five checkpoints from four architecture families across a grid of bit widths and group sizes, and isolating the embedding, attention and feed-forward blocks at each width.
Every heuristic fails to transfer as stated. The embedding table never emerges as the dominant isolated protection priority in any family, despite being the largest tensor in several of them. Module sensitivity does not survive as a transferable ordering: at INT4/g16 the spread between modules is too small to allocate against, at INT3 the ordering becomes family-dependent and joint damage stops being the sum of its parts, and at INT2 comparable reconstruction error accompanies retention ranging from 1.3 to 65.9 percent of full precision. A cheap reconstruction proxy is useful for screening uniform bit widths but substantially less reliable for choosing which tensors to protect; its apparent strength across the whole grid is a range-extension artifact.
A distilled 109M student at INT3 holds 78.04 NDCG@10 in 68.4 MB and dominates the extreme-PTQ arm of its own 0.6B teacher, 297.9 MB at 64.46, on both size and quality -- but only inside the task it was distilled for. Sizes are byte counts of files that exist rather than arithmetic estimates, and the measurement repository carries the byte provenance for every one of them. 

---
# Evaluating Brand Retrieval and Ranking in Large Language Model Recommendations 

**Authors**: Edward Malthouse, Kun-Yu Lee, Jing Yang, Sanchary Pal, Xueyan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2609.16304)  

**Abstract**: Large language models (LLMs) are increasingly used for product recommendation, but evaluating their recommendations presents challenges that differ from conventional information retrieval and recommender systems. LLMs can generate recommendations without an explicit candidate set, and repeated responses to the same query can produce different brands and rankings. We introduce a framework for evaluating open-ended LLM brand recommendations that defines the competitive set independently of model outputs and estimates recommendation prevalence and prominence through repeated sampling. We operationalize these constructs using Brand Recommendation Probability (BRP@$k$) and Mean Reciprocal Rank (MRR@$k$), and apply the framework to six LLMs across five product categories. Category-only queries reveal substantial omission of established brands and limited evidence that recommendation prominence follows conventional brand popularity. Instead, prominence is associated with broader marketplace-visibility signals, particularly search interest and online brand conversation. Needs-based queries show that contextualizing users' goals and constraints changes which brands are retrieved, while diagnostic positioning probes demonstrate that brands omitted from ordinary recommendations can remain conditionally retrievable when distinctive cues are supplied. These findings highlight the need to evaluate LLM recommendation as a stochastic retrieval-and-ranking process rather than from individual generated lists. We provide open-source software and data to support reproducible evaluation of LLM-generated brand recommendations. 

---
# Lexplorer: Navigating the Complexity of Legal Document Landscapes 

**Authors**: Daniel Fürst, Titus Pünder, Maximilian T. Fischer, Corinna Coupette  

**Link**: [PDF](https://arxiv.org/pdf/2609.17366)  

**Abstract**: As technological and social innovations create novel regulatory challenges, legal systems grow in complexity - increasing the need for interfaces that enable effective interactions with legal document collections. Through interviews with legal scholars (n=15), we find that supporting legal work requires going beyond retrieval-centered legal-information-system paradigms. Hence, we propose Lexplorer, a flexible interface for exploring, navigating, and analyzing legal documents, based on a taxonomy capturing user intents. Distinguishing text and data views for one, few, and many documents, Lexplorer enables context-sensitive interactions with evolving collections of interconnected legal texts, facilitating Adaptive Meaning Construction in law. We evaluate Lexplorer with legal scholars (n=20) in the context of European Union law, validating our elicited requirements, intent taxonomy, and prototype design. Resulting from a close collaboration between visual-analytics researchers and legal scholars, our work also provides nuanced insights into the process required to design interactive systems for expert domains driven by implicit methodological knowledge. 

---
# Diagnosing the Fact-Grounding Gap in Multi-Hop Question Answering 

**Authors**: Kevin Mo, Nathan Mo, Richard Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2609.17043)  

**Abstract**: Multi-hop question answering requires combining information from multiple documents to answer complex questions. These systems have grown increasingly capable, yet when they fail, the error is typically attributed to not finding the right documents. Whether this holds at the level of individual reasoning steps remains largely unexamined. We investigate this across three standard multi-hop QA benchmarks and find that failures decompose into two distinct modes: retrieval failures, where the needed passage was not retrieved, and extraction failures, where the passage was retrieved but the needed fact could not be extracted - a phenomenon we term the fact-grounding gap. Extraction failures account for nearly half of all per-hop deficiencies and are invisible to standard retrieval metrics. They remain unresolved by every retrieval intervention we test, establishing a ceiling for retrieval-only improvements. The gap's severity varies across benchmarks and question types, but extraction failures appear on every dataset we measure. Our findings reveal that retrieval failures and extraction failures are fundamentally different bottlenecks requiring different solutions - a distinction absent from current evaluation practice. 

---
# RegRet: Enhancing Region-Level Retrieval in Large Multimodal Models 

**Authors**: Xun Liang, Honghui Yang, Weihang Pan, Ruisi Zhao, Boyuan Pan, Yao Hu, Wenxiao Wang, Binbin Lin, Deng Cai  

**Link**: [PDF](https://arxiv.org/pdf/2609.16847)  

**Abstract**: Region-level retrieval aims to align user-specified image regions with relevant regions or textual descriptions, playing a crucial role in realworld applications such as e-commerce product search and RAG. Although recent Large Multimodal Models (LMMs) have made significant strides in multimodal retrieval, they primarily focus on global-level tasks and struggle to capture effective region-level representations. To bridge this gap, we present RegRet, an LMM-based Region-level Retrieval framework that enhances the regional representations without compromising overall global retrieval performance. At its core, RegRet integrates a Region-Aware Encoder to capture detailed regional features while balancing them with the global background context. To further enhance the fine-grained understanding and discriminability of representations, we design a multi-stage training pipeline that includes detailed localized captioning and regional contrastive learning tasks. In addition, considering the absence of region-level contrastive training data and the limited diversity of evaluation tasks in current benchmarks, we introduce the REGMB benchmark. It comprises 225k contrastive pairs, covering four multimodal retrieval tasks. Extensive experiments validate the effectiveness of our approach. RegRet outperforms strong baselines in the zero-shot setting. Further training with contrastive learning leads to an average improvement of more than 20\% on both REGMB and public benchmarks, while achieving comparable or better results on global-level retrieval tasks. 

---
# Can We Do Interpretable NLI with Graphs Based on Atomic Propositions? 

**Authors**: Younes Boufouss, Luc Pommeret, Thomas Gerald, Patrick Paroubek, Sophie Rosset  

**Link**: [PDF](https://arxiv.org/pdf/2609.16814)  

**Abstract**: While Large Language Model (LLM)-based Natural Language Inference (NLI) systems achieve high accuracy, their decision-making processes lack auditable structures. This paper explores whether NLI can be performed using only interpretable, graph-based representations of evidence. We introduce a fully graph-based pipeline where the classifier never directly processes the input text. Instead, sentences are decomposed into atomic propositions, converted into ConceptNet triples via constrained decoding, and represented as three graphs per pair: premise, hypothesis, and a retrieved ConceptNet subgraph. These graphs are then fed into a fine-tuned 0.8-billion-parameter language model. On the SNLI dataset, our pipeline achieves 89.7% accuracy, just 1.9 points below an identically trained text-based model. On ANLI, it matches the published performance of RoBERTa-large on rounds R2 and R3 (50% accuracy) but trails by 16 points on R1, resulting in an overall gap of 9 to 14 points compared to its text counterpart. We term this gap the price of interpretability and demonstrate that it stems from representational limitations rather than data constraints. Ablation studies further reveal that graphs and text are complementary: combining both modalities achieves 92.1% accuracy on SNLI. 

---
# LSREP: A Longitudinal State-Replay Protocol for Evaluating Conversational Memory, with ICE v2 as an Audited Local-First Architecture 

**Authors**: Deepesh Sonar  

**Link**: [PDF](https://arxiv.org/pdf/2609.16730)  

**Abstract**: Conversational memory changes during use, so endpoint question answering alone cannot establish how a persistent state accumulates, ages, or incorporates revisions. We introduce LSREP, a Longitudinal State-Replay Evaluation Protocol combining ordered replay, explicit lifecycle schedules, repeated probes, evolving reference answers, and mechanism-fidelity checks. Its architectural case study is ICE v2, a local-first memory middleware with typed stores, retrieval fusion, and dynamic context budgets. The private, single-user instantiation contains 1,985 turns, 219 distinct probes, and 1,211 probe-checkpoint observations across 52 checkpoints. On three ordinary-density datasets, ICE v2 has a near-zero mean quality difference from vector-RAG while selecting 32% fewer fragments but using 6.6% more estimated prompt tokens. A fourth, dense dataset exposes catastrophic failures of the unbudgeted baseline. The fidelity audit limits attribution: procedural retrieval is defective, several mechanisms are unexercised, and graph utility is not established. In a complementary matched public diagnostic, ICE v2 loses decisively to pure vector-RAG on LongMemEval: 50.8% versus 72.8% in the evidence-only oracle and 43.0% versus 69.5% in full-S. Paired differences are -22.0 points (95% CI [-26.6, -17.4]) and -26.5 ([-31.3, -21.8]). Conservative abstention accompanies severe multi-session and temporal failures. ICE uses less context in this diagnostic, establishing a quality-cost trade-off rather than superior efficiency. Together, replay, fidelity auditing, and public endpoint testing expose distinct failure modes that neither architectural descriptions nor aggregate scores identify alone. 

---
# Quantifying Organizational Environmental Action from Web Data and Large Language Models 

**Authors**: Quinn Reynolds, Daniel Shore, Vianey Leos Barajas, Tanhum Yoreh, Meredith Franklin  

**Link**: [PDF](https://arxiv.org/pdf/2609.16627)  

**Abstract**: Quantifying organizational environmental action from publicly available web content remains a challenging environmental data science problem because relevant information can be dispersed across multiple webpages and is primarily communicated through unstructured text. We present a scalable computational framework for transforming organizational web content into structured measures of environmental action and demonstrate the approach using Jewish congregations in the United States. We constructed a national database of 4,964 congregations by integrating multiple geospatial, knowledge-base, directory, and manually reviewed sources. Of these, 2,657 had active websites that were successfully crawled, producing a corpus of 154,454 webpages. We compared three approaches for detecting environmental actions: keyword retrieval followed by large language model (LLM) classification, semantic vector retrieval followed by LLM classification, and direct LLM classification classification without preliminary retrieval. Agreement with an expert human reviewer was lowest for keyword retrieval ($\kappa$ = 0.26), higher for semantic vector retrieval ($\kappa$ = 0.42), and similar for direct LLM classification ($\kappa$ = 0.40). Although semantic retrieval achieved the highest agreement, its retrieval recall was 0.87, indicating loss of relevant content before classification. Applied to the complete corpus, direct LLM classification identified at least one environmental action at 1,398 congregations (53%), providing greater coverage than either retrieval-based approach. These results demonstrate that preliminary retrieval can reduce computational cost but may exclude relevant information before it reaches the classifier. The framework provides a reproducible approach for extracting organization-level environmental information from unstructured web content that can be adapted to other institutions. 

---
