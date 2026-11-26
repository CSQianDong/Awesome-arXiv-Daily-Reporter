# Grounding User Preferences in Shared Multimodal Concepts for Recommendation 

**Authors**: Ao Xu, Jinhui Huang, Wenqiang Tang, Tiehua Mei, Hongru Hou, Hengrui Chen, Jiaqing Liang, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34436)  

**Abstract**: Multimodal recommendation has substantially advanced item understanding by incorporating rich visual and textual content. Yet user preferences are still commonly encoded as implicit signals tied to individual items, leaving recurring semantic factors without persistent identities that can be shared across items. We introduce MKGE, a multimodal knowledge graph framework that represents these factors as canonical, addressable concepts. MKGE induces a domain-adaptive concept vocabulary from item images and titles, grounds each user's interaction history in supported concepts, and jointly models direct user-item evidence and concept-mediated preference signals. By turning recurring semantics into shared coordinates, MKGE enables evidence from different items to accumulate around the same preference factors. Extensive experiments on three public benchmarks demonstrate that MKGE consistently outperforms collaborative, knowledge-graph, multimodal, and recent LLM/MLLM-based recommenders across all six reported metrics. Controlled studies further demonstrate the importance of canonical concept identities and personalized user-concept correspondence, with especially pronounced gains over removing or rewiring the user graph for low-activity users. These results establish canonical multimodal concepts as effective units for preference modeling and transfer. Code, constructed knowledge graphs, and prompts are available at this https URL. 

---
# ConTra: A Training Framework for Effective Contextual Retrieval via Prefix Generation 

**Authors**: Jiajun Hou, Jingbang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.34366)  

**Abstract**: Data management systems organize large text collections for tasks such as finding relevant studies and extracting contract clauses. Systems make long documents searchable by splitting them into independently indexed chunks. Chunking can separate text from its identifying details. The indexing challenge is to keep each chunk independently searchable while preserving the context needed to find and rank it. Existing methods either encode that context in chunk vectors or attach it as a text prefix. Context encoded only in vectors is unavailable to text rerankers. Prompted prefixes serve both vector search and reranking, but are costly to generate at indexing time and can underperform contextual encoders. We propose ConTra (Context That Travels), a reinforcement learning framework for training lightweight prefix generators. Its reward combines feedback from frozen embedding and reranking models with factual verification and length control. ConTra generates one prefix per chunk during index construction and stores the augmented text and its embedding for reuse across queries. To evaluate contextual retrieval across workloads and under interference, we introduce RealContextBench, a benchmark with five application collections and three controlled conditions that isolate the effect of distracting text. On ConTEB and RealContextBench, ConTra-2B outperforms the prompted 9B baseline using Anthropic's method and the PPLX-4B contextual encoder, with a margin of up to 3.15 points over their best overall scores, while indexing short documents 7.74 times and 2.15 times as fast as those baselines, respectively. Post-trained only on ConTEB, it leads the standalone methods on RealContextBench after reranking and is less sensitive than the prompted 2B baseline to random and topically similar distractors. Code, datasets, and trained models are available at this https URL. 

---
# RecField: Embedding Recommendations into Spatial Movie Organization as Affordances for Serendipitous Encounters 

**Authors**: Kenta Oku  

**Link**: [PDF](https://arxiv.org/pdf/2611.34330)  

**Abstract**: Research on serendipity-oriented recommender systems has focused on algorithms that recommend unexpected yet useful items. However, serendipity is increasingly viewed as a user experience shaped by both algorithmic outputs and system features. This study positions recommendations not as the primary goal users pursue but as an environmental affordance that creates opportunities for discovery during other activities. We propose RecField, a spatial movie organization interface where users freely arrange movies into collection zones. For each collection, RecField provides attribute-based recommendation lists labeled according to their recommendation perspectives. A between-subjects experiment ($N$ = 57) examined the effects of recommendations on discoveries made during a foreground task of creating and organizing personal movie profiles. Although recommendations did not significantly affect environment-level perceived serendipity, ratings of encounter-related aspects tended to be higher, with small effect sizes. At the individual-movie level, favorites in the recommendation condition received significantly higher ratings for stimulating the desire to watch and broadening preferences, but not for novelty or unexpectedness. Notably, only 14\% of evaluated favorites in the recommendation condition were added directly from recommendation lists. Differences in ratings showed similar effect sizes even when the analysis was restricted to favorites added through other routes. These findings suggest that embedded recommendations can contribute to serendipitous experiences both directly through item acceptance and indirectly by prompting users' own exploration and re-evaluation. They also highlight the value of evaluating serendipity at both the environment and item levels and the limitations of relying solely on direct recommendation acceptance rates. 

---
# MulCLIP: A Multi-level Alignment Framework for Enhancing Fine-grained Long-context CLIP 

**Authors**: Chau Truong, Hieu Ta Quang, Zhenzhen Liu, Dung D. Le  

**Link**: [PDF](https://arxiv.org/pdf/2611.34083)  

**Abstract**: Vision-language models such as CLIP excel at image-text alignment but struggle with long, detailed descriptions due to training on short captions. Recent methods address this limitation using region proposals to align visual regions with sentences, but at substantial deployment cost. We present MulCLIP, an end-to-end multi-level alignment framework that directly exploits natural long-text structure without region proposals. Instead of simply stacking objectives, MulCLIP aligns different textual granularities through compatible training signals: global contrastive alignment for long captions and short summaries, Word-Patch Reconstruction over locally calibrated features for within-sample word-patch semantics, and Subcaption-Aggregated Patch alignment for context-rich subcaption grounding. Experiments on benchmarks spanning varying caption lengths show consistent gains, and ablations confirm that the proposed objectives are complementary and jointly improve the overall framework. Code will be released at this https URL. 

---
# Privacy-Aware Adaptive Client Selection for Federated Retrieval-Augmented Generation 

**Authors**: Ali Shams Saryazd, Abolfazl Gandomi  

**Link**: [PDF](https://arxiv.org/pdf/2611.33890)  

**Abstract**: Federated retrieval-augmented generation retrieves evidence from distributed corpora. However, broadcasting queries to all clients increases the retrieval burden and query exposure. We present Privacy-Aware Adaptive Client Selection for Federated Retrieval-Augmented Generation (PACS-FedRAG), an online selector that takes into account perturbed semantic profiles along with evidence feedback, latency, payload cost, and exploration. In a ten-client setting where three clients are chosen, the archived end-to-end benchmark reports an evidence overlap of 0.372 and an answer accuracy of 0.287. The retrieval-only evaluation, on the other hand, reuses a full-broadcast baseline with a different fusion tie-breaking policy. Rank fusion across disjoint client corpora imposes an overlap ceiling of 0.375, limiting its interpretation as a relevance measure. Answer accuracy is below the fixed-label diagnostic of 0.645, and raw generator responses are unavailable for parser verification. Corrected baselines and a new answer-level evaluation are needed to assess whether quality is preserved. Selecting three clients reduces the number of query recipients by 70%. Profile perturbation does not provide confidentiality of queries or passages. 

---
# OneQuery: Unified Sequence-to-Structure Generation for Multilingual E-commerce Query Understanding 

**Authors**: Mingyu Zhang, Jiejie Qin, Jianhui Ji, Rong Xiao, Meishan Zhang, Xiaoyi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2611.33653)  

**Abstract**: Multilingual e-commerce search requires robust query understanding to convert cross-lingual user queries with varied linguistic patterns into standardized semantic representations that support retrieval and ranking. However, traditional Query Understanding systems suffer from error accumulation, separately optimized subtasks that ignore their mutual correlations, and high maintenance costs. To address these issues, this paper proposes OneQuery, a unified generative framework for multilingual query understanding based on domain-specific large language models in e-commerce. We reformulate four subtasks, namely Language Recognition, Spelling Correction, Translation, and Query Tagging, into a unified Sequenceto-Structure autoregressive generation process constrained by predefined schemas, and design a three-stage progressive training pipeline with multi-task reinforcement learning alignment. OneQuery demonstrates consistent gains over baselines across all tasks and remains effective across different backbone sizes, and online experiments in the AliExpress e-commerce search scenario further confirm its practical value for industrial deployment. 

---
# Structure-Aware Multimodal Retrieval Under Data Scarcity, Privacy Constraints and Out-of-Domain Content 

**Authors**: Argyrios Zafeiriou, Dionysios Koulouris, Ilias Maglogiannis  

**Link**: [PDF](https://arxiv.org/pdf/2611.33649)  

**Abstract**: Purpose: Figure retrieval from reference atlases in expert domains like forensic pathology, dentistry and geology must work without labels or training pairs, on content that often cannot leave local hardware, and on images outside the training data of image-text encoders. This work proposes a structure-aware pipeline that satisfies all three, retrieving figures through the page structure recovered by layout geometry and a local open-weight 8B VLM. Methods: A geometry-based pairing method is introduced, in which the model receives a schematic of the detected boxes and selects each picture's caption without access to pixels or text. Caption retrieval is evaluated on three private corpora (29 books, 10,421 figures, 218 queries) with blind human grading, against a lexical baseline, generated descriptions and five image-text encoders. Results: Geometric pairing was correct for 96%, 92% and 77% of pictures in the three domains, against 76%, 45% and 53% for the better of two document parsers. A hosted model given the schematic was equally accurate, while coordinates as text lost 22 points. Caption retrieval obtained an nDCG@5 of 0.85, 0.87 and 0.76, against 0.54, 0.51 and 0.27 for the best cross-modal encoder and 0.41, 0.28 and 0.30 for generated descriptions. Pairing errors carried over to retrieval. Conclusion: The proposed pipeline achieves accurate figure retrieval with small local models and no in-domain training, where general and biomedical image-text encoders fall far behind. Its geometry-only pairing is the most accurate of five methods and exposes no content, even to a hosted model. Code and prompts are released. 

---
# Multi-Modal Causal Disentanglement for Multi-Domain Sequential Audio Recommendation 

**Authors**: Yu Ren, Pengcheng Li, Weijun Xu, Hengzhong Zhou, Shufeng Ye, Qiang Xiao, Dongjing Wang, Chuanjiang Luo, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.33628)  

**Abstract**: Modern music platforms expose users to heterogeneous audio content domains such as songs, playlists, and podcasts. Multi-domain sequential recommendation (MDSR) seeks to exploit such cross-domain behavior, yet two obstacles persist in practice. First, joint optimization across heterogeneous domains is prone to negative transfer, with high-traffic domains dominating the shared parameter space and suppressing weaker domains. Second, conventional ID-based collaborative signals fail to bridge domains that are semantically related but exhibit little behavioral overlap. This paper presents CAMAR, a multi-modal causal disentanglement framework for multi-domain audio recommendation. From a causal perspective, user behavior is modeled as the joint outcome of a domain-invariant preference and domain-specific preferences. Guided by this view, CAMAR factorizes user representations into domain-invariant and domain-specific components and enforces domain adversarial invariance together with orthogonality, restricting cross-domain interaction to the invariant subspace and suppressing harmful transfer paths. To strengthen cross-domain semantic coupling, CAMAR derives trainable semantic-ID embeddings from pre-trained text, vision, and audio features through residual quantization, yielding an end-to-end multi-modal adaptation tailored to recommendation. A transfer network imputes user representations for inactive domains, mitigating cross-domain cold start. Experiments on both public and industrial datasets demonstrate consistent gains over strong baselines. In online A/B tests on NetEase Cloud Music, one of China's largest music streaming platforms, CAMAR delivers consistent gains across all three domains in their main traffic scenarios, validating its effectiveness in multi-domain audio recommendation at scale. 

---
# Learning the Platform, Not the Learner: Sequencing Shortcuts in the Offline Evaluation of Educational Recommender Systems and LLM Rankers 

**Authors**: Youssef Jdidou  

**Link**: [PDF](https://arxiv.org/pdf/2611.33240)  

**Abstract**: Educational recommenders are usually compared offline, by how well they predict the next activity in logs of tutoring systems, learning applications and MOOCs, where the next activity is largely decided by the platform: teachers assign problem sets, tutors draw problems until mastery, engines select questions. On eight public datasets from five platforms, we compare four structural predictors, which use nothing about the learner beyond the last activity, with ten personalised recommenders and with open-weight LLM rankers from 0.5 to 7 billion parameters, under two protocols. The structural predictors reach between 0.81 and 1.00 of the best accuracy of any method (median 0.95), mostly through population transitions rather than the course order; the best personalised recommender is significantly better on five datasets, by 0.007 to 0.102 NDCG@10, and on Eedi in three of four training runs. Every method is far less accurate on steps along transitions never seen in training and on steps chosen by learners. LLM rankers look competent with random negatives but are never much better than a word-matching baseline, and with negatives that are plausible under the course structure they stay close to random or well behind the transition heuristic, whatever their size. In a simulation with known learner needs, the structural share tracks the share of steps decided by the platform, and offline accuracy ranks an oracle of the learners' needs below imitators of the platform. Weighting test steps by how unlikely the structure makes them restores much of the link with usefulness and, on the real data, reveals a significant personalisation signal on seven of the eight datasets. We propose a syllabus-aware evaluation checklist and release a Python package that reports the lift of any recommender over the structure. 

---
# Procedure Inference as Memory: Learning Recruiter Decision Procedures from OKF-Structured Episodes in Multi-Session Hiring Assistants 

**Authors**: Praveen Kumar Bodigutla, Yizhi Zhang, Sai Vivek Kanaparthy, Arup De  

**Link**: [PDF](https://arxiv.org/pdf/2611.34576)  

**Abstract**: Multi-session hiring assistants must infer evolving recruiter preferences and decision patterns across projects and time. We introduce Open Knowledge Format (OKF)-indexed episodic memory to organize interactions and retrieve related events through their lineage. We also introduce Procedure Memory, which learns reusable decision procedures from these episodes through prompted reflection or reinforcement learning. We compare these approaches with no memory, full-history context, and conversation search, and evaluate Jev-class models for index navigation and GPT-5.6 Luna as a lower-cost alternative to GPT-5.4.
In our evaluation, OKF-indexed retrieval achieves low-priority preference F1 scores of 0.93-0.95 at medium and high preference overlap between hiring projects. It significantly outperforms full-history context and conversation search in both settings, and no memory at high overlap, while using 69% fewer context tokens than full history. Procedure Memory reduces conversational turns by up to 39% relative to no memory. Qualitative and long-horizon evaluations provide evidence of temporal reasoning, with RL-learned procedures achieving 60% on both correct abstention and exact-field identification. Jev-class models improve conversational efficiency but more frequently miss temporally relevant evidence, highlighting the importance of evidence selection.
On PersonaMem-v2, Procedure Memory with OKF retrieval achieves 50.2% accuracy versus 48.6% for full-history context and 29.0% for no memory, using 90.5% fewer reader input tokens than full history. GPT-5.6 Luna performs comparably to GPT-5.4 at lower cost. These findings highlight the value of structured episodic memory and learned decision procedures for efficient personalization and temporal reasoning. 

---
# Agentic Waterfall: An Intent-Driven Information Discovery Framework for Reclaiming User Control in the Attention Economy 

**Authors**: Yige Luo, Shining Zhang, Jikun Liu, Hongjiao Qiang, Ran Guan  

**Link**: [PDF](https://arxiv.org/pdf/2611.34407)  

**Abstract**: We formalize Agentic Waterfall Feed, a new problem of personalized information discovery in which users actively direct recommendations and the system adapts to evolving needs inferred from both explicit requests and browsing behavior. We address this problem with the Agentic Waterfall framework, whose language-model controller maintains persistent user state and coordinates content retrieval, ranking, and state updates. To evaluate systems on this task, we contribute the Agentic Waterfall Dataset and a replay-based evaluation protocol. The dataset combines real content metadata with simulated interaction histories. The protocol compares systems under shared histories using relevance scores and blind judgments of how well recommendations satisfy current user goals and constraints. Agentic Waterfall remains competitive with strong baselines on conventional recommendation tasks across five public benchmarks. On the Agentic Waterfall Dataset, it outperforms the strongest baseline in both evaluations, with relative improvements of up to 16.1%. 

---
# Welfare-Enhancing Mechanism Design for Agent Ecosystems 

**Authors**: Zitian Guo, Chen Xu, Beibei Li, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2611.33627)  

**Abstract**: The emerging Agentic Web forms a multi-agent ecosystem: a decentralized service market where the user agent selects which content agents to query from their submitted profiles, before seeing any documents. Profiles therefore become a new target of optimization. Existing Generative Engine Optimization studies how a single supplier rewrites documents the engine can already retrieve, to improve their citation. In the Agentic Web, multiple content agents optimize profiles to compete for limited opportunities to deliver documents. To study this repeated interaction, we build WildRegistry, a simulation framework and benchmark based on a real agentic-search request stream, and model it as a dynamic game under private monitoring. Long-horizon experiments show that this competition lowers website-selection quality and settles at a harmful stable point, which our empirical criterion identifies as an approximate rewriting equilibrium. We therefore design a welfare-enhancing mechanism, DEC (Delivery-Evidence Credit), a platform-side policy that uses recorded delivery outcomes to reallocate later opportunities and reduce information asymmetry. DEC improves this endpoint, raising website-selection quality while preserving most of providers' exposure gains, for a Welfare Gain of +2.01 points and a healthier long-horizon outcome. More broadly, our findings highlight the importance of mechanism design in agent ecosystems: turning recorded delivery outcomes into evidence that endogenously shapes later user-agent selection and content-agent adaptation can steer long-horizon adaptation toward delivered capabilities. 

---
# Community Notes Across Platforms: Evaluating the Feasibility of Re-using Crowd-sourced Context Notes 

**Authors**: Nishka Arora, Yann Issam Aouidef, Ashish Goel  

**Link**: [PDF](https://arxiv.org/pdf/2611.33546)  

**Abstract**: Crowdsourced fact-checking systems such as Community Notes can provide useful context for misleading content. Producing and rating new notes on such systems requires substantial contributor effort. However, the resulting context is typically confined to the post and platform for which it was created. We ask whether already-vetted notes can be reused when the same claims recur across posts and platforms.
To investigate this question, we construct a human-labelled dataset using Community Notes attached to X posts created during a one-week period and candidate posts retrieved from Bluesky. Within our observation window, 140 of 340 claims had at least one confirmed recurrence on Bluesky, showing that cross-platform note reuse is a substantial opportunity. Recurrence alone, however, does not establish whether reuse would provide timely context. A timing analysis shows that many relevant notes were already Helpful when matching Bluesky posts appeared. Even when a Bluesky post appeared first, reuse could often still provide context sooner after publication than the original X post received its Helpful note. To automate the discovery of these opportunities, we evaluate a three-stage pipeline that searches the Bluesky stream for potential recurrences and determines whether an existing note applies. The pipeline achieves 72.6% precision in a held-out stream evaluation and 70.1% end-to-end recall on our known-positive set. We further characterize the pipeline's costs and principal failure modes. Crossnote, our prototype Bluesky labeler, implements this approach by applying existing Community Notes to matching posts.
Our findings suggest that reusing vetted crowdsourced context is a compelling way to scale the reach and impact of Community Notes. Note reuse can complement the production of new notes by reducing duplicated effort and shortening time to context when misleading claims recur. 

---
# CA-RAG: Constraint-Aligned Retrieval for Structurally Faithful Long-Form Generation 

**Authors**: Vimal Kumar Kumar, Meghana Kshirsagar, Conor Ryan  

**Link**: [PDF](https://arxiv.org/pdf/2611.33497)  

**Abstract**: Retrieval-augmented generation (RAG) typically optimises evidence selection for semantic relevance, whereas compliance-sensitive long-form generation also requires evidence supporting explicit structural obligations. Semantically relevant retrieval may therefore omit information needed for mandatory downstream clauses. We propose Constraint-Aligned RAG (CA-RAG), a retrieval-conditioning framework that augments section-specific queries with ontology-derived structural constraints and policy-grounded knowledge-graph relations before evidence selection. Retrieval-only evaluation with manually annotated evidence shows consistent improvements across retrieval budgets. At $k=6$, CA-RAG increases Precision@$k$ from $0.5268$ to $0.5692$, pooled Recall@$k$ from $0.5402$ to $0.5950$, and F1@$k$ from $0.5389$ to $0.5672$ over standard RAG. Generation experiments further show improvements in compliance and reliability in several model--domain settings, while generation-only constraints achieve higher clause coverage but lower overall compliance and reliability. These results indicate that structurally aligned retrieval can improve evidence sufficiency in compliance-sensitive long-form generation. 

---
