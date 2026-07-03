# Bringing Agentic Search to Earth Observation Data Discovery 

**Authors**: Minghan Yu, Youran Sun, Chugang Yi, Yixin Wen, Haizhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.02387)  

**Abstract**: NASA and its data centers hold thousands of geoscience datasets and tools like Worldview, Giovanni, the Science Discovery Engine, and Harmony. Finding the right one is hard even for domain experts. We present an agentic search system, deployed as a public service for the geoscience community, that takes a natural-language research query and returns the matching datasets and tools. We demonstrate that, in the era of large language models, the latent value of knowledge graphs (KGs) can be substantially amplified through agentic search. From the NASA Earth Observation Knowledge Graph (NASA EO-KG) we derive NASA-EO-Bench, an open benchmark of 47k query-dataset pairs (21k task-based queries). A neural scorer fine-tuned on NASA-EO-Bench beats cosine and BM25 baselines. Further combining it with BM25 via score fusion raises both Recall@10 (R@10) and MRR by over 5x. On top of this supervised pipeline, we add a zero-shot agentic reranking stage that, without any additional training, lifts MRR by 28% on a stratified N=200 subset, showing that LLM reasoning is complementary to supervised retrieval. 

---
# Planning over Matrix-Factorization MDPs for Candidate Generation 

**Authors**: Mikhail Trapeznikov, Maksim Utushkin  

**Link**: [PDF](https://arxiv.org/pdf/2607.02115)  

**Abstract**: For a recommender service, we view the customer journey as a chain of item recommendations: a useful item changes the user's state and therefore what should be retrieved next. Standard matrix-factorization retrieval ignores this -- it builds one user vector and returns the top-$K$ items by a static score, treating them as independent. We ask a narrow question: when is it worth planning over the user-state dynamics that fold-in induces? To answer it we propose casting top-$K$ retrieval as an MDP over the implicit-ALS posterior $(A^{-1},u)$, where an action is an item and the transition is a closed-form rank-one fold-in, and the trajectory reward combines a relevance similarity with a posterior-alignment term. Under the same fixed embeddings we compare static retrieval, one-step planning, and horizon-$K$ MCTS across five datasets and two protocols: a per-user leave-last-$n$ split and a stricter global time split. Dynamics-aware planning tends to overcome static retrieval on all datasets under leave-last-$n$, and the gains hold on MovieLens-1M and the VK-LSVD slices under the global time split. A single step of lookahead already captures most of the gain, so the lightweight planning layer turns static top-$K$ scoring into a short decision and improves retrieval over fixed collaborative-filtering embeddings, with no retraining and no change to the representation. These gains depend on measuring relevance with cosine rather than inner-product similarity, which is otherwise entangled with item popularity. 

---
# Evaluating Chunking Strategies for Retrieval-Augmented Generation on Academic Texts 

**Authors**: Valentin J. J. Kreileder, Johannes Reisinger, Andreas Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2607.01852)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems use the question-answering capabilities of Large Language Models (LLMs) to access information outside their parameters. We evaluate if cluster-based semantic chunking improves retrieval and answer quality compared to fixed-size and recursive chunking evaluating on long, structured academic theses using the Retrieval Augmented Generation Assessment (RAGAs) framework. RAGAs based faithfulness shows limited reliability in this setup. Performance on fixed versus document specific questions varied substantially, likely related to the formatting of documents and preprocessing. Under the tested configuration, cluster-based chunking did not outperform simpler strategies. 

---
# IntentTune: Using user demand and personalization to resolve "unknown" query intents for e-commerce search 

**Authors**: Rachith Aiyappa, Ishita Khan, Chester Palen-Michel, Jayanth Yetukuri, Samarth Agrawal, Mehran Elyasi, Shuang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.01530)  

**Abstract**: Understanding user intent is fundamental to delivering relevant search results in e-commerce. However, substantial fraction of real-world queries are under-specified (e.g., "watch" or "shirt"), lacking explicit attributes such as gender or age group. This ambiguity poses a significant challenge for query intent detection models in e-commerce search systems, which must accurately infer latent user intent (e.g., age, gender) to support effective downstream retrieval. We introduce IntentTune, a framework for resolving ambiguous or under-specified query intents by leveraging either (1) user-specific behavioral signals including search history, browsing activity, and profile attributes or (2) population-level demand patterns aggregated across all users. Through experiments on real-world e-commerce data, we first demonstrate that population-level demand patterns alone are insufficient to reliably infer intent in under-specified queries. We then demonstrate that user-specific behavioral signals -- particularly prior search queries -- outperform both population-level statistics and static profile information for inferring gender, age group, product category, and size intent from underspecified queries. 

---
# CoPersona: Collaborative Persona Graphs for Robust LLM Personalization 

**Authors**: Yangtian Zhang, Leyao Wang, Hiren Madhu, Ngoc Bui, Walter Roznyatovskiy, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2607.01485)  

**Abstract**: Real-world LLM personalization is often constrained by sparse and skewed user histories: most users provide only a handful of interactions, while even frequent users' logs capture an incomplete and biased view of their preferences. As a result, weakly observed user attributes are difficult to infer, leading to brittle personalization when test-time requests shift toward under-supported facets. Motivated by this limitation, we present CoPersona, a graph-based collaborative personalization framework that completes sparse user profiles by borrowing signals from behaviorally similar peers. However, directly transferring signals is difficult because uneven facet coverage introduces bias into interaction histories, obscuring user similarity in the unstructured global space. To address this issue, CoPersona decomposes interaction histories into multiple facet-level representations and explicitly models peer-to-peer, facet-level alignment through a multiplex persona graph. To effectively leverage peer information at inference time, we employ a dual-branch architecture that combines non-parametric peer retrieval with parametric graph reasoning. Experiments across multiple domains and model scales demonstrate consistent improvements over strong baselines, validating CoPersona as an effective approach for robust LLM personalization. 

---
# Bi-NAS: Towards Effective and Personalized Explanation for Recommender Systems via Bi-Level Neural Architecture Search 

**Authors**: Longfeng Wu, Yao Zhou, Tong Zeng, Zhimin Peng, Bhanu Pratap Singh Rawat, Lecheng Zheng, Giovanni Seni, Dawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.01387)  

**Abstract**: Recommender systems are vital in helping users navigate vast amounts of information, offering personalized suggestions and effective explanations for these recommendations. While previous efforts have attempted to provide such explanations, evaluating their effectiveness across various scenarios remains a challenge. Enhancing these explanations is essential for improving user engagement, trust, and decision-making. To facilitate effective explanations within the recommender system, we propose a Bi-level Neural Architecture Search (Bi-NAS) framework to optimize explanations. This approach simultaneously refines cross-attention mechanisms and feature interaction functions by exploring both intra-layer and inter-layer design spaces. Furthermore, we integrate Large Language Models (LLMs) to enhance explanation generation, leveraging zero-shot prompting to produce more effective and personalized justifications. By aligning user feature preferences with item quality scores, our approach ensures that explanations reflect both user intent and item attributes, improving transparency and reasoning depth. Extensive evaluations on four real-world datasets demonstrate that Bi-NAS not only boosts recommendation accuracy but also significantly improves the effectiveness of explanations for recommender systems, providing users with clear and reliable insights into the suggestions they receive. 

---
# Retrieval-Augmented Generation to Support Railways Engineering Tasks: A Case Study 

**Authors**: Andrea Gerardo Russo, Federico Ruggeri, Ivan Tomarchio, Davide Bombini, Nicolò Donati, Gianmarco Pappacoda, Paolo Torroni, Giuseppe-Emiliano La Cara  

**Link**: [PDF](https://arxiv.org/pdf/2607.01244)  

**Abstract**: The growing number and complexity of technical regulations represent an important challenge for all professionals in regulated industries. This paper describes a case study, from design to deployment, of building a Retrieval-Augmented Generation system for the consultation of complex technical regulations in the railway domain. Although developed for the railway sector, this testimony of an industrial experience is of particular value for technical domains where regulatory compliance and accurate information retrieval from complex documentation are essential requirements. It also constitutes a human-centered approach for implementing LLM-powered technical documentation consultation across various regulated industries, balancing technological capabilities with domain expertise. 

---
# STRUCTSURVEY: Structured Agentic Retrieval for Automated Survey Paper Generation 

**Authors**: Paolo Pedinotti, Enrico Santus  

**Link**: [PDF](https://arxiv.org/pdf/2607.01243)  

**Abstract**: The rapid growth of scientific publications makes it increasingly difficult to track and synthesize research progress. While Large Language Models (LLMs) can support automated survey generation, existing methods retrieve unstructured data and require models to infer conceptual, methodological, and taxonomic relations from raw text at generation time. We introduce STRUCTSURVEY, a hierarchical multi-agent framework that shifts structural reasoning from generation to retrieval by dynamically constructing graph-based representations of entities, relations, and topical taxonomies. We evaluate STRUCTSURVEY on a new reference-grounded benchmark of ACL survey papers for reproducible long-form scientific summarization. Compared with embedding-only retrieval baselines, STRUCTSURVEY improves ROUGE-1 recall by +2.9 and ROUGE-2 recall by +1.0 on average, without reducing precision. It also improves LLM-as-a-Judge ratings for logical structure, depth, and synthesis, showing that explicit structural retrieval yields surveys closer to human-written organization and reasoning. 

---
# HNSW with Accuracy Guarantees Using Graph Spanners -- A Technical Report 

**Authors**: Minghao Li, Raghav Mittal, Sanjivni Rana, Suraj Shetiya, Gautam Das, Nick Koudas  

**Link**: [PDF](https://arxiv.org/pdf/2607.02338)  

**Abstract**: Hierarchical Navigable Small World (HNSW) graphs serve as the industry standard due to their logarithmic complexity and strong empirical performance. However, HNSW relies on greedy graph traversal, a heuristic that provides no theoretical guarantees of correctness. In this paper, we propose a novel "Certify-then-Rectify" framework that bridges the gap between the speed of heuristic search and the rigor of exact retrieval. Rather than discarding HNSW, our approach first employs a distribution-free statistical certifier to dynamically evaluate the quality of a standard HNSW search with minimal overhead. If certification indicates that the retrieved neighbors are of low quality, the framework safely escalates to a rigorous exact recovery algorithm. To make this exact recovery computationally feasible, we reinterpret the HNSW graph as a geometric spanner and utilize Extreme Value Theory to stochastically estimate its maximum empirical stretch factor. This allows us to mathematically bound the maximum distance of true nearest neighbors. Extensive evaluations on benchmark datasets demonstrate that our tiered framework delivers the average-case speed of HNSW while ensuring the worst-case correctness of exact search and outperforming other applicable approaches. 

---
# Embedding Inference Attack 

**Authors**: Cedric Fitiavana Raelijohn, Sébastien Gambs, Jean-Francois Rajotte  

**Link**: [PDF](https://arxiv.org/pdf/2607.01276)  

**Abstract**: Embedding models are essential components of modern Information Retrieval (IR) systems, yet they are typically hidden behind APIs. Recent works have shown that dense IR system can lead to security vulnerabilities such as embedding inversion attacks. However, such attacks usually require that the attacker knows the embedding model for the attack to be applicable. In this paper, we study IR systems under a black-box setting in which the adversary observes only the unordered set of retrieved documents, without ranking or similarity scores. We demonstrate that in such contexts, tailored queries allow an adversary to identify which embedding model is in use from a set of known model candidate, which we coin as an embedding inference attack (EIA). We also show that certain queries remain discriminative even when the system includes a reranker as a potential defense mechanism. We further validate our method on a real Retrieval-Augmented Generation (RAG) system, in which the tailored queries bypass the LLM's tendency to reject inputs it does not recognize as well-formed questions. Finally, we propose and evaluate other mitigation strategies such as similarity thresholds. 

---
# Office Comprehension Benchmark 

**Authors**: Firoz Shaik, Mateus Picanço Lima Gomes, Tanvir Aumi, Jingci Wang, Milos Milunovic, Filip Basara, Ivana Jovanovic, Vishwas Suryanarayanan, Neha Nandan Kenkare, Weiyao Xie, Zhipeng Han, Zheng Zhang, Waleed Shahid, Jay Rathi, Russell Scherer, Thong Q. Nguyen, Michael Bentley, Tamara Stankovic, Rasika Chakravarthy, Vishal Chowdhary  

**Link**: [PDF](https://arxiv.org/pdf/2607.01245)  

**Abstract**: We introduce Office Comprehension Bench (OCB), the first public benchmark to jointly evaluate LLM systems on Word, Excel, and PowerPoint comprehension over native file formats (.docx, .xlsx, .pptx) and their variants. OCB consists of two tracks. File Fidelity Q&A tests structural and visual perception of office artifacts - tables, charts, embedded images, formulas, and app-specific elements such as headers, speaker notes, and named ranges. Domain Q&A tests expert-level reasoning grounded in real-world industry documents across 12 professional domains, with queries requiring multi-step analysis and synthesis across documents. Each reference answer is decomposed into atomic, binary-gradable claims, and an ensemble of LLM judges scores responses against each claim independently. Even the strongest frontier system in its default reasoning mode reaches only about 59.3% on Domain Q&A; increasing thinking depth within a tier does not move performance materially, while moving to a higher product tier yields modest gains. We release the dataset, evaluation tooling, judge prompt, and a public leaderboard. 

---
# ExPerT: Personalizing LLM Responses to Users' Domain Expertise via Query-Wise Semantic and Keystroke Behavioral Cues 

**Authors**: Yeji Park, Jiwon Tark, Taesik Gong  

**Link**: [PDF](https://arxiv.org/pdf/2607.01242)  

**Abstract**: Large language models (LLMs) are increasingly used by end users, yet existing personalization methods relying on static profiles or text-only signals fail to capture query-specific expertise variation. We present ExPerT, a query-wise personalization framework that adapts LLM responses to users' query domain expertise by combining semantic and behavioral cues. ExPerT consists of two key components: (i) a semantic-behavioral expertise inference module that jointly interprets query text and keystroke dynamics via in-context LLM prompting, and (ii) an expertise-conditioned response generation that adapts the level of detail, terminology, and conceptual complexity. Our user study with 40 participants and 1270 queries demonstrated that ExPerT reduced expertise inference error by 65.7% compared to the strongest baseline (MAE = 0.398 vs. 1.162) and improved response satisfaction by 17.52% (from 3.71 to 4.36) on a 5-point Likert scale. 

---
