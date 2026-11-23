# Decision Columns as an Index: Interpretable Embeddings for Explainable Entity Matching and Search 

**Authors**: Aadharsh Kannan, Robert Bruckner, Philip A. Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2611.27910)  

**Abstract**: Business data systems must determine whether records refer to the same entity and retrieve records satisfying requests whose decisive properties are implicit. As agents act on these records, both tasks require auditable reasons. Dense embeddings perform well but offer opaque coordinates, cannot be filtered by meaning, and require corpus re-encoding to add new properties. We propose interpretable embeddings built from decision columns: each column stores a decision model's calibrated answer to a natural-language question, with outcomes yes, no, or not stated. A record's embedding is its answers across an ordered set of questions. Similarity decomposes exactly into per-question contributions, and new questions add coordinates without altering existing ones.
On public product benchmarks, question-space retrieval matches dense candidate generation statistically in a retrieve-then-verify entity matcher, while providing exact explanations for accepted matches. Questions derived from retrieval failures triple mean reciprocal rank without re-encoding prior answers. For search, inferring the attributes of a suitable result outperforms encoding only the literal request, and combining question-space and dense scores improves retrieval, though not at our prespecified confidence threshold. A single small neural network predicts the full interpretable vector from an existing dense embedding, enabling corpus backfilling from a few thousand labeled rows at roughly 1/200 the cost. We position interpretable embeddings as a complementary, readable retrieval channel for explaining matches, filtering on attributes, and extending indexes as failures expose missing facts. 

---
# Deep Cross-Division Network for Product-and-Ratio Feature Interactions in Industrial Ranking 

**Authors**: Mikhail Trapeznikov, Sofia Gulevskaia, Maksim Utushkin, Aleksandr Poslavsky, Alexander D'yakonov  

**Link**: [PDF](https://arxiv.org/pdf/2611.27514)  

**Abstract**: Learning effective feature interactions is essential for industrial recommender systems. DCN and DCNv2 efficiently automate polynomial feature crossing. However, their cross layers construct interactions using only non-negative integer powers of the input features, with the maximum degree controlled by network depth. Consequently, they cannot directly represent the reciprocal terms required for ratios, rates, and normalized counts unless such features are explicitly provided as inputs. We introduce DCDN, a Deep Cross-Division Network that augments a DCNv2-style cross layer with a division branch. DCDN jointly learns product, ratio, and mixed product-ratio interactions. We characterize the representable interactions as multivariate Laurent polynomials and show that a $K$-layer DCDN generates Laurent monomials of $\ell_1$-order up to $K+1$, with signed degree between $-K$ and $K+1$. We evaluate DCDN against a production-optimized neural-ranking baseline built around DCNv2 and supplied with manually engineered ratio features. Replacing the DCNv2 block with DCDN allowed us to eliminate the entire class of manually engineered ratio features from the model input, reducing the number of input features by $40.1\%$. DCDN is deployed in the production ranking stack of a major short-video service. In a live A/B test on this service, DCDN increased likes by $3.25\%$, shares by $16.26\%$, total view time (TVT) by $0.33\%$, and clicks to open fullscreen by $0.38\%$; all changes were statistically significant. The number of shown videos decreased slightly ($-0.24\%$). These results demonstrate that DCDN can improve user engagement while eliminating reliance on manually engineered ratio features. Our code is publicly released at this https URL. 

---
# Leader Finding: A New Variation of Expert Finding in Community Question Answering 

**Authors**: Sepideh Entezari Maleki, Mahmood Neshati  

**Link**: [PDF](https://arxiv.org/pdf/2611.26921)  

**Abstract**: Expert finding ranks people by their knowledge of a given topic. Many job positions need more than topical expertise, and a team leader is a clear case. A suitable leader knows the technical areas of the project without being overqualified, has strong managerial knowledge, and communicates well with the team. We study the leader finding problem, where the query is a project and a team and the output is a ranked list of candidate leaders. We model the problem in a probabilistic framework that combines technical fit, managerial knowledge, and communication with the team, and we propose one evaluation metric for each of these qualities. On two large collections built from Stack Overflow, the best configuration of our model more than doubles the NDCG@10 of the strongest baseline. These results show that a ranking of leaders must balance several qualities that topical expertise alone does not capture. 

---
# ERA: Embed, Route, Adapt for Cold-Start Cross-Model Latent Retrieval 

**Authors**: Venkata Chivukula, Zifeng Wang, Jinsung Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2611.26919)  

**Abstract**: Production retrieval systems are tied to the embedding model that indexed their corpus. Moving queries to a different encoder normally requires re-embedding the whole corpus, and at cold start there are no query logs while every document embedded by the new encoder incurs an API or compute cost. We present ERA (Embed, Route, Adapt), a system for cold-start cross-model latent retrieval. ERA spends a small anchor budget on maximally informative documents, selected from the existing target index via Cosine Farthest Point Sampling, trains a lightweight adapter with a Manifold Alignment Loss that couples directional alignment with orthogonal regularization to prevent representation collapse, and pairs the adapter with a deterministic router that escalates the rare, hard queries that define perceived retrieval quality. Across five BEIR datasets, ERA matches or outperforms point-wise L2 and embedding-conversion baselines, with relative gains of up to 35% nDCG@10 on Arguana, and on clustered corpora its anchor selection also improves prior methods under a fixed anchor budget. With 1.2 ms of adapter and router overhead on top of a ~20 ms local encoder, ERA at a 20% nominal escalation budget reduces end-to-end query latency from roughly 160 ms to 38-86 ms while concentrating escalations on the queries where the projection fails most. 

---
# DiTraR: Diverse Scorers and Transcript-Guided Re-ranking for Video Corpus Moment Retrieval 

**Authors**: Mohammad Alijanpour Shalmani, Alale Rezvani Boroujeni, Ali Amini  

**Link**: [PDF](https://arxiv.org/pdf/2611.27931)  

**Abstract**: Video corpus moment retrieval asks a system to find, in a large collection of untrimmed videos with subtitles, the video and the time interval that a natural-language query describes. We find that most top-ranked errors of a strong corpus-level scorer come from choosing the wrong video, not from imprecise boundaries, and that re-weighting the evidence the scorer already uses does not fix them. We present DiTraR, which addresses video selection from two complementary directions. A witness-interval scorer decomposes the query into probes and scores every candidate interval of every corpus video exactly; a multi-scorer energy ensemble averages the energies of independently trained scorers before one exhaustive search; and a transcript verifier, a frozen language model that reads the time-stamped dialogue of the highest-ranked videos, re-orders the moments without changing their boundaries. With all settings frozen on development data and a single evaluation, DiTraR reaches 24.36 and 13.55 R@1 at IoU 0.5 and 0.7 on the TVR validation split. The ensemble improves queries about visual content, the verifier improves queries about dialogue, and their gains largely add up. 

---
# Pre-Crawl Prioritization and Seed Classification for Large-Scale Web Archiving 

**Authors**: Akshith Garapati, Sawood Alam  

**Link**: [PDF](https://arxiv.org/pdf/2611.27743)  

**Abstract**: Pre-crawl prioritization systems decide which pages to fetch first before indexing, aiming to improve crawl efficiency and coverage. Existing approaches based on page popularity and relevance are unsuited to web-archival crawlers that prioritize comprehensive coverage over selective retrieval. We explore prioritization techniques tailored to web archival operations using URL lexical analysis, dynamic domain reputation systems, and HTTP response analysis to inform crawl scheduling decisions. By incorporating predictive models that estimate content change and disappearance probabilities, we improve crawl completeness while reducing redundant visits to stable resources. Through large-scale experiments, we demonstrate that our methods improve archival effectiveness and resource utilization compared to undirected crawling. 

---
# Which Dwarf Is This? Doppeldwarf-306 and Near-Identical Instance Retrieval 

**Authors**: Turhan Can Kargin  

**Link**: [PDF](https://arxiv.org/pdf/2611.27490)  

**Abstract**: Most established instance-retrieval benchmarks emphasise landmarks; small sculptures from visually similar series offer a complementary challenge. We introduce Doppeldwarf-306: 1,691 openly licensed photographs of 306 bronze dwarf statues in Wrocław, Poland, many made in themed series that share one style, with the photographer of each image recorded. With no training, nearest-neighbour search over frozen DINOv2 features names the right statue first for 93.1% of leave-one-out queries, and the best of thirteen backbones from seven pretraining paradigms reaches 94.7%. Accuracy falls slowly as candidates are added: for DINOv2, by about 0.8 points each time their number doubles. A natural design for a phone guide compares a photograph only with statues near the user. In simulation this helps, but less than an equal number of randomly chosen statues does, so random subsampling overstates its value; a permutation test ties the penalty to the real layout (p = 0.005). Used as a weight instead of a filter, location raises DINOv2 from 93.4% to 98.7% for a simulated user 25 m from the statue. With the query's photographer withheld, learned local-feature matching raises DINOv2 accuracy from 81.8% to 85.5% and also improves DINOv3. Comparing the two best candidates cuts DINOv2's false acceptance of unknown statues from 38.3% to 19.6% at 90% known acceptance. 

---
# Explicit Style Extraction for Personalized Long-Form Text Generation 

**Authors**: Jenitha Reddy Patel, Sameeksha Kini, Amulya Rayasam, Anisa Dye, Abhinav Sharma, Sai Karthik Navuluru, Bo Ni, Li Li, Junda Wu, Tong Yu, Nedim Lipka, Xiaolei Huang, Tyler Derr, Yue Zhao, Zhengzhong Tu, Yu Wang, Hongjie Chen, Thien Huu Nguyen, Nesreen K. Ahmed, Franck Dernoncourt, Ryan A. Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2611.27211)  

**Abstract**: Personalized long-form generation asks a language model to write new text the way a specific user would. The standard recipe, retrieval-augmented generation, places a few of the user's past texts in the prompt but leaves the model to decide what to take from them. We introduce StylePrism, a taxonomy of 28 stylistic aspects in seven categories, and SP-Refine, an inference-time method that makes style extraction an explicit step: the model first fills StylePrism with the user's patterns, quoted evidence, and habits to avoid, then drafts from the user's samples and revises the draft twice against this profile, guided by its own style scores. On the three long-form tasks of LongLaMP with three LLMs, SP-Refine outperforms LongLaMP's retrieval-augmented baseline in 25 of 27 ROUGE and METEOR comparisons, by 10.2% on average and by up to 35.8%. Annotating users' own texts with the extracted profiles shows that StylePrism captures concrete, checkable habits, from the order of rhetorical moves to a reviewer's sign-off. Ablations show that the profile helps as revision guidance, not as a replacement for the samples: writing from the profile alone is worse in all 27 comparisons, while a single implicit-inference call is a cheaper alternative that comes close. Oracle variants that let the reviser see the reference, together with reference-free controls, show that larger refinement gains require information unavailable at inference time. 

---
