# PinEqualizer: Full Funnel Content Exploration and Debiasing System at Pinterest 

**Authors**: Olafur Gudmundsson, Bo Zhao, Huayi Liao, Anna Kiyantseva, Sai Xiao, Heath Vinicombe, Mostafa Keikha, Luke DeLuccia, Zihao Chen, Junpeng Hou, Weijie Jiang, Bhawna Juneja, Andreanne Lemay, Wei-Ting Lin, Keyvan Moghadam, Jiaxing Qu, Zhiqing Rao, Zhihua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.22518)  

**Abstract**: In this paper, we propose a new solution for addressing the content cold-start problem in industry-scale search and recommender systems. Compared to prior approaches, we have made the following new contributions: 1) our solution spans the entire multi-stage funnel and generalizes well for both search and recommendation surfaces, 2) our solution reduces bias favoring existing content, allowing more accurate model prediction across content types and reducing short-term tradeoffs associated with high volumes of explicit content exploration, 3) our solution is evaluated with a scalable measurement framework that enables fast short-term experimentation while validating long-term impact. We have iteratively built and successfully deployed this new system at Pinterest in the past two years and observed significant improvements in fresh content exploration, overall user engagement, and content ecosystem health. 

---
# Legal Nugget Extraction for Granular Retrieval over Long Jurisprudential Texts 

**Authors**: Lucas Pereira, Erick Brito, Roberto Lotufo, Jayr Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2607.22479)  

**Abstract**: Legal retrieval over jurisprudential collections is challenging because court decisions are long, heterogeneous documents whose relevant legal thesis may occupy only a small portion of the text. This paper asks whether legal nuggets, defined as short and self-contained legal theses extracted from source documents, can improve dense retrieval over Brazilian legal collections. We propose a pipeline that extracts nuggets from each document, indexes them with embeddings, retrieves nugget-level evidence, and aggregates the retrieved nuggets back to document-level rankings. We evaluate this approach on four Portuguese legal retrieval benchmarks from the JUA ecosystem, reporting NDCG@10, MAP@10, and MRR@10. Nugget retrieval substantially improves the two jurisprudential datasets: on JUA-Juris, NDCG@10 increases from 0.10265 to 0.20461, and on JurisTCU from 0.20898 to 0.32696. However, it underperforms full-document retrieval on NormasTCU and BR-TaxQA, and an embedding-model ablation shows that strong domain-adapted retrievers can remain better in the full-document setting. The results demonstrate that legal nuggets can be useful for jurisprudence search, especially when queries are formulated as legal theses, but they may not transfer equally well to other legal retrieval scenarios. 

---
# The Prompt Is Not the Query: How Request State Evolves Across Multi-Turn AI Conversations 

**Authors**: Benjamin Tannenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2607.22392)  

**Abstract**: AI-search evaluation commonly treats a prompt as a stable query that can be counted, classified, and replayed in isolation. A conversation makes that unit of analysis questionable: each user turn can add a constraint, revise an assumption, request evidence, or refer to alternatives established earlier. We replace latent "intent" with an observable construct, conversation-conditioned request state, and measure how that state is distributed across user turns. The analysis reuses frozen rules and the governed cohort of a preceding conversation study: 670 English commercial multi-turn conversations in a discovery-replication design and 7,463 public PRISM conversations from 1,389 participants. In the commercial corpus, the final prompt contains a median 35.6% of the session's unique user-side content vocabulary; in PRISM, the median is 36.4%. The final prompt contains at most half of that vocabulary in 68.4% and 74.3% of conversations, respectively. More importantly, transparent rules detect at least one request-state dimension in history but not in the final prompt in 50.3% of commercial conversations and 44.8% of PRISM conversations. Among dimension-bearing conversations, the final prompt reproduces the full observed dimension set in only 26.1% and 26.2%. At the same time, the final prompt adds a previously unseen dimension in 17.9% and 19.3%, showing that the endpoint is neither a summary nor merely a reference: it is often another state update. Length-matched nulls show that low lexical coverage is largely a consequence of turn length, so vocabulary results are interpreted as information availability, not semantic drift. The categorical results support session-level measurement for AI search. They do not estimate the causal effect of history on model answers. 

---
# Bringing GRACE to Recommendation: Fine-Tuning for Sustainable and Accurate Personalization 

**Authors**: Yibowen Zhao, Yinan Zhang, Ning Liu, Lizhen Cui, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2607.22341)  

**Abstract**: Growing concern about environmental sustainability (e.g., reducing carbon emissions and resource use) and public health has motivated ``green'' recommender systems that steer users toward more eco-friendly and healthier choices. However, many existing green recommendation approaches require training new models from scratch, incurring substantial computational and energy costs. Reranking-based methods, meanwhile, introduce an additional sorting stage at inference, increasing latency and computational cost. In this work, we propose GRACE (Green Recommendation via Adaptive Conflict-rEsolution), a fine-tuning framework that integrates item-level sustainability signals (e.g., eco-scores or health indices) into pretrained recommendation models. Since these green values are usually discrete and non-differentiable, existing methods often rely on pairwise comparisons to promote greener items. GRACE instead introduces a differentiable approximation that enables direct optimization of the green criterion. To balance sustainability and personalization quality, GRACE further employs a gradient projection mechanism to mitigate conflicts between the green objective and the accuracy objective during fine-tuning. Experiments on real-world datasets demonstrate that GRACE improves sustainability-oriented recommendation outcomes while generally preserving recommendation accuracy through a controllable preference-anchored update mechanism. 

---
# LAMAR: An Open Language-Aware Multilingual Alignment Reranker 

**Authors**: Seongtae Hong, Youngjoon Jang, Jungseob Lee, Seungyoon Lee, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2607.22042)  

**Abstract**: In multilingual retrieval augmented generation, a retriever can retrieve relevant documents written in multiple languages, which are subsequently reranked before answer generation. However, it remains unclear whether existing multilingual rerankers consider document language when ordering semantically relevant candidates. Our analysis shows that these rerankers do not consistently prioritize documents written in the same language as the query when semantically equivalent documents are available across languages, even though document language can affect answer generation. We release LAMAR, a language aware multilingual cross encoder trained to account for both semantic relevance and language coherence. LAMAR first uses English anchored relevance distillation to establish consistent relevance scoring across multilingual inputs and then applies preference alignment for language coherence to encourage documents written in the same language as the query to receive higher rankings while retaining semantic relevance. In a controlled experiment designed to assess language coherence, LAMAR achieves the best performance overall and across all languages examined individually. LAMAR also remains competitive on established multilingual reranking benchmarks. In practical retrieval settings, LAMAR achieves the best results across all reported metrics when reranking candidates retrieved in the first stage. These results demonstrate that LAMAR accounts for language coherence while achieving strong performance on general multilingual reranking benchmarks. 

---
# SIREN (Luring LLMs onto the Rocks): PAIR-Driven Preference Manipulation in Web-RAG Recommenders 

**Authors**: Evan Caville, Siamak Layeghy, Billy Sung, Sara Dolnicar, Marius Portmann  

**Link**: [PDF](https://arxiv.org/pdf/2607.21951)  

**Abstract**: This paper investigates the adversarial manipulation of the ranked recommendations produced by web-augmented large language models (LLMs). When an LLM answers a recommendation query by retrieving and reading live webpages, it acts as a recommender, and each retrieved page becomes a potential attack surface. Prior work has examined fabricated products, retrieval poisoning, and rank promotion. However, these studies do not compare how different edits to an already retrieved page change the model's final ranking while the surrounding source set remains unchanged. To address this gap, we propose SIREN, an automated attacker--judge method that adapts the PAIR jailbreaking loop to competitive rank manipulation, with the goal of moving a chosen entity to rank~1 in an LLM-generated recommendation. SIREN retrieves and captures webpages using Anthropic's web tools, then iteratively edits a retrieved source using an interpretable taxonomy of 23 content-poisoning techniques. The custom-RAG replay platform keeps the same sources in the same order, so changes in the model's ranking can be linked to changes in the supplied content rather than to differences in retrieval. Across two production Claude models, SIREN reaches rank~1 in 62 of 124 technique trials nested within eight query--model contexts. The payloads that reached rank~1 were then tested in fresh sessions, where they reproduced the result with a mean success rate of 0.805. Across the evaluated settings, declarative ranking claims and seeded lists were generally more effective than directive-form injections, although the strength of this difference depended on the target model. To the best of our knowledge, this is among the first controlled studies of competitive rank manipulation in production LLMs where the supplied source context is kept fixed. 

---
# Reflector: Arrangement-Aware Harmonic Retrieval for Sample-Based Composition 

**Authors**: Austin Rockman  

**Link**: [PDF](https://arxiv.org/pdf/2607.22413)  

**Abstract**: Sample retrieval tools can help composers find harmonically compatible material, but querying from a fixed reference sample becomes less informative as arrangements evolve and the harmonic context shifts with each musical decision. We present Reflector, an interactive audio workstation that tracks harmonic combinations as they accumulate on the composer's timeline and adapts retrieval as the arrangement develops. The system is organized around a fixed interval-class oracle: a hand-designed table of weights that scores how pitch-class content combines between sources. An encoder trained entirely on synthetic audio learns to approximate the oracle in a 128-dimensional embedding space, where dot products stand in for compatibility scores at interactive speed. As the composer arranges material on a multi-track timeline, a sweep-line analysis discovers co-sounding regions, computes oracle-weighted centroids, and retrieves against the composite harmonic identity of the session as it evolves. Session centroids projected into a navigable 3-D space reveal structural harmonic relations across the composer's body of work. This paper is a systems account: we give the design rationale for each architectural decision, characterize Reflector's behavior through intrinsic measurements on a working sample library, and describe the implementation. The characterization yields a central finding: the learned embedding preserves the kernel's pairwise judgments while covering the whole library, something the kernel cannot do when used directly as a retrieval rule, because the embedding's normalized geometry cannot express the degenerate solutions that direct scoring favors. The entire pipeline runs locally with no copyrighted training data. Reflector is free, and the training pipeline is open source. 

---
# StARS: Socially Appropriate Robot Actions via a Recommender System-Driven Approach 

**Authors**: Erencem Ozbey, Fethiye Irmak Dogan, Jin Huang, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2607.21802)  

**Abstract**: Social appropriateness in human-robot interaction (HRI) is not universal: different people can judge the same robot action differently in the same situation. To capture this inter-subject variability, we reformulate socially appropriate action generation as a preference modelling problem inspired by recommender systems, treating annotators as users, contexts/scenes as items, and appropriateness scores over a set of candidate robot actions as targets. We propose StARS, a novel model-agnostic framework that integrates collaborative filtering with learnable scene representations to generate user-specific appropriateness scores over candidate robot actions. StARS is model-agnostic: it can be integrated with various scene encoders and backbones, enabling personalisation without redesigning the underlying model. We evaluate StARS on two socially aware robotics datasets, MannersDB+ and SocNav1, and analyse robustness under sparse preference feedback. Across datasets and backbones, StARS consistently improves performance and agreement with annotators, supporting personalised action selection aligned with user norms. Our code is publicly available at this https URL. 

---
