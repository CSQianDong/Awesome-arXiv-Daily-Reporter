# When Stored Evidence Stops Being Usable: Scale-Conditioned Evaluation of Agent Memory 

**Authors**: Jiaqi Shao, Yiyi Lu, Yunzhen Zhang, Bing Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.07313)  

**Abstract**: Memory-agent evaluations report fixed-snapshot accuracy or retrieval quality, but these scores do not show whether evidence remains usable as irrelevant sessions (sessions not annotated as task-relevant evidence for the query) accumulate. We present a scale-conditioned evaluation protocol for agent memory under evidence-preserving growth: for each query, task evidence is held fixed while irrelevant sessions are added. The protocol logs agent--memory trajectories and reports four diagnostics: budget-compliant reliability, tail memory-call burden, failure-regime decomposition, and the usable-scale boundary where reliability falls below the target. Applied to LongMemEval and LoCoMo across flat, planar, and hierarchical memory interfaces, the protocol shows reliability loss is not a single phenomenon. On LongMemEval, HippoRAG stays within the two-call budget but loses 16--20 percentage points in budget-compliant reliability as irrelevant sessions are added; LiCoMemory's observed failures depend strongly on the agent, with Qwen3-8B exceeding the budget while Qwen3-32B and Qwen3-235B remain reliable in the tested range. The result supports a framework for making scalable-memory claims conditional on agent, interface, scale range, and interaction budget. 

---
# LARAG: Link-Aware Retrieval Strategy for RAG Systems in Hyperlinked Technical Documentation 

**Authors**: Giorgia Bolognesi, Claudio Estatico, Ulderico Fugacci, Isabella Mastroianni, Claudio Muselli, Luca Oneto  

**Link**: [PDF](https://arxiv.org/pdf/2605.07517)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances the factual grounding of Large Language Models by conditioning their outputs on external documents. However, standard embedding-based retrievers treat naturally structured corpora, such as technical manuals, as flat collections of passages, thereby overlooking the hyperlink topology that users rely on when navigating such content.
We introduce LARAG (Link-Aware RAG): a lightweight, link-aware retrieval strategy that leverages the author-defined hyperlink structure already present in HTML documentation, encoding hyperlink relations as metadata in the chunk representations and exploiting them to perform a form of graph-like retrieval of locally relevant content.
In a benchmark of twenty expert-designed queries over Rulex Platform technical documentation and four prompting strategies, LARAG consistently improves answer quality, achieving the highest BERTScore F1, while retrieving fewer chunks and generating fewer tokens than a baseline RAG architecture used for comparison. These results show that directly leveraging the existing hyperlink topology of technical documentation, even without explicit graph construction or inference, enables an implicit form of graph-like retrieval that yields a more faithful and efficient RAG pipeline, providing better grounding at lower cost. 

---
# Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding 

**Authors**: Ke Ma, Jiaqi Tang, Bin Guo, Xueting Han, Ruonan Xu, Qingfeng He, Ziheng Wang, Xu Wang, Qifeng Chen, Zhiwen Yu, Yunhao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07575)  

**Abstract**: Proactive streaming video understanding requires Video-LLMs to decide when to respond as a video unfolds, a task where existing methods often fall short due to their implicit, query-agnostic modeling of visual evidence. We introduce Response-G1, a novel framework that establishes explicit, structured alignment between the accumulated video evidence and the query's expected response conditions via scene graphs. The framework operates in three fine-tuning-free stages: (1) online query-guided scene graph generation from streaming clips; (2) memory-based retrieval of the most semantically relevant historical scene graphs; and (3) retrieval-augmented trigger prompting for per-frame "silence/response" this http URL grounding both evidence and conditions in a shared graph representation, Response-G1 achieves more interpretable and accurate response timing decisions. Experimental results on established benchmarks demonstrate the superiority of our method in both proactive and reactive tasks, validating the advantage of explicit scene graph modeling and retrieval in streaming video understanding. 

---
# BioProVLA-Agent: An Affordable, Protocol-Driven, Vision-Enhanced VLA-Enabled Embodied Multi-Agent System with Closed-Loop-Capable Reasoning for Biological Laboratory Manipulation 

**Authors**: Zhaohui Du, Zhe Wang, Hongmei Fei, Xiwen Cao, Ting Xiao, Qi Wang, Huanbo Jin, Jiaming Gu, Quan Lu, Zhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07306)  

**Abstract**: Biological laboratory automation can reduce repetitive manual work and improve reproducibility, but reliable embodied execution in wet-lab environments remains challenging. Protocols are often unstructured, labware is frequently transparent or reflective, and multi-step procedures require state-aware execution beyond one-shot instruction following. Existing robotic systems often rely on costly hardware, fixed workflows, dedicated instruments, or robotics-oriented interfaces. Here, we introduce BioProVLA-Agent, an affordable, protocol-driven, vision-enhanced embodied multi-agent system enabled by Vision-Language-Action (VLA) models for biological manipulation. The system uses protocols as the task interface and integrates protocol parsing, visual state verification, and embodied execution in a closed-loop workflow. A Tailored LLM Protocol Agent converts protocols into verifiable subtasks; a VLM-RAG Verification Agent assesses readiness and completion using observations, robot states, retrieved knowledge, and success/failure examples; and a VLA Embodied Agent executes verified subtasks through a lightweight policy. To improve robustness under wet-lab visual perturbations, we develop AugSmolVLA, an online augmentation strategy targeting transparent labware, reflections, illumination shifts, and overexposure. We evaluate the system on a hierarchical benchmark covering 15 atomic tasks, 6 composite workflows, and 3 bimanual tasks, including tube loading, sorting, waste disposal, cap twisting, and liquid pouring. Across normal and high-exposure settings, AugSmolVLA improves execution stability over ACT, X-VLA, and the original SmolVLA, especially for precise placement, transparent-object manipulation, composite workflows, and visually degraded scenes. These results suggest a practical route toward accessible, protocol-centered, and verification-capable embodied AI for biological manipulation. 

---
# From Clouds to Hallucinations: Atmospheric Retrieval Hijacking in Remote Sensing Vision-Language RAG 

**Authors**: Jiaju Han, Chao Li, Chengyin Hu, Qike Zhang, Xuemeng Sun, Xin Wang, Fengyu Zhang, Xiang Chen, Yiwei Wei, Jiahuan Long, Jiujiang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.07273)  

**Abstract**: Multimodal RAG systems increasingly rely on vision-language retrievers to ground visual queries in external textual evidence. Existing adversarial studies on RAG mainly manipulate the retrieval corpus or memory, while attacks on vision-language and remote sensing models typically target end-task predictions. Input-space threats to the evidence retrieval stage of remote sensing multimodal RAG remain underexplored. To address this gap, we introduce CloudWeb, an atmospheric retrieval hijacking attack that modifies only the input image while keeping the retriever, generator, and knowledge base fixed at deployment. CloudWeb overlays parameterized cloud- and haze-like patterns on remote sensing images and optimizes them with a retrieval-oriented objective that pulls adversarial image embeddings toward target atmospheric evidence, suppresses source-scene evidence, enforces rank separation, and regularizes naturalness and coverage. To the best of our knowledge, this is the first study of retrieval-stage atmospheric evidence hijacking in remote sensing multimodal RAG. We evaluate CloudWeb on a seven-dataset remote sensing RAG benchmark with five CLIP-style retrievers, including GeoRSCLIP, RemoteCLIP, OpenAI CLIP, and OpenCLIP, together with downstream vision-language generators. Across retrievers, CloudWeb consistently outperforms clean retrieval, handcrafted atmospheric baselines, random cloud perturbations, and fixed variants in injecting weather-related evidence into top-ranked results. On GeoRSCLIP ViT-B/32, Weather@5 increases from 0.71\% to 43.29\%. Downstream generation further shows measurable weather hallucination and semantic shift, indicating that retrieval-stage hijacking can propagate to the final RAG response. These findings reveal a practical failure mode: natural-looking atmospheric changes can compromise evidence retrieval before generation begins. 

---
# Hallucination Detection via Activations of Open-Weight Proxy Analyzers 

**Authors**: Akshita Singh, Prabesh Paudel, Siddhartha Roy  

**Link**: [PDF](https://arxiv.org/pdf/2605.07209)  

**Abstract**: We introduce a proxy-analyzer framework for detecting hallucinations in large language models. Instead of looking inside the generating model, our system reads already-generated text through a small locally hosted open-weight model and spots hallucinations using the reader's own internal activations. This works just as well when the generator is a closed API like GPT-4 as when it is any open-weight model. We built eighteen features grounded in how transformers process text, covering residual stream norms, per-head source-document attention, entropy, MLP activations, logit-lens trajectories, and three new token-level grounding statistics. We trained a stacking ensemble on 72,135 samples from five hallucination datasets. We tested across seven analyzer architectures from 0.5 billion to 9 billion parameters: Qwen2.5 at 0.5B and 7B, Gemma-2 at 2B and 9B, Pythia at 1.4B, and LLaMA-3 at both 3B and 8B. Across all seven, we consistently beat ReDeEP's token-level AUC of 0.73 on RAGTruth by 7.4 to 10.3 percentage points. Qwen2.5-7B reached an F1 of 0.717, just above ReDeEP's 0.713, while Qwen2.5-0.5B hit 0.706. The most striking finding is how tightly all seven models cluster: AUC spans only 2.3 percentage points across an eighteen-fold difference in model size. Even more surprising, our 3B LLaMA outperforms our 8B LLaMA on RAGTruth, showing that bigger is not always better even within the same model family. Both RAGTruth and LLM-AggreFact include outputs from multiple LLM families, so our results are not skewed toward any particular generator. 

---
# RRCM: Ranking-Driven Retrieval over Collaborative and Meta Memories for LLM Recommendation 

**Authors**: Shijun Li, Wooseong Yang, Yu Wang, Tianxin Wei, Joydeep Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2605.07129)  

**Abstract**: Large Language Models (LLMs) have emerged as a promising paradigm for next-generation recommender systems, offering strong semantic understanding and natural-language reasoning abilities. Despite recent progress, current LLM-based recommenders still face key challenges in constructing decision-relevant contexts from heterogeneous evidence. First, existing methods often rely on fixed context construction strategies: collaborative behavioral evidence and item-side metadata are typically incorporated through predefined prompts, static retrieval pipelines, or handcrafted injection mechanisms, making it difficult to determine what information is truly beneficial for each instance. Second, heterogeneous evidence introduces a severe context-efficiency bottleneck. Rich metadata and collaborative interaction records can quickly overwhelm the context window, while aggressive compression or heuristic filtering may discard fine-grained evidence critical for accurate recommendation. To address these challenges, we propose RRCM, a ranking-driven retrieval-and-reasoning framework over collaborative and metadata memories for LLM-based agentic recommendation. RRCM starts from a lightweight user-history context and learns whether to recommend directly, retrieve collaborative evidence, retrieve item metadata, or interleave both through reasoning. Both memories are represented in natural language and accessed through a unified retrieval interface, enabling flexible evidence acquisition without handcrafted CF injection or fixed retrieval rules. We optimize this memory-reading policy with an outcome-only ranking reward, instantiated using group relative policy optimization, so that retrieval decisions are directly driven by final top-k recommendation quality. Extensive experiments show that RRCM significantly outperforms traditional baselines and diverse LLM-based recommendation approaches. 

---
# From Surface Learning to Deep Understanding: A Grounded AI Tutoring System for Moodle 

**Authors**: Anna Ostrowska, Michał Kukla, Gabriela Majstrak, Jan Opala, Sebastian Pergała, Jan Skwarek, Anna Wróblewska  

**Link**: [PDF](https://arxiv.org/pdf/2605.06963)  

**Abstract**: This demo paper describes the development of the AI Teaching \& Learning Assistant, a modular Moodle plugin that leverages Retrieval-Augmented Generation (RAG) to deliver high-quality, hallucination-free education. The system employs a dual-centric design, providing students with interactive, Socratic-based tutoring and educators with a "human-in-the-loop" workspace for supervised content generation. By grounding Large Language Model (LLM) responses in teacher-provided materials, the assistant addresses the risks of misinformation while encouraging deep conceptual mastery. Evaluation via the Ragas (LLM-as-a-Judge) framework and a preliminary user study confirms its effectiveness, achieving faithfulness scores up to 0.97 and a 4.00/5.00 recommendation rate. 

---
# WiCER: Wiki-memory Compile, Evaluate, Refine Iterative Knowledge Compilation for LLM Wiki Systems 

**Authors**: Juan M. Huerta  

**Link**: [PDF](https://arxiv.org/pdf/2605.07068)  

**Abstract**: The LLM Wiki pattern, to compile and provide domain knowledge into a persistent artifact and serve it to LLMs via KV cache inference, promises context access at sub-second latency with zero retrieval failure. Realizing this requires solving the compilation gap: LLM compilation distilling raw documents into a wiki without catastrophically discarding critical facts. We characterize this gap across 17 RepLiQA domains (6,800 questions): we observe that full context KV cache inference outperforms RAG on curated knowledge (4.38 vs. 4.08 out of 5, 7.3 faster TTFT) but degrades below RAG at scale due to attention dilution, and blind compilation fails entirely (2.14 to 2.32 vs. 3.46, 53 to 60% catastrophic failure rate). To address the compilation gap, we propose WiCER (Wiki-memory Compile, Evaluate, Refine), an iterative algorithm inspired by counterexample-guided abstraction refinement (CEGAR) that closes this gap. WiCER evaluates compiled wikis against diagnostic probes, identifies dropped facts, and forces their preservation in subsequent compilations. One to two iterations recover 80% of lost quality (mean 3.24 vs. 3.47 for raw full-context across the 15 topics with baselines), reducing catastrophic failures by 55% relative. An ablation across all 17 topics confirms that targeted diagnosis (+0.95), not generic pinning (+0.16), drives the gains. All code and benchmarks are released for reproducible research. 

---
# Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation 

**Authors**: Hongyang Su, Beibei Kong, Lei Cheng, Chengxiang Zhuo, Zang Li, Chenyun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07613)  

**Abstract**: Conversational news recommendation requires grounding each suggestion in a rapidly evolving article corpus while addressing implicit user intents that lack explicit retrievable keywords. To characterize this scenario, we identify 6 intent types from production dialogues: five are implicit and pose fundamental challenges to standard RAG pipelines, forming a critical retrieve-first bottleneck. To address these issues, we introduce intent-driven Semantic ID (SID) generation under a Generate-then-Match paradigm. With two-stage training that consists of multi-task SID alignment and GPT-4 Chain-of-Thought distillation, an LLM maps diverse intents to hierarchical SID prefixes, which are then fuzzy-matched to the current news pool to guarantee fully grounded recommendations. Profile-Aware Dual-Signal Reasoning (PADR) further enables cold-start users to obtain valid recommendations using only profiles. On a mainstream Chinese news platform, our 7B model achieves 0% hallucination and 12.4% L1 match in the 152K open-generation SID space (4x random baseline). It matches GPT-4+Hybrid RAG on L1 while surpassing it on finer-grained metrics (L2 2x, Category +1.2pp) at ~100x lower cost. Cold-start users, where existing baselines score 0%, achieve 18.0% L1 (6x random), the highest among all user groups. 

---
# MIPIAD: Multilingual Indirect Prompt Injection Attack Defense with Qwen -- TF-IDF Hybrid and Meta-Ensemble Learning 

**Authors**: Al Muhit Muhtadi, Mostafa Rifat Tazwar  

**Link**: [PDF](https://arxiv.org/pdf/2605.07269)  

**Abstract**: Indirect prompt injection remains a persistent weakness in retrieval-augmented and tool-using LLM systems, and the problem becomes harder to characterise in multilingual settings. We present MIPIAD, a defense framework evaluated on English and Bangla that combines a sequence classifier fine-tuned from Qwen2.5-1.5B via LoRA (XLPID), TF-IDF lexical features, and validation-tuned ensembling through late fusion, stacking, and gradient boosting. The framework is evaluated on a synthetic benchmark built from BIPIA(Yi et al., 2023) templates spanning five task families -- email, table, QA, abstract, and code-comprising over 1.43 million generated samples, with train and test splits using mutually exclusive attack categories. Across the experiments, lexical signals prove strong (TF-IDF+SVM F1=0.77), and the hybrid XLPID+TF-IDF ensemble achieves the best overall F1 (0.9205) while the Boosting Ensemble achieves the best AUROC (0.9378). Ensemble methods consistently reduce the English-Bangla cross-lingual gap relative to standalone neural models. The pipeline is designed for extensibility: NLLB-200 supports over 200 languages and XLPID's multilingual backbone can be retargeted to additional languages without architectural changes; empirical validation is currently limited to English and Bangla 

---
# Can LLMs Take Retrieved Information with a Grain of Salt? 

**Authors**: Behzad Shayegh, Mohamed Osama Ahmed, Fred Tung, Leo Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.06919)  

**Abstract**: Large language models have demonstrated impressive retrieval-augmented capabilities. However, a crucial area remains underexplored: their ability to appropriately adapt responses to the certainty of the retrieved information. It is a limitation with real consequences in high-stakes domains like medicine and finance. We evaluate eight LLMs on their context-certainty obedience, measuring how well they adjust responses to match expressed context certainty. Our analysis reveals systematic limitations: LLMs struggle to recall prior knowledge after observing an uncertain context, misinterpret expressed certainties, and overtrust complex contexts. To address these, we propose an interaction strategy combining prior reminders, certainty recalibration, and context simplification. This approach reduces obedience errors by 25% on average, without modifying model weights, demonstrating the efficacy of interaction design in enhancing LLM reliability. Our contributions include a principled evaluation metric, empirical insights into LLMs' uncertainty handling, and a portable strategy to improve context-certainty obedience across diverse LLMs. 

---
# InterLV-Search: Benchmarking Interleaved Multimodal Agentic Search 

**Authors**: Bohan Hou, Jiuning Gu, Jiayan Guo, Ronghao Dang, Sicong Leng, Xin Li, Xuemeng Song, Jianfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.07510)  

**Abstract**: Existing benchmarks for multimodal agentic search evaluate multimodal search and visual browsing, but visual evidence is either confined to the input or treated as an answer endpoint rather than part of an interleaved search trajectory. We introduce \textbf{InterLV-Search}, a benchmark for Interleaved Language-Vision Agentic Search, in which textual and visual evidence is repeatedly used to condition later search. It contains 2,061 examples across three levels: active visual evidence seeking, controlled offline interleaved multimodal search, and open-web interleaved multimodal search. Beyond existing benchmarks, it also includes multimodal multi-branch samples that involve comparison between multiple entities during the evidence search. We construct Level 1 and Level 2 with automated pipelines and Level 3 with a machine-led, human-supervised open-web pipeline. We further provide InterLV-Agent for standardized tool use, trajectory logging, and evaluation. Experiments on proprietary and open-source multimodal agents show that current systems remain far from solving interleaved multimodal search, with the best model below 50% overall accuracy, highlighting challenges in visual evidence seeking, search control, and multimodal evidence integration. We release the benchmark data and evaluation code at this https URL 

---
# TRACE: Tourism Recommendation with Accountable Citation Evidence 

**Authors**: Zixu Zhao, Sijin Wang, Yu Hou, Yuanyuan Xu, Yufan Sheng, Xike Xie, Wenjie Zhang, Won-Yong Shin, Xin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2605.07677)  

**Abstract**: Tourism is a high-stakes setting for conversational recommender systems (CRS): a plausible-sounding suggestion can waste real money and trip time once a traveler acts on it. Existing CRS benchmarks primarily evaluate systems with a single Recall@k score over entity mentions, and tourism-specific resources add spatial or knowledge-graph context, yet none of them couple multi-turn recommendation with verbatim review-span evidence and rejection recovery. This leaves an evaluation gap for tourism recommendation that is simultaneously trustworthy, verifiable, and adaptive: recommend the right point of interest (POI) for multi-aspect preferences (such as cuisine, price, atmosphere, walking distance), justify each suggestion with verifiable evidence from prior visitors so the traveler can act without trial and error, and recover when the first recommendation is rejected mid-dialogue. We introduce TRACE, where each item is a multi-turn tourism recommendation dialogue with review-span citations and explicit rejection turns: 10,000 dialogues over 2,400 Yelp POIs and 34,208 reviews across eight U.S. cities, paired with 14 retrieval, planning, and LLM baselines, along with 25 metrics organized under Accuracy, Grounding, and Recovery. Across these baselines, TRACE reveals the Three-Competency Gap: LLM Zero-Shot leads in closed-set Recall@1 and rejection recovery but cites less densely than retrievers; non-LLM retrievers achieve surface-verbatim grounding but with low accuracy; Multi-Review Synthesis fails at recovery. The Grounding Score agrees with human citation precision (Spearman rho=+0.80, p<10^-20), and paired t-tests reproduce the per-baseline ranking (p<0.01 on the dominant contrasts). TRACE reframes accountable tourism recommendation as a joint target (right POI, verifiable evidence, adaptive repair) rather than a single-axis leaderboard. 

---
# DiffRetriever: Parallel Representative Tokens for Retrieval with Diffusion Language Models 

**Authors**: Shuai Wang, Yin Yu, Shengyao Zhuang, Bevan Koopman, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2605.07210)  

**Abstract**: PromptReps showed that an autoregressive language model can be used directly as a retriever by prompting it to generate dense and sparse representations of a query or passage. Extending this to multiple representatives is inefficient for autoregressive models, since tokens must be generated sequentially, and prior multi-token variants did not reliably improve over single-token decoding.
We show that the bottleneck is sequential generation, not the multi-token idea itself. DiffRetriever is a representative-token retriever for diffusion language models: it appends K masked positions to the prompt and reads all K in a single bidirectional forward pass. Across in-domain and out-of-domain evaluation, multi-token DiffRetriever substantially improves over single-token on every diffusion backbone we test, while autoregressive multi-token is flat or negative and pays a latency cost that scales with K where diffusion does not. After supervised fine-tuning, DiffRetriever on Dream is the strongest BEIR-7 retriever in our comparison, ahead of PromptReps, the encoder-style DiffEmbed baseline on the same diffusion backbones, and the contrastively fine-tuned single-vector RepLLaMA. A per-query oracle on the frozen base model exceeds contrastive fine-tuning at the same fixed budget, pointing to adaptive budget selection as future work. Code is available at this https URL. 

---
# Topic Is Not Agenda: A Citation-Community Audit of Text Embeddings 

**Authors**: Junseon Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2605.07158)  

**Abstract**: Vector search and retrieval-augmented generation (RAG) rest on the assumption that cosine similarity between text embeddings reflects conceptual relatedness. We measure where this assumption breaks. We build an augmented citation graph over 3.58M scientific papers and partition it via Leiden CPM at two granularities: sub-field (L1) and research-agenda (L2, hierarchical inside each L1). Four state-of-the-art embeddings (Gemini, Qwen3-8B, Qwen3-0.6B, SPECTER2) clear the L1 bar reasonably (45-52% top-10 same-rate) but stop working at L2: only 15-21% of top-10 neighbors share the query's research agenda. In absolute terms, 8 of every 10 retrieved papers are off-agenda. The failure is universal across eight scientific domains and all four models; SPECTER2, despite its citation-based contrastive training, is the weakest. As a diagnostic probe, we test whether the same augmented graph also functions as a retrieval signal: a deliberately simple citation-count rerank reaches 57.7% top-1 L2 on top of LLM-expanded Boolean retrieval and 59.6% on top of plain BM25, on 80 curated agenda queries -- about 9 points above the best cosine retriever (Gemini, 50.6%) and 20 points above BM25 alone (39.3%). The probe isolates a slice of the agenda-matching signal the graph carries but the embeddings miss, connecting recent theoretical limits on single-vector retrieval to a concrete failure mode of scientific RAG. 

---
# MLAIRE: Multilingual Language-Aware Information Retrieval Evaluation Protocal 

**Authors**: Youngjoon Jang, Seongtae Hong, Hyeonseok Moon, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2605.07249)  

**Abstract**: Multilingual Information Retrieval is increasingly important in real-world search settings, where users issue queries over mixed-language corpora. Existing evaluations mainly reward language-agnostic semantic relevance, treating relevant passages equally regardless of language. Yet retrieval utility also depends on the language of the retrieved passages: users may prefer results they can read and verify in the query language, and query--passage language mismatch can complicate downstream grounding and answer verification in Retrieval-Augmented Generation systems. To evaluate this language-aware dimension, we introduce MLAIRE, a Multilingual Language-Aware Information Retrieval Evaluation protocol that disentangles cross-lingual semantic retrieval from query-language preference. MLAIRE constructs controlled pools with parallel passages across languages, enabling measurement of semantic retrieval accuracy and query-language preference when equivalent translations are available. We propose language-aware metrics, including Language Preference Rate (LPR) and Lang-nDCG, together with a 4-way decomposition separating semantic and query-language preference failures. Evaluating 31 dense, sparse, and late-interaction retrievers, we show that standard metrics obscure distinct behaviors: semantically strong retrievers may return correct content in a non-query language, while retrievers with stronger query-language preference may retrieve less semantically relevant passages. 

---
# FAVOR: Efficient Filter-Agnostic Vector ANNS Based on Selectivity-Aware Exclusion Distances 

**Authors**: Junjie Song, Yu Liu, Guoyu Hu, Zhongle Xie, Ming Yang, Beng Chin Ooi, Ke Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.07770)  

**Abstract**: Modern retrieval systems increasingly require integrating approximate nearest neighbor search (ANNS) with complex attribute filtering to handle hybrid queries in applications such as recommendation systems and retrieval-augmented generation (RAG). While HNSW-based inline-filtering methods show promise, existing approaches struggle to deliver high throughput under low-selectivity scenarios while balancing search efficiency, filtering generality, and index connectivity. To address these challenges, we propose FAVOR, an efficient filter-agnostic vector ANNS that supports arbitrary filtering conditions while maintaining stable performance across varying selectivity levels. FAVOR introduces three novel features: (1) an integrated architecture that unifies selectivity estimation and filtered ANNS execution, providing a cohesive solution for hybrid vector-attribute queries; (2) a HNSW-based inline-filtering algorithm that introduces an exclusion distance mechanism to dynamically reshape the vector distance distribution, pushing non-target vectors away from the query while promoting valid candidates toward the query, thus improving search efficiency without compromising generality or graph connectivity; and (3) a selectivity-driven search selector that estimates query selectivity and dynamically routes queries between a pre-filtering brute-force algorithm for low-selectivity cases and an optimized HNSW-based search algorithm for other scenarios, ensuring consistent performance. Extensive experiments on real-world datasets demonstrate that FAVOR achieves a 1.3-5$\times$ higher QPS at $Recall@10 = 95\%$ compared to state-of-the-art methods for arbitrary filtering conditions, while maintaining competitive performance even against tailored solutions in some filtering conditions. 

---
