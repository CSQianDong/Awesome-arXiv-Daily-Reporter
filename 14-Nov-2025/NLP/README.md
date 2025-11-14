# ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference 

**Authors**: Yesheng Liang, Haisheng Chen, Song Han, Zhijian Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.10645)  

**Abstract**: Weight-only post-training quantization (PTQ) compresses the weights of Large Language Models (LLMs) into low-precision representations to reduce memory footprint and accelerate inference. However, the presence of outliers in weights and activations often leads to large quantization errors and severe accuracy degradation, especially in recent reasoning LLMs where errors accumulate across long chains of thought. Existing PTQ methods either fail to sufficiently suppress outliers or introduce significant overhead during inference. In this paper, we propose Pairwise Rotation Quantization (ParoQuant), a weight-only PTQ method that combines hardware-efficient and optimizable independent Givens rotations with channel-wise scaling to even out the magnitude across channels and narrow the dynamic range within each quantization group. We further co-design the inference kernel to fully exploit GPU parallelism and keep the rotations and scaling lightweight at runtime. ParoQuant achieves an average 2.4% accuracy improvement over AWQ on reasoning tasks with less than 10% overhead. This paves the way for more efficient and accurate deployment of reasoning LLMs. 

---
# Black-Box On-Policy Distillation of Large Language Models 

**Authors**: Tianzhu Ye, Li Dong, Zewen Chi, Xun Wu, Shaohan Huang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2511.10643)  

**Abstract**: Black-box distillation creates student large language models (LLMs) by learning from a proprietary teacher model's text outputs alone, without access to its internal logits or parameters. In this work, we introduce Generative Adversarial Distillation (GAD), which enables on-policy and black-box distillation. GAD frames the student LLM as a generator and trains a discriminator to distinguish its responses from the teacher LLM's, creating a minimax game. The discriminator acts as an on-policy reward model that co-evolves with the student, providing stable, adaptive feedback. Experimental results show that GAD consistently surpasses the commonly used sequence-level knowledge distillation. In particular, Qwen2.5-14B-Instruct (student) trained with GAD becomes comparable to its teacher, GPT-5-Chat, on the LMSYS-Chat automatic evaluation. The results establish GAD as a promising and effective paradigm for black-box LLM distillation. 

---
# Instella: Fully Open Language Models with Stellar Performance 

**Authors**: Jiang Liu, Jialian Wu, Xiaodong Yu, Yusheng Su, Prakamya Mishra, Gowtham Ramesh, Sudhanshu Ranjan, Chaitanya Manem, Ximeng Sun, Ze Wang, Pratik Prabhanjan Brahma, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2511.10628)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance across a wide range of tasks, yet the majority of high-performing models remain closed-source or partially open, limiting transparency and reproducibility. In this work, we introduce Instella, a family of fully open three billion parameter language models trained entirely on openly available data and codebase. Powered by AMD Instinct MI300X GPUs, Instella is developed through large-scale pre-training, general-purpose instruction tuning, and alignment with human preferences. Despite using substantially fewer pre-training tokens than many contemporaries, Instella achieves state-of-the-art results among fully open models and is competitive with leading open-weight models of comparable size. We further release two specialized variants: Instella-Long, capable of handling context lengths up to 128K tokens, and Instella-Math, a reasoning-focused model enhanced through supervised fine-tuning and reinforcement learning on mathematical tasks. Together, these contributions establish Instella as a transparent, performant, and versatile alternative for the community, advancing the goal of open and reproducible language modeling research. 

---
# SSR: Socratic Self-Refine for Large Language Model Reasoning 

**Authors**: Haizhou Shi, Ye Liu, Bo Pang, Zeyu Leo Liu, Hao Wang, Silvio Savarese, Caiming Xiong, Yingbo Zhou, Semih Yavuz  

**Link**: [PDF](https://arxiv.org/pdf/2511.10621)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable reasoning abilities, yet existing test-time frameworks often rely on coarse self-verification and self-correction, limiting their effectiveness on complex tasks. In this paper, we propose Socratic Self-Refine (SSR), a novel framework for fine-grained evaluation and precise refinement of LLM reasoning. Our proposed SSR decomposes model responses into verifiable (sub-question, sub-answer) pairs, enabling step-level confidence estimation through controlled re-solving and self-consistency checks. By pinpointing unreliable steps and iteratively refining them, SSR produces more accurate and interpretable reasoning chains. Empirical results across five reasoning benchmarks and three LLMs show that SSR consistently outperforms state-of-the-art iterative self-refinement baselines. Beyond performance gains, SSR provides a principled black-box approach for evaluating and understanding the internal reasoning processes of LLMs. Code is available at this https URL. 

---
# Know Your Limits: Entropy Estimation Modeling for Compression and Generalization 

**Authors**: Benjamin L. Badger, Matthew Neligeorge  

**Link**: [PDF](https://arxiv.org/pdf/2511.10618)  

**Abstract**: Language prediction is constrained by informational entropy intrinsic to language, such that there exists a limit to how accurate any language model can become and equivalently a lower bound to language compression. The most efficient language compression algorithms today are causal (next token prediction) large language models, but the use of these models to form accurate estimates of language entropy is currently computationally infeasible. We introduce encoder-augmented causal decoder model architectures that exhibit superior training efficiency characteristics and achieve higher compression than causal transformers even when trained on modest hardware. We demonstrate how entropy estimates can be obtained on a per-token basis, and show that the generalization of models trained to approach the entropy of their training data necessarily exceeds the generalization of models trained to minimize loss beyond this value. We show empirically that causal models trained to approach but not exceed estimated per-token entropies exhibit greater generalization than models trained without taking entropy into account. 

---
# Mined Prompting and Metadata-Guided Generation for Wound Care Visual Question Answering 

**Authors**: Bavana Durgapraveen, Sornaraj Sivasankaran, Abhinand Balachandran, Sriram Rajkumar  

**Link**: [PDF](https://arxiv.org/pdf/2511.10591)  

**Abstract**: The rapid expansion of asynchronous remote care has intensified provider workload, creating demand for AI systems that can assist clinicians in managing patient queries more efficiently. The MEDIQA-WV 2025 shared task addresses this challenge by focusing on generating free-text responses to wound care queries paired with images. In this work, we present two complementary approaches developed for the English track. The first leverages a mined prompting strategy, where training data is embedded and the top-k most similar examples are retrieved to serve as few-shot demonstrations during generation. The second approach builds on a metadata ablation study, which identified four metadata attributes that consistently enhance response quality. We train classifiers to predict these attributes for test cases and incorporate them into the generation pipeline, dynamically adjusting outputs based on prediction confidence. Experimental results demonstrate that mined prompting improves response relevance, while metadata-guided generation further refines clinical precision. Together, these methods highlight promising directions for developing AI-driven tools that can provide reliable and efficient wound care support. 

---
# Evaluating Prompting Strategies with MedGemma for Medical Order Extraction 

**Authors**: Abhinand Balachandran, Bavana Durgapraveen, Gowsikkan Sikkan Sudhagar, Vidhya Varshany J S, Sriram Rajkumar  

**Link**: [PDF](https://arxiv.org/pdf/2511.10583)  

**Abstract**: The accurate extraction of medical orders from doctor-patient conversations is a critical task for reducing clinical documentation burdens and ensuring patient safety. This paper details our team submission to the MEDIQA-OE-2025 Shared Task. We investigate the performance of MedGemma, a new domain-specific open-source language model, for structured order extraction. We systematically evaluate three distinct prompting paradigms: a straightforward one-Shot approach, a reasoning-focused ReAct framework, and a multi-step agentic workflow. Our experiments reveal that while more complex frameworks like ReAct and agentic flows are powerful, the simpler one-shot prompting method achieved the highest performance on the official validation set. We posit that on manually annotated transcripts, complex reasoning chains can lead to "overthinking" and introduce noise, making a direct approach more robust and efficient. Our work provides valuable insights into selecting appropriate prompting strategies for clinical information extraction in varied data conditions. 

---
# DESS: DeBERTa Enhanced Syntactic-Semantic Aspect Sentiment Triplet Extraction 

**Authors**: Vishal Thenuwara, Nisansa de Silva  

**Link**: [PDF](https://arxiv.org/pdf/2511.10577)  

**Abstract**: Fine-grained sentiment analysis faces ongoing challenges in Aspect Sentiment Triple Extraction (ASTE), particularly in accurately capturing the relationships between aspects, opinions, and sentiment polarities. While researchers have made progress using BERT and Graph Neural Networks, the full potential of advanced language models in understanding complex language patterns remains unexplored. We introduce DESS, a new approach that builds upon previous work by integrating DeBERTa's enhanced attention mechanism to better understand context and relationships in text. Our framework maintains a dual-channel structure, where DeBERTa works alongside an LSTM channel to process both meaning and grammatical patterns in text. We have carefully refined how these components work together, paying special attention to how different types of language information interact. When we tested DESS on standard datasets, it showed meaningful improvements over current methods, with F1-score increases of 4.85, 8.36, and 2.42 in identifying aspect opinion pairs and determining sentiment accurately. Looking deeper into the results, we found that DeBERTa's sophisticated attention system helps DESS handle complicated sentence structures better, especially when important words are far apart. Our findings suggest that upgrading to more advanced language models when thoughtfully integrated, can lead to real improvements in how well we can analyze sentiments in text. The implementation of our approach is publicly available at: this https URL. 

---
# URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding 

**Authors**: Yongxin Shi, Jiapeng Wang, Zeyu Shan, Dezhi Peng, Zening Lin, Lianwen Jin  

**Link**: [PDF](https://arxiv.org/pdf/2511.10552)  

**Abstract**: Recent multimodal large language models (MLLMs) still struggle with long document understanding due to two fundamental challenges: information interference from abundant irrelevant content, and the quadratic computational cost of Transformer-based architectures. Existing approaches primarily fall into two categories: token compression, which sacrifices fine-grained details; and introducing external retrievers, which increase system complexity and prevent end-to-end optimization. To address these issues, we conduct an in-depth analysis and observe that MLLMs exhibit a human-like coarse-to-fine reasoning pattern: early Transformer layers attend broadly across the document, while deeper layers focus on relevant evidence pages. Motivated by this insight, we posit that the inherent evidence localization capabilities of MLLMs can be explicitly leveraged to perform retrieval during the reasoning process, facilitating efficient long document understanding. To this end, we propose URaG, a simple-yet-effective framework that Unifies Retrieval and Generation within a single MLLM. URaG introduces a lightweight cross-modal retrieval module that converts the early Transformer layers into an efficient evidence selector, identifying and preserving the most relevant pages while discarding irrelevant content. This design enables the deeper layers to concentrate computational resources on pertinent information, improving both accuracy and efficiency. Extensive experiments demonstrate that URaG achieves state-of-the-art performance while reducing computational overhead by 44-56%. The code is available at this https URL. 

---
# Computing the Formal and Institutional Boundaries of Contemporary Genre and Literary Fiction 

**Authors**: Natasha Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2511.10546)  

**Abstract**: Though the concept of genre has been a subject of discussion for millennia, the relatively recent emergence of genre fiction has added a new layer to this ongoing conversation. While more traditional perspectives on genre have emphasized form, contemporary scholarship has invoked both formal and institutional characteristics in its taxonomy of genre, genre fiction, and literary fiction. This project uses computational methods to explore the soundness of genre as a formal designation as opposed to an institutional one. Pulling from Andrew Piper's CONLIT dataset of Contemporary Literature, we assemble a corpus of literary and genre fiction, with the latter category containing romance, mystery, and science fiction novels. We use Welch's ANOVA to compare the distribution of narrative features according to author gender within each genre and within genre versus literary fiction. Then, we use logistic regression to model the effect that each feature has on literary classification and to measure how author gender moderates these effects. Finally, we analyze stylistic and semantic vector representations of our genre categories to understand the importance of form and content in literary classification. This project finds statistically significant formal markers of each literary category and illustrates how female authorship narrows and blurs the target for achieving literary status. 

---
# Convomem Benchmark: Why Your First 150 Conversations Don't Need RAG 

**Authors**: Egor Pakhomov, Erik Nijkamp, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2511.10523)  

**Abstract**: We introduce a comprehensive benchmark for conversational memory evaluation containing 75,336 question-answer pairs across diverse categories including user facts, assistant recall, abstention, preferences, temporal changes, and implicit connections. While existing benchmarks have advanced the field, our work addresses fundamental challenges in statistical power, data generation consistency, and evaluation flexibility that limit current memory evaluation frameworks. We examine the relationship between conversational memory and retrieval-augmented generation (RAG). While these systems share fundamental architectural patterns--temporal reasoning, implicit extraction, knowledge updates, and graph representations--memory systems have a unique characteristic: they start from zero and grow progressively with each conversation. This characteristic enables naive approaches that would be impractical for traditional RAG. Consistent with recent findings on long context effectiveness, we observe that simple full-context approaches achieve 70-82% accuracy even on our most challenging multi-message evidence cases, while sophisticated RAG-based memory systems like Mem0 achieve only 30-45% when operating on conversation histories under 150 interactions. Our analysis reveals practical transition points: long context excels for the first 30 conversations, remains viable with manageable trade-offs up to 150 conversations, and typically requires hybrid or RAG approaches beyond that point as costs and latencies become prohibitive. These patterns indicate that the small-corpus advantage of conversational memory--where exhaustive search and complete reranking are feasible--deserves dedicated research attention rather than simply applying general RAG solutions to conversation histories. 

---
# Say It Differently: Linguistic Styles as Jailbreak Vectors 

**Authors**: Srikant Panda, Avinash Rai  

**Link**: [PDF](https://arxiv.org/pdf/2511.10519)  

**Abstract**: Large Language Models (LLMs) are commonly evaluated for robustness against paraphrased or semantically equivalent jailbreak prompts, yet little attention has been paid to linguistic variation as an attack surface. In this work, we systematically study how linguistic styles such as fear or curiosity can reframe harmful intent and elicit unsafe responses from aligned models. We construct style-augmented jailbreak benchmark by transforming prompts from 3 standard datasets into 11 distinct linguistic styles using handcrafted templates and LLM-based rewrites, while preserving semantic intent. Evaluating 16 open- and close-source instruction-tuned models, we find that stylistic reframing increases jailbreak success rates by up to +57 percentage points. Styles such as fearful, curious and compassionate are most effective and contextualized rewrites outperform templated variants.
To mitigate this, we introduce a style neutralization preprocessing step using a secondary LLM to strip manipulative stylistic cues from user inputs, significantly reducing jailbreak success rates. Our findings reveal a systemic and scaling-resistant vulnerability overlooked in current safety pipelines. 

---
# LOCA-R: Near-Perfect Performance on the Chinese Physics Olympiad 2025 

**Authors**: Dong-Shan Jian, Xiang Li, Chen-Xu Yan, Hui-Wen Zheng, Zhi-Zhang Bian, You-Le Fang, Sheng-Qi Zhang, Bing-Rui Gong, Ren-Xi He, Jing-Tian Zhang, Ce Meng, Yan-Qing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.10515)  

**Abstract**: Olympiad-level physics problem-solving presents a significant challenge for both humans and artificial intelligence (AI), as it requires a sophisticated integration of precise calculation, abstract reasoning, and a fundamental grasp of physical principles. The Chinese Physics Olympiad (CPhO), renowned for its complexity and depth, serves as an ideal and rigorous testbed for these advanced capabilities. In this paper, we introduce LOCA-R (LOgical Chain Augmentation for Reasoning), an improved version of the LOCA framework adapted for complex reasoning, and apply it to the CPhO 2025 theory examination. LOCA-R achieves a near-perfect score of 313 out of 320 points, solidly surpassing the highest-scoring human competitor and significantly outperforming all baseline methods. 

---
# Rubric-Based Benchmarking and Reinforcement Learning for Advancing LLM Instruction Following 

**Authors**: Yun He, Wenzhe Li, Hejia Zhang, Songlin Li, Karishma Mandyam, Sopan Khosla, Yuanhao Xiong, Nanshu Wang, Selina Peng, Beibin Li, Shengjie Bi, Shishir G. Patil, Qi Qi, Shengyu Feng, Julian Katz-Samuels, Richard Yuanzhe Pang, Sujan Gonugondla, Hunter Lang, Yue Yu, Yundi Qian, Maryam Fazel-Zarandi, Licheng Yu, Amine Benhalloum, Hany Awadalla, Manaal Faruqui  

**Link**: [PDF](https://arxiv.org/pdf/2511.10507)  

**Abstract**: Recent progress in large language models (LLMs) has led to impressive performance on a range of tasks, yet advanced instruction following (IF)-especially for complex, multi-turn, and system-prompted instructions-remains a significant challenge. Rigorous evaluation and effective training for such capabilities are hindered by the lack of high-quality, human-annotated benchmarks and reliable, interpretable reward signals. In this work, we introduce AdvancedIF (we will release this benchmark soon), a comprehensive benchmark featuring over 1,600 prompts and expert-curated rubrics that assess LLMs ability to follow complex, multi-turn, and system-level instructions. We further propose RIFL (Rubric-based Instruction-Following Learning), a novel post-training pipeline that leverages rubric generation, a finetuned rubric verifier, and reward shaping to enable effective reinforcement learning for instruction following. Extensive experiments demonstrate that RIFL substantially improves the instruction-following abilities of LLMs, achieving a 6.7% absolute gain on AdvancedIF and strong results on public benchmarks. Our ablation studies confirm the effectiveness of each component in RIFL. This work establishes rubrics as a powerful tool for both training and evaluating advanced IF in LLMs, paving the way for more capable and reliable AI systems. 

---
# Beyond Elicitation: Provision-based Prompt Optimization for Knowledge-Intensive Tasks 

**Authors**: Yunzhe Xu, Zhuosheng Zhang, Zhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.10465)  

**Abstract**: While prompt optimization has emerged as a critical technique for enhancing language model performance, existing approaches primarily focus on elicitation-based strategies that search for optimal prompts to activate models' capabilities. These methods exhibit fundamental limitations when addressing knowledge-intensive tasks, as they operate within fixed parametric boundaries rather than providing the factual knowledge, terminology precision, and reasoning patterns required in specialized domains. To address these limitations, we propose Knowledge-Provision-based Prompt Optimization (KPPO), a framework that reformulates prompt optimization as systematic knowledge integration rather than potential elicitation. KPPO introduces three key innovations: 1) a knowledge gap filling mechanism for knowledge gap identification and targeted remediation; 2) a batch-wise candidate evaluation approach that considers both performance improvement and distributional stability; 3) an adaptive knowledge pruning strategy that balances performance and token efficiency, reducing up to 29% token usage. Extensive evaluation on 15 knowledge-intensive benchmarks from various domains demonstrates KPPO's superiority over elicitation-based methods, with an average performance improvement of ~6% over the strongest baseline while achieving comparable or lower token consumption. Code at: this https URL. 

---
# LocalBench: Benchmarking LLMs on County-Level Local Knowledge and Reasoning 

**Authors**: Zihan Gao, Yifei Xu, Jacob Thebault-Spieker  

**Link**: [PDF](https://arxiv.org/pdf/2511.10459)  

**Abstract**: Large language models (LLMs) have been widely evaluated on macro-scale geographic tasks, such as global factual recall, event summarization, and regional reasoning. Yet, their ability to handle hyper-local knowledge remains poorly understood. This gap is increasingly consequential as real-world applications, from civic platforms to community journalism, demand AI systems that can reason about neighborhood-specific dynamics, cultural narratives, and local governance. Existing benchmarks fall short in capturing this complexity, often relying on coarse-grained data or isolated references. We present LocalBench, the first benchmark designed to systematically evaluate LLMs on county-level local knowledge across the United States. Grounded in the Localness Conceptual Framework, LocalBench includes 14,782 validated question-answer pairs across 526 U.S. counties in 49 states, integrating diverse sources such as Census statistics, local subreddit discourse, and regional news. It spans physical, cognitive, and relational dimensions of locality. Using LocalBench, we evaluate 13 state-of-the-art LLMs under both closed-book and web-augmented settings. Our findings reveal critical limitations: even the best-performing models reach only 56.8% accuracy on narrative-style questions and perform below 15.5% on numerical reasoning. Moreover, larger model size and web augmentation do not guarantee better performance, for example, search improves Gemini's accuracy by +13.6%, but reduces GPT-series performance by -11.4%. These results underscore the urgent need for language models that can support equitable, place-aware AI systems: capable of engaging with the diverse, fine-grained realities of local communities across geographic and cultural contexts. 

---
# Exploring State Tracking Capabilities of Large Language Models 

**Authors**: Kiamehr Rezaee, Jose Camacho-Collados, Mohammad Taher Pilehvar  

**Link**: [PDF](https://arxiv.org/pdf/2511.10457)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in solving complex tasks, including those requiring a certain level of reasoning. In this paper, we focus on state tracking, a problem where models need to keep track of the state governing a number of entities. To isolate the state tracking component from other factors, we propose a benchmark based on three well-defined state tracking tasks and analyse the performance of LLMs in different scenarios. The results indicate that the recent generation of LLMs (specifically, GPT-4 and Llama3) are capable of tracking state, especially when integrated with mechanisms such as Chain of Thought. However, models from the former generation, while understanding the task and being able to solve it at the initial stages, often fail at this task after a certain number of steps. 

---
# Reasoning About Intent for Ambiguous Requests 

**Authors**: Irina Saparina, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2511.10453)  

**Abstract**: Large language models often respond to ambiguous requests by implicitly committing to one interpretation. Intent misunderstandings can frustrate users and create safety risks. To address this, we propose generating multiple interpretation-answer pairs in a single structured response to ambiguous requests. Our models are trained with reinforcement learning and customized reward functions using multiple valid answers as supervision. Experiments on conversational question answering and semantic parsing demonstrate that our method achieves higher coverage of valid answers than baseline approaches. Human evaluation confirms that predicted interpretations are highly aligned with their answers. Our approach promotes transparency with explicit interpretations, achieves efficiency by requiring only one generation step, and supports downstream applications through its structured output format. 

---
# Analogical Structure, Minimal Contextual Cues and Contrastive Distractors: Input Design for Sample-Efficient Linguistic Rule Induction 

**Authors**: Chunyang Jiang, Paola Merlo  

**Link**: [PDF](https://arxiv.org/pdf/2511.10441)  

**Abstract**: Large language models achieve strong performance through training on vast datasets. Can analogical paradigm organization enable lightweight models to match this performance with minimal data? We develop a computational approach implementing three cognitive-inspired principles: analogical structure, contrastive learning, and minimal contextual cues. We test this approach with structured completion tasks where models identify correct sentence completions from analogical patterns with contrastive alternatives. Training lightweight models (BERT+CNN, $0.5M$ parameters) on only one hundred structured examples of English causative/inchoative alternations achieves $F1=0.95$, outperforming zero-shot \texttt{GPT-o3} ($F1=0.87$). Ablation studies confirm that analogical organization and contrastive structure improve performance, consistently surpassing randomly shuffled baselines across architectures. Cross-phenomenon validation using unspecified object alternations replicates these efficiency gains, confirming approach robustness. Our results show that analogical paradigm organization enables competitive linguistic rule learning with orders of magnitude less data than conventional approaches require. 

---
# DELICATE: Diachronic Entity LInking using Classes And Temporal Evidence 

**Authors**: Cristian Santini, Sebastian Barzaghi, Paolo Sernani, Emanuele Frontoni, Mehwish Alam  

**Link**: [PDF](https://arxiv.org/pdf/2511.10404)  

**Abstract**: In spite of the remarkable advancements in the field of Natural Language Processing, the task of Entity Linking (EL) remains challenging in the field of humanities due to complex document typologies, lack of domain-specific datasets and models, and long-tail entities, i.e., entities under-represented in Knowledge Bases (KBs). The goal of this paper is to address these issues with two main contributions. The first contribution is DELICATE, a novel neuro-symbolic method for EL on historical Italian which combines a BERT-based encoder with contextual information from Wikidata to select appropriate KB entities using temporal plausibility and entity type consistency. The second contribution is ENEIDE, a multi-domain EL corpus in historical Italian semi-automatically extracted from two annotated editions spanning from the 19th to the 20th century and including literary and political texts. Results show how DELICATE outperforms other EL models in historical Italian even if compared with larger architectures with billions of parameters. Moreover, further analyses reveal how DELICATE confidence scores and features sensitivity provide results which are more explainable and interpretable than purely neural methods. 

---
# Position: On the Methodological Pitfalls of Evaluating Base LLMs for Reasoning 

**Authors**: Jason Chan, Zhixue Zhao, Robert Gaizauskas  

**Link**: [PDF](https://arxiv.org/pdf/2511.10381)  

**Abstract**: Existing work investigates the reasoning capabilities of large language models (LLMs) to uncover their limitations, human-like biases and underlying processes. Such studies include evaluations of base LLMs (pre-trained on unlabeled corpora only) for this purpose. Our position paper argues that evaluating base LLMs' reasoning capabilities raises inherent methodological concerns that are overlooked in such existing studies. We highlight the fundamental mismatch between base LLMs' pretraining objective and normative qualities, such as correctness, by which reasoning is assessed. In particular, we show how base LLMs generate logically valid or invalid conclusions as coincidental byproducts of conforming to purely linguistic patterns of statistical plausibility. This fundamental mismatch challenges the assumptions that (a) base LLMs' outputs can be assessed as their bona fide attempts at correct answers or conclusions; and (b) conclusions about base LLMs' reasoning can generalize to post-trained LLMs optimized for successful instruction-following. We call for a critical re-examination of existing work that relies implicitly on these assumptions, and for future work to account for these methodological pitfalls. 

---
# TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs 

**Authors**: Shuyi Liu, Yuming Shang, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10375)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful framework for enhancing the capabilities of Large Language Models (LLMs) by integrating retrieval-based methods with generative models. As external knowledge repositories continue to expand and the parametric knowledge within models becomes outdated, a critical challenge for RAG systems is resolving conflicts between retrieved external information and LLMs' internal knowledge, which can significantly compromise the accuracy and reliability of generated content. However, existing approaches to conflict resolution typically operate at the token or semantic level, often leading to fragmented and partial understanding of factual discrepancies between LLMs' knowledge and context, particularly in knowledge-intensive tasks. To address this limitation, we propose TruthfulRAG, the first framework that leverages Knowledge Graphs (KGs) to resolve factual-level knowledge conflicts in RAG systems. Specifically, TruthfulRAG constructs KGs by systematically extracting triples from retrieved content, utilizes query-based graph retrieval to identify relevant knowledge, and employs entropy-based filtering mechanisms to precisely locate conflicting elements and mitigate factual inconsistencies, thereby enabling LLMs to generate faithful and accurate responses. Extensive experiments reveal that TruthfulRAG outperforms existing methods, effectively alleviating knowledge conflicts and improving the robustness and trustworthiness of RAG systems. 

---
# Knowledge Graphs Generation from Cultural Heritage Texts: Combining LLMs and Ontological Engineering for Scholarly Debates 

**Authors**: Andrea Schimmenti, Valentina Pasqual, Fabio Vitali, Marieke van Erp  

**Link**: [PDF](https://arxiv.org/pdf/2511.10354)  

**Abstract**: Cultural Heritage texts contain rich knowledge that is difficult to query systematically due to the challenges of converting unstructured discourse into structured Knowledge Graphs (KGs). This paper introduces ATR4CH (Adaptive Text-to-RDF for Cultural Heritage), a systematic five-step methodology for Large Language Model-based Knowledge Extraction from Cultural Heritage documents. We validate the methodology through a case study on authenticity assessment debates. Methodology - ATR4CH combines annotation models, ontological frameworks, and LLM-based extraction through iterative development: foundational analysis, annotation schema development, pipeline architecture, integration refinement, and comprehensive evaluation. We demonstrate the approach using Wikipedia articles about disputed items (documents, artifacts...), implementing a sequential pipeline with three LLMs (Claude Sonnet 3.7, Llama 3.3 70B, GPT-4o-mini). Findings - The methodology successfully extracts complex Cultural Heritage knowledge: 0.96-0.99 F1 for metadata extraction, 0.7-0.8 F1 for entity recognition, 0.65-0.75 F1 for hypothesis extraction, 0.95-0.97 for evidence extraction, and 0.62 G-EVAL for discourse representation. Smaller models performed competitively, enabling cost-effective deployment. Originality - This is the first systematic methodology for coordinating LLM-based extraction with Cultural Heritage ontologies. ATR4CH provides a replicable framework adaptable across CH domains and institutional resources. Research Limitations - The produced KG is limited to Wikipedia articles. While the results are encouraging, human oversight is necessary during post-processing. Practical Implications - ATR4CH enables Cultural Heritage institutions to systematically convert textual knowledge into queryable KGs, supporting automated metadata enrichment and knowledge discovery. 

---
# BhashaKritika: Building Synthetic Pretraining Data at Scale for Indic Languages 

**Authors**: Guduru Manoj, Neel Prabhanjan Rachamalla, Ashish Kulkarni, Gautam Rajeev, Jay Piplodiya, Arul Menezes, Shaharukh Khan, Souvik Rana, Manya Sah, Chandra Khatri, Shubham Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2511.10338)  

**Abstract**: In the context of pretraining of Large Language Models (LLMs), synthetic data has emerged as an alternative for generating high-quality pretraining data at scale. This is particularly beneficial in low-resource language settings where the benefits of recent LLMs have been unevenly distributed across languages. In this work, we present a systematic study on the generation and evaluation of synthetic multilingual pretraining data for Indic languages, where we construct a large-scale synthetic dataset BhashaKritika, comprising 540B tokens using 5 different techniques for 10 languages. We explore the impact of grounding generation in documents, personas, and topics. We analyze how language choice, both in the prompt instructions and document grounding, affects data quality, and we compare translations of English content with native generation in Indic languages. To support scalable and language-sensitive evaluation, we introduce a modular quality evaluation pipeline that integrates script and language detection, metadata consistency checks, n-gram repetition analysis, and perplexity-based filtering using KenLM models. Our framework enables robust quality control across diverse scripts and linguistic contexts. Empirical results through model runs reveal key trade-offs in generation strategies and highlight best practices for constructing effective multilingual corpora. 

---
# Rectify Evaluation Preference: Improving LLMs' Critique on Math Reasoning via Perplexity-aware Reinforcement Learning 

**Authors**: Changyuan Tian, Zhicong Lu, Shuang Qian, Nayu Liu, Peiguang Li, Li Jin, Leiyi Hu, Zhizhao Zeng, Sirui Wang, Ke Zeng, Zhi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.10303)  

**Abstract**: To improve Multi-step Mathematical Reasoning (MsMR) of Large Language Models (LLMs), it is crucial to obtain scalable supervision from the corpus by automatically critiquing mistakes in the reasoning process of MsMR and rendering a final verdict of the problem-solution. Most existing methods rely on crafting high-quality supervised fine-tuning demonstrations for critiquing capability enhancement and pay little attention to delving into the underlying reason for the poor critiquing performance of LLMs. In this paper, we orthogonally quantify and investigate the potential reason -- imbalanced evaluation preference, and conduct a statistical preference analysis. Motivated by the analysis of the reason, a novel perplexity-aware reinforcement learning algorithm is proposed to rectify the evaluation preference, elevating the critiquing capability. Specifically, to probe into LLMs' critiquing characteristics, a One-to-many Problem-Solution (OPS) benchmark is meticulously constructed to quantify the behavior difference of LLMs when evaluating the problem solutions generated by itself and others. Then, to investigate the behavior difference in depth, we conduct a statistical preference analysis oriented on perplexity and find an intriguing phenomenon -- ``LLMs incline to judge solutions with lower perplexity as correct'', which is dubbed as \textit{imbalanced evaluation preference}. To rectify this preference, we regard perplexity as the baton in the algorithm of Group Relative Policy Optimization, supporting the LLMs to explore trajectories that judge lower perplexity as wrong and higher perplexity as correct. Extensive experimental results on our built OPS and existing available critic benchmarks demonstrate the validity of our method. 

---
# Local Hybrid Retrieval-Augmented Document QA 

**Authors**: Paolo Astrino  

**Link**: [PDF](https://arxiv.org/pdf/2511.10297)  

**Abstract**: Organizations handling sensitive documents face a critical dilemma: adopt cloud-based AI systems that offer powerful question-answering capabilities but compromise data privacy, or maintain local processing that ensures security but delivers poor accuracy. We present a question-answering system that resolves this trade-off by combining semantic understanding with keyword precision, operating entirely on local infrastructure without internet access. Our approach demonstrates that organizations can achieve competitive accuracy on complex queries across legal, scientific, and conversational documents while keeping all data on their machines. By balancing two complementary retrieval strategies and using consumer-grade hardware acceleration, the system delivers reliable answers with minimal errors, letting banks, hospitals, and law firms adopt conversational document AI without transmitting proprietary information to external providers. This work establishes that privacy and performance need not be mutually exclusive in enterprise AI deployment. 

---
# MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models 

**Authors**: He Zhang, Wenqian Cui, Haoning Xu, Xiaohui Li, Lei Zhu, Shaohua Ma, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2511.10262)  

**Abstract**: Full-Duplex Speech Language Models (FD-SLMs) enable real-time, overlapping conversational interactions, offering a more dynamic user experience compared to traditional half-duplex models. However, existing benchmarks primarily focus on evaluating single-round interactions and conversational features, neglecting the complexities of multi-round communication and critical capabilities such as instruction following and safety. Evaluating FD-SLMs in multi-round settings poses significant challenges, including blurred turn boundaries in communication and context inconsistency during model inference. To address these gaps, we introduce MTR-DuplexBench, a novel benchmark that segments continuous full-duplex dialogues into discrete turns, enabling comprehensive, turn-by-turn evaluation of FD-SLMs across dialogue quality, conversational dynamics, instruction following, and safety. Experimental results reveal that current FD-SLMs face difficulties in maintaining consistent performance across multiple rounds and evaluation dimensions, highlighting the necessity and effectiveness of our proposed benchmark. The benchmark and code will be available in the future. 

---
# VocalNet-M2: Advancing Low-Latency Spoken Language Modeling via Integrated Multi-Codebook Tokenization and Multi-Token Prediction 

**Authors**: Yuhao Wang, Ziyang Cheng, Heyang Liu, Ronghua Wu, Qunshan Gu, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10232)  

**Abstract**: Current end-to-end spoken language models (SLMs) have made notable progress, yet they still encounter considerable response latency. This delay primarily arises from the autoregressive generation of speech tokens and the reliance on complex flow-matching models for speech synthesis. To overcome this, we introduce VocalNet-M2, a novel low-latency SLM that integrates a multi-codebook tokenizer and a multi-token prediction (MTP) strategy. Our model directly generates multi-codebook speech tokens, thus eliminating the need for a latency-inducing flow-matching model. Furthermore, our MTP strategy enhances generation efficiency and improves overall performance. Extensive experiments demonstrate that VocalNet-M2 achieves a substantial reduction in first chunk latency (from approximately 725ms to 350ms) while maintaining competitive performance across mainstream SLMs. This work also provides a comprehensive comparison of single-codebook and multi-codebook strategies, offering valuable insights for developing efficient and high-performance SLMs for real-time interactive applications. 

---
# LangGPS: Language Separability Guided Data Pre-Selection for Joint Multilingual Instruction Tuning 

**Authors**: Yangfan Ye, Xiaocheng Feng, Xiachong Feng, Lei Huang, Weitao Ma, Qichen Hong, Yunfei Lu, Duyu Tang, Dandan Tu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2511.10229)  

**Abstract**: Joint multilingual instruction tuning is a widely adopted approach to improve the multilingual instruction-following ability and downstream performance of large language models (LLMs), but the resulting multilingual capability remains highly sensitive to the composition and selection of the training data. Existing selection methods, often based on features like text quality, diversity, or task relevance, typically overlook the intrinsic linguistic structure of multilingual data. In this paper, we propose LangGPS, a lightweight two-stage pre-selection framework guided by language separability which quantifies how well samples in different languages can be distinguished in the model's representation space. LangGPS first filters training data based on separability scores and then refines the subset using existing selection methods. Extensive experiments across six benchmarks and 22 languages demonstrate that applying LangGPS on top of existing selection methods improves their effectiveness and generalizability in multilingual training, especially for understanding tasks and low-resource languages. Further analysis reveals that highly separable samples facilitate the formation of clearer language boundaries and support faster adaptation, while low-separability samples tend to function as bridges for cross-lingual alignment. Besides, we also find that language separability can serve as an effective signal for multilingual curriculum learning, where interleaving samples with diverse separability levels yields stable and generalizable gains. Together, we hope our work offers a new perspective on data utility in multilingual contexts and support the development of more linguistically informed LLMs. 

---
# Persona-Aware Alignment Framework for Personalized Dialogue Generation 

**Authors**: Guanrong Li, Xinyu Liu, Zhen Wu, Xinyu Dai  

**Link**: [PDF](https://arxiv.org/pdf/2511.10215)  

**Abstract**: Personalized dialogue generation aims to leverage persona profiles and dialogue history to generate persona-relevant and consistent responses. Mainstream models typically rely on token-level language model training with persona dialogue data, such as Next Token Prediction, to implicitly achieve personalization, making these methods tend to neglect the given personas and generate generic responses. To address this issue, we propose a novel Persona-Aware Alignment Framework (PAL), which directly treats persona alignment as the training objective of dialogue generation. Specifically, PAL employs a two-stage training method including Persona-aware Learning and Persona Alignment, equipped with an easy-to-use inference strategy Select then Generate, to improve persona sensitivity and generate more persona-relevant responses at the semantics level. Through extensive experiments, we demonstrate that our framework outperforms many state-of-the-art personalized dialogue methods and large language models. 

---
# EffiReason-Bench: A Unified Benchmark for Evaluating and Advancing Efficient Reasoning in Large Language Models 

**Authors**: Junquan Huang, Haotian Wu, Yubo Gao, Yibo Yan, Junyan Zhang, Yonghua Hei, Song Dai, Jie Zhang, Puay Siew Tan, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2511.10201)  

**Abstract**: Large language models (LLMs) with Chain-of-Thought (CoT) prompting achieve strong reasoning but often produce unnecessarily long explanations, increasing cost and sometimes reducing accuracy. Fair comparison of efficiency-oriented approaches is hindered by fragmented evaluation practices. We introduce EffiReason-Bench, a unified benchmark for rigorous cross-paradigm evaluation of efficient reasoning methods across three categories: Reasoning Blueprints, Dynamic Execution, and Post-hoc Refinement. To enable step-by-step evaluation, we construct verified CoT annotations for CommonsenseQA and LogiQA via a pipeline that enforces standardized reasoning structures, comprehensive option-wise analysis, and human verification. We evaluate 7 methods across 6 open-source LLMs (1B-70B) on 4 datasets spanning mathematics, commonsense, and logic, and propose the E3-Score, a principled metric inspired by economic trade-off modeling that provides smooth, stable evaluation without discontinuities or heavy reliance on heuristics. Experiments show that no single method universally dominates; optimal strategies depend on backbone scale, task complexity, and architecture. 

---
# Text2SQL-Flow: A Robust SQL-Aware Data Augmentation Framework for Text-to-SQL 

**Authors**: Qifeng Cai, Hao Liang, Chang Xu, Tao Xie, Wentao Zhang, Bin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2511.10192)  

**Abstract**: The data-centric paradigm has become pivotal in AI, especially for Text-to-SQL, where performance is limited by scarce, simplistic, and low-diversity datasets. To address this, we propose Text2SQL-Flow, a SQL-aware data augmentation framework that generates large-scale, semantically valid, and structurally diverse Text-to-SQL pairs from minimal seed data. It operates across six augmentation dimensions and integrates an end-to-end pipeline featuring SQL execution verification, natural language question generation, chain-of-thought reasoning traces, and data classification. A modular Database Manager ensures cross-database compatibility and scalability. Using this framework, we build SQLFlow, a high-quality dataset of 89,544 annotated examples. We evaluate SQLFlow in two settings: (1) For open-source LLMs, fine-tuning on SQLFlow consistently improves performance across benchmarks under the same data budget. (2) For closed-source LLMs, we introduce a masked alignment retrieval method that treats SQLFlow as both knowledge base and training data for the retriever. This enables structure-aware example matching by modeling fine-grained alignments between questions and SQL queries. Experiments show our retrieval strategy outperforms existing methods, underscoring the value of SQLFlow's high-fidelity data and our novel technique. Our work establishes a scalable, data-centric foundation for advancing Text-to-SQL systems and highlights the critical role of high-quality structured data in modern AI. 

---
# Beyond the Black Box: Demystifying Multi-Turn LLM Reasoning with VISTA 

**Authors**: Yiran Zhang, Mingyang Lin, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2511.10182)  

**Abstract**: Recent research has increasingly focused on the reasoning capabilities of Large Language Models (LLMs) in multi-turn interactions, as these scenarios more closely mirror real-world problem-solving. However, analyzing the intricate reasoning processes within these interactions presents a significant challenge due to complex contextual dependencies and a lack of specialized visualization tools, leading to a high cognitive load for researchers. To address this gap, we present VISTA, an web-based Visual Interactive System for Textual Analytics in multi-turn reasoning tasks. VISTA allows users to visualize the influence of context on model decisions and interactively modify conversation histories to conduct "what-if" analyses across different models. Furthermore, the platform can automatically parse a session and generate a reasoning dependency tree, offering a transparent view of the model's step-by-step logical path. By providing a unified and interactive framework, VISTA significantly reduces the complexity of analyzing reasoning chains, thereby facilitating a deeper understanding of the capabilities and limitations of current LLMs. The platform is open-source and supports easy integration of custom benchmarks and local models. 

---
# Generalizing to Unseen Disaster Events: A Causal View 

**Authors**: Philipp Seeberger, Steffen Freisinger, Tobias Bocklet, Korbinian Riedhammer  

**Link**: [PDF](https://arxiv.org/pdf/2511.10120)  

**Abstract**: Due to the rapid growth of social media platforms, these tools have become essential for monitoring information during ongoing disaster events. However, extracting valuable insights requires real-time processing of vast amounts of data. A major challenge in existing systems is their exposure to event-related biases, which negatively affects their ability to generalize to emerging events. While recent advancements in debiasing and causal learning offer promising solutions, they remain underexplored in the disaster event domain. In this work, we approach bias mitigation through a causal lens and propose a method to reduce event- and domain-related biases, enhancing generalization to future events. Our approach outperforms multiple baselines by up to +1.9% F1 and significantly improves a PLM-based classifier across three disaster classification tasks. 

---
# On the Military Applications of Large Language Models 

**Authors**: Satu Johansson, Taneli Riihonen  

**Link**: [PDF](https://arxiv.org/pdf/2511.10093)  

**Abstract**: In this paper, military use cases or applications and implementation thereof are considered for natural language processing and large language models, which have broken into fame with the invention of the generative pre-trained transformer (GPT) and the extensive foundation model pretraining done by OpenAI for ChatGPT and others. First, we interrogate a GPT-based language model (viz. Microsoft Copilot) to make it reveal its own knowledge about their potential military applications and then critically assess the information. Second, we study how commercial cloud services (viz. Microsoft Azure) could be used readily to build such applications and assess which of them are feasible. We conclude that the summarization and generative properties of language models directly facilitate many applications at large and other features may find particular uses. 

---
# ELYADATA & LIA at NADI 2025: ASR and ADI Subtasks 

**Authors**: Haroun Elleuch, Youssef Saidi, Salima Mdhaffar, Yannick Estève, Fethi Bougares  

**Link**: [PDF](https://arxiv.org/pdf/2511.10090)  

**Abstract**: This paper describes Elyadata \& LIA's joint submission to the NADI multi-dialectal Arabic Speech Processing 2025. We participated in the Spoken Arabic Dialect Identification (ADI) and multi-dialectal Arabic ASR subtasks. Our submission ranked first for the ADI subtask and second for the multi-dialectal Arabic ASR subtask among all participants. Our ADI system is a fine-tuned Whisper-large-v3 encoder with data augmentation. This system obtained the highest ADI accuracy score of \textbf{79.83\%} on the official test set. For multi-dialectal Arabic ASR, we fine-tuned SeamlessM4T-v2 Large (Egyptian variant) separately for each of the eight considered dialects. Overall, we obtained an average WER and CER of \textbf{38.54\%} and \textbf{14.53\%}, respectively, on the test set. Our results demonstrate the effectiveness of large pre-trained speech models with targeted fine-tuning for Arabic speech processing. 

---
# Format Matters: The Robustness of Multimodal LLMs in Reviewing Evidence from Tables and Charts 

**Authors**: Xanh Ho, Yun-Ang Wu, Sunisth Kumar, Florian Boudin, Atsuhiro Takasu, Akiko Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2511.10075)  

**Abstract**: With the growing number of submitted scientific papers, there is an increasing demand for systems that can assist reviewers in evaluating research claims. Experimental results are a core component of scientific work, often presented in varying formats such as tables or charts. Understanding how robust current multimodal large language models (multimodal LLMs) are at verifying scientific claims across different evidence formats remains an important and underexplored challenge. In this paper, we design and conduct a series of experiments to assess the ability of multimodal LLMs to verify scientific claims using both tables and charts as evidence. To enable this evaluation, we adapt two existing datasets of scientific papers by incorporating annotations and structures necessary for a multimodal claim verification task. Using this adapted dataset, we evaluate 12 multimodal LLMs and find that current models perform better with table-based evidence while struggling with chart-based evidence. We further conduct human evaluations and observe that humans maintain strong performance across both formats, unlike the models. Our analysis also reveals that smaller multimodal LLMs (under 8B) show weak correlation in performance between table-based and chart-based tasks, indicating limited cross-modal generalization. These findings highlight a critical gap in current models' multimodal reasoning capabilities. We suggest that future multimodal LLMs should place greater emphasis on improving chart understanding to better support scientific claim verification. 

---
# ADI-20: Arabic Dialect Identification dataset and models 

**Authors**: Haroun Elleuch, Salima Mdhaffar, Yannick Estève, Fethi Bougares  

**Link**: [PDF](https://arxiv.org/pdf/2511.10070)  

**Abstract**: We present ADI-20, an extension of the previously published ADI-17 Arabic Dialect Identification (ADI) dataset. ADI-20 covers all Arabic-speaking countries' dialects. It comprises 3,556 hours from 19 Arabic dialects in addition to Modern Standard Arabic (MSA). We used this dataset to train and evaluate various state-of-the-art ADI systems. We explored fine-tuning pre-trained ECAPA-TDNN-based models, as well as Whisper encoder blocks coupled with an attention pooling layer and a classification dense layer. We investigated the effect of (i) training data size and (ii) the model's number of parameters on identification performance. Our results show a small decrease in F1 score while using only 30% of the original training data. We open-source our collected data and trained models to enable the reproduction of our work, as well as support further research in ADI. 

---
# GraphIF: Enhancing Multi-Turn Instruction Following for Large Language Models with Relation Graph Prompt 

**Authors**: Zhenhe Li, Can Lin, Ling Zheng, Wen-Da Wei, Junli Liang, Qi Song  

**Link**: [PDF](https://arxiv.org/pdf/2511.10051)  

**Abstract**: Multi-turn instruction following is essential for building intelligent conversational systems that can consistently adhere to instructions across dialogue turns. However, existing approaches to enhancing multi-turn instruction following primarily rely on collecting or generating large-scale multi-turn dialogue datasets to fine-tune large language models (LLMs), which treat each response generation as an isolated task and fail to explicitly incorporate multi-turn instruction following into the optimization objectives. As a result, instruction-tuned LLMs often struggle with complex long-distance constraints. In multi-turn dialogues, relational constraints across turns can be naturally modeled as labeled directed edges, making graph structures particularly suitable for modeling multi-turn instruction following. Despite this potential, leveraging graph structures to enhance the multi-turn instruction following capabilities of LLMs remains unexplored. To bridge this gap, we propose GraphIF, a plug-and-play framework that models multi-turn dialogues as directed relation graphs and leverages graph prompts to enhance the instruction following capabilities of LLMs. GraphIF comprises three key components: (1) an agent-based relation extraction module that captures inter-turn semantic relations via action-triggered mechanisms to construct structured graphs; (2) a relation graph prompt generation module that converts structured graph information into natural language prompts; and (3) a response rewriting module that refines initial LLM outputs using the generated graph prompts. Extensive experiments on two long multi-turn dialogue datasets demonstrate that GraphIF can be seamlessly integrated into instruction-tuned LLMs and leads to significant improvements across all four multi-turn instruction-following evaluation metrics. 

---
# Do Language Models Associate Sound with Meaning? A Multimodal Study of Sound Symbolism 

**Authors**: Jinhong Jeong, Sunghyun Lee, Jaeyoung Lee, Seonah Han, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.10045)  

**Abstract**: Sound symbolism is a linguistic concept that refers to non-arbitrary associations between phonetic forms and their meanings. We suggest that this can be a compelling probe into how Multimodal Large Language Models (MLLMs) interpret auditory information in human languages. We investigate MLLMs' performance on phonetic iconicity across textual (orthographic and IPA) and auditory forms of inputs with up to 25 semantic dimensions (e.g., sharp vs. round), observing models' layer-wise information processing by measuring phoneme-level attention fraction scores. To this end, we present LEX-ICON, an extensive mimetic word dataset consisting of 8,052 words from four natural languages (English, French, Japanese, and Korean) and 2,930 systematically constructed pseudo-words, annotated with semantic features applied across both text and audio modalities. Our key findings demonstrate (1) MLLMs' phonetic intuitions that align with existing linguistic research across multiple semantic dimensions and (2) phonosemantic attention patterns that highlight models' focus on iconic phonemes. These results bridge domains of artificial intelligence and cognitive linguistics, providing the first large-scale, quantitative analyses of phonetic iconicity in terms of MLLMs' interpretability. 

---
# ScaleFormer: Span Representation Cumulation for Long-Context Transformer 

**Authors**: Jiangshu Du, Wenpeng Yin, Philip Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.10029)  

**Abstract**: The quadratic complexity of standard self-attention severely limits the application of Transformer-based models to long-context tasks. While efficient Transformer variants exist, they often require architectural changes and costly pre-training from scratch. To circumvent this, we propose ScaleFormer(Span Representation Cumulation for Long-Context Transformer) - a simple and effective plug-and-play framework that adapts off-the-shelf pre-trained encoder-decoder models to process long sequences without requiring architectural modifications. Our approach segments long inputs into overlapping chunks and generates a compressed, context-aware representation for the decoder. The core of our method is a novel, parameter-free fusion mechanism that endows each chunk's representation with structural awareness of its position within the document. It achieves this by enriching each chunk's boundary representations with cumulative context vectors from all preceding and succeeding chunks. This strategy provides the model with a strong signal of the document's narrative flow, achieves linear complexity, and enables pre-trained models to reason effectively over long-form text. Experiments on long-document summarization show that our method is highly competitive with and often outperforms state-of-the-art approaches without requiring architectural modifications or external retrieval mechanisms. 

---
# PustakAI: Curriculum-Aligned and Interactive Textbooks Using Large Language Models 

**Authors**: Shivam Sharma, Riya Naik, Tejas Gawas, Heramb Patil, Kunal Korgaonkar  

**Link**: [PDF](https://arxiv.org/pdf/2511.10002)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in understanding and generating human-like content. This has revolutionized various sectors such as healthcare, software development, and education. In education, LLMs offer potential for personalized and interactive learning experiences, especially in regions with limited teaching resources. However, adapting these models effectively to curriculum-specific content, such as the National Council of Educational Research and Training (NCERT) syllabus in India, presents unique challenges in terms of accuracy, alignment, and pedagogical relevance. In this paper, we present the framework "PustakAI"\footnote{Pustak means `book' in many Indian languages.} for the design and evaluation of a novel question-answering dataset "NCERT-QA" aligned with the NCERT curriculum for English and Science subjects of grades 6 to 8. We classify the curated QA pairs as Factoid, Inferential, and Others (evaluative and reasoning). We evaluate the dataset with various prompting techniques, such as meta-prompt, few-shot, and CoT-style prompting, using diverse evaluation metrics to understand which approach aligns more efficiently with the structure and demands of the curriculum. Along with the usability of the dataset, we analyze the strengths and limitations of current open-source LLMs (Gemma3:1b, Llama3.2:3b, and Nemotron-mini:4b) and high-end LLMs (Llama-4-Scout-17B and Deepseek-r1-70B) as AI-based learning tools in formal education systems. 

---
# FinNuE: Exposing the Risks of Using BERTScore for Numerical Semantic Evaluation in Finance 

**Authors**: Yu-Shiang Huang, Yun-Yu Lee, Tzu-Hsin Chou, Che Lin, Chuan-Ju Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.09997)  

**Abstract**: BERTScore has become a widely adopted metric for evaluating semantic similarity between natural language sentences. However, we identify a critical limitation: BERTScore exhibits low sensitivity to numerical variation, a significant weakness in finance where numerical precision directly affects meaning (e.g., distinguishing a 2% gain from a 20% loss). We introduce FinNuE, a diagnostic dataset constructed with controlled numerical perturbations across earnings calls, regulatory filings, social media, and news articles. Using FinNuE, demonstrate that BERTScore fails to distinguish semantically critical numerical differences, often assigning high similarity scores to financially divergent text pairs. Our findings reveal fundamental limitations of embedding-based metrics for finance and motivate numerically-aware evaluation frameworks for financial NLP. 

---
# Language Drift in Multilingual Retrieval-Augmented Generation: Characterization and Decoding-Time Mitigation 

**Authors**: Bo Li, Zhenghua Xu, Rui Xie  

**Link**: [PDF](https://arxiv.org/pdf/2511.09984)  

**Abstract**: Multilingual Retrieval-Augmented Generation (RAG) enables large language models (LLMs) to perform knowledge-intensive tasks in multilingual settings by leveraging retrieved documents as external evidence. However, when the retrieved evidence differs in language from the user query and in-context exemplars, the model often exhibits language drift by generating responses in an unintended language. This phenomenon is especially pronounced during reasoning-intensive decoding, such as Chain-of-Thought (CoT) generation, where intermediate steps introduce further language instability. In this paper, we systematically study output language drift in multilingual RAG across multiple datasets, languages, and LLM backbones. Our controlled experiments reveal that the drift results not from comprehension failure but from decoder-level collapse, where dominant token distributions and high-frequency English patterns dominate the intended generation language. We further observe that English serves as a semantic attractor under cross-lingual conditions, emerging as both the strongest interference source and the most frequent fallback language.
To mitigate this, we propose Soft Constrained Decoding (SCD), a lightweight, training-free decoding strategy that gently steers generation toward the target language by penalizing non-target-language tokens. SCD is model-agnostic and can be applied to any generation algorithm without modifying the architecture or requiring additional data. Experiments across three multilingual datasets and multiple typologically diverse languages show that SCD consistently improves language alignment and task performance, providing an effective and generalizable solution in multilingual RAG. 

---
# Modeling Uncertainty Trends for Timely Retrieval in Dynamic RAG 

**Authors**: Bo Li, Tian Tian, Zhenghua Xu, Hao Cheng, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2511.09980)  

**Abstract**: Dynamic retrieval-augmented generation (RAG) allows large language models (LLMs) to fetch external knowledge on demand, offering greater adaptability than static RAG. A central challenge in this setting lies in determining the optimal timing for retrieval. Existing methods often trigger retrieval based on low token-level confidence, which may lead to delayed intervention after errors have already propagated. We introduce Entropy-Trend Constraint (ETC), a training-free method that determines optimal retrieval timing by modeling the dynamics of token-level uncertainty. Specifically, ETC utilizes first- and second-order differences of the entropy sequence to detect emerging uncertainty trends, enabling earlier and more precise retrieval. Experiments on six QA benchmarks with three LLM backbones demonstrate that ETC consistently outperforms strong baselines while reducing retrieval frequency. ETC is particularly effective in domain-specific scenarios, exhibiting robust generalization capabilities. Ablation studies and qualitative analyses further confirm that trend-aware uncertainty modeling yields more effective retrieval timing. The method is plug-and-play, model-agnostic, and readily integrable into existing decoding pipelines. Implementation code is included in the supplementary materials. 

---
# NumPert: Numerical Perturbations to Probe Language Models for Veracity Prediction 

**Authors**: Peter Røysland Aarnes, Vinay Setty  

**Link**: [PDF](https://arxiv.org/pdf/2511.09971)  

**Abstract**: Large language models show strong performance on knowledge intensive tasks such as fact-checking and question answering, yet they often struggle with numerical reasoning. We present a systematic evaluation of state-of-the-art models for veracity prediction on numerical claims and evidence pairs using controlled perturbations, including label-flipping probes, to test robustness. Our results indicate that even leading proprietary systems experience accuracy drops of up to 62\% under certain perturbations. No model proves to be robust across all conditions. We further find that increasing context length generally reduces accuracy, but when extended context is enriched with perturbed demonstrations, most models substantially recover. These findings highlight critical limitations in numerical fact-checking and suggest that robustness remains an open challenge for current language models. 

---
# REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering 

**Authors**: Yijie Zhu, Haojie Zhou, Wanting Hong, Tailin Liu, Ning Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.09966)  

**Abstract**: Retrieval-augmented generation (RAG) has been extensively employed to mitigate hallucinations in large language models (LLMs). However, existing methods for multi-hop reasoning tasks often lack global planning, increasing the risk of falling into local reasoning impasses. Insufficient exploitation of retrieved content and the neglect of latent clues fail to ensure the accuracy of reasoning outcomes. To overcome these limitations, we propose Recursive Evaluation and Adaptive Planning (REAP), whose core idea is to explicitly maintain structured sub-tasks and facts related to the current task through the Sub-task Planner (SP) and Fact Extractor (FE) modules. SP maintains a global perspective, guiding the overall reasoning direction and evaluating the task state based on the outcomes of FE, enabling dynamic optimization of the task-solving trajectory. FE performs fine-grained analysis over retrieved content to extract reliable answers and clues. These two modules incrementally enrich a logically coherent representation of global knowledge, enhancing the reliability and the traceability of the reasoning process. Furthermore, we propose a unified task paradigm design that enables effective multi-task fine-tuning, significantly enhancing SP's performance on complex, data-scarce tasks. We conduct extensive experiments on multiple public multi-hop datasets, and the results demonstrate that our method significantly outperforms existing RAG methods in both in-domain and out-of-domain settings, validating its effectiveness in complex multi-hop reasoning tasks. 

---
# Leveraging Large Language Models for Identifying Knowledge Components 

**Authors**: Canwen Wang, Jionghao Lin, Kenneth R. Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2511.09935)  

**Abstract**: Knowledge Components (KCs) are foundational to adaptive learning systems, but their manual identification by domain experts is a significant bottleneck. While Large Language Models (LLMs) offer a promising avenue for automating this process, prior research has been limited to small datasets and has been shown to produce superfluous, redundant KC labels. This study addresses these limitations by first scaling a "simulated textbook" LLM prompting strategy (using GPT-4o-mini) to a larger dataset of 646 multiple-choice questions. We found that this initial automated approach performed significantly worse than an expert-designed KC model (RMSE 0.4285 vs. 0.4206) and generated an excessive number of KCs (569 vs. 101). To address the issue of redundancy, we proposed and evaluated a novel method for merging semantically similar KC labels based on their cosine similarity. This merging strategy significantly improved the model's performance; a model using a cosine similarity threshold of 0.8 achieved the best result, reducing the KC count to 428 and improving the RMSE to 0.4259. This demonstrates that while scaled LLM generation alone is insufficient, combining it with a semantic merging technique offers a viable path toward automating and refining KC identification. 

---
# MINDS: A Cross-cultural Dialogue Corpus for Social Norm Classification and Adherence Detection 

**Authors**: Pritish Sahu, Anirudh Som, Dimitra Vergyri, Ajay Divakaran  

**Link**: [PDF](https://arxiv.org/pdf/2511.09918)  

**Abstract**: Social norms are implicit, culturally grounded expectations that guide interpersonal communication. Unlike factual commonsense, norm reasoning is subjective, context-dependent, and varies across cultures, posing challenges for computational models. Prior works provide valuable normative annotations but mostly target isolated utterances or synthetic dialogues, limiting their ability to capture the fluid, multi-turn nature of real-world conversations. In this work, we present Norm-RAG, a retrieval-augmented, agentic framework for nuanced social norm inference in multi-turn dialogues. Norm-RAG models utterance-level attributes including communicative intent, speaker roles, interpersonal framing, and linguistic cues and grounds them in structured normative documentation retrieved via a novel Semantic Chunking approach. This enables interpretable and context-aware reasoning about norm adherence and violation across multilingual dialogues. We further introduce MINDS (Multilingual Interactions with Norm-Driven Speech), a bilingual dataset comprising 31 multi-turn Mandarin-English and Spanish-English conversations. Each turn is annotated for norm category and adherence status using multi-annotator consensus, reflecting cross-cultural and realistic norm expression. Our experiments show that Norm-RAG improves norm detection and generalization, demonstrates improved performance for culturally adaptive and socially intelligent dialogue systems. 

---
# HI-TransPA: Hearing Impairments Translation Personal Assistant 

**Authors**: Zhiming Ma, Shiyu Gan, Junhao Zhao, Xianming Li, Qingyun Pan, Peidong Wang, Mingjun Pan, Yuhao Mo, Jiajie Cheng, Chengxin Chen, Zhonglun Cao, Chonghan Liu, Shi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.09915)  

**Abstract**: To provide a unified and flexible solution for daily communication among hearing-impaired individuals, we introduce the Omni-Model paradigm into assistive technology and present HI-TransPA, an instruction-driven audio-visual personal assistant. The model fuses indistinct speech with high-frame-rate lip dynamics, enabling both translation and dialogue within a single multimodal framework. To tackle the challenges of noisy and heterogeneous raw data and the limited adaptability of existing Omni-Models to hearing-impaired speech, we construct a comprehensive preprocessing and curation pipeline that detects facial landmarks, isolates and stabilizes the lip region, and quantitatively assesses multimodal sample quality. These quality scores guide a curriculum learning strategy that first trains on clean, high-confidence samples and progressively incorporates harder cases to strengthen model robustness. We further adopt a SigLIP encoder combined with a Unified 3D-Resampler to efficiently encode high-frame-rate lip motion. Experiments on our purpose-built HI-Dialogue dataset show that HI-TransPA achieves state-of-the-art performance in both literal accuracy and semantic fidelity. This work establishes a foundation for applying Omni-Models to assistive communication technology, providing an end-to-end modeling framework and essential processing tools for future research. 

---
# EnchTable: Unified Safety Alignment Transfer in Fine-tuned Large Language Models 

**Authors**: Jialin Wu, Kecen Li, Zhicong Huang, Xinfeng Li, Xiaofeng Wang, Cheng Hong  

**Link**: [PDF](https://arxiv.org/pdf/2511.09880)  

**Abstract**: Many machine learning models are fine-tuned from large language models (LLMs) to achieve high performance in specialized domains like code generation, biomedical analysis, and mathematical problem solving. However, this fine-tuning process often introduces a critical vulnerability: the systematic degradation of safety alignment, undermining ethical guidelines and increasing the risk of harmful outputs. Addressing this challenge, we introduce EnchTable, a novel framework designed to transfer and maintain safety alignment in downstream LLMs without requiring extensive retraining. EnchTable leverages a Neural Tangent Kernel (NTK)-based safety vector distillation method to decouple safety constraints from task-specific reasoning, ensuring compatibility across diverse model architectures and sizes. Additionally, our interference-aware merging technique effectively balances safety and utility, minimizing performance compromises across various task domains. We implemented a fully functional prototype of EnchTable on three different task domains and three distinct LLM architectures, and evaluated its performance through extensive experiments on eleven diverse datasets, assessing both utility and model safety. Our evaluations include LLMs from different vendors, demonstrating EnchTable's generalization capability. Furthermore, EnchTable exhibits robust resistance to static and dynamic jailbreaking attacks, outperforming vendor-released safety models in mitigating adversarial prompts. Comparative analyses with six parameter modification methods and two inference-time alignment baselines reveal that EnchTable achieves a significantly lower unsafe rate, higher utility score, and universal applicability across different task domains. Additionally, we validate EnchTable can be seamlessly integrated into various deployment pipelines without significant overhead. 

---
# HierRouter: Coordinated Routing of Specialized Large Language Models via Reinforcement Learning 

**Authors**: Nikunj Gupta, Bill Guo, Rajgopal Kannan, Viktor K. Prasanna  

**Link**: [PDF](https://arxiv.org/pdf/2511.09873)  

**Abstract**: Large Language Models (LLMs) deliver state-of-the-art performance across many tasks but impose high computational and memory costs, limiting their deployment in resource-constrained or real-time settings. To address this, we propose HierRouter, a hierarchical routing approach that dynamically assembles inference pipelines from a pool of specialized, lightweight language models. Formulated as a finite-horizon Markov Decision Process (MDP), our approach trains a Proximal Policy Optimization (PPO)-based reinforcement learning agent to iteratively select which models to invoke at each stage of multi-hop inference. The agent conditions on the evolving context and accumulated cost to make context-aware routing decisions. Experiments with three open-source candidate LLMs across six benchmarks, including QA, code generation, and mathematical reasoning, show that HierRouter improves response quality by up to 2.4x compared to using individual models independently, while incurring only a minimal additional inference cost on average. These results highlight the promise of hierarchical routing for cost-efficient, high-performance LLM inference. All codes can be found here this https URL Nikunj-Gupta/hierouter. 

---
# In-Token Rationality Optimization: Towards Accurate and Concise LLM Reasoning via Self-Feedback 

**Authors**: Mingye Zhu, Yi Liu, Zheren Fu, Quan Wang, Yongdong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.09865)  

**Abstract**: Training Large Language Models (LLMs) for chain-of-thought reasoning presents a significant challenge: supervised fine-tuning on a single "golden" rationale hurts generalization as it penalizes equally valid alternatives, whereas reinforcement learning with verifiable rewards struggles with credit assignment and prohibitive computational cost. To tackle these limitations, we introduce InTRO (In-Token Rationality Optimization), a new framework that enables both token-level exploration and self-feedback for accurate and concise reasoning. Instead of directly optimizing an intractable objective over all valid reasoning paths, InTRO leverages correction factors-token-wise importance weights estimated by the information discrepancy between the generative policy and its answer-conditioned counterpart, for informative next token selection. This approach allows the model to perform token-level exploration and receive self-generated feedback within a single forward pass, ultimately encouraging accurate and concise rationales. Across six math-reasoning benchmarks, InTRO consistently outperforms other baselines, raising solution accuracy by up to 20% relative to the base model. Its chains of thought are also notably more concise, exhibiting reduced verbosity. Beyond this, InTRO enables cross-domain transfer, successfully adapting to out-of-domain reasoning tasks that extend beyond the realm of mathematics, demonstrating robust generalization. 

---
# TermGPT: Multi-Level Contrastive Fine-Tuning for Terminology Adaptation in Legal and Financial Domain 

**Authors**: Yidan Sun, Mengying Zhu, Feiyue Chen, Yangyang Wu, Xiaolei Dan, Mengyuan Yang, Xiaolin Zheng, Shenglin Ben  

**Link**: [PDF](https://arxiv.org/pdf/2511.09854)  

**Abstract**: Large language models (LLMs) have demonstrated impressive performance in text generation tasks; however, their embedding spaces often suffer from the isotropy problem, resulting in poor discrimination of domain-specific terminology, particularly in legal and financial contexts. This weakness in terminology-level representation can severely hinder downstream tasks such as legal judgment prediction or financial risk analysis, where subtle semantic distinctions are critical. To address this problem, we propose TermGPT, a multi-level contrastive fine-tuning framework designed for terminology adaptation. We first construct a sentence graph to capture semantic and structural relations, and generate semantically consistent yet discriminative positive and negative samples based on contextual and topological cues. We then devise a multi-level contrastive learning approach at both the sentence and token levels, enhancing global contextual understanding and fine-grained terminology discrimination. To support robust evaluation, we construct the first financial terminology dataset derived from official regulatory documents. Experiments show that TermGPT outperforms existing baselines in term discrimination tasks within the finance and legal domains. 

---
# Answering Students' Questions on Course Forums Using Multiple Chain-of-Thought Reasoning and Finetuning RAG-Enabled LLM 

**Authors**: Neo Wang, Sonit Singh  

**Link**: [PDF](https://arxiv.org/pdf/2511.09831)  

**Abstract**: The course forums are increasingly significant and play vital role in facilitating student discussions and answering their questions related to the course. It provides a platform for students to post their questions related to the content and admin issues related to the course. However, there are several challenges due to the increase in the number of students enrolled in the course. The primary challenge is that students' queries cannot be responded immediately and the instructors have to face lots of repetitive questions. To mitigate these issues, we propose a question answering system based on large language model with retrieval augmented generation (RAG) method. This work focuses on designing a question answering system with open source Large Language Model (LLM) and fine-tuning it on the relevant course dataset. To further improve the performance, we use a local knowledge base and applied RAG method to retrieve relevant documents relevant to students' queries, where the local knowledge base contains all the course content. To mitigate the hallucination of LLMs, We also integrate it with multi chain-of-thought reasoning to overcome the challenge of hallucination in LLMs. In this work, we experiment fine-tuned LLM with RAG method on the HotpotQA dataset. The experimental results demonstrate that the fine-tuned LLM with RAG method has a strong performance on question answering task. 

---
# Improving Graduate Outcomes by Identifying Skills Gaps and Recommending Courses Based on Career Interests 

**Authors**: Rahul Soni, Basem Suleiman, Sonit Singh  

**Link**: [PDF](https://arxiv.org/pdf/2511.09819)  

**Abstract**: This paper aims to address the challenge of selecting relevant courses for students by proposing the design and development of a course recommendation system. The course recommendation system utilises a combination of data analytics techniques and machine learning algorithms to recommend courses that align with current industry trends and requirements. In order to provide customised suggestions, the study entails the design and implementation of an extensive algorithmic framework that combines machine learning methods, user preferences, and academic criteria. The system employs data mining and collaborative filtering techniques to examine past courses and individual career goals in order to provide course recommendations. Moreover, to improve the accessibility and usefulness of the recommendation system, special attention is given to the development of an easy-to-use front-end interface. The front-end design prioritises visual clarity, interaction, and simplicity through iterative prototyping and user input revisions, guaranteeing a smooth and captivating user experience. We refined and optimised the proposed system by incorporating user feedback, ensuring that it effectively meets the needs and preferences of its target users. The proposed course recommendation system could be a useful tool for students, instructors, and career advisers to use in promoting lifelong learning and professional progression as it fills the gap between university learning and industry expectations. We hope that the proposed course recommendation system will help university students in making data-drive and industry-informed course decisions, in turn, improving graduate outcomes for the university sector. 

---
# Khmer Spellchecking: A Holistic Approach 

**Authors**: Marry Kong, Rina Buoy, Sovisal Chenda, Nguonly Taing  

**Link**: [PDF](https://arxiv.org/pdf/2511.09812)  

**Abstract**: Compared to English and other high-resource languages, spellchecking for Khmer remains an unresolved problem due to several challenges. First, there are misalignments between words in the lexicon and the word segmentation model. Second, a Khmer word can be written in different forms. Third, Khmer compound words are often loosely and easily formed, and these compound words are not always found in the lexicon. Fourth, some proper nouns may be flagged as misspellings due to the absence of a Khmer named-entity recognition (NER) model. Unfortunately, existing solutions do not adequately address these challenges. This paper proposes a holistic approach to the Khmer spellchecking problem by integrating Khmer subword segmentation, Khmer NER, Khmer grapheme-to-phoneme (G2P) conversion, and a Khmer language model to tackle these challenges, identify potential correction candidates, and rank the most suitable candidate. Experimental results show that the proposed approach achieves a state-of-the-art Khmer spellchecking accuracy of up to 94.4%, compared to existing solutions. The benchmark datasets for Khmer spellchecking and NER tasks in this study will be made publicly available. 

---
# TARG: Training-Free Adaptive Retrieval Gating for Efficient RAG 

**Authors**: Yufeng Wang, Lu wei, Haibin Ling  

**Link**: [PDF](https://arxiv.org/pdf/2511.09803)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves factuality but retrieving for every query often hurts quality while inflating tokens and latency. We propose Training-free Adaptive Retrieval Gating (TARG), a single-shot policy that decides when to retrieve using only a short, no-context draft from the base model. From the draft's prefix logits, TARG computes lightweight uncertainty scores: mean token entropy, a margin signal derived from the top-1/top-2 logit gap via a monotone link, or small-N variance across a handful of stochastic prefixes, and triggers retrieval only when the score exceeds a threshold. The gate is model agnostic, adds only tens to hundreds of draft tokens, and requires no additional training or auxiliary heads. On NQ-Open, TriviaQA, and PopQA, TARG consistently shifts the accuracy-efficiency frontier: compared with Always-RAG, TARG matches or improves EM/F1 while reducing retrieval by 70-90% and cutting end-to-end latency, and it remains close to Never-RAG in overhead. A central empirical finding is that under modern instruction-tuned LLMs the margin signal is a robust default (entropy compresses as backbones sharpen), with small-N variance offering a conservative, budget-first alternative. We provide ablations over gate type and prefix length and use a delta-latency view to make budget trade-offs explicit. 

---
# Predicate-Argument Structure Divergences in Chinese and English Parallel Sentences and their Impact on Language Transfer 

**Authors**: Rocco Tripodi, Xiaoyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.09796)  

**Abstract**: Cross-lingual Natural Language Processing (NLP) has gained significant traction in recent years, offering practical solutions in low-resource settings by transferring linguistic knowledge from resource-rich to low-resource languages. This field leverages techniques like annotation projection and model transfer for language adaptation, supported by multilingual pre-trained language models. However, linguistic divergences hinder language transfer, especially among typologically distant languages. In this paper, we present an analysis of predicate-argument structures in parallel Chinese and English sentences. We explore the alignment and misalignment of predicate annotations, inspecting similarities and differences and proposing a categorization of structural divergences. The analysis and the categorization are supported by a qualitative and quantitative analysis of the results of an annotation projection experiment, in which, in turn, one of the two languages has been used as source language to project annotations into the corresponding parallel sentences. The results of this analysis show clearly that language transfer is asymmetric. An aspect that requires attention when it comes to selecting the source language in transfer learning applications and that needs to be investigated before any scientific claim about cross-lingual NLP is proposed. 

---
# How Small Can You Go? Compact Language Models for On-Device Critical Error Detection in Machine Translation 

**Authors**: Muskaan Chopra, Lorenz Sparrenberg, Sarthak Khanna, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2511.09748)  

**Abstract**: Large Language Models (LLMs) excel at evaluating machine translation (MT), but their scale and cost hinder deployment on edge devices and in privacy-sensitive workflows. We ask: how small can you get while still detecting meaning-altering translation errors? Focusing on English->German Critical Error Detection (CED), we benchmark sub-2B models (LFM2-350M, Qwen-3-0.6B/1.7B, Llama-3.2-1B-Instruct, Gemma-3-1B) across WMT21, WMT22, and SynCED-EnDe-2025. Our framework standardizes prompts, applies lightweight logit-bias calibration and majority voting, and reports both semantic quality (MCC, F1-ERR/F1-NOT) and compute metrics (VRAM, latency, throughput). Results reveal a clear sweet spot around one billion parameters: Gemma-3-1B provides the best quality-efficiency trade-off, reaching MCC=0.77 with F1-ERR=0.98 on SynCED-EnDe-2025 after merged-weights fine-tuning, while maintaining 400 ms single-sample latency on a MacBook Pro M4 Pro (24 GB). At larger scale, Qwen-3-1.7B attains the highest absolute MCC (+0.11 over Gemma) but with higher compute cost. In contrast, ultra-small models (0.6B) remain usable with few-shot calibration yet under-detect entity and number errors. Overall, compact, instruction-tuned LLMs augmented with lightweight calibration and small-sample supervision can deliver trustworthy, on-device CED for MT, enabling private, low-cost error screening in real-world translation pipelines. All datasets, prompts, and scripts are publicly available at our GitHub repository. 

---
# Assessing the Applicability of Natural Language Processing to Traditional Social Science Methodology: A Case Study in Identifying Strategic Signaling Patterns in Presidential Directives 

**Authors**: C. LeMay, A. Lane, J. Seales, M. Winstead, S. Baty  

**Link**: [PDF](https://arxiv.org/pdf/2511.09738)  

**Abstract**: Our research investigates how Natural Language Processing (NLP) can be used to extract main topics from a larger corpus of written data, as applied to the case of identifying signaling themes in Presidential Directives (PDs) from the Reagan through Clinton administrations. Analysts and NLP both identified relevant documents, demonstrating the potential utility of NLPs in research involving large written corpuses. However, we also identified discrepancies between NLP and human-labeled results that indicate a need for more research to assess the validity of NLP in this use case. The research was conducted in 2023, and the rapidly evolving landscape of AIML means existing tools have improved and new tools have been developed; this research displays the inherent capabilities of a potentially dated AI tool in emerging social science applications. 

---
# Contextual morphologically-guided tokenization for Latin encoder models 

**Authors**: Marisa Hudspeth, Patrick J. Burns, Brendan O'Connor  

**Link**: [PDF](https://arxiv.org/pdf/2511.09709)  

**Abstract**: Tokenization is a critical component of language model pretraining, yet standard tokenization methods often prioritize information-theoretical goals like high compression and low fertility rather than linguistic goals like morphological alignment. In fact, they have been shown to be suboptimal for morphologically rich languages, where tokenization quality directly impacts downstream performance. In this work, we investigate morphologically-aware tokenization for Latin, a morphologically rich language that is medium-resource in terms of pretraining data, but high-resource in terms of curated lexical resources -- a distinction that is often overlooked but critical in discussions of low-resource language modeling. We find that morphologically-guided tokenization improves overall performance on four downstream tasks. Performance gains are most pronounced for out of domain texts, highlighting our models' improved generalization ability. Our findings demonstrate the utility of linguistic resources to improve language modeling for morphologically complex languages. For low-resource languages that lack large-scale pretraining data, the development and incorporation of linguistic resources can serve as a feasible alternative to improve LM performance. 

---
# Order Matters: Rethinking Prompt Construction in In-Context Learning 

**Authors**: Warren Li, Yiqian Wang, Zihan Wang, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2511.09700)  

**Abstract**: In-context learning (ICL) enables large language models to perform new tasks by conditioning on a sequence of examples. Most prior work reasonably and intuitively assumes that which examples are chosen has a far greater effect on performance than how those examples are ordered, leading to a focus on example selection. We revisit this assumption and conduct a systematic comparison between the effect of selection and ordering. Through controlled experiments on both classification and generation tasks, using multiple open-source model families (0.5B to 27B parameters) and GPT-5, we find that the variance in performance due to different example orderings is comparable to that from using entirely different example sets. Furthermore, we show that strong orderings can be identified using only a development set, achieving performance close to an oracle that selects the best ordering based on test labels. Our findings highlight the equal and intertwined importance of example selection and ordering in prompt design, calling for a reexamination of the assumptions held in ICL. 

---
# Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages 

**Authors**: Omnilingual ASR team, Gil Keren, Artyom Kozhevnikov, Yen Meng, Christophe Ropers, Matthew Setzler, Skyler Wang, Ife Adebara, Michael Auli, Can Balioglu, Kevin Chan, Chierh Cheng, Joe Chuang, Caley Droof, Mark Duppenthaler, Paul-Ambroise Duquenne, Alexander Erben, Cynthia Gao, Gabriel Mejia Gonzalez, Kehan Lyu, Sagar Miglani, Vineel Pratap, Kaushik Ram Sadagopan, Safiyyah Saleem, Arina Turkatenko, Albert Ventayol-Boada, Zheng-Xin Yong, Yu-An Chung, Jean Maillard, Rashel Moritz, Alexandre Mourachko, Mary Williamson, Shireen Yates  

**Link**: [PDF](https://arxiv.org/pdf/2511.09690)  

**Abstract**: Automatic speech recognition (ASR) has advanced in high-resource languages, but most of the world's 7,000+ languages remain unsupported, leaving thousands of long-tail languages behind. Expanding ASR coverage has been costly and limited by architectures that restrict language support, making extension inaccessible to most--all while entangled with ethical concerns when pursued without community collaboration. To transcend these limitations, we introduce Omnilingual ASR, the first large-scale ASR system designed for extensibility. Omnilingual ASR enables communities to introduce unserved languages with only a handful of data samples. It scales self-supervised pre-training to 7B parameters to learn robust speech representations and introduces an encoder-decoder architecture designed for zero-shot generalization, leveraging a LLM-inspired decoder. This capability is grounded in a massive and diverse training corpus; by combining breadth of coverage with linguistic variety, the model learns representations robust enough to adapt to unseen languages. Incorporating public resources with community-sourced recordings gathered through compensated local partnerships, Omnilingual ASR expands coverage to over 1,600 languages, the largest such effort to date--including over 500 never before served by ASR. Automatic evaluations show substantial gains over prior systems, especially in low-resource conditions, and strong generalization. We release Omnilingual ASR as a family of models, from 300M variants for low-power devices to 7B for maximum accuracy. We reflect on the ethical considerations shaping this design and conclude by discussing its societal impact. In particular, we highlight how open-sourcing models and tools can lower barriers for researchers and communities, inviting new forms of participation. Open-source artifacts are available at this https URL. 

---
# Towards Blind and Low-Vision Accessibility of Lightweight VLMs and Custom LLM-Evals 

**Authors**: Shruti Singh Baghel, Yash Pratap Singh Rathore, Sushovan Jena, Anurag Pradhan, Amit Shukla, Arnav Bhavsar, Pawan Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2511.10615)  

**Abstract**: Large Vision-Language Models (VLMs) excel at understanding and generating video descriptions but their high memory, computation, and deployment demands hinder practical use particularly for blind and low-vision (BLV) users who depend on detailed, context-aware descriptions. To study the effect of model size on accessibility-focused description quality, we evaluate SmolVLM2 variants with 500M and 2.2B parameters across two diverse datasets: AVCaps (outdoor), and Charades (indoor). In this work, we introduce two novel evaluation frameworks specifically designed for BLV accessibility assessment: the Multi-Context BLV Framework evaluating spatial orientation, social interaction, action events, and ambience contexts; and the Navigational Assistance Framework focusing on mobility-critical information. Additionally, we conduct a systematic evaluation of four different prompt design strategies and deploy both models on a smartphone, evaluating FP32 and INT8 precision variants to assess real-world performance constraints on resource-limited mobile devices. 

---
# Towards Emotionally Intelligent and Responsible Reinforcement Learning 

**Authors**: Garapati Keerthana, Manik Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.10573)  

**Abstract**: Personalized decision systems in healthcare and behavioral support often rely on static rule-based or engagement-maximizing heuristics that overlook users' emotional context and ethical constraints. Such approaches risk recommending insensitive or unsafe interventions, especially in domains involving serious mental illness, substance use disorders, or depression. To address this limitation, we propose a Responsible Reinforcement Learning (RRL) framework that integrates emotional and contextual understanding with ethical considerations into the sequential decision-making process. RRL formulates personalization as a Constrained Markov Decision Process (CMDP), where the agent optimizes engagement and adherence while ensuring emotional alignment and ethical safety. We introduce a multi-objective reward function that explicitly balances short-term behavioral engagement with long-term user well-being, and define an emotion-informed state representation that captures fluctuations in emotional readiness, affect, and risk. The proposed architecture can be instantiated with any RL algorithm (e.g., DQN, PPO) augmented with safety constraints or Lagrangian regularization. Conceptually, this framework operationalizes empathy and responsibility within machine learning policy optimization, bridging safe RL, affective computing and responsible AI. We discuss the implications of this approach for human-centric domains such as behavioral health, education, and digital therapeutics, and outline simulation-based validation paths for future empirical work. This paper aims to initiate a methodological conversation about ethically aligned reinforcement learning for emotionally aware and trustworthy personalization systems. 

---
# Impact of Layer Norm on Memorization and Generalization in Transformers 

**Authors**: Rishi Singhal, Jung-Eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.10566)  

**Abstract**: Layer Normalization (LayerNorm) is one of the fundamental components in transformers that stabilizes training and improves optimization. In recent times, Pre-LayerNorm transformers have become the preferred choice over Post-LayerNorm transformers due to their stable gradient flow. However, the impact of LayerNorm on learning and memorization across these architectures remains unclear. In this work, we investigate how LayerNorm influences memorization and learning for Pre- and Post-LayerNorm transformers. We identify that LayerNorm serves as a key factor for stable learning in Pre-LayerNorm transformers, while in Post-LayerNorm transformers, it impacts memorization. Our analysis reveals that eliminating LayerNorm parameters in Pre-LayerNorm models exacerbates memorization and destabilizes learning, while in Post-LayerNorm models, it effectively mitigates memorization by restoring genuine labels. We further precisely identify that early layers LayerNorm are the most critical over middle/later layers and their influence varies across Pre and Post LayerNorm models. We have validated it through 13 models across 6 Vision and Language datasets. These insights shed new light on the role of LayerNorm in shaping memorization and learning in transformers. 

---
# Rethinking the Reliability of Multi-agent System: A Perspective from Byzantine Fault Tolerance 

**Authors**: Lifan Zheng, Jiawei Chen, Qinghong Yin, Jingyuan Zhang, Xinyi Zeng, Yu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2511.10400)  

**Abstract**: Ensuring the reliability of agent architectures and effectively identifying problematic agents when failures occur are crucial challenges in multi-agent systems (MAS). Advances in large language models (LLMs) have established LLM-based agents as a major branch of MAS, enabling major breakthroughs in complex problem solving and world modeling. However, the reliability implications of this shift remain largely unexplored. i.e., whether substituting traditional agents with LLM-based agents can effectively enhance the reliability of MAS. In this work, we investigate and quantify the reliability of LLM-based agents from the perspective of Byzantine fault tolerance. We observe that LLM-based agents demonstrate stronger skepticism when processing erroneous message flows, a characteristic that enables them to outperform traditional agents across different topological structures. Motivated by the results of the pilot experiment, we design CP-WBFT, a confidence probe-based weighted Byzantine Fault Tolerant consensus mechanism to enhance the stability of MAS with different topologies. It capitalizes on the intrinsic reflective and discriminative capabilities of LLMs by employing a probe-based, weighted information flow transmission method to improve the reliability of LLM-based agents. Extensive experiments demonstrate that CP-WBFT achieves superior performance across diverse network topologies under extreme Byzantine conditions (85.7\% fault rate). Notably, our approach surpasses traditional methods by attaining remarkable accuracy on various topologies and maintaining strong reliability in both mathematical reasoning and safety assessment tasks. 

---
# AgentEvolver: Towards Efficient Self-Evolving Agent System 

**Authors**: Yunpeng Zhai, Shuchang Tao, Cheng Chen, Anni Zou, Ziqian Chen, Qingxu Fu, Shinji Mai, Li Yu, Jiaji Deng, Zouying Cao, Zhaoyang Liu, Bolin Ding, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.10395)  

**Abstract**: Autonomous agents powered by large language models (LLMs) have the potential to significantly enhance human productivity by reasoning, using tools, and executing complex tasks in diverse environments. However, current approaches to developing such agents remain costly and inefficient, as they typically require manually constructed task datasets and reinforcement learning (RL) pipelines with extensive random exploration. These limitations lead to prohibitively high data-construction costs, low exploration efficiency, and poor sample utilization. To address these challenges, we present AgentEvolver, a self-evolving agent system that leverages the semantic understanding and reasoning capabilities of LLMs to drive autonomous agent learning. AgentEvolver introduces three synergistic mechanisms: (i) self-questioning, which enables curiosity-driven task generation in novel environments, reducing dependence on handcrafted datasets; (ii) self-navigating, which improves exploration efficiency through experience reuse and hybrid policy guidance; and (iii) self-attributing, which enhances sample efficiency by assigning differentiated rewards to trajectory states and actions based on their contribution. By integrating these mechanisms into a unified framework, AgentEvolver enables scalable, cost-effective, and continual improvement of agent capabilities. Preliminary experiments indicate that AgentEvolver achieves more efficient exploration, better sample utilization, and faster adaptation compared to traditional RL-based baselines. 

---
# Simulating Misinformation Propagation in Social Networks using Large Language Models 

**Authors**: Raj Gaurav Maurya, Vaibhav Shukla, Raj Abhijit Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2511.10384)  

**Abstract**: Misinformation on social media thrives on surprise, emotion, and identity-driven reasoning, often amplified through human cognitive biases. To investigate these mechanisms, we model large language model (LLM) personas as synthetic agents that mimic user-level biases, ideological alignments, and trust heuristics. Within this setup, we introduce an auditor--node framework to simulate and analyze how misinformation evolves as it circulates through networks of such agents. News articles are propagated across networks of persona-conditioned LLM nodes, each rewriting received content. A question--answering-based auditor then measures factual fidelity at every step, offering interpretable, claim-level tracking of misinformation drift. We formalize a misinformation index and a misinformation propagation rate to quantify factual degradation across homogeneous and heterogeneous branches of up to 30 sequential rewrites. Experiments with 21 personas across 10 domains reveal that identity- and ideology-based personas act as misinformation accelerators, especially in politics, marketing, and technology. By contrast, expert-driven personas preserve factual stability. Controlled-random branch simulations further show that once early distortions emerge, heterogeneous persona interactions rapidly escalate misinformation to propaganda-level distortion. Our taxonomy of misinformation severity -- spanning factual errors, lies, and propaganda -- connects observed drift to established theories in misinformation studies. These findings demonstrate the dual role of LLMs as both proxies for human-like biases and as auditors capable of tracing information fidelity. The proposed framework provides an interpretable, empirically grounded approach for studying, simulating, and mitigating misinformation diffusion in digital ecosystems. 

---
# Music Flamingo: Scaling Music Understanding in Audio Language Models 

**Authors**: Sreyan Ghosh, Arushi Goel, Lasha Koroshinadze, Sang-gil Lee, Zhifeng Kong, Joao Felipe Santos, Ramani Duraiswami, Dinesh Manocha, Wei Ping, Mohammad Shoeybi, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2511.10289)  

**Abstract**: We introduce Music Flamingo, a novel large audio-language model designed to advance music (including song) understanding in foundational audio models. While audio-language research has progressed rapidly, music remains challenging due to its dynamic, layered, and information-dense nature. Progress has been further limited by the difficulty of scaling open audio understanding models, primarily because of the scarcity of high-quality music data and annotations. As a result, prior models are restricted to producing short, high-level captions, answering only surface-level questions, and showing limited generalization across diverse musical cultures. To address these challenges, we curate MF-Skills, a large-scale dataset labeled through a multi-stage pipeline that yields rich captions and question-answer pairs covering harmony, structure, timbre, lyrics, and cultural context. We fine-tune an enhanced Audio Flamingo 3 backbone on MF-Skills and further strengthen multiple skills relevant to music understanding. To improve the model's reasoning abilities, we introduce a post-training recipe: we first cold-start with MF-Think, a novel chain-of-thought dataset grounded in music theory, followed by GRPO-based reinforcement learning with custom rewards. Music Flamingo achieves state-of-the-art results across 10+ benchmarks for music understanding and reasoning, establishing itself as a generalist and musically intelligent audio-language model. Beyond strong empirical results, Music Flamingo sets a new standard for advanced music understanding by demonstrating how models can move from surface-level recognition toward layered, human-like perception of songs. We believe this work provides both a benchmark and a foundation for the community to build the next generation of models that engage with music as meaningfully as humans do. 

---
# OutSafe-Bench: A Benchmark for Multimodal Offensive Content Detection in Large Language Models 

**Authors**: Yuping Yan, Yuhan Xie, Yuanshuai Li, Yingchao Yu, Lingjuan Lyu, Yaochu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2511.10287)  

**Abstract**: Since Multimodal Large Language Models (MLLMs) are increasingly being integrated into everyday tools and intelligent agents, growing concerns have arisen regarding their possible output of unsafe contents, ranging from toxic language and biased imagery to privacy violations and harmful misinformation. Current safety benchmarks remain highly limited in both modality coverage and performance evaluations, often neglecting the extensive landscape of content safety. In this work, we introduce OutSafe-Bench, the first most comprehensive content safety evaluation test suite designed for the multimodal era. OutSafe-Bench includes a large-scale dataset that spans four modalities, featuring over 18,000 bilingual (Chinese and English) text prompts, 4,500 images, 450 audio clips and 450 videos, all systematically annotated across nine critical content risk categories. In addition to the dataset, we introduce a Multidimensional Cross Risk Score (MCRS), a novel metric designed to model and assess overlapping and correlated content risks across different categories. To ensure fair and robust evaluation, we propose FairScore, an explainable automated multi-reviewer weighted aggregation framework. FairScore selects top-performing models as adaptive juries, thereby mitigating biases from single-model judgments and enhancing overall evaluation reliability. Our evaluation of nine state-of-the-art MLLMs reveals persistent and substantial safety vulnerabilities, underscoring the pressing need for robust safeguards in MLLMs. 

---
# FactGuard: Event-Centric and Commonsense-Guided Fake News Detection 

**Authors**: Jing He, Han Zhang, Yuanhui Xiao, Wei Guo, Shaowen Yao, Renyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.10281)  

**Abstract**: Fake news detection methods based on writing style have achieved remarkable progress. However, as adversaries increasingly imitate the style of authentic news, the effectiveness of such approaches is gradually diminishing. Recent research has explored incorporating large language models (LLMs) to enhance fake news detection. Yet, despite their transformative potential, LLMs remain an untapped goldmine for fake news detection, with their real-world adoption hampered by shallow functionality exploration, ambiguous usability, and prohibitive inference costs. In this paper, we propose a novel fake news detection framework, dubbed FactGuard, that leverages LLMs to extract event-centric content, thereby reducing the impact of writing style on detection performance. Furthermore, our approach introduces a dynamic usability mechanism that identifies contradictions and ambiguous cases in factual reasoning, adaptively incorporating LLM advice to improve decision reliability. To ensure efficiency and practical deployment, we employ knowledge distillation to derive FactGuard-D, enabling the framework to operate effectively in cold-start and resource-constrained scenarios. Comprehensive experiments on two benchmark datasets demonstrate that our approach consistently outperforms existing methods in both robustness and accuracy, effectively addressing the challenges of style sensitivity and LLM usability in fake news detection. 

---
# ProgRAG: Hallucination-Resistant Progressive Retrieval and Reasoning over Knowledge Graphs 

**Authors**: Minbae Park, Hyemin Yang, Jeonghyun Kim, Kunsoo Park, Hyunjoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.10240)  

**Abstract**: Large Language Models (LLMs) demonstrate strong reasoning capabilities but struggle with hallucinations and limited transparency. Recently, KG-enhanced LLMs that integrate knowledge graphs (KGs) have been shown to improve reasoning performance, particularly for complex, knowledge-intensive tasks. However, these methods still face significant challenges, including inaccurate retrieval and reasoning failures, often exacerbated by long input contexts that obscure relevant information or by context constructions that struggle to capture the richer logical directions required by different question types. Furthermore, many of these approaches rely on LLMs to directly retrieve evidence from KGs, and to self-assess the sufficiency of this evidence, which often results in premature or incorrect reasoning. To address the retrieval and reasoning failures, we propose ProgRAG, a multi-hop knowledge graph question answering (KGQA) framework that decomposes complex questions into sub-questions, and progressively extends partial reasoning paths by answering each sub-question. At each step, external retrievers gather candidate evidence, which is then refined through uncertainty-aware pruning by the LLM. Finally, the context for LLM reasoning is optimized by organizing and rearranging the partial reasoning paths obtained from the sub-question answers. Experiments on three well-known datasets demonstrate that ProgRAG outperforms existing baselines in multi-hop KGQA, offering improved reliability and reasoning quality. 

---
# Enhancing the Medical Context-Awareness Ability of LLMs via Multifaceted Self-Refinement Learning 

**Authors**: Yuxuan Zhou, Yubin Wang, Bin Wang, Chen Ning, Xien Liu, Ji Wu, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2511.10067)  

**Abstract**: Large language models (LLMs) have shown great promise in the medical domain, achieving strong performance on several benchmarks. However, they continue to underperform in real-world medical scenarios, which often demand stronger context-awareness, i.e., the ability to recognize missing or critical details (e.g., user identity, medical history, risk factors) and provide safe, helpful, and contextually appropriate responses. To address this issue, we propose Multifaceted Self-Refinement (MuSeR), a data-driven approach that enhances LLMs' context-awareness along three key facets (decision-making, communication, and safety) through self-evaluation and refinement. Specifically, we first design a attribute-conditioned query generator that simulates diverse real-world user contexts by varying attributes such as role, geographic region, intent, and degree of information ambiguity. An LLM then responds to these queries, self-evaluates its answers along three key facets, and refines its responses to better align with the requirements of each facet. Finally, the queries and refined responses are used for supervised fine-tuning to reinforce the model's context-awareness ability. Evaluation results on the latest HealthBench dataset demonstrate that our method significantly improves LLM performance across multiple aspects, with particularly notable gains in the context-awareness axis. Furthermore, by incorporating knowledge distillation with the proposed method, the performance of a smaller backbone LLM (e.g., Qwen3-32B) surpasses its teacher model, achieving a new SOTA across all open-source LLMs on HealthBench (63.8%) and its hard subset (43.1%). Code and dataset will be released at this https URL. 

---
# Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers 

**Authors**: Xuan Rao, Simian Xu, Zheng Li, Bo Zhao, Derong Liu, Mingming Ha, Cesare Alippi  

**Link**: [PDF](https://arxiv.org/pdf/2511.09926)  

**Abstract**: Recent advances have shown that sequential fine-tuning (SeqFT) of pre-trained vision transformers (ViTs), followed by classifier refinement using approximate distributions of class features, can be an effective strategy for class-incremental learning (CIL). However, this approach is susceptible to distribution drift, caused by the sequential optimization of shared backbone parameters. This results in a mismatch between the distributions of the previously learned classes and that of the updater model, ultimately degrading the effectiveness of classifier performance over time. To address this issue, we introduce a latent space transition operator and propose Sequential Learning with Drift Compensation (SLDC). SLDC aims to align feature distributions across tasks to mitigate the impact of drift. First, we present a linear variant of SLDC, which learns a linear operator by solving a regularized least-squares problem that maps features before and after fine-tuning. Next, we extend this with a weakly nonlinear SLDC variant, which assumes that the ideal transition operator lies between purely linear and fully nonlinear transformations. This is implemented using learnable, weakly nonlinear mappings that balance flexibility and generalization. To further reduce representation drift, we apply knowledge distillation (KD) in both algorithmic variants. Extensive experiments on standard CIL benchmarks demonstrate that SLDC significantly improves the performance of SeqFT. Notably, by combining KD to address representation drift with SLDC to compensate distribution drift, SeqFT achieves performance comparable to joint training across all evaluated datasets. Code: this https URL. 

---
# Regional Attention-Enhanced Swin Transformer for Clinically Relevant Medical Image Captioning 

**Authors**: Zubia Naz, Farhan Asghar, Muhammad Ishfaq Hussain, Yahya Hadadi, Muhammad Aasim Rafique, Wookjin Choi, Moongu Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2511.09893)  

**Abstract**: Automated medical image captioning translates complex radiological images into diagnostic narratives that can support reporting workflows. We present a Swin-BART encoder-decoder system with a lightweight regional attention module that amplifies diagnostically salient regions before cross-attention. Trained and evaluated on ROCO, our model achieves state-of-the-art semantic fidelity while remaining compact and interpretable. We report results as mean$\pm$std over three seeds and include $95\%$ confidence intervals. Compared with baselines, our approach improves ROUGE (proposed 0.603, ResNet-CNN 0.356, BLIP2-OPT 0.255) and BERTScore (proposed 0.807, BLIP2-OPT 0.645, ResNet-CNN 0.623), with competitive BLEU, CIDEr, and METEOR. We further provide ablations (regional attention on/off and token-count sweep), per-modality analysis (CT/MRI/X-ray), paired significance tests, and qualitative heatmaps that visualize the regions driving each description. Decoding uses beam search (beam size $=4$), length penalty $=1.1$, $no\_repeat\_ngram\_size$ $=3$, and max length $=128$. The proposed design yields accurate, clinically phrased captions and transparent regional attributions, supporting safe research use with a human in the loop. 

---
# Probability-Biased Attention over Directed Bipartite Graphs for Long-Tail ICD Coding 

**Authors**: Tianlei Chen, Yuxiao Chen, Yang Li, Feifei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.09559)  

**Abstract**: Automated International Classification of Diseases (ICD) coding aims to assign multiple disease codes to clinical documents, constituting a crucial multi-label text classification task in healthcare informatics. However, the task is challenging due to its large label space (10,000 to 20,000 codes) and long-tail distribution, where a few codes dominate while many rare codes lack sufficient training data. To address this, we propose a learning method that models fine-grained co-occurrence relationships among codes. Specifically, we construct a Directed Bipartite Graph Encoder with disjoint sets of common and rare code nodes. To facilitate a one-way information flow, edges are directed exclusively from common to rare codes. The nature of these connections is defined by a probability-based bias, which is derived from the conditional probability of a common code co-occurring given the presence of a rare code. This bias is then injected into the encoder's attention module, a process we term Co-occurrence Encoding. This structure empowers the graph encoder to enrich rare code representations by aggregating latent comorbidity information reflected in the statistical co-occurrence of their common counterparts. To ensure high-quality input to the graph, we utilize a large language model (LLM) to generate comprehensive descriptions for codes, enriching initial embeddings with clinical context and comorbidity information, serving as external knowledge for the statistical co-occurrence relationships in the code system. Experiments on three automated ICD coding benchmark datasets demonstrate that our method achieves state-of-the-art performance with particularly notable improvements in Macro-F1, which is the key metric for long-tail classification. 

---
