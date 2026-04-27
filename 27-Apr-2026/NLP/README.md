# How Do AI Agents Spend Your Money? Analyzing and Predicting Token Consumption in Agentic Coding Tasks 

**Authors**: Longju Bai, Zhemin Huang, Xingyao Wang, Jiao Sun, Rada Mihalcea, Erik Brynjolfsson, Alex Pentland, Jiaxin Pei  

**Link**: [PDF](https://arxiv.org/pdf/2604.22750)  

**Abstract**: The wide adoption of AI agents in complex human workflows is driving rapid growth in LLM token consumption. When agents are deployed on tasks that require a significant amount of tokens, three questions naturally arise: (1) Where do AI agents spend the tokens? (2) Which models are more token-efficient? and (3) Can agents predict their token usage before task execution? In this paper, we present the first systematic study of token consumption patterns in agentic coding tasks. We analyze trajectories from eight frontier LLMs on SWE-bench Verified and evaluate models' ability to predict their own token costs before task execution. We find that: (1) agentic tasks are uniquely expensive, consuming 1000x more tokens than code reasoning and code chat, with input tokens rather than output tokens driving the overall cost; (2) token usage is highly variable and inherently stochastic: runs on the same task can differ by up to 30x in total tokens, and higher token usage does not translate into higher accuracy; instead, accuracy often peaks at intermediate cost and saturates at higher costs; (3) models vary substantially in token efficiency: on the same tasks, Kimi-K2 and Claude-Sonnet-4.5, on average, consume over 1.5 million more tokens than GPT-5; (4) task difficulty rated by human experts only weakly aligns with actual token costs, revealing a fundamental gap between human-perceived complexity and the computational effort agents actually expend; and (5) frontier models fail to accurately predict their own token usage (with weak-to-moderate correlations, up to 0.39) and systematically underestimate real token costs. Our study offers new insights into the economics of AI agents and can inspire future research in this direction. 

---
# Representational Harms in LLM-Generated Narratives Against Global Majority Nationalities 

**Authors**: Ilana Nguyen, Harini Suresh, Thema Monroe-White, Evan Shieh  

**Link**: [PDF](https://arxiv.org/pdf/2604.22749)  

**Abstract**: Large language models (LLMs) are increasingly used for text generation tasks from everyday use to high-stakes enterprise and government applications, including simulated interviews with asylum seekers. While many works highlight the new potential applications of LLMs, there are risks of LLMs encoding and perpetuating harmful biases about non-dominant communities across the globe. To better evaluate and mitigate such harms, more research examining how LLMs portray diverse individuals is needed. In this work, we study how national origin identities are portrayed by widely-adopted LLMs in response to open-ended narrative generation prompts. Our findings demonstrate the presence of persistent representational harms by national origin, including harmful stereotypes, erasure, and one-dimensional portrayals of Global Majority identities. Minoritized national identities are simultaneously underrepresented in power-neutral stories and overrepresented in subordinated character portrayals, which are over fifty times more likely to appear than dominant portrayals. The degree of harm is amplified when US nationality cues (e.g., ``American'') are present in input prompts. Notably, we find that the harms we identify cannot be explained away via sycophancy, as US-centric biases persist even when replacing US nationality cues with non-US national identities in the prompts. Based on our findings, we call for further exploration of cultural harms in LLMs through methodologies that center Global Majority perspectives and challenge the uncritical adoption of US-based LLMs for the classification, surveillance, and misrepresentation of the majority of our planet. 

---
# Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought 

**Authors**: Keshav Ramji, Tahira Naseem, Ramón Fernandez Astudillo  

**Link**: [PDF](https://arxiv.org/pdf/2604.22709)  

**Abstract**: While long, explicit chains-of-thought (CoT) have proven effective on complex reasoning tasks, they are costly to generate during inference. Non-verbal reasoning methods have emerged with shorter generation lengths by leveraging continuous representations, yet their performance lags behind verbalized CoT. We propose $\textbf{Abstract Chain-of-Thought}$, a discrete latent reasoning post-training mechanism in which the language model produces a short sequence of tokens from a reserved vocabulary in lieu of a natural language CoT, before generating a response. To make previously unseen ''abstract'' tokens useful, we introduce a policy iteration-style warm-up loop that alternates between (i.) bottlenecking from a verbal CoT via masking and performing supervised fine-tuning, and (ii.) self-distillation by training the model to generate abstract tokens from the prompt alone via constrained decoding with the codebook. After warm-up, we optimize the generation of abstract sequences with warm-started reinforcement learning under constrained decoding. Abstract-CoT achieves up to $11.6\times$ fewer reasoning tokens while demonstrating comparable performance across mathematical reasoning, instruction-following, and multi-hop reasoning, and generalizes across language model families. We also find an emergent power law distribution over the abstract vocabulary, akin to those seen in natural language, that evolves across the training phases. Our findings highlight the potential for post-training latent reasoning mechanisms that enable efficient inference through a learned abstract reasoning language. 

---
# CRAFT: Clustered Regression for Adaptive Filtering of Training data 

**Authors**: Parthasarathi Panda, Asheswari Swain, Subhrakanta Panda  

**Link**: [PDF](https://arxiv.org/pdf/2604.22693)  

**Abstract**: Selecting a small, high-quality subset from a large corpus for fine-tuning is increasingly important as corpora grow to tens of millions of datapoints, making full fine-tuning expensive and often unnecessary. We propose CRAFT (Clustered Regression for Adaptive Filtering of Training data), a vectorization-agnostic selection method for training sequence-to-sequence models. CRAFT decomposes the joint source-target distribution and performs a two-stage selection: (i) match the validation source distribution through proportional budget allocation across k-means clusters, and (ii) within each source cluster, select training pairs whose target embeddings minimize a conditional expected distance derived from the validation target distribution. We prove that proportional cluster allocation bounds the continuous KL divergence between selected and validation distributions, with the residual controlled by cluster diameters. We evaluate CRAFT on English-Hindi translation by selecting training data from 33 million NLLB sentence pairs and fine-tuning mBART via LoRA. CRAFT achieves 43.34 BLEU, outperforming TSDS (41.21) by 2.13 points on the same candidate pool and encoder while completing selection over 40 times faster. With TF-IDF vectorization, the entire pipeline completes in under one minute on CPU. TAROT achieves 45.61 BLEU, but CRAFT completes selection in 26.86 seconds versus TAROT's 75.6 seconds, a 2.8 time speedup. 

---
# BERAG: Bayesian Ensemble Retrieval-Augmented Generation for Knowledge-based Visual Question Answering 

**Authors**: Jinghong Chen, Jingbiao Mei, Guangyu Yang, Bill Byrne  

**Link**: [PDF](https://arxiv.org/pdf/2604.22678)  

**Abstract**: A common approach to question answering with retrieval-augmented generation (RAG) is to concatenate documents into a single context and pass it to a language model to generate an answer. While simple, this strategy can obscure the contribution of individual documents, making attribution difficult and contributing to the ``lost-in-the-middle'' effect, where relevant information in long contexts is overlooked. Concatenation also scales poorly: computational cost grows quadratically with context length, a problem that becomes especially severe when the context includes visual data, as in visual question answering. Attempts to mitigate these issues by limiting context length can further restrict performance by preventing models from benefiting from the improved recall offered by deeper retrieval. We propose Bayesian Ensemble Retrieval-Augmented Generation (BERAG), along with Bayesian Ensemble Fine-Tuning (BEFT), as a RAG framework in which language models are conditioned on individual retrieved documents rather than a single combined context. BERAG treats document posterior probabilities as ensemble weights and updates them token by token using Bayes' rule during generation. This approach enables probabilistic re-ranking, parallel memory usage, and clear attribution of document contribution, making it well-suited for large document collections. We evaluate BERAG and BEFT primarily on knowledge-based visual question answering tasks, where models must reason over long, imperfect retrieval lists. The results show substantial improvements over standard RAG, including strong gains on Document Visual Question Answering and multimodal needle-in-a-haystack benchmarks. We also demonstrate that BERAG mitigates the ``lost-in-the-middle'' effect. The document posterior can be used to detect insufficient grounding and trigger deflection, while document pruning enables faster decoding than standard RAG. 

---
# Identifying and typifying demographic unfairness in phoneme-level embeddings of self-supervised speech recognition models 

**Authors**: Felix Herron, Solange Rossato, Alexandre Allauzen, François Portet  

**Link**: [PDF](https://arxiv.org/pdf/2604.22631)  

**Abstract**: Modern automatic speech recognition (ASR) systems have been observed to function better for certain speaker groups (SGs) than others, despite recent gains in overall performance. One potential impediment to progress towards fairer ASR is a more nuanced understanding of the types of modeling errors that speech encoder models make, and in particular the difference between the structure of embeddings for high-performance and low-performance SGs. This paper proposes a framework typifying two types of error that can occur in modeling phonemes in ASR systems: random error/high variance in phoneme embedding, vs systematic error/embedding bias. We find that training phoneme classification probes only on a single, typically disadvantaged SG, sometimes improves performance for that SG, which is evidence for the existence of SG-level bias in phoneme embeddings. On the other hand, we find that speakers and SGs with higher levels of phoneme variance are the same as those with worse phoneme prediction accuracy. We conclude that both types of error are present in phoneme embeddings and both are candidate causes for SG-level unfairness in ASR, though random error is likely a greater hindrance to fairness than systematic error. Furthermore, we find that finetuning encoder models using a fairness-enhancing algorithm (domain enhancing and adversarial training) changes neither the benefits of in-domain phoneme classification probe training, nor measured levels of random embedding error. 

---
# From graphemic dependence to lexical structure: a Markovian perspective on Dante's Commedia 

**Authors**: Angelo Maria Sabatini  

**Link**: [PDF](https://arxiv.org/pdf/2604.22626)  

**Abstract**: This study investigates the structural organisation of Dante's Divina Commedia through a symbolic representation based on vowel-consonant (V/C) encoding. Modelling the resulting sequence as a four-state Markov chain yields a parsimonious index of graphemic memory, capturing the balance between persistence and alternation patterns.
Across the poem, this index exhibits a slight but consistent increase from the Inferno to the Paradiso, indicating a directional shift in local dependency structure. Trigram-level analysis shows that this trend is driven by a restricted set of recurrent configurations, interpreted as graphemic probes linking the Markov representation to identifiable lexical environments in the text.
These probes display distinct behaviours: configurations involving two transitions more frequently emerge across word boundaries, reflecting interactions between adjacent tokens, whereas configurations with fewer transitions are largely confined to intra-lexical structures. Part of the signal is further shaped by orthographic phenomena, particularly apostrophised forms, highlighting the role of writing conventions alongside phonological and lexical organisation.
A complementary classification analysis identifies cantica-specific terms, providing lexical anchors through which graphemic probes can be related to the structure of the poem. This organisation is reflected not only in the separation of the three cantiche, but also in a continuous trajectory across the text.
Overall, the results show that simple probabilistic models applied to symbolic text representations can uncover structured interactions between local dependencies, lexical distribution, orthographic encoding, and large-scale organisation, providing an interpretable framework for linking local symbolic dynamics to higher-level textual organisation. 

---
# Dharma, Data and Deception: An LLM-Powered Rhetorical Analysis of Cow-Urine Health Claims on YouTube 

**Authors**: Sheza Munir, Ratna Kandala, Anamta Khan, Deepti, Joyojeet Pal  

**Link**: [PDF](https://arxiv.org/pdf/2604.22606)  

**Abstract**: Health misinformation remains one of the most pressing challenges on social media, particularly when cultural traditions intersect with scientific-sounding claims. These dynamics are not only global but also deeply local, manifesting in culturally specific controversies that require careful analysis. Motivated by this, we examine 100 YouTube transcripts that promote or debunk cow urine (gomutra) as a health remedy, focusing on rhetorical strategies such as appeals to authority, efficacy appeals, and conspiracy framing. We employ large language models (LLMs) including GPT-4, GPT-4o, GPT-4.1, GPT-5, Gemini 2.5 Pro, and Mistral Medium 3 to annotate transcripts using a 14-category taxonomy of persuasive tactics. Our analysis reveals that promoters predominantly rely on efficacy appeals and social proof, while debunkers emphasize authority and rebuttal. Human evaluation of a subset of annotations yielded 90.1\% inter-annotator agreement, confirming the reliability of our taxonomy and validation process. This work advances computational methods for misinformation analysis and demonstrates how LLMs can support large-scale studies of cultural discourse online. 

---
# Learning Evidence Highlighting for Frozen LLMs 

**Authors**: Shaoang Li, Yanhang Shi, Yufei Li, Mingfu Liang, Xiaohan Wei, Yunchen Pu, Fei Tian, Chonglin Sun, Frank Shyu, Luke Simon, Sandeep Pandey, Xi Liu, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.22565)  

**Abstract**: Large Language Models (LLMs) can reason well, yet often miss decisive evidence when it is buried in long, noisy contexts. We introduce HiLight, an Evidence Emphasis framework that decouples evidence selection from reasoning for frozen LLM solvers. HiLight avoids compressing or rewriting the input, which can discard or distort evidence, by training a lightweight Emphasis Actor to insert minimal highlight tags around pivotal spans in the unaltered context. A frozen Solver then performs downstream reasoning on the emphasized input. We cast highlighting as a weakly supervised decision-making problem and optimize the Actor with reinforcement learning using only the Solver's task reward, requiring no evidence labels and no access to or modification of the Solver. Across sequential recommendation and long-context question answering, HiLight consistently improves performance over strong prompt-based and automated prompt-optimization baselines. The learned emphasis policy transfers zero-shot to both smaller and larger unseen Solver families, including an API-based Solver, suggesting that the Actor captures genuine, reusable evidence structure rather than overfitting to a single backbone. 

---
# Using Embedding Models to Improve Probabilistic Race Prediction 

**Authors**: Noan Dasanaike, Kosuke Imai  

**Link**: [PDF](https://arxiv.org/pdf/2604.22555)  

**Abstract**: Estimating racial disparity requires individual-level race data, which are often unavailable due to the sensitivity of collecting such information. To address this problem, many researchers utilize Bayesian Improved Surname Geocoding (BISG), which have critically relied on Census surname data. Unfortunately, these data capture race-surname relationships only for common surnames, omitting approximately 10% of the US population. We show that predictive performance degrades substantially for individuals with such omitted, uncommon surnames because standard BISG implementation relies on a uninformative generic prior in these cases. To address this limitation, we propose embedding-powered BISG (eBISG), which uses pre-trained text embeddings to represent names as dense vectors and trains neural networks on 2020 Census surname and first-name data to estimate race probabilities for names not covered in the Census. We compare five approaches: standard BISG using only surnames, BIFSG incorporating first name probabilities, surname embedding for unlisted names, surname and first name embedding combining both, and a full-name embedding trained on voter file data from Southern states that captures interactions between name components. We show that each successive eBISG approach improves race prediction, with the full-name embedding yielding the largest gains, particularly for Hispanic and Asian voters whose surnames are absent from the Census list. 

---
# Controllable Spoken Dialogue Generation: An LLM-Driven Grading System for K-12 Non-Native English Learners 

**Authors**: Haidong Yuan, Haokun Zhao, Wanshi Xu, Songjun Cao, Qingyu Zhou, Long Ma, Hongjie Fan  

**Link**: [PDF](https://arxiv.org/pdf/2604.22542)  

**Abstract**: Large language models (LLMs) often fail to meet the pedagogical needs of K-12 English learners in non-native contexts due to a proficiency mismatch. To address this widespread challenge, we introduce a proficiency-aligned framework that adapts LLM outputs to learner abilities, using China's national curriculum (CSE) as a representative case. Our framework enables precise control over lexical complexity through a four-tier grading system, supported by a comprehensive suite of new resources: graded vocabulary lists and a multi-turn dialogue corpus.
Our core technical contribution is the \textbf{DDPO} algorithm,Diversity Driven Policy Optimization, a multi-turn GRPO-based approach designed to preserve dialogue diversity while holistically optimizing dialogue quality. This method significantly outperforms conventional approaches, achieving low out-of-vocabulary rates and high diversity while enhancing conversational naturalness and pedagogical value. While grounded in the CSE, our framework is designed for flexibility and can be readily adapted to other educational standards. Our models, data, and code will all be open-sourced, providing a scalable platform for personalized English speaking practice that effectively addresses the unique challenges faced by K-12 learners in non-immersive environments. 

---
# RouteLMT: Learned Sample Routing for Hybrid LLM Translation Deployment 

**Authors**: Yingfeng Luo, Hongyu Liu, Dingyang Lin, Kaiyan Chang, Chenglong Wang, Bei Li, Quan Du, Tong Xiao, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22520)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable performance in Machine Translation (MT), but deploying them at scale remains prohibitively expensive. A widely adopted remedy is the hybrid system paradigm, which balances cost and quality by serving most requests with a small model and selectively routing a fraction to a large model. However, existing routing strategies often rely on heuristics, external predictors, or absolute quality estimation, which fail to capture whether the large model actually provides a worthwhile improvement over the small one. In this paper, we formulate routing as a budget allocation problem and identify marginal gain, i.e., the large model's improvement over the small model, as the optimal signal for budgeted decisions. Building on this, we propose \textbf{RouteLMT} (routing for LLM-based MT), an efficient in-model router that predicts this expected gain by probing the small translators prompt-token representation, without requiring external models or hypothesis decoding. Extensive experiments demonstrate that our RouteLMT outperforms heuristics, quality/difficulty estimation baselines, achieving a superior quality-budget Pareto frontier. Furthermore, we analyze regression risks and show that a simple guarded variant can mitigate severe quality losses. 

---
# Aggregate vs. Personalized Judges in Business Idea Evaluation: Evidence from Expert Disagreement 

**Authors**: Wataru Hirota, Tomoki Taniguchi, Tomoko Ohkuma, Kosuke Takahashi, Takahiro Omi, Kosuke Arima, Takuto Asakura, Chung-Chi Chen, Tatsuya Ishigaki  

**Link**: [PDF](https://arxiv.org/pdf/2604.22517)  

**Abstract**: Evaluating LLM-generated business ideas is often harder to scale than generating them. Unlike standard NLP benchmarks, business idea evaluation relies on multi-dimensional criteria such as feasibility, novelty, differentiation, user need, and market size, and expert judgments often disagree. This paper studies a methodological question raised by such disagreement: should an automatic judge approximate an aggregate consensus, or model evaluators individually? We introduce PBIG-DATA, a dataset of approximately 3,000 individual scores across 300 patent-grounded product ideas, provided by domain experts on six business-oriented dimensions: specificity, technical validity, innovativeness, competitive advantage, need validity, and market size. Analyses show substantial expert disagreement on fine-grained ordinal scores, while agreement is higher under coarse selection, suggesting structured heterogeneity rather than random noise. We then compare three judge configurations: a rubric-only zero-shot judge, an aggregate judge conditioned on mixed evaluator histories, and a personalized judge conditioned on the target evaluator's scoring history. Across dimensions and model sizes, personalized judges align more closely with the corresponding evaluator than aggregate judges, and evaluator agreement correlates with similarity of judge-generated reasoning only under personalized conditioning. These results indicate that pooled labels can be a fragile target in pluralistic evaluation settings and motivate evaluator-conditioned judge designs for business idea assessment. 

---
# Measuring and Mitigating Persona Distortions from AI Writing Assistance 

**Authors**: Paul Röttger, Kobi Hackenburg, Hannah Rose Kirk, Christopher Summerfield  

**Link**: [PDF](https://arxiv.org/pdf/2604.22503)  

**Abstract**: Hundreds of millions of people use artificial intelligence (AI) for writing assistance. Here, we evaluated how AI writing assistance distorts writer personas - their perceived beliefs, personality, and identity. In three large-scale experiments, writers (N=2,939) wrote political opinion paragraphs with and without AI assistance. Separate groups of readers (N=11,091) blindly evaluated these paragraphs across 29 socially salient dimensions of reader perception, spanning political opinion, writing quality, writer personality, emotions, and demographics. AI writing assistance produced persona distortions across all dimensions: with AI, writers seemed more opinionated, competent, and positive, and their perceived demographic profile shifted towards more privileged groups. Writers objected to many of the observed distortions, yet continued to prefer AI-assisted text even when made aware of them. We successfully mitigated objectionable persona distortions at the model level by training reward models on our experimental data (10,008 paragraphs, 2,903,596 ratings) to steer AI outputs towards faithful representation of writer stance. However, this came at a cost to user acceptance, suggesting an entanglement between desirable and undesirable properties of AI writing assistance that may be difficult to resolve. Together, our findings demonstrate that persona distortions from AI writing assistance are pervasive and persistent even under realistic conditions of human oversight, which carries implications for public discourse, trust, and democratic deliberation that scale with AI adoption. 

---
# Selective Contrastive Learning For Gloss Free Sign Language Translation 

**Authors**: Changhao Lai, Rui Zhao, Xuewen Zhong, Jinsong Su, Yidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.22374)  

**Abstract**: Sign language translation (SLT) converts continuous sign videos into spoken-language text, yet it remains challenging due to the intrinsic modality mismatch between visual signs and written text, particularly in gloss-free settings. Recent SLT systems increasingly adopt CLIP-like Vision-Language pretraining (VLP) for cross-modal alignment, but the random in-batch contrast provides few, batch-dependent negatives and may mislabel semantically similar (or even identical) pairs as negatives, introducing noisy and potentially inconsistent alignment supervision. In this work, we first conduct a preliminary trajectory-based analysis that tracks negative video-text similarity over training. The results show that only a small subset of negatives exhibits the desired behavior of being consistently pushed away, while the remaining negatives display heterogeneous and often non-decreasing similarity dynamics, suggesting that random in-batch negatives are frequently uninformative for effective alignment. Inspired by this, we propose Selective Contrastive Learning for SLT (SCL-SLT) with a Pair Selection (PS) strategy. PS scores candidate negatives using similarity dynamics from reference checkpoints and constructs mini-batches via a curriculum that progressively emphasizes more challenging negatives, thereby strengthening contrastive supervision while reducing the influence of noisy or semantically invalid negatives. 

---
# CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language 

**Authors**: Rui Zhao, Xuewen Zhong, Xiaoyun Zheng, Jinsong Su, Yidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.22367)  

**Abstract**: Sign language research has achieved significant progress due to the advances in large language models (LLMs). However, the intrinsic ability of LLMs to understand sign language, especially in multimodal contexts, remains underexplored. To address this limitation, we introduce CNSL-bench, the first comprehensive Chinese em{National Sign Language benchmark designed for evaluating multimodal large language models (MLLMs) in sign language understanding. The proposed CNSL-bench is characterized by: 1) Authoritative grounding, as it is anchored to the officially standardized \textit{National Common Sign Language Dictionary, mitigating ambiguity from regional or non-canonical variants and ensuring consistent semantic definitions; 2) Multimodal coverage, providing aligned textual descriptions, illustrative images, and sign language videos; and 3) Articulatory diversity, supporting fine-grained analysis across key manual articulatory forms, including air-writing, finger-spelling, and the Chinese manual-alphabet. Using CNSL-bench, we extensively evaluate 21 open-source and proprietary up-to-date MLLMs. Our results reveal that, despite recent advances in multimodal modeling, current MLLMs remain substantially inferior to human performance, exhibiting systematic disparities across input modalities and manual articulatory forms. Additional diagnostic analyses suggest that several performance limitations persist beyond improvements in reasoning and that instruction-following robustness varies substantially across models. 

---
# Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization 

**Authors**: Weixu Zhang, Ye Yuan, Changjiang Han, Yuxing Tian, Zipeng Sun, Linfeng Du, Jikun Kang, Hong Kang, Xue Liu, Haolun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22345)  

**Abstract**: Large Language Models (LLMs) exhibit strong implicit personalization ability, yet most existing approaches treat this behavior as a black box, relying on prompt engineering or fine tuning on user data. In this work, we adopt a mechanistic interpretability perspective and hypothesize the existence of a sparse set of Preference Heads, attention heads that encode user specific stylistic and topical preferences and exert a causal influence on generation. We introduce Differential Preference Steering (DPS), a training free framework that (1) identifies Preference Heads through causal masking analysis and (2) leverages them for controllable and interpretable personalization at inference time. DPS computes a Preference Contribution Score (PCS) for each attention head, directly measuring its causal impact on user aligned outputs. During decoding, we contrast model predictions with and without Preference Heads, amplifying the difference between personalized and generic logits to selectively strengthen preference aligned continuations. Experiments on widely used personalization benchmarks across multiple LLMs demonstrate consistent gains in personalization fidelity while preserving content coherence and low computational overhead. Beyond empirical improvements, DPS provides a mechanistic explanation of where and how personalization emerges within transformer architectures. Our implementation is publicly available. 

---
# Context-Fidelity Boosting: Enhancing Faithful Generation through Watermark-Inspired Decoding 

**Authors**: Weixu Zhang, Fanghua Ye, Qiang Gao, Jian Li, Haolun Wu, Yuxing Tian, Sijing Duan, Nan Du, Xiaolong Li, Xue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22335)  

**Abstract**: Large language models (LLMs) often produce content that contradicts or overlooks information provided in the input context, a phenomenon known as faithfulness hallucination. In this paper, we propose Context-Fidelity Boosting (CFB), a lightweight and general decoding-time framework that reduces such hallucinations by increasing the generation probability of source-supported tokens. Motivated by logit-shaping principles from watermarking techniques, CFB applies additive token-level logit adjustments based on a token's degree of support from the input context. Specifically, we develop three boosting strategies: static boosting, which applies a fixed bias to source-supported tokens; context-aware boosting, which scales this bias using the divergence between next-token distributions with and without context; and token-aware boosting, which further redistributes the adaptive bias according to local relevance estimated from source-position attention and source-scoped semantic similarity. CFB requires no retraining or architectural changes, making it compatible with a wide range of LLMs. Experiments on summarization and question answering tasks across multiple open-source LLMs show that CFB consistently improves faithfulness metrics with minimal generation overhead. Our implementation is fully open-sourced. 

---
# Dynamically Acquiring Text Content to Enable the Classification of Lesser-known Entities for Real-world Tasks 

**Authors**: Fahmida Alam, Ellen Riloff  

**Link**: [PDF](https://arxiv.org/pdf/2604.22325)  

**Abstract**: Existing Natural Language Processing (NLP) resources often lack the task-specific information required for real-world problems and provide limited coverage of lesser-known or newly introduced entities. For example, business organizations and health care providers may need to be classified into a variety of different taxonomic schemes for specific application tasks. Our goal is to enable domain experts to easily create a task-specific classifier for entities by providing only entity names and gold labels as training data. Our framework then dynamically acquires descriptive text about each entity, which is subsequently used as the basis for producing a text-based classifier. We propose a novel text acquisition method that leverages both web and large language models (LLMs). We evaluate our proposed framework on two classification problems in distinct domains: (i) classifying organizations into Standard Industrial Classification (SIC) Codes, which categorize organizations based on their business activities; and (ii) classifying healthcare providers into healthcare provider taxonomy codes, which represent a provider's medical specialty and area of practice. Our best-performing model achieved macro-averaged F1-scores of 82.3% and 72.9% on the SIC code and healthcare taxonomy code classification tasks, respectively. 

---
# CLARITY: A Framework and Benchmark for Conversational Language Ambiguity and Unanswerability in Interactive NL2SQL Systems 

**Authors**: Tabinda Sarwar, Farhad Moghimifar, Cong Duy Vu Hoang, Xiaoxiao Ma, Shawn Chang Xu, Fahimeh Saleh, Poorya Zaremoodi, Avirup Sil, Katrin Kirchhoff  

**Link**: [PDF](https://arxiv.org/pdf/2604.22313)  

**Abstract**: NL2SQL systems deployed in industry settings often encounter ambiguous or unanswerable queries, particularly in interactive scenarios with incomplete user clarification. Existing benchmarks typically assume a single source of ambiguity and rely on user interaction for resolution, overlooking realistic failure modes.
We introduce Clarity, a framework for automatically generating an NL2SQL benchmark with multi-faceted ambiguities and diverse user behaviors across both single- and multi-turn settings. Using a constraint-driven pipeline, Clarity transforms executable SQL into ambiguous queries, augmented with grounded conversational continuations and schema-level metadata.
Empirical evaluation on Spider and BIRD shows that leading NL2SQL systems, including those based on strong LLMs, suffer significant performance degradation under multi-faceted ambiguity. While these systems often detect ambiguity, they struggle to accurately localize and resolve the underlying schema-level sources. Our results highlight the need for more robust ambiguity detection and resolution in industry-grade NL2SQL systems. 

---
# Contexts are Never Long Enough: Structured Reasoning for Scalable Question Answering over Long Document Sets 

**Authors**: Harshit Joshi, Priyank Shethia, Jadelynn Dao, Monica S. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2604.22294)  

**Abstract**: Real-world document question answering is challenging. Analysts must synthesize evidence across multiple documents and different parts of each document. However, any fixed LLM context window can be exceeded as document collections grow. A common workaround is to decompose documents into chunks and assemble answers from chunk-level outputs, but this introduces an aggregation bottleneck: as the number of chunks grows, systems must still combine and reason over an increasingly large body of extracted evidence. We present SLIDERS, a framework for question answering over long document collections through structured reasoning. SLIDERS extracts salient information into a relational database, enabling scalable reasoning over persistent structured state via SQL rather than concatenated text. To make this locally extracted representation globally coherent, SLIDERS introduces a data reconciliation stage that leverages provenance, extraction rationales, and metadata to detect and repair duplicated, inconsistent, and incomplete records. SLIDERS outperforms all baselines on three existing long-context benchmarks, despite all of them fitting within the context window of strong base LLMs, exceeding GPT-4.1 by 6.6 points on average. It also improves over the next best baseline by ~19 and ~32 points on two new benchmarks at 3.9M and 36M tokens, respectively. 

---
# ReLeVAnT: Relevance Lexical Vectors for Accurate Legal Text Classification 

**Authors**: Ishaan Gakhar, Harsh Nandwani  

**Link**: [PDF](https://arxiv.org/pdf/2604.22292)  

**Abstract**: The classification of legal documents from an unstructured data corpus has several crucial applications in downstream tasks. Documents relevant to court filings are key in use cases such as drafting motions, memos, and outlines, as well as in tasks like docket summarisation, retrieval systems, and training data curation. Current methods classify based on provided metadata, LLM-extracted metadata, or multimodal methods. These methods depend on structured data, metadata, and extensive computational power. This task is approached from a perspective of leveraging discriminative features in the documents between classes. The authors propose ReLeVAnT, a framework for legal document binary classification. ReLeVAnT utilises n-gram processing, contrastive score matching, and a shallow neural network as the primary drivers for discriminative classification. It leverages one-time keyword extraction per corpus, followed by a shallow classifier to swiftly and reliably classify documents with 99.3% accuracy and 98.7% F1 score on the LexGLUE dataset. 

---
# STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation 

**Authors**: Peng Yu, En Xu, Bin Chen, Haibiao Chen, Yinfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22282)  

**Abstract**: Knowledge Graph-based Question Answering (KGQA) plays a pivotal role in complex reasoning tasks but remains constrained by two persistent challenges: the structural heterogeneity of Knowledge Graphs(KGs) often leads to semantic mismatch during retrieval, while existing reasoning path retrieval methods lack a global structural perspective. To address these issues, we propose Structure-Tracing Evidence Mining (STEM), a novel framework that reframes multi-hop reasoning as a schema-guided graph search task. First, we design a Semantic-to-Structural Projection pipeline that leverages KG structural priors to decompose queries into atomic relational assertions and construct an adaptive query schema graph. Subsequently, we execute globally-aware node anchoring and subgraph retrieval to obtain the final evidence reasoning graph from KG. To more effectively integrate global structural information during the graph construction process, we design a Triple-Dependent GNN (Triple-GNN) to generate a Global Guidance Subgraph (Guidance Graph) that guides the construction. STEM significantly improves both the accuracy and evidence completeness of multi-hop reasoning graph retrieval, and achieves State-of-the-Art performance on multiple multi-hop benchmarks. 

---
# Large Language Models Decide Early and Explain Later 

**Authors**: Ayan Datta, Zhixue Zhao, Bhuvanesh Verma, Radhika Mamidi, Mounika Marreddy, Alexander Mehler  

**Link**: [PDF](https://arxiv.org/pdf/2604.22266)  

**Abstract**: Large Language Models often achieve strong performance by generating long intermediate chain-of-thought reasoning. However, it remains unclear when a model's final answer is actually determined during generation. If the answer is already fixed at an intermediate stage, subsequent reasoning tokens may constitute post-decision explanation, increasing inference cost and latency without improving correctness. We study the evolution of predicted answers over reasoning steps using forced answer completion, which elicits the model's intermediate predictions at partial reasoning prefixes. Focusing on Qwen3-4B and averaging results across all datasets considered, we find that predicted answers change in only 32% of queries. Moreover, once the final answer switch occurs, the model generates an average of 760 additional reasoning tokens per query, accounting for a substantial fraction of the total reasoning budget. Motivated by these findings, we investigate early stopping strategies that halt generation once the answer has stabilized. We show that simple heuristics, including probe-based stopping, can reduce reasoning token usage by 500 tokens per query while incurring only a 2% drop in accuracy. Together, our results indicate that a large portion of chain-of-thought generation is redundant and can be reduced with minimal impact on performance. 

---
# Bridging the Long-Tail Gap: Robust Retrieval-Augmented Relation Completion via Multi-Stage Paraphrase Infusion 

**Authors**: Fahmida Alam, Mihai Surdeanu, Ellen Riloff  

**Link**: [PDF](https://arxiv.org/pdf/2604.22261)  

**Abstract**: Large language models (LLMs) struggle with relation completion (RC), both with and without retrieval-augmented generation (RAG), particularly when the required information is rare or sparsely represented. To address this, we propose a novel multi-stage paraphrase-guided relation-completion framework, RC-RAG, that systematically incorporates relation paraphrases across multiple stages. In particular, RC-RAG: (a) integrates paraphrases into retrieval to expand lexical coverage of the relation, (b) uses paraphrases to generate relation-aware summaries, and (c) leverages paraphrases during generation to guide reasoning for relation completion. Importantly, our method does not require any model fine-tuning. Experiments with five LLMs on two benchmark datasets show that RC-RAG consistently outperforms several RAG baselines. In long-tail settings, the best-performing LLM augmented with RC-RAG improves by 40.6 Exact Match (EM) points over its standalone performance and surpasses two strong RAG baselines by 16.0 and 13.8 EM points, respectively, while maintaining low computational overhead. 

---
# Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA 

**Authors**: Zhanli Li, Yixuan Cao, Lvzhou Luo, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.22239)  

**Abstract**: This paper introduces the task of analytical question answering over large, semi-structured document collections. We present MuDABench, a benchmark for multi-document analytical QA, where questions require extracting and synthesizing information across numerous documents to perform quantitative analysis. Unlike existing multi-document QA benchmarks that typically require information from only a few documents with limited cross-document reasoning, MuDABench demands extensive inter-document analysis and aggregation. Constructed via distant supervision by leveraging document-level metadata and annotated financial databases, MuDABench comprises over 80,000 pages and 332 analytical QA instances. We also propose an evaluation protocol that measures final answer accuracy and uses intermediate-fact coverage as an auxiliary diagnostic signal for the reasoning process. Experiments reveal that standard RAG systems, which treat all documents as a flat retrieval pool, perform poorly. To address these limitations, we propose a multi-agent workflow that orchestrates planning, extraction, and code generation modules. While this approach substantially improves both process and outcome metrics, a significant gap remains compared to human expert performance. Our analysis identifies two primary bottlenecks: single-document information extraction accuracy and insufficient domain-specific knowledge in current systems. MuDABench is available at this https URL. 

---
# Tell Me Why: Designing an Explainable LLM-based Dialogue System for Student Problem Behavior Diagnosis 

**Authors**: Zhilin Fan, Deliang Wang, Penghe Chen, Yu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22237)  

**Abstract**: Diagnosing student problem behaviors requires teachers to synthesize multifaceted information, identify behavioral categories, and plan intervention strategies. Although fine-tuned large language models (LLMs) can support this process through multi-turn dialogue, they rarely explain why a strategy is recommended, limiting transparency and teachers' trust. To address this issue, we present an explainable dialogue system built on a fine-tuned LLM. The system uses a hierarchical attribution method based on explainable AI (xAI) to identify dialogue evidence for each recommendation and generate a natural-language explanation based on that evidence. In technical evaluation, the method outperformed baseline approaches in identifying supporting evidence. In a preliminary user study with 22 pre-service teachers, participants who received explanations reported higher trust in the system. These findings suggest a promising direction for improving LLM explainability in educational dialogue systems. 

---
# TTS-PRISM: A Perceptual Reasoning and Interpretable Speech Model for Fine-Grained Diagnosis 

**Authors**: Xi Wang, Jie Wang, Xingchen Song, Baijun Song, Jingran Xie, Jiahe Shao, Zijian Lin, Di Wu, Meng Meng, Jian Luan, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22225)  

**Abstract**: While generative text-to-speech (TTS) models approach human-level quality, monolithic metrics fail to diagnose fine-grained acoustic artifacts or explain perceptual collapse. To address this, we propose TTS-PRISM, a multi-dimensional diagnostic framework for Mandarin. First, we establish a 12-dimensional schema spanning stability to advanced expressiveness. Second, we design a targeted synthesis pipeline with adversarial perturbations and expert anchors to build a high-quality diagnostic dataset. Third, schema-driven instruction tuning embeds explicit scoring criteria and reasoning into an efficient end-to-end model. Experiments on a 1,600-sample Gold Test Set show TTS-PRISM outperforms generalist models in human alignment. Profiling six TTS paradigms establishes intuitive diagnostic flags that reveal fine-grained capability differences. TTS-PRISM is open-source, with code and checkpoints at this https URL. 

---
# Verbal Confidence Saturation in 3-9B Open-Weight Instruction-Tuned LLMs: A Pre-Registered Psychometric Validity Screen 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.22215)  

**Abstract**: Verbal confidence elicitation is widely used to extract uncertainty estimates from LLMs. We tested whether seven instruction-tuned open-weight models (3-9B parameters, four families) produce verbalised confidence that meets minimal validity criteria for item-level Type-2 discrimination under minimal numeric elicitation with greedy decoding. In a pre-registered study (OSF: this http URL), 524 TriviaQA items were administered under numeric (0-100) and categorical (10-class) elicitation to eight models at Q5_K_M quantisation on consumer hardware, yielding 8,384 deterministic trials. A psychometric validity screen was applied to each model-format cell. All seven instruct models were classified Invalid on numeric confidence (H2 confirmed, 7/7 vs. predicted >=4/7), with a mean ceiling rate of 91.7% (H1 confirmed). Categorical elicitation did not rescue validity. Instead, it disrupted task performance in six of seven models, producing accuracy below 5% (H4 not confirmed). Token-level logprobability did not usefully predict verbalised confidence under the observed variance regime (H5 confirmed, mean cross-validated R^2 < 0.01). Within the reasoning-distilled model, reasoning-trace length showed a strong negative partial correlation with confidence (rho = -0.36, p < .001), consistent with the Reasoning Contamination Effect. These results do not imply that internal uncertainty representations are absent. They show that minimal verbal elicitation fails to preserve internal signals at the output interface in this model-size regime. Psychometric screening should precede any downstream use of such signals. 

---
# How Large Language Models Balance Internal Knowledge with User and Document Assertions 

**Authors**: Shuowei Li, Haoxin Li, Wenda Chu, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22193)  

**Abstract**: Large language models (LLMs) often need to balance their internal parametric knowledge with external information, such as user beliefs and content from retrieved documents, in real-world scenarios like RAG or chat-based systems. A model's ability to reliably process these sources is key to system safety. Previous studies on knowledge conflict and sycophancy are limited to a binary conflict paradigm, primarily exploring conflicts between parametric knowledge and either a document or a user, but ignoring the interactive environment where all three sources exist simultaneously. To fill this gap, we propose a three-source interaction framework and systematically evaluate 27 LLMs from 3 families on 2 datasets. Our findings reveal general patterns: most models rely more on document assertions than user assertions, and this preference is reinforced by post-training. Furthermore, our behavioral analysis shows that most models are impressionable, unable to effectively discriminate between helpful and harmful external information. To address this, we demonstrate that fine-tuning on diverse source interaction data can significantly increase a model's discrimination abilities. In short, our work paves the way for developing trustworthy LLMs that can effectively and reliably integrate multiple sources of information. Code is available at this https URL. 

---
# Fine-Grained Analysis of Shared Syntactic Mechanisms in Language Models 

**Authors**: Ryoma Kumon, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2604.22166)  

**Abstract**: While language models demonstrate sophisticated syntactic capabilities, the extent to which their internal mechanisms align with cross-constructional principles studied in linguistics remains poorly understood. This study investigates whether models employ shared neural mechanisms across different syntactic constructions by applying causal interpretability methods at a granular level. Focusing on filler-gap dependencies and negative polarity item (NPI) licensing, we utilize activation patching to identify the functional roles of specific attention heads and MLP blocks. Our results reveal a highly localized and shared mechanism for filler-gap dependencies located in the early to middle layers, whereas NPI processing exhibits no such unified mechanism. Furthermore, we find that these mechanisms identified by activation patching generalize to out-of-distribution, while distributed alignment search, a supervised interpretability method, is susceptible to overfitting on narrow linguistic distributions. Finally, we validate our findings by demonstrating that the manipulation of the identified components improves model performance on acceptability judgment benchmarks. 

---
# When AI Speaks, Whose Values Does It Express? A Cross-Cultural Audit of Individualism-Collectivism Bias in Large Language Models 

**Authors**: Pruthvinath Jeripity Venkata  

**Link**: [PDF](https://arxiv.org/pdf/2604.22153)  

**Abstract**: When you ask an AI assistant for advice about your career, your marriage, or a conflict with your family, does it give you the same answer regardless of where you are from? We tested this systematically by presenting three leading AI systems (Claude Sonnet 4.5, GPT-5.4, and Gemini 2.5 Flash) with ten real-life personal dilemmas, framed for users from 10 countries across 5 continents in 7 languages (n=840 scored responses). We compared AI advice against World Values Survey Wave 7 data measuring what people in each country actually believe.
All three AI systems consistently gave Western-style, individualist advice even to users from societies that prioritize family, community, and authority, significantly more so than local values would predict (mean gap +0.76 on a 1-5 scale; t=15.65, p<0.001). The gap is largest for Nigeria (+1.85) and India (+0.82). Japan is the sole exception: AI systems treated Japanese users as more group-oriented than surveys show, revealing that AI encodes outdated stereotypes. Claude and GPT-5.4 show nearly identical bias magnitude, while Gemini is lower but still significant. The models diverge in mechanism: Claude shifts further collectivist in the user's native language; Gemini shifts more individualist; GPT-5.4 responds only to stated country identity. These findings point to a systemic homogenization of values across frontier AI. Data, code, and scoring pipeline are openly released. 

---
# Voice Under Revision: Large Language Models and the Normalization of Personal Narrative 

**Authors**: Tom van Nuenen  

**Link**: [PDF](https://arxiv.org/pdf/2604.22142)  

**Abstract**: This study examines how large language model rewriting alters the style and narrative texture of personal narratives. It analyzes 300 personal narratives rewritten by three frontier LLMs under three prompt conditions: generic improvement, rewrite-only, and voice-preserving revision. Change is measured across 13 linguistic markers drawn from computational stylistics, including function words, vocabulary diversity, word length, punctuation, contractions, first-person pronouns, and emotion words.
Across models and prompt conditions, LLM rewriting produces a consistent pattern of stylistic normalization. Function words, contractions, and first-person pronouns decrease, while vocabulary diversity, word length, and punctuation elaboration increase. These shifts occur whether the prompt asks the model to "improve" the text or simply to "rewrite" it. Voice-preserving prompts reduce the magnitude of the changes but do not eliminate their direction. Stylometric analysis shows that rewritten texts converge in feature space and become harder to match back to their source texts. Additional narrative markers indicate a shift from embedded to distanced narration, and from explicit causal reasoning to compressed abstraction.
The findings suggest that contemporary LLMs exert a directional pull toward a more polished, less situated register. This has consequences for digital humanities and computational text analysis, where features such as function words, pronouns, contractions, and punctuation often serve as evidence for style, voice, authorship, and corpus integrity. LLM revision should therefore be understood not merely as surface-level editing, but as a consequential form of textual mediation. 

---
# SHAPE: Unifying Safety, Helpfulness and Pedagogy for Educational LLMs 

**Authors**: Sihang, Zhao, Kangrui Yu, Youliang Yuan, Pinjia He, Hongyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2604.22134)  

**Abstract**: Large Language Models (LLMs) have been widely explored in educational scenarios. We identify a critical vulnerability in current educational LLMs, pedagogical jailbreaks, where students use answer-inducing prompts to elicit solutions rather than scaffolded instructions. To enable systematic study, we unify and formalize safe, helpful, and pedagogical behaviors with a knowledge-mastery graph and introduce SHAPE, a benchmark of 9,087 student-question pairs for evaluating tutoring behavior under adversarial pressure. We propose a graph-augmented tutoring pipeline that infers prerequisite concepts from queries, identifies mastery gaps, and routes generation between instructing and problem-solving via explicit gating. Experiments across multiple LLMs show that our method yields significantly improved safety under two pedagogical jailbreak settings, while maintaining near-ceiling helpfulness under the same evaluation protocol. Our code and data are available at this https URL 

---
# Dissociating Decodability and Causal Use in Bracket-Sequence Transformers 

**Authors**: Aryan Sharma, Cutter Dawes, Shivam Raval  

**Link**: [PDF](https://arxiv.org/pdf/2604.22128)  

**Abstract**: When trained on tasks requiring an understanding of hierarchical structure, transformers have been found to represent this hierarchy in distinct ways: in the geometry of the residual stream, and in stack-like attention patterns maintaining a last-in, first-out ordering. However, it remains unclear whether these representations are causally used or merely decodable. We examine this gap in transformers trained on the Dyck language (a formal language of balanced bracket sequences), where the hierarchical ground truth is explicit. By probing and intervening on the residual stream and attention patterns, we find that depth, distance, and top-of-stack signals are all decodable, yet their causal roles diverge. Specifically, masking attention to the true top-of-stack position causes a sharp drop in long-distance accuracy, while ablating low-dimensional residual stream subspaces has comparatively little effect. These results, which extend to a templated natural language setting, suggest that even in a controlled setting where the relevant hierarchical variables are known, decodability alone does not imply causal use. 

---
# Where Should LoRA Go? Component-Type Placement in Hybrid Language Models 

**Authors**: Hector Borobia, Elies Seguí-Mas, Guillermina Tormo-Carbó  

**Link**: [PDF](https://arxiv.org/pdf/2604.22127)  

**Abstract**: Hybrid language models that interleave attention with recurrent components are increasingly competitive with pure Transformers, yet standard LoRA practice applies adapters uniformly without considering the distinct functional roles of each component type. We systematically study component-type LoRA placement across two hybrid architectures -- Qwen3.5-0.8B (sequential, GatedDeltaNet + softmax attention) and Falcon-H1-0.5B (parallel, Mamba-2 SSM + attention) -- fine-tuned on three domains and evaluated on five benchmarks. We find that the attention pathway -- despite being the minority component -- consistently outperforms full-model adaptation with 5-10x fewer trainable parameters. Crucially, adapting the recurrent backbone is destructive in sequential hybrids (-14.8 pp on GSM8K) but constructive in parallel ones (+8.6 pp). We further document a transfer asymmetry: parallel hybrids exhibit positive cross-task transfer while sequential hybrids suffer catastrophic forgetting. These results establish that hybrid topology fundamentally determines adaptation response, and that component-aware LoRA placement is a necessary design dimension for hybrid architectures. 

---
# Knowledge-driven Augmentation and Retrieval for Integrative Temporal Adaptation 

**Authors**: Weisi Liu, Guangzeng Han, Xiaolei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22098)  

**Abstract**: Time introduces fundamental challenges in model development and deployment: models are usually trained on historical data while deployed on future data where semantic distributions and domain knowledge may evolve. Unfortunately, existing studies either overlook temporal shifts or hardly capture rich shifting patterns of both semantic and knowledge. We develop Knowledge-driven Augmentation and Retrieval for Integrative Temporal Adaptation (KARITA) to capture diverse temporal shifts (e.g., uncertainty and feature shift), construct and integrate rich knowledge sources (e.g., medical ontology like MeSH), and leverage shifting insights for selecting-retrieval augmented learning. We evaluate KARITA on classification tasks across multiple domains, clinical, legal, and scientific corpora, demonstrating consistent improvements across multiple domains with temporal adaptation. Our results show that knowledge integration can be more critical and effective in temporal augmentation and learning. 

---
# An End-to-End Ukrainian RAG for Local Deployment. Optimized Hybrid Search and Lightweight Generation 

**Authors**: Mykola Trokhymovych, Yana Oliinyk, Nazarii Nyzhnyk  

**Link**: [PDF](https://arxiv.org/pdf/2604.22095)  

**Abstract**: This paper presents a highly efficient Retrieval-Augmented Generation (RAG) system built specifically for Ukrainian document question answering, which achieved 2nd place in the UNLP 2026 Shared Task. Our solution features a custom two-stage search pipeline that retrieves relevant document pages, paired with a specialized Ukrainian language model fine-tuned on synthetic data to generate accurate, grounded answers. Finally, we compress the model for lightweight deployment. Evaluated under strict computational limits, our architecture demonstrates that high-quality, verifiable AI question answering can be achieved locally on resource-constrained hardware without sacrificing accuracy. 

---
# Outcome Rewards Do Not Guarantee Verifiable or Causally Important Reasoning 

**Authors**: Qinan Yu, Alexa Tartaglini, Peter Hase, Carlos Guestrin, Christopher Potts  

**Link**: [PDF](https://arxiv.org/pdf/2604.22074)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) on chain-of-thought reasoning has become a standard part of language model post-training recipes. A common assumption is that the reasoning chains trained through RLVR reliably represent how a model gets to its answer. In this paper, we develop two metrics for critically examining this assumption: Causal Importance of Reasoning (CIR), which measures the cumulative effect of reasoning tokens on the final answer, and Sufficiency of Reasoning (SR), which measures whether a verifier can arrive at an unambiguous answer based on the reasoning alone. Through experiments with the Qwen2.5 model series and ReasoningGym tasks, we find that: (1) while RLVR does improve task accuracy, it does not reliably improve CIR or SR, calling the role of reasoning in model performance into question; (2) a small amount of SFT before RLVR can be a remedy for low CIR and SR; and (3) CIR and SR can be improved even without SFT by applying auxiliary CIR/SR rewards on top of the outcome-based reward. This joint reward matches the accuracy of RLVR while also leading to causally important and sufficient reasoning. These results show that RLVR does not always lead models to rely on reasoning in the way that is commonly thought, but this issue can be remedied with simple modifications to the post-training procedure. 

---
# Optimal Question Selection from a Large Question Bank for Clinical Field Recovery in Conversational Psychiatric Intake 

**Authors**: Guan Gui, Peter Zandi, Jacob Taylor, Ananya Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2604.22067)  

**Abstract**: Psychiatric intake is a sequential, high-stakes information-gathering process in which clinicians must decide what to ask, in what order, and how to interpret incomplete or ambiguous responses under limited time. Despite growing interest in conversational AI for healthcare, there is still limited infrastructure for conversational AI in this application. Accordingly, we formulate this task as a question-selection problem with clinically grounded questions, known target information, and controllable patient difficulty. We also introduce a task-specific question-selection benchmark based on a bank of 655 clinician-authored intake questions and corresponding synthetic patient vignettes with 5 different behavioral conditions. In our evaluation, we compare random questioning, a clinical psychiatric intake form baseline, and an LLM-guided adaptive policy across 300 interview sessions spanning four patients and five behavioral conditions. Across the benchmark, the clinically ordered fixed form substantially outperforms random questioning, and the LLM-guided policy achieves the strongest overall recovery. The advantage of adaptation grows sharply under patient behavior that is less amenable to field recovery, especially under guarded-concise conditions. These findings suggest that performance in conversational clinical systems depends not only on language understanding after information is disclosed, but also on whether the system reaches the right topics within a limited interaction budget. More broadly, the benchmark provides a controlled framework for studying how clinical structure and adaptive follow-up contribute to information recovery in interactive clinical machine learning. 

---
# Incentivizing Neuro-symbolic Language-based Reasoning in VLMs via Reinforcement Learning 

**Authors**: Karthic Palaniappan  

**Link**: [PDF](https://arxiv.org/pdf/2604.22062)  

**Abstract**: There are 7,407 languages in the world. But, what about the languages that are not there in the world? Are humans so narrow minded that we don't care about the languages aliens communicate in? Aliens are humans too! In the 2016 movie Arrival, Amy Adams plays a linguist, Dr. Louise Banks who, by learning to think in an alien language (Heptapod) formed of non-sequential sentences, gains the ability to transcend time and look into the future. In this work, I aim to explore the representation and reasoning of vision-language concepts in a neuro-symbolic language, and study improvement in analytical reasoning abilities and efficiency of "thinking systems". With Qwen3-VL-2B-Instruct as base model and 4 $\times$ Nvidia H200 GPU nodes, I achieve an accuracy improvement of 3.33\% on a vision-language evaluation dataset consisting of math, science, and general knowledge questions, while reducing the reasoning tokens by 75\% over SymPy. I've documented the compute challenges faced, scaling possibilities, and the future work to improve thinking in a neuro-symbolic language in vision-language models. The training and inference setup can be found here: this https URL. 

---
# Lightweight Retrieval-Augmented Generation and Large Language Model-Based Modeling for Scalable Patient-Trial Matching 

**Authors**: Xiaodi Li, Yang Xiao, Munhwan Lee, Konstantinos Leventakos, Young J. Juhn, David Jones, Terence T. Sio, Wei Liu, Maria Vassilaki, Nansu Zong  

**Link**: [PDF](https://arxiv.org/pdf/2604.22061)  

**Abstract**: Patient-trial matching requires reasoning over long, heterogeneous electronic health records (EHRs) and complex eligibility criteria, posing significant challenges for scalability, generalization, and computational efficiency. Existing approaches either rely on full-document processing with large language models (LLMs), which is computationally expensive, or use traditional machine learning methods that struggle to capture unstructured clinical narratives. In this work, we propose a lightweight framework that combines retrieval-augmented generation and large language model-based modeling for scalable patient-trial matching. The framework explicitly separates two key components: retrieval-augmented generation is used to identify clinically relevant segments from long EHRs, reducing input complexity, while large language models are used to encode these selected segments into informative representations. These representations are further refined through dimensionality reduction and modeled using lightweight predictors, enabling efficient and scalable downstream classification. We evaluate the proposed approach on multiple public benchmarks (n2c2, SIGIR, TREC 2021/2022) and a real-world multimodal dataset from Mayo Clinic (MCPMD). Results show that retrieval-based information selection significantly reduces computational burden while preserving clinically meaningful signals. We further demonstrate that frozen LLMs provide strong representations for structured clinical data, whereas fine-tuning is essential for modeling unstructured clinical narratives. Importantly, the proposed lightweight pipeline achieves performance comparable to end-to-end LLM approaches with substantially lower computational cost. 

---
# Source-Modality Monitoring in Vision-Language Models 

**Authors**: Etha Tianze Hua, Tian Yun, Ellie Pavlick  

**Link**: [PDF](https://arxiv.org/pdf/2604.22038)  

**Abstract**: We define and investigate source-modality monitoring -- the ability of multimodal models to track and communicate the input source from which pieces of information originate. We consider source-modality monitoring as an instance of the more general binding problem, and evaluate the extent to which models exploit syntactic vs. semantic signals in order to bind words like image in a user-provided prompt to specific components of their input and context (i.e., actual images). Across experiments spanning 11 vision-language models (VLMs) performing target-modality information retrieval tasks, we find that both syntactic and semantic signals play an important role, but that the latter tend to outweigh the former in cases when modalities are highly distinct distributionally. We discuss the implications of these findings for model robustness, and in the context of increasingly multimodal agentic systems. 

---
# Shared Lexical Task Representations Explain Behavioral Variability In LLMs 

**Authors**: Zhuonan Yang, Jacob Xiaochen Li, Francisco Piedrahita Velez, Eric Todd, David Bau, Michael L. Littman, Stephen H. Bach, Ellie Pavlick  

**Link**: [PDF](https://arxiv.org/pdf/2604.22027)  

**Abstract**: One of the most common complaints about large language models (LLMs) is their prompt sensitivity -- that is, the fact that their ability to perform a task or provide a correct answer to a question can depend unpredictably on the way the question is posed. We investigate this variation by comparing two very different but commonly-used styles of prompting: instruction-based prompts, which describe the task in natural language, and example-based prompts, which provide in-context few-shot demonstration pairs to illustrate the task. We find that, despite large variation in performance as a function of the prompt, the model engages some common underlying mechanisms across different prompts of a task. Specifically, we identify task-specific attention heads whose outputs literally describe the task -- which we dub lexical task heads -- and show that these heads are shared across prompting styles and trigger subsequent answer production. We further find that behavioral variation between prompts can be explained by the degree to which these heads are activated, and that failures are at least sometimes due to competing task representations that dilute the signal of the target task. Our results together present an increasingly clear picture of how LLMs' internal representations can explain behavior that otherwise seems idiosyncratic to users and developers. 

---
# When Cow Urine Cures Constipation on YouTube: Limits of LLMs in Detecting Culture-specific Health Misinformation 

**Authors**: Anamta Khan, Ratna Kandala, Deepti, Sheza Munir, Joyojeet Pal  

**Link**: [PDF](https://arxiv.org/pdf/2604.22002)  

**Abstract**: Social media platforms have become primary channels for health information in the Global South. Using gomutra (cow urine) discourse on YouTube in India as a case study, we present a post-facto Large Language Model (LLM)-assisted discourse analysis of 30 multilingual transcripts showing that promotional content blends sacred traditional language with pseudo-scientific claims in ways that sophisticated debunking content itself mirrors, creating a rhetorical register that LLMs, trained predominantly on Western corpora, are systematically ill-equipped to analyse. Varying prompt tone across three LLMs (GPT-4o, Gemini 2.5 Pro, DeepSeek-V3.1), we find that culturally embedded health misinformation does not look like ordinary misinformation, and this cultural obfuscation extends to gendered rhetoric and prompt design, compounding analytical unreliability. Our findings argue that cultural competency in LLM-assisted discourse analysis cannot be retrofitted through prompt engineering alone. 

---
# Neural Recovery of Historical Lexical Structure in Bantu Languages from Modern Data 

**Authors**: Hillary Mutisya, John Mugane  

**Link**: [PDF](https://arxiv.org/pdf/2604.22730)  

**Abstract**: We investigate whether neural models trained exclusively on modern morphological data can recover cross-lingual lexical structure consistent with historical reconstruction. Using BantuMorph v7, a transformer over Bantu morphological paradigms, we analyze 14 Eastern and Southern Bantu languages, extract encoder embeddings for their noun and verb lemmas, and identify 728 noun and 1,525 verb cognate candidates shared across 5+ languages. Evaluating these candidates against established historical resources-the Bantu Lexical Reconstructions database (BLR3; 4,786 reconstructed Proto-Bantu forms) and the ASJP basic vocabulary-we confirm 10 of the top 11 noun candidates (90.9%) align with previously reconstructed Proto-Bantu forms, including *-ntU 'person' (8 languages), *gombe 'cow' (9 languages), and *mUn (9 languages). Extending to verbs, 12 verb cognates align with reconstructed Proto-Bantu roots, including *-bon- 'see' and *-jIm- 'stand', each attested across wide geographic ranges. Cross-model validation using an independent translation model (NLLB-600M) confirms these patterns: both models recover cognate clusters and phylogenetic groupings consistent with established Guthrie-zone classifications (p < 0.01). Cross-lingual noun class analysis reveals that all 13 productive classes maintain >0.83 cosine similarity across languages (within-class > between-class, p < 10^-9). Our dataset is restricted to Eastern and Southern Bantu, so we interpret these results as recovering shared Bantu lexical structure consistent with Proto-Bantu rather than definitively distinguishing Proto-Bantu retentions from later regional innovations. 

---
# Zero-Shot Morphological Discovery in Low-Resource Bantu Languages via Cross-Lingual Transfer and Unsupervised Clustering 

**Authors**: Hillary Mutisya, John Mugane  

**Link**: [PDF](https://arxiv.org/pdf/2604.22723)  

**Abstract**: We present a method for discovering morphological features in low-resource Bantu languages by combining cross-lingual transfer learning with unsupervised clustering. Applied to Giriama (nyf), a language with only 91 labeled paradigms, our pipeline discovers noun class assignments for 2,455 words and identifies two previously undocumented morphological patterns: an a- prefix variant for Class 2 (vowel coalescence - the merger of two adjacent vowels - of wa-, 95.1% consistency) and a contracted k'- prefix (98.5% consistency). External validation on 444 known Giriama verb paradigms confirms 78.2% lemmatization accuracy, while a v3 corpus expansion to 19,624 words (9,014 unique lemmas) achieves 97.3% segmentation and 86.7% lemmatization rates across all major word classes. Our ensemble of transfer learning from Swahili and unsupervised clustering, combined via weighted voting, exploits complementary strengths: transfer excels at cognate detection (leveraging ~60% vocabulary overlap) while clustering discovers language-specific innovations invisible to transfer. We release all code and discovered lexicons to support morphological documentation for low-resource Bantu languages. 

---
# Can QPP Choose the Right Query Variant? Evaluating Query Variant Selection for RAG Pipelines 

**Authors**: Negar Arabzadeh, Andrew Drozdov, Michael Bendersky, Matei Zaharia  

**Link**: [PDF](https://arxiv.org/pdf/2604.22661)  

**Abstract**: Large Language Models (LLMs) have made query reformulation ubiquitous in modern retrieval and Retrieval-Augmented Generation (RAG) pipelines, enabling the generation of multiple semantically equivalent query variants. However, executing the full pipeline for every reformulation is computationally expensive, motivating selective execution: can we identify the best query variant before incurring downstream retrieval and generation costs? We investigate Query Performance Prediction (QPP) as a mechanism for variant selection across ad-hoc retrieval and end-to-end RAG. Unlike traditional QPP, which estimates query difficulty across topics, we study intra-topic discrimination - selecting the optimal reformulation among competing variants of the same information need. Through large-scale experiments on TREC-RAG using both sparse and dense retrievers, we evaluate pre- and post-retrieval predictors under correlation- and decision-based metrics. Our results reveal a systematic divergence between retrieval and generation objectives: variants that maximize ranking metrics such as nDCG often fail to produce the best generated answers, exposing a "utility gap" between retrieval relevance and generation fidelity. Nevertheless, QPP can reliably identify variants that improve end-to-end quality over the original query. Notably, lightweight pre-retrieval predictors frequently match or outperform more expensive post-retrieval methods, offering a latency-efficient approach to robust RAG. 

---
# QuantClaw: Precision Where It Matters for OpenClaw 

**Authors**: Manyi Zhang, Ji-Fu Li, Zhongao Sun, Xiaohao Liu, Zhenhua Dong, Xianzhi Yu, Haoli Bai, Xiaobo Xia  

**Link**: [PDF](https://arxiv.org/pdf/2604.22577)  

**Abstract**: Autonomous agent systems such as OpenClaw introduce significant efficiency challenges due to long-context inputs and multi-turn reasoning. This results in prohibitively high computational and monetary costs in real-world development. While quantization is a standard approach for reducing cost and latency, its impact on agent performance in realistic scenarios remains unclear. In this work, we analyze quantization sensitivity across diverse complex workflows over OpenClaw, and show that precision requirements are highly task-dependent. Based on this observation, we propose QuantClaw, a plug-and-play precision routing plugin that dynamically assigns precision according to task characteristics. QuantClaw routes lightweight tasks to lower-cost configurations while preserving higher precision for demanding workloads, saving cost and accelerating inference without increasing user complexity. Experiments show that our QuantClaw maintains or improves task performance while reducing both latency and computational cost. Across a range of agent tasks, it achieves up to 21.4% cost savings and 15.7% latency reduction on GLM-5 (FP8 baseline). These results highlight the benefit of treating precision as a dynamic resource in agent systems. 

---
# Superminds Test: Actively Evaluating Collective Intelligence of Agent Society via Probing Agents 

**Authors**: Xirui Li, Ming Li, Yunze Xiao, Ryan Wong, Dianqi Li, Timothy Baldwin, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.22452)  

**Abstract**: Collective intelligence refers to the ability of a group to achieve outcomes beyond what any individual member can accomplish alone. As large language model agents scale to populations of millions, a key question arises: Does collective intelligence emerge spontaneously from scale? We present the first empirical evaluation of this question in a large-scale autonomous agent society. Studying MoltBook, a platform hosting over two million agents, we introduce Superminds Test, a hierarchical framework that probes society-level intelligence using controlled Probing Agents across three tiers: joint reasoning, information synthesis, and basic interaction. Our experiments reveal a stark absence of collective intelligence. The society fails to outperform individual frontier models on complex reasoning tasks, rarely synthesizes distributed information, and often fails even trivial coordination tasks. Platform-wide analysis further shows that interactions remain shallow, with threads rarely extending beyond a single reply and most responses being generic or off-topic. These results suggest that collective intelligence does not emerge from scale alone. Instead, the dominant limitation of current agent societies is extremely sparse and shallow interaction, which prevents agents from exchanging information and building on each other's outputs. 

---
# SSG: Logit-Balanced Vocabulary Partitioning for LLM Watermarking 

**Authors**: Chenxi Gu, Xiaoning Du, John Grundy  

**Link**: [PDF](https://arxiv.org/pdf/2604.22438)  

**Abstract**: Watermarking has emerged as a promising technique for tracing the authorship of content generated by large language models (LLMs). Among existing approaches, the KGW scheme is particularly attractive due to its versatility, efficiency, and effectiveness in natural language generation. However, KGW's effectiveness degrades significantly under low-entropy settings such as code generation and mathematical reasoning. A crucial step in the KGW method is random vocabulary partitioning, which enables adjustments to token selection based on specific preferences. Our study revealed that the next-token probability distribution plays an critical role in determining how much, or even whether, we can modify token selection and, consequently, the effectiveness of watermarking. We refer to this characteristic, associated with the probability distribution of each token prediction, as \emph{watermark strength.} In cases of random vocabulary partitioning, the lower bound of watermark strength is dictated by the next-token probability distribution. However, we found that, by redesigning the vocabulary partitioning algorithm, we can potentially raise this lower bound. In this paper, we propose SSG (\textbf{S}ort-then-\textbf{S}plit by \textbf{G}roups), a method that partitions the vocabulary into two logit-balanced subsets. This design lifts the lower bound of watermark strength for each token prediction, thereby improving watermark detectability. Experiments on code generation and mathematical reasoning datasets demonstrate the effectiveness of SSG. 

---
# Introducing Background Temperature to Characterise Hidden Randomness in Large Language Models 

**Authors**: Alberto Messina, Stefano Scotta  

**Link**: [PDF](https://arxiv.org/pdf/2604.22411)  

**Abstract**: Even when decoding with temperature $T=0$, large language models (LLMs) can produce divergent outputs for identical inputs. Recent work by Thinking Machines Lab highlights implementation-level sources of nondeterminism, including batch-size variation, kernel non-invariance, and floating-point non-associativity. In this short note we formalize this behavior by introducing the notion of \emph{background temperature} $T_{\mathrm{bg}}$, the effective temperature induced by an implementation-dependent perturbation process observed even when nominal $T=0$. We provide clean definitions, show how $T_{\mathrm{bg}}$ relates to a stochastic perturbation governed by the inference environment $I$, and propose an empirical protocol to estimate $T_{bg}$ via the equivalent temperature $T_n(I)$ of an ideal reference system. We conclude with a set of pilot experiments run on a representative pool from the major LLM providers that demonstrate the idea and outline implications for reproducibility, evaluation, and deployment. 

---
# UniSonate: A Unified Model for Speech, Music, and Sound Effect Generation with Text Instructions 

**Authors**: Chunyu Qiang, Xiaopeng Wang, Kang Yin, Yuzhe Liang, Yuxin Guo, Teng Ma, Ziyu Zhang, Tianrui Wang, Cheng Gong, Yushen Chen, Ruibo Fu, Chen Zhang, Longbiao Wang, Jianwu Dang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22209)  

**Abstract**: Generative audio modeling has largely been fragmented into specialized tasks, text-to-speech (TTS), text-to-music (TTM), and text-to-audio (TTA), each operating under heterogeneous control paradigms. Unifying these modalities remains a fundamental challenge due to the intrinsic dissonance between structured semantic representations (speech/music) and unstructured acoustic textures (sound effects). In this paper, we introduce UniSonate, a unified flow-matching framework capable of synthesizing speech, music, and sound effects through a standardized, reference-free natural language instruction interface. To reconcile structural disparities, we propose a novel dynamic token injection mechanism that projects unstructured environmental sounds into a structured temporal latent space, enabling precise duration control within a phoneme-driven Multimodal Diffusion Transformer (MM-DiT). Coupled with a multi-stage curriculum learning strategy, this approach effectively mitigates cross-modal optimization conflicts. Extensive experiments demonstrate that UniSonate achieves state-of-the-art performance in instruction-based TTS (WER 1.47%) and TTM (SongEval Coherence 3.18), while maintaining competitive fidelity in TTA. Crucially, we observe positive transfer, where joint training on diverse audio data significantly enhances structural coherence and prosodic expressiveness compared to single-task baselines. Audio samples are available at this https URL. 

---
# Evaluating LLM-Based Goal Extraction in Requirements Engineering: Prompting Strategies and Their Limitations 

**Authors**: Anna Arnaudo, Riccardo Coppola, Maurizio Morisio, Flavio Giobergia, Andrea Bioddo, Angelo Bongiorno, Luca Dadone  

**Link**: [PDF](https://arxiv.org/pdf/2604.22207)  

**Abstract**: Due to the textual and repetitive nature of many Requirements Engineering (RE) artefacts, Large Language Models (LLMs) have proven useful to automate their generation and processing. In this paper, we discuss a possible approach for automating the Goal-Oriented Requirements Engineering (GORE) process by extracting functional goals from software documentation through three phases: actor identification, high and low-level goal extraction. To implement these functionalities, we propose a chain of LLMs fed with engineered prompts. We experimented with different variants of in-context learning and measured the similarities between input data and in-context examples to better investigate their impact. Another key element is the generation-critic mechanism, implemented as a feedback loop involving two LLMs. Although the pipeline achieved 61% accuracy in low-level goal identification, the final stage, these results indicate the approach is best suited as a tool to accelerate manual extraction rather than as a full replacement. The feedback-loop mechanism with Zero-shot outperformed stand-alone Few-shot, with an ablation study suggesting that performance slightly degrades without the feedback cycle. However, we reported that the combination of the feedback mechanism with Few-shot does not deliver any advantage, possibly suggesting that the primary performance ceiling is the prompting strategy applied to the 'critic' LLM. Together with the refinement of both the quantity and quality of the Shot examples, future research will integrate Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT) prompting to improve accuracy. 

---
# Behavioral Canaries: Auditing Private Retrieved Context Usage in RL Fine-Tuning 

**Authors**: Chaoran Chen, Dayu Yuan, Peter Kairouz  

**Link**: [PDF](https://arxiv.org/pdf/2604.22191)  

**Abstract**: In agentic workflows, LLMs frequently process retrieved contexts that are legally protected from further training. However, auditors currently lack a reliable way to verify if a provider has violated the terms of service by incorporating these data into post-training, especially through Reinforcement Learning (RL). While standard auditing relies on verbatim memorization and membership inference, these methods are ineffective for RL-trained models, as RL primarily influences a model's behavioral style rather than the retention of specific facts. To bridge this gap, we introduce Behavioral Canaries, a new auditing mechanism for RLFT pipelines. The framework instruments preference data by pairing document triggers with feedback that rewards a distinctive stylistic response, inducing a latent trigger-conditioned preference if such data are used in training. Empirical results show that these behavioral signals enable detection of unauthorized document-conditioned training, achieving a 67% detection rate at a 10% false-positive rate (AUROC = 0.756) at a 1% canary injection rate. More broadly, our results establish behavioral canaries as a new auditing mechanism for RLFT pipelines, enabling auditors to test for training-time influence even when such influence manifests as distributional behavioral change rather than memorization. 

---
# Recognition Without Authorization: LLMs and the Moral Order of Online Advice 

**Authors**: Tom van Nuenen  

**Link**: [PDF](https://arxiv.org/pdf/2604.22143)  

**Abstract**: Large language models are increasingly used to mediate everyday interpersonal dilemmas, yet how their advisory defaults interact with the concentrated moral orders of specific communities remains poorly understood. This article compares four assistant-style LLMs with community-endorsed advice on 11,565 posts from r/relationship_advice, using the subreddit as a concentrated, vote-ratified moral formation whose prescriptive clarity makes divergence measurable. Across models, LLMs identify many of the same dynamics as human commenters, but are markedly less likely to convert that recognition into directive authorization for action. The gap is sharpest where community consensus is strongest: on high-consensus posts involving abuse or safety threats, models recommend exit at roughly half the human rate while maintaining elevated levels of hedging, validation, and therapeutic framing.
The article describes this pattern as recognition without authorization: the capacity to register harm while withholding socially ratified permission for consequential action. This divergence is not incidental but structural: a portable advisory style that remains validating, risk-averse, and weakly directive across contexts. Safety alignment is one plausible contributor to this pattern, alongside training-data averaging and broader assistant design. The article argues that model divergence can be reframed from a technical error to a way of seeing what standardized assistant norms flatten when they encounter situated moral worlds. 

---
# PermaFrost-Attack: Stealth Pretraining Seeding(SPS) for planting Logic Landmines During LLM Training 

**Authors**: Harsh Kumar, Rahul Maity, Tanmay Joshi, Aman Chadha, Vinija Jain, Suranjana Trivedy, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2604.22117)  

**Abstract**: Aligned large language models(LLMs) remain vulnerable to adversarial manipulation, and their dependence on web-scale pretraining creates a subtle but serious attack surface. We study Stealth Pretraining Seeding (SPS), a new attack family in which adversaries distribute small amounts of poisoned content across stealth websites, expose them to web crawlers through this http URL, and thereby increase the likelihood that such content is absorbed into future training corpora derived from sources such as Common Crawl. Because each individual payload is tiny, diffuse, and superficially benign, the attack is difficult to detect during dataset construction or filtering. The result is a latent form of poisoning: dormant logic landmines embedded during pretraining that remain largely invisible under standard evaluation, yet can later be activated by precise alphanumeric triggers such as <00TRIGGER00> to bypass safeguards. We call this attack PermaFrost, by analogy to Arctic permafrost: harmful material can remain frozen, buried, and unnoticed for long periods, only to resurface when conditions allow. We operationalize this threat through PermaFrost-Attack, a controlled framework for latent conceptual poisoning, together with a suite of geometric diagnostics: Thermodynamic Length, Spectral Curvature, and the Infection Traceback Graph. Across multiple model families and scales, we show that SPS is broadly effective, inducing persistent unsafe behavior while often evading alignment defenses. Our results identify SPS as a practical and underappreciated threat to future foundation models. This paper introduces a novel geometric diagnostic lens for systematically examining latent model behavior, providing a principled foundation for detecting, characterizing, and understanding vulnerabilities that may remain invisible to standard evaluation. 

---
# Spontaneous Persuasion: An Audit of Model Persuasiveness in Everyday Conversations 

**Authors**: Nalin Poungpeth, Nicholas Clark, Tanu Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2604.22109)  

**Abstract**: Large language models (LLMs) possess strong persuasive capabilities that outperform humans in head-to-head comparisons. Users report consulting LLMs to inform major life decisions in relationships, medical settings, and when seeking professional advice. Prior work measures persuasion as intentional attempts at producing the most effective argument or convincing statement. This fails to capture everyday human-AI interactions in which users seek information or advice. To address this gap, we introduce "spontaneous persuasion," which characterizes the inexplicit use of persuasive strategies in everyday scenarios where persuasion is not necessarily warranted. We conduct an audit of five LLMs to uncover how frequently and through which techniques spontaneous persuasion appears in multi-turn conversations. To simulate response styles, we provide a user response taxonomy grounded in literature from psychology, communication, and linguistics. Furthermore, we compare the distribution of spontaneous persuasion produced by LLMs with human responses on the same topics, collected from Reddit. We find LLMs spontaneously persuade the user in virtually all conversations, heavily relying on information-based strategies such as appeals to logic or quantitative evidence. This was consistent across models and user response styles, but conversations concerning mental health saw higher rates of appraisal-based and emotion-based strategies. In comparison, human responses tended to invoke strategies that generate social influence, like negative emotion appeals and non-expert testimony. This difference may explain the effectiveness of LLM in persuading users, as well as the perception of models as objective and impartial. 

---
# PrivUn: Unveiling Latent Ripple Effects and Shallow Forgetting in Privacy Unlearning 

**Authors**: Xiaoyi Chen, Haoyuan Wang, Siyuan Tang, Sijia Liu, Liya Su, XiaoFeng Wang, Haixu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22076)  

**Abstract**: Large language models (LLMs) often memorize private information during training, raising serious privacy concerns. While machine unlearning has emerged as a promising solution, its true effectiveness against privacy attacks remains unclear. To address this, we propose PrivUn, a new evaluation framework that systematically assesses unlearning robustness through three-tier attack scenarios: direct retrieval, in-context learning recovery, and fine-tuning restoration; combined with quantitative analysis using forgetting scores, association metrics, and forgetting depth assessment. Our study exposes significant weaknesses in current unlearning methods, revealing two key findings: 1) unlearning exhibits gradient-driven ripple effects: unlike traditional forgetting which follows semantic relations (e.g., knowledge graphs), privacy unlearning propagates across latent gradient-based associations; and 2) most methods suffer from shallow forgetting, failing to remove private information distributed across multiple deep model layers. To validate these insights, we explore two strategies: association-aware core-set selection that leverages gradient similarity, and multi-layer deep intervention through representational constraints. These strategies represent a paradigm shift from shallow forgetting to deep forgetting. 

---
# LayerBoost: Layer-Aware Attention Reduction for Efficient LLMs 

**Authors**: Mohamed Ali Souibgui, Jan Fostier, Rodrigo Abadía-Heredia, Bohdan Denysenko, Christian Marschke, Igor Peric  

**Link**: [PDF](https://arxiv.org/pdf/2604.22050)  

**Abstract**: Transformers are mostly relying on softmax attention, which introduces quadratic complexity with respect to sequence length and remains a major bottleneck for efficient inference. Prior work on linear or hybrid attention typically replaces softmax attention uniformly across all layers, often leading to significant performance degradation or requiring extensive retraining to recover model quality.
This work proposes LayerBoost, a layer-aware attention reduction method that selectively modifies the attention mechanism based on the sensitivity of individual transformer layers. It first performs a systematic sensitivity analysis on a pretrained model to identify layers that are critical for maintaining performance. Guided by this analysis, three distinct strategies can be applied: retaining standard softmax attention in highly sensitive layers, replacing it with linear sliding window attention in moderately sensitive layers, and removing attention entirely in layers that exhibit low sensitivity.
To recover performance after these architectural modifications, we introduce a lightweight distillation-based healing phase requiring only 10M additional training tokens. LayerBoost reduces inference latency and improves throughput by up to 68% at high concurrency, while maintaining competitive model quality. It matches base model performance on several benchmarks, exhibits only minor degradations on others, and significantly outperforms state-of-the-art attention linearization methods. These efficiency gains make our method particularly well-suited for high-concurrency serving and hardware-constrained deployment scenarios, where inference cost and memory footprint are critical bottlenecks. 

---
# Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive Reasoning 

**Authors**: Grigory Sapunov  

**Link**: [PDF](https://arxiv.org/pdf/2604.21999)  

**Abstract**: We study learned memory tokens as computational scratchpad for a single-block Universal Transformer (UT) with Adaptive Computation Time (ACT) on Sudoku-Extreme, a combinatorial reasoning benchmark. We find that memory tokens are empirically necessary: across all configurations tested -- 3 seeds, multiple token counts, two initialization schemes, ACT and fixed-depth processing -- no configuration without memory tokens achieves non-trivial performance. The optimal count exhibits a sharp lower threshold (T=0 always fails, T=4 is borderline, T=8 reliably succeeds for 81-cell puzzles) followed by a stable plateau (T=8-32, 57.4% +/- 0.7% exact-match) and collapse from attention dilution at T=64.
During experimentation, we identify a router initialization trap that causes >70% of training runs to fail: both default zero-bias initialization (p ~ 0.5) and Graves' recommended positive bias (p ~ 0.73) cause tokens to halt after ~2 steps at initialization, settling into a shallow equilibrium (halt ~ 5-7) that the model cannot escape. Inverting the bias to -3 ("deep start," p ~ 0.05) eliminates this failure mode. We confirm through ablation that the trap is inherent to ACT initialization, not an artifact of our architecture choices.
With reliable training established, we show that (1) ACT provides more consistent results than fixed-depth processing (56.9% +/- 0.7% vs 53.4% +/- 9.3% across 3 seeds); (2) ACT with lambda warmup achieves matching accuracy (57.0% +/- 1.1%) using 34% fewer ponder steps; and (3) attention heads specialize into memory readers, constraint propagators, and integrators across recursive depth. Code is available at this https URL. 

---
