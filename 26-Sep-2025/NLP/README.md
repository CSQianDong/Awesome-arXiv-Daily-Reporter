# SciReasoner: Laying the Scientific Reasoning Ground Across Disciplines 

**Authors**: Yizhou Wang, Chen Tang, Han Deng, Jiabei Xiao, Jiaqi Liu, Jianyu Wu, Jun Yao, Pengze Li, Encheng Su, Lintao Wang, Guohang Zhuang, Yuchen Ren, Ben Fei, Ming Hu, Xin Chen, Dongzhan Zhou, Junjun He, Xiangyu Yue, Zhenfei Yin, Jiamin Wu, Qihao Zheng, Yuhao Zhou, Huihui Xu, Chenglong Ma, Yan Lu, Wenlong Zhang, Chunfeng Song, Philip Torr, Shixiang Tang, Xinzhu Ma, Wanli Ouyang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2509.21320)  

**Abstract**: We present a scientific reasoning foundation model that aligns natural language with heterogeneous scientific representations. The model is pretrained on a 206B-token corpus spanning scientific text, pure sequences, and sequence-text pairs, then aligned via SFT on 40M instructions, annealed cold-start bootstrapping to elicit long-form chain-of-thought, and reinforcement learning with task-specific reward shaping, which instills deliberate scientific reasoning. It supports four capability families, covering up to 103 tasks across workflows: (i) faithful translation between text and scientific formats, (ii) text/knowledge extraction, (iii) property prediction, (iv) property classification, (v) unconditional and conditional sequence generation and design. Compared with specialist systems, our approach broadens instruction coverage, improves cross-domain generalization, and enhances fidelity. We detail data curation and training and show that cross-discipline learning strengthens transfer and downstream reliability. The model, instruct tuning datasets and the evaluation code are open-sourced at this https URL and this https URL. 

---
# RLBFF: Binary Flexible Feedback to bridge between Human Feedback & Verifiable Rewards 

**Authors**: Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Ellie Evans, Daniel Egert, Hoo-Chang Shin, Felipe Soares, Yi Dong, Oleksii Kuchaiev  

**Link**: [PDF](https://arxiv.org/pdf/2509.21319)  

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) are the main RL paradigms used in LLM post-training, each offering distinct advantages. However, RLHF struggles with interpretability and reward hacking because it relies on human judgments that usually lack explicit criteria, whereas RLVR is limited in scope by its focus on correctness-based verifiers. We propose Reinforcement Learning with Binary Flexible Feedback (RLBFF), which combines the versatility of human-driven preferences with the precision of rule-based verification, enabling reward models to capture nuanced aspects of response quality beyond mere correctness. RLBFF extracts principles that can be answered in a binary fashion (e.g. accuracy of information: yes, or code readability: no) from natural language feedback. Such principles can then be used to ground Reward Model training as an entailment task (response satisfies or does not satisfy an arbitrary principle). We show that Reward Models trained in this manner can outperform Bradley-Terry models when matched for data and achieve top performance on RM-Bench (86.2%) and JudgeBench (81.4%, #1 on leaderboard as of September 24, 2025). Additionally, users can specify principles of interest at inference time to customize the focus of our reward models, in contrast to Bradley-Terry models. Finally, we present a fully open source recipe (including data) to align Qwen3-32B using RLBFF and our Reward Model, to match or exceed the performance of o3-mini and DeepSeek R1 on general alignment benchmarks of MT-Bench, WildBench, and Arena Hard v2 (at <5% of the inference cost). 

---
# Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs 

**Authors**: Daniel Vennemeyer, Phan Anh Duong, Tiffany Zhan, Tianyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21305)  

**Abstract**: Large language models (LLMs) often exhibit sycophantic behaviors -- such as excessive agreement with or flattery of the user -- but it is unclear whether these behaviors arise from a single mechanism or multiple distinct processes. We decompose sycophancy into sycophantic agreement and sycophantic praise, contrasting both with genuine agreement. Using difference-in-means directions, activation additions, and subspace geometry across multiple models and datasets, we show that: (1) the three behaviors are encoded along distinct linear directions in latent space; (2) each behavior can be independently amplified or suppressed without affecting the others; and (3) their representational structure is consistent across model families and scales. These results suggest that sycophantic behaviors correspond to distinct, independently steerable representations. 

---
# The role of synthetic data in Multilingual, Multi-cultural AI systems: Lessons from Indic Languages 

**Authors**: Pranjal A. Chitale, Varun Gumma, Sanchit Ahuja, Prashant Kodali, Manan Uppadhyay, Deepthi Sudharsan, Sunayana Sitaram  

**Link**: [PDF](https://arxiv.org/pdf/2509.21294)  

**Abstract**: Developing AI systems that operate effectively across languages while remaining culturally grounded is a long-standing challenge, particularly in low-resource settings. Synthetic data provides a promising avenue, yet its effectiveness in multilingual and multicultural contexts remains underexplored. We investigate the creation and impact of synthetic, culturally contextualized datasets for Indian languages through a bottom-up generation strategy that prompts large open-source LLMs (>= 235B parameters) to ground data generation in language-specific Wikipedia content. This approach complements the dominant top-down paradigm of translating synthetic datasets from high-resource languages such as English. We introduce Updesh, a high-quality large-scale synthetic instruction-following dataset comprising 9.5M data points across 13 Indian languages, encompassing diverse reasoning and generative tasks with an emphasis on long-context, multi-turn capabilities, and alignment with Indian cultural contexts. A comprehensive evaluation incorporating both automated metrics and human annotation across 10k assessments indicates that generated data is high quality; though, human evaluation highlights areas for further improvement. Additionally, we perform downstream evaluations by fine-tuning models on our dataset and assessing the performance across 15 diverse multilingual datasets. Models trained on Updesh consistently achieve significant gains on generative tasks and remain competitive on multiple-choice style NLU tasks. Notably, relative improvements are most pronounced in low and medium-resource languages, narrowing their gap with high-resource languages. These findings provide empirical evidence that effective multilingual AI requires multi-faceted data curation and generation strategies that incorporate context-aware, culturally grounded methodologies. 

---
# DisCoCLIP: A Distributional Compositional Tensor Network Encoder for Vision-Language Understanding 

**Authors**: Kin Ian Lo, Hala Hawashin, Mina Abbaszadeh, Tilen Limback-Stokin, Hadi Wazni, Mehrnoosh Sadrzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2509.21287)  

**Abstract**: Recent vision-language models excel at large-scale image-text alignment but often neglect the compositional structure of language, leading to failures on tasks that hinge on word order and predicate-argument structure. We introduce DisCoCLIP, a multimodal encoder that combines a frozen CLIP vision transformer with a novel tensor network text encoder that explicitly encodes syntactic structure. Sentences are parsed with a Combinatory Categorial Grammar parser to yield distributional word tensors whose contractions mirror the sentence's grammatical derivation. To keep the model efficient, high-order tensors are factorized with tensor decompositions, reducing parameter count from tens of millions to under one million. Trained end-to-end with a self-supervised contrastive loss, DisCoCLIP markedly improves sensitivity to verb semantics and word order: it raises CLIP's SVO-Probes verb accuracy from 77.6% to 82.4%, boosts ARO attribution and relation scores by over 9% and 4%, and achieves 93.7% on a newly introduced SVO-Swap benchmark. These results demonstrate that embedding explicit linguistic structure via tensor networks yields interpretable, parameter-efficient representations that substantially improve compositional reasoning in vision-language tasks. 

---
# Bounds of Chain-of-Thought Robustness: Reasoning Steps, Embed Norms, and Beyond 

**Authors**: Dingzirui Wang, Xuanliang Zhang, Keyan Xu, Qingfu Zhu, Wanxiang Che, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2509.21284)  

**Abstract**: Existing research indicates that the output of Chain-of-Thought (CoT) is significantly affected by input perturbations. Although many methods aim to mitigate such impact by optimizing prompts, a theoretical explanation of how these perturbations influence CoT outputs remains an open area of research. This gap limits our in-depth understanding of how input perturbations propagate during the reasoning process and hinders further improvements in prompt optimization methods. Therefore, in this paper, we theoretically analyze the effect of input perturbations on the fluctuation of CoT outputs. We first derive an upper bound for input perturbations under the condition that the output fluctuation is within an acceptable range, based on which we prove that: (i) This upper bound is positively correlated with the number of reasoning steps in the CoT; (ii) Even an infinitely long reasoning process cannot eliminate the impact of input perturbations. We then apply these conclusions to the Linear Self-Attention (LSA) model, which can be viewed as a simplified version of the Transformer. For the LSA model, we prove that the upper bound for input perturbation is negatively correlated with the norms of the input embedding and hidden state vectors. To validate this theoretical analysis, we conduct experiments on three mainstream datasets and four mainstream models. The experimental results align with our theoretical analysis, empirically demonstrating the correctness of our findings. 

---
# LLMTrace: A Corpus for Classification and Fine-Grained Localization of AI-Written Text 

**Authors**: Irina Tolstykh, Aleksandra Tsybina, Sergey Yakubson, Maksim Kuprashevich  

**Link**: [PDF](https://arxiv.org/pdf/2509.21269)  

**Abstract**: The widespread use of human-like text from Large Language Models (LLMs) necessitates the development of robust detection systems. However, progress is limited by a critical lack of suitable training data; existing datasets are often generated with outdated models, are predominantly in English, and fail to address the increasingly common scenario of mixed human-AI authorship. Crucially, while some datasets address mixed authorship, none provide the character-level annotations required for the precise localization of AI-generated segments within a text. To address these gaps, we introduce LLMTrace, a new large-scale, bilingual (English and Russian) corpus for AI-generated text detection. Constructed using a diverse range of modern proprietary and open-source LLMs, our dataset is designed to support two key tasks: traditional full-text binary classification (human vs. AI) and the novel task of AI-generated interval detection, facilitated by character-level annotations. We believe LLMTrace will serve as a vital resource for training and evaluating the next generation of more nuanced and practical AI detection models. The project page is available at \href{this https URL}{iitolstykh/LLMTrace}. 

---
# LLM Output Homogenization is Task Dependent 

**Authors**: Shomik Jain, Jack Lanchantin, Maximilian Nickel, Karen Ullrich, Ashia Wilson, Jamelle Watson-Daniels  

**Link**: [PDF](https://arxiv.org/pdf/2509.21267)  

**Abstract**: A large language model can be less helpful if it exhibits output response homogenization. But whether two responses are considered homogeneous, and whether such homogenization is problematic, both depend on the task category. For instance, in objective math tasks, we often expect no variation in the final answer but anticipate variation in the problem-solving strategy. Whereas, for creative writing tasks, we may expect variation in key narrative components (e.g. plot, genre, setting, etc), beyond the vocabulary or embedding diversity produced by temperature-sampling. Previous work addressing output homogenization often fails to conceptualize diversity in a task-dependent way. We address this gap in the literature directly by making the following contributions. (1) We present a task taxonomy comprised of eight task categories that each have distinct conceptualizations of output homogenization. (2) We introduce task-anchored functional diversity to better evaluate output homogenization. (3) We propose a task-anchored sampling technique that increases functional diversity for task categories where homogenization is undesired, while preserving homogenization where it is desired. (4) We challenge the perceived existence of a diversity-quality trade-off by increasing functional diversity while maintaining response quality. Overall, we demonstrate how task dependence improves the evaluation and mitigation of output homogenization. 

---
# Un-Doubling Diffusion: LLM-guided Disambiguation of Homonym Duplication 

**Authors**: Evgeny Kaskov, Elizaveta Petrova, Petr Surovtsev, Anna Kostikova, Ilya Mistiurin, Alexander Kapitanov, Alexander Nagaev  

**Link**: [PDF](https://arxiv.org/pdf/2509.21262)  

**Abstract**: Homonyms are words with identical spelling but distinct meanings, which pose challenges for many generative models. When a homonym appears in a prompt, diffusion models may generate multiple senses of the word simultaneously, which is known as homonym duplication. This issue is further complicated by an Anglocentric bias, which includes an additional translation step before the text-to-image model pipeline. As a result, even words that are not homonymous in the original language may become homonyms and lose their meaning after translation into English. In this paper, we introduce a method for measuring duplication rates and conduct evaluations of different diffusion models using both automatic evaluation utilizing Vision-Language Models (VLM) and human evaluation. Additionally, we investigate methods to mitigate the homonym duplication problem through prompt expansion, demonstrating that this approach also effectively reduces duplication related to Anglocentric bias. The code for the automatic evaluation pipeline is publicly available. 

---
# Query-Centric Graph Retrieval Augmented Generation 

**Authors**: Yaxiong Wu, Jianyuan Bo, Yongyue Zhang, Sheng Liang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21237)  

**Abstract**: Graph-based retrieval-augmented generation (RAG) enriches large language models (LLMs) with external knowledge for long-context understanding and multi-hop reasoning, but existing methods face a granularity dilemma: fine-grained entity-level graphs incur high token costs and lose context, while coarse document-level graphs fail to capture nuanced relations. We introduce QCG-RAG, a query-centric graph RAG framework that enables query-granular indexing and multi-hop chunk retrieval. Our query-centric approach leverages Doc2Query and Doc2Query{-}{-} to construct query-centric graphs with controllable granularity, improving graph quality and interpretability. A tailored multi-hop retrieval mechanism then selects relevant chunks via the generated queries. Experiments on LiHuaWorld and MultiHop-RAG show that QCG-RAG consistently outperforms prior chunk-based and graph-based RAG methods in question answering accuracy, establishing a new paradigm for multi-hop reasoning. 

---
# SGMem: Sentence Graph Memory for Long-Term Conversational Agents 

**Authors**: Yaxiong Wu, Yongyue Zhang, Sheng Liang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21212)  

**Abstract**: Long-term conversational agents require effective memory management to handle dialogue histories that exceed the context window of large language models (LLMs). Existing methods based on fact extraction or summarization reduce redundancy but struggle to organize and retrieve relevant information across different granularities of dialogue and generated memory. We introduce SGMem (Sentence Graph Memory), which represents dialogue as sentence-level graphs within chunked units, capturing associations across turn-, round-, and session-level contexts. By combining retrieved raw dialogue with generated memory such as summaries, facts and insights, SGMem supplies LLMs with coherent and relevant context for response generation. Experiments on LongMemEval and LoCoMo show that SGMem consistently improves accuracy and outperforms strong baselines in long-term conversational question answering. 

---
# CLaw: Benchmarking Chinese Legal Knowledge in Large Language Models - A Fine-grained Corpus and Reasoning Analysis 

**Authors**: Xinzhe Xu, Liang Zhao, Hongshen Xu, Chen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21208)  

**Abstract**: Large Language Models (LLMs) are increasingly tasked with analyzing legal texts and citing relevant statutes, yet their reliability is often compromised by general pre-training that ingests legal texts without specialized focus, obscuring the true depth of their legal knowledge. This paper introduces CLaw, a novel benchmark specifically engineered to meticulously evaluate LLMs on Chinese legal knowledge and its application in reasoning. CLaw comprises two key components: (1) a comprehensive, fine-grained corpus of all 306 Chinese national statutes, segmented to the subparagraph level and incorporating precise historical revision timesteps for rigorous recall evaluation (64,849 entries), and (2) a challenging set of 254 case-based reasoning instances derived from China Supreme Court curated materials to assess the practical application of legal knowledge. Our empirical evaluation reveals that most contemporary LLMs significantly struggle to faithfully reproduce legal provisions. As accurate retrieval and citation of legal provisions form the basis of legal reasoning, this deficiency critically undermines the reliability of their responses. We contend that achieving trustworthy legal reasoning in LLMs requires a robust synergy of accurate knowledge retrieval--potentially enhanced through supervised fine-tuning (SFT) or retrieval-augmented generation (RAG)--and strong general reasoning capabilities. This work provides an essential benchmark and critical insights for advancing domain-specific LLM reasoning, particularly within the complex legal sphere. 

---
# Eigen-1: Adaptive Multi-Agent Refinement with Monitor-Based RAG for Scientific Reasoning 

**Authors**: Xiangru Tang, Wanghan Xu, Yujie Wang, Zijie Guo, Daniel Shao, Jiapeng Chen, Cixuan Zhang, Ziyi Wang, Lixin Zhang, Guancheng Wan, Wenlong Zhang, Lei Bai, Zhenfei Yin, Philip Torr, Hanrui Wang, Di Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.21193)  

**Abstract**: Large language models (LLMs) have recently shown strong progress on scientific reasoning, yet two major bottlenecks remain. First, explicit retrieval fragments reasoning, imposing a hidden "tool tax" of extra tokens and steps. Second, multi-agent pipelines often dilute strong solutions by averaging across all candidates. We address these challenges with a unified framework that combines implicit retrieval and structured collaboration. At its foundation, a Monitor-based retrieval module operates at the token level, integrating external knowledge with minimal disruption to reasoning. On top of this substrate, Hierarchical Solution Refinement (HSR) iteratively designates each candidate as an anchor to be repaired by its peers, while Quality-Aware Iterative Reasoning (QAIR) adapts refinement to solution quality. On Humanity's Last Exam (HLE) Bio/Chem Gold, our framework achieves 48.3\% accuracy -- the highest reported to date, surpassing the strongest agent baseline by 13.4 points and leading frontier LLMs by up to 18.1 points, while simultaneously reducing token usage by 53.5\% and agent steps by 43.7\%. Results on SuperGPQA and TRQA confirm robustness across domains. Error analysis shows that reasoning failures and knowledge gaps co-occur in over 85\% of cases, while diversity analysis reveals a clear dichotomy: retrieval tasks benefit from solution variety, whereas reasoning tasks favor consensus. Together, these findings demonstrate how implicit augmentation and structured refinement overcome the inefficiencies of explicit tool use and uniform aggregation. Code is available at: this https URL. 

---
# GEP: A GCG-Based method for extracting personally identifiable information from chatbots built on small language models 

**Authors**: Jieli Zhu, Vi Ngoc-Nha Tran  

**Link**: [PDF](https://arxiv.org/pdf/2509.21192)  

**Abstract**: Small language models (SLMs) become unprecedentedly appealing due to their approximately equivalent performance compared to large language models (LLMs) in certain fields with less energy and time consumption during training and inference. However, the personally identifiable information (PII) leakage of SLMs for downstream tasks has yet to be explored. In this study, we investigate the PII leakage of the chatbot based on SLM. We first finetune a new chatbot, i.e., ChatBioGPT based on the backbone of BioGPT using medical datasets Alpaca and HealthCareMagic. It shows a matchable performance in BERTscore compared with previous studies of ChatDoctor and ChatGPT. Based on this model, we prove that the previous template-based PII attacking methods cannot effectively extract the PII in the dataset for leakage detection under the SLM condition. We then propose GEP, which is a greedy coordinate gradient-based (GCG) method specifically designed for PII extraction. We conduct experimental studies of GEP and the results show an increment of up to 60$\times$ more leakage compared with the previous template-based methods. We further expand the capability of GEP in the case of a more complicated and realistic situation by conducting free-style insertion where the inserted PII in the dataset is in the form of various syntactic expressions instead of fixed templates, and GEP is still able to reveal a PII leakage rate of up to 4.53%. 

---
# Who's Laughing Now? An Overview of Computational Humour Generation and Explanation 

**Authors**: Tyler Loakman, William Thorne, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.21175)  

**Abstract**: The creation and perception of humour is a fundamental human trait, positioning its computational understanding as one of the most challenging tasks in natural language processing (NLP). As an abstract, creative, and frequently context-dependent construct, humour requires extensive reasoning to understand and create, making it a pertinent task for assessing the common-sense knowledge and reasoning abilities of modern large language models (LLMs). In this work, we survey the landscape of computational humour as it pertains to the generative tasks of creation and explanation. We observe that, despite the task of understanding humour bearing all the hallmarks of a foundational NLP task, work on generating and explaining humour beyond puns remains sparse, while state-of-the-art models continue to fall short of human capabilities. We bookend our literature survey by motivating the importance of computational humour processing as a subdiscipline of NLP and presenting an extensive discussion of future directions for research in the area that takes into account the subjective and ethically ambiguous nature of humour. 

---
# Learning the Wrong Lessons: Syntactic-Domain Spurious Correlations in Language Models 

**Authors**: Chantal Shaib, Vinith M. Suriyakumar, Levent Sagun, Byron C. Wallace, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21155)  

**Abstract**: For an LLM to correctly respond to an instruction it must understand both the semantics and the domain (i.e., subject area) of a given task-instruction pair. However, syntax can also convey implicit information Recent work shows that syntactic templates--frequent sequences of Part-of-Speech (PoS) tags--are prevalent in training data and often appear in model outputs. In this work we characterize syntactic templates, domain, and semantics in task-instruction pairs. We identify cases of spurious correlations between syntax and domain, where models learn to associate a domain with syntax during training; this can sometimes override prompt semantics. Using a synthetic training dataset, we find that the syntactic-domain correlation can lower performance (mean 0.51 +/- 0.06) on entity knowledge tasks in OLMo-2 models (1B-13B). We introduce an evaluation framework to detect this phenomenon in trained models, and show that it occurs on a subset of the FlanV2 dataset in open (OLMo-2-7B; Llama-4-Maverick), and closed (GPT-4o) models. Finally, we present a case study on the implications for safety finetuning, showing that unintended syntactic-domain correlations can be used to bypass refusals in OLMo-2-7B Instruct and GPT-4o. Our findings highlight two needs: (1) to explicitly test for syntactic-domain correlations, and (2) to ensure syntactic diversity in training data, specifically within domains, to prevent such spurious correlations. 

---
# Retrieval over Classification: Integrating Relation Semantics for Multimodal Relation Extraction 

**Authors**: Lei Hei, Tingjing Liao, Yingxin Pei, Yiyang Qi, Jiaqi Wang, Ruiting Li, Feiliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.21151)  

**Abstract**: Relation extraction (RE) aims to identify semantic relations between entities in unstructured text. Although recent work extends traditional RE to multimodal scenarios, most approaches still adopt classification-based paradigms with fused multimodal features, representing relations as discrete labels. This paradigm has two significant limitations: (1) it overlooks structural constraints like entity types and positional cues, and (2) it lacks semantic expressiveness for fine-grained relation understanding. We propose \underline{R}etrieval \underline{O}ver \underline{C}lassification (ROC), a novel framework that reformulates multimodal RE as a retrieval task driven by relation semantics. ROC integrates entity type and positional information through a multimodal encoder, expands relation labels into natural language descriptions using a large language model, and aligns entity-relation pairs via semantic similarity-based contrastive learning. Experiments show that our method achieves state-of-the-art performance on the benchmark datasets MNRE and MORE and exhibits stronger robustness and interpretability. 

---
# AutoIntent: AutoML for Text Classification 

**Authors**: Ilya Alekseev, Roman Solomatin, Darina Rustamova, Denis Kuznetsov  

**Link**: [PDF](https://arxiv.org/pdf/2509.21138)  

**Abstract**: AutoIntent is an automated machine learning tool for text classification tasks. Unlike existing solutions, AutoIntent offers end-to-end automation with embedding model selection, classifier optimization, and decision threshold tuning, all within a modular, sklearn-like interface. The framework is designed to support multi-label classification and out-of-scope detection. AutoIntent demonstrates superior performance compared to existing AutoML tools on standard intent classification datasets and enables users to balance effectiveness and resource consumption. 

---
# Acoustic-based Gender Differentiation in Speech-aware Language Models 

**Authors**: Junhyuk Choi, Jihwan Seol, Nayeon Kim, Chanhee Cho, EunBin Cho, Bugeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.21125)  

**Abstract**: Speech-aware Language Models (SpeechLMs) have fundamentally transformed human-AI interaction by enabling voice-based communication, yet they may exhibit acoustic-based gender differentiation where identical questions lead to different responses based on the speaker's gender. This paper propose a new dataset that enables systematic analysis of this phenomenon, containing 9,208 speech samples across three categories: Gender-Independent, Gender-Stereotypical, and Gender-Dependent. We further evaluated LLaMA-Omni series and discovered a paradoxical pattern; while overall responses seems identical regardless of gender, the pattern is far from unbiased responses. Specifically, in Gender-Stereotypical questions, all models consistently exhibited male-oriented responses; meanwhile, in Gender-Dependent questions where gender differentiation would be contextually appropriate, models exhibited responses independent to gender instead. We also confirm that this pattern does not result from neutral options nor perceived gender of a voice. When we allow neutral response, models tends to respond neutrally also in Gender-Dependent questions. The paradoxical pattern yet retains when we applied gender neutralization methods on speech. Through comparison between SpeechLMs with corresponding backbone LLMs, we confirmed that these paradoxical patterns primarily stem from Whisper speech encoders, which generates male-oriented acoustic tokens. These findings reveal that current SpeechLMs may not successfully remove gender biases though they prioritized general fairness principles over contextual appropriateness, highlighting the need for more sophisticated techniques to utilize gender information properly in speech technology. 

---
# VoiceBBQ: Investigating Effect of Content and Acoustics in Social Bias of Spoken Language Model 

**Authors**: Junhyuk Choi, Ro-hoon Oh, Jihwan Seol, Bugeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.21108)  

**Abstract**: We introduce VoiceBBQ, a spoken extension of the BBQ (Bias Benchmark for Question Answering) - a dataset that measures social bias by presenting ambiguous or disambiguated contexts followed by questions that may elicit stereotypical responses. Due to the nature of speech, social bias in Spoken Language Models (SLMs) can emerge from two distinct sources: 1) content aspect and 2) acoustic aspect. The dataset converts every BBQ context into controlled voice conditions, enabling per-axis accuracy, bias, and consistency scores that remain comparable to the original text benchmark. Using VoiceBBQ, we evaluate two SLMs - LLaMA-Omni and Qwen2-Audio - and observe architectural contrasts: LLaMA-Omni resists acoustic bias while amplifying gender and accent bias, whereas Qwen2-Audio substantially dampens these cues while preserving content fidelity. VoiceBBQ thus provides a compact, drop-in testbed for jointly diagnosing content and acoustic bias across spoken language models. 

---
# BESPOKE: Benchmark for Search-Augmented Large Language Model Personalization via Diagnostic Feedback 

**Authors**: Hyunseo Kim, Sangam Lee, Kwangwook Seo, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.21106)  

**Abstract**: Search-augmented large language models (LLMs) have advanced information-seeking tasks by integrating retrieval into generation, reducing users' cognitive burden compared to traditional search systems. Yet they remain insufficient for fully addressing diverse user needs, which requires recognizing how the same query can reflect different intents across users and delivering information in preferred forms. While recent systems such as ChatGPT and Gemini attempt personalization by leveraging user histories, systematic evaluation of such personalization is under-explored. To address this gap, we propose BESPOKE, the realistic benchmark for evaluating personalization in search-augmented LLMs. BESPOKE is designed to be both realistic, by collecting authentic chat and search histories directly from humans, and diagnostic, by pairing responses with fine-grained preference scores and feedback. The benchmark is constructed through long-term, deeply engaged human annotation, where human annotators contributed their own histories, authored queries with detailed information needs, and evaluated responses with scores and diagnostic feedback. Leveraging BESPOKE, we conduct systematic analyses that reveal key requirements for effective personalization in information-seeking tasks, providing a foundation for fine-grained evaluation of personalized search-augmented LLMs. Our code and data are available at this https URL. 

---
# PerHalluEval: Persian Hallucination Evaluation Benchmark for Large Language Models 

**Authors**: Mohammad Hosseini, Kimia Hosseini, Shayan Bali, Zahra Zanjani, Saeedeh Momtazi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21104)  

**Abstract**: Hallucination is a persistent issue affecting all large language Models (LLMs), particularly within low-resource languages such as Persian. PerHalluEval (Persian Hallucination Evaluation) is the first dynamic hallucination evaluation benchmark tailored for the Persian language. Our benchmark leverages a three-stage LLM-driven pipeline, augmented with human validation, to generate plausible answers and summaries regarding QA and summarization tasks, focusing on detecting extrinsic and intrinsic hallucinations. Moreover, we used the log probabilities of generated tokens to select the most believable hallucinated instances. In addition, we engaged human annotators to highlight Persian-specific contexts in the QA dataset in order to evaluate LLMs' performance on content specifically related to Persian culture. Our evaluation of 12 LLMs, including open- and closed-source models using PerHalluEval, revealed that the models generally struggle in detecting hallucinated Persian text. We showed that providing external knowledge, i.e., the original document for the summarization task, could mitigate hallucination partially. Furthermore, there was no significant difference in terms of hallucination when comparing LLMs specifically trained for Persian with others. 

---
# Which Cultural Lens Do Models Adopt? On Cultural Positioning Bias and Agentic Mitigation in LLMs 

**Authors**: Yixin Wan, Xingrun Chen, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21080)  

**Abstract**: Large language models (LLMs) have unlocked a wide range of downstream generative applications. However, we found that they also risk perpetuating subtle fairness issues tied to culture, positioning their generations from the perspectives of the mainstream US culture while demonstrating salient externality towards non-mainstream ones. In this work, we identify and systematically investigate this novel culture positioning bias, in which an LLM's default generative stance aligns with a mainstream view and treats other cultures as outsiders. We propose the CultureLens benchmark with 4000 generation prompts and 3 evaluation metrics for quantifying this bias through the lens of a culturally situated interview script generation task, in which an LLM is positioned as an onsite reporter interviewing local people across 10 diverse cultures. Empirical evaluation on 5 state-of-the-art LLMs reveals a stark pattern: while models adopt insider tones in over 88 percent of US-contexted scripts on average, they disproportionately adopt mainly outsider stances for less dominant cultures. To resolve these biases, we propose 2 inference-time mitigation methods: a baseline prompt-based Fairness Intervention Pillars (FIP) method, and a structured Mitigation via Fairness Agents (MFA) framework consisting of 2 pipelines: (1) MFA-SA (Single-Agent) introduces a self-reflection and rewriting loop based on fairness guidelines. (2) MFA-MA (Multi-Agent) structures the process into a hierarchy of specialized agents: a Planner Agent(initial script generation), a Critique Agent (evaluates initial script against fairness pillars), and a Refinement Agent (incorporates feedback to produce a polished, unbiased script). Empirical results showcase the effectiveness of agent-based methods as a promising direction for mitigating biases in generative LLMs. 

---
# SoM-1K: A Thousand-Problem Benchmark Dataset for Strength of Materials 

**Authors**: Qixin Wan, Zilong Wang, Jingwen Zhou, Wanting Wang, Ziheng Geng, Jiachen Liu, Ran Cao, Minghui Cheng, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.21079)  

**Abstract**: Foundation models have shown remarkable capabilities in various domains, but their performance on complex, multimodal engineering problems remains largely unexplored. We introduce SoM-1K, the first large-scale multimodal benchmark dataset dedicated to evaluating foundation models on problems in the strength of materials (SoM). The dataset, which contains 1,065 annotated SoM problems, mirrors real-world engineering tasks by including both textual problem statements and schematic diagrams. Due to the limited capabilities of current foundation models in understanding complicated visual information, we propose a novel prompting strategy called Descriptions of Images (DoI), which provides rigorous expert-generated text descriptions of the visual diagrams as the context. We evaluate eight representative foundation models, including both large language models (LLMs) and vision language models (VLMs). Our results show that current foundation models struggle significantly with these engineering problems, with the best-performing model achieving only 56.6% accuracy. Interestingly, we found that LLMs, when provided with DoI, often outperform VLMs provided with visual diagrams. A detailed error analysis reveals that DoI plays a crucial role in mitigating visual misinterpretation errors, suggesting that accurate text-based descriptions can be more effective than direct image input for current foundation models. This work establishes a rigorous benchmark for engineering AI and highlights a critical need for developing more robust multimodal reasoning capabilities in foundation models, particularly in scientific and engineering contexts. 

---
# When Instructions Multiply: Measuring and Estimating LLM Capabilities of Multiple Instructions Following 

**Authors**: Keno Harada, Yudai Yamazaki, Masachika Taniguchi, Edison Marrese-Taylor, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2509.21051)  

**Abstract**: As large language models (LLMs) are increasingly applied to real-world scenarios, it becomes crucial to understand their ability to follow multiple instructions simultaneously. To systematically evaluate these capabilities, we introduce two specialized benchmarks for fundamental domains where multiple instructions following is important: Many Instruction-Following Eval (ManyIFEval) for text generation with up to ten instructions, and Style-aware Mostly Basic Programming Problems (StyleMBPP) for code generation with up to six instructions. Our experiments with the created benchmarks across ten LLMs reveal that performance consistently degrades as the number of instructions increases. Furthermore, given the fact that evaluating all the possible combinations of multiple instructions is computationally impractical in actual use cases, we developed three types of regression models that can estimate performance on both unseen instruction combinations and different numbers of instructions which are not used during training. We demonstrate that a logistic regression model using instruction count as an explanatory variable can predict performance of following multiple instructions with approximately 10% error, even for unseen instruction combinations. We show that relatively modest sample sizes (500 for ManyIFEval and 300 for StyleMBPP) are sufficient for performance estimation, enabling efficient evaluation of LLMs under various instruction combinations. 

---
# Behind RoPE: How Does Causal Mask Encode Positional Information? 

**Authors**: Junu Kim, Xiao Liu, Zhenghao Lin, Lei Ji, Yeyun Gong, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21042)  

**Abstract**: While explicit positional encodings such as RoPE are a primary source of positional information in Transformer decoders, the causal mask also provides positional information. In this work, we prove that the causal mask can induce position-dependent patterns in attention scores, even without parameters or causal dependency in the input. Our theoretical analysis indicates that the induced attention pattern tends to favor nearby query-key pairs, mirroring the behavior of common positional encodings. Empirical analysis confirms that trained models exhibit the same behavior, with learned parameters further amplifying these patterns. Notably, we found that the interaction of causal mask and RoPE distorts RoPE's relative attention score patterns into non-relative ones. We consistently observed this effect in modern large language models, suggesting the importance of considering the causal mask as a source of positional information alongside explicit positional encodings. 

---
# Generative AI for FFRDCs 

**Authors**: Arun S. Maiya  

**Link**: [PDF](https://arxiv.org/pdf/2509.21040)  

**Abstract**: Federally funded research and development centers (FFRDCs) face text-heavy workloads, from policy documents to scientific and engineering papers, that are slow to analyze manually. We show how large language models can accelerate summarization, classification, extraction, and sense-making with only a few input-output examples. To enable use in sensitive government contexts, we apply OnPrem$.$LLM, an open-source framework for secure and flexible application of generative AI. Case studies on defense policy documents and scientific corpora, including the National Defense Authorization Act (NDAA) and National Science Foundation (NSF) Awards, demonstrate how this approach enhances oversight and strategic analysis while maintaining auditability and data sovereignty. 

---
# Analysis of instruction-based LLMs' capabilities to score and judge text-input problems in an academic setting 

**Authors**: Valeria Ramirez-Garcia, David de-Fitero-Dominguez, Antonio Garcia-Cabot, Eva Garcia-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2509.20982)  

**Abstract**: Large language models (LLMs) can act as evaluators, a role studied by methods like LLM-as-a-Judge and fine-tuned judging LLMs. In the field of education, LLMs have been studied as assistant tools for students and teachers. Our research investigates LLM-driven automatic evaluation systems for academic Text-Input Problems using rubrics. We propose five evaluation systems that have been tested on a custom dataset of 110 answers about computer science from higher education students with three models: JudgeLM, Llama-3.1-8B and DeepSeek-R1-Distill-Llama-8B. The evaluation systems include: The JudgeLM evaluation, which uses the model's single answer prompt to obtain a score; Reference Aided Evaluation, which uses a correct answer as a guide aside from the original context of the question; No Reference Evaluation, which ommits the reference answer; Additive Evaluation, which uses atomic criteria; and Adaptive Evaluation, which is an evaluation done with generated criteria fitted to each question. All evaluation methods have been compared with the results of a human evaluator. Results show that the best method to automatically evaluate and score Text-Input Problems using LLMs is Reference Aided Evaluation. With the lowest median absolute deviation (0.945) and the lowest root mean square deviation (1.214) when compared to human evaluation, Reference Aided Evaluation offers fair scoring as well as insightful and complete evaluations. Other methods such as Additive and Adaptive Evaluation fail to provide good results in concise answers, No Reference Evaluation lacks information needed to correctly assess questions and JudgeLM Evaluations have not provided good results due to the model's limitations. As a result, we conclude that Artificial Intelligence-driven automatic evaluation systems, aided with proper methodologies, show potential to work as complementary tools to other academic resources. 

---
# Tool Calling for Arabic LLMs: Data Strategies and Instruction Tuning 

**Authors**: Asim Ersoy, Enes Altinisik, Husrev Taha Sencar, Kareem Darwish  

**Link**: [PDF](https://arxiv.org/pdf/2509.20957)  

**Abstract**: Tool calling is a critical capability that allows Large Language Models (LLMs) to interact with external systems, significantly expanding their utility. However, research and resources for tool calling are predominantly English-centric, leaving a gap in our understanding of how to enable this functionality for other languages, such as Arabic. This paper investigates three key research questions: (1) the necessity of in-language (Arabic) tool-calling data versus relying on cross-lingual transfer, (2) the effect of general-purpose instruction tuning on tool-calling performance, and (3) the value of fine-tuning on specific, high-priority tools. To address these questions, we conduct extensive experiments using base and post-trained variants of an open-weight Arabic LLM. To enable this study, we bridge the resource gap by translating and adapting two open-source tool-calling datasets into Arabic. Our findings provide crucial insights into the optimal strategies for developing robust tool-augmented agents for Arabic. 

---
# Cross-Linguistic Analysis of Memory Load in Sentence Comprehension: Linear Distance and Structural Density 

**Authors**: Krishna Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2509.20916)  

**Abstract**: This study examines whether sentence-level memory load in comprehension is better explained by linear proximity between syntactically related words or by the structural density of the intervening material. Building on locality-based accounts and cross-linguistic evidence for dependency length minimization, the work advances Intervener Complexity-the number of intervening heads between a head and its dependent-as a structurally grounded lens that refines linear distance measures. Using harmonized dependency treebanks and a mixed-effects framework across multiple languages, the analysis jointly evaluates sentence length, dependency length, and Intervener Complexity as predictors of the Memory-load measure. Studies in Psycholinguistics have reported the contributions of feature interference and misbinding to memory load during processing. For this study, I operationalized sentence-level memory load as the linear sum of feature misbinding and feature interference for tractability; current evidence does not establish that their cognitive contributions combine additively. All three factors are positively associated with memory load, with sentence length exerting the broadest influence and Intervener Complexity offering explanatory power beyond linear distance. Conceptually, the findings reconcile linear and hierarchical perspectives on locality by treating dependency length as an important surface signature while identifying intervening heads as a more proximate indicator of integration and maintenance demands. Methodologically, the study illustrates how UD-based graph measures and cross-linguistic mixed-effects modelling can disentangle linear and structural contributions to processing efficiency, providing a principled path for evaluating competing theories of memory load in sentence comprehension. 

---
# MemLens: Uncovering Memorization in LLMs with Activation Trajectories 

**Authors**: Zirui He, Haiyan Zhao, Ali Payani, Mengnan du  

**Link**: [PDF](https://arxiv.org/pdf/2509.20909)  

**Abstract**: Large language models (LLMs) are commonly evaluated on challenging benchmarks such as AIME and Math500, which are susceptible to contamination and risk of being memorized. Existing detection methods, which primarily rely on surface-level lexical overlap and perplexity, demonstrate low generalization and degrade significantly when encountering implicitly contaminated data. In this paper, we propose MemLens (An Activation Lens for Memorization Detection) to detect memorization by analyzing the probability trajectories of numeric tokens during generation. Our method reveals that contaminated samples exhibit ``shortcut'' behaviors, locking onto an answer with high confidence in the model's early layers, whereas clean samples show more gradual evidence accumulation across the model's full depth. We observe that contaminated and clean samples exhibit distinct and well-separated reasoning trajectories. To further validate this, we inject carefully designed samples into the model through LoRA fine-tuning and observe the same trajectory patterns as in naturally contaminated data. These results provide strong evidence that MemLens captures genuine signals of memorization rather than spurious correlations. 

---
# Learning to Summarize by Learning to Quiz: Adversarial Agentic Collaboration for Long Document Summarization 

**Authors**: Weixuan Wang, Minghao Wu, Barry Haddow, Alexandra Birch  

**Link**: [PDF](https://arxiv.org/pdf/2509.20900)  

**Abstract**: Long document summarization remains a significant challenge for current large language models (LLMs), as existing approaches commonly struggle with information loss, factual inconsistencies, and coherence issues when processing excessively long documents. We propose SummQ, a novel adversarial multi-agent framework that addresses these limitations through collaborative intelligence between specialized agents operating in two complementary domains: summarization and quizzing. Our approach employs summary generators and reviewers that work collaboratively to create and evaluate comprehensive summaries, while quiz generators and reviewers create comprehension questions that serve as continuous quality checks for the summarization process. This adversarial dynamic, enhanced by an examinee agent that validates whether the generated summary contains the information needed to answer the quiz questions, enables iterative refinement through multifaceted feedback mechanisms. We evaluate SummQ on three widely used long document summarization benchmarks. Experimental results demonstrate that our framework significantly outperforms existing state-of-the-art methods across ROUGE and BERTScore metrics, as well as in LLM-as-a-Judge and human evaluations. Our comprehensive analyses reveal the effectiveness of the multi-agent collaboration dynamics, the influence of different agent configurations, and the impact of the quizzing mechanism. This work establishes a new approach for long document summarization that uses adversarial agentic collaboration to improve summarization quality. 

---
# Single Answer is Not Enough: On Generating Ranked Lists with Medical Reasoning Models 

**Authors**: Pittawat Taveekitworachai, Natpatchara Pongjirapat, Krittaphas Chaisutyakorn, Piyalitt Ittichaiwong, Tossaporn Saengja, Kunat Pipatanakul  

**Link**: [PDF](https://arxiv.org/pdf/2509.20866)  

**Abstract**: This paper presents a systematic study on enabling medical reasoning models (MRMs) to generate ranked lists of answers for open-ended questions. Clinical decision-making rarely relies on a single answer but instead considers multiple options, reducing the risks of narrow perspectives. Yet current MRMs are typically trained to produce only one answer, even in open-ended settings. We propose an alternative format: ranked lists and investigate two approaches: prompting and fine-tuning. While prompting is a cost-effective way to steer an MRM's response, not all MRMs generalize well across different answer formats: choice, short text, and list answers. Based on our prompting findings, we train and evaluate MRMs using supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT). SFT teaches a model to imitate annotated responses, and RFT incentivizes exploration through the responses that maximize a reward. We propose new reward functions targeted at ranked-list answer formats, and conduct ablation studies for RFT. Our results show that while some SFT models generalize to certain answer formats, models trained with RFT are more robust across multiple formats. We also present a case study on a modified MedQA with multiple valid answers, finding that although MRMs might fail to select the benchmark's preferred ground truth, they can recognize valid answers. To the best of our knowledge, this is the first systematic investigation of approaches for enabling MRMs to generate answers as ranked lists. We hope this work provides a first step toward developing alternative answer formats that are beneficial beyond single answers in medical domains. 

---
# WeFT: Weighted Entropy-driven Fine-Tuning for dLLMs 

**Authors**: Guowei Xu, Wenxin Xu, Jiawang Zhao, Kaisheng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.20863)  

**Abstract**: Diffusion models have recently shown strong potential in language modeling, offering faster generation compared to traditional autoregressive approaches. However, applying supervised fine-tuning (SFT) to diffusion models remains challenging, as they lack precise probability estimates at each denoising step. While the diffusion mechanism enables the model to reason over entire sequences, it also makes the generation process less predictable and often inconsistent. This highlights the importance of controlling key tokens that guide the direction of generation. To address this issue, we propose WeFT, a weighted SFT method for diffusion language models, where tokens are assigned different weights based on their entropy. Derived from diffusion theory, WeFT delivers substantial gains: training on s1K, s1K-1.1, and 3k samples from open-r1, it achieves relative improvements of 39%, 64%, and 83% over standard SFT on four widely used reasoning benchmarks (Sudoku, Countdown, GSM8K, and MATH-500). The code and models will be made publicly available. 

---
# Concise and Sufficient Sub-Sentence Citations for Retrieval-Augmented Generation 

**Authors**: Guo Chen, Qiuyuan Li, Qiuxian Li, Hongliang Dai, Xiang Chen, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.20859)  

**Abstract**: In retrieval-augmented generation (RAG) question answering systems, generating citations for large language model (LLM) outputs enhances verifiability and helps users identify potential hallucinations. However, we observe two problems in the citations produced by existing attribution methods. First, the citations are typically provided at the sentence or even paragraph level. Long sentences or paragraphs may include a substantial amount of irrelevant content. Second, sentence-level citations may omit information that is essential for verifying the output, forcing users to read the surrounding context. In this paper, we propose generating sub-sentence citations that are both concise and sufficient, thereby reducing the effort required by users to confirm the correctness of the generated output. To this end, we first develop annotation guidelines for such citations and construct a corresponding dataset. Then, we propose an attribution framework for generating citations that adhere to our standards. This framework leverages LLMs to automatically generate fine-tuning data for our task and employs a credit model to filter out low-quality examples. Our experiments on the constructed dataset demonstrate that the propose approach can generate high-quality and more readable citations. 

---
# Zero-Shot Privacy-Aware Text Rewriting via Iterative Tree Search 

**Authors**: Shuo Huang, Xingliang Yuan, Gholamreza Haffari, Lizhen Qu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20838)  

**Abstract**: The increasing adoption of large language models (LLMs) in cloud-based services has raised significant privacy concerns, as user inputs may inadvertently expose sensitive information. Existing text anonymization and de-identification techniques, such as rule-based redaction and scrubbing, often struggle to balance privacy preservation with text naturalness and utility. In this work, we propose a zero-shot, tree-search-based iterative sentence rewriting algorithm that systematically obfuscates or deletes private information while preserving coherence, relevance, and naturalness. Our method incrementally rewrites privacy-sensitive segments through a structured search guided by a reward model, enabling dynamic exploration of the rewriting space. Experiments on privacy-sensitive datasets show that our approach significantly outperforms existing baselines, achieving a superior balance between privacy protection and utility preservation. 

---
# Distilling Many-Shot In-Context Learning into a Cheat Sheet 

**Authors**: Ukyo Honda, Soichiro Murakami, Peinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20820)  

**Abstract**: Recent advances in large language models (LLMs) enable effective in-context learning (ICL) with many-shot examples, but at the cost of high computational demand due to longer input tokens. To address this, we propose cheat-sheet ICL, which distills the information from many-shot ICL into a concise textual summary (cheat sheet) used as the context at inference time. Experiments on challenging reasoning tasks show that cheat-sheet ICL achieves comparable or better performance than many-shot ICL with far fewer tokens, and matches retrieval-based ICL without requiring test-time retrieval. These findings demonstrate that cheat-sheet ICL is a practical alternative for leveraging LLMs in downstream tasks. 

---
# Leveraging What's Overfixed: Post-Correction via LLM Grammatical Error Overcorrection 

**Authors**: Taehee Park, Heejin Do, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.20811)  

**Abstract**: Robust supervised fine-tuned small Language Models (sLMs) often show high reliability but tend to undercorrect. They achieve high precision at the cost of low recall. Conversely, Large Language Models (LLMs) often show the opposite tendency, making excessive overcorrection, leading to low precision. To effectively harness the strengths of LLMs to address the recall challenges in sLMs, we propose Post-Correction via Overcorrection (PoCO), a novel approach that strategically balances recall and precision. PoCO first intentionally triggers overcorrection via LLM to maximize recall by allowing comprehensive revisions, then applies a targeted post-correction step via fine-tuning smaller models to identify and refine erroneous outputs. We aim to harmonize both aspects by leveraging the generative power of LLMs while preserving the reliability of smaller supervised models. Our extensive experiments demonstrate that PoCO effectively balances GEC performance by increasing recall with competitive precision, ultimately improving the overall quality of grammatical error correction. 

---
# Enrich-on-Graph: Query-Graph Alignment for Complex Reasoning with LLM Enriching 

**Authors**: Songze Li, Zhiqiang Liu, Zhengke Gui, Huajun Chen, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20810)  

**Abstract**: Large Language Models (LLMs) exhibit strong reasoning capabilities in complex tasks. However, they still struggle with hallucinations and factual errors in knowledge-intensive scenarios like knowledge graph question answering (KGQA). We attribute this to the semantic gap between structured knowledge graphs (KGs) and unstructured queries, caused by inherent differences in their focuses and structures. Existing methods usually employ resource-intensive, non-scalable workflows reasoning on vanilla KGs, but overlook this gap. To address this challenge, we propose a flexible framework, Enrich-on-Graph (EoG), which leverages LLMs' prior knowledge to enrich KGs, bridge the semantic gap between graphs and queries. EoG enables efficient evidence extraction from KGs for precise and robust reasoning, while ensuring low computational costs, scalability, and adaptability across different methods. Furthermore, we propose three graph quality evaluation metrics to analyze query-graph alignment in KGQA task, supported by theoretical validation of our optimization objectives. Extensive experiments on two KGQA benchmark datasets indicate that EoG can effectively generate high-quality KGs and achieve the state-of-the-art performance. Our code and data are available at this https URL. 

---
# Few-Shot and Training-Free Review Generation via Conversational Prompting 

**Authors**: Genki Kusano  

**Link**: [PDF](https://arxiv.org/pdf/2509.20805)  

**Abstract**: Personalized review generation helps businesses understand user preferences, yet most existing approaches assume extensive review histories of the target user or require additional model training. Real-world applications often face few-shot and training-free situations, where only a few user reviews are available and fine-tuning is infeasible. It is well known that large language models (LLMs) can address such low-resource settings, but their effectiveness depends on prompt engineering. In this paper, we propose Conversational Prompting, a lightweight method that reformulates user reviews as multi-turn conversations. Its simple variant, Simple Conversational Prompting (SCP), relies solely on the user's own reviews, while the contrastive variant, Contrastive Conversational Prompting (CCP), inserts reviews from other users or LLMs as incorrect replies and then asks the model to correct them, encouraging the model to produce text in the user's style. Experiments on eight product domains and five LLMs showed that the conventional non-conversational prompt often produced reviews similar to those written by random users, based on text-based metrics such as ROUGE-L and BERTScore, and application-oriented tasks like user identity matching and sentiment analysis. In contrast, both SCP and CCP produced reviews much closer to those of the target user, even when each user had only two reviews. CCP brings further improvements when high-quality negative examples are available, whereas SCP remains competitive when such data cannot be collected. These results suggest that conversational prompting offers a practical solution for review generation under few-shot and training-free constraints. 

---
# Towards Atoms of Large Language Models 

**Authors**: Chenhui Hu, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.20784)  

**Abstract**: The fundamental units of internal representations in large language models (LLMs) remain undefined, limiting further understanding of their mechanisms. Neurons or features are often regarded as such units, yet neurons suffer from polysemy, while features face concerns of unreliable reconstruction and instability. To address this issue, we propose the Atoms Theory, which defines such units as atoms. We introduce the atomic inner product (AIP) to correct representation shifting, formally define atoms, and prove the conditions that atoms satisfy the Restricted Isometry Property (RIP), ensuring stable sparse representations over atom set and linking to compressed sensing. Under stronger conditions, we further establish the uniqueness and exact $\ell_1$ recoverability of the sparse representations, and provide guarantees that single-layer sparse autoencoders (SAEs) with threshold activations can reliably identify the atoms. To validate the Atoms Theory, we train threshold-activated SAEs on Gemma2-2B, Gemma2-9B, and Llama3.1-8B, achieving 99.9% sparse reconstruction across layers on average, and more than 99.8% of atoms satisfy the uniqueness condition, compared to 0.5% for neurons and 68.2% for features, showing that atoms more faithfully capture intrinsic representations of LLMs. Scaling experiments further reveal the link between SAEs size and recovery capacity. Overall, this work systematically introduces and validates Atoms Theory of LLMs, providing a theoretical framework for understanding internal representations and a foundation for mechanistic interpretability. Code available at this https URL. 

---
# SFT Doesn't Always Hurt General Capabilities: Revisiting Domain-Specific Fine-Tuning in LLMs 

**Authors**: Jiacheng Lin, Zhongruo Wang, Kun Qian, Tian Wang, Arvind Srinivasan, Hansi Zeng, Ruochen Jiao, Xie Zhou, Jiri Gesi, Dakuo Wang, Yufan Guo, Kai Zhong, Weiqi Zhang, Sujay Sanghavi, Changyou Chen, Hyokun Yun, Lihong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.20758)  

**Abstract**: Supervised Fine-Tuning (SFT) on domain-specific datasets is a common approach to adapt Large Language Models (LLMs) to specialized tasks but is often believed to degrade their general capabilities. In this work, we revisit this trade-off and present both empirical and theoretical insights. First, we show that SFT does not always hurt: using a smaller learning rate can substantially mitigate general performance degradation while preserving comparable target-domain performance. We then provide a theoretical analysis that explains these phenomena and further motivates a new method, Token-Adaptive Loss Reweighting (TALR). Building on this, and recognizing that smaller learning rates alone do not fully eliminate general-performance degradation in all cases, we evaluate a range of strategies for reducing general capability loss, including L2 regularization, LoRA, model averaging, FLOW, and our proposed TALR. Experimental results demonstrate that while no method completely eliminates the trade-off, TALR consistently outperforms these baselines in balancing domain-specific gains and general capabilities. Finally, we distill our findings into practical guidelines for adapting LLMs to new domains: (i) using a small learning rate to achieve a favorable trade-off, and (ii) when a stronger balance is further desired, adopt TALR as an effective strategy. 

---
# Confidence-guided Refinement Reasoning for Zero-shot Question Answering 

**Authors**: Youwon Jang, Woo Suk Choi, Minjoon Jung, Minsu Lee, Byoung-Tak Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20750)  

**Abstract**: We propose Confidence-guided Refinement Reasoning (C2R), a novel training-free framework applicable to question-answering (QA) tasks across text, image, and video domains. C2R strategically constructs and refines sub-questions and their answers (sub-QAs), deriving a better confidence score for the target answer. C2R first curates a subset of sub-QAs to explore diverse reasoning paths, then compares the confidence scores of the resulting answer candidates to select the most reliable final answer. Since C2R relies solely on confidence scores derived from the model itself, it can be seamlessly integrated with various existing QA models, demonstrating consistent performance improvements across diverse models and benchmarks. Furthermore, we provide essential yet underexplored insights into how leveraging sub-QAs affects model behavior, specifically analyzing the impact of both the quantity and quality of sub-QAs on achieving robust and reliable reasoning. 

---
# Probability Distribution Collapse: A Critical Bottleneck to Compact Unsupervised Neural Grammar Induction 

**Authors**: Jinwook Park, Kangil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.20734)  

**Abstract**: Unsupervised neural grammar induction aims to learn interpretable hierarchical structures from language data. However, existing models face an expressiveness bottleneck, often resulting in unnecessarily large yet underperforming grammars. We identify a core issue, $\textit{probability distribution collapse}$, as the underlying cause of this limitation. We analyze when and how the collapse emerges across key components of neural parameterization and introduce a targeted solution, $\textit{collapse-relaxing neural parameterization}$, to mitigate it. Our approach substantially improves parsing performance while enabling the use of significantly more compact grammars across a wide range of languages, as demonstrated through extensive empirical analysis. 

---
# MI-Fuse: Label Fusion for Unsupervised Domain Adaptation with Closed-Source Large-Audio Language Model 

**Authors**: Hsiao-Ying Huang, Yi-Cheng Lin, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.20706)  

**Abstract**: Large audio-language models (LALMs) show strong zero-shot ability on speech tasks, suggesting promise for speech emotion recognition (SER). However, SER in real-world deployments often fails under domain mismatch, where source data are unavailable and powerful LALMs are accessible only through an API. We ask: given only unlabeled target-domain audio and an API-only LALM, can a student model be adapted to outperform the LALM in the target domain? To this end, we propose MI-Fuse, a denoised label fusion framework that supplements the LALM with a source-domain trained SER classifier as an auxiliary teacher. The framework draws multiple stochastic predictions from both teachers, weights their mean distributions by mutual-information-based uncertainty, and stabilizes training with an exponential moving average teacher. Experiments across three public emotion datasets and six cross-domain transfers show consistent gains, with the student surpassing the LALM and outperforming the strongest baseline by 3.9%. This approach strengthens emotion-aware speech systems without sharing source data, enabling realistic adaptation. 

---
# Overcoming Black-box Attack Inefficiency with Hybrid and Dynamic Select Algorithms 

**Authors**: Abhinay Shankar Belde, Rohit Ramkumar, Jonathan Rusert  

**Link**: [PDF](https://arxiv.org/pdf/2509.20699)  

**Abstract**: Adversarial text attack research plays a crucial role in evaluating the robustness of NLP models. However, the increasing complexity of transformer-based architectures has dramatically raised the computational cost of attack testing, especially for researchers with limited resources (e.g., GPUs). Existing popular black-box attack methods often require a large number of queries, which can make them inefficient and impractical for researchers. To address these challenges, we propose two new attack selection strategies called Hybrid and Dynamic Select, which better combine the strengths of previous selection algorithms. Hybrid Select merges generalized BinarySelect techniques with GreedySelect by introducing a size threshold to decide which selection algorithm to use. Dynamic Select provides an alternative approach of combining the generalized Binary and GreedySelect by learning which lengths of texts each selection method should be applied to. This greatly reduces the number of queries needed while maintaining attack effectiveness (a limitation of BinarySelect). Across 4 datasets and 6 target models, our best method(sentence-level Hybrid Select) is able to reduce the number of required queries per attack up 25.82\% on average against both encoder models and LLMs, without losing the effectiveness of the attack. 

---
# RedHerring Attack: Testing the Reliability of Attack Detection 

**Authors**: Jonathan Rusert  

**Link**: [PDF](https://arxiv.org/pdf/2509.20691)  

**Abstract**: In response to adversarial text attacks, attack detection models have been proposed and shown to successfully identify text modified by adversaries. Attack detection models can be leveraged to provide an additional check for NLP models and give signals for human input. However, the reliability of these models has not yet been thoroughly explored. Thus, we propose and test a novel attack setting and attack, RedHerring. RedHerring aims to make attack detection models unreliable by modifying a text to cause the detection model to predict an attack, while keeping the classifier correct. This creates a tension between the classifier and detector. If a human sees that the detector is giving an ``incorrect'' prediction, but the classifier a correct one, then the human will see the detector as unreliable. We test this novel threat model on 4 datasets against 3 detectors defending 4 classifiers. We find that RedHerring is able to drop detection accuracy between 20 - 71 points, while maintaining (or improving) classifier accuracy. As an initial defense, we propose a simple confidence check which requires no retraining of the classifier or detector and increases detection accuracy greatly. This novel threat model offers new insights into how adversaries may target detection models. 

---
# Enhancing Molecular Property Prediction with Knowledge from Large Language Models 

**Authors**: Peng Zhou, Lai Hou Tim, Zhixiang Cheng, Kun Xie, Chaoyi Li, Wei Liu, Xiangxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2509.20664)  

**Abstract**: Predicting molecular properties is a critical component of drug discovery. Recent advances in deep learning, particularly Graph Neural Networks (GNNs), have enabled end-to-end learning from molecular structures, reducing reliance on manual feature engineering. However, while GNNs and self-supervised learning approaches have advanced molecular property prediction (MPP), the integration of human prior knowledge remains indispensable, as evidenced by recent methods that leverage large language models (LLMs) for knowledge extraction. Despite their strengths, LLMs are constrained by knowledge gaps and hallucinations, particularly for less-studied molecular properties. In this work, we propose a novel framework that, for the first time, integrates knowledge extracted from LLMs with structural features derived from pre-trained molecular models to enhance MPP. Our approach prompts LLMs to generate both domain-relevant knowledge and executable code for molecular vectorization, producing knowledge-based features that are subsequently fused with structural representations. We employ three state-of-the-art LLMs, GPT-4o, GPT-4.1, and DeepSeek-R1, for knowledge extraction. Extensive experiments demonstrate that our integrated method outperforms existing approaches, confirming that the combination of LLM-derived knowledge and structural information provides a robust and effective solution for MPP. 

---
# Building Tailored Speech Recognizers for Japanese Speaking Assessment 

**Authors**: Yotaro Kubo, Richard Sproat, Chihiro Taguchi, Llion Jones  

**Link**: [PDF](https://arxiv.org/pdf/2509.20655)  

**Abstract**: This paper presents methods for building speech recognizers tailored for Japanese speaking assessment tasks. Specifically, we build a speech recognizer that outputs phonemic labels with accent markers. Although Japanese is resource-rich, there is only a small amount of data for training models to produce accurate phonemic transcriptions that include accent marks. We propose two methods to mitigate data sparsity. First, a multitask training scheme introduces auxiliary loss functions to estimate orthographic text labels and pitch patterns of the input signal, so that utterances with only orthographic annotations can be leveraged in training. The second fuses two estimators, one over phonetic alphabet strings, and the other over text token sequences. To combine these estimates we develop an algorithm based on the finite-state transducer framework. Our results indicate that the use of multitask learning and fusion is effective for building an accurate phonemic recognizer. We show that this approach is advantageous compared to the use of generic multilingual recognizers. The relative advantages of the proposed methods were also compared. Our proposed methods reduced the average of mora-label error rates from 12.3% to 7.1% over the CSJ core evaluation sets. 

---
# Look Before you Leap: Estimating LLM Benchmark Scores from Descriptions 

**Authors**: Jungsoo Park, Ethan Mendes, Gabriel Stanovsky, Alan Ritter  

**Link**: [PDF](https://arxiv.org/pdf/2509.20645)  

**Abstract**: Progress in large language models is constrained by an evaluation bottleneck: build a benchmark, evaluate models and settings, then iterate. We therefore ask a simple question: can we forecast outcomes before running any experiments? We study text-only performance forecasting: estimating a model's score from a redacted task description and intended configuration, with no access to dataset instances. To support systematic study, we curate PRECOG, a corpus of redacted description-performance pairs spanning diverse tasks, domains, and metrics. Experiments show the task is challenging but feasible: models equipped with a retrieval module that excludes source papers achieve moderate prediction performance with well-calibrated uncertainty, reaching mean absolute error as low as 8.7 on the Accuracy subset at high-confidence thresholds. Our analysis indicates that stronger reasoning models engage in diverse, iterative querying, whereas current open-source models lag and often skip retrieval or gather evidence with limited diversity. We further test a zero-leakage setting, forecasting on newly released datasets or experiments before their papers are indexed, where GPT-5 with built-in web search still attains nontrivial prediction accuracy. Overall, our corpus and analyses offer an initial step toward open-ended anticipatory evaluation, supporting difficulty estimation and smarter experiment prioritization. 

---
# FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Models 

**Authors**: Amin Karimi Monsefi, Nikhil Bhendawade, Manuel Rafael Ciosici, Dominic Culver, Yizhe Zhang, Irina Belousova  

**Link**: [PDF](https://arxiv.org/pdf/2509.20624)  

**Abstract**: Autoregressive language models (ARMs) deliver strong likelihoods, but are inherently serial: they generate one token per forward pass, which limits throughput and inflates latency for long sequences. Diffusion Language Models (DLMs) parallelize across positions and thus appear promising for language generation, yet standard discrete diffusion typically needs hundreds to thousands of model evaluations to reach high quality, trading serial depth for iterative breadth. We introduce FS-DFM, Few-Step Discrete Flow-Matching. A discrete flow-matching model designed for speed without sacrificing quality. The core idea is simple: make the number of sampling steps an explicit parameter and train the model to be consistent across step budgets, so one big move lands where many small moves would. We pair this with a reliable update rule that moves probability in the right direction without overshooting, and with strong teacher guidance distilled from long-run trajectories. Together, these choices make few-step sampling stable, accurate, and easy to control. On language modeling benchmarks, FS-DFM with 8 sampling steps achieves perplexity parity with a 1,024-step discrete-flow baseline for generating 1,024 tokens using a similar-size model, delivering up to 128 times faster sampling and corresponding latency/throughput gains. 

---
# Hierarchical Resolution Transformers: A Wavelet-Inspired Architecture for Multi-Scale Language Understanding 

**Authors**: Ayan Sar, Sampurna Roy, Kanav Gupta, Anurag Kaushish, Tanupriya Choudhury, Abhijit Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.20581)  

**Abstract**: Transformer architectures have achieved state-of-the-art performance across natural language tasks, yet they fundamentally misrepresent the hierarchical nature of human language by processing text as flat token sequences. This results in quadratic computational cost, weak computational cost, weak compositional generalization, and inadequate discourse-level modeling. We propose Hierarchical Resolution Transformer (HRT), a novel wavelet-inspired neural architecture that processes language simultaneously across multiple resolutions, from characters to discourse-level units. HRT constructs a multi-resolution attention, enabling bottom-up composition and top-down contextualization. By employing exponential sequence reduction across scales, HRT achieves O(nlogn) complexity, offering significant efficiency improvements over standard transformers. We evaluated HRT on a diverse suite of benchmarks, including GLUE, SuperGLUE, Long Range Arena, and WikiText-103, and results demonstrated that HRT outperforms standard transformer baselines by an average of +3.8% on GLUE, +4.5% on SuperGLUE, and +6.1% on Long Range Arena, while reducing memory usage by 42% and inference latency by 37% compared to BERT and GPT style models of similar parameter count. Ablation studies confirm the effectiveness of cross-resolution attention and scale-specialized modules, showing that each contributes independently to both efficiency and accuracy. Our findings establish HRT as the first architecture to align computational structure with the hierarchical organization of human language, demonstrating that multi-scale, wavelet-inspired processing yields both theoretical efficiency gains and practical improvements in language understanding. 

---
# Dynamic Reasoning Chains through Depth-Specialized Mixture-of-Experts in Transformer Architectures 

**Authors**: Sampurna Roy, Ayan Sar, Anurag Kaushish, Kanav Gupta, Tanupriya Choudhury, Abhijit Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.20577)  

**Abstract**: Contemporary transformer architectures apply identical processing depth to all inputs, creating inefficiencies and limiting reasoning quality. Simple factual queries are subjected to the same multilayered computation as complex logical problems, wasting resources while constraining deep inference. To overcome this, we came up with a concept of Dynamic Reasoning Chains through Depth Specialised Mixture of Experts (DS-MoE), a modular framework that extends the Mixture of Experts paradigm from width-based to depth specialised computation. DS-MoE introduces expert modules optimised for distinct reasoning depths, shallow pattern recognition, compositional reasoning, logical inference, memory integration, and meta-cognitive supervision. A learned routing network dynamically assembles custom reasoning chains, activating only the necessary experts to match input complexity. The dataset on which we trained and evaluated DS-MoE is on The Pile, an 800GB corpus covering diverse domains such as scientific papers, legal texts, programming code, and web content, enabling systematic assessment across reasoning depths. Experimental results demonstrate that DS-MoE achieves up to 16 per cent computational savings and 35 per cent faster inference compared to uniform-depth transformers, while delivering 2.8 per cent higher accuracy on complex multi-step reasoning benchmarks. Furthermore, routing decisions yield interpretable reasoning chains, enhancing transparency and scalability. These findings establish DS-MoE as a significant advancement in adaptive neural architectures, demonstrating that depth-specialised modular processing can simultaneously improve efficiency, reasoning quality, and interpretability in large-scale language models. 

---
# SwasthLLM: a Unified Cross-Lingual, Multi-Task, and Meta-Learning Zero-Shot Framework for Medical Diagnosis Using Contrastive Representations 

**Authors**: Ayan Sar, Pranav Singh Puri, Sumit Aich, Tanupriya Choudhury, Abhijit Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.20567)  

**Abstract**: In multilingual healthcare environments, automatic disease diagnosis from clinical text remains a challenging task due to the scarcity of annotated medical data in low-resource languages and the linguistic variability across populations. This paper proposes SwasthLLM, a unified, zero-shot, cross-lingual, and multi-task learning framework for medical diagnosis that operates effectively across English, Hindi, and Bengali without requiring language-specific fine-tuning. At its core, SwasthLLM leverages the multilingual XLM-RoBERTa encoder augmented with a language-aware attention mechanism and a disease classification head, enabling the model to extract medically relevant information regardless of the language structure. To align semantic representations across languages, a Siamese contrastive learning module is introduced, ensuring that equivalent medical texts in different languages produce similar embeddings. Further, a translation consistency module and a contrastive projection head reinforce language-invariant representation learning. SwasthLLM is trained using a multi-task learning strategy, jointly optimizing disease classification, translation alignment, and contrastive learning objectives. Additionally, we employ Model-Agnostic Meta-Learning (MAML) to equip the model with rapid adaptation capabilities for unseen languages or tasks with minimal data. Our phased training pipeline emphasizes robust representation alignment before task-specific fine-tuning. Extensive evaluation shows that SwasthLLM achieves high diagnostic performance, with a test accuracy of 97.22% and an F1-score of 97.17% in supervised settings. Crucially, in zero-shot scenarios, it attains 92.78% accuracy on Hindi and 73.33% accuracy on Bengali medical text, demonstrating strong generalization in low-resource contexts. 

---
# SiniticMTError: A Machine Translation Dataset with Error Annotations for Sinitic Languages 

**Authors**: Hannah Liu, Junghyun Min, Ethan Yue Heng Cheung, Shou-Yi Hung, Syed Mekael Wasti, Runtong Liang, Shiyao Qian, Shizhao Zheng, Elsie Chan, Ka Ieng Charlotte Lo, Wing Yu Yip, Richard Tzong-Han Tsai, En-Shiun Annie Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.20557)  

**Abstract**: Despite major advances in machine translation (MT) in recent years, progress remains limited for many low-resource languages that lack large-scale training data and linguistic resources. Cantonese and Wu Chinese are two Sinitic examples, although each enjoys more than 80 million speakers around the world. In this paper, we introduce SiniticMTError, a novel dataset that builds on existing parallel corpora to provide error span, error type, and error severity annotations in machine-translated examples from English to Mandarin, Cantonese, and Wu Chinese. Our dataset serves as a resource for the MT community to utilize in fine-tuning models with error detection capabilities, supporting research on translation quality estimation, error-aware generation, and low-resource language evaluation. We report our rigorous annotation process by native speakers, with analyses on inter-annotator agreement, iterative feedback, and patterns in error type and severity. 

---
# MARS: toward more efficient multi-agent collaboration for LLM reasoning 

**Authors**: Xiao Wang, Jia Wang, Yijie Wang, Pengtao Dang, Sha Cao, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20502)  

**Abstract**: Large language models (LLMs) have achieved impressive results in natural language understanding, yet their reasoning capabilities remain limited when operating as single agents. Multi-Agent Debate (MAD) has been proposed to address this limitation by enabling collaborative reasoning among multiple models in a round-table debate manner. While effective, MAD introduces substantial computational overhead due to the number of agents involved and the frequent communication required. In this paper, we propose MARS (Multi-Agent Review System), a role-based collaboration framework inspired by the review process. In MARS, an author agent generates an initial solution, reviewer agents provide decisions and comments independently, and a meta-reviewer integrates the feedback to make the final decision and guide further revision. This design enhances reasoning quality while avoiding costly reviewer-to-reviewer interactions, thereby controlling token consumption and inference time. We compared MARS with both MAD and other state-of-the-art reasoning strategies across multiple benchmarks. Extensive experiments with different LLMs show that MARS matches the accuracy of MAD while reducing both token usage and inference time by approximately 50\%. Code is available at this https URL. 

---
# ShortCheck: Checkworthiness Detection of Multilingual Short-Form Videos 

**Authors**: Henrik Vatndal, Vinay Setty  

**Link**: [PDF](https://arxiv.org/pdf/2509.20467)  

**Abstract**: Short-form video platforms like TikTok present unique challenges for misinformation detection due to their multimodal, dynamic, and noisy content. We present ShortCheck, a modular, inference-only pipeline with a user-friendly interface that automatically identifies checkworthy short-form videos to help human fact-checkers. The system integrates speech transcription, OCR, object and deepfake detection, video-to-text summarization, and claim verification. ShortCheck is validated by evaluating it on two manually annotated datasets with TikTok videos in a multilingual setting. The pipeline achieves promising results with F1-weighted score over 70\%. 

---
# Document Summarization with Conformal Importance Guarantees 

**Authors**: Bruce Kuwahara, Chen-Yuan Lin, Xiao Shi Huang, Kin Kwan Leung, Jullian Arta Yapeter, Ilya Stanevich, Felipe Perez, Jesse C. Cresswell  

**Link**: [PDF](https://arxiv.org/pdf/2509.20461)  

**Abstract**: Automatic summarization systems have advanced rapidly with large language models (LLMs), yet they still lack reliable guarantees on inclusion of critical content in high-stakes domains like healthcare, law, and finance. In this work, we introduce Conformal Importance Summarization, the first framework for importance-preserving summary generation which uses conformal prediction to provide rigorous, distribution-free coverage guarantees. By calibrating thresholds on sentence-level importance scores, we enable extractive document summarization with user-specified coverage and recall rates over critical content. Our method is model-agnostic, requires only a small calibration set, and seamlessly integrates with existing black-box LLMs. Experiments on established summarization benchmarks demonstrate that Conformal Importance Summarization achieves the theoretically assured information coverage rate. Our work suggests that Conformal Importance Summarization can be combined with existing techniques to achieve reliable, controllable automatic summarization, paving the way for safer deployment of AI summarization tools in critical applications. Code is available at this https URL. 

---
# USB-Rec: An Effective Framework for Improving Conversational Recommendation Capability of Large Language Model 

**Authors**: Jianyu Wen, Jingyun Wang, Cilin Yan, Jiayin Cai, Xiaolong Jiang, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20381)  

**Abstract**: Recently, Large Language Models (LLMs) have been widely employed in Conversational Recommender Systems (CRSs). Unlike traditional language model approaches that focus on training, all existing LLMs-based approaches are mainly centered around how to leverage the summarization and analysis capabilities of LLMs while ignoring the issue of training. Therefore, in this work, we propose an integrated training-inference framework, User-Simulator-Based framework (USB-Rec), for improving the performance of LLMs in conversational recommendation at the model level. Firstly, we design a LLM-based Preference Optimization (PO) dataset construction strategy for RL training, which helps the LLMs understand the strategies and methods in conversational recommendation. Secondly, we propose a Self-Enhancement Strategy (SES) at the inference stage to further exploit the conversational recommendation potential obtained from RL training. Extensive experiments on various datasets demonstrate that our method consistently outperforms previous state-of-the-art methods. 

---
# Beyond Global Emotion: Fine-Grained Emotional Speech Synthesis with Dynamic Word-Level Modulation 

**Authors**: Sirui Wang, Andong Chen, Tiejun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.20378)  

**Abstract**: Emotional text-to-speech (E-TTS) is central to creating natural and trustworthy human-computer interaction. Existing systems typically rely on sentence-level control through predefined labels, reference audio, or natural language prompts. While effective for global emotion expression, these approaches fail to capture dynamic shifts within a sentence. To address this limitation, we introduce Emo-FiLM, a fine-grained emotion modeling framework for LLM-based TTS. Emo-FiLM aligns frame-level features from emotion2vec to words to obtain word-level emotion annotations, and maps them through a Feature-wise Linear Modulation (FiLM) layer, enabling word-level emotion control by directly modulating text embeddings. To support evaluation, we construct the Fine-grained Emotion Dynamics Dataset (FEDD) with detailed annotations of emotional transitions. Experiments show that Emo-FiLM outperforms existing approaches on both global and fine-grained tasks, demonstrating its effectiveness and generality for expressive speech synthesis. 

---
# SKILL-RAG: Self-Knowledge Induced Learning and Filtering for Retrieval-Augmented Generation 

**Authors**: Tomoaki Isoda  

**Link**: [PDF](https://arxiv.org/pdf/2509.20377)  

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly improved the performance of large language models (LLMs) on knowledge-intensive tasks in recent years. However, since retrieval systems may return irrelevant content, incorporating such information into the model often leads to hallucinations. Thus, identifying and filtering out unhelpful retrieved content is a key challenge for improving RAG this http URL better integrate the internal knowledge of the model with external knowledge from retrieval, it is essential to understand what the model "knows" and "does not know" (which is also called "self-knowledge"). Based on this insight, we propose SKILL-RAG (Self-Knowledge Induced Learning and Filtering for RAG), a novel method that leverages the model's self-knowledge to determine which retrieved documents are beneficial for answering a given query. We design a reinforcement learning-based training framework to explicitly elicit self-knowledge from the model and employs sentence-level granularity to filter out irrelevant content while preserving useful this http URL evaluate SKILL-RAG using Llama2-7B and Qwen3-8B on several question answering benchmarks. Experimental results demonstrate that SKILL-RAG not only improves generation quality but also significantly reduces the number of input documents, validating the importance of self-knowledge in guiding the selection of high-quality retrievals. 

---
# ConceptViz: A Visual Analytics Approach for Exploring Concepts in Large Language Models 

**Authors**: Haoxuan Li, Zhen Wen, Qiqi Jiang, Chenxiao Li, Yuwei Wu, Yuchen Yang, Yiyao Wang, Xiuqi Huang, Minfeng Zhu, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.20376)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance across a wide range of natural language tasks. Understanding how LLMs internally represent knowledge remains a significant challenge. Despite Sparse Autoencoders (SAEs) have emerged as a promising technique for extracting interpretable features from LLMs, SAE features do not inherently align with human-understandable concepts, making their interpretation cumbersome and labor-intensive. To bridge the gap between SAE features and human concepts, we present ConceptViz, a visual analytics system designed for exploring concepts in LLMs. ConceptViz implements a novel dentification => Interpretation => Validation pipeline, enabling users to query SAEs using concepts of interest, interactively explore concept-to-feature alignments, and validate the correspondences through model behavior verification. We demonstrate the effectiveness of ConceptViz through two usage scenarios and a user study. Our results show that ConceptViz enhances interpretability research by streamlining the discovery and validation of meaningful concept representations in LLMs, ultimately aiding researchers in building more accurate mental models of LLM features. Our code and user guide are publicly available at this https URL. 

---
# Assessing Classical Machine Learning and Transformer-based Approaches for Detecting AI-Generated Research Text 

**Authors**: Sharanya Parimanoharan, Ruwan D. Nawarathna  

**Link**: [PDF](https://arxiv.org/pdf/2509.20375)  

**Abstract**: The rapid adoption of large language models (LLMs) such as ChatGPT has blurred the line between human and AI-generated texts, raising urgent questions about academic integrity, intellectual property, and the spread of misinformation. Thus, reliable AI-text detection is needed for fair assessment to safeguard human authenticity and cultivate trust in digital communication. In this study, we investigate how well current machine learning (ML) approaches can distinguish ChatGPT-3.5-generated texts from human-written texts employing a labeled data set of 250 pairs of abstracts from a wide range of research topics. We test and compare both classical (Logistic Regression armed with classical Bag-of-Words, POS, and TF-IDF features) and transformer-based (BERT augmented with N-grams, DistilBERT, BERT with a lightweight custom classifier, and LSTM-based N-gram models) ML detection techniques. As we aim to assess each model's performance in detecting AI-generated research texts, we also aim to test whether an ensemble of these models can outperform any single detector. Results show DistilBERT achieves the overall best performance, while Logistic Regression and BERT-Custom offer solid, balanced alternatives; LSTM- and BERT-N-gram approaches lag. The max voting ensemble of the three best models fails to surpass DistilBERT itself, highlighting the primacy of a single transformer-based representation over mere model diversity. By comprehensively assessing the strengths and weaknesses of these AI-text detection approaches, this work lays a foundation for more robust transformer frameworks with larger, richer datasets to keep pace with ever-improving generative AI models. 

---
# CFD-LLMBench: A Benchmark Suite for Evaluating Large Language Models in Computational Fluid Dynamics 

**Authors**: Nithin Somasekharan, Ling Yue, Yadi Cao, Weichao Li, Patrick Emami, Pochinapeddi Sai Bhargav, Anurag Acharya, Xingyu Xie, Shaowu Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.20374)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong performance across general NLP tasks, but their utility in automating numerical experiments of complex physical system -- a critical and labor-intensive component -- remains underexplored. As the major workhorse of computational science over the past decades, Computational Fluid Dynamics (CFD) offers a uniquely challenging testbed for evaluating the scientific capabilities of LLMs. We introduce CFDLLMBench, a benchmark suite comprising three complementary components -- CFDQuery, CFDCodeBench, and FoamBench -- designed to holistically evaluate LLM performance across three key competencies: graduate-level CFD knowledge, numerical and physical reasoning of CFD, and context-dependent implementation of CFD workflows. Grounded in real-world CFD practices, our benchmark combines a detailed task taxonomy with a rigorous evaluation framework to deliver reproducible results and quantify LLM performance across code executability, solution accuracy, and numerical convergence behavior. CFDLLMBench establishes a solid foundation for the development and evaluation of LLM-driven automation of numerical experiments for complex physical systems. Code and data are available at this https URL. 

---
# Speaker Style-Aware Phoneme Anchoring for Improved Cross-Lingual Speech Emotion Recognition 

**Authors**: Shreya G. Upadhyay, Carlos Busso, Chi-Chun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.20373)  

**Abstract**: Cross-lingual speech emotion recognition (SER) remains a challenging task due to differences in phonetic variability and speaker-specific expressive styles across languages. Effectively capturing emotion under such diverse conditions requires a framework that can align the externalization of emotions across different speakers and languages. To address this problem, we propose a speaker-style aware phoneme anchoring framework that aligns emotional expression at the phonetic and speaker levels. Our method builds emotion-specific speaker communities via graph-based clustering to capture shared speaker traits. Using these groups, we apply dual-space anchoring in speaker and phonetic spaces to enable better emotion transfer across languages. Evaluations on the MSP-Podcast (English) and BIIC-Podcast (Taiwanese Mandarin) corpora demonstrate improved generalization over competitive baselines and provide valuable insights into the commonalities in cross-lingual emotion representation. 

---
# Interpreting Public Sentiment in Diplomacy Events: A Counterfactual Analysis Framework Using Large Language Models 

**Authors**: Leyi Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20367)  

**Abstract**: Diplomatic events consistently prompt widespread public discussion and debate. Public sentiment plays a critical role in diplomacy, as a good sentiment provides vital support for policy implementation, helps resolve international issues, and shapes a nation's international image. Traditional methods for gauging public sentiment, such as large-scale surveys or manual content analysis of media, are typically time-consuming, labor-intensive, and lack the capacity for forward-looking analysis. We propose a novel framework that identifies specific modifications for diplomatic event narratives to shift public sentiment from negative to neutral or positive. First, we train a language model to predict public reaction towards diplomatic events. To this end, we construct a dataset comprising descriptions of diplomatic events and their associated public discussions. Second, guided by communication theories and in collaboration with domain experts, we predetermined several textual features for modification, ensuring that any alterations changed the event's narrative framing while preserving its core this http URL develop a counterfactual generation algorithm that employs a large language model to systematically produce modified versions of an original text. The results show that this framework successfully shifted public sentiment to a more favorable state with a 70\% success rate. This framework can therefore serve as a practical tool for diplomats, policymakers, and communication specialists, offering data-driven insights on how to frame diplomatic initiatives or report on events to foster a more desirable public sentiment. 

---
# Interactive Recommendation Agent with Active User Commands 

**Authors**: Jiakai Tang, Yujie Luo, Xunke Xi, Fei Sun, Xueyang Feng, Sunhao Dai, Chao Yi, Dian Chen, Zhujin Gao, Yang Li, Xu Chen, Wen Chen, Jian Wu, Yuning Jiang, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.21317)  

**Abstract**: Traditional recommender systems rely on passive feedback mechanisms that limit users to simple choices such as like and dislike. However, these coarse-grained signals fail to capture users' nuanced behavior motivations and intentions. In turn, current systems cannot also distinguish which specific item attributes drive user satisfaction or dissatisfaction, resulting in inaccurate preference modeling. These fundamental limitations create a persistent gap between user intentions and system interpretations, ultimately undermining user satisfaction and harming system effectiveness.
To address these limitations, we introduce the Interactive Recommendation Feed (IRF), a pioneering paradigm that enables natural language commands within mainstream recommendation feeds. Unlike traditional systems that confine users to passive implicit behavioral influence, IRF empowers active explicit control over recommendation policies through real-time linguistic commands. To support this paradigm, we develop RecBot, a dual-agent architecture where a Parser Agent transforms linguistic expressions into structured preferences and a Planner Agent dynamically orchestrates adaptive tool chains for on-the-fly policy adjustment. To enable practical deployment, we employ simulation-augmented knowledge distillation to achieve efficient performance while maintaining strong reasoning capabilities. Through extensive offline and long-term online experiments, RecBot shows significant improvements in both user satisfaction and business outcomes. 

---
# Hallucination as an Upper Bound: A New Perspective on Text-to-Image Evaluation 

**Authors**: Seyed Amir Kasaei, Mohammad Hossein Rohban  

**Link**: [PDF](https://arxiv.org/pdf/2509.21257)  

**Abstract**: In language and vision-language models, hallucination is broadly understood as content generated from a model's prior knowledge or biases rather than from the given input. While this phenomenon has been studied in those domains, it has not been clearly framed for text-to-image (T2I) generative models. Existing evaluations mainly focus on alignment, checking whether prompt-specified elements appear, but overlook what the model generates beyond the prompt. We argue for defining hallucination in T2I as bias-driven deviations and propose a taxonomy with three categories: attribute, relation, and object hallucinations. This framing introduces an upper bound for evaluation and surfaces hidden biases, providing a foundation for richer assessment of T2I models. 

---
# Evaluating the Evaluators: Metrics for Compositional Text-to-Image Generation 

**Authors**: Seyed Amir Kasaei, Ali Aghayari, Arash Marioriyad, Niki Sepasian, MohammadAmin Fazli, Mahdieh Soleymani Baghshah, Mohammad Hossein Rohban  

**Link**: [PDF](https://arxiv.org/pdf/2509.21227)  

**Abstract**: Text-image generation has advanced rapidly, but assessing whether outputs truly capture the objects, attributes, and relations described in prompts remains a central challenge. Evaluation in this space relies heavily on automated metrics, yet these are often adopted by convention or popularity rather than validated against human judgment. Because evaluation and reported progress in the field depend directly on these metrics, it is critical to understand how well they reflect human preferences. To address this, we present a broad study of widely used metrics for compositional text-image evaluation. Our analysis goes beyond simple correlation, examining their behavior across diverse compositional challenges and comparing how different metric families align with human judgments. The results show that no single metric performs consistently across tasks: performance varies with the type of compositional problem. Notably, VQA-based metrics, though popular, are not uniformly superior, while certain embedding-based metrics prove stronger in specific cases. Image-only metrics, as expected, contribute little to compositional evaluation, as they are designed for perceptual quality rather than alignment. These findings underscore the importance of careful and transparent metric selection, both for trustworthy evaluation and for their use as reward models in generation. Project page is available at \href{this https URL}{this URL}. 

---
# Sigma: Semantically Informative Pre-training for Skeleton-based Sign Language Understanding 

**Authors**: Muxin Pu, Mei Kuan Lim, Chun Yong Chong, Chen Change Loy  

**Link**: [PDF](https://arxiv.org/pdf/2509.21223)  

**Abstract**: Pre-training has proven effective for learning transferable features in sign language understanding (SLU) tasks. Recently, skeleton-based methods have gained increasing attention because they can robustly handle variations in subjects and backgrounds without being affected by appearance or environmental factors. Current SLU methods continue to face three key limitations: 1) weak semantic grounding, as models often capture low-level motion patterns from skeletal data but struggle to relate them to linguistic meaning; 2) imbalance between local details and global context, with models either focusing too narrowly on fine-grained cues or overlooking them for broader context; and 3) inefficient cross-modal learning, as constructing semantically aligned representations across modalities remains difficult. To address these, we propose Sigma, a unified skeleton-based SLU framework featuring: 1) a sign-aware early fusion mechanism that facilitates deep interaction between visual and textual modalities, enriching visual features with linguistic context; 2) a hierarchical alignment learning strategy that jointly maximises agreements across different levels of paired features from different modalities, effectively capturing both fine-grained details and high-level semantic relationships; and 3) a unified pre-training framework that combines contrastive learning, text matching and language modelling to promote semantic consistency and generalisation. Sigma achieves new state-of-the-art results on isolated sign language recognition, continuous sign language recognition, and gloss-free sign language translation on multiple benchmarks spanning different sign and spoken languages, demonstrating the impact of semantically informative pre-training and the effectiveness of skeletal data as a stand-alone solution for SLU. 

---
# TABLET: A Large-Scale Dataset for Robust Visual Table Understanding 

**Authors**: Iñigo Alonso, Imanol Miranda, Eneko Agirre, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2509.21205)  

**Abstract**: While table understanding increasingly relies on pixel-only settings where tables are processed as visual representations, current benchmarks predominantly use synthetic renderings that lack the complexity and visual diversity of real-world tables. Additionally, existing visual table understanding (VTU) datasets offer fixed examples with single visualizations and pre-defined instructions, providing no access to underlying serialized data for reformulation. We introduce TABLET, a large-scale VTU dataset with 4 million examples across 20 tasks, grounded in 2 million unique tables where 88% preserve original visualizations. Each example includes paired image-HTML representations, comprehensive metadata, and provenance information linking back to the source datasets. Fine-tuning vision-language models like Qwen2.5-VL-7B on TABLET improves performance on seen and unseen VTU tasks while increasing robustness on real-world table visualizations. By preserving original visualizations and maintaining example traceability in a unified large-scale collection, TABLET establishes a foundation for robust training and extensible evaluation of future VTU models. 

---
# Automotive-ENV: Benchmarking Multimodal Agents in Vehicle Interface Systems 

**Authors**: Junfeng Yan, Biao Wu, Meng Fang, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21143)  

**Abstract**: Multimodal agents have demonstrated strong performance in general GUI interactions, but their application in automotive systems has been largely unexplored. In-vehicle GUIs present distinct challenges: drivers' limited attention, strict safety requirements, and complex location-based interaction patterns. To address these challenges, we introduce Automotive-ENV, the first high-fidelity benchmark and interaction environment tailored for vehicle GUIs. This platform defines 185 parameterized tasks spanning explicit control, implicit intent understanding, and safety-aware tasks, and provides structured multimodal observations with precise programmatic checks for reproducible evaluation. Building on this benchmark, we propose ASURADA, a geo-aware multimodal agent that integrates GPS-informed context to dynamically adjust actions based on location, environmental conditions, and regional driving norms. Experiments show that geo-aware information significantly improves success on safety-aware tasks, highlighting the importance of location-based context in automotive environments. We will release Automotive-ENV, complete with all tasks and benchmarking tools, to further the development of safe and adaptive in-vehicle agents. 

---
# Expanding Reasoning Potential in Foundation Model by Learning Diverse Chains of Thought Patterns 

**Authors**: Xuemiao Zhang, Can Ren, Chengying Tu, Rongxiang Weng, Shuo Wang, Hongfei Yan, Jingang Wang, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.21124)  

**Abstract**: Recent progress in large reasoning models for challenging mathematical reasoning has been driven by reinforcement learning (RL). Incorporating long chain-of-thought (CoT) data during mid-training has also been shown to substantially improve reasoning depth. However, current approaches often utilize CoT data indiscriminately, leaving open the critical question of which data types most effectively enhance model reasoning capabilities. In this paper, we define the foundation model's reasoning potential for the first time as the inverse of the number of independent attempts required to correctly answer the question, which is strongly correlated with the final model performance. We then propose utilizing diverse data enriched with high-value reasoning patterns to expand the reasoning potential. Specifically, we abstract atomic reasoning patterns from CoT sequences, characterized by commonality and inductive capabilities, and use them to construct a core reference set enriched with valuable reasoning patterns. Furthermore, we propose a dual-granularity algorithm involving chains of reasoning patterns and token entropy, efficiently selecting high-value CoT data (CoTP) from the data pool that aligns with the core set, thereby training models to master reasoning effectively. Only 10B-token CoTP data enables the 85A6B Mixture-of-Experts (MoE) model to improve by 9.58% on the challenging AIME 2024 and 2025, and to raise the upper bound of downstream RL performance by 7.81%. 

---
# TrustJudge: Inconsistencies of LLM-as-a-Judge and How to Alleviate Them 

**Authors**: Yidong Wang, Yunze Song, Tingyuan Zhu, Xuanwang Zhang, Zhuohao Yu, Hao Chen, Chiyu Song, Qiufeng Wang, Cunxiang Wang, Zhen Wu, Xinyu Dai, Yue Zhang, Wei Ye, Shikun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21117)  

**Abstract**: The adoption of Large Language Models (LLMs) as automated evaluators (LLM-as-a-judge) has revealed critical inconsistencies in current evaluation frameworks. We identify two fundamental types of inconsistencies: (1) Score-Comparison Inconsistency, where lower-rated responses outperform higher-scored ones in pairwise comparisons, and (2) Pairwise Transitivity Inconsistency, manifested through circular preference chains (A>B>C>A) and equivalence contradictions (A=B=C\neq A). We argue that these issues come from information loss in discrete rating systems and ambiguous tie judgments during pairwise evaluation. We propose TrustJudge, a probabilistic framework that addresses these limitations through two key innovations: 1) distribution-sensitive scoring that computes continuous expectations from discrete rating probabilities, preserving information entropy for more precise scoring, and 2) likelihood-aware aggregation that resolves transitivity violations using bidirectional preference probabilities or perplexity. We also formalize the theoretical limitations of current LLM-as-a-judge frameworks and demonstrate how TrustJudge's components overcome them. When evaluated with Llama-3.1-70B-Instruct as judge using our dataset, TrustJudge reduces Score-Comparison inconsistency by 8.43% (from 23.32% to 14.89%) and Pairwise Transitivity inconsistency by 10.82% (from 15.22% to 4.40%), while maintaining higher evaluation accuracy. Our work provides the first systematic analysis of evaluation framework inconsistencies in LLM-as-a-judge paradigms, offering both theoretical insights and practical solutions for reliable automated assessment. The framework demonstrates consistent improvements across various model architectures and scales, enabling more trustworthy LLM evaluation without requiring additional training or human annotations. The codes can be found at this https URL. 

---
# Communication Bias in Large Language Models: A Regulatory Perspective 

**Authors**: Adrian Kuenzler, Stefan Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2509.21075)  

**Abstract**: Large language models (LLMs) are increasingly central to many applications, raising concerns about bias, fairness, and regulatory compliance. This paper reviews risks of biased outputs and their societal impact, focusing on frameworks like the EU's AI Act and the Digital Services Act. We argue that beyond constant regulation, stronger attention to competition and design governance is needed to ensure fair, trustworthy AI. This is a preprint of the Communications of the ACM article of the same title. 

---
# ScaleDiff: Scaling Difficult Problems for Advanced Mathematical Reasoning 

**Authors**: Qizhi Pei, Zhuoshi Pan, Honglin Lin, Xin Gao, Yu Li, Zinan Tang, Conghui He, Rui Yan, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21070)  

**Abstract**: Large Reasoning Models (LRMs) have shown impressive capabilities in complex problem-solving, often benefiting from training on difficult mathematical problems that stimulate intricate reasoning. Recent efforts have explored automated synthesis of mathematical problems by prompting proprietary models or large-scale open-source models from seed data or inherent mathematical concepts. However, scaling up these methods remains challenging due to their high computational/API cost, complexity of prompting, and limited difficulty level of the generated problems. To overcome these limitations, we propose ScaleDiff, a simple yet effective pipeline designed to scale the creation of difficult problems. We efficiently identify difficult problems from existing datasets with only a single forward pass using an adaptive thinking model, which can perceive problem difficulty and automatically switch between "Thinking" and "NoThinking" modes. We then train a specialized difficult problem generator (DiffGen-8B) on this filtered difficult data, which can produce new difficult problems in large scale, eliminating the need for complex, per-instance prompting and its associated high API costs. Fine-tuning Qwen2.5-Math-7B-Instruct on the ScaleDiff-Math dataset yields a substantial performance increase of 11.3% compared to the original dataset and achieves a 65.9% average accuracy on AIME'24, AIME'25, HMMT-Feb'25, BRUMO'25, and MATH500, outperforming recent strong LRMs like OpenThinker3. Notably, this performance is achieved using the cost-efficient Qwen3-8B model as a teacher, demonstrating that our pipeline can effectively transfer advanced reasoning capabilities without relying on larger, more expensive teacher models. Furthermore, we observe a clear scaling phenomenon in model performance on difficult benchmarks as the quantity of difficult problems increases. Code: this https URL. 

---
# PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints 

**Authors**: Jiahao Huo, Shuliang Liu, Bin Wang, Junyan Zhang, Yibo Yan, Aiwei Liu, Xuming Hu, Mingxun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.21057)  

**Abstract**: Semantic-level watermarking (SWM) for large language models (LLMs) enhances watermarking robustness against text modifications and paraphrasing attacks by treating the sentence as the fundamental unit. However, existing methods still lack strong theoretical guarantees of robustness, and reject-sampling-based generation often introduces significant distribution distortions compared with unwatermarked outputs. In this work, we introduce a new theoretical framework on SWM through the concept of proxy functions (PFs) $\unicode{x2013}$ functions that map sentences to scalar values. Building on this framework, we propose PMark, a simple yet powerful SWM method that estimates the PF median for the next sentence dynamically through sampling while enforcing multiple PF constraints (which we call channels) to strengthen watermark evidence. Equipped with solid theoretical guarantees, PMark achieves the desired distortion-free property and improves the robustness against paraphrasing-style attacks. We also provide an empirically optimized version that further removes the requirement for dynamical median estimation for better sampling efficiency. Experimental results show that PMark consistently outperforms existing SWM baselines in both text quality and robustness, offering a more effective paradigm for detecting machine-generated text. Our code will be released at [this URL](this https URL). 

---
# Disagreements in Reasoning: How a Model's Thinking Process Dictates Persuasion in Multi-Agent Systems 

**Authors**: Haodong Zhao, Jidong Li, Zhaomin Wu, Tianjie Ju, Zhuosheng Zhang, Bingsheng He, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21054)  

**Abstract**: The rapid proliferation of recent Multi-Agent Systems (MAS), where Large Language Models (LLMs) and Large Reasoning Models (LRMs) usually collaborate to solve complex problems, necessitates a deep understanding of the persuasion dynamics that govern their interactions. This paper challenges the prevailing hypothesis that persuasive efficacy is primarily a function of model scale. We propose instead that these dynamics are fundamentally dictated by a model's underlying cognitive process, especially its capacity for explicit reasoning. Through a series of multi-agent persuasion experiments, we uncover a fundamental trade-off we term the Persuasion Duality. Our findings reveal that the reasoning process in LRMs exhibits significantly greater resistance to persuasion, maintaining their initial beliefs more robustly. Conversely, making this reasoning process transparent by sharing the "thinking content" dramatically increases their ability to persuade others. We further consider more complex transmission persuasion situations and reveal complex dynamics of influence propagation and decay within multi-hop persuasion between multiple agent networks. This research provides systematic evidence linking a model's internal processing architecture to its external persuasive behavior, offering a novel explanation for the susceptibility of advanced models and highlighting critical implications for the safety, robustness, and design of future MAS. 

---
# CLAUSE: Agentic Neuro-Symbolic Knowledge Graph Reasoning via Dynamic Learnable Context Engineering 

**Authors**: Yang Zhao, Chengxiao Dai, Wei Zhuo, Yue Xiu, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2509.21035)  

**Abstract**: Knowledge graphs provide structured context for multi-hop question answering, but deployed systems must balance answer accuracy with strict latency and cost targets while preserving provenance. Static k-hop expansions and "think-longer" prompting often over-retrieve, inflate context, and yield unpredictable runtime. We introduce CLAUSE, an agentic three-agent neuro-symbolic framework that treats context construction as a sequential decision process over knowledge graphs, deciding what to expand, which paths to follow or backtrack, what evidence to keep, and when to stop. Latency (interaction steps) and prompt cost (selected tokens) are exposed as user-specified budgets or prices, allowing per-query adaptation to trade-offs among accuracy, latency, and cost without retraining. CLAUSE employs the proposed Lagrangian-Constrained Multi-Agent Proximal Policy Optimization (LC-MAPPO) algorithm to coordinate three agents: Subgraph Architect, Path Navigator, and Context Curator, so that subgraph construction, reasoning-path discovery, and evidence selection are jointly optimized under per-query resource budgets on edge edits, interaction steps, and selected tokens. Across HotpotQA, MetaQA, and FactKG, CLAUSE yields higher EM@1 while reducing subgraph growth and end-to-end latency at equal or lower token budgets. On MetaQA-2-hop, relative to the strongest RAG baseline (GraphRAG), CLAUSE achieves +39.3 EM@1 with 18.6% lower latency and 40.9% lower edge growth. The resulting contexts are compact, provenance-preserving, and deliver predictable performance under deployment constraints. 

---
# DELTA-Code: How Does RL Unlock and Transfer New Programming Algorithms in LLMs? 

**Authors**: Yiyou Sun, Yuhan Cao, Pohao Huang, Haoyue Bai, Hannaneh Hajishirzi, Nouha Dziri, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.21016)  

**Abstract**: It remains an open question whether LLMs can acquire or generalize genuinely new reasoning strategies, beyond the sharpened skills encoded in their parameters during pre-training or post-training. To attempt to answer this debate, we introduce DELTA-Code--Distributional Evaluation of Learnability and Transferrability in Algorithmic Coding, a controlled benchmark of synthetic coding problem families designed to probe two fundamental aspects: learnability -- can LLMs, through reinforcement learning (RL), solve problem families where pretrained models exhibit failure with large enough attempts (pass@K=0)? --and transferrability -- if learnability happens, can such skills transfer systematically to out-of-distribution (OOD) test sets? Unlike prior public coding datasets, DELTA isolates reasoning skills through templated problem generators and introduces fully OOD problem families that demand novel strategies rather than tool invocation or memorized patterns. Our experiments reveal a striking grokking phase transition: after an extended period with near-zero reward, RL-trained models abruptly climb to near-perfect accuracy. To enable learnability on previously unsolvable problem families, we explore key training ingredients such as staged warm-up with dense rewards, experience replay, curriculum training, and verification-in-the-loop. Beyond learnability, we use DELTA to evaluate transferability or generalization along exploratory, compositional, and transformative axes, as well as cross-family transfer. Results show solid gains within families and for recomposed skills, but persistent weaknesses in transformative cases. DELTA thus offers a clean testbed for probing the limits of RL-driven reasoning and for understanding how models can move beyond existing priors to acquire new algorithmic skills. 

---
# Mechanism of Task-oriented Information Removal in In-context Learning 

**Authors**: Hakaze Cho, Haolin Yang, Gouki Minegishi, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2509.21012)  

**Abstract**: In-context Learning (ICL) is an emerging few-shot learning paradigm based on modern Language Models (LMs), yet its inner mechanism remains unclear. In this paper, we investigate the mechanism through a novel perspective of information removal. Specifically, we demonstrate that in the zero-shot scenario, LMs encode queries into non-selective representations in hidden states containing information for all possible tasks, leading to arbitrary outputs without focusing on the intended task, resulting in near-zero accuracy. Meanwhile, we find that selectively removing specific information from hidden states by a low-rank filter effectively steers LMs toward the intended task. Building on these findings, by measuring the hidden states on carefully designed metrics, we observe that few-shot ICL effectively simulates such task-oriented information removal processes, selectively removing the redundant information from entangled non-selective representations, and improving the output based on the demonstrations, which constitutes a key mechanism underlying ICL. Moreover, we identify essential attention heads inducing the removal operation, termed Denoising Heads, which enables the ablation experiments blocking the information removal operation from the inference, where the ICL accuracy significantly degrades, especially when the correct label is absent from the few-shot demonstrations, confirming both the critical role of the information removal mechanism and denoising heads. 

---
# Binary Autoencoder for Mechanistic Interpretability of Large Language Models 

**Authors**: Hakaze Cho, Haolin Yang, Brian M. Kurkoski, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2509.20997)  

**Abstract**: Existing works are dedicated to untangling atomized numerical components (features) from the hidden states of Large Language Models (LLMs) for interpreting their mechanism. However, they typically rely on autoencoders constrained by some implicit training-time regularization on single training instances (i.e., $L_1$ normalization, top-k function, etc.), without an explicit guarantee of global sparsity among instances, causing a large amount of dense (simultaneously inactive) features, harming the feature sparsity and atomization. In this paper, we propose a novel autoencoder variant that enforces minimal entropy on minibatches of hidden activations, thereby promoting feature independence and sparsity across instances. For efficient entropy calculation, we discretize the hidden activations to 1-bit via a step function and apply gradient estimation to enable backpropagation, so that we term it as Binary Autoencoder (BAE) and empirically demonstrate two major applications: (1) Feature set entropy calculation. Entropy can be reliably estimated on binary hidden activations, which we empirically evaluate and leverage to characterize the inference dynamics of LLMs and In-context Learning. (2) Feature untangling. Similar to typical methods, BAE can extract atomized features from LLM's hidden states. To robustly evaluate such feature extraction capability, we refine traditional feature-interpretation methods to avoid unreliable handling of numerical tokens, and show that BAE avoids dense features while producing the largest number of interpretable ones among baselines, which confirms the effectiveness of BAE serving as a feature extractor. 

---
# CLUE: Conflict-guided Localization for LLM Unlearning Framework 

**Authors**: Hang Chen, Jiaying Zhu, Xinyu Yang, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20977)  

**Abstract**: The LLM unlearning aims to eliminate the influence of undesirable data without affecting causally unrelated information. This process typically involves using a forget set to remove target information, alongside a retain set to maintain non-target capabilities. While recent localization-based methods demonstrate promise in identifying important neurons to be unlearned, they fail to disentangle neurons responsible for forgetting undesirable knowledge or retaining essential skills, often treating them as a single entangled group. As a result, these methods apply uniform interventions, risking catastrophic over-forgetting or incomplete erasure of the target knowledge. To address this, we turn to circuit discovery, a mechanistic interpretability technique, and propose the Conflict-guided Localization for LLM Unlearning framEwork (CLUE). This framework identifies the forget and retain circuit composed of important neurons, and then the circuits are transformed into conjunctive normal forms (CNF). The assignment of each neuron in the CNF satisfiability solution reveals whether it should be forgotten or retained. We then provide targeted fine-tuning strategies for different categories of neurons. Extensive experiments demonstrate that, compared to existing localization methods, CLUE achieves superior forget efficacy and retain utility through precise neural localization. 

---
# On Theoretical Interpretations of Concept-Based In-Context Learning 

**Authors**: Huaze Tang, Tianren Peng, Shao-lun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20882)  

**Abstract**: In-Context Learning (ICL) has emerged as an important new paradigm in natural language processing and large language model (LLM) applications. However, the theoretical understanding of the ICL mechanism remains limited. This paper aims to investigate this issue by studying a particular ICL approach, called concept-based ICL (CB-ICL). In particular, we propose theoretical analyses on applying CB-ICL to ICL tasks, which explains why and when the CB-ICL performs well for predicting query labels in prompts with only a few demonstrations. In addition, the proposed theory quantifies the knowledge that can be leveraged by the LLMs to the prompt tasks, and leads to a similarity measure between the prompt demonstrations and the query input, which provides important insights and guidance for model pre-training and prompt engineering in ICL. Moreover, the impact of the prompt demonstration size and the dimension of the LLM embeddings in ICL are also explored based on the proposed theory. Finally, several real-data experiments are conducted to validate the practical usefulness of CB-ICL and the corresponding theory. 

---
# StyleBench: Evaluating thinking styles in Large Language Models 

**Authors**: Junyu Guo, Shangding Gu, Ming Jin, Costas Spanos, Javad Lavaei  

**Link**: [PDF](https://arxiv.org/pdf/2509.20868)  

**Abstract**: The effectiveness of Large Language Models (LLMs) is heavily influenced by the reasoning strategies, or styles of thought, employed in their prompts. However, the interplay between these reasoning styles, model architecture, and task type remains poorly understood. To address this, we introduce StyleBench, a comprehensive benchmark for systematically evaluating reasoning styles across diverse tasks and models. We assess five representative reasoning styles, including Chain of Thought (CoT), Tree of Thought (ToT), Algorithm of Thought (AoT), Sketch of Thought (SoT), and Chain-of-Draft (CoD) on five reasoning tasks, using 15 open-source models from major families (LLaMA, Qwen, Mistral, Gemma, GPT-OSS, Phi, and DeepSeek) ranging from 270M to 120B parameters. Our large-scale analysis reveals that no single style is universally optimal. We demonstrate that strategy efficacy is highly contingent on both model scale and task type: search-based methods (AoT, ToT) excel in open-ended problems but require large-scale models, while concise styles (SoT, CoD) achieve radical efficiency gains on well-defined tasks. Furthermore, we identify key behavioral patterns: smaller models frequently fail to follow output instructions and default to guessing, while reasoning robustness emerges as a function of scale. Our findings offer a crucial roadmap for selecting optimal reasoning strategies based on specific constraints, we open source the benchmark in this https URL. 

---
# Verification Limits Code LLM Training 

**Authors**: Srishti Gureja, Elena Tommasone, Jingyi He, Sara Hooker, Matthias Gallé, Marzieh Fadaee  

**Link**: [PDF](https://arxiv.org/pdf/2509.20837)  

**Abstract**: Large language models for code generation increasingly rely on synthetic data, where both problem solutions and verification tests are generated by models. While this enables scalable data creation, it introduces a previously unexplored bottleneck: the verification ceiling, in which the quality and diversity of training data are fundamentally constrained by the capabilities of synthetic verifiers. In this work, we systematically study how verification design and strategies influence model performance. We investigate (i) what we verify by analyzing the impact of test complexity and quantity: richer test suites improve code generation capabilities (on average +3 pass@1), while quantity alone yields diminishing returns, (ii) how we verify by exploring relaxed pass thresholds: rigid 100% pass criteria can be overly restrictive. By allowing for relaxed thresholds or incorporating LLM-based soft verification, we can recover valuable training data, leading to a 2-4 point improvement in pass@1 performance. However, this benefit is contingent upon the strength and diversity of the test cases used, and (iii) why verification remains necessary through controlled comparisons of formally correct versus incorrect solutions and human evaluation: retaining diverse correct solutions per problem yields consistent generalization gains. Our results show that Verification as currently practiced is too rigid, filtering out valuable diversity. But it cannot be discarded, only recalibrated. By combining calibrated verification with diverse, challenging problem-solution pairs, we outline a path to break the verification ceiling and unlock stronger code generation models. 

---
# Seeing Through Words, Speaking Through Pixels: Deep Representational Alignment Between Vision and Language Models 

**Authors**: Zoe Wanying He, Sean Trott, Meenakshi Khosla  

**Link**: [PDF](https://arxiv.org/pdf/2509.20751)  

**Abstract**: Recent studies show that deep vision-only and language-only models--trained on disjoint modalities--nonetheless project their inputs into a partially aligned representational space. Yet we still lack a clear picture of where in each network this convergence emerges, what visual or linguistic cues support it, whether it captures human preferences in many-to-many image-text scenarios, and how aggregating exemplars of the same concept affects alignment. Here, we systematically investigate these questions. We find that alignment peaks in mid-to-late layers of both model types, reflecting a shift from modality-specific to conceptually shared representations. This alignment is robust to appearance-only changes but collapses when semantics are altered (e.g., object removal or word-order scrambling), highlighting that the shared code is truly semantic. Moving beyond the one-to-one image-caption paradigm, a forced-choice "Pick-a-Pic" task shows that human preferences for image-caption matches are mirrored in the embedding spaces across all vision-language model pairs. This pattern holds bidirectionally when multiple captions correspond to a single image, demonstrating that models capture fine-grained semantic distinctions akin to human judgments. Surprisingly, averaging embeddings across exemplars amplifies alignment rather than blurring detail. Together, our results demonstrate that unimodal networks converge on a shared semantic code that aligns with human judgments and strengthens with exemplar aggregation. 

---
# Visual Authority and the Rhetoric of Health Misinformation: A Multimodal Analysis of Social Media Videos 

**Authors**: Mohammad Reza Zarei, Barbara Stead-Coyle, Michael Christensen, Sarah Everts, Majid Komeili  

**Link**: [PDF](https://arxiv.org/pdf/2509.20724)  

**Abstract**: Short form video platforms are central sites for health advice, where alternative narratives mix useful, misleading, and harmful content. Rather than adjudicating truth, this study examines how credibility is packaged in nutrition and supplement videos by analyzing the intersection of authority signals, narrative techniques, and monetization. We assemble a cross platform corpus of 152 public videos from TikTok, Instagram, and YouTube and annotate each on 26 features spanning visual authority, presenter attributes, narrative strategies, and engagement cues. A transparent annotation pipeline integrates automatic speech recognition, principled frame selection, and a multimodal model, with human verification on a stratified subsample showing strong agreement. Descriptively, a confident single presenter in studio or home settings dominates, and clinical contexts are rare. Analytically, authority cues such as titles, slides and charts, and certificates frequently occur with persuasive elements including jargon, references, fear or urgency, critiques of mainstream medicine, and conspiracies, and with monetization including sales links and calls to subscribe. References and science like visuals often travel with emotive and oppositional narratives rather than signaling restraint. 

---
# CE-GPPO: Controlling Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning 

**Authors**: Zhenpeng Su, Leiyu Pan, Minxuan Lv, Yuntao Li, Wenping Hu, Fuzheng Zhang, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.20712)  

**Abstract**: Reinforcement learning (RL) has become a powerful paradigm for optimizing large language models (LLMs) to handle complex reasoning tasks. A core challenge in this process lies in managing policy entropy, which reflects the balance between exploration and exploitation during training. Existing methods, such as proximal policy optimization (PPO) and its variants, discard valuable gradient signals from low-probability tokens due to the clipping mechanism. We systematically analyze the entropy dynamics and reveal that these clipped tokens play a critical yet overlooked role in regulating entropy evolution. We propose \textbf{C}ontrolling \textbf{E}ntropy via \textbf{G}radient-\textbf{P}reserving \textbf{P}olicy \textbf{O}ptimization (CE-GPPO), a novel algorithm that reintroduces gradients from clipped tokens in native PPO in a gentle and bounded manner. By controlling the magnitude of gradients from tokens outside the clipping interval, CE-GPPO is able to achieve an exploration-exploitation trade-off. We provide theoretical justification and empirical evidence showing that CE-GPPO effectively mitigates entropy instability. Extensive experiments on mathematical reasoning benchmarks show that CE-GPPO consistently outperforms strong baselines across different model scales. 

---
# Can Federated Learning Safeguard Private Data in LLM Training? Vulnerabilities, Attacks, and Defense Evaluation 

**Authors**: Wenkai Guo, Xuefeng Liu, Haolin Wang, Jianwei Niu, Shaojie Tang, Jing Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2509.20680)  

**Abstract**: Fine-tuning large language models (LLMs) with local data is a widely adopted approach for organizations seeking to adapt LLMs to their specific domains. Given the shared characteristics in data across different organizations, the idea of collaboratively fine-tuning an LLM using data from multiple sources presents an appealing opportunity. However, organizations are often reluctant to share local data, making centralized fine-tuning impractical. Federated learning (FL), a privacy-preserving framework, enables clients to retain local data while sharing only model parameters for collaborative training, offering a potential solution. While fine-tuning LLMs on centralized datasets risks data leakage through next-token prediction, the iterative aggregation process in FL results in a global model that encapsulates generalized knowledge, which some believe protects client privacy. In this paper, however, we present contradictory findings through extensive experiments. We show that attackers can still extract training data from the global model, even using straightforward generation methods, with leakage increasing as the model size grows. Moreover, we introduce an enhanced attack strategy tailored to FL, which tracks global model updates during training to intensify privacy leakage. To mitigate these risks, we evaluate privacy-preserving techniques in FL, including differential privacy, regularization-constrained updates and adopting LLMs with safety alignment. Our results provide valuable insights and practical guidelines for reducing privacy risks when training LLMs with FL. 

---
# Human Semantic Representations of Social Interactions from Moving Shapes 

**Authors**: Yiling Yun, Hongjing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20673)  

**Abstract**: Humans are social creatures who readily recognize various social interactions from simple display of moving shapes. While previous research has often focused on visual features, we examine what semantic representations that humans employ to complement visual features. In Study 1, we directly asked human participants to label the animations based on their impression of moving shapes. We found that human responses were distributed. In Study 2, we measured the representational geometry of 27 social interactions through human similarity judgments and compared it with model predictions based on visual features, labels, and semantic embeddings from animation descriptions. We found that semantic models provided complementary information to visual features in explaining human judgments. Among the semantic models, verb-based embeddings extracted from descriptions account for human similarity judgments the best. These results suggest that social perception in simple displays reflects the semantic structure of social interactions, bridging visual and abstract representations. 

---
# Every Character Counts: From Vulnerability to Defense in Phishing Detection 

**Authors**: Maria Chiper, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20589)  

**Abstract**: Phishing attacks targeting both organizations and individuals are becoming an increasingly significant threat as technology advances. Current automatic detection methods often lack explainability and robustness in detecting new phishing attacks. In this work, we investigate the effectiveness of character-level deep learning models for phishing detection, which can provide both robustness and interpretability. We evaluate three neural architectures adapted to operate at the character level, namely CharCNN, CharGRU, and CharBiLSTM, on a custom-built email dataset, which combines data from multiple sources. Their performance is analyzed under three scenarios: (i) standard training and testing, (ii) standard training and testing under adversarial attacks, and (iii) training and testing with adversarial examples. Aiming to develop a tool that operates as a browser extension, we test all models under limited computational resources. In this constrained setup, CharGRU proves to be the best-performing model across all scenarios. All models show vulnerability to adversarial attacks, but adversarial training substantially improves their robustness. In addition, by adapting the Gradient-weighted Class Activation Mapping (Grad-CAM) technique to character-level inputs, we are able to visualize which parts of each email influence the decision of each model. Our open-source code and data is released at this https URL. 

---
# Perspectra: Choosing Your Experts Enhances Critical Thinking in Multi-Agent Research Ideation 

**Authors**: Yiren Liu, Viraj Shah, Sangho Suh, Pao Siangliulue, Tal August, Yun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20553)  

**Abstract**: Recent advances in multi-agent systems (MAS) enable tools for information search and ideation by assigning personas to agents. However, how users can effectively control, steer, and critically evaluate collaboration among multiple domain-expert agents remains underexplored. We present Perspectra, an interactive MAS that visualizes and structures deliberation among LLM agents via a forum-style interface, supporting @-mention to invite targeted agents, threading for parallel exploration, with a real-time mind map for visualizing arguments and rationales. In a within-subjects study with 18 participants, we compared Perspectra to a group-chat baseline as they developed research proposals. Our findings show that Perspectra significantly increased the frequency and depth of critical-thinking behaviors, elicited more interdisciplinary replies, and led to more frequent proposal revisions than the group chat condition. We discuss implications for designing multi-agent tools that scaffold critical thinking by supporting user control over multi-agent adversarial discourse. 

---
# InsightGUIDE: An Opinionated AI Assistant for Guided Critical Reading of Scientific Literature 

**Authors**: Paris Koloveas, Serafeim Chatzopoulos, Thanasis Vergoulis, Christos Tryfonopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2509.20493)  

**Abstract**: The proliferation of scientific literature presents an increasingly significant challenge for researchers. While Large Language Models (LLMs) offer promise, existing tools often provide verbose summaries that risk replacing, rather than assisting, the reading of the source material. This paper introduces InsightGUIDE, a novel AI-powered tool designed to function as a reading assistant, not a replacement. Our system provides concise, structured insights that act as a "map" to a paper's key elements by embedding an expert's reading methodology directly into its core AI logic. We present the system's architecture, its prompt-driven methodology, and a qualitative case study comparing its output to a general-purpose LLM. The results demonstrate that InsightGUIDE produces more structured and actionable guidance, serving as a more effective tool for the modern researcher. 

---
# RadAgents: Multimodal Agentic Reasoning for Chest X-ray Interpretation with Radiologist-like Workflows 

**Authors**: Kai Zhang, Corey D Barrett, Jangwon Kim, Lichao Sun, Tara Taghavi, Krishnaram Kenthapadi  

**Link**: [PDF](https://arxiv.org/pdf/2509.20490)  

**Abstract**: Agentic systems offer a potential path to solve complex clinical tasks through collaboration among specialized agents, augmented by tool use and external knowledge bases. Nevertheless, for chest X-ray (CXR) interpretation, prevailing methods remain limited: (i) reasoning is frequently neither clinically interpretable nor aligned with guidelines, reflecting mere aggregation of tool outputs; (ii) multimodal evidence is insufficiently fused, yielding text-only rationales that are not visually grounded; and (iii) systems rarely detect or resolve cross-tool inconsistencies and provide no principled verification mechanisms. To bridge the above gaps, we present RadAgents, a multi-agent framework for CXR interpretation that couples clinical priors with task-aware multimodal reasoning. In addition, we integrate grounding and multimodal retrieval-augmentation to verify and resolve context conflicts, resulting in outputs that are more reliable, transparent, and consistent with clinical practice. 

---
# Blueprints of Trust: AI System Cards for End to End Transparency and Governance 

**Authors**: Huzaifa Sidhpurwala, Emily Fox, Garth Mollett, Florencio Cano Gabarda, Roman Zhukov  

**Link**: [PDF](https://arxiv.org/pdf/2509.20394)  

**Abstract**: This paper introduces the Hazard-Aware System Card (HASC), a novel framework designed to enhance transparency and accountability in the development and deployment of AI systems. The HASC builds upon existing model card and system card concepts by integrating a comprehensive, dynamic record of an AI system's security and safety posture. The framework proposes a standardized system of identifiers, including a novel AI Safety Hazard (ASH) ID, to complement existing security identifiers like CVEs, allowing for clear and consistent communication of fixed flaws. By providing a single, accessible source of truth, the HASC empowers developers and stakeholders to make more informed decisions about AI system safety throughout its lifecycle. Ultimately, we also compare our proposed AI system cards with the ISO/IEC 42001:2023 standard and discuss how they can be used to complement each other, providing greater transparency and accountability for AI systems. 

---
# Leveraging NTPs for Efficient Hallucination Detection in VLMs 

**Authors**: Ofir Azachi, Kfir Eliyahu, Eyal El Ani, Rom Himelstein, Roi Reichart, Yuval Pinter, Nitay Calderon  

**Link**: [PDF](https://arxiv.org/pdf/2509.20379)  

**Abstract**: Hallucinations of vision-language models (VLMs), which are misalignments between visual content and generated text, undermine the reliability of VLMs. One common approach for detecting them employs the same VLM, or a different one, to assess generated outputs. This process is computationally intensive and increases model latency. In this paper, we explore an efficient on-the-fly method for hallucination detection by training traditional ML models over signals based on the VLM's next-token probabilities (NTPs). NTPs provide a direct quantification of model uncertainty. We hypothesize that high uncertainty (i.e., a low NTP value) is strongly associated with hallucinations. To test this, we introduce a dataset of 1,400 human-annotated statements derived from VLM-generated content, each labeled as hallucinated or not, and use it to test our NTP-based lightweight method. Our results demonstrate that NTP-based features are valuable predictors of hallucinations, enabling fast and simple ML models to achieve performance comparable to that of strong VLMs. Furthermore, augmenting these NTPs with linguistic NTPs, computed by feeding only the generated text back into the VLM, enhances hallucination detection performance. Finally, integrating hallucination prediction scores from VLMs into the NTP-based models led to better performance than using either VLMs or NTPs alone. We hope this study paves the way for simple, lightweight solutions that enhance the reliability of VLMs. 

---
# CARINOX: Inference-time Scaling with Category-Aware Reward-based Initial Noise Optimization and Exploration 

**Authors**: Seyed Amir Kasaei, Ali Aghayari, Arash Marioriyad, Niki Sepasian, Shayan Baghayi Nejad, MohammadAmin Fazli, Mahdieh Soleymani Baghshah, Mohammad Hossein Rohban  

**Link**: [PDF](https://arxiv.org/pdf/2509.17458)  

**Abstract**: Text-to-image diffusion models, such as Stable Diffusion, can produce high-quality and diverse images but often fail to achieve compositional alignment, particularly when prompts describe complex object relationships, attributes, or spatial arrangements. Recent inference-time approaches address this by optimizing or exploring the initial noise under the guidance of reward functions that score text-image alignment without requiring model fine-tuning. While promising, each strategy has intrinsic limitations when used alone: optimization can stall due to poor initialization or unfavorable search trajectories, whereas exploration may require a prohibitively large number of samples to locate a satisfactory output. Our analysis further shows that neither single reward metrics nor ad-hoc combinations reliably capture all aspects of compositionality, leading to weak or inconsistent guidance. To overcome these challenges, we present Category-Aware Reward-based Initial Noise Optimization and Exploration (CARINOX), a unified framework that combines noise optimization and exploration with a principled reward selection procedure grounded in correlation with human judgments. Evaluations on two complementary benchmarks covering diverse compositional challenges show that CARINOX raises average alignment scores by +16% on T2I-CompBench++ and +11% on the HRS benchmark, consistently outperforming state-of-the-art optimization and exploration-based methods across all major categories, while preserving image quality and diversity. The project page is available at this https URL{this URL}. 

---
# Intercept Cancer: Cancer Pre-Screening with Large Scale Healthcare Foundation Models 

**Authors**: Liwen Sun, Hao-Ren Yao, Gary Gao, Ophir Frieder, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.00209)  

**Abstract**: Cancer screening, leading to early detection, saves lives. Unfortunately, existing screening techniques require expensive and intrusive medical procedures, not globally available, resulting in too many lost would-be-saved lives. We present CATCH-FM, CATch Cancer early with Healthcare Foundation Models, a cancer pre-screening methodology that identifies high-risk patients for further screening solely based on their historical medical records. With millions of electronic healthcare records (EHR), we establish the scaling law of EHR foundation models pretrained on medical code sequences, pretrain compute-optimal foundation models of up to 2.4 billion parameters, and finetune them on clinician-curated cancer risk prediction cohorts. In our retrospective evaluation comprising of thirty thousand patients, CATCH-FM achieved strong efficacy (60% sensitivity) with low risk (99% specificity and Negative Predictive Value), outperforming feature-based tree models as well as general and medical large language models by large margins. Despite significant demographic, healthcare system, and EHR coding differences, CATCH-FM achieves state-of-the-art pancreatic cancer risk prediction on the EHRSHOT few-shot leaderboard, outperforming EHR foundation models pretrained using on-site patient data. Our analysis demonstrates the robustness of CATCH-FM in various patient distributions, the benefits of operating in the ICD code space, and its ability to capture non-trivial cancer risk factors. Our code will be open-sourced. 

---
