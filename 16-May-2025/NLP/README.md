# Beyond 'Aha!': Toward Systematic Meta-Abilities Alignment in Large Reasoning Models 

**Authors**: Zhiyuan Hu, Yibo Wang, Hanze Dong, Yuhui Xu, Amrita Saha, Caiming Xiong, Bryan Hooi, Junnan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10554)  

**Abstract**: Large reasoning models (LRMs) already possess a latent capacity for long chain-of-thought reasoning. Prior work has shown that outcome-based reinforcement learning (RL) can incidentally elicit advanced reasoning behaviors such as self-correction, backtracking, and verification phenomena often referred to as the model's "aha moment". However, the timing and consistency of these emergent behaviors remain unpredictable and uncontrollable, limiting the scalability and reliability of LRMs' reasoning capabilities. To address these limitations, we move beyond reliance on prompts and coincidental "aha moments". Instead, we explicitly align models with three meta-abilities: deduction, induction, and abduction, using automatically generated, self-verifiable tasks. Our three stage-pipeline individual alignment, parameter-space merging, and domain-specific reinforcement learning, boosting performance by over 10\% relative to instruction-tuned baselines. Furthermore, domain-specific RL from the aligned checkpoint yields an additional 2\% average gain in the performance ceiling across math, coding, and science benchmarks, demonstrating that explicit meta-ability alignment offers a scalable and dependable foundation for reasoning. Code is available at: this https URL 

---
# WorldPM: Scaling Human Preference Modeling 

**Authors**: Binghai Wang, Runji Lin, Keming Lu, Le Yu, Zhenru Zhang, Fei Huang, Chujie Zheng, Kai Dang, Yang Fan, Xingzhang Ren, An Yang, Binyuan Hui, Dayiheng Liu, Tao Gui, Qi Zhang, Xuanjing Huang, Yu-Gang Jiang, Bowen Yu, Jingren Zhou, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.10527)  

**Abstract**: Motivated by scaling laws in language modeling that demonstrate how test loss scales as a power law with model and dataset sizes, we find that similar laws exist in preference modeling. We propose World Preference Modeling$ (WorldPM) to emphasize this scaling potential, where World Preference embodies a unified representation of human preferences. In this paper, we collect preference data from public forums covering diverse user communities, and conduct extensive training using 15M-scale data across models ranging from 1.5B to 72B parameters. We observe distinct patterns across different evaluation metrics: (1) Adversarial metrics (ability to identify deceptive features) consistently scale up with increased training data and base model size; (2) Objective metrics (objective knowledge with well-defined answers) show emergent behavior in larger language models, highlighting WorldPM's scalability potential; (3) Subjective metrics (subjective preferences from a limited number of humans or AI) do not demonstrate scaling trends. Further experiments validate the effectiveness of WorldPM as a foundation for preference fine-tuning. Through evaluations on 7 benchmarks with 20 subtasks, we find that WorldPM broadly improves the generalization performance across human preference datasets of varying sizes (7K, 100K and 800K samples), with performance gains exceeding 5% on many key subtasks. Integrating WorldPM into our internal RLHF pipeline, we observe significant improvements on both in-house and public evaluation sets, with notable gains of 4% to 8% in our in-house evaluations. 

---
# Multi-Token Prediction Needs Registers 

**Authors**: Anastasios Gerontopoulos, Spyros Gidaris, Nikos Komodakis  

**Link**: [PDF](https://arxiv.org/pdf/2505.10518)  

**Abstract**: Multi-token prediction has emerged as a promising objective for improving language model pretraining, but its benefits have not consistently generalized to other settings such as fine-tuning. In this paper, we propose MuToR, a simple and effective approach to multi-token prediction that interleaves learnable register tokens into the input sequence, each tasked with predicting future targets. Compared to existing methods, MuToR offers several key advantages: it introduces only a negligible number of additional parameters, requires no architectural changes--ensuring compatibility with off-the-shelf pretrained language models--and remains aligned with the next-token pretraining objective, making it especially well-suited for supervised fine-tuning. Moreover, it naturally supports scalable prediction horizons. We demonstrate the effectiveness and versatility of MuToR across a range of use cases, including supervised fine-tuning, parameter-efficient fine-tuning (PEFT), and pretraining, on challenging generative tasks in both language and vision domains. Our code will be available at: this https URL. 

---
# The Devil Is in the Word Alignment Details: On Translation-Based Cross-Lingual Transfer for Token Classification Tasks 

**Authors**: Benedikt Ebing, Goran Glavaš  

**Link**: [PDF](https://arxiv.org/pdf/2505.10507)  

**Abstract**: Translation-based strategies for cross-lingual transfer XLT such as translate-train -- training on noisy target language data translated from the source language -- and translate-test -- evaluating on noisy source language data translated from the target language -- are competitive XLT baselines. In XLT for token classification tasks, however, these strategies include label projection, the challenging step of mapping the labels from each token in the original sentence to its counterpart(s) in the translation. Although word aligners (WAs) are commonly used for label projection, the low-level design decisions for applying them to translation-based XLT have not been systematically investigated. Moreover, recent marker-based methods, which project labeled spans by inserting tags around them before (or after) translation, claim to outperform WAs in label projection for XLT. In this work, we revisit WAs for label projection, systematically investigating the effects of low-level design decisions on token-level XLT: (i) the algorithm for projecting labels between (multi-)token spans, (ii) filtering strategies to reduce the number of noisily mapped labels, and (iii) the pre-tokenization of the translated sentences. We find that all of these substantially impact translation-based XLT performance and show that, with optimized choices, XLT with WA offers performance at least comparable to that of marker-based methods. We then introduce a new projection strategy that ensembles translate-train and translate-test predictions and demonstrate that it substantially outperforms the marker-based projection. Crucially, we show that our proposed ensembling also reduces sensitivity to low-level WA design choices, resulting in more robust XLT for token classification tasks. 

---
# Can You Really Trust Code Copilots? Evaluating Large Language Models from a Code Security Perspective 

**Authors**: Yutao Mou, Xiao Deng, Yuxiao Luo, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.10494)  

**Abstract**: Code security and usability are both essential for various coding assistant applications driven by large language models (LLMs). Current code security benchmarks focus solely on single evaluation task and paradigm, such as code completion and generation, lacking comprehensive assessment across dimensions like secure code generation, vulnerability repair and discrimination. In this paper, we first propose CoV-Eval, a multi-task benchmark covering various tasks such as code completion, vulnerability repair, vulnerability detection and classification, for comprehensive evaluation of LLM code security. Besides, we developed VC-Judge, an improved judgment model that aligns closely with human experts and can review LLM-generated programs for vulnerabilities in a more efficient and reliable way. We conduct a comprehensive evaluation of 20 proprietary and open-source LLMs. Overall, while most LLMs identify vulnerable codes well, they still tend to generate insecure codes and struggle with recognizing specific vulnerability types and performing repairs. Extensive experiments and qualitative analyses reveal key challenges and optimization directions, offering insights for future research in LLM code security. 

---
# CL-RAG: Bridging the Gap in Retrieval-Augmented Generation with Curriculum Learning 

**Authors**: Shaohan Wang, Licheng Zhang, Zheren Fu, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2505.10493)  

**Abstract**: Retrieval-Augmented Generation (RAG) is an effective method to enhance the capabilities of large language models (LLMs). Existing methods focus on optimizing the retriever or generator in the RAG system by directly utilizing the top-k retrieved documents. However, the documents effectiveness are various significantly across user queries, i.e. some documents provide valuable knowledge while others totally lack critical information. It hinders the retriever and generator's adaptation during training. Inspired by human cognitive learning, curriculum learning trains models using samples progressing from easy to difficult, thus enhancing their generalization ability, and we integrate this effective paradigm to the training of the RAG system. In this paper, we propose a multi-stage Curriculum Learning based RAG system training framework, named CL-RAG. We first construct training data with multiple difficulty levels for the retriever and generator separately through sample evolution. Then, we train the model in stages based on the curriculum learning approach, thereby optimizing the overall performance and generalization of the RAG system more effectively. Our CL-RAG framework demonstrates consistent effectiveness across four open-domain QA datasets, achieving performance gains of 2% to 4% over multiple advanced methods. 

---
# Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models 

**Authors**: Zemin Huang, Zhiyang Chen, Zijun Wang, Tiancheng Li, Guo-Jun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10446)  

**Abstract**: We introduce the \emph{Diffusion Chain of Lateral Thought (DCoLT)}, a reasoning framework for diffusion language models. DCoLT treats each intermediate step in the reverse diffusion process as a latent "thinking" action and optimizes the entire reasoning trajectory to maximize the reward on the correctness of the final answer with outcome-based Reinforcement Learning (RL). Unlike traditional Chain-of-Thought (CoT) methods that follow a causal, linear thinking process, DCoLT allows bidirectional, non-linear reasoning with no strict rule on grammatical correctness amid its intermediate steps of thought. We implement DCoLT on two representative Diffusion Language Models (DLMs). First, we choose SEDD as a representative continuous-time discrete diffusion model, where its concrete score derives a probabilistic policy to maximize the RL reward over the entire sequence of intermediate diffusion steps. We further consider the discrete-time masked diffusion language model -- LLaDA, and find that the order to predict and unmask tokens plays an essential role to optimize its RL action resulting from the ranking-based Unmasking Policy Module (UPM) defined by the Plackett-Luce model. Experiments on both math and code generation tasks show that using only public data and 16 H800 GPUs, DCoLT-reinforced DLMs outperform other DLMs trained by SFT or RL or even both. Notably, DCoLT-reinforced LLaDA boosts its reasoning accuracy by +9.8%, +5.7%, +11.4%, +19.5% on GSM8K, MATH, MBPP, and HumanEval. 

---
# Hierarchical Document Refinement for Long-context Retrieval-augmented Generation 

**Authors**: Jiajie Jin, Xiaoxi Li, Guanting Dong, Yuyao Zhang, Yutao Zhu, Yongkang Wu, Zhonghua Li, Qi Ye, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2505.10413)  

**Abstract**: Real-world RAG applications often encounter long-context input scenarios, where redundant information and noise results in higher inference costs and reduced performance. To address these challenges, we propose LongRefiner, an efficient plug-and-play refiner that leverages the inherent structural characteristics of long documents. LongRefiner employs dual-level query analysis, hierarchical document structuring, and adaptive refinement through multi-task learning on a single foundation model. Experiments on seven QA datasets demonstrate that LongRefiner achieves competitive performance in various scenarios while using 10x fewer computational costs and latency compared to the best baseline. Further analysis validates that LongRefiner is scalable, efficient, and effective, providing practical insights for real-world long-text RAG applications. Our code is available at this https URL. 

---
# Are LLM-generated plain language summaries truly understandable? A large-scale crowdsourced evaluation 

**Authors**: Yue Guo, Jae Ho Sohn, Gondy Leroy, Trevor Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2505.10409)  

**Abstract**: Plain language summaries (PLSs) are essential for facilitating effective communication between clinicians and patients by making complex medical information easier for laypeople to understand and act upon. Large language models (LLMs) have recently shown promise in automating PLS generation, but their effectiveness in supporting health information comprehension remains unclear. Prior evaluations have generally relied on automated scores that do not measure understandability directly, or subjective Likert-scale ratings from convenience samples with limited generalizability. To address these gaps, we conducted a large-scale crowdsourced evaluation of LLM-generated PLSs using Amazon Mechanical Turk with 150 participants. We assessed PLS quality through subjective Likert-scale ratings focusing on simplicity, informativeness, coherence, and faithfulness; and objective multiple-choice comprehension and recall measures of reader understanding. Additionally, we examined the alignment between 10 automated evaluation metrics and human judgments. Our findings indicate that while LLMs can generate PLSs that appear indistinguishable from human-written ones in subjective evaluations, human-written PLSs lead to significantly better comprehension. Furthermore, automated evaluation metrics fail to reflect human judgment, calling into question their suitability for evaluating PLSs. This is the first study to systematically evaluate LLM-generated PLSs based on both reader preferences and comprehension outcomes. Our findings highlight the need for evaluation frameworks that move beyond surface-level quality and for generation methods that explicitly optimize for layperson comprehension. 

---
# Rethinking Repetition Problems of LLMs in Code Generation 

**Authors**: Yihong Dong, Yuchen Liu, Xue Jiang, Zhi Jin, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10402)  

**Abstract**: With the advent of neural language models, the performance of code generation has been significantly boosted. However, the problem of repetitions during the generation process continues to linger. Previous work has primarily focused on content repetition, which is merely a fraction of the broader repetition problem in code generation. A more prevalent and challenging problem is structural repetition. In structural repetition, the repeated code appears in various patterns but possesses a fixed structure, which can be inherently reflected in grammar. In this paper, we formally define structural repetition and propose an efficient decoding approach called RPG, which stands for Repetition Penalization based on Grammar, to alleviate the repetition problems in code generation for LLMs. Specifically, RPG first leverages grammar rules to identify repetition problems during code generation, and then strategically decays the likelihood of critical tokens that contribute to repetitions, thereby mitigating them in code generation. To facilitate this study, we construct a new dataset CodeRepetEval to comprehensively evaluate approaches for mitigating the repetition problems in code generation. Extensive experimental results demonstrate that RPG substantially outperforms the best-performing baselines on CodeRepetEval dataset as well as HumanEval and MBPP benchmarks, effectively reducing repetitions and enhancing the quality of generated code. 

---
# Multi-domain Multilingual Sentiment Analysis in Industry: Predicting Aspect-based Opinion Quadruples 

**Authors**: Benjamin White, Anastasia Shimorina  

**Link**: [PDF](https://arxiv.org/pdf/2505.10389)  

**Abstract**: This paper explores the design of an aspect-based sentiment analysis system using large language models (LLMs) for real-world use. We focus on quadruple opinion extraction -- identifying aspect categories, sentiment polarity, targets, and opinion expressions from text data across different domains and languages. Using internal datasets, we investigate whether a single fine-tuned model can effectively handle multiple domain-specific taxonomies simultaneously. We demonstrate that a combined multi-domain model achieves performance comparable to specialized single-domain models while reducing operational complexity. We also share lessons learned for handling non-extractive predictions and evaluating various failure modes when developing LLM-based systems for structured prediction tasks. 

---
# Coherent Language Reconstruction from Brain Recordings with Flexible Multi-Modal Input Stimuli 

**Authors**: Chunyu Ye, Shaonan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10356)  

**Abstract**: Decoding thoughts from brain activity offers valuable insights into human cognition and enables promising applications in brain-computer interaction. While prior studies have explored language reconstruction from fMRI data, they are typically limited to single-modality inputs such as images or audio. In contrast, human thought is inherently multimodal. To bridge this gap, we propose a unified and flexible framework for reconstructing coherent language from brain recordings elicited by diverse input modalities-visual, auditory, and textual. Our approach leverages visual-language models (VLMs), using modality-specific experts to jointly interpret information across modalities. Experiments demonstrate that our method achieves performance comparable to state-of-the-art systems while remaining adaptable and extensible. This work advances toward more ecologically valid and generalizable mind decoding. 

---
# LDIR: Low-Dimensional Dense and Interpretable Text Embeddings with Relative Representations 

**Authors**: Yile Wang, Zhanyu Shen, Hui Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10354)  

**Abstract**: Semantic text representation is a fundamental task in the field of natural language processing. Existing text embedding (e.g., SimCSE and LLM2Vec) have demonstrated excellent performance, but the values of each dimension are difficult to trace and interpret. Bag-of-words, as classic sparse interpretable embeddings, suffers from poor performance. Recently, Benara et al. (2024) propose interpretable text embeddings using large language models, which forms "0/1" embeddings based on responses to a series of questions. These interpretable text embeddings are typically high-dimensional (larger than 10,000). In this work, we propose Low-dimensional (lower than 500) Dense and Interpretable text embeddings with Relative representations (LDIR). The numerical values of its dimensions indicate semantic relatedness to different anchor texts through farthest point sampling, offering both semantic representation as well as a certain level of traceability and interpretability. We validate LDIR on multiple semantic textual similarity, retrieval, and clustering tasks. Extensive experimental results show that LDIR performs close to the black-box baseline models and outperforms the interpretable embeddings baselines with much fewer dimensions. Code is available at this https URL. 

---
# J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning 

**Authors**: Chenxi Whitehouse, Tianlu Wang, Ping Yu, Xian Li, Jason Weston, Ilia Kulikov, Swarnadeep Saha  

**Link**: [PDF](https://arxiv.org/pdf/2505.10320)  

**Abstract**: The progress of AI is bottlenecked by the quality of evaluation, and powerful LLM-as-a-Judge models have proved to be a core solution. Improved judgment ability is enabled by stronger chain-of-thought reasoning, motivating the need to find the best recipes for training such models to think. In this work we introduce J1, a reinforcement learning approach to training such models. Our method converts both verifiable and non-verifiable prompts to judgment tasks with verifiable rewards that incentivize thinking and mitigate judgment bias. In particular, our approach outperforms all other existing 8B or 70B models when trained at those sizes, including models distilled from DeepSeek-R1. J1 also outperforms o1-mini, and even R1 on some benchmarks, despite training a smaller model. We provide analysis and ablations comparing Pairwise-J1 vs Pointwise-J1 models, offline vs online training recipes, reward strategies, seed prompts, and variations in thought length and content. We find that our models make better judgments by learning to outline evaluation criteria, comparing against self-generated reference answers, and re-evaluating the correctness of model responses. 

---
# From Questions to Clinical Recommendations: Large Language Models Driving Evidence-Based Clinical Decision Making 

**Authors**: Dubai Li, Nan Jiang, Kangping Huang, Ruiqi Tu, Shuyu Ouyang, Huayu Yu, Lin Qiao, Chen Yu, Tianshu Zhou, Danyang Tong, Qian Wang, Mengtao Li, Xiaofeng Zeng, Yu Tian, Xinping Tian, Jingsong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10282)  

**Abstract**: Clinical evidence, derived from rigorous research and data analysis, provides healthcare professionals with reliable scientific foundations for informed decision-making. Integrating clinical evidence into real-time practice is challenging due to the enormous workload, complex professional processes, and time constraints. This highlights the need for tools that automate evidence synthesis to support more efficient and accurate decision making in clinical settings. This study introduces Quicker, an evidence-based clinical decision support system powered by large language models (LLMs), designed to automate evidence synthesis and generate clinical recommendations modeled after standard clinical guideline development processes. Quicker implements a fully automated chain that covers all phases, from questions to clinical recommendations, and further enables customized decision-making through integrated tools and interactive user interfaces. To evaluate Quicker's capabilities, we developed the Q2CRBench-3 benchmark dataset, based on clinical guideline development records for three different diseases. Experimental results highlighted Quicker's strong performance, with fine-grained question decomposition tailored to user preferences, retrieval sensitivities comparable to human experts, and literature screening performance approaching comprehensive inclusion of relevant studies. In addition, Quicker-assisted evidence assessment effectively supported human reviewers, while Quicker's recommendations were more comprehensive and logically coherent than those of clinicians. In system-level testing, collaboration between a single reviewer and Quicker reduced the time required for recommendation development to 20-40 minutes. In general, our findings affirm the potential of Quicker to help physicians make quicker and more reliable evidence-based clinical decisions. 

---
# The Evolving Landscape of Generative Large Language Models and Traditional Natural Language Processing in Medicine 

**Authors**: Rui Yang, Huitao Li, Matthew Yu Heng Wong, Yuhe Ke, Xin Li, Kunyu Yu, Jingchi Liao, Jonathan Chong Kai Liew, Sabarinath Vinod Nair, Jasmine Chiat Ling Ong, Irene Li, Douglas Teodoro, Chuan Hong, Daniel Shu Wei Ting, Nan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10261)  

**Abstract**: Natural language processing (NLP) has been traditionally applied to medicine, and generative large language models (LLMs) have become prominent recently. However, the differences between them across different medical tasks remain underexplored. We analyzed 19,123 studies, finding that generative LLMs demonstrate advantages in open-ended tasks, while traditional NLP dominates in information extraction and analysis tasks. As these technologies advance, ethical use of them is essential to ensure their potential in medical applications. 

---
# Comparing LLM Text Annotation Skills: A Study on Human Rights Violations in Social Media Data 

**Authors**: Poli Apollinaire Nemkova, Solomon Ubani, Mark V. Albert  

**Link**: [PDF](https://arxiv.org/pdf/2505.10260)  

**Abstract**: In the era of increasingly sophisticated natural language processing (NLP) systems, large language models (LLMs) have demonstrated remarkable potential for diverse applications, including tasks requiring nuanced textual understanding and contextual reasoning. This study investigates the capabilities of multiple state-of-the-art LLMs - GPT-3.5, GPT-4, LLAMA3, Mistral 7B, and Claude-2 - for zero-shot and few-shot annotation of a complex textual dataset comprising social media posts in Russian and Ukrainian. Specifically, the focus is on the binary classification task of identifying references to human rights violations within the dataset.
To evaluate the effectiveness of these models, their annotations are compared against a gold standard set of human double-annotated labels across 1000 samples. The analysis includes assessing annotation performance under different prompting conditions, with prompts provided in both English and Russian. Additionally, the study explores the unique patterns of errors and disagreements exhibited by each model, offering insights into their strengths, limitations, and cross-linguistic adaptability.
By juxtaposing LLM outputs with human annotations, this research contributes to understanding the reliability and applicability of LLMs for sensitive, domain-specific tasks in multilingual contexts. It also sheds light on how language models handle inherently subjective and context-dependent judgments, a critical consideration for their deployment in real-world scenarios. 

---
# RAIDEN-R1: Improving Role-awareness of LLMs via GRPO with Verifiable Reward 

**Authors**: Zongsheng Wang, Kaili Sun, Bowen Wu, Qun Yu, Ying Li, Baoxun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10218)  

**Abstract**: Role-playing conversational agents (RPCAs) face persistent challenges in maintaining role consistency. To address this, we propose RAIDEN-R1, a novel reinforcement learning framework that integrates Verifiable Role-Awareness Reward (VRAR). The method introduces both singular and multi-term mining strategies to generate quantifiable rewards by assessing role-specific keys. Additionally, we construct a high-quality, role-aware Chain-of-Thought dataset through multi-LLM collaboration, and implement experiments to enhance reasoning coherence. Experiments on the RAIDEN benchmark demonstrate RAIDEN-R1's superiority: our 14B-GRPO model achieves 88.04% and 88.65% accuracy on Script-Based Knowledge and Conversation Memory metrics, respectively, outperforming baseline models while maintaining robustness. Case analyses further reveal the model's enhanced ability to resolve conflicting contextual cues and sustain first-person narrative consistency. This work bridges the non-quantifiability gap in RPCA training and provides insights into role-aware reasoning patterns, advancing the development of RPCAs. 

---
# VQ-Logits: Compressing the Output Bottleneck of Large Language Models via Vector Quantized Logits 

**Authors**: Jintian Shao, Hongyi Huang, Jiayi Wu, YiMing Cheng, ZhiYu Wu, You Shan, MingKai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.10202)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success but face significant computational and memory challenges, particularly due to their extensive output vocabularies. The final linear projection layer, mapping hidden states to vocabulary-sized logits, often constitutes a substantial portion of the model's parameters and computational cost during inference. Existing methods like adaptive softmax or hierarchical softmax introduce structural complexities. In this paper, we propose VQ-Logits, a novel approach that leverages Vector Quantization (VQ) to drastically reduce the parameter count and computational load of the LLM output layer. VQ-Logits replaces the large V * dmodel output embedding matrix with a small, shared codebook of K embedding vectors (K << V ). Each token in the vocabulary is mapped to one of these K codebook vectors. The LLM predicts logits over this compact codebook, which are then efficiently "scattered" to the full vocabulary space using the learned or preassigned mapping. We demonstrate through extensive experiments on standard language modeling benchmarks (e.g., WikiText-103, C4) that VQ-Logits can achieve up to 99% parameter reduction in the output layer and 6x speedup in logit computation, with only a marginal 4% increase in perplexity compared to full softmax baselines. We further provide detailed ablation studies on codebook size, initialization, and learning strategies, showcasing the robustness and effectiveness of our approach. 

---
# The CoT Encyclopedia: Analyzing, Predicting, and Controlling how a Reasoning Model will Think 

**Authors**: Seongyun Lee, Seungone Kim, Minju Seo, Yongrae Jo, Dongyoung Go, Hyeonbin Hwang, Jinho Park, Xiang Yue, Sean Welleck, Graham Neubig, Moontae Lee, Minjoon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2505.10185)  

**Abstract**: Long chain-of-thought (CoT) is an essential ingredient in effective usage of modern large language models, but our understanding of the reasoning strategies underlying these capabilities remains limited. While some prior works have attempted to categorize CoTs using predefined strategy types, such approaches are constrained by human intuition and fail to capture the full diversity of model behaviors. In this work, we introduce the CoT Encyclopedia, a bottom-up framework for analyzing and steering model reasoning. Our method automatically extracts diverse reasoning criteria from model-generated CoTs, embeds them into a semantic space, clusters them into representative categories, and derives contrastive rubrics to interpret reasoning behavior. Human evaluations show that this framework produces more interpretable and comprehensive analyses than existing methods. Moreover, we demonstrate that this understanding enables performance gains: we can predict which strategy a model is likely to use and guide it toward more effective alternatives. Finally, we provide practical insights, such as that training data format (e.g., free-form vs. multiple-choice) has a far greater impact on reasoning behavior than data domain, underscoring the importance of format-aware model design. 

---
# Mining Hidden Thoughts from Texts: Evaluating Continual Pretraining with Synthetic Data for LLM Reasoning 

**Authors**: Yoichi Ishibashi, Taro Yano, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2505.10182)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant improvements in reasoning capabilities through supervised fine-tuning and reinforcement learning. However, when training reasoning models, these approaches are primarily applicable to specific domains such as mathematics and programming, which imposes fundamental constraints on the breadth and scalability of training data. In contrast, continual pretraining (CPT) offers the advantage of not requiring task-specific signals. Nevertheless, how to effectively synthesize training data for reasoning and how such data affect a wide range of domains remain largely unexplored. This study provides a detailed evaluation of Reasoning CPT, a form of CPT that uses synthetic data to reconstruct the hidden thought processes underlying texts, based on the premise that texts are the result of the author's thinking process. Specifically, we apply Reasoning CPT to Gemma2-9B using synthetic data with hidden thoughts derived from STEM and Law corpora, and compare it to standard CPT on the MMLU benchmark. Our analysis reveals that Reasoning CPT consistently improves performance across all evaluated domains. Notably, reasoning skills acquired in one domain transfer effectively to others; the performance gap with conventional methods widens as problem difficulty increases, with gains of up to 8 points on the most challenging problems. Furthermore, models trained with hidden thoughts learn to adjust the depth of their reasoning according to problem difficulty. 

---
# GE-Chat: A Graph Enhanced RAG Framework for Evidential Response Generation of LLMs 

**Authors**: Longchao Da, Parth Mitesh Shah, Kuan-Ru Liou, Jiaxing Zhang, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.10143)  

**Abstract**: Large Language Models are now key assistants in human decision-making processes. However, a common note always seems to follow: "LLMs can make mistakes. Be careful with important info." This points to the reality that not all outputs from LLMs are dependable, and users must evaluate them manually. The challenge deepens as hallucinated responses, often presented with seemingly plausible explanations, create complications and raise trust issues among users. To tackle such issue, this paper proposes GE-Chat, a knowledge Graph enhanced retrieval-augmented generation framework to provide Evidence-based response generation. Specifically, when the user uploads a material document, a knowledge graph will be created, which helps construct a retrieval-augmented agent, enhancing the agent's responses with additional knowledge beyond its training corpus. Then we leverage Chain-of-Thought (CoT) logic generation, n-hop sub-graph searching, and entailment-based sentence generation to realize accurate evidence retrieval. We demonstrate that our method improves the existing models' performance in terms of identifying the exact evidence in a free-form context, providing a reliable way to examine the resources of LLM's conclusion and help with the judgment of the trustworthiness. 

---
# What Does Neuro Mean to Cardio? Investigating the Role of Clinical Specialty Data in Medical LLMs 

**Authors**: Xinlan Yan, Di Wu, Yibin Lei, Christof Monz, Iacer Calixto  

**Link**: [PDF](https://arxiv.org/pdf/2505.10113)  

**Abstract**: In this paper, we introduce S-MedQA, an English medical question-answering (QA) dataset for benchmarking large language models in fine-grained clinical specialties. We use S-MedQA to check the applicability of a popular hypothesis related to knowledge injection in the knowledge-intense scenario of medical QA, and show that: 1) training on data from a speciality does not necessarily lead to best performance on that specialty and 2) regardless of the specialty fine-tuned on, token probabilities of clinically relevant terms for all specialties increase consistently. Thus, we believe improvement gains come mostly from domain shifting (e.g., general to medical) rather than knowledge injection and suggest rethinking the role of fine-tuning data in the medical domain. We release S-MedQA and all code needed to reproduce all our experiments to the research community. 

---
# XRAG: Cross-lingual Retrieval-Augmented Generation 

**Authors**: Wei Liu, Sony Trenous, Leonardo F. R. Ribeiro, Bill Byrne, Felix Hieber  

**Link**: [PDF](https://arxiv.org/pdf/2505.10089)  

**Abstract**: We propose XRAG, a novel benchmark designed to evaluate the generation abilities of LLMs in cross-lingual Retrieval-Augmented Generation (RAG) settings where the user language does not match the retrieval results. XRAG is constructed from recent news articles to ensure that its questions require external knowledge to be answered. It covers the real-world scenarios of monolingual and multilingual retrieval, and provides relevancy annotations for each retrieved document. Our novel dataset construction pipeline results in questions that require complex reasoning, as evidenced by the significant gap between human and LLM performance. Consequently, XRAG serves as a valuable benchmark for studying LLM reasoning abilities, even before considering the additional cross-lingual complexity. Experimental results on five LLMs uncover two previously unreported challenges in cross-lingual RAG: 1) in the monolingual retrieval setting, all evaluated models struggle with response language correctness; 2) in the multilingual retrieval setting, the main challenge lies in reasoning over retrieved information across languages rather than generation of non-English text. 

---
# Designing and Contextualising Probes for African Languages 

**Authors**: Wisdom Aduah, Francois Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2505.10081)  

**Abstract**: Pretrained language models (PLMs) for African languages are continually improving, but the reasons behind these advances remain unclear. This paper presents the first systematic investigation into probing PLMs for linguistic knowledge about African languages. We train layer-wise probes for six typologically diverse African languages to analyse how linguistic features are distributed. We also design control tasks, a way to interpret probe performance, for the MasakhaPOS dataset. We find PLMs adapted for African languages to encode more linguistic information about target languages than massively multilingual PLMs. Our results reaffirm previous findings that token-level syntactic information concentrates in middle-to-last layers, while sentence-level semantic information is distributed across all layers. Through control tasks and probing baselines, we confirm that performance reflects the internal knowledge of PLMs rather than probe memorisation. Our study applies established interpretability techniques to African-language PLMs. In doing so, we highlight the internal mechanisms underlying the success of strategies like active learning and multilingual adaptation. 

---
# Dark LLMs: The Growing Threat of Unaligned AI Models 

**Authors**: Michael Fire, Yitzhak Elbazis, Adi Wasenstein, Lior Rokach  

**Link**: [PDF](https://arxiv.org/pdf/2505.10066)  

**Abstract**: Large Language Models (LLMs) rapidly reshape modern life, advancing fields from healthcare to education and beyond. However, alongside their remarkable capabilities lies a significant threat: the susceptibility of these models to jailbreaking. The fundamental vulnerability of LLMs to jailbreak attacks stems from the very data they learn from. As long as this training data includes unfiltered, problematic, or 'dark' content, the models can inherently learn undesirable patterns or weaknesses that allow users to circumvent their intended safety controls. Our research identifies the growing threat posed by dark LLMs models deliberately designed without ethical guardrails or modified through jailbreak techniques. In our research, we uncovered a universal jailbreak attack that effectively compromises multiple state-of-the-art models, enabling them to answer almost any question and produce harmful outputs upon request. The main idea of our attack was published online over seven months ago. However, many of the tested LLMs were still vulnerable to this attack. Despite our responsible disclosure efforts, responses from major LLM providers were often inadequate, highlighting a concerning gap in industry practices regarding AI safety. As model training becomes more accessible and cheaper, and as open-source LLMs proliferate, the risk of widespread misuse escalates. Without decisive intervention, LLMs may continue democratizing access to dangerous knowledge, posing greater risks than anticipated. 

---
# CAFE: Retrieval Head-based Coarse-to-Fine Information Seeking to Enhance Multi-Document QA Capability 

**Authors**: Han Peng, Jinhao Jiang, Zican Dong, Wayne Xin Zhao, Lei Fang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10063)  

**Abstract**: Advancements in Large Language Models (LLMs) have extended their input context length, yet they still struggle with retrieval and reasoning in long-context inputs. Existing methods propose to utilize the prompt strategy and retrieval head to alleviate this limitation. However, they still face challenges in balancing retrieval precision and recall, impacting their efficacy in answering questions. To address this, we introduce $\textbf{CAFE}$, a two-stage coarse-to-fine method to enhance multi-document question-answering capacities. By gradually eliminating the negative impacts of background and distracting documents, CAFE makes the responses more reliant on the evidence documents. Initially, a coarse-grained filtering method leverages retrieval heads to identify and rank relevant documents. Then, a fine-grained steering method guides attention to the most relevant content. Experiments across benchmarks show CAFE outperforms baselines, achieving up to 22.1% and 13.7% SubEM improvement over SFT and RAG methods on the Mistral model, respectively. 

---
# DIF: A Framework for Benchmarking and Verifying Implicit Bias in LLMs 

**Authors**: Lake Yin, Fan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10013)  

**Abstract**: As Large Language Models (LLMs) have risen in prominence over the past few years, there has been concern over the potential biases in LLMs inherited from the training data. Previous studies have examined how LLMs exhibit implicit bias, such as when response generation changes when different social contexts are introduced. We argue that this implicit bias is not only an ethical, but also a technical issue, as it reveals an inability of LLMs to accommodate extraneous information. However, unlike other measures of LLM intelligence, there are no standard methods to benchmark this specific subset of LLM bias. To bridge this gap, we developed a method for calculating an easily interpretable benchmark, DIF (Demographic Implicit Fairness), by evaluating preexisting LLM logic and math problem datasets with sociodemographic personas. We demonstrate that this method can statistically validate the presence of implicit bias in LLM behavior and find an inverse trend between question answering accuracy and implicit bias, supporting our argument. 

---
# Personalizing Large Language Models using Retrieval Augmented Generation and Knowledge Graph 

**Authors**: Deeksha Prahlad, Chanhee Lee, Dongha Kim, Hokeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.09945)  

**Abstract**: The advent of large language models (LLMs) has allowed numerous applications, including the generation of queried responses, to be leveraged in chatbots and other conversational assistants. Being trained on a plethora of data, LLMs often undergo high levels of over-fitting, resulting in the generation of extra and incorrect data, thus causing hallucinations in output generation. One of the root causes of such problems is the lack of timely, factual, and personalized information fed to the LLM. In this paper, we propose an approach to address these problems by introducing retrieval augmented generation (RAG) using knowledge graphs (KGs) to assist the LLM in personalized response generation tailored to the users. KGs have the advantage of storing continuously updated factual information in a structured way. While our KGs can be used for a variety of frequently updated personal data, such as calendar, contact, and location data, we focus on calendar data in this paper. Our experimental results show that our approach works significantly better in understanding personal information and generating accurate responses compared to the baseline LLMs using personal data as text inputs, with a moderate reduction in response time. 

---
# Rethinking Prompt Optimizers: From Prompt Merits to Optimization 

**Authors**: Zixiao Zhu, Hanzhang Zhou, Zijian Feng, Tianjiao Li, Chua Jia Jim Deryl, Mak Lee Onn, Gee Wah Ng, Kezhi Mao  

**Link**: [PDF](https://arxiv.org/pdf/2505.09930)  

**Abstract**: Prompt optimization (PO) offers a practical alternative to fine-tuning large language models (LLMs), enabling performance improvements without altering model weights. Existing methods typically rely on advanced, large-scale LLMs like GPT-4 to generate optimized prompts. However, due to limited downward compatibility, verbose, instruction-heavy prompts from advanced LLMs can overwhelm lightweight inference models and degrade response quality. In this work, we rethink prompt optimization through the lens of interpretable design. We first identify a set of model-agnostic prompt quality merits and empirically validate their effectiveness in enhancing prompt and response quality. We then introduce MePO, a merit-guided, lightweight, and locally deployable prompt optimizer trained on our preference dataset built from merit-aligned prompts generated by a lightweight LLM. Unlike prior work, MePO avoids online optimization reliance, reduces cost and privacy concerns, and, by learning clear, interpretable merits, generalizes effectively to both large-scale and lightweight inference models. Experiments demonstrate that MePO achieves better results across diverse tasks and model types, offering a scalable and robust solution for real-world deployment. Our model and dataset are available at: this https URL 

---
# From Trade-off to Synergy: A Versatile Symbiotic Watermarking Framework for Large Language Models 

**Authors**: Yidan Wang, Yubing Ren, Yanan Cao, Binxing Fang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09924)  

**Abstract**: The rise of Large Language Models (LLMs) has heightened concerns about the misuse of AI-generated text, making watermarking a promising solution. Mainstream watermarking schemes for LLMs fall into two categories: logits-based and sampling-based. However, current schemes entail trade-offs among robustness, text quality, and security. To mitigate this, we integrate logits-based and sampling-based schemes, harnessing their respective strengths to achieve synergy. In this paper, we propose a versatile symbiotic watermarking framework with three strategies: serial, parallel, and hybrid. The hybrid framework adaptively embeds watermarks using token entropy and semantic entropy, optimizing the balance between detectability, robustness, text quality, and security. Furthermore, we validate our approach through comprehensive experiments on various datasets and models. Experimental results indicate that our method outperforms existing baselines and achieves state-of-the-art (SOTA) performance. We believe this framework provides novel insights into diverse watermarking paradigms. Our code is available at \href{this https URL}{this https URL}. 

---
# Crossing Borders Without Crossing Boundaries: How Sociolinguistic Awareness Can Optimize User Engagement with Localized Spanish AI Models Across Hispanophone Countries 

**Authors**: Martin Capdevila, Esteban Villa Turek, Ellen Karina Chumbe Fernandez, Luis Felipe Polo Galvez, Luis Cadavid, Andrea Marroquin, Rebeca Vargas Quesada, Johanna Crew, Nicole Vallejo Galarraga, Christopher Rodriguez, Diego Gutierrez, Radhi Datla  

**Link**: [PDF](https://arxiv.org/pdf/2505.09902)  

**Abstract**: Large language models are, by definition, based on language. In an effort to underscore the critical need for regional localized models, this paper examines primary differences between variants of written Spanish across Latin America and Spain, with an in-depth sociocultural and linguistic contextualization therein. We argue that these differences effectively constitute significant gaps in the quotidian use of Spanish among dialectal groups by creating sociolinguistic dissonances, to the extent that locale-sensitive AI models would play a pivotal role in bridging these divides. In doing so, this approach informs better and more efficient localization strategies that also serve to more adequately meet inclusivity goals, while securing sustainable active daily user growth in a major low-risk investment geographic area. Therefore, implementing at least the proposed five sub variants of Spanish addresses two lines of action: to foment user trust and reliance on AI language models while also demonstrating a level of cultural, historical, and sociolinguistic awareness that reflects positively on any internationalization strategy. 

---
# Do Large Language Models Know Conflict? Investigating Parametric vs. Non-Parametric Knowledge of LLMs for Conflict Forecasting 

**Authors**: Apollinaire Poli Nemkova, Sarath Chandra Lingareddy, Sagnik Ray Choudhury, Mark V. Albert  

**Link**: [PDF](https://arxiv.org/pdf/2505.09852)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance across natural language tasks, but their ability to forecast violent conflict remains underexplored. We investigate whether LLMs possess meaningful parametric knowledge-encoded in their pretrained weights-to predict conflict escalation and fatalities without external data. This is critical for early warning systems, humanitarian planning, and policy-making. We compare this parametric knowledge with non-parametric capabilities, where LLMs access structured and unstructured context from conflict datasets (e.g., ACLED, GDELT) and recent news reports via Retrieval-Augmented Generation (RAG). Incorporating external information could enhance model performance by providing up-to-date context otherwise missing from pretrained weights. Our two-part evaluation framework spans 2020-2024 across conflict-prone regions in the Horn of Africa and the Middle East. In the parametric setting, LLMs predict conflict trends and fatalities relying only on pretrained knowledge. In the non-parametric setting, models receive summaries of recent conflict events, indicators, and geopolitical developments. We compare predicted conflict trend labels (e.g., Escalate, Stable Conflict, De-escalate, Peace) and fatalities against historical data. Our findings highlight the strengths and limitations of LLMs for conflict forecasting and the benefits of augmenting them with structured external knowledge. 

---
# KRISTEVA: Close Reading as a Novel Task for Benchmarking Interpretive Reasoning 

**Authors**: Peiqi Sui, Juan Diego Rodriguez, Philippe Laban, Dean Murphy, Joseph P. Dexter, Richard Jean So, Samuel Baker, Pramit Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2505.09825)  

**Abstract**: Each year, tens of millions of essays are written and graded in college-level English courses. Students are asked to analyze literary and cultural texts through a process known as close reading, in which they gather textual details to formulate evidence-based arguments. Despite being viewed as a basis for critical thinking and widely adopted as a required element of university coursework, close reading has never been evaluated on large language models (LLMs), and multi-discipline benchmarks like MMLU do not include literature as a subject. To fill this gap, we present KRISTEVA, the first close reading benchmark for evaluating interpretive reasoning, consisting of 1331 multiple-choice questions adapted from classroom data. With KRISTEVA, we propose three progressively more difficult sets of tasks to approximate different elements of the close reading process, which we use to test how well LLMs may seem to understand and reason about literary works: 1) extracting stylistic features, 2) retrieving relevant contextual information from parametric knowledge, and 3) multi-hop reasoning between style and external contexts. Our baseline results find that, while state-of-the-art LLMs possess some college-level close reading competency (accuracy 49.7% - 69.7%), their performances still trail those of experienced human evaluators on 10 out of our 11 tasks. 

---
# Exploring the generalization of LLM truth directions on conversational formats 

**Authors**: Timour Ichmoukhamedov, David Martens  

**Link**: [PDF](https://arxiv.org/pdf/2505.09807)  

**Abstract**: Several recent works argue that LLMs have a universal truth direction where true and false statements are linearly separable in the activation space of the model. It has been demonstrated that linear probes trained on a single hidden state of the model already generalize across a range of topics and might even be used for lie detection in LLM conversations. In this work we explore how this truth direction generalizes between various conversational formats. We find good generalization between short conversations that end on a lie, but poor generalization to longer formats where the lie appears earlier in the input prompt. We propose a solution that significantly improves this type of generalization by adding a fixed key phrase at the end of each conversation. Our results highlight the challenges towards reliable LLM lie detectors that generalize to new settings. 

---
# Automated Detection of Clinical Entities in Lung and Breast Cancer Reports Using NLP Techniques 

**Authors**: J. Moreno-Casanova, J.M. Auñón, A. Mártinez-Pérez, M.E. Pérez-Martínez, M.E. Gas-López  

**Link**: [PDF](https://arxiv.org/pdf/2505.09794)  

**Abstract**: Research projects, including those focused on cancer, rely on the manual extraction of information from clinical reports. This process is time-consuming and prone to errors, limiting the efficiency of data-driven approaches in healthcare. To address these challenges, Natural Language Processing (NLP) offers an alternative for automating the extraction of relevant data from electronic health records (EHRs). In this study, we focus on lung and breast cancer due to their high incidence and the significant impact they have on public health. Early detection and effective data management in both types of cancer are crucial for improving patient outcomes. To enhance the accuracy and efficiency of data extraction, we utilized GMV's NLP tool uQuery, which excels at identifying relevant entities in clinical texts and converting them into standardized formats such as SNOMED and OMOP. uQuery not only detects and classifies entities but also associates them with contextual information, including negated entities, temporal aspects, and patient-related details. In this work, we explore the use of NLP techniques, specifically Named Entity Recognition (NER), to automatically identify and extract key clinical information from EHRs related to these two cancers. A dataset from Health Research Institute Hospital La Fe (IIS La Fe), comprising 200 annotated breast cancer and 400 lung cancer reports, was used, with eight clinical entities manually labeled using the Doccano platform. To perform NER, we fine-tuned the bsc-bio-ehr-en3 model, a RoBERTa-based biomedical linguistic model pre-trained in Spanish. Fine-tuning was performed using the Transformers architecture, enabling accurate recognition of clinical entities in these cancer types. Our results demonstrate strong overall performance, particularly in identifying entities like MET and PAT, although challenges remain with less frequent entities like EVOL. 

---
# Achieving Tokenizer Flexibility in Language Models through Heuristic Adaptation and Supertoken Learning 

**Authors**: Shaurya Sharthak, Vinayak Pahalwan, Adithya Kamath, Adarsh Shirawalmath  

**Link**: [PDF](https://arxiv.org/pdf/2505.09738)  

**Abstract**: Pretrained language models (LLMs) are often constrained by their fixed tokenization schemes, leading to inefficiencies and performance limitations, particularly for multilingual or specialized applications. This tokenizer lock-in presents significant challenges. standard methods to overcome this often require prohibitive computational resources. Although tokenizer replacement with heuristic initialization aims to reduce this burden, existing methods often require exhaustive residual fine-tuning and still may not fully preserve semantic nuances or adequately address the underlying compression inefficiencies. Our framework introduces two innovations: first, Tokenadapt, a model-agnostic tokenizer transplantation method, and second, novel pre-tokenization learning for multi-word Supertokens to enhance compression and reduce fragmentation. Tokenadapt initializes new unique token embeddings via a hybrid heuristic that combines two methods: a local estimate based on subword decomposition using the old tokenizer, and a global estimate utilizing the top-k semantically similar tokens from the original vocabulary. This methodology aims to preserve semantics while significantly minimizing retraining requirements. Empirical investigations validate both contributions: the transplantation heuristic successfully initializes unique tokens, markedly outperforming conventional baselines and sophisticated methods including Transtokenizer and ReTok, while our Supertokens achieve notable compression gains. Our zero-shot perplexity results demonstrate that the TokenAdapt hybrid initialization consistently yields lower perplexity ratios compared to both ReTok and TransTokenizer baselines across different base models and newly trained target tokenizers. TokenAdapt typically reduced the overall perplexity ratio significantly compared to ReTok, yielding at least a 2-fold improvement in these aggregate scores. 

---
# An AI-Powered Research Assistant in the Lab: A Practical Guide for Text Analysis Through Iterative Collaboration with LLMs 

**Authors**: Gino Carmona-Díaz, William Jiménez-Leal, María Alejandra Grisales, Chandra Sripada, Santiago Amaya, Michael Inzlicht, Juan Pablo Bermúdez  

**Link**: [PDF](https://arxiv.org/pdf/2505.09724)  

**Abstract**: Analyzing texts such as open-ended responses, headlines, or social media posts is a time- and labor-intensive process highly susceptible to bias. LLMs are promising tools for text analysis, using either a predefined (top-down) or a data-driven (bottom-up) taxonomy, without sacrificing quality. Here we present a step-by-step tutorial to efficiently develop, test, and apply taxonomies for analyzing unstructured data through an iterative and collaborative process between researchers and LLMs. Using personal goals provided by participants as an example, we demonstrate how to write prompts to review datasets and generate a taxonomy of life domains, evaluate and refine the taxonomy through prompt and direct modifications, test the taxonomy and assess intercoder agreements, and apply the taxonomy to categorize an entire dataset with high intercoder reliability. We discuss the possibilities and limitations of using LLMs for text analysis. 

---
# VeriFact: Enhancing Long-Form Factuality Evaluation with Refined Fact Extraction and Reference Facts 

**Authors**: Xin Liu, Lechen Zhang, Sheza Munir, Yiyang Gu, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09701)  

**Abstract**: Large language models (LLMs) excel at generating long-form responses, but evaluating their factuality remains challenging due to complex inter-sentence dependencies within the generated facts. Prior solutions predominantly follow a decompose-decontextualize-verify pipeline but often fail to capture essential context and miss key relational facts. In this paper, we introduce VeriFact, a factuality evaluation framework designed to enhance fact extraction by identifying and resolving incomplete and missing facts to support more accurate verification results. Moreover, we introduce FactRBench , a benchmark that evaluates both precision and recall in long-form model responses, whereas prior work primarily focuses on precision. FactRBench provides reference fact sets from advanced LLMs and human-written answers, enabling recall assessment. Empirical evaluations show that VeriFact significantly enhances fact completeness and preserves complex facts with critical relational information, resulting in more accurate factuality evaluation. Benchmarking various open- and close-weight LLMs on FactRBench indicate that larger models within same model family improve precision and recall, but high precision does not always correlate with high recall, underscoring the importance of comprehensive factuality assessment. 

---
# System Prompt Optimization with Meta-Learning 

**Authors**: Yumin Choi, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09666)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities, with optimizing their input prompts playing a pivotal role in maximizing their performance. However, while LLM prompts consist of both the task-agnostic system prompts and task-specific user prompts, existing work on prompt optimization has focused on user prompts specific to individual queries or tasks, and largely overlooked the system prompt that is, once optimized, applicable across different tasks and domains. Motivated by this, we introduce the novel problem of bilevel system prompt optimization, whose objective is to design system prompts that are robust to diverse user prompts and transferable to unseen tasks. To tackle this problem, we then propose a meta-learning framework, which meta-learns the system prompt by optimizing it over various user prompts across multiple datasets, while simultaneously updating the user prompts in an iterative manner to ensure synergy between them. We conduct experiments on 14 unseen datasets spanning 5 different domains, on which we show that our approach produces system prompts that generalize effectively to diverse user prompts. Also, our findings reveal that the optimized system prompt enables rapid adaptation even to unseen tasks, requiring fewer optimization steps for test-time user prompts while achieving improved performance. 

---
# Large Language Models Are More Persuasive Than Incentivized Human Persuaders 

**Authors**: Philipp Schoenegger, Francesco Salvi, Jiacheng Liu, Xiaoli Nan, Ramit Debnath, Barbara Fasolo, Evelina Leivada, Gabriel Recchia, Fritz Günther, Ali Zarifhonarvar, Joe Kwon, Zahoor Ul Islam, Marco Dehnert, Daryl Y. H. Lee, Madeline G. Reinecke, David G. Kamper, Mert Kobaş, Adam Sandford, Jonas Kgomo, Luke Hewitt, Shreya Kapoor, Kerem Oktar, Eyup Engin Kucuk, Bo Feng, Cameron R. Jones, Izzy Gainsburg, Sebastian Olschewski, Nora Heinzelmann, Francisco Cruz, Ben M. Tappin, Tao Ma, Peter S. Park, Rayan Onyonka, Arthur Hjorth, Peter Slattery, Qingcheng Zeng, Lennart Finke, Igor Grossmann, Alessandro Salatiello, Ezra Karger  

**Link**: [PDF](https://arxiv.org/pdf/2505.09662)  

**Abstract**: We directly compare the persuasion capabilities of a frontier large language model (LLM; Claude Sonnet 3.5) against incentivized human persuaders in an interactive, real-time conversational quiz setting. In this preregistered, large-scale incentivized experiment, participants (quiz takers) completed an online quiz where persuaders (either humans or LLMs) attempted to persuade quiz takers toward correct or incorrect answers. We find that LLM persuaders achieved significantly higher compliance with their directional persuasion attempts than incentivized human persuaders, demonstrating superior persuasive capabilities in both truthful (toward correct answers) and deceptive (toward incorrect answers) contexts. We also find that LLM persuaders significantly increased quiz takers' accuracy, leading to higher earnings, when steering quiz takers toward correct answers, and significantly decreased their accuracy, leading to lower earnings, when steering them toward incorrect answers. Overall, our findings suggest that AI's persuasion capabilities already exceed those of humans that have real-money bonuses tied to performance. Our findings of increasingly capable AI persuaders thus underscore the urgency of emerging alignment and governance frameworks. 

---
# DRA-GRPO: Exploring Diversity-Aware Reward Adjustment for R1-Zero-Like Training of Large Language Models 

**Authors**: Xiwen Chen, Wenhui Zhu, Peijie Qiu, Xuanzhao Dong, Hao Wang, Haiyu Wu, Huayu Li, Aristeidis Sotiras, Yalin Wang, Abolfazl Razi  

**Link**: [PDF](https://arxiv.org/pdf/2505.09655)  

**Abstract**: Recent advances in reinforcement learning for language model post-training, such as Group Relative Policy Optimization (GRPO), have shown promise in low-resource settings. However, GRPO typically relies on solution-level and scalar reward signals that fail to capture the semantic diversity among sampled completions. This leads to what we identify as a diversity-quality inconsistency, where distinct reasoning paths may receive indistinguishable rewards. To address this limitation, we propose $\textit{Diversity-aware Reward Adjustment}$ (DRA), a method that explicitly incorporates semantic diversity into the reward computation. DRA uses Submodular Mutual Information (SMI) to downweight redundant completions and amplify rewards for diverse ones. This encourages better exploration during learning, while maintaining stable exploitation of high-quality samples. Our method integrates seamlessly with both GRPO and its variant DR.~GRPO, resulting in $\textit{DRA-GRPO}$ and $\textit{DGA-DR.~GRPO}$. We evaluate our method on five mathematical reasoning benchmarks and find that it outperforms recent strong baselines. It achieves state-of-the-art performance with an average accuracy of 58.2%, using only 7,000 fine-tuning samples and a total training cost of approximately $55. The code is available at this https URL. 

---
# Next Word Suggestion using Graph Neural Network 

**Authors**: Abisha Thapa Magar, Anup Shakya  

**Link**: [PDF](https://arxiv.org/pdf/2505.09649)  

**Abstract**: Language Modeling is a prevalent task in Natural Language Processing. The currently existing most recent and most successful language models often tend to build a massive model with billions of parameters, feed in a tremendous amount of text data, and train with enormous computation resources which require millions of dollars. In this project, we aim to address an important sub-task in language modeling, i.e., context embedding. We propose an approach to exploit the Graph Convolution operation in GNNs to encode the context and use it in coalition with LSTMs to predict the next word given a local context of preceding words. We test this on the custom Wikipedia text corpus using a very limited amount of resources and show that this approach works fairly well to predict the next word. 

---
# MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning 

**Authors**: Ke Wang, Junting Pan, Linda Wei, Aojun Zhou, Weikang Shi, Zimu Lu, Han Xiao, Yunqiao Yang, Houxing Ren, Mingjie Zhan, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10557)  

**Abstract**: Natural language image-caption datasets, widely used for training Large Multimodal Models, mainly focus on natural scenarios and overlook the intricate details of mathematical figures that are critical for problem-solving, hindering the advancement of current LMMs in multimodal mathematical reasoning. To this end, we propose leveraging code as supervision for cross-modal alignment, since code inherently encodes all information needed to generate corresponding figures, establishing a precise connection between the two modalities. Specifically, we co-develop our image-to-code model and dataset with model-in-the-loop approach, resulting in an image-to-code model, FigCodifier and ImgCode-8.6M dataset, the largest image-code dataset to date. Furthermore, we utilize FigCodifier to synthesize novel mathematical figures and then construct MM-MathInstruct-3M, a high-quality multimodal math instruction fine-tuning dataset. Finally, we present MathCoder-VL, trained with ImgCode-8.6M for cross-modal alignment and subsequently fine-tuned on MM-MathInstruct-3M for multimodal math problem solving. Our model achieves a new open-source SOTA across all six metrics. Notably, it surpasses GPT-4o and Claude 3.5 Sonnet in the geometry problem-solving subset of MathVista, achieving improvements of 8.9% and 9.2%. The dataset and models will be released at this https URL. 

---
# Towards a Deeper Understanding of Reasoning Capabilities in Large Language Models 

**Authors**: Annie Wong, Thomas Bäck, Aske Plaat, Niki van Stein, Anna V. Kononova  

**Link**: [PDF](https://arxiv.org/pdf/2505.10543)  

**Abstract**: While large language models demonstrate impressive performance on static benchmarks, the true potential of large language models as self-learning and reasoning agents in dynamic environments remains unclear. This study systematically evaluates the efficacy of self-reflection, heuristic mutation, and planning as prompting techniques to test the adaptive capabilities of agents. We conduct experiments with various open-source language models in dynamic environments and find that larger models generally outperform smaller ones, but that strategic prompting can close this performance gap. Second, a too-long prompt can negatively impact smaller models on basic reactive tasks, while larger models show more robust behaviour. Third, advanced prompting techniques primarily benefit smaller models on complex games, but offer less improvement for already high-performing large language models. Yet, we find that advanced reasoning methods yield highly variable outcomes: while capable of significantly improving performance when reasoning and decision-making align, they also introduce instability and can lead to big performance drops. Compared to human performance, our findings reveal little evidence of true emergent reasoning. Instead, large language model performance exhibits persistent limitations in crucial areas such as planning, reasoning, and spatial coordination, suggesting that current-generation large language models still suffer fundamental shortcomings that may not be fully overcome through self-reflective prompting alone. Reasoning is a multi-faceted task, and while reasoning methods like Chain of thought improves multi-step reasoning on math word problems, our findings using dynamic benchmarks highlight important shortcomings in general reasoning capabilities, indicating a need to move beyond static benchmarks to capture the complexity of reasoning. 

---
# MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models 

**Authors**: Mugilan Ganesan, Shane Segal, Ankur Aggarwal, Nish Sinnadurai, Sean Lie, Vithursan Thangarasa  

**Link**: [PDF](https://arxiv.org/pdf/2505.10526)  

**Abstract**: Speculative decoding significantly accelerates language model inference by enabling a lightweight draft model to propose multiple tokens that a larger target model verifies simultaneously. However, applying this technique to vision-language models (VLMs) presents two fundamental challenges: small language models that could serve as efficient drafters lack the architectural components to process visual inputs, and their token predictions fail to match those of VLM target models that consider visual context. We introduce Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models (MASSV), which transforms existing small language models into effective multimodal drafters through a two-phase approach. MASSV first connects the target VLM's vision encoder to the draft model via a lightweight trainable projector, then applies self-distilled visual instruction tuning using responses generated by the target VLM to align token predictions. Comprehensive experiments across the Qwen2.5-VL and Gemma3 model families demonstrate that MASSV increases accepted length by up to 30% and delivers end-to-end inference speedups of up to 1.46x on visually-grounded tasks. MASSV provides a scalable, architecture-compatible method for accelerating both current and future VLMs. 

---
# RouteNator: A Router-Based Multi-Modal Architecture for Generating Synthetic Training Data for Function Calling LLMs 

**Authors**: Vibha Belavadi, Tushar Vatsa, Dewang Sultania, Suhas Suresha, Ishita Verma, Cheng Chen, Tracy Holloway King, Michael Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2505.10495)  

**Abstract**: This paper addresses fine-tuning Large Language Models (LLMs) for function calling tasks when real user interaction data is unavailable. In digital content creation tools, where users express their needs through natural language queries that must be mapped to API calls, the lack of real-world task-specific data and privacy constraints for training on it necessitate synthetic data generation. Existing approaches to synthetic data generation fall short in diversity and complexity, failing to replicate real-world data distributions and leading to suboptimal performance after LLM fine-tuning. We present a novel router-based architecture that leverages domain resources like content metadata and structured knowledge graphs, along with text-to-text and vision-to-text language models to generate high-quality synthetic training data. Our architecture's flexible routing mechanism enables synthetic data generation that matches observed real-world distributions, addressing a fundamental limitation of traditional approaches. Evaluation on a comprehensive set of real user queries demonstrates significant improvements in both function classification accuracy and API parameter selection. Models fine-tuned with our synthetic data consistently outperform traditional approaches, establishing new benchmarks for function calling tasks. 

---
# Parallel Scaling Law for Language Models 

**Authors**: Mouxiang Chen, Binyuan Hui, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Jianling Sun, Junyang Lin, Zhongxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10475)  

**Abstract**: It is commonly believed that scaling language models should commit a significant space or time cost, by increasing the parameters (parameter scaling) or output tokens (inference-time scaling). We introduce the third and more inference-efficient scaling paradigm: increasing the model's parallel computation during both training and inference time. We apply $P$ diverse and learnable transformations to the input, execute forward passes of the model in parallel, and dynamically aggregate the $P$ outputs. This method, namely parallel scaling (ParScale), scales parallel computation by reusing existing parameters and can be applied to any model structure, optimization procedure, data, or task. We theoretically propose a new scaling law and validate it through large-scale pre-training, which shows that a model with $P$ parallel streams is similar to scaling the parameters by $O(\log P)$ while showing superior inference efficiency. For example, ParScale can use up to 22$\times$ less memory increase and 6$\times$ less latency increase compared to parameter scaling that achieves the same performance improvement. It can also recycle an off-the-shelf pre-trained model into a parallelly scaled one by post-training on a small amount of tokens, further reducing the training budget. The new scaling law we discovered potentially facilitates the deployment of more powerful models in low-resource scenarios, and provides an alternative perspective for the role of computation in machine learning. 

---
# Superposition Yields Robust Neural Scaling 

**Authors**: Yizhou liu, Ziming Liu, Jeff Gore  

**Link**: [PDF](https://arxiv.org/pdf/2505.10465)  

**Abstract**: The success of today's large language models (LLMs) depends on the observation that larger models perform better. However, the origin of this neural scaling law -- the finding that loss decreases as a power law with model size -- remains unclear. Starting from two empirical principles -- that LLMs represent more things than the model dimensions (widths) they have (i.e., representations are superposed), and that words or concepts in language occur with varying frequencies -- we constructed a toy model to study the loss scaling with model size. We found that when superposition is weak, meaning only the most frequent features are represented without interference, the scaling of loss with model size depends on the underlying feature frequency; if feature frequencies follow a power law, so does the loss. In contrast, under strong superposition, where all features are represented but overlap with each other, the loss becomes inversely proportional to the model dimension across a wide range of feature frequency distributions. This robust scaling behavior is explained geometrically: when many more vectors are packed into a lower dimensional space, the interference (squared overlaps) between vectors scales inversely with that dimension. We then analyzed four families of open-sourced LLMs and found that they exhibit strong superposition and quantitatively match the predictions of our toy model. The Chinchilla scaling law turned out to also agree with our results. We conclude that representation superposition is an important mechanism underlying the observed neural scaling laws. We anticipate that these insights will inspire new training strategies and model architectures to achieve better performance with less computation and fewer parameters. 

---
# StoryReasoning Dataset: Using Chain-of-Thought for Scene Understanding and Grounded Story Generation 

**Authors**: Daniel A. P. Oliveira, David Martins de Matos  

**Link**: [PDF](https://arxiv.org/pdf/2505.10292)  

**Abstract**: Visual storytelling systems struggle to maintain character identity across frames and link actions to appropriate subjects, frequently leading to referential hallucinations. These issues can be addressed through grounding of characters, objects, and other entities on the visual elements. We propose StoryReasoning, a dataset containing 4,178 stories derived from 52,016 movie images, with both structured scene analyses and grounded stories. Each story maintains character and object consistency across frames while explicitly modeling multi-frame relationships through structured tabular representations. Our approach features cross-frame object re-identification using visual similarity and face recognition, chain-of-thought reasoning for explicit narrative modeling, and a grounding scheme that links textual elements to visual entities across multiple frames. We establish baseline performance by fine-tuning Qwen2.5-VL 7B, creating Qwen Storyteller, which performs end-to-end object detection, re-identification, and landmark detection while maintaining consistent object references throughout the story. Evaluation demonstrates a reduction from 4.06 to 3.56 (-12.3%) hallucinations on average per story when compared to a non-fine-tuned model. 

---
# On the Interplay of Human-AI Alignment,Fairness, and Performance Trade-offs in Medical Imaging 

**Authors**: Haozhe Luo, Ziyu Zhou, Zixin Shu, Aurélie Pahud de Mortanges, Robert Berke, Mauricio Reyes  

**Link**: [PDF](https://arxiv.org/pdf/2505.10231)  

**Abstract**: Deep neural networks excel in medical imaging but remain prone to biases, leading to fairness gaps across demographic groups. We provide the first systematic exploration of Human-AI alignment and fairness in this domain. Our results show that incorporating human insights consistently reduces fairness gaps and enhances out-of-domain generalization, though excessive alignment can introduce performance trade-offs, emphasizing the need for calibrated strategies. These findings highlight Human-AI alignment as a promising approach for developing fair, robust, and generalizable medical AI systems, striking a balance between expert guidance and automated efficiency. Our code is available at this https URL. 

---
# ComplexFormer: Disruptively Advancing Transformer Inference Ability via Head-Specific Complex Vector Attention 

**Authors**: Jintian Shao, Hongyi Huang, Jiayi Wu, Beiwen Zhang, ZhiYu Wu, You Shan, MingKai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.10222)  

**Abstract**: Transformer models rely on self-attention to capture token dependencies but face challenges in effectively integrating positional information while allowing multi-head attention (MHA) flexibility. Prior methods often model semantic and positional differences disparately or apply uniform positional adjustments across heads, potentially limiting representational capacity. This paper introduces ComplexFormer, featuring Complex Multi-Head Attention-CMHA. CMHA empowers each head to independently model semantic and positional differences unified within the complex plane, representing interactions as rotations and scaling. ComplexFormer incorporates two key improvements: (1) a per-head Euler transformation, converting real-valued query/key projections into polar-form complex vectors for head-specific complex subspace operation; and (2) a per-head adaptive differential rotation mechanism, exp[i(Adapt(ASmn,i) + Delta(Pmn),i)], allowing each head to learn distinct strategies for integrating semantic angle differences (ASmn,i) with relative positional encodings (Delta(Pmn),i). Extensive experiments on language modeling, text generation, code generation, and mathematical reasoning show ComplexFormer achieves superior performance, significantly lower generation perplexity , and improved long-context coherence compared to strong baselines like RoPE-Transformers. ComplexFormer demonstrates strong parameter efficiency, offering a more expressive, adaptable attention mechanism. 

---
# Why 1 + 1 < 1 in Visual Token Pruning: Beyond Naive Integration via Multi-Objective Balanced Covering 

**Authors**: Yangfu Li, Hongjian Zhan, Tianyi Chen, Qi Liu, Yue Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10118)  

**Abstract**: Existing visual token pruning methods target prompt alignment and visual preservation with static strategies, overlooking the varying relative importance of these objectives across tasks, which leads to inconsistent performance. To address this, we derive the first closed-form error bound for visual token pruning based on the Hausdorff distance, uniformly characterizing the contributions of both objectives. Moreover, leveraging $\epsilon$-covering theory, we reveal an intrinsic trade-off between these objectives and quantify their optimal attainment levels under a fixed budget. To practically handle this trade-off, we propose Multi-Objective Balanced Covering (MoB), which reformulates visual token pruning as a bi-objective covering problem. In this framework, the attainment trade-off reduces to budget allocation via greedy radius trading. MoB offers a provable performance bound and linear scalability with respect to the number of input visual tokens, enabling adaptation to challenging pruning scenarios. Extensive experiments show that MoB preserves 96.4% of performance for LLaVA-1.5-7B using only 11.1% of the original visual tokens and accelerates LLaVA-Next-7B by 1.3-1.5$\times$ with negligible performance loss. Additionally, evaluations on Qwen2-VL and Video-LLaVA confirm that MoB integrates seamlessly into advanced MLLMs and diverse vision-language tasks. 

---
# Learning Virtual Machine Scheduling in Cloud Computing through Language Agents 

**Authors**: JieHao Wu, Ziwei Wang, Junjie Sheng, Wenhao Li, Xiangfei Wang, Jun Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.10117)  

**Abstract**: In cloud services, virtual machine (VM) scheduling is a typical Online Dynamic Multidimensional Bin Packing (ODMBP) problem, characterized by large-scale complexity and fluctuating demands. Traditional optimization methods struggle to adapt to real-time changes, domain-expert-designed heuristic approaches suffer from rigid strategies, and existing learning-based methods often lack generalizability and interpretability. To address these limitations, this paper proposes a hierarchical language agent framework named MiCo, which provides a large language model (LLM)-driven heuristic design paradigm for solving ODMBP. Specifically, ODMBP is formulated as a Semi-Markov Decision Process with Options (SMDP-Option), enabling dynamic scheduling through a two-stage architecture, i.e., Option Miner and Option Composer. Option Miner utilizes LLMs to discover diverse and useful non-context-aware strategies by interacting with constructed environments. Option Composer employs LLMs to discover a composing strategy that integrates the non-context-aware strategies with the contextual ones. Extensive experiments on real-world enterprise datasets demonstrate that MiCo achieves a 96.9\% competitive ratio in large-scale scenarios involving more than 10,000 virtual machines. It maintains high performance even under nonstationary request flows and diverse configurations, thus validating its effectiveness in complex and large-scale cloud environments. 

---
# From Text to Network: Constructing a Knowledge Graph of Taiwan-Based China Studies Using Generative AI 

**Authors**: Hsuan-Lei Shao  

**Link**: [PDF](https://arxiv.org/pdf/2505.10093)  

**Abstract**: Taiwanese China Studies (CS) has developed into a rich, interdisciplinary research field shaped by the unique geopolitical position and long standing academic engagement with Mainland China. This study responds to the growing need to systematically revisit and reorganize decades of Taiwan based CS scholarship by proposing an AI assisted approach that transforms unstructured academic texts into structured, interactive knowledge representations. We apply generative AI (GAI) techniques and large language models (LLMs) to extract and standardize entity relation triples from 1,367 peer reviewed CS articles published between 1996 and 2019. These triples are then visualized through a lightweight this http URL based system, forming the foundation of a domain specific knowledge graph and vector database for the field. This infrastructure allows users to explore conceptual nodes and semantic relationships across the corpus, revealing previously uncharted intellectual trajectories, thematic clusters, and research gaps. By decomposing textual content into graph structured knowledge units, our system enables a paradigm shift from linear text consumption to network based knowledge navigation. In doing so, it enhances scholarly access to CS literature while offering a scalable, data driven alternative to traditional ontology construction. This work not only demonstrates how generative AI can augment area studies and digital humanities but also highlights its potential to support a reimagined scholarly infrastructure for regional knowledge systems. 

---
# Advanced Crash Causation Analysis for Freeway Safety: A Large Language Model Approach to Identifying Key Contributing Factors 

**Authors**: Ahmed S. Abdelrahman, Mohamed Abdel-Aty, Samgyu Yang, Abdulrahman Faden  

**Link**: [PDF](https://arxiv.org/pdf/2505.09949)  

**Abstract**: Understanding the factors contributing to traffic crashes and developing strategies to mitigate their severity is essential. Traditional statistical methods and machine learning models often struggle to capture the complex interactions between various factors and the unique characteristics of each crash. This research leverages large language model (LLM) to analyze freeway crash data and provide crash causation analysis accordingly. By compiling 226 traffic safety studies related to freeway crashes, a training dataset encompassing environmental, driver, traffic, and geometric design factors was created. The Llama3 8B model was fine-tuned using QLoRA to enhance its understanding of freeway crashes and their contributing factors, as covered in these studies. The fine-tuned Llama3 8B model was then used to identify crash causation without pre-labeled data through zero-shot classification, providing comprehensive explanations to ensure that the identified causes were reasonable and aligned with existing research. Results demonstrate that LLMs effectively identify primary crash causes such as alcohol-impaired driving, speeding, aggressive driving, and driver inattention. Incorporating event data, such as road maintenance, offers more profound insights. The model's practical applicability and potential to improve traffic safety measures were validated by a high level of agreement among researchers in the field of traffic safety, as reflected in questionnaire results with 88.89%. This research highlights the complex nature of traffic crashes and how LLMs can be used for comprehensive analysis of crash causation and other contributing factors. Moreover, it provides valuable insights and potential countermeasures to aid planners and policymakers in developing more effective and efficient traffic safety practices. 

---
# PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative In-Context Optimization 

**Authors**: Yidan Wang, Yanan Cao, Yubing Ren, Fang Fang, Zheng Lin, Binxing Fang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09921)  

**Abstract**: Large Language Models (LLMs) excel in various domains but pose inherent privacy risks. Existing methods to evaluate privacy leakage in LLMs often use memorized prefixes or simple instructions to extract data, both of which well-alignment models can easily block. Meanwhile, Jailbreak attacks bypass LLM safety mechanisms to generate harmful content, but their role in privacy scenarios remains underexplored. In this paper, we examine the effectiveness of jailbreak attacks in extracting sensitive information, bridging privacy leakage and jailbreak attacks in LLMs. Moreover, we propose PIG, a novel framework targeting Personally Identifiable Information (PII) and addressing the limitations of current jailbreak methods. Specifically, PIG identifies PII entities and their types in privacy queries, uses in-context learning to build a privacy context, and iteratively updates it with three gradient-based strategies to elicit target PII. We evaluate PIG and existing jailbreak methods using two privacy-related datasets. Experiments on four white-box and two black-box LLMs show that PIG outperforms baseline methods and achieves state-of-the-art (SoTA) results. The results underscore significant privacy risks in LLMs, emphasizing the need for stronger safeguards. Our code is availble at \href{this https URL}{this https URL}. 

---
# Comparing Exploration-Exploitation Strategies of LLMs and Humans: Insights from Standard Multi-armed Bandit Tasks 

**Authors**: Ziyuan Zhang, Darcy Wang, Ningyuan Chen, Rodrigo Mansur, Vahid Sarhangian  

**Link**: [PDF](https://arxiv.org/pdf/2505.09901)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate or automate human behavior in complex sequential decision-making tasks. A natural question is then whether LLMs exhibit similar decision-making behavior to humans, and can achieve comparable (or superior) performance. In this work, we focus on the exploration-exploitation (E&E) tradeoff, a fundamental aspect of dynamic decision-making under uncertainty. We employ canonical multi-armed bandit (MAB) tasks introduced in the cognitive science and psychiatry literature to conduct a comparative study of the E&E strategies of LLMs, humans, and MAB algorithms. We use interpretable choice models to capture the E&E strategies of the agents and investigate how explicit reasoning, through both prompting strategies and reasoning-enhanced models, shapes LLM decision-making. We find that reasoning shifts LLMs toward more human-like behavior, characterized by a mix of random and directed exploration. In simple stationary tasks, reasoning-enabled LLMs exhibit similar levels of random and directed exploration compared to humans. However, in more complex, non-stationary environments, LLMs struggle to match human adaptability, particularly in effective directed exploration, despite achieving similar regret in certain scenarios. Our findings highlight both the promise and limits of LLMs as simulators of human behavior and tools for automated decision-making and point to potential areas of improvements. 

---
# Predictability Shapes Adaptation: An Evolutionary Perspective on Modes of Learning in Transformers 

**Authors**: Alexander Y. Ku, Thomas L. Griffiths, Stephanie C.Y. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2505.09855)  

**Abstract**: Transformer models learn in two distinct modes: in-weights learning (IWL), encoding knowledge into model weights, and in-context learning (ICL), adapting flexibly to context without weight modification. To better understand the interplay between these learning modes, we draw inspiration from evolutionary biology's analogous adaptive strategies: genetic encoding (akin to IWL, adapting over generations and fixed within an individual's lifetime) and phenotypic plasticity (akin to ICL, enabling flexible behavioral responses to environmental cues). In evolutionary biology, environmental predictability dictates the balance between these strategies: stability favors genetic encoding, while reliable predictive cues promote phenotypic plasticity. We experimentally operationalize these dimensions of predictability and systematically investigate their influence on the ICL/IWL balance in Transformers. Using regression and classification tasks, we show that high environmental stability decisively favors IWL, as predicted, with a sharp transition at maximal stability. Conversely, high cue reliability enhances ICL efficacy, particularly when stability is low. Furthermore, learning dynamics reveal task-contingent temporal evolution: while a canonical ICL-to-IWL shift occurs in some settings (e.g., classification with many classes), we demonstrate that scenarios with easier IWL (e.g., fewer classes) or slower ICL acquisition (e.g., regression) can exhibit an initial IWL phase later yielding to ICL dominance. These findings support a relative-cost hypothesis for explaining these learning mode transitions, establishing predictability as a critical factor governing adaptive strategies in Transformers, and offering novel insights for understanding ICL and guiding training methodologies. 

---
# A Survey on Large Language Models in Multimodal Recommender Systems 

**Authors**: Alejo Lopez-Avila, Jinhua Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.09777)  

**Abstract**: Multimodal recommender systems (MRS) integrate heterogeneous user and item data, such as text, images, and structured information, to enhance recommendation performance. The emergence of large language models (LLMs) introduces new opportunities for MRS by enabling semantic reasoning, in-context learning, and dynamic input handling. Compared to earlier pre-trained language models (PLMs), LLMs offer greater flexibility and generalisation capabilities but also introduce challenges related to scalability and model accessibility. This survey presents a comprehensive review of recent work at the intersection of LLMs and MRS, focusing on prompting strategies, fine-tuning methods, and data adaptation techniques. We propose a novel taxonomy to characterise integration patterns, identify transferable techniques from related recommendation domains, provide an overview of evaluation metrics and datasets, and point to possible future directions. We aim to clarify the emerging role of LLMs in multimodal recommendation and support future research in this rapidly evolving field. 

---
# Tales of the 2025 Los Angeles Fire: Hotwash for Public Health Concerns in Reddit via LLM-Enhanced Topic Modeling 

**Authors**: Sulong Zhou, Qunying Huang, Shaoheng Zhou, Yun Hang, Xinyue Ye, Aodong Mei, Kathryn Phung, Yuning Ye, Uma Govindswamy, Zehan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.09665)  

**Abstract**: Wildfires have become increasingly frequent, irregular, and severe in recent years. Understanding how affected populations perceive and respond during wildfire crises is critical for timely and empathetic disaster response. Social media platforms offer a crowd-sourced channel to capture evolving public discourse, providing hyperlocal information and insight into public sentiment. This study analyzes Reddit discourse during the 2025 Los Angeles wildfires, spanning from the onset of the disaster to full containment. We collect 385 posts and 114,879 comments related to the Palisades and Eaton fires. We adopt topic modeling methods to identify the latent topics, enhanced by large language models (LLMs) and human-in-the-loop (HITL) refinement. Furthermore, we develop a hierarchical framework to categorize latent topics, consisting of two main categories, Situational Awareness (SA) and Crisis Narratives (CN). The volume of SA category closely aligns with real-world fire progressions, peaking within the first 2-5 days as the fires reach the maximum extent. The most frequent co-occurring category set of public health and safety, loss and damage, and emergency resources expands on a wide range of health-related latent topics, including environmental health, occupational health, and one health. Grief signals and mental health risks consistently accounted for 60 percentage and 40 percentage of CN instances, respectively, with the highest total volume occurring at night. This study contributes the first annotated social media dataset on the 2025 LA fires, and introduces a scalable multi-layer framework that leverages topic modeling for crisis discourse analysis. By identifying persistent public health concerns, our results can inform more empathetic and adaptive strategies for disaster response, public health communication, and future research in comparable climate-related disaster events. 

---
