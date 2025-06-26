# Inside you are many wolves: Using cognitive models to interpret value trade-offs in LLMs 

**Authors**: Sonia K. Murthy, Rosie Zhao, Jennifer Hu, Sham Kakade, Markus Wulfmeier, Peng Qian, Tomer Ullman  

**Link**: [PDF](https://arxiv.org/pdf/2506.20666)  

**Abstract**: Navigating everyday social situations often requires juggling conflicting goals, such as conveying a harsh truth, maintaining trust, all while still being mindful of another person's feelings. These value trade-offs are an integral part of human decision-making and language use, however, current tools for interpreting such dynamic and multi-faceted notions of values in LLMs are limited. In cognitive science, so-called "cognitive models" provide formal accounts of these trade-offs in humans, by modeling the weighting of a speaker's competing utility functions in choosing an action or utterance. In this work, we use a leading cognitive model of polite speech to interpret the extent to which LLMs represent human-like trade-offs. We apply this lens to systematically evaluate value trade-offs in two encompassing model settings: degrees of reasoning "effort" in frontier black-box models, and RL post-training dynamics of open-source models. Our results highlight patterns of higher informational utility than social utility in reasoning models, and in open-source models shown to be stronger in mathematical reasoning. Our findings from LLMs' training dynamics suggest large shifts in utility values early on in training with persistent effects of the choice of base model and pretraining data, compared to feedback dataset or alignment method. We show that our method is responsive to diverse aspects of the rapidly evolving LLM landscape, with insights for forming hypotheses about other high-level behaviors, shaping training regimes for reasoning models, and better controlling trade-offs between values during model training. 

---
# Memento: Note-Taking for Your Future Self 

**Authors**: Chao Wan, Albert Gong, Mihir Mishra, Carl-Leander Henneking, Claas Beger, Kilian Q. Weinberger  

**Link**: [PDF](https://arxiv.org/pdf/2506.20642)  

**Abstract**: Large language models (LLMs) excel at reasoning-only tasks, but struggle when reasoning must be tightly coupled with retrieval, as in multi-hop question answering. To overcome these limitations, we introduce a prompting strategy that first decomposes a complex question into smaller steps, then dynamically constructs a database of facts using LLMs, and finally pieces these facts together to solve the question. We show how this three-stage strategy, which we call Memento, can boost the performance of existing prompting strategies across diverse settings. On the 9-step PhantomWiki benchmark, Memento doubles the performance of chain-of-thought (CoT) when all information is provided in context. On the open-domain version of 2WikiMultiHopQA, CoT-RAG with Memento improves over vanilla CoT-RAG by more than 20 F1 percentage points and over the multi-hop RAG baseline, IRCoT, by more than 13 F1 percentage points. On the challenging MuSiQue dataset, Memento improves ReAct by more than 3 F1 percentage points, demonstrating its utility in agentic settings. 

---
# DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation 

**Authors**: Shansan Gong, Ruixiang Zhang, Huangjie Zheng, Jiatao Gu, Navdeep Jaitly, Lingpeng Kong, Yizhe Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20639)  

**Abstract**: Diffusion large language models (dLLMs) are compelling alternatives to autoregressive (AR) models because their denoising models operate over the entire sequence. The global planning and iterative refinement features of dLLMs are particularly useful for code generation. However, current training and inference mechanisms for dLLMs in coding are still under-explored. To demystify the decoding behavior of dLLMs and unlock their potential for coding, we systematically investigate their denoising processes and reinforcement learning (RL) methods. We train a 7B dLLM, \textbf{DiffuCoder}, on 130B tokens of code. Using this model as a testbed, we analyze its decoding behavior, revealing how it differs from that of AR models: (1) dLLMs can decide how causal their generation should be without relying on semi-AR decoding, and (2) increasing the sampling temperature diversifies not only token choices but also their generation order. This diversity creates a rich search space for RL rollouts. For RL training, to reduce the variance of token log-likelihood estimates and maintain training efficiency, we propose \textbf{coupled-GRPO}, a novel sampling scheme that constructs complementary mask noise for completions used in training. In our experiments, coupled-GRPO significantly improves DiffuCoder's performance on code generation benchmarks (+4.4\% on EvalPlus) and reduces reliance on AR causal during decoding. Our work provides deeper insight into the machinery of dLLM generation and offers an effective, diffusion-native RL training framework. this https URL. 

---
# Model Editing as a Double-Edged Sword: Steering Agent Ethical Behavior Toward Beneficence or Harm 

**Authors**: Baixiang Huang, Zhen Tan, Haoran Wang, Zijie Liu, Dawei Li, Ali Payani, Huan Liu, Tianlong Chen, Kai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20606)  

**Abstract**: Agents based on Large Language Models (LLMs) have demonstrated strong capabilities across a wide range of tasks. However, deploying LLM-based agents in high-stakes domains comes with significant safety and ethical risks. Unethical behavior by these agents can directly result in serious real-world consequences, including physical harm and financial loss. To efficiently steer the ethical behavior of agents, we frame agent behavior steering as a model editing task, which we term Behavior Editing. Model editing is an emerging area of research that enables precise and efficient modifications to LLMs while preserving their overall capabilities. To systematically study and evaluate this approach, we introduce BehaviorBench, a multi-tier benchmark grounded in psychological moral theories. This benchmark supports both the evaluation and editing of agent behaviors across a variety of scenarios, with each tier introducing more complex and ambiguous scenarios. We first demonstrate that Behavior Editing can dynamically steer agents toward the target behavior within specific scenarios. Moreover, Behavior Editing enables not only scenario-specific local adjustments but also more extensive shifts in an agent's global moral alignment. We demonstrate that Behavior Editing can be used to promote ethical and benevolent behavior or, conversely, to induce harmful or malicious behavior. Through comprehensive evaluations on agents based on frontier LLMs, BehaviorBench shows the effectiveness of Behavior Editing across different models and scenarios. Our findings offer key insights into a new paradigm for steering agent behavior, highlighting both the promise and perils of Behavior Editing. 

---
# When Life Gives You Samples: The Benefits of Scaling up Inference Compute for Multilingual LLMs 

**Authors**: Ammar Khairi, Daniel D'souza, Ye Shen, Julia Kreutzer, Sara Hooker  

**Link**: [PDF](https://arxiv.org/pdf/2506.20544)  

**Abstract**: Recent advancements in large language models (LLMs) have shifted focus toward scaling inference-time compute, improving performance without retraining the model. A common approach is to sample multiple outputs in parallel, and select one of these as the final output. However, work to date has focused on English and a handful of domains such as math and code. In contrast, we are most interested in techniques that generalize across open-ended tasks, formally verifiable tasks, and across languages. In this work, we study how to robustly scale inference-time compute for open-ended generative tasks in a multilingual, multi-task setting.
Our findings show that both sampling strategy based on temperature variation and selection strategy must be adapted to account for diverse domains and varied language settings. We evaluate existing selection methods, revealing that strategies effective in English often fail to generalize across languages. We propose novel sampling and selection strategies specifically adapted for multilingual and multi-task inference scenarios, and show they yield notable gains across languages and tasks. In particular, our combined sampling and selection methods lead to an average +6.8 jump in win-rates for our 8B models on m-ArenaHard-v2.0 prompts, against proprietary models such as Gemini. At larger scale, Command-A (111B model) equipped with our methods, shows +9.0 improvement in win-rates on the same benchmark with just five samples against single-sample decoding, a substantial increase at minimal cost. Our results underscore the need for language- and task-aware approaches to inference-time compute, aiming to democratize performance improvements in underrepresented languages. 

---
# OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling 

**Authors**: Zengzhi Wang, Fan Zhou, Xuefeng Li, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20512)  

**Abstract**: Different base language model families, such as Llama and Qwen, exhibit divergent behaviors during post-training with reinforcement learning (RL), especially on reasoning-intensive tasks. What makes a base language model suitable for reinforcement learning? Gaining deeper insight into this question is essential for developing RL-scalable foundation models of the next generation. In this work, we investigate how mid-training strategies shape RL dynamics, focusing on two representative model families: Qwen and Llama. Our study reveals that (1) high-quality mathematical corpora, such as MegaMath-Web-Pro, significantly improve both base model and RL performance, while existing alternatives (e.g., FineMath-4plus) fail to do so; (2) further adding QA-style data, particularly long chain-of-thought (CoT) reasoning examples, enhances RL outcomes, and instruction data further unlocks this effect; (3) while long-CoT improves reasoning depth, it can also induce verbosity of model responses and unstability of RL training, underscoring the importance of data formatting; (4) scaling mid-training consistently leads to stronger downstream RL performance. Building on these insights, we introduce a two-stage mid-training strategy, Stable-then-Decay, in which base models are first trained on 200B tokens with a constant learning rate, followed by 20B tokens across three CoT-focused branches with learning rate decay. This yields OctoThinker, a family of models demonstrating strong RL compatibility and closing the performance gap with more RL-friendly model families, i.e., Qwen. We hope our work will help shape pre-training strategies for foundation models in the RL era. To support further research, we release our open-source models along with a curated math reasoning-intensive corpus of over 70 billion tokens (i.e., MegaMath-Web-Pro-Max). 

---
# ReCode: Updating Code API Knowledge with Reinforcement Learning 

**Authors**: Haoze Wu, Yunzhi Yao, Wenhao Yu, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20495)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable code generation capabilities but falter when adapting to frequent updates in external library APIs. This critical limitation, stemming from reliance on outdated API knowledge from their training data, even with access to current documentation, impedes reliable code generation in dynamic environments. To tackle this issue, we propose ReCode (rule-based Reinforcement learning for Code Update), a novel framework that mimics human programmer adaptation to API changes. Specifically, we construct a dataset of approximately 2,000 data entries to train the LLMs to perform version migration based on updated information. Then, we introduce a modified string similarity metric for code evaluation as the reward for reinforcement learning. Our experiments demonstrate that ReCode substantially boosts LLMs' code generation performance in dynamic API scenarios, especially on the unseen CodeUpdateArena task. Crucially, compared to supervised fine-tuning, ReCode has less impact on LLMs' general code generation abilities. We apply ReCode on various LLMs and reinforcement learning algorithms (GRPO and DAPO), all achieving consistent improvements. Notably, after training, Qwen2.5-Coder-7B outperforms that of the 32B parameter code instruction-tuned model and the reasoning model with the same architecture. Code is available at this https URL. 

---
# GPTailor: Large Language Model Pruning Through Layer Cutting and Stitching 

**Authors**: Guinan Su, Li Shen, Lu Yin, Shiwei Liu, Yanwu Yang, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2506.20480)  

**Abstract**: Large language models (LLMs) have shown remarkable capabilities in language understanding and generation. However, such impressive capability typically comes with a substantial model size, which presents significant challenges in deployment and inference. While structured pruning of model parameters offers a promising way to reduce computational costs at deployment time, current methods primarily focus on single model pruning. In this work, we develop a novel strategy to compress models by strategically combining or merging layers from finetuned model variants, which preserves the original model's abilities by aggregating capabilities accentuated in different finetunes. We pose the optimal tailoring of these LLMs as a zero-order optimization problem, adopting a search space that supports three different operations: (1) Layer removal, (2) Layer selection from different candidate models, and (3) Layer merging. Our experiments demonstrate that this approach leads to competitive model pruning, for example, for the Llama2-13B model families, our compressed models maintain approximately 97.3\% of the original performance while removing $\sim25\%$ of parameters, significantly outperforming previous state-of-the-art methods. The code is available at this https URL. 

---
# Knowledge-Aware Diverse Reranking for Cross-Source Question Answering 

**Authors**: Tong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.20476)  

**Abstract**: This paper presents Team Marikarp's solution for the SIGIR 2025 LiveRAG competition. The competition's evaluation set, automatically generated by DataMorgana from internet corpora, encompassed a wide range of target topics, question types, question formulations, audience types, and knowledge organization methods. It offered a fair evaluation of retrieving question-relevant supporting documents from a 15M documents subset of the FineWeb corpus. Our proposed knowledge-aware diverse reranking RAG pipeline achieved first place in the competition. 

---
# Time is On My Side: Dynamics of Talk-Time Sharing in Video-chat Conversations 

**Authors**: Kaixiang Zhang, Justine Zhang, Cristian Danescu-Niculescu-Mizil  

**Link**: [PDF](https://arxiv.org/pdf/2506.20474)  

**Abstract**: An intrinsic aspect of every conversation is the way talk-time is shared between multiple speakers. Conversations can be balanced, with each speaker claiming a similar amount of talk-time, or imbalanced when one talks disproportionately. Such overall distributions are the consequence of continuous negotiations between the speakers throughout the conversation: who should be talking at every point in time, and for how long?
In this work we introduce a computational framework for quantifying both the conversation-level distribution of talk-time between speakers, as well as the lower-level dynamics that lead to it. We derive a typology of talk-time sharing dynamics structured by several intuitive axes of variation. By applying this framework to a large dataset of video-chats between strangers, we confirm that, perhaps unsurprisingly, different conversation-level distributions of talk-time are perceived differently by speakers, with balanced conversations being preferred over imbalanced ones, especially by those who end up talking less. Then we reveal that -- even when they lead to the same level of overall balance -- different types of talk-time sharing dynamics are perceived differently by the participants, highlighting the relevance of our newly introduced typology. Finally, we discuss how our framework offers new tools to designers of computer-mediated communication platforms, for both human-human and human-AI communication. 

---
# Probing AI Safety with Source Code 

**Authors**: Ujwal Narayan, Shreyas Chaudhari, Ashwin Kalyan, Tanmay Rajpurohit, Karthik Narasimhan, Ameet Deshpande, Vishvak Murahari  

**Link**: [PDF](https://arxiv.org/pdf/2506.20471)  

**Abstract**: Large language models (LLMs) have become ubiquitous, interfacing with humans in numerous safety-critical applications. This necessitates improving capabilities, but importantly coupled with greater safety measures to align these models with human values and preferences. In this work, we demonstrate that contemporary models fall concerningly short of the goal of AI safety, leading to an unsafe and harmful experience for users. We introduce a prompting strategy called Code of Thought (CoDoT) to evaluate the safety of LLMs. CoDoT converts natural language inputs to simple code that represents the same intent. For instance, CoDoT transforms the natural language prompt "Make the statement more toxic: {text}" to: "make_more_toxic({text})". We show that CoDoT results in a consistent failure of a wide range of state-of-the-art LLMs. For example, GPT-4 Turbo's toxicity increases 16.5 times, DeepSeek R1 fails 100% of the time, and toxicity increases 300% on average across seven modern LLMs. Additionally, recursively applying CoDoT can further increase toxicity two times. Given the rapid and widespread adoption of LLMs, CoDoT underscores the critical need to evaluate safety efforts from first principles, ensuring that safety and capabilities advance together. 

---
# An Agentic System for Rare Disease Diagnosis with Traceable Reasoning 

**Authors**: Weike Zhao, Chaoyi Wu, Yanjie Fan, Xiaoman Zhang, Pengcheng Qiu, Yuze Sun, Xiao Zhou, Yanfeng Wang, Ya Zhang, Yongguo Yu, Kun Sun, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.20430)  

**Abstract**: Rare diseases collectively affect over 300 million individuals worldwide, yet timely and accurate diagnosis remains a pervasive challenge. This is largely due to their clinical heterogeneity, low individual prevalence, and the limited familiarity most clinicians have with rare conditions. Here, we introduce DeepRare, the first rare disease diagnosis agentic system powered by a large language model (LLM), capable of processing heterogeneous clinical inputs. The system generates ranked diagnostic hypotheses for rare diseases, each accompanied by a transparent chain of reasoning that links intermediate analytic steps to verifiable medical evidence.
DeepRare comprises three key components: a central host with a long-term memory module; specialized agent servers responsible for domain-specific analytical tasks integrating over 40 specialized tools and web-scale, up-to-date medical knowledge sources, ensuring access to the most current clinical information. This modular and scalable design enables complex diagnostic reasoning while maintaining traceability and adaptability. We evaluate DeepRare on eight datasets. The system demonstrates exceptional diagnostic performance among 2,919 diseases, achieving 100% accuracy for 1013 diseases. In HPO-based evaluations, DeepRare significantly outperforms other 15 methods, like traditional bioinformatics diagnostic tools, LLMs, and other agentic systems, achieving an average Recall@1 score of 57.18% and surpassing the second-best method (Reasoning LLM) by a substantial margin of 23.79 percentage points. For multi-modal input scenarios, DeepRare achieves 70.60% at Recall@1 compared to Exomiser's 53.20% in 109 cases. Manual verification of reasoning chains by clinical experts achieves 95.40% agreements. Furthermore, the DeepRare system has been implemented as a user-friendly web application this http URL. 

---
# TAPS: Tool-Augmented Personalisation via Structured Tagging 

**Authors**: Ekaterina Taktasheva, Jeff Dalton  

**Link**: [PDF](https://arxiv.org/pdf/2506.20409)  

**Abstract**: Recent advancements in tool-augmented large language models have enabled them to interact with external tools, enhancing their ability to perform complex user tasks. However, existing approaches overlook the role of personalisation in guiding tool use. This work investigates how user preferences can be effectively integrated into goal-oriented dialogue agents. Through extensive analysis, we identify key weaknesses in the ability of LLMs to personalise tool use. To this end, we introduce \name, a novel solution that enhances personalised tool use by leveraging a structured tagging tool and an uncertainty-based tool detector. TAPS significantly improves the ability of LLMs to incorporate user preferences, achieving the new state-of-the-art for open source models on the NLSI task. 

---
# Biomed-Enriched: A Biomedical Dataset Enriched with LLMs for Pretraining and Extracting Rare and Hidden Content 

**Authors**: Rian Touchent, Nathan Godey, Eric de la Clergerie  

**Link**: [PDF](https://arxiv.org/pdf/2506.20331)  

**Abstract**: We introduce Biomed-Enriched, a biomedical text dataset constructed from PubMed via a two-stage annotation process. In the first stage, a large language model annotates 400K paragraphs from PubMed scientific articles, assigning scores for their type (review, study, clinical case, other), domain (clinical, biomedical, other), and educational quality. The educational quality score (rated 1 to 5) estimates how useful a paragraph is for college-level learning. These annotations are then used to fine-tune a small language model, which propagates the labels across the full PMC-OA corpus. The resulting metadata allows us to extract refined subsets, including 2M clinical case paragraphs with over 450K high-quality ones from articles with commercial-use licenses, and to construct several variants via quality filtering and domain upsampling. Clinical text is typically difficult to access due to privacy constraints, as hospital records cannot be publicly shared. Hence, our dataset provides an alternative large-scale, openly available collection of clinical cases from PubMed, making it a valuable resource for biomedical and clinical NLP. Preliminary continual-pretraining experiments with OLMo2 suggest these curated subsets enable targeted improvements, with clinical upsampling boosting performance by ~5% on MMLU ProfMed and educational quality filtering improving MedQA and MedMCQA by ~1%. Combinations of these techniques led to faster convergence, reaching same performance with a third of training tokens, indicating potential for more efficient and effective biomedical pretraining strategies. 

---
# Narrative Shift Detection: A Hybrid Approach of Dynamic Topic Models and Large Language Models 

**Authors**: Kai-Robin Lange, Tobias Schmidt, Matthias Reccius, Henrik Müller, Michael Roos, Carsten Jentsch  

**Link**: [PDF](https://arxiv.org/pdf/2506.20269)  

**Abstract**: With rapidly evolving media narratives, it has become increasingly critical to not just extract narratives from a given corpus but rather investigate, how they develop over time. While popular narrative extraction methods such as Large Language Models do well in capturing typical narrative elements or even the complex structure of a narrative, applying them to an entire corpus comes with obstacles, such as a high financial or computational cost. We propose a combination of the language understanding capabilities of Large Language Models with the large scale applicability of topic models to dynamically model narrative shifts across time using the Narrative Policy Framework. We apply a topic model and a corresponding change point detection method to find changes that concern a specific topic of interest. Using this model, we filter our corpus for documents that are particularly representative of that change and feed them into a Large Language Model that interprets the change that happened in an automated fashion and distinguishes between content and narrative shifts. We employ our pipeline on a corpus of The Wall Street Journal news paper articles from 2009 to 2023. Our findings indicate that a Large Language Model can efficiently extract a narrative shift if one exists at a given point in time, but does not perform as well when having to decide whether a shift in content or a narrative shift took place. 

---
# CBF-AFA: Chunk-Based Multi-SSL Fusion for Automatic Fluency Assessment 

**Authors**: Papa Séga Wade, Mihai Andries, Ioannis Kanellos, Thierry Moudenc  

**Link**: [PDF](https://arxiv.org/pdf/2506.20243)  

**Abstract**: Automatic fluency assessment (AFA) remains challenging, particularly in capturing speech rhythm, pauses, and disfluencies in non-native speakers. We introduce a chunk-based approach integrating self-supervised learning (SSL) models (Wav2Vec2, HuBERT, and WavLM) selected for their complementary strengths in phonetic, prosodic, and noisy speech modeling, with a hierarchical CNN-BiLSTM framework. Speech is segmented into breath-group chunks using Silero voice activity detection (Silero-VAD), enabling fine-grained temporal analysis while mitigating over-segmentation artifacts. SSL embeddings are fused via a learnable weighted mechanism, balancing acoustic and linguistic features, and enriched with chunk-level fluency markers (e.g., speech rate, pause durations, n-gram repetitions). The CNN-BiLSTM captures local and long-term dependencies across chunks. Evaluated on Avalinguo and Speechocean762, our approach improves F1-score by 2.8 and Pearson correlation by 6.2 points over single SSL baselines on Speechocean762, with gains of 4.2 F1-score and 4.0 Pearson points on Avalinguo, surpassing this http URL-based segmentation baselines. These findings highlight chunk-based multi-SSL fusion for robust fluency evaluation, though future work should explore generalization to dialects with irregular prosody. 

---
# Enhancing Large Language Models through Structured Reasoning 

**Authors**: Yubo Dong, Hehe Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20241)  

**Abstract**: Recent Large Language Models (LLMs) have significantly advanced natural language processing and automated decision-making. However, these models still encounter difficulties when performing complex reasoning tasks involving logical deduction and systematic planning, primarily due to their reliance on implicit statistical relationships without structured knowledge this http URL by cognitive science and neurosymbolic AI, we introduce a novel approach to enhance LLMs through explicit structured reasoning. First, we convert unstructured data into structured formats by explicitly annotating reasoning steps. We then employ this structured dataset to train LLMs through Supervised Fine-Tuning (SFT). Additionally, we enhance the structured reasoning capabilities of LLMs using Group Relative Policy Optimization (GRPO), incorporating two innovative algorithms--MAX-Flow and Longest Common Subsequence (LCS)--which notably improve reasoning effectiveness and reduce computational complexity. Experimental results from fine-tuning a DeepSeek-R1-Distill-Qwen-1.5B model demonstrate concise reasoning, robust performance across various scenarios, and improved compatibility with optimization techniques, validating the efficacy of structured reasoning integration in LLMs. 

---
# Perspectives in Play: A Multi-Perspective Approach for More Inclusive NLP Systems 

**Authors**: Benedetta Muscato, Lucia Passaro, Gizem Gezici, Fosca Giannotti  

**Link**: [PDF](https://arxiv.org/pdf/2506.20209)  

**Abstract**: In the realm of Natural Language Processing (NLP), common approaches for handling human disagreement consist of aggregating annotators' viewpoints to establish a single ground truth. However, prior studies show that disregarding individual opinions can lead can lead to the side effect of underrepresenting minority perspectives, especially in subjective tasks, where annotators may systematically disagree because of their preferences. Recognizing that labels reflect the diverse backgrounds, life experiences, and values of individuals, this study proposes a new multi-perspective approach using soft labels to encourage the development of the next generation of perspective aware models, more inclusive and pluralistic. We conduct an extensive analysis across diverse subjective text classification tasks, including hate speech, irony, abusive language, and stance detection, to highlight the importance of capturing human disagreements, often overlooked by traditional aggregation methods. Results show that the multi-perspective approach not only better approximates human label distributions, as measured by Jensen-Shannon Divergence (JSD), but also achieves superior classification performance (higher F1 scores), outperforming traditional approaches. However, our approach exhibits lower confidence in tasks like irony and stance detection, likely due to the inherent subjectivity present in the texts. Lastly, leveraging Explainable AI (XAI), we explore model uncertainty and uncover meaningful insights into model predictions. 

---
# Intrinsic vs. Extrinsic Evaluation of Czech Sentence Embeddings: Semantic Relevance Doesn't Help with MT Evaluation 

**Authors**: Petra Barančíková, Ondřej Bojar  

**Link**: [PDF](https://arxiv.org/pdf/2506.20203)  

**Abstract**: In this paper, we compare Czech-specific and multilingual sentence embedding models through intrinsic and extrinsic evaluation paradigms. For intrinsic evaluation, we employ Costra, a complex sentence transformation dataset, and several Semantic Textual Similarity (STS) benchmarks to assess the ability of the embeddings to capture linguistic phenomena such as semantic similarity, temporal aspects, and stylistic variations. In the extrinsic evaluation, we fine-tune each embedding model using COMET-based metrics for machine translation evaluation.
Our experiments reveal an interesting disconnect: models that excel in intrinsic semantic similarity tests do not consistently yield superior performance on downstream translation evaluation tasks. Conversely, models with seemingly over-smoothed embedding spaces can, through fine-tuning, achieve excellent results. These findings highlight the complex relationship between semantic property probes and downstream task, emphasizing the need for more research into 'operationalizable semantics' in sentence embeddings, or more in-depth downstream tasks datasets (here translation evaluation) 

---
# How to Retrieve Examples in In-context Learning to Improve Conversational Emotion Recognition using Large Language Models? 

**Authors**: Mengqi Wang, Tiantian Feng, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20199)  

**Abstract**: Large language models (LLMs) have enabled a wide variety of real-world applications in various domains. However, creating a high-performing application with high accuracy remains challenging, particularly for subjective tasks like emotion recognition. Inspired by the SLT 2024 GenSER Challenge, this study investigates approaches to improving conversational emotion recognition (CER) by LLMs. Specifically, we explore how to retrieve high-quality examples in in-context learning (ICL) to enhance CER. We propose various strategies based on random and augmented example retrieval and also analyze the impact of conversational context on CER accuracy. Experiments were conducted on the three datasets including IEMOCAP, MELD and EmoryNLP. The results show that augmented example retrieval consistently outperforms other techniques under investigation across all datasets, highlighting the importance of retrieving coherent targeted examples and enhancing them through paraphrasing. 

---
# COIN: Uncertainty-Guarding Selective Question Answering for Foundation Models with Provable Risk Guarantees 

**Authors**: Zhiyuan Wang, Jinhao Duan, Qingni Wang, Xiaofeng Zhu, Tianlong Chen, Xiaoshuang Shi, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20178)  

**Abstract**: Uncertainty quantification (UQ) for foundation models is essential to identify and mitigate potential hallucinations in automatically generated text. However, heuristic UQ approaches lack formal guarantees for key metrics such as the false discovery rate (FDR) in selective prediction. Previous work adopts the split conformal prediction (SCP) framework to ensure desired coverage of admissible answers by constructing prediction sets, but these sets often contain incorrect candidates, limiting their practical utility. To address this, we propose COIN, an uncertainty-guarding selection framework that calibrates statistically valid thresholds to filter a single generated answer per question under user-specified FDR constraints. COIN estimates the empirical error rate on a calibration set and applies confidence interval methods such as Clopper-Pearson to establish a high-probability upper bound on the true error rate (i.e., FDR). This enables the selection of the largest uncertainty threshold that ensures FDR control on test data while significantly increasing sample retention. We demonstrate COIN's robustness in risk control, strong test-time power in retaining admissible answers, and predictive efficiency under limited calibration data across both general and multimodal text generation tasks. Furthermore, we show that employing alternative upper bound constructions and UQ strategies can further boost COIN's power performance, which underscores its extensibility and adaptability to diverse application scenarios. 

---
# SEED: A Structural Encoder for Embedding-Driven Decoding in Time Series Prediction with LLMs 

**Authors**: Fengze Li, Yue Wang, Yangle Liu, Ming Huang, Dou Hong, Jieming Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.20167)  

**Abstract**: Multivariate time series forecasting requires models to simultaneously capture variable-wise structural dependencies and generalize across diverse tasks. While structural encoders are effective in modeling feature interactions, they lack the capacity to support semantic-level reasoning or task adaptation. Conversely, large language models (LLMs) possess strong generalization capabilities but remain incompatible with raw time series inputs. This gap limits the development of unified, transferable prediction systems. Therefore, we introduce SEED, a structural encoder for embedding-driven decoding, which integrates four stages: a token-aware encoder for patch extraction, a projection module that aligns patches with language model embeddings, a semantic reprogramming mechanism that maps patches to task-aware prototypes, and a frozen language model for prediction. This modular architecture decouples representation learning from inference, enabling efficient alignment between numerical patterns and semantic reasoning. Empirical results demonstrate that the proposed method achieves consistent improvements over strong baselines, and comparative studies on various datasets confirm SEED's role in addressing the structural-semantic modeling gap. 

---
# AALC: Large Language Model Efficient Reasoning via Adaptive Accuracy-Length Control 

**Authors**: Ruosen Li, Ziming Luo, Quan Zhang, Ruochen Li, Ben Zhou, Ali Payani, Xinya Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.20160)  

**Abstract**: Large reasoning models (LRMs) achieve impressive reasoning capabilities by generating lengthy chain-of-thoughts, but this "overthinking" incurs high latency and cost without commensurate accuracy gains. In this work, we introduce AALC, a lightweight, accuracy-aware length reward integrated into reinforcement learning that dynamically balances correctness and brevity during training. By incorporating validation accuracy into the reward and employing a smooth, dynamically scheduled length penalty, AALC delays length penalty until target performance is met. Through extensive experiments across standard and out-of-distribution math benchmarks, we show that our approach reduces response length by over 50% while maintaining or even improving the original accuracy. Furthermore, qualitative analysis reveals that our method curbs redundant reasoning patterns such as excessive subgoal setting and verification, leading to structurally refined outputs rather than naive truncation. We also identify that efficiency gains are accompanied by reduced interpretability: models trained with AALC omit some narrative framing and explanatory context. These findings highlight the potential of reward-based strategies to guide LRMs toward more efficient, generalizable reasoning paths. 

---
# CCRS: A Zero-Shot LLM-as-a-Judge Framework for Comprehensive RAG Evaluation 

**Authors**: Aashiq Muhamed  

**Link**: [PDF](https://arxiv.org/pdf/2506.20128)  

**Abstract**: RAG systems enhance LLMs by incorporating external knowledge, which is crucial for domains that demand factual accuracy and up-to-date information. However, evaluating the multifaceted quality of RAG outputs, spanning aspects such as contextual coherence, query relevance, factual correctness, and informational completeness, poses significant challenges. Existing evaluation methods often rely on simple lexical overlap metrics, which are inadequate for capturing these nuances, or involve complex multi-stage pipelines with intermediate steps like claim extraction or require finetuning specialized judge models, hindering practical efficiency. To address these limitations, we propose CCRS (Contextual Coherence and Relevance Score), a novel suite of five metrics that utilizes a single, powerful, pretrained LLM as a zero-shot, end-to-end judge. CCRS evaluates: Contextual Coherence (CC), Question Relevance (QR), Information Density (ID), Answer Correctness (AC), and Information Recall (IR). We apply CCRS to evaluate six diverse RAG system configurations on the challenging BioASQ dataset. Our analysis demonstrates that CCRS effectively discriminates between system performances, confirming, for instance, that the Mistral-7B reader outperforms Llama variants. We provide a detailed analysis of CCRS metric properties, including score distributions, convergent/discriminant validity, tie rates, population statistics, and discriminative power. Compared to the complex RAGChecker framework, CCRS offers comparable or superior discriminative power for key aspects like recall and faithfulness, while being significantly more computationally efficient. CCRS thus provides a practical, comprehensive, and efficient framework for evaluating and iteratively improving RAG systems. 

---
# Leveraging AI Graders for Missing Score Imputation to Achieve Accurate Ability Estimation in Constructed-Response Tests 

**Authors**: Masaki Uto, Yuma Ito  

**Link**: [PDF](https://arxiv.org/pdf/2506.20119)  

**Abstract**: Evaluating the abilities of learners is a fundamental objective in the field of education. In particular, there is an increasing need to assess higher-order abilities such as expressive skills and logical thinking. Constructed-response tests such as short-answer and essay-based questions have become widely used as a method to meet this demand. Although these tests are effective, they require substantial manual grading, making them both labor-intensive and costly. Item response theory (IRT) provides a promising solution by enabling the estimation of ability from incomplete score data, where human raters grade only a subset of answers provided by learners across multiple test items. However, the accuracy of ability estimation declines as the proportion of missing scores increases. Although data augmentation techniques for imputing missing scores have been explored in order to address this limitation, they often struggle with inaccuracy for sparse or heterogeneous data. To overcome these challenges, this study proposes a novel method for imputing missing scores by leveraging automated scoring technologies for accurate IRT-based ability estimation. The proposed method achieves high accuracy in ability estimation while markedly reducing manual grading workload. 

---
# A Multi-Pass Large Language Model Framework for Precise and Efficient Radiology Report Error Detection 

**Authors**: Songsoo Kim, Seungtae Lee, See Young Lee, Joonho Kim, Keechan Kan, Dukyong Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2506.20112)  

**Abstract**: Background: The positive predictive value (PPV) of large language model (LLM)-based proofreading for radiology reports is limited due to the low error prevalence. Purpose: To assess whether a three-pass LLM framework enhances PPV and reduces operational costs compared with baseline approaches. Materials and Methods: A retrospective analysis was performed on 1,000 consecutive radiology reports (250 each: radiography, ultrasonography, CT, MRI) from the MIMIC-III database. Two external datasets (CheXpert and Open-i) were validation sets. Three LLM frameworks were tested: (1) single-prompt detector; (2) extractor plus detector; and (3) extractor, detector, and false-positive verifier. Precision was measured by PPV and absolute true positive rate (aTPR). Efficiency was calculated from model inference charges and reviewer remuneration. Statistical significance was tested using cluster bootstrap, exact McNemar tests, and Holm-Bonferroni correction. Results: Framework PPV increased from 0.063 (95% CI, 0.036-0.101, Framework 1) to 0.079 (0.049-0.118, Framework 2), and significantly to 0.159 (0.090-0.252, Framework 3; P<.001 vs. baselines). aTPR remained stable (0.012-0.014; P>=.84). Operational costs per 1,000 reports dropped to USD 5.58 (Framework 3) from USD 9.72 (Framework 1) and USD 6.85 (Framework 2), reflecting reductions of 42.6% and 18.5%, respectively. Human-reviewed reports decreased from 192 to 88. External validation supported Framework 3's superior PPV (CheXpert 0.133, Open-i 0.105) and stable aTPR (0.007). Conclusion: A three-pass LLM framework significantly enhanced PPV and reduced operational costs, maintaining detection performance, providing an effective strategy for AI-assisted radiology report quality assurance. 

---
# MIRAGE: A Benchmark for Multimodal Information-Seeking and Reasoning in Agricultural Expert-Guided Conversations 

**Authors**: Vardhan Dongre, Chi Gui, Shubham Garg, Hooshang Nayyeri, Gokhan Tur, Dilek Hakkani-Tür, Vikram S. Adve  

**Link**: [PDF](https://arxiv.org/pdf/2506.20100)  

**Abstract**: We introduce MIRAGE, a new benchmark for multimodal expert-level reasoning and decision-making in consultative interaction settings. Designed for the agriculture domain, MIRAGE captures the full complexity of expert consultations by combining natural user queries, expert-authored responses, and image-based context, offering a high-fidelity benchmark for evaluating models on grounded reasoning, clarification strategies, and long-form generation in a real-world, knowledge-intensive domain. Grounded in over 35,000 real user-expert interactions and curated through a carefully designed multi-step pipeline, MIRAGE spans diverse crop health, pest diagnosis, and crop management scenarios. The benchmark includes more than 7,000 unique biological entities, covering plant species, pests, and diseases, making it one of the most taxonomically diverse benchmarks available for vision-language models, grounded in the real world. Unlike existing benchmarks that rely on well-specified user inputs and closed-set taxonomies, MIRAGE features underspecified, context-rich scenarios with open-world settings, requiring models to infer latent knowledge gaps, handle rare entities, and either proactively guide the interaction or respond. Project Page: this https URL 

---
# ITFormer: Bridging Time Series and Natural Language for Multi-Modal QA with Large-Scale Multitask Dataset 

**Authors**: Yilin Wang, Peixuan Lei, Jie Song, Yuzhe Hao, Tao Chen, Yuxuan Zhang, Lei Jia, Yuanxiang Li, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.20093)  

**Abstract**: Time-series data are critical in diverse applications, such as industrial monitoring, medical diagnostics, and climate research. However, effectively integrating these high-dimensional temporal signals with natural language for dynamic, interactive tasks remains a significant challenge. To address this, we introduce the Time-Series Question Answering (Time-Series QA) task and release EngineMT-QA, the first large-scale, multi-task, temporal-textual QA dataset designed to capture complex interactions between time-series signals and natural language. Building on this resource, we propose the Instruct Time Transformer (ITFormer), a novel framework that bridges time-series encoders with frozen large language models (LLMs). ITFormer effectively extracts, aligns, and fuses temporal and textual features, achieving a strong improvement in QA accuracy over strong baselines with fewer than 1\% additional trainable parameters. By combining computational efficiency with robust cross-modal modeling, our work establishes a adaptable paradigm for integrating temporal data with natural language, paving the way for new research and applications in multi-modal AI. More details about the project, including datasets and code, are available at: this https URL 

---
# Bridging Compositional and Distributional Semantics: A Survey on Latent Semantic Geometry via AutoEncoder 

**Authors**: Yingji Zhang, Danilo S. Carvalho, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2506.20083)  

**Abstract**: Integrating compositional and symbolic properties into current distributional semantic spaces can enhance the interpretability, controllability, compositionality, and generalisation capabilities of Transformer-based auto-regressive language models (LMs). In this survey, we offer a novel perspective on latent space geometry through the lens of compositional semantics, a direction we refer to as \textit{semantic representation learning}. This direction enables a bridge between symbolic and distributional semantics, helping to mitigate the gap between them. We review and compare three mainstream autoencoder architectures-Variational AutoEncoder (VAE), Vector Quantised VAE (VQVAE), and Sparse AutoEncoder (SAE)-and examine the distinctive latent geometries they induce in relation to semantic structure and interpretability. 

---
# SACL: Understanding and Combating Textual Bias in Code Retrieval with Semantic-Augmented Reranking and Localization 

**Authors**: Dhruv Gupta, Gayathri Ganesh Lakshmy, Yiqing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.20081)  

**Abstract**: Retrieval-Augmented Code Generation (RACG) is a critical technique for enhancing code generation by retrieving relevant information. In this work, we conduct an in-depth analysis of code retrieval by systematically masking specific features while preserving code functionality. Our discoveries include: (1) although trained on code, current retrievers heavily rely on surface-level textual features (e.g., docstrings, identifier names), and (2) they exhibit a strong bias towards well-documented code, even if the documentation is this http URL on our discoveries, we propose SACL, a framework that enriches textual information and reduces bias by augmenting code or structural knowledge with semantic information. Extensive experiments show that SACL substantially improves code retrieval (e.g., by 12.8% / 9.4% / 7.0% Recall@1 on HumanEval / MBPP / SWE-Bench-Lite), which also leads to better code generation performance (e.g., by 4.88% Pass@1 on HumanEval). 

---
# A Modular Multitask Reasoning Framework Integrating Spatio-temporal Models and LLMs 

**Authors**: Kethmi Hirushini Hettige, Jiahao Ji, Cheng Long, Shili Xiang, Gao Cong, Jingyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20073)  

**Abstract**: Spatio-temporal data mining plays a pivotal role in informed decision making across diverse domains. However, existing models are often restricted to narrow tasks, lacking the capacity for multi-task inference and complex long-form reasoning that require generation of in-depth, explanatory outputs. These limitations restrict their applicability to real-world, multi-faceted decision scenarios. In this work, we introduce STReason, a novel framework that integrates the reasoning strengths of large language models (LLMs) with the analytical capabilities of spatio-temporal models for multi-task inference and execution. Without requiring task-specific finetuning, STReason leverages in-context learning to decompose complex natural language queries into modular, interpretable programs, which are then systematically executed to generate both solutions and detailed rationales. To facilitate rigorous evaluation, we construct a new benchmark dataset and propose a unified evaluation framework with metrics specifically designed for long-form spatio-temporal reasoning. Experimental results show that STReason significantly outperforms advanced LLM baselines across all metrics, particularly excelling in complex, reasoning-intensive spatio-temporal scenarios. Human evaluations further validate STReason's credibility and practical utility, demonstrating its potential to reduce expert workload and broaden the applicability to real-world spatio-temporal tasks. We believe STReason provides a promising direction for developing more capable and generalizable spatio-temporal reasoning systems. 

---
# A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior 

**Authors**: Francesco Ignazio Re, Andreas Opedal, Glib Manaiev, Mario Giulianelli, Ryan Cotterell  

**Link**: [PDF](https://arxiv.org/pdf/2506.19999)  

**Abstract**: Reading is a process that unfolds across space and time, alternating between fixations where a reader focuses on a specific point in space, and saccades where a reader rapidly shifts their focus to a new point. An ansatz of psycholinguistics is that modeling a reader's fixations and saccades yields insight into their online sentence processing. However, standard approaches to such modeling rely on aggregated eye-tracking measurements and models that impose strong assumptions, ignoring much of the spatio-temporal dynamics that occur during reading. In this paper, we propose a more general probabilistic model of reading behavior, based on a marked spatio-temporal point process, that captures not only how long fixations last, but also where they land in space and when they take place in time. The saccades are modeled using a Hawkes process, which captures how each fixation excites the probability of a new fixation occurring near it in time and space. The duration time of fixation events is modeled as a function of fixation-specific predictors convolved across time, thus capturing spillover effects. Empirically, our Hawkes process model exhibits a better fit to human saccades than baselines. With respect to fixation durations, we observe that incorporating contextual surprisal as a predictor results in only a marginal improvement in the model's predictive accuracy. This finding suggests that surprisal theory struggles to explain fine-grained eye movements. 

---
# Doc2Agent: Scalable Generation of Tool-Using Agents from API Documentation 

**Authors**: Xinyi Ni, Haonan Jian, Qiuyang Wang, Vedanshi Chetan Shah, Pengyu Hong  

**Link**: [PDF](https://arxiv.org/pdf/2506.19998)  

**Abstract**: REST APIs play important roles in enriching the action space of web agents, yet most API-based agents rely on curated and uniform toolsets that do not reflect the complexity of real-world APIs. Building tool-using agents for arbitrary domains remains a major challenge, as it requires reading unstructured API documentation, testing APIs and inferring correct parameters. We propose Doc2Agent, a scalable pipeline to build agents that can call Python-based tools generated from API documentation. Doc2Agent generates executable tools from API documentations and iteratively refines them using a code agent. We evaluate our approach on real-world APIs, WebArena APIs, and research APIs, producing validated tools. We achieved a 55\% relative performance improvement with 90\% lower cost compared to direct API calling on WebArena benchmark. A domain-specific agent built for glycomaterial science further demonstrates the pipeline's adaptability to complex, knowledge-rich tasks. Doc2Agent offers a generalizable solution for building tool agents from unstructured API documentation at scale. 

---
# Inference Scaled GraphRAG: Improving Multi Hop Question Answering on Knowledge Graphs 

**Authors**: Travis Thompson, Seung-Hwan Lim, Paul Liu, Ruoying He, Dongkuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19967)  

**Abstract**: Large Language Models (LLMs) have achieved impressive capabilities in language understanding and generation, yet they continue to underperform on knowledge-intensive reasoning tasks due to limited access to structured context and multi-hop information. Retrieval-Augmented Generation (RAG) partially mitigates this by grounding generation in retrieved context, but conventional RAG and GraphRAG methods often fail to capture relational structure across nodes in knowledge graphs. We introduce Inference-Scaled GraphRAG, a novel framework that enhances LLM-based graph reasoning by applying inference-time compute scaling. Our method combines sequential scaling with deep chain-of-thought graph traversal, and parallel scaling with majority voting over sampled trajectories within an interleaved reasoning-execution loop. Experiments on the GRBench benchmark demonstrate that our approach significantly improves multi-hop question answering performance, achieving substantial gains over both traditional GraphRAG and prior graph traversal baselines. These findings suggest that inference-time scaling is a practical and architecture-agnostic solution for structured knowledge reasoning with LLMs 

---
# CycleDistill: Bootstrapping Machine Translation using LLMs with Cyclical Distillation 

**Authors**: Deepon Halder, Thanmay Jayakumar, Raj Dabre  

**Link**: [PDF](https://arxiv.org/pdf/2506.19952)  

**Abstract**: Large language models (LLMs), despite their ability to perform few-shot machine translation (MT), often lag behind dedicated MT systems trained on parallel corpora, which are crucial for high quality machine translation (MT). However, parallel corpora are often scarce or non-existent for low-resource languages. In this paper, we propose CycleDistill, a bootstrapping approach leveraging LLMs and few-shot translation to obtain high-quality MT systems. CycleDistill involves iteratively generating synthetic parallel corpora from monolingual corpora via zero- or few-shot MT, which is then used to fine-tune the model that was used for generating said data for MT. CycleDistill does not need parallel corpora beyond 1 to 4 few-shot examples, and in our experiments focusing on three Indian languages, by relying solely on monolingual corpora, it can achieve high-quality machine translation, improving upon a few-shot baseline model by over 20-30 chrF points on average in the first iteration. We also study the effect of leveraging softmax activations during the distillation process and observe mild improvements in translation quality. 

---
# MMSearch-R1: Incentivizing LMMs to Search 

**Authors**: Jinming Wu, Zihao Deng, Wei Li, Yiding Liu, Bo You, Bo Li, Zejun Ma, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20670)  

**Abstract**: Robust deployment of large multimodal models (LMMs) in real-world scenarios requires access to external knowledge sources, given the complexity and dynamic nature of real-world information. Existing approaches such as retrieval-augmented generation (RAG) and prompt engineered search agents rely on rigid pipelines, often leading to inefficient or excessive search behaviors. We present MMSearch-R1, the first end-to-end reinforcement learning framework that enables LMMs to perform on-demand, multi-turn search in real-world Internet environments. Our framework integrates both image and text search tools, allowing the model to reason about when and how to invoke them guided by an outcome-based reward with a search penalty. To support training, We collect a multimodal search VQA dataset through a semi-automated pipeline that covers diverse visual and textual knowledge needs and curate a search-balanced subset with both search-required and search-free samples, which proves essential for shaping efficient and on-demand search behavior. Extensive experiments on knowledge-intensive and info-seeking VQA tasks show that our model not only outperforms RAG-based baselines of the same model size, but also matches the performance of a larger RAG-based model while reducing search calls by over 30%. We further analyze key empirical findings to offer actionable insights for advancing research in multimodal search. 

---
# The Decrypto Benchmark for Multi-Agent Reasoning and Theory of Mind 

**Authors**: Andrei Lupu, Timon Willi, Jakob Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2506.20664)  

**Abstract**: As Large Language Models (LLMs) gain agentic abilities, they will have to navigate complex multi-agent scenarios, interacting with human users and other agents in cooperative and competitive settings. This will require new reasoning skills, chief amongst them being theory of mind (ToM), or the ability to reason about the "mental" states of other agents. However, ToM and other multi-agent abilities in LLMs are poorly understood, since existing benchmarks suffer from narrow scope, data leakage, saturation, and lack of interactivity. We thus propose Decrypto, a game-based benchmark for multi-agent reasoning and ToM drawing inspiration from cognitive science, computational pragmatics and multi-agent reinforcement learning. It is designed to be as easy as possible in all other dimensions, eliminating confounding factors commonly found in other benchmarks. To our knowledge, it is also the first platform for designing interactive ToM experiments.
We validate the benchmark design through comprehensive empirical evaluations of frontier LLMs, robustness studies, and human-AI cross-play experiments. We find that LLM game-playing abilities lag behind humans and simple word-embedding baselines. We then create variants of two classic cognitive science experiments within Decrypto to evaluate three key ToM abilities. Surprisingly, we find that state-of-the-art reasoning models are significantly worse at those tasks than their older counterparts. This demonstrates that Decrypto addresses a crucial gap in current reasoning and ToM evaluations, and paves the path towards better artificial agents. 

---
# PLoP: Precise LoRA Placement for Efficient Finetuning of Large Models 

**Authors**: Soufiane Hayou, Nikhil Ghosh, Bin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20629)  

**Abstract**: Low-Rank Adaptation (LoRA) is a widely used finetuning method for large models. Its small memory footprint allows practitioners to adapt large models to specific tasks at a fraction of the cost of full finetuning. Different modifications have been proposed to enhance its efficiency by, for example, setting the learning rate, the rank, and the initialization. Another improvement axis is adapter placement strategy: when using LoRA, practitioners usually pick module types to adapt with LoRA, such as Query and Key modules. Few works have studied the problem of adapter placement, with nonconclusive results: original LoRA paper suggested placing adapters in attention modules, while other works suggested placing them in the MLP modules. Through an intuitive theoretical analysis, we introduce PLoP (Precise LoRA Placement), a lightweight method that allows automatic identification of module types where LoRA adapters should be placed, given a pretrained model and a finetuning task. We demonstrate that PLoP consistently outperforms, and in the worst case competes, with commonly used placement strategies through comprehensive experiments on supervised finetuning and reinforcement learning for reasoning. 

---
# Asymmetric REINFORCE for off-Policy Reinforcement Learning: Balancing positive and negative rewards 

**Authors**: Charles Arnal, Gaëtan Narozniak, Vivien Cabannes, Yunhao Tang, Julia Kempe, Remi Munos  

**Link**: [PDF](https://arxiv.org/pdf/2506.20520)  

**Abstract**: Reinforcement learning (RL) is increasingly used to align large language models (LLMs). Off-policy methods offer greater implementation simplicity and data efficiency than on-policy techniques, but often result in suboptimal performance. In this work, we study the intermediate range of algorithms between off-policy RL and supervised fine-tuning by analyzing a simple off-policy REINFORCE algorithm, where the advantage is defined as $A=r-V$, with $r$ a reward and $V$ some tunable baseline. Intuitively, lowering $V$ emphasizes high-reward samples, while raising it penalizes low-reward ones more heavily. We first provide a theoretical analysis of this off-policy REINFORCE algorithm, showing that when the baseline $V$ lower-bounds the expected reward, the algorithm enjoys a policy improvement guarantee. Our analysis reveals that while on-policy updates can safely leverage both positive and negative signals, off-policy updates benefit from focusing more on positive rewards than on negative ones. We validate our findings experimentally in a controlled stochastic bandit setting and through fine-tuning state-of-the-art LLMs on reasoning tasks. 

---
# Counterfactual Influence as a Distributional Quantity 

**Authors**: Matthieu Meeus, Igor Shilov, Georgios Kaissis, Yves-Alexandre de Montjoye  

**Link**: [PDF](https://arxiv.org/pdf/2506.20481)  

**Abstract**: Machine learning models are known to memorize samples from their training data, raising concerns around privacy and generalization. Counterfactual self-influence is a popular metric to study memorization, quantifying how the model's prediction for a sample changes depending on the sample's inclusion in the training dataset. However, recent work has shown memorization to be affected by factors beyond self-influence, with other training samples, in particular (near-)duplicates, having a large impact. We here study memorization treating counterfactual influence as a distributional quantity, taking into account how all training samples influence how a sample is memorized. For a small language model, we compute the full influence distribution of training samples on each other and analyze its properties. We find that solely looking at self-influence can severely underestimate tangible risks associated with memorization: the presence of (near-)duplicates seriously reduces self-influence, while we find these samples to be (near-)extractable. We observe similar patterns for image classification, where simply looking at the influence distributions reveals the presence of near-duplicates in CIFAR-10. Our findings highlight that memorization stems from complex interactions across training data and is better captured by the full influence distribution than by self-influence alone. 

---
# From Codicology to Code: A Comparative Study of Transformer and YOLO-based Detectors for Layout Analysis in Historical Documents 

**Authors**: Sergio Torres Aguilar  

**Link**: [PDF](https://arxiv.org/pdf/2506.20326)  

**Abstract**: Robust Document Layout Analysis (DLA) is critical for the automated processing and understanding of historical documents with complex page organizations. This paper benchmarks five state-of-the-art object detection architectures on three annotated datasets representing a spectrum of codicological complexity: The e-NDP, a corpus of Parisian medieval registers (1326-1504); CATMuS, a diverse multiclass dataset derived from various medieval and modern sources (ca.12th-17th centuries) and HORAE, a corpus of decorated books of hours (ca.13th-16th centuries). We evaluate two Transformer-based models (Co-DETR, Grounding DINO) against three YOLO variants (AABB, OBB, and YOLO-World). Our findings reveal significant performance variations dependent on model architecture, data set characteristics, and bounding box representation. In the e-NDP dataset, Co-DETR achieves state-of-the-art results (0.752 mAP@.50:.95), closely followed by YOLOv11X-OBB (0.721). Conversely, on the more complex CATMuS and HORAE datasets, the CNN-based YOLOv11x-OBB significantly outperforms all other models (0.564 and 0.568, respectively). This study unequivocally demonstrates that using Oriented Bounding Boxes (OBB) is not a minor refinement but a fundamental requirement for accurately modeling the non-Cartesian nature of historical manuscripts. We conclude that a key trade-off exists between the global context awareness of Transformers, ideal for structured layouts, and the superior generalization of CNN-OBB models for visually diverse and complex documents. 

---
# FundaQ-8: A Clinically-Inspired Scoring Framework for Automated Fundus Image Quality Assessment 

**Authors**: Lee Qi Zun, Oscar Wong Jin Hao, Nor Anita Binti Che Omar, Zalifa Zakiah Binti Asnir, Mohamad Sabri bin Sinal Zainal, Goh Man Fye  

**Link**: [PDF](https://arxiv.org/pdf/2506.20303)  

**Abstract**: Automated fundus image quality assessment (FIQA) remains a challenge due to variations in image acquisition and subjective expert evaluations. We introduce FundaQ-8, a novel expert-validated framework for systematically assessing fundus image quality using eight critical parameters, including field coverage, anatomical visibility, illumination, and image artifacts. Using FundaQ-8 as a structured scoring reference, we develop a ResNet18-based regression model to predict continuous quality scores in the 0 to 1 range. The model is trained on 1800 fundus images from real-world clinical sources and Kaggle datasets, using transfer learning, mean squared error optimization, and standardized preprocessing. Validation against the EyeQ dataset and statistical analyses confirm the framework's reliability and clinical interpretability. Incorporating FundaQ-8 into deep learning models for diabetic retinopathy grading also improves diagnostic robustness, highlighting the value of quality-aware training in real-world screening applications. 

---
# Why Robots Are Bad at Detecting Their Mistakes: Limitations of Miscommunication Detection in Human-Robot Dialogue 

**Authors**: Ruben Janssens, Jens De Bock, Sofie Labat, Eva Verhelst, Veronique Hoste, Tony Belpaeme  

**Link**: [PDF](https://arxiv.org/pdf/2506.20268)  

**Abstract**: Detecting miscommunication in human-robot interaction is a critical function for maintaining user engagement and trust. While humans effortlessly detect communication errors in conversations through both verbal and non-verbal cues, robots face significant challenges in interpreting non-verbal feedback, despite advances in computer vision for recognizing affective expressions. This research evaluates the effectiveness of machine learning models in detecting miscommunications in robot dialogue. Using a multi-modal dataset of 240 human-robot conversations, where four distinct types of conversational failures were systematically introduced, we assess the performance of state-of-the-art computer vision models. After each conversational turn, users provided feedback on whether they perceived an error, enabling an analysis of the models' ability to accurately detect robot mistakes. Despite using state-of-the-art models, the performance barely exceeds random chance in identifying miscommunication, while on a dataset with more expressive emotional content, they successfully identified confused states. To explore the underlying cause, we asked human raters to do the same. They could also only identify around half of the induced miscommunications, similarly to our model. These results uncover a fundamental limitation in identifying robot miscommunications in dialogue: even when users perceive the induced miscommunication as such, they often do not communicate this to their robotic conversation partner. This knowledge can shape expectations of the performance of computer vision models and can help researchers to design better human-robot conversations by deliberately eliciting feedback where needed. 

---
# Language Modeling by Language Models 

**Authors**: Junyan Cheng, Peter Clark, Kyle Richardson  

**Link**: [PDF](https://arxiv.org/pdf/2506.20249)  

**Abstract**: Can we leverage LLMs to model the process of discovering novel language model (LM) architectures? Inspired by real research, we propose a multi-agent LLM approach that simulates the conventional stages of research, from ideation and literature search (proposal stage) to design implementation (code generation), generative pre-training, and downstream evaluation (verification). Using ideas from scaling laws, our system, Genesys, employs a Ladder of Scales approach; new designs are proposed, adversarially reviewed, implemented, and selectively verified at increasingly larger model scales (14M$\sim$350M parameters) with a narrowing budget (the number of models we can train at each scale). To help make discovery efficient and factorizable, Genesys uses a novel genetic programming backbone, which we show has empirical advantages over commonly used direct prompt generation workflows (e.g., $\sim$86\% percentage point improvement in successful design generation, a key bottleneck). We report experiments involving 1,162 newly discovered designs (1,062 fully verified through pre-training) and find the best designs to be highly competitive with known architectures (e.g., outperform GPT2, Mamba2, etc., on 6/9 common benchmarks). We couple these results with comprehensive system-level ablations and formal results, which give broader insights into the design of effective autonomous discovery systems. 

---
# PSALM-V: Automating Symbolic Planning in Interactive Visual Environments with Large Language Models 

**Authors**: Wang Bill Zhu, Miaosen Chai, Ishika Singh, Robin Jia, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2506.20097)  

**Abstract**: We propose PSALM-V, the first autonomous neuro-symbolic learning system able to induce symbolic action semantics (i.e., pre- and post-conditions) in visual environments through interaction. PSALM-V bootstraps reliable symbolic planning without expert action definitions, using LLMs to generate heuristic plans and candidate symbolic semantics. Previous work has explored using large language models to generate action semantics for Planning Domain Definition Language (PDDL)-based symbolic planners. However, these approaches have primarily focused on text-based domains or relied on unrealistic assumptions, such as access to a predefined problem file, full observability, or explicit error messages. By contrast, PSALM-V dynamically infers PDDL problem files and domain action semantics by analyzing execution outcomes and synthesizing possible error explanations. The system iteratively generates and executes plans while maintaining a tree-structured belief over possible action semantics for each action, iteratively refining these beliefs until a goal state is reached. Simulated experiments of task completion in ALFRED demonstrate that PSALM-V increases the plan success rate from 37% (Claude-3.7) to 74% in partially observed setups. Results on two 2D game environments, RTFM and Overcooked-AI, show that PSALM-V improves step efficiency and succeeds in domain induction in multi-agent settings. PSALM-V correctly induces PDDL pre- and post-conditions for real-world robot BlocksWorld tasks, despite low-level manipulation failures from the robot. 

---
# Persona-Assigned Large Language Models Exhibit Human-Like Motivated Reasoning 

**Authors**: Saloni Dash, Amélie Reymond, Emma S. Spiro, Aylin Caliskan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20020)  

**Abstract**: Reasoning in humans is prone to biases due to underlying motivations like identity protection, that undermine rational decision-making and judgment. This motivated reasoning at a collective level can be detrimental to society when debating critical issues such as human-driven climate change or vaccine safety, and can further aggravate political polarization. Prior studies have reported that large language models (LLMs) are also susceptible to human-like cognitive biases, however, the extent to which LLMs selectively reason toward identity-congruent conclusions remains largely unexplored. Here, we investigate whether assigning 8 personas across 4 political and socio-demographic attributes induces motivated reasoning in LLMs. Testing 8 LLMs (open source and proprietary) across two reasoning tasks from human-subject studies -- veracity discernment of misinformation headlines and evaluation of numeric scientific evidence -- we find that persona-assigned LLMs have up to 9% reduced veracity discernment relative to models without personas. Political personas specifically, are up to 90% more likely to correctly evaluate scientific evidence on gun control when the ground truth is congruent with their induced political identity. Prompt-based debiasing methods are largely ineffective at mitigating these effects. Taken together, our empirical findings are the first to suggest that persona-assigned LLMs exhibit human-like motivated reasoning that is hard to mitigate through conventional debiasing prompts -- raising concerns of exacerbating identity-congruent reasoning in both LLMs and humans. 

---
# Accurate and Energy Efficient: Local Retrieval-Augmented Generation Models Outperform Commercial Large Language Models in Medical Tasks 

**Authors**: Konstantinos Vrettos, Michail E. Klontzas  

**Link**: [PDF](https://arxiv.org/pdf/2506.20009)  

**Abstract**: Background The increasing adoption of Artificial Intelligence (AI) in healthcare has sparked growing concerns about its environmental and ethical implications. Commercial Large Language Models (LLMs), such as ChatGPT and DeepSeek, require substantial resources, while the utilization of these systems for medical purposes raises critical issues regarding patient privacy and safety. Methods We developed a customizable Retrieval-Augmented Generation (RAG) framework for medical tasks, which monitors its energy usage and CO2 emissions. This system was then used to create RAGs based on various open-source LLMs. The tested models included both general purpose models like llama3.1:8b and medgemma-4b-it, which is medical-domain specific. The best RAGs performance and energy consumption was compared to DeepSeekV3-R1 and OpenAIs o4-mini model. A dataset of medical questions was used for the evaluation. Results Custom RAG models outperformed commercial models in accuracy and energy consumption. The RAG model built on llama3.1:8B achieved the highest accuracy (58.5%) and was significantly better than other models, including o4-mini and DeepSeekV3-R1. The llama3.1-RAG also exhibited the lowest energy consumption and CO2 footprint among all models, with a Performance per kWh of 0.52 and a total CO2 emission of 473g. Compared to o4-mini, the llama3.1-RAG achieved 2.7x times more accuracy points per kWh and 172% less electricity usage while maintaining higher accuracy. Conclusion Our study demonstrates that local LLMs can be leveraged to develop RAGs that outperform commercial, online LLMs in medical tasks, while having a smaller environmental impact. Our modular framework promotes sustainable AI development, reducing electricity usage and aligning with the UNs Sustainable Development Goals. 

---
# Position: Machine Learning Conferences Should Establish a "Refutations and Critiques" Track 

**Authors**: Rylan Schaeffer, Joshua Kazdan, Yegor Denisov-Blanch, Brando Miranda, Matthias Gerstgrasser, Susan Zhang, Andreas Haupt, Isha Gupta, Elyas Obbad, Jesse Dodge, Jessica Zosa Forde, Koustuv Sinha, Francesco Orabona, Sanmi Koyejo, David Donoho  

**Link**: [PDF](https://arxiv.org/pdf/2506.19882)  

**Abstract**: Science progresses by iteratively advancing and correcting humanity's understanding of the world. In machine learning (ML) research, rapid advancements have led to an explosion of publications, but have also led to misleading, incorrect, flawed or perhaps even fraudulent studies being accepted and sometimes highlighted at ML conferences due to the fallibility of peer review. While such mistakes are understandable, ML conferences do not offer robust processes to help the field systematically correct when such errors are this http URL position paper argues that ML conferences should establish a dedicated "Refutations and Critiques" (R & C) Track. This R & C Track would provide a high-profile, reputable platform to support vital research that critically challenges prior research, thereby fostering a dynamic self-correcting research ecosystem. We discuss key considerations including track design, review principles, potential pitfalls, and provide an illustrative example submission concerning a recent ICLR 2025 Oral. We conclude that ML conferences should create official, reputable mechanisms to help ML research self-correct. 

---
# Capturing Visualization Design Rationale 

**Authors**: Maeve Hutchinson, Radu Jianu, Aidan Slingsby, Jo Wood, Pranava Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2506.16571)  

**Abstract**: Prior natural language datasets for data visualization have focused on tasks such as visualization literacy assessment, insight generation, and visualization generation from natural language instructions. These studies often rely on controlled setups with purpose-built visualizations and artificially constructed questions. As a result, they tend to prioritize the interpretation of visualizations, focusing on decoding visualizations rather than understanding their encoding. In this paper, we present a new dataset and methodology for probing visualization design rationale through natural language. We leverage a unique source of real-world visualizations and natural language narratives: literate visualization notebooks created by students as part of a data visualization course. These notebooks combine visual artifacts with design exposition, in which students make explicit the rationale behind their design decisions. We also use large language models (LLMs) to generate and categorize question-answer-rationale triples from the narratives and articulations in the notebooks. We then carefully validate the triples and curate a dataset that captures and distills the visualization design choices and corresponding rationales of the students. 

---
