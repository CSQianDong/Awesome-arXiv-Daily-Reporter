# XAttention: Block Sparse Attention with Antidiagonal Scoring 

**Authors**: Ruyi Xu, Guangxuan Xiao, Haofeng Huang, Junxian Guo, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.16428)  

**Abstract**: Long-Context Transformer Models (LCTMs) are vital for real-world applications but suffer high computational costs due to attention's quadratic complexity. Block-sparse attention mitigates this by focusing computation on critical regions, yet existing methods struggle with balancing accuracy and efficiency due to costly block importance measurements. In this paper, we introduce XAttention, a plug-and-play framework that dramatically accelerates long-context inference in Transformers models using sparse attention. XAttention's key innovation is the insight that the sum of antidiagonal values (i.e., from the lower-left to upper-right) in the attention matrix provides a powerful proxy for block importance. This allows for precise identification and pruning of non-essential blocks, resulting in high sparsity and dramatically accelerated inference. Across comprehensive evaluations on demanding long-context benchmarks-including RULER and LongBench for language, VideoMME for video understanding, and VBench for video generation. XAttention achieves accuracy comparable to full attention while delivering substantial computational gains. We demonstrate up to 13.5x acceleration in attention computation. These results underscore XAttention's ability to unlock the practical potential of block sparse attention, paving the way for scalable and efficient deployment of LCTMs in real-world applications. Code is available at this https URL. 

---
# Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models 

**Authors**: Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen, Zhong, Hanjie Chen, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16419)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in complex tasks. Recent advancements in Large Reasoning Models (LRMs), such as OpenAI o1 and DeepSeek-R1, have further improved performance in System-2 reasoning domains like mathematics and programming by harnessing supervised fine-tuning (SFT) and reinforcement learning (RL) techniques to enhance the Chain-of-Thought (CoT) reasoning. However, while longer CoT reasoning sequences improve performance, they also introduce significant computational overhead due to verbose and redundant outputs, known as the "overthinking phenomenon". In this paper, we provide the first structured survey to systematically investigate and explore the current progress toward achieving efficient reasoning in LLMs. Overall, relying on the inherent mechanism of LLMs, we categorize existing works into several key directions: (1) model-based efficient reasoning, which considers optimizing full-length reasoning models into more concise reasoning models or directly training efficient reasoning models; (2) reasoning output-based efficient reasoning, which aims to dynamically reduce reasoning steps and length during inference; (3) input prompts-based efficient reasoning, which seeks to enhance reasoning efficiency based on input prompt properties such as difficulty or length control. Additionally, we introduce the use of efficient data for training reasoning models, explore the reasoning capabilities of small language models, and discuss evaluation methods and benchmarking. 

---
# CaKE: Circuit-aware Editing Enables Generalizable Knowledge Learners 

**Authors**: Yunzhi Yao, Jizhan Fang, Jia-Chen Gu, Ningyu Zhang, Shumin Deng, Huajun Chen, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.16356)  

**Abstract**: Knowledge Editing (KE) enables the modification of outdated or incorrect information in large language models (LLMs). While existing KE methods can update isolated facts, they struggle to generalize these updates to multi-hop reasoning tasks that depend on the modified knowledge. Through an analysis of reasoning circuits -- the neural pathways LLMs use for knowledge-based inference, we observe that current layer-localized KE approaches, such as MEMIT and WISE, which edit only single or a few model layers, struggle to effectively incorporate updated information into these reasoning pathways. To address this limitation, we propose CaKE (Circuit-aware Knowledge Editing), a novel method that enables more effective integration of updated knowledge in LLMs. CaKE leverages strategically curated data, guided by our circuits-based analysis, that enforces the model to utilize the modified knowledge, stimulating the model to develop appropriate reasoning circuits for newly integrated knowledge. Experimental results show that CaKE enables more accurate and consistent use of updated knowledge across related reasoning tasks, leading to an average of 20% improvement in multi-hop reasoning accuracy on MQuAKE dataset compared to existing KE methods. We release the code and data in this https URL. 

---
# LLM Braces: Straightening Out LLM Predictions with Relevant Sub-Updates 

**Authors**: Ying Shen, Lifu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16334)  

**Abstract**: Recent findings reveal that much of the knowledge in a Transformer-based Large Language Model (LLM) is encoded in its feed-forward (FFN) layers, where each FNN layer can be interpreted as the summation of sub-updates, each corresponding to a weighted column vector from the FFN's value parameter matrix that often encodes human-interpretable concepts. In light of this, we hypothesize that model performance and behaviors can be further enhanced and controlled by modulating the contributions of these sub-updates based on their relevance to the input or target output style, and propose LLMBRACES, a novel and efficient method that computes relevance scores associated with value vectors in FFN layers and leverages these scores to dynamically adjust the contribution of sub-updates. By optimizing sub-update contributions, LLMBRACES refines the prediction process, leading to more accurate and reliable outputs, much like a 'brace' providing support and stability. Moreover, LLMBRACES can be extended to support conditional control over generation characteristics, such as sentiment, thereby offering fine-grained steering of LLM outputs. Extensive experiments on various LLMs-including Qwen2.5-1.5B, Llama2-7B, and Llama3-8B-demonstrate that LLMBRACES outperforms baseline approaches in both fine-tuning and zero-shot settings while requiring significantly fewer tunable parameters, up to 75% fewer compared to LoRA. Furthermore, LLMBRACES excels in sentiment-controlled generation and toxicity reduction, highlighting its potential for flexible, controlled text generation across applications. 

---
# Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning 

**Authors**: Zhaowei Liu, Xin Guo, Fangqi Lou, Lingfeng Zeng, Jinyi Niu, Zixuan Wang, Jiajie Xu, Weige Cai, Ziwei Yang, Xueqian Zhao, Chao Li, Sheng Xu, Dezhi Chen, Yun Chen, Zuo Bai, Liwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16252)  

**Abstract**: Reasoning large language models are rapidly evolving across various domains. However, their capabilities in handling complex financial tasks still require in-depth exploration. In this paper, we introduce Fin-R1, a reasoning large language model specifically designed for the financial sector. Fin-R1 is built using a two-stage architecture, leveraging a financial reasoning dataset distilled and processed based on DeepSeek-R1. Through supervised fine-tuning (SFT) and reinforcement learning (RL) training, it demonstrates performance close to DeepSeek-R1 with a parameter size of 7 billion across a range of financial reasoning tasks. It achieves the state-of-the-art (SOTA) in the FinQA and ConvFinQA tasks between those LLMs in our evaluation, surpassing larger models in other tasks as well. Fin-R1 showcases strong reasoning and decision-making capabilities, providing solutions to various problems encountered in the financial domain. Our code is available at this https URL. 

---
# MathFusion: Enhancing Mathematic Problem-solving of LLM through Instruction Fusion 

**Authors**: Qizhi Pei, Lijun Wu, Zhuoshi Pan, Yu Li, Honglin Lin, Chenlin Ming, Xin Gao, Conghui He, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16212)  

**Abstract**: Large Language Models (LLMs) have shown impressive progress in mathematical reasoning. While data augmentation is promising to enhance mathematical problem-solving ability, current approaches are predominantly limited to instance-level modifications-such as rephrasing or generating syntactic variations-which fail to capture and leverage the intrinsic relational structures inherent in mathematical knowledge. Inspired by human learning processes, where mathematical proficiency develops through systematic exposure to interconnected concepts, we introduce MathFusion, a novel framework that enhances mathematical reasoning through cross-problem instruction synthesis. MathFusion implements this through three fusion strategies: (1) sequential fusion, which chains related problems to model solution dependencies; (2) parallel fusion, which combines analogous problems to reinforce conceptual understanding; and (3) conditional fusion, which creates context-aware selective problems to enhance reasoning flexibility. By applying these strategies, we generate a new dataset, \textbf{MathFusionQA}, followed by fine-tuning models (DeepSeekMath-7B, Mistral-7B, Llama3-8B) on it. Experimental results demonstrate that MathFusion achieves substantial improvements in mathematical reasoning while maintaining high data efficiency, boosting performance by 18.0 points in accuracy across diverse benchmarks while requiring only 45K additional synthetic instructions, representing a substantial improvement over traditional single-instruction approaches. Our datasets, models, and code are publicly available at this https URL. 

---
# SpeCache: Speculative Key-Value Caching for Efficient Generation of LLMs 

**Authors**: Shibo Jie, Yehui Tang, Kai Han, Zhi-Hong Deng, Jing Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.16163)  

**Abstract**: Transformer-based large language models (LLMs) have already achieved remarkable results on long-text tasks, but the limited GPU memory (VRAM) resources struggle to accommodate the linearly growing demand for key-value (KV) cache as the sequence length increases, which has become a bottleneck for the application of LLMs on long sequences. Existing KV cache compression methods include eviction, merging, or quantization of the KV cache to reduce its size. However, compression results in irreversible information forgetting, potentially affecting the accuracy of subsequent decoding. In this paper, we propose SpeCache, which takes full advantage of the large and easily expandable CPU memory to offload the complete KV cache, and dynamically fetches KV pairs back in each decoding step based on their importance measured by low-bit KV cache copy in VRAM. To avoid inference latency caused by CPU-GPU communication, SpeCache speculatively predicts the KV pairs that the next token might attend to, allowing us to prefetch them before the next decoding step which enables parallelization of prefetching and computation. Experiments on LongBench and Needle-in-a-Haystack benchmarks verify that SpeCache effectively reduces VRAM usage while avoiding information forgetting for long sequences without re-training, even with a 10x high KV cache compression ratio. 

---
# Towards Lighter and Robust Evaluation for Retrieval Augmented Generation 

**Authors**: Alex-Razvan Ispas, Charles-Elie Simon, Fabien Caspani, Vincent Guigue  

**Link**: [PDF](https://arxiv.org/pdf/2503.16161)  

**Abstract**: Large Language Models are prompting us to view more NLP tasks from a generative perspective. At the same time, they offer a new way of accessing information, mainly through the RAG framework. While there have been notable improvements for the autoregressive models, overcoming hallucination in the generated answers remains a continuous problem. A standard solution is to use commercial LLMs, such as GPT4, to evaluate these algorithms. However, such frameworks are expensive and not very transparent. Therefore, we propose a study which demonstrates the interest of open-weight models for evaluating RAG hallucination. We develop a lightweight approach using smaller, quantized LLMs to provide an accessible and interpretable metric that gives continuous scores for the generated answer with respect to their correctness and faithfulness. This score allows us to question decisions' reliability and explore thresholds to develop a new AUC metric as an alternative to correlation with human judgment. 

---
# Automatically Generating Chinese Homophone Words to Probe Machine Translation Estimation Systems 

**Authors**: Shenbin Qian, Constantin Orăsan, Diptesh Kanojia, Félix do Carmo  

**Link**: [PDF](https://arxiv.org/pdf/2503.16158)  

**Abstract**: Evaluating machine translation (MT) of user-generated content (UGC) involves unique challenges such as checking whether the nuance of emotions from the source are preserved in the target text. Recent studies have proposed emotion-related datasets, frameworks and models to automatically evaluate MT quality of Chinese UGC, without relying on reference translations. However, whether these models are robust to the challenge of preserving emotional nuances has been left largely unexplored. To address this gap, we introduce a novel method inspired by information theory which generates challenging Chinese homophone words related to emotions, by leveraging the concept of self-information. Our approach generates homophones that were observed to cause translation errors in emotion preservation, and exposes vulnerabilities in MT systems and their evaluation methods when tackling emotional UGC. We evaluate the efficacy of our method using human evaluation for the quality of these generated homophones, and compare it with an existing one, showing that our method achieves higher correlation with human judgments. The generated Chinese homophones, along with their manual translations, are utilized to generate perturbations and to probe the robustness of existing quality evaluation models, including models trained using multi-task learning, fine-tuned variants of multilingual language models, as well as large language models (LLMs). Our results indicate that LLMs with larger size exhibit higher stability and robustness to such perturbations. We release our data and code for reproducibility and further research. 

---
# MKG-Rank: Enhancing Large Language Models with Knowledge Graph for Multilingual Medical Question Answering 

**Authors**: Feiyang Li, Yingjian Chen, Haoran Liu, Rui Yang, Han Yuan, Yuang Jiang, Tianxiao Li, Edison Marrese Taylor, Hossein Rouhizadeh, Yusuke Iwasawa, Douglas Teodoro, Yutaka Matsuo, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.16131)  

**Abstract**: Large Language Models (LLMs) have shown remarkable progress in medical question answering (QA), yet their effectiveness remains predominantly limited to English due to imbalanced multilingual training data and scarce medical resources for low-resource languages. To address this critical language gap in medical QA, we propose Multilingual Knowledge Graph-based Retrieval Ranking (MKG-Rank), a knowledge graph-enhanced framework that enables English-centric LLMs to perform multilingual medical QA. Through a word-level translation mechanism, our framework efficiently integrates comprehensive English-centric medical knowledge graphs into LLM reasoning at a low cost, mitigating cross-lingual semantic distortion and achieving precise medical QA across language barriers. To enhance efficiency, we introduce caching and multi-angle ranking strategies to optimize the retrieval process, significantly reducing response times and prioritizing relevant medical knowledge. Extensive evaluations on multilingual medical QA benchmarks across Chinese, Japanese, Korean, and Swahili demonstrate that MKG-Rank consistently outperforms zero-shot LLMs, achieving maximum 33.89% increase in accuracy, while maintaining an average retrieval time of only 0.0009 seconds. 

---
# Cultural Alignment in Large Language Models Using Soft Prompt Tuning 

**Authors**: Reem I. Masoud, Martin Ferianc, Philip Treleaven, Miguel Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2503.16094)  

**Abstract**: Large Language Model (LLM) alignment conventionally relies on supervised fine-tuning or reinforcement learning based alignment frameworks. These methods typically require labeled or preference datasets and involve updating model weights to align the LLM with the training objective or reward model. Meanwhile, in social sciences such as cross-cultural studies, factor analysis is widely used to uncover underlying dimensions or latent variables that explain observed patterns in survey data. The non-differentiable nature of these measurements deriving from survey data renders the former alignment methods infeasible for alignment with cultural dimensions. To overcome this, we propose a parameter efficient strategy that combines soft prompt tuning, which freezes the model parameters while modifying the input prompt embeddings, with Differential Evolution (DE), a black-box optimization method for cases where a differentiable objective is unattainable. This strategy ensures alignment consistency without the need for preference data or model parameter updates, significantly enhancing efficiency and mitigating overfitting. Our method demonstrates significant improvements in LLama-3-8B-Instruct's cultural dimensions across multiple regions, outperforming both the Naive LLM and the In-context Learning (ICL) baseline, and effectively bridges computational models with human cultural nuances. 

---
# Tuning LLMs by RAG Principles: Towards LLM-native Memory 

**Authors**: Jiale Wei, Shuchi Wu, Ruochen Liu, Xiang Ying, Jingbo Shang, Fangbo Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.16071)  

**Abstract**: Memory, additional information beyond the training of large language models (LLMs), is crucial to various real-world applications, such as personal assistant. The two mainstream solutions to incorporate memory into the generation process are long-context LLMs and retrieval-augmented generation (RAG). In this paper, we first systematically compare these two types of solutions on three renovated/new datasets and show that (1) long-context solutions, although more expensive, shall be easier to capture the big picture and better answer queries which require considering the memory as a whole; and (2) when the queries concern specific information, RAG solutions shall be more competitive especially when the keywords can be explicitly matched. Therefore, we propose a novel method RAG-Tuned-LLM which fine-tunes a relative small (e.g., 7B) LLM using the data generated following the RAG principles, so it can combine the advantages of both solutions. Extensive experiments on three datasets demonstrate that RAG-Tuned-LLM can beat long-context LLMs and RAG methods across a wide range of query types. 

---
# Two-stage Incomplete Utterance Rewriting on Editing Operation 

**Authors**: Zhiyu Cao, Peifeng Li, Qiaoming Zhu, Yaxin Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16063)  

**Abstract**: Previous work on Incomplete Utterance Rewriting (IUR) has primarily focused on generating rewritten utterances based solely on dialogue context, ignoring the widespread phenomenon of coreference and ellipsis in dialogues. To address this issue, we propose a novel framework called TEO (\emph{Two-stage approach on Editing Operation}) for IUR, in which the first stage generates editing operations and the second stage rewrites incomplete utterances utilizing the generated editing operations and the dialogue context. Furthermore, an adversarial perturbation strategy is proposed to mitigate cascading errors and exposure bias caused by the inconsistency between training and inference in the second stage. Experimental results on three IUR datasets show that our TEO outperforms the SOTA models significantly. 

---
# Meta-Learning Neural Mechanisms rather than Bayesian Priors 

**Authors**: Michael Goodale, Salvador Mascarenhas, Yair Lakretz  

**Link**: [PDF](https://arxiv.org/pdf/2503.16048)  

**Abstract**: Children acquire language despite being exposed to several orders of magnitude less data than large language models require. Meta-learning has been proposed as a way to integrate human-like learning biases into neural-network architectures, combining both the structured generalizations of symbolic models with the scalability of neural-network models. But what does meta-learning exactly imbue the model with? We investigate the meta-learning of formal languages and find that, contrary to previous claims, meta-trained models are not learning simplicity-based priors when meta-trained on datasets organised around simplicity. Rather, we find evidence that meta-training imprints neural mechanisms (such as counters) into the model, which function like cognitive primitives for the network on downstream tasks. Most surprisingly, we find that meta-training on a single formal language can provide as much improvement to a model as meta-training on 5000 different formal languages, provided that the formal language incentivizes the learning of useful neural mechanisms. Taken together, our findings provide practical implications for efficient meta-learning paradigms and new theoretical insights into linking symbolic theories and neural mechanisms. 

---
# Incomplete Utterance Rewriting with Editing Operation Guidance and Utterance Augmentation 

**Authors**: Zhiyu Cao, Peifeng Li, Yaxin Fan, Qiaoming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16043)  

**Abstract**: Although existing fashionable generation methods on Incomplete Utterance Rewriting (IUR) can generate coherent utterances, they often result in the inclusion of irrelevant and redundant tokens in rewritten utterances due to their inability to focus on critical tokens in dialogue context. Furthermore, the limited size of the training datasets also contributes to the insufficient training of the IUR model. To address the first issue, we propose a multi-task learning framework EO-IUR (Editing Operation-guided Incomplete Utterance Rewriting) that introduces the editing operation labels generated by sequence labeling module to guide generation model to focus on critical tokens. Furthermore, we introduce a token-level heterogeneous graph to represent dialogues. To address the second issue, we propose a two-dimensional utterance augmentation strategy, namely editing operation-based incomplete utterance augmentation and LLM-based historical utterance augmentation. The experimental results on three datasets demonstrate that our EO-IUR outperforms previous state-of-the-art (SOTA) baselines in both open-domain and task-oriented dialogue. The code will be available at this https URL. 

---
# Evaluating Test-Time Scaling LLMs for Legal Reasoning: OpenAI o1, DeepSeek-R1, and Beyond 

**Authors**: Yaoyao Yu, Leilei Gan, Yinghao Hu, Bin Wei, Kun Kuang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16040)  

**Abstract**: Recently, Test-Time Scaling Large Language Models (LLMs), such as DeepSeek-R1 and OpenAI o1, have demonstrated exceptional capabilities across various domains and tasks, particularly in reasoning. While these models have shown impressive performance on general language tasks, their effectiveness in specialized fields like legal remains unclear. To address this, we present a preliminary evaluation of LLMs in various legal scenarios, covering both Chinese and English legal tasks. Our analysis includes 9 LLMs and 17 legal tasks, with a focus on newly published and more complex challenges such as multi-defendant legal judgments and legal argument reasoning. Our findings indicate that, despite DeepSeek-R1 and OpenAI o1 being among the most powerful models, their legal reasoning capabilities are still lacking. Specifically, these models score below 80\% on seven Chinese legal reasoning tasks and below 80\% on two English legal reasoning tasks. This suggests that, even among the most advanced reasoning models, legal reasoning abilities remain underdeveloped. 

---
# Deceptive Humor: A Synthetic Multilingual Benchmark Dataset for Bridging Fabricated Claims with Humorous Content 

**Authors**: Sai Kartheek Reddy Kasu, Shankar Biradar, Sunil Saumya  

**Link**: [PDF](https://arxiv.org/pdf/2503.16031)  

**Abstract**: This paper presents the Deceptive Humor Dataset (DHD), a novel resource for studying humor derived from fabricated claims and misinformation. In an era of rampant misinformation, understanding how humor intertwines with deception is essential. DHD consists of humor-infused comments generated from false narratives, incorporating fabricated claims and manipulated information using the ChatGPT-4o model. Each instance is labeled with a Satire Level, ranging from 1 for subtle satire to 3 for high-level satire and classified into five distinct Humor Categories: Dark Humor, Irony, Social Commentary, Wordplay, and Absurdity. The dataset spans multiple languages including English, Telugu, Hindi, Kannada, Tamil, and their code-mixed variants (Te-En, Hi-En, Ka-En, Ta-En), making it a valuable multilingual benchmark. By introducing DHD, we establish a structured foundation for analyzing humor in deceptive contexts, paving the way for a new research direction that explores how humor not only interacts with misinformation but also influences its perception and spread. We establish strong baselines for the proposed dataset, providing a foundation for future research to benchmark and advance deceptive humor detection models. 

---
# The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement 

**Authors**: Ruihan Yang, Fanghua Ye, Jian Li, Siyu Yuan, Yikai Zhang, Zhaopeng Tu, Xiaolong Li, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16024)  

**Abstract**: Large language models (LLMs) have recently transformed from text-based assistants to autonomous agents capable of planning, reasoning, and iteratively improving their actions. While numerical reward signals and verifiers can effectively rank candidate actions, they often provide limited contextual guidance. In contrast, natural language feedback better aligns with the generative capabilities of LLMs, providing richer and more actionable suggestions. However, parsing and implementing this feedback effectively can be challenging for LLM-based agents. In this work, we introduce Critique-Guided Improvement (CGI), a novel two-player framework, comprising an actor model that explores an environment and a critic model that generates detailed nature language feedback. By training the critic to produce fine-grained assessments and actionable revisions, and the actor to utilize these critiques, our approach promotes more robust exploration of alternative strategies while avoiding local optima. Experiments in three interactive environments show that CGI outperforms existing baselines by a substantial margin. Notably, even a small critic model surpasses GPT-4 in feedback quality. The resulting actor achieves state-of-the-art performance, demonstrating the power of explicit iterative guidance to enhance decision-making in LLM-based agents. 

---
# Corrective In-Context Learning: Evaluating Self-Correction in Large Language Models 

**Authors**: Mario Sanz-Guerrero, Katharina von der Wense  

**Link**: [PDF](https://arxiv.org/pdf/2503.16022)  

**Abstract**: In-context learning (ICL) has transformed the use of large language models (LLMs) for NLP tasks, enabling few-shot learning by conditioning on labeled examples without finetuning. Despite its effectiveness, ICL is prone to errors, especially for challenging examples. With the goal of improving the performance of ICL, we propose corrective in-context learning (CICL), an approach that incorporates a model's incorrect predictions alongside ground truth corrections into the prompt, aiming to enhance classification accuracy through self-correction. However, contrary to our hypothesis, extensive experiments on text classification tasks demonstrate that CICL consistently underperforms standard ICL, with performance degrading as the proportion of corrections in the prompt increases. Our findings indicate that CICL introduces confusion by disrupting the model's task understanding, rather than refining its predictions. Additionally, we observe that presenting harder examples in standard ICL does not improve performance, suggesting that example difficulty alone may not be a reliable criterion for effective selection. By presenting these negative results, we provide important insights into the limitations of self-corrective mechanisms in LLMs and offer directions for future research. 

---
# ECKGBench: Benchmarking Large Language Models in E-commerce Leveraging Knowledge Graph 

**Authors**: Langming Liu, Haibin Chen, Yuhao Wang, Yujin Yuan, Shilei Liu, Wenbo Su, Xiangyu Zhao, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.15990)  

**Abstract**: Large language models (LLMs) have demonstrated their capabilities across various NLP tasks. Their potential in e-commerce is also substantial, evidenced by practical implementations such as platform search, personalized recommendations, and customer service. One primary concern associated with LLMs is their factuality (e.g., hallucination), which is urgent in e-commerce due to its significant impact on user experience and revenue. Despite some methods proposed to evaluate LLMs' factuality, issues such as lack of reliability, high consumption, and lack of domain expertise leave a gap between effective assessment in e-commerce. To bridge the evaluation gap, we propose ECKGBench, a dataset specifically designed to evaluate the capacities of LLMs in e-commerce knowledge. Specifically, we adopt a standardized workflow to automatically generate questions based on a large-scale knowledge graph, guaranteeing sufficient reliability. We employ the simple question-answering paradigm, substantially improving the evaluation efficiency by the least input and output tokens. Furthermore, we inject abundant e-commerce expertise in each evaluation stage, including human annotation, prompt design, negative sampling, and verification. Besides, we explore the LLMs' knowledge boundaries in e-commerce from a novel perspective. Through comprehensive evaluations of several advanced LLMs on ECKGBench, we provide meticulous analysis and insights into leveraging LLMs for e-commerce. 

---
# InhibiDistilbert: Knowledge Distillation for a ReLU and Addition-based Transformer 

**Authors**: Tony Zhang, Rickard Brännvall  

**Link**: [PDF](https://arxiv.org/pdf/2503.15983)  

**Abstract**: This work explores optimizing transformer-based language models by integrating model compression techniques with inhibitor attention, a novel alternative attention mechanism. Inhibitor attention employs Manhattan distances and ReLU activations instead of the matrix multiplications and softmax activation of the conventional scaled dot-product attention. This shift offers potential computational and energy savings while maintaining model effectiveness. We propose further adjustments to improve the inhibitor mechanism's training efficiency and evaluate its performance on the DistilBERT architecture. Our knowledge distillation experiments indicate that the modified inhibitor transformer model can achieve competitive performance on standard NLP benchmarks, including General Language Understanding Evaluation (GLUE) and sentiment analysis tasks. 

---
# Exploratory Study into Relations between Cognitive Distortions and Emotional Appraisals 

**Authors**: Navneet Agarwal, Kairit Sirts  

**Link**: [PDF](https://arxiv.org/pdf/2503.15979)  

**Abstract**: In recent years, there has been growing interest in studying cognitive distortions and emotional appraisals from both computational and psychological perspectives. Despite considerable similarities between emotional reappraisal and cognitive reframing as emotion regulation techniques, these concepts have largely been examined in isolation. This research explores the relationship between cognitive distortions and emotional appraisal dimensions, examining their potential connections and relevance for future interdisciplinary studies. Under this pretext, we conduct an exploratory computational study, aimed at investigating the relationship between cognitive distortion and emotional appraisals. We show that the patterns of statistically significant relationships between cognitive distortions and appraisal dimensions vary across different distortion categories, giving rise to distinct appraisal profiles for individual distortion classes. Additionally, we analyze the impact of cognitive restructuring on appraisal dimensions, exemplifying the emotion regulation aspect of cognitive restructuring. 

---
# Adaptive Group Policy Optimization: Towards Stable Training and Token-Efficient Reasoning 

**Authors**: Chen Li, Nazhou Liu, Kai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.15952)  

**Abstract**: Since DeepSeek-R1 popularized, Group Relative Policy Optimization (GRPO) has become the core part of Reasoning LLMs training. However, we find some deficiency that influences RL stability and inference efficiency. Thus, we propose Adaptive Group Policy Optimization (AGPO) which contains two simple but effective modifications: a revised advantage estimation method to mitigate zero-variance situations; a length-based reward, incentivizing the model to avoid overthinking. The experiments demonstrate our methods achieve more stable training and comparable or superior performance with significantly fewer tokens in reasoning steps. 

---
# From Chaos to Order: The Atomic Reasoner Framework for Fine-grained Reasoning in Large Language Models 

**Authors**: Jinyi Liu, Yan Zheng, Rong Cheng, Qiyu Wu, Wei Guo, Fei Ni, Hebin Liang, Yifu Yuan, Hangyu Mao, Fuzheng Zhang, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2503.15944)  

**Abstract**: Recent advances in large language models (LLMs) have shown remarkable progress, yet their capacity for logical ``slow-thinking'' reasoning persists as a critical research frontier. Current inference scaling paradigms suffer from two fundamental constraints: fragmented thought flows compromising logical coherence, and intensively computational complexity that escalates with search space dimensions. To overcome these limitations, we present \textbf{Atomic Reasoner} (\textbf{AR}), a cognitive inference strategy that enables fine-grained reasoning through systematic atomic-level operations. AR decomposes the reasoning process into atomic cognitive units, employing a cognitive routing mechanism to dynamically construct reasoning representations and orchestrate inference pathways. This systematic methodology implements stepwise, structured cognition, which ensures logical coherence while significantly reducing cognitive load, effectively simulating the cognitive patterns observed in human deep thinking processes. Extensive experimental results demonstrate AR's superior reasoning capabilities without the computational burden of exhaustive solution searches, particularly excelling in linguistic logic puzzles. These findings substantiate AR's effectiveness in enhancing LLMs' capacity for robust, long-sequence logical reasoning and deliberation. 

---
# Towards Automatic Continual Learning: A Self-Adaptive Framework for Continual Instruction Tuning 

**Authors**: Peiyi Lin, Fukai Zhang, Kai Niu, Hao Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15924)  

**Abstract**: Continual instruction tuning enables large language models (LLMs) to learn incrementally while retaining past knowledge, whereas existing methods primarily focus on how to retain old knowledge rather than on selecting which new knowledge to learn. In domain-specific contexts, maintaining data quality and managing system constraints remain key challenges. To address these issues, we propose an automated continual instruction tuning framework that dynamically filters incoming data, which identify and reduce redundant data across successive updates. Our approach utilizes a small proxy model for efficient perplexity-based filtering, and updates the proxy to ensure that the filtering criteria remain aligned with the evolving state of the deployed model. Compared to existing static data selection methods, our framework can effectively handle incrementally acquired data and shifting distributions. Additionally, it addresses practical deployment challenges by enabling seamless model updates, supporting version rollback and incorporating automatic checkpoint evaluation. We evaluated the system in real-world medical scenarios. It reduced computational costs by 66.7% and improved model performance, and achieved autonomous updates, thus demonstrating its effectiveness for automatic continual instruction tuning. 

---
# From Structured Prompts to Open Narratives: Measuring Gender Bias in LLMs Through Open-Ended Storytelling 

**Authors**: Evan Chen, Run-Jun Zhan, Yan-Bai Lin, Hung-Hsuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.15904)  

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing, yet concerns persist regarding their tendency to reflect or amplify social biases present in their training data. This study introduces a novel evaluation framework to uncover gender biases in LLMs, focusing on their occupational narratives. Unlike previous methods relying on structured scenarios or carefully crafted prompts, our approach leverages free-form storytelling to reveal biases embedded in the models. Systematic analyses show an overrepresentation of female characters across occupations in six widely used LLMs. Additionally, our findings reveal that LLM-generated occupational gender rankings align more closely with human stereotypes than actual labor statistics. These insights underscore the need for balanced mitigation strategies to ensure fairness while avoiding the reinforcement of new stereotypes. 

---
# Parameters vs. Context: Fine-Grained Control of Knowledge Reliance in Language Models 

**Authors**: Baolong Bi, Shenghua Liu, Yiwei Wang, Yilong Xu, Junfeng Fang, Lingrui Mei, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.15888)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates hallucinations in Large Language Models (LLMs) by integrating external knowledge. However, conflicts between parametric knowledge and retrieved context pose challenges, particularly when retrieved information is unreliable or the model's internal knowledge is outdated. In such cases, LLMs struggle to determine whether to rely more on their own parameters or the conflicted context. To address this, we propose **CK-PLUG**, a plug-and-play method for controlling LLMs' reliance on parametric and contextual knowledge. We introduce a novel knowledge consistency metric, Confidence Gain, which detects knowledge conflicts by measuring entropy shifts in token probability distributions after context insertion. CK-PLUG then enables fine-grained control over knowledge preference by adjusting the probability distribution of tokens with negative confidence gain through a single tuning parameter. Experiments demonstrate CK-PLUG's ability to significantly regulate knowledge reliance in counterfactual RAG scenarios while maintaining generation fluency and knowledge accuracy. For instance, on Llama3-8B, memory recall (MR) of RAG response can be adjusted within a broad range (9.9%-71.9%), compared to the baseline of 42.1%. Moreover, CK-PLUG supports adaptive control based on the model's confidence in both internal and external knowledge, achieving consistent performance improvements across various general RAG tasks. Our code is available at: $\href{this https URL}{\text{this https URL}}$. 

---
# Typed-RAG: Type-aware Multi-Aspect Decomposition for Non-Factoid Question Answering 

**Authors**: DongGeon Lee, Ahjeong Park, Hyeri Lee, Hyeonseo Nam, Yunho Maeng  

**Link**: [PDF](https://arxiv.org/pdf/2503.15879)  

**Abstract**: Non-factoid question-answering (NFQA) poses a significant challenge due to its open-ended nature, diverse intents, and the need for multi-aspect reasoning, which renders conventional factoid QA approaches, including retrieval-augmented generation (RAG), inadequate. Unlike factoid questions, non-factoid questions (NFQs) lack definitive answers and require synthesizing information from multiple sources across various reasoning dimensions. To address these limitations, we introduce Typed-RAG, a type-aware multi-aspect decomposition framework within the RAG paradigm for NFQA. Typed-RAG classifies NFQs into distinct types -- such as debate, experience, and comparison -- and applies aspect-based decomposition to refine retrieval and generation strategies. By decomposing multi-aspect NFQs into single-aspect sub-queries and aggregating the results, Typed-RAG generates more informative and contextually relevant responses. To evaluate Typed-RAG, we introduce Wiki-NFQA, a benchmark dataset covering diverse NFQ types. Experimental results demonstrate that Typed-RAG outperforms baselines, thereby highlighting the importance of type-aware decomposition for effective retrieval and generation in NFQA. Our code and dataset are available at \href{this https URL}{this https URL}. 

---
# Uncertainty Quantification and Confidence Calibration in Large Language Models: A Survey 

**Authors**: Xiaoou Liu, Tiejin Chen, Longchao Da, Chacha Chen, Zhen Lin, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2503.15850)  

**Abstract**: Large Language Models (LLMs) excel in text generation, reasoning, and decision-making, enabling their adoption in high-stakes domains such as healthcare, law, and transportation. However, their reliability is a major concern, as they often produce plausible but incorrect responses. Uncertainty quantification (UQ) enhances trustworthiness by estimating confidence in outputs, enabling risk mitigation and selective prediction. However, traditional UQ methods struggle with LLMs due to computational constraints and decoding inconsistencies. Moreover, LLMs introduce unique uncertainty sources, such as input ambiguity, reasoning path divergence, and decoding stochasticity, that extend beyond classical aleatoric and epistemic uncertainty. To address this, we introduce a new taxonomy that categorizes UQ methods based on computational efficiency and uncertainty dimensions (input, reasoning, parameter, and prediction uncertainty). We evaluate existing techniques, assess their real-world applicability, and identify open challenges, emphasizing the need for scalable, interpretable, and robust UQ approaches to enhance LLM reliability. 

---
# Fùxì: A Benchmark for Evaluating Language Models on Ancient Chinese Text Understanding and Generation 

**Authors**: Shangqing Zhao, Yuhao Zhou, Yupei Ren, Zhe Chen, Chenghao Jia, Fang Zhe, Zhaogaung Long, Shu Liu, Man Lan  

**Link**: [PDF](https://arxiv.org/pdf/2503.15837)  

**Abstract**: Ancient Chinese text processing presents unique challenges for large language models (LLMs) due to its distinct linguistic features, complex structural constraints, and rich cultural context. While existing benchmarks have primarily focused on evaluating comprehension through multiple-choice questions, there remains a critical gap in assessing models' generative capabilities in classical Chinese. We introduce Fùxì, a comprehensive benchmark that evaluates both understanding and generation capabilities across 21 diverse tasks. Our benchmark distinguishes itself through three key contributions: (1) balanced coverage of both comprehension and generation tasks, including novel tasks like poetry composition and couplet completion, (2) specialized evaluation metrics designed specifically for classical Chinese text generation, combining rule-based verification with fine-tuned LLM evaluators, and (3) a systematic assessment framework that considers both linguistic accuracy and cultural authenticity. Through extensive evaluation of state-of-the-art LLMs, we reveal significant performance gaps between understanding and generation tasks, with models achieving promising results in comprehension but struggling considerably in generation tasks, particularly those requiring deep cultural knowledge and adherence to classical formats. Our findings highlight the current limitations in ancient Chinese text processing and provide insights for future model development. The benchmark, evaluation toolkit, and baseline results are publicly available to facilitate research in this domain. 

---
# Grammar and Gameplay-aligned RL for Game Description Generation with LLMs 

**Authors**: Tsunehiko Tanaka, Edgar Simo-Serra  

**Link**: [PDF](https://arxiv.org/pdf/2503.15783)  

**Abstract**: Game Description Generation (GDG) is the task of generating a game description written in a Game Description Language (GDL) from natural language text. Previous studies have explored generation methods leveraging the contextual understanding capabilities of Large Language Models (LLMs); however, accurately reproducing the game features of the game descriptions remains a challenge. In this paper, we propose reinforcement learning-based fine-tuning of LLMs for GDG (RLGDG). Our training method simultaneously improves grammatical correctness and fidelity to game concepts by introducing both grammar rewards and concept rewards. Furthermore, we adopt a two-stage training strategy where Reinforcement Learning (RL) is applied following Supervised Fine-Tuning (SFT). Experimental results demonstrate that our proposed method significantly outperforms baseline methods using SFT alone. 

---
# Can one size fit all?: Measuring Failure in Multi-Document Summarization Domain Transfer 

**Authors**: Alexandra DeLucia, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2503.15768)  

**Abstract**: Abstractive multi-document summarization (MDS) is the task of automatically summarizing information in multiple documents, from news articles to conversations with multiple speakers. The training approaches for current MDS models can be grouped into four approaches: end-to-end with special pre-training ("direct"), chunk-then-summarize, extract-then-summarize, and inference with GPT-style models. In this work, we evaluate MDS models across training approaches, domains, and dimensions (reference similarity, quality, and factuality), to analyze how and why models trained on one domain can fail to summarize documents from another (News, Science, and Conversation) in the zero-shot domain transfer setting. We define domain-transfer "failure" as a decrease in factuality, higher deviation from the target, and a general decrease in summary quality. In addition to exploring domain transfer for MDS models, we examine potential issues with applying popular summarization metrics out-of-the-box. 

---
# KoGNER: A Novel Framework for Knowledge Graph Distillation on Biomedical Named Entity Recognition 

**Authors**: Heming Zhang, Wenyu Li, Di Huang, Yinjie Tang, Yixin Chen, Philip Payne, Fuhai Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.15737)  

**Abstract**: Named Entity Recognition (NER) is a fundamental task in Natural Language Processing (NLP) that plays a crucial role in information extraction, question answering, and knowledge-based systems. Traditional deep learning-based NER models often struggle with domain-specific generalization and suffer from data sparsity issues. In this work, we introduce Knowledge Graph distilled for Named Entity Recognition (KoGNER), a novel approach that integrates Knowledge Graph (KG) distillation into NER models to enhance entity recognition performance. Our framework leverages structured knowledge representations from KGs to enrich contextual embeddings, thereby improving entity classification and reducing ambiguity in entity detection. KoGNER employs a two-step process: (1) Knowledge Distillation, where external knowledge sources are distilled into a lightweight representation for seamless integration with NER models, and (2) Entity-Aware Augmentation, which integrates contextual embeddings that have been enriched with knowledge graph information directly into GNN, thereby improving the model's ability to understand and represent entity relationships. Experimental results on benchmark datasets demonstrate that KoGNER achieves state-of-the-art performance, outperforming finetuned NER models and LLMs by a significant margin. These findings suggest that leveraging knowledge graphs as auxiliary information can significantly improve NER accuracy, making KoGNER a promising direction for future research in knowledge-aware NLP. 

---
# Am I eligible? Natural Language Inference for Clinical Trial Patient Recruitment: the Patient's Point of View 

**Authors**: Mathilde Aguiar, Pierre Zweigenbaum, Nona Naderi  

**Link**: [PDF](https://arxiv.org/pdf/2503.15718)  

**Abstract**: Recruiting patients to participate in clinical trials can be challenging and time-consuming. Usually, participation in a clinical trial is initiated by a healthcare professional and proposed to the patient. Promoting clinical trials directly to patients via online recruitment might help to reach them more efficiently. In this study, we address the case where a patient is initiating their own recruitment process and wants to determine whether they are eligible for a given clinical trial, using their own language to describe their medical profile. To study whether this creates difficulties in the patient trial matching process, we design a new dataset and task, Natural Language Inference for Patient Recruitment (NLI4PR), in which patient language profiles must be matched to clinical trials. We create it by adapting the TREC 2022 Clinical Trial Track dataset, which provides patients' medical profiles, and rephrasing them manually using patient language. We also use the associated clinical trial reports where the patients are either eligible or excluded. We prompt several open-source Large Language Models on our task and achieve from 56.5 to 71.8 of F1 score using patient language, against 64.7 to 73.1 for the same task using medical language. When using patient language, we observe only a small loss in performance for the best model, suggesting that having the patient as a starting point could be adopted to help recruit patients for clinical trials. The corpus and code bases are all freely available on our Github and HuggingFace repositories. 

---
# Enhancing Pancreatic Cancer Staging with Large Language Models: The Role of Retrieval-Augmented Generation 

**Authors**: Hisashi Johno, Yuki Johno, Akitomo Amakawa, Junichi Sato, Ryota Tozuka, Atsushi Komaba, Hiroaki Watanabe, Hiroki Watanabe, Chihiro Goto, Hiroyuki Morisaka, Hiroshi Onishi, Kazunori Nakamoto  

**Link**: [PDF](https://arxiv.org/pdf/2503.15664)  

**Abstract**: Purpose: Retrieval-augmented generation (RAG) is a technology to enhance the functionality and reliability of large language models (LLMs) by retrieving relevant information from reliable external knowledge (REK). RAG has gained interest in radiology, and we previously reported the utility of NotebookLM, an LLM with RAG (RAG-LLM), for lung cancer staging. However, since the comparator LLM differed from NotebookLM's internal model, it remained unclear whether its advantage stemmed from RAG or inherent model differences. To better isolate RAG's impact and assess its utility across different cancers, we compared NotebookLM with its internal LLM, Gemini 2.0 Flash, in a pancreatic cancer staging experiment.
Materials and Methods: A summary of Japan's pancreatic cancer staging guidelines was used as REK. We compared three groups - REK+/RAG+ (NotebookLM with REK), REK+/RAG- (Gemini 2.0 Flash with REK), and REK-/RAG- (Gemini 2.0 Flash without REK) - in staging 100 fictional pancreatic cancer cases based on CT findings. Staging criteria included TNM classification, local invasion factors, and resectability classification. In REK+/RAG+, retrieval accuracy was quantified based on the sufficiency of retrieved REK excerpts.
Results: REK+/RAG+ achieved a staging accuracy of 70%, outperforming REK+/RAG- (38%) and REK-/RAG- (35%). For TNM classification, REK+/RAG+ attained 80% accuracy, exceeding REK+/RAG- (55%) and REK-/RAG- (50%). Additionally, REK+/RAG+ explicitly presented retrieved REK excerpts, achieving a retrieval accuracy of 92%.
Conclusion: NotebookLM, a RAG-LLM, outperformed its internal LLM, Gemini 2.0 Flash, in a pancreatic cancer staging experiment, suggesting that RAG may improve LLM's staging accuracy. Furthermore, its ability to retrieve and present REK excerpts provides transparency for physicians, highlighting its applicability for clinical diagnosis and classification. 

---
# Does Context Matter? ContextualJudgeBench for Evaluating LLM-based Judges in Contextual Settings 

**Authors**: Austin Xu, Srijan Bansal, Yifei Ming, Semih Yavuz, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2503.15620)  

**Abstract**: The large language model (LLM)-as-judge paradigm has been used to meet the demand for a cheap, reliable, and fast evaluation of model outputs during AI system development and post-deployment monitoring. While judge models -- LLMs finetuned to specialize in assessing and critiquing model outputs -- have been touted as general purpose evaluators, they are typically evaluated only on non-contextual scenarios, such as instruction following. The omission of contextual settings -- those where external information is used as context to generate an output -- is surprising given the increasing prevalence of retrieval-augmented generation (RAG) and summarization use cases. Contextual assessment is uniquely challenging, as evaluation often depends on practitioner priorities, leading to conditional evaluation criteria (e.g., comparing responses based on factuality and then considering completeness if they are equally factual). To address the gap, we propose ContextualJudgeBench, a judge benchmark with 2,000 challenging response pairs across eight splits inspired by real-world contextual evaluation scenarios. We build our benchmark with a multi-pronged data construction pipeline that leverages both existing human annotations and model-based perturbations. Our comprehensive study across 11 judge models and 9 general purpose models, reveals that the contextual information and its assessment criteria present a significant challenge to even state-of-the-art models. For example, OpenAI's o1, the best-performing model, barely reaches 55% consistent accuracy. 

---
# Survey on Evaluation of LLM-based Agents 

**Authors**: Asaf Yehudai, Lilach Eden, Alan Li, Guy Uziel, Yilun Zhao, Roy Bar-Haim, Arman Cohan, Michal Shmueli-Scheuer  

**Link**: [PDF](https://arxiv.org/pdf/2503.16416)  

**Abstract**: The emergence of LLM-based agents represents a paradigm shift in AI, enabling autonomous systems to plan, reason, use tools, and maintain memory while interacting with dynamic environments. This paper provides the first comprehensive survey of evaluation methodologies for these increasingly capable agents. We systematically analyze evaluation benchmarks and frameworks across four critical dimensions: (1) fundamental agent capabilities, including planning, tool use, self-reflection, and memory; (2) application-specific benchmarks for web, software engineering, scientific, and conversational agents; (3) benchmarks for generalist agents; and (4) frameworks for evaluating agents. Our analysis reveals emerging trends, including a shift toward more realistic, challenging evaluations with continuously updated benchmarks. We also identify critical gaps that future research must address-particularly in assessing cost-efficiency, safety, and robustness, and in developing fine-grained, and scalable evaluation methods. This survey maps the rapidly evolving landscape of agent evaluation, reveals the emerging trends in the field, identifies current limitations, and proposes directions for future research. 

---
# The Emperor's New Clothes in Benchmarking? A Rigorous Examination of Mitigation Strategies for LLM Benchmark Data Contamination 

**Authors**: Yifan Sun, Han Wang, Dongbai Li, Gang Wang, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16402)  

**Abstract**: Benchmark Data Contamination (BDC)-the inclusion of benchmark testing samples in the training set-has raised increasing concerns in Large Language Model (LLM) evaluation, leading to falsely inflated performance estimates and undermining evaluation reliability. To address this, researchers have proposed various mitigation strategies to update existing benchmarks, including modifying original questions or generating new ones based on them. However, a rigorous examination of the effectiveness of these mitigation strategies remains lacking. In this paper, we design a systematic and controlled pipeline along with two novel metrics-fidelity and contamination resistance-to provide a fine-grained and comprehensive assessment of existing BDC mitigation strategies. Previous assessment methods, such as accuracy drop and accuracy matching, focus solely on aggregate accuracy, often leading to incomplete or misleading conclusions. Our metrics address this limitation by emphasizing question-level evaluation result matching. Extensive experiments with 10 LLMs, 5 benchmarks, 20 BDC mitigation strategies, and 2 contamination scenarios reveal that no existing strategy significantly improves resistance over the vanilla case (i.e., no benchmark update) across all benchmarks, and none effectively balances fidelity and contamination resistance. These findings underscore the urgent need for designing more effective BDC mitigation strategies. Our code repository is available at this https URL. 

---
# Do Visual Imaginations Improve Vision-and-Language Navigation Agents? 

**Authors**: Akhil Perincherry, Jacob Krantz, Stefan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.16394)  

**Abstract**: Vision-and-Language Navigation (VLN) agents are tasked with navigating an unseen environment using natural language instructions. In this work, we study if visual representations of sub-goals implied by the instructions can serve as navigational cues and lead to increased navigation performance. To synthesize these visual representations or imaginations, we leverage a text-to-image diffusion model on landmark references contained in segmented instructions. These imaginations are provided to VLN agents as an added modality to act as landmark cues and an auxiliary loss is added to explicitly encourage relating these with their corresponding referring expressions. Our findings reveal an increase in success rate (SR) of around 1 point and up to 0.5 points in success scaled by inverse path length (SPL) across agents. These results suggest that the proposed approach reinforces visual understanding compared to relying on language instructions alone. Code and data for our work can be found at this https URL. 

---
# Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't 

**Authors**: Quy-Anh Dang, Chris Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2503.16219)  

**Abstract**: Enhancing the reasoning capabilities of large language models (LLMs) typically relies on massive computational resources and extensive datasets, limiting accessibility for resource-constrained settings. Our study investigates the potential of reinforcement learning (RL) to improve reasoning in small LLMs, focusing on a 1.5-billion-parameter model, DeepSeek-R1-Distill-Qwen-1.5B, under strict constraints: training on 4 NVIDIA A40 GPUs (48 GB VRAM each) within 24 hours. Adapting the Group Relative Policy Optimization (GRPO) algorithm and curating a compact, high-quality mathematical reasoning dataset, we conducted three experiments to explore model behavior and performance. Our results demonstrate rapid reasoning gains - e.g., AMC23 accuracy rising from 63% to 80% and AIME24 reaching 46.7%, surpassing o1-preview - using only 7,000 samples and a $42 training cost, compared to thousands of dollars for baseline models. However, challenges such as optimization instability and length constraints emerged with prolonged training. These findings highlight the efficacy of RL-based fine-tuning for small LLMs, offering a cost-effective alternative to large-scale approaches. We release our code and datasets as open-source resources, providing insights into trade-offs and laying a foundation for scalable, reasoning-capable LLMs in resource-limited environments. All are available at this https URL. 

---
# Accurate Scene Text Recognition with Efficient Model Scaling and Cloze Self-Distillation 

**Authors**: Andrea Maracani, Savas Ozkan, Sijun Cho, Hyowon Kim, Eunchung Noh, Jeongwon Min, Cho Jung Min, Dookun Park, Mete Ozay  

**Link**: [PDF](https://arxiv.org/pdf/2503.16184)  

**Abstract**: Scaling architectures have been proven effective for improving Scene Text Recognition (STR), but the individual contribution of vision encoder and text decoder scaling remain under-explored. In this work, we present an in-depth empirical analysis and demonstrate that, contrary to previous observations, scaling the decoder yields significant performance gains, always exceeding those achieved by encoder scaling alone. We also identify label noise as a key challenge in STR, particularly in real-world data, which can limit the effectiveness of STR models. To address this, we propose Cloze Self-Distillation (CSD), a method that mitigates label noise by distilling a student model from context-aware soft predictions and pseudolabels generated by a teacher model. Additionally, we enhance the decoder architecture by introducing differential cross-attention for STR. Our methodology achieves state-of-the-art performance on 10 out of 11 benchmarks using only real data, while significantly reducing the parameter size and computational costs. 

---
# CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models 

**Authors**: Hong Yi Lin, Chunhua Liu, Haoyu Gao, Patanamon Thongtanunam, Christoph Treude  

**Link**: [PDF](https://arxiv.org/pdf/2503.16167)  

**Abstract**: State-of-the-art large language models (LLMs) have demonstrated impressive code generation capabilities but struggle with real-world software engineering tasks, such as revising source code to address code reviews, hindering their practical use. Code review comments are often implicit, ambiguous, and colloquial, requiring models to grasp both code and human intent. This challenge calls for evaluating large language models' ability to bridge both technical and conversational contexts. While existing work has employed the automated code refinement (ACR) task to resolve these comments, current evaluation methods fall short, relying on text matching metrics that provide limited insight into model failures and remain susceptible to training data contamination. To address these limitations, we introduce a novel evaluation benchmark, $\textbf{CodeReviewQA}$ that enables us to conduct fine-grained assessment of model capabilities and mitigate data contamination risks. In CodeReviewQA, we decompose the generation task of code refinement into $\textbf{three essential reasoning steps}$: $\textit{change type recognition}$ (CTR), $\textit{change localisation}$ (CL), and $\textit{solution identification}$ (SI). Each step is reformulated as multiple-choice questions with varied difficulty levels, enabling precise assessment of model capabilities, while mitigating data contamination risks. Our comprehensive evaluation spans 72 recently released large language models on $\textbf{900 manually curated, high-quality examples}$ across nine programming languages. Our results show that CodeReviewQA is able to expose specific model weaknesses in code review comprehension, disentangled from their generative automated code refinement results. 

---
# Only a Little to the Left: A Theory-grounded Measure of Political Bias in Large Language Models 

**Authors**: Mats Faulborn, Indira Sen, Max Pellert, Andreas Spitz, David Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2503.16148)  

**Abstract**: Prompt-based language models like GPT4 and LLaMa have been used for a wide variety of use cases such as simulating agents, searching for information, or for content analysis. For all of these applications and others, political biases in these models can affect their performance. Several researchers have attempted to study political bias in language models using evaluation suites based on surveys, such as the Political Compass Test (PCT), often finding a particular leaning favored by these models. However, there is some variation in the exact prompting techniques, leading to diverging findings and most research relies on constrained-answer settings to extract model responses. Moreover, the Political Compass Test is not a scientifically valid survey instrument. In this work, we contribute a political bias measured informed by political science theory, building on survey design principles to test a wide variety of input prompts, while taking into account prompt sensitivity. We then prompt 11 different open and commercial models, differentiating between instruction-tuned and non-instruction-tuned models, and automatically classify their political stances from 88,110 responses. Leveraging this dataset, we compute political bias profiles across different prompt variations and find that while PCT exaggerates bias in certain models like GPT3.5, measures of political bias are often unstable, but generally more left-leaning for instruction-tuned models. 

---
# Redefining Toxicity: An Objective and Context-Aware Approach for Stress-Level-Based Detection 

**Authors**: Sergey Berezin, Reza Farahbakhsh, Noel Crespi  

**Link**: [PDF](https://arxiv.org/pdf/2503.16072)  

**Abstract**: The fundamental problem of toxicity detection lies in the fact that the term "toxicity" is ill-defined. Such uncertainty causes researchers to rely on subjective and vague data during model training, which leads to non-robust and inaccurate results, following the 'garbage in - garbage out' paradigm. This study introduces a novel, objective, and context-aware framework for toxicity detection, leveraging stress levels as a key determinant of toxicity. We propose new definition, metric and training approach as a parts of our framework and demonstrate it's effectiveness using a dataset we collected. 

---
# Hybrid-Level Instruction Injection for Video Token Compression in Multi-modal Large Language Models 

**Authors**: Zhihang Liu, Chen-Wei Xie, Pandeng Li, Liming Zhao, Longxiang Tang, Yun Zheng, Chuanbin Liu, Hongtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2503.16036)  

**Abstract**: Recent Multi-modal Large Language Models (MLLMs) have been challenged by the computational overhead resulting from massive video frames, often alleviated through compression strategies. However, the visual content is not equally contributed to user instructions, existing strategies (\eg, average pool) inevitably lead to the loss of potentially useful information. To tackle this, we propose the Hybrid-level Instruction Injection Strategy for Conditional Token Compression in MLLMs (HICom), utilizing the instruction as a condition to guide the compression from both local and global levels. This encourages the compression to retain the maximum amount of user-focused information while reducing visual tokens to minimize computational burden. Specifically, the instruction condition is injected into the grouped visual tokens at the local level and the learnable tokens at the global level, and we conduct the attention mechanism to complete the conditional compression. From the hybrid-level compression, the instruction-relevant visual parts are highlighted while the temporal-spatial structure is also preserved for easier understanding of LLMs. To further unleash the potential of HICom, we introduce a new conditional pre-training stage with our proposed dataset HICom-248K. Experiments show that our HICom can obtain distinguished video understanding ability with fewer tokens, increasing the performance by 2.43\% average on three multiple-choice QA benchmarks and saving 78.8\% tokens compared with the SOTA method. The code is available at this https URL. 

---
# Autonomous AI imitators increase diversity in homogeneous information ecosystems 

**Authors**: Emil Bakkensen Johansen, Oliver Baumann  

**Link**: [PDF](https://arxiv.org/pdf/2503.16021)  

**Abstract**: Recent breakthroughs in large language models (LLMs) have facilitated autonomous AI agents capable of imitating human-generated content. This technological advancement raises fundamental questions about AI's potential impact on the diversity and democratic value of information ecosystems. Here, we introduce a large-scale simulation framework to examine AI-based imitation in news, a context critically influential for public discourse. By systematically testing two distinct imitation strategies across a range of information environments varying in initial diversity, we demonstrate that AI-generated articles do not uniformly homogenize content. Instead, AI's influence is strongly context-dependent: AI-generated articles can introduce valuable diversity in originally homogeneous news environments, while potentially diminishing diversity in contexts that initially display high heterogeneity. These results illustrate that the baseline diversity of an information space critically shapes AI's impact, challenging assumptions that AI-driven imitation uniformly threatens information diversity. Instead, when information is initially homogeneous, AI-driven imitation can expand perspectives, styles, and topics. This is especially important in news contexts, where information diversity fosters richer public debate by exposing citizens to alternative viewpoints, challenging biases, and preventing narrative monopolies, which is essential for a resilient democracy. 

---
# Don't Fight Hallucinations, Use Them: Estimating Image Realism using NLI over Atomic Facts 

**Authors**: Elisei Rykov, Kseniia Petrushina, Kseniia Titova, Alexander Panchenko, Vasily Konovalov  

**Link**: [PDF](https://arxiv.org/pdf/2503.15948)  

**Abstract**: Quantifying the realism of images remains a challenging problem in the field of artificial intelligence. For example, an image of Albert Einstein holding a smartphone violates common-sense because modern smartphone were invented after Einstein's death. We introduce a novel method for assessing image realism using Large Vision-Language Models (LVLMs) and Natural Language Inference (NLI). Our approach is based on the premise that LVLMs may generate hallucinations when confronted with images that defy common sense. Using LVLM to extract atomic facts from these images, we obtain a mix of accurate facts and erroneous hallucinations. We proceed by calculating pairwise entailment scores among these facts, subsequently aggregating these values to yield a singular reality score. This process serves to identify contradictions between genuine facts and hallucinatory elements, signaling the presence of images that violate common sense. Our approach has achieved a new state-of-the-art performance in zero-shot mode on the WHOOPS! dataset. 

---
# InCo-DPO: Balancing Distribution Shift and Data Quality for Enhanced Preference Optimization 

**Authors**: Yunan Wang, Jijie Li, Bo-Wen Zhang, Liangdong Wang, Guang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15880)  

**Abstract**: Direct Preference Optimization (DPO) optimizes language models to align with human preferences. Utilizing on-policy samples, generated directly by the policy model, typically results in better performance due to its distribution consistency with the model compared to off-policy samples. This paper identifies the quality of candidate preference samples as another critical factor. While the quality of on-policy data is inherently constrained by the capabilities of the policy model, off-policy data, which can be derived from diverse sources, offers greater potential for quality despite experiencing distribution shifts. However, current research mostly relies on on-policy data and neglects the value of off-policy data in terms of data quality, due to the challenge posed by distribution shift. In this paper, we propose InCo-DPO, an efficient method for synthesizing preference data by integrating on-policy and off-policy data, allowing dynamic adjustments to balance distribution shifts and data quality, thus finding an optimal trade-off. Consequently, InCo-DPO overcomes the limitations of distribution shifts in off-policy data and the quality constraints of on-policy data. We evaluated InCo-DPO with the Alpaca-Eval 2.0 and Arena-Hard benchmarks. Experimental results demonstrate that our approach not only outperforms both on-policy and off-policy data but also achieves a state-of-the-art win rate of 60.8 on Arena-Hard with the vanilla DPO using Gemma-2 model. 

---
# Entropy-based Exploration Conduction for Multi-step Reasoning 

**Authors**: Jinghan Zhang, Xiting Wang, Fengran Mo, Yeyang Zhou, Wanfu Gao, Kunpeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15848)  

**Abstract**: In large language model (LLM) reasoning, multi-step processes have proven effective for solving complex tasks. However, the depth of exploration can significantly affect the reasoning performance. Existing methods to automatically decide the depth often bring high costs and lack flexibility, and thus undermine the model's reasoning accuracy. To address these issues, we propose Entropy-based Exploration Depth Conduction (Entro-duction), a novel method that dynamically adjusts the exploration depth during multi-step reasoning by monitoring LLM's output entropy and variance entropy. We employ these two metrics to capture the model's current uncertainty and the fluctuation of uncertainty across consecutive reasoning steps. Based on the observed changes, the LLM selects whether to deepen, expand or stop exploration according to the probability. In this way, we balance the reasoning accuracy and exploration effectiveness. Experimental results across four benchmark datasets demonstrate the efficacy of Entro-duction. We further conduct experiments and analysis on the components of Entro-duction to discuss their contributions to reasoning performance. 

---
# ChatGPT and U(X): A Rapid Review on Measuring the User Experience 

**Authors**: Katie Seaborn  

**Link**: [PDF](https://arxiv.org/pdf/2503.15808)  

**Abstract**: ChatGPT, powered by a large language model (LLM), has revolutionized everyday human-computer interaction (HCI) since its 2022 release. While now used by millions around the world, a coherent pathway for evaluating the user experience (UX) ChatGPT offers remains missing. In this rapid review (N = 58), I explored how ChatGPT UX has been approached quantitatively so far. I focused on the independent variables (IVs) manipulated, the dependent variables (DVs) measured, and the methods used for measurement. Findings reveal trends, gaps, and emerging consensus in UX assessments. This work offers a first step towards synthesizing existing approaches to measuring ChatGPT UX, urgent trajectories to advance standardization and breadth, and two preliminary frameworks aimed at guiding future research and tool development. I seek to elevate the field of ChatGPT UX by empowering researchers and practitioners in optimizing user interactions with ChatGPT and similar LLM-based systems. 

---
# Mixture of Lookup Experts 

**Authors**: Shibo Jie, Yehui Tang, Kai Han, Yitong Li, Duyu Tang, Zhi-Hong Deng, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.15798)  

**Abstract**: Mixture-of-Experts (MoE) activates only a subset of experts during inference, allowing the model to maintain low inference FLOPs and latency even as the parameter count scales up. However, since MoE dynamically selects the experts, all the experts need to be loaded into VRAM. Their large parameter size still limits deployment, and offloading, which load experts into VRAM only when needed, significantly increase inference latency. To address this, we propose Mixture of Lookup Experts (MoLE), a new MoE architecture that is efficient in both communication and VRAM usage. In MoLE, the experts are Feed-Forward Networks (FFNs) during training, taking the output of the embedding layer as input. Before inference, these experts can be re-parameterized as lookup tables (LUTs) that retrieves expert outputs based on input ids, and offloaded to storage devices. Therefore, we do not need to perform expert computations during inference. Instead, we directly retrieve the expert's computation results based on input ids and load them into VRAM, and thus the resulting communication overhead is negligible. Experiments show that, with the same FLOPs and VRAM usage, MoLE achieves inference speeds comparable to dense models and significantly faster than MoE with experts offloading, while maintaining performance on par with MoE. 

---
# UI-Vision: A Desktop-centric GUI Benchmark for Visual Perception and Interaction 

**Authors**: Shravan Nayak, Xiangru Jian, Kevin Qinghong Lin, Juan A. Rodriguez, Montek Kalsi, Rabiul Awal, Nicolas Chapados, M. Tamer Özsu, Aishwarya Agrawal, David Vazquez, Christopher Pal, Perouz Taslakian, Spandana Gella, Sai Rajeswar  

**Link**: [PDF](https://arxiv.org/pdf/2503.15661)  

**Abstract**: Autonomous agents that navigate Graphical User Interfaces (GUIs) to automate tasks like document editing and file management can greatly enhance computer workflows. While existing research focuses on online settings, desktop environments, critical for many professional and everyday tasks, remain underexplored due to data collection challenges and licensing issues. We introduce UI-Vision, the first comprehensive, license-permissive benchmark for offline, fine-grained evaluation of computer use agents in real-world desktop environments. Unlike online benchmarks, UI-Vision provides: (i) dense, high-quality annotations of human demonstrations, including bounding boxes, UI labels, and action trajectories (clicks, drags, and keyboard inputs) across 83 software applications, and (ii) three fine-to-coarse grained tasks-Element Grounding, Layout Grounding, and Action Prediction-with well-defined metrics to rigorously evaluate agents' performance in desktop environments. Our evaluation reveals critical limitations in state-of-the-art models like UI-TARS-72B, including issues with understanding professional software, spatial reasoning, and complex actions like drag-and-drop. These findings highlight the challenges in developing fully autonomous computer use agents. By releasing UI-Vision as open-source, we aim to advance the development of more capable agents for real-world desktop tasks. 

---
# LLaVA-MORE: A Comparative Study of LLMs and Visual Backbones for Enhanced Visual Instruction Tuning 

**Authors**: Federico Cocchi, Nicholas Moratelli, Davide Caffagni, Sara Sarto, Lorenzo Baraldi, Marcella Cornia, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2503.15621)  

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) has highlighted the critical roles of both the visual backbone and the underlying language model. While prior work has primarily focused on scaling these components to billions of parameters, the trade-offs between model size, architecture, and performance remain underexplored. Additionally, inconsistencies in training data and evaluation protocols have hindered direct comparisons, making it difficult to derive optimal design choices. In this paper, we introduce LLaVA-MORE, a new family of MLLMs that integrates recent language models with diverse visual backbones. To ensure fair comparisons, we employ a unified training protocol applied consistently across all architectures. Our analysis systematically explores both small- and medium-scale LLMs -- including Phi-4, LLaMA-3.1, and Gemma-2 -- to evaluate multimodal reasoning, generation, and instruction following, while examining the relationship between model size and performance. Beyond evaluating the LLM impact on final results, we conduct a comprehensive study of various visual encoders, ranging from CLIP-based architectures to alternatives such as DINOv2, SigLIP, and SigLIP2. Additional experiments investigate the effects of increased image resolution and variations in pre-training datasets. Overall, our results provide insights into the design of more effective MLLMs, offering a reproducible evaluation framework that facilitates direct comparisons and can guide future model development. Our source code and trained models are publicly available at: this https URL. 

---
# Personalized Attacks of Social Engineering in Multi-turn Conversations -- LLM Agents for Simulation and Detection 

**Authors**: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15552)  

**Abstract**: The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts. 

---
# From Divergence to Consensus: Evaluating the Role of Large Language Models in Facilitating Agreement through Adaptive Strategies 

**Authors**: Loukas Triantafyllopoulos, Dimitris Kalles  

**Link**: [PDF](https://arxiv.org/pdf/2503.15521)  

**Abstract**: Achieving consensus in group decision-making often involves overcoming significant challenges, particularly in reconciling diverse perspectives and mitigating biases that hinder agreement. Traditional methods relying on human facilitators are often constrained by scalability and efficiency, especially in large-scale, fast-paced discussions. To address these challenges, this study proposes a novel framework employing large language models (LLMs) as automated facilitators within a custom-built multi-user chat system. Leveraging cosine similarity as a core metric, this approach evaluates the ability of three state-of-the-art LLMs- ChatGPT 4.0, Mistral Large 2, and AI21 Jamba Instruct- to synthesize consensus proposals that align with participants' viewpoints. Unlike conventional techniques, the system integrates adaptive facilitation strategies, including clarifying misunderstandings, summarizing discussions, and proposing compromises, enabling the LLMs to iteratively refine consensus proposals based on user feedback. Experimental results demonstrate the superiority of ChatGPT 4.0, which achieves higher alignment with participant opinions, requiring fewer iterations to reach consensus compared to its counterparts. Moreover, analysis reveals the nuanced performance of the models across various sustainability-focused discussion topics, such as climate action, quality education, good health and well-being, and access to clean water and sanitation. These findings highlight the transformative potential of LLM-driven facilitation for improving collective decision-making processes and underscore the importance of advancing evaluation metrics and cross-cultural adaptability in future research. 

---
# Superhuman AI Disclosure: Impacts on Toxicity, Fairness, and Trust Vary by Expertise and Persona Attributes 

**Authors**: Jaymari Chua, Chen Wang, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.15514)  

**Abstract**: As artificial intelligence demonstrates surpassing human performance across real-world tasks, disclosing superhuman capabilities poses challenges for fairness, accountability, and trust. To investigate how transparency impacts attitudes and perceptions, we introduce a grounded and validated set of synthetic personas reflecting diverse fairness concerns and technology acceptance levels. Then we evaluate responses in two contrasting domains: (1) a competitive player in StarCraft II, where strategy and high-skill gameplay often elicit toxic interactions, and (2) a cooperative personal-assistant in providing information. Across numerous interactions spanning persona profiles, we test non-disclosure versus explicit superhuman labelling under controlled game outcomes and usage contexts. Our findings reveal sharp domain-specific effects: in StarCraft II, explicitly labelling AI as superhuman, novice personas who learned of it reported lower toxicity and higher fairness-attributing defeat to advanced skill rather than hidden cheating-whereas expert personas found the disclosure statements irksome but still less deceptive than non-disclosure. Conversely, in the LLM as personal-assistant setting, disclosure of superhuman capabilities improved perceived trustworthiness, though it risked AI overreliance among certain persona segments. We release Dataset X-containing persona cards-including profile attributes, disclosure prompts, and detailed interaction logs, accompanied by reproducible protocols and disclaimers for adapting them to diverse tasks. Our results demonstrate that transparency is not a cure-all: while it reduces suspicion and enhances trust in cooperative contexts, it may inflame resistance or disappointment in competitive domains. 

---
# Representing data in words 

**Authors**: Amandine M. Caut, Amy Rouillard, Beimnet Zenebe, Matthias Green, Ágúst Pálmason Morthens, David J. T. Sumpter  

**Link**: [PDF](https://arxiv.org/pdf/2503.15509)  

**Abstract**: An important part of data science is the use of visualisations to display data in a way that is easy to digest. Visualisations often rely on underlying statistical or machine learning models -- ranging from basic calculations like category means to advanced methods such as principal component analysis of multidimensional datasets -- to convey insights. We introduce an analogous concept for word descriptions of data, which we call wordalisations. Wordalisations describe data in easy to digest words, without necessarily reporting numerical values from the data. We show how to create wordalisations using large language models, through prompt templates engineered according to a task-agnostic structure which can be used to automatically generate prompts from data. We show how to produce reliable and engaging texts on three application areas: scouting football players, personality tests, and international survey data. Using the model cards framework, we emphasise the importance of clearly stating the model we are imposing on the data when creating the wordalisation, detailing how numerical values are translated into words, incorporating background information into prompts for the large language model, and documenting the limitations of the wordalisations. We argue that our model cards approach is a more appropriate framework for setting best practices in wordalisation of data than performance tests on benchmark datasets. 

---
# Agreeing to Interact in Human-Robot Interaction using Large Language Models and Vision Language Models 

**Authors**: Kazuhiro Sasabuchi, Naoki Wake, Atsushi Kanehira, Jun Takamatsu, Katsushi Ikeuchi  

**Link**: [PDF](https://arxiv.org/pdf/2503.15491)  

**Abstract**: In human-robot interaction (HRI), the beginning of an interaction is often complex. Whether the robot should communicate with the human is dependent on several situational factors (e.g., the current human's activity, urgency of the interaction, etc.). We test whether large language models (LLM) and vision language models (VLM) can provide solutions to this problem. We compare four different system-design patterns using LLMs and VLMs, and test on a test set containing 84 human-robot situations. The test set mixes several publicly available datasets and also includes situations where the appropriate action to take is open-ended. Our results using the GPT-4o and Phi-3 Vision model indicate that LLMs and VLMs are capable of handling interaction beginnings when the desired actions are clear, however, challenge remains in the open-ended situations where the model must balance between the human and robot situation. 

---
