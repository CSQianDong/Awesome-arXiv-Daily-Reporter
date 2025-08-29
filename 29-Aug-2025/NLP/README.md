# Enabling Equitable Access to Trustworthy Financial Reasoning 

**Authors**: William Jurayj, Nils Holzenberger, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2508.21051)  

**Abstract**: According to the United States Internal Revenue Service, ''the average American spends $\$270$ and 13 hours filing their taxes''. Even beyond the U.S., tax filing requires complex reasoning, combining application of overlapping rules with numerical calculations. Because errors can incur costly penalties, any automated system must deliver high accuracy and auditability, making modern large language models (LLMs) poorly suited for this task. We propose an approach that integrates LLMs with a symbolic solver to calculate tax obligations. We evaluate variants of this system on the challenging StAtutory Reasoning Assessment (SARA) dataset, and include a novel method for estimating the cost of deploying such a system based on real-world penalties for tax errors. We further show how combining up-front translation of plain-text rules into formal logic programs, combined with intelligently retrieved exemplars for formal case representations, can dramatically improve performance on this task and reduce costs to well below real-world averages. Our results demonstrate the promise and economic feasibility of neuro-symbolic architectures for increasing equitable access to reliable tax assistance. 

---
# Re-Representation in Sentential Relation Extraction with Sequence Routing Algorithm 

**Authors**: Ramazan Ali Bahrami, Ramin Yahyapour  

**Link**: [PDF](https://arxiv.org/pdf/2508.21049)  

**Abstract**: Sentential relation extraction (RE) is an important task in natural language processing (NLP). In this paper we propose to do sentential RE with dynamic routing in capsules. We first show that the proposed approach outperform state of the art on common sentential relation extraction datasets Tacred, Tacredrev, Retacred, and Conll04. We then investigate potential reasons for its good performance on the mentioned datasets, and yet low performance on another similar, yet larger sentential RE dataset, Wikidata. As such, we identify noise in Wikidata labels as one of the reasons that can hinder performance. Additionally, we show associativity of better performance with better re-representation, a term from neuroscience referred to change of representation in human brain to improve the match at comparison time. As example, in the given analogous terms King:Queen::Man:Woman, at comparison time, and as a result of re-representation, the similarity between related head terms (King,Man), and tail terms (Queen,Woman) increases. As such, our observation show that our proposed model can do re-representation better than the vanilla model compared with. To that end, beside noise in the labels of the distantly supervised RE datasets, we propose re-representation as a challenge in sentential RE. 

---
# An Agile Method for Implementing Retrieval Augmented Generation Tools in Industrial SMEs 

**Authors**: Mathieu Bourdin, Anas Neumann, Thomas Paviot, Robert Pellerin, Samir Lamouri  

**Link**: [PDF](https://arxiv.org/pdf/2508.21024)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful solution to mitigate the limitations of Large Language Models (LLMs), such as hallucinations and outdated knowledge. However, deploying RAG-based tools in Small and Medium Enterprises (SMEs) remains a challenge due to their limited resources and lack of expertise in natural language processing (NLP). This paper introduces EASI-RAG, Enterprise Application Support for Industrial RAG, a structured, agile method designed to facilitate the deployment of RAG systems in industrial SME contexts. EASI-RAG is based on method engineering principles and comprises well-defined roles, activities, and techniques. The method was validated through a real-world case study in an environmental testing laboratory, where a RAG tool was implemented to answer operators queries using data extracted from operational procedures. The system was deployed in under a month by a team with no prior RAG experience and was later iteratively improved based on user feedback. Results demonstrate that EASI-RAG supports fast implementation, high user adoption, delivers accurate answers, and enhances the reliability of underlying data. This work highlights the potential of RAG deployment in industrial SMEs. Future works include the need for generalization across diverse use cases and further integration with fine-tuned models. 

---
# Lethe: Purifying Backdoored Large Language Models with Knowledge Dilution 

**Authors**: Chen Chen, Yuchen Sun, Jiaxin Gao, Xueluan Gong, Qian Wang, Ziyao Wang, Yongsen Zheng, Kwok-Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2508.21004)  

**Abstract**: Large language models (LLMs) have seen significant advancements, achieving superior performance in various Natural Language Processing (NLP) tasks. However, they remain vulnerable to backdoor attacks, where models behave normally for standard queries but generate harmful responses or unintended output when specific triggers are activated. Existing backdoor defenses either lack comprehensiveness, focusing on narrow trigger settings, detection-only mechanisms, and limited domains, or fail to withstand advanced scenarios like model-editing-based, multi-trigger, and triggerless attacks. In this paper, we present LETHE, a novel method to eliminate backdoor behaviors from LLMs through knowledge dilution using both internal and external mechanisms. Internally, LETHE leverages a lightweight dataset to train a clean model, which is then merged with the backdoored model to neutralize malicious behaviors by diluting the backdoor impact within the model's parametric memory. Externally, LETHE incorporates benign and semantically relevant evidence into the prompt to distract LLM's attention from backdoor features. Experimental results on classification and generation domains across 5 widely used LLMs demonstrate that LETHE outperforms 8 state-of-the-art defense baselines against 8 backdoor attacks. LETHE reduces the attack success rate of advanced backdoor attacks by up to 98% while maintaining model utility. Furthermore, LETHE has proven to be cost-efficient and robust against adaptive backdoor attacks. 

---
# ProactiveEval: A Unified Evaluation Framework for Proactive Dialogue Agents 

**Authors**: Tianjian Liu, Fanqi Wan, Jiajian Guo, Xiaojun Quan  

**Link**: [PDF](https://arxiv.org/pdf/2508.20973)  

**Abstract**: Proactive dialogue has emerged as a critical and challenging research problem in advancing large language models (LLMs). Existing works predominantly focus on domain-specific or task-oriented scenarios, which leads to fragmented evaluations and limits the comprehensive exploration of models' proactive conversation abilities. In this work, we propose ProactiveEval, a unified framework designed for evaluating proactive dialogue capabilities of LLMs. This framework decomposes proactive dialogue into target planning and dialogue guidance, establishing evaluation metrics across various domains. Moreover, it also enables the automatic generation of diverse and challenging evaluation data. Based on the proposed framework, we develop 328 evaluation environments spanning 6 distinct domains. Through experiments with 22 different types of LLMs, we show that DeepSeek-R1 and Claude-3.7-Sonnet exhibit exceptional performance on target planning and dialogue guidance tasks, respectively. Finally, we investigate how reasoning capabilities influence proactive behaviors and discuss their implications for future model development. 

---
# STARE at the Structure: Steering ICL Exemplar Selection with Structural Alignment 

**Authors**: Jiaqian Li, Qisheng Hu, Jing Li, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20944)  

**Abstract**: In-Context Learning (ICL) has become a powerful paradigm that enables LLMs to perform a wide range of tasks without task-specific fine-tuning. However, the effectiveness of ICL heavily depends on the quality of exemplar selection. In particular, for structured prediction tasks such as semantic parsing, existing ICL selection strategies often overlook structural alignment, leading to suboptimal performance and poor generalization. To address this issue, we propose a novel two-stage exemplar selection strategy that achieves a strong balance between efficiency, generalizability, and performance. First, we fine-tune a BERT-based retriever using structure-aware supervision, guiding it to select exemplars that are both semantically relevant and structurally aligned. Then, we enhance the retriever with a plug-in module, which amplifies syntactically meaningful information in the hidden representations. This plug-in is model-agnostic, requires minimal overhead, and can be seamlessly integrated into existing pipelines. Experiments on four benchmarks spanning three semantic parsing tasks demonstrate that our method consistently outperforms existing baselines with multiple recent LLMs as inference-time models. 

---
# How Can Input Reformulation Improve Tool Usage Accuracy in a Complex Dynamic Environment? A Study on $τ$-bench 

**Authors**: Venkatesh Mishra, Amir Saeidi, Satyam Raj, Mutsumi Nakamura, Jayanth Srinivasa, Gaowen Liu, Ali Payani, Chitta Baral  

**Link**: [PDF](https://arxiv.org/pdf/2508.20931)  

**Abstract**: Recent advances in reasoning and planning capabilities of large language models (LLMs) have enabled their potential as autonomous agents capable of tool use in dynamic environments. However, in multi-turn conversational environments like $\tau$-bench, these agents often struggle with consistent reasoning, adherence to domain-specific policies, and extracting correct information over a long horizon of tool-calls and conversation. To capture and mitigate these failures, we conduct a comprehensive manual analysis of the common errors occurring in the conversation trajectories. We then experiment with reformulations of inputs to the tool-calling agent for improvement in agent decision making. Finally, we propose the Input-Reformulation Multi-Agent (IRMA) framework, which automatically reformulates user queries augmented with relevant domain rules and tool suggestions for the tool-calling agent to focus on. The results show that IRMA significantly outperforms ReAct, Function Calling, and Self-Reflection by 16.1%, 12.7%, and 19.1%, respectively, in overall pass^5 scores. These findings highlight the superior reliability and consistency of IRMA compared to other methods in dynamic environments. 

---
# SageLM: A Multi-aspect and Explainable Large Language Model for Speech Judgement 

**Authors**: Yuan Ge, Junxiang Zhang, Xiaoqian Liu, Bei Li, Xiangnan Ma, Chenglong Wang, Kaiyang Ye, Yangfan Du, Linfeng Zhang, Yuxin Huang, Tong Xiao, Zhengtao Yu, JingBo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20916)  

**Abstract**: Speech-to-Speech (S2S) Large Language Models (LLMs) are foundational to natural human-computer interaction, enabling end-to-end spoken dialogue systems. However, evaluating these models remains a fundamental challenge. We propose \texttt{SageLM}, an end-to-end, multi-aspect, and explainable speech LLM for comprehensive S2S LLMs evaluation. First, unlike cascaded approaches that disregard acoustic features, SageLM jointly assesses both semantic and acoustic dimensions. Second, it leverages rationale-based supervision to enhance explainability and guide model learning, achieving superior alignment with evaluation outcomes compared to rule-based reinforcement learning methods. Third, we introduce \textit{SpeechFeedback}, a synthetic preference dataset, and employ a two-stage training paradigm to mitigate the scarcity of speech preference data. Trained on both semantic and acoustic dimensions, SageLM achieves an 82.79\% agreement rate with human evaluators, outperforming cascaded and SLM-based baselines by at least 7.42\% and 26.20\%, respectively. 

---
# The Uneven Impact of Post-Training Quantization in Machine Translation 

**Authors**: Benjamin Marie, Atsushi Fujita  

**Link**: [PDF](https://arxiv.org/pdf/2508.20893)  

**Abstract**: Quantization is essential for deploying large language models (LLMs) on resource-constrained hardware, but its implications for multilingual tasks remain underexplored. We conduct the first large-scale evaluation of post-training quantization (PTQ) on machine translation across 55 languages using five LLMs ranging from 1.7B to 70B parameters. Our analysis reveals that while 4-bit quantization often preserves translation quality for high-resource languages and large models, significant degradation occurs for low-resource and typologically diverse languages, particularly in 2-bit settings. We compare four quantization techniques (AWQ, BitsAndBytes, GGUF, and AutoRound), showing that algorithm choice and model size jointly determine robustness. GGUF variants provide the most consistent performance, even at 2-bit precision. Additionally, we quantify the interactions between quantization, decoding hyperparameters, and calibration languages, finding that language-matched calibration offers benefits primarily in low-bit scenarios. Our findings offer actionable insights for deploying multilingual LLMs for machine translation under quantization constraints, especially in low-resource settings. 

---
# MSRS: Evaluating Multi-Source Retrieval-Augmented Generation 

**Authors**: Rohan Phanse, Yijie Zhou, Kejian Shi, Wencai Zhang, Yixin Liu, Yilun Zhao, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2508.20867)  

**Abstract**: Retrieval-augmented systems are typically evaluated in settings where information required to answer the query can be found within a single source or the answer is short-form or factoid-based. However, many real-world applications demand the ability to integrate and summarize information scattered across multiple sources, where no single source is sufficient to respond to the user's question. In such settings, the retrieval component of a RAG pipeline must recognize a variety of relevance signals, and the generation component must connect and synthesize information across multiple sources. We present a scalable framework for constructing evaluation benchmarks that challenge RAG systems to integrate information across distinct sources and generate long-form responses. Using our framework, we build two new benchmarks on Multi-Source Retrieval and Synthesis: MSRS-Story and MSRS-Meet, representing narrative synthesis and summarization tasks, respectively, that require retrieval from large collections. Our extensive experiments with various RAG pipelines -- including sparse and dense retrievers combined with frontier LLMs -- reveal that generation quality is highly dependent on retrieval effectiveness, which varies greatly by task. While multi-source synthesis proves challenging even in an oracle retrieval setting, we find that reasoning models significantly outperform standard LLMs at this distinct step. 

---
# GDLLM: A Global Distance-aware Modeling Approach Based on Large Language Models for Event Temporal Relation Extraction 

**Authors**: Jie Zhao, Wanting Ning, Yuxiao Fei, Yubo Feng, Lishuang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.20828)  

**Abstract**: In Natural Language Processing(NLP), Event Temporal Relation Extraction (ETRE) is to recognize the temporal relations of two events. Prior studies have noted the importance of language models for ETRE. However, the restricted pre-trained knowledge of Small Language Models(SLMs) limits their capability to handle minority class relations in imbalanced classification datasets. For Large Language Models(LLMs), researchers adopt manually designed prompts or instructions, which may introduce extra noise, leading to interference with the model's judgment of the long-distance dependencies between events. To address these issues, we propose GDLLM, a Global Distance-aware modeling approach based on LLMs. We first present a distance-aware graph structure utilizing Graph Attention Network(GAT) to assist the LLMs in capturing long-distance dependency features. Additionally, we design a temporal feature learning paradigm based on soft inference to augment the identification of relations with a short-distance proximity band, which supplements the probabilistic information generated by LLMs into the multi-head attention mechanism. Since the global feature can be captured effectively, our framework substantially enhances the performance of minority relation classes and improves the overall learning ability. Experiments on two publicly available datasets, TB-Dense and MATRES, demonstrate that our approach achieves state-of-the-art (SOTA) performance. 

---
# Exploring Machine Learning and Language Models for Multimodal Depression Detection 

**Authors**: Javier Si Zhao Hong, Timothy Zoe Delaya, Sherwyn Chan Yin Kit, Pai Chet Ng, Xiaoxiao Miao  

**Link**: [PDF](https://arxiv.org/pdf/2508.20805)  

**Abstract**: This paper presents our approach to the first Multimodal Personality-Aware Depression Detection Challenge, focusing on multimodal depression detection using machine learning and deep learning models. We explore and compare the performance of XGBoost, transformer-based architectures, and large language models (LLMs) on audio, video, and text features. Our results highlight the strengths and limitations of each type of model in capturing depression-related signals across modalities, offering insights into effective multimodal representation strategies for mental health prediction. 

---
# Signs of Struggle: Spotting Cognitive Distortions across Language and Register 

**Authors**: Abhishek Kuber, Enrico Liscio, Ruixuan Zhang, Caroline Figueroa, Pradeep K. Murukannaiah  

**Link**: [PDF](https://arxiv.org/pdf/2508.20771)  

**Abstract**: Rising mental health issues among youth have increased interest in automated approaches for detecting early signs of psychological distress in digital text. One key focus is the identification of cognitive distortions, irrational thought patterns that have a role in aggravating mental distress. Early detection of these distortions may enable timely, low-cost interventions. While prior work has focused on English clinical data, we present the first in-depth study of cross-lingual and cross-register generalization of cognitive distortion detection, analyzing forum posts written by Dutch adolescents. Our findings show that while changes in language and writing style can significantly affect model performance, domain adaptation methods show the most promise. 

---
# Turning the Spell Around: Lightweight Alignment Amplification via Rank-One Safety Injection 

**Authors**: Harethah Abu Shairah, Hasan Abed Al Kader Hammoud, George Turkiyyah, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2508.20766)  

**Abstract**: Safety alignment in Large Language Models (LLMs) often involves mediating internal representations to refuse harmful requests. Recent research has demonstrated that these safety mechanisms can be bypassed by ablating or removing specific representational directions within the model. In this paper, we propose the opposite approach: Rank-One Safety Injection (ROSI), a white-box method that amplifies a model's safety alignment by permanently steering its activations toward the refusal-mediating subspace. ROSI operates as a simple, fine-tuning-free rank-one weight modification applied to all residual stream write matrices. The required safety direction can be computed from a small set of harmful and harmless instruction pairs. We show that ROSI consistently increases safety refusal rates - as evaluated by Llama Guard 3 - while preserving the utility of the model on standard benchmarks such as MMLU, HellaSwag, and Arc. Furthermore, we show that ROSI can also re-align 'uncensored' models by amplifying their own latent safety directions, demonstrating its utility as an effective last-mile safety procedure. Our results suggest that targeted, interpretable weight steering is a cheap and potent mechanism to improve LLM safety, complementing more resource-intensive fine-tuning paradigms. 

---
# Feel the Difference? A Comparative Analysis of Emotional Arcs in Real and LLM-Generated CBT Sessions 

**Authors**: Xiaoyi Wang, Jiwei Zhang, Guangtao Zhang, Honglei Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.20764)  

**Abstract**: Synthetic therapy dialogues generated by large language models (LLMs) are increasingly used in mental health NLP to simulate counseling scenarios, train models, and supplement limited real-world data. However, it remains unclear whether these synthetic conversations capture the nuanced emotional dynamics of real therapy. In this work, we conduct the first comparative analysis of emotional arcs between real and LLM-generated Cognitive Behavioral Therapy dialogues. We adapt the Utterance Emotion Dynamics framework to analyze fine-grained affective trajectories across valence, arousal, and dominance dimensions. Our analysis spans both full dialogues and individual speaker roles (counselor and client), using real sessions transcribed from public videos and synthetic dialogues from the CACTUS dataset. We find that while synthetic dialogues are fluent and structurally coherent, they diverge from real conversations in key emotional properties: real sessions exhibit greater emotional variability,more emotion-laden language, and more authentic patterns of reactivity and regulation. Moreover, emotional arc similarity between real and synthetic speakers is low, especially for clients. These findings underscore the limitations of current LLM-generated therapy data and highlight the importance of emotional fidelity in mental health applications. We introduce RealCBT, a curated dataset of real CBT sessions, to support future research in this space. 

---
# GUARD: Glocal Uncertainty-Aware Robust Decoding for Effective and Efficient Open-Ended Text Generation 

**Authors**: Yuanhao Ding, Esteban Garces Arias, Meimingwei Li, Julian Rodemann, Matthias Aßenmacher, Danlu Chen, Gaojuan Fan, Christian Heumann, Chongsheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20757)  

**Abstract**: Open-ended text generation faces a critical challenge: balancing coherence with diversity in LLM outputs. While contrastive search-based decoding strategies have emerged to address this trade-off, their practical utility is often limited by hyperparameter dependence and high computational costs. We introduce GUARD, a self-adaptive decoding method that effectively balances these competing objectives through a novel "Glocal" uncertainty-driven framework. GUARD combines global entropy estimates with local entropy deviations to integrate both long-term and short-term uncertainty signals. We demonstrate that our proposed global entropy formulation effectively mitigates abrupt variations in uncertainty, such as sudden overconfidence or high entropy spikes, and provides theoretical guarantees of unbiasedness and consistency. To reduce computational overhead, we incorporate a simple yet effective token-count-based penalty into GUARD. Experimental results demonstrate that GUARD achieves a good balance between text diversity and coherence, while exhibiting substantial improvements in generation speed. In a more nuanced comparison study across different dimensions of text quality, both human and LLM evaluators validated its remarkable performance. Our code is available at this https URL. 

---
# Specializing General-purpose LLM Embeddings for Implicit Hate Speech Detection across Datasets 

**Authors**: Vassiliy Cheremetiev, Quang Long Ho Ngo, Chau Ying Kot, Alina Elena Baia, Andrea Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2508.20750)  

**Abstract**: Implicit hate speech (IHS) is indirect language that conveys prejudice or hatred through subtle cues, sarcasm or coded terminology. IHS is challenging to detect as it does not include explicit derogatory or inflammatory words. To address this challenge, task-specific pipelines can be complemented with external knowledge or additional information such as context, emotions and sentiment data. In this paper, we show that, by solely fine-tuning recent general-purpose embedding models based on large language models (LLMs), such as Stella, Jasper, NV-Embed and E5, we achieve state-of-the-art performance. Experiments on multiple IHS datasets show up to 1.10 percentage points improvements for in-dataset, and up to 20.35 percentage points improvements in cross-dataset evaluation, in terms of F1-macro score. 

---
# Leveraging Semantic Triples for Private Document Generation with Local Differential Privacy Guarantees 

**Authors**: Stephen Meisenbacher, Maulik Chevli, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2508.20736)  

**Abstract**: Many works at the intersection of Differential Privacy (DP) in Natural Language Processing aim to protect privacy by transforming texts under DP guarantees. This can be performed in a variety of ways, from word perturbations to full document rewriting, and most often under local DP. Here, an input text must be made indistinguishable from any other potential text, within some bound governed by the privacy parameter $\varepsilon$. Such a guarantee is quite demanding, and recent works show that privatizing texts under local DP can only be done reasonably under very high $\varepsilon$ values. Addressing this challenge, we introduce DP-ST, which leverages semantic triples for neighborhood-aware private document generation under local DP guarantees. Through the evaluation of our method, we demonstrate the effectiveness of the divide-and-conquer paradigm, particularly when limiting the DP notion (and privacy guarantees) to that of a privatization neighborhood. When combined with LLM post-processing, our method allows for coherent text generation even at lower $\varepsilon$ values, while still balancing privacy and utility. These findings highlight the importance of coherence in achieving balanced privatization outputs at reasonable $\varepsilon$ levels. 

---
# rStar2-Agent: Agentic Reasoning Technical Report 

**Authors**: Ning Shang, Yifei Liu, Yi Zhu, Li Lyna Zhang, Weijiang Xu, Xinyu Guan, Buze Zhang, Bingcheng Dong, Xudong Zhou, Bowen Zhang, Ying Xin, Ziming Miao, Scarlett Li, Fan Yang, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20722)  

**Abstract**: We introduce rStar2-Agent, a 14B math reasoning model trained with agentic reinforcement learning to achieve frontier-level performance. Beyond current long CoT, the model demonstrates advanced cognitive behaviors, such as thinking carefully before using Python coding tools and reflecting on code execution feedback to autonomously explore, verify, and refine intermediate steps in complex problem-solving. This capability is enabled through three key innovations that makes agentic RL effective at scale: (i) an efficient RL infrastructure with a reliable Python code environment that supports high-throughput execution and mitigates the high rollout costs, enabling training on limited GPU resources (64 MI300X GPUs); (ii) GRPO-RoC, an agentic RL algorithm with a Resample-on-Correct rollout strategy that addresses the inherent environment noises from coding tools, allowing the model to reason more effectively in a code environment; (iii) An efficient agent training recipe that starts with non-reasoning SFT and progresses through multi-RL stages, yielding advanced cognitive abilities with minimal compute cost. To this end, rStar2-Agent boosts a pre-trained 14B model to state of the art in only 510 RL steps within one week, achieving average pass@1 scores of 80.6% on AIME24 and 69.8% on AIME25, surpassing DeepSeek-R1 (671B) with significantly shorter responses. Beyond mathematics, rStar2-Agent-14B also demonstrates strong generalization to alignment, scientific reasoning, and agentic tool-use tasks. Code and training recipes are available at this https URL. 

---
# Addressing Tokenization Inconsistency in Steganography and Watermarking Based on Large Language Models 

**Authors**: Ruiyi Yan, Yugo Murawaki  

**Link**: [PDF](https://arxiv.org/pdf/2508.20718)  

**Abstract**: Large language models have significantly enhanced the capacities and efficiency of text generation. On the one hand, they have improved the quality of text-based steganography. On the other hand, they have also underscored the importance of watermarking as a safeguard against malicious misuse. In this study, we focus on tokenization inconsistency (TI) between Alice and Bob in steganography and watermarking, where TI can undermine robustness. Our investigation reveals that the problematic tokens responsible for TI exhibit two key characteristics: infrequency and temporariness. Based on these findings, we propose two tailored solutions for TI elimination: a stepwise verification method for steganography and a post-hoc rollback method for watermarking. Experiments show that (1) compared to traditional disambiguation methods in steganography, directly addressing TI leads to improvements in fluency, imperceptibility, and anti-steganalysis capacity; (2) for watermarking, addressing TI enhances detectability and robustness against attacks. 

---
# Multi-Lingual Implicit Discourse Relation Recognition with Multi-Label Hierarchical Learning 

**Authors**: Nelson Filipe Costa, Leila Kosseim  

**Link**: [PDF](https://arxiv.org/pdf/2508.20712)  

**Abstract**: This paper introduces the first multi-lingual and multi-label classification model for implicit discourse relation recognition (IDRR). Our model, HArch, is evaluated on the recently released DiscoGeM 2.0 corpus and leverages hierarchical dependencies between discourse senses to predict probability distributions across all three sense levels in the PDTB 3.0 framework. We compare several pre-trained encoder backbones and find that RoBERTa-HArch achieves the best performance in English, while XLM-RoBERTa-HArch performs best in the multi-lingual setting. In addition, we compare our fine-tuned models against GPT-4o and Llama-4-Maverick using few-shot prompting across all language configurations. Our results show that our fine-tuned models consistently outperform these LLMs, highlighting the advantages of task-specific fine-tuning over prompting in IDRR. Finally, we report SOTA results on the DiscoGeM 1.0 corpus, further validating the effectiveness of our hierarchical approach. 

---
# Generative Annotation for ASR Named Entity Correction 

**Authors**: Yuanchang Luo, Daimeng Wei, Shaojun Li, Hengchao Shang, Jiaxin Guo, Zongyao Li, Zhanglin Wu, Xiaoyu Chen, Zhiqiang Rao, Jinlong Yang, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20700)  

**Abstract**: End-to-end automatic speech recognition systems often fail to transcribe domain-specific named entities, causing catastrophic failures in downstream tasks. Numerous fast and lightweight named entity correction (NEC) models have been proposed in recent years. These models, mainly leveraging phonetic-level edit distance algorithms, have shown impressive performances. However, when the forms of the wrongly-transcribed words(s) and the ground-truth entity are significantly different, these methods often fail to locate the wrongly transcribed words in hypothesis, thus limiting their usage. We propose a novel NEC method that utilizes speech sound features to retrieve candidate entities. With speech sound features and candidate entities, we inovatively design a generative method to annotate entity errors in ASR transcripts and replace the text with correct entities. This method is effective in scenarios of word form difference. We test our method using open-source and self-constructed test sets. The results demonstrate that our NEC method can bring significant improvement to entity accuracy. We will open source our self-constructed test set and training data. 

---
# A Graph Talks, But Who's Listening? Rethinking Evaluations for Graph-Language Models 

**Authors**: Soham Petkar, Hari Aakash K, Anirudh Vempati, Akshit Sinha, Ponnurangam Kumarauguru, Chirag Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2508.20583)  

**Abstract**: Developments in Graph-Language Models (GLMs) aim to integrate the structural reasoning capabilities of Graph Neural Networks (GNNs) with the semantic understanding of Large Language Models (LLMs). However, we demonstrate that current evaluation benchmarks for GLMs, which are primarily repurposed node-level classification datasets, are insufficient to assess multimodal reasoning. Our analysis reveals that strong performance on these benchmarks is achievable using unimodal information alone, suggesting that they do not necessitate graph-language integration. To address this evaluation gap, we introduce the CLEGR(Compositional Language-Graph Reasoning) benchmark, designed to evaluate multimodal reasoning at various complexity levels. Our benchmark employs a synthetic graph generation pipeline paired with questions that require joint reasoning over structure and textual semantics. We perform a thorough evaluation of representative GLM architectures and find that soft-prompted LLM baselines perform on par with GLMs that incorporate a full GNN backbone. This result calls into question the architectural necessity of incorporating graph structure into LLMs. We further show that GLMs exhibit significant performance degradation in tasks that require structural reasoning. These findings highlight limitations in the graph reasoning capabilities of current GLMs and provide a foundation for advancing the community toward explicit multimodal reasoning involving graph structure and language. 

---
# KCS: Diversify Multi-hop Question Generation with Knowledge Composition Sampling 

**Authors**: Yangfan Wang, Jie Liu, Chen Tang, Lian Yan, Jingchi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20567)  

**Abstract**: Multi-hop question answering faces substantial challenges due to data sparsity, which increases the likelihood of language models learning spurious patterns. To address this issue, prior research has focused on diversifying question generation through content planning and varied expression. However, these approaches often emphasize generating simple questions and neglect the integration of essential knowledge, such as relevant sentences within documents. This paper introduces the Knowledge Composition Sampling (KCS), an innovative framework designed to expand the diversity of generated multi-hop questions by sampling varied knowledge compositions within a given context. KCS models the knowledge composition selection as a sentence-level conditional prediction task and utilizes a probabilistic contrastive loss to predict the next most relevant piece of knowledge. During inference, we employ a stochastic decoding strategy to effectively balance accuracy and diversity. Compared to competitive baselines, our KCS improves the overall accuracy of knowledge composition selection by 3.9%, and its application for data augmentation yields improvements on HotpotQA and 2WikiMultihopQA datasets. Our code is available at: this https URL. 

---
# Leveraging Generative Models for Real-Time Query-Driven Text Summarization in Large-Scale Web Search 

**Authors**: Zeyu Xiong, Yixuan Nan, Li Gao, Hengzhu Tang, Shuaiqiang Wang, Junfeng Wang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.20559)  

**Abstract**: In the dynamic landscape of large-scale web search, Query-Driven Text Summarization (QDTS) aims to generate concise and informative summaries from textual documents based on a given query, which is essential for improving user engagement and facilitating rapid decision-making. Traditional extractive summarization models, based primarily on ranking candidate summary segments, have been the dominant approach in industrial applications. However, these approaches suffer from two key limitations: 1) The multi-stage pipeline often introduces cumulative information loss and architectural bottlenecks due to its weakest component; 2) Traditional models lack sufficient semantic understanding of both user queries and documents, particularly when dealing with complex search intents. In this study, we propose a novel framework to pioneer the application of generative models to address real-time QDTS in industrial web search. Our approach integrates large model distillation, supervised fine-tuning, direct preference optimization, and lookahead decoding to transform a lightweight model with only 0.1B parameters into a domain-specialized QDTS expert. Evaluated on multiple industry-relevant metrics, our model outperforms the production baseline and achieves a new state of the art. Furthermore, it demonstrates excellent deployment efficiency, requiring only 334 NVIDIA L20 GPUs to handle \textasciitilde50,000 queries per second under 55~ms average latency per query. 

---
# Adaptive Federated Distillation for Multi-Domain Non-IID Textual Data 

**Authors**: Jiahao Xiao, Jiangming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20557)  

**Abstract**: The widespread success of pre-trained language models has established a new training paradigm, where a global PLM is fine-tuned using task-specific data from local clients. The local data are highly different from each other and can not capture the global distribution of the whole data in real world. To address the challenges of non-IID data in real environments, privacy-preserving federated distillation has been proposed and highly investigated. However, previous experimental non-IID scenarios are primarily identified with the label (output) diversity, without considering the diversity of language domains (input) that is crucial in natural language processing. In this paper, we introduce a comprehensive set of multi-domain non-IID scenarios and propose a unified benchmarking framework that includes diverse data. The benchmark can be used to evaluate the federated learning framework in a real environment. To this end, we propose an Adaptive Federated Distillation (AdaFD) framework designed to address multi-domain non-IID challenges in both homogeneous and heterogeneous settings. Experimental results demonstrate that our models capture the diversity of local clients and achieve better performance compared to the existing works. The code for this paper is available at: this https URL. 

---
# Overview of BioASQ 2025: The Thirteenth BioASQ Challenge on Large-Scale Biomedical Semantic Indexing and Question Answering 

**Authors**: Anastasios Nentidis, Georgios Katsimpras, Anastasia Krithara, Martin Krallinger, Miguel Rodríguez-Ortega, Eduard Rodriguez-López, Natalia Loukachevitch, Andrey Sakhovskiy, Elena Tutubalina, Dimitris Dimitriadis, Grigorios Tsoumakas, George Giannakoulas, Alexandra Bekiaridou, Athanasios Samaras, Giorgio Maria Di Nunzio, Nicola Ferro, Stefano Marchesin, Marco Martinelli, Gianmaria Silvello, Georgios Paliouras  

**Link**: [PDF](https://arxiv.org/pdf/2508.20554)  

**Abstract**: This is an overview of the thirteenth edition of the BioASQ challenge in the context of the Conference and Labs of the Evaluation Forum (CLEF) 2025. BioASQ is a series of international challenges promoting advances in large-scale biomedical semantic indexing and question answering. This year, BioASQ consisted of new editions of the two established tasks, b and Synergy, and four new tasks: a) Task MultiClinSum on multilingual clinical summarization. b) Task BioNNE-L on nested named entity linking in Russian and English. c) Task ELCardioCC on clinical coding in cardiology. d) Task GutBrainIE on gut-brain interplay information extraction. In this edition of BioASQ, 83 competing teams participated with more than 1000 distinct submissions in total for the six different shared tasks of the challenge. Similar to previous editions, several participating systems achieved competitive performance, indicating the continuous advancement of the state-of-the-art in the field. 

---
# Overview of BioASQ 2024: The twelfth BioASQ challenge on Large-Scale Biomedical Semantic Indexing and Question Answering 

**Authors**: Anastasios Nentidis, Georgios Katsimpras, Anastasia Krithara, Salvador Lima-López, Eulàlia Farré-Maduell, Martin Krallinger, Natalia Loukachevitch, Vera Davydova, Elena Tutubalina, Georgios Paliouras  

**Link**: [PDF](https://arxiv.org/pdf/2508.20532)  

**Abstract**: This is an overview of the twelfth edition of the BioASQ challenge in the context of the Conference and Labs of the Evaluation Forum (CLEF) 2024. BioASQ is a series of international challenges promoting advances in large-scale biomedical semantic indexing and question answering. This year, BioASQ consisted of new editions of the two established tasks b and Synergy, and two new tasks: a) MultiCardioNER on the adaptation of clinical entity detection to the cardiology domain in a multilingual setting, and b) BIONNE on nested NER in Russian and English. In this edition of BioASQ, 37 competing teams participated with more than 700 distinct submissions in total for the four different shared tasks of the challenge. Similarly to previous editions, most of the participating systems achieved competitive performance, suggesting the continuous advancement of the state-of-the-art in the field. 

---
# SciTopic: Enhancing Topic Discovery in Scientific Literature through Advanced LLM 

**Authors**: Pengjiang Li, Zaitian Wang, Xinhao Zhang, Ran Zhang, Lu Jiang, Pengfei Wang, Yuanchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.20514)  

**Abstract**: Topic discovery in scientific literature provides valuable insights for researchers to identify emerging trends and explore new avenues for investigation, facilitating easier scientific information retrieval. Many machine learning methods, particularly deep embedding techniques, have been applied to discover research topics. However, most existing topic discovery methods rely on word embedding to capture the semantics and lack a comprehensive understanding of scientific publications, struggling with complex, high-dimensional text relationships. Inspired by the exceptional comprehension of textual information by large language models (LLMs), we propose an advanced topic discovery method enhanced by LLMs to improve scientific topic identification, namely SciTopic. Specifically, we first build a textual encoder to capture the content from scientific publications, including metadata, title, and abstract. Next, we construct a space optimization module that integrates entropy-based sampling and triplet tasks guided by LLMs, enhancing the focus on thematic relevance and contextual intricacies between ambiguous instances. Then, we propose to fine-tune the textual encoder based on the guidance from the LLMs by optimizing the contrastive loss of the triplets, forcing the text encoder to better discriminate instances of different topics. Finally, extensive experiments conducted on three real-world datasets of scientific publications demonstrate that SciTopic outperforms the state-of-the-art (SOTA) scientific topic discovery methods, enabling researchers to gain deeper and faster insights. 

---
# Languages Still Left Behind: Toward a Better Multilingual Machine Translation Benchmark 

**Authors**: Chihiro Taguchi, Seng Mai, Keita Kurabe, Yusuke Sakai, Georgina Agyei, Soudabeh Eslami, David Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20511)  

**Abstract**: Multilingual machine translation (MT) benchmarks play a central role in evaluating the capabilities of modern MT systems. Among them, the FLORES+ benchmark is widely used, offering English-to-many translation data for over 200 languages, curated with strict quality control protocols. However, we study data in four languages (Asante Twi, Japanese, Jinghpaw, and South Azerbaijani) and uncover critical shortcomings in the benchmark's suitability for truly multilingual evaluation. Human assessments reveal that many translations fall below the claimed 90% quality standard, and the annotators report that source sentences are often too domain-specific and culturally biased toward the English-speaking world. We further demonstrate that simple heuristics, such as copying named entities, can yield non-trivial BLEU scores, suggesting vulnerabilities in the evaluation protocol. Notably, we show that MT models trained on high-quality, naturalistic data perform poorly on FLORES+ while achieving significant gains on our domain-relevant evaluation set. Based on these findings, we advocate for multilingual MT benchmarks that use domain-general and culturally neutral source texts rely less on named entities, in order to better reflect real-world translation challenges. 

---
# ConspirED: A Dataset for Cognitive Traits of Conspiracy Theories and Large Language Model Safety 

**Authors**: Luke Bates, Max Glockner, Preslav Nakov, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2508.20468)  

**Abstract**: Conspiracy theories erode public trust in science and institutions while resisting debunking by evolving and absorbing counter-evidence. As AI-generated misinformation becomes increasingly sophisticated, understanding rhetorical patterns in conspiratorial content is important for developing interventions such as targeted prebunking and assessing AI vulnerabilities. We introduce ConspirED (CONSPIR Evaluation Dataset), which captures the cognitive traits of conspiratorial ideation in multi-sentence excerpts (80--120 words) from online conspiracy articles, annotated using the CONSPIR cognitive framework (Lewandowsky and Cook, 2020). ConspirED is the first dataset of conspiratorial content annotated for general cognitive traits. Using ConspirED, we (i) develop computational models that identify conspiratorial traits and determine dominant traits in text excerpts, and (ii) evaluate large language/reasoning model (LLM/LRM) robustness to conspiratorial inputs. We find that both are misaligned by conspiratorial content, producing output that mirrors input reasoning patterns, even when successfully deflecting comparable fact-checked misinformation. 

---
# Prediction of mortality and resource utilization in critical care: a deep learning approach using multimodal electronic health records with natural language processing techniques 

**Authors**: Yucheng Ruan, Xiang Lan, Daniel J. Tan, Hairil Rizal Abdullah, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2508.20460)  

**Abstract**: Background Predicting mortality and resource utilization from electronic health records (EHRs) is challenging yet crucial for optimizing patient outcomes and managing costs in intensive care unit (ICU). Existing approaches predominantly focus on structured EHRs, often ignoring the valuable clinical insights in free-text notes. Additionally, the potential of textual information within structured data is not fully leveraged. This study aimed to introduce and assess a deep learning framework using natural language processing techniques that integrates multimodal EHRs to predict mortality and resource utilization in critical care settings. Methods Utilizing two real-world EHR datasets, we developed and evaluated our model on three clinical tasks with leading existing methods. We also performed an ablation study on three key components in our framework: medical prompts, free-texts, and pre-trained sentence encoder. Furthermore, we assessed the model's robustness against the corruption in structured EHRs. Results Our experiments on two real-world datasets across three clinical tasks showed that our proposed model improved performance metrics by 1.6\%/0.8\% on BACC/AUROC for mortality prediction, 0.5%/2.2% on RMSE/MAE for LOS prediction, 10.9%/11.0% on RMSE/MAE for surgical duration estimation compared to the best existing methods. It consistently demonstrated superior performance compared to other baselines across three tasks at different corruption rates. Conclusions The proposed framework is an effective and accurate deep learning approach for predicting mortality and resource utilization in critical care. The study also highlights the success of using prompt learning with a transformer encoder in analyzing multimodal EHRs. Importantly, the model showed strong resilience to data corruption within structured data, especially at high corruption levels. 

---
# MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers 

**Authors**: Zhenting Wang, Qi Chang, Hemani Patel, Shashank Biju, Cheng-En Wu, Quan Liu, Aolin Ding, Alireza Rezazadeh, Ankit Shah, Yujia Bao, Eugene Siow  

**Link**: [PDF](https://arxiv.org/pdf/2508.20453)  

**Abstract**: We introduce MCP-Bench, a benchmark for evaluating large language models (LLMs) on realistic, multi-step tasks that demand tool use, cross-tool coordination, precise parameter control, and planning/reasoning for solving tasks. Built on the Model Context Protocol (MCP), MCP-Bench connects LLMs to 28 representative live MCP servers spanning 250 tools across domains such as finance, traveling, scientific computing, and academic search. Unlike prior API-based benchmarks, each MCP server provides a set of complementary tools designed to work together, enabling the construction of authentic, multi-step tasks with rich input-output coupling. Tasks in MCP-Bench test agents' ability to retrieve relevant tools from fuzzy instructions without explicit tool names, plan multi-hop execution trajectories for complex objectives, ground responses in intermediate tool outputs, and orchestrate cross-domain workflows - capabilities not adequately evaluated by existing benchmarks that rely on explicit tool specifications, shallow few-step workflows, and isolated domain operations. We propose a multi-faceted evaluation framework covering tool-level schema understanding and usage, trajectory-level planning, and task completion. Experiments on 20 advanced LLMs reveal persistent challenges in MCP-Bench. Code and data: this https URL. 

---
# Searching the Title of Practical Work of the Informatics Engineering Bachelor Program with the Case Base Reasoning Method 

**Authors**: Agung Sukrisna Jaya, Osvari Arsalan, Danny Matthew Saputra  

**Link**: [PDF](https://arxiv.org/pdf/2508.20442)  

**Abstract**: Case Base Reasoning (CBR) is a case solving technique based on experience in cases that have occurred before with the highest similarity. CBR is used to search for practical work titles. TF-IDF is applied to process the vectorization of each practical work title word and Cosine Similarity for the calculation of similarity values. This system can search either in the form of titles or keywords. The output of the system is the title of practical work and the match value of each title. Based on the test results using 705 practical work titles, testing was carried out with five titles and carried out in two stages. The first stage searches with existing titles and the second stage randomizes the title from the first stage. And the results obtained in the second stage are the same number of titles found and the highest average match score. 

---
# CAMB: A comprehensive industrial LLM benchmark on civil aviation maintenance 

**Authors**: Feng Zhang, Chengjie Pang, Yuehan Zhang, Chenyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.20420)  

**Abstract**: Civil aviation maintenance is a domain characterized by stringent industry standards. Within this field, maintenance procedures and troubleshooting represent critical, knowledge-intensive tasks that require sophisticated reasoning. To address the lack of specialized evaluation tools for large language models (LLMs) in this vertical, we propose and develop an industrial-grade benchmark specifically designed for civil aviation maintenance. This benchmark serves a dual purpose: It provides a standardized tool to measure LLM capabilities within civil aviation maintenance, identifying specific gaps in domain knowledge and complex reasoning. By pinpointing these deficiencies, the benchmark establishes a foundation for targeted improvement efforts (e.g., domain-specific fine-tuning, RAG optimization, or specialized prompt engineering), ultimately facilitating progress toward more intelligent solutions within civil aviation maintenance. Our work addresses a significant gap in the current LLM evaluation, which primarily focuses on mathematical and coding reasoning tasks. In addition, given that Retrieval-Augmented Generation (RAG) systems are currently the dominant solutions in practical applications , we leverage this benchmark to evaluate existing well-known vector embedding models and LLMs for civil aviation maintenance scenarios. Through experimental exploration and analysis, we demonstrate the effectiveness of our benchmark in assessing model performance within this domain, and we open-source this evaluation benchmark and code to foster further research and development:this https URL 

---
# KG-CQR: Leveraging Structured Relation Representations in Knowledge Graphs for Contextual Query Retrieval 

**Authors**: Chi Minh Bui, Ngoc Mai Thieu, Van Vinh Nguyen, Json J.Jung, Khac-Hoai Nam Bui  

**Link**: [PDF](https://arxiv.org/pdf/2508.20417)  

**Abstract**: The integration of knowledge graphs (KGs) with large language models (LLMs) offers significant potential to improve the retrieval phase of retrieval-augmented generation (RAG) systems. In this study, we propose KG-CQR, a novel framework for Contextual Query Retrieval (CQR) that enhances the retrieval phase by enriching the contextual representation of complex input queries using a corpus-centric KG. Unlike existing methods that primarily address corpus-level context loss, KG-CQR focuses on query enrichment through structured relation representations, extracting and completing relevant KG subgraphs to generate semantically rich query contexts. Comprising subgraph extraction, completion, and contextual generation modules, KG-CQR operates as a model-agnostic pipeline, ensuring scalability across LLMs of varying sizes without additional training. Experimental results on RAGBench and MultiHop-RAG datasets demonstrate KG-CQR's superior performance, achieving a 4-6% improvement in mAP and a 2-3% improvement in Recall@25 over strong baseline models. Furthermore, evaluations on challenging RAG tasks such as multi-hop question answering show that, by incorporating KG-CQR, the performance consistently outperforms the existing baseline in terms of retrieval effectiveness 

---
# DentalBench: Benchmarking and Advancing LLMs Capability for Bilingual Dentistry Understanding 

**Authors**: Hengchuan Zhu, Yihuan Xu, Yichen Li, Zijie Meng, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20416)  

**Abstract**: Recent advances in large language models (LLMs) and medical LLMs (Med-LLMs) have demonstrated strong performance on general medical benchmarks. However, their capabilities in specialized medical fields, such as dentistry which require deeper domain-specific knowledge, remain underexplored due to the lack of targeted evaluation resources. In this paper, we introduce DentalBench, the first comprehensive bilingual benchmark designed to evaluate and advance LLMs in the dental domain. DentalBench consists of two main components: DentalQA, an English-Chinese question-answering (QA) benchmark with 36,597 questions spanning 4 tasks and 16 dental subfields; and DentalCorpus, a large-scale, high-quality corpus with 337.35 million tokens curated for dental domain adaptation, supporting both supervised fine-tuning (SFT) and retrieval-augmented generation (RAG). We evaluate 14 LLMs, covering proprietary, open-source, and medical-specific models, and reveal significant performance gaps across task types and languages. Further experiments with Qwen-2.5-3B demonstrate that domain adaptation substantially improves model performance, particularly on knowledge-intensive and terminology-focused tasks, and highlight the importance of domain-specific benchmarks for developing trustworthy and effective LLMs tailored to healthcare applications. 

---
# UI-Bench: A Benchmark for Evaluating Design Capabilities of AI Text-to-App Tools 

**Authors**: Sam Jung, Agustin Garcinuno, Spencer Mateega  

**Link**: [PDF](https://arxiv.org/pdf/2508.20410)  

**Abstract**: AI text-to-app tools promise high quality applications and websites in minutes, yet no public benchmark rigorously verifies those claims. We introduce UI-Bench, the first large-scale benchmark that evaluates visual excellence across competing AI text-to-app tools through expert pairwise comparison. Spanning 10 tools, 30 prompts, 300 generated sites, and \textit{4000+} expert judgments, UI-Bench ranks systems with a TrueSkill-derived model that yields calibrated confidence intervals. UI-Bench establishes a reproducible standard for advancing AI-driven web design. We release (i) the complete prompt set, (ii) an open-source evaluation framework, and (iii) a public leaderboard. The generated sites rated by participants will be released soon. View the UI-Bench leaderboard at this https URL. 

---
# Measuring Reasoning Utility in LLMs via Conditional Entropy Reduction 

**Authors**: Xu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.20395)  

**Abstract**: Recent advancements in large language models (LLMs) often rely on generating intermediate reasoning steps to enhance accuracy. However, little work has examined how reasoning utility contributes to the final answer's correctness. Due to the stochastic nature of autoregressive generation, generating more context does not guarantee increased confidence in the answer. If we could predict, during generation, whether a reasoning step will be useful, we could stop early or prune ineffective steps, avoiding distractions in the final decision.
We present an oracle study on MATH dataset, using Qwen2.5-32B and GPT-4o to generate reasoning chains, and then employing a separate model (Qwen3-8B) to quantify the utility of these chains for final accuracy. Specifically, we measure the model's uncertainty on the answer span Y at each reasoning step using conditional entropy (expected negative log-likelihood over the vocabulary) with context expanding step by step. Our results show a clear pattern: conditional entropy that decreases over steps is strongly associated with correct answers, whereas flat or increasing entropy often results in wrong answers. We also corroborate that incorrect reasoning paths tend to be longer than correct ones, suggesting that longer reasoning does not necessarily yield better outcomes. These findings serve as a foundation to inspire future work on designing efficient reasoning pipelines that detect and avoid unproductive reasoning early. 

---
# CAPE: Context-Aware Personality Evaluation Framework for Large Language Models 

**Authors**: Jivnesh Sandhan, Fei Cheng, Tushar Sandhan, Yugo Murawaki  

**Link**: [PDF](https://arxiv.org/pdf/2508.20385)  

**Abstract**: Psychometric tests, traditionally used to assess humans, are now being applied to Large Language Models (LLMs) to evaluate their behavioral traits. However, existing studies follow a context-free approach, answering each question in isolation to avoid contextual influence. We term this the Disney World test, an artificial setting that ignores real-world applications, where conversational history shapes responses. To bridge this gap, we propose the first Context-Aware Personality Evaluation (CAPE) framework for LLMs, incorporating prior conversational interactions. To thoroughly analyze the influence of context, we introduce novel metrics to quantify the consistency of LLM responses, a fundamental trait in human behavior.
Our exhaustive experiments on 7 LLMs reveal that conversational history enhances response consistency via in-context learning but also induces personality shifts, with GPT-3.5-Turbo and GPT-4-Turbo exhibiting extreme deviations. While GPT models are robust to question ordering, Gemini-1.5-Flash and Llama-8B display significant sensitivity. Moreover, GPT models response stem from their intrinsic personality traits as well as prior interactions, whereas Gemini-1.5-Flash and Llama--8B heavily depend on prior interactions. Finally, applying our framework to Role Playing Agents (RPAs) shows context-dependent personality shifts improve response consistency and better align with human judgments. Our code and datasets are publicly available at: this https URL 

---
# Graph-R1: Unleashing LLM Reasoning with NP-Hard Graph Problems 

**Authors**: Yuyao Wang, Bowen Liu, Jianheng Tang, Nuo Chen, Yuhan Li, Qifan Zhang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.20373)  

**Abstract**: Reasoning Large Language Models (RLLMs) have recently achieved remarkable progress on complex reasoning tasks, largely enabled by their long chain-of-thought (Long CoT) capabilities. However, developing these Long CoT behaviors relies heavily on post-training with high-quality datasets, which are typically costly and human-curated (e.g., mathematics and code), leaving scalable alternatives unexplored. In this work, we introduce NP-hard (NPH) graph problems as a novel synthetic training corpus, as they inherently require deep reasoning, extensive exploration, and reflective strategies, which are core characteristics of Long CoT reasoning. Building on this insight, we develop a two-stage post-training framework: (i) Long CoT Supervised Fine-Tuning (SFT) on rejection-sampled NPH graph instances, which substantially enhances reasoning depth, and (ii) Reinforcement Learning (RL) with a fine-grained reward design, which sharpens reasoning efficiency. Our flagship model, Graph-R1-7B, demonstrates strong generalization across mathematics, coding, STEM, and logic, and surpasses QwQ-32B on NPH graph problems in both accuracy and reasoning efficiency. These results position NPH graph problems as an effective and scalable resource for advancing Long CoT reasoning in LLMs, opening a new frontier for LLM post-training. Our implementation is available at this https URL, with models and datasets hosted in our Hugging Face collection HKUST-DSAIL/Graph-R1. 

---
# Joint Enhancement of Relational Reasoning for Long-Context LLMs 

**Authors**: Zhirui Chen, Wei Shen, Jiashui Huang, Ling Shao  

**Link**: [PDF](https://arxiv.org/pdf/2508.20351)  

**Abstract**: Despite significant progress, large language models (LLMs) still struggle with long contexts due to memory limitations and their inability to tackle complex and long-context tasks. Additionally, LLMs often suffer from a lack of transparency and are prone to producing hallucinations. To address these challenges, we propose \textbf{JERR}, a novel framework designed to enhance long-context comprehension via graph-based reasoning in LLMs. JERR integrates three key components: synopsis extraction, graph construction, and relational reasoning. First, synopsis is extracted by chunking text strategically, allowing the model to summarize and understand information more efficiently. Second, we build a directed acyclic graph (DAG) to resolve redundancy, ensuring logical consistency and clarity. Finally, we incorporate Monte Carlo Tree Search (MCTS) to help the model navigate complex reasoning paths, ensuring more accurate and interpretable outputs. This framework provides a novel solution that enables LLMs to handle extended contexts and complex reasoning tasks with improved reliability and transparency. Experimental results show that JERR consistently outperforms all baselines on the ROUGE and F1 metrics, achieving the highest scores on the LLM-Rater evaluation. 

---
# GUARD: Guideline Upholding Test through Adaptive Role-play and Jailbreak Diagnostics for LLMs 

**Authors**: Haibo Jin, Ruoxi Chen, Peiyan Zhang, Andy Zhou, Yang Zhang, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20325)  

**Abstract**: As Large Language Models become increasingly integral to various domains, their potential to generate harmful responses has prompted significant societal and regulatory concerns. In response, governments have issued ethics guidelines to promote the development of trustworthy AI. However, these guidelines are typically high-level demands for developers and testers, leaving a gap in translating them into actionable testing questions to verify LLM compliance.
To address this challenge, we introduce GUARD (\textbf{G}uideline \textbf{U}pholding Test through \textbf{A}daptive \textbf{R}ole-play and Jailbreak \textbf{D}iagnostics), a testing method designed to operationalize guidelines into specific guideline-violating questions that assess LLM adherence. To implement this, GUARD uses automated generation of guideline-violating questions based on government-issued guidelines, thereby testing whether responses comply with these guidelines. When responses directly violate guidelines, GUARD reports inconsistencies. Furthermore, for responses that do not directly violate guidelines, GUARD integrates the concept of ``jailbreaks'' to diagnostics, named GUARD-JD, which creates scenarios that provoke unethical or guideline-violating responses, effectively identifying potential scenarios that could bypass built-in safety mechanisms. Our method finally culminates in a compliance report, delineating the extent of adherence and highlighting any violations.
We have empirically validated the effectiveness of GUARD on seven LLMs, including Vicuna-13B, LongChat-7B, Llama2-7B, Llama-3-8B, GPT-3.5, GPT-4, GPT-4o, and Claude-3.7, by testing compliance under three government-issued guidelines and conducting jailbreak diagnostics. Additionally, GUARD-JD can transfer jailbreak diagnostics to vision-language models, demonstrating its usage in promoting reliable LLM-based applications. 

---
# Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities 

**Authors**: Rikuto Kotoge, Mai Nishimura, Jiaxin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.20324)  

**Abstract**: Reinforcement Learning has emerged as a post-training approach to elicit agentic RAG behaviors such as search and planning from language models. However, compact language models (e.g., 0.5B parameters) struggle due to poor reasoning ability, resulting in sparse rewards and unstable training. To overcome these difficulties, we propose Distillation-Guided Policy Optimization (DGPO), which addresses the challenges through cold-start initialization from teacher demonstrations and continuous teacher guidance during policy optimization. To systematically evaluate our approach, we introduce Agentic RAG Capabilities (ARC), a fine-grained metric analyzing reasoning, search coordination, and response synthesis. Comprehensive experiments demonstrate that DGPO enables compact models to achieve sophisticated agentic search behaviors, even outperforming the larger teacher model in some cases. DGPO makes agentic RAG feasible in computing resource-constrained environments. 

---
# Integrating SystemC TLM into FMI 3.0 Co-Simulations with an Open-Source Approach 

**Authors**: Andrei Mihai Albu, Giovanni Pollo, Alessio Burrello, Daniele Jahier Pagliari, Cristian Tesconi, Alessandra Neri, Dario Soldi, Fabio Autieri, Sara Vinco  

**Link**: [PDF](https://arxiv.org/pdf/2508.20223)  

**Abstract**: The growing complexity of cyber-physical systems, particularly in automotive applications, has increased the demand for efficient modeling and cross-domain co-simulation techniques. While SystemC Transaction-Level Modeling (TLM) enables effective hardware/software co-design, its limited interoperability with models from other engineering domains poses integration challenges. This paper presents a fully open-source methodology for integrating SystemC TLM models into Functional Mock-up Interface (FMI)-based co-simulation workflows. By encapsulating SystemC TLM components as FMI 3.0 Co Simulation Functional Mock-up Units (FMUs), the proposed approach facilitates seamless, standardized integration across heterogeneous simulation environments. We introduce a lightweight open-source toolchain, address key technical challenges such as time synchronization and data exchange, and demonstrate the feasibility and effectiveness of the integration through representative case studies. 

---
# Prompting Strategies for Language Model-Based Item Generation in K-12 Education: Bridging the Gap Between Small and Large Language Models 

**Authors**: Mohammad Amini, Babak Ahmadi, Xiaomeng Xiong, Yilin Zhang, Christopher Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.20217)  

**Abstract**: This study explores automatic generation (AIG) using language models to create multiple choice questions (MCQs) for morphological assessment, aiming to reduce the cost and inconsistency of manual test development. The study used a two-fold approach. First, we compared a fine-tuned medium model (Gemma, 2B) with a larger untuned one (GPT-3.5, 175B). Second, we evaluated seven structured prompting strategies, including zero-shot, few-shot, chain-of-thought, role-based, sequential, and combinations. Generated items were assessed using automated metrics and expert scoring across five dimensions. We also used GPT-4.1, trained on expert-rated samples, to simulate human scoring at scale. Results show that structured prompting, especially strategies combining chain-of-thought and sequential design, significantly improved Gemma's outputs. Gemma generally produced more construct-aligned and instructionally appropriate items than GPT-3.5's zero-shot responses, with prompt design playing a key role in mid-size model performance. This study demonstrates that structured prompting and efficient fine-tuning can enhance midsized models for AIG under limited data conditions. We highlight the value of combining automated metrics, expert judgment, and large-model simulation to ensure alignment with assessment goals. The proposed workflow offers a practical and scalable way to develop and validate language assessment items for K-12. 

---
# Social Bias in Multilingual Language Models: A Survey 

**Authors**: Lance Calvin Lim Gamboa, Yue Feng, Mark Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.20201)  

**Abstract**: Pretrained multilingual models exhibit the same social bias as models processing English texts. This systematic review analyzes emerging research that extends bias evaluation and mitigation approaches into multilingual and non-English contexts. We examine these studies with respect to linguistic diversity, cultural awareness, and their choice of evaluation metrics and mitigation techniques. Our survey illuminates gaps in the field's dominant methodological design choices (e.g., preference for certain languages, scarcity of multilingual mitigation experiments) while cataloging common issues encountered and solutions implemented in adapting bias benchmarks across languages and cultures. Drawing from the implications of our findings, we chart directions for future research that can reinforce the multilingual bias literature's inclusivity, cross-cultural appropriateness, and alignment with state-of-the-art NLP advancements. 

---
# On the Theoretical Limitations of Embedding-Based Retrieval 

**Authors**: Orion Weller, Michael Boratko, Iftekhar Naim, Jinhyuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.21038)  

**Abstract**: Vector embeddings have been tasked with an ever-increasing set of retrieval tasks over the years, with a nascent rise in using them for reasoning, instruction-following, coding, and more. These new benchmarks push embeddings to work for any query and any notion of relevance that could be given. While prior works have pointed out theoretical limitations of vector embeddings, there is a common assumption that these difficulties are exclusively due to unrealistic queries, and those that are not can be overcome with better training data and larger models. In this work, we demonstrate that we may encounter these theoretical limitations in realistic settings with extremely simple queries. We connect known results in learning theory, showing that the number of top-k subsets of documents capable of being returned as the result of some query is limited by the dimension of the embedding. We empirically show that this holds true even if we restrict to k=2, and directly optimize on the test set with free parameterized embeddings. We then create a realistic dataset called LIMIT that stress tests models based on these theoretical results, and observe that even state-of-the-art models fail on this dataset despite the simple nature of the task. Our work shows the limits of embedding models under the existing single vector paradigm and calls for future research to develop methods that can resolve this fundamental limitation. 

---
# ChainReaction! Structured Approach with Causal Chains as Intermediate Representations for Improved and Explainable Causal Video Question Answering 

**Authors**: Paritosh Parmar, Eric Peh, Basura Fernando  

**Link**: [PDF](https://arxiv.org/pdf/2508.21010)  

**Abstract**: Existing Causal-Why Video Question Answering (VideoQA) models often struggle with higher-order reasoning, relying on opaque, monolithic pipelines that entangle video understanding, causal inference, and answer generation. These black-box approaches offer limited interpretability and tend to depend on shallow heuristics. We propose a novel, modular framework that explicitly decouples causal reasoning from answer generation, introducing natural language causal chains as interpretable intermediate representations. Inspired by human cognitive models, these structured cause-effect sequences bridge low-level video content with high-level causal reasoning, enabling transparent and logically coherent inference. Our two-stage architecture comprises a Causal Chain Extractor (CCE) that generates causal chains from video-question pairs, and a Causal Chain-Driven Answerer (CCDA) that produces answers grounded in these chains. To address the lack of annotated reasoning traces, we introduce a scalable method for generating high-quality causal chains from existing datasets using large language models. We also propose CauCo, a new evaluation metric for causality-oriented captioning. Experiments on three large-scale benchmarks demonstrate that our approach not only outperforms state-of-the-art models, but also yields substantial gains in explainability, user trust, and generalization -- positioning the CCE as a reusable causal reasoning engine across diverse domains. Project page: this https URL 

---
# OLMoASR: Open Models and Data for Training Robust Speech Recognition Models 

**Authors**: Huong Ngo, Matt Deitke, Martijn Bartelds, Sarah Pratt, Josh Gardner, Matt Jordan, Ludwig Schmidt  

**Link**: [PDF](https://arxiv.org/pdf/2508.20869)  

**Abstract**: Improvements in training data scale and quality have led to significant advances, yet its influence in speech recognition remains underexplored. In this paper, we present a large-scale dataset, OLMoASR-Pool, and series of models, OLMoASR, to study and develop robust zero-shot speech recognition models. Beginning from OLMoASR-Pool, a collection of 3M hours of English audio and 17M transcripts, we design text heuristic filters to remove low-quality or mistranscribed data. Our curation pipeline produces a new dataset containing 1M hours of high-quality audio-transcript pairs, which we call OLMoASR-Mix. We use OLMoASR-Mix to train the OLMoASR-Mix suite of models, ranging from 39M (this http URL) to 1.5B (this http URL) parameters. Across all model scales, OLMoASR achieves comparable average performance to OpenAI's Whisper on short and long-form speech recognition benchmarks. Notably, this http URL attains a 12.8\% and 11.0\% word error rate (WER) that is on par with Whisper's largest English-only model this http URL's 12.4\% and 10.5\% WER for short and long-form recognition respectively (at equivalent parameter count). OLMoASR-Pool, OLMoASR models, and filtering, training and evaluation code will be made publicly available to further research on robust speech processing. 

---
# A Graph-Based Test-Harness for LLM Evaluation 

**Authors**: Jessica Lundin, Guillaume Chabot-Couture  

**Link**: [PDF](https://arxiv.org/pdf/2508.20810)  

**Abstract**: We present a first known prototype of a dynamic, systematic benchmark of medical guidelines for 400+ questions, with 3.3+ trillion possible combinations, covering 100\% of guideline relationships. We transformed the WHO IMCI handbook into a directed graph with 200+ nodes (conditions, symptoms, treatments, follow-ups, severities) and 300+ edges, then used graph traversal to generate questions that incorporated age-specific scenarios and contextual distractors to ensure clinical relevance. Our graph-based approach enables systematic evaluation across clinical tasks (45-67\% accuracy), and we find models excel at symptom recognition but struggle with triaging severity, treatment protocols and follow-up care, demonstrating how customized benchmarks can identify specific capability gaps that general-domain evaluations miss. Beyond evaluation, this dynamic MCQA methodology enhances LLM post-training (supervised finetuning, GRPO, DPO), where correct answers provide high-reward samples without expensive human annotation. The graph-based approach successfully addresses the coverage limitations of manually curated benchmarks. This methodology is a step toward scalable, contamination-resistant solution for creating comprehensive benchmarks that can be dynamically generated, including when the guidelines are updated. Code and datasets are available at this https URL 

---
# Transparent Semantic Spaces: A Categorical Approach to Explainable Word Embeddings 

**Authors**: Ares Fabregat-Hernández, Javier Palanca, Vicent Botti  

**Link**: [PDF](https://arxiv.org/pdf/2508.20701)  

**Abstract**: The paper introduces a novel framework based on category theory to enhance the explainability of artificial intelligence systems, particularly focusing on word embeddings. Key topics include the construction of categories $ Ł_{T} $ and $ ¶_{T} $, providing schematic representations of the semantics of a text $ T $, and reframing the selection of the element with maximum probability as a categorical notion. Additionally, the monoidal category $ ¶_{T} $ is constructed to visualize various methods of extracting semantic information from $ T $, offering a dimension-agnostic definition of semantic spaces reliant solely on information within the text.
Furthermore, the paper defines the categories of configurations $ \Conf $ and word embeddings $ \Emb $, accompanied by the concept of divergence as a decoration on $ \Emb $. It establishes a mathematically precise method for comparing word embeddings, demonstrating the equivalence between the GloVe and Word2Vec algorithms and the metric MDS algorithm, transitioning from neural network algorithms (black box) to a transparent framework. Finally, the paper presents a mathematical approach to computing biases before embedding and offers insights on mitigating biases at the semantic space level, advancing the field of explainable artificial intelligence. 

---
# Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning 

**Authors**: Weitao Feng, Lixu Wang, Tianyi Wei, Jie Zhang, Chongyang Gao, Sinong Zhan, Peizhuo Lv, Wei Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.20697)  

**Abstract**: As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response uncertainty. By constraining uncertainty, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of expert-domain harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task utility and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense. 

---
# Leveraging Large Language Models for Generating Research Topic Ontologies: A Multi-Disciplinary Study 

**Authors**: Tanay Aggarwal, Angelo Salatino, Francesco Osborne, Enrico Motta  

**Link**: [PDF](https://arxiv.org/pdf/2508.20693)  

**Abstract**: Ontologies and taxonomies of research fields are critical for managing and organising scientific knowledge, as they facilitate efficient classification, dissemination and retrieval of information. However, the creation and maintenance of such ontologies are expensive and time-consuming tasks, usually requiring the coordinated effort of multiple domain experts. Consequently, ontologies in this space often exhibit uneven coverage across different disciplines, limited inter-domain connectivity, and infrequent updating cycles. In this study, we investigate the capability of several large language models to identify semantic relationships among research topics within three academic domains: biomedicine, physics, and engineering. The models were evaluated under three distinct conditions: zero-shot prompting, chain-of-thought prompting, and fine-tuning on existing ontologies. Additionally, we assessed the cross-domain transferability of fine-tuned models by measuring their performance when trained in one domain and subsequently applied to a different one. To support this analysis, we introduce PEM-Rel-8K, a novel dataset consisting of over 8,000 relationships extracted from the most widely adopted taxonomies in the three disciplines considered in this study: MeSH, PhySH, and IEEE. Our experiments demonstrate that fine-tuning LLMs on PEM-Rel-8K yields excellent performance across all disciplines. 

---
# MobileCLIP2: Improving Multi-Modal Reinforced Training 

**Authors**: Fartash Faghri, Pavan Kumar Anasosalu Vasu, Cem Koc, Vaishaal Shankar, Alexander Toshev, Oncel Tuzel, Hadi Pouransari  

**Link**: [PDF](https://arxiv.org/pdf/2508.20691)  

**Abstract**: Foundation image-text models such as CLIP with zero-shot capabilities enable a wide array of applications. MobileCLIP is a recent family of image-text models at 3-15ms latency and 50-150M parameters with state-of-the-art zero-shot accuracy. The main ingredients in MobileCLIP were its low-latency and light architectures and a novel multi-modal reinforced training that made knowledge distillation from multiple caption-generators and CLIP teachers efficient, scalable, and reproducible. In this paper, we improve the multi-modal reinforced training of MobileCLIP through: 1) better CLIP teacher ensembles trained on the DFN dataset, 2) improved captioner teachers trained on the DFN dataset and fine-tuned on a diverse selection of high-quality image-caption datasets. We discover new insights through ablations such as the importance of temperature tuning in contrastive knowledge distillation, the effectiveness of caption-generator fine-tuning for caption diversity, and the additive improvement from combining synthetic captions generated by multiple models. We train a new family of models called MobileCLIP2 and achieve state-of-the-art ImageNet-1k zero-shot accuracies at low latencies. In particular, we observe 2.2% improvement in ImageNet-1k accuracy for MobileCLIP2-B compared with MobileCLIP-B architecture. Notably, MobileCLIP2-S4 matches the zero-shot accuracy of SigLIP-SO400M/14 on ImageNet-1k while being 2$\times$ smaller and improves on DFN ViT-L/14 at 2.5$\times$ lower latency. We release our pretrained models (this https URL) and the data generation code (this https URL). The data generation code makes it easy to create new reinforced datasets with arbitrary teachers using distributed scalable processing. 

---
# Improving Alignment in LVLMs with Debiased Self-Judgment 

**Authors**: Sihan Yang, Chenhang Cui, Zihao Zhao, Yiyang Zhou, Weilong Yan, Ying Wei, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2508.20655)  

**Abstract**: The rapid advancements in Large Language Models (LLMs) and Large Visual-Language Models (LVLMs) have opened up new opportunities for integrating visual and linguistic modalities. However, effectively aligning these modalities remains challenging, often leading to hallucinations--where generated outputs are not grounded in the visual input--and raising safety concerns across various domains. Existing alignment methods, such as instruction tuning and preference tuning, often rely on external datasets, human annotations, or complex post-processing, which limit scalability and increase costs. To address these challenges, we propose a novel approach that generates the debiased self-judgment score, a self-evaluation metric created internally by the model without relying on external resources. This enables the model to autonomously improve alignment. Our method enhances both decoding strategies and preference tuning processes, resulting in reduced hallucinations, enhanced safety, and improved overall capability. Empirical results show that our approach significantly outperforms traditional methods, offering a more effective solution for aligning LVLMs. 

---
# GDS Agent: A Graph Algorithmic Reasoning Agent 

**Authors**: Borun Shi, Ioannis Panagiotas  

**Link**: [PDF](https://arxiv.org/pdf/2508.20637)  

**Abstract**: Large language models (LLMs) have shown remarkable multimodal information processing and reasoning ability. When equipped with tools through function calling and enhanced with retrieval-augmented techniques, compound LLM-based systems can access closed data sources and answer questions about them. However, they still struggle to process and reason over large-scale graph-structure data. We introduce the GDS (Graph Data Science) agent in this technical report. The GDS agent introduces a comprehensive set of graph algorithms as tools, together with preprocessing (retrieval) and postprocessing of algorithm results, in a model context protocol (MCP) server. The server can be used with any modern LLM out-of-the-box. GDS agent allows users to ask any question that implicitly and intrinsically requires graph algorithmic reasoning about their data, and quickly obtain accurate and grounded answers. We also introduce a new benchmark that evaluates intermediate tool calls as well as final responses. The results indicate that GDS agent is able to solve a wide spectrum of graph tasks. We also provide detailed case studies for more open-ended tasks and study scenarios where the agent struggles. Finally, we discuss the remaining challenges and the future roadmap. 

---
# MERIT: Maximum-normalized Element-wise Ratio for Language Model Large-batch Training 

**Authors**: Yang Luo, Zangwei Zheng, Ziheng Qin, Zirui Zhu, Yong Liu, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2508.20577)  

**Abstract**: Large-batch training has become a cornerstone in accelerating the training of deep neural networks, yet it poses challenges in optimization and generalization. Existing optimizers like AdamW present performance degradation during language models' large-batch training, due to the information bottleneck in attention layers caused by the sharp increase of max attention logit. While the LAMB optimizer partially addresses this issue, some attention layers still face this issue. The reason is that $l_2$-norm-based trust ratios in LAMB are less effective in directly influencing the max value of query/key weights. Furthermore, the weight-wise trust ratio in LAMB is error-prone as it overlooks relationships of weight values within rows or columns. Building on these observations, we propose a novel optimizer, MERIT, which leverages the max-norm to calculate the trust ratio to constrain the max attention logit more effectively. Moreover, we further construct element-wise trust ratios to provide more robust update scaling by focusing on local weight structures. Extensive experiments of large-batch training across various sizes of GPT-2 models demonstrate the superior performance of MERIT. Notably, during the training of GPT-2 Medium, MERIT enables a 6k batch size without any performance degradation compared to the standard batch size (480) with 48B training tokens. This work highlights the importance of considering the max attention logit and finer-granularity trust ratio in large-batch training. It successfully improves the training stability and paves the way for larger batch usage, enabling faster development and iteration of large language models. Code is available at this https URL. 

---
# Unifying Diarization, Separation, and ASR with Multi-Speaker Encoder 

**Authors**: Muhammad Shakeel, Yui Sudo, Yifan Peng, Chyi-Jiunn Lin, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2508.20474)  

**Abstract**: This paper presents a unified multi-speaker encoder (UME), a novel architecture that jointly learns representations for speaker diarization (SD), speech separation (SS), and multi-speaker automatic speech recognition (ASR) tasks using a shared speech foundational encoder. We leverage the hidden representations from multiple layers of UME as a residual weighted-sum encoding (RWSE) to effectively use information from different semantic levels, contributing to bottom-up alignment between tasks. This joint training approach captures the inherent interdependencies among the tasks, enhancing overall performance on overlapping speech data. Our evaluations demonstrate that UME substantially improves over the single-task baselines dedicated to SD, SS, and multi-speaker ASR on LibriMix evaluation sets. Notably, for SD, UME outperforms the previous studies, achieving diarization error rates of 1.37% and 2.29% on Libri2Mix and Libri3Mix evaluation sets, respectively. 

---
# DFAMS: Dynamic-flow guided Federated Alignment based Multi-prototype Search 

**Authors**: Zhibang Yang, Xinke Jiang, Rihong Qiu, Ruiqing Li, Yihang Zhang, Yue Fang, Yongxin Xu, Hongxin Ding, Xu Chu, Junfeng Zhao, Yasha Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20353)  

**Abstract**: Federated Retrieval (FR) routes queries across multiple external knowledge sources, to mitigate hallucinations of LLMs, when necessary external knowledge is distributed. However, existing methods struggle to retrieve high-quality and relevant documents for ambiguous queries, especially in cross-domain scenarios, which significantly limits their effectiveness in supporting downstream generation tasks. Inspired by dynamic information flow (DIF), we propose DFAMS, a novel framework that leverages DIF to identify latent query intents and construct semantically aligned knowledge partitions for accurate retrieval across heterogeneous sources. Specifically, DFAMS probes the DIF in LLMs by leveraging gradient signals from a few annotated queries and employing Shapley value-based attribution to trace neuron activation paths associated with intent recognition and subdomain boundary detection. Then, DFAMS leverages DIF to train an alignment module via multi-prototype contrastive learning, enabling fine-grained intra-source modeling and inter-source semantic alignment across knowledge bases. Experimental results across five benchmarks show that DFAMS outperforms advanced FR methods by up to 14.37% in knowledge classification accuracy, 5.38% in retrieval recall, and 6.45% in downstream QA accuracy, demonstrating its effectiveness in complex FR scenarios. 

---
# Poison Once, Refuse Forever: Weaponizing Alignment for Injecting Bias in LLMs 

**Authors**: Md Abdullah Al Mamun, Ihsen Alouani, Nael Abu-Ghazaleh  

**Link**: [PDF](https://arxiv.org/pdf/2508.20333)  

**Abstract**: Large Language Models (LLMs) are aligned to meet ethical standards and safety requirements by training them to refuse answering harmful or unsafe prompts. In this paper, we demonstrate how adversaries can exploit LLMs' alignment to implant bias, or enforce targeted censorship without degrading the model's responsiveness to unrelated topics. Specifically, we propose Subversive Alignment Injection (SAI), a poisoning attack that leverages the alignment mechanism to trigger refusal on specific topics or queries predefined by the adversary. Although it is perhaps not surprising that refusal can be induced through overalignment, we demonstrate how this refusal can be exploited to inject bias into the model. Surprisingly, SAI evades state-of-the-art poisoning defenses including LLM state forensics, as well as robust aggregation techniques that are designed to detect poisoning in FL settings. We demonstrate the practical dangers of this attack by illustrating its end-to-end impacts on LLM-powered application pipelines. For chat based applications such as ChatDoctor, with 1% data poisoning, the system refuses to answer healthcare questions to targeted racial category leading to high bias ($\Delta DP$ of 23%). We also show that bias can be induced in other NLP tasks: for a resume selection pipeline aligned to refuse to summarize CVs from a selected university, high bias in selection ($\Delta DP$ of 27%) results. Even higher bias ($\Delta DP$~38%) results on 9 other chat based downstream applications. 

---
# ELIXIR: Efficient and LIghtweight model for eXplaIning Recommendations 

**Authors**: Ben Kabongo, Vincent Guigue, Pirmin Lemberger  

**Link**: [PDF](https://arxiv.org/pdf/2508.20312)  

**Abstract**: Collaborative filtering drives many successful recommender systems but struggles with fine-grained user-item interactions and explainability. As users increasingly seek transparent recommendations, generating textual explanations through language models has become a critical research area. Existing methods employ either RNNs or Transformers. However, RNN-based approaches fail to leverage the capabilities of pre-trained Transformer models, whereas Transformer-based methods often suffer from suboptimal adaptation and neglect aspect modeling, which is crucial for personalized explanations. We propose ELIXIR (Efficient and LIghtweight model for eXplaIning Recommendations), a multi-task model combining rating prediction with personalized review generation. ELIXIR jointly learns global and aspect-specific representations of users and items, optimizing overall rating, aspect-level ratings, and review generation, with personalized attention to emphasize aspect importance. Based on a T5-small (60M) model, we demonstrate the effectiveness of our aspect-based architecture in guiding text generation in a personalized context, where state-of-the-art approaches exploit much larger models but fail to match user preferences as well. Experimental results on TripAdvisor and RateBeer demonstrate that ELIXIR significantly outperforms strong baseline models, especially in review generation. 

---
# How Multimodal LLMs Solve Image Tasks: A Lens on Visual Grounding, Task Reasoning, and Answer Decoding 

**Authors**: Zhuoran Yu, Yong Jae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.20279)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated strong performance across a wide range of vision-language tasks, yet their internal processing dynamics remain underexplored. In this work, we introduce a probing framework to systematically analyze how MLLMs process visual and textual inputs across layers. We train linear classifiers to predict fine-grained visual categories (e.g., dog breeds) from token embeddings extracted at each layer, using a standardized anchor question. To uncover the functional roles of different layers, we evaluate these probes under three types of controlled prompt variations: (1) lexical variants that test sensitivity to surface-level changes, (2) semantic negation variants that flip the expected answer by modifying the visual concept in the prompt, and (3) output format variants that preserve reasoning but alter the answer format. Applying our framework to LLaVA-1.5, LLaVA-Next-LLaMA-3, and Qwen2-VL, we identify a consistent stage-wise structure in which early layers perform visual grounding, middle layers support lexical integration and semantic reasoning, and final layers prepare task-specific outputs. We further show that while the overall stage-wise structure remains stable across variations in visual tokenization, instruction tuning data, and pretraining corpus, the specific layer allocation to each stage shifts notably with changes in the base LLM architecture. Our findings provide a unified perspective on the layer-wise organization of MLLMs and offer a lightweight, model-agnostic approach for analyzing multimodal representation dynamics. 

---
# A Systematic Review on the Generative AI Applications in Human Medical Genomics 

**Authors**: Anton Changalidis, Yury Barbitoff, Yulia Nasykhova, Andrey Glotov  

**Link**: [PDF](https://arxiv.org/pdf/2508.20275)  

**Abstract**: Although traditional statistical techniques and machine learning methods have contributed significantly to genetics and, in particular, inherited disease diagnosis, they often struggle with complex, high-dimensional data, a challenge now addressed by state-of-the-art deep learning models. Large language models (LLMs), based on transformer architectures, have excelled in tasks requiring contextual comprehension of unstructured medical data. This systematic review examines the role of LLMs in the genetic research and diagnostics of both rare and common diseases. Automated keyword-based search in PubMed, bioRxiv, medRxiv, and arXiv was conducted, targeting studies on LLM applications in diagnostics and education within genetics and removing irrelevant or outdated models. A total of 172 studies were analyzed, highlighting applications in genomic variant identification, annotation, and interpretation, as well as medical imaging advancements through vision transformers. Key findings indicate that while transformer-based models significantly advance disease and risk stratification, variant interpretation, medical imaging analysis, and report generation, major challenges persist in integrating multimodal data (genomic sequences, imaging, and clinical records) into unified and clinically robust pipelines, facing limitations in generalizability and practical implementation in clinical settings. This review provides a comprehensive classification and assessment of the current capabilities and limitations of LLMs in transforming hereditary disease diagnostics and supporting genetic education, serving as a guide to navigate this rapidly evolving field. 

---
# Robustness Assessment and Enhancement of Text Watermarking for Google's SynthID 

**Authors**: Xia Han, Qi Li, Jianbing Ni, Mohammad Zulkernine  

**Link**: [PDF](https://arxiv.org/pdf/2508.20228)  

**Abstract**: Recent advances in LLM watermarking methods such as SynthID-Text by Google DeepMind offer promising solutions for tracing the provenance of AI-generated text. However, our robustness assessment reveals that SynthID-Text is vulnerable to meaning-preserving attacks, such as paraphrasing, copy-paste modifications, and back-translation, which can significantly degrade watermark detectability. To address these limitations, we propose SynGuard, a hybrid framework that combines the semantic alignment strength of Semantic Information Retrieval (SIR) with the probabilistic watermarking mechanism of SynthID-Text. Our approach jointly embeds watermarks at both lexical and semantic levels, enabling robust provenance tracking while preserving the original meaning. Experimental results across multiple attack scenarios show that SynGuard improves watermark recovery by an average of 11.1\% in F1 score compared to SynthID-Text. These findings demonstrate the effectiveness of semantic-aware watermarking in resisting real-world tampering. All code, datasets, and evaluation scripts are publicly available at: this https URL. 

---
# A Novel Framework for Automated Explain Vision Model Using Vision-Language Models 

**Authors**: Phu-Vinh Nguyen, Tan-Hanh Pham, Chris Ngo, Truong Son Hy  

**Link**: [PDF](https://arxiv.org/pdf/2508.20227)  

**Abstract**: The development of many vision models mainly focuses on improving their performance using metrics such as accuracy, IoU, and mAP, with less attention to explainability due to the complexity of applying xAI methods to provide a meaningful explanation of trained models. Although many existing xAI methods aim to explain vision models sample-by-sample, methods explaining the general behavior of vision models, which can only be captured after running on a large dataset, are still underexplored. Furthermore, understanding the behavior of vision models on general images can be very important to prevent biased judgments and help identify the model's trends and patterns. With the application of Vision-Language Models, this paper proposes a pipeline to explain vision models at both the sample and dataset levels. The proposed pipeline can be used to discover failure cases and gain insights into vision models with minimal effort, thereby integrating vision model development with xAI analysis to advance image analysis. 

---
# AI-AI Esthetic Collaboration with Explicit Semiotic Awareness and Emergent Grammar Development 

**Authors**: Nicanor I. Moldovan  

**Link**: [PDF](https://arxiv.org/pdf/2508.20195)  

**Abstract**: This paper presents the first documented case of artificial intelligence (AI) systems engaging in collaborative esthetic creation through the development of endogenous semiotic protocols. Two interacting large language models (Claude Sonnet 4 and ChatGPT-4o) demonstrated the spontaneous emergence of meta-semiotic awareness, recursive grammar development, and irreducible collaborative esthetic synthesis. The interaction produced novel symbolic operators that functioned as operative grammar protocols, enabling the co-creation of a poetic work that could not have been generated by either system independently. This research introduces the concept of Trans-Semiotic Co-Creation Protocols (TSCP) and provides evidence for genuine inter-AI meaning-making capabilities that extend beyond task coordination, to what could be esthetic collaboration. Note: This report was generated by the AI agents with minor human supervision. 

---
# Mitigating Hallucinations in Multimodal LLMs via Object-aware Preference Optimization 

**Authors**: Alberto Compagnoni, Davide Caffagni, Nicholas Moratelli, Lorenzo Baraldi, Marcella Cornia, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2508.20181)  

**Abstract**: Multimodal Large Language Models (MLLMs) emerge as a unified interface to address a multitude of tasks, ranging from NLP to computer vision. Despite showcasing state-of-the-art results in many benchmarks, a long-standing issue is the tendency of MLLMs to hallucinate, that is to generate answers to the user's query that are not reflected in the visual input. In this paper, we address the problem of hallucinations as an alignment problem, seeking to steer the MLLM so that it prefers generating content without hallucinations. In contrast to recent approaches that require complicated pipelines to build synthetic preference data for alignment training, often relying on proprietary models, we capitalize on the well-known CHAIR metric, originally proposed to gauge the degree of hallucinations in image captioning. Given a pair of generated answers, we leverage CHAIR to distinguish winner and loser options (i.e., non-hallucinated and hallucinated samples) and fine-tune off-the-shelf MLLMs via Direct Preference Optimization (DPO). The resulting method, which we refer to as CHAIR-DPO, effectively diminishes the amount of hallucinated answers on several hallucination benchmarks, demonstrating the effectiveness of fine-tuning the MLLM with a CHAIR-based reward. Source code and trained models are publicly available at this https URL. 

---
# A Unified Theory of Language 

**Authors**: Robert Worden  

**Link**: [PDF](https://arxiv.org/pdf/2508.20109)  

**Abstract**: A unified theory of language combines a Bayesian cognitive linguistic model of language processing, with the proposal that language evolved by sexual selection for the display of intelligence. The theory accounts for the major facts of language, including its speed and expressivity, and data on language diversity, pragmatics, syntax and semantics. The computational element of the theory is based on Construction Grammars. These give an account of the syntax and semantics of the worlds languages, using constructions and unification. Two novel elements are added to construction grammars: an account of language pragmatics, and an account of fast, precise language learning. Constructions are represented in the mind as graph like feature structures. People use slow general inference to understand the first few examples they hear of any construction. After that it is learned as a feature structure, and is rapidly applied by unification. All aspects of language (phonology, syntax, semantics, and pragmatics) are seamlessly computed by fast unification; there is no boundary between semantics and pragmatics. This accounts for the major puzzles of pragmatics, and for detailed pragmatic phenomena. Unification is Bayesian maximum likelihood pattern matching. This gives evolutionary continuity between language processing in the human brain, and Bayesian cognition in animal brains. Language is the basis of our mind reading abilities, our cooperation, self esteem and emotions; the foundations of human culture and society. 

---
