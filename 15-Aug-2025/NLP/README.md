# A Survey on Diffusion Language Models 

**Authors**: Tianyi Li, Mingda Chen, Bowei Guo, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10875)  

**Abstract**: Diffusion Language Models (DLMs) are rapidly emerging as a powerful and promising alternative to the dominant autoregressive (AR) paradigm. By generating tokens in parallel through an iterative denoising process, DLMs possess inherent advantages in reducing inference latency and capturing bidirectional context, thereby enabling fine-grained control over the generation process. While achieving a several-fold speed-up, recent advancements have allowed DLMs to show performance comparable to their autoregressive counterparts, making them a compelling choice for various natural language processing tasks. In this survey, we provide a holistic overview of the current DLM landscape. We trace its evolution and relationship with other paradigms, such as autoregressive and masked language models, and cover both foundational principles and state-of-the-art models. Our work offers an up-to-date, comprehensive taxonomy and an in-depth analysis of current techniques, from pre-training strategies to advanced post-training methods. Another contribution of this survey is a thorough review of DLM inference strategies and optimizations, including improvements in decoding parallelism, caching mechanisms, and generation quality. We also highlight the latest approaches to multimodal extensions of DLMs and delineate their applications across various practical scenarios. Furthermore, our discussion addresses the limitations and challenges of DLMs, including efficiency, long-sequence handling, and infrastructure requirements, while outlining future research directions to sustain progress in this rapidly evolving field. Project GitHub is available at this https URL. 

---
# SSRL: Self-Search Reinforcement Learning 

**Authors**: Yuchen Fan, Kaiyan Zhang, Heng Zhou, Yuxin Zuo, Yanxu Chen, Yu Fu, Xinwei Long, Xuekai Zhu, Che Jiang, Yuchen Zhang, Li Kang, Gang Chen, Cheng Huang, Zhizhou He, Bingning Wang, Lei Bai, Ning Ding, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.10874)  

**Abstract**: We investigate the potential of large language models (LLMs) to serve as efficient simulators for agentic search tasks in reinforcement learning (RL), thereby reducing dependence on costly interactions with external search engines. To this end, we first quantify the intrinsic search capability of LLMs via structured prompting and repeated sampling, which we term Self-Search. Our results reveal that LLMs exhibit strong scaling behavior with respect to the inference budget, achieving high pass@k on question-answering benchmarks, including the challenging BrowseComp task. Building on these observations, we introduce Self-Search RL (SSRL), which enhances LLMs' Self-Search capability through format-based and rule-based rewards. SSRL enables models to iteratively refine their knowledge utilization internally, without requiring access to external tools. Empirical evaluations demonstrate that SSRL-trained policy models provide a cost-effective and stable environment for search-driven RL training, reducing reliance on external search engines and facilitating robust sim-to-real transfer. We draw the following conclusions: 1) LLMs possess world knowledge that can be effectively elicited to achieve high performance; 2) SSRL demonstrates the potential of leveraging internal knowledge to reduce hallucination; 3) SSRL-trained models integrate seamlessly with external search engines without additional effort. Our findings highlight the potential of LLMs to support more scalable RL agent training. 

---
# From Black Box to Transparency: Enhancing Automated Interpreting Assessment with Explainable AI in College Classrooms 

**Authors**: Zhaokun Jiang, Ziyin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10860)  

**Abstract**: Recent advancements in machine learning have spurred growing interests in automated interpreting quality assessment. Nevertheless, existing research suffers from insufficient examination of language use quality, unsatisfactory modeling effectiveness due to data scarcity and imbalance, and a lack of efforts to explain model predictions. To address these gaps, we propose a multi-dimensional modeling framework that integrates feature engineering, data augmentation, and explainable machine learning. This approach prioritizes explainability over ``black box'' predictions by utilizing only construct-relevant, transparent features and conducting Shapley Value (SHAP) analysis. Our results demonstrate strong predictive performance on a novel English-Chinese consecutive interpreting dataset, identifying BLEURT and CometKiwi scores to be the strongest predictive features for fidelity, pause-related features for fluency, and Chinese-specific phraseological diversity metrics for language use. Overall, by placing particular emphasis on explainability, we present a scalable, reliable, and transparent alternative to traditional human evaluation, facilitating the provision of detailed diagnostic feedback for learners and supporting self-regulated learning advantages not afforded by automated scores in isolation. 

---
# Psyche-R1: Towards Reliable Psychological LLMs through Unified Empathy, Expertise, and Reasoning 

**Authors**: Chongyuan Dai, Jinpeng Hu, Hongchang Shi, Zhuo Li, Xun Yang, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10848)  

**Abstract**: Amidst a shortage of qualified mental health professionals, the integration of large language models (LLMs) into psychological applications offers a promising way to alleviate the growing burden of mental health disorders. Recent reasoning-augmented LLMs have achieved remarkable performance in mathematics and programming, while research in the psychological domain has predominantly emphasized emotional support and empathetic dialogue, with limited attention to reasoning mechanisms that are beneficial to generating reliable responses. Therefore, in this paper, we propose Psyche-R1, the first Chinese psychological LLM that jointly integrates empathy, psychological expertise, and reasoning, built upon a novel data curation pipeline. Specifically, we design a comprehensive data synthesis pipeline that produces over 75k high-quality psychological questions paired with detailed rationales, generated through chain-of-thought (CoT) reasoning and iterative prompt-rationale optimization, along with 73k empathetic dialogues. Subsequently, we employ a hybrid training strategy wherein challenging samples are identified through a multi-LLM cross-selection strategy for group relative policy optimization (GRPO) to improve reasoning ability, while the remaining data is used for supervised fine-tuning (SFT) to enhance empathetic response generation and psychological domain knowledge. Extensive experiment results demonstrate the effectiveness of the Psyche-R1 across several psychological benchmarks, where our 7B Psyche-R1 achieves comparable results to 671B DeepSeek-R1. 

---
# Reinforced Language Models for Sequential Decision Making 

**Authors**: Jim Dilkes, Vahid Yazdanpanah, Sebastian Stein  

**Link**: [PDF](https://arxiv.org/pdf/2508.10839)  

**Abstract**: Large Language Models (LLMs) show potential as sequential decision-making agents, but their application is often limited due to a reliance on large, computationally expensive models. This creates a need to improve smaller models, yet existing post-training methods are designed for single-turn interactions and cannot handle credit assignment in multi-step agentic tasks. To address this, we introduce Multi-Step Group-Relative Policy Optimization (MS-GRPO), a new algorithm for post-training LLM agents, grounded in formal Text-Mediated Stochastic Game (TSMG) and Language-Agent Policy (LAP) frameworks. For credit assignment, MS-GRPO attributes the entire cumulative episode reward to each individual episode step. We supplement this algorithm with a novel absolute-advantage-weighted episode sampling strategy that we show improves training performance. We evaluate our approach by post-training a 3-billion parameter model on Snake and Frozen Lake. Our experiments demonstrate that the method is effective in improving decision-making performance: our post-trained 3B parameter model outperforms a 72B parameter baseline by 50% on the Frozen Lake task. This work demonstrates that targeted post-training is a practical and efficient alternative to relying on model scale for creating sequential decision-making agents using LLMs. 

---
# Beyond "Not Novel Enough": Enriching Scholarly Critique with LLM-Assisted Feedback 

**Authors**: Osama Mohammed Afzal, Preslav Nakov, Tom Hope, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2508.10795)  

**Abstract**: Novelty assessment is a central yet understudied aspect of peer review, particularly in high volume fields like NLP where reviewer capacity is increasingly strained. We present a structured approach for automated novelty evaluation that models expert reviewer behavior through three stages: content extraction from submissions, retrieval and synthesis of related work, and structured comparison for evidence based assessment. Our method is informed by a large scale analysis of human written novelty reviews and captures key patterns such as independent claim verification and contextual reasoning. Evaluated on 182 ICLR 2025 submissions with human annotated reviewer novelty assessments, the approach achieves 86.5% alignment with human reasoning and 75.3% agreement on novelty conclusions - substantially outperforming existing LLM based baselines. The method produces detailed, literature aware analyses and improves consistency over ad hoc reviewer judgments. These results highlight the potential for structured LLM assisted approaches to support more rigorous and transparent peer review without displacing human expertise. Data and code are made available. 

---
# Thinking Inside the Mask: In-Place Prompting in Diffusion LLMs 

**Authors**: Xiangqi Jin, Yuxuan Wang, Yifeng Gao, Zichen Wen, Biqing Qi, Dongrui Liu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10736)  

**Abstract**: Despite large language models (LLMs) have achieved remarkable success, their prefix-only prompting paradigm and sequential generation process offer limited flexibility for bidirectional information. Diffusion large language models (dLLMs) present new opportunities through their bidirectional attention mechanisms and iterative refinement processes, enabling more flexible in-place prompting strategies. We introduce ICE (In-Place Chain-of-Thought Prompting with Early Exit), a novel framework that transforms prefix-only prompting into in-place prompting specifically designed for dLLMs. ICE integrates in-place prompts directly within masked token positions during iterative refinement and employs a confidence-aware early exit mechanism to significantly reduce computational overhead. Extensive experiments demonstrate ICE's effectiveness, achieving up to 17.29% accuracy improvement with 4.12$\times$ speedup on GSM8K, and up to 276.67$\times$ acceleration on MMLU while maintaining competitive performance. 

---
# Learning from Natural Language Feedback for Personalized Question Answering 

**Authors**: Alireza Salemi, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2508.10695)  

**Abstract**: Personalization is crucial for enhancing both the effectiveness and user satisfaction of language technologies, particularly in information-seeking tasks like question answering. Current approaches for personalizing large language models (LLMs) often rely on retrieval-augmented generation (RAG), followed by reinforcement learning with scalar reward signals to teach models how to use retrieved personal context. We believe that these scalar rewards sometimes provide weak, non-instructive feedback, limiting learning efficiency and personalization quality. We introduce VAC, a novel framework for personalized response generation that replaces scalar rewards with natural language feedback (NLF) that are generated conditioned on the user profiles and the question narratives. NLF serves as a rich and actionable supervision signal, allowing the policy model to iteratively refine its outputs and internalize effective personalization strategies. Training alternates between optimizing the feedback model and fine-tuning the policy model on the improved responses, resulting in a policy model that no longer requires feedback at inference. Evaluation on the LaMP-QA benchmark that consists of three diverse domains demonstrates consistent and significant improvements over the state-of-the-art results. Human evaluations further confirm the superior quality of the generated responses. These results demonstrate that NLF provides more effective signals for optimizing personalized question answering. 

---
# Continuous Bangla Sign Language Translation: Mitigating the Expense of Gloss Annotation with the Assistance of Graph 

**Authors**: Safaeid Hossain Arib, Rabeya Akter, Sejuti Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2508.10687)  

**Abstract**: Millions of individuals worldwide are affected by deafness and hearing impairment. Sign language serves as a sophisticated means of communication for the deaf and hard of hearing. However, in societies that prioritize spoken languages, sign language often faces underestimation, leading to communication barriers and social exclusion. The Continuous Bangla Sign Language Translation project aims to address this gap by enhancing translation methods. While recent approaches leverage transformer architecture for state-of-the-art results, our method integrates graph-based methods with the transformer architecture. This fusion, combining transformer and STGCN-LSTM architectures, proves more effective in gloss-free translation. Our contributions include architectural fusion, exploring various fusion strategies, and achieving a new state-of-the-art performance on diverse sign language datasets, namely RWTH-PHOENIX-2014T, CSL-Daily, How2Sign, and BornilDB v1.0. Our approach demonstrates superior performance compared to current translation outcomes across all datasets, showcasing notable improvements of BLEU-4 scores of 4.01, 2.07, and 0.5, surpassing those of GASLT, GASLT and slt_how2sign in RWTH-PHOENIX-2014T, CSL-Daily, and How2Sign, respectively. Also, we introduce benchmarking on the BornilDB v1.0 dataset for the first time. Our method sets a benchmark for future research, emphasizing the importance of gloss-free translation to improve communication accessibility for the deaf and hard of hearing. 

---
# Neural Machine Translation for Coptic-French: Strategies for Low-Resource Ancient Languages 

**Authors**: Nasma Chaoui, Richard Khoury  

**Link**: [PDF](https://arxiv.org/pdf/2508.10683)  

**Abstract**: This paper presents the first systematic study of strategies for translating Coptic into French. Our comprehensive pipeline systematically evaluates: pivot versus direct translation, the impact of pre-training, the benefits of multi-version fine-tuning, and model robustness to noise. Utilizing aligned biblical corpora, we demonstrate that fine-tuning with a stylistically-varied and noise-aware training corpus significantly enhances translation quality. Our findings provide crucial practical insights for developing translation tools for historical languages in general. 

---
# eDIF: A European Deep Inference Fabric for Remote Interpretability of LLM 

**Authors**: Irma Heithoff. Marc Guggenberger, Sandra Kalogiannis, Susanne Mayer, Fabian Maag, Sigurd Schacht, Carsten Lanquillon  

**Link**: [PDF](https://arxiv.org/pdf/2508.10553)  

**Abstract**: This paper presents a feasibility study on the deployment of a European Deep Inference Fabric (eDIF), an NDIF-compatible infrastructure designed to support mechanistic interpretability research on large language models. The need for widespread accessibility of LLM interpretability infrastructure in Europe drives this initiative to democratize advanced model analysis capabilities for the research community. The project introduces a GPU-based cluster hosted at Ansbach University of Applied Sciences and interconnected with partner institutions, enabling remote model inspection via the NNsight API. A structured pilot study involving 16 researchers from across Europe evaluated the platform's technical performance, usability, and scientific utility. Users conducted interventions such as activation patching, causal tracing, and representation analysis on models including GPT-2 and DeepSeek-R1-70B. The study revealed a gradual increase in user engagement, stable platform performance throughout, and a positive reception of the remote experimentation capabilities. It also marked the starting point for building a user community around the platform. Identified limitations such as prolonged download durations for activation data as well as intermittent execution interruptions are addressed in the roadmap for future development. This initiative marks a significant step towards widespread accessibility of LLM interpretability infrastructure in Europe and lays the groundwork for broader deployment, expanded tooling, and sustained community collaboration in mechanistic interpretability research. 

---
# When Language Overrules: Revealing Text Dominance in Multimodal Large Language Models 

**Authors**: Huyu Wu, Meng Tang, Xinhan Zheng, Haiyun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10552)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities across a diverse range of multimodal tasks. However, these models suffer from a core problem known as text dominance: they depend heavily on text for their inference, while underutilizing other modalities. While prior work has acknowledged this phenomenon in vision-language tasks, often attributing it to data biases or model architectures. In this paper, we conduct the first systematic investigation of text dominance across diverse data modalities, including images, videos, audio, time-series, and graphs. To measure this imbalance, we propose two evaluation metrics: the Modality Dominance Index (MDI) and the Attention Efficiency Index (AEI). Our comprehensive analysis reveals that text dominance is both significant and pervasive across all tested modalities. Our in-depth analysis identifies three underlying causes: attention dilution from severe token redundancy in non-textual modalities, the influence of fusion architecture design, and task formulations that implicitly favor textual inputs. Furthermore, we propose a simple token compression method that effectively rebalances model attention. Applying this method to LLaVA-7B, for instance, drastically reduces its MDI from 10.23 to a well-balanced value of 0.86. Our analysis and methodological framework offer a foundation for the development of more equitable and comprehensive multimodal language models. 

---
# When Explainability Meets Privacy: An Investigation at the Intersection of Post-hoc Explainability and Differential Privacy in the Context of Natural Language Processing 

**Authors**: Mahdi Dhaini, Stephen Meisenbacher, Ege Erdogan, Florian Matthes, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2508.10482)  

**Abstract**: In the study of trustworthy Natural Language Processing (NLP), a number of important research fields have emerged, including that of \textit{explainability} and \textit{privacy}. While research interest in both explainable and privacy-preserving NLP has increased considerably in recent years, there remains a lack of investigation at the intersection of the two. This leaves a considerable gap in understanding of whether achieving \textit{both} explainability and privacy is possible, or whether the two are at odds with each other. In this work, we conduct an empirical investigation into the privacy-explainability trade-off in the context of NLP, guided by the popular overarching methods of \textit{Differential Privacy} (DP) and Post-hoc Explainability. Our findings include a view into the intricate relationship between privacy and explainability, which is formed by a number of factors, including the nature of the downstream task and choice of the text privatization and explainability method. In this, we highlight the potential for privacy and explainability to co-exist, and we summarize our findings in a collection of practical recommendations for future work at this important intersection. 

---
# DiFaR: Enhancing Multimodal Misinformation Detection with Diverse, Factual, and Relevant Rationales 

**Authors**: Herun Wan, Jiaying Wu, Minnan Luo, Xiangzheng Kong, Zihan Ma, Zhi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2508.10444)  

**Abstract**: Generating textual rationales from large vision-language models (LVLMs) to support trainable multimodal misinformation detectors has emerged as a promising paradigm. However, its effectiveness is fundamentally limited by three core challenges: (i) insufficient diversity in generated rationales, (ii) factual inaccuracies due to hallucinations, and (iii) irrelevant or conflicting content that introduces noise. We introduce DiFaR, a detector-agnostic framework that produces diverse, factual, and relevant rationales to enhance misinformation detection. DiFaR employs five chain-of-thought prompts to elicit varied reasoning traces from LVLMs and incorporates a lightweight post-hoc filtering module to select rationale sentences based on sentence-level factuality and relevance scores. Extensive experiments on four popular benchmarks demonstrate that DiFaR outperforms four baseline categories by up to 5.9% and boosts existing detectors by as much as 8.7%. Both automatic metrics and human evaluations confirm that DiFaR significantly improves rationale quality across all three dimensions. 

---
# Computational Economics in Large Language Models: Exploring Model Behavior and Incentive Design under Resource Constraints 

**Authors**: Sandeep Reddy, Kabir Khan, Rohit Patil, Ananya Chakraborty, Faizan A. Khan, Swati Kulkarni, Arjun Verma, Neha Singh  

**Link**: [PDF](https://arxiv.org/pdf/2508.10426)  

**Abstract**: Large language models (LLMs) are limited by substantial computational cost. We introduce a "computational economics" framework that treats an LLM as an internal economy of resource-constrained agents (attention heads and neuron blocks) that must allocate scarce computation to maximize task utility. First, we show empirically that when computation is scarce, standard LLMs reallocate attention toward high-value tokens while preserving accuracy. Building on this observation, we propose an incentive-driven training paradigm that augments the task loss with a differentiable computation cost term, encouraging sparse and efficient activations. On GLUE (MNLI, STS-B, CoLA) and WikiText-103, the method yields a family of models that trace a Pareto frontier and consistently dominate post-hoc pruning; for a similar accuracy we obtain roughly a forty percent reduction in FLOPS and lower latency, together with more interpretable attention patterns. These results indicate that economic principles offer a principled route to designing efficient, adaptive, and more transparent LLMs under strict resource constraints. 

---
# Evaluating LLMs on Chinese Idiom Translation 

**Authors**: Cai Yang, Yao Dou, David Heineman, Xiaofeng Wu, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10421)  

**Abstract**: Idioms, whose figurative meanings usually differ from their literal interpretations, are common in everyday language, especially in Chinese, where they often contain historical references and follow specific structural patterns. Despite recent progress in machine translation with large language models, little is known about Chinese idiom translation. In this work, we introduce IdiomEval, a framework with a comprehensive error taxonomy for Chinese idiom translation. We annotate 900 translation pairs from nine modern systems, including GPT-4o and Google Translate, across four domains: web, news, Wikipedia, and social media. We find these systems fail at idiom translation, producing incorrect, literal, partial, or even missing translations. The best-performing system, GPT-4, makes errors in 28% of cases. We also find that existing evaluation metrics measure idiom quality poorly with Pearson correlation below 0.48 with human ratings. We thus develop improved models that achieve F$_1$ scores of 0.68 for detecting idiom translation errors. 

---
# ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning 

**Authors**: Juyuan Wang, Rongchen Zhao, Wei Wei, Yufeng Wang, Mo Yu, Jie Zhou, Jin Xu, Liyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10419)  

**Abstract**: Narrative comprehension on long stories and novels has been a challenging domain attributed to their intricate plotlines and entangled, often evolving relations among characters and entities. Given the LLM's diminished reasoning over extended context and high computational cost, retrieval-based approaches remain a pivotal role in practice. However, traditional RAG methods can fall short due to their stateless, single-step retrieval process, which often overlooks the dynamic nature of capturing interconnected relations within long-range context. In this work, we propose ComoRAG, holding the principle that narrative reasoning is not a one-shot process, but a dynamic, evolving interplay between new evidence acquisition and past knowledge consolidation, analogous to human cognition when reasoning with memory-related signals in the brain. Specifically, when encountering a reasoning impasse, ComoRAG undergoes iterative reasoning cycles while interacting with a dynamic memory workspace. In each cycle, it generates probing queries to devise new exploratory paths, then integrates the retrieved evidence of new aspects into a global memory pool, thereby supporting the emergence of a coherent context for the query resolution. Across four challenging long-context narrative benchmarks (200K+ tokens), ComoRAG outperforms strong RAG baselines with consistent relative gains up to 11% compared to the strongest baseline. Further analysis reveals that ComoRAG is particularly advantageous for complex queries requiring global comprehension, offering a principled, cognitively motivated paradigm for retrieval-based long context comprehension towards stateful reasoning. Our code is publicly released at this https URL 

---
# Layer-Wise Perturbations via Sparse Autoencoders for Adversarial Text Generation 

**Authors**: Huizhen Shu, Xuying Li, Qirui Wang, Yuji Kosuga, Mengqiu Tian, Zhuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.10404)  

**Abstract**: With the rapid proliferation of Natural Language Processing (NLP), especially Large Language Models (LLMs), generating adversarial examples to jailbreak LLMs remains a key challenge for understanding model vulnerabilities and improving robustness. In this context, we propose a new black-box attack method that leverages the interpretability of large models. We introduce the Sparse Feature Perturbation Framework (SFPF), a novel approach for adversarial text generation that utilizes sparse autoencoders to identify and manipulate critical features in text. After using the SAE model to reconstruct hidden layer representations, we perform feature clustering on the successfully attacked texts to identify features with higher activations. These highly activated features are then perturbed to generate new adversarial texts. This selective perturbation preserves the malicious intent while amplifying safety signals, thereby increasing their potential to evade existing defenses. Our method enables a new red-teaming strategy that balances adversarial effectiveness with safety alignment. Experimental results demonstrate that adversarial texts generated by SFPF can bypass state-of-the-art defense mechanisms, revealing persistent vulnerabilities in current NLP this http URL, the method's effectiveness varies across prompts and layers, and its generalizability to other architectures and larger models remains to be validated. 

---
# Jailbreaking Commercial Black-Box LLMs with Explicitly Harmful Prompts 

**Authors**: Chiyu Zhang, Lu Zhou, Xiaogang Xu, Jiafei Wu, Liming Fang, Zhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10390)  

**Abstract**: Evaluating jailbreak attacks is challenging when prompts are not overtly harmful or fail to induce harmful outputs. Unfortunately, many existing red-teaming datasets contain such unsuitable prompts. To evaluate attacks accurately, these datasets need to be assessed and cleaned for maliciousness. However, existing malicious content detection methods rely on either manual annotation, which is labor-intensive, or large language models (LLMs), which have inconsistent accuracy in harmful types. To balance accuracy and efficiency, we propose a hybrid evaluation framework named MDH (Malicious content Detection based on LLMs with Human assistance) that combines LLM-based annotation with minimal human oversight, and apply it to dataset cleaning and detection of jailbroken responses. Furthermore, we find that well-crafted developer messages can significantly boost jailbreak success, leading us to propose two new strategies: D-Attack, which leverages context simulation, and DH-CoT, which incorporates hijacked chains of thought. The Codes, datasets, judgements, and detection results will be released in github repository: this https URL. 

---
# Improving Generative Cross-lingual Aspect-Based Sentiment Analysis with Constrained Decoding 

**Authors**: Jakub Šmíd, Pavel Přibáň, Pavel Král  

**Link**: [PDF](https://arxiv.org/pdf/2508.10369)  

**Abstract**: While aspect-based sentiment analysis (ABSA) has made substantial progress, challenges remain for low-resource languages, which are often overlooked in favour of English. Current cross-lingual ABSA approaches focus on limited, less complex tasks and often rely on external translation tools. This paper introduces a novel approach using constrained decoding with sequence-to-sequence models, eliminating the need for unreliable translation tools and improving cross-lingual performance by 5\% on average for the most complex task. The proposed method also supports multi-tasking, which enables solving multiple ABSA tasks with a single model, with constrained decoding boosting results by more than 10\%.
We evaluate our approach across seven languages and six ABSA tasks, surpassing state-of-the-art methods and setting new benchmarks for previously unexplored tasks. Additionally, we assess large language models (LLMs) in zero-shot, few-shot, and fine-tuning scenarios. While LLMs perform poorly in zero-shot and few-shot settings, fine-tuning achieves competitive results compared to smaller multilingual models, albeit at the cost of longer training and inference times.
We provide practical recommendations for real-world applications, enhancing the understanding of cross-lingual ABSA methodologies. This study offers valuable insights into the strengths and limitations of cross-lingual ABSA approaches, advancing the state-of-the-art in this challenging research domain. 

---
# Large Language Models for Summarizing Czech Historical Documents and Beyond 

**Authors**: Václav Tran, Jakub Šmíd, Jiří Martínek, Ladislav Lenc, Pavel Král  

**Link**: [PDF](https://arxiv.org/pdf/2508.10368)  

**Abstract**: Text summarization is the task of shortening a larger body of text into a concise version while retaining its essential meaning and key information. While summarization has been significantly explored in English and other high-resource languages, Czech text summarization, particularly for historical documents, remains underexplored due to linguistic complexities and a scarcity of annotated datasets. Large language models such as Mistral and mT5 have demonstrated excellent results on many natural language processing tasks and languages. Therefore, we employ these models for Czech summarization, resulting in two key contributions: (1) achieving new state-of-the-art results on the modern Czech summarization dataset SumeCzech using these advanced models, and (2) introducing a novel dataset called Posel od Čerchova for summarization of historical Czech documents with baseline results. Together, these contributions provide a great potential for advancing Czech text summarization and open new avenues for research in Czech historical text processing. 

---
# Advancing Cross-lingual Aspect-Based Sentiment Analysis with LLMs and Constrained Decoding for Sequence-to-Sequence Models 

**Authors**: Jakub Šmíd, Pavel Přibáň, Pavel Král  

**Link**: [PDF](https://arxiv.org/pdf/2508.10366)  

**Abstract**: Aspect-based sentiment analysis (ABSA) has made significant strides, yet challenges remain for low-resource languages due to the predominant focus on English. Current cross-lingual ABSA studies often centre on simpler tasks and rely heavily on external translation tools. In this paper, we present a novel sequence-to-sequence method for compound ABSA tasks that eliminates the need for such tools. Our approach, which uses constrained decoding, improves cross-lingual ABSA performance by up to 10\%. This method broadens the scope of cross-lingual ABSA, enabling it to handle more complex tasks and providing a practical, efficient alternative to translation-dependent techniques. Furthermore, we compare our approach with large language models (LLMs) and show that while fine-tuned multilingual LLMs can achieve comparable results, English-centric LLMs struggle with these tasks. 

---
# Making Qwen3 Think in Korean with Reinforcement Learning 

**Authors**: Jungyup Lee, Jemin Kim, Sang Park, SeungJae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.10355)  

**Abstract**: We present a two-stage fine-tuning approach to make the large language model Qwen3 14B "think" natively in Korean. In the first stage, supervised fine-tuning (SFT) on a high-quality Korean reasoning dataset establishes a strong foundation in Korean logical reasoning, yielding notable improvements in Korean-language tasks and even some gains in general reasoning ability. In the second stage, we employ reinforcement learning with a customized Group Relative Policy Optimization (GRPO) algorithm to further enhance both Korean reasoning alignment and overall problem-solving performance. We address critical stability challenges in GRPO training - such as reward hacking and policy collapse - by introducing an oracle judge model that calibrates the reward signal. Our approach achieves stable learning (avoiding the collapse observed in naive GRPO) and leads to steady, incremental performance gains. The final RL-tuned model demonstrates substantially improved results on advanced reasoning benchmarks (particularly math and coding tasks) while maintaining knowledge and language proficiency, successfully conducting its internal chain-of-thought entirely in Korean. 

---
# Cross-Prompt Encoder for Low-Performing Languages 

**Authors**: Beso Mikaberidze, Teimuraz Saghinadze, Simon Ostermann, Philipp Muller  

**Link**: [PDF](https://arxiv.org/pdf/2508.10352)  

**Abstract**: Soft prompts have emerged as a powerful alternative to adapters in parameter-efficient fine-tuning (PEFT), enabling large language models (LLMs) to adapt to downstream tasks without architectural changes or parameter updates. While prior work has focused on stabilizing training via parameter interaction in small neural prompt encoders, their broader potential for transfer across languages remains unexplored. In this paper, we demonstrate that a prompt encoder can play a central role in improving performance on low-performing languages-those that achieve poor accuracy even under full-model fine-tuning. We introduce the Cross-Prompt Encoder (XPE), which combines a lightweight encoding architecture with multi-source training on typologically diverse languages - a design that enables the model to capture abstract and transferable patterns across languages. To complement XPE, we propose a Dual Soft Prompt mechanism that combines an encoder-based prompt with a directly trained standard soft prompt. This hybrid design proves especially effective for target languages that benefit from both broadly shared structure and language-specific alignment. Experiments on the SIB-200 benchmark reveal a consistent trade-off: XPE is most effective for low-performing languages, while hybrid variants offer broader adaptability across multilingual settings. 

---
# Beyond Semantic Understanding: Preserving Collaborative Frequency Components in LLM-based Recommendation 

**Authors**: Minhao Wang, Yunhang He, Cong Xu, Zhangchi Zhu, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10312)  

**Abstract**: Recommender systems in concert with Large Language Models (LLMs) present promising avenues for generating semantically-informed recommendations. However, LLM-based recommenders exhibit a tendency to overemphasize semantic correlations within users' interaction history. When taking pretrained collaborative ID embeddings as input, LLM-based recommenders progressively weaken the inherent collaborative signals as the embeddings propagate through LLM backbones layer by layer, as opposed to traditional Transformer-based sequential models in which collaborative signals are typically preserved or even enhanced for state-of-the-art performance. To address this limitation, we introduce FreLLM4Rec, an approach designed to balance semantic and collaborative information from a spectral perspective. Item embeddings that incorporate both semantic and collaborative information are first purified using a Global Graph Low-Pass Filter (G-LPF) to preliminarily remove irrelevant high-frequency noise. Temporal Frequency Modulation (TFM) then actively preserves collaborative signal layer by layer. Note that the collaborative preservation capability of TFM is theoretically guaranteed by establishing a connection between the optimal but hard-to-implement local graph fourier filters and the suboptimal yet computationally efficient frequency-domain filters. Extensive experiments on four benchmark datasets demonstrate that FreLLM4Rec successfully mitigates collaborative signal attenuation and achieves competitive performance, with improvements of up to 8.00\% in NDCG@10 over the best baseline. Our findings provide insights into how LLMs process collaborative information and offer a principled approach for improving LLM-based recommendation systems. 

---
# From Surface to Semantics: Semantic Structure Parsing for Table-Centric Document Analysis 

**Authors**: Xuan Li, Jialiang Dong, Raymond Wong  

**Link**: [PDF](https://arxiv.org/pdf/2508.10311)  

**Abstract**: Documents are core carriers of information and knowl-edge, with broad applications in finance, healthcare, and scientific research. Tables, as the main medium for structured data, encapsulate key information and are among the most critical document components. Existing studies largely focus on surface-level tasks such as layout analysis, table detection, and data extraction, lacking deep semantic parsing of tables and their contextual associations. This limits advanced tasks like cross-paragraph data interpretation and context-consistent analysis. To address this, we propose DOTABLER, a table-centric semantic document parsing framework designed to uncover deep semantic links between tables and their context. DOTABLER leverages a custom dataset and domain-specific fine-tuning of pre-trained models, integrating a complete parsing pipeline to identify context segments semantically tied to tables. Built on this semantic understanding, DOTABLER implements two core functionalities: table-centric document structure parsing and domain-specific table retrieval, delivering comprehensive table-anchored semantic analysis and precise extraction of semantically relevant tables. Evaluated on nearly 4,000 pages with over 1,000 tables from real-world PDFs, DOTABLER achieves over 90% Precision and F1 scores, demonstrating superior performance in table-context semantic analysis and deep document parsing compared to advanced models such as GPT-4o. 

---
# ReviewRL: Towards Automated Scientific Review with RL 

**Authors**: Sihang Zeng, Kai Tian, Kaiyan Zhang, Yuru wang, Junqi Gao, Runze Liu, Sa Yang, Jingxuan Li, Xinwei Long, Jiaheng Ma, Biqing Qi, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.10308)  

**Abstract**: Peer review is essential for scientific progress but faces growing challenges due to increasing submission volumes and reviewer fatigue. Existing automated review approaches struggle with factual accuracy, rating consistency, and analytical depth, often generating superficial or generic feedback lacking the insights characteristic of high-quality human reviews. We introduce ReviewRL, a reinforcement learning framework for generating comprehensive and factually grounded scientific paper reviews. Our approach combines: (1) an ArXiv-MCP retrieval-augmented context generation pipeline that incorporates relevant scientific literature, (2) supervised fine-tuning that establishes foundational reviewing capabilities, and (3) a reinforcement learning procedure with a composite reward function that jointly enhances review quality and rating accuracy. Experiments on ICLR 2025 papers demonstrate that ReviewRL significantly outperforms existing methods across both rule-based metrics and model-based quality assessments. ReviewRL establishes a foundational framework for RL-driven automatic critique generation in scientific discovery, demonstrating promising potential for future development in this domain. The implementation of ReviewRL will be released at GitHub. 

---
# Yet another algorithmic bias: A Discursive Analysis of Large Language Models Reinforcing Dominant Discourses on Gender and Race 

**Authors**: Gustavo Bonil, Simone Hashiguti, Jhessica Silva, João Gondim, Helena Maia, Nádia Silva, Helio Pedrini, Sandra Avila  

**Link**: [PDF](https://arxiv.org/pdf/2508.10304)  

**Abstract**: With the advance of Artificial Intelligence (AI), Large Language Models (LLMs) have gained prominence and been applied in diverse contexts. As they evolve into more sophisticated versions, it is essential to assess whether they reproduce biases, such as discrimination and racialization, while maintaining hegemonic discourses. Current bias detection approaches rely mostly on quantitative, automated methods, which often overlook the nuanced ways in which biases emerge in natural language. This study proposes a qualitative, discursive framework to complement such methods. Through manual analysis of LLM-generated short stories featuring Black and white women, we investigate gender and racial biases. We contend that qualitative methods such as the one proposed here are fundamental to help both developers and users identify the precise ways in which biases manifest in LLM outputs, thus enabling better conditions to mitigate them. Results show that Black women are portrayed as tied to ancestry and resistance, while white women appear in self-discovery processes. These patterns reflect how language models replicate crystalized discursive representations, reinforcing essentialization and a sense of social immobility. When prompted to correct biases, models offered superficial revisions that maintained problematic meanings, revealing limitations in fostering inclusive narratives. Our results demonstrate the ideological functioning of algorithms and have significant implications for the ethical use and development of AI. The study reinforces the need for critical, interdisciplinary approaches to AI design and deployment, addressing how LLM-generated discourses reflect and perpetuate inequalities. 

---
# Inductive Bias Extraction and Matching for LLM Prompts 

**Authors**: Christian M. Angel, Francis Ferraro  

**Link**: [PDF](https://arxiv.org/pdf/2508.10295)  

**Abstract**: The active research topic of prompt engineering makes it evident that LLMs are sensitive to small changes in prompt wording. A portion of this can be ascribed to the inductive bias that is present in the LLM. By using an LLM's output as a portion of its prompt, we can more easily create satisfactory wording for prompts. This has the effect of creating a prompt that matches the inductive bias in model. Empirically, we show that using this Inductive Bias Extraction and Matching strategy improves LLM Likert ratings used for classification by up to 19% and LLM Likert ratings used for ranking by up to 27%. 

---
# A Computational Approach to Analyzing Language Change and Variation in the Constructed Language Toki Pona 

**Authors**: Daniel Huang, Hyoun-A Joo  

**Link**: [PDF](https://arxiv.org/pdf/2508.10246)  

**Abstract**: This study explores language change and variation in Toki Pona, a constructed language with approximately 120 core words. Taking a computational and corpus-based approach, the study examines features including fluid word classes and transitivity in order to examine (1) changes in preferences of content words for different syntactic positions over time and (2) variation in usage across different corpora. The results suggest that sociolinguistic factors influence Toki Pona in the same way as natural languages, and that even constructed linguistic systems naturally evolve as communities use them. 

---
# Using Large Language Models to Measure Symptom Severity in Patients At Risk for Schizophrenia 

**Authors**: Andrew X. Chen, Guillermo Horga, Sean Escola  

**Link**: [PDF](https://arxiv.org/pdf/2508.10226)  

**Abstract**: Patients who are at clinical high risk (CHR) for schizophrenia need close monitoring of their symptoms to inform appropriate treatments. The Brief Psychiatric Rating Scale (BPRS) is a validated, commonly used research tool for measuring symptoms in patients with schizophrenia and other psychotic disorders; however, it is not commonly used in clinical practice as it requires a lengthy structured interview. Here, we utilize large language models (LLMs) to predict BPRS scores from clinical interview transcripts in 409 CHR patients from the Accelerating Medicines Partnership Schizophrenia (AMP-SCZ) cohort. Despite the interviews not being specifically structured to measure the BPRS, the zero-shot performance of the LLM predictions compared to the true assessment (median concordance: 0.84, ICC: 0.73) approaches human inter- and intra-rater reliability. We further demonstrate that LLMs have substantial potential to improve and standardize the assessment of CHR patients via their accuracy in assessing the BPRS in foreign languages (median concordance: 0.88, ICC: 0.70), and integrating longitudinal information in a one-shot or few-shot learning approach. 

---
# Understanding Textual Emotion Through Emoji Prediction 

**Authors**: Ethan Gordon, Nishank Kuppa, Rigved Tummala, Sriram Anasuri  

**Link**: [PDF](https://arxiv.org/pdf/2508.10222)  

**Abstract**: This project explores emoji prediction from short text sequences using four deep learning architectures: a feed-forward network, CNN, transformer, and BERT. Using the TweetEval dataset, we address class imbalance through focal loss and regularization techniques. Results show BERT achieves the highest overall performance due to its pre-training advantage, while CNN demonstrates superior efficacy on rare emoji classes. This research shows the importance of architecture selection and hyperparameter tuning for sentiment-aware emoji prediction, contributing to improved human-computer interaction. 

---
# Prompt-Response Semantic Divergence Metrics for Faithfulness Hallucination and Misalignment Detection in Large Language Models 

**Authors**: Igor Halperin  

**Link**: [PDF](https://arxiv.org/pdf/2508.10192)  

**Abstract**: The proliferation of Large Language Models (LLMs) is challenged by hallucinations, critical failure modes where models generate non-factual, nonsensical or unfaithful text. This paper introduces Semantic Divergence Metrics (SDM), a novel lightweight framework for detecting Faithfulness Hallucinations -- events of severe deviations of LLMs responses from input contexts. We focus on a specific implementation of these LLM errors, {confabulations, defined as responses that are arbitrary and semantically misaligned with the user's query. Existing methods like Semantic Entropy test for arbitrariness by measuring the diversity of answers to a single, fixed prompt. Our SDM framework improves upon this by being more prompt-aware: we test for a deeper form of arbitrariness by measuring response consistency not only across multiple answers but also across multiple, semantically-equivalent paraphrases of the original prompt. Methodologically, our approach uses joint clustering on sentence embeddings to create a shared topic space for prompts and answers. A heatmap of topic co-occurances between prompts and responses can be viewed as a quantified two-dimensional visualization of the user-machine dialogue. We then compute a suite of information-theoretic metrics to measure the semantic divergence between prompts and responses. Our practical score, $\mathcal{S}_H$, combines the Jensen-Shannon divergence and Wasserstein distance to quantify this divergence, with a high score indicating a Faithfulness hallucination. Furthermore, we identify the KL divergence KL(Answer $||$ Prompt) as a powerful indicator of \textbf{Semantic Exploration}, a key signal for distinguishing different generative behaviors. These metrics are further combined into the Semantic Box, a diagnostic framework for classifying LLM response types, including the dangerous, confident confabulation. 

---
# PakBBQ: A Culturally Adapted Bias Benchmark for QA 

**Authors**: Abdullah Hashmat, Muhammad Arham Mirza, Agha Ali Raza  

**Link**: [PDF](https://arxiv.org/pdf/2508.10186)  

**Abstract**: With the widespread adoption of Large Language Models (LLMs) across various applications, it is empirical to ensure their fairness across all user communities. However, most LLMs are trained and evaluated on Western centric data, with little attention paid to low-resource languages and regional contexts. To address this gap, we introduce PakBBQ, a culturally and regionally adapted extension of the original Bias Benchmark for Question Answering (BBQ) dataset. PakBBQ comprises over 214 templates, 17180 QA pairs across 8 categories in both English and Urdu, covering eight bias dimensions including age, disability, appearance, gender, socio-economic status, religious, regional affiliation, and language formality that are relevant in Pakistan. We evaluate multiple multilingual LLMs under both ambiguous and explicitly disambiguated contexts, as well as negative versus non negative question framings. Our experiments reveal (i) an average accuracy gain of 12\% with disambiguation, (ii) consistently stronger counter bias behaviors in Urdu than in English, and (iii) marked framing effects that reduce stereotypical responses when questions are posed negatively. These findings highlight the importance of contextualized benchmarks and simple prompt engineering strategies for bias mitigation in low resource settings. 

---
# Efficient Forward-Only Data Valuation for Pretrained LLMs and VLMs 

**Authors**: Wenlong Deng, Jiaming Zhang, Qi Zeng, Christos Thrampoulidis, Boying Gong, Xiaoxiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.10180)  

**Abstract**: Quantifying the influence of individual training samples is essential for enhancing the transparency and accountability of large language models (LLMs) and vision-language models (VLMs). However, existing data valuation methods often rely on Hessian information or model retraining, making them computationally prohibitive for billion-parameter models. In this work, we introduce For-Value, a forward-only data valuation framework that enables scalable and efficient influence estimation for both LLMs and VLMs. By leveraging the rich representations of modern foundation models, For-Value computes influence scores using a simple closed-form expression based solely on a single forward pass, thereby eliminating the need for costly gradient computations. Our theoretical analysis demonstrates that For-Value accurately estimates per-sample influence by capturing alignment in hidden representations and prediction errors between training and validation samples. Extensive experiments show that For-Value matches or outperforms gradient-based baselines in identifying impactful fine-tuning examples and effectively detecting mislabeled data. 

---
# Estimating Machine Translation Difficulty 

**Authors**: Lorenzo Proietti, Stefano Perrella, Vilém Zouhar, Roberto Navigli, Tom Kocmi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10175)  

**Abstract**: Machine translation quality has began achieving near-perfect translations in some setups. These high-quality outputs make it difficult to distinguish between state-of-the-art models and to identify areas for future improvement. Automatically identifying texts where machine translation systems struggle holds promise for developing more discriminative evaluations and guiding future research.
We formalize the task of translation difficulty estimation, defining a text's difficulty based on the expected quality of its translations. We introduce a new metric to evaluate difficulty estimators and use it to assess both baselines and novel approaches. Finally, we demonstrate the practical utility of difficulty estimators by using them to construct more challenging machine translation benchmarks. Our results show that dedicated models (dubbed Sentinel-src) outperform both heuristic-based methods (e.g. word rarity or syntactic complexity) and LLM-as-a-judge approaches. We release two improved models for difficulty estimation, Sentinel-src-24 and Sentinel-src-25, which can be used to scan large collections of texts and select those most likely to challenge contemporary machine translation systems. 

---
# LaajMeter: A Framework for LaaJ Evaluation 

**Authors**: Gal Amram, Eitan Farchi, Shmulik Froimovich, Raviv Gal, Avi Ziv  

**Link**: [PDF](https://arxiv.org/pdf/2508.10161)  

**Abstract**: Large Language Models (LLMs) are increasingly used as evaluators in natural language processing tasks, a paradigm known as LLM-as-a-Judge (LaaJ). While effective in general domains, LaaJs pose significant challenges in domain-specific contexts, where annotated data is scarce and expert evaluation is costly. In such cases, meta-evaluation is often performed using metrics that have not been validated for the specific domain in which they are applied. As a result, it becomes difficult to determine which metrics effectively identify LaaJ quality, and further, what threshold indicates sufficient evaluator performance. In this work, we introduce LaaJMeter, a simulation-based framework for controlled meta-evaluation of LaaJs. LaaJMeter enables engineers to generate synthetic data representing virtual models and judges, allowing systematic analysis of evaluation metrics under realistic conditions. This helps practitioners validate and refine LaaJs for specific evaluation tasks: they can test whether their metrics correctly distinguish between better and worse (virtual) LaaJs, and estimate appropriate thresholds for evaluator adequacy.
We demonstrate the utility of LaaJMeter in a code translation task involving a legacy programming language, showing how different metrics vary in sensitivity to evaluator quality. Our results highlight the limitations of common metrics and the importance of principled metric selection. LaaJMeter provides a scalable and extensible solution for assessing LaaJs in low-resource settings, contributing to the broader effort to ensure trustworthy and reproducible evaluation in NLP. 

---
# Multi-Turn Puzzles: Evaluating Interactive Reasoning and Strategic Dialogue in LLMs 

**Authors**: Kartikeya Badola, Jonathan Simon, Arian Hosseini, Sara Marie Mc Carthy, Tsendsuren Munkhdalai, Abhimanyu Goyal, Tomáš Kočiský, Shyam Upadhyay, Bahare Fatemi, Mehran Kazemi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10142)  

**Abstract**: Large language models (LLMs) excel at solving problems with clear and complete statements, but often struggle with nuanced environments or interactive tasks which are common in most real-world scenarios. This highlights the critical need for developing LLMs that can effectively engage in logically consistent multi-turn dialogue, seek information and reason with incomplete data. To this end, we introduce a novel benchmark comprising a suite of multi-turn tasks each designed to test specific reasoning, interactive dialogue, and information-seeking abilities. These tasks have deterministic scoring mechanisms, thus eliminating the need for human intervention. Evaluating frontier models on our benchmark reveals significant headroom. Our analysis shows that most errors emerge from poor instruction following, reasoning failures, and poor planning. This benchmark provides valuable insights into the strengths and weaknesses of current LLMs in handling complex, interactive scenarios and offers a robust platform for future research aimed at improving these critical capabilities. 

---
# mSCoRe: a $M$ultilingual and Scalable Benchmark for $S$kill-based $Co$mmonsense $Re$asoning 

**Authors**: Nghia Trung Ngo, Franck Dernoncourt, Thien Huu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10137)  

**Abstract**: Recent advancements in reasoning-reinforced Large Language Models (LLMs) have shown remarkable capabilities in complex reasoning tasks. However, the mechanism underlying their utilization of different human reasoning skills remains poorly investigated, especially for multilingual commonsense reasoning that involves everyday knowledge across different languages and cultures. To address this gap, we propose a \textbf{M}ultilingual and Scalable Benchmark for \textbf{S}kill-based \textbf{Co}mmonsense \textbf{Re}asoning (\textbf{mSCoRe}). Our benchmark incorporates three key components that are designed to systematically evaluate LLM's reasoning capabilities, including: (1) a novel taxonomy of reasoning skills that enables fine-grained analysis of models' reasoning processes, (2) a robust data synthesis pipeline tailored specifically for commonsense reasoning evaluation, and (3) a complexity scaling framework allowing task difficulty to scale dynamically alongside future improvements in LLM abilities. Extensive experiments on eights state-of-the-art LLMs of varying sizes and training approaches demonstrate that \textbf{mSCoRe} remains significantly challenging for current models, particularly at higher complexity levels. Our results reveal the limitations of such reasoning-reinforced models when confronted with nuanced multilingual general and cultural commonsense. We further provide detailed analysis on the models' reasoning processes, suggesting future directions for improving multilingual commonsense reasoning capabilities. 

---
# Reflect then Learn: Active Prompting for Information Extraction Guided by Introspective Confusion 

**Authors**: Dong Zhao, Yadong Wang, Xiang Chen, Chenxi Wang, Hongliang Dai, Chuanxing Geng, Shengzhong Zhang, Shaoyuan Li, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10036)  

**Abstract**: Large Language Models (LLMs) show remarkable potential for few-shot information extraction (IE), yet their performance is highly sensitive to the choice of in-context examples. Conventional selection strategies often fail to provide informative guidance, as they overlook a key source of model fallibility: confusion stemming not just from semantic content, but also from the generation of well-structured formats required by IE tasks. To address this, we introduce Active Prompting for Information Extraction (APIE), a novel active prompting framework guided by a principle we term introspective confusion. Our method empowers an LLM to assess its own confusion through a dual-component uncertainty metric that uniquely quantifies both Format Uncertainty (difficulty in generating correct syntax) and Content Uncertainty (inconsistency in extracted semantics). By ranking unlabeled data with this comprehensive score, our framework actively selects the most challenging and informative samples to serve as few-shot exemplars. Extensive experiments on four benchmarks show that our approach consistently outperforms strong baselines, yielding significant improvements in both extraction accuracy and robustness. Our work highlights the critical importance of a fine-grained, dual-level view of model uncertainty when it comes to building effective and reliable structured generation systems. 

---
# The Cost of Thinking: Increased Jailbreak Risk in Large Language Models 

**Authors**: Fan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10032)  

**Abstract**: Thinking mode has always been regarded as one of the most valuable modes in LLMs. However, we uncover a surprising and previously overlooked phenomenon: LLMs with thinking mode are more easily broken by Jailbreak attack. We evaluate 9 LLMs on AdvBench and HarmBench and find that the success rate of attacking thinking mode in LLMs is almost higher than that of non-thinking mode. Through large numbers of sample studies, it is found that for educational purposes and excessively long thinking lengths are the characteristics of successfully attacked data, and LLMs also give harmful answers when they mostly know that the questions are harmful. In order to alleviate the above problems, this paper proposes a method of safe thinking intervention for LLMs, which explicitly guides the internal thinking processes of LLMs by adding "specific thinking tokens" of LLMs to the prompt. The results demonstrate that the safe thinking intervention can significantly reduce the attack success rate of LLMs with thinking mode. 

---
# Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models 

**Authors**: Saaduddin Mahmud, Mason Nakamura, Kyle H. Wray, Shlomo Zilberstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.10030)  

**Abstract**: Prompt optimization methods have demonstrated significant effectiveness in aligning black-box large language models (LLMs). In parallel, inference scaling strategies such as Best-of-N Sampling and Majority Voting have also proven to enhance alignment and performance by trading off computation. However, existing prompt optimization approaches are inference strategy agnostic; that is, they optimize prompts without regard to the inference strategy employed during deployment. This constitutes a significant methodological gap, as our empirical and theoretical analysis reveals a strong interdependence between these two paradigms. Moreover, we find that user preferences regarding trade-offs among multiple objectives and inference budgets substantially influence the choice of prompt and inference configuration. To address this gap, we introduce a unified novel framework named IAPO (Inference-Aware Prompt Optimization) that jointly optimizes the prompt and inference scale, while being aware of the inference budget and different task objectives. We then develop a fixed-budget training algorithm for IAPO, which we call PSST (Prompt Scaling via Sequential Trimming), and analyze finite-budget guarantees on error probability. Finally, we evaluate the effectiveness of PSST on six different tasks, including multi-objective text generation and reasoning, and demonstrate the critical role of incorporating inference-awareness when aligning black-box LLMs through prompt optimization. 

---
# Latent Fusion Jailbreak: Blending Harmful and Harmless Representations to Elicit Unsafe LLM Outputs 

**Authors**: Wenpeng Xing, Mohan Li, Chunqiang Hu, Haitao XuNingyu Zhang, Bo Lin, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.10029)  

**Abstract**: Large language models (LLMs) demonstrate impressive capabilities in various language tasks but are susceptible to jailbreak attacks that circumvent their safety alignments. This paper introduces Latent Fusion Jailbreak (LFJ), a representation-based attack that interpolates hidden states from harmful and benign query pairs to elicit prohibited responses. LFJ begins by selecting query pairs with high thematic and syntactic similarity, then performs gradient-guided interpolation at influential layers and tokens, followed by optimization to balance attack success, output fluency, and computational efficiency. Evaluations on models such as Vicuna and LLaMA-2 across benchmarks like AdvBench and MaliciousInstruct yield an average attack success rate (ASR) of 94.01%, outperforming existing methods. To mitigate LFJ, we propose an adversarial training defense that fine-tunes models on interpolated examples, reducing ASR by over 80% without degrading performance on benign inputs. Ablation studies validate the importance of query pair selection, hidden state interpolation components, and optimization strategies in LFJ's effectiveness. 

---
# PREF: Reference-Free Evaluation of Personalised Text Generation in LLMs 

**Authors**: Xiao Fu, Hossein A. Rahmani, Bin Wu, Jerome Ramos, Emine Yilmaz, Aldo Lipani  

**Link**: [PDF](https://arxiv.org/pdf/2508.10028)  

**Abstract**: Personalised text generation is essential for user-centric information systems, yet most evaluation methods overlook the individuality of users. We introduce \textbf{PREF}, a \textbf{P}ersonalised \textbf{R}eference-free \textbf{E}valuation \textbf{F}ramework that jointly measures general output quality and user-specific alignment without requiring gold personalised references. PREF operates in a three-step pipeline: (1) a coverage stage uses a large language model (LLM) to generate a comprehensive, query-specific guideline covering universal criteria such as factuality, coherence, and completeness; (2) a preference stage re-ranks and selectively augments these factors using the target user's profile, stated or inferred preferences, and context, producing a personalised evaluation rubric; and (3) a scoring stage applies an LLM judge to rate candidate answers against this rubric, ensuring baseline adequacy while capturing subjective priorities. This separation of coverage from preference improves robustness, transparency, and reusability, and allows smaller models to approximate the personalised quality of larger ones. Experiments on the PrefEval benchmark, including implicit preference-following tasks, show that PREF achieves higher accuracy, better calibration, and closer alignment with human judgments than strong baselines. By enabling scalable, interpretable, and user-aligned evaluation, PREF lays the groundwork for more reliable assessment and development of personalised language generation systems. 

---
# LLMCARE: Alzheimer's Detection via Transformer Models Enhanced by LLM-Generated Synthetic Data 

**Authors**: Ali Zolnour, Hossein Azadmaleki, Yasaman Haghbin, Fatemeh Taherinezhad, Mohamad Javad Momeni Nezhad, Sina Rashidi, Masoud Khani, AmirSajjad Taleban, Samin Mahdizadeh Sani, Maryam Dadkhah, James M. Noble, Suzanne Bakken, Yadollah Yaghoobzadeh, Abdol-Hossein Vahabie, Masoud Rouhizadeh, Maryam Zolnoori  

**Link**: [PDF](https://arxiv.org/pdf/2508.10027)  

**Abstract**: Alzheimer's disease and related dementias (ADRD) affect approximately five million older adults in the U.S., yet over half remain undiagnosed. Speech-based natural language processing (NLP) offers a promising, scalable approach to detect early cognitive decline through linguistic markers.
To develop and evaluate a screening pipeline that (i) fuses transformer embeddings with handcrafted linguistic features, (ii) tests data augmentation using synthetic speech generated by large language models (LLMs), and (iii) benchmarks unimodal and multimodal LLM classifiers for ADRD detection.
Transcripts from the DementiaBank "cookie-theft" task (n = 237) were used. Ten transformer models were evaluated under three fine-tuning strategies. A fusion model combined embeddings from the top-performing transformer with 110 lexical-derived linguistic features. Five LLMs (LLaMA-8B/70B, MedAlpaca-7B, Ministral-8B, GPT-4o) were fine-tuned to generate label-conditioned synthetic speech, which was used to augment training data. Three multimodal models (GPT-4o, Qwen-Omni, Phi-4) were tested for speech-text classification in zero-shot and fine-tuned settings.
The fusion model achieved F1 = 83.3 (AUC = 89.5), outperforming linguistic or transformer-only baselines. Augmenting training data with 2x MedAlpaca-7B synthetic speech increased F1 to 85.7. Fine-tuning significantly improved unimodal LLM classifiers (e.g., MedAlpaca: F1 = 47.3 -> 78.5 F1). Current multimodal models demonstrated lower performance (GPT-4o = 70.2 F1; Qwen = 66.0). Performance gains aligned with the distributional similarity between synthetic and real speech.
Integrating transformer embeddings with linguistic features enhances ADRD detection from speech. Clinically tuned LLMs effectively support both classification and data augmentation, while further advancement is needed in multimodal modeling. 

---
# SABER: Switchable and Balanced Training for Efficient LLM Reasoning 

**Authors**: Kai Zhao, Yanjun Zhao, Jiaming Song, Shien He, Lusheng Zhang, Qiang Zhang, Tianjiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.10026)  

**Abstract**: Large language models (LLMs) empowered by chain-of-thought reasoning have achieved impressive accuracy on complex tasks but suffer from excessive inference costs and latency when applied uniformly to all problems. We propose SABER (Switchable and Balanced Training for Efficient LLM Reasoning), a reinforcement learning framework that endows LLMs with user-controllable, token-budgeted reasoning. SABER first profiles each training example's base-model thinking token usage and assigns it to one of the predefined budget tiers. During fine-tuning, the model is guided by system prompts and length-aware rewards to respect its assigned budget. In parallel, we incorporate no-think examples to ensure the model remains reliable even when explicit reasoning is turned off. SABER further supports four discrete inference modes - NoThink, FastThink, CoreThink, and DeepThink, enabling flexible trade-offs between latency and reasoning depth. Extensive evaluations on math reasoning (MATH, GSM8K), code generation (MBPP), and logical reasoning (LiveBench-Reasoning) demonstrate that SABER achieves high accuracy under tight budgets, graceful degradation, and effective cross-scale and cross-domain generalization. In particular, SABER-FastThink cuts reasoning length by 65.4% and yields a 3.6% accuracy gain compared with the base model on the MATH benchmark. 

---
# Detecting and explaining postpartum depression in real-time with generative artificial intelligence 

**Authors**: Silvia García-Méndez, Francisco de Arriba-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2508.10025)  

**Abstract**: Among the many challenges mothers undergo after childbirth, postpartum depression (PPD) is a severe condition that significantly impacts their mental and physical well-being. Consequently, the rapid detection of ppd and their associated risk factors is critical for in-time assessment and intervention through specialized prevention procedures. Accordingly, this work addresses the need to help practitioners make decisions with the latest technological advancements to enable real-time screening and treatment recommendations. Mainly, our work contributes to an intelligent PPD screening system that combines Natural Language Processing, Machine Learning (ML), and Large Language Models (LLMs) towards an affordable, real-time, and non-invasive free speech analysis. Moreover, it addresses the black box problem since the predictions are described to the end users thanks to the combination of LLMs with interpretable ml models (i.e., tree-based algorithms) using feature importance and natural language. The results obtained are 90 % on ppd detection for all evaluation metrics, outperforming the competing solutions in the literature. Ultimately, our solution contributes to the rapid detection of PPD and their associated risk factors, critical for in-time and proper assessment and intervention. 

---
# RTTC: Reward-Guided Collaborative Test-Time Compute 

**Authors**: J. Pablo Muñoz, Jinjie Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10024)  

**Abstract**: Test-Time Compute (TTC) has emerged as a powerful paradigm for enhancing the performance of Large Language Models (LLMs) at inference, leveraging strategies such as Test-Time Training (TTT) and Retrieval-Augmented Generation (RAG). However, the optimal adaptation strategy varies across queries, and indiscriminate application of TTC strategy incurs substantial computational overhead. In this work, we introduce Reward-Guided Test-Time Compute (RTTC), a novel framework that adaptively selects the most effective TTC strategy for each query via a pretrained reward model, maximizing downstream accuracy across diverse domains and tasks. RTTC operates in a distributed server-client architecture, retrieving relevant samples from a remote knowledge base and applying RAG or lightweight fine-tuning on client devices only when necessary. To further mitigate redundant computation, we propose Query-State Caching, which enables the efficient reuse of historical query states at both retrieval and adaptation levels. Extensive experiments across multiple LLMs and benchmarks demonstrate that RTTC consistently achieves superior accuracy compared to vanilla RAG or TTT, validating the necessity of adaptive, reward-guided TTC selection and the potential of RTTC for scalable, high-performance language model adaptation. 

---
# Conformal P-Value in Multiple-Choice Question Answering Tasks with Provable Risk Control 

**Authors**: Yuanchang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2508.10022)  

**Abstract**: This study introduces a significance testing-enhanced conformal prediction (CP) framework to improve trustworthiness of large language models (LLMs) in multiple-choice question answering (MCQA). While LLMs have been increasingly deployed in disciplinary QA scenarios, hallucination and nonfactual generation substantially compromise response reliability. Although CP provides statistically rigorous marginal coverage guarantees for prediction sets, and significance testing offers established statistical rigor, their synergistic integration remains unexplored. To mitigate hallucination and factual inaccuracies, our framework integrates $p$-value computation with conformity scoring through self-consistency resampling of MCQA responses. This approach calculates option frequencies to address LLMs' black-box nature, subsequently constructing prediction sets via null hypothesis testing ($\mathcal{H}_0$) with empirically derived $p$-values. Evaluations on MMLU and MMLU-Pro benchmarks using off-the-shelf LLMs demonstrate: (1) The enhanced CP achieves user-specified empirical miscoverage rates; (2) Test-set average prediction set size (APSS) decreases monotonically with increasing risk levels ($\alpha$), validating APSS as an effective uncertainty metric. This work establishes a principled statistical framework for trustworthy LLM deployment in high-stakes QA applications. 

---
# LATTE: Learning Aligned Transactions and Textual Embeddings for Bank Clients 

**Authors**: Egor Fadeev, Dzhambulat Mollaev, Aleksei Shestov, Dima Korolev, Omar Zoloev, Ivan Kireev, Andrey Savchenko, Maksim Makarenko  

**Link**: [PDF](https://arxiv.org/pdf/2508.10021)  

**Abstract**: Learning clients embeddings from sequences of their historic communications is central to financial applications. While large language models (LLMs) offer general world knowledge, their direct use on long event sequences is computationally expensive and impractical in real-world pipelines. In this paper, we propose LATTE, a contrastive learning framework that aligns raw event embeddings with semantic embeddings from frozen LLMs. Behavioral features are summarized into short prompts, embedded by the LLM, and used as supervision via contrastive loss. The proposed approach significantly reduces inference cost and input size compared to conventional processing of complete sequence by LLM. We experimentally show that our method outperforms state-of-the-art techniques for learning event sequence representations on real-world financial datasets while remaining deployable in latency-sensitive environments. 

---
# FedCoT: Communication-Efficient Federated Reasoning Enhancement for Large Language Models 

**Authors**: Chuan Li, Qianyi Zhao, Fengran Mo, Cen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10020)  

**Abstract**: Efficiently enhancing the reasoning capabilities of large language models (LLMs) in federated learning environments remains challenging, particularly when balancing performance gains with strict computational, communication, and privacy constraints. This challenge is especially acute in healthcare, where decisions-spanning clinical, operational, and patient-facing contexts-demand not only accurate outputs but also interpretable, traceable rationales to ensure safety, accountability, and regulatory compliance. Conventional federated tuning approaches on LLM fail to address this need: they optimize primarily for answer correctness while neglecting rationale quality, leaving CoT capabilities dependent on models' innate pre-training abilities. Moreover, existing methods for improving rationales typically rely on privacy-violating knowledge distillation from centralized models. Additionally, the communication overhead in traditional federated fine-tuning on LLMs remains substantial. We addresses this gap by proposing FedCoT, a novel framework specifically designed to enhance reasoning in federated settings. FedCoT leverages a lightweight chain-of-thought enhancement mechanism: local models generate multiple reasoning paths, and a compact discriminator dynamically selects the most promising one. This approach improves reasoning accuracy and robustness while providing valuable interpretability, which is particularly critical for medical applications. To manage client heterogeneity efficiently, we adopt an improved aggregation approach building upon advanced LoRA module stacking, incorporating client classifier-awareness to achieve noise-free aggregation across diverse clients. Comprehensive experiments on medical reasoning tasks demonstrate that FedCoT significantly boosts client-side reasoning performance under stringent resource budgets while fully preserving data privacy. 

---
# Decoupling Understanding from Reasoning via Problem Space Mapping for Small-scale Model Reasoning 

**Authors**: Li Wang, Changhao Zhang, Zengqi Xiu, Kai Lu, Xin Yu, Kui Zhang, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10019)  

**Abstract**: Despite recent advances in the reasoning capabilities of Large Language Models (LLMs), improving the reasoning ability of Small Language Models (SLMs, e.g., $\leq$ 1.5B) remains challenging. A key obstacle lies in the complexity and variability of natural language: essentially equivalent problems often appear in diverse surface forms, often obscured by redundant or distracting details. This imposes a dual burden on SLMs: they must first extract the core problem from complex linguistic input, and then perform reasoning based on that understanding. The resulting vast and noisy problem space hinders optimization, particularly for models with limited capacity. To address this, we propose a new framework that decouples understanding from reasoning by mapping natural language problems into a canonical problem space-a semantically simplified yet expressive domain. This enables SLMs to focus on reasoning over standardized inputs, free from linguistic variability. Within this framework, we introduce DURIT (Decoupled Understanding from Reasoning via Iterative Training), a three-step algorithm that iteratively: (1) mapping natural language problems via reinforcement learning, (2) aligns reasoning trajectories through self-distillation, and (3) trains reasoning policies in the problem space. The mapper and reasoner are co-trained in an alternating loop throughout this process. Experiments show that DURIT substantially improves SLMs' performance on both in-domain and out-of-domain mathematical and logical reasoning tasks. Beyond improving reasoning capabilities, DURIT also improves the robustness of reasoning, validating decoupling understanding from reasoning as an effective strategy for strengthening SLMs. 

---
# A Rose by Any Other Name Would Smell as Sweet: Categorical Homotopy Theory for Large Language Models 

**Authors**: Sridhar Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10018)  

**Abstract**: Natural language is replete with superficially different statements, such as ``Charles Darwin wrote" and ``Charles Darwin is the author of", which carry the same meaning. Large language models (LLMs) should generate the same next-token probabilities in such cases, but usually do not. Empirical workarounds have been explored, such as using k-NN estimates of sentence similarity to produce smoothed estimates. In this paper, we tackle this problem more abstractly, introducing a categorical homotopy framework for LLMs. We introduce an LLM Markov category to represent probability distributions in language generated by an LLM, where the probability of a sentence, such as ``Charles Darwin wrote" is defined by an arrow in a Markov category. However, this approach runs into difficulties as language is full of equivalent rephrases, and each generates a non-isomorphic arrow in the LLM Markov category. To address this fundamental problem, we use categorical homotopy techniques to capture ``weak equivalences" in an LLM Markov category. We present a detailed overview of application of categorical homotopy to LLMs, from higher algebraic K-theory to model categories, building on powerful theoretical results developed over the past half a century. 

---
# Training-Free Multimodal Large Language Model Orchestration 

**Authors**: Tianyu Xie, Yuhang Wu, Yongdong Luo, Jiayi Ji, Xiawu Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.10016)  

**Abstract**: Different Multimodal Large Language Models (MLLMs) cannot be integrated into a unified multimodal input-output system directly. In previous work, training has been considered as an inevitable component due to challenges in modal alignment, Text-to-Speech efficiency and other integration issues. In this paper, we introduce Multimodal Large Language Model Orchestration, an effective approach for creating interactive multimodal AI systems without additional training. MLLM Orchestration leverages the inherent reasoning capabilities of large language models to coordinate specialized models through explicit workflows, enabling natural multimodal interactions while maintaining modularity, improving interpretability, and significantly enhancing computational efficiency. Our orchestration framework is built upon three key innovations: (1) a central controller LLM that analyzes user inputs and dynamically routes tasks to appropriate specialized models through carefully designed agents; (2) a parallel Text-to-Speech architecture that enables true full-duplex interaction with seamless interruption handling and natural conversational flow; and (3) a cross-modal memory integration system that maintains coherent context across modalities through intelligent information synthesis and retrieval, selectively avoiding unnecessary modality calls in certain scenarios to improve response speed. Extensive evaluations demonstrate that MLLM Orchestration achieves comprehensive multimodal capabilities without additional training, performance improvements of up to 7.8% over traditional jointly-trained approaches on standard benchmarks, reduced latency by 10.3%, and significantly enhanced interpretability through explicit orchestration processes. 

---
# RealTalk-CN: A Realistic Chinese Speech-Text Dialogue Benchmark With Cross-Modal Interaction Analysis 

**Authors**: Enzhi Wang, Qicheng Li, Shiwan Zhao, Aobo Kong, Jiaming Zhou, Xi Yang, Yequan Wang, Yonghua Lin, Yong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2508.10015)  

**Abstract**: In recent years, large language models (LLMs) have achieved remarkable advancements in multimodal processing, including end-to-end speech-based language models that enable natural interactions and perform specific tasks in task-oriented dialogue (TOD) systems. However, existing TOD datasets are predominantly text-based, lacking real speech signals that are essential for evaluating the robustness of speech-based LLMs. Moreover, existing speech TOD datasets are primarily English and lack critical aspects such as speech disfluencies and speaker variations. To address these gaps, we introduce RealTalk-CN, the first Chinese multi-turn, multi-domain speech-text dual-modal TOD dataset, comprising 5.4k dialogues (60K utterances, 150 hours) with paired speech-text annotations. RealTalk-CN captures diverse dialogue scenarios with annotated spontaneous speech disfluencies, ensuring comprehensive coverage of real-world complexities in speech dialogue. In addition, we propose a novel cross-modal chat task that authentically simulates real-world user interactions, allowing dynamic switching between speech and text modalities. Our evaluation covers robustness to speech disfluencies, sensitivity to speaker characteristics, and cross-domain performance. Extensive experiments validate the effectiveness of RealTalk-CN, establishing a strong foundation for Chinese speech-based LLMs research. 

---
# PersonaEval: Are LLM Evaluators Human Enough to Judge Role-Play? 

**Authors**: Lingfeng Zhou, Jialing Zhang, Jin Gao, Mohan Jiang, Dequan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10014)  

**Abstract**: Current role-play studies often rely on unvalidated LLM-as-a-judge paradigms, which may fail to reflect how humans perceive role fidelity. A key prerequisite for human-aligned evaluation is role identification, the ability to recognize who is speaking based on dialogue context. We argue that any meaningful judgment of role-playing quality (how well a character is played) fundamentally depends on first correctly attributing words and actions to the correct persona (who is speaking). We present PersonaEval, the first benchmark designed to test whether LLM evaluators can reliably identify human roles. PersonaEval uses human-authored dialogues from novels, scripts, and video transcripts, challenging models to determine the correct persona according to the conversation context. Our experiments, including a human study, show that even the best-performing LLMs reach only around 69% accuracy, well below the level needed for reliable evaluation. In contrast, human participants perform near ceiling with 90.8% accuracy, highlighting that current LLM evaluators are still not human enough to effectively judge role-play scenarios. To better understand this gap, we examine training-time adaptation and test-time compute, suggesting that reliable evaluation requires more than task-specific tuning, but depends on strong, human-like reasoning abilities in LLM evaluators. We release our benchmark at this https URL. 

---
# Semantic Bridge: Universal Multi-Hop Question Generation via AMR-Driven Graph Synthesis 

**Authors**: Linqing Chen, Hanmeng Zhong, Wentao Wu, Weilei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10013)  

**Abstract**: Large language model (LLM) training faces a critical bottleneck: the scarcity of high-quality, reasoning-intensive question-answer pairs, especially from sparse, domain-specific sources like PubMed papers or legal documents. Existing methods rely on surface patterns, fundamentally failing to generate controllable, complex multi-hop reasoning questions that test genuine understanding-essential for advancing LLM training paradigms. We present \textbf{Semantic Bridge}, the first universal framework for controllably generating sophisticated multi-hop reasoning questions from arbitrary sources. Our breakthrough innovation is \textit{semantic graph weaving}-three complementary bridging mechanisms (entity bridging for role-varying shared entities, predicate chain bridging for temporal/causal/logical sequences, and causal bridging for explicit reasoning chains)-that systematically construct complex pathways across documents, with fine-grained control over complexity and types via AMR-driven analysis. Our multi-modal AMR pipeline achieves up to 9.5% better round-trip quality, enabling production-ready controllable QA generation. Extensive evaluation demonstrates performance across both general-purpose datasets (Wikipedia) and specialized domains (biomedicine) It yields consistent 18.3%-25.4% gains over baselines across four languages (English, Chinese, French, German). Question pairs generated from 200 sources outperform 600 native human annotation examples with 67% fewer materials. Human evaluation shows 23.4% higher complexity, 18.7% better answerability, and 31.2% improved pattern coverage. Semantic Bridge establishes a new paradigm for LLM training data synthesis, enabling controllable generation of targeted reasoning questions from sparse sources. We will release our core code and semantic bridge model. 

---
# Guided Navigation in Knowledge-Dense Environments: Structured Semantic Exploration with Guidance Graphs 

**Authors**: Dehao Tao, Guangjie Liu, Weizheng, Yongfeng Huang, Minghu jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10012)  

**Abstract**: While Large Language Models (LLMs) exhibit strong linguistic capabilities, their reliance on static knowledge and opaque reasoning processes limits their performance in knowledge intensive tasks. Knowledge graphs (KGs) offer a promising solution, but current exploration methods face a fundamental trade off: question guided approaches incur redundant exploration due to granularity mismatches, while clue guided methods fail to effectively leverage contextual information for complex scenarios. To address these limitations, we propose Guidance Graph guided Knowledge Exploration (GG Explore), a novel framework that introduces an intermediate Guidance Graph to bridge unstructured queries and structured knowledge retrieval. The Guidance Graph defines the retrieval space by abstracting the target knowledge' s structure while preserving broader semantic context, enabling precise and efficient exploration. Building upon the Guidance Graph, we develop: (1) Structural Alignment that filters incompatible candidates without LLM overhead, and (2) Context Aware Pruning that enforces semantic consistency with graph constraints. Extensive experiments show our method achieves superior efficiency and outperforms SOTA, especially on complex tasks, while maintaining strong performance with smaller LLMs, demonstrating practical value. 

---
# Evaluation of GPT-based large language generative AI models as study aids for the national licensure examination for registered dietitians in Japan 

**Authors**: Yuta Nagamori, Mikoto Kosai, Yuji Kawai, Haruka Marumo, Misaki Shibuya, Tatsuya Negishi, Masaki Imanishi, Yasumasa Ikeda, Koichiro Tsuchiya, Asuka Sawai, Licht Miyamoto  

**Link**: [PDF](https://arxiv.org/pdf/2508.10011)  

**Abstract**: Generative artificial intelligence (AI) based on large language models (LLMs), such as ChatGPT, has demonstrated remarkable progress across various professional fields, including medicine and education. However, their performance in nutritional education, especially in Japanese national licensure examination for registered dietitians, remains underexplored. This study aimed to evaluate the potential of current LLM-based generative AI models as study aids for nutrition students. Questions from the Japanese national examination for registered dietitians were used as prompts for ChatGPT and three Bing models (Precise, Creative, Balanced), based on GPT-3.5 and GPT-4. Each question was entered into independent sessions, and model responses were analyzed for accuracy, consistency, and response time. Additional prompt engineering, including role assignment, was tested to assess potential performance improvements. Bing-Precise (66.2%) and Bing-Creative (61.4%) surpassed the passing threshold (60%), while Bing-Balanced (43.3%) and ChatGPT (42.8%) did not. Bing-Precise and Bing-Creative generally outperformed others across subject fields except Nutrition Education, where all models underperformed. None of the models consistently provided the same correct responses across repeated attempts, highlighting limitations in answer stability. ChatGPT showed greater consistency in response patterns but lower accuracy. Prompt engineering had minimal effect, except for modest improvement when correct answers and explanations were explicitly provided. While some generative AI models marginally exceeded the passing threshold, overall accuracy and answer consistency remained suboptimal. Moreover, all the models demonstrated notable limitations in answer consistency and robustness. Further advancements are needed to ensure reliable and stable AI-based study aids for dietitian licensure preparation. 

---
# An Audit and Analysis of LLM-Assisted Health Misinformation Jailbreaks Against LLMs 

**Authors**: Ayana Hussain, Patrick Zhao, Nicholas Vincent  

**Link**: [PDF](https://arxiv.org/pdf/2508.10010)  

**Abstract**: Large Language Models (LLMs) are a double-edged sword capable of generating harmful misinformation -- inadvertently, or when prompted by "jailbreak" attacks that attempt to produce malicious outputs. LLMs could, with additional research, be used to detect and prevent the spread of misinformation. In this paper, we investigate the efficacy and characteristics of LLM-produced jailbreak attacks that cause other models to produce harmful medical misinformation. We also study how misinformation generated by jailbroken LLMs compares to typical misinformation found on social media, and how effectively it can be detected using standard machine learning approaches. Specifically, we closely examine 109 distinct attacks against three target LLMs and compare the attack prompts to in-the-wild health-related LLM queries. We also examine the resulting jailbreak responses, comparing the generated misinformation to health-related misinformation on Reddit. Our findings add more evidence that LLMs can be effectively used to detect misinformation from both other LLMs and from people, and support a body of work suggesting that with careful design, LLMs can contribute to a healthier overall information ecosystem. 

---
# Beyond Hard Sharing: Efficient Multi-Task Speech-to-Text Modeling with Supervised Mixture of Experts 

**Authors**: Hojun Jin, Eunsoo Hong, Ziwon Hyung, Sungjun Lim, Seungjin Lee, Keunseok Cho  

**Link**: [PDF](https://arxiv.org/pdf/2508.10009)  

**Abstract**: Hard-parameter sharing is a common strategy to train a single model jointly across diverse tasks. However, this often leads to task interference, impeding overall model performance. To address the issue, we propose a simple yet effective Supervised Mixture of Experts (S-MoE). Unlike traditional Mixture of Experts models, S-MoE eliminates the need for training gating functions by utilizing special guiding tokens to route each task to its designated expert. By assigning each task to a separate feedforward network, S-MoE overcomes the limitations of hard-parameter sharing. We further apply S-MoE to a speech-to-text model, enabling the model to process mixed-bandwidth input while jointly performing automatic speech recognition (ASR) and speech translation (ST). Experimental results demonstrate the effectiveness of the proposed S-MoE, achieving a 6.35% relative improvement in Word Error Rate (WER) when applied to both the encoder and decoder. 

---
# Multidimensional classification of posts for online course discussion forum curation 

**Authors**: Antonio Leandro Martins Candido, Jose Everardo Bessa Maia  

**Link**: [PDF](https://arxiv.org/pdf/2508.10008)  

**Abstract**: The automatic curation of discussion forums in online courses requires constant updates, making frequent retraining of Large Language Models (LLMs) a resource-intensive process. To circumvent the need for costly fine-tuning, this paper proposes and evaluates the use of Bayesian fusion. The approach combines the multidimensional classification scores of a pre-trained generic LLM with those of a classifier trained on local data. The performance comparison demonstrated that the proposed fusion improves the results compared to each classifier individually, and is competitive with the LLM fine-tuning approach 

---
# Automated scoring of the Ambiguous Intentions Hostility Questionnaire using fine-tuned large language models 

**Authors**: Y. Lyu, D. Combs, D. Neumann, Y. C. Leong  

**Link**: [PDF](https://arxiv.org/pdf/2508.10007)  

**Abstract**: Hostile attribution bias is the tendency to interpret social interactions as intentionally hostile. The Ambiguous Intentions Hostility Questionnaire (AIHQ) is commonly used to measure hostile attribution bias, and includes open-ended questions where participants describe the perceived intentions behind a negative social situation and how they would respond. While these questions provide insights into the contents of hostile attributions, they require time-intensive scoring by human raters. In this study, we assessed whether large language models can automate the scoring of AIHQ open-ended responses. We used a previously collected dataset in which individuals with traumatic brain injury (TBI) and healthy controls (HC) completed the AIHQ and had their open-ended responses rated by trained human raters. We used half of these responses to fine-tune the two models on human-generated ratings, and tested the fine-tuned models on the remaining half of AIHQ responses. Results showed that model-generated ratings aligned with human ratings for both attributions of hostility and aggression responses, with fine-tuned models showing higher alignment. This alignment was consistent across ambiguous, intentional, and accidental scenario types, and replicated previous findings on group differences in attributions of hostility and aggression responses between TBI and HC groups. The fine-tuned models also generalized well to an independent nonclinical dataset. To support broader adoption, we provide an accessible scoring interface that includes both local and cloud-based options. Together, our findings suggest that large language models can streamline AIHQ scoring in both research and clinical contexts, revealing their potential to facilitate psychological assessments across different populations. 

---
# From Answers to Questions: EQGBench for Evaluating LLMs' Educational Question Generation 

**Authors**: Chengliang Zhou, Mei Wang, Ting Zhang, Qiannan Zhu, Jian Li, Hua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10005)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in mathematical problem-solving. However, the transition from providing answers to generating high-quality educational questions presents significant challenges that remain underexplored. To advance Educational Question Generation (EQG) and facilitate LLMs in generating pedagogically valuable and educationally effective questions, we introduce EQGBench, a comprehensive benchmark specifically designed for evaluating LLMs' performance in Chinese EQG. EQGBench establishes a five-dimensional evaluation framework supported by a dataset of 900 evaluation samples spanning three fundamental middle school disciplines: mathematics, physics, and chemistry. The dataset incorporates user queries with varying knowledge points, difficulty gradients, and question type specifications to simulate realistic educational scenarios. Through systematic evaluation of 46 mainstream large models, we reveal significant room for development in generating questions that reflect educational value and foster students' comprehensive abilities. 

---
# User Perception of Attention Visualizations: Effects on Interpretability Across Evidence-Based Medical Documents 

**Authors**: Andrés Carvallo, Denis Parra, Peter Brusilovsky, Hernan Valdivieso, Gabriel Rada, Ivania Donoso, Vladimir Araujo  

**Link**: [PDF](https://arxiv.org/pdf/2508.10004)  

**Abstract**: The attention mechanism is a core component of the Transformer architecture. Beyond improving performance, attention has been proposed as a mechanism for explainability via attention weights, which are associated with input features (e.g., tokens in a document). In this context, larger attention weights may imply more relevant features for the model's prediction. In evidence-based medicine, such explanations could support physicians' understanding and interaction with AI systems used to categorize biomedical literature. However, there is still no consensus on whether attention weights provide helpful explanations. Moreover, little research has explored how visualizing attention affects its usefulness as an explanation aid. To bridge this gap, we conducted a user study to evaluate whether attention-based explanations support users in biomedical document classification and whether there is a preferred way to visualize them. The study involved medical experts from various disciplines who classified articles based on study design (e.g., systematic reviews, broad synthesis, randomized and non-randomized trials). Our findings show that the Transformer model (XLNet) classified documents accurately; however, the attention weights were not perceived as particularly helpful for explaining the predictions. However, this perception varied significantly depending on how attention was visualized. Contrary to Munzner's principle of visual effectiveness, which favors precise encodings like bar length, users preferred more intuitive formats, such as text brightness or background color. While our results do not confirm the overall utility of attention weights for explanation, they suggest that their perceived helpfulness is influenced by how they are visually presented. 

---
# Semantic Structure in Large Language Model Embeddings 

**Authors**: Austin C. Kozlowski, Callin Dai, Andrei Boutyline  

**Link**: [PDF](https://arxiv.org/pdf/2508.10003)  

**Abstract**: Psychological research consistently finds that human ratings of words across diverse semantic scales can be reduced to a low-dimensional form with relatively little information loss. We find that the semantic associations encoded in the embedding matrices of large language models (LLMs) exhibit a similar structure. We show that the projections of words on semantic directions defined by antonym pairs (e.g. kind - cruel) correlate highly with human ratings, and further find that these projections effectively reduce to a 3-dimensional subspace within LLM embeddings, closely resembling the patterns derived from human survey responses. Moreover, we find that shifting tokens along one semantic direction causes off-target effects on geometrically aligned features proportional to their cosine similarity. These findings suggest that semantic features are entangled within LLMs similarly to how they are interconnected in human language, and a great deal of semantic information, despite its apparent complexity, is surprisingly low-dimensional. Furthermore, accounting for this semantic structure may prove essential for avoiding unintended consequences when steering features. 

---
# HiFACTMix: A Code-Mixed Benchmark and Graph-Aware Model for EvidenceBased Political Claim Verification in Hinglish 

**Authors**: Rakesh Thakur, Sneha Sharma, Gauri Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2508.10001)  

**Abstract**: Fact-checking in code-mixed, low-resource languages such as Hinglish remains an underexplored challenge in natural language processing. Existing fact-verification systems largely focus on high-resource, monolingual settings and fail to generalize to real-world political discourse in linguistically diverse regions like India. Given the widespread use of Hinglish by public figures, particularly political figures, and the growing influence of social media on public opinion, there's a critical need for robust, multilingual and context-aware fact-checking tools. To address this gap a novel benchmark HiFACT dataset is introduced with 1,500 realworld factual claims made by 28 Indian state Chief Ministers in Hinglish, under a highly code-mixed low-resource setting. Each claim is annotated with textual evidence and veracity labels. To evaluate this benchmark, a novel graphaware, retrieval-augmented fact-checking model is proposed that combines multilingual contextual encoding, claim-evidence semantic alignment, evidence graph construction, graph neural reasoning, and natural language explanation generation. Experimental results show that HiFACTMix outperformed accuracy in comparison to state of art multilingual baselines models and provides faithful justifications for its verdicts. This work opens a new direction for multilingual, code-mixed, and politically grounded fact verification research. 

---
# AutoGeTS: Knowledge-based Automated Generation of Text Synthetics for Improving Text Classification 

**Authors**: Chenhao Xue, Yuanzhe Jin, Adrian Carrasco-Revilla, Joyraj Chakraborty, Min Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10000)  

**Abstract**: When developing text classification models for real world applications, one major challenge is the difficulty to collect sufficient data for all text classes. In this work, we address this challenge by utilizing large language models (LLMs) to generate synthetic data and using such data to improve the performance of the models without waiting for more real data to be collected and labelled. As an LLM generates different synthetic data in response to different input examples, we formulate an automated workflow, which searches for input examples that lead to more ``effective'' synthetic data for improving the model concerned. We study three search strategies with an extensive set of experiments, and use experiment results to inform an ensemble algorithm that selects a search strategy according to the characteristics of a class. Our further experiments demonstrate that this ensemble approach is more effective than each individual strategy in our automated workflow for improving classification models using LLMs. 

---
# XFacta: Contemporary, Real-World Dataset and Evaluation for Multimodal Misinformation Detection with Multimodal LLMs 

**Authors**: Yuzhuo Xiao, Zeyu Han, Yuhan Wang, Huaizu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09999)  

**Abstract**: The rapid spread of multimodal misinformation on social media calls for more effective and robust detection methods. Recent advances leveraging multimodal large language models (MLLMs) have shown the potential in addressing this challenge. However, it remains unclear exactly where the bottleneck of existing approaches lies (evidence retrieval v.s. reasoning), hindering the further advances in this field. On the dataset side, existing benchmarks either contain outdated events, leading to evaluation bias due to discrepancies with contemporary social media scenarios as MLLMs can simply memorize these events, or artificially synthetic, failing to reflect real-world misinformation patterns. Additionally, it lacks comprehensive analyses of MLLM-based model design strategies. To address these issues, we introduce XFacta, a contemporary, real-world dataset that is better suited for evaluating MLLM-based detectors. We systematically evaluate various MLLM-based misinformation detection strategies, assessing models across different architectures and scales, as well as benchmarking against existing detection methods. Building on these analyses, we further enable a semi-automatic detection-in-the-loop framework that continuously updates XFacta with new content to maintain its contemporary relevance. Our analysis provides valuable insights and practices for advancing the field of multimodal misinformation detection. The code and data have been released. 

---
# INTIMA: A Benchmark for Human-AI Companionship Behavior 

**Authors**: Lucie-Aimée Kaffee, Giada Pistilli, Yacine Jernite  

**Link**: [PDF](https://arxiv.org/pdf/2508.09998)  

**Abstract**: AI companionship, where users develop emotional bonds with AI systems, has emerged as a significant pattern with positive but also concerning implications. We introduce Interactions and Machine Attachment Benchmark (INTIMA), a benchmark for evaluating companionship behaviors in language models. Drawing from psychological theories and user data, we develop a taxonomy of 31 behaviors across four categories and 368 targeted prompts. Responses to these prompts are evaluated as companionship-reinforcing, boundary-maintaining, or neutral. Applying INTIMA to Gemma-3, Phi-4, o3-mini, and Claude-4 reveals that companionship-reinforcing behaviors remain much more common across all models, though we observe marked differences between models. Different commercial providers prioritize different categories within the more sensitive parts of the benchmark, which is concerning since both appropriate boundary-setting and emotional support matter for user well-being. These findings highlight the need for more consistent approaches to handling emotionally charged interactions. 

---
# Thematic and Task-Based Categorization of K-12 GenAI Usages with Hierarchical Topic Modeling 

**Authors**: Johannes Schneider, Béatrice S. Hasler, Michaela Varrone, Fabian Hoya, Thomas Schroffenegger, Dana-Kristin Mah, Karl Peböck  

**Link**: [PDF](https://arxiv.org/pdf/2508.09997)  

**Abstract**: We analyze anonymous interaction data of minors in class-rooms spanning several months, schools, and subjects employing a novel, simple topic modeling approach. Specifically, we categorize more than 17,000 messages generated by students, teachers, and ChatGPT in two dimensions: content (such as nature and people) and tasks (such as writing and explaining). Our hierarchical categorization done separately for each dimension includes exemplary prompts, and provides both a high-level overview as well as tangible insights. Prior works mostly lack a content or thematic categorization. While task categorizations are more prevalent in education, most have not been supported by real-world data for K-12. In turn, it is not surprising that our analysis yielded a number of novel applications. In deriving these insights, we found that many of the well-established classical and emerging computational methods, i.e., topic modeling, for analysis of large amounts of texts underperform, leading us to directly apply state-of-the-art LLMs with adequate pre-processing to achieve hierarchical topic structures with better human alignment through explicit instructions than prior approaches. Our findings support fellow researchers, teachers and students in enriching the usage of GenAI, while our discussion also highlights a number of concerns and open questions for future research. 

---
# A Transparent Fairness Evaluation Protocol for Open-Source Language Model Benchmarking on the Blockchain 

**Authors**: Hugo Massaroli, Leonardo Iara, Emmanuel Iarussi, Viviana Siless  

**Link**: [PDF](https://arxiv.org/pdf/2508.09993)  

**Abstract**: Large language models (LLMs) are increasingly deployed in realworld applications, yet concerns about their fairness persist especially in highstakes domains like criminal justice, education, healthcare, and finance. This paper introduces transparent evaluation protocol for benchmarking the fairness of opensource LLMs using smart contracts on the Internet Computer Protocol (ICP) blockchain (Foundation, 2023). Our method ensures verifiable, immutable, and reproducible evaluations by executing onchain HTTP requests to hosted Hugging Face endpoints and storing datasets, prompts, and metrics directly onchain. We benchmark the Llama, DeepSeek, and Mistral models on the PISA dataset for academic performance prediction (OECD, 2018), a dataset suitable for fairness evaluation using statistical parity and equal opportunity metrics (Hardt et al., 2016). We also evaluate structured Context Association Metrics derived from the StereoSet dataset (Nadeem et al., 2020) to measure social bias in contextual associations. We further extend our analysis with a multilingual evaluation across English, Spanish, and Portuguese using the Kaleidoscope benchmark (Salazar et al., 2025), revealing cross-linguistic disparities. All code and results are open source, enabling community audits and longitudinal fairness tracking across model versions. 

---
# Bridging AI Innovation and Healthcare Needs: Lessons Learned from Incorporating Modern NLP at The BC Cancer Registry 

**Authors**: Lovedeep Gondara, Gregory Arbour, Raymond Ng, Jonathan Simkin, Shebnum Devji  

**Link**: [PDF](https://arxiv.org/pdf/2508.09991)  

**Abstract**: Automating data extraction from clinical documents offers significant potential to improve efficiency in healthcare settings, yet deploying Natural Language Processing (NLP) solutions presents practical challenges. Drawing upon our experience implementing various NLP models for information extraction and classification tasks at the British Columbia Cancer Registry (BCCR), this paper shares key lessons learned throughout the project lifecycle. We emphasize the critical importance of defining problems based on clear business objectives rather than solely technical accuracy, adopting an iterative approach to development, and fostering deep interdisciplinary collaboration and co-design involving domain experts, end-users, and ML specialists from inception. Further insights highlight the need for pragmatic model selection (including hybrid approaches and simpler methods where appropriate), rigorous attention to data quality (representativeness, drift, annotation), robust error mitigation strategies involving human-in-the-loop validation and ongoing audits, and building organizational AI literacy. These practical considerations, generalizable beyond cancer registries, provide guidance for healthcare organizations seeking to successfully implement AI/NLP solutions to enhance data management processes and ultimately improve patient care and public health outcomes. 

---
# Searching for Privacy Risks in LLM Agents via Simulation 

**Authors**: Yanzhe Zhang, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10880)  

**Abstract**: The widespread deployment of LLM-based agents is likely to introduce a critical privacy threat: malicious agents that proactively engage others in multi-turn interactions to extract sensitive information. These dynamic dialogues enable adaptive attack strategies that can cause severe privacy violations, yet their evolving nature makes it difficult to anticipate and discover sophisticated vulnerabilities manually. To tackle this problem, we present a search-based framework that alternates between improving attacker and defender instructions by simulating privacy-critical agent interactions. Each simulation involves three roles: data subject, data sender, and data recipient. While the data subject's behavior is fixed, the attacker (data recipient) attempts to extract sensitive information from the defender (data sender) through persistent and interactive exchanges. To explore this interaction space efficiently, our search algorithm employs LLMs as optimizers, using parallel search with multiple threads and cross-thread propagation to analyze simulation trajectories and iteratively propose new instructions. Through this process, we find that attack strategies escalate from simple direct requests to sophisticated multi-turn tactics such as impersonation and consent forgery, while defenses advance from rule-based constraints to identity-verification state machines. The discovered attacks and defenses transfer across diverse scenarios and backbone models, demonstrating strong practical utility for building privacy-aware agents. 

---
# Memory-Augmented Transformers: A Systematic Review from Neuroscience Principles to Technical Solutions 

**Authors**: Parsa Omidi, Xingshuai Huang, Axel Laborieux, Bahareh Nikpour, Tianyu Shi, Armaghan Eshaghi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10824)  

**Abstract**: Memory is fundamental to intelligence, enabling learning, reasoning, and adaptability across biological and artificial systems. While Transformer architectures excel at sequence modeling, they face critical limitations in long-range context retention, continual learning, and knowledge integration. This review presents a unified framework bridging neuroscience principles, including dynamic multi-timescale memory, selective attention, and consolidation, with engineering advances in Memory-Augmented Transformers. We organize recent progress through three taxonomic dimensions: functional objectives (context extension, reasoning, knowledge integration, adaptation), memory representations (parameter-encoded, state-based, explicit, hybrid), and integration mechanisms (attention fusion, gated control, associative retrieval). Our analysis of core memory operations (reading, writing, forgetting, and capacity management) reveals a shift from static caches toward adaptive, test-time learning systems. We identify persistent challenges in scalability and interference, alongside emerging solutions including hierarchical buffering and surprise-gated updates. This synthesis provides a roadmap toward cognitively-inspired, lifelong-learning Transformer architectures. 

---
# Pass@k Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning Models 

**Authors**: Zhipeng Chen, Xiaobo Qin, Youbin Wu, Yue Ling, Qinghao Ye, Wayne Xin Zhao, Guang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10751)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR), which typically adopts Pass@1 as the reward, has faced the issues in balancing exploration and exploitation, causing policies to prefer conservative actions, converging to a local optimum. Identifying an appropriate reward metric is therefore crucial. Regarding the prior work, although Pass@k has been used in evaluation, its connection to LLM exploration ability in RLVR remains largely overlooked. To investigate this, we first use Pass@k as the reward to train the policy model (i.e., $\textbf{Pass@k Training}$), and observe the improvement on its exploration ability. Next, we derive an analytical solution for the advantage of Pass@k Training, leading to an efficient and effective process. Building on this, our analysis reveals that exploration and exploitation are not inherently conflicting objectives, while they can mutually enhance each other. Moreover, Pass@k Training with analytical derivation essentially involves directly designing the advantage function. Inspired by this, we preliminarily explore the advantage design for RLVR, showing promising results and highlighting a potential future direction. 

---
# Stabilizing Long-term Multi-turn Reinforcement Learning with Gated Rewards 

**Authors**: Zetian Sun, Dongfang Li, Zhuoen Chen, Yuhuai Qin, Baotian Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10548)  

**Abstract**: Reward sparsity in long-horizon reinforcement learning (RL) tasks remains a significant challenge, while existing outcome-based reward shaping struggles to define meaningful immediate rewards without introducing bias or requiring explicit task decomposition. Alternatively, verification-based reward shaping uses stepwise critics, but misalignment between immediate rewards and long-term objectives can lead to reward hacking and suboptimal policies. In this work, we address this problem in the context of software engineering (SWE) tasks, where multi-turn reasoning and rule-based verification are critical. We introduce the SWE-oriented RL Framework, a unified system supporting multi-turn interaction, docker-based execution, and customizable reward functions. Additionally, we propose Gated Reward Accumulation (G-RA), a novel method that accumulates immediate rewards only when high-level (long-term) rewards meet a predefined threshold, ensuring stable RL optimization. Experiments on SWE-bench Verified and kBench demonstrate that G-RA leads to an increase in completion rates (47.6\% \rightarrow 93.8\% and 22.0\% \rightarrow 86.0\%) and modification rates (19.6\% \rightarrow 23.8\% and 12.0\% \rightarrow 42.0\%), while avoiding policy degradation caused by reward misalignment. Our findings highlight the importance of balanced reward accumulation in long-horizon RL and provide a practical solution. 

---
# Improving Value-based Process Verifier via Low-Cost Variance Reduction 

**Authors**: Zetian Sun, Dongfang Li, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10539)  

**Abstract**: Large language models (LLMs) have achieved remarkable success in a wide range of tasks. However, their reasoning capabilities, particularly in complex domains like mathematics, remain a significant challenge. Value-based process verifiers, which estimate the probability of a partial reasoning chain leading to a correct solution, are a promising approach for improving reasoning. Nevertheless, their effectiveness is often hindered by estimation error in their training annotations, a consequence of the limited number of Monte Carlo (MC) samples feasible due to the high cost of LLM inference. In this paper, we identify that the estimation error primarily arises from high variance rather than bias, and the MC estimator is a Minimum Variance Unbiased Estimator (MVUE). To address the problem, we propose the \textsc{Com}pound \textsc{M}onte \textsc{C}arlo \textsc{S}ampling (ComMCS) method, which constructs an unbiased estimator by linearly combining the MC estimators from the current and subsequent steps. Theoretically, we show that our method leads to a predictable reduction in variance, while maintaining an unbiased estimation without additional LLM inference cost. We also perform empirical experiments on the MATH-500 and GSM8K benchmarks to demonstrate the effectiveness of our method. Notably, ComMCS outperforms regression-based optimization method by 2.8 points, the non-variance-reduced baseline by 2.2 points on MATH-500 on Best-of-32 sampling experiment. 

---
# Diversity First, Quality Later: A Two-Stage Assumption for Language Model Alignment 

**Authors**: Zetian Sun, Dongfang Li, Baotian Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10530)  

**Abstract**: The alignment of language models (LMs) with human preferences is critical for building reliable AI systems. The problem is typically framed as optimizing an LM policy to maximize the expected reward that reflects human preferences. Recently, Direct Preference Optimization (DPO) was proposed as a LM alignment method that directly optimize the policy from static preference data, and further improved by incorporating on-policy sampling (i.e., preference candidates generated during the training loop) for better LM alignment. However, we show on-policy data is not always optimal, with systematic effectiveness difference emerging between static and on-policy preference candidates. For example, on-policy data can result in a 3$\times$ effectiveness compared with static data for Llama-3, and a 0.4$\times$ effectiveness for Zephyr. To explain the phenomenon, we propose the alignment stage assumption, which divides the alignment process into two distinct stages: the preference injection stage, which benefits from diverse data, and the preference fine-tuning stage, which favors high-quality data. Through theoretical and empirical analysis, we characterize these stages and propose an effective algorithm to identify the boundaries between them. We perform experiments on 5 models (Llama, Zephyr, Phi-2, Qwen, Pythia) and 2 alignment methods (DPO, SLiC-HF) to show the generalizability of alignment stage assumption and boundary measurement. 

---
# Reverse Physician-AI Relationship: Full-process Clinical Diagnosis Driven by a Large Language Model 

**Authors**: Shicheng Xu, Xin Huang, Zihao Wei, Liang Pang, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.10492)  

**Abstract**: Full-process clinical diagnosis in the real world encompasses the entire diagnostic workflow that begins with only an ambiguous chief complaint. While artificial intelligence (AI), particularly large language models (LLMs), is transforming clinical diagnosis, its role remains largely as an assistant to physicians. This AI-assisted working pattern makes AI can only answer specific medical questions at certain parts within the diagnostic process, but lack the ability to drive the entire diagnostic process starting from an ambiguous complaint, which still relies heavily on human physicians. This gap limits AI's ability to fully reduce physicians' workload and enhance diagnostic efficiency. To address this, we propose a paradigm shift that reverses the relationship between physicians and AI: repositioning AI as the primary director, with physicians serving as its assistants. So we present DxDirector-7B, an LLM endowed with advanced deep thinking capabilities, enabling it to drive the full-process diagnosis with minimal physician involvement. Furthermore, DxDirector-7B establishes a robust accountability framework for misdiagnoses, delineating responsibility between AI and human physicians. In evaluations across rare, complex, and real-world cases under full-process diagnosis setting, DxDirector-7B not only achieves significant superior diagnostic accuracy but also substantially reduces physician workload than state-of-the-art medical LLMs as well as general-purpose LLMs. Fine-grained analyses across multiple clinical departments and tasks validate its efficacy, with expert evaluations indicating its potential to serve as a viable substitute for medical specialists. These findings mark a new era where AI, traditionally a physicians' assistant, now drives the entire diagnostic process to drastically reduce physicians' workload, indicating an efficient and accurate diagnostic solution. 

---
# CorrectNav: Self-Correction Flywheel Empowers Vision-Language-Action Navigation Model 

**Authors**: Zhuoyuan Yu, Yuxing Long, Zihan Yang, Chengyan Zeng, Hongwei Fan, Jiyao Zhang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.10416)  

**Abstract**: Existing vision-and-language navigation models often deviate from the correct trajectory when executing instructions. However, these models lack effective error correction capability, hindering their recovery from errors. To address this challenge, we propose Self-correction Flywheel, a novel post-training paradigm. Instead of considering the model's error trajectories on the training set as a drawback, our paradigm emphasizes their significance as a valuable data source. We have developed a method to identify deviations in these error trajectories and devised innovative techniques to automatically generate self-correction data for perception and action. These self-correction data serve as fuel to power the model's continued training. The brilliance of our paradigm is revealed when we re-evaluate the model on the training set, uncovering new error trajectories. At this time, the self-correction flywheel begins to spin. Through multiple flywheel iterations, we progressively enhance our monocular RGB-based VLA navigation model CorrectNav. Experiments on R2R-CE and RxR-CE benchmarks show CorrectNav achieves new state-of-the-art success rates of 65.1% and 69.3%, surpassing prior best VLA navigation models by 8.2% and 16.4%. Real robot tests in various indoor and outdoor environments demonstrate \method's superior capability of error correction, dynamic obstacle avoidance, and long instruction following. 

---
# Improving OCR for Historical Texts of Multiple Languages 

**Authors**: Hylke Westerdijk, Ben Blankenborg, Khondoker Ittehadul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2508.10356)  

**Abstract**: This paper presents our methodology and findings from three tasks across Optical Character Recognition (OCR) and Document Layout Analysis using advanced deep learning techniques. First, for the historical Hebrew fragments of the Dead Sea Scrolls, we enhanced our dataset through extensive data augmentation and employed the Kraken and TrOCR models to improve character recognition. In our analysis of 16th to 18th-century meeting resolutions task, we utilized a Convolutional Recurrent Neural Network (CRNN) that integrated DeepLabV3+ for semantic segmentation with a Bidirectional LSTM, incorporating confidence-based pseudolabeling to refine our model. Finally, for modern English handwriting recognition task, we applied a CRNN with a ResNet34 encoder, trained using the Connectionist Temporal Classification (CTC) loss function to effectively capture sequential dependencies. This report offers valuable insights and suggests potential directions for future research. 

---
# Personalized Real-time Jargon Support for Online Meetings 

**Authors**: Yifan Song, Wing Yee Au, Hon Yung Wong, Brian P. Bailey, Tal August  

**Link**: [PDF](https://arxiv.org/pdf/2508.10239)  

**Abstract**: Effective interdisciplinary communication is frequently hindered by domain-specific jargon. To explore the jargon barriers in-depth, we conducted a formative diary study with 16 professionals, revealing critical limitations in current jargon-management strategies during workplace meetings. Based on these insights, we designed ParseJargon, an interactive LLM-powered system providing real-time personalized jargon identification and explanations tailored to users' individual backgrounds. A controlled experiment comparing ParseJargon against baseline (no support) and general-purpose (non-personalized) conditions demonstrated that personalized jargon support significantly enhanced participants' comprehension, engagement, and appreciation of colleagues' work, whereas general-purpose support negatively affected engagement. A follow-up field study validated ParseJargon's usability and practical value in real-time meetings, highlighting both opportunities and limitations for real-world deployment. Our findings contribute insights into designing personalized jargon support tools, with implications for broader interdisciplinary and educational applications. 

---
# Nested-ReFT: Efficient Reinforcement Learning for Large Language Model Fine-Tuning via Off-Policy Rollouts 

**Authors**: Maxime Heuillet, Yufei Cui, Boxing Chen, Audrey Durand, Prasanna Parthasarathi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10123)  

**Abstract**: Advanced reasoning in LLMs on challenging domains like mathematical reasoning can be tackled using verifiable rewards based reinforced fine-tuning (ReFT). In standard ReFT frameworks, a behavior model generates multiple completions with answers per problem, for the answer to be then scored by a reward function. While such RL post-training methods demonstrate significant performance improvements across challenging reasoning domains, the computational cost of generating completions during training with multiple inference steps makes the training cost non-trivial. To address this, we draw inspiration from off-policy RL, and speculative decoding to introduce a novel ReFT framework, dubbed Nested-ReFT, where a subset of layers of the target model acts as the behavior model to generate off-policy completions during training. The behavior model configured with dynamic layer skipping per batch during training decreases the inference cost compared to the standard ReFT frameworks. Our theoretical analysis shows that Nested-ReFT yields unbiased gradient estimates with controlled variance. Our empirical analysis demonstrates improved computational efficiency measured as tokens/sec across multiple math reasoning benchmarks and model sizes. Additionally, we explore three variants of bias mitigation to minimize the off-policyness in the gradient updates that allows for maintaining performance that matches the baseline ReFT performance. 

---
# Amazon Nova AI Challenge -- Trusted AI: Advancing secure, AI-assisted software development 

**Authors**: Sattvik Sahai, Prasoon Goyal, Michael Johnston, Anna Gottardi, Yao Lu, Lucy Hu, Luke Dai, Shaohua Liu, Samyuth Sagi, Hangjie Shi, Desheng Zhang, Lavina Vaz, Leslie Ball, Maureen Murray, Rahul Gupta, Shankar Ananthakrishna  

**Link**: [PDF](https://arxiv.org/pdf/2508.10108)  

**Abstract**: AI systems for software development are rapidly gaining prominence, yet significant challenges remain in ensuring their safety. To address this, Amazon launched the Trusted AI track of the Amazon Nova AI Challenge, a global competition among 10 university teams to drive advances in secure AI. In the challenge, five teams focus on developing automated red teaming bots, while the other five create safe AI assistants. This challenge provides teams with a unique platform to evaluate automated red-teaming and safety alignment methods through head-to-head adversarial tournaments where red teams have multi-turn conversations with the competing AI coding assistants to test their safety alignment. Along with this, the challenge provides teams with a feed of high quality annotated data to fuel iterative improvement. Throughout the challenge, teams developed state-of-the-art techniques, introducing novel approaches in reasoning-based safety alignment, robust model guardrails, multi-turn jail-breaking, and efficient probing of large language models (LLMs). To support these efforts, the Amazon Nova AI Challenge team made substantial scientific and engineering investments, including building a custom baseline coding specialist model for the challenge from scratch, developing a tournament orchestration service, and creating an evaluation harness. This paper outlines the advancements made by university teams and the Amazon Nova AI Challenge team in addressing the safety challenges of AI for software development, highlighting this collaborative effort to raise the bar for AI safety. 

---
# SaraCoder: Orchestrating Semantic and Structural Cues for Profit-Oriented Repository-Level Code Completion 

**Authors**: Xiaohan Chen, Zhongying Pan, Quan Feng, Yu Tian, Shuqun Yang, Mengru Wang, Lina Gong, Yuxia Geng, Piji Li, Xiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10068)  

**Abstract**: Retrieval-augmented generation (RAG) for repository-level code completion commonly relies on superficial text similarity, leading to results plagued by semantic misguidance, redundancy, and homogeneity, while also failing to resolve external symbol ambiguity. To address these challenges, we introduce Saracoder, a Hierarchical Feature-Optimized retrieval framework. Its core Hierarchical Feature Optimization module systematically refines candidates by distilling deep semantic relationships, pruning exact duplicates, assessing structural similarity with a novel graph-based metric that weighs edits by their topological importance, and reranking results to maximize both relevance and diversity. Furthermore, an External-Aware Identifier Disambiguator module accurately resolves cross-file symbol ambiguity via dependency analysis. Extensive experiments on the challenging CrossCodeEval and RepoEval-Updated benchmarks demonstrate that Saracoder significantly outperforms existing baselines across multiple programming languages and models. Our work proves that systematically refining retrieval results across multiple dimensions provides a new paradigm for building more accurate and robust repository-level code completion systems. 

---
# Large Language Models Show Signs of Alignment with Human Neurocognition During Abstract Reasoning 

**Authors**: Christopher Pinier, Sonia Acuña Vargas, Mariia Steeghs-Turchina, Dora Matzke, Claire E. Stevenson, Michael D. Nunez  

**Link**: [PDF](https://arxiv.org/pdf/2508.10057)  

**Abstract**: This study investigates whether large language models (LLMs) mirror human neurocognition during abstract reasoning. We compared the performance and neural representations of human participants with those of eight open-source LLMs on an abstract-pattern-completion task. We leveraged pattern type differences in task performance and in fixation-related potentials (FRPs) as recorded by electroencephalography (EEG) during the task. Our findings indicate that only the largest tested LLMs (~70 billion parameters) achieve human-comparable accuracy, with Qwen-2.5-72B and DeepSeek-R1-70B also showing similarities with the human pattern-specific difficulty profile. Critically, every LLM tested forms representations that distinctly cluster the abstract pattern categories within their intermediate layers, although the strength of this clustering scales with their performance on the task. Moderate positive correlations were observed between the representational geometries of task-optimal LLM layers and human frontal FRPs. These results consistently diverged from comparisons with other EEG measures (response-locked ERPs and resting EEG), suggesting a potential shared representational space for abstract patterns. This indicates that LLMs might mirror human brain mechanisms in abstract reasoning, offering preliminary evidence of shared principles between biological and artificial intelligence. 

---
# Context Misleads LLMs: The Role of Context Filtering in Maintaining Safe Alignment of LLMs 

**Authors**: Jinhwa Kim, Ian G. Harris  

**Link**: [PDF](https://arxiv.org/pdf/2508.10031)  

**Abstract**: While Large Language Models (LLMs) have shown significant advancements in performance, various jailbreak attacks have posed growing safety and ethical risks. Malicious users often exploit adversarial context to deceive LLMs, prompting them to generate responses to harmful queries. In this study, we propose a new defense mechanism called Context Filtering model, an input pre-processing method designed to filter out untrustworthy and unreliable context while identifying the primary prompts containing the real user intent to uncover concealed malicious intent. Given that enhancing the safety of LLMs often compromises their helpfulness, potentially affecting the experience of benign users, our method aims to improve the safety of the LLMs while preserving their original performance. We evaluate the effectiveness of our model in defending against jailbreak attacks through comparative analysis, comparing our approach with state-of-the-art defense mechanisms against six different attacks and assessing the helpfulness of LLMs under these defenses. Our model demonstrates its ability to reduce the Attack Success Rates of jailbreak attacks by up to 88% while maintaining the original LLMs' performance, achieving state-of-the-art Safety and Helpfulness Product results. Notably, our model is a plug-and-play method that can be applied to all LLMs, including both white-box and black-box models, to enhance their safety without requiring any fine-tuning of the models themselves. We will make our model publicly available for research purposes. 

---
# Personalized Product Search Ranking: A Multi-Task Learning Approach with Tabular and Non-Tabular Data 

**Authors**: Lalitesh Morishetti, Abhay Kumar, Jonathan Scott, Kaushiki Nag, Gunjan Sharma, Shanu Vashishtha, Rahul Sridhar, Rohit Chatter, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09636)  

**Abstract**: In this paper, we present a novel model architecture for optimizing personalized product search ranking using a multi-task learning (MTL) framework. Our approach uniquely integrates tabular and non-tabular data, leveraging a pre-trained TinyBERT model for semantic embeddings and a novel sampling technique to capture diverse customer behaviors. We evaluate our model against several baselines, including XGBoost, TabNet, FT-Transformer, DCN-V2, and MMoE, focusing on their ability to handle mixed data types and optimize personalized ranking. Additionally, we propose a scalable relevance labeling mechanism based on click-through rates, click positions, and semantic similarity, offering an alternative to traditional human-annotated labels. Experimental results show that combining non-tabular data with advanced embedding techniques in multi-task learning paradigm significantly enhances model performance. Ablation studies further underscore the benefits of incorporating relevance labels, fine-tuning TinyBERT layers, and TinyBERT query-product embedding interactions. These results demonstrate the effectiveness of our approach in achieving improved personalized product search ranking. 

---
