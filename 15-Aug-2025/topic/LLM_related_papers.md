# Semantic IDs for Joint Generative Search and Recommendation 

**Authors**: Gustavo Penha, Edoardo D'Amico, Marco De Nadai, Enrico Palumbo, Alexandre Tamborrino, Ali Vardasbi, Max Lefarov, Shawn Lin, Timothy Heath, Francesco Fabbri, Hugues Bouchard  

**Link**: [PDF](https://arxiv.org/pdf/2508.10478)  

**Abstract**: Generative models powered by Large Language Models (LLMs) are emerging as a unified solution for powering both recommendation and search tasks. A key design choice in these models is how to represent items, traditionally through unique identifiers (IDs) and more recently with Semantic IDs composed of discrete codes, obtained from embeddings. While task-specific embedding models can improve performance for individual tasks, they may not generalize well in a joint setting. In this paper, we explore how to construct Semantic IDs that perform well both in search and recommendation when using a unified model. We compare a range of strategies to construct Semantic IDs, looking into task-specific and cross-tasks approaches, and also whether each task should have its own semantic ID tokens in a joint search and recommendation generative model. Our results show that using a bi-encoder model fine-tuned on both search and recommendation tasks to obtain item embeddings, followed by the construction of a unified Semantic ID space provides an effective trade-off, enabling strong performance in both tasks. We hope these findings spark follow-up work on generalisable, semantically grounded ID schemes and inform the next wave of unified generative recommender architectures. 

---
# Reflect then Learn: Active Prompting for Information Extraction Guided by Introspective Confusion 

**Authors**: Dong Zhao, Yadong Wang, Xiang Chen, Chenxi Wang, Hongliang Dai, Chuanxing Geng, Shengzhong Zhang, Shaoyuan Li, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10036)  

**Abstract**: Large Language Models (LLMs) show remarkable potential for few-shot information extraction (IE), yet their performance is highly sensitive to the choice of in-context examples. Conventional selection strategies often fail to provide informative guidance, as they overlook a key source of model fallibility: confusion stemming not just from semantic content, but also from the generation of well-structured formats required by IE tasks. To address this, we introduce Active Prompting for Information Extraction (APIE), a novel active prompting framework guided by a principle we term introspective confusion. Our method empowers an LLM to assess its own confusion through a dual-component uncertainty metric that uniquely quantifies both Format Uncertainty (difficulty in generating correct syntax) and Content Uncertainty (inconsistency in extracted semantics). By ranking unlabeled data with this comprehensive score, our framework actively selects the most challenging and informative samples to serve as few-shot exemplars. Extensive experiments on four benchmarks show that our approach consistently outperforms strong baselines, yielding significant improvements in both extraction accuracy and robustness. Our work highlights the critical importance of a fine-grained, dual-level view of model uncertainty when it comes to building effective and reliable structured generation systems. 

---
# RTTC: Reward-Guided Collaborative Test-Time Compute 

**Authors**: J. Pablo Muñoz, Jinjie Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10024)  

**Abstract**: Test-Time Compute (TTC) has emerged as a powerful paradigm for enhancing the performance of Large Language Models (LLMs) at inference, leveraging strategies such as Test-Time Training (TTT) and Retrieval-Augmented Generation (RAG). However, the optimal adaptation strategy varies across queries, and indiscriminate application of TTC strategy incurs substantial computational overhead. In this work, we introduce Reward-Guided Test-Time Compute (RTTC), a novel framework that adaptively selects the most effective TTC strategy for each query via a pretrained reward model, maximizing downstream accuracy across diverse domains and tasks. RTTC operates in a distributed server-client architecture, retrieving relevant samples from a remote knowledge base and applying RAG or lightweight fine-tuning on client devices only when necessary. To further mitigate redundant computation, we propose Query-State Caching, which enables the efficient reuse of historical query states at both retrieval and adaptation levels. Extensive experiments across multiple LLMs and benchmarks demonstrate that RTTC consistently achieves superior accuracy compared to vanilla RAG or TTT, validating the necessity of adaptive, reward-guided TTC selection and the potential of RTTC for scalable, high-performance language model adaptation. 

---
# Bridging Modality Gaps in e-Commerce Products via Vision-Language Alignment 

**Authors**: Yipeng Zhang, Hongju Yu, Aritra Mandal, Canran Xu, Qunzhi Zhou, Zhe Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10116)  

**Abstract**: Item information, such as titles and attributes, is essential for effective user engagement in e-commerce. However, manual or semi-manual entry of structured item specifics often produces inconsistent quality, errors, and slow turnaround, especially for Customer-to-Customer sellers. Generating accurate descriptions directly from item images offers a promising alternative. Existing retrieval-based solutions address some of these issues but often miss fine-grained visual details and struggle with niche or specialized categories.
We propose Optimized Preference-Based AI for Listings (OPAL), a framework for generating schema-compliant, high-quality item descriptions from images using a fine-tuned multimodal large language model (MLLM). OPAL addresses key challenges in multimodal e-commerce applications, including bridging modality gaps and capturing detailed contextual information. It introduces two data refinement methods: MLLM-Assisted Conformity Enhancement, which ensures alignment with structured schema requirements, and LLM-Assisted Contextual Understanding, which improves the capture of nuanced and fine-grained information from visual inputs.
OPAL uses visual instruction tuning combined with direct preference optimization to fine-tune the MLLM, reducing hallucinations and improving robustness across different backbone architectures. We evaluate OPAL on real-world e-commerce datasets, showing that it consistently outperforms baseline methods in both description quality and schema completion rates. These results demonstrate that OPAL effectively bridges the gap between visual and textual modalities, delivering richer, more accurate, and more consistent item descriptions. This work advances automated listing optimization and supports scalable, high-quality content generation in e-commerce platforms. 

---
# Learning from Natural Language Feedback for Personalized Question Answering 

**Authors**: Alireza Salemi, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2508.10695)  

**Abstract**: Personalization is crucial for enhancing both the effectiveness and user satisfaction of language technologies, particularly in information-seeking tasks like question answering. Current approaches for personalizing large language models (LLMs) often rely on retrieval-augmented generation (RAG), followed by reinforcement learning with scalar reward signals to teach models how to use retrieved personal context. We believe that these scalar rewards sometimes provide weak, non-instructive feedback, limiting learning efficiency and personalization quality. We introduce VAC, a novel framework for personalized response generation that replaces scalar rewards with natural language feedback (NLF) that are generated conditioned on the user profiles and the question narratives. NLF serves as a rich and actionable supervision signal, allowing the policy model to iteratively refine its outputs and internalize effective personalization strategies. Training alternates between optimizing the feedback model and fine-tuning the policy model on the improved responses, resulting in a policy model that no longer requires feedback at inference. Evaluation on the LaMP-QA benchmark that consists of three diverse domains demonstrates consistent and significant improvements over the state-of-the-art results. Human evaluations further confirm the superior quality of the generated responses. These results demonstrate that NLF provides more effective signals for optimizing personalized question answering. 

---
# Reinforced Language Models for Sequential Decision Making 

**Authors**: Jim Dilkes, Vahid Yazdanpanah, Sebastian Stein  

**Link**: [PDF](https://arxiv.org/pdf/2508.10839)  

**Abstract**: Large Language Models (LLMs) show potential as sequential decision-making agents, but their application is often limited due to a reliance on large, computationally expensive models. This creates a need to improve smaller models, yet existing post-training methods are designed for single-turn interactions and cannot handle credit assignment in multi-step agentic tasks. To address this, we introduce Multi-Step Group-Relative Policy Optimization (MS-GRPO), a new algorithm for post-training LLM agents, grounded in formal Text-Mediated Stochastic Game (TSMG) and Language-Agent Policy (LAP) frameworks. For credit assignment, MS-GRPO attributes the entire cumulative episode reward to each individual episode step. We supplement this algorithm with a novel absolute-advantage-weighted episode sampling strategy that we show improves training performance. We evaluate our approach by post-training a 3-billion parameter model on Snake and Frozen Lake. Our experiments demonstrate that the method is effective in improving decision-making performance: our post-trained 3B parameter model outperforms a 72B parameter baseline by 50% on the Frozen Lake task. This work demonstrates that targeted post-training is a practical and efficient alternative to relying on model scale for creating sequential decision-making agents using LLMs. 

---
# SSRL: Self-Search Reinforcement Learning 

**Authors**: Yuchen Fan, Kaiyan Zhang, Heng Zhou, Yuxin Zuo, Yanxu Chen, Yu Fu, Xinwei Long, Xuekai Zhu, Che Jiang, Yuchen Zhang, Li Kang, Gang Chen, Cheng Huang, Zhizhou He, Bingning Wang, Lei Bai, Ning Ding, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.10874)  

**Abstract**: We investigate the potential of large language models (LLMs) to serve as efficient simulators for agentic search tasks in reinforcement learning (RL), thereby reducing dependence on costly interactions with external search engines. To this end, we first quantify the intrinsic search capability of LLMs via structured prompting and repeated sampling, which we term Self-Search. Our results reveal that LLMs exhibit strong scaling behavior with respect to the inference budget, achieving high pass@k on question-answering benchmarks, including the challenging BrowseComp task. Building on these observations, we introduce Self-Search RL (SSRL), which enhances LLMs' Self-Search capability through format-based and rule-based rewards. SSRL enables models to iteratively refine their knowledge utilization internally, without requiring access to external tools. Empirical evaluations demonstrate that SSRL-trained policy models provide a cost-effective and stable environment for search-driven RL training, reducing reliance on external search engines and facilitating robust sim-to-real transfer. We draw the following conclusions: 1) LLMs possess world knowledge that can be effectively elicited to achieve high performance; 2) SSRL demonstrates the potential of leveraging internal knowledge to reduce hallucination; 3) SSRL-trained models integrate seamlessly with external search engines without additional effort. Our findings highlight the potential of LLMs to support more scalable RL agent training. 

---
# Beyond "Not Novel Enough": Enriching Scholarly Critique with LLM-Assisted Feedback 

**Authors**: Osama Mohammed Afzal, Preslav Nakov, Tom Hope, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2508.10795)  

**Abstract**: Novelty assessment is a central yet understudied aspect of peer review, particularly in high volume fields like NLP where reviewer capacity is increasingly strained. We present a structured approach for automated novelty evaluation that models expert reviewer behavior through three stages: content extraction from submissions, retrieval and synthesis of related work, and structured comparison for evidence based assessment. Our method is informed by a large scale analysis of human written novelty reviews and captures key patterns such as independent claim verification and contextual reasoning. Evaluated on 182 ICLR 2025 submissions with human annotated reviewer novelty assessments, the approach achieves 86.5% alignment with human reasoning and 75.3% agreement on novelty conclusions - substantially outperforming existing LLM based baselines. The method produces detailed, literature aware analyses and improves consistency over ad hoc reviewer judgments. These results highlight the potential for structured LLM assisted approaches to support more rigorous and transparent peer review without displacing human expertise. Data and code are made available. 

---
# Computational Economics in Large Language Models: Exploring Model Behavior and Incentive Design under Resource Constraints 

**Authors**: Sandeep Reddy, Kabir Khan, Rohit Patil, Ananya Chakraborty, Faizan A. Khan, Swati Kulkarni, Arjun Verma, Neha Singh  

**Link**: [PDF](https://arxiv.org/pdf/2508.10426)  

**Abstract**: Large language models (LLMs) are limited by substantial computational cost. We introduce a "computational economics" framework that treats an LLM as an internal economy of resource-constrained agents (attention heads and neuron blocks) that must allocate scarce computation to maximize task utility. First, we show empirically that when computation is scarce, standard LLMs reallocate attention toward high-value tokens while preserving accuracy. Building on this observation, we propose an incentive-driven training paradigm that augments the task loss with a differentiable computation cost term, encouraging sparse and efficient activations. On GLUE (MNLI, STS-B, CoLA) and WikiText-103, the method yields a family of models that trace a Pareto frontier and consistently dominate post-hoc pruning; for a similar accuracy we obtain roughly a forty percent reduction in FLOPS and lower latency, together with more interpretable attention patterns. These results indicate that economic principles offer a principled route to designing efficient, adaptive, and more transparent LLMs under strict resource constraints. 

---
# eDIF: A European Deep Inference Fabric for Remote Interpretability of LLM 

**Authors**: Irma Heithoff. Marc Guggenberger, Sandra Kalogiannis, Susanne Mayer, Fabian Maag, Sigurd Schacht, Carsten Lanquillon  

**Link**: [PDF](https://arxiv.org/pdf/2508.10553)  

**Abstract**: This paper presents a feasibility study on the deployment of a European Deep Inference Fabric (eDIF), an NDIF-compatible infrastructure designed to support mechanistic interpretability research on large language models. The need for widespread accessibility of LLM interpretability infrastructure in Europe drives this initiative to democratize advanced model analysis capabilities for the research community. The project introduces a GPU-based cluster hosted at Ansbach University of Applied Sciences and interconnected with partner institutions, enabling remote model inspection via the NNsight API. A structured pilot study involving 16 researchers from across Europe evaluated the platform's technical performance, usability, and scientific utility. Users conducted interventions such as activation patching, causal tracing, and representation analysis on models including GPT-2 and DeepSeek-R1-70B. The study revealed a gradual increase in user engagement, stable platform performance throughout, and a positive reception of the remote experimentation capabilities. It also marked the starting point for building a user community around the platform. Identified limitations such as prolonged download durations for activation data as well as intermittent execution interruptions are addressed in the roadmap for future development. This initiative marks a significant step towards widespread accessibility of LLM interpretability infrastructure in Europe and lays the groundwork for broader deployment, expanded tooling, and sustained community collaboration in mechanistic interpretability research. 

---
# Psyche-R1: Towards Reliable Psychological LLMs through Unified Empathy, Expertise, and Reasoning 

**Authors**: Chongyuan Dai, Jinpeng Hu, Hongchang Shi, Zhuo Li, Xun Yang, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10848)  

**Abstract**: Amidst a shortage of qualified mental health professionals, the integration of large language models (LLMs) into psychological applications offers a promising way to alleviate the growing burden of mental health disorders. Recent reasoning-augmented LLMs have achieved remarkable performance in mathematics and programming, while research in the psychological domain has predominantly emphasized emotional support and empathetic dialogue, with limited attention to reasoning mechanisms that are beneficial to generating reliable responses. Therefore, in this paper, we propose Psyche-R1, the first Chinese psychological LLM that jointly integrates empathy, psychological expertise, and reasoning, built upon a novel data curation pipeline. Specifically, we design a comprehensive data synthesis pipeline that produces over 75k high-quality psychological questions paired with detailed rationales, generated through chain-of-thought (CoT) reasoning and iterative prompt-rationale optimization, along with 73k empathetic dialogues. Subsequently, we employ a hybrid training strategy wherein challenging samples are identified through a multi-LLM cross-selection strategy for group relative policy optimization (GRPO) to improve reasoning ability, while the remaining data is used for supervised fine-tuning (SFT) to enhance empathetic response generation and psychological domain knowledge. Extensive experiment results demonstrate the effectiveness of the Psyche-R1 across several psychological benchmarks, where our 7B Psyche-R1 achieves comparable results to 671B DeepSeek-R1. 

---
# When Language Overrules: Revealing Text Dominance in Multimodal Large Language Models 

**Authors**: Huyu Wu, Meng Tang, Xinhan Zheng, Haiyun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10552)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities across a diverse range of multimodal tasks. However, these models suffer from a core problem known as text dominance: they depend heavily on text for their inference, while underutilizing other modalities. While prior work has acknowledged this phenomenon in vision-language tasks, often attributing it to data biases or model architectures. In this paper, we conduct the first systematic investigation of text dominance across diverse data modalities, including images, videos, audio, time-series, and graphs. To measure this imbalance, we propose two evaluation metrics: the Modality Dominance Index (MDI) and the Attention Efficiency Index (AEI). Our comprehensive analysis reveals that text dominance is both significant and pervasive across all tested modalities. Our in-depth analysis identifies three underlying causes: attention dilution from severe token redundancy in non-textual modalities, the influence of fusion architecture design, and task formulations that implicitly favor textual inputs. Furthermore, we propose a simple token compression method that effectively rebalances model attention. Applying this method to LLaVA-7B, for instance, drastically reduces its MDI from 10.23 to a well-balanced value of 0.86. Our analysis and methodological framework offer a foundation for the development of more equitable and comprehensive multimodal language models. 

---
# Jailbreaking Commercial Black-Box LLMs with Explicitly Harmful Prompts 

**Authors**: Chiyu Zhang, Lu Zhou, Xiaogang Xu, Jiafei Wu, Liming Fang, Zhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10390)  

**Abstract**: Evaluating jailbreak attacks is challenging when prompts are not overtly harmful or fail to induce harmful outputs. Unfortunately, many existing red-teaming datasets contain such unsuitable prompts. To evaluate attacks accurately, these datasets need to be assessed and cleaned for maliciousness. However, existing malicious content detection methods rely on either manual annotation, which is labor-intensive, or large language models (LLMs), which have inconsistent accuracy in harmful types. To balance accuracy and efficiency, we propose a hybrid evaluation framework named MDH (Malicious content Detection based on LLMs with Human assistance) that combines LLM-based annotation with minimal human oversight, and apply it to dataset cleaning and detection of jailbroken responses. Furthermore, we find that well-crafted developer messages can significantly boost jailbreak success, leading us to propose two new strategies: D-Attack, which leverages context simulation, and DH-CoT, which incorporates hijacked chains of thought. The Codes, datasets, judgements, and detection results will be released in github repository: this https URL. 

---
# Evaluating LLMs on Chinese Idiom Translation 

**Authors**: Cai Yang, Yao Dou, David Heineman, Xiaofeng Wu, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10421)  

**Abstract**: Idioms, whose figurative meanings usually differ from their literal interpretations, are common in everyday language, especially in Chinese, where they often contain historical references and follow specific structural patterns. Despite recent progress in machine translation with large language models, little is known about Chinese idiom translation. In this work, we introduce IdiomEval, a framework with a comprehensive error taxonomy for Chinese idiom translation. We annotate 900 translation pairs from nine modern systems, including GPT-4o and Google Translate, across four domains: web, news, Wikipedia, and social media. We find these systems fail at idiom translation, producing incorrect, literal, partial, or even missing translations. The best-performing system, GPT-4, makes errors in 28% of cases. We also find that existing evaluation metrics measure idiom quality poorly with Pearson correlation below 0.48 with human ratings. We thus develop improved models that achieve F$_1$ scores of 0.68 for detecting idiom translation errors. 

---
# Improving Generative Cross-lingual Aspect-Based Sentiment Analysis with Constrained Decoding 

**Authors**: Jakub Šmíd, Pavel Přibáň, Pavel Král  

**Link**: [PDF](https://arxiv.org/pdf/2508.10369)  

**Abstract**: While aspect-based sentiment analysis (ABSA) has made substantial progress, challenges remain for low-resource languages, which are often overlooked in favour of English. Current cross-lingual ABSA approaches focus on limited, less complex tasks and often rely on external translation tools. This paper introduces a novel approach using constrained decoding with sequence-to-sequence models, eliminating the need for unreliable translation tools and improving cross-lingual performance by 5\% on average for the most complex task. The proposed method also supports multi-tasking, which enables solving multiple ABSA tasks with a single model, with constrained decoding boosting results by more than 10\%.
We evaluate our approach across seven languages and six ABSA tasks, surpassing state-of-the-art methods and setting new benchmarks for previously unexplored tasks. Additionally, we assess large language models (LLMs) in zero-shot, few-shot, and fine-tuning scenarios. While LLMs perform poorly in zero-shot and few-shot settings, fine-tuning achieves competitive results compared to smaller multilingual models, albeit at the cost of longer training and inference times.
We provide practical recommendations for real-world applications, enhancing the understanding of cross-lingual ABSA methodologies. This study offers valuable insights into the strengths and limitations of cross-lingual ABSA approaches, advancing the state-of-the-art in this challenging research domain. 

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
# Advancing Cross-lingual Aspect-Based Sentiment Analysis with LLMs and Constrained Decoding for Sequence-to-Sequence Models 

**Authors**: Jakub Šmíd, Pavel Přibáň, Pavel Král  

**Link**: [PDF](https://arxiv.org/pdf/2508.10366)  

**Abstract**: Aspect-based sentiment analysis (ABSA) has made significant strides, yet challenges remain for low-resource languages due to the predominant focus on English. Current cross-lingual ABSA studies often centre on simpler tasks and rely heavily on external translation tools. In this paper, we present a novel sequence-to-sequence method for compound ABSA tasks that eliminates the need for such tools. Our approach, which uses constrained decoding, improves cross-lingual ABSA performance by up to 10\%. This method broadens the scope of cross-lingual ABSA, enabling it to handle more complex tasks and providing a practical, efficient alternative to translation-dependent techniques. Furthermore, we compare our approach with large language models (LLMs) and show that while fine-tuned multilingual LLMs can achieve comparable results, English-centric LLMs struggle with these tasks. 

---
# Inductive Bias Extraction and Matching for LLM Prompts 

**Authors**: Christian M. Angel, Francis Ferraro  

**Link**: [PDF](https://arxiv.org/pdf/2508.10295)  

**Abstract**: The active research topic of prompt engineering makes it evident that LLMs are sensitive to small changes in prompt wording. A portion of this can be ascribed to the inductive bias that is present in the LLM. By using an LLM's output as a portion of its prompt, we can more easily create satisfactory wording for prompts. This has the effect of creating a prompt that matches the inductive bias in model. Empirically, we show that using this Inductive Bias Extraction and Matching strategy improves LLM Likert ratings used for classification by up to 19% and LLM Likert ratings used for ranking by up to 27%. 

---
# Large Language Models for Summarizing Czech Historical Documents and Beyond 

**Authors**: Václav Tran, Jakub Šmíd, Jiří Martínek, Ladislav Lenc, Pavel Král  

**Link**: [PDF](https://arxiv.org/pdf/2508.10368)  

**Abstract**: Text summarization is the task of shortening a larger body of text into a concise version while retaining its essential meaning and key information. While summarization has been significantly explored in English and other high-resource languages, Czech text summarization, particularly for historical documents, remains underexplored due to linguistic complexities and a scarcity of annotated datasets. Large language models such as Mistral and mT5 have demonstrated excellent results on many natural language processing tasks and languages. Therefore, we employ these models for Czech summarization, resulting in two key contributions: (1) achieving new state-of-the-art results on the modern Czech summarization dataset SumeCzech using these advanced models, and (2) introducing a novel dataset called Posel od Čerchova for summarization of historical Czech documents with baseline results. Together, these contributions provide a great potential for advancing Czech text summarization and open new avenues for research in Czech historical text processing. 

---
# LaajMeter: A Framework for LaaJ Evaluation 

**Authors**: Gal Amram, Eitan Farchi, Shmulik Froimovich, Raviv Gal, Avi Ziv  

**Link**: [PDF](https://arxiv.org/pdf/2508.10161)  

**Abstract**: Large Language Models (LLMs) are increasingly used as evaluators in natural language processing tasks, a paradigm known as LLM-as-a-Judge (LaaJ). While effective in general domains, LaaJs pose significant challenges in domain-specific contexts, where annotated data is scarce and expert evaluation is costly. In such cases, meta-evaluation is often performed using metrics that have not been validated for the specific domain in which they are applied. As a result, it becomes difficult to determine which metrics effectively identify LaaJ quality, and further, what threshold indicates sufficient evaluator performance. In this work, we introduce LaaJMeter, a simulation-based framework for controlled meta-evaluation of LaaJs. LaaJMeter enables engineers to generate synthetic data representing virtual models and judges, allowing systematic analysis of evaluation metrics under realistic conditions. This helps practitioners validate and refine LaaJs for specific evaluation tasks: they can test whether their metrics correctly distinguish between better and worse (virtual) LaaJs, and estimate appropriate thresholds for evaluator adequacy.
We demonstrate the utility of LaaJMeter in a code translation task involving a legacy programming language, showing how different metrics vary in sensitivity to evaluator quality. Our results highlight the limitations of common metrics and the importance of principled metric selection. LaaJMeter provides a scalable and extensible solution for assessing LaaJs in low-resource settings, contributing to the broader effort to ensure trustworthy and reproducible evaluation in NLP. 

---
# Making Qwen3 Think in Korean with Reinforcement Learning 

**Authors**: Jungyup Lee, Jemin Kim, Sang Park, SeungJae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.10355)  

**Abstract**: We present a two-stage fine-tuning approach to make the large language model Qwen3 14B "think" natively in Korean. In the first stage, supervised fine-tuning (SFT) on a high-quality Korean reasoning dataset establishes a strong foundation in Korean logical reasoning, yielding notable improvements in Korean-language tasks and even some gains in general reasoning ability. In the second stage, we employ reinforcement learning with a customized Group Relative Policy Optimization (GRPO) algorithm to further enhance both Korean reasoning alignment and overall problem-solving performance. We address critical stability challenges in GRPO training - such as reward hacking and policy collapse - by introducing an oracle judge model that calibrates the reward signal. Our approach achieves stable learning (avoiding the collapse observed in naive GRPO) and leads to steady, incremental performance gains. The final RL-tuned model demonstrates substantially improved results on advanced reasoning benchmarks (particularly math and coding tasks) while maintaining knowledge and language proficiency, successfully conducting its internal chain-of-thought entirely in Korean. 

---
# mSCoRe: a $M$ultilingual and Scalable Benchmark for $S$kill-based $Co$mmonsense $Re$asoning 

**Authors**: Nghia Trung Ngo, Franck Dernoncourt, Thien Huu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10137)  

**Abstract**: Recent advancements in reasoning-reinforced Large Language Models (LLMs) have shown remarkable capabilities in complex reasoning tasks. However, the mechanism underlying their utilization of different human reasoning skills remains poorly investigated, especially for multilingual commonsense reasoning that involves everyday knowledge across different languages and cultures. To address this gap, we propose a \textbf{M}ultilingual and Scalable Benchmark for \textbf{S}kill-based \textbf{Co}mmonsense \textbf{Re}asoning (\textbf{mSCoRe}). Our benchmark incorporates three key components that are designed to systematically evaluate LLM's reasoning capabilities, including: (1) a novel taxonomy of reasoning skills that enables fine-grained analysis of models' reasoning processes, (2) a robust data synthesis pipeline tailored specifically for commonsense reasoning evaluation, and (3) a complexity scaling framework allowing task difficulty to scale dynamically alongside future improvements in LLM abilities. Extensive experiments on eights state-of-the-art LLMs of varying sizes and training approaches demonstrate that \textbf{mSCoRe} remains significantly challenging for current models, particularly at higher complexity levels. Our results reveal the limitations of such reasoning-reinforced models when confronted with nuanced multilingual general and cultural commonsense. We further provide detailed analysis on the models' reasoning processes, suggesting future directions for improving multilingual commonsense reasoning capabilities. 

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
# SABER: Switchable and Balanced Training for Efficient LLM Reasoning 

**Authors**: Kai Zhao, Yanjun Zhao, Jiaming Song, Shien He, Lusheng Zhang, Qiang Zhang, Tianjiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.10026)  

**Abstract**: Large language models (LLMs) empowered by chain-of-thought reasoning have achieved impressive accuracy on complex tasks but suffer from excessive inference costs and latency when applied uniformly to all problems. We propose SABER (Switchable and Balanced Training for Efficient LLM Reasoning), a reinforcement learning framework that endows LLMs with user-controllable, token-budgeted reasoning. SABER first profiles each training example's base-model thinking token usage and assigns it to one of the predefined budget tiers. During fine-tuning, the model is guided by system prompts and length-aware rewards to respect its assigned budget. In parallel, we incorporate no-think examples to ensure the model remains reliable even when explicit reasoning is turned off. SABER further supports four discrete inference modes - NoThink, FastThink, CoreThink, and DeepThink, enabling flexible trade-offs between latency and reasoning depth. Extensive evaluations on math reasoning (MATH, GSM8K), code generation (MBPP), and logical reasoning (LiveBench-Reasoning) demonstrate that SABER achieves high accuracy under tight budgets, graceful degradation, and effective cross-scale and cross-domain generalization. In particular, SABER-FastThink cuts reasoning length by 65.4% and yields a 3.6% accuracy gain compared with the base model on the MATH benchmark. 

---
# PREF: Reference-Free Evaluation of Personalised Text Generation in LLMs 

**Authors**: Xiao Fu, Hossein A. Rahmani, Bin Wu, Jerome Ramos, Emine Yilmaz, Aldo Lipani  

**Link**: [PDF](https://arxiv.org/pdf/2508.10028)  

**Abstract**: Personalised text generation is essential for user-centric information systems, yet most evaluation methods overlook the individuality of users. We introduce \textbf{PREF}, a \textbf{P}ersonalised \textbf{R}eference-free \textbf{E}valuation \textbf{F}ramework that jointly measures general output quality and user-specific alignment without requiring gold personalised references. PREF operates in a three-step pipeline: (1) a coverage stage uses a large language model (LLM) to generate a comprehensive, query-specific guideline covering universal criteria such as factuality, coherence, and completeness; (2) a preference stage re-ranks and selectively augments these factors using the target user's profile, stated or inferred preferences, and context, producing a personalised evaluation rubric; and (3) a scoring stage applies an LLM judge to rate candidate answers against this rubric, ensuring baseline adequacy while capturing subjective priorities. This separation of coverage from preference improves robustness, transparency, and reusability, and allows smaller models to approximate the personalised quality of larger ones. Experiments on the PrefEval benchmark, including implicit preference-following tasks, show that PREF achieves higher accuracy, better calibration, and closer alignment with human judgments than strong baselines. By enabling scalable, interpretable, and user-aligned evaluation, PREF lays the groundwork for more reliable assessment and development of personalised language generation systems. 

---
# Detecting and explaining postpartum depression in real-time with generative artificial intelligence 

**Authors**: Silvia García-Méndez, Francisco de Arriba-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2508.10025)  

**Abstract**: Among the many challenges mothers undergo after childbirth, postpartum depression (PPD) is a severe condition that significantly impacts their mental and physical well-being. Consequently, the rapid detection of ppd and their associated risk factors is critical for in-time assessment and intervention through specialized prevention procedures. Accordingly, this work addresses the need to help practitioners make decisions with the latest technological advancements to enable real-time screening and treatment recommendations. Mainly, our work contributes to an intelligent PPD screening system that combines Natural Language Processing, Machine Learning (ML), and Large Language Models (LLMs) towards an affordable, real-time, and non-invasive free speech analysis. Moreover, it addresses the black box problem since the predictions are described to the end users thanks to the combination of LLMs with interpretable ml models (i.e., tree-based algorithms) using feature importance and natural language. The results obtained are 90 % on ppd detection for all evaluation metrics, outperforming the competing solutions in the literature. Ultimately, our solution contributes to the rapid detection of PPD and their associated risk factors, critical for in-time and proper assessment and intervention. 

---
# Efficient Forward-Only Data Valuation for Pretrained LLMs and VLMs 

**Authors**: Wenlong Deng, Jiaming Zhang, Qi Zeng, Christos Thrampoulidis, Boying Gong, Xiaoxiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.10180)  

**Abstract**: Quantifying the influence of individual training samples is essential for enhancing the transparency and accountability of large language models (LLMs) and vision-language models (VLMs). However, existing data valuation methods often rely on Hessian information or model retraining, making them computationally prohibitive for billion-parameter models. In this work, we introduce For-Value, a forward-only data valuation framework that enables scalable and efficient influence estimation for both LLMs and VLMs. By leveraging the rich representations of modern foundation models, For-Value computes influence scores using a simple closed-form expression based solely on a single forward pass, thereby eliminating the need for costly gradient computations. Our theoretical analysis demonstrates that For-Value accurately estimates per-sample influence by capturing alignment in hidden representations and prediction errors between training and validation samples. Extensive experiments show that For-Value matches or outperforms gradient-based baselines in identifying impactful fine-tuning examples and effectively detecting mislabeled data. 

---
# Multi-Turn Puzzles: Evaluating Interactive Reasoning and Strategic Dialogue in LLMs 

**Authors**: Kartikeya Badola, Jonathan Simon, Arian Hosseini, Sara Marie Mc Carthy, Tsendsuren Munkhdalai, Abhimanyu Goyal, Tomáš Kočiský, Shyam Upadhyay, Bahare Fatemi, Mehran Kazemi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10142)  

**Abstract**: Large language models (LLMs) excel at solving problems with clear and complete statements, but often struggle with nuanced environments or interactive tasks which are common in most real-world scenarios. This highlights the critical need for developing LLMs that can effectively engage in logically consistent multi-turn dialogue, seek information and reason with incomplete data. To this end, we introduce a novel benchmark comprising a suite of multi-turn tasks each designed to test specific reasoning, interactive dialogue, and information-seeking abilities. These tasks have deterministic scoring mechanisms, thus eliminating the need for human intervention. Evaluating frontier models on our benchmark reveals significant headroom. Our analysis shows that most errors emerge from poor instruction following, reasoning failures, and poor planning. This benchmark provides valuable insights into the strengths and weaknesses of current LLMs in handling complex, interactive scenarios and offers a robust platform for future research aimed at improving these critical capabilities. 

---
# The Cost of Thinking: Increased Jailbreak Risk in Large Language Models 

**Authors**: Fan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10032)  

**Abstract**: Thinking mode has always been regarded as one of the most valuable modes in LLMs. However, we uncover a surprising and previously overlooked phenomenon: LLMs with thinking mode are more easily broken by Jailbreak attack. We evaluate 9 LLMs on AdvBench and HarmBench and find that the success rate of attacking thinking mode in LLMs is almost higher than that of non-thinking mode. Through large numbers of sample studies, it is found that for educational purposes and excessively long thinking lengths are the characteristics of successfully attacked data, and LLMs also give harmful answers when they mostly know that the questions are harmful. In order to alleviate the above problems, this paper proposes a method of safe thinking intervention for LLMs, which explicitly guides the internal thinking processes of LLMs by adding "specific thinking tokens" of LLMs to the prompt. The results demonstrate that the safe thinking intervention can significantly reduce the attack success rate of LLMs with thinking mode. 

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
# Evaluation of GPT-based large language generative AI models as study aids for the national licensure examination for registered dietitians in Japan 

**Authors**: Yuta Nagamori, Mikoto Kosai, Yuji Kawai, Haruka Marumo, Misaki Shibuya, Tatsuya Negishi, Masaki Imanishi, Yasumasa Ikeda, Koichiro Tsuchiya, Asuka Sawai, Licht Miyamoto  

**Link**: [PDF](https://arxiv.org/pdf/2508.10011)  

**Abstract**: Generative artificial intelligence (AI) based on large language models (LLMs), such as ChatGPT, has demonstrated remarkable progress across various professional fields, including medicine and education. However, their performance in nutritional education, especially in Japanese national licensure examination for registered dietitians, remains underexplored. This study aimed to evaluate the potential of current LLM-based generative AI models as study aids for nutrition students. Questions from the Japanese national examination for registered dietitians were used as prompts for ChatGPT and three Bing models (Precise, Creative, Balanced), based on GPT-3.5 and GPT-4. Each question was entered into independent sessions, and model responses were analyzed for accuracy, consistency, and response time. Additional prompt engineering, including role assignment, was tested to assess potential performance improvements. Bing-Precise (66.2%) and Bing-Creative (61.4%) surpassed the passing threshold (60%), while Bing-Balanced (43.3%) and ChatGPT (42.8%) did not. Bing-Precise and Bing-Creative generally outperformed others across subject fields except Nutrition Education, where all models underperformed. None of the models consistently provided the same correct responses across repeated attempts, highlighting limitations in answer stability. ChatGPT showed greater consistency in response patterns but lower accuracy. Prompt engineering had minimal effect, except for modest improvement when correct answers and explanations were explicitly provided. While some generative AI models marginally exceeded the passing threshold, overall accuracy and answer consistency remained suboptimal. Moreover, all the models demonstrated notable limitations in answer consistency and robustness. Further advancements are needed to ensure reliable and stable AI-based study aids for dietitian licensure preparation. 

---
# Guided Navigation in Knowledge-Dense Environments: Structured Semantic Exploration with Guidance Graphs 

**Authors**: Dehao Tao, Guangjie Liu, Weizheng, Yongfeng Huang, Minghu jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10012)  

**Abstract**: While Large Language Models (LLMs) exhibit strong linguistic capabilities, their reliance on static knowledge and opaque reasoning processes limits their performance in knowledge intensive tasks. Knowledge graphs (KGs) offer a promising solution, but current exploration methods face a fundamental trade off: question guided approaches incur redundant exploration due to granularity mismatches, while clue guided methods fail to effectively leverage contextual information for complex scenarios. To address these limitations, we propose Guidance Graph guided Knowledge Exploration (GG Explore), a novel framework that introduces an intermediate Guidance Graph to bridge unstructured queries and structured knowledge retrieval. The Guidance Graph defines the retrieval space by abstracting the target knowledge' s structure while preserving broader semantic context, enabling precise and efficient exploration. Building upon the Guidance Graph, we develop: (1) Structural Alignment that filters incompatible candidates without LLM overhead, and (2) Context Aware Pruning that enforces semantic consistency with graph constraints. Extensive experiments show our method achieves superior efficiency and outperforms SOTA, especially on complex tasks, while maintaining strong performance with smaller LLMs, demonstrating practical value. 

---
# Prompt-Response Semantic Divergence Metrics for Faithfulness Hallucination and Misalignment Detection in Large Language Models 

**Authors**: Igor Halperin  

**Link**: [PDF](https://arxiv.org/pdf/2508.10192)  

**Abstract**: The proliferation of Large Language Models (LLMs) is challenged by hallucinations, critical failure modes where models generate non-factual, nonsensical or unfaithful text. This paper introduces Semantic Divergence Metrics (SDM), a novel lightweight framework for detecting Faithfulness Hallucinations -- events of severe deviations of LLMs responses from input contexts. We focus on a specific implementation of these LLM errors, {confabulations, defined as responses that are arbitrary and semantically misaligned with the user's query. Existing methods like Semantic Entropy test for arbitrariness by measuring the diversity of answers to a single, fixed prompt. Our SDM framework improves upon this by being more prompt-aware: we test for a deeper form of arbitrariness by measuring response consistency not only across multiple answers but also across multiple, semantically-equivalent paraphrases of the original prompt. Methodologically, our approach uses joint clustering on sentence embeddings to create a shared topic space for prompts and answers. A heatmap of topic co-occurances between prompts and responses can be viewed as a quantified two-dimensional visualization of the user-machine dialogue. We then compute a suite of information-theoretic metrics to measure the semantic divergence between prompts and responses. Our practical score, $\mathcal{S}_H$, combines the Jensen-Shannon divergence and Wasserstein distance to quantify this divergence, with a high score indicating a Faithfulness hallucination. Furthermore, we identify the KL divergence KL(Answer $||$ Prompt) as a powerful indicator of \textbf{Semantic Exploration}, a key signal for distinguishing different generative behaviors. These metrics are further combined into the Semantic Box, a diagnostic framework for classifying LLM response types, including the dangerous, confident confabulation. 

---
# A Rose by Any Other Name Would Smell as Sweet: Categorical Homotopy Theory for Large Language Models 

**Authors**: Sridhar Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10018)  

**Abstract**: Natural language is replete with superficially different statements, such as ``Charles Darwin wrote" and ``Charles Darwin is the author of", which carry the same meaning. Large language models (LLMs) should generate the same next-token probabilities in such cases, but usually do not. Empirical workarounds have been explored, such as using k-NN estimates of sentence similarity to produce smoothed estimates. In this paper, we tackle this problem more abstractly, introducing a categorical homotopy framework for LLMs. We introduce an LLM Markov category to represent probability distributions in language generated by an LLM, where the probability of a sentence, such as ``Charles Darwin wrote" is defined by an arrow in a Markov category. However, this approach runs into difficulties as language is full of equivalent rephrases, and each generates a non-isomorphic arrow in the LLM Markov category. To address this fundamental problem, we use categorical homotopy techniques to capture ``weak equivalences" in an LLM Markov category. We present a detailed overview of application of categorical homotopy to LLMs, from higher algebraic K-theory to model categories, building on powerful theoretical results developed over the past half a century. 

---
# Multidimensional classification of posts for online course discussion forum curation 

**Authors**: Antonio Leandro Martins Candido, Jose Everardo Bessa Maia  

**Link**: [PDF](https://arxiv.org/pdf/2508.10008)  

**Abstract**: The automatic curation of discussion forums in online courses requires constant updates, making frequent retraining of Large Language Models (LLMs) a resource-intensive process. To circumvent the need for costly fine-tuning, this paper proposes and evaluates the use of Bayesian fusion. The approach combines the multidimensional classification scores of a pre-trained generic LLM with those of a classifier trained on local data. The performance comparison demonstrated that the proposed fusion improves the results compared to each classifier individually, and is competitive with the LLM fine-tuning approach 

---
# An Audit and Analysis of LLM-Assisted Health Misinformation Jailbreaks Against LLMs 

**Authors**: Ayana Hussain, Patrick Zhao, Nicholas Vincent  

**Link**: [PDF](https://arxiv.org/pdf/2508.10010)  

**Abstract**: Large Language Models (LLMs) are a double-edged sword capable of generating harmful misinformation -- inadvertently, or when prompted by "jailbreak" attacks that attempt to produce malicious outputs. LLMs could, with additional research, be used to detect and prevent the spread of misinformation. In this paper, we investigate the efficacy and characteristics of LLM-produced jailbreak attacks that cause other models to produce harmful medical misinformation. We also study how misinformation generated by jailbroken LLMs compares to typical misinformation found on social media, and how effectively it can be detected using standard machine learning approaches. Specifically, we closely examine 109 distinct attacks against three target LLMs and compare the attack prompts to in-the-wild health-related LLM queries. We also examine the resulting jailbreak responses, comparing the generated misinformation to health-related misinformation on Reddit. Our findings add more evidence that LLMs can be effectively used to detect misinformation from both other LLMs and from people, and support a body of work suggesting that with careful design, LLMs can contribute to a healthier overall information ecosystem. 

---
# Automated scoring of the Ambiguous Intentions Hostility Questionnaire using fine-tuned large language models 

**Authors**: Y. Lyu, D. Combs, D. Neumann, Y. C. Leong  

**Link**: [PDF](https://arxiv.org/pdf/2508.10007)  

**Abstract**: Hostile attribution bias is the tendency to interpret social interactions as intentionally hostile. The Ambiguous Intentions Hostility Questionnaire (AIHQ) is commonly used to measure hostile attribution bias, and includes open-ended questions where participants describe the perceived intentions behind a negative social situation and how they would respond. While these questions provide insights into the contents of hostile attributions, they require time-intensive scoring by human raters. In this study, we assessed whether large language models can automate the scoring of AIHQ open-ended responses. We used a previously collected dataset in which individuals with traumatic brain injury (TBI) and healthy controls (HC) completed the AIHQ and had their open-ended responses rated by trained human raters. We used half of these responses to fine-tune the two models on human-generated ratings, and tested the fine-tuned models on the remaining half of AIHQ responses. Results showed that model-generated ratings aligned with human ratings for both attributions of hostility and aggression responses, with fine-tuned models showing higher alignment. This alignment was consistent across ambiguous, intentional, and accidental scenario types, and replicated previous findings on group differences in attributions of hostility and aggression responses between TBI and HC groups. The fine-tuned models also generalized well to an independent nonclinical dataset. To support broader adoption, we provide an accessible scoring interface that includes both local and cloud-based options. Together, our findings suggest that large language models can streamline AIHQ scoring in both research and clinical contexts, revealing their potential to facilitate psychological assessments across different populations. 

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
# Semantic Structure in Large Language Model Embeddings 

**Authors**: Austin C. Kozlowski, Callin Dai, Andrei Boutyline  

**Link**: [PDF](https://arxiv.org/pdf/2508.10003)  

**Abstract**: Psychological research consistently finds that human ratings of words across diverse semantic scales can be reduced to a low-dimensional form with relatively little information loss. We find that the semantic associations encoded in the embedding matrices of large language models (LLMs) exhibit a similar structure. We show that the projections of words on semantic directions defined by antonym pairs (e.g. kind - cruel) correlate highly with human ratings, and further find that these projections effectively reduce to a 3-dimensional subspace within LLM embeddings, closely resembling the patterns derived from human survey responses. Moreover, we find that shifting tokens along one semantic direction causes off-target effects on geometrically aligned features proportional to their cosine similarity. These findings suggest that semantic features are entangled within LLMs similarly to how they are interconnected in human language, and a great deal of semantic information, despite its apparent complexity, is surprisingly low-dimensional. Furthermore, accounting for this semantic structure may prove essential for avoiding unintended consequences when steering features. 

---
# Semantic Bridge: Universal Multi-Hop Question Generation via AMR-Driven Graph Synthesis 

**Authors**: Linqing Chen, Hanmeng Zhong, Wentao Wu, Weilei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10013)  

**Abstract**: Large language model (LLM) training faces a critical bottleneck: the scarcity of high-quality, reasoning-intensive question-answer pairs, especially from sparse, domain-specific sources like PubMed papers or legal documents. Existing methods rely on surface patterns, fundamentally failing to generate controllable, complex multi-hop reasoning questions that test genuine understanding-essential for advancing LLM training paradigms. We present \textbf{Semantic Bridge}, the first universal framework for controllably generating sophisticated multi-hop reasoning questions from arbitrary sources. Our breakthrough innovation is \textit{semantic graph weaving}-three complementary bridging mechanisms (entity bridging for role-varying shared entities, predicate chain bridging for temporal/causal/logical sequences, and causal bridging for explicit reasoning chains)-that systematically construct complex pathways across documents, with fine-grained control over complexity and types via AMR-driven analysis. Our multi-modal AMR pipeline achieves up to 9.5% better round-trip quality, enabling production-ready controllable QA generation. Extensive evaluation demonstrates performance across both general-purpose datasets (Wikipedia) and specialized domains (biomedicine) It yields consistent 18.3%-25.4% gains over baselines across four languages (English, Chinese, French, German). Question pairs generated from 200 sources outperform 600 native human annotation examples with 67% fewer materials. Human evaluation shows 23.4% higher complexity, 18.7% better answerability, and 31.2% improved pattern coverage. Semantic Bridge establishes a new paradigm for LLM training data synthesis, enabling controllable generation of targeted reasoning questions from sparse sources. We will release our core code and semantic bridge model. 

---
# AutoGeTS: Knowledge-based Automated Generation of Text Synthetics for Improving Text Classification 

**Authors**: Chenhao Xue, Yuanzhe Jin, Adrian Carrasco-Revilla, Joyraj Chakraborty, Min Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10000)  

**Abstract**: When developing text classification models for real world applications, one major challenge is the difficulty to collect sufficient data for all text classes. In this work, we address this challenge by utilizing large language models (LLMs) to generate synthetic data and using such data to improve the performance of the models without waiting for more real data to be collected and labelled. As an LLM generates different synthetic data in response to different input examples, we formulate an automated workflow, which searches for input examples that lead to more ``effective'' synthetic data for improving the model concerned. We study three search strategies with an extensive set of experiments, and use experiment results to inform an ensemble algorithm that selects a search strategy according to the characteristics of a class. Our further experiments demonstrate that this ensemble approach is more effective than each individual strategy in our automated workflow for improving classification models using LLMs. 

---
# Training-Free Multimodal Large Language Model Orchestration 

**Authors**: Tianyu Xie, Yuhang Wu, Yongdong Luo, Jiayi Ji, Xiawu Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.10016)  

**Abstract**: Different Multimodal Large Language Models (MLLMs) cannot be integrated into a unified multimodal input-output system directly. In previous work, training has been considered as an inevitable component due to challenges in modal alignment, Text-to-Speech efficiency and other integration issues. In this paper, we introduce Multimodal Large Language Model Orchestration, an effective approach for creating interactive multimodal AI systems without additional training. MLLM Orchestration leverages the inherent reasoning capabilities of large language models to coordinate specialized models through explicit workflows, enabling natural multimodal interactions while maintaining modularity, improving interpretability, and significantly enhancing computational efficiency. Our orchestration framework is built upon three key innovations: (1) a central controller LLM that analyzes user inputs and dynamically routes tasks to appropriate specialized models through carefully designed agents; (2) a parallel Text-to-Speech architecture that enables true full-duplex interaction with seamless interruption handling and natural conversational flow; and (3) a cross-modal memory integration system that maintains coherent context across modalities through intelligent information synthesis and retrieval, selectively avoiding unnecessary modality calls in certain scenarios to improve response speed. Extensive evaluations demonstrate that MLLM Orchestration achieves comprehensive multimodal capabilities without additional training, performance improvements of up to 7.8% over traditional jointly-trained approaches on standard benchmarks, reduced latency by 10.3%, and significantly enhanced interpretability through explicit orchestration processes. 

---
# Conformal P-Value in Multiple-Choice Question Answering Tasks with Provable Risk Control 

**Authors**: Yuanchang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2508.10022)  

**Abstract**: This study introduces a significance testing-enhanced conformal prediction (CP) framework to improve trustworthiness of large language models (LLMs) in multiple-choice question answering (MCQA). While LLMs have been increasingly deployed in disciplinary QA scenarios, hallucination and nonfactual generation substantially compromise response reliability. Although CP provides statistically rigorous marginal coverage guarantees for prediction sets, and significance testing offers established statistical rigor, their synergistic integration remains unexplored. To mitigate hallucination and factual inaccuracies, our framework integrates $p$-value computation with conformity scoring through self-consistency resampling of MCQA responses. This approach calculates option frequencies to address LLMs' black-box nature, subsequently constructing prediction sets via null hypothesis testing ($\mathcal{H}_0$) with empirically derived $p$-values. Evaluations on MMLU and MMLU-Pro benchmarks using off-the-shelf LLMs demonstrate: (1) The enhanced CP achieves user-specified empirical miscoverage rates; (2) Test-set average prediction set size (APSS) decreases monotonically with increasing risk levels ($\alpha$), validating APSS as an effective uncertainty metric. This work establishes a principled statistical framework for trustworthy LLM deployment in high-stakes QA applications. 

---
# Reverse Physician-AI Relationship: Full-process Clinical Diagnosis Driven by a Large Language Model 

**Authors**: Shicheng Xu, Xin Huang, Zihao Wei, Liang Pang, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.10492)  

**Abstract**: Full-process clinical diagnosis in the real world encompasses the entire diagnostic workflow that begins with only an ambiguous chief complaint. While artificial intelligence (AI), particularly large language models (LLMs), is transforming clinical diagnosis, its role remains largely as an assistant to physicians. This AI-assisted working pattern makes AI can only answer specific medical questions at certain parts within the diagnostic process, but lack the ability to drive the entire diagnostic process starting from an ambiguous complaint, which still relies heavily on human physicians. This gap limits AI's ability to fully reduce physicians' workload and enhance diagnostic efficiency. To address this, we propose a paradigm shift that reverses the relationship between physicians and AI: repositioning AI as the primary director, with physicians serving as its assistants. So we present DxDirector-7B, an LLM endowed with advanced deep thinking capabilities, enabling it to drive the full-process diagnosis with minimal physician involvement. Furthermore, DxDirector-7B establishes a robust accountability framework for misdiagnoses, delineating responsibility between AI and human physicians. In evaluations across rare, complex, and real-world cases under full-process diagnosis setting, DxDirector-7B not only achieves significant superior diagnostic accuracy but also substantially reduces physician workload than state-of-the-art medical LLMs as well as general-purpose LLMs. Fine-grained analyses across multiple clinical departments and tasks validate its efficacy, with expert evaluations indicating its potential to serve as a viable substitute for medical specialists. These findings mark a new era where AI, traditionally a physicians' assistant, now drives the entire diagnostic process to drastically reduce physicians' workload, indicating an efficient and accurate diagnostic solution. 

---
# Personalized Real-time Jargon Support for Online Meetings 

**Authors**: Yifan Song, Wing Yee Au, Hon Yung Wong, Brian P. Bailey, Tal August  

**Link**: [PDF](https://arxiv.org/pdf/2508.10239)  

**Abstract**: Effective interdisciplinary communication is frequently hindered by domain-specific jargon. To explore the jargon barriers in-depth, we conducted a formative diary study with 16 professionals, revealing critical limitations in current jargon-management strategies during workplace meetings. Based on these insights, we designed ParseJargon, an interactive LLM-powered system providing real-time personalized jargon identification and explanations tailored to users' individual backgrounds. A controlled experiment comparing ParseJargon against baseline (no support) and general-purpose (non-personalized) conditions demonstrated that personalized jargon support significantly enhanced participants' comprehension, engagement, and appreciation of colleagues' work, whereas general-purpose support negatively affected engagement. A follow-up field study validated ParseJargon's usability and practical value in real-time meetings, highlighting both opportunities and limitations for real-world deployment. Our findings contribute insights into designing personalized jargon support tools, with implications for broader interdisciplinary and educational applications. 

---
# Searching for Privacy Risks in LLM Agents via Simulation 

**Authors**: Yanzhe Zhang, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10880)  

**Abstract**: The widespread deployment of LLM-based agents is likely to introduce a critical privacy threat: malicious agents that proactively engage others in multi-turn interactions to extract sensitive information. These dynamic dialogues enable adaptive attack strategies that can cause severe privacy violations, yet their evolving nature makes it difficult to anticipate and discover sophisticated vulnerabilities manually. To tackle this problem, we present a search-based framework that alternates between improving attacker and defender instructions by simulating privacy-critical agent interactions. Each simulation involves three roles: data subject, data sender, and data recipient. While the data subject's behavior is fixed, the attacker (data recipient) attempts to extract sensitive information from the defender (data sender) through persistent and interactive exchanges. To explore this interaction space efficiently, our search algorithm employs LLMs as optimizers, using parallel search with multiple threads and cross-thread propagation to analyze simulation trajectories and iteratively propose new instructions. Through this process, we find that attack strategies escalate from simple direct requests to sophisticated multi-turn tactics such as impersonation and consent forgery, while defenses advance from rule-based constraints to identity-verification state machines. The discovered attacks and defenses transfer across diverse scenarios and backbone models, demonstrating strong practical utility for building privacy-aware agents. 

---
# Pass@k Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning Models 

**Authors**: Zhipeng Chen, Xiaobo Qin, Youbin Wu, Yue Ling, Qinghao Ye, Wayne Xin Zhao, Guang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10751)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR), which typically adopts Pass@1 as the reward, has faced the issues in balancing exploration and exploitation, causing policies to prefer conservative actions, converging to a local optimum. Identifying an appropriate reward metric is therefore crucial. Regarding the prior work, although Pass@k has been used in evaluation, its connection to LLM exploration ability in RLVR remains largely overlooked. To investigate this, we first use Pass@k as the reward to train the policy model (i.e., $\textbf{Pass@k Training}$), and observe the improvement on its exploration ability. Next, we derive an analytical solution for the advantage of Pass@k Training, leading to an efficient and effective process. Building on this, our analysis reveals that exploration and exploitation are not inherently conflicting objectives, while they can mutually enhance each other. Moreover, Pass@k Training with analytical derivation essentially involves directly designing the advantage function. Inspired by this, we preliminarily explore the advantage design for RLVR, showing promising results and highlighting a potential future direction. 

---
# XFacta: Contemporary, Real-World Dataset and Evaluation for Multimodal Misinformation Detection with Multimodal LLMs 

**Authors**: Yuzhuo Xiao, Zeyu Han, Yuhan Wang, Huaizu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09999)  

**Abstract**: The rapid spread of multimodal misinformation on social media calls for more effective and robust detection methods. Recent advances leveraging multimodal large language models (MLLMs) have shown the potential in addressing this challenge. However, it remains unclear exactly where the bottleneck of existing approaches lies (evidence retrieval v.s. reasoning), hindering the further advances in this field. On the dataset side, existing benchmarks either contain outdated events, leading to evaluation bias due to discrepancies with contemporary social media scenarios as MLLMs can simply memorize these events, or artificially synthetic, failing to reflect real-world misinformation patterns. Additionally, it lacks comprehensive analyses of MLLM-based model design strategies. To address these issues, we introduce XFacta, a contemporary, real-world dataset that is better suited for evaluating MLLM-based detectors. We systematically evaluate various MLLM-based misinformation detection strategies, assessing models across different architectures and scales, as well as benchmarking against existing detection methods. Building on these analyses, we further enable a semi-automatic detection-in-the-loop framework that continuously updates XFacta with new content to maintain its contemporary relevance. Our analysis provides valuable insights and practices for advancing the field of multimodal misinformation detection. The code and data have been released. 

---
# Large Language Models Show Signs of Alignment with Human Neurocognition During Abstract Reasoning 

**Authors**: Christopher Pinier, Sonia Acuña Vargas, Mariia Steeghs-Turchina, Dora Matzke, Claire E. Stevenson, Michael D. Nunez  

**Link**: [PDF](https://arxiv.org/pdf/2508.10057)  

**Abstract**: This study investigates whether large language models (LLMs) mirror human neurocognition during abstract reasoning. We compared the performance and neural representations of human participants with those of eight open-source LLMs on an abstract-pattern-completion task. We leveraged pattern type differences in task performance and in fixation-related potentials (FRPs) as recorded by electroencephalography (EEG) during the task. Our findings indicate that only the largest tested LLMs (~70 billion parameters) achieve human-comparable accuracy, with Qwen-2.5-72B and DeepSeek-R1-70B also showing similarities with the human pattern-specific difficulty profile. Critically, every LLM tested forms representations that distinctly cluster the abstract pattern categories within their intermediate layers, although the strength of this clustering scales with their performance on the task. Moderate positive correlations were observed between the representational geometries of task-optimal LLM layers and human frontal FRPs. These results consistently diverged from comparisons with other EEG measures (response-locked ERPs and resting EEG), suggesting a potential shared representational space for abstract patterns. This indicates that LLMs might mirror human brain mechanisms in abstract reasoning, offering preliminary evidence of shared principles between biological and artificial intelligence. 

---
# Using Large Language Models to Measure Symptom Severity in Patients At Risk for Schizophrenia 

**Authors**: Andrew X. Chen, Guillermo Horga, Sean Escola  

**Link**: [PDF](https://arxiv.org/pdf/2508.10226)  

**Abstract**: Patients who are at clinical high risk (CHR) for schizophrenia need close monitoring of their symptoms to inform appropriate treatments. The Brief Psychiatric Rating Scale (BPRS) is a validated, commonly used research tool for measuring symptoms in patients with schizophrenia and other psychotic disorders; however, it is not commonly used in clinical practice as it requires a lengthy structured interview. Here, we utilize large language models (LLMs) to predict BPRS scores from clinical interview transcripts in 409 CHR patients from the Accelerating Medicines Partnership Schizophrenia (AMP-SCZ) cohort. Despite the interviews not being specifically structured to measure the BPRS, the zero-shot performance of the LLM predictions compared to the true assessment (median concordance: 0.84, ICC: 0.73) approaches human inter- and intra-rater reliability. We further demonstrate that LLMs have substantial potential to improve and standardize the assessment of CHR patients via their accuracy in assessing the BPRS in foreign languages (median concordance: 0.88, ICC: 0.70), and integrating longitudinal information in a one-shot or few-shot learning approach. 

---
# Context Misleads LLMs: The Role of Context Filtering in Maintaining Safe Alignment of LLMs 

**Authors**: Jinhwa Kim, Ian G. Harris  

**Link**: [PDF](https://arxiv.org/pdf/2508.10031)  

**Abstract**: While Large Language Models (LLMs) have shown significant advancements in performance, various jailbreak attacks have posed growing safety and ethical risks. Malicious users often exploit adversarial context to deceive LLMs, prompting them to generate responses to harmful queries. In this study, we propose a new defense mechanism called Context Filtering model, an input pre-processing method designed to filter out untrustworthy and unreliable context while identifying the primary prompts containing the real user intent to uncover concealed malicious intent. Given that enhancing the safety of LLMs often compromises their helpfulness, potentially affecting the experience of benign users, our method aims to improve the safety of the LLMs while preserving their original performance. We evaluate the effectiveness of our model in defending against jailbreak attacks through comparative analysis, comparing our approach with state-of-the-art defense mechanisms against six different attacks and assessing the helpfulness of LLMs under these defenses. Our model demonstrates its ability to reduce the Attack Success Rates of jailbreak attacks by up to 88% while maintaining the original LLMs' performance, achieving state-of-the-art Safety and Helpfulness Product results. Notably, our model is a plug-and-play method that can be applied to all LLMs, including both white-box and black-box models, to enhance their safety without requiring any fine-tuning of the models themselves. We will make our model publicly available for research purposes. 

---
# The Knowledge-Reasoning Dissociation: Fundamental Limitations of LLMs in Clinical Natural Language Inference 

**Authors**: Maël Jullien, Marco Valentino, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2508.10777)  

**Abstract**: Large language models are often assumed to acquire increasingly structured, generalizable internal representations simply by scaling data and parameters. We interrogate this assumption by introducing a Clinical Trial Natural Language Inference benchmark comprising four reasoning families, Causal Attribution, Compositional Grounding, Epistemic Verification, and Risk State Abstraction. Each item is paired with a targeted Ground Knowledge and Meta-Level Reasoning Verification (GKMRV) probe, allowing us to dissociate failures of factual access from failures of inference. We evaluate six contemporary LLMs under both direct and chain of thought prompting.
Models achieve near-ceiling GKMRV accuracy (mean accuracy 0.918) yet perform poorly on the main reasoning tasks (mean accuracy 0.25). Despite low accuracy, output inferences are highly consistent across samples (mean 0.87), indicating a systematic application of underlying heuristics and shortcuts.
These results reveal fundamental structural and representational limitations: current LLMs often possess the relevant clinical knowledge but lack the structured, composable internal representations needed to deploy it reliably (e.g., integrating constraints, weighing evidence, or simulating counterfactuals). Decoupling knowledge from reasoning with GKMRV makes this dissociation explicit and measurable, providing an effective framework for probing the reliability of LLMs in high-stakes domains. 

---
# GenOM: Ontology Matching with Description Generation and Large Language Model 

**Authors**: Yiping Song, Jiaoyan Chen, Renate A. Schmidt  

**Link**: [PDF](https://arxiv.org/pdf/2508.10703)  

**Abstract**: Ontology matching (OM) plays an essential role in enabling semantic interoperability and integration across heterogeneous knowledge sources, particularly in the biomedical domain which contains numerous complex concepts related to diseases and pharmaceuticals. This paper introduces GenOM, a large language model (LLM)-based ontology alignment framework, which enriches the semantic representations of ontology concepts via generating textual definitions, retrieves alignment candidates with an embedding model, and incorporates exact matching-based tools to improve precision. Extensive experiments conducted on the OAEI Bio-ML track demonstrate that GenOM can often achieve competitive performance, surpassing many baselines including traditional OM systems and recent LLM-based methods. Further ablation studies confirm the effectiveness of semantic enrichment and few-shot prompting, highlighting the framework's robustness and adaptability. 

---
# SEQ-GPT: LLM-assisted Spatial Query via Example 

**Authors**: Ivan Khai Ze Lim, Ningyi Liao, Yiming Yang, Gerald Wei Yong Yip, Siqiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.10486)  

**Abstract**: Contemporary spatial services such as online maps predominantly rely on user queries for location searches. However, the user experience is limited when performing complex tasks, such as searching for a group of locations simultaneously. In this study, we examine the extended scenario known as Spatial Exemplar Query (SEQ), where multiple relevant locations are jointly searched based on user-specified examples. We introduce SEQ-GPT, a spatial query system powered by Large Language Models (LLMs) towards more versatile SEQ search using natural language. The language capabilities of LLMs enable unique interactive operations in the SEQ process, including asking users to clarify query details and dynamically adjusting the search based on user feedback. We also propose a tailored LLM adaptation pipeline that aligns natural language with structured spatial data and queries through dialogue synthesis and multi-model cooperation. SEQ-GPT offers an end-to-end demonstration for broadening spatial search with realistic data and application scenarios. 

---
# FIRESPARQL: A LLM-based Framework for SPARQL Query Generation over Scholarly Knowledge Graphs 

**Authors**: Xueli Pan, Victor de Boer, Jacco van Ossenbruggen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10467)  

**Abstract**: Question answering over Scholarly Knowledge Graphs (SKGs) remains a challenging task due to the complexity of scholarly content and the intricate structure of these graphs. Large Language Model (LLM) approaches could be used to translate natural language questions (NLQs) into SPARQL queries; however, these LLM-based approaches struggle with SPARQL query generation due to limited exposure to SKG-specific content and the underlying schema. We identified two main types of errors in the LLM-generated SPARQL queries: (i) structural inconsistencies, such as missing or redundant triples in the queries, and (ii) semantic inaccuracies, where incorrect entities or properties are shown in the queries despite a correct query structure. To address these issues, we propose FIRESPARQL, a modular framework that supports fine-tuned LLMs as a core component, with optional context provided via retrieval-augmented generation (RAG) and a SPARQL query correction layer. We evaluate the framework on the SciQA Benchmark using various configurations (zero-shot, zero-shot with RAG, one-shot, fine-tuning, and fine-tuning with RAG) and compare the performance with baseline and state-of-the-art approaches. We measure query accuracy using BLEU and ROUGE metrics, and query result accuracy using relaxed exact match(RelaxedEM), with respect to the gold standards containing the NLQs, SPARQL queries, and the results of the queries. Experimental results demonstrate that fine-tuning achieves the highest overall performance, reaching 0.90 ROUGE-L for query accuracy and 0.85 RelaxedEM for result accuracy on the test set. 

---
# What to Ask Next? Probing the Imaginative Reasoning of LLMs with TurtleSoup Puzzles 

**Authors**: Mengtao Zhou, Sifan Wu, Huan Zhang, Qi Sima, Bang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10358)  

**Abstract**: We investigate the capacity of Large Language Models (LLMs) for imaginative reasoning--the proactive construction, testing, and revision of hypotheses in information-sparse environments. Existing benchmarks, often static or focused on social deduction, fail to capture the dynamic, exploratory nature of this reasoning process. To address this gap, we introduce a comprehensive research framework based on the classic "Turtle Soup" game, integrating a benchmark, an agent, and an evaluation protocol. We present TurtleSoup-Bench, the first large-scale, bilingual, interactive benchmark for imaginative reasoning, comprising 800 turtle soup puzzles sourced from both the Internet and expert authors. We also propose Mosaic-Agent, a novel agent designed to assess LLMs' performance in this setting. To evaluate reasoning quality, we develop a multi-dimensional protocol measuring logical consistency, detail completion, and conclusion alignment. Experiments with leading LLMs reveal clear capability limits, common failure patterns, and a significant performance gap compared to humans. Our work offers new insights into LLMs' imaginative reasoning and establishes a foundation for future research on exploratory agent behavior. 

---
# Promoting Efficient Reasoning with Verifiable Stepwise Reward 

**Authors**: Chuhuai Yue, Chengqi Dong, Yinan Gao, Hang He, Jiajun Chai, Guojun Yin, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.10293)  

**Abstract**: Large reasoning models (LRMs) have recently achieved significant progress in complex reasoning tasks, aided by reinforcement learning with verifiable rewards. However, LRMs often suffer from overthinking, expending excessive computation on simple problems and reducing efficiency. Existing efficient reasoning methods typically require accurate task assessment to preset token budgets or select reasoning modes, which limits their flexibility and reliability. In this work, we revisit the essence of overthinking and identify that encouraging effective steps while penalizing ineffective ones is key to its solution. To this end, we propose a novel rule-based verifiable stepwise reward mechanism (VSRM), which assigns rewards based on the performance of intermediate states in the reasoning trajectory. This approach is intuitive and naturally fits the step-by-step nature of reasoning tasks. We conduct extensive experiments on standard mathematical reasoning benchmarks, including AIME24 and AIME25, by integrating VSRM with PPO and Reinforce++. Results show that our method achieves substantial output length reduction while maintaining original reasoning performance, striking an optimal balance between efficiency and accuracy. Further analysis of overthinking frequency and pass@k score before and after training demonstrates that our approach in deed effectively suppresses ineffective steps and encourages effective reasoning, fundamentally alleviating the overthinking problem. All code will be released upon acceptance. 

---
# Agentic Design Review System 

**Authors**: Sayan Nag, K J Joseph, Koustava Goswami, Vlad I Morariu, Balaji Vasan Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10745)  

**Abstract**: Evaluating graphic designs involves assessing it from multiple facets like alignment, composition, aesthetics and color choices. Evaluating designs in a holistic way involves aggregating feedback from individual expert reviewers. Towards this, we propose an Agentic Design Review System (AgenticDRS), where multiple agents collaboratively analyze a design, orchestrated by a meta-agent. A novel in-context exemplar selection approach based on graph matching and a unique prompt expansion method plays central role towards making each agent design aware. Towards evaluating this framework, we propose DRS-BENCH benchmark. Thorough experimental evaluation against state-of-the-art baselines adapted to the problem setup, backed-up with critical ablation experiments brings out the efficacy of Agentic-DRS in evaluating graphic designs and generating actionable feedback. We hope that this work will attract attention to this pragmatic, yet under-explored research direction. 

---
# Modeling Human Responses to Multimodal AI Content 

**Authors**: Zhiqi Shen, Shaojing Fan, Danni Xu, Terence Sim, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2508.10769)  

**Abstract**: As AI-generated content becomes widespread, so does the risk of misinformation. While prior research has primarily focused on identifying whether content is authentic, much less is known about how such content influences human perception and behavior. In domains like trading or the stock market, predicting how people react (e.g., whether a news post will go viral), can be more critical than verifying its factual accuracy. To address this, we take a human-centered approach and introduce the MhAIM Dataset, which contains 154,552 online posts (111,153 of them AI-generated), enabling large-scale analysis of how people respond to AI-generated content. Our human study reveals that people are better at identifying AI content when posts include both text and visuals, particularly when inconsistencies exist between the two. We propose three new metrics: trustworthiness, impact, and openness, to quantify how users judge and engage with online content. We present T-Lens, an LLM-based agent system designed to answer user queries by incorporating predicted human responses to multimodal information. At its core is HR-MCP (Human Response Model Context Protocol), built on the standardized Model Context Protocol (MCP), enabling seamless integration with any LLM. This integration allows T-Lens to better align with human reactions, enhancing both interpretability and interaction capabilities. Our work provides empirical insights and practical tools to equip LLMs with human-awareness capabilities. By highlighting the complex interplay among AI, human cognition, and information reception, our findings suggest actionable strategies for mitigating the risks of AI-driven misinformation. 

---
# A Curriculum Learning Approach to Reinforcement Learning: Leveraging RAG for Multimodal Question Answering 

**Authors**: Chenliang Zhang, Lin Wang, Yuanyuan Lu, Yusheng Qi, Kexin Wang, Peixu Hou, Wenshi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10337)  

**Abstract**: This paper describes the solutions of the Dianping-Trust-Safety team for the META CRAG-MM challenge. The challenge requires building a comprehensive retrieval-augmented generation system capable for multi-modal multi-turn question answering. The competition consists of three tasks: (1) answering questions using structured data retrieved from an image-based mock knowledge graph, (2) synthesizing information from both knowledge graphs and web search results, and (3) handling multi-turn conversations that require context understanding and information aggregation from multiple sources. For Task 1, our solution is based on the vision large language model, enhanced by supervised fine-tuning with knowledge distilled from GPT-4.1. We further applied curriculum learning strategies to guide reinforcement learning, resulting in improved answer accuracy and reduced hallucination. For Task 2 and Task 3, we additionally leveraged web search APIs to incorporate external knowledge, enabling the system to better handle complex queries and multi-turn conversations. Our approach achieved 1st place in Task 1 with a significant lead of 52.38\%, and 3rd place in Task 3, demonstrating the effectiveness of the integration of curriculum learning with reinforcement learning in our training pipeline. 

---
# KompeteAI: Accelerated Autonomous Multi-Agent System for End-to-End Pipeline Generation for Machine Learning Problems 

**Authors**: Stepan Kulibaba, Artem Dzhalilov, Roman Pakhomov, Oleg Svidchenko, Alexander Gasnikov, Aleksei Shpilman  

**Link**: [PDF](https://arxiv.org/pdf/2508.10177)  

**Abstract**: Recent Large Language Model (LLM)-based AutoML systems demonstrate impressive capabilities but face significant limitations such as constrained exploration strategies and a severe execution bottleneck. Exploration is hindered by one-shot methods lacking diversity and Monte Carlo Tree Search (MCTS) approaches that fail to recombine strong partial solutions. The execution bottleneck arises from lengthy code validation cycles that stifle iterative refinement. To overcome these challenges, we introduce KompeteAI, a novel AutoML framework with dynamic solution space exploration. Unlike previous MCTS methods that treat ideas in isolation, KompeteAI introduces a merging stage that composes top candidates. We further expand the hypothesis space by integrating Retrieval-Augmented Generation (RAG), sourcing ideas from Kaggle notebooks and arXiv papers to incorporate real-world strategies. KompeteAI also addresses the execution bottleneck via a predictive scoring model and an accelerated debugging method, assessing solution potential using early stage metrics to avoid costly full-code execution. This approach accelerates pipeline evaluation 6.9 times. KompeteAI outperforms leading methods (e.g., RD-agent, AIDE, and Ml-Master) by an average of 3\% on the primary AutoML benchmark, MLE-Bench. Additionally, we propose Kompete-bench to address limitations in MLE-Bench, where KompeteAI also achieves state-of-the-art results 

---
# MCP-Orchestrated Multi-Agent System for Automated Disinformation Detection 

**Authors**: Alexandru-Andrei Avram, Adrian Groza, Alexandru Lecu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10143)  

**Abstract**: The large spread of disinformation across digital platforms creates significant challenges to information integrity. This paper presents a multi-agent system that uses relation extraction to detect disinformation in news articles, focusing on titles and short text snippets. The proposed Agentic AI system combines four agents: (i) a machine learning agent (logistic regression), (ii) a Wikipedia knowledge check agent (which relies on named entity recognition), (iii) a coherence detection agent (using LLM prompt engineering), and (iv) a web-scraped data analyzer that extracts relational triplets for fact checking. The system is orchestrated via the Model Context Protocol (MCP), offering shared context and live learning across components. Results demonstrate that the multi-agent ensemble achieves 95.3% accuracy with an F1 score of 0.964, significantly outperforming individual agents and traditional approaches. The weighted aggregation method, mathematically derived from individual agent misclassification rates, proves superior to algorithmic threshold optimization. The modular architecture makes the system easily scalable, while also maintaining details of the decision processes. 

---
# We-Math 2.0: A Versatile MathBook System for Incentivizing Visual Mathematical Reasoning 

**Authors**: Runqi Qiao, Qiuna Tan, Peiqing Yang, Yanzi Wang, Xiaowan Wang, Enhui Wan, Sitong Zhou, Guanting Dong, Yuchen Zeng, Yida Xu, Jie Wang, Chong Sun, Chen Li, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10433)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities across various tasks, but still struggle with complex mathematical reasoning. Existing research primarily focuses on dataset construction and method optimization, often overlooking two critical aspects: comprehensive knowledge-driven design and model-centric data space modeling. In this paper, we introduce We-Math 2.0, a unified system that integrates a structured mathematical knowledge system, model-centric data space modeling, and a reinforcement learning (RL)-based training paradigm to comprehensively enhance the mathematical reasoning abilities of MLLMs. The key contributions of We-Math 2.0 are fourfold: (1) MathBook Knowledge System: We construct a five-level hierarchical system encompassing 491 knowledge points and 1,819 fundamental principles. (2) MathBook-Standard & Pro: We develop MathBook-Standard, a dataset that ensures broad conceptual coverage and flexibility through dual expansion. Additionally, we define a three-dimensional difficulty space and generate 7 progressive variants per problem to build MathBook-Pro, a challenging dataset for robust training. (3) MathBook-RL: We propose a two-stage RL framework comprising: (i) Cold-Start Fine-tuning, which aligns the model with knowledge-oriented chain-of-thought reasoning; and (ii) Progressive Alignment RL, leveraging average-reward learning and dynamic data scheduling to achieve progressive alignment across difficulty levels. (4) MathBookEval: We introduce a comprehensive benchmark covering all 491 knowledge points with diverse reasoning step distributions. Experimental results show that MathBook-RL performs competitively with existing baselines on four widely-used benchmarks and achieves strong results on MathBookEval, suggesting promising generalization in mathematical reasoning. 

---
# Performance of GPT-5 in Brain Tumor MRI Reasoning 

**Authors**: Mojtaba Safari, Shansong Wang, Mingzhe Hu, Zach Eidex, Qiang Li, Xiaofeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10865)  

**Abstract**: Accurate differentiation of brain tumor types on magnetic resonance imaging (MRI) is critical for guiding treatment planning in neuro-oncology. Recent advances in large language models (LLMs) have enabled visual question answering (VQA) approaches that integrate image interpretation with natural language reasoning. In this study, we evaluated GPT-4o, GPT-5-nano, GPT-5-mini, and GPT-5 on a curated brain tumor VQA benchmark derived from 3 Brain Tumor Segmentation (BraTS) datasets - glioblastoma (GLI), meningioma (MEN), and brain metastases (MET). Each case included multi-sequence MRI triplanar mosaics and structured clinical features transformed into standardized VQA items. Models were assessed in a zero-shot chain-of-thought setting for accuracy on both visual and reasoning tasks. Results showed that GPT-5-mini achieved the highest macro-average accuracy (44.19%), followed by GPT-5 (43.71%), GPT-4o (41.49%), and GPT-5-nano (35.85%). Performance varied by tumor subtype, with no single model dominating across all cohorts. These findings suggest that GPT-5 family models can achieve moderate accuracy in structured neuro-oncological VQA tasks, but not at a level acceptable for clinical use. 

---
# Pruning Long Chain-of-Thought of Large Reasoning Models via Small-Scale Preference Optimization 

**Authors**: Bin Hong, Jiayu Liu, Zhenya Huang, Kai Zhang, Mengdi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10164)  

**Abstract**: Recent advances in Large Reasoning Models (LRMs) have demonstrated strong performance on complex tasks through long Chain-of-Thought (CoT) reasoning. However, their lengthy outputs increase computational costs and may lead to overthinking, raising challenges in balancing reasoning effectiveness and efficiency. Current methods for efficient reasoning often compromise reasoning quality or require extensive resources. This paper investigates efficient methods to reduce the generation length of LRMs. We analyze generation path distributions and filter generated trajectories through difficulty estimation. Subsequently, we analyze the convergence behaviors of the objectives of various preference optimization methods under a Bradley-Terry loss based framework. Based on the analysis, we propose Length Controlled Preference Optimization (LCPO) that directly balances the implicit reward related to NLL loss. LCPO can effectively learn length preference with limited data and training. Extensive experiments demonstrate that our approach significantly reduces the average output length by over 50\% across multiple benchmarks while maintaining the reasoning performance. Our work highlights the potential for computationally efficient approaches in guiding LRMs toward efficient reasoning. 

---
# A Survey of Optimization Modeling Meets LLMs: Progress and Future Directions 

**Authors**: Ziyang Xiao, Jingrong Xie, Lilin Xu, Shisi Guan, Jingyan Zhu, Xiongwei Han, Xiaojin Fu, WingYin Yu, Han Wu, Wei Shi, Qingcan Kang, Jiahui Duan, Tao Zhong, Mingxuan Yuan, Jia Zeng, Yuan Wang, Gang Chen, Dongxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10047)  

**Abstract**: By virtue of its great utility in solving real-world problems, optimization modeling has been widely employed for optimal decision-making across various sectors, but it requires substantial expertise from operations research professionals. With the advent of large language models (LLMs), new opportunities have emerged to automate the procedure of mathematical modeling. This survey presents a comprehensive and timely review of recent advancements that cover the entire technical stack, including data synthesis and fine-tuning for the base model, inference frameworks, benchmark datasets, and performance evaluation. In addition, we conducted an in-depth analysis on the quality of benchmark datasets, which was found to have a surprisingly high error rate. We cleaned the datasets and constructed a new leaderboard with fair performance evaluation in terms of base LLM model and datasets. We also build an online portal that integrates resources of cleaned datasets, code and paper repository to benefit the community. Finally, we identify limitations in current methodologies and outline future research opportunities. 

---
# EgoCross: Benchmarking Multimodal Large Language Models for Cross-Domain Egocentric Video Question Answering 

**Authors**: Yanjun Li, Yuqian Fu, Tianwen Qian, Qi'ao Xu, Silong Dai, Danda Pani Paudel, Luc Van Gool, Xiaoling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10729)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have significantly pushed the frontier of egocentric video question answering (EgocentricQA). However, existing benchmarks and studies are mainly limited to common daily activities such as cooking and cleaning. In contrast, real-world deployment inevitably encounters domain shifts, where target domains differ substantially in both visual style and semantic content. To bridge this gap, we introduce \textbf{EgoCross}, a comprehensive benchmark designed to evaluate the cross-domain generalization of MLLMs in EgocentricQA. EgoCross covers four diverse and challenging domains, including surgery, industry, extreme sports, and animal perspective, representing realistic and high-impact application scenarios. It comprises approximately 1,000 QA pairs across 798 video clips, spanning four key QA tasks: prediction, recognition, localization, and counting. Each QA pair provides both OpenQA and CloseQA formats to support fine-grained evaluation. Extensive experiments show that most existing MLLMs, whether general-purpose or egocentric-specialized, struggle to generalize to domains beyond daily life, highlighting the limitations of current models. Furthermore, we conduct several pilot studies, \eg, fine-tuning and reinforcement learning, to explore potential improvements. We hope EgoCross and our accompanying analysis will serve as a foundation for advancing domain-adaptive, robust egocentric video understanding. Data and codes will be released at: \href{this https URL}{this https URL.} 

---
# REFN: A Reinforcement-Learning-From-Network Framework against 1-day/n-day Exploitations 

**Authors**: Tianlong Yu, Lihong Liu, Ziyi Zhou, Fudu Xing, Kailong Wang, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10701)  

**Abstract**: The exploitation of 1 day or n day vulnerabilities poses severe threats to networked devices due to massive deployment scales and delayed patching (average Mean Time To Patch exceeds 60 days). Existing defenses, including host based patching and network based filtering, are inadequate due to limited scalability across diverse devices, compatibility issues especially with embedded or legacy systems, and error prone deployment process (manual patch validation). To address these issues, we introduce REFN (Reinforcement Learning From Network), a novel framework that trains Large Language Models (LLMs) to autonomously generate network filters to prevent 1 day or n day exploitations. REFN ensures scalability by uniquely employs Reinforcement Learning (RL) driven by online network rewards instead of traditional Human Feedback (RLHF). REFN guarantees compatibility via unified deployment on edge security gateways (Amazon Eero). REFN provides robustness via online validation using real network traffic. Crucially, REFN addresses three core challenges in training LLMs for exploit prevention: 1) expanding current LLMs limited vulnerability fixing expertise via Agentic RAG based Knowledge Distillation, 2) bridging current LLMs language to network gaps through an RL From VNF Pipeline that translates language context (vulnerability description) into network enforcement, 3) addressing the LLM hallucination and non determinism via the Online Agentic Validation that penalizes erroneous outputs. Evaluated across 22 families of 1 day or n day exploits, REFN demonstrates effectiveness (21.1 percent higher accuracy than alternatives), efficiency (Mean Time To Patch of 3.65 hours) and scalability (easily scale to 10K devices). REFN serves as an initial step toward training LLMs to rapidly prevent massive scale 1 day or n day exploitations. 

---
# FROGENT: An End-to-End Full-process Drug Design Agent 

**Authors**: Qihua Pan, Dong Xu, Jenna Xinyi Yao, Lijia Ma, Zexuan Zhu, Junkai Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.10760)  

**Abstract**: Powerful AI tools for drug discovery reside in isolated web apps, desktop programs, and code libraries. Such fragmentation forces scientists to manage incompatible interfaces and specialized scripts, which can be a cumbersome and repetitive process. To address this issue, a Full-pROcess druG dEsign ageNT, named FROGENT, has been proposed. Specifically, FROGENT utilizes a Large Language Model and the Model Context Protocol to integrate multiple dynamic biochemical databases, extensible tool libraries, and task-specific AI models. This agentic framework allows FROGENT to execute complicated drug discovery workflows dynamically, including component tasks such as target identification, molecule generation and retrosynthetic planning. FROGENT has been evaluated on eight benchmarks that cover various aspects of drug discovery, such as knowledge retrieval, property prediction, virtual screening, mechanistic analysis, molecular design, and synthesis. It was compared against six increasingly advanced ReAct-style agents that support code execution and literature searches. Empirical results demonstrated that FROGENT triples the best baseline performance in hit-finding and doubles it in interaction profiling, significantly outperforming both the open-source model Qwen3-32B and the commercial model GPT-4o. In addition, real-world cases have been utilized to validate the practicability and generalization of FROGENT. This development suggests that streamlining the agentic drug discovery pipeline can significantly enhance researcher productivity. 

---
# ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning 

**Authors**: Juyuan Wang, Rongchen Zhao, Wei Wei, Yufeng Wang, Mo Yu, Jie Zhou, Jin Xu, Liyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10419)  

**Abstract**: Narrative comprehension on long stories and novels has been a challenging domain attributed to their intricate plotlines and entangled, often evolving relations among characters and entities. Given the LLM's diminished reasoning over extended context and high computational cost, retrieval-based approaches remain a pivotal role in practice. However, traditional RAG methods can fall short due to their stateless, single-step retrieval process, which often overlooks the dynamic nature of capturing interconnected relations within long-range context. In this work, we propose ComoRAG, holding the principle that narrative reasoning is not a one-shot process, but a dynamic, evolving interplay between new evidence acquisition and past knowledge consolidation, analogous to human cognition when reasoning with memory-related signals in the brain. Specifically, when encountering a reasoning impasse, ComoRAG undergoes iterative reasoning cycles while interacting with a dynamic memory workspace. In each cycle, it generates probing queries to devise new exploratory paths, then integrates the retrieved evidence of new aspects into a global memory pool, thereby supporting the emergence of a coherent context for the query resolution. Across four challenging long-context narrative benchmarks (200K+ tokens), ComoRAG outperforms strong RAG baselines with consistent relative gains up to 11% compared to the strongest baseline. Further analysis reveals that ComoRAG is particularly advantageous for complex queries requiring global comprehension, offering a principled, cognitively motivated paradigm for retrieval-based long context comprehension towards stateful reasoning. Our code is publicly released at this https URL 

---
# MCP2OSC: Parametric Control by Natural Language 

**Authors**: Yuan-Yi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10414)  

**Abstract**: Text prompts enable intuitive content creation but may fall short in achieving high precision for intricate tasks; knob or slider controls offer precise adjustments at the cost of increased complexity. To address the gap between knobs and prompts, a new MCP (Model Context Protocol) server and a unique set of prompt design criteria are presented to enable exploring parametric OSC (OpenSoundControl) control by natural language prompts. Demonstrated by 14 practical QA examples with best practices and the generalized prompt templates, this study finds Claude integrated with the MCP2OSC server effective in generating OSC messages by natural language, interpreting, searching, and visualizing OSC messages, validating and debugging OSC messages, and managing OSC address patterns. MCP2OSC enhances human-machine collaboration by leveraging LLM (Large Language Model) to handle intricate OSC development tasks, and by empowering human creativity with an intuitive language interface featuring flexible precision controls: a prompt-based OSC tool. This study provides a novel perspective on the creative MCP application at the network protocol level by utilizing LLM's strength in directly processing and generating human-readable OSC messages. The results suggest its potential for a LLM-based universal control mechanism for multimedia devices. 

---
# A Unified Multi-Agent Framework for Universal Multimodal Understanding and Generation 

**Authors**: Jiulin Li, Ping Huang, Yexin Li, Shuo Chen, Juewen Hu, Ye Tian  

**Link**: [PDF](https://arxiv.org/pdf/2508.10494)  

**Abstract**: Real-world multimodal applications often require any-to-any capabilities, enabling both understanding and generation across modalities including text, image, audio, and video. However, integrating the strengths of autoregressive language models (LLMs) for reasoning and diffusion models for high-fidelity generation remains challenging. Existing approaches rely on rigid pipelines or tightly coupled architectures, limiting flexibility and scalability. We propose MAGUS (Multi-Agent Guided Unified Multimodal System), a modular framework that unifies multimodal understanding and generation via two decoupled phases: Cognition and Deliberation. MAGUS enables symbolic multi-agent collaboration within a shared textual workspace. In the Cognition phase, three role-conditioned multimodal LLM agents - Perceiver, Planner, and Reflector - engage in collaborative dialogue to perform structured understanding and planning. The Deliberation phase incorporates a Growth-Aware Search mechanism that orchestrates LLM-based reasoning and diffusion-based generation in a mutually reinforcing manner. MAGUS supports plug-and-play extensibility, scalable any-to-any modality conversion, and semantic alignment - all without the need for joint training. Experiments across multiple benchmarks, including image, video, and audio generation, as well as cross-modal instruction following, demonstrate that MAGUS outperforms strong baselines and state-of-the-art systems. Notably, on the MME benchmark, MAGUS surpasses the powerful closed-source model GPT-4o. 

---
# MRFD: Multi-Region Fusion Decoding with Self-Consistency for Mitigating Hallucinations in LVLMs 

**Authors**: Haonan Ge, Yiwei Wang, Ming-Hsuan Yang, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2508.10264)  

**Abstract**: Large Vision-Language Models (LVLMs) have shown strong performance across multimodal tasks. However, they often produce hallucinations -- text that is inconsistent with visual input, due to the limited ability to verify information in different regions of the image. To address this, we propose Multi-Region Fusion Decoding (MRFD), a training-free decoding method that improves factual grounding by modeling inter-region consistency. MRFD identifies salient regions using cross-attention, generates initial responses for each, and computes reliability weights based on Jensen-Shannon Divergence (JSD) among the responses. These weights guide a consistency-aware fusion of per-region predictions, using region-aware prompts inspired by Chain-of-Thought reasoning. Experiments across multiple LVLMs and benchmarks show that MRFD significantly reduces hallucinations and improves response factuality without requiring model updates. 

---
# Less is More: Learning Graph Tasks with Just LLMs 

**Authors**: Sola Shirai, Kavitha Srinivas, Julian Dolby, Michael Katz, Horst Samulowitz, Shirin Sohrabi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10115)  

**Abstract**: For large language models (LLMs), reasoning over graphs could help solve many problems. Prior work has tried to improve LLM graph reasoning by examining how best to serialize graphs as text and by combining GNNs and LLMs. However, the merits of such approaches remain unclear, so we empirically answer the following research questions: (1) Can LLMs learn to solve fundamental graph tasks without specialized graph encoding models?, (2) Can LLMs generalize learned solutions to unseen graph structures or tasks?, and (3) What are the merits of competing approaches to learn graph tasks? We show that even small LLMs can learn to solve graph tasks by training them with instructive chain-of-thought solutions, and this training generalizes, without specialized graph encoders, to new tasks and graph structures. 

---
