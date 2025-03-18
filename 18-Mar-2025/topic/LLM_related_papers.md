# Computation Mechanism Behind LLM Position Generalization 

**Authors**: Chi Han, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2503.13305)  

**Abstract**: Most written natural languages are composed of sequences of words and sentences. Similar to humans, large language models (LLMs) exhibit flexibility in handling textual positions - a phenomenon we term position generalization. They can understand texts with position perturbations and generalize to longer texts than those encountered during training with the latest techniques. These phenomena suggest that LLMs handle positions tolerantly, but how LLMs computationally process positional relevance remains largely unexplored. This work connects the linguistic phenomenon with LLMs' computational mechanisms. We show how LLMs enforce certain computational mechanisms for the aforementioned tolerance in position perturbations. Despite the complex design of the self-attention mechanism, this work reveals that LLMs learn a counterintuitive disentanglement of attention logits. Their values show a 0.959 linear correlation with an approximation of the arithmetic sum of positional relevance and semantic importance. Furthermore, we identify a prevalent pattern in intermediate features, which we prove theoretically enables this effect. The pattern, which is different from how randomly initialized parameters would behave, suggests that it is a learned behavior rather than a natural result of the model architecture. Based on these findings, we provide computational explanations and criteria for LLMs' position flexibilities. This work takes a pioneering step in linking position generalization with modern LLMs' internal mechanisms. 

---
# Reliable and Efficient Amortized Model-based Evaluation 

**Authors**: Sang Truong, Yuheng Tu, Percy Liang, Bo Li, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2503.13335)  

**Abstract**: Comprehensive evaluations of language models (LM) during both development and deployment phases are necessary because these models possess numerous capabilities (e.g., mathematical reasoning, legal support, or medical diagnostic) as well as safety risks (e.g., racial bias, toxicity, or misinformation). The average score across a wide range of benchmarks provides a signal that helps guide the use of these LMs in practice. Currently, holistic evaluations are costly due to the large volume of benchmark questions, making frequent evaluations impractical. A popular attempt to lower the cost is to compute the average score on a subset of the benchmark. This approach, unfortunately, often renders an unreliable measure of LM performance because the average score is often confounded with the difficulty of the questions in the benchmark subset. Item response theory (IRT) was designed to address this challenge, providing a reliable measurement by careful controlling for question difficulty. Unfortunately, question difficulty is expensive to estimate. Facing this challenge, we train a model that predicts question difficulty from its content, enabling a reliable measurement at a fraction of the cost. In addition, we leverage this difficulty predictor to further improve the evaluation efficiency through training a question generator given a difficulty level. This question generator is essential in adaptive testing, where, instead of using a random subset of the benchmark questions, informative questions are adaptively chosen based on the current estimation of LLM performance. Experiments on 22 common natural language benchmarks and 172 LMs show that this approach is more reliable and efficient compared to current common practice. 

---
# MetaScale: Test-Time Scaling with Evolving Meta-Thoughts 

**Authors**: Qin Liu, Wenxuan Zhou, Nan Xu, James Y. Huang, Fei Wang, Sheng Zhang, Hoifung Poon, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.13447)  

**Abstract**: One critical challenge for large language models (LLMs) for making complex reasoning is their reliance on matching reasoning patterns from training data, instead of proactively selecting the most appropriate cognitive strategy to solve a given task. Existing approaches impose fixed cognitive structures that enhance performance in specific tasks but lack adaptability across diverse scenarios. To address this limitation, we introduce METASCALE, a test-time scaling framework based on meta-thoughts -- adaptive thinking strategies tailored to each task. METASCALE initializes a pool of candidate meta-thoughts, then iteratively selects and evaluates them using a multi-armed bandit algorithm with upper confidence bound selection, guided by a reward model. To further enhance adaptability, a genetic algorithm evolves high-reward meta-thoughts, refining and extending the strategy pool over time. By dynamically proposing and optimizing meta-thoughts at inference time, METASCALE improves both accuracy and generalization across a wide range of tasks. Experimental results demonstrate that MetaScale consistently outperforms standard inference approaches, achieving an 11% performance gain in win rate on Arena-Hard for GPT-4o, surpassing o1-mini by 0.9% under style control. Notably, METASCALE scales more effectively with increasing sampling budgets and produces more structured, expert-level responses. 

---
# LLM-Match: An Open-Sourced Patient Matching Model Based on Large Language Models and Retrieval-Augmented Generation 

**Authors**: Xiaodi Li, Shaika Chowdhury, Chung Il Wi, Maria Vassilaki, Ken Liu, Terence T Sio, Owen Garrick, Young J Juhn, James R Cerhan, Cui Tao, Nansu Zong  

**Link**: [PDF](https://arxiv.org/pdf/2503.13281)  

**Abstract**: Patient matching is the process of linking patients to appropriate clinical trials by accurately identifying and matching their medical records with trial eligibility criteria. We propose LLM-Match, a novel framework for patient matching leveraging fine-tuned open-source large language models. Our approach consists of four key components. First, a retrieval-augmented generation (RAG) module extracts relevant patient context from a vast pool of electronic health records (EHRs). Second, a prompt generation module constructs input prompts by integrating trial eligibility criteria (both inclusion and exclusion criteria), patient context, and system instructions. Third, a fine-tuning module with a classification head optimizes the model parameters using structured prompts and ground-truth labels. Fourth, an evaluation module assesses the fine-tuned model's performance on the testing datasets. We evaluated LLM-Match on four open datasets, n2c2, SIGIR, TREC 2021, and TREC 2022, using open-source models, comparing it against TrialGPT, Zero-Shot, and GPT-4-based closed models. LLM-Match outperformed all baselines. 

---
# Faithfulness of LLM Self-Explanations for Commonsense Tasks: Larger Is Better, and Instruction-Tuning Allows Trade-Offs but Not Pareto Dominance 

**Authors**: Noah Y. Siegel, Nicolas Heess, Maria Perez-Ortiz, Oana-Maria Camburu  

**Link**: [PDF](https://arxiv.org/pdf/2503.13445)  

**Abstract**: As large language models (LLMs) become increasingly capable, ensuring that their self-generated explanations are faithful to their internal decision-making process is critical for safety and oversight. In this work, we conduct a comprehensive counterfactual faithfulness analysis across 62 models from 8 families, encompassing both pretrained and instruction-tuned variants and significantly extending prior studies of counterfactual tests. We introduce phi-CCT, a simplified variant of the Correlational Counterfactual Test, which avoids the need for token probabilities while explaining most of the variance of the original test. Our findings reveal clear scaling trends: larger models are consistently more faithful on our metrics. However, when comparing instruction-tuned and human-imitated explanations, we find that observed differences in faithfulness can often be attributed to explanation verbosity, leading to shifts along the true-positive/false-positive Pareto frontier. While instruction-tuning and prompting can influence this trade-off, we find limited evidence that they fundamentally expand the frontier of explanatory faithfulness beyond what is achievable with pretrained models of comparable size. Our analysis highlights the nuanced relationship between instruction-tuning, verbosity, and the faithful representation of model decision processes. 

---
# TablePilot; Recommending Human-Preferred Tabular Data Analysis with Large Language Models 

**Authors**: Deyin Yi, Yihao Liu, Lang Cao, Mengyu Zhou, Haoyu Dong, Shi Han, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13262)  

**Abstract**: Tabular data analysis is crucial in many scenarios, yet efficiently identifying the most relevant data analysis queries and results for a new table remains a significant challenge. The complexity of tabular data, diverse analytical operations, and the demand for high-quality analysis make the process tedious. To address these challenges, we aim to recommend query-code-result triplets tailored for new tables in tabular data analysis workflows. In this paper, we present TablePilot, a pioneering tabular data analysis framework leveraging large language models to autonomously generate comprehensive and superior analytical results without relying on user profiles or prior interactions. The framework incorporates key designs in analysis preparation and analysis optimization to enhance accuracy. Additionally, we propose Rec-Align, a novel method to further improve recommendation quality and better align with human preferences. Experiments on DART, a dataset specifically designed for comprehensive tabular data analysis recommendation, demonstrate the effectiveness of our framework. Based on GPT-4o, the tuned TablePilot achieves 77.0% top-5 recommendation recall. Human evaluations further highlight its effectiveness in optimizing tabular data analysis workflows. 

---
# Improving Complex Reasoning with Dynamic Prompt Corruption: A soft prompt Optimization Approach 

**Authors**: Sinan Fan, Liang Xie, Chen Shen, Ge Teng, Xiaosong Yuan, Xiaofeng Zhang, Chenxi Huang, Wenxiao Wang, Xiaofei He, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.13208)  

**Abstract**: Prompt-tuning (PT) for large language models (LLMs) can facilitate the performance on various conventional NLP tasks with significantly fewer trainable parameters. However, our investigation reveals that PT provides limited improvement and may even degrade the primitive performance of LLMs on complex reasoning tasks. Such a phenomenon suggests that soft prompts can positively impact certain instances while negatively affecting others, particularly during the later phases of reasoning. To address these challenges, We first identify an information accumulation within the soft prompts. Through detailed analysis, we demonstrate that this phenomenon is often accompanied by erroneous information flow patterns in the deeper layers of the model, which ultimately lead to incorrect reasoning outcomes. we propose a novel method called \textbf{D}ynamic \textbf{P}rompt \textbf{C}orruption (DPC) to take better advantage of soft prompts in complex reasoning tasks, which dynamically adjusts the influence of soft prompts based on their impact on the reasoning process. Specifically, DPC consists of two stages: Dynamic Trigger and Dynamic Corruption. First, Dynamic Trigger measures the impact of soft prompts, identifying whether beneficial or detrimental. Then, Dynamic Corruption mitigates the negative effects of soft prompts by selectively masking key tokens that interfere with the reasoning process. We validate the proposed approach through extensive experiments on various LLMs and reasoning tasks, including GSM8K, MATH, and AQuA. Experimental results demonstrate that DPC can consistently enhance the performance of PT, achieving 4\%-8\% accuracy gains compared to vanilla prompt tuning, highlighting the effectiveness of our approach and its potential to enhance complex reasoning in LLMs. 

---
# REPA: Russian Error Types Annotation for Evaluating Text Generation and Judgment Capabilities 

**Authors**: Alexander Pugachev, Alena Fenogenova, Vladislav Mikhailov, Ekaterina Artemova  

**Link**: [PDF](https://arxiv.org/pdf/2503.13102)  

**Abstract**: Recent advances in large language models (LLMs) have introduced the novel paradigm of using LLMs as judges, where an LLM evaluates and scores the outputs of another LLM, which often correlates highly with human preferences. However, the use of LLM-as-a-judge has been primarily studied in English. In this paper, we evaluate this framework in Russian by introducing the Russian Error tyPes Annotation dataset (REPA), a dataset of 1k user queries and 2k LLM-generated responses. Human annotators labeled each response pair expressing their preferences across ten specific error types, as well as selecting an overall preference. We rank six generative LLMs across the error types using three rating systems based on human preferences. We also evaluate responses using eight LLM judges in zero-shot and few-shot settings. We describe the results of analyzing the judges and position and length biases. Our findings reveal a notable gap between LLM judge performance in Russian and English. However, rankings based on human and LLM preferences show partial alignment, suggesting that while current LLM judges struggle with fine-grained evaluation in Russian, there is potential for improvement. 

---
# ClusComp: A Simple Paradigm for Model Compression and Efficient Finetuning 

**Authors**: Baohao Liao, Christian Herold, Seyyed Hadi Hashemi, Stefan Vasilev, Shahram Khadivi, Christof Monz  

**Link**: [PDF](https://arxiv.org/pdf/2503.13089)  

**Abstract**: As large language models (LLMs) scale, model compression is crucial for edge deployment and accessibility. Weight-only quantization reduces model size but suffers from performance degradation at lower bit widths. Moreover, standard finetuning is incompatible with quantized models, and alternative methods often fall short of full finetuning. In this paper, we propose ClusComp, a simple yet effective compression paradigm that clusters weight matrices into codebooks and finetunes them block-by-block. ClusComp (1) achieves superior performance in 2-4 bit quantization, (2) pushes compression to 1-bit while outperforming ultra-low-bit methods with minimal finetuning, and (3) enables efficient finetuning, even surpassing existing quantization-based approaches and rivaling full FP16 finetuning. Notably, ClusComp supports compression and finetuning of 70B LLMs on a single A6000-48GB GPU. 

---
# A Framework to Assess Multilingual Vulnerabilities of LLMs 

**Authors**: Likai Tang, Niruth Bogahawatta, Yasod Ginige, Jiarui Xu, Shixuan Sun, Surangika Ranathunga, Suranga Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2503.13081)  

**Abstract**: Large Language Models (LLMs) are acquiring a wider range of capabilities, including understanding and responding in multiple languages. While they undergo safety training to prevent them from answering illegal questions, imbalances in training data and human evaluation resources can make these models more susceptible to attacks in low-resource languages (LRL). This paper proposes a framework to automatically assess the multilingual vulnerabilities of commonly used LLMs. Using our framework, we evaluated six LLMs across eight languages representing varying levels of resource availability. We validated the assessments generated by our automated framework through human evaluation in two languages, demonstrating that the framework's results align with human judgments in most cases. Our findings reveal vulnerabilities in LRL; however, these may pose minimal risk as they often stem from the model's poor performance, resulting in incoherent responses. 

---
# A Multi-Stage Framework with Taxonomy-Guided Reasoning for Occupation Classification Using Large Language Models 

**Authors**: Palakorn Achananuparp, Ee-Peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2503.12989)  

**Abstract**: Automatically annotating job data with standardized occupations from taxonomies, known as occupation classification, is crucial for labor market analysis. However, this task is often hindered by data scarcity and the challenges of manual annotations. While large language models (LLMs) hold promise due to their extensive world knowledge and in-context learning capabilities, their effectiveness depends on their knowledge of occupational taxonomies, which remains unclear. In this study, we assess the ability of LLMs to generate precise taxonomic entities from taxonomy, highlighting their limitations. To address these challenges, we propose a multi-stage framework consisting of inference, retrieval, and reranking stages, which integrates taxonomy-guided reasoning examples to enhance performance by aligning outputs with taxonomic knowledge. Evaluations on a large-scale dataset show significant improvements in classification accuracy. Furthermore, we demonstrate the framework's adaptability for multi-label skill classification. Our results indicate that the framework outperforms existing LLM-based methods, offering a practical and scalable solution for occupation classification and related tasks across LLMs. 

---
# HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model 

**Authors**: Haiyang Guo, Fanhu Zeng, Ziwei Xiang, Fei Zhu, Da-Han Wang, Xu-Yao Zhang, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12941)  

**Abstract**: Instruction tuning is widely used to improve a pre-trained Multimodal Large Language Model (MLLM) by training it on curated task-specific datasets, enabling better comprehension of human instructions. However, it is infeasible to collect all possible instruction datasets simultaneously in real-world scenarios. Thus, enabling MLLM with continual instruction tuning is essential for maintaining their adaptability. However, existing methods often trade off memory efficiency for performance gains, significantly compromising overall efficiency. In this paper, we propose a task-specific expansion and task-general fusion framework based on the variations in Centered Kernel Alignment (CKA) similarity across different model layers when trained on diverse datasets. Furthermore, we analyze the information leakage present in the existing benchmark and propose a new and more challenging benchmark to rationally evaluate the performance of different methods. Comprehensive experiments showcase a significant performance improvement of our method compared to existing state-of-the-art methods. Our code will be public available. 

---
# HICD: Hallucination-Inducing via Attention Dispersion for Contrastive Decoding to Mitigate Hallucinations in Large Language Models 

**Authors**: Xinyan Jiang, Hang Ye, Yongxin Zhu, Xiaoying Zheng, Zikang Chen, Jun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.12908)  

**Abstract**: Large Language Models (LLMs) often generate hallucinations, producing outputs that are contextually inaccurate or factually incorrect. We introduce HICD, a novel method designed to induce hallucinations for contrastive decoding to mitigate hallucinations. Unlike existing contrastive decoding methods, HICD selects attention heads crucial to the model's prediction as inducing heads, then induces hallucinations by dispersing attention of these inducing heads and compares the hallucinated outputs with the original outputs to obtain the final result. Our approach significantly improves performance on tasks requiring contextual faithfulness, such as context completion, reading comprehension, and question answering. It also improves factuality in tasks requiring accurate knowledge recall. We demonstrate that our inducing heads selection and attention dispersion method leads to more "contrast-effective" hallucinations for contrastive decoding, outperforming other hallucination-inducing methods. Our findings provide a promising strategy for reducing hallucinations by inducing hallucinations in a controlled manner, enhancing the performance of LLMs in a wide range of tasks. 

---
# ThinkPatterns-21k: A Systematic Study on the Impact of Thinking Patterns in LLMs 

**Authors**: Pengcheng Wen, Jiaming Ji, Chi-Min Chan, Juntao Dai, Donghai Hong, Yaodong Yang, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2503.12918)  

**Abstract**: Large language models (LLMs) have demonstrated enhanced performance through the \textit{Thinking then Responding} paradigm, where models generate internal thoughts before final responses (aka, System 2 thinking). However, existing research lacks a systematic understanding of the mechanisms underlying how thinking patterns affect performance across model sizes. In this work, we conduct a comprehensive analysis of the impact of various thinking types on model performance and introduce ThinkPatterns-21k, a curated dataset comprising 21k instruction-response pairs (QA) collected from existing instruction-following datasets with five thinking types. For each pair, we augment it with five distinct internal thinking patterns: one unstructured thinking (monologue) and four structured variants (decomposition, self-ask, self-debate and self-critic), while maintaining the same instruction and response. Through extensive evaluation across different model sizes (3B-32B parameters), we have two key findings: (1) smaller models (<30B parameters) can benefit from most of structured thinking patterns, while larger models (32B) with structured thinking like decomposition would degrade performance and (2) unstructured monologue demonstrates broad effectiveness across different model sizes. Finally, we released all of our datasets, checkpoints, training logs of diverse thinking patterns to reproducibility, aiming to facilitate further research in this direction. 

---
# A Survey on Transformer Context Extension: Approaches and Evaluation 

**Authors**: Yijun Liu, Jinzheng Yu, Yang Xu, Zhongyang Li, Qingfu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.13299)  

**Abstract**: Large language models (LLMs) based on Transformer have been widely applied in the filed of natural language processing (NLP), demonstrating strong performance, particularly in handling short text tasks. However, when it comes to long context scenarios, the performance of LLMs degrades due to some challenges. To alleviate this phenomenon, there is a number of work proposed recently. In this survey, we first list the challenges of applying pre-trained LLMs to process long contexts. Then systematically review the approaches related to long context and propose our taxonomy categorizing them into four main types: positional encoding, context compression, retrieval augmented, and attention pattern. In addition to the approaches, we focus on the evaluation of long context, organizing relevant data, tasks, and metrics based on existing long context benchmarks. Finally, we summarize unresolved issues in the long context domain and put forward our views on future developments. 

---
# nvBench 2.0: A Benchmark for Natural Language to Visualization under Ambiguity 

**Authors**: Tianqi Luo, Chuhan Huang, Leixian Shen, Boyan Li, Shuyu Shen, Wei Zeng, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.12880)  

**Abstract**: Natural Language to Visualization (NL2VIS) enables users to create visualizations from natural language queries, making data insights more accessible. However, NL2VIS faces challenges in interpreting ambiguous queries, as users often express their visualization needs in imprecise language. To address this challenge, we introduce nvBench 2.0, a new benchmark designed to evaluate NL2VIS systems in scenarios involving ambiguous queries. nvBench 2.0 includes 7,878 natural language queries and 24,076 corresponding visualizations, derived from 780 tables across 153 domains. It is built using a controlled ambiguity-injection pipeline that generates ambiguous queries through a reverse-generation workflow. By starting with unambiguous seed visualizations and selectively injecting ambiguities, the pipeline yields multiple valid interpretations for each query, with each ambiguous query traceable to its corresponding visualization through step-wise reasoning paths. We evaluate various Large Language Models (LLMs) on their ability to perform ambiguous NL2VIS tasks using nvBench 2.0. We also propose Step-NL2VIS, an LLM-based model trained on nvBench 2.0, which enhances performance in ambiguous scenarios through step-wise preference optimization. Our results show that Step-NL2VIS outperforms all baselines, setting a new state-of-the-art for ambiguous NL2VIS tasks. 

---
# Enhancing LLM Reasoning with Iterative DPO: A Comprehensive Empirical Investigation 

**Authors**: Songjun Tu, Jiahao Lin, Xiangyu Tian, Qichao Zhang, Linjing Li, Yuqian Fu, Nan Xu, Wei He, Xiangyuan Lan, Dongmei Jiang, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.12854)  

**Abstract**: Recent advancements in post-training methodologies for large language models (LLMs) have highlighted reinforcement learning (RL) as a critical component for enhancing reasoning. However, the substantial computational costs associated with RL-based approaches have led to growing interest in alternative paradigms, such as Direct Preference Optimization (DPO). In this study, we investigate the effectiveness of DPO in facilitating self-improvement for LLMs through iterative preference-based learning. We demonstrate that a single round of DPO with coarse filtering significantly enhances mathematical reasoning performance, particularly for strong base model. Furthermore, we design an iterative enhancement framework for both the generator and the reward model (RM), enabling their mutual improvement through online interaction across multiple rounds of DPO. Finally, with simple verifiable rewards, our model DPO-VP achieves RL-level performance with significantly lower computational overhead. These findings highlight DPO as a scalable and cost-effective alternative to RL, offering a practical solution for enhancing LLM reasoning in resource-constrained situations. 

---
# Can Language Models Follow Multiple Turns of Entangled Instructions? 

**Authors**: Chi Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.13222)  

**Abstract**: Despite significant achievements in improving the instruction-following capabilities of large language models (LLMs), the ability to process multiple potentially entangled or conflicting instructions remains a considerable challenge. Real-world scenarios often require consistency across multiple instructions over time, such as secret privacy, personal preferences, and prioritization, which demand sophisticated abilities to integrate multiple turns and carefully balance competing objectives when instructions intersect or conflict. This work presents a systematic investigation of LLMs' capabilities in handling multiple turns of instructions, covering three levels of difficulty: (1) retrieving information from instructions, (2) tracking and reasoning across turns, and (3) resolving conflicts among instructions. We construct MultiTurnInstruct with around 1.1K high-quality multi-turn conversations through the human-in-the-loop approach and result in nine capability categories, including statics and dynamics, reasoning, and multitasking. Our finding reveals an intriguing trade-off between different capabilities. While GPT models demonstrate superior memorization, they show reduced effectiveness in privacy-protection tasks requiring selective information withholding. Larger models exhibit stronger reasoning capabilities but still struggle with resolving conflicting instructions. Importantly, these performance gaps cannot be attributed solely to information loss, as models demonstrate strong BLEU scores on memorization tasks but their attention mechanisms fail to integrate multiple related instructions effectively. These findings highlight critical areas for improvement in complex real-world tasks involving multi-turn instructions. 

---
# Plausibility Vaccine: Injecting LLM Knowledge for Event Plausibility 

**Authors**: Jacob Chmura, Jonah Dauvet, Sebastian Sabry  

**Link**: [PDF](https://arxiv.org/pdf/2503.12667)  

**Abstract**: Despite advances in language modelling, distributional methods that build semantic representations from co-occurrences fail to discriminate between plausible and implausible events. In this work, we investigate how plausibility prediction can be improved by injecting latent knowledge prompted from large language models using parameter-efficient fine-tuning. We train 12 task adapters to learn various physical properties and association measures and perform adapter fusion to compose latent semantic knowledge from each task on top of pre-trained AlBERT embeddings. We automate auxiliary task data generation, which enables us to scale our approach and fine-tune our learned representations across two plausibility datasets. Our code is available at this https URL. 

---
# RaSA: Rank-Sharing Low-Rank Adaptation 

**Authors**: Zhiwei He, Zhaopeng Tu, Xing Wang, Xingyu Chen, Zhijie Wang, Jiahao Xu, Tian Liang, Wenxiang Jiao, Zhuosheng Zhang, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12576)  

**Abstract**: Low-rank adaptation (LoRA) has been prominently employed for parameter-efficient fine-tuning of large language models (LLMs). However, the limited expressive capacity of LoRA, stemming from the low-rank constraint, has been recognized as a bottleneck, particularly in rigorous tasks like code generation and mathematical reasoning. To address this limitation, we introduce Rank-Sharing Low-Rank Adaptation (RaSA), an innovative extension that enhances the expressive capacity of LoRA by leveraging partial rank sharing across layers. By forming a shared rank pool and applying layer-specific weighting, RaSA effectively increases the number of ranks without augmenting parameter overhead. Our theoretically grounded and empirically validated approach demonstrates that RaSA not only maintains the core advantages of LoRA but also significantly boosts performance in challenging code and math tasks. Code, data and scripts are available at: this https URL. 

---
# Code-Driven Inductive Synthesis: Enhancing Reasoning Abilities of Large Language Models with Sequences 

**Authors**: Kedi Chen, Zhikai Lei, Fan Zhang, Yinqi Zhang, Qin Chen, Jie Zhou, Liang He, Qipeng Guo, Kai Chen, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13109)  

**Abstract**: Large language models make remarkable progress in reasoning capabilities. Existing works focus mainly on deductive reasoning tasks (e.g., code and math), while another type of reasoning mode that better aligns with human learning, inductive reasoning, is not well studied. We attribute the reason to the fact that obtaining high-quality process supervision data is challenging for inductive reasoning. Towards this end, we novelly employ number sequences as the source of inductive reasoning data. We package sequences into algorithmic problems to find the general term of each sequence through a code solution. In this way, we can verify whether the code solution holds for any term in the current sequence, and inject case-based supervision signals by using code unit tests. We build a sequence synthetic data pipeline and form a training dataset CodeSeq. Experimental results show that the models tuned with CodeSeq improve on both code and comprehensive reasoning benchmarks. 

---
# EXAONE Deep: Reasoning Enhanced Language Models 

**Authors**: LG AI Research, Kyunghoon Bae, Eunbi Choi, Kibong Choi, Stanley Jungkyu Choi, Yemuk Choi, Seokhee Hong, Junwon Hwang, Hyojin Jeon, Kijeong Jeon, Gerrard Jeongwon Jo, Hyunjik Jo, Jiyeon Jung, Hyosang Kim, Joonkee Kim, Seonghwan Kim, Soyeon Kim, Sunkyoung Kim, Yireun Kim, Yongil Kim, Youchul Kim, Edward Hwayoung Lee, Haeju Lee, Honglak Lee, Jinsik Lee, Kyungmin Lee, Sangha Park, Yongmin Park, Sihoon Yang, Heuiyeen Yeen, Sihyuk Yi, Hyeongu Yun  

**Link**: [PDF](https://arxiv.org/pdf/2503.12524)  

**Abstract**: We present EXAONE Deep series, which exhibits superior capabilities in various reasoning tasks, including math and coding benchmarks. We train our models mainly on the reasoning-specialized dataset that incorporates long streams of thought processes. Evaluation results show that our smaller models, EXAONE Deep 2.4B and 7.8B, outperform other models of comparable size, while the largest model, EXAONE Deep 32B, demonstrates competitive performance against leading open-weight models. All EXAONE Deep models are openly available for research purposes and can be downloaded from this https URL 

---
# HKCanto-Eval: A Benchmark for Evaluating Cantonese Language Understanding and Cultural Comprehension in LLMs 

**Authors**: Tsz Chung Cheng, Chung Shing Cheng, Chaak Ming Lau, Eugene Tin-Ho Lam, Chun Yat Wong, Hoi On Yu, Cheuk Hei Chong  

**Link**: [PDF](https://arxiv.org/pdf/2503.12440)  

**Abstract**: The ability of language models to comprehend and interact in diverse linguistic and cultural landscapes is crucial. The Cantonese language used in Hong Kong presents unique challenges for natural language processing due to its rich cultural nuances and lack of dedicated evaluation datasets. The HKCanto-Eval benchmark addresses this gap by evaluating the performance of large language models (LLMs) on Cantonese language understanding tasks, extending to English and Written Chinese for cross-lingual evaluation. HKCanto-Eval integrates cultural and linguistic nuances intrinsic to Hong Kong, providing a robust framework for assessing language models in realistic scenarios. Additionally, the benchmark includes questions designed to tap into the underlying linguistic metaknowledge of the models. Our findings indicate that while proprietary models generally outperform open-weight models, significant limitations remain in handling Cantonese-specific linguistic and cultural knowledge, highlighting the need for more targeted training data and evaluation methods. The code can be accessed at this https URL 

---
# General Table Question Answering via Answer-Formula Joint Generation 

**Authors**: Zhongyuan Wang, Richong Zhang, Zhijie Nie  

**Link**: [PDF](https://arxiv.org/pdf/2503.12345)  

**Abstract**: Advanced table question answering (TableQA) methods prompt large language models (LLMs) to generate answer text, SQL query, Python code, or custom operations, which impressively improve the complex reasoning problems in the TableQA task. However, these methods lack the versatility to cope with specific question types or table structures. In contrast, the Spreadsheet Formula, the widely-used and well-defined operation language for tabular data, has not been thoroughly explored to solve TableQA. In this paper, we first attempt to use Formula as the logical form for solving complex reasoning on the tables with different structures. Specifically, we construct a large Formula-annotated TableQA dataset \texttt{FromulaQA} from existing datasets. In addition, we propose \texttt{TabAF}, a general table answering framework to solve multiple types of tasks over multiple types of tables simultaneously. Unlike existing methods, \texttt{TabAF} decodes answers and Formulas with a single LLM backbone, demonstrating great versatility and generalization. \texttt{TabAF} based on Llama3.1-70B achieves new state-of-the-art performance on the WikiTableQuestion, HiTab and TabFact. 

---
# SVD-LLM V2: Optimizing Singular Value Truncation for Large Language Model Compression 

**Authors**: Xin Wang, Samiul Alam, Zhongwei Wan, Hui Shen, Mi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12340)  

**Abstract**: Despite significant advancements, the practical deployment of Large Language Models (LLMs) is often hampered by their immense sizes, highlighting the need for effective compression techniques. Singular Value Decomposition (SVD) is a promising LLM compression technique. However, existing SVD-based compression methods fall short in reducing truncation losses, leading to less competitive performance in compressed models. In this work, we introduce SVD-LLM V2, a SVD-based LLM compression method that optimizes singular value truncation in SVD compression with two techniques. First, SVD-LLM V2 proposes to use theoretical truncation loss of weight matrices to assign a unique compression ratio to each weight matrix at different layers to accommodate weight redundancy heterogeneity. Second, SVD-LLM V2 proposes loss-optimized weight truncation to ensure that the truncated singular values result in a lower and more stable truncation loss in practice. We evaluate SVD-LLM V2 on ten datasets and five LLMs at various scales. Our results show SVD-LLM V2 outperforms state-of-the-art SVD-based LLM compression methods. Our code is available at this https URL 

---
# From Guessing to Asking: An Approach to Resolving the Persona Knowledge Gap in LLMs during Multi-Turn Conversations 

**Authors**: Sarvesh Baskar, Tanmay Tulsidas Verelakar, Srinivasan Parthasarathy, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2503.12556)  

**Abstract**: In multi-turn dialogues, large language models (LLM) face a critical challenge of ensuring coherence while adapting to user-specific information. This study introduces the persona knowledge gap, the discrepancy between a model's internal understanding and the knowledge required for coherent, personalized conversations. While prior research has recognized these gaps, computational methods for their identification and resolution remain underexplored. We propose Conversation Preference Elicitation and Recommendation (CPER), a novel framework that dynamically detects and resolves persona knowledge gaps using intrinsic uncertainty quantification and feedback-driven refinement. CPER consists of three key modules: a Contextual Understanding Module for preference extraction, a Dynamic Feedback Module for measuring uncertainty and refining persona alignment, and a Persona-Driven Response Generation module for adapting responses based on accumulated user context. We evaluate CPER on two real-world datasets: CCPE-M for preferential movie recommendations and ESConv for mental health support. Using A/B testing, human evaluators preferred CPER's responses 42% more often than baseline models in CCPE-M and 27% more often in ESConv. A qualitative human evaluation confirms that CPER's responses are preferred for maintaining contextual relevance and coherence, particularly in longer (12+ turn) conversations. 

---
# Interpretation Gaps in LLM-Assisted Comprehension of Privacy Documents 

**Authors**: Rinku Dewri  

**Link**: [PDF](https://arxiv.org/pdf/2503.12225)  

**Abstract**: This article explores the gaps that can manifest when using a large language model (LLM) to obtain simplified interpretations of data practices from a complex privacy policy. We exemplify these gaps to showcase issues in accuracy, completeness, clarity and representation, while advocating for continued research to realize an LLM's true potential in revolutionizing privacy management through personal assistants and automated compliance checking. 

---
# Improving LLM-based Document-level Machine Translation with Multi-Knowledge Fusion 

**Authors**: Bin Liu, Xinglin Lyu, Junhui Li, Daimeng Wei, Min Zhang, Shimin Tao, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12152)  

**Abstract**: Recent studies in prompting large language model (LLM) for document-level machine translation (DMT) primarily focus on the inter-sentence context by flatting the source document into a long sequence. This approach relies solely on the sequence of sentences within the document. However, the complexity of document-level sequences is greater than that of shorter sentence-level sequences, which may limit LLM's ability in DMT when only this single-source knowledge is used. In this paper, we propose an enhanced approach by incorporating multiple sources of knowledge, including both the document summarization and entity translation, to enhance the performance of LLM-based DMT. Given a source document, we first obtain its summarization and translation of entities via LLM as the additional knowledge. We then utilize LLMs to generate two translations of the source document by fusing these two single knowledge sources, respectively. Finally, recognizing that different sources of knowledge may aid or hinder the translation of different sentences, we refine and rank the translations by leveraging a multi-knowledge fusion strategy to ensure the best results. Experimental results in eight document-level translation tasks show that our approach achieves an average improvement of 0.8, 0.6, and 0.4 COMET scores over the baseline without extra knowledge for LLaMA3-8B-Instruct, Mistral-Nemo-Instruct, and GPT-4o-mini, respectively. 

---
# PLM: Efficient Peripheral Language Models Hardware-Co-Designed for Ubiquitous Computing 

**Authors**: Cheng Deng, Luoyang Sun, Jiwen Jiang, Yongcheng Zeng, Xinjian Wu, Wenxin Zhao, Qingfa Xiao, Jiachuan Wang, Lei Chen, Lionel M. Ni, Haifeng Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12167)  

**Abstract**: While scaling laws have been continuously validated in large language models (LLMs) with increasing model parameters, the inherent tension between the inference demands of LLMs and the limited resources of edge devices poses a critical challenge to the development of edge intelligence. Recently, numerous small language models have emerged, aiming to distill the capabilities of LLMs into smaller footprints. However, these models often retain the fundamental architectural principles of their larger counterparts, still imposing considerable strain on the storage and bandwidth capacities of edge devices. In this paper, we introduce the PLM, a Peripheral Language Model, developed through a co-design process that jointly optimizes model architecture and edge system constraints. The PLM utilizes a Multi-head Latent Attention mechanism and employs the squared ReLU activation function to encourage sparsity, thereby reducing peak memory footprint during inference. During training, we collect and reorganize open-source datasets, implement a multi-phase training strategy, and empirically investigate the Warmup-Stable-Decay-Constant (WSDC) learning rate scheduler. Additionally, we incorporate Reinforcement Learning from Human Feedback (RLHF) by adopting the ARIES preference learning approach. Following a two-phase SFT process, this method yields performance gains of 2% in general tasks, 9% in the GSM8K task, and 11% in coding tasks. In addition to its novel architecture, evaluation results demonstrate that PLM outperforms existing small language models trained on publicly available data while maintaining the lowest number of activated parameters. Furthermore, deployment across various edge devices, including consumer-grade GPUs, mobile phones, and Raspberry Pis, validates PLM's suitability for peripheral applications. The PLM series models are publicly available at this https URL. 

---
# CAKE: Cascading and Adaptive KV Cache Eviction with Layer Preferences 

**Authors**: Ziran Qin, Yuchen Cao, Mingbao Lin, Wen Hu, Shixuan Fan, Ke Cheng, Weiyao Lin, Jianguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.12491)  

**Abstract**: Large language models (LLMs) excel at processing long sequences, boosting demand for key-value (KV) caching. While recent efforts to evict KV cache have alleviated the inference burden, they often fail to allocate resources rationally across layers with different attention patterns. In this paper, we introduce Cascading and Adaptive KV cache Eviction (CAKE), a novel approach that frames KV cache eviction as a "cake-slicing problem." CAKE assesses layer-specific preferences by considering attention dynamics in both spatial and temporal dimensions, allocates rational cache size for layers accordingly, and manages memory constraints in a cascading manner. This approach enables a global view of cache allocation, adaptively distributing resources across diverse attention mechanisms while maintaining memory budgets. CAKE also employs a new eviction indicator that considers the shifting importance of tokens over time, addressing limitations in existing methods that overlook temporal dynamics. Comprehensive experiments on LongBench and NeedleBench show that CAKE maintains model performance with only 3.2% of the KV cache and consistently outperforms current baselines across various models and memory constraints, particularly in low-memory settings. Additionally, CAKE achieves over 10x speedup in decoding latency compared to full cache when processing contexts of 128K tokens with FlashAttention-2. Our code is available at this https URL. 

---
# Large Language Models in Legislative Content Analysis: A Dataset from the Polish Parliament 

**Authors**: Arkadiusz Bryłkowski, Jakub Klikowski  

**Link**: [PDF](https://arxiv.org/pdf/2503.12100)  

**Abstract**: Large language models (LLMs) are among the best methods for processing natural language, partly due to their versatility. At the same time, domain-specific LLMs are more practical in real-life applications. This work introduces a novel natural language dataset created by acquired data from official legislative authorities' websites. The study focuses on formulating three natural language processing (NLP) tasks to evaluate the effectiveness of LLMs on legislative content analysis within the context of the Polish legal system. Key findings highlight the potential of LLMs in automating and enhancing legislative content analysis while emphasizing specific challenges, such as understanding legal context. The research contributes to the advancement of NLP in the legal field, particularly in the Polish language. It has been demonstrated that even commonly accessible data can be practically utilized for legislative content analysis. 

---
# Information-Guided Identification of Training Data Imprint in (Proprietary) Large Language Models 

**Authors**: Abhilasha Ravichander, Jillian Fisher, Taylor Sorensen, Ximing Lu, Yuchen Lin, Maria Antoniak, Niloofar Mireshghallah, Chandra Bhagavatula, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2503.12072)  

**Abstract**: High-quality training data has proven crucial for developing performant large language models (LLMs). However, commercial LLM providers disclose few, if any, details about the data used for training. This lack of transparency creates multiple challenges: it limits external oversight and inspection of LLMs for issues such as copyright infringement, it undermines the agency of data authors, and it hinders scientific research on critical issues such as data contamination and data selection. How can we recover what training data is known to LLMs? In this work, we demonstrate a new method to identify training data known to proprietary LLMs like GPT-4 without requiring any access to model weights or token probabilities, by using information-guided probes. Our work builds on a key observation: text passages with high surprisal are good search material for memorization probes. By evaluating a model's ability to successfully reconstruct high-surprisal tokens in text, we can identify a surprising number of texts memorized by LLMs. 

---
# Applications of Large Language Model Reasoning in Feature Generation 

**Authors**: Dharani Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2503.11989)  

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing through their state of art reasoning capabilities. This paper explores the convergence of LLM reasoning techniques and feature generation for machine learning tasks. We examine four key reasoning approaches: Chain of Thought, Tree of Thoughts, Retrieval-Augmented Generation, and Thought Space Exploration. Our analysis reveals how these approaches can be used to identify effective feature generation rules without having to manually specify search spaces. The paper categorizes LLM-based feature generation methods across various domains including finance, healthcare, and text analytics. LLMs can extract key information from clinical notes and radiology reports in healthcare, by enabling more efficient data utilization. In finance, LLMs facilitate text generation, summarization, and entity extraction from complex documents. We analyze evaluation methodologies for assessing feature quality and downstream performance, with particular attention to OCTree's decision tree reasoning approach that provides language-based feedback for iterative improvements. Current challenges include hallucination, computational efficiency, and domain adaptation. As of March 2025, emerging approaches include inference-time compute scaling, reinforcement learning, and supervised fine-tuning with model distillation. Future directions point toward multimodal feature generation, self-improving systems, and neuro-symbolic approaches. This paper provides a detailed overview of an emerging field that promises to automate and enhance feature engineering through language model reasoning. 

---
# No LLM is Free From Bias: A Comprehensive Study of Bias Evaluation in Large Language models 

**Authors**: Charaka Vinayak Kumar, Ashok Urlana, Gopichand Kanumolu, Bala Mallikarjunarao Garlapati, Pruthwik Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2503.11985)  

**Abstract**: Advancements in Large Language Models (LLMs) have increased the performance of different natural language understanding as well as generation tasks. Although LLMs have breached the state-of-the-art performance in various tasks, they often reflect different forms of bias present in the training data. In the light of this perceived limitation, we provide a unified evaluation of benchmarks using a set of representative LLMs that cover different forms of biases starting from physical characteristics to socio-economic categories. Moreover, we propose five prompting approaches to carry out the bias detection task across different aspects of bias. Further, we formulate three research questions to gain valuable insight in detecting biases in LLMs using different approaches and evaluation metrics across benchmarks. The results indicate that each of the selected LLMs suffer from one or the other form of bias with the LLaMA3.1-8B model being the least biased. Finally, we conclude the paper with the identification of key challenges and possible future directions. 

---
# RECSIP: REpeated Clustering of Scores Improving the Precision 

**Authors**: André Schamschurko, Nenad Petrovic, Alois Christian Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.12108)  

**Abstract**: The latest research on Large Language Models (LLMs) has demonstrated significant advancement in the field of Natural Language Processing (NLP). However, despite this progress, there is still a lack of reliability in these models. This is due to the stochastic architecture of LLMs, which presents a challenge for users attempting to ascertain the reliability of a model's response. These responses may cause serious harm in high-risk environments or expensive failures in industrial contexts. Therefore, we introduce the framework REpeated Clustering of Scores Improving the Precision (RECSIP) which focuses on improving the precision of LLMs by asking multiple models in parallel, scoring and clustering their responses to ensure a higher reliability on the response. The evaluation of our reference implementation recsip on the benchmark MMLU-Pro using the models GPT-4o, Claude and Gemini shows an overall increase of 5.8 per cent points compared to the best used model. 

---
# Integration of Explainable AI Techniques with Large Language Models for Enhanced Interpretability for Sentiment Analysis 

**Authors**: Thivya Thogesan, Anupiya Nugaliyadde, Kok Wai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2503.11948)  

**Abstract**: Interpretability remains a key difficulty in sentiment analysis with Large Language Models (LLMs), particularly in high-stakes applications where it is crucial to comprehend the rationale behind forecasts. This research addressed this by introducing a technique that applies SHAP (Shapley Additive Explanations) by breaking down LLMs into components such as embedding layer,encoder,decoder and attention layer to provide a layer-by-layer knowledge of sentiment prediction. The approach offers a clearer overview of how model interpret and categorise sentiment by breaking down LLMs into these parts. The method is evaluated using the Stanford Sentiment Treebank (SST-2) dataset, which shows how different sentences affect different layers. The effectiveness of layer-wise SHAP analysis in clarifying sentiment-specific token attributions is demonstrated by experimental evaluations, which provide a notable enhancement over current whole-model explainability techniques. These results highlight how the suggested approach could improve the reliability and transparency of LLM-based sentiment analysis in crucial applications. 

---
# LAG-MMLU: Benchmarking Frontier LLM Understanding in Latvian and Giriama 

**Authors**: Naome A. Etori, Kevin Lu, Randu Karisa, Arturs Kanepajs  

**Link**: [PDF](https://arxiv.org/pdf/2503.11911)  

**Abstract**: As large language models (LLMs) rapidly advance, evaluating their performance is critical. LLMs are trained on multilingual data, but their reasoning abilities are mainly evaluated using English datasets. Hence, robust evaluation frameworks are needed using high-quality non-English datasets, especially low-resource languages (LRLs). This study evaluates eight state-of-the-art (SOTA) LLMs on Latvian and Giriama using a Massive Multitask Language Understanding (MMLU) subset curated with native speakers for linguistic and cultural relevance. Giriama is benchmarked for the first time. Our evaluation shows that OpenAI's o1 model outperforms others across all languages, scoring 92.8\% in English, 88.8\% in Latvian, and 70.8\% in Giriama on 0-shot tasks. Mistral-large (35.6\%) and Llama-70B IT (41\%) have weak performance, on both Latvian and Giriama. Our results underscore the need for localized benchmarks and human evaluations in advancing cultural AI contextualization. 

---
# LLMs for Translation: Historical, Low-Resourced Languages and Contemporary AI Models 

**Authors**: Merve Tekgurler  

**Link**: [PDF](https://arxiv.org/pdf/2503.11898)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable adaptability in performing various tasks, including machine translation (MT), without explicit training. Models such as OpenAI's GPT-4 and Google's Gemini are frequently evaluated on translation benchmarks and utilized as translation tools due to their high performance. This paper examines Gemini's performance in translating an 18th-century Ottoman Turkish manuscript, Prisoner of the Infidels: The Memoirs of Osman Agha of Timisoara, into English. The manuscript recounts the experiences of Osman Agha, an Ottoman subject who spent 11 years as a prisoner of war in Austria, and includes his accounts of warfare and violence. Our analysis reveals that Gemini's safety mechanisms flagged between 14 and 23 percent of the manuscript as harmful, resulting in untranslated passages. These safety settings, while effective in mitigating potential harm, hinder the model's ability to provide complete and accurate translations of historical texts. Through real historical examples, this study highlights the inherent challenges and limitations of current LLM safety implementations in the handling of sensitive and context-rich materials. These real-world instances underscore potential failures of LLMs in contemporary translation scenarios, where accurate and comprehensive translations are crucial-for example, translating the accounts of modern victims of war for legal proceedings or humanitarian documentation. 

---
# Resolving UnderEdit & OverEdit with Iterative & Neighbor-Assisted Model Editing 

**Authors**: Bhiman Kumar Baghel, Scott M. Jordan, Zheyuan Ryan Shi, Xiang Lorraine Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.11895)  

**Abstract**: Large Language Models (LLMs) are used in various downstream language tasks, making it crucial to keep their knowledge up-to-date, but both retraining and fine-tuning the model can be costly. Model editing offers an efficient and effective alternative by a single update to only a key subset of model parameters. While being efficient, these methods are not perfect. Sometimes knowledge edits are unsuccessful, i.e., UnderEdit, or the edit contaminated neighboring knowledge that should remain unchanged, i.e., OverEdit. To address these limitations, we propose iterative model editing, based on our hypothesis that a single parameter update is often insufficient, to mitigate UnderEdit, and neighbor-assisted model editing, which incorporates neighboring knowledge during editing to minimize OverEdit. Extensive experiments demonstrate that our methods effectively reduce UnderEdit up to 38 percentage points and OverEdit up to 6 percentage points across multiple model editing algorithms, LLMs, and benchmark datasets. 

---
# GPT's Devastated and LLaMA's Content: Emotion Representation Alignment in LLMs for Keyword-based Generation 

**Authors**: Shadab Choudhury, Asha Kumar, Lara J. Martin  

**Link**: [PDF](https://arxiv.org/pdf/2503.11881)  

**Abstract**: In controlled text generation using large language models (LLMs), gaps arise between the language model's interpretation and human expectations. We look at the problem of controlling emotions in keyword-based sentence generation for both GPT-4 and LLaMA-3. We selected four emotion representations: Words, Valence-Arousal-Dominance (VAD) dimensions expressed in both Lexical and Numeric forms, and Emojis. Our human evaluation looked at the Human-LLM alignment for each representation, as well as the accuracy and realism of the generated sentences. While representations like VAD break emotions into easy-to-compute components, our findings show that people agree more with how LLMs generate when conditioned on English words (e.g., "angry") rather than VAD scales. This difference is especially visible when comparing Numeric VAD to words. However, we found that converting the originally-numeric VAD scales to Lexical scales (e.g., +4.0 becomes "High") dramatically improved agreement. Furthermore, the perception of how much a generated sentence conveys an emotion is highly dependent on the LLM, representation type, and which emotion it is. 

---
# OpeNLGauge: An Explainable Metric for NLG Evaluation with Open-Weights LLMs 

**Authors**: Ivan Kartáč, Mateusz Lango, Ondřej Dušek  

**Link**: [PDF](https://arxiv.org/pdf/2503.11858)  

**Abstract**: Large Language Models (LLMs) have demonstrated great potential as evaluators of NLG systems, allowing for high-quality, reference-free, and multi-aspect assessments. However, existing LLM-based metrics suffer from two major drawbacks: reliance on proprietary models to generate training data or perform evaluations, and a lack of fine-grained, explanatory feedback. In this paper, we introduce OpeNLGauge, a fully open-source, reference-free NLG evaluation metric that provides accurate explanations based on error spans. OpeNLGauge is available as a two-stage ensemble of larger open-weight LLMs, or as a small fine-tuned evaluation model, with confirmed generalizability to unseen tasks, domains and aspects. Our extensive meta-evaluation shows that OpeNLGauge achieves competitive correlation with human judgments, outperforming state-of-the-art models on certain tasks while maintaining full reproducibility and providing explanations more than twice as accurate. 

---
# Bridging the LLM Accessibility Divide? Performance, Fairness, and Cost of Closed versus Open LLMs for Automated Essay Scoring 

**Authors**: Kezia Oketch, John P. Lalor, Yi Yang, Ahmed Abbasi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11827)  

**Abstract**: Closed large language models (LLMs) such as GPT-4 have set state-of-the-art results across a number of NLP tasks and have become central to NLP and machine learning (ML)-driven solutions. Closed LLMs' performance and wide adoption has sparked considerable debate about their accessibility in terms of availability, cost, and transparency. In this study, we perform a rigorous comparative analysis of nine leading LLMs, spanning closed, open, and open-source LLM ecosystems, across text assessment and generation tasks related to automated essay scoring. Our findings reveal that for few-shot learning-based assessment of human generated essays, open LLMs such as Llama 3 and Qwen2.5 perform comparably to GPT-4 in terms of predictive performance, with no significant differences in disparate impact scores when considering age- or race-related fairness. Moreover, Llama 3 offers a substantial cost advantage, being up to 37 times more cost-efficient than GPT-4. For generative tasks, we find that essays generated by top open LLMs are comparable to closed LLMs in terms of their semantic composition/embeddings and ML assessed scores. Our findings challenge the dominance of closed LLMs and highlight the democratizing potential of open LLMs, suggesting they can effectively bridge accessibility divides while maintaining competitive performance and fairness. 

---
# HInter: Exposing Hidden Intersectional Bias in Large Language Models 

**Authors**: Badr Souani, Ezekiel Soremekun, Mike Papadakis, Setsuko Yokoyama, Sudipta Chattopadhyay, Yves Le Traon  

**Link**: [PDF](https://arxiv.org/pdf/2503.11962)  

**Abstract**: Large Language Models (LLMs) may portray discrimination towards certain individuals, especially those characterized by multiple attributes (aka intersectional bias). Discovering intersectional bias in LLMs is challenging, as it involves complex inputs on multiple attributes (e.g. race and gender). To address this challenge, we propose HInter, a test technique that synergistically combines mutation analysis, dependency parsing and metamorphic oracles to automatically detect intersectional bias in LLMs. HInter generates test inputs by systematically mutating sentences using multiple mutations, validates inputs via a dependency invariant and detects biases by checking the LLM response on the original and mutated sentences. We evaluate HInter using six LLM architectures and 18 LLM models (GPT3.5, Llama2, BERT, etc) and find that 14.61% of the inputs generated by HInter expose intersectional bias. Results also show that our dependency invariant reduces false positives (incorrect test inputs) by an order of magnitude. Finally, we observed that 16.62% of intersectional bias errors are hidden, meaning that their corresponding atomic cases do not trigger biases. Overall, this work emphasize the importance of testing LLMs for intersectional bias. 

---
# LogitLens4LLMs: Extending Logit Lens Analysis to Modern Large Language Models 

**Authors**: Zhenyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11667)  

**Abstract**: This paper introduces LogitLens4LLMs, a toolkit that extends the Logit Lens technique to modern large language models. While Logit Lens has been a crucial method for understanding internal representations of language models, it was previously limited to earlier model architectures. Our work overcomes the limitations of existing implementations, enabling the technique to be applied to state-of-the-art architectures (such as Qwen-2.5 and Llama-3.1) while automating key analytical workflows. By developing component-specific hooks to capture both attention mechanisms and MLP outputs, our implementation achieves full compatibility with the HuggingFace transformer library while maintaining low inference overhead. The toolkit provides both interactive exploration and batch processing capabilities, supporting large-scale layer-wise analyses. Through open-sourcing our implementation, we aim to facilitate deeper investigations into the internal mechanisms of large-scale language models. The toolkit is openly available at this https URL. 

---
# Integrating Chain-of-Thought and Retrieval Augmented Generation Enhances Rare Disease Diagnosis from Clinical Notes 

**Authors**: Da Wu, Zhanliang Wang, Quan Nguyen, Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12286)  

**Abstract**: Background: Several studies show that large language models (LLMs) struggle with phenotype-driven gene prioritization for rare diseases. These studies typically use Human Phenotype Ontology (HPO) terms to prompt foundation models like GPT and LLaMA to predict candidate genes. However, in real-world settings, foundation models are not optimized for domain-specific tasks like clinical diagnosis, yet inputs are unstructured clinical notes rather than standardized terms. How LLMs can be instructed to predict candidate genes or disease diagnosis from unstructured clinical notes remains a major challenge. Methods: We introduce RAG-driven CoT and CoT-driven RAG, two methods that combine Chain-of-Thought (CoT) and Retrieval Augmented Generation (RAG) to analyze clinical notes. A five-question CoT protocol mimics expert reasoning, while RAG retrieves data from sources like HPO and OMIM (Online Mendelian Inheritance in Man). We evaluated these approaches on rare disease datasets, including 5,980 Phenopacket-derived notes, 255 literature-based narratives, and 220 in-house clinical notes from Childrens Hospital of Philadelphia. Results: We found that recent foundations models, including Llama 3.3-70B-Instruct and DeepSeek-R1-Distill-Llama-70B, outperformed earlier versions such as Llama 2 and GPT-3.5. We also showed that RAG-driven CoT and CoT-driven RAG both outperform foundation models in candidate gene prioritization from clinical notes; in particular, both methods with DeepSeek backbone resulted in a top-10 gene accuracy of over 40% on Phenopacket-derived clinical notes. RAG-driven CoT works better for high-quality notes, where early retrieval can anchor the subsequent reasoning steps in domain-specific evidence, while CoT-driven RAG has advantage when processing lengthy and noisy notes. 

---
# Automating Mathematical Proof Generation Using Large Language Model Agents and Knowledge Graphs 

**Authors**: Vincent Li, Yule Fu, Tim Knappe, Kevin Han, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11657)  

**Abstract**: Large Language Models have demonstrated remarkable capabilities in natural language processing tasks, including mathematical problem-solving that requires multi-step logical reasoning. However, challenges persist in automating the identification of key mathematical concepts, understanding their interrelations, and formalizing proofs within a rigorous framework. We present a novel framework that leverages knowledge graphs to augment LLMs to construct and formalize mathematical proofs. Our results demonstrate significant performance improvements across multiple datasets, with using knowledge graphs, achieving up to a 34% success rate on the MUSTARDSAUCE dataset on o1-mini and consistently outperforming baseline approaches by 2-11% across different models. We show how this approach bridges the gap between natural language understanding and formal logic proof systems and achieve elevated results for foundation models over baseline. 

---
# MicroVQA: A Multimodal Reasoning Benchmark for Microscopy-Based Scientific Research 

**Authors**: James Burgess, Jeffrey J Nirschl, Laura Bravo-Sánchez, Alejandro Lozano, Sanket Rajan Gupte, Jesus G. Galaz-Montoya, Yuhui Zhang, Yuchang Su, Disha Bhowmik, Zachary Coman, Sarina M. Hasan, Alexandra Johannesson, William D. Leineweber, Malvika G Nair, Ridhi Yarlagadda, Connor Zuraski, Wah Chiu, Sarah Cohen, Jan N. Hansen, Manuel D Leonetti, Chad Liu, Emma Lundberg, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2503.13399)  

**Abstract**: Scientific research demands sophisticated reasoning over multimodal data, a challenge especially prevalent in biology. Despite recent advances in multimodal large language models (MLLMs) for AI-assisted research, existing multimodal reasoning benchmarks only target up to college-level difficulty, while research-level benchmarks emphasize lower-level perception, falling short of the complex multimodal reasoning needed for scientific discovery. To bridge this gap, we introduce MicroVQA, a visual-question answering (VQA) benchmark designed to assess three reasoning capabilities vital in research workflows: expert image understanding, hypothesis generation, and experiment proposal. MicroVQA consists of 1,042 multiple-choice questions (MCQs) curated by biology experts across diverse microscopy modalities, ensuring VQA samples represent real scientific practice. In constructing the benchmark, we find that standard MCQ generation methods induce language shortcuts, motivating a new two-stage pipeline: an optimized LLM prompt structures question-answer pairs into MCQs; then, an agent-based `RefineBot' updates them to remove shortcuts. Benchmarking on state-of-the-art MLLMs reveal a peak performance of 53\%; models with smaller LLMs only slightly underperform top models, suggesting that language-based reasoning is less challenging than multimodal reasoning; and tuning with scientific articles enhances performance. Expert analysis of chain-of-thought responses shows that perception errors are the most frequent, followed by knowledge errors and then overgeneralization errors. These insights highlight the challenges in multimodal scientific reasoning, showing MicroVQA is a valuable resource advancing AI-driven biomedical research. MicroVQA is available at this https URL, and project page at this https URL. 

---
# REGEN: A Dataset and Benchmarks with Natural Language Critiques and Narratives 

**Authors**: Kun Su, Krishna Sayana, Hubert Pham, James Pine, Yuri Vasilevski, Raghavendra Vasudeva, Marialena Kyriakidi, Liam Hebert, Ambarish Jash, Anushya Subbiah, Sukhdeep Sodhi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11924)  

**Abstract**: This paper introduces a novel dataset REGEN (Reviews Enhanced with GEnerative Narratives), designed to benchmark the conversational capabilities of recommender Large Language Models (LLMs), addressing the limitations of existing datasets that primarily focus on sequential item prediction. REGEN extends the Amazon Product Reviews dataset by inpainting two key natural language features: (1) user critiques, representing user "steering" queries that lead to the selection of a subsequent item, and (2) narratives, rich textual outputs associated with each recommended item taking into account prior context. The narratives include product endorsements, purchase explanations, and summaries of user preferences.
Further, we establish an end-to-end modeling benchmark for the task of conversational recommendation, where models are trained to generate both recommendations and corresponding narratives conditioned on user history (items and critiques). For this joint task, we introduce a modeling framework LUMEN (LLM-based Unified Multi-task Model with Critiques, Recommendations, and Narratives) which uses an LLM as a backbone for critiquing, retrieval and generation. We also evaluate the dataset's quality using standard auto-rating techniques and benchmark it by training both traditional and LLM-based recommender models. Our results demonstrate that incorporating critiques enhances recommendation quality by enabling the recommender to learn language understanding and integrate it with recommendation signals. Furthermore, LLMs trained on our dataset effectively generate both recommendations and contextual narratives, achieving performance comparable to state-of-the-art recommenders and language models. 

---
# A Semantic-based Optimization Approach for Repairing LLMs: Case Study on Code Generation 

**Authors**: Jian Gu, Aldeida Aleti, Chunyang Chen, Hongyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12899)  

**Abstract**: Language Models (LMs) are widely used in software engineering for code generation, but they may produce code with errors. Rather than repairing the generated code, an alternative way is to address the underlying failures of models. LM repair offers a lightweight solution to this challenge: it requires minimal data, reduces computational costs, and reduces the side effects. Unlike retraining, LM repair focuses on applying tailored updates to targeted neurons, making it ideal for scenarios with limited resources, high-performance demands, or strict safety requirements. In this paper, we propose \ul{S}emantic \ul{T}argeting for \ul{A}nalytical \ul{R}epair (\textsc{STAR}), a pioneering and novel semantic-based optimization approach for repairing LLMs. \textsc{STAR} realizes main operations in LM repair methods in an optimization process, including locating ``buggy neurons'', solving ``neuron patches'', and patching ``buggy neurons''. Correspondingly, it computes the deltas of weight matrix as the prior information to guide optimization; and attributes the targeted layers and neurons leveraging statistical insights. The neuron patches are computed with a solid semantic-based analytical formula, which directly bridges the changes to logits with the deltas of neurons, by steering latent representations. Compared to the prior work of LM repair (\textsc{MINT}) and optimization methods (\textsc{SGD}), \textsc{STAR} integrates their strengths while mitigating their limitations. \textsc{STAR} supports solving multiple failures together, significantly improving the usefulness. Evaluated on three code generation tasks using popular code LMs, \textsc{STAR} demonstrates superior effectiveness. Additionally, \textsc{STAR} exhibits better efficiency. In terms of side effects, namely the balance between generalization and specificity, \textsc{STAR} outperforms prior work by a significant margin. 

---
# R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization 

**Authors**: Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.12937)  

**Abstract**: Recent studies generally enhance MLLMs' reasoning capabilities via supervised fine-tuning on high-quality chain-of-thought reasoning data, which often leads models to merely imitate successful reasoning paths without understanding what the wrong reasoning paths are. In this work, we aim to enhance the MLLMs' reasoning ability beyond passively imitating positive reasoning paths. To this end, we design Step-wise Group Relative Policy Optimization (StepGRPO), a new online reinforcement learning framework that enables MLLMs to self-improve reasoning ability via simple, effective and dense step-wise rewarding. Specifically, StepGRPO introduces two novel rule-based reasoning rewards: Step-wise Reasoning Accuracy Reward (StepRAR) and Step-wise Reasoning Validity Reward (StepRVR). StepRAR rewards the reasoning paths that contain necessary intermediate reasoning steps via a soft key-step matching technique, while StepRAR rewards reasoning paths that follow a well-structured and logically consistent reasoning process through a reasoning completeness and logic evaluation strategy. With the proposed StepGRPO, we introduce R1-VL, a series of MLLMs with outstanding capabilities in step-by-step reasoning. Extensive experiments over 8 benchmarks demonstrate the superiority of our methods. 

---
# MAP: Evaluation and Multi-Agent Enhancement of Large Language Models for Inpatient Pathways 

**Authors**: Zhen Chen, Zhihao Peng, Xusheng Liang, Cheng Wang, Peigan Liang, Linsheng Zeng, Minjie Ju, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.13205)  

**Abstract**: Inpatient pathways demand complex clinical decision-making based on comprehensive patient information, posing critical challenges for clinicians. Despite advancements in large language models (LLMs) in medical applications, limited research focused on artificial intelligence (AI) inpatient pathways systems, due to the lack of large-scale inpatient datasets. Moreover, existing medical benchmarks typically concentrated on medical question-answering and examinations, ignoring the multifaceted nature of clinical decision-making in inpatient settings. To address these gaps, we first developed the Inpatient Pathway Decision Support (IPDS) benchmark from the MIMIC-IV database, encompassing 51,274 cases across nine triage departments and 17 major disease categories alongside 16 standardized treatment options. Then, we proposed the Multi-Agent Inpatient Pathways (MAP) framework to accomplish inpatient pathways with three clinical agents, including a triage agent managing the patient admission, a diagnosis agent serving as the primary decision maker at the department, and a treatment agent providing treatment plans. Additionally, our MAP framework includes a chief agent overseeing the inpatient pathways to guide and promote these three clinician agents. Extensive experiments showed our MAP improved the diagnosis accuracy by 25.10% compared to the state-of-the-art LLM HuatuoGPT2-13B. It is worth noting that our MAP demonstrated significant clinical compliance, outperforming three board-certified clinicians by 10%-12%, establishing a foundation for inpatient pathways systems. 

---
# Identifying Cooperative Personalities in Multi-agent Contexts through Personality Steering with Representation Engineering 

**Authors**: Kenneth J. K. Ong, Lye Jia Jun, Hieu Minh "Jord" Nguyen, Seong Hah Cho, Natalia Pérez-Campanero Antolín  

**Link**: [PDF](https://arxiv.org/pdf/2503.12722)  

**Abstract**: As Large Language Models (LLMs) gain autonomous capabilities, their coordination in multi-agent settings becomes increasingly important. However, they often struggle with cooperation, leading to suboptimal outcomes. Inspired by Axelrod's Iterated Prisoner's Dilemma (IPD) tournaments, we explore how personality traits influence LLM cooperation. Using representation engineering, we steer Big Five traits (e.g., Agreeableness, Conscientiousness) in LLMs and analyze their impact on IPD decision-making. Our results show that higher Agreeableness and Conscientiousness improve cooperation but increase susceptibility to exploitation, highlighting both the potential and limitations of personality-based steering for aligning AI agents. 

---
# DeepPerception: Advancing R1-like Cognitive Visual Perception in MLLMs for Knowledge-Intensive Visual Grounding 

**Authors**: Xinyu Ma, Ziyang Ding, Zhicong Luo, Chi Chen, Zonghao Guo, Derek F. Wong, Xiaoyi Feng, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.12797)  

**Abstract**: Human experts excel at fine-grained visual discrimination by leveraging domain knowledge to refine perceptual features, a capability that remains underdeveloped in current Multimodal Large Language Models (MLLMs). Despite possessing vast expert-level knowledge, MLLMs struggle to integrate reasoning into visual perception, often generating direct responses without deeper analysis. To bridge this gap, we introduce knowledge-intensive visual grounding (KVG), a novel visual grounding task that requires both fine-grained perception and domain-specific knowledge integration. To address the challenges of KVG, we propose DeepPerception, an MLLM enhanced with cognitive visual perception capabilities. Our approach consists of (1) an automated data synthesis pipeline that generates high-quality, knowledge-aligned training samples, and (2) a two-stage training framework combining supervised fine-tuning for cognitive reasoning scaffolding and reinforcement learning to optimize perception-cognition synergy. To benchmark performance, we introduce KVG-Bench a comprehensive dataset spanning 10 domains with 1.3K manually curated test cases. Experimental results demonstrate that DeepPerception significantly outperforms direct fine-tuning, achieving +8.08\% accuracy improvements on KVG-Bench and exhibiting +4.60\% superior cross-domain generalization over baseline approaches. Our findings highlight the importance of integrating cognitive processes into MLLMs for human-like visual perception and open new directions for multimodal reasoning research. The data, codes, and models are released at this https URL. 

---
# VeriLA: A Human-Centered Evaluation Framework for Interpretable Verification of LLM Agent Failures 

**Authors**: Yoo Yeon Sung, Hannah Kim, Dan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12651)  

**Abstract**: AI practitioners increasingly use large language model (LLM) agents in compound AI systems to solve complex reasoning tasks, these agent executions often fail to meet human standards, leading to errors that compromise the system's overall performance. Addressing these failures through human intervention is challenging due to the agents' opaque reasoning processes, misalignment with human expectations, the complexity of agent dependencies, and the high cost of manual inspection. This paper thus introduces a human-centered evaluation framework for Verifying LLM Agent failures (VeriLA), which systematically assesses agent failures to reduce human effort and make these agent failures interpretable to humans. The framework first defines clear expectations of each agent by curating human-designed agent criteria. Then, it develops a human-aligned agent verifier module, trained with human gold standards, to assess each agent's execution output. This approach enables granular evaluation of each agent's performance by revealing failures from a human standard, offering clear guidelines for revision, and reducing human cognitive load. Our case study results show that VeriLA is both interpretable and efficient in helping practitioners interact more effectively with the system. By upholding accountability in human-agent collaboration, VeriLA paves the way for more trustworthy and human-aligned compound AI systems. 

---
# LLM Agents for Education: Advances and Applications 

**Authors**: Zhendong Chu, Shen Wang, Jian Xie, Tinghui Zhu, Yibo Yan, Jinheng Ye, Aoxiao Zhong, Xuming Hu, Jing Liang, Philip S. Yu, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11733)  

**Abstract**: Large Language Model (LLM) agents have demonstrated remarkable capabilities in automating tasks and driving innovation across diverse educational applications. In this survey, we provide a systematic review of state-of-the-art research on LLM agents in education, categorizing them into two broad classes: (1) \emph{Pedagogical Agents}, which focus on automating complex pedagogical tasks to support both teachers and students; and (2) \emph{Domain-Specific Educational Agents}, which are tailored for specialized fields such as science education, language learning, and professional development. We comprehensively examine the technological advancements underlying these LLM agents, including key datasets, benchmarks, and algorithmic frameworks that drive their effectiveness. Furthermore, we discuss critical challenges such as privacy, bias and fairness concerns, hallucination mitigation, and integration with existing educational ecosystems. This survey aims to provide a comprehensive technological overview of LLM agents for education, fostering further research and collaboration to enhance their impact for the greater good of learners and educators alike. 

---
# Toward a method for LLM-enabled Indoor Navigation 

**Authors**: Alberto Coffrini, Mohammad Amin Zadenoori, Paolo Barsocchi, Francesco Furfari, Antonino Crivello, Alessio Ferrari  

**Link**: [PDF](https://arxiv.org/pdf/2503.11702)  

**Abstract**: Indoor navigation presents unique challenges due to complex layouts, lack of GPS signals, and accessibility concerns. Existing solutions often struggle with real-time adaptability and user-specific needs. In this work, we explore the potential of a Large Language Model (LLM), i.e., ChatGPT, to generate natural, context-aware navigation instructions from indoor map images. We design and evaluate test cases across different real-world environments, analyzing the effectiveness of LLMs in interpreting spatial layouts, handling user constraints, and planning efficient routes. Our findings demonstrate the potential of LLMs for supporting personalized indoor navigation, with an average of 52% correct indications and a maximum of 62%. The results do not appear to depend on the complexity of the layout or the complexity of the expected path, but rather on the number of points of interest and the abundance of visual information, which negatively affect the performance. 

---
# An LLM-Based Approach for Insight Generation in Data Analysis 

**Authors**: Alberto Sánchez Pérez, Alaa Boukhary, Paolo Papotti, Luis Castejón Lozano, Adam Elwood  

**Link**: [PDF](https://arxiv.org/pdf/2503.11664)  

**Abstract**: Generating insightful and actionable information from databases is critical in data analysis. This paper introduces a novel approach using Large Language Models (LLMs) to automatically generate textual insights. Given a multi-table database as input, our method leverages LLMs to produce concise, text-based insights that reflect interesting patterns in the tables. Our framework includes a Hypothesis Generator to formulate domain-relevant questions, a Query Agent to answer such questions by generating SQL queries against a database, and a Summarization module to verbalize the insights. The insights are evaluated for both correctness and subjective insightfulness using a hybrid model of human judgment and automated metrics. Experimental results on public and enterprise databases demonstrate that our approach generates more insightful insights than other approaches while maintaining correctness. 

---
# Are LLMs (Really) Ideological? An IRT-based Analysis and Alignment Tool for Perceived Socio-Economic Bias in LLMs 

**Authors**: Jasmin Wachter, Michael Radloff, Maja Smolej, Katharina Kinder-Kurlanda  

**Link**: [PDF](https://arxiv.org/pdf/2503.13149)  

**Abstract**: We introduce an Item Response Theory (IRT)-based framework to detect and quantify socioeconomic bias in large language models (LLMs) without relying on subjective human judgments. Unlike traditional methods, IRT accounts for item difficulty, improving ideological bias estimation. We fine-tune two LLM families (Meta-LLaMa 3.2-1B-Instruct and Chat- GPT 3.5) to represent distinct ideological positions and introduce a two-stage approach: (1) modeling response avoidance and (2) estimating perceived bias in answered responses. Our results show that off-the-shelf LLMs often avoid ideological engagement rather than exhibit bias, challenging prior claims of partisanship. This empirically validated framework enhances AI alignment research and promotes fairer AI governance. 

---
# MT-RewardTree: A Comprehensive Framework for Advancing LLM-Based Machine Translation via Reward Modeling 

**Authors**: Zhaopeng Feng, Jiahan Ren, Jiayuan Su, Jiamei Zheng, Zhihang Tang, Hongwei Wang, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12123)  

**Abstract**: Process reward models (PRMs) have shown success in complex reasoning tasks for large language models (LLMs). However, their application to machine translation (MT) remains underexplored due to the lack of systematic methodologies and evaluation benchmarks. To address this gap, we introduce \textbf{MT-RewardTree}, a comprehensive framework for constructing, evaluating, and deploying process reward models in MT. Unlike traditional vanilla preference pair construction, we propose a novel method for automatically generating token-level preference pairs using approximate Monte Carlo Tree Search (MCTS), which mitigates the prohibitive cost of human annotation for fine-grained steps. Then, we establish the first MT-specific reward model benchmark and provide a systematic comparison of different reward modeling architectures, revealing that token-level supervision effectively captures fine-grained preferences. Experimental results demonstrate that our MT-PRM-Qwen-2.5-3B achieves state-of-the-art performance in both token-level and sequence-level evaluation given the same input prefix. Furthermore, we showcase practical applications where PRMs enable test-time alignment for LLMs without additional alignment training and significantly improve performance in hypothesis ensembling. Our work provides valuable insights into the role of reward models in MT research. Our code and data are released in \href{this https URL}{this https URL\_RewardTreePage}. 

---
# Can Reasoning Models Reason about Hardware? An Agentic HLS Perspective 

**Authors**: Luca Collini, Andrew Hennessee, Ramesh Karri, Siddharth Garg  

**Link**: [PDF](https://arxiv.org/pdf/2503.12721)  

**Abstract**: Recent Large Language Models (LLMs) such as OpenAI o3-mini and DeepSeek-R1 use enhanced reasoning through Chain-of-Thought (CoT). Their potential in hardware design, which relies on expert-driven iterative optimization, remains unexplored. This paper investigates whether reasoning LLMs can address challenges in High-Level Synthesis (HLS) design space exploration and optimization. During HLS, engineers manually define pragmas/directives to balance performance and resource constraints. We propose an LLM-based optimization agentic framework that automatically restructures code, inserts pragmas, and identifies optimal design points via feedback from HLs tools and access to integer-linear programming (ILP) solvers. Experiments compare reasoning models against conventional LLMs on benchmarks using success rate, efficiency, and design quality (area/latency) metrics, and provide the first-ever glimpse into the CoTs produced by a powerful open-source reasoning model like DeepSeek-R1. 

---
# MPBench: A Comprehensive Multimodal Reasoning Benchmark for Process Errors Identification 

**Authors**: Zhaopan Xu, Pengfei Zhou, Jiaxin Ai, Wangbo Zhao, Kai Wang, Xiaojiang Peng, Wenqi Shao, Hongxun Yao, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12505)  

**Abstract**: Reasoning is an essential capacity for large language models (LLMs) to address complex tasks, where the identification of process errors is vital for improving this ability. Recently, process-level reward models (PRMs) were proposed to provide step-wise rewards that facilitate reinforcement learning and data production during training and guide LLMs toward correct steps during inference, thereby improving reasoning accuracy. However, existing benchmarks of PRMs are text-based and focus on error detection, neglecting other scenarios like reasoning search. To address this gap, we introduce MPBench, a comprehensive, multi-task, multimodal benchmark designed to systematically assess the effectiveness of PRMs in diverse scenarios. MPBench employs three evaluation paradigms, each targeting a specific role of PRMs in the reasoning process: (1) Step Correctness, which assesses the correctness of each intermediate reasoning step; (2) Answer Aggregation, which aggregates multiple solutions and selects the best one; and (3) Reasoning Process Search, which guides the search for optimal reasoning steps during inference. Through these paradigms, MPBench makes comprehensive evaluations and provides insights into the development of multimodal PRMs. 

---
# A Survey on the Optimization of Large Language Model-based Agents 

**Authors**: Shangheng Du, Jiabao Zhao, Jinxin Shi, Zhentao Xie, Xin Jiang, Yanhong Bai, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2503.12434)  

**Abstract**: With the rapid development of Large Language Models (LLMs), LLM-based agents have been widely adopted in various fields, becoming essential for autonomous decision-making and interactive tasks. However, current work typically relies on prompt design or fine-tuning strategies applied to vanilla LLMs, which often leads to limited effectiveness or suboptimal performance in complex agent-related environments. Although LLM optimization techniques can improve model performance across many general tasks, they lack specialized optimization towards critical agent functionalities such as long-term planning, dynamic environmental interaction, and complex decision-making. Although numerous recent studies have explored various strategies to optimize LLM-based agents for complex agent tasks, a systematic review summarizing and comparing these methods from a holistic perspective is still lacking. In this survey, we provide a comprehensive review of LLM-based agent optimization approaches, categorizing them into parameter-driven and parameter-free methods. We first focus on parameter-driven optimization, covering fine-tuning-based optimization, reinforcement learning-based optimization, and hybrid strategies, analyzing key aspects such as trajectory data construction, fine-tuning techniques, reward function design, and optimization algorithms. Additionally, we briefly discuss parameter-free strategies that optimize agent behavior through prompt engineering and external knowledge retrieval. Finally, we summarize the datasets and benchmarks used for evaluation and tuning, review key applications of LLM-based agents, and discuss major challenges and promising future directions. Our repository for related references is available at this https URL. 

---
# Knowledge-Aware Iterative Retrieval for Multi-Agent Systems 

**Authors**: Seyoung Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.13275)  

**Abstract**: We introduce a novel large language model (LLM)-driven agent framework, which iteratively refines queries and filters contextual evidence by leveraging dynamically evolving knowledge. A defining feature of the system is its decoupling of external sources from an internal knowledge cache that is progressively updated to guide both query generation and evidence selection. This design mitigates bias-reinforcement loops and enables dynamic, trackable search exploration paths, thereby optimizing the trade-off between exploring diverse information and maintaining accuracy through autonomous agent decision-making. Our approach is evaluated on a broad range of open-domain question answering benchmarks, including multi-step tasks that mirror real-world scenarios where integrating information from multiple sources is critical, especially given the vulnerabilities of LLMs that lack explicit reasoning or planning capabilities. The results show that the proposed system not only outperforms single-step baselines regardless of task difficulty but also, compared to conventional iterative retrieval methods, demonstrates pronounced advantages in complex tasks through precise evidence-based reasoning and enhanced efficiency. The proposed system supports both competitive and collaborative sharing of updated context, enabling multi-agent extension. The benefits of multi-agent configurations become especially prominent as task difficulty increases. The number of convergence steps scales with task difficulty, suggesting cost-effective scalability. 

---
# Automating the loop in traffic incident management on highway 

**Authors**: Matteo Cercola, Nicola Gatti, Pedro Huertas Leyva, Benedetto Carambia, Simone Formentin  

**Link**: [PDF](https://arxiv.org/pdf/2503.12085)  

**Abstract**: Effective traffic incident management is essential for ensuring safety, minimizing congestion, and reducing response times in emergency situations. Traditional highway incident management relies heavily on radio room operators, who must make rapid, informed decisions in high-stakes environments. This paper proposes an innovative solution to support and enhance these decisions by integrating Large Language Models (LLMs) into a decision-support system for traffic incident management. We introduce two approaches: (1) an LLM + Optimization hybrid that leverages both the flexibility of natural language interaction and the robustness of optimization techniques, and (2) a Full LLM approach that autonomously generates decisions using only LLM capabilities. We tested our solutions using historical event data from Autostrade per l'Italia. Experimental results indicate that while both approaches show promise, the LLM + Optimization solution demonstrates superior reliability, making it particularly suited to critical applications where consistency and accuracy are paramount. This research highlights the potential for LLMs to transform highway incident management by enabling accessible, data-driven decision-making support. 

---
# SPIN-Bench: How Well Do LLMs Plan Strategically and Reason Socially? 

**Authors**: Jianzhu Yao, Kevin Wang, Ryan Hsieh, Haisu Zhou, Tianqing Zou, Zerui Cheng, Zhangyang Wang, Pramod Viswanath  

**Link**: [PDF](https://arxiv.org/pdf/2503.12349)  

**Abstract**: Reasoning and strategic behavior in \emph{social interactions} is a hallmark of intelligence. This form of reasoning is significantly more sophisticated than isolated planning or reasoning tasks in static settings (e.g., math problem solving). In this paper, we present \textit{Strategic Planning, Interaction, and Negotiation} (\textbf{SPIN-Bench}), a new multi-domain evaluation designed to measure the intelligence of \emph{strategic planning} and \emph{social reasoning}. While many existing benchmarks focus on narrow planning or single-agent reasoning, SPIN-Bench combines classical PDDL tasks, competitive board games, cooperative card games, and multi-agent negotiation scenarios in one unified framework. The framework includes both a benchmark as well as an arena to simulate and evaluate the variety of social settings to test reasoning and strategic behavior of AI agents. We formulate the benchmark SPIN-Bench by systematically varying action spaces, state complexity, and the number of interacting agents to simulate a variety of social settings where success depends on not only methodical and step-wise decision making, but also \emph{conceptual inference} of other (adversarial or cooperative) participants. Our experiments reveal that while contemporary LLMs handle \emph{basic fact retrieval} and \emph{short-range planning} reasonably well, they encounter significant performance bottlenecks in tasks requiring \emph{deep multi-hop reasoning} over large state spaces and \emph{socially adept} coordination under uncertainty. We envision SPIN-Bench as a catalyst for future research on robust multi-agent planning, social reasoning, and human--AI teaming. 

---
# Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation 

**Authors**: Bowen Baker, Joost Huizinga, Leo Gao, Zehao Dou, Melody Y. Guan, Aleksander Madry, Wojciech Zaremba, Jakub Pachocki, David Farhi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11926)  

**Abstract**: Mitigating reward hacking--where AI systems misbehave due to flaws or misspecifications in their learning objectives--remains a key challenge in constructing capable and aligned models. We show that we can monitor a frontier reasoning model, such as OpenAI o3-mini, for reward hacking in agentic coding environments by using another LLM that observes the model's chain-of-thought (CoT) reasoning. CoT monitoring can be far more effective than monitoring agent actions and outputs alone, and we further found that a LLM weaker than o3-mini, namely GPT-4o, can effectively monitor a stronger model. Because CoT monitors can be effective at detecting exploits, it is natural to ask whether those exploits can be suppressed by incorporating a CoT monitor directly into the agent's training objective. While we show that integrating CoT monitors into the reinforcement learning reward can indeed produce more capable and more aligned agents in the low optimization regime, we find that with too much optimization, agents learn obfuscated reward hacking, hiding their intent within the CoT while still exhibiting a significant rate of reward hacking. Because it is difficult to tell when CoTs have become obfuscated, it may be necessary to pay a monitorability tax by not applying strong optimization pressures directly to the chain-of-thought, ensuring that CoTs remain monitorable and useful for detecting misaligned behavior. 

---
# Visualizing Thought: Conceptual Diagrams Enable Robust Planning in LMMs 

**Authors**: Nasim Borazjanizadeh, Roei Herzig, Eduard Oks, Trevor Darrell, Rogerio Feris, Leonid Karlinsky  

**Link**: [PDF](https://arxiv.org/pdf/2503.11790)  

**Abstract**: Human reasoning relies on constructing and manipulating mental models-simplified internal representations of situations that we use to understand and solve problems. Conceptual diagrams (for example, sketches drawn by humans to aid reasoning) externalize these mental models, abstracting irrelevant details to efficiently capture relational and spatial information. In contrast, Large Language Models (LLMs) and Large Multimodal Models (LMMs) predominantly reason through textual representations, limiting their effectiveness in complex multi-step combinatorial and planning tasks. In this paper, we propose a zero-shot fully automatic framework that enables LMMs to reason through multiple chains of self-generated intermediate conceptual diagrams, significantly enhancing their combinatorial planning capabilities. Our approach does not require any human initialization beyond a natural language description of the task. It integrates both textual and diagrammatic reasoning within an optimized graph-of-thought inference framework, enhanced by beam search and depth-wise backtracking. Evaluated on multiple challenging PDDL planning domains, our method substantially improves GPT-4o's performance (for example, from 35.5% to 90.2% in Blocksworld). On more difficult planning domains with solution depths up to 40, our approach outperforms even the o1-preview reasoning model (for example, over 13% improvement in Parking). These results highlight the value of conceptual diagrams as a complementary reasoning medium in LMMs. 

---
# xLSTM 7B: A Recurrent LLM for Fast and Efficient Inference 

**Authors**: Maximilian Beck, Korbinian Pöppel, Phillip Lippe, Richard Kurle, Patrick M. Blies, Günter Klambauer, Sebastian Böck, Sepp Hochreiter  

**Link**: [PDF](https://arxiv.org/pdf/2503.13427)  

**Abstract**: Recent breakthroughs in solving reasoning, math and coding problems with Large Language Models (LLMs) have been enabled by investing substantial computation budgets at inference time. Therefore, inference speed is one of the most critical properties of LLM architectures, and there is a growing need for LLMs that are efficient and fast at inference. Recently, LLMs built on the xLSTM architecture have emerged as a powerful alternative to Transformers, offering linear compute scaling with sequence length and constant memory usage, both highly desirable properties for efficient inference. However, such xLSTM-based LLMs have yet to be scaled to larger models and assessed and compared with respect to inference speed and efficiency. In this work, we introduce xLSTM 7B, a 7-billion-parameter LLM that combines xLSTM's architectural benefits with targeted optimizations for fast and efficient inference. Our experiments demonstrate that xLSTM 7B achieves performance on downstream tasks comparable to other similar-sized LLMs, while providing significantly faster inference speeds and greater efficiency compared to Llama- and Mamba-based LLMs. These results establish xLSTM 7B as the fastest and most efficient 7B LLM, offering a solution for tasks that require large amounts of test-time computation. Our work highlights xLSTM's potential as a foundational architecture for methods building on heavy use of LLM inference. Our model weights, model code and training code are open-source. 

---
# Mitigating Visual Forgetting via Take-along Visual Conditioning for Multi-modal Long CoT Reasoning 

**Authors**: Hai-Long Sun, Zhun Sun, Houwen Peng, Han-Jia Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.13360)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have demonstrated enhanced reasoning capabilities, evolving from Chain-of-Thought (CoT) prompting to advanced, product-oriented solutions like OpenAI o1. During our re-implementation of this model, we noticed that in multimodal tasks requiring visual input (e.g., geometry problems), Multimodal LLMs (MLLMs) struggle to maintain focus on the visual information, in other words, MLLMs suffer from a gradual decline in attention to visual information as reasoning progresses, causing text-over-relied outputs. To investigate this, we ablate image inputs during long-chain reasoning. Concretely, we truncate the reasoning process midway, then re-complete the reasoning process with the input image removed. We observe only a ~2% accuracy drop on MathVista's test-hard subset, revealing the model's textual outputs dominate the following reasoning process. Motivated by this, we propose Take-along Visual Conditioning (TVC), a strategy that shifts image input to critical reasoning stages and compresses redundant visual tokens via dynamic pruning. This methodology helps the model retain attention to the visual components throughout the reasoning. Our approach achieves state-of-the-art performance on average across five mathematical reasoning benchmarks (+3.4% vs previous sota), demonstrating the effectiveness of TVC in enhancing multimodal reasoning systems. 

---
# A Comprehensive Survey on Multi-Agent Cooperative Decision-Making: Scenarios, Approaches, Challenges and Perspectives 

**Authors**: Weiqiang Jin, Hongyang Du, Biao Zhao, Xingwu Tian, Bohang Shi, Guang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13415)  

**Abstract**: With the rapid development of artificial intelligence, intelligent decision-making techniques have gradually surpassed human levels in various human-machine competitions, especially in complex multi-agent cooperative task scenarios. Multi-agent cooperative decision-making involves multiple agents working together to complete established tasks and achieve specific objectives. These techniques are widely applicable in real-world scenarios such as autonomous driving, drone navigation, disaster rescue, and simulated military confrontations. This paper begins with a comprehensive survey of the leading simulation environments and platforms used for multi-agent cooperative decision-making. Specifically, we provide an in-depth analysis for these simulation environments from various perspectives, including task formats, reward allocation, and the underlying technologies employed. Subsequently, we provide a comprehensive overview of the mainstream intelligent decision-making approaches, algorithms and models for multi-agent systems (MAS). Theseapproaches can be broadly categorized into five types: rule-based (primarily fuzzy logic), game theory-based, evolutionary algorithms-based, deep multi-agent reinforcement learning (MARL)-based, and large language models(LLMs)reasoning-based. Given the significant advantages of MARL andLLMs-baseddecision-making methods over the traditional rule, game theory, and evolutionary algorithms, this paper focuses on these multi-agent methods utilizing MARL and LLMs-based techniques. We provide an in-depth discussion of these approaches, highlighting their methodology taxonomies, advantages, and drawbacks. Further, several prominent research directions in the future and potential challenges of multi-agent cooperative decision-making are also detailed. 

---
# LEAVS: An LLM-based Labeler for Abdominal CT Supervision 

**Authors**: Ricardo Bigolin Lanfredi, Yan Zhuang, Mark Finkelstein, Praveen Thoppey Srinivasan Balamuralikrishna, Luke Krembs, Brandon Khoury, Arthi Reddy, Pritam Mukherjee, Neil M. Rofsky, Ronald M. Summers  

**Link**: [PDF](https://arxiv.org/pdf/2503.13330)  

**Abstract**: Extracting structured labels from radiology reports has been employed to create vision models to simultaneously detect several types of abnormalities. However, existing works focus mainly on the chest region. Few works have been investigated on abdominal radiology reports due to more complex anatomy and a wider range of pathologies in the abdomen. We propose LEAVS (Large language model Extractor for Abdominal Vision Supervision). This labeler can annotate the certainty of presence and the urgency of seven types of abnormalities for nine abdominal organs on CT radiology reports. To ensure broad coverage, we chose abnormalities that encompass most of the finding types from CT reports. Our approach employs a specialized chain-of-thought prompting strategy for a locally-run LLM using sentence extraction and multiple-choice questions in a tree-based decision system. We demonstrate that the LLM can extract several abnormality types across abdominal organs with an average F1 score of 0.89, significantly outperforming competing labelers and humans. Additionally, we show that extraction of urgency labels achieved performance comparable to human annotations. Finally, we demonstrate that the abnormality labels contain valuable information for training a single vision model that classifies several organs as normal or abnormal. We release our code and structured annotations for a public CT dataset containing over 1,000 CT volumes. 

---
# Goal2Story: A Multi-Agent Fleet based on Privately Enabled sLLMs for Impacting Mapping on Requirements Elicitation 

**Authors**: Xinkai Zou, Yan Liu, Xiongbo Shi, Chen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13279)  

**Abstract**: As requirements drift with rapid iterations, agile development becomes the dominant paradigm. Goal-driven Requirements Elicitation (RE) is a pivotal yet challenging task in agile project development due to its heavy tangling with adaptive planning and efficient collaboration. Recently, AI agents have shown promising ability in supporting requirements analysis by saving significant time and effort for stakeholders. However, current research mainly focuses on functional RE, and research works have not been reported bridging the long journey from goal to user stories. Moreover, considering the cost of LLM facilities and the need for data and idea protection, privately hosted small-sized LLM should be further utilized in RE. To address these challenges, we propose Goal2Story, a multi-agent fleet that adopts the Impact Mapping (IM) framework while merely using cost-effective sLLMs for goal-driven RE. Moreover, we introduce a StorySeek dataset that contains over 1,000 user stories (USs) with corresponding goals and project context information, as well as the semi-automatic dataset construction method. For evaluation, we proposed two metrics: Factuality Hit Rate (FHR) to measure consistency between the generated USs with the dataset and Quality And Consistency Evaluation (QuACE) to evaluate the quality of the generated USs. Experimental results demonstrate that Goal2Story outperforms the baseline performance of the Super-Agent adopting powerful LLMs, while also showcasing the performance improvements in key metrics brought by CoT and Agent Profile to Goal2Story, as well as its exploration in identifying latent needs. 

---
# Cream of the Crop: Harvesting Rich, Scalable and Transferable Multi-Modal Data for Instruction Fine-Tuning 

**Authors**: Mengyao Lyu, Yan Li, Huasong Zhong, Wenhao Yang, Hui Chen, Jungong Han, Guiguang Ding, Zhenheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13383)  

**Abstract**: The hypothesis that pretrained large language models (LLMs) necessitate only minimal supervision during the fine-tuning (SFT) stage (Zhou et al., 2024) has been substantiated by recent advancements in data curation and selection research. However, their stability and generalizability are compromised due to the vulnerability to experimental setups and validation protocols, falling short of surpassing random sampling (Diddee & Ippolito, 2024; Xia et al., 2024b). Built upon LLMs, multi-modal LLMs (MLLMs), combined with the sheer token volume and heightened heterogeneity of data sources, amplify both the significance and complexity of data selection.
To harvest multi-modal instructional data in a robust and efficient manner, we re-define the granularity of the quality metric by decomposing it into 14 vision-language-related capabilities, and introduce multi-modal rich scorers to evaluate the capabilities of each data candidate. To promote diversity, in light of the inherent objective of the alignment stage, we take interaction style as diversity indicator and use a multi-modal rich styler to identify data instruction patterns. In doing so, our multi-modal rich scorers and styler (mmSSR) guarantee that high-scoring information is conveyed to users in diversified forms. Free from embedding-based clustering or greedy sampling, mmSSR efficiently scales to millions of data with varying budget constraints, supports customization for general or specific capability acquisition, and facilitates training-free generalization to new domains for curation. Across 10+ experimental settings, validated by 14 multi-modal benchmarks, we demonstrate consistent improvements over random sampling, baseline strategies and state-of-the-art selection methods, achieving 99.1% of full performance with only 30% of the 2.6M data. 

---
# 3DAxisPrompt: Promoting the 3D Grounding and Reasoning in GPT-4o 

**Authors**: Dingning Liu, Cheng Wang, Peng Gao, Renrui Zhang, Xinzhu Ma, Yuan Meng, Zhihui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13185)  

**Abstract**: Multimodal Large Language Models (MLLMs) exhibit impressive capabilities across a variety of tasks, especially when equipped with carefully designed visual prompts. However, existing studies primarily focus on logical reasoning and visual understanding, while the capability of MLLMs to operate effectively in 3D vision remains an ongoing area of exploration. In this paper, we introduce a novel visual prompting method, called 3DAxisPrompt, to elicit the 3D understanding capabilities of MLLMs in real-world scenes. More specifically, our method leverages the 3D coordinate axis and masks generated from the Segment Anything Model (SAM) to provide explicit geometric priors to MLLMs and then extend their impressive 2D grounding and reasoning ability to real-world 3D scenarios. Besides, we first provide a thorough investigation of the potential visual prompting formats and conclude our findings to reveal the potential and limits of 3D understanding capabilities in GPT-4o, as a representative of MLLMs. Finally, we build evaluation environments with four datasets, i.e., ScanRefer, ScanNet, FMB, and nuScene datasets, covering various 3D tasks. Based on this, we conduct extensive quantitative and qualitative experiments, which demonstrate the effectiveness of the proposed method. Overall, our study reveals that MLLMs, with the help of 3DAxisPrompt, can effectively perceive an object's 3D position in real-world scenarios. Nevertheless, a single prompt engineering approach does not consistently achieve the best outcomes for all 3D tasks. This study highlights the feasibility of leveraging MLLMs for 3D vision grounding/reasoning with prompt engineering techniques. 

---
# ClearSight: Visual Signal Enhancement for Object Hallucination Mitigation in Multimodal Large language Models 

**Authors**: Hao Yin, Guangzong Si, Zilei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13107)  

**Abstract**: Contrastive decoding strategies are widely used to mitigate object hallucinations in multimodal large language models (MLLMs). By reducing over-reliance on language priors, these strategies ensure that generated content remains closely grounded in visual inputs, producing contextually accurate outputs. Since contrastive decoding requires no additional training or external tools, it offers both computational efficiency and versatility, making it highly attractive. However, these methods present two main limitations: (1) bluntly suppressing language priors can compromise coherence and accuracy of generated content, and (2) processing contrastive inputs adds computational load, significantly slowing inference speed. To address these challenges, we propose Visual Amplification Fusion (VAF), a plug-and-play technique that enhances attention to visual signals within the model's middle layers, where modality fusion predominantly occurs. This approach enables more effective capture of visual features, reducing the model's bias toward language modality. Experimental results demonstrate that VAF significantly reduces hallucinations across various MLLMs without affecting inference speed, while maintaining coherence and accuracy in generated outputs. 

---
# Lifting the Veil on Visual Information Flow in MLLMs: Unlocking Pathways to Faster Inference 

**Authors**: Hao Yin, Guangzong Si, Zilei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13108)  

**Abstract**: Multimodal large language models (MLLMs) improve performance on vision-language tasks by integrating visual features from pre-trained vision encoders into large language models (LLMs). However, how MLLMs process and utilize visual information remains unclear. In this paper, a shift in the dominant flow of visual information is uncovered: (1) in shallow layers, strong interactions are observed between image tokens and instruction tokens, where most visual information is injected into instruction tokens to form cross-modal semantic representations; (2) in deeper layers, image tokens primarily interact with each other, aggregating the remaining visual information to optimize semantic representations within visual modality. Based on these insights, we propose Hierarchical Modality-Aware Pruning (HiMAP), a plug-and-play inference acceleration method that dynamically prunes image tokens at specific layers, reducing computational costs by approximately 65% without sacrificing performance. Our findings offer a new understanding of visual information processing in MLLMs and provide a state-of-the-art solution for efficient inference. 

---
# MAP: Multi-user Personalization with Collaborative LLM-powered Agents 

**Authors**: Christine Lee, Jihye Choi, Bilge Mutlu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12757)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) and LLM-powered agents in multi-user settings underscores the need for reliable, usable methods to accommodate diverse preferences and resolve conflicting directives. Drawing on conflict resolution theory, we introduce a user-centered workflow for multi-user personalization comprising three stages: Reflection, Analysis, and Feedback. We then present MAP -- a \textbf{M}ulti-\textbf{A}gent system for multi-user \textbf{P}ersonalization -- to operationalize this workflow. By delegating subtasks to specialized agents, MAP (1) retrieves and reflects on relevant user information, while enhancing reliability through agent-to-agent interactions, (2) provides detailed analysis for improved transparency and usability, and (3) integrates user feedback to iteratively refine results. Our user study findings (n=12) highlight MAP's effectiveness and usability for conflict resolution while emphasizing the importance of user involvement in resolution verification and failure management. This work highlights the potential of multi-agent systems to implement user-centered, multi-user personalization workflows and concludes by offering insights for personalization in multi-user contexts. 

---
# MoECollab: Democratizing LLM Development Through Collaborative Mixture of Experts 

**Authors**: Harshit  

**Link**: [PDF](https://arxiv.org/pdf/2503.12592)  

**Abstract**: Large Language Model (LLM) development has become increasingly centralized, limiting participation to well-resourced organizations. This paper introduces MoECollab, a novel framework leveraging Mixture of Experts (MoE) architecture to enable distributed, collaborative LLM development. By decomposing monolithic models into specialized expert modules coordinated by a trainable gating network, our framework allows diverse contributors to participate regardless of computational resources. We provide a complete technical implementation with mathematical foundations for expert dynamics, gating mechanisms, and integration strategies. Experiments on multiple datasets demonstrate that our approach achieves accuracy improvements of 3-7% over baseline models while reducing computational requirements by 34%. Expert specialization yields significant domain-specific gains, with improvements from 51% to 88% F1 score in general classification and from 23% to 44% accuracy in news categorization. We formalize the routing entropy optimization problem and demonstrate how proper regularization techniques lead to 14% higher expert utilization rates. These results validate MoECollab as an effective approach for democratizing LLM development through architecturally-supported collaboration. 

---
# Aligning Vision to Language: Text-Free Multimodal Knowledge Graph Construction for Enhanced LLMs Reasoning 

**Authors**: Junming Liu, Siyuan Meng, Yanting Gao, Song Mao, Pinlong Cai, Guohang Yan, Yirong Chen, Zilin Bian, Botian Shi, Ding Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12972)  

**Abstract**: Multimodal reasoning in Large Language Models (LLMs) struggles with incomplete knowledge and hallucination artifacts, challenges that textual Knowledge Graphs (KGs) only partially mitigate due to their modality isolation. While Multimodal Knowledge Graphs (MMKGs) promise enhanced cross-modal understanding, their practical construction is impeded by semantic narrowness of manual text annotations and inherent noise in visual-semantic entity linkages. In this paper, we propose Vision-align-to-Language integrated Knowledge Graph (VaLiK), a novel approach for constructing MMKGs that enhances LLMs reasoning through cross-modal information supplementation. Specifically, we cascade pre-trained Vision-Language Models (VLMs) to align image features with text, transforming them into descriptions that encapsulate image-specific information. Furthermore, we developed a cross-modal similarity verification mechanism to quantify semantic consistency, effectively filtering out noise introduced during feature alignment. Even without manually annotated image captions, the refined descriptions alone suffice to construct the MMKG. Compared to conventional MMKGs construction paradigms, our approach achieves substantial storage efficiency gains while maintaining direct entity-to-image linkage capability. Experimental results on multimodal reasoning tasks demonstrate that LLMs augmented with VaLiK outperform previous state-of-the-art models. Our code is published at this https URL. 

---
# Facilitating Automated Online Consensus Building through Parallel Thinking 

**Authors**: Wen Gu, Zhaoxing Li, Jan Buermann, Jim Dilkes, Dimitris Michailidis, Shinobu Hasegawa, Vahid Yazdanpanah, Sebastian Stein  

**Link**: [PDF](https://arxiv.org/pdf/2503.12499)  

**Abstract**: Consensus building is inherently challenging due to the diverse opinions held by stakeholders. Effective facilitation is crucial to support the consensus building process and enable efficient group decision making. However, the effectiveness of facilitation is often constrained by human factors such as limited experience and scalability. In this research, we propose a Parallel Thinking-based Facilitation Agent (PTFA) that facilitates online, text-based consensus building processes. The PTFA automatically collects textual posts and leverages large language models (LLMs) to perform all of the six distinct roles of the well-established Six Thinking Hats technique in parallel thinking. To illustrate the potential of PTFA, a pilot study was carried out and PTFA's ability in idea generation, emotional probing, and deeper analysis of ideas was demonstrated. Furthermore, a comprehensive dataset that contains not only the conversational content among the participants but also between the participants and the agent is constructed for future study. 

---
# Unveiling Pitfalls: Understanding Why AI-driven Code Agents Fail at GitHub Issue Resolution 

**Authors**: Zhi Chen, Wei Ma, Lingxiao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12374)  

**Abstract**: AI-driven software development has rapidly advanced with the emergence of software development agents that leverage large language models (LLMs) to tackle complex, repository-level software engineering tasks. These agents go beyond just generation of final code; they engage in multi-step reasoning, utilize various tools for code modification and debugging, and interact with execution environments to diagnose and iteratively resolve issues. However, most existing evaluations focus primarily on static analyses of final code outputs, yielding limited insights into the agents' dynamic problem-solving processes. To fill this gap, we conduct an in-depth empirical study on 3,977 solving-phase trajectories and 3,931 testing-phase logs from 8 top-ranked agents evaluated on 500 GitHub issues in the SWE-Bench benchmark. Our exploratory analysis shows that Python execution errors during the issue resolution phase correlate with lower resolution rates and increased reasoning overheads. We have identified the most prevalent errors -- such as ModuleNotFoundError and TypeError -- and highlighted particularly challenging errors like OSError and database-related issues (e.g., IntegrityError) that demand significantly more debugging effort. Furthermore, we have discovered 3 bugs in the SWE-Bench platform that affect benchmark fairness and accuracy; these issues have been reported to and confirmed by the maintainers. To promote transparency and foster future research, we publicly share our datasets and analysis scripts. 

---
# GeoRSMLLM: A Multimodal Large Language Model for Vision-Language Tasks in Geoscience and Remote Sensing 

**Authors**: Zilun Zhang, Haozhan Shen, Tiancheng Zhao, Bin Chen, Zian Guan, Yuhao Wang, Xu Jia, Yuxiang Cai, Yongheng Shang, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2503.12490)  

**Abstract**: The application of Vision-Language Models (VLMs) in remote sensing (RS) has demonstrated significant potential in traditional tasks such as scene classification, object detection, and image captioning. However, current models, which excel in Referring Expression Comprehension (REC), struggle with tasks involving complex instructions (e.g., exists multiple conditions) or pixel-level operations like segmentation and change detection. In this white paper, we provide a comprehensive hierarchical summary of vision-language tasks in RS, categorized by the varying levels of cognitive capability required. We introduce the Remote Sensing Vision-Language Task Set (RSVLTS), which includes Open-Vocabulary Tasks (OVT), Referring Expression Tasks (RET), and Described Object Tasks (DOT) with increased difficulty, and Visual Question Answering (VQA) aloneside. Moreover, we propose a novel unified data representation using a set-of-points approach for RSVLTS, along with a condition parser and a self-augmentation strategy based on cyclic referring. These features are integrated into the GeoRSMLLM model, and this enhanced model is designed to handle a broad range of tasks of RSVLTS, paving the way for a more generalized solution for vision-language tasks in geoscience and remote sensing. 

---
# Leveraging Vision Capabilities of Multimodal LLMs for Automated Data Extraction from Plots 

**Authors**: Maciej P. Polak, Dane Morgan  

**Link**: [PDF](https://arxiv.org/pdf/2503.12326)  

**Abstract**: Automated data extraction from research texts has been steadily improving, with the emergence of large language models (LLMs) accelerating progress even further. Extracting data from plots in research papers, however, has been such a complex task that it has predominantly been confined to manual data extraction. We show that current multimodal large language models, with proper instructions and engineered workflows, are capable of accurately extracting data from plots. This capability is inherent to the pretrained models and can be achieved with a chain-of-thought sequence of zero-shot engineered prompts we call PlotExtract, without the need to fine-tune. We demonstrate PlotExtract here and assess its performance on synthetic and published plots. We consider only plots with two axes in this analysis. For plots identified as extractable, PlotExtract finds points with over 90% precision (and around 90% recall) and errors in x and y position of around 5% or lower. These results prove that multimodal LLMs are a viable path for high-throughput data extraction for plots and in many circumstances can replace the current manual methods of data extraction. 

---
# Toward Foundation Models for Online Complex Event Detection in CPS-IoT: A Case Study 

**Authors**: Liying Han, Gaofeng Dong, Xiaomin Ouyang, Lance Kaplan, Federico Cerutti, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2503.12282)  

**Abstract**: Complex events (CEs) play a crucial role in CPS-IoT applications, enabling high-level decision-making in domains such as smart monitoring and autonomous systems. However, most existing models focus on short-span perception tasks, lacking the long-term reasoning required for CE detection. CEs consist of sequences of short-time atomic events (AEs) governed by spatiotemporal dependencies. Detecting them is difficult due to long, noisy sensor data and the challenge of filtering out irrelevant AEs while capturing meaningful patterns. This work explores CE detection as a case study for CPS-IoT foundation models capable of long-term reasoning. We evaluate three approaches: (1) leveraging large language models (LLMs), (2) employing various neural architectures that learn CE rules from data, and (3) adopting a neurosymbolic approach that integrates neural models with symbolic engines embedding human knowledge. Our results show that the state-space model, Mamba, which belongs to the second category, outperforms all methods in accuracy and generalization to longer, unseen sensor traces. These findings suggest that state-space models could be a strong backbone for CPS-IoT foundation models for long-span reasoning tasks. 

---
# Agentic Search Engine for Real-Time IoT Data 

**Authors**: Abdelrahman Elewah, Khalid Elgazzar  

**Link**: [PDF](https://arxiv.org/pdf/2503.12255)  

**Abstract**: The Internet of Things (IoT) has enabled diverse devices to communicate over the Internet, yet the fragmentation of IoT systems limits seamless data sharing and coordinated management. We have recently introduced SensorsConnect, a unified framework to enable seamless content and sensor data sharing in collaborative IoT systems, inspired by how the World Wide Web (WWW) enabled a shared and accessible space for information among humans. This paper presents the IoT Agentic Search Engine (IoT-ASE), a real-time search engine tailored for IoT environments. IoT-ASE leverages Large Language Models (LLMs) and Retrieval Augmented Generation (RAG) techniques to address the challenge of searching vast, real-time IoT data, enabling it to handle complex queries and deliver accurate, contextually relevant results. We implemented a use-case scenario in Toronto to demonstrate how IoT-ASE can improve service quality recommendations by leveraging real-time IoT data. Our evaluation shows that IoT-ASE achieves a 92\% accuracy in retrieving intent-based services and produces responses that are concise, relevant, and context-aware, outperforming generalized responses from systems like Gemini. These findings highlight the potential IoT-ASE to make real-time IoT data accessible and support effective, real-time decision-making. 

---
# When neural implant meets multimodal LLM: A dual-loop system for neuromodulation and naturalistic neuralbehavioral research 

**Authors**: Edward Hong Wang, Cynthia Xin Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.12334)  

**Abstract**: We propose a novel dual-loop system that synergistically combines responsive neurostimulation (RNS) implants with artificial intelligence-driven wearable devices for treating post-traumatic stress disorder (PTSD) and enabling naturalistic brain research. In PTSD Therapy Mode, an implanted closed-loop neural device monitors amygdala activity and provides on-demand stimulation upon detecting pathological theta oscillations, while an ensemble of wearables (smart glasses, smartwatches, smartphones) uses multimodal large language model (LLM) analysis of sensory data to detect environmental or physiological PTSD triggers and deliver timely audiovisual interventions. Logged events from both the neural and wearable loops are analyzed to personalize trigger detection and progressively transition patients to non-invasive interventions. In Neuroscience Research Mode, the same platform is adapted for real-world brain activity capture. Wearable-LLM systems recognize naturalistic events (social interactions, emotional situations, compulsive behaviors, decision making) and signal implanted RNS devices (via wireless triggers) to record synchronized intracranial data during these moments. This approach builds on recent advances in mobile intracranial EEG recording and closed-loop neuromodulation in humans (BRAIN Initiative, 2023) (Mobbs et al., 2021). We discuss how our interdisciplinary system could revolutionize PTSD therapy and cognitive neuroscience by enabling 24/7 monitoring, context-aware intervention, and rich data collection outside traditional labs. The vision is a future where AI-enhanced devices continuously collaborate with the human brain, offering therapeutic support and deep insights into neural function, with the resulting real-world context rich neural data, in turn, accelerating the development of more biologically-grounded and human-centric AI. 

---
# Language Models for Automated Classification of Brain MRI Reports and Growth Chart Generation 

**Authors**: Maryam Daniali, Shivaram Karandikar, Dabriel Zimmerman, J. Eric Schmitt, Matthew J. Buczek, Benjamin Jung, Laura Mercedes, Jakob Seidlitz, Vanessa Troiani, Lena Dorfschmidt, Eren Kafadar, Remo Williams, Susan Sotardi, Arastoo Vosough, Scott Haag, Jenna M. Schabdach, Aaron Alexander-Bloch  

**Link**: [PDF](https://arxiv.org/pdf/2503.12143)  

**Abstract**: Clinically acquired brain MRIs and radiology reports are valuable but underutilized resources due to the challenges of manual analysis and data heterogeneity. We developed fine-tuned language models (LMs) to classify brain MRI reports as normal (reports with limited pathology) or abnormal, fine-tuning BERT, BioBERT, ClinicalBERT, and RadBERT on 44,661 reports. We also explored the reasoning capabilities of a leading LM, Gemini 1.5-Pro, for normal report categorization. Automated image processing and modeling generated brain growth charts from LM-classified normal scans, comparing them to human-derived charts. Fine-tuned LMs achieved high classification performance (F1-Score >97%), with unbalanced training mitigating class imbalance. Performance was robust on out-of-distribution data, with full text outperforming summary (impression) sections. Gemini 1.5-Pro showed a promising categorization performance, especially with clinical inference. LM-derived brain growth charts were nearly identical to human-annotated charts (r = 0.99, p < 2.2e-16). Our LMs offer scalable analysis of radiology reports, enabling automated classification of brain MRIs in large datasets. One application is automated generation of brain growth charts for benchmarking quantitative image features. Further research is needed to address data heterogeneity and optimize LM reasoning. 

---
# V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents 

**Authors**: Zhengrong Yue, Shaobin Zhuang, Kunchang Li, Yanbo Ding, Yali Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12077)  

**Abstract**: Despite the recent advancement in video stylization, most existing methods struggle to render any video with complex transitions, based on an open style description of user query. To fill this gap, we introduce a generic multi-agent system for video stylization, V-Stylist, by a novel collaboration and reflection paradigm of multi-modal large language models. Specifically, our V-Stylist is a systematical workflow with three key roles: (1) Video Parser decomposes the input video into a number of shots and generates their text prompts of key shot content. Via a concise video-to-shot prompting paradigm, it allows our V-Stylist to effectively handle videos with complex transitions. (2) Style Parser identifies the style in the user query and progressively search the matched style model from a style tree. Via a robust tree-of-thought searching paradigm, it allows our V-Stylist to precisely specify vague style preference in the open user query. (3) Style Artist leverages the matched model to render all the video shots into the required style. Via a novel multi-round self-reflection paradigm, it allows our V-Stylist to adaptively adjust detail control, according to the style requirement. With such a distinct design of mimicking human professionals, our V-Stylist achieves a major breakthrough over the primary challenges for effective and automatic video stylization. Moreover,we further construct a new benchmark Text-driven Video Stylization Benchmark (TVSBench), which fills the gap to assess stylization of complex videos on open user queries. Extensive experiments show that, V-Stylist achieves the state-of-the-art, e.g.,V-Stylist surpasses FRESCO and ControlVideo by 6.05% and 4.51% respectively in overall average metrics, marking a significant advance in video stylization. 

---
# An LLM-Integrated Framework for Completion, Management, and Tracing of STPA 

**Authors**: Ali Raeisdanaei, Juho Kim, Michael Liao, Sparsh Kochhar  

**Link**: [PDF](https://arxiv.org/pdf/2503.12043)  

**Abstract**: In many safety-critical engineering domains, hazard analysis techniques are an essential part of requirement elicitation. Of the methods proposed for this task, STPA (System-Theoretic Process Analysis) represents a relatively recent development in the field. The completion, management, and traceability of this hazard analysis technique present a time-consuming challenge to the requirements and safety engineers involved. In this paper, we introduce a free, open-source software framework to build STPA models with several automated workflows powered by large language models (LLMs). In past works, LLMs have been successfully integrated into a myriad of workflows across various fields. Here, we demonstrate that LLMs can be used to complete tasks associated with STPA with a high degree of accuracy, saving the time and effort of the human engineers involved. We experimentally validate our method on real-world STPA models built by requirement engineers and researchers. The source code of our software framework is available at the following link: this https URL. 

---
# Maritime Mission Planning for Unmanned Surface Vessel using Large Language Model 

**Authors**: Muhayy Ud Din, Waseem Akram, Ahsan B Bakht, Yihao Dong, Irfan Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2503.12065)  

**Abstract**: Unmanned Surface Vessels (USVs) are essential for various maritime operations. USV mission planning approach offers autonomous solutions for monitoring, surveillance, and logistics. Existing approaches, which are based on static methods, struggle to adapt to dynamic environments, leading to suboptimal performance, higher costs, and increased risk of failure. This paper introduces a novel mission planning framework that uses Large Language Models (LLMs), such as GPT-4, to address these challenges. LLMs are proficient at understanding natural language commands, executing symbolic reasoning, and flexibly adjusting to changing situations. Our approach integrates LLMs into maritime mission planning to bridge the gap between high-level human instructions and executable plans, allowing real-time adaptation to environmental changes and unforeseen obstacles. In addition, feedback from low-level controllers is utilized to refine symbolic mission plans, ensuring robustness and adaptability. This framework improves the robustness and effectiveness of USV operations by integrating the power of symbolic planning with the reasoning abilities of LLMs. In addition, it simplifies the mission specification, allowing operators to focus on high-level objectives without requiring complex programming. The simulation results validate the proposed approach, demonstrating its ability to optimize mission execution while seamlessly adapting to dynamic maritime conditions. 

---
# LLM-Driven Multi-step Translation from C to Rust using Static Analysis 

**Authors**: Tianyang Zhou, Haowen Lin, Somesh Jha, Mihai Christodorescu, Kirill Levchenko, Varun Chandrasekaran  

**Link**: [PDF](https://arxiv.org/pdf/2503.12511)  

**Abstract**: Translating software written in legacy languages to modern languages, such as C to Rust, has significant benefits in improving memory safety while maintaining high performance. However, manual translation is cumbersome, error-prone, and produces unidiomatic code. Large language models (LLMs) have demonstrated promise in producing idiomatic translations, but offer no correctness guarantees as they lack the ability to capture all the semantics differences between the source and target languages. To resolve this issue, we propose SACTOR, an LLM-driven C-to-Rust zero-shot translation tool using a two-step translation methodology: an "unidiomatic" step to translate C into Rust while preserving semantics, and an "idiomatic" step to refine the code to follow Rust's semantic standards. SACTOR utilizes information provided by static analysis of the source C program to address challenges such as pointer semantics and dependency resolution. To validate the correctness of the translated result from each step, we use end-to-end testing via the foreign function interface to embed our translated code segment into the original code. We evaluate the translation of 200 programs from two datasets and two case studies, comparing the performance of GPT-4o, Claude 3.5 Sonnet, Gemini 2.0 Flash, Llama 3.3 70B and DeepSeek-R1 in SACTOR. Our results demonstrate that SACTOR achieves high correctness and improved idiomaticity, with the best-performing model (DeepSeek-R1) reaching 93% and (GPT-4o, Claude 3.5, DeepSeek-R1) reaching 84% correctness (on each dataset, respectively), while producing more natural and Rust-compliant translations compared to existing methods. 

---
# Compose Your Aesthetics: Empowering Text-to-Image Models with the Principles of Art 

**Authors**: Zhe Jin, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2503.12018)  

**Abstract**: Text-to-Image (T2I) diffusion models (DM) have garnered widespread adoption due to their capability in generating high-fidelity outputs and accessibility to anyone able to put imagination into words. However, DMs are often predisposed to generate unappealing outputs, much like the random images on the internet they were trained on. Existing approaches to address this are founded on the implicit premise that visual aesthetics is universal, which is limiting. Aesthetics in the T2I context should be about personalization and we propose the novel task of aesthetics alignment which seeks to align user-specified aesthetics with the T2I generation output. Inspired by how artworks provide an invaluable perspective to approach aesthetics, we codify visual aesthetics using the compositional framework artists employ, known as the Principles of Art (PoA). To facilitate this study, we introduce CompArt, a large-scale compositional art dataset building on top of WikiArt with PoA analysis annotated by a capable Multimodal LLM. Leveraging the expressive power of LLMs and training a lightweight and transferrable adapter, we demonstrate that T2I DMs can effectively offer 10 compositional controls through user-specified PoA conditions. Additionally, we design an appropriate evaluation framework to assess the efficacy of our approach. 

---
# End-to-End Edge AI Service Provisioning Framework in 6G ORAN 

**Authors**: Yun Tang, Udhaya Chandhar Srinivasan, Benjamin James Scott, Obumneme Umealor, Dennis Kevogo, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2503.11933)  

**Abstract**: With the advent of 6G, Open Radio Access Network (O-RAN) architectures are evolving to support intelligent, adaptive, and automated network orchestration. This paper proposes a novel Edge AI and Network Service Orchestration framework that leverages Large Language Model (LLM) agents deployed as O-RAN rApps. The proposed LLM-agent-powered system enables interactive and intuitive orchestration by translating the user's use case description into deployable AI services and corresponding network configurations. The LLM agent automates multiple tasks, including AI model selection from repositories (e.g., Hugging Face), service deployment, network adaptation, and real-time monitoring via xApps. We implement a prototype using open-source O-RAN projects (OpenAirInterface and FlexRIC) to demonstrate the feasibility and functionality of our framework. Our demonstration showcases the end-to-end flow of AI service orchestration, from user interaction to network adaptation, ensuring Quality of Service (QoS) compliance. This work highlights the potential of integrating LLM-driven automation into 6G O-RAN ecosystems, paving the way for more accessible and efficient edge AI ecosystems. 

---
# Comparing Human Expertise and Large Language Models Embeddings in Content Validity Assessment of Personality Tests 

**Authors**: Nicola Milano, Michela Ponticorvo, Davide Marocco  

**Link**: [PDF](https://arxiv.org/pdf/2503.12080)  

**Abstract**: In this article we explore the application of Large Language Models (LLMs) in assessing the content validity of psychometric instruments, focusing on the Big Five Questionnaire (BFQ) and Big Five Inventory (BFI). Content validity, a cornerstone of test construction, ensures that psychological measures adequately cover their intended constructs. Using both human expert evaluations and advanced LLMs, we compared the accuracy of semantic item-construct alignment. Graduate psychology students employed the Content Validity Ratio (CVR) to rate test items, forming the human baseline. In parallel, state-of-the-art LLMs, including multilingual and fine-tuned models, analyzed item embeddings to predict construct mappings. The results reveal distinct strengths and limitations of human and AI approaches. Human validators excelled in aligning the behaviorally rich BFQ items, while LLMs performed better with the linguistically concise BFI items. Training strategies significantly influenced LLM performance, with models tailored for lexical relationships outperforming general-purpose LLMs. Here we highlights the complementary potential of hybrid validation systems that integrate human expertise and AI precision. The findings underscore the transformative role of LLMs in psychological assessment, paving the way for scalable, objective, and robust test development methodologies. 

---
# CoLLMLight: Cooperative Large Language Model Agents for Network-Wide Traffic Signal Control 

**Authors**: Zirui Yuan, Siqi Lai, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11739)  

**Abstract**: Traffic Signal Control (TSC) plays a critical role in urban traffic management by optimizing traffic flow and mitigating congestion. While Large Language Models (LLMs) have recently emerged as promising tools for TSC due to their exceptional problem-solving and generalization capabilities, existing approaches fail to address the essential need for inter-agent coordination, limiting their effectiveness in achieving network-wide optimization. To bridge this gap, we propose CoLLMLight, a cooperative LLM agent framework for TSC. Specifically, we first construct a structured spatiotemporal graph to capture real-time traffic dynamics and spatial relationships among neighboring intersections, enabling the LLM to reason about complex traffic interactions. Moreover, we introduce a complexity-aware reasoning mechanism that dynamically adapts reasoning depth based on real-time traffic conditions, ensuring optimal computational efficiency without sacrificing decision quality. Besides, we propose a fine-tuning strategy that leverages iterative simulation-driven data collection and environmental feedback to build a lightweight LLM tailored for cooperative TSC. Extensive experiments on both synthetic and real-world datasets demonstrate that CoLLMLight outperforms state-of-the-art methods in diverse traffic scenarios, showcasing its effectiveness, scalability, and robustness. 

---
# LLMSeR: Enhancing Sequential Recommendation via LLM-based Data Augmentation 

**Authors**: Yuqi Sun, Qidong Liu, Haiping Zhu, Feng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2503.12547)  

**Abstract**: Sequential Recommender Systems (SRS) have become a cornerstone of online platforms, leveraging users' historical interaction data to forecast their next potential engagement. Despite their widespread adoption, SRS often grapple with the long-tail user dilemma, resulting in less effective recommendations for individuals with limited interaction records. The advent of Large Language Models (LLMs), with their profound capability to discern semantic relationships among items, has opened new avenues for enhancing SRS through data augmentation. Nonetheless, current methodologies encounter obstacles, including the absence of collaborative signals and the prevalence of hallucination this http URL this work, we present LLMSeR, an innovative framework that utilizes Large Language Models (LLMs) to generate pseudo-prior items, thereby improving the efficacy of Sequential Recommender Systems (SRS). To alleviate the challenge of insufficient collaborative signals, we introduce the Semantic Interaction Augmentor (SIA), a method that integrates both semantic and collaborative information to comprehensively augment user interaction data. Moreover, to weaken the adverse effects of hallucination in SRS, we develop the Adaptive Reliability Validation (ARV), a validation technique designed to assess the reliability of the generated pseudo items. Complementing these advancements, we also devise a Dual-Channel Training strategy, ensuring seamless integration of data augmentation into the SRS training this http URL experiments conducted with three widely-used SRS models demonstrate the generalizability and efficacy of LLMSeR. 

---
# Genicious: Contextual Few-shot Prompting for Insights Discovery 

**Authors**: Vineet Kumar, Ronald Tony, Darshita Rathore, Vipasha Rana, Bhuvanesh Mandora, Kanishka, Chetna Bansal, Anindya Moitra  

**Link**: [PDF](https://arxiv.org/pdf/2503.12062)  

**Abstract**: Data and insights discovery is critical for decision-making in modern organizations. We present Genicious, an LLM-aided interface that enables users to interact with tabular datasets and ask complex queries in natural language. By benchmarking various prompting strategies and language models, we have developed an end-to-end tool that leverages contextual few-shot prompting, achieving superior performance in terms of latency, accuracy, and scalability. Genicious empowers stakeholders to explore, analyze and visualize their datasets efficiently while ensuring data security through role-based access control and a Text-to-SQL approach. 

---
