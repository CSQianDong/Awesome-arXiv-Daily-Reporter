# MetaScale: Test-Time Scaling with Evolving Meta-Thoughts 

**Authors**: Qin Liu, Wenxuan Zhou, Nan Xu, James Y. Huang, Fei Wang, Sheng Zhang, Hoifung Poon, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.13447)  

**Abstract**: One critical challenge for large language models (LLMs) for making complex reasoning is their reliance on matching reasoning patterns from training data, instead of proactively selecting the most appropriate cognitive strategy to solve a given task. Existing approaches impose fixed cognitive structures that enhance performance in specific tasks but lack adaptability across diverse scenarios. To address this limitation, we introduce METASCALE, a test-time scaling framework based on meta-thoughts -- adaptive thinking strategies tailored to each task. METASCALE initializes a pool of candidate meta-thoughts, then iteratively selects and evaluates them using a multi-armed bandit algorithm with upper confidence bound selection, guided by a reward model. To further enhance adaptability, a genetic algorithm evolves high-reward meta-thoughts, refining and extending the strategy pool over time. By dynamically proposing and optimizing meta-thoughts at inference time, METASCALE improves both accuracy and generalization across a wide range of tasks. Experimental results demonstrate that MetaScale consistently outperforms standard inference approaches, achieving an 11% performance gain in win rate on Arena-Hard for GPT-4o, surpassing o1-mini by 0.9% under style control. Notably, METASCALE scales more effectively with increasing sampling budgets and produces more structured, expert-level responses. 

---
# Faithfulness of LLM Self-Explanations for Commonsense Tasks: Larger Is Better, and Instruction-Tuning Allows Trade-Offs but Not Pareto Dominance 

**Authors**: Noah Y. Siegel, Nicolas Heess, Maria Perez-Ortiz, Oana-Maria Camburu  

**Link**: [PDF](https://arxiv.org/pdf/2503.13445)  

**Abstract**: As large language models (LLMs) become increasingly capable, ensuring that their self-generated explanations are faithful to their internal decision-making process is critical for safety and oversight. In this work, we conduct a comprehensive counterfactual faithfulness analysis across 62 models from 8 families, encompassing both pretrained and instruction-tuned variants and significantly extending prior studies of counterfactual tests. We introduce phi-CCT, a simplified variant of the Correlational Counterfactual Test, which avoids the need for token probabilities while explaining most of the variance of the original test. Our findings reveal clear scaling trends: larger models are consistently more faithful on our metrics. However, when comparing instruction-tuned and human-imitated explanations, we find that observed differences in faithfulness can often be attributed to explanation verbosity, leading to shifts along the true-positive/false-positive Pareto frontier. While instruction-tuning and prompting can influence this trade-off, we find limited evidence that they fundamentally expand the frontier of explanatory faithfulness beyond what is achievable with pretrained models of comparable size. Our analysis highlights the nuanced relationship between instruction-tuning, verbosity, and the faithful representation of model decision processes. 

---
# SuperBPE: Space Travel for Language Models 

**Authors**: Alisa Liu, Jonathan Hayase, Valentin Hofmann, Sewoong Oh, Noah A. Smith, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2503.13423)  

**Abstract**: The assumption across nearly all language model (LM) tokenization schemes is that tokens should be subwords, i.e., contained within word boundaries. While providing a seemingly reasonable inductive bias, is this common practice limiting the potential of modern LMs? Whitespace is not a reliable delimiter of meaning, as evidenced by multi-word expressions (e.g., "by the way"), crosslingual variation in the number of words needed to express a concept (e.g., "spacesuit helmet" in German is "raumanzughelm"), and languages that do not use whitespace at all (e.g., Chinese). To explore the potential of tokenization beyond subwords, we introduce a "superword" tokenizer, SuperBPE, which incorporates a simple pretokenization curriculum into the byte-pair encoding (BPE) algorithm to first learn subwords, then superwords that bridge whitespace. This brings dramatic improvements in encoding efficiency: when fixing the vocabulary size to 200k, SuperBPE encodes a fixed piece of text with up to 33% fewer tokens than BPE on average. In experiments, we pretrain 8B transformer LMs from scratch while fixing the model size, vocabulary size, and train compute, varying *only* the algorithm for learning the vocabulary. Our model trained with SuperBPE achieves an average +4.0% absolute improvement over the BPE baseline across 30 downstream tasks (including +8.2% on MMLU), while simultaneously requiring 27% less compute at inference time. In analysis, we find that SuperBPE results in segmentations of text that are more uniform in per-token difficulty. Qualitatively, this may be because SuperBPE tokens often capture common multi-word expressions that function semantically as a single unit. SuperBPE is a straightforward, local modification to tokenization that improves both encoding efficiency and downstream performance, yielding better language models overall. 

---
# DLPO: Towards a Robust, Efficient, and Generalizable Prompt Optimization Framework from a Deep-Learning Perspective 

**Authors**: Dengyun Peng, Yuhang Zhou, Qiguang Chen, Jinhao Liu, Jingjing Chen, Libo Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.13413)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across diverse tasks, largely driven by well-designed prompts. However, crafting and selecting such prompts often requires considerable human effort, significantly limiting its scalability. To mitigate this, recent studies have explored automated prompt optimization as a promising solution. Despite these efforts, existing methods still face critical challenges in robustness, efficiency, and generalization. To systematically address these challenges, we first conduct an empirical analysis to identify the limitations of current reflection-based prompt optimization paradigm. Building on these insights, we propose 7 innovative approaches inspired by traditional deep learning paradigms for prompt optimization (DLPO), seamlessly integrating these concepts into text-based gradient optimization. Through these advancements, we progressively tackle the aforementioned challenges and validate our methods through extensive experimentation. We hope our study not only provides valuable guidance for future research but also offers a comprehensive understanding of the challenges and potential solutions in prompt optimization. Our code is available at this https URL. 

---
# Using the Tools of Cognitive Science to Understand Large Language Models at Different Levels of Analysis 

**Authors**: Alexander Ku, Declan Campbell, Xuechunzi Bai, Jiayi Geng, Ryan Liu, Raja Marjieh, R. Thomas McCoy, Andrew Nam, Ilia Sucholutsky, Veniamin Veselovsky, Liyi Zhang, Jian-Qiao Zhu, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2503.13401)  

**Abstract**: Modern artificial intelligence systems, such as large language models, are increasingly powerful but also increasingly hard to understand. Recognizing this problem as analogous to the historical difficulties in understanding the human mind, we argue that methods developed in cognitive science can be useful for understanding large language models. We propose a framework for applying these methods based on Marr's three levels of analysis. By revisiting established cognitive science techniques relevant to each level and illustrating their potential to yield insights into the behavior and internal organization of large language models, we aim to provide a toolkit for making sense of these new kinds of minds. 

---
# Aligned Probing: Relating Toxic Behavior and Model Internals 

**Authors**: Andreas Waldis, Vagrant Gautam, Anne Lauscher, Dietrich Klakow, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2503.13390)  

**Abstract**: We introduce aligned probing, a novel interpretability framework that aligns the behavior of language models (LMs), based on their outputs, and their internal representations (internals). Using this framework, we examine over 20 OLMo, Llama, and Mistral models, bridging behavioral and internal perspectives for toxicity for the first time. Our results show that LMs strongly encode information about the toxicity level of inputs and subsequent outputs, particularly in lower layers. Focusing on how unique LMs differ offers both correlative and causal evidence that they generate less toxic output when strongly encoding information about the input toxicity. We also highlight the heterogeneity of toxicity, as model behavior and internals vary across unique attributes such as Threat. Finally, four case studies analyzing detoxification, multi-prompt evaluations, model quantization, and pre-training dynamics underline the practical impact of aligned probing with further concrete insights. Our findings contribute to a more holistic understanding of LMs, both within and beyond the context of toxicity. 

---
# Valid Text-to-SQL Generation with Unification-based DeepStochLog 

**Authors**: Ying Jiao, Luc De Raedt, Giuseppe Marra  

**Link**: [PDF](https://arxiv.org/pdf/2503.13342)  

**Abstract**: Large language models have been used to translate natural language questions to SQL queries. Without hard constraints on syntax and database schema, they occasionally produce invalid queries that are not executable. These failures limit the usage of these systems in real-life scenarios. We propose a neurosymbolic framework that imposes SQL syntax and schema constraints with unification-based definite clause grammars and thus guarantees the generation of valid queries. Our framework also builds a bi-directional interface to language models to leverage their natural language understanding abilities. The evaluation results on a subset of SQL grammars show that all our output queries are valid. This work is the first step towards extending language models with unification-based grammars. We demonstrate this extension enhances the validity, execution accuracy, and ground truth alignment of the underlying language model by a large margin. Our code is available at this https URL. 

---
# Reliable and Efficient Amortized Model-based Evaluation 

**Authors**: Sang Truong, Yuheng Tu, Percy Liang, Bo Li, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2503.13335)  

**Abstract**: Comprehensive evaluations of language models (LM) during both development and deployment phases are necessary because these models possess numerous capabilities (e.g., mathematical reasoning, legal support, or medical diagnostic) as well as safety risks (e.g., racial bias, toxicity, or misinformation). The average score across a wide range of benchmarks provides a signal that helps guide the use of these LMs in practice. Currently, holistic evaluations are costly due to the large volume of benchmark questions, making frequent evaluations impractical. A popular attempt to lower the cost is to compute the average score on a subset of the benchmark. This approach, unfortunately, often renders an unreliable measure of LM performance because the average score is often confounded with the difficulty of the questions in the benchmark subset. Item response theory (IRT) was designed to address this challenge, providing a reliable measurement by careful controlling for question difficulty. Unfortunately, question difficulty is expensive to estimate. Facing this challenge, we train a model that predicts question difficulty from its content, enabling a reliable measurement at a fraction of the cost. In addition, we leverage this difficulty predictor to further improve the evaluation efficiency through training a question generator given a difficulty level. This question generator is essential in adaptive testing, where, instead of using a random subset of the benchmark questions, informative questions are adaptively chosen based on the current estimation of LLM performance. Experiments on 22 common natural language benchmarks and 172 LMs show that this approach is more reliable and efficient compared to current common practice. 

---
# Computation Mechanism Behind LLM Position Generalization 

**Authors**: Chi Han, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2503.13305)  

**Abstract**: Most written natural languages are composed of sequences of words and sentences. Similar to humans, large language models (LLMs) exhibit flexibility in handling textual positions - a phenomenon we term position generalization. They can understand texts with position perturbations and generalize to longer texts than those encountered during training with the latest techniques. These phenomena suggest that LLMs handle positions tolerantly, but how LLMs computationally process positional relevance remains largely unexplored. This work connects the linguistic phenomenon with LLMs' computational mechanisms. We show how LLMs enforce certain computational mechanisms for the aforementioned tolerance in position perturbations. Despite the complex design of the self-attention mechanism, this work reveals that LLMs learn a counterintuitive disentanglement of attention logits. Their values show a 0.959 linear correlation with an approximation of the arithmetic sum of positional relevance and semantic importance. Furthermore, we identify a prevalent pattern in intermediate features, which we prove theoretically enables this effect. The pattern, which is different from how randomly initialized parameters would behave, suggests that it is a learned behavior rather than a natural result of the model architecture. Based on these findings, we provide computational explanations and criteria for LLMs' position flexibilities. This work takes a pioneering step in linking position generalization with modern LLMs' internal mechanisms. 

---
# A Survey on Transformer Context Extension: Approaches and Evaluation 

**Authors**: Yijun Liu, Jinzheng Yu, Yang Xu, Zhongyang Li, Qingfu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.13299)  

**Abstract**: Large language models (LLMs) based on Transformer have been widely applied in the filed of natural language processing (NLP), demonstrating strong performance, particularly in handling short text tasks. However, when it comes to long context scenarios, the performance of LLMs degrades due to some challenges. To alleviate this phenomenon, there is a number of work proposed recently. In this survey, we first list the challenges of applying pre-trained LLMs to process long contexts. Then systematically review the approaches related to long context and propose our taxonomy categorizing them into four main types: positional encoding, context compression, retrieval augmented, and attention pattern. In addition to the approaches, we focus on the evaluation of long context, organizing relevant data, tasks, and metrics based on existing long context benchmarks. Finally, we summarize unresolved issues in the long context domain and put forward our views on future developments. 

---
# LLM-Match: An Open-Sourced Patient Matching Model Based on Large Language Models and Retrieval-Augmented Generation 

**Authors**: Xiaodi Li, Shaika Chowdhury, Chung Il Wi, Maria Vassilaki, Ken Liu, Terence T Sio, Owen Garrick, Young J Juhn, James R Cerhan, Cui Tao, Nansu Zong  

**Link**: [PDF](https://arxiv.org/pdf/2503.13281)  

**Abstract**: Patient matching is the process of linking patients to appropriate clinical trials by accurately identifying and matching their medical records with trial eligibility criteria. We propose LLM-Match, a novel framework for patient matching leveraging fine-tuned open-source large language models. Our approach consists of four key components. First, a retrieval-augmented generation (RAG) module extracts relevant patient context from a vast pool of electronic health records (EHRs). Second, a prompt generation module constructs input prompts by integrating trial eligibility criteria (both inclusion and exclusion criteria), patient context, and system instructions. Third, a fine-tuning module with a classification head optimizes the model parameters using structured prompts and ground-truth labels. Fourth, an evaluation module assesses the fine-tuned model's performance on the testing datasets. We evaluated LLM-Match on four open datasets, n2c2, SIGIR, TREC 2021, and TREC 2022, using open-source models, comparing it against TrialGPT, Zero-Shot, and GPT-4-based closed models. LLM-Match outperformed all baselines. 

---
# TablePilot; Recommending Human-Preferred Tabular Data Analysis with Large Language Models 

**Authors**: Deyin Yi, Yihao Liu, Lang Cao, Mengyu Zhou, Haoyu Dong, Shi Han, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13262)  

**Abstract**: Tabular data analysis is crucial in many scenarios, yet efficiently identifying the most relevant data analysis queries and results for a new table remains a significant challenge. The complexity of tabular data, diverse analytical operations, and the demand for high-quality analysis make the process tedious. To address these challenges, we aim to recommend query-code-result triplets tailored for new tables in tabular data analysis workflows. In this paper, we present TablePilot, a pioneering tabular data analysis framework leveraging large language models to autonomously generate comprehensive and superior analytical results without relying on user profiles or prior interactions. The framework incorporates key designs in analysis preparation and analysis optimization to enhance accuracy. Additionally, we propose Rec-Align, a novel method to further improve recommendation quality and better align with human preferences. Experiments on DART, a dataset specifically designed for comprehensive tabular data analysis recommendation, demonstrate the effectiveness of our framework. Based on GPT-4o, the tuned TablePilot achieves 77.0% top-5 recommendation recall. Human evaluations further highlight its effectiveness in optimizing tabular data analysis workflows. 

---
# Can Language Models Follow Multiple Turns of Entangled Instructions? 

**Authors**: Chi Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.13222)  

**Abstract**: Despite significant achievements in improving the instruction-following capabilities of large language models (LLMs), the ability to process multiple potentially entangled or conflicting instructions remains a considerable challenge. Real-world scenarios often require consistency across multiple instructions over time, such as secret privacy, personal preferences, and prioritization, which demand sophisticated abilities to integrate multiple turns and carefully balance competing objectives when instructions intersect or conflict. This work presents a systematic investigation of LLMs' capabilities in handling multiple turns of instructions, covering three levels of difficulty: (1) retrieving information from instructions, (2) tracking and reasoning across turns, and (3) resolving conflicts among instructions. We construct MultiTurnInstruct with around 1.1K high-quality multi-turn conversations through the human-in-the-loop approach and result in nine capability categories, including statics and dynamics, reasoning, and multitasking. Our finding reveals an intriguing trade-off between different capabilities. While GPT models demonstrate superior memorization, they show reduced effectiveness in privacy-protection tasks requiring selective information withholding. Larger models exhibit stronger reasoning capabilities but still struggle with resolving conflicting instructions. Importantly, these performance gaps cannot be attributed solely to information loss, as models demonstrate strong BLEU scores on memorization tasks but their attention mechanisms fail to integrate multiple related instructions effectively. These findings highlight critical areas for improvement in complex real-world tasks involving multi-turn instructions. 

---
# Improving Complex Reasoning with Dynamic Prompt Corruption: A soft prompt Optimization Approach 

**Authors**: Sinan Fan, Liang Xie, Chen Shen, Ge Teng, Xiaosong Yuan, Xiaofeng Zhang, Chenxi Huang, Wenxiao Wang, Xiaofei He, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.13208)  

**Abstract**: Prompt-tuning (PT) for large language models (LLMs) can facilitate the performance on various conventional NLP tasks with significantly fewer trainable parameters. However, our investigation reveals that PT provides limited improvement and may even degrade the primitive performance of LLMs on complex reasoning tasks. Such a phenomenon suggests that soft prompts can positively impact certain instances while negatively affecting others, particularly during the later phases of reasoning. To address these challenges, We first identify an information accumulation within the soft prompts. Through detailed analysis, we demonstrate that this phenomenon is often accompanied by erroneous information flow patterns in the deeper layers of the model, which ultimately lead to incorrect reasoning outcomes. we propose a novel method called \textbf{D}ynamic \textbf{P}rompt \textbf{C}orruption (DPC) to take better advantage of soft prompts in complex reasoning tasks, which dynamically adjusts the influence of soft prompts based on their impact on the reasoning process. Specifically, DPC consists of two stages: Dynamic Trigger and Dynamic Corruption. First, Dynamic Trigger measures the impact of soft prompts, identifying whether beneficial or detrimental. Then, Dynamic Corruption mitigates the negative effects of soft prompts by selectively masking key tokens that interfere with the reasoning process. We validate the proposed approach through extensive experiments on various LLMs and reasoning tasks, including GSM8K, MATH, and AQuA. Experimental results demonstrate that DPC can consistently enhance the performance of PT, achieving 4\%-8\% accuracy gains compared to vanilla prompt tuning, highlighting the effectiveness of our approach and its potential to enhance complex reasoning in LLMs. 

---
# Code-Driven Inductive Synthesis: Enhancing Reasoning Abilities of Large Language Models with Sequences 

**Authors**: Kedi Chen, Zhikai Lei, Fan Zhang, Yinqi Zhang, Qin Chen, Jie Zhou, Liang He, Qipeng Guo, Kai Chen, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13109)  

**Abstract**: Large language models make remarkable progress in reasoning capabilities. Existing works focus mainly on deductive reasoning tasks (e.g., code and math), while another type of reasoning mode that better aligns with human learning, inductive reasoning, is not well studied. We attribute the reason to the fact that obtaining high-quality process supervision data is challenging for inductive reasoning. Towards this end, we novelly employ number sequences as the source of inductive reasoning data. We package sequences into algorithmic problems to find the general term of each sequence through a code solution. In this way, we can verify whether the code solution holds for any term in the current sequence, and inject case-based supervision signals by using code unit tests. We build a sequence synthetic data pipeline and form a training dataset CodeSeq. Experimental results show that the models tuned with CodeSeq improve on both code and comprehensive reasoning benchmarks. 

---
# REPA: Russian Error Types Annotation for Evaluating Text Generation and Judgment Capabilities 

**Authors**: Alexander Pugachev, Alena Fenogenova, Vladislav Mikhailov, Ekaterina Artemova  

**Link**: [PDF](https://arxiv.org/pdf/2503.13102)  

**Abstract**: Recent advances in large language models (LLMs) have introduced the novel paradigm of using LLMs as judges, where an LLM evaluates and scores the outputs of another LLM, which often correlates highly with human preferences. However, the use of LLM-as-a-judge has been primarily studied in English. In this paper, we evaluate this framework in Russian by introducing the Russian Error tyPes Annotation dataset (REPA), a dataset of 1k user queries and 2k LLM-generated responses. Human annotators labeled each response pair expressing their preferences across ten specific error types, as well as selecting an overall preference. We rank six generative LLMs across the error types using three rating systems based on human preferences. We also evaluate responses using eight LLM judges in zero-shot and few-shot settings. We describe the results of analyzing the judges and position and length biases. Our findings reveal a notable gap between LLM judge performance in Russian and English. However, rankings based on human and LLM preferences show partial alignment, suggesting that while current LLM judges struggle with fine-grained evaluation in Russian, there is potential for improvement. 

---
# Who Wrote This? Identifying Machine vs Human-Generated Text in Hausa 

**Authors**: Babangida Sani, Aakansha Soy, Sukairaj Hafiz Imam, Ahmad Mustapha, Lukman Jibril Aliyu, Idris Abdulmumin, Ibrahim Said Ahmad, Shamsuddeen Hassan Muhammad  

**Link**: [PDF](https://arxiv.org/pdf/2503.13101)  

**Abstract**: The advancement of large language models (LLMs) has allowed them to be proficient in various tasks, including content generation. However, their unregulated usage can lead to malicious activities such as plagiarism and generating and spreading fake news, especially for low-resource languages. Most existing machine-generated text detectors are trained on high-resource languages like English, French, etc. In this study, we developed the first large-scale detector that can distinguish between human- and machine-generated content in Hausa. We scrapped seven Hausa-language media outlets for the human-generated text and the Gemini-2.0 flash model to automatically generate the corresponding Hausa-language articles based on the human-generated article headlines. We fine-tuned four pre-trained Afri-centric models (AfriTeVa, AfriBERTa, AfroXLMR, and AfroXLMR-76L) on the resulting dataset and assessed their performance using accuracy and F1-score metrics. AfroXLMR achieved the highest performance with an accuracy of 99.23% and an F1 score of 99.21%, demonstrating its effectiveness for Hausa text detection. Our dataset is made publicly available to enable further research. 

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
# Overview of the NTCIR-18 Automatic Evaluation of LLMs (AEOLLM) Task 

**Authors**: Junjie Chen, Haitao Li, Zhumin Chu, Yiqun Liu, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2503.13038)  

**Abstract**: In this paper, we provide an overview of the NTCIR-18 Automatic Evaluation of LLMs (AEOLLM) task. As large language models (LLMs) grow popular in both academia and industry, how to effectively evaluate the capacity of LLMs becomes an increasingly critical but still challenging issue. Existing methods can be divided into two types: manual evaluation, which is expensive, and automatic evaluation, which faces many limitations including task format (the majority belong to multiple-choice questions) and evaluation criteria (occupied by reference-based metrics). To advance the innovation of automatic evaluation, we propose the AEOLLM task which focuses on generative tasks and encourages reference-free methods. Besides, we set up diverse subtasks such as dialogue generation, text expansion, summary generation and non-factoid question answering to comprehensively test different methods. This year, we received 48 runs from 4 teams in total. This paper will describe the background of the task, the data set, the evaluation measures and the evaluation results, respectively. 

---
# Halving transcription time: A fast, user-friendly and GDPR-compliant workflow to create AI-assisted transcripts for content analysis 

**Authors**: Jakob Sponholz, Andreas Weilinghoff, Juliane Schopf  

**Link**: [PDF](https://arxiv.org/pdf/2503.13031)  

**Abstract**: In qualitative research, data transcription is often labor-intensive and time-consuming. To expedite this process, a workflow utilizing artificial intelligence (AI) was developed. This workflow not only enhances transcription speed but also addresses the issue of AI-generated transcripts often lacking compatibility with standard content analysis software. Within this workflow, automatic speech recognition is employed to create initial transcripts from audio recordings, which are then formatted to be compatible with content analysis software such as this http URL or MAXQDA. Empirical data from a study of 12 interviews suggests that this workflow can reduce transcription time by up to 46.2%. Furthermore, by using widely used standard software, this process is suitable for both students and researchers while also being adaptable to a variety of learning, teaching, and research environments. It is also particularly beneficial for non-native speakers. In addition, the workflow is GDPR-compliant and facilitates local, offline transcript generation, which is crucial when dealing with sensitive data. 

---
# Dynamic Relation Inference via Verb Embeddings 

**Authors**: Omri Suissa, Muhiim Ali, Ariana Azarbal, Hui Shen, Shekhar Pradhan  

**Link**: [PDF](https://arxiv.org/pdf/2503.13021)  

**Abstract**: CLIP has demonstrated exceptional image-text matching capabilities due to its training on contrastive learning tasks. Past research has suggested that whereas CLIP effectively matches text to images when the matching can be achieved just by matching the text with the objects in the image, CLIP struggles when the matching depends on representing the relationship among the objects in the images (i.e., inferring relations). Previous attempts to address this limitation by training CLIP on relation detection datasets with only linguistic supervision have met with limited success. In this paper, we offer insights and practical methods to advance the field of relation inference from images. This paper approaches the task of creating a model that effectively detects relations among the objects in images by producing text and image embeddings that capture relationships through linguistic supervision. To this end, we propose Dynamic Relation Inference via Verb Embeddings (DRIVE), which augments the COCO dataset, fine-tunes CLIP with hard negatives subject-relation-object triples and corresponding images, and introduces a novel loss function to improve relation detection. Evaluated on multiple CLIP-based models, our method significantly improves zero-shot relation inference accuracy in both frozen and fine-tuned settings, significantly outperforming CLIP and state-of-the-art models while generalizing well on unseen data. 

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
# ThinkPatterns-21k: A Systematic Study on the Impact of Thinking Patterns in LLMs 

**Authors**: Pengcheng Wen, Jiaming Ji, Chi-Min Chan, Juntao Dai, Donghai Hong, Yaodong Yang, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2503.12918)  

**Abstract**: Large language models (LLMs) have demonstrated enhanced performance through the \textit{Thinking then Responding} paradigm, where models generate internal thoughts before final responses (aka, System 2 thinking). However, existing research lacks a systematic understanding of the mechanisms underlying how thinking patterns affect performance across model sizes. In this work, we conduct a comprehensive analysis of the impact of various thinking types on model performance and introduce ThinkPatterns-21k, a curated dataset comprising 21k instruction-response pairs (QA) collected from existing instruction-following datasets with five thinking types. For each pair, we augment it with five distinct internal thinking patterns: one unstructured thinking (monologue) and four structured variants (decomposition, self-ask, self-debate and self-critic), while maintaining the same instruction and response. Through extensive evaluation across different model sizes (3B-32B parameters), we have two key findings: (1) smaller models (<30B parameters) can benefit from most of structured thinking patterns, while larger models (32B) with structured thinking like decomposition would degrade performance and (2) unstructured monologue demonstrates broad effectiveness across different model sizes. Finally, we released all of our datasets, checkpoints, training logs of diverse thinking patterns to reproducibility, aiming to facilitate further research in this direction. 

---
# HICD: Hallucination-Inducing via Attention Dispersion for Contrastive Decoding to Mitigate Hallucinations in Large Language Models 

**Authors**: Xinyan Jiang, Hang Ye, Yongxin Zhu, Xiaoying Zheng, Zikang Chen, Jun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.12908)  

**Abstract**: Large Language Models (LLMs) often generate hallucinations, producing outputs that are contextually inaccurate or factually incorrect. We introduce HICD, a novel method designed to induce hallucinations for contrastive decoding to mitigate hallucinations. Unlike existing contrastive decoding methods, HICD selects attention heads crucial to the model's prediction as inducing heads, then induces hallucinations by dispersing attention of these inducing heads and compares the hallucinated outputs with the original outputs to obtain the final result. Our approach significantly improves performance on tasks requiring contextual faithfulness, such as context completion, reading comprehension, and question answering. It also improves factuality in tasks requiring accurate knowledge recall. We demonstrate that our inducing heads selection and attention dispersion method leads to more "contrast-effective" hallucinations for contrastive decoding, outperforming other hallucination-inducing methods. Our findings provide a promising strategy for reducing hallucinations by inducing hallucinations in a controlled manner, enhancing the performance of LLMs in a wide range of tasks. 

---
# DAPI: Domain Adaptive Toxicity Probe Vector Intervention for Fine-Grained Detoxification 

**Authors**: Cho Hyeonsu, Dooyoung Kim, Youngjoong Ko  

**Link**: [PDF](https://arxiv.org/pdf/2503.12882)  

**Abstract**: There have been attempts to utilize linear probe for detoxification, with existing studies relying on a single toxicity probe vector to reduce toxicity. However, toxicity can be fine-grained into various subcategories, making it difficult to remove certain types of toxicity by using a single toxicity probe vector. To address this limitation, we propose a category-specific toxicity probe vector approach. First, we train multiple toxicity probe vectors for different toxicity categories. During generation, we dynamically select the most relevant toxicity probe vector based on the current context. Finally, the selected vector is dynamically scaled and subtracted from model. Our method successfully mitigated toxicity from categories that the single probe vector approach failed to detoxify. Experiments demonstrate that our approach achieves up to a 78.52% reduction in toxicity on the evaluation dataset, while fluency remains nearly unchanged, with only a 0.052% drop compared to the unsteered model. 

---
# nvBench 2.0: A Benchmark for Natural Language to Visualization under Ambiguity 

**Authors**: Tianqi Luo, Chuhan Huang, Leixian Shen, Boyan Li, Shuyu Shen, Wei Zeng, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.12880)  

**Abstract**: Natural Language to Visualization (NL2VIS) enables users to create visualizations from natural language queries, making data insights more accessible. However, NL2VIS faces challenges in interpreting ambiguous queries, as users often express their visualization needs in imprecise language. To address this challenge, we introduce nvBench 2.0, a new benchmark designed to evaluate NL2VIS systems in scenarios involving ambiguous queries. nvBench 2.0 includes 7,878 natural language queries and 24,076 corresponding visualizations, derived from 780 tables across 153 domains. It is built using a controlled ambiguity-injection pipeline that generates ambiguous queries through a reverse-generation workflow. By starting with unambiguous seed visualizations and selectively injecting ambiguities, the pipeline yields multiple valid interpretations for each query, with each ambiguous query traceable to its corresponding visualization through step-wise reasoning paths. We evaluate various Large Language Models (LLMs) on their ability to perform ambiguous NL2VIS tasks using nvBench 2.0. We also propose Step-NL2VIS, an LLM-based model trained on nvBench 2.0, which enhances performance in ambiguous scenarios through step-wise preference optimization. Our results show that Step-NL2VIS outperforms all baselines, setting a new state-of-the-art for ambiguous NL2VIS tasks. 

---
# Harnessing Test-time Adaptation for NLU tasks Involving Dialects of English 

**Authors**: Duke Nguyen, Aditya Joshi, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2503.12858)  

**Abstract**: Test-time adaptation (TTA) is an excellent method which helps generalize models across domains, tasks, and distributions without the use of labeled datasets. Thus, TTA is very useful in natural language processing (NLP) in the dialectal setting, since oftentimes, models are trained on Standard American English (SAE), evaluated on Indian English or Nigerian English, of which distribution differs significantly from the former. This is especially useful since dialectal datasets are scarce. In this paper, we explore one of the most famous TTA techniques, SHOT, in dialectal NLP. We finetune and evaluate SHOT on different combinations of dialectal GLUE. Our findings show that SHOT is a viable technique when labeled datasets are unavailable. We also theoretically propose the concept of dialectal gap and show that it has a positive correlation with the effectiveness of SHOT. We also find that in many cases, finetuning on SAE yields higher performance than finetuning on dialectal data. Our code is available at this https URL 

---
# Enhancing LLM Reasoning with Iterative DPO: A Comprehensive Empirical Investigation 

**Authors**: Songjun Tu, Jiahao Lin, Xiangyu Tian, Qichao Zhang, Linjing Li, Yuqian Fu, Nan Xu, Wei He, Xiangyuan Lan, Dongmei Jiang, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.12854)  

**Abstract**: Recent advancements in post-training methodologies for large language models (LLMs) have highlighted reinforcement learning (RL) as a critical component for enhancing reasoning. However, the substantial computational costs associated with RL-based approaches have led to growing interest in alternative paradigms, such as Direct Preference Optimization (DPO). In this study, we investigate the effectiveness of DPO in facilitating self-improvement for LLMs through iterative preference-based learning. We demonstrate that a single round of DPO with coarse filtering significantly enhances mathematical reasoning performance, particularly for strong base model. Furthermore, we design an iterative enhancement framework for both the generator and the reward model (RM), enabling their mutual improvement through online interaction across multiple rounds of DPO. Finally, with simple verifiable rewards, our model DPO-VP achieves RL-level performance with significantly lower computational overhead. These findings highlight DPO as a scalable and cost-effective alternative to RL, offering a practical solution for enhancing LLM reasoning in resource-constrained situations. 

---
# Modelling Child Learning and Parsing of Long-range Syntactic Dependencies 

**Authors**: Louis Mahon, Mark Johnson, Mark Steedman  

**Link**: [PDF](https://arxiv.org/pdf/2503.12832)  

**Abstract**: This work develops a probabilistic child language acquisition model to learn a range of linguistic phenonmena, most notably long-range syntactic dependencies of the sort found in object wh-questions, among other constructions. The model is trained on a corpus of real child-directed speech, where each utterance is paired with a logical form as a meaning representation. It then learns both word meanings and language-specific syntax simultaneously. After training, the model can deduce the correct parse tree and word meanings for a given utterance-meaning pair, and can infer the meaning if given only the utterance. The successful modelling of long-range dependencies is theoretically important because it exploits aspects of the model that are, in general, trans-context-free. 

---
# Leveraging Deep Neural Networks for Aspect-Based Sentiment Classification 

**Authors**: Chen Li, Debo Cheng, Yasuhiko Morimoto  

**Link**: [PDF](https://arxiv.org/pdf/2503.12803)  

**Abstract**: Aspect-based sentiment analysis seeks to determine sentiment with a high level of detail. While graph convolutional networks (GCNs) are commonly used for extracting sentiment features, their straightforward use in syntactic feature extraction can lead to a loss of crucial information. This paper presents a novel edge-enhanced GCN, called EEGCN, which improves performance by preserving feature integrity as it processes syntactic graphs. We incorporate a bidirectional long short-term memory (Bi-LSTM) network alongside a self-attention-based transformer for effective text encoding, ensuring the retention of long-range dependencies. A bidirectional GCN (Bi-GCN) with message passing then captures the relationships between entities, while an aspect-specific masking technique removes extraneous information. Extensive evaluations and ablation studies on four benchmark datasets show that EEGCN significantly enhances aspect-based sentiment analysis, overcoming issues with syntactic feature extraction and advancing the field's methodologies. 

---
# RAG-RL: Advancing Retrieval-Augmented Generation via RL and Curriculum Learning 

**Authors**: Jerry Huang, Siddarth Madala, Risham Sidhu, Cheng Niu, Julia Hockenmaier, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12759)  

**Abstract**: Recent research highlights the challenges retrieval models face in retrieving useful contexts and the limitations of generation models in effectively utilizing those contexts in retrieval-augmented generation (RAG) settings. To address these challenges, we introduce RAG-RL, the first reasoning language model (RLM) specifically trained for RAG. RAG-RL demonstrates that stronger answer generation models can identify relevant contexts within larger sets of retrieved information -- thereby alleviating the burden on retrievers -- while also being able to utilize those contexts more effectively. Moreover, we show that curriculum design in the reinforcement learning (RL) post-training process is a powerful approach to enhancing model performance. We benchmark our method on two open-domain question-answering datasets and achieve state-of-the-art results, surpassing previous SOTA generative reader models. In addition, we offers empirical insights into various curriculum learning strategies, providing a deeper understanding of their impact on model performance. 

---
# TNCSE: Tensor's Norm Constraints for Unsupervised Contrastive Learning of Sentence Embeddings 

**Authors**: Tianyu Zong, Bingkang Shi, Hongzhu Yi, Jungang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12739)  

**Abstract**: Unsupervised sentence embedding representation has become a hot research topic in natural language processing. As a tensor, sentence embedding has two critical properties: direction and norm. Existing works have been limited to constraining only the orientation of the samples' representations while ignoring the features of their module lengths. To address this issue, we propose a new training objective that optimizes the training of unsupervised contrastive learning by constraining the module length features between positive samples. We combine the training objective of Tensor's Norm Constraints with ensemble learning to propose a new Sentence Embedding representation framework, TNCSE. We evaluate seven semantic text similarity tasks, and the results show that TNCSE and derived models are the current state-of-the-art approach; in addition, we conduct extensive zero-shot evaluations, and the results show that TNCSE outperforms other baselines. 

---
# Plausibility Vaccine: Injecting LLM Knowledge for Event Plausibility 

**Authors**: Jacob Chmura, Jonah Dauvet, Sebastian Sabry  

**Link**: [PDF](https://arxiv.org/pdf/2503.12667)  

**Abstract**: Despite advances in language modelling, distributional methods that build semantic representations from co-occurrences fail to discriminate between plausible and implausible events. In this work, we investigate how plausibility prediction can be improved by injecting latent knowledge prompted from large language models using parameter-efficient fine-tuning. We train 12 task adapters to learn various physical properties and association measures and perform adapter fusion to compose latent semantic knowledge from each task on top of pre-trained AlBERT embeddings. We automate auxiliary task data generation, which enables us to scale our approach and fine-tune our learned representations across two plausibility datasets. Our code is available at this https URL. 

---
# UniBERTs: Adversarial Training for Language-Universal Representations 

**Authors**: Andrei-Marius Avram, Marian Lupaşcu, Dumitru-Clementin Cercel, Ionuţ Mironică, Ştefan Trăuşan-Matu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12608)  

**Abstract**: This paper presents UniBERT, a compact multilingual language model that leverages an innovative training framework integrating three components: masked language modeling, adversarial training, and knowledge distillation. Pre-trained on a meticulously curated Wikipedia corpus spanning 107 languages, UniBERT is designed to reduce the computational demands of large-scale models while maintaining competitive performance across various natural language processing tasks. Comprehensive evaluations on four tasks -- named entity recognition, natural language inference, question answering, and semantic textual similarity -- demonstrate that our multilingual training strategy enhanced by an adversarial objective significantly improves cross-lingual generalization. Specifically, UniBERT models show an average relative improvement of 7.72% over traditional baselines, which achieved an average relative improvement of only 1.17%, with statistical analysis confirming the significance of these gains (p-value = 0.0181). This work highlights the benefits of combining adversarial training and knowledge distillation to build scalable and robust language models, thereby advancing the field of multilingual and cross-lingual natural language processing. 

---
# RaSA: Rank-Sharing Low-Rank Adaptation 

**Authors**: Zhiwei He, Zhaopeng Tu, Xing Wang, Xingyu Chen, Zhijie Wang, Jiahao Xu, Tian Liang, Wenxiang Jiao, Zhuosheng Zhang, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12576)  

**Abstract**: Low-rank adaptation (LoRA) has been prominently employed for parameter-efficient fine-tuning of large language models (LLMs). However, the limited expressive capacity of LoRA, stemming from the low-rank constraint, has been recognized as a bottleneck, particularly in rigorous tasks like code generation and mathematical reasoning. To address this limitation, we introduce Rank-Sharing Low-Rank Adaptation (RaSA), an innovative extension that enhances the expressive capacity of LoRA by leveraging partial rank sharing across layers. By forming a shared rank pool and applying layer-specific weighting, RaSA effectively increases the number of ranks without augmenting parameter overhead. Our theoretically grounded and empirically validated approach demonstrates that RaSA not only maintains the core advantages of LoRA but also significantly boosts performance in challenging code and math tasks. Code, data and scripts are available at: this https URL. 

---
# Multi-Granular Multimodal Clue Fusion for Meme Understanding 

**Authors**: Li Zheng, Hao Fei, Ting Dai, Zuquan Peng, Fei Li, Huisheng Ma, Chong Teng, Donghong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2503.12560)  

**Abstract**: With the continuous emergence of various social media platforms frequently used in daily life, the multimodal meme understanding (MMU) task has been garnering increasing attention. MMU aims to explore and comprehend the meanings of memes from various perspectives by performing tasks such as metaphor recognition, sentiment analysis, intention detection, and offensiveness detection. Despite making progress, limitations persist due to the loss of fine-grained metaphorical visual clue and the neglect of multimodal text-image weak correlation. To overcome these limitations, we propose a multi-granular multimodal clue fusion model (MGMCF) to advance MMU. Firstly, we design an object-level semantic mining module to extract object-level image feature clues, achieving fine-grained feature clue extraction and enhancing the model's ability to capture metaphorical details and semantics. Secondly, we propose a brand-new global-local cross-modal interaction model to address the weak correlation between text and images. This model facilitates effective interaction between global multimodal contextual clues and local unimodal feature clues, strengthening their representations through a bidirectional cross-modal attention mechanism. Finally, we devise a dual-semantic guided training strategy to enhance the model's understanding and alignment of multimodal representations in the semantic space. Experiments conducted on the widely-used MET-MEME bilingual dataset demonstrate significant improvements over state-of-the-art baselines. Specifically, there is an 8.14% increase in precision for offensiveness detection task, and respective accuracy enhancements of 3.53%, 3.89%, and 3.52% for metaphor recognition, sentiment analysis, and intention detection tasks. These results, underpinned by in-depth analyses, underscore the effectiveness and potential of our approach for advancing MMU. 

---
# From Guessing to Asking: An Approach to Resolving the Persona Knowledge Gap in LLMs during Multi-Turn Conversations 

**Authors**: Sarvesh Baskar, Tanmay Tulsidas Verelakar, Srinivasan Parthasarathy, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2503.12556)  

**Abstract**: In multi-turn dialogues, large language models (LLM) face a critical challenge of ensuring coherence while adapting to user-specific information. This study introduces the persona knowledge gap, the discrepancy between a model's internal understanding and the knowledge required for coherent, personalized conversations. While prior research has recognized these gaps, computational methods for their identification and resolution remain underexplored. We propose Conversation Preference Elicitation and Recommendation (CPER), a novel framework that dynamically detects and resolves persona knowledge gaps using intrinsic uncertainty quantification and feedback-driven refinement. CPER consists of three key modules: a Contextual Understanding Module for preference extraction, a Dynamic Feedback Module for measuring uncertainty and refining persona alignment, and a Persona-Driven Response Generation module for adapting responses based on accumulated user context. We evaluate CPER on two real-world datasets: CCPE-M for preferential movie recommendations and ESConv for mental health support. Using A/B testing, human evaluators preferred CPER's responses 42% more often than baseline models in CCPE-M and 27% more often in ESConv. A qualitative human evaluation confirms that CPER's responses are preferred for maintaining contextual relevance and coherence, particularly in longer (12+ turn) conversations. 

---
# Basic Category Usage in Vision Language Models 

**Authors**: Hunter Sawyer, Jesse Roberts, Kyle Moore  

**Link**: [PDF](https://arxiv.org/pdf/2503.12530)  

**Abstract**: The field of psychology has long recognized a basic level of categorization that humans use when labeling visual stimuli, a term coined by Rosch in 1976. This level of categorization has been found to be used most frequently, to have higher information density, and to aid in visual language tasks with priming in humans. Here, we investigate basic level categorization in two recently released, open-source vision-language models (VLMs). This paper demonstrates that Llama 3.2 Vision Instruct (11B) and Molmo 7B-D both prefer basic level categorization consistent with human behavior. Moreover, the models' preferences are consistent with nuanced human behaviors like the biological versus non-biological basic level effects and the well established expert basic level shift, further suggesting that VLMs acquire cognitive categorization behaviors from the human data on which they are trained. 

---
# Investigating Human-Aligned Large Language Model Uncertainty 

**Authors**: Kyle Moore, Jesse Roberts, Daryl Watson, Pamela Wisniewski  

**Link**: [PDF](https://arxiv.org/pdf/2503.12528)  

**Abstract**: Recent work has sought to quantify large language model uncertainty to facilitate model control and modulate user trust. Previous works focus on measures of uncertainty that are theoretically grounded or reflect the average overt behavior of the model. In this work, we investigate a variety of uncertainty measures, in order to identify measures that correlate with human group-level uncertainty. We find that Bayesian measures and a variation on entropy measures, top-k entropy, tend to agree with human behavior as a function of model size. We find that some strong measures decrease in human-similarity with model size, but, by multiple linear regression, we find that combining multiple uncertainty measures provide comparable human-alignment with reduced size-dependency. 

---
# EXAONE Deep: Reasoning Enhanced Language Models 

**Authors**: LG AI Research, Kyunghoon Bae, Eunbi Choi, Kibong Choi, Stanley Jungkyu Choi, Yemuk Choi, Seokhee Hong, Junwon Hwang, Hyojin Jeon, Kijeong Jeon, Gerrard Jeongwon Jo, Hyunjik Jo, Jiyeon Jung, Hyosang Kim, Joonkee Kim, Seonghwan Kim, Soyeon Kim, Sunkyoung Kim, Yireun Kim, Yongil Kim, Youchul Kim, Edward Hwayoung Lee, Haeju Lee, Honglak Lee, Jinsik Lee, Kyungmin Lee, Sangha Park, Yongmin Park, Sihoon Yang, Heuiyeen Yeen, Sihyuk Yi, Hyeongu Yun  

**Link**: [PDF](https://arxiv.org/pdf/2503.12524)  

**Abstract**: We present EXAONE Deep series, which exhibits superior capabilities in various reasoning tasks, including math and coding benchmarks. We train our models mainly on the reasoning-specialized dataset that incorporates long streams of thought processes. Evaluation results show that our smaller models, EXAONE Deep 2.4B and 7.8B, outperform other models of comparable size, while the largest model, EXAONE Deep 32B, demonstrates competitive performance against leading open-weight models. All EXAONE Deep models are openly available for research purposes and can be downloaded from this https URL 

---
# CAKE: Cascading and Adaptive KV Cache Eviction with Layer Preferences 

**Authors**: Ziran Qin, Yuchen Cao, Mingbao Lin, Wen Hu, Shixuan Fan, Ke Cheng, Weiyao Lin, Jianguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.12491)  

**Abstract**: Large language models (LLMs) excel at processing long sequences, boosting demand for key-value (KV) caching. While recent efforts to evict KV cache have alleviated the inference burden, they often fail to allocate resources rationally across layers with different attention patterns. In this paper, we introduce Cascading and Adaptive KV cache Eviction (CAKE), a novel approach that frames KV cache eviction as a "cake-slicing problem." CAKE assesses layer-specific preferences by considering attention dynamics in both spatial and temporal dimensions, allocates rational cache size for layers accordingly, and manages memory constraints in a cascading manner. This approach enables a global view of cache allocation, adaptively distributing resources across diverse attention mechanisms while maintaining memory budgets. CAKE also employs a new eviction indicator that considers the shifting importance of tokens over time, addressing limitations in existing methods that overlook temporal dynamics. Comprehensive experiments on LongBench and NeedleBench show that CAKE maintains model performance with only 3.2% of the KV cache and consistently outperforms current baselines across various models and memory constraints, particularly in low-memory settings. Additionally, CAKE achieves over 10x speedup in decoding latency compared to full cache when processing contexts of 128K tokens with FlashAttention-2. Our code is available at this https URL. 

---
# HKCanto-Eval: A Benchmark for Evaluating Cantonese Language Understanding and Cultural Comprehension in LLMs 

**Authors**: Tsz Chung Cheng, Chung Shing Cheng, Chaak Ming Lau, Eugene Tin-Ho Lam, Chun Yat Wong, Hoi On Yu, Cheuk Hei Chong  

**Link**: [PDF](https://arxiv.org/pdf/2503.12440)  

**Abstract**: The ability of language models to comprehend and interact in diverse linguistic and cultural landscapes is crucial. The Cantonese language used in Hong Kong presents unique challenges for natural language processing due to its rich cultural nuances and lack of dedicated evaluation datasets. The HKCanto-Eval benchmark addresses this gap by evaluating the performance of large language models (LLMs) on Cantonese language understanding tasks, extending to English and Written Chinese for cross-lingual evaluation. HKCanto-Eval integrates cultural and linguistic nuances intrinsic to Hong Kong, providing a robust framework for assessing language models in realistic scenarios. Additionally, the benchmark includes questions designed to tap into the underlying linguistic metaknowledge of the models. Our findings indicate that while proprietary models generally outperform open-weight models, significant limitations remain in handling Cantonese-specific linguistic and cultural knowledge, highlighting the need for more targeted training data and evaluation methods. The code can be accessed at this https URL 

---
# Understanding Common Ground Misalignment in Goal-Oriented Dialog: A Case-Study with Ubuntu Chat Logs 

**Authors**: Rupak Sarkar, Neha Srikanth, Taylor Hudson, Rachel Rudinger, Claire Bonial, Philip Resnik  

**Link**: [PDF](https://arxiv.org/pdf/2503.12370)  

**Abstract**: While it is commonly accepted that maintaining common ground plays a role in conversational success, little prior research exists connecting conversational grounding to success in task-oriented conversations. We study failures of grounding in the Ubuntu IRC dataset, where participants use text-only communication to resolve technical issues. We find that disruptions in conversational flow often stem from a misalignment in common ground, driven by a divergence in beliefs and assumptions held by participants. These disruptions, which we call conversational friction, significantly correlate with task success. We find that although LLMs can identify overt cases of conversational friction, they struggle with subtler and more context-dependent instances requiring pragmatic or domain-specific reasoning. 

---
# Numerical Words and Linguistic Loops: The Perpetual Four-Letter Routine 

**Authors**: Krishna Chaitanya Polavaram  

**Link**: [PDF](https://arxiv.org/pdf/2503.12357)  

**Abstract**: This study presents a fascinating linguistic property related to the number of letters in words and their corresponding numerical values. By selecting any arbitrary word, counting its constituent letters, and subsequently spelling out the resulting count and tallying the letters anew, an unanticipated pattern is observed. Remarkably, this iterative sequence, conducted on a dataset of 100,000 random words, invariably converges to the numeral four (4), termed the Linguistic Loop (LL) constant. Examining 73 languages utilizing the Latin alphabet, this research reveals distinctive patterns. Among them, 28 languages exhibit LL-positive behavior adhering to the established property, while 31 languages deviate as LL-negative. Additionally, 13 languages display nuanced tendencies: eight feature two LL constants (bi-positivity), and five feature three constants (tri-positivity). This discovery highlights a linguistic quirk within Latin alphabet-based language number-word representations, uncovering an intriguing facet across diverse alphabetic systems. It also raises questions about the underlying linguistic and cognitive mechanisms responsible for this phenomenon. 

---
# Synthesizing Privacy-Preserving Text Data via Finetuning without Finetuning Billion-Scale LLMs 

**Authors**: Bowen Tan, Zheng Xu, Eric Xing, Zhiting Hu, Shanshan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12347)  

**Abstract**: Synthetic data offers a promising path to train models while preserving data privacy. Differentially private (DP) finetuning of large language models (LLMs) as data generator is effective, but is impractical when computation resources are limited. Meanwhile, prompt-based methods such as private evolution, depend heavily on the manual prompts, and ineffectively use private information in their iterative data selection process. To overcome these limitations, we propose CTCL (Data Synthesis with ConTrollability and CLustering), a novel framework for generating privacy-preserving synthetic data without extensive prompt engineering or billion-scale LLM finetuning. CTCL pretrains a lightweight 140M conditional generator and a clustering-based topic model on large-scale public data. To further adapt to the private domain, the generator is DP finetuned on private data for fine-grained textual information, while the topic model extracts a DP histogram representing distributional information. The DP generator then samples according to the DP histogram to synthesize a desired number of data examples. Evaluation across five diverse domains demonstrates the effectiveness of our framework, particularly in the strong privacy regime. Systematic ablation validates the design of each framework component and highlights the scalability of our approach. 

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
# The Lucie-7B LLM and the Lucie Training Dataset: Open resources for multilingual language generation 

**Authors**: Olivier Gouvert, Julie Hunter, Jérôme Louradour, Christophe Cerisara, Evan Dufraisse, Yaya Sy, Laura Rivière, Jean-Pierre Lorré, OpenLLM-France community  

**Link**: [PDF](https://arxiv.org/pdf/2503.12294)  

**Abstract**: We present both the Lucie Training Dataset and the Lucie-7B foundation model. The Lucie Training Dataset is a multilingual collection of textual corpora centered around French and designed to offset anglo-centric biases found in many datasets for large language model pretraining. Its French data is pulled not only from traditional web sources, but also from French cultural heritage documents, filling an important gap in modern datasets. Beyond French, which makes up the largest share of the data, we added documents to support several other European languages, including English, Spanish, German, and Italian. Apart from its value as a resource for French language and culture, an important feature of this dataset is that it prioritizes data rights by minimizing copyrighted material. In addition, building on the philosophy of past open projects, it is redistributed in the form used for training and its processing is described on Hugging Face and GitHub. The Lucie-7B foundation model is trained on equal amounts of data in French and English -- roughly 33% each -- in an effort to better represent cultural aspects of French-speaking communities. We also describe two instruction fine-tuned models, Lucie-7B-Instruct-v1.1 and Lucie-7B-Instruct-human-data, which we release as demonstrations of Lucie-7B in use. These models achieve promising results compared to state-of-the-art models, demonstrating that an open approach prioritizing data rights can still deliver strong performance. We see these models as an initial step toward developing more performant, aligned models in the near future. Model weights for Lucie-7B and the Lucie instruct models, along with intermediate checkpoints for the former, are published on Hugging Face, while model training and data preparation code is available on GitHub. This makes Lucie-7B one of the first OSI compliant language models according to the new OSI definition. 

---
# Integrating Chain-of-Thought and Retrieval Augmented Generation Enhances Rare Disease Diagnosis from Clinical Notes 

**Authors**: Da Wu, Zhanliang Wang, Quan Nguyen, Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12286)  

**Abstract**: Background: Several studies show that large language models (LLMs) struggle with phenotype-driven gene prioritization for rare diseases. These studies typically use Human Phenotype Ontology (HPO) terms to prompt foundation models like GPT and LLaMA to predict candidate genes. However, in real-world settings, foundation models are not optimized for domain-specific tasks like clinical diagnosis, yet inputs are unstructured clinical notes rather than standardized terms. How LLMs can be instructed to predict candidate genes or disease diagnosis from unstructured clinical notes remains a major challenge. Methods: We introduce RAG-driven CoT and CoT-driven RAG, two methods that combine Chain-of-Thought (CoT) and Retrieval Augmented Generation (RAG) to analyze clinical notes. A five-question CoT protocol mimics expert reasoning, while RAG retrieves data from sources like HPO and OMIM (Online Mendelian Inheritance in Man). We evaluated these approaches on rare disease datasets, including 5,980 Phenopacket-derived notes, 255 literature-based narratives, and 220 in-house clinical notes from Childrens Hospital of Philadelphia. Results: We found that recent foundations models, including Llama 3.3-70B-Instruct and DeepSeek-R1-Distill-Llama-70B, outperformed earlier versions such as Llama 2 and GPT-3.5. We also showed that RAG-driven CoT and CoT-driven RAG both outperform foundation models in candidate gene prioritization from clinical notes; in particular, both methods with DeepSeek backbone resulted in a top-10 gene accuracy of over 40% on Phenopacket-derived clinical notes. RAG-driven CoT works better for high-quality notes, where early retrieval can anchor the subsequent reasoning steps in domain-specific evidence, while CoT-driven RAG has advantage when processing lengthy and noisy notes. 

---
# Interpretation Gaps in LLM-Assisted Comprehension of Privacy Documents 

**Authors**: Rinku Dewri  

**Link**: [PDF](https://arxiv.org/pdf/2503.12225)  

**Abstract**: This article explores the gaps that can manifest when using a large language model (LLM) to obtain simplified interpretations of data practices from a complex privacy policy. We exemplify these gaps to showcase issues in accuracy, completeness, clarity and representation, while advocating for continued research to realize an LLM's true potential in revolutionizing privacy management through personal assistants and automated compliance checking. 

---
# PLM: Efficient Peripheral Language Models Hardware-Co-Designed for Ubiquitous Computing 

**Authors**: Cheng Deng, Luoyang Sun, Jiwen Jiang, Yongcheng Zeng, Xinjian Wu, Wenxin Zhao, Qingfa Xiao, Jiachuan Wang, Lei Chen, Lionel M. Ni, Haifeng Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12167)  

**Abstract**: While scaling laws have been continuously validated in large language models (LLMs) with increasing model parameters, the inherent tension between the inference demands of LLMs and the limited resources of edge devices poses a critical challenge to the development of edge intelligence. Recently, numerous small language models have emerged, aiming to distill the capabilities of LLMs into smaller footprints. However, these models often retain the fundamental architectural principles of their larger counterparts, still imposing considerable strain on the storage and bandwidth capacities of edge devices. In this paper, we introduce the PLM, a Peripheral Language Model, developed through a co-design process that jointly optimizes model architecture and edge system constraints. The PLM utilizes a Multi-head Latent Attention mechanism and employs the squared ReLU activation function to encourage sparsity, thereby reducing peak memory footprint during inference. During training, we collect and reorganize open-source datasets, implement a multi-phase training strategy, and empirically investigate the Warmup-Stable-Decay-Constant (WSDC) learning rate scheduler. Additionally, we incorporate Reinforcement Learning from Human Feedback (RLHF) by adopting the ARIES preference learning approach. Following a two-phase SFT process, this method yields performance gains of 2% in general tasks, 9% in the GSM8K task, and 11% in coding tasks. In addition to its novel architecture, evaluation results demonstrate that PLM outperforms existing small language models trained on publicly available data while maintaining the lowest number of activated parameters. Furthermore, deployment across various edge devices, including consumer-grade GPUs, mobile phones, and Raspberry Pis, validates PLM's suitability for peripheral applications. The PLM series models are publicly available at this https URL. 

---
# Improving LLM-based Document-level Machine Translation with Multi-Knowledge Fusion 

**Authors**: Bin Liu, Xinglin Lyu, Junhui Li, Daimeng Wei, Min Zhang, Shimin Tao, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12152)  

**Abstract**: Recent studies in prompting large language model (LLM) for document-level machine translation (DMT) primarily focus on the inter-sentence context by flatting the source document into a long sequence. This approach relies solely on the sequence of sentences within the document. However, the complexity of document-level sequences is greater than that of shorter sentence-level sequences, which may limit LLM's ability in DMT when only this single-source knowledge is used. In this paper, we propose an enhanced approach by incorporating multiple sources of knowledge, including both the document summarization and entity translation, to enhance the performance of LLM-based DMT. Given a source document, we first obtain its summarization and translation of entities via LLM as the additional knowledge. We then utilize LLMs to generate two translations of the source document by fusing these two single knowledge sources, respectively. Finally, recognizing that different sources of knowledge may aid or hinder the translation of different sentences, we refine and rank the translations by leveraging a multi-knowledge fusion strategy to ensure the best results. Experimental results in eight document-level translation tasks show that our approach achieves an average improvement of 0.8, 0.6, and 0.4 COMET scores over the baseline without extra knowledge for LLaMA3-8B-Instruct, Mistral-Nemo-Instruct, and GPT-4o-mini, respectively. 

---
# Seeing Sarcasm Through Different Eyes: Analyzing Multimodal Sarcasm Perception in Large Vision-Language Models 

**Authors**: Junjie Chen, Xuyang Liu, Subin Huang, Linfeng Zhang, Hang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12149)  

**Abstract**: With the advent of large vision-language models (LVLMs) demonstrating increasingly human-like abilities, a pivotal question emerges: do different LVLMs interpret multimodal sarcasm differently, and can a single model grasp sarcasm from multiple perspectives like humans? To explore this, we introduce an analytical framework using systematically designed prompts on existing multimodal sarcasm datasets. Evaluating 12 state-of-the-art LVLMs over 2,409 samples, we examine interpretive variations within and across models, focusing on confidence levels, alignment with dataset labels, and recognition of ambiguous "neutral" cases. Our findings reveal notable discrepancies -- across LVLMs and within the same model under varied prompts. While classification-oriented prompts yield higher internal consistency, models diverge markedly when tasked with interpretive reasoning. These results challenge binary labeling paradigms by highlighting sarcasm's subjectivity. We advocate moving beyond rigid annotation schemes toward multi-perspective, uncertainty-aware modeling, offering deeper insights into multimodal sarcasm comprehension. Our code and data are available at: this https URL 

---
# Enhanced Sentiment Analysis of Iranian Restaurant Reviews Utilizing Sentiment Intensity Analyzer & Fuzzy Logic 

**Authors**: Shayan Rokhva, Babak Teimourpour, Romina Babaei  

**Link**: [PDF](https://arxiv.org/pdf/2503.12141)  

**Abstract**: This research presents an advanced sentiment analysis framework studied on Iranian restaurant reviews, combining fuzzy logic with conventional sentiment analysis techniques to assess both sentiment polarity and intensity. A dataset of 1266 reviews, alongside corresponding star ratings, was compiled and preprocessed for analysis. Initial sentiment analysis was conducted using the Sentiment Intensity Analyzer (VADER), a rule-based tool that assigns sentiment scores across positive, negative, and neutral categories. However, a noticeable bias toward neutrality often led to an inaccurate representation of sentiment intensity. To mitigate this issue, based on a fuzzy perspective, two refinement techniques were introduced, applying square-root and fourth-root transformations to amplify positive and negative sentiment scores while maintaining neutrality. This led to three distinct methodologies: Approach 1, utilizing unaltered VADER scores; Approach 2, modifying sentiment values using the square root; and Approach 3, applying the fourth root for further refinement. A Fuzzy Inference System incorporating comprehensive fuzzy rules was then developed to process these refined scores and generate a single, continuous sentiment value for each review based on each approach. Comparative analysis, including human supervision and alignment with customer star ratings, revealed that the refined approaches significantly improved sentiment analysis by reducing neutrality bias and better capturing sentiment intensity. Despite these advancements, minor over-amplification and persistent neutrality in domain-specific cases were identified, leading us to propose several future studies to tackle these occasional barriers. The study's methodology and outcomes offer valuable insights for businesses seeking a more precise understanding of consumer sentiment, enhancing sentiment analysis across various industries. 

---
# MT-RewardTree: A Comprehensive Framework for Advancing LLM-Based Machine Translation via Reward Modeling 

**Authors**: Zhaopeng Feng, Jiahan Ren, Jiayuan Su, Jiamei Zheng, Zhihang Tang, Hongwei Wang, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12123)  

**Abstract**: Process reward models (PRMs) have shown success in complex reasoning tasks for large language models (LLMs). However, their application to machine translation (MT) remains underexplored due to the lack of systematic methodologies and evaluation benchmarks. To address this gap, we introduce \textbf{MT-RewardTree}, a comprehensive framework for constructing, evaluating, and deploying process reward models in MT. Unlike traditional vanilla preference pair construction, we propose a novel method for automatically generating token-level preference pairs using approximate Monte Carlo Tree Search (MCTS), which mitigates the prohibitive cost of human annotation for fine-grained steps. Then, we establish the first MT-specific reward model benchmark and provide a systematic comparison of different reward modeling architectures, revealing that token-level supervision effectively captures fine-grained preferences. Experimental results demonstrate that our MT-PRM-Qwen-2.5-3B achieves state-of-the-art performance in both token-level and sequence-level evaluation given the same input prefix. Furthermore, we showcase practical applications where PRMs enable test-time alignment for LLMs without additional alignment training and significantly improve performance in hypothesis ensembling. Our work provides valuable insights into the role of reward models in MT research. Our code and data are released in \href{this https URL}{this https URL\_RewardTreePage}. 

---
# RECSIP: REpeated Clustering of Scores Improving the Precision 

**Authors**: André Schamschurko, Nenad Petrovic, Alois Christian Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.12108)  

**Abstract**: The latest research on Large Language Models (LLMs) has demonstrated significant advancement in the field of Natural Language Processing (NLP). However, despite this progress, there is still a lack of reliability in these models. This is due to the stochastic architecture of LLMs, which presents a challenge for users attempting to ascertain the reliability of a model's response. These responses may cause serious harm in high-risk environments or expensive failures in industrial contexts. Therefore, we introduce the framework REpeated Clustering of Scores Improving the Precision (RECSIP) which focuses on improving the precision of LLMs by asking multiple models in parallel, scoring and clustering their responses to ensure a higher reliability on the response. The evaluation of our reference implementation recsip on the benchmark MMLU-Pro using the models GPT-4o, Claude and Gemini shows an overall increase of 5.8 per cent points compared to the best used model. 

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
# TLUE: A Tibetan Language Understanding Evaluation Benchmark 

**Authors**: Fan Gao, Cheng Huang, Nyima Tashi, Xiangxiang Wang, Thupten Tsering, Ban Ma-bao, Renzeg Duojie, Gadeng Luosang, Rinchen Dongrub, Dorje Tashi, Xiao Feng, Yongbin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12051)  

**Abstract**: Large language models (LLMs) have made tremendous progress in recent years, but low-resource languages, such as Tibetan, remain significantly underrepresented in their evaluation. Despite Tibetan being spoken by over seven million people, it has largely been neglected in the development and assessment of LLMs. To address this gap, we present TLUE (A Tibetan Language Understanding Evaluation Benchmark), the first large-scale benchmark for assessing LLMs' capabilities in Tibetan. TLUE comprises two major components: (1) a comprehensive multi-task understanding benchmark spanning 5 domains and 67 subdomains, and (2) a safety benchmark covering 7 subdomains. We evaluate a diverse set of state-of-the-art LLMs. Experimental results demonstrate that most LLMs perform below the random baseline, highlighting the considerable challenges LLMs face in processing Tibetan, a low-resource language. TLUE provides an essential foundation for driving future research and progress in Tibetan language understanding and underscores the need for greater inclusivity in LLM development. 

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
# HInter: Exposing Hidden Intersectional Bias in Large Language Models 

**Authors**: Badr Souani, Ezekiel Soremekun, Mike Papadakis, Setsuko Yokoyama, Sudipta Chattopadhyay, Yves Le Traon  

**Link**: [PDF](https://arxiv.org/pdf/2503.11962)  

**Abstract**: Large Language Models (LLMs) may portray discrimination towards certain individuals, especially those characterized by multiple attributes (aka intersectional bias). Discovering intersectional bias in LLMs is challenging, as it involves complex inputs on multiple attributes (e.g. race and gender). To address this challenge, we propose HInter, a test technique that synergistically combines mutation analysis, dependency parsing and metamorphic oracles to automatically detect intersectional bias in LLMs. HInter generates test inputs by systematically mutating sentences using multiple mutations, validates inputs via a dependency invariant and detects biases by checking the LLM response on the original and mutated sentences. We evaluate HInter using six LLM architectures and 18 LLM models (GPT3.5, Llama2, BERT, etc) and find that 14.61% of the inputs generated by HInter expose intersectional bias. Results also show that our dependency invariant reduces false positives (incorrect test inputs) by an order of magnitude. Finally, we observed that 16.62% of intersectional bias errors are hidden, meaning that their corresponding atomic cases do not trigger biases. Overall, this work emphasize the importance of testing LLMs for intersectional bias. 

---
# Integration of Explainable AI Techniques with Large Language Models for Enhanced Interpretability for Sentiment Analysis 

**Authors**: Thivya Thogesan, Anupiya Nugaliyadde, Kok Wai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2503.11948)  

**Abstract**: Interpretability remains a key difficulty in sentiment analysis with Large Language Models (LLMs), particularly in high-stakes applications where it is crucial to comprehend the rationale behind forecasts. This research addressed this by introducing a technique that applies SHAP (Shapley Additive Explanations) by breaking down LLMs into components such as embedding layer,encoder,decoder and attention layer to provide a layer-by-layer knowledge of sentiment prediction. The approach offers a clearer overview of how model interpret and categorise sentiment by breaking down LLMs into these parts. The method is evaluated using the Stanford Sentiment Treebank (SST-2) dataset, which shows how different sentences affect different layers. The effectiveness of layer-wise SHAP analysis in clarifying sentiment-specific token attributions is demonstrated by experimental evaluations, which provide a notable enhancement over current whole-model explainability techniques. These results highlight how the suggested approach could improve the reliability and transparency of LLM-based sentiment analysis in crucial applications. 

---
# REGEN: A Dataset and Benchmarks with Natural Language Critiques and Narratives 

**Authors**: Kun Su, Krishna Sayana, Hubert Pham, James Pine, Yuri Vasilevski, Raghavendra Vasudeva, Marialena Kyriakidi, Liam Hebert, Ambarish Jash, Anushya Subbiah, Sukhdeep Sodhi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11924)  

**Abstract**: This paper introduces a novel dataset REGEN (Reviews Enhanced with GEnerative Narratives), designed to benchmark the conversational capabilities of recommender Large Language Models (LLMs), addressing the limitations of existing datasets that primarily focus on sequential item prediction. REGEN extends the Amazon Product Reviews dataset by inpainting two key natural language features: (1) user critiques, representing user "steering" queries that lead to the selection of a subsequent item, and (2) narratives, rich textual outputs associated with each recommended item taking into account prior context. The narratives include product endorsements, purchase explanations, and summaries of user preferences.
Further, we establish an end-to-end modeling benchmark for the task of conversational recommendation, where models are trained to generate both recommendations and corresponding narratives conditioned on user history (items and critiques). For this joint task, we introduce a modeling framework LUMEN (LLM-based Unified Multi-task Model with Critiques, Recommendations, and Narratives) which uses an LLM as a backbone for critiquing, retrieval and generation. We also evaluate the dataset's quality using standard auto-rating techniques and benchmark it by training both traditional and LLM-based recommender models. Our results demonstrate that incorporating critiques enhances recommendation quality by enabling the recommender to learn language understanding and integrate it with recommendation signals. Furthermore, LLMs trained on our dataset effectively generate both recommendations and contextual narratives, achieving performance comparable to state-of-the-art recommenders and language models. 

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
# A Transformer and Prototype-based Interpretable Model for Contextual Sarcasm Detection 

**Authors**: Ximing Wen, Rezvaneh Rezapour  

**Link**: [PDF](https://arxiv.org/pdf/2503.11838)  

**Abstract**: Sarcasm detection, with its figurative nature, poses unique challenges for affective systems designed to perform sentiment analysis. While these systems typically perform well at identifying direct expressions of emotion, they struggle with sarcasm's inherent contradiction between literal and intended sentiment. Since transformer-based language models (LMs) are known for their efficient ability to capture contextual meanings, we propose a method that leverages LMs and prototype-based networks, enhanced by sentiment embeddings to conduct interpretable sarcasm detection. Our approach is intrinsically interpretable without extra post-hoc interpretability techniques. We test our model on three public benchmark datasets and show that our model outperforms the current state-of-the-art. At the same time, the prototypical layer enhances the model's inherent interpretability by generating explanations through similar examples in the reference time. Furthermore, we demonstrate the effectiveness of incongruity loss in the ablation study, which we construct using sentiment prototypes. 

---
# Transfer Learning for Automated Feedback Generation on Small Datasets 

**Authors**: Oscar Morris  

**Link**: [PDF](https://arxiv.org/pdf/2503.11836)  

**Abstract**: Feedback is a very important part the learning process. However, it is challenging to make this feedback both timely and accurate when relying on human markers. This is the challenge that Automated Feedback Generation attempts to address. In this paper, a technique to train such a system on a very small dataset with very long sequences is presented. Both of these attributes make this a very challenging task, however, by using a three stage transfer learning pipeline state-of-the-art results can be achieved with qualitatively accurate but unhuman sounding results. The use of both Automated Essay Scoring and Automated Feedback Generation systems in the real world is also discussed. 

---
# Bridging the LLM Accessibility Divide? Performance, Fairness, and Cost of Closed versus Open LLMs for Automated Essay Scoring 

**Authors**: Kezia Oketch, John P. Lalor, Yi Yang, Ahmed Abbasi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11827)  

**Abstract**: Closed large language models (LLMs) such as GPT-4 have set state-of-the-art results across a number of NLP tasks and have become central to NLP and machine learning (ML)-driven solutions. Closed LLMs' performance and wide adoption has sparked considerable debate about their accessibility in terms of availability, cost, and transparency. In this study, we perform a rigorous comparative analysis of nine leading LLMs, spanning closed, open, and open-source LLM ecosystems, across text assessment and generation tasks related to automated essay scoring. Our findings reveal that for few-shot learning-based assessment of human generated essays, open LLMs such as Llama 3 and Qwen2.5 perform comparably to GPT-4 in terms of predictive performance, with no significant differences in disparate impact scores when considering age- or race-related fairness. Moreover, Llama 3 offers a substantial cost advantage, being up to 37 times more cost-efficient than GPT-4. For generative tasks, we find that essays generated by top open LLMs are comparable to closed LLMs in terms of their semantic composition/embeddings and ML assessed scores. Our findings challenge the dominance of closed LLMs and highlight the democratizing potential of open LLMs, suggesting they can effectively bridge accessibility divides while maintaining competitive performance and fairness. 

---
# Key, Value, Compress: A Systematic Exploration of KV Cache Compression Techniques 

**Authors**: Neusha Javidnia, Bita Darvish Rouhani, Farinaz Koushanfar  

**Link**: [PDF](https://arxiv.org/pdf/2503.11816)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional capabilities in generating text, images, and video content. However, as context length grows, the computational cost of attention increases quadratically with the number of tokens, presenting significant efficiency challenges. This paper presents an analysis of various Key-Value (KV) cache compression strategies, offering a comprehensive taxonomy that categorizes these methods by their underlying principles and implementation techniques. Furthermore, we evaluate their impact on performance and inference latency, providing critical insights into their effectiveness. Our findings highlight the trade-offs involved in KV cache compression and its influence on handling long-context scenarios, paving the way for more efficient LLM implementations. 

---
# reWordBench: Benchmarking and Improving the Robustness of Reward Models with Transformed Inputs 

**Authors**: Zhaofeng Wu, Michihiro Yasunaga, Andrew Cohen, Yoon Kim, Asli Celikyilmaz, Marjan Ghazvininejad  

**Link**: [PDF](https://arxiv.org/pdf/2503.11751)  

**Abstract**: Reward models have become a staple in modern NLP, serving as not only a scalable text evaluator, but also an indispensable component in many alignment recipes and inference-time algorithms. However, while recent reward models increase performance on standard benchmarks, this may partly be due to overfitting effects, which would confound an understanding of their true capability. In this work, we scrutinize the robustness of reward models and the extent of such overfitting. We build **reWordBench**, which systematically transforms reward model inputs in meaning- or ranking-preserving ways. We show that state-of-the-art reward models suffer from substantial performance degradation even with minor input transformations, sometimes dropping to significantly below-random accuracy, suggesting brittleness. To improve reward model robustness, we propose to explicitly train them to assign similar scores to paraphrases, and find that this approach also improves robustness to other distinct kinds of transformations. For example, our robust reward model reduces such degradation by roughly half for the Chat Hard subset in RewardBench. Furthermore, when used in alignment, our robust reward models demonstrate better utility and lead to higher-quality outputs, winning in up to 59% of instances against a standardly trained RM. 

---
# LogitLens4LLMs: Extending Logit Lens Analysis to Modern Large Language Models 

**Authors**: Zhenyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11667)  

**Abstract**: This paper introduces LogitLens4LLMs, a toolkit that extends the Logit Lens technique to modern large language models. While Logit Lens has been a crucial method for understanding internal representations of language models, it was previously limited to earlier model architectures. Our work overcomes the limitations of existing implementations, enabling the technique to be applied to state-of-the-art architectures (such as Qwen-2.5 and Llama-3.1) while automating key analytical workflows. By developing component-specific hooks to capture both attention mechanisms and MLP outputs, our implementation achieves full compatibility with the HuggingFace transformer library while maintaining low inference overhead. The toolkit provides both interactive exploration and batch processing capabilities, supporting large-scale layer-wise analyses. Through open-sourcing our implementation, we aim to facilitate deeper investigations into the internal mechanisms of large-scale language models. The toolkit is openly available at this https URL. 

---
# Automating Mathematical Proof Generation Using Large Language Model Agents and Knowledge Graphs 

**Authors**: Vincent Li, Yule Fu, Tim Knappe, Kevin Han, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11657)  

**Abstract**: Large Language Models have demonstrated remarkable capabilities in natural language processing tasks, including mathematical problem-solving that requires multi-step logical reasoning. However, challenges persist in automating the identification of key mathematical concepts, understanding their interrelations, and formalizing proofs within a rigorous framework. We present a novel framework that leverages knowledge graphs to augment LLMs to construct and formalize mathematical proofs. Our results demonstrate significant performance improvements across multiple datasets, with using knowledge graphs, achieving up to a 34% success rate on the MUSTARDSAUCE dataset on o1-mini and consistently outperforming baseline approaches by 2-11% across different models. We show how this approach bridges the gap between natural language understanding and formal logic proof systems and achieve elevated results for foundation models over baseline. 

---
# TRUTH DECAY: Quantifying Multi-Turn Sycophancy in Language Models 

**Authors**: Joshua Liu, Aarav Jain, Soham Takuri, Srihan Vege, Aslihan Akalin, Kevin Zhu, Sean O'Brien, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2503.11656)  

**Abstract**: Rapid improvements in large language models have unveiled a critical challenge in human-AI interaction: sycophancy. In this context, sycophancy refers to the tendency of models to excessively agree with or flatter users, often at the expense of factual accuracy. While previous studies have primarily analyzed this behavior in single-turn interactions, its persistence and evolution in multi-step conversations remain largely unexplored. We introduce TRUTH DECAY, a benchmark specifically designed to evaluate sycophancy in extended dialogues, where language models must navigate iterative user feedback, challenges, and persuasion. We prompt models to elicit four types of sycophantic biases. We then propose and test sycophancy reduction strategies, evaluating their effectiveness beyond single-step interactions. 

---
# Explainable Sentiment Analysis with DeepSeek-R1: Performance, Efficiency, and Few-Shot Learning 

**Authors**: Donghao Huang, Zhaoxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11655)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly enhanced sentiment analysis capabilities. However, the trade-offs between model performance, efficiency, and explainability of some latest models remain underexplored. This study presents the first comprehensive evaluation of the DeepSeek-R1 series of models, reasoning open-source LLMs, for sentiment analysis, comparing them against OpenAI's GPT-4 and GPT-4-mini. We systematically analyze their performance under few-shot prompting conditions, scaling up to 50-shot configurations to assess in-context learning effectiveness. Our experiments reveal that DeepSeek-R1 demonstrates competitive accuracy, particularly in multi-class sentiment tasks, while offering enhanced interpretability through its detailed reasoning process. Additionally, we highlight the impact of increasing few-shot examples on model performance and discuss key trade-offs between explainability and computational efficiency. 

---
# MicroVQA: A Multimodal Reasoning Benchmark for Microscopy-Based Scientific Research 

**Authors**: James Burgess, Jeffrey J Nirschl, Laura Bravo-Sánchez, Alejandro Lozano, Sanket Rajan Gupte, Jesus G. Galaz-Montoya, Yuhui Zhang, Yuchang Su, Disha Bhowmik, Zachary Coman, Sarina M. Hasan, Alexandra Johannesson, William D. Leineweber, Malvika G Nair, Ridhi Yarlagadda, Connor Zuraski, Wah Chiu, Sarah Cohen, Jan N. Hansen, Manuel D Leonetti, Chad Liu, Emma Lundberg, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2503.13399)  

**Abstract**: Scientific research demands sophisticated reasoning over multimodal data, a challenge especially prevalent in biology. Despite recent advances in multimodal large language models (MLLMs) for AI-assisted research, existing multimodal reasoning benchmarks only target up to college-level difficulty, while research-level benchmarks emphasize lower-level perception, falling short of the complex multimodal reasoning needed for scientific discovery. To bridge this gap, we introduce MicroVQA, a visual-question answering (VQA) benchmark designed to assess three reasoning capabilities vital in research workflows: expert image understanding, hypothesis generation, and experiment proposal. MicroVQA consists of 1,042 multiple-choice questions (MCQs) curated by biology experts across diverse microscopy modalities, ensuring VQA samples represent real scientific practice. In constructing the benchmark, we find that standard MCQ generation methods induce language shortcuts, motivating a new two-stage pipeline: an optimized LLM prompt structures question-answer pairs into MCQs; then, an agent-based `RefineBot' updates them to remove shortcuts. Benchmarking on state-of-the-art MLLMs reveal a peak performance of 53\%; models with smaller LLMs only slightly underperform top models, suggesting that language-based reasoning is less challenging than multimodal reasoning; and tuning with scientific articles enhances performance. Expert analysis of chain-of-thought responses shows that perception errors are the most frequent, followed by knowledge errors and then overgeneralization errors. These insights highlight the challenges in multimodal scientific reasoning, showing MicroVQA is a valuable resource advancing AI-driven biomedical research. MicroVQA is available at this https URL, and project page at this https URL. 

---
# TimeZero: Temporal Video Grounding with Reasoning-Guided LVLM 

**Authors**: Ye Wang, Boshen Xu, Zihao Yue, Zihan Xiao, Ziheng Wang, Liang Zhang, Dingyi Yang, Wenxuan Wang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2503.13377)  

**Abstract**: We introduce TimeZero, a reasoning-guided LVLM designed for the temporal video grounding (TVG) task. This task requires precisely localizing relevant video segments within long videos based on a given language query. TimeZero tackles this challenge by extending the inference process, enabling the model to reason about video-language relationships solely through reinforcement learning. To evaluate the effectiveness of TimeZero, we conduct experiments on two benchmarks, where TimeZero achieves state-of-the-art performance on Charades-STA. Code is available at this https URL. 

---
# $ϕ$-Decoding: Adaptive Foresight Sampling for Balanced Inference-Time Exploration and Exploitation 

**Authors**: Fangzhi Xu, Hang Yan, Chang Ma, Haiteng Zhao, Jun Liu, Qika Lin, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.13288)  

**Abstract**: Inference-time optimization scales computation to derive deliberate reasoning steps for effective performance. While previous search-based strategies address the short-sightedness of auto-regressive generation, the vast search space leads to excessive exploration and insufficient exploitation. To strike an efficient balance to derive the optimal step, we frame the decoding strategy as foresight sampling, leveraging simulated future steps to obtain globally optimal step estimation. Built on it, we propose a novel decoding strategy, named $\phi$-Decoding. To provide a precise and expressive estimation of step value, $\phi$-Decoding approximates two distributions via foresight and clustering. Sampling from the joint distribution, the optimal steps can be selected for exploitation. To support adaptive computation allocation, we propose in-width and in-depth pruning strategies, featuring a light-weight solution to achieve inference efficiency. Extensive experiments across seven benchmarks show $\phi$-Decoding outperforms strong baselines in both performance and efficiency. Additional analysis demonstrates its generalization across various LLMs and scalability across a wide range of computing budgets. The code will be released at this https URL, and the open-source PyPI package is coming soon. 

---
# MAP: Evaluation and Multi-Agent Enhancement of Large Language Models for Inpatient Pathways 

**Authors**: Zhen Chen, Zhihao Peng, Xusheng Liang, Cheng Wang, Peigan Liang, Linsheng Zeng, Minjie Ju, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.13205)  

**Abstract**: Inpatient pathways demand complex clinical decision-making based on comprehensive patient information, posing critical challenges for clinicians. Despite advancements in large language models (LLMs) in medical applications, limited research focused on artificial intelligence (AI) inpatient pathways systems, due to the lack of large-scale inpatient datasets. Moreover, existing medical benchmarks typically concentrated on medical question-answering and examinations, ignoring the multifaceted nature of clinical decision-making in inpatient settings. To address these gaps, we first developed the Inpatient Pathway Decision Support (IPDS) benchmark from the MIMIC-IV database, encompassing 51,274 cases across nine triage departments and 17 major disease categories alongside 16 standardized treatment options. Then, we proposed the Multi-Agent Inpatient Pathways (MAP) framework to accomplish inpatient pathways with three clinical agents, including a triage agent managing the patient admission, a diagnosis agent serving as the primary decision maker at the department, and a treatment agent providing treatment plans. Additionally, our MAP framework includes a chief agent overseeing the inpatient pathways to guide and promote these three clinician agents. Extensive experiments showed our MAP improved the diagnosis accuracy by 25.10% compared to the state-of-the-art LLM HuatuoGPT2-13B. It is worth noting that our MAP demonstrated significant clinical compliance, outperforming three board-certified clinicians by 10%-12%, establishing a foundation for inpatient pathways systems. 

---
# Are LLMs (Really) Ideological? An IRT-based Analysis and Alignment Tool for Perceived Socio-Economic Bias in LLMs 

**Authors**: Jasmin Wachter, Michael Radloff, Maja Smolej, Katharina Kinder-Kurlanda  

**Link**: [PDF](https://arxiv.org/pdf/2503.13149)  

**Abstract**: We introduce an Item Response Theory (IRT)-based framework to detect and quantify socioeconomic bias in large language models (LLMs) without relying on subjective human judgments. Unlike traditional methods, IRT accounts for item difficulty, improving ideological bias estimation. We fine-tune two LLM families (Meta-LLaMa 3.2-1B-Instruct and Chat- GPT 3.5) to represent distinct ideological positions and introduce a two-stage approach: (1) modeling response avoidance and (2) estimating perceived bias in answered responses. Our results show that off-the-shelf LLMs often avoid ideological engagement rather than exhibit bias, challenging prior claims of partisanship. This empirically validated framework enhances AI alignment research and promotes fairer AI governance. 

---
# Logic-in-Frames: Dynamic Keyframe Search via Visual Semantic-Logical Verification for Long Video Understanding 

**Authors**: Weiyu Guo, Ziyang Chen, Shaoguang Wang, Jianxiang He, Yijie Xu, Jinhui Ye, Ying Sun, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.13139)  

**Abstract**: Understanding long video content is a complex endeavor that often relies on densely sampled frame captions or end-to-end feature selectors, yet these techniques commonly overlook the logical relationships between textual queries and visual elements. In practice, computational constraints necessitate coarse frame subsampling, a challenge analogous to ``finding a needle in a haystack.'' To address this issue, we introduce a semantics-driven search framework that reformulates keyframe selection under the paradigm of Visual Semantic-Logical Search. Specifically, we systematically define four fundamental logical dependencies: 1) spatial co-occurrence, 2) temporal proximity, 3) attribute dependency, and 4) causal order. These relations dynamically update frame sampling distributions through an iterative refinement process, enabling context-aware identification of semantically critical frames tailored to specific query requirements. Our method establishes new SOTA performance on the manually annotated benchmark in key-frame selection metrics. Furthermore, when applied to downstream video question-answering tasks, the proposed approach demonstrates the best performance gains over existing methods on LongVideoBench and Video-MME, validating its effectiveness in bridging the logical gap between textual queries and visual-temporal reasoning. The code will be publicly available. 

---
# MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs 

**Authors**: Erik Daxberger, Nina Wenzel, David Griffiths, Haiming Gang, Justin Lazarow, Gefen Kohavi, Kai Kang, Marcin Eichner, Yinfei Yang, Afshin Dehghan, Peter Grasch  

**Link**: [PDF](https://arxiv.org/pdf/2503.13111)  

**Abstract**: Multimodal large language models (MLLMs) excel at 2D visual understanding but remain limited in their ability to reason about 3D space. In this work, we leverage large-scale high-quality 3D scene data with open-set annotations to introduce 1) a novel supervised fine-tuning dataset and 2) a new evaluation benchmark, focused on indoor scenes. Our Cubify Anything VQA (CA-VQA) data covers diverse spatial tasks including spatial relationship prediction, metric size and distance estimation, and 3D grounding. We show that CA-VQA enables us to train MM-Spatial, a strong generalist MLLM that also achieves state-of-the-art performance on 3D spatial understanding benchmarks, including our own. We show how incorporating metric depth and multi-view inputs (provided in CA-VQA) can further improve 3D understanding, and demonstrate that data alone allows our model to achieve depth perception capabilities comparable to dedicated monocular depth estimation models. We will publish our SFT dataset and benchmark. 

---
# Intra-neuronal attention within language models Relationships between activation and semantics 

**Authors**: Michael Pichat, William Pogrund, Paloma Pichat, Armanouche Gasparian, Samuel Demarchi, Corbet Alois Georgeon, Michael Veillet-Guillem  

**Link**: [PDF](https://arxiv.org/pdf/2503.12992)  

**Abstract**: This study investigates the ability of perceptron-type neurons in language models to perform intra-neuronal attention; that is, to identify different homogeneous categorical segments within the synthetic thought category they encode, based on a segmentation of specific activation zones for the tokens to which they are particularly responsive. The objective of this work is therefore to determine to what extent formal neurons can establish a homomorphic relationship between activation-based and categorical segmentations. The results suggest the existence of such a relationship, albeit tenuous, only at the level of tokens with very high activation levels. This intra-neuronal attention subsequently enables categorical restructuring processes at the level of neurons in the following layer, thereby contributing to the progressive formation of high-level categorical abstractions. 

---
# R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization 

**Authors**: Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.12937)  

**Abstract**: Recent studies generally enhance MLLMs' reasoning capabilities via supervised fine-tuning on high-quality chain-of-thought reasoning data, which often leads models to merely imitate successful reasoning paths without understanding what the wrong reasoning paths are. In this work, we aim to enhance the MLLMs' reasoning ability beyond passively imitating positive reasoning paths. To this end, we design Step-wise Group Relative Policy Optimization (StepGRPO), a new online reinforcement learning framework that enables MLLMs to self-improve reasoning ability via simple, effective and dense step-wise rewarding. Specifically, StepGRPO introduces two novel rule-based reasoning rewards: Step-wise Reasoning Accuracy Reward (StepRAR) and Step-wise Reasoning Validity Reward (StepRVR). StepRAR rewards the reasoning paths that contain necessary intermediate reasoning steps via a soft key-step matching technique, while StepRAR rewards reasoning paths that follow a well-structured and logically consistent reasoning process through a reasoning completeness and logic evaluation strategy. With the proposed StepGRPO, we introduce R1-VL, a series of MLLMs with outstanding capabilities in step-by-step reasoning. Extensive experiments over 8 benchmarks demonstrate the superiority of our methods. 

---
# A Semantic-based Optimization Approach for Repairing LLMs: Case Study on Code Generation 

**Authors**: Jian Gu, Aldeida Aleti, Chunyang Chen, Hongyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12899)  

**Abstract**: Language Models (LMs) are widely used in software engineering for code generation, but they may produce code with errors. Rather than repairing the generated code, an alternative way is to address the underlying failures of models. LM repair offers a lightweight solution to this challenge: it requires minimal data, reduces computational costs, and reduces the side effects. Unlike retraining, LM repair focuses on applying tailored updates to targeted neurons, making it ideal for scenarios with limited resources, high-performance demands, or strict safety requirements. In this paper, we propose \ul{S}emantic \ul{T}argeting for \ul{A}nalytical \ul{R}epair (\textsc{STAR}), a pioneering and novel semantic-based optimization approach for repairing LLMs. \textsc{STAR} realizes main operations in LM repair methods in an optimization process, including locating ``buggy neurons'', solving ``neuron patches'', and patching ``buggy neurons''. Correspondingly, it computes the deltas of weight matrix as the prior information to guide optimization; and attributes the targeted layers and neurons leveraging statistical insights. The neuron patches are computed with a solid semantic-based analytical formula, which directly bridges the changes to logits with the deltas of neurons, by steering latent representations. Compared to the prior work of LM repair (\textsc{MINT}) and optimization methods (\textsc{SGD}), \textsc{STAR} integrates their strengths while mitigating their limitations. \textsc{STAR} supports solving multiple failures together, significantly improving the usefulness. Evaluated on three code generation tasks using popular code LMs, \textsc{STAR} demonstrates superior effectiveness. Additionally, \textsc{STAR} exhibits better efficiency. In terms of side effects, namely the balance between generalization and specificity, \textsc{STAR} outperforms prior work by a significant margin. 

---
# VITED: Video Temporal Evidence Distillation 

**Authors**: Yujie Lu, Yale Song, William Wang, Lorenzo Torresani, Tushar Nagarajan  

**Link**: [PDF](https://arxiv.org/pdf/2503.12855)  

**Abstract**: We investigate complex video question answering via chain-of-evidence reasoning -- identifying sequences of temporal spans from multiple relevant parts of the video, together with visual evidence within them. Existing models struggle with multi-step reasoning as they uniformly sample a fixed number of frames, which can miss critical evidence distributed nonuniformly throughout the video. Moreover, they lack the ability to temporally localize such evidence in the broader context of the full video, which is required for answering complex questions. We propose a framework to enhance existing VideoQA datasets with evidence reasoning chains, automatically constructed by searching for optimal intervals of interest in the video with supporting evidence, that maximizes the likelihood of answering a given question. We train our model (VITED) to generate these evidence chains directly, enabling it to both localize evidence windows as well as perform multi-step reasoning across them in long-form video content. We show the value of our evidence-distilled models on a suite of long video QA benchmarks where we outperform state-of-the-art approaches that lack evidence reasoning capabilities. 

---
# A Multi-Power Law for Loss Curve Prediction Across Learning Rate Schedules 

**Authors**: Kairong Luo, Haodong Wen, Shengding Hu, Zhenbo Sun, Zhiyuan Liu, Maosong Sun, Kaifeng Lyu, Wenguang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.12811)  

**Abstract**: Training large models is both resource-intensive and time-consuming, making it crucial to understand the quantitative relationship between model performance and hyperparameters. In this paper, we present an empirical law that describes how the pretraining loss of large language models evolves under different learning rate schedules, such as constant, cosine, and step decay schedules. Our proposed law takes a multi-power form, combining a power law based on the sum of learning rates and additional power laws to account for a loss reduction effect induced by learning rate decay. We extensively validate this law on various model sizes and architectures, and demonstrate that after fitting on a few learning rate schedules, the law accurately predicts the loss curves for unseen schedules of different shapes and horizons. Moreover, by minimizing the predicted final pretraining loss across learning rate schedules, we are able to find a schedule that outperforms the widely used cosine learning rate schedule. Interestingly, this automatically discovered schedule bears some resemblance to the recently proposed Warmup-Stable-Decay (WSD) schedule (Hu et al, 2024) but achieves a slightly lower final loss. We believe these results could offer valuable insights for understanding the dynamics of pretraining and designing learning rate schedules to improve efficiency. 

---
# DeepPerception: Advancing R1-like Cognitive Visual Perception in MLLMs for Knowledge-Intensive Visual Grounding 

**Authors**: Xinyu Ma, Ziyang Ding, Zhicong Luo, Chi Chen, Zonghao Guo, Derek F. Wong, Xiaoyi Feng, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.12797)  

**Abstract**: Human experts excel at fine-grained visual discrimination by leveraging domain knowledge to refine perceptual features, a capability that remains underdeveloped in current Multimodal Large Language Models (MLLMs). Despite possessing vast expert-level knowledge, MLLMs struggle to integrate reasoning into visual perception, often generating direct responses without deeper analysis. To bridge this gap, we introduce knowledge-intensive visual grounding (KVG), a novel visual grounding task that requires both fine-grained perception and domain-specific knowledge integration. To address the challenges of KVG, we propose DeepPerception, an MLLM enhanced with cognitive visual perception capabilities. Our approach consists of (1) an automated data synthesis pipeline that generates high-quality, knowledge-aligned training samples, and (2) a two-stage training framework combining supervised fine-tuning for cognitive reasoning scaffolding and reinforcement learning to optimize perception-cognition synergy. To benchmark performance, we introduce KVG-Bench a comprehensive dataset spanning 10 domains with 1.3K manually curated test cases. Experimental results demonstrate that DeepPerception significantly outperforms direct fine-tuning, achieving +8.08\% accuracy improvements on KVG-Bench and exhibiting +4.60\% superior cross-domain generalization over baseline approaches. Our findings highlight the importance of integrating cognitive processes into MLLMs for human-like visual perception and open new directions for multimodal reasoning research. The data, codes, and models are released at this https URL. 

---
# A Reinforcement Learning-Driven Transformer GAN for Molecular Generation 

**Authors**: Chen Li, Huidong Tang, Ye Zhu, Yoshihiro Yamanishi  

**Link**: [PDF](https://arxiv.org/pdf/2503.12796)  

**Abstract**: Generating molecules with desired chemical properties presents a critical challenge in fields such as chemical synthesis and drug discovery. Recent advancements in artificial intelligence (AI) and deep learning have significantly contributed to data-driven molecular generation. However, challenges persist due to the inherent sensitivity of simplified molecular input line entry system (SMILES) representations and the difficulties in applying generative adversarial networks (GANs) to discrete data. This study introduces RL-MolGAN, a novel Transformer-based discrete GAN framework designed to address these challenges. Unlike traditional Transformer architectures, RL-MolGAN utilizes a first-decoder-then-encoder structure, facilitating the generation of drug-like molecules from both $de~novo$ and scaffold-based designs. In addition, RL-MolGAN integrates reinforcement learning (RL) and Monte Carlo tree search (MCTS) techniques to enhance the stability of GAN training and optimize the chemical properties of the generated molecules. To further improve the model's performance, RL-MolWGAN, an extension of RL-MolGAN, incorporates Wasserstein distance and mini-batch discrimination, which together enhance the stability of the GAN. Experimental results on two widely used molecular datasets, QM9 and ZINC, validate the effectiveness of our models in generating high-quality molecular structures with diverse and desirable chemical properties. 

---
# Identifying Cooperative Personalities in Multi-agent Contexts through Personality Steering with Representation Engineering 

**Authors**: Kenneth J. K. Ong, Lye Jia Jun, Hieu Minh "Jord" Nguyen, Seong Hah Cho, Natalia Pérez-Campanero Antolín  

**Link**: [PDF](https://arxiv.org/pdf/2503.12722)  

**Abstract**: As Large Language Models (LLMs) gain autonomous capabilities, their coordination in multi-agent settings becomes increasingly important. However, they often struggle with cooperation, leading to suboptimal outcomes. Inspired by Axelrod's Iterated Prisoner's Dilemma (IPD) tournaments, we explore how personality traits influence LLM cooperation. Using representation engineering, we steer Big Five traits (e.g., Agreeableness, Conscientiousness) in LLMs and analyze their impact on IPD decision-making. Our results show that higher Agreeableness and Conscientiousness improve cooperation but increase susceptibility to exploitation, highlighting both the potential and limitations of personality-based steering for aligning AI agents. 

---
# Logic-RAG: Augmenting Large Multimodal Models with Visual-Spatial Knowledge for Road Scene Understanding 

**Authors**: Imran Kabir, Md Alimoor Reza, Syed Billah  

**Link**: [PDF](https://arxiv.org/pdf/2503.12663)  

**Abstract**: Large multimodal models (LMMs) are increasingly integrated into autonomous driving systems for user interaction. However, their limitations in fine-grained spatial reasoning pose challenges for system interpretability and user trust. We introduce Logic-RAG, a novel Retrieval-Augmented Generation (RAG) framework that improves LMMs' spatial understanding in driving scenarios. Logic-RAG constructs a dynamic knowledge base (KB) about object-object relationships in first-order logic (FOL) using a perception module, a query-to-logic embedder, and a logical inference engine. We evaluated Logic-RAG on visual-spatial queries using both synthetic and real-world driving videos. When using popular LMMs (GPT-4V, Claude 3.5) as proxies for an autonomous driving system, these models achieved only 55% accuracy on synthetic driving scenes and under 75% on real-world driving scenes. Augmenting them with Logic-RAG increased their accuracies to over 80% and 90%, respectively. An ablation study showed that even without logical inference, the fact-based context constructed by Logic-RAG alone improved accuracy by 15%. Logic-RAG is extensible: it allows seamless replacement of individual components with improved versions and enables domain experts to compose new knowledge in both FOL and natural language. In sum, Logic-RAG addresses critical spatial reasoning deficiencies in LMMs for autonomous driving applications. Code and data are available at this https URL. 

---
# VeriLA: A Human-Centered Evaluation Framework for Interpretable Verification of LLM Agent Failures 

**Authors**: Yoo Yeon Sung, Hannah Kim, Dan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12651)  

**Abstract**: AI practitioners increasingly use large language model (LLM) agents in compound AI systems to solve complex reasoning tasks, these agent executions often fail to meet human standards, leading to errors that compromise the system's overall performance. Addressing these failures through human intervention is challenging due to the agents' opaque reasoning processes, misalignment with human expectations, the complexity of agent dependencies, and the high cost of manual inspection. This paper thus introduces a human-centered evaluation framework for Verifying LLM Agent failures (VeriLA), which systematically assesses agent failures to reduce human effort and make these agent failures interpretable to humans. The framework first defines clear expectations of each agent by curating human-designed agent criteria. Then, it develops a human-aligned agent verifier module, trained with human gold standards, to assess each agent's execution output. This approach enables granular evaluation of each agent's performance by revealing failures from a human standard, offering clear guidelines for revision, and reducing human cognitive load. Our case study results show that VeriLA is both interpretable and efficient in helping practitioners interact more effectively with the system. By upholding accountability in human-agent collaboration, VeriLA paves the way for more trustworthy and human-aligned compound AI systems. 

---
# Online Misinformation Detection in Live Streaming Videos 

**Authors**: Rui Cao  

**Link**: [PDF](https://arxiv.org/pdf/2503.12627)  

**Abstract**: Online misinformation detection is an important issue and methods are proposed to detect and curb misinformation in various forms. However, previous studies are conducted in an offline manner. We claim a realistic misinformation detection setting that has not been studied yet is online misinformation detection in live streaming videos (MDLS). In the proposal, we formulate the problem of MDLS and illustrate the importance and the challenge of the task. Besides, we propose feasible ways of developing the problem into AI challenges as well as potential solutions to the problem. 

---
# AdaReTaKe: Adaptive Redundancy Reduction to Perceive Longer for Video-language Understanding 

**Authors**: Xiao Wang, Qingyi Si, Jianlong Wu, Shiyu Zhu, Li Cao, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2503.12559)  

**Abstract**: Multimodal Large Language Models (MLLMs) have revolutionized video understanding, yet are still limited by context length when processing long videos. Recent methods compress videos by leveraging visual redundancy uniformly, yielding promising results. Nevertheless, our quantitative analysis shows that redundancy varies significantly across time and model layers, necessitating a more flexible compression strategy. We propose AdaReTaKe, a training-free method that flexibly reduces visual redundancy by allocating compression ratios among time and layers with theoretical guarantees. Integrated into state-of-the-art MLLMs, AdaReTaKe improves processing capacity from 256 to 2048 frames while preserving critical information. Experiments on VideoMME, MLVU, LongVideoBench, and LVBench datasets demonstrate that AdaReTaKe outperforms existing methods by 2.3% and 2.8% for 7B and 72B models, respectively, with even greater improvements of 5.9% and 6.0% on the longest LVBench. Our code is available at this https URL. 

---
# CorpusStudio: Surfacing Emergent Patterns in a Corpus of Prior Work while Writing 

**Authors**: Hai Dang, Chelse Swoopes, Daniel Buschek, Elena L. Glassman  

**Link**: [PDF](https://arxiv.org/pdf/2503.12436)  

**Abstract**: Many communities, including the scientific community, develop implicit writing norms. Understanding them is crucial for effective communication with that community. Writers gradually develop an implicit understanding of norms by reading papers and receiving feedback on their writing. However, it is difficult to both externalize this knowledge and apply it to one's own writing. We propose two new writing support concepts that reify document and sentence-level patterns in a given text corpus: (1) an ordered distribution over section titles and (2) given the user's draft and cursor location, many retrieved contextually relevant sentences. Recurring words in the latter are algorithmically highlighted to help users see any emergent norms. Study results (N=16) show that participants revised the structure and content using these concepts, gaining confidence in aligning with or breaking norms after reviewing many examples. These results demonstrate the value of reifying distributions over other authors' writing choices during the writing process. 

---
# IPCGRL: Language-Instructed Reinforcement Learning for Procedural Level Generation 

**Authors**: In-Chang Baek, Sung-Hyun Kim, Seo-yung Lee, Dong-Hyun Lee, Kyung-Joong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.12358)  

**Abstract**: Recent research has highlighted the significance of natural language in enhancing the controllability of generative models. While various efforts have been made to leverage natural language for content generation, research on deep reinforcement learning (DRL) agents utilizing text-based instructions for procedural content generation remains limited. In this paper, we propose IPCGRL, an instruction-based procedural content generation method via reinforcement learning, which incorporates a sentence embedding model. IPCGRL fine-tunes task-specific embedding representations to effectively compress game-level conditions. We evaluate IPCGRL in a two-dimensional level generation task and compare its performance with a general-purpose embedding method. The results indicate that IPCGRL achieves up to a 21.4% improvement in controllability and a 17.2% improvement in generalizability for unseen instructions. Furthermore, the proposed method extends the modality of conditional input, enabling a more flexible and expressive interaction framework for procedural content generation. 

---
# CapArena: Benchmarking and Analyzing Detailed Image Captioning in the LLM Era 

**Authors**: Kanzhi Cheng, Wenpo Song, Jiaxin Fan, Zheng Ma, Qiushi Sun, Fangzhi Xu, Chenyang Yan, Nuo Chen, Jianbing Zhang, Jiajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.12329)  

**Abstract**: Image captioning has been a longstanding challenge in vision-language research. With the rise of LLMs, modern Vision-Language Models (VLMs) generate detailed and comprehensive image descriptions. However, benchmarking the quality of such captions remains unresolved. This paper addresses two key questions: (1) How well do current VLMs actually perform on image captioning, particularly compared to humans? We built CapArena, a platform with over 6000 pairwise caption battles and high-quality human preference votes. Our arena-style evaluation marks a milestone, showing that leading models like GPT-4o achieve or even surpass human performance, while most open-source models lag behind. (2) Can automated metrics reliably assess detailed caption quality? Using human annotations from CapArena, we evaluate traditional and recent captioning metrics, as well as VLM-as-a-Judge. Our analysis reveals that while some metrics (e.g., METEOR) show decent caption-level agreement with humans, their systematic biases lead to inconsistencies in model ranking. In contrast, VLM-as-a-Judge demonstrates robust discernment at both the caption and model levels. Building on these insights, we release CapArena-Auto, an accurate and efficient automated benchmark for detailed captioning, achieving 94.3% correlation with human rankings at just $4 per test. Data and resources will be open-sourced at this https URL. 

---
# One Goal, Many Challenges: Robust Preference Optimization Amid Content-Aware and Multi-Source Noise 

**Authors**: Amirabbas Afzali, Amirhossein Afsharrad, Seyed Shahabeddin Mousavi, Sanjay Lall  

**Link**: [PDF](https://arxiv.org/pdf/2503.12301)  

**Abstract**: Large Language Models (LLMs) have made significant strides in generating human-like responses, largely due to preference alignment techniques. However, these methods often assume unbiased human feedback, which is rarely the case in real-world scenarios. This paper introduces Content-Aware Noise-Resilient Preference Optimization (CNRPO), a novel framework that addresses multiple sources of content-dependent noise in preference learning. CNRPO employs a multi-objective optimization approach to separate true preferences from content-aware noises, effectively mitigating their impact. We leverage backdoor attack mechanisms to efficiently learn and control various noise sources within a single model. Theoretical analysis and extensive experiments on different synthetic noisy datasets demonstrate that CNRPO significantly improves alignment with primary human preferences while controlling for secondary noises and biases, such as response length and harmfulness. 

---
# Hyperbolic Safety-Aware Vision-Language Models 

**Authors**: Tobia Poppi, Tejaswi Kasarla, Pascal Mettes, Lorenzo Baraldi, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2503.12127)  

**Abstract**: Addressing the retrieval of unsafe content from vision-language models such as CLIP is an important step towards real-world integration. Current efforts have relied on unlearning techniques that try to erase the model's knowledge of unsafe concepts. While effective in reducing unwanted outputs, unlearning limits the model's capacity to discern between safe and unsafe content. In this work, we introduce a novel approach that shifts from unlearning to an awareness paradigm by leveraging the inherent hierarchical properties of the hyperbolic space. We propose to encode safe and unsafe content as an entailment hierarchy, where both are placed in different regions of hyperbolic space. Our HySAC, Hyperbolic Safety-Aware CLIP, employs entailment loss functions to model the hierarchical and asymmetrical relations between safe and unsafe image-text pairs. This modelling, ineffective in standard vision-language models due to their reliance on Euclidean embeddings, endows the model with awareness of unsafe content, enabling it to serve as both a multimodal unsafe classifier and a flexible content retriever, with the option to dynamically redirect unsafe queries toward safer alternatives or retain the original output. Extensive experiments show that our approach not only enhances safety recognition but also establishes a more adaptable and interpretable framework for content moderation in vision-language models. Our source code is available at this https URL. 

---
# Semantic-Clipping: Efficient Vision-Language Modeling with Semantic-Guidedd Visual Selection 

**Authors**: Bangzheng Li, Fei Wang, Wenxuan Zhou, Nan Xu, Ben Zhou, Sheng Zhang, Hoifung Poon, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11794)  

**Abstract**: Vision-Language Models (VLMs) leverage aligned visual encoders to transform images into visual tokens, allowing them to be processed similarly to text by the backbone large language model (LLM). This unified input paradigm enables VLMs to excel in vision-language tasks such as visual question answering (VQA). To improve fine-grained visual reasoning, recent advancements in vision-language modeling introduce image cropping techniques that feed all encoded sub-images into the model. However, this approach significantly increases the number of visual tokens, leading to inefficiency and potential distractions for the LLM. To address the generalization challenges of image representation in VLMs, we propose a lightweight, universal framework that seamlessly integrates with existing VLMs to enhance their ability to process finegrained details. Our method leverages textual semantics to identify key visual areas, improving VQA performance without requiring any retraining of the VLM. Additionally, it incorporates textual signals into the visual encoding process, enhancing both efficiency and effectiveness. The proposed method, SEMCLIP, strengthens the visual understanding of a 7B VLM, LLaVA-1.5 by 3.3% on average across 7 benchmarks, and particularly by 5.3% on the challenging detailed understanding benchmark V*. 

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
# Lorecast: Layout-Aware Performance and Power Forecasting from Natural Language 

**Authors**: Runzhi Wang, Prianka Sengupta, Yiran Chen, Jiang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11662)  

**Abstract**: In chip design planning, obtaining reliable performance and power forecasts for various design options is of critical importance. Traditionally, this involves using system-level models, which often lack accuracy, or trial synthesis, which is both labor-intensive and time-consuming. We introduce a new methodology, called Lorecast, which accepts English prompts as input to rapidly generate layout-aware performance and power estimates. This approach bypasses the need for HDL code development or synthesis, making it both fast and user-friendly. Experimental results demonstrate that Lorecast achieves accuracy within a few percent of error compared to post-layout analysis. 

---
