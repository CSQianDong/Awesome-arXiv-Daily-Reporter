# Causal Estimation of Tokenisation Bias 

**Authors**: Pietro Lesci, Clara Meister, Thomas Hofmann, Andreas Vlachos, Tiago Pimentel  

**Link**: [PDF](https://arxiv.org/pdf/2506.03149)  

**Abstract**: Modern language models are typically trained over subword sequences, but ultimately define probabilities over character-strings. Ideally, the choice of the tokeniser -- which maps character-strings to subwords -- should not affect the probability assigned to the underlying character-string; in practice, it does. We define this mismatch as tokenisation bias. In this work, we quantify one particular type of tokenisation bias: the effect of including or not a subword (e.g., $\langle hello \rangle$) in a tokeniser's vocabulary on the probability a trained model assigns to the corresponding characters (i.e., \textit{``hello''}). Estimating this effect is challenging because each model is trained with only one tokeniser. We address this by framing tokenisation bias as a causal effect and estimating it using the regression discontinuity design. Specifically, we exploit the fact that tokenisation algorithms rank subwords and add the first $K$ to a tokeniser's vocabulary, where $K$ is an arbitrary cutoff point. As such, we can estimate a causal effect by comparing similar subwords around this cutoff. Experimentally, we find that tokenisation consistently affects models' outputs across scales, vocabularies, and tokenisers. Notably, a subword's presence in a small model's vocabulary may increase its characters' probability by up to 17 times, highlighting tokenisation as a key design choice in language modelling. 

---
# Entity-Augmented Neuroscience Knowledge Retrieval Using Ontology and Semantic Understanding Capability of LLM 

**Authors**: Pralaypati Ta, Sriram Venkatesaperumal, Keerthi Ram, Mohanasankar Sivaprakasam  

**Link**: [PDF](https://arxiv.org/pdf/2506.03145)  

**Abstract**: Neuroscience research publications encompass a vast wealth of knowledge. Accurately retrieving existing information and discovering new insights from this extensive literature is essential for advancing the field. However, when knowledge is dispersed across multiple sources, current state-of-the-art retrieval methods often struggle to extract the necessary information. A knowledge graph (KG) can integrate and link knowledge from multiple sources, but existing methods for constructing KGs in neuroscience often rely on labeled data and require domain expertise. Acquiring large-scale, labeled data for a specialized area like neuroscience presents significant challenges. This work proposes novel methods for constructing KG from unlabeled large-scale neuroscience research corpus utilizing large language models (LLM), neuroscience ontology, and text embeddings. We analyze the semantic relevance of neuroscience text segments identified by LLM for building the knowledge graph. We also introduce an entity-augmented information retrieval algorithm to extract knowledge from the KG. Several experiments were conducted to evaluate the proposed approaches, and the results demonstrate that our methods significantly enhance knowledge discovery from the unlabeled neuroscience research corpus. It achieves an F1 score of 0.84 for entity extraction, and the knowledge obtained from the KG improves answers to over 54% of the questions. 

---
# GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents 

**Authors**: Qianhui Wu, Kanzhi Cheng, Rui Yang, Chaoyun Zhang, Jianwei Yang, Huiqiang Jiang, Jian Mu, Baolin Peng, Bo Qiao, Reuben Tan, Si Qin, Lars Liden, Qingwei Lin, Huan Zhang, Tong Zhang, Jianbing Zhang, Dongmei Zhang, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.03143)  

**Abstract**: One of the principal challenges in building VLM-powered GUI agents is visual grounding, i.e., localizing the appropriate screen region for action execution based on both the visual content and the textual plans. Most existing work formulates this as a text-based coordinate generation task. However, these approaches suffer from several limitations: weak spatial-semantic alignment, inability to handle ambiguous supervision targets, and a mismatch between the dense nature of screen coordinates and the coarse, patch-level granularity of visual features extracted by models like Vision Transformers. In this paper, we propose GUI-Actor, a VLM-based method for coordinate-free GUI grounding. At its core, GUI-Actor introduces an attention-based action head that learns to align a dedicated <ACTOR> token with all relevant visual patch tokens, enabling the model to propose one or more action regions in a single forward pass. In line with this, we further design a grounding verifier to evaluate and select the most plausible action region from the candidates proposed for action execution. Extensive experiments show that GUI-Actor outperforms prior state-of-the-art methods on multiple GUI action grounding benchmarks, with improved generalization to unseen screen resolutions and layouts. Notably, GUI-Actor-7B even surpasses UI-TARS-72B (38.1) on ScreenSpot-Pro, achieving scores of 40.7 with Qwen2-VL and 44.6 with Qwen2.5-VL as backbones. Furthermore, by incorporating the verifier, we find that fine-tuning only the newly introduced action head (~100M parameters for 7B model) while keeping the VLM backbone frozen is sufficient to achieve performance comparable to previous state-of-the-art models, highlighting that GUI-Actor can endow the underlying VLM with effective grounding capabilities without compromising its general-purpose strengths. 

---
# Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning 

**Authors**: Yinjie Wang, Ling Yang, Ye Tian, Ke Shen, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03136)  

**Abstract**: We propose CURE, a novel reinforcement learning framework with a dedicated reward design that co-evolves coding and unit test generation capabilities based on their interaction outcomes, without any ground-truth code as supervision. This approach enables flexible and scalable training and allows the unit tester to learn directly from the coder's mistakes. Our derived ReasonFlux-Coder-7B and 14B models improve code generation accuracy by 5.3% and Best-of-N accuracy by 9.0% after optimization on Qwen2.5-Instruct models, outperforming similarly sized Qwen-Coder, DeepSeek-Coder, and Seed-Coder. They naturally extend to downstream tasks such as test-time scaling and agentic coding-achieving a 8.1% improvement over the base model. For the long-CoT model, our ReasonFlux-Coder-4B consistently outperforms Qwen3-4B while achieving 64.8% inference efficiency in unit test generation. Notably, we also find that our model can serve as an effective reward model for reinforcement learning on base models. Project: this https URL 

---
# AUTOCIRCUIT-RL: Reinforcement Learning-Driven LLM for Automated Circuit Topology Generation 

**Authors**: Prashanth Vijayaraghavan, Luyao Shi, Ehsan Degan, Vandana Mukherjee, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03122)  

**Abstract**: Analog circuit topology synthesis is integral to Electronic Design Automation (EDA), enabling the automated creation of circuit structures tailored to specific design requirements. However, the vast design search space and strict constraint adherence make efficient synthesis challenging. Leveraging the versatility of Large Language Models (LLMs), we propose AUTOCIRCUIT-RL,a novel reinforcement learning (RL)-based framework for automated analog circuit synthesis. The framework operates in two phases: instruction tuning, where an LLM learns to generate circuit topologies from structured prompts encoding design constraints, and RL refinement, which further improves the instruction-tuned model using reward models that evaluate validity, efficiency, and output voltage. The refined model is then used directly to generate topologies that satisfy the design constraints. Empirical results show that AUTOCIRCUIT-RL generates ~12% more valid circuits and improves efficiency by ~14% compared to the best baselines, while reducing duplicate generation rates by ~38%. It achieves over 60% success in synthesizing valid circuits with limited training data, demonstrating strong generalization. These findings highlight the framework's effectiveness in scaling to complex circuits while maintaining efficiency and constraint adherence, marking a significant advancement in AI-driven circuit design. 

---
# Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback 

**Authors**: Xiaoying Zhang, Hao Sun, Yipeng Zhang, Kaituo Feng, Chao Yang, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2506.03106)  

**Abstract**: Recent advances in reinforcement learning (RL) with numerical feedback, such as scalar rewards, have significantly enhanced the complex reasoning capabilities of large language models (LLMs). Despite this success, we identify three key challenges encountered by RL with solely numerical feedback: performance plateaus, limited effectiveness of self-reflection, and persistent failures. We then demonstrate that RL-finetuned models, even after exhibiting performance plateaus, can generate correct refinements on persistently failed problems by leveraging natural language feedback in the form of critiques. Building on this insight, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided refinements simultaneously while maintaining exploration. Extensive experiments using Qwen2.5-7B-Base and Qwen3-8B-Base show that Critique-GRPO consistently outperforms supervised learning-based and RL-based fine-tuning approaches across eight challenging mathematical, STEM, and general reasoning tasks, improving average pass@1 scores by approximately 4.5% and 5%, respectively. Notably, Critique-GRPO surpasses a strong baseline that incorporates expert demonstrations within online RL. Further analysis reveals two critical insights about policy exploration: (1) higher entropy does not always guarantee efficient learning from exploration, and (2) longer responses do not necessarily lead to more effective exploration. 

---
# Beyond Text Compression: Evaluating Tokenizers Across Scales 

**Authors**: Jonas F. Lotz, António V. Lopes, Stephan Peitz, Hendra Setiawan, Leonardo Emili  

**Link**: [PDF](https://arxiv.org/pdf/2506.03101)  

**Abstract**: The choice of tokenizer can profoundly impact language model performance, yet accessible and reliable evaluations of tokenizer quality remain an open challenge. Inspired by scaling consistency, we show that smaller models can accurately predict significant differences in tokenizer impact on larger models at a fraction of the compute cost. By systematically evaluating both English-centric and multilingual tokenizers, we find that tokenizer choice has negligible effects on tasks in English but results in consistent performance differences in multilingual settings. We propose new intrinsic tokenizer metrics inspired by Zipf's law that correlate more strongly with downstream performance than text compression when modeling unseen languages. By combining several metrics to capture multiple aspects of tokenizer behavior, we develop a reliable framework for intrinsic tokenizer evaluations. Our work offers a more efficient path to informed tokenizer selection in future language model development. 

---
# Literary Evidence Retrieval via Long-Context Language Models 

**Authors**: Katherine Thai, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2506.03090)  

**Abstract**: How well do modern long-context language models understand literary fiction? We explore this question via the task of literary evidence retrieval, repurposing the RELiC dataset of That et al. (2022) to construct a benchmark where the entire text of a primary source (e.g., The Great Gatsby) is provided to an LLM alongside literary criticism with a missing quotation from that work. This setting, in which the model must generate the missing quotation, mirrors the human process of literary analysis by requiring models to perform both global narrative reasoning and close textual examination. We curate a high-quality subset of 292 examples through extensive filtering and human verification. Our experiments show that recent reasoning models, such as Gemini Pro 2.5 can exceed human expert performance (62.5% vs. 50% accuracy). In contrast, the best open-weight model achieves only 29.1% accuracy, highlighting a wide gap in interpretive reasoning between open and closed-weight models. Despite their speed and apparent accuracy, even the strongest models struggle with nuanced literary signals and overgeneration, signaling open challenges for applying LLMs to literary analysis. We release our dataset and evaluation code to encourage future work in this direction. 

---
# Facts Do Care About Your Language: Assessing Answer Quality of Multilingual LLMs 

**Authors**: Yuval Kansal, Shmuel Berman, Lydia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03051)  

**Abstract**: Factuality is a necessary precursor to useful educational tools. As adoption of Large Language Models (LLMs) in education continues of grow, ensuring correctness in all settings is paramount. Despite their strong English capabilities, LLM performance in other languages is largely untested. In this work, we evaluate the correctness of the Llama3.1 family of models in answering factual questions appropriate for middle and high school students. We demonstrate that LLMs not only provide extraneous and less truthful information, but also exacerbate existing biases against rare languages. 

---
# Towards Analyzing and Understanding the Limitations of VAPO: A Theoretical Perspective 

**Authors**: Jintian Shao, Yiming Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.03038)  

**Abstract**: Reinforcement learning (RL) enhances large language models (LLMs) in complex, long-chain-of-thought (long-CoT) reasoning. The advanced VAPO framework, despite sophisticated mechanisms like Decoupled GAE, theoretically faces fundamental limitations in comprehensively modeling and leveraging deep, long-term value for fine-grained, step-by-step policy guidance in extended reasoning chains. We argue these limitations stem from inherent difficulties in credit assignment, value function representational capacity with temporally abstracted goals, and translating global value signals into local policy improvements, especially with sparse rewards. Our theoretical analysis examines these aspects to illuminate VAPO's boundaries in long-term value modeling, aiming to deepen understanding of current RL for advanced reasoning and suggest future research for more robust LLM agents. 

---
# Leveraging Information Retrieval to Enhance Spoken Language Understanding Prompts in Few-Shot Learning 

**Authors**: Pierre Lepagnol, Sahar Ghannay, Thomas Gerald, Christophe Servan, Sophie Rosset  

**Link**: [PDF](https://arxiv.org/pdf/2506.03035)  

**Abstract**: Understanding user queries is fundamental in many applications, such as home assistants, booking systems, or recommendations. Accordingly, it is crucial to develop accurate Spoken Language Understanding (SLU) approaches to ensure the reliability of the considered system. Current State-of-the-Art SLU techniques rely on large amounts of training data; however, only limited annotated examples are available for specific tasks or languages.
In the meantime, instruction-tuned large language models (LLMs) have shown exceptional performance on unseen tasks in a few-shot setting when provided with adequate prompts. In this work, we propose to explore example selection by leveraging Information retrieval (IR) approaches to build an enhanced prompt that is applied to an SLU task. We evaluate the effectiveness of the proposed method on several SLU benchmarks. Experimental results show that lexical IR methods significantly enhance performance without increasing prompt length. 

---
# Coding Agents with Multimodal Browsing are Generalist Problem Solvers 

**Authors**: Aditya Bharat Soni, Boxuan Li, Xingyao Wang, Valerie Chen, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2506.03011)  

**Abstract**: Modern human labor is characterized by specialization; we train for years and develop particular tools that allow us to perform well across a variety of tasks. In addition, AI agents have been specialized for domains such as software engineering, web navigation, and workflow automation. However, this results in agents that are good for one thing but fail to generalize beyond their intended scope. One reason for this is that agent developers provide a highly specialized set of tools or make architectural decisions optimized for a specific use case or benchmark. In this work, we ask the question: what is the minimal set of general tools that can be used to achieve high performance across a diverse set of tasks? Our answer is OpenHands-Versa, a generalist agent built with a modest number of general tools: code editing and execution, web search, as well as multimodal web browsing and file access. Importantly, OpenHands-Versa demonstrates superior or competitive performance over leading specialized agents across three diverse and challenging benchmarks: SWE-Bench Multimodal, GAIA, and The Agent Company, outperforming the best-performing previously published results with absolute improvements in success rate of 9.1, 1.3, and 9.1 points respectively. Further, we show how existing state-of-the-art multi-agent systems fail to generalize beyond their target domains. These results demonstrate the feasibility of developing a generalist agent to solve diverse tasks and establish OpenHands-Versa as a strong baseline for future research. 

---
# Conditioning Large Language Models on Legal Systems? Detecting Punishable Hate Speech 

**Authors**: Florian Ludwig, Torsten Zesch, Frederike Zufall  

**Link**: [PDF](https://arxiv.org/pdf/2506.03009)  

**Abstract**: The assessment of legal problems requires the consideration of a specific legal system and its levels of abstraction, from constitutional law to statutory law to case law. The extent to which Large Language Models (LLMs) internalize such legal systems is unknown. In this paper, we propose and investigate different approaches to condition LLMs at different levels of abstraction in legal systems. This paper examines different approaches to conditioning LLMs at multiple levels of abstraction in legal systems to detect potentially punishable hate speech. We focus on the task of classifying whether a specific social media posts falls under the criminal offense of incitement to hatred as prescribed by the German Criminal Code. The results show that there is still a significant performance gap between models and legal experts in the legal assessment of hate speech, regardless of the level of abstraction with which the models were conditioned. Our analysis revealed, that models conditioned on abstract legal knowledge lacked deep task understanding, often contradicting themselves and hallucinating answers, while models using concrete legal knowledge performed reasonably well in identifying relevant target groups, but struggled with classifying target conducts. 

---
# A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems 

**Authors**: Đorđe Klisura, Astrid R Bernaga Torres, Anna Karen Gárate-Escamilla, Rajesh Roshan Biswal, Ke Yang, Hilal Pataci, Anthony Rios  

**Link**: [PDF](https://arxiv.org/pdf/2506.02998)  

**Abstract**: Privacy policies inform users about data collection and usage, yet their complexity limits accessibility for diverse populations. Existing Privacy Policy Question Answering (QA) systems exhibit performance disparities across English dialects, disadvantaging speakers of non-standard varieties. We propose a novel multi-agent framework inspired by human-centered design principles to mitigate dialectal biases. Our approach integrates a Dialect Agent, which translates queries into Standard American English (SAE) while preserving dialectal intent, and a Privacy Policy Agent, which refines predictions using domain expertise. Unlike prior approaches, our method does not require retraining or dialect-specific fine-tuning, making it broadly applicable across models and domains. Evaluated on PrivacyQA and PolicyQA, our framework improves GPT-4o-mini's zero-shot accuracy from 0.394 to 0.601 on PrivacyQA and from 0.352 to 0.464 on PolicyQA, surpassing or matching few-shot baselines without additional training data. These results highlight the effectiveness of structured agent collaboration in mitigating dialect biases and underscore the importance of designing NLP systems that account for linguistic diversity to ensure equitable access to privacy information. 

---
# It's Not a Walk in the Park! Challenges of Idiom Translation in Speech-to-text Systems 

**Authors**: Iuliia Zaitova, Badr M. Abdullah, Wei Xue, Dietrich Klakow, Bernd Möbius, Tania Avgustinova  

**Link**: [PDF](https://arxiv.org/pdf/2506.02995)  

**Abstract**: Idioms are defined as a group of words with a figurative meaning not deducible from their individual components. Although modern machine translation systems have made remarkable progress, translating idioms remains a major challenge, especially for speech-to-text systems, where research on this topic is notably sparse. In this paper, we systematically evaluate idiom translation as compared to conventional news translation in both text-to-text machine translation (MT) and speech-to-text translation (SLT) systems across two language pairs (German to English, Russian to English). We compare state-of-the-art end-to-end SLT systems (SeamlessM4T SLT-to-text, Whisper Large v3) with MT systems (SeamlessM4T SLT-to-text, No Language Left Behind), Large Language Models (DeepSeek, LLaMA) and cascaded alternatives. Our results reveal that SLT systems experience a pronounced performance drop on idiomatic data, often reverting to literal translations even in higher layers, whereas MT systems and Large Language Models demonstrate better handling of idioms. These findings underscore the need for idiom-specific strategies and improved internal representations in SLT architectures. 

---
# Performance of leading large language models in May 2025 in Membership of the Royal College of General Practitioners-style examination questions: a cross-sectional analysis 

**Authors**: Richard Armitage  

**Link**: [PDF](https://arxiv.org/pdf/2506.02987)  

**Abstract**: Background: Large language models (LLMs) have demonstrated substantial potential to support clinical practice. Other than Chat GPT4 and its predecessors, few LLMs, especially those of the leading and more powerful reasoning model class, have been subjected to medical specialty examination questions, including in the domain of primary care. This paper aimed to test the capabilities of leading LLMs as of May 2025 (o3, Claude Opus 4, Grok3, and Gemini 2.5 Pro) in primary care education, specifically in answering Member of the Royal College of General Practitioners (MRCGP) style examination questions.
Methods: o3, Claude Opus 4, Grok3, and Gemini 2.5 Pro were tasked to answer 100 randomly chosen multiple choice questions from the Royal College of General Practitioners GP SelfTest on 25 May 2025. Questions included textual information, laboratory results, and clinical images. Each model was prompted to answer as a GP in the UK and was provided with full question information. Each question was attempted once by each model. Responses were scored against correct answers provided by GP SelfTest.
Results: The total score of o3, Claude Opus 4, Grok3, and Gemini 2.5 Pro was 99.0%, 95.0%, 95.0%, and 95.0%, respectively. The average peer score for the same questions was 73.0%.
Discussion: All models performed remarkably well, and all substantially exceeded the average performance of GPs and GP registrars who had answered the same questions. o3 demonstrated the best performance, while the performances of the other leading models were comparable with each other and were not substantially lower than that of o3. These findings strengthen the case for LLMs, particularly reasoning models, to support the delivery of primary care, especially those that have been specifically trained on primary care clinical data. 

---
# Towards a Japanese Full-duplex Spoken Dialogue System 

**Authors**: Atsumoto Ohashi, Shinya Iizuka, Jingjing Jiang, Ryuichiro Higashinaka  

**Link**: [PDF](https://arxiv.org/pdf/2506.02979)  

**Abstract**: Full-duplex spoken dialogue systems, which can model simultaneous bidirectional features of human conversations such as speech overlaps and backchannels, have attracted significant attention recently. However, the study of full-duplex spoken dialogue systems for the Japanese language has been limited, and the research on their development in Japanese remains scarce. In this paper, we present the first publicly available full-duplex spoken dialogue model in Japanese, which is built upon Moshi, a full-duplex dialogue model in English. Our model is trained through a two-stage process: pre-training on a large-scale spoken dialogue data in Japanese, followed by fine-tuning on high-quality stereo spoken dialogue data. We further enhance the model's performance by incorporating synthetic dialogue data generated by a multi-stream text-to-speech system. Evaluation experiments demonstrate that the trained model outperforms Japanese baseline models in both naturalness and meaningfulness. 

---
# Expanding before Inferring: Enhancing Factuality in Large Language Models through Premature Layers Interpolation 

**Authors**: Dingwei Chen, Ziqiang Liu, Feiteng Fang, Chak Tou Leong, Shiwen Ni, Ahmadreza Argha, Hamid Alinejad-Rokny, Min Yang, Chengming Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.02973)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities in text understanding and generation. However, their tendency to produce factually inconsistent outputs, commonly referred to as ''hallucinations'', remains a critical challenge. Existing approaches, such as retrieval-based and inference-time correction methods, primarily address this issue at the input or output level, often overlooking the intrinsic information refinement process and the role of premature layers. Meanwhile, alignment- and fine-tuning-based methods are resource-intensive. In this paper, we propose PLI (Premature Layers Interpolation), a novel, training-free, and plug-and-play intervention designed to enhance factuality. PLI mitigates hallucinations by inserting premature layers formed through mathematical interpolation with adjacent layers. Inspired by stable diffusion and sampling steps, PLI extends the depth of information processing and transmission in LLMs, improving factual coherence. Experiments on four publicly available datasets demonstrate that PLI effectively reduces hallucinations while outperforming existing baselines in most cases. Further analysis suggests that the success of layer interpolation is closely linked to LLMs' internal mechanisms. To promote reproducibility, we will release our code and data upon acceptance. 

---
# FlowerTune: A Cross-Domain Benchmark for Federated Fine-Tuning of Large Language Models 

**Authors**: Yan Gao, Massimo Roberto Scamarcia, Javier Fernandez-Marques, Mohammad Naseri, Chong Shen Ng, Dimitris Stripelis, Zexi Li, Tao Shen, Jiamu Bai, Daoyuan Chen, Zikai Zhang, Rui Hu, InSeo Song, Lee KangYoon, Hong Jia, Ting Dang, Junyan Wang, Zheyuan Liu, Daniel Janes Beutel, Lingjuan Lyu, Nicholas D. Lane  

**Link**: [PDF](https://arxiv.org/pdf/2506.02961)  

**Abstract**: Large Language Models (LLMs) have achieved state-of-the-art results across diverse domains, yet their development remains reliant on vast amounts of publicly available data, raising concerns about data scarcity and the lack of access to domain-specific, sensitive information. Federated Learning (FL) presents a compelling framework to address these challenges by enabling decentralized fine-tuning on pre-trained LLMs without sharing raw data. However, the compatibility and performance of pre-trained LLMs in FL settings remain largely under explored. We introduce the FlowerTune LLM Leaderboard, a first-of-its-kind benchmarking suite designed to evaluate federated fine-tuning of LLMs across four diverse domains: general NLP, finance, medical, and coding. Each domain includes federated instruction-tuning datasets and domain-specific evaluation metrics. Our results, obtained through a collaborative, open-source and community-driven approach, provide the first comprehensive comparison across 26 pre-trained LLMs with different aggregation and fine-tuning strategies under federated settings, offering actionable insights into model performance, resource constraints, and domain adaptation. This work lays the foundation for developing privacy-preserving, domain-specialized LLMs for real-world applications. 

---
# HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring 

**Authors**: Zhixiong Su, Yichen Wang, Herun Wan, Zhaohan Zhang, Minnan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.02959)  

**Abstract**: The misuse of large language models (LLMs) poses potential risks, motivating the development of machine-generated text (MGT) detection. Existing literature primarily concentrates on binary, document-level detection, thereby neglecting texts that are composed jointly by human and LLM contributions. Hence, this paper explores the possibility of fine-grained MGT detection under human-AI coauthoring. We suggest fine-grained detectors can pave pathways toward coauthored text detection with a numeric AI ratio. Specifically, we propose a dataset, HACo-Det, which produces human-AI coauthored texts via an automatic pipeline with word-level attribution labels. We retrofit seven prevailing document-level detectors to generalize them to word-level detection. Then we evaluate these detectors on HACo-Det on both word- and sentence-level detection tasks. Empirical results show that metric-based methods struggle to conduct fine-grained detection with a 0.462 average F1 score, while finetuned models show superior performance and better generalization across domains. However, we argue that fine-grained co-authored text detection is far from solved. We further analyze factors influencing performance, e.g., context window, and highlight the limitations of current methods, pointing to potential avenues for improvement. 

---
# Adaptive Graph Pruning for Multi-Agent Communication 

**Authors**: Boyi Li, Zhonghan Zhao, Der-Horng Lee, Gaoang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02951)  

**Abstract**: Large Language Model (LLM) based multi-agent systems have shown remarkable performance in various tasks, especially when enhanced through collaborative communication. However, current methods often rely on a fixed number of agents and static communication structures, limiting their ability to adapt to varying task complexities. In this paper, we propose Adaptive Graph Pruning (AGP), a novel task-adaptive multi-agent collaboration framework that jointly optimizes agent quantity (hard-pruning) and communication topology (soft-pruning). Specifically, our method employs a two-stage training strategy: firstly, independently training soft-pruning networks for different agent quantities to determine optimal agent-quantity-specific complete graphs and positional masks across specific tasks; and then jointly optimizing hard-pruning and soft-pruning within a maximum complete graph to dynamically configure the number of agents and their communication topologies per task. Extensive experiments demonstrate that our approach is: (1) High-performing, achieving state-of-the-art results across six benchmarks and consistently generalizes across multiple mainstream LLM architectures, with a increase in performance of $2.58\%\sim 9.84\%$; (2) Task-adaptive, dynamically constructing optimized communication topologies tailored to specific tasks, with an extremely high performance in all three task categories (general reasoning, mathematical reasoning, and code generation); (3) Token-economical, having fewer training steps and token consumption at the same time, with a decrease in token consumption of $90\%+$; and (4) Training-efficient, achieving high performance with very few training steps compared with other methods. The performance will surpass the existing baselines after about ten steps of training under six benchmarks. 

---
# Quantitative LLM Judges 

**Authors**: Aishwarya Sahoo, Jeevana Kruthi Karnuthala, Tushar Parmanand Budhwani, Pranchal Agarwal, Sankaran Vaidyanathan, Alexa Siu, Franck Dernoncourt, Jennifer Healey, Nedim Lipka, Ryan Rossi, Uttaran Bhattacharya, Branislav Kveton  

**Link**: [PDF](https://arxiv.org/pdf/2506.02945)  

**Abstract**: LLM-as-a-judge is a framework in which a large language model (LLM) automatically evaluates the output of another LLM. We propose quantitative LLM judges, which align evaluation scores of existing LLM judges to human scores in a given domain using regression models. The models are trained to improve the score of the original judge by using the judge's textual evaluation and score. We present four quantitative judges for different types of absolute and relative feedback, which showcases the generality and versatility of our framework. Our framework is more computationally efficient than supervised fine-tuning and can be more statistically efficient when human feedback is limited, which is expected in most applications of our work. We validate these claims empirically on four datasets using two base judges. Our experiments show that quantitative judges can effectively improve the predictive power of existing judges through post-hoc modeling. 

---
# INESC-ID @ eRisk 2025: Exploring Fine-Tuned, Similarity-Based, and Prompt-Based Approaches to Depression Symptom Identification 

**Authors**: Diogo A.P. Nunes, Eugénio Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2506.02924)  

**Abstract**: In this work, we describe our team's approach to eRisk's 2025 Task 1: Search for Symptoms of Depression. Given a set of sentences and the Beck's Depression Inventory - II (BDI) questionnaire, participants were tasked with submitting up to 1,000 sentences per depression symptom in the BDI, sorted by relevance. Participant submissions were evaluated according to standard Information Retrieval (IR) metrics, including Average Precision (AP) and R-Precision (R-PREC). The provided training data, however, consisted of sentences labeled as to whether a given sentence was relevant or not w.r.t. one of BDI's symptoms. Due to this labeling limitation, we framed our development as a binary classification task for each BDI symptom, and evaluated accordingly. To that end, we split the available labeled data into training and validation sets, and explored foundation model fine-tuning, sentence similarity, Large Language Model (LLM) prompting, and ensemble techniques. The validation results revealed that fine-tuning foundation models yielded the best performance, particularly when enhanced with synthetic data to mitigate class imbalance. We also observed that the optimal approach varied by symptom. Based on these insights, we devised five independent test runs, two of which used ensemble methods. These runs achieved the highest scores in the official IR evaluation, outperforming submissions from 16 other teams. 

---
# A Controllable Examination for Long-Context Language Models 

**Authors**: Yijun Yang, Zeyu Huang, Wenhao Zhu, Zihan Qiu, Fei Yuan, Jeff Z.Pan, Ivan Titov  

**Link**: [PDF](https://arxiv.org/pdf/2506.02921)  

**Abstract**: Existing frameworks for evaluating long-context language models (LCLM) can be broadly categorized into real-world and synthetic tasks. Despite their utility, both approaches are accompanied by certain intrinsic limitations. Real-world tasks are too complex to interpret or characterize and are susceptible to data contamination. In contrast, synthetic tasks often adopt the needle-in-the-haystack (NIAH) format, wherein a lack of coherence between the "needle" and the "haystack" compromises their validity as proxies for realistic applications. In response to these challenges, we posit that an ideal long-context evaluation framework should be characterized by three essential features: $\textit{seamless context}$, $\textit{controllable setting}$, and $\textit{sound evaluation}$. This study introduces $\textbf{LongBioBench}$, a novel benchmark that utilizes artificially generated biographies as a controlled environment for assessing LCLMs across dimensions of $\textit{understanding}$, $\textit{reasoning}$, and $\textit{trustworthiness}$. Our experimental evaluation, which includes $\textbf{18}$ LCLMs in total, demonstrates that most models still exhibit deficiencies in semantic understanding and elementary reasoning over retrieved results and are less trustworthy as context length increases. Our further analysis indicates some design choices employed by existing synthetic benchmarks, such as contextual non-coherence, numerical needles, and the absence of distractors, rendering them vulnerable to test the model long-context capabilities. Moreover, we also reveal that long-context continual pretraining primarily adjusts RoPE embedding to accommodate extended context lengths. To sum up, compared to previous synthetic benchmarks, LongBioBench achieves a better trade-off between mirroring authentic language tasks and maintaining controllability, and is highly interpretable and configurable. 

---
# Cell-o1: Training LLMs to Solve Single-Cell Reasoning Puzzles with Reinforcement Learning 

**Authors**: Yin Fang, Qiao Jin, Guangzhi Xiong, Bowen Jin, Xianrui Zhong, Siru Ouyang, Aidong Zhang, Jiawei Han, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02911)  

**Abstract**: Cell type annotation is a key task in analyzing the heterogeneity of single-cell RNA sequencing data. Although recent foundation models automate this process, they typically annotate cells independently, without considering batch-level cellular context or providing explanatory reasoning. In contrast, human experts often annotate distinct cell types for different cell clusters based on their domain knowledge. To mimic this workflow, we introduce the CellPuzzles task, where the objective is to assign unique cell types to a batch of cells. This benchmark spans diverse tissues, diseases, and donor conditions, and requires reasoning across the batch-level cellular context to ensure label uniqueness. We find that off-the-shelf large language models (LLMs) struggle on CellPuzzles, with the best baseline (OpenAI's o1) achieving only 19.0% batch-level accuracy. To fill this gap, we propose Cell-o1, a 7B LLM trained via supervised fine-tuning on distilled reasoning traces, followed by reinforcement learning with batch-level rewards. Cell-o1 achieves state-of-the-art performance, outperforming o1 by over 73% and generalizing well across contexts. Further analysis of training dynamics and reasoning behaviors provides insights into batch-level annotation performance and emergent expert-like reasoning. Code and data are available at this https URL. 

---
# IMPARA-GED: Grammatical Error Detection is Boosting Reference-free Grammatical Error Quality Estimator 

**Authors**: Yusuke Sakai, Takumi Goto, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.02899)  

**Abstract**: We propose IMPARA-GED, a novel reference-free automatic grammatical error correction (GEC) evaluation method with grammatical error detection (GED) capabilities. We focus on the quality estimator of IMPARA, an existing automatic GEC evaluation method, and construct that of IMPARA-GED using a pre-trained language model with enhanced GED capabilities. Experimental results on SEEDA, a meta-evaluation dataset for automatic GEC evaluation methods, demonstrate that IMPARA-GED achieves the highest correlation with human sentence-level evaluations. 

---
# A Multi-Dialectal Dataset for German Dialect ASR and Dialect-to-Standard Speech Translation 

**Authors**: Verena Blaschke, Miriam Winkler, Constantin Förster, Gabriele Wenger-Glemser, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2506.02894)  

**Abstract**: Although Germany has a diverse landscape of dialects, they are underrepresented in current automatic speech recognition (ASR) research. To enable studies of how robust models are towards dialectal variation, we present Betthupferl, an evaluation dataset containing four hours of read speech in three dialect groups spoken in Southeast Germany (Franconian, Bavarian, Alemannic), and half an hour of Standard German speech. We provide both dialectal and Standard German transcriptions, and analyze the linguistic differences between them. We benchmark several multilingual state-of-the-art ASR models on speech translation into Standard German, and find differences between how much the output resembles the dialectal vs. standardized transcriptions. Qualitative error analyses of the best ASR model reveal that it sometimes normalizes grammatical differences, but often stays closer to the dialectal constructions. 

---
# CoT is Not True Reasoning, It Is Just a Tight Constraint to Imitate: A Theory Perspective 

**Authors**: Jintian Shao, Yiming Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.02878)  

**Abstract**: Chain-of-Thought (CoT) prompting has demonstrably enhanced the performance of Large Language Models on tasks requiring multi-step inference. This success has led to widespread claims of emergent reasoning capabilities in these models. In this paper, we present a theoretical counter-perspective: Chain-of-Thought (CoT) does not elicit genuine, abstract reasoning. Instead, we argue that Chain-of-Thought functions as a powerful structural constraint that guides Large Language Models to imitate the form of reasoning. By forcing the generation of intermediate steps, Chain-of-Thought leverages the model immense capacity for sequence prediction and pattern matching, effectively constraining its output to sequences that resemble coherent thought processes. Chain-of-Thought (CoT) prompting has demonstrably enhanced the performance of Large Language Models on tasks requiring multi-step inference. This success has led to widespread claims of emergent reasoning capabilities in these models. In this paper, we present a theoretical counter-perspective: Chain-of-Thought (CoT) does not elicit genuine, abstract reasoning. Instead, we argue that Chain-of-Thought functions as a powerful structural constraint that guides Large Language Models to imitate the form of reasoning. By forcing the generation of intermediate steps, Chain-of-Thought leverages the model immense capacity for sequence prediction and pattern matching, effectively constraining its output to sequences that resemble coherent thought processes. 

---
# Token and Span Classification for Entity Recognition in French Historical Encyclopedias 

**Authors**: Ludovic Moncla, Hédi Zeghidi  

**Link**: [PDF](https://arxiv.org/pdf/2506.02872)  

**Abstract**: Named Entity Recognition (NER) in historical texts presents unique challenges due to non-standardized language, archaic orthography, and nested or overlapping entities. This study benchmarks a diverse set of NER approaches, ranging from classical Conditional Random Fields (CRFs) and spaCy-based models to transformer-based architectures such as CamemBERT and sequence-labeling models like Flair. Experiments are conducted on the GeoEDdA dataset, a richly annotated corpus derived from 18th-century French encyclopedias. We propose framing NER as both token-level and span-level classification to accommodate complex nested entity structures typical of historical documents. Additionally, we evaluate the emerging potential of few-shot prompting with generative language models for low-resource scenarios. Our results demonstrate that while transformer-based models achieve state-of-the-art performance, especially on nested entities, generative models offer promising alternatives when labeled data are scarce. The study highlights ongoing challenges in historical NER and suggests avenues for hybrid approaches combining symbolic and neural methods to better capture the intricacies of early modern French text. 

---
# TO-GATE: Clarifying Questions and Summarizing Responses with Trajectory Optimization for Eliciting Human Preference 

**Authors**: Yulin Dou, Jiangming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02827)  

**Abstract**: Large language models (LLMs) can effectively elicit human preferences through multi-turn dialogue. Complex tasks can be accomplished through iterative clarifying questions and final responses generated by an LLM acting as a questioner (STaR-GATE; Andukuri et al., 2024}). However, existing approaches based on self-taught reasoning struggle to identify optimal dialogue trajectories and avoid irrelevant questions to the tasks. To address this limitation, we propose TO-GATE, a novel framework that enhances question generation through trajectory optimization, which consists of two key components: a clarification resolver that generates optimal questioning trajectories, and a summarizer that ensures task-aligned final responses. The trajectory optimization enables the model to produce effective elicitation questions and summary responses tailored to specific tasks. Experimental results demonstrate that TO-GATE significantly outperforms baseline methods, achieving a 9.32% improvement on standard preference elicitation tasks. 

---
# ProcrustesGPT: Compressing LLMs with Structured Matrices and Orthogonal Transformations 

**Authors**: Ekaterina Grishina, Mikhail Gorbunov, Maxim Rakhuba  

**Link**: [PDF](https://arxiv.org/pdf/2506.02818)  

**Abstract**: Large language models (LLMs) demonstrate impressive results in natural language processing tasks but require a significant amount of computational and memory resources. Structured matrix representations are a promising way for reducing the number of parameters of these models. However, it seems unrealistic to expect that weight matrices of pretrained models can be accurately represented by structured matrices without any fine-tuning. To overcome this issue, we utilize the fact that LLM output is invariant under certain orthogonal transformations of weight matrices. This insight can be leveraged to identify transformations that significantly improve the compressibility of weights within structured classes. The proposed approach is applicable to various types of structured matrices that support efficient projection operations. Code is available at this https URL 

---
# SemVink: Advancing VLMs' Semantic Understanding of Optical Illusions via Visual Global Thinking 

**Authors**: Sifan Li, Yujun Cai, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02803)  

**Abstract**: Vision-language models (VLMs) excel in semantic tasks but falter at a core human capability: detecting hidden content in optical illusions or AI-generated images through perceptual adjustments like zooming. We introduce HC-Bench, a benchmark of 112 images with hidden text, objects, and illusions, revealing that leading VLMs achieve near-zero accuracy (0-5.36%)-even with explicit prompting. Humans resolve such ambiguities instinctively, yet VLMs fail due to an overreliance on high-level semantics. Strikingly, we propose SemVink (Semantic Visual Thinking) by simply scaling images to low resolutions (32-128 pixels), which unlocks >99% accuracy by eliminating redundant visual noise. This exposes a critical architectural flaw: VLMs prioritize abstract reasoning over low-level visual operations crucial for real-world robustness. Our work urges a shift toward hybrid models integrating multi-scale processing, bridging the gap between computational vision and human cognition for applications in medical imaging, security, and beyond. 

---
# Exploiting the English Vocabulary Profile for L2 word-level vocabulary assessment with LLMs 

**Authors**: Stefano Bannò, Kate Knill, Mark Gales  

**Link**: [PDF](https://arxiv.org/pdf/2506.02758)  

**Abstract**: Vocabulary use is a fundamental aspect of second language (L2) proficiency. To date, its assessment by automated systems has typically examined the context-independent, or part-of-speech (PoS) related use of words. This paper introduces a novel approach to enable fine-grained vocabulary evaluation exploiting the precise use of words within a sentence. The scheme combines large language models (LLMs) with the English Vocabulary Profile (EVP). The EVP is a standard lexical resource that enables in-context vocabulary use to be linked with proficiency level. We evaluate the ability of LLMs to assign proficiency levels to individual words as they appear in L2 learner writing, addressing key challenges such as polysemy, contextual variation, and multi-word expressions. We compare LLMs to a PoS-based baseline. LLMs appear to exploit additional semantic information that yields improved performance. We also explore correlations between word-level proficiency and essay-level proficiency. Finally, the approach is applied to examine the consistency of the EVP proficiency levels. Results show that LLMs are well-suited for the task of vocabulary assessment. 

---
# Multi-task Learning with Active Learning for Arabic Offensive Speech Detection 

**Authors**: Aisha Alansari, Hamzah Luqman  

**Link**: [PDF](https://arxiv.org/pdf/2506.02753)  

**Abstract**: The rapid growth of social media has amplified the spread of offensive, violent, and vulgar speech, which poses serious societal and cybersecurity concerns. Detecting such content in Arabic text is particularly complex due to limited labeled data, dialectal variations, and the language's inherent complexity. This paper proposes a novel framework that integrates multi-task learning (MTL) with active learning to enhance offensive speech detection in Arabic social media text. By jointly training on two auxiliary tasks, violent and vulgar speech, the model leverages shared representations to improve the detection accuracy of the offensive speech. Our approach dynamically adjusts task weights during training to balance the contribution of each task and optimize performance. To address the scarcity of labeled data, we employ an active learning strategy through several uncertainty sampling techniques to iteratively select the most informative samples for model training. We also introduce weighted emoji handling to better capture semantic cues. Experimental results on the OSACT2022 dataset show that the proposed framework achieves a state-of-the-art macro F1-score of 85.42%, outperforming existing methods while using significantly fewer fine-tuning samples. The findings of this study highlight the potential of integrating MTL with active learning for efficient and accurate offensive language detection in resource-constrained settings. 

---
# Stereotypical gender actions can be extracted from Web text 

**Authors**: Amaç Herdağdelen, Marco Baroni  

**Link**: [PDF](https://arxiv.org/pdf/2506.02740)  

**Abstract**: We extracted gender-specific actions from text corpora and Twitter, and compared them to stereotypical expectations of people. We used Open Mind Common Sense (OMCS), a commonsense knowledge repository, to focus on actions that are pertinent to common sense and daily life of humans. We use the gender information of Twitter users and Web-corpus-based pronoun/name gender heuristics to compute the gender bias of the actions. With high recall, we obtained a Spearman correlation of 0.47 between corpus-based predictions and a human gold standard, and an area under the ROC curve of 0.76 when predicting the polarity of the gold standard. We conclude that it is feasible to use natural text (and a Twitter-derived corpus in particular) in order to augment commonsense repositories with the stereotypical gender expectations of actions. We also present a dataset of 441 commonsense actions with human judges' ratings on whether the action is typically/slightly masculine/feminine (or neutral), and another larger dataset of 21,442 actions automatically rated by the methods we investigate in this study. 

---
# RACE-Align: Retrieval-Augmented and Chain-of-Thought Enhanced Preference Alignment for Large Language Models 

**Authors**: Qihang Yan, Xinyu Zhang, Luming Guo, Qi Zhang, Feifan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02726)  

**Abstract**: Large Language Models (LLMs) struggle with accuracy, domain-specific reasoning, and interpretability in vertical domains. Traditional preference alignment methods like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) often overlook the underlying knowledge sources and reasoning logic. This paper introduces RACE-Align (Retrieval-Augmented and Chain-of-Thought Enhanced Alignment), a novel framework designed to address these limitations. RACE-Align systematically constructs a binary preference dataset incorporating external knowledge support and explicit Chain-of-Thought (CoT) reasoning, then aligns LLMs using the DPO algorithm. The core innovation lies in its preference data construction strategy: it integrates AI-driven retrieval for factual grounding, enhancing knowledgeability and accuracy, and emphasizes the optimization of domain-specific CoT, treating the reasoning process itself as a key preference dimension. A multi-stage, AI-driven refinement pipeline cost-effectively generates these preference pairs. Experimental validation in Traditional Chinese Medicine (TCM) using Qwen3-1.7B as the base model demonstrates that RACE-Align significantly outperforms the original base model and a model fine-tuned only with Supervised Fine-Tuning (SFT). Improvements were observed across multiple dimensions, including answer accuracy, information richness, application of TCM thinking patterns, logicality and depth of reasoning, and interpretability. These findings suggest RACE-Align offers an effective pathway to enhance LLMs' knowledge application, reasoning reliability, and process transparency in complex vertical domains. 

---
# On Entity Identification in Language Models 

**Authors**: Masaki Sakata, Sho Yokoi, Benjamin Heinzerling, Takumi Ito, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2506.02701)  

**Abstract**: We analyze the extent to which internal representations of language models (LMs) identify and distinguish mentions of named entities, focusing on the many-to-many correspondence between entities and their mentions. We first formulate two problems of entity mentions -- ambiguity and variability -- and propose a framework analogous to clustering quality metrics. Specifically, we quantify through cluster analysis of LM internal representations the extent to which mentions of the same entity cluster together and mentions of different entities remain separated. Our experiments examine five Transformer-based autoregressive models, showing that they effectively identify and distinguish entities with metrics analogous to precision and recall ranging from 0.66 to 0.9. Further analysis reveals that entity-related information is compactly represented in a low-dimensional linear subspace at early LM layers. Additionally, we clarify how the characteristics of entity representations influence word prediction performance. These findings are interpreted through the lens of isomorphism between LM representations and entity-centric knowledge structures in the real world, providing insights into how LMs internally organize and use entity information. 

---
# MASTER: Enhancing Large Language Model via Multi-Agent Simulated Teaching 

**Authors**: Liang Yue, Yihong Tang, Kehai Chen, Jie Liu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02689)  

**Abstract**: Instruction fine-tuning is crucial in NLP tasks, enhancing pretrained models' instruction-following capabilities and task-specific performance. However, obtaining high-quality fine-tuning data for large models is challenging due to data collection difficulties and high production costs. To address this, we propose MASTER, a novel data augmentation method that enriches original data through interactions among multiple agents with varying cognitive levels. We simulate three pedagogically grounded teaching scenarios, leveraging multi-agent conversations to generate high-quality teacher-student interaction data. Utilizing MASTER, we construct BOOST-QA, a fine-tuning dataset augmented from existing datasets like Orca-Math-200k, ProcQA, and OpenHermes2.5. Experiments show that models fine-tuned with BOOST-QA perform excellently across multiple benchmarks, demonstrating strong multitask generalization. Notably, MASTER significantly improves models' reasoning abilities in complex tasks, providing valuable insights for future research. 

---
# Decompose, Plan in Parallel, and Merge: A Novel Paradigm for Large Language Models based Planning with Multiple Constraints 

**Authors**: Zhengdong Lu, Weikai Lu, Yiling Tao, Yun Dai, ZiXuan Chen, Huiping Zhuang, Cen Chen, Hao Peng, Ziqian Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.02683)  

**Abstract**: Despite significant advances in Large Language Models (LLMs), planning tasks still present challenges for LLM-based agents. Existing planning methods face two key limitations: heavy constraints and cascading errors. To address these limitations, we propose a novel parallel planning paradigm, which Decomposes, Plans for subtasks in Parallel, and Merges subplans into a final plan (DPPM). Specifically, DPPM decomposes the complex task based on constraints into subtasks, generates the subplan for each subtask in parallel, and merges them into a global plan. In addition, our approach incorporates a verification and refinement module, enabling error correction and conflict resolution. Experimental results demonstrate that DPPM significantly outperforms existing methods in travel planning tasks. 

---
# TL;DR: Too Long, Do Re-weighting for Effcient LLM Reasoning Compression 

**Authors**: Zhong-Zhi Li, Xiao Liang, Zihao Tang, Lei Ji, Peijie Wang, Haotian Xu, Xing W, Haizhen Huang, Weiwei Deng, Ying Nian Wu, Yeyun Gong, Zhijiang Guo, Xiao Liu, Fei Yin, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02678)  

**Abstract**: Large Language Models (LLMs) have recently achieved remarkable progress by leveraging Reinforcement Learning and extended Chain-of-Thought (CoT) techniques. However, the challenge of performing efficient language reasoning--especially during inference with extremely long outputs--has drawn increasing attention from the research community. In this work, we propose a dynamic ratio-based training pipeline that does not rely on sophisticated data annotations or interpolation between multiple models. We continuously balance the weights between the model's System-1 and System-2 data to eliminate redundant reasoning processes while preserving the model's reasoning capability. We validate our approach across models on DeepSeek-R1-Distill-7B and DeepSeek-R1-Distill-14B and on a diverse set of benchmarks with varying difficulty levels. Our method significantly reduces the number of output tokens by nearly 40% while maintaining the accuracy of the reasoning. Our code and data will be available soon. 

---
# EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving 

**Authors**: Shihan Dou, Ming Zhang, Chenhao Huang, Jiayi Chen, Feng Chen, Shichun Liu, Yan Liu, Chenxiao Liu, Cheng Zhong, Zongzhang Zhang, Tao Gui, Chao Xin, Wei Chengzhi, Lin Yan, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02672)  

**Abstract**: We introduce EvaLearn, a pioneering benchmark designed to evaluate large language models (LLMs) on their learning capability and efficiency in challenging tasks, a critical, yet underexplored aspect of model potential. EvaLearn contains 648 challenging problems across six task types, grouped into 182 sequences, each sequence dedicated to one task type. Diverging from most existing benchmarks that evaluate models in parallel, EvaLearn requires models to solve problems sequentially, allowing them to leverage the experience gained from previous solutions. EvaLearn provides five comprehensive automated metrics to evaluate models and quantify their learning capability and efficiency. We extensively benchmark nine frontier models and observe varied performance profiles: some models, such as Claude-3.7-sonnet, start with moderate initial performance but exhibit strong learning ability, while some models struggle to benefit from experience and may even show negative transfer. Moreover, we investigate model performance under two learning settings and find that instance-level rubrics and teacher-model feedback further facilitate model learning. Importantly, we observe that current LLMs with stronger static abilities do not show a clear advantage in learning capability across all tasks, highlighting that EvaLearn evaluates a new dimension of model performance. We hope EvaLearn provides a novel evaluation perspective for assessing LLM potential and understanding the gap between models and human capabilities, promoting the development of deeper and more dynamic evaluation approaches. All datasets, the automatic evaluation framework, and the results studied in this paper are available at the GitHub repository. 

---
# Are Economists Always More Introverted? Analyzing Consistency in Persona-Assigned LLMs 

**Authors**: Manon Reusens, Bart Baesens, David Jurgens  

**Link**: [PDF](https://arxiv.org/pdf/2506.02659)  

**Abstract**: Personalized Large Language Models (LLMs) are increasingly used in diverse applications, where they are assigned a specific persona - such as a happy high school teacher - to guide their responses. While prior research has examined how well LLMs adhere to predefined personas in writing style, a comprehensive analysis of consistency across different personas and task types is lacking. In this paper, we introduce a new standardized framework to analyze consistency in persona-assigned LLMs. We define consistency as the extent to which a model maintains coherent responses when assigned the same persona across different tasks and runs. Our framework evaluates personas across four different categories (happiness, occupation, personality, and political stance) spanning multiple task dimensions (survey writing, essay generation, social media post generation, single turn, and multi-turn conversations). Our findings reveal that consistency is influenced by multiple factors, including the assigned persona, stereotypes, and model design choices. Consistency also varies across tasks, increasing with more structured tasks and additional context. All code is available on GitHub. 

---
# Overcoming Data Scarcity in Multi-Dialectal Arabic ASR via Whisper Fine-Tuning 

**Authors**: Ömer Tarik Özyilmaz, Matt Coler, Matias Valdenegro-Toro  

**Link**: [PDF](https://arxiv.org/pdf/2506.02627)  

**Abstract**: Although commercial Arabic automatic speech recognition (ASR) systems support Modern Standard Arabic (MSA), they struggle with dialectal speech. We investigate the effect of fine-tuning OpenAI's Whisper on five major Arabic dialects (Gulf, Levantine, Iraqi, Egyptian, Maghrebi) using Mozilla Common Voice for MSA and the MASC dataset for dialectal speech. We evaluate MSA training size effects, benefits of pre-training on MSA data, and dialect-specific versus dialect-pooled models. We find that small amounts of MSA fine-tuning data yield substantial improvements for smaller models, matching larger non-fine-tuned models. While MSA pre-training shows minimal benefit, suggesting limited shared features between MSA and dialects, our dialect-pooled models perform comparably to dialect-specific ones. This indicates that pooling dialectal data, when properly balanced, can help address data scarcity in low-resource ASR without significant performance loss. 

---
# EssayBench: Evaluating Large Language Models in Multi-Genre Chinese Essay Writing 

**Authors**: Fan Gao, Dongyuan Li, Ding Xia, Fei Mi, Yasheng Wang, Lifeng Shang, Baojun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02596)  

**Abstract**: Chinese essay writing and its evaluation are critical in educational contexts, yet the capabilities of Large Language Models (LLMs) in this domain remain largely underexplored. Existing benchmarks often rely on coarse-grained text quality metrics, largely overlooking the structural and rhetorical complexities of Chinese essays, particularly across diverse genres. To address this gap, we propose \benchName, a multi-genre benchmark specifically designed for Chinese essay writing across four major genres: Argumentative, Narrative, Descriptive, and Expository. We curate and refine a total of 728 real-world prompts to ensure authenticity and meticulously categorize them into the \textit{Open-Ended} and \textit{Constrained} sets to capture diverse writing scenarios. To reliably evaluate generated essays, we develop a fine-grained, genre-specific scoring framework that hierarchically aggregates scores. We further validate our evaluation protocol through a comprehensive human agreement study. Finally, we benchmark 15 large-sized LLMs, analyzing their strengths and limitations across genres and instruction types. With \benchName, we aim to advance LLM-based Chinese essay evaluation and inspire future research on improving essay generation in educational settings. 

---
# Beyond the Surface: Measuring Self-Preference in LLM Judgments 

**Authors**: Zhi-Yuan Chen, Hao Wang, Xinyu Zhang, Enrui Hu, Yankai Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.02592)  

**Abstract**: Recent studies show that large language models (LLMs) exhibit self-preference bias when serving as judges, meaning they tend to favor their own responses over those generated by other models. Existing methods typically measure this bias by calculating the difference between the scores a judge model assigns to its own responses and those it assigns to responses from other models. However, this approach conflates self-preference bias with response quality, as higher-quality responses from the judge model may also lead to positive score differences, even in the absence of bias. To address this issue, we introduce gold judgments as proxies for the actual quality of responses and propose the DBG score, which measures self-preference bias as the difference between the scores assigned by the judge model to its own responses and the corresponding gold judgments. Since gold judgments reflect true response quality, the DBG score mitigates the confounding effect of response quality on bias measurement. Using the DBG score, we conduct comprehensive experiments to assess self-preference bias across LLMs of varying versions, sizes, and reasoning abilities. Additionally, we investigate two factors that influence and help alleviate self-preference bias: response text style and the post-training data of judge models. Finally, we explore potential underlying mechanisms of self-preference bias from an attention-based perspective. Our code and data are available at this https URL. 

---
# On Generalization across Measurement Systems: LLMs Entail More Test-Time Compute for Underrepresented Cultures 

**Authors**: Minh Duc Bui, Kyung Eun Park, Goran Glavaš, Fabian David Schmidt, Katharina von der Wense  

**Link**: [PDF](https://arxiv.org/pdf/2506.02591)  

**Abstract**: Measurement systems (e.g., currencies) differ across cultures, but the conversions between them are well defined so that humans can state facts using any measurement system of their choice. Being available to users from diverse cultural backgrounds, large language models (LLMs) should also be able to provide accurate information irrespective of the measurement system at hand. Using newly compiled datasets we test if this is the case for seven open-source LLMs, addressing three key research questions: (RQ1) What is the default system used by LLMs for each type of measurement? (RQ2) Do LLMs' answers and their accuracy vary across different measurement systems? (RQ3) Can LLMs mitigate potential challenges w.r.t. underrepresented systems via reasoning? Our findings show that LLMs default to the measurement system predominantly used in the data. Additionally, we observe considerable instability and variance in performance across different measurement systems. While this instability can in part be mitigated by employing reasoning methods such as chain-of-thought (CoT), this implies longer responses and thereby significantly increases test-time compute (and inference costs), marginalizing users from cultural backgrounds that use underrepresented measurement systems. 

---
# Evaluating Named Entity Recognition Models for Russian Cultural News Texts: From BERT to LLM 

**Authors**: Maria Levchenko  

**Link**: [PDF](https://arxiv.org/pdf/2506.02589)  

**Abstract**: This paper addresses the challenge of Named Entity Recognition (NER) for person names within the specialized domain of Russian news texts concerning cultural events. The study utilizes the unique SPbLitGuide dataset, a collection of event announcements from Saint Petersburg spanning 1999 to 2019. A comparative evaluation of diverse NER models is presented, encompassing established transformer-based architectures such as DeepPavlov, RoBERTa, and SpaCy, alongside recent Large Language Models (LLMs) including GPT-3.5, GPT-4, and GPT-4o. Key findings highlight the superior performance of GPT-4o when provided with specific prompting for JSON output, achieving an F1 score of 0.93. Furthermore, GPT-4 demonstrated the highest precision at 0.99. The research contributes to a deeper understanding of current NER model capabilities and limitations when applied to morphologically rich languages like Russian within the cultural heritage domain, offering insights for researchers and practitioners. Follow-up evaluation with GPT-4.1 (April 2025) achieves F1=0.94 for both simple and structured prompts, demonstrating rapid progress across model families and simplified deployment requirements. 

---
# Prosodic Structure Beyond Lexical Content: A Study of Self-Supervised Learning 

**Authors**: Sarenne Wallbridge, Christoph Minixhofer, Catherine Lai, Peter Bell  

**Link**: [PDF](https://arxiv.org/pdf/2506.02584)  

**Abstract**: People exploit the predictability of lexical structures during text comprehension. Though predictable structure is also present in speech, the degree to which prosody, e.g. intonation, tempo, and loudness, contributes to such structure independently of the lexical content is unclear. This study leverages self-supervised learning (SSL) to examine the temporal granularity of structures in the acoustic correlates of prosody. Representations from our proposed Masked Prosody Model can predict perceptual labels dependent on local information, such as word boundaries, but provide the most value for labels involving longer-term structures, like emotion recognition. Probing experiments across various perceptual labels show strong relative gains over untransformed pitch, energy, and voice activity features. Our results reveal the importance of SSL training objective timescale and highlight the value of complex SSL-encoded structures compared to more constrained classical structures. 

---
# IndoSafety: Culturally Grounded Safety for LLMs in Indonesian Languages 

**Authors**: Muhammad Falensi Azmi, Muhammad Dehan Al Kautsar, Alfan Farizki Wicaksono, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2506.02573)  

**Abstract**: Although region-specific large language models (LLMs) are increasingly developed, their safety remains underexplored, particularly in culturally diverse settings like Indonesia, where sensitivity to local norms is essential and highly valued by the community. In this work, we present IndoSafety, the first high-quality, human-verified safety evaluation dataset tailored for the Indonesian context, covering five language varieties: formal and colloquial Indonesian, along with three major local languages: Javanese, Sundanese, and Minangkabau. IndoSafety is constructed by extending prior safety frameworks to develop a taxonomy that captures Indonesia's sociocultural context. We find that existing Indonesian-centric LLMs often generate unsafe outputs, particularly in colloquial and local language settings, while fine-tuning on IndoSafety significantly improves safety while preserving task performance. Our work highlights the critical need for culturally grounded safety evaluation and provides a concrete step toward responsible LLM deployment in multilingual settings. Warning: This paper contains example data that may be offensive, harmful, or biased. 

---
# Pruning General Large Language Models into Customized Expert Models 

**Authors**: Yirao Zhao, Guizhen Chen, Kenji Kawaguchi, Lidong Bing, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02561)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing, yet their substantial model sizes often require substantial computational resources. To preserve computing resources and accelerate inference speed, it is crucial to prune redundant parameters, especially for experienced users who often need compact expert models tailored to specific downstream scenarios. However, most existing pruning methods focus on preserving the model's general capabilities, often requiring extensive post-training or suffering from degraded performance due to coarse-grained pruning. In this work, we design a $\underline{Cus}$tom $\underline{Prun}$ing method ($\texttt{Cus-Prun}$) to prune a large general model into a smaller lightweight expert model, which is positioned along the "language", "domain" and "task" dimensions. By identifying and pruning irrelevant neurons of each dimension, $\texttt{Cus-Prun}$ creates expert models without any post-training. Our experiments demonstrate that $\texttt{Cus-Prun}$ consistently outperforms other methods, achieving minimal loss in both expert and general capabilities across various models from different model families and sizes. 

---
# CoRe-MMRAG: Cross-Source Knowledge Reconciliation for Multimodal RAG 

**Authors**: Yang Tian, Fan Liu, Jingyuan Zhang, Victoria W., Yupeng Hu, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.02544)  

**Abstract**: Multimodal Retrieval-Augmented Generation (MMRAG) has been introduced to enhance Multimodal Large Language Models by incorporating externally retrieved multimodal knowledge, but it introduces two challenges: Parametric-Retrieved Knowledge Inconsistency (PRKI), where discrepancies between parametric and retrieved knowledge create uncertainty in determining reliability, and Visual-Textual Knowledge Inconsistency (VTKI), where misalignment between visual and textual sources disrupts entity representation. To address these challenges, we propose \textbf{C}r\textbf{o}ss-source knowledge \textbf{Re}conciliation for \textbf{M}ulti\textbf{M}odal \textbf{RAG} (CoRe-MMRAG), a novel end-to-end framework that effectively reconciles inconsistencies across knowledge sources. CoRe-MMRAG follows a four-stage pipeline: it first generates an internal response from parametric knowledge, then selects the most relevant multimodal evidence via joint similarity assessment, generates an external response, and finally integrates both to produce a reliable answer. Additionally, a specialized training paradigm enhances knowledge source discrimination, multimodal integration, and unified answer generation. Experiments on KB-VQA benchmarks show that CoRe-MMRAG achieves substantial improvements over baseline methods, achieving 5.6\% and 9.3\% performance gains on InfoSeek and Encyclopedic-VQA, respectively. We release code and data at \href{this https URL}{this https URL}. 

---
# Answer Convergence as a Signal for Early Stopping in Reasoning 

**Authors**: Xin Liu, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02536)  

**Abstract**: Chain-of-thought (CoT) prompting enhances reasoning in large language models (LLMs) but often leads to verbose and redundant outputs, thus increasing inference cost. We hypothesize that many reasoning steps are unnecessary for producing correct answers. To investigate this, we start with a systematic study to examine what is the minimum reasoning required for a model to reach a stable decision. We find that on math reasoning tasks like math, models typically converge to their final answers after 60\% of the reasoning steps, suggesting substantial redundancy in the remaining content. Based on these insights, we propose three inference-time strategies to improve efficiency: (1) early stopping via answer consistency, (2) boosting the probability of generating end-of-reasoning signals, and (3) a supervised method that learns when to stop based on internal activations. Experiments across five benchmarks and five open-weights LLMs show that our methods significantly reduce token usage with little or no accuracy drop. In particular, on NaturalQuestions, Answer Consistency reduces tokens by over 40\% while further improving accuracy. Our work underscores the importance of cost-effective reasoning methods that operate at inference time, offering practical benefits for real-world applications. 

---
# Natural Language Processing to Enhance Deliberation in Political Online Discussions: A Survey 

**Authors**: Maike Behrendt, Stefan Sylvius Wagner, Carina Weinmann, Marike Bormann, Mira Warne, Stefan Harmeling  

**Link**: [PDF](https://arxiv.org/pdf/2506.02533)  

**Abstract**: Political online participation in the form of discussing political issues and exchanging opinions among citizens is gaining importance with more and more formats being held digitally. To come to a decision, a careful discussion and consideration of opinions and a civil exchange of arguments, which is defined as the act of deliberation, is desirable. The quality of discussions and participation processes in terms of their deliberativeness highly depends on the design of platforms and processes. To facilitate online communication for both participants and initiators, machine learning methods offer a lot of potential. In this work we want to showcase which issues occur in political online discussions and how machine learning can be used to counteract these issues and enhance deliberation. 

---
# ReasoningFlow: Semantic Structure of Complex Reasoning Traces 

**Authors**: Jinu Lee, Sagnik Mukherjee, Dilek Hakkani-Tur, Julia Hockenmaier  

**Link**: [PDF](https://arxiv.org/pdf/2506.02532)  

**Abstract**: Large reasoning models (LRMs) generate complex reasoning traces with planning, reflection, verification, and backtracking. In this work, we introduce ReasoningFlow, a unified schema for analyzing the semantic structures of these complex traces. ReasoningFlow parses traces into directed acyclic graphs, enabling the characterization of distinct reasoning patterns as subgraph structures. This human-interpretable representation offers promising applications in understanding, evaluating, and enhancing the reasoning processes of LRMs. 

---
# Multilingual Information Retrieval with a Monolingual Knowledge Base 

**Authors**: Yingying Zhuang, Aman Gupta, Anurag Beniwal  

**Link**: [PDF](https://arxiv.org/pdf/2506.02527)  

**Abstract**: Multilingual information retrieval has emerged as powerful tools for expanding knowledge sharing across languages. On the other hand, resources on high quality knowledge base are often scarce and in limited languages, therefore an effective embedding model to transform sentences from different languages into a feature vector space same as the knowledge base language becomes the key ingredient for cross language knowledge sharing, especially to transfer knowledge available in high-resource languages to low-resource ones. In this paper we propose a novel strategy to fine-tune multilingual embedding models with weighted sampling for contrastive learning, enabling multilingual information retrieval with a monolingual knowledge base. We demonstrate that the weighted sampling strategy produces performance gains compared to standard ones by up to 31.03\% in MRR and up to 33.98\% in Recall@3. Additionally, our proposed methodology is language agnostic and applicable for both multilingual and code switching use cases. 

---
# Learning Together to Perform Better: Teaching Small-Scale LLMs to Collaborate via Preferential Rationale Tuning 

**Authors**: Sohan Patnaik, Milan Aggarwal, Sumit Bhatia, Balaji Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2506.02519)  

**Abstract**: LLMssuch as GPT-4 have shown a remarkable ability to solve complex questions by generating step-by-step rationales. Prior works have utilized this capability to improve smaller and cheaper LMs (say, with 7B parameters). However, various practical constraints, such as copyright and legal issues, owing to lack of transparency in the pre-training data of large (often closed) models, prevent their use in commercial settings. Little focus has been given to improving the innate reasoning ability of smaller models without distilling information from larger LLMs. To address this, we propose COLLATE, a trainable framework that tunes a (small) LLM to generate those outputs from a pool of diverse rationales that selectively improves the downstream task. COLLATE enforces multiple instances of the same LLM to exhibit distinct behavior and employs them to generate rationales to obtain diverse outputs. The LLM is then tuned via preference optimization to choose the candidate rationale which maximizes the likelihood of ground-truth answer. COLLATE outperforms several trainable and prompting baselines on 5 datasets across 3 domains: maths problem solving, natural language inference, and commonsense reasoning. We show the eff icacy of COLLATE on LLMs from different model families across varying parameter scales (1B to 8B) and demonstrate the benefit of multiple rationale providers guided by the end task through ablations. Code is released here (this https URL). 

---
# FinChain: A Symbolic Benchmark for Verifiable Chain-of-Thought Financial Reasoning 

**Authors**: Zhuohan Xie, Dhruv Sahnan, Debopriyo Banerjee, Georgi Georgiev, Rushil Thareja, Hachem Madmoun, Jinyan Su, Aaryamonvikram Singh, Yuxia Wang, Rui Xing, Fajri Koto, Haonan Li, Ivan Koychev, Tanmoy Chakraborty, Salem Lahlou, Veselin Stoyanov, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2506.02515)  

**Abstract**: Multi-step symbolic reasoning is critical for advancing downstream performance on financial tasks. Yet, benchmarks for systematically evaluating this capability are lacking. Existing datasets like FinQA and ConvFinQA supervise only final numerical answers, without assessing intermediate reasoning steps. To address this, we introduce FinChain, the first symbolic benchmark designed for verifiable Chain-of- Thought (CoT) financial reasoning. Spanning 54 topics across 12 financial domains, Fin- Chain offers five parameterized templates per topic, each varying in reasoning complexity and domain expertise required. Each dataset instance includes an executable Python trace, enabling automatic generation of extensive training data and easy adaptation to other domains. We also introduce ChainEval, a new metric for automatic evaluation of both final answers and intermediate reasoning. Benchmarking 30 LLMs on our dataset, we find that even state-of-the-art models have considerable room for improvement in multi-step financial reasoning. All templates and evaluation metrics for FinChain are available at https: //github.com/mbzuai-nlp/finchain. 

---
# M$^3$FinMeeting: A Multilingual, Multi-Sector, and Multi-Task Financial Meeting Understanding Evaluation Dataset 

**Authors**: Jie Zhu, Junhui Li, Yalong Wen, Xiandong Li, Lifan Guo, Feng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.02510)  

**Abstract**: Recent breakthroughs in large language models (LLMs) have led to the development of new benchmarks for evaluating their performance in the financial domain. However, current financial benchmarks often rely on news articles, earnings reports, or announcements, making it challenging to capture the real-world dynamics of financial meetings. To address this gap, we propose a novel benchmark called $\texttt{M$^3$FinMeeting}$, which is a multilingual, multi-sector, and multi-task dataset designed for financial meeting understanding. First, $\texttt{M$^3$FinMeeting}$ supports English, Chinese, and Japanese, enhancing comprehension of financial discussions in diverse linguistic contexts. Second, it encompasses various industry sectors defined by the Global Industry Classification Standard (GICS), ensuring that the benchmark spans a broad range of financial activities. Finally, $\texttt{M$^3$FinMeeting}$ includes three tasks: summarization, question-answer (QA) pair extraction, and question answering, facilitating a more realistic and comprehensive evaluation of understanding. Experimental results with seven popular LLMs reveal that even the most advanced long-context models have significant room for improvement, demonstrating the effectiveness of $\texttt{M$^3$FinMeeting}$ as a benchmark for assessing LLMs' financial meeting comprehension skills. 

---
# KARE-RAG: Knowledge-Aware Refinement and Enhancement for RAG 

**Authors**: Yongjian Li, HaoCheng Chu, Yukun Yan, Zhenghao Liu, Shi Yu, Zheni Zeng, Ruobing Wang, Sen Song, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.02503)  

**Abstract**: Retrieval-Augmented Generation (RAG) enables large language models (LLMs) to access broader knowledge sources, yet factual inconsistencies persist due to noise in retrieved documents-even with advanced retrieval methods. We demonstrate that enhancing generative models' capacity to process noisy content is equally critical for robust performance. In this paper, we present KARE-RAG (Knowledge-Aware Refinement and Enhancement for RAG), which improves knowledge utilization through three key innovations: (1) structured knowledge representations that facilitate error detection during training, (2) Dense Direct Preference Optimization (DDPO)-a refined training objective that prioritizes correction of critical errors, and (3) a contrastive data generation pipeline that maintains semantic consistency while rectifying factual inaccuracies. Experiments show our method significantly enhances standard RAG pipelines across model scales, improving both in-domain and out-of-domain task performance without compromising general capabilities. Notably, these gains are achieved with modest training data, suggesting data-efficient optimization is possible through targeted learning strategies. Our findings establish a new direction for RAG improvement: by improving how models learn to process retrieved content, we can enhance performance across diverse inference paradigms. All data and code will be publicly available on Github. 

---
# Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text 

**Authors**: Junzhe Zhang, Huixuan Zhang, Xinyu Hu, Li Lin, Mingqi Gao, Shi Qiu, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2506.02494)  

**Abstract**: Evaluation is important for multimodal generation tasks. With the rapid progress of MLLMs, there is growing interest in applying MLLMs to build general evaluation systems. However, existing work overlooks two aspects: (1) the development of evaluation capabilities for text-to-image (T2I) generation task, and (2) the incorporation of large-scale human evaluation data. In this paper, we introduce Minos-Corpus, a large-scale multimodal evaluation dataset that combines evaluation data from both human and GPT. The corpus contains evaluation data across both image-to-text(I2T) and T2I generation tasks. Based on this corpus, we propose Data Selection and Balance, Mix-SFT training methods, and apply DPO to develop Minos, a multimodal evaluation model built upon a 7B backbone. Minos achieves state-of-the-art (SoTA) performance among all open-source evaluation models of similar scale on the average of evaluation performance on all tasks, and outperforms all open-source and closed-source models on evaluation of T2I generation task. Extensive experiments demonstrate the importance of leveraging high-quality human evaluation data and jointly training on evaluation data from both I2T and T2I generation tasks. 

---
# Enhancing Large Language Models with Neurosymbolic Reasoning for Multilingual Tasks 

**Authors**: Sina Bagheri Nezhad, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2506.02483)  

**Abstract**: Large language models (LLMs) often struggle to perform multi-target reasoning in long-context scenarios where relevant information is scattered across extensive documents. To address this challenge, we introduce NeuroSymbolic Augmented Reasoning (NSAR), which combines the benefits of neural and symbolic reasoning during inference. NSAR explicitly extracts symbolic facts from text and generates executable Python code to handle complex reasoning steps. Through extensive experiments across seven languages and diverse context lengths, we demonstrate that NSAR significantly outperforms both a vanilla RAG baseline and advanced prompting strategies in accurately identifying and synthesizing multiple pieces of information. Our results highlight the effectiveness of combining explicit symbolic operations with neural inference for robust, interpretable, and scalable reasoning in multilingual settings. 

---
# Do Language Models Think Consistently? A Study of Value Preferences Across Varying Response Lengths 

**Authors**: Inderjeet Nair, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02481)  

**Abstract**: Evaluations of LLMs' ethical risks and value inclinations often rely on short-form surveys and psychometric tests, yet real-world use involves long-form, open-ended responses -- leaving value-related risks and preferences in practical settings largely underexplored. In this work, we ask: Do value preferences inferred from short-form tests align with those expressed in long-form outputs? To address this question, we compare value preferences elicited from short-form reactions and long-form responses, varying the number of arguments in the latter to capture users' differing verbosity preferences. Analyzing five LLMs (llama3-8b, gemma2-9b, mistral-7b, qwen2-7b, and olmo-7b), we find (1) a weak correlation between value preferences inferred from short-form and long-form responses across varying argument counts, and (2) similarly weak correlation between preferences derived from any two distinct long-form generation settings. (3) Alignment yields only modest gains in the consistency of value expression. Further, we examine how long-form generation attributes relate to value preferences, finding that argument specificity negatively correlates with preference strength, while representation across scenarios shows a positive correlation. Our findings underscore the need for more robust methods to ensure consistent value expression across diverse applications. 

---
# ORPP: Self-Optimizing Role-playing Prompts to Enhance Language Model Capabilities 

**Authors**: Yifan Duan, Yihong Tang, Kehai Chen, Liqiang Nie, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02480)  

**Abstract**: High-quality prompts are crucial for eliciting outstanding performance from large language models (LLMs) on complex tasks. Existing research has explored model-driven strategies for prompt optimization. However, these methods often suffer from high computational overhead or require strong optimization capabilities from the model itself, which limits their broad this http URL address these challenges, we propose ORPP (Optimized Role-Playing Prompt),a framework that enhances model performance by optimizing and generating role-playing prompts. The core idea of ORPP is to confine the prompt search space to role-playing scenarios, thereby fully activating the model's intrinsic capabilities through carefully crafted, high-quality role-playing prompts. Specifically, ORPP first performs iterative optimization on a small subset of training samples to generate high-quality role-playing prompts. Then, leveraging the model's few-shot learning capability, it transfers the optimization experience to efficiently generate suitable prompts for the remaining this http URL experimental results show that ORPP not only matches but in most cases surpasses existing mainstream prompt optimization methods in terms of performance. Notably, ORPP demonstrates superior "plug-and-play" capability. In most cases, it can be integrated with various other prompt methods and further enhance their effectiveness. 

---
# FroM: Frobenius Norm-Based Data-Free Adaptive Model Merging 

**Authors**: Zijian Li, Xiaocheng Feng, Huixin Liu, Yichong Huang, Ting Liu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.02478)  

**Abstract**: With the development of large language models, fine-tuning has emerged as an effective method to enhance performance in specific scenarios by injecting domain-specific knowledge. In this context, model merging techniques provide a solution for fusing knowledge from multiple fine-tuning models by combining their parameters. However, traditional methods often encounter task interference when merging full fine-tuning models, and this problem becomes even more evident in parameter-efficient fine-tuning scenarios. In this paper, we introduce an improvement to the RegMean method, which indirectly leverages the training data to approximate the outputs of the linear layers before and after merging. We propose an adaptive merging method called FroM, which directly measures the model parameters using the Frobenius norm, without any training data. By introducing an additional hyperparameter for control, FroM outperforms baseline methods across various fine-tuning scenarios, alleviating the task interference problem. 

---
# XToM: Exploring the Multilingual Theory of Mind for Large Language Models 

**Authors**: Chunkit Chan, Yauwai Yim, Hongchuan Zeng, Zhiying Zou, Xinyuan Cheng, Zhifan Sun, Zheye Deng, Kawai Chung, Yuzhuo Ao, Yixiang Fan, Cheng Jiayang, Ercong Nie, Ginny Y. Wong, Helmut Schmid, Hinrich Schütze, Simon See, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.02461)  

**Abstract**: Theory of Mind (ToM), the ability to infer mental states in others, is pivotal for human social cognition. Existing evaluations of ToM in LLMs are largely limited to English, neglecting the linguistic diversity that shapes human cognition. This limitation raises a critical question: can LLMs exhibit Multilingual Theory of Mind, which is the capacity to reason about mental states across diverse linguistic contexts? To address this gap, we present XToM, a rigorously validated multilingual benchmark that evaluates ToM across five languages and incorporates diverse, contextually rich task scenarios. Using XToM, we systematically evaluate LLMs (e.g., DeepSeek R1), revealing a pronounced dissonance: while models excel in multilingual language understanding, their ToM performance varies across languages. Our findings expose limitations in LLMs' ability to replicate human-like mentalizing across linguistic contexts. 

---
# MidPO: Dual Preference Optimization for Safety and Helpfulness in Large Language Models via a Mixture of Experts Framework 

**Authors**: Yupeng Qi, Ziyu Lyu, Min Yang, Yanlin Wang, Lu Bai, Lixin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2506.02460)  

**Abstract**: As large language models (LLMs) are increasingly applied across various domains, enhancing safety while maintaining the helpfulness of LLMs has become a critical challenge. Recent studies solve this problem through safety-constrained online preference optimization or safety-constrained offline preference optimization. However, the safety-constrained online methods often suffer from excessive safety, which might reduce helpfulness, while the safety-constrained offline methods perform poorly in adaptively balancing safety and helpfulness. To address these limitations, we propose MidPO, a \textbf{\underline{Mi}}xture of Experts (MoE) framework for safety-helpfulness \textbf{\underline{d}}ual \textbf{\underline{P}}reference \textbf{\underline{O}}ptimization. Firstly, MidPO devises single-preference enhanced direct preference optimization approach to transform the base model into two independent experts, termed safety and helpfulness experts, and fine-tunes the two independent experts for optimal safety or helpfulness performance. Secondly, to achieve an effective balance between safety and helpfulness, MidPO incorporates the two experts into the MoE framework and designs a dynamic routing mechanism to allocate contributions from each expert adaptively. We conduct quantitative and qualitative experiments on three popular datasets to demonstrate the proposed MidPO significantly outperforms state-of-the-art approaches in both safety and helpfulness. The code and models will be released. 

---
# Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework 

**Authors**: Zhaorui Yang, Bo Pan, Han Wang, Yiyao Wang, Xingyu Liu, Minfeng Zhu, Bo Zhang, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.02454)  

**Abstract**: Visualizations play a crucial part in effective communication of concepts and information. Recent advances in reasoning and retrieval augmented generation have enabled Large Language Models (LLMs) to perform deep research and generate comprehensive reports. Despite its progress, existing deep research frameworks primarily focus on generating text-only content, leaving the automated generation of interleaved texts and visualizations underexplored. This novel task poses key challenges in designing informative visualizations and effectively integrating them with text reports. To address these challenges, we propose Formal Description of Visualization (FDV), a structured textual representation of charts that enables LLMs to learn from and generate diverse, high-quality visualizations. Building on this representation, we introduce Multimodal DeepResearcher, an agentic framework that decomposes the task into four stages: (1) researching, (2) exemplar report textualization, (3) planning, and (4) multimodal report generation. For the evaluation of generated multimodal reports, we develop MultimodalReportBench, which contains 100 diverse topics served as inputs along with 5 dedicated metrics. Extensive experiments across models and evaluation methods demonstrate the effectiveness of Multimodal DeepResearcher. Notably, utilizing the same Claude 3.7 Sonnet model, Multimodal DeepResearcher achieves an 82\% overall win rate over the baseline method. 

---
# IP-Dialog: Evaluating Implicit Personalization in Dialogue Systems with Synthetic Data 

**Authors**: Bo Peng, Zhiheng Wang, Heyang Gong, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02449)  

**Abstract**: In modern dialogue systems, the ability to implicitly infer user backgrounds from conversations and leverage this information for personalized assistance is crucial. However, the scarcity of high-quality data remains a fundamental challenge to evaluating and improving this capability. Traditional dataset construction methods are labor-intensive, resource-demanding, and raise privacy concerns. To address these issues, we propose a novel approach for automatic synthetic data generation and introduce the Implicit Personalized Dialogue (IP-Dialog) benchmark along with a training dataset, covering 10 tasks and 12 user attribute types. Additionally, we develop a systematic evaluation framework with four metrics to assess both attribute awareness and reasoning capabilities. We further propose five causal graphs to elucidate models' reasoning pathways during implicit personalization. Extensive experiments yield insightful observations and prove the reliability of our dataset. 

---
# Should LLM Safety Be More Than Refusing Harmful Instructions? 

**Authors**: Utsav Maskey, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2506.02442)  

**Abstract**: This paper presents a systematic evaluation of Large Language Models' (LLMs) behavior on long-tail distributed (encrypted) texts and their safety implications. We introduce a two-dimensional framework for assessing LLM safety: (1) instruction refusal-the ability to reject harmful obfuscated instructions, and (2) generation safety-the suppression of generating harmful responses. Through comprehensive experiments, we demonstrate that models that possess capabilities to decrypt ciphers may be susceptible to mismatched-generalization attacks: their safety mechanisms fail on at least one safety dimension, leading to unsafe responses or over-refusal. Based on these findings, we evaluate a number of pre-LLM and post-LLM safeguards and discuss their strengths and limitations. This work contributes to understanding the safety of LLM in long-tail text scenarios and provides directions for developing robust safety mechanisms. 

---
# From Anger to Joy: How Nationality Personas Shape Emotion Attribution in Large Language Models 

**Authors**: Mahammed Kamruzzaman, Abdullah Al Monsur, Gene Louis Kim, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2506.02431)  

**Abstract**: Emotions are a fundamental facet of human experience, varying across individuals, cultural contexts, and nationalities. Given the recent success of Large Language Models (LLMs) as role-playing agents, we examine whether LLMs exhibit emotional stereotypes when assigned nationality-specific personas. Specifically, we investigate how different countries are represented in pre-trained LLMs through emotion attributions and whether these attributions align with cultural norms. Our analysis reveals significant nationality-based differences, with emotions such as shame, fear, and joy being disproportionately assigned across regions. Furthermore, we observe notable misalignment between LLM-generated and human emotional responses, particularly for negative emotions, highlighting the presence of reductive and potentially biased stereotypes in LLM outputs. 

---
# Comparative Analysis of AI Agent Architectures for Entity Relationship Classification 

**Authors**: Maryam Berijanian, Kuldeep Singh, Amin Sehati  

**Link**: [PDF](https://arxiv.org/pdf/2506.02426)  

**Abstract**: Entity relationship classification remains a challenging task in information extraction, especially in scenarios with limited labeled data and complex relational structures. In this study, we conduct a comparative analysis of three distinct AI agent architectures designed to perform relation classification using large language models (LLMs). The agentic architectures explored include (1) reflective self-evaluation, (2) hierarchical task decomposition, and (3) a novel multi-agent dynamic example generation mechanism, each leveraging different modes of reasoning and prompt adaptation. In particular, our dynamic example generation approach introduces real-time cooperative and adversarial prompting. We systematically compare their performance across multiple domains and model backends. Our experiments demonstrate that multi-agent coordination consistently outperforms standard few-shot prompting and approaches the performance of fine-tuned models. These findings offer practical guidance for the design of modular, generalizable LLM-based systems for structured relation extraction. The source codes and dataset are available at \href{this https URL}{this https URL}. 

---
# Gender Inequality in English Textbooks Around the World: an NLP Approach 

**Authors**: Tairan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02425)  

**Abstract**: Textbooks play a critical role in shaping children's understanding of the world. While previous studies have identified gender inequality in individual countries' textbooks, few have examined the issue cross-culturally. This study applies natural language processing methods to quantify gender inequality in English textbooks from 22 countries across 7 cultural spheres. Metrics include character count, firstness (which gender is mentioned first), and TF-IDF word associations by gender. The analysis also identifies gender patterns in proper names appearing in TF-IDF word lists, tests whether large language models can distinguish between gendered word lists, and uses GloVe embeddings to examine how closely keywords associate with each gender. Results show consistent overrepresentation of male characters in terms of count, firstness, and named entities. All regions exhibit gender inequality, with the Latin cultural sphere showing the least disparity. 

---
# SingaKids: A Multilingual Multimodal Dialogic Tutor for Language Learning 

**Authors**: Zhengyuan Liu, Geyu Lin, Hui Li Tan, Huayun Zhang, Yanfeng Lu, Xiaoxue Gao, Stella Xin Yin, He Sun, Hock Huan Goh, Lung Hsiang Wong, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.02412)  

**Abstract**: The integration of generative artificial intelligence into educational applications has enhanced personalized and interactive learning experiences, and it shows strong potential to promote young learners language acquisition. However, it is still challenging to ensure consistent and robust performance across different languages and cultural contexts, and kids-friendly design requires simplified instructions, engaging interactions, and age-appropriate scaffolding to maintain motivation and optimize learning outcomes. In this work, we introduce SingaKids, a dialogic tutor designed to facilitate language learning through picture description tasks. Our system integrates dense image captioning, multilingual dialogic interaction, speech understanding, and engaging speech generation to create an immersive learning environment in four languages: English, Mandarin, Malay, and Tamil. We further improve the system through multilingual pre-training, task-specific tuning, and scaffolding optimization. Empirical studies with elementary school students demonstrate that SingaKids provides effective dialogic teaching, benefiting learners at different performance levels. 

---
# GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation 

**Authors**: Yilin Xiao, Junnan Dong, Chuang Zhou, Su Dong, Qianwen Zhang, Di Yin, Xing Sun, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02404)  

**Abstract**: Graph Retrieval Augmented Generation (GraphRAG) has garnered increasing recognition for its potential to enhance large language models (LLMs) by structurally organizing domain-specific corpora and facilitating complex reasoning. However, current evaluations of GraphRAG models predominantly rely on traditional question-answering datasets. Their limited scope in questions and evaluation metrics fails to comprehensively assess the reasoning capacity improvements enabled by GraphRAG models. To address this gap, we introduce GraphRAG-Bench, a large-scale, domain-specific benchmark designed to rigorously evaluate GraphRAG models. Our benchmark offers three key superiorities: \((i)\) Challenging question design. Featuring college-level, domain-specific questions that demand multi-hop reasoning, the benchmark ensures that simple content retrieval is insufficient for problem-solving. For example, some questions require mathematical reasoning or programming. \((ii)\) Diverse task coverage. The dataset includes a broad spectrum of reasoning tasks, multiple-choice, true/false, multi-select, open-ended, and fill-in-the-blank. It spans 16 disciplines in twenty core textbooks. \((iii)\) Holistic evaluation framework. GraphRAG-Bench provides comprehensive assessment across the entire GraphRAG pipeline, including graph construction, knowledge retrieval, and answer generation. Beyond final-answer correctness, it evaluates the logical coherence of the reasoning process. By applying nine contemporary GraphRAG methods to GraphRAG-Bench, we demonstrate its utility in quantifying how graph-based structuring improves model reasoning capabilities. Our analysis reveals critical insights about graph architectures, retrieval efficacy, and reasoning capabilities, offering actionable guidance for the research community. 

---
# Consultant Decoding: Yet Another Synergistic Mechanism 

**Authors**: Chuanghao Ding, Jiaping Wang, Ziqing Yang, Xiaoliang Wang, Dahua Lin, Cam-Tu Nguyen, Fei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.02391)  

**Abstract**: The synergistic mechanism based on Speculative Decoding (SD) has garnered considerable attention as a simple yet effective approach for accelerating the inference of large language models (LLMs). Nonetheless, the high rejection rates require repeated LLMs calls to validate draft tokens, undermining the overall efficiency gain of SD. In this work, we revisit existing verification mechanisms and propose a novel synergetic mechanism Consultant Decoding (CD). Unlike SD, which relies on a metric derived from importance sampling for verification, CD verifies candidate drafts using token-level likelihoods computed solely by the LLM. CD achieves up to a 2.5-fold increase in inference speed compared to the target model, while maintaining comparable generation quality (around 100% of the target model's performance). Interestingly, this is achieved by combining models whose parameter sizes differ by two orders of magnitude. In addition, CD reduces the call frequency of the large target model to below 10%, particularly in more demanding tasks. CD's performance was even found to surpass that of the large target model, which theoretically represents the upper bound for speculative decoding. 

---
# Exploring Explanations Improves the Robustness of In-Context Learning 

**Authors**: Ukyo Honda, Tatsushi Oka  

**Link**: [PDF](https://arxiv.org/pdf/2506.02378)  

**Abstract**: In-context learning (ICL) has emerged as a successful paradigm for leveraging large language models (LLMs). However, it often struggles to generalize beyond the distribution of the provided demonstrations. A recent advancement in enhancing robustness is ICL with explanations (X-ICL), which improves prediction reliability by guiding LLMs to understand and articulate the reasoning behind correct labels. Building on this approach, we introduce an advanced framework that extends X-ICL by systematically exploring explanations for all possible labels (X$^2$-ICL), thereby enabling more comprehensive and robust decision-making. Experimental results on multiple natural language understanding datasets validate the effectiveness of X$^2$-ICL, demonstrating significantly improved robustness to out-of-distribution data compared to the existing ICL approaches. 

---
# AnswerCarefully: A Dataset for Improving the Safety of Japanese LLM Output 

**Authors**: Hisami Suzuki, Satoru Katsumata, Takashi Kodama, Tetsuro Takahashi, Kouta Nakayama, Satoshi Sekine  

**Link**: [PDF](https://arxiv.org/pdf/2506.02372)  

**Abstract**: In this paper we present AnswerCarefully, a dataset for promoting the safety and appropriateness of Japanese LLM outputs. The dataset consists of 1,800 pairs of questions and reference answers, where the questions require special attention in answering. It covers a wide range of risk categories established in prior English-language datasets, but the data samples are original in that they are manually created to reflect the socio-cultural context of LLM usage in Japan. We show that using this dataset for instruction to fine-tune a Japanese LLM led to improved output safety without compromising the utility of general responses. We also report the results of a safety evaluation of 12 Japanese LLMs using this dataset as a benchmark. Finally, we describe the latest update on the dataset which provides English translations and annotations of the questions, aimed at facilitating the derivation of similar datasets in different languages and regions. 

---
# DIAMOND: An LLM-Driven Agent for Context-Aware Baseball Highlight Summarization 

**Authors**: Jeonghun Kang, Soonmok Kwon, Joonseok Lee, Byung-Hak Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.02351)  

**Abstract**: Traditional approaches -- such as Win Probability Added (WPA)-based ranking or computer vision-driven event detection -- can identify scoring plays but often miss strategic depth, momentum shifts, and storyline progression. Manual curation remains the gold standard but is resource-intensive and not scalable. We introduce DIAMOND, an LLM-driven agent for context-aware baseball highlight summarization that integrates structured sports analytics with natural language reasoning. DIAMOND leverages sabermetric features -- Win Expectancy, WPA, and Leverage Index -- to quantify play importance, while an LLM module enhances selection based on contextual narrative value. This hybrid approach ensures both quantitative rigor and qualitative richness, surpassing the limitations of purely statistical or vision-based systems. Evaluated on five diverse Korean Baseball Organization League games, DIAMOND improves F1-score from 42.9% (WPA-only) to 84.8%, outperforming both commercial and statistical baselines. Though limited in scale, our results highlight the potential of modular, interpretable agent-based frameworks for event-level summarization in sports and beyond. 

---
# Truth over Tricks: Measuring and Mitigating Shortcut Learning in Misinformation Detection 

**Authors**: Herun Wan, Jiaying Wu, Minnan Luo, Zhi Zeng, Zhixiong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.02350)  

**Abstract**: Misinformation detection models often rely on superficial cues (i.e., \emph{shortcuts}) that correlate with misinformation in training data but fail to generalize to the diverse and evolving nature of real-world misinformation. This issue is exacerbated by large language models (LLMs), which can easily generate convincing misinformation through simple prompts. We introduce TruthOverTricks, a unified evaluation paradigm for measuring shortcut learning in misinformation detection. TruthOverTricks categorizes shortcut behaviors into intrinsic shortcut induction and extrinsic shortcut injection, and evaluates seven representative detectors across 14 popular benchmarks, along with two new factual misinformation datasets, NQ-Misinfo and Streaming-Misinfo. Empirical results reveal that existing detectors suffer severe performance degradation when exposed to both naturally occurring and adversarially crafted shortcuts. To address this, we propose SMF, an LLM-augmented data augmentation framework that mitigates shortcut reliance through paraphrasing, factual summarization, and sentiment normalization. SMF consistently enhances robustness across 16 benchmarks, encouraging models to rely on deeper semantic understanding rather than shortcut cues. To promote the development of misinformation detectors, we have published the resources publicly at this https URL. 

---
# STORYTELLER: An Enhanced Plot-Planning Framework for Coherent and Cohesive Story Generation 

**Authors**: Jiaming Li, Yukun Chen, Ziqiang Liu, Minghuan Tan, Lei Zhang, Yunshui Li, Run Luo, Longze Chen, Jing Luo, Ahmadreza Argha, Hamid Alinejad-Rokny, Wei Zhou, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02347)  

**Abstract**: Stories are central to human culture, serving to share ideas, preserve traditions, and foster connections. Automatic story generation, a key advancement in artificial intelligence (AI), offers new possibilities for creating personalized content, exploring creative ideas, and enhancing interactive experiences. However, existing methods struggle to maintain narrative coherence and logical consistency. This disconnect compromises the overall storytelling experience, underscoring the need for substantial improvements. Inspired by human cognitive processes, we introduce Storyteller, a novel approach that systemically improves the coherence and consistency of automatically generated stories. Storyteller introduces a plot node structure based on linguistically grounded subject verb object (SVO) triplets, which capture essential story events and ensure a consistent logical flow. Unlike previous methods, Storyteller integrates two dynamic modules, the STORYLINE and narrative entity knowledge graph (NEKG),that continuously interact with the story generation process. This integration produces structurally sound, cohesive and immersive narratives. Extensive experiments demonstrate that Storyteller significantly outperforms existing approaches, achieving an 84.33% average win rate through human preference evaluation. At the same time, it is also far ahead in other aspects including creativity, coherence, engagement, and relevance. 

---
# One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL 

**Authors**: Hyungjoo Chae, Dongjin Kang, Jihyuk Kim, Beong-woo Kwak, Sunghyun Park, Haeju Park, Jinyoung Yeo, Moontae Lee, Kyungjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.02338)  

**Abstract**: With the release of R1, a publicly available large reasoning model (LRM), researchers commonly train new LRMs by training language models on R1's long chain-of-thought (CoT) inferences. While prior works show that LRMs' capabilities can be reproduced through direct distillation, the continued reliance on the existing models (e.g., R1) remains a critical limitation in advancing the field. As a first step toward independent LRM development, this paper explores the possibility of constructing a long CoT dataset with LLMs that are not trained for inference-time scaling. To this end, we present the Long CoT Collection, a dataset of 100K CoT rationales annotated using existing short CoT LLMs. We develop a pipeline that induces o1's novel reasoning strategies into short CoT LLMs, enabling them to think longer and introducing controllability over the thought budget to better manage the overthinking problem. Our extensive analyses validate that our dataset achieves quality comparable to--or slightly below--R1. Furthermore, our experiments demonstrate that training on our dataset not only strengthens general reasoning skills, but also provides a strong foundation for reinforcement learning--models initialized on our data achieve 2-3x larger gains with RLVR. 

---
# Something Just Like TRuST : Toxicity Recognition of Span and Target 

**Authors**: Berk Atil, Namrata Sureddy, Rebecca J. Passonneau  

**Link**: [PDF](https://arxiv.org/pdf/2506.02326)  

**Abstract**: Toxicity in online content, including content generated by language models, has become a critical concern due to its potential for negative psychological and social impact. This paper introduces TRuST, a comprehensive dataset designed to improve toxicity detection that merges existing datasets, and has labels for toxicity, target social group, and toxic spans. It includes a diverse range of target groups such as ethnicity, gender, religion, disability, and politics, with both human/machine-annotated and human machine-generated data. We benchmark state-of-the-art large language models (LLMs) on toxicity detection, target group identification, and toxic span extraction. We find that fine-tuned models consistently outperform zero-shot and few-shot prompting, though performance remains low for certain social groups. Further, reasoning capabilities do not significantly improve performance, indicating that LLMs have weak social reasoning skills. 

---
# Quantifying Misattribution Unfairness in Authorship Attribution 

**Authors**: Pegah Alipoormolabashi, Ajay Patel, Niranjan Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2506.02321)  

**Abstract**: Authorship misattribution can have profound consequences in real life. In forensic settings simply being considered as one of the potential authors of an evidential piece of text or communication can result in undesirable scrutiny. This raises a fairness question: Is every author in the candidate pool at equal risk of misattribution? Standard evaluation measures for authorship attribution systems do not explicitly account for this notion of fairness. We introduce a simple measure, Misattribution Unfairness Index (MAUIk), which is based on how often authors are ranked in the top k for texts they did not write. Using this measure we quantify the unfairness of five models on two different datasets. All models exhibit high levels of unfairness with increased risks for some authors. Furthermore, we find that this unfairness relates to how the models embed the authors as vectors in the latent search space. In particular, we observe that the risk of misattribution is higher for authors closer to the centroid (or center) of the embedded authors in the haystack. These results indicate the potential for harm and the need for communicating with and calibrating end users on misattribution risk when building and providing such models for downstream use. 

---
# Explain-then-Process: Using Grammar Prompting to Enhance Grammatical Acceptability Judgments 

**Authors**: Russell Scheinberg, Ameeta Agrawal, Amber Shore, So Young Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.02302)  

**Abstract**: Large language models (LLMs) can explain grammatical rules, yet they often fail to apply those rules when judging sentence acceptability. We present "grammar prompting", an explain-then-process paradigm: a large LLM first produces a concise explanation of the relevant syntactic phenomenon, then that explanation is fed back as additional context to the target model -- either an LLM or a smaller language model (SLM) -- before deciding which sentence of a minimal pair is grammatical. On the English BLiMP, Chinese SLING, and Russian RuBLiMP benchmarks, this simple prompt design yields substantial improvements over strong baselines across many syntactic phenomena. Feeding an LLM's metalinguistic explanation back to the target model bridges the gap between knowing a rule and using it. On SLMs, grammar prompting alone trims the average LLM-SLM accuracy gap by about 20%, and when paired with chain-of-thought, by 56% (13.0 pp -> 5.8 pp), all at negligible cost. The lightweight, language-agnostic cue lets low-cost SLMs approach frontier-LLM performance in multilingual settings. 

---
# LAM SIMULATOR: Advancing Data Generation for Large Action Model Training via Online Exploration and Trajectory Feedback 

**Authors**: Thai Hoang, Kung-Hsiang Huang, Shirley Kokane, Jianguo Zhang, Zuxin Liu, Ming Zhu, Jake Grigsby, Tian Lan, Michael S Ryoo, Chien-Sheng Wu, Shelby Heinecke, Huan Wang, Silvio Savarese, Caiming Xiong, Juan Carlos Niebles  

**Link**: [PDF](https://arxiv.org/pdf/2506.02298)  

**Abstract**: Large Action Models (LAMs) for AI Agents offer incredible potential but face challenges due to the need for high-quality training data, especially for multi-steps tasks that involve planning, executing tool calls, and responding to feedback. To address these issues, we present LAM SIMULATOR, a comprehensive framework designed for online exploration of agentic tasks with high-quality feedback. Our framework features a dynamic task query generator, an extensive collection of tools, and an interactive environment where Large Language Model (LLM) Agents can call tools and receive real-time feedback. This setup enables LLM Agents to explore and solve tasks autonomously, facilitating the discovery of multiple approaches to tackle any given task. The resulting action trajectory data are then used to create high-quality training datasets for LAMs. Our experiments on popular agentic benchmarks, ToolBench and CRMArena, highlight the effectiveness of LAM SIMULATOR: models trained with self-generated datasets using our framework achieve significant performance gains, up to a 49.3\% improvement over their original baselines. LAM SIMULATOR requires minimal human input during dataset creation, highlighting LAM SIMULATOR's efficiency and effectiveness in speeding up development of AI agents. 

---
# Sounding Like a Winner? Prosodic Differences in Post-Match Interviews 

**Authors**: Sofoklis Kakouros, Haoyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.02283)  

**Abstract**: This study examines the prosodic characteristics associated with winning and losing in post-match tennis interviews. Additionally, this research explores the potential to classify match outcomes solely based on post-match interview recordings using prosodic features and self-supervised learning (SSL) representations. By analyzing prosodic elements such as pitch and intensity, alongside SSL models like Wav2Vec 2.0 and HuBERT, the aim is to determine whether an athlete has won or lost their match. Traditional acoustic features and deep speech representations are extracted from the data, and machine learning classifiers are employed to distinguish between winning and losing players. Results indicate that SSL representations effectively differentiate between winning and losing outcomes, capturing subtle speech patterns linked to emotional states. At the same time, prosodic cues -- such as pitch variability -- remain strong indicators of victory. 

---
# ImpRAG: Retrieval-Augmented Generation with Implicit Queries 

**Authors**: Wenzheng Zhang, Xi Victoria Lin, Karl Stratos, Wen-tau Yih, Mingda Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.02279)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems traditionally treat retrieval and generation as separate processes, requiring explicit textual queries to connect them. This separation can limit the ability of models to generalize across diverse tasks. In this work, we propose a query-free RAG system, named ImpRAG, which integrates retrieval and generation into a unified model. ImpRAG allows models to implicitly express their information needs, eliminating the need for human-specified queries. By dividing pretrained decoder-only language models into specialized layer groups, ImpRAG optimizes retrieval and generation tasks simultaneously. Our approach employs a two-stage inference process, using the same model parameters and forward pass for both retrieval and generation, thereby minimizing the disparity between retrievers and language models. Experiments on 8 knowledge-intensive tasks demonstrate that ImpRAG achieves 3.6-11.5 improvements in exact match scores on unseen tasks with diverse formats, highlighting its effectiveness in enabling models to articulate their own information needs and generalize across tasks. Our analysis underscores the importance of balancing retrieval and generation parameters and leveraging generation perplexities as retrieval training objectives for enhanced performance. 

---
# CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment 

**Authors**: Radin Shayanfar, Chu Fei Luo, Rohan Bhambhoria, Samuel Dahan, Xiaodan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02264)  

**Abstract**: It is often challenging to teach specialized, unseen tasks to dialogue systems due to the high cost of expert knowledge, training data, and high technical difficulty. To support domain-specific applications - such as law, medicine, or finance - it is essential to build frameworks that enable non-technical experts to define, test, and refine system behaviour with minimal effort. Achieving this requires cross-disciplinary collaboration between developers and domain specialists. In this work, we introduce a novel framework, CoDial (Code for Dialogue), that converts expert knowledge, represented as a novel structured heterogeneous graph, into executable conversation logic. CoDial can be easily implemented in existing guardrailing languages, such as Colang, to enable interpretable, modifiable, and true zero-shot specification of task-oriented dialogue systems. Empirically, CoDial achieves state-of-the-art performance on the STAR dataset for inference-based models and is competitive with similar baselines on the well-known MultiWOZ dataset. We also demonstrate CoDial's iterative improvement via manual and LLM-aided feedback, making it a practical tool for expert-guided alignment of LLMs in high-stakes domains. 

---
# Investigating the Impact of Word Informativeness on Speech Emotion Recognition 

**Authors**: Sofoklis Kakouros  

**Link**: [PDF](https://arxiv.org/pdf/2506.02239)  

**Abstract**: In emotion recognition from speech, a key challenge lies in identifying speech signal segments that carry the most relevant acoustic variations for discerning specific emotions. Traditional approaches compute functionals for features such as energy and F0 over entire sentences or longer speech portions, potentially missing essential fine-grained variation in the long-form statistics. This research investigates the use of word informativeness, derived from a pre-trained language model, to identify semantically important segments. Acoustic features are then computed exclusively for these identified segments, enhancing emotion recognition accuracy. The methodology utilizes standard acoustic prosodic features, their functionals, and self-supervised representations. Results indicate a notable improvement in recognition performance when features are computed on segments selected based on word informativeness, underscoring the effectiveness of this approach. 

---
# Leveraging Natural Language Processing to Unravel the Mystery of Life: A Review of NLP Approaches in Genomics, Transcriptomics, and Proteomics 

**Authors**: Ella Rannon, David Burstein  

**Link**: [PDF](https://arxiv.org/pdf/2506.02212)  

**Abstract**: Natural Language Processing (NLP) has transformed various fields beyond linguistics by applying techniques originally developed for human language to the analysis of biological sequences. This review explores the application of NLP methods to biological sequence data, focusing on genomics, transcriptomics, and proteomics. We examine how various NLP methods, from classic approaches like word2vec to advanced models employing transformers and hyena operators, are being adapted to analyze DNA, RNA, protein sequences, and entire genomes. The review also examines tokenization strategies and model architectures, evaluating their strengths, limitations, and suitability for different biological tasks. We further cover recent advances in NLP applications for biological data, such as structure prediction, gene expression, and evolutionary analysis, highlighting the potential of these methods for extracting meaningful insights from large-scale genomic data. As language models continue to advance, their integration into bioinformatics holds immense promise for advancing our understanding of biological processes in all domains of life. 

---
# BehaviorBox: Automated Discovery of Fine-Grained Performance Differences Between Language Models 

**Authors**: Lindia Tjuatja, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2506.02204)  

**Abstract**: Language model evaluation is a daunting task: prompts are brittle, corpus-level perplexities are vague, and the choice of benchmarks are endless. Finding examples that show meaningful, generalizable differences between two LMs is crucial to understanding where one model succeeds and another fails. Can this process be done automatically? In this work, we propose methodology for automated comparison of language models that uses performance-aware contextual embeddings to find fine-grained features of text where one LM outperforms another. Our method, which we name BehaviorBox, extracts coherent features that demonstrate differences with respect to the ease of generation between two LMs. Specifically, BehaviorBox finds features that describe groups of words in fine-grained contexts, such as "conditional 'were' in the phrase 'if you were'" and "exclamation marks after emotional statements", where one model outperforms another within a particular datatset. We apply BehaviorBox to compare models that vary in size, model family, and post-training, and enumerate insights into specific contexts that illustrate meaningful differences in performance which cannot be found by measures such as corpus-level perplexity alone. 

---
# Echoes of Phonetics: Unveiling Relevant Acoustic Cues for ASR via Feature Attribution 

**Authors**: Dennis Fucci, Marco Gaido, Matteo Negri, Mauro Cettolo, Luisa Bentivogli  

**Link**: [PDF](https://arxiv.org/pdf/2506.02181)  

**Abstract**: Despite significant advances in ASR, the specific acoustic cues models rely on remain unclear. Prior studies have examined such cues on a limited set of phonemes and outdated models. In this work, we apply a feature attribution technique to identify the relevant acoustic cues for a modern Conformer-based ASR system. By analyzing plosives, fricatives, and vowels, we assess how feature attributions align with their acoustic properties in the time and frequency domains, also essential for human speech perception. Our findings show that the ASR model relies on vowels' full time spans, particularly their first two formants, with greater saliency in male speech. It also better captures the spectral characteristics of sibilant fricatives than non-sibilants and prioritizes the release phase in plosives, especially burst characteristics. These insights enhance the interpretability of ASR models and highlight areas for future research to uncover potential gaps in model robustness. 

---
# AI Debate Aids Assessment of Controversial Claims 

**Authors**: Salman Rahman, Sheriff Issaka, Ashima Suvarna, Genglin Liu, James Shiffer, Jaeyoung Lee, Md Rizwan Parvez, Hamid Palangi, Shi Feng, Nanyun Peng, Yejin Choi, Julian Michael, Liwei Jiang, Saadia Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2506.02175)  

**Abstract**: As AI grows more powerful, it will increasingly shape how we understand the world. But with this influence comes the risk of amplifying misinformation and deepening social divides-especially on consequential topics like public health where factual accuracy directly impacts well-being. Scalable Oversight aims to ensure AI truthfulness by enabling humans to supervise systems that may exceed human capabilities--yet humans themselves hold different beliefs and biases that impair their judgment. We study whether AI debate can guide biased judges toward the truth by having two AI systems debate opposing sides of controversial COVID-19 factuality claims where people hold strong prior beliefs. We conduct two studies: one with human judges holding either mainstream or skeptical beliefs evaluating factuality claims through AI-assisted debate or consultancy protocols, and a second examining the same problem with personalized AI judges designed to mimic these different human belief systems. In our human study, we find that debate-where two AI advisor systems present opposing evidence-based arguments-consistently improves judgment accuracy and confidence calibration, outperforming consultancy with a single-advisor system by 10% overall. The improvement is most significant for judges with mainstream beliefs (+15.2% accuracy), though debate also helps skeptical judges who initially misjudge claims move toward accurate views (+4.7% accuracy). In our AI judge study, we find that AI judges with human-like personas achieve even higher accuracy (78.5%) than human judges (70.1%) and default AI judges without personas (69.8%), suggesting their potential for supervising frontier AI models. These findings highlight AI debate as a promising path toward scalable, bias-resilient oversight--leveraging both diverse human and AI judgments to move closer to truth in contested domains. 

---
# Different Speech Translation Models Encode and Translate Speaker Gender Differently 

**Authors**: Dennis Fucci, Marco Gaido, Matteo Negri, Luisa Bentivogli, Andre Martins, Giuseppe Attanasio  

**Link**: [PDF](https://arxiv.org/pdf/2506.02172)  

**Abstract**: Recent studies on interpreting the hidden states of speech models have shown their ability to capture speaker-specific features, including gender. Does this finding also hold for speech translation (ST) models? If so, what are the implications for the speaker's gender assignment in translation? We address these questions from an interpretability perspective, using probing methods to assess gender encoding across diverse ST models. Results on three language directions (English-French/Italian/Spanish) indicate that while traditional encoder-decoder models capture gender information, newer architectures -- integrating a speech encoder with a machine translation system via adapters -- do not. We also demonstrate that low gender encoding capabilities result in systems' tendency toward a masculine default, a translation bias that is more pronounced in newer architectures. 

---
# HENT-SRT: Hierarchical Efficient Neural Transducer with Self-Distillation for Joint Speech Recognition and Translation 

**Authors**: Amir Hussein, Cihan Xiao, Matthew Wiesner, Dan Povey, Leibny Paola Garcia, Sanjeev Khudanpur  

**Link**: [PDF](https://arxiv.org/pdf/2506.02157)  

**Abstract**: Neural transducers (NT) provide an effective framework for speech streaming, demonstrating strong performance in automatic speech recognition (ASR). However, the application of NT to speech translation (ST) remains challenging, as existing approaches struggle with word reordering and performance degradation when jointly modeling ASR and ST, resulting in a gap with attention-based encoder-decoder (AED) models. Existing NT-based ST approaches also suffer from high computational training costs. To address these issues, we propose HENT-SRT (Hierarchical Efficient Neural Transducer for Speech Recognition and Translation), a novel framework that factorizes ASR and translation tasks to better handle reordering. To ensure robust ST while preserving ASR performance, we use self-distillation with CTC consistency regularization. Moreover, we improve computational efficiency by incorporating best practices from ASR transducers, including a down-sampled hierarchical encoder, a stateless predictor, and a pruned transducer loss to reduce training complexity. Finally, we introduce a blank penalty during decoding, reducing deletions and improving translation quality. Our approach is evaluated on three conversational datasets Arabic, Spanish, and Mandarin achieving new state-of-the-art performance among NT models and substantially narrowing the gap with AED-based systems. 

---
# BabyLM's First Constructions: Causal interventions provide a signal of learning 

**Authors**: Joshua Rozner, Leonie Weissweiler, Cory Shain  

**Link**: [PDF](https://arxiv.org/pdf/2506.02147)  

**Abstract**: Construction grammar posits that children acquire constructions (form-meaning pairings) from the statistics of their environment. Recent work supports this hypothesis by showing sensitivity to constructions in pretrained language models (PLMs), including one recent study (Rozner et al., 2025) demonstrating that constructions shape the PLM's output distribution. However, models under study have generally been trained on developmentally implausible amounts of data, casting doubt on their relevance to human language learning. Here we use Rozner et al.'s methods to evaluate constructional learning in models from the 2024 BabyLM challenge. Our results show that even when trained on developmentally plausible quantities of data, models represent diverse constructions, even hard cases that are superficially indistinguishable. We further find correlational evidence that constructional performance may be functionally relevant: models that better represent constructions perform better on the BabyLM benchmarks. 

---
# Model Internal Sleuthing: Finding Lexical Identity and Inflectional Morphology in Modern Language Models 

**Authors**: Michael Li, Nishant Subramani  

**Link**: [PDF](https://arxiv.org/pdf/2506.02132)  

**Abstract**: Large transformer-based language models dominate modern NLP, yet our understanding of how they encode linguistic information is rooted in studies of early models like BERT and GPT-2. To better understand today's language models, we investigate how both classical architectures (BERT, DeBERTa, GPT-2)and contemporary large language models (Pythia, OLMo-2, Gemma-2, Qwen2.5, Llama-3.1) represent lexical identity and inflectional morphology. We train linear and nonlinear classifiers on layer-wise activations to predict word lemmas and inflectional features. We discover that models concentrate lexical information linearly in early layers and increasingly nonlinearly in later layers, while keeping inflectional information uniformly accessible and linearly separable throughout the layers. Further analysis reveals that these models encode inflectional morphology through generalizable abstractions, but rely predominantly on memorization to encode lexical identity. Remarkably, these patterns emerge across all 16 models we test, despite differences in architecture, size, and training regime (including pretrained and instruction-tuned variants). This consistency suggests that, despite substantial advances in LLM technologies, transformer models organize linguistic information in similar ways, indicating that these properties could be fundamental for next token prediction and are learned early during pretraining. Our code is available at this https URL. 

---
# Knowledge or Reasoning? A Close Look at How LLMs Think Across Domains 

**Authors**: Juncheng Wu, Sheng Liu, Haoqin Tu, Hang Yu, Xiaoke Huang, James Zou, Cihang Xie, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.02126)  

**Abstract**: Recent advances in reasoning-enhanced Large Language Models such as OpenAI-o1/3 and DeepSeek-R1 have significantly improved performance on complex tasks. However, the quality and transparency of their internal reasoning processes remain underexplored. This work moves beyond the final-answer accuracy and investigates step-by-step reasoning in the medical and mathematical domains by explicitly decomposing the thinking trajectories into two parts: knowledge and reasoning. Specifically, we introduce a fine-grained evaluation framework that judges: (1) the correctness of knowledge used (measured by Knowledge Index (KI)) and (2) the quality of reasoning (measured by Information Gain (InfoGain)). Using this framework, we study R1-distilled and base Qwen models trained with supervised fine-tuning (SFT) and/or reinforcement learning (RL) in the medical and math domains. Three intriguing findings emerge: (1) The general reasoning abilities in R1-distilled models do not transfer effectively to the medical domain through either SFT or RL. (2) SFT raises final-answer accuracy in both domains, but often at the cost of reasoning quality: InfoGain drops by 38.9% on average compared with untrained models; In the medical domain, however, SFT remains crucial because domain knowledge is indispensable. (3) RL enhances medical reasoning by pruning inaccurate or irrelevant knowledge from reasoning paths, thereby improving both reasoning accuracy and knowledge correctness. 

---
# Evaluating the Unseen Capabilities: How Many Theorems Do LLMs Know? 

**Authors**: Xiang Li, Jiayi Xin, Qi Long, Weijie J. Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.02058)  

**Abstract**: Accurate evaluation of large language models (LLMs) is crucial for understanding their capabilities and guiding their development. However, current evaluations often inconsistently reflect the actual capacities of these models. In this paper, we demonstrate that one of many contributing factors to this \textit{evaluation crisis} is the oversight of unseen knowledge -- information encoded by LLMs but not directly observed or not yet observed during evaluations. We introduce KnowSum, a statistical framework designed to provide a more comprehensive assessment by quantifying the unseen knowledge for a class of evaluation tasks. KnowSum estimates the unobserved portion by extrapolating from the appearance frequencies of observed knowledge instances. We demonstrate the effectiveness and utility of KnowSum across three critical applications: estimating total knowledge, evaluating information retrieval effectiveness, and measuring output diversity. Our experiments reveal that a substantial volume of knowledge is omitted when relying solely on observed LLM performance. Importantly, KnowSum yields significantly different comparative rankings for several common LLMs based on their internal knowledge. 

---
# Enhancing Multimodal Continual Instruction Tuning with BranchLoRA 

**Authors**: Duzhen Zhang, Yong Ren, Zhong-Zhi Li, Yahan Yu, Jiahua Dong, Chenxing Li, Zhilong Ji, Jinfeng Bai  

**Link**: [PDF](https://arxiv.org/pdf/2506.02041)  

**Abstract**: Multimodal Continual Instruction Tuning (MCIT) aims to finetune Multimodal Large Language Models (MLLMs) to continually align with human intent across sequential tasks. Existing approaches often rely on the Mixture-of-Experts (MoE) LoRA framework to preserve previous instruction alignments. However, these methods are prone to Catastrophic Forgetting (CF), as they aggregate all LoRA blocks via simple summation, which compromises performance over time. In this paper, we identify a critical parameter inefficiency in the MoELoRA framework within the MCIT context. Based on this insight, we propose BranchLoRA, an asymmetric framework to enhance both efficiency and performance. To mitigate CF, we introduce a flexible tuning-freezing mechanism within BranchLoRA, enabling branches to specialize in intra-task knowledge while fostering inter-task collaboration. Moreover, we incrementally incorporate task-specific routers to ensure an optimal branch distribution over time, rather than favoring the most recent task. To streamline inference, we introduce a task selector that automatically routes test inputs to the appropriate router without requiring task identity. Extensive experiments on the latest MCIT benchmark demonstrate that BranchLoRA significantly outperforms MoELoRA and maintains its superiority across various MLLM sizes. 

---
# FinS-Pilot: A Benchmark for Online Financial System 

**Authors**: Feng Wang, Yiding Sun, Jiaxin Mao, Wei Xue, Danqing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02037)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various professional domains, with their performance typically evaluated through standardized benchmarks. However, the development of financial RAG benchmarks has been constrained by data confidentiality issues and the lack of dynamic data integration. To address this issue, we introduces FinS-Pilot, a novel benchmark for evaluating RAG systems in online financial applications. Constructed from real-world financial assistant interactions, our benchmark incorporates both real-time API data and structured text sources, organized through an intent classification framework covering critical financial domains such as equity analysis and macroeconomic forecasting. The benchmark enables comprehensive evaluation of financial assistants' capabilities in handling both static knowledge and time-sensitive market information. Through systematic experiments with multiple Chinese leading LLMs, we demonstrate FinS-Pilot's effectiveness in identifying models suitable for financial applications while addressing the current gap in specialized evaluation tools for the financial domain. Our work contributes both a practical evaluation framework and a curated dataset to advance research in financial NLP systems. The code and dataset are accessible on GitHub\footnote{this https URL\_rag\_benchmark}. 

---
# ChatCFD: an End-to-End CFD Agent with Domain-specific Structured Thinking 

**Authors**: E Fan, Weizong Wang, Tianhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02019)  

**Abstract**: Computational Fluid Dynamics (CFD) is essential for scientific and engineering advancements but is limited by operational complexity and the need for extensive expertise. This paper presents ChatCFD, a large language model-driven pipeline that automates CFD workflows within the OpenFOAM framework. It enables users to configure and execute complex simulations from natural language prompts or published literature with minimal expertise. The innovation is its structured approach to database construction, configuration validation, and error reflection, integrating CFD and OpenFOAM knowledge with general language models to improve accuracy and adaptability. Validation shows ChatCFD can autonomously reproduce published CFD results, handling complex, unseen configurations beyond basic examples, a task challenging for general language models. 

---
# Enhancing Paraphrase Type Generation: The Impact of DPO and RLHF Evaluated with Human-Ranked Data 

**Authors**: Christopher Lee Lübbers  

**Link**: [PDF](https://arxiv.org/pdf/2506.02018)  

**Abstract**: Paraphrasing re-expresses meaning to enhance applications like text simplification, machine translation, and question-answering. Specific paraphrase types facilitate accurate semantic analysis and robust language models. However, existing paraphrase-type generation methods often misalign with human preferences due to reliance on automated metrics and limited human-annotated training data, obscuring crucial aspects of semantic fidelity and linguistic transformations.
This study addresses this gap by leveraging a human-ranked paraphrase-type dataset and integrating Direct Preference Optimization (DPO) to align model outputs directly with human judgments. DPO-based training increases paraphrase-type generation accuracy by 3 percentage points over a supervised baseline and raises human preference ratings by 7 percentage points. A newly created human-annotated dataset supports more rigorous future evaluations. Additionally, a paraphrase-type detection model achieves F1 scores of 0.91 for addition/deletion, 0.78 for same polarity substitution, and 0.70 for punctuation changes.
These findings demonstrate that preference data and DPO training produce more reliable, semantically accurate paraphrases, enabling downstream applications such as improved summarization and more robust question-answering. The PTD model surpasses automated metrics and provides a more reliable framework for evaluating paraphrase quality, advancing paraphrase-type research toward richer, user-aligned language generation and establishing a stronger foundation for future evaluations grounded in human-centric criteria. 

---
# Pruning for Performance: Efficient Idiom and Metaphor Classification in Low-Resource Konkani Using mBERT 

**Authors**: Timothy Do, Pranav Saran, Harshita Poojary, Pranav Prabhu, Sean O'Brien, Vasu Sharma, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02005)  

**Abstract**: In this paper, we address the persistent challenges that figurative language expressions pose for natural language processing (NLP) systems, particularly in low-resource languages such as Konkani. We present a hybrid model that integrates a pre-trained Multilingual BERT (mBERT) with a bidirectional LSTM and a linear classifier. This architecture is fine-tuned on a newly introduced annotated dataset for metaphor classification, developed as part of this work. To improve the model's efficiency, we implement a gradient-based attention head pruning strategy. For metaphor classification, the pruned model achieves an accuracy of 78%. We also applied our pruning approach to expand on an existing idiom classification task, achieving 83% accuracy. These results demonstrate the effectiveness of attention head pruning for building efficient NLP tools in underrepresented languages. 

---
# NovelHopQA: Diagnosing Multi-Hop Reasoning Failures in Long Narrative Contexts 

**Authors**: Abhay Gupta, Michael Lu, Kevin Zhu, Sean O'Brien, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2506.02000)  

**Abstract**: Current large language models (LLMs) struggle to answer questions that span tens of thousands of tokens, especially when multi-hop reasoning is involved. While prior benchmarks explore long-context comprehension or multi-hop reasoning in isolation, none jointly vary context length and reasoning depth in natural narrative settings. We introduce NovelHopQA, the first benchmark to evaluate k1-4 hop QA over 64k-128k-token excerpts from 83 full-length public-domain novels. A keyword-guided pipeline builds hop-separated chains grounded in coherent storylines. We evaluate six state-of-the-art (SOTA) models and apply oracle-context filtering to ensure all questions are genuinely answerable. Human annotators validate both alignment and hop depth. We noticed consistent accuracy drops with increased hops and context length, even in frontier models-revealing that sheer scale does not guarantee robust reasoning. Our failure mode analysis highlights common breakdowns, such as missed final-hop integration and long-range drift. NovelHopQA offers a controlled diagnostic setting to stress-test multi-hop reasoning at scale. 

---
# No Free Lunch in Active Learning: LLM Embedding Quality Dictates Query Strategy Success 

**Authors**: Lukas Rauch, Moritz Wirth, Denis Huseljic, Marek Herde, Bernhard Sick, Matthias Aßenmacher  

**Link**: [PDF](https://arxiv.org/pdf/2506.01992)  

**Abstract**: The advent of large language models (LLMs) capable of producing general-purpose representations lets us revisit the practicality of deep active learning (AL): By leveraging frozen LLM embeddings, we can mitigate the computational costs of iteratively fine-tuning large backbones. This study establishes a benchmark and systematically investigates the influence of LLM embedding quality on query strategies in deep AL. We employ five top-performing models from the massive text embedding benchmark (MTEB) leaderboard and two baselines for ten diverse text classification tasks. Our findings reveal key insights: First, initializing the labeled pool using diversity-based sampling synergizes with high-quality embeddings, boosting performance in early AL iterations. Second, the choice of the optimal query strategy is sensitive to embedding quality. While the computationally inexpensive Margin sampling can achieve performance spikes on specific datasets, we find that strategies like Badge exhibit greater robustness across tasks. Importantly, their effectiveness is often enhanced when paired with higher-quality embeddings. Our results emphasize the need for context-specific evaluation of AL strategies, as performance heavily depends on embedding quality and the target task. 

---
# Research on Medical Named Entity Identification Based On Prompt-Biomrc Model and Its Application in Intelligent Consultation System 

**Authors**: Jinzhu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01961)  

**Abstract**: This study is dedicated to exploring the application of prompt learning methods to advance Named Entity Recognition (NER) within the medical domain. In recent years, the emergence of large-scale models has driven significant progress in NER tasks, particularly with the introduction of the BioBERT language model, which has greatly enhanced NER capabilities in medical texts. Our research introduces the Prompt-bioMRC model, which integrates both hard template and soft prompt designs aimed at refining the precision and efficiency of medical entity recognition. Through extensive experimentation across diverse medical datasets, our findings consistently demonstrate that our approach surpasses traditional models. This enhancement not only validates the efficacy of our methodology but also highlights its potential to provide reliable technological support for applications like intelligent diagnosis systems. By leveraging advanced NER techniques, this study contributes to advancing automated medical data processing, facilitating more accurate medical information extraction, and supporting efficient healthcare decision-making processes. 

---
# UniWorld: High-Resolution Semantic Encoders for Unified Visual Understanding and Generation 

**Authors**: Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, Yatian Pang, Li Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2506.03147)  

**Abstract**: Although existing unified models deliver strong performance on vision-language understanding and text-to-image generation, their models are limited in exploring image perception and manipulation tasks, which are urgently desired by users for wide applications. Recently, OpenAI released their powerful GPT-4o-Image model for comprehensive image perception and manipulation, achieving expressive capability and attracting community interests. By observing the performance of GPT-4o-Image in our carefully constructed experiments, we infer that GPT-4o-Image leverages features extracted by semantic encoders instead of VAE, while VAEs are considered essential components in many image manipulation models. Motivated by such inspiring observations, we present a unified generative framework named UniWorld based on semantic features provided by powerful visual-language models and contrastive semantic encoders. As a result, we build a strong unified model using only 1% amount of BAGEL's data, which consistently outperforms BAGEL on image editing benchmarks. UniWorld also maintains competitive image understanding and generation capabilities, achieving strong performance across multiple image perception tasks. We fully open-source our models, including model weights, training and evaluation scripts, and datasets. 

---
# MERIT: Multilingual Semantic Retrieval with Interleaved Multi-Condition Query 

**Authors**: Wei Chow, Yuan Gao, Linfeng Li, Xian Wang, Qi Xu, Hang Song, Lingdong Kong, Ran Zhou, Yi Zeng, Yidong Cai, Botian Jiang, Shilin Xu, Jiajun Zhang, Minghui Qiu, Xiangtai Li, Tianshu Yang, Siliang Tang, Juncheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.03144)  

**Abstract**: Semantic retrieval is crucial for modern applications yet remains underexplored in current research. Existing datasets are limited to single languages, single images, or singular retrieval conditions, often failing to fully exploit the expressive capacity of visual information as evidenced by maintained performance when images are replaced with captions. However, practical retrieval scenarios frequently involve interleaved multi-condition queries with multiple images. Hence, this paper introduces MERIT, the first multilingual dataset for interleaved multi-condition semantic retrieval, comprising 320,000 queries with 135,000 products in 5 languages, covering 7 distinct product categories. Extensive experiments on MERIT identify existing models's limitation: focusing solely on global semantic information while neglecting specific conditional elements in queries. Consequently, we propose Coral, a novel fine-tuning framework that adapts pre-trained MLLMs by integrating embedding reconstruction to preserve fine-grained conditional elements and contrastive learning to extract comprehensive global semantics. Experiments demonstrate that Coral achieves a 45.9% performance improvement over conventional approaches on MERIT, with strong generalization capabilities validated across 8 established retrieval benchmarks. Collectively, our contributions - a novel dataset, identification of critical limitations in existing approaches, and an innovative fine-tuning framework - establish a foundation for future research in interleaved multi-condition semantic retrieval. 

---
# OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models 

**Authors**: Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, Li Yi  

**Link**: [PDF](https://arxiv.org/pdf/2506.03135)  

**Abstract**: Spatial reasoning is a key aspect of cognitive psychology and remains a major bottleneck for current vision-language models (VLMs). While extensive research has aimed to evaluate or improve VLMs' understanding of basic spatial relations, such as distinguishing left from right, near from far, and object counting, these tasks represent only the most fundamental level of spatial reasoning. In this work, we introduce OmniSpatial, a comprehensive and challenging benchmark for spatial reasoning, grounded in cognitive psychology. OmniSpatial covers four major categories: dynamic reasoning, complex spatial logic, spatial interaction, and perspective-taking, with 50 fine-grained subcategories. Through Internet data crawling and careful manual annotation, we construct over 1.5K question-answer pairs. Extensive experiments show that both open- and closed-source VLMs, as well as existing reasoning and spatial understanding models, exhibit significant limitations in comprehensive spatial understanding. We further analyze failure cases and propose potential directions for future research. 

---
# Retrieval-Augmented Generation as Noisy In-Context Learning: A Unified Theory and Risk Bounds 

**Authors**: Yang Guo, Yutian Tao, Yifei Ming, Robert D. Nowak, Yingyu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03100)  

**Abstract**: Retrieval-augmented generation (RAG) has seen many empirical successes in recent years by aiding the LLM with external knowledge. However, its theoretical aspect has remained mostly unexplored. In this paper, we propose the first finite-sample generalization bound for RAG in in-context linear regression and derive an exact bias-variance tradeoff. Our framework views the retrieved texts as query-dependent noisy in-context examples and recovers the classical in-context learning (ICL) and standard RAG as the limit cases. Our analysis suggests that an intrinsic ceiling on generalization error exists on RAG as opposed to the ICL. Furthermore, our framework is able to model retrieval both from the training data and from external corpora by introducing uniform and non-uniform RAG noise. In line with our theory, we show the sample efficiency of ICL and RAG empirically with experiments on common QA benchmarks, such as Natural Questions and TriviaQA. 

---
# MAEBE: Multi-Agent Emergent Behavior Framework 

**Authors**: Sinem Erisken, Timothy Gothard, Martin Leitgab, Ram Potham  

**Link**: [PDF](https://arxiv.org/pdf/2506.03053)  

**Abstract**: Traditional AI safety evaluations on isolated LLMs are insufficient as multi-agent AI ensembles become prevalent, introducing novel emergent risks. This paper introduces the Multi-Agent Emergent Behavior Evaluation (MAEBE) framework to systematically assess such risks. Using MAEBE with the Greatest Good Benchmark (and a novel double-inversion question technique), we demonstrate that: (1) LLM moral preferences, particularly for Instrumental Harm, are surprisingly brittle and shift significantly with question framing, both in single agents and ensembles. (2) The moral reasoning of LLM ensembles is not directly predictable from isolated agent behavior due to emergent group dynamics. (3) Specifically, ensembles exhibit phenomena like peer pressure influencing convergence, even when guided by a supervisor, highlighting distinct safety and alignment challenges. Our findings underscore the necessity of evaluating AI systems in their interactive, multi-agent contexts. 

---
# Mitigating Manipulation and Enhancing Persuasion: A Reflective Multi-Agent Approach for Legal Argument Generation 

**Authors**: Li Zhang, Kevin D. Ashley  

**Link**: [PDF](https://arxiv.org/pdf/2506.02992)  

**Abstract**: Large Language Models (LLMs) are increasingly explored for legal argument generation, yet they pose significant risks of manipulation through hallucination and ungrounded persuasion, and often fail to utilize provided factual bases effectively or abstain when arguments are untenable. This paper introduces a novel reflective multi-agent method designed to address these challenges in the context of legally compliant persuasion. Our approach employs specialized agents--a Factor Analyst and an Argument Polisher--in an iterative refinement process to generate 3-ply legal arguments (plaintiff, defendant, rebuttal). We evaluate Reflective Multi-Agent against single-agent, enhanced-prompt single-agent, and non-reflective multi-agent baselines using four diverse LLMs (GPT-4o, GPT-4o-mini, Llama-4-Maverick-17b-128e, Llama-4-Scout-17b-16e) across three legal scenarios: "arguable", "mismatched", and "non-arguable". Results demonstrate Reflective Multi-Agent's significant superiority in successful abstention (preventing generation when arguments cannot be grounded), marked improvements in hallucination accuracy (reducing fabricated and misattributed factors), particularly in "non-arguable" scenarios, and enhanced factor utilization recall (improving the use of provided case facts). These findings suggest that structured reflection within a multi-agent framework offers a robust computable method for fostering ethical persuasion and mitigating manipulation in LLM-based legal argumentation systems, a critical step towards trustworthy AI in law. Project page: this https URL 

---
# Scaling Fine-Grained MoE Beyond 50B Parameters: Empirical Evaluation and Practical Insights 

**Authors**: Jakub Krajewski, Marcin Chochowski, Daniel Korzekwa  

**Link**: [PDF](https://arxiv.org/pdf/2506.02890)  

**Abstract**: Mixture of Experts (MoE) architectures have emerged as pivotal for scaling Large Language Models (LLMs) efficiently. Fine-grained MoE approaches - utilizing more numerous, smaller experts - have demonstrated potential in improving model convergence and quality. This work proposes a set of training recipes and provides a comprehensive empirical evaluation of fine-grained MoE, directly comparing its scaling properties against standard MoE configurations for models with up to 56B total (17B active) parameters. We investigate convergence speed, model performance on downstream benchmarks, and practical training considerations across various setups. Overall, at the largest scale we show that fine-grained MoE achieves better validation loss and higher accuracy across a set of downstream benchmarks. This study offers empirical grounding and practical insights for leveraging fine-grained MoE in the development of future large-scale models. 

---
# Demystifying Reasoning Dynamics with Mutual Information: Thinking Tokens are Information Peaks in LLM Reasoning 

**Authors**: Chen Qian, Dongrui Liu, Haochen Wen, Zhen Bai, Yong Liu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.02867)  

**Abstract**: Large reasoning models (LRMs) have demonstrated impressive capabilities in complex problem-solving, yet their internal reasoning mechanisms remain poorly understood. In this paper, we investigate the reasoning trajectories of LRMs from an information-theoretic perspective. By tracking how mutual information (MI) between intermediate representations and the correct answer evolves during LRM reasoning, we observe an interesting MI peaks phenomenon: the MI at specific generative steps exhibits a sudden and significant increase during LRM's reasoning process. We theoretically analyze such phenomenon and show that as MI increases, the probability of model's prediction error decreases. Furthermore, these MI peaks often correspond to tokens expressing reflection or transition, such as ``Hmm'', ``Wait'' and ``Therefore,'' which we term as the thinking tokens. We then demonstrate that these thinking tokens are crucial for LRM's reasoning performance, while other tokens has minimal impacts. Building on these analyses, we propose two simple yet effective methods to improve LRM's reasoning performance, by delicately leveraging these thinking tokens. Overall, our work provides novel insights into the reasoning mechanisms of LRMs and offers practical ways to improve their reasoning capabilities. The code is available at this https URL. 

---
# Rethinking Machine Unlearning in Image Generation Models 

**Authors**: Renyang Liu, Wenjie Feng, Tianwei Zhang, Wei Zhou, Xueqi Cheng, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2506.02761)  

**Abstract**: With the surge and widespread application of image generation models, data privacy and content safety have become major concerns and attracted great attention from users, service providers, and policymakers. Machine unlearning (MU) is recognized as a cost-effective and promising means to address these challenges. Despite some advancements, image generation model unlearning (IGMU) still faces remarkable gaps in practice, e.g., unclear task discrimination and unlearning guidelines, lack of an effective evaluation framework, and unreliable evaluation metrics. These can hinder the understanding of unlearning mechanisms and the design of practical unlearning algorithms. We perform exhaustive assessments over existing state-of-the-art unlearning algorithms and evaluation standards, and discover several critical flaws and challenges in IGMU tasks. Driven by these limitations, we make several core contributions, to facilitate the comprehensive understanding, standardized categorization, and reliable evaluation of IGMU. Specifically, (1) We design CatIGMU, a novel hierarchical task categorization framework. It provides detailed implementation guidance for IGMU, assisting in the design of unlearning algorithms and the construction of testbeds. (2) We introduce EvalIGMU, a comprehensive evaluation framework. It includes reliable quantitative metrics across five critical aspects. (3) We construct DataIGM, a high-quality unlearning dataset, which can be used for extensive evaluations of IGMU, training content detectors for judgment, and benchmarking the state-of-the-art unlearning algorithms. With EvalIGMU and DataIGM, we discover that most existing IGMU algorithms cannot handle the unlearning well across different evaluation dimensions, especially for preservation and robustness. Code and models are available at this https URL. 

---
# An Exploratory Framework for Future SETI Applications: Detecting Generative Reactivity via Language Models 

**Authors**: Po-Chieh Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02730)  

**Abstract**: We present an exploratory framework to test whether noise-like input can induce structured responses in language models. Instead of assuming that extraterrestrial signals must be decoded, we evaluate whether inputs can trigger linguistic behavior in generative systems. This shifts the focus from decoding to viewing structured output as a sign of underlying regularity in the input. We tested GPT-2 small, a 117M-parameter model trained on English text, using four types of acoustic input: human speech, humpback whale vocalizations, Phylloscopus trochilus birdsong, and algorithmically generated white noise. All inputs were treated as noise-like, without any assumed symbolic encoding. To assess reactivity, we defined a composite score called Semantic Induction Potential (SIP), combining entropy, syntax coherence, compression gain, and repetition penalty. Results showed that whale and bird vocalizations had higher SIP scores than white noise, while human speech triggered only moderate responses. This suggests that language models may detect latent structure even in data without conventional semantics. We propose that this approach could complement traditional SETI methods, especially in cases where communicative intent is unknown. Generative reactivity may offer a different way to identify data worth closer attention. 

---
# Benchmarking and Advancing Large Language Models for Local Life Services 

**Authors**: Xiaochong Lan, Jie Feng, Jiahuan Lei, Xinlei Shi, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.02720)  

**Abstract**: Large language models (LLMs) have exhibited remarkable capabilities and achieved significant breakthroughs across various domains, leading to their widespread adoption in recent years. Building on this progress, we investigate their potential in the realm of local life services. In this study, we establish a comprehensive benchmark and systematically evaluate the performance of diverse LLMs across a wide range of tasks relevant to local life services. To further enhance their effectiveness, we explore two key approaches: model fine-tuning and agent-based workflows. Our findings reveal that even a relatively compact 7B model can attain performance levels comparable to a much larger 72B model, effectively balancing inference cost and model capability. This optimization greatly enhances the feasibility and efficiency of deploying LLMs in real-world online services, making them more practical and accessible for local life applications. 

---
# Iterative Self-Improvement of Vision Language Models for Image Scoring and Self-Explanation 

**Authors**: Naoto Tanji, Toshihiko Yamasaki  

**Link**: [PDF](https://arxiv.org/pdf/2506.02708)  

**Abstract**: Image scoring is a crucial task in numerous real-world applications. To trust a model's judgment, understanding its rationale is essential. This paper proposes a novel training method for Vision Language Models (VLMs) to generate not only image scores but also corresponding justifications in natural language. Leveraging only an image scoring dataset and an instruction-tuned VLM, our method enables self-training, utilizing the VLM's generated text without relying on external data or models. In addition, we introduce a simple method for creating a dataset designed to improve alignment between predicted scores and their textual justifications. By iteratively training the model with Direct Preference Optimization on two distinct datasets and merging them, we can improve both scoring accuracy and the coherence of generated explanations. 

---
# Synthetic Speech Source Tracing using Metric Learning 

**Authors**: Dimitrios Koutsianos, Stavros Zacharopoulos, Yannis Panagakis, Themos Stafylakis  

**Link**: [PDF](https://arxiv.org/pdf/2506.02590)  

**Abstract**: This paper addresses source tracing in synthetic speech-identifying generative systems behind manipulated audio via speaker recognition-inspired pipelines. While prior work focuses on spoofing detection, source tracing lacks robust solutions. We evaluate two approaches: classification-based and metric-learning. We tested our methods on the MLAADv5 benchmark using ResNet and self-supervised learning (SSL) backbones. The results show that ResNet achieves competitive performance with the metric learning approach, matching and even exceeding SSL-based systems. Our work demonstrates ResNet's viability for source tracing while underscoring the need to optimize SSL representations for this task. Our work bridges speaker recognition methodologies with audio forensic challenges, offering new directions for combating synthetic media manipulation. 

---
# Response-Level Rewards Are All You Need for Online Reinforcement Learning in LLMs: A Mathematical Perspective 

**Authors**: Shenghua He, Tian Xia, Xuan Zhou, Hui Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.02553)  

**Abstract**: We study a common challenge in reinforcement learning for large language models (LLMs): the Zero-Reward Assumption, where non-terminal actions (i.e., intermediate token generations) receive zero task-specific immediate reward, while only the final token receives a reward for the entire response. This assumption arises frequently in practice, as precise token-level rewards are often difficult or infeasible to obtain in LLM applications. In this work, we provide a unifying theoretical perspective. We introduce the Trajectory Policy Gradient Theorem, which shows that the policy gradient based on true, unknown token-level rewards can be unbiasedly estimated using only a response-level reward model, regardless of whether the Zero-Reward Assumption holds or not, for algorithms in the REINFORCE and Actor-Critic families. This result reveals that widely used methods such as PPO, GRPO, ReMax, and RLOO inherently possess the capacity to model token-level reward signals, offering a theoretical justification for response-level reward approaches. Our findings pave the way for more practical, efficient LLM fine-tuning, allowing developers to treat training algorithms as black boxes and focus on improving the response-level reward model with auxiliary sub-models. We also offer a detailed analysis of popular RL and non-RL methods, comparing their theoretical foundations and practical advantages across common LLM tasks. Finally, we propose a new algorithm: Token-Reinforced Policy Optimization (TRePO), a theoretically grounded method that is simpler than PPO, matches GRPO in memory efficiency, and holds promise for broad applicability. 

---
# Automated Web Application Testing: End-to-End Test Case Generation with Large Language Models and Screen Transition Graphs 

**Authors**: Nguyen-Khang Le, Quan Minh Bui, Minh Ngoc Nguyen, Hiep Nguyen, Trung Vo, Son T. Luu, Shoshin Nomura, Minh Le Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.02529)  

**Abstract**: Web applications are critical to modern software ecosystems, yet ensuring their reliability remains challenging due to the complexity and dynamic nature of web interfaces. Recent advances in large language models (LLMs) have shown promise in automating complex tasks, but limitations persist in handling dynamic navigation flows and complex form interactions. This paper presents an automated system for generating test cases for two key aspects of web application testing: site navigation and form filling. For site navigation, the system employs screen transition graphs and LLMs to model navigation flows and generate test scenarios. For form filling, it uses state graphs to handle conditional forms and automates Selenium script generation. Key contributions include: (1) a novel integration of graph structures and LLMs for site navigation testing, (2) a state graph-based approach for automating form-filling test cases, and (3) a comprehensive dataset for evaluating form-interaction testing. Experimental results demonstrate the system's effectiveness in improving test coverage and robustness, advancing the state of web application testing. 

---
# BitBypass: A New Direction in Jailbreaking Aligned Large Language Models with Bitstream Camouflage 

**Authors**: Kalyan Nakka, Nitesh Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2506.02479)  

**Abstract**: The inherent risk of generating harmful and unsafe content by Large Language Models (LLMs), has highlighted the need for their safety alignment. Various techniques like supervised fine-tuning, reinforcement learning from human feedback, and red-teaming were developed for ensuring the safety alignment of LLMs. However, the robustness of these aligned LLMs is always challenged by adversarial attacks that exploit unexplored and underlying vulnerabilities of the safety alignment. In this paper, we develop a novel black-box jailbreak attack, called BitBypass, that leverages hyphen-separated bitstream camouflage for jailbreaking aligned LLMs. This represents a new direction in jailbreaking by exploiting fundamental information representation of data as continuous bits, rather than leveraging prompt engineering or adversarial manipulations. Our evaluation of five state-of-the-art LLMs, namely GPT-4o, Gemini 1.5, Claude 3.5, Llama 3.1, and Mixtral, in adversarial perspective, revealed the capabilities of BitBypass in bypassing their safety alignment and tricking them into generating harmful and unsafe content. Further, we observed that BitBypass outperforms several state-of-the-art jailbreak attacks in terms of stealthiness and attack success. Overall, these results highlights the effectiveness and efficiency of BitBypass in jailbreaking these state-of-the-art LLMs. 

---
# Comba: Improving Nonlinear RNNs with Closed-loop Control 

**Authors**: Jiaxi Hu, Yongqi Pan, Jusen Du, Disen Lan, Xiaqiang Tang, Qingsong Wen, Yuxuan Liang, Weigao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.02475)  

**Abstract**: Recent efficient sequence modeling methods such as Gated DeltaNet, TTT, and RWKV-7 have achieved performance improvements by supervising the recurrent memory management through Delta learning rule. Unlike previous state-space models (e.g., Mamba) and gated linear attentions (e.g., GLA), these models introduce interactions between the recurrent state and the key vector, resulting in a nonlinear recursive structure. In this paper, we first introduce the concept of Nonlinear RNNs with a comprehensive analysis on the advantages and limitations of these models. Then, based on closed-loop control theory, we propose a novel Nonlinear RNN variant named Comba, which adopts a scalar-plus-low-rank state transition, with both state feedback and output feedback corrections. We also implement a hardware-efficient chunk-wise parallel kernel in Triton and train models with 340M/1.3B parameters on large-scale corpus. Comba demonstrates its superior performance and computation efficiency in both language and vision modeling. 

---
# StarVC: A Unified Auto-Regressive Framework for Joint Text and Speech Generation in Voice Conversion 

**Authors**: Fengjin Li, Jie Wang, Yadong Niu, Yongqing Wang, Meng Meng, Jian Luan, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02414)  

**Abstract**: Voice Conversion (VC) modifies speech to match a target speaker while preserving linguistic content. Traditional methods usually extract speaker information directly from speech while neglecting the explicit utilization of linguistic content. Since VC fundamentally involves disentangling speaker identity from linguistic content, leveraging structured semantic features could enhance conversion performance. However, previous attempts to incorporate semantic features into VC have shown limited effectiveness, motivating the integration of explicit text modeling. We propose StarVC, a unified autoregressive VC framework that first predicts text tokens before synthesizing acoustic features. The experiments demonstrate that StarVC outperforms conventional VC methods in preserving both linguistic content (i.e., WER and CER) and speaker characteristics (i.e., SECS and MOS). Audio demo can be found at: this https URL. 

---
# ResearchCodeBench: Benchmarking LLMs on Implementing Novel Machine Learning Research Code 

**Authors**: Tianyu Hua, Harper Hua, Violet Xiang, Benjamin Klieger, Sang T. Truong, Weixin Liang, Fan-Yun Sun, Nick Haber  

**Link**: [PDF](https://arxiv.org/pdf/2506.02314)  

**Abstract**: Large language models (LLMs) have shown promise in transforming machine learning research, yet their capability to faithfully implement novel ideas from recent research papers-ideas unseen during pretraining-remains unclear. We introduce ResearchCodeBench, a benchmark of 212 coding challenges that evaluates LLMs' ability to translate cutting-edge ML contributions from top 2024-2025 research papers into executable code. We assessed 30+ proprietary and open-source LLMs, finding that even the best models correctly implement less than 40% of the code. We find Gemini-2.5-Pro-Preview to perform best at 37.3% success rate, with O3 (High) and O4-mini (High) following behind at 32.3% and 30.8% respectively. We present empirical findings on performance comparison, contamination, and error patterns. By providing a rigorous and community-driven evaluation platform, ResearchCodeBench enables continuous understanding and advancement of LLM-driven innovation in research code generation. 

---
# VLCD: Vision-Language Contrastive Distillation for Accurate and Efficient Automatic Placenta Analysis 

**Authors**: Manas Mehta, Yimu Pan, Kelly Gallagher, Alison D. Gernand, Jeffery A. Goldstein, Delia Mwinyelle, Leena Mithal, James Z. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02229)  

**Abstract**: Pathological examination of the placenta is an effective method for detecting and mitigating health risks associated with childbirth. Recent advancements in AI have enabled the use of photographs of the placenta and pathology reports for detecting and classifying signs of childbirth-related pathologies. However, existing automated methods are computationally extensive, which limits their deployability. We propose two modifications to vision-language contrastive learning (VLC) frameworks to enhance their accuracy and efficiency: (1) text-anchored vision-language contrastive knowledge distillation (VLCD)-a new knowledge distillation strategy for medical VLC pretraining, and (2) unsupervised predistillation using a large natural images dataset for improved initialization. Our approach distills efficient neural networks that match or surpass the teacher model in performance while achieving model compression and acceleration. Our results showcase the value of unsupervised predistillation in improving the performance and robustness of our approach, specifically for lower-quality images. VLCD serves as an effective way to improve the efficiency and deployability of medical VLC approaches, making AI-based healthcare solutions more accessible, especially in resource-constrained environments. 

---
# KDRL: Post-Training Reasoning LLMs via Unified Knowledge Distillation and Reinforcement Learning 

**Authors**: Hongling Xu, Qi Zhu, Heyuan Deng, Jinpeng Li, Lu Hou, Yasheng Wang, Lifeng Shang, Ruifeng Xu, Fei Mi  

**Link**: [PDF](https://arxiv.org/pdf/2506.02208)  

**Abstract**: Recent advances in large language model (LLM) post-training have leveraged two distinct paradigms to enhance reasoning capabilities: reinforcement learning (RL) and knowledge distillation (KD). While RL enables the emergence of complex reasoning behaviors, it often suffers from low sample efficiency when the initial policy struggles to explore high-reward trajectories. Conversely, KD improves learning efficiency via mimicking the teacher model but tends to generalize poorly to out-of-domain scenarios. In this work, we present \textbf{KDRL}, a \textit{unified post-training framework} that jointly optimizes a reasoning model through teacher supervision (KD) and self-exploration (RL). Specifically, KDRL leverages policy gradient optimization to simultaneously minimize the reverse Kullback-Leibler divergence (RKL) between the student and teacher distributions while maximizing the expected rule-based rewards. We first formulate a unified objective that integrates GRPO and KD, and systematically explore how different KL approximations, KL coefficients, and reward-guided KD strategies affect the overall post-training dynamics and performance. Empirical results on multiple reasoning benchmarks demonstrate that KDRL outperforms GRPO and various KD baselines while achieving a favorable balance between performance and reasoning token efficiency. These findings indicate that integrating KD and RL serves as an effective and efficient strategy to train reasoning LLMs. 

---
# Cocktail-Party Audio-Visual Speech Recognition 

**Authors**: Thai-Binh Nguyen, Ngoc-Quan Pham, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2506.02178)  

**Abstract**: Audio-Visual Speech Recognition (AVSR) offers a robust solution for speech recognition in challenging environments, such as cocktail-party scenarios, where relying solely on audio proves insufficient. However, current AVSR models are often optimized for idealized scenarios with consistently active speakers, overlooking the complexities of real-world settings that include both speaking and silent facial segments. This study addresses this gap by introducing a novel audio-visual cocktail-party dataset designed to benchmark current AVSR systems and highlight the limitations of prior approaches in realistic noisy conditions. Additionally, we contribute a 1526-hour AVSR dataset comprising both talking-face and silent-face segments, enabling significant performance gains in cocktail-party environments. Our approach reduces WER by 67% relative to the state-of-the-art, reducing WER from 119% to 39.2% in extreme noise, without relying on explicit segmentation cues. 

---
# A Dynamic Framework for Semantic Grouping of Common Data Elements (CDE) Using Embeddings and Clustering 

**Authors**: Madan Krishnamurthy, Daniel Korn, Melissa A Haendel, Christopher J Mungall, Anne E Thessen  

**Link**: [PDF](https://arxiv.org/pdf/2506.02160)  

**Abstract**: This research aims to develop a dynamic and scalable framework to facilitate harmonization of Common Data Elements (CDEs) across heterogeneous biomedical datasets by addressing challenges such as semantic heterogeneity, structural variability, and context dependence to streamline integration, enhance interoperability, and accelerate scientific discovery. Our methodology leverages Large Language Models (LLMs) for context-aware text embeddings that convert CDEs into dense vectors capturing semantic relationships and patterns. These embeddings are clustered using Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) to group semantically similar CDEs. The framework incorporates four key steps: (1) LLM-based text embedding to mathematically represent semantic context, (2) unsupervised clustering of embeddings via HDBSCAN, (3) automated labeling using LLM summarization, and (4) supervised learning to train a classifier assigning new or unclustered CDEs to labeled clusters. Evaluated on the NIH NLM CDE Repository with over 24,000 CDEs, the system identified 118 meaningful clusters at an optimized minimum cluster size of 20. The classifier achieved 90.46 percent overall accuracy, performing best in larger categories. External validation against Gravity Projects Social Determinants of Health domains showed strong agreement (Adjusted Rand Index 0.52, Normalized Mutual Information 0.78), indicating that embeddings effectively capture cluster characteristics. This adaptable and scalable approach offers a practical solution to CDE harmonization, improving selection efficiency and supporting ongoing data interoperability. 

---
# SynthRL: Scaling Visual Reasoning with Verifiable Data Synthesis 

**Authors**: Zijian Wu, Jinjie Ni, Xiangyan Liu, Zichen Liu, Hang Yan, Michael Qizhe Shieh  

**Link**: [PDF](https://arxiv.org/pdf/2506.02096)  

**Abstract**: Vision-language models (VLMs) trained via reinforcement learning with verifiable reward (RLVR) have shown notable progress in scaling test-time compute effectively. In this work, we investigate how synthesized RL data can further improve RLVR. To this end, we propose \textbf{SynthRL}-a scalable and guaranteed pipeline for automatic data scaling in reasoning-oriented RL training. SynthRL comprises three key stages: (1) selecting seed questions with appropriate distribution, (2) augmenting them into more challenging variants while preserving the original answers, and (3) a guaranteed verification stage that ensures near-perfect correctness and difficulty enhancement. Our empirical experiments demonstrate SynthRL's scalability and effectiveness. When applied to the MMK12 dataset, SynthRL synthesizes over 3.3K additional verifiable, challenging questions from approximately 8K seed samples. Models trained with our synthesized data achieve consistent gains across five out-of-domain visual math reasoning benchmarks, with a significant improvement over baseline models trained on seed data alone. Notably, detailed analysis reveals that the gains are more pronounced on the most challenging evaluation samples, highlighting SynthRL's effectiveness in eliciting deeper and more complex reasoning patterns. 

---
# Enhancing Speech Emotion Recognition with Graph-Based Multimodal Fusion and Prosodic Features for the Speech Emotion Recognition in Naturalistic Conditions Challenge at Interspeech 2025 

**Authors**: Alef Iury Siqueira Ferreira, Lucas Rafael Gris, Alexandre Ferro Filho, Lucas Ólives, Daniel Ribeiro, Luiz Fernando, Fernanda Lustosa, Rodrigo Tanaka, Frederico Santos de Oliveira, Arlindo Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2506.02088)  

**Abstract**: Training SER models in natural, spontaneous speech is especially challenging due to the subtle expression of emotions and the unpredictable nature of real-world audio. In this paper, we present a robust system for the INTERSPEECH 2025 Speech Emotion Recognition in Naturalistic Conditions Challenge, focusing on categorical emotion recognition. Our method combines state-of-the-art audio models with text features enriched by prosodic and spectral cues. In particular, we investigate the effectiveness of Fundamental Frequency (F0) quantization and the use of a pretrained audio tagging model. We also employ an ensemble model to improve robustness. On the official test set, our system achieved a Macro F1-score of 39.79% (42.20% on validation). Our results underscore the potential of these methods, and analysis of fusion techniques confirmed the effectiveness of Graph Attention Networks. Our source code is publicly available. 

---
# Unveiling Audio Deepfake Origins: A Deep Metric learning And Conformer Network Approach With Ensemble Fusion 

**Authors**: Ajinkya Kulkarni, Sandipana Dowerah, Tanel Alumae, Mathew Magimai.-Doss  

**Link**: [PDF](https://arxiv.org/pdf/2506.02085)  

**Abstract**: Audio deepfakes are acquiring an unprecedented level of realism with advanced AI. While current research focuses on discerning real speech from spoofed speech, tracing the source system is equally crucial. This work proposes a novel audio source tracing system combining deep metric multi-class N-pair loss with Real Emphasis and Fake Dispersion framework, a Conformer classification network, and ensemble score-embedding fusion. The N-pair loss improves discriminative ability, while Real Emphasis and Fake Dispersion enhance robustness by focusing on differentiating real and fake speech patterns. The Conformer network captures both global and local dependencies in the audio signal, crucial for source tracing. The proposed ensemble score-embedding fusion shows an optimal trade-off between in-domain and out-of-domain source tracing scenarios. We evaluate our method using Frechet Distance and standard metrics, demonstrating superior performance in source tracing over the baseline system. 

---
# Assigning Distinct Roles to Quantized and Low-Rank Matrices Toward Optimal Weight Decomposition 

**Authors**: Yoonjun Cho, Soeun Kim, Dongjae Jeon, Kyelim Lee, Beomsoo Lee, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2506.02077)  

**Abstract**: Decomposing weight matrices into quantization and low-rank components ($\mathbf{W} \approx \mathbf{Q} + \mathbf{L}\mathbf{R}$) is a widely used technique for compressing large language models (LLMs). Existing joint optimization methods iteratively alternate between quantization and low-rank approximation. However, these methods tend to prioritize one component at the expense of the other, resulting in suboptimal decompositions that fail to leverage each component's unique strengths. In this work, we introduce Outlier-Driven Low-Rank Initialization (ODLRI), which assigns low-rank components the specific role of capturing activation-sensitive weights. This structured decomposition mitigates outliers' negative impact on quantization, enabling more effective balance between quantization and low-rank approximation. Experiments on Llama2 (7B, 13B, 70B), Llama3-8B, and Mistral-7B demonstrate that incorporating ODLRI into the joint optimization framework consistently reduces activation-aware error, minimizes quantization scale, and improves perplexity and zero-shot accuracy in low-bit settings. 

---
# Learning More with Less: Self-Supervised Approaches for Low-Resource Speech Emotion Recognition 

**Authors**: Ziwei Gong, Pengyuan Shi, Kaan Donbekci, Lin Ai, Run Chen, David Sasu, Zehui Wu, Julia Hirschberg  

**Link**: [PDF](https://arxiv.org/pdf/2506.02059)  

**Abstract**: Speech Emotion Recognition (SER) has seen significant progress with deep learning, yet remains challenging for Low-Resource Languages (LRLs) due to the scarcity of annotated data. In this work, we explore unsupervised learning to improve SER in low-resource settings. Specifically, we investigate contrastive learning (CL) and Bootstrap Your Own Latent (BYOL) as self-supervised approaches to enhance cross-lingual generalization. Our methods achieve notable F1 score improvements of 10.6% in Urdu, 15.2% in German, and 13.9% in Bangla, demonstrating their effectiveness in LRLs. Additionally, we analyze model behavior to provide insights on key factors influencing performance across languages, and also highlighting challenges in low-resource SER. This work provides a foundation for developing more inclusive, explainable, and robust emotion recognition systems for underrepresented languages. 

---
# Enhancing Speech Instruction Understanding and Disambiguation in Robotics via Speech Prosody 

**Authors**: David Sasu, Kweku Andoh Yamoah, Benedict Quartey, Natalie Schluter  

**Link**: [PDF](https://arxiv.org/pdf/2506.02057)  

**Abstract**: Enabling robots to accurately interpret and execute spoken language instructions is essential for effective human-robot collaboration. Traditional methods rely on speech recognition to transcribe speech into text, often discarding crucial prosodic cues needed for disambiguating intent. We propose a novel approach that directly leverages speech prosody to infer and resolve instruction intent. Predicted intents are integrated into large language models via in-context learning to disambiguate and select appropriate task plans. Additionally, we present the first ambiguous speech dataset for robotics, designed to advance research in speech disambiguation. Our method achieves 95.79% accuracy in detecting referent intents within an utterance and determines the intended task plan of ambiguous instructions with 71.96% accuracy, demonstrating its potential to significantly improve human-robot communication. 

---
# Inter(sectional) Alia(s): Ambiguity in Voice Agent Identity via Intersectional Japanese Self-Referents 

**Authors**: Takao Fujii, Katie Seaborn, Madeleine Steeds, Jun Kato  

**Link**: [PDF](https://arxiv.org/pdf/2506.01998)  

**Abstract**: Conversational agents that mimic people have raised questions about the ethics of anthropomorphizing machines with human social identity cues. Critics have also questioned assumptions of identity neutrality in humanlike agents. Recent work has revealed that intersectional Japanese pronouns can elicit complex and sometimes evasive impressions of agent identity. Yet, the role of other "neutral" non-pronominal self-referents (NPSR) and voice as a socially expressive medium remains unexplored. In a crowdsourcing study, Japanese participants (N = 204) evaluated three ChatGPT voices (Juniper, Breeze, and Ember) using seven self-referents. We found strong evidence of voice gendering alongside the potential of intersectional self-referents to evade gendering, i.e., ambiguity through neutrality and elusiveness. Notably, perceptions of age and formality intersected with gendering as per sociolinguistic theories, especially boku and watakushi. This work provides a nuanced take on agent identity perceptions and champions intersectional and culturally-sensitive work on voice agents. 

---
# Turning LLM Activations Quantization-Friendly 

**Authors**: Patrik Czakó, Gábor Kertész, Sándor Szénási  

**Link**: [PDF](https://arxiv.org/pdf/2506.01967)  

**Abstract**: Quantization effectively reduces the serving costs of Large Language Models (LLMs) by speeding up data movement through compressed parameters and enabling faster operations via integer arithmetic. However, activating integer arithmetic requires quantizing both weights and activations, which poses challenges due to the significant outliers in LLMs that increase quantization error. In this work, we investigate these outliers with an emphasis on their effect on layer-wise quantization error, then examine how smoothing and rotation transform the observed values. Our primary contributions include introducing a new metric to measure and visualize quantization difficulty based on channel magnitudes, as well as proposing a hybrid approach that applies channel-wise scaling before rotation, supported by a mathematical formulation of its benefits. 

---
# Breaking Quadratic Barriers: A Non-Attention LLM for Ultra-Long Context Horizons 

**Authors**: Andrew Kiruluta, Preethi Raju, Priscilla Burity  

**Link**: [PDF](https://arxiv.org/pdf/2506.01963)  

**Abstract**: We present a novel non attention based architecture for large language models (LLMs) that efficiently handles very long context windows, on the order of hundreds of thousands to potentially millions of tokens. Unlike traditional Transformer designs, which suffer from quadratic memory and computation overload due to the nature of the self attention mechanism, our model avoids token to token attention entirely. Instead, it combines the following complementary components: State Space blocks (inspired by S4) that learn continuous time convolution kernels and scale near linearly with sequence length, Multi Resolution Convolution layers that capture local context at different dilation levels, a lightweight Recurrent Supervisor to maintain a global hidden state across sequential chunks, and Retrieval Augmented External Memory that stores and retrieves high-level chunk embeddings without reintroducing quadratic operations. 

---
# Generate, Not Recommend: Personalized Multimodal Content Generation 

**Authors**: Jiongnan Liu, Zhicheng Dou, Ning Hu, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.01704)  

**Abstract**: To address the challenge of information overload from massive web contents, recommender systems are widely applied to retrieve and present personalized results for users. However, recommendation tasks are inherently constrained to filtering existing items and lack the ability to generate novel concepts, limiting their capacity to fully satisfy user demands and preferences. In this paper, we propose a new paradigm that goes beyond content filtering and selecting: directly generating personalized items in a multimodal form, such as images, tailored to individual users. To accomplish this, we leverage any-to-any Large Multimodal Models (LMMs) and train them in both supervised fine-tuning and online reinforcement learning strategy to equip them with the ability to yield tailored next items for users. Experiments on two benchmark datasets and user study confirm the efficacy of the proposed method. Notably, the generated images not only align well with users' historical preferences but also exhibit relevance to their potential future interests. 

---
