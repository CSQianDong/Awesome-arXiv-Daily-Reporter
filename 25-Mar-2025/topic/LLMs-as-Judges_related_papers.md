# J&H: Evaluating the Robustness of Large Language Models Under Knowledge-Injection Attacks in Legal Domain 

**Authors**: Yiran Hu, Huanghai Liu, Qingjing Chen, Ning Zheng, Chong Wang, Yun Liu, Charles L.A. Clarke, Weixing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2503.18360)  

**Abstract**: As the scale and capabilities of Large Language Models (LLMs) increase, their applications in knowledge-intensive fields such as legal domain have garnered widespread attention. However, it remains doubtful whether these LLMs make judgments based on domain knowledge for reasoning. If LLMs base their judgments solely on specific words or patterns, rather than on the underlying logic of the language, the ''LLM-as-judges'' paradigm poses substantial risks in the real-world applications. To address this question, we propose a method of legal knowledge injection attacks for robustness testing, thereby inferring whether LLMs have learned legal knowledge and reasoning logic. In this paper, we propose J&H: an evaluation framework for detecting the robustness of LLMs under knowledge injection attacks in the legal domain. The aim of the framework is to explore whether LLMs perform deductive reasoning when accomplishing legal tasks. To further this aim, we have attacked each part of the reasoning logic underlying these tasks (major premise, minor premise, and conclusion generation). We have collected mistakes that legal experts might make in judicial decisions in the real world, such as typos, legal synonyms, inaccurate external legal statutes retrieval. However, in real legal practice, legal experts tend to overlook these mistakes and make judgments based on logic. However, when faced with these errors, LLMs are likely to be misled by typographical errors and may not utilize logic in their judgments. We conducted knowledge injection attacks on existing general and domain-specific LLMs. Current LLMs are not robust against the attacks employed in our experiments. In addition we propose and compare several methods to enhance the knowledge robustness of LLMs. 

---
# Improving Preference Extraction In LLMs By Identifying Latent Knowledge Through Classifying Probes 

**Authors**: Sharan Maiya, Yinhong Liu, Ramit Debnath, Anna Korhonen  

**Link**: [PDF](https://arxiv.org/pdf/2503.17755)  

**Abstract**: Large Language Models (LLMs) are often used as automated judges to evaluate text, but their effectiveness can be hindered by various unintentional biases. We propose using linear classifying probes, trained by leveraging differences between contrasting pairs of prompts, to directly access LLMs' latent knowledge and extract more accurate preferences. Through extensive experiments using models of varying size from four different families and six diverse datasets assessing text quality evaluation and common sense reasoning, we demonstrate that both supervised and unsupervised probing approaches consistently outperform traditional generation-based judgement while maintaining similar computational costs. These probes generalise under domain shifts and can even outperform finetuned evaluators with the same training data size. Our results suggest linear probing offers an accurate, robust and computationally efficient approach for LLM-as-judge tasks while providing interpretable insights into how models encode judgement-relevant knowledge. Our data and code will be openly released in the future. 

---
# Judge Anything: MLLM as a Judge Across Any Modality 

**Authors**: Shu Pu, Yaochen Wang, Dongping Chen, Yuhang Chen, Guohao Wang, Qi Qin, Zhongyi Zhang, Zhiyuan Zhang, Zetong Zhou, Shuang Gong, Yi Gui, Yao Wan, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17489)  

**Abstract**: Evaluating generative foundation models on open-ended multimodal understanding (MMU) and generation (MMG) tasks across diverse modalities (e.g., images, audio, video) poses significant challenges due to the complexity of cross-modal interactions. To this end, the idea of utilizing Multimodal LLMs (MLLMs) as automated judges has emerged, with encouraging results in assessing vision-language understanding tasks. Moving further, this paper extends MLLM-as-a-Judge across modalities to a unified manner by introducing two benchmarks, TaskAnything and JudgeAnything, to respectively evaluate the overall performance and judging capabilities of MLLMs across any-to-any modality tasks. Specifically, TaskAnything evaluates the MMU and MMG capabilities across 15 any-to-any modality categories, employing 1,500 queries curated from well-established benchmarks. Furthermore, JudgeAnything evaluates the judging capabilities of 5 advanced (e.g., GPT-4o and Gemini-2.0-Flash) from the perspectives of Pair Comparison and Score Evaluation, providing a standardized testbed that incorporates human judgments and detailed rubrics. Our extensive experiments reveal that while these MLLMs show promise in assessing MMU (i.e., achieving an average of 66.55% in Pair Comparison setting and 42.79% in Score Evaluation setting), they encounter significant challenges with MMG tasks (i.e., averaging only 53.37% in Pair Comparison setting and 30.05% in Score Evaluation setting), exposing cross-modality biases and hallucination issues. To address this, we present OmniArena, an automated platform for evaluating omni-models and multimodal reward models. Our work highlights the need for fairer evaluation protocols and stronger alignment with human preferences. The source code and dataset are publicly available at: this https URL. 

---
# On the effectiveness of LLMs for automatic grading of open-ended questions in Spanish 

**Authors**: Germán Capdehourat, Isabel Amigo, Brian Lorenzo, Joaquín Trigo  

**Link**: [PDF](https://arxiv.org/pdf/2503.18072)  

**Abstract**: Grading is a time-consuming and laborious task that educators must face. It is an important task since it provides feedback signals to learners, and it has been demonstrated that timely feedback improves the learning process. In recent years, the irruption of LLMs has shed light on the effectiveness of automatic grading. In this paper, we explore the performance of different LLMs and prompting techniques in automatically grading short-text answers to open-ended questions. Unlike most of the literature, our study focuses on a use case where the questions, answers, and prompts are all in Spanish. Experimental results comparing automatic scores to those of human-expert evaluators show good outcomes in terms of accuracy, precision and consistency for advanced LLMs, both open and proprietary. Results are notably sensitive to prompt styles, suggesting biases toward certain words or content in the prompt. However, the best combinations of models and prompt strategies, consistently surpasses an accuracy of 95% in a three-level grading task, which even rises up to more than 98% when the it is simplified to a binary right or wrong rating problem, which demonstrates the potential that LLMs have to implement this type of automation in education applications. 

---
