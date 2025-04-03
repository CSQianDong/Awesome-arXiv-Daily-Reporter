# FAIRE: Assessing Racial and Gender Bias in AI-Driven Resume Evaluations 

**Authors**: Athena Wen, Tanush Patil, Ansh Saxena, Yicheng Fu, Sean O'Brien, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01420)  

**Abstract**: In an era where AI-driven hiring is transforming recruitment practices, concerns about fairness and bias have become increasingly important. To explore these issues, we introduce a benchmark, FAIRE (Fairness Assessment In Resume Evaluation), to test for racial and gender bias in large language models (LLMs) used to evaluate resumes across different industries. We use two methods-direct scoring and ranking-to measure how model performance changes when resumes are slightly altered to reflect different racial or gender identities. Our findings reveal that while every model exhibits some degree of bias, the magnitude and direction vary considerably. This benchmark provides a clear way to examine these differences and offers valuable insights into the fairness of AI-based hiring tools. It highlights the urgent need for strategies to reduce bias in AI-driven recruitment. Our benchmark code and dataset are open-sourced at our repository: this https URL. 

---
# LITE: LLM-Impelled efficient Taxonomy Evaluation 

**Authors**: Lin Zhang, Zhouhong Gu, Suhang Zheng, Tao Wang, Tianyu Li, Hongwei Feng, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.01369)  

**Abstract**: This paper presents LITE, an LLM-based evaluation method designed for efficient and flexible assessment of taxonomy quality. To address challenges in large-scale taxonomy evaluation, such as efficiency, fairness, and consistency, LITE adopts a top-down hierarchical evaluation strategy, breaking down the taxonomy into manageable substructures and ensuring result reliability through cross-validation and standardized input formats. LITE also introduces a penalty mechanism to handle extreme cases and provides both quantitative performance analysis and qualitative insights by integrating evaluation metrics closely aligned with task objectives. Experimental results show that LITE demonstrates high reliability in complex evaluation tasks, effectively identifying semantic errors, logical contradictions, and structural flaws in taxonomies, while offering directions for improvement. Code is available at this https URL . 

---
# Prompt-Reverse Inconsistency: LLM Self-Inconsistency Beyond Generative Randomness and Prompt Paraphrasing 

**Authors**: Jihyun Janice Ahn, Wenpeng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2504.01282)  

**Abstract**: While the inconsistency of LLMs is not a novel topic, prior research has predominantly addressed two types of generative inconsistencies: i) Randomness Inconsistency: running the same LLM multiple trials, yielding varying responses; ii) Paraphrase Inconsistency: paraphrased prompts result in different responses from the same LLM. Randomness Inconsistency arises from the inherent randomness due to stochastic sampling in generative models, while Paraphrase Inconsistency is a consequence of the language modeling objectives, where paraphrased prompts alter the distribution of vocabulary logits. This research discovers Prompt-Reverse Inconsistency (PRIN), a new form of LLM self-inconsistency: given a question and a couple of LLM-generated answer candidates, the LLM often has conflicting responses when prompted "Which are correct answers?" and "Which are incorrect answers?". PRIN poses a big concern as it undermines the credibility of LLM-as-a-judge, and suggests a challenge for LLMs to adhere to basic logical rules. We conduct a series of experiments to investigate PRIN, examining the extent of PRIN across different LLMs, methods to mitigate it, potential applications, and its relationship with Randomness Inconsistency and Paraphrase Inconsistency. As the first study to explore PRIN, our findings offer valuable insights into the inner workings of LLMs and contribute to advancing trustworthy AI. 

---
# Grade Guard: A Smart System for Short Answer Automated Grading 

**Authors**: Niharika Dadu, Harsh Vardhan Singh, Romi Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2504.01253)  

**Abstract**: The advent of large language models (LLMs) in the education sector has provided impetus to automate grading short answer questions. LLMs make evaluating short answers very efficient, thus addressing issues like staff shortage. However, in the task of Automated Short Answer Grading (ASAG), LLM responses are influenced by diverse perspectives in their training dataset, leading to inaccuracies in evaluating nuanced or partially correct answers. To address this challenge, we propose a novel framework, Grade Guard.
1. To enhance the task-based specialization of the LLMs, the temperature parameter has been fine-tuned using Root Mean Square Error (RMSE).
2. Unlike traditional approaches, LLMs in Grade Guard compute an Indecisiveness Score (IS) along with the grade to reflect uncertainty in predicted grades.
3. Introduced Confidence-Aware Loss (CAL) to generate an optimized Indecisiveness Score (IS).
4. To improve reliability, self-reflection based on the optimized IS has been introduced into the framework, enabling human re-evaluation to minimize incorrect grade assignments.
Our experimentation shows that the best setting of Grade Guard outperforms traditional methods by 19.16% RMSE in Upstage Solar Pro, 23.64% RMSE in Upstage Solar Mini, 4.00% RMSE in Gemini 1.5 Flash, and 10.20% RMSE in GPT 4-o Mini. Future work includes improving interpretability by generating rationales for grades to enhance accuracy. Expanding benchmark datasets and annotating them with domain-specific nuances will enhance grading accuracy. Finally, analyzing feedback to enhance confidence in predicted grades, reduce biases, optimize grading criteria, and personalize learning while supporting multilingual grading systems will make the solution more accurate, adaptable, fair, and inclusive. 

---
# An Illusion of Progress? Assessing the Current State of Web Agents 

**Authors**: Tianci Xue, Weijian Qi, Tianneng Shi, Chan Hee Song, Boyu Gou, Dawn Song, Huan Sun, Yu Su  

**Link**: [PDF](https://arxiv.org/pdf/2504.01382)  

**Abstract**: As digitalization and cloud technologies evolve, the web is becoming increasingly important in the modern society. Autonomous web agents based on large language models (LLMs) hold a great potential in work automation. It is therefore important to accurately measure and monitor the progression of their capabilities. In this work, we conduct a comprehensive and rigorous assessment of the current state of web agents. Our results depict a very different picture of the competency of current agents, suggesting over-optimism in previously reported results. This gap can be attributed to shortcomings in existing benchmarks. We introduce Online-Mind2Web, an online evaluation benchmark consisting of 300 diverse and realistic tasks spanning 136 websites. It enables us to evaluate web agents under a setting that approximates how real users use these agents. To facilitate more scalable evaluation and development, we also develop a novel LLM-as-a-Judge automatic evaluation method and show that it can achieve around 85% agreement with human judgment, substantially higher than existing methods. Finally, we present the first comprehensive comparative analysis of current web agents, highlighting both their strengths and limitations to inspire future research. 

---
