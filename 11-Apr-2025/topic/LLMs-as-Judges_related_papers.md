# Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge 

**Authors**: Riccardo Cantini, Alessio Orsino, Massimo Ruggiero, Domenico Talia  

**Link**: [PDF](https://arxiv.org/pdf/2504.07887)  

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence, driving advancements in machine translation, summarization, and conversational agents. However, their increasing integration into critical societal domains has raised concerns about embedded biases, which can perpetuate stereotypes and compromise fairness. These biases stem from various sources, including historical inequalities in training data, linguistic imbalances, and adversarial manipulation. Despite mitigation efforts, recent studies indicate that LLMs remain vulnerable to adversarial attacks designed to elicit biased responses. This work proposes a scalable benchmarking framework to evaluate LLM robustness against adversarial bias elicitation. Our methodology involves (i) systematically probing models with a multi-task approach targeting biases across various sociocultural dimensions, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach for automated assessment of model responses, and (iii) employing jailbreak techniques to investigate vulnerabilities in safety mechanisms. Our analysis examines prevalent biases in both small and large state-of-the-art models and their impact on model safety. Additionally, we assess the safety of domain-specific models fine-tuned for critical fields, such as medicine. Finally, we release a curated dataset of bias-related prompts, CLEAR-Bias, to facilitate systematic vulnerability benchmarking. Our findings reveal critical trade-offs between model size and safety, aiding the development of fairer and more robust future language models. 

---
# AgentAda: Skill-Adaptive Data Analytics for Tailored Insight Discovery 

**Authors**: Amirhossein Abaskohi, Amrutha Varshini Ramesh, Shailesh Nanisetty, Chirag Goel, David Vazquez, Christopher Pal, Spandana Gella, Giuseppe Carenini, Issam H. Laradji  

**Link**: [PDF](https://arxiv.org/pdf/2504.07421)  

**Abstract**: We introduce AgentAda, the first LLM-powered analytics agent that can learn and use new analytics skills to extract more specialized insights. Unlike existing methods that require users to manually decide which data analytics method to apply, AgentAda automatically identifies the skill needed from a library of analytical skills to perform the analysis. This also allows AgentAda to use skills that existing LLMs cannot perform out of the box. The library covers a range of methods, including clustering, predictive modeling, and NLP techniques like BERT, which allow AgentAda to handle complex analytics tasks based on what the user needs. AgentAda's dataset-to-insight extraction strategy consists of three key steps: (I) a question generator to generate queries relevant to the user's goal and persona, (II) a hybrid Retrieval-Augmented Generation (RAG)-based skill matcher to choose the best data analytics skill from the skill library, and (III) a code generator that produces executable code based on the retrieved skill's documentation to extract key patterns. We also introduce KaggleBench, a benchmark of curated notebooks across diverse domains, to evaluate AgentAda's performance. We conducted a human evaluation demonstrating that AgentAda provides more insightful analytics than existing tools, with 48.78% of evaluators preferring its analyses, compared to 27.67% for the unskilled agent. We also propose a novel LLM-as-a-judge approach that we show is aligned with human evaluation as a way to automate insight quality evaluation at larger scale. 

---
# Revisiting LLM Evaluation through Mechanism Interpretability: a New Metric and Model Utility Law 

**Authors**: Yixin Cao, Jiahao Ying, Yaoning Wang, Xipeng Qiu, Xuanjing Huang, Yugang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07440)  

**Abstract**: Large Language Models (LLMs) have become indispensable across academia, industry, and daily applications, yet current evaluation methods struggle to keep pace with their rapid development. In this paper, we analyze the core limitations of traditional evaluation pipelines and propose a novel metric, the Model Utilization Index (MUI), which introduces mechanism interpretability techniques to complement traditional performance metrics. MUI quantifies the extent to which a model leverages its capabilities to complete tasks. The core idea is that to assess an LLM's overall ability, we must evaluate not only its task performance but also the effort expended to achieve the outcome. Our extensive experiments reveal an inverse relationship between MUI and performance, from which we deduce a common trend observed in popular LLMs, which we term the Utility Law. Based on this, we derive four corollaries that address key challenges, including training judgement, the issue of data contamination, fairness in model comparison, and data diversity. We hope that our survey, novel metric, and utility law will foster mutual advancement in both evaluation and mechanism interpretability. Our code can be found at this https URL. 

---
# TALE: A Tool-Augmented Framework for Reference-Free Evaluation of Large Language Models 

**Authors**: Sher Badshah, Ali Emami, Hassan Sajjad  

**Link**: [PDF](https://arxiv.org/pdf/2504.07385)  

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into real-world, autonomous applications, relying on static, pre-annotated references for evaluation poses significant challenges in cost, scalability, and completeness. We propose Tool-Augmented LLM Evaluation (TALE), a framework to assess LLM outputs without predetermined ground-truth answers. Unlike conventional metrics that compare to fixed references or depend solely on LLM-as-a-judge knowledge, TALE employs an agent with tool-access capabilities that actively retrieves and synthesizes external evidence. It iteratively generates web queries, collects information, summarizes findings, and refines subsequent searches through reflection. By shifting away from static references, TALE aligns with free-form question-answering tasks common in real-world scenarios. Experimental results on multiple free-form QA benchmarks show that TALE not only outperforms standard reference-based metrics for measuring response accuracy but also achieves substantial to near-perfect agreement with human evaluations. TALE enhances the reliability of LLM evaluations in real-world, dynamic scenarios without relying on static references. 

---
# HypoEval: Hypothesis-Guided Evaluation for Natural Language Generation 

**Authors**: Mingxuan Li, Hanchen Li, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.07174)  

**Abstract**: Large language models (LLMs) have demonstrated great potential for automating the evaluation of natural language generation. Previous frameworks of LLM-as-a-judge fall short in two ways: they either use zero-shot setting without consulting any human input, which leads to low alignment, or fine-tune LLMs on labeled data, which requires a non-trivial number of samples. Moreover, previous methods often provide little reasoning behind automated evaluations. In this paper, we propose HypoEval, Hypothesis-guided Evaluation framework, which first uses a small corpus of human evaluations to generate more detailed rubrics for human judgments and then incorporates a checklist-like approach to combine LLM's assigned scores on each decomposed dimension to acquire overall scores. With only 30 human evaluations, HypoEval achieves state-of-the-art performance in alignment with both human rankings (Spearman correlation) and human scores (Pearson correlation), on average outperforming G-Eval by 11.86% and fine-tuned Llama-3.1-8B-Instruct with at least 3 times more human evaluations by 11.95%. Furthermore, we conduct systematic studies to assess the robustness of HypoEval, highlighting its effectiveness as a reliable and interpretable automated evaluation framework. 

---
