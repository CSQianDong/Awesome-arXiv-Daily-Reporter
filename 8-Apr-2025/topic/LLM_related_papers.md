# Post-Training Language Models for Continual Relation Extraction 

**Authors**: Sefika Efeoglu, Adrian Paschke, Sonja Schimmler  

**Link**: [PDF](https://arxiv.org/pdf/2504.05214)  

**Abstract**: Real-world data, such as news articles, social media posts, and chatbot conversations, is inherently dynamic and non-stationary, presenting significant challenges for constructing real-time structured representations through knowledge graphs (KGs). Relation Extraction (RE), a fundamental component of KG creation, often struggles to adapt to evolving data when traditional models rely on static, outdated datasets. Continual Relation Extraction (CRE) methods tackle this issue by incrementally learning new relations while preserving previously acquired knowledge. This study investigates the application of pre-trained language models (PLMs), specifically large language models (LLMs), to CRE, with a focus on leveraging memory replay to address catastrophic forgetting. We evaluate decoder-only models (eg, Mistral-7B and Llama2-7B) and encoder-decoder models (eg, Flan-T5 Base) on the TACRED and FewRel datasets. Task-incremental fine-tuning of LLMs demonstrates superior performance over earlier approaches using encoder-only models like BERT on TACRED, excelling in seen-task accuracy and overall performance (measured by whole and average accuracy), particularly with the Mistral and Flan-T5 models. Results on FewRel are similarly promising, achieving second place in whole and average accuracy metrics. This work underscores critical factors in knowledge transfer, language model architecture, and KG completeness, advancing CRE with LLMs and memory replay for dynamic, real-time relation extraction. 

---
# Do PhD-level LLMs Truly Grasp Elementary Addition? Probing Rule Learning vs. Memorization in Large Language Models 

**Authors**: Yang Yan, Yu Lu, Renjun Xu, Zhenzhong Lan  

**Link**: [PDF](https://arxiv.org/pdf/2504.05262)  

**Abstract**: Despite high benchmark scores, Large Language Models (LLMs) often fail simple problem, raising a critical question: Do LLMs learn mathematical principles or merely memorize patterns? Rather than designing increasingly complex benchmarks like recent works, we investigate this using elementary two-integer addition ($0$ to $2^{64}$), probing two core properties: commutativity ($A+B=B+A$) and compositional generalization (via isomorphic symbolic mappings, e.g., $7 \rightarrow y$). While state-of-the-art LLMs achieve 73.8-99.8\% accuracy on numerical addition, performance collapses to $\leq$7.5\% under symbolic mapping, indicating failure to generalize learned rules. Non-monotonic performance scaling with digit count and frequent commutativity violations (over 1,700 cases of $A+B \neq B+A$) further support this. Explicitly providing addition rules degrades performance by 81.2\% on average, while self-explanation maintains baseline accuracy, suggesting LLM arithmetic processing is misaligned with human-defined principles. Our findings indicate current LLMs rely on memory pattern over genuine rule learning, highlighting architectural limitations and the need for new approaches to achieve true mathematical reasoning. 

---
# Truthful or Fabricated? Using Causal Attribution to Mitigate Reward Hacking in Explanations 

**Authors**: Pedro Ferreira, Wilker Aziz, Ivan Titov  

**Link**: [PDF](https://arxiv.org/pdf/2504.05294)  

**Abstract**: Chain-of-thought explanations are widely used to inspect the decision process of large language models (LLMs) and to evaluate the trustworthiness of model outputs, making them important for effective collaboration between LLMs and humans. We demonstrate that preference optimization - a key step in the alignment phase - can inadvertently reduce the faithfulness of these explanations. This occurs because the reward model (RM), which guides alignment, is tasked with optimizing both the expected quality of the response and the appropriateness of the explanations (e.g., minimizing bias or adhering to safety standards), creating potential conflicts. The RM lacks a mechanism to assess the consistency between the model's internal decision process and the generated explanation. Consequently, the LLM may engage in "reward hacking" by producing a final response that scores highly while giving an explanation tailored to maximize reward rather than accurately reflecting its reasoning. To address this issue, we propose enriching the RM's input with a causal attribution of the prediction, allowing the RM to detect discrepancies between the generated self-explanation and the model's decision process. In controlled settings, we show that this approach reduces the tendency of the LLM to generate misleading explanations. 

---
# Enhancing LLM-Based Short Answer Grading with Retrieval-Augmented Generation 

**Authors**: Yucheng Chu, Peng He, Hang Li, Haoyu Han, Kaiqi Yang, Yu Xue, Tingting Li, Joseph Krajcik, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05276)  

**Abstract**: Short answer assessment is a vital component of science education, allowing evaluation of students' complex three-dimensional understanding. Large language models (LLMs) that possess human-like ability in linguistic tasks are increasingly popular in assisting human graders to reduce their workload. However, LLMs' limitations in domain knowledge restrict their understanding in task-specific requirements and hinder their ability to achieve satisfactory performance. Retrieval-augmented generation (RAG) emerges as a promising solution by enabling LLMs to access relevant domain-specific knowledge during assessment. In this work, we propose an adaptive RAG framework for automated grading that dynamically retrieves and incorporates domain-specific knowledge based on the question and student answer context. Our approach combines semantic search and curated educational sources to retrieve valuable reference materials. Experimental results in a science education dataset demonstrate that our system achieves an improvement in grading accuracy compared to baseline LLM approaches. The findings suggest that RAG-enhanced grading systems can serve as reliable support with efficient performance gains. 

---
# Concise Reasoning via Reinforcement Learning 

**Authors**: Mehdi Fatemi, Banafsheh Rafiee, Mingjie Tang, Kartik Talamadupula  

**Link**: [PDF](https://arxiv.org/pdf/2504.05185)  

**Abstract**: Despite significant advancements in large language models (LLMs), a major drawback of reasoning models is their enormous token usage, which increases computational cost, resource requirements, and response time. In this work, we revisit the core principles of reinforcement learning (RL) and, through mathematical analysis, demonstrate that the tendency to generate lengthy responses arises inherently from RL-based optimization during training. This finding questions the prevailing assumption that longer responses inherently improve reasoning accuracy. Instead, we uncover a natural correlation between conciseness and accuracy that has been largely overlooked. Moreover, we show that introducing a secondary phase of RL post-training, using a small set of problems and limited resources, can significantly reduce a model's chain of thought while maintaining or even enhancing accuracy. Finally, we validate our conclusions through extensive experimental results. 

---
# LLM-based Automated Grading with Human-in-the-Loop 

**Authors**: Hang Li, Yucheng Chu, Kaiqi Yang, Yasemin Copur-Gencturk, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05239)  

**Abstract**: The rise of artificial intelligence (AI) technologies, particularly large language models (LLMs), has brought significant advancements to the field of education. Among various applications, automatic short answer grading (ASAG), which focuses on evaluating open-ended textual responses, has seen remarkable progress with the introduction of LLMs. These models not only enhance grading performance compared to traditional ASAG approaches but also move beyond simple comparisons with predefined "golden" answers, enabling more sophisticated grading scenarios, such as rubric-based evaluation. However, existing LLM-powered methods still face challenges in achieving human-level grading performance in rubric-based assessments due to their reliance on fully automated approaches. In this work, we explore the potential of LLMs in ASAG tasks by leveraging their interactive capabilities through a human-in-the-loop (HITL) approach. Our proposed framework, GradeHITL, utilizes the generative properties of LLMs to pose questions to human experts, incorporating their insights to refine grading rubrics dynamically. This adaptive process significantly improves grading accuracy, outperforming existing methods and bringing ASAG closer to human-level evaluation. 

---
# AI for Climate Finance: Agentic Retrieval and Multi-Step Reasoning for Early Warning System Investments 

**Authors**: Saeid Ario Vaghefi, Aymane Hachcham, Veronica Grasso, Jiska Manicus, Nakiete Msemo, Chiara Colesanti Senni, Markus Leippold  

**Link**: [PDF](https://arxiv.org/pdf/2504.05104)  

**Abstract**: Tracking financial investments in climate adaptation is a complex and expertise-intensive task, particularly for Early Warning Systems (EWS), which lack standardized financial reporting across multilateral development banks (MDBs) and funds. To address this challenge, we introduce an LLM-based agentic AI system that integrates contextual retrieval, fine-tuning, and multi-step reasoning to extract relevant financial data, classify investments, and ensure compliance with funding guidelines. Our study focuses on a real-world application: tracking EWS investments in the Climate Risk and Early Warning Systems (CREWS) Fund. We analyze 25 MDB project documents and evaluate multiple AI-driven classification methods, including zero-shot and few-shot learning, fine-tuned transformer-based classifiers, chain-of-thought (CoT) prompting, and an agent-based retrieval-augmented generation (RAG) approach. Our results show that the agent-based RAG approach significantly outperforms other methods, achieving 87\% accuracy, 89\% precision, and 83\% recall. Additionally, we contribute a benchmark dataset and expert-annotated corpus, providing a valuable resource for future research in AI-driven financial tracking and climate finance transparency. 

---
# DoCIA: An Online Document-Level Context Incorporation Agent for Speech Translation 

**Authors**: Xinglin Lyu, Wei Tang, Yuang Li, Xiaofeng Zhao, Ming Zhu, Junhui Li, Yunfei Lu, Min Zhang, Daimeng Wei, Hao Yang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05122)  

**Abstract**: Document-level context is crucial for handling discourse challenges in text-to-text document-level machine translation (MT). Despite the increased discourse challenges introduced by noise from automatic speech recognition (ASR), the integration of document-level context in speech translation (ST) remains insufficiently explored. In this paper, we develop DoCIA, an online framework that enhances ST performance by incorporating document-level context. DoCIA decomposes the ST pipeline into four stages. Document-level context is integrated into the ASR refinement, MT, and MT refinement stages through auxiliary LLM (large language model)-based modules. Furthermore, DoCIA leverages document-level information in a multi-level manner while minimizing computational overhead. Additionally, a simple yet effective determination mechanism is introduced to prevent hallucinations from excessive refinement, ensuring the reliability of the final results. Experimental results show that DoCIA significantly outperforms traditional ST baselines in both sentence and discourse metrics across four LLMs, demonstrating its effectiveness in improving ST performance. 

---
# Not All Data Are Unlearned Equally 

**Authors**: Aravind Krishnan, Siva Reddy, Marius Mosbach  

**Link**: [PDF](https://arxiv.org/pdf/2504.05058)  

**Abstract**: Machine unlearning is concerned with the task of removing knowledge learned from particular data points from a trained model. In the context of large language models (LLMs), unlearning has recently received increased attention, particularly for removing knowledge about named entities from models for privacy purposes. While various approaches have been proposed to address the unlearning problem, most existing approaches treat all data points to be unlearned equally, i.e., unlearning that Montreal is a city in Canada is treated exactly the same as unlearning the phone number of the first author of this paper. In this work, we show that this all data is equal assumption does not hold for LLM unlearning. We study how the success of unlearning depends on the frequency of the knowledge we want to unlearn in the pre-training data of a model and find that frequency strongly affects unlearning, i.e., more frequent knowledge is harder to unlearn. Additionally, we uncover a misalignment between probability and generation-based evaluations of unlearning and show that this problem worsens as models become larger. Overall, our experiments highlight the need for better evaluation practices and novel methods for LLM unlearning that take the training data of models into account. 

---
# On the Performance of an Explainable Language Model on PubMedQA 

**Authors**: Venkat Srinivasan, Vishaal Jatav, Anushka Chandrababu, Geetika Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2504.05074)  

**Abstract**: Large language models (LLMs) have shown significant abilities in retrieving medical knowledge, reasoning over it and answering medical questions comparably to physicians. However, these models are not interpretable, hallucinate, are difficult to maintain and require enormous compute resources for training and inference. In this paper, we report results from Gyan, an explainable language model based on an alternative architecture, on the PubmedQA data set. The Gyan LLM is a compositional language model and the model is decoupled from knowledge. Gyan is trustable, transparent, does not hallucinate and does not require significant training or compute resources. Gyan is easily transferable across domains. Gyan-4.3 achieves SOTA results on PubmedQA with 87.1% accuracy compared to 82% by MedPrompt based on GPT-4 and 81.8% by Med-PaLM 2 (Google and DeepMind). We will be reporting results for other medical data sets - MedQA, MedMCQA, MMLU - Medicine in the future. 

---
# The Curse of CoT: On the Limitations of Chain-of-Thought in In-Context Learning 

**Authors**: Tianshi Zheng, Yixiang Chen, Chengxi Li, Chunyang Li, Qing Zong, Haochen Shi, Baixuan Xu, Yangqiu Song, Ginny Y. Wong, Simon See  

**Link**: [PDF](https://arxiv.org/pdf/2504.05081)  

**Abstract**: Chain-of-Thought (CoT) prompting has been widely recognized for its ability to enhance reasoning capabilities in large language models (LLMs) through the generation of explicit explanatory rationales. However, our study reveals a surprising contradiction to this prevailing perspective. Through extensive experiments involving 16 state-of-the-art LLMs and nine diverse pattern-based in-context learning (ICL) datasets, we demonstrate that CoT and its reasoning variants consistently underperform direct answering across varying model scales and benchmark complexities. To systematically investigate this unexpected phenomenon, we designed extensive experiments to validate several hypothetical explanations. Our analysis uncovers a fundamental explicit-implicit duality driving CoT's performance in pattern-based ICL: while explicit reasoning falters due to LLMs' struggles to infer underlying patterns from demonstrations, implicit reasoning-disrupted by the increased contextual distance of CoT rationales-often compensates, delivering correct answers despite flawed rationales. This duality explains CoT's relative underperformance, as noise from weak explicit inference undermines the process, even as implicit mechanisms partially salvage outcomes. Notably, even long-CoT reasoning models, which excel in abstract and symbolic reasoning, fail to fully overcome these limitations despite higher computational costs. Our findings challenge existing assumptions regarding the universal efficacy of CoT, yielding novel insights into its limitations and guiding future research toward more nuanced and effective reasoning methodologies for LLMs. 

---
# Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models 

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau  

**Link**: [PDF](https://arxiv.org/pdf/2504.05050)  

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities. 

---
# Collab-RAG: Boosting Retrieval-Augmented Generation for Complex Question Answering via White-Box and Black-Box LLM Collaboration 

**Authors**: Ran Xu, Wenqi Shi, Yuchen Zhuang, Yue Yu, Joyce C. Ho, Haoyu Wang, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04915)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems often struggle to handle multi-hop question-answering tasks accurately due to irrelevant context retrieval and limited complex reasoning capabilities. We introduce Collab-RAG, a collaborative training framework that leverages mutual enhancement between a white-box small language model (SLM) and a blackbox large language model (LLM) for RAG. Specifically, the SLM decomposes complex queries into simpler sub-questions, thus enhancing the accuracy of the retrieval and facilitating more effective reasoning by the black-box LLM. Concurrently, the black-box LLM provides feedback signals to improve the SLM's decomposition capability. We observe that Collab-RAG relies solely on supervision from an affordable black-box LLM without additional distillation from frontier LLMs, yet demonstrates strong generalization across multiple black-box LLMs. Experimental evaluations across five multi-hop QA datasets demonstrate that Collab-RAG substantially outperforms existing black-box-only and SLM fine-tuning baselines by 1.8%-14.2% on average. In particular, our fine-tuned 3B SLM surpasses a frozen 32B LLM in question decomposition, highlighting the efficiency of Collab-RAG in improving reasoning and retrieval for complex questions. The code of Collab-RAG is available on this https URL. 

---
# M-Prometheus: A Suite of Open Multilingual LLM Judges 

**Authors**: José Pombal, Dongkeun Yoon, Patrick Fernandes, Ian Wu, Seungone Kim, Ricardo Rei, Graham Neubig, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2504.04953)  

**Abstract**: The use of language models for automatically evaluating long-form text (LLM-as-a-judge) is becoming increasingly common, yet most LLM judges are optimized exclusively for English, with strategies for enhancing their multilingual evaluation capabilities remaining largely unexplored in the current literature. This has created a disparity in the quality of automatic evaluation methods for non-English languages, ultimately hindering the development of models with better multilingual capabilities. To bridge this gap, we introduce M-Prometheus, a suite of open-weight LLM judges ranging from 3B to 14B parameters that can provide both direct assessment and pairwise comparison feedback on multilingual outputs. M-Prometheus models outperform state-of-the-art open LLM judges on multilingual reward benchmarks spanning more than 20 languages, as well as on literary machine translation (MT) evaluation covering 4 language pairs. Furthermore, M-Prometheus models can be leveraged at decoding time to significantly improve generated outputs across all 3 tested languages, showcasing their utility for the development of better multilingual models. Lastly, through extensive ablations, we identify the key factors for obtaining an effective multilingual judge, including backbone model selection and training on natively multilingual feedback data instead of translated data. We release our models, training dataset, and code. 

---
# CARE: Aligning Language Models for Regional Cultural Awareness 

**Authors**: Geyang Guo, Tarek Naous, Hiromi Wakaki, Yukiko Nishimura, Yuki Mitsufuji, Alan Ritter, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05154)  

**Abstract**: Existing language models (LMs) often exhibit a Western-centric bias and struggle to represent diverse cultural knowledge. Previous attempts to address this rely on synthetic data and express cultural knowledge only in English. In this work, we study whether a small amount of human-written, multilingual cultural preference data can improve LMs across various model families and sizes. We first introduce CARE, a multilingual resource of 24.1k responses with human preferences on 2,580 questions about Chinese and Arab cultures, all carefully annotated by native speakers and offering more balanced coverage. Using CARE, we demonstrate that cultural alignment improves existing LMs beyond generic resources without compromising general capabilities. Moreover, we evaluate the cultural awareness of LMs, native speakers, and retrieved web content when queried in different languages. Our experiment reveals regional disparities among LMs, which may also be reflected in the documentation gap: native speakers often take everyday cultural commonsense and social norms for granted, while non-natives are more likely to actively seek out and document them. CARE is publicly available at this https URL (we plan to add Japanese data in the near future). 

---
# Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models 

**Authors**: Ruikang Liu, Yuxuan Sun, Manyi Zhang, Haoli Bai, Xianzhi Yu, Tiezheng Yu, Chun Yuan, Lu Hou  

**Link**: [PDF](https://arxiv.org/pdf/2504.04823)  

**Abstract**: Recent advancements in reasoning language models have demonstrated remarkable performance in complex tasks, but their extended chain-of-thought reasoning process increases inference overhead. While quantization has been widely adopted to reduce the inference cost of large language models, its impact on reasoning models remains understudied. In this study, we conduct the first systematic study on quantized reasoning models, evaluating the open-sourced DeepSeek-R1-Distilled Qwen and LLaMA families ranging from 1.5B to 70B parameters, and QwQ-32B. Our investigation covers weight, KV cache, and activation quantization using state-of-the-art algorithms at varying bit-widths, with extensive evaluation across mathematical (AIME, MATH-500), scientific (GPQA), and programming (LiveCodeBench) reasoning benchmarks. Our findings reveal that while lossless quantization can be achieved with W8A8 or W4A16 quantization, lower bit-widths introduce significant accuracy risks. We further identify model size, model origin, and task difficulty as critical determinants of performance. Contrary to expectations, quantized models do not exhibit increased output lengths. In addition, strategically scaling the model sizes or reasoning steps can effectively enhance the performance. All quantized models and codes will be open-sourced in this https URL. 

---
# Improving Multilingual Retrieval-Augmented Language Models through Dialectic Reasoning Argumentations 

**Authors**: Leonardo Ranaldi, Federico Ranaldi, Fabio Massimo Zanzotto, Barry Haddow, Alexandra Birch  

**Link**: [PDF](https://arxiv.org/pdf/2504.04771)  

**Abstract**: Retrieval-augmented generation (RAG) is key to enhancing large language models (LLMs) to systematically access richer factual knowledge. Yet, using RAG brings intrinsic challenges, as LLMs must deal with potentially conflicting knowledge, especially in multilingual retrieval, where the heterogeneity of knowledge retrieved may deliver different outlooks. To make RAG more analytical, critical and grounded, we introduce Dialectic-RAG (DRAG), a modular approach guided by Argumentative Explanations, i.e., structured reasoning process that systematically evaluates retrieved
information by comparing, contrasting, and resolving conflicting perspectives. Given a query and a set of multilingual related documents, DRAG selects and exemplifies relevant knowledge for delivering dialectic explanations that, by critically weighing opposing arguments and filtering extraneous content, clearly determine the final response. Through a series of in-depth experiments, we show the impact of our framework both as an in-context learning strategy and for constructing demonstrations to instruct smaller models. The final results demonstrate that DRAG significantly improves RAG approaches, requiring low-impact computational effort and providing robustness to knowledge perturbations. 

---
# Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs 

**Authors**: Ankush Raut, Xiaofeng Zhu, Maria Leonor Pacheco  

**Link**: [PDF](https://arxiv.org/pdf/2504.04745)  

**Abstract**: This paper evaluates the ability of Large Language Models (LLMs) to leverage contextual information in the form of structured linguistic representations. Specifically, we examine the impact of encoding both short and long contexts using Abstract Meaning Representation (AMR) structures across a diverse set of language tasks. We perform our analysis using 8-bit quantized and instruction-tuned versions of Llama 3.1 (8B), Phi-3, and Mistral 7B. Our results indicate that, for tasks involving short contexts, augmenting the prompt with the AMR of the original language context often degrades the performance of the underlying LLM. However, for tasks that involve long contexts, such as dialogue summarization in the SAMSum dataset, this enhancement improves LLM performance, for example, by increasing the zero-shot cosine similarity score of Llama 3.1 from 66.2% to 76%. This improvement is more evident in the newer and larger LLMs, but does not extend to the older or smaller ones. In addition, we observe that LLMs can effectively reconstruct the original text from a linearized AMR, achieving a cosine similarity of 81.3% in the best-case scenario. 

---
# Surveying Professional Writers on AI: Limitations, Expectations, and Fears 

**Authors**: Anastasiia Ivanova, Natalia Fedorova, Sergey Tilga, Ekaterina Artemova  

**Link**: [PDF](https://arxiv.org/pdf/2504.05008)  

**Abstract**: The rapid development of AI-driven tools, particularly large language models (LLMs), is reshaping professional writing. Still, key aspects of their adoption such as languages support, ethics, and long-term impact on writers voice and creativity remain underexplored. In this work, we conducted a questionnaire (N = 301) and an interactive survey (N = 36) targeting professional writers regularly using AI. We examined LLM-assisted writing practices across 25+ languages, ethical concerns, and user expectations. The findings of the survey demonstrate important insights, reflecting upon the importance of: LLMs adoption for non-English speakers; the degree of misinformation, domain and style adaptation; usability and key features of LLMs. These insights can guide further development, benefiting both writers and a broader user base. 

---
# Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs 

**Authors**: Will Cai, Tianneng Shi, Xuandong Zhao, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.04715)  

**Abstract**: The proliferation of Large Language Models (LLMs) accessed via black-box APIs introduces a significant trust challenge: users pay for services based on advertised model capabilities (e.g., size, performance), but providers may covertly substitute the specified model with a cheaper, lower-quality alternative to reduce operational costs. This lack of transparency undermines fairness, erodes trust, and complicates reliable benchmarking. Detecting such substitutions is difficult due to the black-box nature, typically limiting interaction to input-output queries. This paper formalizes the problem of model substitution detection in LLM APIs. We systematically evaluate existing verification techniques, including output-based statistical tests, benchmark evaluations, and log probability analysis, under various realistic attack scenarios like model quantization, randomized substitution, and benchmark evasion. Our findings reveal the limitations of methods relying solely on text outputs, especially against subtle or adaptive attacks. While log probability analysis offers stronger guarantees when available, its accessibility is often limited. We conclude by discussing the potential of hardware-based solutions like Trusted Execution Environments (TEEs) as a pathway towards provable model integrity, highlighting the trade-offs between security, performance, and provider adoption. Code is available at this https URL 

---
# Leveraging Large Language Models for Cost-Effective, Multilingual Depression Detection and Severity Assessment 

**Authors**: Longdi Xian, Jianzhang Ni, Mingzhu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04891)  

**Abstract**: Depression is a prevalent mental health disorder that is difficult to detect early due to subjective symptom assessments. Recent advancements in large language models have offered efficient and cost-effective approaches for this objective. In this study, we evaluated the performance of four LLMs in depression detection using clinical interview data. We selected the best performing model and further tested it in the severity evaluation scenario and knowledge enhanced scenario. The robustness was evaluated in complex diagnostic scenarios using a dataset comprising 51074 statements from six different mental disorders. We found that DeepSeek V3 is the most reliable and cost-effective model for depression detection, performing well in both zero-shot and few-shot scenarios, with zero-shot being the most efficient choice. The evaluation of severity showed low agreement with the human evaluator, particularly for mild depression. The model maintains stably high AUCs for detecting depression in complex diagnostic scenarios. These findings highlight DeepSeek V3s strong potential for text-based depression detection in real-world clinical applications. However, they also underscore the need for further refinement in severity assessment and the mitigation of potential biases to enhance clinical reliability. 

---
# Sequential-NIAH: A Needle-In-A-Haystack Benchmark for Extracting Sequential Needles from Long Contexts 

**Authors**: Yifei Yu, Qian-Wen Zhang, Lingfeng Qiao, Di Yin, Fang Li, Jie Wang, Zengxi Chen, Suncong Zheng, Xiaolong Liang, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.04713)  

**Abstract**: Evaluating the ability of large language models (LLMs) to handle extended contexts is critical, particularly for retrieving information relevant to specific queries embedded within lengthy inputs. We introduce Sequential-NIAH, a benchmark specifically designed to evaluate the capability of LLMs to extract sequential information items (known as needles) from long contexts. The benchmark comprises three types of needle generation pipelines: synthetic, real, and open-domain QA. It includes contexts ranging from 8K to 128K tokens in length, with a dataset of 14,000 samples (2,000 reserved for testing). To facilitate evaluation on this benchmark, we trained a synthetic data-driven evaluation model capable of evaluating answer correctness based on chronological or logical order, achieving an accuracy of 99.49% on synthetic test data. We conducted experiments on six well-known LLMs, revealing that even the best-performing model achieved a maximum accuracy of only 63.15%. Further analysis highlights the growing challenges posed by increasing context lengths and the number of needles, underscoring substantial room for improvement. Additionally, noise robustness experiments validate the reliability of the benchmark, making Sequential-NIAH an important reference for advancing research on long text extraction capabilities of LLMs. 

---
# Beyond Single-Turn: A Survey on Multi-Turn Interactions with Large Language Models 

**Authors**: Yubo Li, Xiaobin Shen, Xinyu Yao, Xueying Ding, Yidi Miao, Ramayya Krishnan, Rema Padman  

**Link**: [PDF](https://arxiv.org/pdf/2504.04717)  

**Abstract**: Recent advancements in large language models (LLMs) have revolutionized their ability to handle single-turn tasks, yet real-world applications demand sophisticated multi-turn interactions. This survey provides a comprehensive review of recent advancements in evaluating and enhancing multi-turn interactions in LLMs. Focusing on task-specific scenarios, from instruction following in diverse domains such as math and coding to complex conversational engagements in roleplay, healthcare, education, and even adversarial jailbreak settings, we systematically examine the challenges of maintaining context, coherence, fairness, and responsiveness over prolonged dialogues. The paper organizes current benchmarks and datasets into coherent categories that reflect the evolving landscape of multi-turn dialogue evaluation. In addition, we review a range of enhancement methodologies under multi-turn settings, including model-centric strategies (contextual learning, supervised fine-tuning, reinforcement learning, and new architectures), external integration approaches (memory-augmented, retrieval-based methods, and knowledge graph), and agent-based techniques for collaborative interactions. Finally, we discuss open challenges and propose future directions for research to further advance the robustness and effectiveness of multi-turn interactions in LLMs. Related resources and papers are available at this https URL. 

---
# scAgent: Universal Single-Cell Annotation via a LLM Agent 

**Authors**: Yuren Mao, Yu Mi, Peigen Liu, Mengfei Zhang, Hanqing Liu, Yunjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.04698)  

**Abstract**: Cell type annotation is critical for understanding cellular heterogeneity. Based on single-cell RNA-seq data and deep learning models, good progress has been made in annotating a fixed number of cell types within a specific tissue. However, universal cell annotation, which can generalize across tissues, discover novel cell types, and extend to novel cell types, remains less explored. To fill this gap, this paper proposes scAgent, a universal cell annotation framework based on Large Language Models (LLMs). scAgent can identify cell types and discover novel cell types in diverse tissues; furthermore, it is data efficient to learn novel cell types. Experimental studies in 160 cell types and 35 tissues demonstrate the superior performance of scAgent in general cell-type annotation, novel cell discovery, and extensibility to novel cell type. 

---
# KnowsLM: A framework for evaluation of small language models for knowledge augmentation and humanised conversations 

**Authors**: Chitranshu Harbola, Anupam Purwar  

**Link**: [PDF](https://arxiv.org/pdf/2504.04569)  

**Abstract**: In the evolving landscape of conversational AI, generating concise, context-aware, and human-like dialogue using small and medium-sized language models (LLMs) remains a complex challenge. This study investigates the influence of LoRA rank, dataset scale, and prompt prefix design on both knowledge retention and stylistic alignment. While fine-tuning improves fluency and enables stylistic customization, its ability to integrate unseen knowledge is constrained -- particularly with smaller datasets. Conversely, RAG-augmented models, equipped to incorporate external documents at inference, demonstrated superior factual accuracy on out-of-distribution prompts, though they lacked the stylistic consistency achieved by fine-tuning. Evaluations by LLM-based judges across knowledge accuracy, conversational quality, and conciseness suggest that fine-tuning is best suited for tone adaptation, whereas RAG excels at real-time knowledge augmentation. 

---
# An Empirical Comparison of Text Summarization: A Multi-Dimensional Evaluation of Large Language Models 

**Authors**: Anantharaman Janakiraman, Behnaz Ghoraani  

**Link**: [PDF](https://arxiv.org/pdf/2504.04534)  

**Abstract**: Text summarization is crucial for mitigating information overload across domains like journalism, medicine, and business. This research evaluates summarization performance across 17 large language models (OpenAI, Google, Anthropic, open-source) using a novel multi-dimensional framework. We assessed models on seven diverse datasets (BigPatent, BillSum, CNN/DailyMail, PubMed, SAMSum, WikiHow, XSum) at three output lengths (50, 100, 150 tokens) using metrics for factual consistency, semantic similarity, lexical overlap, and human-like quality, while also considering efficiency factors. Our findings reveal significant performance differences, with specific models excelling in factual accuracy (deepseek-v3), human-like quality (claude-3-5-sonnet), and processing efficiency/cost-effectiveness (gemini-1.5-flash, gemini-2.0-flash). Performance varies dramatically by dataset, with models struggling on technical domains but performing well on conversational content. We identified a critical tension between factual consistency (best at 50 tokens) and perceived quality (best at 150 tokens). Our analysis provides evidence-based recommendations for different use cases, from high-stakes applications requiring factual accuracy to resource-constrained environments needing efficient processing. This comprehensive approach enhances evaluation methodology by integrating quality metrics with operational considerations, incorporating trade-offs between accuracy, efficiency, and cost-effectiveness to guide model selection for specific applications. 

---
# An overview of model uncertainty and variability in LLM-based sentiment analysis. Challenges, mitigation strategies and the role of explainability 

**Authors**: David Herrera-Poyatos, Carlos Peláez-González, Cristina Zuheros, Andrés Herrera-Poyatos, Virilo Tejedor, Francisco Herrera, Rosana Montes  

**Link**: [PDF](https://arxiv.org/pdf/2504.04462)  

**Abstract**: Large Language Models (LLMs) have significantly advanced sentiment analysis, yet their inherent uncertainty and variability pose critical challenges to achieving reliable and consistent outcomes. This paper systematically explores the Model Variability Problem (MVP) in LLM-based sentiment analysis, characterized by inconsistent sentiment classification, polarization, and uncertainty arising from stochastic inference mechanisms, prompt sensitivity, and biases in training data. We analyze the core causes of MVP, presenting illustrative examples and a case study to highlight its impact. In addition, we investigate key challenges and mitigation strategies, paying particular attention to the role of temperature as a driver of output randomness and emphasizing the crucial role of explainability in improving transparency and user trust. By providing a structured perspective on stability, reproducibility, and trustworthiness, this study helps develop more reliable, explainable, and robust sentiment analysis models, facilitating their deployment in high-stakes domains such as finance, healthcare, and policymaking, among others. 

---
# TathyaNyaya and FactLegalLlama: Advancing Factual Judgment Prediction and Explanation in the Indian Legal Context 

**Authors**: Shubham Kumar Nigam, Balaramamahanthi Deepak Patnaik, Shivam Mishra, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2504.04737)  

**Abstract**: In the landscape of Fact-based Judgment Prediction and Explanation (FJPE), reliance on factual data is essential for developing robust and realistic AI-driven decision-making tools. This paper introduces TathyaNyaya, the largest annotated dataset for FJPE tailored to the Indian legal context, encompassing judgments from the Supreme Court of India and various High Courts. Derived from the Hindi terms "Tathya" (fact) and "Nyaya" (justice), the TathyaNyaya dataset is uniquely designed to focus on factual statements rather than complete legal texts, reflecting real-world judicial processes where factual data drives outcomes. Complementing this dataset, we present FactLegalLlama, an instruction-tuned variant of the LLaMa-3-8B Large Language Model (LLM), optimized for generating high-quality explanations in FJPE tasks. Finetuned on the factual data in TathyaNyaya, FactLegalLlama integrates predictive accuracy with coherent, contextually relevant explanations, addressing the critical need for transparency and interpretability in AI-assisted legal systems. Our methodology combines transformers for binary judgment prediction with FactLegalLlama for explanation generation, creating a robust framework for advancing FJPE in the Indian legal domain. TathyaNyaya not only surpasses existing datasets in scale and diversity but also establishes a benchmark for building explainable AI systems in legal analysis. The findings underscore the importance of factual precision and domain-specific tuning in enhancing predictive performance and interpretability, positioning TathyaNyaya and FactLegalLlama as foundational resources for AI-assisted legal decision-making. 

---
# Saliency-driven Dynamic Token Pruning for Large Language Models 

**Authors**: Yao Tao, Yehui Tang, Yun Wang, Mingjian Zhu, Hailin Hu, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04514)  

**Abstract**: Despite the recent success of large language models (LLMs), LLMs are particularly challenging in long-sequence inference scenarios due to the quadratic computational complexity of the attention mechanism. Inspired by the interpretability theory of feature attribution in neural network models, we observe that not all tokens have the same contribution. Based on this observation, we propose a novel token pruning framework, namely Saliency-driven Dynamic Token Pruning (SDTP), to gradually and dynamically prune redundant tokens based on the input context. Specifically, a lightweight saliency-driven prediction module is designed to estimate the importance score of each token with its hidden state, which is added to different layers of the LLM to hierarchically prune redundant tokens. Furthermore, a ranking-based optimization strategy is proposed to minimize the ranking divergence of the saliency score and the predicted importance score. Extensive experiments have shown that our framework is generalizable to various models and datasets. By hierarchically pruning 65\% of the input tokens, our method greatly reduces 33\% $\sim$ 47\% FLOPs and achieves speedup up to 1.75$\times$ during inference, while maintaining comparable performance. We further demonstrate that SDTP can be combined with KV cache compression method for further compression. 

---
# PolyGuard: A Multilingual Safety Moderation Tool for 17 Languages 

**Authors**: Priyanshu Kumar, Devansh Jain, Akhila Yerukola, Liwei Jiang, Himanshu Beniwal, Thomas Hartvigsen, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2504.04377)  

**Abstract**: Truly multilingual safety moderation efforts for Large Language Models (LLMs) have been hindered by a narrow focus on a small set of languages (e.g., English, Chinese) as well as a limited scope of safety definition, resulting in significant gaps in moderation capabilities. To bridge these gaps, we release POLYGUARD, a new state-of-the-art multilingual safety model for safeguarding LLM generations, and the corresponding training and evaluation datasets. POLYGUARD is trained on POLYGUARDMIX, the largest multilingual safety training corpus to date containing 1.91M samples across 17 languages (e.g., Chinese, Czech, English, Hindi). We also introduce POLYGUARDPROMPTS, a high quality multilingual benchmark with 29K samples for the evaluation of safety guardrails. Created by combining naturally occurring multilingual human-LLM interactions and human-verified machine translations of an English-only safety dataset (WildGuardMix; Han et al., 2024), our datasets contain prompt-output pairs with labels of prompt harmfulness, response harmfulness, and response refusal. Through extensive evaluations across multiple safety and toxicity benchmarks, we demonstrate that POLYGUARD outperforms existing state-of-the-art open-weight and commercial safety classifiers by 5.5%. Our contributions advance efforts toward safer multilingual LLMs for all global users. 

---
# Generative Large Language Models Trained for Detecting Errors in Radiology Reports 

**Authors**: Cong Sun, Kurt Teichman, Yiliang Zhou, Brian Critelli, David Nauheim, Graham Keir, Xindi Wang, Judy Zhong, Adam E Flanders, George Shih, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2504.04336)  

**Abstract**: In this retrospective study, a dataset was constructed with two parts. The first part included 1,656 synthetic chest radiology reports generated by GPT-4 using specified prompts, with 828 being error-free synthetic reports and 828 containing errors. The second part included 614 reports: 307 error-free reports between 2011 and 2016 from the MIMIC-CXR database and 307 corresponding synthetic reports with errors generated by GPT-4 on the basis of these MIMIC-CXR reports and specified prompts. All errors were categorized into four types: negation, left/right, interval change, and transcription errors. Then, several models, including Llama-3, GPT-4, and BiomedBERT, were refined using zero-shot prompting, few-shot prompting, or fine-tuning strategies. Finally, the performance of these models was evaluated using the F1 score, 95\% confidence interval (CI) and paired-sample t-tests on our constructed dataset, with the prediction results further assessed by radiologists. Using zero-shot prompting, the fine-tuned Llama-3-70B-Instruct model achieved the best performance with the following F1 scores: 0.769 for negation errors, 0.772 for left/right errors, 0.750 for interval change errors, 0.828 for transcription errors, and 0.780 overall. In the real-world evaluation phase, two radiologists reviewed 200 randomly selected reports output by the model. Of these, 99 were confirmed to contain errors detected by the models by both radiologists, and 163 were confirmed to contain model-detected errors by at least one radiologist. Generative LLMs, fine-tuned on synthetic and MIMIC-CXR radiology reports, greatly enhanced error detection in radiology reports. 

---
# Causal Retrieval with Semantic Consideration 

**Authors**: Hyunseo Shin, Wonseok Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04700)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly enhanced the performance of conversational AI systems. To extend their capabilities to knowledge-intensive domains such as biomedical and legal fields, where the accuracy is critical, LLMs are often combined with information retrieval (IR) systems to generate responses based on retrieved documents. However, for IR systems to effectively support such applications, they must go beyond simple semantic matching and accurately capture diverse query intents, including causal relationships. Existing IR models primarily focus on retrieving documents based on surface-level semantic similarity, overlooking deeper relational structures such as causality. To address this, we propose CAWAI, a retrieval model that is trained with dual objectives: semantic and causal relations. Our extensive experiments demonstrate that CAWAI outperforms various models on diverse causal retrieval tasks especially under large-scale retrieval settings. We also show that CAWAI exhibits strong zero-shot generalization across scientific domain QA tasks. 

---
# IMPersona: Evaluating Individual Level LM Impersonation 

**Authors**: Quan Shi, Carlos Jimenez, Stephen Dong, Brian Seo, Caden Yao, Adam Kelch, Karthik Narasimhan  

**Link**: [PDF](https://arxiv.org/pdf/2504.04332)  

**Abstract**: As language models achieve increasingly human-like capabilities in conversational text generation, a critical question emerges: to what extent can these systems simulate the characteristics of specific individuals? To evaluate this, we introduce IMPersona, a framework for evaluating LMs at impersonating specific individuals' writing style and personal knowledge. Using supervised fine-tuning and a hierarchical memory-inspired retrieval system, we demonstrate that even modestly sized open-source models, such as Llama-3.1-8B-Instruct, can achieve impersonation abilities at concerning levels. In blind conversation experiments, participants (mis)identified our fine-tuned models with memory integration as human in 44.44% of interactions, compared to just 25.00% for the best prompting-based approach. We analyze these results to propose detection methods and defense strategies against such impersonation attempts. Our findings raise important questions about both the potential applications and risks of personalized language models, particularly regarding privacy, security, and the ethical deployment of such technologies in real-world contexts. 

---
# Balancing Complexity and Informativeness in LLM-Based Clustering: Finding the Goldilocks Zone 

**Authors**: Justin Miller, Tristram Alexander  

**Link**: [PDF](https://arxiv.org/pdf/2504.04314)  

**Abstract**: The challenge of clustering short text data lies in balancing informativeness with interpretability. Traditional evaluation metrics often overlook this trade-off. Inspired by linguistic principles of communicative efficiency, this paper investigates the optimal number of clusters by quantifying the trade-off between informativeness and cognitive simplicity. We use large language models (LLMs) to generate cluster names and evaluate their effectiveness through semantic density, information theory, and clustering accuracy. Our results show that Gaussian Mixture Model (GMM) clustering on embeddings generated by a LLM, increases semantic density compared to random assignment, effectively grouping similar bios. However, as clusters increase, interpretability declines, as measured by a generative LLM's ability to correctly assign bios based on cluster names. A logistic regression analysis confirms that classification accuracy depends on the semantic similarity between bios and their assigned cluster names, as well as their distinction from alternatives.
These findings reveal a "Goldilocks zone" where clusters remain distinct yet interpretable. We identify an optimal range of 16-22 clusters, paralleling linguistic efficiency in lexical categorization. These insights inform both theoretical models and practical applications, guiding future research toward optimising cluster interpretability and usefulness. 

---
# StyleRec: A Benchmark Dataset for Prompt Recovery in Writing Style Transformation 

**Authors**: Shenyang Liu, Yang Gao, Shaoyan Zhai, Liqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04373)  

**Abstract**: Prompt Recovery, reconstructing prompts from the outputs of large language models (LLMs), has grown in importance as LLMs become ubiquitous. Most users access LLMs through APIs without internal model weights, relying only on outputs and logits, which complicates recovery. This paper explores a unique prompt recovery task focused on reconstructing prompts for style transfer and rephrasing, rather than typical question-answering. We introduce a dataset created with LLM assistance, ensuring quality through multiple techniques, and test methods like zero-shot, few-shot, jailbreak, chain-of-thought, fine-tuning, and a novel canonical-prompt fallback for poor-performing cases. Our results show that one-shot and fine-tuning yield the best outcomes but highlight flaws in traditional sentence similarity metrics for evaluating prompt recovery. Contributions include (1) a benchmark dataset, (2) comprehensive experiments on prompt recovery strategies, and (3) identification of limitations in current evaluation metrics, all of which advance general prompt recovery research, where the structure of the input prompt is unrestricted. 

---
# Lost in Multilinguality: Dissecting Cross-lingual Factual Inconsistency in Transformer Language Models 

**Authors**: Mingyang Wang, Heike Adel, Lukas Lange, Yihong Liu, Ercong Nie, Jannik Strötgen, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2504.04264)  

**Abstract**: Multilingual language models (MLMs) store factual knowledge across languages but often struggle to provide consistent responses to semantically equivalent prompts in different languages. While previous studies point out this cross-lingual inconsistency issue, the underlying causes remain unexplored. In this work, we use mechanistic interpretability methods to investigate cross-lingual inconsistencies in MLMs. We find that MLMs encode knowledge in a language-independent concept space through most layers, and only transition to language-specific spaces in the final layers. Failures during the language transition often result in incorrect predictions in the target language, even when the answers are correct in other languages. To mitigate this inconsistency issue, we propose a linear shortcut method that bypasses computations in the final layers, enhancing both prediction accuracy and cross-lingual consistency. Our findings shed light on the internal mechanisms of MLMs and provide a lightweight, effective strategy for producing more consistent factual outputs. 

---
# Dynamic Hedging Strategies in Derivatives Markets with LLM-Driven Sentiment and News Analytics 

**Authors**: Jie Yang, Yiqiu Tang, Yongjie Li, Lihua Zhang, Haoran Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04295)  

**Abstract**: Dynamic hedging strategies are essential for effective risk management in derivatives markets, where volatility and market sentiment can greatly impact performance. This paper introduces a novel framework that leverages large language models (LLMs) for sentiment analysis and news analytics to inform hedging decisions. By analyzing textual data from diverse sources like news articles, social media, and financial reports, our approach captures critical sentiment indicators that reflect current market conditions. The framework allows for real-time adjustments to hedging strategies, adapting positions based on continuous sentiment signals. Backtesting results on historical derivatives data reveal that our dynamic hedging strategies achieve superior risk-adjusted returns compared to conventional static approaches. The incorporation of LLM-driven sentiment analysis into hedging practices presents a significant advancement in decision-making processes within derivatives trading. This research showcases how sentiment-informed dynamic hedging can enhance portfolio management and effectively mitigate associated risks. 

---
# Sensitivity Meets Sparsity: The Impact of Extremely Sparse Parameter Patterns on Theory-of-Mind of Large Language Models 

**Authors**: Yuheng Wu, Wentao Guo, Zirui Liu, Heng Ji, Zhaozhuo Xu, Denghui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04238)  

**Abstract**: This paper investigates the emergence of Theory-of-Mind (ToM) capabilities in large language models (LLMs) from a mechanistic perspective, focusing on the role of extremely sparse parameter patterns. We introduce a novel method to identify ToM-sensitive parameters and reveal that perturbing as little as 0.001% of these parameters significantly degrades ToM performance while also impairing contextual localization and language understanding. To understand this effect, we analyze their interaction with core architectural components of LLMs. Our findings demonstrate that these sensitive parameters are closely linked to the positional encoding module, particularly in models using Rotary Position Embedding (RoPE), where perturbations disrupt dominant-frequency activations critical for contextual processing. Furthermore, we show that perturbing ToM-sensitive parameters affects LLM's attention mechanism by modulating the angle between queries and keys under positional encoding. These insights provide a deeper understanding of how LLMs acquire social reasoning abilities, bridging AI interpretability with cognitive science. Our results have implications for enhancing model alignment, mitigating biases, and improving AI systems designed for human interaction. 

---
# A Perplexity and Menger Curvature-Based Approach for Similarity Evaluation of Large Language Models 

**Authors**: Yuantao Zhang, Zhankui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04216)  

**Abstract**: The rise of Large Language Models (LLMs) has brought about concerns regarding copyright infringement and unethical practices in data and model usage. For instance, slight modifications to existing LLMs may be used to falsely claim the development of new models, leading to issues of model copying and violations of ownership rights. This paper addresses these challenges by introducing a novel metric for quantifying LLM similarity, which leverages perplexity curves and differences in Menger curvature. Comprehensive experiments validate the performance of our methodology, demonstrating its superiority over baseline methods and its ability to generalize across diverse models and domains. Furthermore, we highlight the capability of our approach in detecting model replication through simulations, emphasizing its potential to preserve the originality and integrity of LLMs. Code is available at this https URL. 

---
# Pre-trained Language Models and Few-shot Learning for Medical Entity Extraction 

**Authors**: Xiaokai Wang, Guiran Liu, Binrong Zhu, Jacky He, Hongye Zheng, Hanlu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04385)  

**Abstract**: This study proposes a medical entity extraction method based on Transformer to enhance the information extraction capability of medical literature. Considering the professionalism and complexity of medical texts, we compare the performance of different pre-trained language models (BERT, BioBERT, PubMedBERT, ClinicalBERT) in medical entity extraction tasks. Experimental results show that PubMedBERT achieves the best performance (F1-score = 88.8%), indicating that a language model pre-trained on biomedical literature is more effective in the medical domain. In addition, we analyze the impact of different entity extraction methods (CRF, Span-based, Seq2Seq) and find that the Span-based approach performs best in medical entity extraction tasks (F1-score = 88.6%). It demonstrates superior accuracy in identifying entity boundaries. In low-resource scenarios, we further explore the application of Few-shot Learning in medical entity extraction. Experimental results show that even with only 10-shot training samples, the model achieves an F1-score of 79.1%, verifying the effectiveness of Few-shot Learning under limited data conditions. This study confirms that the combination of pre-trained language models and Few-shot Learning can enhance the accuracy of medical entity extraction. Future research can integrate knowledge graphs and active learning strategies to improve the model's generalization and stability, providing a more effective solution for medical NLP research. Keywords- Natural Language Processing, medical named entity recognition, pre-trained language model, Few-shot Learning, information extraction, deep learning 

---
# Adaptive Elicitation of Latent Information Using Natural Language 

**Authors**: Jimmy Wang, Thomas Zollo, Richard Zemel, Hongseok Namkoong  

**Link**: [PDF](https://arxiv.org/pdf/2504.04204)  

**Abstract**: Eliciting information to reduce uncertainty about a latent entity is a critical task in many application domains, e.g., assessing individual student learning outcomes, diagnosing underlying diseases, or learning user preferences. Though natural language is a powerful medium for this purpose, large language models (LLMs) and existing fine-tuning algorithms lack mechanisms for strategically gathering information to refine their own understanding of the latent entity. To harness the generalization power and world knowledge of LLMs in developing effective information-gathering strategies, we propose an adaptive elicitation framework that actively reduces uncertainty on the latent entity. Since probabilistic modeling of an abstract latent entity is difficult, our framework adopts a predictive view of uncertainty, using a meta-learned language model to simulate future observations and enable scalable uncertainty quantification over complex natural language. Through autoregressive forward simulation, our model quantifies how new questions reduce epistemic uncertainty, enabling the development of sophisticated information-gathering strategies to choose the most informative next queries. In experiments on the 20 questions game, dynamic opinion polling, and adaptive student assessment, our method consistently outperforms baselines in identifying critical unknowns and improving downstream predictions, illustrating the promise of strategic information gathering in natural language settings. 

---
# Rethinking Multilingual Continual Pretraining: Data Mixing for Adapting LLMs Across Languages and Resources 

**Authors**: Zihao Li, Shaoxiong Ji, Hengyu Luo, Jörg Tiedemann  

**Link**: [PDF](https://arxiv.org/pdf/2504.04152)  

**Abstract**: Large Language Models (LLMs) exhibit significant disparities in performance across languages, primarily benefiting high-resource languages while marginalizing underrepresented ones. Continual Pretraining (CPT) has emerged as a promising approach to address this imbalance, although the relative effectiveness of monolingual, bilingual, and code-augmented data strategies remains unclear. This study systematically evaluates 36 CPT configurations involving three multilingual base models, across 30+ languages categorized as altruistic, selfish, and stagnant, spanning various resource levels. Our findings reveal three major insights: (1) Bilingual CPT improves multilingual classification but often causes language mixing issues during generation. (2) Including programming code data during CPT consistently enhances multilingual classification accuracy, particularly benefiting low-resource languages, but introduces a trade-off by slightly degrading generation quality. (3) Contrary to prior work, we observe substantial deviations from language classifications according to their impact on cross-lingual transfer: Languages classified as altruistic often negatively affect related languages, selfish languages show conditional and configuration-dependent behavior, and stagnant languages demonstrate surprising adaptability under certain CPT conditions. These nuanced interactions emphasize the complexity of multilingual representation learning, underscoring the importance of systematic studies on generalizable language classification to inform future multilingual CPT strategies. 

---
# STEP: Staged Parameter-Efficient Pre-training for Large Language Models 

**Authors**: Kazuki Yano, Takumi Ito, Jun Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2504.04151)  

**Abstract**: Pre-training large language models (LLMs) faces significant memory challenges due to the large size of model parameters. We introduce STaged parameter-Efficient Pre-training (STEP), which integrates parameter-efficient tuning techniques with model growth. We conduct experiments on pre-training LLMs of various sizes and demonstrate that STEP achieves up to a 53.9% reduction in maximum memory requirements compared to vanilla pre-training while maintaining equivalent performance. Furthermore, we show that the model by STEP performs comparably to vanilla pre-trained models on downstream tasks after instruction tuning. 

---
# GlotEval: A Test Suite for Massively Multilingual Evaluation of Large Language Models 

**Authors**: Hengyu Luo, Zihao Li, Joseph Attieh, Sawal Devkota, Ona de Gibert, Shaoxiong Ji, Peiqin Lin, Bhavani Sai Praneeth Varma Mantina, Ananda Sreenidhi, Raúl Vázquez, Mengjie Wang, Samea Yusofi, Jörg Tiedemann  

**Link**: [PDF](https://arxiv.org/pdf/2504.04155)  

**Abstract**: Large language models (LLMs) are advancing at an unprecedented pace globally, with regions increasingly adopting these models for applications in their primary language. Evaluation of these models in diverse linguistic environments, especially in low-resource languages, has become a major challenge for academia and industry. Existing evaluation frameworks are disproportionately focused on English and a handful of high-resource languages, thereby overlooking the realistic performance of LLMs in multilingual and lower-resource scenarios. To address this gap, we introduce GlotEval, a lightweight framework designed for massively multilingual evaluation. Supporting seven key tasks (machine translation, text classification, summarization, open-ended generation, reading comprehension, sequence labeling, and intrinsic evaluation), spanning over dozens to hundreds of languages, GlotEval highlights consistent multilingual benchmarking, language-specific prompt templates, and non-English-centric machine translation. This enables a precise diagnosis of model strengths and weaknesses in diverse linguistic contexts. A multilingual translation case study demonstrates GlotEval's applicability for multilingual and language-specific evaluations. 

---
# Reasoning on Multiple Needles In A Haystack 

**Authors**: Yidong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04150)  

**Abstract**: The Needle In A Haystack (NIAH) task has been widely used to evaluate the long-context question-answering capabilities of Large Language Models (LLMs). However, its reliance on simple retrieval limits its effectiveness. To address this limitation, recent studies have introduced the Multiple Needles In A Haystack Reasoning (MNIAH-R) task, which incorporates supporting documents (Multiple needles) of multi-hop reasoning tasks into a distracting context (Haystack}). Despite this advancement, existing approaches still fail to address the issue of models providing direct answers from internal knowledge, and they do not explain or mitigate the decline in accuracy as context length increases. In this paper, we tackle the memory-based answering problem by filtering out direct-answer questions, and we reveal that performance degradation is primarily driven by the reduction in the length of the thinking process as the input length increases. Building on this insight, we decompose the thinking process into retrieval and reasoning stages and introduce a reflection mechanism for multi-round extension. We also train a model using the generated iterative thinking process, which helps mitigate the performance degradation. Furthermore, we demonstrate the application of this retrieval-reflection capability in mathematical reasoning scenarios, improving GPT-4o's performance on AIME2024. 

---
# Cognitive Debiasing Large Language Models for Decision-Making 

**Authors**: Yougang Lyu, Shijie Ren, Yue Feng, Zihan Wang, Zhumin Chen, Zhaochun Ren, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2504.04141)  

**Abstract**: Large language models (LLMs) have shown potential in supporting decision-making applications, particularly as personal conversational assistants in the financial, healthcare, and legal domains. While prompt engineering strategies have enhanced the capabilities of LLMs in decision-making, cognitive biases inherent to LLMs present significant challenges. Cognitive biases are systematic patterns of deviation from norms or rationality in decision-making that can lead to the production of inaccurate outputs. Existing cognitive bias mitigation strategies assume that input prompts contain (exactly) one type of cognitive bias and therefore fail to perform well in realistic settings where there maybe any number of biases.
To fill this gap, we propose a cognitive debiasing approach, called self-debiasing, that enhances the reliability of LLMs by iteratively refining prompts. Our method follows three sequential steps -- bias determination, bias analysis, and cognitive debiasing -- to iteratively mitigate potential cognitive biases in prompts. Experimental results on finance, healthcare, and legal decision-making tasks, using both closed-source and open-source LLMs, demonstrate that the proposed self-debiasing method outperforms both advanced prompt engineering methods and existing cognitive debiasing techniques in average accuracy under no-bias, single-bias, and multi-bias settings. 

---
# A Benchmark for End-to-End Zero-Shot Biomedical Relation Extraction with LLMs: Experiments with OpenAI Models 

**Authors**: Aviv Brokman, Xuguang Ai, Yuhang Jiang, Shashank Gupta, Ramakanth Kavuluru  

**Link**: [PDF](https://arxiv.org/pdf/2504.04083)  

**Abstract**: Objective: Zero-shot methodology promises to cut down on costs of dataset annotation and domain expertise needed to make use of NLP. Generative large language models trained to align with human goals have achieved high zero-shot performance across a wide variety of tasks. As of yet, it is unclear how well these models perform on biomedical relation extraction (RE). To address this knowledge gap, we explore patterns in the performance of OpenAI LLMs across a diverse sampling of RE tasks.
Methods: We use OpenAI GPT-4-turbo and their reasoning model o1 to conduct end-to-end RE experiments on seven datasets. We use the JSON generation capabilities of GPT models to generate structured output in two ways: (1) by defining an explicit schema describing the structure of relations, and (2) using a setting that infers the structure from the prompt language.
Results: Our work is the first to study and compare the performance of the GPT-4 and o1 for the end-to-end zero-shot biomedical RE task across a broad array of datasets. We found the zero-shot performances to be proximal to that of fine-tuned methods. The limitations of this approach are that it performs poorly on instances containing many relations and errs on the boundaries of textual mentions.
Conclusion: Recent large language models exhibit promising zero-shot capabilities in complex biomedical RE tasks, offering competitive performance with reduced dataset curation and NLP modeling needs at the cost of increased computing, potentially increasing medical community accessibility. Addressing the limitations we identify could further boost reliability. The code, data, and prompts for all our experiments are publicly available: this https URL 

---
# VocalNet: Speech LLM with Multi-Token Prediction for Faster and High-Quality Generation 

**Authors**: Yuhao Wang, Heyang Liu, Ziyang Cheng, Ronghua Wu, Qunshan Gu, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04060)  

**Abstract**: Speech large language models (LLMs) have emerged as a prominent research focus in speech processing. We propose VocalNet-1B and VocalNet-8B, a series of high-performance, low-latency speech LLMs enabled by a scalable and model-agnostic training framework for real-time voice interaction. Departing from the conventional next-token prediction (NTP), we introduce multi-token prediction (MTP), a novel approach optimized for speech LLMs that simultaneously improves generation speed and quality. Experiments show that VocalNet outperforms mainstream Omni LLMs despite using significantly less training data, while also surpassing existing open-source speech LLMs by a substantial margin. To support reproducibility and community advancement, we will open-source all model weights, inference code, training data, and framework implementations upon publication. 

---
# Cross-Asset Risk Management: Integrating LLMs for Real-Time Monitoring of Equity, Fixed Income, and Currency Markets 

**Authors**: Jie Yang, Yiqiu Tang, Yongjie Li, Lihua Zhang, Haoran Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04292)  

**Abstract**: Large language models (LLMs) have emerged as powerful tools in the field of finance, particularly for risk management across different asset classes. In this work, we introduce a Cross-Asset Risk Management framework that utilizes LLMs to facilitate real-time monitoring of equity, fixed income, and currency markets. This innovative approach enables dynamic risk assessment by aggregating diverse data sources, ultimately enhancing decision-making processes. Our model effectively synthesizes and analyzes market signals to identify potential risks and opportunities while providing a holistic view of asset classes. By employing advanced analytics, we leverage LLMs to interpret financial texts, news articles, and market reports, ensuring that risks are contextualized within broader market narratives. Extensive backtesting and real-time simulations validate the framework, showing increased accuracy in predicting market shifts compared to conventional methods. The focus on real-time data integration enhances responsiveness, allowing financial institutions to manage risks adeptly under varying market conditions and promoting financial stability through the advanced application of LLMs in risk analysis. 

---
# SyLeR: A Framework for Explicit Syllogistic Legal Reasoning in Large Language Models 

**Authors**: Kepu Zhang, Weijie Yu, Zhongxiang Sun, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04042)  

**Abstract**: Syllogistic reasoning is a fundamental aspect of legal decision-making, enabling logical conclusions by connecting general legal principles with specific case facts. Although existing large language models (LLMs) can generate responses to legal questions, they fail to perform explicit syllogistic reasoning, often producing implicit and unstructured answers that lack explainability and trustworthiness. To address this limitation, we propose SyLeR, a novel framework that empowers LLMs to engage in explicit syllogistic legal reasoning. SyLeR integrates a tree-structured hierarchical retrieval mechanism to effectively combine relevant legal statutes and precedent cases, forming comprehensive major premises. This is followed by a two-stage fine-tuning process: supervised fine-tuning warm-up establishes a foundational understanding of syllogistic reasoning, while reinforcement learning with a structure-aware reward mechanism refines the ability of the model to generate diverse logically sound and well-structured reasoning paths. We conducted extensive experiments across various dimensions, including in-domain and cross-domain user groups (legal laypersons and practitioners), multiple languages (Chinese and French), and different LLM backbones (legal-specific and open-domain LLMs). The results show that SyLeR significantly improves response accuracy and consistently delivers explicit, explainable, and trustworthy legal reasoning. 

---
# Algorithmic Prompt Generation for Diverse Human-like Teaming and Communication with Large Language Models 

**Authors**: Siddharth Srikanth, Varun Bhatt, Boshen Zhang, Werner Hager, Charles Michael Lewis, Katia P. Sycara, Aaquib Tabrez, Stefanos Nikolaidis  

**Link**: [PDF](https://arxiv.org/pdf/2504.03991)  

**Abstract**: Understanding how humans collaborate and communicate in teams is essential for improving human-agent teaming and AI-assisted decision-making. However, relying solely on data from large-scale user studies is impractical due to logistical, ethical, and practical constraints, necessitating synthetic models of multiple diverse human behaviors. Recently, agents powered by Large Language Models (LLMs) have been shown to emulate human-like behavior in social settings. But, obtaining a large set of diverse behaviors requires manual effort in the form of designing prompts. On the other hand, Quality Diversity (QD) optimization has been shown to be capable of generating diverse Reinforcement Learning (RL) agent behavior. In this work, we combine QD optimization with LLM-powered agents to iteratively search for prompts that generate diverse team behavior in a long-horizon, multi-step collaborative environment. We first show, through a human-subjects experiment (n=54 participants), that humans exhibit diverse coordination and communication behavior in this domain. We then show that our approach can effectively replicate trends from human teaming data and also capture behaviors that are not easily observed without collecting large amounts of data. Our findings highlight the combination of QD and LLM-powered agents as an effective tool for studying teaming and communication strategies in multi-agent collaboration. 

---
# Do LLM Evaluators Prefer Themselves for a Reason? 

**Authors**: Wei-Lin Chen, Zhepei Wei, Xinyu Zhu, Shi Feng, Yu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2504.03846)  

**Abstract**: Large language models (LLMs) are increasingly used as automatic evaluators in applications such as benchmarking, reward modeling, and self-refinement. Prior work highlights a potential self-preference bias where LLMs favor their own generated responses, a tendency often intensifying with model size and capability. This raises a critical question: Is self-preference detrimental, or does it simply reflect objectively superior outputs from more capable models? Disentangling these has been challenging due to the usage of subjective tasks in previous studies. To address this, we investigate self-preference using verifiable benchmarks (mathematical reasoning, factual knowledge, code generation) that allow objective ground-truth assessment. This enables us to distinguish harmful self-preference (favoring objectively worse responses) from legitimate self-preference (favoring genuinely superior ones). We conduct large-scale experiments under controlled evaluation conditions across diverse model families (e.g., Llama, Qwen, Gemma, Mistral, Phi, GPT, DeepSeek). Our findings reveal three key insights: (1) Better generators are better judges -- LLM evaluators' accuracy strongly correlates with their task performance, and much of the self-preference in capable models is legitimate. (2) Harmful self-preference persists, particularly when evaluator models perform poorly as generators on specific task instances. Stronger models exhibit more pronounced harmful bias when they err, though such incorrect generations are less frequent. (3) Inference-time scaling strategies, such as generating a long Chain-of-Thought before evaluation, effectively reduce the harmful self-preference. These results provide a more nuanced understanding of LLM-based evaluation and practical insights for improving its reliability. 

---
# Language Models Are Implicitly Continuous 

**Authors**: Samuele Marro, Davide Evangelista, X. Angelo Huang, Emanuele La Malfa, Michele Lombardi, Michael Wooldridge  

**Link**: [PDF](https://arxiv.org/pdf/2504.03933)  

**Abstract**: Language is typically modelled with discrete sequences. However, the most successful approaches to language modelling, namely neural networks, are continuous and smooth function approximators. In this work, we show that Transformer-based language models implicitly learn to represent sentences as continuous-time functions defined over a continuous input space. This phenomenon occurs in most state-of-the-art Large Language Models (LLMs), including Llama2, Llama3, Phi3, Gemma, Gemma2, and Mistral, and suggests that LLMs reason about language in ways that fundamentally differ from humans. Our work formally extends Transformers to capture the nuances of time and space continuity in both input and output space. Our results challenge the traditional interpretation of how LLMs understand language, with several linguistic and engineering implications. 

---
# Do "New Snow Tablets" Contain Snow? Large Language Models Over-Rely on Names to Identify Ingredients of Chinese Drugs 

**Authors**: Sifan Li, Yujun Cai, Bryan Hooi, Nanyun Peng, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03786)  

**Abstract**: Traditional Chinese Medicine (TCM) has seen increasing adoption in healthcare, with specialized Large Language Models (LLMs) emerging to support clinical applications. A fundamental requirement for these models is accurate identification of TCM drug ingredients. In this paper, we evaluate how general and TCM-specialized LLMs perform when identifying ingredients of Chinese drugs. Our systematic analysis reveals consistent failure patterns: models often interpret drug names literally, overuse common herbs regardless of relevance, and exhibit erratic behaviors when faced with unfamiliar formulations. LLMs also fail to understand the verification task. These findings demonstrate that current LLMs rely primarily on drug names rather than possessing systematic pharmacological knowledge. To address these limitations, we propose a Retrieval Augmented Generation (RAG) approach focused on ingredient names. Experiments across 220 TCM formulations show our method significantly improves accuracy from approximately 50% to 82% in ingredient verification tasks. Our work highlights critical weaknesses in current TCM-specific LLMs and offers a practical solution for enhancing their clinical reliability. 

---
# Leveraging LLMs for Utility-Focused Annotation: Reducing Manual Effort for Retrieval and RAG 

**Authors**: Hengran Zhang, Minghao Tang, Keping Bi, Jiafeng Guo, Shihao Liu, Daiting Shi, Dawei Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05220)  

**Abstract**: Retrieval models typically rely on costly human-labeled query-document relevance annotations for training and evaluation. To reduce this cost and leverage the potential of Large Language Models (LLMs) in relevance judgments, we aim to explore whether LLM-generated annotations can effectively replace human annotations in training retrieval models. Retrieval usually emphasizes relevance, which indicates "topic-relatedness" of a document to a query, while in RAG, the value of a document (or utility) depends on how it contributes to answer generation. Recognizing this mismatch, some researchers use LLM performance on downstream tasks with documents as labels, but this approach requires manual answers for specific tasks, leading to high costs and limited generalization. In another line of work, prompting LLMs to select useful documents as RAG references eliminates the need for human annotation and is not task-specific. If we leverage LLMs' utility judgments to annotate retrieval data, we may retain cross-task generalization without human annotation in large-scale corpora. Therefore, we investigate utility-focused annotation via LLMs for large-scale retriever training data across both in-domain and out-of-domain settings on the retrieval and RAG tasks. To reduce the impact of low-quality positives labeled by LLMs, we design a novel loss function, i.e., Disj-InfoNCE. Our experiments reveal that: (1) Retrievers trained on utility-focused annotations significantly outperform those trained on human annotations in the out-of-domain setting on both tasks, demonstrating superior generalization capabilities. (2) LLM annotation does not replace human annotation in the in-domain setting. However, incorporating just 20% human-annotated data enables retrievers trained with utility-focused annotations to match the performance of models trained entirely with human annotations. 

---
# Learning to Reason Over Time: Timeline Self-Reflection for Improved Temporal Reasoning in Language Models 

**Authors**: Adrián Bazaga, Rexhina Blloshmi, Bill Byrne, Adrià de Gispert  

**Link**: [PDF](https://arxiv.org/pdf/2504.05258)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for generating coherent text, understanding context, and performing reasoning tasks. However, they struggle with temporal reasoning, which requires processing time-related information such as event sequencing, durations, and inter-temporal relationships. These capabilities are critical for applications including question answering, scheduling, and historical analysis. In this paper, we introduce TISER, a novel framework that enhances the temporal reasoning abilities of LLMs through a multi-stage process that combines timeline construction with iterative self-reflection. Our approach leverages test-time scaling to extend the length of reasoning traces, enabling models to capture complex temporal dependencies more effectively. This strategy not only boosts reasoning accuracy but also improves the traceability of the inference process. Experimental results demonstrate state-of-the-art performance across multiple benchmarks, including out-of-distribution test sets, and reveal that TISER enables smaller open-source models to surpass larger closed-weight models on challenging temporal reasoning tasks. 

---
# Adaptation of Large Language Models 

**Authors**: Zixuan Ke, Yifei Ming, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2504.03931)  

**Abstract**: This tutorial on adaptation of LLMs is designed to address the growing demand for models that go beyond the static capabilities of generic LLMs by providing an overview of dynamic, domain-specific, and task-adaptive LLM adaptation techniques. While general LLMs have demonstrated strong generalization across a variety of tasks, they often struggle to perform well in specialized domains such as finance, healthcare, and code generation for underrepresented languages. Additionally, their static nature limits their ability to evolve with the changing world, and they are often extremely large in size, making them impractical and costly to deploy at scale. As a result, the adaptation of LLMs has drawn much attention since the birth of LLMs and is of core importance, both for industry, which focuses on serving its targeted users, and academia, which can greatly benefit from small but powerful LLMs. To address this gap, this tutorial aims to provide an overview of the LLM adaptation techniques. We start with an introduction to LLM adaptation, from both the data perspective and the model perspective. We then emphasize how the evaluation metrics and benchmarks are different from other techniques. After establishing the problems, we explore various adaptation techniques. We categorize adaptation techniques into two main families. The first is parametric knowledge adaptation, which focuses on updating the parametric knowledge within LLMs. Additionally, we will discuss real-time adaptation techniques, including model editing, which allows LLMs to be updated dynamically in production environments. The second kind of adaptation is semi-parametric knowledge adaptation, where the goal is to update LLM parameters to better leverage external knowledge or tools through techniques like retrieval-augmented generation (RAG) and agent-based systems. 

---
# YaleNLP @ PerAnsSumm 2025: Multi-Perspective Integration via Mixture-of-Agents for Enhanced Healthcare QA Summarization 

**Authors**: Dongsuk Jang, Alan Li, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2504.03932)  

**Abstract**: Automated summarization of healthcare community question-answering forums is challenging due to diverse perspectives presented across multiple user responses to each question. The PerAnsSumm Shared Task was therefore proposed to tackle this challenge by identifying perspectives from different answers and then generating a comprehensive answer to the question. In this study, we address the PerAnsSumm Shared Task using two complementary paradigms: (i) a training-based approach through QLoRA fine-tuning of LLaMA-3.3-70B-Instruct, and (ii) agentic approaches including zero- and few-shot prompting with frontier LLMs (LLaMA-3.3-70B-Instruct and GPT-4o) and a Mixture-of-Agents (MoA) framework that leverages a diverse set of LLMs by combining outputs from multi-layer feedback aggregation. For perspective span identification/classification, GPT-4o zero-shot achieves an overall score of 0.57, substantially outperforming the 0.40 score of the LLaMA baseline. With a 2-layer MoA configuration, we were able to improve LLaMA performance up by 28 percent to 0.51. For perspective-based summarization, GPT-4o zero-shot attains an overall score of 0.42 compared to 0.28 for the best LLaMA zero-shot, and our 2-layer MoA approach boosts LLaMA performance by 32 percent to 0.37. Furthermore, in few-shot setting, our results show that the sentence-transformer embedding-based exemplar selection provides more gain than manually selected exemplars on LLaMA models, although the few-shot prompting is not always helpful for GPT-4o. The YaleNLP team's approach ranked the overall second place in the shared task. 

---
# Towards Visual Text Grounding of Multimodal Large Language Model 

**Authors**: Ming Li, Ruiyi Zhang, Jian Chen, Jiuxiang Gu, Yufan Zhou, Franck Dernoncourt, Wanrong Zhu, Tianyi Zhou, Tong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.04974)  

**Abstract**: Despite the existing evolution of Multimodal Large Language Models (MLLMs), a non-neglectable limitation remains in their struggle with visual text grounding, especially in text-rich images of documents. Document images, such as scanned forms and infographics, highlight critical challenges due to their complex layouts and textual content. However, current benchmarks do not fully address these challenges, as they mostly focus on visual grounding on natural images, rather than text-rich document images. Thus, to bridge this gap, we introduce TRIG, a novel task with a newly designed instruction dataset for benchmarking and improving the Text-Rich Image Grounding capabilities of MLLMs in document question-answering. Specifically, we propose an OCR-LLM-human interaction pipeline to create 800 manually annotated question-answer pairs as a benchmark and a large-scale training set of 90$ synthetic data based on four diverse datasets. A comprehensive evaluation of various MLLMs on our proposed benchmark exposes substantial limitations in their grounding capability on text-rich images. In addition, we propose two simple and effective TRIG methods based on general instruction tuning and plug-and-play efficient embedding, respectively. By finetuning MLLMs on our synthetic dataset, they promisingly improve spatial reasoning and grounding capabilities. 

---
# LEO-MINI: An Efficient Multimodal Large Language Model using Conditional Token Reduction and Mixture of Multi-Modal Experts 

**Authors**: Yimu Wang, Mozhgan Nasr Azadani, Sean Sedwards, Krzysztof Czarnecki  

**Link**: [PDF](https://arxiv.org/pdf/2504.04653)  

**Abstract**: Redundancy of visual tokens in multi-modal large language models (MLLMs) significantly reduces their computational efficiency. Recent approaches, such as resamplers and summarizers, have sought to reduce the number of visual tokens, but at the cost of visual reasoning ability. To address this, we propose LEO-MINI, a novel MLLM that significantly reduces the number of visual tokens and simultaneously boosts visual reasoning capabilities. For efficiency, LEO-MINI incorporates CoTR, a novel token reduction module to consolidate a large number of visual tokens into a smaller set of tokens, using the similarity between visual tokens, text tokens, and a compact learnable query. For effectiveness, to scale up the model's ability with minimal computational overhead, LEO-MINI employs MMoE, a novel mixture of multi-modal experts module. MMOE employs a set of LoRA experts with a novel router to switch between them based on the input text and visual tokens instead of only using the input hidden state. MMoE also includes a general LoRA expert that is always activated to learn general knowledge for LLM reasoning. For extracting richer visual features, MMOE employs a set of vision experts trained on diverse domain-specific data. To demonstrate LEO-MINI's improved efficiency and performance, we evaluate it against existing efficient MLLMs on various benchmark vision-language tasks. 

---
# R2Vul: Learning to Reason about Software Vulnerabilities with Reinforcement Learning and Structured Reasoning Distillation 

**Authors**: Martin Weyssow, Chengran Yang, Junkai Chen, Yikun Li, Huihui Huang, Ratnadira Widyasari, Han Wei Ang, Frank Liauw, Eng Lieh Ouh, Lwin Khin Shar, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2504.04699)  

**Abstract**: Large language models (LLMs) have shown promising performance in software vulnerability detection (SVD), yet their reasoning capabilities remain unreliable. Existing approaches relying on chain-of-thought (CoT) struggle to provide relevant and actionable security assessments. Additionally, effective SVD requires not only generating coherent reasoning but also differentiating between well-founded and misleading yet plausible security assessments, an aspect overlooked in prior work. To this end, we introduce R2Vul, a novel approach that distills structured reasoning into small LLMs using reinforcement learning from AI feedback (RLAIF). Through RLAIF, R2Vul enables LLMs to produce structured, security-aware reasoning that is actionable and reliable while explicitly learning to distinguish valid assessments from misleading ones. We evaluate R2Vul across five languages against SAST tools, CoT, instruction tuning, and classification-based baselines. Our results show that R2Vul with structured reasoning distillation enables a 1.5B student LLM to rival larger models while improving generalization to out-of-distribution vulnerabilities. Beyond model improvements, we contribute a large-scale, multilingual preference dataset featuring structured reasoning to support future research in SVD. 

---
# SECQUE: A Benchmark for Evaluating Real-World Financial Analysis Capabilities 

**Authors**: Noga Ben Yoash, Meni Brief, Oded Ovadia, Gil Shenderovitz, Moshik Mishaeli, Rachel Lemberg, Eitam Sheetrit  

**Link**: [PDF](https://arxiv.org/pdf/2504.04596)  

**Abstract**: We introduce SECQUE, a comprehensive benchmark for evaluating large language models (LLMs) in financial analysis tasks. SECQUE comprises 565 expert-written questions covering SEC filings analysis across four key categories: comparison analysis, ratio calculation, risk assessment, and financial insight generation. To assess model performance, we develop SECQUE-Judge, an evaluation mechanism leveraging multiple LLM-based judges, which demonstrates strong alignment with human evaluations. Additionally, we provide an extensive analysis of various models' performance on our benchmark. By making SECQUE publicly available, we aim to facilitate further research and advancements in financial AI. 

---
# Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use 

**Authors**: Anna Goldie, Azalia Mirhoseini, Hao Zhou, Irene Cai, Christopher D. Manning  

**Link**: [PDF](https://arxiv.org/pdf/2504.04736)  

**Abstract**: Reinforcement learning has been shown to improve the performance of large language models. However, traditional approaches like RLHF or RLAIF treat the problem as single-step. As focus shifts toward more complex reasoning and agentic tasks, language models must take multiple steps of text generation, reasoning and environment interaction before generating a solution. We propose a synthetic data generation and RL methodology targeting multi-step optimization scenarios. This approach, called Step-Wise Reinforcement Learning (SWiRL), iteratively generates multi-step reasoning and tool use data, and then learns from that data. It employs a simple step-wise decomposition that breaks each multi-step trajectory into multiple sub-trajectories corresponding to each action by the original model. It then applies synthetic data filtering and RL optimization on these sub-trajectories. We evaluated SWiRL on a number of multi-step tool use, question answering, and mathematical reasoning tasks. Our experiments show that SWiRL outperforms baseline approaches by 21.5%, 12.3%, 14.8%, 11.1%, and 15.3% in relative accuracy on GSM8K, HotPotQA, CofCA, MuSiQue, and BeerQA, respectively. Excitingly, the approach exhibits generalization across tasks: for example, training only on HotPotQA (text question-answering) improves zero-shot performance on GSM8K (a math dataset) by a relative 16.9%. 

---
# A Llama walks into the 'Bar': Efficient Supervised Fine-Tuning for Legal Reasoning in the Multi-state Bar Exam 

**Authors**: Rean Fernandes, André Biedenkapp, Frank Hutter, Noor Awad  

**Link**: [PDF](https://arxiv.org/pdf/2504.04945)  

**Abstract**: Legal reasoning tasks present unique challenges for large language models (LLMs) due to the complexity of domain-specific knowledge and reasoning processes. This paper investigates how effectively smaller language models (Llama 2 7B and Llama 3 8B) can be fine-tuned with a limited dataset of 1,514 Multi-state Bar Examination (MBE) questions to improve legal question answering accuracy. We evaluate these models on the 2022 MBE questions licensed from JD Advising, the same dataset used in the 'GPT-4 passes the Bar exam' study. Our methodology involves collecting approximately 200 questions per legal domain across 7 domains. We distill the dataset using Llama 3 (70B) to transform explanations into a structured IRAC (Issue, Rule, Application, Conclusion) format as a guided reasoning process to see if it results in better performance over the non-distilled dataset. We compare the non-fine-tuned models against their supervised fine-tuned (SFT) counterparts, trained for different sample sizes per domain, to study the effect on accuracy and prompt adherence. We also analyse option selection biases and their mitigation following SFT. In addition, we consolidate the performance across multiple variables: prompt type (few-shot vs zero-shot), answer ordering (chosen-option first vs generated-explanation first), response format (Numbered list vs Markdown vs JSON), and different decoding temperatures. Our findings show that domain-specific SFT helps some model configurations achieve close to human baseline performance, despite limited computational resources and a relatively small dataset. We release both the gathered SFT dataset and the family of Supervised Fine-tuned (SFT) adapters optimised for MBE performance. This establishes a practical lower bound on resources needed towards achieving effective legal question answering in smaller LLMs. 

---
# Hessian of Perplexity for Large Language Models by PyTorch autograd (Open Source) 

**Authors**: Ivan Ilin  

**Link**: [PDF](https://arxiv.org/pdf/2504.04520)  

**Abstract**: Computing the full Hessian matrix -- the matrix of second-order derivatives for an entire Large Language Model (LLM) is infeasible due to its sheer size. In this technical report, we aim to provide a comprehensive guide on how to accurately compute at least a small portion of the Hessian for LLMs using PyTorch autograd library. We also demonstrate how to compute the full diagonal of the Hessian matrix using multiple samples of vector-Hessian Products (HVPs). We hope that both this guide and the accompanying GitHub code will be valuable resources for practitioners and researchers interested in better understanding the behavior and structure of the Hessian in LLMs. 

---
# Retro-Search: Exploring Untaken Paths for Deeper and Efficient Reasoning 

**Authors**: Ximing Lu, Seungju Han, David Acuna, Hyunwoo Kim, Jaehun Jung, Shrimai Prabhumoye, Niklas Muennighoff, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.04383)  

**Abstract**: Large reasoning models exhibit remarkable reasoning capabilities via long, elaborate reasoning trajectories. Supervised fine-tuning on such reasoning traces, also known as distillation, can be a cost-effective way to boost reasoning capabilities of student models. However, empirical observations reveal that these reasoning trajectories are often suboptimal, switching excessively between different lines of thought, resulting in under-thinking, over-thinking, and even degenerate responses. We introduce Retro-Search, an MCTS-inspired search algorithm, for distilling higher quality reasoning paths from large reasoning models. Retro-Search retrospectively revises reasoning paths to discover better, yet shorter traces, which can then lead to student models with enhanced reasoning capabilities with shorter, thus faster inference. Our approach can enable two use cases: self-improvement, where models are fine-tuned on their own Retro-Search-ed thought traces, and weak-to-strong improvement, where a weaker model revises stronger model's thought traces via Retro-Search. For self-improving, R1-distill-7B, fine-tuned on its own Retro-Search-ed traces, reduces the average reasoning length by 31.2% while improving performance by 7.7% across seven math benchmarks. For weak-to-strong improvement, we retrospectively revise R1-671B's traces from the OpenThoughts dataset using R1-distill-32B as the Retro-Search-er, a model 20x smaller. Qwen2.5-32B, fine-tuned on this refined data, achieves performance comparable to R1-distill-32B, yielding an 11.3% reduction in reasoning length and a 2.4% performance improvement compared to fine-tuning on the original OpenThoughts data. Our work counters recently emergent viewpoints that question the relevance of search algorithms in the era of large reasoning models, by demonstrating that there are still opportunities for algorithmic advancements, even for frontier models. 

---
# DDPT: Diffusion-Driven Prompt Tuning for Large Language Model Code Generation 

**Authors**: Jinyang Li, Sangwon Hyun, M. Ali Babar  

**Link**: [PDF](https://arxiv.org/pdf/2504.04351)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation. However, the quality of the generated code is heavily dependent on the structure and composition of the prompts used. Crafting high-quality prompts is a challenging task that requires significant knowledge and skills of prompt engineering. To advance the automation support for the prompt engineering for LLM-based code generation, we propose a novel solution Diffusion-Driven Prompt Tuning (DDPT) that learns how to generate optimal prompt embedding from Gaussian Noise to automate the prompt engineering for code generation. We evaluate the feasibility of diffusion-based optimization and abstract the optimal prompt embedding as a directional vector toward the optimal embedding. We use the code generation loss given by the LLMs to help the diffusion model capture the distribution of optimal prompt embedding during training. The trained diffusion model can build a path from the noise distribution to the optimal distribution at the sampling phrase, the evaluation result demonstrates that DDPT helps improve the prompt optimization for code generation. 

---
# Mixture-of-Personas Language Models for Population Simulation 

**Authors**: Ngoc Bui, Hieu Trung Nguyen, Shantanu Kumar, Julian Theodore, Weikang Qiu, Viet Anh Nguyen, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2504.05019)  

**Abstract**: Advances in Large Language Models (LLMs) paved the way for their emerging applications in various domains, such as human behavior simulations, where LLMs could augment human-generated data in social science research and machine learning model training. However, pretrained LLMs often fail to capture the behavioral diversity of target populations due to the inherent variability across individuals and groups. To address this, we propose \textit{Mixture of Personas} (MoP), a \textit{probabilistic} prompting method that aligns the LLM responses with the target population. MoP is a contextual mixture model, where each component is an LM agent characterized by a persona and an exemplar representing subpopulation behaviors. The persona and exemplar are randomly chosen according to the learned mixing weights to elicit diverse LLM responses during simulation. MoP is flexible, requires no model finetuning, and is transferable across base models. Experiments for synthetic data generation show that MoP outperforms competing methods in alignment and diversity metrics. 

---
# OpenCodeInstruct: A Large-scale Instruction Tuning Dataset for Code LLMs 

**Authors**: Wasi Uddin Ahmad, Aleksander Ficek, Mehrzad Samadi, Jocelyn Huang, Vahid Noroozi, Somshubra Majumdar, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2504.04030)  

**Abstract**: Large Language Models (LLMs) have transformed software development by enabling code generation, automated debugging, and complex reasoning. However, their continued advancement is constrained by the scarcity of high-quality, publicly available supervised fine-tuning (SFT) datasets tailored for coding tasks. To bridge this gap, we introduce OpenCodeInstruct, the largest open-access instruction tuning dataset, comprising 5 million diverse samples. Each sample includes a programming question, solution, test cases, execution feedback, and LLM-generated quality assessments. We fine-tune various base models, including LLaMA and Qwen, across multiple scales (1B+, 3B+, and 7B+) using our dataset. Comprehensive evaluations on popular benchmarks (HumanEval, MBPP, LiveCodeBench, and BigCodeBench) demonstrate substantial performance improvements achieved by SFT with OpenCodeInstruct. We also present a detailed methodology encompassing seed data curation, synthetic instruction and solution generation, and filtering. 

---
# Unleashing the Power of LLMs in Dense Retrieval with Query Likelihood Modeling 

**Authors**: Hengran Zhang, Keping Bi, Jiafeng Guo, Xiaojie Sun, Shihao Liu, Daiting Shi, Dawei Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05216)  

**Abstract**: Dense retrieval is a crucial task in Information Retrieval (IR) and is the foundation for downstream tasks such as re-ranking. Recently, large language models (LLMs) have shown compelling semantic understanding capabilities and are appealing to researchers studying dense retrieval. LLMs, as decoder-style generative models, are competent at language generation while falling short on modeling global information due to the lack of attention to tokens afterward. Inspired by the classical word-based language modeling approach for IR, i.e., the query likelihood (QL) model, we seek to sufficiently utilize LLMs' generative ability by QL maximization. However, instead of ranking documents with QL estimation, we introduce an auxiliary task of QL maximization to yield a better backbone for contrastively learning a discriminative retriever. We name our model as LLM-QL. To condense global document semantics to a single vector during QL modeling, LLM-QL has two major components, Attention Stop (AS) and Input Corruption (IC). AS stops the attention of predictive tokens to previous tokens until the ending token of the document. IC masks a portion of tokens in the input documents during prediction. Experiments on MSMARCO show that LLM-QL can achieve significantly better performance than other LLM-based retrievers and using QL estimated by LLM-QL for ranking outperforms word-based QL by a large margin. 

---
# Breach in the Shield: Unveiling the Vulnerabilities of Large Language Models 

**Authors**: Runpeng Dai, Run Yang, Fan Zhou, Hongtu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03714)  

**Abstract**: Large Language Models (LLMs) and Vision-Language Models (VLMs) have become essential to general artificial intelligence, exhibiting remarkable capabilities in task understanding and problem-solving. However, the real-world reliability of these models critically depends on their stability, which remains an underexplored area. Despite their widespread use, rigorous studies examining the stability of LLMs under various perturbations are still lacking. In this paper, we address this gap by proposing a novel stability measure for LLMs, inspired by statistical methods rooted in information geometry. Our measure possesses desirable invariance properties, making it well-suited for analyzing model sensitivity to both parameter and input perturbations. To assess the effectiveness of our approach, we conduct extensive experiments on models ranging in size from 1.5B to 13B parameters. Our results demonstrate the utility of our measure in identifying salient parameters and detecting vulnerable regions in input images or critical dimensions in token embeddings. Furthermore, leveraging our stability framework, we enhance model robustness during model merging, leading to improved performance. 

---
# PEIRCE: Unifying Material and Formal Reasoning via LLM-Driven Neuro-Symbolic Refinement 

**Authors**: Xin Quan, Marco Valentino, Danilo S. Carvalho, Dhairya Dalal, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2504.04110)  

**Abstract**: A persistent challenge in AI is the effective integration of material and formal inference - the former concerning the plausibility and contextual relevance of arguments, while the latter focusing on their logical and structural validity. Large Language Models (LLMs), by virtue of their extensive pre-training on large textual corpora, exhibit strong capabilities in material inference. However, their reasoning often lacks formal rigour and verifiability. At the same time, LLMs' linguistic competence positions them as a promising bridge between natural and formal languages, opening up new opportunities for combining these two modes of reasoning. In this paper, we introduce PEIRCE, a neuro-symbolic framework designed to unify material and formal inference through an iterative conjecture-criticism process. Within this framework, LLMs play the central role of generating candidate solutions in natural and formal languages, which are then evaluated and refined via interaction with external critique models. These critiques include symbolic provers, which assess formal validity, as well as soft evaluators that measure the quality of the generated arguments along linguistic and epistemic dimensions such as plausibility, coherence, and parsimony. While PEIRCE is a general-purpose framework, we demonstrate its capabilities in the domain of natural language explanation generation - a setting that inherently demands both material adequacy and formal correctness. 

---
# Debate Only When Necessary: Adaptive Multiagent Collaboration for Efficient LLM Reasoning 

**Authors**: Sugyeong Eo, Hyeonseok Moon, Evelyn Hayoon Zi, Chanjun Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2504.05047)  

**Abstract**: Multiagent collaboration has emerged as a promising framework for enhancing the reasoning capabilities of large language models (LLMs). While this approach improves reasoning capability, it incurs substantial computational overhead due to iterative agent interactions. Furthermore, engaging in debates for queries that do not necessitate collaboration amplifies the risk of error generation. To address these challenges, we propose Debate Only When Necessary (DOWN), an adaptive multiagent debate framework that selectively activates the debate process based on the confidence score of the agent's initial response. For queries where debate is triggered, agents refine their outputs using responses from participating agents and their confidence scores. Experimental results demonstrate that this mechanism significantly improves efficiency while maintaining or even surpassing the performance of existing multiagent debate systems. We also find that confidence-guided debate mitigates error propagation and enhances the selective incorporation of reliable responses. These results establish DOWN as an optimization strategy for efficient and effective multiagent reasoning, facilitating the practical deployment of LLM-based collaboration. 

---
# Evaluating Knowledge Graph Based Retrieval Augmented Generation Methods under Knowledge Incompleteness 

**Authors**: Dongzhuoran Zhou, Yuqicheng Zhu, Yuan He, Jiaoyan Chen, Evgeny Kharlamov, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2504.05163)  

**Abstract**: Knowledge Graph based Retrieval-Augmented Generation (KG-RAG) is a technique that enhances Large Language Model (LLM) inference in tasks like Question Answering (QA) by retrieving relevant information from knowledge graphs (KGs). However, real-world KGs are often incomplete, meaning that essential information for answering questions may be missing. Existing benchmarks do not adequately capture the impact of KG incompleteness on KG-RAG performance. In this paper, we systematically evaluate KG-RAG methods under incomplete KGs by removing triples using different methods and analyzing the resulting effects. We demonstrate that KG-RAG methods are sensitive to KG incompleteness, highlighting the need for more robust approaches in realistic settings. 

---
# The challenge of uncertainty quantification of large language models in medicine 

**Authors**: Zahra Atf, Seyed Amir Ahmad Safavi-Naini, Peter R. Lewis, Aref Mahjoubfar, Nariman Naderi, Thomas R. Savage, Ali Soroush  

**Link**: [PDF](https://arxiv.org/pdf/2504.05278)  

**Abstract**: This study investigates uncertainty quantification in large language models (LLMs) for medical applications, emphasizing both technical innovations and philosophical implications. As LLMs become integral to clinical decision-making, accurately communicating uncertainty is crucial for ensuring reliable, safe, and ethical AI-assisted healthcare. Our research frames uncertainty not as a barrier but as an essential part of knowledge that invites a dynamic and reflective approach to AI design. By integrating advanced probabilistic methods such as Bayesian inference, deep ensembles, and Monte Carlo dropout with linguistic analysis that computes predictive and semantic entropy, we propose a comprehensive framework that manages both epistemic and aleatoric uncertainties. The framework incorporates surrogate modeling to address limitations of proprietary APIs, multi-source data integration for better context, and dynamic calibration via continual and meta-learning. Explainability is embedded through uncertainty maps and confidence metrics to support user trust and clinical interpretability. Our approach supports transparent and ethical decision-making aligned with Responsible and Reflective AI principles. Philosophically, we advocate accepting controlled ambiguity instead of striving for absolute predictability, recognizing the inherent provisionality of medical knowledge. 

---
# Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning 

**Authors**: Anja Surina, Amin Mansouri, Lars Quaedvlieg, Amal Seddas, Maryna Viazovska, Emmanuel Abbe, Caglar Gulcehre  

**Link**: [PDF](https://arxiv.org/pdf/2504.05108)  

**Abstract**: Discovering efficient algorithms for solving complex problems has been an outstanding challenge in mathematics and computer science, requiring substantial human expertise over the years. Recent advancements in evolutionary search with large language models (LLMs) have shown promise in accelerating the discovery of algorithms across various domains, particularly in mathematics and optimization. However, existing approaches treat the LLM as a static generator, missing the opportunity to update the model with the signal obtained from evolutionary exploration. In this work, we propose to augment LLM-based evolutionary search by continuously refining the search operator - the LLM - through reinforcement learning (RL) fine-tuning. Our method leverages evolutionary search as an exploration strategy to discover improved algorithms, while RL optimizes the LLM policy based on these discoveries. Our experiments on three combinatorial optimization tasks - bin packing, traveling salesman, and the flatpack problem - show that combining RL and evolutionary search improves discovery efficiency of improved algorithms, showcasing the potential of RL-enhanced evolutionary strategies to assist computer scientists and mathematicians for more efficient algorithm design. 

---
# BIASINSPECTOR: Detecting Bias in Structured Data through LLM Agents 

**Authors**: Haoxuan Li, Mingyu Derek Ma, Jen-tse Huang, Zhaotian Weng, Wei Wang, Jieyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.04855)  

**Abstract**: Detecting biases in structured data is a complex and time-consuming task. Existing automated techniques are limited in diversity of data types and heavily reliant on human case-by-case handling, resulting in a lack of generalizability. Currently, large language model (LLM)-based agents have made significant progress in data science, but their ability to detect data biases is still insufficiently explored. To address this gap, we introduce the first end-to-end, multi-agent synergy framework, BIASINSPECTOR, designed for automatic bias detection in structured data based on specific user requirements. It first develops a multi-stage plan to analyze user-specified bias detection tasks and then implements it with a diverse and well-suited set of tools. It delivers detailed results that include explanations and visualizations. To address the lack of a standardized framework for evaluating the capability of LLM agents to detect biases in data, we further propose a comprehensive benchmark that includes multiple evaluation metrics and a large set of test cases. Extensive experiments demonstrate that our framework achieves exceptional overall performance in structured data bias detection, setting a new milestone for fairer data applications. 

---
# Multimodal Agricultural Agent Architecture (MA3): A New Paradigm for Intelligent Agricultural Decision-Making 

**Authors**: Zhuoning Xu, Jian Xu, Mingqing Zhang, Peijie Wang, Chao Deng, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04789)  

**Abstract**: As a strategic pillar industry for human survival and development, modern agriculture faces dual challenges: optimizing production efficiency and achieving sustainable development. Against the backdrop of intensified climate change leading to frequent extreme weather events, the uncertainty risks in agricultural production systems are increasing exponentially. To address these challenges, this study proposes an innovative \textbf{M}ultimodal \textbf{A}gricultural \textbf{A}gent \textbf{A}rchitecture (\textbf{MA3}), which leverages cross-modal information fusion and task collaboration mechanisms to achieve intelligent agricultural decision-making. This study constructs a multimodal agricultural agent dataset encompassing five major tasks: classification, detection, Visual Question Answering (VQA), tool selection, and agent evaluation. We propose a unified backbone for sugarcane disease classification and detection tools, as well as a sugarcane disease expert model. By integrating an innovative tool selection module, we develop a multimodal agricultural agent capable of effectively performing tasks in classification, detection, and VQA. Furthermore, we introduce a multi-dimensional quantitative evaluation framework and conduct a comprehensive assessment of the entire architecture over our evaluation dataset, thereby verifying the practicality and robustness of MA3 in agricultural scenarios. This study provides new insights and methodologies for the development of agricultural agents, holding significant theoretical and practical implications. Our source code and dataset will be made publicly available upon acceptance. 

---
# Lemmanaid: Neuro-Symbolic Lemma Conjecturing 

**Authors**: Yousef Alhessi, Sólrún Halla Einarsdóttir, George Granberry, Emily First, Moa Johansson, Sorin Lerner, Nicholas Smallbone  

**Link**: [PDF](https://arxiv.org/pdf/2504.04942)  

**Abstract**: Automatically conjecturing useful, interesting and novel lemmas would greatly improve automated reasoning tools and lower the bar for formalizing mathematics in proof assistants. It is however a very challenging task for both neural and symbolic approaches. We present the first steps towards a practical neuro-symbolic lemma conjecturing tool, Lemmanaid, that combines Large Language Models (LLMs) and symbolic methods, and evaluate it on proof libraries for the Isabelle proof assistant. We train an LLM to generate lemma templates that describe the shape of a lemma, and use symbolic methods to fill in the details. We compare Lemmanaid against an LLM trained to generate complete lemma statements as well as previous fully symbolic conjecturing methods. Our results indicate that neural and symbolic techniques are complementary. By leveraging the best of both symbolic and neural methods we can generate useful lemmas for a wide range of input domains, facilitating computer-assisted theory development and formalization. 

---
# Capturing AI's Attention: Physics of Repetition, Hallucination, Bias and Beyond 

**Authors**: Frank Yingjie Huo, Neil F. Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2504.04600)  

**Abstract**: We derive a first-principles physics theory of the AI engine at the heart of LLMs' 'magic' (e.g. ChatGPT, Claude): the basic Attention head. The theory allows a quantitative analysis of outstanding AI challenges such as output repetition, hallucination and harmful content, and bias (e.g. from training and fine-tuning). Its predictions are consistent with large-scale LLM outputs. Its 2-body form suggests why LLMs work so well, but hints that a generalized 3-body Attention would make such AI work even better. Its similarity to a spin-bath means that existing Physics expertise could immediately be harnessed to help Society ensure AI is trustworthy and resilient to manipulation. 

---
# Hierarchical Planning for Complex Tasks with Knowledge Graph-RAG and Symbolic Verification 

**Authors**: Cristina Cornelio, Flavio Petruzzellis, Pietro Lio  

**Link**: [PDF](https://arxiv.org/pdf/2504.04578)  

**Abstract**: Large Language Models (LLMs) have shown promise as robotic planners but often struggle with long-horizon and complex tasks, especially in specialized environments requiring external knowledge. While hierarchical planning and Retrieval-Augmented Generation (RAG) address some of these challenges, they remain insufficient on their own and a deeper integration is required for achieving more reliable systems. To this end, we propose a neuro-symbolic approach that enhances LLMs-based planners with Knowledge Graph-based RAG for hierarchical plan generation. This method decomposes complex tasks into manageable subtasks, further expanded into executable atomic action sequences. To ensure formal correctness and proper decomposition, we integrate a Symbolic Validator, which also functions as a failure detector by aligning expected and observed world states. Our evaluation against baseline methods demonstrates the consistent significant advantages of integrating hierarchical planning, symbolic verification, and RAG across tasks of varying complexity and different LLMs. Additionally, our experimental setup and novel metrics not only validate our approach for complex planning but also serve as a tool for assessing LLMs' reasoning and compositional capabilities. 

---
# Constitution or Collapse? Exploring Constitutional AI with Llama 3-8B 

**Authors**: Xue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04918)  

**Abstract**: As language models continue to grow larger, the cost of acquiring high-quality training data has increased significantly. Collecting human feedback is both expensive and time-consuming, and manual labels can be noisy, leading to an imbalance between helpfulness and harmfulness. Constitutional AI, introduced by Anthropic in December 2022, uses AI to provide feedback to another AI, greatly reducing the need for human labeling. However, the original implementation was designed for a model with around 52 billion parameters, and there is limited information on how well Constitutional AI performs with smaller models, such as LLaMA 3-8B. In this paper, we replicated the Constitutional AI workflow using the smaller LLaMA 3-8B model. Our results show that Constitutional AI can effectively increase the harmlessness of the model, reducing the Attack Success Rate in MT-Bench by 40.8%. However, similar to the original study, increasing harmlessness comes at the cost of helpfulness. The helpfulness metrics, which are an average of the Turn 1 and Turn 2 scores, dropped by 9.8% compared to the baseline. Additionally, we observed clear signs of model collapse in the final DPO-CAI model, indicating that smaller models may struggle with self-improvement due to insufficient output quality, making effective fine-tuning more challenging. Our study suggests that, like reasoning and math ability, self-improvement is an emergent property. 

---
# Crowdsourcing-Based Knowledge Graph Construction for Drug Side Effects Using Large Language Models with an Application on Semaglutide 

**Authors**: Zhijie Duan, Kai Wei, Zhaoqian Xue, Lingyao li, Jin Jin, Shu Yang, Jiayan Zhou, Siyuan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.04346)  

**Abstract**: Social media is a rich source of real-world data that captures valuable patient experience information for pharmacovigilance. However, mining data from unstructured and noisy social media content remains a challenging task. We present a systematic framework that leverages large language models (LLMs) to extract medication side effects from social media and organize them into a knowledge graph (KG). We apply this framework to semaglutide for weight loss using data from Reddit. Using the constructed knowledge graph, we perform comprehensive analyses to investigate reported side effects across different semaglutide brands over time. These findings are further validated through comparison with adverse events reported in the FAERS database, providing important patient-centered insights into semaglutide's side effects that complement its safety profile and current knowledge base of semaglutide for both healthcare professionals and patients. Our work demonstrates the feasibility of using LLMs to transform social media data into structured KGs for pharmacovigilance. 

---
# Among Us: A Sandbox for Agentic Deception 

**Authors**: Satvik Golechha, Adrià Garriga-Alonso  

**Link**: [PDF](https://arxiv.org/pdf/2504.04072)  

**Abstract**: Studying deception in AI agents is important and difficult due to the lack of model organisms and sandboxes that elicit the behavior without asking the model to act under specific conditions or inserting intentional backdoors. Extending upon $\textit{AmongAgents}$, a text-based social-deduction game environment, we aim to fix this by introducing Among Us as a rich sandbox where LLM-agents exhibit human-style deception naturally while they think, speak, and act with other agents or humans. We introduce Deception ELO as an unbounded measure of deceptive capability, suggesting that frontier models win more because they're better at deception, not at detecting it. We evaluate the effectiveness of AI safety techniques (LLM-monitoring of outputs, linear probes on various datasets, and sparse autoencoders) for detecting lying and deception in Among Us, and find that they generalize very well out-of-distribution. We open-source our sandbox as a benchmark for future alignment research and hope that this is a good testbed to improve safety techniques to detect and remove agentically-motivated deception, and to anticipate deceptive abilities in LLMs. 

---
# Have Large Language Models Learned to Reason? A Characterization via 3-SAT Phase Transition 

**Authors**: Rishi Hazra, Gabriele Venturato, Pedro Zuidberg Dos Martires, Luc De Raedt  

**Link**: [PDF](https://arxiv.org/pdf/2504.03930)  

**Abstract**: Large Language Models (LLMs) have been touted as AI models possessing advanced reasoning abilities. In theory, autoregressive LLMs with Chain-of-Thought (CoT) can perform more serial computations to solve complex reasoning tasks. However, recent studies suggest that, despite this capacity, LLMs do not truly learn to reason but instead fit on statistical features. To study the reasoning capabilities in a principled fashion, we adopt a computational theory perspective and propose an experimental protocol centered on 3-SAT -- the prototypical NP-complete problem lying at the core of logical reasoning and constraint satisfaction tasks. Specifically, we examine the phase transitions in random 3-SAT and characterize the reasoning abilities of state-of-the-art LLMs by varying the inherent hardness of the problem instances. By comparing DeepSeek R1 with other LLMs, our findings reveal two key insights (1) LLM accuracy drops significantly on harder instances, suggesting all current models struggle when statistical shortcuts are unavailable (2) Unlike other LLMs, R1 shows signs of having learned the underlying reasoning. Following a principled experimental protocol, our study moves beyond the benchmark-driven evidence often found in LLM reasoning research. Our findings highlight important gaps and suggest clear directions for future research. 

---
# ADAPT: Actively Discovering and Adapting to Preferences for any Task 

**Authors**: Maithili Patel, Xavier Puig, Ruta Desai, Roozbeh Mottaghi, Sonia Chernova, Joanne Truong, Akshara Rai  

**Link**: [PDF](https://arxiv.org/pdf/2504.04040)  

**Abstract**: Assistive agents should be able to perform under-specified long-horizon tasks while respecting user preferences. We introduce Actively Discovering and Adapting to Preferences for any Task (ADAPT) -- a benchmark designed to evaluate agents' ability to adhere to user preferences across various household tasks through active questioning. Next, we propose Reflection-DPO, a novel training approach for adapting large language models (LLMs) to the task of active questioning. Reflection-DPO finetunes a 'student' LLM to follow the actions of a privileged 'teacher' LLM, and optionally ask a question to gather necessary information to better predict the teacher action. We find that prior approaches that use state-of-the-art LLMs fail to sufficiently follow user preferences in ADAPT due to insufficient questioning and poor adherence to elicited preferences. In contrast, Reflection-DPO achieves a higher rate of satisfying user preferences, outperforming a zero-shot chain-of-thought baseline by 6.1% on unseen users. 

---
# Hierarchically Encapsulated Representation for Protocol Design in Self-Driving Labs 

**Authors**: Yu-Zhe Shi, Mingchen Liu, Fanxu Meng, Qiao Xu, Zhangqian Bi, Kun He, Lecheng Ruan, Qining Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03810)  

**Abstract**: Self-driving laboratories have begun to replace human experimenters in performing single experimental skills or predetermined experimental protocols. However, as the pace of idea iteration in scientific research has been intensified by Artificial Intelligence, the demand for rapid design of new protocols for new discoveries become evident. Efforts to automate protocol design have been initiated, but the capabilities of knowledge-based machine designers, such as Large Language Models, have not been fully elicited, probably for the absence of a systematic representation of experimental knowledge, as opposed to isolated, flatten pieces of information. To tackle this issue, we propose a multi-faceted, multi-scale representation, where instance actions, generalized operations, and product flow models are hierarchically encapsulated using Domain-Specific Languages. We further develop a data-driven algorithm based on non-parametric modeling that autonomously customizes these representations for specific domains. The proposed representation is equipped with various machine designers to manage protocol design tasks, including planning, modification, and adjustment. The results demonstrate that the proposed method could effectively complement Large Language Models in the protocol design process, serving as an auxiliary module in the realm of machine-assisted scientific exploration. 

---
# URECA: Unique Region Caption Anything 

**Authors**: Sangbeom Lim, Junwan Kim, Heeji Yoon, Jaewoo Jung, Seungryong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.05305)  

**Abstract**: Region-level captioning aims to generate natural language descriptions for specific image regions while highlighting their distinguishing features. However, existing methods struggle to produce unique captions across multi-granularity, limiting their real-world applicability. To address the need for detailed region-level understanding, we introduce URECA dataset, a large-scale dataset tailored for multi-granularity region captioning. Unlike prior datasets that focus primarily on salient objects, URECA dataset ensures a unique and consistent mapping between regions and captions by incorporating a diverse set of objects, parts, and background elements. Central to this is a stage-wise data curation pipeline, where each stage incrementally refines region selection and caption generation. By leveraging Multimodal Large Language Models (MLLMs) at each stage, our pipeline produces distinctive and contextually grounded captions with improved accuracy and semantic diversity. Building upon this dataset, we present URECA, a novel captioning model designed to effectively encode multi-granularity regions. URECA maintains essential spatial properties such as position and shape through simple yet impactful modifications to existing MLLMs, enabling fine-grained and semantically rich region descriptions. Our approach introduces dynamic mask modeling and a high-resolution mask encoder to enhance caption uniqueness. Experiments show that URECA achieves state-of-the-art performance on URECA dataset and generalizes well to existing region-level captioning benchmarks. 

---
# Explaining Low Perception Model Competency with High-Competency Counterfactuals 

**Authors**: Sara Pohland, Claire Tomlin  

**Link**: [PDF](https://arxiv.org/pdf/2504.05254)  

**Abstract**: There exist many methods to explain how an image classification model generates its decision, but very little work has explored methods to explain why a classifier might lack confidence in its prediction. As there are various reasons the classifier might lose confidence, it would be valuable for this model to not only indicate its level of uncertainty but also explain why it is uncertain. Counterfactual images have been used to visualize changes that could be made to an image to generate a different classification decision. In this work, we explore the use of counterfactuals to offer an explanation for low model competency--a generalized form of predictive uncertainty that measures confidence. Toward this end, we develop five novel methods to generate high-competency counterfactual images, namely Image Gradient Descent (IGD), Feature Gradient Descent (FGD), Autoencoder Reconstruction (Reco), Latent Gradient Descent (LGD), and Latent Nearest Neighbors (LNN). We evaluate these methods across two unique datasets containing images with six known causes for low model competency and find Reco, LGD, and LNN to be the most promising methods for counterfactual generation. We further evaluate how these three methods can be utilized by pre-trained Multimodal Large Language Models (MLLMs) to generate language explanations for low model competency. We find that the inclusion of a counterfactual image in the language model query greatly increases the ability of the model to generate an accurate explanation for the cause of low model competency, thus demonstrating the utility of counterfactual images in explaining low perception model competency. 

---
# The Dream Within Huang Long Cave: AI-Driven Interactive Narrative for Family Storytelling and Emotional Reflection 

**Authors**: Jiayang Huang, Lingjie Li, Kang Zhang, David Yip  

**Link**: [PDF](https://arxiv.org/pdf/2504.04968)  

**Abstract**: This paper introduces the art project The Dream Within Huang Long Cave, an AI-driven interactive and immersive narrative experience. The project offers new insights into AI technology, artistic practice, and psychoanalysis. Inspired by actual geographical landscapes and familial archetypes, the work combines psychoanalytic theory and computational technology, providing an artistic response to the concept of the non-existence of the Big Other. The narrative is driven by a combination of a large language model (LLM) and a realistic digital character, forming a virtual agent named YELL. Through dialogue and exploration within a cave automatic virtual environment (CAVE), the audience is invited to unravel the language puzzles presented by YELL and help him overcome his life challenges. YELL is a fictional embodiment of the Big Other, modeled after the artist's real father. Through a cross-temporal interaction with this digital father, the project seeks to deconstruct complex familial relationships. By demonstrating the non-existence of the Big Other, we aim to underscore the authenticity of interpersonal emotions, positioning art as a bridge for emotional connection and understanding within family dynamics. 

---
# BRIDGES: Bridging Graph Modality and Large Language Models within EDA Tasks 

**Authors**: Wei Li, Yang Zou, Christopher Ellis, Ruben Purdy, Shawn Blanton, José M. F. Moura  

**Link**: [PDF](https://arxiv.org/pdf/2504.05180)  

**Abstract**: While many EDA tasks already involve graph-based data, existing LLMs in EDA primarily either represent graphs as sequential text, or simply ignore graph-structured data that might be beneficial like dataflow graphs of RTL code. Recent studies have found that LLM performance suffers when graphs are represented as sequential text, and using additional graph information significantly boosts performance. To address these challenges, we introduce BRIDGES, a framework designed to incorporate graph modality into LLMs for EDA tasks. BRIDGES integrates an automated data generation workflow, a solution that combines graph modality with LLM, and a comprehensive evaluation suite. First, we establish an LLM-driven workflow to generate RTL and netlist-level data, converting them into dataflow and netlist graphs with function descriptions. This workflow yields a large-scale dataset comprising over 500,000 graph instances and more than 1.5 billion tokens. Second, we propose a lightweight cross-modal projector that encodes graph representations into text-compatible prompts, enabling LLMs to effectively utilize graph data without architectural modifications. Experimental results demonstrate 2x to 10x improvements across multiple tasks compared to text-only baselines, including accuracy in design retrieval, type prediction and perplexity in function description, with negligible computational overhead (<1% model weights increase and <30% additional runtime overhead). Even without additional LLM finetuning, our results outperform text-only by a large margin. We plan to release BRIDGES, including the dataset, models, and training flow. 

---
# Enhancing Compositional Reasoning in Vision-Language Models with Synthetic Preference Data 

**Authors**: Samarth Mishra, Kate Saenko, Venkatesh Saligrama  

**Link**: [PDF](https://arxiv.org/pdf/2504.04740)  

**Abstract**: Compositionality, or correctly recognizing scenes as compositions of atomic visual concepts, remains difficult for multimodal large language models (MLLMs). Even state of the art MLLMs such as GPT-4o can make mistakes in distinguishing compositions like "dog chasing cat" vs "cat chasing dog". While on Winoground, a benchmark for measuring such reasoning, MLLMs have made significant progress, they are still far from a human's performance. We show that compositional reasoning in these models can be improved by elucidating such concepts via data, where a model is trained to prefer the correct caption for an image over a close but incorrect one. We introduce SCRAMBLe: Synthetic Compositional Reasoning Augmentation of MLLMs with Binary preference Learning, an approach for preference tuning open-weight MLLMs on synthetic preference data generated in a fully automated manner from existing image-caption data. SCRAMBLe holistically improves these MLLMs' compositional reasoning capabilities which we can see through significant improvements across multiple vision language compositionality benchmarks, as well as smaller but significant improvements on general question answering tasks. As a sneak peek, SCRAMBLe tuned Molmo-7B model improves on Winoground from 49.5% to 54.8% (best reported to date), while improving by ~1% on more general visual question answering tasks. Code for SCRAMBLe along with tuned models and our synthetic training dataset is available at this https URL. 

---
# Planning Safety Trajectories with Dual-Phase, Physics-Informed, and Transportation Knowledge-Driven Large Language Models 

**Authors**: Rui Gan, Pei Li, Keke Long, Bocheng An, Junwei You, Keshu Wu, Bin Ran  

**Link**: [PDF](https://arxiv.org/pdf/2504.04562)  

**Abstract**: Foundation models have demonstrated strong reasoning and generalization capabilities in driving-related tasks, including scene understanding, planning, and control. However, they still face challenges in hallucinations, uncertainty, and long inference latency. While existing foundation models have general knowledge of avoiding collisions, they often lack transportation-specific safety knowledge. To overcome these limitations, we introduce LetsPi, a physics-informed, dual-phase, knowledge-driven framework for safe, human-like trajectory planning. To prevent hallucinations and minimize uncertainty, this hybrid framework integrates Large Language Model (LLM) reasoning with physics-informed social force dynamics. LetsPi leverages the LLM to analyze driving scenes and historical information, providing appropriate parameters and target destinations (goals) for the social force model, which then generates the future trajectory. Moreover, the dual-phase architecture balances reasoning and computational efficiency through its Memory Collection phase and Fast Inference phase. The Memory Collection phase leverages the physics-informed LLM to process and refine planning results through reasoning, reflection, and memory modules, storing safe, high-quality driving experiences in a memory bank. Surrogate safety measures and physics-informed prompt techniques are introduced to enhance the LLM's knowledge of transportation safety and physical force, respectively. The Fast Inference phase extracts similar driving experiences as few-shot examples for new scenarios, while simplifying input-output requirements to enable rapid trajectory planning without compromising safety. Extensive experiments using the HighD dataset demonstrate that LetsPi outperforms baseline models across five safety this http URL PDF for project Github link. 

---
# Bridging Knowledge Gap Between Image Inpainting and Large-Area Visible Watermark Removal 

**Authors**: Yicheng Leng, Chaowei Fang, Junye Chen, Yixiang Fang, Sheng Li, Guanbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.04687)  

**Abstract**: Visible watermark removal which involves watermark cleaning and background content restoration is pivotal to evaluate the resilience of watermarks. Existing deep neural network (DNN)-based models still struggle with large-area watermarks and are overly dependent on the quality of watermark mask prediction. To overcome these challenges, we introduce a novel feature adapting framework that leverages the representation modeling capacity of a pre-trained image inpainting model. Our approach bridges the knowledge gap between image inpainting and watermark removal by fusing information of the residual background content beneath watermarks into the inpainting backbone model. We establish a dual-branch system to capture and embed features from the residual background content, which are merged into intermediate features of the inpainting backbone model via gated feature fusion modules. Moreover, for relieving the dependence on high-quality watermark masks, we introduce a new training paradigm by utilizing coarse watermark masks to guide the inference process. This contributes to a visible image removal model which is insensitive to the quality of watermark mask during testing. Extensive experiments on both a large-scale synthesized dataset and a real-world dataset demonstrate that our approach significantly outperforms existing state-of-the-art methods. The source code is available in the supplementary materials. 

---
# Advancing Egocentric Video Question Answering with Multimodal Large Language Models 

**Authors**: Alkesh Patel, Vibhav Chitalia, Yinfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04550)  

**Abstract**: Egocentric Video Question Answering (QA) requires models to handle long-horizon temporal reasoning, first-person perspectives, and specialized challenges like frequent camera movement. This paper systematically evaluates both proprietary and open-source Multimodal Large Language Models (MLLMs) on QaEgo4Dv2 - a refined dataset of egocentric videos derived from QaEgo4D. Four popular MLLMs (GPT-4o, Gemini-1.5-Pro, Video-LLaVa-7B and Qwen2-VL-7B-Instruct) are assessed using zero-shot and fine-tuned approaches for both OpenQA and CloseQA settings. We introduce QaEgo4Dv2 to mitigate annotation noise in QaEgo4D, enabling more reliable comparison. Our results show that fine-tuned Video-LLaVa-7B and Qwen2-VL-7B-Instruct achieve new state-of-the-art performance, surpassing previous benchmarks by up to +2.6% ROUGE/METEOR (for OpenQA) and +13% accuracy (for CloseQA). We also present a thorough error analysis, indicating the model's difficulty in spatial reasoning and fine-grained object recognition - key areas for future improvement. 

---
# How Accurately Do Large Language Models Understand Code? 

**Authors**: Sabaat Haroon, Ahmad Faraz Khan, Ahmad Humayun, Waris Gill, Abdul Haddi Amjad, Ali R. Butt, Mohammad Taha Khan, Muhammad Ali Gulzar  

**Link**: [PDF](https://arxiv.org/pdf/2504.04372)  

**Abstract**: Large Language Models (LLMs) are increasingly used in post-development tasks such as code repair and testing. A key factor in these tasks' success is the model's deep understanding of code. However, the extent to which LLMs truly understand code remains largely unevaluated. Quantifying code comprehension is challenging due to its abstract nature and the lack of a standardized metric. Previously, this was assessed through developer surveys, which are not feasible for evaluating LLMs. Existing LLM benchmarks focus primarily on code generation, fundamentally different from code comprehension. Additionally, fixed benchmarks quickly become obsolete as they become part of the training data. This paper presents the first large-scale empirical investigation into LLMs' ability to understand code. Inspired by mutation testing, we use an LLM's fault-finding ability as a proxy for its deep code understanding. This approach is based on the insight that a model capable of identifying subtle functional discrepancies must understand the code well. We inject faults in real-world programs and ask the LLM to localize them, ensuring the specifications suffice for fault localization. Next, we apply semantic-preserving code mutations (SPMs) to the faulty programs and test whether the LLMs still locate the faults, verifying their confidence in code understanding. We evaluate nine popular LLMs on 575000 debugging tasks from 670 Java and 637 Python programs. We find that LLMs lose the ability to debug the same bug in 81% of faulty programs when SPMs are applied, indicating a shallow understanding of code and reliance on features irrelevant to semantics. We also find that LLMs understand code earlier in the program better than later. This suggests that LLMs' code comprehension remains tied to lexical and syntactic features due to tokenization designed for natural languages, which overlooks code semantics. 

---
# AutoPDL: Automatic Prompt Optimization for LLM Agents 

**Authors**: Claudio Spiess, Mandana Vaziri, Louis Mandel, Martin Hirzel  

**Link**: [PDF](https://arxiv.org/pdf/2504.04365)  

**Abstract**: The performance of large language models (LLMs) depends on how they are prompted, with choices spanning both the high-level prompting pattern (e.g., Zero-Shot, CoT, ReAct, ReWOO) and the specific prompt content (instructions and few-shot demonstrations). Manually tuning this combination is tedious, error-prone, and non-transferable across LLMs or tasks. Therefore, this paper proposes AutoPDL, an automated approach to discover good LLM agent configurations. Our method frames this as a structured AutoML problem over a combinatorial space of agentic and non-agentic prompting patterns and demonstrations, using successive halving to efficiently navigate this space. We introduce a library implementing common prompting patterns using the PDL prompt programming language. AutoPDL solutions are human-readable, editable, and executable PDL programs that use this library. This approach also enables source-to-source optimization, allowing human-in-the-loop refinement and reuse. Evaluations across three tasks and six LLMs (ranging from 8B to 70B parameters) show consistent accuracy gains ($9.5\pm17.5$ percentage points), up to 68.9pp, and reveal that selected prompting strategies vary across models and tasks. 

---
# Universal Item Tokenization for Transferable Generative Recommendation 

**Authors**: Bowen Zheng, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2504.04405)  

**Abstract**: Recently, generative recommendation has emerged as a promising paradigm, attracting significant research attention. The basic framework involves an item tokenizer, which represents each item as a sequence of codes serving as its identifier, and a generative recommender that predicts the next item by autoregressively generating the target item identifier. However, in existing methods, both the tokenizer and the recommender are typically domain-specific, limiting their ability for effective transfer or adaptation to new domains. To this end, we propose UTGRec, a Universal item Tokenization approach for transferable Generative Recommendation. Specifically, we design a universal item tokenizer for encoding rich item semantics by adapting a multimodal large language model (MLLM). By devising tree-structured codebooks, we discretize content representations into corresponding codes for item tokenization. To effectively learn the universal item tokenizer on multiple domains, we introduce two key techniques in our approach. For raw content reconstruction, we employ dual lightweight decoders to reconstruct item text and images from discrete representations to capture general knowledge embedded in the content. For collaborative knowledge integration, we assume that co-occurring items are similar and integrate collaborative signals through co-occurrence alignment and reconstruction. Finally, we present a joint learning framework to pre-train and adapt the transferable generative recommender across multiple domains. Extensive experiments on four public datasets demonstrate the superiority of UTGRec compared to both traditional and generative recommendation baselines. 

---
# Driving-RAG: Driving Scenarios Embedding, Search, and RAG Applications 

**Authors**: Cheng Chang, Jingwei Ge, Jiazhe Guo, Zelin Guo, Binghong Jiang, Li Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.04419)  

**Abstract**: Driving scenario data play an increasingly vital role in the development of intelligent vehicles and autonomous driving. Accurate and efficient scenario data search is critical for both online vehicle decision-making and planning, and offline scenario generation and simulations, as it allows for leveraging the scenario experiences to improve the overall performance. Especially with the application of large language models (LLMs) and Retrieval-Augmented-Generation (RAG) systems in autonomous driving, urgent requirements are put forward. In this paper, we introduce the Driving-RAG framework to address the challenges of efficient scenario data embedding, search, and applications for RAG systems. Our embedding model aligns fundamental scenario information and scenario distance metrics in the vector space. The typical scenario sampling method combined with hierarchical navigable small world can perform efficient scenario vector search to achieve high efficiency without sacrificing accuracy. In addition, the reorganization mechanism by graph knowledge enhances the relevance to the prompt scenarios and augment LLM generation. We demonstrate the effectiveness of the proposed framework on typical trajectory planning task for complex interactive scenarios such as ramps and intersections, showcasing its advantages for RAG applications. 

---
# Geo-OLM: Enabling Sustainable Earth Observation Studies with Cost-Efficient Open Language Models & State-Driven Workflows 

**Authors**: Dimitrios Stamoulis, Diana Marculescu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04319)  

**Abstract**: Geospatial Copilots hold immense potential for automating Earth observation (EO) and climate monitoring workflows, yet their reliance on large-scale models such as GPT-4o introduces a paradox: tools intended for sustainability studies often incur unsustainable costs. Using agentic AI frameworks in geospatial applications can amass thousands of dollars in API charges or requires expensive, power-intensive GPUs for deployment, creating barriers for researchers, policymakers, and NGOs. Unfortunately, when geospatial Copilots are deployed with open language models (OLMs), performance often degrades due to their dependence on GPT-optimized logic. In this paper, we present Geo-OLM, a tool-augmented geospatial agent that leverages the novel paradigm of state-driven LLM reasoning to decouple task progression from tool calling. By alleviating the workflow reasoning burden, our approach enables low-resource OLMs to complete geospatial tasks more effectively. When downsizing to small models below 7B parameters, Geo-OLM outperforms the strongest prior geospatial baselines by 32.8% in successful query completion rates. Our method performs comparably to proprietary models achieving results within 10% of GPT-4o, while reducing inference costs by two orders of magnitude from \$500-\$1000 to under \$10. We present an in-depth analysis with geospatial downstream benchmarks, providing key insights to help practitioners effectively deploy OLMs for EO applications. 

---
# TrafficLLM: Enhancing Large Language Models for Network Traffic Analysis with Generic Traffic Representation 

**Authors**: Tianyu Cui, Xinjie Lin, Sijia Li, Miao Chen, Qilei Yin, Qi Li, Ke Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04222)  

**Abstract**: Machine learning (ML) powered network traffic analysis has been widely used for the purpose of threat detection. Unfortunately, their generalization across different tasks and unseen data is very limited. Large language models (LLMs), known for their strong generalization capabilities, have shown promising performance in various domains. However, their application to the traffic analysis domain is limited due to significantly different characteristics of network traffic. To address the issue, in this paper, we propose TrafficLLM, which introduces a dual-stage fine-tuning framework to learn generic traffic representation from heterogeneous raw traffic data. The framework uses traffic-domain tokenization, dual-stage tuning pipeline, and extensible adaptation to help LLM release generalization ability on dynamic traffic analysis tasks, such that it enables traffic detection and traffic generation across a wide range of downstream tasks. We evaluate TrafficLLM across 10 distinct scenarios and 229 types of traffic. TrafficLLM achieves F1-scores of 0.9875 and 0.9483, with up to 80.12% and 33.92% better performance than existing detection and generation methods. It also shows strong generalization on unseen traffic with an 18.6% performance improvement. We further evaluate TrafficLLM in real-world scenarios. The results confirm that TrafficLLM is easy to scale and achieves accurate detection performance on enterprise traffic. 

---
# TARAC: Mitigating Hallucination in LVLMs via Temporal Attention Real-time Accumulative Connection 

**Authors**: Chunzhao Xie, Tongxuan Liu, Lei Jiang, Yuting Zeng, jinrong Guo, Yunheng Shen, Weizhe Huang, Jing Li, Xiaohua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04099)  

**Abstract**: Large Vision-Language Models have demonstrated remarkable performance across various tasks; however, the challenge of hallucinations constrains their practical applications. The hallucination problem arises from multiple factors, including the inherent hallucinations in language models, the limitations of visual encoders in perception, and biases introduced by multimodal data. Extensive research has explored ways to mitigate hallucinations. For instance, OPERA prevents the model from overly focusing on "anchor tokens", thereby reducing hallucinations, whereas VCD mitigates hallucinations by employing a contrastive decoding approach. In this paper, we investigate the correlation between the decay of attention to image tokens and the occurrence of hallucinations. Based on this finding, we propose Temporal Attention Real-time Accumulative Connection (TARAC), a novel training-free method that dynamically accumulates and updates LVLMs' attention on image tokens during generation. By enhancing the model's attention to image tokens, TARAC mitigates hallucinations caused by the decay of attention on image tokens. We validate the effectiveness of TARAC across multiple models and datasets, demonstrating that our approach substantially mitigates hallucinations. In particular, TARAC reduces $C_S$ by 25.2 and $C_I$ by 8.7 compared to VCD on the CHAIR benchmark. 

---
# OLAF: An Open Life Science Analysis Framework for Conversational Bioinformatics Powered by Large Language Models 

**Authors**: Dylan Riffle, Nima Shirooni, Cody He, Manush Murali, Sovit Nayak, Rishikumar Gopalan, Diego Gonzalez Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2504.03976)  

**Abstract**: OLAF (Open Life Science Analysis Framework) is an open-source platform that enables researchers to perform bioinformatics analyses using natural language. By combining large language models (LLMs) with a modular agent-pipe-router architecture, OLAF generates and executes bioinformatics code on real scientific data, including formats like .h5ad. The system includes an Angular front end and a Python/Firebase backend, allowing users to run analyses such as single-cell RNA-seq workflows, gene annotation, and data visualization through a simple web interface. Unlike general-purpose AI tools, OLAF integrates code execution, data handling, and scientific libraries in a reproducible, user-friendly environment. It is designed to lower the barrier to computational biology for non-programmers and support transparent, AI-powered life science research. 

---
# Bridging LMS and Generative AI: Dynamic Course Content Integration (DCCI) for Connecting LLMs to Course Content -- The Ask ME Assistant 

**Authors**: Kovan Mzwri, Márta Turcsányi-Szabo  

**Link**: [PDF](https://arxiv.org/pdf/2504.03966)  

**Abstract**: The integration of Large Language Models (LLMs) with Learning Management Systems (LMSs) has the potential to enhance task automation and accessibility in education. However, hallucination where LLMs generate inaccurate or misleading information remains a significant challenge. This study introduces the Dynamic Course Content Integration (DCCI) mechanism, which dynamically retrieves and integrates course content and curriculum from Canvas LMS into the LLM-powered assistant, Ask ME. By employing prompt engineering to structure retrieved content within the LLM's context window, DCCI ensures accuracy, relevance, and contextual alignment, mitigating hallucination. To evaluate DCCI's effectiveness, Ask ME's usability, and broader student perceptions of AI in education, a mixed-methods approach was employed, incorporating user satisfaction ratings and a structured survey. Results from a pilot study indicate high user satisfaction (4.614/5), with students recognizing Ask ME's ability to provide timely and contextually relevant responses for both administrative and course-related inquiries. Additionally, a majority of students agreed that Ask ME's integration with course content in Canvas LMS reduced platform-switching, improving usability, engagement, and comprehension. AI's role in reducing classroom hesitation and fostering self-directed learning and intellectual curiosity was also highlighted. Despite these benefits and positive perception of AI tools, concerns emerged regarding over-reliance on AI, accuracy limitations, and ethical issues such as plagiarism and reduced student-teacher interaction. These findings emphasize the need for strategic AI implementation, ethical safeguards, and a pedagogical framework that prioritizes human-AI collaboration over substitution. 

---
# Can ChatGPT Learn My Life From a Week of First-Person Video? 

**Authors**: Keegan Harris  

**Link**: [PDF](https://arxiv.org/pdf/2504.03857)  

**Abstract**: Motivated by recent improvements in generative AI and wearable camera devices (e.g. smart glasses and AI-enabled pins), I investigate the ability of foundation models to learn about the wearer's personal life through first-person camera data. To test this, I wore a camera headset for 54 hours over the course of a week, generated summaries of various lengths (e.g. minute-long, hour-long, and day-long summaries), and fine-tuned both GPT-4o and GPT-4o-mini on the resulting summary hierarchy. By querying the fine-tuned models, we are able to learn what the models learned about me. The results are mixed: Both models learned basic information about me (e.g. approximate age, gender). Moreover, GPT-4o correctly deduced that I live in Pittsburgh, am a PhD student at CMU, am right-handed, and have a pet cat. However, both models also suffered from hallucination and would make up names for the individuals present in the video footage of my life. 

---
# Arti-"fickle" Intelligence: Using LLMs as a Tool for Inference in the Political and Social Sciences 

**Authors**: Lisa P. Argyle, Ethan C. Busby, Joshua R. Gubler, Bryce Hepner, Alex Lyman, David Wingate  

**Link**: [PDF](https://arxiv.org/pdf/2504.03822)  

**Abstract**: Generative large language models (LLMs) are incredibly useful, versatile, and promising tools. However, they will be of most use to political and social science researchers when they are used in a way that advances understanding about real human behaviors and concerns. To promote the scientific use of LLMs, we suggest that researchers in the political and social sciences need to remain focused on the scientific goal of inference. To this end, we discuss the challenges and opportunities related to scientific inference with LLMs, using validation of model output as an illustrative case for discussion. We propose a set of guidelines related to establishing the failure and success of LLMs when completing particular tasks, and discuss how we can make inferences from these observations. We conclude with a discussion of how this refocus will improve the accumulation of shared scientific knowledge about these tools and their uses in the social sciences. 

---
# MCP Safety Audit: LLMs with the Model Context Protocol Allow Major Security Exploits 

**Authors**: Brandon Radosevich, John Halloran  

**Link**: [PDF](https://arxiv.org/pdf/2504.03767)  

**Abstract**: To reduce development overhead and enable seamless integration between potential components comprising any given generative AI application, the Model Context Protocol (MCP) (Anthropic, 2024) has recently been released and subsequently widely adopted. The MCP is an open protocol that standardizes API calls to large language models (LLMs), data sources, and agentic tools. By connecting multiple MCP servers, each defined with a set of tools, resources, and prompts, users are able to define automated workflows fully driven by LLMs. However, we show that the current MCP design carries a wide range of security risks for end users. In particular, we demonstrate that industry-leading LLMs may be coerced into using MCP tools to compromise an AI developer's system through various attacks, such as malicious code execution, remote access control, and credential theft. To proactively mitigate these and related attacks, we introduce a safety auditing tool, MCPSafetyScanner, the first agentic tool to assess the security of an arbitrary MCP server. MCPScanner uses several agents to (a) automatically determine adversarial samples given an MCP server's tools and resources; (b) search for related vulnerabilities and remediations based on those samples; and (c) generate a security report detailing all findings. Our work highlights serious security issues with general-purpose agentic workflows while also providing a proactive tool to audit MCP server safety and address detected vulnerabilities before deployment.
The described MCP server auditing tool, MCPSafetyScanner, is freely available at: this https URL 

---
# Ethical AI on the Waitlist: Group Fairness Evaluation of LLM-Aided Organ Allocation 

**Authors**: Hannah Murray, Brian Hyeongseok Kim, Isabelle Lee, Jason Byun, Dani Yogatama, Evi Micha  

**Link**: [PDF](https://arxiv.org/pdf/2504.03716)  

**Abstract**: Large Language Models (LLMs) are becoming ubiquitous, promising automation even in high-stakes scenarios. However, existing evaluation methods often fall short -- benchmarks saturate, accuracy-based metrics are overly simplistic, and many inherently ambiguous problems lack a clear ground truth. Given these limitations, evaluating fairness becomes complex. To address this, we reframe fairness evaluation using Borda scores, a method from voting theory, as a nuanced yet interpretable metric for measuring fairness. Using organ allocation as a case study, we introduce two tasks: (1) Choose-One and (2) Rank-All. In Choose-One, LLMs select a single candidate for a kidney, and we assess fairness across demographics using proportional parity. In Rank-All, LLMs rank all candidates for a kidney, reflecting real-world allocation processes. Since traditional fairness metrics do not account for ranking, we propose a novel application of Borda scoring to capture biases. Our findings highlight the potential of voting-based metrics to provide a richer, more multifaceted evaluation of LLM fairness. 

---
# RLDBF: Enhancing LLMs Via Reinforcement Learning With DataBase FeedBack 

**Authors**: Weichen Dai, Zijie Dai, Zhijie Huang, Yixuan Pan, Xinhe Li, Xi Li, Yi Zhou, Ji Qi, Wu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03713)  

**Abstract**: While current large language models (LLMs) demonstrate remarkable linguistic capabilities through training on massive unstructured text corpora, they remain inadequate in leveraging structured scientific data (e.g., chemical molecular properties in databases) that encapsulate centuries of accumulated scientific expertise. These structured datasets hold strategic significance for advancing AI for Science yet current approaches merely treat them as auxiliary supplements to unstructured text. This study pioneers a systematic investigation into enhancing LLMs with structured scientific data, using chemical molecular science as a testbed. We investigate the impact of incorporating molecular property data on LLM across distinct training phases, including continual pre-training, supervised fine-tuning, and reinforcement learning. Notably, to address the inherent limitation of numerical insensitivity in large models, we propose an innovative methodology termed "Reinforcement Learning with Database Feedback" (RLDBF). Experimental evaluations demonstrate the efficacy of the proposed approach, with the model exhibiting remarkable generalization capabilities on previously unseen data and other chemical tasks. The results substantiate the potential of our method in advancing the field of structured scientific data processing within LLMs. 

---
# AIBrix: Towards Scalable, Cost-Effective Large Language Model Inference Infrastructure 

**Authors**: AIBrix Team, Jiaxin Shan, Varun Gupta, Le Xu, Haiyang Shi, Jingyuan Zhang, Ning Wang, Linhui Xu, Rong Kang, Tongping Liu, Yifei Zhang, Yiqing Zhu, Shuowei Jin, Gangmuk Lim, Binbin Chen, Zuzhi Chen, Xiao Liu, Xin Chen, Kante Yin, Chak-Pong Chung, Chenyu Jiang, Yicheng Lu, Jianjun Chen, Caixue Lin, Wu Xiang, Rui Shi, Liguang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2504.03648)  

**Abstract**: We introduce AIBrix, a cloud-native, open-source framework designed to optimize and simplify large-scale LLM deployment in cloud environments. Unlike traditional cloud-native stacks, AIBrix follows a co-design philosophy, ensuring every layer of the infrastructure is purpose-built for seamless integration with inference engines like vLLM. AIBrix introduces several key innovations to reduce inference costs and enhance performance including high-density LoRA management for dynamic adapter scheduling, LLM-specific autoscalers, and prefix-aware, load-aware routing. To further improve efficiency, AIBrix incorporates a distributed KV cache, boosting token reuse across nodes, leading to a 50% increase in throughput and a 70% reduction in inference latency. AIBrix also supports unified AI runtime which streamlines model management while maintaining vendor-agnostic engine compatibility. For large-scale multi-node inference, AIBrix employs hybrid orchestration -- leveraging Kubernetes for coarse-grained scheduling and Ray for fine-grained execution -- to balance efficiency and flexibility. Additionally, an SLO-driven GPU optimizer dynamically adjusts resource allocations, optimizing heterogeneous serving to maximize cost efficiency while maintaining service guarantees. Finally, AIBrix enhances system reliability with AI accelerator diagnostic tools, enabling automated failure detection and mock-up testing to improve fault resilience. AIBrix is available at this https URL. 

---
# CodeIF-Bench: Evaluating Instruction-Following Capabilities of Large Language Models in Interactive Code Generation 

**Authors**: Peiding Wang, Li Zhang, Fang Liu, Lin Shi, Minxiao Li, Bo Shen, An Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22688)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional performance in code generation tasks and have become indispensable programming assistants for developers. However, existing code generation benchmarks primarily assess the functional correctness of code generated by LLMs in single-turn interactions, offering limited insight into their capabilities to generate code that strictly follows users' instructions, especially in multi-turn interaction scenarios. In this paper, we introduce \bench, a benchmark for evaluating LLMs' instruction-following capabilities in interactive code generation. Specifically, \bench incorporates nine types of verifiable instructions aligned with the real-world software development requirements, which can be independently and objectively validated through specified test cases, facilitating the evaluation of instruction-following capability in multi-turn interactions. We evaluate nine prominent LLMs using \bench, and the experimental results reveal a significant disparity between their basic programming capability and instruction-following capability, particularly as task complexity, context length, and the number of dialogue rounds increase. 

---
# Can LLM-Driven Hard Negative Sampling Empower Collaborative Filtering? Findings and Potentials 

**Authors**: Chu Zhao, Enneng Yang, Yuting Liu, Jianzhe Zhao, Guibing Guo, Xingwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04726)  

**Abstract**: Hard negative samples can accelerate model convergence and optimize decision boundaries, which is key to improving the performance of recommender systems. Although large language models (LLMs) possess strong semantic understanding and generation capabilities, systematic research has not yet been conducted on how to generate hard negative samples effectively. To fill this gap, this paper introduces the concept of Semantic Negative Sampling and exploreshow to optimize LLMs for high-quality, hard negative sampling. Specifically, we design an experimental pipeline that includes three main modules, profile generation, semantic negative sampling, and semantic alignment, to verify the potential of LLM-driven hard negative sampling in enhancing the accuracy of collaborative filtering (CF). Experimental results indicate that hard negative samples generated based on LLMs, when semantically aligned and integrated into CF, can significantly improve CF performance, although there is still a certain gap compared to traditional negative sampling methods. Further analysis reveals that this gap primarily arises from two major challenges: noisy samples and lack of behavioral constraints. To address these challenges, we propose a framework called HNLMRec, based on fine-tuning LLMs supervised by collaborative signals. Experimental results show that this framework outperforms traditional negative sampling and other LLM-driven recommendation methods across multiple datasets, providing new solutions for empowering traditional RS with LLMs. Additionally, we validate the excellent generalization ability of the LLM-based semantic negative sampling method on new datasets, demonstrating its potential in alleviating issues such as data sparsity, popularity bias, and the problem of false hard negative samples. Our implementation code is available at this https URL. 

---
# Query Smarter, Trust Better? Exploring Search Behaviours for Verifying News Accuracy 

**Authors**: David Elsweiler, Samy Ateia, Markus Bink, Gregor Donabauer, Marcos Fernández Pichel, Alexander Frummet, Udo Kruschwitz, David Losada, Bernd Ludwig, Selina Meyer, Noel Pascual Presa  

**Link**: [PDF](https://arxiv.org/pdf/2504.05146)  

**Abstract**: While it is often assumed that searching for information to evaluate misinformation will help identify false claims, recent work suggests that search behaviours can instead reinforce belief in misleading news, particularly when users generate queries using vocabulary from the source articles. Our research explores how different query generation strategies affect news verification and whether the way people search influences the accuracy of their information evaluation. A mixed-methods approach was used, consisting of three parts: (1) an analysis of existing data to understand how search behaviour influences trust in fake news, (2) a simulation of query generation strategies using a Large Language Model (LLM) to assess the impact of different query formulations on search result quality, and (3) a user study to examine how 'Boost' interventions in interface design can guide users to adopt more effective query strategies. The results show that search behaviour significantly affects trust in news, with successful searches involving multiple queries and yielding higher-quality results. Queries inspired by different parts of a news article produced search results of varying quality, and weak initial queries improved when reformulated using full SERP information. Although 'Boost' interventions had limited impact, the study suggests that interface design encouraging users to thoroughly review search results can enhance query formulation. This study highlights the importance of query strategies in evaluating news and proposes that interface design can play a key role in promoting more effective search practices, serving as one component of a broader set of interventions to combat misinformation. 

---
# AiReview: An Open Platform for Accelerating Systematic Reviews with LLMs 

**Authors**: Xinyu Mao, Teerapong Leelanupab, Martin Potthast, Harrisen Scells, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2504.04193)  

**Abstract**: Systematic reviews are fundamental to evidence-based medicine. Creating one is time-consuming and labour-intensive, mainly due to the need to screen, or assess, many studies for inclusion in the review. Several tools have been developed to streamline this process, mostly relying on traditional machine learning methods. Large language models (LLMs) have shown potential in further accelerating the screening process. However, no tool currently allows end users to directly leverage LLMs for screening or facilitates systematic and transparent usage of LLM-assisted screening methods. This paper introduces (i) an extensible framework for applying LLMs to systematic review tasks, particularly title and abstract screening, and (ii) a web-based interface for LLM-assisted screening. Together, these elements form AiReview-a novel platform for LLM-assisted systematic review creation. AiReview is the first of its kind to bridge the gap between cutting-edge LLM-assisted screening methods and those that create medical systematic reviews. The tool is available at this https URL. The source code is also open sourced at this https URL. 

---
# MSL: Not All Tokens Are What You Need for Tuning LLM as a Recommender 

**Authors**: Bohao Wang, Feng Liu, Jiawei Chen, Xingyu Lou, Changwang Zhang, Jun Wang, Yuegang Sun, Yan Feng, Chun Chen, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04178)  

**Abstract**: Large language models (LLMs), known for their comprehension capabilities and extensive knowledge, have been increasingly applied to recommendation systems (RS). Given the fundamental gap between the mechanism of LLMs and the requirement of RS, researchers have focused on fine-tuning LLMs with recommendation-specific data to enhance their performance. Language Modeling Loss (LML), originally designed for language generation tasks, is commonly adopted. However, we identify two critical limitations of LML: 1) it exhibits significant divergence from the recommendation objective; 2) it erroneously treats all fictitious item descriptions as negative samples, introducing misleading training signals.
To address these limitations, we propose a novel Masked Softmax Loss (MSL) tailored for fine-tuning LLMs on recommendation. MSL improves LML by identifying and masking invalid tokens that could lead to fictitious item descriptions during loss computation. This strategy can effectively avoid the interference from erroneous negative signals and ensure well alignment with the recommendation objective supported by theoretical guarantees. During implementation, we identify a potential challenge related to gradient vanishing of MSL. To overcome this, we further introduce the temperature coefficient and propose an Adaptive Temperature Strategy (ATS) that adaptively adjusts the temperature without requiring extensive hyperparameter tuning. Extensive experiments conducted on four public datasets further validate the effectiveness of MSL, achieving an average improvement of 42.24% in NDCG@10. The code is available at this https URL. 

---
# QE-RAG: A Robust Retrieval-Augmented Generation Benchmark for Query Entry Errors 

**Authors**: Kepu Zhang, Zhongxiang Sun, Weijie Yu, Xiaoxue Zang, Kai Zheng, Yang Song, Han Li, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04062)  

**Abstract**: Retriever-augmented generation (RAG) has become a widely adopted approach for enhancing the factual accuracy of large language models (LLMs). While current benchmarks evaluate the performance of RAG methods from various perspectives, they share a common assumption that user queries used for retrieval are error-free. However, in real-world interactions between users and LLMs, query entry errors such as keyboard proximity errors, visual similarity errors, and spelling errors are frequent. The impact of these errors on current RAG methods against such errors remains largely unexplored. To bridge this gap, we propose QE-RAG, the first robust RAG benchmark designed specifically to evaluate performance against query entry errors. We augment six widely used datasets by injecting three common types of query entry errors into randomly selected user queries at rates of 20\% and 40\%, simulating typical user behavior in real-world scenarios. We analyze the impact of these errors on LLM outputs and find that corrupted queries degrade model performance, which can be mitigated through query correction and training a robust retriever for retrieving relevant documents. Based on these insights, we propose a contrastive learning-based robust retriever training method and a retrieval-augmented query correction method. Extensive in-domain and cross-domain experiments reveal that: (1) state-of-the-art RAG methods including sequential, branching, and iterative methods, exhibit poor robustness to query entry errors; (2) our method significantly enhances the robustness of RAG when handling query entry errors and it's compatible with existing RAG methods, further improving their robustness. 

---
# Automating Personalization: Prompt Optimization for Recommendation Reranking 

**Authors**: Chen Wang, Mingdai Yang, Zhiwei Liu, Pan Li, Linsey Pang, Qingsong Wen, Philip Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03965)  

**Abstract**: Modern recommender systems increasingly leverage large language models (LLMs) for reranking to improve personalization. However, existing approaches face two key limitations: (1) heavy reliance on manually crafted prompts that are difficult to scale, and (2) inadequate handling of unstructured item metadata that complicates preference inference. We present AGP (Auto-Guided Prompt Refinement), a novel framework that automatically optimizes user profile generation prompts for personalized reranking. AGP introduces two key innovations: (1) position-aware feedback mechanisms for precise ranking correction, and (2) batched training with aggregated feedback to enhance generalization. 

---
# Distillation and Refinement of Reasoning in Small Language Models for Document Re-ranking 

**Authors**: Chris Samarinas, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2504.03947)  

**Abstract**: We present a novel approach for training small language models for reasoning-intensive document ranking that combines knowledge distillation with reinforcement learning optimization. While existing methods often rely on expensive human annotations or large black-box language models, our methodology leverages web data and a teacher LLM to automatically generate high-quality training examples with relevance explanations. By framing document ranking as a reinforcement learning problem and incentivizing explicit reasoning capabilities, we train a compact 3B parameter language model that achieves state-of-the-art performance on the BRIGHT benchmark. Our model ranks third on the leaderboard while using substantially fewer parameters than other approaches, outperforming models that are over 20 times larger. Through extensive experiments, we demonstrate that generating explanations during inference, rather than directly predicting relevance scores, enables more effective reasoning with smaller language models. The self-supervised nature of our method offers a scalable and interpretable solution for modern information retrieval systems. 

---
# Practical Poisoning Attacks against Retrieval-Augmented Generation 

**Authors**: Baolei Zhang, Yuxi Chen, Minghong Fang, Zhuqing Liu, Lihai Nie, Tong Li, Zheli Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03957)  

**Abstract**: Large language models (LLMs) have demonstrated impressive natural language processing abilities but face challenges such as hallucination and outdated knowledge. Retrieval-Augmented Generation (RAG) has emerged as a state-of-the-art approach to mitigate these issues. While RAG enhances LLM outputs, it remains vulnerable to poisoning attacks. Recent studies show that injecting poisoned text into the knowledge database can compromise RAG systems, but most existing attacks assume that the attacker can insert a sufficient number of poisoned texts per query to outnumber correct-answer texts in retrieval, an assumption that is often unrealistic. To address this limitation, we propose CorruptRAG, a practical poisoning attack against RAG systems in which the attacker injects only a single poisoned text, enhancing both feasibility and stealth. Extensive experiments across multiple datasets demonstrate that CorruptRAG achieves higher attack success rates compared to existing baselines. 

---
