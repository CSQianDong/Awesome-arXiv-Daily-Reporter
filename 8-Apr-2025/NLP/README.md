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
# Do PhD-level LLMs Truly Grasp Elementary Addition? Probing Rule Learning vs. Memorization in Large Language Models 

**Authors**: Yang Yan, Yu Lu, Renjun Xu, Zhenzhong Lan  

**Link**: [PDF](https://arxiv.org/pdf/2504.05262)  

**Abstract**: Despite high benchmark scores, Large Language Models (LLMs) often fail simple problem, raising a critical question: Do LLMs learn mathematical principles or merely memorize patterns? Rather than designing increasingly complex benchmarks like recent works, we investigate this using elementary two-integer addition ($0$ to $2^{64}$), probing two core properties: commutativity ($A+B=B+A$) and compositional generalization (via isomorphic symbolic mappings, e.g., $7 \rightarrow y$). While state-of-the-art LLMs achieve 73.8-99.8\% accuracy on numerical addition, performance collapses to $\leq$7.5\% under symbolic mapping, indicating failure to generalize learned rules. Non-monotonic performance scaling with digit count and frequent commutativity violations (over 1,700 cases of $A+B \neq B+A$) further support this. Explicitly providing addition rules degrades performance by 81.2\% on average, while self-explanation maintains baseline accuracy, suggesting LLM arithmetic processing is misaligned with human-defined principles. Our findings indicate current LLMs rely on memory pattern over genuine rule learning, highlighting architectural limitations and the need for new approaches to achieve true mathematical reasoning. 

---
# LLM-based Automated Grading with Human-in-the-Loop 

**Authors**: Hang Li, Yucheng Chu, Kaiqi Yang, Yasemin Copur-Gencturk, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05239)  

**Abstract**: The rise of artificial intelligence (AI) technologies, particularly large language models (LLMs), has brought significant advancements to the field of education. Among various applications, automatic short answer grading (ASAG), which focuses on evaluating open-ended textual responses, has seen remarkable progress with the introduction of LLMs. These models not only enhance grading performance compared to traditional ASAG approaches but also move beyond simple comparisons with predefined "golden" answers, enabling more sophisticated grading scenarios, such as rubric-based evaluation. However, existing LLM-powered methods still face challenges in achieving human-level grading performance in rubric-based assessments due to their reliance on fully automated approaches. In this work, we explore the potential of LLMs in ASAG tasks by leveraging their interactive capabilities through a human-in-the-loop (HITL) approach. Our proposed framework, GradeHITL, utilizes the generative properties of LLMs to pose questions to human experts, incorporating their insights to refine grading rubrics dynamically. This adaptive process significantly improves grading accuracy, outperforming existing methods and bringing ASAG closer to human-level evaluation. 

---
# NoveltyBench: Evaluating Creativity and Diversity in Language Models 

**Authors**: Yiming Zhang, Harshita Diddee, Susan Holm, Hanchen Liu, Xinyue Liu, Vinay Samuel, Barry Wang, Daphne Ippolito  

**Link**: [PDF](https://arxiv.org/pdf/2504.05228)  

**Abstract**: Language models have demonstrated remarkable capabilities on standard benchmarks, yet they struggle increasingly from mode collapse, the inability to generate diverse and novel outputs. Our work introduces NoveltyBench, a benchmark specifically designed to evaluate the ability of language models to produce multiple distinct and high-quality outputs. NoveltyBench utilizes prompts curated to elicit diverse answers and filtered real-world user queries. Evaluating 20 leading language models, we find that current state-of-the-art systems generate significantly less diversity than human writers. Notably, larger models within a family often exhibit less diversity than their smaller counterparts, challenging the notion that capability on standard benchmarks translates directly to generative utility. While prompting strategies like in-context regeneration can elicit diversity, our findings highlight a fundamental lack of distributional diversity in current models, reducing their utility for users seeking varied responses and suggesting the need for new training and evaluation paradigms that prioritize creativity alongside quality. 

---
# Proposing TAGbank as a Corpus of Tree-Adjoining Grammar Derivations 

**Authors**: Jungyeul Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.05226)  

**Abstract**: The development of lexicalized grammars, particularly Tree-Adjoining Grammar (TAG), has significantly advanced our understanding of syntax and semantics in natural language processing (NLP). While existing syntactic resources like the Penn Treebank and Universal Dependencies offer extensive annotations for phrase-structure and dependency parsing, there is a lack of large-scale corpora grounded in lexicalized grammar formalisms. To address this gap, we introduce TAGbank, a corpus of TAG derivations automatically extracted from existing syntactic treebanks. This paper outlines a methodology for mapping phrase-structure annotations to TAG derivations, leveraging the generative power of TAG to support parsing, grammar induction, and semantic analysis. Our approach builds on the work of CCGbank, extending it to incorporate the unique structural properties of TAG, including its transparent derivation trees and its ability to capture long-distance dependencies. We also discuss the challenges involved in the extraction process, including ensuring consistency across treebank schemes and dealing with language-specific syntactic idiosyncrasies. Finally, we propose the future extension of TAGbank to include multilingual corpora, focusing on the Penn Korean and Penn Chinese Treebanks, to explore the cross-linguistic application of TAG's formalism. By providing a robust, derivation-based resource, TAGbank aims to support a wide range of computational tasks and contribute to the theoretical understanding of TAG's generative capacity. 

---
# Post-Training Language Models for Continual Relation Extraction 

**Authors**: Sefika Efeoglu, Adrian Paschke, Sonja Schimmler  

**Link**: [PDF](https://arxiv.org/pdf/2504.05214)  

**Abstract**: Real-world data, such as news articles, social media posts, and chatbot conversations, is inherently dynamic and non-stationary, presenting significant challenges for constructing real-time structured representations through knowledge graphs (KGs). Relation Extraction (RE), a fundamental component of KG creation, often struggles to adapt to evolving data when traditional models rely on static, outdated datasets. Continual Relation Extraction (CRE) methods tackle this issue by incrementally learning new relations while preserving previously acquired knowledge. This study investigates the application of pre-trained language models (PLMs), specifically large language models (LLMs), to CRE, with a focus on leveraging memory replay to address catastrophic forgetting. We evaluate decoder-only models (eg, Mistral-7B and Llama2-7B) and encoder-decoder models (eg, Flan-T5 Base) on the TACRED and FewRel datasets. Task-incremental fine-tuning of LLMs demonstrates superior performance over earlier approaches using encoder-only models like BERT on TACRED, excelling in seen-task accuracy and overall performance (measured by whole and average accuracy), particularly with the Mistral and Flan-T5 models. Results on FewRel are similarly promising, achieving second place in whole and average accuracy metrics. This work underscores critical factors in knowledge transfer, language model architecture, and KG completeness, advancing CRE with LLMs and memory replay for dynamic, real-time relation extraction. 

---
# Exploiting individual differences to bootstrap communication 

**Authors**: Richard A. Blythe, Casimir Fisch  

**Link**: [PDF](https://arxiv.org/pdf/2504.05211)  

**Abstract**: Establishing a communication system is hard because the intended meaning of a signal is unknown to its receiver when first produced, and the signaller also has no idea how that signal will be interpreted. Most theoretical accounts of the emergence of communication systems rely on feedback to reinforce behaviours that have led to successful communication in the past. However, providing such feedback requires already being able to communicate the meaning that was intended or interpreted. Therefore these accounts cannot explain how communication can be bootstrapped from non-communicative behaviours. Here we present a model that shows how a communication system, capable of expressing an unbounded number of meanings, can emerge as a result of individual behavioural differences in a large population without any pre-existing means to determine communicative success. The two key cognitive capabilities responsible for this outcome are behaving predictably in a given situation, and an alignment of psychological states ahead of signal production that derives from shared intentionality. Since both capabilities can exist independently of communication, our results are compatible with theories in which large flexible socially-learned communication systems like language are the product of a general but well-developed capacity for social cognition. 

---
# Concise Reasoning via Reinforcement Learning 

**Authors**: Mehdi Fatemi, Banafsheh Rafiee, Mingjie Tang, Kartik Talamadupula  

**Link**: [PDF](https://arxiv.org/pdf/2504.05185)  

**Abstract**: Despite significant advancements in large language models (LLMs), a major drawback of reasoning models is their enormous token usage, which increases computational cost, resource requirements, and response time. In this work, we revisit the core principles of reinforcement learning (RL) and, through mathematical analysis, demonstrate that the tendency to generate lengthy responses arises inherently from RL-based optimization during training. This finding questions the prevailing assumption that longer responses inherently improve reasoning accuracy. Instead, we uncover a natural correlation between conciseness and accuracy that has been largely overlooked. Moreover, we show that introducing a secondary phase of RL post-training, using a small set of problems and limited resources, can significantly reduce a model's chain of thought while maintaining or even enhancing accuracy. Finally, we validate our conclusions through extensive experimental results. 

---
# CARE: Aligning Language Models for Regional Cultural Awareness 

**Authors**: Geyang Guo, Tarek Naous, Hiromi Wakaki, Yukiko Nishimura, Yuki Mitsufuji, Alan Ritter, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05154)  

**Abstract**: Existing language models (LMs) often exhibit a Western-centric bias and struggle to represent diverse cultural knowledge. Previous attempts to address this rely on synthetic data and express cultural knowledge only in English. In this work, we study whether a small amount of human-written, multilingual cultural preference data can improve LMs across various model families and sizes. We first introduce CARE, a multilingual resource of 24.1k responses with human preferences on 2,580 questions about Chinese and Arab cultures, all carefully annotated by native speakers and offering more balanced coverage. Using CARE, we demonstrate that cultural alignment improves existing LMs beyond generic resources without compromising general capabilities. Moreover, we evaluate the cultural awareness of LMs, native speakers, and retrieved web content when queried in different languages. Our experiment reveals regional disparities among LMs, which may also be reflected in the documentation gap: native speakers often take everyday cultural commonsense and social norms for granted, while non-natives are more likely to actively seek out and document them. CARE is publicly available at this https URL (we plan to add Japanese data in the near future). 

---
# DoCIA: An Online Document-Level Context Incorporation Agent for Speech Translation 

**Authors**: Xinglin Lyu, Wei Tang, Yuang Li, Xiaofeng Zhao, Ming Zhu, Junhui Li, Yunfei Lu, Min Zhang, Daimeng Wei, Hao Yang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05122)  

**Abstract**: Document-level context is crucial for handling discourse challenges in text-to-text document-level machine translation (MT). Despite the increased discourse challenges introduced by noise from automatic speech recognition (ASR), the integration of document-level context in speech translation (ST) remains insufficiently explored. In this paper, we develop DoCIA, an online framework that enhances ST performance by incorporating document-level context. DoCIA decomposes the ST pipeline into four stages. Document-level context is integrated into the ASR refinement, MT, and MT refinement stages through auxiliary LLM (large language model)-based modules. Furthermore, DoCIA leverages document-level information in a multi-level manner while minimizing computational overhead. Additionally, a simple yet effective determination mechanism is introduced to prevent hallucinations from excessive refinement, ensuring the reliability of the final results. Experimental results show that DoCIA significantly outperforms traditional ST baselines in both sentence and discourse metrics across four LLMs, demonstrating its effectiveness in improving ST performance. 

---
# AI for Climate Finance: Agentic Retrieval and Multi-Step Reasoning for Early Warning System Investments 

**Authors**: Saeid Ario Vaghefi, Aymane Hachcham, Veronica Grasso, Jiska Manicus, Nakiete Msemo, Chiara Colesanti Senni, Markus Leippold  

**Link**: [PDF](https://arxiv.org/pdf/2504.05104)  

**Abstract**: Tracking financial investments in climate adaptation is a complex and expertise-intensive task, particularly for Early Warning Systems (EWS), which lack standardized financial reporting across multilateral development banks (MDBs) and funds. To address this challenge, we introduce an LLM-based agentic AI system that integrates contextual retrieval, fine-tuning, and multi-step reasoning to extract relevant financial data, classify investments, and ensure compliance with funding guidelines. Our study focuses on a real-world application: tracking EWS investments in the Climate Risk and Early Warning Systems (CREWS) Fund. We analyze 25 MDB project documents and evaluate multiple AI-driven classification methods, including zero-shot and few-shot learning, fine-tuned transformer-based classifiers, chain-of-thought (CoT) prompting, and an agent-based retrieval-augmented generation (RAG) approach. Our results show that the agent-based RAG approach significantly outperforms other methods, achieving 87\% accuracy, 89\% precision, and 83\% recall. Additionally, we contribute a benchmark dataset and expert-annotated corpus, providing a valuable resource for future research in AI-driven financial tracking and climate finance transparency. 

---
# State Tuning: State-based Test-Time Scaling on RWKV-7 

**Authors**: Liu Xiao, Li Zhiyuan, Lin Yueyu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05097)  

**Abstract**: Test-time scaling has emerged as a prominent research direction in machine learning, enabling models to enhance their expressive capabilities during this http URL, renowned for striking a delicate balance between efficiency and expressiveness, have benefited from test-time scaling techniques that leverage an expanding key-value (KV) cache to significantly improve this http URL this paper, we introduce a novel state-based approach to test-time scaling, which we term state tuning, tailored to the RNN-based RWKV-7 this http URL exploiting the unique strengths of RWKV-7, our method achieves state-of-the-art performance on the target task without altering the model's pre-trained weights. Our approach centers on three key innovations. First, we develop an observer framework that allows a smaller model to replicate and learn the state dynamics of the RWKV-7 model. Second, we employ a kernel method to dynamically upscale the state size, enhancing the model's capacity to capture intricate patterns. Third, we integrate Decorrelated Backpropagation (DBP) to optimize the upscaled state matrix, thereby improving convergence and expressivity. By tuning only the state matrix, we demonstrate that a smaller model can outperform larger models on the given task. This method preserves the efficiency of the original RWKV-7 architecture while harnessing the power of test-time scaling to deliver superior results. Our findings underscore the potential of state tuning as an effective strategy for advancing model performance in resource-constrained settings. Our code is this https URL. 

---
# The Curse of CoT: On the Limitations of Chain-of-Thought in In-Context Learning 

**Authors**: Tianshi Zheng, Yixiang Chen, Chengxi Li, Chunyang Li, Qing Zong, Haochen Shi, Baixuan Xu, Yangqiu Song, Ginny Y. Wong, Simon See  

**Link**: [PDF](https://arxiv.org/pdf/2504.05081)  

**Abstract**: Chain-of-Thought (CoT) prompting has been widely recognized for its ability to enhance reasoning capabilities in large language models (LLMs) through the generation of explicit explanatory rationales. However, our study reveals a surprising contradiction to this prevailing perspective. Through extensive experiments involving 16 state-of-the-art LLMs and nine diverse pattern-based in-context learning (ICL) datasets, we demonstrate that CoT and its reasoning variants consistently underperform direct answering across varying model scales and benchmark complexities. To systematically investigate this unexpected phenomenon, we designed extensive experiments to validate several hypothetical explanations. Our analysis uncovers a fundamental explicit-implicit duality driving CoT's performance in pattern-based ICL: while explicit reasoning falters due to LLMs' struggles to infer underlying patterns from demonstrations, implicit reasoning-disrupted by the increased contextual distance of CoT rationales-often compensates, delivering correct answers despite flawed rationales. This duality explains CoT's relative underperformance, as noise from weak explicit inference undermines the process, even as implicit mechanisms partially salvage outcomes. Notably, even long-CoT reasoning models, which excel in abstract and symbolic reasoning, fail to fully overcome these limitations despite higher computational costs. Our findings challenge existing assumptions regarding the universal efficacy of CoT, yielding novel insights into its limitations and guiding future research toward more nuanced and effective reasoning methodologies for LLMs. 

---
# On the Performance of an Explainable Language Model on PubMedQA 

**Authors**: Venkat Srinivasan, Vishaal Jatav, Anushka Chandrababu, Geetika Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2504.05074)  

**Abstract**: Large language models (LLMs) have shown significant abilities in retrieving medical knowledge, reasoning over it and answering medical questions comparably to physicians. However, these models are not interpretable, hallucinate, are difficult to maintain and require enormous compute resources for training and inference. In this paper, we report results from Gyan, an explainable language model based on an alternative architecture, on the PubmedQA data set. The Gyan LLM is a compositional language model and the model is decoupled from knowledge. Gyan is trustable, transparent, does not hallucinate and does not require significant training or compute resources. Gyan is easily transferable across domains. Gyan-4.3 achieves SOTA results on PubmedQA with 87.1% accuracy compared to 82% by MedPrompt based on GPT-4 and 81.8% by Med-PaLM 2 (Google and DeepMind). We will be reporting results for other medical data sets - MedQA, MedMCQA, MMLU - Medicine in the future. 

---
# Not All Data Are Unlearned Equally 

**Authors**: Aravind Krishnan, Siva Reddy, Marius Mosbach  

**Link**: [PDF](https://arxiv.org/pdf/2504.05058)  

**Abstract**: Machine unlearning is concerned with the task of removing knowledge learned from particular data points from a trained model. In the context of large language models (LLMs), unlearning has recently received increased attention, particularly for removing knowledge about named entities from models for privacy purposes. While various approaches have been proposed to address the unlearning problem, most existing approaches treat all data points to be unlearned equally, i.e., unlearning that Montreal is a city in Canada is treated exactly the same as unlearning the phone number of the first author of this paper. In this work, we show that this all data is equal assumption does not hold for LLM unlearning. We study how the success of unlearning depends on the frequency of the knowledge we want to unlearn in the pre-training data of a model and find that frequency strongly affects unlearning, i.e., more frequent knowledge is harder to unlearn. Additionally, we uncover a misalignment between probability and generation-based evaluations of unlearning and show that this problem worsens as models become larger. Overall, our experiments highlight the need for better evaluation practices and novel methods for LLM unlearning that take the training data of models into account. 

---
# Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models 

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau  

**Link**: [PDF](https://arxiv.org/pdf/2504.05050)  

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities. 

---
# Batch Aggregation: An Approach to Enhance Text Classification with Correlated Augmented Data 

**Authors**: Charco Hui, Yalu Wen  

**Link**: [PDF](https://arxiv.org/pdf/2504.05020)  

**Abstract**: Natural language processing models often face challenges due to limited labeled data, especially in domain specific areas, e.g., clinical trials. To overcome this, text augmentation techniques are commonly used to increases sample size by transforming the original input data into artificial ones with the label preserved. However, traditional text classification methods ignores the relationship between augmented texts and treats them as independent samples which may introduce classification error. Therefore, we propose a novel approach called 'Batch Aggregation' (BAGG) which explicitly models the dependence of text inputs generated through augmentation by incorporating an additional layer that aggregates results from correlated texts. Through studying multiple benchmark data sets across different domains, we found that BAGG can improve classification accuracy. We also found that the increase of performance with BAGG is more obvious in domain specific data sets, with accuracy improvements of up to 10-29%. Through the analysis of benchmark data, the proposed method addresses limitations of traditional techniques and improves robustness in text classification tasks. Our result demonstrates that BAGG offers more robust results and outperforms traditional approaches when training data is limited. 

---
# Surveying Professional Writers on AI: Limitations, Expectations, and Fears 

**Authors**: Anastasiia Ivanova, Natalia Fedorova, Sergey Tilga, Ekaterina Artemova  

**Link**: [PDF](https://arxiv.org/pdf/2504.05008)  

**Abstract**: The rapid development of AI-driven tools, particularly large language models (LLMs), is reshaping professional writing. Still, key aspects of their adoption such as languages support, ethics, and long-term impact on writers voice and creativity remain underexplored. In this work, we conducted a questionnaire (N = 301) and an interactive survey (N = 36) targeting professional writers regularly using AI. We examined LLM-assisted writing practices across 25+ languages, ethical concerns, and user expectations. The findings of the survey demonstrate important insights, reflecting upon the importance of: LLMs adoption for non-English speakers; the degree of misinformation, domain and style adaptation; usability and key features of LLMs. These insights can guide further development, benefiting both writers and a broader user base. 

---
# Following the Whispers of Values: Unraveling Neural Mechanisms Behind Value-Oriented Behaviors in LLMs 

**Authors**: Ling Hu, Yuemei Xu, Xiaoyang Gu, Letao Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.04994)  

**Abstract**: Despite the impressive performance of large language models (LLMs), they can present unintended biases and harmful behaviors driven by encoded values, emphasizing the urgent need to understand the value mechanisms behind them. However, current research primarily evaluates these values through external responses with a focus on AI safety, lacking interpretability and failing to assess social values in real-world contexts. In this paper, we propose a novel framework called ValueExploration, which aims to explore the behavior-driven mechanisms of National Social Values within LLMs at the neuron level. As a case study, we focus on Chinese Social Values and first construct C-voice, a large-scale bilingual benchmark for identifying and evaluating Chinese Social Values in LLMs. By leveraging C-voice, we then identify and locate the neurons responsible for encoding these values according to activation difference. Finally, by deactivating these neurons, we analyze shifts in model behavior, uncovering the internal mechanism by which values influence LLM decision-making. Extensive experiments on four representative LLMs validate the efficacy of our framework. The benchmark and code will be available. 

---
# A Domain-Based Taxonomy of Jailbreak Vulnerabilities in Large Language Models 

**Authors**: Carlos Peláez-González, Andrés Herrera-Poyatos, Cristina Zuheros, David Herrera-Poyatos, Virilo Tejedor, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2504.04976)  

**Abstract**: The study of large language models (LLMs) is a key area in open-world machine learning. Although LLMs demonstrate remarkable natural language processing capabilities, they also face several challenges, including consistency issues, hallucinations, and jailbreak vulnerabilities. Jailbreaking refers to the crafting of prompts that bypass alignment safeguards, leading to unsafe outputs that compromise the integrity of LLMs. This work specifically focuses on the challenge of jailbreak vulnerabilities and introduces a novel taxonomy of jailbreak attacks grounded in the training domains of LLMs. It characterizes alignment failures through generalization, objectives, and robustness gaps. Our primary contribution is a perspective on jailbreak, framed through the different linguistic domains that emerge during LLM training and alignment. This viewpoint highlights the limitations of existing approaches and enables us to classify jailbreak attacks on the basis of the underlying model deficiencies they exploit. Unlike conventional classifications that categorize attacks based on prompt construction methods (e.g., prompt templating), our approach provides a deeper understanding of LLM behavior. We introduce a taxonomy with four categories -- mismatched generalization, competing objectives, adversarial robustness, and mixed attacks -- offering insights into the fundamental nature of jailbreak vulnerabilities. Finally, we present key lessons derived from this taxonomic study. 

---
# Few Dimensions are Enough: Fine-tuning BERT with Selected Dimensions Revealed Its Redundant Nature 

**Authors**: Shion Fukuhata, Yoshinobu Kano  

**Link**: [PDF](https://arxiv.org/pdf/2504.04966)  

**Abstract**: When fine-tuning BERT models for specific tasks, it is common to select part of the final layer's output and input it into a newly created fully connected layer. However, it remains unclear which part of the final layer should be selected and what information each dimension of the layers holds. In this study, we comprehensively investigated the effectiveness and redundancy of token vectors, layers, and dimensions through BERT fine-tuning on GLUE tasks. The results showed that outputs other than the CLS vector in the final layer contain equivalent information, most tasks require only 2-3 dimensions, and while the contribution of lower layers decreases, there is little difference among higher layers. We also evaluated the impact of freezing pre-trained layers and conducted cross-fine-tuning, where fine-tuning is applied sequentially to different tasks. The findings suggest that hidden layers may change significantly during fine-tuning, BERT has considerable redundancy, enabling it to handle multiple tasks simultaneously, and its number of dimensions may be excessive. 

---
# Constraint Multi-class Positive and Unlabeled Learning for Distantly Supervised Named Entity Recognition 

**Authors**: Yuzhe Zhang, Min Cen, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04963)  

**Abstract**: Distantly supervised named entity recognition (DS-NER) has been proposed to exploit the automatically labeled training data by external knowledge bases instead of human annotations. However, it tends to suffer from a high false negative rate due to the inherent incompleteness. To address this issue, we present a novel approach called \textbf{C}onstraint \textbf{M}ulti-class \textbf{P}ositive and \textbf{U}nlabeled Learning (CMPU), which introduces a constraint factor on the risk estimator of multiple positive classes. It suggests that the constraint non-negative risk estimator is more robust against overfitting than previous PU learning methods with limited positive data. Solid theoretical analysis on CMPU is provided to prove the validity of our approach. Extensive experiments on two benchmark datasets that were labeled using diverse external knowledge sources serve to demonstrate the superior performance of CMPU in comparison to existing DS-NER methods. 

---
# M-Prometheus: A Suite of Open Multilingual LLM Judges 

**Authors**: José Pombal, Dongkeun Yoon, Patrick Fernandes, Ian Wu, Seungone Kim, Ricardo Rei, Graham Neubig, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2504.04953)  

**Abstract**: The use of language models for automatically evaluating long-form text (LLM-as-a-judge) is becoming increasingly common, yet most LLM judges are optimized exclusively for English, with strategies for enhancing their multilingual evaluation capabilities remaining largely unexplored in the current literature. This has created a disparity in the quality of automatic evaluation methods for non-English languages, ultimately hindering the development of models with better multilingual capabilities. To bridge this gap, we introduce M-Prometheus, a suite of open-weight LLM judges ranging from 3B to 14B parameters that can provide both direct assessment and pairwise comparison feedback on multilingual outputs. M-Prometheus models outperform state-of-the-art open LLM judges on multilingual reward benchmarks spanning more than 20 languages, as well as on literary machine translation (MT) evaluation covering 4 language pairs. Furthermore, M-Prometheus models can be leveraged at decoding time to significantly improve generated outputs across all 3 tested languages, showcasing their utility for the development of better multilingual models. Lastly, through extensive ablations, we identify the key factors for obtaining an effective multilingual judge, including backbone model selection and training on natively multilingual feedback data instead of translated data. We release our models, training dataset, and code. 

---
# Collab-RAG: Boosting Retrieval-Augmented Generation for Complex Question Answering via White-Box and Black-Box LLM Collaboration 

**Authors**: Ran Xu, Wenqi Shi, Yuchen Zhuang, Yue Yu, Joyce C. Ho, Haoyu Wang, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04915)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems often struggle to handle multi-hop question-answering tasks accurately due to irrelevant context retrieval and limited complex reasoning capabilities. We introduce Collab-RAG, a collaborative training framework that leverages mutual enhancement between a white-box small language model (SLM) and a blackbox large language model (LLM) for RAG. Specifically, the SLM decomposes complex queries into simpler sub-questions, thus enhancing the accuracy of the retrieval and facilitating more effective reasoning by the black-box LLM. Concurrently, the black-box LLM provides feedback signals to improve the SLM's decomposition capability. We observe that Collab-RAG relies solely on supervision from an affordable black-box LLM without additional distillation from frontier LLMs, yet demonstrates strong generalization across multiple black-box LLMs. Experimental evaluations across five multi-hop QA datasets demonstrate that Collab-RAG substantially outperforms existing black-box-only and SLM fine-tuning baselines by 1.8%-14.2% on average. In particular, our fine-tuned 3B SLM surpasses a frozen 32B LLM in question decomposition, highlighting the efficiency of Collab-RAG in improving reasoning and retrieval for complex questions. The code of Collab-RAG is available on this https URL. 

---
# Leveraging Large Language Models for Cost-Effective, Multilingual Depression Detection and Severity Assessment 

**Authors**: Longdi Xian, Jianzhang Ni, Mingzhu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04891)  

**Abstract**: Depression is a prevalent mental health disorder that is difficult to detect early due to subjective symptom assessments. Recent advancements in large language models have offered efficient and cost-effective approaches for this objective. In this study, we evaluated the performance of four LLMs in depression detection using clinical interview data. We selected the best performing model and further tested it in the severity evaluation scenario and knowledge enhanced scenario. The robustness was evaluated in complex diagnostic scenarios using a dataset comprising 51074 statements from six different mental disorders. We found that DeepSeek V3 is the most reliable and cost-effective model for depression detection, performing well in both zero-shot and few-shot scenarios, with zero-shot being the most efficient choice. The evaluation of severity showed low agreement with the human evaluator, particularly for mild depression. The model maintains stably high AUCs for detecting depression in complex diagnostic scenarios. These findings highlight DeepSeek V3s strong potential for text-based depression detection in real-world clinical applications. However, they also underscore the need for further refinement in severity assessment and the mitigation of potential biases to enhance clinical reliability. 

---
# SAFT: Structure-aware Transformers for Textual Interaction Classification 

**Authors**: Hongtao Wang, Renchi Yang, Hewen Wang, Haoran Zheng, Jianliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04861)  

**Abstract**: Textual interaction networks (TINs) are an omnipresent data structure used to model the interplay between users and items on e-commerce websites, social networks, etc., where each interaction is associated with a text description. Classifying such textual interactions (TIC) finds extensive use in detecting spam reviews in e-commerce, fraudulent transactions in finance, and so on. Existing TIC solutions either (i) fail to capture the rich text semantics due to the use of context-free text embeddings, and/or (ii) disregard the bipartite structure and node heterogeneity of TINs, leading to compromised TIC performance. In this work, we propose SAFT, a new architecture that integrates language- and graph-based modules for the effective fusion of textual and structural semantics in the representation learning of interactions. In particular, line graph attention (LGA)/gated attention units (GAUs) and pretrained language models (PLMs) are capitalized on to model the interaction-level and token-level signals, which are further coupled via the proxy token in an iterative and contextualized fashion. Additionally, an efficient and theoretically-grounded approach is developed to encode the local and global topology information pertaining to interactions into structural embeddings. The resulting embeddings not only inject the structural features underlying TINs into the textual interaction encoding but also facilitate the design of graph sampling strategies. Extensive empirical evaluations on multiple real TIN datasets demonstrate the superiority of SAFT over the state-of-the-art baselines in TIC accuracy. 

---
# Discovering dynamical laws for speech gestures 

**Authors**: Sam Kirkham  

**Link**: [PDF](https://arxiv.org/pdf/2504.04849)  

**Abstract**: A fundamental challenge in the cognitive sciences is discovering the dynamics that govern behaviour. Take the example of spoken language, which is characterised by a highly variable and complex set of physical movements that map onto the small set of cognitive units that comprise language. What are the fundamental dynamical principles behind the movements that structure speech production? In this study, we discover models in the form of symbolic equations that govern articulatory gestures during speech. A sparse symbolic regression algorithm is used to discover models from kinematic data on the tongue and lips. We explore these candidate models using analytical techniques and numerical simulations, and find that a second-order linear model achieves high levels of accuracy, but a nonlinear force is required to properly model articulatory dynamics in approximately one third of cases. This supports the proposal that an autonomous, nonlinear, second-order differential equation is a viable dynamical law for articulatory gestures in speech. We conclude by identifying future opportunities and obstacles in data-driven model discovery and outline prospects for discovering the dynamical principles that govern language, brain and behaviour. 

---
# Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models 

**Authors**: Ruikang Liu, Yuxuan Sun, Manyi Zhang, Haoli Bai, Xianzhi Yu, Tiezheng Yu, Chun Yuan, Lu Hou  

**Link**: [PDF](https://arxiv.org/pdf/2504.04823)  

**Abstract**: Recent advancements in reasoning language models have demonstrated remarkable performance in complex tasks, but their extended chain-of-thought reasoning process increases inference overhead. While quantization has been widely adopted to reduce the inference cost of large language models, its impact on reasoning models remains understudied. In this study, we conduct the first systematic study on quantized reasoning models, evaluating the open-sourced DeepSeek-R1-Distilled Qwen and LLaMA families ranging from 1.5B to 70B parameters, and QwQ-32B. Our investigation covers weight, KV cache, and activation quantization using state-of-the-art algorithms at varying bit-widths, with extensive evaluation across mathematical (AIME, MATH-500), scientific (GPQA), and programming (LiveCodeBench) reasoning benchmarks. Our findings reveal that while lossless quantization can be achieved with W8A8 or W4A16 quantization, lower bit-widths introduce significant accuracy risks. We further identify model size, model origin, and task difficulty as critical determinants of performance. Contrary to expectations, quantized models do not exhibit increased output lengths. In addition, strategically scaling the model sizes or reasoning steps can effectively enhance the performance. All quantized models and codes will be open-sourced in this https URL. 

---
# I only read it for the plot! Maturity Ratings Affect Fanfiction Style and Community Engagement 

**Authors**: Mia Jacobsen, Ross Deans Kristensen-McLachlan  

**Link**: [PDF](https://arxiv.org/pdf/2504.04782)  

**Abstract**: We consider the textual profiles of different fanfiction maturity ratings, how they vary across fan groups, and how this relates to reader engagement metrics. Previous studies have shown that fanfiction writing is motivated by a combination of admiration for and frustration with the fan object. These findings emerge when looking at fanfiction as a whole, as well as when it is divided into subgroups, also called fandoms. However, maturity ratings are used to indicate the intended audience of the fanfiction, as well as whether the story includes mature themes and explicit scenes. Since these ratings can be used to filter readers and writers, they can also be seen as a proxy for different reader/writer motivations and desires. We find that explicit fanfiction in particular has a distinct textual profile when compared to other maturity ratings. These findings thus nuance our understanding of reader/writer motivations in fanfiction communities, and also highlights the influence of the community norms and fan behavior more generally on these cultural products. 

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
# TathyaNyaya and FactLegalLlama: Advancing Factual Judgment Prediction and Explanation in the Indian Legal Context 

**Authors**: Shubham Kumar Nigam, Balaramamahanthi Deepak Patnaik, Shivam Mishra, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2504.04737)  

**Abstract**: In the landscape of Fact-based Judgment Prediction and Explanation (FJPE), reliance on factual data is essential for developing robust and realistic AI-driven decision-making tools. This paper introduces TathyaNyaya, the largest annotated dataset for FJPE tailored to the Indian legal context, encompassing judgments from the Supreme Court of India and various High Courts. Derived from the Hindi terms "Tathya" (fact) and "Nyaya" (justice), the TathyaNyaya dataset is uniquely designed to focus on factual statements rather than complete legal texts, reflecting real-world judicial processes where factual data drives outcomes. Complementing this dataset, we present FactLegalLlama, an instruction-tuned variant of the LLaMa-3-8B Large Language Model (LLM), optimized for generating high-quality explanations in FJPE tasks. Finetuned on the factual data in TathyaNyaya, FactLegalLlama integrates predictive accuracy with coherent, contextually relevant explanations, addressing the critical need for transparency and interpretability in AI-assisted legal systems. Our methodology combines transformers for binary judgment prediction with FactLegalLlama for explanation generation, creating a robust framework for advancing FJPE in the Indian legal domain. TathyaNyaya not only surpasses existing datasets in scale and diversity but also establishes a benchmark for building explainable AI systems in legal analysis. The findings underscore the importance of factual precision and domain-specific tuning in enhancing predictive performance and interpretability, positioning TathyaNyaya and FactLegalLlama as foundational resources for AI-assisted legal decision-making. 

---
# T1: Tool-integrated Self-verification for Test-time Compute Scaling in Small Language Models 

**Authors**: Minki Kang, Jongwon Jeong, Jaewoong Cho  

**Link**: [PDF](https://arxiv.org/pdf/2504.04718)  

**Abstract**: Recent studies have demonstrated that test-time compute scaling effectively improves the performance of small language models (sLMs). However, prior research has mainly examined test-time compute scaling with an additional larger model as a verifier, leaving self-verification by sLMs underexplored. In this work, we investigate whether sLMs can reliably self-verify their outputs under test-time scaling. We find that even with knowledge distillation from larger verifiers, sLMs struggle with verification tasks requiring memorization, such as numerical calculations and fact-checking. To address this limitation, we propose Tool-integrated self-verification (T1), which delegates memorization-heavy verification steps to external tools, such as a code interpreter. Our theoretical analysis shows that tool integration reduces memorization demands and improves test-time scaling performance. Experiments on the MATH benchmark demonstrate that, with T1, a Llama-3.2 1B model under test-time scaling outperforms the significantly larger Llama-3.1 8B model. Moreover, T1 generalizes effectively to both mathematical (MATH500) and multi-domain knowledge-intensive tasks (MMLU-Pro). Our findings highlight the potential of tool integration to substantially improve the self-verification abilities of sLMs. 

---
# Beyond Single-Turn: A Survey on Multi-Turn Interactions with Large Language Models 

**Authors**: Yubo Li, Xiaobin Shen, Xinyu Yao, Xueying Ding, Yidi Miao, Ramayya Krishnan, Rema Padman  

**Link**: [PDF](https://arxiv.org/pdf/2504.04717)  

**Abstract**: Recent advancements in large language models (LLMs) have revolutionized their ability to handle single-turn tasks, yet real-world applications demand sophisticated multi-turn interactions. This survey provides a comprehensive review of recent advancements in evaluating and enhancing multi-turn interactions in LLMs. Focusing on task-specific scenarios, from instruction following in diverse domains such as math and coding to complex conversational engagements in roleplay, healthcare, education, and even adversarial jailbreak settings, we systematically examine the challenges of maintaining context, coherence, fairness, and responsiveness over prolonged dialogues. The paper organizes current benchmarks and datasets into coherent categories that reflect the evolving landscape of multi-turn dialogue evaluation. In addition, we review a range of enhancement methodologies under multi-turn settings, including model-centric strategies (contextual learning, supervised fine-tuning, reinforcement learning, and new architectures), external integration approaches (memory-augmented, retrieval-based methods, and knowledge graph), and agent-based techniques for collaborative interactions. Finally, we discuss open challenges and propose future directions for research to further advance the robustness and effectiveness of multi-turn interactions in LLMs. Related resources and papers are available at this https URL. 

---
# Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs 

**Authors**: Will Cai, Tianneng Shi, Xuandong Zhao, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.04715)  

**Abstract**: The proliferation of Large Language Models (LLMs) accessed via black-box APIs introduces a significant trust challenge: users pay for services based on advertised model capabilities (e.g., size, performance), but providers may covertly substitute the specified model with a cheaper, lower-quality alternative to reduce operational costs. This lack of transparency undermines fairness, erodes trust, and complicates reliable benchmarking. Detecting such substitutions is difficult due to the black-box nature, typically limiting interaction to input-output queries. This paper formalizes the problem of model substitution detection in LLM APIs. We systematically evaluate existing verification techniques, including output-based statistical tests, benchmark evaluations, and log probability analysis, under various realistic attack scenarios like model quantization, randomized substitution, and benchmark evasion. Our findings reveal the limitations of methods relying solely on text outputs, especially against subtle or adaptive attacks. While log probability analysis offers stronger guarantees when available, its accessibility is often limited. We conclude by discussing the potential of hardware-based solutions like Trusted Execution Environments (TEEs) as a pathway towards provable model integrity, highlighting the trade-offs between security, performance, and provider adoption. Code is available at this https URL 

---
# Sequential-NIAH: A Needle-In-A-Haystack Benchmark for Extracting Sequential Needles from Long Contexts 

**Authors**: Yifei Yu, Qian-Wen Zhang, Lingfeng Qiao, Di Yin, Fang Li, Jie Wang, Zengxi Chen, Suncong Zheng, Xiaolong Liang, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.04713)  

**Abstract**: Evaluating the ability of large language models (LLMs) to handle extended contexts is critical, particularly for retrieving information relevant to specific queries embedded within lengthy inputs. We introduce Sequential-NIAH, a benchmark specifically designed to evaluate the capability of LLMs to extract sequential information items (known as needles) from long contexts. The benchmark comprises three types of needle generation pipelines: synthetic, real, and open-domain QA. It includes contexts ranging from 8K to 128K tokens in length, with a dataset of 14,000 samples (2,000 reserved for testing). To facilitate evaluation on this benchmark, we trained a synthetic data-driven evaluation model capable of evaluating answer correctness based on chronological or logical order, achieving an accuracy of 99.49% on synthetic test data. We conducted experiments on six well-known LLMs, revealing that even the best-performing model achieved a maximum accuracy of only 63.15%. Further analysis highlights the growing challenges posed by increasing context lengths and the number of needles, underscoring substantial room for improvement. Additionally, noise robustness experiments validate the reliability of the benchmark, making Sequential-NIAH an important reference for advancing research on long text extraction capabilities of LLMs. 

---
# Causal Retrieval with Semantic Consideration 

**Authors**: Hyunseo Shin, Wonseok Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04700)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly enhanced the performance of conversational AI systems. To extend their capabilities to knowledge-intensive domains such as biomedical and legal fields, where the accuracy is critical, LLMs are often combined with information retrieval (IR) systems to generate responses based on retrieved documents. However, for IR systems to effectively support such applications, they must go beyond simple semantic matching and accurately capture diverse query intents, including causal relationships. Existing IR models primarily focus on retrieving documents based on surface-level semantic similarity, overlooking deeper relational structures such as causality. To address this, we propose CAWAI, a retrieval model that is trained with dual objectives: semantic and causal relations. Our extensive experiments demonstrate that CAWAI outperforms various models on diverse causal retrieval tasks especially under large-scale retrieval settings. We also show that CAWAI exhibits strong zero-shot generalization across scientific domain QA tasks. 

---
# scAgent: Universal Single-Cell Annotation via a LLM Agent 

**Authors**: Yuren Mao, Yu Mi, Peigen Liu, Mengfei Zhang, Hanqing Liu, Yunjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.04698)  

**Abstract**: Cell type annotation is critical for understanding cellular heterogeneity. Based on single-cell RNA-seq data and deep learning models, good progress has been made in annotating a fixed number of cell types within a specific tissue. However, universal cell annotation, which can generalize across tissues, discover novel cell types, and extend to novel cell types, remains less explored. To fill this gap, this paper proposes scAgent, a universal cell annotation framework based on Large Language Models (LLMs). scAgent can identify cell types and discover novel cell types in diverse tissues; furthermore, it is data efficient to learn novel cell types. Experimental studies in 160 cell types and 35 tissues demonstrate the superior performance of scAgent in general cell-type annotation, novel cell discovery, and extensibility to novel cell type. 

---
# Splits! A Flexible Dataset for Evaluating a Model's Demographic Social Inference 

**Authors**: Eylon Caplan, Tania Chakraborty, Dan Goldwasser  

**Link**: [PDF](https://arxiv.org/pdf/2504.04640)  

**Abstract**: Understanding how people of various demographics think, feel, and express themselves (collectively called group expression) is essential for social science and underlies the assessment of bias in Large Language Models (LLMs). While LLMs can effectively summarize group expression when provided with empirical examples, coming up with generalizable theories of how a group's expression manifests in real-world text is challenging. In this paper, we define a new task called Group Theorization, in which a system must write theories that differentiate expression across demographic groups. We make available a large dataset on this task, Splits!, constructed by splitting Reddit posts by neutral topics (e.g. sports, cooking, and movies) and by demographics (e.g. occupation, religion, and race). Finally, we suggest a simple evaluation framework for assessing how effectively a method can generate 'better' theories about group expression, backed by human validation. We publicly release the raw corpora and evaluation scripts for Splits! to help researchers assess how methods infer--and potentially misrepresent--group differences in expression. We make Splits! and our evaluation module available at this https URL. 

---
# Steering off Course: Reliability Challenges in Steering Language Models 

**Authors**: Patrick Queiroz Da Silva, Hari Sethuraman, Dheeraj Rajagopal, Hannaneh Hajishirzi, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2504.04635)  

**Abstract**: Steering methods for language models (LMs) have gained traction as lightweight alternatives to fine-tuning, enabling targeted modifications to model activations. However, prior studies primarily report results on a few models, leaving critical gaps in understanding the robustness of these methods. In this work, we systematically examine three prominent steering methods -- DoLa, function vectors, and task vectors. In contrast to the original studies, which evaluated a handful of models, we test up to 36 models belonging to 14 families with sizes ranging from 1.5B to 70B parameters. Our experiments reveal substantial variability in the effectiveness of the steering approaches, with a large number of models showing no improvement and at times degradation in steering performance. Our analysis demonstrate fundamental flaws in the assumptions underlying these methods, challenging their reliability as scalable steering solutions. 

---
# DynClean: Training Dynamics-based Label Cleaning for Distantly-Supervised Named Entity Recognition 

**Authors**: Qi Zhang, Huitong Pan, Zhijia Chen, Longin Jan Latecki, Cornelia Caragea, Eduard Dragut  

**Link**: [PDF](https://arxiv.org/pdf/2504.04616)  

**Abstract**: Distantly Supervised Named Entity Recognition (DS-NER) has attracted attention due to its scalability and ability to automatically generate labeled data. However, distant annotation introduces many mislabeled instances, limiting its performance. Most of the existing work attempt to solve this problem by developing intricate models to learn from the noisy labels. An alternative approach is to attempt to clean the labeled data, thus increasing the quality of distant labels. This approach has received little attention for NER. In this paper, we propose a training dynamics-based label cleaning approach, which leverages the behavior of a model as training progresses to characterize the distantly annotated samples. We also introduce an automatic threshold estimation strategy to locate the errors in distant labels. Extensive experimental results demonstrate that: (1) models trained on our cleaned DS-NER datasets, which were refined by directly removing identified erroneous annotations, achieve significant improvements in F1-score, ranging from 3.18% to 8.95%; and (2) our method outperforms numerous advanced DS-NER approaches across four datasets. 

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
# Saliency-driven Dynamic Token Pruning for Large Language Models 

**Authors**: Yao Tao, Yehui Tang, Yun Wang, Mingjian Zhu, Hailin Hu, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04514)  

**Abstract**: Despite the recent success of large language models (LLMs), LLMs are particularly challenging in long-sequence inference scenarios due to the quadratic computational complexity of the attention mechanism. Inspired by the interpretability theory of feature attribution in neural network models, we observe that not all tokens have the same contribution. Based on this observation, we propose a novel token pruning framework, namely Saliency-driven Dynamic Token Pruning (SDTP), to gradually and dynamically prune redundant tokens based on the input context. Specifically, a lightweight saliency-driven prediction module is designed to estimate the importance score of each token with its hidden state, which is added to different layers of the LLM to hierarchically prune redundant tokens. Furthermore, a ranking-based optimization strategy is proposed to minimize the ranking divergence of the saliency score and the predicted importance score. Extensive experiments have shown that our framework is generalizable to various models and datasets. By hierarchically pruning 65\% of the input tokens, our method greatly reduces 33\% $\sim$ 47\% FLOPs and achieves speedup up to 1.75$\times$ during inference, while maintaining comparable performance. We further demonstrate that SDTP can be combined with KV cache compression method for further compression. 

---
# Directed Graph-alignment Approach for Identification of Gaps in Short Answers 

**Authors**: Archana Sahu, Plaban Kumar Bhowmick  

**Link**: [PDF](https://arxiv.org/pdf/2504.04473)  

**Abstract**: In this paper, we have presented a method for identifying missing items known as gaps in the student answers by comparing them against the corresponding model answer/reference answers, automatically. The gaps can be identified at word, phrase or sentence level. The identified gaps are useful in providing feedback to the students for formative assessment. The problem of gap identification has been modelled as an alignment of a pair of directed graphs representing a student answer and the corresponding model answer for a given question. To validate the proposed approach, the gap annotated student answers considering answers from three widely known datasets in the short answer grading domain, namely, University of North Texas (UNT), SciEntsBank, and Beetle have been developed and this gap annotated student answers' dataset is available at: this https URL. Evaluation metrics used in the traditional machine learning tasks have been adopted to evaluate the task of gap identification. Though performance of the proposed approach varies across the datasets and the types of the answers, overall the performance is observed to be promising. 

---
# An overview of model uncertainty and variability in LLM-based sentiment analysis. Challenges, mitigation strategies and the role of explainability 

**Authors**: David Herrera-Poyatos, Carlos Peláez-González, Cristina Zuheros, Andrés Herrera-Poyatos, Virilo Tejedor, Francisco Herrera, Rosana Montes  

**Link**: [PDF](https://arxiv.org/pdf/2504.04462)  

**Abstract**: Large Language Models (LLMs) have significantly advanced sentiment analysis, yet their inherent uncertainty and variability pose critical challenges to achieving reliable and consistent outcomes. This paper systematically explores the Model Variability Problem (MVP) in LLM-based sentiment analysis, characterized by inconsistent sentiment classification, polarization, and uncertainty arising from stochastic inference mechanisms, prompt sensitivity, and biases in training data. We analyze the core causes of MVP, presenting illustrative examples and a case study to highlight its impact. In addition, we investigate key challenges and mitigation strategies, paying particular attention to the role of temperature as a driver of output randomness and emphasizing the crucial role of explainability in improving transparency and user trust. By providing a structured perspective on stability, reproducibility, and trustworthiness, this study helps develop more reliable, explainable, and robust sentiment analysis models, facilitating their deployment in high-stakes domains such as finance, healthcare, and policymaking, among others. 

---
# On the Spatial Structure of Mixture-of-Experts in Transformers 

**Authors**: Daniel Bershatsky, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2504.04444)  

**Abstract**: A common assumption is that MoE routers primarily leverage semantic features for expert selection. However, our study challenges this notion by demonstrating that positional token information also plays a crucial role in routing decisions. Through extensive empirical analysis, we provide evidence supporting this hypothesis, develop a phenomenological explanation of the observed behavior, and discuss practical implications for MoE-based architectures. 

---
# Pre-trained Language Models and Few-shot Learning for Medical Entity Extraction 

**Authors**: Xiaokai Wang, Guiran Liu, Binrong Zhu, Jacky He, Hongye Zheng, Hanlu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04385)  

**Abstract**: This study proposes a medical entity extraction method based on Transformer to enhance the information extraction capability of medical literature. Considering the professionalism and complexity of medical texts, we compare the performance of different pre-trained language models (BERT, BioBERT, PubMedBERT, ClinicalBERT) in medical entity extraction tasks. Experimental results show that PubMedBERT achieves the best performance (F1-score = 88.8%), indicating that a language model pre-trained on biomedical literature is more effective in the medical domain. In addition, we analyze the impact of different entity extraction methods (CRF, Span-based, Seq2Seq) and find that the Span-based approach performs best in medical entity extraction tasks (F1-score = 88.6%). It demonstrates superior accuracy in identifying entity boundaries. In low-resource scenarios, we further explore the application of Few-shot Learning in medical entity extraction. Experimental results show that even with only 10-shot training samples, the model achieves an F1-score of 79.1%, verifying the effectiveness of Few-shot Learning under limited data conditions. This study confirms that the combination of pre-trained language models and Few-shot Learning can enhance the accuracy of medical entity extraction. Future research can integrate knowledge graphs and active learning strategies to improve the model's generalization and stability, providing a more effective solution for medical NLP research. Keywords- Natural Language Processing, medical named entity recognition, pre-trained language model, Few-shot Learning, information extraction, deep learning 

---
# PolyGuard: A Multilingual Safety Moderation Tool for 17 Languages 

**Authors**: Priyanshu Kumar, Devansh Jain, Akhila Yerukola, Liwei Jiang, Himanshu Beniwal, Thomas Hartvigsen, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2504.04377)  

**Abstract**: Truly multilingual safety moderation efforts for Large Language Models (LLMs) have been hindered by a narrow focus on a small set of languages (e.g., English, Chinese) as well as a limited scope of safety definition, resulting in significant gaps in moderation capabilities. To bridge these gaps, we release POLYGUARD, a new state-of-the-art multilingual safety model for safeguarding LLM generations, and the corresponding training and evaluation datasets. POLYGUARD is trained on POLYGUARDMIX, the largest multilingual safety training corpus to date containing 1.91M samples across 17 languages (e.g., Chinese, Czech, English, Hindi). We also introduce POLYGUARDPROMPTS, a high quality multilingual benchmark with 29K samples for the evaluation of safety guardrails. Created by combining naturally occurring multilingual human-LLM interactions and human-verified machine translations of an English-only safety dataset (WildGuardMix; Han et al., 2024), our datasets contain prompt-output pairs with labels of prompt harmfulness, response harmfulness, and response refusal. Through extensive evaluations across multiple safety and toxicity benchmarks, we demonstrate that POLYGUARD outperforms existing state-of-the-art open-weight and commercial safety classifiers by 5.5%. Our contributions advance efforts toward safer multilingual LLMs for all global users. 

---
# StyleRec: A Benchmark Dataset for Prompt Recovery in Writing Style Transformation 

**Authors**: Shenyang Liu, Yang Gao, Shaoyan Zhai, Liqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04373)  

**Abstract**: Prompt Recovery, reconstructing prompts from the outputs of large language models (LLMs), has grown in importance as LLMs become ubiquitous. Most users access LLMs through APIs without internal model weights, relying only on outputs and logits, which complicates recovery. This paper explores a unique prompt recovery task focused on reconstructing prompts for style transfer and rephrasing, rather than typical question-answering. We introduce a dataset created with LLM assistance, ensuring quality through multiple techniques, and test methods like zero-shot, few-shot, jailbreak, chain-of-thought, fine-tuning, and a novel canonical-prompt fallback for poor-performing cases. Our results show that one-shot and fine-tuning yield the best outcomes but highlight flaws in traditional sentence similarity metrics for evaluating prompt recovery. Contributions include (1) a benchmark dataset, (2) comprehensive experiments on prompt recovery strategies, and (3) identification of limitations in current evaluation metrics, all of which advance general prompt recovery research, where the structure of the input prompt is unrestricted. 

---
# Compression Laws for Large Language Models 

**Authors**: Ayan Sengupta, Siddhant Chaudhary, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2504.04342)  

**Abstract**: We introduce compression laws for language language models (LLMs). While recent scaling laws have sought to understand how LLMs scale with respect to model size, pre-training data, and computational resources, we focus on understanding how model compression affects the performance of a pre-trained LLM on downstream tasks. We empirically examine the effects of structured model compression on LLMs through over $1000$ experiments across eight models with sizes ranging from $0.5B$ to $14B$ parameters. Our findings indicate that the test cross-entropy loss increases quadratically with the compression ratio, whereas performance on downstream tasks declines only linearly. Our study emphasizes the importance of recovery fine-tuning in enhancing generation loss, showing that the test loss of compressed LLMs can improve by up to 55% with recovery fine-tuning. At higher compression ratios (up to 90%), compressed LLMs demonstrate a speed increase of 60% during inference compared to their uncompressed counterparts, compensating for the performance degradation at this level. However, for smaller models ($\le 7B$), the computational gains are limited, peaking at just 35%. We conclude that model compression can be highly beneficial for larger models, especially when a smaller model within the same computational budget is not available. These insights provide the practical guidelines for utilizing model compression techniques for adopting LLMs in real-life applications in resource-constrained settings. 

---
# Generative Large Language Models Trained for Detecting Errors in Radiology Reports 

**Authors**: Cong Sun, Kurt Teichman, Yiliang Zhou, Brian Critelli, David Nauheim, Graham Keir, Xindi Wang, Judy Zhong, Adam E Flanders, George Shih, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2504.04336)  

**Abstract**: In this retrospective study, a dataset was constructed with two parts. The first part included 1,656 synthetic chest radiology reports generated by GPT-4 using specified prompts, with 828 being error-free synthetic reports and 828 containing errors. The second part included 614 reports: 307 error-free reports between 2011 and 2016 from the MIMIC-CXR database and 307 corresponding synthetic reports with errors generated by GPT-4 on the basis of these MIMIC-CXR reports and specified prompts. All errors were categorized into four types: negation, left/right, interval change, and transcription errors. Then, several models, including Llama-3, GPT-4, and BiomedBERT, were refined using zero-shot prompting, few-shot prompting, or fine-tuning strategies. Finally, the performance of these models was evaluated using the F1 score, 95\% confidence interval (CI) and paired-sample t-tests on our constructed dataset, with the prediction results further assessed by radiologists. Using zero-shot prompting, the fine-tuned Llama-3-70B-Instruct model achieved the best performance with the following F1 scores: 0.769 for negation errors, 0.772 for left/right errors, 0.750 for interval change errors, 0.828 for transcription errors, and 0.780 overall. In the real-world evaluation phase, two radiologists reviewed 200 randomly selected reports output by the model. Of these, 99 were confirmed to contain errors detected by the models by both radiologists, and 163 were confirmed to contain model-detected errors by at least one radiologist. Generative LLMs, fine-tuned on synthetic and MIMIC-CXR radiology reports, greatly enhanced error detection in radiology reports. 

---
# Hallucination Detection using Multi-View Attention Features 

**Authors**: Yuya Ogasa, Yuki Arase  

**Link**: [PDF](https://arxiv.org/pdf/2504.04335)  

**Abstract**: This study tackles token-level hallucination detection in outputs of large language models. Previous studies revealed that attention exhibits irregular patterns when hallucination occurs. Inspired by this, we extract features from the attention matrix that provide complementary views of (a) the average attention each token receives, which helps identify whether certain tokens are overly influential or ignored, (b) the diversity of attention each token receives, which reveals whether attention is biased toward specific subsets, and (c) the diversity of tokens a token attends to during generation, which indicates whether the model references a narrow or broad range of information. These features are input to a Transformer-based classifier to conduct token-level classification to identify hallucinated spans. Experimental results indicate that the proposed method outperforms strong baselines on hallucination detection with longer input contexts, i.e., data-to-text and summarization tasks. 

---
# IMPersona: Evaluating Individual Level LM Impersonation 

**Authors**: Quan Shi, Carlos Jimenez, Stephen Dong, Brian Seo, Caden Yao, Adam Kelch, Karthik Narasimhan  

**Link**: [PDF](https://arxiv.org/pdf/2504.04332)  

**Abstract**: As language models achieve increasingly human-like capabilities in conversational text generation, a critical question emerges: to what extent can these systems simulate the characteristics of specific individuals? To evaluate this, we introduce IMPersona, a framework for evaluating LMs at impersonating specific individuals' writing style and personal knowledge. Using supervised fine-tuning and a hierarchical memory-inspired retrieval system, we demonstrate that even modestly sized open-source models, such as Llama-3.1-8B-Instruct, can achieve impersonation abilities at concerning levels. In blind conversation experiments, participants (mis)identified our fine-tuned models with memory integration as human in 44.44% of interactions, compared to just 25.00% for the best prompting-based approach. We analyze these results to propose detection methods and defense strategies against such impersonation attempts. Our findings raise important questions about both the potential applications and risks of personalized language models, particularly regarding privacy, security, and the ethical deployment of such technologies in real-world contexts. 

---
# Constructing the Truth: Text Mining and Linguistic Networks in Public Hearings of Case 03 of the Special Jurisdiction for Peace (JEP) 

**Authors**: Juan Sosa, Alejandro Urrego, Cesar Prieto, Emma J. Camargo-Díaz  

**Link**: [PDF](https://arxiv.org/pdf/2504.04325)  

**Abstract**: Case 03 of the Special Jurisdiction for Peace (JEP), focused on the so-called false positives in Colombia, represents one of the most harrowing episodes of the Colombian armed conflict. This article proposes an innovative methodology based on natural language analysis and semantic co-occurrence models to explore, systematize, and visualize narrative patterns present in the public hearings of victims and appearing parties. By constructing skipgram networks and analyzing their modularity, the study identifies thematic clusters that reveal regional and procedural status differences, providing empirical evidence on dynamics of victimization, responsibility, and acknowledgment in this case. This computational approach contributes to the collective construction of both judicial and extrajudicial truth, offering replicable tools for other transitional justice cases. The work is grounded in the pillars of truth, justice, reparation, and non-repetition, proposing a critical and in-depth reading of contested memories. 

---
# Balancing Complexity and Informativeness in LLM-Based Clustering: Finding the Goldilocks Zone 

**Authors**: Justin Miller, Tristram Alexander  

**Link**: [PDF](https://arxiv.org/pdf/2504.04314)  

**Abstract**: The challenge of clustering short text data lies in balancing informativeness with interpretability. Traditional evaluation metrics often overlook this trade-off. Inspired by linguistic principles of communicative efficiency, this paper investigates the optimal number of clusters by quantifying the trade-off between informativeness and cognitive simplicity. We use large language models (LLMs) to generate cluster names and evaluate their effectiveness through semantic density, information theory, and clustering accuracy. Our results show that Gaussian Mixture Model (GMM) clustering on embeddings generated by a LLM, increases semantic density compared to random assignment, effectively grouping similar bios. However, as clusters increase, interpretability declines, as measured by a generative LLM's ability to correctly assign bios based on cluster names. A logistic regression analysis confirms that classification accuracy depends on the semantic similarity between bios and their assigned cluster names, as well as their distinction from alternatives.
These findings reveal a "Goldilocks zone" where clusters remain distinct yet interpretable. We identify an optimal range of 16-22 clusters, paralleling linguistic efficiency in lexical categorization. These insights inform both theoretical models and practical applications, guiding future research toward optimising cluster interpretability and usefulness. 

---
# CO-Bench: Benchmarking Language Model Agents in Algorithm Search for Combinatorial Optimization 

**Authors**: Weiwei Sun, Shengyu Feng, Shanda Li, Yiming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04310)  

**Abstract**: Although LLM-based agents have attracted significant attention in domains such as software engineering and machine learning research, their role in advancing combinatorial optimization (CO) remains relatively underexplored. This gap underscores the need for a deeper understanding of their potential in tackling structured, constraint-intensive problems-a pursuit currently limited by the absence of comprehensive benchmarks for systematic investigation. To address this, we introduce CO-Bench, a benchmark suite featuring 36 real-world CO problems drawn from a broad range of domains and complexity levels. CO-Bench includes structured problem formulations and curated data to support rigorous investigation of LLM agents. We evaluate multiple agent frameworks against established human-designed algorithms, revealing key strengths and limitations of current approaches and identifying promising directions for future research. CO-Bench is publicly available at this https URL. 

---
# Dynamic Hedging Strategies in Derivatives Markets with LLM-Driven Sentiment and News Analytics 

**Authors**: Jie Yang, Yiqiu Tang, Yongjie Li, Lihua Zhang, Haoran Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04295)  

**Abstract**: Dynamic hedging strategies are essential for effective risk management in derivatives markets, where volatility and market sentiment can greatly impact performance. This paper introduces a novel framework that leverages large language models (LLMs) for sentiment analysis and news analytics to inform hedging decisions. By analyzing textual data from diverse sources like news articles, social media, and financial reports, our approach captures critical sentiment indicators that reflect current market conditions. The framework allows for real-time adjustments to hedging strategies, adapting positions based on continuous sentiment signals. Backtesting results on historical derivatives data reveal that our dynamic hedging strategies achieve superior risk-adjusted returns compared to conventional static approaches. The incorporation of LLM-driven sentiment analysis into hedging practices presents a significant advancement in decision-making processes within derivatives trading. This research showcases how sentiment-informed dynamic hedging can enhance portfolio management and effectively mitigate associated risks. 

---
# Cross-Asset Risk Management: Integrating LLMs for Real-Time Monitoring of Equity, Fixed Income, and Currency Markets 

**Authors**: Jie Yang, Yiqiu Tang, Yongjie Li, Lihua Zhang, Haoran Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04292)  

**Abstract**: Large language models (LLMs) have emerged as powerful tools in the field of finance, particularly for risk management across different asset classes. In this work, we introduce a Cross-Asset Risk Management framework that utilizes LLMs to facilitate real-time monitoring of equity, fixed income, and currency markets. This innovative approach enables dynamic risk assessment by aggregating diverse data sources, ultimately enhancing decision-making processes. Our model effectively synthesizes and analyzes market signals to identify potential risks and opportunities while providing a holistic view of asset classes. By employing advanced analytics, we leverage LLMs to interpret financial texts, news articles, and market reports, ensuring that risks are contextualized within broader market narratives. Extensive backtesting and real-time simulations validate the framework, showing increased accuracy in predicting market shifts compared to conventional methods. The focus on real-time data integration enhances responsiveness, allowing financial institutions to manage risks adeptly under varying market conditions and promoting financial stability through the advanced application of LLMs in risk analysis. 

---
# Could AI Trace and Explain the Origins of AI-Generated Images and Text? 

**Authors**: Hongchao Fang, Can Qin, Ran Xu, Feng Liu, Yixin Liu, Lichao Sun, Dongwon Lee, Lifu Huang, Wenpeng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2504.04279)  

**Abstract**: AI-generated content is becoming increasingly prevalent in the real world, leading to serious ethical and societal concerns. For instance, adversaries might exploit large multimodal models (LMMs) to create images that violate ethical or legal standards, while paper reviewers may misuse large language models (LLMs) to generate reviews without genuine intellectual effort. While prior work has explored detecting AI-generated images and texts, and occasionally tracing their source models, there is a lack of a systematic and fine-grained comparative study. Important dimensions--such as AI-generated images vs. text, fully vs. partially AI-generated images, and general vs. malicious use cases--remain underexplored. Furthermore, whether AI systems like GPT-4o can explain why certain forged content is attributed to specific generative models is still an open question, with no existing benchmark addressing this. To fill this gap, we introduce AI-FAKER, a comprehensive multimodal dataset with over 280,000 samples spanning multiple LLMs and LMMs, covering both general and malicious use cases for AI-generated images and texts. Our experiments reveal two key findings: (i) AI authorship detection depends not only on the generated output but also on the model's original training intent; and (ii) GPT-4o provides highly consistent but less specific explanations when analyzing content produced by OpenAI's own models, such as DALL-E and GPT-4o itself. 

---
# negativas: a prototype for searching and classifying sentential negation in speech data 

**Authors**: Túlio Sousa de Gois, Paloma Batista Cardoso  

**Link**: [PDF](https://arxiv.org/pdf/2504.04275)  

**Abstract**: Negation is a universal feature of natural languages. In Brazilian Portuguese, the most commonly used negation particle is não, which can scope over nouns or verbs. When it scopes over a verb, não can occur in three positions: pre-verbal (NEG1), double negation (NEG2), or post-verbal (NEG3), e.g., não gosto, não gosto não, gosto não ("I do not like it"). From a variationist perspective, these structures are different forms of expressing negation. Pragmatically, they serve distinct communicative functions, such as politeness and modal evaluation. Despite their grammatical acceptability, these forms differ in frequency. NEG1 dominates across Brazilian regions, while NEG2 and NEG3 appear more rarely, suggesting its use is contextually restricted. This low-frequency challenges research, often resulting in subjective, non-generalizable interpretations of verbal negation with não. To address this, we developed negativas, a tool for automatically identifying NEG1, NEG2, and NEG3 in transcribed data. The tool's development involved four stages: i) analyzing a dataset of 22 interviews from the Falares Sergipanos database, annotated by three linguists, ii) creating a code using natural language processing (NLP) techniques, iii) running the tool, iv) evaluating accuracy. Inter-annotator consistency, measured using Fleiss' Kappa, was moderate (0.57). The tool identified 3,338 instances of não, classifying 2,085 as NEG1, NEG2, or NEG3, achieving a 93% success rate. However, negativas has limitations. NEG1 accounted for 91.5% of identified structures, while NEG2 and NEG3 represented 7.2% and 1.2%, respectively. The tool struggled with NEG2, sometimes misclassifying instances as overlapping structures (NEG1/NEG2/NEG3). 

---
# Lost in Multilinguality: Dissecting Cross-lingual Factual Inconsistency in Transformer Language Models 

**Authors**: Mingyang Wang, Heike Adel, Lukas Lange, Yihong Liu, Ercong Nie, Jannik Strötgen, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2504.04264)  

**Abstract**: Multilingual language models (MLMs) store factual knowledge across languages but often struggle to provide consistent responses to semantically equivalent prompts in different languages. While previous studies point out this cross-lingual inconsistency issue, the underlying causes remain unexplored. In this work, we use mechanistic interpretability methods to investigate cross-lingual inconsistencies in MLMs. We find that MLMs encode knowledge in a language-independent concept space through most layers, and only transition to language-specific spaces in the final layers. Failures during the language transition often result in incorrect predictions in the target language, even when the answers are correct in other languages. To mitigate this inconsistency issue, we propose a linear shortcut method that bypasses computations in the final layers, enhancing both prediction accuracy and cross-lingual consistency. Our findings shed light on the internal mechanisms of MLMs and provide a lightweight, effective strategy for producing more consistent factual outputs. 

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
# Towards Understanding and Improving Refusal in Compressed Models via Mechanistic Interpretability 

**Authors**: Vishnu Kabir Chhabra, Mohammad Mahdi Khalili  

**Link**: [PDF](https://arxiv.org/pdf/2504.04215)  

**Abstract**: The rapid growth of large language models has spurred significant interest in model compression as a means to enhance their accessibility and practicality. While extensive research has explored model compression through the lens of safety, findings suggest that safety-aligned models often lose elements of trustworthiness post-compression. Simultaneously, the field of mechanistic interpretability has gained traction, with notable discoveries, such as the identification of a single direction in the residual stream mediating refusal behaviors across diverse model architectures. In this work, we investigate the safety of compressed models by examining the mechanisms of refusal, adopting a novel interpretability-driven perspective to evaluate model safety. Furthermore, leveraging insights from our interpretability analysis, we propose a lightweight, computationally efficient method to enhance the safety of compressed models without compromising their performance or utility. 

---
# Adaptive Elicitation of Latent Information Using Natural Language 

**Authors**: Jimmy Wang, Thomas Zollo, Richard Zemel, Hongseok Namkoong  

**Link**: [PDF](https://arxiv.org/pdf/2504.04204)  

**Abstract**: Eliciting information to reduce uncertainty about a latent entity is a critical task in many application domains, e.g., assessing individual student learning outcomes, diagnosing underlying diseases, or learning user preferences. Though natural language is a powerful medium for this purpose, large language models (LLMs) and existing fine-tuning algorithms lack mechanisms for strategically gathering information to refine their own understanding of the latent entity. To harness the generalization power and world knowledge of LLMs in developing effective information-gathering strategies, we propose an adaptive elicitation framework that actively reduces uncertainty on the latent entity. Since probabilistic modeling of an abstract latent entity is difficult, our framework adopts a predictive view of uncertainty, using a meta-learned language model to simulate future observations and enable scalable uncertainty quantification over complex natural language. Through autoregressive forward simulation, our model quantifies how new questions reduce epistemic uncertainty, enabling the development of sophisticated information-gathering strategies to choose the most informative next queries. In experiments on the 20 questions game, dynamic opinion polling, and adaptive student assessment, our method consistently outperforms baselines in identifying critical unknowns and improving downstream predictions, illustrating the promise of strategic information gathering in natural language settings. 

---
# GlotEval: A Test Suite for Massively Multilingual Evaluation of Large Language Models 

**Authors**: Hengyu Luo, Zihao Li, Joseph Attieh, Sawal Devkota, Ona de Gibert, Shaoxiong Ji, Peiqin Lin, Bhavani Sai Praneeth Varma Mantina, Ananda Sreenidhi, Raúl Vázquez, Mengjie Wang, Samea Yusofi, Jörg Tiedemann  

**Link**: [PDF](https://arxiv.org/pdf/2504.04155)  

**Abstract**: Large language models (LLMs) are advancing at an unprecedented pace globally, with regions increasingly adopting these models for applications in their primary language. Evaluation of these models in diverse linguistic environments, especially in low-resource languages, has become a major challenge for academia and industry. Existing evaluation frameworks are disproportionately focused on English and a handful of high-resource languages, thereby overlooking the realistic performance of LLMs in multilingual and lower-resource scenarios. To address this gap, we introduce GlotEval, a lightweight framework designed for massively multilingual evaluation. Supporting seven key tasks (machine translation, text classification, summarization, open-ended generation, reading comprehension, sequence labeling, and intrinsic evaluation), spanning over dozens to hundreds of languages, GlotEval highlights consistent multilingual benchmarking, language-specific prompt templates, and non-English-centric machine translation. This enables a precise diagnosis of model strengths and weaknesses in diverse linguistic contexts. A multilingual translation case study demonstrates GlotEval's applicability for multilingual and language-specific evaluations. 

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
# Reasoning on Multiple Needles In A Haystack 

**Authors**: Yidong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04150)  

**Abstract**: The Needle In A Haystack (NIAH) task has been widely used to evaluate the long-context question-answering capabilities of Large Language Models (LLMs). However, its reliance on simple retrieval limits its effectiveness. To address this limitation, recent studies have introduced the Multiple Needles In A Haystack Reasoning (MNIAH-R) task, which incorporates supporting documents (Multiple needles) of multi-hop reasoning tasks into a distracting context (Haystack}). Despite this advancement, existing approaches still fail to address the issue of models providing direct answers from internal knowledge, and they do not explain or mitigate the decline in accuracy as context length increases. In this paper, we tackle the memory-based answering problem by filtering out direct-answer questions, and we reveal that performance degradation is primarily driven by the reduction in the length of the thinking process as the input length increases. Building on this insight, we decompose the thinking process into retrieval and reasoning stages and introduce a reflection mechanism for multi-round extension. We also train a model using the generated iterative thinking process, which helps mitigate the performance degradation. Furthermore, we demonstrate the application of this retrieval-reflection capability in mathematical reasoning scenarios, improving GPT-4o's performance on AIME2024. 

---
# My Life in Artificial Intelligence: People, anecdotes, and some lessons learnt 

**Authors**: Kees van Deemter  

**Link**: [PDF](https://arxiv.org/pdf/2504.04142)  

**Abstract**: In this very personal workography, I relate my 40-year experiences as a researcher and educator in and around Artificial Intelligence (AI), more specifically Natural Language Processing. I describe how curiosity, and the circumstances of the day, led me to work in both industry and academia, and in various countries, including The Netherlands (Amsterdam, Eindhoven, and Utrecht), the USA (Stanford), England (Brighton), Scotland (Aberdeen), and China (Beijing and Harbin). People and anecdotes play a large role in my story; the history of AI forms its backdrop. I focus on things that might be of interest to (even) younger colleagues, given the choices they face in their own work and life at a time when AI is finally emerging from the shadows. 

---
# Cognitive Debiasing Large Language Models for Decision-Making 

**Authors**: Yougang Lyu, Shijie Ren, Yue Feng, Zihan Wang, Zhumin Chen, Zhaochun Ren, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2504.04141)  

**Abstract**: Large language models (LLMs) have shown potential in supporting decision-making applications, particularly as personal conversational assistants in the financial, healthcare, and legal domains. While prompt engineering strategies have enhanced the capabilities of LLMs in decision-making, cognitive biases inherent to LLMs present significant challenges. Cognitive biases are systematic patterns of deviation from norms or rationality in decision-making that can lead to the production of inaccurate outputs. Existing cognitive bias mitigation strategies assume that input prompts contain (exactly) one type of cognitive bias and therefore fail to perform well in realistic settings where there maybe any number of biases.
To fill this gap, we propose a cognitive debiasing approach, called self-debiasing, that enhances the reliability of LLMs by iteratively refining prompts. Our method follows three sequential steps -- bias determination, bias analysis, and cognitive debiasing -- to iteratively mitigate potential cognitive biases in prompts. Experimental results on finance, healthcare, and legal decision-making tasks, using both closed-source and open-source LLMs, demonstrate that the proposed self-debiasing method outperforms both advanced prompt engineering methods and existing cognitive debiasing techniques in average accuracy under no-bias, single-bias, and multi-bias settings. 

---
# Precise Legal Sentence Boundary Detection for Retrieval at Scale: NUPunkt and CharBoundary 

**Authors**: Michael J Bommarito, Daniel Martin Katz, Jillian Bommarito  

**Link**: [PDF](https://arxiv.org/pdf/2504.04131)  

**Abstract**: We present NUPunkt and CharBoundary, two sentence boundary detection libraries optimized for high-precision, high-throughput processing of legal text in large-scale applications such as due diligence, e-discovery, and legal research. These libraries address the critical challenges posed by legal documents containing specialized citations, abbreviations, and complex sentence structures that confound general-purpose sentence boundary detectors.
Our experimental evaluation on five diverse legal datasets comprising over 25,000 documents and 197,000 annotated sentence boundaries demonstrates that NUPunkt achieves 91.1% precision while processing 10 million characters per second with modest memory requirements (432 MB). CharBoundary models offer balanced and adjustable precision-recall tradeoffs, with the large model achieving the highest F1 score (0.782) among all tested methods.
Notably, NUPunkt provides a 29-32% precision improvement over general-purpose tools while maintaining exceptional throughput, processing multi-million document collections in minutes rather than hours. Both libraries run efficiently on standard CPU hardware without requiring specialized accelerators. NUPunkt is implemented in pure Python with zero external dependencies, while CharBoundary relies only on scikit-learn and optional ONNX runtime integration for optimized performance. Both libraries are available under the MIT license, can be installed via PyPI, and can be interactively tested at this https URL.
These libraries address critical precision issues in retrieval-augmented generation systems by preserving coherent legal concepts across sentences, where each percentage improvement in precision yields exponentially greater reductions in context fragmentation, creating cascading benefits throughout retrieval pipelines and significantly enhancing downstream reasoning quality. 

---
# A Benchmark for End-to-End Zero-Shot Biomedical Relation Extraction with LLMs: Experiments with OpenAI Models 

**Authors**: Aviv Brokman, Xuguang Ai, Yuhang Jiang, Shashank Gupta, Ramakanth Kavuluru  

**Link**: [PDF](https://arxiv.org/pdf/2504.04083)  

**Abstract**: Objective: Zero-shot methodology promises to cut down on costs of dataset annotation and domain expertise needed to make use of NLP. Generative large language models trained to align with human goals have achieved high zero-shot performance across a wide variety of tasks. As of yet, it is unclear how well these models perform on biomedical relation extraction (RE). To address this knowledge gap, we explore patterns in the performance of OpenAI LLMs across a diverse sampling of RE tasks.
Methods: We use OpenAI GPT-4-turbo and their reasoning model o1 to conduct end-to-end RE experiments on seven datasets. We use the JSON generation capabilities of GPT models to generate structured output in two ways: (1) by defining an explicit schema describing the structure of relations, and (2) using a setting that infers the structure from the prompt language.
Results: Our work is the first to study and compare the performance of the GPT-4 and o1 for the end-to-end zero-shot biomedical RE task across a broad array of datasets. We found the zero-shot performances to be proximal to that of fine-tuned methods. The limitations of this approach are that it performs poorly on instances containing many relations and errs on the boundaries of textual mentions.
Conclusion: Recent large language models exhibit promising zero-shot capabilities in complex biomedical RE tasks, offering competitive performance with reduced dataset curation and NLP modeling needs at the cost of increased computing, potentially increasing medical community accessibility. Addressing the limitations we identify could further boost reliability. The code, data, and prompts for all our experiments are publicly available: this https URL 

---
# Collaboration and Controversy Among Experts: Rumor Early Detection by Tuning a Comment Generator 

**Authors**: Bing Wang, Bingrui Zhao, Ximing Li, Changchun Li, Wanfu Gao, Shengsheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04076)  

**Abstract**: Over the past decade, social media platforms have been key in spreading rumors, leading to significant negative impacts. To counter this, the community has developed various Rumor Detection (RD) algorithms to automatically identify them using user comments as evidence. However, these RD methods often fail in the early stages of rumor propagation when only limited user comments are available, leading the community to focus on a more challenging topic named Rumor Early Detection (RED). Typically, existing RED methods learn from limited semantics in early comments. However, our preliminary experiment reveals that the RED models always perform best when the number of training and test comments is consistent and extensive. This inspires us to address the RED issue by generating more human-like comments to support this hypothesis. To implement this idea, we tune a comment generator by simulating expert collaboration and controversy and propose a new RED framework named CAMERED. Specifically, we integrate a mixture-of-expert structure into a generative language model and present a novel routing network for expert collaboration. Additionally, we synthesize a knowledgeable dataset and design an adversarial learning strategy to align the style of generated comments with real-world comments. We further integrate generated and original comments with a mutual controversy fusion module. Experimental results show that CAMERED outperforms state-of-the-art RED baseline models and generation methods, demonstrating its effectiveness. 

---
# VocalNet: Speech LLM with Multi-Token Prediction for Faster and High-Quality Generation 

**Authors**: Yuhao Wang, Heyang Liu, Ziyang Cheng, Ronghua Wu, Qunshan Gu, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04060)  

**Abstract**: Speech large language models (LLMs) have emerged as a prominent research focus in speech processing. We propose VocalNet-1B and VocalNet-8B, a series of high-performance, low-latency speech LLMs enabled by a scalable and model-agnostic training framework for real-time voice interaction. Departing from the conventional next-token prediction (NTP), we introduce multi-token prediction (MTP), a novel approach optimized for speech LLMs that simultaneously improves generation speed and quality. Experiments show that VocalNet outperforms mainstream Omni LLMs despite using significantly less training data, while also surpassing existing open-source speech LLMs by a substantial margin. To support reproducibility and community advancement, we will open-source all model weights, inference code, training data, and framework implementations upon publication. 

---
# FISH-Tuning: Enhancing PEFT Methods with Fisher Information 

**Authors**: Kang Xue, Ming Dong, Xinhui Tu, Tingting He  

**Link**: [PDF](https://arxiv.org/pdf/2504.04050)  

**Abstract**: The rapid growth in the parameter size of Large Language Models (LLMs) has led to the development of Parameter-Efficient Fine-Tuning (PEFT) methods to alleviate the computational costs of fine-tuning. Among these, Fisher Induced Sparse uncHanging (FISH) Mask is a selection-based PEFT technique that identifies a subset of pre-trained parameters for fine-tuning based on approximate Fisher information. However, the integration of FISH Mask with other PEFT methods, such as LoRA and Adapters, remains underexplored. In this paper, we propose FISH-Tuning, a novel approach that incorporates FISH Mask into addition-based and reparameterization-based PEFT methods, including LoRA, Adapters, and their variants. By leveraging Fisher information to select critical parameters within these methods, FISH-Tuning achieves superior performance without additional memory overhead or inference latency. Experimental results across various datasets and pre-trained models demonstrate that FISH-Tuning consistently outperforms the vanilla PEFT methods with the same proportion of trainable parameters. 

---
# SyLeR: A Framework for Explicit Syllogistic Legal Reasoning in Large Language Models 

**Authors**: Kepu Zhang, Weijie Yu, Zhongxiang Sun, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04042)  

**Abstract**: Syllogistic reasoning is a fundamental aspect of legal decision-making, enabling logical conclusions by connecting general legal principles with specific case facts. Although existing large language models (LLMs) can generate responses to legal questions, they fail to perform explicit syllogistic reasoning, often producing implicit and unstructured answers that lack explainability and trustworthiness. To address this limitation, we propose SyLeR, a novel framework that empowers LLMs to engage in explicit syllogistic legal reasoning. SyLeR integrates a tree-structured hierarchical retrieval mechanism to effectively combine relevant legal statutes and precedent cases, forming comprehensive major premises. This is followed by a two-stage fine-tuning process: supervised fine-tuning warm-up establishes a foundational understanding of syllogistic reasoning, while reinforcement learning with a structure-aware reward mechanism refines the ability of the model to generate diverse logically sound and well-structured reasoning paths. We conducted extensive experiments across various dimensions, including in-domain and cross-domain user groups (legal laypersons and practitioners), multiple languages (Chinese and French), and different LLM backbones (legal-specific and open-domain LLMs). The results show that SyLeR significantly improves response accuracy and consistently delivers explicit, explainable, and trustworthy legal reasoning. 

---
# myNER: Contextualized Burmese Named Entity Recognition with Bidirectional LSTM and fastText Embeddings via Joint Training with POS Tagging 

**Authors**: Kaung Lwin Thant, Kwankamol Nongpong, Ye Kyaw Thu, Thura Aung, Khaing Hsu Wai, Thazin Myint Oo  

**Link**: [PDF](https://arxiv.org/pdf/2504.04038)  

**Abstract**: Named Entity Recognition (NER) involves identifying and categorizing named entities within textual data. Despite its significance, NER research has often overlooked low-resource languages like Myanmar (Burmese), primarily due to the lack of publicly available annotated datasets. To address this, we introduce myNER, a novel word-level NER corpus featuring a 7-tag annotation scheme, enriched with Part-of-Speech (POS) tagging to provide additional syntactic information. Alongside the corpus, we conduct a comprehensive evaluation of NER models, including Conditional Random Fields (CRF), Bidirectional LSTM (BiLSTM)-CRF, and their combinations with fastText embeddings in different settings. Our experiments reveal the effectiveness of contextualized word embeddings and the impact of joint training with POS tagging, demonstrating significant performance improvements across models. The traditional CRF joint-task model with fastText embeddings as a feature achieved the best result, with a 0.9818 accuracy and 0.9811 weighted F1 score with 0.7429 macro F1 score. BiLSTM-CRF with fine-tuned fastText embeddings gets the best result of 0.9791 accuracy and 0.9776 weighted F1 score with 0.7395 macro F1 score. 

---
# Rethinking Reflection in Pre-Training 

**Authors**: Essential AI, Darsh J Shah, Peter Rushton, Somanshu Singla, Mohit Parmar, Kurt Smith, Yash Vanjani, Ashish Vaswani, Adarsh Chaluvaraju, Andrew Hojel, Andrew Ma, Anil Thomas, Anthony Polloreno, Ashish Tanwer, Burhan Drak Sibai, Divya S Mansingka, Divya Shivaprasad, Ishaan Shah, Karl Stratos, Khoi Nguyen, Michael Callahan, Michael Pust, Mrinal Iyer, Philip Monk, Platon Mazarakis, Ritvik Kapila, Saurabh Srivastava, Tim Romanski  

**Link**: [PDF](https://arxiv.org/pdf/2504.04022)  

**Abstract**: A language model's ability to reflect on its own reasoning provides a key advantage for solving complex problems. While most recent research has focused on how this ability develops during reinforcement learning, we show that it actually begins to emerge much earlier - during the model's pre-training. To study this, we introduce deliberate errors into chains-of-thought and test whether the model can still arrive at the correct answer by recognizing and correcting these mistakes. By tracking performance across different stages of pre-training, we observe that this self-correcting ability appears early and improves steadily over time. For instance, an OLMo2-7B model pre-trained on 4 trillion tokens displays self-correction on our six self-reflection tasks. 

---
# Algorithmic Prompt Generation for Diverse Human-like Teaming and Communication with Large Language Models 

**Authors**: Siddharth Srikanth, Varun Bhatt, Boshen Zhang, Werner Hager, Charles Michael Lewis, Katia P. Sycara, Aaquib Tabrez, Stefanos Nikolaidis  

**Link**: [PDF](https://arxiv.org/pdf/2504.03991)  

**Abstract**: Understanding how humans collaborate and communicate in teams is essential for improving human-agent teaming and AI-assisted decision-making. However, relying solely on data from large-scale user studies is impractical due to logistical, ethical, and practical constraints, necessitating synthetic models of multiple diverse human behaviors. Recently, agents powered by Large Language Models (LLMs) have been shown to emulate human-like behavior in social settings. But, obtaining a large set of diverse behaviors requires manual effort in the form of designing prompts. On the other hand, Quality Diversity (QD) optimization has been shown to be capable of generating diverse Reinforcement Learning (RL) agent behavior. In this work, we combine QD optimization with LLM-powered agents to iteratively search for prompts that generate diverse team behavior in a long-horizon, multi-step collaborative environment. We first show, through a human-subjects experiment (n=54 participants), that humans exhibit diverse coordination and communication behavior in this domain. We then show that our approach can effectively replicate trends from human teaming data and also capture behaviors that are not easily observed without collecting large amounts of data. Our findings highlight the combination of QD and LLM-powered agents as an effective tool for studying teaming and communication strategies in multi-agent collaboration. 

---
# Structured Extraction of Process Structure Properties Relationships in Materials Science 

**Authors**: Amit K Verma, Zhisong Zhang, Junwon Seo, Robin Kuo, Runbo Jiang, Emma Strubell, Anthony D Rollett  

**Link**: [PDF](https://arxiv.org/pdf/2504.03979)  

**Abstract**: With the advent of large language models (LLMs), the vast unstructured text within millions of academic papers is increasingly accessible for materials discovery, although significant challenges remain. While LLMs offer promising few- and zero-shot learning capabilities, particularly valuable in the materials domain where expert annotations are scarce, general-purpose LLMs often fail to address key materials-specific queries without further adaptation. To bridge this gap, fine-tuning LLMs on human-labeled data is essential for effective structured knowledge extraction. In this study, we introduce a novel annotation schema designed to extract generic process-structure-properties relationships from scientific literature. We demonstrate the utility of this approach using a dataset of 128 abstracts, with annotations drawn from two distinct domains: high-temperature materials (Domain I) and uncertainty quantification in simulating materials microstructure (Domain II). Initially, we developed a conditional random field (CRF) model based on MatBERT, a domain-specific BERT variant, and evaluated its performance on Domain I. Subsequently, we compared this model with a fine-tuned LLM (GPT-4o from OpenAI) under identical conditions. Our results indicate that fine-tuning LLMs can significantly improve entity extraction performance over the BERT-CRF baseline on Domain I. However, when additional examples from Domain II were incorporated, the performance of the BERT-CRF model became comparable to that of the GPT-4o model. These findings underscore the potential of our schema for structured knowledge extraction and highlight the complementary strengths of both modeling approaches. 

---
# Clinical ModernBERT: An efficient and long context encoder for biomedical text 

**Authors**: Simon A. Lee, Anthony Wu, Jeffrey N. Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03964)  

**Abstract**: We introduce Clinical ModernBERT, a transformer based encoder pretrained on large scale biomedical literature, clinical notes, and medical ontologies, incorporating PubMed abstracts, MIMIC IV clinical data, and medical codes with their textual descriptions. Building on ModernBERT the current state of the art natural language text encoder featuring architectural upgrades such as rotary positional embeddings (RoPE), Flash Attention, and extended context length up to 8,192 tokens our model adapts these innovations specifically for biomedical and clinical domains. Clinical ModernBERT excels at producing semantically rich representations tailored for long context tasks. We validate this both by analyzing its pretrained weights and through empirical evaluation on a comprehensive suite of clinical NLP benchmarks. 

---
# Language Models Are Implicitly Continuous 

**Authors**: Samuele Marro, Davide Evangelista, X. Angelo Huang, Emanuele La Malfa, Michele Lombardi, Michael Wooldridge  

**Link**: [PDF](https://arxiv.org/pdf/2504.03933)  

**Abstract**: Language is typically modelled with discrete sequences. However, the most successful approaches to language modelling, namely neural networks, are continuous and smooth function approximators. In this work, we show that Transformer-based language models implicitly learn to represent sentences as continuous-time functions defined over a continuous input space. This phenomenon occurs in most state-of-the-art Large Language Models (LLMs), including Llama2, Llama3, Phi3, Gemma, Gemma2, and Mistral, and suggests that LLMs reason about language in ways that fundamentally differ from humans. Our work formally extends Transformers to capture the nuances of time and space continuity in both input and output space. Our results challenge the traditional interpretation of how LLMs understand language, with several linguistic and engineering implications. 

---
# YaleNLP @ PerAnsSumm 2025: Multi-Perspective Integration via Mixture-of-Agents for Enhanced Healthcare QA Summarization 

**Authors**: Dongsuk Jang, Alan Li, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2504.03932)  

**Abstract**: Automated summarization of healthcare community question-answering forums is challenging due to diverse perspectives presented across multiple user responses to each question. The PerAnsSumm Shared Task was therefore proposed to tackle this challenge by identifying perspectives from different answers and then generating a comprehensive answer to the question. In this study, we address the PerAnsSumm Shared Task using two complementary paradigms: (i) a training-based approach through QLoRA fine-tuning of LLaMA-3.3-70B-Instruct, and (ii) agentic approaches including zero- and few-shot prompting with frontier LLMs (LLaMA-3.3-70B-Instruct and GPT-4o) and a Mixture-of-Agents (MoA) framework that leverages a diverse set of LLMs by combining outputs from multi-layer feedback aggregation. For perspective span identification/classification, GPT-4o zero-shot achieves an overall score of 0.57, substantially outperforming the 0.40 score of the LLaMA baseline. With a 2-layer MoA configuration, we were able to improve LLaMA performance up by 28 percent to 0.51. For perspective-based summarization, GPT-4o zero-shot attains an overall score of 0.42 compared to 0.28 for the best LLaMA zero-shot, and our 2-layer MoA approach boosts LLaMA performance by 32 percent to 0.37. Furthermore, in few-shot setting, our results show that the sentence-transformer embedding-based exemplar selection provides more gain than manually selected exemplars on LLaMA models, although the few-shot prompting is not always helpful for GPT-4o. The YaleNLP team's approach ranked the overall second place in the shared task. 

---
# Adaptation of Large Language Models 

**Authors**: Zixuan Ke, Yifei Ming, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2504.03931)  

**Abstract**: This tutorial on adaptation of LLMs is designed to address the growing demand for models that go beyond the static capabilities of generic LLMs by providing an overview of dynamic, domain-specific, and task-adaptive LLM adaptation techniques. While general LLMs have demonstrated strong generalization across a variety of tasks, they often struggle to perform well in specialized domains such as finance, healthcare, and code generation for underrepresented languages. Additionally, their static nature limits their ability to evolve with the changing world, and they are often extremely large in size, making them impractical and costly to deploy at scale. As a result, the adaptation of LLMs has drawn much attention since the birth of LLMs and is of core importance, both for industry, which focuses on serving its targeted users, and academia, which can greatly benefit from small but powerful LLMs. To address this gap, this tutorial aims to provide an overview of the LLM adaptation techniques. We start with an introduction to LLM adaptation, from both the data perspective and the model perspective. We then emphasize how the evaluation metrics and benchmarks are different from other techniques. After establishing the problems, we explore various adaptation techniques. We categorize adaptation techniques into two main families. The first is parametric knowledge adaptation, which focuses on updating the parametric knowledge within LLMs. Additionally, we will discuss real-time adaptation techniques, including model editing, which allows LLMs to be updated dynamically in production environments. The second kind of adaptation is semi-parametric knowledge adaptation, where the goal is to update LLM parameters to better leverage external knowledge or tools through techniques like retrieval-augmented generation (RAG) and agent-based systems. 

---
# CliME: Evaluating Multimodal Climate Discourse on Social Media and the Climate Alignment Quotient (CAQ) 

**Authors**: Abhilekh Borah, Hasnat Md Abdullah, Kangda Wei, Ruihong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03906)  

**Abstract**: The rise of Large Language Models (LLMs) has raised questions about their ability to understand climate-related contexts. Though climate change dominates social media, analyzing its multimodal expressions is understudied, and current tools have failed to determine whether LLMs amplify credible solutions or spread unsubstantiated claims. To address this, we introduce CliME (Climate Change Multimodal Evaluation), a first-of-its-kind multimodal dataset, comprising 2579 Twitter and Reddit posts. The benchmark features a diverse collection of humorous memes and skeptical posts, capturing how these formats distill complex issues into viral narratives that shape public opinion and policy discussions. To systematically evaluate LLM performance, we present the Climate Alignment Quotient (CAQ), a novel metric comprising five distinct dimensions: Articulation, Evidence, Resonance, Transition, and Specificity. Additionally, we propose three analytical lenses: Actionability, Criticality, and Justice, to guide the assessment of LLM-generated climate discourse using CAQ. Our findings, based on the CAQ metric, indicate that while most evaluated LLMs perform relatively well in Criticality and Justice, they consistently underperform on the Actionability axis. Among the models evaluated, Claude 3.7 Sonnet achieves the highest overall performance. We publicly release our CliME dataset and code to foster further research in this domain. 

---
# Do LLM Evaluators Prefer Themselves for a Reason? 

**Authors**: Wei-Lin Chen, Zhepei Wei, Xinyu Zhu, Shi Feng, Yu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2504.03846)  

**Abstract**: Large language models (LLMs) are increasingly used as automatic evaluators in applications such as benchmarking, reward modeling, and self-refinement. Prior work highlights a potential self-preference bias where LLMs favor their own generated responses, a tendency often intensifying with model size and capability. This raises a critical question: Is self-preference detrimental, or does it simply reflect objectively superior outputs from more capable models? Disentangling these has been challenging due to the usage of subjective tasks in previous studies. To address this, we investigate self-preference using verifiable benchmarks (mathematical reasoning, factual knowledge, code generation) that allow objective ground-truth assessment. This enables us to distinguish harmful self-preference (favoring objectively worse responses) from legitimate self-preference (favoring genuinely superior ones). We conduct large-scale experiments under controlled evaluation conditions across diverse model families (e.g., Llama, Qwen, Gemma, Mistral, Phi, GPT, DeepSeek). Our findings reveal three key insights: (1) Better generators are better judges -- LLM evaluators' accuracy strongly correlates with their task performance, and much of the self-preference in capable models is legitimate. (2) Harmful self-preference persists, particularly when evaluator models perform poorly as generators on specific task instances. Stronger models exhibit more pronounced harmful bias when they err, though such incorrect generations are less frequent. (3) Inference-time scaling strategies, such as generating a long Chain-of-Thought before evaluation, effectively reduce the harmful self-preference. These results provide a more nuanced understanding of LLM-based evaluation and practical insights for improving its reliability. 

---
# What Large Language Models Do Not Talk About: An Empirical Study of Moderation and Censorship Practices 

**Authors**: Sander Noels, Guillaume Bied, Maarten Buyl, Alexander Rogiers, Yousra Fettach, Jefrey Lijffijt, Tijl De Bie  

**Link**: [PDF](https://arxiv.org/pdf/2504.03803)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed as gateways to information, yet their content moderation practices remain underexplored. This work investigates the extent to which LLMs refuse to answer or omit information when prompted on political topics. To do so, we distinguish between hard censorship (i.e., generated refusals, error messages, or canned denial responses) and soft censorship (i.e., selective omission or downplaying of key elements), which we identify in LLMs' responses when asked to provide information on a broad range of political figures. Our analysis covers 14 state-of-the-art models from Western countries, China, and Russia, prompted in all six official United Nations (UN) languages. Our analysis suggests that although censorship is observed across the board, it is predominantly tailored to an LLM provider's domestic audience and typically manifests as either hard censorship or soft censorship (though rarely both concurrently). These findings underscore the need for ideological and geographic diversity among publicly available LLMs, and greater transparency in LLM moderation strategies to facilitate informed user choices. All data are made freely available. 

---
# Entropy-Based Block Pruning for Efficient Large Language Models 

**Authors**: Liangwei Yang, Yuhui Xu, Juntao Tan, Doyen Sahoo, Silvio Savarese, Caiming Xiong, Huan Wang, Shelby Heinecke  

**Link**: [PDF](https://arxiv.org/pdf/2504.03794)  

**Abstract**: As large language models continue to scale, their growing computational and storage demands pose significant challenges for real-world deployment. In this work, we investigate redundancy within Transformer-based models and propose an entropy-based pruning strategy to enhance efficiency while maintaining performance. Empirical analysis reveals that the entropy of hidden representations decreases in the early blocks but progressively increases across most subsequent blocks. This trend suggests that entropy serves as a more effective measure of information richness within computation blocks. Unlike cosine similarity, which primarily captures geometric relationships, entropy directly quantifies uncertainty and information content, making it a more reliable criterion for pruning. Extensive experiments demonstrate that our entropy-based pruning approach surpasses cosine similarity-based methods in reducing model size while preserving accuracy, offering a promising direction for efficient model deployment. 

---
# Sample, Don't Search: Rethinking Test-Time Alignment for Language Models 

**Authors**: Gonçalo Faria, Noah A. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2504.03790)  

**Abstract**: Increasing test-time computation has emerged as a promising direction for improving language model performance, particularly in scenarios where model finetuning is impractical or impossible due to computational constraints or private model weights. However, existing test-time search methods using a reward model (RM) often degrade in quality as compute scales, due to the over-optimization of what are inherently imperfect reward proxies. We introduce QAlign, a new test-time alignment approach. As we scale test-time compute, QAlign converges to sampling from the optimal aligned distribution for each individual prompt. By adopting recent advances in Markov chain Monte Carlo for text generation, our method enables better-aligned outputs without modifying the underlying model or even requiring logit access. We demonstrate the effectiveness of QAlign on mathematical reasoning benchmarks (GSM8K and GSM-Symbolic) using a task-specific RM, showing consistent improvements over existing test-time compute methods like best-of-n and majority voting. Furthermore, when applied with more realistic RMs trained on the Tulu 3 preference dataset, QAlign outperforms direct preference optimization (DPO), best-of-n, majority voting, and weighted majority voting on a diverse range of datasets (GSM8K, MATH500, IFEval, MMLU-Redux, and TruthfulQA). A practical solution to aligning language models at test time using additional computation without degradation, our approach expands the limits of the capability that can be obtained from off-the-shelf language models without further training. 

---
# Do "New Snow Tablets" Contain Snow? Large Language Models Over-Rely on Names to Identify Ingredients of Chinese Drugs 

**Authors**: Sifan Li, Yujun Cai, Bryan Hooi, Nanyun Peng, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03786)  

**Abstract**: Traditional Chinese Medicine (TCM) has seen increasing adoption in healthcare, with specialized Large Language Models (LLMs) emerging to support clinical applications. A fundamental requirement for these models is accurate identification of TCM drug ingredients. In this paper, we evaluate how general and TCM-specialized LLMs perform when identifying ingredients of Chinese drugs. Our systematic analysis reveals consistent failure patterns: models often interpret drug names literally, overuse common herbs regardless of relevance, and exhibit erratic behaviors when faced with unfamiliar formulations. LLMs also fail to understand the verification task. These findings demonstrate that current LLMs rely primarily on drug names rather than possessing systematic pharmacological knowledge. To address these limitations, we propose a Retrieval Augmented Generation (RAG) approach focused on ingredient names. Experiments across 220 TCM formulations show our method significantly improves accuracy from approximately 50% to 82% in ingredient verification tasks. Our work highlights critical weaknesses in current TCM-specific LLMs and offers a practical solution for enhancing their clinical reliability. 

---
# A Unified Virtual Mixture-of-Experts Framework:Enhanced Inference and Hallucination Mitigation in Single-Model System 

**Authors**: Mingyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03739)  

**Abstract**: Generative models, such as GPT and BERT, have significantly improved performance in tasks like text generation and summarization. However, hallucinations "where models generate non-factual or misleading content" are especially problematic in smaller-scale architectures, limiting their real-world this http URL this paper, we propose a unified Virtual Mixture-of-Experts (MoE) fusion strategy that enhances inference performance and mitigates hallucinations in a single Qwen 1.5 0.5B model without increasing the parameter count. Our method leverages multiple domain-specific expert prompts (with the number of experts being adjustable) to guide the model from different perspectives. We apply a statistical outlier truncation strategy based on the mean and standard deviation to filter out abnormally high probability predictions, and we inject noise into the embedding space to promote output diversity. To clearly assess the contribution of each module, we adopt a fixed voting mechanism rather than a dynamic gating network, thereby avoiding additional confounding factors. We provide detailed theoretical derivations from both statistical and ensemble learning perspectives to demonstrate how our method reduces output variance and suppresses hallucinations. Extensive ablation experiments on dialogue generation tasks show that our approach significantly improves inference accuracy and robustness in small models. Additionally, we discuss methods for evaluating the orthogonality of virtual experts and outline the potential for future work involving dynamic expert weight allocation using gating networks. 

---
# LiveVQA: Live Visual Knowledge Seeking 

**Authors**: Mingyang Fu, Yuyang Peng, Benlin Liu, Yao Wan, Dongping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.05288)  

**Abstract**: We introduce LiveVQA, an automatically collected dataset of latest visual knowledge from the Internet with synthesized VQA problems. LiveVQA consists of 3,602 single- and multi-hop visual questions from 6 news websites across 14 news categories, featuring high-quality image-text coherence and authentic information. Our evaluation across 15 MLLMs (e.g., GPT-4o, Gemma-3, and Qwen-2.5-VL family) demonstrates that stronger models perform better overall, with advanced visual reasoning capabilities proving crucial for complex multi-hop questions. Despite excellent performance on textual problems, models with tools like search engines still show significant gaps when addressing visual questions requiring latest visual knowledge, highlighting important areas for future research. 

---
# Learning to Reason Over Time: Timeline Self-Reflection for Improved Temporal Reasoning in Language Models 

**Authors**: Adrián Bazaga, Rexhina Blloshmi, Bill Byrne, Adrià de Gispert  

**Link**: [PDF](https://arxiv.org/pdf/2504.05258)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for generating coherent text, understanding context, and performing reasoning tasks. However, they struggle with temporal reasoning, which requires processing time-related information such as event sequencing, durations, and inter-temporal relationships. These capabilities are critical for applications including question answering, scheduling, and historical analysis. In this paper, we introduce TISER, a novel framework that enhances the temporal reasoning abilities of LLMs through a multi-stage process that combines timeline construction with iterative self-reflection. Our approach leverages test-time scaling to extend the length of reasoning traces, enabling models to capture complex temporal dependencies more effectively. This strategy not only boosts reasoning accuracy but also improves the traceability of the inference process. Experimental results demonstrate state-of-the-art performance across multiple benchmarks, including out-of-distribution test sets, and reveal that TISER enables smaller open-source models to surpass larger closed-weight models on challenging temporal reasoning tasks. 

---
# Leveraging LLMs for Utility-Focused Annotation: Reducing Manual Effort for Retrieval and RAG 

**Authors**: Hengran Zhang, Minghao Tang, Keping Bi, Jiafeng Guo, Shihao Liu, Daiting Shi, Dawei Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05220)  

**Abstract**: Retrieval models typically rely on costly human-labeled query-document relevance annotations for training and evaluation. To reduce this cost and leverage the potential of Large Language Models (LLMs) in relevance judgments, we aim to explore whether LLM-generated annotations can effectively replace human annotations in training retrieval models. Retrieval usually emphasizes relevance, which indicates "topic-relatedness" of a document to a query, while in RAG, the value of a document (or utility) depends on how it contributes to answer generation. Recognizing this mismatch, some researchers use LLM performance on downstream tasks with documents as labels, but this approach requires manual answers for specific tasks, leading to high costs and limited generalization. In another line of work, prompting LLMs to select useful documents as RAG references eliminates the need for human annotation and is not task-specific. If we leverage LLMs' utility judgments to annotate retrieval data, we may retain cross-task generalization without human annotation in large-scale corpora. Therefore, we investigate utility-focused annotation via LLMs for large-scale retriever training data across both in-domain and out-of-domain settings on the retrieval and RAG tasks. To reduce the impact of low-quality positives labeled by LLMs, we design a novel loss function, i.e., Disj-InfoNCE. Our experiments reveal that: (1) Retrievers trained on utility-focused annotations significantly outperform those trained on human annotations in the out-of-domain setting on both tasks, demonstrating superior generalization capabilities. (2) LLM annotation does not replace human annotation in the in-domain setting. However, incorporating just 20% human-annotated data enables retrievers trained with utility-focused annotations to match the performance of models trained entirely with human annotations. 

---
# Unleashing the Power of LLMs in Dense Retrieval with Query Likelihood Modeling 

**Authors**: Hengran Zhang, Keping Bi, Jiafeng Guo, Xiaojie Sun, Shihao Liu, Daiting Shi, Dawei Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05216)  

**Abstract**: Dense retrieval is a crucial task in Information Retrieval (IR) and is the foundation for downstream tasks such as re-ranking. Recently, large language models (LLMs) have shown compelling semantic understanding capabilities and are appealing to researchers studying dense retrieval. LLMs, as decoder-style generative models, are competent at language generation while falling short on modeling global information due to the lack of attention to tokens afterward. Inspired by the classical word-based language modeling approach for IR, i.e., the query likelihood (QL) model, we seek to sufficiently utilize LLMs' generative ability by QL maximization. However, instead of ranking documents with QL estimation, we introduce an auxiliary task of QL maximization to yield a better backbone for contrastively learning a discriminative retriever. We name our model as LLM-QL. To condense global document semantics to a single vector during QL modeling, LLM-QL has two major components, Attention Stop (AS) and Input Corruption (IC). AS stops the attention of predictive tokens to previous tokens until the ending token of the document. IC masks a portion of tokens in the input documents during prediction. Experiments on MSMARCO show that LLM-QL can achieve significantly better performance than other LLM-based retrievers and using QL estimated by LLM-QL for ranking outperforms word-based QL by a large margin. 

---
# Mixture-of-Personas Language Models for Population Simulation 

**Authors**: Ngoc Bui, Hieu Trung Nguyen, Shantanu Kumar, Julian Theodore, Weikang Qiu, Viet Anh Nguyen, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2504.05019)  

**Abstract**: Advances in Large Language Models (LLMs) paved the way for their emerging applications in various domains, such as human behavior simulations, where LLMs could augment human-generated data in social science research and machine learning model training. However, pretrained LLMs often fail to capture the behavioral diversity of target populations due to the inherent variability across individuals and groups. To address this, we propose \textit{Mixture of Personas} (MoP), a \textit{probabilistic} prompting method that aligns the LLM responses with the target population. MoP is a contextual mixture model, where each component is an LM agent characterized by a persona and an exemplar representing subpopulation behaviors. The persona and exemplar are randomly chosen according to the learned mixing weights to elicit diverse LLM responses during simulation. MoP is flexible, requires no model finetuning, and is transferable across base models. Experiments for synthetic data generation show that MoP outperforms competing methods in alignment and diversity metrics. 

---
# Towards Visual Text Grounding of Multimodal Large Language Model 

**Authors**: Ming Li, Ruiyi Zhang, Jian Chen, Jiuxiang Gu, Yufan Zhou, Franck Dernoncourt, Wanrong Zhu, Tianyi Zhou, Tong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.04974)  

**Abstract**: Despite the existing evolution of Multimodal Large Language Models (MLLMs), a non-neglectable limitation remains in their struggle with visual text grounding, especially in text-rich images of documents. Document images, such as scanned forms and infographics, highlight critical challenges due to their complex layouts and textual content. However, current benchmarks do not fully address these challenges, as they mostly focus on visual grounding on natural images, rather than text-rich document images. Thus, to bridge this gap, we introduce TRIG, a novel task with a newly designed instruction dataset for benchmarking and improving the Text-Rich Image Grounding capabilities of MLLMs in document question-answering. Specifically, we propose an OCR-LLM-human interaction pipeline to create 800 manually annotated question-answer pairs as a benchmark and a large-scale training set of 90$ synthetic data based on four diverse datasets. A comprehensive evaluation of various MLLMs on our proposed benchmark exposes substantial limitations in their grounding capability on text-rich images. In addition, we propose two simple and effective TRIG methods based on general instruction tuning and plug-and-play efficient embedding, respectively. By finetuning MLLMs on our synthetic dataset, they promisingly improve spatial reasoning and grounding capabilities. 

---
# A Llama walks into the 'Bar': Efficient Supervised Fine-Tuning for Legal Reasoning in the Multi-state Bar Exam 

**Authors**: Rean Fernandes, André Biedenkapp, Frank Hutter, Noor Awad  

**Link**: [PDF](https://arxiv.org/pdf/2504.04945)  

**Abstract**: Legal reasoning tasks present unique challenges for large language models (LLMs) due to the complexity of domain-specific knowledge and reasoning processes. This paper investigates how effectively smaller language models (Llama 2 7B and Llama 3 8B) can be fine-tuned with a limited dataset of 1,514 Multi-state Bar Examination (MBE) questions to improve legal question answering accuracy. We evaluate these models on the 2022 MBE questions licensed from JD Advising, the same dataset used in the 'GPT-4 passes the Bar exam' study. Our methodology involves collecting approximately 200 questions per legal domain across 7 domains. We distill the dataset using Llama 3 (70B) to transform explanations into a structured IRAC (Issue, Rule, Application, Conclusion) format as a guided reasoning process to see if it results in better performance over the non-distilled dataset. We compare the non-fine-tuned models against their supervised fine-tuned (SFT) counterparts, trained for different sample sizes per domain, to study the effect on accuracy and prompt adherence. We also analyse option selection biases and their mitigation following SFT. In addition, we consolidate the performance across multiple variables: prompt type (few-shot vs zero-shot), answer ordering (chosen-option first vs generated-explanation first), response format (Numbered list vs Markdown vs JSON), and different decoding temperatures. Our findings show that domain-specific SFT helps some model configurations achieve close to human baseline performance, despite limited computational resources and a relatively small dataset. We release both the gathered SFT dataset and the family of Supervised Fine-tuned (SFT) adapters optimised for MBE performance. This establishes a practical lower bound on resources needed towards achieving effective legal question answering in smaller LLMs. 

---
# How Is Generative AI Used for Persona Development?: A Systematic Review of 52 Research Articles 

**Authors**: Danial Amin, Joni Salminen, Farhan Ahmed, Sonja M.H. Tervola, Sankalp Sethi, Bernard J. Jansen  

**Link**: [PDF](https://arxiv.org/pdf/2504.04927)  

**Abstract**: Although Generative AI (GenAI) has the potential for persona development, many challenges must be addressed. This research systematically reviews 52 articles from 2022-2024, with important findings. First, closed commercial models are frequently used in persona development, creating a monoculture Second, GenAI is used in various stages of persona development (data collection, segmentation, enrichment, and evaluation). Third, similar to other quantitative persona development techniques, there are major gaps in persona evaluation for AI generated personas. Fourth, human-AI collaboration models are underdeveloped, despite human oversight being crucial for maintaining ethical standards. These findings imply that realizing the full potential of AI-generated personas will require substantial efforts across academia and industry. To that end, we provide a list of research avenues to inspire future work. 

---
# Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use 

**Authors**: Anna Goldie, Azalia Mirhoseini, Hao Zhou, Irene Cai, Christopher D. Manning  

**Link**: [PDF](https://arxiv.org/pdf/2504.04736)  

**Abstract**: Reinforcement learning has been shown to improve the performance of large language models. However, traditional approaches like RLHF or RLAIF treat the problem as single-step. As focus shifts toward more complex reasoning and agentic tasks, language models must take multiple steps of text generation, reasoning and environment interaction before generating a solution. We propose a synthetic data generation and RL methodology targeting multi-step optimization scenarios. This approach, called Step-Wise Reinforcement Learning (SWiRL), iteratively generates multi-step reasoning and tool use data, and then learns from that data. It employs a simple step-wise decomposition that breaks each multi-step trajectory into multiple sub-trajectories corresponding to each action by the original model. It then applies synthetic data filtering and RL optimization on these sub-trajectories. We evaluated SWiRL on a number of multi-step tool use, question answering, and mathematical reasoning tasks. Our experiments show that SWiRL outperforms baseline approaches by 21.5%, 12.3%, 14.8%, 11.1%, and 15.3% in relative accuracy on GSM8K, HotPotQA, CofCA, MuSiQue, and BeerQA, respectively. Excitingly, the approach exhibits generalization across tasks: for example, training only on HotPotQA (text question-answering) improves zero-shot performance on GSM8K (a math dataset) by a relative 16.9%. 

---
# LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important 

**Authors**: Manlai Liang, JiaMing Zhang, Xiong Li, Jinlong Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.04704)  

**Abstract**: The increasing size of the Key-Value (KV) cache during the Large Language Models long-context inference is the main obstacle for its balance between the deployment cost and task accuracy. To reduce the KV cache size in such scenarios, most previous efforts leveraged on the attention weight to evict non-critical cache tokens. But there is a trade-off in those methods, they usually require major modifiation of the inference infrastructure and significant computation overhead. Base on the fact that the Large Lanuage models are autoregresssive models, we propose {\it LagKV}, a KV allocation strategy only relying on straight forward comparison among KV themself. It is a totally attention free method which offers easy integration to the main stream inference platform and comparable performance comparing to other complicated KV compression methods. Results on LongBench and PasskeyRetrieval show that, our approach achieves nearly zero loss when the ratio is $2\times$ and $\approx 90\%$ of the original model performance for $8\times$. Especially in the 64-digit passkey retrieval task, our mehod outperforms the attention weight based method $H_2O$ over $60\%$ with same compression ratios. Our code is available at \url{this https URL}. 

---
# R2Vul: Learning to Reason about Software Vulnerabilities with Reinforcement Learning and Structured Reasoning Distillation 

**Authors**: Martin Weyssow, Chengran Yang, Junkai Chen, Yikun Li, Huihui Huang, Ratnadira Widyasari, Han Wei Ang, Frank Liauw, Eng Lieh Ouh, Lwin Khin Shar, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2504.04699)  

**Abstract**: Large language models (LLMs) have shown promising performance in software vulnerability detection (SVD), yet their reasoning capabilities remain unreliable. Existing approaches relying on chain-of-thought (CoT) struggle to provide relevant and actionable security assessments. Additionally, effective SVD requires not only generating coherent reasoning but also differentiating between well-founded and misleading yet plausible security assessments, an aspect overlooked in prior work. To this end, we introduce R2Vul, a novel approach that distills structured reasoning into small LLMs using reinforcement learning from AI feedback (RLAIF). Through RLAIF, R2Vul enables LLMs to produce structured, security-aware reasoning that is actionable and reliable while explicitly learning to distinguish valid assessments from misleading ones. We evaluate R2Vul across five languages against SAST tools, CoT, instruction tuning, and classification-based baselines. Our results show that R2Vul with structured reasoning distillation enables a 1.5B student LLM to rival larger models while improving generalization to out-of-distribution vulnerabilities. Beyond model improvements, we contribute a large-scale, multilingual preference dataset featuring structured reasoning to support future research in SVD. 

---
# LEO-MINI: An Efficient Multimodal Large Language Model using Conditional Token Reduction and Mixture of Multi-Modal Experts 

**Authors**: Yimu Wang, Mozhgan Nasr Azadani, Sean Sedwards, Krzysztof Czarnecki  

**Link**: [PDF](https://arxiv.org/pdf/2504.04653)  

**Abstract**: Redundancy of visual tokens in multi-modal large language models (MLLMs) significantly reduces their computational efficiency. Recent approaches, such as resamplers and summarizers, have sought to reduce the number of visual tokens, but at the cost of visual reasoning ability. To address this, we propose LEO-MINI, a novel MLLM that significantly reduces the number of visual tokens and simultaneously boosts visual reasoning capabilities. For efficiency, LEO-MINI incorporates CoTR, a novel token reduction module to consolidate a large number of visual tokens into a smaller set of tokens, using the similarity between visual tokens, text tokens, and a compact learnable query. For effectiveness, to scale up the model's ability with minimal computational overhead, LEO-MINI employs MMoE, a novel mixture of multi-modal experts module. MMOE employs a set of LoRA experts with a novel router to switch between them based on the input text and visual tokens instead of only using the input hidden state. MMoE also includes a general LoRA expert that is always activated to learn general knowledge for LLM reasoning. For extracting richer visual features, MMOE employs a set of vision experts trained on diverse domain-specific data. To demonstrate LEO-MINI's improved efficiency and performance, we evaluate it against existing efficient MLLMs on various benchmark vision-language tasks. 

---
# Ineffectiveness for Search and Undecidability of PCSP Meta-Problems 

**Authors**: Alberto Larrauri  

**Link**: [PDF](https://arxiv.org/pdf/2504.04639)  

**Abstract**: It is an open question whether the search and decision versions of promise CSPs are equivalent. Most known algorithms for PCSPs solve only their \emph{decision} variant, and it is unknown whether they can be adapted to solve \emph{search} as well. The main approaches, called BLP, AIP and BLP+AIP, handle a PCSP by finding a solution to a relaxation of some integer program. We prove that rounding those solutions to a proper search certificate can be as hard as any problem in the class TFNP. In other words, these algorithms are ineffective for search. Building on the algebraic approach to PCSPs, we find sufficient conditions that imply ineffectiveness for search. Our tools are tailored to algorithms that are characterized by minions in a suitable way, and can also be used to prove undecidability results for meta-problems. This way, we show that the families of templates solvable via BLP, AIP, and BLP+AIP are undecidable.
Using the same techniques we also analyze several algebraic conditions that are known to guarantee the tractability of finite-template CSPs. We prove that several meta-problems related to cyclic polymorphims and WNUs are undecidable for PCSPs. In particular, there is no algorithm deciding whether a finite PCSP template (1) admits cyclic a polymorphism, (2) admits a WNU. 

---
# SECQUE: A Benchmark for Evaluating Real-World Financial Analysis Capabilities 

**Authors**: Noga Ben Yoash, Meni Brief, Oded Ovadia, Gil Shenderovitz, Moshik Mishaeli, Rachel Lemberg, Eitam Sheetrit  

**Link**: [PDF](https://arxiv.org/pdf/2504.04596)  

**Abstract**: We introduce SECQUE, a comprehensive benchmark for evaluating large language models (LLMs) in financial analysis tasks. SECQUE comprises 565 expert-written questions covering SEC filings analysis across four key categories: comparison analysis, ratio calculation, risk assessment, and financial insight generation. To assess model performance, we develop SECQUE-Judge, an evaluation mechanism leveraging multiple LLM-based judges, which demonstrates strong alignment with human evaluations. Additionally, we provide an extensive analysis of various models' performance on our benchmark. By making SECQUE publicly available, we aim to facilitate further research and advancements in financial AI. 

---
# Hessian of Perplexity for Large Language Models by PyTorch autograd (Open Source) 

**Authors**: Ivan Ilin  

**Link**: [PDF](https://arxiv.org/pdf/2504.04520)  

**Abstract**: Computing the full Hessian matrix -- the matrix of second-order derivatives for an entire Large Language Model (LLM) is infeasible due to its sheer size. In this technical report, we aim to provide a comprehensive guide on how to accurately compute at least a small portion of the Hessian for LLMs using PyTorch autograd library. We also demonstrate how to compute the full diagonal of the Hessian matrix using multiple samples of vector-Hessian Products (HVPs). We hope that both this guide and the accompanying GitHub code will be valuable resources for practitioners and researchers interested in better understanding the behavior and structure of the Hessian in LLMs. 

---
# Prot42: a Novel Family of Protein Language Models for Target-aware Protein Binder Generation 

**Authors**: Mohammad Amaan Sayeed, Engin Tekin, Maryam Nadeem, Nancy A. ElNaker, Aahan Singh, Natalia Vassilieva, Boulbaba Ben Amor  

**Link**: [PDF](https://arxiv.org/pdf/2504.04453)  

**Abstract**: Unlocking the next generation of biotechnology and therapeutic innovation demands overcoming the inherent complexity and resource-intensity of conventional protein engineering methods. Recent GenAI-powered computational techniques often rely on the availability of the target protein's 3D structures and specific binding sites to generate high-affinity binders, constraints exhibited by models such as AlphaProteo and RFdiffusion. In this work, we explore the use of Protein Language Models (pLMs) for high-affinity binder generation. We introduce Prot42, a novel family of Protein Language Models (pLMs) pretrained on vast amounts of unlabeled protein sequences. By capturing deep evolutionary, structural, and functional insights through an advanced auto-regressive, decoder-only architecture inspired by breakthroughs in natural language processing, Prot42 dramatically expands the capabilities of computational protein design based on language only. Remarkably, our models handle sequences up to 8,192 amino acids, significantly surpassing standard limitations and enabling precise modeling of large proteins and complex multi-domain sequences. Demonstrating powerful practical applications, Prot42 excels in generating high-affinity protein binders and sequence-specific DNA-binding proteins. Our innovative models are publicly available, offering the scientific community an efficient and precise computational toolkit for rapid protein engineering. 

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
# Gating is Weighting: Understanding Gated Linear Attention through In-context Learning 

**Authors**: Yingcong Li, Davoud Ataee Tarzanagh, Ankit Singh Rawat, Maryam Fazel, Samet Oymak  

**Link**: [PDF](https://arxiv.org/pdf/2504.04308)  

**Abstract**: Linear attention methods offer a compelling alternative to softmax attention due to their efficiency in recurrent decoding. Recent research has focused on enhancing standard linear attention by incorporating gating while retaining its computational benefits. Such Gated Linear Attention (GLA) architectures include competitive models such as Mamba and RWKV. In this work, we investigate the in-context learning capabilities of the GLA model and make the following contributions. We show that a multilayer GLA can implement a general class of Weighted Preconditioned Gradient Descent (WPGD) algorithms with data-dependent weights. These weights are induced by the gating mechanism and the input, enabling the model to control the contribution of individual tokens to prediction. To further understand the mechanics of this weighting, we introduce a novel data model with multitask prompts and characterize the optimization landscape of learning a WPGD algorithm. Under mild conditions, we establish the existence and uniqueness (up to scaling) of a global minimum, corresponding to a unique WPGD solution. Finally, we translate these findings to explore the optimization landscape of GLA and shed light on how gating facilitates context-aware learning and when it is provably better than vanilla linear attention. 

---
# Beyond the Hype: Embeddings vs. Prompting for Multiclass Classification Tasks 

**Authors**: Marios Kokkodis, Richard Demsyn-Jones, Vijay Raghavan  

**Link**: [PDF](https://arxiv.org/pdf/2504.04277)  

**Abstract**: Are traditional classification approaches irrelevant in this era of AI hype? We show that there are multiclass classification problems where predictive models holistically outperform LLM prompt-based frameworks. Given text and images from home-service project descriptions provided by Thumbtack customers, we build embeddings-based softmax models that predict the professional category (e.g., handyman, bathroom remodeling) associated with each problem description. We then compare against prompts that ask state-of-the-art LLM models to solve the same problem. We find that the embeddings approach outperforms the best LLM prompts in terms of accuracy, calibration, latency, and financial cost. In particular, the embeddings approach has 49.5% higher accuracy than the prompting approach, and its superiority is consistent across text-only, image-only, and text-image problem descriptions. Furthermore, it yields well-calibrated probabilities, which we later use as confidence signals to provide contextualized user experience during deployment. On the contrary, prompting scores are overly uninformative. Finally, the embeddings approach is 14 and 81 times faster than prompting in processing images and text respectively, while under realistic deployment assumptions, it can be up to 10 times cheaper. Based on these results, we deployed a variation of the embeddings approach, and through A/B testing we observed performance consistent with our offline analysis. Our study shows that for multiclass classification problems that can leverage proprietary datasets, an embeddings-based approach may yield unequivocally better results. Hence, scientists, practitioners, engineers, and business leaders can use our study to go beyond the hype and consider appropriate predictive models for their classification use cases. 

---
# PEIRCE: Unifying Material and Formal Reasoning via LLM-Driven Neuro-Symbolic Refinement 

**Authors**: Xin Quan, Marco Valentino, Danilo S. Carvalho, Dhairya Dalal, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2504.04110)  

**Abstract**: A persistent challenge in AI is the effective integration of material and formal inference - the former concerning the plausibility and contextual relevance of arguments, while the latter focusing on their logical and structural validity. Large Language Models (LLMs), by virtue of their extensive pre-training on large textual corpora, exhibit strong capabilities in material inference. However, their reasoning often lacks formal rigour and verifiability. At the same time, LLMs' linguistic competence positions them as a promising bridge between natural and formal languages, opening up new opportunities for combining these two modes of reasoning. In this paper, we introduce PEIRCE, a neuro-symbolic framework designed to unify material and formal inference through an iterative conjecture-criticism process. Within this framework, LLMs play the central role of generating candidate solutions in natural and formal languages, which are then evaluated and refined via interaction with external critique models. These critiques include symbolic provers, which assess formal validity, as well as soft evaluators that measure the quality of the generated arguments along linguistic and epistemic dimensions such as plausibility, coherence, and parsimony. While PEIRCE is a general-purpose framework, we demonstrate its capabilities in the domain of natural language explanation generation - a setting that inherently demands both material adequacy and formal correctness. 

---
# OpenCodeInstruct: A Large-scale Instruction Tuning Dataset for Code LLMs 

**Authors**: Wasi Uddin Ahmad, Aleksander Ficek, Mehrzad Samadi, Jocelyn Huang, Vahid Noroozi, Somshubra Majumdar, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2504.04030)  

**Abstract**: Large Language Models (LLMs) have transformed software development by enabling code generation, automated debugging, and complex reasoning. However, their continued advancement is constrained by the scarcity of high-quality, publicly available supervised fine-tuning (SFT) datasets tailored for coding tasks. To bridge this gap, we introduce OpenCodeInstruct, the largest open-access instruction tuning dataset, comprising 5 million diverse samples. Each sample includes a programming question, solution, test cases, execution feedback, and LLM-generated quality assessments. We fine-tune various base models, including LLaMA and Qwen, across multiple scales (1B+, 3B+, and 7B+) using our dataset. Comprehensive evaluations on popular benchmarks (HumanEval, MBPP, LiveCodeBench, and BigCodeBench) demonstrate substantial performance improvements achieved by SFT with OpenCodeInstruct. We also present a detailed methodology encompassing seed data curation, synthetic instruction and solution generation, and filtering. 

---
# VideoComp: Advancing Fine-Grained Compositional and Temporal Alignment in Video-Text Models 

**Authors**: Dahun Kim, AJ Piergiovanni, Ganesh Mallya, Anelia Angelova  

**Link**: [PDF](https://arxiv.org/pdf/2504.03970)  

**Abstract**: We introduce VideoComp, a benchmark and learning framework for advancing video-text compositionality understanding, aimed at improving vision-language models (VLMs) in fine-grained temporal alignment. Unlike existing benchmarks focused on static image-text compositionality or isolated single-event videos, our benchmark targets alignment in continuous multi-event videos. Leveraging video-text datasets with temporally localized event captions (e.g. ActivityNet-Captions, YouCook2), we construct two compositional benchmarks, ActivityNet-Comp and YouCook2-Comp. We create challenging negative samples with subtle temporal disruptions such as reordering, action word replacement, partial captioning, and combined disruptions. These benchmarks comprehensively test models' compositional sensitivity across extended, cohesive video-text sequences. To improve model performance, we propose a hierarchical pairwise preference loss that strengthens alignment with temporally accurate pairs and gradually penalizes increasingly disrupted ones, encouraging fine-grained compositional learning. To mitigate the limited availability of densely annotated video data, we introduce a pretraining strategy that concatenates short video-caption pairs to simulate multi-event sequences. We evaluate video-text foundational models and large multimodal models (LMMs) on our benchmark, identifying both strengths and areas for improvement in compositionality. Overall, our work provides a comprehensive framework for evaluating and enhancing model capabilities in achieving fine-grained, temporally coherent video-text alignment. 

---
# Distillation and Refinement of Reasoning in Small Language Models for Document Re-ranking 

**Authors**: Chris Samarinas, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2504.03947)  

**Abstract**: We present a novel approach for training small language models for reasoning-intensive document ranking that combines knowledge distillation with reinforcement learning optimization. While existing methods often rely on expensive human annotations or large black-box language models, our methodology leverages web data and a teacher LLM to automatically generate high-quality training examples with relevance explanations. By framing document ranking as a reinforcement learning problem and incentivizing explicit reasoning capabilities, we train a compact 3B parameter language model that achieves state-of-the-art performance on the BRIGHT benchmark. Our model ranks third on the leaderboard while using substantially fewer parameters than other approaches, outperforming models that are over 20 times larger. Through extensive experiments, we demonstrate that generating explanations during inference, rather than directly predicting relevance scores, enables more effective reasoning with smaller language models. The self-supervised nature of our method offers a scalable and interpretable solution for modern information retrieval systems. 

---
# Recursive Training Loops in LLMs: How training data properties modulate distribution shift in generated data? 

**Authors**: Grgur Kovač, Jérémy Perez, Rémy Portelas, Peter Ford Dominey, Pierre-Yves Oudeyer  

**Link**: [PDF](https://arxiv.org/pdf/2504.03814)  

**Abstract**: Large language models (LLMs) are increasingly contributing to the creation of content on the Internet. This creates a feedback loop as subsequent generations of models will be trained on this generated, synthetic data. This phenomenon is receiving increasing interest, in particular because previous studies have shown that it may lead to distribution shift - models misrepresent and forget the true underlying distributions of human data they are expected to approximate (e.g. resulting in a drastic loss of quality). In this study, we study the impact of human data properties on distribution shift dynamics in iterated training loops. We first confirm that the distribution shift dynamics greatly vary depending on the human data by comparing four datasets (two based on Twitter and two on Reddit). We then test whether data quality may influence the rate of this shift. We find that it does on the twitter, but not on the Reddit datasets. We then focus on a Reddit dataset and conduct a more exhaustive evaluation of a large set of dataset properties. This experiment associated lexical diversity with larger, and semantic diversity with smaller detrimental shifts, suggesting that incorporating text with high lexical (but limited semantic) diversity could exacerbate the degradation of generated text. We then focus on the evolution of political bias, and find that the type of shift observed (bias reduction, amplification or inversion) depends on the political lean of the human (true) distribution. Overall, our work extends the existing literature on the consequences of recursive fine-tuning by showing that this phenomenon is highly dependent on features of the human data on which training occurs. This suggests that different parts of internet (e.g. GitHub, Reddit) may undergo different types of shift depending on their properties. 

---
# FlowKV: A Disaggregated Inference Framework with Low-Latency KV Cache Transfer and Load-Aware Scheduling 

**Authors**: Weiqing Li, Guochao Jiang, Xiangyong Ding, Zhangcheng Tao, Chuzhan Hao, Chenfeng Xu, Yuewei Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03775)  

**Abstract**: Disaggregated inference has become an essential framework that separates the prefill (P) and decode (D) stages in large language model inference to improve throughput. However, the KV cache transfer faces significant delays between prefill and decode nodes. The block-wise calling method and discontinuous KV cache memory allocation increase the number of calls to the transmission kernel. Additionally, existing frameworks often fix the roles of P and D nodes, leading to computational imbalances. In this paper, we propose FlowKV, a novel disaggregated inference framework, which reduces the average transmission latency of KV cache by 96%, from 0.944s to 0.053s, almost eliminating the transfer time relative to the total request latency by optimizing the KV cache transfer. FlowKV introduces the Load-Aware Scheduler for balanced request scheduling and flexible PD node allocation. This design maximizes hardware resource utilization, achieving peak system throughput across various scenarios, including normal, computational imbalance, and extreme overload conditions. Experimental results demonstrate that FlowKV significantly accelerates inference by 15.2%-48.9% on LongBench dataset compared to the baseline and supports applications with heterogeneous GPUs. 

---
# TDBench: Benchmarking Vision-Language Models in Understanding Top-Down Images 

**Authors**: Kaiyuan Hou, Minghui Zhao, Lilin Xu, Yuang Fan, Xiaofan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03748)  

**Abstract**: The rapid emergence of Vision-Language Models (VLMs) has significantly advanced multimodal understanding, enabling applications in scene comprehension and visual reasoning. While these models have been primarily evaluated and developed for front-view image understanding, their capabilities in interpreting top-down images have received limited attention, partly due to the scarcity of diverse top-down datasets and the challenges in collecting such data. In contrast, top-down vision provides explicit spatial overviews and improved contextual understanding of scenes, making it particularly valuable for tasks like autonomous navigation, aerial imaging, and spatial planning. In this work, we address this gap by introducing TDBench, a comprehensive benchmark for VLMs in top-down image understanding. TDBench is constructed from public top-down view datasets and high-quality simulated images, including diverse real-world and synthetic scenarios. TDBench consists of visual question-answer pairs across ten evaluation dimensions of image understanding. Moreover, we conduct four case studies that commonly happen in real-world scenarios but are less explored. By revealing the strengths and limitations of existing VLM through evaluation results, we hope TDBench to provide insights for motivating future research. Project homepage: this https URL 

---
# Misaligned Roles, Misplaced Images: Structural Input Perturbations Expose Multimodal Alignment Blind Spots 

**Authors**: Erfan Shayegani, G M Shahariar, Sara Abdali, Lei Yu, Nael Abu-Ghazaleh, Yue Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.03735)  

**Abstract**: Multimodal Language Models (MMLMs) typically undergo post-training alignment to prevent harmful content generation. However, these alignment stages focus primarily on the assistant role, leaving the user role unaligned, and stick to a fixed input prompt structure of special tokens, leaving the model vulnerable when inputs deviate from these expectations. We introduce Role-Modality Attacks (RMA), a novel class of adversarial attacks that exploit role confusion between the user and assistant and alter the position of the image token to elicit harmful outputs. Unlike existing attacks that modify query content, RMAs manipulate the input structure without altering the query itself. We systematically evaluate these attacks across multiple Vision Language Models (VLMs) on eight distinct settings, showing that they can be composed to create stronger adversarial prompts, as also evidenced by their increased projection in the negative refusal direction in the residual stream, a property observed in prior successful attacks. Finally, for mitigation, we propose an adversarial training approach that makes the model robust against input prompt perturbations. By training the model on a range of harmful and benign prompts all perturbed with different RMA settings, it loses its sensitivity to Role Confusion and Modality Manipulation attacks and is trained to only pay attention to the content of the query in the input prompt structure, effectively reducing Attack Success Rate (ASR) while preserving the model's general utility. 

---
# CrowdVLM-R1: Expanding R1 Ability to Vision Language Model for Crowd Counting using Fuzzy Group Relative Policy Reward 

**Authors**: Zhiqiang Wang, Pengbin Feng, Yanbin Lin, Shuzhang Cai, Zongao Bian, Jinghua Yan, Xingquan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03724)  

**Abstract**: We propose Fuzzy Group Relative Policy Reward (FGRPR), a novel framework that integrates Group Relative Policy Optimization (GRPO) with a fuzzy reward function to enhance learning efficiency. Unlike the conventional binary 0/1 accuracy reward, our fuzzy reward model provides nuanced incentives, encouraging more precise outputs. Experimental results demonstrate that GRPO with a standard 0/1 accuracy reward underperforms compared to supervised fine-tuning (SFT). In contrast, FGRPR, applied to Qwen2.5-VL(3B and 7B), surpasses all baseline models, including GPT4o, LLaMA2(90B), and SFT, across five in-domain datasets. On an out-of-domain dataset, FGRPR achieves performance comparable to SFT but excels when target values are larger, as its fuzzy reward function assigns higher rewards to closer approximations. This approach is broadly applicable to tasks where the precision of the answer is critical. Code and data: this https URL 

---
# Breach in the Shield: Unveiling the Vulnerabilities of Large Language Models 

**Authors**: Runpeng Dai, Run Yang, Fan Zhou, Hongtu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03714)  

**Abstract**: Large Language Models (LLMs) and Vision-Language Models (VLMs) have become essential to general artificial intelligence, exhibiting remarkable capabilities in task understanding and problem-solving. However, the real-world reliability of these models critically depends on their stability, which remains an underexplored area. Despite their widespread use, rigorous studies examining the stability of LLMs under various perturbations are still lacking. In this paper, we address this gap by proposing a novel stability measure for LLMs, inspired by statistical methods rooted in information geometry. Our measure possesses desirable invariance properties, making it well-suited for analyzing model sensitivity to both parameter and input perturbations. To assess the effectiveness of our approach, we conduct extensive experiments on models ranging in size from 1.5B to 13B parameters. Our results demonstrate the utility of our measure in identifying salient parameters and detecting vulnerable regions in input images or critical dimensions in token embeddings. Furthermore, leveraging our stability framework, we enhance model robustness during model merging, leading to improved performance. 

---
