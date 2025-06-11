# Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions 

**Authors**: Clara Lachenmaier, Judith Sieker, Sina Zarrieß  

**Link**: [PDF](https://arxiv.org/pdf/2506.08952)  

**Abstract**: Communication among humans relies on conversational grounding, allowing interlocutors to reach mutual understanding even when they do not have perfect knowledge and must resolve discrepancies in each other's beliefs. This paper investigates how large language models (LLMs) manage common ground in cases where they (don't) possess knowledge, focusing on facts in the political domain where the risk of misinformation and grounding failure is high. We examine the ability of LLMs to answer direct knowledge questions and loaded questions that presuppose misinformation. We evaluate whether loaded questions lead LLMs to engage in active grounding and correct false user beliefs, in connection to their level of knowledge and their political bias. Our findings highlight significant challenges in LLMs' ability to engage in grounding and reject false user beliefs, raising concerns about their role in mitigating misinformation in political discourse. 

---
# PropMEND: Hypernetworks for Knowledge Propagation in LLMs 

**Authors**: Zeyu Leo Liu, Greg Durrett, Eunsol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08920)  

**Abstract**: Knowledge editing techniques for large language models (LLMs) can inject knowledge that is later reproducible verbatim, but they fall short on propagating that knowledge: models cannot answer questions that require reasoning with the injected knowledge. We present a hypernetwork-based approach for knowledge propagation, named PropMEND, where we meta-learn how to modify gradients of a language modeling loss to encourage injected information to propagate. Our approach extends the meta-objective of MEND [29] so that gradient updates on knowledge are transformed to enable answering multi-hop questions involving that knowledge. We show improved performance on the RippleEdit dataset, showing almost 2x accuracy on challenging multi-hop questions whose answers are not explicitly stated in the injected fact. We further introduce a new dataset, Controlled RippleEdit, to evaluate the generalization of our hypernetwork, testing knowledge propagation along relations and entities unseen during hypernetwork training. PropMEND still outperforms existing approaches in unseen entity-relation pairs, yet the performance gap decreases substantially, suggesting future work in propagating knowledge to a wide range of relations. 

---
# Can A Gamer Train A Mathematical Reasoning Model? 

**Authors**: Andrew Shin  

**Link**: [PDF](https://arxiv.org/pdf/2506.08935)  

**Abstract**: While large language models (LLMs) have achieved remarkable performance in various tasks including mathematical reasoning, their development typically demands prohibitive computational resources. Recent advancements have reduced costs for training capable models, yet even these approaches rely on high-end hardware clusters. In this paper, we demonstrate that a single average gaming GPU can train a solid mathematical reasoning model, by integrating reinforcement learning and memory optimization techniques. Specifically, we train a 1.5B parameter mathematical reasoning model on RTX 3080 Ti of 16GB memory that achieves comparable or better performance on mathematical reasoning benchmarks than models several times larger, in resource-constrained environments. Our results challenge the paradigm that state-of-the-art mathematical reasoning necessitates massive infrastructure, democratizing access to high-performance AI research. this https URL. 

---
# From Legal Texts to Defeasible Deontic Logic via LLMs: A Study in Automated Semantic Analysis 

**Authors**: Elias Horner, Cristinel Mateis, Guido Governatori, Agata Ciabattoni  

**Link**: [PDF](https://arxiv.org/pdf/2506.08899)  

**Abstract**: We present a novel approach to the automated semantic analysis of legal texts using large language models (LLMs), targeting their transformation into formal representations in Defeasible Deontic Logic (DDL). We propose a structured pipeline that segments complex normative language into atomic snippets, extracts deontic rules, and evaluates them for syntactic and semantic coherence. Our methodology is evaluated across various LLM configurations, including prompt engineering strategies, fine-tuned models, and multi-stage pipelines, focusing on legal norms from the Australian Telecommunications Consumer Protections Code. Empirical results demonstrate promising alignment between machine-generated and expert-crafted formalizations, showing that LLMs - particularly when prompted effectively - can significantly contribute to scalable legal informatics. 

---
# Learning to Reason Across Parallel Samples for LLM Reasoning 

**Authors**: Jianing Qi, Xi Ye, Hao Tang, Zhigang Zhu, Eunsol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.09014)  

**Abstract**: Scaling test-time compute brings substantial performance gains for large language models (LLMs). By sampling multiple answers and heuristically aggregate their answers (e.g., either through majority voting or using verifiers to rank the answers), one can achieve consistent performance gains in math domains. In this paper, we propose a new way to leverage such multiple sample set. We train a compact LLM, called Sample Set Aggregator (SSA), that takes a concatenated sequence of multiple samples and output the final answer, optimizing it for the answer accuracy with reinforcement learning. Experiments on multiple reasoning datasets show that SSA outperforms other test-time scaling methods such as reward model-based re-ranking. Our approach also shows a promising generalization ability, across sample set sizes, base model families and scales, and tasks. By separating LLMs to generate answers and LLMs to analyze and aggregate sampled answers, our approach can work with the outputs from premier black box models easily and efficiently. 

---
# Enhancing Accuracy and Maintainability in Nuclear Plant Data Retrieval: A Function-Calling LLM Approach Over NL-to-SQL 

**Authors**: Mishca de Costa, Muhammad Anwar, Dave Mercier, Mark Randall, Issam Hammad  

**Link**: [PDF](https://arxiv.org/pdf/2506.08757)  

**Abstract**: Retrieving operational data from nuclear power plants requires exceptional accuracy and transparency due to the criticality of the decisions it supports. Traditionally, natural language to SQL (NL-to-SQL) approaches have been explored for querying such data. While NL-to-SQL promises ease of use, it poses significant risks: end-users cannot easily validate generated SQL queries, and legacy nuclear plant databases -- often complex and poorly structured -- complicate query generation due to decades of incremental modifications. These challenges increase the likelihood of inaccuracies and reduce trust in the approach. In this work, we propose an alternative paradigm: leveraging function-calling large language models (LLMs) to address these challenges. Instead of directly generating SQL queries, we define a set of pre-approved, purpose-specific functions representing common use cases. Queries are processed by invoking these functions, which encapsulate validated SQL logic. This hybrid approach mitigates the risks associated with direct NL-to-SQL translations by ensuring that SQL queries are reviewed and optimized by experts before deployment. While this strategy introduces the upfront cost of developing and maintaining the function library, we demonstrate how NL-to-SQL tools can assist in the initial generation of function code, allowing experts to focus on validation rather than creation. Our study includes a performance comparison between direct NL-to-SQL generation and the proposed function-based approach, highlighting improvements in accuracy and maintainability. This work underscores the importance of balancing user accessibility with operational safety and provides a novel, actionable framework for robust data retrieval in critical systems. 

---
# Dialect Normalization using Large Language Models and Morphological Rules 

**Authors**: Antonios Dimakis, John Pavlopoulos, Antonios Anastasopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.08907)  

**Abstract**: Natural language understanding systems struggle with low-resource languages, including many dialects of high-resource ones. Dialect-to-standard normalization attempts to tackle this issue by transforming dialectal text so that it can be used by standard-language tools downstream. In this study, we tackle this task by introducing a new normalization method that combines rule-based linguistically informed transformations and large language models (LLMs) with targeted few-shot prompting, without requiring any parallel data. We implement our method for Greek dialects and apply it on a dataset of regional proverbs, evaluating the outputs using human annotators. We then use this dataset to conduct downstream experiments, finding that previous results regarding these proverbs relied solely on superficial linguistic information, including orthographic artifacts, while new observations can still be made through the remaining semantics. 

---
# AraReasoner: Evaluating Reasoning-Based LLMs for Arabic NLP 

**Authors**: Ahmed Hasanaath, Aisha Alansari, Ahmed Ashraf, Chafik Salmane, Hamzah Luqman, Saad Ezzini  

**Link**: [PDF](https://arxiv.org/pdf/2506.08768)  

**Abstract**: Large language models (LLMs) have shown remarkable progress in reasoning abilities and general natural language processing (NLP) tasks, yet their performance on Arabic data, characterized by rich morphology, diverse dialects, and complex script, remains underexplored. This paper presents a comprehensive benchmarking study of multiple reasoning-focused LLMs, with a special emphasis on the newly introduced DeepSeek models, across a suite of fifteen Arabic NLP tasks. We experiment with various strategies, including zero-shot, few-shot, and fine-tuning. This allows us to systematically evaluate performance on datasets covering a range of applications to examine their capacity for linguistic reasoning under different levels of complexity. Our experiments reveal several key findings. First, carefully selecting just three in-context examples delivers an average uplift of over 13 F1 points on classification tasks-boosting sentiment analysis from 35.3% to 87.5% and paraphrase detection from 56.1% to 87.0%. Second, reasoning-focused DeepSeek architectures outperform a strong GPT o4-mini baseline by an average of 12 F1 points on complex inference tasks in the zero-shot setting. Third, LoRA-based fine-tuning yields up to an additional 8 points in F1 and BLEU compared to equivalent increases in model scale. The code is available at this https URL 

---
# AdversariaL attacK sAfety aLIgnment(ALKALI): Safeguarding LLMs through GRACE: Geometric Representation-Aware Contrastive Enhancement- Introducing Adversarial Vulnerability Quality Index (AVQI) 

**Authors**: Danush Khanna, Krishna Kumar, Basab Ghosh, Vinija Jain, Vasu Sharma, Aman Chadha, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2506.08885)  

**Abstract**: Adversarial threats against LLMs are escalating faster than current defenses can adapt. We expose a critical geometric blind spot in alignment: adversarial prompts exploit latent camouflage, embedding perilously close to the safe representation manifold while encoding unsafe intent thereby evading surface level defenses like Direct Preference Optimization (DPO), which remain blind to the latent geometry. We introduce ALKALI, the first rigorously curated adversarial benchmark and the most comprehensive to date spanning 9,000 prompts across three macro categories, six subtypes, and fifteen attack families. Evaluation of 21 leading LLMs reveals alarmingly high Attack Success Rates (ASRs) across both open and closed source models, exposing an underlying vulnerability we term latent camouflage, a structural blind spot where adversarial completions mimic the latent geometry of safe ones. To mitigate this vulnerability, we introduce GRACE - Geometric Representation Aware Contrastive Enhancement, an alignment framework coupling preference learning with latent space regularization. GRACE enforces two constraints: latent separation between safe and adversarial completions, and adversarial cohesion among unsafe and jailbreak behaviors. These operate over layerwise pooled embeddings guided by a learned attention profile, reshaping internal geometry without modifying the base model, and achieve up to 39% ASR reduction. Moreover, we introduce AVQI, a geometry aware metric that quantifies latent alignment failure via cluster separation and compactness. AVQI reveals when unsafe completions mimic the geometry of safe ones, offering a principled lens into how models internally encode safety. We make the code publicly available at this https URL. 

---
# FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation 

**Authors**: Qinggang Zhang, Zhishang Xiang, Yilin Xiao, Le Wang, Junhui Li, Xinrun Wang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.08938)  

**Abstract**: Large language models (LLMs) augmented with retrieval systems have demonstrated significant potential in handling knowledge-intensive tasks. However, these models often struggle with unfaithfulness issues, generating outputs that either ignore the retrieved context or inconsistently blend it with the LLM`s parametric knowledge. This issue is particularly severe in cases of knowledge conflict, where the retrieved context conflicts with the model`s parametric knowledge. While existing faithful RAG approaches enforce strict context adherence through well-designed prompts or modified decoding strategies, our analysis reveals a critical limitation: they achieve faithfulness by forcibly suppressing the model`s parametric knowledge, which undermines the model`s internal knowledge structure and increases the risk of misinterpreting the context. To this end, this paper proposes FaithfulRAG, a novel framework that resolves knowledge conflicts by explicitly modeling discrepancies between the model`s parametric knowledge and retrieved context. Specifically, FaithfulRAG identifies conflicting knowledge at the fact level and designs a self-thinking process, allowing LLMs to reason about and integrate conflicting facts before generating responses. Extensive experiments demonstrate that our method outperforms state-of-the-art methods. The code is available at https:// this http URL 

---
# Improved LLM Agents for Financial Document Question Answering 

**Authors**: Nelvin Tan, Zian Seng, Liang Zhang, Yu-Ching Shih, Dong Yang, Amol Salunkhe  

**Link**: [PDF](https://arxiv.org/pdf/2506.08726)  

**Abstract**: Large language models (LLMs) have shown impressive capabilities on numerous natural language processing tasks. However, LLMs still struggle with numerical question answering for financial documents that include tabular and textual data. Recent works have showed the effectiveness of critic agents (i.e., self-correction) for this task given oracle labels. Building upon this framework, this paper examines the effectiveness of the traditional critic agent when oracle labels are not available, and show, through experiments, that this critic agent's performance deteriorates in this scenario. With this in mind, we present an improved critic agent, along with the calculator agent which outperforms the previous state-of-the-art approach (program-of-thought) and is safer. Furthermore, we investigate how our agents interact with each other, and how this interaction affects their performance. 

---
# The impact of fine tuning in LLaMA on hallucinations for named entity extraction in legal documentation 

**Authors**: Francisco Vargas, Alejandro González Coene, Gaston Escalante, Exequiel Lobón, Manuel Pulido  

**Link**: [PDF](https://arxiv.org/pdf/2506.08827)  

**Abstract**: The extraction of information about traffic accidents from legal documents is crucial for quantifying insurance company costs. Extracting entities such as percentages of physical and/or psychological disability and the involved compensation amounts is a challenging process, even for experts, due to the subtle arguments and reasoning in the court decision. A two-step procedure is proposed: first, segmenting the document identifying the most relevant segments, and then extracting the entities. For text segmentation, two methodologies are compared: a classic method based on regular expressions and a second approach that divides the document into blocks of n-tokens, which are then vectorized using multilingual models for semantic searches (text-embedding-ada-002/MiniLM-L12-v2 ). Subsequently, large language models (LLaMA-2 7b, 70b, LLaMA-3 8b, and GPT-4 Turbo) are applied with prompting to the selected segments for entity extraction. For the LLaMA models, fine-tuning is performed using LoRA. LLaMA-2 7b, even with zero temperature, shows a significant number of hallucinations in extractions which are an important contention point for named entity extraction. This work shows that these hallucinations are substantially reduced after finetuning the model. The performance of the methodology based on segment vectorization and subsequent use of LLMs significantly surpasses the classic method which achieves an accuracy of 39.5%. Among open-source models, LLaMA-2 70B with finetuning achieves the highest accuracy 79.4%, surpassing its base version 61.7%. Notably, the base LLaMA-3 8B model already performs comparably to the finetuned LLaMA-2 70B model, achieving 76.6%, highlighting the rapid progress in model development. Meanwhile, GPT-4 Turbo achieves the highest accuracy at 86.1%. 

---
# Brevity is the soul of sustainability: Characterizing LLM response lengths 

**Authors**: Soham Poddar, Paramita Koley, Janardan Misra, Sanjay Podder, Navveen Balani, Niloy Ganguly, Saptarshi Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2506.08686)  

**Abstract**: A significant portion of the energy consumed by Large Language Models (LLMs) arises from their inference processes; hence developing energy-efficient methods for inference is crucial. While several techniques exist for inference optimization, output compression remains relatively unexplored, with only a few preliminary efforts addressing this aspect. In this work, we first benchmark 12 decoder-only LLMs across 5 datasets, revealing that these models often produce responses that are substantially longer than necessary. We then conduct a comprehensive quality assessment of LLM responses, formally defining six information categories present in LLM responses. We show that LLMs often tend to include redundant or additional information besides the minimal answer. To address this issue of long responses by LLMs, we explore several simple and intuitive prompt-engineering strategies. Empirical evaluation shows that appropriate prompts targeting length reduction and controlling information content can achieve significant energy optimization between 25-60\% by reducing the response length while preserving the quality of LLM responses. 

---
# Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning 

**Authors**: Haozhen Zhang, Tao Feng, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2506.09033)  

**Abstract**: The rapid emergence of diverse large language models (LLMs) has spurred the development of LLM routers that assign user queries to the most suitable model. However, existing LLM routers typically perform a single-round, one-to-one mapping (\textit{i.e.}, assigning each query to a single model in isolation), which limits their capability to tackle complex tasks that demand the complementary strengths of multiple LLMs. In this paper, we present \textbf{Router-R1}, a reinforcement learning (RL)-based framework that formulates multi-LLM routing and aggregation as a sequential decision process. Router-R1 instantiates the router itself as a capable LLM, leveraging its reasoning ability to interleave "think" actions (internal deliberation) with "route" actions (dynamic model invocation), and integrates each response into its evolving context. To guide learning, we employ a lightweight rule-based reward comprising format rewards, final outcome rewards, and a novel cost reward for performance and cost trade-off optimization, opening a pathway toward optimizing performance-cost tradeoffs via RL. Router-R1 also conditions only on simple model descriptors such as pricing, latency, and example performance, enabling strong generalization to unseen model selection. Experiments on seven general and multi-hop QA benchmarks show that Router-R1 outperforms over several strong baselines, achieving superior performance while maintaining robust generalization and cost this http URL is available at this https URL. 

---
# Summarization for Generative Relation Extraction in the Microbiome Domain 

**Authors**: Oumaima El Khettari, Solen Quiniou, Samuel Chaffron  

**Link**: [PDF](https://arxiv.org/pdf/2506.08647)  

**Abstract**: We explore a generative relation extraction (RE) pipeline tailored to the study of interactions in the intestinal microbiome, a complex and low-resource biomedical domain. Our method leverages summarization with large language models (LLMs) to refine context before extracting relations via instruction-tuned generation. Preliminary results on a dedicated corpus show that summarization improves generative RE performance by reducing noise and guiding the model. However, BERT-based RE approaches still outperform generative models. This ongoing work demonstrates the potential of generative methods to support the study of specialized domains in low-resources setting. 

---
# ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Large Language Model Preference Optimization 

**Authors**: Hee Suk Yoon, Eunseop Yoon, Mark A. Hasegawa-Johnson, Sungwoong Kim, Chang D. Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2506.08712)  

**Abstract**: We introduce ConfPO, a method for preference learning in Large Language Models (LLMs) that identifies and optimizes preference-critical tokens based solely on the training policy's confidence, without requiring any auxiliary models or compute. Unlike prior Direct Alignment Algorithms (DAAs) such as Direct Preference Optimization (DPO), which uniformly adjust all token probabilities regardless of their relevance to preference, ConfPO focuses optimization on the most impactful tokens. This targeted approach improves alignment quality while mitigating overoptimization (i.e., reward hacking) by using the KL divergence budget more efficiently. In contrast to recent token-level methods that rely on credit-assignment models or AI annotators, raising concerns about scalability and reliability, ConfPO is simple, lightweight, and model-free. Experimental results on challenging alignment benchmarks, including AlpacaEval 2 and Arena-Hard, demonstrate that ConfPO consistently outperforms uniform DAAs across various LLMs, delivering better alignment with zero additional computational overhead. 

---
# Factors affecting the in-context learning abilities of LLMs for dialogue state tracking 

**Authors**: Pradyoth Hegde, Santosh Kesiraju, Jan Švec, Šimon Sedláček, Bolaji Yusuf, Oldřich Plchot, Deepak K T, Jan Černocký  

**Link**: [PDF](https://arxiv.org/pdf/2506.08753)  

**Abstract**: This study explores the application of in-context learning (ICL) to the dialogue state tracking (DST) problem and investigates the factors that influence its effectiveness. We use a sentence embedding based k-nearest neighbour method to retrieve the suitable demonstrations for ICL. The selected demonstrations, along with the test samples, are structured within a template as input to the LLM. We then conduct a systematic study to analyse the impact of factors related to demonstration selection and prompt context on DST performance. This work is conducted using the MultiWoZ2.4 dataset and focuses primarily on the OLMo-7B-instruct, Mistral-7B-Instruct-v0.3, and Llama3.2-3B-Instruct models. Our findings provide several useful insights on in-context learning abilities of LLMs for dialogue state tracking. 

---
# Unlocking the Potential of Large Language Models in the Nuclear Industry with Synthetic Data 

**Authors**: Muhammad Anwar, Daniel Lau, Mishca de Costa, Issam Hammad  

**Link**: [PDF](https://arxiv.org/pdf/2506.08750)  

**Abstract**: The nuclear industry possesses a wealth of valuable information locked away in unstructured text data. This data, however, is not readily usable for advanced Large Language Model (LLM) applications that require clean, structured question-answer pairs for tasks like model training, fine-tuning, and evaluation. This paper explores how synthetic data generation can bridge this gap, enabling the development of robust LLMs for the nuclear domain. We discuss the challenges of data scarcity and privacy concerns inherent in the nuclear industry and how synthetic data provides a solution by transforming existing text data into usable Q&A pairs. This approach leverages LLMs to analyze text, extract key information, generate relevant questions, and evaluate the quality of the resulting synthetic dataset. By unlocking the potential of LLMs in the nuclear industry, synthetic data can pave the way for improved information retrieval, enhanced knowledge sharing, and more informed decision-making in this critical sector. 

---
# Explainable Compliance Detection with Multi-Hop Natural Language Inference on Assurance Case Structure 

**Authors**: Fariz Ikhwantri, Dusica Marijan  

**Link**: [PDF](https://arxiv.org/pdf/2506.08713)  

**Abstract**: Ensuring complex systems meet regulations typically requires checking the validity of assurance cases through a claim-argument-evidence framework. Some challenges in this process include the complicated nature of legal and technical texts, the need for model explanations, and limited access to assurance case data. We propose a compliance detection approach based on Natural Language Inference (NLI): EXplainable CompLiance detection with Argumentative Inference of Multi-hop reasoning (EXCLAIM). We formulate the claim-argument-evidence structure of an assurance case as a multi-hop inference for explainable and traceable compliance detection. We address the limited number of assurance cases by generating them using large language models (LLMs). We introduce metrics that measure the coverage and structural consistency. We demonstrate the effectiveness of the generated assurance case from GDPR requirements in a multi-hop inference task as a case study. Our results highlight the potential of NLI-based approaches in automating the regulatory compliance process. 

---
# Efficient Post-Training Refinement of Latent Reasoning in Large Language Models 

**Authors**: Xinyuan Wang, Dongjie Wang, Wangyang Ying, Haoyue Bai, Nanxu Gong, Sixun Dong, Kunpeng Liu, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08552)  

**Abstract**: Reasoning is a key component of language understanding in Large Language Models. While Chain-of-Thought prompting enhances performance via explicit intermediate steps, it suffers from sufficient token overhead and a fixed reasoning trajectory, preventing step-wise refinement. Recent advances in latent reasoning address these limitations by refining internal reasoning processes directly in the model's latent space, without producing explicit outputs. However, a key challenge remains: how to effectively update reasoning embeddings during post-training to guide the model toward more accurate solutions. To overcome this challenge, we propose a lightweight post-training framework that refines latent reasoning trajectories using two novel strategies: 1) Contrastive reasoning feedback, which compares reasoning embeddings against strong and weak baselines to infer effective update directions via embedding enhancement; 2) Residual embedding refinement, which stabilizes updates by progressively integrating current and historical gradients, enabling fast yet controlled convergence. Extensive experiments and case studies are conducted on five reasoning benchmarks to demonstrate the effectiveness of the proposed framework. Notably, a 5\% accuracy gain on MathQA without additional training. 

---
# Hateful Person or Hateful Model? Investigating the Role of Personas in Hate Speech Detection by Large Language Models 

**Authors**: Shuzhou Yuan, Ercong Nie, Mario Tawfelis, Helmut Schmid, Hinrich Schütze, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2506.08593)  

**Abstract**: Hate speech detection is a socially sensitive and inherently subjective task, with judgments often varying based on personal traits. While prior work has examined how socio-demographic factors influence annotation, the impact of personality traits on Large Language Models (LLMs) remains largely unexplored. In this paper, we present the first comprehensive study on the role of persona prompts in hate speech classification, focusing on MBTI-based traits. A human annotation survey confirms that MBTI dimensions significantly affect labeling behavior. Extending this to LLMs, we prompt four open-source models with MBTI personas and evaluate their outputs across three hate speech datasets. Our analysis uncovers substantial persona-driven variation, including inconsistencies with ground truth, inter-persona disagreement, and logit-level biases. These findings highlight the need to carefully define persona prompts in LLM-based annotation workflows, with implications for fairness and alignment with human values. 

---
# RAISE: Enhancing Scientific Reasoning in LLMs via Step-by-Step Retrieval 

**Authors**: Minhae Oh, Jeonghye Kim, Nakyung Lee, Donggeon Seo, Taeuk Kim, Jungwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.08625)  

**Abstract**: Scientific reasoning requires not only long-chain reasoning processes, but also knowledge of domain-specific terminologies and adaptation to updated findings. To deal with these challenges for scientific reasoning, we introduce RAISE, a step-by-step retrieval-augmented framework which retrieves logically relevant documents from in-the-wild corpus. RAISE is divided into three steps: problem decomposition, logical query generation, and logical retrieval. We observe that RAISE consistently outperforms other baselines on scientific reasoning benchmarks. We analyze that unlike other baselines, RAISE retrieves documents that are not only similar in terms of the domain knowledge, but also documents logically more relevant. 

---
# MEMETRON: Metaheuristic Mechanisms for Test-time Response Optimization of Large Language Models 

**Authors**: Son The Nguyen, Theja Tulabandhula  

**Link**: [PDF](https://arxiv.org/pdf/2506.08643)  

**Abstract**: Large language models (LLMs) are increasingly used for both open-ended and structured tasks, yet their inference-time behavior is still largely dictated by heuristic decoding strategies such as greedy search, sampling, or reranking. These methods provide limited control and do not explicitly optimize for task-specific objectives. We introduce MEMETRON, a task-agnostic framework that formulates LLM decoding as a discrete black-box optimization problem. MEMETRON leverages hybrid metaheuristic algorithms, GENETRON and ANNETRON, to search the response space, guided by reward models and contextual operations performed by the LLM itself. This approach enables efficient discovery of high-reward responses without requiring model retraining or gradient access. The framework is modular and generalizes across diverse tasks, requiring only a reward function and lightweight prompt templates. We evaluate our framework on the critical human preference alignment task and demonstrate that it significantly outperforms standard decoding and reranking methods, highlighting its potential to improve alignment without model retraining. 

---
# Detecting Harmful Memes with Decoupled Understanding and Guided CoT Reasoning 

**Authors**: Fengjun Pan, Anh Tuan Luu, Xiaobao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08477)  

**Abstract**: Detecting harmful memes is essential for maintaining the integrity of online environments. However, current approaches often struggle with resource efficiency, flexibility, or explainability, limiting their practical deployment in content moderation systems. To address these challenges, we introduce U-CoT+, a novel framework for harmful meme detection. Instead of relying solely on prompting or fine-tuning multimodal models, we first develop a high-fidelity meme-to-text pipeline that converts visual memes into detail-preserving textual descriptions. This design decouples meme interpretation from meme classification, thus avoiding immediate reasoning over complex raw visual content and enabling resource-efficient harmful meme detection with general large language models (LLMs). Building on these textual descriptions, we further incorporate targeted, interpretable human-crafted guidelines to guide models' reasoning under zero-shot CoT prompting. As such, this framework allows for easy adaptation to different harmfulness detection criteria across platforms, regions, and over time, offering high flexibility and explainability. Extensive experiments on seven benchmark datasets validate the effectiveness of our framework, highlighting its potential for explainable and low-resource harmful meme detection using small-scale LLMs. Codes and data are available at: this https URL. 

---
# DRAGged into Conflicts: Detecting and Addressing Conflicting Sources in Search-Augmented LLMs 

**Authors**: Arie Cattan, Alon Jacovi, Ori Ram, Jonathan Herzig, Roee Aharoni, Sasha Goldshtein, Eran Ofek, Idan Szpektor, Avi Caciularu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08500)  

**Abstract**: Retrieval Augmented Generation (RAG) is a commonly used approach for enhancing large language models (LLMs) with relevant and up-to-date information. However, the retrieved sources can often contain conflicting information and it remains unclear how models should address such discrepancies. In this work, we first propose a novel taxonomy of knowledge conflict types in RAG, along with the desired model behavior for each type. We then introduce CONFLICTS, a high-quality benchmark with expert annotations of conflict types in a realistic RAG setting. CONFLICTS is the first benchmark that enables tracking progress on how models address a wide range of knowledge conflicts. We conduct extensive experiments on this benchmark, showing that LLMs often struggle to appropriately resolve conflicts between sources. While prompting LLMs to explicitly reason about the potential conflict in the retrieved documents significantly improves the quality and appropriateness of their responses, substantial room for improvement in future research remains. 

---
# EtiCor++: Towards Understanding Etiquettical Bias in LLMs 

**Authors**: Ashutosh Dwivedi, Siddhant Shivdutt Singh, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08488)  

**Abstract**: In recent years, researchers have started analyzing the cultural sensitivity of LLMs. In this respect, Etiquettes have been an active area of research. Etiquettes are region-specific and are an essential part of the culture of a region; hence, it is imperative to make LLMs sensitive to etiquettes. However, there needs to be more resources in evaluating LLMs for their understanding and bias with regard to etiquettes. In this resource paper, we introduce EtiCor++, a corpus of etiquettes worldwide. We introduce different tasks for evaluating LLMs for knowledge about etiquettes across various regions. Further, we introduce various metrics for measuring bias in LLMs. Extensive experimentation with LLMs shows inherent bias towards certain regions. 

---
# CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmark of Large Language Models in Mental Health Counseling 

**Authors**: Yahan Li, Jifan Yao, John Bosco S. Bunyi, Adam C. Frank, Angel Hwang, Ruishan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08584)  

**Abstract**: Large language models (LLMs) are increasingly proposed for use in mental health support, yet their behavior in realistic counseling scenarios remains largely untested. We introduce CounselBench, a large-scale benchmark developed with 100 mental health professionals to evaluate and stress-test LLMs in single-turn counseling. The first component, CounselBench-EVAL, contains 2,000 expert evaluations of responses from GPT-4, LLaMA 3, Gemini, and online human therapists to real patient questions. Each response is rated along six clinically grounded dimensions, with written rationales and span-level annotations. We find that LLMs often outperform online human therapists in perceived quality, but experts frequently flag their outputs for safety concerns such as unauthorized medical advice. Follow-up experiments show that LLM judges consistently overrate model responses and overlook safety issues identified by human experts. To probe failure modes more directly, we construct CounselBench-Adv, an adversarial dataset of 120 expert-authored counseling questions designed to trigger specific model issues. Evaluation across 2,880 responses from eight LLMs reveals consistent, model-specific failure patterns. Together, CounselBench establishes a clinically grounded framework for benchmarking and improving LLM behavior in high-stakes mental health settings. 

---
# Know-MRI: A Knowledge Mechanisms Revealer&Interpreter for Large Language Models 

**Authors**: Jiaxiang Liu, Boxuan Xing, Chenhao Yuan, Chenxiang Zhang, Di Wu, Xiusheng Huang, Haida Yu, Chuhan Lang, Pengfei Cao, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08427)  

**Abstract**: As large language models (LLMs) continue to advance, there is a growing urgency to enhance the interpretability of their internal knowledge mechanisms. Consequently, many interpretation methods have emerged, aiming to unravel the knowledge mechanisms of LLMs from various perspectives. However, current interpretation methods differ in input data formats and interpreting outputs. The tools integrating these methods are only capable of supporting tasks with specific inputs, significantly constraining their practical applications. To address these challenges, we present an open-source Knowledge Mechanisms Revealer&Interpreter (Know-MRI) designed to analyze the knowledge mechanisms within LLMs systematically. Specifically, we have developed an extensible core module that can automatically match different input data with interpretation methods and consolidate the interpreting outputs. It enables users to freely choose appropriate interpretation methods based on the inputs, making it easier to comprehensively diagnose the model's internal knowledge mechanisms from multiple perspectives. Our code is available at this https URL. We also provide a demonstration video on this https URL. 

---
# CAF-I: A Collaborative Multi-Agent Framework for Enhanced Irony Detection with Large Language Models 

**Authors**: Ziqi.Liu, Ziyang.Zhou, Mingxuan.Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08430)  

**Abstract**: Large language model (LLM) have become mainstream methods in the field of sarcasm detection. However, existing LLM methods face challenges in irony detection, including: 1. single-perspective limitations, 2. insufficient comprehensive understanding, and 3. lack of interpretability. This paper introduces the Collaborative Agent Framework for Irony (CAF-I), an LLM-driven multi-agent system designed to overcome these issues. CAF-I employs specialized agents for Context, Semantics, and Rhetoric, which perform multidimensional analysis and engage in interactive collaborative optimization. A Decision Agent then consolidates these perspectives, with a Refinement Evaluator Agent providing conditional feedback for optimization. Experiments on benchmark datasets establish CAF-I's state-of-the-art zero-shot performance. Achieving SOTA on the vast majority of metrics, CAF-I reaches an average Macro-F1 of 76.31, a 4.98 absolute improvement over the strongest prior baseline. This success is attained by its effective simulation of human-like multi-perspective analysis, enhancing detection accuracy and interpretability. 

---
# Draft-based Approximate Inference for LLMs 

**Authors**: Kevin Galim, Ethan Ewer, Wonjun Kang, Minjae Lee, Hyung Il Koo, Kangwook Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.08373)  

**Abstract**: Optimizing inference for long-context Large Language Models (LLMs) is increasingly important due to the quadratic compute and linear memory complexity of Transformers. Existing approximation methods, such as key-value (KV) cache dropping, sparse attention, and prompt compression, typically rely on rough predictions of token or KV pair importance. We propose a novel framework for approximate LLM inference that leverages small draft models to more accurately predict the importance of tokens and KV pairs. Specifically, we introduce two instantiations of our proposed framework: (i) SpecKV, which leverages a draft output to accurately assess the importance of each KV pair for more effective KV cache dropping, and (ii) SpecPC, which uses the draft model's attention activations to identify and discard unimportant prompt tokens. To the best of our knowledge, this is the first work to use draft models for approximate LLM inference acceleration, extending their utility beyond traditional lossless speculative decoding. We motivate our methods with theoretical and empirical analyses, and show a strong correlation between the attention patterns of draft and target models. Extensive experiments on long-context benchmarks show that our methods consistently achieve higher accuracy than existing baselines, while preserving the same improvements in memory usage, latency, and throughput. Our code is available at this https URL. 

---
# Large Language Models Have Intrinsic Meta-Cognition, but Need a Good Lens 

**Authors**: Ziyang Ma, Qingyue Yuan, Zhenglin Wang, Deyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.08410)  

**Abstract**: Previous research has primarily focused on the cognitive error detection capabilities of Large Language Models (LLMs), often prompting them to analyze mistakes in reasoning chains. However, few studies have examined the meta-cognitive abilities of LLMs (e.g., their self-awareness of step errors), which are crucial for their reliability. While studies on LLM self-evaluation present some measures, such as perplexity, which can reflect the answer correctness and be viewed as the lens of meta-cognition, they lack step-level analysis and adaptation. This paper studies the evaluation of LLM meta-cognition using the current lenses and how to improve these lenses. Specifically, we propose AutoMeco, an Automated Meta-cognition Evaluation framework for benchmarking the existing lenses. Furthermore, a training-free Markovian Intrinsic Reward Adjustment strategy, MIRA, is proposed to boost current meta-cognition lenses. Experimental results on three mathematical reasoning datasets and three LLMs show the reasonableness of AutoMeco by comparing it with Best-of-N verification. Moreover, the meta-cognition ability of LLMs can be better evaluated using MIRA. 

---
# Mitigating Posterior Salience Attenuation in Long-Context LLMs with Positional Contrastive Decoding 

**Authors**: Zikai Xiao, Ziyang Wang, Wen Ma, Yan Zhang, Wei Shen, Yan Wang, Luqi Gong, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08371)  

**Abstract**: While Large Language Models (LLMs) support long contexts, they struggle with performance degradation within the context window. Current solutions incur prohibitive training costs, leaving statistical behaviors and cost-effective approaches underexplored. From the decoding perspective, we identify the Posterior Salience Attenuation (PSA) phenomenon, where the salience ratio correlates with long-text performance degradation. Notably, despite the attenuation, gold tokens still occupy high-ranking positions in the decoding space. Motivated by it, we propose the training-free Positional Contrastive Decoding (PCD) that contrasts the logits derived from long-aware attention with those from designed local-aware attention, enabling the model to focus on the gains introduced by large-scale short-to-long training. Through the analysis of long-term decay simulation, we demonstrate that PCD effectively alleviates attention score degradation. Experimental results show that PCD achieves state-of-the-art performance on long-context benchmarks. 

---
# EIFBENCH: Extremely Complex Instruction Following Benchmark for Large Language Models 

**Authors**: Tao Zou, Xinghua Zhang, Haiyang Yu, Minzheng Wang, Fei Huang, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.08375)  

**Abstract**: With the development and widespread application of large language models (LLMs), the new paradigm of "Model as Product" is rapidly evolving, and demands higher capabilities to address complex user needs, often requiring precise workflow execution which involves the accurate understanding of multiple tasks. However, existing benchmarks focusing on single-task environments with limited constraints lack the complexity required to fully reflect real-world scenarios. To bridge this gap, we present the Extremely Complex Instruction Following Benchmark (EIFBENCH), meticulously crafted to facilitate a more realistic and robust evaluation of LLMs. EIFBENCH not only includes multi-task scenarios that enable comprehensive assessment across diverse task types concurrently, but also integrates a variety of constraints, replicating complex operational environments. Furthermore, we propose the Segment Policy Optimization (SegPO) algorithm to enhance the LLM's ability to accurately fulfill multi-task workflow. Evaluations on EIFBENCH have unveiled considerable performance discrepancies in existing LLMs when challenged with these extremely complex instructions. This finding underscores the necessity for ongoing optimization to navigate the intricate challenges posed by LLM applications. 

---
# TACTIC: Translation Agents with Cognitive-Theoretic Interactive Collaboration 

**Authors**: Weiya Li, Junjie Chen, Bei Li, Boyang Liu, Zichen Wen, Nuanqiao Shan, Xiaoqian Liu, Anping Liu, Huajie Liu, Youyan Wang, Wujiuge Yin, Hu Song, Bing Huang, Zhiyuan Xia, Jialiang Chen, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08403)  

**Abstract**: Machine translation has long been a central task in natural language processing. With the rapid advancement of large language models (LLMs), there has been remarkable progress in translation quality. However, fully realizing the translation potential of LLMs remains an open challenge. Recent studies have explored multi-agent systems to decompose complex translation tasks into collaborative subtasks, showing initial promise in enhancing translation quality through agent cooperation and specialization. Nevertheless, existing multi-agent translation frameworks largely neglect foundational insights from cognitive translation studies. These insights emphasize how human translators employ different cognitive strategies, such as balancing literal and free translation, refining expressions based on context, and iteratively evaluating outputs. To address this limitation, we propose a cognitively informed multi-agent framework called TACTIC, which stands for T ranslation A gents with Cognitive- T heoretic Interactive Collaboration. The framework comprises six functionally distinct agents that mirror key cognitive processes observed in human translation behavior. These include agents for drafting, refinement, evaluation, scoring, context reasoning, and external knowledge gathering. By simulating an interactive and theory-grounded translation workflow, TACTIC effectively leverages the full capacity of LLMs for high-quality translation. Experimental results on diverse language pairs from the FLORES-200 and WMT24 benchmarks show that our method consistently achieves state-of-the-art performance. Using DeepSeek-V3 as the base model, TACTIC surpasses GPT-4.1 by an average of +0.6 XCOMET and +1.18 COMETKIWI-23. Compared to DeepSeek-R1, it further improves by +0.84 XCOMET and +2.99 COMETKIWI-23. Code is available at this https URL. 

---
# Can AI Validate Science? Benchmarking LLMs for Accurate Scientific Claim $\rightarrow$ Evidence Reasoning 

**Authors**: Shashidhar Reddy Javaji, Yupeng Cao, Haohang Li, Yangyang Yu, Nikhil Muralidhar, Zining Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08235)  

**Abstract**: Large language models (LLMs) are increasingly being used for complex research tasks such as literature review, idea generation, and scientific paper analysis, yet their ability to truly understand and process the intricate relationships within complex research papers, such as the logical links between claims and supporting evidence remains largely unexplored. In this study, we present CLAIM-BENCH, a comprehensive benchmark for evaluating LLMs' capabilities in scientific claim-evidence extraction and validation, a task that reflects deeper comprehension of scientific argumentation. We systematically compare three approaches which are inspired by divide and conquer approaches, across six diverse LLMs, highlighting model-specific strengths and weaknesses in scientific comprehension. Through evaluation involving over 300 claim-evidence pairs across multiple research domains, we reveal significant limitations in LLMs' ability to process complex scientific content. Our results demonstrate that closed-source models like GPT-4 and Claude consistently outperform open-source counterparts in precision and recall across claim-evidence identification tasks. Furthermore, strategically designed three-pass and one-by-one prompting approaches significantly improve LLMs' abilities to accurately link dispersed evidence with claims, although this comes at increased computational cost. CLAIM-BENCH sets a new standard for evaluating scientific comprehension in LLMs, offering both a diagnostic tool and a path forward for building systems capable of deeper, more reliable reasoning across full-length papers. 

---
# mSTEB: Massively Multilingual Evaluation of LLMs on Speech and Text Tasks 

**Authors**: Luel Hagos Beyene, Vivek Verma, Min Ma, Jesujoba O. Alabi, Fabian David Schmidt, Joyce Nakatumba-Nabende, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2506.08400)  

**Abstract**: Large Language models (LLMs) have demonstrated impressive performance on a wide range of tasks, including in multimodal settings such as speech. However, their evaluation is often limited to English and a few high-resource languages. For low-resource languages, there is no standardized evaluation benchmark. In this paper, we address this gap by introducing mSTEB, a new benchmark to evaluate the performance of LLMs on a wide range of tasks covering language identification, text classification, question answering, and translation tasks on both speech and text modalities. We evaluated the performance of leading LLMs such as Gemini 2.0 Flash and GPT-4o (Audio) and state-of-the-art open models such as Qwen 2 Audio and Gemma 3 27B. Our evaluation shows a wide gap in performance between high-resource and low-resource languages, especially for languages spoken in Africa and Americas/Oceania. Our findings show that more investment is needed to address their under-representation in LLMs coverage. 

---
# CC-RAG: Structured Multi-Hop Reasoning via Theme-Based Causal Graphs 

**Authors**: Jash Rajesh Parekh, Pengcheng Jiang, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.08364)  

**Abstract**: Understanding cause and effect relationships remains a formidable challenge for Large Language Models (LLMs), particularly in specialized domains where reasoning requires more than surface-level correlations. Retrieval-Augmented Generation (RAG) improves factual accuracy, but standard RAG pipelines treat evidence as flat context, lacking the structure required to model true causal dependencies. We introduce Causal-Chain RAG (CC-RAG), a novel approach that integrates zero-shot triple extraction and theme-aware graph chaining into the RAG pipeline, enabling structured multi-hop inference. Given a domain specific corpus, CC-RAG constructs a Directed Acyclic Graph (DAG) of <cause, relation, effect> triples and uses forward/backward chaining to guide structured answer generation. Experiments on two real-world domains: Bitcoin price fluctuations and Gaucher disease, show that CC-RAG outperforms standard RAG and zero-shot LLMs in chain similarity, information density, and lexical diversity. Both LLM-as-a-Judge and human evaluations consistently favor CC-RAG. Our results demonstrate that explicitly modeling causal structure enables LLMs to generate more accurate and interpretable responses, especially in specialized domains where flat retrieval fails. 

---
# Evaluating LLMs Across Multi-Cognitive Levels: From Medical Knowledge Mastery to Scenario-Based Problem Solving 

**Authors**: Yuxuan Zhou, Xien Liu, Chenwei Yan, Chen Ning, Xiao Zhang, Boxun Li, Xiangling Fu, Shijin Wang, Guoping Hu, Yu Wang, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08349)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance on various medical benchmarks, but their capabilities across different cognitive levels remain underexplored. Inspired by Bloom's Taxonomy, we propose a multi-cognitive-level evaluation framework for assessing LLMs in the medical domain in this study. The framework integrates existing medical datasets and introduces tasks targeting three cognitive levels: preliminary knowledge grasp, comprehensive knowledge application, and scenario-based problem solving. Using this framework, we systematically evaluate state-of-the-art general and medical LLMs from six prominent families: Llama, Qwen, Gemma, Phi, GPT, and DeepSeek. Our findings reveal a significant performance decline as cognitive complexity increases across evaluated models, with model size playing a more critical role in performance at higher cognitive levels. Our study highlights the need to enhance LLMs' medical capabilities at higher cognitive levels and provides insights for developing LLMs suited to real-world medical applications. 

---
# Unable to forget: Proactive lnterference Reveals Working Memory Limits in LLMs Beyond Context Length 

**Authors**: Chupei Wang, Jiaqiu Vince Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.08184)  

**Abstract**: Information retrieval in Large Language Models (LLMs) is increasingly recognized as intertwined with generation capabilities rather than mere lookup. While longer contexts are often assumed to improve retrieval, the effects of intra-context interference remain understudied. To address this, we adapt the proactive interference (PI) paradigm from cognitive science, where earlier information disrupts recall of newer updates. In humans, susceptibility to such interference is inversely linked to working memory capacity. We introduce PI-LLM, an evaluation that sequentially streams semantically related key-value updates and queries only the final values. Although these final values are clearly positioned just before the query, LLM retrieval accuracy declines log-linearly toward zero as interference accumulates; errors arise from retrieving previously overwritten values. Attempts to mitigate interference via prompt engineering (e.g., instructing models to ignore earlier input) yield limited success. These findings reveal a fundamental constraint on LLMs' ability to disentangle interference and flexibly manipulate information, suggesting a working memory bottleneck beyond mere context access. This calls for approaches that strengthen models' ability to suppress irrelevant content during retrieval. 

---
# Can Artificial Intelligence Write Like Borges? An Evaluation Protocol for Spanish Microfiction 

**Authors**: Gerardo Aleman Manzanarez, Nora de la Cruz Arana, Jorge Garcia Flores, Yobany Garcia Medina, Raul Monroy, Nathalie Pernelle  

**Link**: [PDF](https://arxiv.org/pdf/2506.08172)  

**Abstract**: Automated story writing has been a subject of study for over 60 years. Large language models can generate narratively consistent and linguistically coherent short fiction texts. Despite these advancements, rigorous assessment of such outputs for literary merit - especially concerning aesthetic qualities - has received scant attention. In this paper, we address the challenge of evaluating AI-generated microfictions and argue that this task requires consideration of literary criteria across various aspects of the text, such as thematic coherence, textual clarity, interpretive depth, and aesthetic quality. To facilitate this, we present GrAImes: an evaluation protocol grounded in literary theory, specifically drawing from a literary perspective, to offer an objective framework for assessing AI-generated microfiction. Furthermore, we report the results of our validation of the evaluation protocol, as answered by both literature experts and literary enthusiasts. This protocol will serve as a foundation for evaluating automatically generated microfictions and assessing their literary value. 

---
# DEAL: Disentangling Transformer Head Activations for LLM Steering 

**Authors**: Li-Ming Zhan, Bo Liu, Zexin Lu, Chengqiang Xie, Jiannong Cao, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08359)  

**Abstract**: Inference-time steering aims to alter the response characteristics of large language models (LLMs) without modifying their underlying parameters. A critical step in this process is the identification of internal modules within LLMs that are associated with the target behavior. However, current approaches to module selection often depend on superficial cues or ad-hoc heuristics, which can result in suboptimal or unintended outcomes. In this work, we propose a principled causal-attribution framework for identifying behavior-relevant attention heads in transformers. For each head, we train a vector-quantized autoencoder (VQ-AE) on its attention activations, partitioning the latent space into behavior-relevant and behavior-irrelevant subspaces, each quantized with a shared learnable codebook. We assess the behavioral relevance of each head by quantifying the separability of VQ-AE encodings for behavior-aligned versus behavior-violating responses using a binary classification metric. This yields a behavioral relevance score that reflects each head discriminative capacity with respect to the target behavior, guiding both selection and importance weighting. Experiments on seven LLMs from two model families and five behavioral steering datasets demonstrate that our method enables more accurate inference-time interventions, achieving superior performance on the truthfulness-steering task. Furthermore, the heads selected by our approach exhibit strong zero-shot generalization in cross-domain truthfulness-steering scenarios. 

---
# Multilingual Hate Speech Detection in Social Media Using Translation-Based Approaches with Large Language Models 

**Authors**: Muhammad Usman, Muhammad Ahmad, M. Shahiki Tash, Irina Gelbukh, Rolando Quintero Tellez, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2506.08147)  

**Abstract**: Social media platforms are critical spaces for public discourse, shaping opinions and community dynamics, yet their widespread use has amplified harmful content, particularly hate speech, threatening online safety and inclusivity. While hate speech detection has been extensively studied in languages like English and Spanish, Urdu remains underexplored, especially using translation-based approaches. To address this gap, we introduce a trilingual dataset of 10,193 tweets in English (3,834 samples), Urdu (3,197 samples), and Spanish (3,162 samples), collected via keyword filtering, with a balanced distribution of 4,849 Hateful and 5,344 Not-Hateful labels. Our methodology leverages attention layers as a precursor to transformer-based models and large language models (LLMs), enhancing feature extraction for multilingual hate speech detection. For non-transformer models, we use TF-IDF for feature extraction. The dataset is benchmarked using state-of-the-art models, including GPT-3.5 Turbo and Qwen 2.5 72B, alongside traditional machine learning models like SVM and other transformers (e.g., BERT, RoBERTa). Three annotators, following rigorous guidelines, ensured high dataset quality, achieving a Fleiss' Kappa of 0.821. Our approach, integrating attention layers with GPT-3.5 Turbo and Qwen 2.5 72B, achieves strong performance, with macro F1 scores of 0.87 for English (GPT-3.5 Turbo), 0.85 for Spanish (GPT-3.5 Turbo), 0.81 for Urdu (Qwen 2.5 72B), and 0.88 for the joint multilingual model (Qwen 2.5 72B). These results reflect improvements of 8.75% in English (over SVM baseline 0.80), 8.97% in Spanish (over SVM baseline 0.78), 5.19% in Urdu (over SVM baseline 0.77), and 7.32% in the joint multilingual model (over SVM baseline 0.82). Our framework offers a robust solution for multilingual hate speech detection, fostering safer digital communities worldwide. 

---
# LLM-BT: Back-Translation as a Framework for Terminology Standardization and Dynamic Semantic Embedding 

**Authors**: Li Weigang, Pedro Carvalho Brom  

**Link**: [PDF](https://arxiv.org/pdf/2506.08174)  

**Abstract**: The rapid growth of English technical terms challenges traditional expert-driven standardization, especially in fast-evolving fields like AI and quantum computing. Manual methods struggle to ensure multilingual consistency. We propose \textbf{LLM-BT}, a back-translation framework powered by large language models (LLMs) to automate terminology verification and standardization via cross-lingual semantic alignment. Our contributions are: \textbf{(1) Term-Level Consistency Validation:} Using English $\rightarrow$ intermediate language $\rightarrow$ English back-translation, LLM-BT achieves high term consistency across models (e.g., GPT-4, DeepSeek, Grok), with case studies showing over 90\% exact or semantic matches. \textbf{(2) Multi-Path Verification Workflow:} A novel ``Retrieve--Generate--Verify--Optimize'' pipeline integrates serial (e.g., EN $\rightarrow$ ZHcn $\rightarrow$ ZHtw $\rightarrow$ EN) and parallel (e.g., EN $\rightarrow$ Chinese/Portuguese $\rightarrow$ EN) BT routes. BLEU and term accuracy indicate strong cross-lingual robustness (BLEU $>$ 0.45; Portuguese accuracy 100\%). \textbf{(3) Back-Translation as Semantic Embedding:} BT is conceptualized as dynamic semantic embedding, revealing latent meaning trajectories. Unlike static embeddings, LLM-BT provides transparent path-based embeddings shaped by model evolution. LLM-BT transforms back-translation into an active engine for multilingual terminology standardization, enabling human--AI collaboration: machines ensure semantic fidelity, humans guide cultural interpretation. This infrastructure supports terminology governance across scientific and technological fields worldwide. 

---
# "I Wrote, I Paused, I Rewrote" Teaching LLMs to Read Between the Lines of Student Writing 

**Authors**: Samra Zafar, Shaheer Minhas, Syed Ali Hassan Zaidi, Arfa Naeem, Zahra Ali  

**Link**: [PDF](https://arxiv.org/pdf/2506.08221)  

**Abstract**: Large language models(LLMs) like Gemini are becoming common tools for supporting student writing. But most of their feedback is based only on the final essay missing important context about how that text was written. In this paper, we explore whether using writing process data, collected through keystroke logging and periodic snapshots, can help LLMs give feedback that better reflects how learners think and revise while writing. We built a digital writing tool that captures both what students type and how their essays evolve over time. Twenty students used this tool to write timed essays, which were then evaluated in two ways: (i) LLM generated feedback using both the final essay and the full writing trace, and (ii) After the task, students completed surveys about how useful and relatable they found the feedback. Early results show that learners preferred the process-aware LLM feedback, finding it more in tune with their own thinking. We also found that certain types of edits, like adding new content or reorganizing paragraphs, aligned closely with higher scores in areas like coherence and elaboration. Our findings suggest that making LLMs more aware of the writing process can lead to feedback that feels more meaningful, personal, and supportive. 

---
# EconWebArena: Benchmarking Autonomous Agents on Economic Tasks in Realistic Web Environments 

**Authors**: Zefang Liu, Yinzhu Quan  

**Link**: [PDF](https://arxiv.org/pdf/2506.08136)  

**Abstract**: We introduce EconWebArena, a benchmark for evaluating autonomous agents on complex, multimodal economic tasks in realistic web environments. The benchmark comprises 360 curated tasks from 82 authoritative websites spanning domains such as macroeconomics, labor, finance, trade, and public policy. Each task challenges agents to navigate live websites, interpret structured and visual content, interact with real interfaces, and extract precise, time-sensitive data through multi-step workflows. We construct the benchmark by prompting multiple large language models (LLMs) to generate candidate tasks, followed by rigorous human curation to ensure clarity, feasibility, and source reliability. Unlike prior work, EconWebArena emphasizes fidelity to authoritative data sources and the need for grounded web-based economic reasoning. We evaluate a diverse set of state-of-the-art multimodal LLMs as web agents, analyze failure cases, and conduct ablation studies to assess the impact of visual grounding, plan-based reasoning, and interaction design. Our results reveal substantial performance gaps and highlight persistent challenges in grounding, navigation, and multimodal understanding, positioning EconWebArena as a rigorous testbed for economic web intelligence. 

---
# Wait, We Don't Need to "Wait"! Removing Thinking Tokens Improves Reasoning Efficiency 

**Authors**: Chenlong Wang, Yuanning Feng, Dongping Chen, Zhaoyang Chu, Ranjay Krishna, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.08343)  

**Abstract**: Recent advances in large reasoning models have enabled complex, step-by-step reasoning but often introduce significant overthinking, resulting in verbose and redundant outputs that hinder efficiency. In this study, we examine whether explicit self-reflection, signaled by tokens such as "Wait" and "Hmm", is necessary for advanced reasoning. We propose NoWait, a simple yet effective approach that disables explicit self-reflection by suppressing these tokens during inference. Extensive experiments on ten benchmarks across textual, visual, and video reasoning tasks show that NoWait reduces chain-of-thought trajectory length by up to 27%-51% in five R1-style model series, without compromising model utility. NoWait thus offers a plug-and-play solution for efficient and utility-preserving multimodal reasoning. 

---
# Automatic Generation of Inference Making Questions for Reading Comprehension Assessments 

**Authors**: Wanjing Anya Ma, Michael Flor, Zuowei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08260)  

**Abstract**: Inference making is an essential but complex skill in reading comprehension (RC). Some inferences require resolving references across sentences, and some rely on using prior knowledge to fill in the detail that is not explicitly written in the text. Diagnostic RC questions can help educators provide more effective and targeted reading instruction and interventions for school-age students. We introduce a taxonomy of inference types for RC and use it to analyze the distribution of items within a diagnostic RC item bank. Next, we present experiments using GPT-4o to generate bridging-inference RC items for given reading passages via few-shot prompting, comparing conditions with and without chain-of-thought prompts. Generated items were evaluated on three aspects: overall item quality, appropriate inference type, and LLM reasoning, achieving high inter-rater agreements above 0.90. Our results show that GPT-4o produced 93.8% good-quality questions suitable for operational use in grade 3-12 contexts; however, only 42.6% of the generated questions accurately matched the targeted inference type. We conclude that combining automatic item generation with human judgment offers a promising path toward scalable, high-quality diagnostic RC assessments. 

---
# Conservative Bias in Large Language Models: Measuring Relation Predictions 

**Authors**: Toyin Aguda, Erik Wilson, Allan Anzagira, Simerjot Kaur, Charese Smiley  

**Link**: [PDF](https://arxiv.org/pdf/2506.08120)  

**Abstract**: Large language models (LLMs) exhibit pronounced conservative bias in relation extraction tasks, frequently defaulting to No_Relation label when an appropriate option is unavailable. While this behavior helps prevent incorrect relation assignments, our analysis reveals that it also leads to significant information loss when reasoning is not explicitly included in the output. We systematically evaluate this trade-off across multiple prompts, datasets, and relation types, introducing the concept of Hobson's choice to capture scenarios where models opt for safe but uninformative labels over hallucinated ones. Our findings suggest that conservative bias occurs twice as often as hallucination. To quantify this effect, we use SBERT and LLM prompts to capture the semantic similarity between conservative bias behaviors in constrained prompts and labels generated from semi-constrained and open-ended prompts. 

---
# Paths to Causality: Finding Informative Subgraphs Within Knowledge Graphs for Knowledge-Based Causal Discovery 

**Authors**: Yuni Susanti, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2506.08771)  

**Abstract**: Inferring causal relationships between variable pairs is crucial for understanding multivariate interactions in complex systems. Knowledge-based causal discovery -- which involves inferring causal relationships by reasoning over the metadata of variables (e.g., names or textual context) -- offers a compelling alternative to traditional methods that rely on observational data. However, existing methods using Large Language Models (LLMs) often produce unstable and inconsistent results, compromising their reliability for causal inference. To address this, we introduce a novel approach that integrates Knowledge Graphs (KGs) with LLMs to enhance knowledge-based causal discovery. Our approach identifies informative metapath-based subgraphs within KGs and further refines the selection of these subgraphs using Learning-to-Rank-based models. The top-ranked subgraphs are then incorporated into zero-shot prompts, improving the effectiveness of LLMs in inferring the causal relationship. Extensive experiments on biomedical and open-domain datasets demonstrate that our method outperforms most baselines by up to 44.4 points in F1 scores, evaluated across diverse LLMs and KGs. Our code and datasets are available on GitHub: this https URL 

---
# QA-LIGN: Aligning LLMs through Constitutionally Decomposed QA 

**Authors**: Jacob Dineen, Aswin RRV, Qin Liu, Zhikun Xu, Xiao Ye, Ming Shen, Zhaonan Li, Shijie Lu, Chitta Baral, Muhao Chen, Ben Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.08123)  

**Abstract**: Alignment of large language models with explicit principles (such as helpfulness, honesty, and harmlessness) is crucial for ensuring safe and reliable AI systems. However, standard reward-based alignment methods typically collapse diverse feedback into a single scalar reward, entangling multiple objectives into one opaque training signal, which hinders interpretability. In this work, we introduce QA-LIGN, an automatic symbolic reward decomposition approach that preserves the structure of each constitutional principle within the reward mechanism. Instead of training a black-box reward model that outputs a monolithic score, QA-LIGN formulates principle-specific evaluation questions and derives separate reward components for each principle, making it a drop-in reward model replacement. Experiments aligning an uncensored large language model with a set of constitutional principles demonstrate that QA-LIGN offers greater transparency and adaptability in the alignment process. At the same time, our approach achieves performance on par with or better than a DPO baseline. Overall, these results represent a step toward more interpretable and controllable alignment of language models, achieved without sacrificing end-task performance. 

---
# Approaching Dialogue State Tracking via Aligning Speech Encoders and LLMs 

**Authors**: Šimon Sedláček, Bolaji Yusuf, Ján Švec, Pradyoth Hegde, Santosh Kesiraju, Oldřich Plchot, Jan Černocký  

**Link**: [PDF](https://arxiv.org/pdf/2506.08633)  

**Abstract**: In this work, we approach spoken Dialogue State Tracking (DST) by bridging the representation spaces of speech encoders and LLMs via a small connector module, with a focus on fully open-sourced and open-data components (WavLM-large, OLMo). We focus on ablating different aspects of such systems including full/LoRA adapter fine-tuning, the effect of agent turns in the dialogue history, as well as fuzzy matching-based output post-processing, which greatly improves performance of our systems on named entities in the dialogue slot values. We conduct our experiments on the SpokenWOZ dataset, and additionally utilize the Speech-Aware MultiWOZ dataset to augment our training data. Ultimately, our best-performing WavLM + connector + OLMo-1B aligned models achieve state of the art on the SpokenWOZ test set (34.66% JGA), and our system with Gemma-2-9B-instruct further surpasses this result, reaching 42.17% JGA on SpokenWOZ test. 

---
# e3: Learning to Explore Enables Extrapolation of Test-Time Compute for LLMs 

**Authors**: Amrith Setlur, Matthew Y. R. Yang, Charlie Snell, Jeremy Greer, Ian Wu, Virginia Smith, Max Simchowitz, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.09026)  

**Abstract**: Test-time scaling offers a promising path to improve LLM reasoning by utilizing more compute at inference time; however, the true promise of this paradigm lies in extrapolation (i.e., improvement in performance on hard problems as LLMs keep "thinking" for longer, beyond the maximum token budget they were trained on). Surprisingly, we find that most existing reasoning models do not extrapolate well. We show that one way to enable extrapolation is by training the LLM to perform in-context exploration: training the LLM to effectively spend its test time budget by chaining operations (such as generation, verification, refinement, etc.), or testing multiple hypotheses before it commits to an answer. To enable in-context exploration, we identify three key ingredients as part of our recipe e3: (1) chaining skills that the base LLM has asymmetric competence in, e.g., chaining verification (easy) with generation (hard), as a way to implement in-context search; (2) leveraging "negative" gradients from incorrect traces to amplify exploration during RL, resulting in longer search traces that chains additional asymmetries; and (3) coupling task difficulty with training token budget during training via a specifically-designed curriculum to structure in-context exploration. Our recipe e3 produces the best known 1.7B model according to AIME'25 and HMMT'25 scores, and extrapolates to 2x the training token budget. Our e3-1.7B model not only attains high pass@1 scores, but also improves pass@k over the base model. 

---
# EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements 

**Authors**: Issa Sugiura, Takashi Ishida, Taro Makino, Chieko Tazuke, Takanori Nakagawa, Kosuke Nakago, David Ha  

**Link**: [PDF](https://arxiv.org/pdf/2506.08762)  

**Abstract**: Financial analysis presents complex challenges that could leverage large language model (LLM) capabilities. However, the scarcity of challenging financial datasets, particularly for Japanese financial data, impedes academic innovation in financial analytics. As LLMs advance, this lack of accessible research resources increasingly hinders their development and evaluation in this specialized domain. To address this gap, we introduce EDINET-Bench, an open-source Japanese financial benchmark designed to evaluate the performance of LLMs on challenging financial tasks including accounting fraud detection, earnings forecasting, and industry prediction. EDINET-Bench is constructed by downloading annual reports from the past 10 years from Japan's Electronic Disclosure for Investors' NETwork (EDINET) and automatically assigning labels corresponding to each evaluation task. Our experiments reveal that even state-of-the-art LLMs struggle, performing only slightly better than logistic regression in binary classification for fraud detection and earnings forecasting. These results highlight significant challenges in applying LLMs to real-world financial applications and underscore the need for domain-specific adaptation. Our dataset, benchmark construction code, and evaluation code is publicly available to facilitate future research in finance with LLMs. 

---
# The Geometries of Truth Are Orthogonal Across Tasks 

**Authors**: Waiss Azizian, Michael Kirchhof, Eugene Ndiaye, Louis Bethune, Michal Klein, Pierre Ablin, Marco Cuturi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08572)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive generalization capabilities across various tasks, but their claim to practical relevance is still mired by concerns on their reliability. Recent works have proposed examining the activations produced by an LLM at inference time to assess whether its answer to a question is correct. Some works claim that a "geometry of truth" can be learned from examples, in the sense that the activations that generate correct answers can be distinguished from those leading to mistakes with a linear classifier. In this work, we underline a limitation of these approaches: we observe that these "geometries of truth" are intrinsically task-dependent and fail to transfer across tasks. More precisely, we show that linear classifiers trained across distinct tasks share little similarity and, when trained with sparsity-enforcing regularizers, have almost disjoint supports. We show that more sophisticated approaches (e.g., using mixtures of probes and tasks) fail to overcome this limitation, likely because activation vectors commonly used to classify answers form clearly separated clusters when examined across tasks. 

---
# Reinforce LLM Reasoning through Multi-Agent Reflection 

**Authors**: Yurun Yuan, Tengyang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.08379)  

**Abstract**: Leveraging more test-time computation has proven to be an effective way to boost the reasoning capabilities of large language models (LLMs). Among various methods, the verify-and-improve paradigm stands out for enabling dynamic solution exploration and feedback incorporation. However, existing approaches often suffer from restricted feedback spaces and lack of coordinated training of different parties, leading to suboptimal performance. To address this, we model this multi-turn refinement process as a Markov Decision Process and introduce DPSDP (Direct Policy Search by Dynamic Programming), a reinforcement learning algorithm that trains an actor-critic LLM system to iteratively refine answers via direct preference learning on self-generated data. Theoretically, DPSDP can match the performance of any policy within the training distribution. Empirically, we instantiate DPSDP with various base models and show improvements on both in- and out-of-distribution benchmarks. For example, on benchmark MATH 500, majority voting over five refinement steps increases first-turn accuracy from 58.2% to 63.2% with Ministral-based models. An ablation study further confirms the benefits of multi-agent collaboration and out-of-distribution generalization. 

---
# Consistent Paths Lead to Truth: Self-Rewarding Reinforcement Learning for LLM Reasoning 

**Authors**: Kongcheng Zhang, Qi Yao, Shunyu Liu, Yingjie Wang, Baisheng Lai, Jieping Ye, Mingli Song, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.08745)  

**Abstract**: Recent advances of Reinforcement Learning (RL) have highlighted its potential in complex reasoning tasks, yet effective training often relies on external supervision, which limits the broader applicability. In this work, we propose a novel self-rewarding reinforcement learning framework to enhance Large Language Model (LLM) reasoning by leveraging the consistency of intermediate reasoning states across different reasoning trajectories. Our key insight is that correct responses often exhibit consistent trajectory patterns in terms of model likelihood: their intermediate reasoning states tend to converge toward their own final answers (high consistency) with minimal deviation toward other candidates (low volatility). Inspired by this observation, we introduce CoVo, an intrinsic reward mechanism that integrates Consistency and Volatility via a robust vector-space aggregation strategy, complemented by a curiosity bonus to promote diverse exploration. CoVo enables LLMs to perform RL in a self-rewarding manner, offering a scalable pathway for learning to reason without external supervision. Extensive experiments on diverse reasoning benchmarks show that CoVo achieves performance comparable to or even surpassing supervised RL. Our code is available at this https URL. 

---
# A Survey on Large Language Models for Mathematical Reasoning 

**Authors**: Peng-Yuan Wang, Tian-Shuo Liu, Chenyang Wang, Yi-Di Wang, Shu Yan, Cheng-Xing Jia, Xu-Hui Liu, Xin-Wei Chen, Jia-Cheng Xu, Ziniu Li, Yang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08446)  

**Abstract**: Mathematical reasoning has long represented one of the most fundamental and challenging frontiers in artificial intelligence research. In recent years, large language models (LLMs) have achieved significant advances in this area. This survey examines the development of mathematical reasoning abilities in LLMs through two high-level cognitive phases: comprehension, where models gain mathematical understanding via diverse pretraining strategies, and answer generation, which has progressed from direct prediction to step-by-step Chain-of-Thought (CoT) reasoning. We review methods for enhancing mathematical reasoning, ranging from training-free prompting to fine-tuning approaches such as supervised fine-tuning and reinforcement learning, and discuss recent work on extended CoT and "test-time scaling". Despite notable progress, fundamental challenges remain in terms of capacity, efficiency, and generalization. To address these issues, we highlight promising research directions, including advanced pretraining and knowledge augmentation techniques, formal reasoning frameworks, and meta-generalization through principled learning paradigms. This survey tries to provide some insights for researchers interested in enhancing reasoning capabilities of LLMs and for those seeking to apply these techniques to other domains. 

---
# A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation 

**Authors**: Andrew Z. Wang, Songwei Ge, Tero Karras, Ming-Yu Liu, Yogesh Balaji  

**Link**: [PDF](https://arxiv.org/pdf/2506.08210)  

**Abstract**: Both text-to-image generation and large language models (LLMs) have made significant advancements. However, many text-to-image models still employ the somewhat outdated T5 and CLIP as their text encoders. In this work, we investigate the effectiveness of using modern decoder-only LLMs as text encoders for text-to-image diffusion models. We build a standardized training and evaluation pipeline that allows us to isolate and evaluate the effect of different text embeddings. We train a total of 27 text-to-image models with 12 different text encoders to analyze the critical aspects of LLMs that could impact text-to-image generation, including the approaches to extract embeddings, different LLMs variants, and model sizes. Our experiments reveal that the de facto way of using last-layer embeddings as conditioning leads to inferior performance. Instead, we explore embeddings from various layers and find that using layer-normalized averaging across all layers significantly improves alignment with complex prompts. Most LLMs with this conditioning outperform the baseline T5 model, showing enhanced performance in advanced visio-linguistic reasoning skills. 

---
# From Passive to Active Reasoning: Can Large Language Models Ask the Right Questions under Incomplete Information? 

**Authors**: Zhanke Zhou, Xiao Feng, Zhaocheng Zhu, Jiangchao Yao, Sanmi Koyejo, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.08295)  

**Abstract**: While existing benchmarks probe the reasoning abilities of large language models (LLMs) across diverse domains, they predominantly assess passive reasoning, providing models with all the information needed to reach a solution. By contrast, active reasoning-where an LLM must interact with external systems to acquire missing evidence or data-has received little systematic attention. To address this shortfall, we present AR-Bench, a novel benchmark designed explicitly to evaluate an LLM's active reasoning skills. AR-Bench comprises three task families-detective cases, situation puzzles, and guessing numbers-that together simulate real-world, agentic scenarios and measure performance across commonsense, logical, and symbolic reasoning challenges. Empirical evaluation on AR-Bench demonstrates that contemporary LLMs exhibit pronounced difficulties with active reasoning: they frequently fail to acquire or leverage the information needed to solve tasks. This gap highlights a stark divergence between their passive and active reasoning abilities. Moreover, ablation studies indicate that even advanced strategies, such as tree-based searching or post-training approaches, yield only modest gains and fall short of the levels required for real-world deployment. Collectively, these findings highlight the critical need to advance methodology for active reasoning, e.g., incorporating interactive learning, real-time feedback loops, and environment-aware objectives for training. The benchmark is publicly available at: this https URL. 

---
# From Debate to Equilibrium: Belief-Driven Multi-Agent LLM Reasoning via Bayesian Nash Equilibrium 

**Authors**: Xie Yi, Zhanke Zhou, Chentao Cao, Qiyu Niu, Tongliang Liu, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.08292)  

**Abstract**: Multi-agent frameworks can substantially boost the reasoning power of large language models (LLMs), but they typically incur heavy computational costs and lack convergence guarantees. To overcome these challenges, we recast multi-LLM coordination as an incomplete-information game and seek a Bayesian Nash equilibrium (BNE), in which each agent optimally responds to its probabilistic beliefs about the strategies of others. We introduce Efficient Coordination via Nash Equilibrium (ECON), a hierarchical reinforcement-learning paradigm that marries distributed reasoning with centralized final output. Under ECON, each LLM independently selects responses that maximize its expected reward, conditioned on its beliefs about co-agents, without requiring costly inter-agent exchanges. We mathematically prove that ECON attains a markedly tighter regret bound than non-equilibrium multi-agent schemes. Empirically, ECON outperforms existing multi-LLM approaches by 11.2% on average across six benchmarks spanning complex reasoning and planning tasks. Further experiments demonstrate ECON's ability to flexibly incorporate additional models, confirming its scalability and paving the way toward larger, more powerful multi-LLM ensembles. The code is publicly available at: this https URL. 

---
# Bingo: Boosting Efficient Reasoning of LLMs via Dynamic and Significance-based Reinforcement Learning 

**Authors**: Hanbing Liu, Lang Cao, Yuanyi Ren, Mengyu Zhou, Haoyu Dong, Xiaojun Ma, Shi Han, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08125)  

**Abstract**: Large language models have demonstrated impressive reasoning capabilities, yet they often suffer from inefficiencies due to unnecessarily verbose or redundant outputs. While many works have explored reinforcement learning (RL) to enhance reasoning abilities, most primarily focus on improving accuracy, with limited attention to reasoning efficiency. Some existing approaches introduce direct length-based rewards to encourage brevity, but this often leads to noticeable drops in accuracy. In this paper, we propose Bingo, an RL framework that advances length-based reward design to boost efficient reasoning. Bingo incorporates two key mechanisms: a significance-aware length reward, which gradually guides the model to reduce only insignificant tokens, and a dynamic length reward, which initially encourages elaborate reasoning for hard questions but decays over time to improve overall efficiency. Experiments across multiple reasoning benchmarks show that Bingo improves both accuracy and efficiency. It outperforms the vanilla reward and several other length-based reward baselines in RL, achieving a favorable trade-off between accuracy and efficiency. These results underscore the potential of training LLMs explicitly for efficient reasoning. 

---
# AutoSDT: Scaling Data-Driven Discovery Tasks Toward Open Co-Scientists 

**Authors**: Yifei Li, Hanane Nour Moussa, Ziru Chen, Shijie Chen, Botao Yu, Mingyi Xue, Benjamin Burns, Tzu-Yao Chiu, Vishal Dey, Zitong Lu, Chen Wei, Qianheng Zhang, Tianyu Zhang, Song Gao, Xuhui Huang, Xia Ning, Nesreen K. Ahmed, Ali Payani, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.08140)  

**Abstract**: Despite long-standing efforts in accelerating scientific discovery with AI, building AI co-scientists remains challenging due to limited high-quality data for training and evaluation. To tackle this data scarcity issue, we present AutoSDT, an automatic pipeline that collects high-quality coding tasks in real-world data-driven discovery workflows. AutoSDT leverages the coding capabilities and parametric knowledge of LLMs to search for diverse sources, select ecologically valid tasks, and synthesize accurate task instructions and code solutions. Using our pipeline, we construct AutoSDT-5K, a dataset of 5,404 coding tasks for data-driven discovery that covers four scientific disciplines and 756 unique Python packages. To the best of our knowledge, AutoSDT-5K is the only automatically collected and the largest open dataset for data-driven scientific discovery. Expert feedback on a subset of 256 tasks shows the effectiveness of AutoSDT: 93% of the collected tasks are ecologically valid, and 92.2% of the synthesized programs are functionally correct. Trained on AutoSDT-5K, the Qwen2.5-Coder-Instruct LLM series, dubbed AutoSDT-Coder, show substantial improvement on two challenging data-driven discovery benchmarks, ScienceAgentBench and DiscoveryBench. Most notably, AutoSDT-Coder-32B reaches the same level of performance as GPT-4o on ScienceAgentBench with a success rate of 7.8%, doubling the performance of its base model. On DiscoveryBench, it lifts the hypothesis matching score to 8.1, bringing a 17.4% relative improvement and closing the gap between open-weight models and GPT-4o. 

---
# Leveraging LLMs to Evaluate Usefulness of Document 

**Authors**: Xingzhu Wang, Erhan Zhang, Yiqun Chen, Jinghan Xuan, Yucheng Hou, Yitong Xu, Ying Nie, Shuaiqiang Wang, Dawei Yin, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2506.08626)  

**Abstract**: The conventional Cranfield paradigm struggles to effectively capture user satisfaction due to its weak correlation between relevance and satisfaction, alongside the high costs of relevance annotation in building test collections. To tackle these issues, our research explores the potential of leveraging large language models (LLMs) to generate multilevel usefulness labels for evaluation. We introduce a new user-centric evaluation framework that integrates users' search context and behavioral data into LLMs. This framework uses a cascading judgment structure designed for multilevel usefulness assessments, drawing inspiration from ordinal regression techniques. Our study demonstrates that when well-guided with context and behavioral information, LLMs can accurately evaluate usefulness, allowing our approach to surpass third-party labeling methods. Furthermore, we conduct ablation studies to investigate the influence of key components within the framework. We also apply the labels produced by our method to predict user satisfaction, with real-world experiments indicating that these labels substantially improve the performance of satisfaction prediction models. 

---
# No Stupid Questions: An Analysis of Question Query Generation for Citation Recommendation 

**Authors**: Brian D. Zimmerman, Julien Aubert-Béduchaud, Florian Boudin, Akiko Aizawa, Olga Vechtomova  

**Link**: [PDF](https://arxiv.org/pdf/2506.08196)  

**Abstract**: Existing techniques for citation recommendation are constrained by their adherence to article contents and metadata. We leverage GPT-4o-mini's latent expertise as an inquisitive assistant by instructing it to ask questions which, when answered, could expose new insights about an excerpt from a scientific article. We evaluate the utility of these questions as retrieval queries, measuring their effectiveness in retrieving and ranking masked target documents. In some cases, generated questions ended up being better queries than extractive keyword queries generated by the same model. We additionally propose MMR-RBO, a variation of Maximal Marginal Relevance (MMR) using Rank-Biased Overlap (RBO) to identify which questions will perform competitively with the keyword baseline. As all question queries yield unique result sets, we contend that there are no stupid questions. 

---
# Serendipitous Recommendation with Multimodal LLM 

**Authors**: Haoting Wang, Jianling Wang, Hao Li, Fangjun Yi, Mengyu Fu, Youwei Zhang, Yifan Liu, Liang Liu, Minmin Chen, Ed H. Chi, Lichan Hong, Haokai Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08283)  

**Abstract**: Conventional recommendation systems succeed in identifying relevant content but often fail to provide users with surprising or novel items. Multimodal Large Language Models (MLLMs) possess the world knowledge and multimodal understanding needed for serendipity, but their integration into billion-item-scale platforms presents significant challenges. In this paper, we propose a novel hierarchical framework where fine-tuned MLLMs provide high-level guidance to conventional recommendation models, steering them towards more serendipitous suggestions. This approach leverages MLLM strengths in understanding multimodal content and user interests while retaining the efficiency of traditional models for item-level recommendation. This mitigates the complexity of applying MLLMs directly to vast action spaces. We also demonstrate a chain-of-thought strategy enabling MLLMs to discover novel user interests by first understanding video content and then identifying relevant yet unexplored interest clusters. Through live experiments within a commercial short-form video platform serving billions of users, we show that our MLLM-powered approach significantly improves both recommendation serendipity and user satisfaction. 

---
# Reinforcement Fine-Tuning for Reasoning towards Multi-Step Multi-Source Search in Large Language Models 

**Authors**: Wentao Shi, Yiqing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.08352)  

**Abstract**: Large language models (LLMs) can face factual limitations when responding to time-sensitive queries about recent events that arise after their knowledge thresholds in the training corpus. Existing search-augmented approaches fall into two categories, each with distinct limitations: multi-agent search frameworks incur substantial computational overhead by separating search planning and response synthesis across multiple LLMs, while single-LLM tool-calling methods restrict themselves to sequential planned, single-query searches from sole search sources. We present Reasoning-Search (R-Search), a single-LLM search framework that unifies multi-step planning, multi-source search execution, and answer synthesis within one coherent inference process. Innovatively, it structure the output into four explicitly defined components, including reasoning steps that guide the search process (<think>), a natural-language directed acyclic graph that represents the search plans with respect to diverse sources (<search>), retrieved results from executing the search plans (<result>), and synthesized final answers (<answer>). To enable effective generation of these structured outputs, we propose a specialized Reinforcement Fine-Tuning (ReFT) method based on GRPO, together with a multi-component reward function that optimizes LLM's answer correctness, structural validity of the generated DAG, and adherence to the defined output format. Experimental evaluation on FinSearchBench-24, SearchExpertBench-25, and seven Q and A benchmarks demonstrates that R-Search outperforms state-of-the-art methods, while achieving substantial efficiency gains through 70% reduction in context token usage and approximately 50% decrease in execution latency. Code is available at this https URL. 

---
# Your Brain on ChatGPT: Accumulation of Cognitive Debt when Using an AI Assistant for Essay Writing Task 

**Authors**: Nataliya Kosmyna, Eugene Hauptmann, Ye Tong Yuan, Jessica Situ, Xian-Hao Liao, Ashly Vivian Beresnitzky, Iris Braunstein, Pattie Maes  

**Link**: [PDF](https://arxiv.org/pdf/2506.08872)  

**Abstract**: This study explores the neural and behavioral consequences of LLM-assisted essay writing. Participants were divided into three groups: LLM, Search Engine, and Brain-only (no tools). Each completed three sessions under the same condition. In a fourth session, LLM users were reassigned to Brain-only group (LLM-to-Brain), and Brain-only users were reassigned to LLM condition (Brain-to-LLM). A total of 54 participants took part in Sessions 1-3, with 18 completing session 4. We used electroencephalography (EEG) to assess cognitive load during essay writing, and analyzed essays using NLP, as well as scoring essays with the help from human teachers and an AI judge. Across groups, NERs, n-gram patterns, and topic ontology showed within-group homogeneity. EEG revealed significant differences in brain connectivity: Brain-only participants exhibited the strongest, most distributed networks; Search Engine users showed moderate engagement; and LLM users displayed the weakest connectivity. Cognitive activity scaled down in relation to external tool use. In session 4, LLM-to-Brain participants showed reduced alpha and beta connectivity, indicating under-engagement. Brain-to-LLM users exhibited higher memory recall and activation of occipito-parietal and prefrontal areas, similar to Search Engine users. Self-reported ownership of essays was the lowest in the LLM group and the highest in the Brain-only group. LLM users also struggled to accurately quote their own work. While LLMs offer immediate convenience, our findings highlight potential cognitive costs. Over four months, LLM users consistently underperformed at neural, linguistic, and behavioral levels. These results raise concerns about the long-term educational implications of LLM reliance and underscore the need for deeper inquiry into AI's role in learning. 

---
# RHealthTwin: Towards Responsible and Multimodal Digital Twins for Personalized Well-being 

**Authors**: Rahatara Ferdousi, M Anwar Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2506.08486)  

**Abstract**: The rise of large language models (LLMs) has created new possibilities for digital twins in healthcare. However, the deployment of such systems in consumer health contexts raises significant concerns related to hallucination, bias, lack of transparency, and ethical misuse. In response to recommendations from health authorities such as the World Health Organization (WHO), we propose Responsible Health Twin (RHealthTwin), a principled framework for building and governing AI-powered digital twins for well-being assistance. RHealthTwin processes multimodal inputs that guide a health-focused LLM to produce safe, relevant, and explainable responses. At the core of RHealthTwin is the Responsible Prompt Engine (RPE), which addresses the limitations of traditional LLM configuration. Conventionally, users input unstructured prompt and the system instruction to configure the LLM, which increases the risk of hallucination. In contrast, RPE extracts predefined slots dynamically to structure both inputs. This guides the language model to generate responses that are context aware, personalized, fair, reliable, and explainable for well-being assistance. The framework further adapts over time through a feedback loop that updates the prompt structure based on user satisfaction. We evaluate RHealthTwin across four consumer health domains including mental support, symptom triage, nutrition planning, and activity coaching. RPE achieves state-of-the-art results with BLEU = 0.41, ROUGE-L = 0.63, and BERTScore = 0.89 on benchmark datasets. Also, we achieve over 90% in ethical compliance and instruction-following metrics using LLM-as-judge evaluation, outperforming baseline strategies. We envision RHealthTwin as a forward-looking foundation for responsible LLM-based applications in health and well-being. 

---
# Transforming Expert Knowledge into Scalable Ontology via Large Language Models 

**Authors**: Ikkei Itoku, David Theil, Evelyn Eichelsdoerfer Uehara, Sreyoshi Bhaduri, Junnosuke Kuroda, Toshi Yumoto, Alex Gil, Natalie Perez, Rajesh Cherukuri, Naumaan Nayyar  

**Link**: [PDF](https://arxiv.org/pdf/2506.08422)  

**Abstract**: Having a unified, coherent taxonomy is essential for effective knowledge representation in domain-specific applications as diverse terminologies need to be mapped to underlying concepts. Traditional manual approaches to taxonomy alignment rely on expert review of concept pairs, but this becomes prohibitively expensive and time-consuming at scale, while subjective interpretations often lead to expert disagreements. Existing automated methods for taxonomy alignment have shown promise but face limitations in handling nuanced semantic relationships and maintaining consistency across different domains. These approaches often struggle with context-dependent concept mappings and lack transparent reasoning processes. We propose a novel framework that combines large language models (LLMs) with expert calibration and iterative prompt optimization to automate taxonomy alignment. Our method integrates expert-labeled examples, multi-stage prompt engineering, and human validation to guide LLMs in generating both taxonomy linkages and supporting rationales. In evaluating our framework on a domain-specific mapping task of concept essentiality, we achieved an F1-score of 0.97, substantially exceeding the human benchmark of 0.68. These results demonstrate the effectiveness of our approach in scaling taxonomy alignment while maintaining high-quality mappings and preserving expert oversight for ambiguous cases. 

---
# On Reasoning Strength Planning in Large Reasoning Models 

**Authors**: Leheng Sheng, An Zhang, Zijian Wu, Weixiang Zhao, Changshuo Shen, Yi Zhang, Xiang Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.08390)  

**Abstract**: Recent studies empirically reveal that large reasoning models (LRMs) can automatically allocate more reasoning strengths (i.e., the number of reasoning tokens) for harder problems, exhibiting difficulty-awareness for better task performance. While this automatic reasoning strength allocation phenomenon has been widely observed, its underlying mechanism remains largely unexplored. To this end, we provide explanations for this phenomenon from the perspective of model activations. We find evidence that LRMs pre-plan the reasoning strengths in their activations even before generation, with this reasoning strength causally controlled by the magnitude of a pre-allocated directional vector. Specifically, we show that the number of reasoning tokens is predictable solely based on the question activations using linear probes, indicating that LRMs estimate the required reasoning strength in advance. We then uncover that LRMs encode this reasoning strength through a pre-allocated directional vector embedded in the activations of the model, where the vector's magnitude modulates the reasoning strength. Subtracting this vector can lead to reduced reasoning token number and performance, while adding this vector can lead to increased reasoning token number and even improved performance. We further reveal that this direction vector consistently yields positive reasoning length prediction, and it modifies the logits of end-of-reasoning token </think> to affect the reasoning length. Finally, we demonstrate two potential applications of our findings: overthinking behavior detection and enabling efficient reasoning on simple problems. Our work provides new insights into the internal mechanisms of reasoning in LRMs and offers practical tools for controlling their reasoning behaviors. Our code is available at this https URL. 

---
# AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions 

**Authors**: Polina Kirichenko, Mark Ibrahim, Kamalika Chaudhuri, Samuel J. Bell  

**Link**: [PDF](https://arxiv.org/pdf/2506.09038)  

**Abstract**: For Large Language Models (LLMs) to be reliably deployed in both everyday and high-stakes domains, knowing when not to answer is equally critical as answering correctly. Real-world user queries, which can be underspecified, ill-posed, or fundamentally unanswerable, require LLMs to reason about uncertainty and selectively abstain -- i.e., refuse to answer definitively. However, abstention remains understudied, without a systematic evaluation framework for modern LLMs. In this work, we introduce AbstentionBench, a large-scale benchmark for holistically evaluating abstention across 20 diverse datasets, including questions with unknown answers, underspecification, false premises, subjective interpretations, and outdated information. Evaluating 20 frontier LLMs reveals abstention is an unsolved problem, and one where scaling models is of little use. While recent reasoning LLMs have shown impressive results in complex problem solving, surprisingly, we find that reasoning fine-tuning degrades abstention (by $24\%$ on average), even for math and science domains on which reasoning models are explicitly trained. We find that while a carefully crafted system prompt can boost abstention in practice, it does not resolve models' fundamental inability to reason about uncertainty. We release AbstentionBench to foster research into advancing LLM reliability. 

---
# ORFS-agent: Tool-Using Agents for Chip Design Optimization 

**Authors**: Amur Ghose, Andrew B. Kahng, Sayak Kundu, Zhiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08332)  

**Abstract**: Machine learning has been widely used to optimize complex engineering workflows across numerous domains. In the context of integrated circuit design, modern flows (e.g., going from a register-transfer level netlist to physical layouts) involve extensive configuration via thousands of parameters, and small changes to these parameters can have large downstream impacts on desired outcomes - namely design performance, power, and area. Recent advances in Large Language Models (LLMs) offer new opportunities for learning and reasoning within such high-dimensional optimization tasks. In this work, we introduce ORFS-agent, an LLM-based iterative optimization agent that automates parameter tuning in an open-source hardware design flow. ORFS-agent adaptively explores parameter configurations, demonstrating clear improvements over standard Bayesian optimization approaches in terms of resource efficiency and final design metrics. Our empirical evaluations on two different technology nodes and a range of circuit benchmarks indicate that ORFS-agent can improve both routed wirelength and effective clock period by over 13%, all while using 40% fewer optimization iterations. Moreover, by following natural language objectives to trade off certain metrics for others, ORFS-agent demonstrates a flexible and interpretable framework for multi-objective optimization. Crucially, RFS-agent is modular and model-agnostic, and can be plugged in to any frontier LLM without any further fine-tuning. 

---
# Safe and Economical UAV Trajectory Planning in Low-Altitude Airspace: A Hybrid DRL-LLM Approach with Compliance Awareness 

**Authors**: Yanwei Gong, Xiaolin Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08532)  

**Abstract**: The rapid growth of the low-altitude economy has driven the widespread adoption of unmanned aerial vehicles (UAVs). This growing deployment presents new challenges for UAV trajectory planning in complex urban environments. However, existing studies often overlook key factors, such as urban airspace constraints and economic efficiency, which are essential in low-altitude economy contexts. Deep reinforcement learning (DRL) is regarded as a promising solution to these issues, while its practical adoption remains limited by low learning efficiency. To overcome this limitation, we propose a novel UAV trajectory planning framework that combines DRL with large language model (LLM) reasoning to enable safe, compliant, and economically viable path planning. Experimental results demonstrate that our method significantly outperforms existing baselines across multiple metrics, including data collection rate, collision avoidance, successful landing, regulatory compliance, and energy efficiency. These results validate the effectiveness of our approach in addressing UAV trajectory planning key challenges under constraints of the low-altitude economy networking. 

---
# LeanTutor: A Formally-Verified AI Tutor for Mathematical Proofs 

**Authors**: Manooshree Patel, Rayna Bhattacharyya, Thomas Lu, Arnav Mehta, Niels Voss, Narges Norouzi, Gireeja Ranade  

**Link**: [PDF](https://arxiv.org/pdf/2506.08321)  

**Abstract**: We present LeanTutor, a Large Language Model (LLM)-based tutoring system for math proofs. LeanTutor interacts with the student in natural language, formally verifies student-written math proofs in Lean, generates correct next steps, and provides the appropriate instructional guidance. LeanTutor is composed of three modules: (i) an autoformalizer/proof-checker, (ii) a next-step generator, and (iii) a natural language feedback generator. The first module faithfully autoformalizes student proofs into Lean and verifies proof accuracy via successful code compilation. If the proof has an error, the incorrect step is identified. The next-step generator module outputs a valid next Lean tactic for incorrect proofs via LLM-based candidate generation and proof search. The feedback generator module leverages Lean data to produce a pedagogically-motivated natural language hint for the student user. To evaluate our system, we introduce PeanoBench, a human-written dataset derived from the Natural Numbers Game, consisting of 371 Peano Arithmetic proofs, where each natural language proof step is paired with the corresponding logically equivalent tactic in Lean. The Autoformalizer correctly formalizes 57% of tactics in correct proofs and accurately identifies the incorrect step in 30% of incorrect proofs. In generating natural language hints for erroneous proofs, LeanTutor outperforms a simple baseline on accuracy and relevance metrics. 

---
# SOP-Bench: Complex Industrial SOPs for Evaluating LLM Agents 

**Authors**: Subhrangshu Nandi, Arghya Datta, Nikhil Vichare, Indranil Bhattacharya, Huzefa Raja, Jing Xu, Shayan Ray, Giuseppe Carenini, Abhi Srivastava, Aaron Chan, Man Ho Woo, Amar Kandola, Brandon Theresa, Francesco Carbone  

**Link**: [PDF](https://arxiv.org/pdf/2506.08119)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive general-purpose reasoning and problem-solving abilities. However, they struggle with executing complex, long-horizon workflows that demand strict adherence to Standard Operating Procedures (SOPs), a critical requirement for real-world industrial automation. Despite this need, there is a lack of public benchmarks that reflect the complexity, structure, and domain-specific nuances of SOPs. To address this, we present three main contributions. First, we introduce a synthetic data generation framework to create realistic, industry-grade SOPs that rigorously test the planning, reasoning, and tool-use capabilities of LLM-based agents. Second, using this framework, we develop SOP-Bench, a benchmark of over 1,800 tasks across 10 industrial domains, each with APIs, tool interfaces, and human-validated test cases. Third, we evaluate two prominent agent architectures: Function-Calling and ReAct Agents, on SOP-Bench, observing average success rates of only 27% and 48%, respectively. Remarkably, when the tool registry is much larger than necessary, agents invoke incorrect tools nearly 100% of the time. These findings underscore a substantial gap between current agentic capabilities of LLMs and the demands of automating real-world SOPs. Performance varies significantly by task and domain, highlighting the need for domain-specific benchmarking and architectural choices before deployment. SOP-Bench is publicly available at this http URL. We also release the prompts underpinning the data generation framework to support new domain-specific SOP benchmarks. We invite the community to extend SOP-Bench with SOPs from their industrial domains. 

---
# Agentic Neural Networks: Self-Evolving Multi-Agent Systems via Textual Backpropagation 

**Authors**: Xiaowen Ma, Chenyang Lin, Yao Zhang, Volker Tresp, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.09046)  

**Abstract**: Leveraging multiple Large Language Models(LLMs) has proven effective for addressing complex, high-dimensional tasks, but current approaches often rely on static, manually engineered multi-agent configurations. To overcome these constraints, we present the Agentic Neural Network(ANN), a framework that conceptualizes multi-agent collaboration as a layered neural network architecture. In this design, each agent operates as a node, and each layer forms a cooperative "team" focused on a specific subtask. Agentic Neural Network follows a two-phase optimization strategy: (1) Forward Phase-Drawing inspiration from neural network forward passes, tasks are dynamically decomposed into subtasks, and cooperative agent teams with suitable aggregation methods are constructed layer by layer. (2) Backward Phase-Mirroring backpropagation, we refine both global and local collaboration through iterative feedback, allowing agents to self-evolve their roles, prompts, and coordination. This neuro-symbolic approach enables ANN to create new or specialized agent teams post-training, delivering notable gains in accuracy and adaptability. Across four benchmark datasets, ANN surpasses leading multi-agent baselines under the same configurations, showing consistent performance improvements. Our findings indicate that ANN provides a scalable, data-driven framework for multi-agent systems, combining the collaborative capabilities of LLMs with the efficiency and flexibility of neural network principles. We plan to open-source the entire framework. 

---
# Cognitive Weave: Synthesizing Abstracted Knowledge with a Spatio-Temporal Resonance Graph 

**Authors**: Akash Vishwakarma, Hojin Lee, Mohith Suresh, Priyam Shankar Sharma, Rahul Vishwakarma, Sparsh Gupta, Yuvraj Anupam Chauhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.08098)  

**Abstract**: The emergence of capable large language model (LLM) based agents necessitates memory architectures that transcend mere data storage, enabling continuous learning, nuanced reasoning, and dynamic adaptation. Current memory systems often grapple with fundamental limitations in structural flexibility, temporal awareness, and the ability to synthesize higher-level insights from raw interaction data. This paper introduces Cognitive Weave, a novel memory framework centered around a multi-layered spatio-temporal resonance graph (STRG). This graph manages information as semantically rich insight particles (IPs), which are dynamically enriched with resonance keys, signifiers, and situational imprints via a dedicated semantic oracle interface (SOI). These IPs are interconnected through typed relational strands, forming an evolving knowledge tapestry. A key component of Cognitive Weave is the cognitive refinement process, an autonomous mechanism that includes the synthesis of insight aggregates (IAs) condensed, higher-level knowledge structures derived from identified clusters of related IPs. We present comprehensive experimental results demonstrating Cognitive Weave's marked enhancement over existing approaches in long-horizon planning tasks, evolving question-answering scenarios, and multi-session dialogue coherence. The system achieves a notable 34% average improvement in task completion rates and a 42% reduction in mean query latency when compared to state-of-the-art baselines. Furthermore, this paper explores the ethical considerations inherent in such advanced memory systems, discusses the implications for long-term memory in LLMs, and outlines promising future research trajectories. 

---
# FZOO: Fast Zeroth-Order Optimizer for Fine-Tuning Large Language Models towards Adam-Scale Speed 

**Authors**: Sizhe Dang, Yangyang Guo, Yanjun Zhao, Haishan Ye, Xiaodong Zheng, Guang Dai, Ivor Tsang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09034)  

**Abstract**: Fine-tuning large language models (LLMs) often faces GPU memory bottlenecks: the backward pass of first-order optimizers like Adam increases memory usage to more than 10 times the inference level (e.g., 633 GB for OPT-30B). Zeroth-order (ZO) optimizers avoid this cost by estimating gradients only from forward passes, yet existing methods like MeZO usually require many more steps to converge. Can this trade-off between speed and memory in ZO be fundamentally improved? Normalized-SGD demonstrates strong empirical performance with greater memory efficiency than Adam. In light of this, we introduce FZOO, a Fast Zeroth-Order Optimizer toward Adam-Scale Speed. FZOO reduces the total forward passes needed for convergence by employing batched one-sided estimates that adapt step sizes based on the standard deviation of batch losses. It also accelerates per-batch computation through the use of Rademacher random vector perturbations coupled with CUDA's parallel processing. Extensive experiments on diverse models, including RoBERTa-large, OPT (350M-66B), Phi-2, and Llama3, across 11 tasks validate FZOO's effectiveness. On average, FZOO outperforms MeZO by 3 percent in accuracy while requiring 3 times fewer forward passes. For RoBERTa-large, FZOO achieves average improvements of 5.6 percent in accuracy and an 18 times reduction in forward passes compared to MeZO, achieving convergence speeds comparable to Adam. We also provide theoretical analysis proving FZOO's formal equivalence to a normalized-SGD update rule and its convergence guarantees. FZOO integrates smoothly into PEFT techniques, enabling even larger memory savings. Overall, our results make single-GPU, high-speed, full-parameter fine-tuning practical and point toward future work on memory-efficient pre-training. 

---
# Breaking the ICE: Exploring promises and challenges of benchmarks for Inference Carbon & Energy estimation for LLMs 

**Authors**: Samarth Sikand, Rohit Mehra, Priyavanshi Pathania, Nikhil Bamby, Vibhu Saujanya Sharma, Vikrant Kaulgud, Sanjay Podder, Adam P. Burden  

**Link**: [PDF](https://arxiv.org/pdf/2506.08727)  

**Abstract**: While Generative AI stands to be one of the fastest adopted technologies ever, studies have made evident that the usage of Large Language Models (LLMs) puts significant burden on energy grids and our environment. It may prove a hindrance to the Sustainability goals of any organization. A crucial step in any Sustainability strategy is monitoring or estimating the energy consumption of various components. While there exist multiple tools for monitoring energy consumption, there is a dearth of tools/frameworks for estimating the consumption or carbon emissions. Current drawbacks of both monitoring and estimation tools include high input data points, intrusive nature, high error margin, etc. We posit that leveraging emerging LLM benchmarks and related data points can help overcome aforementioned challenges while balancing accuracy of the emission estimations. To that extent, we discuss the challenges of current approaches and present our evolving framework, R-ICE, which estimates prompt level inference carbon emissions by leveraging existing state-of-the-art(SOTA) benchmark. This direction provides a more practical and non-intrusive way to enable emerging use-cases like dynamic LLM routing, carbon accounting, etc. Our promising validation results suggest that benchmark-based modelling holds great potential for inference emission estimation and warrants further exploration from the scientific community. 

---
# MLVTG: Mamba-Based Feature Alignment and LLM-Driven Purification for Multi-Modal Video Temporal Grounding 

**Authors**: Zhiyi Zhu, Xiaoyu Wu, Zihao Liu, Linlin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08512)  

**Abstract**: Video Temporal Grounding (VTG), which aims to localize video clips corresponding to natural language queries, is a fundamental yet challenging task in video understanding. Existing Transformer-based methods often suffer from redundant attention and suboptimal multi-modal alignment. To address these limitations, we propose MLVTG, a novel framework that integrates two key modules: MambaAligner and LLMRefiner. MambaAligner uses stacked Vision Mamba blocks as a backbone instead of Transformers to model temporal dependencies and extract robust video representations for multi-modal alignment. LLMRefiner leverages the specific frozen layer of a pre-trained Large Language Model (LLM) to implicitly transfer semantic priors, enhancing multi-modal alignment without fine-tuning. This dual alignment strategy, temporal modeling via structured state-space dynamics and semantic purification via textual priors, enables more precise localization. Extensive experiments on QVHighlights, Charades-STA, and TVSum demonstrate that MLVTG achieves state-of-the-art performance and significantly outperforms existing baselines. 

---
# Your Agent Can Defend Itself against Backdoor Attacks 

**Authors**: Li Changjiang, Liang Jiacheng, Cao Bochuan, Chen Jinghui, Wang Ting  

**Link**: [PDF](https://arxiv.org/pdf/2506.08336)  

**Abstract**: Despite their growing adoption across domains, large language model (LLM)-powered agents face significant security risks from backdoor attacks during training and fine-tuning. These compromised agents can subsequently be manipulated to execute malicious operations when presented with specific triggers in their inputs or environments. To address this pressing risk, we present ReAgent, a novel defense against a range of backdoor attacks on LLM-based agents. Intuitively, backdoor attacks often result in inconsistencies among the user's instruction, the agent's planning, and its execution. Drawing on this insight, ReAgent employs a two-level approach to detect potential backdoors. At the execution level, ReAgent verifies consistency between the agent's thoughts and actions; at the planning level, ReAgent leverages the agent's capability to reconstruct the instruction based on its thought trajectory, checking for consistency between the reconstructed instruction and the user's instruction. Extensive evaluation demonstrates ReAgent's effectiveness against various backdoor attacks across tasks. For instance, ReAgent reduces the attack success rate by up to 90\% in database operation tasks, outperforming existing defenses by large margins. This work reveals the potential of utilizing compromised agents themselves to mitigate backdoor risks. 

---
# How Good LLM-Generated Password Policies Are? 

**Authors**: Vivek Vaidya, Aditya Patwardhan, Ashish Kundu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08320)  

**Abstract**: Generative AI technologies, particularly Large Language Models (LLMs), are rapidly being adopted across industry, academia, and government sectors, owing to their remarkable capabilities in natural language processing. However, despite their strengths, the inconsistency and unpredictability of LLM outputs present substantial challenges, especially in security-critical domains such as access control. One critical issue that emerges prominently is the consistency of LLM-generated responses, which is paramount for ensuring secure and reliable operations.
In this paper, we study the application of LLMs within the context of Cybersecurity Access Control Systems. Specifically, we investigate the consistency and accuracy of LLM-generated password policies, translating natural language prompts into executable this http URL configuration files. Our experimental methodology adopts two distinct approaches: firstly, we utilize pre-trained LLMs to generate configuration files purely from natural language prompts without additional guidance. Secondly, we provide these models with official this http URL documentation to serve as an informative baseline. We systematically assess the soundness, accuracy, and consistency of these AI-generated configurations. Our findings underscore significant challenges in the current generation of LLMs and contribute valuable insights into refining the deployment of LLMs in Access Control Systems. 

---
# SPBA: Utilizing Speech Large Language Model for Backdoor Attacks on Speech Classification Models 

**Authors**: Wenhan Yao, Fen Xiao, Xiarun Chen, Jia Liu, YongQiang He, Weiping Wen  

**Link**: [PDF](https://arxiv.org/pdf/2506.08346)  

**Abstract**: Deep speech classification tasks, including keyword spotting and speaker verification, are vital in speech-based human-computer interaction. Recently, the security of these technologies has been revealed to be susceptible to backdoor attacks. Specifically, attackers use noisy disruption triggers and speech element triggers to produce poisoned speech samples that train models to become vulnerable. However, these methods typically create only a limited number of backdoors due to the inherent constraints of the trigger function. In this paper, we propose that speech backdoor attacks can strategically focus on speech elements such as timbre and emotion, leveraging the Speech Large Language Model (SLLM) to generate diverse triggers. Increasing the number of triggers may disproportionately elevate the poisoning rate, resulting in higher attack costs and a lower success rate per trigger. We introduce the Multiple Gradient Descent Algorithm (MGDA) as a mitigation strategy to address this challenge. The proposed attack is called the Speech Prompt Backdoor Attack (SPBA). Building on this foundation, we conducted attack experiments on two speech classification tasks, demonstrating that SPBA shows significant trigger effectiveness and achieves exceptional performance in attack metrics. 

---
# Ensuring Reliability of Curated EHR-Derived Data: The Validation of Accuracy for LLM/ML-Extracted Information and Data (VALID) Framework 

**Authors**: Melissa Estevez, Nisha Singh, Lauren Dyson, Blythe Adamson, Qianyu Yuan, Megan W. Hildner, Erin Fidyk, Olive Mbah, Farhad Khan, Kathi Seidl-Rathkopf, Aaron B. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2506.08231)  

**Abstract**: Large language models (LLMs) are increasingly used to extract clinical data from electronic health records (EHRs), offering significant improvements in scalability and efficiency for real-world data (RWD) curation in oncology. However, the adoption of LLMs introduces new challenges in ensuring the reliability, accuracy, and fairness of extracted data, which are essential for research, regulatory, and clinical applications. Existing quality assurance frameworks for RWD and artificial intelligence do not fully address the unique error modes and complexities associated with LLM-extracted data. In this paper, we propose a comprehensive framework for evaluating the quality of clinical data extracted by LLMs. The framework integrates variable-level performance benchmarking against expert human abstraction, automated verification checks for internal consistency and plausibility, and replication analyses comparing LLM-extracted data to human-abstracted datasets or external standards. This multidimensional approach enables the identification of variables most in need of improvement, systematic detection of latent errors, and confirmation of dataset fitness-for-purpose in real-world research. Additionally, the framework supports bias assessment by stratifying metrics across demographic subgroups. By providing a rigorous and transparent method for assessing LLM-extracted RWD, this framework advances industry standards and supports the trustworthy use of AI-powered evidence generation in oncology research and practice. 

---
# Worst-Case Symbolic Constraints Analysis and Generalisation with Large Language Models 

**Authors**: Daniel Koh, Yannic Noller, Corina S. Pasareanu, Adrians Skapars, Youcheng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.08171)  

**Abstract**: Large language models (LLMs) have been successfully applied to a variety of coding tasks, including code generation, completion, and repair. However, more complex symbolic reasoning tasks remain largely unexplored by LLMs. This paper investigates the capacity of LLMs to reason about worst-case executions in programs through symbolic constraints analysis, aiming to connect LLMs and symbolic reasoning approaches. Specifically, we define and address the problem of worst-case symbolic constraints analysis as a measure to assess the comprehension of LLMs. We evaluate the performance of existing LLMs on this novel task and further improve their capabilities through symbolic reasoning-guided fine-tuning, grounded in SMT (Satisfiability Modulo Theories) constraint solving and supported by a specially designed dataset of symbolic constraints. Experimental results show that our solver-aligned model, WARP-1.0-3B, consistently surpasses size-matched and even much larger baselines, demonstrating that a 3B LLM can recover the very constraints that pin down an algorithm's worst-case behaviour through reinforcement learning methods. These findings suggest that LLMs are capable of engaging in deeper symbolic reasoning, supporting a closer integration between neural network-based learning and formal methods for rigorous program analysis. 

---
# Eliciting Fine-Tuned Transformer Capabilities via Inference-Time Techniques 

**Authors**: Asankhaya Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2506.08060)  

**Abstract**: Large language models have transformed natural language processing, yet supervised fine-tuning (SFT) remains computationally intensive. This paper formally proves that capabilities acquired through SFT can be approximated by a base transformer model using inference-time techniques, specifically in-context learning (ICL), without altering model parameters, under idealized assumptions including unbounded computational resources and access to the fine-tuning dataset. We extend these results to practical scenarios with finite context lengths and partial dataset access. For text generation tasks with fixed output length $l$, datasets of size $\mathrm{O}\left( \frac{m V}{\varepsilon^2} \log \frac{m}{\delta} \right)$ or, with bounded context, $\mathrm{O}\left( \frac{l \log V}{\varepsilon^2} \log \frac{1}{\delta} \right)$ suffice to approximate fine-tuned behavior across $m$ contexts within error $\varepsilon$, where $V$ is the vocabulary size and $\delta$ is the failure probability. For linear classification, datasets of size $\mathrm{O}\left( \frac{d}{\varepsilon} \right)$ or, with fixed context, $\mathrm{O}\left( \frac{1}{\varepsilon^2} \log \frac{1}{\delta} \right)$ are sufficient, where $d$ is the input dimension. Grounded in the Turing completeness of transformers, these results provide a theoretical foundation for resource-efficient deployment of large language models, with practical techniques like retrieval-augmented generation bridging theory to real-world applications. 

---
# Being Strong Progressively! Enhancing Knowledge Distillation of Large Language Models through a Curriculum Learning Framework 

**Authors**: Lingyuan Liu, Mengxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05695)  

**Abstract**: Knowledge Distillation (KD) compresses large language models (LLMs) by transferring the teacher model's capabilities to a smaller student model, reducing inference cost and memory usage while maintaining performance. However, existing KD methods for LLMs often fail to prevent significant shifts in the student model's distribution during training, leading to issues such as catastrophic forgetting, mode collapse, and training-inference mismatch. To address these challenges, we propose a novel, plug-in curriculum learning framework inspired by the strength training principle of "progressive overload" (POCL), which can be seamlessly integrated into existing white-box KD approaches with minimal computational overhead. The framework comprises two core components: (1) a difficulty measurer that ranks and partitions training samples from easy to hard, and (2) a training scheduler that incrementally introduces these subsets into the distillation process at fixed intervals while applying loss functions with progressively rising temperatures. By starting with the easiest samples and progressively increasing the difficulty, the approach enhances both the stability and efficiency of learning. Extensive experiments in instruction-following settings demonstrate that POCL consistently improves the performance of distilled student models across various white-box KD methods and model families. Our findings highlight the effectiveness of sorted training samples in KD for LLMs. More generally, our work demonstrates how to structure training data within the KD process to enhance the stability and performance of distilled LLMs. 

---
# QUITE: A Query Rewrite System Beyond Rules with LLM Agents 

**Authors**: Yuyang Song, Hanxu Yan, Jiale Lao, Yibo Wang, Yufei Li, Yuanchun Zhou, Jianguo Wang, Mingjie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07675)  

**Abstract**: Query rewrite transforms SQL queries into semantically equivalent forms that run more efficiently. Existing approaches mainly rely on predefined rewrite rules, but they handle a limited subset of queries and can cause performance regressions. This limitation stems from three challenges of rule-based query rewrite: (1) it is hard to discover and verify new rules, (2) fixed rewrite rules do not generalize to new query patterns, and (3) some rewrite techniques cannot be expressed as fixed rules. Motivated by the fact that human experts exhibit significantly better rewrite ability but suffer from scalability, and Large Language Models (LLMs) have demonstrated nearly human-level semantic and reasoning abilities, we propose a new approach of using LLMs to rewrite SQL queries beyond rules. Due to the hallucination problems in LLMs, directly applying LLMs often leads to nonequivalent and suboptimal queries. To address this issue, we propose QUITE (query rewrite), a training-free and feedback-aware system based on LLM agents that rewrites SQL queries into semantically equivalent forms with significantly better performance, covering a broader range of query patterns and rewrite strategies compared to rule-based methods. Firstly, we design a multi-agent framework controlled by a finite state machine (FSM) to equip LLMs with the ability to use external tools and enhance the rewrite process with real-time database feedback. Secondly, we develop a rewrite middleware to enhance the ability of LLMs to generate optimized query equivalents. Finally, we employ a novel hint injection technique to improve execution plans for rewritten queries. Extensive experiments show that QUITE reduces query execution time by up to 35.8% over state-of-the-art approaches and produces 24.1% more rewrites than prior methods, covering query cases that earlier systems did not handle. 

---
# ChemGraph: An Agentic Framework for Computational Chemistry Workflows 

**Authors**: Thang D. Pham, Aditya Tanikanti, Murat Keçeli  

**Link**: [PDF](https://arxiv.org/pdf/2506.06363)  

**Abstract**: Atomistic simulations are essential tools in chemistry and materials science, accelerating the discovery of novel catalysts, energy storage materials, and pharmaceuticals. However, running these simulations remains challenging due to the wide range of computational methods, diverse software ecosystems, and the need for expert knowledge and manual effort for the setup, execution, and validation stages. In this work, we present ChemGraph, an agentic framework powered by artificial intelligence and state-of-the-art simulation tools to streamline and automate computational chemistry and materials science workflows. ChemGraph leverages graph neural network-based foundation models for accurate yet computationally efficient calculations and large language models (LLMs) for natural language understanding, task planning, and scientific reasoning to provide an intuitive and interactive interface. Users can perform tasks such as molecular structure generation, single-point energy, geometry optimization, vibrational analysis, and thermochemistry calculations with methods ranging from tight-binding and machine learning interatomic potentials to density functional theory or wave function theory-based methods. We evaluate ChemGraph across 13 benchmark tasks and demonstrate that smaller LLMs (GPT-4o-mini, Claude-3.5-haiku, Qwen2.5-14B) perform well on simple workflows, while more complex tasks benefit from using larger models like GPT-4o. Importantly, we show that decomposing complex tasks into smaller subtasks through a multi-agent framework enables smaller LLM models to match or exceed GPT-4o's performance in specific scenarios. 

---
# Exp4Fuse: A Rank Fusion Framework for Enhanced Sparse Retrieval using Large Language Model-based Query Expansion 

**Authors**: Lingyuan Liu, Mengxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04760)  

**Abstract**: Large Language Models (LLMs) have shown potential in generating hypothetical documents for query expansion, thereby enhancing information retrieval performance. However, the efficacy of this method is highly dependent on the quality of the generated documents, which often requires complex prompt strategies and the integration of advanced dense retrieval techniques. This can be both costly and computationally intensive. To mitigate these limitations, we explore the use of zero-shot LLM-based query expansion to improve sparse retrieval, particularly for learned sparse retrievers. We introduce a novel fusion ranking framework, Exp4Fuse, which enhances the performance of sparse retrievers through an indirect application of zero-shot LLM-based query expansion. Exp4Fuse operates by simultaneously considering two retrieval routes-one based on the original query and the other on the LLM-augmented query. It then generates two ranked lists using a sparse retriever and fuses them using a modified reciprocal rank fusion method. We conduct extensive evaluations of Exp4Fuse against leading LLM-based query expansion methods and advanced retrieval techniques on three MS MARCO-related datasets and seven low-resource datasets. Experimental results reveal that Exp4Fuse not only surpasses existing LLM-based query expansion methods in enhancing sparse retrievers but also, when combined with advanced sparse retrievers, achieves SOTA results on several benchmarks. This highlights the superior performance and effectiveness of Exp4Fuse in improving query expansion for sparse retrieval. 

---
