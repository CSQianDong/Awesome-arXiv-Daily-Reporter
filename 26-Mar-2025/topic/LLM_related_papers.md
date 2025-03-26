# Enhanced Bloom's Educational Taxonomy for Fostering Information Literacy in the Era of Large Language Models 

**Authors**: Yiming Luo, Ting Liu, Patrick Cheong-Iao Pang, Dana McKay, Ziqi Chen, George Buchanan, Shanton Chang  

**Link**: [PDF](https://arxiv.org/pdf/2503.19434)  

**Abstract**: The advent of Large Language Models (LLMs) has profoundly transformed the paradigms of information retrieval and problem-solving, enabling students to access information acquisition more efficiently to support learning. However, there is currently a lack of standardized evaluation frameworks that guide learners in effectively leveraging LLMs. This paper proposes an LLM-driven Bloom's Educational Taxonomy that aims to recognize and evaluate students' information literacy (IL) with LLMs, and to formalize and guide students practice-based activities of using LLMs to solve complex problems. The framework delineates the IL corresponding to the cognitive abilities required to use LLM into two distinct stages: Exploration & Action and Creation & Metacognition. It further subdivides these into seven phases: Perceiving, Searching, Reasoning, Interacting, Evaluating, Organizing, and Curating. Through the case presentation, the analysis demonstrates the framework's applicability and feasibility, supporting its role in fostering IL among students with varying levels of prior knowledge. This framework fills the existing gap in the analysis of LLM usage frameworks and provides theoretical support for guiding learners to improve IL. 

---
# Rankers, Judges, and Assistants: Towards Understanding the Interplay of LLMs in Information Retrieval Evaluation 

**Authors**: Krisztian Balog, Donald Metzler, Zhen Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.19092)  

**Abstract**: Large language models (LLMs) are increasingly integral to information retrieval (IR), powering ranking, evaluation, and AI-assisted content creation. This widespread adoption necessitates a critical examination of potential biases arising from the interplay between these LLM-based components. This paper synthesizes existing research and presents novel experiment designs that explore how LLM-based rankers and assistants influence LLM-based judges. We provide the first empirical evidence of LLM judges exhibiting significant bias towards LLM-based rankers. Furthermore, we observe limitations in LLM judges' ability to discern subtle system performance differences. Contrary to some previous findings, our preliminary study does not find evidence of bias against AI-generated content. These results highlight the need for a more holistic view of the LLM-driven information ecosystem. To this end, we offer initial guidelines and a research agenda to ensure the reliable use of LLMs in IR evaluation. 

---
# Understanding and Improving Information Preservation in Prompt Compression for LLMs 

**Authors**: Weronika Łajewska, Momchil Hardalov, Laura Aina, Neha Anna John, Hang Su, Lluís Màrquez  

**Link**: [PDF](https://arxiv.org/pdf/2503.19114)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled their successful application to a broad range of tasks. However, in information-intensive tasks, the prompt length can grow fast, leading to increased computational requirements, performance degradation, and induced biases from irrelevant or redundant information. Recently, various prompt compression techniques have been introduced to optimize the trade-off between reducing input length and retaining performance. We propose a holistic evaluation framework that allows for in-depth analysis of prompt compression methods. We focus on three key aspects, besides compression ratio: (i) downstream task performance, (ii) grounding in the input context, and (iii) information preservation. Through this framework, we investigate state-of-the-art soft and hard compression methods, showing that they struggle to preserve key details from the original prompt, limiting their performance on complex tasks. We demonstrate that modifying soft prompting methods to control better the granularity of the compressed information can significantly improve their effectiveness -- up to +23\% in downstream task performance, more than +8 BERTScore points in grounding, and 2.7x more entities preserved in compression. 

---
# Context-Efficient Retrieval with Factual Decomposition 

**Authors**: Yanhong Li, David Yunis, David McAllester, Jiawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.19574)  

**Abstract**: There has recently been considerable interest in incorporating information retrieval into large language models (LLMs). Retrieval from a dynamically expanding external corpus of text allows a model to incorporate current events and can be viewed as a form of episodic memory. Here we demonstrate that pre-processing the external corpus into semi-structured ''atomic facts'' makes retrieval more efficient. More specifically, we demonstrate that our particular form of atomic facts improves performance on various question answering tasks when the amount of retrieved text is limited. Limiting the amount of retrieval reduces the size of the context and improves inference efficiency. 

---
# CoLLM: A Large Language Model for Composed Image Retrieval 

**Authors**: Chuong Huynh, Jinyu Yang, Ashish Tawari, Mubarak Shah, Son Tran, Raffay Hamid, Trishul Chilimbi, Abhinav Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2503.19910)  

**Abstract**: Composed Image Retrieval (CIR) is a complex task that aims to retrieve images based on a multimodal query. Typical training data consists of triplets containing a reference image, a textual description of desired modifications, and the target image, which are expensive and time-consuming to acquire. The scarcity of CIR datasets has led to zero-shot approaches utilizing synthetic triplets or leveraging vision-language models (VLMs) with ubiquitous web-crawled image-caption pairs. However, these methods have significant limitations: synthetic triplets suffer from limited scale, lack of diversity, and unnatural modification text, while image-caption pairs hinder joint embedding learning of the multimodal query due to the absence of triplet data. Moreover, existing approaches struggle with complex and nuanced modification texts that demand sophisticated fusion and understanding of vision and language modalities. We present CoLLM, a one-stop framework that effectively addresses these limitations. Our approach generates triplets on-the-fly from image-caption pairs, enabling supervised training without manual annotation. We leverage Large Language Models (LLMs) to generate joint embeddings of reference images and modification texts, facilitating deeper multimodal fusion. Additionally, we introduce Multi-Text CIR (MTCIR), a large-scale dataset comprising 3.4M samples, and refine existing CIR benchmarks (CIRR and Fashion-IQ) to enhance evaluation reliability. Experimental results demonstrate that CoLLM achieves state-of-the-art performance across multiple CIR benchmarks and settings. MTCIR yields competitive results, with up to 15% performance improvement. Our refined benchmarks provide more reliable evaluation metrics for CIR models, contributing to the advancement of this important field. 

---
# Think Twice: Enhancing LLM Reasoning by Scaling Multi-round Test-time Thinking 

**Authors**: Xiaoyu Tian, Sitong Zhao, Haotian Wang, Shuaiting Chen, Yunjie Ji, Yiping Peng, Han Zhao, Xiangang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.19855)  

**Abstract**: Recent advances in large language models (LLMs), such as OpenAI-o1 and DeepSeek-R1, have demonstrated the effectiveness of test-time scaling, where extended reasoning processes substantially enhance model performance. Despite this, current models are constrained by limitations in handling long texts and reinforcement learning (RL) training efficiency. To address these issues, we propose a simple yet effective test-time scaling approach Multi-round Thinking. This method iteratively refines model reasoning by leveraging previous answers as prompts for subsequent rounds. Extensive experiments across multiple models, including QwQ-32B and DeepSeek-R1, consistently show performance improvements on various benchmarks such as AIME 2024, MATH-500, GPQA-diamond, and LiveCodeBench. For instance, the accuracy of QwQ-32B improved from 80.3% (Round 1) to 82.1% (Round 2) on the AIME 2024 dataset, while DeepSeek-R1 showed a similar increase from 79.7% to 82.0%. These results confirm that Multi-round Thinking is a broadly applicable, straightforward approach to achieving stable enhancements in model performance, underscoring its potential for future developments in test-time scaling techniques. The key prompt: {Original question prompt} The assistant's previous answer is: <answer> {last round answer} </answer>, and please re-answer. 

---
# A Comparative Analysis of Word Segmentation, Part-of-Speech Tagging, and Named Entity Recognition for Historical Chinese Sources, 1900-1950 

**Authors**: Zhao Fang, Liang-Chun Wu, Xuening Kong, Spencer Dean Stewart  

**Link**: [PDF](https://arxiv.org/pdf/2503.19844)  

**Abstract**: This paper compares large language models (LLMs) and traditional natural language processing (NLP) tools for performing word segmentation, part-of-speech (POS) tagging, and named entity recognition (NER) on Chinese texts from 1900 to 1950. Historical Chinese documents pose challenges for text analysis due to their logographic script, the absence of natural word boundaries, and significant linguistic changes. Using a sample dataset from the Shanghai Library Republican Journal corpus, traditional tools such as Jieba and spaCy are compared to LLMs, including GPT-4o, Claude 3.5, and the GLM series. The results show that LLMs outperform traditional methods in all metrics, albeit at considerably higher computational costs, highlighting a trade-off between accuracy and efficiency. Additionally, LLMs better handle genre-specific challenges such as poetry and temporal variations (i.e., pre-1920 versus post-1920 texts), demonstrating that their contextual learning capabilities can advance NLP approaches to historical texts by reducing the need for domain-specific training data. 

---
# AdaptiVocab: Enhancing LLM Efficiency in Focused Domains through Lightweight Vocabulary Adaptation 

**Authors**: Itay Nakash, Nitay Calderon, Eyal Ben David, Elad Hoffer, Roi Reichart  

**Link**: [PDF](https://arxiv.org/pdf/2503.19693)  

**Abstract**: Large Language Models (LLMs) have shown impressive versatility as general purpose models. However, their broad applicability comes at a high-cost computational overhead, particularly in auto-regressive decoding where each step requires a forward pass. In domain-specific settings, general-purpose capabilities are unnecessary and can be exchanged for efficiency. In this work, we take a novel perspective on domain adaptation, reducing latency and computational costs by adapting the vocabulary to focused domains of interest. We introduce AdaptiVocab, an end-to-end approach for vocabulary adaptation, designed to enhance LLM efficiency in low-resource domains. AdaptiVocab can be applied to any tokenizer and architecture, modifying the vocabulary by replacing tokens with domain-specific n-gram-based tokens, thereby reducing the number of tokens required for both input processing and output generation. AdaptiVocab initializes new n-token embeddings using an exponentially weighted combination of existing embeddings and employs a lightweight fine-tuning phase that can be efficiently performed on a single GPU. We evaluate two 7B LLMs across three niche domains, assessing efficiency, generation quality, and end-task performance. Our results show that AdaptiVocab reduces token usage by over 25% without compromising performance 

---
# Scaling Evaluation-time Compute with Reasoning Models as Process Evaluators 

**Authors**: Seungone Kim, Ian Wu, Jinu Lee, Xiang Yue, Seongyun Lee, Mingyeong Moon, Kiril Gashteovski, Carolin Lawrence, Julia Hockenmaier, Graham Neubig, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2503.19877)  

**Abstract**: As language model (LM) outputs get more and more natural, it is becoming more difficult than ever to evaluate their quality. Simultaneously, increasing LMs' "thinking" time through scaling test-time compute has proven an effective technique to solve challenging problems in domains such as math and code. This raises a natural question: can an LM's evaluation capability also be improved by spending more test-time compute? To answer this, we investigate employing reasoning models-LMs that natively generate long chain-of-thought reasoning-as evaluators. Specifically, we examine methods to leverage more test-time compute by (1) using reasoning models, and (2) prompting these models to evaluate not only the response as a whole (i.e., outcome evaluation) but also assess each step in the response separately (i.e., process evaluation). In experiments, we observe that the evaluator's performance improves monotonically when generating more reasoning tokens, similar to the trends observed in LM-based generation. Furthermore, we use these more accurate evaluators to rerank multiple generations, and demonstrate that spending more compute at evaluation time can be as effective as using more compute at generation time in improving an LM's problem-solving capability. 

---
# CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation 

**Authors**: Nengbo Wang, Xiaotian Han, Jagdip Singh, Jing Ma, Vipin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2503.19878)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing (NLP), particularly through Retrieval-Augmented Generation (RAG), which enhances LLM capabilities by integrating external knowledge. However, traditional RAG systems face critical limitations, including disrupted contextual integrity due to text chunking, and over-reliance on semantic similarity for retrieval. To address these issues, we propose CausalRAG, a novel framework that incorporates causal graphs into the retrieval process. By constructing and tracing causal relationships, CausalRAG preserves contextual continuity and improves retrieval precision, leading to more accurate and interpretable responses. We evaluate CausalRAG against regular RAG and graph-based RAG approaches, demonstrating its superiority across several metrics. Our findings suggest that grounding retrieval in causal reasoning provides a promising approach to knowledge-intensive tasks. 

---
# Writing as a testbed for open ended agents 

**Authors**: Sian Gooding, Lucia Lopez-Rivilla, Edward Grefenstette  

**Link**: [PDF](https://arxiv.org/pdf/2503.19711)  

**Abstract**: Open-ended tasks are particularly challenging for LLMs due to the vast solution space, demanding both expansive exploration and adaptable strategies, especially when success lacks a clear, objective definition. Writing, with its vast solution space and subjective evaluation criteria, provides a compelling testbed for studying such problems. In this paper, we investigate the potential of LLMs to act as collaborative co-writers, capable of suggesting and implementing text improvements autonomously. We analyse three prominent LLMs - Gemini 1.5 Pro, Claude 3.5 Sonnet, and GPT-4o - focusing on how their action diversity, human alignment, and iterative improvement capabilities impact overall performance. This work establishes a framework for benchmarking autonomous writing agents and, more broadly, highlights fundamental challenges and potential solutions for building systems capable of excelling in diverse open-ended domains. 

---
# HausaNLP at SemEval-2025 Task 3: Towards a Fine-Grained Model-Aware Hallucination Detection 

**Authors**: Maryam Bala, Amina Imam Abubakar, Abdulhamid Abubakar, Abdulkadir Shehu Bichi, Hafsa Kabir Ahmad, Sani Abdullahi Sani, Idris Abdulmumin, Shamsuddeen Hassan Muhamad, Ibrahim Said Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2503.19650)  

**Abstract**: This paper presents our findings of the Multilingual Shared Task on Hallucinations and Related Observable Overgeneration Mistakes, MU-SHROOM, which focuses on identifying hallucinations and related overgeneration errors in large language models (LLMs). The shared task involves detecting specific text spans that constitute hallucinations in the outputs generated by LLMs in 14 languages. To address this task, we aim to provide a nuanced, model-aware understanding of hallucination occurrences and severity in English. We used natural language inference and fine-tuned a ModernBERT model using a synthetic dataset of 400 samples, achieving an Intersection over Union (IoU) score of 0.032 and a correlation score of 0.422. These results indicate a moderately positive correlation between the model's confidence scores and the actual presence of hallucinations. The IoU score indicates that our model has a relatively low overlap between the predicted hallucination span and the truth annotation. The performance is unsurprising, given the intricate nature of hallucination detection. Hallucinations often manifest subtly, relying on context, making pinpointing their exact boundaries formidable. 

---
# The Greatest Good Benchmark: Measuring LLMs' Alignment with Utilitarian Moral Dilemmas 

**Authors**: Giovanni Franco Gabriel Marraffini, Andrés Cotton, Noe Fabian Hsueh, Axel Fridman, Juan Wisznia, Luciano Del Corro  

**Link**: [PDF](https://arxiv.org/pdf/2503.19598)  

**Abstract**: The question of how to make decisions that maximise the well-being of all persons is very relevant to design language models that are beneficial to humanity and free from harm. We introduce the Greatest Good Benchmark to evaluate the moral judgments of LLMs using utilitarian dilemmas. Our analysis across 15 diverse LLMs reveals consistently encoded moral preferences that diverge from established moral theories and lay population moral standards. Most LLMs have a marked preference for impartial beneficence and rejection of instrumental harm. These findings showcase the 'artificial moral compass' of LLMs, offering insights into their moral alignment. 

---
# KSHSeek: Data-Driven Approaches to Mitigating and Detecting Knowledge-Shortcut Hallucinations in Generative Models 

**Authors**: Zhiwei Wang, Zhongxin Liu, Ying Li, Hongyu Sun, Meng Xu, Yuqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.19482)  

**Abstract**: The emergence of large language models (LLMs) has significantly advanced the development of natural language processing (NLP), especially in text generation tasks like question answering. However, model hallucinations remain a major challenge in natural language generation (NLG) tasks due to their complex causes. We systematically expand on the causes of factual hallucinations from the perspective of knowledge shortcuts, analyzing hallucinations arising from correct and defect-free data and demonstrating that knowledge-shortcut hallucinations are prevalent in generative models. To mitigate this issue, we propose a high similarity pruning algorithm at the data preprocessing level to reduce spurious correlations in the data. Additionally, we design a specific detection method for knowledge-shortcut hallucinations to evaluate the effectiveness of our mitigation strategy. Experimental results show that our approach effectively reduces knowledge-shortcut hallucinations, particularly in fine-tuning tasks, without negatively impacting model performance in question answering. This work introduces a new paradigm for mitigating specific hallucination issues in generative models, enhancing their robustness and reliability in real-world applications. 

---
# DeCAP: Context-Adaptive Prompt Generation for Debiasing Zero-shot Question Answering in Large Language Models 

**Authors**: Suyoung Bae, YunSeok Choi, Jee-Hyong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.19426)  

**Abstract**: While Large Language Models (LLMs) excel in zero-shot Question Answering (QA), they tend to expose biases in their internal knowledge when faced with socially sensitive questions, leading to a degradation in performance. Existing zero-shot methods are efficient but fail to consider context and prevent bias propagation in the answers. To address this, we propose DeCAP, a method for debiasing LLMs using Context-Adaptive Prompt Generation. DeCAP leverages a Question Ambiguity Detection to take appropriate debiasing actions based on the context and a Neutral Answer Guidance Generation to suppress the LLMs make objective judgments about the context, minimizing the propagation of bias from their internal knowledge. Our various experiments across eight LLMs show that DeCAP achieves state-of-the-art zero-shot debiased QA performance. This demonstrates DeCAP's efficacy in enhancing the fairness and accuracy of LLMs in diverse QA settings. 

---
# 1.4 Million Open-Source Distilled Reasoning Dataset to Empower Large Language Model Training 

**Authors**: Han Zhao, Haotian Wang, Yiping Peng, Sitong Zhao, Xiaoyu Tian, Shuaiting Chen, Yunjie Ji, Xiangang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.19633)  

**Abstract**: The AM-DeepSeek-R1-Distilled is a large-scale dataset with thinking traces for general reasoning tasks, composed of high-quality and challenging reasoning problems. These problems are collected from a multitude of open-source datasets, subjected to semantic deduplication and meticulous cleaning to eliminate test set contamination. All responses within the dataset are distilled from reasoning models (predominantly DeepSeek-R1) and have undergone rigorous verification procedures. Mathematical problems are validated by checking against reference answers, code problems are verified using test cases, and other tasks are evaluated with the aid of a reward model. The AM-Distill-Qwen-32B model, which was trained through only simple Supervised Fine-Tuning (SFT) using this batch of data, outperformed the DeepSeek-R1-Distill-Qwen-32B model on four benchmarks: AIME2024, MATH-500, GPQA-Diamond, and LiveCodeBench. Additionally, the AM-Distill-Qwen-72B model surpassed the DeepSeek-R1-Distill-Llama-70B model on all benchmarks as well. We are releasing these 1.4 million problems and their corresponding responses to the research community with the objective of fostering the development of powerful reasoning-oriented Large Language Models (LLMs). The dataset was published in \href{this https URL}{this https URL}. 

---
# PHEONA: An Evaluation Framework for Large Language Model-based Approaches to Computational Phenotyping 

**Authors**: Sarah Pungitore, Shashank Yadav, Vignesh Subbian  

**Link**: [PDF](https://arxiv.org/pdf/2503.19265)  

**Abstract**: Computational phenotyping is essential for biomedical research but often requires significant time and resources, especially since traditional methods typically involve extensive manual data review. While machine learning and natural language processing advancements have helped, further improvements are needed. Few studies have explored using Large Language Models (LLMs) for these tasks despite known advantages of LLMs for text-based tasks. To facilitate further research in this area, we developed an evaluation framework, Evaluation of PHEnotyping for Observational Health Data (PHEONA), that outlines context-specific considerations. We applied and demonstrated PHEONA on concept classification, a specific task within a broader phenotyping process for Acute Respiratory Failure (ARF) respiratory support therapies. From the sample concepts tested, we achieved high classification accuracy, suggesting the potential for LLM-based methods to improve computational phenotyping processes. 

---
# Linguistic Blind Spots of Large Language Models 

**Authors**: Jiali Cheng, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2503.19260)  

**Abstract**: Large language models (LLMs) are the foundation of many AI applications today. However, despite their remarkable proficiency in generating coherent text, questions linger regarding their ability to perform fine-grained linguistic annotation tasks, such as detecting nouns or verbs, or identifying more complex syntactic structures like clauses in input texts. These tasks require precise syntactic and semantic understanding of input text, and when LLMs underperform on specific linguistic structures, it raises concerns about their reliability for detailed linguistic analysis and whether their (even correct) outputs truly reflect an understanding of the inputs. In this paper, we empirically study the performance of recent LLMs on fine-grained linguistic annotation tasks. Through a series of experiments, we find that recent LLMs show limited efficacy in addressing linguistic queries and often struggle with linguistically complex inputs. We show that the most capable LLM (Llama3-70b) makes notable errors in detecting linguistic structures, such as misidentifying embedded clauses, failing to recognize verb phrases, and confusing complex nominals with clauses. Our results provide insights to inform future advancements in LLM design and development. 

---
# SCI-IDEA: Context-Aware Scientific Ideation Using Token and Sentence Embeddings 

**Authors**: Farhana Keya, Gollam Rabby, Prasenjit Mitra, Sahar Vahdati, Sören Auer, Yaser Jaradeh  

**Link**: [PDF](https://arxiv.org/pdf/2503.19257)  

**Abstract**: Every scientific discovery starts with an idea inspired by prior work, interdisciplinary concepts, and emerging challenges. Recent advancements in large language models (LLMs) trained on scientific corpora have driven interest in AI-supported idea generation. However, generating context-aware, high-quality, and innovative ideas remains challenging. We introduce SCI-IDEA, a framework that uses LLM prompting strategies and Aha Moment detection for iterative idea refinement. SCI-IDEA extracts essential facets from research publications, assessing generated ideas on novelty, excitement, feasibility, and effectiveness. Comprehensive experiments validate SCI-IDEA's effectiveness, achieving average scores of 6.84, 6.86, 6.89, and 6.84 (on a 1-10 scale) across novelty, excitement, feasibility, and effectiveness, respectively. Evaluations employed GPT-4o, GPT-4.5, DeepSeek-32B (each under 2-shot prompting), and DeepSeek-70B (3-shot prompting), with token-level embeddings used for Aha Moment detection. Similarly, it achieves scores of 6.87, 6.86, 6.83, and 6.87 using GPT-4o under 5-shot prompting, GPT-4.5 under 3-shot prompting, DeepSeek-32B under zero-shot chain-of-thought prompting, and DeepSeek-70B under 5-shot prompting with sentence-level embeddings. We also address ethical considerations such as intellectual credit, potential misuse, and balancing human creativity with AI-driven ideation. Our results highlight SCI-IDEA's potential to facilitate the structured and flexible exploration of context-aware scientific ideas, supporting innovation while maintaining ethical standards. 

---
# A Survey of Large Language Model Agents for Question Answering 

**Authors**: Murong Yue  

**Link**: [PDF](https://arxiv.org/pdf/2503.19213)  

**Abstract**: This paper surveys the development of large language model (LLM)-based agents for question answering (QA). Traditional agents face significant limitations, including substantial data requirements and difficulty in generalizing to new environments. LLM-based agents address these challenges by leveraging LLMs as their core reasoning engine. These agents achieve superior QA results compared to traditional QA pipelines and naive LLM QA systems by enabling interaction with external environments. We systematically review the design of LLM agents in the context of QA tasks, organizing our discussion across key stages: planning, question understanding, information retrieval, and answer generation. Additionally, this paper identifies ongoing challenges and explores future research directions to enhance the performance of LLM agent QA systems. 

---
# Evaluating Bias in LLMs for Job-Resume Matching: Gender, Race, and Education 

**Authors**: Hayate Iso, Pouya Pezeshkpour, Nikita Bhutani, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2503.19182)  

**Abstract**: Large Language Models (LLMs) offer the potential to automate hiring by matching job descriptions with candidate resumes, streamlining recruitment processes, and reducing operational costs. However, biases inherent in these models may lead to unfair hiring practices, reinforcing societal prejudices and undermining workplace diversity. This study examines the performance and fairness of LLMs in job-resume matching tasks within the English language and U.S. context. It evaluates how factors such as gender, race, and educational background influence model decisions, providing critical insights into the fairness and reliability of LLMs in HR applications. Our findings indicate that while recent models have reduced biases related to explicit attributes like gender and race, implicit biases concerning educational background remain significant. These results highlight the need for ongoing evaluation and the development of advanced bias mitigation strategies to ensure equitable hiring practices when using LLMs in industry settings. 

---
# Language Model Uncertainty Quantification with Attention Chain 

**Authors**: Yinghao Li, Rushi Qiang, Lama Moukheiber, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.19168)  

**Abstract**: Accurately quantifying a large language model's (LLM) predictive uncertainty is crucial for judging the reliability of its answers. While most existing research focuses on short, directly answerable questions with closed-form outputs (e.g., multiple-choice), involving intermediate reasoning steps in LLM responses is increasingly important. This added complexity complicates uncertainty quantification (UQ) because the probabilities assigned to answer tokens are conditioned on a vast space of preceding reasoning tokens. Direct marginalization is infeasible, and the dependency inflates probability estimates, causing overconfidence in UQ. To address this, we propose UQAC, an efficient method that narrows the reasoning space to a tractable size for marginalization. UQAC iteratively constructs an "attention chain" of tokens deemed "semantically crucial" to the final answer via a backtracking procedure. Starting from the answer tokens, it uses attention weights to identify the most influential predecessors, then iterates this process until reaching the input tokens. Similarity filtering and probability thresholding further refine the resulting chain, allowing us to approximate the marginal probabilities of the answer tokens, which serve as the LLM's confidence. We validate UQAC on multiple reasoning benchmarks with advanced open-source LLMs, demonstrating that it consistently delivers reliable UQ estimates with high computational efficiency. 

---
# MIRAGE: Multimodal Immersive Reasoning and Guided Exploration for Red-Team Jailbreak Attacks 

**Authors**: Wenhao You, Bryan Hooi, Yiwei Wang, Youke Wang, Zong Ke, Ming-Hsuan Yang, Zi Huang, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2503.19134)  

**Abstract**: While safety mechanisms have significantly progressed in filtering harmful text inputs, MLLMs remain vulnerable to multimodal jailbreaks that exploit their cross-modal reasoning capabilities. We present MIRAGE, a novel multimodal jailbreak framework that exploits narrative-driven context and role immersion to circumvent safety mechanisms in Multimodal Large Language Models (MLLMs). By systematically decomposing the toxic query into environment, role, and action triplets, MIRAGE constructs a multi-turn visual storytelling sequence of images and text using Stable Diffusion, guiding the target model through an engaging detective narrative. This process progressively lowers the model's defences and subtly guides its reasoning through structured contextual cues, ultimately eliciting harmful responses. In extensive experiments on the selected datasets with six mainstream MLLMs, MIRAGE achieves state-of-the-art performance, improving attack success rates by up to 17.5% over the best baselines. Moreover, we demonstrate that role immersion and structured semantic reconstruction can activate inherent model biases, facilitating the model's spontaneous violation of ethical safeguards. These results highlight critical weaknesses in current multimodal safety mechanisms and underscore the urgent need for more robust defences against cross-modal threats. 

---
# LLM-Based Insight Extraction for Contact Center Analytics and Cost-Efficient Deployment 

**Authors**: Varsha Embar, Ritvik Shrivastava, Vinay Damodaran, Travis Mehlinger, Yu-Chung Hsiao, Karthik Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2503.19090)  

**Abstract**: Large Language Models have transformed the Contact Center industry, manifesting in enhanced self-service tools, streamlined administrative processes, and augmented agent productivity. This paper delineates our system that automates call driver generation, which serves as the foundation for tasks such as topic modeling, incoming call classification, trend detection, and FAQ generation, delivering actionable insights for contact center agents and administrators to consume. We present a cost-efficient LLM system design, with 1) a comprehensive evaluation of proprietary, open-weight, and fine-tuned models and 2) cost-efficient strategies, and 3) the corresponding cost analysis when deployed in production environments. 

---
# Masks and Mimicry: Strategic Obfuscation and Impersonation Attacks on Authorship Verification 

**Authors**: Kenneth Alperin, Rohan Leekha, Adaku Uchendu, Trang Nguyen, Srilakshmi Medarametla, Carlos Levya Capote, Seth Aycock, Charlie Dagli  

**Link**: [PDF](https://arxiv.org/pdf/2503.19099)  

**Abstract**: The increasing use of Artificial Intelligence (AI) technologies, such as Large Language Models (LLMs) has led to nontrivial improvements in various tasks, including accurate authorship identification of documents. However, while LLMs improve such defense techniques, they also simultaneously provide a vehicle for malicious actors to launch new attack vectors. To combat this security risk, we evaluate the adversarial robustness of authorship models (specifically an authorship verification model) to potent LLM-based attacks. These attacks include untargeted methods - \textit{authorship obfuscation} and targeted methods - \textit{authorship impersonation}. For both attacks, the objective is to mask or mimic the writing style of an author while preserving the original texts' semantics, respectively. Thus, we perturb an accurate authorship verification model, and achieve maximum attack success rates of 92\% and 78\% for both obfuscation and impersonation attacks, respectively. 

---
# LookAhead Tuning: Safer Language Models via Partial Answer Previews 

**Authors**: Kangwei Liu, Mengru Wang, Yujie Luo, Lin Yuan, Mengshu Sun, Ningyu Zhang, Lei Liang, Zhiqiang Zhang, Jun Zhou, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.19041)  

**Abstract**: Fine-tuning enables large language models (LLMs) to adapt to specific domains, but often undermines their previously established safety alignment. To mitigate the degradation of model safety during fine-tuning, we introduce LookAhead Tuning, which comprises two simple, low-resource, and effective data-driven methods that modify training data by previewing partial answer prefixes. Both methods aim to preserve the model's inherent safety mechanisms by minimizing perturbations to initial token distributions. Comprehensive experiments demonstrate that LookAhead Tuning effectively maintains model safety without sacrificing robust performance on downstream tasks. Our findings position LookAhead Tuning as a reliable and efficient solution for the safe and effective adaptation of LLMs. Code is released at this https URL. 

---
# Machine-assisted writing evaluation: Exploring pre-trained language models in analyzing argumentative moves 

**Authors**: Wenjuan Qin, Weiran Wang, Yuming Yang, Tao Gui  

**Link**: [PDF](https://arxiv.org/pdf/2503.19279)  

**Abstract**: The study investigates the efficacy of pre-trained language models (PLMs) in analyzing argumentative moves in a longitudinal learner corpus. Prior studies on argumentative moves often rely on qualitative analysis and manual coding, limiting their efficiency and generalizability. The study aims to: 1) to assess the reliability of PLMs in analyzing argumentative moves; 2) to utilize PLM-generated annotations to illustrate developmental patterns and predict writing quality. A longitudinal corpus of 1643 argumentative texts from 235 English learners in China is collected and annotated into six move types: claim, data, counter-claim, counter-data, rebuttal, and non-argument. The corpus is divided into training, validation, and application sets annotated by human experts and PLMs. We use BERT as one of the implementations of PLMs. The results indicate a robust reliability of PLMs in analyzing argumentative moves, with an overall F1 score of 0.743, surpassing existing models in the field. Additionally, PLM-labeled argumentative moves effectively capture developmental patterns and predict writing quality. Over time, students exhibit an increase in the use of data and counter-claims and a decrease in non-argument moves. While low-quality texts are characterized by a predominant use of claims and data supporting only oneside position, mid- and high-quality texts demonstrate an integrative perspective with a higher ratio of counter-claims, counter-data, and rebuttals. This study underscores the transformative potential of integrating artificial intelligence into language education, enhancing the efficiency and accuracy of evaluating students' writing. The successful application of PLMs can catalyze the development of educational technology, promoting a more data-driven and personalized learning environment that supports diverse educational needs. 

---
# QUAD: Quantization and Parameter-Efficient Tuning of LLM with Activation Decomposition 

**Authors**: Yuxuan Hu, Xiaodong Chen, Cuiping Li, Hong Chen, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.19353)  

**Abstract**: Large Language Models (LLMs) excel in diverse applications but suffer inefficiency due to massive scale. While quantization reduces computational costs, existing methods degrade accuracy in medium-sized LLMs (e.g., Llama-3-8B) due to activation outliers. To address this, we propose QUAD (Quantization with Activation Decomposition), a framework leveraging Singular Value Decomposition (SVD) to suppress activation outliers for effective 4-bit quantization. QUAD estimates activation singular vectors offline using calibration data to construct an orthogonal transformation matrix P, shifting outliers to additional dimensions in full precision while quantizing rest components to 4-bit. Additionally, QUAD enables parameter-efficient fine-tuning via adaptable full-precision outlier weights, narrowing the accuracy gap between quantized and full-precision models. Experiments demonstrate that QUAD achieves 94% ~ 96% accuracy under W4A4 quantization and 98% accuracy with W4A4/A8 and parameter-efficient fine-tuning for Llama-3 and Qwen-2.5 models. Our code is available at \href{this https URL}{repository}. 

---
# ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning 

**Authors**: Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Fan Yang, Zenan Zhou, Weipeng Chen, Haofen Wang, Jeff Z. Pan, Wen Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.19470)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities in reasoning, exemplified by the success of OpenAI-o1 and DeepSeek-R1. However, integrating reasoning with external search processes remains challenging, especially for complex multi-hop questions requiring multiple retrieval steps. We propose ReSearch, a novel framework that trains LLMs to Reason with Search via reinforcement learning without using any supervised data on reasoning steps. Our approach treats search operations as integral components of the reasoning chain, where when and how to perform searches is guided by text-based thinking, and search results subsequently influence further reasoning. We train ReSearch on Qwen2.5-7B(-Instruct) and Qwen2.5-32B(-Instruct) models and conduct extensive experiments. Despite being trained on only one dataset, our models demonstrate strong generalizability across various benchmarks. Analysis reveals that ReSearch naturally elicits advanced reasoning capabilities such as reflection and self-correction during the reinforcement learning process. 

---
# SRMIR: Shadow Reward Models Based on Introspective Reasoning for LLM Alignment 

**Authors**: Ruoxi Cheng, Shuirong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2503.18991)  

**Abstract**: Aligning large language models (LLMs) with human preferences and values is vital for application. However, current alignment methods face three main limitations: (1) reliance on costly human annotation; (2) alignment tax; (3) shallow alignment vulnerable to jailbreak attacks. Additionally, current alignment datasets often suffer from uneven distributions, leading to overrepresentation of some topics and neglect of others. To address these issues, we propose SRMIR (Shadow Reward Models Based on Introspective Reasoning), inspired by shadow models in membership inference attacks. We first construct a balanced safety Chain of Draft (CoD) dataset across $7$ harmful types with structured prompt leveraging the introspective reasoning capabilities of LLMs, then train a set of specialized reward models to guide policy optimization through Group Relative Policy Optimization (GRPO). We apply two strategies, linear combination and categorized approach, to integrate shadow reward models for policy optimization. By comparison, we find that the latter achieves superior alignment despite higher computational costs. Experiments across several LLMs demonstrate SRMIR significantly outperforms existing methods. 

---
# Innate Reasoning is Not Enough: In-Context Learning Enhances Reasoning Large Language Models with Less Overthinking 

**Authors**: Yuyao Ge, Shenghua Liu, Yiwei Wang, Lingrui Mei, Lizhe Chen, Baolong Bi, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.19602)  

**Abstract**: Recent advances in Large Language Models (LLMs) have introduced Reasoning Large Language Models (RLLMs), which employ extended thinking processes with reflection and self-correction capabilities, demonstrating the effectiveness of test-time scaling. RLLMs exhibit innate Chain-of-Thought (CoT) reasoning capability obtained from training, leading to a natural question: "Is CoT prompting, a popular In-Context Learning (ICL) method for chat LLMs, necessary to enhance the reasoning capability of RLLMs?" In this work, we present the first comprehensive analysis of the impacts of Zero-shot CoT and Few-shot CoT on RLLMs across mathematical reasoning tasks. We examine models ranging from 1.5B to 32B parameters, finding that contrary to concerns, CoT prompting significantly enhances RLLMs' performance in most scenarios. Our results reveal distinct patterns: large-capacity models show minimal improvement on simple tasks but substantial gains on complex problems, while smaller models exhibit the opposite behavior. Further analysis demonstrates that CoT prompting effectively controls the distribution of the numbers of thinking tokens and reasoning steps, reducing excessive reflections by approximately 90% in some cases. Moreover, attention logits analysis reveals the RLLMs' overfitting to reflection-related words, which is mitigated by external CoT guidance. Notably, our experiments indicate that for RLLMs, one-shot CoT consistently yields superior performance compared to Few-shot CoT approaches. Our findings provide important insights for optimizing RLLMs' performance through appropriate prompting strategies. 

---
# LLMs as Planning Modelers: A Survey for Leveraging Large Language Models to Construct Automated Planning Models 

**Authors**: Marcus Tantakoun, Xiaodan Zhu, Christian Muise  

**Link**: [PDF](https://arxiv.org/pdf/2503.18971)  

**Abstract**: Large Language Models (LLMs) excel in various natural language tasks but often struggle with long-horizon planning problems requiring structured reasoning. This limitation has drawn interest in integrating neuro-symbolic approaches within the Automated Planning (AP) and Natural Language Processing (NLP) communities. However, identifying optimal AP deployment frameworks can be daunting. This paper aims to provide a timely survey of the current research with an in-depth analysis, positioning LLMs as tools for extracting and refining planning models to support reliable AP planners. By systematically reviewing the current state of research, we highlight methodologies, and identify critical challenges and future directions, hoping to contribute to the joint research on NLP and Automated Planning. 

---
# Process or Result? Manipulated Ending Tokens Can Mislead Reasoning LLMs to Ignore the Correct Reasoning Steps 

**Authors**: Yu Cui, Bryan Hooi, Yujun Cai, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.19326)  

**Abstract**: Recent reasoning large language models (LLMs) have demonstrated remarkable improvements in mathematical reasoning capabilities through long Chain-of-Thought. The reasoning tokens of these models enable self-correction within reasoning chains, enhancing robustness. This motivates our exploration: how vulnerable are reasoning LLMs to subtle errors in their input reasoning chains? We introduce "Compromising Thought" (CPT), a vulnerability where models presented with reasoning tokens containing manipulated calculation results tend to ignore correct reasoning steps and adopt incorrect results instead. Through systematic evaluation across multiple reasoning LLMs, we design three increasingly explicit prompting methods to measure CPT resistance, revealing that models struggle significantly to identify and correct these manipulations. Notably, contrary to existing research suggesting structural alterations affect model performance more than content modifications, we find that local ending token manipulations have greater impact on reasoning outcomes than structural changes. Moreover, we discover a security vulnerability in DeepSeek-R1 where tampered reasoning tokens can trigger complete reasoning cessation. Our work enhances understanding of reasoning robustness and highlights security considerations for reasoning-intensive applications. 

---
# MedAgent-Pro: Towards Multi-modal Evidence-based Medical Diagnosis via Reasoning Agentic Workflow 

**Authors**: Ziyue Wang, Junde Wu, Chang Han Low, Yueming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2503.18968)  

**Abstract**: Developing reliable AI systems to assist human clinicians in multi-modal medical diagnosis has long been a key objective for researchers. Recently, Multi-modal Large Language Models (MLLMs) have gained significant attention and achieved success across various domains. With strong reasoning capabilities and the ability to perform diverse tasks based on user instructions, they hold great potential for enhancing medical diagnosis. However, directly applying MLLMs to the medical domain still presents challenges. They lack detailed perception of visual inputs, limiting their ability to perform quantitative image analysis, which is crucial for medical diagnostics. Additionally, MLLMs often exhibit hallucinations and inconsistencies in reasoning, whereas clinical diagnoses must adhere strictly to established criteria. To address these challenges, we propose MedAgent-Pro, an evidence-based reasoning agentic system designed to achieve reliable, explainable, and precise medical diagnoses. This is accomplished through a hierarchical workflow: at the task level, knowledge-based reasoning generate reliable diagnostic plans for specific diseases following retrieved clinical criteria. While at the case level, multiple tool agents process multi-modal inputs, analyze different indicators according to the plan, and provide a final diagnosis based on both quantitative and qualitative evidence. Comprehensive experiments on both 2D and 3D medical diagnosis tasks demonstrate the superiority and effectiveness of MedAgent-Pro, while case studies further highlight its reliability and interpretability. The code is available at this https URL. 

---
# PAVE: Patching and Adapting Video Large Language Models 

**Authors**: Zhuoming Liu, Yiquan Li, Khoi Duc Nguyen, Yiwu Zhong, Yin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.19794)  

**Abstract**: Pre-trained video large language models (Video LLMs) exhibit remarkable reasoning capabilities, yet adapting these models to new tasks involving additional modalities or data types (e.g., audio or 3D information) remains challenging. In this paper, we present PAVE, a flexible framework for adapting pre-trained Video LLMs to downstream tasks with side-channel signals, such as audio, 3D cues, or multi-view videos. PAVE introduces lightweight adapters, referred to as "patches," which add a small number of parameters and operations to a base model without modifying its architecture or pre-trained weights. In doing so, PAVE can effectively adapt the pre-trained base model to support diverse downstream tasks, including audio-visual question answering, 3D reasoning, multi-view video recognition, and high frame rate video understanding. Across these tasks, PAVE significantly enhances the performance of the base model, surpassing state-of-the-art task-specific models while incurring a minor cost of ~0.1% additional FLOPs and parameters. Further, PAVE supports multi-task learning and generalizes well across different Video LLMs. Our code is available at this https URL. 

---
# HoarePrompt: Structural Reasoning About Program Correctness in Natural Language 

**Authors**: Dimitrios Stamatios Bouras, Yihan Dai, Tairan Wang, Yingfei Xiong, Sergey Mechtaev  

**Link**: [PDF](https://arxiv.org/pdf/2503.19599)  

**Abstract**: While software requirements are often expressed in natural language, verifying the correctness of a program against natural language requirements is a hard and underexplored problem. Large language models (LLMs) are promising candidates for addressing this challenge, however our experience shows that they are ineffective in this task, often failing to detect even straightforward bugs. To address this gap, we introduce HoarePrompt, a novel approach that adapts fundamental ideas from program analysis and verification to natural language artifacts. Drawing inspiration from the strongest postcondition calculus, HoarePrompt employs a systematic, step-by-step process in which an LLM generates natural language descriptions of reachable program states at various points in the code. To manage loops, we propose few-shot-driven k-induction, an adaptation of the k-induction method widely used in model checking. Once program states are described, HoarePrompt leverages the LLM to assess whether the program, annotated with these state descriptions, conforms to the natural language requirements. For evaluating the quality of classifiers of program correctness with respect to natural language requirements, we constructed CoCoClaNeL, a challenging dataset of solutions to programming competition problems. Our experiments show that HoarePrompt improves the MCC by 62% compared to directly using Zero-shot-CoT prompts for correctness classification. Furthermore, HoarePrompt outperforms a classifier that assesses correctness via LLM-based test generation by increasing the MCC by 93%. The inductive reasoning mechanism contributes a 28% boost to MCC, underscoring its effectiveness in managing loops. 

---
# VecTrans: LLM Transformation Framework for Better Auto-vectorization on High-performance CPU 

**Authors**: Zhongchun Zheng, Long Cheng, Lu Li, Rodrigo C. O. Rocha, Tianyi Liu, Wei Wei, Xianwei Zhang, Yaoqing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.19449)  

**Abstract**: Large language models (LLMs) have demonstrated great capabilities in code generation, yet their effective application in compiler optimizations remains an open challenge due to issues such as hallucinations and a lack of domain-specific reasoning. Vectorization, a crucial optimization for enhancing code performance, often fails because of the compiler's inability to recognize complex code patterns, which commonly require extensive empirical expertise. LLMs, with their ability to capture intricate patterns, thus providing a promising solution to this challenge. This paper presents VecTrans, a novel framework that leverages LLMs to enhance compiler-based code vectorization. VecTrans first employs compiler analysis to identify potentially vectorizable code regions. It then utilizes an LLM to refactor these regions into patterns that are more amenable to the compiler's auto-vectorization. To ensure semantic correctness, VecTrans further integrates a hybrid validation mechanism at the intermediate representation (IR) level. With the above efforts, VecTrans combines the adaptability of LLMs with the precision of compiler vectorization, thereby effectively opening up the vectorization opportunities. Experimental results show that among all 50 TSVC functions unvectorizable by Clang, GCC, and BiShengCompiler, VecTrans successfully vectorizes 23 cases (46%) and achieves an average speedup of 2.02x, greatly surpassing state-of-the-art performance. 

---
# LLM Benchmarking with LLaMA2: Evaluating Code Development Performance Across Multiple Programming Languages 

**Authors**: Patrick Diehl, Nojoud Nader, Maxim Moraru, Steven R. Brandt  

**Link**: [PDF](https://arxiv.org/pdf/2503.19217)  

**Abstract**: The rapid evolution of large language models (LLMs) has opened new possibilities for automating various tasks in software development. This paper evaluates the capabilities of the Llama 2-70B model in automating these tasks for scientific applications written in commonly used programming languages. Using representative test problems, we assess the model's capacity to generate code, documentation, and unit tests, as well as its ability to translate existing code between commonly used programming languages. Our comprehensive analysis evaluates the compilation, runtime behavior, and correctness of the generated and translated code. Additionally, we assess the quality of automatically generated code, documentation and unit tests. Our results indicate that while Llama 2-70B frequently generates syntactically correct and functional code for simpler numerical tasks, it encounters substantial difficulties with more complex, parallelized, or distributed computations, requiring considerable manual corrections. We identify key limitations and suggest areas for future improvements to better leverage AI-driven automation in scientific computing workflows. 

---
# Data-centric Federated Graph Learning with Large Language Models 

**Authors**: Bo Yan, Zhongjian Zhang, Huabin Sun, Mengmei Zhang, Yang Cao, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.19455)  

**Abstract**: In federated graph learning (FGL), a complete graph is divided into multiple subgraphs stored in each client due to privacy concerns, and all clients jointly train a global graph model by only transmitting model parameters. A pain point of FGL is the heterogeneity problem, where nodes or structures present non-IID properties among clients (e.g., different node label distributions), dramatically undermining the convergence and performance of FGL. To address this, existing efforts focus on design strategies at the model level, i.e., they design models to extract common knowledge to mitigate heterogeneity. However, these model-level strategies fail to fundamentally address the heterogeneity problem as the model needs to be designed from scratch when transferring to other tasks. Motivated by large language models (LLMs) having achieved remarkable success, we aim to utilize LLMs to fully understand and augment local text-attributed graphs, to address data heterogeneity at the data level. In this paper, we propose a general framework LLM4FGL that innovatively decomposes the task of LLM for FGL into two sub-tasks theoretically. Specifically, for each client, it first utilizes the LLM to generate missing neighbors and then infers connections between generated nodes and raw nodes. To improve the quality of generated nodes, we design a novel federated generation-and-reflection mechanism for LLMs, without the need to modify the parameters of the LLM but relying solely on the collective feedback from all clients. After neighbor generation, all the clients utilize a pre-trained edge predictor to infer the missing edges. Furthermore, our framework can seamlessly integrate as a plug-in with existing FGL methods. Experiments on three real-world datasets demonstrate the superiority of our method compared to advanced baselines. 

---
# A-MESS: Anchor based Multimodal Embedding with Semantic Synchronization for Multimodal Intent Recognition 

**Authors**: Yaomin Shen, Xiaojian Lin, Wei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.19474)  

**Abstract**: In the domain of multimodal intent recognition (MIR), the objective is to recognize human intent by integrating a variety of modalities, such as language text, body gestures, and tones. However, existing approaches face difficulties adequately capturing the intrinsic connections between the modalities and overlooking the corresponding semantic representations of intent. To address these limitations, we present the Anchor-based Mul- timodal Embedding with Semantic Synchronization (A-MESS) framework. We first design an Anchor-based Multimodal Embed- ding (A-ME) module that employs an anchor-based embedding fusion mechanism to integrate multimodal inputs. Furthermore, we develop a Semantic Synchronization (SS) strategy with the Triplet Contrastive Learning pipeline, which optimizes the pro- cess by synchronizing multimodal representation with label de- scriptions produced by the large language model. Comprehensive experiments indicate that our A-MESS achieves state-of-the-art and provides substantial insight into multimodal representation and downstream tasks. 

---
# Context-Aware Semantic Segmentation: Enhancing Pixel-Level Understanding with Large Language Models for Advanced Vision Applications 

**Authors**: Ben Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2503.19276)  

**Abstract**: Semantic segmentation has made significant strides in pixel-level image understanding, yet it remains limited in capturing contextual and semantic relationships between objects. Current models, such as CNN and Transformer-based architectures, excel at identifying pixel-level features but fail to distinguish semantically similar objects (e.g., "doctor" vs. "nurse" in a hospital scene) or understand complex contextual scenarios (e.g., differentiating a running child from a regular pedestrian in autonomous driving). To address these limitations, we proposed a novel Context-Aware Semantic Segmentation framework that integrates Large Language Models (LLMs) with state-of-the-art vision backbones. Our hybrid model leverages the Swin Transformer for robust visual feature extraction and GPT-4 for enriching semantic understanding through text embeddings. A Cross-Attention Mechanism is introduced to align vision and language features, enabling the model to reason about context more effectively. Additionally, Graph Neural Networks (GNNs) are employed to model object relationships within the scene, capturing dependencies that are overlooked by traditional models. Experimental results on benchmark datasets (e.g., COCO, Cityscapes) demonstrate that our approach outperforms the existing methods in both pixel-level accuracy (mIoU) and contextual understanding (mAP). This work bridges the gap between vision and language, paving the path for more intelligent and context-aware vision systems in applications including autonomous driving, medical imaging, and robotics. 

---
# Option Discovery Using LLM-guided Semantic Hierarchical Reinforcement Learning 

**Authors**: Chak Lam Shek, Pratap Tokekar  

**Link**: [PDF](https://arxiv.org/pdf/2503.19007)  

**Abstract**: Large Language Models (LLMs) have shown remarkable promise in reasoning and decision-making, yet their integration with Reinforcement Learning (RL) for complex robotic tasks remains underexplored. In this paper, we propose an LLM-guided hierarchical RL framework, termed LDSC, that leverages LLM-driven subgoal selection and option reuse to enhance sample efficiency, generalization, and multi-task adaptability. Traditional RL methods often suffer from inefficient exploration and high computational cost. Hierarchical RL helps with these challenges, but existing methods often fail to reuse options effectively when faced with new tasks. To address these limitations, we introduce a three-stage framework that uses LLMs for subgoal generation given natural language description of the task, a reusable option learning and selection method, and an action-level policy, enabling more effective decision-making across diverse tasks. By incorporating LLMs for subgoal prediction and policy guidance, our approach improves exploration efficiency and enhances learning performance. On average, LDSC outperforms the baseline by 55.9\% in average reward, demonstrating its effectiveness in complex RL settings. More details and experiment videos could be found in \href{this https URL}{this link\footnote{this https URL}}. 

---
# LLMs in the Classroom: Outcomes and Perceptions of Questions Written with the Aid of AI 

**Authors**: Gavin Witsken, Igor Crk, Eren Gultepe  

**Link**: [PDF](https://arxiv.org/pdf/2503.18995)  

**Abstract**: We randomly deploy questions constructed with and without use of the LLM tool and gauge the ability of the students to correctly answer, as well as their ability to correctly perceive the difference between human-authored and LLM-authored questions. In determining whether the questions written with the aid of ChatGPT were consistent with the instructor's questions and source text, we computed representative vectors of both the human and ChatGPT questions using SBERT and compared cosine similarity to the course textbook. A non-significant Mann-Whitney U test (z = 1.018, p = .309) suggests that students were unable to perceive whether questions were written with or without the aid of ChatGPT. However, student scores on LLM-authored questions were almost 9% lower (z = 2.702, p < .01). This result may indicate that either the AI questions were more difficult or that the students were more familiar with the instructor's style of questions. Overall, the study suggests that while there is potential for using LLM tools to aid in the construction of assessments, care must be taken to ensure that the questions are fair, well-composed, and relevant to the course material. 

---
# SplitFrozen: Split Learning with Device-side Model Frozen for Fine-Tuning LLM on Heterogeneous Resource-Constrained Devices 

**Authors**: Jian Ma, Xinchen Lyu, Jun Jiang, Qimei Cui, Haipeng Yao, Xiaofeng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.18986)  

**Abstract**: Fine-tuning large language models (LLMs) on private, on-device data can empower tailored personalized AI agents. However, fine-tuning LLMs on resource-constrained edge devices faces significant challenges, including excessive computation overhead, device heterogeneity, and data imbalance. This paper proposes SplitFrozen, a split learning framework that enables efficient LLM fine-tuning by strategically freezing device-side model layers while centralizing parameter-efficient fine-tuning on the server. Our framework partitions LLMs into device-side frozen layers and server-side fine-tuning layers, where heterogeneous resource-constrained devices execute only forward propagation. To minimize server-side training costs, we integrate Low-Rank Adaptation (LoRA) into the server-side layers. A pipeline parallelism strategy further optimizes training efficiency by decoupling device-server computations and leveraging decomposed backward propagation. Experiments on GPT-2 with the MRPC, MNLI-matched, and SST-2 datasets demonstrate that SplitFrozen outperforms FedLoRA and SplitLoRA by 69.4\% model accuracy under extremely imbalanced data, while reducing up to 86.8\% device-side computations and 50.2\% total training time. Experiments also validate the scalability of SplitFrozen on content generation task using Llama-3.2 model on GSM8K dataset. 

---
