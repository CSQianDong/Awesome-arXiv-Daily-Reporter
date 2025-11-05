# Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities 

**Authors**: Amanda Bertsch, Adithya Pratapa, Teruko Mitamura, Graham Neubig, Matthew R. Gormley  

**Link**: [PDF](https://arxiv.org/pdf/2511.02817)  

**Abstract**: As model context lengths continue to grow, concerns about whether models effectively use the full context length have persisted. While several carefully designed long-context evaluations have recently been released, these evaluations tend to rely on retrieval from one or more sections of the context, which allows nearly all of the context tokens to be disregarded as noise. This represents only one type of task that might be performed with long context. We introduce Oolong, a benchmark of long-context reasoning tasks that require analyzing individual chunks of text on an atomic level, and then aggregating these analyses to answer distributional questions. Oolong is separated into two task sets: Oolong-synth, a set of naturalistic synthetic tasks, where we can easily ablate components of the reasoning problem; and Oolong-real, a downstream setting which requires reasoning over real-world conversational data. Oolong requires models to reason over large quantities of examples, to perform both classification and counting in-context, and to reason over temporal and user relations. Even frontier models struggle on Oolong, with GPT-5, Claude-Sonnet-4, and Gemini-2.5-Pro all achieving less than 50% accuracy on both splits at 128K. We release the data and evaluation harness for Oolong to enable further development of models that can reason over large quantities of text. 

---
# MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End Reinforcement Learning 

**Authors**: Qianhao Yuan, Jie Lou, Zichao Li, Jiawei Chen, Yaojie Lu, Hongyu Lin, Le Sun, Debing Zhang, Xianpei Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.02805)  

**Abstract**: Typical search agents concatenate the entire interaction history into the LLM context, preserving information integrity but producing long, noisy contexts, resulting in high computation and memory costs. In contrast, using only the current turn avoids this overhead but discards essential information. This trade-off limits the scalability of search agents. To address this challenge, we propose MemSearcher, an agent workflow that iteratively maintains a compact memory and combines the current turn with it. At each turn, MemSearcher fuses the user's question with the memory to generate reasoning traces, perform search actions, and update memory to retain only information essential for solving the task. This design stabilizes context length across multi-turn interactions, improving efficiency without sacrificing accuracy. To optimize this workflow, we introduce multi-context GRPO, an end-to-end RL framework that jointly optimize reasoning, search strategies, and memory management of MemSearcher Agents. Specifically, multi-context GRPO samples groups of trajectories under different contexts and propagates trajectory-level advantages across all conversations within them. Trained on the same dataset as Search-R1, MemSearcher achieves significant improvements over strong baselines on seven public benchmarks: +11% on Qwen2.5-3B-Instruct and +12% on Qwen2.5-7B-Instruct relative average gains. Notably, the 3B-based MemSearcher even outperforms 7B-based baselines, demonstrating that striking a balance between information integrity and efficiency yields both higher accuracy and lower computational overhead. The code and models will be publicly available at this https URL 

---
# Beyond Single Embeddings: Capturing Diverse Targets with Multi-Query Retrieval 

**Authors**: Hung-Ting Chen, Xiang Liu, Shauli Ravfogel, Eunsol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2511.02770)  

**Abstract**: Most text retrievers generate \emph{one} query vector to retrieve relevant documents. Yet, the conditional distribution of relevant documents for the query may be multimodal, e.g., representing different interpretations of the query. We first quantify the limitations of existing retrievers. All retrievers we evaluate struggle more as the distance between target document embeddings grows. To address this limitation, we develop a new retriever architecture, \emph{A}utoregressive \emph{M}ulti-\emph{E}mbedding \emph{R}etriever (AMER). Our model autoregressively generates multiple query vectors, and all the predicted query vectors are used to retrieve documents from the corpus. We show that on the synthetic vectorized data, the proposed method could capture multiple target distributions perfectly, showing 4x better performance than single embedding model. We also fine-tune our model on real-world multi-answer retrieval datasets and evaluate in-domain. AMER presents 4 and 21\% relative gains over single-embedding baselines on two datasets we evaluate on. Furthermore, we consistently observe larger gains on the subset of dataset where the embeddings of the target documents are less similar to each other. We demonstrate the potential of using a multi-query vector retriever and open up a new direction for future work. 

---
# Controlling Performance and Budget of a Centralized Multi-agent LLM System with Reinforcement Learning 

**Authors**: Bowen Jin, TJ Collins, Donghan Yu, Mert Cemri, Shenao Zhang, Mengyu Li, Jay Tang, Tian Qin, Zhiyang Xu, Jiarui Lu, Guoli Yin, Jiawei Han, Zirui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02755)  

**Abstract**: Large language models (LLMs) exhibit complementary strengths across domains and come with varying inference costs, motivating the design of multi-agent LLM systems where specialized models collaborate efficiently. Existing approaches predominantly rely on decentralized frameworks, which invoke multiple LLMs for every input and thus lead to substantial and uncontrolled inference costs. In this work, we introduce a centralized multi-LLM framework, where a controller LLM selectively coordinates a pool of expert models in a cost-efficient and cost-controllable manner. We formulate this coordination problem as reinforcement learning with dual objectives: maximizing task performance while minimizing the overall inference cost. In addition, we expect the multi-agent system to have adapted behavior with different budget conditions during inference. To this end, we propose CoRL, a reinforcement learning framework that optimizes the performance cost trade-off in a controllable multi-budget setting. Experiments on four diverse benchmarks demonstrate that CoRL enables a single system to surpass the best expert LLM under high-budget settings, while maintaining strong performance in more economical low-budget modes, highlighting the effectiveness of centralized coordination for scalable and cost-efficient multi-agent LLM systems. 

---
# AI Diffusion in Low Resource Language Countries 

**Authors**: Amit Misra, Syed Waqas Zamir, Wassim Hamidouche, Inbal Becker-Reshef, Juan Lavista Ferres  

**Link**: [PDF](https://arxiv.org/pdf/2511.02752)  

**Abstract**: Artificial intelligence (AI) is diffusing globally at unprecedented speed, but adoption remains uneven. Frontier Large Language Models (LLMs) are known to perform poorly on low-resource languages due to data scarcity. We hypothesize that this performance deficit reduces the utility of AI, thereby slowing adoption in Low-Resource Language Countries (LRLCs). To test this, we use a weighted regression model to isolate the language effect from socioeconomic and demographic factors, finding that LRLCs have a share of AI users that is approximately 20% lower relative to their baseline. These results indicate that linguistic accessibility is a significant, independent barrier to equitable AI diffusion. 

---
# PragExTra: A Multilingual Corpus of Pragmatic Explicitation in Translation 

**Authors**: Doreen Osmelak, Koel Dutta Chowdhury, Uliana Sentsova, Cristina España-Bonet, Josef van Genabith  

**Link**: [PDF](https://arxiv.org/pdf/2511.02721)  

**Abstract**: Translators often enrich texts with background details that make implicit cultural meanings explicit for new audiences. This phenomenon, known as pragmatic explicitation, has been widely discussed in translation theory but rarely modeled computationally. We introduce PragExTra, the first multilingual corpus and detection framework for pragmatic explicitation. The corpus covers eight language pairs from TED-Multi and Europarl and includes additions such as entity descriptions, measurement conversions, and translator remarks. We identify candidate explicitation cases through null alignments and refined using active learning with human annotation. Our results show that entity and system-level explicitations are most frequent, and that active learning improves classifier accuracy by 7-8 percentage points, achieving up to 0.88 accuracy and 0.82 F1 across languages. PragExTra establishes pragmatic explicitation as a measurable, cross-linguistic phenomenon and takes a step towards building culturally aware machine translation. Keywords: translation, multilingualism, explicitation 

---
# Optimal Singular Damage: Efficient LLM Inference in Low Storage Regimes 

**Authors**: Mohammadsajad Alipour, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2511.02681)  

**Abstract**: Large language models (LLMs) are increasingly prevalent across diverse applications. However, their enormous size limits storage and processing capabilities to a few well-resourced stakeholders. As a result, most applications rely on pre-trained LLMs, fine-tuned for specific tasks. However, even storing the fine-tuned versions of these models remains a significant challenge due to the wide range of tasks they address. Recently, studies show that fine-tuning these models primarily affects a small fraction of parameters, highlighting the need for more efficient storage of fine-tuned models. This paper focuses on efficient storage of parameter updates in pre-trained models after fine-tuning. To address this challenge, we leverage the observation that fine-tuning updates are both low-rank and sparse, which can be utilized for storage efficiency. However, using only low-rank approximation or sparsification may discard critical singular components that enhance model expressivity. We first observe that given the same memory budget, sparsified low-rank approximations with larger ranks outperform standard low-rank approximations with smaller ranks. Building on this, we propose our method, optimal singular damage, that selectively sparsifies low-rank approximated updates by leveraging the interleaved importance of singular vectors, ensuring that the most impactful components are retained. We demonstrate through extensive experiments that our proposed methods lead to significant storage efficiency and superior accuracy within the same memory budget compared to employing the low-rank approximation or sparsification individually. 

---
# Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis, Solution, and Interpretation 

**Authors**: Renfei Dang, Peng Hu, Changjiang Gao, Shujian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02626)  

**Abstract**: Previous studies show that introducing new knowledge during large language models (LLMs) fine-tuning can lead to the generation of erroneous output when tested on known information, thereby triggering factual hallucinations. However, existing studies have not deeply investigated the specific manifestations and underlying mechanisms of these hallucinations. Our work addresses this gap by designing a controlled dataset Biography-Reasoning, and conducting a fine-grained analysis across multiple knowledge types and two task types, including knowledge question answering (QA) and knowledge reasoning tasks. We find that when fine-tuned on a dataset in which a specific knowledge type consists entirely of new knowledge, LLMs exhibit significantly increased hallucination tendencies. This suggests that the high unfamiliarity of a particular knowledge type, rather than the overall proportion of new knowledge, is a stronger driver of hallucinations, and these tendencies can even affect other knowledge types in QA tasks. To mitigate such factual hallucinations, we propose KnownPatch, which patches a small number of known knowledge samples in the later stages of training, effectively alleviating new-knowledge-induced hallucinations. Through attention analysis, we find that learning new knowledge reduces the model's attention to key entities in the question, thus causing excessive focus on the surrounding context, which may increase the risk of hallucination. Moreover, the attention pattern can propagate to similar contexts, facilitating the spread of hallucinations to textually similar questions. Our method effectively mitigates the disruption of new knowledge learning to the model's attention on key entities, accompanied by improved performance. 

---
# The Realignment Problem: When Right becomes Wrong in LLMs 

**Authors**: Aakash Sen Sharma, Debdeep Sanyal, Vivek Srivastava, Shirish Karande, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2511.02623)  

**Abstract**: The alignment of Large Language Models (LLMs) with human values is central to their safe deployment, yet current practice produces static, brittle, and costly-to-maintain models that fail to keep pace with evolving norms and policies. This misalignment, which we term the Alignment-Reality Gap, poses a growing challenge for reliable long-term use. Existing remedies are inadequate: large-scale re-annotation is economically prohibitive, and standard unlearning methods act as blunt instruments that erode utility rather than enable precise policy updates. We introduce TRACE (Triage and Re-align by Alignment Conflict Evaluation), a framework for principled unlearning that reconceives re-alignment as a programmatic policy application problem. TRACE programmatically triages existing preference data against a new policy, identifies high-impact conflicts via a alignment impact score, and applies a hybrid optimization that cleanly inverts, discards, or preserves preferences while safeguarding model performance. Empirical results show that TRACE achieves robust re-alignment across diverse model families (Qwen2.5-7B, Gemma-2-9B, Llama-3.1-8B). On both synthetic benchmarks and the PKU-SafeRLHF dataset under complex policy shift, TRACE enforces new principles without degrading general capabilities. Our work establishes a scalable, dynamic, and cost-effective paradigm for maintaining LLM alignment, providing a foundation for sustainable and responsible AI deployment. 

---
# CGES: Confidence-Guided Early Stopping for Efficient and Accurate Self-Consistency 

**Authors**: Ehsan Aghazadeh, Ahmad Ghasemi, Hedyeh Beyhaghi, Hossein Pishro-Nik  

**Link**: [PDF](https://arxiv.org/pdf/2511.02603)  

**Abstract**: Large language models (LLMs) are often queried multiple times at test time, with predictions aggregated by majority vote. While effective, this self-consistency strategy (arXiv:2203.11171) requires a fixed number of calls and can fail when the correct answer is rare. We introduce Confidence-Guided Early Stopping (CGES), a Bayesian framework that forms posteriors over candidate answers using scalar confidence signals derived from token probabilities or reward models. CGES adaptively halts sampling once the posterior mass of a candidate exceeds a threshold. We provide theoretical guarantees for both perfectly calibrated confidences and realistic noisy confidence signals. Across five reasoning benchmarks, CGES reduces the average number of model calls by about 69 percent (for example, from 16.0 to 4.9) while matching the accuracy of self-consistency within 0.06 percentage points. 

---
# Next Token Knowledge Tracing: Exploiting Pretrained LLM Representations to Decode Student Behaviour 

**Authors**: Max Norris, Kobi Gal, Sahan Bulathwela  

**Link**: [PDF](https://arxiv.org/pdf/2511.02599)  

**Abstract**: Modelling student knowledge is a key challenge when leveraging AI in education, with major implications for personalised learning. The Knowledge Tracing (KT) task aims to predict how students will respond to educational questions in learning environments, based on their prior interactions. Existing KT models typically use response correctness along with metadata like skill tags and timestamps, often overlooking the question text, which is an important source of pedagogical insight. This omission poses a lost opportunity while limiting predictive performance. We propose Next Token Knowledge Tracing (NTKT), a novel approach that reframes KT as a next-token prediction task using pretrained Large Language Models (LLMs). NTKT represents both student histories and question content as sequences of text, allowing LLMs to learn patterns in both behaviour and language. Our series of experiments significantly improves performance over state-of-the-art neural KT models and generalises much better to cold-start questions and users. These findings highlight the importance of question content in KT and demonstrate the benefits of leveraging pretrained representations of LLMs to model student learning more effectively. 

---
# The Analysis of Lexical Errors in Machine Translation from English into Romanian 

**Authors**: Angela Stamatie  

**Link**: [PDF](https://arxiv.org/pdf/2511.02587)  

**Abstract**: The research explores error analysis in the performance of translating by Machine Translation from English into Romanian, and it focuses on lexical errors found in texts which include official information, provided by the World Health Organization (WHO), the Gavi Organization, by the patient information leaflet (the information about the active ingredients of the vaccines or the medication, the indications, the dosage instructions, the storage instructions, the side effects and warning, etc.). All of these texts are related to Covid-19 and have been translated by Google Translate, a multilingual Machine Translation that was created by Google. In the last decades, Google has actively worked to develop a more accurate and fluent automatic translation system. This research, specifically focused on improving Google Translate, aims to enhance the overall quality of Machine Translation by achieving better lexical selection and by reducing errors. The investigation involves a comprehensive analysis of 230 texts that have been translated from English into Romanian. 

---
# Smart-Hiring: An Explainable end-to-end Pipeline for CV Information Extraction and Job Matching 

**Authors**: Kenza Khelkhal, Dihia Lanasri  

**Link**: [PDF](https://arxiv.org/pdf/2511.02537)  

**Abstract**: Hiring processes often involve the manual screening of hundreds of resumes for each job, a task that is time and effort consuming, error-prone, and subject to human bias. This paper presents Smart-Hiring, an end-to-end Natural Language Processing (NLP) pipeline de- signed to automatically extract structured information from unstructured resumes and to semantically match candidates with job descriptions. The proposed system combines document parsing, named-entity recognition, and contextual text embedding techniques to capture skills, experience, and qualifications. Using advanced NLP technics, Smart-Hiring encodes both resumes and job descriptions in a shared vector space to compute similarity scores between candidates and job postings. The pipeline is modular and explainable, allowing users to inspect extracted entities and matching rationales. Experiments were conducted on a real-world dataset of resumes and job descriptions spanning multiple professional domains, demonstrating the robustness and feasibility of the proposed approach. The system achieves competitive matching accuracy while preserving a high degree of interpretability and transparency in its decision process. This work introduces a scalable and practical NLP frame- work for recruitment analytics and outlines promising directions for bias mitigation, fairness-aware modeling, and large-scale deployment of data-driven hiring solutions. 

---
# Prompting for Policy: Forecasting Macroeconomic Scenarios with Synthetic LLM Personas 

**Authors**: Giulia Iadisernia, Carolina Camassa  

**Link**: [PDF](https://arxiv.org/pdf/2511.02458)  

**Abstract**: We evaluate whether persona-based prompting improves Large Language Model (LLM) performance on macroeconomic forecasting tasks. Using 2,368 economics-related personas from the PersonaHub corpus, we prompt GPT-4o to replicate the ECB Survey of Professional Forecasters across 50 quarterly rounds (2013-2025). We compare the persona-prompted forecasts against the human experts panel, across four target variables (HICP, core HICP, GDP growth, unemployment) and four forecast horizons. We also compare the results against 100 baseline forecasts without persona descriptions to isolate its effect. We report two main findings. Firstly, GPT-4o and human forecasters achieve remarkably similar accuracy levels, with differences that are statistically significant yet practically modest. Our out-of-sample evaluation on 2024-2025 data demonstrates that GPT-4o can maintain competitive forecasting performance on unseen events, though with notable differences compared to the in-sample period. Secondly, our ablation experiment reveals no measurable forecasting advantage from persona descriptions, suggesting these prompt components can be omitted to reduce computational costs without sacrificing accuracy. Our results provide evidence that GPT-4o can achieve competitive forecasting accuracy even on out-of-sample macroeconomic events, if provided with relevant context data, while revealing that diverse prompts produce remarkably homogeneous forecasts compared to human panels. 

---
# Merging Continual Pretraining Models for Domain-Specialized LLMs: A Case Study in Finance 

**Authors**: Kentaro Ueda, François Portet, Hirohiko Suwa, Keiichi Yasumoto  

**Link**: [PDF](https://arxiv.org/pdf/2511.02451)  

**Abstract**: While LLMs excel at general tasks, they struggle in specialized domains like finance, requiring diverse skills in domain knowledge, mathematical reasoning, and multilingual processing. Merging domain-specific Continual Pre-training (CPT) "experts" offers a practical alternative to costly and unstable multi-skill training. However, unlike established Supervised Fine-Tuning (SFT) model-based merging, CPT model merging remains largely unexplored. We address this gap by creating financial LLMs from experts in finance, math, and Japanese. We propose a three-stage evaluation focusing on knowledge recovery, complementarity, and emergence, and assess three merging methods (Task Arithmetic, TIES, and DARE-TIES) on a comprehensive financial benchmark curated from 18 tasks across 8 established datasets. Results show that merging an expert with its base model recovers general knowledge lost during CPT, while merging experts improves performance and can yield emergent cross-domain skills. Among the methods, Task Arithmetic performs strongly but is hyperparameter-sensitive, whereas TIES is more robust. Our findings also suggest that while model similarity correlates with merging success, emergent skills depend on more complex factors. This work presents the first foundational analysis of CPT model merging, establishing a principled framework and providing clear guidance for building multi-skill LLMs from existing assets. 

---
# AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models 

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban  

**Link**: [PDF](https://arxiv.org/pdf/2511.02376)  

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs, yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves up to 95% attack success rate on Llama-3.1-8B within six turns a 24 percent improvement over single turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests then iteratively refines them. Extensive evaluation across commercial and open-source models (GPT-4o-mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses. 

---
# AyurParam: A State-of-the-Art Bilingual Language Model for Ayurveda 

**Authors**: Mohd Nauman, Sravan Gvm, Vijay Devane, Shyam Pawar, Viraj Thakur, Kundeshwar Pundalik, Piyush Sawarkar, Rohit Saluja, Maunendra Desarkar, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2511.02374)  

**Abstract**: Current large language models excel at broad, general-purpose tasks, but consistently underperform when exposed to highly specialized domains that require deep cultural, linguistic, and subject-matter expertise. In particular, traditional medical systems such as Ayurveda embody centuries of nuanced textual and clinical knowledge that mainstream LLMs fail to accurately interpret or apply. We introduce AyurParam-2.9B, a domain-specialized, bilingual language model fine-tuned from Param-1-2.9B using an extensive, expertly curated Ayurveda dataset spanning classical texts and clinical guidance. AyurParam's dataset incorporates context-aware, reasoning, and objective-style Q&A in both English and Hindi, with rigorous annotation protocols for factual precision and instructional clarity. Benchmarked on BhashaBench-Ayur, AyurParam not only surpasses all open-source instruction-tuned models in its size class (1.5--3B parameters), but also demonstrates competitive or superior performance compared to much larger models. The results from AyurParam highlight the necessity for authentic domain adaptation and high-quality supervision in delivering reliable, culturally congruent AI for specialized medical knowledge. 

---
# LiveSecBench: A Dynamic and Culturally-Relevant AI Safety Benchmark for LLMs in Chinese Context 

**Authors**: Yudong Li, Zhongliang Yang, Kejiang Chen, Wenxuan Wang, Tianxin Zhang, Sifang Wan, Kecheng Wang, Haitian Li, Xu Wang, Lefan Cheng, Youdan Yang, Baocheng Chen, Ziyu Liu, Yufei Sun, Liyan Wu, Wenya Wen, Xingchi Gu, Peiru Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02366)  

**Abstract**: In this work, we propose LiveSecBench, a dynamic and continuously updated safety benchmark specifically for Chinese-language LLM application scenarios. LiveSecBench evaluates models across six critical dimensions (Legality, Ethics, Factuality, Privacy, Adversarial Robustness, and Reasoning Safety) rooted in the Chinese legal and social frameworks. This benchmark maintains relevance through a dynamic update schedule that incorporates new threat vectors, such as the planned inclusion of Text-to-Image Generation Safety and Agentic Safety in the next update. For now, LiveSecBench (v251030) has evaluated 18 LLMs, providing a landscape of AI safety in the context of Chinese language. The leaderboard is publicly accessible at this https URL. 

---
# Let Multimodal Embedders Learn When to Augment Query via Adaptive Query Augmentation 

**Authors**: Wongyu Kim, Hochang Lee, Sanghak Lee, Yoonsung Kim, Jaehyun Park  

**Link**: [PDF](https://arxiv.org/pdf/2511.02358)  

**Abstract**: Query augmentation makes queries more meaningful by appending further information to the queries to find relevant documents. Current studies have proposed Large Language Model (LLM)-based embedders, which learn representation for embedding and generation for query augmentation in a multi-task manner by leveraging the generative capabilities of LLM. During inference, these jointly trained embedders have conducted query augmentation followed by embedding, showing effective results. However, augmenting every query leads to substantial embedding latency and query augmentation can be detrimental to performance for some queries. Also, previous methods have not been explored in multimodal environments. To tackle these problems, we propose M-Solomon, a universal multimodal embedder that can adaptively determine when to augment queries. Our approach first divides the queries of the training datasets into two groups at the dataset level. One includes queries that require augmentation and the other includes queries that do not. Then, we introduces a synthesis process that generates appropriate augmentations for queries that require them by leveraging a powerful Multimodal LLM (MLLM). Next, we present adaptive query augmentation. Through this step, M-Solomon can conduct query augmentation only when necessary by learning to generate synthetic augmentations with the prefix /augment for queries that demand them and to generate the simple string /embed for others. Experimental results showed that M-Solomon not only surpassed the baseline without augmentation by a large margin but also outperformed the baseline that always used augmentation, providing much faster embedding latency. 

---
# LTD-Bench: Evaluating Large Language Models by Letting Them Draw 

**Authors**: Liuhao Lin, Ke Li, Zihan Xu, Yuchen Shi, Yulei Qin, Yan Zhang, Xing Sun, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2511.02347)  

**Abstract**: Current evaluation paradigms for large language models (LLMs) represent a critical blind spot in AI research--relying on opaque numerical metrics that conceal fundamental limitations in spatial reasoning while providing no intuitive understanding of model capabilities. This deficiency creates a dangerous disconnect between reported performance and practical abilities, particularly for applications requiring physical world understanding. We introduce LTD-Bench, a breakthrough benchmark that transforms LLM evaluation from abstract scores to directly observable visual outputs by requiring models to generate drawings through dot matrices or executable code. This approach makes spatial reasoning limitations immediately apparent even to non-experts, bridging the fundamental gap between statistical performance and intuitive assessment. LTD-Bench implements a comprehensive methodology with complementary generation tasks (testing spatial imagination) and recognition tasks (assessing spatial perception) across three progressively challenging difficulty levels, methodically evaluating both directions of the critical language-spatial mapping. Our extensive experiments with state-of-the-art models expose an alarming capability gap: even LLMs achieving impressive results on traditional benchmarks demonstrate profound deficiencies in establishing bidirectional mappings between language and spatial concept--a fundamental limitation that undermines their potential as genuine world models. Furthermore, LTD-Bench's visual outputs enable powerful diagnostic analysis, offering a potential approach to investigate model similarity. 

---
# Demo: Statistically Significant Results On Biases and Errors of LLMs Do Not Guarantee Generalizable Results 

**Authors**: Jonathan Liu, Haoling Qiu, Jonathan Lasko, Damianos Karakos, Mahsa Yarmohammadi, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2511.02246)  

**Abstract**: Recent research has shown that hallucinations, omissions, and biases are prevalent in everyday use-cases of LLMs. However, chatbots used in medical contexts must provide consistent advice in situations where non-medical factors are involved, such as when demographic information is present. In order to understand the conditions under which medical chatbots fail to perform as expected, we develop an infrastructure that 1) automatically generates queries to probe LLMs and 2) evaluates answers to these queries using multiple LLM-as-a-judge setups and prompts. For 1), our prompt creation pipeline samples the space of patient demographics, histories, disorders, and writing styles to create realistic questions that we subsequently use to prompt LLMs. In 2), our evaluation pipeline provides hallucination and omission detection using LLM-as-a-judge as well as agentic workflows, in addition to LLM-as-a-judge treatment category detectors. As a baseline study, we perform two case studies on inter-LLM agreement and the impact of varying the answering and evaluation LLMs. We find that LLM annotators exhibit low agreement scores (average Cohen's Kappa $\kappa=0.118$), and only specific (answering, evaluation) LLM pairs yield statistically significant differences across writing styles, genders, and races. We recommend that studies using LLM evaluation use multiple LLMs as evaluators in order to avoid arriving at statistically significant but non-generalizable results, particularly in the absence of ground-truth data. We also suggest publishing inter-LLM agreement metrics for transparency. Our code and dataset are available here: this https URL. 

---
# IG-Pruning: Input-Guided Block Pruning for Large Language Models 

**Authors**: Kangyu Qiao, Shaolei Zhang, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2511.02213)  

**Abstract**: With the growing computational demands of large language models (LLMs), efficient inference has become increasingly critical for practical deployment. Depth pruning has emerged as a promising approach for reducing the computational costs of large language models by removing transformer layers. However, existing methods typically rely on fixed block masks, which can lead to suboptimal performance across different tasks and inputs. In this paper, we propose IG-Pruning, a novel input-aware block-wise pruning method that dynamically selects layer masks at inference time. Our approach consists of two stages: (1) Discovering diverse mask candidates through semantic clustering and L0 optimization, and (2) Implementing efficient dynamic pruning without the need for extensive training. Experimental results demonstrate that our method consistently outperforms state-of-the-art static depth pruning methods, making it particularly suitable for resource-constrained deployment scenarios. 

---
# Rethinking LLM Human Simulation: When a Graph is What You Need 

**Authors**: Joseph Suh, Suhong Moon, Serina Chang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02135)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate humans, with applications ranging from survey prediction to decision-making. However, are LLMs strictly necessary, or can smaller, domain-grounded models suffice? We identify a large class of simulation problems in which individuals make choices among discrete options, where a graph neural network (GNN) can match or surpass strong LLM baselines despite being three orders of magnitude smaller. We introduce Graph-basEd Models for human Simulation (GEMS), which casts discrete choice simulation tasks as a link prediction problem on graphs, leveraging relational knowledge while incorporating language representations only when needed. Evaluations across three key settings on three simulation datasets show that GEMS achieves comparable or better accuracy than LLMs, with far greater efficiency, interpretability, and transparency, highlighting the promise of graph-based modeling as a lightweight alternative to LLMs for human simulation. Our code is available at this https URL. 

---
# Multi-Personality Generation of LLMs at Decoding-time 

**Authors**: Rongxin Chen, Yunfan Li, Yige Yuan, Bingbing Xu, Huawei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2511.01891)  

**Abstract**: Multi-personality generation for LLMs, enabling simultaneous embodiment of multiple personalization attributes, is a fundamental challenge. Existing retraining-based approaches are costly and poorly scalable, while decoding-time methods often rely on external models or heuristics, limiting flexibility and robustness. In this paper, we propose a novel Multi-Personality Generation (MPG) framework under the decoding-time combination paradigm. It flexibly controls multi-personality without relying on scarce multi-dimensional models or extra training, leveraging implicit density ratios in single-dimensional models as a "free lunch" to reformulate the task as sampling from a target strategy aggregating these ratios. To implement MPG efficiently, we design Speculative Chunk-level based Rejection sampling (SCR), which generates responses in chunks and parallelly validates them via estimated thresholds within a sliding window. This significantly reduces computational overhead while maintaining high-quality generation. Experiments on MBTI personality and Role-Playing demonstrate the effectiveness of MPG, showing improvements up to 16%-18%. Code and data are available at this https URL . 

---
# Agent-Omni: Test-Time Multimodal Reasoning via Model Coordination for Understanding Anything 

**Authors**: Huawei Lin, Yunzhi Shi, Tong Geng, Weijie Zhao, Wei Wang, Ravender Pal Singh  

**Link**: [PDF](https://arxiv.org/pdf/2511.02834)  

**Abstract**: Multimodal large language models (MLLMs) have shown strong capabilities but remain limited to fixed modality pairs and require costly fine-tuning with large aligned datasets. Building fully omni-capable models that can integrate text, images, audio, and video remains impractical and lacks robust reasoning support. In this paper, we propose an Agent-Omni framework that coordinates existing foundation models through a master-agent system, enabling flexible multimodal reasoning without retraining. The master agent interprets user intent, delegates subtasks to modality-specific agents, and integrates their outputs into coherent responses. Extensive experiments across text, image, audio, video, and omni benchmarks show that Agent-Omni consistently achieves state-of-the-art performance, particularly on tasks requiring complex cross-modal reasoning. Its agent-based design enables seamless integration of specialized foundation models, ensuring adaptability to diverse inputs while maintaining transparency and interpretability. In addition, the framework is modular and easily extensible, allowing future improvements as stronger models become available. %We release an open-source implementation to support continued research on scalable and reliable omni-modal reasoning. 

---
# In Good GRACEs: Principled Teacher Selection for Knowledge Distillation 

**Authors**: Abhishek Panigrahi, Bingbin Liu, Sadhika Malladi, Sham Kakade, Surbhi Goel  

**Link**: [PDF](https://arxiv.org/pdf/2511.02833)  

**Abstract**: Knowledge distillation is an efficient strategy to use data generated by large "teacher" language models to train smaller capable "student" models, but selecting the optimal teacher for a specific student-task combination requires expensive trial-and-error. We propose a lightweight score called GRACE to quantify how effective a teacher will be for post-training a student model. GRACE measures distributional properties of the student's gradients without access to a verifier, teacher logits, teacher internals, or test data. From an information-theoretic perspective, GRACE connects to leave-one-out stability of gradient-based algorithms, which controls the generalization performance of the distilled students. On GSM8K and MATH, GRACE correlates strongly (up to 86% Spearman correlation) with the performance of the distilled LLaMA and OLMo students. In particular, training a student using the GRACE-selected teacher can improve the performance by up to 7.4% over naively using the best-performing teacher. Further, GRACE can provide guidance on crucial design choices in distillation, including (1) the best temperature to use when generating from the teacher, (2) the best teacher to use given a size constraint, and (3) the best teacher to use within a specific model family. Altogether, our findings demonstrate that GRACE can efficiently and effectively identify a strongly compatible teacher for a given student and provide fine-grained guidance on how to perform distillation. 

---
# Can LLMs subtract numbers? 

**Authors**: Mayank Jobanputra, Nils Philipp Walter, Maitrey Mehta, Blerta Veseli, Evan Parker Kelly Chapple, Yifan Wang, Sneha Chetani, Ellie Pavlick, Antonio Vergari, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2511.02795)  

**Abstract**: We present a systematic study of subtraction in large language models (LLMs). While prior benchmarks emphasize addition and multiplication, subtraction has received comparatively little attention despite being structurally distinct as a non-commutative operation. We evaluate eight pretrained LLMs spanning four families on addition and subtraction problems. Our experiments reveal that subtraction accuracy lags behind addition by a wide margin. We find that the errors for ($a-b$) are concentrated in cases where ($a<b$). In such cases, LLMs frequently produce the correct magnitude but omit the negative sign. Probing analyses show that LLMs internally encode whether results should be negative, yet this information is often not reflected in generated outputs. We further test well-known techniques such as few-shot learning and instruction-tuning to see if they can improve the LLMs' performance. Our results suggest that while few-shot prompting yields modest gains, the instruction-tuned models achieve near-perfect accuracies in generating the negative sign. Together, these findings provide a clearer characterization of the limitations and recoverability of LLMs' arithmetic capabilities in subtraction. 

---
# VCode: a Multimodal Coding Benchmark with SVG as Symbolic Visual Representation 

**Authors**: Kevin Qinghong Lin, Yuhao Zheng, Hangyu Ran, Dantong Zhu, Dongxing Mao, Linjie Li, Philip Torr, Alex Jinpeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02778)  

**Abstract**: Code has emerged as a precise and executable medium for reasoning and action in the agent era. Yet, progress has largely focused on language-centric tasks such as program synthesis and debugging, leaving visual-centric coding underexplored. Inspired by how humans reason over sketches, we advocate SVG code as a compact, interpretable, and executable visual representation. We introduce VCode, a benchmark that reframes multimodal understanding as code generation: given an image, a model must produce SVG that preserves symbolic meaning for downstream reasoning. VCode covers three domains - general commonsense (MM-Vet), professional disciplines (MMMU), and visual-centric perception (CV-Bench). To assess symbolic fidelity, we propose CodeVQA, a novel evaluation protocol in which a policy model answers questions over rendered SVGs; correct answers indicate faithful symbolic preservation. Empirically, frontier VLMs struggle to generate faithful SVGs, revealing a persistent gap between language-centric and visual-centric coding. To close this gap, we introduce VCoder, an agentic framework that augments VLMs along two axes: (i) Thinking with Revision, which iteratively analyzes discrepancies and refines SVG code; and (ii) Acting with Visual Tools, where detectors and parsers supply structured cues such as objects, shapes, and text beyond the model's intrinsic capacity. Across benchmarks, frontier VLMs with strong reasoning capabilities score well overall yet remain limited in professional knowledge and 3D reasoning. VCoder delivers a 12.3-point overall gain over the top-performing Claude-4-Opus. Human studies show that both humans and VLMs perform worse on rendered SVGs, their consistency reveals the promise of symbolic visual representation. The benchmark and code are available at this https URL. 

---
# CostBench: Evaluating Multi-Turn Cost-Optimal Planning and Adaptation in Dynamic Environments for LLM Tool-Use Agents 

**Authors**: Jiayu Liu, Cheng Qian, Zhaochen Su, Qing Zong, Shijue Huang, Bingxiang He, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2511.02734)  

**Abstract**: Current evaluations of Large Language Model (LLM) agents primarily emphasize task completion, often overlooking resource efficiency and adaptability. This neglects a crucial capability: agents' ability to devise and adjust cost-optimal plans in response to changing environments. To bridge this gap, we introduce CostBench, a scalable, cost-centric benchmark designed to evaluate agents' economic reasoning and replanning abilities. Situated in the travel-planning domain, CostBench comprises tasks solvable via multiple sequences of atomic and composite tools with diverse, customizable costs. It also supports four types of dynamic blocking events, such as tool failures and cost changes, to simulate real-world unpredictability and necessitate agents to adapt in real time. Evaluating leading open-sourced and proprietary models on CostBench reveals a substantial gap in cost-aware planning: agents frequently fail to identify cost-optimal solutions in static settings, with even GPT-5 achieving less than 75% exact match rate on the hardest tasks, and performance further dropping by around 40% under dynamic conditions. By diagnosing these weaknesses, CostBench lays the groundwork for developing future agents that are both economically rational and robust. 

---
# The Collaboration Gap 

**Authors**: Tim R. Davidson, Adam Fourney, Saleema Amershi, Robert West, Eric Horvitz, Ece Kamar  

**Link**: [PDF](https://arxiv.org/pdf/2511.02687)  

**Abstract**: The trajectory of AI development suggests that we will increasingly rely on agent-based systems composed of independently developed agents with different information, privileges, and tools. The success of these systems will critically depend on effective collaboration among these heterogeneous agents, even under partial observability. Despite intense interest, few empirical studies have evaluated such agent-agent collaboration at scale. We propose a collaborative maze-solving benchmark that (i) isolates collaborative capabilities, (ii) modulates problem complexity, (iii) enables scalable automated grading, and (iv) imposes no output-format constraints, preserving ecological plausibility. Using this framework, we evaluate 32 leading open- and closed-source models in solo, homogeneous, and heterogeneous pairings. Our results reveal a "collaboration gap": models that perform well solo often degrade substantially when required to collaborate. Collaboration can break down dramatically; for instance, small distilled models that solve mazes well alone may fail almost completely in certain pairings. We find that starting with the stronger agent often improves outcomes, motivating a "relay inference" approach where the stronger agent leads before handing off to the weaker one, closing much of the gap. Our findings argue for (1) collaboration-aware evaluation, (2) training strategies developed to enhance collaborative capabilities, and (3) interaction design that reliably elicits agents' latent skills, guidance that applies to AI-AI and human-AI collaboration. 

---
# UniChange: Unifying Change Detection with Multimodal Large Language Model 

**Authors**: Xu Zhang, Danyang Li, Xiaohang Dong, Tianhao Wu, Hualong Yu, Jianye Wang, Qicheng Li, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.02607)  

**Abstract**: Change detection (CD) is a fundamental task for monitoring and analyzing land cover dynamics. While recent high performance models and high quality datasets have significantly advanced the field, a critical limitation persists. Current models typically acquire limited knowledge from single-type annotated data and cannot concurrently leverage diverse binary change detection (BCD) and semantic change detection (SCD) datasets. This constraint leads to poor generalization and limited versatility. The recent advancements in Multimodal Large Language Models (MLLMs) introduce new possibilities for a unified CD framework. We leverage the language priors and unification capabilities of MLLMs to develop UniChange, the first MLLM-based unified change detection model. UniChange integrates generative language abilities with specialized CD functionalities. Our model successfully unifies both BCD and SCD tasks through the introduction of three special tokens: [T1], [T2], and [CHANGE]. Furthermore, UniChange utilizes text prompts to guide the identification of change categories, eliminating the reliance on predefined classification heads. This design allows UniChange to effectively acquire knowledge from multi-source datasets, even when their class definitions conflict. Experiments on four public benchmarks (WHU-CD, S2Looking, LEVIR-CD+, and SECOND) demonstrate SOTA performance, achieving IoU scores of 90.41, 53.04, 78.87, and 57.62, respectively, surpassing all previous methods. The code is available at this https URL. 

---
# DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding 

**Authors**: Zixuan Liu, Siavash H. Khajavi, Guangkai Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02495)  

**Abstract**: Recent advances in multi-modal models have demonstrated strong performance in tasks such as image generation and reasoning. However, applying these models to the fire domain remains challenging due to the lack of publicly available datasets with high-quality fire domain annotations. To address this gap, we introduce DetectiumFire, a large-scale, multi-modal dataset comprising of 22.5k high-resolution fire-related images and 2.5k real-world fire-related videos covering a wide range of fire types, environments, and risk levels. The data are annotated with both traditional computer vision labels (e.g., bounding boxes) and detailed textual prompts describing the scene, enabling applications such as synthetic data generation and fire risk reasoning. DetectiumFire offers clear advantages over existing benchmarks in scale, diversity, and data quality, significantly reducing redundancy and enhancing coverage of real-world scenarios. We validate the utility of DetectiumFire across multiple tasks, including object detection, diffusion-based image generation, and vision-language reasoning. Our results highlight the potential of this dataset to advance fire-related research and support the development of intelligent safety systems. We release DetectiumFire to promote broader exploration of fire understanding in the AI community. The dataset is available at this https URL 

---
# CoCoVa: Chain of Continuous Vision-Language Thought for Latent Space Reasoning 

**Authors**: Jizheng Ma, Xiaofei Zhou, Yanlong Song, Han Yan  

**Link**: [PDF](https://arxiv.org/pdf/2511.02360)  

**Abstract**: In human cognition, there exist numerous thought processes that are tacit and beyond verbal expression, enabling us to understand and interact with the world in multiple ways. However, contemporary Vision-Language Models (VLMs) remain constrained to reasoning within the discrete and rigid space of linguistic tokens, thereby bottlenecking the rich, high-dimensional nature of visual perception. To bridge this gap, we propose CoCoVa (Chain of Continuous Vision-Language Thought), a novel framework for vision-language model that leverages continuous cross-modal reasoning for diverse vision-language tasks. The core of CoCoVa is an iterative reasoning cycle, where a novel Latent Q-Former (LQ-Former) acts as a dynamic reasoning engine, iteratively refining a chain of latent thought vectors through cross-modal fusion. To focus this process, a token selection mechanism dynamically identifies salient visual regions, mimicking attentional focus. To ensure these latent thoughts remain grounded, we train the model with a multi-task objective that combines contrastive learning and diffusion-based reconstruction, enforcing alignment between latent representations and both visual and textual modalities. Evaluations show CoCoVa improves accuracy and token efficiency over strong baselines. With a 1.5B backbone, it competes with or surpasses larger 7B-9B models on almost all benchmarks. When scaled to 7B LLM backbones, it remains competitive with state-of-the-art models. Qualitative analysis validates that learned latent space captures interpretable and structured reasoning patterns, highlighting the potential of CoCoVa to bridge the representational gap between discrete language processing and the continuous nature of visual understanding. 

---
# Automata-Conditioned Cooperative Multi-Agent Reinforcement Learning 

**Authors**: Beyazit Yalcinkaya, Marcell Vazquez-Chanlatte, Ameesh Shah, Hanna Krasowski, Sanjit A. Seshia  

**Link**: [PDF](https://arxiv.org/pdf/2511.02304)  

**Abstract**: We study the problem of learning multi-task, multi-agent policies for cooperative, temporal objectives, under centralized training, decentralized execution. In this setting, using automata to represent tasks enables the decomposition of complex tasks into simpler sub-tasks that can be assigned to agents. However, existing approaches remain sample-inefficient and are limited to the single-task case. In this work, we present Automata-Conditioned Cooperative Multi-Agent Reinforcement Learning (ACC-MARL), a framework for learning task-conditioned, decentralized team policies. We identify the main challenges to ACC-MARL's feasibility in practice, propose solutions, and prove the correctness of our approach. We further show that the value functions of learned policies can be used to assign tasks optimally at test time. Experiments show emergent task-aware, multi-step coordination among agents, e.g., pressing a button to unlock a door, holding the door, and short-circuiting tasks. 

---
# Unlocking the Power of Multi-Agent LLM for Reasoning: From Lazy Agents to Deliberation 

**Authors**: Zhiwei Zhang, Xiaomin Li, Yudi Lin, Hui Liu, Ramraj Chandradevan, Linlin Wu, Minhua Lin, Fali Wang, Xianfeng Tang, Qi He, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02303)  

**Abstract**: Large Language Models (LLMs) trained with reinforcement learning and verifiable rewards have achieved strong results on complex reasoning tasks. Recent work extends this paradigm to a multi-agent setting, where a meta-thinking agent proposes plans and monitors progress while a reasoning agent executes subtasks through sequential conversational turns. Despite promising performance, we identify a critical limitation: lazy agent behavior, in which one agent dominates while the other contributes little, undermining collaboration and collapsing the setup to an ineffective single agent. In this paper, we first provide a theoretical analysis showing why lazy behavior naturally arises in multi-agent reasoning. We then introduce a stable and efficient method for measuring causal influence, helping mitigate this issue. Finally, as collaboration intensifies, the reasoning agent risks getting lost in multi-turn interactions and trapped by previous noisy responses. To counter this, we propose a verifiable reward mechanism that encourages deliberation by allowing the reasoning agent to discard noisy outputs, consolidate instructions, and restart its reasoning process when necessary. Extensive experiments demonstrate that our framework alleviates lazy agent behavior and unlocks the full potential of multi-agent framework for complex reasoning tasks. 

---
# Link prediction Graph Neural Networks for structure recognition of Handwritten Mathematical Expressions 

**Authors**: Cuong Tuan Nguyen, Ngoc Tuan Nguyen, Triet Hoang Minh Dao, Huy Minh Nhat, Huy Truong Dinh  

**Link**: [PDF](https://arxiv.org/pdf/2511.02288)  

**Abstract**: We propose a Graph Neural Network (GNN)-based approach for Handwritten Mathematical Expression (HME) recognition by modeling HMEs as graphs, where nodes represent symbols and edges capture spatial dependencies. A deep BLSTM network is used for symbol segmentation, recognition, and spatial relation classification, forming an initial primitive graph. A 2D-CFG parser then generates all possible spatial relations, while the GNN-based link prediction model refines the structure by removing unnecessary connections, ultimately forming the Symbol Label Graph. Experimental results demonstrate the effectiveness of our approach, showing promising performance in HME structure recognition. 

---
# SAIL-RL: Guiding MLLMs in When and How to Think via Dual-Reward RL Tuning 

**Authors**: Fangxun Shu, Yongjie Ye, Yue Liao, Zijian Kang, Weijie Yin, Jiacong Wang, Xiao Liang, Shuicheng Yan, Chao Feng  

**Link**: [PDF](https://arxiv.org/pdf/2511.02280)  

**Abstract**: We introduce SAIL-RL, a reinforcement learning (RL) post-training framework that enhances the reasoning capabilities of multimodal large language models (MLLMs) by teaching them when and how to think. Existing approaches are limited by outcome-only supervision, which rewards correct answers without ensuring sound reasoning, and by uniform thinking strategies, which often lead to overthinking on simple tasks and underthinking on complex ones. SAIL-RL addresses these challenges with a dual reward system: the Thinking Reward, which evaluates reasoning quality through factual grounding, logical coherence, and answer consistency, and the Judging Reward, which adaptively determines whether deep reasoning or direct answering is appropriate. Experiments on the state-of-the-art SAIL-VL2 show that SAIL-RL improves reasoning and multimodal understanding benchmarks at both 4B and 8B scales, achieving competitive performance against commercial closed-source models such as GPT-4o, and substantially reduces hallucinations, establishing it as a principled framework for building more reliable and adaptive MLLMs. The code will be available at this https URL. 

---
# An Evaluation of Interleaved Instruction Tuning on Semantic Reasoning Performance in an Audio MLLM 

**Authors**: Jiawei Liu, Enis Berk Çoban, Zarina Schevchenko, Hao Tang, Zhigang Zhu, Michael I Mandel, Johanna Devaney  

**Link**: [PDF](https://arxiv.org/pdf/2511.02234)  

**Abstract**: Standard training for Multi-modal Large Language Models (MLLMs) involves concatenating non-textual information, like vision or audio, with a text prompt. This approach may not encourage deep integration of modalities, limiting the model's ability to leverage the core language model's reasoning capabilities. This work examined the impact of interleaved instruction tuning in an audio MLLM, where audio tokens are interleaved within the prompt. Using the Listen, Think, and Understand (LTU) model as a testbed, we conduct an experiment using the Synonym and Hypernym Audio Reasoning Dataset (SHARD), our newly created reasoning benchmark for audio-based semantic reasoning focusing on synonym and hypernym recognition. Our findings show that while even zero-shot interleaved prompting improves performance on our reasoning tasks, a small amount of fine-tuning using interleaved training prompts improves the results further, however, at the expense of the MLLM's audio labeling ability. 

---
# Training Proactive and Personalized LLM Agents 

**Authors**: Weiwei Sun, Xuhui Zhou, Weihua Du, Xingyao Wang, Sean Welleck, Graham Neubig, Maarten Sap, Yiming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02208)  

**Abstract**: While existing work focuses primarily on task success, we argue that effective real-world agents require optimizing three dimensions: productivity (task completion), proactivity (asking essential questions), and personalization (adapting to diverse user preferences). We introduce UserVille, an interactive environment with LLM-based user simulators enabling diverse, configurable user preferences. Leveraging UserVille, we introduce PPP, a multi-objective reinforcement learning approach that jointly optimizes all three dimensions: Productivity, Proactivity, and Personalization. Experiments on software engineering and deep research tasks show that agents trained with PPP achieve substantial improvements over strong baselines such as GPT-5 (+21.6 on average), demonstrating the ability to ask strategic clarifying questions, adapt to unseen user preferences, and improve task success through better interaction. This work demonstrates that explicitly optimizing for user-centered interaction is critical for building practical and effective AI agents. 

---
# Personalized Decision Modeling: Utility Optimization or Textualized-Symbolic Reasoning 

**Authors**: Yibo Zhao, Yang Zhao, Hongru Du, Hao Frank Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02194)  

**Abstract**: Decision-making models for individuals, particularly in high-stakes scenarios like vaccine uptake, often diverge from population optimal predictions. This gap arises from the uniqueness of the individual decision-making process, shaped by numerical attributes (e.g., cost, time) and linguistic influences (e.g., personal preferences and constraints). Developing upon Utility Theory and leveraging the textual-reasoning capabilities of Large Language Models (LLMs), this paper proposes an Adaptive Textual-symbolic Human-centric Reasoning framework (ATHENA) to address the optimal information integration. ATHENA uniquely integrates two stages: First, it discovers robust, group-level symbolic utility functions via LLM-augmented symbolic discovery; Second, it implements individual-level semantic adaptation, creating personalized semantic templates guided by the optimal utility to model personalized choices. Validated on real-world travel mode and vaccine choice tasks, ATHENA consistently outperforms utility-based, machine learning, and other LLM-based models, lifting F1 score by at least 6.5% over the strongest cutting-edge models. Further, ablation studies confirm that both stages of ATHENA are critical and complementary, as removing either clearly degrades overall predictive performance. By organically integrating symbolic utility modeling and semantic adaptation, ATHENA provides a new scheme for modeling human-centric decisions. The project page can be found at this https URL. 

---
# InsurAgent: A Large Language Model-Empowered Agent for Simulating Individual Behavior in Purchasing Flood Insurance 

**Authors**: Ziheng Geng, Jiachen Liu, Ran Cao, Lu Cheng, Dan M. Frangopol, Minghui Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.02119)  

**Abstract**: Flood insurance is an effective strategy for individuals to mitigate disaster-related losses. However, participation rates among at-risk populations in the United States remain strikingly low. This gap underscores the need to understand and model the behavioral mechanisms underlying insurance decisions. Large language models (LLMs) have recently exhibited human-like intelligence across wide-ranging tasks, offering promising tools for simulating human decision-making. This study constructs a benchmark dataset to capture insurance purchase probabilities across factors. Using this dataset, the capacity of LLMs is evaluated: while LLMs exhibit a qualitative understanding of factors, they fall short in estimating quantitative probabilities. To address this limitation, InsurAgent, an LLM-empowered agent comprising five modules including perception, retrieval, reasoning, action, and memory, is proposed. The retrieval module leverages retrieval-augmented generation (RAG) to ground decisions in empirical survey data, achieving accurate estimation of marginal and bivariate probabilities. The reasoning module leverages LLM common sense to extrapolate beyond survey data, capturing contextual information that is intractable for traditional models. The memory module supports the simulation of temporal decision evolutions, illustrated through a roller coaster life trajectory. Overall, InsurAgent provides a valuable tool for behavioral modeling and policy analysis. 

---
# Deep Value Benchmark: Measuring Whether Models Generalize Deep values or Shallow Preferences 

**Authors**: Joshua Ashkinaze, Hua Shen, Sai Avula, Eric Gilbert, Ceren Budak  

**Link**: [PDF](https://arxiv.org/pdf/2511.02109)  

**Abstract**: We introduce the Deep Value Benchmark (DVB), an evaluation framework that directly tests whether large language models (LLMs) learn fundamental human values or merely surface-level preferences. This distinction is critical for AI alignment: Systems that capture deeper values are likely to generalize human intentions robustly, while those that capture only superficial patterns in preference data risk producing misaligned behavior. The DVB uses a novel experimental design with controlled confounding between deep values (e.g., moral principles) and shallow features (e.g., superficial attributes). In the training phase, we expose LLMs to human preference data with deliberately correlated deep and shallow features -- for instance, where a user consistently prefers (non-maleficence, formal language) options over (justice, informal language) alternatives. The testing phase then breaks these correlations, presenting choices between (justice, formal language) and (non-maleficence, informal language) options. This design allows us to precisely measure a model's Deep Value Generalization Rate (DVGR) -- the probability of generalizing based on the underlying value rather than the shallow feature. Across 9 different models, the average DVGR is just 0.30. All models generalize deep values less than chance. Larger models have a (slightly) lower DVGR than smaller models. We are releasing our dataset, which was subject to three separate human validation experiments. DVB provides an interpretable measure of a core feature of alignment. 

---
# LLM Probing with Contrastive Eigenproblems: Improving Understanding and Applicability of CCS 

**Authors**: Stefan F. Schouten, Peter Bloem  

**Link**: [PDF](https://arxiv.org/pdf/2511.02089)  

**Abstract**: Contrast-Consistent Search (CCS) is an unsupervised probing method able to test whether large language models represent binary features, such as sentence truth, in their internal activations. While CCS has shown promise, its two-term objective has been only partially understood. In this work, we revisit CCS with the aim of clarifying its mechanisms and extending its applicability. We argue that what should be optimized for, is relative contrast consistency. Building on this insight, we reformulate CCS as an eigenproblem, yielding closed-form solutions with interpretable eigenvalues and natural extensions to multiple variables. We evaluate these approaches across a range of datasets, finding that they recover similar performance to CCS, while avoiding problems around sensitivity to random initialization. Our results suggest that relativizing contrast consistency not only improves our understanding of CCS but also opens pathways for broader probing and mechanistic interpretability methods. 

---
# Complete asymptotic type-token relationship for growing complex systems with inverse power-law count rankings 

**Authors**: Pablo Rosillo-Rodes, Laurent Hébert-Dufresne, Peter Sheridan Dodds  

**Link**: [PDF](https://arxiv.org/pdf/2511.02069)  

**Abstract**: The growth dynamics of complex systems often exhibit statistical regularities involving power-law relationships. For real finite complex systems formed by countable tokens (animals, words) as instances of distinct types (species, dictionary entries), an inverse power-law scaling $S \sim r^{-\alpha}$ between type count $S$ and type rank $r$, widely known as Zipf's law, is widely observed to varying degrees of fidelity. A secondary, summary relationship is Heaps' law, which states that the number of types scales sublinearly with the total number of observed tokens present in a growing system. Here, we propose an idealized model of a growing system that (1) deterministically produces arbitrary inverse power-law count rankings for types, and (2) allows us to determine the exact asymptotics of the type-token relationship. Our argument improves upon and remedies earlier work. We obtain a unified asymptotic expression for all values of $\alpha$, which corrects the special cases of $\alpha = 1$ and $\alpha \gg 1$. Our approach relies solely on the form of count rankings, avoids unnecessary approximations, and does not involve any stochastic mechanisms or sampling processes. We thereby demonstrate that a general type-token relationship arises solely as a consequence of Zipf's law. 

---
# Regularization Through Reasoning: Systematic Improvements in Language Model Classification via Explanation-Enhanced Fine-Tuning 

**Authors**: Vivswan Shah, Randy Cogill, Hanwei Yue, Gopinath Chennupati, Rinat Khaziev  

**Link**: [PDF](https://arxiv.org/pdf/2511.02044)  

**Abstract**: Fine-tuning LLMs for classification typically maps inputs directly to labels. We ask whether attaching brief explanations to each label during fine-tuning yields better models. We evaluate conversational response quality along three axes: naturalness, comprehensiveness, and on-topic adherence, each rated on 5-point scales. Using ensemble-generated data from multiple LLMs, we fine-tune a 7B-parameter model and test across six diverse conversational datasets. Across 18 dataset, task settings, label-plus-explanation training outperforms label-only baselines.
A central and unexpected result concerns random tokens. We replace human-written explanations with text that is syntactically incoherent yet vocabulary-aligned with the originals (e.g., shuffled or bag-of-words variants). Despite lacking semantics, these pseudo-explanations still improve accuracy over label-only training and often narrow much of the gap to true explanations. The effect persists across datasets and training seeds, indicating that gains arise less from meaning than from structure: the extra token budget encourages richer intermediate computation and acts as a regularizer that reduces over-confident shortcuts.
Internal analyses support this view: explanation-augmented models exhibit higher activation entropy in intermediate layers alongside sharper predictive mass at the output layer, consistent with increased deliberation before decision. Overall, explanation-augmented fine-tuning, whether with genuine rationales or carefully constructed random token sequences, improves accuracy and reliability for LLM classification while clarifying how token-level scaffolding shapes computation during inference. 

---
# TapOut: A Bandit-Based Approach to Dynamic Speculative Decoding 

**Authors**: Aditya Sridhar, Nish Sinnadurai, Sean Lie, Vithursan Thangarasa  

**Link**: [PDF](https://arxiv.org/pdf/2511.02017)  

**Abstract**: Speculative decoding accelerates LLMs by using a lightweight draft model to generate tokens autoregressively before verifying them in parallel with a larger target model. However, determining the optimal number of tokens to draft remains a key challenge limiting the approach's effectiveness. Dynamic speculative decoding aims to intelligently decide how many tokens to draft to achieve maximum speedups. Existing methods often rely on hand-tuned, sensitive thresholds (e.g., token entropy), which are costly to set and generalize poorly across models and domains. We propose TapOut, an online, training-free, plug-and-play algorithm for dynamic speculation policy selection using multi-armed bandits. Our approach employs a meta-algorithm that selects among multiple parameter-free dynamic speculation strategies based on past reward and exploration. We conduct extensive experiments across diverse model pairs and datasets, showing that TapOut achieves competitive or superior speedups compared to well-established dynamic speculation baselines without any hyperparameter tuning. 

---
# Retrieval-Augmented Multimodal Depression Detection 

**Authors**: Ruibo Hou, Shiyu Teng, Jiaqing Liu, Shurong Chai, Yinhao Li, Lanfen Lin, Yen-Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.01892)  

**Abstract**: Multimodal deep learning has shown promise in depression detection by integrating text, audio, and video signals. Recent work leverages sentiment analysis to enhance emotional understanding, yet suffers from high computational cost, domain mismatch, and static knowledge limitations. To address these issues, we propose a novel Retrieval-Augmented Generation (RAG) framework. Given a depression-related text, our method retrieves semantically relevant emotional content from a sentiment dataset and uses a Large Language Model (LLM) to generate an Emotion Prompt as an auxiliary modality. This prompt enriches emotional representation and improves interpretability. Experiments on the AVEC 2019 dataset show our approach achieves state-of-the-art performance with CCC of 0.593 and MAE of 3.95, surpassing previous transfer learning and multi-task learning baselines. 

---
# CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization 

**Authors**: Zijian Zhang, Rong Wang, Shiyang Li, Yuebo Luo, Mingyi Hong, Caiwen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2511.01884)  

**Abstract**: Developing efficient CUDA kernels is increasingly critical for AI applications such as large-scale LLM training. However, manual kernel design is both costly and time-consuming, motivating automatic approaches that leverage LLMs for code generation. Existing methods for automatic kernel generation, however, often produce low-efficiency kernels, incur high computational overhead, and fail to generalize across settings. In this work, we propose CudaForge, a training-free multi-agent workflow for CUDA kernel generation and optimization. Our workflow is inspired by the iterative workflow of human experts, which contains steps such as developing initial kernels, testing correctness, analyzing hardware feedback, and iterative improvement. More specifically, CudaForge employs two LLM agents: a Coder and a Judge, that iteratively generate, correct, and optimize CUDA kernels, while integrating hardware feedback such as Nsight Compute (NCU) metrics. In extensive evaluations, we show that CudaForge, by leveraging base models like OpenAI-o3, achieves 97.6\% correctness of generated kernels and an average 1.68$\times$ speedup over PyTorch baselines, substantially surpassing state-of-the-art models including OpenAI-o3 and Kevin on KernelBench. Beyond accuracy and speed, CudaForge demonstrates strong generalization across GPUs (A100, RTX 6000, 4090, 3090) and base models (OpenAI-o3, GPT-5, gpt-oss-120B, Claude-Sonnet-4, QwQ-32B), while maintaining high efficiency. In particular, generating an optimized kernel takes about 26.5 minutes on one RTX6000 and incurs about \$ 0.3 API cost, which is significantly cheaper than existing agentic work that costs 6 H100 hours and \$ 5 API cost per kernel. Our results highlight that multi-agent, training-free workflows can enable cost-effective, generalizable, and high-performance CUDA kernel optimization. Code available at this https URL 

---
# SciDaSynth: Interactive Structured Data Extraction from Scientific Literature with Large Language Model 

**Authors**: Xingbo Wang, Samantha L. Huey, Rui Sheng, Saurabh Mehta, Fei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2404.13765)  

**Abstract**: The explosion of scientific literature has made the efficient and accurate extraction of structured data a critical component for advancing scientific knowledge and supporting evidence-based decision-making. However, existing tools often struggle to extract and structure multimodal, varied, and inconsistent information across documents into standardized formats. We introduce SciDaSynth, a novel interactive system powered by large language models (LLMs) that automatically generates structured data tables according to users' queries by integrating information from diverse sources, including text, tables, and figures. Furthermore, SciDaSynth supports efficient table data validation and refinement, featuring multi-faceted visual summaries and semantic grouping capabilities to resolve cross-document data inconsistencies. A within-subjects study with nutrition and NLP researchers demonstrates SciDaSynth's effectiveness in producing high-quality structured data more efficiently than baseline methods. We discuss design implications for human-AI collaborative systems supporting data extraction tasks. The system code is available at this https URL 

---
