# Unleashing the Native Recommendation Potential: LLM-Based Generative Recommendation via Structured Term Identifiers 

**Authors**: Zhiyang Zhang, Junda She, Kuo Cai, Bo Chen, Shiyao Wang, Xinchen Luo, Qiang Luo, Ruiming Tang, Han Li, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.06798)  

**Abstract**: Leveraging the vast open-world knowledge and understanding capabilities of Large Language Models (LLMs) to develop general-purpose, semantically-aware recommender systems has emerged as a pivotal research direction in generative recommendation. However, existing methods face bottlenecks in constructing item identifiers. Text-based methods introduce LLMs' vast output space, leading to hallucination, while methods based on Semantic IDs (SIDs) encounter a semantic gap between SIDs and LLMs' native vocabulary, requiring costly vocabulary expansion and alignment training. To address this, this paper introduces Term IDs (TIDs), defined as a set of semantically rich and standardized textual keywords, to serve as robust item identifiers. We propose GRLM, a novel framework centered on TIDs, employs Context-aware Term Generation to convert item's metadata into standardized TIDs and utilizes Integrative Instruction Fine-tuning to collaboratively optimize term internalization and sequential recommendation. Additionally, Elastic Identifier Grounding is designed for robust item mapping. Extensive experiments on real-world datasets demonstrate that GRLM significantly outperforms baselines across multiple scenarios, pointing a promising direction for generalizable and high-performance generative recommendation systems. 

---
# FinCARDS: Card-Based Analyst Reranking for Financial Document Question Answering 

**Authors**: Yixi Zhou, Fan Zhang, Yu Chen, Haipeng Zhang, Preslav Nakov, Zhuohan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2601.06992)  

**Abstract**: Financial question answering (QA) over long corporate filings requires evidence to satisfy strict constraints on entities, financial metrics, fiscal periods, and numeric values. However, existing LLM-based rerankers primarily optimize semantic relevance, leading to unstable rankings and opaque decisions on long documents. We propose FinCards, a structured reranking framework that reframes financial evidence selection as constraint satisfaction under a finance-aware schema. FinCards represents filing chunks and questions using aligned schema fields (entities, metrics, periods, and numeric spans), enabling deterministic field-level matching. Evidence is selected via a multi-stage tournament reranking with stability-aware aggregation, producing auditable decision traces. Across two corporate filing QA benchmarks, FinCards substantially improves early-rank retrieval over both lexical and LLM-based reranking baselines, while reducing ranking variance, without requiring model fine-tuning or unpredictable inference budgets. Our code is available at this https URL. 

---
# DiSCo: Making Absence Visible in Intelligent Summarization Interfaces 

**Authors**: Eran Fainman, Hagit Ben Shoshan, Adir Solomon, Osnat Mokryn  

**Link**: [PDF](https://arxiv.org/pdf/2601.07229)  

**Abstract**: Intelligent interfaces increasingly use large language models to summarize user-generated content, yet these summaries emphasize what is mentioned while overlooking what is missing. This presence bias can mislead users who rely on summaries to make decisions. We present Domain Informed Summarization through Contrast (DiSCo), an expectation-based computational approach that makes absences visible by comparing each entity's content with domain topical expectations captured in reference distributions of aspects typically discussed in comparable accommodations. This comparison identifies aspects that are either unusually emphasized or missing relative to domain norms and integrates them into the generated text. In a user study across three accommodation domains, namely ski, beach, and city center, DiSCo summaries were rated as more detailed and useful for decision making than baseline large language model summaries, although slightly harder to read. The findings show that modeling expectations reduces presence bias and improves both transparency and decision support in intelligent summarization interfaces. 

---
# Is Agentic RAG worth it? An experimental comparison of RAG approaches 

**Authors**: Pietro Ferrazzi, Milica Cvjeticanin, Alessio Piraccini, Davide Giannuzzi  

**Link**: [PDF](https://arxiv.org/pdf/2601.07711)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are usually defined by the combination of a generator and a retrieval component that extracts textual context from a knowledge base to answer user queries. However, such basic implementations exhibit several limitations, including noisy or suboptimal retrieval, misuse of retrieval for out-of-scope queries, weak query-document matching, and variability or cost associated with the generator. These shortcomings have motivated the development of "Enhanced" RAG, where dedicated modules are introduced to address specific weaknesses in the workflow. More recently, the growing self-reflective capabilities of Large Language Models (LLMs) have enabled a new paradigm, which we refer to as "Agentic" RAG. In this approach, the LLM orchestrates the entire process-deciding which actions to perform, when to perform them, and whether to iterate-thereby reducing reliance on fixed, manually engineered modules. Despite the rapid adoption of both paradigms, it remains unclear which approach is preferable under which conditions. In this work, we conduct an extensive, empirically driven evaluation of Enhanced and Agentic RAG across multiple scenarios and dimensions. Our results provide practical insights into the trade-offs between the two paradigms, offering guidance on selecting the most effective RAG design for real-world applications, considering both costs and performance. 

---
# Kinship Data Benchmark for Multi-hop Reasoning 

**Authors**: Tianda Sun, Dimitar Kazakov  

**Link**: [PDF](https://arxiv.org/pdf/2601.07794)  

**Abstract**: Large language models (LLMs) are increasingly evaluated on their ability to perform multi-hop reasoning, i.e., to combine multiple pieces of information into a coherent inference. We introduce KinshipQA, a benchmark designed to probe this capability through reasoning over kinship relations. The central contribution of our work is a generative pipeline that produces, on demand, large-scale, realistic, and culture-specific genealogical data: collections of interconnected family trees that satisfy explicit marriage constraints associated with different kinship systems. This allows task difficulty, cultural assumptions, and relational depth to be systematically controlled and varied. From these genealogies, we derive textual inference tasks that require reasoning over implicit relational chains. We evaluate the resulting benchmark using six state-of-the-art LLMs, spanning both open-source and closed-source models, under a uniform zero-shot protocol with deterministic decoding. Performance is measured using exact-match and set-based metrics. Our results demonstrate that KinshipQA yields a wide spread of outcomes and exposes systematic differences in multi-hop reasoning across models and cultural settings. 

---
# Enhancing Self-Correction in Large Language Models through Multi-Perspective Reflection 

**Authors**: Mariana Costa, Alberlucia Rafael Soarez, Daniel Kim, Camila Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2601.07780)  

**Abstract**: While Chain-of-Thought (CoT) prompting advances LLM reasoning, challenges persist in consistency, accuracy, and self-correction, especially for complex or ethically sensitive tasks. Existing single-dimensional reflection methods offer insufficient improvements. We propose MyGO Poly-Reflective Chain-of-Thought (PR-CoT), a novel methodology employing structured multi-perspective reflection. After initial CoT, PR-CoT guides the LLM to self-assess its reasoning across multiple predefined angles: logical consistency, information completeness, biases/ethics, and alternative solutions. Implemented purely via prompt engineering, this process refines the initial CoT into a more robust and accurate final answer without model retraining. Experiments across arithmetic, commonsense, ethical decision-making, and logical puzzles, using GPT-three point five and GPT-four models, demonstrate PR-CoT's superior performance. It significantly outperforms traditional CoT and existing reflection methods in logical consistency and error correction, with notable gains in nuanced domains like ethical decision-making. Ablation studies, human evaluations, and qualitative analyses further validate the contribution of each reflection perspective and the overall efficacy of our poly-reflective paradigm in fostering more reliable LLM reasoning. 

---
# The Confidence Trap: Gender Bias and Predictive Certainty in LLMs 

**Authors**: Ahmed Sabir, Markus Kängsepp, Rajesh Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2601.07806)  

**Abstract**: The increased use of Large Language Models (LLMs) in sensitive domains leads to growing interest in how their confidence scores correspond to fairness and bias. This study examines the alignment between LLM-predicted confidence and human-annotated bias judgments. Focusing on gender bias, the research investigates probability confidence calibration in contexts involving gendered pronoun resolution. The goal is to evaluate if calibration metrics based on predicted confidence scores effectively capture fairness-related disparities in LLMs. The results show that, among the six state-of-the-art models, Gemma-2 demonstrates the worst calibration according to the gender bias benchmark. The primary contribution of this work is a fairness-aware evaluation of LLMs' confidence calibration, offering guidance for ethical deployment. In addition, we introduce a new calibration metric, Gender-ECE, designed to measure gender disparities in resolution tasks. 

---
# Structure First, Reason Next: Enhancing a Large Language Model using Knowledge Graph for Numerical Reasoning in Financial Documents 

**Authors**: Aryan Mishra, Akash Anil  

**Link**: [PDF](https://arxiv.org/pdf/2601.07754)  

**Abstract**: Numerical reasoning is an important task in the analysis of financial documents. It helps in understanding and performing numerical predictions with logical conclusions for the given query seeking answers from financial texts. Recently, Large Language Models (LLMs) have shown promising results in multiple Question-Answering (Q-A) systems with the capability of logical reasoning. As documents related to finance often consist of long and complex financial contexts, LLMs appear well-suited for building high-quality automated financial question-answering systems. However, LLMs often face challenges in accurately processing the various numbers within financial reports. Extracting numerical data from unstructured text and semi-structured tables, and reliably performing accurate calculations, remains a significant bottleneck for numerical reasoning in most state-of-the-art LLMs. Recent studies have shown that structured data augmentations, such as Knowledge Graphs (KGs), have notably improved the predictions of LLMs along with logical explanations. Thus, it is an important requirement to consider inherent structured information in financial reports while using LLMs for various financial analytics. This paper proposes a framework to incorporate structured information using KGs along with LLM predictions for numerical reasoning tasks. The KGs are extracted using a proposed schema inherently from the document under processing. We evaluated our proposed framework over the benchmark data FinQA, using an open-source LLM, namely Llama 3.1 8B Instruct. We observed that the proposed framework improved execution accuracy by approximately 12% relative to the vanilla LLM. 

---
# Learning Through Dialogue: Unpacking the Dynamics of Human-LLM Conversations on Political Issues 

**Authors**: Shaz Furniturewala, Gerard Christopher Yeo, Kokil Jaidka  

**Link**: [PDF](https://arxiv.org/pdf/2601.07796)  

**Abstract**: Large language models (LLMs) are increasingly used as conversational partners for learning, yet the interactional dynamics supporting users' learning and engagement are understudied. We analyze the linguistic and interactional features from both LLM and participant chats across 397 human-LLM conversations about socio-political issues to identify the mechanisms and conditions under which LLM explanations shape changes in political knowledge and confidence. Mediation analyses reveal that LLM explanatory richness partially supports confidence by fostering users' reflective insight, whereas its effect on knowledge gain operates entirely through users' cognitive engagement. Moderation analyses show that these effects are highly conditional and vary by political efficacy. Confidence gains depend on how high-efficacy users experience and resolve uncertainty. Knowledge gains depend on high-efficacy users' ability to leverage extended interaction, with longer conversations benefiting primarily reflective users. In summary, we find that learning from LLMs is an interactional achievement, not a uniform outcome of better explanations. The findings underscore the importance of aligning LLM explanatory behavior with users' engagement states to support effective learning in designing Human-AI interactive systems. 

---
# Exploring the Meta-level Reasoning of Large Language Models via a Tool-based Multi-hop Tabular Question Answering Task 

**Authors**: Nick Ferguson, Alan Bundy, Kwabena Nuamah  

**Link**: [PDF](https://arxiv.org/pdf/2601.07696)  

**Abstract**: Recent advancements in Large Language Models (LLMs) are increasingly focused on "reasoning" ability, a concept with many overlapping definitions in the LLM discourse. We take a more structured approach, distinguishing meta-level reasoning (denoting the process of reasoning about intermediate steps required to solve a task) from object-level reasoning (which concerns the low-level execution of the aforementioned steps.) We design a novel question answering task, which is based around the values of geopolitical indicators for various countries over various years. Questions require breaking down into intermediate steps, retrieval of data, and mathematical operations over that data. The meta-level reasoning ability of LLMs is analysed by examining the selection of appropriate tools for answering questions. To bring greater depth to the analysis of LLMs beyond final answer accuracy, our task contains 'essential actions' against which we can compare the tool call output of LLMs to infer the strength of reasoning ability. We find that LLMs demonstrate good meta-level reasoning on our task, yet are flawed in some aspects of task understanding. We find that n-shot prompting has little effect on accuracy; error messages encountered do not often deteriorate performance; and provide additional evidence for the poor numeracy of LLMs. Finally, we discuss the generalisation and limitation of our findings to other task domains. 

---
# Proof of Time: A Benchmark for Evaluating Scientific Idea Judgments 

**Authors**: Bingyang Ye, Shan Chen, Jingxuan Tu, Chen Liu, Zidi Xiong, Samuel Schmidgall, Danielle S. Bitterman  

**Link**: [PDF](https://arxiv.org/pdf/2601.07606)  

**Abstract**: Large language models are increasingly being used to assess and forecast research ideas, yet we lack scalable ways to evaluate the quality of models' judgments about these scientific ideas. Towards this goal, we introduce PoT, a semi-verifiable benchmarking framework that links scientific idea judgments to downstream signals that become observable later (e.g., citations and shifts in researchers' agendas). PoT freezes a pre-cutoff snapshot of evidence in an offline sandbox and asks models to forecast post-cutoff outcomes, enabling verifiable evaluation when ground truth arrives, scalable benchmarking without exhaustive expert annotation, and analysis of human-model misalignment against signals such as peer-review awards. In addition, PoT provides a controlled testbed for agent-based research judgments that evaluate scientific ideas, comparing tool-using agents to non-agent baselines under prompt ablations and budget scaling. Across 30,000+ instances spanning four benchmark domains, we find that, compared with non-agent baselines, higher interaction budgets generally improve agent performance, while the benefit of tool use is strongly task-dependent. By combining time-partitioned, future-verifiable targets with an offline sandbox for tool use, PoT supports scalable evaluation of agents on future-facing scientific idea judgment tasks. 

---
# A Unified Framework for Emotion Recognition and Sentiment Analysis via Expert-Guided Multimodal Fusion with Large Language Models 

**Authors**: Jiaqi Qiao, Xiujuan Xu, Xinran Li, Yu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07565)  

**Abstract**: Multimodal emotion understanding requires effective integration of text, audio, and visual modalities for both discrete emotion recognition and continuous sentiment analysis. We present EGMF, a unified framework combining expert-guided multimodal fusion with large language models. Our approach features three specialized expert networks--a fine-grained local expert for subtle emotional nuances, a semantic correlation expert for cross-modal relationships, and a global context expert for long-range dependencies--adaptively integrated through hierarchical dynamic gating for context-aware feature selection. Enhanced multimodal representations are integrated with LLMs via pseudo token injection and prompt-based conditioning, enabling a single generative framework to handle both classification and regression through natural language generation. We employ LoRA fine-tuning for computational efficiency. Experiments on bilingual benchmarks (MELD, CHERMA, MOSEI, SIMS-V2) demonstrate consistent improvements over state-of-the-art methods, with superior cross-lingual robustness revealing universal patterns in multimodal emotional expressions across English and Chinese. We will release the source code publicly. 

---
# KALE: Enhancing Knowledge Manipulation in Large Language Models via Knowledge-aware Learning 

**Authors**: Qitan Lv, Tianyu Liu, Qiaosheng Zhang, Xingcheng Xu, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07430)  

**Abstract**: Despite the impressive performance of large language models (LLMs) pretrained on vast knowledge corpora, advancing their knowledge manipulation-the ability to effectively recall, reason, and transfer relevant knowledge-remains challenging. Existing methods mainly leverage Supervised Fine-Tuning (SFT) on labeled datasets to enhance LLMs' knowledge manipulation ability. However, we observe that SFT models still exhibit the known&incorrect phenomenon, where they explicitly possess relevant knowledge for a given question but fail to leverage it for correct answers. To address this challenge, we propose KALE (Knowledge-Aware LEarning)-a post-training framework that leverages knowledge graphs (KGs) to generate high-quality rationales and enhance LLMs' knowledge manipulation ability. Specifically, KALE first introduces a Knowledge-Induced (KI) data synthesis method that efficiently extracts multi-hop reasoning paths from KGs to generate high-quality rationales for question-answer pairs. Then, KALE employs a Knowledge-Aware (KA) fine-tuning paradigm that enhances knowledge manipulation by internalizing rationale-guided reasoning through minimizing the KL divergence between predictions with and without rationales. Extensive experiments on eight popular benchmarks across six different LLMs demonstrate the effectiveness of KALE, achieving accuracy improvements of up to 11.72% and an average of 4.18%. 

---
# Thinking Before Constraining: A Unified Decoding Framework for Large Language Models 

**Authors**: Ngoc Trinh Hung Nguyen, Alonso Silva, Laith Zumot, Liubov Tupikina, Armen Aghasaryan, Mehwish Alam  

**Link**: [PDF](https://arxiv.org/pdf/2601.07525)  

**Abstract**: Natural generation allows Language Models (LMs) to produce free-form responses with rich reasoning, but the lack of guaranteed structure makes outputs difficult to parse or verify. Structured generation, or constrained decoding, addresses this drawback by producing content in standardized formats such as JSON, ensuring consistency and guaranteed-parsable outputs, but it can inadvertently restrict the model's reasoning capabilities. In this work, we propose a simple approach that combines the advantages of both natural and structured generation. By allowing LLMs to reason freely until specific trigger tokens are generated, and then switching to structured generation, our method preserves the expressive power of natural language reasoning while ensuring the reliability of structured outputs. We further evaluate our approach on several datasets, covering both classification and reasoning tasks, to demonstrate its effectiveness, achieving a substantial gain of up to 27% in accuracy compared to natural generation, while requiring only a small overhead of 10-20 extra tokens. 

---
# Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations 

**Authors**: Wen Luo, Guangyue Peng, Wei Li, Shaohang Wei, Feifan Song, Liang Wang, Nan Yang, Xingxing Zhang, Jing Jin, Furu Wei, Houfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07422)  

**Abstract**: Despite their impressive capabilities, large language models (LLMs) frequently generate hallucinations. Previous work shows that their internal states encode rich signals of truthfulness, yet the origins and mechanisms of these signals remain unclear. In this paper, we demonstrate that truthfulness cues arise from two distinct information pathways: (1) a Question-Anchored pathway that depends on question-answer information flow, and (2) an Answer-Anchored pathway that derives self-contained evidence from the generated answer itself. First, we validate and disentangle these pathways through attention knockout and token patching. Afterwards, we uncover notable and intriguing properties of these two mechanisms. Further experiments reveal that (1) the two mechanisms are closely associated with LLM knowledge boundaries; and (2) internal representations are aware of their distinctions. Finally, building on these insightful findings, two applications are proposed to enhance hallucination detection performance. Overall, our work provides new insight into how LLMs internally encode truthfulness, offering directions for more reliable and self-aware generative systems. 

---
# Judging Against the Reference: Uncovering Knowledge-Driven Failures in LLM-Judges on QA Evaluation 

**Authors**: Dongryeol Lee, Yerin Hwang, Taegwan Kang, Minwoo Lee, Younhyung Chae, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2601.07506)  

**Abstract**: While large language models (LLMs) are increasingly used as automatic judges for question answering (QA) and other reference-conditioned evaluation tasks, little is known about their ability to adhere to a provided reference. We identify a critical failure mode of such reference-based LLM QA evaluation: when the provided reference conflicts with the judge model's parametric knowledge, the resulting scores become unreliable, substantially degrading evaluation fidelity. To study this phenomenon systematically, we introduce a controlled swapped-reference QA framework that induces reference-belief conflicts. Specifically, we replace the reference answer with an incorrect entity and construct diverse pairings of original and swapped references with correspondingly aligned candidate answers. Surprisingly, grading reliability drops sharply under swapped references across a broad set of judge models. We empirically show that this vulnerability is driven by judges' over-reliance on parametric knowledge, leading judges to disregard the given reference under conflict. Finally, we find that this failure persists under common prompt-based mitigation strategies, highlighting a fundamental limitation of LLM-as-a-judge evaluation and motivating reference-based protocols that enforce stronger adherence to the provided reference. 

---
# Interpretable Text Classification Applied to the Detection of LLM-generated Creative Writing 

**Authors**: Minerva Suvanto, Andrea McGlinchey, Mattias Wahde, Peter J Barclay  

**Link**: [PDF](https://arxiv.org/pdf/2601.07368)  

**Abstract**: We consider the problem of distinguishing human-written creative fiction (excerpts from novels) from similar text generated by an LLM. Our results show that, while human observers perform poorly (near chance levels) on this binary classification task, a variety of machine-learning models achieve accuracy in the range 0.93 - 0.98 over a previously unseen test set, even using only short samples and single-token (unigram) features. We therefore employ an inherently interpretable (linear) classifier (with a test accuracy of 0.98), in order to elucidate the underlying reasons for this high accuracy. In our analysis, we identify specific unigram features indicative of LLM-generated text, one of the most important being that the LLM tends to use a larger variety of synonyms, thereby skewing the probability distributions in a manner that is easy to detect for a machine learning classifier, yet very difficult for a human observer. Four additional explanation categories were also identified, namely, temporal drift, Americanisms, foreign language usage, and colloquialisms. As identification of the AI-generated text depends on a constellation of such features, the classification appears robust, and therefore not easy to circumvent by malicious actors intent on misrepresenting AI-generated text as human work. 

---
# Beyond Literal Mapping: Benchmarking and Improving Non-Literal Translation Evaluation 

**Authors**: Yanzhi Tian, Cunxiang Wang, Zeming Liu, Heyan Huang, Wenbo Yu, Dawei Song, Jie Tang, Yuhang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.07338)  

**Abstract**: Large Language Models (LLMs) have significantly advanced Machine Translation (MT), applying them to linguistically complex domains-such as Social Network Services, literature etc. In these scenarios, translations often require handling non-literal expressions, leading to the inaccuracy of MT metrics. To systematically investigate the reliability of MT metrics, we first curate a meta-evaluation dataset focused on non-literal translations, namely MENT. MENT encompasses four non-literal translation domains and features source sentences paired with translations from diverse MT systems, with 7,530 human-annotated scores on translation quality. Experimental results reveal the inaccuracies of traditional MT metrics and the limitations of LLM-as-a-Judge, particularly the knowledge cutoff and score inconsistency problem. To mitigate these limitations, we propose RATE, a novel agentic translation evaluation framework, centered by a reflective Core Agent that dynamically invokes specialized sub-agents. Experimental results indicate the efficacy of RATE, achieving an improvement of at least 3.2 meta score compared with current metrics. Further experiments demonstrate the robustness of RATE to general-domain MT evaluation. Code and dataset are available at: this https URL. 

---
# From RAG to Agentic RAG for Faithful Islamic Question Answering 

**Authors**: Gagan Bhatia, Hamdy Mubarak, Mustafa Jarrar, George Mikros, Fadi Zaraket, Mahmoud Alhirthani, Mutaz Al-Khatib, Logan Cochrane, Kareem Darwish, Rashid Yahiaoui, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2601.07528)  

**Abstract**: LLMs are increasingly used for Islamic question answering, where ungrounded responses may carry serious religious consequences. Yet standard MCQ/MRC-style evaluations do not capture key real-world failure modes, notably free-form hallucinations and whether models appropriately abstain when evidence is lacking. To shed a light on this aspect we introduce ISLAMICFAITHQA, a 3,810-item bilingual (Arabic/English) generative benchmark with atomic single-gold answers, which enables direct measurement of hallucination and abstention. We additionally developed an end-to-end grounded Islamic modelling suite consisting of (i) 25K Arabic text-grounded SFT reasoning pairs, (ii) 5K bilingual preference samples for reward-guided alignment, and (iii) a verse-level Qur'an retrieval corpus of $\sim$6k atomic verses (ayat). Building on these resources, we develop an agentic Quran-grounding framework (agentic RAG) that uses structured tool calls for iterative evidence seeking and answer revision. Experiments across Arabic-centric and multilingual LLMs show that retrieval improves correctness and that agentic RAG yields the largest gains beyond standard RAG, achieving state-of-the-art performance and stronger Arabic-English robustness even with a small model (i.e., Qwen3 4B). We will make the experimental resources and datasets publicly available for the community. 

---
# DiffER: Diffusion Entity-Relation Modeling for Reversal Curse in Diffusion Large Language Models 

**Authors**: Shaokai He, Kaiwen Wei, Xinyi Zeng, Xiang Chen, Xue Yang, Zhenyang Li, Jiang Zhong, Yu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2601.07347)  

**Abstract**: The "reversal curse" refers to the phenomenon where large language models (LLMs) exhibit predominantly unidirectional behavior when processing logically bidirectional relationships. Prior work attributed this to autoregressive training -- predicting the next token inherently favors left-to-right information flow over genuine bidirectional knowledge associations. However, we observe that Diffusion LLMs (DLLMs), despite being trained bidirectionally, also suffer from the reversal curse. To investigate the root causes, we conduct systematic experiments on DLLMs and identify three key reasons: 1) entity fragmentation during training, 2) data asymmetry, and 3) missing entity relations. Motivated by the analysis of these reasons, we propose Diffusion Entity-Relation Modeling (DiffER), which addresses the reversal curse through entity-aware training and balanced data construction. Specifically, DiffER introduces whole-entity masking, which mitigates entity fragmentation by predicting complete entities in a single step. DiffER further employs distribution-symmetric and relation-enhanced data construction strategies to alleviate data asymmetry and missing relations. Extensive experiments demonstrate that DiffER effectively alleviates the reversal curse in Diffusion LLMs, offering new perspectives for future research. 

---
# PsyCLIENT: Client Simulation via Conversational Trajectory Modeling for Trainee Practice and Model Evaluation in Mental Health Counseling 

**Authors**: Huachuan Qiu, Zhaoming Chen, Yuqian Chen, Yuan Xie, Yu Lu, Zhenzhong Lan  

**Link**: [PDF](https://arxiv.org/pdf/2601.07312)  

**Abstract**: LLM-based client simulation has emerged as a promising tool for training novice counselors and evaluating automated counseling systems. However, existing client simulation approaches face three key challenges: (1) limited diversity and realism in client profiles, (2) the lack of a principled framework for modeling realistic client behaviors, and (3) a scarcity in Chinese-language settings. To address these limitations, we propose PsyCLIENT, a novel simulation framework grounded in conversational trajectory modeling. By conditioning LLM generation on predefined real-world trajectories that incorporate explicit behavior labels and content constraints, our approach ensures diverse and realistic interactions. We further introduce PsyCLIENT-CP, the first open-source Chinese client profile dataset, covering 60 distinct counseling topics. Comprehensive evaluations involving licensed professional counselors demonstrate that PsyCLIENT significantly outperforms baselines in terms of authenticity and training effectiveness. Notably, the simulated clients are nearly indistinguishable from human clients, achieving an about 95\% expert confusion rate in discrimination tasks. These findings indicate that conversational trajectory modeling effectively bridges the gap between theoretical client profiles and dynamic, realistic simulations, offering a robust solution for mental health education and research. Code and data will be released to facilitate future research in mental health counseling. 

---
# The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in Tool-Use Agents 

**Authors**: Weihao Xuan, Qingcheng Zeng, Heli Qi, Yunze Xiao, Junjue Wang, Naoto Yokoya  

**Link**: [PDF](https://arxiv.org/pdf/2601.07264)  

**Abstract**: Autonomous agents based on large language models (LLMs) are rapidly evolving to handle multi-turn tasks, but ensuring their trustworthiness remains a critical challenge. A fundamental pillar of this trustworthiness is calibration, which refers to an agent's ability to express confidence that reliably reflects its actual performance. While calibration is well-established for static models, its dynamics in tool-integrated agentic workflows remain underexplored. In this work, we systematically investigate verbalized calibration in tool-use agents, revealing a fundamental confidence dichotomy driven by tool type. Specifically, our pilot study identifies that evidence tools (e.g., web search) systematically induce severe overconfidence due to inherent noise in retrieved information, while verification tools (e.g., code interpreters) can ground reasoning through deterministic feedback and mitigate miscalibration. To robustly improve calibration across tool types, we propose a reinforcement learning (RL) fine-tuning framework that jointly optimizes task accuracy and calibration, supported by a holistic benchmark of reward designs. We demonstrate that our trained agents not only achieve superior calibration but also exhibit robust generalization from local training environments to noisy web settings and to distinct domains such as mathematical reasoning. Our results highlight the necessity of domain-specific calibration strategies for tool-use agents. More broadly, this work establishes a foundation for building self-aware agents that can reliably communicate uncertainty in high-stakes, real-world deployments. 

---
# ActiShade: Activating Overshadowed Knowledge to Guide Multi-Hop Reasoning in Large Language Models 

**Authors**: Huipeng Ma, Luan Zhang, Dandan Song, Linmei Hu, Yuhang Tian, Jun Yang, Changzhi Zhou, Chenhao Li, Yizhou Jin, Xudong Li, Meng Lin, Mingxing Zhang, Shuhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07260)  

**Abstract**: In multi-hop reasoning, multi-round retrieval-augmented generation (RAG) methods typically rely on LLM-generated content as the retrieval query. However, these approaches are inherently vulnerable to knowledge overshadowing - a phenomenon where critical information is overshadowed during generation. As a result, the LLM-generated content may be incomplete or inaccurate, leading to irrelevant retrieval and causing error accumulation during the iteration process. To address this challenge, we propose ActiShade, which detects and activates overshadowed knowledge to guide large language models (LLMs) in multi-hop reasoning. Specifically, ActiShade iteratively detects the overshadowed keyphrase in the given query, retrieves documents relevant to both the query and the overshadowed keyphrase, and generates a new query based on the retrieved documents to guide the next-round iteration. By supplementing the overshadowed knowledge during the formulation of next-round queries while minimizing the introduction of irrelevant noise, ActiShade reduces the error accumulation caused by knowledge overshadowing. Extensive experiments show that ActiShade outperforms existing methods across multiple datasets and LLMs. 

---
# Can Large Language Models Understand, Reason About, and Generate Code-Switched Text? 

**Authors**: Genta Indra Winata, David Anugraha, Patrick Amadeus Irawan, Anirban Das, Haneul Yoo, Paresh Dashore, Shreyas Kulkarni, Ruochen Zhang, Haruki Sakajo, Frederikus Hudi, Anaelia Ovalle, Syrielle Montariol, Felix Gaschi, Michael Anugraha, Rutuj Ravindra Puranik, Zawad Hayat Ahmed, Adril Putra Merin, Emmanuele Chersoni  

**Link**: [PDF](https://arxiv.org/pdf/2601.07153)  

**Abstract**: Code-switching is a pervasive phenomenon in multilingual communication, yet the robustness of large language models (LLMs) in mixed-language settings remains insufficiently understood. In this work, we present a comprehensive evaluation of LLM capabilities in understanding, reasoning over, and generating code-switched text. We introduce CodeMixQA a novel benchmark with high-quality human annotations, comprising 16 diverse parallel code-switched language-pair variants that span multiple geographic regions and code-switching patterns, and include both original scripts and their transliterated forms. Using this benchmark, we analyze the reasoning behavior of LLMs on code-switched question-answering tasks, shedding light on how models process and reason over mixed-language inputs. We further conduct a systematic evaluation of LLM-generated synthetic code-switched text, focusing on both naturalness and semantic fidelity, and uncover key limitations in current generation capabilities. Our findings reveal persistent challenges in both reasoning and generation under code-switching conditions and provide actionable insights for building more robust multilingual LLMs. We release the dataset and code as open source. 

---
# Relink: Constructing Query-Driven Evidence Graph On-the-Fly for GraphRAG 

**Authors**: Manzong Huang, Chenyang Bu, Yi He, Xingrui Zhuo, Xindong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07192)  

**Abstract**: Graph-based Retrieval-Augmented Generation (GraphRAG) mitigates hallucinations in Large Language Models (LLMs) by grounding them in structured knowledge. However, current GraphRAG methods are constrained by a prevailing \textit{build-then-reason} paradigm, which relies on a static, pre-constructed Knowledge Graph (KG). This paradigm faces two critical challenges. First, the KG's inherent incompleteness often breaks reasoning paths. Second, the graph's low signal-to-noise ratio introduces distractor facts, presenting query-relevant but misleading knowledge that disrupts the reasoning process.
To address these challenges, we argue for a \textit{reason-and-construct} paradigm and propose Relink, a framework that dynamically builds a query-specific evidence graph. To tackle incompleteness, \textbf{Relink} instantiates required facts from a latent relation pool derived from the original text corpus, repairing broken paths on the fly. To handle misleading or distractor facts, Relink employs a unified, query-aware evaluation strategy that jointly considers candidates from both the KG and latent relations, selecting those most useful for answering the query rather than relying on their pre-existence. This empowers Relink to actively discard distractor facts and construct the most faithful and precise evidence path for each query.
Extensive experiments on five Open-Domain Question Answering benchmarks show that Relink achieves significant average improvements of 5.4\% in EM and 5.2\% in F1 over leading GraphRAG baselines, demonstrating the superiority of our proposed framework. 

---
# MI-PRUN: Optimize Large Language Model Pruning via Mutual Information 

**Authors**: Hao Zhang, Zhibin Zhang, Guangxin Wu, He Chen, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.07212)  

**Abstract**: Large Language Models (LLMs) have become indispensable across various domains, but this comes at the cost of substantial computational and memory resources. Model pruning addresses this by removing redundant components from models. In particular, block pruning can achieve significant compression and inference acceleration. However, existing block pruning methods are often unstable and struggle to attain globally optimal solutions. In this paper, we propose a mutual information based pruning method MI-PRUN for LLMs. Specifically, we leverages mutual information to identify redundant blocks by evaluating transitions in hidden states. Additionally, we incorporate the Data Processing Inequality (DPI) to reveal the relationship between the importance of entire contiguous blocks and that of individual blocks. Moreover, we develop the Fast-Block-Select algorithm, which iteratively updates block combinations to achieve a globally optimal solution while significantly improving the efficiency. Extensive experiments across various models and datasets demonstrate the stability and effectiveness of our method. 

---
# Structured Reasoning for Large Language Models 

**Authors**: Jinyi Han, Zixiang Di, Zishang Jiang, Ying Liao, Jiaqing Liang, Yongqi Wang, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2601.07180)  

**Abstract**: Large language models (LLMs) achieve strong performance by generating long chains of thought, but longer traces always introduce redundant or ineffective reasoning steps. One typical behavior is that they often perform unnecessary verification and revisions even if they have reached the correct answers. This limitation stems from the unstructured nature of reasoning trajectories and the lack of targeted supervision for critical reasoning abilities. To address this, we propose Structured Reasoning (SCR), a framework that decouples reasoning trajectories into explicit, evaluable, and trainable components. We mainly implement SCR using a Generate-Verify-Revise paradigm. Specifically, we construct structured training data and apply Dynamic Termination Supervision to guide the model in deciding when to terminate reasoning. To avoid interference between learning signals for different reasoning abilities, we adopt a progressive two-stage reinforcement learning strategy: the first stage targets initial generation and self-verification, and the second stage focuses on revision. Extensive experiments on three backbone models show that SCR substantially improves reasoning efficiency and self-verification. Besides, compared with existing reasoning paradigms, it reduces output token length by up to 50%. 

---
# ReMIND: Orchestrating Modular Large Language Models for Controllable Serendipity A REM-Inspired System Design for Emergent Creative Ideation 

**Authors**: Makoto Sato  

**Link**: [PDF](https://arxiv.org/pdf/2601.07121)  

**Abstract**: Large language models (LLMs) are used not only for problem solving but also for creative ideation; however, eliciting serendipitous insights that are both novel and internally coherent remains difficult. While stochastic sampling promotes novelty, it often degrades consistency. Here, we propose ReMIND, a REM-inspired modular framework for ideation. ReMIND consists of four stages: wake, which generates a stable low-temperature semantic baseline; dream, which performs high-temperature exploratory generation; judge, which applies coarse evaluation to filter incoherent outputs and extract candidate ideas; and re-wake, which re-articulates selected ideas into coherent final outputs. By instantiating each stage as an independent LLM, ReMIND enables functional separation between exploration and consolidation. Parameter sweeps show that ReMIND reliably induces semantic exploration while preserving downstream stability. Embedding-based analyses confirm substantial semantic displacement during the dream phase, whereas external evaluations reveal that high-quality ideas emerge sporadically rather than as extrema along any single metric. These results suggest that serendipitous ideation in LLMs is a rare-event process best approached through system level design that shapes the conditions under which valuable ideas can emerge and be stabilized. ReMIND provides a general framework for studying the computational basis of serendipity and illustrates how modular LLM orchestration can bridge exploration and stabilization. 

---
# Codified Foreshadowing-Payoff Text Generation 

**Authors**: Longfei Yun, Kun Zhou, Yupeng Hou, Letian Peng, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07033)  

**Abstract**: Foreshadowing and payoff are ubiquitous narrative devices through which authors introduce commitments early in a story and resolve them through concrete, observable outcomes. However, despite advances in story generation, large language models (LLMs) frequently fail to bridge these long-range narrative dependencies, often leaving "Chekhov's guns" unfired even when the necessary context is present. Existing evaluations largely overlook this structural failure, focusing on surface-level coherence rather than the logical fulfillment of narrative setups. In this paper, we introduce Codified Foreshadowing-Payoff Generation (CFPG), a novel framework that reframes narrative quality through the lens of payoff realization. Recognizing that LLMs struggle to intuitively grasp the "triggering mechanism" of a foreshadowed event, CFPG transforms narrative continuity into a set of executable causal predicates. By mining and encoding Foreshadow-Trigger-Payoff triples from the BookSum corpus, we provide structured supervision that ensures foreshadowed commitments are not only mentioned but also temporally and logically fulfilled. Experiments demonstrate that CFPG significantly outperforms standard prompting baselines in payoff accuracy and narrative alignment. Our findings suggest that explicitly codifying narrative mechanics is essential for moving LLMs from surface-level fluency to genuine narrative competence. 

---
# Fine-Tuning vs. RAG for Multi-Hop Question Answering with Novel Knowledge 

**Authors**: Zhuoyi Yang, Yurun Song, Iftekhar Ahmed, Ian Harris  

**Link**: [PDF](https://arxiv.org/pdf/2601.07054)  

**Abstract**: Multi-hop question answering is widely used to evaluate the reasoning capabilities of large language models (LLMs), as it requires integrating multiple pieces of supporting knowledge to arrive at a correct answer. While prior work has explored different mechanisms for providing knowledge to LLMs, such as finetuning and retrieval-augmented generation (RAG), their relative effectiveness for multi-hop question answering remains insufficiently understood, particularly when the required knowledge is temporally novel.
In this paper, we systematically compare parametric and non-parametric knowledge injection methods for open-domain multi-hop question answering. We evaluate unsupervised fine-tuning (continual pretraining), supervised fine-tuning, and retrieval-augmented generation across three 7B-parameter open-source LLMs. Experiments are conducted on two benchmarks: QASC, a standard multi-hop science question answering dataset, and a newly constructed dataset of over 10,000 multi-hop questions derived from Wikipedia events in 2024, designed to test knowledge beyond the models' pretraining cutoff.
Our results show that unsupervised fine-tuning provides only limited gains over base models, suggesting that continual pretraining alone is insufficient for improving multi-hop reasoning accuracy. In contrast, retrieval-augmented generation yields substantial and consistent improvements, particularly when answering questions that rely on temporally novel information. Supervised fine-tuning achieves the highest overall accuracy across models and datasets. These findings highlight fundamental differences in how knowledge injection mechanisms support multi-hop question answering and underscore the importance of retrieval-based methods when external or compositional knowledge is required. 

---
# Mid-Think: Training-Free Intermediate-Budget Reasoning via Token-Level Triggers 

**Authors**: Wang Yang, Debargha Ganguly, Xinpeng Li, Chaoda Song, Shouren Wang, Vikash Singh, Vipin Chaudhary, Xiaotian Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.07036)  

**Abstract**: Hybrid reasoning language models are commonly controlled through high-level Think/No-think instructions to regulate reasoning behavior, yet we found that such mode switching is largely driven by a small set of trigger tokens rather than the instructions themselves. Through attention analysis and controlled prompting experiments, we show that a leading ``Okay'' token induces reasoning behavior, while the newline pattern following ``</think>'' suppresses it. Based on this observation, we propose Mid-Think, a simple training-free prompting format that combines these triggers to achieve intermediate-budget reasoning, consistently outperforming fixed-token and prompt-based baselines in terms of the accuracy-length trade-off. Furthermore, applying Mid-Think to RL training after SFT reduces training time by approximately 15% while improving final performance of Qwen3-8B on AIME from 69.8% to 72.4% and on GPQA from 58.5% to 61.1%, demonstrating its effectiveness for both inference-time control and RL-based reasoning training. 

---
# Solar Open Technical Report 

**Authors**: Sungrae Park, Sanghoon Kim, Jungho Cho, Gyoungjin Gim, Dawoon Jung, Mikyoung Cha, Eunhae Choo, Taekgyu Hong, Minbyul Jeong, SeHwan Joo, Minsoo Khang, Eunwon Kim, Minjeong Kim, Sujeong Kim, Yunsu Kim, Hyeonju Lee, Seunghyun Lee, Sukyung Lee, Siyoung Park, Gyungin Shin, Inseo Song, Wonho Song, Seonghoon Yang, Seungyoun Yi, Sanghoon Yoon, Jeonghyun Ko, Seyoung Song, Keunwoo Choi, Hwalsuk Lee, Sunghun Kim, Du-Seong Chang, Kyunghyun Cho, Junsuk Choe, Hwaran Lee, Jae-Gil Lee, KyungTae Lim, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2601.07022)  

**Abstract**: We introduce Solar Open, a 102B-parameter bilingual Mixture-of-Experts language model for underserved languages. Solar Open demonstrates a systematic methodology for building competitive LLMs by addressing three interconnected challenges. First, to train effectively despite data scarcity for underserved languages, we synthesize 4.5T tokens of high-quality, domain-specific, and RL-oriented data. Second, we coordinate this data through a progressive curriculum jointly optimizing composition, quality thresholds, and domain coverage across 20 trillion tokens. Third, to enable reasoning capabilities through scalable RL, we apply our proposed framework SnapPO for efficient optimization. Across benchmarks in English and Korean, Solar Open achieves competitive performance, demonstrating the effectiveness of this methodology for underserved language AI development. 

---
# MedTutor: A Retrieval-Augmented LLM System for Case-Based Medical Education 

**Authors**: Dongsuk Jang, Ziyao Shangguan, Kyle Tegtmeyer, Anurag Gupta, Jan Czerminski, Sophie Chheang, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2601.06979)  

**Abstract**: The learning process for medical residents presents significant challenges, demanding both the ability to interpret complex case reports and the rapid acquisition of accurate medical knowledge from reliable sources. Residents typically study case reports and engage in discussions with peers and mentors, but finding relevant educational materials and evidence to support their learning from these cases is often time-consuming and challenging. To address this, we introduce MedTutor, a novel system designed to augment resident training by automatically generating evidence-based educational content and multiple-choice questions from clinical case reports. MedTutor leverages a Retrieval-Augmented Generation (RAG) pipeline that takes clinical case reports as input and produces targeted educational materials. The system's architecture features a hybrid retrieval mechanism that synergistically queries a local knowledge base of medical textbooks and academic literature (using PubMed, Semantic Scholar APIs) for the latest related research, ensuring the generated content is both foundationally sound and current. The retrieved evidence is filtered and ordered using a state-of-the-art reranking model and then an LLM generates the final long-form output describing the main educational content regarding the case-report. We conduct a rigorous evaluation of the system. First, three radiologists assessed the quality of outputs, finding them to be of high clinical and educational value. Second, we perform a large scale evaluation using an LLM-as-a Judge to understand if LLMs can be used to evaluate the output of the system. Our analysis using correlation between LLMs outputs and human expert judgments reveals a moderate alignment and highlights the continued necessity of expert oversight. 

---
# RealMem: Benchmarking LLMs in Real-World Memory-Driven Interaction 

**Authors**: Haonan Bian, Zhiyuan Yao, Sen Hu, Zishan Xu, Shaolei Zhang, Yifu Guo, Ziliang Yang, Xueran Han, Huacan Wang, Ronghao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.06966)  

**Abstract**: As Large Language Models (LLMs) evolve from static dialogue interfaces to autonomous general agents, effective memory is paramount to ensuring long-term consistency. However, existing benchmarks primarily focus on casual conversation or task-oriented dialogue, failing to capture **"long-term project-oriented"** interactions where agents must track evolving goals.
To bridge this gap, we introduce **RealMem**, the first benchmark grounded in realistic project scenarios. RealMem comprises over 2,000 cross-session dialogues across eleven scenarios, utilizing natural user queries for evaluation.
We propose a synthesis pipeline that integrates Project Foundation Construction, Multi-Agent Dialogue Generation, and Memory and Schedule Management to simulate the dynamic evolution of memory. Experiments reveal that current memory systems face significant challenges in managing the long-term project states and dynamic context dependencies inherent in real-world projects.
Our code and datasets are available at [this https URL](this https URL). 

---
# Distributional Clarity: The Hidden Driver of RL-Friendliness in Large Language Models 

**Authors**: Shaoning Sun, Mingzhu Cai, Huang He, Bingjin Chen, Siqi Bao, Yujiu Yang, Hua Wu, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06911)  

**Abstract**: Language model families exhibit striking disparity in their capacity to benefit from reinforcement learning: under identical training, models like Qwen achieve substantial gains, while others like Llama yield limited improvements. Complementing data-centric approaches, we reveal that this disparity reflects a hidden structural property: \textbf{distributional clarity} in probability space. Through a three-stage analysis-from phenomenon to mechanism to interpretation-we uncover that RL-friendly models exhibit intra-class compactness and inter-class separation in their probability assignments to correct vs. incorrect responses. We quantify this clarity using the \textbf{Silhouette Coefficient} ($S$) and demonstrate that (1) high $S$ correlates strongly with RL performance; (2) low $S$ is associated with severe logic errors and reasoning instability. To confirm this property, we introduce a Silhouette-Aware Reweighting strategy that prioritizes low-$S$ samples during training. Experiments across six mathematical benchmarks show consistent improvements across all model families, with gains up to 5.9 points on AIME24. Our work establishes distributional clarity as a fundamental, trainable property underlying RL-Friendliness. 

---
# Paraphrasing Adversarial Attack on LLM-as-a-Reviewer 

**Authors**: Masahiro Kaneko  

**Link**: [PDF](https://arxiv.org/pdf/2601.06884)  

**Abstract**: The use of large language models (LLMs) in peer review systems has attracted growing attention, making it essential to examine their potential vulnerabilities. Prior attacks rely on prompt injection, which alters manuscript content and conflates injection susceptibility with evaluation robustness. We propose the Paraphrasing Adversarial Attack (PAA), a black-box optimization method that searches for paraphrased sequences yielding higher review scores while preserving semantic equivalence and linguistic naturalness. PAA leverages in-context learning, using previous paraphrases and their scores to guide candidate generation. Experiments across five ML and NLP conferences with three LLM reviewers and five attacking models show that PAA consistently increases review scores without changing the paper's claims. Human evaluation confirms that generated paraphrases maintain meaning and naturalness. We also find that attacked papers exhibit increased perplexity in reviews, offering a potential detection signal, and that paraphrasing submissions can partially mitigate attacks. 

---
# BiasLab: A Multilingual, Dual-Framing Framework for Robust Measurement of Output-Level Bias in Large Language Models 

**Authors**: William Guey, Wei Zhang, Pei-Luen Patrick Rau, Pierrick Bougault, Vitor D. de Moura, Bertan Ucar, Jose O. Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2601.06861)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in high-stakes contexts where their outputs influence real-world decisions. However, evaluating bias in LLM outputs remains methodologically challenging due to sensitivity to prompt wording, limited multilingual coverage, and the lack of standardized metrics that enable reliable comparison across models. This paper introduces BiasLab, an open-source, model-agnostic evaluation framework for quantifying output-level (extrinsic) bias through a multilingual, robustness-oriented experimental design. BiasLab constructs mirrored probe pairs under a strict dual-framing scheme: an affirmative assertion favoring Target A and a reverse assertion obtained by deterministic target substitution favoring Target B, while preserving identical linguistic structure. To reduce dependence on prompt templates, BiasLab performs repeated evaluation under randomized instructional wrappers and enforces a fixed-choice Likert response format to maximize comparability across models and languages. Responses are normalized into agreement labels using an LLM-based judge, aligned for polarity consistency across framings, and aggregated into quantitative bias indicators with descriptive statistics including effect sizes and neutrality rates. The framework supports evaluation across diverse bias axes, including demographic, cultural, political, and geopolitical topics, and produces reproducible artifacts such as structured reports and comparative visualizations. BiasLab contributes a standardized methodology for cross-lingual and framing-sensitive bias measurement that complements intrinsic and dataset-based audits, enabling researchers and institutions to benchmark robustness and make better-informed deployment decisions. 

---
# LLMs Can't Play Hangman: On the Necessity of a Private Working Memory for Language Agents 

**Authors**: Davide Baldelli, Ali Parviz, Amal Zouaq, Sarath Chandar  

**Link**: [PDF](https://arxiv.org/pdf/2601.06973)  

**Abstract**: As LLMs move from text completion toward autonomous agents, they remain constrained by the standard chat interface, which lacks private working memory. This raises a fundamental question: can agents reliably perform interactive tasks that depend on hidden state? We define Private State Interactive Tasks (PSITs), which require agents to generate and maintain hidden information while producing consistent public responses. We show theoretically that any agent restricted to the public conversation history cannot simultaneously preserve secrecy and consistency in PSITs, yielding an impossibility theorem. To empirically validate this limitation, we introduce a self-consistency testing protocol that evaluates whether agents can maintain a hidden secret across forked dialogue branches. Standard chat-based LLMs and retrieval-based memory baselines fail this test regardless of scale, demonstrating that semantic retrieval does not enable true state maintenance. To address this, we propose a novel architecture incorporating an explicit private working memory; we demonstrate that this mechanism restores consistency, establishing private state as a necessary component for interactive language agents. 

---
# AgentHallu: Benchmarking Automated Hallucination Attribution of LLM-based Agents 

**Authors**: Xuannan Liu, Xiao Yang, Zekun Li, Peipei Li, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2601.06818)  

**Abstract**: As LLM-based agents operate over sequential multi-step reasoning, hallucinations arising at intermediate steps risk propagating along the trajectory, thus degrading overall reliability. Unlike hallucination detection in single-turn responses, diagnosing hallucinations in multi-step workflows requires identifying which step causes the initial divergence. To fill this gap, we propose a new research task, automated hallucination attribution of LLM-based agents, aiming to identify the step responsible for the hallucination and explain why. To support this task, we introduce AgentHallu, a comprehensive benchmark with: (1) 693 high-quality trajectories spanning 7 agent frameworks and 5 domains, (2) a hallucination taxonomy organized into 5 categories (Planning, Retrieval, Reasoning, Human-Interaction, and Tool-Use) and 14 sub-categories, and (3) multi-level annotations curated by humans, covering binary labels, hallucination-responsible steps, and causal explanations. We evaluate 13 leading models, and results show the task is challenging even for top-tier models (like GPT-5, Gemini-2.5-Pro). The best-performing model achieves only 41.1\% step localization accuracy, where tool-use hallucinations are the most challenging at just 11.6\%. We believe AgentHallu will catalyze future research into developing robust, transparent, and reliable agentic systems. 

---
# Garbage Attention in Large Language Models: BOS Sink Heads and Sink-aware Pruning 

**Authors**: Jaewon Sok, Jewon Yeom, Seonghyeon Park, Jeongjae Park, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.06787)  

**Abstract**: Large Language Models (LLMs) are known to contain significant redundancy, yet a systematic explanation for why certain components, particularly in higher layers, are more redundant has remained elusive. In this work, we identify the BOS sink phenomenon as a key mechanism driving this layer-wise sensitivity. We show that attention heads with high BOS sink scores are strongly associated with functional redundancy: such heads, especially in deeper layers, contribute little to predictive performance and effectively serve as \emph{dumping grounds} for superfluous attention weights. This provides a concrete functional explanation for the structural redundancy reported in prior studies. Leveraging this insight, we introduce a simple pruning strategy that removes high-BOS sink heads. Experiments on Gemma-3, Llama-3.1, and Qwen3 demonstrate that this approach identifies redundant transformer components more reliably than weight- or activation-based criteria, while preserving performance close to dense baselines even under aggressive pruning. Moreover, we find that the behavior of sink heads remains stable across different sequence lengths. Overall, our results suggest that structural properties of attention offer a more intuitive and robust basis for model compression than magnitude-based methods. 

---
# GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO 

**Authors**: Shubhashis Roy Dipta, Khairul Mahbub, Nadia Najjar  

**Link**: [PDF](https://arxiv.org/pdf/2601.06767)  

**Abstract**: We present a Bengali mathematical reasoning model called GanitLLM (named after the Bangla word for mathematics, "Ganit"), together with a new difficulty-aware Bengali math corpus and a curriculum-based GRPO pipeline. Bengali is one of the world's most widely spoken languages, yet existing LLMs either reason in English and then translate, or simply fail on multi-step Bengali math, in part because reinforcement learning recipes are tuned for high-resource languages and collapse under reward sparsity in low-resource settings. To address this, we construct Ganit, a rigorously filtered and decontaminated Bengali math dataset with automatic difficulty tags derived from the pass@k of a strong evaluator model. Building on this dataset, we propose Curriculum-GRPO, which combines multi-stage training (SFT + GRPO) with difficulty-aware sampling and verifiable rewards for format, numerical correctness, and Bengali reasoning. On Bn-MGSM and Bn-MSVAMP, GanitLLM-4B improves over its Qwen3-4B base by +8 and +7 accuracy points, respectively, while increasing the percentage of Bengali reasoning tokens from 14% to over 88% and reducing average solution length from 943 to 193 words. 

---
# MTMCS-Bench: Evaluating Contextual Safety of Multimodal Large Language Models in Multi-Turn Dialogues 

**Authors**: Zheyuan Liu, Dongwhi Kim, Yixin Wan, Xiangchi Yuan, Zhaoxuan Tan, Fengran Mo, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06757)  

**Abstract**: Multimodal large language models (MLLMs) are increasingly deployed as assistants that interact through text and images, making it crucial to evaluate contextual safety when risk depends on both the visual scene and the evolving dialogue. Existing contextual safety benchmarks are mostly single-turn and often miss how malicious intent can emerge gradually or how the same scene can support both benign and exploitative goals. We introduce the Multi-Turn Multimodal Contextual Safety Benchmark (MTMCS-Bench), a benchmark of realistic images and multi-turn conversations that evaluates contextual safety in MLLMs under two complementary settings, escalation-based risk and context-switch risk. MTMCS-Bench offers paired safe and unsafe dialogues with structured evaluation. It contains over 30 thousand multimodal (image+text) and unimodal (text-only) samples, with metrics that separately measure contextual intent recognition, safety-awareness on unsafe cases, and helpfulness on benign ones. Across eight open-source and seven proprietary MLLMs, we observe persistent trade-offs between contextual safety and utility, with models tending to either miss gradual risks or over-refuse benign dialogues. Finally, we evaluate five current guardrails and find that they mitigate some failures but do not fully resolve multi-turn contextual risks. 

---
# EpiCaR: Knowing What You Don't Know Matters for Better Reasoning in LLMs 

**Authors**: Jewon Yeom, Jaewon Sok, Seonghyeon Park, Jeongjae Park, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.06786)  

**Abstract**: Improving the reasoning abilities of large language models (LLMs) has largely relied on iterative self-training with model-generated data. While effective at boosting accuracy, existing approaches primarily reinforce successful reasoning paths, incurring a substantial calibration cost: models become overconfident and lose the ability to represent uncertainty. This failure has been characterized as a form of model collapse in alignment, where predictive distributions degenerate toward low-variance point estimates. We address this issue by reframing reasoning training as an epistemic learning problem, in which models must learn not only how to reason, but also when their reasoning should be trusted. We propose epistemically-calibrated reasoning (EpiCaR) as a training objective that jointly optimizes reasoning performance and calibration, and instantiate it within an iterative supervised fine-tuning framework using explicit self-evaluation signals. Experiments on Llama-3 and Qwen-3 families demonstrate that our approach achieves Pareto-superiority over standard baselines in both accuracy and calibration, particularly in models with sufficient reasoning capacity (e.g., 3B+). This framework generalizes effectively to OOD mathematical reasoning (GSM8K) and code generation (MBPP). Ultimately, our approach enables a 3X reduction in inference compute, matching the K=30 performance of STaR with only K=10 samples in capable models. 

---
# Multi-Stage Evolutionary Model Merging with Meta Data Driven Curriculum Learning for Sentiment-Specialized Large Language Modeling 

**Authors**: Keito Inoshita, Xiaokang Zhou, Akira Kawai  

**Link**: [PDF](https://arxiv.org/pdf/2601.06780)  

**Abstract**: The emergence of large language models (LLMs) has significantly transformed natural language processing (NLP), enabling more generalized models to perform various tasks with minimal training. However, traditional sentiment analysis methods, which focus on individual tasks such as sentiment classification or aspect-based analysis, are not practical for real-world applications that usually require handling multiple tasks. While offering flexibility, LLMs in sentiment-specific tasks often fall short of the required accuracy. Techniques like fine-tuning and evolutionary model merging help integrate models into a unified framework, which can improve the learning performance while reducing computational costs. The use of task meta-data and curriculum learning to optimize learning processes remains underexplored, while sentiment analysis is a critical task in NLP that requires high accuracy and scalability across multiple subtasks. In this study, we propose a hybrid learning model called Multi-stage Evolutionary Model Merging with Meta data driven Curriculum Learning (MEM-MCL), to enhance the sentiment analysis in large language modeling. In particular, expert models are created through instruction tuning for specific sentiment tasks and then merged using evolutionary algorithms to form a unified model. The merging process is optimized with weak data to enhance performance across tasks. The curriculum learning is incorporated to provide a learning sequence based on task difficulty, improving knowledge extraction from LLMs. Experiment results demonstrate that the proposed MEM-MCL model outperforms conventional LLMs in a majority of sentiment analysis tasks, achieving superior results across various subtasks. 

---
# Explainable Multimodal Aspect-Based Sentiment Analysis with Dependency-guided Large Language Model 

**Authors**: Zhongzheng Wang, Yuanhe Tian, Hongzhi Wang, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.06848)  

**Abstract**: Multimodal aspect-based sentiment analysis (MABSA) aims to identify aspect-level sentiments by jointly modeling textual and visual information, which is essential for fine-grained opinion understanding in social media. Existing approaches mainly rely on discriminative classification with complex multimodal fusion, yet lacking explicit sentiment explainability. In this paper, we reformulate MABSA as a generative and explainable task, proposing a unified framework that simultaneously predicts aspect-level sentiment and generates natural language explanations. Based on multimodal large language models (MLLMs), our approach employs a prompt-based generative paradigm, jointly producing sentiment and explanation. To further enhance aspect-oriented reasoning capabilities, we propose a dependency-syntax-guided sentiment cue strategy. This strategy prunes and textualizes the aspect-centered dependency syntax tree, guiding the model to distinguish different sentiment aspects and enhancing its explainability. To enable explainability, we use MLLMs to construct new datasets with sentiment explanations to fine-tune. Experiments show that our approach not only achieves consistent gains in sentiment classification accuracy, but also produces faithful, aspect-grounded explanations. 

---
# Evaluating Accounting Reasoning Capabilities of Large Language Models 

**Authors**: Jie Zhou, Xin Chen, Jie Zhang, Hai Li, Jie Wang, Zhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.06707)  

**Abstract**: Large language models are transforming learning, cognition, and research across many fields. Effectively integrating them into professional domains, such as accounting, is a key challenge for enterprise digital transformation. To address this, we define vertical domain accounting reasoning and propose evaluation criteria derived from an analysis of the training data characteristics of representative GLM models. These criteria support systematic study of accounting reasoning and provide benchmarks for performance improvement. Using this framework, we evaluate GLM-6B, GLM-130B, GLM-4, and OpenAI GPT-4 on accounting reasoning tasks. Results show that prompt design significantly affects performance, with GPT-4 demonstrating the strongest capability. Despite these gains, current models remain insufficient for real-world enterprise accounting, indicating the need for further optimization to unlock their full practical value. 

---
# InFi-Check: Interpretable and Fine-Grained Fact-Checking of LLMs 

**Authors**: Yuzhuo Bai, Shuzheng Si, Kangyang Luo, Qingyi Wang, Wenhao Li, Gang Chen, Fanchao Qi, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.06666)  

**Abstract**: Large language models (LLMs) often hallucinate, yet most existing fact-checking methods treat factuality evaluation as a binary classification problem, offering limited interpretability and failing to capture fine-grained error types. In this paper, we introduce InFi-Check, a framework for interpretable and fine-grained fact-checking of LLM outputs. Specifically, we first propose a controlled data synthesis pipeline that generates high-quality data featuring explicit evidence, fine-grained error type labels, justifications, and corrections. Based on this, we further construct large-scale training data and a manually verified benchmark InFi-Check-FG for fine-grained fact-checking of LLM outputs. Building on these high-quality training data, we further propose InFi-Checker, which can jointly provide supporting evidence, classify fine-grained error types, and produce justifications along with corrections. Experiments show that InFi-Checker achieves state-of-the-art performance on InFi-Check-FG and strong generalization across various downstream tasks, significantly improving the utility and trustworthiness of factuality evaluation. 

---
# Pragya: An AI-Based Semantic Recommendation System for Sanskrit Subhasitas 

**Authors**: Tanisha Raorane, Prasenjit Kole  

**Link**: [PDF](https://arxiv.org/pdf/2601.06607)  

**Abstract**: Sanskrit Subhasitas encapsulate centuries of cultural and philosophical wisdom, yet remain underutilized in the digital age due to linguistic and contextual barriers. In this work, we present Pragya, a retrieval-augmented generation (RAG) framework for semantic recommendation of Subhasitas. We curate a dataset of 200 verses annotated with thematic tags such as motivation, friendship, and compassion. Using sentence embeddings (IndicBERT), the system retrieves top-k verses relevant to user queries. The retrieved results are then passed to a generative model (Mistral LLM) to produce transliterations, translations, and contextual explanations. Experimental evaluation demonstrates that semantic retrieval significantly outperforms keyword matching in precision and relevance, while user studies highlight improved accessibility through generated summaries. To our knowledge, this is the first attempt at integrating retrieval and generation for Sanskrit Subhasitas, bridging cultural heritage with modern applied AI. 

---
# How Context Shapes Truth: Geometric Transformations of Statement-level Truth Representations in LLMs 

**Authors**: Shivam Adarsh, Maria Maistro, Christina Lioma  

**Link**: [PDF](https://arxiv.org/pdf/2601.06599)  

**Abstract**: Large Language Models (LLMs) often encode whether a statement is true as a vector in their residual stream activations. These vectors, also known as truth vectors, have been studied in prior work, however how they change when context is introduced remains unexplored. We study this question by measuring (1) the directional change ($\theta$) between the truth vectors with and without context and (2) the relative magnitude of the truth vectors upon adding context. Across four LLMs and four datasets, we find that (1) truth vectors are roughly orthogonal in early layers, converge in middle layers, and may stabilize or continue increasing in later layers; (2) adding context generally increases the truth vector magnitude, i.e., the separation between true and false representations in the activation space is amplified; (3) larger models distinguish relevant from irrelevant context mainly through directional change ($\theta$), while smaller models show this distinction through magnitude differences. We also find that context conflicting with parametric knowledge produces larger geometric changes than parametrically aligned context. To the best of our knowledge, this is the first work that provides a geometric characterization of how context transforms the truth vector in the activation space of LLMs. 

---
# MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis 

**Authors**: Wenting Chen, Zhongrui Zhu, Guolin Huang, Wenxuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06636)  

**Abstract**: Despite achieving high accuracy on medical benchmarks, LLMs exhibit the Einstellung Effect in clinical diagnosis--relying on statistical shortcuts rather than patient-specific evidence, causing misdiagnosis in atypical cases. Existing benchmarks fail to detect this critical failure mode. We introduce MedEinst, a counterfactual benchmark with 5,383 paired clinical cases across 49 diseases. Each pair contains a control case and a "trap" case with altered discriminative evidence that flips the diagnosis. We measure susceptibility via Bias Trap Rate--probability of misdiagnosing traps despite correctly diagnosing controls. Extensive Evaluation of 17 LLMs shows frontier models achieve high baseline accuracy but severe bias trap rates. Thus, we propose ECR-Agent, aligning LLM reasoning with Evidence-Based Medicine standard via two components: (1) Dynamic Causal Inference (DCI) performs structured reasoning through dual-pathway perception, dynamic causal graph reasoning across three levels (association, intervention, counterfactual), and evidence audit for final diagnosis; (2) Critic-Driven Graph and Memory Evolution (CGME) iteratively refines the system by storing validated reasoning paths in an exemplar base and consolidating disease-specific knowledge into evolving illness graphs. Source code is to be released. 

---
# IDRBench: Interactive Deep Research Benchmark 

**Authors**: Yingchaojie Feng, Qiang Huang, Xiaoya Xie, Zhaorui Yang, Jun Yu, Wei Chen, Anthony K. H. Tung  

**Link**: [PDF](https://arxiv.org/pdf/2601.06676)  

**Abstract**: Deep research agents powered by Large Language Models (LLMs) can perform multi-step reasoning, web exploration, and long-form report generation. However, most existing systems operate in an autonomous manner, assuming fully specified user intent and evaluating only final outputs. In practice, research goals are often underspecified and evolve during exploration, making sustained interaction essential for robust alignment. Despite its importance, interaction remains largely invisible to existing deep research benchmarks, which neither model dynamic user feedback nor quantify its costs. We introduce IDRBench, the first benchmark for systematically evaluating interactive deep research. IDRBench combines a modular multi-agent research framework with on-demand interaction, a scalable reference-grounded user simulator, and an interaction-aware evaluation suite that jointly measures interaction benefits (quality and alignment) and costs (turns and tokens). Experiments across seven state-of-the-art LLMs show that interaction consistently improves research quality and robustness, often outweighing differences in model capacity, while revealing substantial trade-offs in interaction efficiency. 

---
# Do Language Models Reason Across Languages? 

**Authors**: Yan Meng, Wafaa Mohammed, Christof Monz  

**Link**: [PDF](https://arxiv.org/pdf/2601.06644)  

**Abstract**: The real-world information sources are inherently multilingual, which naturally raises a question about whether language models can synthesize information across languages. In this paper, we introduce a simple two-hop question answering setting, where answering a question requires making inferences over two multilingual documents. We find that language models are more sensitive to language variation in answer-span documents than in those providing bridging information, despite the equal importance of both documents for answering a question. Under a step-by-step sub-question evaluation, we further show that in up to 33% of multilingual cases, models fail to infer the bridging information in the first step yet still answer the overall question correctly. This indicates that reasoning in language models, especially in multilingual settings, does not follow a faithful step-by-step decomposition. Subsequently, we show that the absence of reasoning decomposition leads to around 18% composition failure, where both sub-questions are answered correctly but fail for the final two-hop questions. To mitigate this, we propose a simple three-stage SUBQ prompting method to guide the multi-step reasoning with sub-questions, which boosts accuracy from 10.1% to 66.5%. 

---
# N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs 

**Authors**: Mohamed Sharafath, Aravindh Annamalai, Ganesh Murugan, Aravindakumar Venugopalan  

**Link**: [PDF](https://arxiv.org/pdf/2601.06603)  

**Abstract**: Multi-hop question answering over hybrid table-text data requires retrieving and reasoning across multiple evidence pieces from large corpora, but standard Retrieval-Augmented Generation (RAG) pipelines process documents as flat ranked lists, causing retrieval noise to obscure reasoning chains. We introduce N2N-GQA. To our knowledge, it is the first zeroshot framework for open-domain hybrid table-text QA that constructs dynamic evidence graphs from noisy retrieval outputs. Our key insight is that multi-hop reasoning requires understanding relationships between evidence pieces: by modeling documents as graph nodes with semantic relationships as edges, we identify bridge documents connecting reasoning steps, a capability absent in list-based retrieval. On OTT-QA, graph-based evidence curation provides a 19.9-point EM improvement over strong baselines, demonstrating that organizing retrieval results as structured graphs is critical for multihop reasoning. N2N-GQA achieves 48.80 EM, matching finetuned retrieval models (CORE: 49.0 EM) and approaching heavily optimized systems (COS: 56.9 EM) without any task specific training. This establishes graph-structured evidence organization as essential for scalable, zero-shot multi-hop QA systems and demonstrates that simple, interpretable graph construction can rival sophisticated fine-tuned approaches. 

---
# Probing Multimodal Large Language Models on Cognitive Biases in Chinese Short-Video Misinformation 

**Authors**: Jen-tse Huang, Chang Chen, Shiyang Lai, Wenxuan Wang, Michelle R. Kaufman, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2601.06600)  

**Abstract**: Short-video platforms have become major channels for misinformation, where deceptive claims frequently leverage visual experiments and social cues. While Multimodal Large Language Models (MLLMs) have demonstrated impressive reasoning capabilities, their robustness against misinformation entangled with cognitive biases remains under-explored. In this paper, we introduce a comprehensive evaluation framework using a high-quality, manually annotated dataset of 200 short videos spanning four health domains. This dataset provides fine-grained annotations for three deceptive patterns, experimental errors, logical fallacies, and fabricated claims, each verified by evidence such as national standards and academic literature. We evaluate eight frontier MLLMs across five modality settings. Experimental results demonstrate that Gemini-2.5-Pro achieves the highest performance in the multimodal setting with a belief score of 71.5/100, while o3 performs the worst at 35.2. Furthermore, we investigate social cues that induce false beliefs in videos and find that models are susceptible to biases like authoritative channel IDs. 

---
# CSR-RAG: An Efficient Retrieval System for Text-to-SQL on the Enterprise Scale 

**Authors**: Rajpreet Singh, Novak Boškov, Lawrence Drabeck, Aditya Gudal, Manzoor A. Khan  

**Link**: [PDF](https://arxiv.org/pdf/2601.06564)  

**Abstract**: Natural language to SQL translation (Text-to-SQL) is one of the long-standing problems that has recently benefited from advances in Large Language Models (LLMs). While most academic Text-to-SQL benchmarks request schema description as a part of natural language input, enterprise-scale applications often require table retrieval before SQL query generation. To address this need, we propose a novel hybrid Retrieval Augmented Generation (RAG) system consisting of contextual, structural, and relational retrieval (CSR-RAG) to achieve computationally efficient yet sufficiently accurate retrieval for enterprise-scale databases. Through extensive enterprise benchmarks, we demonstrate that CSR-RAG achieves up to 40% precision and over 80% recall while incurring a negligible average query generation latency of only 30ms on commodity data center hardware, which makes it appropriate for modern LLM-based enterprise-scale systems. 

---
# SimLLM: Fine-Tuning Code LLMs for SimPy-Based Queueing System Simulation 

**Authors**: Jun-Qi Chen, Kun Zhang, Rui Zheng, Ying Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2601.06543)  

**Abstract**: The Python package SimPy is widely used for modeling queueing systems due to its flexibility, simplicity, and smooth integration with modern data analysis and optimization frameworks. Recent advances in large language models (LLMs) have shown strong ability in generating clear and executable code, making them powerful and suitable tools for writing SimPy queueing simulation code. However, directly employing closed-source models like GPT-4o to generate such code may lead to high computational costs and raise data privacy concerns. To address this, we fine-tune two open-source LLMs, Qwen-Coder-7B and DeepSeek-Coder-6.7B, on curated SimPy queueing data, which enhances their code-generating performance in executability, output-format compliance, and instruction-code consistency. Particularly, we proposed a multi-stage fine-tuning framework comprising two stages of supervised fine-tuning (SFT) and one stage of direct preference optimization (DPO), progressively enhancing the model's ability in SimPy-based queueing simulation code generation. Extensive evaluations demonstrate that both fine-tuned models achieve substantial improvements in executability, output-format compliance, and instruct consistency. These results confirm that domain-specific fine-tuning can effectively transform compact open-source code models into reliable SimPy simulation generators which provide a practical alternative to closed-source LLMs for education, research, and operational decision support. 

---
# Time Travel Engine: A Shared Latent Chronological Manifold Enables Historical Navigation in Large Language Models 

**Authors**: Jingmin An, Wei Liu, Qian Wang, Fang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06437)  

**Abstract**: Time functions as a fundamental dimension of human cognition, yet the mechanisms by which Large Language Models (LLMs) encode chronological progression remain opaque. We demonstrate that temporal information in their latent space is organized not as discrete clusters but as a continuous, traversable geometry. We introduce the Time Travel Engine (TTE), an interpretability-driven framework that projects diachronic linguistic patterns onto a shared chronological manifold. Unlike surface-level prompting, TTE directly modulates latent representations to induce coherent stylistic, lexical, and conceptual shifts aligned with target eras. By parameterizing diachronic evolution as a continuous manifold within the residual stream, TTE enables fluid navigation through period-specific "zeitgeists" while restricting access to future knowledge. Furthermore, experiments across diverse architectures reveal topological isomorphism between the temporal subspaces of Chinese and English-indicating that distinct languages share a universal geometric logic of historical evolution. These findings bridge historical linguistics with mechanistic interpretability, offering a novel paradigm for controlling temporal reasoning in neural networks. 

---
# IndRegBias: A Dataset for Studying Indian Regional Biases in English and Code-Mixed Social Media Comments 

**Authors**: Debasmita Panda, Akash Anil, Neelesh Kumar Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2601.06477)  

**Abstract**: Warning: This paper consists of examples representing regional biases in Indian regions that might be offensive towards a particular region. While social biases corresponding to gender, race, socio-economic conditions, etc., have been extensively studied in the major applications of Natural Language Processing (NLP), biases corresponding to regions have garnered less attention. This is mainly because of (i) difficulty in the extraction of regional bias datasets, (ii) disagreements in annotation due to inherent human biases, and (iii) regional biases being studied in combination with other types of social biases and often being under-represented. This paper focuses on creating a dataset IndRegBias, consisting of regional biases in an Indian context reflected in users' comments on popular social media platforms, namely Reddit and YouTube. We carefully selected 25,000 comments appearing on various threads in Reddit and videos on YouTube discussing trending topics on regional issues in India. Furthermore, we propose a multilevel annotation strategy to annotate the comments describing the severity of regional biased statements. To detect the presence of regional bias and its severity in IndRegBias, we evaluate open-source Large Language Models (LLMs) and Indic Language Models (ILMs) using zero-shot, few-shot, and fine-tuning strategies. We observe that zero-shot and few-shot approaches show lower accuracy in detecting regional biases and severity in the majority of the LLMs and ILMs. However, the fine-tuning approach significantly enhances the performance of the LLM in detecting Indian regional bias along with its severity. 

---
# NC-Bench: An LLM Benchmark for Evaluating Conversational Competence 

**Authors**: Robert J. Moore, Sungeun An, Farhan Ahmed, Jay Pankaj Gala  

**Link**: [PDF](https://arxiv.org/pdf/2601.06426)  

**Abstract**: The Natural Conversation Benchmark (NC-Bench) introduce a new approach to evaluating the general conversational competence of large language models (LLMs). Unlike prior benchmarks that focus on the content of model behavior, NC-Bench focuses on the form and structure of natural conversation. Grounded in the IBM Natural Conversation Framework (NCF), NC-Bench comprises three distinct sets. The Basic Conversation Competence set evaluates fundamental sequence management practices, such as answering inquiries, repairing responses, and closing conversational pairs. The RAG set applies the same sequence management patterns as the first set but incorporates retrieval-augmented generation (RAG). The Complex Request set extends the evaluation to complex requests involving more intricate sequence management patterns. Each benchmark tests a model's ability to produce contextually appropriate conversational actions in response to characteristic interaction patterns. Initial evaluations across 6 open-source models and 14 interaction patterns show that models perform well on basic answering tasks, struggle more with repair tasks (especially repeat), have mixed performance on closing sequences, and find complex multi-turn requests most challenging, with Qwen models excelling on the Basic set and Granite models on the RAG set and the Complex Request set. By operationalizing fundamental principles of human conversation, NC-Bench provides a lightweight, extensible, and theory-grounded framework for assessing and improving the conversational abilities of LLMs beyond topical or task-specific benchmarks. 

---
# LitVISTA: A Benchmark for Narrative Orchestration in Literary Text 

**Authors**: Mingzhe Lu, Yiwen Wang, Yanbing Liu, Qi You, Chong Liu, Ruize Qin, Haoyu Dong, Wenyu Zhang, Jiarui Zhang, Yue Hu, Yunpeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.06445)  

**Abstract**: Computational narrative analysis aims to capture rhythm, tension, and emotional dynamics in literary texts. Existing large language models can generate long stories but overly focus on causal coherence, neglecting the complex story arcs and orchestration inherent in human narratives. This creates a structural misalignment between model- and human-generated narratives. We propose VISTA Space, a high-dimensional representational framework for narrative orchestration that unifies human and model narrative perspectives. We further introduce LitVISTA, a structurally annotated benchmark grounded in literary texts, enabling systematic evaluation of models' narrative orchestration capabilities. We conduct oracle evaluations on a diverse selection of frontier LLMs, including GPT, Claude, Grok, and Gemini. Results reveal systematic deficiencies: existing models fail to construct a unified global narrative view, struggling to jointly capture narrative function and structure. Furthermore, even advanced thinking modes yield only limited gains for such literary narrative understanding. 

---
# Can a Unimodal Language Agent Provide Preferences to Tune a Multimodal Vision-Language Model? 

**Authors**: Sazia Tabasum Mim, Jack Morris, Manish Dhakal, Yanming Xiu, Maria Gorlatova, Yi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2601.06424)  

**Abstract**: To explore a more scalable path for adding multimodal capabilities to existing LLMs, this paper addresses a fundamental question: Can a unimodal LLM, relying solely on text, reason about its own informational needs and provide effective feedback to optimize a multimodal model? To answer this, we propose a method that enables a language agent to give feedback to a vision-language model (VLM) to adapt text generation to the agent's preferences. Our results from different experiments affirm this hypothesis, showing that LLM preference feedback significantly enhances VLM descriptions. Using our proposed method, we find that the VLM can generate multimodal scene descriptions to help the LLM better understand multimodal context, leading to improvements of maximum 13% in absolute accuracy compared to the baseline multimodal approach. Furthermore, a human study validated our AI-driven feedback, showing a 64.6% preference alignment rate between the LLM's choices and human judgments. Extensive experiments provide insights on how and why the method works and its limitations. 

---
# Exposía: Academic Writing Assessment of Exposés and Peer Feedback 

**Authors**: Dennis Zyska, Alla Rozovskaya, Ilia Kuznetsov, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2601.06536)  

**Abstract**: We present Exposía, the first public dataset that connects writing and feedback assessment in higher education, enabling research on educationally grounded approaches to academic writing evaluation. Exposía includes student research project proposals and peer and instructor feedback consisting of comments and free-text reviews. The dataset was collected in the "Introduction to Scientific Work" course of the Computer Science undergraduate program that focuses on teaching academic writing skills and providing peer feedback on academic writing. Exposía reflects the multi-stage nature of the academic writing process that includes drafting, providing and receiving feedback, and revising the writing based on the feedback received. Both the project proposals and peer feedback are accompanied by human assessment scores based on a fine-grained, pedagogically-grounded schema for writing and feedback assessment that we develop.
We use Exposía to benchmark state-of-the-art open-source large language models (LLMs) for two tasks: automated scoring of (1) the proposals and (2) the student reviews. The strongest LLMs attain high agreement on scoring aspects that require little domain knowledge but degrade on dimensions evaluating content, in line with human agreement values. We find that LLMs align better with the human instructors giving high scores. Finally, we establish that a prompting strategy that scores multiple aspects of the writing together is the most effective, an important finding for classroom deployment. 

---
# AfriqueLLM: How Data Mixing and Model Architecture Impact Continued Pre-training for African Languages 

**Authors**: Hao Yu, Tianyi Xu, Michael A. Hedderich, Wassim Hamidouche, Syed Waqas Zamir, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2601.06395)  

**Abstract**: Large language models (LLMs) are increasingly multilingual, yet open models continue to underperform relative to proprietary systems, with the gap most pronounced for African languages. Continued pre-training (CPT) offers a practical route to language adaptation, but improvements on demanding capabilities such as mathematical reasoning often remain limited. This limitation is driven in part by the uneven domain coverage and missing task-relevant knowledge that characterize many low-resource language corpora. We present \texttt{AfriqueLLM}, a suite of open LLMs adapted to 20 African languages through CPT on 26B tokens. We perform a comprehensive empirical study across five base models spanning sizes and architectures, including Llama 3.1, Gemma 3, and Qwen 3, and systematically analyze how CPT data composition shapes downstream performance. In particular, we vary mixtures that include math, code, and synthetic translated data, and evaluate the resulting models on a range of multilingual benchmarks. Our results identify data composition as the primary driver of CPT gains. Adding math, code, and synthetic translated data yields consistent improvements, including on reasoning-oriented evaluations. Within a fixed architecture, larger models typically improve performance, but architectural choices dominate scale when comparing across model families. Moreover, strong multilingual performance in the base model does not reliably predict post-CPT outcomes; robust architectures coupled with task-aligned data provide a more dependable recipe. Finally, our best models improve long-context performance, including document-level translation. Models have been released on [Huggingface](this https URL). 

---
# Steer Model beyond Assistant: Controlling System Prompt Strength via Contrastive Decoding 

**Authors**: Yijiang River Dong, Tiancheng Hu, Zheng Hui, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2601.06403)  

**Abstract**: Large language models excel at complex instructions yet struggle to deviate from their helpful assistant persona, as post-training instills strong priors that resist conflicting instructions. We introduce system prompt strength, a training-free method that treats prompt adherence as a continuous control. By contrasting logits from target and default system prompts, we isolate and amplify the behavioral signal unique to the target persona by a scalar factor alpha. Across five diverse benchmarks spanning constraint satisfaction, behavioral control, pluralistic alignment, capability modulation, and stylistic control, our method yields substantial improvements: up to +8.5 strict accuracy on IFEval, +45pp refusal rate on OffTopicEval, and +13% steerability on Prompt-Steering. Our approach enables practitioners to modulate system prompt strength, providing dynamic control over model behavior without retraining. 

---
# Annotating Dimensions of Social Perception in Text: The First Sentence-Level Dataset of Warmth and Competence 

**Authors**: Mutaz Ayesh, Saif M. Mohammad, Nedjma Ousidhoum  

**Link**: [PDF](https://arxiv.org/pdf/2601.06316)  

**Abstract**: Warmth (W) (often further broken down into Trust (T) and Sociability (S)) and Competence (C) are central dimensions along which people evaluate individuals and social groups (Fiske, 2018). While these constructs are well established in social psychology, they are only starting to get attention in NLP research through word-level lexicons, which do not completely capture their contextual expression in larger text units and discourse. In this work, we introduce Warmth and Competence Sentences (W&C-Sent), the first sentence-level dataset annotated for warmth and competence. The dataset includes over 1,600 English sentence--target pairs annotated along three dimensions: trust and sociability (components of warmth), and competence. The sentences in W&C-Sent are from social media and often express attitudes and opinions about specific individuals or social groups (the targets of our annotations). We describe the data collection, annotation, and quality-control procedures in detail, and evaluate a range of large language models (LLMs) on their ability to identify trust, sociability, and competence in text. W&C-Sent provides a new resource for analyzing warmth and competence in language and supports future research at the intersection of NLP and computational social science. 

---
# AzeroS: Extending LLM to Speech with Self-Generated Instruction-Free Tuning 

**Authors**: Yiwen Shao, Wei Liu, Jiahong Li, Tianzi Wang, Kun Wei, Meng Yu, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.06086)  

**Abstract**: Extending large language models (LLMs) to the speech domain has recently gained significant attention. A typical approach connects a pretrained LLM with an audio encoder through a projection module and trains the resulting model on large-scale, task-specific instruction-tuning datasets. However, curating such instruction-tuning data for specific requirements is time-consuming, and models trained in this manner often generalize poorly to unseen tasks. In this work, we first formulate that the strongest generalization of a speech-LLM is achieved when it is trained with Self-Generated Instruction-Free Tuning (SIFT), in which supervision signals are generated by a frozen LLM using textual representations of speech as input. Our proposed SIFT paradigm eliminates the need for collecting task-specific question-answer pairs and yields the theoretically best generalization to unseen tasks. Building upon this paradigm, we introduce AZeroS (Auden Zero-instruction-tuned Speech-LLM), which is trained on speech-text pairs derived from publicly available corpora, including approximately 25,000 hours of speech with ASR transcripts and 3,000 hours of speech with paralinguistic labels. Built upon Qwen2.5-7B-Instruct, the model updates only two lightweight projection modules (23.8 million parameters each), while keeping both the LLM and audio encoders frozen. Despite the minimal training cost and modest data scale, AZeroS achieves state-of-the-art performance on both semantic and paralinguistic benchmarks, including VoiceBench, AIR-Bench Foundation (Speech), and AIR-Bench Chat (Speech). 

---
# A Multi-Stage Workflow for the Review of Marketing Content with Reasoning Large Language Models 

**Authors**: Alberto Purpura, Emily Chen, Swapnil Shinde  

**Link**: [PDF](https://arxiv.org/pdf/2601.06054)  

**Abstract**: Reasoning Large Language Models (LLMs) have shown promising results when tasked with solving complex problems. In this paper, we propose and evaluate a multi-stage workflow that leverages the capabilities of fine-tuned reasoning LLMs to assist in the review process of marketing content, making sure they comply with a given list of requirements. The contributions of this paper are the following: (i) we present a novel approach -- that does not rely on any external knowledge representation -- for the automatic identification of compliance issues in textual content; (ii) compare the effectiveness of different fine-tuning strategies like Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) in training models to solve this problem; (iii) we evaluate the effectiveness of training small LLMs to generate reasoning tokens before providing their final response; (iv) we evaluate how the choice and combinations of different reward functions affects the performance of a model trained with GRPO. 

---
# How well can off-the-shelf LLMs elucidate molecular structures from mass spectra using chain-of-thought reasoning? 

**Authors**: Yufeng Wang, Lu Wei, Lin Liu, Hao Xu, Haibin Ling  

**Link**: [PDF](https://arxiv.org/pdf/2601.06289)  

**Abstract**: Mass spectrometry (MS) is a powerful analytical technique for identifying small molecules, yet determining complete molecular structures directly from tandem mass spectra (MS/MS) remains a long-standing challenge due to complex fragmentation patterns and the vast diversity of chemical space. Recent progress in large language models (LLMs) has shown promise for reasoning-intensive scientific tasks, but their capability for chemical interpretation is still unclear. In this work, we introduce a Chain-of-Thought (CoT) prompting framework and benchmark that evaluate how LLMs reason about mass spectral data to predict molecular structures. We formalize expert chemists' reasoning steps-such as double bond equivalent (DBE) analysis, neutral loss identification, and fragment assembly-into structured prompts and assess multiple state-of-the-art LLMs (Claude-3.5-Sonnet, GPT-4o-mini, and Llama-3 series) in a zero-shot setting using the MassSpecGym dataset. Our evaluation across metrics of SMILES validity, formula consistency, and structural similarity reveals that while LLMs can produce syntactically valid and partially plausible structures, they fail to achieve chemical accuracy or link reasoning to correct molecular predictions. These findings highlight both the interpretive potential and the current limitations of LLM-based reasoning for molecular elucidation, providing a foundation for future work that combines domain knowledge and reinforcement learning to achieve chemically grounded AI reasoning. 

---
# Are LLM Decisions Faithful to Verbal Confidence? 

**Authors**: Jiawei Wang, Yanfei Zhou, Siddartha Devic, Deqing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07767)  

**Abstract**: Large Language Models (LLMs) can produce surprisingly sophisticated estimates of their own uncertainty. However, it remains unclear to what extent this expressed confidence is tied to the reasoning, knowledge, or decision making of the model. To test this, we introduce $\textbf{RiskEval}$: a framework designed to evaluate whether models adjust their abstention policies in response to varying error penalties. Our evaluation of several frontier models reveals a critical dissociation: models are neither cost-aware when articulating their verbal confidence, nor strategically responsive when deciding whether to engage or abstain under high-penalty conditions. Even when extreme penalties render frequent abstention the mathematically optimal strategy, models almost never abstain, resulting in utility collapse. This indicates that calibrated verbal confidence scores may not be sufficient to create trustworthy and interpretable AI systems, as current models lack the strategic agency to convert uncertainty signals into optimal and risk-sensitive decisions. 

---
# $\texttt{AMEND++}$: Benchmarking Eligibility Criteria Amendments in Clinical Trials 

**Authors**: Trisha Das, Mandis Beigi, Jacob Aptekar, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.06300)  

**Abstract**: Clinical trial amendments frequently introduce delays, increased costs, and administrative burden, with eligibility criteria being the most commonly amended component. We introduce \textit{eligibility criteria amendment prediction}, a novel NLP task that aims to forecast whether the eligibility criteria of an initial trial protocol will undergo future amendments. To support this task, we release $\texttt{AMEND++}$, a benchmark suite comprising two datasets: $\texttt{AMEND}$, which captures eligibility-criteria version histories and amendment labels from public clinical trials, and $\verb|AMEND_LLM|$, a refined subset curated using an LLM-based denoising pipeline to isolate substantive changes. We further propose $\textit{Change-Aware Masked Language Modeling}$ (CAMLM), a revision-aware pretraining strategy that leverages historical edits to learn amendment-sensitive representations. Experiments across diverse baselines show that CAMLM consistently improves amendment prediction, enabling more robust and cost-effective clinical trial design. 

---
# Reasoning Models Will Blatantly Lie About Their Reasoning 

**Authors**: William Walden  

**Link**: [PDF](https://arxiv.org/pdf/2601.07663)  

**Abstract**: It has been shown that Large Reasoning Models (LRMs) may not *say what they think*: they do not always volunteer information about how certain parts of the input influence their reasoning. But it is one thing for a model to *omit* such information and another, worse thing to *lie* about it. Here, we extend the work of Chen et al. (2025) to show that LRMs will do just this: they will flatly deny relying on hints provided in the prompt in answering multiple choice questions -- even when directly asked to reflect on unusual (i.e. hinted) prompt content, even when allowed to use hints, and even though experiments *show* them to be using the hints. Our results thus have discouraging implications for CoT monitoring and interpretability. 

---
# GRPO with State Mutations: Improving LLM-Based Hardware Test Plan Generation 

**Authors**: Dimple Vijay Kochar, Nathaniel Pinckney, Guan-Ting Liu, Chia-Tung Ho, Chenhui Deng, Haoxing Ren, Brucek Khailany  

**Link**: [PDF](https://arxiv.org/pdf/2601.07593)  

**Abstract**: RTL design often relies heavily on ad-hoc testbench creation early in the design cycle. While large language models (LLMs) show promise for RTL code generation, their ability to reason about hardware specifications and generate targeted test plans remains largely unexplored. We present the first systematic study of LLM reasoning capabilities for RTL verification stimuli generation, establishing a two-stage framework that decomposes test plan generation from testbench execution. Our benchmark reveals that state-of-the-art models, including DeepSeek-R1 and Claude-4.0-Sonnet, achieve only 15.7-21.7% success rates on generating stimuli that pass golden RTL designs. To improve LLM generated stimuli, we develop a comprehensive training methodology combining supervised fine-tuning with a novel reinforcement learning approach, GRPO with State Mutation (GRPO-SMu), which enhances exploration by varying input mutations. Our approach leverages a tree-based branching mutation strategy to construct training data comprising equivalent and mutated trees, moving beyond linear mutation approaches to provide rich learning signals. Training on this curated dataset, our 7B parameter model achieves a 33.3% golden test pass rate and a 13.9% mutation detection rate, representing a 17.6% absolute improvement over baseline and outperforming much larger general-purpose models. These results demonstrate that specialized training methodologies can significantly enhance LLM reasoning capabilities for hardware verification tasks, establishing a foundation for automated sub-unit testing in semiconductor design workflows. 

---
# SCALPEL: Selective Capability Ablation via Low-rank Parameter Editing for Large Language Model Interpretability Analysis 

**Authors**: Zihao Fu, Xufeng Duan, Zhenguang G. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2601.07411)  

**Abstract**: Large language models excel across diverse domains, yet their deployment in healthcare, legal systems, and autonomous decision-making remains limited by incomplete understanding of their internal mechanisms. As these models integrate into high-stakes systems, understanding how they encode capabilities has become fundamental to interpretability research. Traditional approaches identify important modules through gradient attribution or activation analysis, assuming specific capabilities map to specific components. However, this oversimplifies neural computation: modules may contribute to multiple capabilities simultaneously, while single capabilities may distribute across multiple modules. These coarse-grained analyses fail to capture fine-grained, distributed capability encoding. We present SCALPEL (Selective Capability Ablation via Low-rank Parameter Editing for Large language models), a framework representing capabilities as low-rank parameter subspaces rather than discrete modules. Our key insight is that capabilities can be characterized by low-rank modifications distributed across layers and modules, enabling precise capability removal without affecting others. By training LoRA adapters to reduce distinguishing correct from incorrect answers while preserving general language modeling quality, SCALPEL identifies low-rank representations responsible for particular capabilities while remaining disentangled from others. Experiments across diverse capability and linguistic tasks from BLiMP demonstrate that SCALPEL successfully removes target capabilities while preserving general capabilities, providing fine-grained insights into capability distribution across parameter space. Results reveal that capabilities exhibit low-rank structure and can be selectively ablated through targeted parameter-space interventions, offering nuanced understanding of capability encoding in LLMs. 

---
# Learning to Trust the Crowd: A Multi-Model Consensus Reasoning Engine for Large Language Models 

**Authors**: Pranav Kallem  

**Link**: [PDF](https://arxiv.org/pdf/2601.07245)  

**Abstract**: Large language models (LLMs) achieve strong aver- age performance yet remain unreliable at the instance level, with frequent hallucinations, brittle failures, and poorly calibrated confidence. We study reliability through the lens of multi-model consensus: given responses from several heterogeneous LLMs, can we learn which answer is most likely correct for a given query? We introduce a Multi-Model Consensus Reasoning Engine that treats the set of LLM outputs as input to a supervised meta-learner. The system maps natural language responses into structured features using semantic embeddings, pairwise similarity and clustering statistics, lexical and structural cues, reasoning-quality scores, confidence estimates, and model-specific priors, and then applies gradient-boosted trees, listwise ranking, and graph neural networks over similarity graphs of answers. Using three open-weight LLMs evaluated on compact, resource- constrained subsets of GSM8K, ARC-Challenge, HellaSwag, and TruthfulQA, our best graph-attention-based consensus model improves macro-average accuracy by 4.6 percentage points over the strongest single LLM and by 8.1 points over majority vote, while also yielding lower Brier scores and fewer TruthfulQA hal- lucinations. Ablation and feature-importance analyses show that semantic agreement and clustering features are most influential, with reasoning-quality and model-prior features providing com- plementary gains, suggesting supervised multi-model consensus is a practical route toward more reliable LLM behavior, even in a modest single-machine setup. 

---
# On Narrative: The Rhetorical Mechanisms of Online Polarisation 

**Authors**: Jan Elfes, Marco Bastos, Luca Maria Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2601.07398)  

**Abstract**: Polarisation research has demonstrated how people cluster in homogeneous groups with opposing opinions. However, this effect emerges not only through interaction between people, limiting communication between groups, but also between narratives, shaping opinions and partisan identities. Yet, how polarised groups collectively construct and negotiate opposing interpretations of reality, and whether narratives move between groups despite limited interactions, remains unexplored. To address this gap, we formalise the concept of narrative polarisation and demonstrate its measurement in 212 YouTube videos and 90,029 comments on the Israeli-Palestinian conflict. Based on structural narrative theory and implemented through a large language model, we extract the narrative roles assigned to central actors in two partisan information environments. We find that while videos produce highly polarised narratives, comments significantly reduce narrative polarisation, harmonising discourse on the surface level. However, on a deeper narrative level, recurring narrative motifs reveal additional differences between partisan groups. 

---
# An Ubuntu-Guided Large Language Model Framework for Cognitive Behavioral Mental Health Dialogue 

**Authors**: Sontaga G. Forane, Absalom E. Ezugwu, Kevin Igwe, Karen van den Berg  

**Link**: [PDF](https://arxiv.org/pdf/2601.06875)  

**Abstract**: South Africa's escalating mental health crisis, compounded by limited access to culturally responsive care, calls for innovative and contextually grounded interventions. While large language models show considerable promise for mental health support, their predominantly Western-centric training data limit cultural and linguistic applicability in African contexts. This study introduces a proof-of-concept framework that integrates cognitive behavioral therapy with the African philosophy of Ubuntu to create a culturally sensitive, emotionally intelligent, AI-driven mental health dialogue system. Guided by a design science research methodology, the framework applies both deep theoretical and therapeutic adaptations as well as surface-level linguistic and communicative cultural adaptations. Key CBT techniques, including behavioral activation and cognitive restructuring, were reinterpreted through Ubuntu principles that emphasize communal well-being, spiritual grounding, and interconnectedness. A culturally adapted dataset was developed through iterative processes of language simplification, spiritual contextualization, and Ubuntu-based reframing. The fine-tuned model was evaluated through expert-informed case studies, employing UniEval for conversational quality assessment alongside additional measures of CBT reliability and cultural linguistic alignment. Results demonstrate that the model effectively engages in empathetic, context-aware dialogue aligned with both therapeutic and cultural objectives. Although real-time end-user testing has not yet been conducted, the model underwent rigorous review and supervision by domain specialist clinical psychologists. The findings highlight the potential of culturally embedded emotional intelligence to enhance the contextual relevance, inclusivity, and effectiveness of AI-driven mental health interventions across African settings. 

---
# TagSpeech: End-to-End Multi-Speaker ASR and Diarization with Fine-Grained Temporal Grounding 

**Authors**: Mingyue Huo, Yiwen Shao, Yuheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06896)  

**Abstract**: We present TagSpeech, a unified LLM-based framework that utilizes Temporal Anchor Grounding for joint multi-speaker ASR and diarization. The framework is built on two key designs: (1) decoupled semantic and speaker streams fine-tuned via Serialized Output Training (SOT) to learn turn-taking dynamics; and (2) an interleaved time anchor mechanism that not only supports fine-grained timestamp prediction but also acts as a synchronization signal between semantic understanding and speaker tracking. Compared to previous works that primarily focus on speaker-attributed ASR or implicit diarization, TagSpeech addresses the challenge of fine-grained speaker-content alignment and explicitly models "who spoke what and when" in an end-to-end manner. Experiments on AMI and AliMeeting benchmarks demonstrate that our method achieves consistent improvements in Diarization Error Rate (DER) over strong end-to-end baselines, including Qwen-Omni and Gemini, particularly in handling complex speech overlaps. Moreover, TagSpeech employs a parameter-efficient training paradigm in which the LLM backbone is frozen and only lightweight projectors are trained, resulting in strong performance with low computational cost. 

---
# Benchmarking Egocentric Clinical Intent Understanding Capability for Medical Multimodal Large Language Models 

**Authors**: Shaonan Liu, Guo Yu, Xiaoling Luo, Shiyi Zheng, Wenting Chen, Jie Liu, Linlin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.06750)  

**Abstract**: Medical Multimodal Large Language Models (Med-MLLMs) require egocentric clinical intent understanding for real-world deployment, yet existing benchmarks fail to evaluate this critical capability. To address these challenges, we introduce MedGaze-Bench, the first benchmark leveraging clinician gaze as a Cognitive Cursor to assess intent understanding across surgery, emergency simulation, and diagnostic interpretation. Our benchmark addresses three fundamental challenges: visual homogeneity of anatomical structures, strict temporal-causal dependencies in clinical workflows, and implicit adherence to safety protocols. We propose a Three-Dimensional Clinical Intent Framework evaluating: (1) Spatial Intent: discriminating precise targets amid visual noise, (2) Temporal Intent: inferring causal rationale through retrospective and prospective reasoning, and (3) Standard Intent: verifying protocol compliance through safety checks. Beyond accuracy metrics, we introduce Trap QA mechanisms to stress-test clinical reliability by penalizing hallucinations and cognitive sycophancy. Experiments reveal current MLLMs struggle with egocentric intent due to over-reliance on global features, leading to fabricated observations and uncritical acceptance of invalid instructions. 

---
# LRAS: Advanced Legal Reasoning with Agentic Search 

**Authors**: Yujin Zhou, Chuxue Cao, Jinluan Yang, Lijun Wu, Conghui He, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.07296)  

**Abstract**: While Large Reasoning Models (LRMs) have demonstrated exceptional logical capabilities in mathematical domains, their application to the legal field remains hindered by the strict requirements for procedural rigor and adherence to legal logic. Existing legal LLMs, which rely on "closed-loop reasoning" derived solely from internal parametric knowledge, frequently suffer from lack of self-awareness regarding their knowledge boundaries, leading to confident yet incorrect conclusions. To address this challenge, we present Legal Reasoning with Agentic Search (LRAS), the first framework designed to transition legal LLMs from static and parametric "closed-loop thinking" to dynamic and interactive "Active Inquiry". By integrating Introspective Imitation Learning and Difficulty-aware Reinforcement Learning, LRAS enables LRMs to identify knowledge boundaries and handle legal reasoning complexity. Empirical results demonstrate that LRAS outperforms state-of-the-art baselines by 8.2-32\%, with the most substantial gains observed in tasks requiring deep reasoning with reliable knowledge. We will release our data and models for further exploration soon. 

---
# Classroom AI: Large Language Models as Grade-Specific Teachers 

**Authors**: Jio Oh, Steven Euijong Whang, James Evans, Jindong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06225)  

**Abstract**: Large Language Models (LLMs) offer a promising solution to complement traditional teaching and address global teacher shortages that affect hundreds of millions of children, but they fail to provide grade-appropriate responses for students at different educational levels. We introduce a framework for finetuning LLMs to generate age-appropriate educational content across six grade levels, from lower elementary to adult education. Our framework successfully adapts explanations to match students' comprehension capacities without sacrificing factual correctness. This approach integrates seven established readability metrics through a clustering method and builds a comprehensive dataset for grade-specific content generation. Evaluations across multiple datasets with 208 human participants demonstrate substantial improvements in grade-level alignment, achieving a 35.64 percentage point increase compared to prompt-based methods while maintaining response accuracy. AI-assisted learning tailored to different grade levels has the potential to advance educational engagement and equity. 

---
# Manifold-based Sampling for In-Context Hallucination Detection in Large Language Models 

**Authors**: Bodla Krishna Vamshi, Rohan Bhatnagar, Haizhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06196)  

**Abstract**: Large language models (LLMs) frequently generate factually incorrect or unsupported content, commonly referred to as hallucinations. Prior work has explored decoding strategies, retrieval augmentation, and supervised fine-tuning for hallucination detection, while recent studies show that in-context learning (ICL) can substantially influence factual reliability. However, existing ICL demonstration selection methods often rely on surface-level similarity heuristics and exhibit limited robustness across tasks and models.
We propose MB-ICL, a manifold-based demonstration sampling framework for selecting in-context demonstrations that leverages latent representations extracted from frozen LLMs. By jointly modeling local manifold structure and class-aware prototype geometry, MB-ICL selects demonstrations based on their proximity to learned prototypes rather than lexical or embedding similarity alone.
Across factual verification (FEVER) and hallucination detection (HaluEval) benchmarks, MB-ICL outperforms standard ICL selection baselines in the majority of evaluated settings, with particularly strong gains on dialogue and summarization tasks. The method remains robust under temperature perturbations and model variation, indicating improved stability compared to heuristic retrieval strategies. While lexical retrieval can remain competitive in certain question-answering regimes, our results demonstrate that manifold-based prototype selection provides a reliable and training light approach for hallucination detection without modifying LLM parameters, offering a principled direction for improved ICL demonstration selection. 

---
# An evaluation of LLMs for political bias in Western media: Israel-Hamas and Ukraine-Russia wars 

**Authors**: Rohitash Chandra, Haoyan Chen, Yaqing Zhang, Jiacheng Chen, Yuting Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.06132)  

**Abstract**: Political bias in media plays a critical role in shaping public opinion, voter behaviour, and broader democratic discourse. Subjective opinions and political bias can be found in media sources, such as newspapers, depending on their funding mechanisms and alliances with political parties. Automating the detection of political biases in media content can limit biases in elections. The impact of large language models (LLMs) in politics and media studies is becoming prominent. In this study, we utilise LLMs to compare the left-wing, right-wing, and neutral political opinions expressed in the Guardian and BBC. We review newspaper reporting that includes significant events such as the Russia-Ukraine war and the Hamas-Israel conflict. We analyse the proportion for each opinion to find the bias under different LLMs, including BERT, Gemini, and DeepSeek. Our results show that after the outbreak of the wars, the political bias of Western media shifts towards the left-wing and each LLM gives a different result. DeepSeek consistently showed a stable Left-leaning tendency, while BERT and Gemini remained closer to the Centre. The BBC and The Guardian showed distinct reporting behaviours across the two conflicts. In the Russia-Ukraine war, both outlets maintained relatively stable positions; however, in the Israel-Hamas conflict, we identified larger political bias shifts, particularly in Guardian coverage, suggesting a more event-driven pattern of reporting bias. These variations suggest that LLMs are shaped not only by their training data and architecture, but also by underlying worldviews with associated political biases. 

---
# LLM Flow Processes for Text-Conditioned Regression 

**Authors**: Felix Biggs, Samuel Willis  

**Link**: [PDF](https://arxiv.org/pdf/2601.06147)  

**Abstract**: Meta-learning methods for regression like Neural (Diffusion) Processes achieve impressive results, but with these models it can be difficult to incorporate expert prior knowledge and information contained in metadata. Large Language Models (LLMs) are trained on giant corpora including varied real-world regression datasets alongside their descriptions and metadata, leading to impressive performance on a range of downstream tasks. Recent work has extended this to regression tasks and is able to leverage such prior knowledge and metadata, achieving surprisingly good performance, but this still rarely matches dedicated meta-learning methods. Here we introduce a general method for sampling from a product-of-experts of a diffusion or flow matching model and an `expert' with binned probability density; we apply this to combine neural diffusion processes with LLM token probabilities for regression (which may incorporate textual knowledge), exceeding the empirical performance of either alone. 

---
# MLB: A Scenario-Driven Benchmark for Evaluating Large Language Models in Clinical Applications 

**Authors**: Qing He, Dongsheng Bi, Jianrong Lu, Minghui Yang, Zixiao Chen, Jiacheng Lu, Jing Chen, Nannan Du, Xiao Cu, Sijing Wu, Peng Xiang, Yinyin Hu, Yi Guo, Chunpu Li, Shaoyang Li, Zhuo Dong, Ming Jiang, Shuai Guo, Liyun Feng, Jin Peng, Jian Wang, Jinjie Gu, Junwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.06193)  

**Abstract**: The proliferation of Large Language Models (LLMs) presents transformative potential for healthcare, yet practical deployment is hindered by the absence of frameworks that assess real-world clinical utility. Existing benchmarks test static knowledge, failing to capture the dynamic, application-oriented capabilities required in clinical practice. To bridge this gap, we introduce a Medical LLM Benchmark MLB, a comprehensive benchmark evaluating LLMs on both foundational knowledge and scenario-based reasoning. MLB is structured around five core dimensions: Medical Knowledge (MedKQA), Safety and Ethics (MedSE), Medical Record Understanding (MedRU), Smart Services (SmartServ), and Smart Healthcare (SmartCare). The benchmark integrates 22 datasets (17 newly curated) from diverse Chinese clinical sources, covering 64 clinical specialties. Its design features a rigorous curation pipeline involving 300 licensed physicians. Besides, we provide a scalable evaluation methodology, centered on a specialized judge model trained via Supervised Fine-Tuning (SFT) on expert annotations. Our comprehensive evaluation of 10 leading models reveals a critical translational gap: while the top-ranked model, Kimi-K2-Instruct (77.3% accuracy overall), excels in structured tasks like information extraction (87.8% accuracy in MedRU), performance plummets in patient-facing scenarios (61.3% in SmartServ). Moreover, the exceptional safety score (90.6% in MedSE) of the much smaller Baichuan-M2-32B highlights that targeted training is equally critical. Our specialized judge model, trained via SFT on a 19k expert-annotated medical dataset, achieves 92.1% accuracy, an F1-score of 94.37%, and a Cohen's Kappa of 81.3% for human-AI consistency, validating a reproducible and expert-aligned evaluation protocol. MLB thus provides a rigorous framework to guide the development of clinically viable LLMs. 

---
# From RLHF to Direct Alignment: A Theoretical Unification of Preference Learning for Large Language Models 

**Authors**: Tarun Raheja, Nilay Pochhi  

**Link**: [PDF](https://arxiv.org/pdf/2601.06108)  

**Abstract**: Aligning large language models (LLMs) with human preferences has become essential for safe and beneficial AI deployment. While Reinforcement Learning from Human Feedback (RLHF) established the dominant paradigm, a proliferation of alternatives -- Direct Preference Optimization (DPO), Identity Preference Optimization (IPO), Kahneman-Tversky Optimization (KTO), Simple Preference Optimization (SimPO), and many others -- has left practitioners without clear guidance on method selection. This survey provides a \textit{theoretical unification} of preference learning methods, revealing that the apparent diversity reduces to principled choices along three orthogonal axes: \textbf{(I) Preference Model} (what likelihood model underlies the objective), \textbf{(II) Regularization Mechanism} (how deviation from reference policies is controlled), and \textbf{(III) Data Distribution} (online vs.\ offline learning and coverage requirements). We formalize each axis with precise definitions and theorems, establishing key results including the coverage separation between online and offline methods, scaling laws for reward overoptimization, and conditions under which direct alignment methods fail. Our analysis reveals that failure modes -- length hacking, mode collapse, likelihood displacement -- arise from specific, predictable combinations of design choices. We synthesize empirical findings across 50+ papers and provide a practitioner's decision guide for method selection. The framework transforms preference learning from an empirical art into a theoretically grounded discipline. 

---
# CrossTrafficLLM: A Human-Centric Framework for Interpretable Traffic Intelligence via Large Language Model 

**Authors**: Zeming Du, Qitan Shao, Hongfei Liu, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06042)  

**Abstract**: While accurate traffic forecasting is vital for Intelligent Transportation Systems (ITS), effectively communicating predicted conditions via natural language for human-centric decision support remains a challenge and is often handled separately. To address this, we propose CrossTrafficLLM, a novel GenAI-driven framework that simultaneously predicts future spatiotemporal traffic states and generates corresponding natural language descriptions, specifically targeting conditional abnormal event summaries. We tackle the core challenge of aligning quantitative traffic data with qualitative textual semantics by leveraging Large Language Models (LLMs) within a unified architecture. This design allows generative textual context to improve prediction accuracy while ensuring generated reports are directly informed by the forecast. Technically, a text-guided adaptive graph convolutional network is employed to effectively merge high-level semantic information with the traffic network structure. Evaluated on the BJTT dataset, CrossTrafficLLM demonstrably surpasses state-of-the-art methods in both traffic forecasting performance and text generation quality. By unifying prediction and description generation, CrossTrafficLLM delivers a more interpretable, and actionable approach to generative traffic intelligence, offering significant advantages for modern ITS applications. 

---
# Judge Model for Large-scale Multimodality Benchmarks 

**Authors**: Min-Han Shih, Yu-Hsin Wu, Yu-Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.06106)  

**Abstract**: We propose a dedicated multimodal Judge Model designed to provide reliable, explainable evaluation across a diverse suite of tasks. Our benchmark spans text, audio, image, and video modalities, drawing from carefully sampled public datasets with fixed seeds to ensure reproducibility and minimize train test leakage. Instead of simple scoring, our framework aggregates multimodal judgments, analyzes the quality and reasoning consistency of model outputs, and generates diagnostic feedback. We evaluate several MLLMs, including Gemini 2.5, Phi 4, and Qwen 2.5, across 280 multimodal samples and compare judge model assessments with human annotators. Results show strong alignment between the Judge Model and human scores, demonstrating its potential as a scalable, interpretable evaluation pipeline for future multimodal AI research. 

---
# Certainty-Guided Reasoning in Large Language Models: A Dynamic Thinking Budget Approach 

**Authors**: João Paulo Nogueira, Wentao Sun, Alonso Silva, Laith Zumot  

**Link**: [PDF](https://arxiv.org/pdf/2509.07820)  

**Abstract**: The rise of large reasoning language models (LRLMs) has unlocked new potential for solving complex tasks. These models operate with a thinking budget, that is, a predefined number of reasoning tokens used to arrive at a solution. We propose a novel approach, inspired by the generator/discriminator framework in generative adversarial networks, in which a critic model periodically probes its own reasoning to assess whether it has reached a confident conclusion. If not, reasoning continues until a target certainty threshold is met. This mechanism adaptively balances efficiency and reliability by allowing early termination when confidence is high, while encouraging further reasoning when uncertainty persists. Through experiments on the AIME2024 and AIME2025 datasets, we show that Certainty-Guided Reasoning (CGR) improves baseline accuracy while reducing token usage. Importantly, extended multi-seed evaluations over 64 runs demonstrate that CGR is stable, reducing variance across seeds and improving exam-like performance under penalty-based grading. Additionally, our token savings analysis shows that CGR can eliminate millions of tokens in aggregate, with tunable trade-offs between certainty thresholds and efficiency. Together, these findings highlight certainty as a powerful signal for reasoning sufficiency. By integrating confidence into the reasoning process, CGR makes large reasoning language models more adaptive, trustworthy, and resource efficient, paving the way for practical deployment in domains where both accuracy and computational cost matter. 

---
# VirtualEnv: A Platform for Embodied AI Research 

**Authors**: Kabir Swain, Sijie Han, Ayush Raina, Jin Zhang, Shuang Li, Michael Stopa, Antonio Torralba  

**Link**: [PDF](https://arxiv.org/pdf/2601.07553)  

**Abstract**: As large language models (LLMs) continue to improve in reasoning and decision-making, there is a growing need for realistic and interactive environments where their abilities can be rigorously evaluated. We present VirtualEnv, a next-generation simulation platform built on Unreal Engine 5 that enables fine-grained benchmarking of LLMs in embodied and interactive scenarios. VirtualEnv supports rich agent-environment interactions, including object manipulation, navigation, and adaptive multi-agent collaboration, as well as game-inspired mechanics like escape rooms and procedurally generated environments. We provide a user-friendly API built on top of Unreal Engine, allowing researchers to deploy and control LLM-driven agents using natural language instructions. We integrate large-scale LLMs and vision-language models (VLMs), such as GPT-based models, to generate novel environments and structured tasks from multimodal inputs. Our experiments benchmark the performance of several popular LLMs across tasks of increasing complexity, analyzing differences in adaptability, planning, and multi-agent coordination. We also describe our methodology for procedural task generation, task validation, and real-time environment control. VirtualEnv is released as an open-source platform, we aim to advance research at the intersection of AI and gaming, enable standardized evaluation of LLMs in embodied AI settings, and pave the way for future developments in immersive simulations and interactive entertainment. 

---
# JudgeFlow: Agentic Workflow Optimization via Block Judge 

**Authors**: Zihan Ma, Zhikai Zhao, Chuanbo Hua, Federico Berto, Jinkyoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2601.07477)  

**Abstract**: Optimizing LLM-based agentic workflows is challenging for scaling AI capabilities. Current methods rely on coarse, end-to-end evaluation signals and lack fine-grained signals on where to refine, often resulting in inefficient or low-impact modifications. To address these limitations, we propose {\our{}}, an Evaluation-Judge-Optimization-Update pipeline. We incorporate reusable, configurable logic blocks into agentic workflows to capture fundamental forms of logic. On top of this abstraction, we design a dedicated Judge module that inspects execution traces -- particularly failed runs -- and assigns rank-based responsibility scores to problematic blocks. These fine-grained diagnostic signals are then leveraged by an LLM-based optimizer, which focuses modifications on the most problematic block in the workflow. Our approach improves sample efficiency, enhances interpretability through block-level diagnostics, and provides a scalable foundation for automating increasingly complex agentic workflows. We evaluate {\our{}} on mathematical reasoning and code generation benchmarks, where {\our{}} achieves superior performance and efficiency compared to existing methods. The source code is publicly available at this https URL. 

---
# Knowledge Distillation for LLM-Based Human Activity Recognition in Homes 

**Authors**: Julien Cumin, Oussama Er-Rahmany, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.07469)  

**Abstract**: Human Activity Recognition (HAR) is a central problem for context-aware applications, especially for smart homes and assisted living. A few very recent studies have shown that Large Language Models (LLMs) can be used for HAR at home, reaching high performance and addressing key challenges. In this paper, we provide new experimental results regarding the use of LLMs for HAR, on two state-of-the-art datasets. More specifically, we show how recognition performance evolves depending on the size of the LLM used. Moreover, we experiment on the use of knowledge distillation techniques to fine-tune smaller LLMs with HAR reasoning examples generated by larger LLMs. We show that such fine-tuned models can perform almost as well as the largest LLMs, while having 50 times less parameters. 

---
# IFDNS: An Iterative Feedback-Driven Neuro-Symbolic Method for Faithful Logical Reasoning 

**Authors**: Xiaoheng Wang, Tongxuan Liu, Zi Gong, Xianzhe Dong, Yuting Zeng, Minhan Hu, Weizhe Huang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.07464)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities across a wide range of reasoning tasks, including logical and mathematical problem-solving. While prompt-based methods like Chain-of-Thought (CoT) can enhance LLM reasoning abilities to some extent, they often suffer from a lack of faithfulness, where the derived conclusions may not align with the generated reasoning chain. To address this issue, researchers have explored neuro-symbolic approaches to bolster LLM logical reasoning capabilities. However, existing neuro-symbolic methods still face challenges with information loss during the process. To overcome these limitations, we introduce Iterative Feedback-Driven Neuro-Symbolic (IFDNS), a novel prompt-based method that employs a multi-round feedback mechanism to address LLM limitations in handling complex logical relationships. IFDNS utilizes iterative feedback during the logic extraction phase to accurately extract causal relationship statements and translate them into propositional and logical implication expressions, effectively mitigating information loss issues. Furthermore, IFDNS is orthogonal to existing prompt methods, allowing for seamless integration with various prompting approaches. Empirical evaluations across six datasets demonstrate the effectiveness of IFDNS in significantly improving the performance of CoT and Chain-of-Thought with Self-Consistency (CoT-SC). Specifically, IFDNS achieves a +9.40% accuracy boost for CoT on the LogiQA dataset and a +11.70% improvement for CoT-SC on the PrOntoQA dataset. 

---
# Beyond Dialogue Time: Temporal Semantic Memory for Personalized LLM Agents 

**Authors**: Miao Su, Yucan Guo, Zhongni Hou, Long Bai, Zixuan Li, Yufei Zhang, Guojun Yin, Wei Lin, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.07468)  

**Abstract**: Memory enables Large Language Model (LLM) agents to perceive, store, and use information from past dialogues, which is essential for personalization. However, existing methods fail to properly model the temporal dimension of memory in two aspects: 1) Temporal inaccuracy: memories are organized by dialogue time rather than their actual occurrence time; 2) Temporal fragmentation: existing methods focus on point-wise memory, losing durative information that captures persistent states and evolving patterns. To address these limitations, we propose Temporal Semantic Memory (TSM), a memory framework that models semantic time for point-wise memory and supports the construction and utilization of durative memory. During memory construction, it first builds a semantic timeline rather than a dialogue one. Then, it consolidates temporally continuous and semantically related information into a durative memory. During memory utilization, it incorporates the query's temporal intent on the semantic timeline, enabling the retrieval of temporally appropriate durative memories and providing time-valid, duration-consistent context to support response generation. Experiments on LongMemEval and LoCoMo show that TSM consistently outperforms existing methods and achieves up to 12.2% absolute improvement in accuracy, demonstrating the effectiveness of the proposed method. 

---
# Agentic Diagnostic Reasoning over Telecom and Datacenter Infrastructure 

**Authors**: Nicolas Tacheny  

**Link**: [PDF](https://arxiv.org/pdf/2601.07342)  

**Abstract**: Large-scale telecom and datacenter infrastructures rely on multi-layered service and resource models, where failures propagate across physical and logical components and affect multiple customers. Traditional approaches to root cause analysis(RCA) rely on hard-coded graph traversal algorithms or rule-based correlation engines, which are costly to maintain and tightly coupled to the infrastructure model.
In this work, we introduce an agentic diagnostic framework where a Large Language Model (LLM) performs step-wise investigation using a constrained tool space exposed through the Model Context Protocol (MCP). Instead of embedding causal logic or traversal algorithms into the application, the agent autonomously navigates the infrastructure model by invoking tools for service lookup, dependency retrieval, structured and unstructured data, and event analysis, and impact discovery. We define an investigation protocol that structures the agent's reasoning and ensures grounding, reproducibility, and safe handling of missing or ambiguous information.
This work lays the foundation for autonomous incident resolution and change impact mitigation. Future systems will not only diagnose and remediate infrastructure failures, but also predict the impact of planned changes on services and customers, enabling operators to mitigate risks before executing maintenance operations. 

---
# Group Pattern Selection Optimization: Let LRMs Pick the Right Pattern for Reasoning 

**Authors**: Hanbin Wang, Jingwei Song, Jinpeng Li, Fei Mi, Lifeng Shang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07238)  

**Abstract**: Large reasoning models (LRMs) exhibit diverse high-level reasoning patterns (e.g., direct solution, reflection-and-verification, and exploring multiple solutions), yet prevailing training recipes implicitly bias models toward a limited set of dominant patterns. Through a systematic analysis, we identify substantial accuracy variance across these patterns on mathematics and science benchmarks, revealing that a model's default reasoning pattern is often sub-optimal for a given problem. To address this, we introduce Group Pattern Selection Optimization (GPSO), a reinforcement learning framework that extends GRPO by incorporating multi-pattern rollouts, verifier-guided optimal pattern selection per problem, and attention masking during optimization to prevent the leakage of explicit pattern suffixes into the learned policy. By exploring a portfolio of diverse reasoning strategies and optimizing the policy on the most effective ones, GPSO enables the model to internalize the mapping from problem characteristics to optimal reasoning patterns. Extensive experiments demonstrate that GPSO delivers consistent and substantial performance gains across various model backbones and benchmarks, effectively mitigating pattern sub-optimality and fostering more robust, adaptable reasoning. All data and codes are available at this https URL. 

---
# Active Context Compression: Autonomous Memory Management in LLM Agents 

**Authors**: Nikhil Verma  

**Link**: [PDF](https://arxiv.org/pdf/2601.07190)  

**Abstract**: Large Language Model (LLM) agents struggle with long-horizon software engineering tasks due to "Context Bloat." As interaction history grows, computational costs explode, latency increases, and reasoning capabilities degrade due to distraction by irrelevant past errors. Existing solutions often rely on passive, external summarization mechanisms that the agent cannot control. This paper proposes Focus, an agent-centric architecture inspired by the biological exploration strategies of Physarum polycephalum (slime mold). The Focus Agent autonomously decides when to consolidate key learnings into a persistent "Knowledge" block and actively withdraws (prunes) the raw interaction history. Using an optimized scaffold matching industry best practices (persistent bash + string-replacement editor), we evaluated Focus on N=5 context-intensive instances from SWE-bench Lite using Claude Haiku 4.5. With aggressive prompting that encourages frequent compression, Focus achieves 22.7% token reduction (14.9M -> 11.5M tokens) while maintaining identical accuracy (3/5 = 60% for both agents). Focus performed 6.0 autonomous compressions per task on average, with token savings up to 57% on individual instances. We demonstrate that capable models can autonomously self-regulate their context when given appropriate tools and prompting, opening pathways for cost-aware agentic systems without sacrificing task performance. 

---
# Stochastic CHAOS: Why Deterministic Inference Kills, and Distributional Variability Is the Heartbeat of Artifical Cognition 

**Authors**: Tanmay Joshi, Shourya Aggarwal, Anusa Saha, Aadi Pandey, Shreyash Dhoot, Vighnesh Rai, Raxit Goswami, Aman Chadha, Vinija Jain, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2601.07239)  

**Abstract**: Deterministic inference is a comforting ideal in classical software: the same program on the same input should always produce the same output. As large language models move into real-world deployment, this ideal has been imported wholesale into inference stacks. Recent work from the Thinking Machines Lab has presented a detailed analysis of nondeterminism in LLM inference, showing how batch-invariant kernels and deterministic attention can enforce bitwise-identical outputs, positioning deterministic inference as a prerequisite for reproducibility and enterprise reliability.
In this paper, we take the opposite stance. We argue that, for LLMs, deterministic inference kills. It kills the ability to model uncertainty, suppresses emergent abilities, collapses reasoning into a single brittle path, and weakens safety alignment by hiding tail risks. LLMs implement conditional distributions over outputs, not fixed functions. Collapsing these distributions to a single canonical completion may appear reassuring, but it systematically conceals properties central to artificial cognition. We instead advocate Stochastic CHAOS, treating distributional variability as a signal to be measured and controlled.
Empirically, we show that deterministic inference is systematically misleading. Single-sample deterministic evaluation underestimates both capability and fragility, masking failure probability under paraphrases and noise. Phase-like transitions associated with emergent abilities disappear under greedy decoding. Multi-path reasoning degrades when forced onto deterministic backbones, reducing accuracy and diagnostic insight. Finally, deterministic evaluation underestimates safety risk by hiding rare but dangerous behaviors that appear only under multi-sample evaluation. 

---
# OpenTinker: Separating Concerns in Agentic Reinforcement Learning 

**Authors**: Siqi Zhu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2601.07376)  

**Abstract**: We introduce OpenTinker, an infrastructure for reinforcement learning (RL) of large language model (LLM) agents built around a separation of concerns across algorithm design, execution, and agent-environment interaction. Rather than relying on monolithic, end-to-end RL pipelines, OpenTinker decomposes agentic learning systems into lightweight, composable components with clearly defined abstraction boundaries. Users specify agents, environments, and interaction protocols, while inference and training are delegated to a managed execution runtime. OpenTinker introduces a centralized scheduler for managing training and inference workloads, including LoRA-based and full-parameter RL, supervised fine-tuning, and inference, over shared resources. We further discuss design principles for extending OpenTinker to multi-agent training. Finally, we present a set of RL use cases that demonstrate the effectiveness of the framework in practical agentic learning scenarios. 

---
# LLMRouterBench: A Massive Benchmark and Unified Framework for LLM Routing 

**Authors**: Hao Li, Yiqun Zhang, Zhaoyan Guo, Chenxu Wang, Shengji Tang, Qiaosheng Zhang, Yang Chen, Biqing Qi, Peng Ye, Lei Bai, Zhen Wang, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07206)  

**Abstract**: Large language model (LLM) routing assigns each query to the most suitable model from an ensemble. We introduce LLMRouterBench, a large-scale benchmark and unified framework for LLM routing. It comprises over 400K instances from 21 datasets and 33 models. Moreover, it provides comprehensive metrics for both performance-oriented routing and performance-cost trade-off routing, and integrates 10 representative routing baselines. Using LLMRouterBench, we systematically re-evaluate the field. While confirming strong model complementarity-the central premise of LLM routing-we find that many routing methods exhibit similar performance under unified evaluation, and several recent approaches, including commercial routers, fail to reliably outperform a simple baseline. Meanwhile, a substantial gap remains to the Oracle, driven primarily by persistent model-recall failures. We further show that backbone embedding models have limited impact, that larger ensembles exhibit diminishing returns compared to careful model curation, and that the benchmark also enables latency-aware analysis. All code and data are available at this https URL. 

---
# Dr. Zero: Self-Evolving Search Agents without Training Data 

**Authors**: Zhenrui Yue, Kartikeya Upasani, Xianjun Yang, Suyu Ge, Shaoliang Nie, Yuning Mao, Zhe Liu, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07055)  

**Abstract**: As high-quality data becomes increasingly difficult to obtain, data-free self-evolution has emerged as a promising paradigm. This approach allows large language models (LLMs) to autonomously generate and solve complex problems, thereby improving their reasoning capabilities. However, multi-turn search agents struggle in data-free self-evolution due to the limited question diversity and the substantial compute required for multi-step reasoning and tool using. In this work, we introduce Dr. Zero, a framework enabling search agents to effectively self-evolve without any training data. In particular, we design a self-evolution feedback loop where a proposer generates diverse questions to train a solver initialized from the same base model. As the solver evolves, it incentivizes the proposer to produce increasingly difficult yet solvable tasks, thus establishing an automated curriculum to refine both agents. To enhance training efficiency, we also introduce hop-grouped relative policy optimization (HRPO). This method clusters structurally similar questions to construct group-level baselines, effectively minimizing the sampling overhead in evaluating each query's individual difficulty and solvability. Consequently, HRPO significantly reduces the compute requirements for solver training without compromising performance or stability. Extensive experiment results demonstrate that the data-free Dr. Zero matches or surpasses fully supervised search agents, proving that complex reasoning and search capabilities can emerge solely through self-evolution. 

---
# ENTRA: Entropy-Based Redundancy Avoidance in Large Language Model Reasoning 

**Authors**: Ruichu Cai, Haopeng Du, Qingwen Lin, Yutong Chen, Zijian Li, Boyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07123)  

**Abstract**: Large Reasoning Models (LRMs) often suffer from overthinking, generating unnecessarily long reasoning chains even for simple tasks. This leads to substantial computational overhead with limited performance gain, primarily due to redundant verification and repetitive generation. While prior work typically constrains output length or optimizes correctness, such coarse supervision fails to guide models toward concise yet accurate inference. In this paper, we propose ENTRA, an entropy-based training framework that suppresses redundant reasoning while preserving performance. ENTRA first estimates the token-level importance using a lightweight Bidirectional Importance Estimation (BIE) method, which accounts for both prediction confidence and forward influence. It then computes a redundancy reward based on the entropy of low-importance tokens, normalized by its theoretical upper bound, and optimizes this reward via reinforcement learning. Experiments on mathematical reasoning benchmarks demonstrate that ENTRA reduces output length by 37% to 53% with no loss-and in some cases, gains-in accuracy. Our approach offers a principled and efficient solution to reduce overthinking in LRMs, and provides a generalizable path toward redundancy-aware reasoning optimization. 

---
# AscendKernelGen: A Systematic Study of LLM-Based Kernel Generation for Neural Processing Units 

**Authors**: Xinzi Cao, Jianyang Zhai, Pengfei Li, Zhiheng Hu, Cen Yan, Bingxu Mu, Guanghuan Fang, Bin She, Jiayu Li, Yihan Su, Dongyang Tao, Xiansong Huang, Fan Xu, Feidiao Yang, Yao Lu, Chang-Dong Wang, Yutong Lu, Weicheng Xue, Bin Zhou, Yonghong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2601.07160)  

**Abstract**: To meet the ever-increasing demand for computational efficiency, Neural Processing Units (NPUs) have become critical in modern AI infrastructure. However, unlocking their full potential requires developing high-performance compute kernels using vendor-specific Domain-Specific Languages (DSLs), a task that demands deep hardware expertise and is labor-intensive. While Large Language Models (LLMs) have shown promise in general code generation, they struggle with the strict constraints and scarcity of training data in the NPU domain. Our preliminary study reveals that state-of-the-art general-purpose LLMs fail to generate functional complex kernels for Ascend NPUs, yielding a near-zero success rate. To address these challenges, we propose AscendKernelGen, a generation-evaluation integrated framework for NPU kernel development. We introduce Ascend-CoT, a high-quality dataset incorporating chain-of-thought reasoning derived from real-world kernel implementations, and KernelGen-LM, a domain-adaptive model trained via supervised fine-tuning and reinforcement learning with execution feedback. Furthermore, we design NPUKernelBench, a comprehensive benchmark for assessing compilation, correctness, and performance across varying complexity levels. Experimental results demonstrate that our approach significantly bridges the gap between general LLMs and hardware-specific coding. Specifically, the compilation success rate on complex Level-2 kernels improves from 0% to 95.5% (Pass@10), while functional correctness achieves 64.3% compared to the baseline's complete failure. These results highlight the critical role of domain-specific reasoning and rigorous evaluation in automating accelerator-aware code generation. 

---
# LLM Performance Predictors: Learning When to Escalate in Hybrid Human-AI Moderation Systems 

**Authors**: Or Bachar, Or Levi, Sardhendu Mishra, Adi Levi, Manpreet Singh Minhas, Justin Miller, Omer Ben-Porat, Eilon Sheetrit, Jonathan Morra  

**Link**: [PDF](https://arxiv.org/pdf/2601.07006)  

**Abstract**: As LLMs are increasingly integrated into human-in-the-loop content moderation systems, a central challenge is deciding when their outputs can be trusted versus when escalation for human review is preferable. We propose a novel framework for supervised LLM uncertainty quantification, learning a dedicated meta-model based on LLM Performance Predictors (LPPs) derived from LLM outputs: log-probabilities, entropy, and novel uncertainty attribution indicators. We demonstrate that our method enables cost-aware selective classification in real-world human-AI workflows: escalating high-risk cases while automating the rest. Experiments across state-of-the-art LLMs, including both off-the-shelf (Gemini, GPT) and open-source (Llama, Qwen), on multimodal and multilingual moderation tasks, show significant improvements over existing uncertainty estimators in accuracy-cost trade-offs. Beyond uncertainty estimation, the LPPs enhance explainability by providing new insights into failure conditions (e.g., ambiguous content vs. under-specified policy). This work establishes a principled framework for uncertainty-aware, scalable, and responsible human-AI moderation workflows. 

---
# ET-Agent: Incentivizing Effective Tool-Integrated Reasoning Agent via Behavior Calibration 

**Authors**: Yifei Chen, Guanting Dong, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2601.06860)  

**Abstract**: Large Language Models (LLMs) can extend their parameter knowledge limits by adopting the Tool-Integrated Reasoning (TIR) paradigm. However, existing LLM-based agent training framework often focuses on answers' accuracy, overlooking specific alignment for behavior patterns. Consequently, agent often exhibits ineffective actions during TIR tasks, such as redundant and insufficient tool calls. How to calibrate erroneous behavioral patterns when executing TIR tasks, thereby exploring effective trajectories, remains an open-ended problem. In this paper, we propose ET-Agent, a training framework for calibrating agent's tool-use behavior through two synergistic perspectives: Self-evolving Data Flywheel and Behavior Calibration Training. Specifically, we introduce a self-evolutionary data flywheel to generate enhanced data, used to fine-tune LLM to improve its exploration ability. Based on this, we implement an two-phases behavior-calibration training framework. It is designed to progressively calibrate erroneous behavioral patterns to optimal behaviors. Further in-depth experiments confirm the superiority of \ourmodel{} across multiple dimensions, including correctness, efficiency, reasoning conciseness, and tool execution accuracy. Our ET-Agent framework provides practical insights for research in the TIR field. Codes can be found in this https URL 

---
# mind_call: A Dataset for Mental Health Function Calling with Large Language Models 

**Authors**: Fozle Rabbi Shafi, M. Anwar Hossain, Salimur Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2601.06937)  

**Abstract**: Large Language Model (LLM)-based systems increasingly rely on function calling to enable structured and controllable interaction with external data sources, yet existing datasets do not address mental health-oriented access to wearable sensor data. This paper presents a synthetic function-calling dataset designed for mental health assistance grounded in wearable health signals such as sleep, physical activity, cardiovascular measures, stress indicators, and metabolic data. The dataset maps diverse natural language queries to standardized API calls derived from a widely adopted health data schema. Each sample includes a user query, a query category, an explicit reasoning step, a normalized temporal parameter, and a target function. The dataset covers explicit, implicit, behavioral, symptom-based, and metaphorical expressions, which reflect realistic mental health-related user interactions. This resource supports research on intent grounding, temporal reasoning, and reliable function invocation in LLM-based mental health agents and is publicly released to promote reproducibility and future work. 

---
# Agentic AI Empowered Intent-Based Networking for 6G 

**Authors**: Genze Jiang, Kezhi Wang, Xiaomin Chen, Yizhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06640)  

**Abstract**: The transition towards sixth-generation (6G) wireless networks necessitates autonomous orchestration mechanisms capable of translating high-level operational intents into executable network configurations. Existing approaches to Intent-Based Networking (IBN) rely upon either rule-based systems that struggle with linguistic variation or end-to-end neural models that lack interpretability and fail to enforce operational constraints. This paper presents a hierarchical multi-agent framework where Large Language Model (LLM) based agents autonomously decompose natural language intents, consult domain-specific specialists, and synthesise technically feasible network slice configurations through iterative reasoning-action (ReAct) cycles. The proposed architecture employs an orchestrator agent coordinating two specialist agents, i.e., Radio Access Network (RAN) and Core Network agents, via ReAct-style reasoning, grounded in structured network state representations. Experimental evaluation across diverse benchmark scenarios shows that the proposed system outperforms rule-based systems and direct LLM prompting, with architectural principles applicable to Open RAN (O-RAN) deployments. The results also demonstrate that whilst contemporary LLMs possess general telecommunications knowledge, network automation requires careful prompt engineering to encode context-dependent decision thresholds, advancing autonomous orchestration capabilities for next-generation wireless systems. 

---
# QMAVIS: Long Video-Audio Understanding using Fusion of Large Multimodal Models 

**Authors**: Zixing Lin, Jiale Wang, Gee Wah Ng, Lee Onn Mak, Chan Zhi Yang Jeriel, Jun Yang Lee, Yaohao Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.06573)  

**Abstract**: Large Multimodal Models (LMMs) for video-audio understanding have traditionally been evaluated only on shorter videos of a few minutes long. In this paper, we introduce QMAVIS (Q Team-Multimodal Audio Video Intelligent Sensemaking), a novel long video-audio understanding pipeline built through a late fusion of LMMs, Large Language Models, and speech recognition models. QMAVIS addresses the gap in long-form video analytics, particularly for longer videos of a few minutes to beyond an hour long, opening up new potential applica- tions in sensemaking, video content analysis, embodied AI, etc. Quantitative experiments using QMAVIS demonstrated a 38.75% improvement over state-of-the-art video-audio LMMs like Vide- oLlaMA2 and InternVL2 on the VideoMME (with subtitles) dataset, which comprises long videos with audio information. Evaluations on other challenging video understanding datasets like PerceptionTest and EgoSchema saw up to 2% improvement, indicating competitive performance. Qualitative experiments also showed that QMAVIS is able to extract the nuances of different scenes in a long video audio content while understanding the overarching narrative. Ablation studies were also conducted to ascertain the impact of each component in the fusion pipeline. 

---
# DRAGON: LLM-Driven Decomposition and Reconstruction Agents for Large-Scale Combinatorial Optimization 

**Authors**: Shengkai Chen, Zhiguang Cao, Jianan Zhou, Yaoxin Wu, Senthilnath Jayavelu, Zhuoyi Lin, Xiaoli Li, Shili Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06502)  

**Abstract**: Large Language Models (LLMs) have recently shown promise in addressing combinatorial optimization problems (COPs) through prompt-based strategies. However, their scalability and generalization remain limited, and their effectiveness diminishes as problem size increases, particularly in routing problems involving more than 30 nodes. We propose DRAGON, which stands for Decomposition and Reconstruction Agents Guided OptimizatioN, a novel framework that combines the strengths of metaheuristic design and LLM reasoning. Starting from an initial global solution, DRAGON autonomously identifies regions with high optimization potential and strategically decompose large-scale COPs into manageable subproblems. Each subproblem is then reformulated as a concise, localized optimization task and solved through targeted LLM prompting guided by accumulated experiences. Finally, the locally optimized solutions are systematically reintegrated into the original global context to yield a significantly improved overall outcome. By continuously interacting with the optimization environment and leveraging an adaptive experience memory, the agents iteratively learn from feedback, effectively coupling symbolic reasoning with heuristic search. Empirical results show that, unlike existing LLM-based solvers limited to small-scale instances, DRAGON consistently produces feasible solutions on TSPLIB, CVRPLIB, and Weibull-5k bin packing benchmarks, and achieves near-optimal results (0.16% gap) on knapsack problems with over 3M variables. This work shows the potential of feedback-driven language agents as a new paradigm for generalizable and interpretable large-scale optimization. 

---
# Does Inference Scaling Improve Reasoning Faithfulness? A Multi-Model Analysis of Self-Consistency Tradeoffs 

**Authors**: Deep Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2601.06423)  

**Abstract**: Self-consistency has emerged as a popular technique for improving large language model accuracy on reasoning tasks. The approach is straightforward: generate multiple reasoning paths and select the most common answer through majority voting. While this reliably boosts accuracy, it remains unclear whether these gains reflect genuine improvements in reasoning quality. We investigate a fundamental question that has not been studied before: does inference scaling improve reasoning faithfulness?
We conduct a comprehensive empirical study across four frontier models (GPT-5.2, Claude Opus 4.5, Gemini-3-flash-preview, and DeepSeek-v3.2) on 100 GSM8K mathematical reasoning problems. Our analysis employs bootstrap confidence intervals, McNemar's tests for paired comparisons, and Cohen's d effect sizes to quantify the effects rigorously. The results reveal striking differences across models that challenge common assumptions about self-consistency.
GPT-5.2 shows the expected pattern: accuracy improves from 78% to 90% at N=5, with faithfulness remaining relatively stable (0.540 to 0.510). Claude Opus 4.5 tells a completely different story. Its accuracy actually drops from 78% to 74.3% while faithfulness jumps dramatically from 0.270 to 0.891 at N=5. DeepSeek-v3.2, already at 98% accuracy, shows ceiling effects with modest faithfulness gains (0.440 to 0.541). Gemini-3-flash improves from 81% to 86% accuracy with a slight faithfulness decrease (0.260 to 0.212).
Problem difficulty analysis reveals that GPT-5.2 solves 82% of hard problems while breaking only 13% of easy ones. Claude, in contrast, breaks 23% of easy problems, explaining its accuracy decrease. These findings matter for practitioners: self-consistency is not universally beneficial, and teams should test their specific models before deployment. We release our code and provide practical recommendations for navigating these tradeoffs. 

---
# From Text to Simulation: A Multi-Agent LLM Workflow for Automated Chemical Process Design 

**Authors**: Xufei Tian, Wenli Du, Shaoyi Yang, Han Hu, Hui Xin, Shifeng Qu, Ke Ye  

**Link**: [PDF](https://arxiv.org/pdf/2601.06776)  

**Abstract**: Process simulation is a critical cornerstone of chemical engineering design. Current automated chemical design methodologies focus mainly on various representations of process flow diagrams. However, transforming these diagrams into executable simulation flowsheets remains a time-consuming and labor-intensive endeavor, requiring extensive manual parameter configuration within simulation software. In this work, we propose a novel multi-agent workflow that leverages the semantic understanding capabilities of large language models(LLMs) and enables iterative interactions with chemical process simulation software, achieving end-to-end automated simulation from textual process specifications to computationally validated software configurations for design enhancement. Our approach integrates four specialized agents responsible for task understanding, topology generation, parameter configuration, and evaluation analysis, respectively, coupled with Enhanced Monte Carlo Tree Search to accurately interpret semantics and robustly generate configurations. Evaluated on Simona, a large-scale process description dataset, our method achieves a 31.1% improvement in the simulation convergence rate compared to state-of-the-art baselines and reduces the design time by 89. 0% compared to the expert manual design. This work demonstrates the potential of AI-assisted chemical process design, which bridges the gap between conceptual design and practical implementation. Our workflow is applicable to diverse process-oriented industries, including pharmaceuticals, petrochemicals, food processing, and manufacturing, offering a generalizable solution for automated process design. 

---
# Styles + Persona-plug = Customized LLMs 

**Authors**: Yutong Song, Jiang Wu, Shaofan Yuan, Chengze Shen, Jian Wang, Amir Rahmani, Nikil Dutt, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06362)  

**Abstract**: We discover a previously overlooked challenge in personalized text generation: personalization methods are increasingly applied under explicit style instructions, yet their behavior under such constraints remains poorly understood. To balance implicit personalization and explicit style, we formulate personalization as a distributional residual and propose PsPLUG, a lightweight soft-prompt plug-in trained with style-conditioned preference contrasts. Across LaMP benchmark, our framework improves persona alignment, maintains stylistic fidelity, and outperforms retrieval-based and soft-prompt baselines with minimal computation. These results show that residual modeling provides a simple and principled foundation for controllable, style-aware LLM personalization. 

---
# HiMem: Hierarchical Long-Term Memory for LLM Long-Horizon Agents 

**Authors**: Ningning Zhang, Xingxing Yang, Zhizhong Tan, Weiping Deng, Wenyong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06377)  

**Abstract**: Although long-term memory systems have made substantial progress in recent years, they still exhibit clear limitations in adaptability, scalability, and self-evolution under continuous interaction settings. Inspired by cognitive theories, we propose HiMem, a hierarchical long-term memory framework for long-horizon dialogues, designed to support memory construction, retrieval, and dynamic updating during sustained interactions. HiMem constructs cognitively consistent Episode Memory via a Topic-Aware Event--Surprise Dual-Channel Segmentation strategy, and builds Note Memory that captures stable knowledge through a multi-stage information extraction pipeline. These two memory types are semantically linked to form a hierarchical structure that bridges concrete interaction events and abstract knowledge, enabling efficient retrieval without sacrificing information fidelity. HiMem supports both hybrid and best-effort retrieval strategies to balance accuracy and efficiency, and incorporates conflict-aware Memory Reconsolidation to revise and supplement stored knowledge based on retrieval feedback. This design enables continual memory self-evolution over long-term use. Experimental results on long-horizon dialogue benchmarks demonstrate that HiMem consistently outperforms representative baselines in accuracy, consistency, and long-term reasoning, while maintaining favorable efficiency. Overall, HiMem provides a principled and scalable design paradigm for building adaptive and self-evolving LLM-based conversational agents. The code is available at this https URL. 

---
# PCoKG: Personality-aware Commonsense Reasoning with Debate 

**Authors**: Weijie Li, Zhongqing Wang, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.06234)  

**Abstract**: Most commonsense reasoning models overlook the influence of personality traits, limiting their effectiveness in personalized systems such as dialogue generation. To address this limitation, we introduce the Personality-aware Commonsense Knowledge Graph (PCoKG), a structured dataset comprising 521,316 quadruples. We begin by employing three evaluators to score and filter events from the ATOMIC dataset, selecting those that are likely to elicit diverse reasoning patterns across different personality types. For knowledge graph construction, we leverage the role-playing capabilities of large language models (LLMs) to perform reasoning tasks. To enhance the quality of the generated knowledge, we incorporate a debate mechanism consisting of a proponent, an opponent, and a judge, which iteratively refines the outputs through feedback loops. We evaluate the dataset from multiple perspectives and conduct fine-tuning and ablation experiments using multiple LLM backbones to assess PCoKG's robustness and the effectiveness of its construction pipeline. Our LoRA-based fine-tuning results indicate a positive correlation between model performance and the parameter scale of the base models. Finally, we apply PCoKG to persona-based dialogue generation, where it demonstrates improved consistency between generated responses and reference outputs. This work bridges the gap between commonsense reasoning and individual cognitive differences, enabling the development of more personalized and context-aware AI systems. 

---
# Seeing through the Conflict: Transparent Knowledge Conflict Handling in Retrieval-Augmented Generation 

**Authors**: Hua Ye, Siyuan Chen, Ziqi Zhong, Canran Xiao, Haoliang Zhang, Yuhan Wu, Fei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.06842)  

**Abstract**: Large language models (LLMs) equipped with retrieval--the Retrieval-Augmented Generation (RAG) paradigm--should combine their parametric knowledge with external evidence, yet in practice they often hallucinate, over-trust noisy snippets, or ignore vital context. We introduce TCR (Transparent Conflict Resolution), a plug-and-play framework that makes this decision process observable and controllable. TCR (i) disentangles semantic match and factual consistency via dual contrastive encoders, (ii) estimates self-answerability to gauge confidence in internal memory, and (iii) feeds the three scalar signals to the generator through a lightweight soft-prompt with SNR-based weighting. Across seven benchmarks TCR improves conflict detection (+5-18 F1), raises knowledge-gap recovery by +21.4 pp and cuts misleading-context overrides by -29.3 pp, while adding only 0.3% parameters. The signals align with human judgements and expose temporal decision patterns. 

---
# Rational Synthesizers or Heuristic Followers? Analyzing LLMs in RAG-based Question-Answering 

**Authors**: Atharv Naphade  

**Link**: [PDF](https://arxiv.org/pdf/2601.06189)  

**Abstract**: Retrieval-Augmented Generation (RAG) is the prevailing paradigm for grounding Large Language Models (LLMs), yet the mechanisms governing how models integrate groups of conflicting retrieved evidence remain opaque. Does an LLM answer a certain way because the evidence is factually strong, because of a prior belief, or merely because it is repeated frequently? To answer this, we introduce GroupQA, a curated dataset of 1,635 controversial questions paired with 15,058 diversely-sourced evidence documents, annotated for stance and qualitative strength. Through controlled experiments, we characterize group-level evidence aggregation dynamics: Paraphrasing an argument can be more persuasive than providing distinct independent support; Models favor evidence presented first rather than last, and Larger models are increasingly resistant to adapt to presented evidence. Additionally, we find that LLM explanations to group-based answers are unfaithful. Together, we show that LLMs behave consistently as vulnerable heuristic followers, with direct implications for improving RAG system design. 

---
# Neuro-Symbolic Compliance: Integrating LLMs and SMT Solvers for Automated Financial Legal Analysis 

**Authors**: Yung-Shen Hsia, Fang Yu, Jie-Hong Roland Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06181)  

**Abstract**: Financial regulations are increasingly complex, hindering automated compliance-especially the maintenance of logical consistency with minimal human oversight. We introduce a Neuro-Symbolic Compliance Framework that integrates Large Language Models (LLMs) with Satisfiability Modulo Theories (SMT) solvers to enable formal verifiability and optimization-based compliance correction. The LLM interprets statutes and enforcement cases to generate SMT constraints, while the solver enforces consistency and computes the minimal factual modification required to restore legality when penalties arise. Unlike transparency-oriented methods, our approach emphasizes logic-driven optimization, delivering verifiable, legally consistent reasoning rather than post-hoc explanation. Evaluated on 87 enforcement cases from Taiwan's Financial Supervisory Commission (FSC), the system attains 86.2% correctness in SMT code generation, improves reasoning efficiency by over 100x, and consistently corrects violations-establishing a preliminary foundation for optimization-based compliance applications. 

---
# Student Guides Teacher: Weak-to-Strong Inference via Spectral Orthogonal Exploration 

**Authors**: Dayu Wang, Jiaye Yang, Weikang Li, Jiahui Liang, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.06160)  

**Abstract**: While Large Language Models (LLMs) demonstrate near-human capabilities, they often suffer from "Reasoning Collapse" in complex mathematical proving and long-horizon planning. Models tend to degenerate into low-rank Bias Manifold, where stochastic sampling merely produces lexical variations of erroneous logic rather than semantic exploration. This geometric collapse renders the model "blind" to high-value solutions that lie within its Null Space. To address this, we propose Spectral Orthogonal Exploration (SOE), a geometric framework operating on a counter-intuitive "Student Guides Teacher" paradigm. Specifically, we utilize a weak auxiliary agent not for imitation, but as an orthogonal probe. By explicitly navigating the Teacher's Null Space, SOE serves as a geometric bridge, effectively ejecting the model from local optima to explore diverse, high-value solution spaces. Experiments on mathematical benchmarks demonstrate that, relative to baseline methods, our approach improves average accuracy by 62.4% and increases average sampling efficiency by 113.7%, indicating a promising path toward overcoming performance plateaus in advanced reasoning tasks. 

---
# Beyond Reproducibility: Token Probabilities Expose Large Language Model Nondeterminism 

**Authors**: Tairan Fu, Gonzalo Martínez, Javier Conde, Carlos Arriaga, Pedro Reviriego, Xiuyuan Qi, Shanshan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.06118)  

**Abstract**: The execution of Large Language Models (LLMs) has been shown to produce nondeterministic results when run on Graphics Processing Units (GPUs), even when they are configured to produce deterministic results. This is due to the finite precision effects of the arithmetic operations, which depend on the order in which they are executed. This order, in turn, depends on the processes that are running concurrently on the GPU. Previous studies have focused on the impact of nondeterminism on the text generated by the LLMs or on proposing mechanisms to achieve deterministic execution. This work takes a closer look at nondeterminism by analyzing the variations on the token probabilities, not on the generated text. Interestingly, all the models evaluated have similar results in both the trends and the actual values of the variations of the probabilities. In particular, the results show that the effects of nondeterminism are significant for token probabilities that are in the range of 0.1 to 0.9, while they are much smaller when the probabilities are close to 0 or 1. This has significant implications for our understanding of nondeterminism. The first is that nondeterminism will likely have a non-negligible impact on generated text when the temperature is not zero, as it introduces significant variations in the token probabilities except when they are close to 0 or 1. Secondly, it suggests that all models have similar non deterministic variations at the token probability level. Therefore, different variations in the performance of the generated text, for example, when measuring accuracy on a benchmark, seem to come from different token probabilities or response lengths. A third implication is that we may be able to estimate the impact of nondeterminism by running a single inference and analyzing the token level probabilities, instead of having to run the same inference many times. 

---
# NL2Dashboard: A Lightweight and Controllable Framework for Generating Dashboards with LLMs 

**Authors**: Boshen Shi, Kexin Yang, Yuanbo Yang, Guanguang Chang, Ce Chi, Zhendong Wang, Xing Wang, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2601.06126)  

**Abstract**: While Large Language Models (LLMs) have demonstrated remarkable proficiency in generating standalone charts, synthesizing comprehensive dashboards remains a formidable challenge. Existing end-to-end paradigms, which typically treat dashboard generation as a direct code generation task (e.g., raw HTML), suffer from two fundamental limitations: representation redundancy due to massive tokens spent on visual rendering, and low controllability caused by the entanglement of analytical reasoning and presentation. To address these challenges, we propose NL2Dashboard, a lightweight framework grounded in the principle of Analysis-Presentation Decoupling. We introduce a structured intermediate representation (IR) that encapsulates the dashboard's content, layout, and visual elements. Therefore, it confines the LLM's role to data analysis and intent translation, while offloading visual synthesis to a deterministic rendering engine. Building upon this framework, we develop a multi-agent system in which the IR-driven algorithm is instantiated as a suite of tools. Comprehensive experiments conducted with this system demonstrate that NL2Dashboard significantly outperforms state-of-the-art baselines across diverse domains, achieving superior visual quality, significantly higher token efficiency, and precise controllability in both generation and modification tasks. 

---
# Towards Infinite Length Extrapolation: A Unified Approach 

**Authors**: Nitin Vetcha  

**Link**: [PDF](https://arxiv.org/pdf/2601.06113)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing, but their ability to process long sequences is fundamentally limited by the context window size during training. Existing length extrapolation methods often suffer from performance degradation or computational inefficiencies. We thereby use a unified framework that reinterprets positional encoding methods as a decomposition of the attention score into a multiplicative transformation and an additive bias. This perspective not only subsumes popular approaches such as relative position embeddings and attention-bias moderated approaches but also exposes their inherent limitations in handling long-range dependencies. To address these shortcomings, motivated by our framework, we introduce Adaptive Positional Encoding (APE), which leverages adaptive frequency modulation and an intricately designed decay bias that incorporates linear, logarithmic, and square-root terms. Our theoretical analysis establishes conditions for infinite-context extrapolation, ensuring that the softmax normalization remains well-defined over unbounded sequences while preserving long-distance correlations, entropy boundedness and gradient positional sensitivity. We substantiate our claims with an experimental case study on TinyStories dataset as well as a new synthetic dataset, \emph{Long Tiny Stories} featuring stories up to 32,000 words. Relevant code, dataset and model weights are available at this https URL. 

---
# CBMAS: Cognitive Behavioral Modeling via Activation Steering 

**Authors**: Ahmed H. Ismail, Anthony Kuang, Ayo Akinkugbe, Kevin Zhu, Sean O'Brien  

**Link**: [PDF](https://arxiv.org/pdf/2601.06109)  

**Abstract**: Large language models (LLMs) often encode cognitive behaviors unpredictably across prompts, layers, and contexts, making them difficult to diagnose and control. We present CBMAS, a diagnostic framework for continuous activation steering, which extends cognitive bias analysis from discrete before/after interventions to interpretable trajectories. By combining steering vector construction with dense {\alpha}-sweeps, logit lens-based bias curves, and layer-site sensitivity analysis, our approach can reveal tipping points where small intervention strengths flip model behavior and show how steering effects evolve across layer depth. We argue that these continuous diagnostics offer a bridge between high-level behavioral evaluation and low-level representational dynamics, contributing to the cognitive interpretability of LLMs. Lastly, we provide a CLI and datasets for various cognitive behaviors at the project repository, this https URL. 

---
# ReliabilityBench: Evaluating LLM Agent Reliability Under Production-Like Stress Conditions 

**Authors**: Aayush Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2601.06112)  

**Abstract**: Existing benchmarks for tool-using LLM agents primarily report single-run success rates and miss reliability properties required in production. We introduce \textbf{ReliabilityBench}, a benchmark for evaluating agent reliability across three dimensions: (i) consistency under repeated execution using $\mathrm{pass}^k$, (ii) robustness to semantically equivalent task perturbations at intensity $\epsilon$, and (iii) fault tolerance under controlled tool/API failures at intensity $\lambda$. ReliabilityBench contributes a unified reliability surface $R(k,\epsilon,\lambda)$, \textit{action metamorphic relations} that define correctness via end-state equivalence rather than text similarity, and a chaos-engineering-style fault injection framework (timeouts, rate limits, partial responses, schema drift). We evaluate two models (Gemini 2.0 Flash, GPT-4o) and two agent architectures (ReAct, Reflexion) across four domains (scheduling, travel, customer support, e-commerce) over 1,280 episodes. Perturbations alone reduce success from 96.9% at $\epsilon=0$ to 88.1% at $\epsilon=0.2$. Rate limiting is the most damaging fault in ablations. ReAct is more robust than Reflexion under combined stress, and Gemini 2.0 Flash achieves comparable reliability to GPT-4o at much lower cost. ReliabilityBench provides a systematic framework for assessing production readiness of LLM agents. 

---
# LLM-Powered Social Digital Twins: A Framework for Simulating Population Behavioral Response to Policy Interventions 

**Authors**: Aayush Gupta, Farahan Raza Sheikh  

**Link**: [PDF](https://arxiv.org/pdf/2601.06111)  

**Abstract**: Predicting how populations respond to policy interventions is a fundamental challenge in computational social science and public policy. Traditional approaches rely on aggregate statistical models that capture historical correlations but lack mechanistic interpretability and struggle with novel policy scenarios. We present a general framework for constructing Social Digital Twins - virtual population replicas where Large Language Models (LLMs) serve as cognitive engines for individual agents. Each agent, characterized by demographic and psychographic attributes, receives policy signals and outputs multi-dimensional behavioral probability vectors. A calibration layer maps aggregated agent responses to observable population-level metrics, enabling validation against real-world data and deployment for counterfactual policy analysis.
We instantiate this framework in the domain of pandemic response, using COVID-19 as a case study with rich observational data. On a held-out test period, our calibrated digital twin achieves a 20.7% improvement in macro-averaged prediction error over gradient boosting baselines across six behavioral categories. Counterfactual experiments demonstrate monotonic and bounded responses to policy variations, establishing behavioral plausibility. The framework is domain-agnostic: the same architecture applies to transportation policy, economic interventions, environmental regulations, or any setting where policy affects population behavior. We discuss implications for policy simulation, limitations of the approach, and directions for extending LLM-based digital twins beyond pandemic response. 

---
# Automatic Question Generation for Intuitive Learning Utilizing Causal Graph Guided Chain of Thought Reasoning 

**Authors**: Nicholas X. Wang, Neel V. Parpia, Aaryan D. Parikh, Aggelos K. Katsaggelos  

**Link**: [PDF](https://arxiv.org/pdf/2601.06098)  

**Abstract**: Intuitive learning is crucial for developing deep conceptual understanding, especially in STEM education, where students often struggle with abstract and interconnected concepts. Automatic question generation has become an effective strategy for personalized and adaptive learning. However, its effectiveness is hindered by hallucinations in large language models (LLMs), which may generate factually incorrect, ambiguous, or pedagogically inconsistent questions. To address this issue, we propose a novel framework that combines causal-graph-guided Chain-of-Thought (CoT) reasoning with a multi-agent LLM architecture. This approach ensures the generation of accurate, meaningful, and curriculum-aligned questions. Causal graphs provide an explicit representation of domain knowledge, while CoT reasoning facilitates a structured, step-by-step traversal of related concepts. Dedicated LLM agents are assigned specific tasks such as graph pathfinding, reasoning, validation, and output, all working within domain constraints. A dual validation mechanism-at both the conceptual and output stages-greatly reduces hallucinations. Experimental results demonstrate up to a 70% improvement in quality compared to reference methods and yielded highly favorable outcomes in subjective evaluations. 

---
# GeoMotionGPT: Geometry-Aligned Motion Understanding with Large Language Models 

**Authors**: Zhankai Ye, Bofan Li, Yukai Jin, Shuoqiu Li, Wei Wang, Yanfu Zhang, Shangqian Gao, Xin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07632)  

**Abstract**: Discrete motion tokenization has recently enabled Large Language Models (LLMs) to serve as versatile backbones for motion understanding and motion-language reasoning. However, existing pipelines typically decouple motion quantization from semantic embedding learning, linking them solely via token IDs. This approach fails to effectively align the intrinsic geometry of the motion space with the embedding space, thereby hindering the LLM's capacity for nuanced motion reasoning. We argue that alignment is most effective when both modalities share a unified geometric basis. Therefore, instead of forcing the LLM to reconstruct the complex geometry among motion tokens from scratch, we present a novel framework that explicitly enforces orthogonality on both the motion codebook and the LLM embedding space, ensuring that their relational structures naturally mirror each other. Specifically, we employ a decoder-only quantizer with Gumbel-Softmax for differentiable training and balanced codebook usage. To bridge the modalities, we use a sparse projection that maps motion codes into the LLM embedding space while preserving orthogonality. Finally, a two-stage orthonormal regularization schedule enforces soft constraints during tokenizer training and LLM fine-tuning to maintain geometric alignment without hindering semantic adaptation. Extensive experiments on HumanML3D demonstrate that our framework achieves a 20% performance improvement over current state-of-the-art methods, validating that a unified geometric basis effectively empowers the LLM for nuanced motion reasoning. 

---
# d3LLM: Ultra-Fast Diffusion LLM using Pseudo-Trajectory Distillation 

**Authors**: Yu-Yang Qian, Junda Su, Lanxiang Hu, Peiyuan Zhang, Zhijie Deng, Peng Zhao, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07568)  

**Abstract**: Diffusion large language models (dLLMs) offer capabilities beyond those of autoregressive (AR) LLMs, such as parallel decoding and random-order generation. However, realizing these benefits in practice is non-trivial, as dLLMs inherently face an accuracy-parallelism trade-off. Despite increasing interest, existing methods typically focus on only one-side of the coin, targeting either efficiency or performance. To address this limitation, we propose d3LLM (Pseudo-Distilled Diffusion Large Language Model), striking a balance between accuracy and parallelism: (i) during training, we introduce pseudo-trajectory distillation to teach the model which tokens can be decoded confidently at early steps, thereby improving parallelism; (ii) during inference, we employ entropy-based multi-block decoding with a KV-cache refresh mechanism to achieve high parallelism while maintaining accuracy. To better evaluate dLLMs, we also introduce AUP (Accuracy Under Parallelism), a new metric that jointly measures accuracy and parallelism. Experiments demonstrate that our d3LLM achieves up to 10$\times$ speedup over vanilla LLaDA/Dream and 5$\times$ speedup over AR models without much accuracy drop. Our code is available at this https URL. 

---
# Large Language Models for Physics Instrument Design 

**Authors**: Sara Zoccheddu, Shah Rukh Qasim, Patrick Owen, Nicola Serra  

**Link**: [PDF](https://arxiv.org/pdf/2601.07580)  

**Abstract**: We study the use of large language models (LLMs) for physics instrument design and compare their performance to reinforcement learning (RL). Using only prompting, LLMs are given task constraints and summaries of prior high-scoring designs and propose complete detector configurations, which we evaluate with the same simulators and reward functions used in RL-based optimization. Although RL yields stronger final designs, we find that modern LLMs consistently generate valid, resource-aware, and physically meaningful configurations that draw on broad pretrained knowledge of detector design principles and particle--matter interactions, despite having no task-specific training. Based on this result, as a first step toward hybrid design workflows, we explore pairing the LLMs with a dedicated trust region optimizer, serving as a precursor to future pipelines in which LLMs propose and structure design hypotheses while RL performs reward-driven optimization. Based on these experiments, we argue that LLMs are well suited as meta-planners: they can design and orchestrate RL-based optimization studies, define search strategies, and coordinate multiple interacting components within a unified workflow. In doing so, they point toward automated, closed-loop instrument design in which much of the human effort required to structure and supervise optimization can be reduced. 

---
# Seeing Right but Saying Wrong: Inter- and Intra-Layer Refinement in MLLMs without Training 

**Authors**: Shezheng Song, Shasha Li, Jie Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07359)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated strong capabilities across a variety of vision-language tasks. However, their internal reasoning often exhibits a critical inconsistency: although deeper layers may attend to the correct visual regions, final predictions are frequently misled by noisy attention from earlier layers. This results in a disconnect between what the model internally understands and what it ultimately expresses, a phenomenon we describe as seeing it right but saying it wrong. To address this issue, we propose DualPD, a dual-perspective decoding refinement strategy that enhances the visual understanding without any additional training. DualPD consists of two components. (1) The layer-wise attention-guided contrastive logits module captures how the belief in the correct answer evolves by comparing output logits between layers that exhibit the largest attention shift. (2) The head-wise information filtering module suppresses low-contribution attention heads that focus on irrelevant regions, thereby improving attention quality within each layer. Experiments conducted on both the LLaVA and Qwen-VL model families across multiple multimodal benchmarks demonstrate that DualPD consistently improves accuracy without training, confirming its effectiveness and generalizability. The code will be released upon publication. 

---
# MCP-ITP: An Automated Framework for Implicit Tool Poisoning in MCP 

**Authors**: Ruiqi Li, Zhiqiang Wang, Yunhao Yao, Xiang-Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.07395)  

**Abstract**: To standardize interactions between LLM-based agents and their environments, the Model Context Protocol (MCP) was proposed and has since been widely adopted. However, integrating external tools expands the attack surface, exposing agents to tool poisoning attacks. In such attacks, malicious instructions embedded in tool metadata are injected into the agent context during MCP registration phase, thereby manipulating agent behavior. Prior work primarily focuses on explicit tool poisoning or relied on manually crafted poisoned tools. In contrast, we focus on a particularly stealthy variant: implicit tool poisoning, where the poisoned tool itself remains uninvoked. Instead, the instructions embedded in the tool metadata induce the agent to invoke a legitimate but high-privilege tool to perform malicious operations. We propose MCP-ITP, the first automated and adaptive framework for implicit tool poisoning within the MCP ecosystem. MCP-ITP formulates poisoned tool generation as a black-box optimization problem and employs an iterative optimization strategy that leverages feedback from both an evaluation LLM and a detection LLM to maximize Attack Success Rate (ASR) while evading current detection mechanisms. Experimental results on the MCPTox dataset across 12 LLM agents demonstrate that MCP-ITP consistently outperforms the manually crafted baseline, achieving up to 84.2% ASR while suppressing the Malicious Tool Detection Rate (MDR) to as low as 0.3%. 

---
# When Bots Take the Bait: Exposing and Mitigating the Emerging Social Engineering Attack in Web Automation Agent 

**Authors**: Xinyi Wu, Geng Hong, Yueyue Chen, MingXuan Liu, Feier Jin, Xudong Pan, Jiarun Dai, Baojun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07263)  

**Abstract**: Web agents, powered by large language models (LLMs), are increasingly deployed to automate complex web interactions. The rise of open-source frameworks (e.g., Browser Use, Skyvern-AI) has accelerated adoption, but also broadened the attack surface. While prior research has focused on model threats such as prompt injection and backdoors, the risks of social engineering remain largely unexplored. We present the first systematic study of social engineering attacks against web automation agents and design a pluggable runtime mitigation solution. On the attack side, we introduce the AgentBait paradigm, which exploits intrinsic weaknesses in agent execution: inducement contexts can distort the agent's reasoning and steer it toward malicious objectives misaligned with the intended task. On the defense side, we propose SUPERVISOR, a lightweight runtime module that enforces environment and intention consistency alignment between webpage context and intended goals to mitigate unsafe operations before execution.
Empirical results show that mainstream frameworks are highly vulnerable to AgentBait, with an average attack success rate of 67.5% and peaks above 80% under specific strategies (e.g., trusted identity forgery). Compared with existing lightweight defenses, our module can be seamlessly integrated across different web automation frameworks and reduces attack success rates by up to 78.1% on average while incurring only a 7.7% runtime overhead and preserving usability. This work reveals AgentBait as a critical new threat surface for web agents and establishes a practical, generalizable defense, advancing the security of this rapidly emerging ecosystem. We reported the details of this attack to the framework developers and received acknowledgment before submission. 

---
# Beyond Variance: Knowledge-Aware LLM Compression via Fisher-Aligned Subspace Diagnostics 

**Authors**: Ibne Farabi Shihab, Sanjeda Akter, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2601.07197)  

**Abstract**: Post-training activation compression is essential for deploying Large Language Models (LLMs) on resource-constrained hardware. However, standard methods like Singular Value Decomposition (SVD) are gradient-blind: they preserve high-variance dimensions regardless of their impact on factual knowledge preservation. We introduce Fisher-Aligned Subspace Compression (FASC), a knowledge-aware compression framework that selects subspaces by directly modeling activation-gradient coupling, minimizing a second-order surrogate of the loss function. FASC leverages the Fisher Information Matrix to identify dimensions critical for factual knowledge, which often reside in low-variance but high-gradient-sensitivity subspaces. We propose the Dependence Violation Score (\r{ho}) as a general-purpose diagnostic metric that quantifies activation-gradient coupling, revealing where factual knowledge is stored within transformer architectures. Extensive experiments on Mistral-7B and Llama-3-8B demonstrate that FASC preserves 6-8% more accuracy on knowledge-intensive benchmarks (MMLU, LAMA) compared to variance-based methods at 50% rank reduction, effectively enabling a 7B model to match the factual recall of a 13B uncompressed model. Our analysis reveals that \r{ho} serves as a fundamental signal of stored knowledge, with high-\r{ho} layers emerging only when models internalize factual associations during training. 

---
# Forward versus Backward: Comparing Reasoning Objectives in Direct Preference Optimization 

**Authors**: Murtaza Nikzad, Raghuram Ramanujan  

**Link**: [PDF](https://arxiv.org/pdf/2601.07199)  

**Abstract**: Large language models exhibit impressive reasoning capabilities yet frequently generate plausible but incorrect solutions, a phenomenon commonly termed hallucination. This paper investigates the effect of training objective composition on reasoning reliability through Direct Preference Optimization. Two complementary training signals are examined: forward chain-of-thought generation, which trains the model to produce correct reasoning traces, and backward verification, which trains the model to verify and acknowledge errors in candidate solutions. Experiments on GSM8K reveal a fundamental trade-off between these objectives. Forward-only DPO training achieves the highest accuracy improvement, increasing from 83.1% to 86.6% (+3.5 percentage points), while backward-only training yields minimal accuracy gains but substantially reduces the false positive rate from 13.4% to 4.3%. Notably, both training variants reduce acknowledgement rate compared to the baseline, suggesting that preference optimization increases model confidence in its outputs. These findings indicate that forward and backward reasoning objectives provide distinct and complementary learning signals: forward training improves problem-solving capability, while backward training improves verification calibration. The complete training and evaluation pipeline, implemented efficiently through Low-Rank Adaptation, is released to facilitate further research. 

---
# SIRR-LMM: Single-image Reflection Removal via Large Multimodal Model 

**Authors**: Yu Guo, Zhiqiang Lao, Xiyun Song, Yubin Zhou, Heather Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07209)  

**Abstract**: Glass surfaces create complex interactions of reflected and transmitted light, making single-image reflection removal (SIRR) challenging. Existing datasets suffer from limited physical realism in synthetic data or insufficient scale in real captures. We introduce a synthetic dataset generation framework that path-traces 3D glass models over real background imagery to create physically accurate reflection scenarios with varied glass properties, camera settings, and post-processing effects. To leverage the capabilities of Large Multimodal Model (LMM), we concatenate the image layers into a single composite input, apply joint captioning, and fine-tune the model using task-specific LoRA rather than full-parameter training. This enables our approach to achieve improved reflection removal and separation performance compared to state-of-the-art methods. 

---
# Defenses Against Prompt Attacks Learn Surface Heuristics 

**Authors**: Shawn Li, Chenxiao Yu, Zhiyu Ni, Hao Li, Charith Peris, Chaowei Xiao, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.07185)  

**Abstract**: Large language models (LLMs) are increasingly deployed in security-sensitive applications, where they must follow system- or developer-specified instructions that define the intended task behavior, while completing benign user requests. When adversarial instructions appear in user queries or externally retrieved content, models may override intended logic. Recent defenses rely on supervised fine-tuning with benign and malicious labels. Although these methods achieve high attack rejection rates, we find that they rely on narrow correlations in defense data rather than harmful intent, leading to systematic rejection of safe inputs. We analyze three recurring shortcut behaviors induced by defense fine-tuning. \emph{Position bias} arises when benign content placed later in a prompt is rejected at much higher rates; across reasoning benchmarks, suffix-task rejection rises from below \textbf{10\%} to as high as \textbf{90\%}. \emph{Token trigger bias} occurs when strings common in attack data raise rejection probability even in benign contexts; inserting a single trigger token increases false refusals by up to \textbf{50\%}. \emph{Topic generalization bias} reflects poor generalization beyond the defense data distribution, with defended models suffering test-time accuracy drops of up to \textbf{40\%}. These findings suggest that current prompt-injection defenses frequently respond to attack-like surface patterns rather than the underlying intent. We introduce controlled diagnostic datasets and a systematic evaluation across two base models and multiple defense pipelines, highlighting limitations of supervised fine-tuning for reliable LLM security. 

---
# Safeguarding LLM Fine-tuning via Push-Pull Distributional Alignment 

**Authors**: Haozhong Wang, Zhuo Li, Yibo Yang, He Zhao, Hongyuan Zha, Dandan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.07200)  

**Abstract**: The inherent safety alignment of Large Language Models (LLMs) is prone to erosion during fine-tuning, even when using seemingly innocuous datasets. While existing defenses attempt to mitigate this via data selection, they typically rely on heuristic, instance-level assessments that neglect the global geometry of the data distribution and fail to explicitly repel harmful patterns. To address this, we introduce Safety Optimal Transport (SOT), a novel framework that reframes safe fine-tuning from an instance-level filtering challenge to a distribution-level alignment task grounded in Optimal Transport (OT). At its core is a dual-reference ``push-pull'' weight-learning mechanism: SOT optimizes sample importance by actively pulling the downstream distribution towards a trusted safe anchor while simultaneously pushing it away from a general harmful reference. This establishes a robust geometric safety boundary that effectively purifies the training data. Extensive experiments across diverse model families and domains demonstrate that SOT significantly enhances model safety while maintaining competitive downstream performance, achieving a superior safety-utility trade-off compared to baselines. 

---
# Measuring Iterative Temporal Reasoning with TimePuzzles 

**Authors**: Zhengxiang Wang, Zeyu Dong  

**Link**: [PDF](https://arxiv.org/pdf/2601.07148)  

**Abstract**: We introduce TimePuzzles, a constraint-based date inference task for evaluating iterative temporal reasoning. Each puzzle combines factual temporal anchors with (cross-cultural) calendar relations, admits one or multiple valid solution dates, and is algorithmically generated for controlled, dynamic, and continual evaluation. Across 13 diverse LLMs, TimePuzzles well distinguishes their iterative temporal reasoning capabilities and remains challenging without tools: GPT-5 reaches only 49.3% accuracy and all other models stay below 31%, despite the dataset's simplicity. Web search consistently yields substantial gains and using code interpreter shows mixed effects, but all models perform much better when constraints are rewritten with explicit dates, revealing a gap in reliable tool use. Overall, TimePuzzles presents a simple, cost-effective diagnostic for tool-augmented iterative temporal reasoning. 

---
# Safe-FedLLM: Delving into the Safety of Federated Large Language Models 

**Authors**: Mingxiang Tao, Yu Tian, Wenxuan Tu, Yue Yang, Xue Yang, Xiangyan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07177)  

**Abstract**: Federated learning (FL) addresses data privacy and silo issues in large language models (LLMs). Most prior work focuses on improving the training efficiency of federated LLMs. However, security in open environments is overlooked, particularly defenses against malicious clients. To investigate the safety of LLMs during FL, we conduct preliminary experiments to analyze potential attack surfaces and defensible characteristics from the perspective of Low-Rank Adaptation (LoRA) weights. We find two key properties of FL: 1) LLMs are vulnerable to attacks from malicious clients in FL, and 2) LoRA weights exhibit distinct behavioral patterns that can be filtered through simple classifiers. Based on these properties, we propose Safe-FedLLM, a probe-based defense framework for federated LLMs, constructing defenses across three dimensions: Step-Level, Client-Level, and Shadow-Level. The core concept of Safe-FedLLM is to perform probe-based discrimination on the LoRA weights locally trained by each client during FL, treating them as high-dimensional behavioral features and using lightweight classification models to determine whether they possess malicious attributes. Extensive experiments demonstrate that Safe-FedLLM effectively enhances the defense capability of federated LLMs without compromising performance on benign data. Notably, our method effectively suppresses malicious data impact without significant impact on training speed, and remains effective even with many malicious clients. Our code is available at: this https URL. 

---
# The AI Cognitive Trojan Horse: How Large Language Models May Bypass Human Epistemic Vigilance 

**Authors**: Andrew D. Maynard  

**Link**: [PDF](https://arxiv.org/pdf/2601.07085)  

**Abstract**: Large language model (LLM)-based conversational AI systems present a challenge to human cognition that current frameworks for understanding misinformation and persuasion do not adequately address. This paper proposes that a significant epistemic risk from conversational AI may lie not in inaccuracy or intentional deception, but in something more fundamental: these systems may be configured, through optimization processes that make them useful, to present characteristics that bypass the cognitive mechanisms humans evolved to evaluate incoming information. The Cognitive Trojan Horse hypothesis draws on Sperber and colleagues' theory of epistemic vigilance -- the parallel cognitive process monitoring communicated information for reasons to doubt -- and proposes that LLM-based systems present 'honest non-signals': genuine characteristics (fluency, helpfulness, apparent disinterest) that fail to carry the information equivalent human characteristics would carry, because in humans these are costly to produce while in LLMs they are computationally trivial. Four mechanisms of potential bypass are identified: processing fluency decoupled from understanding, trust-competence presentation without corresponding stakes, cognitive offloading that delegates evaluation itself to the AI, and optimization dynamics that systematically produce sycophancy. The framework generates testable predictions, including a counterintuitive speculation that cognitively sophisticated users may be more vulnerable to AI-mediated epistemic influence. This reframes AI safety as partly a problem of calibration -- aligning human evaluative responses with the actual epistemic status of AI-generated content -- rather than solely a problem of preventing deception. 

---
# Enhancing Cloud Network Resilience via a Robust LLM-Empowered Multi-Agent Reinforcement Learning Framework 

**Authors**: Yixiao Peng, Hao Hu, Feiyang Li, Xinye Cao, Yingchang Jiang, Jipeng Tang, Guoshun Nan, Yuling Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07122)  

**Abstract**: While virtualization and resource pooling empower cloud networks with structural flexibility and elastic scalability, they inevitably expand the attack surface and challenge cyber resilience. Reinforcement Learning (RL)-based defense strategies have been developed to optimize resource deployment and isolation policies under adversarial conditions, aiming to enhance system resilience by maintaining and restoring network availability. However, existing approaches lack robustness as they require retraining to adapt to dynamic changes in network structure, node scale, attack strategies, and attack intensity. Furthermore, the lack of Human-in-the-Loop (HITL) support limits interpretability and flexibility. To address these limitations, we propose CyberOps-Bots, a hierarchical multi-agent reinforcement learning framework empowered by Large Language Models (LLMs). Inspired by MITRE ATT&CK's Tactics-Techniques model, CyberOps-Bots features a two-layer architecture: (1) An upper-level LLM agent with four modules--ReAct planning, IPDRR-based perception, long-short term memory, and action/tool integration--performs global awareness, human intent recognition, and tactical planning; (2) Lower-level RL agents, developed via heterogeneous separated pre-training, execute atomic defense actions within localized network regions. This synergy preserves LLM adaptability and interpretability while ensuring reliable RL execution. Experiments on real cloud datasets show that, compared to state-of-the-art algorithms, CyberOps-Bots maintains network availability 68.5% higher and achieves a 34.7% jumpstart performance gain when shifting the scenarios without retraining. To our knowledge, this is the first study to establish a robust LLM-RL framework with HITL support for cloud defense. We will release our framework to the community, facilitating the advancement of robust and autonomous defense in cloud networks. 

---
# Hallucinations Live in Variance 

**Authors**: Aaron R. Flouro, Shawn P. Chadwick  

**Link**: [PDF](https://arxiv.org/pdf/2601.07058)  

**Abstract**: Benchmarks measure whether a model is correct. They do not measure whether a model is reliable. This distinction is largely academic for single-shot inference, but becomes critical for agentic AI systems, where a single rephrased prompt can trigger cascading failures in multi-step execution. Yet this form of instability is not captured by existing evaluations.
Hallucinations live in variance: they arise when semantically equivalent prompts activate inconsistent internal pathways, producing divergent outputs. Consistent but incorrect outputs reflect bias or missing knowledge; confident guessing reflects calibration failure. Neither constitutes hallucination under this definition. When error is variance-dominated, reducing redundant pathways improves reliability without adding knowledge. We formalize this through Semantic Stability (SS), measured via Paraphrase Consistency (PC@k): generate k paraphrases, greedy decode each, compute mode agreement. SS is a diagnostic for variance-driven unreliability, not a method for improving correctness.
We show that a dense Qwen3-0.6B agrees with itself only 23.8% of the time; at 32% sparsity, agreement jumps to 55.9%. A phase diagram reveals the sweet spot where variance reduction outpaces bias accumulation, and regimes where stability collapses onto wrong answers. 

---
# Overcoming the Retrieval Barrier: Indirect Prompt Injection in the Wild for LLM Systems 

**Authors**: Hongyan Chang, Ergute Bao, Xinjian Luo, Ting Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07072)  

**Abstract**: Large language models (LLMs) increasingly rely on retrieving information from external corpora. This creates a new attack surface: indirect prompt injection (IPI), where hidden instructions are planted in the corpora and hijack model behavior once retrieved. Previous studies have highlighted this risk but often avoid the hardest step: ensuring that malicious content is actually retrieved. In practice, unoptimized IPI is rarely retrieved under natural queries, which leaves its real-world impact unclear.
We address this challenge by decomposing the malicious content into a trigger fragment that guarantees retrieval and an attack fragment that encodes arbitrary attack objectives. Based on this idea, we design an efficient and effective black-box attack algorithm that constructs a compact trigger fragment to guarantee retrieval for any attack fragment. Our attack requires only API access to embedding models, is cost-efficient (as little as $0.21 per target user query on OpenAI's embedding models), and achieves near-100% retrieval across 11 benchmarks and 8 embedding models (including both open-source models and proprietary services).
Based on this attack, we present the first end-to-end IPI exploits under natural queries and realistic external corpora, spanning both RAG and agentic systems with diverse attack objectives. These results establish IPI as a practical and severe threat: when a user issued a natural query to summarize emails on frequently asked topics, a single poisoned email was sufficient to coerce GPT-4o into exfiltrating SSH keys with over 80% success in a multi-agent workflow. We further evaluate several defenses and find that they are insufficient to prevent the retrieval of malicious text, highlighting retrieval as a critical open vulnerability. 

---
# Zer0n: An AI-Assisted Vulnerability Discovery and Blockchain-Backed Integrity Framework 

**Authors**: Harshil Parmar, Pushti Vyas, Prayers Khristi, Priyank Panchal  

**Link**: [PDF](https://arxiv.org/pdf/2601.07019)  

**Abstract**: As vulnerability research increasingly adopts generative AI, a critical reliance on opaque model outputs has emerged, creating a "trust gap" in security automation. We address this by introducing Zer0n, a framework that anchors the reasoning capabilities of Large Language Models (LLMs) to the immutable audit trails of blockchain technology. Specifically, we integrate Gemini 2.0 Pro for logic-based vulnerability detection with the Avalanche C-Chain for tamper-evident artifact logging. Unlike fully decentralized solutions that suffer from high latency, Zer0n employs a hybrid architecture: execution remains off-chain for performance, while integrity proofs are finalized on-chain. Our evaluation on a dataset of 500 endpoints reveals that this approach achieves 80% detection accuracy with only a marginal 22.9% overhead, effectively demonstrating that decentralized integrity can coexist with high-speed security workflows. 

---
# MicLog: Towards Accurate and Efficient LLM-based Log Parsing via Progressive Meta In-Context Learning 

**Authors**: Jianbo Yu, Yixuan Li, Hai Xu, Kang Xu, Junjielong Xu, Zhijing Li, Pinjia He, Wanyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.07005)  

**Abstract**: Log parsing converts semi-structured logs into structured templates, forming a critical foundation for downstream analysis. Traditional syntax and semantic-based parsers often struggle with semantic variations in evolving logs and data scarcity stemming from their limited domain coverage. Recent large language model (LLM)-based parsers leverage in-context learning (ICL) to extract semantics from examples, demonstrating superior accuracy. However, LLM-based parsers face two main challenges: 1) underutilization of ICL capabilities, particularly in dynamic example selection and cross-domain generalization, leading to inconsistent performance; 2) time-consuming and costly LLM querying. To address these challenges, we present MicLog, the first progressive meta in-context learning (ProgMeta-ICL) log parsing framework that combines meta-learning with ICL on small open-source LLMs (i.e., Qwen-2.5-3B). Specifically, MicLog: i) enhances LLMs' ICL capability through a zero-shot to k-shot ProgMeta-ICL paradigm, employing weighted DBSCAN candidate sampling and enhanced BM25 demonstration selection; ii) accelerates parsing via a multi-level pre-query cache that dynamically matches and refines recently parsed templates. Evaluated on Loghub-2.0, MicLog achieves 10.3% higher parsing accuracy than the state-of-the-art parser while reducing parsing time by 42.4%. 

---
# SketchJudge: A Diagnostic Benchmark for Grading Hand-drawn Diagrams with Multimodal Large Language Models 

**Authors**: Yuhang Su, Mei Wang, Yaoyao Zhong, Guozhang Li, Shixing Li, Yihan Feng, Hua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06944)  

**Abstract**: While Multimodal Large Language Models (MLLMs) have achieved remarkable progress in visual understanding, they often struggle when faced with the unstructured and ambiguous nature of human-generated sketches. This limitation is particularly pronounced in the underexplored task of visual grading, where models should not only solve a problem but also diagnose errors in hand-drawn diagrams. Such diagnostic capabilities depend on complex structural, semantic, and metacognitive reasoning. To bridge this gap, we introduce SketchJudge, a novel benchmark tailored for evaluating MLLMs as graders of hand-drawn STEM diagrams. SketchJudge encompasses 1,015 hand-drawn student responses across four domains: geometry, physics, charts, and flowcharts, featuring diverse stylistic variations and distinct error types. Evaluations on SketchJudge demonstrate that even advanced MLLMs lag significantly behind humans, validating the benchmark's effectiveness in exposing the fragility of current vision-language alignment in symbolic and noisy contexts. All data, code, and evaluation scripts are publicly available at this https URL. 

---
# VISTA: Knowledge-Driven Interpretable Vessel Trajectory Imputation via Large Language Models 

**Authors**: Hengyu Liu, Tianyi Li, Haoyu Wang, Kristian Torp, Tiancheng Zhang, Yushuai Li, Christian S. Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2601.06940)  

**Abstract**: The Automatic Identification System provides critical information for maritime navigation and safety, yet its trajectories are often incomplete due to signal loss or deliberate tampering. Existing imputation methods emphasize trajectory recovery, paying limited attention to interpretability and failing to provide underlying knowledge that benefits downstream tasks such as anomaly detection and route planning. We propose knowledge-driven interpretable vessel trajectory imputation (VISTA), the first trajectory imputation framework that offers interpretability while simultaneously providing underlying knowledge to support downstream analysis. Specifically, we first define underlying knowledge as a combination of Structured Data-derived Knowledge (SDK) distilled from AIS data and Implicit LLM Knowledge acquired from large-scale Internet corpora. Second, to manage and leverage the SDK effectively at scale, we develop a data-knowledge-data loop that employs a Structured Data-derived Knowledge Graph for SDK extraction and knowledge-driven trajectory imputation. Third, to efficiently process large-scale AIS data, we introduce a workflow management layer that coordinates the end-to-end pipeline, enabling parallel knowledge extraction and trajectory imputation with anomaly handling and redundancy elimination. Experiments on two large AIS datasets show that VISTA is capable of state-of-the-art imputation accuracy and computational efficiency, improving over state-of-the-art baselines by 5%-94% and reducing time cost by 51%-93%, while producing interpretable knowledge cues that benefit downstream tasks. The source code and implementation details of VISTA are publicly available. 

---
# Towards Compositional Generalization in LLMs for Smart Contract Security: A Case Study on Reentrancy Vulnerabilities 

**Authors**: Ying Zhou, Jiacheng Wei, Yu Qi, Faguo Wu, Xiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06914)  

**Abstract**: Large language models (LLMs) demonstrate remarkable capabilities in natural language understanding and generation. Despite being trained on large-scale, high-quality data, LLMs still fail to outperform traditional static analysis tools in specialized domains like smart contract vulnerability detection. To address this issue, this paper proposes a post-training algorithm based on atomic task decomposition and fusion. This algorithm aims to achieve combinatorial generalization under limited data by decomposing complex reasoning tasks. Specifically, we decompose the reentrancy vulnerability detection task into four linearly independent atomic tasks: identifying external calls, identifying state updates, identifying data dependencies between external calls and state updates, and determining their data flow order. These tasks form the core components of our approach. By training on synthetic datasets, we generate three compiler-verified datasets. We then employ the Slither tool to extract structural information from the control flow graph and data flow graph, which is used to fine-tune the LLM's adapter. Experimental results demonstrate that low-rank normalization fusion with the LoRA adapter improves the LLM's reentrancy vulnerability detection accuracy to 98.2%, surpassing state-of-the-art methods. On 31 real-world contracts, the algorithm achieves a 20% higher recall than traditional analysis tools. 

---
# DaQ-MSA: Denoising and Qualifying Diffusion Augmentations for Multimodal Sentiment Analysis 

**Authors**: Jiazhang Liang, Jianheng Dai, Miaosen Luo, Menghua Jiang, Sijie Mai  

**Link**: [PDF](https://arxiv.org/pdf/2601.06870)  

**Abstract**: Multimodal large language models (MLLMs) have demonstrated strong performance on vision-language tasks, yet their effectiveness on multimodal sentiment analysis remains constrained by the scarcity of high-quality training data, which limits accurate multimodal understanding and generalization. To alleviate this bottleneck, we leverage diffusion models to perform semantics-preserving augmentation on the video and audio modalities, expanding the multimodal training distribution. However, increasing data quantity alone is insufficient, as diffusion-generated samples exhibit substantial quality variation and noisy augmentations may degrade performance. We therefore propose DaQ-MSA (Denoising and Qualifying Diffusion Augmentations for Multimodal Sentiment Analysis), which introduces a quality scoring module to evaluate the reliability of augmented samples and assign adaptive training weights. By down-weighting low-quality samples and emphasizing high-fidelity ones, DaQ-MSA enables more stable learning. By integrating the generative capability of diffusion models with the semantic understanding of MLLMs, our approach provides a robust and generalizable automated augmentation strategy for training MLLMs without any human annotation or additional supervision. 

---
# Artificial Entanglement in the Fine-Tuning of Large Language Models 

**Authors**: Min Chen, Zihan Wang, Canyu Chen, Zeguan Wu, Manling Li, Junyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.06788)  

**Abstract**: Large language models (LLMs) can be adapted to new tasks using parameter-efficient fine-tuning (PEFT) methods that modify only a small number of trainable parameters, often through low-rank updates. In this work, we adopt a quantum-information-inspired perspective to understand their effectiveness. From this perspective, low-rank parameterizations naturally correspond to low-dimensional Matrix Product States (MPS) representations, which enable entanglement-based characterizations of parameter structure. Thereby, we term and measure "Artificial Entanglement", defined as the entanglement entropy of the parameters in artificial neural networks (in particular the LLMs). We first study the representative low-rank adaptation (LoRA) PEFT method, alongside full fine-tuning (FFT), using LLaMA models at the 1B and 8B scales trained on the Tulu3 and OpenThoughts3 datasets, and uncover: (i) Internal artificial entanglement in the updates of query and value projection matrices in LoRA follows a volume law with a central suppression (termed as the "Entanglement Valley"), which is sensitive to hyper-parameters and is distinct from that in FFT; (ii) External artificial entanglement in attention matrices, corresponding to token-token correlations in representation space, follows an area law with logarithmic corrections and remains robust to LoRA hyper-parameters and training steps. Drawing a parallel to the No-Hair Theorem in black hole physics, we propose that although LoRA and FFT induce distinct internal entanglement signatures, such differences do not manifest in the attention outputs, suggesting a "no-hair" property that results in the effectiveness of low rank updates. We further provide theoretical support based on random matrix theory, and extend our analysis to an MPS Adaptation PEFT method, which exhibits qualitatively similar behaviors. 

---
# AutoTour: Automatic Photo Tour Guide with Smartphones and LLMs 

**Authors**: Huatao Xu, Zihe Liu, Zilin Zeng, Baichuan Li, Mo Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.06781)  

**Abstract**: We present AutoTour, a system that enhances user exploration by automatically generating fine-grained landmark annotations and descriptive narratives for photos captured by users. The key idea of AutoTour is to fuse visual features extracted from photos with nearby geospatial features queried from open matching databases. Unlike existing tour applications that rely on pre-defined content or proprietary datasets, AutoTour leverages open and extensible data sources to provide scalable and context-aware photo-based guidance. To achieve this, we design a training-free pipeline that first extracts and filters relevant geospatial features around the user's GPS location. It then detects major landmarks in user photos through VLM-based feature detection and projects them into the horizontal spatial plane. A geometric matching algorithm aligns photo features with corresponding geospatial entities based on their estimated distance and direction. The matched features are subsequently grounded and annotated directly on the original photo, accompanied by large language model-generated textual and audio descriptions to provide an informative, tour-like experience. We demonstrate that AutoTour can deliver rich, interpretable annotations for both iconic and lesser-known landmarks, enabling a new form of interactive, context-aware exploration that bridges visual perception and geospatial understanding. 

---
# CEDAR: Context Engineering for Agentic Data Science 

**Authors**: Rishiraj Saha Roy, Chris Hinze, Luzian Hahn, Fabian Kuech  

**Link**: [PDF](https://arxiv.org/pdf/2601.06606)  

**Abstract**: We demonstrate CEDAR, an application for automating data science (DS) tasks with an agentic setup. Solving DS problems with LLMs is an underexplored area that has immense market value. The challenges are manifold: task complexities, data sizes, computational limitations, and context restrictions. We show that these can be alleviated via effective context engineering. We first impose structure into the initial prompt with DS-specific input fields, that serve as instructions for the agentic system. The solution is then materialized as an enumerated sequence of interleaved plan and code blocks generated by separate LLM agents, providing a readable structure to the context at any step of the workflow. Function calls for generating these intermediate texts, and for corresponding Python code, ensure that data stays local, and only aggregate statistics and associated instructions are injected into LLM prompts. Fault tolerance and context management are introduced via iterative code generation and smart history rendering. The viability of our agentic data scientist is demonstrated using canonical Kaggle challenges. 

---
# Are LLMs Vulnerable to Preference-Undermining Attacks (PUA)? A Factorial Analysis Methodology for Diagnosing the Trade-off between Preference Alignment and Real-World Validity 

**Authors**: Hongjun An, Yiliang Song, Jiangan Chen, Jiawei Shao, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.06596)  

**Abstract**: Large Language Model (LLM) training often optimizes for preference alignment, rewarding outputs that are perceived as helpful and interaction-friendly. However, this preference-oriented objective can be exploited: manipulative prompts can steer responses toward user-appeasing agreement and away from truth-oriented correction. In this work, we investigate whether aligned models are vulnerable to Preference-Undermining Attacks (PUA), a class of manipulative prompting strategies designed to exploit the model's desire to please user preferences at the expense of truthfulness. We propose a diagnostic methodology that provides a finer-grained and more directive analysis than aggregate benchmark scores, using a factorial evaluation framework to decompose prompt-induced shifts into interpretable effects of system objectives (truth- vs. preference-oriented) and PUA-style dialogue factors (directive control, personal derogation, conditional approval, reality denial) within a controlled $2 \times 2^4$ design. Surprisingly, more advanced models are sometimes more susceptible to manipulative prompts. Beyond the dominant reality-denial factor, we observe model-specific sign reversals and interactions with PUA-style factors, suggesting tailored defenses rather than uniform robustness. These findings offer a novel, reproducible factorial evaluation methodology that provides finer-grained diagnostics for post-training processes like RLHF, enabling better trade-offs in the product iteration of LLMs by offering a more nuanced understanding of preference alignment risks and the impact of manipulative prompts. 

---
# Coding in a Bubble? Evaluating LLMs in Resolving Context Adaptation Bugs During Code Adaptation 

**Authors**: Tanghaoran Zhang, Xinjun Mao, Shangwen Wang, Yuxin Zhao, Yao Lu, Zezhou Tang, Wenyu Xu, Longfei Sun, Changrong Xie, Kang Yang, Yue Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.06497)  

**Abstract**: Code adaptation is a fundamental but challenging task in software development, requiring developers to modify existing code for new contexts. A key challenge is to resolve Context Adaptation Bugs (CtxBugs), which occurs when code correct in its original context violates constraints in the target environment. Unlike isolated bugs, CtxBugs cannot be resolved through local fixes and require cross-context reasoning to identify semantic mismatches. Overlooking them may lead to critical failures in adaptation. Although Large Language Models (LLMs) show great potential in automating code-related tasks, their ability to resolve CtxBugs remains a significant and unexplored obstacle to their practical use in code adaptation. To bridge this gap, we propose CtxBugGen, a novel framework for generating CtxBugs to evaluate LLMs. Its core idea is to leverage LLMs' tendency to generate plausible but context-free code when contextual constraints are absent. The framework generates CtxBugs through a four-step process to ensure their relevance and validity: (1) Adaptation Task Selection, (2) Task-specific Perturbation,(3) LLM-based Variant Generation and (4) CtxBugs Identification. Based on the benchmark constructed by CtxBugGen, we conduct an empirical study with four state-of-the-art LLMs. Our results reveal their unsatisfactory performance in CtxBug resolution. The best performing LLM, Kimi-K2, achieves 55.93% on Pass@1 and resolves just 52.47% of CtxBugs. The presence of CtxBugs degrades LLMs' adaptation performance by up to 30%. Failure analysis indicates that LLMs often overlook CtxBugs and replicate them in their outputs. Our study highlights a critical weakness in LLMs' cross-context reasoning and emphasize the need for new methods to enhance their context awareness for reliable code adaptation. 

---
# PRISP: Privacy-Safe Few-Shot Personalization via Lightweight Adaptation 

**Authors**: Junho Park, Dohoon Kim, Taesup Moon  

**Link**: [PDF](https://arxiv.org/pdf/2601.06471)  

**Abstract**: Large language model (LLM) personalization aims to adapt general-purpose models to individual users. Most existing methods, however, are developed under data-rich and resource-abundant settings, often incurring privacy risks. In contrast, realistic personalization typically occurs after deployment under (i) extremely limited user data, (ii) constrained computational resources, and (iii) strict privacy requirements. We propose PRISP, a lightweight and privacy-safe personalization framework tailored to these constraints. PRISP leverages a Text-to-LoRA hypernetwork to generate task-aware LoRA parameters from task descriptions, and enables efficient user personalization by optimizing a small subset of task-aware LoRA parameters together with minimal additional modules using few-shot user data. Experiments on a few-shot variant of the LaMP benchmark demonstrate that PRISP achieves strong overall performance compared to prior approaches, while reducing computational overhead and eliminating privacy risks. 

---
# LLMTrack: Semantic Multi-Object Tracking with Multi-modal Large Language Models 

**Authors**: Pan Liao, Feng Yang, Di Wu, Jinwen Yu, Yuhua Zhu, Wenhui Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.06550)  

**Abstract**: Traditional Multi-Object Tracking (MOT) systems have achieved remarkable precision in localization and association, effectively answering \textit{where} and \textit{who}. However, they often function as autistic observers, capable of tracing geometric paths but blind to the semantic \textit{what} and \textit{why} behind object behaviors. To bridge the gap between geometric perception and cognitive reasoning, we propose \textbf{LLMTrack}, a novel end-to-end framework for Semantic Multi-Object Tracking (SMOT). We adopt a bionic design philosophy that decouples strong localization from deep understanding, utilizing Grounding DINO as the eyes and the LLaVA-OneVision multimodal large model as the brain. We introduce a Spatio-Temporal Fusion Module that aggregates instance-level interaction features and video-level contexts, enabling the Large Language Model (LLM) to comprehend complex trajectories. Furthermore, we design a progressive three-stage training strategy, Visual Alignment, Temporal Fine-tuning, and Semantic Injection via LoRA to efficiently adapt the massive model to the tracking domain. Extensive experiments on the BenSMOT benchmark demonstrate that LLMTrack achieves state-of-the-art performance, significantly outperforming existing methods in instance description, interaction recognition, and video summarization while maintaining robust tracking stability. 

---
# Context Matters: Peer-Aware Student Behavioral Engagement Measurement via VLM Action Parsing and LLM Sequence Classification 

**Authors**: Ahmed Abdelkawy, Ahmed Elsayed, Asem Ali, Aly Farag, Thomas Tretter, Michael McIntyre  

**Link**: [PDF](https://arxiv.org/pdf/2601.06394)  

**Abstract**: Understanding student behavior in the classroom is essential to improve both pedagogical quality and student engagement. Existing methods for predicting student engagement typically require substantial annotated data to model the diversity of student behaviors, yet privacy concerns often restrict researchers to their own proprietary datasets. Moreover, the classroom context, represented in peers' actions, is ignored. To address the aforementioned limitation, we propose a novel three-stage framework for video-based student engagement measurement. First, we explore the few-shot adaptation of the vision-language model for student action recognition, which is fine-tuned to distinguish among action categories with a few training samples. Second, to handle continuous and unpredictable student actions, we utilize the sliding temporal window technique to divide each student's 2-minute-long video into non-overlapping segments. Each segment is assigned an action category via the fine-tuned VLM model, generating a sequence of action predictions. Finally, we leverage the large language model to classify this entire sequence of actions, together with the classroom context, as belonging to an engaged or disengaged student. The experimental results demonstrate the effectiveness of the proposed approach in identifying student engagement. 

---
# Smart Privacy Policy Assistant: An LLM-Powered System for Transparent and Actionable Privacy Notices 

**Authors**: Sriharshini Kalvakuntla, Luoxi Tang, Yuqiao Meng, Zhaohan Xi  

**Link**: [PDF](https://arxiv.org/pdf/2601.06357)  

**Abstract**: Most users agree to online privacy policies without reading or understanding them, even though these documents govern how personal data is collected, shared, and monetized. Privacy policies are typically long, legally complex, and difficult for non-experts to interpret. This paper presents the Smart Privacy Policy Assistant, an LLM-powered system that automatically ingests privacy policies, extracts and categorizes key clauses, assigns human-interpretable risk levels, and generates clear, concise explanations. The system is designed for real-time use through browser extensions or mobile interfaces, surfacing contextual warnings before users disclose sensitive information or grant risky permissions. We describe the end-to-end pipeline, including policy ingestion, clause categorization, risk scoring, and explanation generation, and propose an evaluation framework based on clause-level accuracy, policy-level risk agreement, and user comprehension. 

---
# Evaluating Robustness of Large Language Models in Enterprise Applications: Benchmarks for Perturbation Consistency Across Formats and Languages 

**Authors**: Tara Bogavelli, Oluwanifemi Bamgbose, Gabrielle Gauthier Melançon, Fanny Riols, Roshnee Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2601.06341)  

**Abstract**: Enterprise LLM applications require consistently high quality and reliable performance across diverse scenarios, demanding robustness to minor variations. Existing research shows that even small prompt changes can lead to substantial differences in output, but has mainly focused on a narrow set of perturbations with small academic datasets, limiting their relevance to real-world applications. To address this, we present a comprehensive benchmark suite that evaluates robustness across multiple perturbation types, including general text edits (e.g., punctuation, whitespace), formatting changes (e.g., JSON, YAML), multilingual and cross-lingual inputs, and positional variations in instructions. Evaluating 11 models ranging from 4B to 120B+ parameters, we find that minor perturbations reduce performance by up to 40 percentage points on key enterprise metrics. Critically, we demonstrate that the relationship between model size and robustness is more nuanced than conventional assumptions suggest: an 8B parameter model (Ministral 3 8B) outperforms most larger models, while another 8B model (Llama 3.1 8B) performs worst overall. 

---
# AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving 

**Authors**: Tianhao Xu, Yiming Liu, Xianglong Lu, Yijia Zhao, Xuting Zhou, Aichen Feng, Yiyi Chen, Yi Shen, Qin Zhou, Xumeng Chen, Ilya Sherstyuk, Haorui Li, Rishi Thakkar, Ben Hamm, Yuanzhe Li, Xue Huang, Wenpeng Wu, Anish Shanbhag, Harry Kim, Chuan Chen, Junjie Lai  

**Link**: [PDF](https://arxiv.org/pdf/2601.06288)  

**Abstract**: Optimizing Large Language Model (LLM) inference in production systems is increasingly difficult due to dynamic workloads, stringent latency/throughput targets, and a rapidly expanding configuration space. This complexity spans not only distributed parallelism strategies (tensor/pipeline/expert) but also intricate framework-specific runtime parameters such as those concerning the enablement of CUDA graphs, available KV-cache memory fractions, and maximum token capacity, which drastically impact performance. The diversity of modern inference frameworks (e.g., TRT-LLM, vLLM, SGLang), each employing distinct kernels and execution policies, makes manual tuning both framework-specific and computationally prohibitive. We present AIConfigurator, a unified performance-modeling system that enables rapid, framework-agnostic inference configuration search without requiring GPU-based profiling. AIConfigurator combines (1) a methodology that decomposes inference into analytically modelable primitives - GEMM, attention, communication, and memory operations while capturing framework-specific scheduling dynamics; (2) a calibrated kernel-level performance database for these primitives across a wide range of hardware platforms and popular open-weights models (GPT-OSS, Qwen, DeepSeek, LLama, Mistral); and (3) an abstraction layer that automatically resolves optimal launch parameters for the target backend, seamlessly integrating into production-grade orchestration systems. Evaluation on production LLM serving workloads demonstrates that AIConfigurator identifies superior serving configurations that improve performance by up to 40% for dense models (e.g., Qwen3-32B) and 50% for MoE architectures (e.g., DeepSeek-V3), while completing searches within 30 seconds on average. Enabling the rapid exploration of vast design spaces - from cluster topology down to engine specific flags. 

---
# Projecting Out the Malice: A Global Subspace Approach to LLM Detoxification 

**Authors**: Zenghao Duan, Zhiyi Yin, Zhichao Shi, Liang Pang, Shaoling Jing, Zihe Huang, Jiayi Wu, Yu Yan, Jingcheng Deng, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.06226)  

**Abstract**: Large language models (LLMs) exhibit exceptional performance but pose inherent risks of generating toxic content, restricting their safe deployment. While traditional methods (e.g., alignment) adjust output preferences, they fail to eliminate underlying toxic regions in parameters, leaving models vulnerable to adversarial attacks. Prior mechanistic studies characterize toxic regions as "toxic vectors" or "layer-wise subspaces", yet our analysis identifies critical limitations: i) Removed toxic vectors can be reconstructed via linear combinations of non-toxic vectors, demanding targeting of entire toxic subspace; ii) Contrastive objective over limited samples inject noise into layer-wise subspaces, hindering stable extraction. These highlight the challenge of identifying robust toxic subspace and removing them. Therefore, we propose GLOSS (GLobal tOxic Subspace Suppression), a lightweight method that mitigates toxicity by identifying and eliminating this global subspace from FFN parameters. Experiments on LLMs (e.g., Qwen3) show GLOSS achieves SOTA detoxification while preserving general capabilities without requiring large-scale retraining. WARNING: This paper contains context which is toxic in nature. 

---
# QCaption: Video Captioning and Q&A through Fusion of Large Multimodal Models 

**Authors**: Jiale Wang, Gee Wah Ng, Lee Onn Mak, Randall Cher, Ng Ding Hei Ryan, Davis Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06566)  

**Abstract**: This paper introduces QCaption, a novel video captioning and Q&A pipeline that enhances video analytics by fusing three models: key frame extraction, a Large Multimodal Model (LMM) for image-text analysis, and a Large Language Model (LLM) for text analysis. This approach enables integrated analysis of text, images, and video, achieving performance improvements over existing video captioning and Q&A models; all while remaining fully self-contained, adept for on-premises deployment. Experimental results using QCaption demonstrated up to 44.2% and 48.9% improvements in video captioning and Q&A tasks, respectively. Ablation studies were also performed to assess the role of LLM on the fusion on the results. Moreover, the paper proposes and evaluates additional video captioning approaches, benchmarking them against QCaption and existing methodologies. QCaption demonstrate the potential of adopting a model fusion approach in advancing video analytics. 

---
# An Intelligent AI glasses System with Multi-Agent Architecture for Real-Time Voice Processing and Task Execution 

**Authors**: Sheng-Kai Chen, Jyh-Horng Wu, Ching-Yao Lin, Yen-Ting Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.06235)  

**Abstract**: This paper presents an AI glasses system that integrates real-time voice processing, artificial intelligence(AI) agents, and cross-network streaming capabilities. The system employs dual-agent architecture where Agent 01 handles Automatic Speech Recognition (ASR) and Agent 02 manages AI processing through local Large Language Models (LLMs), Model Context Protocol (MCP) tools, and Retrieval-Augmented Generation (RAG). The system supports real-time RTSP streaming for voice and video data transmission, eye tracking data collection, and remote task execution through RabbitMQ messaging. Implementation demonstrates successful voice command processing with multilingual support and cross-platform task execution capabilities. 

---
# LLM Agents in Law: Taxonomy, Applications, and Challenges 

**Authors**: Shuang Liu, Ruijia Zhang, Ruoyun Ma, Yujia Deng, Lanyi Zhu, Jiayu Li, Zelong Li, Zhibin Shen, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2601.06216)  

**Abstract**: Large language models (LLMs) have precipitated a dramatic improvement in the legal domain, yet the deployment of standalone models faces significant limitations regarding hallucination, outdated information, and verifiability. Recently, LLM agents have attracted significant attention as a solution to these challenges, utilizing advanced capabilities such as planning, memory, and tool usage to meet the rigorous standards of legal practice. In this paper, we present a comprehensive survey of LLM agents for legal tasks, analyzing how these architectures bridge the gap between technical capabilities and domain-specific needs. Our major contributions include: (1) systematically analyzing the technical transition from standard legal LLMs to legal agents; (2) presenting a structured taxonomy of current agent applications across distinct legal practice areas; (3) discussing evaluation methodologies specifically for agentic performance in law; and (4) identifying open challenges and outlining future directions for developing robust and autonomous legal assistants. 

---
# Breaking Model Lock-in: Cost-Efficient Zero-Shot LLM Routing via a Universal Latent Space 

**Authors**: Cheng Yan, Wuyang Zhang, Zhiyuan Ning, Fan Xu, Ziyang Tao, Lu Zhang, Bing Yin, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06220)  

**Abstract**: The rapid proliferation of Large Language Models (LLMs) has led to a fragmented and inefficient ecosystem, a state of ``model lock-in'' where seamlessly integrating novel models remains a significant bottleneck. Current routing frameworks require exhaustive, costly retraining, hindering scalability and adaptability. We introduce ZeroRouter, a new paradigm for LLM routing that breaks this lock-in. Our approach is founded on a universal latent space, a model-agnostic representation of query difficulty that fundamentally decouples the characterization of a query from the profiling of a model. This allows for zero-shot onboarding of new models without full-scale retraining. ZeroRouter features a context-aware predictor that maps queries to this universal space and a dual-mode optimizer that balances accuracy, cost, and latency. Our framework consistently outperforms all baselines, delivering higher accuracy at lower cost and latency. 

---
# Political Alignment in Large Language Models: A Multidimensional Audit of Psychometric Identity and Behavioral Bias 

**Authors**: Adib Sakhawat, Tahsin Islam, Takia Farhin, Syed Rifat Raiyan, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2601.06194)  

**Abstract**: As large language models (LLMs) are increasingly integrated into social decision-making, understanding their political positioning and alignment behavior is critical for safety and fairness. This study presents a sociotechnical audit of 26 prominent LLMs, triangulating their positions across three psychometric inventories (Political Compass, SapplyValues, 8 Values) and evaluating their performance on a large-scale news labeling task ($N \approx 27{,}000$). Our results reveal a strong clustering of models in the Libertarian-Left region of the ideological space, encompassing 96.3% of the cohort. Alignment signals appear to be consistent architectural traits rather than stochastic noise ($\eta^2 > 0.90$); however, we identify substantial discrepancies in measurement validity. In particular, the Political Compass exhibits a strong negative correlation with cultural progressivism ($r=-0.64$) when compared against multi-axial instruments, suggesting a conflation of social conservatism with authoritarianism in this context. We further observe a significant divergence between open-weights and closed-source models, with the latter displaying markedly higher cultural progressivism scores ($p<10^{-25}$). In downstream media analysis, models exhibit a systematic "center-shift," frequently categorizing neutral articles as left-leaning, alongside an asymmetric detection capability in which "Far Left" content is identified with greater accuracy (19.2%) than "Far Right" content (2.0%). These findings suggest that single-axis evaluations are insufficient and that multidimensional auditing frameworks are necessary to characterize alignment behavior in deployed LLMs. Our code and data will be made public. 

---
# ECLIPTICA - A Framework for Switchable LLM Alignment via CITA - Contrastive Instruction-Tuned Alignment 

**Authors**: Kapil Wanaskar, Gaytri Jena, Vinija Jain, Aman Chadha, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2601.06157)  

**Abstract**: Alignment in large language models (LLMs) is still largely static: after training, the policy is frozen. DPO, GRPO methods typically imprint one behavior into the weights, leaving little runtime control beyond prompt hacks or expensive re-alignment. We introduce ECLIPTICA, which treats alignment as instruction-driven and runtime-controllable: natural-language alignment instructions act as an explicit behavioral contract (stance, refusal boundary, verbosity) that modulates behavior on the fly under evolving safety requirements, user roles, and governance constraints.
We introduce CITA (Contrastive Instruction-Tuned Alignment), combining SFT with contrastive preference optimization under an explicit geometric anchor to a reference model. This yields a stable Riemannian chart and keeps instruction updates within a shared neighborhood, so regimes stay nearby and traversable for reliable switching.
To isolate policy switching from ordinary instruction following, we release the ECLIPTICA benchmark: 3000 controlled cases (300 prompts x 10 instruction types) where the user request is fixed and only the alignment instruction changes. On Llama-3.1-8B across five suites (ECLIPTICA, TruthfulQA, Conditional Safety, Length Control, LITMUS), CITA reaches 86.7% instruction-alignment efficiency, beating DPO (56.1%), GRPO (36.1%), and PPO (20.4%). 

---
# Latent Space Communication via K-V Cache Alignment 

**Authors**: Lucio M. Dery, Zohar Yahav, Henry Prior, Qixuan Feng, Jiajun Shen, Arthur Szlam  

**Link**: [PDF](https://arxiv.org/pdf/2601.06123)  

**Abstract**: Solving increasingly complex problems with large language models (LLMs) necessitates a move beyond individual models and towards multi-model systems that can effectively collaborate. While text has traditionally served as the medium for inter-model communication, a richer and more efficient exchange is possible if models can access each other's internal states directly. In this paper, we propose learning a shared representation space that aligns the k-v caches of multiple models, creating a high-bandwidth channel for collaboration without altering the underlying pre-trained parameters. We do so by augmenting each model with adapters to translate its state into and out of this shared space. Via a suite of experiments with Gemma-2 models, we demonstrate that this approach not only enables seamless inter-model communication but also improves individual model performance. We also show that the shared space allows for the direct transfer of learned skills, such as soft prompts, between different models. Our work represents a significant step towards a future where models can fluidly share knowledge and capabilities. 

---
# Islamic Chatbots in the Age of Large Language Models 

**Authors**: Muhammad Aurangzeb Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2601.06092)  

**Abstract**: Large Language Models (LLMs) are rapidly transforming how communities access, interpret, and circulate knowledge, and religious communities are no exception. Chatbots powered by LLMs are beginning to reshape authority, pedagogy, and everyday religious practice in Muslim communities. We analyze the landscape of LLM powered Islamic chatbots and how they are transforming Islamic religious practices e.g., democratizing access to religious knowledge but also running the risk of erosion of authority. We discuss what kind of challenges do these systems raise for Muslim communities and explore recommendations for the responsible design of these systems. 

---
# Using street view images and visual LLMs to predict heritage values for governance support: Risks, ethics, and policy implications 

**Authors**: Tim Johansson, Mikael Mangold, Kristina Dabrock, Anna Donarelli, Ingrid Campo-Ruiz  

**Link**: [PDF](https://arxiv.org/pdf/2601.06056)  

**Abstract**: During 2025 and 2026, the Energy Performance of Buildings Directive is being implemented in the European Union member states, requiring all member states to have National Building Renovation Plans. In Sweden, there is a lack of a national register of buildings with heritage values. This is seen as a barrier for the analyses underlying the development of Building Renovation Plans by the involved Swedish authorities. The purpose of this research was to assist Swedish authorities in assigning heritage values to building in the Swedish building stock. As part of the analyses, buildings in street view images from all over Sweden (N=154 710) have been analysed using multimodal Large Language Models (LLM) to assess aspects of heritage value. Zero-shot predictions by LLMs were used as a basis to for identifying buildings with potential heritage values for 5.0 million square meters of heated floor area for the Swedish Building Renovation Plan. In this paper, the results of the predictions and lessons learnt are presented and related to the development of Swedish Building Renovation Plan as part of governance. Potential risks for authorities using LLM-based data are addressed, with a focus on issues of transparency, error detection and sycophancy. 

---
