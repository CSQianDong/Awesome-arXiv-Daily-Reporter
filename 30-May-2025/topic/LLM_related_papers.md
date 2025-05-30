# From Token to Action: State Machine Reasoning to Mitigate Overthinking in Information Retrieval 

**Authors**: Dohyeon Lee, Yeonseok Jeong, Seung-won Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23059)  

**Abstract**: Chain-of-Thought (CoT) prompting enables complex reasoning in large language models (LLMs), including applications in information retrieval (IR). However, it often leads to overthinking, where models produce excessively long and semantically redundant traces with little or no benefit. We identify two key challenges in IR: redundant trajectories that revisit similar states and misguided reasoning that diverges from user intent. To address these, we propose State Machine Reasoning (SMR), a transition-based reasoning framework composed of discrete actions (Refine, Rerank, Stop) that support early stopping and fine-grained control. Experiments on the BEIR and BRIGHT benchmarks show that SMR improves retrieval performance (nDCG@10) by 3.4% while reducing token usage by 74.4%. It generalizes across LLMs and retrievers without requiring task-specific tuning, offering a practical alternative to conventional CoT reasoning. The code and details are available at this https URL. 

---
# Verify-in-the-Graph: Entity Disambiguation Enhancement for Complex Claim Verification with Interactive Graph Representation 

**Authors**: Hoang Pham, Thanh-Do Nguyen, Khac-Hoai Nam Bui  

**Link**: [PDF](https://arxiv.org/pdf/2505.22993)  

**Abstract**: Claim verification is a long-standing and challenging task that demands not only high accuracy but also explainability of the verification process. This task becomes an emerging research issue in the era of large language models (LLMs) since real-world claims are often complex, featuring intricate semantic structures or obfuscated entities. Traditional approaches typically address this by decomposing claims into sub-claims and querying a knowledge base to resolve hidden or ambiguous entities. However, the absence of effective disambiguation strategies for these entities can compromise the entire verification process. To address these challenges, we propose Verify-in-the-Graph (VeGraph), a novel framework leveraging the reasoning and comprehension abilities of LLM agents. VeGraph operates in three phases: (1) Graph Representation - an input claim is decomposed into structured triplets, forming a graph-based representation that integrates both structured and unstructured information; (2) Entity Disambiguation -VeGraph iteratively interacts with the knowledge base to resolve ambiguous entities within the graph for deeper sub-claim verification; and (3) Verification - remaining triplets are verified to complete the fact-checking process. Experiments using Meta-Llama-3-70B (instruct version) show that VeGraph achieves competitive performance compared to baselines on two benchmarks HoVer and FEVEROUS, effectively addressing claim verification challenges. Our source code and data are available for further exploitation. 

---
# Augment or Not? A Comparative Study of Pure and Augmented Large Language Model Recommenders 

**Authors**: Wei-Hsiang Huang, Chen-Wei Ke, Wei-Ning Chiu, Yu-Xuan Su, Chun-Chun Yang, Chieh-Yuan Cheng, Yun-Nung Chen, Pu-Jen Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.23053)  

**Abstract**: Large language models (LLMs) have introduced new paradigms for recommender systems by enabling richer semantic understanding and incorporating implicit world knowledge. In this study, we propose a systematic taxonomy that classifies existing approaches into two categories: (1) Pure LLM Recommenders, which rely solely on LLMs, and (2) Augmented LLM Recommenders, which integrate additional non-LLM techniques to enhance performance. This taxonomy provides a novel lens through which to examine the evolving landscape of LLM-based recommendation. To support fair comparison, we introduce a unified evaluation platform that benchmarks representative models under consistent experimental settings, highlighting key design choices that impact effectiveness. We conclude by discussing open challenges and outlining promising directions for future research. This work offers both a comprehensive overview and practical guidance for advancing next-generation LLM-powered recommender. 

---
# Deep Retrieval at CheckThat! 2025: Identifying Scientific Papers from Implicit Social Media Mentions via Hybrid Retrieval and Re-Ranking 

**Authors**: Pascal J. Sager, Ashwini Kamaraj, Benjamin F. Grewe, Thilo Stadelmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.23250)  

**Abstract**: We present the methodology and results of the Deep Retrieval team for subtask 4b of the CLEF CheckThat! 2025 competition, which focuses on retrieving relevant scientific literature for given social media posts. To address this task, we propose a hybrid retrieval pipeline that combines lexical precision, semantic generalization, and deep contextual re-ranking, enabling robust retrieval that bridges the informal-to-formal language gap. Specifically, we combine BM25-based keyword matching with a FAISS vector store using a fine-tuned INF-Retriever-v1 model for dense semantic retrieval. BM25 returns the top 30 candidates, and semantic search yields 100 candidates, which are then merged and re-ranked via a large language model (LLM)-based cross-encoder.
Our approach achieves a mean reciprocal rank at 5 (MRR@5) of 76.46% on the development set and 66.43% on the hidden test set, securing the 1st position on the development leaderboard and ranking 3rd on the test leaderboard (out of 31 teams), with a relative performance gap of only 2 percentage points compared to the top-ranked system. We achieve this strong performance by running open-source models locally and without external training data, highlighting the effectiveness of a carefully designed and fine-tuned retrieval pipeline. 

---
# Fortune: Formula-Driven Reinforcement Learning for Symbolic Table Reasoning in Language Models 

**Authors**: Lang Cao, Jingxian Xu, Hanbing Liu, Jinyu Wang, Mengyu Zhou, Haoyu Dong, Shi Han, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23667)  

**Abstract**: Tables are a fundamental structure for organizing and analyzing data, making effective table understanding a critical capability for intelligent systems. While large language models (LMs) demonstrate strong general reasoning abilities, they continue to struggle with accurate numerical or symbolic reasoning over tabular data, especially in complex scenarios. Spreadsheet formulas provide a powerful and expressive medium for representing executable symbolic operations, encoding rich reasoning patterns that remain largely underutilized. In this paper, we propose Formula Tuning (Fortune), a reinforcement learning (RL) framework that trains LMs to generate executable spreadsheet formulas for question answering over general tabular data. Formula Tuning reduces the reliance on supervised formula annotations by using binary answer correctness as a reward signal, guiding the model to learn formula derivation through reasoning. We provide a theoretical analysis of its advantages and demonstrate its effectiveness through extensive experiments on seven table reasoning benchmarks. Formula Tuning substantially enhances LM performance, particularly on multi-step numerical and symbolic reasoning tasks, enabling a 7B model to outperform O1 on table understanding. This highlights the potential of formula-driven RL to advance symbolic table reasoning in LMs. 

---
# Autoformalization in the Era of Large Language Models: A Survey 

**Authors**: Ke Weng, Lun Du, Sirui Li, Wangyue Lu, Haozhe Sun, Hengyu Liu, Tiancheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23486)  

**Abstract**: Autoformalization, the process of transforming informal mathematical propositions into verifiable formal representations, is a foundational task in automated theorem proving, offering a new perspective on the use of mathematics in both theoretical and applied domains. Driven by the rapid progress in artificial intelligence, particularly large language models (LLMs), this field has witnessed substantial growth, bringing both new opportunities and unique challenges. In this survey, we provide a comprehensive overview of recent advances in autoformalization from both mathematical and LLM-centric perspectives. We examine how autoformalization is applied across various mathematical domains and levels of difficulty, and analyze the end-to-end workflow from data preprocessing to model design and evaluation. We further explore the emerging role of autoformalization in enhancing the verifiability of LLM-generated outputs, highlighting its potential to improve both the trustworthiness and reasoning capabilities of LLMs. Finally, we summarize key open-source models and datasets supporting current research, and discuss open challenges and promising future directions for the field. 

---
# EVOREFUSE: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions 

**Authors**: Xiaorui Wu, Xiaofeng Mao, Fei Li, Xin Zhang, Xiaolu Zhang, Jun Zhou, Yuxiang Peng, Li Zheng, Chong Teng, Donghong Ji, Zhuang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.23473)  

**Abstract**: Large language models (LLMs) frequently refuse to respond to pseudo-malicious instructions: semantically harmless input queries triggering unnecessary LLM refusals due to conservative safety alignment, significantly impairing user experience. Collecting such instructions is crucial for evaluating and mitigating over-refusals, but existing instruction curation methods, like manual creation or instruction rewriting, either lack scalability or fail to produce sufficiently diverse and effective refusal-inducing prompts. To address these limitations, we introduce EVOREFUSE, a prompt optimization approach that generates diverse pseudo-malicious instructions consistently eliciting confident refusals across LLMs. EVOREFUSE employs an evolutionary algorithm exploring the instruction space in more diverse directions than existing methods via mutation strategies and recombination, and iteratively evolves seed instructions to maximize evidence lower bound on LLM refusal probability. Using EVOREFUSE, we create two novel datasets: EVOREFUSE-TEST, a benchmark of 582 pseudo-malicious instructions that outperforms the next-best benchmark with 140.41% higher average refusal triggering rate across 9 LLMs, 34.86% greater lexical diversity, and 40.03% improved LLM response confidence scores; and EVOREFUSE-ALIGN, which provides 3,000 pseudo-malicious instructions with responses for supervised and preference-based alignment training. LLAMA3.1-8B-INSTRUCT supervisedly fine-tuned on EVOREFUSE-ALIGN achieves up to 14.31% fewer over-refusals than models trained on the second-best alignment dataset, without compromising safety. Our analysis with EVOREFUSE-TEST reveals models trigger over-refusals by overly focusing on sensitive keywords while ignoring broader context. 

---
# MathArena: Evaluating LLMs on Uncontaminated Math Competitions 

**Authors**: Mislav Balunović, Jasper Dekoninck, Ivo Petrov, Nikola Jovanović, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2505.23281)  

**Abstract**: The rapid advancement of reasoning capabilities in large language models (LLMs) has led to notable improvements on mathematical benchmarks. However, many of the most commonly used evaluation datasets (e.g., AIME 2024) are widely available online, making it difficult to disentangle genuine reasoning from potential memorization. Furthermore, these benchmarks do not evaluate proof-writing capabilities, which are crucial for many mathematical tasks. To address this, we introduce MathArena, a new benchmark based on the following key insight: recurring math competitions provide a stream of high-quality, challenging problems that can be used for real-time evaluation of LLMs. By evaluating models as soon as new problems are released, we effectively eliminate the risk of contamination. Using this framework, we find strong signs of contamination in AIME 2024. Nonetheless, evaluations on harder competitions, such as SMT 2025 -- published well after model release dates -- demonstrate impressive reasoning capabilities in top-performing models. MathArena is also the first benchmark for proof-writing capabilities. On USAMO 2025, even top models score below 25%, far behind their performance on final-answer tasks. So far, we have evaluated 30 models across five competitions, totaling 149 problems. As an evolving benchmark, MathArena will continue to track the progress of LLMs on newly released competitions, ensuring rigorous and up-to-date evaluation of mathematical reasoning. 

---
# SafeScientist: Toward Risk-Aware Scientific Discoveries by LLM Agents 

**Authors**: Kunlun Zhu, Jiaxun Zhang, Ziheng Qi, Nuoxing Shang, Zijia Liu, Peixuan Han, Yue Su, Haofei Yu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2505.23559)  

**Abstract**: Recent advancements in large language model (LLM) agents have significantly accelerated scientific discovery automation, yet concurrently raised critical ethical and safety concerns. To systematically address these challenges, we introduce \textbf{SafeScientist}, an innovative AI scientist framework explicitly designed to enhance safety and ethical responsibility in AI-driven scientific exploration. SafeScientist proactively refuses ethically inappropriate or high-risk tasks and rigorously emphasizes safety throughout the research process. To achieve comprehensive safety oversight, we integrate multiple defensive mechanisms, including prompt monitoring, agent-collaboration monitoring, tool-use monitoring, and an ethical reviewer component. Complementing SafeScientist, we propose \textbf{SciSafetyBench}, a novel benchmark specifically designed to evaluate AI safety in scientific contexts, comprising 240 high-risk scientific tasks across 6 domains, alongside 30 specially designed scientific tools and 120 tool-related risk tasks. Extensive experiments demonstrate that SafeScientist significantly improves safety performance by 35\% compared to traditional AI scientist frameworks, without compromising scientific output quality. Additionally, we rigorously validate the robustness of our safety pipeline against diverse adversarial attack methods, further confirming the effectiveness of our integrated approach. The code and data will be available at this https URL. \textcolor{red}{Warning: this paper contains example data that may be offensive or harmful.} 

---
# Be.FM: Open Foundation Models for Human Behavior 

**Authors**: Yutong Xie, Zhuoheng Li, Xiyuan Wang, Yijun Pan, Qijia Liu, Xingzhi Cui, Kuang-Yu Lo, Ruoyi Gao, Xingjian Zhang, Jin Huang, Walter Yuan, Matthew O. Jackson, Qiaozhu Mei  

**Link**: [PDF](https://arxiv.org/pdf/2505.23058)  

**Abstract**: Despite their success in numerous fields, the potential of foundation models for modeling and understanding human behavior remains largely unexplored. We introduce this http URL, one of the first open foundation models designed for human behavior modeling. Built upon open-source large language models and fine-tuned on a diverse range of behavioral data, this http URL can be used to understand and predict human decision-making. We construct a comprehensive set of benchmark tasks for testing the capabilities of behavioral foundation models. Our results demonstrate that this http URL can predict behaviors, infer characteristics of individuals and populations, generate insights about contexts, and apply behavioral science knowledge. 

---
# Case-Based Reasoning Enhances the Predictive Power of LLMs in Drug-Drug Interaction 

**Authors**: Guangyi Liu, Yongqi Zhang, Xunyuan Liu, Quanming Yao  

**Link**: [PDF](https://arxiv.org/pdf/2505.23034)  

**Abstract**: Drug-drug interaction (DDI) prediction is critical for treatment safety. While large language models (LLMs) show promise in pharmaceutical tasks, their effectiveness in DDI prediction remains challenging. Inspired by the well-established clinical practice where physicians routinely reference similar historical cases to guide their decisions through case-based reasoning (CBR), we propose CBR-DDI, a novel framework that distills pharmacological principles from historical cases to improve LLM reasoning for DDI tasks. CBR-DDI constructs a knowledge repository by leveraging LLMs to extract pharmacological insights and graph neural networks (GNNs) to model drug associations. A hybrid retrieval mechanism and dual-layer knowledge-enhanced prompting allow LLMs to effectively retrieve and reuse relevant cases. We further introduce a representative sampling strategy for dynamic case refinement. Extensive experiments demonstrate that CBR-DDI achieves state-of-the-art performance, with a significant 28.7% accuracy improvement over both popular LLMs and CBR baseline, while maintaining high interpretability and flexibility. 

---
# Let's Reason Formally: Natural-Formal Hybrid Reasoning Enhances LLM's Math Capability 

**Authors**: Ruida Wang, Yuxin Li, Yi R., Fung, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23703)  

**Abstract**: Enhancing the mathematical reasoning capabilities of LLMs has garnered significant attention in both the mathematical and computer science communities. Recent works have made substantial progress in both Natural Language (NL) reasoning and Formal Language (FL) reasoning by leveraging the potential of pure Reinforcement Learning (RL) methods on base models. However, RL approaches struggle to impart new capabilities not presented in the base model, highlighting the need to integrate more knowledge like FL into NL math reasoning effectively. Yet, this integration is challenging due to inherent disparities in problem structure and reasoning format between NL and FL. To address these challenges, we introduce **NL-FL HybridReasoning**, an end-to-end framework designed to incorporate the FL expert into NL math problem-solving. To bridge the NL and FL input format gap, we propose the *NL-FL Problem Alignment* method, which reformulates the Question-Answering (QA) problems in NL as existence theorems in FL. Subsequently, the *Mixed Problem Input* technique we provide enables the FL reasoner to handle both QA and existence problems concurrently. Lastly, we mitigate the NL and FL output format gap in reasoning through an LLM-based *Answer Extraction* mechanism. Comprehensive experiments demonstrate that the **HybridReasoning** framework achieves **89.80%** and **84.34%** accuracy rates on the MATH-500 and the AMC benchmarks, surpassing the NL baseline by 4.60% and 4.82%, respectively. Notably, some problems resolved by our framework remain unsolved by the NL baseline model even under a larger number of trials. 

---
# Decomposing Elements of Problem Solving: What "Math" Does RL Teach? 

**Authors**: Tian Qin, Core Francisco Park, Mujin Kwun, Aaron Walsman, Eran Malach, Nikhil Anand, Hidenori Tanaka, David Alvarez-Melis  

**Link**: [PDF](https://arxiv.org/pdf/2505.22756)  

**Abstract**: Mathematical reasoning tasks have become prominent benchmarks for assessing the reasoning capabilities of LLMs, especially with reinforcement learning (RL) methods such as GRPO showing significant performance gains. However, accuracy metrics alone do not support fine-grained assessment of capabilities and fail to reveal which problem-solving skills have been internalized. To better understand these capabilities, we propose to decompose problem solving into fundamental capabilities: Plan (mapping questions to sequences of steps), Execute (correctly performing solution steps), and Verify (identifying the correctness of a solution). Empirically, we find that GRPO mainly enhances the execution skill-improving execution robustness on problems the model already knows how to solve-a phenomenon we call temperature distillation. More importantly, we show that RL-trained models struggle with fundamentally new problems, hitting a 'coverage wall' due to insufficient planning skills. To explore RL's impact more deeply, we construct a minimal, synthetic solution-tree navigation task as an analogy for mathematical problem-solving. This controlled setup replicates our empirical findings, confirming RL primarily boosts execution robustness. Importantly, in this setting, we identify conditions under which RL can potentially overcome the coverage wall through improved exploration and generalization to new solution paths. Our findings provide insights into the role of RL in enhancing LLM reasoning, expose key limitations, and suggest a path toward overcoming these barriers. Code is available at this https URL. 

---
# Design and testing of an agent chatbot supporting decision making with public transport data 

**Authors**: Luca Fantin, Marco Antonelli, Margherita Cesetti, Daniele Irto, Bruno Zamengo, Francesco Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2505.22698)  

**Abstract**: Assessing the quality of public transportation services requires the analysis of large quantities of data on the scheduled and actual trips and documents listing the quality constraints each service needs to meet. Interrogating such datasets with SQL queries, organizing and visualizing the data can be quite complex for most users. This paper presents a chatbot offering a user-friendly tool to interact with these datasets and support decision making. It is based on an agent architecture, which expands the capabilities of the core Large Language Model (LLM) by allowing it to interact with a series of tools that can execute several tasks, like performing SQL queries, plotting data and creating maps from the coordinates of a trip and its stops. This paper also tackles one of the main open problems of such Generative AI projects: collecting data to measure the system's performance. Our chatbot has been extensively tested with a workflow that asks several questions and stores the generated query, the retrieved data and the natural language response for each of them. Such questions are drawn from a set of base examples which are then completed with actual data from the database. This procedure yields a dataset for the evaluation of the chatbot's performance, especially the consistency of its answers and the correctness of the generated queries. 

---
# From Chat Logs to Collective Insights: Aggregative Question Answering 

**Authors**: Wentao Zhang, Woojeong Kim, Yuntian Deng  

**Link**: [PDF](https://arxiv.org/pdf/2505.23765)  

**Abstract**: Conversational agents powered by large language models (LLMs) are rapidly becoming integral to our daily interactions, generating unprecedented amounts of conversational data. Such datasets offer a powerful lens into societal interests, trending topics, and collective concerns. Yet, existing approaches typically treat these interactions as independent and miss critical insights that could emerge from aggregating and reasoning across large-scale conversation logs. In this paper, we introduce Aggregative Question Answering, a novel task requiring models to reason explicitly over thousands of user-chatbot interactions to answer aggregative queries, such as identifying emerging concerns among specific demographics. To enable research in this direction, we construct a benchmark, WildChat-AQA, comprising 6,027 aggregative questions derived from 182,330 real-world chatbot conversations. Experiments show that existing methods either struggle to reason effectively or incur prohibitive computational costs, underscoring the need for new approaches capable of extracting collective insights from large-scale conversational data. 

---
# DeepTheorem: Advancing LLM Reasoning for Theorem Proving Through Natural Language and Reinforcement Learning 

**Authors**: Ziyin Zhang, Jiahao Xu, Zhiwei He, Tian Liang, Qiuzhi Liu, Yansi Li, Linfeng Song, Zhengwen Liang, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.23754)  

**Abstract**: Theorem proving serves as a major testbed for evaluating complex reasoning abilities in large language models (LLMs). However, traditional automated theorem proving (ATP) approaches rely heavily on formal proof systems that poorly align with LLMs' strength derived from informal, natural language knowledge acquired during pre-training. In this work, we propose DeepTheorem, a comprehensive informal theorem-proving framework exploiting natural language to enhance LLM mathematical reasoning. DeepTheorem includes a large-scale benchmark dataset consisting of 121K high-quality IMO-level informal theorems and proofs spanning diverse mathematical domains, rigorously annotated for correctness, difficulty, and topic categories, accompanied by systematically constructed verifiable theorem variants. We devise a novel reinforcement learning strategy (RL-Zero) explicitly tailored to informal theorem proving, leveraging the verified theorem variants to incentivize robust mathematical inference. Additionally, we propose comprehensive outcome and process evaluation metrics examining proof correctness and the quality of reasoning steps. Extensive experimental analyses demonstrate DeepTheorem significantly improves LLM theorem-proving performance compared to existing datasets and supervised fine-tuning protocols, achieving state-of-the-art accuracy and reasoning quality. Our findings highlight DeepTheorem's potential to fundamentally advance automated informal theorem proving and mathematical exploration. 

---
# Bounded Rationality for LLMs: Satisficing Alignment at Inference-Time 

**Authors**: Mohamad Chehade, Soumya Suvra Ghosal, Souradip Chakraborty, Avinash Reddy, Dinesh Manocha, Hao Zhu, Amrit Singh Bedi  

**Link**: [PDF](https://arxiv.org/pdf/2505.23729)  

**Abstract**: Aligning large language models with humans is challenging due to the inherently multifaceted nature of preference feedback. While existing approaches typically frame this as a multi-objective optimization problem, they often overlook how humans actually make decisions. Research on bounded rationality suggests that human decision making follows satisficing strategies-optimizing primary objectives while ensuring others meet acceptable thresholds. To bridge this gap and operationalize the notion of satisficing alignment, we propose SITAlign: an inference time framework that addresses the multifaceted nature of alignment by maximizing a primary objective while satisfying threshold-based constraints on secondary criteria. We provide theoretical insights by deriving sub-optimality bounds of our satisficing based inference alignment approach. We empirically validate SITAlign's performance through extensive experimentation on multiple benchmarks. For instance, on the PKU-SafeRLHF dataset with the primary objective of maximizing helpfulness while ensuring a threshold on harmlessness, SITAlign outperforms the state-of-the-art multi objective decoding strategy by a margin of 22.3% in terms of GPT-4 win-tie rate for helpfulness reward while adhering to the threshold on harmlessness. 

---
# VF-Eval: Evaluating Multimodal LLMs for Generating Feedback on AIGC Videos 

**Authors**: Tingyu Song, Tongyan Hu, Guo Gan, Yilun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.23693)  

**Abstract**: MLLMs have been widely studied for video question answering recently. However, most existing assessments focus on natural videos, overlooking synthetic videos, such as AI-generated content (AIGC). Meanwhile, some works in video generation rely on MLLMs to evaluate the quality of generated videos, but the capabilities of MLLMs on interpreting AIGC videos remain largely underexplored. To address this, we propose a new benchmark, VF-Eval, which introduces four tasks-coherence validation, error awareness, error type detection, and reasoning evaluation-to comprehensively evaluate the abilities of MLLMs on AIGC videos. We evaluate 13 frontier MLLMs on VF-Eval and find that even the best-performing model, GPT-4.1, struggles to achieve consistently good performance across all tasks. This highlights the challenging nature of our benchmark. Additionally, to investigate the practical applications of VF-Eval in improving video generation, we conduct an experiment, RePrompt, demonstrating that aligning MLLMs more closely with human feedback can benefit video generation. 

---
# AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction from Web-Scale Corpora 

**Authors**: Jiaxin Bai, Wei Fan, Qi Hu, Qing Zong, Chunyang Li, Hong Ting Tsang, Hongyu Luo, Yauwai Yim, Haoyu Huang, Xiao Zhou, Feng Qin, Tianshi Zheng, Xi Peng, Xin Yao, Huiwen Yang, Leijie Wu, Yi Ji, Gong Zhang, Renhai Chen, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.23628)  

**Abstract**: We present AutoSchemaKG, a framework for fully autonomous knowledge graph construction that eliminates the need for predefined schemas. Our system leverages large language models to simultaneously extract knowledge triples and induce comprehensive schemas directly from text, modeling both entities and events while employing conceptualization to organize instances into semantic categories. Processing over 50 million documents, we construct ATLAS (Automated Triple Linking And Schema induction), a family of knowledge graphs with 900+ million nodes and 5.9 billion edges. This approach outperforms state-of-the-art baselines on multi-hop QA tasks and enhances LLM factuality. Notably, our schema induction achieves 95\% semantic alignment with human-crafted schemas with zero manual intervention, demonstrating that billion-scale knowledge graphs with dynamically induced schemas can effectively complement parametric knowledge in large language models. 

---
# One Trajectory, One Token: Grounded Video Tokenization via Panoptic Sub-object Trajectory 

**Authors**: Chenhao Zheng, Jieyu Zhang, Mohammadreza Salehi, Ziqi Gao, Vishnu Iyengar, Norimasa Kobori, Quan Kong, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2505.23617)  

**Abstract**: Effective video tokenization is critical for scaling transformer models for long videos. Current approaches tokenize videos using space-time patches, leading to excessive tokens and computational inefficiencies. The best token reduction strategies degrade performance and barely reduce the number of tokens when the camera moves. We introduce grounded video tokenization, a paradigm that organizes tokens based on panoptic sub-object trajectories rather than fixed patches. Our method aligns with fundamental perceptual principles, ensuring that tokenization reflects scene complexity rather than video duration. We propose TrajViT, a video encoder that extracts object trajectories and converts them into semantically meaningful tokens, significantly reducing redundancy while maintaining temporal coherence. Trained with contrastive learning, TrajViT significantly outperforms space-time ViT (ViT3D) across multiple video understanding benchmarks, e.g., TrajViT outperforms ViT3D by a large margin of 6% top-5 recall in average at video-text retrieval task with 10x token deduction. We also show TrajViT as a stronger model than ViT3D for being the video encoder for modern VideoLLM, obtaining an average of 5.2% performance improvement across 6 VideoQA benchmarks while having 4x faster training time and 18x less inference FLOPs. TrajViT is the first efficient encoder to consistently outperform ViT3D across diverse video analysis tasks, making it a robust and scalable solution. 

---
# Spatial-MLLM: Boosting MLLM Capabilities in Visual-based Spatial Intelligence 

**Authors**: Diankun Wu, Fangfu Liu, Yi-Hsin Hung, Yueqi Duan  

**Link**: [PDF](https://arxiv.org/pdf/2505.23747)  

**Abstract**: Recent advancements in Multimodal Large Language Models (MLLMs) have significantly enhanced performance on 2D visual tasks. However, improving their spatial intelligence remains a challenge. Existing 3D MLLMs always rely on additional 3D or 2.5D data to incorporate spatial awareness, restricting their utility in scenarios with only 2D inputs, such as images or videos. In this paper, we present Spatial-MLLM, a novel framework for visual-based spatial reasoning from purely 2D observations. Unlike conventional video MLLMs which rely on CLIP-based visual encoders optimized for semantic understanding, our key insight is to unleash the strong structure prior from the feed-forward visual geometry foundation model. Specifically, we propose a dual-encoder architecture: a pretrained 2D visual encoder to extract semantic features, and a spatial encoder-initialized from the backbone of the visual geometry model-to extract 3D structure features. A connector then integrates both features into unified visual tokens for enhanced spatial understanding. Furthermore, we propose a space-aware frame sampling strategy at inference time, which selects the spatially informative frames of a video sequence, ensuring that even under limited token length, the model focuses on frames critical for spatial reasoning. Beyond architecture improvements, we construct the Spatial-MLLM-120k dataset and train the model on it using supervised fine-tuning and GRPO. Extensive experiments on various real-world datasets demonstrate that our spatial-MLLM achieves state-of-the-art performance in a wide range of visual-based spatial understanding and reasoning tasks. Project page: this https URL. 

---
# Active Layer-Contrastive Decoding Reduces Hallucination in Large Language Model Generation 

**Authors**: Hongxiang Zhang, Hao Chen, Tianyi Zhang, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.23657)  

**Abstract**: Recent decoding methods improve the factuality of large language models~(LLMs) by refining how the next token is selected during generation. These methods typically operate at the token level, leveraging internal representations to suppress superficial patterns. Nevertheless, LLMs remain prone to hallucinations, especially over longer contexts. In this paper, we propose Active Layer-Contrastive Decoding (ActLCD), a novel decoding strategy that actively decides when to apply contrasting layers during generation. By casting decoding as a sequential decision-making problem, ActLCD employs a reinforcement learning policy guided by a reward-aware classifier to optimize factuality beyond the token level. Our experiments demonstrate that ActLCD surpasses state-of-the-art methods across five benchmarks, showcasing its effectiveness in mitigating hallucinations in diverse generation scenarios. 

---
# Cognitive Guardrails for Open-World Decision Making in Autonomous Drone Swarms 

**Authors**: Jane Cleland-Huang, Pedro Antonio Alarcon Granadeno, Arturo Miguel Russell Bernal, Demetrius Hernandez, Michael Murphy, Maureen Petterson, Walter Scheirer  

**Link**: [PDF](https://arxiv.org/pdf/2505.23576)  

**Abstract**: Small Uncrewed Aerial Systems (sUAS) are increasingly deployed as autonomous swarms in search-and-rescue and other disaster-response scenarios. In these settings, they use computer vision (CV) to detect objects of interest and autonomously adapt their missions. However, traditional CV systems often struggle to recognize unfamiliar objects in open-world environments or to infer their relevance for mission planning. To address this, we incorporate large language models (LLMs) to reason about detected objects and their implications. While LLMs can offer valuable insights, they are also prone to hallucinations and may produce incorrect, misleading, or unsafe recommendations. To ensure safe and sensible decision-making under uncertainty, high-level decisions must be governed by cognitive guardrails. This article presents the design, simulation, and real-world integration of these guardrails for sUAS swarms in search-and-rescue missions. 

---
# Can Large Language Models Challenge CNNS in Medical Image Analysis? 

**Authors**: Shibbir Ahmed, Shahnewaz Karim Sakib, Anindya Bijoy Das  

**Link**: [PDF](https://arxiv.org/pdf/2505.23503)  

**Abstract**: This study presents a multimodal AI framework designed for precisely classifying medical diagnostic images. Utilizing publicly available datasets, the proposed system compares the strengths of convolutional neural networks (CNNs) and different large language models (LLMs). This in-depth comparative analysis highlights key differences in diagnostic performance, execution efficiency, and environmental impacts. Model evaluation was based on accuracy, F1-score, average execution time, average energy consumption, and estimated $CO_2$ emission. The findings indicate that although CNN-based models can outperform various multimodal techniques that incorporate both images and contextual information, applying additional filtering on top of LLMs can lead to substantial performance gains. These findings highlight the transformative potential of multimodal AI systems to enhance the reliability, efficiency, and scalability of medical diagnostics in clinical settings. 

---
# SWE-bench Goes Live! 

**Authors**: Linghao Zhang, Shilin He, Chaoyun Zhang, Yu Kang, Bowen Li, Chengxing Xie, Junhao Wang, Maoquan Wang, Yufan Huang, Shengyu Fu, Elsie Nallipogu, Qingwei Lin, Yingnong Dang, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23419)  

**Abstract**: The issue-resolving task, where a model generates patches to fix real-world bugs, has emerged as a critical benchmark for evaluating the capabilities of large language models (LLMs). While SWE-bench and its variants have become standard in this domain, they suffer from key limitations: they have not been updated since their initial releases, cover a narrow set of repositories, and depend heavily on manual effort for instance construction and environment setup. These factors hinder scalability and introduce risks of overfitting and data contamination. In this work, we present \textbf{SWE-bench-Live}, a \textit{live-updatable} benchmark designed to overcome these challenges. Our initial release consists of 1,319 tasks derived from real GitHub issues created since 2024, spanning 93 repositories. Each task is accompanied by a dedicated Docker image to ensure reproducible execution. Central to our benchmark is \method, an automated curation pipeline that streamlines the entire process from instance creation to environment setup, removing manual bottlenecks and enabling scalability and continuous updates. We evaluate a range of state-of-the-art agent frameworks and LLMs on SWE-bench-Live, revealing a substantial performance gap compared to static benchmarks like SWE-bench, even under controlled evaluation conditions. To better understand this discrepancy, we perform detailed analyses across repository origin, issue recency, and task difficulty. By providing a fresh, diverse, and executable benchmark grounded in live repository activity, SWE-bench-Live facilitates rigorous, contamination-resistant evaluation of LLMs and agents in dynamic, real-world software development settings. 

---
# Engineering Serendipity through Recommendations of Items with Atypical Aspects 

**Authors**: Ramit Aditya, Razvan Bunescu, Smita Nannaware, Erfan Al-Hossami  

**Link**: [PDF](https://arxiv.org/pdf/2505.23580)  

**Abstract**: A restaurant dinner or a hotel stay may lead to memorable experiences when guests encounter unexpected aspects that also match their interests. For example, an origami-making station in the waiting area of a restaurant may be both surprising and enjoyable for a customer who is passionate about paper crafts. Similarly, an exhibit of 18th century harpsichords would be atypical for a hotel lobby and likely pique the interest of a guest who has a passion for Baroque music. Motivated by this insight, in this paper we introduce the new task of engineering serendipity through recommendations of items with atypical aspects. We describe an LLM-based system pipeline that extracts atypical aspects from item reviews, then estimates and aggregates their user-specific utility in a measure of serendipity potential that is used to rerank a list of items recommended to the user. To facilitate system development and evaluation, we introduce a dataset of Yelp reviews that are manually annotated with atypical aspects and a dataset of artificially generated user profiles, together with crowdsourced annotations of user-aspect utility values. Furthermore, we introduce a custom procedure for dynamic selection of in-context learning examples, which is shown to improve LLM-based judgments of atypicality and utility. Experimental evaluations show that serendipity-based rankings generated by the system are highly correlated with ground truth rankings for which serendipity scores are computed from manual annotations of atypical aspects and their user-dependent utility. Overall, we hope that the new recommendation task and the associated system presented in this paper catalyze further research into recommendation approaches that go beyond accuracy in their pursuit of enhanced user satisfaction.
The datasets and the code are made publicly available at this https URL . 

---
# Does Machine Unlearning Truly Remove Model Knowledge? A Framework for Auditing Unlearning in LLMs 

**Authors**: Haokun Chen, Yueqi Zhang, Yuan Bi, Yao Zhang, Tong Liu, Jinhe Bi, Jian Lan, Jindong Gu, Claudia Grosser, Denis Krompass, Nassir Navab, Volker Tresp  

**Link**: [PDF](https://arxiv.org/pdf/2505.23270)  

**Abstract**: In recent years, Large Language Models (LLMs) have achieved remarkable advancements, drawing significant attention from the research community. Their capabilities are largely attributed to large-scale architectures, which require extensive training on massive datasets. However, such datasets often contain sensitive or copyrighted content sourced from the public internet, raising concerns about data privacy and ownership. Regulatory frameworks, such as the General Data Protection Regulation (GDPR), grant individuals the right to request the removal of such sensitive information. This has motivated the development of machine unlearning algorithms that aim to remove specific knowledge from models without the need for costly retraining. Despite these advancements, evaluating the efficacy of unlearning algorithms remains a challenge due to the inherent complexity and generative nature of LLMs. In this work, we introduce a comprehensive auditing framework for unlearning evaluation, comprising three benchmark datasets, six unlearning algorithms, and five prompt-based auditing methods. By using various auditing algorithms, we evaluate the effectiveness and robustness of different unlearning strategies. To explore alternatives beyond prompt-based auditing, we propose a novel technique that leverages intermediate activation perturbations, addressing the limitations of auditing methods that rely solely on model inputs and outputs. 

---
# How Does Response Length Affect Long-Form Factuality 

**Authors**: James Xu Zhao, Jimmy Z.J. Liu, Bryan Hooi, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2505.23295)  

**Abstract**: Large language models (LLMs) are widely used for long-form text generation. However, factual errors in the responses would undermine their reliability. Despite growing attention to LLM factuality, the effect of response length on factuality remains underexplored. In this work, we systematically investigate this relationship by first introducing an automatic and bi-level long-form factuality evaluation framework, which achieves high agreement with human annotations while being cost-effective. Using this framework, we conduct controlled experiments and find that longer responses exhibit lower factual precision, confirming the presence of length bias. To explain this phenomenon, we empirically examine three hypotheses: error propagation, long context, and facts exhaustion. Our results reveal that facts exhaustion, where the model gradually exhausts more reliable knowledge, is the primary cause of factual degradation, rather than the other two hypotheses. 

---
# MCTSr-Zero: Self-Reflective Psychological Counseling Dialogues Generation via Principles and Adaptive Exploration 

**Authors**: Hao Lu, Yanchi Gu, Haoyuan Huang, Yulin Zhou, Ningxin Zhu, Chen Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.23229)  

**Abstract**: The integration of Monte Carlo Tree Search (MCTS) with Large Language Models (LLMs) has demonstrated significant success in structured, problem-oriented tasks. However, applying these methods to open-ended dialogues, such as those in psychological counseling, presents unique challenges. Unlike tasks with objective correctness, success in therapeutic conversations depends on subjective factors like empathetic engagement, ethical adherence, and alignment with human preferences, for which strict "correctness" criteria are ill-defined. Existing result-oriented MCTS approaches can therefore produce misaligned responses. To address this, we introduce MCTSr-Zero, an MCTS framework designed for open-ended, human-centric dialogues. Its core innovation is "domain alignment", which shifts the MCTS search objective from predefined end-states towards conversational trajectories that conform to target domain principles (e.g., empathy in counseling). Furthermore, MCTSr-Zero incorporates "Regeneration" and "Meta-Prompt Adaptation" mechanisms to substantially broaden exploration by allowing the MCTS to consider fundamentally different initial dialogue strategies. We evaluate MCTSr-Zero in psychological counseling by generating multi-turn dialogue data, which is used to fine-tune an LLM, PsyLLM. We also introduce PsyEval, a benchmark for assessing multi-turn psychological counseling dialogues. Experiments demonstrate that PsyLLM achieves state-of-the-art performance on PsyEval and other relevant metrics, validating MCTSr-Zero's effectiveness in generating high-quality, principle-aligned conversational data for human-centric domains and addressing the LLM challenge of consistently adhering to complex psychological standards. 

---
# The Arabic AI Fingerprint: Stylometric Analysis and Detection of Large Language Models Text 

**Authors**: Maged S. Al-Shaibani, Moataz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2505.23276)  

**Abstract**: Large Language Models (LLMs) have achieved unprecedented capabilities in generating human-like text, posing subtle yet significant challenges for information integrity across critical domains, including education, social media, and academia, enabling sophisticated misinformation campaigns, compromising healthcare guidance, and facilitating targeted propaganda. This challenge becomes severe, particularly in under-explored and low-resource languages like Arabic. This paper presents a comprehensive investigation of Arabic machine-generated text, examining multiple generation strategies (generation from the title only, content-aware generation, and text refinement) across diverse model architectures (ALLaM, Jais, Llama, and GPT-4) in academic, and social media domains. Our stylometric analysis reveals distinctive linguistic patterns differentiating human-written from machine-generated Arabic text across these varied contexts. Despite their human-like qualities, we demonstrate that LLMs produce detectable signatures in their Arabic outputs, with domain-specific characteristics that vary significantly between different contexts. Based on these insights, we developed BERT-based detection models that achieved exceptional performance in formal contexts (up to 99.9\% F1-score) with strong precision across model architectures. Our cross-domain analysis confirms generalization challenges previously reported in the literature. To the best of our knowledge, this work represents the most comprehensive investigation of Arabic machine-generated text to date, uniquely combining multiple prompt generation methods, diverse model architectures, and in-depth stylometric analysis across varied textual domains, establishing a foundation for developing robust, linguistically-informed detection systems essential for preserving information integrity in Arabic-language contexts. 

---
# ExpeTrans: LLMs Are Experiential Transfer Learners 

**Authors**: Jinglong Gao, Xiao Ding, Lingxiao Zou, Bibo Cai, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.23191)  

**Abstract**: Recent studies provide large language models (LLMs) with textual task-solving experiences via prompts to improve their performance. However, previous methods rely on substantial human labor or time to gather such experiences for each task, which is impractical given the growing variety of task types in user queries to LLMs. To address this issue, we design an autonomous experience transfer framework to explore whether LLMs can mimic human cognitive intelligence to autonomously transfer experience from existing source tasks to newly encountered target tasks. This not only allows the acquisition of experience without extensive costs of previous methods, but also offers a novel path for the generalization of LLMs. Experimental results on 13 datasets demonstrate that our framework effectively improves the performance of LLMs. Furthermore, we provide a detailed analysis of each module in the framework. 

---
# Cross-Task Experiential Learning on LLM-based Multi-Agent Collaboration 

**Authors**: Yilong Li, Chen Qian, Yu Xia, Ruijie Shi, Yufan Dang, Zihao Xie, Ziming You, Weize Chen, Cheng Yang, Weichuan Liu, Ye Tian, Xuantang Xiong, Lei Han, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.23187)  

**Abstract**: Large Language Model-based multi-agent systems (MAS) have shown remarkable progress in solving complex tasks through collaborative reasoning and inter-agent critique. However, existing approaches typically treat each task in isolation, resulting in redundant computations and limited generalization across structurally similar tasks. To address this, we introduce multi-agent cross-task experiential learning (MAEL), a novel framework that endows LLM-driven agents with explicit cross-task learning and experience accumulation. We model the task-solving workflow on a graph-structured multi-agent collaboration network, where agents propagate information and coordinate via explicit connectivity. During the experiential learning phase, we quantify the quality for each step in the task-solving workflow and store the resulting rewards along with the corresponding inputs and outputs into each agent's individual experience pool. During inference, agents retrieve high-reward, task-relevant experiences as few-shot examples to enhance the effectiveness of each reasoning step, thereby enabling more accurate and efficient multi-agent collaboration. Experimental results on diverse datasets demonstrate that MAEL empowers agents to learn from prior task experiences effectively-achieving faster convergence and producing higher-quality solutions on current tasks. 

---
# OSS-UAgent: An Agent-based Usability Evaluation Framework for Open Source Software 

**Authors**: Lingkai Meng, Yu Shao, Long Yuan, Longbin Lai, Peng Cheng, Wenyuan Yu, Wenjie Zhang, Xuemin Lin, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.23239)  

**Abstract**: Usability evaluation is critical to the impact and adoption of open source software (OSS), yet traditional methods relying on human evaluators suffer from high costs and limited scalability. To address these limitations, we introduce OSS-UAgent, an automated, configurable, and interactive agent-based usability evaluation framework specifically designed for open source software. Our framework employs intelligent agents powered by large language models (LLMs) to simulate developers performing programming tasks across various experience levels (from Junior to Expert). By dynamically constructing platform-specific knowledge bases, OSS-UAgent ensures accurate and context-aware code generation. The generated code is automatically evaluated across multiple dimensions, including compliance, correctness, and readability, providing a comprehensive measure of the software's usability. Additionally, our demonstration showcases OSS-UAgent's practical application in evaluating graph analytics platforms, highlighting its effectiveness in automating usability evaluation. 

---
# VERINA: Benchmarking Verifiable Code Generation 

**Authors**: Zhe Ye, Zhengxu Yan, Jingxuan He, Timothe Kasriel, Kaiyu Yang, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.23135)  

**Abstract**: Large language models (LLMs) are increasingly integrated in software development, but ensuring correctness in LLM-generated code remains challenging and often requires costly manual review. Verifiable code generation -- jointly generating code, specifications, and proofs of code-specification alignment -- offers a promising path to address this limitation and further unleash LLMs' benefits in coding. Yet, there exists a significant gap in evaluation: current benchmarks often lack support for end-to-end verifiable code generation. In this paper, we introduce Verina (Verifiable Code Generation Arena), a high-quality benchmark enabling a comprehensive and modular evaluation of code, specification, and proof generation as well as their compositions. Verina consists of 189 manually curated coding tasks in Lean, with detailed problem descriptions, reference implementations, formal specifications, and extensive test suites. Our extensive evaluation of state-of-the-art LLMs reveals significant challenges in verifiable code generation, especially in proof generation, underscoring the need for improving LLM-based theorem provers in verification domains. The best model, OpenAI o4-mini, generates only 61.4% correct code, 51.0% sound and complete specifications, and 3.6% successful proofs, with one trial per task. We hope Verina will catalyze progress in verifiable code generation by providing a rigorous and comprehensive benchmark. We release our dataset on this https URL and our evaluation code on this https URL. 

---
# Context Robust Knowledge Editing for Language Models 

**Authors**: Haewon Park, Gyubin Choi, Minjun Kim, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2505.23026)  

**Abstract**: Knowledge editing (KE) methods offer an efficient way to modify knowledge in large language models. Current KE evaluations typically assess editing success by considering only the edited knowledge without any preceding contexts. In real-world applications, however, preceding contexts often trigger the retrieval of the original knowledge and undermine the intended edit. To address this issue, we develop CHED -- a benchmark designed to evaluate the context robustness of KE methods. Evaluations on CHED show that they often fail when preceding contexts are present. To mitigate this shortcoming, we introduce CoRE, a KE method designed to strengthen context robustness by minimizing context-sensitive variance in hidden states of the model for edited knowledge. This method not only improves the editing success rate in situations where a preceding context is present but also preserves the overall capabilities of the model. We provide an in-depth analysis of the differing impacts of preceding contexts when introduced as user utterances versus assistant responses, and we dissect attention-score patterns to assess how specific tokens influence editing success. 

---
# AgentAlign: Navigating Safety Alignment in the Shift from Informative to Agentic Large Language Models 

**Authors**: Jinchuan Zhang, Lu Yin, Yan Zhou, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.23020)  

**Abstract**: The acquisition of agentic capabilities has transformed LLMs from "knowledge providers" to "action executors", a trend that while expanding LLMs' capability boundaries, significantly increases their susceptibility to malicious use. Previous work has shown that current LLM-based agents execute numerous malicious tasks even without being attacked, indicating a deficiency in agentic use safety alignment during the post-training phase. To address this gap, we propose AgentAlign, a novel framework that leverages abstract behavior chains as a medium for safety alignment data synthesis. By instantiating these behavior chains in simulated environments with diverse tool instances, our framework enables the generation of highly authentic and executable instructions while capturing complex multi-step dynamics. The framework further ensures model utility by proportionally synthesizing benign instructions through non-malicious interpretations of behavior chains, precisely calibrating the boundary between helpfulness and harmlessness. Evaluation results on AgentHarm demonstrate that fine-tuning three families of open-source models using our method substantially improves their safety (35.8% to 79.5% improvement) while minimally impacting or even positively enhancing their helpfulness, outperforming various prompting methods. The dataset and code have both been open-sourced. 

---
# A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs 

**Authors**: Chiwan Park, Wonjun Jang, Daeryong Kim, Aelim Ahn, Kichang Yang, Woosung Hwang, Jihyeon Roh, Hyerin Park, Hyosun Wang, Min Seok Kim, Jihoon Kang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23006)  

**Abstract**: The advancement of Large Language Models (LLMs) has led to significant improvements in various service domains, including search, recommendation, and chatbot applications. However, applying state-of-the-art (SOTA) research to industrial settings presents challenges, as it requires maintaining flexible conversational abilities while also strictly complying with service-specific constraints. This can be seen as two conflicting requirements due to the probabilistic nature of LLMs. In this paper, we propose our approach to addressing this challenge and detail the strategies we employed to overcome their inherent limitations in real-world applications. We conduct a practical case study of a conversational agent designed for the e-commerce domain, detailing our implementation workflow and optimizations. Our findings provide insights into bridging the gap between academic research and real-world application, introducing a framework for developing scalable, controllable, and reliable AI-driven agents. 

---
# WorkForceAgent-R1: Incentivizing Reasoning Capability in LLM-based Web Agents via Reinforcement Learning 

**Authors**: Yuchen Zhuang, Di Jin, Jiaao Chen, Wenqi Shi, Hanrui Wang, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22942)  

**Abstract**: Large language models (LLMs)-empowered web agents enables automating complex, real-time web navigation tasks in enterprise environments. However, existing web agents relying on supervised fine-tuning (SFT) often struggle with generalization and robustness due to insufficient reasoning capabilities when handling the inherently dynamic nature of web interactions. In this study, we introduce WorkForceAgent-R1, an LLM-based web agent trained using a rule-based R1-style reinforcement learning framework designed explicitly to enhance single-step reasoning and planning for business-oriented web navigation tasks. We employ a structured reward function that evaluates both adherence to output formats and correctness of actions, enabling WorkForceAgent-R1 to implicitly learn robust intermediate reasoning without explicit annotations or extensive expert demonstrations. Extensive experiments on the WorkArena benchmark demonstrate that WorkForceAgent-R1 substantially outperforms SFT baselines by 10.26-16.59%, achieving competitive performance relative to proprietary LLM-based agents (gpt-4o) in workplace-oriented web navigation tasks. 

---
# Scalable Parameter and Memory Efficient Pretraining for LLM: Recent Algorithmic Advances and Benchmarking 

**Authors**: Athanasios Glentis, Jiaxiang Li, Qiulin Shang, Andi Han, Ioannis Tsaknakis, Quan Wei, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2505.22922)  

**Abstract**: Fueled by their remarkable ability to tackle diverse tasks across multiple domains, large language models (LLMs) have grown at an unprecedented rate, with some recent models containing trillions of parameters. This growth is accompanied by substantial computational challenges, particularly regarding the memory and compute resources required for training and fine-tuning. Numerous approaches have been explored to address these issues, such as LoRA. While these methods are effective for fine-tuning, their application to pre-training is significantly more challenging due to the need to learn vast datasets. Motivated by this issue, we aim to address the following questions: Can parameter- or memory-efficient methods enhance pre-training efficiency while achieving performance comparable to full-model training? How can the performance gap be narrowed? To this end, the contributions of this work are the following. (1) We begin by conducting a comprehensive survey that summarizes state-of-the-art methods for efficient pre-training. (2) We perform a benchmark evaluation of several representative memory efficient pre-training approaches to comprehensively evaluate their performance across model sizes. We observe that with a proper choice of optimizer and hyperparameters, full-rank training delivers the best performance, as expected. We also notice that incorporating high-rank updates in low-rank approaches is the key to improving their performance. (3) Finally, we propose two practical techniques, namely weight refactorization and momentum reset, to enhance the performance of efficient pre-training methods. We observe that applying these techniques to the low-rank method (on a 1B model) can achieve a lower perplexity than popular memory efficient algorithms such as GaLore and Fira, while simultaneously using about 25% less memory. 

---
# Generative Social Choice: The Next Generation 

**Authors**: Niclas Boehmer, Sara Fish, Ariel D. Procaccia  

**Link**: [PDF](https://arxiv.org/pdf/2505.22939)  

**Abstract**: A key task in certain democratic processes is to produce a concise slate of statements that proportionally represents the full spectrum of user opinions. This task is similar to committee elections, but unlike traditional settings, the candidate set comprises all possible statements of varying lengths, and so it can only be accessed through specific queries. Combining social choice and large language models, prior work has approached this challenge through a framework of generative social choice. We extend the framework in two fundamental ways, providing theoretical guarantees even in the face of approximately optimal queries and a budget limit on the overall length of the slate. Using GPT-4o to implement queries, we showcase our approach on datasets related to city improvement measures and drug reviews, demonstrating its effectiveness in generating representative slates from unstructured user opinions. 

---
# BugWhisperer: Fine-Tuning LLMs for SoC Hardware Vulnerability Detection 

**Authors**: Shams Tarek, Dipayan Saha, Sujan Kumar Saha, Farimah Farahmandi  

**Link**: [PDF](https://arxiv.org/pdf/2505.22878)  

**Abstract**: The current landscape of system-on-chips (SoCs) security verification faces challenges due to manual, labor-intensive, and inflexible methodologies. These issues limit the scalability and effectiveness of security protocols, making bug detection at the Register-Transfer Level (RTL) difficult. This paper proposes a new framework named BugWhisperer that utilizes a specialized, fine-tuned Large Language Model (LLM) to address these challenges. By enhancing the LLM's hardware security knowledge and leveraging its capabilities for text inference and knowledge transfer, this approach automates and improves the adaptability and reusability of the verification process. We introduce an open-source, fine-tuned LLM specifically designed for detecting security vulnerabilities in SoC designs. Our findings demonstrate that this tailored LLM effectively enhances the efficiency and flexibility of the security verification process. Additionally, we introduce a comprehensive hardware vulnerability database that supports this work and will further assist the research community in enhancing the security verification process. 

---
# Can LLMs Deceive CLIP? Benchmarking Adversarial Compositionality of Pre-trained Multimodal Representation via Text Updates 

**Authors**: Jaewoo Ahn, Heeseung Yun, Dayoon Ko, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.22943)  

**Abstract**: While pre-trained multimodal representations (e.g., CLIP) have shown impressive capabilities, they exhibit significant compositional vulnerabilities leading to counterintuitive judgments. We introduce Multimodal Adversarial Compositionality (MAC), a benchmark that leverages large language models (LLMs) to generate deceptive text samples to exploit these vulnerabilities across different modalities and evaluates them through both sample-wise attack success rate and group-wise entropy-based diversity. To improve zero-shot methods, we propose a self-training approach that leverages rejection-sampling fine-tuning with diversity-promoting filtering, which enhances both attack success rate and sample diversity. Using smaller language models like Llama-3.1-8B, our approach demonstrates superior performance in revealing compositional vulnerabilities across various multimodal representations, including images, videos, and audios. 

---
# Operationalizing CaMeL: Strengthening LLM Defenses for Enterprise Deployment 

**Authors**: Krti Tallam, Emma Miller  

**Link**: [PDF](https://arxiv.org/pdf/2505.22852)  

**Abstract**: CaMeL (Capabilities for Machine Learning) introduces a capability-based sandbox to mitigate prompt injection attacks in large language model (LLM) agents. While effective, CaMeL assumes a trusted user prompt, omits side-channel concerns, and incurs performance tradeoffs due to its dual-LLM design. This response identifies these issues and proposes engineering improvements to expand CaMeL's threat coverage and operational usability. We introduce: (1) prompt screening for initial inputs, (2) output auditing to detect instruction leakage, (3) a tiered-risk access model to balance usability and control, and (4) a verified intermediate language for formal guarantees. Together, these upgrades align CaMeL with best practices in enterprise security and support scalable deployment. 

---
# A Tool for Generating Exceptional Behavior Tests With Large Language Models 

**Authors**: Linghan Zhong, Samuel Yuan, Jiyang Zhang, Yu Liu, Pengyu Nie, Junyi Jessy Li, Milos Gligoric  

**Link**: [PDF](https://arxiv.org/pdf/2505.22818)  

**Abstract**: Exceptional behavior tests (EBTs) are crucial in software development for verifying that code correctly handles unwanted events and throws appropriate exceptions. However, prior research has shown that developers often prioritize testing "happy paths", e.g., paths without unwanted events over exceptional scenarios. We present exLong, a framework that automatically generates EBTs to address this gap. exLong leverages a large language model (LLM) fine-tuned from CodeLlama and incorporates reasoning about exception-throwing traces, conditional expressions that guard throw statements, and non-exceptional behavior tests that execute similar traces. Our demonstration video illustrates how exLong can effectively assist developers in creating comprehensive EBTs for their project (available at this https URL). 

---
# What Has Been Lost with Synthetic Evaluation? 

**Authors**: Alexander Gill, Abhilasha Ravichander, Ana Marasović  

**Link**: [PDF](https://arxiv.org/pdf/2505.22830)  

**Abstract**: Large language models (LLMs) are increasingly used for data generation. However, creating evaluation benchmarks raises the bar for this emerging paradigm. Benchmarks must target specific phenomena, penalize exploiting shortcuts, and be challenging. Through two case studies, we investigate whether LLMs can meet these demands by generating reasoning over-text benchmarks and comparing them to those created through careful crowdsourcing. Specifically, we evaluate both the validity and difficulty of LLM-generated versions of two high-quality reading comprehension datasets: CondaQA, which evaluates reasoning about negation, and DROP, which targets reasoning about quantities. We find that prompting LLMs can produce variants of these datasets that are often valid according to the annotation guidelines, at a fraction of the cost of the original crowdsourcing effort. However, we show that they are less challenging for LLMs than their human-authored counterparts. This finding sheds light on what may have been lost by generating evaluation data with LLMs, and calls for critically reassessing the immediate use of this increasingly prevalent approach to benchmark creation. 

---
# Permissioned LLMs: Enforcing Access Control in Large Language Models 

**Authors**: Bargav Jayaraman, Virendra J. Marathe, Hamid Mozaffari, William F. Shen, Krishnaram Kenthapadi  

**Link**: [PDF](https://arxiv.org/pdf/2505.22860)  

**Abstract**: In enterprise settings, organizational data is segregated, siloed and carefully protected by elaborate access control frameworks. These access control structures can completely break down if an LLM fine-tuned on the siloed data serves requests, for downstream tasks, from individuals with disparate access privileges. We propose Permissioned LLMs (PermLLM), a new class of LLMs that superimpose the organizational data access control structures on query responses they generate. We formalize abstractions underpinning the means to determine whether access control enforcement happens correctly over LLM query responses. Our formalism introduces the notion of a relevant response that can be used to prove whether a PermLLM mechanism has been implemented correctly. We also introduce a novel metric, called access advantage, to empirically evaluate the efficacy of a PermLLM mechanism. We introduce three novel PermLLM mechanisms that build on Parameter Efficient Fine-Tuning to achieve the desired access control. We furthermore present two instantiations of access advantage--(i) Domain Distinguishability Index (DDI) based on Membership Inference Attacks, and (ii) Utility Gap Index (UGI) based on LLM utility evaluation. We demonstrate the efficacy of our PermLLM mechanisms through extensive experiments on four public datasets (GPQA, RCV1, SimpleQA, and WMDP), in addition to evaluating the validity of DDI and UGI metrics themselves for quantifying access control in LLMs. 

---
# In Dialogue with Intelligence: Rethinking Large Language Models as Collective Knowledge 

**Authors**: Eleni Vasilaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.22767)  

**Abstract**: Large Language Models (LLMs) are typically analysed through architectural, behavioural, or training-data lenses. This article offers a theoretical and experiential re-framing: LLMs as dynamic instantiations of Collective human Knowledge (CK), where intelligence is evoked through dialogue rather than stored statically. Drawing on concepts from neuroscience and AI, and grounded in sustained interaction with ChatGPT-4, I examine emergent dialogue patterns, the implications of fine-tuning, and the notion of co-augmentation: mutual enhancement between human and machine cognition. This perspective offers a new lens for understanding interaction, representation, and agency in contemporary AI systems. 

---
# A Large Language Model-Enabled Control Architecture for Dynamic Resource Capability Exploration in Multi-Agent Manufacturing Systems 

**Authors**: Jonghan Lim, Ilya Kovalenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.22814)  

**Abstract**: Manufacturing environments are becoming more complex and unpredictable due to factors such as demand variations and shorter product lifespans. This complexity requires real-time decision-making and adaptation to disruptions. Traditional control approaches highlight the need for advanced control strategies capable of overcoming unforeseen challenges, as they demonstrate limitations in responsiveness within dynamic industrial settings. Multi-agent systems address these challenges through decentralization of decision-making, enabling systems to respond dynamically to operational changes. However, current multi-agent systems encounter challenges related to real-time adaptation, context-aware decision-making, and the dynamic exploration of resource capabilities. Large language models provide the possibility to overcome these limitations through context-aware decision-making capabilities. This paper introduces a large language model-enabled control architecture for multi-agent manufacturing systems to dynamically explore resource capabilities in response to real-time disruptions. A simulation-based case study demonstrates that the proposed architecture improves system resilience and flexibility. The case study findings show improved throughput and efficient resource utilization compared to existing approaches. 

---
# Automated Essay Scoring Incorporating Annotations from Automated Feedback Systems 

**Authors**: Christopher Ormerod  

**Link**: [PDF](https://arxiv.org/pdf/2505.22771)  

**Abstract**: This study illustrates how incorporating feedback-oriented annotations into the scoring pipeline can enhance the accuracy of automated essay scoring (AES). This approach is demonstrated with the Persuasive Essays for Rating, Selecting, and Understanding Argumentative and Discourse Elements (PERSUADE) corpus. We integrate two types of feedback-driven annotations: those that identify spelling and grammatical errors, and those that highlight argumentative components. To illustrate how this method could be applied in real-world scenarios, we employ two LLMs to generate annotations -- a generative language model used for spell-correction and an encoder-based token classifier trained to identify and mark argumentative elements. By incorporating annotations into the scoring process, we demonstrate improvements in performance using encoder-based large language models fine-tuned as classifiers. 

---
# Training Language Models to Generate Quality Code with Program Analysis Feedback 

**Authors**: Feng Yao, Zilong Wang, Liyuan Liu, Junxia Cui, Li Zhong, Xiaohan Fu, Haohui Mai, Vish Krishnan, Jianfeng Gao, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22704)  

**Abstract**: Code generation with large language models (LLMs), often termed vibe coding, is increasingly adopted in production but fails to ensure code quality, particularly in security (e.g., SQL injection vulnerabilities) and maintainability (e.g., missing type annotations). Existing methods, such as supervised fine-tuning and rule-based post-processing, rely on labor-intensive annotations or brittle heuristics, limiting their scalability and effectiveness. We propose REAL, a reinforcement learning framework that incentivizes LLMs to generate production-quality code using program analysis-guided feedback. Specifically, REAL integrates two automated signals: (1) program analysis detecting security or maintainability defects and (2) unit tests ensuring functional correctness. Unlike prior work, our framework is prompt-agnostic and reference-free, enabling scalable supervision without manual intervention. Experiments across multiple datasets and model scales demonstrate that REAL outperforms state-of-the-art methods in simultaneous assessments of functionality and code quality. Our work bridges the gap between rapid prototyping and production-ready code, enabling LLMs to deliver both speed and quality. 

---
# Don't Take the Premise for Granted: Evaluating the Premise Critique Ability of Large Language Models 

**Authors**: Jinzhe Li, Gengxu Li, Yi Chang, Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.23715)  

**Abstract**: Large language models (LLMs) have witnessed rapid advancements, demonstrating remarkable capabilities. However, a notable vulnerability persists: LLMs often uncritically accept flawed or contradictory premises, leading to inefficient reasoning and unreliable outputs. This emphasizes the significance of possessing the \textbf{Premise Critique Ability} for LLMs, defined as the capacity to proactively identify and articulate errors in input premises. Most existing studies assess LLMs' reasoning ability in ideal settings, largely ignoring their vulnerabilities when faced with flawed premises. Thus, we introduce the \textbf{Premise Critique Bench (PCBench)}, designed by incorporating four error types across three difficulty levels, paired with multi-faceted evaluation metrics. We conducted systematic evaluations of 15 representative LLMs. Our findings reveal: (1) Most models rely heavily on explicit prompts to detect errors, with limited autonomous critique; (2) Premise critique ability depends on question difficulty and error type, with direct contradictions being easier to detect than complex or procedural errors; (3) Reasoning ability does not consistently correlate with the premise critique ability; (4) Flawed premises trigger overthinking in reasoning models, markedly lengthening responses due to repeated attempts at resolving conflicts. These insights underscore the urgent need to enhance LLMs' proactive evaluation of input validity, positioning premise critique as a foundational capability for developing reliable, human-centric systems. The code is available at this https URL. 

---
# SocialMaze: A Benchmark for Evaluating Social Reasoning in Large Language Models 

**Authors**: Zixiang Xu, Yanbo Wang, Yue Huang, Jiayi Ye, Haomin Zhuang, Zirui Song, Lang Gao, Chenxi Wang, Zhaorun Chen, Yujun Zhou, Sixian Li, Wang Pan, Yue Zhao, Jieyu Zhao, Xiangliang Zhang, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.23713)  

**Abstract**: Large language models (LLMs) are increasingly applied to socially grounded tasks, such as online community moderation, media content analysis, and social reasoning games. Success in these contexts depends on a model's social reasoning ability - the capacity to interpret social contexts, infer others' mental states, and assess the truthfulness of presented information. However, there is currently no systematic evaluation framework that comprehensively assesses the social reasoning capabilities of LLMs. Existing efforts often oversimplify real-world scenarios and consist of tasks that are too basic to challenge advanced models. To address this gap, we introduce SocialMaze, a new benchmark specifically designed to evaluate social reasoning. SocialMaze systematically incorporates three core challenges: deep reasoning, dynamic interaction, and information uncertainty. It provides six diverse tasks across three key settings: social reasoning games, daily-life interactions, and digital community platforms. Both automated and human validation are used to ensure data quality. Our evaluation reveals several key insights: models vary substantially in their ability to handle dynamic interactions and integrate temporally evolving information; models with strong chain-of-thought reasoning perform better on tasks requiring deeper inference beyond surface-level cues; and model reasoning degrades significantly under uncertainty. Furthermore, we show that targeted fine-tuning on curated reasoning examples can greatly improve model performance in complex social scenarios. The dataset is publicly available at: this https URL 

---
# Can LLMs Reason Abstractly Over Math Word Problems Without CoT? Disentangling Abstract Formulation From Arithmetic Computation 

**Authors**: Ziling Cheng, Meng Cao, Leila Pishdad, Yanshuai Cao, Jackie Chi Kit Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2505.23701)  

**Abstract**: Final-answer-based metrics are commonly used for evaluating large language models (LLMs) on math word problems, often taken as proxies for reasoning ability. However, such metrics conflate two distinct sub-skills: abstract formulation (capturing mathematical relationships using expressions) and arithmetic computation (executing the calculations). Through a disentangled evaluation on GSM8K and SVAMP, we find that the final-answer accuracy of Llama-3 and Qwen2.5 (1B-32B) without CoT is overwhelmingly bottlenecked by the arithmetic computation step and not by the abstract formulation step. Contrary to the common belief, we show that CoT primarily aids in computation, with limited impact on abstract formulation. Mechanistically, we show that these two skills are composed conjunctively even in a single forward pass without any reasoning steps via an abstract-then-compute mechanism: models first capture problem abstractions, then handle computation. Causal patching confirms these abstractions are present, transferable, composable, and precede computation. These behavioural and mechanistic findings highlight the need for disentangled evaluation to accurately assess LLM reasoning and to guide future improvements. 

---
# ToolHaystack: Stress-Testing Tool-Augmented Language Models in Realistic Long-Term Interactions 

**Authors**: Beong-woo Kwak, Minju Kim, Dongha Lim, Hyungjoo Chae, Dongjin Kang, Sunghwan Kim, Dongil Yang, Jinyoung Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2505.23662)  

**Abstract**: Large language models (LLMs) have demonstrated strong capabilities in using external tools to address user inquiries. However, most existing evaluations assume tool use in short contexts, offering limited insight into model behavior during realistic long-term interactions. To fill this gap, we introduce ToolHaystack, a benchmark for testing the tool use capabilities in long-term interactions. Each test instance in ToolHaystack includes multiple tasks execution contexts and realistic noise within a continuous conversation, enabling assessment of how well models maintain context and handle various disruptions. By applying this benchmark to 14 state-of-the-art LLMs, we find that while current models perform well in standard multi-turn settings, they often significantly struggle in ToolHaystack, highlighting critical gaps in their long-term robustness not revealed by previous tool benchmarks. 

---
# Label-Guided In-Context Learning for Named Entity Recognition 

**Authors**: Fan Bai, Hamid Hassanzadeh, Ardavan Saeedi, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2505.23722)  

**Abstract**: In-context learning (ICL) enables large language models (LLMs) to perform new tasks using only a few demonstrations. In Named Entity Recognition (NER), demonstrations are typically selected based on semantic similarity to the test instance, ignoring training labels and resulting in suboptimal performance. We introduce DEER, a new method that leverages training labels through token-level statistics to improve ICL performance. DEER first enhances example selection with a label-guided, token-based retriever that prioritizes tokens most informative for entity recognition. It then prompts the LLM to revisit error-prone tokens, which are also identified using label statistics, and make targeted corrections. Evaluated on five NER datasets using four different LLMs, DEER consistently outperforms existing ICL methods and approaches the performance of supervised fine-tuning. Further analysis shows its effectiveness on both seen and unseen entities and its robustness in low-resource settings. 

---
# Table-R1: Inference-Time Scaling for Table Reasoning 

**Authors**: Zheyuan Yang, Lyuhao Chen, Arman Cohan, Yilun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.23621)  

**Abstract**: In this work, we present the first study to explore inference-time scaling on table reasoning tasks. We develop and evaluate two post-training strategies to enable inference-time scaling: distillation from frontier model reasoning traces and reinforcement learning with verifiable rewards (RLVR). For distillation, we introduce a large-scale dataset of reasoning traces generated by DeepSeek-R1, which we use to fine-tune LLMs into the Table-R1-SFT model. For RLVR, we propose task-specific verifiable reward functions and apply the GRPO algorithm to obtain the Table-R1-Zero model. We evaluate our Table-R1-series models across diverse table reasoning tasks, including short-form QA, fact verification, and free-form QA. Notably, the Table-R1-Zero model matches or exceeds the performance of GPT-4.1 and DeepSeek-R1, while using only a 7B-parameter LLM. It also demonstrates strong generalization to out-of-domain datasets. Extensive ablation and qualitative analyses reveal the benefits of instruction tuning, model architecture choices, and cross-task generalization, as well as emergence of essential table reasoning skills during RL training. 

---
# Translation in the Wild 

**Authors**: Yuri Balashov  

**Link**: [PDF](https://arxiv.org/pdf/2505.23548)  

**Abstract**: Large Language Models (LLMs) excel in translation among other things, demonstrating competitive performance for many language pairs in zero- and few-shot settings. But unlike dedicated neural machine translation models, LLMs are not trained on any translation-related objective. What explains their remarkable translation abilities? Are these abilities grounded in "incidental bilingualism" (Briakou et al. 2023) in training data? Does instruction tuning contribute to it? Are LLMs capable of aligning and leveraging semantically identical or similar monolingual contents from different corners of the internet that are unlikely to fit in a single context window? I offer some reflections on this topic, informed by recent studies and growing user experience. My working hypothesis is that LLMs' translation abilities originate in two different types of pre-training data that may be internalized by the models in different ways. I discuss the prospects for testing the "duality" hypothesis empirically and its implications for reconceptualizing translation, human and machine, in the age of deep learning. 

---
# Probability-Consistent Preference Optimization for Enhanced LLM Reasoning 

**Authors**: Yunqiao Yang, Houxing Ren, Zimu Lu, Ke Wang, Weikang Shi, Aojun Zhou, Junting Pan, Mingjie Zhan, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.23540)  

**Abstract**: Recent advances in preference optimization have demonstrated significant potential for improving mathematical reasoning capabilities in large language models (LLMs). While current approaches leverage high-quality pairwise preference data through outcome-based criteria like answer correctness or consistency, they fundamentally neglect the internal logical coherence of responses. To overcome this, we propose Probability-Consistent Preference Optimization (PCPO), a novel framework that establishes dual quantitative metrics for preference selection: (1) surface-level answer correctness and (2) intrinsic token-level probability consistency across responses. Extensive experiments show that our PCPO consistently outperforms existing outcome-only criterion approaches across a diverse range of LLMs and benchmarks. Our code is publicly available at this https URL. 

---
# Revisiting Overthinking in Long Chain-of-Thought from the Perspective of Self-Doubt 

**Authors**: Keqin Peng, Liang Ding, Yuanxin Ouyang, Meng Fang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.23480)  

**Abstract**: Reasoning Large Language Models (RLLMs) have demonstrated impressive performance on complex tasks, largely due to the adoption of Long Chain-of-Thought (Long CoT) reasoning. However, they often exhibit overthinking -- performing unnecessary reasoning steps even after arriving at the correct answer. Prior work has largely focused on qualitative analyses of overthinking through sample-based observations of long CoTs. In contrast, we present a quantitative analysis of overthinking from the perspective of self-doubt, characterized by excessive token usage devoted to re-verifying already-correct answer. We find that self-doubt significantly contributes to overthinking. In response, we introduce a simple and effective prompting method to reduce the model's over-reliance on input questions, thereby avoiding self-doubt. Specifically, we first prompt the model to question the validity of the input question, and then respond concisely based on the outcome of that evaluation. Experiments on three mathematical reasoning tasks and four datasets with missing premises demonstrate that our method substantially reduces answer length and yields significant improvements across nearly all datasets upon 4 widely-used RLLMs. Further analysis demonstrates that our method effectively minimizes the number of reasoning steps and reduces self-doubt. 

---
# UAQFact: Evaluating Factual Knowledge Utilization of LLMs on Unanswerable Questions 

**Authors**: Chuanyuan Tan, Wenbiao Shao, Hao Xiong, Tong Zhu, Zhenhua Liu, Kai Shi, Wenliang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.23461)  

**Abstract**: Handling unanswerable questions (UAQ) is crucial for LLMs, as it helps prevent misleading responses in complex situations. While previous studies have built several datasets to assess LLMs' performance on UAQ, these datasets lack factual knowledge support, which limits the evaluation of LLMs' ability to utilize their factual knowledge when handling UAQ. To address the limitation, we introduce a new unanswerable question dataset UAQFact, a bilingual dataset with auxiliary factual knowledge created from a Knowledge Graph. Based on UAQFact, we further define two new tasks to measure LLMs' ability to utilize internal and external factual knowledge, respectively. Our experimental results across multiple LLM series show that UAQFact presents significant challenges, as LLMs do not consistently perform well even when they have factual knowledge stored. Additionally, we find that incorporating external knowledge may enhance performance, but LLMs still cannot make full use of the knowledge which may result in incorrect responses. 

---
# From Parameters to Prompts: Understanding and Mitigating the Factuality Gap between Fine-Tuned LLMs 

**Authors**: Xuan Gong, Hanbo Huang, Shiyu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23410)  

**Abstract**: Factual knowledge extraction aims to explicitly extract knowledge parameterized in pre-trained language models for application in downstream tasks. While prior work has been investigating the impact of supervised fine-tuning data on the factuality of large language models (LLMs), its mechanism remains poorly understood. We revisit this impact through systematic experiments, with a particular focus on the factuality gap that arises when fine-tuning on known versus unknown knowledge. Our findings show that this gap can be mitigated at the inference stage, either under out-of-distribution (OOD) settings or by using appropriate in-context learning (ICL) prompts (i.e., few-shot learning and Chain of Thought (CoT)). We prove this phenomenon theoretically from the perspective of knowledge graphs, showing that the test-time prompt may diminish or even overshadow the impact of fine-tuning data and play a dominant role in knowledge extraction. Ultimately, our results shed light on the interaction between finetuning data and test-time prompt, demonstrating that ICL can effectively compensate for shortcomings in fine-tuning data, and highlighting the need to reconsider the use of ICL prompting as a means to evaluate the effectiveness of fine-tuning data selection methods. 

---
# Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking 

**Authors**: Liangliang Zhang, Zhuorui Jiang, Hongliang Chi, Haoyang Chen, Mohammed Elkoumy, Fali Wang, Qiong Wu, Zhengyi Zhou, Shirui Pan, Suhang Wang, Yao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.23495)  

**Abstract**: Knowledge Graph Question Answering (KGQA) systems rely on high-quality benchmarks to evaluate complex multi-hop reasoning. However, despite their widespread use, popular datasets such as WebQSP and CWQ suffer from critical quality issues, including inaccurate or incomplete ground-truth annotations, poorly constructed questions that are ambiguous, trivial, or unanswerable, and outdated or inconsistent knowledge. Through a manual audit of 16 popular KGQA datasets, including WebQSP and CWQ, we find that the average factual correctness rate is only 57 %. To address these issues, we introduce KGQAGen, an LLM-in-the-loop framework that systematically resolves these pitfalls. KGQAGen combines structured knowledge grounding, LLM-guided generation, and symbolic verification to produce challenging and verifiable QA instances. Using KGQAGen, we construct KGQAGen-10k, a ten-thousand scale benchmark grounded in Wikidata, and evaluate a diverse set of KG-RAG models. Experimental results demonstrate that even state-of-the-art systems struggle on this benchmark, highlighting its ability to expose limitations of existing models. Our findings advocate for more rigorous benchmark construction and position KGQAGen as a scalable framework for advancing KGQA evaluation. 

---
# Evaluating AI capabilities in detecting conspiracy theories on YouTube 

**Authors**: Leonardo La Rocca, Francesco Corso, Francesco Pierri  

**Link**: [PDF](https://arxiv.org/pdf/2505.23570)  

**Abstract**: As a leading online platform with a vast global audience, YouTube's extensive reach also makes it susceptible to hosting harmful content, including disinformation and conspiracy theories. This study explores the use of open-weight Large Language Models (LLMs), both text-only and multimodal, for identifying conspiracy theory videos shared on YouTube. Leveraging a labeled dataset of thousands of videos, we evaluate a variety of LLMs in a zero-shot setting and compare their performance to a fine-tuned RoBERTa baseline. Results show that text-based LLMs achieve high recall but lower precision, leading to increased false positives. Multimodal models lag behind their text-only counterparts, indicating limited benefits from visual data integration. To assess real-world applicability, we evaluate the most accurate models on an unlabeled dataset, finding that RoBERTa achieves performance close to LLMs with a larger number of parameters. Our work highlights the strengths and limitations of current LLM-based approaches for online harmful content detection, emphasizing the need for more precise and robust systems. 

---
# Generalized Category Discovery in Event-Centric Contexts: Latent Pattern Mining with LLMs 

**Authors**: Yi Luo, Qiwen Wang, Junqi Yang, Luyao Tang, Zhenghao Lin, Zhenzhe Ying, Weiqiang Wang, Chen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.23304)  

**Abstract**: Generalized Category Discovery (GCD) aims to classify both known and novel categories using partially labeled data that contains only known classes. Despite achieving strong performance on existing benchmarks, current textual GCD methods lack sufficient validation in realistic settings. We introduce Event-Centric GCD (EC-GCD), characterized by long, complex narratives and highly imbalanced class distributions, posing two main challenges: (1) divergent clustering versus classification groupings caused by subjective criteria, and (2) Unfair alignment for minority classes. To tackle these, we propose PaMA, a framework leveraging LLMs to extract and refine event patterns for improved cluster-class alignment. Additionally, a ranking-filtering-mining pipeline ensures balanced representation of prototypes across imbalanced categories. Evaluations on two EC-GCD benchmarks, including a newly constructed Scam Report dataset, demonstrate that PaMA outperforms prior methods with up to 12.58% H-score gains, while maintaining strong generalization on base GCD datasets. 

---
# Evaluating the performance and fragility of large language models on the self-assessment for neurological surgeons 

**Authors**: Krithik Vishwanath, Anton Alyakin, Mrigayu Ghosh, Jin Vivian Lee, Daniel Alexander Alber, Karl L. Sangwon, Douglas Kondziolka, Eric Karl Oermann  

**Link**: [PDF](https://arxiv.org/pdf/2505.23477)  

**Abstract**: The Congress of Neurological Surgeons Self-Assessment for Neurological Surgeons (CNS-SANS) questions are widely used by neurosurgical residents to prepare for written board examinations. Recently, these questions have also served as benchmarks for evaluating large language models' (LLMs) neurosurgical knowledge. This study aims to assess the performance of state-of-the-art LLMs on neurosurgery board-like questions and to evaluate their robustness to the inclusion of distractor statements. A comprehensive evaluation was conducted using 28 large language models. These models were tested on 2,904 neurosurgery board examination questions derived from the CNS-SANS. Additionally, the study introduced a distraction framework to assess the fragility of these models. The framework incorporated simple, irrelevant distractor statements containing polysemous words with clinical meanings used in non-clinical contexts to determine the extent to which such distractions degrade model performance on standard medical benchmarks. 6 of the 28 tested LLMs achieved board-passing outcomes, with the top-performing models scoring over 15.7% above the passing threshold. When exposed to distractions, accuracy across various model architectures was significantly reduced-by as much as 20.4%-with one model failing that had previously passed. Both general-purpose and medical open-source models experienced greater performance declines compared to proprietary variants when subjected to the added distractors. While current LLMs demonstrate an impressive ability to answer neurosurgery board-like exam questions, their performance is markedly vulnerable to extraneous, distracting information. These findings underscore the critical need for developing novel mitigation strategies aimed at bolstering LLM resilience against in-text distractions, particularly for safe and effective clinical deployment. 

---
# Sentinel: Attention Probing of Proxy Models for LLM Context Compression with an Understanding Perspective 

**Authors**: Yong Zhang, Yanwen Huang, Ning Cheng, Yang Guo, Yun Zhu, Yanmeng Wang, Shaojun Wang, Jing Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.23277)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) with external context, but retrieved passages are often lengthy, noisy, or exceed input limits. Existing compression methods typically require supervised training of dedicated compression models, increasing cost and reducing portability. We propose Sentinel, a lightweight sentence-level compression framework that reframes context filtering as an attention-based understanding task. Rather than training a compression model, Sentinel probes decoder attention from an off-the-shelf 0.5B proxy LLM using a lightweight classifier to identify sentence relevance. Empirically, we find that query-context relevance estimation is consistent across model scales, with 0.5B proxies closely matching the behaviors of larger models. On the LongBench benchmark, Sentinel achieves up to 5$\times$ compression while matching the QA performance of 7B-scale compression systems. Our results suggest that probing native attention signals enables fast, effective, and question-aware context compression. Code available at: this https URL. 

---
# Threading the Needle: Reweaving Chain-of-Thought Reasoning to Explain Human Label Variation 

**Authors**: Beiduo Chen, Yang Janet Liu, Anna Korhonen, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2505.23368)  

**Abstract**: The recent rise of reasoning-tuned Large Language Models (LLMs)--which generate chains of thought (CoTs) before giving the final answer--has attracted significant attention and offers new opportunities for gaining insights into human label variation, which refers to plausible differences in how multiple annotators label the same data instance. Prior work has shown that LLM-generated explanations can help align model predictions with human label distributions, but typically adopt a reverse paradigm: producing explanations based on given answers. In contrast, CoTs provide a forward reasoning path that may implicitly embed rationales for each answer option, before generating the answers. We thus propose a novel LLM-based pipeline enriched with linguistically-grounded discourse segmenters to extract supporting and opposing statements for each answer option from CoTs with improved accuracy. We also propose a rank-based HLV evaluation framework that prioritizes the ranking of answers over exact scores, which instead favor direct comparison of label distributions. Our method outperforms a direct generation method as well as baselines on three datasets, and shows better alignment of ranking methods with humans, highlighting the effectiveness of our approach. 

---
# ARC: Argument Representation and Coverage Analysis for Zero-Shot Long Document Summarization with Instruction Following LLMs 

**Authors**: Mohamed Elaraby, Diane Litman  

**Link**: [PDF](https://arxiv.org/pdf/2505.23654)  

**Abstract**: Integrating structured information has long improved the quality of abstractive summarization, particularly in retaining salient content. In this work, we focus on a specific form of structure: argument roles, which are crucial for summarizing documents in high-stakes domains such as law. We investigate whether instruction-tuned large language models (LLMs) adequately preserve this information. To this end, we introduce Argument Representation Coverage (ARC), a framework for measuring how well LLM-generated summaries capture salient arguments. Using ARC, we analyze summaries produced by three open-weight LLMs in two domains where argument roles are central: long legal opinions and scientific articles. Our results show that while LLMs cover salient argument roles to some extent, critical information is often omitted in generated summaries, particularly when arguments are sparsely distributed throughout the input. Further, we use ARC to uncover behavioral patterns -- specifically, how the positional bias of LLM context windows and role-specific preferences impact the coverage of key arguments in generated summaries, emphasizing the need for more argument-aware summarization strategies. 

---
# Infinite-Instruct: Synthesizing Scaling Code instruction Data with Bidirectional Synthesis and Static Verification 

**Authors**: Wenjing Xing, Wenke Lu, Yeheng Duan, Bing Zhao, Zhenghui kang, Yaolong Wang, Kai Gao, Lei Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.23177)  

**Abstract**: Traditional code instruction data synthesis methods suffer from limited diversity and poor logic. We introduce Infinite-Instruct, an automated framework for synthesizing high-quality question-answer pairs, designed to enhance the code generation capabilities of large language models (LLMs). The framework focuses on improving the internal logic of synthesized problems and the quality of synthesized code. First, "Reverse Construction" transforms code snippets into diverse programming problems. Then, through "Backfeeding Construction," keywords in programming problems are structured into a knowledge graph to reconstruct them into programming problems with stronger internal logic. Finally, a cross-lingual static code analysis pipeline filters invalid samples to ensure data quality. Experiments show that on mainstream code generation benchmarks, our fine-tuned models achieve an average performance improvement of 21.70% on 7B-parameter models and 36.95% on 32B-parameter models. Using less than one-tenth of the instruction fine-tuning data, we achieved performance comparable to the Qwen-2.5-Coder-Instruct. Infinite-Instruct provides a scalable solution for LLM training in programming. We open-source the datasets used in the experiments, including both unfiltered versions and filtered versions via static analysis. The data are available at this https URL 

---
# Data-efficient Meta-models for Evaluation of Context-based Questions and Answers in LLMs 

**Authors**: Julia Belikova, Konstantin Polev, Rauf Parchiev, Dmitry Simakov  

**Link**: [PDF](https://arxiv.org/pdf/2505.23299)  

**Abstract**: Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems are increasingly deployed in industry applications, yet their reliability remains hampered by challenges in detecting hallucinations. While supervised state-of-the-art (SOTA) methods that leverage LLM hidden states -- such as activation tracing and representation analysis -- show promise, their dependence on extensively annotated datasets limits scalability in real-world applications. This paper addresses the critical bottleneck of data annotation by investigating the feasibility of reducing training data requirements for two SOTA hallucination detection frameworks: Lookback Lens, which analyzes attention head dynamics, and probing-based approaches, which decode internal model representations. We propose a methodology combining efficient classification algorithms with dimensionality reduction techniques to minimize sample size demands while maintaining competitive performance. Evaluations on standardized question-answering RAG benchmarks show that our approach achieves performance comparable to strong proprietary LLM-based baselines with only 250 training samples. These results highlight the potential of lightweight, data-efficient paradigms for industrial deployment, particularly in annotation-constrained scenarios. 

---
# Enhancing Large Language Models'Machine Translation via Dynamic Focus Anchoring 

**Authors**: Qiuyu Ding, Zhiqiang Cao, Hailong Cao, Tiejun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.23140)  

**Abstract**: Large language models have demonstrated exceptional performance across multiple crosslingual NLP tasks, including machine translation (MT). However, persistent challenges remain in addressing context-sensitive units (CSUs), such as polysemous words. These CSUs not only affect the local translation accuracy of LLMs, but also affect LLMs' understanding capability for sentences and tasks, and even lead to translation failure. To address this problem, we propose a simple but effective method to enhance LLMs' MT capabilities by acquiring CSUs and applying semantic focus. Specifically, we dynamically analyze and identify translation challenges, then incorporate them into LLMs in a structured manner to mitigate mistranslations or misunderstandings of CSUs caused by information flattening. Efficiently activate LLMs to identify and apply relevant knowledge from its vast data pool in this way, ensuring more accurate translations for translating difficult terms. On a benchmark dataset of MT, our proposed method achieved competitive performance compared to multiple existing open-sourced MT baseline models. It demonstrates effectiveness and robustness across multiple language pairs, including both similar language pairs and distant language pairs. Notably, the proposed method requires no additional model training and enhances LLMs' performance across multiple NLP tasks with minimal resource consumption. 

---
# Dataset Cartography for Large Language Model Alignment: Mapping and Diagnosing Preference Data 

**Authors**: Seohyeong Lee, Eunwon Kim, Hwaran Lee, Buru Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23114)  

**Abstract**: Human preference data plays a critical role in aligning large language models (LLMs) with human values. However, collecting such data is often expensive and inefficient, posing a significant scalability challenge. To address this, we introduce Alignment Data Map, a GPT-4o-assisted tool for analyzing and diagnosing preference data. Using GPT-4o as a proxy for LLM alignment, we compute alignment scores for LLM-generated responses to instructions from existing preference datasets. These scores are then used to construct an Alignment Data Map based on their mean and variance. Our experiments show that using only 33 percent of the data, specifically samples in the high-mean, low-variance region, achieves performance comparable to or better than using the entire dataset. This finding suggests that the Alignment Data Map can significantly improve data collection efficiency by identifying high-quality samples for LLM alignment without requiring explicit annotations. Moreover, the Alignment Data Map can diagnose existing preference datasets. Our analysis shows that it effectively detects low-impact or potentially misannotated samples. Source code is available online. 

---
# Generating Diverse Training Samples for Relation Extraction with Large Language Models 

**Authors**: Zexuan Li, Hongliang Dai, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.23108)  

**Abstract**: Using Large Language Models (LLMs) to generate training data can potentially be a preferable way to improve zero or few-shot NLP tasks. However, many problems remain to be investigated for this direction. For the task of Relation Extraction (RE), we find that samples generated by directly prompting LLMs may easily have high structural similarities with each other. They tend to use a limited variety of phrasing while expressing the relation between a pair of entities. Therefore, in this paper, we study how to effectively improve the diversity of the training samples generated with LLMs for RE, while also maintaining their correctness. We first try to make the LLMs produce dissimilar samples by directly giving instructions in In-Context Learning (ICL) prompts. Then, we propose an approach to fine-tune LLMs for diversity training sample generation through Direct Preference Optimization (DPO). Our experiments on commonly used RE datasets show that both attempts can improve the quality of the generated training data. We also find that comparing with directly performing RE with an LLM, training a non-LLM RE model with its generated samples may lead to better performance. 

---
# SNS-Bench-VL: Benchmarking Multimodal Large Language Models in Social Networking Services 

**Authors**: Hongcheng Guo, Zheyong Xie, Shaosheng Cao, Boyang Wang, Weiting Liu, Anjie Le, Lei Li, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.23065)  

**Abstract**: With the increasing integration of visual and textual content in Social Networking Services (SNS), evaluating the multimodal capabilities of Large Language Models (LLMs) is crucial for enhancing user experience, content understanding, and platform intelligence. Existing benchmarks primarily focus on text-centric tasks, lacking coverage of the multimodal contexts prevalent in modern SNS ecosystems. In this paper, we introduce SNS-Bench-VL, a comprehensive multimodal benchmark designed to assess the performance of Vision-Language LLMs in real-world social media scenarios. SNS-Bench-VL incorporates images and text across 8 multimodal tasks, including note comprehension, user engagement analysis, information retrieval, and personalized recommendation. It comprises 4,001 carefully curated multimodal question-answer pairs, covering single-choice, multiple-choice, and open-ended tasks. We evaluate over 25 state-of-the-art multimodal LLMs, analyzing their performance across tasks. Our findings highlight persistent challenges in multimodal social context comprehension. We hope SNS-Bench-VL will inspire future research towards robust, context-aware, and human-aligned multimodal intelligence for next-generation social networking services. 

---
# PBEBench: A Multi-Step Programming by Examples Reasoning Benchmark inspired by Historical Linguistics 

**Authors**: Atharva Naik, Darsh Agrawal, Manav Kapadnis, Yuwei An, Yash Mathur, Carolyn Rose, David Mortensen  

**Link**: [PDF](https://arxiv.org/pdf/2505.23126)  

**Abstract**: Recently, long chain of thought (LCoT), Large Language Models (LLMs), have taken the machine learning world by storm with their breathtaking reasoning capabilities. However, are the abstract reasoning abilities of these models general enough for problems of practical importance? Unlike past work, which has focused mainly on math, coding, and data wrangling, we focus on a historical linguistics-inspired inductive reasoning problem, formulated as Programming by Examples. We develop a fully automated pipeline for dynamically generating a benchmark for this task with controllable difficulty in order to tackle scalability and contamination issues to which many reasoning benchmarks are subject. Using our pipeline, we generate a test set with nearly 1k instances that is challenging for all state-of-the-art reasoning LLMs, with the best model (Claude-3.7-Sonnet) achieving a mere 54% pass rate, demonstrating that LCoT LLMs still struggle with a class or reasoning that is ubiquitous in historical linguistics as well as many other domains. 

---
# Query Routing for Retrieval-Augmented Language Models 

**Authors**: Jiarui Zhang, Xiangyu Liu, Yong Hu, Chaoyue Niu, Fan Wu, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.23052)  

**Abstract**: Retrieval-Augmented Generation (RAG) significantly improves the performance of Large Language Models (LLMs) on knowledge-intensive tasks. However, varying response quality across LLMs under RAG necessitates intelligent routing mechanisms, which select the most suitable model for each query from multiple retrieval-augmented LLMs via a dedicated router model. We observe that external documents dynamically affect LLMs' ability to answer queries, while existing routing methods, which rely on static parametric knowledge representations, exhibit suboptimal performance in RAG scenarios. To address this, we formally define the new retrieval-augmented LLM routing problem, incorporating the influence of retrieved documents into the routing framework. We propose RAGRouter, a RAG-aware routing design, which leverages document embeddings and RAG capability embeddings with contrastive learning to capture knowledge representation shifts and enable informed routing decisions. Extensive experiments on diverse knowledge-intensive tasks and retrieval settings show that RAGRouter outperforms the best individual LLM by 3.61% on average and existing routing methods by 3.29%-9.33%. With an extended score-threshold-based mechanism, it also achieves strong performance-efficiency trade-offs under low-latency constraints. 

---
# MMBoundary: Advancing MLLM Knowledge Boundary Awareness through Reasoning Step Confidence Calibration 

**Authors**: Zhitao He, Sandeep Polisetty, Zhiyuan Fan, Yuchen Huang, Shujin Wu, Yi R., Fung  

**Link**: [PDF](https://arxiv.org/pdf/2505.23224)  

**Abstract**: In recent years, multimodal large language models (MLLMs) have made significant progress but continue to face inherent challenges in multimodal reasoning, which requires multi-level (e.g., perception, reasoning) and multi-granular (e.g., multi-step reasoning chain) advanced inferencing. Prior work on estimating model confidence tends to focus on the overall response for training and calibration, but fails to assess confidence in each reasoning step, leading to undesirable hallucination snowballing. In this work, we present MMBoundary, a novel framework that advances the knowledge boundary awareness of MLLMs through reasoning step confidence calibration. To achieve this, we propose to incorporate complementary textual and cross-modal self-rewarding signals to estimate confidence at each step of the MLLM reasoning process. In addition to supervised fine-tuning MLLM on this set of self-rewarded confidence estimation signal for initial confidence expression warm-up, we introduce a reinforcement learning stage with multiple reward functions for further aligning model knowledge and calibrating confidence at each reasoning step, enhancing reasoning chain self-correction. Empirical results show that MMBoundary significantly outperforms existing methods across diverse domain datasets and metrics, achieving an average of 7.5% reduction in multimodal confidence calibration errors and up to 8.3% improvement in task performance. 

---
# Tell, Don't Show: Leveraging Language Models' Abstractive Retellings to Model Literary Themes 

**Authors**: Li Lucy, Camilla Griffiths, Sarah Levine, Jennifer L. Eberhardt, Dorottya Demszky, David Bamman  

**Link**: [PDF](https://arxiv.org/pdf/2505.23166)  

**Abstract**: Conventional bag-of-words approaches for topic modeling, like latent Dirichlet allocation (LDA), struggle with literary text. Literature challenges lexical methods because narrative language focuses on immersive sensory details instead of abstractive description or exposition: writers are advised to "show, don't tell." We propose Retell, a simple, accessible topic modeling approach for literature. Here, we prompt resource-efficient, generative language models (LMs) to tell what passages show, thereby translating narratives' surface forms into higher-level concepts and themes. By running LDA on LMs' retellings of passages, we can obtain more precise and informative topics than by running LDA alone or by directly asking LMs to list topics. To investigate the potential of our method for cultural analytics, we compare our method's outputs to expert-guided annotations in a case study on racial/cultural identity in high school English language arts books. 

---
# EL4NER: Ensemble Learning for Named Entity Recognition via Multiple Small-Parameter Large Language Models 

**Authors**: Yuzhen Xiao, Jiahe Song, Yongxin Xu, Ruizhe Zhang, Yiqi Xiao, Xin Lu, Runchuan Zhu, Bowen Jiang, Junfeng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.23038)  

**Abstract**: In-Context Learning (ICL) technique based on Large Language Models (LLMs) has gained prominence in Named Entity Recognition (NER) tasks for its lower computing resource consumption, less manual labeling overhead, and stronger generalizability. Nevertheless, most ICL-based NER methods depend on large-parameter LLMs: the open-source models demand substantial computational resources for deployment and inference, while the closed-source ones incur high API costs, raise data-privacy concerns, and hinder community collaboration. To address this question, we propose an Ensemble Learning Method for Named Entity Recognition (EL4NER), which aims at aggregating the ICL outputs of multiple open-source, small-parameter LLMs to enhance overall performance in NER tasks at less deployment and inference cost. Specifically, our method comprises three key components. First, we design a task decomposition-based pipeline that facilitates deep, multi-stage ensemble learning. Second, we introduce a novel span-level sentence similarity algorithm to establish an ICL demonstration retrieval mechanism better suited for NER tasks. Third, we incorporate a self-validation mechanism to mitigate the noise introduced during the ensemble process. We evaluated EL4NER on multiple widely adopted NER datasets from diverse domains. Our experimental results indicate that EL4NER surpasses most closed-source, large-parameter LLM-based methods at a lower parameter cost and even attains state-of-the-art (SOTA) performance among ICL-based methods on certain datasets. These results show the parameter efficiency of EL4NER and underscore the feasibility of employing open-source, small-parameter LLMs within the ICL paradigm for NER tasks. 

---
# ToMAP: Training Opponent-Aware LLM Persuaders with Theory of Mind 

**Authors**: Peixuan Han, Zijia Liu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2505.22961)  

**Abstract**: Large language models (LLMs) have shown promising potential in persuasion, but existing works on training LLM persuaders are still preliminary. Notably, while humans are skilled in modeling their opponent's thoughts and opinions proactively and dynamically, current LLMs struggle with such Theory of Mind (ToM) reasoning, resulting in limited diversity and opponent awareness. To address this limitation, we introduce Theory of Mind Augmented Persuader (ToMAP), a novel approach for building more flexible persuader agents by incorporating two theory of mind modules that enhance the persuader's awareness and analysis of the opponent's mental state. Specifically, we begin by prompting the persuader to consider possible objections to the target central claim, and then use a text encoder paired with a trained MLP classifier to predict the opponent's current stance on these counterclaims. Our carefully designed reinforcement learning schema enables the persuader learns how to analyze opponent-related information and utilize it to generate more effective arguments. Experiments show that the ToMAP persuader, while containing only 3B parameters, outperforms much larger baselines, like GPT-4o, with a relative gain of 39.4% across multiple persuadee models and diverse corpora. Notably, ToMAP exhibits complex reasoning chains and reduced repetition during training, which leads to more diverse and effective arguments. The opponent-aware feature of ToMAP also makes it suitable for long conversations and enables it to employ more logical and opponent-aware strategies. These results underscore our method's effectiveness and highlight its potential for developing more persuasive language agents. Code is available at: this https URL. 

---
# LLMs for Argument Mining: Detection, Extraction, and Relationship Classification of pre-defined Arguments in Online Comments 

**Authors**: Matteo Guida, Yulia Otmakhova, Eduard Hovy, Lea Frermann  

**Link**: [PDF](https://arxiv.org/pdf/2505.22956)  

**Abstract**: Automated large-scale analysis of public discussions around contested issues like abortion requires detecting and understanding the use of arguments. While Large Language Models (LLMs) have shown promise in language processing tasks, their performance in mining topic-specific, pre-defined arguments in online comments remains underexplored. We evaluate four state-of-the-art LLMs on three argument mining tasks using datasets comprising over 2,000 opinion comments across six polarizing topics. Quantitative evaluation suggests an overall strong performance across the three tasks, especially for large and fine-tuned LLMs, albeit at a significant environmental cost. However, a detailed error analysis revealed systematic shortcomings on long and nuanced comments and emotionally charged language, raising concerns for downstream applications like content moderation or opinion analysis. Our results highlight both the promise and current limitations of LLMs for automated argument analysis in online comments. 

---
# Improving Multilingual Social Media Insights: Aspect-based Comment Analysis 

**Authors**: Longyin Zhang, Bowei Zou, Ai Ti Aw  

**Link**: [PDF](https://arxiv.org/pdf/2505.23037)  

**Abstract**: The inherent nature of social media posts, characterized by the freedom of language use with a disjointed array of diverse opinions and topics, poses significant challenges to downstream NLP tasks such as comment clustering, comment summarization, and social media opinion analysis. To address this, we propose a granular level of identifying and generating aspect terms from individual comments to guide model attention. Specifically, we leverage multilingual large language models with supervised fine-tuning for comment aspect term generation (CAT-G), further aligning the model's predictions with human expectations through DPO. We demonstrate the effectiveness of our method in enhancing the comprehension of social media discourse on two NLP tasks. Moreover, this paper contributes the first multilingual CAT-G test set on English, Chinese, Malay, and Bahasa Indonesian. As LLM capabilities vary among languages, this test set allows for a comparative analysis of performance across languages with varying levels of LLM proficiency. 

---
# Structured Memory Mechanisms for Stable Context Representation in Large Language Models 

**Authors**: Yue Xing, Tao Yang, Yijiashun Qi, Minggu Wei, Yu Cheng, Honghui Xin  

**Link**: [PDF](https://arxiv.org/pdf/2505.22921)  

**Abstract**: This paper addresses the limitations of large language models in understanding long-term context. It proposes a model architecture equipped with a long-term memory mechanism to improve the retention and retrieval of semantic information across paragraphs and dialogue turns. The model integrates explicit memory units, gated writing mechanisms, and attention-based reading modules. A forgetting function is introduced to enable dynamic updates of memory content, enhancing the model's ability to manage historical information. To further improve the effectiveness of memory operations, the study designs a joint training objective. This combines the main task loss with constraints on memory writing and forgetting. It guides the model to learn better memory strategies during task execution. Systematic evaluation across multiple subtasks shows that the model achieves clear advantages in text generation consistency, stability in multi-turn question answering, and accuracy in cross-context reasoning. In particular, the model demonstrates strong semantic retention and contextual coherence in long-text tasks and complex question answering scenarios. It effectively mitigates the context loss and semantic drift problems commonly faced by traditional language models when handling long-term dependencies. The experiments also include analysis of different memory structures, capacity sizes, and control strategies. These results further confirm the critical role of memory mechanisms in language understanding. They demonstrate the feasibility and effectiveness of the proposed approach in both architectural design and performance outcomes. 

---
# LLM-based HSE Compliance Assessment: Benchmark, Performance, and Advancements 

**Authors**: Jianwei Wang, Mengqi Wang, Yinsi Zhou, Zhenchang Xing, Qing Liu, Xiwei Xu, Wenjie Zhang, Liming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22959)  

**Abstract**: Health, Safety, and Environment (HSE) compliance assessment demands dynamic real-time decision-making under complicated regulations and complex human-machine-environment interactions. While large language models (LLMs) hold significant potential for decision intelligence and contextual dialogue, their capacity for domain-specific knowledge in HSE and structured legal reasoning remains underexplored. We introduce HSE-Bench, the first benchmark dataset designed to evaluate the HSE compliance assessment capabilities of LLM. HSE-Bench comprises over 1,000 manually curated questions drawn from regulations, court cases, safety exams, and fieldwork videos, and integrates a reasoning flow based on Issue spotting, rule Recall, rule Application, and rule Conclusion (IRAC) to assess the holistic reasoning pipeline. We conduct extensive evaluations on different prompting strategies and more than 10 LLMs, including foundation models, reasoning models and multimodal vision models. The results show that, although current LLMs achieve good performance, their capabilities largely rely on semantic matching rather than principled reasoning grounded in the underlying HSE compliance context. Moreover, their native reasoning trace lacks the systematic legal reasoning required for rigorous HSE compliance assessment. To alleviate these, we propose a new prompting technique, Reasoning of Expert (RoE), which guides LLMs to simulate the reasoning process of different experts for compliance assessment and reach a more accurate unified decision. We hope our study highlights reasoning gaps in LLMs for HSE compliance and inspires further research on related tasks. 

---
# StrucSum: Graph-Structured Reasoning for Long Document Extractive Summarization with LLMs 

**Authors**: Haohan Yuan, Sukhwa Hong, Haopeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22950)  

**Abstract**: Large language models (LLMs) have shown strong performance in zero-shot summarization, but often struggle to model document structure and identify salient information in long texts. In this work, we introduce StrucSum, a training-free prompting framework that enhances LLM reasoning through sentence-level graph structures. StrucSum injects structural signals into prompts via three targeted strategies: Neighbor-Aware Prompting (NAP) for local context, Centrality-Aware Prompting (CAP) for importance estimation, and Centrality-Guided Masking (CGM) for efficient input reduction. Experiments on ArXiv, PubMed, and Multi-News demonstrate that StrucSum consistently improves both summary quality and factual consistency over unsupervised baselines and vanilla prompting. Notably, on ArXiv, it boosts FactCC and SummaC by 19.2 and 9.7 points, indicating stronger alignment between summaries and source content. These findings suggest that structure-aware prompting is a simple yet effective approach for zero-shot extractive summarization with LLMs, without any training or task-specific tuning. 

---
# Self-Critique and Refinement for Faithful Natural Language Explanations 

**Authors**: Yingming Wang, Pepa Atanasova  

**Link**: [PDF](https://arxiv.org/pdf/2505.22823)  

**Abstract**: With the rapid development of large language models (LLMs), natural language explanations (NLEs) have become increasingly important for understanding model predictions. However, these explanations often fail to faithfully represent the model's actual reasoning process. While existing work has demonstrated that LLMs can self-critique and refine their initial outputs for various tasks, this capability remains unexplored for improving explanation faithfulness. To address this gap, we introduce Self-critique and Refinement for Natural Language Explanations (SR-NLE), a framework that enables models to improve the faithfulness of their own explanations -- specifically, post-hoc NLEs -- through an iterative critique and refinement process without external supervision. Our framework leverages different feedback mechanisms to guide the refinement process, including natural language self-feedback and, notably, a novel feedback approach based on feature attribution that highlights important input words. Our experiments across three datasets and four state-of-the-art LLMs demonstrate that SR-NLE significantly reduces unfaithfulness rates, with our best method achieving an average unfaithfulness rate of 36.02%, compared to 54.81% for baseline -- an absolute reduction of 18.79%. These findings reveal that the investigated LLMs can indeed refine their explanations to better reflect their actual reasoning process, requiring only appropriate guidance through feedback without additional training or fine-tuning. 

---
# GateNLP at SemEval-2025 Task 10: Hierarchical Three-Step Prompting for Multilingual Narrative Classification 

**Authors**: Iknoor Singh, Carolina Scarton, Kalina Bontcheva  

**Link**: [PDF](https://arxiv.org/pdf/2505.22867)  

**Abstract**: The proliferation of online news and the increasing spread of misinformation necessitate robust methods for automatic data analysis. Narrative classification is emerging as a important task, since identifying what is being said online is critical for fact-checkers, policy markers and other professionals working on information studies. This paper presents our approach to SemEval 2025 Task 10 Subtask 2, which aims to classify news articles into a pre-defined two-level taxonomy of main narratives and sub-narratives across multiple languages.
We propose Hierarchical Three-Step Prompting (H3Prompt) for multilingual narrative classification. Our methodology follows a three-step Large Language Model (LLM) prompting strategy, where the model first categorises an article into one of two domains (Ukraine-Russia War or Climate Change), then identifies the most relevant main narratives, and finally assigns sub-narratives. Our approach secured the top position on the English test set among 28 competing teams worldwide. The code is available at this https URL. 

---
# Can Large Language Models Match the Conclusions of Systematic Reviews? 

**Authors**: Christopher Polzak, Alejandro Lozano, Min Woo Sun, James Burgess, Yuhui Zhang, Kevin Wu, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2505.22787)  

**Abstract**: Systematic reviews (SR), in which experts summarize and analyze evidence across individual studies to provide insights on a specialized topic, are a cornerstone for evidence-based clinical decision-making, research, and policy. Given the exponential growth of scientific articles, there is growing interest in using large language models (LLMs) to automate SR generation. However, the ability of LLMs to critically assess evidence and reason across multiple documents to provide recommendations at the same proficiency as domain experts remains poorly characterized. We therefore ask: Can LLMs match the conclusions of systematic reviews written by clinical experts when given access to the same studies? To explore this question, we present MedEvidence, a benchmark pairing findings from 100 SRs with the studies they are based on. We benchmark 24 LLMs on MedEvidence, including reasoning, non-reasoning, medical specialist, and models across varying sizes (from 7B-700B). Through our systematic evaluation, we find that reasoning does not necessarily improve performance, larger models do not consistently yield greater gains, and knowledge-based fine-tuning degrades accuracy on MedEvidence. Instead, most models exhibit similar behavior: performance tends to degrade as token length increases, their responses show overconfidence, and, contrary to human experts, all models show a lack of scientific skepticism toward low-quality findings. These results suggest that more work is still required before LLMs can reliably match the observations from expert-conducted SRs, even though these systems are already deployed and being used by clinicians. We release our codebase and benchmark to the broader research community to further investigate LLM-based SR systems. 

---
# Climate Finance Bench 

**Authors**: Rafik Mankour, Yassine Chafai, Hamada Saleh, Ghassen Ben Hassine, Thibaud Barreau, Peter Tankov  

**Link**: [PDF](https://arxiv.org/pdf/2505.22752)  

**Abstract**: Climate Finance Bench introduces an open benchmark that targets question-answering over corporate climate disclosures using Large Language Models. We curate 33 recent sustainability reports in English drawn from companies across all 11 GICS sectors and annotate 330 expert-validated question-answer pairs that span pure extraction, numerical reasoning, and logical reasoning. Building on this dataset, we propose a comparison of RAG (retrieval-augmented generation) approaches. We show that the retriever's ability to locate passages that actually contain the answer is the chief performance bottleneck. We further argue for transparent carbon reporting in AI-for-climate applications, highlighting advantages of techniques such as Weight Quantization. 

---
# When Models Reason in Your Language: Controlling Thinking Trace Language Comes at the Cost of Accuracy 

**Authors**: Jirui Qi, Shan Chen, Zidi Xiong, Raquel Fernández, Danielle S. Bitterman, Arianna Bisazza  

**Link**: [PDF](https://arxiv.org/pdf/2505.22888)  

**Abstract**: Recent Large Reasoning Models (LRMs) with thinking traces have shown strong performance on English reasoning tasks. However, their ability to think in other languages is less studied. This capability is as important as answer accuracy for real world applications because users may find the reasoning trace useful for oversight only when it is expressed in their own language. We comprehensively evaluate two leading families of LRMs on our XReasoning benchmark and find that even the most advanced models often revert to English or produce fragmented reasoning in other languages, revealing a substantial gap in multilingual reasoning. Prompt based interventions that force models to reason in the users language improve readability and oversight but reduce answer accuracy, exposing an important trade off. We further show that targeted post training on just 100 examples mitigates this mismatch, though some accuracy loss remains. Our results highlight the limited multilingual reasoning capabilities of current LRMs and outline directions for future work. Code and data are available at this https URL. 

---
# MEDAL: A Framework for Benchmarking LLMs as Multilingual Open-Domain Chatbots and Dialogue Evaluators 

**Authors**: John Mendonça, Alon Lavie, Isabel Trancoso  

**Link**: [PDF](https://arxiv.org/pdf/2505.22777)  

**Abstract**: As the capabilities of chatbots and their underlying LLMs continue to dramatically improve, evaluating their performance has increasingly become a major blocker to their further development. A major challenge is the available benchmarking datasets, which are largely static, outdated, and lacking in multilingual coverage, limiting their ability to capture subtle linguistic and cultural variations. This paper introduces MEDAL, an automated multi-agent framework for generating, evaluating, and curating more representative and diverse open-domain dialogue evaluation benchmarks. Our approach leverages several state-of-the-art LLMs to generate user-chatbot multilingual dialogues, conditioned on varied seed contexts. A strong LLM (GPT-4.1) is then used for a multidimensional analysis of the performance of the chatbots, uncovering noticeable cross-lingual performance differences. Guided by this large-scale evaluation, we curate a new meta-evaluation multilingual benchmark and human-annotate samples with nuanced quality judgments. This benchmark is then used to assess the ability of several reasoning and non-reasoning LLMs to act as evaluators of open-domain dialogues. We find that current LLMs struggle to detect nuanced issues, particularly those involving empathy and reasoning. 

---
# ER-REASON: A Benchmark Dataset for LLM-Based Clinical Reasoning in the Emergency Room 

**Authors**: Nikita Mehandru, Niloufar Golchini, David Bamman, Travis Zack, Melanie F. Molina, Ahmed Alaa  

**Link**: [PDF](https://arxiv.org/pdf/2505.22919)  

**Abstract**: Large language models (LLMs) have been extensively evaluated on medical question answering tasks based on licensing exams. However, real-world evaluations often depend on costly human annotators, and existing benchmarks tend to focus on isolated tasks that rarely capture the clinical reasoning or full workflow underlying medical decisions. In this paper, we introduce ER-Reason, a benchmark designed to evaluate LLM-based clinical reasoning and decision-making in the emergency room (ER)--a high-stakes setting where clinicians make rapid, consequential decisions across diverse patient presentations and medical specialties under time pressure. ER-Reason includes data from 3,984 patients, encompassing 25,174 de-identified longitudinal clinical notes spanning discharge summaries, progress notes, history and physical exams, consults, echocardiography reports, imaging notes, and ER provider documentation. The benchmark includes evaluation tasks that span key stages of the ER workflow: triage intake, initial assessment, treatment selection, disposition planning, and final diagnosis--each structured to reflect core clinical reasoning processes such as differential diagnosis via rule-out reasoning. We also collected 72 full physician-authored rationales explaining reasoning processes that mimic the teaching process used in residency training, and are typically absent from ER documentation. Evaluations of state-of-the-art LLMs on ER-Reason reveal a gap between LLM-generated and clinician-authored clinical reasoning for ER decisions, highlighting the need for future research to bridge this divide. 

---
# Domain-Aware Tensor Network Structure Search 

**Authors**: Giorgos Iacovides, Wuyang Zhou, Chao Li, Qibin Zhao, Danilo Mandic  

**Link**: [PDF](https://arxiv.org/pdf/2505.23537)  

**Abstract**: Tensor networks (TNs) provide efficient representations of high-dimensional data, yet identification of the optimal TN structures, the so called tensor network structure search (TN-SS) problem, remains a challenge. Current state-of-the-art (SOTA) algorithms are computationally expensive as they require extensive function evaluations, which is prohibitive for real-world applications. In addition, existing methods ignore valuable domain information inherent in real-world tensor data and lack transparency in their identified TN structures. To this end, we propose a novel TN-SS framework, termed the tnLLM, which incorporates domain information about the data and harnesses the reasoning capabilities of large language models (LLMs) to directly predict suitable TN structures. The proposed framework involves a domain-aware prompting pipeline which instructs the LLM to infer suitable TN structures based on the real-world relationships between tensor modes. In this way, our approach is capable of not only iteratively optimizing the objective function, but also generating domain-aware explanations for the identified structures. Experimental results demonstrate that tnLLM achieves comparable TN-SS objective function values with much fewer function evaluations compared to SOTA algorithms. Furthermore, we demonstrate that the LLM-enabled domain information can be used to find good initializations in the search space for sampling-based SOTA methods to accelerate their convergence while preserving theoretical performance guarantees. 

---
# Conversational Alignment with Artificial Intelligence in Context 

**Authors**: Rachel Katharine Sterken, James Ravi Kirkpatrick  

**Link**: [PDF](https://arxiv.org/pdf/2505.22907)  

**Abstract**: The development of sophisticated artificial intelligence (AI) conversational agents based on large language models raises important questions about the relationship between human norms, values, and practices and AI design and performance. This article explores what it means for AI agents to be conversationally aligned to human communicative norms and practices for handling context and common ground and proposes a new framework for evaluating developers' design choices. We begin by drawing on the philosophical and linguistic literature on conversational pragmatics to motivate a set of desiderata, which we call the CONTEXT-ALIGN framework, for conversational alignment with human communicative practices. We then suggest that current large language model (LLM) architectures, constraints, and affordances may impose fundamental limitations on achieving full conversational alignment. 

---
# Large Language Models for Depression Recognition in Spoken Language Integrating Psychological Knowledge 

**Authors**: Yupei Li, Shuaijie Shao, Manuel Milling, Björn W. Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2505.22863)  

**Abstract**: Depression is a growing concern gaining attention in both public discourse and AI research. While deep neural networks (DNNs) have been used for recognition, they still lack real-world effectiveness. Large language models (LLMs) show strong potential but require domain-specific fine-tuning and struggle with non-textual cues. Since depression is often expressed through vocal tone and behaviour rather than explicit text, relying on language alone is insufficient. Diagnostic accuracy also suffers without incorporating psychological expertise. To address these limitations, we present, to the best of our knowledge, the first application of LLMs to multimodal depression detection using the DAIC-WOZ dataset. We extract the audio features using the pre-trained model Wav2Vec, and mapped it to text-based LLMs for further processing. We also propose a novel strategy for incorporating psychological knowledge into LLMs to enhance diagnostic performance, specifically using a question and answer set to grant authorised knowledge to LLMs. Our approach yields a notable improvement in both Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) compared to a base score proposed by the related original paper. The codes are available at this https URL 

---
# TailorSQL: An NL2SQL System Tailored to Your Query Workload 

**Authors**: Kapil Vaidya, Jialin Ding, Sebastian Kosak, David Kernert, Chuan Lei, Xiao Qin, Abhinav Tripathy, Ramesh Balan, Balakrishnan Narayanaswamy, Tim Kraska  

**Link**: [PDF](https://arxiv.org/pdf/2505.23039)  

**Abstract**: NL2SQL (natural language to SQL) translates natural language questions into SQL queries, thereby making structured data accessible to non-technical users, serving as the foundation for intelligent data applications. State-of-the-art NL2SQL techniques typically perform translation by retrieving database-specific information, such as the database schema, and invoking a pre-trained large language model (LLM) using the question and retrieved information to generate the SQL query.
However, existing NL2SQL techniques miss a key opportunity which is present in real-world settings: NL2SQL is typically applied on existing databases which have already served many SQL queries in the past. The past query workload implicitly contains information which is helpful for accurate NL2SQL translation and is not apparent from the database schema alone, such as common join paths and the semantics of obscurely-named tables and columns. We introduce TailorSQL, a NL2SQL system that takes advantage of information in the past query workload to improve both the accuracy and latency of translating natural language questions into SQL. By specializing to a given workload, TailorSQL achieves up to 2$\times$ improvement in execution accuracy on standardized benchmarks. 

---
