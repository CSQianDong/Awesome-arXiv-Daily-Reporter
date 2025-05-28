# How does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective 

**Authors**: Shimao Zhang, Zhejian Lai, Xiang Liu, Shuaijie She, Xiao Liu, Yeyun Gong, Shujian Huang, Jiajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.21505)  

**Abstract**: Multilingual Alignment is an effective and representative paradigm to enhance LLMs' multilingual capabilities, which transfers the capabilities from the high-resource languages to the low-resource languages. Meanwhile, some researches on language-specific neurons reveal that there are language-specific neurons that are selectively activated in LLMs when processing different languages. This provides a new perspective to analyze and understand LLMs' mechanisms more specifically in multilingual scenarios. In this work, we propose a new finer-grained neuron identification algorithm, which detects language neurons~(including language-specific neurons and language-related neurons) and language-agnostic neurons. Furthermore, based on the distributional characteristics of different types of neurons, we divide the LLMs' internal process for multilingual inference into four parts: (1) multilingual understanding, (2) shared semantic space reasoning, (3) multilingual output space transformation, and (4) vocabulary space outputting. Additionally, we systematically analyze the models before and after alignment with a focus on different types of neurons. We also analyze the phenomenon of ''Spontaneous Multilingual Alignment''. Overall, our work conducts a comprehensive investigation based on different types of neurons, providing empirical results and valuable insights for better understanding multilingual alignment and multilingual capabilities of LLMs. 

---
# Silence is Not Consensus: Disrupting Agreement Bias in Multi-Agent LLMs via Catfish Agent for Clinical Decision Making 

**Authors**: Yihan Wang, Qiao Yan, Zhenghao Xing, Lihao Liu, Junjun He, Chi-Wing Fu, Xiaowei Hu, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2505.21503)  

**Abstract**: Large language models (LLMs) have demonstrated strong potential in clinical question answering, with recent multi-agent frameworks further improving diagnostic accuracy via collaborative reasoning. However, we identify a recurring issue of Silent Agreement, where agents prematurely converge on diagnoses without sufficient critical analysis, particularly in complex or ambiguous cases. We present a new concept called Catfish Agent, a role-specialized LLM designed to inject structured dissent and counter silent agreement. Inspired by the ``catfish effect'' in organizational psychology, the Catfish Agent is designed to challenge emerging consensus to stimulate deeper reasoning. We formulate two mechanisms to encourage effective and context-aware interventions: (i) a complexity-aware intervention that modulates agent engagement based on case difficulty, and (ii) a tone-calibrated intervention articulated to balance critique and collaboration. Evaluations on nine medical Q&A and three medical VQA benchmarks show that our approach consistently outperforms both single- and multi-agent LLMs frameworks, including leading commercial models such as GPT-4o and DeepSeek-R1. 

---
# UI-Genie: A Self-Improving Approach for Iteratively Boosting MLLM-based Mobile GUI Agents 

**Authors**: Han Xiao, Guozhi Wang, Yuxiang Chai, Zimu Lu, Weifeng Lin, Hao He, Lue Fan, Liuyang Bian, Rui Hu, Liang Liu, Shuai Ren, Yafei Wen, Xiaoxin Chen, Aojun Zhou, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.21496)  

**Abstract**: In this paper, we introduce UI-Genie, a self-improving framework addressing two key challenges in GUI agents: verification of trajectory outcome is challenging and high-quality training data are not scalable. These challenges are addressed by a reward model and a self-improving pipeline, respectively. The reward model, UI-Genie-RM, features an image-text interleaved architecture that efficiently pro- cesses historical context and unifies action-level and task-level rewards. To sup- port the training of UI-Genie-RM, we develop deliberately-designed data genera- tion strategies including rule-based verification, controlled trajectory corruption, and hard negative mining. To address the second challenge, a self-improvement pipeline progressively expands solvable complex GUI tasks by enhancing both the agent and reward models through reward-guided exploration and outcome verification in dynamic environments. For training the model, we generate UI- Genie-RM-517k and UI-Genie-Agent-16k, establishing the first reward-specific dataset for GUI agents while demonstrating high-quality synthetic trajectory gen- eration without manual annotation. Experimental results show that UI-Genie achieves state-of-the-art performance across multiple GUI agent benchmarks with three generations of data-model self-improvement. We open-source our complete framework implementation and generated datasets to facilitate further research in this https URL. 

---
# Are Language Models Consequentialist or Deontological Moral Reasoners? 

**Authors**: Keenan Samway, Max Kleiman-Weiner, David Guzman Piedrahita, Rada Mihalcea, Bernhard Schölkopf, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.21479)  

**Abstract**: As AI systems increasingly navigate applications in healthcare, law, and governance, understanding how they handle ethically complex scenarios becomes critical. Previous work has mainly examined the moral judgments in large language models (LLMs), rather than their underlying moral reasoning process. In contrast, we focus on a large-scale analysis of the moral reasoning traces provided by LLMs. Furthermore, unlike prior work that attempted to draw inferences from only a handful of moral dilemmas, our study leverages over 600 distinct trolley problems as probes for revealing the reasoning patterns that emerge within different LLMs. We introduce and test a taxonomy of moral rationales to systematically classify reasoning traces according to two main normative ethical theories: consequentialism and deontology. Our analysis reveals that LLM chains-of-thought tend to favor deontological principles based on moral obligations, while post-hoc explanations shift notably toward consequentialist rationales that emphasize utility. Our framework provides a foundation for understanding how LLMs process and articulate ethical considerations, an important step toward safe and interpretable deployment of LLMs in high-stakes decision-making environments. Our code is available at this https URL . 

---
# Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration 

**Authors**: Zijun Liu, Zhennan Wan, Peng Li, Ming Yan, Ji Zhang, Fei Huang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21471)  

**Abstract**: With the rapid advancement of post-training techniques for reasoning and information seeking, large language models (LLMs) can incorporate a large quantity of retrieved knowledge to solve complex tasks. However, the limited context window of LLMs obstructs scaling the amount of external knowledge input, prohibiting further improvement, especially for tasks requiring significant amount of external knowledge. Existing context window extension methods inevitably cause information loss. LLM-based multi-agent methods emerge as a new paradigm to handle massive input in a distributional manner, where we identify two core bottlenecks in existing knowledge synchronization and reasoning processes. In this work, we develop a multi-agent framework, $\textbf{ExtAgents}$, to overcome the bottlenecks and enable better scalability in inference-time knowledge integration without longer-context training. Benchmarked with our enhanced multi-hop question answering test, $\textbf{$\boldsymbol{\infty}$Bench+}$, and other public test sets including long survey generation, ExtAgents significantly enhances the performance over existing non-training methods with the same amount of external knowledge input, regardless of whether it falls $\textit{within or exceeds the context window}$. Moreover, the method maintains high efficiency due to high parallelism. Further study in the coordination of LLM agents on increasing external knowledge input could benefit real-world applications. 

---
# Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion 

**Authors**: Zhanqiu Hu, Jian Meng, Yash Akhauri, Mohamed S. Abdelfattah, Jae-sun Seo, Zhiru Zhang, Udit Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.21467)  

**Abstract**: Diffusion language models offer parallel token generation and inherent bidirectionality, promising more efficient and powerful sequence modeling compared to autoregressive approaches. However, state-of-the-art diffusion models (e.g., Dream 7B, LLaDA 8B) suffer from slow inference. While they match the quality of similarly sized Autoregressive (AR) Models (e.g., Qwen2.5 7B, Llama3 8B), their iterative denoising requires multiple full-sequence forward passes, resulting in high computational costs and latency, particularly for long input prompts and long-context scenarios. Furthermore, parallel token generation introduces token incoherence problems, and current sampling heuristics suffer from significant quality drops with decreasing denoising steps. We address these limitations with two training-free techniques. First, we propose FreeCache, a Key-Value (KV) approximation caching technique that reuses stable KV projections across denoising steps, effectively reducing the computational cost of DLM inference. Second, we introduce Guided Diffusion, a training-free method that uses a lightweight pretrained autoregressive model to supervise token unmasking, dramatically reducing the total number of denoising iterations without sacrificing quality. We conduct extensive evaluations on open-source reasoning benchmarks, and our combined methods deliver up to a 34x end-to-end speedup without compromising accuracy. For the first time, diffusion language models achieve a comparable and even faster latency as the widely adopted autoregressive models. Our work successfully paved the way for scaling up the diffusion language model to a broader scope of applications across different domains. 

---
# Do LLMs Need to Think in One Language? Correlation between Latent Language and Task Performance 

**Authors**: Shintaro Ozaki, Tatsuya Hiraoka, Hiroto Otake, Hiroki Ouchi, Masaru Isonuma, Benjamin Heinzerling, Kentaro Inui, Taro Watanabe, Yusuke Miyao, Yohei Oseki, Yu Takagi  

**Link**: [PDF](https://arxiv.org/pdf/2505.21458)  

**Abstract**: Large Language Models (LLMs) are known to process information using a proficient internal language consistently, referred to as latent language, which may differ from the input or output languages. However, how the discrepancy between the latent language and the input and output language affects downstream task performance remains largely unexplored. While many studies research the latent language of LLMs, few address its importance in influencing task performance. In our study, we hypothesize that thinking in latent language consistently enhances downstream task performance. To validate this, our work varies the input prompt languages across multiple downstream tasks and analyzes the correlation between consistency in latent language and task performance. We create datasets consisting of questions from diverse domains such as translation and geo-culture, which are influenced by the choice of latent language. Experimental results across multiple LLMs on translation and geo-culture tasks, which are sensitive to the choice of language, indicate that maintaining consistency in latent language is not always necessary for optimal downstream task performance. This is because these models adapt their internal representations near the final layers to match the target language, reducing the impact of consistency on overall performance. 

---
# Words Like Knives: Backstory-Personalized Modeling and Detection of Violent Communication 

**Authors**: Jocelyn Shen, Akhila Yerukola, Xuhui Zhou, Cynthia Breazeal, Maarten Sap, Hae Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.21451)  

**Abstract**: Conversational breakdowns in close relationships are deeply shaped by personal histories and emotional context, yet most NLP research treats conflict detection as a general task, overlooking the relational dynamics that influence how messages are perceived. In this work, we leverage nonviolent communication (NVC) theory to evaluate LLMs in detecting conversational breakdowns and assessing how relationship backstory influences both human and model perception of conflicts. Given the sensitivity and scarcity of real-world datasets featuring conflict between familiar social partners with rich personal backstories, we contribute the PersonaConflicts Corpus, a dataset of N=5,772 naturalistic simulated dialogues spanning diverse conflict scenarios between friends, family members, and romantic partners. Through a controlled human study, we annotate a subset of dialogues and obtain fine-grained labels of communication breakdown types on individual turns, and assess the impact of backstory on human and model perception of conflict in conversation. We find that the polarity of relationship backstories significantly shifted human perception of communication breakdowns and impressions of the social partners, yet models struggle to meaningfully leverage those backstories in the detection task. Additionally, we find that models consistently overestimate how positively a message will make a listener feel. Our findings underscore the critical role of personalization to relationship contexts in enabling LLMs to serve as effective mediators in human communication for authentic connection. 

---
# Towards Better Instruction Following Retrieval Models 

**Authors**: Yuchen Zhuang, Aaron Trinh, Rushi Qiang, Haotian Sun, Chao Zhang, Hanjun Dai, Bo Dai  

**Link**: [PDF](https://arxiv.org/pdf/2505.21439)  

**Abstract**: Modern information retrieval (IR) models, trained exclusively on standard <query, passage> pairs, struggle to effectively interpret and follow explicit user instructions. We introduce InF-IR, a large-scale, high-quality training corpus tailored for enhancing retrieval models in Instruction-Following IR. InF-IR expands traditional training pairs into over 38,000 expressive <instruction, query, passage> triplets as positive samples. In particular, for each positive triplet, we generate two additional hard negative examples by poisoning both instructions and queries, then rigorously validated by an advanced reasoning model (o3-mini) to ensure semantic plausibility while maintaining instructional incorrectness. Unlike existing corpora that primarily support computationally intensive reranking tasks for decoder-only language models, the highly contrastive positive-negative triplets in InF-IR further enable efficient representation learning for smaller encoder-only models, facilitating direct embedding-based retrieval. Using this corpus, we train InF-Embed, an instruction-aware Embedding model optimized through contrastive learning and instruction-query attention mechanisms to align retrieval outcomes precisely with user intents. Extensive experiments across five instruction-based retrieval benchmarks demonstrate that InF-Embed significantly surpasses competitive baselines by 8.1% in p-MRR, measuring the instruction-following capabilities. 

---
# RefTool: Enhancing Model Reasoning with Reference-Guided Tool Creation 

**Authors**: Xiao Liu, Da Yin, Zirui Wu, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.21413)  

**Abstract**: Tools enhance the reasoning capabilities of large language models (LLMs) in complex problem-solving tasks, but not all tasks have available tools. In the absence of predefined tools, prior works have explored instructing LLMs to generate tools on their own. However, such approaches rely heavily on the models' internal knowledge and would fail in domains beyond the LLMs' knowledge scope. To address this limitation, we propose RefTool, a reference-guided framework for automatic tool creation that leverages structured external materials such as textbooks. RefTool consists of two modules: (1) tool creation, where LLMs generate executable tools from reference content, validate them using illustrative examples, and organize them hierarchically into a toolbox; and (2) tool utilization, where LLMs navigate the toolbox structure to select and apply the appropriate tools to solve problems. Experiments on causality, physics, and chemistry benchmarks demonstrate that RefTool outperforms existing tool-creation and domain-specific reasoning methods by 11.3% on average accuracy, while being cost-efficient and broadly generalizable. Analyses reveal that grounding tool creation in references produces accurate and faithful tools, and that the hierarchical structure facilitates effective tool selection. RefTool enables LLMs to overcome knowledge limitations, demonstrating the value of grounding tool creation in external references for enhanced and generalizable reasoning. 

---
# Pangu Pro MoE: Mixture of Grouped Experts for Efficient Sparsity 

**Authors**: Yehui Tang, Xiaosong Li, Fangcheng Liu, Wei Guo, Hang Zhou, Yaoyuan Wang, Kai Han, Xianzhi Yu, Jinpeng Li, Hui Zang, Fei Mi, Xiaojun Meng, Zhicheng Liu, Hanting Chen, Binfan Zheng, Can Chen, Youliang Yan, Ruiming Tang, Peifeng Qin, Xinghao Chen, Dacheng Tao, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21411)  

**Abstract**: The surgence of Mixture of Experts (MoE) in Large Language Models promises a small price of execution cost for a much larger model parameter count and learning capacity, because only a small fraction of parameters are activated for each input token. However, it is commonly observed that some experts are activated far more often than others, leading to system inefficiency when running the experts on different devices in parallel. Therefore, we introduce Mixture of Grouped Experts (MoGE), which groups the experts during selection and balances the expert workload better than MoE in nature. It constrains tokens to activate an equal number of experts within each predefined expert group. When a model execution is distributed on multiple devices, this architectural design ensures a balanced computational load across devices, significantly enhancing throughput, particularly for the inference phase. Further, we build Pangu Pro MoE on Ascend NPUs, a sparse model based on MoGE with 72 billion total parameters, 16 billion of which are activated for each token. The configuration of Pangu Pro MoE is optimized for Ascend 300I Duo and 800I A2 through extensive system simulation studies. Our experiments indicate that MoGE indeed leads to better expert load balancing and more efficient execution for both model training and inference on Ascend NPUs. The inference performance of Pangu Pro MoE achieves 1148 tokens/s per card and can be further improved to 1528 tokens/s per card by speculative acceleration, outperforming comparable 32B and 72B Dense models. Furthermore, we achieve an excellent cost-to-performance ratio for model inference on Ascend 300I this http URL studies show that Ascend NPUs are capable of training Pangu Pro MoE with massive parallelization to make it a leading model within the sub-100B total parameter class, outperforming prominent open-source models like GLM-Z1-32B and Qwen3-32B. 

---
# RelationalFactQA: A Benchmark for Evaluating Tabular Fact Retrieval from Large Language Models 

**Authors**: Dario Satriani, Enzo Veltri, Donatello Santoro, Paolo Papotti  

**Link**: [PDF](https://arxiv.org/pdf/2505.21409)  

**Abstract**: Factuality in Large Language Models (LLMs) is a persistent challenge. Current benchmarks often assess short factual answers, overlooking the critical ability to generate structured, multi-record tabular outputs from parametric knowledge. We demonstrate that this relational fact retrieval is substantially more difficult than isolated point-wise queries, even when individual facts are known to the model, exposing distinct failure modes sensitive to output dimensionality (e.g., number of attributes or records). To systematically evaluate this under-explored capability, we introduce RelationalFactQA, a new benchmark featuring diverse natural language questions (paired with SQL) and gold-standard tabular answers, specifically designed to assess knowledge retrieval in a structured format. RelationalFactQA enables analysis across varying query complexities, output sizes, and data characteristics. Our experiments reveal that even state-of-the-art LLMs struggle significantly, not exceeding 25% factual accuracy in generating relational outputs, with performance notably degrading as output dimensionality increases. These findings underscore critical limitations in current LLMs' ability to synthesize structured factual knowledge and establish RelationalFactQA as a crucial resource for measuring future progress in LLM factuality. 

---
# Factual Self-Awareness in Language Models: Representation, Robustness, and Scaling 

**Authors**: Hovhannes Tamoyan, Subhabrata Dutta, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2505.21399)  

**Abstract**: Factual incorrectness in generated content is one of the primary concerns in ubiquitous deployment of large language models (LLMs). Prior findings suggest LLMs can (sometimes) detect factual incorrectness in their generated content (i.e., fact-checking post-generation). In this work, we provide evidence supporting the presence of LLMs' internal compass that dictate the correctness of factual recall at the time of generation. We demonstrate that for a given subject entity and a relation, LLMs internally encode linear features in the Transformer's residual stream that dictate whether it will be able to recall the correct attribute (that forms a valid entity-relation-attribute triplet). This self-awareness signal is robust to minor formatting variations. We investigate the effects of context perturbation via different example selection strategies. Scaling experiments across model sizes and training dynamics highlight that self-awareness emerges rapidly during training and peaks in intermediate layers. These findings uncover intrinsic self-monitoring capabilities within LLMs, contributing to their interpretability and reliability. 

---
# DecisionFlow: Advancing Large Language Model as Principled Decision Maker 

**Authors**: Xiusi Chen, Shanyong Wang, Cheng Qian, Hongru Wang, Peixuan Han, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.21397)  

**Abstract**: In high-stakes domains such as healthcare and finance, effective decision-making demands not just accurate outcomes but transparent and explainable reasoning. However, current language models often lack the structured deliberation needed for such tasks, instead generating decisions and justifications in a disconnected, post-hoc manner. To address this, we propose DecisionFlow, a novel decision modeling framework that guides models to reason over structured representations of actions, attributes, and constraints. Rather than predicting answers directly from prompts, DecisionFlow builds a semantically grounded decision space and infers a latent utility function to evaluate trade-offs in a transparent, utility-driven manner. This process produces decisions tightly coupled with interpretable rationales reflecting the model's reasoning. Empirical results on two high-stakes benchmarks show that DecisionFlow not only achieves up to 30% accuracy gains over strong prompting baselines but also enhances alignment in outcomes. Our work is a critical step toward integrating symbolic reasoning with LLMs, enabling more accountable, explainable, and reliable LLM decision support systems. We release the data and code at this https URL. 

---
# Improving Research Idea Generation Through Data: An Empirical Investigation in Social Science 

**Authors**: Xiao Liu, Xinyi Dong, Xinyang Gao, Yansong Feng, Xun Pang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21396)  

**Abstract**: Recent advancements in large language models (LLMs) have shown promise in generating novel research ideas. However, these ideas often face challenges related to feasibility and expected effectiveness. This paper explores how augmenting LLMs with relevant data during the idea generation process can enhance the quality of generated ideas. We introduce two ways of incorporating data: (1) providing metadata during the idea generation stage to guide LLMs toward feasible directions, and (2) adding automatic validation during the idea selection stage to assess the empirical plausibility of hypotheses within ideas. We conduct experiments in the social science domain, specifically with climate negotiation topics, and find that metadata improves the feasibility of generated ideas by 20%, while automatic validation improves the overall quality of selected ideas by 7%. A human study shows that LLM-generated ideas, along with their related data and validation processes, inspire researchers to propose research ideas with higher quality. Our work highlights the potential of data-driven research idea generation, and underscores the practical utility of LLM-assisted ideation in real-world academic settings. 

---
# AutoJudger: An Agent-Driven Framework for Efficient Benchmarking of MLLMs 

**Authors**: Xuanwen Ding, Chengjun Pan, Zejun Li, Jiwen Zhang, Siyuan Wang, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.21389)  

**Abstract**: Evaluating multimodal large language models (MLLMs) is increasingly expensive, as the growing size and cross-modality complexity of benchmarks demand significant scoring efforts. To tackle with this difficulty, we introduce AutoJudger, an agent-driven framework for efficient and adaptive benchmarking of MLLMs that tackles this escalating cost. AutoJudger employs the Item Response Theory (IRT) to estimate the question difficulty and an autonomous evaluation agent to dynamically select the most informative test questions based on the model's real-time performance. Specifically, AutoJudger incorporates two pivotal components: a semantic-aware retrieval mechanism to ensure that selected questions cover diverse and challenging scenarios across both vision and language modalities, and a dynamic memory that maintains contextual statistics of previously evaluated questions to guide coherent and globally informed question selection throughout the evaluation process. Extensive experiments on four representative multimodal benchmarks demonstrate that our adaptive framework dramatically reduces evaluation expenses, i.e. AutoJudger uses only 4% of the data to achieve over 90% ranking accuracy with the full benchmark evaluation on MMT-Bench. 

---
# PHISH in MESH: Korean Adversarial Phonetic Substitution and Phonetic-Semantic Feature Integration Defense 

**Authors**: Byungjun Kim, Minju Kim, Hyeonchu Park, Bugeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.21380)  

**Abstract**: As malicious users increasingly employ phonetic substitution to evade hate speech detection, researchers have investigated such strategies. However, two key challenges remain. First, existing studies have overlooked the Korean language, despite its vulnerability to phonetic perturbations due to its phonographic nature. Second, prior work has primarily focused on constructing datasets rather than developing architectural defenses. To address these challenges, we propose (1) PHonetic-Informed Substitution for Hangul (PHISH) that exploits the phonological characteristics of the Korean writing system, and (2) Mixed Encoding of Semantic-pHonetic features (MESH) that enhances the detector's robustness by incorporating phonetic information at the architectural level. Our experimental results demonstrate the effectiveness of our proposed methods on both perturbed and unperturbed datasets, suggesting that they not only improve detection performance but also reflect realistic adversarial behaviors employed by malicious users. 

---
# Analyzing values about gendered language reform in LLMs' revisions 

**Authors**: Jules Watson, Xi Wang, Raymond Liu, Suzanne Stevenson, Barend Beekhuizen  

**Link**: [PDF](https://arxiv.org/pdf/2505.21378)  

**Abstract**: Within the common LLM use case of text revision, we study LLMs' revision of gendered role nouns (e.g., outdoorsperson/woman/man) and their justifications of such revisions. We evaluate their alignment with feminist and trans-inclusive language reforms for English. Drawing on insight from sociolinguistics, we further assess if LLMs are sensitive to the same contextual effects in the application of such reforms as people are, finding broad evidence of such effects. We discuss implications for value alignment. 

---
# Evaluating LLM Adaptation to Sociodemographic Factors: User Profile vs. Dialogue History 

**Authors**: Qishuai Zhong, Zongmin Li, Siqi Fan, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.21362)  

**Abstract**: Effective engagement by large language models (LLMs) requires adapting responses to users' sociodemographic characteristics, such as age, occupation, and education level. While many real-world applications leverage dialogue history for contextualization, existing evaluations of LLMs' behavioral adaptation often focus on single-turn prompts. In this paper, we propose a framework to evaluate LLM adaptation when attributes are introduced either (1) explicitly via user profiles in the prompt or (2) implicitly through multi-turn dialogue history. We assess the consistency of model behavior across these modalities. Using a multi-agent pipeline, we construct a synthetic dataset pairing dialogue histories with distinct user profiles and employ questions from the Value Survey Module (VSM 2013) (Hofstede and Hofstede, 2016) to probe value expression. Our findings indicate that most models adjust their expressed values in response to demographic changes, particularly in age and education level, but consistency varies. Models with stronger reasoning capabilities demonstrate greater alignment, indicating the importance of reasoning in robust sociodemographic adaptation. 

---
# Leveraging Large Language Models for Bengali Math Word Problem Solving with Chain of Thought Reasoning 

**Authors**: Bidyarthi Paul, Jalisha Jashim Era, Mirazur Rahman Zim, Tahmid Sattar Aothoi, Faisal Muhammad Shah  

**Link**: [PDF](https://arxiv.org/pdf/2505.21354)  

**Abstract**: Solving Bengali Math Word Problems (MWPs) remains a major challenge in natural language processing (NLP) due to the language's low-resource status and the multi-step reasoning required. Existing models struggle with complex Bengali MWPs, largely because no human-annotated Bengali dataset has previously addressed this task. This gap has limited progress in Bengali mathematical reasoning. To address this, we created SOMADHAN, a dataset of 8792 complex Bengali MWPs with manually written, step-by-step solutions. We designed this dataset to support reasoning-focused evaluation and model development in a linguistically underrepresented context. Using SOMADHAN, we evaluated a range of large language models (LLMs) - including GPT-4o, GPT-3.5 Turbo, LLaMA series models, Deepseek, and Qwen - through both zero-shot and few-shot prompting with and without Chain of Thought (CoT) reasoning. CoT prompting consistently improved performance over standard prompting, especially in tasks requiring multi-step logic. LLaMA-3.3 70B achieved the highest accuracy of 88% with few-shot CoT prompting. We also applied Low-Rank Adaptation (LoRA) to fine-tune models efficiently, enabling them to adapt to Bengali MWPs with minimal computational cost. Our work fills a critical gap in Bengali NLP by providing a high-quality reasoning dataset and a scalable framework for solving complex MWPs. We aim to advance equitable research in low-resource languages and enhance reasoning capabilities in educational and language technologies. 

---
# PEDANTIC: A Dataset for the Automatic Examination of Definiteness in Patent Claims 

**Authors**: Valentin Knappich, Annemarie Friedrich, Anna Hätty, Simon Razniewski  

**Link**: [PDF](https://arxiv.org/pdf/2505.21342)  

**Abstract**: Patent claims define the scope of protection for an invention. If there are ambiguities in a claim, it is rejected by the patent office. In the US, this is referred to as indefiniteness (35 U.S.C § 112(b)) and is among the most frequent reasons for patent application rejection. The development of automatic methods for patent definiteness examination has the potential to make patent drafting and examination more efficient, but no annotated dataset has been published to date.
We introduce PEDANTIC (\underline{P}at\underline{e}nt \underline{D}efiniteness Ex\underline{a}mi\underline{n}a\underline{ti}on \underline{C}orpus), a novel dataset of 14k US patent claims from patent applications relating to Natural Language Processing (NLP), annotated with reasons for indefiniteness. We construct PEDANTIC using a fully automatic pipeline that retrieves office action documents from the USPTO and uses Large Language Models (LLMs) to extract the reasons for indefiniteness. A human validation study confirms the pipeline's accuracy in generating high-quality annotations. To gain insight beyond binary classification metrics, we implement an LLM-as-Judge evaluation that compares the free-form reasoning of every model-cited reason with every examiner-cited reason. We show that LLM agents based on Qwen 2.5 32B and 72B struggle to outperform logistic regression baselines on definiteness prediction, even though they often correctly identify the underlying reasons. PEDANTIC provides a valuable resource for patent AI researchers, enabling the development of advanced examination models. We will publicly release the dataset and code. 

---
# Leveraging large language models and traditional machine learning ensembles for ADHD detection from narrative transcripts 

**Authors**: Yuxin Zhu, Yuting Guo, Noah Marchuck, Abeed Sarker, Yun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21324)  

**Abstract**: Despite rapid advances in large language models (LLMs), their integration with traditional supervised machine learning (ML) techniques that have proven applicability to medical data remains underexplored. This is particularly true for psychiatric applications, where narrative data often exhibit nuanced linguistic and contextual complexity, and can benefit from the combination of multiple models with differing characteristics. In this study, we introduce an ensemble framework for automatically classifying Attention-Deficit/Hyperactivity Disorder (ADHD) diagnosis (binary) using narrative transcripts. Our approach integrates three complementary models: LLaMA3, an open-source LLM that captures long-range semantic structure; RoBERTa, a pre-trained transformer model fine-tuned on labeled clinical narratives; and a Support Vector Machine (SVM) classifier trained using TF-IDF-based lexical features. These models are aggregated through a majority voting mechanism to enhance predictive robustness. The dataset includes 441 instances, including 352 for training and 89 for validation. Empirical results show that the ensemble outperforms individual models, achieving an F$_1$ score of 0.71 (95\% CI: [0.60-0.80]). Compared to the best-performing individual model (SVM), the ensemble improved recall while maintaining competitive precision. This indicates the strong sensitivity of the ensemble in identifying ADHD-related linguistic cues. These findings demonstrate the promise of hybrid architectures that leverage the semantic richness of LLMs alongside the interpretability and pattern recognition capabilities of traditional supervised ML, offering a new direction for robust and generalizable psychiatric text classification. 

---
# Charting the Landscape of African NLP: Mapping Progress and Shaping the Road Ahead 

**Authors**: Jesujoba O. Alabi, Michael A. Hedderich, David Ifeoluwa Adelani, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2505.21315)  

**Abstract**: With over 2,000 languages and potentially millions of speakers, Africa represents one of the richest linguistic regions in the world. Yet, this diversity is scarcely reflected in state-of-the-art natural language processing (NLP) systems and large language models (LLMs), which predominantly support a narrow set of high-resource languages. This exclusion not only limits the reach and utility of modern NLP technologies but also risks widening the digital divide across linguistic communities. Nevertheless, NLP research on African languages is active and growing. In recent years, there has been a surge of interest in this area, driven by several factors-including the creation of multilingual language resources, the rise of community-led initiatives, and increased support through funding programs. In this survey, we analyze 734 research papers on NLP for African languages published over the past five years, offering a comprehensive overview of recent progress across core tasks. We identify key trends shaping the field and conclude by outlining promising directions to foster more inclusive and sustainable NLP research for African languages. 

---
# How Humans and LLMs Organize Conceptual Knowledge: Exploring Subordinate Categories in Italian 

**Authors**: Andrea Pedrotti, Giulia Rambelli, Caterina Villani, Marianna Bolognesi  

**Link**: [PDF](https://arxiv.org/pdf/2505.21301)  

**Abstract**: People can categorize the same entity at multiple taxonomic levels, such as basic (bear), superordinate (animal), and subordinate (grizzly bear). While prior research has focused on basic-level categories, this study is the first attempt to examine the organization of categories by analyzing exemplars produced at the subordinate level. We present a new Italian psycholinguistic dataset of human-generated exemplars for 187 concrete words. We then use these data to evaluate whether textual and vision LLMs produce meaningful exemplars that align with human category organization across three key tasks: exemplar generation, category induction, and typicality judgment. Our findings show a low alignment between humans and LLMs, consistent with previous studies. However, their performance varies notably across different semantic domains. Ultimately, this study highlights both the promises and the constraints of using AI-generated exemplars to support psychological and linguistic research. 

---
# rStar-Coder: Scaling Competitive Code Reasoning with a Large-Scale Verified Dataset 

**Authors**: Yifei Liu, Li Lyna Zhang, Yi Zhu, Bingcheng Dong, Xudong Zhou, Ning Shang, Fan Yang, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21297)  

**Abstract**: Advancing code reasoning in large language models (LLMs) is fundamentally limited by the scarcity of high-difficulty datasets, especially those with verifiable input-output test cases necessary for rigorous solution validation at scale. We introduce rStar-Coder, which significantly improves LLM code reasoning capabilities by constructing a large-scale, verified dataset of 418K competition-level code problems, 580K long-reasoning solutions along with rich test cases of varying difficulty. This is achieved through three core contributions: (1) we curate competitive programming code problems and oracle solutions to synthesize new, solvable problems; (2) we introduce a reliable input-output test case synthesis pipeline that decouples the generation into a three-step input generation method and a mutual verification mechanism for effective output labeling; (3) we augment problems with high-quality, test-case-verified long-reasoning solutions. Extensive experiments on Qwen models (1.5B-14B) across various code reasoning benchmarks demonstrate the superiority of rStar-Coder dataset, achieving leading performance comparable to frontier reasoning LLMs with much smaller model sizes. On LiveCodeBench, rStar-Coder improves Qwen2.5-7B from 17.4% to an impressive 57.3%, and Qwen2.5-14B from 23.3% to 62.5%, surpassing o3-mini (low) by3.1%. On the more challenging USA Computing Olympiad, our 7B model achieves an average pass@1 accuracy of 16.15%, outperforming the frontier-level QWQ-32B. Code and the dataset will be released at this https URL. 

---
# Multilingual Pretraining for Pixel Language Models 

**Authors**: Ilker Kesen, Jonas F. Lotz, Ingo Ziegler, Phillip Rust, Desmond Elliott  

**Link**: [PDF](https://arxiv.org/pdf/2505.21265)  

**Abstract**: Pixel language models operate directly on images of rendered text, eliminating the need for a fixed vocabulary. While these models have demonstrated strong capabilities for downstream cross-lingual transfer, multilingual pretraining remains underexplored. We introduce PIXEL-M4, a model pretrained on four visually and linguistically diverse languages: English, Hindi, Ukrainian, and Simplified Chinese. Multilingual evaluations on semantic and syntactic tasks show that PIXEL-M4 outperforms an English-only counterpart on non-Latin scripts. Word-level probing analyses confirm that PIXEL-M4 captures rich linguistic features, even in languages not seen during pretraining. Furthermore, an analysis of its hidden representations shows that multilingual pretraining yields a semantic embedding space closely aligned across the languages used for pretraining. This work demonstrates that multilingual pretraining substantially enhances the capability of pixel language models to effectively support a diverse set of languages. 

---
# ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision 

**Authors**: Dosung Lee, Wonjun Oh, Boyoung Kim, Minyoung Kim, Joonsuk Park, Paul Hongsuck Seo  

**Link**: [PDF](https://arxiv.org/pdf/2505.21250)  

**Abstract**: Multi-hop question answering (MHQA) involves reasoning across multiple documents to answer complex questions. Dense retrievers typically outperform sparse methods like BM25 by leveraging semantic embeddings; however, they require labeled query-document pairs for fine-tuning. This poses a significant challenge in MHQA due to the high variability of queries (reformulated) questions throughout the reasoning steps. To overcome this limitation, we introduce Retriever Supervision with Consistency and Relevance (ReSCORE), a novel method for training dense retrievers for MHQA without labeled documents. ReSCORE leverages large language models to capture each documents relevance to the question and consistency with the correct answer and use them to train a retriever within an iterative question-answering framework. Experiments on three MHQA benchmarks demonstrate the effectiveness of ReSCORE, with significant improvements in retrieval, and in turn, the state-of-the-art MHQA performance. Our implementation is available at: this https URL. 

---
# Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings 

**Authors**: Gunjan Balde, Soumyadeep Roy, Mainack Mondal, Niloy Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2505.21242)  

**Abstract**: Large Language Models (LLMs) recently achieved great success in medical text summarization by simply using in-context learning. However, these recent efforts do not perform fine-grained evaluations under difficult settings where LLMs might fail. They typically report performance scores over the entire dataset. Through our benchmarking study, we show that LLMs show a significant performance drop for data points with high concentration of out-of-vocabulary (OOV) words or with high novelty. Vocabulary adaptation is an intuitive solution to this vocabulary mismatch issue where the LLM vocabulary gets updated with certain expert domain (here, medical) words or subwords. An interesting finding from our study is that Llama-3.1, even with a vocabulary size of around 128K tokens, still faces over-fragmentation issue with medical words. To that end, we show vocabulary adaptation helps improve the LLM summarization performance even in difficult settings. Through extensive experimentation of multiple vocabulary adaptation strategies, two continual pretraining strategies, and three benchmark medical summarization datasets, we gain valuable insights into the role of vocabulary adaptation strategies for customizing LLMs to the medical domain. We also performed a human evaluation study with medical experts where they found that vocabulary adaptation results in more relevant and faithful summaries. Our codebase is made publicly available at this https URL. 

---
# LMCD: Language Models are Zeroshot Cognitive Diagnosis Learners 

**Authors**: Yu He, Zihan Yao, Chentao Song, Tianyu Qi, Jun Liu, Ming Li, Qing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21239)  

**Abstract**: Cognitive Diagnosis (CD) has become a critical task in AI-empowered education, supporting personalized learning by accurately assessing students' cognitive states. However, traditional CD models often struggle in cold-start scenarios due to the lack of student-exercise interaction data. Recent NLP-based approaches leveraging pre-trained language models (PLMs) have shown promise by utilizing textual features but fail to fully bridge the gap between semantic understanding and cognitive profiling. In this work, we propose Language Models as Zeroshot Cognitive Diagnosis Learners (LMCD), a novel framework designed to handle cold-start challenges by harnessing large language models (LLMs). LMCD operates via two primary phases: (1) Knowledge Diffusion, where LLMs generate enriched contents of exercises and knowledge concepts (KCs), establishing stronger semantic links; and (2) Semantic-Cognitive Fusion, where LLMs employ causal attention mechanisms to integrate textual information and student cognitive states, creating comprehensive profiles for both students and exercises. These representations are efficiently trained with off-the-shelf CD models. Experiments on two real-world datasets demonstrate that LMCD significantly outperforms state-of-the-art methods in both exercise-cold and domain-cold settings. The code is publicly available at this https URL 

---
# A Representation Level Analysis of NMT Model Robustness to Grammatical Errors 

**Authors**: Abderrahmane Issam, Yusuf Can Semerci, Jan Scholtes, Gerasimos Spanakis  

**Link**: [PDF](https://arxiv.org/pdf/2505.21224)  

**Abstract**: Understanding robustness is essential for building reliable NLP systems. Unfortunately, in the context of machine translation, previous work mainly focused on documenting robustness failures or improving robustness. In contrast, we study robustness from a model representation perspective by looking at internal model representations of ungrammatical inputs and how they evolve through model layers. For this purpose, we perform Grammatical Error Detection (GED) probing and representational similarity analysis. Our findings indicate that the encoder first detects the grammatical error, then corrects it by moving its representation toward the correct form. To understand what contributes to this process, we turn to the attention mechanism where we identify what we term Robustness Heads. We find that Robustness Heads attend to interpretable linguistic units when responding to grammatical errors, and that when we fine-tune models for robustness, they tend to rely more on Robustness Heads for updating the ungrammatical word representation. 

---
# Pretrained LLMs Learn Multiple Types of Uncertainty 

**Authors**: Roi Cohen, Omri Fahn, Gerard de Melo  

**Link**: [PDF](https://arxiv.org/pdf/2505.21218)  

**Abstract**: Large Language Models are known to capture real-world knowledge, allowing them to excel in many downstream tasks. Despite recent advances, these models are still prone to what are commonly known as hallucinations, causing them to emit unwanted and factually incorrect text. In this work, we study how well LLMs capture uncertainty, without explicitly being trained for that. We show that, if considering uncertainty as a linear concept in the model's latent space, it might indeed be captured, even after only pretraining. We further show that, though unintuitive, LLMs appear to capture several different types of uncertainty, each of which can be useful to predict the correctness for a specific task or benchmark. Furthermore, we provide in-depth results such as demonstrating a correlation between our correction prediction and the model's ability to abstain from misinformation using words, and the lack of impact of model scaling for capturing uncertainty. Finally, we claim that unifying the uncertainty types as a single one using instruction-tuning or [IDK]-token tuning is helpful for the model in terms of correctness prediction. 

---
# Unveiling Instruction-Specific Neurons & Experts: An Analytical Framework for LLM's Instruction-Following Capabilities 

**Authors**: Junyan Zhang, Yubo Gao, Yibo Yan, Jungang Li, Zhaorui Hou, Sicheng Tao, Shuliang Liu, Song Dai, Yonghua Hei, Junzhuo Li, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21191)  

**Abstract**: The finetuning of Large Language Models (LLMs) has significantly advanced their instruction-following capabilities, yet the underlying computational mechanisms driving these improvements remain poorly understood. This study systematically examines how fine-tuning reconfigures LLM computations by isolating and analyzing instruction-specific sparse components, i.e., neurons in dense models and both neurons and experts in Mixture-of-Experts (MoE) architectures. In particular, we introduce HexaInst, a carefully curated and balanced instructional dataset spanning six distinct categories, and propose SPARCOM, a novel analytical framework comprising three key contributions: (1) a method for identifying these sparse components, (2) an evaluation of their functional generality and uniqueness, and (3) a systematic comparison of their alterations. Through experiments, we demonstrate functional generality, uniqueness, and the critical role of these components in instruction execution. By elucidating the relationship between fine-tuning-induced adaptations and sparse computational substrates, this work provides deeper insights into how LLMs internalize instruction-following behavior for the trustworthy LLM community. 

---
# Lunguage: A Benchmark for Structured and Sequential Chest X-ray Interpretation 

**Authors**: Jong Hak Moon, Geon Choi, Paloma Rabaey, Min Gwan Kim, Hyuk Gi Hong, Jung-Oh Lee, Hangyul Yoon, Eun Woo Doe, Jiyoun Kim, Harshita Sharma, Daniel C. Castro, Javier Alvarez-Valle, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.21190)  

**Abstract**: Radiology reports convey detailed clinical observations and capture diagnostic reasoning that evolves over time. However, existing evaluation methods are limited to single-report settings and rely on coarse metrics that fail to capture fine-grained clinical semantics and temporal dependencies. We introduce LUNGUAGE,a benchmark dataset for structured radiology report generation that supports both single-report evaluation and longitudinal patient-level assessment across multiple studies. It contains 1,473 annotated chest X-ray reports, each reviewed by experts, and 80 of them contain longitudinal annotations to capture disease progression and inter-study intervals, also reviewed by experts. Using this benchmark, we develop a two-stage framework that transforms generated reports into fine-grained, schema-aligned structured representations, enabling longitudinal interpretation. We also propose LUNGUAGESCORE, an interpretable metric that compares structured outputs at the entity, relation, and attribute level while modeling temporal consistency across patient timelines. These contributions establish the first benchmark dataset, structuring framework, and evaluation metric for sequential radiology reporting, with empirical results demonstrating that LUNGUAGESCORE effectively supports structured report evaluation. The code is available at: this https URL 

---
# Exploring the Latent Capacity of LLMs for One-Step Text Generation 

**Authors**: Gleb Mezentsev, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2505.21189)  

**Abstract**: A recent study showed that large language models (LLMs) can reconstruct surprisingly long texts - up to thousands of tokens - via autoregressive generation from just one specially trained input embedding. In this work, we explore whether such reconstruction is possible without autoregression. We show that frozen LLMs can generate hundreds of accurate tokens in just one forward pass, when provided with only two learned embeddings. This reveals a surprising and underexplored capability of LLMs - multi-token generation without iterative decoding. We investigate the behaviour of these embeddings and provide insight into the type of information they encode. We also empirically show that although these representations are not unique for a given text, they form connected and local regions in embedding space - a property that suggests the potential of learning a dedicated encoder into that space. 

---
# Walk Before You Run! Concise LLM Reasoning via Reinforcement Learning 

**Authors**: Mingyang Song, Mao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.21178)  

**Abstract**: As test-time scaling becomes a pivotal research frontier in Large Language Models (LLMs) development, contemporary and advanced post-training methodologies increasingly focus on extending the generation length of long Chain-of-Thought (CoT) responses to enhance reasoning capabilities toward DeepSeek R1-like performance. However, recent studies reveal a persistent overthinking phenomenon in state-of-the-art reasoning models, manifesting as excessive redundancy or repetitive thinking patterns in long CoT responses. To address this issue, in this paper, we propose a simple yet effective two-stage reinforcement learning framework for achieving concise reasoning in LLMs, named ConciseR. Specifically, the first stage, using more training steps, aims to incentivize the model's reasoning capabilities via Group Relative Policy Optimization with clip-higher and dynamic sampling components (GRPO++), and the second stage, using fewer training steps, explicitly enforces conciseness and improves efficiency via Length-aware Group Relative Policy Optimization (L-GRPO). Significantly, ConciseR only optimizes response length once all rollouts of a sample are correct, following the "walk before you run" principle. Extensive experimental results demonstrate that our ConciseR model, which generates more concise CoT reasoning responses, outperforms recent state-of-the-art reasoning models with zero RL paradigm across AIME 2024, MATH-500, AMC 2023, Minerva, and Olympiad benchmarks. 

---
# TAT-R1: Terminology-Aware Translation with Reinforcement Learning and Word Alignment 

**Authors**: Zheng Li, Mao Zheng, Mingyang Song, Wenjie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21172)  

**Abstract**: Recently, deep reasoning large language models(LLMs) like DeepSeek-R1 have made significant progress in tasks such as mathematics and coding. Inspired by this, several studies have employed reinforcement learning(RL) to enhance models' deep reasoning capabilities and improve machine translation(MT) quality. However, the terminology translation, an essential task in MT, remains unexplored in deep reasoning LLMs. In this paper, we propose \textbf{TAT-R1}, a terminology-aware translation model trained with reinforcement learning and word alignment. Specifically, we first extract the keyword translation pairs using a word alignment model. Then we carefully design three types of rule-based alignment rewards with the extracted alignment relationships. With those alignment rewards, the RL-trained translation model can learn to focus on the accurate translation of key information, including terminology in the source text. Experimental results show the effectiveness of TAT-R1. Our model significantly improves terminology translation accuracy compared to the baseline models while maintaining comparable performance on general translation tasks. In addition, we conduct detailed ablation studies of the DeepSeek-R1-like training paradigm for machine translation and reveal several key findings. 

---
# M-Wanda: Improving One-Shot Pruning for Multilingual LLMs 

**Authors**: Rochelle Choenni, Ivan Titov  

**Link**: [PDF](https://arxiv.org/pdf/2505.21171)  

**Abstract**: Multilingual LLM performance is often critically dependent on model size. With an eye on efficiency, this has led to a surge in interest in one-shot pruning methods that retain the benefits of large-scale pretraining while shrinking the model size. However, as pruning tends to come with performance loss, it is important to understand the trade-offs between multilinguality and sparsification. In this work, we study multilingual performance under different sparsity constraints and show that moderate ratios already substantially harm performance. To help bridge this gap, we propose M-Wanda, a pruning method that models cross-lingual variation by incorporating language-aware activation statistics into its pruning criterion and dynamically adjusts layerwise sparsity based on cross-lingual importance. We show that M-Wanda consistently improves performance at minimal additional costs. We are the first to explicitly optimize pruning to retain multilingual performance, and hope to inspire future advances in multilingual pruning. 

---
# Assessment of L2 Oral Proficiency using Speech Large Language Models 

**Authors**: Rao Ma, Mengjie Qian, Siyuan Tang, Stefano Bannò, Kate M. Knill, Mark J.F. Gales  

**Link**: [PDF](https://arxiv.org/pdf/2505.21148)  

**Abstract**: The growing population of L2 English speakers has increased the demand for developing automatic graders for spoken language assessment (SLA). Historically, statistical models, text encoders, and self-supervised speech models have been utilised for this task. However, cascaded systems suffer from the loss of information, while E2E graders also have limitations. With the recent advancements of multi-modal large language models (LLMs), we aim to explore their potential as L2 oral proficiency graders and overcome these issues. In this work, we compare various training strategies using regression and classification targets. Our results show that speech LLMs outperform all previous competitive baselines, achieving superior performance on two datasets. Furthermore, the trained grader demonstrates strong generalisation capabilities in the cross-part or cross-task evaluation, facilitated by the audio understanding knowledge acquired during LLM pre-training. 

---
# Leveraging LLM and Self-Supervised Training Models for Speech Recognition in Chinese Dialects: A Comparative Analysis 

**Authors**: Tianyi Xu, Hongjie Chen, Wang Qing, Lv Hang, Jian Kang, Li Jie, Zhennan Lin, Yongxiang Li, Xie Lei  

**Link**: [PDF](https://arxiv.org/pdf/2505.21138)  

**Abstract**: Large-scale training corpora have significantly improved the performance of ASR models. Unfortunately, due to the relative scarcity of data, Chinese accents and dialects remain a challenge for most ASR models. Recent advancements in self-supervised learning have shown that self-supervised pre- training, combined with large language models (LLM), can effectively enhance ASR performance in low-resource scenarios. We aim to investigate the effectiveness of this paradigm for Chinese dialects. Specifically, we pre-train a Data2vec2 model on 300,000 hours of unlabeled dialect and accented speech data and do alignment training on a supervised dataset of 40,000 hours. Then, we systematically examine the impact of various projectors and LLMs on Mandarin, dialect, and accented speech recognition performance under this paradigm. Our method achieved SOTA results on multiple dialect datasets, including Kespeech. We will open-source our work to promote reproducible research 

---
# Scaling and Prompting for Improved End-to-End Spoken Grammatical Error Correction 

**Authors**: Mengjie Qian, Rao Ma, Stefano Bannò, Kate M. Knill, Mark J.F. Gales  

**Link**: [PDF](https://arxiv.org/pdf/2505.21137)  

**Abstract**: Spoken Grammatical Error Correction (SGEC) and Feedback (SGECF) are crucial for second language learners, teachers and test takers. Traditional SGEC systems rely on a cascaded pipeline consisting of an ASR, a module for disfluency detection (DD) and removal and one for GEC. With the rise of end-to-end (E2E) speech foundation models, we investigate their effectiveness in SGEC and feedback generation. This work introduces a pseudo-labelling process to address the challenge of limited labelled data, expanding the training data size from 77 hours to approximately 2500 hours, leading to improved performance. Additionally, we prompt an E2E Whisper-based SGEC model with fluent transcriptions, showing a slight improvement in SGEC performance, with more significant gains in feedback generation. Finally, we assess the impact of increasing model size, revealing that while pseudo-labelled data does not yield performance gain for a larger Whisper model, training with prompts proves beneficial. 

---
# Will It Still Be True Tomorrow? Multilingual Evergreen Question Classification to Improve Trustworthy QA 

**Authors**: Sergey Pletenev, Maria Marina, Nikolay Ivanov, Daria Galimzianova, Nikita Krayko, Mikhail Salnikov, Vasily Konovalov, Alexander Panchenko, Viktor Moskvoretskii  

**Link**: [PDF](https://arxiv.org/pdf/2505.21115)  

**Abstract**: Large Language Models (LLMs) often hallucinate in question answering (QA) tasks. A key yet underexplored factor contributing to this is the temporality of questions -- whether they are evergreen (answers remain stable over time) or mutable (answers change). In this work, we introduce EverGreenQA, the first multilingual QA dataset with evergreen labels, supporting both evaluation and training. Using EverGreenQA, we benchmark 12 modern LLMs to assess whether they encode question temporality explicitly (via verbalized judgments) or implicitly (via uncertainty signals). We also train EG-E5, a lightweight multilingual classifier that achieves SoTA performance on this task. Finally, we demonstrate the practical utility of evergreen classification across three applications: improving self-knowledge estimation, filtering QA datasets, and explaining GPT-4o retrieval behavior. 

---
# A Lightweight Multi-Expert Generative Language Model System for Engineering Information and Knowledge Extraction 

**Authors**: Bogdan Bogachov, Yaoyao Fiona Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.21109)  

**Abstract**: Despite recent advancements in domain adaptation techniques for large language models, these methods remain computationally intensive, and the resulting models can still exhibit hallucination issues. Most existing adaptation methods do not prioritize reducing the computational resources required for fine-tuning and inference of language models. Hallucination issues have gradually decreased with each new model release. However, they remain prevalent in engineering contexts, where generating well-structured text with minimal errors and inconsistencies is critical. This work introduces a novel approach called the Small Language Graph (SLG), which is a lightweight adaptation solution designed to address the two key challenges outlined above. The system is structured in the form of a graph, where each node represents a lightweight expert - a small language model fine-tuned on specific and concise texts. The results of this study have shown that SLG was able to surpass conventional fine-tuning methods on the Exact Match metric by 3 times. Additionally, the fine-tuning process was 1.7 times faster compared to that of a larger stand-alone language model. These findings introduce a potential for small to medium-sized engineering companies to confidently use generative AI technologies, such as LLMs, without the necessity to invest in expensive computational resources. Also, the graph architecture and the small size of expert nodes offer a possible opportunity for distributed AI systems, thus potentially diverting the global need for expensive centralized compute clusters. 

---
# Thinker: Learning to Think Fast and Slow 

**Authors**: Stephen Chung, Wenyu Du, Jie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21097)  

**Abstract**: Recent studies show that the reasoning capabilities of Large Language Models (LLMs) can be improved by applying Reinforcement Learning (RL) to question-answering (QA) tasks in areas such as math and coding. With a long context length, LLMs may learn to perform search, as indicated by the self-correction behavior observed in DeepSeek R1. However, this search behavior is often imprecise and lacks confidence, resulting in long, redundant responses and highlighting deficiencies in intuition and verification. Inspired by the Dual Process Theory in psychology, we introduce a simple modification to the QA task that includes four stages: Fast Thinking, where the LLM must answer within a strict token budget; Verification, where the model evaluates its initial response; Slow Thinking, where it refines the initial response with more deliberation; and Summarization, where it distills the refinement from the previous stage into precise steps. Our proposed task improves average accuracy from 24.9% to 27.9% for Qwen2.5-1.5B, and from 45.9% to 49.8% for DeepSeek-R1-Qwen-1.5B. Notably, for Qwen2.5-1.5B, the Fast Thinking mode alone achieves 26.8% accuracy using fewer than 1000 tokens, demonstrating substantial inference efficiency gains. These findings suggest that intuition and deliberative reasoning are distinct, complementary systems benefiting from targeted training. 

---
# BLUCK: A Benchmark Dataset for Bengali Linguistic Understanding and Cultural Knowledge 

**Authors**: Daeen Kabir, Minhajur Rahman Chowdhury Mahim, Sheikh Shafayat, Adnan Sadik, Arian Ahmed, Eunsu Kim, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2505.21092)  

**Abstract**: In this work, we introduce BLUCK, a new dataset designed to measure the performance of Large Language Models (LLMs) in Bengali linguistic understanding and cultural knowledge. Our dataset comprises 2366 multiple-choice questions (MCQs) carefully curated from compiled collections of several college and job level examinations and spans 23 categories covering knowledge on Bangladesh's culture and history and Bengali linguistics. We benchmarked BLUCK using 6 proprietary and 3 open-source LLMs - including GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Pro, Llama-3.3-70B-Instruct, and DeepSeekV3. Our results show that while these models perform reasonably well overall, they, however, struggles in some areas of Bengali phonetics. Although current LLMs' performance on Bengali cultural and linguistic contexts is still not comparable to that of mainstream languages like English, our results indicate Bengali's status as a mid-resource language. Importantly, BLUCK is also the first MCQ-based evaluation benchmark that is centered around native Bengali culture, history, and linguistics. 

---
# LLMs Think, But Not In Your Flow: Reasoning-Level Personalization for Black-Box Large Language Models 

**Authors**: Jieyong Kim, Tongyoung Kim, Soonjin Yoon, Jaehyung Kim, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.21082)  

**Abstract**: Large language models (LLMs) have recently achieved impressive performance across a wide range of natural language tasks and are now widely used in real-world applications. Among them, black-box LLMs--served via APIs without access to model internals--are especially dominant due to their scalability and ease of deployment. Despite their strong capabilities, these models typically produce generalized responses that overlook personal preferences and reasoning styles. This has led to growing interest in black-box LLM personalization, which aims to tailor model outputs to user-specific context without modifying model parameters. However, existing approaches primarily focus on response-level personalization, attempting to match final outputs without modeling personal thought process. To address this limitation, we propose RPM, a framework for reasoning-level personalization that aligns the model's reasoning process with a user's personalized logic. RPM first constructs statistical user-specific factors by extracting and grouping response-influential features from user history. It then builds personalized reasoning paths that reflect how these factors are used in context. In the inference stage, RPM retrieves reasoning-aligned examples for new queries via feature-level similarity and performs inference conditioned on the structured factors and retrieved reasoning paths, enabling the model to follow user-specific reasoning trajectories. This reasoning-level personalization enhances both predictive accuracy and interpretability by grounding model outputs in user-specific logic through structured information. Extensive experiments across diverse tasks show that RPM consistently outperforms response-level personalization methods, demonstrating the effectiveness of reasoning-level personalization in black-box LLMs. 

---
# Faithfulness-Aware Uncertainty Quantification for Fact-Checking the Output of Retrieval Augmented Generation 

**Authors**: Ekaterina Fadeeva, Aleksandr Rubashevskii, Roman Vashurin, Shehzaad Dhuliawala, Artem Shelmanov, Timothy Baldwin, Preslav Nakov, Mrinmaya Sachan, Maxim Panov  

**Link**: [PDF](https://arxiv.org/pdf/2505.21072)  

**Abstract**: Large Language Models (LLMs) enhanced with external knowledge retrieval, an approach known as Retrieval-Augmented Generation (RAG), have shown strong performance in open-domain question answering. However, RAG systems remain susceptible to hallucinations: factually incorrect outputs that may arise either from inconsistencies in the model's internal knowledge or incorrect use of the retrieved context. Existing approaches often conflate factuality with faithfulness to the retrieved context, misclassifying factually correct statements as hallucinations if they are not directly supported by the retrieval. In this paper, we introduce FRANQ (Faithfulness-based Retrieval Augmented UNcertainty Quantification), a novel method for hallucination detection in RAG outputs. FRANQ applies different Uncertainty Quantification (UQ) techniques to estimate factuality based on whether a statement is faithful to the retrieved context or not. To evaluate FRANQ and other UQ techniques for RAG, we present a new long-form Question Answering (QA) dataset annotated for both factuality and faithfulness, combining automated labeling with manual validation of challenging examples. Extensive experiments on long- and short-form QA across multiple datasets and LLMs show that FRANQ achieves more accurate detection of factual errors in RAG-generated responses compared to existing methods. 

---
# Predicting Implicit Arguments in Procedural Video Instructions 

**Authors**: Anil Batra, Laura Sevilla-Lara, Marcus Rohrbach, Frank Keller  

**Link**: [PDF](https://arxiv.org/pdf/2505.21068)  

**Abstract**: Procedural texts help AI enhance reasoning about context and action sequences. Transforming these into Semantic Role Labeling (SRL) improves understanding of individual steps by identifying predicate-argument structure like {verb,what,where/with}. Procedural instructions are highly elliptic, for instance, (i) add cucumber to the bowl and (ii) add sliced tomatoes, the second step's where argument is inferred from the context, referring to where the cucumber was placed. Prior SRL benchmarks often miss implicit arguments, leading to incomplete understanding. To address this, we introduce Implicit-VidSRL, a dataset that necessitates inferring implicit and explicit arguments from contextual information in multimodal cooking procedures. Our proposed dataset benchmarks multimodal models' contextual reasoning, requiring entity tracking through visual changes in recipes. We study recent multimodal LLMs and reveal that they struggle to predict implicit arguments of what and where/with from multi-modal procedural data given the verb. Lastly, we propose iSRL-Qwen2-VL, which achieves a 17% relative improvement in F1-score for what-implicit and a 14.7% for where/with-implicit semantic roles over GPT-4o. 

---
# Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction 

**Authors**: Sam O'Connor Russell, Naomi Harte  

**Link**: [PDF](https://arxiv.org/pdf/2505.21043)  

**Abstract**: Turn-taking is richly multimodal. Predictive turn-taking models (PTTMs) facilitate naturalistic human-robot interaction, yet most rely solely on speech. We introduce MM-VAP, a multimodal PTTM which combines speech with visual cues including facial expression, head pose and gaze. We find that it outperforms the state-of-the-art audio-only in videoconferencing interactions (84% vs. 79% hold/shift prediction accuracy). Unlike prior work which aggregates all holds and shifts, we group by duration of silence between turns. This reveals that through the inclusion of visual features, MM-VAP outperforms a state-of-the-art audio-only turn-taking model across all durations of speaker transitions. We conduct a detailed ablation study, which reveals that facial expression features contribute the most to model performance. Thus, our working hypothesis is that when interlocutors can see one another, visual cues are vital for turn-taking and must therefore be included for accurate turn-taking prediction. We additionally validate the suitability of automatic speech alignment for PTTM training using telephone speech. This work represents the first comprehensive analysis of multimodal PTTMs. We discuss implications for future work and make all code publicly available. 

---
# FCKT: Fine-Grained Cross-Task Knowledge Transfer with Semantic Contrastive Learning for Targeted Sentiment Analysis 

**Authors**: Wei Chen, Zhao Zhang, Meng Yuan, Kepeng Xu, Fuzhen Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21040)  

**Abstract**: In this paper, we address the task of targeted sentiment analysis (TSA), which involves two sub-tasks, i.e., identifying specific aspects from reviews and determining their corresponding sentiments. Aspect extraction forms the foundation for sentiment prediction, highlighting the critical dependency between these two tasks for effective cross-task knowledge transfer. While most existing studies adopt a multi-task learning paradigm to align task-specific features in the latent space, they predominantly rely on coarse-grained knowledge transfer. Such approaches lack fine-grained control over aspect-sentiment relationships, often assuming uniform sentiment polarity within related aspects. This oversimplification neglects contextual cues that differentiate sentiments, leading to negative transfer. To overcome these limitations, we propose FCKT, a fine-grained cross-task knowledge transfer framework tailored for TSA. By explicitly incorporating aspect-level information into sentiment prediction, FCKT achieves fine-grained knowledge transfer, effectively mitigating negative transfer and enhancing task performance. Experiments on three datasets, including comparisons with various baselines and large language models (LLMs), demonstrate the effectiveness of FCKT. The source code is available on this https URL. 

---
# Def-DTS: Deductive Reasoning for Open-domain Dialogue Topic Segmentation 

**Authors**: Seungmin Lee, Yongsang Yoo, Minhwa Jung, Min Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.21033)  

**Abstract**: Dialogue Topic Segmentation (DTS) aims to divide dialogues into coherent segments. DTS plays a crucial role in various NLP downstream tasks, but suffers from chronic problems: data shortage, labeling ambiguity, and incremental complexity of recently proposed solutions. On the other hand, Despite advances in Large Language Models (LLMs) and reasoning strategies, these have rarely been applied to DTS. This paper introduces Def-DTS: Deductive Reasoning for Open-domain Dialogue Topic Segmentation, which utilizes LLM-based multi-step deductive reasoning to enhance DTS performance and enable case study using intermediate result. Our method employs a structured prompting approach for bidirectional context summarization, utterance intent classification, and deductive topic shift detection. In the intent classification process, we propose the generalizable intent list for domain-agnostic dialogue intent classification. Experiments in various dialogue settings demonstrate that Def-DTS consistently outperforms traditional and state-of-the-art approaches, with each subtask contributing to improved performance, particularly in reducing type 2 error. We also explore the potential for autolabeling, emphasizing the importance of LLM reasoning techniques in DTS. 

---
# LLMs are Frequency Pattern Learners in Natural Language Inference 

**Authors**: Liang Cheng, Zhaowei Wang, Mark Steedman  

**Link**: [PDF](https://arxiv.org/pdf/2505.21011)  

**Abstract**: While fine-tuning LLMs on NLI corpora improves their inferential performance, the underlying mechanisms driving this improvement remain largely opaque. In this work, we conduct a series of experiments to investigate what LLMs actually learn during fine-tuning. We begin by analyzing predicate frequencies in premises and hypotheses across NLI datasets and identify a consistent frequency bias, where predicates in hypotheses occur more frequently than those in premises for positive instances. To assess the impact of this bias, we evaluate both standard and NLI fine-tuned LLMs on bias-consistent and bias-adversarial cases. We find that LLMs exploit frequency bias for inference and perform poorly on adversarial instances. Furthermore, fine-tuned LLMs exhibit significantly increased reliance on this bias, suggesting that they are learning these frequency patterns from datasets. Finally, we compute the frequencies of hyponyms and their corresponding hypernyms from WordNet, revealing a correlation between frequency bias and textual entailment. These findings help explain why learning frequency patterns can enhance model performance on inference tasks. 

---
# Uncertainty Unveiled: Can Exposure to More In-context Examples Mitigate Uncertainty for Large Language Models? 

**Authors**: Yifei Wang, Yu Sheng, Linjing Li, Daniel Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.21003)  

**Abstract**: Recent advances in handling long sequences have facilitated the exploration of long-context in-context learning (ICL). While much of the existing research emphasizes performance improvements driven by additional in-context examples, the influence on the trustworthiness of generated responses remains underexplored. This paper addresses this gap by investigating how increased examples influence predictive uncertainty, an essential aspect in trustworthiness. We begin by systematically quantifying the uncertainty of ICL with varying shot counts, analyzing the impact of example quantity. Through uncertainty decomposition, we introduce a novel perspective on performance enhancement, with a focus on epistemic uncertainty (EU). Our results reveal that additional examples reduce total uncertainty in both simple and complex tasks by injecting task-specific knowledge, thereby diminishing EU and enhancing performance. For complex tasks, these advantages emerge only after addressing the increased noise and uncertainty associated with longer inputs. Finally, we explore the evolution of internal confidence across layers, unveiling the mechanisms driving the reduction in uncertainty. 

---
# Articulatory strategy in vowel production as a basis for speaker discrimination 

**Authors**: Justin J. H. Lo, Patrycja Strycharczuk, Sam Kirkham  

**Link**: [PDF](https://arxiv.org/pdf/2505.20995)  

**Abstract**: The way speakers articulate is well known to be variable across individuals while at the same time subject to anatomical and biomechanical constraints. In this study, we ask whether articulatory strategy in vowel production can be sufficiently speaker-specific to form the basis for speaker discrimination. We conducted Generalised Procrustes Analyses of tongue shape data from 40 English speakers from the North West of England, and assessed the speaker-discriminatory potential of orthogonal tongue shape features within the framework of likelihood ratios. Tongue size emerged as the individual dimension with the strongest discriminatory power, while tongue shape variation in the more anterior part of the tongue generally outperformed tongue shape variation in the posterior part. When considered in combination, shape-only information may offer comparable levels of speaker specificity to size-and-shape information, but only when features do not exhibit speaker-level co-variation. 

---
# Who Reasons in the Large Language Models? 

**Authors**: Jie Shao, Jianxin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20993)  

**Abstract**: Despite the impressive performance of large language models (LLMs), the process of endowing them with new capabilities--such as mathematical reasoning--remains largely empirical and opaque. A critical open question is whether reasoning abilities stem from the entire model, specific modules, or are merely artifacts of overfitting. In this work, we hypothesize that the reasoning capabilities in well-trained LLMs are primarily attributed to the output projection module (oproj) in the Transformer's multi-head self-attention (MHSA) mechanism. To support this hypothesis, we introduce Stethoscope for Networks (SfN), a suite of diagnostic tools designed to probe and analyze the internal behaviors of LLMs. Using SfN, we provide both circumstantial and empirical evidence suggesting that oproj plays a central role in enabling reasoning, whereas other modules contribute more to fluent dialogue. These findings offer a new perspective on LLM interpretability and open avenues for more targeted training strategies, potentially enabling more efficient and specialized LLMs. 

---
# Evaluating and Steering Modality Preferences in Multimodal Large Language Model 

**Authors**: Yu Zhang, Jinlong Ma, Yongshuai Hou, Xuefeng Bai, Kehai Chen, Yang Xiang, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20977)  

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable performance on complex tasks with multimodal context. However, it is still understudied whether they exhibit modality preference when processing multimodal contexts. To study this question, we first build a \textbf{MC\textsuperscript{2}} benchmark under controlled evidence conflict scenarios to systematically evaluate modality preference, which is the tendency to favor one modality over another when making decisions based on multimodal conflicting evidence. Our extensive evaluation reveals that all 18 tested MLLMs generally demonstrate clear modality bias, and modality preference can be influenced by external interventions. An in-depth analysis reveals that the preference direction can be captured within the latent representations of MLLMs. Built on this, we propose a probing and steering method based on representation engineering to explicitly control modality preference without additional fine-tuning or carefully crafted prompts. Our method effectively amplifies modality preference toward a desired direction and applies to downstream tasks such as hallucination mitigation and multimodal machine translation, yielding promising improvements. 

---
# Contrastive Learning on LLM Back Generation Treebank for Cross-domain Constituency Parsing 

**Authors**: Peiming Guo, Meishan Zhang, Jianling Li, Min Zhang, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20976)  

**Abstract**: Cross-domain constituency parsing is still an unsolved challenge in computational linguistics since the available multi-domain constituency treebank is limited. We investigate automatic treebank generation by large language models (LLMs) in this paper. The performance of LLMs on constituency parsing is poor, therefore we propose a novel treebank generation method, LLM back generation, which is similar to the reverse process of constituency parsing. LLM back generation takes the incomplete cross-domain constituency tree with only domain keyword leaf nodes as input and fills the missing words to generate the cross-domain constituency treebank. Besides, we also introduce a span-level contrastive learning pre-training strategy to make full use of the LLM back generation treebank for cross-domain constituency parsing. We verify the effectiveness of our LLM back generation treebank coupled with contrastive learning pre-training on five target domains of MCTB. Experimental results show that our approach achieves state-of-the-art performance on average results compared with various baselines. 

---
# Reason-Align-Respond: Aligning LLM Reasoning with Knowledge Graphs for KGQA 

**Authors**: Xiangqing Shen, Fanfan Wang, Rui Xia  

**Link**: [PDF](https://arxiv.org/pdf/2505.20971)  

**Abstract**: LLMs have demonstrated remarkable capabilities in complex reasoning tasks, yet they often suffer from hallucinations and lack reliable factual grounding. Meanwhile, knowledge graphs (KGs) provide structured factual knowledge but lack the flexible reasoning abilities of LLMs. In this paper, we present Reason-Align-Respond (RAR), a novel framework that systematically integrates LLM reasoning with knowledge graphs for KGQA. Our approach consists of three key components: a Reasoner that generates human-like reasoning chains, an Aligner that maps these chains to valid KG paths, and a Responser that synthesizes the final answer. We formulate this process as a probabilistic model and optimize it using the Expectation-Maximization algorithm, which iteratively refines the reasoning chains and knowledge paths. Extensive experiments on multiple benchmarks demonstrate the effectiveness of RAR, achieving state-of-the-art performance with Hit@1 scores of 93.3% and 91.0% on WebQSP and CWQ respectively. Human evaluation confirms that RAR generates high-quality, interpretable reasoning chains well-aligned with KG paths. Furthermore, RAR exhibits strong zero-shot generalization capabilities and maintains computational efficiency during inference. 

---
# Personalized Query Auto-Completion for Long and Short-Term Interests with Adaptive Detoxification Generation 

**Authors**: Zhibo Wang, Xiaoze Jiang, Zhiheng Qin, Enyun Yu, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20966)  

**Abstract**: Query auto-completion (QAC) plays a crucial role in modern search systems. However, in real-world applications, there are two pressing challenges that still need to be addressed. First, there is a need for hierarchical personalized representations for users. Previous approaches have typically used users' search behavior as a single, overall representation, which proves inadequate in more nuanced generative scenarios. Additionally, query prefixes are typically short and may contain typos or sensitive information, increasing the likelihood of generating toxic content compared to traditional text generation tasks. Such toxic content can degrade user experience and lead to public relations issues. Therefore, the second critical challenge is detoxifying QAC systems.
To address these two limitations, we propose a novel model (LaD) that captures personalized information from both long-term and short-term interests, incorporating adaptive detoxification. In LaD, personalized information is captured hierarchically at both coarse-grained and fine-grained levels. This approach preserves as much personalized information as possible while enabling online generation within time constraints. To move a futher step, we propose an online training method based on Reject Preference Optimization (RPO). By incorporating a special token [Reject] during both the training and inference processes, the model achieves adaptive detoxification. Consequently, the generated text presented to users is both non-toxic and relevant to the given prefix. We conduct comprehensive experiments on industrial-scale datasets and perform online A/B tests, delivering the largest single-experiment metric improvement in nearly two years of our product. Our model has been deployed on Kuaishou search, driving the primary traffic for hundreds of millions of active users. The code is available at this https URL. 

---
# Context-Aware Content Moderation for German Newspaper Comments 

**Authors**: Felix Krejca, Tobias Kietreiber, Alexander Buchelt, Sebastian Neumaier  

**Link**: [PDF](https://arxiv.org/pdf/2505.20963)  

**Abstract**: The increasing volume of online discussions requires advanced automatic content moderation to maintain responsible discourse. While hate speech detection on social media is well-studied, research on German-language newspaper forums remains limited. Existing studies often neglect platform-specific context, such as user history and article themes. This paper addresses this gap by developing and evaluating binary classification models for automatic content moderation in German newspaper forums, incorporating contextual information. Using LSTM, CNN, and ChatGPT-3.5 Turbo, and leveraging the One Million Posts Corpus from the Austrian newspaper Der Standard, we assess the impact of context-aware models. Results show that CNN and LSTM models benefit from contextual information and perform competitively with state-of-the-art approaches. In contrast, ChatGPT's zero-shot classification does not improve with added context and underperforms. 

---
# Research Community Perspectives on "Intelligence" and Large Language Models 

**Authors**: Bertram Højer, Terne Sasha Thorn Jakobsen, Anna Rogers, Stefan Heinrich  

**Link**: [PDF](https://arxiv.org/pdf/2505.20959)  

**Abstract**: Despite the widespread use of ''artificial intelligence'' (AI) framing in Natural Language Processing (NLP) research, it is not clear what researchers mean by ''intelligence''. To that end, we present the results of a survey on the notion of ''intelligence'' among researchers and its role in the research agenda. The survey elicited complete responses from 303 researchers from a variety of fields including NLP, Machine Learning (ML), Cognitive Science, Linguistics, and Neuroscience. We identify 3 criteria of intelligence that the community agrees on the most: generalization, adaptability, & reasoning. Our results suggests that the perception of the current NLP systems as ''intelligent'' is a minority position (29%). Furthermore, only 16.2% of the respondents see developing intelligent systems as a research goal, and these respondents are more likely to consider the current systems intelligent. 

---
# On VLMs for Diverse Tasks in Multimodal Meme Classification 

**Authors**: Deepesh Gavit, Debajyoti Mazumder, Samiran Das, Jasabanta Patro  

**Link**: [PDF](https://arxiv.org/pdf/2505.20937)  

**Abstract**: In this paper, we present a comprehensive and systematic analysis of vision-language models (VLMs) for disparate meme classification tasks. We introduced a novel approach that generates a VLM-based understanding of meme images and fine-tunes the LLMs on textual understanding of the embedded meme text for improving the performance. Our contributions are threefold: (1) Benchmarking VLMs with diverse prompting strategies purposely to each sub-task; (2) Evaluating LoRA fine-tuning across all VLM components to assess performance gains; and (3) Proposing a novel approach where detailed meme interpretations generated by VLMs are used to train smaller language models (LLMs), significantly improving classification. The strategy of combining VLMs with LLMs improved the baseline performance by 8.34%, 3.52% and 26.24% for sarcasm, offensive and sentiment classification, respectively. Our results reveal the strengths and limitations of VLMs and present a novel strategy for meme understanding. 

---
# Information-Theoretic Complementary Prompts for Improved Continual Text Classification 

**Authors**: Duzhen Zhang, Yong Ren, Chenxing Li, Dong Yu, Tielin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20933)  

**Abstract**: Continual Text Classification (CTC) aims to continuously classify new text data over time while minimizing catastrophic forgetting of previously acquired knowledge. However, existing methods often focus on task-specific knowledge, overlooking the importance of shared, task-agnostic knowledge. Inspired by the complementary learning systems theory, which posits that humans learn continually through the interaction of two systems -- the hippocampus, responsible for forming distinct representations of specific experiences, and the neocortex, which extracts more general and transferable representations from past experiences -- we introduce Information-Theoretic Complementary Prompts (InfoComp), a novel approach for CTC. InfoComp explicitly learns two distinct prompt spaces: P(rivate)-Prompt and S(hared)-Prompt. These respectively encode task-specific and task-invariant knowledge, enabling models to sequentially learn classification tasks without relying on data replay. To promote more informative prompt learning, InfoComp uses an information-theoretic framework that maximizes mutual information between different parameters (or encoded representations). Within this framework, we design two novel loss functions: (1) to strengthen the accumulation of task-specific knowledge in P-Prompt, effectively mitigating catastrophic forgetting, and (2) to enhance the retention of task-invariant knowledge in S-Prompt, improving forward knowledge transfer. Extensive experiments on diverse CTC benchmarks show that our approach outperforms previous state-of-the-art methods. 

---
# Multi-objective Large Language Model Alignment with Hierarchical Experts 

**Authors**: Zhuo Li, Guodong Du, Weiyang Guo, Yigeng Zhou, Xiucheng Li, Wenya Wang, Fangming Liu, Yequan Wang, Deheng Ye, Min Zhang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20925)  

**Abstract**: Aligning large language models (LLMs) to simultaneously satisfy multiple objectives remains a significant challenge, especially given the diverse and often conflicting nature of human preferences. Existing alignment methods struggle to balance trade-offs effectively, often requiring costly retraining or yielding suboptimal results across the Pareto frontier of preferences. In this paper, we introduce \textit{HoE}(Hierarchical Mixture-of-Experts), a \textit{lightweight}, \textit{parameter-efficient}, and \textit{plug-and-play} approach that eliminates the need for model training, while enabling LLMs to adapt across the entire Pareto frontier and accommodate diverse user preferences. In particular, \textit{HoE} consists of three hierarchical components: LoRA Experts, Router Experts and Preference Routing, reaching optimal Pareto frontiers and achieving a trade-off between parameter size, training cost, and performance. We evaluate \textit{HoE} across various tasks on 14 objectives and 200 different preferences among 6 benchmarks, demonstrating superior performance over 15 recent baselines. Code is available in the supplementary materials. 

---
# Automatic Transmission for LLM Tiers: Optimizing Cost and Accuracy in Large Language Models 

**Authors**: Injae Na, Keonwoong Noh, Woohwan Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.20921)  

**Abstract**: LLM providers typically offer multiple LLM tiers, varying in performance and price. As NLP tasks become more complex and modularized, selecting the suitable LLM tier for each subtask is a key challenge to balance between cost and performance. To address the problem, we introduce LLM Automatic Transmission (LLM-AT) framework that automatically selects LLM tiers without training. LLM-AT consists of Starter, Generator, and Judge. The starter selects the initial LLM tier expected to solve the given question, the generator produces a response using the LLM of the selected tier, and the judge evaluates the validity of the response. If the response is invalid, LLM-AT iteratively upgrades to a higher-tier model, generates a new response, and re-evaluates until a valid response is obtained. Additionally, we propose accuracy estimator, which enables the suitable initial LLM tier selection without training. Given an input question, accuracy estimator estimates the expected accuracy of each LLM tier by computing the valid response rate across top-k similar queries from past inference records. Experiments demonstrate that LLM-AT achieves superior performance while reducing costs, making it a practical solution for real-world applications. 

---
# Automated Privacy Information Annotation in Large Language Model Interactions 

**Authors**: Hang Zeng, Xiangyu Liu, Yong Hu, Chaoyue Niu, Fan Wu, Shaojie Tang, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.20910)  

**Abstract**: Users interacting with large language models (LLMs) under their real identifiers often unknowingly risk disclosing private information. Automatically notifying users whether their queries leak privacy and which phrases leak what private information has therefore become a practical need. Existing privacy detection methods, however, were designed for different objectives and application scenarios, typically tagging personally identifiable information (PII) in anonymous content. In this work, to support the development and evaluation of privacy detection models for LLM interactions that are deployable on local user devices, we construct a large-scale multilingual dataset with 249K user queries and 154K annotated privacy phrases. In particular, we build an automated privacy annotation pipeline with cloud-based strong LLMs to automatically extract privacy phrases from dialogue datasets and annotate leaked information. We also design evaluation metrics at the levels of privacy leakage, extracted privacy phrase, and privacy information. We further establish baseline methods using light-weight LLMs with both tuning-free and tuning-based methods, and report a comprehensive evaluation of their performance. Evaluation results reveal a gap between current performance and the requirements of real-world LLM applications, motivating future research into more effective local privacy detection methods grounded in our dataset. 

---
# Towards Objective Fine-tuning: How LLMs' Prior Knowledge Causes Potential Poor Calibration? 

**Authors**: Ziming Wang, Zeyu Shi, Haoyi Zhou, Shiqi Gao, Qingyun Sun, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20903)  

**Abstract**: Fine-tuned Large Language Models (LLMs) often demonstrate poor calibration, with their confidence scores misaligned with actual performance. While calibration has been extensively studied in models trained from scratch, the impact of LLMs' prior knowledge on calibration during fine-tuning remains understudied. Our research reveals that LLMs' prior knowledge causes potential poor calibration due to the ubiquitous presence of known data in real-world fine-tuning, which appears harmful for calibration. Specifically, data aligned with LLMs' prior knowledge would induce overconfidence, while new knowledge improves calibration. Our findings expose a tension: LLMs' encyclopedic knowledge, while enabling task versatility, undermines calibration through unavoidable knowledge overlaps. To address this, we propose CogCalib, a cognition-aware framework that applies targeted learning strategies according to the model's prior knowledge. Experiments across 7 tasks using 3 LLM families prove that CogCalib significantly improves calibration while maintaining performance, achieving an average 57\% reduction in ECE compared to standard fine-tuning in Llama3-8B. These improvements generalize well to out-of-domain tasks, enhancing the objectivity and reliability of domain-specific LLMs, and making them more trustworthy for critical human-AI interaction applications. 

---
# A Stereotype Content Analysis on Color-related Social Bias in Large Vision Language Models 

**Authors**: Junhyuk Choi, Minju Kim, Yeseon Hong, Bugeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.20901)  

**Abstract**: As large vision language models(LVLMs) rapidly advance, concerns about their potential to learn and generate social biases and stereotypes are increasing. Previous studies on LVLM's stereotypes face two primary limitations: metrics that overlooked the importance of content words, and datasets that overlooked the effect of color. To address these limitations, this study introduces new evaluation metrics based on the Stereotype Content Model (SCM). We also propose BASIC, a benchmark for assessing gender, race, and color stereotypes. Using SCM metrics and BASIC, we conduct a study with eight LVLMs to discover stereotypes. As a result, we found three findings. (1) The SCM-based evaluation is effective in capturing stereotypes. (2) LVLMs exhibit color stereotypes in the output along with gender and race ones. (3) Interaction between model architecture and parameter sizes seems to affect stereotypes. We release BASIC publicly on [anonymized for review]. 

---
# Dub-S2ST: Textless Speech-to-Speech Translation for Seamless Dubbing 

**Authors**: Jeongsoo Choi, Jaehun Kim, Joon Son Chung  

**Link**: [PDF](https://arxiv.org/pdf/2505.20899)  

**Abstract**: This paper introduces a cross-lingual dubbing system that translates speech from one language to another while preserving key characteristics such as duration, speaker identity, and speaking speed. Despite the strong translation quality of existing speech translation approaches, they often overlook the transfer of speech patterns, leading to mismatches with source speech and limiting their suitability for dubbing applications. To address this, we propose a discrete diffusion-based speech-to-unit translation model with explicit duration control, enabling time-aligned translation. We then synthesize speech based on the predicted units and source identity with a conditional flow matching model. Additionally, we introduce a unit-based speed adaptation mechanism that guides the translation model to produce speech at a rate consistent with the source, without relying on any text. Extensive experiments demonstrate that our framework generates natural and fluent translations that align with the original speech's duration and speaking pace, while achieving competitive translation performance. 

---
# EasyDistill: A Comprehensive Toolkit for Effective Knowledge Distillation of Large Language Models 

**Authors**: Chengyu Wang, Junbing Yan, Wenrui Cai, Yuanhao Yue, Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20888)  

**Abstract**: In this paper, we present EasyDistill, a comprehensive toolkit designed for effective black-box and white-box knowledge distillation (KD) of large language models (LLMs). Our framework offers versatile functionalities, including data synthesis, supervised fine-tuning, ranking optimization, and reinforcement learning techniques specifically tailored for KD scenarios. The toolkit accommodates KD functionalities for both System 1 (fast, intuitive) and System 2 (slow, analytical) models. With its modular design and user-friendly interface, EasyDistill empowers researchers and industry practitioners to seamlessly experiment with and implement state-of-the-art KD strategies for LLMs. In addition, EasyDistill provides a series of robust distilled models and KD-based industrial solutions developed by us, along with the corresponding open-sourced datasets, catering to a variety of use cases. Furthermore, we describe the seamless integration of EasyDistill into Alibaba Cloud's Platform for AI (PAI). Overall, the EasyDistill toolkit makes advanced KD techniques for LLMs more accessible and impactful within the NLP community. 

---
# MSA at SemEval-2025 Task 3: High Quality Weak Labeling and LLM Ensemble Verification for Multilingual Hallucination Detection 

**Authors**: Baraa Hikal, Ahmed Nasreldin, Ali Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2505.20880)  

**Abstract**: This paper describes our submission for SemEval-2025 Task 3: Mu-SHROOM, the Multilingual Shared-task on Hallucinations and Related Observable Overgeneration Mistakes. The task involves detecting hallucinated spans in text generated by instruction-tuned Large Language Models (LLMs) across multiple languages. Our approach combines task-specific prompt engineering with an LLM ensemble verification mechanism, where a primary model extracts hallucination spans and three independent LLMs adjudicate their validity through probability-based voting. This framework simulates the human annotation workflow used in the shared task validation and test data. Additionally, fuzzy matching refines span alignment. Our system ranked 1st in Arabic and Basque, 2nd in German, Swedish, and Finnish, and 3rd in Czech, Farsi, and French. 

---
# Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties 

**Authors**: Jiyoung Lee, Seungho Kim, Jieun Han, Jun-Min Lee, Kitaek Kim, Alice Oh, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.20875)  

**Abstract**: Large Language Models (LLMs) are predominantly evaluated on Standard American English (SAE), often overlooking the diversity of global English varieties. This narrow focus may raise fairness concerns as degraded performance on non-standard varieties can lead to unequal benefits for users worldwide. Therefore, it is critical to extensively evaluate the linguistic robustness of LLMs on multiple non-standard English varieties. We introduce Trans-EnV, a framework that automatically transforms SAE datasets into multiple English varieties to evaluate the linguistic robustness. Our framework combines (1) linguistics expert knowledge to curate variety-specific features and transformation guidelines from linguistic literature and corpora, and (2) LLM-based transformations to ensure both linguistic validity and scalability. Using Trans-EnV, we transform six benchmark datasets into 38 English varieties and evaluate seven state-of-the-art LLMs. Our results reveal significant performance disparities, with accuracy decreasing by up to 46.3% on non-standard varieties. These findings highlight the importance of comprehensive linguistic robustness evaluation across diverse English varieties. Each construction of Trans-EnV was validated through rigorous statistical testing and consultation with a researcher in the field of second language acquisition, ensuring its linguistic validity. Our \href{this https URL}{code} and \href{this https URL}{datasets} are publicly available. 

---
# Can LLMs Learn to Map the World from Local Descriptions? 

**Authors**: Sirui Xia, Aili Chen, Xintao Wang, Tinghui Zhu, Yikai Zhang, Jiangjie Chen, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.20874)  

**Abstract**: Recent advances in Large Language Models (LLMs) have demonstrated strong capabilities in tasks such as code and mathematics. However, their potential to internalize structured spatial knowledge remains underexplored. This study investigates whether LLMs, grounded in locally relative human observations, can construct coherent global spatial cognition by integrating fragmented relational descriptions. We focus on two core aspects of spatial cognition: spatial perception, where models infer consistent global layouts from local positional relationships, and spatial navigation, where models learn road connectivity from trajectory data and plan optimal paths between unconnected locations. Experiments conducted in a simulated urban environment demonstrate that LLMs not only generalize to unseen spatial relationships between points of interest (POIs) but also exhibit latent representations aligned with real-world spatial distributions. Furthermore, LLMs can learn road connectivity from trajectory descriptions, enabling accurate path planning and dynamic spatial awareness during navigation. 

---
# Divide-Then-Align: Honest Alignment based on the Knowledge Boundary of RAG 

**Authors**: Xin Sun, Jianan Xie, Zhongqi Chen, Qiang Liu, Shu Wu, Yuehe Chen, Bowen Song, Weiqiang Wang, Zilei Wang, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20871)  

**Abstract**: Large language models (LLMs) augmented with retrieval systems have significantly advanced natural language processing tasks by integrating external knowledge sources, enabling more accurate and contextually rich responses. To improve the robustness of such systems against noisy retrievals, Retrieval-Augmented Fine-Tuning (RAFT) has emerged as a widely adopted method. However, RAFT conditions models to generate answers even in the absence of reliable knowledge. This behavior undermines their reliability in high-stakes domains, where acknowledging uncertainty is critical. To address this issue, we propose Divide-Then-Align (DTA), a post-training approach designed to endow RAG systems with the ability to respond with "I don't know" when the query is out of the knowledge boundary of both the retrieved passages and the model's internal knowledge. DTA divides data samples into four knowledge quadrants and constructs tailored preference data for each quadrant, resulting in a curated dataset for Direct Preference Optimization (DPO). Experimental results on three benchmark datasets demonstrate that DTA effectively balances accuracy with appropriate abstention, enhancing the reliability and trustworthiness of retrieval-augmented systems. 

---
# Concealment of Intent: A Game-Theoretic Analysis 

**Authors**: Xinbo Wu, Abhishek Umrawal, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2505.20841)  

**Abstract**: As large language models (LLMs) grow more capable, concerns about their safe deployment have also grown. Although alignment mechanisms have been introduced to deter misuse, they remain vulnerable to carefully designed adversarial prompts. In this work, we present a scalable attack strategy: intent-hiding adversarial prompting, which conceals malicious intent through the composition of skills. We develop a game-theoretic framework to model the interaction between such attacks and defense systems that apply both prompt and response filtering. Our analysis identifies equilibrium points and reveals structural advantages for the attacker. To counter these threats, we propose and analyze a defense mechanism tailored to intent-hiding attacks. Empirically, we validate the attack's effectiveness on multiple real-world LLMs across a range of malicious behaviors, demonstrating clear advantages over existing adversarial prompting techniques. 

---
# AdParaphrase v2.0: Generating Attractive Ad Texts Using a Preference-Annotated Paraphrase Dataset 

**Authors**: Soichiro Murakami, Peinan Zhang, Hidetaka Kamigaito, Hiroya Takamura, Manabu Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2505.20826)  

**Abstract**: Identifying factors that make ad text attractive is essential for advertising success. This study proposes AdParaphrase v2.0, a dataset for ad text paraphrasing, containing human preference data, to enable the analysis of the linguistic factors and to support the development of methods for generating attractive ad texts. Compared with v1.0, this dataset is 20 times larger, comprising 16,460 ad text paraphrase pairs, each annotated with preference data from ten evaluators, thereby enabling a more comprehensive and reliable analysis. Through the experiments, we identified multiple linguistic features of engaging ad texts that were not observed in v1.0 and explored various methods for generating attractive ad texts. Furthermore, our analysis demonstrated the relationships between human preference and ad performance, and highlighted the potential of reference-free metrics based on large language models for evaluating ad text attractiveness. The dataset is publicly available at: this https URL. 

---
# Reinforced Informativeness Optimization for Long-Form Retrieval-Augmented Generation 

**Authors**: Yuhao Wang, Ruiyang Ren, Yucheng Wang, Wayne Xin Zhao, Jing Liu, Hua Wu, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20825)  

**Abstract**: Long-form question answering (LFQA) presents unique challenges for large language models, requiring the synthesis of coherent, paragraph-length answers. While retrieval-augmented generation (RAG) systems have emerged as a promising solution, existing research struggles with key limitations: the scarcity of high-quality training data for long-form generation, the compounding risk of hallucination in extended outputs, and the absence of reliable evaluation metrics for factual completeness. In this paper, we propose RioRAG, a novel reinforcement learning (RL) framework that advances long-form RAG through reinforced informativeness optimization. Our approach introduces two fundamental innovations to address the core challenges. First, we develop an RL training paradigm of reinforced informativeness optimization that directly optimizes informativeness and effectively addresses the slow-thinking deficit in conventional RAG systems, bypassing the need for expensive supervised data. Second, we propose a nugget-centric hierarchical reward modeling approach that enables precise assessment of long-form answers through a three-stage process: extracting the nugget from every source webpage, constructing a nugget claim checklist, and computing rewards based on factual alignment. Extensive experiments on two LFQA benchmarks LongFact and RAGChecker demonstrate the effectiveness of the proposed method. Our codes are available at this https URL. 

---
# Tracing and Reversing Rank-One Model Edits 

**Authors**: Paul Youssef, Zhixue Zhao, Christin Seifert, Jörg Schlötterer  

**Link**: [PDF](https://arxiv.org/pdf/2505.20819)  

**Abstract**: Knowledge editing methods (KEs) are a cost-effective way to update the factual content of large language models (LLMs), but they pose a dual-use risk. While KEs are beneficial for updating outdated or incorrect information, they can be exploited maliciously to implant misinformation or bias. In order to defend against these types of malicious manipulation, we need robust techniques that can reliably detect, interpret, and mitigate adversarial edits. This work investigates the traceability and reversibility of knowledge edits, focusing on the widely used Rank-One Model Editing (ROME) method. We first show that ROME introduces distinctive distributional patterns in the edited weight matrices, which can serve as effective signals for locating the edited weights. Second, we show that these altered weights can reliably be used to predict the edited factual relation, enabling partial reconstruction of the modified fact. Building on this, we propose a method to infer the edited object entity directly from the modified weights, without access to the editing prompt, achieving over 95% accuracy. Finally, we demonstrate that ROME edits can be reversed, recovering the model's original outputs with $\geq$ 80% accuracy. Our findings highlight the feasibility of detecting, tracing, and reversing edits based on the edited weights, offering a robust framework for safeguarding LLMs against adversarial manipulations. 

---
# Rethinking Information Synthesis in Multimodal Question Answering A Multi-Agent Perspective 

**Authors**: Krishna Singh Rajput, Tejas Anvekar, Chitta Baral, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.20816)  

**Abstract**: Recent advances in multimodal question answering have primarily focused on combining heterogeneous modalities or fine-tuning multimodal large language models. While these approaches have shown strong performance, they often rely on a single, generalized reasoning strategy, overlooking the unique characteristics of each modality ultimately limiting both accuracy and interpretability. To address these limitations, we propose MAMMQA, a multi-agent QA framework for multimodal inputs spanning text, tables, and images. Our system includes two Visual Language Model (VLM) agents and one text-based Large Language Model (LLM) agent. The first VLM decomposes the user query into sub-questions and sequentially retrieves partial answers from each modality. The second VLM synthesizes and refines these results through cross-modal reasoning. Finally, the LLM integrates the insights into a cohesive answer. This modular design enhances interpretability by making the reasoning process transparent and allows each agent to operate within its domain of expertise. Experiments on diverse multimodal QA benchmarks demonstrate that our cooperative, multi-agent framework consistently outperforms existing baselines in both accuracy and robustness. 

---
# RSCF: Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph 

**Authors**: Junsik Kim, Jinwook Park, Kangil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.20813)  

**Abstract**: In knowledge graph embedding, leveraging relation-specific entity-transformation has markedly enhanced performance. However, the consistency of embedding differences before and after transformation remains unaddressed, risking the loss of valuable inductive bias inherent in the embeddings. This inconsistency stems from two problems. First, transformation representations are specified for relations in a disconnected manner, allowing dissimilar transformations and corresponding entity-embeddings for similar relations. Second, a generalized plug-in approach as a SFBR (Semantic Filter Based on Relations) disrupts this consistency through excessive concentration of entity embeddings under entity-based regularization, generating indistinguishable score distributions among relations. In this paper, we introduce a plug-in KGE method, Relation-Semantics Consistent Filter (RSCF), containing more consistent entity-transformation characterized by three features: 1) shared affine transformation of relation embeddings across all relations, 2) rooted entity-transformation that adds an entity embedding to its change represented by the transformed vector, and 3) normalization of the change to prevent scale reduction. To amplify the advantages of consistency that preserve semantics on embeddings, RSCF adds relation transformation and prediction modules for enhancing the semantics. In knowledge graph completion tasks with distance-based and tensor decomposition models, RSCF significantly outperforms state-of-the-art KGE methods, showing robustness across all relations and their frequencies. 

---
# Improved Representation Steering for Language Models 

**Authors**: Zhengxuan Wu, Qinan Yu, Aryaman Arora, Christopher D. Manning, Christopher Potts  

**Link**: [PDF](https://arxiv.org/pdf/2505.20809)  

**Abstract**: Steering methods for language models (LMs) seek to provide fine-grained and interpretable control over model generations by variously changing model inputs, weights, or representations to adjust behavior. Recent work has shown that adjusting weights or representations is often less effective than steering by prompting, for instance when wanting to introduce or suppress a particular concept. We demonstrate how to improve representation steering via our new Reference-free Preference Steering (RePS), a bidirectional preference-optimization objective that jointly does concept steering and suppression. We train three parameterizations of RePS and evaluate them on AxBench, a large-scale model steering benchmark. On Gemma models with sizes ranging from 2B to 27B, RePS outperforms all existing steering methods trained with a language modeling objective and substantially narrows the gap with prompting -- while promoting interpretability and minimizing parameter count. In suppression, RePS matches the language-modeling objective on Gemma-2 and outperforms it on the larger Gemma-3 variants while remaining resilient to prompt-based jailbreaking attacks that defeat prompting. Overall, our results suggest that RePS provides an interpretable and robust alternative to prompting for both steering and suppression. 

---
# CHIMERA: A Knowledge Base of Idea Recombination in Scientific Literature 

**Authors**: Noy Sternlicht, Tom Hope  

**Link**: [PDF](https://arxiv.org/pdf/2505.20779)  

**Abstract**: A hallmark of human innovation is the process of recombination -- creating original ideas by integrating elements of existing mechanisms and concepts. In this work, we automatically mine the scientific literature and build CHIMERA: a large-scale knowledge base (KB) of recombination examples. CHIMERA can be used to empirically explore at scale how scientists recombine concepts and take inspiration from different areas, or to train supervised machine learning models that learn to predict new creative cross-domain directions. To build this KB, we present a novel information extraction task of extracting recombination from scientific paper abstracts, collect a high-quality corpus of hundreds of manually annotated abstracts, and use it to train an LLM-based extraction model. The model is applied to a large corpus of papers in the AI domain, yielding a KB of over 28K recombination examples. We analyze CHIMERA to explore the properties of recombination in different subareas of AI. Finally, we train a scientific hypothesis generation model using the KB, which predicts new recombination directions that real-world researchers find inspiring. Our data and code are available at this https URL 

---
# SpecExtend: A Drop-in Enhancement for Speculative Decoding of Long Sequences 

**Authors**: Jungyoub Cha, Hyunjong Kim, Sungzoon Cho  

**Link**: [PDF](https://arxiv.org/pdf/2505.20776)  

**Abstract**: Speculative decoding is a widely adopted technique for accelerating inference in large language models (LLMs), but its performance degrades on long inputs due to increased attention cost and reduced draft accuracy. We introduce SpecExtend, a drop-in enhancement that improves the performance of speculative decoding on long sequences without any additional training. SpecExtend integrates efficient attention mechanisms such as FlashAttention and Hybrid Tree Attention into both the draft and target models, reducing latency across all stages. To improve draft accuracy and speed, we propose Cross-model Retrieval, a novel KV cache update strategy that uses the target model's attention scores to dynamically select relevant context for the draft model. Extensive evaluations on three long-context understanding datasets show that SpecExtend accelerates standard tree-based speculative decoding by up to 2.22x for inputs up to 16K tokens, providing an effective solution for speculative decoding of long sequences. The code is available at this https URL . 

---
# CogniBench: A Legal-inspired Framework and Dataset for Assessing Cognitive Faithfulness of Large Language Models 

**Authors**: Xiaqiang Tang, Jian Li, Keyu Hu, Du Nan, Xiaolong Li, Xi Zhang, Weigao Sun, Sihong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.20767)  

**Abstract**: Faithfulness hallucination are claims generated by a Large Language Model (LLM) not supported by contexts provided to the LLM. Lacking assessment standard, existing benchmarks only contain "factual statements" that rephrase source materials without marking "cognitive statements" that make inference from the given context, making the consistency evaluation and optimization of cognitive statements difficult. Inspired by how an evidence is assessed in the legislative domain, we design a rigorous framework to assess different levels of faithfulness of cognitive statements and create a benchmark dataset where we reveal insightful statistics. We design an annotation pipeline to create larger benchmarks for different LLMs automatically, and the resulting larger-scale CogniBench-L dataset can be used to train accurate cognitive hallucination detection model. We release our model and dataset at: this https URL 

---
# Silencer: From Discovery to Mitigation of Self-Bias in LLM-as-Benchmark-Generator 

**Authors**: Peiwen Yuan, Yiwei Li, Shaoxiong Feng, Xinglin Wang, Yueqi Zhang, Jiayi Shi, Chuyi Tan, Boyuan Pan, Yao Hu, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20738)  

**Abstract**: LLM-as-Benchmark-Generator methods have been widely studied as a supplement to human annotators for scalable evaluation, while the potential biases within this paradigm remain underexplored. In this work, we systematically define and validate the phenomenon of inflated performance in models evaluated on their self-generated benchmarks, referred to as self-bias, and attribute it to sub-biases arising from question domain, language style, and wrong labels. On this basis, we propose Silencer, a general framework that leverages the heterogeneity between multiple generators at both the sample and benchmark levels to neutralize bias and generate high-quality, self-bias-silenced benchmark. Experimental results across various settings demonstrate that Silencer can suppress self-bias to near zero, significantly improve evaluation effectiveness of the generated benchmark (with an average improvement from 0.655 to 0.833 in Pearson correlation with high-quality human-annotated benchmark), while also exhibiting strong generalizability. 

---
# SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution 

**Authors**: Hanlin Wang, Chak Tou Leong, Jiashuo Wang, Jian Wang, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20732)  

**Abstract**: Reinforcement learning (RL) holds significant promise for training LLM agents to handle complex, goal-oriented tasks that require multi-step interactions with external environments. However, a critical challenge when applying RL to these agentic tasks arises from delayed rewards: feedback signals are typically available only after the entire task is completed. This makes it non-trivial to assign delayed rewards to earlier actions, providing insufficient guidance regarding environmental constraints and hindering agent training. In this work, we draw on the insight that the ultimate completion of a task emerges from the cumulative progress an agent makes across individual steps. We propose Stepwise Progress Attribution (SPA), a general reward redistribution framework that decomposes the final reward into stepwise contributions, each reflecting its incremental progress toward overall task completion. To achieve this, we train a progress estimator that accumulates stepwise contributions over a trajectory to match the task completion. During policy optimization, we combine the estimated per-step contribution with a grounding signal for actions executed in the environment as the fine-grained, intermediate reward for effective agent training. Extensive experiments on common agent benchmarks (including Webshop, ALFWorld, and VirtualHome) demonstrate that SPA consistently outperforms the state-of-the-art method in both success rate (+2.5\% on average) and grounding accuracy (+1.9\% on average). Further analyses demonstrate that our method remarkably provides more effective intermediate rewards for RL training. Our code is available at this https URL. 

---
# Dissecting Physics Reasoning in Small Language Models: A Multi-Dimensional Analysis from an Educational Perspective 

**Authors**: Nicy Scaria, Silvester John Joseph Kennedy, Diksha Seth, Deepak Subramani  

**Link**: [PDF](https://arxiv.org/pdf/2505.20707)  

**Abstract**: Small Language Models (SLMs) offer computational efficiency and accessibility, making them promising for educational applications. However, their capacity for complex reasoning, particularly in domains such as physics, remains underexplored. This study investigates the high school physics reasoning capabilities of state-of-the-art SLMs (under 4 billion parameters), including instruct versions of Llama 3.2, Phi 4 Mini, Gemma 3, and Qwen series. We developed a comprehensive physics dataset from the OpenStax High School Physics textbook, annotated according to Bloom's Taxonomy, with LaTeX and plaintext mathematical notations. A novel cultural contextualization approach was applied to a subset, creating culturally adapted problems for Asian, African, and South American/Australian contexts while preserving core physics principles. Using an LLM-as-a-judge framework with Google's Gemini 2.5 Flash, we evaluated answer and reasoning chain correctness, along with calculation accuracy. The results reveal significant differences between the SLMs. Qwen 3 1.7B achieved high `answer accuracy' (85%), but `fully correct reasoning' was substantially low (38%). The format of the mathematical notation had a negligible impact on performance. SLMs exhibited varied performance across the physics topics and showed a decline in reasoning quality with increasing cognitive and knowledge complexity. In particular, the consistency of reasoning was largely maintained in diverse cultural contexts, especially by better performing models. These findings indicate that, while SLMs can often find correct answers, their underlying reasoning is frequently flawed, suggesting an overreliance on pattern recognition. For SLMs to become reliable educational tools in physics, future development must prioritize enhancing genuine understanding and the generation of sound, verifiable reasoning chains over mere answer accuracy. 

---
# Beyond Templates: Dynamic Adaptation of Reasoning Demonstrations via Feasibility-Aware Exploration 

**Authors**: Yong Wu, Weihang Pan, Ke Li, Chen Binhui, Ping Li, Binbin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.20700)  

**Abstract**: Large language models (LLMs) have shown remarkable reasoning capabilities, yet aligning such abilities to small language models (SLMs) remains a challenge due to distributional mismatches and limited model capacity. Existing reasoning datasets, typically designed for powerful LLMs, often lead to degraded performance when directly applied to weaker models. In this work, we introduce Dynamic Adaptation of Reasoning Trajectories (DART), a novel data adaptation framework that bridges the capability gap between expert reasoning trajectories and diverse SLMs. Instead of uniformly imitating expert steps, DART employs a selective imitation strategy guided by step-wise adaptability estimation via solution simulation. When expert steps surpass the student's capacity -- signaled by an Imitation Gap -- the student autonomously explores alternative reasoning paths, constrained by outcome consistency. We validate DART across multiple reasoning benchmarks and model scales, demonstrating that it significantly improves generalization and data efficiency over static fine-tuning. Our method enhances supervision quality by aligning training signals with the student's reasoning capabilities, offering a scalable solution for reasoning alignment in resource-constrained models. 

---
# Phir Hera Fairy: An English Fairytaler is a Strong Faker of Fluent Speech in Low-Resource Indian Languages 

**Authors**: Praveen Srinivasa Varadhan, Srija Anand, Soma Siddhartha, Mitesh M.Khapra  

**Link**: [PDF](https://arxiv.org/pdf/2505.20693)  

**Abstract**: What happens when an English Fairytaler is fine-tuned on Indian languages? We evaluate how the English F5-TTS model adapts to 11 Indian languages, measuring polyglot fluency, voice-cloning, style-cloning, and code-mixing. We compare: (i) training from scratch, (ii) fine-tuning English F5 on Indian data, and (iii) fine-tuning on both Indian and English data to prevent forgetting. Fine-tuning with only Indian data proves most effective and the resultant IN-F5 is a near-human polyglot; that enables speakers of one language (e.g., Odia) to fluently speak in another (e.g., Hindi). Our results show English pretraining aids low-resource TTS in reaching human parity. To aid progress in other low-resource languages, we study data-constrained setups and arrive at a compute optimal strategy. Finally, we show IN-F5 can synthesize unseen languages like Bhojpuri and Tulu using a human-in-the-loop approach for zero-resource TTS via synthetic data generation. 

---
# SELF-PERCEPT: Introspection Improves Large Language Models' Detection of Multi-Person Mental Manipulation in Conversations 

**Authors**: Danush Khanna, Pratinav Seth, Sidhaarth Sredharan Murali, Aditya Kumar Guru, Siddharth Shukla, Tanuj Tyagi, Sandeep Chaurasia, Kripabandhu Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2505.20679)  

**Abstract**: Mental manipulation is a subtle yet pervasive form of abuse in interpersonal communication, making its detection critical for safeguarding potential victims. However, due to manipulation's nuanced and context-specific nature, identifying manipulative language in complex, multi-turn, and multi-person conversations remains a significant challenge for large language models (LLMs). To address this gap, we introduce the MultiManip dataset, comprising 220 multi-turn, multi-person dialogues balanced between manipulative and non-manipulative interactions, all drawn from reality shows that mimic real-world scenarios. For manipulative interactions, it includes 11 distinct manipulations depicting real-life scenarios. We conduct extensive evaluations of state-of-the-art LLMs, such as GPT-4o and Llama-3.1-8B, employing various prompting strategies. Despite their capabilities, these models often struggle to detect manipulation effectively. To overcome this limitation, we propose SELF-PERCEPT, a novel, two-stage prompting framework inspired by Self-Perception Theory, demonstrating strong performance in detecting multi-person, multi-turn mental manipulation. Our code and data are publicly available at this https URL . 

---
# Pretraining Language Models to Ponder in Continuous Space 

**Authors**: Boyi Zeng, Shixiang Song, Siyuan Huang, Yixuan Wang, He Li, Ziwei He, Xinbing Wang, Zhiyu Li, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.20674)  

**Abstract**: Humans ponder before articulating complex sentence elements, enabling deeper cognitive processing through focused effort. In this work, we introduce this pondering process into language models by repeatedly invoking the forward process within a single token generation step. During pondering, instead of generating an actual token sampled from the prediction distribution, the model ponders by yielding a weighted sum of all token embeddings according to the predicted token distribution. The generated embedding is then fed back as input for another forward pass. We show that the model can learn to ponder in this way through self-supervised learning, without any human annotations. Our method is straightforward and can be seamlessly integrated with various existing language models. Experiments across three widely used open-source architectures-GPT-2, Pythia, and LLaMA-and extensive downstream task evaluations demonstrate the effectiveness and generality of our method. For language modeling tasks, pondering language models achieve performance comparable to vanilla models with twice the number of parameters. On 9 downstream benchmarks, our pondering-enhanced Pythia models significantly outperform the official Pythia models. Notably, pondering-enhanced Pythia-1B is comparable to TinyLlama-1.1B, which is trained on 10 times more data. The code is available at this https URL. 

---
# Self-Route: Automatic Mode Switching via Capability Estimation for Efficient Reasoning 

**Authors**: Yang He, Xiao Ding, Bibo Cai, Yufei Zhang, Kai Xiong, Zhouhao Sun, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20664)  

**Abstract**: While reasoning-augmented large language models (RLLMs) significantly enhance complex task performance through extended reasoning chains, they inevitably introduce substantial unnecessary token consumption, particularly for simpler problems where Short Chain-of-Thought (Short CoT) suffices. This overthinking phenomenon leads to inefficient resource usage without proportional accuracy gains. To address this issue, we propose Self-Route, a dynamic reasoning framework that automatically selects between general and reasoning modes based on model capability estimation. Our approach introduces a lightweight pre-inference stage to extract capability-aware embeddings from hidden layer representations, enabling real-time evaluation of the model's ability to solve problems. We further construct Gradient-10K, a model difficulty estimation-based dataset with dense complexity sampling, to train the router for precise capability boundary detection. Extensive experiments demonstrate that Self-Route achieves comparable accuracy to reasoning models while reducing token consumption by 30-55\% across diverse benchmarks. The proposed framework demonstrates consistent effectiveness across models with different parameter scales and reasoning paradigms, highlighting its general applicability and practical value. 

---
# BacktrackAgent: Enhancing GUI Agent with Error Detection and Backtracking Mechanism 

**Authors**: Qinzhuo Wu, Pengzhi Gao, Wei Liu, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2505.20660)  

**Abstract**: Graphical User Interface (GUI) agents have gained substantial attention due to their impressive capabilities to complete tasks through multiple interactions within GUI environments. However, existing agents primarily focus on enhancing the accuracy of individual actions and often lack effective mechanisms for detecting and recovering from errors. To address these shortcomings, we propose the BacktrackAgent, a robust framework that incorporates a backtracking mechanism to improve task completion efficiency. BacktrackAgent includes verifier, judger, and reflector components as modules for error detection and recovery, while also applying judgment rewards to further enhance the agent's performance. Additionally, we develop a training dataset specifically designed for the backtracking mechanism, which considers the outcome pages after action executions. Experimental results show that BacktrackAgent has achieved performance improvements in both task success rate and step accuracy on Mobile3M and Auto-UI benchmarks. Our data and code will be released upon acceptance. 

---
# Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge 

**Authors**: Yue Fang, Zhi Jin, Jie An, Hongshen Chen, Xiaohong Chen, Naijun Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2505.20658)  

**Abstract**: Temporal Logic (TL), especially Signal Temporal Logic (STL), enables precise formal specification, making it widely used in cyber-physical systems such as autonomous driving and robotics. Automatically transforming NL into STL is an attractive approach to overcome the limitations of manual transformation, which is time-consuming and error-prone. However, due to the lack of datasets, automatic transformation currently faces significant challenges and has not been fully explored. In this paper, we propose an NL-STL dataset named STL-Diversity-Enhanced (STL-DivEn), which comprises 16,000 samples enriched with diverse patterns. To develop the dataset, we first manually create a small-scale seed set of NL-STL pairs. Next, representative examples are identified through clustering and used to guide large language models (LLMs) in generating additional NL-STL pairs. Finally, diversity and accuracy are ensured through rigorous rule-based filters and human validation. Furthermore, we introduce the Knowledge-Guided STL Transformation (KGST) framework, a novel approach for transforming natural language into STL, involving a generate-then-refine process based on external knowledge. Statistical analysis shows that the STL-DivEn dataset exhibits more diversity than the existing NL-STL dataset. Moreover, both metric-based and human evaluations indicate that our KGST approach outperforms baseline models in transformation accuracy on STL-DivEn and DeepSTL datasets. 

---
# Chinese Cyberbullying Detection: Dataset, Method, and Validation 

**Authors**: Yi Zhu, Xin Zou, Xindong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20654)  

**Abstract**: Existing cyberbullying detection benchmarks were organized by the polarity of speech, such as "offensive" and "non-offensive", which were essentially hate speech detection. However, in the real world, cyberbullying often attracted widespread social attention through incidents. To address this problem, we propose a novel annotation method to construct a cyberbullying dataset that organized by incidents. The constructed CHNCI is the first Chinese cyberbullying incident detection dataset, which consists of 220,676 comments in 91 incidents. Specifically, we first combine three cyberbullying detection methods based on explanations generation as an ensemble method to generate the pseudo labels, and then let human annotators judge these labels. Then we propose the evaluation criteria for validating whether it constitutes a cyberbullying incident. Experimental results demonstrate that the constructed dataset can be a benchmark for the tasks of cyberbullying detection and incident prediction. To the best of our knowledge, this is the first study for the Chinese cyberbullying incident detection task. 

---
# FinTagging: An LLM-ready Benchmark for Extracting and Structuring Financial Information 

**Authors**: Yan Wang, Yang Ren, Lingfei Qian, Xueqing Peng, Keyi Wang, Yi Han, Dongji Feng, Xiao-Yang Liu, Jimin Huang, Qianqian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.20650)  

**Abstract**: We introduce FinTagging, the first full-scope, table-aware XBRL benchmark designed to evaluate the structured information extraction and semantic alignment capabilities of large language models (LLMs) in the context of XBRL-based financial reporting. Unlike prior benchmarks that oversimplify XBRL tagging as flat multi-class classification and focus solely on narrative text, FinTagging decomposes the XBRL tagging problem into two subtasks: FinNI for financial entity extraction and FinCL for taxonomy-driven concept alignment. It requires models to jointly extract facts and align them with the full 10k+ US-GAAP taxonomy across both unstructured text and structured tables, enabling realistic, fine-grained evaluation. We assess a diverse set of LLMs under zero-shot settings, systematically analyzing their performance on both subtasks and overall tagging accuracy. Our results reveal that, while LLMs demonstrate strong generalization in information extraction, they struggle with fine-grained concept alignment, particularly in disambiguating closely related taxonomy entries. These findings highlight the limitations of existing LLMs in fully automating XBRL tagging and underscore the need for improved semantic reasoning and schema-aware modeling to meet the demands of accurate financial disclosure. Code is available at our GitHub repository and data is at our Hugging Face repository. 

---
# STEER-BENCH: A Benchmark for Evaluating the Steerability of Large Language Models 

**Authors**: Kai Chen, Zihao He, Taiwei Shi, Kristina Lerman  

**Link**: [PDF](https://arxiv.org/pdf/2505.20645)  

**Abstract**: Steerability, or the ability of large language models (LLMs) to adapt outputs to align with diverse community-specific norms, perspectives, and communication styles, is critical for real-world applications but remains under-evaluated. We introduce Steer-Bench, a benchmark for assessing population-specific steering using contrasting Reddit communities. Covering 30 contrasting subreddit pairs across 19 domains, Steer-Bench includes over 10,000 instruction-response pairs and validated 5,500 multiple-choice question with corresponding silver labels to test alignment with diverse community norms. Our evaluation of 13 popular LLMs using Steer-Bench reveals that while human experts achieve an accuracy of 81% with silver labels, the best-performing models reach only around 65% accuracy depending on the domain and configuration. Some models lag behind human-level alignment by over 15 percentage points, highlighting significant gaps in community-sensitive steerability. Steer-Bench is a benchmark to systematically assess how effectively LLMs understand community-specific instructions, their resilience to adversarial steering attempts, and their ability to accurately represent diverse cultural and ideological perspectives. 

---
# Test-Time Learning for Large Language Models 

**Authors**: Jinwu Hu, Zhitian Zhang, Guohao Chen, Xutao Wen, Chao Shuai, Wei Luo, Bin Xiao, Yuanqing Li, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.20633)  

**Abstract**: While Large Language Models (LLMs) have exhibited remarkable emergent capabilities through extensive pre-training, they still face critical limitations in generalizing to specialized domains and handling diverse linguistic variations, known as distribution shifts. In this paper, we propose a Test-Time Learning (TTL) paradigm for LLMs, namely TLM, which dynamically adapts LLMs to target domains using only unlabeled test data during testing. Specifically, we first provide empirical evidence and theoretical insights to reveal that more accurate predictions from LLMs can be achieved by minimizing the input perplexity of the unlabeled test data. Based on this insight, we formulate the Test-Time Learning process of LLMs as input perplexity minimization, enabling self-supervised enhancement of LLM performance. Furthermore, we observe that high-perplexity samples tend to be more informative for model optimization. Accordingly, we introduce a Sample Efficient Learning Strategy that actively selects and emphasizes these high-perplexity samples for test-time updates. Lastly, to mitigate catastrophic forgetting and ensure adaptation stability, we adopt Low-Rank Adaptation (LoRA) instead of full-parameter optimization, which allows lightweight model updates while preserving more original knowledge from the model. We introduce the AdaptEval benchmark for TTL and demonstrate through experiments that TLM improves performance by at least 20% compared to original LLMs on domain knowledge adaptation. 

---
# Long Context Scaling: Divide and Conquer via Multi-Agent Question-driven Collaboration 

**Authors**: Sibo Xiao, Zixin Lin, Wenyang Gao, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20625)  

**Abstract**: Processing long contexts has become a critical capability for modern large language models (LLMs). Existing works leverage agent-based divide-and-conquer methods for processing long contexts. But these methods face crucial limitations, including prohibitive accumulated latency and amplified information loss from excessive agent invocations, and the disruption of inherent textual dependencies by immoderate partitioning. In this paper, we propose a novel multi-agent framework XpandA (Expand-Agent) coupled with question-driven workflow and dynamic partitioning for robust long-context processing. XpandA overcomes these limitations through: 1) dynamic partitioning of long texts, which adaptively modulates the filling rate of context windows for input sequences of vastly varying lengths; 2) question-guided protocol to update flat information ensembles within centralized shared memory, constructing consistent inter-agent knowledge across partitions; and 3) selectively replaying specific partitions based on the state-tracking of question-information couples to promote the resolution of inverted-order structures across partitions (e.g., flashbacks). We perform a comprehensive evaluation of XpandA on multiple long-context benchmarks with length varying from 1k to 1M, demonstrating XpandA's feasibility for processing ultra-long sequences and its significant effectiveness in enhancing the long-context capabilities of various LLMs by achieving 20\% improvements and 1.5x inference speedup over baselines of full-context, RAG and previous agent-based methods. 

---
# POLAR: A Benchmark for Multilingual, Multicultural, and Multi-Event Online Polarization 

**Authors**: Usman Naseem, Juan Ren, Saba Anwar, Sarah Kohail, Rudy Alexandro Garrido Veliz, Robert Geislinger, Aisha Jabr, Idris Abdulmumin, Laiba Qureshi, Aarushi Ajay Borkar, Maryam Ibrahim Mukhtar, Abinew Ali Ayele, Ibrahim Said Ahmad, Adem Ali, Martin Semmann, Shamsuddeen Hassan Muhammad, Seid Muhie Yimam  

**Link**: [PDF](https://arxiv.org/pdf/2505.20624)  

**Abstract**: Online polarization poses a growing challenge for democratic discourse, yet most computational social science research remains monolingual, culturally narrow, or event-specific. We introduce POLAR, a multilingual, multicultural, and multievent dataset with over 23k instances in seven languages from diverse online platforms and real-world events. Polarization is annotated along three axes: presence, type, and manifestation, using a variety of annotation platforms adapted to each cultural context. We conduct two main experiments: (1) we fine-tune six multilingual pretrained language models in both monolingual and cross-lingual setups; and (2) we evaluate a range of open and closed large language models (LLMs) in few-shot and zero-shot scenarios. Results show that while most models perform well on binary polarization detection, they achieve substantially lower scores when predicting polarization types and manifestations. These findings highlight the complex, highly contextual nature of polarization and the need for robust, adaptable approaches in NLP and computational social science. All resources will be released to support further research and effective mitigation of digital polarization globally. 

---
# SeqPO-SiMT: Sequential Policy Optimization for Simultaneous Machine Translation 

**Authors**: Ting Xu, Zhichao Huang, Jiankai Sun, Shanbo Cheng, Wai Lam  

**Link**: [PDF](https://arxiv.org/pdf/2505.20622)  

**Abstract**: We present Sequential Policy Optimization for Simultaneous Machine Translation (SeqPO-SiMT), a new policy optimization framework that defines the simultaneous machine translation (SiMT) task as a sequential decision making problem, incorporating a tailored reward to enhance translation quality while reducing latency. In contrast to popular Reinforcement Learning from Human Feedback (RLHF) methods, such as PPO and DPO, which are typically applied in single-step tasks, SeqPO-SiMT effectively tackles the multi-step SiMT task. This intuitive framework allows the SiMT LLMs to simulate and refine the SiMT process using a tailored reward. We conduct experiments on six datasets from diverse domains for En to Zh and Zh to En SiMT tasks, demonstrating that SeqPO-SiMT consistently achieves significantly higher translation quality with lower latency. In particular, SeqPO-SiMT outperforms the supervised fine-tuning (SFT) model by 1.13 points in COMET, while reducing the Average Lagging by 6.17 in the NEWSTEST2021 En to Zh dataset. While SiMT operates with far less context than offline translation, the SiMT results of SeqPO-SiMT on 7B LLM surprisingly rival the offline translation of high-performing LLMs, including Qwen-2.5-7B-Instruct and LLaMA-3-8B-Instruct. 

---
# REAL-Prover: Retrieval Augmented Lean Prover for Mathematical Reasoning 

**Authors**: Ziju Shen, Naohao Huang, Fanyi Yang, Yutong Wang, Guoxiong Gao, Tianyi Xu, Jiedong Jiang, Wanyi He, Pu Yang, Mengzhou Sun, Haocheng Ju, Peihao Wu, Bryan Dai, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.20613)  

**Abstract**: Nowadays, formal theorem provers have made monumental progress on high-school and competition-level mathematics, but few of them generalize to more advanced mathematics. In this paper, we present REAL-Prover, a new open-source stepwise theorem prover for Lean 4 to push this boundary. This prover, based on our fine-tuned large language model (REAL-Prover-v1) and integrated with a retrieval system (Leansearch-PS), notably boosts performance on solving college-level mathematics problems. To train REAL-Prover-v1, we developed HERALD-AF, a data extraction pipeline that converts natural language math problems into formal statements, and a new open-source Lean 4 interactive environment (Jixia-interactive) to facilitate synthesis data collection. In our experiments, our prover using only supervised fine-tune achieves competitive results with a 23.7% success rate (Pass@64) on the ProofNet dataset-comparable to state-of-the-art (SOTA) models. To further evaluate our approach, we introduce FATE-M, a new benchmark focused on algebraic problems, where our prover achieves a SOTA success rate of 56.7% (Pass@64). 

---
# Towards Pretraining Robust ASR Foundation Model with Acoustic-Aware Data Augmentation 

**Authors**: Dancheng Liu, Amir Nassereldine, Chenhui Xu, Jinjun Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.20606)  

**Abstract**: Whisper's robust performance in automatic speech recognition (ASR) is often attributed to its massive 680k-hour training set, an impractical scale for most researchers. In this work, we examine how linguistic and acoustic diversity in training data affect the robustness of the ASR model and reveal that transcription generalization is primarily driven by acoustic variation rather than linguistic richness. We find that targeted acoustic augmentation methods could significantly improve the generalization ability of ASR models, reducing word-error rates by up to 19.24 percent on unseen datasets when training on the 960-hour Librispeech dataset. These findings highlight strategic acoustically focused data augmentation as a promising alternative to massive datasets for building robust ASR models, offering a potential solution to future foundation ASR models when massive human speech data is lacking. 

---
# Effectiveness of Prompt Optimization in NL2SQL Systems 

**Authors**: Sairam Gurajada, Eser Kandogan, Sajjadur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2505.20591)  

**Abstract**: NL2SQL approaches have greatly benefited from the impressive capabilities of large language models (LLMs). In particular, bootstrapping an NL2SQL system for a specific domain can be as simple as instructing an LLM with sufficient contextual information, such as schema details and translation demonstrations. However, building an accurate system still requires the rigorous task of selecting the right context for each query-including identifying relevant schema elements, cell values, and suitable exemplars that help the LLM understand domain-specific nuances. Retrieval-based methods have become the go-to approach for identifying such context. While effective, these methods introduce additional inference-time costs due to the retrieval process.
In this paper, we argue that production scenarios demand high-precision, high-performance NL2SQL systems, rather than simply high-quality SQL generation, which is the focus of most current NL2SQL approaches. In such scenarios, the careful selection of a static set of exemplars-capturing the intricacies of the query log, target database, SQL constructs, and execution latencies-plays a more crucial role than exemplar selection based solely on similarity. The key challenge, however, lies in identifying a representative set of exemplars for a given production setting. To this end, we propose a prompt optimization framework that not only addresses the high-precision requirement but also optimizes the performance of the generated SQL through multi-objective optimization. Preliminary empirical analysis demonstrates the effectiveness of the proposed framework. 

---
# Emotion Classification In-Context in Spanish 

**Authors**: Bipul Thapa, Gabriel Cofre  

**Link**: [PDF](https://arxiv.org/pdf/2505.20571)  

**Abstract**: Classifying customer feedback into distinct emotion categories is essential for understanding sentiment and improving customer experience. In this paper, we classify customer feedback in Spanish into three emotion categories--positive, neutral, and negative--using advanced NLP and ML techniques. Traditional methods translate feedback from widely spoken languages to less common ones, resulting in a loss of semantic integrity and contextual nuances inherent to the original language. To address this limitation, we propose a hybrid approach that combines TF-IDF with BERT embeddings, effectively transforming Spanish text into rich numerical representations that preserve the semantic depth of the original language by using a Custom Stacking Ensemble (CSE) approach. To evaluate emotion classification, we utilize a range of models, including Logistic Regression, KNN, Bagging classifier with LGBM, and AdaBoost. The CSE model combines these classifiers as base models and uses a one-vs-all Logistic Regression as the meta-model. Our experimental results demonstrate that CSE significantly outperforms the individual and BERT model, achieving a test accuracy of 93.3% on the native Spanish dataset--higher than the accuracy obtained from the translated version. These findings underscore the challenges of emotion classification in Spanish and highlight the advantages of combining vectorization techniques like TF-IDF with BERT for improved accuracy. Our results provide valuable insights for businesses seeking to leverage emotion classification to enhance customer feedback analysis and service improvements. 

---
# The NaijaVoices Dataset: Cultivating Large-Scale, High-Quality, Culturally-Rich Speech Data for African Languages 

**Authors**: Chris Emezue, NaijaVoices Community, Busayo Awobade, Abraham Owodunni, Handel Emezue, Gloria Monica Tobechukwu Emezue, Nefertiti Nneoma Emezue, Sewade Ogun, Bunmi Akinremi, David Ifeoluwa Adelani, Chris Pal  

**Link**: [PDF](https://arxiv.org/pdf/2505.20564)  

**Abstract**: The development of high-performing, robust, and reliable speech technologies depends on large, high-quality datasets. However, African languages -- including our focus, Igbo, Hausa, and Yoruba -- remain under-represented due to insufficient data. Popular voice-enabled technologies do not support any of the 2000+ African languages, limiting accessibility for circa one billion people. While previous dataset efforts exist for the target languages, they lack the scale and diversity needed for robust speech models. To bridge this gap, we introduce the NaijaVoices dataset, a 1,800-hour speech-text dataset with 5,000+ speakers. We outline our unique data collection approach, analyze its acoustic diversity, and demonstrate its impact through finetuning experiments on automatic speech recognition, averagely achieving 75.86% (Whisper), 52.06% (MMS), and 42.33% (XLSR) WER improvements. These results highlight NaijaVoices' potential to advance multilingual speech processing for African languages. 

---
# Paths Not Taken: Understanding and Mending the Multilingual Factual Recall Pipeline 

**Authors**: Meng Lu, Ruochen Zhang, Ellie Pavlick, Carsten Eickhoff  

**Link**: [PDF](https://arxiv.org/pdf/2505.20546)  

**Abstract**: Multilingual large language models (LLMs) often exhibit factual inconsistencies across languages, with significantly better performance in factual recall tasks in English than in other languages. The causes of these failures, however, remain poorly understood. Using mechanistic analysis techniques, we uncover the underlying pipeline that LLMs employ, which involves using the English-centric factual recall mechanism to process multilingual queries and then translating English answers back into the target language. We identify two primary sources of error: insufficient engagement of the reliable English-centric mechanism for factual recall, and incorrect translation from English back into the target language for the final answer. To address these vulnerabilities, we introduce two vector interventions, both independent of languages and datasets, to redirect the model toward better internal paths for higher factual consistency. Our interventions combined increase the recall accuracy by over 35 percent for the lowest-performing language. Our findings demonstrate how mechanistic insights can be used to unlock latent multilingual capabilities in LLMs. 

---
# AstroVisBench: A Code Benchmark for Scientific Computing and Visualization in Astronomy 

**Authors**: Sebastian Antony Joseph, Syed Murtaza Husain, Stella S. R. Offner, Stéphanie Juneau, Paul Torrey, Adam S. Bolton, Juan P. Farias, Niall Gaffney, Greg Durrett, Junyi Jessy Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20538)  

**Abstract**: Large Language Models (LLMs) are being explored for applications in scientific research, including their capabilities to synthesize literature, answer research questions, generate research ideas, and even conduct computational experiments. Ultimately, our goal is for these to help scientists derive novel scientific insights. In many areas of science, such insights often arise from processing and visualizing data to understand its patterns. However, evaluating whether an LLM-mediated scientific workflow produces outputs conveying the correct scientific insights is challenging to evaluate and has not been addressed in past work. We introduce AstroVisBench, the first benchmark for both scientific computing and visualization in the astronomy domain. AstroVisBench judges a language model's ability to both (1) create astronomy-specific workflows to process and analyze data and (2) visualize the results of these workflows through complex plots. Our evaluation of visualizations uses a novel LLM-as-a-judge workflow, which is validated against annotation by five professional astronomers. Using AstroVisBench we present an evaluation of state-of-the-art language models, showing a significant gap in their ability to engage in astronomy research as useful assistants. This evaluation provides a strong end-to-end evaluation for AI scientists that offers a path forward for the development of visualization-based workflows, which are central to a broad range of domains from physics to biology. 

---
# Multimodal Emotion Recognition in Conversations: A Survey of Methods, Trends, Challenges and Prospects 

**Authors**: Chengyan Wu, Yiqiang Cai, Yang Liu, Pengxu Zhu, Yun Xue, Ziwei Gong, Julia Hirschberg, Bolei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.20511)  

**Abstract**: While text-based emotion recognition methods have achieved notable success, real-world dialogue systems often demand a more nuanced emotional understanding than any single modality can offer. Multimodal Emotion Recognition in Conversations (MERC) has thus emerged as a crucial direction for enhancing the naturalness and emotional understanding of human-computer interaction. Its goal is to accurately recognize emotions by integrating information from various modalities such as text, speech, and visual signals.
This survey offers a systematic overview of MERC, including its motivations, core tasks, representative methods, and evaluation strategies. We further examine recent trends, highlight key challenges, and outline future directions. As interest in emotionally intelligent systems grows, this survey provides timely guidance for advancing MERC research. 

---
# ArVoice: A Multi-Speaker Dataset for Arabic Speech Synthesis 

**Authors**: Hawau Olamide Toyin, Rufael Marew, Humaid Alblooshi, Samar M. Magdy, Hanan Aldarmaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.20506)  

**Abstract**: We introduce ArVoice, a multi-speaker Modern Standard Arabic (MSA) speech corpus with diacritized transcriptions, intended for multi-speaker speech synthesis, and can be useful for other tasks such as speech-based diacritic restoration, voice conversion, and deepfake detection. ArVoice comprises: (1) a new professionally recorded set from six voice talents with diverse demographics, (2) a modified subset of the Arabic Speech Corpus; and (3) high-quality synthetic speech from two commercial systems. The complete corpus consists of a total of 83.52 hours of speech across 11 voices; around 10 hours consist of human voices from 7 speakers. We train three open-source TTS and two voice conversion systems to illustrate the use cases of the dataset. The corpus is available for research use. 

---
# Large Language Models for IT Automation Tasks: Are We There Yet? 

**Authors**: Md Mahadi Hassan, John Salvador, Akond Rahman, Santu Karmaker  

**Link**: [PDF](https://arxiv.org/pdf/2505.20505)  

**Abstract**: LLMs show promise in code generation, yet their effectiveness for IT automation tasks, particularly for tools like Ansible, remains understudied. Existing benchmarks rely primarily on synthetic tasks that fail to capture the needs of practitioners who use IT automation tools, such as Ansible. We present ITAB (IT Automation Task Benchmark), a benchmark of 126 diverse tasks (e.g., configuring servers, managing files) where each task accounts for state reconciliation: a property unique to IT automation tools. ITAB evaluates LLMs' ability to generate functional Ansible automation scripts via dynamic execution in controlled environments. We evaluate 14 open-source LLMs, none of which accomplish pass@10 at a rate beyond 12%. To explain these low scores, we analyze 1,411 execution failures across the evaluated LLMs and identify two main categories of prevalent semantic errors: failures in state reconciliation related reasoning (44.87% combined from variable (11.43%), host (11.84%), path(11.63%), and template (9.97%) issues) and deficiencies in module-specific execution knowledge (24.37% combined from Attribute and parameter (14.44%) and module (9.93%) errors). Our findings reveal key limitations in open-source LLMs' ability to track state changes and apply specialized module knowledge, indicating that reliable IT automation will require major advances in state reasoning and domain-specific execution understanding. 

---
# Gatsby Without the 'E': Crafting Lipograms with LLMs 

**Authors**: Rohan Balasubramanian, Nitish Gokulakrishnan, Syeda Jannatus Saba, Steven Skiena  

**Link**: [PDF](https://arxiv.org/pdf/2505.20501)  

**Abstract**: Lipograms are a unique form of constrained writing where all occurrences of a particular letter are excluded from the text, typified by the novel Gadsby, which daringly avoids all usage of the letter 'e'. In this study, we explore the power of modern large language models (LLMs) by transforming the novel F. Scott Fitzgerald's The Great Gatsby into a fully 'e'-less text. We experimented with a range of techniques, from baseline methods like synonym replacement to sophisticated generative models enhanced with beam search and named entity analysis. We show that excluding up to 3.6% of the most common letters (up to the letter 'u') had minimal impact on the text's meaning, although translation fidelity rapidly and predictably decays with stronger lipogram constraints. Our work highlights the surprising flexibility of English under strict constraints, revealing just how adaptable and creative language can be. 

---
# Beyond Keywords: Evaluating Large Language Model Classification of Nuanced Ableism 

**Authors**: Naba Rizvi, Harper Strickland, Saleha Ahmedi, Aekta Kallepalli, Isha Khirwadkar, William Wu, Imani N. S. Munyaka, Nedjma Ousidhoum  

**Link**: [PDF](https://arxiv.org/pdf/2505.20500)  

**Abstract**: Large language models (LLMs) are increasingly used in decision-making tasks like résumé screening and content moderation, giving them the power to amplify or suppress certain perspectives. While previous research has identified disability-related biases in LLMs, little is known about how they conceptualize ableism or detect it in text. We evaluate the ability of four LLMs to identify nuanced ableism directed at autistic individuals. We examine the gap between their understanding of relevant terminology and their effectiveness in recognizing ableist content in context. Our results reveal that LLMs can identify autism-related language but often miss harmful or offensive connotations. Further, we conduct a qualitative comparison of human and LLM explanations. We find that LLMs tend to rely on surface-level keyword matching, leading to context misinterpretations, in contrast to human annotators who consider context, speaker identity, and potential impact. On the other hand, both LLMs and humans agree on the annotation scheme, suggesting that a binary classification is adequate for evaluating LLM performance, which is consistent with findings from prior studies involving human annotators. 

---
# Inceptive Transformers: Enhancing Contextual Representations through Multi-Scale Feature Learning Across Domains and Languages 

**Authors**: Asif Shahriar, Rifat Shahriyar, M Saifur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2505.20496)  

**Abstract**: Conventional transformer models typically compress the information from all tokens in a sequence into a single \texttt{[CLS]} token to represent global context-- an approach that can lead to information loss in tasks requiring localized or hierarchical cues. In this work, we introduce \textit{Inceptive Transformer}, a modular and lightweight architecture that enriches transformer-based token representations by integrating a multi-scale feature extraction module inspired by inception networks. Our model is designed to balance local and global dependencies by dynamically weighting tokens based on their relevance to a particular task. Evaluation across a diverse range of tasks including emotion recognition (both English and Bangla), irony detection, disease identification, and anti-COVID vaccine tweets classification shows that our models consistently outperform the baselines by 1\% to 14\% while maintaining efficiency. These findings highlight the versatility and cross-lingual applicability of our method for enriching transformer-based representations across diverse domains. 

---
# InFact: Informativeness Alignment for Improved LLM Factuality 

**Authors**: Roi Cohen, Russa Biswas, Gerard de Melo  

**Link**: [PDF](https://arxiv.org/pdf/2505.20487)  

**Abstract**: Factual completeness is a general term that captures how detailed and informative a factually correct text is. For instance, the factual sentence ``Barack Obama was born in the United States'' is factually correct, though less informative than the factual sentence ``Barack Obama was born in Honolulu, Hawaii, United States''. Despite the known fact that LLMs tend to hallucinate and generate factually incorrect text, they might also tend to choose to generate factual text that is indeed factually correct and yet less informative than other, more informative choices. In this work, we tackle this problem by proposing an informativeness alignment mechanism. This mechanism takes advantage of recent factual benchmarks to propose an informativeness alignment objective. This objective prioritizes answers that are both correct and informative. A key finding of our work is that when training a model to maximize this objective or optimize its preference, we can improve not just informativeness but also factuality. 

---
# Conversation Kernels: A Flexible Mechanism to Learn Relevant Context for Online Conversation Understanding 

**Authors**: Vibhor Agarwal, Arjoo Gupta, Suparna De, Nishanth Sastry  

**Link**: [PDF](https://arxiv.org/pdf/2505.20482)  

**Abstract**: Understanding online conversations has attracted research attention with the growth of social networks and online discussion forums. Content analysis of posts and replies in online conversations is difficult because each individual utterance is usually short and may implicitly refer to other posts within the same conversation. Thus, understanding individual posts requires capturing the conversational context and dependencies between different parts of a conversation tree and then encoding the context dependencies between posts and comments/replies into the language model.
To this end, we propose a general-purpose mechanism to discover appropriate conversational context for various aspects about an online post in a conversation, such as whether it is informative, insightful, interesting or funny. Specifically, we design two families of Conversation Kernels, which explore different parts of the neighborhood of a post in the tree representing the conversation and through this, build relevant conversational context that is appropriate for each task being considered. We apply our developed method to conversations crawled from this http URL, which allows users to apply highly different labels to posts, such as 'insightful', 'funny', etc., and therefore provides an ideal experimental platform to study whether a framework such as Conversation Kernels is general-purpose and flexible enough to be adapted to disparately different conversation understanding tasks. 

---
# Amulet: Putting Complex Multi-Turn Conversations on the Stand with LLM Juries 

**Authors**: Sahana Ramnath, Anurag Mudgil, Brihi Joshi, Skyler Hallinan, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.20451)  

**Abstract**: Today, large language models are widely used as judges to evaluate responses from other language models. Hence, it is imperative to benchmark and improve these LLM-judges on real-world language model usage: a typical human-assistant conversation is lengthy, and shows significant diversity in topics, intents, and requirements across turns, e.g. social interactions, task requests, feedback. We present Amulet, a framework that leverages pertinent linguistic concepts of dialog-acts and maxims to improve the accuracy of LLM-judges on preference data with complex, multi-turn conversational context. Amulet presents valuable insights about (a) the communicative structures and intents present in the conversation (dialog acts), and (b) the satisfaction of conversational principles (maxims) by the preference responses, and uses them to make judgments. On four challenging datasets, Amulet shows that (a) humans frequently (60 to 70 percent of the time) change their intents from one turn of the conversation to the next, and (b) in 75 percent of instances, the preference responses can be differentiated via dialog acts and/or maxims, reiterating the latter's significance in judging such data. Amulet can be used either as a judge by applying the framework to a single LLM, or integrated into a jury with different LLM judges; our judges and juries show strong improvements on relevant baselines for all four datasets. 

---
# In-context Language Learning for Endangered Languages in Speech Recognition 

**Authors**: Zhaolin Li, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2505.20445)  

**Abstract**: With approximately 7,000 languages spoken worldwide, current large language models (LLMs) support only a small subset. Prior research indicates LLMs can learn new languages for certain tasks without supervised data. We extend this investigation to speech recognition, investigating whether LLMs can learn unseen, low-resource languages through in-context learning (ICL). With experiments on four diverse endangered languages that LLMs have not been trained on, we find that providing more relevant text samples enhances performance in both language modelling and Automatic Speech Recognition (ASR) tasks. Furthermore, we show that the probability-based approach outperforms the traditional instruction-based approach in language learning. Lastly, we show ICL enables LLMs to achieve ASR performance that is comparable to or even surpasses dedicated language models trained specifically for these languages, while preserving the original capabilities of the LLMs. 

---
# HAMburger: Accelerating LLM Inference via Token Smashing 

**Authors**: Jingyu Liu, Ce Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20438)  

**Abstract**: The growing demand for efficient Large Language Model (LLM) inference requires a holistic optimization on algorithms, systems, and hardware. However, very few works have fundamentally changed the generation pattern: each token needs one forward pass and one KV cache. This can be sub-optimal because we found that LLMs are extremely capable of self-identifying the exact dose of information that a single KV cache can store, and many tokens can be generated confidently without global context. Based on this insight, we introduce HAMburger, a Hierarchically Auto-regressive Model that redefines resource allocation in LLMs by moving beyond uniform computation and storage per token during inference. Stacking a compositional embedder and a micro-step decoder in between a base LLM, HAMburger smashes multiple tokens into a single KV and generates several tokens per step. Additionally, HAMburger functions as a speculative decoding framework where it can blindly trust self-drafted tokens. As a result, HAMburger shifts the growth of KV cache and forward FLOPs from linear to sub-linear with respect to output length, and adjusts its inference speed based on query perplexity and output structure. Extensive evaluations show that HAMburger reduces the KV cache computation by up to 2$\times$ and achieves up to 2$\times$ TPS, while maintaining quality in both short- and long-context tasks. Our method explores an extremely challenging inference regime that requires both computation- and memory-efficiency with a hardware-agnostic design. 

---
# PreP-OCR: A Complete Pipeline for Document Image Restoration and Enhanced OCR Accuracy 

**Authors**: Shuhao Guan, Moule Lin, Cheng Xu, Xinyi Liu, Jinman Zhao, Jiexin Fan, Qi Xu, Derek Greene  

**Link**: [PDF](https://arxiv.org/pdf/2505.20429)  

**Abstract**: This paper introduces PreP-OCR, a two-stage pipeline that combines document image restoration with semantic-aware post-OCR correction to improve text extraction from degraded historical documents. Our key innovation lies in jointly optimizing image clarity and linguistic consistency. First, we generate synthetic image pairs with randomized text fonts, layouts, and degradations. An image restoration model is trained on this synthetic data, using multi-directional patch extraction and fusion to process large images. Second, a ByT5 post-corrector, fine-tuned on synthetic historical text training pairs, addresses any remaining OCR errors. Detailed experiments on 13,831 pages of real historical documents in English, French, and Spanish show that PreP-OCR pipeline reduces character error rates by 63.9-70.3\% compared to OCR on raw images. Our pipeline demonstrates the potential of integrating image restoration with linguistic error correction for digitizing historical archives. 

---
# The UD-NewsCrawl Treebank: Reflections and Challenges from a Large-scale Tagalog Syntactic Annotation Project 

**Authors**: Angelina A. Aquino, Lester James V. Miranda, Elsie Marie T. Or  

**Link**: [PDF](https://arxiv.org/pdf/2505.20428)  

**Abstract**: This paper presents UD-NewsCrawl, the largest Tagalog treebank to date, containing 15.6k trees manually annotated according to the Universal Dependencies framework. We detail our treebank development process, including data collection, pre-processing, manual annotation, and quality assurance procedures. We provide baseline evaluations using multiple transformer-based models to assess the performance of state-of-the-art dependency parsers on Tagalog. We also highlight challenges in the syntactic analysis of Tagalog given its distinctive grammatical properties, and discuss its implications for the annotation of this treebank. We anticipate that UD-NewsCrawl and our baseline model implementations will serve as valuable resources for advancing computational linguistics research in underrepresented languages like Tagalog. 

---
# SEMMA: A Semantic Aware Knowledge Graph Foundation Model 

**Authors**: Arvindh Arun, Sumit Kumar, Mojtaba Nayyeri, Bo Xiong, Ponnurangam Kumaraguru, Antonio Vergari, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2505.20422)  

**Abstract**: Knowledge Graph Foundation Models (KGFMs) have shown promise in enabling zero-shot reasoning over unseen graphs by learning transferable patterns. However, most existing KGFMs rely solely on graph structure, overlooking the rich semantic signals encoded in textual attributes. We introduce SEMMA, a dual-module KGFM that systematically integrates transferable textual semantics alongside structure. SEMMA leverages Large Language Models (LLMs) to enrich relation identifiers, generating semantic embeddings that subsequently form a textual relation graph, which is fused with the structural component. Across 54 diverse KGs, SEMMA outperforms purely structural baselines like ULTRA in fully inductive link prediction. Crucially, we show that in more challenging generalization settings, where the test-time relation vocabulary is entirely unseen, structural methods collapse while SEMMA is 2x more effective. Our findings demonstrate that textual semantics are critical for generalization in settings where structure alone fails, highlighting the need for foundation models that unify structural and linguistic signals in knowledge reasoning. 

---
# GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation 

**Authors**: Zihong Chen, Wanli Jiang, Jinzhe Li, Zhonghang Yuan, Huanjun Kong, Wanli Ouyang, Nanqing Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.20416)  

**Abstract**: Fine-tuning for large language models (LLMs) typically requires substantial amounts of high-quality supervised data, which is both costly and labor-intensive to acquire. While synthetic data generation has emerged as a promising solution, existing approaches frequently suffer from factual inaccuracies, insufficient long-tail coverage, simplistic knowledge structures, and homogenized outputs. To address these challenges, we introduce GraphGen, a knowledge graph-guided framework designed for three key question-answering (QA) scenarios: atomic QA, aggregated QA, and multi-hop QA. It begins by constructing a fine-grained knowledge graph from the source text. It then identifies knowledge gaps in LLMs using the expected calibration error metric, prioritizing the generation of QA pairs that target high-value, long-tail knowledge. Furthermore, GraphGen incorporates multi-hop neighborhood sampling to capture complex relational information and employs style-controlled generation to diversify the resulting QA data. Experimental results on knowledge-intensive tasks under closed-book settings demonstrate that GraphGen outperforms conventional synthetic data methods, offering a more reliable and comprehensive solution to the data scarcity challenge in supervised fine-tuning. The code and data are publicly available at this https URL. 

---
# Enhancing Logical Reasoning in Language Models via Symbolically-Guided Monte Carlo Process Supervision 

**Authors**: Xingwei Tan, Marco Valentino, Mahmud Akhter, Maria Liakata, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2505.20415)  

**Abstract**: Large language models (LLMs) have shown promising performance in mathematical and logical reasoning benchmarks. However, recent studies have pointed to memorization, rather than generalization, as one of the leading causes for such performance. LLMs, in fact, are susceptible to content variations, demonstrating a lack of robust symbolic abstractions supporting their reasoning process. To improve reliability, many attempts have been made to combine LLMs with symbolic methods. Nevertheless, existing approaches fail to effectively leverage symbolic representations due to the challenges involved in developing reliable and scalable verification mechanisms. In this paper, we propose to overcome such limitations by generating symbolic reasoning trajectories and select the high-quality ones using a process reward model automatically tuned based on Monte Carlo estimation. The trajectories are then employed via fine-tuning methods to improve logical reasoning and generalization. Our results on logical reasoning benchmarks such as FOLIO and LogicAsker show the effectiveness of the proposed method with large gains on frontier and open-weight models. Moreover, additional experiments on claim verification reveal that fine-tuning on the generated symbolic reasoning trajectories enhances out-of-domain generalizability, suggesting the potential impact of symbolically-guided process supervision in alleviating the effect of memorization on LLM reasoning. 

---
# Rethinking Text-based Protein Understanding: Retrieval or LLM? 

**Authors**: Juntong Wu, Zijing Liu, He Cao, Hao Li, Bin Feng, Zishan Shu, Ke Yu, Li Yuan, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20354)  

**Abstract**: In recent years, protein-text models have gained significant attention for their potential in protein generation and understanding. Current approaches focus on integrating protein-related knowledge into large language models through continued pretraining and multi-modal alignment, enabling simultaneous comprehension of textual descriptions and protein sequences. Through a thorough analysis of existing model architectures and text-based protein understanding benchmarks, we identify significant data leakage issues present in current benchmarks. Moreover, conventional metrics derived from natural language processing fail to accurately assess the model's performance in this domain. To address these limitations, we reorganize existing datasets and introduce a novel evaluation framework based on biological entities. Motivated by our observation, we propose a retrieval-enhanced method, which significantly outperforms fine-tuned LLMs for protein-to-text generation and shows accuracy and efficiency in training-free scenarios. Our code and data can be seen at this https URL. 

---
# SeRL: Self-Play Reinforcement Learning for Large Language Models with Limited Data 

**Authors**: Wenkai Fang, Shunyu Liu, Yang Zhou, Kongcheng Zhang, Tongya Zheng, Kaixuan Chen, Mingli Song, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.20347)  

**Abstract**: Recent advances have demonstrated the effectiveness of Reinforcement Learning (RL) in improving the reasoning capabilities of Large Language Models (LLMs). However, existing works inevitably rely on high-quality instructions and verifiable rewards for effective training, both of which are often difficult to obtain in specialized domains. In this paper, we propose Self-play Reinforcement Learning(SeRL) to bootstrap LLM training with limited initial data. Specifically, SeRL comprises two complementary modules: self-instruction and self-rewarding. The former module generates additional instructions based on the available data at each training step, employing robust online filtering strategies to ensure instruction quality, diversity, and difficulty. The latter module introduces a simple yet effective majority-voting mechanism to estimate response rewards for additional instructions, eliminating the need for external annotations. Finally, SeRL performs conventional RL based on the generated data, facilitating iterative self-play learning. Extensive experiments on various reasoning benchmarks and across different LLM backbones demonstrate that the proposed SeRL yields results superior to its counterparts and achieves performance on par with those obtained by high-quality data with verifiable rewards. Our code is available at this https URL. 

---
# Do LLMs have a Gender (Entropy) Bias? 

**Authors**: Sonal Prabhune, Balaji Padmanabhan, Kaushik Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2505.20343)  

**Abstract**: We investigate the existence and persistence of a specific type of gender bias in some of the popular LLMs and contribute a new benchmark dataset, RealWorldQuestioning (released on HuggingFace ), developed from real-world questions across four key domains in business and health contexts: education, jobs, personal financial management, and general health. We define and study entropy bias, which we define as a discrepancy in the amount of information generated by an LLM in response to real questions users have asked. We tested this using four different LLMs and evaluated the generated responses both qualitatively and quantitatively by using ChatGPT-4o (as "LLM-as-judge"). Our analyses (metric-based comparisons and "LLM-as-judge" evaluation) suggest that there is no significant bias in LLM responses for men and women at a category level. However, at a finer granularity (the individual question level), there are substantial differences in LLM responses for men and women in the majority of cases, which "cancel" each other out often due to some responses being better for males and vice versa. This is still a concern since typical users of these tools often ask a specific question (only) as opposed to several varied ones in each of these common yet important areas of life. We suggest a simple debiasing approach that iteratively merges the responses for the two genders to produce a final result. Our approach demonstrates that a simple, prompt-based debiasing strategy can effectively debias LLM outputs, thus producing responses with higher information content than both gendered variants in 78% of the cases, and consistently achieving a balanced integration in the remaining cases. 

---
# Dynamic Manifold Evolution Theory: Modeling and Stability Analysis of Latent Representations in Large Language Models 

**Authors**: Yukun Zhang, Qi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.20340)  

**Abstract**: We introduce Dynamic Manifold Evolution Theory (DMET),a unified framework that models large language model generation as a controlled dynamical system evolving on a low_dimensional semantic manifold. By casting latent_state updates as discrete time Euler approximations of continuous dynamics, we map intrinsic energy_driven flows and context_dependent forces onto Transformer components (residual connections, attention, feed-forward networks). Leveraging Lyapunov stability theory We define three empirical metrics (state continuity, clustering quality, topological persistence) that quantitatively link latent_trajectory properties to text fluency, grammaticality, and semantic coherence. Extensive experiments across decoding parameters validate DMET's predictions and yield principled guidelines for balancing creativity and consistency in text generation. 

---
# Assessing the Capability of LLMs in Solving POSCOMP Questions 

**Authors**: Cayo Viegas, Rohit Gheyi, Márcio Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2505.20338)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have significantly expanded the capabilities of artificial intelligence in natural language processing tasks. Despite this progress, their performance in specialized domains such as computer science remains relatively unexplored. Understanding the proficiency of LLMs in these domains is critical for evaluating their practical utility and guiding future developments. The POSCOMP, a prestigious Brazilian examination used for graduate admissions in computer science promoted by the Brazlian Computer Society (SBC), provides a challenging benchmark. This study investigates whether LLMs can match or surpass human performance on the POSCOMP exam. Four LLMs - ChatGPT-4, Gemini 1.0 Advanced, Claude 3 Sonnet, and Le Chat Mistral Large - were initially evaluated on the 2022 and 2023 POSCOMP exams. The assessments measured the models' proficiency in handling complex questions typical of the exam. LLM performance was notably better on text-based questions than on image interpretation tasks. In the 2022 exam, ChatGPT-4 led with 57 correct answers out of 69 questions, followed by Gemini 1.0 Advanced (49), Le Chat Mistral (48), and Claude 3 Sonnet (44). Similar trends were observed in the 2023 exam. ChatGPT-4 achieved the highest performance, surpassing all students who took the POSCOMP 2023 exam. LLMs, particularly ChatGPT-4, show promise in text-based tasks on the POSCOMP exam, although image interpretation remains a challenge. Given the rapid evolution of LLMs, we expanded our analysis to include more recent models - o1, Gemini 2.5 Pro, Claude 3.7 Sonnet, and o3-mini-high - evaluated on the 2022-2024 POSCOMP exams. These newer models demonstrate further improvements and consistently surpass both the average and top-performing human participants across all three years. 

---
# MOSLIM:Align with diverse preferences in prompts through reward classification 

**Authors**: Yu Zhang, Wanli Jiang, Zhengyu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20336)  

**Abstract**: The multi-objective alignment of Large Language Models (LLMs) is essential for ensuring foundational models conform to diverse human preferences. Current research in this field typically involves either multiple policies or multiple reward models customized for various preferences, or the need to train a preference-specific supervised fine-tuning (SFT) model. In this work, we introduce a novel multi-objective alignment method, MOSLIM, which utilizes a single reward model and policy model to address diverse objectives. MOSLIM provides a flexible way to control these objectives through prompting and does not require preference training during SFT phase, allowing thousands of off-the-shelf models to be directly utilized within this training framework. MOSLIM leverages a multi-head reward model that classifies question-answer pairs instead of scoring them and then optimize policy model with a scalar reward derived from a mapping function that converts classification results from reward model into reward scores. We demonstrate the efficacy of our proposed method across several multi-objective benchmarks and conduct ablation studies on various reward model sizes and policy optimization methods. The MOSLIM method outperforms current multi-objective approaches in most results while requiring significantly fewer GPU computing resources compared with existing policy optimization methods. 

---
# Language Model Distillation: A Temporal Difference Imitation Learning Perspective 

**Authors**: Zishun Yu, Shangzhe Li, Xinhua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20335)  

**Abstract**: Large language models have led to significant progress across many NLP tasks, although their massive sizes often incur substantial computational costs. Distillation has become a common practice to compress these large and highly capable models into smaller, more efficient ones. Many existing language model distillation methods can be viewed as behavior cloning from the perspective of imitation learning or inverse reinforcement learning. This viewpoint has inspired subsequent studies that leverage (inverse) reinforcement learning techniques, including variations of behavior cloning and temporal difference learning methods. Rather than proposing yet another specific temporal difference method, we introduce a general framework for temporal difference-based distillation by exploiting the distributional sparsity of the teacher model. Specifically, it is often observed that language models assign most probability mass to a small subset of tokens. Motivated by this observation, we design a temporal difference learning framework that operates on a reduced action space (a subset of vocabulary), and demonstrate how practical algorithms can be derived and the resulting performance improvements. 

---
# Lookahead Q-Cache: Achieving More Consistent KV Cache Eviction via Pseudo Query 

**Authors**: Yixuan Wang, Shiyu Ji, Yijun Liu, Yuzhuang Xu, Yang Xu, Qingfu Zhu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2505.20334)  

**Abstract**: Large language models (LLMs) rely on key-value cache (KV cache) to accelerate decoding by reducing redundant computations. However, the KV cache memory usage grows substantially with longer text sequences, posing challenges for efficient deployment. Existing KV cache eviction methods prune tokens using prefilling-stage attention scores, causing inconsistency with actual inference queries, especially under tight memory budgets. In this paper, we propose Lookahead Q-Cache (LAQ), a novel eviction framework that generates low-cost pseudo lookahead queries to better approximate the true decoding-stage queries. By using these lookahead queries as the observation window for importance estimation, LAQ achieves more consistent and accurate KV cache eviction aligned with real inference scenarios. Experimental results on LongBench and Needle-in-a-Haystack benchmarks show that LAQ outperforms existing methods across various budget levels, achieving a 1 $\sim$ 4 point improvement on LongBench under limited cache budget. Moreover, LAQ is complementary to existing approaches and can be flexibly combined to yield further improvements. 

---
# Multi-Scale Manifold Alignment: A Unified Framework for Enhanced Explainability of Large Language Models 

**Authors**: Yukun Zhang, Qi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.20333)  

**Abstract**: Recent advances in Large Language Models (LLMs) have achieved strong performance, yet their internal reasoning remains opaque, limiting interpretability and trust in critical applications. We propose a novel Multi_Scale Manifold Alignment framework that decomposes the latent space into global, intermediate, and local semantic manifolds capturing themes, context, and word-level details. Our method introduces cross_scale mapping functions that jointly enforce geometric alignment (e.g., Procrustes analysis) and information preservation (via mutual information constraints like MINE or VIB). We further incorporate curvature regularization and hyperparameter tuning for stable optimization. Theoretical analysis shows that alignment error, measured by KL divergence, can be bounded under mild assumptions. This framework offers a unified explanation of how LLMs structure multi-scale semantics, advancing interpretability and enabling applications such as bias detection and robustness enhancement. 

---
# Guided by Gut: Efficient Test-Time Scaling with Reinforced Intrinsic Confidence 

**Authors**: Amirhosein Ghasemabadi, Keith G. Mills, Baochun Li, Di Niu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20325)  

**Abstract**: Test-Time Scaling (TTS) methods for enhancing Large Language Model (LLM) reasoning often incur substantial computational costs, primarily due to extensive reliance on external Process Reward Models (PRMs) or sampling methods like Best-of-N (BoN). This paper introduces Guided by Gut (GG), an efficient self-guided TTS framework that achieves PRM-level performance without costly external verifier models. Our method employs a lightweight tree search guided solely by intrinsic LLM signals, token-level confidence and step novelty. One critical innovation is improving the reliability of internal confidence estimates via a targeted reinforcement learning fine-tuning phase. Empirical evaluations on challenging mathematical reasoning benchmarks demonstrate that GG enables smaller models (e.g., 1.5B parameters) to achieve accuracy matching or surpassing significantly larger models (e.g., 32B-70B parameters), while reducing GPU memory usage by up to 10x. Compared to PRM-based methods, GG achieves comparable accuracy with 8x faster inference speeds and 4-5x lower memory usage. Additionally, GG reduces KV cache memory usage by approximately 50% compared to the BoN strategy, facilitating more efficient and practical deployment of TTS techniques. 

---
# PMOA-TTS: Introducing the PubMed Open Access Textual Times Series Corpus 

**Authors**: Shahriar Noroozizadeh, Sayantan Kumar, George H. Chen, Jeremy C. Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2505.20323)  

**Abstract**: Understanding temporal dynamics in clinical narratives is essential for modeling patient trajectories, yet large-scale temporally annotated resources remain limited. We present PMOA-TTS, the first openly available dataset of 124,699 PubMed Open Access (PMOA) case reports, each converted into structured (event, time) timelines via a scalable LLM-based pipeline. Our approach combines heuristic filtering with Llama 3.3 to identify single-patient case reports, followed by prompt-driven extraction using Llama 3.3 and DeepSeek R1, resulting in over 5.6 million timestamped clinical events. To assess timeline quality, we evaluate against a clinician-curated reference set using three metrics: (i) event-level matching (80% match at a cosine similarity threshold of 0.1), (ii) temporal concordance (c-index > 0.90), and (iii) Area Under the Log-Time CDF (AULTC) for timestamp alignment. Corpus-level analysis shows wide diagnostic and demographic coverage. In a downstream survival prediction task, embeddings from extracted timelines achieve time-dependent concordance indices up to 0.82 $\pm$ 0.01, demonstrating the predictive value of temporally structured narratives. PMOA-TTS provides a scalable foundation for timeline extraction, temporal reasoning, and longitudinal modeling in biomedical NLP. The dataset is available at: this https URL . 

---
# Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms 

**Authors**: Mengru Wang, Ziwen Xu, Shengyu Mao, Shumin Deng, Zhaopeng Tu, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20322)  

**Abstract**: Precise control over language model generation is vital for ensuring both safety and reliability. Although prompt engineering and steering are commonly used to intervene in model behaviors, the vast number of parameters in models often results in highly intertwined internal representations. This interdependency can limit control precision and sometimes lead to unintended side effects. Recent research has explored the use of sparse autoencoders (SAE) to disentangle knowledge in high-dimensional spaces for steering. However, these applications have been limited to toy tasks owing to the nontrivial issue of locating atomic knowledge components. In this paper, we propose Steering Target Atoms (STA), a novel method that isolates and manipulates disentangled knowledge components to enhance safety. Comprehensive experiments demonstrate the effectiveness of our approach. Further analysis reveals that steering exhibits superior robustness and flexibility, particularly in adversarial scenarios. We also apply the steering strategy to the large reasoning model, confirming its effectiveness in precise reasoning control. 

---
# BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases 

**Authors**: Mathew J. Koretsky, Maya Willey, Adi Asija, Owen Bianchi, Chelsea X. Alvarado, Tanay Nayak, Nicole Kuznetsov, Sungwon Kim, Mike A. Nalls, Daniel Khashabi, Faraz Faghri  

**Link**: [PDF](https://arxiv.org/pdf/2505.20321)  

**Abstract**: Biomedical researchers increasingly rely on large-scale structured databases for complex analytical tasks. However, current text-to-SQL systems often struggle to map qualitative scientific questions into executable SQL, particularly when implicit domain reasoning is required. We introduce BiomedSQL, the first benchmark explicitly designed to evaluate scientific reasoning in text-to-SQL generation over a real-world biomedical knowledge base. BiomedSQL comprises 68,000 question/SQL query/answer triples grounded in a harmonized BigQuery knowledge base that integrates gene-disease associations, causal inference from omics data, and drug approval records. Each question requires models to infer domain-specific criteria, such as genome-wide significance thresholds, effect directionality, or trial phase filtering, rather than rely on syntactic translation alone. We evaluate a range of open- and closed-source LLMs across prompting strategies and interaction paradigms. Our results reveal a substantial performance gap: GPT-o3-mini achieves 59.0% execution accuracy, while our custom multi-step agent, BMSQL, reaches 62.6%, both well below the expert baseline of 90.0%. BiomedSQL provides a new foundation for advancing text-to-SQL systems capable of supporting scientific discovery through robust reasoning over structured biomedical knowledge bases. Our dataset is publicly available at this https URL, and our code is open-source at this https URL. 

---
# Less Context, Same Performance: A RAG Framework for Resource-Efficient LLM-Based Clinical NLP 

**Authors**: Satya Narayana Cheetirala, Ganesh Raut, Dhavalkumar Patel, Fabio Sanatana, Robert Freeman, Matthew A Levin, Girish N. Nadkarni, Omar Dawkins, Reba Miller, Randolph M. Steinhagen, Eyal Klang, Prem Timsina  

**Link**: [PDF](https://arxiv.org/pdf/2505.20320)  

**Abstract**: Long text classification is challenging for Large Language Models (LLMs) due to token limits and high computational costs. This study explores whether a Retrieval Augmented Generation (RAG) approach using only the most relevant text segments can match the performance of processing entire clinical notes with large context LLMs. We begin by splitting clinical documents into smaller chunks, converting them into vector embeddings, and storing these in a FAISS index. We then retrieve the top 4,000 words most pertinent to the classification query and feed these consolidated segments into an LLM. We evaluated three LLMs (GPT4o, LLaMA, and Mistral) on a surgical complication identification task. Metrics such as AUC ROC, precision, recall, and F1 showed no statistically significant differences between the RAG based approach and whole-text processing (p > 0.05p > 0.05). These findings indicate that RAG can significantly reduce token usage without sacrificing classification accuracy, providing a scalable and cost effective solution for analyzing lengthy clinical documents. 

---
# Beyond Demonstrations: Dynamic Vector Construction from Latent Representations 

**Authors**: Wang Cai, Hsiu-Yuan Huang, Zhixiang Wang, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20318)  

**Abstract**: In-Context derived Vector (ICV) methods extract task-relevant representations from large language models (LLMs) and reinject them during inference, achieving comparable performance to few-shot In-Context Learning (ICL) without repeated demonstration processing. However, existing ICV methods remain sensitive to ICL-specific factors, often use coarse or semantically fragmented representations as the source of the vector, and rely on heuristic-based injection positions, limiting their applicability.
To address these issues, we propose Dynamic Vector (DyVec), which incorporates an Exhaustive Query Rotation (EQR) strategy to extract robust semantically aggregated latent representations by mitigating variance introduced by ICL. It then applies Dynamic Latent Segmentation and Injection to adaptively partition representations based on task complexity and leverages REINFORCE-based optimization to learn optimal injection positions for each segment.
Experiments results show that DyVec outperforms few-shot ICL, LoRA, and prior ICV baselines. Further analysis highlights the effectiveness of dynamically segmenting and injecting semantically aggregated latent representations. DyVec provides a lightweight and data-efficient solution for inference-time task adaptation. 

---
# Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL 

**Authors**: Zhewei Yao, Guoheng Sun, Lukasz Borchmann, Zheyu Shen, Minghang Deng, Bohan Zhai, Hao Zhang, Ang Li, Yuxiong He  

**Link**: [PDF](https://arxiv.org/pdf/2505.20315)  

**Abstract**: Translating natural language into SQL (Test2SQL) is a longstanding challenge at the intersection of natural language understanding and structured data access. While large language models (LLMs) have significantly improved fluency in SQL generation, producing correct and executable SQL--particularly for complex queries--remains a bottleneck. We present Arctic-Text2SQL-R1, a reinforcement learning (RL) framework and model family designed to generate accurate, executable SQL using a lightweight reward signal based solely on execution correctness. Our approach avoids brittle intermediate supervision and complex reward shaping, promoting stable training and alignment with the end task. Combined with carefully curated data, strong supervised initialization, and effective training practices, Arctic-Text2SQL-R1 achieves state-of-the-art execution accuracy across six diverse Test2SQL benchmarks, including the top position on the BIRD leaderboard. Notably, our 7B model outperforms prior 70B-class systems, highlighting the framework's scalability and efficiency. We further demonstrate inference-time robustness through simple extensions like value retrieval and majority voting. Extensive experiments and ablation studies offer both positive and negative insights, providing practical guidance for future Test2SQL research. 

---
# Guiding Giants: Lightweight Controllers for Weighted Activation Steering in LLMs 

**Authors**: Amr Hegazy, Mostafa Elhoushi, Amr Alanwar  

**Link**: [PDF](https://arxiv.org/pdf/2505.20309)  

**Abstract**: Controlling undesirable Large Language Model (LLM) behaviors, such as the generation of unsafe content or failing to adhere to safety guidelines, often relies on costly fine-tuning. Activation steering provides an alternative for inference-time control, but existing methods typically lack fine-grained, adaptive mechanisms. We introduce a novel approach using a lightweight, trainable controller network integrated during inference. This controller network observes specific intermediate LLM activations and predicts both a global scaling factor and layer-specific weights. The predicted global scaling factor and layer-specific weights then dynamically modulate the intensity of a steering patch, derived from a pre-computed "refusal direction" vector, applied across the LLM's layers during generation. Trained on activations from both harmful and benign prompts, our controller learns to discriminatively apply nuanced, layer-aware interventions, activating steering primarily for harmful inputs. Experiments using safety benchmarks like ToxicChat & In-The-Wild Jailbreak Prompts demonstrate that our weighted steering controller significantly increases refusal rates compared to the base LLM, achieving targeted behavioral modification without altering the original model parameters. Our experiments with Llama-3.1-8B, Llama-3.2-1B & Mistral-7B show our approach outperforms existing methods, presenting an efficient and adaptive method for fine-grained control over LLM behavior at inference time. 

---
# ViewSpatial-Bench: Evaluating Multi-perspective Spatial Localization in Vision-Language Models 

**Authors**: Dingming Li, Hongxing Li, Zixuan Wang, Yuchen Yan, Hang Zhang, Siqi Chen, Guiyang Hou, Shengpei Jiang, Wenqi Zhang, Yongliang Shen, Weiming Lu, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21500)  

**Abstract**: Vision-language models (VLMs) have demonstrated remarkable capabilities in understanding and reasoning about visual content, but significant challenges persist in tasks requiring cross-viewpoint understanding and spatial reasoning. We identify a critical limitation: current VLMs excel primarily at egocentric spatial reasoning (from the camera's perspective) but fail to generalize to allocentric viewpoints when required to adopt another entity's spatial frame of reference. We introduce ViewSpatial-Bench, the first comprehensive benchmark designed specifically for multi-viewpoint spatial localization recognition evaluation across five distinct task types, supported by an automated 3D annotation pipeline that generates precise directional labels. Comprehensive evaluation of diverse VLMs on ViewSpatial-Bench reveals a significant performance disparity: models demonstrate reasonable performance on camera-perspective tasks but exhibit reduced accuracy when reasoning from a human viewpoint. By fine-tuning VLMs on our multi-perspective spatial dataset, we achieve an overall performance improvement of 46.24% across tasks, highlighting the efficacy of our approach. Our work establishes a crucial benchmark for spatial intelligence in embodied AI systems and provides empirical evidence that modeling 3D spatial relationships enhances VLMs' corresponding spatial comprehension capabilities. 

---
# Paper2Poster: Towards Multimodal Poster Automation from Scientific Papers 

**Authors**: Wei Pang, Kevin Qinghong Lin, Xiangru Jian, Xi He, Philip Torr  

**Link**: [PDF](https://arxiv.org/pdf/2505.21497)  

**Abstract**: Academic poster generation is a crucial yet challenging task in scientific communication, requiring the compression of long-context interleaved documents into a single, visually coherent page. To address this challenge, we introduce the first benchmark and metric suite for poster generation, which pairs recent conference papers with author-designed posters and evaluates outputs on (i)Visual Quality-semantic alignment with human posters, (ii)Textual Coherence-language fluency, (iii)Holistic Assessment-six fine-grained aesthetic and informational criteria scored by a VLM-as-judge, and notably (iv)PaperQuiz-the poster's ability to convey core paper content as measured by VLMs answering generated quizzes. Building on this benchmark, we propose PosterAgent, a top-down, visual-in-the-loop multi-agent pipeline: the (a)Parser distills the paper into a structured asset library; the (b)Planner aligns text-visual pairs into a binary-tree layout that preserves reading order and spatial balance; and the (c)Painter-Commenter loop refines each panel by executing rendering code and using VLM feedback to eliminate overflow and ensure alignment. In our comprehensive evaluation, we find that GPT-4o outputs-though visually appealing at first glance-often exhibit noisy text and poor PaperQuiz scores, and we find that reader engagement is the primary aesthetic bottleneck, as human-designed posters rely largely on visual semantics to convey meaning. Our fully open-source variants (e.g. based on the Qwen-2.5 series) outperform existing 4o-driven multi-agent systems across nearly all metrics, while using 87% fewer tokens. It transforms a 22-page paper into a finalized yet editable .pptx poster - all for just $0.005. These findings chart clear directions for the next generation of fully automated poster-generation models. The code and datasets are available at this https URL. 

---
# Reinforcing General Reasoning without Verifiers 

**Authors**: Xiangxin Zhou, Zichen Liu, Anya Sims, Haonan Wang, Tianyu Pang, Chongxuan Li, Liang Wang, Min Lin, Chao Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.21493)  

**Abstract**: The recent paradigm shift towards training large language models (LLMs) using DeepSeek-R1-Zero-style reinforcement learning (RL) on verifiable rewards has led to impressive advancements in code and mathematical reasoning. However, this methodology is limited to tasks where rule-based answer verification is possible and does not naturally extend to real-world domains such as chemistry, healthcare, engineering, law, biology, business, and economics. Current practical workarounds use an additional LLM as a model-based verifier; however, this introduces issues such as reliance on a strong verifier LLM, susceptibility to reward hacking, and the practical burden of maintaining the verifier model in memory during training. To address this and extend DeepSeek-R1-Zero-style training to general reasoning domains, we propose a verifier-free method (VeriFree) that bypasses answer verification and instead uses RL to directly maximize the probability of generating the reference answer. We compare VeriFree with verifier-based methods and demonstrate that, in addition to its significant practical benefits and reduced compute requirements, VeriFree matches and even surpasses verifier-based methods on extensive evaluations across MMLU-Pro, GPQA, SuperGPQA, and math-related benchmarks. Moreover, we provide insights into this method from multiple perspectives: as an elegant integration of training both the policy and implicit verifier in a unified model, and as a variational optimization approach. Code is available at this https URL. 

---
# Hardware-Efficient Attention for Fast Decoding 

**Authors**: Ted Zadouri, Hubert Strauss, Tri Dao  

**Link**: [PDF](https://arxiv.org/pdf/2505.21487)  

**Abstract**: LLM decoding is bottlenecked for large batches and long contexts by loading the key-value (KV) cache from high-bandwidth memory, which inflates per-token latency, while the sequential nature of decoding limits parallelism. We analyze the interplay among arithmetic intensity, parallelization, and model quality and question whether current architectures fully exploit modern hardware. This work redesigns attention to perform more computation per byte loaded from memory to maximize hardware efficiency without trading off parallel scalability. We first propose Grouped-Tied Attention (GTA), a simple variant that combines and reuses key and value states, reducing memory transfers without compromising model quality. We then introduce Grouped Latent Attention (GLA), a parallel-friendly latent attention paired with low-level optimizations for fast decoding while maintaining high model quality. Experiments show that GTA matches Grouped-Query Attention (GQA) quality while using roughly half the KV cache and that GLA matches Multi-head Latent Attention (MLA) and is easier to shard. Our optimized GLA kernel is up to 2$\times$ faster than FlashMLA, for example, in a speculative decoding setting when the query length exceeds one. Furthermore, by fetching a smaller KV cache per device, GLA reduces end-to-end latency and increases throughput in online serving benchmarks by up to 2$\times$. 

---
# Mitigating Hallucination in Large Vision-Language Models via Adaptive Attention Calibration 

**Authors**: Mehrdad Fazli, Bowen Wei, Ziwei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21472)  

**Abstract**: Large vision-language models (LVLMs) achieve impressive performance on multimodal tasks but often suffer from hallucination, and confidently describe objects or attributes not present in the image. Current inference-time interventions, while training-free, struggle to maintain accuracy in open-ended and long-form generation scenarios. We introduce the Confidence-Aware Attention Calibration (CAAC) framework to address this challenge by targeting two key biases: spatial perception bias, which distributes attention disproportionately across image tokens, and modality bias, which shifts focus from visual to textual inputs over time. CAAC employs a two-step approach: Visual-Token Calibration (VTC) to balance attention across visual tokens, and Adaptive Attention Re-Scaling (AAR) to reinforce visual grounding based on the model's confidence. This confidence-driven adjustment ensures consistent visual alignment during generation. Experiments on CHAIR, AMBER, and POPE benchmarks demonstrate that CAAC outperforms baselines, particularly in long-form generations, effectively reducing hallucination. 

---
# ID-Align: RoPE-Conscious Position Remapping for Dynamic High-Resolution Adaptation in Vision-Language Models 

**Authors**: Bozhou Li, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21465)  

**Abstract**: Currently, a prevalent approach for enhancing Vision-Language Models (VLMs) performance is to encode both the high-resolution version and the thumbnail of an image simultaneously. While effective, this method generates a large number of image tokens. When combined with the widely used Rotary Position Embedding (RoPE), its long-term decay property hinders the interaction between high-resolution tokens and thumbnail tokens, as well as between text and image. To address these issues, we propose ID-Align, which alleviates these problems by reordering position IDs. In this method, high-resolution tokens inherit IDs from their corresponding thumbnail token while constraining the overexpansion of positional indices. Our experiments conducted within the LLaVA-Next framework demonstrate that ID-Align achieves significant improvements, including a 6.09% enhancement on MMBench's relation reasoning tasks and notable gains across multiple benchmarks. Our code is available at the following link: this https URL. 

---
# The Multilingual Divide and Its Impact on Global AI Safety 

**Authors**: Aidan Peppin, Julia Kreutzer, Alice Schoenauer Sebag, Kelly Marchisio, Beyza Ermis, John Dang, Samuel Cahyawijaya, Shivalika Singh, Seraphina Goldfarb-Tarrant, Viraat Aryabumi, Aakanksha, Wei-Yin Ko, Ahmet Üstün, Matthias Gallé, Marzieh Fadaee, Sara Hooker  

**Link**: [PDF](https://arxiv.org/pdf/2505.21344)  

**Abstract**: Despite advances in large language model capabilities in recent years, a large gap remains in their capabilities and safety performance for many languages beyond a relatively small handful of globally dominant languages. This paper provides researchers, policymakers and governance experts with an overview of key challenges to bridging the "language gap" in AI and minimizing safety risks across languages. We provide an analysis of why the language gap in AI exists and grows, and how it creates disparities in global AI safety. We identify barriers to address these challenges, and recommend how those working in policy and governance can help address safety concerns associated with the language gap by supporting multilingual dataset creation, transparency, and research. 

---
# Something's Fishy In The Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks 

**Authors**: Allaa Boutaleb, Bernd Amann, Hubert Naacke, Rafael Angarita  

**Link**: [PDF](https://arxiv.org/pdf/2505.21329)  

**Abstract**: Recent table representation learning and data discovery methods tackle table union search (TUS) within data lakes, which involves identifying tables that can be unioned with a given query table to enrich its content. These methods are commonly evaluated using benchmarks that aim to assess semantic understanding in real-world TUS tasks. However, our analysis of prominent TUS benchmarks reveals several limitations that allow simple baselines to perform surprisingly well, often outperforming more sophisticated approaches. This suggests that current benchmark scores are heavily influenced by dataset-specific characteristics and fail to effectively isolate the gains from semantic understanding. To address this, we propose essential criteria for future benchmarks to enable a more realistic and reliable evaluation of progress in semantic table union search. 

---
# Optimizing fMRI Data Acquisition for Decoding Natural Speech with Limited Participants 

**Authors**: Louis Jalouzot, Alexis Thual, Yair Lakretz, Christophe Pallier, Bertrand Thirion  

**Link**: [PDF](https://arxiv.org/pdf/2505.21304)  

**Abstract**: We investigate optimal strategies for decoding perceived natural speech from fMRI data acquired from a limited number of participants. Leveraging Lebel et al. (2023)'s dataset of 8 participants, we first demonstrate the effectiveness of training deep neural networks to predict LLM-derived text representations from fMRI activity. Then, in this data regime, we observe that multi-subject training does not improve decoding accuracy compared to single-subject approach. Furthermore, training on similar or different stimuli across subjects has a negligible effect on decoding accuracy. Finally, we find that our decoders better model syntactic than semantic features, and that stories containing sentences with complex syntax or rich semantic content are more challenging to decode. While our results demonstrate the benefits of having extensive data per participant (deep phenotyping), they suggest that leveraging multi-subject for natural speech decoding likely requires deeper phenotyping or a substantially larger cohort. 

---
# Breaking the Ceiling: Exploring the Potential of Jailbreak Attacks through Expanding Strategy Space 

**Authors**: Yao Huang, Yitong Sun, Shouwei Ruan, Yichi Zhang, Yinpeng Dong, Xingxing Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.21277)  

**Abstract**: Large Language Models (LLMs), despite advanced general capabilities, still suffer from numerous safety risks, especially jailbreak attacks that bypass safety protocols. Understanding these vulnerabilities through black-box jailbreak attacks, which better reflect real-world scenarios, offers critical insights into model robustness. While existing methods have shown improvements through various prompt engineering techniques, their success remains limited against safety-aligned models, overlooking a more fundamental problem: the effectiveness is inherently bounded by the predefined strategy spaces. However, expanding this space presents significant challenges in both systematically capturing essential attack patterns and efficiently navigating the increased complexity. To better explore the potential of expanding the strategy space, we address these challenges through a novel framework that decomposes jailbreak strategies into essential components based on the Elaboration Likelihood Model (ELM) theory and develops genetic-based optimization with intention evaluation mechanisms. To be striking, our experiments reveal unprecedented jailbreak capabilities by expanding the strategy space: we achieve over 90% success rate on Claude-3.5 where prior methods completely fail, while demonstrating strong cross-model transferability and surpassing specialized safeguard models in evaluation accuracy. The code is open-sourced at: this https URL. 

---
# PSRB: A Comprehensive Benchmark for Evaluating Persian ASR Systems 

**Authors**: Nima Sedghiyeh, Sara Sadeghi, Reza Khodadadi, Farzin Kashani, Omid Aghdaei, Somayeh Rahimi, Mohammad Sadegh Safari  

**Link**: [PDF](https://arxiv.org/pdf/2505.21230)  

**Abstract**: Although Automatic Speech Recognition (ASR) systems have become an integral part of modern technology, their evaluation remains challenging, particularly for low-resource languages such as Persian. This paper introduces Persian Speech Recognition Benchmark(PSRB), a comprehensive benchmark designed to address this gap by incorporating diverse linguistic and acoustic conditions. We evaluate ten ASR systems, including state-of-the-art commercial and open-source models, to examine performance variations and inherent biases. Additionally, we conduct an in-depth analysis of Persian ASR transcriptions, identifying key error types and proposing a novel metric that weights substitution errors. This metric enhances evaluation robustness by reducing the impact of minor and partial errors, thereby improving the precision of performance assessment. Our findings indicate that while ASR models generally perform well on standard Persian, they struggle with regional accents, children's speech, and specific linguistic challenges. These results highlight the necessity of fine-tuning and incorporating diverse, representative training datasets to mitigate biases and enhance overall ASR performance. PSRB provides a valuable resource for advancing ASR research in Persian and serves as a framework for developing benchmarks in other low-resource languages. A subset of the PSRB dataset is publicly available at this https URL. 

---
# PoisonSwarm: Universal Harmful Information Synthesis via Model Crowdsourcing 

**Authors**: Yu Yan, Sheng Sun, Zhifei Zheng, Ziji Hao, Teli Liu, Min Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21184)  

**Abstract**: To construct responsible and secure AI applications, harmful information data is widely utilized for adversarial testing and the development of safeguards. Existing studies mainly leverage Large Language Models (LLMs) to synthesize data to obtain high-quality task datasets at scale, thereby avoiding costly human annotation. However, limited by the safety alignment mechanisms of LLMs, the synthesis of harmful data still faces challenges in generation reliability and content diversity. In this study, we propose a novel harmful information synthesis framework, PoisonSwarm, which applies the model crowdsourcing strategy to generate diverse harmful data while maintaining a high success rate. Specifically, we generate abundant benign data as the based templates in a counterfactual manner. Subsequently, we decompose each based template into multiple semantic units and perform unit-by-unit toxification and final refinement through dynamic model switching, thus ensuring the success of synthesis. Experimental results demonstrate that PoisonSwarm achieves state-of-the-art performance in synthesizing different categories of harmful data with high scalability and diversity. 

---
# Leveraging GANs for citation intent classification and its impact on citation network analysis 

**Authors**: Davi A. Bezerra, Filipi N. Silva, Diego R. Amancio  

**Link**: [PDF](https://arxiv.org/pdf/2505.21162)  

**Abstract**: Citations play a fundamental role in the scientific ecosystem, serving as a foundation for tracking the flow of knowledge, acknowledging prior work, and assessing scholarly influence. In scientometrics, they are also central to the construction of quantitative indicators. Not all citations, however, serve the same function: some provide background, others introduce methods, or compare results. Therefore, understanding citation intent allows for a more nuanced interpretation of scientific impact. In this paper, we adopted a GAN-based method to classify citation intents. Our results revealed that the proposed method achieves competitive classification performance, closely matching state-of-the-art results with substantially fewer parameters. This demonstrates the effectiveness and efficiency of leveraging GAN architectures combined with contextual embeddings in intent classification task. We also investigated whether filtering citation intents affects the centrality of papers in citation networks. Analyzing the network constructed from the unArXiv dataset, we found that paper rankings can be significantly influenced by citation intent. All four centrality metrics examined- degree, PageRank, closeness, and betweenness - were sensitive to the filtering of citation types. The betweenness centrality displayed the greatest sensitivity, showing substantial changes in ranking when specific citation intents were removed. 

---
# Creativity in LLM-based Multi-Agent Systems: A Survey 

**Authors**: Yi-Cheng Lin, Kang-Chieh Chen, Zhe-Yan Li, Tzu-Heng Wu, Tzu-Hsuan Wu, Kuan-Yu Chen, Hung-yi Lee, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.21116)  

**Abstract**: Large language model (LLM)-driven multi-agent systems (MAS) are transforming how humans and AIs collaboratively generate ideas and artifacts. While existing surveys provide comprehensive overviews of MAS infrastructures, they largely overlook the dimension of \emph{creativity}, including how novel outputs are generated and evaluated, how creativity informs agent personas, and how creative workflows are coordinated. This is the first survey dedicated to creativity in MAS. We focus on text and image generation tasks, and present: (1) a taxonomy of agent proactivity and persona design; (2) an overview of generation techniques, including divergent exploration, iterative refinement, and collaborative synthesis, as well as relevant datasets and evaluation metrics; and (3) a discussion of key challenges, such as inconsistent evaluation standards, insufficient bias mitigation, coordination conflicts, and the lack of unified benchmarks. This survey offers a structured framework and roadmap for advancing the development, evaluation, and standardization of creative MAS. 

---
# Position is Power: System Prompts as a Mechanism of Bias in Large Language Models (LLMs) 

**Authors**: Anna Neumann, Elisabeth Kirsten, Muhammad Bilal Zafar, Jatinder Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.21091)  

**Abstract**: System prompts in Large Language Models (LLMs) are predefined directives that guide model behaviour, taking precedence over user inputs in text processing and generation. LLM deployers increasingly use them to ensure consistent responses across contexts. While model providers set a foundation of system prompts, deployers and third-party developers can append additional prompts without visibility into others' additions, while this layered implementation remains entirely hidden from end-users. As system prompts become more complex, they can directly or indirectly introduce unaccounted for side effects. This lack of transparency raises fundamental questions about how the position of information in different directives shapes model outputs. As such, this work examines how the placement of information affects model behaviour. To this end, we compare how models process demographic information in system versus user prompts across six commercially available LLMs and 50 demographic groups. Our analysis reveals significant biases, manifesting in differences in user representation and decision-making scenarios. Since these variations stem from inaccessible and opaque system-level configurations, they risk representational, allocative and potential other biases and downstream harms beyond the user's ability to detect or correct. Our findings draw attention to these critical issues, which have the potential to perpetuate harms if left unexamined. Further, we argue that system prompt analysis must be incorporated into AI auditing processes, particularly as customisable system prompts become increasingly prevalent in commercial AI deployments. 

---
# Pause Tokens Strictly Increase the Expressivity of Constant-Depth Transformers 

**Authors**: Charles London, Varun Kanade  

**Link**: [PDF](https://arxiv.org/pdf/2505.21024)  

**Abstract**: Pause tokens, simple filler symbols such as "...", consistently improve Transformer performance on both language and mathematical tasks, yet their theoretical effect remains unexplained. We provide the first formal separation result, proving that adding pause tokens to constant-depth, logarithmic-width Transformers strictly increases their computational expressivity. With bounded-precision activations, Transformers without pause tokens compute only a strict subset of $\mathsf{AC}^0$ functions, while adding a polynomial number of pause tokens allows them to express the entire class. For logarithmic-precision Transformers, we show that adding pause tokens achieves expressivity equivalent to $\mathsf{TC}^0$, matching known upper bounds. Empirically, we demonstrate that two-layer causally masked Transformers can learn parity when supplied with pause tokens, a function that they appear unable to learn without them. Our results provide a rigorous theoretical explanation for prior empirical findings, clarify how pause tokens interact with width, depth, and numeric precision, and position them as a distinct mechanism, complementary to chain-of-thought prompting, for enhancing Transformer reasoning. 

---
# RefAV: Towards Planning-Centric Scenario Mining 

**Authors**: Cainan Davidson, Deva Ramanan, Neehar Peri  

**Link**: [PDF](https://arxiv.org/pdf/2505.20981)  

**Abstract**: Autonomous Vehicles (AVs) collect and pseudo-label terabytes of multi-modal data localized to HD maps during normal fleet testing. However, identifying interesting and safety-critical scenarios from uncurated driving logs remains a significant challenge. Traditional scenario mining techniques are error-prone and prohibitively time-consuming, often relying on hand-crafted structured queries. In this work, we revisit spatio-temporal scenario mining through the lens of recent vision-language models (VLMs) to detect whether a described scenario occurs in a driving log and, if so, precisely localize it in both time and space. To address this problem, we introduce RefAV, a large-scale dataset of 10,000 diverse natural language queries that describe complex multi-agent interactions relevant to motion planning derived from 1000 driving logs in the Argoverse 2 Sensor dataset. We evaluate several referential multi-object trackers and present an empirical analysis of our baselines. Notably, we find that naively repurposing off-the-shelf VLMs yields poor performance, suggesting that scenario mining presents unique challenges. Our code and dataset are available at this https URL and this https URL 

---
# Cross from Left to Right Brain: Adaptive Text Dreamer for Vision-and-Language Navigation 

**Authors**: Pingrui Zhang, Yifei Su, Pengyuan Wu, Dong An, Li Zhang, Zhigang Wang, Dong Wang, Yan Ding, Bin Zhao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20897)  

**Abstract**: Vision-and-Language Navigation (VLN) requires the agent to navigate by following natural instructions under partial observability, making it difficult to align perception with language. Recent methods mitigate this by imagining future scenes, yet they rely on vision-based synthesis, leading to high computational cost and redundant details. To this end, we propose to adaptively imagine key environmental semantics via \textit{language} form, enabling a more reliable and efficient strategy. Specifically, we introduce a novel Adaptive Text Dreamer (ATD), a dual-branch self-guided imagination policy built upon a large language model (LLM). ATD is designed with a human-like left-right brain architecture, where the left brain focuses on logical integration, and the right brain is responsible for imaginative prediction of future scenes. To achieve this, we fine-tune only the Q-former within both brains to efficiently activate domain-specific knowledge in the LLM, enabling dynamic updates of logical reasoning and imagination during navigation. Furthermore, we introduce a cross-interaction mechanism to regularize the imagined outputs and inject them into a navigation expert module, allowing ATD to jointly exploit both the reasoning capacity of the LLM and the expertise of the navigation model. We conduct extensive experiments on the R2R benchmark, where ATD achieves state-of-the-art performance with fewer parameters. The code is \href{this https URL}{here}. 

---
# How Do Transformers Learn Variable Binding in Symbolic Programs? 

**Authors**: Yiwei Wu, Atticus Geiger, Raphaël Millière  

**Link**: [PDF](https://arxiv.org/pdf/2505.20896)  

**Abstract**: Variable binding -- the ability to associate variables with values -- is fundamental to symbolic computation and cognition. Although classical architectures typically implement variable binding via addressable memory, it is not well understood how modern neural networks lacking built-in binding operations may acquire this capacity. We investigate this by training a Transformer to dereference queried variables in symbolic programs where variables are assigned either numerical constants or other variables. Each program requires following chains of variable assignments up to four steps deep to find the queried value, and also contains irrelevant chains of assignments acting as distractors. Our analysis reveals a developmental trajectory with three distinct phases during training: (1) random prediction of numerical constants, (2) a shallow heuristic prioritizing early variable assignments, and (3) the emergence of a systematic mechanism for dereferencing assignment chains. Using causal interventions, we find that the model learns to exploit the residual stream as an addressable memory space, with specialized attention heads routing information across token positions. This mechanism allows the model to dynamically track variable bindings across layers, resulting in accurate dereferencing. Our results show how Transformer models can learn to implement systematic variable binding without explicit architectural support, bridging connectionist and symbolic approaches. 

---
# An LLM-as-Judge Metric for Bridging the Gap with Human Evaluation in SE Tasks 

**Authors**: Xin Zhou, Kisub Kim, Ting Zhang, Martin Weyssow, Luis F. Gomes, Guang Yang, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2505.20854)  

**Abstract**: Large Language Models (LLMs) and other automated techniques have been increasingly used to support software developers by generating software artifacts such as code snippets, patches, and comments. However, accurately assessing the correctness of these generated artifacts remains a significant challenge. On one hand, human evaluation provides high accuracy but is labor-intensive and lacks scalability. On the other hand, other existing automatic evaluation metrics are scalable and require minimal human effort, but they often fail to accurately reflect the actual correctness of generated software artifacts.
In this paper, we present SWE-Judge, the first evaluation metric for LLM-as-Ensemble-Judge specifically designed to accurately assess the correctness of generated software artifacts. SWE-Judge first defines five distinct evaluation strategies, each implemented as an independent judge. A dynamic team selection mechanism then identifies the most appropriate subset of judges to produce a final correctness score through ensembling. We evaluate SWE-Judge across a diverse set of software engineering (SE) benchmarks, including CoNaLa, Card2Code, HumanEval-X, APPS, APR-Assess, and Summary-Assess. These benchmarks span three SE tasks: code generation, automated program repair, and code summarization. Experimental results demonstrate that SWE-Judge consistently achieves a higher correlation with human judgments, with improvements ranging from 5.9% to 183.8% over existing automatic metrics. Furthermore, SWE-Judge reaches agreement levels with human annotators that are comparable to inter-annotator agreement in code generation and program repair tasks. These findings underscore SWE-Judge's potential as a scalable and reliable alternative to human evaluation. 

---
# What LLMs Miss in Recommendations: Bridging the Gap with Retrieval-Augmented Collaborative Signals 

**Authors**: Shahrooz Pouryousef  

**Link**: [PDF](https://arxiv.org/pdf/2505.20730)  

**Abstract**: User-item interactions contain rich collaborative signals that form the backbone of many successful recommender systems. While recent work has explored the use of large language models (LLMs) for recommendation, it remains unclear whether LLMs can effectively reason over this type of collaborative information. In this paper, we conduct a systematic comparison between LLMs and classical matrix factorization (MF) models to assess LLMs' ability to leverage user-item interaction data. We further introduce a simple retrieval-augmented generation (RAG) method that enhances LLMs by grounding their predictions in structured interaction data. Our experiments reveal that current LLMs often fall short in capturing collaborative patterns inherent to MF models, but that our RAG-based approach substantially improves recommendation quality-highlighting a promising direction for future LLM-based recommenders. 

---
# MUSEG: Reinforcing Video Temporal Understanding via Timestamp-Aware Multi-Segment Grounding 

**Authors**: Fuwen Luo, Shengfeng Lou, Chi Chen, Ziyue Wang, Chenliang Li, Weizhou Shen, Jiyue Guo, Peng Li, Ming Yan, Ji Zhang, Fei Huang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20715)  

**Abstract**: Video temporal understanding is crucial for multimodal large language models (MLLMs) to reason over events in videos. Despite recent advances in general video understanding, current MLLMs still struggle with fine-grained temporal reasoning. While reinforcement learning (RL) has been explored to address this issue recently, existing RL approaches remain limited in effectiveness. In this work, we propose MUSEG, a novel RL-based method that enhances temporal understanding by introducing timestamp-aware multi-segment grounding. MUSEG enables MLLMs to align queries with multiple relevant video segments, promoting more comprehensive temporal reasoning. To facilitate effective learning, we design a customized RL training recipe with phased rewards that progressively guides the model toward temporally grounded reasoning. Extensive experiments on temporal grounding and time-sensitive video QA tasks demonstrate that MUSEG significantly outperforms existing methods and generalizes well across diverse temporal understanding scenarios. View our project at this https URL. 

---
# Can we Debias Social Stereotypes in AI-Generated Images? Examining Text-to-Image Outputs and User Perceptions 

**Authors**: Saharsh Barve, Andy Mao, Jiayue Melissa Shi, Prerna Juneja, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2505.20692)  

**Abstract**: Recent advances in generative AI have enabled visual content creation through text-to-image (T2I) generation. However, despite their creative potential, T2I models often replicate and amplify societal stereotypes -- particularly those related to gender, race, and culture -- raising important ethical concerns. This paper proposes a theory-driven bias detection rubric and a Social Stereotype Index (SSI) to systematically evaluate social biases in T2I outputs. We audited three major T2I model outputs -- DALL-E-3, Midjourney-6.1, and Stability AI Core -- using 100 queries across three categories -- geocultural, occupational, and adjectival. Our analysis reveals that initial outputs are prone to include stereotypical visual cues, including gendered professions, cultural markers, and western beauty norms. To address this, we adopted our rubric to conduct targeted prompt refinement using LLMs, which significantly reduced bias -- SSI dropped by 61% for geocultural, 69% for occupational, and 51% for adjectival queries. We complemented our quantitative analysis through a user study examining perceptions, awareness, and preferences around AI-generated biased imagery. Our findings reveal a key tension -- although prompt refinement can mitigate stereotypes, it can limit contextual alignment. Interestingly, users often perceived stereotypical images to be more aligned with their expectations. We discuss the need to balance ethical debiasing with contextual relevance and call for T2I systems that support global diversity and inclusivity while not compromising the reflection of real-world social complexity. 

---
# TeroSeek: An AI-Powered Knowledge Base and Retrieval Generation Platform for Terpenoid Research 

**Authors**: Xu Kang, Siqi Jiang, Kangwei Xu, Jiahao Li, Ruibo Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20663)  

**Abstract**: Terpenoids are a crucial class of natural products that have been studied for over 150 years, but their interdisciplinary nature (spanning chemistry, pharmacology, and biology) complicates knowledge integration. To address this, the authors developed TeroSeek, a curated knowledge base (KB) built from two decades of terpenoid literature, coupled with an AI-powered question-answering chatbot and web service. Leveraging a retrieval-augmented generation (RAG) framework, TeroSeek provides structured, high-quality information and outperforms general-purpose large language models (LLMs) in terpenoid-related queries. It serves as a domain-specific expert tool for multidisciplinary research and is publicly available at this http URL. 

---
# SV-TrustEval-C: Evaluating Structure and Semantic Reasoning in Large Language Models for Source Code Vulnerability Analysis 

**Authors**: Yansong Li, Paula Branco, Alexander M. Hoole, Manish Marwah, Hari Manassery Koduvely, Guy-Vincent Jourdan, Stephan Jou  

**Link**: [PDF](https://arxiv.org/pdf/2505.20630)  

**Abstract**: As Large Language Models (LLMs) evolve in understanding and generating code, accurately evaluating their reliability in analyzing source code vulnerabilities becomes increasingly vital. While studies have examined LLM capabilities in tasks like vulnerability detection and repair, they often overlook the importance of both structure and semantic reasoning crucial for trustworthy vulnerability analysis. To address this gap, we introduce SV-TrustEval-C, a benchmark designed to evaluate LLMs' abilities for vulnerability analysis of code written in the C programming language through two key dimensions: structure reasoning - assessing how models identify relationships between code elements under varying data and control flow complexities; and semantic reasoning - examining their logical consistency in scenarios where code is structurally and semantically perturbed. Our results show that current LLMs are far from satisfactory in understanding complex code relationships and that their vulnerability analyses rely more on pattern matching than on robust logical reasoning. These findings underscore the effectiveness of the SV-TrustEval-C benchmark and highlight critical areas for enhancing the reasoning capabilities and trustworthiness of LLMs in real-world vulnerability analysis tasks. Our initial benchmark dataset is publicly available. 

---
# Roboflow100-VL: A Multi-Domain Object Detection Benchmark for Vision-Language Models 

**Authors**: Peter Robicheaux, Matvei Popov, Anish Madan, Isaac Robinson, Joseph Nelson, Deva Ramanan, Neehar Peri  

**Link**: [PDF](https://arxiv.org/pdf/2505.20612)  

**Abstract**: Vision-language models (VLMs) trained on internet-scale data achieve remarkable zero-shot detection performance on common objects like car, truck, and pedestrian. However, state-of-the-art models still struggle to generalize to out-of-distribution classes, tasks and imaging modalities not typically found in their pre-training. Rather than simply re-training VLMs on more visual data, we argue that one should align VLMs to new concepts with annotation instructions containing a few visual examples and rich textual descriptions. To this end, we introduce Roboflow100-VL, a large-scale collection of 100 multi-modal object detection datasets with diverse concepts not commonly found in VLM pre-training. We evaluate state-of-the-art models on our benchmark in zero-shot, few-shot, semi-supervised, and fully-supervised settings, allowing for comparison across data regimes. Notably, we find that VLMs like GroundingDINO and Qwen2.5-VL achieve less than 2% zero-shot accuracy on challenging medical imaging datasets within Roboflow100-VL, demonstrating the need for few-shot concept alignment. Our code and dataset are available at this https URL and this https URL 

---
# Comparisons between a Large Language Model-based Real-Time Compound Diagnostic Medical AI Interface and Physicians for Common Internal Medicine Cases using Simulated Patients 

**Authors**: Hyungjun Park, Chang-Yun Woo, Seungjo Lim, Seunghwan Lim, Keunho Kwak, Ju Young Jeong, Chong Hyun Suh  

**Link**: [PDF](https://arxiv.org/pdf/2505.20609)  

**Abstract**: Objective To develop an LLM based realtime compound diagnostic medical AI interface and performed a clinical trial comparing this interface and physicians for common internal medicine cases based on the United States Medical License Exam (USMLE) Step 2 Clinical Skill (CS) style exams. Methods A nonrandomized clinical trial was conducted on August 20, 2024. We recruited one general physician, two internal medicine residents (2nd and 3rd year), and five simulated patients. The clinical vignettes were adapted from the USMLE Step 2 CS style exams. We developed 10 representative internal medicine cases based on actual patients and included information available on initial diagnostic evaluation. Primary outcome was the accuracy of the first differential diagnosis. Repeatability was evaluated based on the proportion of agreement. Results The accuracy of the physicians' first differential diagnosis ranged from 50% to 70%, whereas the realtime compound diagnostic medical AI interface achieved an accuracy of 80%. The proportion of agreement for the first differential diagnosis was 0.7. The accuracy of the first and second differential diagnoses ranged from 70% to 90% for physicians, whereas the AI interface achieved an accuracy rate of 100%. The average time for the AI interface (557 sec) was 44.6% shorter than that of the physicians (1006 sec). The AI interface ($0.08) also reduced costs by 98.1% compared to the physicians' average ($4.2). Patient satisfaction scores ranged from 4.2 to 4.3 for care by physicians and were 3.9 for the AI interface Conclusion An LLM based realtime compound diagnostic medical AI interface demonstrated diagnostic accuracy and patient satisfaction comparable to those of a physician, while requiring less time and lower costs. These findings suggest that AI interfaces may have the potential to assist primary care consultations for common internal medicine cases. 

---
# Beyond Markovian: Reflective Exploration via Bayes-Adaptive RL for LLM Reasoning 

**Authors**: Shenao Zhang, Yaqing Wang, Yinxiao Liu, Tianqi Liu, Peter Grabowski, Eugene Ie, Zhaoran Wang, Yunxuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20561)  

**Abstract**: Large Language Models (LLMs) trained via Reinforcement Learning (RL) have exhibited strong reasoning capabilities and emergent reflective behaviors, such as backtracking and error correction. However, conventional Markovian RL confines exploration to the training phase to learn an optimal deterministic policy and depends on the history contexts only through the current state. Therefore, it remains unclear whether reflective reasoning will emerge during Markovian RL training, or why they are beneficial at test time. To remedy this, we recast reflective exploration within the Bayes-Adaptive RL framework, which explicitly optimizes the expected return under a posterior distribution over Markov decision processes. This Bayesian formulation inherently incentivizes both reward-maximizing exploitation and information-gathering exploration via belief updates. Our resulting algorithm, BARL, instructs the LLM to stitch and switch strategies based on the observed outcomes, offering principled guidance on when and how the model should reflectively explore. Empirical results on both synthetic and mathematical reasoning tasks demonstrate that BARL outperforms standard Markovian RL approaches at test time, achieving superior token efficiency with improved exploration effectiveness. Our code is available at this https URL. 

---
# Scaling over Scaling: Exploring Test-Time Scaling Pareto in Large Reasoning Models 

**Authors**: Jian Wang, Boyan Zhu, Chak Tou Leong, Yongqi Li, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20522)  

**Abstract**: Large reasoning models (LRMs) have exhibited the capacity of enhancing reasoning performance via internal test-time scaling. Building upon this, a promising direction is to further scale test-time compute to unlock even greater reasoning capabilities. However, as we push these scaling boundaries, systematically understanding the practical limits and achieving optimal resource allocation becomes a critical challenge. In this paper, we investigate the scaling Pareto of test-time scaling and introduce the Test-Time Scaling Performance Model (TTSPM). We theoretically analyze two fundamental paradigms for such extended scaling, parallel scaling and sequential scaling, from a probabilistic modeling perspective. Our primary contribution is the derivation of the saturation point on the scaling budget for both strategies, identifying thresholds beyond which additional computation yields diminishing returns. Remarkably, despite their distinct mechanisms, both paradigms converge to a unified mathematical structure in their upper bounds. We empirically validate our theoretical findings on challenging reasoning benchmarks, including AIME, MATH-500, and GPQA, demonstrating the practical utility of these bounds for test-time resource allocation. We hope that this work provides insights into the cost-benefit trade-offs of test-time scaling, guiding the development of more resource-efficient inference strategies for large reasoning models. 

---
# Project Riley: Multimodal Multi-Agent LLM Collaboration with Emotional Reasoning and Voting 

**Authors**: Ana Rita Ortigoso, Gabriel Vieira, Daniel Fuentes, Luis Frazão, Nuno Costa, António Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2505.20521)  

**Abstract**: This paper presents Project Riley, a novel multimodal and multi-model conversational AI architecture oriented towards the simulation of reasoning influenced by emotional states. Drawing inspiration from Pixar's Inside Out, the system comprises five distinct emotional agents - Joy, Sadness, Fear, Anger, and Disgust - that engage in structured multi-round dialogues to generate, criticise, and iteratively refine responses. A final reasoning mechanism synthesises the contributions of these agents into a coherent output that either reflects the dominant emotion or integrates multiple perspectives. The architecture incorporates both textual and visual large language models (LLMs), alongside advanced reasoning and self-refinement processes. A functional prototype was deployed locally in an offline environment, optimised for emotional expressiveness and computational efficiency. From this initial prototype, another one emerged, called Armando, which was developed for use in emergency contexts, delivering emotionally calibrated and factually accurate information through the integration of Retrieval-Augmented Generation (RAG) and cumulative context tracking. The Project Riley prototype was evaluated through user testing, in which participants interacted with the chatbot and completed a structured questionnaire assessing three dimensions: Emotional Appropriateness, Clarity and Utility, and Naturalness and Human-likeness. The results indicate strong performance in structured scenarios, particularly with respect to emotional alignment and communicative clarity. 

---
# Embodied AI with Foundation Models for Mobile Service Robots: A Systematic Review 

**Authors**: Matthew Lisondra, Beno Benhabib, Goldie Nejat  

**Link**: [PDF](https://arxiv.org/pdf/2505.20503)  

**Abstract**: Rapid advancements in foundation models, including Large Language Models, Vision-Language Models, Multimodal Large Language Models, and Vision-Language-Action Models have opened new avenues for embodied AI in mobile service robotics. By combining foundation models with the principles of embodied AI, where intelligent systems perceive, reason, and act through physical interactions, robots can improve understanding, adapt to, and execute complex tasks in dynamic real-world environments. However, embodied AI in mobile service robots continues to face key challenges, including multimodal sensor fusion, real-time decision-making under uncertainty, task generalization, and effective human-robot interactions (HRI). In this paper, we present the first systematic review of the integration of foundation models in mobile service robotics, identifying key open challenges in embodied AI and examining how foundation models can address them. Namely, we explore the role of such models in enabling real-time sensor fusion, language-conditioned control, and adaptive task execution. Furthermore, we discuss real-world applications in the domestic assistance, healthcare, and service automation sectors, demonstrating the transformative impact of foundation models on service robotics. We also include potential future research directions, emphasizing the need for predictive scaling laws, autonomous long-term adaptation, and cross-embodiment generalization to enable scalable, efficient, and robust deployment of foundation models in human-centric robotic systems. 

---
# BrainStratify: Coarse-to-Fine Disentanglement of Intracranial Neural Dynamics 

**Authors**: Hui Zheng, Hai-Teng Wang, Yi-Tao Jing, Pei-Yang Lin, Han-Qing Zhao, Wei Chen, Peng-Hu Wei, Yong-Zhi Shan, Guo-Guang Zhao, Yun-Zhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20480)  

**Abstract**: Decoding speech directly from neural activity is a central goal in brain-computer interface (BCI) research. In recent years, exciting advances have been made through the growing use of intracranial field potential recordings, such as stereo-ElectroEncephaloGraphy (sEEG) and ElectroCorticoGraphy (ECoG). These neural signals capture rich population-level activity but present key challenges: (i) task-relevant neural signals are sparsely distributed across sEEG electrodes, and (ii) they are often entangled with task-irrelevant neural signals in both sEEG and ECoG. To address these challenges, we introduce a unified Coarse-to-Fine neural disentanglement framework, BrainStratify, which includes (i) identifying functional groups through spatial-context-guided temporal-spatial modeling, and (ii) disentangling distinct neural dynamics within the target functional group using Decoupled Product Quantization (DPQ). We evaluate BrainStratify on two open-source sEEG datasets and one (epidural) ECoG dataset, spanning tasks like vocal production and speech perception. Extensive experiments show that BrainStratify, as a unified framework for decoding speech from intracranial neural signals, significantly outperforms previous decoding methods. Overall, by combining data-driven stratification with neuroscience-inspired modularity, BrainStratify offers a robust and interpretable solution for speech decoding from intracranial recordings. 

---
# The Impact of a Chatbot's Ephemerality-Framing on Self-Disclosure Perceptions 

**Authors**: Samuel Rhys Cox, Rune Møberg Jacobsen, Niels van Berkel  

**Link**: [PDF](https://arxiv.org/pdf/2505.20464)  

**Abstract**: Self-disclosure, the sharing of one's thoughts and feelings, is affected by the perceived relationship between individuals. While chatbots are increasingly used for self-disclosure, the impact of a chatbot's framing on users' self-disclosure remains under-explored. We investigated how a chatbot's description of its relationship with users, particularly in terms of ephemerality, affects self-disclosure. Specifically, we compared a Familiar chatbot, presenting itself as a companion remembering past interactions, with a Stranger chatbot, presenting itself as a new, unacquainted entity in each conversation. In a mixed factorial design, participants engaged with either the Familiar or Stranger chatbot in two sessions across two days, with one conversation focusing on Emotional- and another Factual-disclosure. When Emotional-disclosure was sought in the first chatting session, Stranger-condition participants felt more comfortable self-disclosing. However, when Factual-disclosure was sought first, these differences were replaced by more enjoyment among Familiar-condition participants. Qualitative findings showed Stranger afforded anonymity and reduced judgement, whereas Familiar sometimes felt intrusive unless rapport was built via low-risk Factual-disclosure. 

---
# SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents 

**Authors**: Ibragim Badertdinov, Alexander Golubev, Maksim Nekrashevich, Anton Shevtsov, Simon Karasik, Andrei Andriushchenko, Maria Trofimova, Daria Litvintseva, Boris Yangel  

**Link**: [PDF](https://arxiv.org/pdf/2505.20411)  

**Abstract**: LLM-based agents have shown promising capabilities in a growing range of software engineering (SWE) tasks. However, advancing this field faces two critical challenges. First, high-quality training data is scarce, especially data that reflects real-world SWE scenarios, where agents must interact with development environments, execute code and adapt behavior based on the outcomes of their actions. Existing datasets are either limited to one-shot code generation or comprise small, manually curated collections of interactive tasks, lacking both scale and diversity. Second, the lack of fresh interactive SWE tasks affects evaluation of rapidly improving models, as static benchmarks quickly become outdated due to contamination issues. To address these limitations, we introduce a novel, automated, and scalable pipeline to continuously extract real-world interactive SWE tasks from diverse GitHub repositories. Using this pipeline, we construct SWE-rebench, a public dataset comprising over 21,000 interactive Python-based SWE tasks, suitable for reinforcement learning of SWE agents at scale. Additionally, we use continuous supply of fresh tasks collected using SWE-rebench methodology to build a contamination-free benchmark for agentic software engineering. We compare results of various LLMs on this benchmark to results on SWE-bench Verified and show that performance of some language models might be inflated due to contamination issues. 

---
# What Changed? Detecting and Evaluating Instruction-Guided Image Edits with Multimodal Large Language Models 

**Authors**: Lorenzo Baraldi, Davide Bucciarelli, Federico Betti, Marcella Cornia, Lorenzo Baraldi, Nicu Sebe, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2505.20405)  

**Abstract**: Instruction-based image editing models offer increased personalization opportunities in generative tasks. However, properly evaluating their results is challenging, and most of the existing metrics lag in terms of alignment with human judgment and explainability. To tackle these issues, we introduce DICE (DIfference Coherence Estimator), a model designed to detect localized differences between the original and the edited image and to assess their relevance to the given modification request. DICE consists of two key components: a difference detector and a coherence estimator, both built on an autoregressive Multimodal Large Language Model (MLLM) and trained using a strategy that leverages self-supervision, distillation from inpainting networks, and full supervision. Through extensive experiments, we evaluate each stage of our pipeline, comparing different MLLMs within the proposed framework. We demonstrate that DICE effectively identifies coherent edits, effectively evaluating images generated by different editing models with a strong correlation with human judgment. We publicly release our source code, models, and data. 

---
# Hierarchical Retrieval with Evidence Curation for Open-Domain Financial Question Answering on Standardized Documents 

**Authors**: Jaeyoung Choe, Jihoon Kim, Woohwan Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.20368)  

**Abstract**: Retrieval-augmented generation (RAG) based large language models (LLMs) are widely used in finance for their excellent performance on knowledge-intensive tasks. However, standardized documents (e.g., SEC filing) share similar formats such as repetitive boilerplate texts, and similar table structures. This similarity forces traditional RAG methods to misidentify near-duplicate text, leading to duplicate retrieval that undermines accuracy and completeness. To address these issues, we propose the Hierarchical Retrieval with Evidence Curation (HiREC) framework. Our approach first performs hierarchical retrieval to reduce confusion among similar texts. It first retrieve related documents and then selects the most relevant passages from the documents. The evidence curation process removes irrelevant passages. When necessary, it automatically generates complementary queries to collect missing information. To evaluate our approach, we construct and release a Large-scale Open-domain Financial (LOFin) question answering benchmark that includes 145,897 SEC documents and 1,595 question-answer pairs. Our code and data are available at this https URL. 

---
# Towards Emotionally Consistent Text-Based Speech Editing: Introducing EmoCorrector and The ECD-TSE Dataset 

**Authors**: Rui Liu, Pu Gao, Jiatian Xi, Berrak Sisman, Carlos Busso, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20341)  

**Abstract**: Text-based speech editing (TSE) modifies speech using only text, eliminating re-recording. However, existing TSE methods, mainly focus on the content accuracy and acoustic consistency of synthetic speech segments, and often overlook the emotional shifts or inconsistency issues introduced by text changes. To address this issue, we propose EmoCorrector, a novel post-correction scheme for TSE. EmoCorrector leverages Retrieval-Augmented Generation (RAG) by extracting the edited text's emotional features, retrieving speech samples with matching emotions, and synthesizing speech that aligns with the desired emotion while preserving the speaker's identity and quality. To support the training and evaluation of emotional consistency modeling in TSE, we pioneer the benchmarking Emotion Correction Dataset for TSE (ECD-TSE). The prominent aspect of ECD-TSE is its inclusion of $<$text, speech$>$ paired data featuring diverse text variations and a range of emotional expressions. Subjective and objective experiments and comprehensive analysis on ECD-TSE confirm that EmoCorrector significantly enhances the expression of intended emotion while addressing emotion inconsistency limitations in current TSE methods. Code and audio examples are available at this https URL. 

---
# Cultural Awareness in Vision-Language Models: A Cross-Country Exploration 

**Authors**: Avinash Madasu, Vasudev Lal, Phillip Howard  

**Link**: [PDF](https://arxiv.org/pdf/2505.20326)  

**Abstract**: Vision-Language Models (VLMs) are increasingly deployed in diverse cultural contexts, yet their internal biases remain poorly understood. In this work, we propose a novel framework to systematically evaluate how VLMs encode cultural differences and biases related to race, gender, and physical traits across countries. We introduce three retrieval-based tasks: (1) Race to Country retrieval, which examines the association between individuals from specific racial groups (East Asian, White, Middle Eastern, Latino, South Asian, and Black) and different countries; (2) Personal Traits to Country retrieval, where images are paired with trait-based prompts (e.g., Smart, Honest, Criminal, Violent) to investigate potential stereotypical associations; and (3) Physical Characteristics to Country retrieval, focusing on visual attributes like skinny, young, obese, and old to explore how physical appearances are culturally linked to nations. Our findings reveal persistent biases in VLMs, highlighting how visual representations may inadvertently reinforce societal stereotypes. 

---
# InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning 

**Authors**: Zifu Wan, Yaqi Xie, Ce Zhang, Zhiqiu Lin, Zihan Wang, Simon Stepputtis, Deva Ramanan, Katia Sycara  

**Link**: [PDF](https://arxiv.org/pdf/2505.18291)  

**Abstract**: Large multimodal foundation models, particularly in the domains of language and vision, have significantly advanced various tasks, including robotics, autonomous driving, information retrieval, and grounding. However, many of these models perceive objects as indivisible, overlooking the components that constitute them. Understanding these components and their associated affordances provides valuable insights into an object's functionality, which is fundamental for performing a wide range of tasks. In this work, we introduce a novel real-world benchmark, InstructPart, comprising hand-labeled part segmentation annotations and task-oriented instructions to evaluate the performance of current models in understanding and executing part-level tasks within everyday contexts. Through our experiments, we demonstrate that task-oriented part segmentation remains a challenging problem, even for state-of-the-art Vision-Language Models (VLMs). In addition to our benchmark, we introduce a simple baseline that achieves a twofold performance improvement through fine-tuning with our dataset. With our dataset and benchmark, we aim to facilitate research on task-oriented part segmentation and enhance the applicability of VLMs across various domains, including robotics, virtual reality, information retrieval, and other related fields. Project website: this https URL. 

---
