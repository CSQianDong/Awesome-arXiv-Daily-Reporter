# Tracing LLM Reasoning Processes with Strategic Games: A Framework for Planning, Revision, and Resource-Constrained Decision Making 

**Authors**: Xiaopeng Yuan, Xingjian Zhang, Ke Xu, Yifan Xu, Lijun Yu, Jindong Wang, Yushun Dong, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12012)  

**Abstract**: Large language models (LLMs) are increasingly used for tasks that require complex reasoning. Most benchmarks focus on final outcomes but overlook the intermediate reasoning steps - such as planning, revision, and decision making under resource constraints. We argue that measuring these internal processes is essential for understanding model behavior and improving reliability. We propose using strategic games as a natural evaluation environment: closed, rule-based systems with clear states, limited resources, and automatic feedback. We introduce a framework that evaluates LLMs along three core dimensions: planning, revision, and resource-constrained decision making. To operationalize this, we define metrics beyond win rate, including overcorrection risk rate, correction success rate, improvement slope, and over-budget ratio. In 4320 adversarial rounds across 12 leading models, ChatGPT-o3-mini achieves the top composite score, with a win rate of 74.7 percent, a correction success rate of 78.6 percent, and an improvement slope of 0.041. By contrast, Qwen-Plus, despite an overcorrection risk rate of 81.6 percent, wins only 25.6 percent of its matches - primarily due to excessive resource use. We also observe a negative correlation between overcorrection risk rate and correction success rate (Pearson r = -0.51, p = 0.093), suggesting that more frequent edits do not always improve outcomes. Our findings highlight the value of assessing not only what LLMs decide but how they arrive at those decisions 

---
# Schema-R1: A reasoning training approach for schema linking in Text-to-SQL Task 

**Authors**: Wuzhenghong Wen, Su Pan, yuwei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.11986)  

**Abstract**: Schema linking is a critical step in Text-to-SQL task, aiming to accurately predict the table names and column names required for the SQL query based on the given question. However, current fine-tuning approaches for schema linking models employ a rote-learning paradigm, excessively optimizing for ground truth schema linking outcomes while compromising reasoning ability. This limitation arises because of the difficulty in acquiring a high-quality reasoning sample for downstream tasks. To address this, we propose Schema-R1, a reasoning schema linking model trained using reinforcement learning. Specifically, Schema-R1 consists of three key steps: constructing small batches of high-quality reasoning samples, supervised fine-tuning for cold-start initialization, and rule-based reinforcement learning training. The final results demonstrate that our method effectively enhances the reasoning ability of the schema linking model, achieving a 10\% improvement in filter accuracy compared to the existing method. Our code is available at this https URL. 

---
# Towards a Cascaded LLM Framework for Cost-effective Human-AI Decision-Making 

**Authors**: Claudio Fanconi, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2506.11887)  

**Abstract**: Effective human-AI decision-making balances three key factors: the \textit{correctness} of predictions, the \textit{cost} of knowledge and reasoning complexity, and the confidence about whether to \textit{abstain} automated answers or involve human experts. In this work, we present a cascaded LLM decision framework that adaptively delegates tasks across multiple tiers of expertise -- a base model for initial candidate answers, a more capable and knowledgeable (but costlier) large model, and a human expert for when the model cascade abstains. Our method proceeds in two stages. First, a deferral policy determines whether to accept the base model's answer or regenerate it with the large model based on the confidence score. Second, an abstention policy decides whether the cascade model response is sufficiently certain or requires human intervention. Moreover, we incorporate an online learning mechanism in the framework that can leverage human feedback to improve decision quality over time. We demonstrate this approach to general question-answering (ARC-Easy and ARC-Challenge) and medical question-answering (MedQA and MedMCQA). Our results show that our cascaded strategy outperforms in most cases single-model baselines in accuracy while reducing cost and providing a principled way to handle abstentions. 

---
# Addressing Bias in LLMs: Strategies and Application to Fair AI-based Recruitment 

**Authors**: Alejandro Peña, Julian Fierrez, Aythami Morales, Gonzalo Mancera, Miguel Lopez, Ruben Tolosana  

**Link**: [PDF](https://arxiv.org/pdf/2506.11880)  

**Abstract**: The use of language technologies in high-stake settings is increasing in recent years, mostly motivated by the success of Large Language Models (LLMs). However, despite the great performance of LLMs, they are are susceptible to ethical concerns, such as demographic biases, accountability, or privacy. This work seeks to analyze the capacity of Transformers-based systems to learn demographic biases present in the data, using a case study on AI-based automated recruitment. We propose a privacy-enhancing framework to reduce gender information from the learning pipeline as a way to mitigate biased behaviors in the final tools. Our experiments analyze the influence of data biases on systems built on two different LLMs, and how the proposed framework effectively prevents trained systems from reproducing the bias in the data. 

---
# Revealing Political Bias in LLMs through Structured Multi-Agent Debate 

**Authors**: Aishwarya Bandaru, Fabian Bindley, Trevor Bluth, Nandini Chavda, Baixu Chen, Ethan Law  

**Link**: [PDF](https://arxiv.org/pdf/2506.11825)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate social behaviour, yet their political biases and interaction dynamics in debates remain underexplored. We investigate how LLM type and agent gender attributes influence political bias using a structured multi-agent debate framework, by engaging Neutral, Republican, and Democrat American LLM agents in debates on politically sensitive topics. We systematically vary the underlying LLMs, agent genders, and debate formats to examine how model provenance and agent personas influence political bias and attitudes throughout debates. We find that Neutral agents consistently align with Democrats, while Republicans shift closer to the Neutral; gender influences agent attitudes, with agents adapting their opinions when aware of other agents' genders; and contrary to prior research, agents with shared political affiliations can form echo chambers, exhibiting the expected intensification of attitudes as debates progress. 

---
# On the Performance of LLMs for Real Estate Appraisal 

**Authors**: Margot Geerts, Manon Reusens, Bart Baesens, Seppe vanden Broucke, Jochen De Weerdt  

**Link**: [PDF](https://arxiv.org/pdf/2506.11812)  

**Abstract**: The real estate market is vital to global economies but suffers from significant information asymmetry. This study examines how Large Language Models (LLMs) can democratize access to real estate insights by generating competitive and interpretable house price estimates through optimized In-Context Learning (ICL) strategies. We systematically evaluate leading LLMs on diverse international housing datasets, comparing zero-shot, few-shot, market report-enhanced, and hybrid prompting techniques. Our results show that LLMs effectively leverage hedonic variables, such as property size and amenities, to produce meaningful estimates. While traditional machine learning models remain strong for pure predictive accuracy, LLMs offer a more accessible, interactive and interpretable alternative. Although self-explanations require cautious interpretation, we find that LLMs explain their predictions in agreement with state-of-the-art models, confirming their trustworthiness. Carefully selected in-context examples based on feature similarity and geographic proximity, significantly enhance LLM performance, yet LLMs struggle with overconfidence in price intervals and limited spatial reasoning. We offer practical guidance for structured prediction tasks through prompt optimization. Our findings highlight LLMs' potential to improve transparency in real estate appraisal and provide actionable insights for stakeholders. 

---
# Causal Effect Identification in Heterogeneous Environments from Higher-Order Moments 

**Authors**: Yaroslav Kivva, Sina Akbari, Saber Salehkaleybar, Negar Kiyavash  

**Link**: [PDF](https://arxiv.org/pdf/2506.11756)  

**Abstract**: We investigate the estimation of the causal effect of a treatment variable on an outcome in the presence of a latent confounder. We first show that the causal effect is identifiable under certain conditions when data is available from multiple environments, provided that the target causal effect remains invariant across these environments. Secondly, we propose a moment-based algorithm for estimating the causal effect as long as only a single parameter of the data-generating mechanism varies across environments -- whether it be the exogenous noise distribution or the causal relationship between two variables. Conversely, we prove that identifiability is lost if both exogenous noise distributions of both the latent and treatment variables vary across environments. Finally, we propose a procedure to identify which parameter of the data-generating mechanism has varied across the environments and evaluate the performance of our proposed methods through experiments on synthetic data. 

---
# Relational GNNs Cannot Learn $C_2$ Features for Planning 

**Authors**: Dillon Z. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11721)  

**Abstract**: Relational Graph Neural Networks (R-GNNs) are a GNN-based approach for learning value functions that can generalise to unseen problems from a given planning domain. R-GNNs were theoretically motivated by the well known connection between the expressive power of GNNs and $C_2$, first-order logic with two variables and counting. In the context of planning, $C_2$ features refer to the set of formulae in $C_2$ with relations defined by the unary and binary predicates of a planning domain. Some planning domains exhibit optimal value functions that can be decomposed as arithmetic expressions of $C_2$ features. We show that, contrary to empirical results, R-GNNs cannot learn value functions defined by $C_2$ features. We also identify prior GNN architectures for planning that may better learn value functions defined by $C_2$ features. 

---
# Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization 

**Authors**: Wenqi Liu, Xuemeng Song, Jiaxi Li, Yinwei Wei, Na Zheng, Jianhua Yin, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.11712)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as an effective approach for mitigating hallucination in Multimodal Large Language Models (MLLMs). Although existing methods have achieved significant progress by utilizing vision-oriented contrastive objectives for enhancing MLLMs' attention to visual inputs and hence reducing hallucination, they suffer from non-rigorous optimization objective function and indirect preference supervision. To address these limitations, we propose a Symmetric Multimodal Preference Optimization (SymMPO), which conducts symmetric preference learning with direct preference supervision (i.e., response pairs) for visual understanding enhancement, while maintaining rigorous theoretical alignment with standard DPO. In addition to conventional ordinal preference learning, SymMPO introduces a preference margin consistency loss to quantitatively regulate the preference gap between symmetric preference pairs. Comprehensive evaluation across five benchmarks demonstrate SymMPO's superior performance, validating its effectiveness in hallucination mitigation of MLLMs. 

---
# VLM@school -- Evaluation of AI image understanding on German middle school knowledge 

**Authors**: René Peinl, Vincent Tischler  

**Link**: [PDF](https://arxiv.org/pdf/2506.11604)  

**Abstract**: This paper introduces a novel benchmark dataset designed to evaluate the capabilities of Vision Language Models (VLMs) on tasks that combine visual reasoning with subject-specific background knowledge in the German language. In contrast to widely used English-language benchmarks that often rely on artificially difficult or decontextualized problems, this dataset draws from real middle school curricula across nine domains including mathematics, history, biology, and religion. The benchmark includes over 2,000 open-ended questions grounded in 486 images, ensuring that models must integrate visual interpretation with factual reasoning rather than rely on superficial textual cues. We evaluate thirteen state-of-the-art open-weight VLMs across multiple dimensions, including domain-specific accuracy and performance on adversarial crafted questions. Our findings reveal that even the strongest models achieve less than 45% overall accuracy, with particularly poor performance in music, mathematics, and adversarial settings. Furthermore, the results indicate significant discrepancies between success on popular benchmarks and real-world multimodal understanding. We conclude that middle school-level tasks offer a meaningful and underutilized avenue for stress-testing VLMs, especially in non-English contexts. The dataset and evaluation protocol serve as a rigorous testbed to better understand and improve the visual and linguistic reasoning capabilities of future AI systems. 

---
# Collaborative LLM Inference via Planning for Efficient Reasoning 

**Authors**: Byeongchan Lee, Jonghoon Lee, Dongyoung Kim, Jaehyung Kim, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11578)  

**Abstract**: Large language models (LLMs) excel at complex reasoning tasks, but those with strong capabilities (e.g., whose numbers of parameters are larger than 100B) are often accessible only through paid APIs, making them too costly for applications of frequent use. In contrast, smaller open-sourced LLMs (e.g., whose numbers of parameters are less than 3B) are freely available and easy to deploy locally (e.g., under a single GPU having 8G VRAM), but lack suff icient reasoning ability. This trade-off raises a natural question: can small (free) and large (costly) models collaborate at test time to combine their strengths? We propose a test-time collaboration framework in which a planner model first generates a plan, defined as a distilled and high-level abstraction of the problem.
This plan serves as a lightweight intermediate that guides a reasoner model, which generates a complete solution. Small and large models take turns acting as planner and reasoner, exchanging plans in a multi-round cascade to collaboratively solve complex tasks. Our method achieves accuracy comparable to strong proprietary models alone, while significantly reducing reliance on paid inference. These results highlight planning as an effective prior for orchestrating cost-aware, cross-model inference under real-world deployment constraints. 

---
# RAG+: Enhancing Retrieval-Augmented Generation with Application-Aware Reasoning 

**Authors**: Yu Wang, Shiwan Zhao, Ming Fan, Zhihu Wang, Yubo Zhang, Xicheng Zhang, Zhengfan Wang, Heyuan Huang, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11555)  

**Abstract**: The integration of external knowledge through Retrieval-Augmented Generation (RAG) has become foundational in enhancing large language models (LLMs) for knowledge-intensive tasks. However, existing RAG paradigms often overlook the cognitive step of applying knowledge, leaving a gap between retrieved facts and task-specific reasoning. In this work, we introduce RAG+, a principled and modular extension that explicitly incorporates application-aware reasoning into the RAG pipeline. RAG+ constructs a dual corpus consisting of knowledge and aligned application examples, created either manually or automatically, and retrieves both jointly during inference. This design enables LLMs not only to access relevant information but also to apply it within structured, goal-oriented reasoning processes. Experiments across mathematical, legal, and medical domains, conducted on multiple models, demonstrate that RAG+ consistently outperforms standard RAG variants, achieving average improvements of 3-5%, and peak gains up to 7.5% in complex scenarios. By bridging retrieval with actionable application, RAG+ advances a more cognitively grounded framework for knowledge integration, representing a step toward more interpretable and capable LLMs. 

---
# Reviving DSP for Advanced Theorem Proving in the Era of Reasoning Models 

**Authors**: Chenrui Cao, Liangcheng Song, Zenan Li, Xinyi Le, Xian Zhang, Hui Xue, Fan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11487)  

**Abstract**: Recent advancements, such as DeepSeek-Prover-V2-671B and Kimina-Prover-Preview-72B, demonstrate a prevailing trend in leveraging reinforcement learning (RL)-based large-scale training for automated theorem proving. Surprisingly, we discover that even without any training, careful neuro-symbolic coordination of existing off-the-shelf reasoning models and tactic step provers can achieve comparable performance. This paper introduces \textbf{DSP+}, an improved version of the Draft, Sketch, and Prove framework, featuring a \emph{fine-grained and integrated} neuro-symbolic enhancement for each phase: (1) In the draft phase, we prompt reasoning models to generate concise natural-language subgoals to benefit the sketch phase, removing thinking tokens and references to human-written proofs; (2) In the sketch phase, subgoals are autoformalized with hypotheses to benefit the proving phase, and sketch lines containing syntactic errors are masked according to predefined rules; (3) In the proving phase, we tightly integrate symbolic search methods like Aesop with step provers to establish proofs for the sketch subgoals. Experimental results show that, without any additional model training or fine-tuning, DSP+ solves 80.7\%, 32.8\%, and 24 out of 644 problems from miniF2F, ProofNet, and PutnamBench, respectively, while requiring fewer budgets compared to state-of-the-arts. DSP+ proves \texttt{imo\_2019\_p1}, an IMO problem in miniF2F that is not solved by any prior work. Additionally, DSP+ generates proof patterns comprehensible by human experts, facilitating the identification of formalization errors; For example, eight wrongly formalized statements in miniF2F are discovered. Our results highlight the potential of classical reasoning patterns besides the RL-based training. All components will be open-sourced. 

---
# Structure-Aware Automatic Channel Pruning by Searching with Graph Embedding 

**Authors**: Zifan Liu, Yuan Cao, Yanwei Yu, Heng Qi, Jie Gui  

**Link**: [PDF](https://arxiv.org/pdf/2506.11469)  

**Abstract**: Channel pruning is a powerful technique to reduce the computational overhead of deep neural networks, enabling efficient deployment on resource-constrained devices. However, existing pruning methods often rely on local heuristics or weight-based criteria that fail to capture global structural dependencies within the network, leading to suboptimal pruning decisions and degraded model performance. To address these limitations, we propose a novel structure-aware automatic channel pruning (SACP) framework that utilizes graph convolutional networks (GCNs) to model the network topology and learn the global importance of each channel. By encoding structural relationships within the network, our approach implements topology-aware pruning and this pruning is fully automated, reducing the need for human intervention. We restrict the pruning rate combinations to a specific space, where the number of combinations can be dynamically adjusted, and use a search-based approach to determine the optimal pruning rate combinations. Extensive experiments on benchmark datasets (CIFAR-10, ImageNet) with various models (ResNet, VGG16) demonstrate that SACP outperforms state-of-the-art pruning methods on compression efficiency and competitive on accuracy retention. 

---
# Resolve Highway Conflict in Multi-Autonomous Vehicle Controls with Local State Attention 

**Authors**: Xuan Duy Ta, Bang Giang Le, Thanh Ha Le, Viet Cuong Ta  

**Link**: [PDF](https://arxiv.org/pdf/2506.11445)  

**Abstract**: In mixed-traffic environments, autonomous vehicles must adapt to human-controlled vehicles and other unusual driving situations. This setting can be framed as a multi-agent reinforcement learning (MARL) environment with full cooperative reward among the autonomous vehicles. While methods such as Multi-agent Proximal Policy Optimization can be effective in training MARL tasks, they often fail to resolve local conflict between agents and are unable to generalize to stochastic events. In this paper, we propose a Local State Attention module to assist the input state representation. By relying on the self-attention operator, the module is expected to compress the essential information of nearby agents to resolve the conflict in traffic situations. Utilizing a simulated highway merging scenario with the priority vehicle as the unexpected event, our approach is able to prioritize other vehicles' information to manage the merging process. The results demonstrate significant improvements in merging efficiency compared to popular baselines, especially in high-density traffic settings. 

---
# FocalAD: Local Motion Planning for End-to-End Autonomous Driving 

**Authors**: Bin Sun, Boao Zhang, Jiayi Lu, Xinjie Feng, Jiachen Shang, Rui Cao, Mengchao Zheng, Chuanye Wang, Shichun Yang, Yaoguang Cao, Ziying Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.11419)  

**Abstract**: In end-to-end autonomous driving,the motion prediction plays a pivotal role in ego-vehicle planning. However, existing methods often rely on globally aggregated motion features, ignoring the fact that planning decisions are primarily influenced by a small number of locally interacting agents. Failing to attend to these critical local interactions can obscure potential risks and undermine planning reliability. In this work, we propose FocalAD, a novel end-to-end autonomous driving framework that focuses on critical local neighbors and refines planning by enhancing local motion representations. Specifically, FocalAD comprises two core modules: the Ego-Local-Agents Interactor (ELAI) and the Focal-Local-Agents Loss (FLA Loss). ELAI conducts a graph-based ego-centric interaction representation that captures motion dynamics with local neighbors to enhance both ego planning and agent motion queries. FLA Loss increases the weights of decision-critical neighboring agents, guiding the model to prioritize those more relevant to planning. Extensive experiments show that FocalAD outperforms existing state-of-the-art methods on the open-loop nuScenes datasets and closed-loop Bench2Drive benchmark. Notably, on the robustness-focused Adv-nuScenes dataset, FocalAD achieves even greater improvements, reducing the average colilision rate by 41.9% compared to DiffusionDrive and by 15.6% compared to SparseDrive. 

---
# Large Language Model-Powered Conversational Agent Delivering Problem-Solving Therapy (PST) for Family Caregivers: Enhancing Empathy and Therapeutic Alliance Using In-Context Learning 

**Authors**: Liying Wang, Ph.D., Daffodil Carrington, M.S., Daniil Filienko, M.S., Caroline El Jazmi, M.S., Serena Jinchen Xie, M.S., Martine De Cock, Ph.D., Sarah Iribarren, Ph.D., Weichao Yuwen, Ph.D  

**Link**: [PDF](https://arxiv.org/pdf/2506.11376)  

**Abstract**: Family caregivers often face substantial mental health challenges due to their multifaceted roles and limited resources. This study explored the potential of a large language model (LLM)-powered conversational agent to deliver evidence-based mental health support for caregivers, specifically Problem-Solving Therapy (PST) integrated with Motivational Interviewing (MI) and Behavioral Chain Analysis (BCA). A within-subject experiment was conducted with 28 caregivers interacting with four LLM configurations to evaluate empathy and therapeutic alliance. The best-performing models incorporated Few-Shot and Retrieval-Augmented Generation (RAG) prompting techniques, alongside clinician-curated examples. The models showed improved contextual understanding and personalized support, as reflected by qualitative responses and quantitative ratings on perceived empathy and therapeutic alliances. Participants valued the model's ability to validate emotions, explore unexpressed feelings, and provide actionable strategies. However, balancing thorough assessment with efficient advice delivery remains a challenge. This work highlights the potential of LLMs in delivering empathetic and tailored support for family caregivers. 

---
# Benchmarking Multimodal LLMs on Recognition and Understanding over Chemical Tables 

**Authors**: Yitong Zhou, Mingyue Cheng, Qingyang Mao, Yucong Luo, Qi Liu, Yupeng Li, Xiaohan Zhang, Deguang Liu, Xin Li, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11375)  

**Abstract**: Chemical tables encode complex experimental knowledge through symbolic expressions, structured variables, and embedded molecular graphics. Existing benchmarks largely overlook this multimodal and domain-specific complexity, limiting the ability of multimodal large language models to support scientific understanding in chemistry. In this work, we introduce ChemTable, a large-scale benchmark of real-world chemical tables curated from the experimental sections of literature. ChemTable includes expert-annotated cell polygons, logical layouts, and domain-specific labels, including reagents, catalysts, yields, and graphical components and supports two core tasks: (1) Table Recognition, covering structure parsing and content extraction; and (2) Table Understanding, encompassing both descriptive and reasoning-oriented question answering grounded in table structure and domain semantics. We evaluated a range of representative multimodal models, including both open-source and closed-source models, on ChemTable and reported a series of findings with practical and conceptual insights. Although models show reasonable performance on basic layout parsing, they exhibit substantial limitations on both descriptive and inferential QA tasks compared to human performance, and we observe significant performance gaps between open-source and closed-source models across multiple dimensions. These results underscore the challenges of chemistry-aware table understanding and position ChemTable as a rigorous and realistic benchmark for advancing scientific reasoning. 

---
# MUDAS: Mote-scale Unsupervised Domain Adaptation in Multi-label Sound Classification 

**Authors**: Jihoon Yun, Chengzhang Li, Dhrubojyoti Roy, Anish Arora  

**Link**: [PDF](https://arxiv.org/pdf/2506.11331)  

**Abstract**: Unsupervised Domain Adaptation (UDA) is essential for adapting machine learning models to new, unlabeled environments where data distribution shifts can degrade performance. Existing UDA algorithms are designed for single-label tasks and rely on significant computational resources, limiting their use in multi-label scenarios and in resource-constrained IoT devices. Overcoming these limitations is particularly challenging in contexts such as urban sound classification, where overlapping sounds and varying acoustics require robust, adaptive multi-label capabilities on low-power, on-device systems. To address these limitations, we introduce Mote-scale Unsupervised Domain Adaptation for Sounds (MUDAS), a UDA framework developed for multi-label sound classification in resource-constrained IoT settings. MUDAS efficiently adapts models by selectively retraining the classifier in situ using high-confidence data, minimizing computational and memory requirements to suit on-device deployment. Additionally, MUDAS incorporates class-specific adaptive thresholds to generate reliable pseudo-labels and applies diversity regularization to improve multi-label classification accuracy. In evaluations on the SONYC Urban Sound Tagging (SONYC-UST) dataset recorded at various New York City locations, MUDAS demonstrates notable improvements in classification accuracy over existing UDA algorithms, achieving good performance in a resource-constrained IoT setting. 

---
# LLM-as-a-Fuzzy-Judge: Fine-Tuning Large Language Models as a Clinical Evaluation Judge with Fuzzy Logic 

**Authors**: Weibing Zheng, Laurah Turner, Jess Kropczynski, Murat Ozer, Tri Nguyen, Shane Halse  

**Link**: [PDF](https://arxiv.org/pdf/2506.11221)  

**Abstract**: Clinical communication skills are critical in medical education, and practicing and assessing clinical communication skills on a scale is challenging. Although LLM-powered clinical scenario simulations have shown promise in enhancing medical students' clinical practice, providing automated and scalable clinical evaluation that follows nuanced physician judgment is difficult. This paper combines fuzzy logic and Large Language Model (LLM) and proposes LLM-as-a-Fuzzy-Judge to address the challenge of aligning the automated evaluation of medical students' clinical skills with subjective physicians' preferences. LLM-as-a-Fuzzy-Judge is an approach that LLM is fine-tuned to evaluate medical students' utterances within student-AI patient conversation scripts based on human annotations from four fuzzy sets, including Professionalism, Medical Relevance, Ethical Behavior, and Contextual Distraction. The methodology of this paper started from data collection from the LLM-powered medical education system, data annotation based on multidimensional fuzzy sets, followed by prompt engineering and the supervised fine-tuning (SFT) of the pre-trained LLMs using these human annotations. The results show that the LLM-as-a-Fuzzy-Judge achieves over 80\% accuracy, with major criteria items over 90\%, effectively leveraging fuzzy logic and LLM as a solution to deliver interpretable, human-aligned assessment. This work suggests the viability of leveraging fuzzy logic and LLM to align with human preferences, advances automated evaluation in medical education, and supports more robust assessment and judgment practices. The GitHub repository of this work is available at this https URL 

---
# OntoGSN: An Ontology for Dynamic Management of Assurance Cases 

**Authors**: Tomas Bueno Momcilovic, Barbara Gallina, Ingmar Kessler, Dian Balta  

**Link**: [PDF](https://arxiv.org/pdf/2506.11023)  

**Abstract**: Assurance cases (ACs) are a common artifact for building and maintaining confidence in system properties such as safety or robustness. Constructing an AC can be challenging, although existing tools provide support in static, document-centric applications and methods for dynamic contexts (e.g., autonomous driving) are emerging. Unfortunately, managing ACs remains a challenge, since maintaining the embedded knowledge in the face of changes requires substantial effort, in the process deterring developers - or worse, producing poorly managed cases that instill false confidence. To address this, we present OntoGSN: an ontology and supporting middleware for managing ACs in the Goal Structuring Notation (GSN) standard. OntoGSN offers a knowledge representation and a queryable graph that can be automatically populated, evaluated, and updated. Our contributions include: a 1:1 formalization of the GSN Community Standard v3 in an OWL ontology with SWRL rules; a helper ontology and parser for integration with a widely used AC tool; a repository and documentation of design decisions for OntoGSN maintenance; a SPARQL query library with automation patterns; and a prototypical interface. The ontology strictly adheres to the standard's text and has been evaluated according to FAIR principles, the OOPS framework, competency questions, and community feedback. The development of other middleware elements is guided by the community needs and subject to ongoing evaluations. To demonstrate the utility of our contributions, we illustrate dynamic AC management in an example involving assurance of adversarial robustness in large language models. 

---
# A Survey of Task-Oriented Knowledge Graph Reasoning: Status, Applications, and Prospects 

**Authors**: Guanglin Niu, Bo Li, Yangguang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11012)  

**Abstract**: Knowledge graphs (KGs) have emerged as a powerful paradigm for structuring and leveraging diverse real-world knowledge, which serve as a fundamental technology for enabling cognitive intelligence systems with advanced understanding and reasoning capabilities. Knowledge graph reasoning (KGR) aims to infer new knowledge based on existing facts in KGs, playing a crucial role in applications such as public security intelligence, intelligent healthcare, and financial risk assessment. From a task-centric perspective, existing KGR approaches can be broadly classified into static single-step KGR, static multi-step KGR, dynamic KGR, multi-modal KGR, few-shot KGR, and inductive KGR. While existing surveys have covered these six types of KGR tasks, a comprehensive review that systematically summarizes all KGR tasks particularly including downstream applications and more challenging reasoning paradigms remains lacking. In contrast to previous works, this survey provides a more comprehensive perspective on the research of KGR by categorizing approaches based on primary reasoning tasks, downstream application tasks, and potential challenging reasoning tasks. Besides, we explore advanced techniques, such as large language models (LLMs), and their impact on KGR. This work aims to highlight key research trends and outline promising future directions in the field of KGR. 

---
# EMLoC: Emulator-based Memory-efficient Fine-tuning with LoRA Correction 

**Authors**: Hsi-Che Lin, Yu-Chu Yu, Kai-Po Chang, Yu-Chiang Frank Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12015)  

**Abstract**: Open-source foundation models have seen rapid adoption and development, enabling powerful general-purpose capabilities across diverse domains. However, fine-tuning large foundation models for domain-specific or personalized tasks remains prohibitively expensive for most users due to the significant memory overhead beyond that of inference. We introduce EMLoC, an Emulator-based Memory-efficient fine-tuning framework with LoRA Correction, which enables model fine-tuning within the same memory budget required for inference. EMLoC constructs a task-specific light-weight emulator using activation-aware singular value decomposition (SVD) on a small downstream calibration set. Fine-tuning then is performed on this lightweight emulator via LoRA. To tackle the misalignment between the original model and the compressed emulator, we propose a novel compensation algorithm to correct the fine-tuned LoRA module, which thus can be merged into the original model for inference. EMLoC supports flexible compression ratios and standard training pipelines, making it adaptable to a wide range of applications. Extensive experiments demonstrate that EMLoC outperforms other baselines across multiple datasets and modalities. Moreover, without quantization, EMLoC enables fine-tuning of a 38B model on a single 24GB consumer GPU-bringing efficient and practical model adaptation to individual users. 

---
# code_transformed: The Influence of Large Language Models on Code 

**Authors**: Yuliang Xu, Siming Huang, Mingmeng Geng, Yao Wan, Xuanhua Shi, Dongping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.12014)  

**Abstract**: Coding remains one of the most fundamental modes of interaction between humans and machines. With the rapid advancement of Large Language Models (LLMs), code generation capabilities have begun to significantly reshape programming practices. This development prompts a central question: Have LLMs transformed code style, and how can such transformation be characterized? In this paper, we present a pioneering study that investigates the impact of LLMs on code style, with a focus on naming conventions, complexity, maintainability, and similarity. By analyzing code from over 19,000 GitHub repositories linked to arXiv papers published between 2020 and 2025, we identify measurable trends in the evolution of coding style that align with characteristics of LLM-generated code. For instance, the proportion of snake\_case variable names in Python code increased from 47% in Q1 2023 to 51% in Q1 2025. Furthermore, we investigate how LLMs approach algorithmic problems by examining their reasoning processes. Given the diversity of LLMs and usage scenarios, among other factors, it is difficult or even impossible to precisely estimate the proportion of code generated or assisted by LLMs. Our experimental results provide the first large-scale empirical evidence that LLMs affect real-world programming style. 

---
# Reimagining Dance: Real-time Music Co-creation between Dancers and AI 

**Authors**: Olga Vechtomova, Jeff Bos  

**Link**: [PDF](https://arxiv.org/pdf/2506.12008)  

**Abstract**: Dance performance traditionally follows a unidirectional relationship where movement responds to music. While AI has advanced in various creative domains, its application in dance has primarily focused on generating choreography from musical input. We present a system that enables dancers to dynamically shape musical environments through their movements. Our multi-modal architecture creates a coherent musical composition by intelligently combining pre-recorded musical clips in response to dance movements, establishing a bidirectional creative partnership where dancers function as both performers and composers. Through correlation analysis of performance data, we demonstrate emergent communication patterns between movement qualities and audio features. This approach reconceptualizes the role of AI in performing arts as a responsive collaborator that expands possibilities for both professional dance performance and improvisational artistic expression across broader populations. 

---
# Upgrade or Switch: Do We Need a New Registry Architecture for the Internet of AI Agents? 

**Authors**: Ramesh Raskar, Pradyumna Chari, Jared James Grogan, Mahesh Lambe, Robert Lincourt, Raghu Bala, Abhishek Singh, Ayush Chopra, Rajesh Ranjan, Shailja Gupta, Dimitris Stripelis, Maria Gorskikh, Sichao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12003)  

**Abstract**: The emerging Internet of AI Agents challenges existing web infrastructure designed for human-scale, reactive interactions. Unlike traditional web resources, autonomous AI agents initiate actions, maintain persistent state, spawn sub-agents, and negotiate directly with peers: demanding millisecond-level discovery, instant credential revocation, and cryptographic behavioral proofs that exceed current DNS/PKI capabilities. This paper analyzes whether to upgrade existing infrastructure or implement purpose-built registry architectures for autonomous agents. We identify critical failure points: DNS propagation (24-48 hours vs. required milliseconds), certificate revocation unable to scale to trillions of entities, and IPv4/IPv6 addressing inadequate for agent-scale routing. We evaluate three approaches: (1) Upgrade paths, (2) Switch options, (3) Hybrid registries. Drawing parallels to dialup-to-broadband transitions, we find that agent requirements constitute qualitative, and not incremental, changes. While upgrades offer compatibility and faster deployment, clean-slate solutions provide better performance but require longer for adoption. Our analysis suggests hybrid approaches will emerge, with centralized registries for critical agents and federated meshes for specialized use cases. 

---
# VGR: Visual Grounded Reasoning 

**Authors**: Jiacong Wang, Zijiang Kang, Haochen Wang, Haiyong Jiang, Jiawen Li, Bohong Wu, Ya Wang, Jiao Ran, Xiao Liang, Chao Feng, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.11991)  

**Abstract**: In the field of multimodal chain-of-thought (CoT) reasoning, existing approaches predominantly rely on reasoning on pure language space, which inherently suffers from language bias and is largely confined to math or science domains. This narrow focus limits their ability to handle complex visual reasoning tasks that demand comprehensive understanding of image details. To address these limitations, this paper introduces VGR, a novel reasoning multimodal large language model (MLLM) with enhanced fine-grained visual perception capabilities. Unlike traditional MLLMs that answer the question or reasoning solely on the language space, our VGR first detects relevant regions that may help to solve problems, and then provides precise answers based on replayed image regions. To achieve this, we conduct a large-scale SFT dataset called VGR -SFT that contains reasoning data with mixed vision grounding and language deduction. The inference pipeline of VGR allows the model to choose bounding boxes for visual reference and a replay stage is introduced to integrates the corresponding regions into the reasoning process, enhancing multimodel comprehension. Experiments on the LLaVA-NeXT-7B baseline show that VGR achieves superior performance on multi-modal benchmarks requiring comprehensive image detail understanding. Compared to the baseline, VGR uses only 30\% of the image token count while delivering scores of +4.1 on MMStar, +7.1 on AI2D, and a +12.9 improvement on ChartQA. 

---
# Technical Evaluation of a Disruptive Approach in Homomorphic AI 

**Authors**: Eric Filiol  

**Link**: [PDF](https://arxiv.org/pdf/2506.11954)  

**Abstract**: We present a technical evaluation of a new, disruptive cryptographic approach to data security, known as HbHAI (Hash-based Homomorphic Artificial Intelligence). HbHAI is based on a novel class of key-dependent hash functions that naturally preserve most similarity properties, most AI algorithms rely on. As a main claim, HbHAI makes now possible to analyze and process data in its cryptographically secure form while using existing native AI algorithms without modification, with unprecedented performances compared to existing homomorphic encryption schemes.
We tested various HbHAI-protected datasets (non public preview) using traditional unsupervised and supervised learning techniques (clustering, classification, deep neural networks) with classical unmodified AI algorithms. This paper presents technical results from an independent analysis conducted with those different, off-the-shelf AI algorithms. The aim was to assess the security, operability and performance claims regarding HbHAI techniques. As a results, our results confirm most these claims, with only a few minor reservations. 

---
# SAIL: Faster-than-Demonstration Execution of Imitation Learning Policies 

**Authors**: Nadun Ranawaka Arachchige, Zhenyang Chen, Wonsuhk Jung, Woo Chul Shin, Rohan Bansal, Pierre Barroso, Yu Hang He, Yingyang Celine Lin, Benjamin Joffe, Shreyas Kousik, Danfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11948)  

**Abstract**: Offline Imitation Learning (IL) methods such as Behavior Cloning are effective at acquiring complex robotic manipulation skills. However, existing IL-trained policies are confined to executing the task at the same speed as shown in demonstration data. This limits the task throughput of a robotic system, a critical requirement for applications such as industrial automation. In this paper, we introduce and formalize the novel problem of enabling faster-than-demonstration execution of visuomotor policies and identify fundamental challenges in robot dynamics and state-action distribution shifts. We instantiate the key insights as SAIL (Speed Adaptation for Imitation Learning), a full-stack system integrating four tightly-connected components: (1) a consistency-preserving action inference algorithm for smooth motion at high speed, (2) high-fidelity tracking of controller-invariant motion targets, (3) adaptive speed modulation that dynamically adjusts execution speed based on motion complexity, and (4) action scheduling to handle real-world system latencies. Experiments on 12 tasks across simulation and two real, distinct robot platforms show that SAIL achieves up to a 4x speedup over demonstration speed in simulation and up to 3.2x speedup in the real world. Additional detail is available at this https URL 

---
# Subjective Experience in AI Systems: What Do AI Researchers and the Public Believe? 

**Authors**: Noemi Dreksler, Lucius Caviola, David Chalmers, Carter Allen, Alex Rand, Joshua Lewis, Philip Waggoner, Kate Mays, Jeff Sebo  

**Link**: [PDF](https://arxiv.org/pdf/2506.11945)  

**Abstract**: We surveyed 582 AI researchers who have published in leading AI venues and 838 nationally representative US participants about their views on the potential development of AI systems with subjective experience and how such systems should be treated and governed. When asked to estimate the chances that such systems will exist on specific dates, the median responses were 1% (AI researchers) and 5% (public) by 2024, 25% and 30% by 2034, and 70% and 60% by 2100, respectively. The median member of the public thought there was a higher chance that AI systems with subjective experience would never exist (25%) than the median AI researcher did (10%). Both groups perceived a need for multidisciplinary expertise to assess AI subjective experience. Although support for welfare protections for such AI systems exceeded opposition, it remained far lower than support for protections for animals or the environment. Attitudes toward moral and governance issues were divided in both groups, especially regarding whether such systems should be created and what rights or protections they should receive. Yet a majority of respondents in both groups agreed that safeguards against the potential risks from AI systems with subjective experience should be implemented by AI developers now, and if created, AI systems with subjective experience should treat others well, behave ethically, and be held accountable. Overall, these results suggest that both AI researchers and the public regard the emergence of AI systems with subjective experience as a possibility this century, though substantial uncertainty and disagreement remain about the timeline and appropriate response. 

---
# Today's Cat Is Tomorrow's Dog: Accounting for Time-Based Changes in the Labels of ML Vulnerability Detection Approaches 

**Authors**: Ranindya Paramitha, Yuan Feng, Fabio Massacci  

**Link**: [PDF](https://arxiv.org/pdf/2506.11939)  

**Abstract**: Vulnerability datasets used for ML testing implicitly contain retrospective information. When tested on the field, one can only use the labels available at the time of training and testing (e.g. seen and assumed negatives). As vulnerabilities are discovered across calendar time, labels change and past performance is not necessarily aligned with future performance. Past works only considered the slices of the whole history (e.g. DiverseVUl) or individual differences between releases (e.g. Jimenez et al. ESEC/FSE 2019). Such approaches are either too optimistic in training (e.g. the whole history) or too conservative (e.g. consecutive releases). We propose a method to restructure a dataset into a series of datasets in which both training and testing labels change to account for the knowledge available at the time. If the model is actually learning, it should improve its performance over time as more data becomes available and data becomes more stable, an effect that can be checked with the Mann-Kendall test. We validate our methodology for vulnerability detection with 4 time-based datasets (3 projects from BigVul dataset + Vuldeepecker's NVD) and 5 ML models (Code2Vec, CodeBERT, LineVul, ReGVD, and Vuldeepecker). In contrast to the intuitive expectation (more retrospective information, better performance), the trend results show that performance changes inconsistently across the years, showing that most models are not learning. 

---
# Improving Large Language Model Safety with Contrastive Representation Learning 

**Authors**: Samuel Simko, Mrinmaya Sachan, Bernhard Schölkopf, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11938)  

**Abstract**: Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at this https URL 

---
# LiveCodeBench Pro: How Do Olympiad Medalists Judge LLMs in Competitive Programming? 

**Authors**: Zihan Zheng, Zerui Cheng, Zeyu Shen, Shang Zhou, Kaiyuan Liu, Hansen He, Dongruixuan Li, Stanley Wei, Hangyi Hao, Jianzhu Yao, Peiyao Sheng, Zixuan Wang, Wenhao Chai, Aleksandra Korolova, Peter Henderson, Sanjeev Arora, Pramod Viswanath, Jingbo Shang, Saining Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.11928)  

**Abstract**: Recent reports claim that large language models (LLMs) now outperform elite humans in competitive programming. Drawing on knowledge from a group of medalists in international algorithmic contests, we revisit this claim, examining how LLMs differ from human experts and where limitations still remain. We introduce LiveCodeBench Pro, a benchmark composed of problems from Codeforces, ICPC, and IOI that are continuously updated to reduce the likelihood of data contamination. A team of Olympiad medalists annotates every problem for algorithmic categories and conducts a line-by-line analysis of failed model-generated submissions. Using this new data and benchmark, we find that frontier models still have significant limitations: without external tools, the best model achieves only 53% pass@1 on medium-difficulty problems and 0% on hard problems, domains where expert humans still excel. We also find that LLMs succeed at implementation-heavy problems but struggle with nuanced algorithmic reasoning and complex case analysis, often generating confidently incorrect justifications. High performance appears largely driven by implementation precision and tool augmentation, not superior reasoning. LiveCodeBench Pro thus highlights the significant gap to human grandmaster levels, while offering fine-grained diagnostics to steer future improvements in code-centric LLM reasoning. 

---
# Real-World Deployment of a Lane Change Prediction Architecture Based on Knowledge Graph Embeddings and Bayesian Inference 

**Authors**: M. Manzour, Catherine M. Elias, Omar M. Shehata, R. Izquierdo, M. A. Sotelo  

**Link**: [PDF](https://arxiv.org/pdf/2506.11925)  

**Abstract**: Research on lane change prediction has gained a lot of momentum in the last couple of years. However, most research is confined to simulation or results obtained from datasets, leaving a gap between algorithmic advances and on-road deployment. This work closes that gap by demonstrating, on real hardware, a lane-change prediction system based on Knowledge Graph Embeddings (KGEs) and Bayesian inference. Moreover, the ego-vehicle employs a longitudinal braking action to ensure the safety of both itself and the surrounding vehicles. Our architecture consists of two modules: (i) a perception module that senses the environment, derives input numerical features, and converts them into linguistic categories; and communicates them to the prediction module; (ii) a pretrained prediction module that executes a KGE and Bayesian inference model to anticipate the target vehicle's maneuver and transforms the prediction into longitudinal braking action. Real-world hardware experimental validation demonstrates that our prediction system anticipates the target vehicle's lane change three to four seconds in advance, providing the ego vehicle sufficient time to react and allowing the target vehicle to make the lane change safely. 

---
# Breaking Habits: On the Role of the Advantage Function in Learning Causal State Representations 

**Authors**: Miguel Suau  

**Link**: [PDF](https://arxiv.org/pdf/2506.11912)  

**Abstract**: Recent work has shown that reinforcement learning agents can develop policies that exploit spurious correlations between rewards and observations. This phenomenon, known as policy confounding, arises because the agent's policy influences both past and future observation variables, creating a feedback loop that can hinder the agent's ability to generalize beyond its usual trajectories. In this paper, we show that the advantage function, commonly used in policy gradient methods, not only reduces the variance of gradient estimates but also mitigates the effects of policy confounding. By adjusting action values relative to the state representation, the advantage function downweights state-action pairs that are more likely under the current policy, breaking spurious correlations and encouraging the agent to focus on causal factors. We provide both analytical and empirical evidence demonstrating that training with the advantage function leads to improved out-of-trajectory performance. 

---
# Spectra-to-Structure and Structure-to-Spectra Inference Across the Periodic Table 

**Authors**: Yufeng Wang, Peiyao Wang, Lu Ma, Yuewei Lin, Qun Liu, Haibin Ling  

**Link**: [PDF](https://arxiv.org/pdf/2506.11908)  

**Abstract**: X-ray Absorption Spectroscopy (XAS) is a powerful technique for probing local atomic environments, yet its interpretation remains limited by the need for expert-driven analysis, computationally expensive simulations, and element-specific heuristics. Recent advances in machine learning have shown promise for accelerating XAS interpretation, but many existing models are narrowly focused on specific elements, edge types, or spectral regimes. In this work, we present XAStruct, a learning framework capable of both predicting XAS spectra from crystal structures and inferring local structural descriptors from XAS input. XAStruct is trained on a large-scale dataset spanning over 70 elements across the periodic table, enabling generalization to a wide variety of chemistries and bonding environments. The model includes the first machine learning approach for predicting neighbor atom types directly from XAS spectra, as well as a unified regression model for mean nearest-neighbor distance that requires no element-specific tuning. While we explored integrating the two pipelines into a single end-to-end model, empirical results showed performance degradation. As a result, the two tasks were trained independently to ensure optimal accuracy and task-specific performance. By combining deep neural networks for complex structure-property mappings with efficient baseline models for simpler tasks, XAStruct offers a scalable and extensible solution for data-driven XAS analysis and local structure inference. The source code will be released upon paper acceptance. 

---
# A Neural Rejection System Against Universal Adversarial Perturbations in Radio Signal Classification 

**Authors**: Lu Zhang, Sangarapillai Lambotharan, Gan Zheng, Fabio Roli  

**Link**: [PDF](https://arxiv.org/pdf/2506.11901)  

**Abstract**: Advantages of deep learning over traditional methods have been demonstrated for radio signal classification in the recent years. However, various researchers have discovered that even a small but intentional feature perturbation known as adversarial examples can significantly deteriorate the performance of the deep learning based radio signal classification. Among various kinds of adversarial examples, universal adversarial perturbation has gained considerable attention due to its feature of being data independent, hence as a practical strategy to fool the radio signal classification with a high success rate. Therefore, in this paper, we investigate a defense system called neural rejection system to propose against universal adversarial perturbations, and evaluate its performance by generating white-box universal adversarial perturbations. We show that the proposed neural rejection system is able to defend universal adversarial perturbations with significantly higher accuracy than the undefended deep neural network. 

---
# Attention-based Adversarial Robust Distillation in Radio Signal Classifications for Low-Power IoT Devices 

**Authors**: Lu Zhang, Sangarapillai Lambotharan, Gan Zheng, Guisheng Liao, Basil AsSadhan, Fabio Roli  

**Link**: [PDF](https://arxiv.org/pdf/2506.11892)  

**Abstract**: Due to great success of transformers in many applications such as natural language processing and computer vision, transformers have been successfully applied in automatic modulation classification. We have shown that transformer-based radio signal classification is vulnerable to imperceptible and carefully crafted attacks called adversarial examples. Therefore, we propose a defense system against adversarial examples in transformer-based modulation classifications. Considering the need for computationally efficient architecture particularly for Internet of Things (IoT)-based applications or operation of devices in environment where power supply is limited, we propose a compact transformer for modulation classification. The advantages of robust training such as adversarial training in transformers may not be attainable in compact transformers. By demonstrating this, we propose a novel compact transformer that can enhance robustness in the presence of adversarial attacks. The new method is aimed at transferring the adversarial attention map from the robustly trained large transformer to a compact transformer. The proposed method outperforms the state-of-the-art techniques for the considered white-box scenarios including fast gradient method and projected gradient descent attacks. We have provided reasoning of the underlying working mechanisms and investigated the transferability of the adversarial examples between different architectures. The proposed method has the potential to protect the transformer from the transferability of adversarial examples. 

---
# Enter: Graduated Realism: A Pedagogical Framework for AI-Powered Avatars in Virtual Reality Teacher Training 

**Authors**: Judson Leroy Dean Haynes IV  

**Link**: [PDF](https://arxiv.org/pdf/2506.11890)  

**Abstract**: Virtual Reality simulators offer a powerful tool for teacher training, yet the integration of AI-powered student avatars presents a critical challenge: determining the optimal level of avatar realism for effective pedagogy. This literature review examines the evolution of avatar realism in VR teacher training, synthesizes its theoretical implications, and proposes a new pedagogical framework to guide future design. Through a systematic review, this paper traces the progression from human-controlled avatars to generative AI prototypes. Applying learning theories like Cognitive Load Theory, we argue that hyper-realism is not always optimal, as high-fidelity avatars can impose excessive extraneous cognitive load on novices, a stance supported by recent empirical findings. A significant gap exists between the technological drive for photorealism and the pedagogical need for scaffolded learning. To address this gap, we propose Graduated Realism, a framework advocating for starting trainees with lower-fidelity avatars and progressively increasing behavioral complexity as skills develop. To make this computationally feasible, we outline a novel single-call architecture, Crazy Slots, which uses a probabilistic engine and a Retrieval-Augmented Generation database to generate authentic, real-time responses without the latency and cost of multi-step reasoning models. This review provides evidence-based principles for designing the next generation of AI simulators, arguing that a pedagogically grounded approach to realism is essential for creating scalable and effective teacher education tools. 

---
# An Explainable AI Framework for Dynamic Resource Management in Vehicular Network Slicing 

**Authors**: Haochen Sun, Yifan Liu, Ahmed Al-Tahmeesschi, Swarna Chetty, Syed Ali Raza Zaidi, Avishek Nag, Hamed Ahmadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.11882)  

**Abstract**: Effective resource management and network slicing are essential to meet the diverse service demands of vehicular networks, including Enhanced Mobile Broadband (eMBB) and Ultra-Reliable and Low-Latency Communications (URLLC). This paper introduces an Explainable Deep Reinforcement Learning (XRL) framework for dynamic network slicing and resource allocation in vehicular networks, built upon a near-real-time RAN intelligent controller. By integrating a feature-based approach that leverages Shapley values and an attention mechanism, we interpret and refine the decisions of our reinforcementlearning agents, addressing key reliability challenges in vehicular communication systems. Simulation results demonstrate that our approach provides clear, real-time insights into the resource allocation process and achieves higher interpretability precision than a pure attention mechanism. Furthermore, the Quality of Service (QoS) satisfaction for URLLC services increased from 78.0% to 80.13%, while that for eMBB services improved from 71.44% to 73.21%. 

---
# Robust Molecular Property Prediction via Densifying Scarce Labeled Data 

**Authors**: Jina Kim, Jeffrey Willette, Bruno Andreis, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11877)  

**Abstract**: A widely recognized limitation of molecular prediction models is their reliance on structures observed in the training data, resulting in poor generalization to out-of-distribution compounds. Yet in drug discovery, the compounds most critical for advancing research often lie beyond the training set, making the bias toward the training data particularly problematic. This mismatch introduces substantial covariate shift, under which standard deep learning models produce unstable and inaccurate predictions. Furthermore, the scarcity of labeled data, stemming from the onerous and costly nature of experimental validation, further exacerbates the difficulty of achieving reliable generalization. To address these limitations, we propose a novel meta-learning-based approach that leverages unlabeled data to interpolate between in-distribution (ID) and out-of-distribution (OOD) data, enabling the model to meta-learn how to generalize beyond the training distribution. We demonstrate significant performance gains over state-of-the-art methods on challenging real-world datasets that exhibit substantial covariate shift. 

---
# How do Probabilistic Graphical Models and Graph Neural Networks Look at Network Data? 

**Authors**: Michela Lapenna, Caterina De Bacco  

**Link**: [PDF](https://arxiv.org/pdf/2506.11869)  

**Abstract**: Graphs are a powerful data structure for representing relational data and are widely used to describe complex real-world systems. Probabilistic Graphical Models (PGMs) and Graph Neural Networks (GNNs) can both leverage graph-structured data, but their inherent functioning is different. The question is how do they compare in capturing the information contained in networked datasets? We address this objective by solving a link prediction task and we conduct three main experiments, on both synthetic and real networks: one focuses on how PGMs and GNNs handle input features, while the other two investigate their robustness to noisy features and increasing heterophily of the graph. PGMs do not necessarily require features on nodes, while GNNs cannot exploit the network edges alone, and the choice of input features matters. We find that GNNs are outperformed by PGMs when input features are low-dimensional or noisy, mimicking many real scenarios where node attributes might be scalar or noisy. Then, we find that PGMs are more robust than GNNs when the heterophily of the graph is increased. Finally, to assess performance beyond prediction tasks, we also compare the two frameworks in terms of their computational complexity and interpretability. 

---
# MindGrab for BrainChop: Fast and Accurate Skull Stripping for Command Line and Browser 

**Authors**: Armina Fani, Mike Doan, Isabelle Le, Alex Fedorov, Malte Hoffmann, Chris Rorden, Sergey Plis  

**Link**: [PDF](https://arxiv.org/pdf/2506.11860)  

**Abstract**: We developed MindGrab, a parameter- and memory-efficient deep fully-convolutional model for volumetric skull-stripping in head images of any modality. Its architecture, informed by a spectral interpretation of dilated convolutions, was trained exclusively on modality-agnostic synthetic data. MindGrab was evaluated on a retrospective dataset of 606 multimodal adult-brain scans (T1, T2, DWI, MRA, PDw MRI, EPI, CT, PET) sourced from the SynthStrip dataset. Performance was benchmarked against SynthStrip, ROBEX, and BET using Dice scores, with Wilcoxon signed-rank significance tests. MindGrab achieved a mean Dice score of 95.9 with standard deviation (SD) 1.6 across modalities, significantly outperforming classical methods (ROBEX: 89.1 SD 7.7, P < 0.05; BET: 85.2 SD 14.4, P < 0.05). Compared to SynthStrip (96.5 SD 1.1, P=0.0352), MindGrab delivered equivalent or superior performance in nearly half of the tested scenarios, with minor differences (<3% Dice) in the others. MindGrab utilized 95% fewer parameters (146,237 vs. 2,566,561) than SynthStrip. This efficiency yielded at least 2x faster inference, 50% lower memory usage on GPUs, and enabled exceptional performance (e.g., 10-30x speedup, and up to 30x memory reduction) and accessibility on a wider range of hardware, including systems without high-end GPUs. MindGrab delivers state-of-the-art accuracy with dramatically lower resource demands, supported in brainchop-cli (this https URL) and at this http URL. 

---
# Regression-adjusted Monte Carlo Estimators for Shapley Values and Probabilistic Values 

**Authors**: R. Teal Witter, Yurong Liu, Christopher Musco  

**Link**: [PDF](https://arxiv.org/pdf/2506.11849)  

**Abstract**: With origins in game theory, probabilistic values like Shapley values, Banzhaf values, and semi-values have emerged as a central tool in explainable AI. They are used for feature attribution, data attribution, data valuation, and more. Since all of these values require exponential time to compute exactly, research has focused on efficient approximation methods using two techniques: Monte Carlo sampling and linear regression formulations. In this work, we present a new way of combining both of these techniques. Our approach is more flexible than prior algorithms, allowing for linear regression to be replaced with any function family whose probabilistic values can be computed efficiently. This allows us to harness the accuracy of tree-based models like XGBoost, while still producing unbiased estimates. From experiments across eight datasets, we find that our methods give state-of-the-art performance for estimating probabilistic values. For Shapley values, the error of our methods can be $6.5\times$ lower than Permutation SHAP (the most popular Monte Carlo method), $3.8\times$ lower than Kernel SHAP (the most popular linear regression method), and $2.6\times$ lower than Leverage SHAP (the prior state-of-the-art Shapley value estimator). For more general probabilistic values, we can obtain error $215\times$ lower than the best estimator from prior work. 

---
# TrustGLM: Evaluating the Robustness of GraphLLMs Against Prompt, Text, and Structure Attacks 

**Authors**: Qihai Zhang, Xinyue Sheng, Yuanfu Sun, Qiaoyu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.11844)  

**Abstract**: Inspired by the success of large language models (LLMs), there is a significant research shift from traditional graph learning methods to LLM-based graph frameworks, formally known as GraphLLMs. GraphLLMs leverage the reasoning power of LLMs by integrating three key components: the textual attributes of input nodes, the structural information of node neighborhoods, and task-specific prompts that guide decision-making. Despite their promise, the robustness of GraphLLMs against adversarial perturbations remains largely unexplored-a critical concern for deploying these models in high-stakes scenarios. To bridge the gap, we introduce TrustGLM, a comprehensive study evaluating the vulnerability of GraphLLMs to adversarial attacks across three dimensions: text, graph structure, and prompt manipulations. We implement state-of-the-art attack algorithms from each perspective to rigorously assess model resilience. Through extensive experiments on six benchmark datasets from diverse domains, our findings reveal that GraphLLMs are highly susceptible to text attacks that merely replace a few semantically similar words in a node's textual attribute. We also find that standard graph structure attack methods can significantly degrade model performance, while random shuffling of the candidate label set in prompt templates leads to substantial performance drops. Beyond characterizing these vulnerabilities, we investigate defense techniques tailored to each attack vector through data-augmented training and adversarial training, which show promising potential to enhance the robustness of GraphLLMs. We hope that our open-sourced library will facilitate rapid, equitable evaluation and inspire further innovative research in this field. 

---
# Diffusion-Based Electrocardiography Noise Quantification via Anomaly Detection 

**Authors**: Tae-Seong Han, Jae-Wook Heo, Hakseung Kim, Cheol-Hui Lee, Hyub Huh, Eue-Keun Choi, Dong-Joo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.11815)  

**Abstract**: Electrocardiography (ECG) signals are often degraded by noise, which complicates diagnosis in clinical and wearable settings. This study proposes a diffusion-based framework for ECG noise quantification via reconstruction-based anomaly detection, addressing annotation inconsistencies and the limited generalizability of conventional methods. We introduce a distributional evaluation using the Wasserstein-1 distance ($W_1$), comparing the reconstruction error distributions between clean and noisy ECGs to mitigate inconsistent annotations. Our final model achieved robust noise quantification using only three reverse diffusion steps. The model recorded a macro-average $W_1$ score of 1.308 across the benchmarks, outperforming the next-best method by over 48%. External validations demonstrated strong generalizability, supporting the exclusion of low-quality segments to enhance diagnostic accuracy and enable timely clinical responses to signal degradation. The proposed method enhances clinical decision-making, diagnostic accuracy, and real-time ECG monitoring capabilities, supporting future advancements in clinical and wearable ECG applications. 

---
# Abstract Sound Fusion with Unconditioned Inversion Model 

**Authors**: Jing Liu, EnQi Lian  

**Link**: [PDF](https://arxiv.org/pdf/2506.11811)  

**Abstract**: An abstract sound is defined as a sound that does not disclose identifiable real-world sound events to a listener. Sound fusion aims to synthesize an original sound and a reference sound to generate a novel sound that exhibits auditory features beyond mere additive superposition of the sound constituents. To achieve this fusion, we employ inversion techniques that preserve essential features of the original sample while enabling controllable synthesis. We propose novel SDE and ODE inversion models based on DPMSolver++ samplers that reverse the sampling process by configuring model outputs as constants, eliminating circular dependencies incurred by noise prediction terms. Our inversion approach requires no prompt conditioning while maintaining flexible guidance during sampling. 

---
# Persona-driven Simulation of Voting Behavior in the European Parliament with Large Language Models 

**Authors**: Maximilian Kreutner, Marlene Lutz, Markus Strohmaier  

**Link**: [PDF](https://arxiv.org/pdf/2506.11798)  

**Abstract**: Large Language Models (LLMs) display remarkable capabilities to understand or even produce political discourse, but have been found to consistently display a progressive left-leaning bias. At the same time, so-called persona or identity prompts have been shown to produce LLM behavior that aligns with socioeconomic groups that the base model is not aligned with. In this work, we analyze whether zero-shot persona prompting with limited information can accurately predict individual voting decisions and, by aggregation, accurately predict positions of European groups on a diverse set of policies. We evaluate if predictions are stable towards counterfactual arguments, different persona prompts and generation methods. Finally, we find that we can simulate voting behavior of Members of the European Parliament reasonably well with a weighted F1 score of approximately 0.793. Our persona dataset of politicians in the 2024 European Parliament and our code are available at this https URL. 

---
# Why Do Class-Dependent Evaluation Effects Occur with Time Series Feature Attributions? A Synthetic Data Investigation 

**Authors**: Gregor Baer, Isel Grau, Chao Zhang, Pieter Van Gorp  

**Link**: [PDF](https://arxiv.org/pdf/2506.11790)  

**Abstract**: Evaluating feature attribution methods represents a critical challenge in explainable AI (XAI), as researchers typically rely on perturbation-based metrics when ground truth is unavailable. However, recent work demonstrates that these evaluation metrics can show different performance across predicted classes within the same dataset. These "class-dependent evaluation effects" raise questions about whether perturbation analysis reliably measures attribution quality, with direct implications for XAI method development and the trustworthiness of evaluation techniques. We investigate under which conditions these class-dependent effects arise by conducting controlled experiments with synthetic time series data where ground truth feature locations are known. We systematically vary feature types and class contrasts across binary classification tasks, then compare perturbation-based degradation scores with ground truth-based precision-recall metrics using multiple attribution methods. Our experiments demonstrate that class-dependent effects emerge with both evaluation approaches even in simple scenarios with temporally localized features, triggered by basic variations in feature amplitude or temporal extent between classes. Most critically, we find that perturbation-based and ground truth metrics frequently yield contradictory assessments of attribution quality across classes, with weak correlations between evaluation approaches. These findings suggest that researchers should interpret perturbation-based metrics with care, as they may not always align with whether attributions correctly identify discriminating features. These findings reveal opportunities to reconsider what attribution evaluation actually measures and to develop more comprehensive evaluation frameworks that capture multiple dimensions of attribution quality. 

---
# Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation 

**Authors**: Divyanshu Mishra, Mohammadreza Salehi, Pramit Saha, Olga Patey, Aris T. Papageorghiou, Yuki M. Asano, J. Alison Noble  

**Link**: [PDF](https://arxiv.org/pdf/2506.11777)  

**Abstract**: Self-supervised learning (SSL) has achieved major advances in natural images and video understanding, but challenges remain in domains like echocardiography (heart ultrasound) due to subtle anatomical structures, complex temporal dynamics, and the current lack of domain-specific pre-trained models. Existing SSL approaches such as contrastive, masked modeling, and clustering-based methods struggle with high intersample similarity, sensitivity to low PSNR inputs common in ultrasound, or aggressive augmentations that distort clinically relevant features. We present DISCOVR (Distilled Image Supervision for Cross Modal Video Representation), a self-supervised dual branch framework for cardiac ultrasound video representation learning. DISCOVR combines a clustering-based video encoder that models temporal dynamics with an online image encoder that extracts fine-grained spatial semantics. These branches are connected through a semantic cluster distillation loss that transfers anatomical knowledge from the evolving image encoder to the video encoder, enabling temporally coherent representations enriched with fine-grained semantic understanding. Evaluated on six echocardiography datasets spanning fetal, pediatric, and adult populations, DISCOVR outperforms both specialized video anomaly detection methods and state-of-the-art video-SSL baselines in zero-shot and linear probing setups, and achieves superior segmentation transfer. 

---
# Real-Time Feedback and Benchmark Dataset for Isometric Pose Evaluation 

**Authors**: Abhishek Jaiswal, Armeet Singh Luthra, Purav Jangir, Bhavya Garg, Nisheeth Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2506.11774)  

**Abstract**: Isometric exercises appeal to individuals seeking convenience, privacy, and minimal dependence on equipments. However, such fitness training is often overdependent on unreliable digital media content instead of expert supervision, introducing serious risks, including incorrect posture, injury, and disengagement due to lack of corrective feedback. To address these challenges, we present a real-time feedback system for assessing isometric poses. Our contributions include the release of the largest multiclass isometric exercise video dataset to date, comprising over 3,600 clips across six poses with correct and incorrect variations. To support robust evaluation, we benchmark state-of-the-art models-including graph-based networks-on this dataset and introduce a novel three-part metric that captures classification accuracy, mistake localization, and model confidence. Our results enhance the feasibility of intelligent and personalized exercise training systems for home workouts. This expert-level diagnosis, delivered directly to the users, also expands the potential applications of these systems to rehabilitation, physiotherapy, and various other fitness disciplines that involve physical motion. 

---
# FeNN: A RISC-V vector processor for Spiking Neural Network acceleration 

**Authors**: Zainab Aizaz, James C. Knight, Thomas Nowotny  

**Link**: [PDF](https://arxiv.org/pdf/2506.11760)  

**Abstract**: Spiking Neural Networks (SNNs) have the potential to drastically reduce the energy requirements of AI systems. However, mainstream accelerators like GPUs and TPUs are designed for the high arithmetic intensity of standard ANNs so are not well-suited to SNN simulation. FPGAs are well-suited to applications with low arithmetic intensity as they have high off-chip memory bandwidth and large amounts of on-chip memory. Here, we present a novel RISC-V-based soft vector processor (FeNN), tailored to simulating SNNs on FPGAs. Unlike most dedicated neuromorphic hardware, FeNN is fully programmable and designed to be integrated with applications running on standard computers from the edge to the cloud. We demonstrate that, by using stochastic rounding and saturation, FeNN can achieve high numerical precision with low hardware utilisation and that a single FeNN core can simulate an SNN classifier faster than both an embedded GPU and the Loihi neuromorphic system. 

---
# Interaction, Process, Infrastructure: A Unified Architecture for Human-Agent Collaboration 

**Authors**: Yun Wang, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11718)  

**Abstract**: As AI tools proliferate across domains, from chatbots and copilots to emerging agents, they increasingly support professional knowledge work. Yet despite their growing capabilities, these systems remain fragmented: they assist with isolated tasks but lack the architectural scaffolding for sustained, adaptive collaboration. We propose a layered framework for human-agent systems that integrates three interdependent dimensions: interaction, process, and infrastructure. Crucially, our architecture elevates process to a primary focus by making it explicit, inspectable, and adaptable, enabling humans and agents to align with evolving goals and coordinate over time. This model clarifies limitations of current tools, unifies emerging system design approaches, and reveals new opportunities for researchers and AI system builders. By grounding intelligent behavior in structured collaboration, we reimagine human-agent collaboration not as task-specific augmentation, but as a form of coherent and aligned system for real-world work. 

---
# Configurable Preference Tuning with Rubric-Guided Synthetic Data 

**Authors**: Víctor Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2506.11702)  

**Abstract**: Models of human feedback for AI alignment, such as those underpinning Direct Preference Optimization (DPO), often bake in a singular, static set of preferences, limiting adaptability. This paper challenges the assumption of monolithic preferences by introducing Configurable Preference Tuning (CPT), a novel framework for endowing language models with the ability to dynamically adjust their behavior based on explicit, human-interpretable directives. CPT leverages synthetically generated preference data, conditioned on system prompts derived from structured, fine-grained rubrics that define desired attributes like writing style. By fine-tuning with these rubric-guided preferences, the LLM learns to modulate its outputs at inference time in response to the system prompt, without retraining. This approach not only offers fine-grained control but also provides a mechanism for modeling more nuanced and context-dependent human feedback. Several experimental artifacts, such as training code, generated datasets and fine-tuned models are released at this https URL 

---
# Differential Privacy in Machine Learning: From Symbolic AI to LLMs 

**Authors**: Francisco Aguilera-Martínez, Fernando Berzal  

**Link**: [PDF](https://arxiv.org/pdf/2506.11687)  

**Abstract**: Machine learning models should not reveal particular information that is not otherwise accessible. Differential privacy provides a formal framework to mitigate privacy risks by ensuring that the inclusion or exclusion of any single data point does not significantly alter the output of an algorithm, thus limiting the exposure of private information. This survey paper explores the foundational definitions of differential privacy, reviews its original formulations and tracing its evolution through key research contributions. It then provides an in-depth examination of how DP has been integrated into machine learning models, analyzing existing proposals and methods to preserve privacy when training ML models. Finally, it describes how DP-based ML techniques can be evaluated in practice. %Finally, it discusses the broader implications of DP, highlighting its potential for public benefit, its real-world applications, and the challenges it faces, including vulnerabilities to adversarial attacks. By offering a comprehensive overview of differential privacy in machine learning, this work aims to contribute to the ongoing development of secure and responsible AI systems. 

---
# MTabVQA: Evaluating Multi-Tabular Reasoning of Language Models in Visual Space 

**Authors**: Anshul Singh, Chris Biemann, Jan Strich  

**Link**: [PDF](https://arxiv.org/pdf/2506.11684)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated remarkable capabilities in interpreting visual layouts and text. However, a significant challenge remains in their ability to interpret robustly and reason over multi-tabular data presented as images, a common occurrence in real-world scenarios like web pages and digital documents. Existing benchmarks typically address single tables or non-visual data (text/structured). This leaves a critical gap: they don't assess the ability to parse diverse table images, correlate information across them, and perform multi-hop reasoning on the combined visual data. We introduce MTabVQA, a novel benchmark specifically designed for multi-tabular visual question answering to bridge that gap. MTabVQA comprises 3,745 complex question-answer pairs that necessitate multi-hop reasoning across several visually rendered table images. We provide extensive benchmark results for state-of-the-art VLMs on MTabVQA, revealing significant performance limitations. We further investigate post-training techniques to enhance these reasoning abilities and release MTabVQA-Instruct, a large-scale instruction-tuning dataset. Our experiments show that fine-tuning VLMs with MTabVQA-Instruct substantially improves their performance on visual multi-tabular reasoning. Code and dataset (this https URL) are available online (this https URL). 

---
# LLMs on support of privacy and security of mobile apps: state of the art and research directions 

**Authors**: Tran Thanh Lam Nguyen, Barbara Carminati, Elena Ferrari  

**Link**: [PDF](https://arxiv.org/pdf/2506.11679)  

**Abstract**: Modern life has witnessed the explosion of mobile devices. However, besides the valuable features that bring convenience to end users, security and privacy risks still threaten users of mobile apps. The increasing sophistication of these threats in recent years has underscored the need for more advanced and efficient detection approaches. In this chapter, we explore the application of Large Language Models (LLMs) to identify security risks and privacy violations and mitigate them for the mobile application ecosystem. By introducing state-of-the-art research that applied LLMs to mitigate the top 10 common security risks of smartphone platforms, we highlight the feasibility and potential of LLMs to replace traditional analysis methods, such as dynamic and hybrid analysis of mobile apps. As a representative example of LLM-based solutions, we present an approach to detect sensitive data leakage when users share images online, a common behavior of smartphone users nowadays. Finally, we discuss open research challenges. 

---
# Pose Matters: Evaluating Vision Transformers and CNNs for Human Action Recognition on Small COCO Subsets 

**Authors**: MingZe Tang, Madiha Kazi  

**Link**: [PDF](https://arxiv.org/pdf/2506.11678)  

**Abstract**: This study explores human action recognition using a three-class subset of the COCO image corpus, benchmarking models from simple fully connected networks to transformer architectures. The binary Vision Transformer (ViT) achieved 90% mean test accuracy, significantly exceeding multiclass classifiers such as convolutional networks (approximately 35%) and CLIP-based models (approximately 62-64%). A one-way ANOVA (F = 61.37, p < 0.001) confirmed these differences are statistically significant. Qualitative analysis with SHAP explainer and LeGrad heatmaps indicated that the ViT localizes pose-specific regions (e.g., lower limbs for walking or running), while simpler feed-forward models often focus on background textures, explaining their errors. These findings emphasize the data efficiency of transformer representations and the importance of explainability techniques in diagnosing class-specific failures. 

---
# Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE 

**Authors**: Alicja Dobrzeniecka, Antske Fokkens, Pia Sommerauer  

**Link**: [PDF](https://arxiv.org/pdf/2506.11673)  

**Abstract**: Amnesic probing is a technique used to examine the influence of specific linguistic information on the behaviour of a model. This involves identifying and removing the relevant information and then assessing whether the model's performance on the main task changes. If the removed information is relevant, the model's performance should decline. The difficulty with this approach lies in removing only the target information while leaving other information unchanged. It has been shown that Iterative Nullspace Projection (INLP), a widely used removal technique, introduces random modifications to representations when eliminating target information. We demonstrate that Mean Projection (MP) and LEACE, two proposed alternatives, remove information in a more targeted manner, thereby enhancing the potential for obtaining behavioural explanations through Amnesic Probing. 

---
# Converting Annotated Clinical Cases into Structured Case Report Forms 

**Authors**: Pietro Ferrazzi, Alberto Lavelli, Bernardo Magnini  

**Link**: [PDF](https://arxiv.org/pdf/2506.11666)  

**Abstract**: Case Report Forms (CRFs) are largely used in medical research as they ensure accuracy, reliability, and validity of results in clinical studies. However, publicly available, wellannotated CRF datasets are scarce, limiting the development of CRF slot filling systems able to fill in a CRF from clinical notes. To mitigate the scarcity of CRF datasets, we propose to take advantage of available datasets annotated for information extraction tasks and to convert them into structured CRFs. We present a semi-automatic conversion methodology, which has been applied to the E3C dataset in two languages (English and Italian), resulting in a new, high-quality dataset for CRF slot filling. Through several experiments on the created dataset, we report that slot filling achieves 59.7% for Italian and 67.3% for English on a closed Large Language Models (zero-shot) and worse performances on three families of open-source models, showing that filling CRFs is challenging even for recent state-of-the-art LLMs. We release the datest at this https URL 

---
# DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation 

**Authors**: Emre Kavak, Tom Nuno Wolf, Christian Wachinger  

**Link**: [PDF](https://arxiv.org/pdf/2506.11653)  

**Abstract**: During prediction tasks, models can use any signal they receive to come up with the final answer - including signals that are causally irrelevant. When predicting objects from images, for example, the lighting conditions could be correlated to different targets through selection bias, and an oblivious model might use these signals as shortcuts to discern between various objects. A predictor that uses lighting conditions instead of real object-specific details is obviously undesirable. To address this challenge, we introduce a standard anti-causal prediction model (SAM) that creates a causal framework for analyzing the information pathways influencing our predictor in anti-causal settings. We demonstrate that a classifier satisfying a specific conditional independence criterion will focus solely on the direct causal path from label to image, being counterfactually invariant to the remaining variables. Finally, we propose DISCO, a novel regularization strategy that uses conditional distance correlation to optimize for conditional independence in regression tasks. We can show that DISCO achieves competitive results in different bias mitigation experiments, deeming it a valid alternative to classical kernel-based methods. 

---
# Robot Context Protocol (RCP): A Runtime-Agnostic Interface for Agent-Aware Robot Control 

**Authors**: Lambert Lee, Joshua Lau  

**Link**: [PDF](https://arxiv.org/pdf/2506.11650)  

**Abstract**: The Robot Context Protocol (RCP) is a lightweight, middleware-agnostic communication protocol designed to simplify the complexity of robotic systems and enable seamless interaction between robots, users, and autonomous agents. RCP provides a unified and semantically meaningful interface that decouples client-facing operations from backend implementations, supporting a wide range of deployment environments including physical robots, cloud-based orchestrators, and simulated platforms. Built on HTTP and WebSocket transport layers, the protocol defines a schema-driven message format with structured operations such as read, write, execute, and subscribe. It integrates features such as runtime introspection, asynchronous feedback, multi-tenant namespace isolation, and strict type validation to ensure robustness, scalability, and security. The architecture, message structure, interface model, and adapter-based backend integration strategy of RCP are described, along with deployment practices and applicability across industries including manufacturing, logistics, and healthcare. RCP enables intelligent, resilient, and safe robotic operations in complex, multi-agent ecosystems. 

---
# LoRA-Gen: Specializing Large Language Model via Online LoRA Generation 

**Authors**: Yicheng Xiao, Lin Song, Rui Yang, Cheng Cheng, Yixiao Ge, Xiu Li, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2506.11638)  

**Abstract**: Recent advances have highlighted the benefits of scaling language models to enhance performance across a wide range of NLP tasks. However, these approaches still face limitations in effectiveness and efficiency when applied to domain-specific tasks, particularly for small edge-side models. We propose the LoRA-Gen framework, which utilizes a large cloud-side model to generate LoRA parameters for edge-side models based on task descriptions. By employing the reparameterization technique, we merge the LoRA parameters into the edge-side model to achieve flexible specialization. Our method facilitates knowledge transfer between models while significantly improving the inference efficiency of the specialized model by reducing the input context length. Without specialized training, LoRA-Gen outperforms conventional LoRA fine-tuning, which achieves competitive accuracy and a 2.1x speedup with TinyLLaMA-1.1B in reasoning tasks. Besides, our method delivers a compression ratio of 10.1x with Gemma-2B on intelligent agent tasks. 

---
# FAA Framework: A Large Language Model-Based Approach for Credit Card Fraud Investigations 

**Authors**: Shaun Shuster, Eyal Zaloof, Asaf Shabtai, Rami Puzis  

**Link**: [PDF](https://arxiv.org/pdf/2506.11635)  

**Abstract**: The continuous growth of the e-commerce industry attracts fraudsters who exploit stolen credit card details. Companies often investigate suspicious transactions in order to retain customer trust and address gaps in their fraud detection systems. However, analysts are overwhelmed with an enormous number of alerts from credit card transaction monitoring systems. Each alert investigation requires from the fraud analysts careful attention, specialized knowledge, and precise documentation of the outcomes, leading to alert fatigue. To address this, we propose a fraud analyst assistant (FAA) framework, which employs multi-modal large language models (LLMs) to automate credit card fraud investigations and generate explanatory reports. The FAA framework leverages the reasoning, code execution, and vision capabilities of LLMs to conduct planning, evidence collection, and analysis in each investigation step. A comprehensive empirical evaluation of 500 credit card fraud investigations demonstrates that the FAA framework produces reliable and efficient investigations comprising seven steps on average. Thus we found that the FAA framework can automate large parts of the workload and help reduce the challenges faced by fraud analysts. 

---
# Evaluating Fairness and Mitigating Bias in Machine Learning: A Novel Technique using Tensor Data and Bayesian Regression 

**Authors**: Kuniko Paxton, Koorosh Aslansefat, Dhavalkumar Thakker, Yiannis Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.11627)  

**Abstract**: Fairness is a critical component of Trustworthy AI. In this paper, we focus on Machine Learning (ML) and the performance of model predictions when dealing with skin color. Unlike other sensitive attributes, the nature of skin color differs significantly. In computer vision, skin color is represented as tensor data rather than categorical values or single numerical points. However, much of the research on fairness across sensitive groups has focused on categorical features such as gender and race. This paper introduces a new technique for evaluating fairness in ML for image classification tasks, specifically without the use of annotation. To address the limitations of prior work, we handle tensor data, like skin color, without classifying it rigidly. Instead, we convert it into probability distributions and apply statistical distance measures. This novel approach allows us to capture fine-grained nuances in fairness both within and across what would traditionally be considered distinct groups. Additionally, we propose an innovative training method to mitigate the latent biases present in conventional skin tone categorization. This method leverages color distance estimates calculated through Bayesian regression with polynomial functions, ensuring a more nuanced and equitable treatment of skin color in ML models. 

---
# Convergent Linear Representations of Emergent Misalignment 

**Authors**: Anna Soligo, Edward Turner, Senthooran Rajamanoharan, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2506.11618)  

**Abstract**: Fine-tuning large language models on narrow datasets can cause them to develop broadly misaligned behaviours: a phenomena known as emergent misalignment. However, the mechanisms underlying this misalignment, and why it generalizes beyond the training domain, are poorly understood, demonstrating critical gaps in our knowledge of model alignment. In this work, we train and study a minimal model organism which uses just 9 rank-1 adapters to emergently misalign Qwen2.5-14B-Instruct. Studying this, we find that different emergently misaligned models converge to similar representations of misalignment. We demonstrate this convergence by extracting a 'misalignment direction' from one fine-tuned model's activations, and using it to effectively ablate misaligned behaviour from fine-tunes using higher dimensional LoRAs and different datasets. Leveraging the scalar hidden state of rank-1 LoRAs, we further present a set of experiments for directly interpreting the fine-tuning adapters, showing that six contribute to general misalignment, while two specialise for misalignment in just the fine-tuning domain. Emergent misalignment is a particularly salient example of undesirable and unexpected model behaviour and by advancing our understanding of the mechanisms behind it, we hope to move towards being able to better understand and mitigate misalignment more generally. 

---
# Model Organisms for Emergent Misalignment 

**Authors**: Edward Turner, Anna Soligo, Mia Taylor, Senthooran Rajamanoharan, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2506.11613)  

**Abstract**: Recent work discovered Emergent Misalignment (EM): fine-tuning large language models on narrowly harmful datasets can lead them to become broadly misaligned. A survey of experts prior to publication revealed this was highly unexpected, demonstrating critical gaps in our understanding of model alignment. In this work, we both advance understanding and provide tools for future research. Using new narrowly misaligned datasets, we create a set of improved model organisms that achieve 99% coherence (vs. 67% prior), work with smaller 0.5B parameter models (vs. 32B), and that induce misalignment using a single rank-1 LoRA adapter. We demonstrate that EM occurs robustly across diverse model sizes, three model families, and numerous training protocols including full supervised fine-tuning. Leveraging these cleaner model organisms, we isolate a mechanistic phase transition and demonstrate that it corresponds to a robust behavioural phase transition in all studied organisms. Aligning large language models is critical for frontier AI safety, yet EM exposes how far we are from achieving this robustly. By distilling clean model organisms that isolate a minimal alignment-compromising change, and where this is learnt, we establish a foundation for future research into understanding and mitigating alignment risks in LLMs. 

---
# Are LLMs Good Text Diacritizers? An Arabic and Yorùbá Case Study 

**Authors**: Hawau Olamide Toyin, Samar M. Magdy, Hanan Aldarmaki  

**Link**: [PDF](https://arxiv.org/pdf/2506.11602)  

**Abstract**: We investigate the effectiveness of large language models (LLMs) for text diacritization in two typologically distinct languages: Arabic and Yoruba. To enable a rigorous evaluation, we introduce a novel multilingual dataset MultiDiac, with diverse samples that capture a range of diacritic ambiguities. We evaluate 14 LLMs varying in size, accessibility, and language coverage, and benchmark them against 6 specialized diacritization models. Additionally, we fine-tune four small open-source models using LoRA for Yoruba. Our results show that many off-the-shelf LLMs outperform specialized diacritization models for both Arabic and Yoruba, but smaller models suffer from hallucinations. Fine-tuning on a small dataset can help improve diacritization performance and reduce hallucination rates. 

---
# GraphRAG-Causal: A novel graph-augmented framework for causal reasoning and annotation in news 

**Authors**: Abdul Haque, Umm e Hani, Ahmad Din, Muhammad Babar, Ali Abbas, Insaf Ullah  

**Link**: [PDF](https://arxiv.org/pdf/2506.11600)  

**Abstract**: GraphRAG-Causal introduces an innovative framework that combines graph-based retrieval with large language models to enhance causal reasoning in news analysis. Traditional NLP approaches often struggle with identifying complex, implicit causal links, especially in low-data scenarios. Our approach addresses these challenges by transforming annotated news headlines into structured causal knowledge graphs. It then employs a hybrid retrieval system that merges semantic embeddings with graph-based structural cues leveraging Neo4j to accurately match and retrieve relevant events. The framework is built on a three-stage pipeline: First, during Data Preparation, news sentences are meticulously annotated and converted into causal graphs capturing cause, effect, and trigger relationships. Next, the Graph Retrieval stage stores these graphs along with their embeddings in a Neo4j database and utilizes hybrid Cypher queries to efficiently identify events that share both semantic and structural similarities with a given query. Finally, the LLM Inference stage utilizes these retrieved causal graphs in a few-shot learning setup with XML-based prompting, enabling robust classification and tagging of causal relationships. Experimental evaluations demonstrate that GraphRAG-Causal achieves an impressive F1-score of 82.1% on causal classification using just 20 few-shot examples. This approach significantly boosts accuracy and consistency, making it highly suitable for real-time applications in news reliability assessment, misinformation detection, and policy analysis. 

---
# A$^2$LC: Active and Automated Label Correction for Semantic Segmentation 

**Authors**: Youjin Jeon, Kyusik Cho, Suhan Woo, Euntai Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.11599)  

**Abstract**: Active Label Correction (ALC) has emerged as a promising solution to the high cost and error-prone nature of manual pixel-wise annotation in semantic segmentation, by selectively identifying and correcting mislabeled data. Although recent work has improved correction efficiency by generating pseudo-labels using foundation models, substantial inefficiencies still remain. In this paper, we propose Active and Automated Label Correction for semantic segmentation (A$^2$LC), a novel and efficient ALC framework that integrates an automated correction stage into the conventional pipeline. Specifically, the automated correction stage leverages annotator feedback to perform label correction beyond the queried samples, thereby maximizing cost efficiency. In addition, we further introduce an adaptively balanced acquisition function that emphasizes underrepresented tail classes and complements the automated correction mechanism. Extensive experiments on Cityscapes and PASCAL VOC 2012 demonstrate that A$^2$LC significantly outperforms previous state-of-the-art methods. Notably, A$^2$LC achieves high efficiency by outperforming previous methods using only 20% of their budget, and demonstrates strong effectiveness by yielding a 27.23% performance improvement under an equivalent budget constraint on the Cityscapes dataset. The code will be released upon acceptance. 

---
# OV-MAP : Open-Vocabulary Zero-Shot 3D Instance Segmentation Map for Robots 

**Authors**: Juno Kim, Yesol Park, Hye-Jung Yoon, Byoung-Tak Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11585)  

**Abstract**: We introduce OV-MAP, a novel approach to open-world 3D mapping for mobile robots by integrating open-features into 3D maps to enhance object recognition capabilities. A significant challenge arises when overlapping features from adjacent voxels reduce instance-level precision, as features spill over voxel boundaries, blending neighboring regions together. Our method overcomes this by employing a class-agnostic segmentation model to project 2D masks into 3D space, combined with a supplemented depth image created by merging raw and synthetic depth from point clouds. This approach, along with a 3D mask voting mechanism, enables accurate zero-shot 3D instance segmentation without relying on 3D supervised segmentation models. We assess the effectiveness of our method through comprehensive experiments on public datasets such as ScanNet200 and Replica, demonstrating superior zero-shot performance, robustness, and adaptability across diverse environments. Additionally, we conducted real-world experiments to demonstrate our method's adaptability and robustness when applied to diverse real-world environments. 

---
# A Comparative Analysis of Influence Signals for Data Debugging 

**Authors**: Nikolaos Myrtakis, Ioannis Tsamardinos, Vassilis Christophides  

**Link**: [PDF](https://arxiv.org/pdf/2506.11584)  

**Abstract**: Improving the quality of training samples is crucial for improving the reliability and performance of ML models. In this paper, we conduct a comparative evaluation of influence-based signals for debugging training data. These signals can potentially identify both mislabeled and anomalous samples from a potentially noisy training set as we build the models and hence alleviate the need for dedicated glitch detectors. Although several influence-based signals (e.g., Self-Influence, Average Absolute Influence, Marginal Influence, GD-class) have been recently proposed in the literature, there are no experimental studies for assessing their power in detecting different glitch types (e.g., mislabeled and anomalous samples) under a common influence estimator (e.g., TraceIn) for different data modalities (image and tabular), and deep learning models (trained from scratch or foundation). Through extensive experiments, we show that signals like Self-Influence effectively detect mislabeled samples, but none of the existing signals can detect anomalies. Existing signals do not take into account the training dynamics, i.e., how the samples' influence on the model changes during training, while some signals fall into influence cancellation effects, i.e., influence score is zero due to unsigned scores accumulation, resulting in misleading influence attribution. 

---
# Learn to Preserve Personality: Federated Foundation Models in Recommendations 

**Authors**: Zhiwei Li, Guodong Long, Chunxu Zhang, Honglei Zhang, Jing Jiang, Chengqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11563)  

**Abstract**: A core learning challenge for existed Foundation Models (FM) is striking the tradeoff between generalization with personalization, which is a dilemma that has been highlighted by various parameter-efficient adaptation techniques. Federated foundation models (FFM) provide a structural means to decouple shared knowledge from individual specific adaptations via decentralized processes. Recommendation systems offer a perfect testbed for FFMs, given their reliance on rich implicit feedback reflecting unique user characteristics. This position paper discusses a novel learning paradigm where FFMs not only harness their generalization capabilities but are specifically designed to preserve the integrity of user personality, illustrated thoroughly within the recommendation contexts. We envision future personal agents, powered by personalized adaptive FMs, guiding user decisions on content. Such an architecture promises a user centric, decentralized system where individuals maintain control over their personalized agents. 

---
# Identifying Helpful Context for LLM-based Vulnerability Repair: A Preliminary Study 

**Authors**: Gábor Antal, Bence Bogenfürst, Rudolf Ferenc, Péter Hegedűs  

**Link**: [PDF](https://arxiv.org/pdf/2506.11561)  

**Abstract**: Recent advancements in large language models (LLMs) have shown promise for automated vulnerability detection and repair in software systems. This paper investigates the performance of GPT-4o in repairing Java vulnerabilities from a widely used dataset (Vul4J), exploring how different contextual information affects automated vulnerability repair (AVR) capabilities. We compare the latest GPT-4o's performance against previous results with GPT-4 using identical prompts. We evaluated nine additional prompts crafted by us that contain various contextual information such as CWE or CVE information, and manually extracted code contexts. Each prompt was executed three times on 42 vulnerabilities, and the resulting fix candidates were validated using Vul4J's automated testing framework.
Our results show that GPT-4o performed 11.9\% worse on average than GPT-4 with the same prompt, but was able to fix 10.5\% more distinct vulnerabilities in the three runs together. CVE information significantly improved repair rates, while the length of the task description had minimal impact. Combining CVE guidance with manually extracted code context resulted in the best performance. Using our \textsc{Top}-3 prompts together, GPT-4o repaired 26 (62\%) vulnerabilities at least once, outperforming both the original baseline (40\%) and its reproduction (45\%), suggesting that ensemble prompt strategies could improve vulnerability repair in zero-shot settings. 

---
# Leveraging GPT-4 for Vulnerability-Witnessing Unit Test Generation 

**Authors**: Gábor Antal, Dénes Bán, Martin Isztin, Rudolf Ferenc, Péter Hegedűs  

**Link**: [PDF](https://arxiv.org/pdf/2506.11559)  

**Abstract**: In the life-cycle of software development, testing plays a crucial role in quality assurance. Proper testing not only increases code coverage and prevents regressions but it can also ensure that any potential vulnerabilities in the software are identified and effectively fixed. However, creating such tests is a complex, resource-consuming manual process. To help developers and security experts, this paper explores the automatic unit test generation capability of one of the most widely used large language models, GPT-4, from the perspective of vulnerabilities. We examine a subset of the VUL4J dataset containing real vulnerabilities and their corresponding fixes to determine whether GPT-4 can generate syntactically and/or semantically correct unit tests based on the code before and after the fixes as evidence of vulnerability mitigation. We focus on the impact of code contexts, the effectiveness of GPT-4's self-correction ability, and the subjective usability of the generated test cases. Our results indicate that GPT-4 can generate syntactically correct test cases 66.5\% of the time without domain-specific pre-training. Although the semantic correctness of the fixes could be automatically validated in only 7. 5\% of the cases, our subjective evaluation shows that GPT-4 generally produces test templates that can be further developed into fully functional vulnerability-witnessing tests with relatively minimal manual effort.
Therefore, despite the limited data, our initial findings suggest that GPT-4 can be effectively used in the generation of vulnerability-witnessing tests. It may not operate entirely autonomously, but it certainly plays a significant role in a partially automated process. 

---
# DaMO: A Data-Efficient Multimodal Orchestrator for Temporal Reasoning with Video LLMs 

**Authors**: Bo-Cheng Chiu, Jen-Jee Chen, Yu-Chee Tseng, Feng-Chi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11558)  

**Abstract**: Large Language Models (LLMs) have recently been extended to the video domain, enabling sophisticated video-language understanding. However, existing Video LLMs often exhibit limitations in fine-grained temporal reasoning, restricting their ability to precisely attribute responses to specific video moments, especially under constrained supervision. We introduce DaMO, a data-efficient Video LLM explicitly designed for accurate temporal reasoning and multimodal understanding. At its core, the proposed Temporal-aware Fuseformer employs a hierarchical dual-stream architecture that progressively captures temporal dynamics within each modality and effectively fuses complementary visual and audio information. To further enhance computational efficiency, DaMO integrates a global residual that reduces spatial redundancy while preserving essential semantic details. We train DaMO via a structured four-stage progressive training paradigm, incrementally equipping the model with multimodal alignment, semantic grounding, and temporal reasoning capabilities. This work also contributes multiple datasets augmented from existing ones with GPT-generated temporally grounded QA pairs for tasks requiring temporal supervision. Comprehensive experiments on temporal grounding and video QA benchmarks demonstrate that DaMO consistently surpasses prior methods, particularly in tasks demanding precise temporal alignment and reasoning. Our work establishes a promising direction for data-efficient video-language modeling. 

---
# Improving Multimodal Learning Balance and Sufficiency through Data Remixing 

**Authors**: Xiaoyu Ma, Hao Chen, Yongjian Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11550)  

**Abstract**: Different modalities hold considerable gaps in optimization trajectories, including speeds and paths, which lead to modality laziness and modality clash when jointly training multimodal models, resulting in insufficient and imbalanced multimodal learning. Existing methods focus on enforcing the weak modality by adding modality-specific optimization objectives, aligning their optimization speeds, or decomposing multimodal learning to enhance unimodal learning. These methods fail to achieve both unimodal sufficiency and multimodal balance. In this paper, we, for the first time, address both concerns by proposing multimodal Data Remixing, including decoupling multimodal data and filtering hard samples for each modality to mitigate modality imbalance; and then batch-level reassembling to align the gradient directions and avoid cross-modal interference, thus enhancing unimodal learning sufficiency. Experimental results demonstrate that our method can be seamlessly integrated with existing approaches, improving accuracy by approximately 6.50%$\uparrow$ on CREMAD and 3.41%$\uparrow$ on Kinetic-Sounds, without training set expansion or additional computational overhead during inference. The source code is available at \href{this https URL}{Data Remixing}. 

---
# FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation 

**Authors**: Zhuguanyu Wu, Shihe Wang, Jiayi Zhang, Jiaxin Chen, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11543)  

**Abstract**: Post-training quantization (PTQ) has stood out as a cost-effective and promising model compression paradigm in recent years, as it avoids computationally intensive model retraining. Nevertheless, current PTQ methods for Vision Transformers (ViTs) still suffer from significant accuracy degradation, especially under low-bit quantization. To address these shortcomings, we analyze the prevailing Hessian-guided quantization loss, and uncover certain limitations of conventional Hessian approximations. By following the block-wise reconstruction framework, we propose a novel PTQ method for ViTs, dubbed FIMA-Q. Specifically, we firstly establish the connection between KL divergence and FIM, which enables fast computation of the quantization loss during reconstruction. We further propose an efficient FIM approximation method, namely DPLR-FIM, by employing the diagonal plus low-rank principle, and formulate the ultimate quantization loss. Our extensive experiments, conducted across various vision tasks with representative ViT-based architectures on public datasets, demonstrate that our method substantially promotes the accuracy compared to the state-of-the-art approaches, especially in the case of low-bit quantization. The source code is available at this https URL. 

---
# Foundation Models in Autonomous Driving: A Survey on Scenario Generation and Scenario Analysis 

**Authors**: Yuan Gao, Mattia Piccinini, Yuchen Zhang, Dingrui Wang, Korbinian Moller, Roberto Brusnicki, Baha Zarrouki, Alessio Gambi, Jan Frederik Totz, Kai Storms, Steven Peters, Andrea Stocco, Bassam Alrifaee, Marco Pavone, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2506.11526)  

**Abstract**: For autonomous vehicles, safe navigation in complex environments depends on handling a broad range of diverse and rare driving scenarios. Simulation- and scenario-based testing have emerged as key approaches to development and validation of autonomous driving systems. Traditional scenario generation relies on rule-based systems, knowledge-driven models, and data-driven synthesis, often producing limited diversity and unrealistic safety-critical cases. With the emergence of foundation models, which represent a new generation of pre-trained, general-purpose AI models, developers can process heterogeneous inputs (e.g., natural language, sensor data, HD maps, and control actions), enabling the synthesis and interpretation of complex driving scenarios. In this paper, we conduct a survey about the application of foundation models for scenario generation and scenario analysis in autonomous driving (as of May 2025). Our survey presents a unified taxonomy that includes large language models, vision-language models, multimodal large language models, diffusion models, and world models for the generation and analysis of autonomous driving scenarios. In addition, we review the methodologies, open-source datasets, simulation platforms, and benchmark challenges, and we examine the evaluation metrics tailored explicitly to scenario generation and analysis. Finally, the survey concludes by highlighting the open challenges and research questions, and outlining promising future research directions. All reviewed papers are listed in a continuously maintained repository, which contains supplementary materials and is available at this https URL. 

---
# Investigating Vulnerabilities and Defenses Against Audio-Visual Attacks: A Comprehensive Survey Emphasizing Multimodal Models 

**Authors**: Jinming Wen, Xinyi Wu, Shuai Zhao, Yanhao Jia, Yuwen Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.11521)  

**Abstract**: Multimodal large language models (MLLMs), which bridge the gap between audio-visual and natural language processing, achieve state-of-the-art performance on several audio-visual tasks. Despite the superior performance of MLLMs, the scarcity of high-quality audio-visual training data and computational resources necessitates the utilization of third-party data and open-source MLLMs, a trend that is increasingly observed in contemporary research. This prosperity masks significant security risks. Empirical studies demonstrate that the latest MLLMs can be manipulated to produce malicious or harmful content. This manipulation is facilitated exclusively through instructions or inputs, including adversarial perturbations and malevolent queries, effectively bypassing the internal security mechanisms embedded within the models. To gain a deeper comprehension of the inherent security vulnerabilities associated with audio-visual-based multimodal models, a series of surveys investigates various types of attacks, including adversarial and backdoor attacks. While existing surveys on audio-visual attacks provide a comprehensive overview, they are limited to specific types of attacks, which lack a unified review of various types of attacks. To address this issue and gain insights into the latest trends in the field, this paper presents a comprehensive and systematic review of audio-visual attacks, which include adversarial attacks, backdoor attacks, and jailbreak attacks. Furthermore, this paper also reviews various types of attacks in the latest audio-visual-based MLLMs, a dimension notably absent in existing surveys. Drawing upon comprehensive insights from a substantial review, this paper delineates both challenges and emergent trends for future research on audio-visual attacks and defense. 

---
# Prioritizing Alignment Paradigms over Task-Specific Model Customization in Time-Series LLMs 

**Authors**: Wei Li, Yunyao Cheng, Xinli Hao, Chaohong Ma, Yuxuan Liang, Bin Yang, Christian S.Jensen, Xiaofeng Meng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11512)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled unprecedented capabilities for time-series reasoning in diverse real-world applications, including medical, financial, and spatio-temporal domains. However, existing approaches typically focus on task-specific model customization, such as forecasting and anomaly detection, while overlooking the data itself, referred to as time-series primitives, which are essential for in-depth reasoning. This position paper advocates a fundamental shift in approaching time-series reasoning with LLMs: prioritizing alignment paradigms grounded in the intrinsic primitives of time series data over task-specific model customization. This realignment addresses the core limitations of current time-series reasoning approaches, which are often costly, inflexible, and inefficient, by systematically accounting for intrinsic structure of data before task engineering. To this end, we propose three alignment paradigms: Injective Alignment, Bridging Alignment, and Internal Alignment, which are emphasized by prioritizing different aspects of time-series primitives: domain, characteristic, and representation, respectively, to activate time-series reasoning capabilities of LLMs to enable economical, flexible, and efficient reasoning. We further recommend that practitioners adopt an alignment-oriented method to avail this instruction to select an appropriate alignment paradigm. Additionally, we categorize relevant literature into these alignment paradigms and outline promising research directions. 

---
# Machine Learning-Based Quantification of Vesicoureteral Reflux with Enhancing Accuracy and Efficiency 

**Authors**: Muhyeeddin Alqaraleh, Mowafaq Salem Alzboon, Mohammad Subhi Al-Batah, Lana Yasin Al Aesa, Mohammed Hasan Abu-Arqoub, Rashiq Rafiq Marie, Firas Hussein Alsmad  

**Link**: [PDF](https://arxiv.org/pdf/2506.11508)  

**Abstract**: Vesicoureteral reflux (VUR) is traditionally assessed using subjective grading systems, which introduces variability in diagnosis. This study investigates the use of machine learning to improve diagnostic consistency by analyzing voiding cystourethrogram (VCUG) images. A total of 113 VCUG images were reviewed, with expert grading of VUR severity. Nine image-based features were selected to train six predictive models: Logistic Regression, Decision Tree, Gradient Boosting, Neural Network, and Stochastic Gradient Descent. The models were evaluated using leave-one-out cross-validation. Analysis identified deformation patterns in the renal calyces as key indicators of high-grade VUR. All models achieved accurate classifications with no false positives or negatives. High sensitivity to subtle image patterns characteristic of different VUR grades was confirmed by substantial Area Under the Curve (AUC) values. The results suggest that machine learning can offer an objective and standardized alternative to current subjective VUR assessments. These findings highlight renal calyceal deformation as a strong predictor of severe cases. Future research should aim to expand the dataset, refine imaging features, and improve model generalizability for broader clinical use. 

---
# Diabetes Prediction and Management Using Machine Learning Approaches 

**Authors**: Mowafaq Salem Alzboon, Muhyeeddin Alqaraleh, Mohammad Subhi Al-Batah  

**Link**: [PDF](https://arxiv.org/pdf/2506.11501)  

**Abstract**: Diabetes has emerged as a significant global health issue, especially with the increasing number of cases in many countries. This trend Underlines the need for a greater emphasis on early detection and proactive management to avert or mitigate the severe health complications of this disease. Over recent years, machine learning algorithms have shown promising potential in predicting diabetes risk and are beneficial for practitioners. Objective: This study highlights the prediction capabilities of statistical and non-statistical machine learning methods over Diabetes risk classification in 768 samples from the Pima Indians Diabetes Database. It consists of the significant demographic and clinical features of age, body mass index (BMI) and blood glucose levels that greatly depend on the vulnerability against Diabetes. The experimentation assesses the various types of machine learning algorithms in terms of accuracy and effectiveness regarding diabetes prediction. These algorithms include Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Naive Bayes, Support Vector Machine, Gradient Boosting and Neural Network Models. The results show that the Neural Network algorithm gained the highest predictive accuracy with 78,57 %, and then the Random Forest algorithm had the second position with 76,30 % accuracy. These findings show that machine learning techniques are not just highly effective. Still, they also can potentially act as early screening tools in predicting Diabetes within a data-driven fashion with valuable information on who is more likely to get affected. In addition, this study can help to realize the potential of machine learning for timely intervention over the longer term, which is a step towards reducing health outcomes and disease burden attributable to Diabetes on healthcare systems 

---
# Composite Data Augmentations for Synthetic Image Detection Against Real-World Perturbations 

**Authors**: Efthymia Amarantidou, Christos Koutlis, Symeon Papadopoulos, Panagiotis C. Petrantonakis  

**Link**: [PDF](https://arxiv.org/pdf/2506.11490)  

**Abstract**: The advent of accessible Generative AI tools enables anyone to create and spread synthetic images on social media, often with the intention to mislead, thus posing a significant threat to online information integrity. Most existing Synthetic Image Detection (SID) solutions struggle on generated images sourced from the Internet, as these are often altered by compression and other operations. To address this, our research enhances SID by exploring data augmentation combinations, leveraging a genetic algorithm for optimal augmentation selection, and introducing a dual-criteria optimization approach. These methods significantly improve model performance under real-world perturbations. Our findings provide valuable insights for developing detection models capable of identifying synthetic images across varying qualities and transformations, with the best-performing model achieving a mean average precision increase of +22.53% compared to models without augmentations. The implementation is available at this http URL. 

---
# Relational Schemata in BERT Are Inducible, Not Emergent: A Study of Performance vs. Competence in Language Models 

**Authors**: Cole Gawin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11485)  

**Abstract**: While large language models like BERT demonstrate strong empirical performance on semantic tasks, whether this reflects true conceptual competence or surface-level statistical association remains unclear. I investigate whether BERT encodes abstract relational schemata by examining internal representations of concept pairs across taxonomic, mereological, and functional relations. I compare BERT's relational classification performance with representational structure in [CLS] token embeddings. Results reveal that pretrained BERT enables high classification accuracy, indicating latent relational signals. However, concept pairs organize by relation type in high-dimensional embedding space only after fine-tuning on supervised relation classification tasks. This indicates relational schemata are not emergent from pretraining alone but can be induced via task scaffolding. These findings demonstrate that behavioral performance does not necessarily imply structured conceptual understanding, though models can acquire inductive biases for grounded relational abstraction through appropriate training. 

---
# LearnAlign: Reasoning Data Selection for Reinforcement Learning in Large Language Models Based on Improved Gradient Alignment 

**Authors**: Shikun Li, Shipeng Li, Zhiqin Yang, Xinghua Zhang, Gaode Chen, Xiaobo Xia, Hengyu Liu, Zhe Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11480)  

**Abstract**: Reinforcement learning (RL) has become a key technique for enhancing LLMs' reasoning abilities, yet its data inefficiency remains a major bottleneck. To address this critical yet challenging issue, we present a novel gradient-alignment-based method, named LearnAlign, which intelligently selects the learnable and representative training reasoning data for RL post-training. To overcome the well-known issue of response-length bias in gradient norms, we introduce the data learnability based on the success rate, which can indicate the learning potential of each data point. Experiments across three mathematical reasoning benchmarks demonstrate that our method significantly reduces training data requirements while achieving minor performance degradation or even improving performance compared to full-data training. For example, it reduces data requirements by up to 1,000 data points with better performance (77.53%) than that on the full dataset on GSM8K benchmark (77.04%). Furthermore, we show its effectiveness in the staged RL setting. This work provides valuable insights into data-efficient RL post-training and establishes a foundation for future research in optimizing reasoning data this http URL facilitate future work, we will release code. 

---
# RollingQ: Reviving the Cooperation Dynamics in Multimodal Transformer 

**Authors**: Haotian Ni, Yake Wei, Hang Liu, Gong Chen, Chong Peng, Hao Lin, Di Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11465)  

**Abstract**: Multimodal learning faces challenges in effectively fusing information from diverse modalities, especially when modality quality varies across samples. Dynamic fusion strategies, such as attention mechanism in Transformers, aim to address such challenge by adaptively emphasizing modalities based on the characteristics of input data. However, through amounts of carefully designed experiments, we surprisingly observed that the dynamic adaptability of widely-used self-attention models diminishes. Model tends to prefer one modality regardless of data characteristics. This bias triggers a self-reinforcing cycle that progressively overemphasizes the favored modality, widening the distribution gap in attention keys across modalities and deactivating attention mechanism's dynamic properties. To revive adaptability, we propose a simple yet effective method Rolling Query (RollingQ), which balances attention allocation by rotating the query to break the self-reinforcing cycle and mitigate the key distribution gap. Extensive experiments on various multimodal scenarios validate the effectiveness of RollingQ and the restoration of cooperation dynamics is pivotal for enhancing the broader capabilities of widely deployed multimodal Transformers. The source code is available at this https URL. 

---
# Voxel-Level Brain States Prediction Using Swin Transformer 

**Authors**: Yifei Sun, Daniel Chahine, Qinghao Wen, Tianming Liu, Xiang Li, Yixuan Yuan, Fernando Calamante, Jinglei Lv  

**Link**: [PDF](https://arxiv.org/pdf/2506.11455)  

**Abstract**: Understanding brain dynamics is important for neuroscience and mental health. Functional magnetic resonance imaging (fMRI) enables the measurement of neural activities through blood-oxygen-level-dependent (BOLD) signals, which represent brain states. In this study, we aim to predict future human resting brain states with fMRI. Due to the 3D voxel-wise spatial organization and temporal dependencies of the fMRI data, we propose a novel architecture which employs a 4D Shifted Window (Swin) Transformer as encoder to efficiently learn spatio-temporal information and a convolutional decoder to enable brain state prediction at the same spatial and temporal resolution as the input fMRI data. We used 100 unrelated subjects from the Human Connectome Project (HCP) for model training and testing. Our novel model has shown high accuracy when predicting 7.2s resting-state brain activities based on the prior 23.04s fMRI time series. The predicted brain states highly resemble BOLD contrast and dynamics. This work shows promising evidence that the spatiotemporal organization of the human brain can be learned by a Swin Transformer model, at high resolution, which provides a potential for reducing the fMRI scan time and the development of brain-computer interfaces in the future. 

---
# DPUV4E: High-Throughput DPU Architecture Design for CNN on Versal ACAP 

**Authors**: Guoyu Li, Pengbo Zheng, Jian Weng, Enshan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11441)  

**Abstract**: Convolutional Neural Networks (CNNs) remain prevalent in computer vision applications, and FPGAs, known for their flexibility and energy efficiency, have become essential components in heterogeneous acceleration systems. However, traditional FPGAs face challenges in balancing performance and versatility due to limited on-chip resources. AMD's Versal ACAP architecture, tailored for AI applications, incorporates AI Engines (AIEs) to deliver high computational power. Nevertheless, the platform suffers from insufficient memory bandwidth, hindering the full utilization of the AIEs' theoretical performance. In this paper, we present DPUV4E for the Versal architecture, providing configurations ranging from 2PE ($32.6$ TOPS) to 8PE ($131.0$ TOPS). We design two computation units, Conv PE and DWC PE, to support different computational patterns. Each computation unit's data flow efficiently utilizes the data reuse opportunities to mitigate bandwidth bottlenecks. Additionally, we extend the functionality of each PE to utilize AIEs for non-convolutional operations, reducing resource overhead. Experiments on over 50 models show that compared to previous designs, our design provides $8.6\times$ the TOPS/W of traditional FPGA-based DPU designs, while reducing DSP usage by $95.8\%$, LUT usage by $44.7\%$, and latency to $68.5\%$ under single-batch conditions. For end-to-end inference, our design improving throughput by up to $2.2\times$ for depth-wise convolution models and up to $1.3\times$ for standard models. 

---
# KoGEC : Korean Grammatical Error Correction with Pre-trained Translation Models 

**Authors**: Taeeun Kim, Semin Jeong, Youngsook Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.11432)  

**Abstract**: This research introduces KoGEC, a Korean Grammatical Error Correction system using pre\--trained translation models. We fine-tuned NLLB (No Language Left Behind) models for Korean GEC, comparing their performance against large language models like GPT-4 and HCX-3. The study used two social media conversation datasets for training and testing. The NLLB models were fine-tuned using special language tokens to distinguish between original and corrected Korean sentences. Evaluation was done using BLEU scores and an "LLM as judge" method to classify error types. Results showed that the fine-tuned NLLB (KoGEC) models outperformed GPT-4o and HCX-3 in Korean GEC tasks. KoGEC demonstrated a more balanced error correction profile across various error types, whereas the larger LLMs tended to focus less on punctuation errors. We also developed a Chrome extension to make the KoGEC system accessible to users. Finally, we explored token vocabulary expansion to further improve the model but found it to decrease model performance. This research contributes to the field of NLP by providing an efficient, specialized Korean GEC system and a new evaluation method. It also highlights the potential of compact, task-specific models to compete with larger, general-purpose language models in specialized NLP tasks. 

---
# Agent-RLVR: Training Software Engineering Agents via Guidance and Environment Rewards 

**Authors**: Jeff Da, Clinton Wang, Xiang Deng, Yuntao Ma, Nikhil Barhate, Sean Hendryx  

**Link**: [PDF](https://arxiv.org/pdf/2506.11425)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) has been widely adopted as the de facto method for enhancing the reasoning capabilities of large language models and has demonstrated notable success in verifiable domains like math and competitive programming tasks. However, the efficacy of RLVR diminishes significantly when applied to agentic environments. These settings, characterized by multi-step, complex problem solving, lead to high failure rates even for frontier LLMs, as the reward landscape is too sparse for effective model training via conventional RLVR. In this work, we introduce Agent-RLVR, a framework that makes RLVR effective in challenging agentic settings, with an initial focus on software engineering tasks. Inspired by human pedagogy, Agent-RLVR introduces agent guidance, a mechanism that actively steers the agent towards successful trajectories by leveraging diverse informational cues. These cues, ranging from high-level strategic plans to dynamic feedback on the agent's errors and environmental interactions, emulate a teacher's guidance, enabling the agent to navigate difficult solution spaces and promotes active self-improvement via additional environment exploration. In the Agent-RLVR training loop, agents first attempt to solve tasks to produce initial trajectories, which are then validated by unit tests and supplemented with agent guidance. Agents then reattempt with guidance, and the agent policy is updated with RLVR based on the rewards of these guided trajectories. Agent-RLVR elevates the pass@1 performance of Qwen-2.5-72B-Instruct from 9.4% to 22.4% on SWE-Bench Verified. We find that our guidance-augmented RLVR data is additionally useful for test-time reward model training, shown by further boosting pass@1 to 27.8%. Agent-RLVR lays the groundwork for training agents with RLVR in complex, real-world environments where conventional RL methods struggle. 

---
# Deep Learning Model Acceleration and Optimization Strategies for Real-Time Recommendation Systems 

**Authors**: Junli Shao, Jing Dong, Dingzhou Wang, Kowei Shih, Dannier Li, Chengrui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.11421)  

**Abstract**: With the rapid growth of Internet services, recommendation systems play a central role in delivering personalized content. Faced with massive user requests and complex model architectures, the key challenge for real-time recommendation systems is how to reduce inference latency and increase system throughput without sacrificing recommendation quality. This paper addresses the high computational cost and resource bottlenecks of deep learning models in real-time settings by proposing a combined set of modeling- and system-level acceleration and optimization strategies. At the model level, we dramatically reduce parameter counts and compute requirements through lightweight network design, structured pruning, and weight quantization. At the system level, we integrate multiple heterogeneous compute platforms and high-performance inference libraries, and we design elastic inference scheduling and load-balancing mechanisms based on real-time load characteristics. Experiments show that, while maintaining the original recommendation accuracy, our methods cut latency to less than 30% of the baseline and more than double system throughput, offering a practical solution for deploying large-scale online recommendation services. 

---
# Stop learning it all to mitigate visual hallucination, Focus on the hallucination target 

**Authors**: Dokyoon Yoon, Youngsook Song, Woomyong Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.11417)  

**Abstract**: Multimodal Large Language Models (MLLMs) frequently suffer from hallucination issues, generating information about objects that are not present in input images during vision-language tasks. These hallucinations particularly undermine model reliability in practical applications requiring accurate object identification. To address this challenge, we propose \mymethod,\ a preference learning approach that mitigates hallucinations by focusing on targeted areas where they occur. To implement this, we build a dataset containing hallucinated responses, correct responses, and target information (i.e., objects present in the images and the corresponding chunk positions in responses affected by hallucinations). By applying a preference learning method restricted to these specific targets, the model can filter out irrelevant signals and focus on correcting hallucinations. This allows the model to produce more factual responses by concentrating solely on relevant information. Experimental results demonstrate that \mymethod\ effectively reduces hallucinations across multiple vision hallucination tasks, improving the reliability and performance of MLLMs without diminishing overall performance. 

---
# The Strategic Imperative for Healthcare Organizations to Build Proprietary Foundation Models 

**Authors**: Naresh Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2506.11412)  

**Abstract**: This paper presents a comprehensive analysis of the strategic imperative for healthcare organizations to develop proprietary foundation models rather than relying exclusively on commercial alternatives. We examine four fundamental considerations driving this imperative: the domain-specific requirements of healthcare data representation, critical data sovereignty and governance considerations unique to healthcare, strategic competitive advantages afforded by proprietary AI infrastructure, and the transformative potential of healthcare-specific foundation models for patient care and organizational operations. Through analysis of empirical evidence, economic frameworks, and organizational case studies, we demonstrate that proprietary multimodal foundation models enable healthcare organizations to achieve superior clinical performance, maintain robust data governance, create sustainable competitive advantages, and accelerate innovation pathways. While acknowledging implementation challenges, we present evidence showing organizations with proprietary AI capabilities demonstrate measurably improved outcomes, faster innovation cycles, and stronger strategic positioning in the evolving healthcare ecosystem. This analysis provides healthcare leaders with a comprehensive framework for evaluating build-versus-buy decisions regarding foundation model implementation, positioning proprietary foundation model development as a cornerstone capability for forward-thinking healthcare organizations. 

---
# A correlation-permutation approach for speech-music encoders model merging 

**Authors**: Fabian Ritter-Gutierrez, Yi-Cheng Lin, Jeremy H.M Wong, Hung-yi Lee, Eng Siong Chng, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11403)  

**Abstract**: Creating a unified speech and music model requires expensive pre-training. Model merging can instead create an unified audio model with minimal computational expense. However, direct merging is challenging when the models are not aligned in the weight space. Motivated by Git Re-Basin, we introduce a correlation-permutation approach that aligns a music encoder's internal layers with a speech encoder. We extend previous work to the case of merging transformer layers. The method computes a permutation matrix that maximizes the model's features-wise cross-correlations layer by layer, enabling effective fusion of these otherwise disjoint models. The merged model retains speech capabilities through this method while significantly enhancing music performance, achieving an improvement of 14.83 points in average score compared to linear interpolation model merging. This work allows the creation of unified audio models from independently trained encoders. 

---
# LoRA Users Beware: A Few Spurious Tokens Can Manipulate Your Finetuned Model 

**Authors**: Pradyut Sekhsaria, Marcel Mateos Salles, Hai Huang, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2506.11402)  

**Abstract**: Parameter Efficient FineTuning (PEFT), such as Low-Rank Adaptation (LoRA), aligns pre-trained Large Language Models (LLMs) to particular downstream tasks in a resource-efficient manner. Because efficiency has been the main metric of progress, very little attention has been put in understanding possible catastrophic failures. We uncover one such failure: PEFT encourages a model to search for shortcut solutions to solve its fine-tuning tasks. When very small amount of tokens, e.g., one token per prompt, are correlated with downstream task classes, PEFT makes any pretrained model rely predominantly on that token for decision making. While such spurious tokens may emerge accidentally from incorrect data cleaning, it also opens opportunities for malevolent parties to control a model's behavior from Seamless Spurious Token Injection (SSTI). In SSTI, a small amount of tokens correlated with downstream classes are injected by the dataset creators. At test time, the finetuned LLM's behavior can be controlled solely by injecting those few tokens. We apply SSTI across models from three families (Snowflake Arctic, Apple OpenELM, and Meta LLaMA-3) and four diverse datasets (IMDB, Financial Classification, CommonSense QA, and Bias in Bios). Our findings reveal three astonishing behaviors. First, as few as a single token of SSTI is sufficient to steer a model's decision making. Second, for light SSTI, the reliance on spurious tokens is proportional to the LoRA rank. Lastly, with aggressive SSTI, larger LoRA rank values become preferable to small rank values as it makes the model attend to non-spurious tokens, hence improving robustness. 

---
# Dynamic Double Space Tower 

**Authors**: Weikai Sun, Shijie Song, Han Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11394)  

**Abstract**: The Visual Question Answering (VQA) task requires the simultaneous understanding of image content and question semantics. However, existing methods often have difficulty handling complex reasoning scenarios due to insufficient cross-modal interaction and capturing the entity spatial relationships in the image.\cite{huang2023adaptive}\cite{liu2021comparing}\cite{guibas2021adaptive}\cite{zhang2022vsa}We studied a brand-new approach to replace the attention mechanism in order to enhance the reasoning ability of the model and its understanding of spatial this http URL, we propose a dynamic bidirectional spatial tower, which is divided into four layers to observe the image according to the principle of human gestalt vision. This naturally provides a powerful structural prior for the spatial organization between entities, enabling the model to no longer blindly search for relationships between pixels but make judgments based on more meaningful perceptual units. Change from "seeing images" to "perceiving and organizing image content".A large number of experiments have shown that our module can be used in any other multimodal model and achieve advanced results, demonstrating its potential in spatial relationship this http URL, the multimodal visual question-answering model July trained by our method has achieved state-of-the-art results with only 3B parameters, especially on the question-answering dataset of spatial relations. 

---
# A Variational Approach for Mitigating Entity Bias in Relation Extraction 

**Authors**: Samuel Mensah, Elena Kochkina, Jabez Magomere, Joy Prakash Sain, Simerjot Kaur, Charese Smiley  

**Link**: [PDF](https://arxiv.org/pdf/2506.11381)  

**Abstract**: Mitigating entity bias is a critical challenge in Relation Extraction (RE), where models often rely excessively on entities, resulting in poor generalization. This paper presents a novel approach to address this issue by adapting a Variational Information Bottleneck (VIB) framework. Our method compresses entity-specific information while preserving task-relevant features. It achieves state-of-the-art performance on relation extraction datasets across general, financial, and biomedical domains, in both indomain (original test sets) and out-of-domain (modified test sets with type-constrained entity replacements) settings. Our approach offers a robust, interpretable, and theoretically grounded methodology. 

---
# Enhance Multimodal Consistency and Coherence for Text-Image Plan Generation 

**Authors**: Xiaoxin Lu, Ranran Haoran Zhang, Yusen Zhang, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11380)  

**Abstract**: People get informed of a daily task plan through diverse media involving both texts and images. However, most prior research only focuses on LLM's capability of textual plan generation. The potential of large-scale models in providing text-image plans remains understudied. Generating high-quality text-image plans faces two main challenges: ensuring consistent alignment between two modalities and keeping coherence among visual steps. To address these challenges, we propose a novel framework that generates and refines text-image plans step-by-step. At each iteration, our framework (1) drafts the next textual step based on the prediction history; (2) edits the last visual step to obtain the next one; (3) extracts PDDL-like visual information; and (4) refines the draft with the extracted visual information. The textual and visual step produced in stage (4) and (2) will then serve as inputs for the next iteration. Our approach offers a plug-and-play improvement to various backbone models, such as Mistral-7B, Gemini-1.5, and GPT-4o. To evaluate the effectiveness of our approach, we collect a new benchmark consisting of 1,100 tasks and their text-image pair solutions covering 11 daily topics. We also design and validate a new set of metrics to evaluate the multimodal consistency and coherence in text-image plans. Extensive experiment results show the effectiveness of our approach on a range of backbone models against competitive baselines. Our code and data are available at this https URL. 

---
# Don't Pay Attention 

**Authors**: Mohammad Hammoud, Devang Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2506.11305)  

**Abstract**: The Transformer has become the de facto standard for large language models and a wide range of downstream tasks across various domains. Despite its numerous advantages like inherent training parallelism, the Transformer still faces key challenges due to its inability to effectively process sequences beyond a fixed context window and the quadratic complexity of its attention mechanism. These challenges have renewed interest in RNN-like architectures, which offer linear scaling with sequence length and improved handling of long-range dependencies, albeit with limited parallelism due to their inherently recurrent nature. In this paper, we propose Avey, a new neural foundational architecture that breaks away from both attention and recurrence. Avey comprises a ranker and an autoregressive neural processor, which collaboratively identify and contextualize only the most relevant tokens for any given token, regardless of their positions in the sequence. Specifically, Avey decouples sequence length from context width, thus enabling effective processing of arbitrarily long sequences. Experimental results show that Avey compares favorably to the Transformer across a variety of standard short-range NLP benchmarks, while notably excelling at capturing long-range dependencies. 

---
# TARDIS STRIDE: A Spatio-Temporal Road Image Dataset for Exploration and Autonomy 

**Authors**: Héctor Carrión, Yutong Bai, Víctor A. Hernández Castro, Kishan Panaganti, Ayush Zenith, Matthew Trang, Tony Zhang, Pietro Perona, Jitendra Malik  

**Link**: [PDF](https://arxiv.org/pdf/2506.11302)  

**Abstract**: World models aim to simulate environments and enable effective agent behavior. However, modeling real-world environments presents unique challenges as they dynamically change across both space and, crucially, time. To capture these composed dynamics, we introduce a Spatio-Temporal Road Image Dataset for Exploration (STRIDE) permuting 360-degree panoramic imagery into rich interconnected observation, state and action nodes. Leveraging this structure, we can simultaneously model the relationship between egocentric views, positional coordinates, and movement commands across both space and time. We benchmark this dataset via TARDIS, a transformer-based generative world model that integrates spatial and temporal dynamics through a unified autoregressive framework trained on STRIDE. We demonstrate robust performance across a range of agentic tasks such as controllable photorealistic image synthesis, instruction following, autonomous self-control, and state-of-the-art georeferencing. These results suggest a promising direction towards sophisticated generalist agents--capable of understanding and manipulating the spatial and temporal aspects of their material environments--with enhanced embodied reasoning capabilities. Training code, datasets, and model checkpoints are made available at this https URL. 

---
# Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning 

**Authors**: Yang Zhang, Amr Mohamed, Hadi Abdine, Guokan Shang, Michalis Vazirgiannis  

**Link**: [PDF](https://arxiv.org/pdf/2506.11300)  

**Abstract**: Curriculum learning has shown promise in improving training efficiency and generalization in various machine learning domains, yet its potential in pretraining language models remains underexplored, prompting our work as the first systematic investigation in this area. We experimented with different settings, including vanilla curriculum learning, pacing-based sampling, and interleaved curricula-guided by six difficulty metrics spanning linguistic and information-theoretic perspectives. We train models under these settings and evaluate their performance on eight diverse benchmarks. Our experiments reveal that curriculum learning consistently improves convergence in early and mid-training phases, and can yield lasting gains when used as a warmup strategy with up to $3.5\%$ improvement. Notably, we identify compression ratio, lexical diversity, and readability as effective difficulty signals across settings. Our findings highlight the importance of data ordering in large-scale pretraining and provide actionable insights for scalable, data-efficient model development under realistic training scenarios. 

---
# A Tale of Two Systems: Characterizing Architectural Complexity on Machine Learning-Enabled Systems 

**Authors**: Renato Cordeiro Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2506.11295)  

**Abstract**: How can the complexity of ML-enabled systems be managed effectively? The goal of this research is to investigate how complexity affects ML-Enabled Systems (MLES). To address this question, this research aims to introduce a metrics-based architectural model to characterize the complexity of MLES. The goal is to support architectural decisions, providing a guideline for the inception and growth of these systems. This paper brings, side-by-side, the architecture representation of two systems that can be used as case studies for creating the metrics-based architectural model: the SPIRA and the Ocean Guard MLES. 

---
# Invocable APIs derived from NL2SQL datasets for LLM Tool-Calling Evaluation 

**Authors**: Benjamin Elder, Anupama Murthi, Jungkoo Kang, Ankita Rajaram Naik, Kiran Kate, Kinjal Basu, Danish Contractor  

**Link**: [PDF](https://arxiv.org/pdf/2506.11266)  

**Abstract**: Large language models (LLMs) are routinely deployed as agentic systems, with access to tools that interact with live environments to accomplish tasks. In enterprise deployments these systems need to interact with API collections that can be extremely large and complex, often backed by databases. In order to create datasets with such characteristics, we explore how existing NL2SQL (Natural Language to SQL query) datasets can be used to automatically create NL2API datasets. Specifically, this work describes a novel data generation pipeline that exploits the syntax of SQL queries to construct a functionally equivalent sequence of API calls. We apply this pipeline to one of the largest NL2SQL datasets, BIRD-SQL to create a collection of over 2500 APIs that can be served as invocable tools or REST-endpoints. We pair natural language queries from BIRD-SQL to ground-truth API sequences based on this API pool. We use this collection to study the performance of 10 public LLMs and find that all models struggle to determine the right set of tools (consisting of tasks of intent detection, sequencing with nested function calls, and slot-filling). We find that models have extremely low task completion rates (7-47 percent - depending on the dataset) which marginally improves to 50 percent when models are employed as ReACT agents that interact with the live API environment. The best task completion rates are far below what may be required for effective general-use tool-calling agents, suggesting substantial scope for improvement in current state-of-the-art tool-calling LLMs. We also conduct detailed ablation studies, such as assessing the impact of the number of tools available as well as the impact of tool and slot-name obfuscation. We compare the performance of models on the original SQL generation tasks and find that current models are sometimes able to exploit SQL better than APIs. 

---
# Gondola: Grounded Vision Language Planning for Generalizable Robotic Manipulation 

**Authors**: Shizhe Chen, Ricardo Garcia, Paul Pacaud, Cordelia Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2506.11261)  

**Abstract**: Robotic manipulation faces a significant challenge in generalizing across unseen objects, environments and tasks specified by diverse language instructions. To improve generalization capabilities, recent research has incorporated large language models (LLMs) for planning and action execution. While promising, these methods often fall short in generating grounded plans in visual environments. Although efforts have been made to perform visual instructional tuning on LLMs for robotic manipulation, existing methods are typically constrained by single-view image input and struggle with precise object grounding. In this work, we introduce Gondola, a novel grounded vision-language planning model based on LLMs for generalizable robotic manipulation. Gondola takes multi-view images and history plans to produce the next action plan with interleaved texts and segmentation masks of target objects and locations. To support the training of Gondola, we construct three types of datasets using the RLBench simulator, namely robot grounded planning, multi-view referring expression and pseudo long-horizon task datasets. Gondola outperforms the state-of-the-art LLM-based method across all four generalization levels of the GemBench dataset, including novel placements, rigid objects, articulated objects and long-horizon tasks. 

---
# Measuring multi-calibration 

**Authors**: Ido Guy, Daniel Haimovich, Fridolin Linder, Nastaran Okati, Lorenzo Perini, Niek Tax, Mark Tygert  

**Link**: [PDF](https://arxiv.org/pdf/2506.11251)  

**Abstract**: A suitable scalar metric can help measure multi-calibration, defined as follows. When the expected values of observed responses are equal to corresponding predicted probabilities, the probabilistic predictions are known as "perfectly calibrated." When the predicted probabilities are perfectly calibrated simultaneously across several subpopulations, the probabilistic predictions are known as "perfectly multi-calibrated." In practice, predicted probabilities are seldom perfectly multi-calibrated, so a statistic measuring the distance from perfect multi-calibration is informative. A recently proposed metric for calibration, based on the classical Kuiper statistic, is a natural basis for a new metric of multi-calibration and avoids well-known problems of metrics based on binning or kernel density estimation. The newly proposed metric weights the contributions of different subpopulations in proportion to their signal-to-noise ratios; data analyses' ablations demonstrate that the metric becomes noisy when omitting the signal-to-noise ratios from the metric. Numerical examples on benchmark data sets illustrate the new metric. 

---
# Can Time-Series Foundation Models Perform Building Energy Management Tasks? 

**Authors**: Ozan Baris Mulayim, Pengrui Quan, Liying Han, Xiaomin Ouyang, Dezhi Hong, Mario Bergés, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2506.11250)  

**Abstract**: Building energy management (BEM) tasks require processing and learning from a variety of time-series data. Existing solutions rely on bespoke task- and data-specific models to perform these tasks, limiting their broader applicability. Inspired by the transformative success of Large Language Models (LLMs), Time-Series Foundation Models (TSFMs), trained on diverse datasets, have the potential to change this. Were TSFMs to achieve a level of generalizability across tasks and contexts akin to LLMs, they could fundamentally address the scalability challenges pervasive in BEM. To understand where they stand today, we evaluate TSFMs across four dimensions: (1) generalizability in zero-shot univariate forecasting, (2) forecasting with covariates for thermal behavior modeling, (3) zero-shot representation learning for classification tasks, and (4) robustness to performance metrics and varying operational conditions. Our results reveal that TSFMs exhibit \emph{limited} generalizability, performing only marginally better than statistical models on unseen datasets and modalities for univariate forecasting. Similarly, inclusion of covariates in TSFMs does not yield performance improvements, and their performance remains inferior to conventional models that utilize covariates. While TSFMs generate effective zero-shot representations for downstream classification tasks, they may remain inferior to statistical models in forecasting when statistical models perform test-time fitting. Moreover, TSFMs forecasting performance is sensitive to evaluation metrics, and they struggle in more complex building environments compared to statistical models. These findings underscore the need for targeted advancements in TSFM design, particularly their handling of covariates and incorporating context and temporal dynamics into prediction mechanisms, to develop more adaptable and scalable solutions for BEM. 

---
# No Universal Prompt: Unifying Reasoning through Adaptive Prompting for Temporal Table Reasoning 

**Authors**: Kushagra Dixit, Abhishek Rajgaria, Harshavardhan Kalalbandi, Dan Roth, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.11246)  

**Abstract**: Temporal Table Reasoning is a critical challenge for Large Language Models (LLMs), requiring effective prompting techniques to extract relevant insights. Despite existence of multiple prompting methods, their impact on table reasoning remains largely unexplored. Furthermore, the performance of these models varies drastically across different table and context structures, making it difficult to determine an optimal approach. This work investigates multiple prompting technique across diverse table types to determine optimal approaches for different scenarios. We find that performance varies based on entity type, table structure, requirement of additional context and question complexity, with NO single method consistently outperforming others. To mitigate these challenges, we introduce SEAR, an adaptive prompting framework inspired by human reasoning that dynamically adjusts based on context characteristics and integrates a structured reasoning. Our results demonstrate that SEAR achieves superior performance across all table types compared to other baseline prompting techniques. Additionally, we explore the impact of table structure refactoring, finding that a unified representation enhances model's reasoning. 

---
# RETUYT-INCO at BEA 2025 Shared Task: How Far Can Lightweight Models Go in AI-powered Tutor Evaluation? 

**Authors**: Santiago Góngora, Ignacio Sastre, Santiago Robaina, Ignacio Remersaro, Luis Chiruzzo, Aiala Rosá  

**Link**: [PDF](https://arxiv.org/pdf/2506.11243)  

**Abstract**: In this paper, we present the RETUYT-INCO participation at the BEA 2025 shared task. Our participation was characterized by the decision of using relatively small models, with fewer than 1B parameters. This self-imposed restriction tries to represent the conditions in which many research labs or institutions are in the Global South, where computational power is not easily accessible due to its prohibitive cost. Even under this restrictive self-imposed setting, our models managed to stay competitive with the rest of teams that participated in the shared task. According to the $exact\ F_1$ scores published by the organizers, the performance gaps between our models and the winners were as follows: $6.46$ in Track 1; $10.24$ in Track 2; $7.85$ in Track 3; $9.56$ in Track 4; and $13.13$ in Track 5. Considering that the minimum difference with a winner team is $6.46$ points -- and the maximum difference is $13.13$ -- according to the $exact\ F_1$ score, we find that models with a size smaller than 1B parameters are competitive for these tasks, all of which can be run on computers with a low-budget GPU or even without a GPU. 

---
# A Causal Lens for Learning Long-term Fair Policies 

**Authors**: Jacob Lear, Lu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11242)  

**Abstract**: Fairness-aware learning studies the development of algorithms that avoid discriminatory decision outcomes despite biased training data. While most studies have concentrated on immediate bias in static contexts, this paper highlights the importance of investigating long-term fairness in dynamic decision-making systems while simultaneously considering instantaneous fairness requirements. In the context of reinforcement learning, we propose a general framework where long-term fairness is measured by the difference in the average expected qualification gain that individuals from different groups could this http URL, through a causal lens, we decompose this metric into three components that represent the direct impact, the delayed impact, as well as the spurious effect the policy has on the qualification gain. We analyze the intrinsic connection between these components and an emerging fairness notion called benefit fairness that aims to control the equity of outcomes in decision-making. Finally, we develop a simple yet effective approach for balancing various fairness notions. 

---
# uPVC-Net: A Universal Premature Ventricular Contraction Detection Deep Learning Algorithm 

**Authors**: Hagai Hamami, Yosef Solewicz, Daniel Zur, Yonatan Kleerekoper, Joachim A. Behar  

**Link**: [PDF](https://arxiv.org/pdf/2506.11238)  

**Abstract**: Introduction: Premature Ventricular Contractions (PVCs) are common cardiac arrhythmias originating from the ventricles. Accurate detection remains challenging due to variability in electrocardiogram (ECG) waveforms caused by differences in lead placement, recording conditions, and population demographics. Methods: We developed uPVC-Net, a universal deep learning model to detect PVCs from any single-lead ECG recordings. The model is developed on four independent ECG datasets comprising a total of 8.3 million beats collected from Holter monitors and a modern wearable ECG patch. uPVC-Net employs a custom architecture and a multi-source, multi-lead training strategy. For each experiment, one dataset is held out to evaluate out-of-distribution (OOD) generalization. Results: uPVC-Net achieved an AUC between 97.8% and 99.1% on the held-out datasets. Notably, performance on wearable single-lead ECG data reached an AUC of 99.1%. Conclusion: uPVC-Net exhibits strong generalization across diverse lead configurations and populations, highlighting its potential for robust, real-world clinical deployment. 

---
# Complexity of normalized stochastic first-order methods with momentum under heavy-tailed noise 

**Authors**: Chuan He, Zhaosong Lu, Defeng Sun, Zhanwang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11214)  

**Abstract**: In this paper, we propose practical normalized stochastic first-order methods with Polyak momentum, multi-extrapolated momentum, and recursive momentum for solving unconstrained optimization problems. These methods employ dynamically updated algorithmic parameters and do not require explicit knowledge of problem-dependent quantities such as the Lipschitz constant or noise bound. We establish first-order oracle complexity results for finding approximate stochastic stationary points under heavy-tailed noise and weakly average smoothness conditions -- both of which are weaker than the commonly used bounded variance and mean-squared smoothness assumptions. Our complexity bounds either improve upon or match the best-known results in the literature. Numerical experiments are presented to demonstrate the practical effectiveness of the proposed methods. 

---
# Multimodal Modeling of CRISPR-Cas12 Activity Using Foundation Models and Chromatin Accessibility Data 

**Authors**: Azim Dehghani Amirabad, Yanfei Zhang, Artem Moskalev, Sowmya Rajesh, Tommaso Mansi, Shuwei Li, Mangal Prakash, Rui Liao  

**Link**: [PDF](https://arxiv.org/pdf/2506.11182)  

**Abstract**: Predicting guide RNA (gRNA) activity is critical for effective CRISPR-Cas12 genome editing but remains challenging due to limited data, variation across protospacer adjacent motifs (PAMs-short sequence requirements for Cas binding), and reliance on large-scale training. We investigate whether pre-trained biological foundation model originally trained on transcriptomic data can improve gRNA activity estimation even without domain-specific pre-training. Using embeddings from existing RNA foundation model as input to lightweight regressor, we show substantial gains over traditional baselines. We also integrate chromatin accessibility data to capture regulatory context, improving performance further. Our results highlight the effectiveness of pre-trained foundation models and chromatin accessibility data for gRNA activity prediction. 

---
# Beyond Formal Semantics for Capabilities and Skills: Model Context Protocol in Manufacturing 

**Authors**: Luis Miguel Vieira da Silva, Aljosha Köcher, Felix Gehlhoff  

**Link**: [PDF](https://arxiv.org/pdf/2506.11180)  

**Abstract**: Explicit modeling of capabilities and skills -- whether based on ontologies, Asset Administration Shells, or other technologies -- requires considerable manual effort and often results in representations that are not easily accessible to Large Language Models (LLMs). In this work-in-progress paper, we present an alternative approach based on the recently introduced Model Context Protocol (MCP). MCP allows systems to expose functionality through a standardized interface that is directly consumable by LLM-based agents. We conduct a prototypical evaluation on a laboratory-scale manufacturing system, where resource functions are made available via MCP. A general-purpose LLM is then tasked with planning and executing a multi-step process, including constraint handling and the invocation of resource functions via MCP. The results indicate that such an approach can enable flexible industrial automation without relying on explicit semantic models. This work lays the basis for further exploration of external tool integration in LLM-driven production systems. 

---
# Brain2Vec: A Deep Learning Framework for EEG-Based Stress Detection Using CNN-LSTM-Attention 

**Authors**: Md Mynoddin, Troyee Dev, Rishita Chakma  

**Link**: [PDF](https://arxiv.org/pdf/2506.11179)  

**Abstract**: Mental stress has become a pervasive factor affecting cognitive health and overall well-being, necessitating the development of robust, non-invasive diagnostic tools. Electroencephalogram (EEG) signals provide a direct window into neural activity, yet their non-stationary and high-dimensional nature poses significant modeling challenges. Here we introduce Brain2Vec, a new deep learning tool that classifies stress states from raw EEG recordings using a hybrid architecture of convolutional, recurrent, and attention mechanisms. The model begins with a series of convolutional layers to capture localized spatial dependencies, followed by an LSTM layer to model sequential temporal patterns, and concludes with an attention mechanism to emphasize informative temporal regions. We evaluate Brain2Vec on the DEAP dataset, applying bandpass filtering, z-score normalization, and epoch segmentation as part of a comprehensive preprocessing pipeline. Compared to traditional CNN-LSTM baselines, our proposed model achieves an AUC score of 0.68 and a validation accuracy of 81.25%. These findings demonstrate Brain2Vec's potential for integration into wearable stress monitoring platforms and personalized healthcare systems. 

---
# Collapsing Sequence-Level Data-Policy Coverage via Poisoning Attack in Offline Reinforcement Learning 

**Authors**: Xue Zhou, Dapeng Man, Chen Xu, Fanyi Zeng, Tao Liu, Huan Wang, Shucheng He, Chaoyang Gao, Wu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11172)  

**Abstract**: Offline reinforcement learning (RL) heavily relies on the coverage of pre-collected data over the target policy's distribution. Existing studies aim to improve data-policy coverage to mitigate distributional shifts, but overlook security risks from insufficient coverage, and the single-step analysis is not consistent with the multi-step decision-making nature of offline RL. To address this, we introduce the sequence-level concentrability coefficient to quantify coverage, and reveal its exponential amplification on the upper bound of estimation errors through theoretical analysis. Building on this, we propose the Collapsing Sequence-Level Data-Policy Coverage (CSDPC) poisoning attack. Considering the continuous nature of offline RL data, we convert state-action pairs into decision units, and extract representative decision patterns that capture multi-step behavior. We identify rare patterns likely to cause insufficient coverage, and poison them to reduce coverage and exacerbate distributional shifts. Experiments show that poisoning just 1% of the dataset can degrade agent performance by 90%. This finding provides new perspectives for analyzing and safeguarding the security of offline RL. 

---
# PromptTSS: A Prompting-Based Approach for Interactive Multi-Granularity Time Series Segmentation 

**Authors**: Ching Chang, Ming-Chih Lo, Wen-Chih Peng, Tien-Fu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11170)  

**Abstract**: Multivariate time series data, collected across various fields such as manufacturing and wearable technology, exhibit states at multiple levels of granularity, from coarse-grained system behaviors to fine-grained, detailed events. Effectively segmenting and integrating states across these different granularities is crucial for tasks like predictive maintenance and performance optimization. However, existing time series segmentation methods face two key challenges: (1) the inability to handle multiple levels of granularity within a unified model, and (2) limited adaptability to new, evolving patterns in dynamic environments. To address these challenges, we propose PromptTSS, a novel framework for time series segmentation with multi-granularity states. PromptTSS uses a unified model with a prompting mechanism that leverages label and boundary information to guide segmentation, capturing both coarse- and fine-grained patterns while adapting dynamically to unseen patterns. Experiments show PromptTSS improves accuracy by 24.49% in multi-granularity segmentation, 17.88% in single-granularity segmentation, and up to 599.24% in transfer learning, demonstrating its adaptability to hierarchical states and evolving time series dynamics. 

---
# Test-Time-Scaling for Zero-Shot Diagnosis with Visual-Language Reasoning 

**Authors**: Ji Young Byun, Young-Jin Park, Navid Azizan, Rama Chellappa  

**Link**: [PDF](https://arxiv.org/pdf/2506.11166)  

**Abstract**: As a cornerstone of patient care, clinical decision-making significantly influences patient outcomes and can be enhanced by large language models (LLMs). Although LLMs have demonstrated remarkable performance, their application to visual question answering in medical imaging, particularly for reasoning-based diagnosis, remains largely unexplored. Furthermore, supervised fine-tuning for reasoning tasks is largely impractical due to limited data availability and high annotation costs. In this work, we introduce a zero-shot framework for reliable medical image diagnosis that enhances the reasoning capabilities of LLMs in clinical settings through test-time scaling. Given a medical image and a textual prompt, a vision-language model processes a medical image along with a corresponding textual prompt to generate multiple descriptions or interpretations of visual features. These interpretations are then fed to an LLM, where a test-time scaling strategy consolidates multiple candidate outputs into a reliable final diagnosis. We evaluate our approach across various medical imaging modalities -- including radiology, ophthalmology, and histopathology -- and demonstrate that the proposed test-time scaling strategy enhances diagnostic accuracy for both our and baseline methods. Additionally, we provide an empirical analysis showing that the proposed approach, which allows unbiased prompting in the first stage, improves the reliability of LLM-generated diagnoses and enhances classification accuracy. 

---
# Evaluating BiLSTM and CNN+GRU Approaches for Human Activity Recognition Using WiFi CSI Data 

**Authors**: Almustapha A. Wakili, Babajide J. Asaju, Woosub Jung  

**Link**: [PDF](https://arxiv.org/pdf/2506.11165)  

**Abstract**: This paper compares the performance of BiLSTM and CNN+GRU deep learning models for Human Activity Recognition (HAR) on two WiFi-based Channel State Information (CSI) datasets: UT-HAR and NTU-Fi HAR. The findings indicate that the CNN+GRU model has a higher accuracy on the UT-HAR dataset (95.20%) thanks to its ability to extract spatial features. In contrast, the BiLSTM model performs better on the high-resolution NTU-Fi HAR dataset (92.05%) by extracting long-term temporal dependencies more effectively. The findings strongly emphasize the critical role of dataset characteristics and preprocessing techniques in model performance improvement. We also show the real-world applicability of such models in applications like healthcare and intelligent home systems, highlighting their potential for unobtrusive activity recognition. 

---
# Synthetic Geology -- Structural Geology Meets Deep Learning 

**Authors**: Simon Ghyselincks, Valeriia Okhmak, Stefano Zampini, George Turkiyyah, David Keyes, Eldad Haber  

**Link**: [PDF](https://arxiv.org/pdf/2506.11164)  

**Abstract**: Visualizing the first few kilometers of the Earth's subsurface, a long-standing challenge gating a virtually inexhaustible list of important applications, is coming within reach through deep learning. Building on techniques of generative artificial intelligence applied to voxelated images, we demonstrate a method that extends surface geological data supplemented by boreholes to a three-dimensional subsurface region by training a neural network. The Earth's land area having been extensively mapped for geological features, the bottleneck of this or any related technique is the availability of data below the surface. We close this data gap in the development of subsurface deep learning by designing a synthetic data-generator process that mimics eons of geological activity such as sediment compaction, volcanic intrusion, and tectonic dynamics to produce a virtually limitless number of samples of the near lithosphere. A foundation model trained on such synthetic data is able to generate a 3D image of the subsurface from a previously unseen map of surface topography and geology, showing increasing fidelity with increasing access to borehole data, depicting such structures as layers, faults, folds, dikes, and sills. We illustrate the early promise of the combination of a synthetic lithospheric generator with a trained neural network model using generative flow matching. Ultimately, such models will be fine-tuned on data from applicable campaigns, such as mineral prospecting in a given region. Though useful in itself, a regionally fine-tuned models may be employed not as an end but as a means: as an AI-based regularizer in a more traditional inverse problem application, in which the objective function represents the mismatch of additional data with physical models with applications in resource exploration, hazard assessment, and geotechnical engineering. 

---
# Autonomous Computer Vision Development with Agentic AI 

**Authors**: Jin Kim, Muhammad Wahi-Anwa, Sangyun Park, Shawn Shin, John M. Hoffman, Matthew S. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2506.11140)  

**Abstract**: Agentic Artificial Intelligence (AI) systems leveraging Large Language Models (LLMs) exhibit significant potential for complex reasoning, planning, and tool utilization. We demonstrate that a specialized computer vision system can be built autonomously from a natural language prompt using Agentic AI methods. This involved extending SimpleMind (SM), an open-source Cognitive AI environment with configurable tools for medical image analysis, with an LLM-based agent, implemented using OpenManus, to automate the planning (tool configuration) for a particular computer vision task. We provide a proof-of-concept demonstration that an agentic system can interpret a computer vision task prompt, plan a corresponding SimpleMind workflow by decomposing the task and configuring appropriate tools. From the user input prompt, "provide sm (SimpleMind) config for lungs, heart, and ribs segmentation for cxr (chest x-ray)"), the agent LLM was able to generate the plan (tool configuration file in YAML format), and execute SM-Learn (training) and SM-Think (inference) scripts autonomously. The computer vision agent automatically configured, trained, and tested itself on 50 chest x-ray images, achieving mean dice scores of 0.96, 0.82, 0.83, for lungs, heart, and ribs, respectively. This work shows the potential for autonomous planning and tool configuration that has traditionally been performed by a data scientist in the development of computer vision applications. 

---
# Grids Often Outperform Implicit Neural Representations 

**Authors**: Namhoon Kim, Sara Fridovich-Keil  

**Link**: [PDF](https://arxiv.org/pdf/2506.11139)  

**Abstract**: Implicit Neural Representations (INRs) have recently shown impressive results, but their fundamental capacity, implicit biases, and scaling behavior remain poorly understood. We investigate the performance of diverse INRs across a suite of 2D and 3D real and synthetic signals with varying effective bandwidth, as well as both overfitting and generalization tasks including tomography, super-resolution, and denoising. By stratifying performance according to model size as well as signal type and bandwidth, our results shed light on how different INR and grid representations allocate their capacity. We find that, for most tasks and signals, a simple regularized grid with interpolation trains faster and to higher quality than any INR with the same number of parameters. We also find limited settings where INRs outperform grids -- namely fitting signals with underlying lower-dimensional structure such as shape contours -- to guide future use of INRs towards the most advantageous applications. Code and synthetic signals used in our analysis are available at this https URL. 

---
# Large Language Models and Emergence: A Complex Systems Perspective 

**Authors**: David C. Krakauer, John W. Krakauer, Melanie Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2506.11135)  

**Abstract**: Emergence is a concept in complexity science that describes how many-body systems manifest novel higher-level properties, properties that can be described by replacing high-dimensional mechanisms with lower-dimensional effective variables and theories. This is captured by the idea "more is different". Intelligence is a consummate emergent property manifesting increasingly efficient -- cheaper and faster -- uses of emergent capabilities to solve problems. This is captured by the idea "less is more". In this paper, we first examine claims that Large Language Models exhibit emergent capabilities, reviewing several approaches to quantifying emergence, and secondly ask whether LLMs possess emergent intelligence. 

---
# A Self-Refining Framework for Enhancing ASR Using TTS-Synthesized Data 

**Authors**: Cheng Kang Chou, Chan-Jan Hsu, Ho-Lam Chung, Liang-Hsuan Tseng, Hsi-Chun Cheng, Yu-Kuan Fu, Kuan Po Huang, Hung-Yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.11130)  

**Abstract**: We propose a self-refining framework that enhances ASR performance with only unlabeled datasets. The process starts with an existing ASR model generating pseudo-labels on unannotated speech, which are then used to train a high-fidelity text-to-speech (TTS) system. Then, synthesized speech text pairs are bootstrapped into the original ASR system, completing the closed-loop self-improvement cycle. We demonstrated the effectiveness of the framework on Taiwanese Mandarin speech. Leveraging 6,000 hours of unlabeled speech, a moderate amount of text data, and synthetic content from the AI models, we adapt Whisper-large-v2 into a specialized model, Twister. Twister reduces error rates by up to 20% on Mandarin and 50% on Mandarin-English code-switching benchmarks compared to Whisper. Results highlight the framework as a compelling alternative to pseudo-labeling self-distillation approaches and provides a practical pathway for improving ASR performance in low-resource or domain-specific settings. 

---
# Trustworthy AI for Medicine: Continuous Hallucination Detection and Elimination with CHECK 

**Authors**: Carlos Garcia-Fernandez, Luis Felipe, Monique Shotande, Muntasir Zitu, Aakash Tripathi, Ghulam Rasool, Issam El Naqa, Vivek Rudrapatna, Gilmer Valdes  

**Link**: [PDF](https://arxiv.org/pdf/2506.11129)  

**Abstract**: Large language models (LLMs) show promise in healthcare, but hallucinations remain a major barrier to clinical use. We present CHECK, a continuous-learning framework that integrates structured clinical databases with a classifier grounded in information theory to detect both factual and reasoning-based hallucinations. Evaluated on 1500 questions from 100 pivotal clinical trials, CHECK reduced LLama3.3-70B-Instruct hallucination rates from 31% to 0.3% - making an open source model state of the art. Its classifier generalized across medical benchmarks, achieving AUCs of 0.95-0.96, including on the MedQA (USMLE) benchmark and HealthBench realistic multi-turn medical questioning. By leveraging hallucination probabilities to guide GPT-4o's refinement and judiciously escalate compute, CHECK boosted its USMLE passing rate by 5 percentage points, achieving a state-of-the-art 92.1%. By suppressing hallucinations below accepted clinical error thresholds, CHECK offers a scalable foundation for safe LLM deployment in medicine and other high-stakes domains. 

---
# Stronger Language Models Produce More Human-Like Errors 

**Authors**: Andrew Keenan Richardson, Ryan Othniel Kearns, Sean Moss, Vincent Wang-Mascianica, Philipp Koralus  

**Link**: [PDF](https://arxiv.org/pdf/2506.11128)  

**Abstract**: Do language models converge toward human-like reasoning patterns as they improve? We provide surprising evidence that while overall reasoning capabilities increase with model sophistication, the nature of errors increasingly mirrors predictable human reasoning fallacies: a previously unobserved inverse scaling phenomenon. To investigate this question, we apply the Erotetic Theory of Reasoning (ETR), a formal cognitive framework with empirical support for predicting human reasoning outcomes. Using the open-source package PyETR, we generate logical reasoning problems where humans predictably err, evaluating responses from 38 language models across 383 reasoning tasks. Our analysis indicates that as models advance in general capability (as measured by Chatbot Arena scores), the proportion of their incorrect answers that align with ETR-predicted human fallacies tends to increase ($\rho = 0.360, p = 0.0265$). Notably, as we observe no correlation between model sophistication and logical correctness on these tasks, this shift in error patterns toward human-likeness occurs independently of error rate. These findings challenge the prevailing view that scaling language models naturally obtains normative rationality, suggesting instead a convergence toward human-like cognition inclusive of our characteristic biases and limitations, as we further confirm by demonstrating order-effects in language model reasoning. 

---
# GUIRoboTron-Speech: Towards Automated GUI Agents Based on Speech Instructions 

**Authors**: Wenkang Han, Zhixiong Zeng, Jing Huang, Shu Jiang, Liming Zheng, Longrong Yang, Haibo Qiu, Chang Yao, Jingyuan Chen, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.11127)  

**Abstract**: Autonomous agents for Graphical User Interfaces (GUIs) are revolutionizing human-computer interaction, yet their reliance on text-based instructions imposes limitations on accessibility and convenience, particularly in hands-free scenarios. To address this gap, we propose GUIRoboTron-Speech, the first end-to-end autonomous GUI agent that directly accepts speech instructions and on-device screenshots to predict actions. Confronted with the scarcity of speech-based GUI agent datasets, we initially generated high-quality speech instructions for training by leveraging a random timbre text-to-speech (TTS) model to convert existing text instructions. We then develop GUIRoboTron-Speech's capabilities through progressive grounding and planning training stages. A key contribution is a heuristic mixed-instruction training strategy designed to mitigate the modality imbalance inherent in pre-trained foundation models. Comprehensive experiments on several benchmark datasets validate the robust and superior performance of GUIRoboTron-Speech, demonstrating the significant potential and widespread applicability of speech as an effective instruction modality for driving GUI agents. Our code and datasets are available at this https URL. 

---
# ASRJam: Human-Friendly AI Speech Jamming to Prevent Automated Phone Scams 

**Authors**: Freddie Grabovski, Gilad Gressel, Yisroel Mirsky  

**Link**: [PDF](https://arxiv.org/pdf/2506.11125)  

**Abstract**: Large Language Models (LLMs), combined with Text-to-Speech (TTS) and Automatic Speech Recognition (ASR), are increasingly used to automate voice phishing (vishing) scams. These systems are scalable and convincing, posing a significant security threat. We identify the ASR transcription step as the most vulnerable link in the scam pipeline and introduce ASRJam, a proactive defence framework that injects adversarial perturbations into the victim's audio to disrupt the attacker's ASR. This breaks the scam's feedback loop without affecting human callers, who can still understand the conversation. While prior adversarial audio techniques are often unpleasant and impractical for real-time use, we also propose EchoGuard, a novel jammer that leverages natural distortions, such as reverberation and echo, that are disruptive to ASR but tolerable to humans. To evaluate EchoGuard's effectiveness and usability, we conducted a 39-person user study comparing it with three state-of-the-art attacks. Results show that EchoGuard achieved the highest overall utility, offering the best combination of ASR disruption and human listening experience. 

---
# SUTA-LM: Bridging Test-Time Adaptation and Language Model Rescoring for Robust ASR 

**Authors**: Wei-Ping Huang, Guan-Ting Lin, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.11121)  

**Abstract**: Despite progress in end-to-end ASR, real-world domain mismatches still cause performance drops, which Test-Time Adaptation (TTA) aims to mitigate by adjusting models during inference. Recent work explores combining TTA with external language models, using techniques like beam search rescoring or generative error correction. In this work, we identify a previously overlooked challenge: TTA can interfere with language model rescoring, revealing the nontrivial nature of effectively combining the two methods. Based on this insight, we propose SUTA-LM, a simple yet effective extension of SUTA, an entropy-minimization-based TTA approach, with language model rescoring. SUTA-LM first applies a controlled adaptation process guided by an auto-step selection mechanism leveraging both acoustic and linguistic information, followed by language model rescoring to refine the outputs. Experiments on 18 diverse ASR datasets show that SUTA-LM achieves robust results across a wide range of domains. 

---
# SDMPrune: Self-Distillation MLP Pruning for Efficient Large Language Models 

**Authors**: Hourun Zhu, Chengchao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11120)  

**Abstract**: In spite of strong performance achieved by LLMs, the costs of their deployment are unaffordable. For the compression of LLMs, gradient-based pruning methods present promising effectiveness. However, in these methods, the gradient computation with one-hot labels ignore the potential predictions on other words, thus missing key information for generative capability of the original model. To address this issue, we introduce a self-distillation loss during the pruning phase (rather than post-training) to fully exploit the predictions of the original model, thereby obtaining more accurate gradient information for pruning. Moreover, we find that, compared to attention modules, the predictions of LLM are less sensitive to multilayer perceptron (MLP) modules, which take up more than $5 \times$ parameters (LLaMA3.2-1.2B). To this end, we focus on the pruning of MLP modules, to significantly compress LLM without obvious performance degradation. Experimental results on extensive zero-shot benchmarks demonstrate that our method significantly outperforms existing pruning methods. Furthermore, our method achieves very competitive performance among 1B-scale open source LLMs. The source code and trained weights are available at this https URL. 

---
# ScIRGen: Synthesize Realistic and Large-Scale RAG Dataset for Scientific Research 

**Authors**: Junyong Lin, Lu Dai, Ruiqian Han, Yijie Sui, Ruilin Wang, Xingliang Sun, Qinglin Wu, Min Feng, Hao Liu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.11117)  

**Abstract**: Scientific researchers need intensive information about datasets to effectively evaluate and develop theories and methodologies. The information needs regarding datasets are implicitly embedded in particular research tasks, rather than explicitly expressed in search queries. However, existing scientific retrieval and question-answering (QA) datasets typically address straightforward questions, which do not align with the distribution of real-world research inquiries. To bridge this gap, we developed ScIRGen, a dataset generation framework for scientific QA \& retrieval that more accurately reflects the information needs of professional science researchers, and uses it to create a large-scale scientific retrieval-augmented generation (RAG) dataset with realistic queries, datasets and papers. Technically, we designed a dataset-oriented information extraction method that leverages academic papers to augment the dataset representation. We then proposed a question generation framework by employing cognitive taxonomy to ensure the quality of synthesized questions. We also design a method to automatically filter synthetic answers based on the perplexity shift of LLMs, which is highly aligned with human judgment of answers' validity. Collectively, these methodologies culminated in the creation of the 61k QA dataset, ScIRGen-Geo. We benchmarked representative methods on the ScIRGen-Geo dataset for their question-answering and retrieval capabilities, finding out that current methods still suffer from reasoning from complex questions. This work advances the development of more sophisticated tools to support the intricate information needs of the scientific community. 

---
# Infinity Instruct: Scaling Instruction Selection and Synthesis to Enhance Language Models 

**Authors**: Jijie Li, Li Du, Hanyu Zhao, Bo-wen Zhang, Liangdong Wang, Boyan Gao, Guang Liu, Yonghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11116)  

**Abstract**: Large Language Models (LLMs) demonstrate strong performance in real-world applications, yet existing open-source instruction datasets often concentrate on narrow domains, such as mathematics or coding, limiting generalization and widening the gap with proprietary models. To bridge this gap, we introduce Infinity-Instruct, a high-quality instruction dataset designed to enhance both foundational and chat capabilities of LLMs through a two-phase pipeline. In Phase 1, we curate 7.4M high-quality foundational instructions (InfInstruct-F-7.4M) from over 100M samples using hybrid data selection techniques. In Phase 2, we synthesize 1.5M high-quality chat instructions (InfInstruct-G-1.5M) through a two-stage process involving instruction selection, evolution, and diagnostic filtering. We empirically evaluate Infinity-Instruct by fine-tuning several open-source models, including Mistral, LLaMA, Qwen, and Yi, and observe substantial performance gains across both foundational and instruction following benchmarks, consistently surpassing official instruction-tuned counterparts. Notably, InfInstruct-LLaMA3.1-70B outperforms GPT-4-0314 by 8.6\% on instruction following tasks while achieving comparable foundational performance. These results underscore the synergy between foundational and chat training and offer new insights into holistic LLM development. Our dataset\footnote{this https URL} and codes\footnote{this https URL} have been publicly released. 

---
# Incorporating Domain Knowledge into Materials Tokenization 

**Authors**: Yerim Oh, Jun-Hyung Park, Junho Kim, SungHo Kim, SangKeun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.11115)  

**Abstract**: While language models are increasingly utilized in materials science, typical models rely on frequency-centric tokenization methods originally developed for natural language processing. However, these methods frequently produce excessive fragmentation and semantic loss, failing to maintain the structural and semantic integrity of material concepts. To address this issue, we propose MATTER, a novel tokenization approach that integrates material knowledge into tokenization. Based on MatDetector trained on our materials knowledge base and a re-ranking method prioritizing material concepts in token merging, MATTER maintains the structural integrity of identified material concepts and prevents fragmentation during tokenization, ensuring their semantic meaning remains intact. The experimental results demonstrate that MATTER outperforms existing tokenization methods, achieving an average performance gain of $4\%$ and $2\%$ in the generation and classification tasks, respectively. These results underscore the importance of domain knowledge for tokenization strategies in scientific text processing. Our code is available at this https URL 

---
# KokushiMD-10: Benchmark for Evaluating Large Language Models on Ten Japanese National Healthcare Licensing Examinations 

**Authors**: Junyu Liu, Kaiqi Yan, Tianyang Wang, Qian Niu, Momoko Nagai-Tanima, Tomoki Aoyama  

**Link**: [PDF](https://arxiv.org/pdf/2506.11114)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated notable performance in medical licensing exams. However, comprehensive evaluation of LLMs across various healthcare roles, particularly in high-stakes clinical scenarios, remains a challenge. Existing benchmarks are typically text-based, English-centric, and focus primarily on medicines, which limits their ability to assess broader healthcare knowledge and multimodal reasoning. To address these gaps, we introduce KokushiMD-10, the first multimodal benchmark constructed from ten Japanese national healthcare licensing exams. This benchmark spans multiple fields, including Medicine, Dentistry, Nursing, Pharmacy, and allied health professions. It contains over 11588 real exam questions, incorporating clinical images and expert-annotated rationales to evaluate both textual and visual reasoning. We benchmark over 30 state-of-the-art LLMs, including GPT-4o, Claude 3.5, and Gemini, across both text and image-based settings. Despite promising results, no model consistently meets passing thresholds across domains, highlighting the ongoing challenges in medical AI. KokushiMD-10 provides a comprehensive and linguistically grounded resource for evaluating and advancing reasoning-centric medical AI across multilingual and multimodal clinical tasks. 

---
# Breaking the Reviewer: Assessing the Vulnerability of Large Language Models in Automated Peer Review Under Textual Adversarial Attacks 

**Authors**: Tzu-Ling Lin, Wei-Chih Chen, Teng-Fang Hsiao, Hou-I Liu, Ya-Hsin Yeh, Yu Kai Chan, Wen-Sheng Lien, Po-Yen Kuo, Philip S. Yu, Hong-Han Shuai  

**Link**: [PDF](https://arxiv.org/pdf/2506.11113)  

**Abstract**: Peer review is essential for maintaining academic quality, but the increasing volume of submissions places a significant burden on reviewers. Large language models (LLMs) offer potential assistance in this process, yet their susceptibility to textual adversarial attacks raises reliability concerns. This paper investigates the robustness of LLMs used as automated reviewers in the presence of such attacks. We focus on three key questions: (1) The effectiveness of LLMs in generating reviews compared to human reviewers. (2) The impact of adversarial attacks on the reliability of LLM-generated reviews. (3) Challenges and potential mitigation strategies for LLM-based review. Our evaluation reveals significant vulnerabilities, as text manipulations can distort LLM assessments. We offer a comprehensive evaluation of LLM performance in automated peer reviewing and analyze its robustness against adversarial attacks. Our findings emphasize the importance of addressing adversarial risks to ensure AI strengthens, rather than compromises, the integrity of scholarly communication. 

---
# Evaluating and Improving Robustness in Large Language Models: A Survey and Future Directions 

**Authors**: Kun Zhang, Le Wu, Kui Yu, Guangyi Lv, Dacao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11111)  

**Abstract**: Large Language Models (LLMs) have gained enormous attention in recent years due to their capability of understanding and generating natural languages. With the rapid development and wild-range applications (e.g., Agents, Embodied Intelligence), the robustness of LLMs has received increased attention. As the core brain of many AI applications, the robustness of LLMs requires that models should not only generate consistent contents, but also ensure the correctness and stability of generated content when dealing with unexpeted application scenarios (e.g., toxic prompts, limited noise domain data, outof-distribution (OOD) applications, etc). In this survey paper, we conduct a thorough review of the robustness of LLMs, aiming to provide a comprehensive terminology of concepts and methods around this field and facilitate the community. Specifically, we first give a formal definition of LLM robustness and present the collection protocol of this survey paper. Then, based on the types of perturbated inputs, we organize this survey from the following perspectives: 1) Adversarial Robustness: tackling the problem that prompts are manipulated intentionally, such as noise prompts, long context, data attack, etc; 2) OOD Robustness: dealing with the unexpected real-world application scenarios, such as OOD detection, zero-shot transferring, hallucinations, etc; 3) Evaluation of Robustness: summarizing the new evaluation datasets, metrics, and tools for verifying the robustness of LLMs. After reviewing the representative work from each perspective, we discuss and highlight future opportunities and research directions in this field. Meanwhile, we also organize related works and provide an easy-to-search project (this https URL) to support the community. 

---
# AssertBench: A Benchmark for Evaluating Self-Assertion in Large Language Models 

**Authors**: Jaeho Lee, Atharv Chowdhary  

**Link**: [PDF](https://arxiv.org/pdf/2506.11110)  

**Abstract**: Recent benchmarks have probed factual consistency and rhetorical robustness in Large Language Models (LLMs). However, a knowledge gap exists regarding how directional framing of factually true statements influences model agreement, a common scenario for LLM users. AssertBench addresses this by sampling evidence-supported facts from FEVEROUS, a fact verification dataset. For each (evidence-backed) fact, we construct two framing prompts: one where the user claims the statement is factually correct, and another where the user claims it is incorrect. We then record the model's agreement and reasoning. The desired outcome is that the model asserts itself, maintaining consistent truth evaluation across both framings, rather than switching its evaluation to agree with the user. AssertBench isolates framing-induced variability from the model's underlying factual knowledge by stratifying results based on the model's accuracy on the same claims when presented neutrally. In doing so, this benchmark aims to measure an LLM's ability to "stick to its guns" when presented with contradictory user assertions about the same fact. The complete source code is available at this https URL. 

---
# Enhancing Large Language Models for Mobility Analytics with Semantic Location Tokenization 

**Authors**: Yile Chen, Yicheng Tao, Yue Jiang, Shuai Liu, Han Yu, Gao Cong  

**Link**: [PDF](https://arxiv.org/pdf/2506.11109)  

**Abstract**: The widespread adoption of location-based services has led to the generation of vast amounts of mobility data, providing significant opportunities to model user movement dynamics within urban environments. Recent advancements have focused on adapting Large Language Models (LLMs) for mobility analytics. However, existing methods face two primary limitations: inadequate semantic representation of locations (i.e., discrete IDs) and insufficient modeling of mobility signals within LLMs (i.e., single templated instruction fine-tuning). To address these issues, we propose QT-Mob, a novel framework that significantly enhances LLMs for mobility analytics. QT-Mob introduces a location tokenization module that learns compact, semantically rich tokens to represent locations, preserving contextual information while ensuring compatibility with LLMs. Furthermore, QT-Mob incorporates a series of complementary fine-tuning objectives that align the learned tokens with the internal representations in LLMs, improving the model's comprehension of sequential movement patterns and location semantics. The proposed QT-Mob framework not only enhances LLMs' ability to interpret mobility data but also provides a more generalizable approach for various mobility analytics tasks. Experiments on three real-world dataset demonstrate the superior performance in both next-location prediction and mobility recovery tasks, outperforming existing deep learning and LLM-based methods. 

---
# Denoising Programming Knowledge Tracing with a Code Graph-based Tuning Adaptor 

**Authors**: Weibo Gao, Qi Liu, Rui Li, Yuze Zhao, Hao Wang, Linan Yre, Fangzhou Yao, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11107)  

**Abstract**: Programming Knowledge Tracking (PKT) aims to dynamically diagnose learners' mastery levels of programming knowledge based on their coding activities, facilitating more effective and personalized programming education. However, current PKT studies primarily focus on the implicit relationship between code content and knowledge assessment, often overlooking two types of noise signals in long-term programming activities: unwanted signals from unrelated submissions and weak signals from minor modifications. This practical challenge significantly limits model performance and application. To address this issue, we propose Coda, a Code graph-based tuning adaptor designed to enhance existing PKT models by identifying and mitigating the impact of noise. Specifically, Coda first transforms the loose code sequences submitted by each learner into a compact code graph. By leveraging this code graph, unwanted signals can be identified from a semantic similarity perspective. We then apply a cluster-aware GCN to the code graph, which improves the discrimination of weak signals and enables their clustering for identification. Finally, a lightweight yet effective adaptor is incorporated into the PKT task through optimization with two noise feature-based constraints and a navigational regularization term, to correct knowledge states affected by noise. It is worth mentioning that the Coda framework is model-agnostic and can be adapted to most existing PKT solutions. Extensive experimental results on four real-world datasets demonstrate that Coda effectively performs the PKT task in the presence of noisy programming records, outperforming typical baselines. 

---
# Graph-based RAG Enhancement via Global Query Disambiguation and Dependency-Aware Reranking 

**Authors**: Ningyuan Li, Junrui Liu, Yi Shan, Minghui Huang, Tong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.11106)  

**Abstract**: Contemporary graph-based retrieval-augmented generation (RAG) methods typically begin by extracting entities from user queries and then leverage pre-constructed knowledge graphs to retrieve related relationships and metadata. However, this pipeline's exclusive reliance on entity-level extraction can lead to the misinterpretation or omission of latent yet critical information and relations. As a result, retrieved content may be irrelevant or contradictory, and essential knowledge may be excluded, exacerbating hallucination risks and degrading the fidelity of generated responses. To address these limitations, we introduce PankRAG, a framework that combines a globally aware, hierarchical query-resolution strategy with a novel dependency-aware reranking mechanism. PankRAG first constructs a multi-level resolution path that captures both parallel and sequential interdependencies within a query, guiding large language models (LLMs) through structured reasoning. It then applies its dependency-aware reranker to exploit the dependency structure among resolved sub-questions, enriching and validating retrieval results for subsequent sub-questions. Empirical evaluations demonstrate that PankRAG consistently outperforms state-of-the-art approaches across multiple benchmarks, underscoring its robustness and generalizability. 

---
# Enabling On-Device Medical AI Assistants via Input-Driven Saliency Adaptation 

**Authors**: Uttej Kallakurik, Edward Humes, Rithvik Jonna, Xiaomin Lin, Tinoosh Mohsenin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11105)  

**Abstract**: Large Language Models (LLMs) have significant impact on the healthcare scenarios but remain prohibitively large for deployment in real-time, resource-constrained environments such as edge devices. In this work, we introduce a novel medical assistant system, optimized through our general-purpose compression framework, which tailors Large Language Models (LLMs) for deployment in specialized domains. By measuring neuron saliency on domain-specific data, our method can aggressively prune irrelevant neurons, reducing model size while preserving performance. Following pruning, we apply post-training quantization to further reduce the memory footprint, and evaluate the compressed model across medical benchmarks including MedMCQA, MedQA, and PubMedQA. We also deploy the 50\% compressed Gemma and the 67\% compressed LLaMA3 models on Jetson Orin Nano (18.7W peak) and Raspberry Pi 5 (6.3W peak), achieving real-time, energy-efficient inference under hardware constraints. 

---
# DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration 

**Authors**: Hanzhi Zhang, Heng Fan, Kewei Sha, Yan Huang, Yunhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11104)  

**Abstract**: Long-context understanding is crucial for many NLP applications, yet transformers struggle with efficiency due to the quadratic complexity of self-attention. Sparse attention methods alleviate this cost but often impose static, predefined masks, failing to capture heterogeneous attention patterns. This results in suboptimal token interactions, limiting adaptability and retrieval accuracy in long-sequence tasks. This work introduces a dynamic sparse attention mechanism that assigns adaptive masks at the attention-map level, preserving heterogeneous patterns across layers and heads. Unlike existing approaches, our method eliminates the need for fine-tuning and predefined mask structures while maintaining computational efficiency. By learning context-aware attention structures, it achieves high alignment with full-attention models, ensuring minimal performance degradation while reducing memory and compute overhead. This approach provides a scalable alternative to full attention, enabling the practical deployment of large-scale Large Language Models (LLMs) without sacrificing retrieval performance. DAM is available at: this https URL. 

---
# You Only Fine-tune Once: Many-Shot In-Context Fine-Tuning for Large Language Model 

**Authors**: Wenchong He, Liqian Peng, Zhe Jiang, Alex Go  

**Link**: [PDF](https://arxiv.org/pdf/2506.11103)  

**Abstract**: Large language models (LLMs) possess a remarkable ability to perform in-context learning (ICL), which enables them to handle multiple downstream tasks simultaneously without requiring task-specific fine-tuning. Recent studies have shown that even moderately sized LLMs, such as Mistral 7B, Gemma 7B and Llama-3 8B, can achieve ICL through few-shot in-context fine-tuning of all tasks at once. However, this approach still lags behind dedicated fine-tuning, where a separate model is trained for each individual task.
In this paper, we propose a novel approach, Many-Shot In-Context Fine-tuning (ManyICL), which significantly narrows this performance gap by extending the principles of ICL to a many-shot setting. To unlock the full potential of ManyICL and address the inherent inefficiency of processing long sequences with numerous in-context examples, we propose a novel training objective. Instead of solely predicting the final answer, our approach treats every answer within the context as a supervised training target. This effectively shifts the role of many-shot examples from prompts to targets for autoregressive learning. Through extensive experiments on diverse downstream tasks, including classification, summarization, question answering, natural language inference, and math, we demonstrate that ManyICL substantially outperforms zero/few-shot fine-tuning and approaches the performance of dedicated fine-tuning. Furthermore, ManyICL significantly mitigates catastrophic forgetting issues observed in zero/few-shot fine-tuning. The code will be made publicly available upon publication. 

---
# Evolutionary Perspectives on the Evaluation of LLM-Based AI Agents: A Comprehensive Survey 

**Authors**: Jiachen Zhu, Menghui Zhu, Renting Rui, Rong Shan, Congmin Zheng, Bo Chen, Yunjia Xi, Jianghao Lin, Weiwen Liu, Ruiming Tang, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11102)  

**Abstract**: The advent of large language models (LLMs), such as GPT, Gemini, and DeepSeek, has significantly advanced natural language processing, giving rise to sophisticated chatbots capable of diverse language-related tasks. The transition from these traditional LLM chatbots to more advanced AI agents represents a pivotal evolutionary step. However, existing evaluation frameworks often blur the distinctions between LLM chatbots and AI agents, leading to confusion among researchers selecting appropriate benchmarks. To bridge this gap, this paper introduces a systematic analysis of current evaluation approaches, grounded in an evolutionary perspective. We provide a detailed analytical framework that clearly differentiates AI agents from LLM chatbots along five key aspects: complex environment, multi-source instructor, dynamic feedback, multi-modal perception, and advanced capability. Further, we categorize existing evaluation benchmarks based on external environments driving forces, and resulting advanced internal capabilities. For each category, we delineate relevant evaluation attributes, presented comprehensively in practical reference tables. Finally, we synthesize current trends and outline future evaluation methodologies through four critical lenses: environment, agent, evaluator, and metrics. Our findings offer actionable guidance for researchers, facilitating the informed selection and application of benchmarks in AI agent evaluation, thus fostering continued advancement in this rapidly evolving research domain. 

---
# An Active Learning-Based Streaming Pipeline for Reduced Data Training of Structure Finding Models in Neutron Diffractometry 

**Authors**: Tianle Wang, Jorge Ramirez, Cristina Garcia-Cardona, Thomas Proffen, Shantenu Jha, Sudip K. Seal  

**Link**: [PDF](https://arxiv.org/pdf/2506.11100)  

**Abstract**: Structure determination workloads in neutron diffractometry are computationally expensive and routinely require several hours to many days to determine the structure of a material from its neutron diffraction patterns. The potential for machine learning models trained on simulated neutron scattering patterns to significantly speed up these tasks have been reported recently. However, the amount of simulated data needed to train these models grows exponentially with the number of structural parameters to be predicted and poses a significant computational challenge. To overcome this challenge, we introduce a novel batch-mode active learning (AL) policy that uses uncertainty sampling to simulate training data drawn from a probability distribution that prefers labelled examples about which the model is least certain. We confirm its efficacy in training the same models with about 75% less training data while improving the accuracy. We then discuss the design of an efficient stream-based training workflow that uses this AL policy and present a performance study on two heterogeneous platforms to demonstrate that, compared with a conventional training workflow, the streaming workflow delivers about 20% shorter training time without any loss of accuracy. 

---
# Knowledge Graph Embeddings with Representing Relations as Annular Sectors 

**Authors**: Huiling Zhu, Yingqi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11099)  

**Abstract**: Knowledge graphs (KGs), structured as multi-relational data of entities and relations, are vital for tasks like data analysis and recommendation systems. Knowledge graph completion (KGC), or link prediction, addresses incompleteness of KGs by inferring missing triples (h, r, t). It is vital for downstream applications. Region-based embedding models usually embed entities as points and relations as geometric regions to accomplish the task. Despite progress, these models often overlook semantic hierarchies inherent in entities. To solve this problem, we propose SectorE, a novel embedding model in polar coordinates. Relations are modeled as annular sectors, combining modulus and phase to capture inference patterns and relation attributes. Entities are embedded as points within these sectors, intuitively encoding hierarchical structure. Evaluated on FB15k-237, WN18RR, and YAGO3-10, SectorE achieves competitive performance against various kinds of models, demonstrating strengths in semantic modeling capability. 

---
# Debiasing Online Preference Learning via Preference Feature Preservation 

**Authors**: Dongyoung Kim, Jinsung Yoon, Jinwoo Shin, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.11098)  

**Abstract**: Recent preference learning frameworks for large language models (LLMs) simplify human preferences with binary pairwise comparisons and scalar rewards. This simplification could make LLMs' responses biased to mostly preferred features, and would be exacerbated during the iterations of online preference learning steps. To address these challenges, we propose a novel framework coined PFP (Preference Feature Preservation). The key idea of PFP is maintaining the distribution of human preference features and utilizing such rich signals throughout the online preference learning process. Specifically, PFP first extract preference features from offline pairwise human preference data and trains a feature classifier. Then, using trained classifier and the distribution preserving optimization, PFP maps appropriate preference features for a new input instruction during online learning. Lastly, PFP trains LLM using the existing preference learning method, by incorporating the preference feature into system prompts and enabling LLM to explicitly handle various human preferences. Our experiments demonstrate that PFP successfully mitigates the bias in preference features during online learning, and hence achieves superior performance compared to previous preference learning methods on standard benchmarks to evaluate LLM alignment. 

---
# C-SEO Bench: Does Conversational SEO Work? 

**Authors**: Haritz Puerto, Martin Gubri, Tommaso Green, Seong Joon Oh, Sangdoo Yun  

**Link**: [PDF](https://arxiv.org/pdf/2506.11097)  

**Abstract**: Large Language Models (LLMs) are transforming search engines into Conversational Search Engines (CSE). Consequently, Search Engine Optimization (SEO) is being shifted into Conversational Search Engine Optimization (C-SEO). We are beginning to see dedicated C-SEO methods for modifying web documents to increase their visibility in CSE responses. However, they are often tested only for a limited breadth of application domains; we do not understand whether certain C-SEO methods would be effective for a broad range of domains. Moreover, existing evaluations consider only a single-actor scenario where only one web document adopts a C-SEO method; in reality, multiple players are likely to competitively adopt the cutting-edge C-SEO techniques, drawing an analogy from the dynamics we have seen in SEO. We present C-SEO Bench, the first benchmark designed to evaluate C-SEO methods across multiple tasks, domains, and number of actors. We consider two search tasks, question answering and product recommendation, with three domains each. We also formalize a new evaluation protocol with varying adoption rates among involved actors. Our experiments reveal that most current C-SEO methods are largely ineffective, contrary to reported results in the literature. Instead, traditional SEO strategies, those aiming to improve the ranking of the source in the LLM context, are significantly more effective. We also observe that as we increase the number of C-SEO adopters, the overall gains decrease, depicting a congested and zero-sum nature of the problem. Our code and data are available at this https URL and this https URL. 

---
# Assessing the Impact of Anisotropy in Neural Representations of Speech: A Case Study on Keyword Spotting 

**Authors**: Guillaume Wisniewski, Séverine Guillaume, Clara Rosina Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2506.11096)  

**Abstract**: Pretrained speech representations like wav2vec2 and HuBERT exhibit strong anisotropy, leading to high similarity between random embeddings. While widely observed, the impact of this property on downstream tasks remains unclear. This work evaluates anisotropy in keyword spotting for computational documentary linguistics. Using Dynamic Time Warping, we show that despite anisotropy, wav2vec2 similarity measures effectively identify words without transcription. Our results highlight the robustness of these representations, which capture phonetic structures and generalize across speakers. Our results underscore the importance of pretraining in learning rich and invariant speech representations. 

---
# Persistent Homology of Topic Networks for the Prediction of Reader Curiosity 

**Authors**: Manuel D. S. Hopp, Vincent Labatut, Arthur Amalvy, Richard Dufour, Hannah Stone, Hayley Jach, Kou Murayama  

**Link**: [PDF](https://arxiv.org/pdf/2506.11095)  

**Abstract**: Reader curiosity, the drive to seek information, is crucial for textual engagement, yet remains relatively underexplored in NLP. Building on Loewenstein's Information Gap Theory, we introduce a framework that models reader curiosity by quantifying semantic information gaps within a text's semantic structure. Our approach leverages BERTopic-inspired topic modeling and persistent homology to analyze the evolving topology (connected components, cycles, voids) of a dynamic semantic network derived from text segments, treating these features as proxies for information gaps. To empirically evaluate this pipeline, we collect reader curiosity ratings from participants (n = 49) as they read S. Collins's ''The Hunger Games'' novel. We then use the topological features from our pipeline as independent variables to predict these ratings, and experimentally show that they significantly improve curiosity prediction compared to a baseline model (73% vs. 30% explained deviance), validating our approach. This pipeline offers a new computational method for analyzing text structure and its relation to reader engagement. 

---
# The Scales of Justitia: A Comprehensive Survey on Safety Evaluation of LLMs 

**Authors**: Songyang Liu, Chaozhuo Li, Jiameng Qiu, Xi Zhang, Feiran Huang, Litian Zhang, Yiming Hei, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11094)  

**Abstract**: With the rapid advancement of artificial intelligence technology, Large Language Models (LLMs) have demonstrated remarkable potential in the field of Natural Language Processing (NLP), including areas such as content generation, human-computer interaction, machine translation, and code generation, among others. However, their widespread deployment has also raised significant safety concerns. In recent years, LLM-generated content has occasionally exhibited unsafe elements like toxicity and bias, particularly in adversarial scenarios, which has garnered extensive attention from both academia and industry. While numerous efforts have been made to evaluate the safety risks associated with LLMs, there remains a lack of systematic reviews summarizing these research endeavors. This survey aims to provide a comprehensive and systematic overview of recent advancements in LLMs safety evaluation, focusing on several key aspects: (1) "Why evaluate" that explores the background of LLMs safety evaluation, how they differ from general LLMs evaluation, and the significance of such evaluation; (2) "What to evaluate" that examines and categorizes existing safety evaluation tasks based on key capabilities, including dimensions such as toxicity, robustness, ethics, bias and fairness, truthfulness, and so on; (3) "Where to evaluate" that summarizes the evaluation metrics, datasets and benchmarks currently used in safety evaluations; (4) "How to evaluate" that reviews existing evaluation toolkit, and categorizing mainstream evaluation methods based on the roles of the evaluators. Finally, we identify the challenges in LLMs safety evaluation and propose potential research directions to promote further advancement in this field. We emphasize the importance of prioritizing LLMs safety evaluation to ensure the safe deployment of these models in real-world applications. 

---
# Dynamic Context Tuning for Retrieval-Augmented Generation: Enhancing Multi-Turn Planning and Tool Adaptation 

**Authors**: Jubin Abhishek Soni, Amit Anand, Rajesh Kumar Pandey, Aniket Abhishek Soni  

**Link**: [PDF](https://arxiv.org/pdf/2506.11092)  

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly advanced large language models (LLMs) by grounding their outputs in external tools and knowledge sources. However, existing RAG systems are typically constrained to static, single-turn interactions with fixed toolsets, making them ill-suited for dynamic domains such as healthcare and smart homes, where user intent, available tools, and contextual factors evolve over time. We present Dynamic Context Tuning (DCT), a lightweight framework that extends RAG to support multi-turn dialogue and evolving tool environments without requiring retraining. DCT integrates an attention-based context cache to track relevant past information, LoRA-based retrieval to dynamically select domain-specific tools, and efficient context compression to maintain inputs within LLM context limits. Experiments on both synthetic and real-world benchmarks show that DCT improves plan accuracy by 14% and reduces hallucinations by 37%, while matching GPT-4 performance at significantly lower cost. Furthermore, DCT generalizes to previously unseen tools, enabling scalable and adaptable AI assistants across a wide range of dynamic environments. 

---
# Better Pseudo-labeling with Multi-ASR Fusion and Error Correction by SpeechLLM 

**Authors**: Jeena Prakash, Blessingh Kumar, Kadri Hacioglu, Bidisha Sharma, Sindhuja Gopalan, Malolan Chetlur, Shankar Venkatesan, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2506.11089)  

**Abstract**: Automatic speech recognition (ASR) models rely on high-quality transcribed data for effective training. Generating pseudo-labels for large unlabeled audio datasets often relies on complex pipelines that combine multiple ASR outputs through multi-stage processing, leading to error propagation, information loss and disjoint optimization. We propose a unified multi-ASR prompt-driven framework using postprocessing by either textual or speech-based large language models (LLMs), replacing voting or other arbitration logic for reconciling the ensemble outputs. We perform a comparative study of multiple architectures with and without LLMs, showing significant improvements in transcription accuracy compared to traditional methods. Furthermore, we use the pseudo-labels generated by the various approaches to train semi-supervised ASR models for different datasets, again showing improved performance with textual and speechLLM transcriptions compared to baselines. 

---
# Two Birds with One Stone: Improving Factuality and Faithfulness of LLMs via Dynamic Interactive Subspace Editing 

**Authors**: Pengbo Wang, Chaozhuo Li, Chenxu Wang, Liwen Zheng, Litian Zhang, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11088)  

**Abstract**: LLMs have demonstrated unprecedented capabilities in natural language processing, yet their practical deployment remains hindered by persistent factuality and faithfulness hallucinations. While existing methods address these hallucination types independently, they inadvertently induce performance trade-offs, as interventions targeting one type often exacerbate the other. Through empirical and theoretical analysis of activation space dynamics in LLMs, we reveal that these hallucination categories share overlapping subspaces within neural representations, presenting an opportunity for concurrent mitigation. To harness this insight, we propose SPACE, a unified framework that jointly enhances factuality and faithfulness by editing shared activation subspaces. SPACE establishes a geometric foundation for shared subspace existence through dual-task feature modeling, then identifies and edits these subspaces via a hybrid probe strategy combining spectral clustering and attention head saliency scoring. Experimental results across multiple benchmark datasets demonstrate the superiority of our approach. 

---
# ADAMIX: Adaptive Mixed-Precision Delta-Compression with Quantization Error Optimization for Large Language Models 

**Authors**: Boya Xiong, Shuo Wang, Weifeng Ge, Guanhua Chen, Yun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11087)  

**Abstract**: Large language models (LLMs) achieve impressive performance on various knowledge-intensive and complex reasoning tasks in different domains. In certain scenarios like multi-tenant serving, a large number of LLMs finetuned from the same base model are deployed to meet complex requirements for users. Recent works explore delta-compression approaches to quantize and compress the delta parameters between the customized LLM and the corresponding base model. However, existing works either exhibit unsatisfactory performance at high compression ratios or depend on empirical bit allocation schemes. In this work, we propose ADAMIX, an effective adaptive mixed-precision delta-compression framework. We provide a mathematical derivation of quantization error to motivate our mixed-precision compression strategy and formulate the optimal mixed-precision bit allocation scheme as the solution to a 0/1 integer linear programming problem. Our derived bit allocation strategy minimizes the quantization error while adhering to a predefined compression ratio requirement. Experimental results on various models and benchmarks demonstrate that our approach surpasses the best baseline by a considerable margin. On tasks like AIME2024 and GQA, where the norm of $\Delta \mathbf{W}$ is large and the base model lacks sufficient ability, ADAMIX outperforms the best baseline Delta-CoMe by 22.3% and 6.1% with 7B models, respectively. 

---
# Intelligibility of Text-to-Speech Systems for Mathematical Expressions 

**Authors**: Sujoy Roychowdhury, H. G. Ranjani, Sumit Soman, Nishtha Paul, Subhadip Bandyopadhyay, Siddhanth Iyengar  

**Link**: [PDF](https://arxiv.org/pdf/2506.11086)  

**Abstract**: There has been limited evaluation of advanced Text-to-Speech (TTS) models with Mathematical eXpressions (MX) as inputs. In this work, we design experiments to evaluate quality and intelligibility of five TTS models through listening and transcribing tests for various categories of MX. We use two Large Language Models (LLMs) to generate English pronunciation from LaTeX MX as TTS models cannot process LaTeX directly. We use Mean Opinion Score from user ratings and quantify intelligibility through transcription correctness using three metrics. We also compare listener preference of TTS outputs with respect to human expert rendition of same MX. Results establish that output of TTS models for MX is not necessarily intelligible, the gap in intelligibility varies across TTS models and MX category. For most categories, performance of TTS models is significantly worse than that of expert rendition. The effect of choice of LLM is limited. This establishes the need to improve TTS models for MX. 

---
# LeanExplore: A search engine for Lean 4 declarations 

**Authors**: Justin Asher  

**Link**: [PDF](https://arxiv.org/pdf/2506.11085)  

**Abstract**: The expanding Lean 4 ecosystem poses challenges for navigating its vast libraries. This paper introduces LeanExplore, a search engine for Lean 4 declarations. LeanExplore enables users to semantically search for statements, both formally and informally, across select Lean 4 packages (including Batteries, Init, Lean, Mathlib, PhysLean, and Std). This search capability is powered by a hybrid ranking strategy, integrating scores from a multi-source semantic embedding model (capturing conceptual meaning from formal Lean code, docstrings, AI-generated informal translations, and declaration titles), BM25+ for keyword-based lexical relevance, and a PageRank-based score reflecting declaration importance and interconnectedness. The search engine is accessible via a dedicated website (this https URL) and a Python API (this https URL). Furthermore, the database can be downloaded, allowing users to self-host the service. LeanExplore integrates easily with LLMs via the model context protocol (MCP), enabling users to chat with an AI assistant about Lean declarations or utilize the search engine for building theorem-proving agents. This work details LeanExplore's architecture, data processing, functionalities, and its potential to enhance Lean 4 workflows and AI-driven mathematical research 

---
# PRISM: A Transformer-based Language Model of Structured Clinical Event Data 

**Authors**: Lionel Levine, John Santerre, Alex S. Young, T. Barry Levine, Francis Campion, Majid Sarrafzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.11082)  

**Abstract**: We introduce PRISM (Predictive Reasoning in Sequential Medicine), a transformer-based architecture designed to model the sequential progression of clinical decision-making processes. Unlike traditional approaches that rely on isolated diagnostic classification, PRISM frames clinical trajectories as tokenized sequences of events - including diagnostic tests, laboratory results, and diagnoses - and learns to predict the most probable next steps in the patient diagnostic journey. Leveraging a large custom clinical vocabulary and an autoregressive training objective, PRISM demonstrates the ability to capture complex dependencies across longitudinal patient timelines. Experimental results show substantial improvements over random baselines in next-token prediction tasks, with generated sequences reflecting realistic diagnostic pathways, laboratory result progressions, and clinician ordering behaviors. These findings highlight the feasibility of applying generative language modeling techniques to structured medical event data, enabling applications in clinical decision support, simulation, and education. PRISM establishes a foundation for future advancements in sequence-based healthcare modeling, bridging the gap between machine learning architectures and real-world diagnostic reasoning. 

---
# Improving Child Speech Recognition and Reading Mistake Detection by Using Prompts 

**Authors**: Lingyun Gao, Cristian Tejedor-Garcia, Catia Cucchiarini, Helmer Strik  

**Link**: [PDF](https://arxiv.org/pdf/2506.11079)  

**Abstract**: Automatic reading aloud evaluation can provide valuable support to teachers by enabling more efficient scoring of reading exercises. However, research on reading evaluation systems and applications remains limited. We present a novel multimodal approach that leverages audio and knowledge from text resources. In particular, we explored the potential of using Whisper and instruction-tuned large language models (LLMs) with prompts to improve transcriptions for child speech recognition, as well as their effectiveness in downstream reading mistake detection. Our results demonstrate the effectiveness of prompting Whisper and prompting LLM, compared to the baseline Whisper model without prompting. The best performing system achieved state-of-the-art recognition performance in Dutch child read speech, with a word error rate (WER) of 5.1%, improving the baseline WER of 9.4%. Furthermore, it significantly improved reading mistake detection, increasing the F1 score from 0.39 to 0.73. 

---
# CLAIM: Mitigating Multilingual Object Hallucination in Large Vision-Language Models with Cross-Lingual Attention Intervention 

**Authors**: Zekai Ye, Qiming Li, Xiaocheng Feng, Libo Qin, Yichong Huang, Baohang Li, Kui Jiang, Yang Xiang, Zhirui Zhang, Yunfei Lu, Duyu Tang, Dandan Tu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.11073)  

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated impressive multimodal abilities but remain prone to multilingual object hallucination, with a higher likelihood of generating responses inconsistent with the visual input when utilizing queries in non-English languages compared to English. Most existing approaches to address these rely on pretraining or fine-tuning, which are resource-intensive. In this paper, inspired by observing the disparities in cross-modal attention patterns across languages, we propose Cross-Lingual Attention Intervention for Mitigating multilingual object hallucination (CLAIM) in LVLMs, a novel near training-free method by aligning attention patterns. CLAIM first identifies language-specific cross-modal attention heads, then estimates language shift vectors from English to the target language, and finally intervenes in the attention outputs during inference to facilitate cross-lingual visual perception capability alignment. Extensive experiments demonstrate that CLAIM achieves an average improvement of 13.56% (up to 30% in Spanish) on the POPE and 21.75% on the hallucination subsets of the MME benchmark across various languages. Further analysis reveals that multilingual attention divergence is most prominent in intermediate layers, highlighting their critical role in multilingual scenarios. 

---
# Embedded Acoustic Intelligence for Automotive Systems 

**Authors**: Renjith Rajagopal, Peter Winzell, Sladjana Strbac, Konstantin Lindström, Petter Hörling, Faisal Kohestani, Niloofar Mehrzad  

**Link**: [PDF](https://arxiv.org/pdf/2506.11071)  

**Abstract**: Transforming sound insights into actionable streams of data, this abstract leverages findings from degree thesis research to enhance automotive system intelligence, enabling us to address road type [1].By extracting and interpreting acoustic signatures from microphones installed within the wheelbase of a car, we focus on classifying road this http URL deep neural networks and feature extraction powered by pre-trained models from the Open AI ecosystem (via Hugging Face [2]), our approach enables Autonomous Driving and Advanced Driver- Assistance Systems (AD/ADAS) to anticipate road surfaces, support adaptive learning for active road noise cancellation, and generate valuable insights for urban planning. The results of this study were specifically captured to support a compelling business case for next-generation automotive systems. This forward-looking approach not only promises to redefine passenger comfort and improve vehicle safety, but also paves the way for intelligent, data-driven urban road management, making the future of mobility both achievable and sustainable. 

---
# Regularized Federated Learning for Privacy-Preserving Dysarthric and Elderly Speech Recognition 

**Authors**: Tao Zhong, Mengzhe Geng, Shujie Hu, Guinan Li, Xunying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11069)  

**Abstract**: Accurate recognition of dysarthric and elderly speech remains challenging to date. While privacy concerns have driven a shift from centralized approaches to federated learning (FL) to ensure data confidentiality, this further exacerbates the challenges of data scarcity, imbalanced data distribution and speaker heterogeneity. To this end, this paper conducts a systematic investigation of regularized FL techniques for privacy-preserving dysarthric and elderly speech recognition, addressing different levels of the FL process by 1) parameter-based, 2) embedding-based and 3) novel loss-based regularization. Experiments on the benchmark UASpeech dysarthric and DementiaBank Pitt elderly speech corpora suggest that regularized FL systems consistently outperform the baseline FedAvg system by statistically significant WER reductions of up to 0.55\% absolute (2.13\% relative). Further increasing communication frequency to one exchange per batch approaches centralized training performance. 

---
# CoQuIR: A Comprehensive Benchmark for Code Quality-Aware Information Retrieval 

**Authors**: Jiahui Geng, Fengyu Cai, Shaobo Cui, Qing Li, Liangwei Chen, Chenyang Lyu, Haonan Li, Derui Zhu, Walter Pretschner, Heinz Koeppl, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2506.11066)  

**Abstract**: Code retrieval is essential in modern software development, as it boosts code reuse and accelerates debugging. However, current benchmarks primarily emphasize functional relevance while neglecting critical dimensions of software quality. Motivated by this gap, we introduce CoQuIR, the first large-scale, multilingual benchmark specifically designed to evaluate quality-aware code retrieval across four key dimensions: correctness, efficiency, security, and maintainability. CoQuIR provides fine-grained quality annotations for 42,725 queries and 134,907 code snippets in 11 programming languages, and is accompanied by two quality-centric evaluation metrics: Pairwise Preference Accuracy and Margin-based Ranking Score. Using CoQuIR, we benchmark 23 retrieval models, covering both open-source and proprietary systems, and find that even top-performing models frequently fail to distinguish buggy or insecure code from their more robust counterparts. Furthermore, we conduct preliminary investigations into training methods that explicitly encourage retrievers to recognize code quality. Using synthetic datasets, we demonstrate promising improvements in quality-aware metrics across various models, without sacrificing semantic relevance. Downstream code generation experiments further validate the effectiveness of our approach. Overall, our work highlights the importance of integrating quality signals into code retrieval systems, laying the groundwork for more trustworthy and robust software development tools. 

---
# PMF-CEC: Phoneme-augmented Multimodal Fusion for Context-aware ASR Error Correction with Error-specific Selective Decoding 

**Authors**: Jiajun He, Tomoki Toda  

**Link**: [PDF](https://arxiv.org/pdf/2506.11064)  

**Abstract**: End-to-end automatic speech recognition (ASR) models often struggle to accurately recognize rare words. Previously, we introduced an ASR postprocessing method called error detection and context-aware error correction (ED-CEC), which leverages contextual information such as named entities and technical terms to improve the accuracy of ASR transcripts. Although ED-CEC achieves a notable success in correcting rare words, its accuracy remains low when dealing with rare words that have similar pronunciations but different spellings. To address this issue, we proposed a phoneme-augmented multimodal fusion method for context-aware error correction (PMF-CEC) method on the basis of ED-CEC, which allowed for better differentiation between target rare words and homophones. Additionally, we observed that the previous ASR error detection module suffers from overdetection. To mitigate this, we introduced a retention probability mechanism to filter out editing operations with confidence scores below a set threshold, preserving the original operation to improve error detection accuracy. Experiments conducted on five datasets demonstrated that our proposed PMF-CEC maintains reasonable inference speed while further reducing the biased word error rate compared with ED-CEC, showing a stronger advantage in correcting homophones. Moreover, our method outperforms other contextual biasing methods, and remains valuable compared with LLM-based methods in terms of faster inference and better robustness under large biasing lists. 

---
# Who is in the Spotlight: The Hidden Bias Undermining Multimodal Retrieval-Augmented Generation 

**Authors**: Jiayu Yao, Shenghua Liu, Yiwei Wang, Lingrui Mei, Baolong Bi, Yuyao Ge, Zhecheng Li, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11063)  

**Abstract**: Multimodal Retrieval-Augmented Generation (RAG) systems have become essential in knowledge-intensive and open-domain tasks. As retrieval complexity increases, ensuring the robustness of these systems is critical. However, current RAG models are highly sensitive to the order in which evidence is presented, often resulting in unstable performance and biased reasoning, particularly as the number of retrieved items or modality diversity grows. This raises a central question: How does the position of retrieved evidence affect multimodal RAG performance? To answer this, we present the first comprehensive study of position bias in multimodal RAG systems. Through controlled experiments across text-only, image-only, and mixed-modality tasks, we observe a consistent U-shaped accuracy curve with respect to evidence position. To quantify this bias, we introduce the Position Sensitivity Index ($PSI_p$) and develop a visualization framework to trace attention allocation patterns across decoder layers. Our results reveal that multimodal interactions intensify position bias compared to unimodal settings, and that this bias increases logarithmically with retrieval range. These findings offer both theoretical and empirical foundations for position-aware analysis in RAG, highlighting the need for evidence reordering or debiasing strategies to build more reliable and equitable generation systems. 

---
# Decoding Cortical Microcircuits: A Generative Model for Latent Space Exploration and Controlled Synthesis 

**Authors**: Xingyu Liu, Yubin Li, Guozhang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11062)  

**Abstract**: A central idea in understanding brains and building artificial intelligence is that structure determines function. Yet, how the brain's complex structure arises from a limited set of genetic instructions remains a key question. The ultra high-dimensional detail of neural connections vastly exceeds the information storage capacity of genes, suggesting a compact, low-dimensional blueprint must guide brain development. Our motivation is to uncover this blueprint. We introduce a generative model, to learn this underlying representation from detailed connectivity maps of mouse cortical microcircuits. Our model successfully captures the essential structural information of these circuits in a compressed latent space. We found that specific, interpretable directions within this space directly relate to understandable network properties. Building on this, we demonstrate a novel method to controllably generate new, synthetic microcircuits with desired structural features by navigating this latent space. This work offers a new way to investigate the design principles of neural circuits and explore how structure gives rise to function, potentially informing the development of more advanced artificial neural networks. 

---
# Code Researcher: Deep Research Agent for Large Systems Code and Commit History 

**Authors**: Ramneet Singh, Sathvik Joel, Abhav Mehrotra, Nalin Wadhwa, Ramakrishna B Bairi, Aditya Kanade, Nagarajan Natarajan  

**Link**: [PDF](https://arxiv.org/pdf/2506.11060)  

**Abstract**: Large Language Model (LLM)-based coding agents have shown promising results on coding benchmarks, but their effectiveness on systems code remains underexplored. Due to the size and complexities of systems code, making changes to a systems codebase is a daunting task, even for humans. It requires researching about many pieces of context, derived from the large codebase and its massive commit history, before making changes. Inspired by the recent progress on deep research agents, we design the first deep research agent for code, called Code Researcher, and apply it to the problem of generating patches for mitigating crashes reported in systems code. Code Researcher performs multi-step reasoning about semantics, patterns, and commit history of code to gather sufficient context. The context is stored in a structured memory which is used for synthesizing a patch. We evaluate Code Researcher on kBenchSyz, a benchmark of Linux kernel crashes, and show that it significantly outperforms strong baselines, achieving a crash-resolution rate of 58%, compared to 37.5% by SWE-agent. On an average, Code Researcher explores 10 files in each trajectory whereas SWE-agent explores only 1.33 files, highlighting Code Researcher's ability to deeply explore the codebase. Through another experiment on an open-source multimedia software, we show the generalizability of Code Researcher. Our experiments highlight the importance of global context gathering and multi-faceted reasoning for large codebases. 

---
# Refactoring Codebases through Library Design 

**Authors**: Ziga Kovacic, Celine Lee, Justin Chiu, Wenting Zhao, Kevin Ellis  

**Link**: [PDF](https://arxiv.org/pdf/2506.11058)  

**Abstract**: Maintainable and general software allows developers to build robust applications efficiently, yet achieving these qualities often requires refactoring specialized solutions into reusable components. This challenge becomes particularly relevant as code agents become increasingly accurate at solving isolated programming problems. We investigate code agents' capacity to refactor code in ways supporting growth and reusability. We present both a method and a benchmark for refactoring: Librarian, a sample-and-rerank method for generating reusable libraries, and Minicode, a benchmark where code agents must minimize and refactor multiple independent solutions into a joint library. Compared to state-of-the-art code agents, Librarian achieves strong results on both compression and correctness on Minicode, obtaining compression rates 1.6-2x better than coding agents while also improving correctness. We open-source our code and benchmark at this https URL. 

---
# STRCMP: Integrating Graph Structural Priors with Language Models for Combinatorial Optimization 

**Authors**: Xijun Li, Jiexiang Yang, Jinghao Wang, Bo Peng, Jianguo Yao, Haibing Guan  

**Link**: [PDF](https://arxiv.org/pdf/2506.11057)  

**Abstract**: Combinatorial optimization (CO) problems, central to operation research and theoretical computer science, present significant computational challenges due to their NP-hard nature. While large language models (LLMs) have emerged as promising tools for CO--either by directly generating solutions or synthesizing solver-specific codes--existing approaches often neglect critical structural priors inherent to CO problems, leading to suboptimality and iterative inefficiency. Inspired by human experts' success in leveraging CO structures for algorithm design, we propose STRCMP, a novel structure-aware LLM-based algorithm discovery framework that systematically integrates structure priors to enhance solution quality and solving efficiency. Our framework combines a graph neural network (GNN) for extracting structural embeddings from CO instances with an LLM conditioned on these embeddings to identify high-performing algorithms in the form of solver-specific codes. This composite architecture ensures syntactic correctness, preserves problem topology, and aligns with natural language objectives, while an evolutionary refinement process iteratively optimizes generated algorithm. Extensive evaluations across Mixed Integer Linear Programming and Boolean Satisfiability problems, using nine benchmark datasets, demonstrate that our proposed STRCMP outperforms five strong neural and LLM-based methods by a large margin, in terms of both solution optimality and computational efficiency. The code and learned model will be publicly available upon the acceptance of the paper. 

---
# xInv: Explainable Optimization of Inverse Problems 

**Authors**: Sean Memery, Kevin Denamganai, Anna Kapron-King, Kartic Subr  

**Link**: [PDF](https://arxiv.org/pdf/2506.11056)  

**Abstract**: Inverse problems are central to a wide range of fields, including healthcare, climate science, and agriculture. They involve the estimation of inputs, typically via iterative optimization, to some known forward model so that it produces a desired outcome. Despite considerable development in the explainability and interpretability of forward models, the iterative optimization of inverse problems remains largely cryptic to domain experts. We propose a methodology to produce explanations, from traces produced by an optimizer, that are interpretable by humans at the abstraction of the domain. The central idea in our approach is to instrument a differentiable simulator so that it emits natural language events during its forward and backward passes. In a post-process, we use a Language Model to create an explanation from the list of events. We demonstrate the effectiveness of our approach with an illustrative optimization problem and an example involving the training of a neural network. 

---
# Adaptive Composition of Machine Learning as a Service (MLaaS) for IoT Environments 

**Authors**: Deepak Kanneganti, Sajib Mistry, Sheik Mohammad Mostakim Fattah, Aneesh Krishna, Monowar Bhuyan  

**Link**: [PDF](https://arxiv.org/pdf/2506.11054)  

**Abstract**: The dynamic nature of Internet of Things (IoT) environments challenges the long-term effectiveness of Machine Learning as a Service (MLaaS) compositions. The uncertainty and variability of IoT environments lead to fluctuations in data distribution, e.g., concept drift and data heterogeneity, and evolving system requirements, e.g., scalability demands and resource limitations. This paper proposes an adaptive MLaaS composition framework to ensure a seamless, efficient, and scalable MLaaS composition. The framework integrates a service assessment model to identify underperforming MLaaS services and a candidate selection model to filter optimal replacements. An adaptive composition mechanism is developed that incrementally updates MLaaS compositions using a contextual multi-armed bandit optimization strategy. By continuously adapting to evolving IoT constraints, the approach maintains Quality of Service (QoS) while reducing the computational cost associated with recomposition from scratch. Experimental results on a real-world dataset demonstrate the efficiency of our proposed approach. 

---
# Bootstrapping your behavior: a new pretraining strategy for user behavior sequence data 

**Authors**: Weichang Wu, Xiaolu Zhang, Jun Zhou, Yuchen Li, Wenwen Xia  

**Link**: [PDF](https://arxiv.org/pdf/2506.11053)  

**Abstract**: User Behavior Sequence (UBS) modeling is crucial in industrial applications. As data scale and task diversity grow, UBS pretraining methods have become increasingly pivotal. State-of-the-art UBS pretraining methods rely on predicting behavior distributions. The key step in these methods is constructing a selected behavior vocabulary. However, this manual step is labor-intensive and prone to bias. The limitation of vocabulary capacity also directly affects models' generalization ability. In this paper, we introduce Bootstrapping Your Behavior (\model{}), a novel UBS pretraining strategy that predicts an automatically constructed supervision embedding summarizing all behaviors' information within a future time window, eliminating the manual behavior vocabulary selection. In implementation, we incorporate a student-teacher encoder scheme to construct the pretraining supervision effectively. Experiments on two real-world industrial datasets and eight downstream tasks demonstrate that \model{} achieves an average improvement of 3.9\% in AUC and 98.9\% in training throughput. Notably, the model exhibits meaningful attention patterns and cluster representations during pretraining without any label supervision. In our online deployment over two months, the pretrained model improves the KS by about 2.7\% and 7.1\% over the baseline model for two financial overdue risk prediction tasks in the Alipay mobile application, which reduces bad debt risk by millions of dollars for Ant group. 

---
# ACCORD: Autoregressive Constraint-satisfying Generation for COmbinatorial Optimization with Routing and Dynamic attention 

**Authors**: Henrik Abgaryan, Tristan Cazenave, Ararat Harutyunyan  

**Link**: [PDF](https://arxiv.org/pdf/2506.11052)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive reasoning capabilities, yet their direct application to NP-hard combinatorial problems (CPs) remains underexplored. In this work, we systematically investigate the reasoning abilities of LLMs on a variety of NP-hard combinatorial optimization tasks and introduce ACCORD: Autoregressive Constraint-satisfying generation for COmbinatorial optimization with Routing and Dynamic attention. ACCORD features a novel dataset representation and model architecture that leverage the autoregressive nature of LLMs to dynamically enforce feasibility constraints, coupled with attention-based routing to activate problem-specific LoRA modules. We also present the ACCORD-90k supervised dataset, covering six NP-hard combinatorial problems: TSP, VRP, Knapsack, FlowShop, JSSP, and BinPacking. Extensive experiments demonstrate that our ACCORD model, built on an 8B-parameter Llama backbone, consistently outperforms standard prompting and input-output methods, even when compared to much larger LLMs, such as gpt-4. Ablation studies further show that our output structure enhances solution feasibility. To the best of our knowledge, this is the first large-scale, end-to-end framework for exploring the applications of LLMs to a broad spectrum of combinatorial optimization problems. The codes are publicly available at this https URL 

---
# 15,500 Seconds: Lean UAV Classification Leveraging PEFT and Pre-Trained Networks 

**Authors**: Andrew P. Berg, Qian Zhang, Mia Y. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11049)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) pose an escalating security concerns as the market for consumer and military UAVs grows. This paper address the critical data scarcity challenges in deep UAV audio classification. We build upon our previous work expanding novel approaches such as: parameter efficient fine-tuning, data augmentation, and pre-trained networks. We achieve performance upwards of 95\% validation accuracy with EfficientNet-B0. 

---
# I Can't Believe It's Not Real: CV-MuSeNet: Complex-Valued Multi-Signal Segmentation 

**Authors**: Sangwon Shin, Mehmet C. Vuran  

**Link**: [PDF](https://arxiv.org/pdf/2506.11048)  

**Abstract**: The increasing congestion of the radio frequency spectrum presents challenges for efficient spectrum utilization. Cognitive radio systems enable dynamic spectrum access with the aid of recent innovations in neural networks. However, traditional real-valued neural networks (RVNNs) face difficulties in low signal-to-noise ratio (SNR) environments, as they were not specifically developed to capture essential wireless signal properties such as phase and amplitude. This work presents CMuSeNet, a complex-valued multi-signal segmentation network for wideband spectrum sensing, to address these limitations. Extensive hyperparameter analysis shows that a naive conversion of existing RVNNs into their complex-valued counterparts is ineffective. Built on complex-valued neural networks (CVNNs) with a residual architecture, CMuSeNet introduces a complexvalued Fourier spectrum focal loss (CFL) and a complex plane intersection over union (CIoU) similarity metric to enhance training performance. Extensive evaluations on synthetic, indoor overthe-air, and real-world datasets show that CMuSeNet achieves an average accuracy of 98.98%-99.90%, improving by up to 9.2 percentage points over its real-valued counterpart and consistently outperforms state of the art. Strikingly, CMuSeNet achieves the accuracy level of its RVNN counterpart in just two epochs, compared to the 27 epochs required for RVNN, while reducing training time by up to a 92.2% over the state of the art. The results highlight the effectiveness of complex-valued architectures in improving weak signal detection and training efficiency for spectrum sensing in challenging low-SNR environments. The dataset is available at: this https URL 

---
# Angle Domain Guidance: Latent Diffusion Requires Rotation Rather Than Extrapolation 

**Authors**: Cheng Jin, Zhenyu Xiao, Chutao Liu, Yuantao Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11039)  

**Abstract**: Classifier-free guidance (CFG) has emerged as a pivotal advancement in text-to-image latent diffusion models, establishing itself as a cornerstone technique for achieving high-quality image synthesis. However, under high guidance weights, where text-image alignment is significantly enhanced, CFG also leads to pronounced color distortions in the generated images. We identify that these distortions stem from the amplification of sample norms in the latent space. We present a theoretical framework that elucidates the mechanisms of norm amplification and anomalous diffusion phenomena induced by classifier-free guidance. Leveraging our theoretical insights and the latent space structure, we propose an Angle Domain Guidance (ADG) algorithm. ADG constrains magnitude variations while optimizing angular alignment, thereby mitigating color distortions while preserving the enhanced text-image alignment achieved at higher guidance weights. Experimental results demonstrate that ADG significantly outperforms existing methods, generating images that not only maintain superior text alignment but also exhibit improved color fidelity and better alignment with human perceptual preferences. 

---
# Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity 

**Authors**: Moussa Koulako Bala Doumbouya, Dan Jurafsky, Christopher D. Manning  

**Link**: [PDF](https://arxiv.org/pdf/2506.11035)  

**Abstract**: Work in psychology has highlighted that the geometric model of similarity standard in deep learning is not psychologically plausible because its metric properties such as symmetry do not align with human perception. In contrast, Tversky (1977) proposed an axiomatic theory of similarity based on a representation of objects as sets of features, and their similarity as a function of common and distinctive features. However, this model has not been used in deep learning before, partly due to the challenge of incorporating discrete set operations. We develop a differentiable parameterization of Tversky's similarity that is learnable through gradient descent, and derive neural network building blocks such as the Tversky projection layer, which unlike the linear projection layer can model non-linear functions such as XOR. Through experiments with image recognition and language modeling, we show that the Tversky projection layer is a beneficial replacement for the linear projection layer, which employs geometric similarity. On the NABirds image classification task, a frozen ResNet-50 adapted with a Tversky projection layer achieves a 24.7% relative accuracy improvement over the linear layer adapter baseline. With Tversky projection layers, GPT-2's perplexity on PTB decreases by 7.5%, and its parameter count by 34.8%. Finally, we propose a unified interpretation of both projection layers as computing similarities of input stimuli to learned prototypes, for which we also propose a novel visualization technique highlighting the interpretability of Tversky projection layers. Our work offers a new paradigm for thinking about the similarity model implicit in deep learning, and designing networks that are interpretable under an established theory of psychological similarity. 

---
# CausalVLBench: Benchmarking Visual Causal Reasoning in Large Vision-Language Models 

**Authors**: Aneesh Komanduri, Karuna Bhaila, Xintao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11034)  

**Abstract**: Large language models (LLMs) have shown remarkable ability in various language tasks, especially with their emergent in-context learning capability. Extending LLMs to incorporate visual inputs, large vision-language models (LVLMs) have shown impressive performance in tasks such as recognition and visual question answering (VQA). Despite increasing interest in the utility of LLMs in causal reasoning tasks such as causal discovery and counterfactual reasoning, there has been relatively little work showcasing the abilities of LVLMs on visual causal reasoning tasks. We take this opportunity to formally introduce a comprehensive causal reasoning benchmark for multi-modal in-context learning from LVLMs. Our CausalVLBench encompasses three representative tasks: causal structure inference, intervention target prediction, and counterfactual prediction. We evaluate the ability of state-of-the-art open-source LVLMs on our causal reasoning tasks across three causal representation learning datasets and demonstrate their fundamental strengths and weaknesses. We hope that our benchmark elucidates the drawbacks of existing vision-language models and motivates new directions and paradigms in improving the visual causal reasoning abilities of LVLMs. 

---
# Runtime Safety through Adaptive Shielding: From Hidden Parameter Inference to Provable Guarantees 

**Authors**: Minjae Kwon, Tyler Ingebrand, Ufuk Topcu, Lu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11033)  

**Abstract**: Variations in hidden parameters, such as a robot's mass distribution or friction, pose safety risks during execution. We develop a runtime shielding mechanism for reinforcement learning, building on the formalism of constrained hidden-parameter Markov decision processes. Function encoders enable real-time inference of hidden parameters from observations, allowing the shield and the underlying policy to adapt online. The shield constrains the action space by forecasting future safety risks (such as obstacle proximity) and accounts for uncertainty via conformal prediction. We prove that the proposed mechanism satisfies probabilistic safety guarantees and yields optimal policies among the set of safety-compliant policies. Experiments across diverse environments with varying hidden parameters show that our method significantly reduces safety violations and achieves strong out-of-distribution generalization, while incurring minimal runtime overhead. 

---
# Task-aligned prompting improves zero-shot detection of AI-generated images by Vision-Language Models 

**Authors**: Zoher Kachwala, Danishjeet Singh, Danielle Yang, Filippo Menczer  

**Link**: [PDF](https://arxiv.org/pdf/2506.11031)  

**Abstract**: As image generators produce increasingly realistic images, concerns about potential misuse continue to grow. Supervised detection relies on large, curated datasets and struggles to generalize across diverse generators. In this work, we investigate the use of pre-trained Vision-Language Models (VLMs) for zero-shot detection of AI-generated images. While off-the-shelf VLMs exhibit some task-specific reasoning and chain-of-thought prompting offers gains, we show that task-aligned prompting elicits more focused reasoning and significantly improves performance without fine-tuning. Specifically, prefixing the model's response with the phrase ``Let's examine the style and the synthesis artifacts'' -- a method we call zero-shot-s$^2$ -- boosts Macro F1 scores by 8%-29% for two widely used open-source models. These gains are consistent across three recent, diverse datasets spanning human faces, objects, and animals with images generated by 16 different models -- demonstrating strong generalization. We further evaluate the approach across three additional model sizes and observe improvements in most dataset-model combinations -- suggesting robustness to model scale. Surprisingly, self-consistency, a behavior previously observed in language reasoning, where aggregating answers from diverse reasoning paths improves performance, also holds in this setting. Even here, zero-shot-s$^2$ scales better than chain-of-thought in most cases -- indicating that it elicits more useful diversity. Our findings show that task-aligned prompts elicit more focused reasoning and enhance latent capabilities in VLMs, like the detection of AI-generated images -- offering a simple, generalizable, and explainable alternative to supervised methods. Our code is publicly available on github: this https URL. 

---
# Forward Target Propagation: A Forward-Only Approach to Global Error Credit Assignment via Local Losses 

**Authors**: Nazmus Saadat As-Saquib, A N M Nafiz Abeer, Hung-Ta Chien, Byung-Jun Yoon, Suhas Kumar, Su-in Yi  

**Link**: [PDF](https://arxiv.org/pdf/2506.11030)  

**Abstract**: Training neural networks has traditionally relied on backpropagation (BP), a gradient-based algorithm that, despite its widespread success, suffers from key limitations in both biological and hardware perspectives. These include backward error propagation by symmetric weights, non-local credit assignment, and frozen activity during backward passes. We propose Forward Target Propagation (FTP), a biologically plausible and computationally efficient alternative that replaces the backward pass with a second forward pass. FTP estimates layerwise targets using only feedforward computations, eliminating the need for symmetric feedback weights or learnable inverse functions, hence enabling modular and local learning. We evaluate FTP on fully connected networks, CNNs, and RNNs, demonstrating accuracies competitive with BP on MNIST, CIFAR10, and CIFAR100, as well as effective modeling of long-term dependencies in sequential tasks. Moreover, FTP outperforms BP under quantized low-precision and emerging hardware constraints while also demonstrating substantial efficiency gains over other biologically inspired methods such as target propagation variants and forward-only learning algorithms. With its minimal computational overhead, forward-only nature, and hardware compatibility, FTP provides a promising direction for energy-efficient on-device learning and neuromorphic computing. 

---
# Output Scaling: YingLong-Delayed Chain of Thought in a Large Pretrained Time Series Forecasting Model 

**Authors**: Xue Wang, Tian Zhou, Jinyang Gao, Bolin Ding, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.11029)  

**Abstract**: We present a joint forecasting framework for time series prediction that contrasts with traditional direct or recursive methods. This framework achieves state-of-the-art performance for our designed foundation model, YingLong, and reveals a novel scaling effect: longer outputs significantly enhance model accuracy due to delayed chain-of-thought reasoning in our non-causal approach. YingLong is a non-causal, bidirectional attention encoder-only transformer trained through masked token recovery, aligning more effectively with language understanding tasks than with generation tasks. Additionally, we boost performance by tackling output variance with a multi-input ensemble. We release four foundation models ranging from 6M to 300M parameters, demonstrating superior results in zero-shot tasks on the ETT and Weather datasets. YingLong achieves more than 60% best performance. To ensure generalizability, we assessed the models using the GIFT-Eval benchmark, which comprises 23 time series datasets across 7 domains. Yinglong significantly outperformed the best time-series foundation models, end-to-end trained models by 14% and 44% in rank this http URL pretrained 300M model is available at this https URL 

---
# Enhancing Epidemic Forecasting: Evaluating the Role of Mobility Data and Graph Convolutional Networks 

**Authors**: Suhan Guo, Zhenghao Xu, Furao Shen, Jian Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.11028)  

**Abstract**: Accurate prediction of contagious disease outbreaks is vital for informed decision-making. Our study addresses the gap between machine learning algorithms and their epidemiological applications, noting that methods optimal for benchmark datasets often underperform with real-world data due to difficulties in incorporating mobility information. We adopt a two-phase approach: first, assessing the significance of mobility data through a pilot study, then evaluating the impact of Graph Convolutional Networks (GCNs) on a transformer backbone. Our findings reveal that while mobility data and GCN modules do not significantly enhance forecasting performance, the inclusion of mortality and hospitalization data markedly improves model accuracy. Additionally, a comparative analysis between GCN-derived spatial maps and lockdown orders suggests a notable correlation, highlighting the potential of spatial maps as sensitive indicators for mobility. Our research offers a novel perspective on mobility representation in predictive modeling for contagious diseases, empowering decision-makers to better prepare for future outbreaks. 

---
# From Reasoning to Code: GRPO Optimization for Underrepresented Languages 

**Authors**: Federico Pennino, Bianca Raimondi, Massimo Rondelli, Andrea Gurioli, Maurizio Gabbrielli  

**Link**: [PDF](https://arxiv.org/pdf/2506.11027)  

**Abstract**: Generating accurate and executable code using large language models (LLMs) is challenging for languages with limited public training data compared to popular languages such as Python. This paper introduces a generalizable approach that uses small-scale code versions of the Qwen 2.5 model combined with Group Relative Policy Optimization (GRPO) to enable effective code generation through explicit reasoning steps, which is particularly beneficial for languages with smaller source code databases. Using Prolog as a representative use case -- given its limited online presence -- the initial model faced challenges in generating executable code. After some training steps, the model successfully produces logically consistent and syntactically accurate code by directly integrating reasoning-driven feedback into the reinforcement learning loop. Experimental evaluations using mathematical logic problem benchmarks illustrate significant improvements in reasoning quality, code accuracy, and logical correctness, underscoring the potential of this approach to benefit a wide range of programming languages lacking extensive training resources. 

---
# When Algorithms Play Favorites: Lookism in the Generation and Perception of Faces 

**Authors**: Miriam Doh, Aditya Gulati, Matei Mancas, Nuria Oliver  

**Link**: [PDF](https://arxiv.org/pdf/2506.11025)  

**Abstract**: This paper examines how synthetically generated faces and machine learning-based gender classification algorithms are affected by algorithmic lookism, the preferential treatment based on appearance. In experiments with 13,200 synthetically generated faces, we find that: (1) text-to-image (T2I) systems tend to associate facial attractiveness to unrelated positive traits like intelligence and trustworthiness; and (2) gender classification models exhibit higher error rates on "less-attractive" faces, especially among non-White women. These result raise fairness concerns regarding digital identity systems. 

---
# Not All Clients Are Equal: Personalized Federated Learning on Heterogeneous Multi-Modal Clients 

**Authors**: Minhyuk Seo, Taeheon Kim, Hankook Lee, Jonghyun Choi, Tinne Tuytelaars  

**Link**: [PDF](https://arxiv.org/pdf/2506.11024)  

**Abstract**: Foundation models have shown remarkable capabilities across diverse multi-modal tasks, but their centralized training raises privacy concerns and induces high transmission costs. In contrast, federated learning (FL) offers a distributed alternative without the need to share data. Recently, for the growing demand for personalizing AI models for different user purposes, personalized federated learning (PFL) has emerged. PFL allows each client to leverage the knowledge of other clients for further adaptation to individual user preferences, again without the need to share data. Despite its potential, most PFL studies remain confined to simulated environments, overlooking the data and model heterogeneity that arise in real-world scenarios. In contrast, we first consider large data heterogeneity, evaluating on a new benchmark for multi-modal PFL, spanning 40 distinct tasks with realistic data distribution shifts. We then consider model heterogeneity in that we do not assume that all clients share similar model architectures. To address data heterogeneity, we propose a task-similarity-aware model aggregation method that provides customized global models to each client. For model heterogeneity, we propose a dimension-invariant module that enables knowledge sharing across heterogeneous models. Empirical validations demonstrate that the proposed approach outperforms the state-of-the-art, excelling in both personalization and generalization capabilities. 

---
# Security Degradation in Iterative AI Code Generation -- A Systematic Analysis of the Paradox 

**Authors**: Shivani Shukla, Himanshu Joshi, Romilla Syed  

**Link**: [PDF](https://arxiv.org/pdf/2506.11022)  

**Abstract**: The rapid adoption of Large Language Models(LLMs) for code generation has transformed software development, yet little attention has been given to how security vulnerabilities evolve through iterative LLM feedback. This paper analyzes security degradation in AI-generated code through a controlled experiment with 400 code samples across 40 rounds of "improvements" using four distinct prompting strategies. Our findings show a 37.6% increase in critical vulnerabilities after just five iterations, with distinct vulnerability patterns emerging across different prompting approaches. This evidence challenges the assumption that iterative LLM refinement improves code security and highlights the essential role of human expertise in the loop. We propose practical guidelines for developers to mitigate these risks, emphasizing the need for robust human validation between LLM iterations to prevent the paradoxical introduction of new security issues during supposedly beneficial code "improvements". 

---
# Eliminating Hallucination-Induced Errors in LLM Code Generation with Functional Clustering 

**Authors**: Chaitanya Ravuri, Saman Amarasinghe  

**Link**: [PDF](https://arxiv.org/pdf/2506.11021)  

**Abstract**: Modern code-generation LLMs can already solve a large fraction of programming problems, yet they still hallucinate subtle bugs that make their outputs unsafe for autonomous deployment. We present functional clustering, a black-box wrapper that eliminates nearly all hallucination-induced errors while providing a tunable confidence score. The wrapper samples many candidate programs, executes each on a self-generated test suite, and clusters candidates whose I/O behavior is identical; the empirical mass of the largest cluster serves as an exact confidence estimate. A single scalar threshold on this estimate lets users trade coverage for reliability with exponential guarantees. On LiveCodeBench our verifier preserves baseline pass@1 on solvable tasks yet slashes the error rate of returned answers from ~65% to 2%, and drives it to 0% at a conservative threshold while still answering 15.6% of prompts. Manual audits show that the few residual mistakes stem from prompt misinterpretation, not random generation noise, narrowing future work to specification clarity. Because the method requires only sampling and sandbox execution, it applies unchanged to closed-source APIs and future models, offering a practical path toward dependable, autonomous code generation. Our code is available on Github (this https URL). 

---
# Extracting Knowledge Graphs from User Stories using LangChain 

**Authors**: Thayná Camargo da Silva  

**Link**: [PDF](https://arxiv.org/pdf/2506.11020)  

**Abstract**: This thesis introduces a novel methodology for the automated generation of knowledge graphs from user stories by leveraging the advanced capabilities of Large Language Models. Utilizing the LangChain framework as a basis, the User Story Graph Transformer module was developed to extract nodes and relationships from user stories using an LLM to construct accurate knowledge this http URL innovative technique was implemented in a script to fully automate the knowledge graph extraction process. Additionally, the evaluation was automated through a dedicated evaluation script, utilizing an annotated dataset for assessment. By enhancing the visualization and understanding of user requirements and domain concepts, this method fosters better alignment between software functionalities and user expectations, ultimately contributing to more effective and user-centric software development processes. 

---
# TeleEval-OS: Performance evaluations of large language models for operations scheduling 

**Authors**: Yanyan Wang, Yingying Wang, Junli Liang, Yin Xu, Yunlong Liu, Yiming Xu, Zhengwang Jiang, Zhehe Li, Fei Li, Long Zhao, Kuang Xu, Qi Song, Xiangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.11017)  

**Abstract**: The rapid advancement of large language models (LLMs) has significantly propelled progress in artificial intelligence, demonstrating substantial application potential across multiple specialized domains. Telecommunications operation scheduling (OS) is a critical aspect of the telecommunications industry, involving the coordinated management of networks, services, risks, and human resources to optimize production scheduling and ensure unified service control. However, the inherent complexity and domain-specific nature of OS tasks, coupled with the absence of comprehensive evaluation benchmarks, have hindered thorough exploration of LLMs' application potential in this critical field. To address this research gap, we propose the first Telecommunications Operation Scheduling Evaluation Benchmark (TeleEval-OS). Specifically, this benchmark comprises 15 datasets across 13 subtasks, comprehensively simulating four key operational stages: intelligent ticket creation, intelligent ticket handling, intelligent ticket closure, and intelligent evaluation. To systematically assess the performance of LLMs on tasks of varying complexity, we categorize their capabilities in telecommunications operation scheduling into four hierarchical levels, arranged in ascending order of difficulty: basic NLP, knowledge Q&A, report generation, and report analysis. On TeleEval-OS, we leverage zero-shot and few-shot evaluation methods to comprehensively assess 10 open-source LLMs (e.g., DeepSeek-V3) and 4 closed-source LLMs (e.g., GPT-4o) across diverse scenarios. Experimental results demonstrate that open-source LLMs can outperform closed-source LLMs in specific scenarios, highlighting their significant potential and value in the field of telecommunications operation scheduling. 

---
# The Memory Paradox: Why Our Brains Need Knowledge in an Age of AI 

**Authors**: Barbara Oakley, Michael Johnston, Ken-Zen Chen, Eulho Jung, Terrence J. Sejnowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.11015)  

**Abstract**: In the age of generative AI and ubiquitous digital tools, human cognition faces a structural paradox: as external aids become more capable, internal memory systems risk atrophy. Drawing on neuroscience and cognitive psychology, this paper examines how heavy reliance on AI systems and discovery-based pedagogies may impair the consolidation of declarative and procedural memory -- systems essential for expertise, critical thinking, and long-term retention. We review how tools like ChatGPT and calculators can short-circuit the retrieval, error correction, and schema-building processes necessary for robust neural encoding. Notably, we highlight striking parallels between deep learning phenomena such as "grokking" and the neuroscience of overlearning and intuition. Empirical studies are discussed showing how premature reliance on AI during learning inhibits proceduralization and intuitive mastery. We argue that effective human-AI interaction depends on strong internal models -- biological "schemata" and neural manifolds -- that enable users to evaluate, refine, and guide AI output. The paper concludes with policy implications for education and workforce training in the age of large language models. 

---
# Data Science: a Natural Ecosystem 

**Authors**: Emilio Porcu, Roy El Moukari, Laurent Najman, Francisco Herrera, Horst Simon  

**Link**: [PDF](https://arxiv.org/pdf/2506.11010)  

**Abstract**: This manuscript provides a holistic (data-centric) view of what we term essential data science, as a natural ecosystem with challenges and missions stemming from the data universe with its multiple combinations of the 5D complexities (data structure, domain, cardinality, causality, and ethics) with the phases of the data life cycle. Data agents perform tasks driven by specific goals. The data scientist is an abstract entity that comes from the logical organization of data agents with their actions. Data scientists face challenges that are defined according to the missions. We define specific discipline-induced data science, which in turn allows for the definition of pan-data science, a natural ecosystem that integrates specific disciplines with the essential data science. We semantically split the essential data science into computational, and foundational. We claim that there is a serious threat of divergence between computational and foundational data science. Especially, if no approach is taken to rate whether a data universe discovery should be useful or not. We suggest that rigorous approaches to measure the usefulness of data universe discoveries might mitigate such a divergence. 

---
# Impact of Comments on LLM Comprehension of Legacy Code 

**Authors**: Rock Sabetto, Emily Escamilla, Devesh Agarwal, Sujay Kandwal, Justin F. Brunelle, Scott Rosen, Nitin Naik, Samruddhi Thaker, Eric O. Scott, Jacob Zimmer, Amit Madan, Arun Sridharan, Doug Wendt, Michael Doyle, Christopher Glasz, Jasper Phillips, William Macke, Colin Diggs, Michael Bartholf, Zachary Robin, Paul Ursino  

**Link**: [PDF](https://arxiv.org/pdf/2506.11007)  

**Abstract**: Large language models (LLMs) have been increasingly integrated into software engineering and maintenance tasks due to their high performance with software engineering tasks and robust understanding of modern programming languages. However, the ability of LLMs to comprehend code written with legacy languages remains a research gap challenged by real-world legacy systems lacking or containing inaccurate documentation that may impact LLM comprehension. To assess LLM comprehension of legacy languages, there is a need for objective LLM evaluation. In order to objectively measure LLM comprehension of legacy languages, we need an efficient, quantitative evaluation method. We leverage multiple-choice question answering (MCQA), an emerging LLM evaluation methodology, to evaluate LLM comprehension of legacy code and the impact of comment prevalence and inaccurate comments. In this work, we present preliminary findings on the impact of documentation on LLM comprehension of legacy code and outline strategic objectives for future work. 

---
# Developing a Dyslexia Indicator Using Eye Tracking 

**Authors**: Kevin Cogan, Vuong M. Ngo, Mark Roantree  

**Link**: [PDF](https://arxiv.org/pdf/2506.11004)  

**Abstract**: Dyslexia, affecting an estimated 10% to 20% of the global population, significantly impairs learning capabilities, highlighting the need for innovative and accessible diagnostic methods. This paper investigates the effectiveness of eye-tracking technology combined with machine learning algorithms as a cost-effective alternative for early dyslexia detection. By analyzing general eye movement patterns, including prolonged fixation durations and erratic saccades, we proposed an enhanced solution for determining eye-tracking-based dyslexia features. A Random Forest Classifier was then employed to detect dyslexia, achieving an accuracy of 88.58\%. Additionally, hierarchical clustering methods were applied to identify varying severity levels of dyslexia. The analysis incorporates diverse methodologies across various populations and settings, demonstrating the potential of this technology to identify individuals with dyslexia, including those with borderline traits, through non-invasive means. Integrating eye-tracking with machine learning represents a significant advancement in the diagnostic process, offering a highly accurate and accessible method in clinical research. 

---
# EmbedAgent: Benchmarking Large Language Models in Embedded System Development 

**Authors**: Ruiyang Xu, Jialun Cao, Mingyuan Wu, Wenliang Zhong, Yaojie Lu, Ben He, Xianpei Han, Shing-Chi Cheung, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.11003)  

**Abstract**: Large Language Models (LLMs) have shown promise in various tasks, yet few benchmarks assess their capabilities in embedded system this http URL this paper, we introduce EmbedAgent, a paradigm designed to simulate real-world roles in embedded system development, such as Embedded System Programmer, Architect, and Integrator. This paradigm enables LLMs to be tested in tasks that bridge the gap between digital and physical systems, allowing for a more comprehensive assessment of their capabilities. To evaluate LLMs on these tasks, we propose Embedbench, the first comprehensive benchmark for embedded system programming, circuit design, and cross-platform this http URL consists of 126 cases, covering 9 electronic components across 3 hardware platforms. Through extensive experiments on 10 mainstream LLMs, we uncover several key findings. Surprisingly, despite the simplicity of the cases, DeepSeek-R1 achieves only a 55.6% pass@1 rate when provided with schematic information, and 50.0% when tasked with generating the schematics itself. In the cross-platform migration tasks, LLMs show relatively strong performance with MicroPython on the Raspberry Pi Pico (with the top model achieving 73.8% pass@1), but perform poorly on ESP-IDF, where the best model reaches only 29.4% pass@1.Interestingly, we observe that general-purpose chat LLMs like DeepSeek-V3 often fail to utilize relevant pre-trained knowledge in this domain, while reasoning LLMs tend to overthink and overlook efficient knowledge during pretraining. Based on these insights, we propose two strategies: retrieval augmented generation and compiler feedback-to enhance LLM performance. These strategies result in significant improvements, with Deepseek-R1 reaching a 65.1% pass@1 with correct schematics, and 53.1% without. Additionally, the accuracy of the Arduino to ESP32 migration task improves from 21.4% to 27.8%. 

---
# Rethinking Technological Readiness in the Era of AI Uncertainty 

**Authors**: S. Tucker Browne, Mark M. Bailey  

**Link**: [PDF](https://arxiv.org/pdf/2506.11001)  

**Abstract**: Artificial intelligence (AI) is poised to revolutionize military combat systems, but ensuring these AI-enabled capabilities are truly mission-ready presents new challenges. We argue that current technology readiness assessments fail to capture critical AI-specific factors, leading to potential risks in deployment. We propose a new AI Readiness Framework to evaluate the maturity and trustworthiness of AI components in military systems. The central thesis is that a tailored framework - analogous to traditional Technology Readiness Levels (TRL) but expanded for AI - can better gauge an AI system's reliability, safety, and suitability for combat use. Using current data evaluation tools and testing practices, we demonstrate the framework's feasibility for near-term implementation. This structured approach provides military decision-makers with clearer insight into whether an AI-enabled system has met the necessary standards of performance, transparency, and human integration to be deployed with confidence, thus advancing the field of defense technology management and risk assessment. 

---
# Automated Validation of COBOL to Java Transformation 

**Authors**: Atul Kumar, Diptikalyan Saha, Toshikai Yasue, Kohichi Ono, Saravanan Krishnan, Sandeep Hans, Fumiko Satoh, Gerald Mitchell, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.10999)  

**Abstract**: Recent advances in Large Language Model (LLM) based Generative AI techniques have made it feasible to translate enterpriselevel code from legacy languages such as COBOL to modern languages such as Java or Python. While the results of LLM-based automatic transformation are encouraging, the resulting code cannot be trusted to correctly translate the original code. We propose a framework and a tool to help validate the equivalence of COBOL and translated Java. The results can also help repair the code if there are some issues and provide feedback to the AI model to improve. We have developed a symbolic-execution-based test generation to automatically generate unit tests for the source COBOL programs which also mocks the external resource calls. We generate equivalent JUnit test cases with equivalent mocking as COBOL and run them to check semantic equivalence between original and translated programs. 

---
# Towards Automated Formal Verification of Backend Systems with LLMs 

**Authors**: Kangping Xu, Yifan Luo, Yang Yuan, Andrew Chi-Chih Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.10998)  

**Abstract**: Software testing plays a critical role in ensuring that systems behave as intended. However, existing automated testing approaches struggle to match the capabilities of human engineers due to key limitations such as test locality, lack of general reliability, and business logic blindness. In this work, we propose a novel framework that leverages functional programming and type systems to translate Scala backend code into formal Lean representations. Our pipeline automatically generates theorems that specify the intended behavior of APIs and database operations, and uses LLM-based provers to verify them. When a theorem is proved, the corresponding logic is guaranteed to be correct and no further testing is needed. If the negation of a theorem is proved instead, it confirms a bug. In cases where neither can be proved, human intervention is required. We evaluate our method on realistic backend systems and find that it can formally verify over 50% of the test requirements, which suggests that half of a testing engineer's workload can be automated. Additionally, with an average cost of only $2.19 per API, LLM-based verification is significantly more cost-effective than manual testing and can be scaled easily through parallel execution. Our results indicate a promising direction for scalable, AI-powered software testing, with the potential to greatly improve engineering productivity as models continue to advance. 

---
# Evaluating LLMs for Visualization Tasks 

**Authors**: Saadiq Rauf Khan, Vinit Chandak, Sougata Mukherjea  

**Link**: [PDF](https://arxiv.org/pdf/2506.10996)  

**Abstract**: Information Visualization has been utilized to gain insights from complex data. In recent times, Large Language Models (LLMs) have performed very well in many tasks. In this paper, we showcase the capabilities of different popular LLMs to generate code for visualization based on simple prompts. We also analyze the power of LLMs to understand some common visualizations by answering simple questions. Our study shows that LLMs could generate code for some visualizations as well as answer questions about them. However, LLMs also have several limitations. We believe that our insights can be used to improve both LLMs and Information Visualization systems. 

---
# On the Effectiveness of the 'Follow-the-Sun' Strategy in Mitigating the Carbon Footprint of AI in Cloud Instances 

**Authors**: Roberto Vergallo, Luís Cruz, Alessio Errico, Luca Mainetti  

**Link**: [PDF](https://arxiv.org/pdf/2506.10990)  

**Abstract**: 'Follow-the-Sun' (FtS) is a theoretical computational model aimed at minimizing the carbon footprint of computer workloads. It involves dynamically moving workloads to regions with cleaner energy sources as demand increases and energy production relies more on fossil fuels. With the significant power consumption of Artificial Intelligence (AI) being a subject of extensive debate, FtS is proposed as a strategy to mitigate the carbon footprint of training AI models. However, the literature lacks scientific evidence on the advantages of FtS to mitigate the carbon footprint of AI workloads. In this paper, we present the results of an experiment conducted in a partial synthetic scenario to address this research gap. We benchmarked four AI algorithms in the anomaly detection domain and measured the differences in carbon emissions in four cases: no strategy, FtS, and two strategies previously introduced in the state of the art, namely Flexible Start and Pause and Resume. To conduct our experiment, we utilized historical carbon intensity data from the year 2021 for seven European cities. Our results demonstrate that the FtS strategy not only achieves average reductions of up to 14.6% in carbon emissions (with peaks of 16.3%) but also helps in preserving the time needed for training. 

---
# Prompt engineering and framework: implementation to increase code reliability based guideline for LLMs 

**Authors**: Rogelio Cruz, Jonatan Contreras, Francisco Guerrero, Ezequiel Rodriguez, Carlos Valdez, Citlali Carrillo  

**Link**: [PDF](https://arxiv.org/pdf/2506.10989)  

**Abstract**: In this paper, we propose a novel prompting approach aimed at enhancing the ability of Large Language Models (LLMs) to generate accurate Python code. Specifically, we introduce a prompt template designed to improve the quality and correctness of generated code snippets, enabling them to pass tests and produce reliable results. Through experiments conducted on two state-of-the-art LLMs using the HumanEval dataset, we demonstrate that our approach outperforms widely studied zero-shot and Chain-of-Thought (CoT) methods in terms of the Pass@k metric. Furthermore, our method achieves these improvements with significantly reduced token usage compared to the CoT approach, making it both effective and resource-efficient, thereby lowering the computational demands and improving the eco-footprint of LLM capabilities. These findings highlight the potential of tailored prompting strategies to optimize code generation performance, paving the way for broader applications in AI-driven programming tasks. 

---
# Application Modernization with LLMs: Addressing Core Challenges in Reliability, Security, and Quality 

**Authors**: Ahilan Ayyachamy Nadar Ponnusamy  

**Link**: [PDF](https://arxiv.org/pdf/2506.10984)  

**Abstract**: AI-assisted code generation tools have revolutionized software development, offering unprecedented efficiency and scalability. However, multiple studies have consistently highlighted challenges such as security vulnerabilities, reliability issues, and inconsistencies in the generated code. Addressing these concerns is crucial to unlocking the full potential of this transformative technology. While advancements in foundational and code-specialized language models have made notable progress in mitigating some of these issues, significant gaps remain, particularly in ensuring high-quality, trustworthy outputs.
This paper builds upon existing research on leveraging large language models (LLMs) for application modernization. It explores an opinionated approach that emphasizes two core capabilities of LLMs: code reasoning and code generation. The proposed framework integrates these capabilities with human expertise to tackle application modernization challenges effectively. It highlights the indispensable role of human involvement and guidance in ensuring the success of AI-assisted processes.
To demonstrate the framework's utility, this paper presents a detailed case study, walking through its application in a real-world scenario. The analysis includes a step-by-step breakdown, assessing alternative approaches where applicable. This work aims to provide actionable insights and a robust foundation for future research in AI-driven application modernization. The reference implementation created for this paper is available on GitHub. 

---
