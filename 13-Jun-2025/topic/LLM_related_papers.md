# Conversational Search: From Fundamentals to Frontiers in the LLM Era 

**Authors**: Fengran Mo, Chuan Meng, Mohammad Aliannejadi, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.10635)  

**Abstract**: Conversational search enables multi-turn interactions between users and systems to fulfill users' complex information needs. During this interaction, the system should understand the users' search intent within the conversational context and then return the relevant information through a flexible, dialogue-based interface. The recent powerful large language models (LLMs) with capacities of instruction following, content generation, and reasoning, attract significant attention and advancements, providing new opportunities and challenges for building up intelligent conversational search systems. This tutorial aims to introduce the connection between fundamentals and the emerging topics revolutionized by LLMs in the context of conversational search. It is designed for students, researchers, and practitioners from both academia and industry. Participants will gain a comprehensive understanding of both the core principles and cutting-edge developments driven by LLMs in conversational search, equipping them with the knowledge needed to contribute to the development of next-generation conversational search systems. 

---
# Towards Understanding Bias in Synthetic Data for Evaluation 

**Authors**: Hossein A. Rahmani, Varsha Ramineni, Nick Craswell, Bhaskar Mitra, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.10301)  

**Abstract**: Test collections are crucial for evaluating Information Retrieval (IR) systems. Creating a diverse set of user queries for these collections can be challenging, and obtaining relevance judgments, which indicate how well retrieved documents match a query, is often costly and resource-intensive. Recently, generating synthetic datasets using Large Language Models (LLMs) has gained attention in various applications. While previous work has used LLMs to generate synthetic queries or documents to improve ranking models, using LLMs to create synthetic test collections is still relatively unexplored. Previous work~\cite{rahmani2024synthetic} showed that synthetic test collections have the potential to be used for system evaluation, however, more analysis is needed to validate this claim. In this paper, we thoroughly investigate the reliability of synthetic test collections constructed using LLMs, where LLMs are used to generate synthetic queries, labels, or both. In particular, we examine the potential biases that might occur when such test collections are used for evaluation. We first empirically show the presence of such bias in evaluation results and analyse the effects it might have on system evaluation. We further validate the presence of such bias using a linear mixed-effects model. Our analysis shows that while the effect of bias present in evaluation results obtained using synthetic test collections could be significant, for e.g.~computing absolute system performance, its effect may not be as significant in comparing relative system performance. Codes and data are available at: this https URL. 

---
# Precise Zero-Shot Pointwise Ranking with LLMs through Post-Aggregated Global Context Information 

**Authors**: Kehan Long, Shasha Li, Chen Xu, Jintao Tang, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10859)  

**Abstract**: Recent advancements have successfully harnessed the power of Large Language Models (LLMs) for zero-shot document ranking, exploring a variety of prompting strategies. Comparative approaches like pairwise and listwise achieve high effectiveness but are computationally intensive and thus less practical for larger-scale applications. Scoring-based pointwise approaches exhibit superior efficiency by independently and simultaneously generating the relevance scores for each candidate document. However, this independence ignores critical comparative insights between documents, resulting in inconsistent scoring and suboptimal performance. In this paper, we aim to improve the effectiveness of pointwise methods while preserving their efficiency through two key innovations: (1) We propose a novel Global-Consistent Comparative Pointwise Ranking (GCCP) strategy that incorporates global reference comparisons between each candidate and an anchor document to generate contrastive relevance scores. We strategically design the anchor document as a query-focused summary of pseudo-relevant candidates, which serves as an effective reference point by capturing the global context for document comparison. (2) These contrastive relevance scores can be efficiently Post-Aggregated with existing pointwise methods, seamlessly integrating essential Global Context information in a training-free manner (PAGC). Extensive experiments on the TREC DL and BEIR benchmark demonstrate that our approach significantly outperforms previous pointwise methods while maintaining comparable efficiency. Our method also achieves competitive performance against comparative methods that require substantially more computational resources. More analyses further validate the efficacy of our anchor construction strategy. 

---
# TaxoAdapt: Aligning LLM-Based Multidimensional Taxonomy Construction to Evolving Research Corpora 

**Authors**: Priyanka Kargupta, Nan Zhang, Yunyi Zhang, Rui Zhang, Prasenjit Mitra, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.10737)  

**Abstract**: The rapid evolution of scientific fields introduces challenges in organizing and retrieving scientific literature. While expert-curated taxonomies have traditionally addressed this need, the process is time-consuming and expensive. Furthermore, recent automatic taxonomy construction methods either (1) over-rely on a specific corpus, sacrificing generalizability, or (2) depend heavily on the general knowledge of large language models (LLMs) contained within their pre-training datasets, often overlooking the dynamic nature of evolving scientific domains. Additionally, these approaches fail to account for the multi-faceted nature of scientific literature, where a single research paper may contribute to multiple dimensions (e.g., methodology, new tasks, evaluation metrics, benchmarks). To address these gaps, we propose TaxoAdapt, a framework that dynamically adapts an LLM-generated taxonomy to a given corpus across multiple dimensions. TaxoAdapt performs iterative hierarchical classification, expanding both the taxonomy width and depth based on corpus' topical distribution. We demonstrate its state-of-the-art performance across a diverse set of computer science conferences over the years to showcase its ability to structure and capture the evolution of scientific fields. As a multidimensional method, TaxoAdapt generates taxonomies that are 26.51% more granularity-preserving and 50.41% more coherent than the most competitive baselines judged by LLMs. 

---
# ChineseHarm-Bench: A Chinese Harmful Content Detection Benchmark 

**Authors**: Kangwei Liu, Siyuan Cheng, Bozhong Tian, Xiaozhuan Liang, Yuyang Yin, Meng Han, Ningyu Zhang, Bryan Hooi, Xi Chen, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.10960)  

**Abstract**: Large language models (LLMs) have been increasingly applied to automated harmful content detection tasks, assisting moderators in identifying policy violations and improving the overall efficiency and accuracy of content review. However, existing resources for harmful content detection are predominantly focused on English, with Chinese datasets remaining scarce and often limited in scope. We present a comprehensive, professionally annotated benchmark for Chinese content harm detection, which covers six representative categories and is constructed entirely from real-world data. Our annotation process further yields a knowledge rule base that provides explicit expert knowledge to assist LLMs in Chinese harmful content detection. In addition, we propose a knowledge-augmented baseline that integrates both human-annotated knowledge rules and implicit knowledge from large language models, enabling smaller models to achieve performance comparable to state-of-the-art LLMs. Code and data are available at this https URL. 

---
# A Study on Individual Spatiotemporal Activity Generation Method Using MCP-Enhanced Chain-of-Thought Large Language Models 

**Authors**: Yu Zhang, Yang Hu, De Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10853)  

**Abstract**: Human spatiotemporal behavior simulation is critical for urban planning research, yet traditional rule-based and statistical approaches suffer from high computational costs, limited generalizability, and poor scalability. While large language models (LLMs) show promise as "world simulators," they face challenges in spatiotemporal reasoning including limited spatial cognition, lack of physical constraint understanding, and group homogenization tendencies. This paper introduces a framework integrating chain-of-thought (CoT) reasoning with Model Context Protocol (MCP) to enhance LLMs' capability in simulating spatiotemporal behaviors that correspond with validation data patterns. The methodology combines human-like progressive reasoning through a five-stage cognitive framework with comprehensive data processing via six specialized MCP tool categories: temporal management, spatial navigation, environmental perception, personal memory, social collaboration, and experience evaluation. Experiments in Shanghai's Lujiazui district validate the framework's effectiveness across 1,000 generated samples. Results demonstrate high similarity with real mobile signaling data, achieving generation quality scores of 7.86 to 8.36 across different base models. Parallel processing experiments show efficiency improvements, with generation times decreasing from 1.30 to 0.17 minutes per sample when scaling from 2 to 12 processes. This work contributes to integrating CoT reasoning with MCP for urban behavior modeling, advancing LLMs applications in urban computing and providing a practical approach for synthetic mobility data generation. The framework offers a foundation for smart city planning, transportation forecasting, and participatory urban design applications. 

---
# GenPlanX. Generation of Plans and Execution 

**Authors**: Daniel Borrajo, Giuseppe Canonaco, Tomás de la Rosa, Alfredo Garrachón, Sriram Gopalakrishnan, Simerjot Kaur, Marianela Morales, Sunandita Patra, Alberto Pozanco, Keshav Ramani, Charese Smiley, Pietro Totis, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2506.10897)  

**Abstract**: Classical AI Planning techniques generate sequences of actions for complex tasks. However, they lack the ability to understand planning tasks when provided using natural language. The advent of Large Language Models (LLMs) has introduced novel capabilities in human-computer interaction. In the context of planning tasks, LLMs have shown to be particularly good in interpreting human intents among other uses. This paper introduces GenPlanX that integrates LLMs for natural language-based description of planning tasks, with a classical AI planning engine, alongside an execution and monitoring framework. We demonstrate the efficacy of GenPlanX in assisting users with office-related tasks, highlighting its potential to streamline workflows and enhance productivity through seamless human-AI collaboration. 

---
# Breaking Bad Molecules: Are MLLMs Ready for Structure-Level Molecular Detoxification? 

**Authors**: Fei Lin, Ziyang Gong, Cong Wang, Yonglin Tian, Tengchao Zhang, Xue Yang, Gen Luo, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10912)  

**Abstract**: Toxicity remains a leading cause of early-stage drug development failure. Despite advances in molecular design and property prediction, the task of molecular toxicity repair - generating structurally valid molecular alternatives with reduced toxicity - has not yet been systematically defined or benchmarked. To fill this gap, we introduce ToxiMol, the first benchmark task for general-purpose Multimodal Large Language Models (MLLMs) focused on molecular toxicity repair. We construct a standardized dataset covering 11 primary tasks and 560 representative toxic molecules spanning diverse mechanisms and granularities. We design a prompt annotation pipeline with mechanism-aware and task-adaptive capabilities, informed by expert toxicological knowledge. In parallel, we propose an automated evaluation framework, ToxiEval, which integrates toxicity endpoint prediction, synthetic accessibility, drug-likeness, and structural similarity into a high-throughput evaluation chain for repair success. We systematically assess nearly 30 mainstream general-purpose MLLMs and design multiple ablation studies to analyze key factors such as evaluation criteria, candidate diversity, and failure attribution. Experimental results show that although current MLLMs still face significant challenges on this task, they begin to demonstrate promising capabilities in toxicity understanding, semantic constraint adherence, and structure-aware molecule editing. 

---
# Primender Sequence: A Novel Mathematical Construct for Testing Symbolic Inference and AI Reasoning 

**Authors**: Mohd Anwar Jamal Faiz  

**Link**: [PDF](https://arxiv.org/pdf/2506.10585)  

**Abstract**: This paper introduces the Primender sequence, a novel integer sequence defined by a hybrid rule that combines classical primality with modular digit-based conditions. Specifically, a number n is included in the sequence if it is prime or ends with a prime number of unit digit or any length. In other words, numbers which are primes or have at least one prime suffix. The resulting sequence exhibits a deterministic yet non-trivial structure, blending number-theoretic properties with symbolic patterning. We propose the Primender sequence as a benchmark for evaluating the symbolic reasoning capabilities of Large Language Models (LLMs). The study is motivated by the need for interpretable, rule-based testbeds that can assess an LLM's ability to infer hidden rules, validate mathematical hypotheses, and generalize symbolic logic at scale. A key hypothesis explored is: Whenever a number in the Primender sequence is exactly one more than the largest prime less than or equal to it, the difference between it and the previous number in the sequence is also 1. We design a structured prompt and evaluation framework to test this hypothesis across multiple state-of-the-art LLMs, including ChatGPT, Copilot, DeepSeek, Gemini, Grok, and LLaMA. The models are tasked with identifying the underlying rule, validating the hypothesis, and generating the next 100,000 terms of the sequence. Comparative metrics such as rule inference accuracy, hypothesis evaluation, sequence validity, and symbolic explanation quality are used to assess model performance. This work contributes a novel mathematical construct and a reproducible methodology for benchmarking LLMs in symbolic reasoning, hypothesis testing, and scalable pattern generalization - bridging the domains of number theory, artificial intelligence, and software engineering. 

---
# Scientists' First Exam: Probing Cognitive Abilities of MLLM via Perception, Understanding, and Reasoning 

**Authors**: Yuhao Zhou, Yiheng Wang, Xuming He, Ruoyao Xiao, Zhiwei Li, Qiantai Feng, Zijie Guo, Yuejin Yang, Hao Wu, Wenxuan Huang, Jiaqi Wei, Dan Si, Xiuqi Yao, Jia Bu, Haiwen Huang, Tianfan Fu, Shixiang Tang, Ben Fei, Dongzhan Zhou, Fenghua Ling, Yan Lu, Siqi Sun, Chenhui Li, Guanjie Zheng, Jiancheng Lv, Wenlong Zhang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2506.10521)  

**Abstract**: Scientific discoveries increasingly rely on complex multimodal reasoning based on information-intensive scientific data and domain-specific expertise. Empowered by expert-level scientific benchmarks, scientific Multimodal Large Language Models (MLLMs) hold the potential to significantly enhance this discovery process in realistic workflows. However, current scientific benchmarks mostly focus on evaluating the knowledge understanding capabilities of MLLMs, leading to an inadequate assessment of their perception and reasoning abilities. To address this gap, we present the Scientists' First Exam (SFE) benchmark, designed to evaluate the scientific cognitive capacities of MLLMs through three interconnected levels: scientific signal perception, scientific attribute understanding, scientific comparative reasoning. Specifically, SFE comprises 830 expert-verified VQA pairs across three question types, spanning 66 multimodal tasks across five high-value disciplines. Extensive experiments reveal that current state-of-the-art GPT-o3 and InternVL-3 achieve only 34.08% and 26.52% on SFE, highlighting significant room for MLLMs to improve in scientific realms. We hope the insights obtained in SFE will facilitate further developments in AI-enhanced scientific discoveries. 

---
# LogiPlan: A Structured Benchmark for Logical Planning and Relational Reasoning in LLMs 

**Authors**: Yanan Cai, Ahmed Salem, Besmira Nushi, Mark Russinovich  

**Link**: [PDF](https://arxiv.org/pdf/2506.10527)  

**Abstract**: We introduce LogiPlan, a novel benchmark designed to evaluate the capabilities of large language models (LLMs) in logical planning and reasoning over complex relational structures. Logical relational reasoning is important for applications that may rely on LLMs to generate and query structured graphs of relations such as network infrastructure, knowledge bases, or business process schema. Our framework allows for dynamic variation of task complexity by controlling the number of objects, relations, and the minimum depth of relational chains, providing a fine-grained assessment of model performance across difficulty levels. LogiPlan encompasses three complementary tasks: (1) Plan Generation, where models must construct valid directed relational graphs meeting specified structural constraints; (2) Consistency Detection, testing models' ability to identify inconsistencies in relational structures; and (3) Comparison Question, evaluating models' capacity to determine the validity of queried relationships within a given graph. Additionally, we assess models' self-correction capabilities by prompting them to verify and refine their initial solutions. We evaluate state-of-the-art models including DeepSeek R1, Gemini 2.0 Pro, Gemini 2 Flash Thinking, GPT-4.5, GPT-4o, Llama 3.1 405B, O3-mini, O1, and Claude 3.7 Sonnet across these tasks, revealing significant performance gaps that correlate with model scale and architecture. Our analysis demonstrates that while recent reasoning-enhanced models show promising results on simpler instances, they struggle with more complex configurations requiring deeper logical planning. 

---
# OPT-BENCH: Evaluating LLM Agent on Large-Scale Search Spaces Optimization Problems 

**Authors**: Xiaozhe Li, Jixuan Chen, Xinyu Fang, Shengyuan Ding, Haodong Duan, Qingwen Liu, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10764)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities in solving diverse tasks. However, their proficiency in iteratively optimizing complex solutions through learning from previous feedback remains insufficiently explored. To bridge this gap, we present OPT-BENCH, a comprehensive benchmark designed to evaluate LLM agents on large-scale search space optimization problems. OPT-BENCH includes 20 real-world machine learning tasks sourced from Kaggle and 10 classical NP problems, offering a diverse and challenging environment for assessing LLM agents on iterative reasoning and solution refinement. To enable rigorous evaluation, we introduce OPT-Agent, an end-to-end optimization framework that emulates human reasoning when tackling complex problems by generating, validating, and iteratively improving solutions through leveraging historical feedback. Through extensive experiments on 9 state-of-the-art LLMs from 6 model families, we analyze the effects of optimization iterations, temperature settings, and model architectures on solution quality and convergence. Our results demonstrate that incorporating historical context significantly enhances optimization performance across both ML and NP tasks. All datasets, code, and evaluation tools are open-sourced to promote further research in advancing LLM-driven optimization and iterative reasoning. Project page: \href{this https URL}{this https URL}. 

---
# WGSR-Bench: Wargame-based Game-theoretic Strategic Reasoning Benchmark for Large Language Models 

**Authors**: Qiyue Yin, Pei Xu, Qiaozhe Li, Shengda Liu, Shengqi Shen, Tong Wang, Yihong Han, Xiaonan Zhao, Likun Yang, Shiyue Cao, Shiyu Qiu, Yuxuan Liu, Shizhao Yu, Lei Cui, Chengxin Yan, Jie Sun, Xiangquan Tang, Kaiqi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10264)  

**Abstract**: Recent breakthroughs in Large Language Models (LLMs) have led to a qualitative leap in artificial intelligence' s performance on reasoning tasks, particularly demonstrating remarkable capabilities in mathematical, symbolic, and commonsense reasoning. However, as a critical component of advanced human cognition, strategic reasoning, i.e., the ability to assess multi-agent behaviors in dynamic environments, formulate action plans, and adapt strategies, has yet to be systematically evaluated or modeled. To address this gap, this paper introduces WGSR-Bench, the first strategy reasoning benchmark for LLMs using wargame as its evaluation environment. Wargame, a quintessential high-complexity strategic scenario, integrates environmental uncertainty, adversarial dynamics, and non-unique strategic choices, making it an effective testbed for assessing LLMs' capabilities in multi-agent decision-making, intent inference, and counterfactual reasoning. WGSR-Bench designs test samples around three core tasks, i.e., Environmental situation awareness, Opponent risk modeling and Policy generation, which serve as the core S-POE architecture, to systematically assess main abilities of strategic reasoning. Finally, an LLM-based wargame agent is designed to integrate these parts for a comprehensive strategy reasoning assessment. With WGSR-Bench, we hope to assess the strengths and limitations of state-of-the-art LLMs in game-theoretic strategic reasoning and to advance research in large model-driven strategic intelligence. 

---
# TeleMath: A Benchmark for Large Language Models in Telecom Mathematical Problem Solving 

**Authors**: Vincenzo Colle, Mohamed Sana, Nicola Piovesan, Antonio De Domenico, Fadhel Ayed, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2506.10674)  

**Abstract**: The increasing adoption of artificial intelligence in telecommunications has raised interest in the capability of Large Language Models (LLMs) to address domain-specific, mathematically intensive tasks. Although recent advancements have improved the performance of LLMs in general mathematical reasoning, their effectiveness within specialized domains, such as signal processing, network optimization, and performance analysis, remains largely unexplored. To address this gap, we introduce TeleMath, the first benchmark dataset specifically designed to evaluate LLM performance in solving mathematical problems with numerical solutions in the telecommunications domain. Comprising 500 question-answer (QnA) pairs, TeleMath covers a wide spectrum of topics in the telecommunications field. This paper outlines the proposed QnAs generation pipeline, starting from a selected seed of problems crafted by Subject Matter Experts. The evaluation of a wide range of open-source LLMs reveals that best performance on TeleMath is achieved by recent models explicitly designed for mathematical or logical reasoning. In contrast, general-purpose models, even those with a large number of parameters, often struggle with these challenges. We have released the dataset and the evaluation code to ease result reproducibility and support future research. 

---
# AutoMind: Adaptive Knowledgeable Agent for Automated Data Science 

**Authors**: Yixin Ou, Yujie Luo, Jingsheng Zheng, Lanning Wei, Shuofei Qiao, Jintian Zhang, Da Zheng, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10974)  

**Abstract**: Large Language Model (LLM) agents have shown great potential in addressing real-world data science problems. LLM-driven data science agents promise to automate the entire machine learning pipeline, yet their real-world effectiveness remains limited. Existing frameworks depend on rigid, pre-defined workflows and inflexible coding strategies; consequently, they excel only on relatively simple, classical problems and fail to capture the empirical expertise that human practitioners bring to complex, innovative tasks. In this work, we introduce AutoMind, an adaptive, knowledgeable LLM-agent framework that overcomes these deficiencies through three key advances: (1) a curated expert knowledge base that grounds the agent in domain expert knowledge, (2) an agentic knowledgeable tree search algorithm that strategically explores possible solutions, and (3) a self-adaptive coding strategy that dynamically tailors code generation to task complexity. Evaluations on two automated data science benchmarks demonstrate that AutoMind delivers superior performance versus state-of-the-art baselines. Additional analyses confirm favorable effectiveness, efficiency, and qualitative solution quality, highlighting AutoMind as an efficient and robust step toward fully automated data science. 

---
# Farseer: A Refined Scaling Law in Large Language Models 

**Authors**: Houyi Li, Wenzhen Zheng, Qiufeng Wang, Zhenyu Ding, Haoying Wang, Zili Wang, Shijie Xuyang, Ning Ding, Shuigeng Zhou, Xiangyu Zhang, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10972)  

**Abstract**: Training Large Language Models (LLMs) is prohibitively expensive, creating a critical scaling gap where insights from small-scale experiments often fail to transfer to resource-intensive production systems, thereby hindering efficient innovation. To bridge this, we introduce Farseer, a novel and refined scaling law offering enhanced predictive accuracy across scales. By systematically constructing a model loss surface $L(N,D)$, Farseer achieves a significantly better fit to empirical data than prior laws (e.g., Chinchilla's law). Our methodology yields accurate, robust, and highly generalizable predictions, demonstrating excellent extrapolation capabilities, improving upon Chinchilla's law by reducing extrapolation error by 433\%. This allows for the reliable evaluation of competing training strategies across all $(N,D)$ settings, enabling conclusions from small-scale ablation studies to be confidently extrapolated to predict large-scale performance. Furthermore, Farseer provides new insights into optimal compute allocation, better reflecting the nuanced demands of modern LLM training. To validate our approach, we trained an extensive suite of approximately 1,000 LLMs across diverse scales and configurations, consuming roughly 3 million NVIDIA H100 GPU hours. We are comprehensively open-sourcing all models, data, results, and logs at this https URL to foster further research. 

---
# Automated Validation of Textual Constraints Against AutomationML via LLMs and SHACL 

**Authors**: Tom Westermann, Aljosha Köcher, Felix Gehlhoff  

**Link**: [PDF](https://arxiv.org/pdf/2506.10678)  

**Abstract**: AutomationML (AML) enables standardized data exchange in engineering, yet existing recommendations for proper AML modeling are typically formulated as informal and textual constraints. These constraints cannot be validated automatically within AML itself. This work-in-progress paper introduces a pipeline to formalize and verify such constraints. First, AML models are mapped to OWL ontologies via RML and SPARQL. In addition, a Large Language Model translates textual rules into SHACL constraints, which are then validated against the previously generated AML ontology. Finally, SHACL validation results are automatically interpreted in natural language. The approach is demonstrated on a sample AML recommendation. Results show that even complex modeling rules can be semi-automatically checked -- without requiring users to understand formal methods or ontology technologies. 

---
# GUARD: Guided Unlearning and Retention via Data Attribution for Large Language Models 

**Authors**: Evelyn Ma, Duo Zhou, Peizhi Niu, Huiting Zhou, Huan Zhang, Olgica Milenkovic, S. Rasoul Etesami  

**Link**: [PDF](https://arxiv.org/pdf/2506.10946)  

**Abstract**: Unlearning in large language models (LLMs) is becoming increasingly important due to regulatory compliance, copyright protection, and privacy concerns. However, a key challenge in LLM unlearning is unintended forgetting, where the removal of specific data inadvertently impairs the utility of the model and its retention of valuable, desired information. While prior work has primarily focused on architectural innovations, the influence of data-level factors on unlearning performance remains underexplored. As a result, existing methods often suffer from degraded retention when forgetting high-impact data. To address this, we propose GUARD-a novel framework for Guided Unlearning And Retention via Data attribution. At its core, GUARD introduces a lightweight proxy data attribution metric tailored for LLM unlearning, which quantifies the "alignment" between the forget and retain sets while remaining computationally efficient. Building on this, we design a novel unlearning objective that assigns adaptive, nonuniform unlearning weights to samples, inversely proportional to their proxy attribution scores. Through such a reallocation of unlearning power, GUARD mitigates unintended losses in retention. We provide rigorous theoretical guarantees that GUARD significantly enhances retention while maintaining forgetting metrics comparable to prior methods. Extensive experiments on the TOFU benchmark across multiple LLM architectures demonstrate that GUARD substantially improves utility preservation while ensuring effective unlearning. Notably, GUARD reduces utility sacrifice on the Retain Set by up to 194.92% in terms of Truth Ratio when forgetting 10% of the training data. 

---
# Robustly Improving LLM Fairness in Realistic Settings via Interpretability 

**Authors**: Adam Karvonen, Samuel Marks  

**Link**: [PDF](https://arxiv.org/pdf/2506.10922)  

**Abstract**: Large language models (LLMs) are increasingly deployed in high-stakes hiring applications, making decisions that directly impact people's careers and livelihoods. While prior studies suggest simple anti-bias prompts can eliminate demographic biases in controlled evaluations, we find these mitigations fail when realistic contextual details are introduced. We address these failures through internal bias mitigation: by identifying and neutralizing sensitive attribute directions within model activations, we achieve robust bias reduction across all tested scenarios. Across leading commercial (GPT-4o, Claude 4 Sonnet, Gemini 2.5 Flash) and open-source models (Gemma-2 27B, Gemma-3, Mistral-24B), we find that adding realistic context such as company names, culture descriptions from public careers pages, and selective hiring constraints (e.g.,``only accept candidates in the top 10\%") induces significant racial and gender biases (up to 12\% differences in interview rates). When these biases emerge, they consistently favor Black over White candidates and female over male candidates across all tested models and scenarios. Moreover, models can infer demographics and become biased from subtle cues like college affiliations, with these biases remaining invisible even when inspecting the model's chain-of-thought reasoning. To address these limitations, our internal bias mitigation identifies race and gender-correlated directions and applies affine concept editing at inference time. Despite using directions from a simple synthetic dataset, the intervention generalizes robustly, consistently reducing bias to very low levels (typically under 1\%, always below 2.5\%) while largely maintaining model performance. Our findings suggest that practitioners deploying LLMs for hiring should adopt more realistic evaluation methodologies and consider internal mitigation strategies for equitable outcomes. 

---
# LLM-Driven Personalized Answer Generation and Evaluation 

**Authors**: Mohammadreza Molavi, Mohammadreza Tavakoli, Mohammad Moein, Abdolali Faraji, Gábor Kismihók  

**Link**: [PDF](https://arxiv.org/pdf/2506.10829)  

**Abstract**: Online learning has experienced rapid growth due to its flexibility and accessibility. Personalization, adapted to the needs of individual learners, is crucial for enhancing the learning experience, particularly in online settings. A key aspect of personalization is providing learners with answers customized to their specific questions. This paper therefore explores the potential of Large Language Models (LLMs) to generate personalized answers to learners' questions, thereby enhancing engagement and reducing the workload on educators. To evaluate the effectiveness of LLMs in this context, we conducted a comprehensive study using the StackExchange platform in two distinct areas: language learning and programming. We developed a framework and a dataset for validating automatically generated personalized answers. Subsequently, we generated personalized answers using different strategies, including 0-shot, 1-shot, and few-shot scenarios. The generated answers were evaluated using three methods: 1. BERTScore, 2. LLM evaluation, and 3. human evaluation. Our findings indicated that providing LLMs with examples of desired answers (from the learner or similar learners) can significantly enhance the LLMs' ability to tailor responses to individual learners' needs. 

---
# Improving Named Entity Transcription with Contextual LLM-based Revision 

**Authors**: Viet Anh Trinh, Xinlu He, Jacob Whitehill  

**Link**: [PDF](https://arxiv.org/pdf/2506.10779)  

**Abstract**: With recent advances in modeling and the increasing amount of supervised training data, automatic speech recognition (ASR) systems have achieved remarkable performance on general speech. However, the word error rate (WER) of state-of-the-art ASR remains high for named entities. Since named entities are often the most critical keywords, misrecognizing them can affect all downstream applications, especially when the ASR system functions as the front end of a complex system. In this paper, we introduce a large language model (LLM) revision mechanism to revise incorrect named entities in ASR predictions by leveraging the LLM's reasoning ability as well as local context (e.g., lecture notes) containing a set of correct named entities. Finally, we introduce the NER-MIT-OpenCourseWare dataset, containing 45 hours of data from MIT courses for development and testing. On this dataset, our proposed technique achieves up to 30\% relative WER reduction for named entities. 

---
# Grounded Vision-Language Navigation for UAVs with Open-Vocabulary Goal Understanding 

**Authors**: Yuhang Zhang, Haosheng Yu, Jiaping Xiao, Mir Feroskhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.10756)  

**Abstract**: Vision-and-language navigation (VLN) is a long-standing challenge in autonomous robotics, aiming to empower agents with the ability to follow human instructions while navigating complex environments. Two key bottlenecks remain in this field: generalization to out-of-distribution environments and reliance on fixed discrete action spaces. To address these challenges, we propose Vision-Language Fly (VLFly), a framework tailored for Unmanned Aerial Vehicles (UAVs) to execute language-guided flight. Without the requirement for localization or active ranging sensors, VLFly outputs continuous velocity commands purely from egocentric observations captured by an onboard monocular camera. The VLFly integrates three modules: an instruction encoder based on a large language model (LLM) that reformulates high-level language into structured prompts, a goal retriever powered by a vision-language model (VLM) that matches these prompts to goal images via vision-language similarity, and a waypoint planner that generates executable trajectories for real-time UAV control. VLFly is evaluated across diverse simulation environments without additional fine-tuning and consistently outperforms all baselines. Moreover, real-world VLN tasks in indoor and outdoor environments under direct and indirect instructions demonstrate that VLFly achieves robust open-vocabulary goal understanding and generalized navigation capabilities, even in the presence of abstract language input. 

---
# Large Language Models for Detection of Life-Threatening Texts 

**Authors**: Thanh Thi Nguyen, Campbell Wilson, Janis Dalins  

**Link**: [PDF](https://arxiv.org/pdf/2506.10687)  

**Abstract**: Detecting life-threatening language is essential for safeguarding individuals in distress, promoting mental health and well-being, and preventing potential harm and loss of life. This paper presents an effective approach to identifying life-threatening texts using large language models (LLMs) and compares them with traditional methods such as bag of words, word embedding, topic modeling, and Bidirectional Encoder Representations from Transformers. We fine-tune three open-source LLMs including Gemma, Mistral, and Llama-2 using their 7B parameter variants on different datasets, which are constructed with class balance, imbalance, and extreme imbalance scenarios. Experimental results demonstrate a strong performance of LLMs against traditional methods. More specifically, Mistral and Llama-2 models are top performers in both balanced and imbalanced data scenarios while Gemma is slightly behind. We employ the upsampling technique to deal with the imbalanced data scenarios and demonstrate that while this method benefits traditional approaches, it does not have as much impact on LLMs. This study demonstrates a great potential of LLMs for real-world life-threatening language detection problems. 

---
# NeuralNexus at BEA 2025 Shared Task: Retrieval-Augmented Prompting for Mistake Identification in AI Tutors 

**Authors**: Numaan Naeem, Sarfraz Ahmad, Momina Ahsan, Hasan Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2506.10627)  

**Abstract**: This paper presents our system for Track 1: Mistake Identification in the BEA 2025 Shared Task on Pedagogical Ability Assessment of AI-powered Tutors. The task involves evaluating whether a tutor's response correctly identifies a mistake in a student's mathematical reasoning. We explore four approaches: (1) an ensemble of machine learning models over pooled token embeddings from multiple pretrained language models (LMs); (2) a frozen sentence-transformer using [CLS] embeddings with an MLP classifier; (3) a history-aware model with multi-head attention between token-level history and response embeddings; and (4) a retrieval-augmented few-shot prompting system with a large language model (LLM) i.e. GPT 4o. Our final system retrieves semantically similar examples, constructs structured prompts, and uses schema-guided output parsing to produce interpretable predictions. It outperforms all baselines, demonstrating the effectiveness of combining example-driven prompting with LLM reasoning for pedagogical feedback assessment. Our code is available at this https URL. 

---
# ConTextTab: A Semantics-Aware Tabular In-Context Learner 

**Authors**: Marco Spinaci, Marek Polewczyk, Maximilian Schambach, Sam Thelin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10707)  

**Abstract**: Tabular in-context learning (ICL) has recently achieved state-of-the-art (SOTA) performance on several tabular prediction tasks. Previously restricted to classification problems on small tables, recent advances such as TabPFN and TabICL have extended its use to larger datasets. While being architecturally efficient and well-adapted to tabular data structures, current table-native ICL architectures, being trained exclusively on synthetic data, do not fully leverage the rich semantics and world knowledge contained in real-world tabular data. On another end of this spectrum, tabular ICL models based on pretrained large language models such as TabuLa-8B integrate deep semantic understanding and world knowledge but are only able to make use of a small amount of context due to inherent architectural limitations. With the aim to combine the best of both these worlds, we introduce ConTextTab, integrating semantic understanding and alignment into a table-native ICL framework. By employing specialized embeddings for different data modalities and by training on large-scale real-world tabular data, our model is competitive with SOTA across a broad set of benchmarks while setting a new standard on the semantically rich CARTE benchmark. 

---
# PREMISE: Scalable and Strategic Prompt Optimization for Efficient Mathematical Reasoning in Large Models 

**Authors**: Ye Yu, Yaoning Yu, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10716)  

**Abstract**: Large reasoning models (LRMs) such as Claude 3.7 Sonnet and OpenAI o1 achieve strong performance on mathematical benchmarks using lengthy chain-of-thought (CoT) reasoning, but the resulting traces are often unnecessarily verbose. This inflates token usage and cost, limiting deployment in latency-sensitive or API-constrained settings. We introduce PREMISE (PRompt-based Efficient Mathematical Inference with Strategic Evaluation), a prompt-only framework that reduces reasoning overhead without modifying model weights. PREMISE combines trace-level diagnostics with gradient-inspired prompt optimization to minimize redundant computation while preserving answer accuracy. The approach jointly optimizes brevity and correctness through a multi-objective textual search that balances token length and answer validity. Unlike prior work, PREMISE runs in a single-pass black-box interface, so it can be applied directly to commercial LLMs. On GSM8K, SVAMP, and Math500 we match or exceed baseline accuracy ($96\%\rightarrow96\%$ with Claude, $91\%\rightarrow92\%$ with Gemini) while reducing reasoning tokens by up to $87.5\%$ and cutting dollar cost by $69$--$82\%$. These results show that prompt-level optimization is a practical and scalable path to efficient LRM inference without compromising reasoning quality. 

---
# StepProof: Step-by-step verification of natural language mathematical proofs 

**Authors**: Xiaolin Hu, Qinghua Zhou, Bogdan Grechuk, Ivan Y. Tyukin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10558)  

**Abstract**: Interactive theorem provers (ITPs) are powerful tools for the formal verification of mathematical proofs down to the axiom level. However, their lack of a natural language interface remains a significant limitation. Recent advancements in large language models (LLMs) have enhanced the understanding of natural language inputs, paving the way for autoformalization - the process of translating natural language proofs into formal proofs that can be verified. Despite these advancements, existing autoformalization approaches are limited to verifying complete proofs and lack the capability for finer, sentence-level verification. To address this gap, we propose StepProof, a novel autoformalization method designed for granular, step-by-step verification. StepProof breaks down complete proofs into multiple verifiable subproofs, enabling sentence-level verification. Experimental results demonstrate that StepProof significantly improves proof success rates and efficiency compared to traditional methods. Additionally, we found that minor manual adjustments to the natural language proofs, tailoring them for step-level verification, further enhanced StepProof's performance in autoformalization. 

---
# CogStream: Context-guided Streaming Video Question Answering 

**Authors**: Zicheng Zhao, Kangyu Wang, Shijie Li, Rui Qian, Weiyao Lin, Huabin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10516)  

**Abstract**: Despite advancements in Video Large Language Models (Vid-LLMs) improving multimodal understanding, challenges persist in streaming video reasoning due to its reliance on contextual information. Existing paradigms feed all available historical contextual information into Vid-LLMs, resulting in a significant computational burden for visual data processing. Furthermore, the inclusion of irrelevant context distracts models from key details. This paper introduces a challenging task called Context-guided Streaming Video Reasoning (CogStream), which simulates real-world streaming video scenarios, requiring models to identify the most relevant historical contextual information to deduce answers for questions about the current stream. To support CogStream, we present a densely annotated dataset featuring extensive and hierarchical question-answer pairs, generated by a semi-automatic pipeline. Additionally, we present CogReasoner as a baseline model. It efficiently tackles this task by leveraging visual stream compression and historical dialogue retrieval. Extensive experiments prove the effectiveness of this method. Code will be released soon. 

---
# From Images to Insights: Explainable Biodiversity Monitoring with Plain Language Habitat Explanations 

**Authors**: Yutong Zhou, Masahiro Ryo  

**Link**: [PDF](https://arxiv.org/pdf/2506.10559)  

**Abstract**: Explaining why the species lives at a particular location is important for understanding ecological systems and conserving biodiversity. However, existing ecological workflows are fragmented and often inaccessible to non-specialists. We propose an end-to-end visual-to-causal framework that transforms a species image into interpretable causal insights about its habitat preference. The system integrates species recognition, global occurrence retrieval, pseudo-absence sampling, and climate data extraction. We then discover causal structures among environmental features and estimate their influence on species occurrence using modern causal inference methods. Finally, we generate statistically grounded, human-readable causal explanations from structured templates and large language models. We demonstrate the framework on a bee and a flower species and report early results as part of an ongoing project, showing the potential of the multimodal AI assistant backed up by a recommended ecological modeling practice for describing species habitat in human-understandable language. 

---
# Time Series Forecasting as Reasoning: A Slow-Thinking Approach with Reinforced LLMs 

**Authors**: Yucong Luo, Yitong Zhou, Mingyue Cheng, Jiahao Wang, Daoyu Wang, Tingyue Pan, Jintao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10630)  

**Abstract**: To advance time series forecasting (TSF), various methods have been proposed to improve prediction accuracy, evolving from statistical techniques to data-driven deep learning architectures. Despite their effectiveness, most existing methods still adhere to a fast thinking paradigm-relying on extracting historical patterns and mapping them to future values as their core modeling philosophy, lacking an explicit thinking process that incorporates intermediate time series reasoning. Meanwhile, emerging slow-thinking LLMs (e.g., OpenAI-o1) have shown remarkable multi-step reasoning capabilities, offering an alternative way to overcome these issues. However, prompt engineering alone presents several limitations - including high computational cost, privacy risks, and limited capacity for in-depth domain-specific time series reasoning. To address these limitations, a more promising approach is to train LLMs to develop slow thinking capabilities and acquire strong time series reasoning skills. For this purpose, we propose Time-R1, a two-stage reinforcement fine-tuning framework designed to enhance multi-step reasoning ability of LLMs for time series forecasting. Specifically, the first stage conducts supervised fine-tuning for warmup adaptation, while the second stage employs reinforcement learning to improve the model's generalization ability. Particularly, we design a fine-grained multi-objective reward specifically for time series forecasting, and then introduce GRIP (group-based relative importance for policy optimization), which leverages non-uniform sampling to further encourage and optimize the model's exploration of effective reasoning paths. Experiments demonstrate that Time-R1 significantly improves forecast performance across diverse datasets. 

---
# SDialog: A Python Toolkit for Synthetic Dialogue Generation and Analysis 

**Authors**: Sergio Burdisso, Esaú Villatoro-Tello, Petr Motlicek  

**Link**: [PDF](https://arxiv.org/pdf/2506.10622)  

**Abstract**: The advancement of conversational AI systems relies on the availability of high-quality, flexible, and reproducible synthetic dialogues for training, evaluation, and benchmarking. SDialog is a modular, extensible Python toolkit designed to address the challenges of synthetic dialogue generation and analysis. By leveraging instruction-tuned Large Language Models (LLMs), SDialog provides abstractions for personas, orchestration, and scenario management, enabling the creation of realistic, diverse, and controllable conversational data for research and development. SDialog supports workflows such as multi-agent simulation and scenario-driven generation, and represents a step forward in the standardization of tools and frameworks for synthetic data generation, a crucial advancement for ensuring reproducibility in today's fast-evolving research landscape. 

---
# Beyond Single-User Dialogue: Assessing Multi-User Dialogue State Tracking Capabilities of Large Language Models 

**Authors**: Sangmin Song, Juhwan Choi, JungMin Yun, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.10504)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance in zero-shot dialogue state tracking (DST), reducing the need for task-specific training. However, conventional DST benchmarks primarily focus on structured user-agent conversations, failing to capture the complexities of real-world multi-user interactions. In this study, we assess the robustness of LLMs in multi-user DST while minimizing dataset construction costs. Inspired by recent advances in LLM-based data annotation, we extend an existing DST dataset by generating utterances of a second user based on speech act theory. Our methodology systematically incorporates a second user's utterances into conversations, enabling a controlled evaluation of LLMs in multi-user settings. Experimental results reveal a significant performance drop compared to single-user DST, highlighting the limitations of current LLMs in extracting and tracking dialogue states amidst multiple speakers. Our findings emphasize the need for future research to enhance LLMs for multi-user DST scenarios, paving the way for more realistic and robust DST models. 

---
# PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier 

**Authors**: Yuhua Jiang, Yuwen Xiong, Yufeng Yuan, Chao Xin, Wenyuan Xu, Yu Yue, Qianchuan Zhao, Lin Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.10406)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in complex reasoning tasks, yet they still struggle to reliably verify the correctness of their own outputs. Existing solutions to this verification challenge often depend on separate verifier models or require multi-stage self-correction training pipelines, which limit scalability. In this paper, we propose Policy as Generative Verifier (PAG), a simple and effective framework that empowers LLMs to self-correct by alternating between policy and verifier roles within a unified multi-turn reinforcement learning (RL) paradigm. Distinct from prior approaches that always generate a second attempt regardless of model confidence, PAG introduces a selective revision mechanism: the model revises its answer only when its own generative verification step detects an error. This verify-then-revise workflow not only alleviates model collapse but also jointly enhances both reasoning and verification abilities. Extensive experiments across diverse reasoning benchmarks highlight PAG's dual advancements: as a policy, it enhances direct generation and self-correction accuracy; as a verifier, its self-verification outperforms self-consistency. 

---
# Specification and Evaluation of Multi-Agent LLM Systems -- Prototype and Cybersecurity Applications 

**Authors**: Felix Härer  

**Link**: [PDF](https://arxiv.org/pdf/2506.10467)  

**Abstract**: Recent advancements in LLMs indicate potential for novel applications, e.g., through reasoning capabilities in the latest OpenAI and DeepSeek models. For applying these models in specific domains beyond text generation, LLM-based multi-agent approaches can be utilized that solve complex tasks by combining reasoning techniques, code generation, and software execution. Applications might utilize these capabilities and the knowledge of specialized LLM agents. However, while many evaluations are performed on LLMs, reasoning techniques, and applications individually, their joint specification and combined application is not explored well. Defined specifications for multi-agent LLM systems are required to explore their potential and their suitability for specific applications, allowing for systematic evaluations of LLMs, reasoning techniques, and related aspects. This paper reports the results of exploratory research to specify and evaluate these aspects through a multi-agent system. The system architecture and prototype are extended from previous research and a specification is introduced for multi-agent systems. Test cases involving cybersecurity tasks indicate feasibility of the architecture and evaluation approach. In particular, the results show the evaluation of question answering, server security, and network security tasks that were completed correctly by agents with LLMs from OpenAI and DeepSeek. 

---
# Time To Impeach LLM-as-a-Judge: Programs are the Future of Evaluation 

**Authors**: Tzu-Heng Huang, Harit Vishwakarma, Frederic Sala  

**Link**: [PDF](https://arxiv.org/pdf/2506.10403)  

**Abstract**: Large language models (LLMs) are widely used to evaluate the quality of LLM generations and responses, but this leads to significant challenges: high API costs, uncertain reliability, inflexible pipelines, and inherent biases. To address these, we introduce PAJAMA (Program-As-a-Judge for Automated Model Assessment), a new alternative that uses LLMs to synthesize executable judging programs instead of directly scoring responses. These synthesized programs can be stored and run locally, costing orders of magnitude less while providing interpretable, and auditable judging logic that can be easily adapted. Program-based judges mitigate biases, improving judgment consistency by 15.83% and reducing biased responses by 23.7% on average compared to a Qwen2.5-14B-based LLM-as-a-judge. When program judgments are distilled into a model, PAJAMA outperforms LLM-as-a-judge on the challenging CHAT-HARD subset of RewardBench, outperforming metrics by 2.19% on Prometheus and 8.67% on the JudgeLM dataset, all at three orders of magnitude lower cost. 

---
# Pisces: An Auto-regressive Foundation Model for Image Understanding and Generation 

**Authors**: Zhiyang Xu, Jiuhai Chen, Zhaojiang Lin, Xichen Pan, Lifu Huang, Tianyi Zhou, Madian Khabsa, Qifan Wang, Di Jin, Michihiro Yasunaga, Lili Yu, Xi Victoria Lin, Shaoliang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.10395)  

**Abstract**: Recent advances in large language models (LLMs) have enabled multimodal foundation models to tackle both image understanding and generation within a unified framework. Despite these gains, unified models often underperform compared to specialized models in either task. A key challenge in developing unified models lies in the inherent differences between the visual features needed for image understanding versus generation, as well as the distinct training processes required for each modality. In this work, we introduce Pisces, an auto-regressive multimodal foundation model that addresses this challenge through a novel decoupled visual encoding architecture and tailored training techniques optimized for multimodal generation. Combined with meticulous data curation, pretraining, and finetuning, Pisces achieves competitive performance in both image understanding and image generation. We evaluate Pisces on over 20 public benchmarks for image understanding, where it demonstrates strong performance across a wide range of tasks. Additionally, on GenEval, a widely adopted benchmark for image generation, Pisces exhibits robust generative capabilities. Our extensive analysis reveals the synergistic relationship between image understanding and generation, and the benefits of using separate visual encoders, advancing the field of unified multimodal models. 

---
# Discovering Hierarchical Latent Capabilities of Language Models via Causal Representation Learning 

**Authors**: Jikai Jin, Vasilis Syrgkanis, Sham Kakade, Hanlin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10378)  

**Abstract**: Faithful evaluation of language model capabilities is crucial for deriving actionable insights that can inform model development. However, rigorous causal evaluations in this domain face significant methodological challenges, including complex confounding effects and prohibitive computational costs associated with extensive retraining. To tackle these challenges, we propose a causal representation learning framework wherein observed benchmark performance is modeled as a linear transformation of a few latent capability factors. Crucially, these latent factors are identified as causally interrelated after appropriately controlling for the base model as a common confounder. Applying this approach to a comprehensive dataset encompassing over 1500 models evaluated across six benchmarks from the Open LLM Leaderboard, we identify a concise three-node linear causal structure that reliably explains the observed performance variations. Further interpretation of this causal structure provides substantial scientific insights beyond simple numerical rankings: specifically, we reveal a clear causal direction starting from general problem-solving capabilities, advancing through instruction-following proficiency, and culminating in mathematical reasoning ability. Our results underscore the essential role of carefully controlling base model variations during evaluation, a step critical to accurately uncovering the underlying causal relationships among latent model capabilities. 

---
# Reliable Reasoning Path: Distilling Effective Guidance for LLM Reasoning with Knowledge Graphs 

**Authors**: Yilin Xiao, Chuang Zhou, Qinggang Zhang, Bo Li, Qing Li, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10508)  

**Abstract**: Large language models (LLMs) often struggle with knowledge-intensive tasks due to a lack of background knowledge and a tendency to hallucinate. To address these limitations, integrating knowledge graphs (KGs) with LLMs has been intensively studied. Existing KG-enhanced LLMs focus on supplementary factual knowledge, but still struggle with solving complex questions. We argue that refining the relationships among facts and organizing them into a logically consistent reasoning path is equally important as factual knowledge itself. Despite their potential, extracting reliable reasoning paths from KGs poses the following challenges: the complexity of graph structures and the existence of multiple generated paths, making it difficult to distinguish between useful and redundant ones. To tackle these challenges, we propose the RRP framework to mine the knowledge graph, which combines the semantic strengths of LLMs with structural information obtained through relation embedding and bidirectional distribution learning. Additionally, we introduce a rethinking module that evaluates and refines reasoning paths according to their significance. Experimental results on two public datasets show that RRP achieves state-of-the-art performance compared to existing baseline methods. Moreover, RRP can be easily integrated into various LLMs to enhance their reasoning abilities in a plug-and-play manner. By generating high-quality reasoning paths tailored to specific questions, RRP distills effective guidance for LLM reasoning. 

---
# Augmenting Large Language Models with Static Code Analysis for Automated Code Quality Improvements 

**Authors**: Seyed Moein Abtahi, Akramul Azim  

**Link**: [PDF](https://arxiv.org/pdf/2506.10330)  

**Abstract**: This study examined code issue detection and revision automation by integrating Large Language Models (LLMs) such as OpenAI's GPT-3.5 Turbo and GPT-4o into software development workflows. A static code analysis framework detects issues such as bugs, vulnerabilities, and code smells within a large-scale software project. Detailed information on each issue was extracted and organized to facilitate automated code revision using LLMs. An iterative prompt engineering process is applied to ensure that prompts are structured to produce accurate and organized outputs aligned with the project requirements. Retrieval-augmented generation (RAG) is implemented to enhance the relevance and precision of the revisions, enabling LLM to access and integrate real-time external knowledge. The issue of LLM hallucinations - where the model generates plausible but incorrect outputs - is addressed by a custom-built "Code Comparison App," which identifies and corrects erroneous changes before applying them to the codebase. Subsequent scans using the static code analysis framework revealed a significant reduction in code issues, demonstrating the effectiveness of combining LLMs, static analysis, and RAG to improve code quality, streamline the software development process, and reduce time and resource expenditure. 

---
# PAL: Probing Audio Encoders via LLMs -- A Study of Information Transfer from Audio Encoders to LLMs 

**Authors**: Tony Alex, Wish Suharitdamrong, Sara Atito, Armin Mustafa, Philip J. B. Jackson, Imran Razzak, Muhammad Awais  

**Link**: [PDF](https://arxiv.org/pdf/2506.10423)  

**Abstract**: The integration of audio perception capabilities into Large Language Models (LLMs) has enabled significant advances in Audio-LLMs. Although application-focused developments, particularly in curating training data for specific capabilities e.g., audio reasoning, have progressed rapidly, the underlying mechanisms that govern efficient transfer of rich semantic representations from audio encoders to LLMs remain under-explored. We conceptualize effective audio-LLM interaction as the LLM's ability to proficiently probe the audio encoder representations to satisfy textual queries. This paper presents a systematic investigation on how architectural design choices can affect that. Beginning with a standard Pengi/LLaVA-style audio-LLM architecture, we propose and evaluate several modifications guided by hypotheses derived from mechanistic interpretability studies and LLM operational principles. Our experiments demonstrate that: (1) delaying audio integration until the LLM's initial layers establish textual context that enhances its ability to probe the audio representations for relevant information; (2) the LLM can proficiently probe audio representations exclusively through LLM layer's attention submodule, without requiring propagation to its Feed-Forward Network (FFN) submodule; (3) an efficiently integrated ensemble of diverse audio encoders provides richer, complementary representations, thereby broadening the LLM's capacity to probe a wider spectrum of audio information. All hypotheses are evaluated using an identical three-stage training curriculum on a dataset of 5.6 million audio-text pairs, ensuring controlled comparisons. Our final architecture, which incorporates all proposed modifications, achieves relative improvements from 10\% to 60\% over the baseline, validating our approach to optimizing cross-modal information transfer in audio-LLMs. Project page: this https URL 

---
# Do Language Models Have Bayesian Brains? Distinguishing Stochastic and Deterministic Decision Patterns within Large Language Models 

**Authors**: Andrea Yaoyun Cui, Pengfei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10268)  

**Abstract**: Language models are essentially probability distributions over token sequences. Auto-regressive models generate sentences by iteratively computing and sampling from the distribution of the next token. This iterative sampling introduces stochasticity, leading to the assumption that language models make probabilistic decisions, similar to sampling from unknown distributions. Building on this assumption, prior research has used simulated Gibbs sampling, inspired by experiments designed to elicit human priors, to infer the priors of language models. In this paper, we revisit a critical question: Do language models possess Bayesian brains? Our findings show that under certain conditions, language models can exhibit near-deterministic decision-making, such as producing maximum likelihood estimations, even with a non-zero sampling temperature. This challenges the sampling assumption and undermines previous methods for eliciting human-like priors. Furthermore, we demonstrate that without proper scrutiny, a system with deterministic behavior undergoing simulated Gibbs sampling can converge to a "false prior." To address this, we propose a straightforward approach to distinguish between stochastic and deterministic decision patterns in Gibbs sampling, helping to prevent the inference of misleading language model priors. We experiment on a variety of large language models to identify their decision patterns under various circumstances. Our results provide key insights in understanding decision making of large language models. 

---
# HPCTransCompile: An AI Compiler Generated Dataset for High-Performance CUDA Transpilation and LLM Preliminary Exploration 

**Authors**: Jiaqi Lv, Xufeng He, Yanchen Liu, Xu Dai, Yang Hu, Shouyi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10401)  

**Abstract**: The rapid growth of deep learning has driven exponential increases in model parameters and computational demands. NVIDIA GPUs and their CUDA-based software ecosystem provide robust support for parallel computing, significantly alleviating computational bottlenecks. Meanwhile, due to the cultivation of user programming habits and the high performance of GPUs, the CUDA ecosystem has established a dominant position in the field of parallel software. This dominance requires other hardware platforms to support CUDA-based software with performance portability. However, translating CUDA code to other platforms poses significant challenges due to differences in parallel programming paradigms and hardware architectures. Existing approaches rely on language extensions, domain-specific languages (DSLs), or compilers but face limitations in workload coverage and generalizability. Moreover, these methods often incur substantial development costs. Recently, LLMs have demonstrated extraordinary potential in various vertical domains, especially in code-related tasks. However, the performance of existing LLMs in CUDA transpilation, particularly for high-performance code, remains suboptimal. The main reason for this limitation lies in the lack of high-quality training datasets. To address these challenges, we propose a novel framework for generating high-performance CUDA and corresponding platform code pairs, leveraging AI compiler and automatic optimization technology. We further enhance the framework with a graph-based data augmentation method and introduce HPCTransEval, a benchmark for evaluating LLM performance on CUDA transpilation. We conduct experiments using CUDA-to-CPU transpilation as a case study on leading LLMs. The result demonstrates that our framework significantly improves CUDA transpilation, highlighting the potential of LLMs to address compatibility challenges within the CUDA ecosystem. 

---
# Code Execution as Grounded Supervision for LLM Reasoning 

**Authors**: Dongwon Jung, Wenxuan Zhou, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10343)  

**Abstract**: Training large language models (LLMs) with chain-of-thought (CoT) supervision has proven effective for enhancing their reasoning abilities. However, obtaining reliable and accurate reasoning supervision remains a significant challenge. We propose a scalable method for generating a high-quality CoT supervision dataset by leveraging the determinism of program execution. Unlike existing reasoning dataset generation methods that rely on costly human annotations or error-prone LLM-generated CoT, our approach extracts verifiable, step-by-step reasoning traces from code execution and transforms them into a natural language CoT reasoning. Experiments on reasoning benchmarks across various domains show that our method effectively equips LLMs with transferable reasoning abilities across diverse tasks. Furthermore, the ablation studies validate that our method produces highly accurate reasoning data and reduces overall token length during inference by reducing meaningless repetition and overthinking. 

---
# Unsupervised Elicitation of Language Models 

**Authors**: Jiaxin Wen, Zachary Ankner, Arushi Somani, Peter Hase, Samuel Marks, Jacob Goldman-Wetzler, Linda Petrini, Henry Sleight, Collin Burns, He He, Shi Feng, Ethan Perez, Jan Leike  

**Link**: [PDF](https://arxiv.org/pdf/2506.10139)  

**Abstract**: To steer pretrained language models for downstream tasks, today's post-training paradigm relies on humans to specify desired behaviors. However, for models with superhuman capabilities, it is difficult or impossible to get high-quality human supervision. To address this challenge, we introduce a new unsupervised algorithm, Internal Coherence Maximization (ICM), to fine-tune pretrained language models on their own generated labels, \emph{without external supervision}. On GSM8k-verification, TruthfulQA, and Alpaca reward modeling tasks, our method matches the performance of training on golden supervision and outperforms training on crowdsourced human supervision. On tasks where LMs' capabilities are strongly superhuman, our method can elicit those capabilities significantly better than training on human labels. Finally, we show that our method can improve the training of frontier LMs: we use our method to train an unsupervised reward model and use reinforcement learning to train a Claude 3.5 Haiku-based assistant. Both the reward model and the assistant outperform their human-supervised counterparts. 

---
# Can LLMs Generate Good Stories? Insights and Challenges from a Narrative Planning Perspective 

**Authors**: Yi Wang, Max Kreminski  

**Link**: [PDF](https://arxiv.org/pdf/2506.10161)  

**Abstract**: Story generation has been a prominent application of Large Language Models (LLMs). However, understanding LLMs' ability to produce high-quality stories remains limited due to challenges in automatic evaluation methods and the high cost and subjectivity of manual evaluation. Computational narratology offers valuable insights into what constitutes a good story, which has been applied in the symbolic narrative planning approach to story generation. This work aims to deepen the understanding of LLMs' story generation capabilities by using them to solve narrative planning problems. We present a benchmark for evaluating LLMs on narrative planning based on literature examples, focusing on causal soundness, character intentionality, and dramatic conflict. Our experiments show that GPT-4 tier LLMs can generate causally sound stories at small scales, but planning with character intentionality and dramatic conflict remains challenging, requiring LLMs trained with reinforcement learning for complex reasoning. The results offer insights on the scale of stories that LLMs can generate while maintaining quality from different aspects. Our findings also highlight interesting problem solving behaviors and shed lights on challenges and considerations for applying LLM narrative planning in game environments. 

---
# One For All: LLM-based Heterogeneous Mission Planning in Precision Agriculture 

**Authors**: Marcos Abel Zuzuárregui, Mustafa Melih Toslak, Stefano Carpin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10106)  

**Abstract**: Artificial intelligence is transforming precision agriculture, offering farmers new tools to streamline their daily operations. While these technological advances promise increased efficiency, they often introduce additional complexity and steep learning curves that are particularly challenging for non-technical users who must balance tech adoption with existing workloads. In this paper, we present a natural language (NL) robotic mission planner that enables non-specialists to control heterogeneous robots through a common interface. By leveraging large language models (LLMs) and predefined primitives, our architecture seamlessly translates human language into intermediate descriptions that can be executed by different robotic platforms. With this system, users can formulate complex agricultural missions without writing any code. In the work presented in this paper, we extend our previous system tailored for wheeled robot mission planning through a new class of experiments involving robotic manipulation and computer vision tasks. Our results demonstrate that the architecture is both general enough to support a diverse set of robots and powerful enough to execute complex mission requests. This work represents a significant step toward making robotic automation in precision agriculture more accessible to non-technical users. 

---
# Leveraging LLMs for Mission Planning in Precision Agriculture 

**Authors**: Marcos Abel Zuzuárregui, Stefano Carpin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10093)  

**Abstract**: Robotics and artificial intelligence hold significant potential for advancing precision agriculture. While robotic systems have been successfully deployed for various tasks, adapting them to perform diverse missions remains challenging, particularly because end users often lack technical expertise. In this paper, we present an end-to-end system that leverages large language models (LLMs), specifically ChatGPT, to enable users to assign complex data collection tasks to autonomous robots using natural language instructions. To enhance reusability, mission plans are encoded using an existing IEEE task specification standard, and are executed on robots via ROS2 nodes that bridge high-level mission descriptions with existing ROS libraries. Through extensive experiments, we highlight the strengths and limitations of LLMs in this context, particularly regarding spatial reasoning and solving complex routing challenges, and show how our proposed implementation overcomes them. 

---
# Textual Bayes: Quantifying Uncertainty in LLM-Based Systems 

**Authors**: Brendan Leigh Ross, Noël Vouitsis, Atiyeh Ashari Ghomi, Rasa Hosseinzadeh, Ji Xin, Zhaoyan Liu, Yi Sui, Shiyi Hou, Kin Kwan Leung, Gabriel Loaiza-Ganem, Jesse C. Cresswell  

**Link**: [PDF](https://arxiv.org/pdf/2506.10060)  

**Abstract**: Although large language models (LLMs) are becoming increasingly capable of solving challenging real-world tasks, accurately quantifying their uncertainty remains a critical open problem, which limits their applicability in high-stakes domains. This challenge is further compounded by the closed-source, black-box nature of many state-of-the-art LLMs. Moreover, LLM-based systems can be highly sensitive to the prompts that bind them together, which often require significant manual tuning (i.e., prompt engineering). In this work, we address these challenges by viewing LLM-based systems through a Bayesian lens. We interpret prompts as textual parameters in a statistical model, allowing us to use a small training dataset to perform Bayesian inference over these prompts. This novel perspective enables principled uncertainty quantification over both the model's textual parameters and its downstream predictions, while also incorporating prior beliefs about these parameters expressed in free-form text. To perform Bayesian inference, a difficult problem even for well-studied data modalities, we introduce Metropolis-Hastings through LLM Proposals (MHLP), a novel Markov chain Monte Carlo (MCMC) algorithm that combines prompt optimization techniques with standard MCMC methods. MHLP is a turnkey modification to existing LLM pipelines, including those that rely exclusively on closed-source models. Empirically, we demonstrate that our method yields improvements in both predictive accuracy and uncertainty quantification (UQ) on a range of LLM benchmarks and UQ tasks. More broadly, our work demonstrates a viable path for incorporating methods from the rich Bayesian literature into the era of LLMs, paving the way for more reliable and calibrated LLM-based systems. 

---
# From Tool Calling to Symbolic Thinking: LLMs in a Persistent Lisp Metaprogramming Loop 

**Authors**: Jordi de la Torre  

**Link**: [PDF](https://arxiv.org/pdf/2506.10021)  

**Abstract**: We propose a novel architecture for integrating large language models (LLMs) with a persistent, interactive Lisp environment. This setup enables LLMs to define, invoke, and evolve their own tools through programmatic interaction with a live REPL. By embedding Lisp expressions within generation and intercepting them via a middleware layer, the system allows for stateful external memory, reflective programming, and dynamic tool creation. We present a design framework and architectural principles to guide future implementations of interactive AI systems that integrate symbolic programming with neural language generation. 

---
# LLMs Caught in the Crossfire: Malware Requests and Jailbreak Challenges 

**Authors**: Haoyang Li, Huan Gao, Zhiyuan Zhao, Zhiyu Lin, Junyu Gao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.10022)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) has heightened concerns about their security, particularly their vulnerability to jailbreak attacks that leverage crafted prompts to generate malicious outputs. While prior research has been conducted on general security capabilities of LLMs, their specific susceptibility to jailbreak attacks in code generation remains largely unexplored. To fill this gap, we propose MalwareBench, a benchmark dataset containing 3,520 jailbreaking prompts for malicious code-generation, designed to evaluate LLM robustness against such threats. MalwareBench is based on 320 manually crafted malicious code generation requirements, covering 11 jailbreak methods and 29 code functionality categories. Experiments show that mainstream LLMs exhibit limited ability to reject malicious code-generation requirements, and the combination of multiple jailbreak methods further reduces the model's security capabilities: specifically, the average rejection rate for malicious content is 60.93%, dropping to 39.92% when combined with jailbreak attack algorithms. Our work highlights that the code security capabilities of LLMs still pose significant challenges. 

---
# From Threat to Tool: Leveraging Refusal-Aware Injection Attacks for Safety Alignment 

**Authors**: Kyubyung Chae, Hyunbin Jin, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.10020)  

**Abstract**: Safely aligning large language models (LLMs) often demands extensive human-labeled preference data, a process that's both costly and time-consuming. While synthetic data offers a promising alternative, current methods frequently rely on complex iterative prompting or auxiliary models. To address this, we introduce Refusal-Aware Adaptive Injection (RAAI), a straightforward, training-free, and model-agnostic framework that repurposes LLM attack techniques. RAAI works by detecting internal refusal signals and adaptively injecting predefined phrases to elicit harmful, yet fluent, completions. Our experiments show RAAI effectively jailbreaks LLMs, increasing the harmful response rate from a baseline of 2.15% to up to 61.04% on average across four benchmarks. Crucially, fine-tuning LLMs with the synthetic data generated by RAAI improves model robustness against harmful prompts while preserving general capabilities on standard tasks like MMLU and ARC. This work highlights how LLM attack methodologies can be reframed as practical tools for scalable and controllable safety alignment. 

---
# Tina: Tiny Reasoning Models via LoRA 

**Authors**: Shangshang Wang, Julian Asilis, Ömer Faruk Akgül, Enes Burak Bilgin, Ollie Liu, Willie Neiswanger  

**Link**: [PDF](https://arxiv.org/pdf/2504.15777)  

**Abstract**: How cost-effectively can strong reasoning abilities be achieved in language models? Driven by this fundamental question, we present Tina, a family of tiny reasoning models achieved with high cost-efficiency. Notably, Tina demonstrates that substantial reasoning performance can be developed using only minimal resources, by applying parameter-efficient updates during reinforcement learning (RL), using low-rank adaptation (LoRA), to an already tiny 1.5B parameter base model. This minimalist approach produces models that achieve reasoning performance which is competitive with, and sometimes surpasses, SOTA RL reasoning models built upon the same base model. Crucially, this is achieved at a tiny fraction of the computational post-training cost employed by existing SOTA models. In fact, the best Tina model achieves a >20\% reasoning performance increase and 43.33\% Pass@1 accuracy on AIME24, at only \$9 USD post-training and evaluation cost (i.e., an estimated 260x cost reduction). Our work reveals the surprising effectiveness of efficient RL reasoning via LoRA. We validate this across multiple open-source reasoning datasets and various ablation settings starting with a single, fixed set of hyperparameters. Furthermore, we hypothesize that this effectiveness and efficiency stem from LoRA rapidly adapting the model to the structural format of reasoning rewarded by RL, while largely preserving the base model's underlying knowledge. In service of accessibility and open research, we fully open-source all code, training logs, and model weights \& checkpoints. 

---
# Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers 

**Authors**: Yixiao Huang, Hanlin Zhu, Tianyu Guo, Jiantao Jiao, Somayeh Sojoudi, Michael I. Jordan, Stuart Russell, Song Mei  

**Link**: [PDF](https://arxiv.org/pdf/2506.10887)  

**Abstract**: Large language models (LLMs) can acquire new knowledge through fine-tuning, but this process exhibits a puzzling duality: models can generalize remarkably from new facts, yet are also prone to hallucinating incorrect information. However, the reasons for this phenomenon remain poorly understood. In this work, we argue that both behaviors stem from a single mechanism known as out-of-context reasoning (OCR): the ability to deduce implications by associating concepts, even those without a causal link. Our experiments across five prominent LLMs confirm that OCR indeed drives both generalization and hallucination, depending on whether the associated concepts are causally related. To build a rigorous theoretical understanding of this phenomenon, we then formalize OCR as a synthetic factual recall task. We empirically show that a one-layer single-head attention-only transformer with factorized output and value matrices can learn to solve this task, while a model with combined weights cannot, highlighting the crucial role of matrix factorization. Our theoretical analysis shows that the OCR capability can be attributed to the implicit bias of gradient descent, which favors solutions that minimize the nuclear norm of the combined output-value matrix. This mathematical structure explains why the model learns to associate facts and implications with high sample efficiency, regardless of whether the correlation is causal or merely spurious. Ultimately, our work provides a theoretical foundation for understanding the OCR phenomenon, offering a new lens for analyzing and mitigating undesirable behaviors from knowledge injection. 

---
# Magistral 

**Authors**: Mistral-AI, Abhinav Rastogi, Albert Q. Jiang, Andy Lo, Gabrielle Berrada, Guillaume Lample, Jason Rute, Joep Barmentlo, Karmesh Yadav, Kartik Khandelwal, Khyathi Raghavi Chandu, Léonard Blier, Lucile Saulnier, Matthieu Dinot, Maxime Darrin, Neha Gupta, Roman Soletskyi, Sagar Vaze, Teven Le Scao, Yihan Wang, Adam Yang, Alexander H. Liu, Alexandre Sablayrolles, Amélie Héliou, Amélie Martin, Andy Ehrenberg, Anmol Agarwal, Antoine Roux, Arthur Darcet, Arthur Mensch, Baptiste Bout, Baptiste Rozière, Baudouin De Monicault, Chris Bamford, Christian Wallenwein, Christophe Renaudin, Clémence Lanfranchi, Darius Dabert, Devon Mizelle, Diego de las Casas, Elliot Chane-Sane, Emilien Fugier, Emma Bou Hanna, Gauthier Delerce, Gauthier Guinet, Georgii Novikov, Guillaume Martin, Himanshu Jaju, Jan Ludziejewski, Jean-Hadrien Chabran, Jean-Malo Delignon, Joachim Studnia, Jonas Amar, Josselin Somerville Roberts, Julien Denize, Karan Saxena, Kush Jain, Lingxiao Zhao, Louis Martin, Luyu Gao, Lélio Renard Lavaud, Marie Pellat, Mathilde Guillaumin, Mathis Felardos, Maximilian Augustin, Mickaël Seznec, Nikhil Raghuraman, Olivier Duchenne, Patricia Wang, Patrick von Platen, Patryk Saffer, Paul Jacob, Paul Wambergue, Paula Kurylowicz, Pavankumar Reddy Muddireddy, Philomène Chagniot, Pierre Stock, Pravesh Agrawal, Romain Sauvestre, Rémi Delacourt, Sanchit Gandhi, Sandeep Subramanian, Shashwat Dalal, Siddharth Gandhi, Soham Ghosh, Srijan Mishra, Sumukh Aithal, Szymon Antoniak, Thibault Schueller, Thibaut Lavril, Thomas Robert, Thomas Wang, Timothée Lacroix, Valeriia Nemychnikova, Victor Paltz, Virgile Richard, Wen-Ding Li, William Marshall, Xuanyu Zhang, Yunhao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10910)  

**Abstract**: We introduce Magistral, Mistral's first reasoning model and our own scalable reinforcement learning (RL) pipeline. Instead of relying on existing implementations and RL traces distilled from prior models, we follow a ground up approach, relying solely on our own models and infrastructure. Notably, we demonstrate a stack that enabled us to explore the limits of pure RL training of LLMs, present a simple method to force the reasoning language of the model, and show that RL on text data alone maintains most of the initial checkpoint's capabilities. We find that RL on text maintains or improves multimodal understanding, instruction following and function calling. We present Magistral Medium, trained for reasoning on top of Mistral Medium 3 with RL alone, and we open-source Magistral Small (Apache 2.0) which further includes cold-start data from Magistral Medium. 

---
# ReCUT: Balancing Reasoning Length and Accuracy in LLMs via Stepwise Trails and Preference Optimization 

**Authors**: Zhensheng Jin, Xinze Li, Yifan Ji, Chunyi Peng, Zhenghao Liu, Qi Shi, Yukun Yan, Shuo Wang, Furong Peng, Ge Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10822)  

**Abstract**: Recent advances in Chain-of-Thought (CoT) prompting have substantially improved the reasoning capabilities of Large Language Models (LLMs). However, these methods often suffer from overthinking, leading to unnecessarily lengthy or redundant reasoning traces. Existing approaches attempt to mitigate this issue through curating multiple reasoning chains for training LLMs, but their effectiveness is often constrained by the quality of the generated data and prone to overfitting. To address the challenge, we propose Reasoning Compression ThroUgh Stepwise Trials (ReCUT), a novel method aimed at balancing the accuracy and length of reasoning trajectory. Specifically, ReCUT employs a stepwise exploration mechanism and a long-short switched sampling strategy, enabling LLMs to incrementally generate diverse reasoning paths. These paths are evaluated and used to construct preference pairs to train two specialized models (Gemini LLMs)-one optimized for reasoning accuracy, the other for shorter reasoning. A final integrated model is obtained by interpolating the parameters of these two models. Experimental results across multiple math reasoning datasets and backbone models demonstrate that ReCUT significantly reduces reasoning lengths by approximately 30-50%, while maintaining or improving reasoning accuracy compared to various baselines. All codes and data will be released via this https URL. 

---
# Beyond Gold Standards: Epistemic Ensemble of LLM Judges for Formal Mathematical Reasoning 

**Authors**: Lan Zhang, Marco Valentino, Andre Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2506.10903)  

**Abstract**: Autoformalization plays a crucial role in formal mathematical reasoning by enabling the automatic translation of natural language statements into formal languages. While recent advances using large language models (LLMs) have shown promising results, methods for automatically evaluating autoformalization remain underexplored. As one moves to more complex domains (e.g., advanced mathematics), human evaluation requires significant time and domain expertise, especially as the complexity of the underlying statements and background knowledge increases. LLM-as-a-judge presents a promising approach for automating such evaluation. However, existing methods typically employ coarse-grained and generic evaluation criteria, which limit their effectiveness for advanced formal mathematical reasoning, where quality hinges on nuanced, multi-granular dimensions. In this work, we take a step toward addressing this gap by introducing a systematic, automatic method to evaluate autoformalization tasks. The proposed method is based on an epistemically and formally grounded ensemble (EFG) of LLM judges, defined on criteria encompassing logical preservation (LP), mathematical consistency (MC), formal validity (FV), and formal quality (FQ), resulting in a transparent assessment that accounts for different contributing factors. We validate the proposed framework to serve as a proxy for autoformalization assessment within the domain of formal mathematics. Overall, our experiments demonstrate that the EFG ensemble of LLM judges is a suitable emerging proxy for evaluation, more strongly correlating with human assessments than a coarse-grained model, especially when assessing formal qualities. These findings suggest that LLM-as-judges, especially when guided by a well-defined set of atomic properties, could offer a scalable, interpretable, and reliable support for evaluating formal mathematical reasoning. 

---
# Mitigating Negative Interference in Multilingual Sequential Knowledge Editing through Null-Space Constraints 

**Authors**: Wei Sun, Tingyu Qu, Mingxiao Li, Jesse Davis, Marie-Francine Moens  

**Link**: [PDF](https://arxiv.org/pdf/2506.10800)  

**Abstract**: Efficiently updating multilingual knowledge in large language models (LLMs), while preserving consistent factual representations across languages, remains a long-standing and unresolved challenge. While deploying separate editing systems for each language might seem viable, this approach incurs substantial costs due to the need to manage multiple models. A more efficient solution involves integrating knowledge updates across all languages into a unified model. However, performing sequential edits across languages often leads to destructive parameter interference, significantly degrading multilingual generalization and the accuracy of injected knowledge. To address this challenge, we propose LangEdit, a novel null-space constrained framework designed to precisely isolate language-specific knowledge updates. The core innovation of LangEdit lies in its ability to project parameter updates for each language onto the orthogonal complement of previous updated subspaces. This approach mathematically guarantees update independence while preserving multilingual generalization capabilities. We conduct a comprehensive evaluation across three model architectures, six languages, and four downstream tasks, demonstrating that LangEdit effectively mitigates parameter interference and outperforms existing state-of-the-art editing methods. Our results highlight its potential for enabling efficient and accurate multilingual knowledge updates in LLMs. The code is available at this https URL. 

---
# Different Questions, Different Models: Fine-Grained Evaluation of Uncertainty and Calibration in Clinical QA with LLMs 

**Authors**: Alberto Testoni, Iacer Calixto  

**Link**: [PDF](https://arxiv.org/pdf/2506.10769)  

**Abstract**: Accurate and well-calibrated uncertainty estimates are essential for deploying large language models (LLMs) in high-stakes domains such as clinical decision support. We present a fine-grained evaluation of uncertainty estimation methods for clinical multiple-choice question answering, covering ten open-source LLMs (general-purpose, biomedical, and reasoning models) across two datasets, eleven medical specialties, and six question types. We compare standard single-generation and sampling-based methods, and present a case study exploring simple, single-pass estimators based on behavioral signals in reasoning traces. These lightweight methods approach the performance of Semantic Entropy while requiring only one generation. Our results reveal substantial variation across specialties and question types, underscoring the importance of selecting models based on both the nature of the question and model-specific strengths. 

---
# One Tokenizer To Rule Them All: Emergent Language Plasticity via Multilingual Tokenizers 

**Authors**: Diana Abagyan, Alejandro R. Salamanca, Andres Felipe Cruz-Salinas, Kris Cao, Hangyu Lin, Acyr Locatelli, Marzieh Fadaee, Ahmet Üstün, Sara Hooker  

**Link**: [PDF](https://arxiv.org/pdf/2506.10766)  

**Abstract**: Pretraining massively multilingual Large Language Models (LLMs) for many languages at once is challenging due to limited model capacity, scarce high-quality data, and compute constraints. Moreover, the lack of language coverage of the tokenizer makes it harder to address the gap for new languages purely at the post-training stage. In this work, we study what relatively cheap interventions early on in training improve "language plasticity", or adaptation capabilities of the model post-training to new languages. We focus on tokenizer design and propose using a universal tokenizer that is trained for more languages than the primary pretraining languages to enable efficient adaptation in expanding language coverage after pretraining. Our systematic experiments across diverse groups of languages and different training strategies show that a universal tokenizer enables significantly higher language adaptation, with up to 20.2% increase in win rates compared to tokenizers specific to pretraining languages. Furthermore, a universal tokenizer also leads to better plasticity towards languages that are completely unseen in the tokenizer and pretraining, by up to 5% win rate gain. We achieve this adaptation to an expanded set of languages with minimal compromise in performance on the majority of languages included in pretraining. 

---
# Inferring Adjective Hypernyms with Language Models to Increase the Connectivity of Open English Wordnet 

**Authors**: Lorenzo Augello, John P. McCrae  

**Link**: [PDF](https://arxiv.org/pdf/2506.10715)  

**Abstract**: Open English Wordnet is a key resource published in OntoLex-lemon as part of the linguistic linked open data cloud. There are, however, many links missing in the resource, and in this paper, we look at how we can establish hypernymy between adjectives. We present a theoretical discussion of the hypernymy relation and how it differs for adjectives in contrast to nouns and verbs. We develop a new resource for adjective hypernymy and fine-tune large language models to predict adjective hypernymy, showing that the methodology of TaxoLLaMa can be adapted to this task. 

---
# Spelling-out is not Straightforward: LLMs' Capability of Tokenization from Token to Characters 

**Authors**: Tatsuya Hiraoka, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2506.10641)  

**Abstract**: Large language models (LLMs) can spell out tokens character by character with high accuracy, yet they struggle with more complex character-level tasks, such as identifying compositional subcomponents within tokens. In this work, we investigate how LLMs internally represent and utilize character-level information during the spelling-out process. Our analysis reveals that, although spelling out is a simple task for humans, it is not handled in a straightforward manner by LLMs. Specifically, we show that the embedding layer does not fully encode character-level information, particularly beyond the first character. As a result, LLMs rely on intermediate and higher Transformer layers to reconstruct character-level knowledge, where we observe a distinct "breakthrough" in their spelling behavior. We validate this mechanism through three complementary analyses: probing classifiers, identification of knowledge neurons, and inspection of attention weights. 

---
# Surface Fairness, Deep Bias: A Comparative Study of Bias in Language Models 

**Authors**: Aleksandra Sorokovikova, Pavel Chizhov, Iuliia Eremenko, Ivan P. Yamshchikov  

**Link**: [PDF](https://arxiv.org/pdf/2506.10491)  

**Abstract**: Modern language models are trained on large amounts of data. These data inevitably include controversial and stereotypical content, which contains all sorts of biases related to gender, origin, age, etc. As a result, the models express biased points of view or produce different results based on the assigned personality or the personality of the user. In this paper, we investigate various proxy measures of bias in large language models (LLMs). We find that evaluating models with pre-prompted personae on a multi-subject benchmark (MMLU) leads to negligible and mostly random differences in scores. However, if we reformulate the task and ask a model to grade the user's answer, this shows more significant signs of bias. Finally, if we ask the model for salary negotiation advice, we see pronounced bias in the answers. With the recent trend for LLM assistant memory and personalization, these problems open up from a different angle: modern LLM users do not need to pre-prompt the description of their persona since the model already knows their socio-demographics. 

---
# Fast on the Easy, Deep on the Hard: Efficient Reasoning via Powered Length Penalty 

**Authors**: Zehui Ling, Deshu Chen, Hongwei Zhang, Yifeng Jiao, Xin Guo, Yuan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.10446)  

**Abstract**: Large language models (LLMs) have demonstrated significant advancements in reasoning capabilities, performing well on various challenging benchmarks. Techniques like Chain-of-Thought prompting have been introduced to further improve reasoning. However, these approaches frequently generate longer outputs, which in turn increase computational latency. Although some methods use reinforcement learning to shorten reasoning, they often apply uniform penalties without considering the problem's complexity, leading to suboptimal outcomes. In this study, we seek to enhance the efficiency of LLM reasoning by promoting conciseness for simpler problems while preserving sufficient reasoning for more complex ones for accuracy, thus improving the model's overall performance. Specifically, we manage the model's reasoning efficiency by dividing the reward function and including a novel penalty for output length. Our approach has yielded impressive outcomes in benchmark evaluations across three datasets: GSM8K, MATH500, and AIME2024. For the comparatively simpler datasets GSM8K and MATH500, our method has effectively shortened output lengths while preserving or enhancing accuracy. On the more demanding AIME2024 dataset, our approach has resulted in improved accuracy. 

---
# Decomposing MLP Activations into Interpretable Features via Semi-Nonnegative Matrix Factorization 

**Authors**: Or Shafran, Atticus Geiger, Mor Geva  

**Link**: [PDF](https://arxiv.org/pdf/2506.10920)  

**Abstract**: A central goal for mechanistic interpretability has been to identify the right units of analysis in large language models (LLMs) that causally explain their outputs. While early work focused on individual neurons, evidence that neurons often encode multiple concepts has motivated a shift toward analyzing directions in activation space. A key question is how to find directions that capture interpretable features in an unsupervised manner. Current methods rely on dictionary learning with sparse autoencoders (SAEs), commonly trained over residual stream activations to learn directions from scratch. However, SAEs often struggle in causal evaluations and lack intrinsic interpretability, as their learning is not explicitly tied to the computations of the model. Here, we tackle these limitations by directly decomposing MLP activations with semi-nonnegative matrix factorization (SNMF), such that the learned features are (a) sparse linear combinations of co-activated neurons, and (b) mapped to their activating inputs, making them directly interpretable. Experiments on Llama 3.1, Gemma 2 and GPT-2 show that SNMF derived features outperform SAEs and a strong supervised baseline (difference-in-means) on causal steering, while aligning with human-interpretable concepts. Further analysis reveals that specific neuron combinations are reused across semantically-related features, exposing a hierarchical structure in the MLP's activation space. Together, these results position SNMF as a simple and effective tool for identifying interpretable features and dissecting concept representations in LLMs. 

---
# Burn After Reading: Do Multimodal Large Language Models Truly Capture Order of Events in Image Sequences? 

**Authors**: Yingjin Song, Yupei Du, Denis Paperno, Albert Gatt  

**Link**: [PDF](https://arxiv.org/pdf/2506.10415)  

**Abstract**: This paper introduces the TempVS benchmark, which focuses on temporal grounding and reasoning capabilities of Multimodal Large Language Models (MLLMs) in image sequences. TempVS consists of three main tests (i.e., event relation inference, sentence ordering and image ordering), each accompanied with a basic grounding test. TempVS requires MLLMs to rely on both visual and linguistic modalities to understand the temporal order of events. We evaluate 38 state-of-the-art MLLMs, demonstrating that models struggle to solve TempVS, with a substantial performance gap compared to human capabilities. We also provide fine-grained insights that suggest promising directions for future research. Our TempVS benchmark data and code are available at this https URL. 

---
# Scheduled Interleaved Speech-Text Training for Speech-to-Speech Translation with LLMs 

**Authors**: Hayato Futami, Emiru Tsunoo, Yosuke Kashiwagi, Yuki Ito, Hassan Shahmohammadi, Siddhant Arora, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.10299)  

**Abstract**: Speech-to-speech translation (S2ST) has been advanced with large language models (LLMs), which are fine-tuned on discrete speech units. In such approaches, modality adaptation from text to speech has been an issue. LLMs are trained on text-only data, which presents challenges to adapt them to speech modality with limited speech-to-speech data. To address the training difficulty, we propose scheduled interleaved speech--text training in this study. We use interleaved speech--text units instead of speech units during training, where aligned text tokens are interleaved at the word level. We gradually decrease the ratio of text as training progresses, to facilitate progressive modality adaptation from text to speech. We conduct experimental evaluations by fine-tuning LLaMA3.2-1B for S2ST on the CVSS dataset. We show that the proposed method consistently improves the translation performances, especially for languages with limited training data. 

---
# Classifying Unreliable Narrators with Large Language Models 

**Authors**: Anneliese Brei, Katharine Henry, Abhisheik Sharma, Shashank Srivastava, Snigdha Chaturvedi  

**Link**: [PDF](https://arxiv.org/pdf/2506.10231)  

**Abstract**: Often when we interact with a first-person account of events, we consider whether or not the narrator, the primary speaker of the text, is reliable. In this paper, we propose using computational methods to identify unreliable narrators, i.e. those who unintentionally misrepresent information. Borrowing literary theory from narratology to define different types of unreliable narrators based on a variety of textual phenomena, we present TUNa, a human-annotated dataset of narratives from multiple domains, including blog posts, subreddit posts, hotel reviews, and works of literature. We define classification tasks for intra-narrational, inter-narrational, and inter-textual unreliabilities and analyze the performance of popular open-weight and proprietary LLMs for each. We propose learning from literature to perform unreliable narrator classification on real-world text data. To this end, we experiment with few-shot, fine-tuning, and curriculum learning settings. Our results show that this task is very challenging, and there is potential for using LLMs to identify unreliable narrators. We release our expert-annotated dataset and code and invite future research in this area. 

---
# Q2E: Query-to-Event Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval 

**Authors**: Shubhashis Roy Dipta, Francis Ferraro  

**Link**: [PDF](https://arxiv.org/pdf/2506.10202)  

**Abstract**: Recent approaches have shown impressive proficiency in extracting and leveraging parametric knowledge from Large-Language Models (LLMs) and Vision-Language Models (VLMs). In this work, we consider how we can improve the identification and retrieval of videos related to complex real-world events by automatically extracting latent parametric knowledge about those events. We present Q2E: a Query-to-Event decomposition method for zero-shot multilingual text-to-video retrieval, adaptable across datasets, domains, LLMs, or VLMs. Our approach demonstrates that we can enhance the understanding of otherwise overly simplified human queries by decomposing the query using the knowledge embedded in LLMs and VLMs. We additionally show how to apply our approach to both visual and speech-based inputs. To combine this varied multimodal knowledge, we adopt entropy-based fusion scoring for zero-shot fusion. Through evaluations on two diverse datasets and multiple retrieval metrics, we demonstrate that Q2E outperforms several state-of-the-art baselines. Our evaluation also shows that integrating audio information can significantly improve text-to-video retrieval. We have released code and data for future research. 

---
# "Check My Work?": Measuring Sycophancy in a Simulated Educational Context 

**Authors**: Chuck Arvin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10297)  

**Abstract**: This study examines how user-provided suggestions affect Large Language Models (LLMs) in a simulated educational context, where sycophancy poses significant risks. Testing five different LLMs from the OpenAI GPT-4o and GPT-4.1 model classes across five experimental conditions, we show that response quality varies dramatically based on query framing. In cases where the student mentions an incorrect answer, the LLM correctness can degrade by as much as 15 percentage points, while mentioning the correct answer boosts accuracy by the same margin. Our results also show that this bias is stronger in smaller models, with an effect of up to 30% for the GPT-4.1-nano model, versus 8% for the GPT-4o model. Our analysis of how often LLMs "flip" their answer, and an investigation into token level probabilities, confirm that the models are generally changing their answers to answer choices mentioned by students in line with the sycophancy hypothesis. This sycophantic behavior has important implications for educational equity, as LLMs may accelerate learning for knowledgeable students while the same tools may reinforce misunderstanding for less knowledgeable students. Our results highlight the need to better understand the mechanism, and ways to mitigate, such bias in the educational context. 

---
# When Large Language Models are Reliable for Judging Empathic Communication 

**Authors**: Aakriti Kumar, Nalin Poungpeth, Diyi Yang, Erina Farrell, Bruce Lambert, Matthew Groh  

**Link**: [PDF](https://arxiv.org/pdf/2506.10150)  

**Abstract**: Large language models (LLMs) excel at generating empathic responses in text-based conversations. But, how reliably do they judge the nuances of empathic communication? We investigate this question by comparing how experts, crowdworkers, and LLMs annotate empathic communication across four evaluative frameworks drawn from psychology, natural language processing, and communications applied to 200 real-world conversations where one speaker shares a personal problem and the other offers support. Drawing on 3,150 expert annotations, 2,844 crowd annotations, and 3,150 LLM annotations, we assess inter-rater reliability between these three annotator groups. We find that expert agreement is high but varies across the frameworks' sub-components depending on their clarity, complexity, and subjectivity. We show that expert agreement offers a more informative benchmark for contextualizing LLM performance than standard classification metrics. Across all four frameworks, LLMs consistently approach this expert level benchmark and exceed the reliability of crowdworkers. These results demonstrate how LLMs, when validated on specific tasks with appropriate benchmarks, can support transparency and oversight in emotionally sensitive applications including their use as conversational companions. 

---
# Chat-of-Thought: Collaborative Multi-Agent System for Generating Domain Specific Information 

**Authors**: Christodoulos Constantinides, Shuxin Lin, Nianjun Zhou, Dhaval Patel  

**Link**: [PDF](https://arxiv.org/pdf/2506.10086)  

**Abstract**: This paper presents a novel multi-agent system called Chat-of-Thought, designed to facilitate the generation of Failure Modes and Effects Analysis (FMEA) documents for industrial assets. Chat-of-Thought employs multiple collaborative Large Language Model (LLM)-based agents with specific roles, leveraging advanced AI techniques and dynamic task routing to optimize the generation and validation of FMEA tables. A key innovation in this system is the introduction of a Chat of Thought, where dynamic, multi-persona-driven discussions enable iterative refinement of content. This research explores the application domain of industrial equipment monitoring, highlights key challenges, and demonstrates the potential of Chat-of-Thought in addressing these challenges through interactive, template-driven workflows and context-aware agent collaboration. 

---
# ClusterUCB: Efficient Gradient-Based Data Selection for Targeted Fine-Tuning of LLMs 

**Authors**: Zige Wang, Qi Zhu, Fei Mi, Minghui Xu, Ruochun Jin, Wenjing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10288)  

**Abstract**: Gradient-based data influence approximation has been leveraged to select useful data samples in the supervised fine-tuning of large language models. However, the computation of gradients throughout the fine-tuning process requires too many resources to be feasible in practice. In this paper, we propose an efficient gradient-based data selection framework with clustering and a modified Upper Confidence Bound (UCB) algorithm. Based on the intuition that data samples with similar gradient features will have similar influences, we first perform clustering on the training data pool. Then, we frame the inter-cluster data selection as a constrained computing budget allocation problem and consider it a multi-armed bandit problem. A modified UCB algorithm is leveraged to solve this problem. Specifically, during the iterative sampling process, historical data influence information is recorded to directly estimate the distributions of each cluster, and a cold start is adopted to balance exploration and exploitation. Experimental results on various benchmarks show that our proposed framework, ClusterUCB, can achieve comparable results to the original gradient-based data selection methods while greatly reducing computing consumption. 

---
# Build the web for agents, not agents for the web 

**Authors**: Xing Han Lù, Gaurav Kamath, Marius Mosbach, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2506.10953)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and multimodal counterparts have spurred significant interest in developing web agents -- AI systems capable of autonomously navigating and completing tasks within web environments. While holding tremendous promise for automating complex web interactions, current approaches face substantial challenges due to the fundamental mismatch between human-designed interfaces and LLM capabilities. Current methods struggle with the inherent complexity of web inputs, whether processing massive DOM trees, relying on screenshots augmented with additional information, or bypassing the user interface entirely through API interactions. This position paper advocates for a paradigm shift in web agent research: rather than forcing web agents to adapt to interfaces designed for humans, we should develop a new interaction paradigm specifically optimized for agentic capabilities. To this end, we introduce the concept of an Agentic Web Interface (AWI), an interface specifically designed for agents to navigate a website. We establish six guiding principles for AWI design, emphasizing safety, efficiency, and standardization, to account for the interests of all primary stakeholders. This reframing aims to overcome fundamental limitations of existing interfaces, paving the way for more efficient, reliable, and transparent web agent design, which will be a collaborative effort involving the broader ML community. 

---
# FASCIST-O-METER: Classifier for Neo-fascist Discourse Online 

**Authors**: Rudy Alexandro Garrido Veliz, Martin Semmann, Chris Biemann, Seid Muhie Yimam  

**Link**: [PDF](https://arxiv.org/pdf/2506.10789)  

**Abstract**: Neo-fascism is a political and societal ideology that has been having remarkable growth in the last decade in the United States of America (USA), as well as in other Western societies. It poses a grave danger to democracy and the minorities it targets, and it requires active actions against it to avoid escalation. This work presents the first-of-its-kind neo-fascist coding scheme for digital discourse in the USA societal context, overseen by political science researchers. Our work bridges the gap between Natural Language Processing (NLP) and political science against this phenomena. Furthermore, to test the coding scheme, we collect a tremendous amount of activity on the internet from notable neo-fascist groups (the forums of Iron March and this http URL), and the guidelines are applied to a subset of the collected posts. Through crowdsourcing, we annotate a total of a thousand posts that are labeled as neo-fascist or non-neo-fascist. With this labeled data set, we fine-tune and test both Small Language Models (SLMs) and Large Language Models (LLMs), obtaining the very first classification models for neo-fascist discourse. We find that the prevalence of neo-fascist rhetoric in this kind of forum is ever-present, making them a good target for future research. The societal context is a key consideration for neo-fascist speech when conducting NLP research. Finally, the work against this kind of political movement must be pressed upon and continued for the well-being of a democratic society. Disclaimer: This study focuses on detecting neo-fascist content in text, similar to other hate speech analyses, without labeling individuals or organizations. 

---
# When Meaning Stays the Same, but Models Drift: Evaluating Quality of Service under Token-Level Behavioral Instability in LLMs 

**Authors**: Xiao Li, Joel Kreuzwieser, Alan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2506.10095)  

**Abstract**: We investigate how large language models respond to prompts that differ only in their token-level realization but preserve the same semantic intent, a phenomenon we call prompt variance. We propose Prompt-Based Semantic Shift (PBSS), a diagnostic framework for measuring behavioral drift in LLMs under semantically equivalent prompt rewordings. Applied to ten constrained tasks, PBSS reveals consistent, model-specific response shifts, suggesting statistical regularities linked to tokenization and decoding. These results highlight an overlooked dimension of model evaluation stability under rephrasing and suggest that tokenization strategies and decoding dynamics may contribute to post-training quality of service instability. 

---
# Can We Infer Confidential Properties of Training Data from LLMs? 

**Authors**: Penguin Huang, Chhavi Yadav, Ruihan Wu, Kamalika Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2506.10364)  

**Abstract**: Large language models (LLMs) are increasingly fine-tuned on domain-specific datasets to support applications in fields such as healthcare, finance, and law. These fine-tuning datasets often have sensitive and confidential dataset-level properties -- such as patient demographics or disease prevalence -- that are not intended to be revealed. While prior work has studied property inference attacks on discriminative models (e.g., image classification models) and generative models (e.g., GANs for image data), it remains unclear if such attacks transfer to LLMs. In this work, we introduce PropInfer, a benchmark task for evaluating property inference in LLMs under two fine-tuning paradigms: question-answering and chat-completion. Built on the ChatDoctor dataset, our benchmark includes a range of property types and task configurations. We further propose two tailored attacks: a prompt-based generation attack and a shadow-model attack leveraging word frequency signals. Empirical evaluations across multiple pretrained LLMs show the success of our attacks, revealing a previously unrecognized vulnerability in LLMs. 

---
# Provably Learning from Language Feedback 

**Authors**: Wanqiao Xu, Allen Nie, Ruijie Zheng, Aditya Modi, Adith Swaminathan, Ching-An Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.10341)  

**Abstract**: Interactively learning from observation and language feedback is an increasingly studied area driven by the emergence of large language model (LLM) agents. While impressive empirical demonstrations have been shown, so far a principled framing of these decision problems remains lacking. In this paper, we formalize the Learning from Language Feedback (LLF) problem, assert sufficient assumptions to enable learning despite latent rewards, and introduce $\textit{transfer eluder dimension}$ as a complexity measure to characterize the hardness of LLF problems. We show that transfer eluder dimension captures the intuition that information in the feedback changes the learning complexity of the LLF problem. We demonstrate cases where learning from rich language feedback can be exponentially faster than learning from reward. We develop a no-regret algorithm, called $\texttt{HELiX}$, that provably solves LLF problems through sequential interactions, with performance guarantees that scale with the transfer eluder dimension of the problem. Across several empirical domains, we show that $\texttt{HELiX}$ performs well even when repeatedly prompting LLMs does not work reliably. Our contributions mark a first step towards designing principled interactive learning algorithms from generic language feedback. 

---
# AC/DC: LLM-based Audio Comprehension via Dialogue Continuation 

**Authors**: Yusuke Fujita, Tomoya Mizumoto, Atsushi Kojima, Lianbo Liu, Yui Sudo  

**Link**: [PDF](https://arxiv.org/pdf/2506.10312)  

**Abstract**: We propose an instruction-following audio comprehension model that leverages the dialogue continuation ability of large language models (LLMs). Instead of directly generating target captions in training data, the proposed method trains a model to produce responses as if the input caption triggered a dialogue. This dialogue continuation training mitigates the caption variation problem. Learning to continue a dialogue effectively captures the caption's meaning beyond its surface-level words. As a result, our model enables zero-shot instruction-following capability without multitask instruction tuning, even trained solely on audio captioning datasets. Experiments on AudioCaps, WavCaps, and Clotho datasets with AudioBench audio-scene question-answering tests demonstrate our model's ability to follow various unseen instructions. 

---
# GenBreak: Red Teaming Text-to-Image Generators Using Large Language Models 

**Authors**: Zilong Wang, Xiang Zheng, Xiaosen Wang, Bo Wang, Xingjun Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10047)  

**Abstract**: Text-to-image (T2I) models such as Stable Diffusion have advanced rapidly and are now widely used in content creation. However, these models can be misused to generate harmful content, including nudity or violence, posing significant safety risks. While most platforms employ content moderation systems, underlying vulnerabilities can still be exploited by determined adversaries. Recent research on red-teaming and adversarial attacks against T2I models has notable limitations: some studies successfully generate highly toxic images but use adversarial prompts that are easily detected and blocked by safety filters, while others focus on bypassing safety mechanisms but fail to produce genuinely harmful outputs, neglecting the discovery of truly high-risk prompts. Consequently, there remains a lack of reliable tools for evaluating the safety of defended T2I models. To address this gap, we propose GenBreak, a framework that fine-tunes a red-team large language model (LLM) to systematically explore underlying vulnerabilities in T2I generators. Our approach combines supervised fine-tuning on curated datasets with reinforcement learning via interaction with a surrogate T2I model. By integrating multiple reward signals, we guide the LLM to craft adversarial prompts that enhance both evasion capability and image toxicity, while maintaining semantic coherence and diversity. These prompts demonstrate strong effectiveness in black-box attacks against commercial T2I generators, revealing practical and concerning safety weaknesses. 

---
# Multimodal Large Language Models: A Survey 

**Authors**: Longzhen Han, Awes Mubarak, Almas Baimagambetov, Nikolaos Polatidis, Thar Baker  

**Link**: [PDF](https://arxiv.org/pdf/2506.10016)  

**Abstract**: Multimodal Large Language Models (MLLMs) have rapidly evolved beyond text generation, now spanning diverse output modalities including images, music, video, human motion, and 3D objects, by integrating language with other sensory modalities under unified architectures. This survey categorises six primary generative modalities and examines how foundational techniques, namely Self-Supervised Learning (SSL), Mixture of Experts (MoE), Reinforcement Learning from Human Feedback (RLHF), and Chain-of-Thought (CoT) prompting, enable cross-modal capabilities. We analyze key models, architectural trends, and emergent cross-modal synergies, while highlighting transferable techniques and unresolved challenges. Architectural innovations like transformers and diffusion models underpin this convergence, enabling cross-modal transfer and modular specialization. We highlight emerging patterns of synergy, and identify open challenges in evaluation, modularity, and structured reasoning. This survey offers a unified perspective on MLLM development and identifies critical paths toward more general-purpose, adaptive, and interpretable multimodal systems. 

---
# Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models 

**Authors**: Elena Sofia Ruzzetti, Giancarlo A. Xompero, Davide Venditti, Fabio Massimo Zanzotto  

**Link**: [PDF](https://arxiv.org/pdf/2506.10024)  

**Abstract**: Large Language Models (LLMs) memorize, and thus, among huge amounts of uncontrolled data, may memorize Personally Identifiable Information (PII), which should not be stored and, consequently, not leaked. In this paper, we introduce Private Memorization Editing (PME), an approach for preventing private data leakage that turns an apparent limitation, that is, the LLMs' memorization ability, into a powerful privacy defense strategy. While attacks against LLMs have been performed exploiting previous knowledge regarding their training data, our approach aims to exploit the same kind of knowledge in order to make a model more robust. We detect a memorized PII and then mitigate the memorization of PII by editing a model knowledge of its training data. We verify that our procedure does not affect the underlying language model while making it more robust against privacy Training Data Extraction attacks. We demonstrate that PME can effectively reduce the number of leaked PII in a number of configurations, in some cases even reducing the accuracy of the privacy attacks to zero. 

---
