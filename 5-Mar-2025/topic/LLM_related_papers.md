# Hierarchical Re-ranker Retriever (HRR) 

**Authors**: Ashish Singh, Priti Mohapatra  

**Link**: [PDF](https://arxiv.org/pdf/2503.02401)  

**Abstract**: Retrieving the right level of context for a given query is a perennial challenge in information retrieval - too large a chunk dilutes semantic specificity, while chunks that are too small lack broader context. This paper introduces the Hierarchical Re-ranker Retriever (HRR), a framework designed to achieve both fine-grained and high-level context retrieval for large language model (LLM) applications. In HRR, documents are split into sentence-level and intermediate-level (512 tokens) chunks to maximize vector-search quality for both short and broad queries. We then employ a reranker that operates on these 512-token chunks, ensuring an optimal balance neither too coarse nor too fine for robust relevance scoring. Finally, top-ranked intermediate chunks are mapped to parent chunks (2048 tokens) to provide an LLM with sufficiently large context. 

---
# Towards Explainable Doctor Recommendation with Large Language Models 

**Authors**: Ziyang Zeng, Dongyuan Li, Yuqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02298)  

**Abstract**: The advent of internet medicine provides patients with unprecedented convenience in searching and communicating with doctors relevant to their diseases and desired treatments online. However, the current doctor recommendation systems fail to fully ensure the professionalism and interpretability of the recommended results. In this work, we formulate doctor recommendation as a ranking task and develop a large language model (LLM)-based pointwise ranking framework. Our framework ranks doctors according to their relevance regarding specific diseases-treatment pairs in a zero-shot setting. The advantage of our framework lies in its ability to generate precise and explainable doctor ranking results. Additionally, we construct DrRank, a new expertise-driven doctor ranking dataset comprising over 38 disease-treatment pairs. Experiment results on the DrRank dataset demonstrate that our framework significantly outperforms the strongest cross-encoder baseline, achieving a notable gain of +5.45 in the NDCG@10 score while maintaining affordable latency consumption. Furthermore, we comprehensively present the fairness analysis results of our framework from three perspectives of different diseases, patient gender, and geographical regions. Meanwhile, the interpretability of our framework is rigorously verified by three human experts, providing further evidence of the reliability of our proposed framework for doctor recommendation. 

---
# Zero-Shot Complex Question-Answering on Long Scientific Documents 

**Authors**: Wanting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02695)  

**Abstract**: With the rapid development in Transformer-based language models, the reading comprehension tasks on short documents and simple questions have been largely addressed. Long documents, specifically the scientific documents that are densely packed with knowledge discovered and developed by humans, remain relatively unexplored. These documents often come with a set of complex and more realistic questions, adding to their complexity. We present a zero-shot pipeline framework that enables social science researchers to perform question-answering tasks that are complex yet of predetermined question formats on full-length research papers without requiring machine learning expertise. Our approach integrates pre-trained language models to handle challenging scenarios including multi-span extraction, multi-hop reasoning, and long-answer generation. Evaluating on MLPsych, a novel dataset of social psychology papers with annotated complex questions, we demonstrate that our framework achieves strong performance through combination of extractive and generative models. This work advances document understanding capabilities for social sciences while providing practical tools for researchers. 

---
# PersonaX: A Recommendation Agent Oriented User Modeling Framework for Long Behavior Sequence 

**Authors**: Yunxiao Shi, Wujiang Xu, Zeqi Zhang, Xing Zi, Qiang Wu, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02398)  

**Abstract**: Recommendation agents leverage large language models for user modeling LLM UM to construct textual personas guiding alignment with real users. However existing LLM UM methods struggle with long user generated content UGC due to context limitations and performance degradation. To address this sampling strategies prioritize relevance or recency are often applied yet they inevitably neglect the diverse user interests embedded within the discarded behaviors resulting in incomplete modeling and degraded profiling quality. Furthermore relevance based sampling requires real time retrieval forcing the user modeling process to operate online which introduces significant latency overhead. In this paper we propose PersonaX an agent agnostic LLM UM framework that tackles these challenges through sub behavior sequence SBS selection and offline multi persona construction. PersonaX extracts compact SBS segments offline to capture diverse user interests generating fine grained textual personas that are cached for efficient online retrieval. This approach ensures that the user persona used for prompting remains highly relevant to the current context while eliminating the need for online user modeling. For SBS selection we ensure both efficiency length less than five and high representational quality by balancing prototypicality and diversity within the sampled data. Extensive experiments validate the effectiveness and versatility of PersonaX in high quality user profiling. Utilizing only 30 to 50 percent of the behavioral data with a sequence length of 480 integrating PersonaX with AgentCF yields an absolute performance improvement of 3 to 11 percent while integration with Agent4Rec results in a gain of 10 to 50 percent. PersonaX as an agent agnostic framework sets a new benchmark for scalable user modeling paving the way for more accurate and efficient LLM driven recommendation agents. 

---
# MindBridge: Scalable and Cross-Model Knowledge Editing via Memory-Augmented Modality 

**Authors**: Shuaike Li, Kai Zhang, Qi Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.02701)  

**Abstract**: Knowledge editing is a technique for efficiently and accurately updating the knowledge of large language models (LLMs) to alleviate obsolescence and correct errors. However, most existing methods overfit to specific models, causing edited knowledge to be discarded during each LLM update and requiring frequent re-editing, which is particularly burdensome in today's rapidly evolving open-source community. To address this issue, we propose the problem of cross-model knowledge editing and introduce MindBridge, a scalable solution inspired by the low coupling between modality processing and LLMs in multi-modal models. MindBridge introduces the novel concept of memory modality, which encodes edited knowledge as an independent modality. It first performs LLM-agnostic pre-training of the memory modality and then integrates it with various LLMs. Extensive experiments on multiple LLMs and popular knowledge editing datasets demonstrate that MindBridge achieves superior performance even in editing tens of thousands of knowledge entries and can flexibly adapt to different LLMs. Our code is available at this https URL. 

---
# Playing games with Large language models: Randomness and strategy 

**Authors**: Alicia Vidler, Toby Walsh  

**Link**: [PDF](https://arxiv.org/pdf/2503.02582)  

**Abstract**: Playing games has a long history of describing intricate interactions in simplified forms. In this paper we explore if large language models (LLMs) can play games, investigating their capabilities for randomisation and strategic adaptation through both simultaneous and sequential game interactions. We focus on GPT-4o-Mini-2024-08-17 and test two games between LLMs: Rock Paper Scissors (RPS) and games of strategy (Prisoners Dilemma PD). LLMs are often described as stochastic parrots, and while they may indeed be parrots, our results suggest that they are not very stochastic in the sense that their outputs - when prompted to be random - are often very biased. Our research reveals that LLMs appear to develop loss aversion strategies in repeated games, with RPS converging to stalemate conditions while PD shows systematic shifts between cooperative and competitive outcomes based on prompt design. We detail programmatic tools for independent agent interactions and the Agentic AI challenges faced in implementation. We show that LLMs can indeed play games, just not very well. These results have implications for the use of LLMs in multi-agent LLM systems and showcase limitations in current approaches to model output for strategic decision-making. 

---
# The Effectiveness of Large Language Models in Transforming Unstructured Text to Standardized Formats 

**Authors**: William Brach, Kristián Košťál, Michal Ries  

**Link**: [PDF](https://arxiv.org/pdf/2503.02650)  

**Abstract**: The exponential growth of unstructured text data presents a fundamental challenge in modern data management and information retrieval. While Large Language Models (LLMs) have shown remarkable capabilities in natural language processing, their potential to transform unstructured text into standardized, structured formats remains largely unexplored - a capability that could revolutionize data processing workflows across industries. This study breaks new ground by systematically evaluating LLMs' ability to convert unstructured recipe text into the structured Cooklang format. Through comprehensive testing of four models (GPT-4o, GPT-4o-mini, Llama3.1:70b, and Llama3.1:8b), an innovative evaluation approach is introduced that combines traditional metrics (WER, ROUGE-L, TER) with specialized metrics for semantic element identification. Our experiments reveal that GPT-4o with few-shot prompting achieves breakthrough performance (ROUGE-L: 0.9722, WER: 0.0730), demonstrating for the first time that LLMs can reliably transform domain-specific unstructured text into structured formats without extensive training. Although model performance generally scales with size, we uncover surprising potential in smaller models like Llama3.1:8b for optimization through targeted fine-tuning. These findings open new possibilities for automated structured data generation across various domains, from medical records to technical documentation, potentially transforming the way organizations process and utilize unstructured information. 

---
# Memorize or Generalize? Evaluating LLM Code Generation with Evolved Questions 

**Authors**: Wentao Chen, Lizhe Zhang, Li Zhong, Letian Peng, Zilong Wang, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02296)  

**Abstract**: Large Language Models (LLMs) are known to exhibit a memorization phenomenon in code generation: instead of truly understanding the underlying principles of a programming problem, they tend to memorize the original prompt and its solution together in the training. Consequently, when facing variants of the original problem, their answers very likely resemble the memorized solutions and fail to generalize. In this paper, we investigate this phenomenon by designing three evolution strategies to create variants: mutation, paraphrasing, and code-rewriting. By comparing the performance and AST similarity of the LLM-generated codes before and after these three evolutions, we develop a memorization score that positively correlates with the level of memorization. As expected, as supervised fine-tuning goes on, the memorization score rises before overfitting, suggesting more severe memorization. We demonstrate that common mitigation approaches, such as prompt translation and using evolved variants as data augmentation in supervised learning and reinforcement learning, either compromise the performance or fail to alleviate the memorization issue. Therefore, memorization remains a significant challenge in LLM code generation, highlighting the need for a more effective solution. 

---
# AppAgentX: Evolving GUI Agents as Proficient Smartphone Users 

**Authors**: Wenjia Jiang, Yangyang Zhuang, Chenxi Song, Xu Yang, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02268)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have led to the development of intelligent LLM-based agents capable of interacting with graphical user interfaces (GUIs). These agents demonstrate strong reasoning and adaptability, enabling them to perform complex tasks that traditionally required predefined rules. However, the reliance on step-by-step reasoning in LLM-based agents often results in inefficiencies, particularly for routine tasks. In contrast, traditional rule-based systems excel in efficiency but lack the intelligence and flexibility to adapt to novel scenarios. To address this challenge, we propose a novel evolutionary framework for GUI agents that enhances operational efficiency while retaining intelligence and flexibility. Our approach incorporates a memory mechanism that records the agent's task execution history. By analyzing this history, the agent identifies repetitive action sequences and evolves high-level actions that act as shortcuts, replacing these low-level operations and improving efficiency. This allows the agent to focus on tasks requiring more complex reasoning, while simplifying routine actions. Experimental results on multiple benchmark tasks demonstrate that our approach significantly outperforms existing methods in both efficiency and accuracy. The code will be open-sourced to support further research. 

---
# V2X-LLM: Enhancing V2X Integration and Understanding in Connected Vehicle Corridors 

**Authors**: Keshu Wu, Pei Li, Yang Zhou, Rui Gan, Junwei You, Yang Cheng, Jingwen Zhu, Steven T. Parker, Bin Ran, David A. Noyce, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02239)  

**Abstract**: The advancement of Connected and Automated Vehicles (CAVs) and Vehicle-to-Everything (V2X) offers significant potential for enhancing transportation safety, mobility, and sustainability. However, the integration and analysis of the diverse and voluminous V2X data, including Basic Safety Messages (BSMs) and Signal Phase and Timing (SPaT) data, present substantial challenges, especially on Connected Vehicle Corridors. These challenges include managing large data volumes, ensuring real-time data integration, and understanding complex traffic scenarios. Although these projects have developed an advanced CAV data pipeline that enables real-time communication between vehicles, infrastructure, and other road users for managing connected vehicle and roadside unit (RSU) data, significant hurdles in data comprehension and real-time scenario analysis and reasoning persist. To address these issues, we introduce the V2X-LLM framework, a novel enhancement to the existing CV data pipeline. V2X-LLM leverages Large Language Models (LLMs) to improve the understanding and real-time analysis of V2X data. The framework includes four key tasks: Scenario Explanation, offering detailed narratives of traffic conditions; V2X Data Description, detailing vehicle and infrastructure statuses; State Prediction, forecasting future traffic states; and Navigation Advisory, providing optimized routing instructions. By integrating LLM-driven reasoning with V2X data within the data pipeline, the V2X-LLM framework offers real-time feedback and decision support for traffic management. This integration enhances the accuracy of traffic analysis, safety, and traffic optimization. Demonstrations in a real-world urban corridor highlight the framework's potential to advance intelligent transportation systems. 

---
# Don't Get Too Excited -- Eliciting Emotions in LLMs 

**Authors**: Gino Franco Fazzi, Julie Skoven Hinge, Stefan Heinrich, Paolo Burelli  

**Link**: [PDF](https://arxiv.org/pdf/2503.02457)  

**Abstract**: This paper investigates the challenges of affect control in large language models (LLMs), focusing on their ability to express appropriate emotional states during extended dialogues. We evaluated state-of-the-art open-weight LLMs to assess their affective expressive range in terms of arousal and valence. Our study employs a novel methodology combining LLM-based sentiment analysis with multiturn dialogue simulations between LLMs. We quantify the models' capacity to express a wide spectrum of emotions and how they fluctuate during interactions. Our findings reveal significant variations among LLMs in their ability to maintain consistent affect, with some models demonstrating more stable emotional trajectories than others. Furthermore, we identify key challenges in affect control, including difficulties in producing and maintaining extreme emotional states and limitations in adapting affect to changing conversational contexts. These findings have important implications for the development of more emotionally intelligent AI systems and highlight the need for improved affect modelling in LLMs. 

---
# TMIQ: Quantifying Test and Measurement Domain Intelligence in Large Language Models 

**Authors**: Emmanuel A. Olowe, Danial Chitnis  

**Link**: [PDF](https://arxiv.org/pdf/2503.02123)  

**Abstract**: The Test and Measurement domain, known for its strict requirements for accuracy and efficiency, is increasingly adopting Generative AI technologies to enhance the performance of data analysis, automation, and decision-making processes. Among these, Large Language Models (LLMs) show significant promise for advancing automation and precision in testing. However, the evaluation of LLMs in this specialized area remains insufficiently explored. To address this gap, we introduce the Test and Measurement Intelligence Quotient (TMIQ), a benchmark designed to quantitatively assess LLMs across a wide range of electronic engineering tasks. TMIQ offers a comprehensive set of scenarios and metrics for detailed evaluation, including SCPI command matching accuracy, ranked response evaluation, Chain-of-Thought Reasoning (CoT), and the impact of output formatting variations required by LLMs on performance. In testing various LLMs, our findings indicate varying levels of proficiency, with exact SCPI command match accuracy ranging from around 56% to 73%, and ranked matching first-position scores achieving around 33% for the best-performing model. We also assess token usage, cost-efficiency, and response times, identifying trade-offs between accuracy and operational efficiency. Additionally, we present a command-line interface (CLI) tool that enables users to generate datasets using the same methodology, allowing for tailored assessments of LLMs. TMIQ and the CLI tool provide a rigorous, reproducible means of evaluating LLMs for production environments, facilitating continuous monitoring and identifying strengths and areas for improvement, and driving innovation in their selections for applications within the Test and Measurement industry. 

---
# Wikipedia in the Era of LLMs: Evolution and Risks 

**Authors**: Siming Huang, Yuliang Xu, Mingmeng Geng, Yao Wan, Dongping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.02879)  

**Abstract**: In this paper, we present a thorough analysis of the impact of Large Language Models (LLMs) on Wikipedia, examining the evolution of Wikipedia through existing data and using simulations to explore potential risks. We begin by analyzing page views and article content to study Wikipedia's recent changes and assess the impact of LLMs. Subsequently, we evaluate how LLMs affect various Natural Language Processing (NLP) tasks related to Wikipedia, including machine translation and retrieval-augmented generation (RAG). Our findings and simulation results reveal that Wikipedia articles have been influenced by LLMs, with an impact of approximately 1%-2% in certain categories. If the machine translation benchmark based on Wikipedia is influenced by LLMs, the scores of the models may become inflated, and the comparative results among models might shift as well. Moreover, the effectiveness of RAG might decrease if the knowledge base becomes polluted by LLM-generated content. While LLMs have not yet fully changed Wikipedia's language and knowledge structures, we believe that our empirical findings signal the need for careful consideration of potential future risks. 

---
# AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation 

**Authors**: Songming Zhang, Xue Zhang, Tong Zhang, Bojie Hu, Yufeng Chen, Jinan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02832)  

**Abstract**: In modern large language models (LLMs), LLM alignment is of crucial importance and is typically achieved through methods such as reinforcement learning from human feedback (RLHF) and direct preference optimization (DPO). However, in most existing methods for LLM alignment, all tokens in the response are optimized using a sparse, response-level reward or preference annotation. The ignorance of token-level rewards may erroneously punish high-quality tokens or encourage low-quality tokens, resulting in suboptimal performance and slow convergence speed. To address this issue, we propose AlignDistil, an RLHF-equivalent distillation method for token-level reward optimization. Specifically, we introduce the reward learned by DPO into the RLHF objective and theoretically prove the equivalence between this objective and a token-level distillation process, where the teacher distribution linearly combines the logits from the DPO model and a reference model. On this basis, we further bridge the accuracy gap between the reward from the DPO model and the pure reward model, by building a contrastive DPO reward with a normal and a reverse DPO model. Moreover, to avoid under- and over-optimization on different tokens, we design a token adaptive logit extrapolation mechanism to construct an appropriate teacher distribution for each token. Experimental results demonstrate the superiority of our AlignDistil over existing methods and showcase fast convergence due to its token-level distributional reward optimization. 

---
# Language Models can Self-Improve at State-Value Estimation for Better Search 

**Authors**: Ethan Mendes, Alan Ritter  

**Link**: [PDF](https://arxiv.org/pdf/2503.02878)  

**Abstract**: Collecting ground truth task completion rewards or human demonstrations for multi-step reasoning tasks is often cost-prohibitive and time-consuming, especially in interactive domains like web tasks. To address this bottleneck, we present self-taught lookahead, a self-supervised method that leverages state-transition dynamics to train a value model capable of effectively guiding language model-controlled search. We find that moderately sized (8 billion parameters) open-weight value models improved with self-taught lookahead can match the performance of using a frontier LLM such as gpt-4o as the value model. Furthermore, we find that self-taught lookahead improves performance by 20% while reducing costs 37x compared to previous LLM-based tree search, without relying on ground truth rewards. 

---
# IterPref: Focal Preference Learning for Code Generation via Iterative Debugging 

**Authors**: Jie Wu, Haoling Li, Xin Zhang, Jianwen Luo, Yangyu Huang, Ruihang Chu, Yujiu Yang, Scarlett Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.02783)  

**Abstract**: Preference learning enhances Code LLMs beyond supervised fine-tuning by leveraging relative quality comparisons. Existing methods construct preference pairs from
candidates based on test case success, treating the higher pass rate sample as positive and the lower as negative. However, this approach does not pinpoint specific errors in the code, which prevents the model from learning more informative error correction patterns, as aligning failing code as a whole lacks the granularity needed to capture meaningful error-resolution relationships. To address these issues, we propose IterPref, a new preference alignment framework that mimics human iterative debugging to refine Code LLMs. IterPref explicitly locates error regions and aligns the corresponding tokens via a tailored DPO algorithm. To generate informative pairs, we introduce the CodeFlow dataset, where samples are iteratively refined until passing tests, with modifications capturing error corrections. Extensive experiments show that a diverse suite of Code LLMs equipped with IterPref achieves significant performance gains in code generation and improves on challenging tasks like BigCodeBench. In-depth analysis reveals that IterPref yields fewer errors. Our code and data will be made publicaly available. 

---
# MPO: Boosting LLM Agents with Meta Plan Optimization 

**Authors**: Weimin Xiong, Yifan Song, Qingxiu Dong, Bingchan Zhao, Feifan Song, Xun Wang, Sujian Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.02682)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled LLM-based agents to successfully tackle interactive planning tasks. However, despite their successes, existing approaches often suffer from planning hallucinations and require retraining for each new agent. To address these challenges, we propose the Meta Plan Optimization (MPO) framework, which enhances agent planning capabilities by directly incorporating explicit guidance. Unlike previous methods that rely on complex knowledge, which either require significant human effort or lack quality assurance, MPO leverages high-level general guidance through meta plans to assist agent planning and enables continuous optimization of the meta plans based on feedback from the agent's task execution. Our experiments conducted on two representative tasks demonstrate that MPO significantly outperforms existing baselines. Moreover, our analysis indicates that MPO provides a plug-and-play solution that enhances both task completion efficiency and generalization capabilities in previous unseen scenarios. 

---
# Towards Event Extraction with Massive Types: LLM-based Collaborative Annotation and Partitioning Extraction 

**Authors**: Wenxuan Liu, Zixuan Li, Long Bai, Yuxin Zuo, Daozhu Xu, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.02628)  

**Abstract**: Developing a general-purpose extraction system that can extract events with massive types is a long-standing target in Event Extraction (EE). In doing so, the challenge comes from two aspects: 1) The absence of an efficient and effective annotation method. 2) The absence of a powerful extraction method can handle massive types. For the first challenge, we propose a collaborative annotation method based on Large Language Models (LLMs). Through collaboration among multiple LLMs, it first refines annotations of trigger words from distant supervision and then carries out argument annotation. Next, a voting phase consolidates the annotation preferences across different LLMs. Finally, we create the EEMT dataset, the largest EE dataset to date, featuring over 200,000 samples, 3,465 event types, and 6,297 role types. For the second challenge, we propose an LLM-based Partitioning EE method called LLM-PEE. To overcome the limited context length of LLMs, LLM-PEE first recalls candidate event types and then splits them into multiple partitions for LLMs to extract events. The results in the supervised setting show that LLM-PEE outperforms the state-of-the-art methods by 5.4 in event detection and 6.1 in argument extraction. In the zero-shot setting, LLM-PEE achieves up to 12.9 improvement compared to mainstream LLMs, demonstrating its strong generalization capabilities. 

---
# Reflection on Data Storytelling Tools in the Generative AI Era from the Human-AI Collaboration Perspective 

**Authors**: Haotian Li, Yun Wang, Huamin Qu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02631)  

**Abstract**: Human-AI collaborative tools attract attentions from the data storytelling community to lower the barrier of expertise and streamline the workflow. The recent advance in large-scale generative AI techniques, e.g., large language models (LLMs) and text-to-image models, has the potential to enhance data storytelling with their power in visual and narration generation. After two years since these techniques were publicly available, it is important to reflect our progress of applying them and have an outlook for future opportunities. To achieve the goal, we compare the collaboration patterns of the latest tools with those of earlier ones using a dedicated framework for understanding human-AI collaboration in data storytelling. Through comparison, we identify persistent collaboration patterns, e.g., human-creator + AI-assistant, and emerging ones, e.g., AI-creator + human-reviewer. The benefits of these AI techniques and other implications to human-AI collaboration are also revealed. We further propose future directions to hopefully ignite innovations. 

---
# Rewarding Doubt: A Reinforcement Learning Approach to Confidence Calibration of Large Language Models 

**Authors**: Paul Stangel, David Bani-Harouni, Chantal Pellegrini, Ege Özsoy, Kamilia Zaripova, Matthias Keicher, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2503.02623)  

**Abstract**: A safe and trustworthy use of Large Language Models (LLMs) requires an accurate expression of confidence in their answers. We introduce a novel Reinforcement Learning (RL) approach for LLM calibration that fine-tunes LLMs to elicit calibrated confidence estimations in their answers to factual questions. We model the problem as a betting game where the model predicts a confidence score together with every answer, and design a reward function that penalizes both over and under-confidence. We prove that under our reward design an optimal policy would result in a perfectly calibrated confidence estimation. Our experiments demonstrate significantly improved confidence calibration and generalization to new tasks without re-training, indicating that our approach teaches a general confidence awareness. This approach enables the training of inherently calibrated LLMs. 

---
# LLM-Safety Evaluations Lack Robustness 

**Authors**: Tim Beyer, Sophie Xhonneux, Simon Geisler, Gauthier Gidel, Leo Schwinn, Stephan Günnemann  

**Link**: [PDF](https://arxiv.org/pdf/2503.02574)  

**Abstract**: In this paper, we argue that current safety alignment research efforts for large language models are hindered by many intertwined sources of noise, such as small datasets, methodological inconsistencies, and unreliable evaluation setups. This can, at times, make it impossible to evaluate and compare attacks and defenses fairly, thereby slowing progress. We systematically analyze the LLM safety evaluation pipeline, covering dataset curation, optimization strategies for automated red-teaming, response generation, and response evaluation using LLM judges. At each stage, we identify key issues and highlight their practical impact. We also propose a set of guidelines for reducing noise and bias in evaluations of future attack and defense papers. Lastly, we offer an opposing perspective, highlighting practical reasons for existing limitations. We believe that addressing the outlined problems in future research will improve the field's ability to generate easily comparable results and make measurable progress. 

---
# An Efficient and Precise Training Data Construction Framework for Process-supervised Reward Model in Mathematical Reasoning 

**Authors**: Wei Sun, Qianlong Du, Fuwei Cui, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02382)  

**Abstract**: Enhancing the mathematical reasoning capabilities of Large Language Models (LLMs) is of great scientific and practical significance. Researchers typically employ process-supervised reward models (PRMs) to guide the reasoning process, effectively improving the models' reasoning abilities. However, existing methods for constructing process supervision training data, such as manual annotation and per-step Monte Carlo estimation, are often costly or suffer from poor quality. To address these challenges, this paper introduces a framework called EpicPRM, which annotates each intermediate reasoning step based on its quantified contribution and uses an adaptive binary search algorithm to enhance both annotation precision and efficiency. Using this approach, we efficiently construct a high-quality process supervision training dataset named Epic50k, consisting of 50k annotated intermediate steps. Compared to other publicly available datasets, the PRM trained on Epic50k demonstrates significantly superior performance. Getting Epic50k at this https URL. 

---
# Large Language Models as Natural Selector for Embodied Soft Robot Design 

**Authors**: Changhe Chen, Xiaohao Xu, Xiangdong Wang, Xiaonan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02249)  

**Abstract**: Designing soft robots is a complex and iterative process that demands cross-disciplinary expertise in materials science, mechanics, and control, often relying on intuition and extensive experimentation. While Large Language Models (LLMs) have demonstrated impressive reasoning abilities, their capacity to learn and apply embodied design principles--crucial for creating functional robotic systems--remains largely unexplored. This paper introduces RoboCrafter-QA, a novel benchmark to evaluate whether LLMs can learn representations of soft robot designs that effectively bridge the gap between high-level task descriptions and low-level morphological and material choices. RoboCrafter-QA leverages the EvoGym simulator to generate a diverse set of soft robot design challenges, spanning robotic locomotion, manipulation, and balancing tasks. Our experiments with state-of-the-art multi-modal LLMs reveal that while these models exhibit promising capabilities in learning design representations, they struggle with fine-grained distinctions between designs with subtle performance differences. We further demonstrate the practical utility of LLMs for robot design initialization. Our code and benchmark will be available to encourage the community to foster this exciting research direction. 

---
# Enhancing LLM Reliability via Explicit Knowledge Boundary Modeling 

**Authors**: Hang Zheng, Hongshen Xu, Yuncong Liu, Lu Chen, Pascale Fung, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02233)  

**Abstract**: Large language models (LLMs) frequently hallucinate due to misaligned self-awareness, generating erroneous outputs when addressing queries beyond their knowledge boundaries. While existing approaches mitigate hallucinations via uncertainty estimation or query rejection, they suffer from computational inefficiency or sacrificed helpfulness. To address these issues, we propose the Explicit Knowledge Boundary Modeling (EKBM) framework, integrating fast and slow reasoning systems to harmonize reliability and usability. The framework first employs a fast-thinking model to generate confidence-labeled responses, enabling immediate use of high-confidence outputs. For uncertain predictions, a slow refinement model conducts targeted reasoning to improve accuracy. To align model behavior with our proposed object, we propose a hybrid training pipeline, enhancing self-awareness without degrading task performance. Evaluations on dialogue state tracking tasks demonstrate that EKBM achieves superior model reliability over uncertainty-based baselines. Further analysis reveals that refinement substantially boosts accuracy while maintaining low computational overhead. Our work establishes a scalable paradigm for advancing LLM reliability and balancing accuracy and practical utility in error-sensitive applications. 

---
# Audio-Reasoner: Improving Reasoning Capability in Large Audio Language Models 

**Authors**: Zhifei Xie, Mingbao Lin, Zihang Liu, Pengcheng Wu, Shuicheng Yan, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2503.02318)  

**Abstract**: Recent advancements in multimodal reasoning have largely overlooked the audio modality. We introduce Audio-Reasoner, a large-scale audio language model for deep reasoning in audio tasks. We meticulously curated a large-scale and diverse multi-task audio dataset with simple annotations. Then, we leverage closed-source models to conduct secondary labeling, QA generation, along with structured COT process. These datasets together form a high-quality reasoning dataset with 1.2 million reasoning-rich samples, which we name CoTA. Following inference scaling principles, we train Audio-Reasoner on CoTA, enabling it to achieve great logical capabilities in audio reasoning. Experiments show state-of-the-art performance across key benchmarks, including MMAU-mini (+25.42%), AIR-Bench chat/foundation(+14.57%/+10.13%), and MELD (+8.01%). Our findings stress the core of structured CoT training in advancing audio reasoning. 

---
# ATLaS: Agent Tuning via Learning Critical Steps 

**Authors**: Zhixun Chen, Ming Li, Yuxuan Huang, Yali Du, Meng Fang, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.02197)  

**Abstract**: Large Language Model (LLM) agents have demonstrated remarkable generalization capabilities across multi-domain tasks. Existing agent tuning approaches typically employ supervised finetuning on entire expert trajectories. However, behavior-cloning of full trajectories can introduce expert bias and weaken generalization to states not covered by the expert data. Additionally, critical steps, such as planning, complex reasoning for intermediate subtasks, and strategic decision-making, are essential to success in agent tasks, so learning these steps is the key to improving LLM agents. For more effective and efficient agent tuning, we propose ATLaS that identifies the critical steps in expert trajectories and finetunes LLMs solely on these steps with reduced costs. By steering the training's focus to a few critical steps, our method mitigates the risk of overfitting entire trajectories and promotes generalization across different environments and tasks. In extensive experiments, an LLM finetuned on only 30% critical steps selected by ATLaS outperforms the LLM finetuned on all steps and recent open-source LLM agents. ATLaS maintains and improves base LLM skills as generalist agents interacting with diverse environments. 

---
# Adversarial Tokenization 

**Authors**: Renato Lui Geh, Zilei Shao, Guy Van den Broeck  

**Link**: [PDF](https://arxiv.org/pdf/2503.02174)  

**Abstract**: Current LLM pipelines account for only one possible tokenization for a given string, ignoring exponentially many alternative tokenizations during training and inference. For example, the standard Llama3 tokenization of penguin is [p,enguin], yet [peng,uin] is another perfectly valid alternative. In this paper, we show that despite LLMs being trained solely on one tokenization, they still retain semantic understanding of other tokenizations, raising questions about their implications in LLM safety. Put succinctly, we answer the following question: can we adversarially tokenize an obviously malicious string to evade safety and alignment restrictions? We show that not only is adversarial tokenization an effective yet previously neglected axis of attack, but it is also competitive against existing state-of-the-art adversarial approaches without changing the text of the harmful request. We empirically validate this exploit across three state-of-the-art LLMs and adversarial datasets, revealing a previously unknown vulnerability in subword models. 

---
# Linear Representations of Political Perspective Emerge in Large Language Models 

**Authors**: Junsol Kim, James Evans, Aaron Schein  

**Link**: [PDF](https://arxiv.org/pdf/2503.02080)  

**Abstract**: Large language models (LLMs) have demonstrated the ability to generate text that realistically reflects a range of different subjective human perspectives. This paper studies how LLMs are seemingly able to reflect more liberal versus more conservative viewpoints among other political perspectives in American politics. We show that LLMs possess linear representations of political perspectives within activation space, wherein more similar perspectives are represented closer together. To do so, we probe the attention heads across the layers of three open transformer-based LLMs (\texttt{Llama-2-7b-chat}, \texttt{Mistral-7b-instruct}, \texttt{Vicuna-7b}). We first prompt models to generate text from the perspectives of different U.S.~lawmakers. We then identify sets of attention heads whose activations linearly predict those lawmakers' DW-NOMINATE scores, a widely-used and validated measure of political ideology. We find that highly predictive heads are primarily located in the middle layers, often speculated to encode high-level concepts and tasks. Using probes only trained to predict lawmakers' ideology, we then show that the same probes can predict measures of news outlets' slant from the activations of models prompted to simulate text from those news outlets. These linear probes allow us to visualize, interpret, and monitor ideological stances implicitly adopted by an LLM as it generates open-ended responses. Finally, we demonstrate that by applying linear interventions to these attention heads, we can steer the model outputs toward a more liberal or conservative stance. Overall, our research suggests that LLMs possess a high-level linear representation of American political ideology and that by leveraging recent advances in mechanistic interpretability, we can identify, monitor, and steer the subjective perspective underlying generated text. 

---
# Superscopes: Amplifying Internal Feature Representations for Language Model Interpretation 

**Authors**: Jonathan Jacobi, Gal Niv  

**Link**: [PDF](https://arxiv.org/pdf/2503.02078)  

**Abstract**: Understanding and interpreting the internal representations of large language models (LLMs) remains an open challenge. Patchscopes introduced a method for probing internal activations by patching them into new prompts, prompting models to self-explain their hidden representations. We introduce Superscopes, a technique that systematically amplifies superposed features in MLP outputs (multilayer perceptron) and hidden states before patching them into new contexts. Inspired by the "features as directions" perspective and the Classifier-Free Guidance (CFG) approach from diffusion models, Superscopes amplifies weak but meaningful features, enabling the interpretation of internal representations that previous methods failed to explain-all without requiring additional training. This approach provides new insights into how LLMs build context and represent complex concepts, further advancing mechanistic interpretability. 

---
# LLMs as Educational Analysts: Transforming Multimodal Data Traces into Actionable Reading Assessment Reports 

**Authors**: Eduardo Davalos, Yike Zhang, Namrata Srivastava, Jorge Alberto Salas, Sara McFadden, Sun-Joo Cho, Gautam Biswas, Amanda Goodwin  

**Link**: [PDF](https://arxiv.org/pdf/2503.02099)  

**Abstract**: Reading assessments are essential for enhancing students' comprehension, yet many EdTech applications focus mainly on outcome-based metrics, providing limited insights into student behavior and cognition. This study investigates the use of multimodal data sources -- including eye-tracking data, learning outcomes, assessment content, and teaching standards -- to derive meaningful reading insights. We employ unsupervised learning techniques to identify distinct reading behavior patterns, and then a large language model (LLM) synthesizes the derived information into actionable reports for educators, streamlining the interpretation process. LLM experts and human educators evaluate these reports for clarity, accuracy, relevance, and pedagogical usefulness. Our findings indicate that LLMs can effectively function as educational analysts, turning diverse data into teacher-friendly insights that are well-received by educators. While promising for automating insight generation, human oversight remains crucial to ensure reliability and fairness. This research advances human-centered AI in education, connecting data-driven analytics with practical classroom applications. 

---
# Mind the (Belief) Gap: Group Identity in the World of LLMs 

**Authors**: Angana Borah, Marwa Houalla, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2503.02016)  

**Abstract**: Social biases and belief-driven behaviors can significantly impact Large Language Models (LLMs) decisions on several tasks. As LLMs are increasingly used in multi-agent systems for societal simulations, their ability to model fundamental group psychological characteristics remains critical yet under-explored. In this study, we present a multi-agent framework that simulates belief congruence, a classical group psychology theory that plays a crucial role in shaping societal interactions and preferences. Our findings reveal that LLMs exhibit amplified belief congruence compared to humans, across diverse contexts. We further investigate the implications of this behavior on two downstream tasks: (1) misinformation dissemination and (2) LLM learning, finding that belief congruence in LLMs increases misinformation dissemination and impedes learning. To mitigate these negative impacts, we propose strategies inspired by: (1) contact hypothesis, (2) accuracy nudges, and (3) global citizenship framework. Our results show that the best strategies reduce misinformation dissemination by up to 37% and enhance learning by 11%. Bridging social psychology and AI, our work provides insights to navigate real-world interactions using LLMs while addressing belief-driven biases. 

---
# AI persuading AI vs AI persuading Humans: LLMs' Differential Effectiveness in Promoting Pro-Environmental Behavior 

**Authors**: Alexander Doudkin, Pat Pataranutaporn, Pattie Maes  

**Link**: [PDF](https://arxiv.org/pdf/2503.02067)  

**Abstract**: Pro-environmental behavior (PEB) is vital to combat climate change, yet turning awareness into intention and action remains elusive. We explore large language models (LLMs) as tools to promote PEB, comparing their impact across 3,200 participants: real humans (n=1,200), simulated humans based on actual participant data (n=1,200), and fully synthetic personas (n=1,200). All three participant groups faced personalized or standard chatbots, or static statements, employing four persuasion strategies (moral foundations, future self-continuity, action orientation, or "freestyle" chosen by the LLM). Results reveal a "synthetic persuasion paradox": synthetic and simulated agents significantly affect their post-intervention PEB stance, while human responses barely shift. Simulated participants better approximate human trends but still overestimate effects. This disconnect underscores LLM's potential for pre-evaluating PEB interventions but warns of its limits in predicting real-world behavior. We call for refined synthetic modeling and sustained and extended human trials to align conversational AI's promise with tangible sustainability outcomes. 

---
# Adaptively evaluating models with task elicitation 

**Authors**: Davis Brown, Prithvi Balehannina, Helen Jin, Shreya Havaldar, Hamed Hassani, Eric Wong  

**Link**: [PDF](https://arxiv.org/pdf/2503.01986)  

**Abstract**: Manual curation of evaluation datasets is struggling to keep up with the rapidly expanding capabilities and deployment scenarios of language models. Towards scalable model profiling, we introduce and validate a framework for evaluating LLMs, called Adaptive Evaluations. Adaptive evaluations use scaffolded language models (evaluator agents) to search through a target model's behavior on a domain dataset and create difficult questions (tasks) that can discover and probe the model's failure modes. We find that frontier models lack consistency when adaptively probed with our framework on a diverse suite of datasets and tasks, including but not limited to legal reasoning, forecasting, and online harassment. Generated questions pass human validity checks and often transfer to other models with different capability profiles, demonstrating that adaptive evaluations can also be used to create difficult domain-specific datasets. 

---
# AskToAct: Enhancing LLMs Tool Use via Self-Correcting Clarification 

**Authors**: Xuan Zhang, Yongliang Shen, Zhe Zheng, Linjuan Wu, Wenqi Zhang, Yuchen Yan, Qiuying Peng, Jun Wang, Weiming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01940)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in tool learning. In real-world scenarios, user queries are often ambiguous and incomplete, requiring effective clarification. However, existing interactive clarification approaches face two critical limitations: reliance on manually constructed datasets and lack of error correction mechanisms during multi-turn clarification. We present AskToAct, which addresses these challenges by exploiting the structural mapping between queries and their tool invocation solutions. Our key insight is that tool parameters naturally represent explicit user intents. By systematically removing key parameters from queries while retaining them as ground truth, we enable automated construction of high-quality training data. We further enhance model robustness by fine-tuning on error-correction augmented data using selective masking mechanism, enabling dynamic error detection during clarification interactions. Comprehensive experiments demonstrate that AskToAct significantly outperforms existing approaches, achieving above 79% accuracy in recovering critical unspecified intents and enhancing clarification efficiency by an average of 48.34% while maintaining high accuracy in tool invocation. Our framework exhibits robust performance across varying complexity levels and successfully generalizes to entirely unseen APIs without additional training, achieving performance comparable to GPT-4 with substantially fewer computational resources. 

---
# Unnatural Languages Are Not Bugs but Features for LLMs 

**Authors**: Keyu Duan, Yiran Zhao, Zhili Feng, Jinjie Ni, Tianyu Pang, Qian Liu, Tianle Cai, Longxu Dou, Kenji Kawaguchi, Anirudh Goyal, J. Zico Kolter, Michael Qizhe Shieh  

**Link**: [PDF](https://arxiv.org/pdf/2503.01926)  

**Abstract**: Large Language Models (LLMs) have been observed to process non-human-readable text sequences, such as jailbreak prompts, often viewed as a bug for aligned LLMs. In this work, we present a systematic investigation challenging this perception, demonstrating that unnatural languages - strings that appear incomprehensible to humans but maintain semantic meanings for LLMs - contain latent features usable by models. Notably, unnatural languages possess latent features that can be generalized across different models and tasks during inference. Furthermore, models fine-tuned on unnatural versions of instruction datasets perform on-par with those trained on natural language, achieving 49.71 win rates in Length-controlled AlpacaEval 2.0 in average across various base models. In addition, through comprehensive analysis, we demonstrate that LLMs process unnatural languages by filtering noise and inferring contextual meaning from filtered words. 

---
# Output Length Effect on DeepSeek-R1's Safety in Forced Thinking 

**Authors**: Xuying Li, Zhuo Li, Yuji Kosuga, Victor Bian  

**Link**: [PDF](https://arxiv.org/pdf/2503.01923)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong reasoning capabilities, but their safety under adversarial conditions remains a challenge. This study examines the impact of output length on the robustness of DeepSeek-R1, particularly in Forced Thinking scenarios. We analyze responses across various adversarial prompts and find that while longer outputs can improve safety through self-correction, certain attack types exploit extended generations. Our findings suggest that output length should be dynamically controlled to balance reasoning effectiveness and security. We propose reinforcement learning-based policy adjustments and adaptive token length regulation to enhance LLM safety. 

---
# NCL-UoR at SemEval-2025 Task 3: Detecting Multilingual Hallucination and Related Observable Overgeneration Text Spans with Modified RefChecker and Modified SeflCheckGPT 

**Authors**: Jiaying Hong, Thanet Markchom, Jianfei Xu, Tong Wu, Huizhi Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01921)  

**Abstract**: SemEval-2025 Task 3 (Mu-SHROOM) focuses on detecting hallucinations in content generated by various large language models (LLMs) across multiple languages. This task involves not only identifying the presence of hallucinations but also pinpointing their specific occurrences. To tackle this challenge, this study introduces two methods: modified RefChecker and modified SelfCheckGPT. The modified RefChecker integrates prompt-based factual verification into References, structuring them as claim-based tests rather than single external knowledge sources. The modified SelfCheckGPT incorporates external knowledge to overcome its reliance on internal knowledge. In addition, both methods' original prompt designs are enhanced to identify hallucinated words within LLM-generated texts. Experimental results demonstrate the effectiveness of the approach, achieving a high ranking on the test dataset in detecting hallucinations across various languages, with an average IoU of 0.5310 and an average COR of 0.5669. 

---
# How to Steer LLM Latents for Hallucination Detection? 

**Authors**: Seongheon Park, Xuefeng Du, Min-Hsuan Yeh, Haobo Wang, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.01917)  

**Abstract**: Hallucinations in LLMs pose a significant concern to their safe deployment in real-world applications. Recent approaches have leveraged the latent space of LLMs for hallucination detection, but their embeddings, optimized for linguistic coherence rather than factual accuracy, often fail to clearly separate truthful and hallucinated content. To this end, we propose the Truthfulness Separator Vector (TSV), a lightweight and flexible steering vector that reshapes the LLM's representation space during inference to enhance the separation between truthful and hallucinated outputs, without altering model parameters. Our two-stage framework first trains TSV on a small set of labeled exemplars to form compact and well-separated clusters. It then augments the exemplar set with unlabeled LLM generations, employing an optimal transport-based algorithm for pseudo-labeling combined with a confidence-based filtering process. Extensive experiments demonstrate that TSV achieves state-of-the-art performance with minimal labeled data, exhibiting strong generalization across datasets and providing a practical solution for real-world LLM applications. 

---
# PsychBench: A comprehensive and professional benchmark for evaluating the performance of LLM-assisted psychiatric clinical practice 

**Authors**: Ruoxi Wang, Shuyu Liu, Ling Zhang, Xuequan Zhu, Rui Yang, Xinzhu Zhou, Fei Wu, Zhi Yang, Cheng Jin, Gang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01903)  

**Abstract**: The advent of Large Language Models (LLMs) offers potential solutions to address problems such as shortage of medical resources and low diagnostic consistency in psychiatric clinical practice. Despite this potential, a robust and comprehensive benchmarking framework to assess the efficacy of LLMs in authentic psychiatric clinical environments is absent. This has impeded the advancement of specialized LLMs tailored to psychiatric applications. In response to this gap, by incorporating clinical demands in psychiatry and clinical data, we proposed a benchmarking system, PsychBench, to evaluate the practical performance of LLMs in psychiatric clinical settings. We conducted a comprehensive quantitative evaluation of 16 LLMs using PsychBench, and investigated the impact of prompt design, chain-of-thought reasoning, input text length, and domain-specific knowledge fine-tuning on model performance. Through detailed error analysis, we identified strengths and potential limitations of the existing models and suggested directions for improvement. Subsequently, a clinical reader study involving 60 psychiatrists of varying seniority was conducted to further explore the practical benefits of existing LLMs as supportive tools for psychiatrists of varying seniority. Through the quantitative and reader evaluation, we show that while existing models demonstrate significant potential, they are not yet adequate as decision-making tools in psychiatric clinical practice. The reader study further indicates that, as an auxiliary tool, LLM could provide particularly notable support for junior psychiatrists, effectively enhancing their work efficiency and overall clinical quality. To promote research in this area, we will make the dataset and evaluation framework publicly available, with the hope of advancing the application of LLMs in psychiatric clinical settings. 

---
# An Empirical Analysis of LLMs for Countering Misinformation 

**Authors**: Adiba Mahbub Proma, Neeley Pate, James Druckman, Gourab Ghoshal, Hangfeng He, Ehsan Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2503.01902)  

**Abstract**: While Large Language Models (LLMs) can amplify online misinformation, they also show promise in tackling misinformation. In this paper, we empirically study the capabilities of three LLMs -- ChatGPT, Gemini, and Claude -- in countering political misinformation. We implement a two-step, chain-of-thought prompting approach, where models first identify credible sources for a given claim and then generate persuasive responses. Our findings suggest that models struggle to ground their responses in real news sources, and tend to prefer citing left-leaning sources. We also observe varying degrees of response diversity among models. Our findings highlight concerns about using LLMs for fact-checking through only prompt-engineering, emphasizing the need for more robust guardrails. Our results have implications for both researchers and non-technical users. 

---
# UDora: A Unified Red Teaming Framework against LLM Agents by Dynamically Hijacking Their Own Reasoning 

**Authors**: Jiawei Zhang, Shuang Yang, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.01908)  

**Abstract**: Large Language Model (LLM) agents equipped with external tools have become increasingly powerful for handling complex tasks such as web shopping, automated email replies, and financial trading. However, these advancements also amplify the risks of adversarial attacks, particularly when LLM agents can access sensitive external functionalities. Moreover, because LLM agents engage in extensive reasoning or planning before executing final actions, manipulating them into performing targeted malicious actions or invoking specific tools remains a significant challenge. Consequently, directly embedding adversarial strings in malicious instructions or injecting malicious prompts into tool interactions has become less effective against modern LLM agents. In this work, we present UDora, a unified red teaming framework designed for LLM Agents that dynamically leverages the agent's own reasoning processes to compel it toward malicious behavior. Specifically, UDora first samples the model's reasoning for the given task, then automatically identifies multiple optimal positions within these reasoning traces to insert targeted perturbations. Subsequently, it uses the modified reasoning as the objective to optimize the adversarial strings. By iteratively applying this process, the LLM agent will then be induced to undertake designated malicious actions or to invoke specific malicious tools. Our approach demonstrates superior effectiveness compared to existing methods across three LLM agent datasets. 

---
# When Continue Learning Meets Multimodal Large Language Model: A Survey 

**Authors**: Yukang Huo, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01887)  

**Abstract**: Recent advancements in Artificial Intelligence have led to the development of Multimodal Large Language Models (MLLMs). However, adapting these pre-trained models to dynamic data distributions and various tasks efficiently remains a challenge. Fine-tuning MLLMs for specific tasks often causes performance degradation in the model's prior knowledge domain, a problem known as 'Catastrophic Forgetting'. While this issue has been well-studied in the Continual Learning (CL) community, it presents new challenges for MLLMs. This review paper, the first of its kind in MLLM continual learning, presents an overview and analysis of 440 research papers in this this http URL review is structured into four sections. First, it discusses the latest research on MLLMs, covering model innovations, benchmarks, and applications in various fields. Second, it categorizes and overviews the latest studies on continual learning, divided into three parts: non-large language models unimodal continual learning (Non-LLM Unimodal CL), non-large language models multimodal continual learning (Non-LLM Multimodal CL), and continual learning in large language models (CL in LLM). The third section provides a detailed analysis of the current state of MLLM continual learning research, including benchmark evaluations, architectural innovations, and a summary of theoretical and empirical this http URL, the paper discusses the challenges and future directions of continual learning in MLLMs, aiming to inspire future research and development in the field. This review connects the foundational concepts, theoretical insights, method innovations, and practical applications of continual learning for multimodal large models, providing a comprehensive understanding of the research progress and challenges in this field, aiming to inspire researchers in the field and promote the advancement of related technologies. 

---
# Evaluating System 1 vs. 2 Reasoning Approaches for Zero-Shot Time-Series Forecasting: A Benchmark and Insights 

**Authors**: Haoxin Liu, Zhiyuan Zhao, Shiduo Li, B. Aditya Prakash  

**Link**: [PDF](https://arxiv.org/pdf/2503.01895)  

**Abstract**: Reasoning ability is crucial for solving challenging tasks. With the advancement of foundation models, such as the emergence of large language models (LLMs), a wide range of reasoning strategies has been proposed, including test-time enhancements, such as Chain-ofThought, and post-training optimizations, as used in DeepSeek-R1. While these reasoning strategies have demonstrated effectiveness across various challenging language or vision tasks, their applicability and impact on time-series forecasting (TSF), particularly the challenging zero-shot TSF, remain largely unexplored. In particular, it is unclear whether zero-shot TSF benefits from reasoning and, if so, what types of reasoning strategies are most effective. To bridge this gap, we propose ReC4TS, the first benchmark that systematically evaluates the effectiveness of popular reasoning strategies when applied to zero-shot TSF tasks. ReC4TS conducts comprehensive evaluations across datasets spanning eight domains, covering both unimodal and multimodal with short-term and longterm forecasting tasks. More importantly, ReC4TS provides key insights: (1) Self-consistency emerges as the most effective test-time reasoning strategy; (2) Group-relative policy optimization emerges as a more suitable approach for incentivizing reasoning ability during post-training; (3) Multimodal TSF benefits more from reasoning strategies compared to unimodal TSF. Beyond these insights, ReC4TS establishes two pioneering starting blocks to support future zero-shot TSF reasoning research: (1) A novel dataset, TimeThinking, containing forecasting samples annotated with reasoning trajectories from multiple advanced LLMs, and (2) A new and simple test-time scaling-law validated on foundational TSF models enabled by self-consistency reasoning strategy. All data and code are publicly accessible at: this https URL 

---
# Starjob: Dataset for LLM-Driven Job Shop Scheduling 

**Authors**: Henrik Abgaryan, Tristan Cazenave, Ararat Harutyunyan  

**Link**: [PDF](https://arxiv.org/pdf/2503.01877)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities across various domains, but their potential for solving combinatorial optimization problems remains largely unexplored. In this paper, we investigate the applicability of LLMs to the Job Shop Scheduling Problem (JSSP), a classic challenge in combinatorial optimization that requires efficient job allocation to machines to minimize makespan. To this end, we introduce Starjob, the first supervised dataset for JSSP, comprising 130k instances specifically designed for training LLMs. Leveraging this dataset, we fine-tune the LLaMA 8B 4-bit quantized model with the LoRA method to develop an end-to-end scheduling approach. Our evaluation on standard benchmarks demonstrates that the proposed LLM-based method not only surpasses traditional Priority Dispatching Rules (PDRs) but also achieves notable improvements over state-of-the-art neural approaches like L2D, with an average improvement of 15.36% on DMU and 7.85% on Taillard benchmarks. These results highlight the untapped potential of LLMs in tackling combinatorial optimization problems, paving the way for future advancements in this area. 

---
# Can Large Language Models Extract Customer Needs as well as Professional Analysts? 

**Authors**: Artem Timoshenko, Chengfeng Mao, John R. Hauser  

**Link**: [PDF](https://arxiv.org/pdf/2503.01870)  

**Abstract**: Identifying customer needs (CNs) is important for product management, product development, and marketing. Applications rely on professional analysts interpreting textual data (e.g., interview transcripts, online reviews) to understand the nuances of customer experience and concisely formulate "jobs to be done." The task is cognitively complex and time-consuming. Current practice facilitates the process with keyword search and machine learning but relies on human judgment to formulate CNs. We examine whether Large Language Models (LLMs) can automatically extract CNs. Because evaluating CNs requires professional judgment, we partnered with a marketing consulting firm to conduct a blind study of CNs extracted by: (1) a foundational LLM with prompt engineering only (Base LLM), (2) an LLM fine-tuned with professionally identified CNs (SFT LLM), and (3) professional analysts. The SFT LLM performs as well as or better than professional analysts when extracting CNs. The extracted CNs are well-formulated, sufficiently specific to identify opportunities, and justified by source content (no hallucinations). The SFT LLM is efficient and provides more complete coverage of CNs. The Base LLM was not sufficiently accurate or specific. Organizations can rely on SFT LLMs to reduce manual effort, enhance the precision of CN articulation, and provide improved insight for innovation and marketing strategy. 

---
# Larger or Smaller Reward Margins to Select Preferences for Alignment? 

**Authors**: Kexin Huang, Junkang Wu, Ziqian Chen, Xue Wang, Jinyang Gao, Bolin Ding, Jiancan Wu, Xiangnan He, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01864)  

**Abstract**: Preference learning is critical for aligning large language models (LLMs) with human values, with the quality of preference datasets playing a crucial role in this process. While existing metrics primarily assess data quality based on either explicit or implicit reward margins, they often provide contradictory evaluations for the same data. To address this issue, we introduce the alignment potential metric, which quantifies the gap from the model's current implicit reward margin to the target explicit reward margin, thereby estimating the model's potential to align with the preference data. Empirical results demonstrate that training on data selected by this metric consistently enhances alignment performance, surpassing existing metrics across different base models and optimization objectives. Furthermore, our method extends to self-play data generation frameworks, where the metric is used to identify high-quality data within the self-generated content by LLMs. Under this data generation scenario, our method surpasses current state-of-the-art (SOTA) results across various training settings and demonstrates continuous improvements in alignment performance as dataset size and training iterations increase. 

---
# A Review of Artificial Intelligence Impacting Statistical Process Monitoring and Future Directions 

**Authors**: Shing I Chang, Parviz Ghafariasl  

**Link**: [PDF](https://arxiv.org/pdf/2503.01858)  

**Abstract**: It has been 100 years since statistical process control (SPC) or statistical process monitoring (SPM) was first introduced for production processes and later applied to service, healthcare, and other industries. The techniques applied to SPM applications are mostly statistically oriented. Recent advances in Artificial Intelligence (AI) have reinvigorated the imagination of adopting AI for SPM applications. This manuscript begins with a concise review of the historical development of the statistically based SPM methods. Next, this manuscript explores AI and Machine Learning (ML) algorithms and methods applied in various SPM applications, addressing quality characteristics of univariate, multivariate, profile, and image. These AI methods can be classified into the following categories: classification, pattern recognition, time series applications, and generative AI. Specifically, different kinds of neural networks, such as artificial neural networks (ANN), convolutional neural networks (CNN), recurrent neural networks (RNN), and generative adversarial networks (GAN), are among the most implemented AI methods impacting SPM. Finally, this manuscript outlines a couple of future directions that harness the potential of the Large Multimodal Model (LMM) for advancing SPM research and applications in complex systems. The ultimate objective is to transform statistical process monitoring (SPM) into smart process control (SMPC), where corrective actions are autonomously implemented to either prevent quality issues or restore process performance. 

---
# LLM-Empowered Class Imbalanced Graph Prompt Learning for Online Drug Trafficking Detection 

**Authors**: Tianyi Ma, Yiyue Qian, Zehong Wang, Zheyuan Zhang, Chuxu Zhang, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.01900)  

**Abstract**: As the market for illicit drugs remains extremely profitable, major online platforms have become direct-to-consumer intermediaries for illicit drug trafficking participants. These online activities raise significant social concerns that require immediate actions. Existing approaches to combating this challenge are generally impractical, due to the imbalance of classes and scarcity of labeled samples in real-world applications. To this end, we propose a novel Large Language Model-empowered Heterogeneous Graph Prompt Learning framework for illicit Drug Trafficking detection, called LLM-HetGDT, that leverages LLM to facilitate heterogeneous graph neural networks (HGNNs) to effectively identify drug trafficking activities in the class-imbalanced scenarios. Specifically, we first pre-train HGNN over a contrastive pretext task to capture the inherent node and structure information over the unlabeled drug trafficking heterogeneous graph (HG). Afterward, we employ LLM to augment the HG by generating high-quality synthetic user nodes in minority classes. Then, we fine-tune the soft prompts on the augmented HG to capture the important information in the minority classes for the downstream drug trafficking detection task. To comprehensively study online illicit drug trafficking activities, we collect a new HG dataset over Twitter, called Twitter-HetDrug. Extensive experiments on this dataset demonstrate the effectiveness, efficiency, and applicability of LLM-HetGDT. 

---
# FairSense-AI: Responsible AI Meets Sustainability 

**Authors**: Shaina Raza, Mukund Sayeeganesh Chettiar, Matin Yousefabadi, Tahniat Khan, Marcelo Lotif  

**Link**: [PDF](https://arxiv.org/pdf/2503.02865)  

**Abstract**: In this paper, we introduce FairSense-AI: a multimodal framework designed to detect and mitigate bias in both text and images. By leveraging Large Language Models (LLMs) and Vision-Language Models (VLMs), FairSense-AI uncovers subtle forms of prejudice or stereotyping that can appear in content, providing users with bias scores, explanatory highlights, and automated recommendations for fairness enhancements. In addition, FairSense-AI integrates an AI risk assessment component that aligns with frameworks like the MIT AI Risk Repository and NIST AI Risk Management Framework, enabling structured identification of ethical and safety concerns. The platform is optimized for energy efficiency via techniques such as model pruning and mixed-precision computation, thereby reducing its environmental footprint. Through a series of case studies and applications, we demonstrate how FairSense-AI promotes responsible AI use by addressing both the social dimension of fairness and the pressing need for sustainability in large-scale AI deployments. this https URL, this https URL 

---
# The First Few Tokens Are All You Need: An Efficient and Effective Unsupervised Prefix Fine-Tuning Method for Reasoning Models 

**Authors**: Ke Ji, Jiahao Xu, Tian Liang, Qiuzhi Liu, Zhiwei He, Xingyu Chen, Xiaoyuan Liu, Zhijie Wang, Junying Chen, Benyou Wang, Zhaopeng Tu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02875)  

**Abstract**: Improving the reasoning capabilities of large language models (LLMs) typically requires supervised fine-tuning with labeled data or computationally expensive sampling. We introduce Unsupervised Prefix Fine-Tuning (UPFT), which leverages the observation of Prefix Self-Consistency -- the shared initial reasoning steps across diverse solution trajectories -- to enhance LLM reasoning efficiency. By training exclusively on the initial prefix substrings (as few as 8 tokens), UPFT removes the need for labeled data or exhaustive sampling. Experiments on reasoning benchmarks show that UPFT matches the performance of supervised methods such as Rejection Sampling Fine-Tuning, while reducing training time by 75% and sampling cost by 99%. Further analysis reveals that errors tend to appear in later stages of the reasoning process and that prefix-based training preserves the model's structural knowledge. This work demonstrates how minimal unsupervised fine-tuning can unlock substantial reasoning gains in LLMs, offering a scalable and resource-efficient alternative to conventional approaches. 

---
# Large Language Models for Multilingual Previously Fact-Checked Claim Detection 

**Authors**: Ivan Vykopal, Matúš Pikuliak, Simon Ostermann, Tatiana Anikina, Michal Gregor, Marián Šimko  

**Link**: [PDF](https://arxiv.org/pdf/2503.02737)  

**Abstract**: In our era of widespread false information, human fact-checkers often face the challenge of duplicating efforts when verifying claims that may have already been addressed in other countries or languages. As false information transcends linguistic boundaries, the ability to automatically detect previously fact-checked claims across languages has become an increasingly important task. This paper presents the first comprehensive evaluation of large language models (LLMs) for multilingual previously fact-checked claim detection. We assess seven LLMs across 20 languages in both monolingual and cross-lingual settings. Our results show that while LLMs perform well for high-resource languages, they struggle with low-resource languages. Moreover, translating original texts into English proved to be beneficial for low-resource languages. These findings highlight the potential of LLMs for multilingual previously fact-checked claim detection and provide a foundation for further research on this promising application of LLMs. 

---
# Shakespearean Sparks: The Dance of Hallucination and Creativity in LLMs' Decoding Layers 

**Authors**: Zicong He, Boxuan Zhang, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.02851)  

**Abstract**: Large language models (LLMs) are known to hallucinate, a phenomenon often linked to creativity. While previous research has primarily explored this connection through theoretical or qualitative lenses, our work takes a quantitative approach to systematically examine the relationship between hallucination and creativity in LLMs. Given the complex nature of creativity, we propose a narrow definition tailored to LLMs and introduce an evaluation framework, HCL, which quantifies Hallucination and Creativity across different Layers of LLMs during decoding. Our empirical analysis reveals a tradeoff between hallucination and creativity that is consistent across layer depth, model type, and model size. Notably, across different model architectures, we identify a specific layer at each model size that optimally balances this tradeoff. Additionally, the optimal layer tends to appear in the early layers of larger models, and the confidence of the model is also significantly higher at this layer. These findings provide a quantitative perspective that offers new insights into the interplay between LLM creativity and hallucination. The code and data for our experiments are available at this https URL. 

---
# Evaluating Knowledge Generation and Self-Refinement Strategies for LLM-based Column Type Annotation 

**Authors**: Keti Korini, Christian Bizer  

**Link**: [PDF](https://arxiv.org/pdf/2503.02718)  

**Abstract**: Understanding the semantics of columns in relational tables is an important pre-processing step for indexing data lakes in order to provide rich data search. An approach to establishing such understanding is column type annotation (CTA) where the goal is to annotate table columns with terms from a given vocabulary. This paper experimentally compares different knowledge generation and self-refinement strategies for LLM-based column type annotation. The strategies include using LLMs to generate term definitions, error-based refinement of term definitions, self-correction, and fine-tuning using examples and term definitions. We evaluate these strategies along two dimensions: effectiveness measured as F1 performance and efficiency measured in terms of token usage and cost. Our experiments show that the best performing strategy depends on the model/dataset combination. We find that using training data to generate label definitions outperforms using the same data as demonstrations for in-context learning for two out of three datasets using OpenAI models. The experiments further show that using the LLMs to refine label definitions brings an average increase of 3.9% F1 in 10 out of 12 setups compared to the performance of the non-refined definitions. Combining fine-tuned models with self-refined term definitions results in the overall highest performance, outperforming zero-shot prompting fine-tuned models by at least 3% in F1 score. The costs analysis shows that while reaching similar F1 score, self-refinement via prompting is more cost efficient for use cases requiring smaller amounts of tables to be annotated while fine-tuning is more efficient for large amounts of tables. 

---
# Calibrating LLM Confidence with Semantic Steering: A Multi-Prompt Aggregation Framework 

**Authors**: Ziang Zhou, Tianyuan Jin, Jieming Shi, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.02863)  

**Abstract**: Large Language Models (LLMs) often exhibit misaligned confidence scores, usually overestimating the reliability of their predictions. While verbalized confidence in Large Language Models (LLMs) has gained attention, prior work remains divided on whether confidence scores can be systematically steered through prompting. Recent studies even argue that such prompt-induced confidence shifts are negligible, suggesting LLMs' confidence calibration is rigid to linguistic interventions. Contrary to these claims, we first rigorously confirm the existence of directional confidence shifts by probing three models (including GPT3.5, LLAMA3-70b, GPT4) across 7 benchmarks, demonstrating that explicit instructions can inflate or deflate confidence scores in a regulated manner. Based on this observation, we propose a novel framework containing three components: confidence steering, steered confidence aggregation and steered answers selection, named SteeringConf. Our method, SteeringConf, leverages a confidence manipulation mechanism to steer the confidence scores of LLMs in several desired directions, followed by a summarization module that aggregates the steered confidence scores to produce a final prediction. We evaluate our method on 7 benchmarks and it consistently outperforms the baselines in terms of calibration metrics in task of confidence calibration and failure detection. 

---
# Multidimensional Consistency Improves Reasoning in Language Models 

**Authors**: Huiyuan Lai, Xiao Zhang, Malvina Nissim  

**Link**: [PDF](https://arxiv.org/pdf/2503.02670)  

**Abstract**: While Large language models (LLMs) have proved able to address some complex reasoning tasks, we also know that they are highly sensitive to input variation, which can lead to different solution paths and final answers. Answer consistency across input variations can thus be taken as a sign of stronger confidence. Leveraging this insight, we introduce a framework, {\em Multidimensional Reasoning Consistency} where, focusing on math problems, models are systematically pushed to diversify solution paths towards a final answer, thereby testing them for answer consistency across multiple input variations. We induce variations in (i) order of shots in prompt, (ii) problem phrasing, and (iii) languages used. Extensive experiments on a large range of open-source state-of-the-art LLMs of various sizes show that reasoning consistency differs by variation dimension, and that by aggregating consistency across dimensions, our framework consistently enhances mathematical reasoning performance on both monolingual dataset GSM8K and multilingual dataset MGSM, especially for smaller models. 

---
# Mask-DPO: Generalizable Fine-grained Factuality Alignment of LLMs 

**Authors**: Yuzhe Gu, Wenwei Zhang, Chengqi Lyu, Dahua Lin, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.02846)  

**Abstract**: Large language models (LLMs) exhibit hallucinations (i.e., unfaithful or nonsensical information) when serving as AI assistants in various domains. Since hallucinations always come with truthful content in the LLM responses, previous factuality alignment methods that conduct response-level preference learning inevitably introduced noises during training. Therefore, this paper proposes a fine-grained factuality alignment method based on Direct Preference Optimization (DPO), called Mask-DPO. Incorporating sentence-level factuality as mask signals, Mask-DPO only learns from factually correct sentences in the preferred samples and prevents the penalty on factual contents in the not preferred samples, which resolves the ambiguity in the preference learning. Extensive experimental results demonstrate that Mask-DPO can significantly improve the factuality of LLMs responses to questions from both in-domain and out-of-domain datasets, although these questions and their corresponding topics are unseen during training. Only trained on the ANAH train set, the score of Llama3.1-8B-Instruct on the ANAH test set is improved from 49.19% to 77.53%, even surpassing the score of Llama3.1-70B-Instruct (53.44%), while its FactScore on the out-of-domain Biography dataset is also improved from 30.29% to 39.39%. We further study the generalization property of Mask-DPO using different training sample scaling strategies and find that scaling the number of topics in the dataset is more effective than the number of questions. We provide a hypothesis of what factual alignment is doing with LLMs, on the implication of this phenomenon, and conduct proof-of-concept experiments to verify it. We hope the method and the findings pave the way for future research on scaling factuality alignment. 

---
# OkraLong: A Flexible Retrieval-Augmented Framework for Long-Text Query Processing 

**Authors**: Yulong Hui, Yihao Liu, Yao Lu, Huanchen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02603)  

**Abstract**: Large Language Models (LLMs) encounter challenges in efficiently processing long-text queries, as seen in applications like enterprise document analysis and financial report comprehension. While conventional solutions employ long-context processing or Retrieval-Augmented Generation (RAG), they suffer from prohibitive input expenses or incomplete information. Recent advancements adopt context compression and dynamic retrieval loops, but still sacrifice critical details or incur iterative this http URL address these limitations, we propose OkraLong, a novel framework that flexibly optimizes the entire processing workflow. Unlike prior static or coarse-grained adaptive strategies, OkraLong adopts fine-grained orchestration through three synergistic components: analyzer, organizer and executor. The analyzer characterizes the task states, which guide the organizer in dynamically scheduling the workflow. The executor carries out the execution and generates the final answer. Experimental results demonstrate that OkraLong not only enhances answer accuracy but also achieves cost-effectiveness across a variety of datasets. 

---
# LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs 

**Authors**: Jianghao Chen, Junhong Wu, Yangyifan Xu, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02502)  

**Abstract**: Long-context modeling has drawn more and more attention in the area of Large Language Models (LLMs). Continual training with long-context data becomes the de-facto method to equip LLMs with the ability to process long inputs. However, it still remains an open challenge to measure the quality of long-context training data. To address this issue, we propose a Long-context data selection framework with Attention-based Dependency Measurement (LADM), which can efficiently identify high-quality long-context data from a large-scale, multi-domain pre-training corpus. LADM leverages the retrieval capabilities of the attention mechanism to capture contextual dependencies, ensuring a comprehensive quality measurement of long-context data. Experimental results show that our LADM framework significantly boosts the performance of LLMs on multiple long-context tasks with only 1B tokens for continual training. 

---
# It Helps to Take a Second Opinion: Teaching Smaller LLMs to Deliberate Mutually via Selective Rationale Optimisation 

**Authors**: Sohan Patnaik, Milan Aggarwal, Sumit Bhatia, Balaji Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2503.02463)  

**Abstract**: Very large language models (LLMs) such as GPT-4 have shown the ability to handle complex tasks by generating and self-refining step-by-step rationales. Smaller language models (SLMs), typically with < 13B parameters, have been improved by using the data generated from very-large LMs through knowledge distillation. However, various practical constraints such as API costs, copyright, legal and ethical policies restrict using large (often opaque) models to train smaller models for commercial use. Limited success has been achieved at improving the ability of an SLM to explore the space of possible rationales and evaluate them by itself through self-deliberation. To address this, we propose COALITION, a trainable framework that facilitates interaction between two variants of the same SLM and trains them to generate and refine rationales optimized for the end-task. The variants exhibit different behaviors to produce a set of diverse candidate rationales during the generation and refinement steps. The model is then trained via Selective Rationale Optimization (SRO) to prefer generating rationale candidates that maximize the likelihood of producing the ground-truth answer. During inference, COALITION employs a controller to select the suitable variant for generating and refining the rationales. On five different datasets covering mathematical problems, commonsense reasoning, and natural language inference, COALITION outperforms several baselines by up to 5%. Our ablation studies reveal that cross-communication between the two variants performs better than using the single model to self-refine the rationales. We also demonstrate the applicability of COALITION for LMs of varying scales (4B to 14B parameters) and model families (Mistral, Llama, Qwen, Phi). We release the code for this work at this https URL. 

---
# AILS-NTUA at SemEval-2025 Task 3: Leveraging Large Language Models and Translation Strategies for Multilingual Hallucination Detection 

**Authors**: Dimitra Karkani, Maria Lymperaiou, Giorgos Filandrianos, Nikolaos Spanos, Athanasios Voulodimos, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2503.02442)  

**Abstract**: Multilingual hallucination detection stands as an underexplored challenge, which the Mu-SHROOM shared task seeks to address. In this work, we propose an efficient, training-free LLM prompting strategy that enhances detection by translating multilingual text spans into English. Our approach achieves competitive rankings across multiple languages, securing two first positions in low-resource languages. The consistency of our results highlights the effectiveness of our translation strategy for hallucination detection, demonstrating its applicability regardless of the source language. 

---
# MedEthicEval: Evaluating Large Language Models Based on Chinese Medical Ethics 

**Authors**: Haoan Jin, Jiacheng Shi, Hanhui Xu, Kenny Q. Zhu, Mengyue Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02374)  

**Abstract**: Large language models (LLMs) demonstrate significant potential in advancing medical applications, yet their capabilities in addressing medical ethics challenges remain underexplored. This paper introduces MedEthicEval, a novel benchmark designed to systematically evaluate LLMs in the domain of medical ethics. Our framework encompasses two key components: knowledge, assessing the models' grasp of medical ethics principles, and application, focusing on their ability to apply these principles across diverse scenarios. To support this benchmark, we consulted with medical ethics researchers and developed three datasets addressing distinct ethical challenges: blatant violations of medical ethics, priority dilemmas with clear inclinations, and equilibrium dilemmas without obvious resolutions. MedEthicEval serves as a critical tool for understanding LLMs' ethical reasoning in healthcare, paving the way for their responsible and effective use in medical contexts. 

---
# Add-One-In: Incremental Sample Selection for Large Language Models via a Choice-Based Greedy Paradigm 

**Authors**: Zhuo Li, Yuhao Du, Xiaoqi Jiao, Yiwen Guo, Yuege Feng, Xiang Wan, Anningzhe Gao, Jinpeng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02359)  

**Abstract**: Selecting high-quality and diverse training samples from extensive datasets plays a crucial role in reducing training overhead and enhancing the performance of Large Language Models (LLMs). However, existing studies fall short in assessing the overall value of selected data, focusing primarily on individual quality, and struggle to strike an effective balance between ensuring diversity and minimizing data point traversals. Therefore, this paper introduces a novel choice-based sample selection framework that shifts the focus from evaluating individual sample quality to comparing the contribution value of different samples when incorporated into the subset. Thanks to the advanced language understanding capabilities of LLMs, we utilize LLMs to evaluate the value of each option during the selection process. Furthermore, we design a greedy sampling process where samples are incrementally added to the subset, thereby improving efficiency by eliminating the need for exhaustive traversal of the entire dataset with the limited budget. Extensive experiments demonstrate that selected data from our method not only surpass the performance of the full dataset but also achieves competitive results with state-of-the-art (SOTA) studies, while requiring fewer selections. Moreover, we validate our approach on a larger medical dataset, highlighting its practical applicability in real-world applications. 

---
# DeLTa: A Decoding Strategy based on Logit Trajectory Prediction Improves Factuality and Reasoning Ability 

**Authors**: Yunzhen He, Yusuke Takase, Yoichi Ishibashi, Hidetoshi Shimodaira  

**Link**: [PDF](https://arxiv.org/pdf/2503.02343)  

**Abstract**: Large Language Models (LLMs) are increasingly being used in real-world applications. However, concerns about the reliability of the content they generate persist, as it frequently deviates from factual correctness or exhibits deficiencies in logical reasoning. This paper proposes a novel decoding strategy aimed at enhancing both factual accuracy and inferential reasoning without requiring any modifications to the architecture or pre-trained parameters of LLMs. Our approach adjusts next-token probabilities by analyzing the trajectory of logits from lower to higher layers in Transformers and applying linear regression. We find that this Decoding by Logit Trajectory-based approach (DeLTa) effectively reinforces factuality and reasoning while mitigating incorrect generation. Experiments on TruthfulQA demonstrate that DeLTa attains up to a 4.9% improvement over the baseline. Furthermore, it enhances performance by up to 8.1% on StrategyQA and 7.3% on GSM8K, both of which demand strong reasoning capabilities. 

---
# BatchGEMBA: Token-Efficient Machine Translation Evaluation with Batched Prompting and Prompt Compression 

**Authors**: Daniil Larionov, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2503.02756)  

**Abstract**: Recent advancements in Large Language Model (LLM)-based Natural Language Generation evaluation have largely focused on single-example prompting, resulting in significant token overhead and computational inefficiencies. In this work, we introduce BatchGEMBA-MQM, a framework that integrates batched prompting with the GEMBA-MQM metric for machine translation evaluation. Our approach aggregates multiple translation examples into a single prompt, reducing token usage by 2-4 times (depending on the batch size) relative to single-example prompting. Furthermore, we propose a batching-aware prompt compression model that achieves an additional token reduction of 13-15% on average while also showing ability to help mitigate batching-induced quality degradation. Evaluations across several LLMs (GPT-4o, GPT-4o-mini, Mistral Small, Phi4, and CommandR7B) and varying batch sizes reveal that while batching generally negatively affects quality (but sometimes not substantially), prompt compression does not degrade further, and in some cases, recovers quality loss. For instance, GPT-4o retains over 90% of its baseline performance at a batch size of 4 when compression is applied, compared to a 44.6% drop without compression. We plan to release our code and trained models at this https URL to support future research in this domain. 

---
# Limited Effectiveness of LLM-based Data Augmentation for COVID-19 Misinformation Stance Detection 

**Authors**: Eun Cheol Choi, Ashwin Balasubramanian, Jinhu Qi, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2503.02328)  

**Abstract**: Misinformation surrounding emerging outbreaks poses a serious societal threat, making robust countermeasures essential. One promising approach is stance detection (SD), which identifies whether social media posts support or oppose misleading claims. In this work, we finetune classifiers on COVID-19 misinformation SD datasets consisting of claims and corresponding tweets. Specifically, we test controllable misinformation generation (CMG) using large language models (LLMs) as a method for data augmentation. While CMG demonstrates the potential for expanding training datasets, our experiments reveal that performance gains over traditional augmentation methods are often minimal and inconsistent, primarily due to built-in safeguards within LLMs. We release our code and datasets to facilitate further research on misinformation detection and generation. 

---
# LoRA-Null: Low-Rank Adaptation via Null Space for Large Language Models 

**Authors**: Pengwei Tang, Yong Liu, Dongjie Zhang, Xing Wu, Debing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02659)  

**Abstract**: Low-Rank Adaptation (LoRA) is the leading parameter-efficient fine-tuning method for Large Language Models (LLMs). However, the fine-tuned LLMs encounter the issue of catastrophic forgetting of the pre-trained world knowledge. To address this issue, inspired by theoretical insights of null space, we propose LoRA-Null, i.e., Low-Rank Adaptation via null space, which builds adapters initialized from the null space of the pre-trained knowledge activation. Concretely, we randomly collect a few data samples and capture their activations after passing through the LLM layer. We perform Singular Value Decomposition on the input activations to obtain their null space. We use the projection of the pre-trained weights onto the null space as the initialization for adapters. Experimental results demonstrate that this initialization approach can effectively preserve the original pre-trained world knowledge of the LLMs during fine-tuning. Additionally, if we freeze the values of the down-projection matrices during fine-tuning, it achieves even better preservation of the pre-trained world knowledge. LoRA-Null effectively preserves pre-trained world knowledge while maintaining strong fine-tuning performance, as validated by extensive experiments on LLaMA series (LLaMA2, LLaMA3, LLaMA3.1, and LLaMA3.2) across Code, Math, and Instruction Following tasks. We also provide a theoretical guarantee for the capacity of LoRA-Null to retain pre-trained knowledge. Code is in this https URL. 

---
# Superficial Self-Improved Reasoners Benefit from Model Merging 

**Authors**: Xiangchi Yuan, Chunhui Zhang, Zheyuan Liu, Dachuan Shi, Soroush Vosoughi, Wenke Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.02103)  

**Abstract**: As scaled language models (LMs) approach human-level reasoning capabilities, self-improvement emerges as a solution to synthesizing high-quality data corpus. While previous research has identified model collapse as a risk in self-improvement, where model outputs become increasingly deterministic, we discover a more fundamental challenge: the superficial self-improved reasoners phenomenon. In particular, our analysis reveals that even when LMs show improved in-domain (ID) reasoning accuracy, they actually compromise their generalized reasoning capabilities on out-of-domain (OOD) tasks due to memorization rather than genuine. Through a systematic investigation of LM architecture, we discover that during self-improvement, LM weight updates are concentrated in less reasoning-critical layers, leading to superficial learning. To address this, we propose Iterative Model Merging (IMM), a method that strategically combines weights from original and self-improved models to preserve generalization while incorporating genuine reasoning improvements. Our approach effectively mitigates both LM collapse and superficial learning, moving towards more stable self-improving systems. 

---
# Persuasion at Play: Understanding Misinformation Dynamics in Demographic-Aware Human-LLM Interactions 

**Authors**: Angana Borah, Rada Mihalcea, Verónica Pérez-Rosas  

**Link**: [PDF](https://arxiv.org/pdf/2503.02038)  

**Abstract**: Existing challenges in misinformation exposure and susceptibility vary across demographic groups, as some populations are more vulnerable to misinformation than others. Large language models (LLMs) introduce new dimensions to these challenges through their ability to generate persuasive content at scale and reinforcing existing biases. This study investigates the bidirectional persuasion dynamics between LLMs and humans when exposed to misinformative content. We analyze human-to-LLM influence using human-stance datasets and assess LLM-to-human influence by generating LLM-based persuasive arguments. Additionally, we use a multi-agent LLM framework to analyze the spread of misinformation under persuasion among demographic-oriented LLM agents. Our findings show that demographic factors influence susceptibility to misinformation in LLMs, closely reflecting the demographic-based patterns seen in human susceptibility. We also find that, similar to human demographic groups, multi-agent LLMs exhibit echo chamber behavior. This research explores the interplay between humans and LLMs, highlighting demographic differences in the context of misinformation and offering insights for future interventions. 

---
# HoT: Highlighted Chain of Thought for Referencing Supportive Facts from Inputs 

**Authors**: Tin Nguyen, Logan Bolton, Mohammad Reza Taesiri, Anh Totti Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2503.02003)  

**Abstract**: An Achilles heel of Large Language Models (LLMs) is their tendency to hallucinate non-factual statements. A response mixed of factual and non-factual statements poses a challenge for humans to verify and accurately base their decisions on. To combat this problem, we propose Highlighted Chain-of-Thought Prompting (HoT), a technique for prompting LLMs to generate responses with XML tags that ground facts to those provided in the query. That is, given an input question, LLMs would first re-format the question to add XML tags highlighting key facts, and then, generate a response with highlights over the facts referenced from the input. Interestingly, in few-shot settings, HoT outperforms vanilla chain of thought prompting (CoT) on a wide range of 17 tasks from arithmetic, reading comprehension to logical reasoning. When asking humans to verify LLM responses, highlights help time-limited participants to more accurately and efficiently recognize when LLMs are correct. Yet, surprisingly, when LLMs are wrong, HoTs tend to make users believe that an answer is correct. 

---
# Measuring What Makes You Unique: Difference-Aware User Modeling for Enhancing LLM Personalization 

**Authors**: Yilun Qiu, Xiaoyan Zhao, Yang Zhang, Yimeng Bai, Wenjie Wang, Hong Cheng, Fuli Feng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2503.02450)  

**Abstract**: Personalizing Large Language Models (LLMs) has become a critical step in facilitating their widespread application to enhance individual life experiences. In pursuit of personalization, distilling key preference information from an individual's historical data as instructional preference context to customize LLM generation has emerged as a promising direction. However, these methods face a fundamental limitation by overlooking the inter-user comparative analysis, which is essential for identifying the inter-user differences that truly shape preferences. To address this limitation, we propose Difference-aware Personalization Learning (DPL), a novel approach that emphasizes extracting inter-user differences to enhance LLM personalization. DPL strategically selects representative users for comparison and establishes a structured standard to extract meaningful, task-relevant differences for customizing LLM generation. Extensive experiments on real-world datasets demonstrate that DPL significantly enhances LLM personalization. We release our code at this https URL. 

---
# Haste Makes Waste: Evaluating Planning Abilities of LLMs for Efficient and Feasible Multitasking with Time Constraints Between Actions 

**Authors**: Zirui Wu, Xiao Liu, Jiayi Li, Lingpeng Kong, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.02238)  

**Abstract**: While Large Language Model-based agents have demonstrated substantial progress in task completion, existing evaluation benchmarks tend to overemphasize single-task performance, with insufficient attention given to the crucial aspects of multitask planning and execution efficiency required in real-world scenarios. To bridge this gap, we present Recipe2Plan, a novel benchmark framework based on real-world cooking scenarios. Unlike conventional benchmarks, Recipe2Plan challenges agents to optimize cooking time through parallel task execution while respecting temporal constraints i.e. specific actions need to be performed within a particular time intervals following the preceding steps. Overly aggressive local parallelization may disrupt this constraint, potentially compromising the entire cooking process. This strict time constraint between actions raises a unique challenge for agents to balance between maximizing concurrent operations and adhering to critical timing constraints. Extensive experiments with state-of-the-art models reveal challenges in maintaining this balance between efficiency and feasibility. The results highlight the need for improved temporal awareness and global multitasking capabilities in large language models. We open-source our benchmark and code at this https URL. 

---
# Unlocking a New Rust Programming Experience: Fast and Slow Thinking with LLMs to Conquer Undefined Behaviors 

**Authors**: Renshuang Jiang, Pan Dong, Zhenling Duan, Yu Shi, Xiaoxiang Fang, Yan Ding, Jun Ma, Shuai Zhao, Zhe Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02335)  

**Abstract**: To provide flexibility and low-level interaction capabilities, the unsafe tag in Rust is essential in many projects, but undermines memory safety and introduces Undefined Behaviors (UBs) that reduce safety. Eliminating these UBs requires a deep understanding of Rust's safety rules and strong typing. Traditional methods require depth analysis of code, which is laborious and depends on knowledge design. The powerful semantic understanding capabilities of LLM offer new opportunities to solve this problem. Although existing large model debugging frameworks excel in semantic tasks, limited by fixed processes and lack adaptive and dynamic adjustment capabilities. Inspired by the dual process theory of decision-making (Fast and Slow Thinking), we present a LLM-based framework called RustBrain that automatically and flexibly minimizes UBs in Rust projects. Fast thinking extracts features to generate solutions, while slow thinking decomposes, verifies, and generalizes them abstractly. To apply verification and generalization results to solution generation, enabling dynamic adjustments and precise outputs, RustBrain integrates two thinking through a feedback mechanism. Experimental results on Miri dataset show a 94.3% pass rate and 80.4% execution rate, improving flexibility and Rust projects safety. 

---
# Tabby: Tabular Data Synthesis with Language Models 

**Authors**: Sonia Cromp, Satya Sai Srinath Namburi GNVV, Mohammed Alkhudhayri, Catherine Cao, Samuel Guo, Nicholas Roberts, Frederic Sala  

**Link**: [PDF](https://arxiv.org/pdf/2503.02152)  

**Abstract**: While advances in large language models (LLMs) have greatly improved the quality of synthetic text data in recent years, synthesizing tabular data has received relatively less attention. We address this disparity with Tabby, a simple but powerful post-training modification to the standard Transformer language model architecture, enabling its use for tabular dataset synthesis. Tabby enables the representation of differences across columns using Gated Mixture-of-Experts, with column-specific sets of parameters. Empirically, Tabby results in data quality near or equal to that of real data. By pairing our novel LLM table training technique, Plain, with Tabby, we observe up to a 44% improvement in quality over previous methods. We also show that Tabby extends beyond tables to more general structured data, reaching parity with real data on a nested JSON dataset as well. 

---
# Network Traffic Classification Using Machine Learning, Transformer, and Large Language Models 

**Authors**: Ahmad Antari, Yazan Abo-Aisheh, Jehad Shamasneh, Huthaifa I. Ashqar  

**Link**: [PDF](https://arxiv.org/pdf/2503.02141)  

**Abstract**: This study uses various models to address network traffic classification, categorizing traffic into web, browsing, IPSec, backup, and email. We collected a comprehensive dataset from Arbor Edge Defender (AED) devices, comprising of 30,959 observations and 19 features. Multiple models were evaluated, including Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, XGBoost, Deep Neural Networks (DNN), Transformer, and two Large Language Models (LLMs) including GPT-4o and Gemini with zero- and few-shot learning. Transformer and XGBoost showed the best performance, achieving the highest accuracy of 98.95 and 97.56%, respectively. GPT-4o and Gemini showed promising results with few-shot learning, improving accuracy significantly from initial zero-shot performance. While Gemini Few-Shot and GPT-4o Few-Shot performed well in categories like Web and Email, misclassifications occurred in more complex categories like IPSec and Backup. The study highlights the importance of model selection, fine-tuning, and the balance between training data size and model complexity for achieving reliable classification results. 

---
# Generator-Assistant Stepwise Rollback Framework for Large Language Model Agent 

**Authors**: Xingzuo Li, Kehai Chen, Yunfei Long, Xuefeng Bai, Yong Xu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02519)  

**Abstract**: Large language model (LLM) agents typically adopt a step-by-step reasoning framework, in which they interleave the processes of thinking and acting to accomplish the given task. However, this paradigm faces a deep-rooted one-pass issue whereby each generated intermediate thought is plugged into the trajectory regardless of its correctness, which can cause irreversible error propagation. To address the issue, this paper proposes a novel framework called Generator-Assistant Stepwise Rollback (GA-Rollback) to induce better decision-making for LLM agents. Particularly, GA-Rollback utilizes a generator to interact with the environment and an assistant to examine each action produced by the generator, where the assistant triggers a rollback operation upon detection of incorrect actions. Moreover, we introduce two additional strategies tailored for the rollback scenario to further improve its effectiveness. Extensive experiments show that GA-Rollback achieves significant improvements over several strong baselines on three widely used benchmarks. Our analysis further reveals that GA-Rollback can function as a robust plug-and-play module, integrating seamlessly with other methods. 

---
# MMSciBench: Benchmarking Language Models on Multimodal Scientific Problems 

**Authors**: Xinwu Ye, Chengfan Li, Siming Chen, Xiangru Tang, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2503.01891)  

**Abstract**: Recent advances in large language models (LLMs) and vision-language models (LVLMs) have shown promise across many tasks, yet their scientific reasoning capabilities remain untested, particularly in multimodal settings. We present MMSciBench, a benchmark for evaluating mathematical and physical reasoning through text-only and text-image formats, with human-annotated difficulty levels, solutions with detailed explanations, and taxonomic mappings. Evaluation of state-of-the-art models reveals significant limitations, with even the best model achieving only \textbf{63.77\%} accuracy and particularly struggling with visual reasoning tasks. Our analysis exposes critical gaps in complex reasoning and visual-textual integration, establishing MMSciBench as a rigorous standard for measuring progress in multimodal scientific understanding. The code for MMSciBench is open-sourced at GitHub, and the dataset is available at Hugging Face. 

---
