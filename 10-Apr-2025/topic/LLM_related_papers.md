# KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs 

**Authors**: Elan Markowitz, Krupa Galiya, Greg Ver Steeg, Aram Galstyan  

**Link**: [PDF](https://arxiv.org/pdf/2504.07087)  

**Abstract**: Knowledge graphs have emerged as a popular method for injecting up-to-date, factual knowledge into large language models (LLMs). This is typically achieved by converting the knowledge graph into text that the LLM can process in context. While multiple methods of encoding knowledge graphs have been proposed, the impact of this textualization process on LLM performance remains under-explored. We introduce KG-LLM-Bench, a comprehensive and extensible benchmark spanning five knowledge graph understanding tasks, and evaluate how different encoding strategies affect performance across various base models. Our extensive experiments with seven language models and five textualization strategies provide insights for optimizing LLM performance on KG reasoning tasks. 

---
# Review of Case-Based Reasoning for LLM Agents: Theoretical Foundations, Architectural Components, and Cognitive Integration 

**Authors**: Kostas Hatalis, Despina Christou, Vyshnavi Kondapalli  

**Link**: [PDF](https://arxiv.org/pdf/2504.06943)  

**Abstract**: Agents powered by Large Language Models (LLMs) have recently demonstrated impressive capabilities in various tasks. Still, they face limitations in tasks requiring specific, structured knowledge, flexibility, or accountable decision-making. While agents are capable of perceiving their environments, forming inferences, planning, and executing actions towards goals, they often face issues such as hallucinations and lack of contextual memory across interactions. This paper explores how Case-Based Reasoning (CBR), a strategy that solves new problems by referencing past experiences, can be integrated into LLM agent frameworks. This integration allows LLMs to leverage explicit knowledge, enhancing their effectiveness. We systematically review the theoretical foundations of these enhanced agents, identify critical framework components, and formulate a mathematical model for the CBR processes of case retrieval, adaptation, and learning. We also evaluate CBR-enhanced agents against other methods like Chain-of-Thought reasoning and standard Retrieval-Augmented Generation, analyzing their relative strengths. Moreover, we explore how leveraging CBR's cognitive dimensions (including self-reflection, introspection, and curiosity) via goal-driven autonomy mechanisms can further enhance the LLM agent capabilities. Contributing to the ongoing research on neuro-symbolic hybrid systems, this work posits CBR as a viable technique for enhancing the reasoning skills and cognitive aspects of autonomous LLM agents. 

---
# FamilyTool: A Multi-hop Personalized Tool Use Benchmark 

**Authors**: Yuxin Wang, Yiran Guo, Yining Zheng, Zhangyue Yin, Shuo Chen, Jie Yang, Jiajun Chen, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06766)  

**Abstract**: The integration of tool learning with Large Language Models (LLMs) has expanded their capabilities in handling complex tasks by leveraging external tools. However, existing benchmarks for tool learning inadequately address critical real-world personalized scenarios, particularly those requiring multi-hop reasoning and inductive knowledge adaptation in dynamic environments. To bridge this gap, we introduce FamilyTool, a novel benchmark grounded in a family-based knowledge graph (KG) that simulates personalized, multi-hop tool use scenarios. FamilyTool challenges LLMs with queries spanning 1 to 3 relational hops (e.g., inferring familial connections and preferences) and incorporates an inductive KG setting where models must adapt to unseen user preferences and relationships without re-training, a common limitation in prior approaches that compromises generalization. We further propose KGETool: a simple KG-augmented evaluation pipeline to systematically assess LLMs' tool use ability in these settings. Experiments reveal significant performance gaps in state-of-the-art LLMs, with accuracy dropping sharply as hop complexity increases and inductive scenarios exposing severe generalization deficits. These findings underscore the limitations of current LLMs in handling personalized, evolving real-world contexts and highlight the urgent need for advancements in tool-learning frameworks. FamilyTool serves as a critical resource for evaluating and advancing LLM agents' reasoning, adaptability, and scalability in complex, dynamic environments. Code and dataset are available at Github. 

---
# Missing Premise exacerbates Overthinking: Are Reasoning Models losing Critical Thinking Skill? 

**Authors**: Chenrui Fan, Ming Li, Lichao Sun, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.06514)  

**Abstract**: We find that the response length of reasoning LLMs, whether trained by reinforcement learning or supervised learning, drastically increases for ill-posed questions with missing premises (MiP), ending up with redundant and ineffective thinking. This newly introduced scenario exacerbates the general overthinking issue to a large extent, which we name as the MiP-Overthinking. Such failures are against the ``test-time scaling law'' but have been widely observed on multiple datasets we curated with MiP, indicating the harm of cheap overthinking and a lack of critical thinking. Surprisingly, LLMs not specifically trained for reasoning exhibit much better performance on the MiP scenario, producing much shorter responses that quickly identify ill-posed queries. This implies a critical flaw of the current training recipe for reasoning LLMs, which does not encourage efficient thinking adequately, leading to the abuse of thinking patterns. To further investigate the reasons behind such failures, we conduct fine-grained analyses of the reasoning length, overthinking patterns, and location of critical thinking on different types of LLMs. Moreover, our extended ablation study reveals that the overthinking is contagious through the distillation of reasoning models' responses. These results improve the understanding of overthinking and shed novel insights into mitigating the problem. 

---
# Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning 

**Authors**: Nikhil Shivakumar Nayak, Krishnateja Killamsetty, Ligong Han, Abhishek Bhandwaldar, Prateek Chanda, Kai Xu, Hao Wang, Aldo Pareja, Oleg Silkin, Mustafa Eyceoz, Akash Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2504.07097)  

**Abstract**: Continual learning in large language models (LLMs) is prone to catastrophic forgetting, where adapting to new tasks significantly degrades performance on previously learned ones. Existing methods typically rely on low-rank, parameter-efficient updates that limit the model's expressivity and introduce additional parameters per task, leading to scalability issues. To address these limitations, we propose a novel continual full fine-tuning approach leveraging adaptive singular value decomposition (SVD). Our method dynamically identifies task-specific low-rank parameter subspaces and constrains updates to be orthogonal to critical directions associated with prior tasks, thus effectively minimizing interference without additional parameter overhead or storing previous task gradients. We evaluate our approach extensively on standard continual learning benchmarks using both encoder-decoder (T5-Large) and decoder-only (LLaMA-2 7B) models, spanning diverse tasks including classification, generation, and reasoning. Empirically, our method achieves state-of-the-art results, up to 7% higher average accuracy than recent baselines like O-LoRA, and notably maintains the model's general linguistic capabilities, instruction-following accuracy, and safety throughout the continual learning process by reducing forgetting to near-negligible levels. Our adaptive SVD framework effectively balances model plasticity and knowledge retention, providing a practical, theoretically grounded, and computationally scalable solution for continual learning scenarios in large language models. 

---
# Self-Steering Language Models 

**Authors**: Gabriel Grand, Joshua B. Tenenbaum, Vikash K. Mansinghka, Alexander K. Lew, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2504.07081)  

**Abstract**: While test-time reasoning enables language models to tackle complex tasks, searching or planning in natural language can be slow, costly, and error-prone. But even when LMs struggle to emulate the precise reasoning steps needed to solve a problem, they often excel at describing its abstract structure--both how to verify solutions and how to search for them. This paper introduces DisCIPL, a method for "self-steering" LMs where a Planner model generates a task-specific inference program that is executed by a population of Follower models. Our approach equips LMs with the ability to write recursive search procedures that guide LM inference, enabling new forms of verifiable and efficient reasoning. When instantiated with a small Follower (e.g., Llama-3.2-1B), DisCIPL matches (and sometimes outperforms) much larger models, including GPT-4o and o1, on challenging constrained generation tasks. In decoupling planning from execution, our work opens up a design space of highly-parallelized Monte Carlo inference strategies that outperform standard best-of-N sampling, require no finetuning, and can be implemented automatically by existing LMs. 

---
# HalluciNot: Hallucination Detection Through Context and Common Knowledge Verification 

**Authors**: Bibek Paudel, Alexander Lyzhov, Preetam Joshi, Puneet Anand  

**Link**: [PDF](https://arxiv.org/pdf/2504.07069)  

**Abstract**: This paper introduces a comprehensive system for detecting hallucinations in large language model (LLM) outputs in enterprise settings. We present a novel taxonomy of LLM responses specific to hallucination in enterprise applications, categorizing them into context-based, common knowledge, enterprise-specific, and innocuous statements. Our hallucination detection model HDM-2 validates LLM responses with respect to both context and generally known facts (common knowledge). It provides both hallucination scores and word-level annotations, enabling precise identification of problematic content. To evaluate it on context-based and common-knowledge hallucinations, we introduce a new dataset HDMBench. Experimental results demonstrate that HDM-2 out-performs existing approaches across RagTruth, TruthfulQA, and HDMBench datasets. This work addresses the specific challenges of enterprise deployment, including computational efficiency, domain specialization, and fine-grained error identification. Our evaluation dataset, model weights, and inference code are publicly available. 

---
# DeduCE: Deductive Consistency as a Framework to Evaluate LLM Reasoning 

**Authors**: Atharva Pandey, Kshitij Dubey, Rahul Sharma, Amit Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2504.07080)  

**Abstract**: Despite great performance on Olympiad-level reasoning problems, frontier large language models can still struggle on high school math when presented with novel problems outside standard benchmarks. Going beyond final accuracy, we propose a deductive consistency metric to analyze chain-of-thought output from language models (LMs).Formally, deductive reasoning involves two subtasks: understanding a set of input premises and inferring the conclusions that follow from them. The proposed metric studies LMs' performance on these subtasks, with the goal of explaining LMs' reasoning errors on novel problems: how well do LMs understand input premises with increasing context lengths, and how well can they infer conclusions over multiple reasoning hops? Since existing benchmarks may be memorized, we develop a pipeline to evaluate LMs' deductive consistency on novel, perturbed versions of benchmark problems. On novel grade school math problems (GSM-8k), we find that LMs are fairly robust to increasing number of input premises, but suffer significant accuracy decay as the number of reasoning hops is increased. Interestingly, these errors are masked in the original benchmark as all models achieve near 100% accuracy. As we increase the number of solution steps using a synthetic dataset, prediction over multiple hops still remains the major source of error compared to understanding input premises. Other factors, such as shifts in language style or natural propagation of early errors do not explain the trends. Our analysis provides a new view to characterize LM reasoning -- as computations over a window of input premises and reasoning hops -- that can provide unified evaluation across problem domains. 

---
# Zero-Shot Image-Based Large Language Model Approach to Road Pavement Monitoring 

**Authors**: Shuoshuo Xu, Kai Zhao, James Loney, Zili Li, Andrea Visentin  

**Link**: [PDF](https://arxiv.org/pdf/2504.06785)  

**Abstract**: Effective and rapid evaluation of pavement surface condition is critical for prioritizing maintenance, ensuring transportation safety, and minimizing vehicle wear and tear. While conventional manual inspections suffer from subjectivity, existing machine learning-based methods are constrained by their reliance on large and high-quality labeled datasets, which require significant resources and limit adaptability across varied road conditions. The revolutionary advancements in Large Language Models (LLMs) present significant potential for overcoming these challenges. In this study, we propose an innovative automated zero-shot learning approach that leverages the image recognition and natural language understanding capabilities of LLMs to assess road conditions effectively. Multiple LLM-based assessment models were developed, employing prompt engineering strategies aligned with the Pavement Surface Condition Index (PSCI) standards. These models' accuracy and reliability were evaluated against official PSCI results, with an optimized model ultimately selected. Extensive tests benchmarked the optimized model against evaluations from various levels experts using Google Street View road images. The results reveal that the LLM-based approach can effectively assess road conditions, with the optimized model -employing comprehensive and structured prompt engineering strategies -outperforming simpler configurations by achieving high accuracy and consistency, even surpassing expert evaluations. Moreover, successfully applying the optimized model to Google Street View images demonstrates its potential for future city-scale deployments. These findings highlight the transformative potential of LLMs in automating road damage evaluations and underscore the pivotal role of detailed prompt engineering in achieving reliable assessments. 

---
# Right Prediction, Wrong Reasoning: Uncovering LLM Misalignment in RA Disease Diagnosis 

**Authors**: Umakanta Maharana, Sarthak Verma, Avarna Agarwal, Prakashini Mruthyunjaya, Dwarikanath Mahapatra, Sakir Ahmed, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2504.06581)  

**Abstract**: Large language models (LLMs) offer a promising pre-screening tool, improving early disease detection and providing enhanced healthcare access for underprivileged communities. The early diagnosis of various diseases continues to be a significant challenge in healthcare, primarily due to the nonspecific nature of early symptoms, the shortage of expert medical practitioners, and the need for prolonged clinical evaluations, all of which can delay treatment and adversely affect patient outcomes. With impressive accuracy in prediction across a range of diseases, LLMs have the potential to revolutionize clinical pre-screening and decision-making for various medical conditions. In this work, we study the diagnostic capability of LLMs for Rheumatoid Arthritis (RA) with real world patients data. Patient data was collected alongside diagnoses from medical experts, and the performance of LLMs was evaluated in comparison to expert diagnoses for RA disease prediction. We notice an interesting pattern in disease diagnosis and find an unexpected \textit{misalignment between prediction and explanation}. We conduct a series of multi-round analyses using different LLM agents. The best-performing model accurately predicts rheumatoid arthritis (RA) diseases approximately 95\% of the time. However, when medical experts evaluated the reasoning generated by the model, they found that nearly 68\% of the reasoning was incorrect. This study highlights a clear misalignment between LLMs high prediction accuracy and its flawed reasoning, raising important questions about relying on LLM explanations in clinical settings. \textbf{LLMs provide incorrect reasoning to arrive at the correct answer for RA disease diagnosis.} 

---
# A Neuro-inspired Interpretation of Unlearning in Large Language Models through Sample-level Unlearning Difficulty 

**Authors**: Xiaohua Feng, Yuyuan Li, Chengye Wang, Junlin Liu, Li Zhang, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06658)  

**Abstract**: Driven by privacy protection laws and regulations, unlearning in Large Language Models (LLMs) is gaining increasing attention. However, current research often neglects the interpretability of the unlearning process, particularly concerning sample-level unlearning difficulty. Existing studies typically assume a uniform unlearning difficulty across samples. This simplification risks attributing the performance of unlearning algorithms to sample selection rather than the algorithm's design, potentially steering the development of LLM unlearning in the wrong direction. Thus, we investigate the relationship between LLM unlearning and sample characteristics, with a focus on unlearning difficulty. Drawing inspiration from neuroscience, we propose a Memory Removal Difficulty ($\mathrm{MRD}$) metric to quantify sample-level unlearning difficulty. Using $\mathrm{MRD}$, we analyze the characteristics of hard-to-unlearn versus easy-to-unlearn samples. Furthermore, we propose an $\mathrm{MRD}$-based weighted sampling method to optimize existing unlearning algorithms, which prioritizes easily forgettable samples, thereby improving unlearning efficiency and effectiveness. We validate the proposed metric and method using public benchmarks and datasets, with results confirming its effectiveness. 

---
# Automated Business Process Analysis: An LLM-Based Approach to Value Assessment 

**Authors**: William De Michele, Abel Armas Cervantes, Lea Frermann  

**Link**: [PDF](https://arxiv.org/pdf/2504.06600)  

**Abstract**: Business processes are fundamental to organizational operations, yet their optimization remains challenging due to the timeconsuming nature of manual process analysis. Our paper harnesses Large Language Models (LLMs) to automate value-added analysis, a qualitative process analysis technique that aims to identify steps in the process that do not deliver value. To date, this technique is predominantly manual, time-consuming, and subjective. Our method offers a more principled approach which operates in two phases: first, decomposing high-level activities into detailed steps to enable granular analysis, and second, performing a value-added analysis to classify each step according to Lean principles. This approach enables systematic identification of waste while maintaining the semantic understanding necessary for qualitative analysis. We develop our approach using 50 business process models, for which we collect and publish manual ground-truth labels. Our evaluation, comparing zero-shot baselines with more structured prompts reveals (a) a consistent benefit of structured prompting and (b) promising performance for both tasks. We discuss the potential for LLMs to augment human expertise in qualitative process analysis while reducing the time and subjectivity inherent in manual approaches. 

---
# Lugha-Llama: Adapting Large Language Models for African Languages 

**Authors**: Happy Buzaaba, Alexander Wettig, David Ifeoluwa Adelani, Christiane Fellbaum  

**Link**: [PDF](https://arxiv.org/pdf/2504.06536)  

**Abstract**: Large language models (LLMs) have achieved impressive results in a wide range of natural language applications. However, they often struggle to recognize low-resource languages, in particular African languages, which are not well represented in large training corpora. In this paper, we consider how to adapt LLMs to low-resource African languages. We find that combining curated data from African languages with high-quality English educational texts results in a training mix that substantially improves the model's performance on these languages. On the challenging IrokoBench dataset, our models consistently achieve the best performance amongst similarly sized baselines, particularly on knowledge-intensive multiple-choice questions (AfriMMLU). Additionally, on the cross-lingual question answering benchmark AfriQA, our models outperform the base model by over 10%. To better understand the role of English data during training, we translate a subset of 200M tokens into Swahili language and perform an analysis which reveals that the content of these data is primarily responsible for the strong performance. We release our models and data to encourage future research on African languages. 

---
# Don't Let It Hallucinate: Premise Verification via Retrieval-Augmented Logical Reasoning 

**Authors**: Yuehan Qin, Shawn Li, Yi Nian, Xinyan Velocity Yu, Yue Zhao, Xuezhe Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.06438)  

**Abstract**: Large language models (LLMs) have shown substantial capacity for generating fluent, contextually appropriate responses. However, they can produce hallucinated outputs, especially when a user query includes one or more false premises-claims that contradict established facts. Such premises can mislead LLMs into offering fabricated or misleading details. Existing approaches include pretraining, fine-tuning, and inference-time techniques that often rely on access to logits or address hallucinations after they occur. These methods tend to be computationally expensive, require extensive training data, or lack proactive mechanisms to prevent hallucination before generation, limiting their efficiency in real-time applications. We propose a retrieval-based framework that identifies and addresses false premises before generation. Our method first transforms a user's query into a logical representation, then applies retrieval-augmented generation (RAG) to assess the validity of each premise using factual sources. Finally, we incorporate the verification results into the LLM's prompt to maintain factual consistency in the final output. Experiments show that this approach effectively reduces hallucinations, improves factual accuracy, and does not require access to model logits or large-scale fine-tuning. 

---
# Dynamic Evaluation Framework for Personalized and Trustworthy Agents: A Multi-Session Approach to Preference Adaptability 

**Authors**: Chirag Shah, Hideo Joho, Kirandeep Kaur, Preetam Prabhu Srikar Dammu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06277)  

**Abstract**: Recent advancements in generative AI have significantly increased interest in personalized agents. With increased personalization, there is also a greater need for being able to trust decision-making and action taking capabilities of these agents. However, the evaluation methods for these agents remain outdated and inadequate, often failing to capture the dynamic and evolving nature of user interactions. In this conceptual article, we argue for a paradigm shift in evaluating personalized and adaptive agents. We propose a comprehensive novel framework that models user personas with unique attributes and preferences. In this framework, agents interact with these simulated users through structured interviews to gather their preferences and offer customized recommendations. These recommendations are then assessed dynamically using simulations driven by Large Language Models (LLMs), enabling an adaptive and iterative evaluation process. Our flexible framework is designed to support a variety of agents and applications, ensuring a comprehensive and versatile evaluation of recommendation strategies that focus on proactive, personalized, and trustworthy aspects. 

---
# A Geometric-Aware Perspective and Beyond: Hybrid Quantum-Classical Machine Learning Methods 

**Authors**: Azadeh Alavia, Hossein Akhoundib, Fatemeh Kouchmeshkib, Mojtaba Mahmoodianc, Sanduni Jayasinghec, Yongli Rena, Abdolrahman Alavi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06328)  

**Abstract**: Geometric Machine Learning (GML) has shown that respecting non-Euclidean geometry in data spaces can significantly improve performance over naive Euclidean assumptions. In parallel, Quantum Machine Learning (QML) has emerged as a promising paradigm that leverages superposition, entanglement, and interference within quantum state manifolds for learning tasks. This paper offers a unifying perspective by casting QML as a specialized yet more expressive branch of GML. We argue that quantum states, whether pure or mixed, reside on curved manifolds (e.g., projective Hilbert spaces or density-operator manifolds), mirroring how covariance matrices inhabit the manifold of symmetric positive definite (SPD) matrices or how image sets occupy Grassmann manifolds. However, QML also benefits from purely quantum properties, such as entanglement-induced curvature, that can yield richer kernel structures and more nuanced data embeddings.
We illustrate these ideas with published and newly discussed results, including hybrid classical -quantum pipelines for diabetic foot ulcer classification and structural health monitoring. Despite near-term hardware limitations that constrain purely quantum solutions, hybrid architectures already demonstrate tangible benefits by combining classical manifold-based feature extraction with quantum embeddings. We present a detailed mathematical treatment of the geometrical underpinnings of quantum states, emphasizing parallels to classical Riemannian geometry and manifold-based optimization. Finally, we outline open research challenges and future directions, including Quantum Large Language Models (LLMs), quantum reinforcement learning, and emerging hardware approaches, demonstrating how synergizing GML and QML principles can unlock the next generation of machine intelligence. 

---
# ER-RAG: Enhance RAG with ER-Based Unified Modeling of Heterogeneous Data Sources 

**Authors**: Yikuan Xia, Jiazun Chen, Yirui Zhan, Suifeng Zhao, Weipeng Jiang, Chaorui Zhang, Wei Han, Bo Bai, Jun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.06271)  

**Abstract**: Large language models (LLMs) excel in question-answering (QA) tasks, and retrieval-augmented generation (RAG) enhances their precision by incorporating external evidence from diverse sources like web pages, databases, and knowledge graphs. However, current RAG methods rely on agent-specific strategies for individual data sources, posing challenges low-resource or black-box environments and complicates operations when evidence is fragmented across sources. To address these limitations, we propose ER-RAG, a framework that unifies evidence integration across heterogeneous data sources using the Entity-Relationship (ER) model. ER-RAG standardizes entity retrieval and relationship querying through ER-based APIs with GET and JOIN operations. It employs a two-stage generation process: first, a preference optimization module selects optimal sources; second, another module constructs API chains based on source schemas. This unified approach allows efficient fine-tuning and seamless integration across diverse data sources. ER-RAG demonstrated its effectiveness by winning all three tracks of the 2024 KDDCup CRAG Challenge, achieving performance on par with commercial RAG pipelines using an 8B LLM backbone. It outperformed hybrid competitors by 3.1% in LLM score and accelerated retrieval by 5.5X. 

---
# StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization 

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05804)  

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present StealthRank, a novel adversarial ranking attack that manipulates LLM-driven product recommendation systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within product descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target products while avoiding explicit manipulation traces that can be easily detected. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven recommendation systems. 

---
# On the Effectiveness and Generalization of Race Representations for Debiasing High-Stakes Decisions 

**Authors**: Dang Nguyen, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.06303)  

**Abstract**: Understanding and mitigating biases is critical for the adoption of large language models (LLMs) in high-stakes decision-making. We introduce Admissions and Hiring, decision tasks with hypothetical applicant profiles where a person's race can be inferred from their name, as simplified test beds for racial bias. We show that Gemma 2B Instruct and LLaMA 3.2 3B Instruct exhibit strong biases. Gemma grants admission to 26% more White than Black applicants, and LLaMA hires 60% more Asian than White applicants. We demonstrate that these biases are resistant to prompt engineering: multiple prompting strategies all fail to promote fairness. In contrast, using distributed alignment search, we can identify "race subspaces" within model activations and intervene on them to debias model decisions. Averaging the representation across all races within the subspaces reduces Gemma's bias by 37-57%. Finally, we examine the generalizability of Gemma's race subspaces, and find limited evidence for generalization, where changing the prompt format can affect the race representation. Our work suggests mechanistic approaches may provide a promising venue for improving the fairness of LLMs, but a universal race representation remains elusive. 

---
# Leveraging LLMs for User Stories in AI Systems: UStAI Dataset 

**Authors**: Asma Yamani, Malak Baslyman, Moataz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2504.00513)  

**Abstract**: AI systems are gaining widespread adoption across various sectors and domains. Creating high-quality AI system requirements is crucial for aligning the AI system with business goals and consumer values and for social responsibility. However, with the uncertain nature of AI systems and the heavy reliance on sensitive data, more research is needed to address the elicitation and analysis of AI systems requirements. With the proprietary nature of many AI systems, there is a lack of open-source requirements artifacts and technical requirements documents for AI systems, limiting broader research and investigation. With Large Language Models (LLMs) emerging as a promising alternative to human-generated text, this paper investigates the potential use of LLMs to generate user stories for AI systems based on abstracts from scholarly papers. We conducted an empirical evaluation using three LLMs and generated $1260$ user stories from $42$ abstracts from $26$ domains. We assess their quality using the Quality User Story (QUS) framework. Moreover, we identify relevant non-functional requirements (NFRs) and ethical principles. Our analysis demonstrates that the investigated LLMs can generate user stories inspired by the needs of various stakeholders, offering a promising approach for generating user stories for research purposes and for aiding in the early requirements elicitation phase of AI systems. We have compiled and curated a collection of stories generated by various LLMs into a dataset (UStAI), which is now publicly available for use. 

---
# A Survey on Personalized and Pluralistic Preference Alignment in Large Language Models 

**Authors**: Zhouhang Xie, Junda Wu, Yiran Shen, Yu Xia, Xintong Li, Aaron Chang, Ryan Rossi, Sachin Kumar, Bodhisattwa Prasad Majumder, Jingbo Shang, Prithviraj Ammanabrolu, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2504.07070)  

**Abstract**: Personalized preference alignment for large language models (LLMs), the process of tailoring LLMs to individual users' preferences, is an emerging research direction spanning the area of NLP and personalization. In this survey, we present an analysis of works on personalized alignment and modeling for LLMs. We introduce a taxonomy of preference alignment techniques, including training time, inference time, and additionally, user-modeling based methods. We provide analysis and discussion on the strengths and limitations of each group of techniques and then cover evaluation, benchmarks, as well as open problems in the field. 

---
# Evaluating Retrieval Augmented Generative Models for Document Queries in Transportation Safety 

**Authors**: Chad Melton, Alex Sorokine, Steve Peterson  

**Link**: [PDF](https://arxiv.org/pdf/2504.07022)  

**Abstract**: Applications of generative Large Language Models LLMs are rapidly expanding across various domains, promising significant improvements in workflow efficiency and information retrieval. However, their implementation in specialized, high-stakes domains such as hazardous materials transportation is challenging due to accuracy and reliability concerns. This study evaluates the performance of three fine-tuned generative models, ChatGPT, Google's Vertex AI, and ORNL Retrieval Augmented Generation augmented LLaMA 2 and LLaMA in retrieving regulatory information essential for hazardous material transportation compliance in the United States. Utilizing approximately 40 publicly available federal and state regulatory documents, we developed 100 realistic queries relevant to route planning and permitting requirements. Responses were qualitatively rated based on accuracy, detail, and relevance, complemented by quantitative assessments of semantic similarity between model outputs. Results demonstrated that the RAG-augmented LLaMA models significantly outperformed Vertex AI and ChatGPT, providing more detailed and generally accurate information, despite occasional inconsistencies. This research introduces the first known application of RAG in transportation safety, emphasizing the need for domain-specific fine-tuning and rigorous evaluation methodologies to ensure reliability and minimize the risk of inaccuracies in high-stakes environments. 

---
# Towards LLMs Robustness to Changes in Prompt Format Styles 

**Authors**: Lilian Ngweta, Kiran Kate, Jason Tsay, Yara Rizk  

**Link**: [PDF](https://arxiv.org/pdf/2504.06969)  

**Abstract**: Large language models (LLMs) have gained popularity in recent years for their utility in various applications. However, they are sensitive to non-semantic changes in prompt formats, where small changes in the prompt format can lead to significant performance fluctuations. In the literature, this problem is commonly referred to as prompt brittleness. Previous research on prompt engineering has focused mainly on developing techniques for identifying the optimal prompt for specific tasks. Some studies have also explored the issue of prompt brittleness and proposed methods to quantify performance variations; however, no simple solution has been found to address this challenge. We propose Mixture of Formats (MOF), a simple and efficient technique for addressing prompt brittleness in LLMs by diversifying the styles used in the prompt few-shot examples. MOF was inspired by computer vision techniques that utilize diverse style datasets to prevent models from associating specific styles with the target variable. Empirical results show that our proposed technique reduces style-induced prompt brittleness in various LLMs while also enhancing overall performance across prompt variations and different datasets. 

---
# Data Augmentation for Fake Reviews Detection in Multiple Languages and Multiple Domains 

**Authors**: Ming Liu, Massimo Poesio  

**Link**: [PDF](https://arxiv.org/pdf/2504.06917)  

**Abstract**: With the growth of the Internet, buying habits have changed, and customers have become more dependent on the online opinions of other customers to guide their purchases. Identifying fake reviews thus became an important area for Natural Language Processing (NLP) research. However, developing high-performance NLP models depends on the availability of large amounts of training data, which are often not available for low-resource languages or domains. In this research, we used large language models to generate datasets to train fake review detectors. Our approach was used to generate fake reviews in different domains (book reviews, restaurant reviews, and hotel reviews) and different languages (English and Chinese). Our results demonstrate that our data augmentation techniques result in improved performance at fake review detection for all domains and languages. The accuracy of our fake review detection model can be improved by 0.3 percentage points on DeRev TEST, 10.9 percentage points on Amazon TEST, 8.3 percentage points on Yelp TEST and 7.2 percentage points on DianPing TEST using the augmented datasets. 

---
# Open Problems and a Hypothetical Path Forward in LLM Knowledge Paradigms 

**Authors**: Xiaotian Ye, Mengqi Zhang, Shu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06823)  

**Abstract**: Knowledge is fundamental to the overall capabilities of Large Language Models (LLMs). The knowledge paradigm of a model, which dictates how it encodes and utilizes knowledge, significantly affects its performance. Despite the continuous development of LLMs under existing knowledge paradigms, issues within these frameworks continue to constrain model potential.
This blog post highlight three critical open problems limiting model capabilities: (1) challenges in knowledge updating for LLMs, (2) the failure of reverse knowledge generalization (the reversal curse), and (3) conflicts in internal knowledge. We review recent progress made in addressing these issues and discuss potential general solutions. Based on observations in these areas, we propose a hypothetical paradigm based on Contextual Knowledge Scaling, and further outline implementation pathways that remain feasible within contemporary techniques. Evidence suggests this approach holds potential to address current shortcomings, serving as our vision for future model paradigms.
This blog post aims to provide researchers with a brief overview of progress in LLM knowledge systems, while provide inspiration for the development of next-generation model architectures. 

---
# SEE: Continual Fine-tuning with Sequential Ensemble of Experts 

**Authors**: Zhilin Wang, Yafu Li, Xiaoye Qu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.06664)  

**Abstract**: Continual fine-tuning of large language models (LLMs) suffers from catastrophic forgetting. Rehearsal-based methods mitigate this problem by retaining a small set of old data. Nevertheless, they still suffer inevitable performance loss. Although training separate experts for each task can help prevent forgetting, effectively assembling them remains a challenge. Some approaches use routers to assign tasks to experts, but in continual learning, they often require retraining for optimal performance. To address these challenges, we introduce the Sequential Ensemble of Experts (SEE) framework. SEE removes the need for an additional router, allowing each expert to independently decide whether a query should be handled. The framework employs distributed routing, and during continual fine-tuning, SEE only requires the training of new experts for incoming tasks rather than retraining the entire system. Experiments reveal that the SEE outperforms prior approaches, including multi-task learning, in continual fine-tuning. It also demonstrates remarkable generalization ability, as the expert can effectively identify out-of-distribution queries, which can then be directed to a more generalized model for resolution. This work highlights the promising potential of integrating routing and response mechanisms within each expert, paving the way for the future of distributed model ensembling. 

---
# ThoughtProbe: Classifier-Guided Thought Space Exploration Leveraging LLM Intrinsic Reasoning 

**Authors**: Zijian Wang, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06650)  

**Abstract**: Pre-trained large language models (LLMs) have been demonstrated to possess intrinsic reasoning capabilities that can emerge naturally when expanding the response space. However, the neural representation mechanisms underlying these intrinsic capabilities and approaches for their optimal utilization remain inadequately understood. In this work, we make the key discovery that a simple linear classifier can effectively detect intrinsic reasoning capabilities in LLMs' activation space, particularly within specific representation types and network layers. Based on this finding, we propose a classifier-guided search framework that strategically explore a tree-structured response space. In each node expansion, the classifier serves as a scoring and ranking mechanism that efficiently allocates computational resources by identifying and prioritizing more thoughtful reasoning directions for continuation. After completing the tree expansion, we collect answers from all branches to form a candidate answer pool. We propose a branch-aggregation selection method that marginalizes over all supporting branches by aggregating their thoughtfulness scores, thereby identifying the optimal answer from the pool. Experimental results show that our framework's comprehensive exploration not only covers valid reasoning chains but also effectively identifies them, achieving significant improvements across multiple arithmetic reasoning benchmarks. 

---
# Integrating Cognitive Processing Signals into Language Models: A Review of Advances, Applications and Future Directions 

**Authors**: Angela Lopez-Cardona, Sebastian Idesis, Ioannis Arapakis  

**Link**: [PDF](https://arxiv.org/pdf/2504.06843)  

**Abstract**: Recently, the integration of cognitive neuroscience in Natural Language Processing (NLP) has gained significant attention. This article provides a critical and timely overview of recent advancements in leveraging cognitive signals, particularly Eye-tracking (ET) signals, to enhance Language Models (LMs) and Multimodal Large Language Models (MLLMs). By incorporating user-centric cognitive signals, these approaches address key challenges, including data scarcity and the environmental costs of training large-scale models. Cognitive signals enable efficient data augmentation, faster convergence, and improved human alignment. The review emphasises the potential of ET data in tasks like Visual Question Answering (VQA) and mitigating hallucinations in MLLMs, and concludes by discussing emerging challenges and research trends. 

---
# Bypassing Safety Guardrails in LLMs Using Humor 

**Authors**: Pedro Cisneros-Velarde  

**Link**: [PDF](https://arxiv.org/pdf/2504.06577)  

**Abstract**: In this paper, we show it is possible to bypass the safety guardrails of large language models (LLMs) through a humorous prompt including the unsafe request. In particular, our method does not edit the unsafe request and follows a fixed template -- it is simple to implement and does not need additional LLMs to craft prompts. Extensive experiments show the effectiveness of our method across different LLMs. We also show that both removing and adding more humor to our method can reduce its effectiveness -- excessive humor possibly distracts the LLM from fulfilling its unsafe request. Thus, we argue that LLM jailbreaking occurs when there is a proper balance between focus on the unsafe request and presence of humor. 

---
# Do Reasoning Models Show Better Verbalized Calibration? 

**Authors**: Qingcheng Zeng, Weihao Xuan, Leyang Cui, Rob Voigt  

**Link**: [PDF](https://arxiv.org/pdf/2504.06564)  

**Abstract**: Large reasoning models (LRMs) have recently shown impressive capabilities in complex reasoning by leveraging increased test-time computation and exhibiting behaviors akin to human-like deliberation. Despite these advances, it remains an open question whether LRMs are better calibrated - particularly in their verbalized confidence - compared to instruction-tuned counterparts. In this paper, we investigate the calibration properties of LRMs trained via supervised fine-tuning distillation on long reasoning traces (henceforth SFT reasoning models) and outcome-based reinforcement learning for reasoning (henceforth RL reasoning models) across diverse domains. Our findings reveal that LRMs significantly outperform instruction-tuned models on complex reasoning tasks in both accuracy and confidence calibration. In contrast, we find surprising trends in the domain of factuality in particular. On factuality tasks, while Deepseek-R1 shows strong calibration behavior, smaller QwQ-32B shows no improvement over instruct models; moreover, SFT reasoning models display worse calibration (greater overconfidence) compared to instruct models. Our results provide evidence for a potentially critical role of reasoning-oriented RL training in improving LLMs' capacity for generating trustworthy, self-aware outputs. 

---
# NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables 

**Authors**: Lanrui Wang, Mingyu Zheng, Hongyin Tang, Zheng Lin, Yanan Cao, Jingang Wang, Xunliang Cai, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06560)  

**Abstract**: Processing structured tabular data, particularly lengthy tables, constitutes a fundamental yet challenging task for large language models (LLMs). However, existing long-context benchmarks primarily focus on unstructured text, neglecting the challenges of long and complex structured tables. To address this gap, we introduce NeedleInATable (NIAT), a novel task that treats each table cell as a "needle" and requires the model to extract the target cell under different queries. Evaluation results of mainstream LLMs on this benchmark show they lack robust long-table comprehension, often relying on superficial correlations or shortcuts for complex table understanding tasks, revealing significant limitations in processing intricate tabular data. To this end, we propose a data synthesis method to enhance models' long-table comprehension capabilities. Experimental results show that our synthesized training data significantly enhances LLMs' performance on the NIAT task, outperforming both long-context LLMs and long-table agent methods. This work advances the evaluation of LLMs' genuine long-structured table comprehension capabilities and paves the way for progress in long-context and table understanding applications. 

---
# Can LLMs Simulate Personas with Reversed Performance? A Benchmark for Counterfactual Instruction Following 

**Authors**: Sai Adith Senthil Kumar, Hao Yan, Saipavan Perepa, Murong Yue, Ziyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.06460)  

**Abstract**: Large Language Models (LLMs) are now increasingly widely used to simulate personas in virtual environments, leveraging their instruction-following capability. However, we discovered that even state-of-the-art LLMs cannot simulate personas with reversed performance (e.g., student personas with low proficiency in educational settings), which impairs the simulation diversity and limits the practical applications of the simulated environments. In this work, using mathematical reasoning as a representative scenario, we propose the first benchmark dataset for evaluating LLMs on simulating personas with reversed performance, a capability that we dub "counterfactual instruction following". We evaluate both open-weight and closed-source LLMs on this task and find that LLMs, including the OpenAI o1 reasoning model, all struggle to follow counterfactual instructions for simulating reversedly performing personas. Intersectionally simulating both the performance level and the race population of a persona worsens the effect even further. These results highlight the challenges of counterfactual instruction following and the need for further research. 

---
# S'MoRE: Structural Mixture of Residual Experts for LLM Fine-tuning 

**Authors**: Hanqing Zeng, Yinglong Xia, Zhuokai Zhao, Gilbert Jiang, Qiang Zhang, Jiayi Liu, Lizhu Zhang, Xiangjun Fan, Benyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06426)  

**Abstract**: Fine-tuning pre-trained large language models (LLMs) presents a dual challenge of balancing parameter efficiency and model capacity. Existing methods like low-rank adaptations (LoRA) are efficient but lack flexibility, while Mixture-of-Experts (MoE) architectures enhance model capacity at the cost of more & under-utilized parameters. To address these limitations, we propose Structural Mixture of Residual Experts (S'MoRE), a novel framework that seamlessly integrates the efficiency of LoRA with the flexibility of MoE. Specifically, S'MoRE employs hierarchical low-rank decomposition of expert weights, yielding residuals of varying orders interconnected in a multi-layer structure. By routing input tokens through sub-trees of residuals, S'MoRE emulates the capacity of many experts by instantiating and assembling just a few low-rank matrices. We craft the inter-layer propagation of S'MoRE's residuals as a special type of Graph Neural Network (GNN), and prove that under similar parameter budget, S'MoRE improves "structural flexibility" of traditional MoE (or Mixture-of-LoRA) by exponential order. Comprehensive theoretical analysis and empirical results demonstrate that S'MoRE achieves superior fine-tuning performance, offering a transformative approach for efficient LLM adaptation. 

---
# Query Understanding in LLM-based Conversational Information Seeking 

**Authors**: Yifei Yuan, Zahra Abbasiantaeb, Yang Deng, Mohammad Aliannejadi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06356)  

**Abstract**: Query understanding in Conversational Information Seeking (CIS) involves accurately interpreting user intent through context-aware interactions. This includes resolving ambiguities, refining queries, and adapting to evolving information needs. Large Language Models (LLMs) enhance this process by interpreting nuanced language and adapting dynamically, improving the relevance and precision of search results in real-time. In this tutorial, we explore advanced techniques to enhance query understanding in LLM-based CIS systems. We delve into LLM-driven methods for developing robust evaluation metrics to assess query understanding quality in multi-turn interactions, strategies for building more interactive systems, and applications like proactive query management and query reformulation. We also discuss key challenges in integrating LLMs for query understanding in conversational search systems and outline future research directions. Our goal is to deepen the audience's understanding of LLM-based conversational query understanding and inspire discussions to drive ongoing advancements in this field. 

---
# A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility 

**Authors**: Andreas Hochlehnert, Hardik Bhatnagar, Vishaal Udandarao, Samuel Albanie, Ameya Prabhu, Matthias Bethge  

**Link**: [PDF](https://arxiv.org/pdf/2504.07086)  

**Abstract**: Reasoning has emerged as the next major frontier for language models (LMs), with rapid advances from both academic and industrial labs. However, this progress often outpaces methodological rigor, with many evaluations relying on benchmarking practices that lack transparency, robustness, or statistical grounding. In this work, we conduct a comprehensive empirical study and find that current mathematical reasoning benchmarks are highly sensitive to subtle implementation choices - including decoding parameters, random seeds, prompt formatting, and even hardware and software-framework configurations. Performance gains reported in recent studies frequently hinge on unclear comparisons or unreported sources of variance. To address these issues, we propose a standardized evaluation framework with clearly defined best practices and reporting standards. Using this framework, we reassess recent methods and find that reinforcement learning (RL) approaches yield only modest improvements - far below prior claims - and are prone to overfitting, especially on small-scale benchmarks like AIME24. In contrast, supervised finetuning (SFT) methods show consistently stronger generalization. To foster reproducibility, we release all code, prompts, and model outputs, for reasoning benchmarks, establishing more rigorous foundations for future work. 

---
# CHIME: A Compressive Framework for Holistic Interest Modeling 

**Authors**: Yong Bai, Rui Xiang, Kaiyuan Li, Yongxiang Tang, Yanhua Cheng, Xialong Liu, Peng Jiang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2504.06780)  

**Abstract**: Modeling holistic user interests is important for improving recommendation systems but is challenged by high computational cost and difficulty in handling diverse information with full behavior context. Existing search-based methods might lose critical signals during behavior selection. To overcome these limitations, we propose CHIME: A Compressive Framework for Holistic Interest Modeling. It uses adapted large language models to encode complete user behaviors with heterogeneous inputs. We introduce multi-granular contrastive learning objectives to capture both persistent and transient interest patterns and apply residual vector quantization to generate compact embeddings. CHIME demonstrates superior ranking performance across diverse datasets, establishing a robust solution for scalable holistic interest modeling in recommendation systems. 

---
